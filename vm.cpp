#include "vm.h"
#include "simd.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

static_assert(kImageSize % kTargetTileSize == 0, "kImageSize must be divisible by kTargetTileSize");

struct Interval
{
    float lower, upper;
};

struct Interval4
{
    simd::v4f lower;
    simd::v4f upper;
};

#ifndef PROSPERO_BATCH_UNROLL
#define PROSPERO_BATCH_UNROLL 2
#endif

#define PROSPERO_STRINGIFY_IMPL(x) #x
#define PROSPERO_STRINGIFY(x) PROSPERO_STRINGIFY_IMPL(x)
#define PROSPERO_PRAGMA(x) _Pragma(PROSPERO_STRINGIFY(x))
#define PROSPERO_BATCH_UNROLL_PRAGMA PROSPERO_PRAGMA(clang loop unroll_count(PROSPERO_BATCH_UNROLL))

constexpr int kBatchLanes = kBatchSize / simd::v4f::width;
constexpr int kBatchVectorsPerRow = kTargetTileSize / simd::v4f::width;
constexpr int kBatchUnroll = PROSPERO_BATCH_UNROLL;
using Batch = std::array<simd::v4f, kBatchLanes>;

static_assert(kBatchSize % simd::v4f::width == 0, "kBatchSize must be divisible by the SIMD width");
static_assert(kTargetTileSize % simd::v4f::width == 0, "kTargetTileSize must be divisible by the SIMD width");
static_assert(kBatchLanes % kBatchUnroll == 0, "kBatchLanes must be divisible by PROSPERO_BATCH_UNROLL");
static_assert(kBatchUnroll > 0, "PROSPERO_BATCH_UNROLL must be positive");

static std::vector<Batch> thread_batch_vars;
static std::vector<Interval4> thread_interval4_vars;
static std::vector<std::array<int, 4>> thread_remap4;
#pragma omp threadprivate(thread_batch_vars, thread_interval4_vars, thread_remap4)

float parse_float(std::string_view sv)
{
    char* endptr = nullptr;
    return std::strtof(sv.data(), &endptr);
}

int parse_hex_var(std::string_view sv)
{
    std::string_view hex_part = sv.substr(1);
    char* endptr = nullptr;
    errno = 0;
    long value_long = std::strtol(hex_part.data(), &endptr, 16);
    return static_cast<int>(value_long);
}

void split(std::string_view line, std::vector<std::string_view>& out)
{
    out.clear();

    size_t start = 0;
    while (start < line.size())
    {
        while (start < line.size() && line[start] == ' ')
        {
            ++start;
        }

        if (start >= line.size())
        {
            break;
        }

        size_t end = line.find(' ', start);
        if (end == std::string_view::npos)
        {
            out.push_back(line.substr(start));
            break;
        }

        out.push_back(line.substr(start, end - start));
        start = end + 1;
    }
}

Instruction parse_instruction(const std::vector<std::string_view>& tokens)
{
    Instruction inst{};
    if (tokens[1] == "var-x")
    {
        inst.op = OpCode::VarX;
    }
    else if (tokens[1] == "var-y")
    {
        inst.op = OpCode::VarY;
    }
    else if (tokens[1] == "const")
    {
        inst.op = OpCode::Const;
        inst.constant = parse_float(tokens[2]);
    }
    else if (tokens[1] == "add")
    {
        inst.op = OpCode::Add;
        inst.input0 = parse_hex_var(tokens[2]);
        inst.input1 = parse_hex_var(tokens[3]);
    }
    else if (tokens[1] == "sub")
    {
        inst.op = OpCode::Sub;
        inst.input0 = parse_hex_var(tokens[2]);
        inst.input1 = parse_hex_var(tokens[3]);
    }
    else if (tokens[1] == "mul")
    {
        inst.op = OpCode::Mul;
        inst.input0 = parse_hex_var(tokens[2]);
        inst.input1 = parse_hex_var(tokens[3]);
    }
    else if (tokens[1] == "max")
    {
        inst.op = OpCode::Max;
        inst.input0 = parse_hex_var(tokens[2]);
        inst.input1 = parse_hex_var(tokens[3]);
    }
    else if (tokens[1] == "min")
    {
        inst.op = OpCode::Min;
        inst.input0 = parse_hex_var(tokens[2]);
        inst.input1 = parse_hex_var(tokens[3]);
    }
    else if (tokens[1] == "neg")
    {
        inst.op = OpCode::Neg;
        inst.input0 = parse_hex_var(tokens[2]);
    }
    else if (tokens[1] == "square")
    {
        inst.op = OpCode::Square;
        inst.input0 = parse_hex_var(tokens[2]);
    }
    else if (tokens[1] == "sqrt")
    {
        inst.op = OpCode::Sqrt;
        inst.input0 = parse_hex_var(tokens[2]);
    }
    else
    {
        std::fprintf(stderr, "Unknown operation: %.*s\n", static_cast<int>(tokens[1].size()), tokens[1].data());
        std::exit(1);
    }

    return inst;
}

const Batch& evaluate_batch(
    const std::vector<Instruction>& instructions,
    const Batch& x_coords,
    const Batch& y_coords)
{
    using simd::v4f;

    Batch* batch_vars = thread_batch_vars.data();
    const size_t num_instructions = instructions.size();

    for (size_t i = 0; i < num_instructions; ++i)
    {
        const Instruction& inst = instructions[i];
        Batch& out = batch_vars[i];
        switch (inst.op)
        {
            case OpCode::VarX:
                out = x_coords;
                break;
            case OpCode::VarY:
                out = y_coords;
                break;
            case OpCode::Const:
            {
                const v4f value = v4f::splat(inst.constant);
                out.fill(value);
                break;
            }
            case OpCode::Add:
            {
                const v4f* lhs = batch_vars[inst.input0].data();
                const v4f* rhs = batch_vars[inst.input1].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    dst[j] = lhs[j] + rhs[j];
                }
                break;
            }
            case OpCode::Sub:
            {
                const v4f* lhs = batch_vars[inst.input0].data();
                const v4f* rhs = batch_vars[inst.input1].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    dst[j] = lhs[j] - rhs[j];
                }
                break;
            }
            case OpCode::Mul:
            {
                const v4f* lhs = batch_vars[inst.input0].data();
                const v4f* rhs = batch_vars[inst.input1].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    dst[j] = lhs[j] * rhs[j];
                }
                break;
            }
            case OpCode::Max:
            {
                const v4f* lhs = batch_vars[inst.input0].data();
                const v4f* rhs = batch_vars[inst.input1].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    dst[j] = simd::max(lhs[j], rhs[j]);
                }
                break;
            }
            case OpCode::Min:
            {
                const v4f* lhs = batch_vars[inst.input0].data();
                const v4f* rhs = batch_vars[inst.input1].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    dst[j] = simd::min(lhs[j], rhs[j]);
                }
                break;
            }
            case OpCode::Neg:
            {
                const v4f* src = batch_vars[inst.input0].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    dst[j] = -src[j];
                }
                break;
            }
            case OpCode::Square:
            {
                const v4f* src = batch_vars[inst.input0].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    const v4f value = src[j];
                    dst[j] = value * value;
                }
                break;
            }
            case OpCode::Sqrt:
            {
                const v4f* src = batch_vars[inst.input0].data();
                v4f* dst = out.data();
                PROSPERO_BATCH_UNROLL_PRAGMA
                for (int j = 0; j < kBatchLanes; ++j)
                {
                    dst[j] = simd::sqrt(src[j]);
                }
                break;
            }
        }
    }

    return batch_vars[num_instructions - 1];
}

float min2(float a, float b) { return a < b ? a : b; }
float min4(float a, float b, float c, float d) { return min2(min2(a, b), min2(c, d)); }
float max2(float a, float b) { return a > b ? a : b; }
float max4(float a, float b, float c, float d) { return max2(max2(a, b), max2(c, d)); }

Interval4 evaluate_interval4(const std::vector<Instruction>& instructions, const Interval4& x, const Interval4& y)
{
    using simd::mask4;
    using simd::v4f;

    Interval4* interval_vars = thread_interval4_vars.data();
    const size_t num_instructions = instructions.size();
    assert(thread_interval4_vars.size() >= num_instructions);
    const v4f zero = v4f::splat(0.0f);
    const v4f x_lower = x.lower;
    const v4f x_upper = x.upper;
    const v4f y_lower = y.lower;
    const v4f y_upper = y.upper;

    auto load_lower = [&](int idx) { return interval_vars[idx].lower; };
    auto load_upper = [&](int idx) { return interval_vars[idx].upper; };
    auto store_interval = [&](size_t idx, v4f lower, v4f upper)
    {
        interval_vars[idx] = {lower, upper};
    };

    for (size_t i = 0; i < num_instructions; ++i)
    {
        const Instruction& inst = instructions[i];
        switch (inst.op)
        {
            case OpCode::VarX:
                store_interval(i, x_lower, x_upper);
                break;
            case OpCode::VarY:
                store_interval(i, y_lower, y_upper);
                break;
            case OpCode::Const:
            {
                const v4f value = v4f::splat(inst.constant);
                store_interval(i, value, value);
                break;
            }
            case OpCode::Add:
                store_interval(i,
                               load_lower(inst.input0) + load_lower(inst.input1),
                               load_upper(inst.input0) + load_upper(inst.input1));
                break;
            case OpCode::Sub:
                store_interval(i,
                               load_lower(inst.input0) - load_upper(inst.input1),
                               load_upper(inst.input0) - load_lower(inst.input1));
                break;
            case OpCode::Mul:
            {
                const v4f a = load_lower(inst.input0);
                const v4f b = load_upper(inst.input0);
                const v4f c = load_lower(inst.input1);
                const v4f d = load_upper(inst.input1);
                const v4f p1 = a * c;
                const v4f p2 = a * d;
                const v4f p3 = b * c;
                const v4f p4 = b * d;
                store_interval(i,
                               simd::min(simd::min(p1, p2), simd::min(p3, p4)),
                               simd::max(simd::max(p1, p2), simd::max(p3, p4)));
                break;
            }
            case OpCode::Max:
                store_interval(i,
                               simd::max(load_lower(inst.input0), load_lower(inst.input1)),
                               simd::max(load_upper(inst.input0), load_upper(inst.input1)));
                break;
            case OpCode::Min:
                store_interval(i,
                               simd::min(load_lower(inst.input0), load_lower(inst.input1)),
                               simd::min(load_upper(inst.input0), load_upper(inst.input1)));
                break;
            case OpCode::Neg:
                store_interval(i, -load_upper(inst.input0), -load_lower(inst.input0));
                break;
            case OpCode::Square:
            {
                const v4f a = load_lower(inst.input0);
                const v4f b = load_upper(inst.input0);
                const v4f sq_a = a * a;
                const v4f sq_b = b * b;
                const mask4 crosses_zero = (a <= zero) & (b >= zero);
                const v4f lower = simd::select(crosses_zero, zero, simd::min(sq_a, sq_b));
                const v4f upper = simd::max(sq_a, sq_b);
                store_interval(i, lower, upper);
                break;
            }
            case OpCode::Sqrt:
            {
                const v4f a = load_lower(inst.input0);
                const v4f b = load_upper(inst.input0);
                const v4f sqrt_a = simd::sqrt(simd::max(zero, a));
                const v4f sqrt_b = simd::sqrt(simd::max(zero, b));
                const mask4 invalid = b < zero;
                store_interval(i,
                               simd::select(invalid, zero, sqrt_a),
                               simd::select(invalid, zero, sqrt_b));
                break;
            }
        }
    }

    return interval_vars[num_instructions - 1];
}

void prune_instructions4(const std::vector<Instruction>& original_instructions, std::array<std::vector<Instruction>, 4>& compacted_instructions)
{
    Interval4* interval_vars = thread_interval4_vars.data();
    std::array<int, 4>* remap = thread_remap4.data();
    int remap_size = static_cast<int>(original_instructions.size());
    assert(remap_size <= static_cast<int>(thread_remap4.size()));
    assert(thread_interval4_vars.size() >= static_cast<size_t>(remap_size));

    memset(remap, -1, remap_size * 4 * sizeof(int));

    for (int j = 0; j < 4; ++j)
    {
        remap[remap_size - 1][j] = 1;
    }

    for (int i = remap_size - 1; i >= 0; --i)
    {
        const Instruction& inst = original_instructions[i];
        float i0_lower[4];
        float i0_upper[4];
        float i1_lower[4];
        float i1_upper[4];

        if (inst.op == OpCode::Max || inst.op == OpCode::Min)
        {
            assert(inst.input0 < i && inst.input1 < i);
            interval_vars[inst.input0].lower.store(i0_lower);
            interval_vars[inst.input0].upper.store(i0_upper);
            interval_vars[inst.input1].lower.store(i1_lower);
            interval_vars[inst.input1].upper.store(i1_upper);
        }

        for (int j = 0; j < 4; ++j)
        {
            if (remap[i][j] == -1)
            {
                continue;
            }

            if (inst.op == OpCode::Max)
            {
                if (i0_lower[j] >= i1_upper[j])
                {
                    remap[inst.input0][j] = 1;
                    remap[i][j] = 0;
                }
                else if (i1_lower[j] >= i0_upper[j])
                {
                    remap[inst.input1][j] = 1;
                    assert(remap[i][j] == 1);
                }
                else
                {
                    remap[inst.input0][j] = 1;
                    remap[inst.input1][j] = 1;
                    remap[i][j] = 2;
                }
            }
            else if (inst.op == OpCode::Min)
            {
                if (i0_upper[j] <= i1_lower[j])
                {
                    remap[inst.input0][j] = 1;
                    remap[i][j] = 0;
                }
                else if (i1_upper[j] <= i0_lower[j])
                {
                    remap[inst.input1][j] = 1;
                    assert(remap[i][j] == 1);
                }
                else
                {
                    remap[inst.input0][j] = 1;
                    remap[inst.input1][j] = 1;
                    remap[i][j] = 2;
                }
            }
            else
            {
                if (inst.input0 != -1)
                {
                    remap[inst.input0][j] = 1;
                }
                if (inst.input1 != -1)
                {
                    remap[inst.input1][j] = 1;
                }
            }
        }
    }

    for (int j = 0; j < 4; ++j)
    {
        compacted_instructions[j].reserve(original_instructions.size());
    }

    for (int i = 0; i < remap_size; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            Instruction inst = original_instructions[i];
            if (remap[i][j] == -1)
            {
                continue;
            }

            if (inst.op == OpCode::Max || inst.op == OpCode::Min)
            {
                if (remap[i][j] != 2)
                {
                    remap[i][j] = remap[i][j] == 0 ? remap[inst.input0][j] : remap[inst.input1][j];
                    continue;
                }
            }

            if (inst.input0 != -1)
            {
                inst.input0 = remap[inst.input0][j];
            }
            if (inst.input1 != -1)
            {
                inst.input1 = remap[inst.input1][j];
            }
            compacted_instructions[j].push_back(inst);
            remap[i][j] = static_cast<int>(compacted_instructions[j].size()) - 1;
        }
    }
}

struct Tile
{
    int px, py, size;
    Interval ix, iy;
};

void fill_region(std::vector<float>& image, Tile tile, float value)
{
    int px = tile.px;
    int py = tile.py;
    int size = tile.size;
    for (int j = 0; j < size; ++j)
    {
        for (int i = 0; i < size; ++i)
        {
            int global_x = px + i;
            int global_y = kImageSize - 1 - (py + j);
            if (global_x >= 0 && global_x < kImageSize && global_y >= 0 && global_y < kImageSize)
            {
                image[global_y * kImageSize + global_x] = value;
            }
        }
    }
}

void solve_region(const std::vector<Instruction>& instructions, std::vector<float>& image, Tile tile)
{
    int px = tile.px;
    int py = tile.py;
    int size = tile.size;
    Interval ix = tile.ix;
    Interval iy = tile.iy;

    if (tile.size == kTargetTileSize)
    {
        float x_range = tile.ix.upper - tile.ix.lower;
        float y_range = tile.iy.upper - tile.iy.lower;
        float x_step = x_range / tile.size;

        Batch x_coords;
        Batch y_coords;

        for (int dy = 0; dy < tile.size; ++dy)
        {
            float y = tile.iy.lower + (dy + 0.5f) * y_range / tile.size;
            for (int dx = 0; dx < tile.size; dx += simd::v4f::width)
            {
                const int batch_idx = dy * kBatchVectorsPerRow + dx / simd::v4f::width;
                x_coords[batch_idx] = simd::v4f::make(
                    tile.ix.lower + (dx + 0.5f) * x_step,
                    tile.ix.lower + (dx + 1.5f) * x_step,
                    tile.ix.lower + (dx + 2.5f) * x_step,
                    tile.ix.lower + (dx + 3.5f) * x_step);
                y_coords[batch_idx] = simd::v4f::splat(y);
            }
        }

        const Batch& values = evaluate_batch(instructions, x_coords, y_coords);
        float lanes[simd::v4f::width];

        for (int dy = 0; dy < size; ++dy)
        {
            for (int dx = 0; dx < size; dx += simd::v4f::width)
            {
                const int batch_idx = dy * kBatchVectorsPerRow + dx / simd::v4f::width;
                values[batch_idx].store(lanes);
                for (int lane = 0; lane < simd::v4f::width; ++lane)
                {
                    int global_x = px + dx + lane;
                    int global_y = kImageSize - 1 - (py + dy);
                    assert(global_x >= 0 && global_x < kImageSize && global_y >= 0 && global_y < kImageSize);
                    image[global_y * kImageSize + global_x] = (lanes[lane] < 0.0f) ? -1.0f : 0.0f;
                }
            }
        }

        return;
    }

    float mid_x = ix.lower + (ix.upper - ix.lower) / 2.0f;
    float mid_y = iy.lower + (iy.upper - iy.lower) / 2.0f;

    int new_size = size / 2;
    assert(new_size >= kTargetTileSize);

    Tile ll = {px, py, new_size, {ix.lower, mid_x}, {iy.lower, mid_y}};
    Tile lr = {px + new_size, py, new_size, {mid_x, ix.upper}, {iy.lower, mid_y}};
    Tile ul = {px, py + new_size, new_size, {ix.lower, mid_x}, {mid_y, iy.upper}};
    Tile ur = {px + new_size, py + new_size, new_size, {mid_x, ix.upper}, {mid_y, iy.upper}};

    Tile tiles[4] = {ll, lr, ul, ur};

    Interval4 ix4 = {
        simd::v4f::make(ll.ix.lower, lr.ix.lower, ul.ix.lower, ur.ix.lower),
        simd::v4f::make(ll.ix.upper, lr.ix.upper, ul.ix.upper, ur.ix.upper)
    };
    Interval4 iy4 = {
        simd::v4f::make(ll.iy.lower, lr.iy.lower, ul.iy.lower, ur.iy.lower),
        simd::v4f::make(ll.iy.upper, lr.iy.upper, ul.iy.upper, ur.iy.upper)
    };

    Interval4 ir4 = evaluate_interval4(instructions, ix4, iy4);
    std::array<std::vector<Instruction>, 4> compacted_instructions;
    prune_instructions4(instructions, compacted_instructions);
    float lower[4];
    float upper[4];
    ir4.lower.store(lower);
    ir4.upper.store(upper);

    for (size_t i = 0; i < 4; ++i)
    {
        if (upper[i] < 0.0f)
        {
            fill_region(image, tiles[i], -1.0f);
            continue;
        }
        if (lower[i] >= 0.0f)
        {
            continue;
        }

        Tile& t = tiles[i];
        std::vector<Instruction>& intrcts = compacted_instructions[i];

        #pragma omp task default(shared) if(size > 64)
        solve_region(intrcts, image, t);
    }

    #pragma omp taskwait
}

void evaluate_recursive(const std::vector<Instruction>& instructions, std::vector<float>& image)
{
    Tile initial_tile = {0, 0, kImageSize, {-1.0f, 1.0f}, {-1.0f, 1.0f}};
    #pragma omp parallel default(shared)
    {
        #pragma omp single
        solve_region(instructions, image, initial_tile);
    }
}

std::vector<Instruction> parse_instructions(std::string_view text)
{
    std::vector<std::string_view> tokens;
    tokens.reserve(16);

    std::vector<Instruction> instructions;
    instructions.reserve(8000);

    size_t start = 0;
    while (start <= text.size())
    {
        size_t end = text.find('\n', start);
        if (end == std::string_view::npos)
        {
            end = text.size();
        }

        std::string_view line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r')
        {
            line.remove_suffix(1);
        }

        if (!line.empty() && line[0] != '#')
        {
            split(line, tokens);
            if (!tokens.empty())
            {
                instructions.push_back(parse_instruction(tokens));
            }
        }

        if (end == text.size())
        {
            break;
        }

        start = end + 1;
    }

    return instructions;
}

int prepare_thread_locals(std::size_t instruction_count)
{
    std::atomic_int num_threads = 0;

    #pragma omp parallel
    {
        thread_batch_vars.resize(instruction_count);
        thread_interval4_vars.resize(instruction_count);
        thread_remap4.resize(instruction_count);
        ++num_threads;
    }

    return num_threads.load();
}

void render_image(const std::vector<Instruction>& instructions, std::vector<float>& image)
{
    if (image.size() != static_cast<size_t>(kImageSize * kImageSize))
    {
        image.assign(kImageSize * kImageSize, 0.0f);
    }
    else
    {
        std::fill(image.begin(), image.end(), 0.0f);
    }

    evaluate_recursive(instructions, image);
}
#undef PROSPERO_BATCH_UNROLL_PRAGMA
#undef PROSPERO_PRAGMA
#undef PROSPERO_STRINGIFY
#undef PROSPERO_STRINGIFY_IMPL
