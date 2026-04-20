#include "vm.h"

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
    alignas(16) float lower[4];
    alignas(16) float upper[4];
};

static std::vector<std::array<float, kBatchSize>> thread_batch_vars;
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

std::array<float, kBatchSize> evaluate_batch(
    const std::vector<Instruction>& instructions,
    const std::array<float, kBatchSize>& x_coords,
    const std::array<float, kBatchSize>& y_coords)
{
    std::array<float, kBatchSize>* batch_vars = thread_batch_vars.data();
    const size_t num_instructions = instructions.size();

#define LOOP(expr) for (int j = 0; j < kBatchSize; ++j) { batch_vars[i][j] = (expr); }

    for (size_t i = 0; i < num_instructions; ++i)
    {
        const Instruction& inst = instructions[i];
        switch (inst.op)
        {
            case OpCode::VarX:
                LOOP(x_coords[j]);
                break;
            case OpCode::VarY:
                LOOP(y_coords[j]);
                break;
            case OpCode::Const:
                LOOP(inst.constant);
                break;
            case OpCode::Add:
                LOOP(batch_vars[inst.input0][j] + batch_vars[inst.input1][j]);
                break;
            case OpCode::Sub:
                LOOP(batch_vars[inst.input0][j] - batch_vars[inst.input1][j]);
                break;
            case OpCode::Mul:
                LOOP(batch_vars[inst.input0][j] * batch_vars[inst.input1][j]);
                break;
            case OpCode::Max:
                LOOP(fmax(batch_vars[inst.input0][j], batch_vars[inst.input1][j]));
                break;
            case OpCode::Min:
                LOOP(fmin(batch_vars[inst.input0][j], batch_vars[inst.input1][j]));
                break;
            case OpCode::Neg:
                LOOP(-batch_vars[inst.input0][j]);
                break;
            case OpCode::Square:
                LOOP(batch_vars[inst.input0][j] * batch_vars[inst.input0][j]);
                break;
            case OpCode::Sqrt:
                LOOP(sqrt(batch_vars[inst.input0][j]));
                break;
        }
    }

#undef LOOP

    return batch_vars[num_instructions - 1];
}

float min2(float a, float b) { return a < b ? a : b; }
float min4(float a, float b, float c, float d) { return min2(min2(a, b), min2(c, d)); }
float max2(float a, float b) { return a > b ? a : b; }
float max4(float a, float b, float c, float d) { return max2(max2(a, b), max2(c, d)); }

Interval4 evaluate_interval4(const std::vector<Instruction>& instructions, const Interval4& x, const Interval4& y)
{
    Interval4* interval_vars = thread_interval4_vars.data();
    const size_t num_instructions = instructions.size();
    assert(thread_interval4_vars.size() >= num_instructions);

    for (size_t i = 0; i < num_instructions; ++i)
    {
        const Instruction& inst = instructions[i];
        switch (inst.op)
        {
            case OpCode::VarX:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = x.lower[j];
                    interval_vars[i].upper[j] = x.upper[j];
                }
                break;
            case OpCode::VarY:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = y.lower[j];
                    interval_vars[i].upper[j] = y.upper[j];
                }
                break;
            case OpCode::Const:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = inst.constant;
                    interval_vars[i].upper[j] = inst.constant;
                }
                break;
            case OpCode::Add:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = interval_vars[inst.input0].lower[j] + interval_vars[inst.input1].lower[j];
                    interval_vars[i].upper[j] = interval_vars[inst.input0].upper[j] + interval_vars[inst.input1].upper[j];
                }
                break;
            case OpCode::Sub:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = interval_vars[inst.input0].lower[j] - interval_vars[inst.input1].upper[j];
                    interval_vars[i].upper[j] = interval_vars[inst.input0].upper[j] - interval_vars[inst.input1].lower[j];
                }
                break;
            case OpCode::Mul:
                for (int j = 0; j < 4; ++j)
                {
                    float a = interval_vars[inst.input0].lower[j];
                    float b = interval_vars[inst.input0].upper[j];
                    float c = interval_vars[inst.input1].lower[j];
                    float d = interval_vars[inst.input1].upper[j];
                    float p1 = a * c;
                    float p2 = a * d;
                    float p3 = b * c;
                    float p4 = b * d;
                    interval_vars[i].lower[j] = min4(p1, p2, p3, p4);
                    interval_vars[i].upper[j] = max4(p1, p2, p3, p4);
                }
                break;
            case OpCode::Max:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = max2(interval_vars[inst.input0].lower[j], interval_vars[inst.input1].lower[j]);
                    interval_vars[i].upper[j] = max2(interval_vars[inst.input0].upper[j], interval_vars[inst.input1].upper[j]);
                }
                break;
            case OpCode::Min:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = min2(interval_vars[inst.input0].lower[j], interval_vars[inst.input1].lower[j]);
                    interval_vars[i].upper[j] = min2(interval_vars[inst.input0].upper[j], interval_vars[inst.input1].upper[j]);
                }
                break;
            case OpCode::Neg:
                for (int j = 0; j < 4; ++j)
                {
                    interval_vars[i].lower[j] = -interval_vars[inst.input0].upper[j];
                    interval_vars[i].upper[j] = -interval_vars[inst.input0].lower[j];
                }
                break;
            case OpCode::Square:
                for (int j = 0; j < 4; ++j)
                {
                    float a = interval_vars[inst.input0].lower[j];
                    float b = interval_vars[inst.input0].upper[j];
                    float sq_a = a * a;
                    float sq_b = b * b;
                    float min_val = min2(sq_a, sq_b);
                    if (a <= 0.0f && b >= 0.0f)
                    {
                        min_val = 0.0f;
                    }
                    interval_vars[i].lower[j] = min_val;
                    interval_vars[i].upper[j] = max2(sq_a, sq_b);
                }
                break;
            case OpCode::Sqrt:
                for (int j = 0; j < 4; ++j)
                {
                    float a = interval_vars[inst.input0].lower[j];
                    float b = interval_vars[inst.input0].upper[j];
                    float sqrt_a = sqrt(max2(0.0f, a));
                    float sqrt_b = sqrt(max2(0.0f, b));
                    interval_vars[i].lower[j] = (b < 0.0f) ? 0.0f : sqrt_a;
                    interval_vars[i].upper[j] = (b < 0.0f) ? 0.0f : sqrt_b;
                }
                break;
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
        for (int j = 0; j < 4; ++j)
        {
            if (remap[i][j] == -1)
            {
                continue;
            }

            if (inst.op == OpCode::Max)
            {
                assert(inst.input0 < i && inst.input1 < i);
                const float i0_lower = interval_vars[inst.input0].lower[j];
                const float i0_upper = interval_vars[inst.input0].upper[j];
                const float i1_lower = interval_vars[inst.input1].lower[j];
                const float i1_upper = interval_vars[inst.input1].upper[j];

                if (i0_lower >= i1_upper)
                {
                    remap[inst.input0][j] = 1;
                    remap[i][j] = 0;
                }
                else if (i1_lower >= i0_upper)
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
                assert(inst.input0 < i && inst.input1 < i);
                const float i0_lower = interval_vars[inst.input0].lower[j];
                const float i0_upper = interval_vars[inst.input0].upper[j];
                const float i1_lower = interval_vars[inst.input1].lower[j];
                const float i1_upper = interval_vars[inst.input1].upper[j];

                if (i0_upper <= i1_lower)
                {
                    remap[inst.input0][j] = 1;
                    remap[i][j] = 0;
                }
                else if (i1_upper <= i0_lower)
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

        std::array<float, kBatchSize> x_coords;
        std::array<float, kBatchSize> y_coords;

        for (int dy = 0; dy < tile.size; ++dy)
        {
            float y = tile.iy.lower + (dy + 0.5f) * y_range / tile.size;
            for (int dx = 0; dx < tile.size; ++dx)
            {
                x_coords[dy * tile.size + dx] = tile.ix.lower + (dx + 0.5f) * x_range / tile.size;
                y_coords[dy * tile.size + dx] = y;
            }
        }

        const std::array<float, kBatchSize> values = evaluate_batch(instructions, x_coords, y_coords);

        for (int dy = 0; dy < size; ++dy)
        {
            for (int dx = 0; dx < size; ++dx)
            {
                int global_x = px + dx;
                int global_y = kImageSize - 1 - (py + dy);
                assert(global_x >= 0 && global_x < kImageSize && global_y >= 0 && global_y < kImageSize);
                float v = values[dy * size + dx];
                image[global_y * kImageSize + global_x] = (v < 0.0f) ? -1.0f : 0.0f;
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

    Interval4 ix4 = {{ll.ix.lower, lr.ix.lower, ul.ix.lower, ur.ix.lower}, {ll.ix.upper, lr.ix.upper, ul.ix.upper, ur.ix.upper}};
    Interval4 iy4 = {{ll.iy.lower, lr.iy.lower, ul.iy.lower, ur.iy.lower}, {ll.iy.upper, lr.iy.upper, ul.iy.upper, ur.iy.upper}};

    Interval4 ir4 = evaluate_interval4(instructions, ix4, iy4);
    std::array<std::vector<Instruction>, 4> compacted_instructions;
    prune_instructions4(instructions, compacted_instructions);

    for (size_t i = 0; i < 4; ++i)
    {
        float lower = ir4.lower[i];
        float upper = ir4.upper[i];
        if (upper < 0.0f)
        {
            fill_region(image, tiles[i], -1.0f);
            continue;
        }
        if (lower >= 0.0f)
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
