#include "vm.h"
#include "simd.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>
#include <chrono>

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

#ifndef PROSPERO_FLAT_JOBS_CUTOFF
#define PROSPERO_FLAT_JOBS_CUTOFF 0
#endif

#ifndef PROSPERO_MODE_STATS
#define PROSPERO_MODE_STATS 0
#endif

#ifndef PROSPERO_PRUNE_STATS
#define PROSPERO_PRUNE_STATS 0
#endif

#ifndef PROSPERO_PRUNE_TIMING
#define PROSPERO_PRUNE_TIMING 0
#endif

#ifndef PROSPERO_PHASE_TIMING
#define PROSPERO_PHASE_TIMING 0
#endif

#ifndef PROSPERO_AXIS_MODES
#define PROSPERO_AXIS_MODES 0
#endif

#define PROSPERO_STRINGIFY_IMPL(x) #x
#define PROSPERO_STRINGIFY(x) PROSPERO_STRINGIFY_IMPL(x)
#define PROSPERO_PRAGMA(x) _Pragma(PROSPERO_STRINGIFY(x))
#define PROSPERO_BATCH_UNROLL_PRAGMA PROSPERO_PRAGMA(clang loop unroll_count(PROSPERO_BATCH_UNROLL))

constexpr int kBatchLanes = kBatchSize / simd::v4f::width;
constexpr int kBatchVectorsPerRow = kTargetTileSize / simd::v4f::width;
constexpr int kBatchUnroll = PROSPERO_BATCH_UNROLL;
constexpr int kTilesPerAxis = kImageSize / kTargetTileSize;
using Batch = std::array<simd::v4f, kBatchLanes>;
using XAxisBatch = std::array<simd::v4f, kBatchVectorsPerRow>;
using YAxisBatch = std::array<simd::v4f, kTargetTileSize>;

enum class ValueMode : uint8_t
{
    Scalar = 0,
    XOnly = 1,
    YOnly = 2,
    Full = 3,
};

static_assert(kBatchSize % simd::v4f::width == 0, "kBatchSize must be divisible by the SIMD width");
static_assert(kTargetTileSize % simd::v4f::width == 0, "kTargetTileSize must be divisible by the SIMD width");
static_assert(kBatchLanes % kBatchUnroll == 0, "kBatchLanes must be divisible by PROSPERO_BATCH_UNROLL");
static_assert(kBatchUnroll > 0, "PROSPERO_BATCH_UNROLL must be positive");

static std::vector<Batch> thread_batch_vars;
static std::vector<Interval4> thread_interval4_vars;
static std::vector<std::array<int, 4>> thread_remap4;
static std::vector<uint8_t> thread_live_mask4;
static std::vector<uint8_t> thread_emit_mask4;
static std::vector<uint8_t> thread_use0_mask4;
static std::vector<uint8_t> thread_is_scalar;
static std::vector<float> thread_scalar_value;
static std::vector<uint8_t> thread_value_mode;
static std::vector<XAxisBatch> thread_xonly_vars;
static std::vector<YAxisBatch> thread_yonly_vars;
static std::array<Batch, kTilesPerAxis> g_leaf_x_coords;
static std::array<Batch, kTilesPerAxis> g_leaf_y_coords;
static std::array<XAxisBatch, kTilesPerAxis> g_leaf_x_axis;
static std::array<YAxisBatch, kTilesPerAxis> g_leaf_y_axis;
static std::once_flag g_leaf_coord_once;
static constexpr std::array<std::uint32_t, 16> kMaskToBytes = [] {
    std::array<std::uint32_t, 16> table{};
    for (int mask = 0; mask < 16; ++mask)
    {
        std::uint32_t packed = 0;
        for (int lane = 0; lane < simd::v4f::width; ++lane)
        {
            if ((mask & (1 << lane)) != 0)
            {
                packed |= 0xFFu << (lane * 8);
            }
        }
        table[static_cast<std::size_t>(mask)] = packed;
    }
    return table;
}();

#if PROSPERO_MODE_STATS
static std::atomic<unsigned long long> g_mode_leaf_count{0};
static std::atomic<unsigned long long> g_mode_instruction_total{0};
static std::atomic<unsigned long long> g_mode_counts[4];
static std::vector<uint8_t> thread_mode_stats;
#pragma omp threadprivate(thread_mode_stats)
#endif

#if PROSPERO_PRUNE_STATS
static std::atomic<unsigned long long> g_prune_calls{0};
static std::atomic<unsigned long long> g_prune_input_total{0};
static std::atomic<unsigned long long> g_prune_live_total{0};
static std::atomic<unsigned long long> g_prune_emit_total{0};
static std::atomic<unsigned long long> g_prune_root_lane_total{0};
static std::atomic<unsigned long long> g_render_call_count{0};
#endif

#if PROSPERO_PRUNE_TIMING
static std::atomic<unsigned long long> g_prune_reverse_ns{0};
static std::atomic<unsigned long long> g_prune_forward_ns{0};
#endif

#if PROSPERO_PHASE_TIMING
enum class TimedPhase : uint8_t
{
    Interval = 0,
    Prune = 1,
    Leaf = 2,
    Fill = 3,
};

static std::atomic<unsigned long long> g_phase_ns[4];
static std::atomic<unsigned long long> g_phase_calls[4];
static std::atomic<unsigned long long> g_render_wall_ns{0};

struct ScopedPhaseTimer
{
    TimedPhase phase;
    std::chrono::high_resolution_clock::time_point start;

    explicit ScopedPhaseTimer(TimedPhase phase_in)
        : phase(phase_in), start(std::chrono::high_resolution_clock::now())
    {
        g_phase_calls[static_cast<size_t>(phase)].fetch_add(1, std::memory_order_relaxed);
    }

    ~ScopedPhaseTimer()
    {
        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        g_phase_ns[static_cast<size_t>(phase)].fetch_add(
            static_cast<unsigned long long>(elapsed),
            std::memory_order_relaxed);
    }
};
#endif

struct ProgramView
{
    const Instruction* data = nullptr;
    std::size_t count = 0;

    const Instruction& operator[](std::size_t i) const { return data[i]; }
    std::size_t size() const { return count; }
};

struct InstructionBuffer
{
    std::unique_ptr<Instruction[]> storage;
    std::size_t capacity = 0;
    std::size_t count = 0;

    void ensure_capacity(std::size_t required)
    {
        if (capacity < required)
        {
            storage = std::make_unique<Instruction[]>(required);
            capacity = required;
        }
        count = 0;
    }

    ProgramView view() const
    {
        return ProgramView{storage.get(), count};
    }
};

struct InstructionFrame
{
    std::array<InstructionBuffer, 4> compacted;
};

static std::deque<InstructionFrame> thread_instruction_frames;
static int thread_instruction_frame_depth = 0;
#pragma omp threadprivate(thread_batch_vars, thread_interval4_vars, thread_remap4, thread_live_mask4, thread_emit_mask4, thread_use0_mask4, thread_is_scalar, thread_scalar_value, thread_value_mode, thread_xonly_vars, thread_yonly_vars, thread_instruction_frames, thread_instruction_frame_depth)

InstructionFrame& acquire_instruction_frame(std::size_t instruction_count)
{
    if (thread_instruction_frame_depth == static_cast<int>(thread_instruction_frames.size()))
    {
        thread_instruction_frames.emplace_back();
    }

    InstructionFrame& frame = thread_instruction_frames[thread_instruction_frame_depth++];
    for (InstructionBuffer& instructions : frame.compacted)
    {
        instructions.ensure_capacity(instruction_count);
    }
    return frame;
}

void prepare_leaf_coord_batches()
{
    std::call_once(g_leaf_coord_once, [] {
        const float step = 2.0f / static_cast<float>(kImageSize);
        for (int tile_x = 0; tile_x < kTilesPerAxis; ++tile_x)
        {
            XAxisBatch& x_axis = g_leaf_x_axis[static_cast<std::size_t>(tile_x)];
            Batch& batch = g_leaf_x_coords[static_cast<std::size_t>(tile_x)];
            const int px = tile_x * kTargetTileSize;
            for (int dx = 0; dx < kTargetTileSize; dx += simd::v4f::width)
            {
                x_axis[static_cast<std::size_t>(dx / simd::v4f::width)] = simd::v4f::make(
                    -1.0f + static_cast<float>(px + dx + 0) * step + 0.5f * step,
                    -1.0f + static_cast<float>(px + dx + 1) * step + 0.5f * step,
                    -1.0f + static_cast<float>(px + dx + 2) * step + 0.5f * step,
                    -1.0f + static_cast<float>(px + dx + 3) * step + 0.5f * step);
            }
            for (int dy = 0; dy < kTargetTileSize; ++dy)
            {
                for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
                {
                    const int batch_idx = dy * kBatchVectorsPerRow + dxv;
                    batch[static_cast<std::size_t>(batch_idx)] = x_axis[static_cast<std::size_t>(dxv)];
                }
            }
        }

        for (int tile_y = 0; tile_y < kTilesPerAxis; ++tile_y)
        {
            YAxisBatch& y_axis = g_leaf_y_axis[static_cast<std::size_t>(tile_y)];
            Batch& batch = g_leaf_y_coords[static_cast<std::size_t>(tile_y)];
            const int py = tile_y * kTargetTileSize;
            for (int dy = 0; dy < kTargetTileSize; ++dy)
            {
                const float y = -1.0f + static_cast<float>(py + dy) * step + 0.5f * step;
                y_axis[static_cast<std::size_t>(dy)] = simd::v4f::splat(y);
                for (int dx = 0; dx < kTargetTileSize; dx += simd::v4f::width)
                {
                    const int batch_idx = dy * kBatchVectorsPerRow + dx / simd::v4f::width;
                    batch[static_cast<std::size_t>(batch_idx)] = y_axis[static_cast<std::size_t>(dy)];
                }
            }
        }
    });
}

void ensure_thread_local_capacity(std::size_t instruction_count)
{
    prepare_leaf_coord_batches();
    thread_batch_vars.resize(instruction_count);
    thread_interval4_vars.resize(instruction_count);
    thread_remap4.resize(instruction_count);
    thread_live_mask4.resize(instruction_count);
    thread_emit_mask4.resize(instruction_count);
    thread_use0_mask4.resize(instruction_count);
    thread_is_scalar.resize(instruction_count);
    thread_scalar_value.resize(instruction_count);
#if PROSPERO_AXIS_MODES
    thread_value_mode.resize(instruction_count);
    thread_xonly_vars.resize(instruction_count);
    thread_yonly_vars.resize(instruction_count);
#endif
}

void release_instruction_frame()
{
    assert(thread_instruction_frame_depth > 0);
    --thread_instruction_frame_depth;
}

struct InstructionFrameGuard
{
    explicit InstructionFrameGuard(std::size_t instruction_count) : frame(acquire_instruction_frame(instruction_count)) {}
    ~InstructionFrameGuard() { release_instruction_frame(); }

    InstructionFrame& frame;
};

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
    if (value_long < 0)
    {
        std::fprintf(stderr, "Register index out of range: %.*s\n", static_cast<int>(sv.size()), sv.data());
        std::exit(1);
    }
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

#if PROSPERO_AXIS_MODES
template <size_t N, typename Op>
inline void apply_unary_array(std::array<simd::v4f, N>& dst, const std::array<simd::v4f, N>& src, Op op)
{
    for (size_t i = 0; i < N; ++i)
    {
        dst[i] = op(src[i]);
    }
}

template <size_t N, typename Op>
inline void apply_binary_array(std::array<simd::v4f, N>& dst,
                               const std::array<simd::v4f, N>& lhs,
                               const std::array<simd::v4f, N>& rhs,
                               Op op)
{
    for (size_t i = 0; i < N; ++i)
    {
        dst[i] = op(lhs[i], rhs[i]);
    }
}

template <size_t N, typename Op>
inline void apply_scalar_left_array(std::array<simd::v4f, N>& dst,
                                    simd::v4f lhs,
                                    const std::array<simd::v4f, N>& rhs,
                                    Op op)
{
    for (size_t i = 0; i < N; ++i)
    {
        dst[i] = op(lhs, rhs[i]);
    }
}

template <size_t N, typename Op>
inline void apply_scalar_right_array(std::array<simd::v4f, N>& dst,
                                     const std::array<simd::v4f, N>& lhs,
                                     simd::v4f rhs,
                                     Op op)
{
    for (size_t i = 0; i < N; ++i)
    {
        dst[i] = op(lhs[i], rhs);
    }
}

template <typename Op>
inline void apply_x_y_to_full(Batch& dst, const XAxisBatch& x, const YAxisBatch& y, Op op)
{
    for (int dy = 0; dy < kTargetTileSize; ++dy)
    {
        const simd::v4f y_value = y[static_cast<size_t>(dy)];
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            dst[static_cast<size_t>(dy * kBatchVectorsPerRow + dxv)] =
                op(x[static_cast<size_t>(dxv)], y_value);
        }
    }
}

template <typename Op>
inline void apply_y_x_to_full(Batch& dst, const YAxisBatch& y, const XAxisBatch& x, Op op)
{
    for (int dy = 0; dy < kTargetTileSize; ++dy)
    {
        const simd::v4f y_value = y[static_cast<size_t>(dy)];
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            dst[static_cast<size_t>(dy * kBatchVectorsPerRow + dxv)] =
                op(y_value, x[static_cast<size_t>(dxv)]);
        }
    }
}

template <typename Op>
inline void apply_x_full_to_full(Batch& dst, const XAxisBatch& x, const Batch& full, Op op)
{
    for (int dy = 0; dy < kTargetTileSize; ++dy)
    {
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            const size_t idx = static_cast<size_t>(dy * kBatchVectorsPerRow + dxv);
            dst[idx] = op(x[static_cast<size_t>(dxv)], full[idx]);
        }
    }
}

template <typename Op>
inline void apply_full_x_to_full(Batch& dst, const Batch& full, const XAxisBatch& x, Op op)
{
    for (int dy = 0; dy < kTargetTileSize; ++dy)
    {
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            const size_t idx = static_cast<size_t>(dy * kBatchVectorsPerRow + dxv);
            dst[idx] = op(full[idx], x[static_cast<size_t>(dxv)]);
        }
    }
}

template <typename Op>
inline void apply_y_full_to_full(Batch& dst, const YAxisBatch& y, const Batch& full, Op op)
{
    for (int dy = 0; dy < kTargetTileSize; ++dy)
    {
        const simd::v4f y_value = y[static_cast<size_t>(dy)];
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            const size_t idx = static_cast<size_t>(dy * kBatchVectorsPerRow + dxv);
            dst[idx] = op(y_value, full[idx]);
        }
    }
}

template <typename Op>
inline void apply_full_y_to_full(Batch& dst, const Batch& full, const YAxisBatch& y, Op op)
{
    for (int dy = 0; dy < kTargetTileSize; ++dy)
    {
        const simd::v4f y_value = y[static_cast<size_t>(dy)];
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            const size_t idx = static_cast<size_t>(dy * kBatchVectorsPerRow + dxv);
            dst[idx] = op(full[idx], y_value);
        }
    }
}

template <typename Op>
inline void apply_unary_full(Batch& dst, const Batch& src, Op op)
{
    for (int i = 0; i < kBatchLanes; ++i)
    {
        dst[static_cast<size_t>(i)] = op(src[static_cast<size_t>(i)]);
    }
}

template <typename Op>
inline void apply_binary_full(Batch& dst, const Batch& lhs, const Batch& rhs, Op op)
{
    for (int i = 0; i < kBatchLanes; ++i)
    {
        dst[static_cast<size_t>(i)] = op(lhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(i)]);
    }
}

template <typename Op>
inline void apply_scalar_left_full(Batch& dst, simd::v4f lhs, const Batch& rhs, Op op)
{
    for (int i = 0; i < kBatchLanes; ++i)
    {
        dst[static_cast<size_t>(i)] = op(lhs, rhs[static_cast<size_t>(i)]);
    }
}

template <typename Op>
inline void apply_scalar_right_full(Batch& dst, const Batch& lhs, simd::v4f rhs, Op op)
{
    for (int i = 0; i < kBatchLanes; ++i)
    {
        dst[static_cast<size_t>(i)] = op(lhs[static_cast<size_t>(i)], rhs);
    }
}

void evaluate_batch(
    ProgramView instructions,
    std::size_t tile_x,
    std::size_t tile_y)
{
    using simd::v4f;

    Batch* batch_vars = thread_batch_vars.data();
    XAxisBatch* x_vars = thread_xonly_vars.data();
    YAxisBatch* y_vars = thread_yonly_vars.data();
    uint8_t* value_mode = thread_value_mode.data();
    float* scalar_value = thread_scalar_value.data();
    const XAxisBatch& x_coords = g_leaf_x_axis[tile_x];
    const YAxisBatch& y_coords = g_leaf_y_axis[tile_y];
    const size_t num_instructions = instructions.size();

    auto eval_binary = [&](size_t i, int lhs_idx, int rhs_idx, auto scalar_op, auto vector_op)
    {
        const ValueMode lhs_mode = static_cast<ValueMode>(value_mode[lhs_idx]);
        const ValueMode rhs_mode = static_cast<ValueMode>(value_mode[rhs_idx]);

        if (lhs_mode == ValueMode::Scalar && rhs_mode == ValueMode::Scalar)
        {
            value_mode[i] = static_cast<uint8_t>(ValueMode::Scalar);
            scalar_value[i] = scalar_op(scalar_value[lhs_idx], scalar_value[rhs_idx]);
            return;
        }

        if (lhs_mode == ValueMode::Scalar)
        {
            const v4f lhs = v4f::splat(scalar_value[lhs_idx]);
            switch (rhs_mode)
            {
                case ValueMode::XOnly:
                    value_mode[i] = static_cast<uint8_t>(ValueMode::XOnly);
                    apply_scalar_left_array(x_vars[i], lhs, x_vars[rhs_idx], vector_op);
                    return;
                case ValueMode::YOnly:
                    value_mode[i] = static_cast<uint8_t>(ValueMode::YOnly);
                    apply_scalar_left_array(y_vars[i], lhs, y_vars[rhs_idx], vector_op);
                    return;
                case ValueMode::Full:
                    value_mode[i] = static_cast<uint8_t>(ValueMode::Full);
                    apply_scalar_left_full(batch_vars[i], lhs, batch_vars[rhs_idx], vector_op);
                    return;
                case ValueMode::Scalar:
                    break;
            }
        }

        if (rhs_mode == ValueMode::Scalar)
        {
            const v4f rhs = v4f::splat(scalar_value[rhs_idx]);
            switch (lhs_mode)
            {
                case ValueMode::XOnly:
                    value_mode[i] = static_cast<uint8_t>(ValueMode::XOnly);
                    apply_scalar_right_array(x_vars[i], x_vars[lhs_idx], rhs, vector_op);
                    return;
                case ValueMode::YOnly:
                    value_mode[i] = static_cast<uint8_t>(ValueMode::YOnly);
                    apply_scalar_right_array(y_vars[i], y_vars[lhs_idx], rhs, vector_op);
                    return;
                case ValueMode::Full:
                    value_mode[i] = static_cast<uint8_t>(ValueMode::Full);
                    apply_scalar_right_full(batch_vars[i], batch_vars[lhs_idx], rhs, vector_op);
                    return;
                case ValueMode::Scalar:
                    break;
            }
        }

        if (lhs_mode == ValueMode::XOnly && rhs_mode == ValueMode::XOnly)
        {
            value_mode[i] = static_cast<uint8_t>(ValueMode::XOnly);
            apply_binary_array(x_vars[i], x_vars[lhs_idx], x_vars[rhs_idx], vector_op);
            return;
        }

        if (lhs_mode == ValueMode::YOnly && rhs_mode == ValueMode::YOnly)
        {
            value_mode[i] = static_cast<uint8_t>(ValueMode::YOnly);
            apply_binary_array(y_vars[i], y_vars[lhs_idx], y_vars[rhs_idx], vector_op);
            return;
        }

        value_mode[i] = static_cast<uint8_t>(ValueMode::Full);
        if (lhs_mode == ValueMode::Full && rhs_mode == ValueMode::Full)
        {
            apply_binary_full(batch_vars[i], batch_vars[lhs_idx], batch_vars[rhs_idx], vector_op);
        }
        else if (lhs_mode == ValueMode::XOnly && rhs_mode == ValueMode::YOnly)
        {
            apply_x_y_to_full(batch_vars[i], x_vars[lhs_idx], y_vars[rhs_idx], vector_op);
        }
        else if (lhs_mode == ValueMode::YOnly && rhs_mode == ValueMode::XOnly)
        {
            apply_y_x_to_full(batch_vars[i], y_vars[lhs_idx], x_vars[rhs_idx], vector_op);
        }
        else if (lhs_mode == ValueMode::XOnly && rhs_mode == ValueMode::Full)
        {
            apply_x_full_to_full(batch_vars[i], x_vars[lhs_idx], batch_vars[rhs_idx], vector_op);
        }
        else if (lhs_mode == ValueMode::Full && rhs_mode == ValueMode::XOnly)
        {
            apply_full_x_to_full(batch_vars[i], batch_vars[lhs_idx], x_vars[rhs_idx], vector_op);
        }
        else if (lhs_mode == ValueMode::YOnly && rhs_mode == ValueMode::Full)
        {
            apply_y_full_to_full(batch_vars[i], y_vars[lhs_idx], batch_vars[rhs_idx], vector_op);
        }
        else if (lhs_mode == ValueMode::Full && rhs_mode == ValueMode::YOnly)
        {
            apply_full_y_to_full(batch_vars[i], batch_vars[lhs_idx], y_vars[rhs_idx], vector_op);
        }
        else
        {
            assert(false);
        }
    };

    auto eval_unary = [&](size_t i, int src_idx, auto scalar_op, auto vector_op)
    {
        const ValueMode src_mode = static_cast<ValueMode>(value_mode[src_idx]);
        switch (src_mode)
        {
            case ValueMode::Scalar:
                value_mode[i] = static_cast<uint8_t>(ValueMode::Scalar);
                scalar_value[i] = scalar_op(scalar_value[src_idx]);
                break;
            case ValueMode::XOnly:
                value_mode[i] = static_cast<uint8_t>(ValueMode::XOnly);
                apply_unary_array(x_vars[i], x_vars[src_idx], vector_op);
                break;
            case ValueMode::YOnly:
                value_mode[i] = static_cast<uint8_t>(ValueMode::YOnly);
                apply_unary_array(y_vars[i], y_vars[src_idx], vector_op);
                break;
            case ValueMode::Full:
                value_mode[i] = static_cast<uint8_t>(ValueMode::Full);
                apply_unary_full(batch_vars[i], batch_vars[src_idx], vector_op);
                break;
        }
    };

    for (size_t i = 0; i < num_instructions; ++i)
    {
        const Instruction& inst = instructions[i];
        switch (inst.op)
        {
            case OpCode::VarX:
                value_mode[i] = static_cast<uint8_t>(ValueMode::XOnly);
                x_vars[i] = x_coords;
                break;
            case OpCode::VarY:
                value_mode[i] = static_cast<uint8_t>(ValueMode::YOnly);
                y_vars[i] = y_coords;
                break;
            case OpCode::Const:
                value_mode[i] = static_cast<uint8_t>(ValueMode::Scalar);
                scalar_value[i] = inst.constant;
                break;
            case OpCode::Add:
                eval_binary(i, inst.input0, inst.input1,
                            [](float a, float b) { return a + b; },
                            [](v4f a, v4f b) { return a + b; });
                break;
            case OpCode::Sub:
                eval_binary(i, inst.input0, inst.input1,
                            [](float a, float b) { return a - b; },
                            [](v4f a, v4f b) { return a - b; });
                break;
            case OpCode::Mul:
                eval_binary(i, inst.input0, inst.input1,
                            [](float a, float b) { return a * b; },
                            [](v4f a, v4f b) { return a * b; });
                break;
            case OpCode::Max:
                eval_binary(i, inst.input0, inst.input1,
                            [](float a, float b) { return std::fmax(a, b); },
                            [](v4f a, v4f b) { return simd::max(a, b); });
                break;
            case OpCode::Min:
                eval_binary(i, inst.input0, inst.input1,
                            [](float a, float b) { return std::fmin(a, b); },
                            [](v4f a, v4f b) { return simd::min(a, b); });
                break;
            case OpCode::Neg:
                eval_unary(i, inst.input0,
                           [](float a) { return -a; },
                           [](v4f a) { return -a; });
                break;
            case OpCode::Square:
                eval_unary(i, inst.input0,
                           [](float a) { return a * a; },
                           [](v4f a) { return a * a; });
                break;
            case OpCode::Sqrt:
                eval_unary(i, inst.input0,
                           [](float a) { return std::sqrt(a); },
                           [](v4f a) { return simd::sqrt(a); });
                break;
        }
    }
}
#else
const Batch& evaluate_batch(
    ProgramView instructions,
    const Batch& x_coords,
    const Batch& y_coords)
{
    using simd::v4f;

    Batch* batch_vars = thread_batch_vars.data();
    uint8_t* is_scalar = thread_is_scalar.data();
    float* scalar_value = thread_scalar_value.data();
    const size_t num_instructions = instructions.size();

    for (size_t i = 0; i < num_instructions; ++i)
    {
        const Instruction& inst = instructions[i];
        Batch& out = batch_vars[i];
        switch (inst.op)
        {
            case OpCode::VarX:
                out = x_coords;
                is_scalar[i] = 0;
                break;
            case OpCode::VarY:
                out = y_coords;
                is_scalar[i] = 0;
                break;
            case OpCode::Const:
            {
                is_scalar[i] = 1;
                scalar_value[i] = inst.constant;
                break;
            }
            case OpCode::Add:
            {
                const bool lhs_scalar = is_scalar[inst.input0] != 0;
                const bool rhs_scalar = is_scalar[inst.input1] != 0;
                if (lhs_scalar && rhs_scalar)
                {
                    is_scalar[i] = 1;
                    scalar_value[i] = scalar_value[inst.input0] + scalar_value[inst.input1];
                    break;
                }

                is_scalar[i] = 0;
                v4f* dst = out.data();
                if (lhs_scalar)
                {
                    const v4f lhs = v4f::splat(scalar_value[inst.input0]);
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs + rhs[j];
                    }
                }
                else if (rhs_scalar)
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f rhs = v4f::splat(scalar_value[inst.input1]);
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs[j] + rhs;
                    }
                }
                else
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs[j] + rhs[j];
                    }
                }
                break;
            }
            case OpCode::Sub:
            {
                const bool lhs_scalar = is_scalar[inst.input0] != 0;
                const bool rhs_scalar = is_scalar[inst.input1] != 0;
                if (lhs_scalar && rhs_scalar)
                {
                    is_scalar[i] = 1;
                    scalar_value[i] = scalar_value[inst.input0] - scalar_value[inst.input1];
                    break;
                }

                is_scalar[i] = 0;
                v4f* dst = out.data();
                if (lhs_scalar)
                {
                    const v4f lhs = v4f::splat(scalar_value[inst.input0]);
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs - rhs[j];
                    }
                }
                else if (rhs_scalar)
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f rhs = v4f::splat(scalar_value[inst.input1]);
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs[j] - rhs;
                    }
                }
                else
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs[j] - rhs[j];
                    }
                }
                break;
            }
            case OpCode::Mul:
            {
                const bool lhs_scalar = is_scalar[inst.input0] != 0;
                const bool rhs_scalar = is_scalar[inst.input1] != 0;
                if (lhs_scalar && rhs_scalar)
                {
                    is_scalar[i] = 1;
                    scalar_value[i] = scalar_value[inst.input0] * scalar_value[inst.input1];
                    break;
                }

                is_scalar[i] = 0;
                v4f* dst = out.data();
                if (lhs_scalar)
                {
                    const v4f lhs = v4f::splat(scalar_value[inst.input0]);
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs * rhs[j];
                    }
                }
                else if (rhs_scalar)
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f rhs = v4f::splat(scalar_value[inst.input1]);
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs[j] * rhs;
                    }
                }
                else
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = lhs[j] * rhs[j];
                    }
                }
                break;
            }
            case OpCode::Max:
            {
                const bool lhs_scalar = is_scalar[inst.input0] != 0;
                const bool rhs_scalar = is_scalar[inst.input1] != 0;
                if (lhs_scalar && rhs_scalar)
                {
                    is_scalar[i] = 1;
                    scalar_value[i] = std::fmax(scalar_value[inst.input0], scalar_value[inst.input1]);
                    break;
                }

                is_scalar[i] = 0;
                v4f* dst = out.data();
                if (lhs_scalar)
                {
                    const v4f lhs = v4f::splat(scalar_value[inst.input0]);
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = simd::max(lhs, rhs[j]);
                    }
                }
                else if (rhs_scalar)
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f rhs = v4f::splat(scalar_value[inst.input1]);
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = simd::max(lhs[j], rhs);
                    }
                }
                else
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = simd::max(lhs[j], rhs[j]);
                    }
                }
                break;
            }
            case OpCode::Min:
            {
                const bool lhs_scalar = is_scalar[inst.input0] != 0;
                const bool rhs_scalar = is_scalar[inst.input1] != 0;
                if (lhs_scalar && rhs_scalar)
                {
                    is_scalar[i] = 1;
                    scalar_value[i] = std::fmin(scalar_value[inst.input0], scalar_value[inst.input1]);
                    break;
                }

                is_scalar[i] = 0;
                v4f* dst = out.data();
                if (lhs_scalar)
                {
                    const v4f lhs = v4f::splat(scalar_value[inst.input0]);
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = simd::min(lhs, rhs[j]);
                    }
                }
                else if (rhs_scalar)
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f rhs = v4f::splat(scalar_value[inst.input1]);
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = simd::min(lhs[j], rhs);
                    }
                }
                else
                {
                    const v4f* lhs = batch_vars[inst.input0].data();
                    const v4f* rhs = batch_vars[inst.input1].data();
                    PROSPERO_BATCH_UNROLL_PRAGMA
                    for (int j = 0; j < kBatchLanes; ++j)
                    {
                        dst[j] = simd::min(lhs[j], rhs[j]);
                    }
                }
                break;
            }
            case OpCode::Neg:
            {
                if (is_scalar[inst.input0] != 0)
                {
                    is_scalar[i] = 1;
                    scalar_value[i] = -scalar_value[inst.input0];
                    break;
                }

                is_scalar[i] = 0;
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
                if (is_scalar[inst.input0] != 0)
                {
                    is_scalar[i] = 1;
                    const float value = scalar_value[inst.input0];
                    scalar_value[i] = value * value;
                    break;
                }

                is_scalar[i] = 0;
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
                if (is_scalar[inst.input0] != 0)
                {
                    is_scalar[i] = 1;
                    scalar_value[i] = std::sqrt(scalar_value[inst.input0]);
                    break;
                }

                is_scalar[i] = 0;
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
#endif

float min2(float a, float b) { return a < b ? a : b; }
float min4(float a, float b, float c, float d) { return min2(min2(a, b), min2(c, d)); }
float max2(float a, float b) { return a > b ? a : b; }
float max4(float a, float b, float c, float d) { return max2(max2(a, b), max2(c, d)); }

Interval4 evaluate_interval4(ProgramView instructions, const Interval4& x, const Interval4& y)
{
#if PROSPERO_PHASE_TIMING
    ScopedPhaseTimer phase_timer(TimedPhase::Interval);
#endif
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
            {
                const v4f lower0 = load_lower(inst.input0);
                const v4f upper0 = load_upper(inst.input0);
                const v4f lower1 = load_lower(inst.input1);
                const v4f upper1 = load_upper(inst.input1);
                store_interval(i,
                               simd::max(lower0, lower1),
                               simd::max(upper0, upper1));
                break;
            }
            case OpCode::Min:
            {
                const v4f lower0 = load_lower(inst.input0);
                const v4f upper0 = load_upper(inst.input0);
                const v4f lower1 = load_lower(inst.input1);
                const v4f upper1 = load_upper(inst.input1);
                store_interval(i,
                               simd::min(lower0, lower1),
                               simd::min(upper0, upper1));
                break;
            }
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

void prune_instructions4(
    ProgramView original_instructions,
    uint8_t root_live_mask,
    std::array<InstructionBuffer, 4>& compacted_instructions)
{
#if PROSPERO_PHASE_TIMING
    ScopedPhaseTimer phase_timer(TimedPhase::Prune);
#endif
    Interval4* interval_vars = thread_interval4_vars.data();
    std::array<int, 4>* remap = thread_remap4.data();
    uint8_t* live_mask = thread_live_mask4.data();
    uint8_t* emit_mask = thread_emit_mask4.data();
    uint8_t* use0_mask = thread_use0_mask4.data();
    const int instruction_count = static_cast<int>(original_instructions.size());
    assert(instruction_count > 0);
    assert(instruction_count <= static_cast<int>(thread_remap4.size()));
    assert(thread_interval4_vars.size() >= static_cast<size_t>(instruction_count));
    assert(thread_live_mask4.size() >= static_cast<size_t>(instruction_count));
    assert(thread_emit_mask4.size() >= static_cast<size_t>(instruction_count));
    assert(thread_use0_mask4.size() >= static_cast<size_t>(instruction_count));

    root_live_mask &= 0x0F;
    if (root_live_mask == 0)
    {
        return;
    }

#if PROSPERO_PRUNE_STATS
    unsigned long long live_count = 0;
    unsigned long long emit_count = 0;
#endif
#if PROSPERO_PRUNE_TIMING
    const auto reverse_start = std::chrono::high_resolution_clock::now();
#endif

    memset(live_mask, 0, static_cast<size_t>(instruction_count) * sizeof(*live_mask));
    live_mask[instruction_count - 1] = root_live_mask;

    for (int i = instruction_count - 1; i >= 0; --i)
    {
        const uint8_t live = live_mask[i];
        if (live == 0)
        {
            continue;
        }

        const Instruction& inst = original_instructions[i];

        if (inst.op == OpCode::Max)
        {
            assert(inst.input0 < i && inst.input1 < i);
            const uint8_t choose0 = static_cast<uint8_t>(simd::mask_bits(interval_vars[inst.input0].lower >= interval_vars[inst.input1].upper) & live);
            const uint8_t choose1 = static_cast<uint8_t>(simd::mask_bits(interval_vars[inst.input1].lower >= interval_vars[inst.input0].upper) & (live & ~choose0));
            const uint8_t keep = static_cast<uint8_t>(live & ~(choose0 | choose1));
            use0_mask[i] = choose0;
            emit_mask[i] = keep;
            live_mask[inst.input0] |= static_cast<uint8_t>(choose0 | keep);
            live_mask[inst.input1] |= static_cast<uint8_t>(choose1 | keep);
        }
        else if (inst.op == OpCode::Min)
        {
            assert(inst.input0 < i && inst.input1 < i);
            const uint8_t choose0 = static_cast<uint8_t>(simd::mask_bits(interval_vars[inst.input1].lower >= interval_vars[inst.input0].upper) & live);
            const uint8_t choose1 = static_cast<uint8_t>(simd::mask_bits(interval_vars[inst.input0].lower >= interval_vars[inst.input1].upper) & (live & ~choose0));
            const uint8_t keep = static_cast<uint8_t>(live & ~(choose0 | choose1));
            use0_mask[i] = choose0;
            emit_mask[i] = keep;
            live_mask[inst.input0] |= static_cast<uint8_t>(choose0 | keep);
            live_mask[inst.input1] |= static_cast<uint8_t>(choose1 | keep);
        }
        else
        {
            emit_mask[i] = live;
            if (inst.input0 != -1)
            {
                live_mask[inst.input0] |= live;
            }
            if (inst.input1 != -1)
            {
                live_mask[inst.input1] |= live;
            }
        }
    }

#if PROSPERO_PRUNE_TIMING
    const auto forward_start = std::chrono::high_resolution_clock::now();
    g_prune_reverse_ns.fetch_add(
        static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(forward_start - reverse_start).count()),
        std::memory_order_relaxed);
#endif

    for (int i = 0; i < instruction_count; ++i)
    {
        const uint8_t live = live_mask[i];
        if (live == 0)
        {
            continue;
        }

#if PROSPERO_PRUNE_STATS
        ++live_count;
#endif

        const uint8_t emitted = emit_mask[i];
        const uint8_t use0 = use0_mask[i];
        uint8_t active_lanes = live;
        while (active_lanes != 0)
        {
            const int j = __builtin_ctz(active_lanes);
            const uint8_t bit = static_cast<uint8_t>(1u << j);
            active_lanes &= static_cast<uint8_t>(active_lanes - 1);

            Instruction inst = original_instructions[i];
            if ((emitted & bit) == 0)
            {
                remap[i][j] = (use0 & bit) ? remap[inst.input0][j] : remap[inst.input1][j];
                continue;
            }

            if (inst.input0 != -1)
            {
                inst.input0 = remap[inst.input0][j];
            }
            if (inst.input1 != -1)
            {
                inst.input1 = remap[inst.input1][j];
            }
            InstructionBuffer& out = compacted_instructions[static_cast<size_t>(j)];
            out.storage[out.count] = inst;
            remap[i][j] = static_cast<int>(out.count);
            ++out.count;
#if PROSPERO_PRUNE_STATS
            ++emit_count;
#endif
        }
    }

#if PROSPERO_PRUNE_TIMING
    const auto forward_end = std::chrono::high_resolution_clock::now();
    g_prune_forward_ns.fetch_add(
        static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(forward_end - forward_start).count()),
        std::memory_order_relaxed);
#endif

#if PROSPERO_PRUNE_STATS
    g_prune_calls.fetch_add(1, std::memory_order_relaxed);
    g_prune_input_total.fetch_add(static_cast<unsigned long long>(instruction_count), std::memory_order_relaxed);
    g_prune_live_total.fetch_add(live_count, std::memory_order_relaxed);
    g_prune_emit_total.fetch_add(emit_count, std::memory_order_relaxed);
    g_prune_root_lane_total.fetch_add(static_cast<unsigned long long>(__builtin_popcount(static_cast<unsigned int>(root_live_mask))), std::memory_order_relaxed);
#endif
}

struct Tile
{
    int px, py, size;
    Interval ix, iy;
};

void fill_region(std::vector<std::uint8_t>& image, Tile tile, std::uint8_t value)
{
#if PROSPERO_PHASE_TIMING
    ScopedPhaseTimer phase_timer(TimedPhase::Fill);
#endif
    int px = tile.px;
    int py = tile.py;
    int size = tile.size;
    for (int j = 0; j < size; ++j)
    {
        int global_y = kImageSize - 1 - (py + j);
        assert(px >= 0 && px + size <= kImageSize && global_y >= 0 && global_y < kImageSize);
        std::memset(image.data() + global_y * kImageSize + px, value, static_cast<std::size_t>(size));
    }
}

void render_leaf_tile(ProgramView instructions, std::vector<std::uint8_t>& image, Tile tile)
{
#if PROSPERO_PHASE_TIMING
    ScopedPhaseTimer phase_timer(TimedPhase::Leaf);
#endif
#if PROSPERO_MODE_STATS
    thread_mode_stats.resize(instructions.size());
    unsigned long long counts[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < instructions.size(); ++i)
    {
        const Instruction& inst = instructions[i];
        uint8_t mode = 0;
        switch (inst.op)
        {
            case OpCode::VarX:
                mode = 1;
                break;
            case OpCode::VarY:
                mode = 2;
                break;
            case OpCode::Const:
                mode = 0;
                break;
            case OpCode::Neg:
            case OpCode::Square:
            case OpCode::Sqrt:
                mode = thread_mode_stats[inst.input0];
                break;
            default:
            {
                const uint8_t lhs = thread_mode_stats[inst.input0];
                const uint8_t rhs = thread_mode_stats[inst.input1];
                if (lhs == 3 || rhs == 3)
                {
                    mode = 3;
                }
                else if (lhs == 0)
                {
                    mode = rhs;
                }
                else if (rhs == 0)
                {
                    mode = lhs;
                }
                else if (lhs == rhs)
                {
                    mode = lhs;
                }
                else
                {
                    mode = 3;
                }
                break;
            }
        }
        thread_mode_stats[i] = mode;
        ++counts[mode];
    }
    ++g_mode_leaf_count;
    g_mode_instruction_total += instructions.size();
    for (int i = 0; i < 4; ++i)
    {
        g_mode_counts[i] += counts[i];
    }
#endif
    int px = tile.px;
    int py = tile.py;
    int size = tile.size;
#if PROSPERO_AXIS_MODES
    const std::size_t tile_x = static_cast<std::size_t>(px / kTargetTileSize);
    const std::size_t tile_y = static_cast<std::size_t>(py / kTargetTileSize);
    const size_t result_idx = instructions.size() - 1;
    evaluate_batch(instructions, tile_x, tile_y);
    const ValueMode mode = static_cast<ValueMode>(thread_value_mode[result_idx]);
    const simd::v4f zero = simd::v4f::splat(0.0f);

    if (mode == ValueMode::Scalar)
    {
        fill_region(image, tile, thread_scalar_value[result_idx] < 0.0f ? 255 : 0);
        return;
    }

    if (mode == ValueMode::XOnly)
    {
        const XAxisBatch& values = thread_xonly_vars[result_idx];
        std::uint32_t packed_row[kBatchVectorsPerRow];
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            const uint8_t mask = static_cast<uint8_t>(simd::mask_bits(values[static_cast<size_t>(dxv)] < zero));
            packed_row[dxv] = kMaskToBytes[mask];
        }
        for (int dy = 0; dy < size; ++dy)
        {
            const int global_y = kImageSize - 1 - (py + dy);
            std::uint8_t* row = image.data() + global_y * kImageSize + px;
            for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
            {
                std::memcpy(row + dxv * simd::v4f::width, &packed_row[dxv], sizeof(packed_row[dxv]));
            }
        }
        return;
    }

    if (mode == ValueMode::YOnly)
    {
        const YAxisBatch& values = thread_yonly_vars[result_idx];
        for (int dy = 0; dy < size; ++dy)
        {
            const int global_y = kImageSize - 1 - (py + dy);
            std::uint8_t* row = image.data() + global_y * kImageSize + px;
            const uint8_t mask = static_cast<uint8_t>(simd::mask_bits(values[static_cast<size_t>(dy)] < zero));
            std::memset(row, mask == 0 ? 0 : 255, static_cast<size_t>(size));
        }
        return;
    }

    const Batch& values = thread_batch_vars[result_idx];
    for (int dy = 0; dy < size; ++dy)
    {
        const int global_y = kImageSize - 1 - (py + dy);
        std::uint8_t* row = image.data() + global_y * kImageSize + px;
        for (int dxv = 0; dxv < kBatchVectorsPerRow; ++dxv)
        {
            const int batch_idx = dy * kBatchVectorsPerRow + dxv;
            const uint8_t mask = static_cast<uint8_t>(simd::mask_bits(values[static_cast<size_t>(batch_idx)] < zero));
            const std::uint32_t packed = kMaskToBytes[mask];
            std::memcpy(row + dxv * simd::v4f::width, &packed, sizeof(packed));
        }
    }
#else
    const Batch& x_coords = g_leaf_x_coords[static_cast<std::size_t>(px / kTargetTileSize)];
    const Batch& y_coords = g_leaf_y_coords[static_cast<std::size_t>(py / kTargetTileSize)];
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
                image[global_y * kImageSize + global_x] = (lanes[lane] < 0.0f) ? 255 : 0;
            }
        }
    }
#endif
}

void assign_constant_tile_programs(Tile tile,
                                   std::uint8_t constant_value,
                                   std::vector<ExportedTileProgram>& programs)
{
    const int tiles_per_axis = kImageSize / kTargetTileSize;
    const int start_x = tile.px / kTargetTileSize;
    const int start_y = tile.py / kTargetTileSize;
    const int span = tile.size / kTargetTileSize;

    for (int dy = 0; dy < span; ++dy)
    {
        for (int dx = 0; dx < span; ++dx)
        {
            const int tile_x = start_x + dx;
            const int tile_y = start_y + dy;
            const int index = tile_y * tiles_per_axis + tile_x;
            ExportedTileProgram& program = programs[static_cast<size_t>(index)];
            program.px = tile_x * kTargetTileSize;
            program.py = tile_y * kTargetTileSize;
            program.is_constant = true;
            program.constant_value = constant_value;
            program.instructions.clear();
        }
    }
}

void assign_dynamic_tile_program(ProgramView instructions,
                                 Tile tile,
                                 std::vector<ExportedTileProgram>& programs)
{
    const int tiles_per_axis = kImageSize / kTargetTileSize;
    const int tile_x = tile.px / kTargetTileSize;
    const int tile_y = tile.py / kTargetTileSize;
    const int index = tile_y * tiles_per_axis + tile_x;
    ExportedTileProgram& program = programs[static_cast<size_t>(index)];
    program.px = tile.px;
    program.py = tile.py;
    program.is_constant = false;
    program.constant_value = 0;
    program.instructions.assign(instructions.data, instructions.data + instructions.size());
}

void extract_tile_programs_impl(ProgramView instructions,
                                Tile tile,
                                std::vector<ExportedTileProgram>& programs)
{
    const int size = tile.size;
    const Interval ix = tile.ix;
    const Interval iy = tile.iy;

    if (size == kTargetTileSize)
    {
        assign_dynamic_tile_program(instructions, tile, programs);
        return;
    }

    const float mid_x = ix.lower + (ix.upper - ix.lower) / 2.0f;
    const float mid_y = iy.lower + (iy.upper - iy.lower) / 2.0f;
    const int new_size = size / 2;

    Tile tiles[4] = {
        {tile.px, tile.py, new_size, {ix.lower, mid_x}, {iy.lower, mid_y}},
        {tile.px + new_size, tile.py, new_size, {mid_x, ix.upper}, {iy.lower, mid_y}},
        {tile.px, tile.py + new_size, new_size, {ix.lower, mid_x}, {mid_y, iy.upper}},
        {tile.px + new_size, tile.py + new_size, new_size, {mid_x, ix.upper}, {mid_y, iy.upper}},
    };

    const Interval4 ix4 = {
        simd::v4f::make(tiles[0].ix.lower, tiles[1].ix.lower, tiles[2].ix.lower, tiles[3].ix.lower),
        simd::v4f::make(tiles[0].ix.upper, tiles[1].ix.upper, tiles[2].ix.upper, tiles[3].ix.upper)
    };
    const Interval4 iy4 = {
        simd::v4f::make(tiles[0].iy.lower, tiles[1].iy.lower, tiles[2].iy.lower, tiles[3].iy.lower),
        simd::v4f::make(tiles[0].iy.upper, tiles[1].iy.upper, tiles[2].iy.upper, tiles[3].iy.upper)
    };

    const Interval4 ir4 = evaluate_interval4(instructions, ix4, iy4);
    float lower[4];
    float upper[4];
    ir4.lower.store(lower);
    ir4.upper.store(upper);
    uint8_t recurse_mask = 0;

    for (int i = 0; i < 4; ++i)
    {
        if (upper[i] < 0.0f)
        {
            assign_constant_tile_programs(tiles[i], 255, programs);
        }
        else if (lower[i] >= 0.0f)
        {
            assign_constant_tile_programs(tiles[i], 0, programs);
        }
        else
        {
            recurse_mask |= static_cast<uint8_t>(1u << i);
        }
    }

    if (recurse_mask == 0)
    {
        return;
    }

    InstructionFrameGuard frame_guard(instructions.size());
    std::array<InstructionBuffer, 4>& compacted_instructions = frame_guard.frame.compacted;
    prune_instructions4(instructions, recurse_mask, compacted_instructions);

    for (uint8_t active = recurse_mask; active != 0; active &= static_cast<uint8_t>(active - 1))
    {
        const int i = __builtin_ctz(active);
        const ProgramView child_program = compacted_instructions[static_cast<size_t>(i)].view();
        extract_tile_programs_impl(child_program, tiles[i], programs);
    }
}

template <bool UseTasks>
void solve_region_impl(ProgramView instructions, std::vector<std::uint8_t>& image, Tile tile)
{
    int px = tile.px;
    int py = tile.py;
    int size = tile.size;
    Interval ix = tile.ix;
    Interval iy = tile.iy;

    if (tile.size == kTargetTileSize)
    {
        render_leaf_tile(instructions, image, tile);
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
    float lower[4];
    float upper[4];
    ir4.lower.store(lower);
    ir4.upper.store(upper);
    uint8_t recurse_mask = 0;

    for (size_t i = 0; i < 4; ++i)
    {
        if (upper[i] < 0.0f)
        {
            fill_region(image, tiles[i], 255);
            continue;
        }
        if (lower[i] >= 0.0f)
        {
            continue;
        }
        recurse_mask |= static_cast<uint8_t>(1u << i);
    }

    if (recurse_mask == 0)
    {
        return;
    }

    InstructionFrameGuard frame_guard(instructions.size());
    std::array<InstructionBuffer, 4>& compacted_instructions = frame_guard.frame.compacted;
    prune_instructions4(instructions, recurse_mask, compacted_instructions);

    for (uint8_t active = recurse_mask; active != 0; active &= static_cast<uint8_t>(active - 1))
    {
        const int i = __builtin_ctz(active);
        Tile child_tile = tiles[i];
        ProgramView child_program = compacted_instructions[static_cast<size_t>(i)].view();
#if defined(_OPENMP)
        if constexpr (UseTasks)
        {
            #pragma omp task default(shared) firstprivate(child_program, child_tile) if(size > 64)
            solve_region_impl<true>(child_program, image, child_tile);
        }
        else
#endif
        {
            solve_region_impl<false>(child_program, image, child_tile);
        }
    }

#if defined(_OPENMP)
    if constexpr (UseTasks)
    {
        #pragma omp taskwait
    }
#endif
}

void evaluate_recursive(const std::vector<Instruction>& instructions, std::vector<std::uint8_t>& image)
{
    Tile initial_tile = {0, 0, kImageSize, {-1.0f, 1.0f}, {-1.0f, 1.0f}};
    const ProgramView program{instructions.data(), instructions.size()};
    #pragma omp parallel default(shared)
    {
        #pragma omp single
        solve_region_impl<true>(program, image, initial_tile);
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
        ensure_thread_local_capacity(instruction_count);
        ++num_threads;
    }

    return num_threads.load();
}

void render_image(const std::vector<Instruction>& instructions, std::vector<std::uint8_t>& image)
{
#if PROSPERO_PHASE_TIMING
    const auto render_start = std::chrono::high_resolution_clock::now();
#endif
    if (image.size() != static_cast<size_t>(kImageSize * kImageSize))
    {
        image.assign(kImageSize * kImageSize, 0);
    }
    else
    {
        std::fill(image.begin(), image.end(), 0);
    }

    evaluate_recursive(instructions, image);

#if PROSPERO_PHASE_TIMING
    const auto render_end = std::chrono::high_resolution_clock::now();
    g_render_wall_ns.fetch_add(
        static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(render_end - render_start).count()),
        std::memory_order_relaxed);
#endif

#if PROSPERO_MODE_STATS
    std::fprintf(stderr,
                 "mode_stats leaves=%llu avg_inst=%.2f scalar=%llu x=%llu y=%llu full=%llu\n",
                 g_mode_leaf_count.load(),
                 g_mode_leaf_count.load() == 0 ? 0.0 :
                     static_cast<double>(g_mode_instruction_total.load()) / static_cast<double>(g_mode_leaf_count.load()),
                 g_mode_counts[0].load(),
                 g_mode_counts[1].load(),
                 g_mode_counts[2].load(),
                 g_mode_counts[3].load());
#endif

#if PROSPERO_PRUNE_STATS
    const unsigned long long renders = g_render_call_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (renders == 500)
    {
#if PROSPERO_PHASE_TIMING
        const unsigned long long interval_ns = g_phase_ns[static_cast<size_t>(TimedPhase::Interval)].load(std::memory_order_relaxed);
        const unsigned long long prune_ns = g_phase_ns[static_cast<size_t>(TimedPhase::Prune)].load(std::memory_order_relaxed);
        const unsigned long long leaf_ns = g_phase_ns[static_cast<size_t>(TimedPhase::Leaf)].load(std::memory_order_relaxed);
        const unsigned long long fill_ns = g_phase_ns[static_cast<size_t>(TimedPhase::Fill)].load(std::memory_order_relaxed);
        const unsigned long long interval_calls = g_phase_calls[static_cast<size_t>(TimedPhase::Interval)].load(std::memory_order_relaxed);
        const unsigned long long prune_calls_timed = g_phase_calls[static_cast<size_t>(TimedPhase::Prune)].load(std::memory_order_relaxed);
        const unsigned long long leaf_calls = g_phase_calls[static_cast<size_t>(TimedPhase::Leaf)].load(std::memory_order_relaxed);
        const unsigned long long fill_calls = g_phase_calls[static_cast<size_t>(TimedPhase::Fill)].load(std::memory_order_relaxed);
        const unsigned long long render_wall_ns = g_render_wall_ns.load(std::memory_order_relaxed);
        const unsigned long long phase_total_ns = interval_ns + prune_ns + leaf_ns + fill_ns;
        std::fprintf(stderr,
                     "phase_timing wall=%.3fms interval=%.3fms prune=%.3fms leaf=%.3fms fill=%.3fms shares(interval=%.3f prune=%.3f leaf=%.3f fill=%.3f) calls(interval=%llu prune=%llu leaf=%llu fill=%llu)\n",
                     static_cast<double>(render_wall_ns) / 1.0e6,
                     static_cast<double>(interval_ns) / 1.0e6,
                     static_cast<double>(prune_ns) / 1.0e6,
                     static_cast<double>(leaf_ns) / 1.0e6,
                     static_cast<double>(fill_ns) / 1.0e6,
                     phase_total_ns == 0 ? 0.0 : static_cast<double>(interval_ns) / static_cast<double>(phase_total_ns),
                     phase_total_ns == 0 ? 0.0 : static_cast<double>(prune_ns) / static_cast<double>(phase_total_ns),
                     phase_total_ns == 0 ? 0.0 : static_cast<double>(leaf_ns) / static_cast<double>(phase_total_ns),
                     phase_total_ns == 0 ? 0.0 : static_cast<double>(fill_ns) / static_cast<double>(phase_total_ns),
                     interval_calls,
                     prune_calls_timed,
                     leaf_calls,
                     fill_calls);
#endif
        const unsigned long long calls = g_prune_calls.load(std::memory_order_relaxed);
        const unsigned long long input_total = g_prune_input_total.load(std::memory_order_relaxed);
        const unsigned long long live_total = g_prune_live_total.load(std::memory_order_relaxed);
        const unsigned long long emit_total = g_prune_emit_total.load(std::memory_order_relaxed);
        const unsigned long long root_lane_total = g_prune_root_lane_total.load(std::memory_order_relaxed);
        std::fprintf(stderr,
                     "prune_stats calls=%llu avg_input=%.2f avg_live=%.2f avg_emit=%.2f avg_root_lanes=%.2f live_ratio=%.3f\n",
                     calls,
                     calls == 0 ? 0.0 : static_cast<double>(input_total) / static_cast<double>(calls),
                     calls == 0 ? 0.0 : static_cast<double>(live_total) / static_cast<double>(calls),
                     calls == 0 ? 0.0 : static_cast<double>(emit_total) / static_cast<double>(calls),
                     calls == 0 ? 0.0 : static_cast<double>(root_lane_total) / static_cast<double>(calls),
                     input_total == 0 ? 0.0 : static_cast<double>(live_total) / static_cast<double>(input_total));
#if PROSPERO_PRUNE_TIMING
        const unsigned long long reverse_ns = g_prune_reverse_ns.load(std::memory_order_relaxed);
        const unsigned long long forward_ns = g_prune_forward_ns.load(std::memory_order_relaxed);
        std::fprintf(stderr,
                     "prune_timing reverse=%.3fms forward=%.3fms reverse_share=%.3f\n",
                     static_cast<double>(reverse_ns) / 1.0e6,
                     static_cast<double>(forward_ns) / 1.0e6,
                     (reverse_ns + forward_ns) == 0 ? 0.0 :
                        static_cast<double>(reverse_ns) / static_cast<double>(reverse_ns + forward_ns));
#endif
    }
#endif
}

std::vector<ExportedTileProgram> extract_tile_programs(const std::vector<Instruction>& instructions)
{
    ensure_thread_local_capacity(instructions.size());

    const int tiles_per_axis = kImageSize / kTargetTileSize;
    std::vector<ExportedTileProgram> programs(static_cast<size_t>(tiles_per_axis * tiles_per_axis));
    const Tile initial_tile = {0, 0, kImageSize, {-1.0f, 1.0f}, {-1.0f, 1.0f}};
    const ProgramView program{instructions.data(), instructions.size()};
    extract_tile_programs_impl(program, initial_tile, programs);

    for (const ExportedTileProgram& tile_program : programs)
    {
        assert(tile_program.is_constant || !tile_program.instructions.empty());
    }

    return programs;
}
#undef PROSPERO_BATCH_UNROLL_PRAGMA
#undef PROSPERO_PRAGMA
#undef PROSPERO_STRINGIFY
#undef PROSPERO_STRINGIFY_IMPL
