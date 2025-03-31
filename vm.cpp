#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <fstream>
#include <string>
#include <vector>
#include <string_view>
#include <array>
#include <chrono>

constexpr int IMAGE_SIZE = 1024;
constexpr int TARGET_TILE_SIZE = 16; 
constexpr int BATCH_SIZE = TARGET_TILE_SIZE * TARGET_TILE_SIZE;
static_assert(IMAGE_SIZE % TARGET_TILE_SIZE == 0, "IMAGE_SIZE must be divisible by TARGET_TILE_SIZE");

struct Interval 
{ 
    float lower, upper; 
};

struct Interval4 
{ 
    alignas(16) float lower[4]; 
    alignas(16) float upper[4]; 
};

static std::vector<std::array<float, BATCH_SIZE>> thread_batch_vars;
static std::vector<Interval4> thread_interval4_vars;
static std::vector<std::array<int, 4>> thread_remap4;
#pragma omp threadprivate(thread_batch_vars, thread_interval4_vars, thread_remap4)

enum class OpCode { VarX, VarY, Const, Add, Sub, Mul, Max, Min, Neg, Square, Sqrt };

struct Instruction 
{
    float constant;
    int input0;
    int input1;
    OpCode op;
};

float parse_float(std::string_view sv) 
{
    char* endptr = nullptr;
    float value = std::strtof(sv.data(), &endptr);
    return value;
}

int parse_hex_var(std::string_view sv) 
{
    std::string_view hex_part = sv.substr(1);
    char* endptr = nullptr;
    errno = 0;
    long value_long = std::strtol(hex_part.data(), &endptr, 16);
    return (int)value_long;
}

void split(std::string_view line, std::vector<std::string_view>& out) 
{
    out.clear();
    size_t start = 0;
    size_t end = 0;

    while (start < line.size()) 
    {
        end = line.find(' ', start);
        if (end == std::string_view::npos) 
        {
            out.push_back(line.substr(start));
            break;
        } 
        else 
        {
            out.push_back(line.substr(start, end - start));
            start = end + 1;
        }
    }
}

void load_instructions(const char *filename, std::vector<Instruction>& instructions) 
{
    std::ifstream file(filename);
    if (!file) 
    {
        fprintf(stderr, "Failed to open file %s\n", filename);
        exit(1);
    }

    std::vector<std::string_view> tokens;
    tokens.reserve(16);
    std::string line;
    line.reserve(256);
    instructions.reserve(8000);

    while (std::getline(file, line)) 
    {
        if (line.empty() || line[0] == '#') 
        {
            continue;
        }

        split(line, tokens);

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
            fprintf(stderr, "Unknown operation: %s\n", tokens[1].data());
            continue;
        }

        instructions.push_back(inst);
    }
}

std::array<float, BATCH_SIZE> evaluate_batch(const std::vector<Instruction>& instructions, const std::array<float, BATCH_SIZE>& x_coords, const std::array<float, BATCH_SIZE>& y_coords) {
    std::array<float, BATCH_SIZE>* batch_vars = thread_batch_vars.data(); // get thread local data
    const size_t num_instructions = instructions.size();

#define LOOP(expr) for(int j = 0; j < BATCH_SIZE; j++) { batch_vars[i][j] = expr; }

    for(int i = 0; i < num_instructions; i++) {
        const Instruction& inst = instructions[i];
        switch(inst.op) {
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

Interval4 evaluate_interval4(const std::vector<Instruction>& instructions, const Interval4& x, const Interval4& y) {
    Interval4* interval_vars = thread_interval4_vars.data(); // get thread local data

    const size_t num_instructions = instructions.size();
    assert(thread_interval4_vars.size() >= num_instructions);

    for(size_t i = 0; i < num_instructions; ++i) {
        const Instruction& inst = instructions[i];
        switch(inst.op) {
            case OpCode::VarX: 
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = x.lower[j];
                    interval_vars[i].upper[j] = x.upper[j];
                }
                break;
            case OpCode::VarY: 
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = y.lower[j];
                    interval_vars[i].upper[j] = y.upper[j];
                }
                break;
            case OpCode::Const: 
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = inst.constant;
                    interval_vars[i].upper[j] = inst.constant;
                }
                break;
            case OpCode::Add:
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = interval_vars[inst.input0].lower[j] + interval_vars[inst.input1].lower[j];
                    interval_vars[i].upper[j] = interval_vars[inst.input0].upper[j] + interval_vars[inst.input1].upper[j];
                }
                break;
            case OpCode::Sub:
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = interval_vars[inst.input0].lower[j] - interval_vars[inst.input1].upper[j];
                    interval_vars[i].upper[j] = interval_vars[inst.input0].upper[j] - interval_vars[inst.input1].lower[j];
                }
                break;
            case OpCode::Mul: {
                for(int j = 0; j < 4; j++) {
                    float a = interval_vars[inst.input0].lower[j], b = interval_vars[inst.input0].upper[j];
                    float c = interval_vars[inst.input1].lower[j], d = interval_vars[inst.input1].upper[j];
                    float p1 = a*c, p2 = a*d, p3 = b*c, p4 = b*d;
                    interval_vars[i].lower[j] = min4(p1, p2, p3, p4);
                    interval_vars[i].upper[j] = max4(p1, p2, p3, p4);
                }
                break;
            }
            case OpCode::Max: {
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = max2(interval_vars[inst.input0].lower[j], interval_vars[inst.input1].lower[j]);
                    interval_vars[i].upper[j] = max2(interval_vars[inst.input0].upper[j], interval_vars[inst.input1].upper[j]);
                }
                break;
            }
            case OpCode::Min: {
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = min2(interval_vars[inst.input0].lower[j], interval_vars[inst.input1].lower[j]);
                    interval_vars[i].upper[j] = min2(interval_vars[inst.input0].upper[j], interval_vars[inst.input1].upper[j]);
                }
                break;
            }
            case OpCode::Neg:
                for(int j = 0; j < 4; j++) {
                    interval_vars[i].lower[j] = -interval_vars[inst.input0].upper[j];
                    interval_vars[i].upper[j] = -interval_vars[inst.input0].lower[j];
                }
                break;
            case OpCode::Square: {
                for(int j = 0; j < 4; j++) {
                    float a = interval_vars[inst.input0].lower[j], b = interval_vars[inst.input0].upper[j];
                    float sq_a = a*a, sq_b = b*b;
                    float min_val = min2(sq_a, sq_b);
                    if (a <= 0.0f && b >= 0.0f) min_val = 0.0f;
                    interval_vars[i].lower[j] = min_val;
                    interval_vars[i].upper[j] = max2(sq_a, sq_b);
                }
                break;
            }
            case OpCode::Sqrt: {
                for(int j = 0; j < 4; j++) {
                    float a = interval_vars[inst.input0].lower[j], b = interval_vars[inst.input0].upper[j];
                    float sqrt_a = sqrt(max2(0.0f, a));
                    float sqrt_b = sqrt(max2(0.0f, b));
                    interval_vars[i].lower[j] = (b < 0.0f) ? 0.0f : sqrt_a;
                    interval_vars[i].upper[j] = (b < 0.0f) ? 0.0f : sqrt_b;
                }
                break;
            }
        }
    }

    return interval_vars[num_instructions - 1];
}

void prune_instructions4(const std::vector<Instruction>& original_instructions, std::array<std::vector<Instruction>, 4>& compacted_instructions) {
    // Get thread local data
    Interval4* interval_vars = thread_interval4_vars.data();
    std::array<int, 4>* remap = thread_remap4.data();
    int remap_size = original_instructions.size();
    assert(remap_size <= thread_remap4.size());
    assert(thread_interval4_vars.size() >= remap_size);

    memset(remap, -1, remap_size * 4 * sizeof(int));

    // Mark the final instruction as needed 
    for(int j = 0; j < 4; j++) {
        remap[remap_size - 1][j] = 1;
    }

    // First we do a backwards pass to determine which instructions are needed
    for (int i = remap_size - 1; i >= 0; --i) {
        const Instruction& inst = original_instructions[i];
        for(int j = 0; j < 4; j++) {
            if(remap[i][j] == -1) continue;

            if(inst.op == OpCode::Max) {
                assert(inst.input0 < i && inst.input1 < i);
                const float i0_lower = interval_vars[inst.input0].lower[j];
                const float i0_upper = interval_vars[inst.input0].upper[j];
                const float i1_lower = interval_vars[inst.input1].lower[j];
                const float i1_upper = interval_vars[inst.input1].upper[j];

                // We "misuse" the remap array to store which input dominates the other one
                if (i0_lower >= i1_upper) { remap[inst.input0][j] = 1; remap[i][j] = 0; } // i0 dominates, mark with 0
                else if (i1_lower >= i0_upper) { remap[inst.input1][j] = 1; assert(remap[i][j] == 1); } // i1 dominates, already marked with 1
                else { remap[inst.input0][j] = 1; remap[inst.input1][j] = 1; remap[i][j] = 2; } // Overlap, mark with 2
            } else if(inst.op == OpCode::Min) {
                assert(inst.input0 < i && inst.input1 < i);
                const float i0_lower = interval_vars[inst.input0].lower[j];
                const float i0_upper = interval_vars[inst.input0].upper[j];
                const float i1_lower = interval_vars[inst.input1].lower[j];
                const float i1_upper = interval_vars[inst.input1].upper[j];

                if (i0_upper <= i1_lower) { remap[inst.input0][j] = 1; remap[i][j] = 0; } // i0 dominates, mark with 0
                else if (i1_upper <= i0_lower) { remap[inst.input1][j] = 1; assert(remap[i][j] == 1); } // i1 dominates, already marked with 1
                else { remap[inst.input0][j] = 1; remap[inst.input1][j] = 1; remap[i][j] = 2; } // Overlap, mark with 2
            } else {
                // propagate needed instructions
                if(inst.input0 != -1) remap[inst.input0][j] = 1;
                if(inst.input1 != -1) remap[inst.input1][j] = 1;
            }
        }
    }

    // Initialize the four compacted instruction streams
    for(int j = 0; j < 4; j++) {
        compacted_instructions[j].reserve(original_instructions.size());
    }

    // Second we do a forwards pass to compact the instructions and compute the input remapping
    for (int i = 0; i < remap_size; ++i) {
        for(int j = 0; j < 4; j++) {
            Instruction inst = original_instructions[i];
            if(remap[i][j] == -1) continue;

            if(inst.op == OpCode::Max || inst.op == OpCode::Min) {
                // if one of the inputs dominates the other one, we can get rid of the max/min and 
                // remap its output to the still valid remapped input
                if(remap[i][j] != 2) {
                    remap[i][j] = remap[i][j] == 0 ? remap[inst.input0][j] : remap[inst.input1][j];
                    continue;
                }
            }

            if(inst.input0 != -1) inst.input0 = remap[inst.input0][j];
            if(inst.input1 != -1) inst.input1 = remap[inst.input1][j];
            compacted_instructions[j].push_back(inst);
            remap[i][j] = compacted_instructions[j].size() - 1;
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
    int px = tile.px, py = tile.py, size = tile.size;
    for (int j = 0; j < size; ++j) 
    {
        for (int i = 0; i < size; ++i) 
        {
            int global_x = px + i;
            int global_y = IMAGE_SIZE - 1 - (py + j); // flip y, since image coordinates have origin in top left
            if (global_x >= 0 && global_x < IMAGE_SIZE && global_y >= 0 && global_y < IMAGE_SIZE) 
            {
                image[global_y * IMAGE_SIZE + global_x] = value;
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

    if (tile.size == TARGET_TILE_SIZE) 
    {
        float x_range = tile.ix.upper - tile.ix.lower;
        float y_range = tile.iy.upper - tile.iy.lower;

        std::array<float, BATCH_SIZE> x_coords;
        std::array<float, BATCH_SIZE> y_coords;

        for (int dy = 0; dy < tile.size; ++dy) 
        {
            float y = tile.iy.lower + (dy + 0.5f) * y_range / tile.size;
            for (int dx = 0; dx < tile.size; ++dx) 
            {
                x_coords[dy*tile.size + dx] = tile.ix.lower + (dx + 0.5f) * x_range / tile.size;
                y_coords[dy*tile.size + dx] = y;
            }
        }

        const std::array<float, BATCH_SIZE> values = evaluate_batch(instructions, x_coords, y_coords);

        for (int dy = 0; dy < size; ++dy) 
        {
            for (int dx = 0; dx < size; ++dx) 
            {
                int global_x = px + dx;
                int global_y = IMAGE_SIZE - 1 - (py + dy); // flip y, since image coordinates have origin in top left
                assert(global_x >= 0 && global_x < IMAGE_SIZE && global_y >= 0 && global_y < IMAGE_SIZE);
                float v = values[dy*size + dx];
                image[global_y * IMAGE_SIZE + global_x] = (v < 0.0f) ? -1.0f : 0.0f;
            }
        }

        return;
    } 

    float mid_x = ix.lower + (ix.upper - ix.lower) / 2.0f;
    float mid_y = iy.lower + (iy.upper - iy.lower) / 2.0f;

    int new_size = size / 2;
    assert(new_size >= TARGET_TILE_SIZE);

    Tile ll = {px, py, new_size, {ix.lower, mid_x}, {iy.lower, mid_y}};                       // lower left
    Tile lr = {px + new_size, py, new_size, {mid_x, ix.upper}, {iy.lower, mid_y}};            // lower right
    Tile ul = {px, py + new_size, new_size, {ix.lower, mid_x}, {mid_y, iy.upper}};            // upper left
    Tile ur = {px + new_size, py + new_size, new_size, {mid_x, ix.upper}, {mid_y, iy.upper}}; // upper right

    Tile tiles[4] = {ll, lr, ul, ur};

    Interval4 ix4 = {{ll.ix.lower, lr.ix.lower, ul.ix.lower, ur.ix.lower}, {ll.ix.upper, lr.ix.upper, ul.ix.upper, ur.ix.upper}}; 
    Interval4 iy4 = {{ll.iy.lower, lr.iy.lower, ul.iy.lower, ur.iy.lower}, {ll.iy.upper, lr.iy.upper, ul.iy.upper, ur.iy.upper}};

    Interval4 ir4 = evaluate_interval4(instructions, ix4, iy4);
    std::array<std::vector<Instruction>, 4> compacted_instructions;
    prune_instructions4(instructions, compacted_instructions);

    for(size_t i = 0; i < 4; i++) 
    {
        float lower = ir4.lower[i];
        float upper = ir4.upper[i];
        if (upper < 0.0f)
        {
            fill_region(image, tiles[i], -1.0f); // Entirely inside
            continue;
        } 
        if (lower >= 0.0f) 
        {
            // Entire outside, already 0, so we can return immediately
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
    Tile initial_tile = {0, 0, IMAGE_SIZE, {-1.0f, 1.0f}, {-1.0f, 1.0f}};
    #pragma omp parallel default(shared)
    {
        #pragma omp single
        solve_region(instructions, image, initial_tile);
    }
}

void write_image(const std::vector<float>& image, const char* filename) 
{
    FILE *file = fopen(filename, "wb");
    if (!file) 
    {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n255\n", IMAGE_SIZE, IMAGE_SIZE);

    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) 
    {
        unsigned char pixel = (image[i] < 0) ? 255 : 0; // inside white
        fwrite(&pixel, 1, 1, file);
    }

    fclose(file);
}

int main(int argc, char *argv[])  
{
    auto start_load = std::chrono::high_resolution_clock::now();
    std::vector<Instruction> instructions;
    load_instructions("prospero.vm", instructions);
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_load = end_load - start_load;
    printf("Loaded instructions in %.2f ms\n", duration_load.count());

    std::atomic_int num_threads = 0;
    auto start_setup = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        thread_batch_vars.resize(instructions.size());
        thread_interval4_vars.resize(instructions.size());
        thread_remap4.resize(instructions.size());
        ++num_threads;
    }
    auto end_setup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_setup = end_setup - start_setup;
    printf("Allocating tls for %d threads took %.2f ms\n", num_threads.load(), duration_setup.count());

    auto start_alloc = std::chrono::high_resolution_clock::now();
    std::vector<float> image(IMAGE_SIZE * IMAGE_SIZE, 0.0f);
    auto end_alloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_alloc = end_alloc - start_alloc;
    printf("Allocating image took %.2f ms\n", duration_alloc.count());

    constexpr int num_runs = 500;
    std::chrono::duration<double, std::milli> render_time{};
    for(size_t i = 0; i < num_runs; i++) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        evaluate_recursive(instructions, image);
        auto end = std::chrono::high_resolution_clock::now();
        render_time += end - start;
    }

    printf("Rendered %dx at %.2fms/frame\n", num_runs, render_time.count() / num_runs);
    write_image(image, "out.ppm");
    return 0;
}