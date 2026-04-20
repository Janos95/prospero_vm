#include "vm.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

std::string read_file(const char* filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::fprintf(stderr, "Failed to open file %s\n", filename);
        std::exit(1);
    }

    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void write_image(const std::vector<std::uint8_t>& image, const char* filename)
{
    FILE* file = std::fopen(filename, "wb");
    if (!file)
    {
        std::fprintf(stderr, "Failed to open file %s for writing\n", filename);
        std::exit(1);
    }

    std::fprintf(file, "P5\n%d %d\n255\n", kImageSize, kImageSize);
    std::fwrite(image.data(), 1, image.size(), file);

    std::fclose(file);
}

int main(int argc, char* argv[])
{
    auto start_load = std::chrono::high_resolution_clock::now();
    std::string program = read_file("prospero.vm");
    std::vector<Instruction> instructions = parse_instructions(program);
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_load = end_load - start_load;
    std::printf("Loaded instructions in %.2f ms\n", duration_load.count());

    auto start_setup = std::chrono::high_resolution_clock::now();
    int num_threads = prepare_thread_locals(instructions.size());
    auto end_setup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_setup = end_setup - start_setup;
    std::printf("Allocating tls for %d threads took %.2f ms\n", num_threads, duration_setup.count());

    auto start_alloc = std::chrono::high_resolution_clock::now();
    std::vector<std::uint8_t> image(kImageSize * kImageSize, 0);
    auto end_alloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_alloc = end_alloc - start_alloc;
    std::printf("Allocating image took %.2f ms\n", duration_alloc.count());

    constexpr int num_runs = 500;
    std::chrono::duration<double, std::milli> render_time{};
    for (int i = 0; i < num_runs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        render_image(instructions, image);
        auto end = std::chrono::high_resolution_clock::now();
        render_time += end - start;
    }

    std::printf("Rendered %dx at %.2fms/frame\n", num_runs, render_time.count() / num_runs);
    write_image(image, "out.ppm");
    return 0;
}
