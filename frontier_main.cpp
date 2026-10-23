// Benchmark driver for the term-based ("frontier") renderer.
//
//   ./frontier [prospero.vm]
//
// Prints timing statistics, the term count and the image checksum, and writes
// the image to out_frontier.pgm (y pointing up).

#include "frontier.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

namespace
{
// Expected results for prospero.vm, matching the published WebAssembly build.
constexpr long long kExpectedChecksum = 33668415;
constexpr int kExpectedTerms = 666;

long long checksum()
{
    const std::uint8_t* img = frontier::image();
    long long sum = 0;
    for (int i = 0; i < frontier::kImageSize * frontier::kImageSize; i++) sum += img[i];
    return sum;
}

void write_pgm(const char* filename)
{
    FILE* file = std::fopen(filename, "wb");
    if (!file)
    {
        std::fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    const int n = frontier::kImageSize;
    std::fprintf(file, "P5\n%d %d\n255\n", n, n);
    const std::uint8_t* img = frontier::image();
    for (int r = n - 1; r >= 0; r--) std::fwrite(img + r * n, 1, static_cast<std::size_t>(n), file);
    std::fclose(file);
}
}  // namespace

int main(int argc, char* argv[])
{
    const char* path = argc > 1 ? argv[1] : "prospero.vm";
    std::ifstream file(path);
    if (!file)
    {
        std::fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }
    const std::string program((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    const auto load_start = std::chrono::steady_clock::now();
    const int count = frontier::load(program.c_str());
    const auto load_end = std::chrono::steady_clock::now();
    if (count <= 0)
    {
        std::fprintf(stderr, "Failed to parse %s (error %d)\n", path, count);
        return 1;
    }
    std::printf("Loaded %d instructions in %.2f ms\n", count,
                std::chrono::duration<double, std::milli>(load_end - load_start).count());

    constexpr int kWarmup = 4;
    constexpr int kRuns = 500;
    for (int i = 0; i < kWarmup; i++) frontier::render();

    double sum = 0.0, sum_sq = 0.0, min_ms = 1e9, max_ms = 0.0;
    for (int i = 0; i < kRuns; i++)
    {
        const auto start = std::chrono::steady_clock::now();
        frontier::render();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        sum += ms;
        sum_sq += ms * ms;
        min_ms = std::fmin(min_ms, ms);
        max_ms = std::fmax(max_ms, ms);
    }
    const double mean = sum / kRuns;
    const double stddev = std::sqrt(std::fmax(0.0, sum_sq / kRuns - mean * mean));
    std::printf("Rendered %dx: mean %.3f ms, std %.3f ms, min %.3f ms, max %.3f ms\n", kRuns, mean, stddev, min_ms,
                max_ms);

    const long long sum_pixels = checksum();
    std::printf("Terms: %d (%d not rasterized), checksum: %lld\n", frontier::term_count(), frontier::skipped_terms(),
                sum_pixels);
    write_pgm("out_frontier.pgm");

    if (argc <= 1 && (sum_pixels != kExpectedChecksum || frontier::term_count() != kExpectedTerms))
    {
        std::fprintf(stderr, "Unexpected result for prospero.vm: expected checksum %lld and %d terms\n",
                     kExpectedChecksum, kExpectedTerms);
        return 2;
    }
    return 0;
}
