// Browser entry points for the term-based ("frontier") renderer, matching the
// functions the blog post's runner.js calls via ccall / Module._prospero_*.

#include "frontier.h"

#include <cmath>
#include <cstdint>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define PROSPERO_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
static double now_ms() { return emscripten_get_now(); }
#else
#include <chrono>
#define PROSPERO_EXPORT extern "C"
static double now_ms()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif

namespace
{
double g_min_ms = 0.0;
double g_max_ms = 0.0;
double g_std_ms = 0.0;
}  // namespace

PROSPERO_EXPORT int prospero_load(const char* text) { return frontier::load(text); }

// The renderer is single-threaded; kept so the page's runner can call it unconditionally.
PROSPERO_EXPORT void prospero_set_threads(int) {}

// Renders `warmup` untimed frames, then `iterations` timed ones. Returns the mean
// frame time in milliseconds; std (population), min and max are available below.
PROSPERO_EXPORT double prospero_benchmark(int warmup, int iterations)
{
    for (int i = 0; i < warmup; i++) frontier::render();
    double sum = 0.0, sum_sq = 0.0;
    g_min_ms = 1e9;
    g_max_ms = 0.0;
    for (int i = 0; i < iterations; i++)
    {
        const double start = now_ms();
        frontier::render();
        const double ms = now_ms() - start;
        sum += ms;
        sum_sq += ms * ms;
        if (ms < g_min_ms) g_min_ms = ms;
        if (ms > g_max_ms) g_max_ms = ms;
    }
    if (iterations <= 0)
    {
        g_min_ms = g_max_ms = g_std_ms = 0.0;
        return 0.0;
    }
    const double mean = sum / iterations;
    g_std_ms = std::sqrt(std::fmax(0.0, sum_sq / iterations - mean * mean));
    return mean;
}

PROSPERO_EXPORT double prospero_std_ms() { return g_std_ms; }
PROSPERO_EXPORT double prospero_min_ms() { return g_min_ms; }
PROSPERO_EXPORT double prospero_max_ms() { return g_max_ms; }
PROSPERO_EXPORT int prospero_instructions() { return frontier::instruction_count(); }
PROSPERO_EXPORT int prospero_frontier_terms() { return frontier::term_count(); }

PROSPERO_EXPORT int prospero_checksum()
{
    const std::uint8_t* img = frontier::image();
    int sum = 0;
    for (int i = 0; i < frontier::kImageSize * frontier::kImageSize; i++) sum += img[i];
    return sum;
}

// Pointer to the 1024x1024 image (one byte per pixel, row 0 is y = -1).
PROSPERO_EXPORT const std::uint8_t* prospero_image_data() { return frontier::image(); }
