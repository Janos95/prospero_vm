#pragma once

// Term-based ("frontier") Prospero renderer.
//
// The Prospero expression is a big union, min(t_1, ..., t_n), of small shapes.
// This renderer splits that min-tree into its terms and renders each one on its
// own: it folds constants, finds the affine (linear in x and y) sub-expressions,
// pushes the condition "term <= 0" backwards to bound those affine pieces, clips
// the square [-1, 1]^2 against the resulting half-planes, and only evaluates
// pixels inside the clipped bounding box.
//
// This file was reconstructed in October 2026 from the WebAssembly build shipped
// with the blog post (April 2026), whose source had been lost. It reproduces that
// build's output pixel for pixel on prospero.vm (checksum 33,668,415, 666 terms).
// Pixel-exact output needs -ffp-contract=off (see Makefile).

#include <cstdint>

namespace frontier
{
constexpr int kImageSize = 1024;

// Parses a Prospero program. Returns the instruction count, -1 for an unknown
// opcode, or -2 if the program has more than 10,000 instructions.
int load(const char* text);

// Renders the loaded program. Pixels inside the shape (value < 0 or NaN) are 255,
// everything else is 0. Row r of the image is y = -1 + 2r/1023 (row 0 is the bottom).
void render();

const std::uint8_t* image();
int instruction_count();
int term_count();      // terms found in the top-level min-tree by the last render()
int skipped_terms();   // terms the last render() did not rasterize (proven empty, or too large)
}
