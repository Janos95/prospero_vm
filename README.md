# Prospero Virtual Machine

Self-contained C++ renderers for Matt Keeter's [Prospero Challenge](https://www.mattkeeter.com/projects/prospero/): evaluate a program of about 7,900 math instructions over [-1, 1]², color each pixel by the sign of the result, and do it as fast as possible.

<img src="out_4k.png" alt="4K Render" width="400">

There are two renderers in this repo:

- **`frontier`** (`frontier.cpp`): splits the expression into its ~700 separate shapes and renders each one inside a tight bounding box. This is the fastest version and the one running on [the blog post](https://janosmeny.com/blog/prospero-challenge/).
- **`vm`** (`vm.cpp`): recursive tiling with interval arithmetic and expression pruning, multi-threaded with OpenMP.

## Building

```
make check          # build frontier and verify it reproduces the published image
make run-frontier   # benchmark the term-based renderer
make run            # benchmark the tiling renderer (OpenMP)
make run-single     # same, single-threaded
make wasm-web       # browser build of frontier for the blog post (needs emsdk in ../emsdk)
```

On macOS the Makefile uses Apple clang with Homebrew's libomp (`brew install libomp`); elsewhere it uses `-fopenmp` with GCC or clang.

## Performance

Single-threaded frame times for the 1024×1024 image, all measured on the same machine (a 2-core cloud VM, so absolute numbers are slow; the ratios are what matter):

| Version | Renderer | ms / frame |
| --- | --- | --- |
| March 2025 | first tiling version | ~21 |
| April 2026 | `vm`: tiling with pruning, axis modes, SIMD | ~11 |
| April 2026 | `frontier`: per-shape rendering | ~6 |

For reference, the March 2025 version rendered the image in about 2 ms on a MacBook Air (M2) with 8 threads, and in about 8.7 ms single-threaded.

## The term-based renderer (`frontier`)

If you look at the expression long enough, you notice that it is basically a big union of about 700 fairly simple shapes (about 20 instructions each on average): it has the form min(e₁, …, e₇₀₀), and each shape is localized somewhere in the image. So instead of evaluating the whole expression everywhere, `frontier` renders each shape on its own:

1. **Split.** Walk down from the root through the nested `min` nodes. Every operand that is not itself a `min` is a term (666 for `prospero.vm`).
2. **Compile each term.** Collect the term's sub-expression (at most 128 instructions) and renumber it. In one forward pass, fold constants, classify each node as constant / x-only / y-only / both, track affine nodes (`cx·x + cy·y + c0`) and nodes that are provably non-negative. Rewrite `sqrt(e) − k` as `e − k²` where that keeps the sign.
3. **Bound it.** Push the condition "term ≤ 0" backwards through the term with interval arithmetic. That gives a range for each affine node, i.e. a pair of half-planes. Clip the square [-1, 1]² against them (Sutherland–Hodgman) to get a tight polygon and its pixel bounding box. Terms that are provably empty are skipped (87 of the 666).
4. **Rasterize.** If the term has no `min` and nothing mixing x and y except `max`, the box is simply filled. Otherwise subdivide the box with interval arithmetic: drop rectangles that are outside, fill ones that are inside, and brute-force rectangles of at most 4,096 pixels. In the brute-force step, x-only nodes are evaluated once per span and y-only nodes once per row.

I originally dismissed this approach (see below) because extracting the terms was slow. The trick that makes it work is to never copy the full program: each frame walks only the `min` tree and each term's own nodes, with generation stamps instead of clearing per-instruction arrays.

### Reconstruction note

The source for this renderer was lost after it shipped on the blog. In October 2026 it was reconstructed from the WebAssembly build on the blog post. `frontier.cpp` reproduces that build **pixel for pixel** on `prospero.vm` (checksum 33,668,415, 666 terms), which `make check` verifies. Two notes:

- Exact output needs `-ffp-contract=off`, and the code mirrors WebAssembly's `min`/`max` behavior for NaN and signed zero.
- The blog also had a multi-threaded browser build. Its threading was not reconstructed; `frontier` is single-threaded. The `wasm-web` target builds a single-threaded drop-in for the page.

## The tiling renderer (`vm`)

The image is recursively subdivided into four tiles. Interval arithmetic on each tile discards tiles that are entirely inside or outside the shape; all four children are evaluated at once with 4-wide SIMD.

Each subdivision also simplifies the expression for the children. A reverse pass marks which instructions each child still needs, dropping the losing side of a `min` or `max` whenever the intervals decide it. A forward pass then compacts the surviving instructions into a smaller program per child.

16×16 leaf tiles are evaluated in batches. The evaluator tracks whether each value depends only on x, only on y, or on neither, and only does full per-pixel work where both are involved. There is a small portable SIMD layer (`simd.h`) for NEON, SSE and WebAssembly. Subdivision is multi-threaded with OpenMP tasks, and there is a threaded WebAssembly build (`make wasm-omp`).

### Things that didn't help (March 2025)

I attempted to explicitly use SIMD instructions for both the batch evaluator and interval evaluator. However, this did not yield better performance. Apparently, the loops are already SIMD-friendly enough for the compiler to automatically vectorize them.

I also explored various approaches to reduce memory allocations and overall memory footprint. They didn't provide measurable performance improvements though. I guess the system allocator is smarter than what I could come up with :)

I also tried parsing out the ~700 sub-expressions, finding a conservative bounding box for each and rasterizing them in parallel. My first implementation took more than a millisecond just to create the 700 instruction lists. A later version of this idea became `frontier`.
