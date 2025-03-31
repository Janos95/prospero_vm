# Prospero Challenge Implementation

A small, self-contained C++ implementation of Matt Keeter's [Prospero Challenge](https://www.mattkeeter.com/projects/prospero/). 

## Performance

On a MacBook Air (M2), it renders the 1024x1024 image in about 2ms. Running `make run` yields:

```
Loaded instructions in 2.36 ms
Generating image with 8 threads...
Rendered 500x at 1.945671ms/frame
```

## Implementation Details

The implementation uses several techniques to optimize expression evaluation:

1. **Divide and Conquer**: Recursively subdivides the image into smaller tiles, using interval arithmetic to discard tiles guaranteed to be outside or inside the implicit function.

2. **Expression Simplification**: Records intermediate results during interval arithmetic tracing to simplify subsequent evaluations.

3. **Batched Evaluation**: Uses a simple batched evaluator for small tiles, processing all pixels in a 16x16 tile in one pass.

4. **Batch Processing**: Performs interval arithmetic and expression simplification in batches of four.

5. **Parallelization**: Uses OpenMP for multi-threading.

### Additional Optimizations

I experimented with several other optimizations:
- Explicit SIMD instructions for batch and interval evaluation (removed as compiler already optimized these loops)
- Various memory allocation reduction techniques (minimal performance impact)

Currently, the main bottleneck is interval evaluation, particularly expression simplification, which requires two passes through the instruction list.