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

The implementation uses several techniques to optimize expression evaluation. The core approach uses a divide and conquer strategy, recursively subdividing the image into smaller tiles. Interval arithmetic is used to efficiently discard tiles where all pixels are guaranteed to be either entirely positive or entirely negative.

During the interval arithmetic tracing process, we record intermediate results which can be used to simplify the expression for subsequent evaluations.

For small tiles, we use a simple batched evaluator that processes all pixels in a 16x16 tile in one pass. The interval arithmetic and expression simplification are also performed in batches of four for better instruction parallelism.

The divide and conquer process is multi-threaded using OpenMP.

### Failed Optimizations

I experimented with several other optimization techniques that didn't provide significant performance improvements:

I attempted to explicitly use SIMD instructions for both the batch evaluator and interval evaluator. However, this
did not yield better performance. Apparently, the loops are already SIMD-friendly enough for the compiler to 
automatically vectorize them.

I also explored various approaches to reduce memory allocations. They didn't provide measurable performance 
improvements though. I guess the system allocator is smarter than what I could come up with :)

Currently, the main bottleneck is interval evaluation, particularly expression simplification, which requires 
two passes through the instruction list.