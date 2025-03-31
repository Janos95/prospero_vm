# Prospero Virtual Machine

A small, self-contained C++ implementation of Matt Keeter's [Prospero Challenge](https://www.mattkeeter.com/projects/prospero/). 

<img src="out_4k.png" alt="4K Render" width="800">

## Performance

On a MacBook Air (M2), it renders the 1024x1024 image in about **2ms**. Running `make run` produces:

```
Loaded instructions in 1.67 ms
Allocating tls for 8 threads took 2.97 ms
Allocating image took 0.35 ms
Rendered 500x at 1.96ms/frame
```

A 4096×4096 render takes about **4ms** for the expression evaluation. This was somewhat surprising to me, but 
I think it can be explained by a combination of factors. For example for a 1024x1024 image the leaf tiles
are too small to be contained inside of the shape, so we have to explicitly evaluate all leaf tiles. 
Also the multi-threading seems to be slightly more effective at the larger size.
Anyhow, the full breakdown for 4k is:

```
Loaded instructions in 1.66 ms
Allocating tls for 8 threads took 2.85 ms
Allocating image took 5.55 ms
Rendered 500x at 4.30ms/frame
```

The single threaded performance comes in at about **8.69ms** for an image of size 1024 and **20.98ms** for a 4k image.

## Implementation Details

The implementation uses several techniques to optimize expression evaluation. The core approach uses a divide and conquer strategy, recursively subdividing the image into smaller tiles. Interval arithmetic is used to efficiently discard tiles where all pixels are guaranteed to be either entirely positive or entirely negative.

During the interval arithmetic tracing process, we record intermediate results which can be used to simplify the expression for subsequent evaluations.

For small tiles, we use a simple batched evaluator that processes all pixels in a 16x16 tile in one pass. The interval arithmetic and expression simplification are also performed in batches of four for better instruction parallelism.

The divide and conquer process is multi-threaded using OpenMP.

### Failed Optimizations Attempts

I experimented with several other optimization techniques that didn't provide significant performance improvements:

I attempted to explicitly use SIMD instructions for both the batch evaluator and interval evaluator. However, this
did not yield better performance. Apparently, the loops are already SIMD-friendly enough for the compiler to 
automatically vectorize them.

I also explored various approaches to reduce memory allocations and overall memory footpring. 
They didn't provide measurable performance improvements though. I guess the system allocator 
is smarter than what I could come up with :)

I also investigated a completely different approach: If you look at the expression long enough, you notice that it is 
basically a big union of about 700 fairly simple expressions (on average about 20 instructions), i.e. it has 
the form min(e1, ... , e_700). 
If you visualize the expressions, they are all primitives localized in various regions of the image. 
So it seems natural to parse out all these subexpressions, find a conservative bounding box for each of 
them and then in parallel rasterize each expression into the final image.
Unfortunately, it turns out that extracing the individual expressions from their joint instruction list is already
fairly slow. At least my somewhat optimized implementation took more than a millisecond to create the 
700 instruction lists.