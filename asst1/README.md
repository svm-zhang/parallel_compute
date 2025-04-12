# Assignment 1: Performance Analysis on a Quad-Core CPU #

## Overview ##

This assignment is intended to help you develop an understanding of the two primary forms of parallel execution present in a modern multi-core CPU:

1. SIMD execution within a single processing core
2. Parallel execution using multiple cores

You will also gain experience measuring and reasoning about the
performance of parallel programs (a challenging, but important, skill you will
use throughout this class). This assignment involves only a small amount of
programming, but a lot of analysis!

__All write-ups are currently based on results obtained from running on
Apple M1 chip.__

## Write-up

- [program 1](./prog1_mandelbrot_threads/README.md)
- [program 2](./prog2_vecintrin/README.md)
- [program 3](./prog3_mandelbrot_ispc/README.md)


## What About ARM-Based Macs? ##

For those with access to a new Apple ARM-based laptop, try changing the ISPC compilation target to `neon-i32x8` and the compilation arch to `aarch64` in Programs 3, 4, and 5. The other programs only use GCC and should produce the correct target. Produce a report of performance of the various programs on a new Apple ARM-based laptop. The staff is curious about what you will find.  What speedups are you observing from SIMD execution? Those without access to a modern Macbook could try to use ARM-based servers that are available on a cloud provider like AWS, although it has not been tested by the staff. Make sure that you reset the ISPC compilation target to `avx2-i32x8` and the compilation arch to `x86-64` before you submit the assignment because we will be testing your solutions on the myth machines!

## For the Curious (highly recommended) ##

Want to know about ISPC and how it was created? One of the two creators of ISPC, Matt Pharr, wrote an __amazing blog post__ on the history of its development called [The story of ispc](https://pharr.org/matt/blog/2018/04/30/ispc-all).  It really touches on many issues of parallel system design -- in particular the value of limited scope vs general programming languages.  IMHO it's a must read for CS149 students!


## Resources and Notes ##

-  Extensive ISPC documentation and examples can be found at
  <http://ispc.github.io/>
-  Zooming into different locations of the mandelbrot image can be quite
  fascinating
-  Intel provides a lot of supporting material about AVX2 vector instructions at
  <http://software.intel.com/en-us/avx/>.  
-  The [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/) is very useful.
