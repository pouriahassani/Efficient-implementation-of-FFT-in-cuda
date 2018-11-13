# Efficient-implementation-of-FFT-in-cuda


this is an implementation of the [fast furier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) on the [Cuda](https://en.wikipedia.org/wiki/CUDA) platform.
to the best of our knowledge optimizations proposed in [this paper](https://dl.acm.org/citation.cfm?id=1413373) will yield the best performance.

some of the optimizations are listed below:
1. Use shared memory to compute FFT for small values of N, e.g., 1024 or 512.
2. Use hierarchical FFT on small shared memory FFT modules for larger values of N.
3. Use a higher radix (4 or 8). The following figure is 16-point FFT using radix 4. In most cases, radix 4 is faster than 2.
4. Use different radix values at different stages of your algorithm. Note that N=2^M and M = 23, 24 and 25. Therefore, for example for N=25, you may decide to execute 12 stages with radix 4 and one last stage with radix 2, or any other combination which you think is faster.

## Running the Code

#### Compile: nvcc    -O2    fft_main.cu     fft.cu    -o efft

#### Execute: ./fft   0  M


#### Contriutions

contribuations are welcome. please [email](mailto:poria19964@gmail.com) me.
