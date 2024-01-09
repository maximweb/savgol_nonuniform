# Savitzky Golay filter for nonuniformly distributed independent data

A python implementation of Savitzky Golay filter based on the algorithm proposed by
Gorry, 1991 [1].

It is based on the calculation of orthogonal polynomail parameters for each window. The algorithim proposed by Gorry requires the recursive calculation of parameters over `np.arange(0, polyorder+1)`, while the parameters for different windows can be calculated simultaneously.

With partially uniformly distributed independent data, e.g. `x=[0, 1, 2, 3, 4, 5, 10, 20, 30, 40 50, 100, 200, 300, 400, 500]`, in mind, the performance is increased by identifying repeating windows in $\Delta x_{window}-x_{window,center}$. The Gorry algorithm is then only calculated on the `np.unique` windows.

## Performance

I tested the performance only rudimentarly on a sine signal superimposed with noise. For small to medium length signals ($N_{points}=10^2$ to $10^5$), it ranges between the performance of `scipy.signal.savgol_filter` (only for uniform data) and a simple reference implementation of bare looped polynomail fitting for many cases, except for very large window sizes.

## Reference

[1] Gorry, P. A. (2002). "General least-squares smoothing and differentiation of nonuniformly spaced data by the convolution method." Analytical Chemistry 63(5): 534-536. (Link to publisher.)[https://pubs.acs.org/doi/abs/10.1021/ac00005a031]