# Deep Unrolled Projected Gradient Descent for WSRMax

This GitHub repository hosts the Python code of a deep-unfolding algorithm for weighted sum rate maximization (WSRMax) precoding design in multiuser MIMO systems subject to a sum power constraint.

## Problem formulation

Consider a downlink multiuser MIMO syste, where a base station equipped with $N$ antennas serves $K$ single-antenna users. Let $s_ k$ be the data symbol intended to user $k$ and let $\mathbf{h}_ k\in\mathbb{C}^{N\times 1}$ be the channel between the base station and user $k$. With linear precoding, the $N\times 1$ transmitted signal vector is

$$\mathbf{x}=\mathbf{V}\mathbf{s}=\sum_ {k=1}^{K}{\mathbf{v}_ ks_ k},\tag{1}$$

where $\mathbf{s}=[s_ 1,\cdots,s_ k,\cdots,s_ K]^\mathrm{T}$, and $\mathbf{V}=[\mathbf{v}_ 1,\cdots,\mathbf{v}_ k,\cdots,\mathbf{v}_ K]$ with $\mathbf{v}_ k$ being  the precoding vector for user $k$. Symbols are assumed to be zero-mean and uncorrelated across users so that $\mathbb{E}\{\mathbf{s}\mathbf{s}^\mathrm{H}\}=\mathbf{I}_ K$. The received signal at user $k$ is

$$y_ k=\mathbf{h}_ k^\mathrm{H}\mathbf{v}_ ks_ k + \sum_ {j\neq k}\mathbf{h}_ k\mathbf{v}_ js_ j + n_ k,\tag{2}$$

where $n_ k\sim\mathcal{CN}(0,\sigma_ \mathrm{n}^2)$ represents independent additive white Gaussian noise with power $\sigma_ \mathrm{n}^2$. It follows that the signal-to-interference-plus-noise-ratio (SINR) of user $k$ is

$$\mathrm{SINR}_ k=\frac{\lvert\mathbf{h}_ k^\mathrm{H}\mathbf{v}_ k\rvert}{\sum_ {j\neq k}\lvert\mathbf{h}_ k^\mathrm{H}\mathbf{v}_ j\rvert+\sigma_ \mathrm{n}^2}.\tag{3}$$

We seek to maximize the weighted sum rate (WSR) subject to a total transmit power constraint, i.e., 

$$\max_ {\mathbf{V}}\quad f(\mathbf{V})=\sum_ {k=1}^{K}\alpha_ k\log_ 2(1+\mathrm{SINR}_ k),\qquad\mathrm{s.t.}\quad\mathrm{tr}(\mathbf{V}\mathbf{V}^\mathrm{H})\leq P,\tag{4}$$

where $\alpha_ k$ indicates the user priority and <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;P" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;\small&space;P" title="\small P" /></a> is the maximum transmit power at the base station. 

The WSRMax problem (4) is known to be NP-hard. Currently,  the iterative weighted minimum mean square error (WMMSE) algorithm developed in [[1]](#WMMSE_Shi)  is the most popular method to handle this problem, which is guaranteed to find a stationary solution. The iterative progress can be summarized as follows:

1. Update the receivers:  $\beta_ k\leftarrow\frac{\mathbf{h}_ k^\mathrm{H}\mathbf{v}_ k}{\mathbf{h}_ k^\mathrm{H}(\sum_ {j=1}^K\mathbf{w}_ j\mathbf{w}_ j^\mathrm{H})\mathbf{h}_ k+\sigma_ \mathrm{n}^2},\\;\forall k$.

2. Update the MSE weights:  $\mu_ k\leftarrow\frac{\mathbf{h}_ k^\mathrm{H}(\sum_ {j=1}^K\mathbf{w}_ j\mathbf{w}_ j^\mathrm{H})\mathbf{h}_ k+\sigma_ \mathrm{n}^2}{\mathbf{h}_ k^\mathrm{H}(\sum_ {j\neq k}\mathbf{w}_ j\mathbf{w}_ j^\mathrm{H})\mathbf{h}_ k+\sigma_ \mathrm{n}^2},\\;\forall k$.

3. Update the precoding matrix:

   $$\mathbf{V}\leftarrow\arg\min_ {\mathbf{V}}\\;\mathrm{tr}(\mathbf{V}^\mathrm{H}\mathbf{A}\mathbf{V}-\mathbf{V}^\mathrm{H}\mathbf{B}-\mathbf{B}^\mathrm{H}\mathbf{V}),\quad\mathrm{s.t.}\\;\mathrm{tr}(\mathbf{V}\mathbf{V}^\mathrm{H})\leq P,\tag{5}$$
   
   where $\mathbf{A}=\sum_ {k=1}^K{\mu_ k\lvert\beta_ k\rvert^2\mathbf{h}_ k\mathbf{h}_ k^\mathrm{H}}$, and $\mathbf{B}=[\mathbf{b}_ 1,\cdots,\mathbf{b}_ k,\cdots,\mathbf{b}_ K]$ with $\mathbf{b}_ k=\mu_ k\beta_ k\mathbf{h}_ k$. Problem (5) is a convex quadratically constrained quadratic programming (QCQP) problem and thus can be easily solved (e.g., by using Lagrange dual method).

## Proposed Algorithm

We propose the novel application of **deep unfolding** [[2]](#L2O_Chen) to the WSRmax problem. The idea is to map each iteration of a **projected gradient descent (PGD) algorithm** for solving (4) to a network layer and thereby obtain a network architecture called **deep unrolled PGD**, in which the iterative parameter is modeled as learnable structures.

Specifically, the WSR objective has a gradient with respect to $\mathbf{V}$ as

$$\nabla f(\mathbf{V})=\mathbf{A}\mathbf{V}-\mathbf{B},\tag{5}$$

where $\mathbf{A}=\sum_{k=1}^{K}{\mu_ k\lvert\beta_ k\rvert^2\mathbf{h}_ k\mathbf{h}_ k^\mathrm{H}}$, and $\mathbf{B}=[\mathbf{b}_ 1,\cdots,\mathbf{b}_ k,\cdots,\mathbf{b}_ K]$ with $\mathbf{b}_ k=\mu_ k\beta_ k\mathbf{h}_ k$. Applying PGD to (4), at the $t$-th iteration, we have the following updating formula:

$$\mathbf{V}^{t+1}=\Pi_ {\mathcal{C}}\left(\mathbf{V}^t-\gamma\left[\mathbf{A}\mathbf{V}^t-\mathbf{B}\right]\right),\tag{6}$$

where $\Pi_ \mathcal{C}(\mathbf{V})=\frac{\sqrt{P}\mathbf{V}}{\mathrm{ReLU}(\lVert\mathbf{V})\rVert-\sqrt{P})+\sqrt{P}}$ denotes the projected operator. The intuition is that (6) can be interpreted as a a network layer, in which $\mathbf{A}$ refers to the 'weights', $\mathbf{B}$ refers to  the 'bias', and the projected operator $\Pi_ {\mathcal{C}}$ represents the 'nonlinear activation function'. As a result, the PGD algorithm applied to the WSRMax problem can be implemented using a deep network, as illustrated in Fig. 1.

![unrolled PGD for WSRmax precoding](/Unrolled_PGD_WSRmax.png)

<center><p><font size="3"><em>Fig 1. Deep Unrolled PGD based WSRMax precoding</em></font><br/></p></center>

In this way, the iterative parameter $\{\beta_ k\}_ {k=1}^{K}$ and $\{\mu_ k\}_ {k=1}^K$ can be treated as learnable structures and their values can be learned from data instead of hand-tuning.

## Usage

In this repository, the user can find:

- `run_unrolled_PGD_WSRMax.py`, which implements the proposed deep unrolled PGD in Python 3.6.13 and Tensorflow 1.15.0.
- `wmmse_algorithm.py`, which implements the WMMSE in [[1]](#ourpaper)  in Python 3.6.13.
- `utility_functions.py`, which includes subroutines that will be needed in `run_unrolled_PGD_WSRMax.py` and `wmmse_algorithm.py`. It also provides the implementation in Python 3 of the zero forcing (ZF) solution and regularized zero-forcing (RZF) solution to (5).

Note that the training time can vary from an hour to many hours, depending on the parameter settings, e.g., the number of iterations and the number of PGD steps, and on the user hardware. 

## Computation environment
In order to run the code in this repository, the following software packages are required:
* `Python 3` (for reference we use Python 3.6.13), with the following packages:`numpy`, `matplotlib`, `copy`, `time`.
* `tensorflow` (version 1.x - for reference we use version 1.15.0).

## Reference

<a id='WMMSE_Shi'></a> [1] Q. Shi, M. Razaviyayn, Z. Luo and C. He, "An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel," in IEEE Transactions on Signal Processing, vol. 59, no. 9, pp. 4331-4340, Sept. 2011, doi: 10.1109/TSP.2011.2147784.

<a id='L2O_Chen'></a> [2] Chen T, Chen X, Chen W, et al. Learning to optimize: A primer and a benchmark[J]. The Journal of Machine Learning Research, 2022, 23(1): 8562-8620.

