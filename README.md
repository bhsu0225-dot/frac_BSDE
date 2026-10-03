# Stochastic approximation methods for high-dimensional semilinear parabolic equations

This repository contains MATLAB implementations accompanying the paper *Stochastic approximation method for semilinear parabolic equations driven by killed Lévy processes in high dimensions*, by Jingchao Li, Changtao Sheng, Bihao Su and Zhi Zhou.

The programs approximate semilinear parabolic equations using forward particle simulation and a backward implicit recursion. Conditional expectations are computed in a two-dimensional feature space by binning particles, smoothing target sums and particle counts, and taking their ratio. The examples cover a fractional problem on the unit ball, a nonhomogeneous fractional problem on a perforated domain, and a highly oscillatory Brownian problem on a cube.

## Requirements

- MATLAB. The files use local functions and, in some places, implicit array expansion.
- Parallel Computing Toolbox for the parallel runs. In particular, `AC_HFO.m` uses `parfor` and `parallel.pool.DataQueue` directly.

The numerical routines are included as local functions in each file. No external datasets, pretrained models, or third-party numerical packages are required. The confluent hypergeometric function used in the nonhomogeneous example is evaluated by a numerical series routine included in the code.

## Files and numerical examples

Each file can be run independently. Helper functions with the same names appear in several files, but their implementations are local to the corresponding experiment.

### Example 1: homogeneous exterior data on the unit ball

The prescribed exact solution is

$$
u(t,x)=\frac{\cos t}{1+10t^2}\bigl(1-\lVert x\rVert^2\bigr)_+^{\alpha/2},
$$

with zero exterior data and nonlinearity $\sin u+\exp(\cos u)$.

- **`frac_IBSDE_ball_homog.m`** computes the numerical solution along the line $x=a(1,\ldots,1)$ and compares it with the exact solution for the particle numbers specified in `M_list`. It also reports errors, exit rates, scalar-solver residuals, and elapsed time.
- **`frac_IBSDE_ball_semilinear_err_t.m`** studies the error as the time step is refined. `N_list` specifies the time discretizations and `M_list` contains the two particle numbers used for the comparison curves. The plot includes an $\mathcal O(\Delta t)$ reference line.
- **`frac_IBSDE_ball_semilinear_err_M.m`** studies the error as the particle number increases. `M_list` specifies the particle numbers and `N_list` contains the two time discretizations used for the comparison curves. The plot includes an $\mathcal O(M^{-1/2})$ reference line.

These three programs use

$$
\Phi(x)=\left(\frac{1}{\sqrt d}\sum_{j=1}^d x_j,\ \sqrt{\lVert x\rVert^2-\left(\frac{1}{\sqrt d}\sum_{j=1}^d x_j\right)^2}\right).
$$

### Example 2: nonhomogeneous exterior data on a perforated domain

The domain is an ellipsoid with an interior ball removed. The prescribed exact solution is

$$
u(t,x)=\exp^{\cos t}x_{k_1}x_{k_2}\exp(-\lVert x\rVert^2),\qquad k_1\ne k_2,
$$

and the exterior data are given by the same expression. The nonlinearity is $\eta(\exp(\beta u)-u)$.

- **`frac_IBSDE_complex_nonhomog.m`** compares numerical and exact solution profiles along $x=a(e_{k_1}+e_{k_2})$ for the particle numbers in `M_list`. The default terminal time is `T = 50`.
- **`frac_IBSDE_complex_nonhomog_err_t.m`** computes time-step convergence curves for this problem, using two particle numbers and an $\mathcal O(\Delta t)$ reference line. Errors, observed orders, solver residuals, and timings are saved during the computation.

Both programs use the feature map $\Phi(x)=(x_{k_1}x_{k_2},\lVert x\rVert^2)$.

### Example 3: a highly oscillatory solution on a cube

- **`AC_HFO.m`** solves a Brownian problem with generator $\Delta$ on the cube $(-L,L)^d$. It compares the numerical solution with

  $$
  u(t,x)=\cos(\lambda_t t)\prod_{j=1}^d\sin(k\pi x_j)
  $$

  along diagonal query points. The frequencies are set in `k_list`, and the feature map is $\Phi(x)=(x_1x_2,\lVert x\rVert^2)$. This file is a script with local functions. Its default values `L = 1` and integer frequencies are consistent with the zero boundary data.

## Running the programs

Download the repository and set the MATLAB current folder to the directory containing the six `.m` files. Open the desired file and edit its parameter block before running it. No other experiment needs to be run first.

For a solution comparison on the unit ball, enter:

```matlab
frac_IBSDE_ball_homog
```

The other experiments are launched by their file names:

```matlab
% Time-step convergence on the unit ball
frac_IBSDE_ball_semilinear_err_t

% Particle-number convergence on the unit ball
frac_IBSDE_ball_semilinear_err_M

% Solution comparison on the perforated domain
frac_IBSDE_complex_nonhomog

% Time-step convergence on the perforated domain
frac_IBSDE_complex_nonhomog_err_t

% Highly oscillatory solution on the cube
AC_HFO
```


### Parallel execution

The three convergence programs and `frac_IBSDE_ball_homog.m` attempt to start a local pool when `usePar = true`. Set `numWorkers` near the top of the file to suit the available hardware. These programs include a serial fallback if pool startup is unavailable.

`frac_IBSDE_complex_nonhomog.m` uses an existing pool but does not create one. Start a pool before running it if parallel execution is desired:

```matlab
if isempty(gcp('nocreate'))
    parpool('local', 4);
end
frac_IBSDE_complex_nonhomog
```

Replace `4` with a suitable worker count. Without an existing pool, this program selects its serial loop. Set `usePar = false` when running it without Parallel Computing Toolbox. `AC_HFO.m` requires the toolbox as written and can also be run after starting a pool explicitly.

Parallelism is across query points. Each worker carries out its own particle simulation and backward recursion, so increasing the worker count also increases memory use.

## Parameters and implementation details

The main settings are collected near the beginning of each file:

- `d`: dimension of the physical space.
- `alpha` or `s`: fractional order, with `alpha = 2*s`. For `frac_IBSDE_complex_nonhomog.m`, change `s` because `alpha` is derived from it. The five fractional-example programs include a Brownian branch for `alpha = 2`. `AC_HFO.m` is Brownian only.
- `T`, `t0`, `N`, and `N_list`: time interval and time discretization. The time step is `(T-t0)/N`.
- `M` and `M_list`: number of particles simulated for each query point.
- `K`: number of spatial query points, not the number of feature-grid cells.
- `NBIN1`, `NBIN2`: numbers of bins along the two feature coordinates.
- `SMOOTHIT`: number of smoothing sweeps applied to the target sums and counts.
- `SOLVE_MAXIT`, `SOLVE_TOL`, or `NEWTON_IT`: scalar-solver settings, where exposed in the parameter block.
- `nRepeats`: number of repeated runs in the convergence programs.
- `reference_height`: vertical placement of the reference line. It does not change any computed errors.

All six implementations use two features. The `kappa = 2` variable in some files documents this choice. Changing that variable alone does not implement a higher-dimensional feature regression.

Feature-grid ranges are determined from the current particle features, with a small padding. The separable smoothing stencil is `[1; 2; 1]/4`. The grid is in feature space, not in the original physical space.


## Output files

The following describes the current save settings. Output folders are created beside the corresponding `.m` files.

| Program | Output folder | Saved formats |
| --- | --- | --- |
| `frac_IBSDE_ball_homog.m` | `output_ball_endpoint` | `.fig`, `.eps`, `.png`, `.mat` when `save_results = true` |
| `frac_IBSDE_ball_semilinear_err_t.m` | `output_ball_err_t` | `.fig`, `.eps`, `.pdf`, `.mat` |
| `frac_IBSDE_ball_semilinear_err_M.m` | `output_ball_err_M` | `.fig`, `.eps`, `.pdf`, `.mat` |
| `frac_IBSDE_complex_nonhomog_err_t.m` | `output_complex_err_t` | `.fig`, `.eps`, `.pdf`, `.mat` |
| `frac_IBSDE_complex_nonhomog.m` | No automatic export enabled | Figure displayed. Uncomment the save block to export `.fig`, `.eps`, and `.mat` files. |
| `AC_HFO.m` | No automatic export enabled | Figure displayed. Uncomment the save block to export `.fig` and `.eps` files. |

The convergence programs save a `.mat` checkpoint after each completed `(N,M)` pair. The files include computed errors, observed orders, numerical and exact values, timings, and parameter settings.
