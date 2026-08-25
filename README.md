# Dissipation-enhanced scrambling in the SYK model coupled to a lossy cavity

This repository contains the numerical Python code accompanying
[Dissipation-enhanced scrambling in the SYK model coupled to a lossy cavity](https://arxiv.org/abs/2608.19310).

The scripts solve the large - $N$ Schwinger-Dyson equations while retaining the
full frequency dependence of the bosonic propagators. They generate the
numerical relaxation-rate and Lyapunov-exponent curves shown in Figures 2 and
3 of the paper.

## Repository contents

| Script | Corresponding result | Output |
| --- | --- | --- |
| [`YSYK_dissipative_relaxation_rate_non_auxiliary.py`](./YSYK_dissipative_relaxation_rate_non_auxiliary.py) | Section III B and Figure 2 | `Relaxation_rate_p_2.pdf` |
| [`YSYK_dissipative_Lyapunov_non_auxiliary.py`](./YSYK_dissipative_Lyapunov_non_auxiliary.py) | Section IV C and Figure 3(a) | `Lyapunov_p_2.pdf` |
| [`YSYK_dissipative_Lyapunov_non_auxiliary_different_p.py`](./YSYK_dissipative_Lyapunov_non_auxiliary_different_p.py) | Sections IV B-IV C and Figure 3(b) | `Lyapunov_p_2468.pdf` |

No external data files are needed. Each script performs the complete numerical
calculation and creates its figure from scratch.

Keep all three Python files in the same directory. The two Lyapunov scripts
import functions from the other scripts.

## 1. Fermionic relaxation rate

### Relation to the paper

[`YSYK_dissipative_relaxation_rate_non_auxiliary.py`](./YSYK_dissipative_relaxation_rate_non_auxiliary.py)
implements the numerical calculation described in Section III B and generates
Figure 2. It solves the Schwinger-Dyson equations (20)-(29), uses the damped
fixed-point iteration in Eq. (68), and extracts the late-time relaxation rate
from Eq. (69),

$$
\Gamma=-\text{Im} \, \Sigma_R(\omega=0).
$$

### Code synopsis

The script:

1. Constructs centered real-time and frequency grids.
2. Initializes the free infinite-temperature real-time fermionic propagator and the
   lossy-boson propagator.
3. Iterates the retarded and Keldysh Schwinger-Dyson equations to a fixed point.
4. Scans the dissipation strength from large to small $\kappa$, reusing each
   converged solution as the starting point for the next value.
5. Computes $\Gamma$ for each $\gamma$ and compares the numerical curves with
   the large-$p$ prediction in Eq. (54).

The default calculation uses

- $p=2$, $J=0.05$, and $\Delta=1$;
- $\gamma=0.15, 0.5, 1, 1.5, 2, 2.5$;
- $\kappa/\Delta=0.2, 0.4, \ldots, 5.8$.

The numerical dots and the dashed large - $p$ curves are saved to
`Relaxation_rate_p_2.pdf`.

## 2. Lyapunov exponent for $p=2$

### Relation to the paper

[`YSYK_dissipative_Lyapunov_non_auxiliary.py`](./YSYK_dissipative_Lyapunov_non_auxiliary.py)
generates the numerical curves corresponding to Figure 3(a). It implements the
ladder kernels introduced in Eqs. (76)-(78) and solves the unit-eigenvalue
condition in Eqs. (79)-(82), following the numerical procedure described in
Section IV C. The large-dissipation behavior is discussed around Eqs. (92) and
(96).

### Code synopsis

For each pair $(\gamma,\kappa)$, the script first obtains the full
Schwinger-Dyson solution using the functions in the relaxation-rate script. It
then:

1. Builds the retarded rails and the bosonic and fermionic ladder rungs.
2. Applies the ladder kernel through FFT-based convolutions.
3. Finds its largest eigenvalue by power iteration.
4. Adjusts $\lambda$ until the largest kernel eigenvalue equals one.
5. Plots the resulting dissipative Lyapunov exponent.

The default calculation uses $p=2$, $J=0.05$, $\Delta=1$,
$\gamma=0.25, 0.5, 1, 2, 4$, and
$\kappa/\Delta=0.2, 0.4, \ldots, 5.8$.

The dashed large - $\kappa$ guide is fitted to

$$
\lambda=A\frac{\Delta^2J^2}{\kappa^3}.
$$

The code fits $A$ independently for every $\gamma$ curve over
$4.0\leq\kappa/\Delta\leq5.8$ and plots their average. The fitted value of $A$
is printed after the scan.

The result is saved to `Lyapunov_p_2.pdf`.

## 3. Lyapunov exponent for different interaction orders

### Relation to the paper

[`YSYK_dissipative_Lyapunov_non_auxiliary_different_p.py`](./YSYK_dissipative_Lyapunov_non_auxiliary_different_p.py)
generates the numerical curves corresponding to Figure 3(b). It compares the
finite - $p$ ladder-kernel calculation with the large - $p$ Lyapunov exponent in
Eq. (88), testing the dissipation-enhanced scrambling discussed in Sections IV
B and IV C.

### Code synopsis

The script repeats the full Schwinger-Dyson and ladder-kernel calculation for
$p=2,4,6,8$ at fixed $\gamma=1$. It uses damped fixed-point iteration for
$p=2$ and Anderson mixing for the higher interaction orders. The couplings are
chosen so that all curves share the scale

$$
2^{2-p}J_p^2=2.5\times10^{-3},
$$

with $\Delta=1$. The numerical curves are plotted together with the dashed
large - $p$ predictions. The $p=2$ scan covers $0.2\leq\kappa/\Delta\leq6$,
while the $p=4,6,8$ scans cover $0.2\leq\kappa/\Delta\leq10$.

The result is saved to `Lyapunov_p_2468.pdf`.

## Reference

If you use these codes, please cite:

> Pietro Pelliconi, Bastien Lapierre, and Shinsei Ryu, "Dissipation-enhanced
> scrambling in the SYK model coupled to a lossy cavity," arXiv:2608.19310
> (2026).

```bibtex
@article{Pelliconi2026DissipationEnhanced,
  title         = {Dissipation-enhanced scrambling in the {SYK} model coupled to a lossy cavity},
  author        = {Pelliconi, Pietro and Lapierre, Bastien and Ryu, Shinsei},
  year          = {2026},
  eprint        = {2608.19310},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
