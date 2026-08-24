"""Compute the dissipative Yukawa-SYK Lyapunov exponent for ``p = 2``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import YSYK_dissipative_relaxation_rate_non_auxiliary as sd


# Use the same model parameters as the shared SD solver.
P = sd.P
J = sd.J
DELTA = sd.DELTA
GAMMA_VALUES = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
KAPPA_VALUES = 0.2 * np.arange(1, 30)

# The SD grid resolves the oscillator poles and the slow fermion decay.
SD_N = 2**17
SD_DT = 0.25

# A power-of-two crop makes the repeated kernel convolutions much faster.
KERNEL_POINTS = 2**18
KERNEL_TAPER_FRACTION = 0.05
REMOVE_FERMION_REGULATOR = True

# Numerical controls for the largest eigenvalue and the lambda root search.
POWER_MAX_ITERATIONS = 250
POWER_TOLERANCE = 1.0e-7
LAMBDA_TOLERANCE = 1.0e-7
INITIAL_LAMBDA_UPPER = 0.01
MAX_LAMBDA_UPPER = 0.02

TAIL_FIT_MIN = 4.0
TAIL_FIT_MAX = 5.8
OUTPUT_FILE = Path(__file__).with_name("Lyapunov_p_2.pdf")


@dataclass
class KernelData:
    """Time-domain lines and frequency grid used by the ladder kernel."""

    time: np.ndarray
    frequency: np.ndarray
    fermion_retarded: np.ndarray
    boson_retarded: np.ndarray
    boson_rung: np.ndarray
    fermion_rung: np.ndarray
    gamma: float


@dataclass
class LyapunovPoint:
    """A converged Lyapunov exponent at one dissipation strength."""

    kappa: float
    lyapunov: float


def require_finite(values: np.ndarray) -> np.ndarray:
    """Reject non-finite values before they enter a convolution."""

    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("The ladder kernel produced non-finite values.")
    return values


def kernel_frequency_grid(time: np.ndarray) -> np.ndarray:
    """Return the centered angular-frequency grid conjugate to ``time``."""

    dt = time[1] - time[0]
    return 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(len(time), d=dt)
    )


def centered_crop(values: np.ndarray, points: int) -> np.ndarray:
    """Take a centered crop containing exactly ``points`` entries."""

    if points > len(values) or points <= 0:
        raise ValueError("KERNEL_POINTS must lie between 1 and the SD grid size.")
    center = len(values) // 2
    start = center - points // 2
    return np.array(values[start : start + points], copy=True)


def tukey_window(time: np.ndarray, taper_fraction: float) -> np.ndarray:
    """Smoothly suppress the outer edge of the cropped time interval."""

    if taper_fraction <= 0.0:
        return np.ones_like(time)
    taper_fraction = float(np.clip(taper_fraction, 1.0e-6, 0.95))
    cutoff = float(np.max(np.abs(time)))
    inner = cutoff * (1.0 - taper_fraction)
    distance = np.abs(time)
    window = np.ones_like(time)
    edge = distance > inner
    phase = (distance[edge] - inner) / (cutoff - inner)
    window[edge] = 0.5 * (1.0 + np.cos(np.pi * phase))
    window[distance >= cutoff] = 0.0
    return window


def make_kernel_data(
    solution: sd.SDSolution,
    time: np.ndarray,
    frequency: np.ndarray,
    gamma: float,
) -> KernelData:
    """Convert a converged R/A/K solution into the ladder-kernel lines."""

    state = solution.state
    fermion_greater_w, fermion_lesser_w = sd.greater_and_lesser(
        state.fermion_retarded, state.fermion_keldysh
    )
    boson_greater_w, boson_lesser_w = sd.greater_and_lesser(
        state.boson_retarded, state.boson_keldysh
    )

    fermion_greater = sd.frequency_to_time(
        frequency, fermion_greater_w
    )
    fermion_lesser = sd.frequency_to_time(
        frequency, fermion_lesser_w
    )
    boson_greater = sd.frequency_to_time(frequency, boson_greater_w)
    boson_lesser = sd.frequency_to_time(frequency, boson_lesser_w)

    theta = np.heaviside(time, 0.5)
    fermion_retarded = theta * (fermion_greater - fermion_lesser)
    boson_retarded = theta * (boson_greater - boson_lesser)

    if REMOVE_FERMION_REGULATOR:
        regulator_correction = np.exp(
            sd.ETA * np.maximum(time, 0.0)
        )
        fermion_retarded *= regulator_correction

    time = centered_crop(time, KERNEL_POINTS)
    fermion_retarded = centered_crop(fermion_retarded, KERNEL_POINTS)
    boson_retarded = centered_crop(boson_retarded, KERNEL_POINTS)
    fermion_greater = centered_crop(fermion_greater, KERNEL_POINTS)
    boson_greater = centered_crop(boson_greater, KERNEL_POINTS)

    window = tukey_window(time, KERNEL_TAPER_FRACTION)
    fermion_retarded *= window
    boson_retarded *= window
    fermion_greater *= window
    boson_greater *= window

    return KernelData(
        time=time,
        frequency=kernel_frequency_grid(time),
        fermion_retarded=require_finite(fermion_retarded),
        boson_retarded=require_finite(boson_retarded),
        boson_rung=require_finite(
            boson_greater * fermion_greater ** (P - 2)
        ),
        fermion_rung=require_finite(fermion_greater ** (P - 1)),
        gamma=gamma,
    )


def retarded_pair_spectrum(
    data: KernelData,
    retarded_line: np.ndarray,
    lyapunov: float,
) -> np.ndarray:
    """Build the two retarded rails after inserting the growth ansatz."""

    causal_line = np.where(data.time >= 0.0, retarded_line, 0.0)
    damped_line = (
        np.exp(-0.5 * lyapunov * np.maximum(data.time, 0.0))
        * causal_line
    )
    damped_w = sd.time_to_frequency(data.time, damped_line)
    pair_w = damped_w * sd.values_at_negative_frequency(damped_w)
    return require_finite(pair_w)


def convolve_with_spectrum(
    data: KernelData,
    fixed_spectrum: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Convolve ``values`` with a line whose transform is precomputed."""

    values_w = sd.time_to_frequency(data.time, values)
    return require_finite(
        sd.frequency_to_time(
            data.frequency, require_finite(fixed_spectrum * values_w)
        )
    )


def apply_ladder_kernel(
    eigenfunction: np.ndarray,
    data: KernelData,
    fermion_pair_w: np.ndarray,
    boson_pair_w: np.ndarray,
) -> np.ndarray:
    """Apply the sum of the bosonic and fermionic ladder kernels."""

    bosonic_part = convolve_with_spectrum(
        data,
        fermion_pair_w,
        data.boson_rung * eigenfunction,
    )
    bosonic_part *= (
        -(1j ** (P - 1))
        * 2.0
        * DELTA
        * data.gamma
        * J**2
        * (P - 1)
    )

    inner = convolve_with_spectrum(
        data,
        boson_pair_w,
        data.fermion_rung * eigenfunction,
    )
    fermionic_part = convolve_with_spectrum(
        data,
        fermion_pair_w,
        data.fermion_rung * inner,
    )
    fermionic_part *= 4.0 * data.gamma * DELTA**2 * J**4

    return require_finite(bosonic_part + fermionic_part)


def initial_eigenfunction(time: np.ndarray) -> np.ndarray:
    """Return a localized, even seed for the power iteration."""

    scaled = np.abs(time / 5.0)
    values = 2.0 * np.exp(-scaled) / (1.0 + np.exp(-2.0 * scaled))
    return values / np.linalg.norm(values)


def leading_kernel_eigenvalue(
    data: KernelData,
    lyapunov: float,
    initial: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    """Find the largest kernel eigenvalue by normalized power iteration."""

    fermion_pair_w = retarded_pair_spectrum(
        data, data.fermion_retarded, lyapunov
    )
    boson_pair_w = retarded_pair_spectrum(
        data, data.boson_retarded, lyapunov
    )
    vector = (
        initial_eigenfunction(data.time)
        if initial is None
        else np.array(initial, dtype=complex, copy=True)
    )
    vector /= np.linalg.norm(vector)
    previous_eigenvalue: complex | None = None

    for iteration in range(1, POWER_MAX_ITERATIONS + 1):
        image = apply_ladder_kernel(
            vector, data, fermion_pair_w, boson_pair_w
        )
        norm = float(np.linalg.norm(image))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("Power iteration produced a zero or non-finite vector.")

        eigenvalue = np.vdot(vector, image) / np.vdot(vector, vector)
        next_vector = image / norm
        center_value = next_vector[len(next_vector) // 2]
        if abs(center_value) > 0.0:
            next_vector *= np.exp(-1j * np.angle(center_value))

        eigenvalue_error = (
            np.inf
            if previous_eigenvalue is None
            else abs(eigenvalue - previous_eigenvalue)
            / max(1.0, abs(eigenvalue))
        )
        vector_error = float(np.linalg.norm(next_vector - vector))
        vector = next_vector
        if (
            eigenvalue_error < POWER_TOLERANCE
            and vector_error < np.sqrt(POWER_TOLERANCE)
        ):
            imaginary_ratio = abs(eigenvalue.imag) / max(
                abs(eigenvalue.real), 1.0e-14
            )
            if imaginary_ratio > 1.0e-6:
                raise RuntimeError(
                    "The leading kernel eigenvalue is not real within "
                    f"tolerance (relative imaginary part {imaginary_ratio:.3e})."
                )
            return float(eigenvalue.real), vector, iteration
        previous_eigenvalue = eigenvalue

    raise RuntimeError(
        f"Power iteration did not converge at lambda={lyapunov:.8g}."
    )


def find_lyapunov_exponent(data: KernelData) -> tuple[float, int]:
    """Find the positive lambda for which the largest eigenvalue equals one."""

    evaluations = 0
    eigenvalue_low, warm_vector, _ = leading_kernel_eigenvalue(data, 0.0)
    evaluations += 1
    if eigenvalue_low <= 1.0:
        raise RuntimeError(
            f"The kernel eigenvalue at lambda=0 is {eigenvalue_low:.8g}; "
            "no positive Lyapunov crossing is bracketed."
        )

    lambda_low = 0.0
    lambda_high = INITIAL_LAMBDA_UPPER
    eigenvalue_high, warm_vector, _ = leading_kernel_eigenvalue(
        data, lambda_high, warm_vector
    )
    evaluations += 1
    while eigenvalue_high > 1.0 and lambda_high < MAX_LAMBDA_UPPER:
        lambda_high = min(2.0 * lambda_high, MAX_LAMBDA_UPPER)
        eigenvalue_high, warm_vector, _ = leading_kernel_eigenvalue(
            data, lambda_high, warm_vector
        )
        evaluations += 1
    if eigenvalue_high >= 1.0:
        raise RuntimeError(
            "The unit-eigenvalue crossing lies above MAX_LAMBDA_UPPER."
        )

    while lambda_high - lambda_low > LAMBDA_TOLERANCE:
        lambda_mid = 0.5 * (lambda_low + lambda_high)
        eigenvalue_mid, warm_vector, _ = leading_kernel_eigenvalue(
            data, lambda_mid, warm_vector
        )
        evaluations += 1
        if eigenvalue_mid > 1.0:
            lambda_low = lambda_mid
            eigenvalue_low = eigenvalue_mid
        else:
            lambda_high = lambda_mid
            eigenvalue_high = eigenvalue_mid

    lyapunov = lambda_low + (
        (1.0 - eigenvalue_low)
        * (lambda_high - lambda_low)
        / (eigenvalue_high - eigenvalue_low)
    )
    return float(lyapunov), evaluations


def scan_gamma(
    gamma: float,
    time: np.ndarray,
    frequency: np.ndarray,
) -> list[LyapunovPoint]:
    """Scan from large to small kappa using each SD state as the next seed."""

    points_by_index: dict[int, LyapunovPoint] = {}
    previous_state: sd.SDState | None = None

    for count, index in enumerate(
        range(len(KAPPA_VALUES) - 1, -1, -1), start=1
    ):
        kappa = float(KAPPA_VALUES[index])
        solution = sd.solve_sd_equations(
            time,
            frequency,
            gamma=gamma,
            kappa=kappa,
            initial_state=previous_state,
        )
        previous_state = solution.state
        data = make_kernel_data(solution, time, frequency, gamma)
        lyapunov, evaluations = find_lyapunov_exponent(data)
        points_by_index[index] = LyapunovPoint(
            kappa=kappa,
            lyapunov=lyapunov,
        )
        print(
            f"gamma={gamma:4.2f}, kappa={kappa:3.1f}: "
            f"lambda={lyapunov:.8f} "
            f"(SD {solution.iterations} iterations, "
            f"residual={solution.residual:.2e}; "
            f"kernel {evaluations} evaluations; {count}/{len(KAPPA_VALUES)})"
        )

    return [points_by_index[index] for index in range(len(KAPPA_VALUES))]


def configure_plot_style() -> None:
    """Set a compact publication-style Matplotlib theme."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 18,
            "axes.linewidth": 1.15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )


def fit_tail_coefficient(
    results_by_gamma: dict[float, list[LyapunovPoint]],
) -> float:
    """Fit each curve to ``A Delta^2 J^2 / kappa^3`` and average A."""

    lyapunov_unit = J**2 / DELTA
    coefficients = []

    for gamma in GAMMA_VALUES:
        points = results_by_gamma[float(gamma)]
        scaled_kappa = np.array([point.kappa for point in points]) / DELTA
        scaled_lyapunov = (
            np.array([point.lyapunov for point in points]) / lyapunov_unit
        )
        fit_mask = (
            (scaled_kappa >= TAIL_FIT_MIN - 1.0e-12)
            & (scaled_kappa <= TAIL_FIT_MAX + 1.0e-12)
        )
        inverse_cubic = scaled_kappa[fit_mask] ** -3
        coefficient = np.dot(
            inverse_cubic, scaled_lyapunov[fit_mask]
        ) / np.dot(inverse_cubic, inverse_cubic)
        coefficients.append(coefficient)

    return float(np.mean(coefficients))


def plot_results(results_by_gamma: dict[float, list[LyapunovPoint]]) -> None:
    """Plot the numerical Lyapunov curves and the inverse-cubic tail."""

    configure_plot_style()
    figure, axis = plt.subplots(figsize=(5.5, 5.2))
    gamma_order = GAMMA_VALUES[::-1]
    colors = plt.cm.viridis(
        np.linspace(0.12, 0.88, len(GAMMA_VALUES))
    )[::-1]
    lyapunov_unit = J**2 / DELTA
    tail_coefficient = fit_tail_coefficient(results_by_gamma)

    for gamma, color in zip(gamma_order, colors):
        points = results_by_gamma[float(gamma)]
        kappa = np.array([point.kappa for point in points])
        lyapunov = np.array([point.lyapunov for point in points])
        axis.plot(
            kappa / DELTA,
            lyapunov / lyapunov_unit,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1.35,
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=rf"$\gamma={gamma:g}$",
        )

    tail_kappa = np.linspace(1.2 * DELTA, 6.0 * DELTA, 400)
    axis.plot(
        tail_kappa / DELTA,
        tail_coefficient * (DELTA / tail_kappa) ** 3,
        color="red",
        linestyle="--",
        linewidth=1.35,
        label=r"$\sim\kappa^{-3}$",
    )

    axis.set_xlabel(r"$\kappa/\Delta$")
    axis.set_ylabel(r"$\lambda\;[J^2/\Delta]$")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.17, 6.5)
    axis.set_ylim(0.012, 3.0)
    axis.minorticks_on()
    axis.tick_params(axis="x", which="minor", labelbottom=False)
    axis.legend(fontsize=14, loc="lower left")
    figure.tight_layout()
    figure.savefig(OUTPUT_FILE)
    print(f"Saved figure to {OUTPUT_FILE}")
    print(f"Large-kappa fit coefficient: A = {tail_coefficient:.6g}")
    plt.show()


def main() -> None:
    """Solve every gamma line and create the p=2 Lyapunov figure."""

    time, frequency = sd.make_frequency_grid(SD_N, SD_DT)
    results_by_gamma = {
        float(gamma): scan_gamma(float(gamma), time, frequency)
        for gamma in GAMMA_VALUES
    }
    plot_results(results_by_gamma)


if __name__ == "__main__":
    main()
