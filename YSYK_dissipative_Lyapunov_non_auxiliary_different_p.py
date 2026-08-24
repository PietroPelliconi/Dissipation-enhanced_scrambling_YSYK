"""Compute the dissipative Yukawa-SYK Lyapunov exponent for several p."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import YSYK_dissipative_Lyapunov_non_auxiliary as otoc
import YSYK_dissipative_relaxation_rate_non_auxiliary as sd


# Model parameters used for the multi-p Lyapunov figure.
P_VALUES = np.array([2, 4, 6, 8], dtype=int)
GAMMA = 1.0
DELTA = sd.DELTA
BASE_COUPLING = 0.05
COUPLING_BY_P = {
    int(p): BASE_COUPLING * 2.0 ** ((int(p) - 2) / 2)
    for p in P_VALUES
}
KAPPA_VALUES_BY_P = {
    2: np.geomspace(0.2, 6.0, 19),
    4: np.geomspace(0.2, 10.0, 22),
    6: np.geomspace(0.2, 10.0, 22),
    8: np.geomspace(0.2, 10.0, 22),
}

# Numerical parameters for the SD and ladder equations.
SD_N = 2**17
SD_DT = 0.25
SD_ETA = sd.ETA
SD_MIXING = sd.MIXING
SD_MAX_ITERATIONS = sd.MAX_ITERATIONS
SD_TOLERANCE = sd.TOLERANCE
ANDERSON_DEPTH = 5
ANDERSON_REGULARIZATION = 1.0e-8

KERNEL_POINTS = 2**16
KERNEL_TAPER_FRACTION = 0.05
REMOVE_FERMION_REGULATOR = True

POWER_MAX_ITERATIONS = 250
POWER_TOLERANCE = 1.0e-7
EIGENVALUE_IMAGINARY_TOLERANCE = 1.0e-5
LAMBDA_TOLERANCE = 1.0e-7
INITIAL_LAMBDA_UPPER = 0.01
MAX_LAMBDA_UPPER = 0.02

OUTPUT_FILE = Path(__file__).with_name("Lyapunov_p_2468.pdf")


@dataclass
class KernelData:
    """Time-domain lines and model parameters used by the ladder kernel."""

    time: np.ndarray
    frequency: np.ndarray
    fermion_retarded: np.ndarray
    boson_retarded: np.ndarray
    boson_rung: np.ndarray
    fermion_rung: np.ndarray
    interaction_order: int
    coupling: float


@dataclass
class LyapunovPoint:
    """A converged Lyapunov exponent at one value of kappa."""

    kappa: float
    lyapunov: float


# Schwinger-Dyson equations

def pack_state(state: sd.SDState) -> np.ndarray:
    """Flatten the four propagators into one vector for Anderson mixing."""

    return np.concatenate(
        [
            state.fermion_retarded,
            state.fermion_keldysh,
            state.boson_retarded,
            state.boson_keldysh,
        ]
    )


def unpack_state(values: np.ndarray, size: int) -> sd.SDState:
    """Restore an SD state from a flattened mixing vector."""

    return sd.SDState(
        values[0:size].copy(),
        values[size : 2 * size].copy(),
        values[2 * size : 3 * size].copy(),
        values[3 * size : 4 * size].copy(),
    )


def anderson_step(
    state: sd.SDState,
    updated: sd.SDState,
    state_history: list[np.ndarray],
    residual_history: list[np.ndarray],
) -> tuple[sd.SDState, np.ndarray, np.ndarray]:
    """Apply a damped Anderson step to the latest SD update."""

    values = pack_state(state)
    residual = pack_state(updated) - values
    if not state_history:
        mixed = values + SD_MIXING * residual
    else:
        states = state_history[-ANDERSON_DEPTH:] + [values]
        residuals = residual_history[-ANDERSON_DEPTH:] + [residual]
        columns = len(states) - 1
        delta_state = np.column_stack(
            [states[index + 1] - states[index] for index in range(columns)]
        )
        delta_residual = np.column_stack(
            [
                residuals[index + 1] - residuals[index]
                for index in range(columns)
            ]
        )
        gram = delta_residual.conj().T @ delta_residual
        gram += ANDERSON_REGULARIZATION * np.eye(columns)
        right_hand_side = delta_residual.conj().T @ residual
        try:
            coefficients = np.linalg.solve(gram, right_hand_side)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                gram, right_hand_side, rcond=None
            )[0]
        mixed = (
            values
            + SD_MIXING * residual
            - (delta_state + SD_MIXING * delta_residual) @ coefficients
        )

    next_state = unpack_state(mixed, len(state.fermion_retarded))
    return next_state, values, residual

def solve_sd_equations(
    time: np.ndarray,
    frequency: np.ndarray,
    *,
    interaction_order: int,
    coupling: float,
    kappa: float,
    initial_state: sd.SDState | None = None,
) -> sd.SDSolution:
    """Solve the SD equations for one interaction order and loss rate."""

    state = (
        sd.free_state(frequency, DELTA, kappa, SD_ETA)
        if initial_state is None
        else initial_state
    )
    state_history: list[np.ndarray] = []
    residual_history: list[np.ndarray] = []

    for iteration in range(1, SD_MAX_ITERATIONS + 1):
        updated, _ = sd.sd_update(
            state,
            time,
            frequency,
            gamma=GAMMA,
            kappa=kappa,
            coupling=coupling,
            delta=DELTA,
            interaction_order=interaction_order,
            eta=SD_ETA,
        )
        residual = sd.state_residual(updated, state)
        if interaction_order == 2:
            state = sd.mix_states(state, updated, SD_MIXING)
        else:
            state, old_values, update_residual = anderson_step(
                state, updated, state_history, residual_history
            )
            state_history.append(old_values)
            residual_history.append(update_residual)
            if len(state_history) > ANDERSON_DEPTH + 1:
                del state_history[0]
                del residual_history[0]
        if residual < SD_TOLERANCE:
            break
    else:
        raise RuntimeError(
            f"SD iteration did not converge for p={interaction_order}, "
            f"kappa={kappa:g}; final residual={residual:.3e}"
        )

    _, sigma_retarded = sd.sd_update(
        state,
        time,
        frequency,
        gamma=GAMMA,
        kappa=kappa,
        coupling=coupling,
        delta=DELTA,
        interaction_order=interaction_order,
        eta=SD_ETA,
    )
    return sd.SDSolution(state, sigma_retarded, iteration, residual)


# Ladder kernel

def make_kernel_data(
    solution: sd.SDSolution,
    time: np.ndarray,
    frequency: np.ndarray,
    interaction_order: int,
    coupling: float,
) -> KernelData:
    """Convert a converged SD solution into retarded rails and rungs."""

    state = solution.state
    fermion_greater_w, fermion_lesser_w = sd.greater_and_lesser(
        state.fermion_retarded, state.fermion_keldysh
    )
    boson_greater_w, boson_lesser_w = sd.greater_and_lesser(
        state.boson_retarded, state.boson_keldysh
    )

    fermion_greater = sd.frequency_to_time(frequency, fermion_greater_w)
    fermion_lesser = sd.frequency_to_time(frequency, fermion_lesser_w)
    boson_greater = sd.frequency_to_time(frequency, boson_greater_w)
    boson_lesser = sd.frequency_to_time(frequency, boson_lesser_w)

    theta = np.heaviside(time, 0.5)
    fermion_retarded = theta * (fermion_greater - fermion_lesser)
    boson_retarded = theta * (boson_greater - boson_lesser)

    if REMOVE_FERMION_REGULATOR:
        fermion_retarded *= np.exp(SD_ETA * np.maximum(time, 0.0))

    time = otoc.centered_crop(time, KERNEL_POINTS)
    fermion_retarded = otoc.centered_crop(
        fermion_retarded, KERNEL_POINTS
    )
    boson_retarded = otoc.centered_crop(boson_retarded, KERNEL_POINTS)
    fermion_greater = otoc.centered_crop(fermion_greater, KERNEL_POINTS)
    boson_greater = otoc.centered_crop(boson_greater, KERNEL_POINTS)

    window = otoc.tukey_window(time, KERNEL_TAPER_FRACTION)
    fermion_retarded *= window
    boson_retarded *= window
    fermion_greater *= window
    boson_greater *= window

    return KernelData(
        time=time,
        frequency=otoc.kernel_frequency_grid(time),
        fermion_retarded=otoc.require_finite(fermion_retarded),
        boson_retarded=otoc.require_finite(boson_retarded),
        boson_rung=otoc.require_finite(
            boson_greater * fermion_greater ** (interaction_order - 2)
        ),
        fermion_rung=otoc.require_finite(
            fermion_greater ** (interaction_order - 1)
        ),
        interaction_order=interaction_order,
        coupling=coupling,
    )


def apply_ladder_kernel(
    eigenfunction: np.ndarray,
    data: KernelData,
    fermion_pair_w: np.ndarray,
    boson_pair_w: np.ndarray,
) -> np.ndarray:
    """Apply the bosonic and fermionic parts of the ladder kernel."""

    p = data.interaction_order
    coupling = data.coupling

    bosonic_part = otoc.convolve_with_spectrum(
        data,
        fermion_pair_w,
        data.boson_rung * eigenfunction,
    )
    bosonic_part *= (
        -(1j ** (p - 1))
        * 2.0
        * DELTA
        * GAMMA
        * coupling**2
        * (p - 1)
    )

    inner = otoc.convolve_with_spectrum(
        data,
        boson_pair_w,
        data.fermion_rung * eigenfunction,
    )
    fermionic_part = otoc.convolve_with_spectrum(
        data,
        fermion_pair_w,
        data.fermion_rung * inner,
    )
    fermionic_part *= 4.0 * GAMMA * DELTA**2 * coupling**4

    return otoc.require_finite(bosonic_part + fermionic_part)


def leading_kernel_eigenvalue(
    data: KernelData,
    lyapunov: float,
    initial: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    """Find the largest kernel eigenvalue by normalized power iteration."""

    fermion_pair_w = otoc.retarded_pair_spectrum(
        data, data.fermion_retarded, lyapunov
    )
    boson_pair_w = otoc.retarded_pair_spectrum(
        data, data.boson_retarded, lyapunov
    )
    vector = (
        otoc.initial_eigenfunction(data.time)
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
            raise RuntimeError(
                "Power iteration produced a zero or non-finite vector."
            )

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
            if imaginary_ratio > EIGENVALUE_IMAGINARY_TOLERANCE:
                raise RuntimeError(
                    "The leading kernel eigenvalue is not real within "
                    f"tolerance (relative imaginary part "
                    f"{imaginary_ratio:.3e})."
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


# Parameter scans

def scan_interaction_order(
    interaction_order: int,
    time: np.ndarray,
    frequency: np.ndarray,
) -> list[LyapunovPoint]:
    """Scan kappa from large to small and reuse each converged SD state."""

    coupling = COUPLING_BY_P[interaction_order]
    kappa_values = KAPPA_VALUES_BY_P[interaction_order]
    points_by_index: dict[int, LyapunovPoint] = {}
    previous_state: sd.SDState | None = None

    for count, index in enumerate(
        range(len(kappa_values) - 1, -1, -1), start=1
    ):
        kappa = float(kappa_values[index])
        solution = solve_sd_equations(
            time,
            frequency,
            interaction_order=interaction_order,
            coupling=coupling,
            kappa=kappa,
            initial_state=previous_state,
        )
        previous_state = solution.state
        data = make_kernel_data(
            solution,
            time,
            frequency,
            interaction_order,
            coupling,
        )
        lyapunov, evaluations = find_lyapunov_exponent(data)
        points_by_index[index] = LyapunovPoint(kappa, lyapunov)
        print(
            f"p={interaction_order}, kappa={kappa:5.3f}: "
            f"lambda={lyapunov:.8f} "
            f"(SD {solution.iterations} iterations, "
            f"residual={solution.residual:.2e}; "
            f"kernel {evaluations} evaluations; "
            f"{count}/{len(kappa_values)})"
        )

    return [points_by_index[index] for index in range(len(kappa_values))]


def large_p_lyapunov(
    kappa: np.ndarray,
    interaction_order: int,
    coupling: float,
) -> np.ndarray:
    """Return the large-p Lyapunov estimate for the plotted parameters."""

    denominator = DELTA**2 + 0.25 * kappa**2
    effective_coupling = (
        np.sqrt(GAMMA)
        * DELTA
        * coupling**2
        / (2.0 ** (interaction_order - 2) * denominator)
    )
    purcell_rate = (
        interaction_order
        * GAMMA
        * coupling**2
        * kappa
        / (2.0 ** (interaction_order - 1) * denominator)
    )
    return (
        np.sqrt(
            (2.0 * interaction_order) ** 2 * effective_coupling**2
            + (2.0 * interaction_order - 1.0) ** 2 * purcell_rate**2
        )
        - 2.0 * np.sqrt(effective_coupling**2 + purcell_rate**2)
        - purcell_rate
    ) / interaction_order


# Figure

def plot_results(results_by_p: dict[int, list[LyapunovPoint]]) -> None:
    """Plot the numerical curves together with the large-p estimates."""

    otoc.configure_plot_style()
    figure, axis = plt.subplots(figsize=(5.5, 5.2))
    p_order = P_VALUES[::-1]
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(P_VALUES)))

    for index, (p, color) in enumerate(zip(p_order, colors)):
        interaction_order = int(p)
        coupling = COUPLING_BY_P[interaction_order]
        points = results_by_p[interaction_order]
        kappa = np.array([point.kappa for point in points])
        lyapunov = np.array([point.lyapunov for point in points])
        lyapunov_unit = (
            2.0 ** (2 - interaction_order) * coupling**2 / DELTA
        )

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
            label=rf"$p={interaction_order}$",
        )

        theory_kappa = np.geomspace(float(kappa[0]), float(kappa[-1]), 400)
        theory = large_p_lyapunov(
            theory_kappa, interaction_order, coupling
        )
        axis.plot(
            theory_kappa / DELTA,
            theory / lyapunov_unit,
            color="0.20",
            linestyle="--",
            linewidth=1.25,
            label=(
                r"Large-$p$"
                if index == len(P_VALUES) - 1
                else "_nolegend_"
            ),
            zorder=4,
        )

    axis.set_xlabel(r"$\kappa/\Delta$")
    axis.set_ylabel(r"$\lambda\;[2^{2-p}J^2/\Delta]$")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.17, 12.0)
    axis.set_ylim(0.008, 8.0)
    axis.minorticks_on()
    axis.tick_params(axis="x", which="minor", labelbottom=False)
    axis.legend(fontsize=14, loc="lower left")
    figure.tight_layout()
    figure.savefig(OUTPUT_FILE)
    print(f"Saved figure to {OUTPUT_FILE}")
    plt.show()


def main() -> None:
    """Solve every interaction order and create the multi-p figure."""

    time, frequency = sd.make_frequency_grid(SD_N, SD_DT)
    results_by_p = {
        int(p): scan_interaction_order(int(p), time, frequency)
        for p in P_VALUES
    }
    plot_results(results_by_p)


if __name__ == "__main__":
    main()
