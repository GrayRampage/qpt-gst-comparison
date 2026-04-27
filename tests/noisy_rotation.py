import numpy as np

# ============================================================
# Basic single-qubit operators
# ============================================================

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

# Orthonormal Pauli basis for chi-matrix
# E_m = {I, X, Y, Z}/sqrt(2)
PAULI_BASIS = [
    I2 / np.sqrt(2),
    X  / np.sqrt(2),
    Y  / np.sqrt(2),
    Z  / np.sqrt(2),
]


# ============================================================
# Helpers: vectorization / devectorization
# We use column-stacking vec convention:
# vec([[a,b],[c,d]]) = [a,c,b,d]^T
# ============================================================

def vec(A: np.ndarray) -> np.ndarray:
    return A.reshape(-1, order="F")

def unvec(v: np.ndarray, dim: int = 2) -> np.ndarray:
    return v.reshape((dim, dim), order="F")


# ============================================================
# Noisy unitary U(phi, delta)
# U = [[cos(delta/2), -i sin(delta/2) e^{-i phi}],
#      [-i sin(delta/2) e^{+i phi}, cos(delta/2)]]
# ============================================================

def U_phi_delta(phi: float, delta: float) -> np.ndarray:
    c = np.cos(delta / 2.0)
    s = np.sin(delta / 2.0)
    return np.array([
        [c, -1j * s * np.exp(-1j * phi)],
        [-1j * s * np.exp(+1j * phi), c]
    ], dtype=complex)


# ============================================================
# Moments for Gaussian variables:
# x ~ N(mean, sigma^2), then E[e^{ikx}] = exp(i k mean - 1/2 k^2 sigma^2)
# ============================================================

def gaussian_characteristic_moment(k: int | float, mean: float, sigma: float) -> complex:
    return np.exp(1j * k * mean - 0.5 * (k ** 2) * (sigma ** 2))


# ============================================================
# Analytic average channel parameters
#
# delta ~ N(delta_mean, delta_sigma^2)
# phi   ~ N(phi_mean,   phi_sigma^2)
# independent, resampled every gate use
#
# Define:
# mu_c = E[cos delta]
# mu_s = E[sin delta]
# eta  = E[e^{+i phi}]
# nu   = E[e^{-2i phi}]
#
# alpha = (1 + mu_c)/2
# beta  = (1 - mu_c)/2
# gamma = mu_s / 2
# ============================================================

def average_channel_parameters(
    phi_mean: float,
    phi_sigma: float,
    delta_mean: float,
    delta_sigma: float
) -> dict:
    mu_c = np.exp(-0.5 * delta_sigma**2) * np.cos(delta_mean)
    mu_s = np.exp(-0.5 * delta_sigma**2) * np.sin(delta_mean)

    eta = gaussian_characteristic_moment(+1, phi_mean, phi_sigma)
    nu  = gaussian_characteristic_moment(-2, phi_mean, phi_sigma)

    alpha = 0.5 * (1.0 + mu_c)
    beta  = 0.5 * (1.0 - mu_c)
    gamma = 0.5 * mu_s

    return {
        "mu_c": mu_c,
        "mu_s": mu_s,
        "eta": eta,
        "nu": nu,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


# ============================================================
# Analytic averaged channel acting on 2x2 density matrix rho
# ============================================================

def apply_average_channel_analytic(
    rho: np.ndarray,
    phi_mean: float,
    phi_sigma: float,
    delta_mean: float,
    delta_sigma: float
) -> np.ndarray:
    p = average_channel_parameters(phi_mean, phi_sigma, delta_mean, delta_sigma)

    alpha = p["alpha"]
    beta  = p["beta"]
    gamma = p["gamma"]
    eta   = p["eta"]
    nu    = p["nu"]

    rho00 = rho[0, 0]
    rho01 = rho[0, 1]
    rho10 = rho[1, 0]
    rho11 = rho[1, 1]

    out = np.array([
        [
            alpha * rho00 + beta * rho11 + 1j * gamma * (eta * rho01 - np.conjugate(eta) * rho10),
            alpha * rho01 + beta * nu * rho10 + 1j * gamma * np.conjugate(eta) * (rho00 - rho11),
        ],
        [
            alpha * rho10 + beta * np.conjugate(nu) * rho01 - 1j * gamma * eta * (rho00 - rho11),
            beta * rho00 + alpha * rho11 - 1j * gamma * (eta * rho01 - np.conjugate(eta) * rho10),
        ],
    ], dtype=complex)

    return out


# ============================================================
# Build superoperator S in vec convention:
# vec(E(rho)) = S vec(rho)
#
# We construct S by acting on basis matrices |i><j|.
# ============================================================

def analytic_superoperator(
    phi_mean: float,
    phi_sigma: float,
    delta_mean: float,
    delta_sigma: float
) -> np.ndarray:
    basis_ops = [
        np.array([[1, 0], [0, 0]], dtype=complex),  # |0><0|
        np.array([[0, 1], [0, 0]], dtype=complex),  # |0><1|
        np.array([[0, 0], [1, 0]], dtype=complex),  # |1><0|
        np.array([[0, 0], [0, 1]], dtype=complex),  # |1><1|
    ]

    cols = []
    for B in basis_ops:
        EB = apply_average_channel_analytic(B, phi_mean, phi_sigma, delta_mean, delta_sigma)
        cols.append(vec(EB))

    return np.column_stack(cols)


# ============================================================
# Choi matrix J(E)
#
# With column-stacking vec convention:
# J_{ik,jl} = E(|i><j|)_{kl}
#
# Equivalent here to reshuffling S.
# We build J directly from E(|i><j|).
# ============================================================

def choi_from_channel_action(
    phi_mean: float,
    phi_sigma: float,
    delta_mean: float,
    delta_sigma: float
) -> np.ndarray:
    e00 = np.array([[1, 0], [0, 0]], dtype=complex)
    e01 = np.array([[0, 1], [0, 0]], dtype=complex)
    e10 = np.array([[0, 0], [1, 0]], dtype=complex)
    e11 = np.array([[0, 0], [0, 1]], dtype=complex)

    E00 = apply_average_channel_analytic(e00, phi_mean, phi_sigma, delta_mean, delta_sigma)
    E01 = apply_average_channel_analytic(e01, phi_mean, phi_sigma, delta_mean, delta_sigma)
    E10 = apply_average_channel_analytic(e10, phi_mean, phi_sigma, delta_mean, delta_sigma)
    E11 = apply_average_channel_analytic(e11, phi_mean, phi_sigma, delta_mean, delta_sigma)

    # Block form:
    # J = [[E(|0><0|), E(|0><1|)],
    #      [E(|1><0|), E(|1><1|)]]
    J = np.block([
        [E00, E01],
        [E10, E11],
    ])
    return J


# ============================================================
# Chi matrix in orthonormal Pauli basis {I, X, Y, Z}/sqrt(2)
#
# If J = sum_{mn} chi_{mn} |E_m>><<E_n|,
# then chi = T^\dagger J T
# where T = [|E_0>>, |E_1>>, |E_2>>, |E_3>>]
# since basis is orthonormal under Hilbert-Schmidt inner product.
# ============================================================

def chi_matrix_from_choi(J: np.ndarray) -> np.ndarray:
    T = np.column_stack([vec(E) for E in PAULI_BASIS])  # 4x4
    # Because {E_m} is orthonormal, T is unitary
    chi = T.conj().T @ J @ T
    return chi


# ============================================================
# Monte Carlo average channel for validation
# ============================================================

def apply_average_channel_monte_carlo(
    rho: np.ndarray,
    phi_mean: float,
    phi_sigma: float,
    delta_mean: float,
    delta_sigma: float,
    nsamples: int = 100000,
    seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    phis = rng.normal(loc=phi_mean, scale=phi_sigma, size=nsamples)
    deltas = rng.normal(loc=delta_mean, scale=delta_sigma, size=nsamples)

    out = np.zeros((2, 2), dtype=complex)
    for phi, delta in zip(phis, deltas):
        U = U_phi_delta(phi, delta)
        out += U @ rho @ U.conj().T

    return out / nsamples


def monte_carlo_superoperator(
    phi_mean: float,
    phi_sigma: float,
    delta_mean: float,
    delta_sigma: float,
    nsamples: int = 100000,
    seed: int | None = None
) -> np.ndarray:
    basis_ops = [
        np.array([[1, 0], [0, 0]], dtype=complex),
        np.array([[0, 1], [0, 0]], dtype=complex),
        np.array([[0, 0], [1, 0]], dtype=complex),
        np.array([[0, 0], [0, 1]], dtype=complex),
    ]

    cols = []
    for i, B in enumerate(basis_ops):
        EB = apply_average_channel_monte_carlo(
            B, phi_mean, phi_sigma, delta_mean, delta_sigma,
            nsamples=nsamples, seed=None if seed is None else seed + i
        )
        cols.append(vec(EB))
    return np.column_stack(cols)


# ============================================================
# Diagnostics
# ============================================================

def partial_trace_out_choi(J: np.ndarray) -> np.ndarray:
    # For J_{ik,jl}, this computes sum_k J_{ik,jk}
    J4 = J.reshape(2, 2, 2, 2)
    return np.einsum("ikjk->ij", J4)

def is_trace_preserving_from_choi(J: np.ndarray, tol: float = 1e-10) -> bool:
    ptr_out = partial_trace_out_choi(J)
    return np.allclose(ptr_out, I2, atol=tol)


def choi_eigenvalues(J: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh((J + J.conj().T) / 2)
    return np.real_if_close(vals)


# ============================================================
# Example usage
# ============================================================

if True:
    # Example parameters
    phi_mean = 0.3
    phi_sigma = 0.2
    delta_mean = np.pi / 2
    delta_sigma = 0.15

    print("=== Parameters ===")
    p = average_channel_parameters(phi_mean, phi_sigma, delta_mean, delta_sigma)
    for k, v in p.items():
        print(f"{k:>6s} = {v}")

    print("\n=== Analytic superoperator S ===")
    S = analytic_superoperator(phi_mean, phi_sigma, delta_mean, delta_sigma)
    print(S)

    print("\n=== Choi matrix J ===")
    J = choi_from_channel_action(phi_mean, phi_sigma, delta_mean, delta_sigma)
    print(J)

    print("\n=== chi matrix in normalized Pauli basis {I,X,Y,Z}/sqrt(2) ===")
    chi = chi_matrix_from_choi(J)
    print(chi)

    print("\n=== Checks ===")
    # test
    print("Partial trace over output:")
    print(partial_trace_out_choi(J))
    print("Trace preserving:", is_trace_preserving_from_choi(J))

    # Compare analytic vs Monte Carlo on a random state
    rng = np.random.default_rng(123)
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    rho_an = apply_average_channel_analytic(rho, phi_mean, phi_sigma, delta_mean, delta_sigma)
    rho_mc = apply_average_channel_monte_carlo(
        rho, phi_mean, phi_sigma, delta_mean, delta_sigma,
        nsamples=50000, seed=42
    )

    print("\n=== Test on a random pure state ===")
    print("rho_in =")
    print(rho)
    print("\nAnalytic E(rho) =")
    print(rho_an)
    print("\nMonte Carlo E(rho) =")
    print(rho_mc)
    print("\n||analytic - MC||_F =", np.linalg.norm(rho_an - rho_mc))