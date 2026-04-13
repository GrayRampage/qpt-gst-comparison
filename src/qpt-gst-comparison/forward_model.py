from __future__ import annotations

import torch as t
import operations as ops


def probabilities(
    gate_set: t.Tensor,
    prep: t.Tensor,
    circuits: list[list[int]],
    meas: t.Tensor,
    dim: int,
) -> t.Tensor:
    """
    Returns:
        probs: (n_circuits, n_outcomes) real (float64)
    """
    eps = 1e-8
    device = gate_set.device

    gate_set = gate_set.to(t.cdouble)
    prep = prep.to(device=device, dtype=t.cdouble)
    meas = meas.to(device=device, dtype=t.cdouble)

    n_circuits = len(circuits[0])
    n_outcomes = meas.shape[0]
    out = t.empty((n_circuits, n_outcomes), device=device, dtype=t.float64)

    for i, circ in enumerate(circuits[0]):
        G = ops.generate_g(circ, gate_set)   # (d2, d2) complex
        transformed = G @ prep               # (d2,) complex
        p = t.real(meas @ transformed)       # (n_outcomes,) real
        p = t.clamp(p, eps, 1 - eps)
        p = p / p.sum().clamp_min(1e-15)     # нормализация на всякий случай
        out[i] = p

    return out


def normalize_shots(
    shots: int | list[int] | t.Tensor,
    n_circuits: int,
    device: t.device,
) -> t.Tensor:
    """
    Converts shots into tensor of shape (n_circuits,), dtype=int64.
    """
    if isinstance(shots, int):
        if shots <= 0:
            raise ValueError("shots must be positive")
        return t.full((n_circuits,), shots, dtype=t.int64, device=device)

    if isinstance(shots, list):
        if len(shots) != n_circuits:
            raise ValueError(
                f"shots length mismatch: got {len(shots)}, expected {n_circuits}"
            )
        shots_tensor = t.tensor(shots, dtype=t.int64, device=device)

    elif isinstance(shots, t.Tensor):
        if shots.ndim != 1:
            raise ValueError("shots tensor must be 1D")
        if shots.numel() != n_circuits:
            raise ValueError(
                f"shots length mismatch: got {shots.numel()}, expected {n_circuits}"
            )
        shots_tensor = shots.to(device=device, dtype=t.int64)

    else:
        raise TypeError("shots must be int, list[int], or torch.Tensor")

    if t.any(shots_tensor <= 0):
        raise ValueError("all shots values must be positive")

    return shots_tensor


def sample_counts(probs: t.Tensor, shots: int | list[int]) -> t.Tensor:
    if probs.ndim != 2:
        raise ValueError("probs must be 2D")

    probs = probs.to(dtype=t.float64)
    n_circuits, n_outcomes = probs.shape
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-15)

    if isinstance(shots, int):
        shots = [shots] * n_circuits

    if len(shots) != n_circuits:
        raise ValueError("len(shots) must match number of circuits")

    counts = t.zeros((n_circuits, n_outcomes), dtype=t.int64, device=probs.device)

    for i in range(n_circuits):
        n = int(shots[i])
        if n <= 0:
            raise ValueError(f"shots[{i}] must be positive")

        idx = t.multinomial(probs[i], n, replacement=True)
        counts[i].scatter_add_(0, idx, t.ones_like(idx, dtype=t.int64))

    return counts


def simulate_experiment(
    dim: int,
    gate_set: t.Tensor,
    prep: t.Tensor,
    meas: t.Tensor,
    circuits: list[list[int]],
    shots: int | list[int],
) -> tuple[t.Tensor, t.Tensor]:
    probs = probabilities(gate_set, prep, circuits, meas, dim)
    counts = sample_counts(probs, shots)
    return probs, counts