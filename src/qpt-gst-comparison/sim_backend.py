from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import json

import torch

from backend_config import Config
import sequence_parsing as sp
import operations as ops
import forward_model as fm


# =========================
# Result objects
# =========================

@dataclass
class SimRunResult:
    """
    Результат одного запуска backend-а.
    """
    counts: torch.Tensor   # shape: (N_schemes, N_outcomes)
    probs: torch.Tensor    # shape: (N_schemes, N_outcomes)


# =========================
# Main backend
# =========================

class SimBackend:
    """
    Симуляционный backend.

    Жизненный цикл:
        1. backend = SimBackend(cfg)
        2. result = backend.run(circuits_json)
        3. counts = result.counts
           probs  = result.probs
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        #self.device = torch.device(cfg.device)

        if cfg.sizes.n_qubits == 1:
            self.model = ops.build_gate_set_1q(cfg)
        if cfg.sizes.n_qubits == 2:
            self.model = ops.build_gate_set_2q(cfg)

    # -------------------------
    # Public API
    # -------------------------

    def run(self, circuits_json: str | Path | dict[str, Any] | list[dict[str, Any]]) -> SimRunResult:
        """
        Запускает симуляцию по набору схем.

        Returns
        -------
        SimRunResult
            counts и probs для всех схем.
        """
        circuits = sp.load_circuits(circuits_json, default_shots=self.cfg.data.default_shots)
        dim = self.cfg.sizes.d_hilbert_1q if self.cfg.sizes.n_qubits == 1 else self.cfg.sizes.d_hilbert_2q
        [probs, counts] = fm.simulate_experiment(
            dim=self.cfg.sizes.n_qubits,
            gate_set=self.model,
            prep=ops.rho_prep(dim=2**(self.cfg.sizes.n_qubits), spam_cfg=self.cfg.spam),
            meas=ops.meas_proj(dim=2**(self.cfg.sizes.n_qubits), spam_cfg=self.cfg.spam),
            circuits=circuits,
            shots=self.cfg.data.default_shots,
        )

        self.data_probs = probs
        self.data_counts = counts

        return SimRunResult(counts=counts, probs=probs)

    def get_counts(self) -> torch.Tensor:
        """
        Возвращает counts последнего запуска.
        """
        if self.data_counts is None:
            raise RuntimeError("No simulation data available. Call run(...) first.")
        return self.data_counts

    def get_probs(self) -> torch.Tensor:
        """
        Возвращает probabilities последнего запуска.
        """
        if self.data_probs is None:
            raise RuntimeError("No simulation data available. Call run(...) first.")
        return self.data_probs

    def getCounts(self) -> torch.Tensor:
        return self.get_counts()

    def getProbs(self) -> torch.Tensor:
        return self.get_probs()

    def reset(self) -> None:
        self.data_counts = None
        self.data_probs = None