from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch as t

import tomllib  # Python 3.11+


# ---------- sections ----------

@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int


@dataclass(frozen=True)
class SizesConfig:
    n_qubits: int
    d_hilbert_1q: int
    d_hilbert_2q: int


@dataclass(frozen=True)
class NoiseConfig:
    p_depol: float
    p_ampl_damp: float
    p_phase_damp: float
    d_theta: float
    d_phi: float
    d_delta: float


@dataclass(frozen=True)
class SpamConfig:
    prep_err: float
    meas_01_err: float
    meas_10_err: float


@dataclass(frozen=True)
class DTypeConfig:
    complex: str
    real: str

    @property
    def complex_torch(self) -> t.dtype:
        mapping = {
            "cdouble": t.cdouble,
            "cfloat": t.cfloat,
        }
        try:
            return mapping[self.complex]
        except KeyError as e:
            raise ValueError(f"Unsupported complex dtype: {self.complex}") from e

    @property
    def real_torch(self) -> t.dtype:
        mapping = {
            "double": t.double,
            "float64": t.float64,
            "float": t.float,
            "float32": t.float32,
        }
        try:
            return mapping[self.real]
        except KeyError as e:
            raise ValueError(f"Unsupported real dtype: {self.real}") from e


@dataclass(frozen=True)
class IOConfig:
    artifacts_root: str
    run_name: str

@dataclass(frozen=True)
class DataConfig:
    default_shots: int

@dataclass(frozen=True)
class Config:
    reproducibility: ReproducibilityConfig
    sizes: SizesConfig
    noise: NoiseConfig
    spam: SpamConfig
    dtype: DTypeConfig
    io: IOConfig
    data: DataConfig


# ---------- loader ----------

def load_config(path: str | Path = "../configs/backend_config.toml") -> Config:
    path = Path(path)

    with path.open("rb") as f:
        raw = tomllib.load(f)

    return Config(
        reproducibility=ReproducibilityConfig(**raw["reproducibility"]),
        sizes=SizesConfig(**raw["sizes"]),
        noise=NoiseConfig(**raw["noise"]),
        spam=SpamConfig(**raw["spam"]),
        dtype=DTypeConfig(**raw["dtype"]),
        io=IOConfig(**raw["io"]),
        data=DataConfig(**raw["data"])
    )