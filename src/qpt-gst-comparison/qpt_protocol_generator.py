from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal


@dataclass
class QPTScheme:
    prep_label: tuple[str, str]
    meas_label: tuple[str, str]
    prep_ops: list[int]
    process_ops: list[int]
    meas_ops: list[int]
    full_ops: list[int]
    shots: int


# ----------------------------
# Gate index convention
# q0: sqrtX -> 0, sqrtY -> 1
# q1: sqrtX -> 2, sqrtY -> 3
# ----------------------------

def sx_idx(qubit: int) -> int:
    if qubit == 0:
        return 0
    if qubit == 1:
        return 2
    raise ValueError(f"Unsupported qubit index: {qubit}")


def sy_idx(qubit: int) -> int:
    if qubit == 0:
        return 1
    if qubit == 1:
        return 3
    raise ValueError(f"Unsupported qubit index: {qubit}")


# ----------------------------
# 1-qubit preparation library
# ----------------------------

def prep_ops_1q(label: str, qubit: int) -> list[int]:
    """
    Minimal 1-qubit linearly independent preparation set from |0>:

    Z+ : |0>
    X+ : sqrt(Y)|0>
    Y- : sqrt(X)|0>
    Z- : X|0> = sqrt(X)sqrt(X)|0>
    """
    if label == "Z+":
        return []
    if label == "X+":
        return [sy_idx(qubit)]
    if label == "Y-":
        return [sx_idx(qubit)]
    if label == "Z-":
        return [sx_idx(qubit), sx_idx(qubit)]

    raise ValueError(f"Unknown prep label: {label}")


def prep_labels_minimal_1q() -> list[str]:
    # порядок фиксируем явно
    return ["Z+", "X+", "Y-", "Z-"]


# ----------------------------
# 1-qubit measurement library
# ----------------------------

def meas_ops_1q(label: str, qubit: int) -> list[int]:
    """
    Pre-rotations before standard computational-basis readout.

    Z : no rotation
    X : R_y(-pi/2) = (sqrtY)^3
    Y : R_x(+pi/2) = sqrtX
    """
    if label == "Z":
        return []
    if label == "X":
        return [sy_idx(qubit), sy_idx(qubit), sy_idx(qubit)]
    if label == "Y":
        return [sx_idx(qubit)]

    raise ValueError(f"Unknown measurement label: {label}")


def meas_labels_cube_1q() -> list[str]:
    return ["Z", "X", "Y"]


# ----------------------------
# 2-qubit protocol generator
# ----------------------------

def build_prep_ops_2q(prep_label: tuple[str, str]) -> list[int]:
    """
    Порядок операций: сначала операции на q0, потом на q1.
    """
    q0_label, q1_label = prep_label
    ops = []
    ops.extend(prep_ops_1q(q0_label, 0))
    ops.extend(prep_ops_1q(q1_label, 1))
    return ops


def build_meas_ops_2q(meas_label: tuple[str, str]) -> list[int]:
    """
    Порядок операций: сначала операции на q0, потом на q1.
    """
    q0_label, q1_label = meas_label
    ops = []
    ops.extend(meas_ops_1q(q0_label, 0))
    ops.extend(meas_ops_1q(q1_label, 1))
    return ops


def generate_qpt_protocol_2q(
    shots: int,
    process_ops: list[int] | None = None,
) -> list[QPTScheme]:
    """
    Генерирует минимальный 2-qubit QPT protocol:
        16 input states x 9 measurement bases = 144 schemes

    Порядок:
        for prep in preps:
            for meas in meases:
                scheme = prep + process + meas
    """
    if process_ops is None:
        process_ops = []

    prep_labels_1q = prep_labels_minimal_1q()
    meas_labels_1q = meas_labels_cube_1q()

    prep_space = list(product(prep_labels_1q, repeat=2))   # 16
    meas_space = list(product(meas_labels_1q, repeat=2))   # 9

    schemes: list[QPTScheme] = []

    for prep_label in prep_space:
        prep_ops = build_prep_ops_2q(prep_label)

        for meas_label in meas_space:
            meas_ops = build_meas_ops_2q(meas_label)
            full_ops = prep_ops + list(process_ops) + meas_ops

            schemes.append(
                QPTScheme(
                    prep_label=prep_label,
                    meas_label=meas_label,
                    prep_ops=prep_ops,
                    process_ops=list(process_ops),
                    meas_ops=meas_ops,
                    full_ops=full_ops,
                    shots=shots,
                )
            )

    return schemes


# ----------------------------
# Convenience helpers
# ----------------------------

def extract_full_circuits(schemes: list[QPTScheme]) -> list[list[int]]:
    """
    Возвращает только списки индексов гейтов.
    """
    return [scheme.full_ops for scheme in schemes]


def protocol_summary(schemes: list[QPTScheme]) -> None:
    print(f"N_schemes = {len(schemes)}")
    if schemes:
        print(f"shots     = {schemes[0].shots}")
        print(f"first     = {schemes[0]}")