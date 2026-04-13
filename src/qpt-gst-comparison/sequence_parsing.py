from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def gate_to_symbol(gate: dict[str, Any]) -> int:
    gate_type = gate.get("type")

    if gate_type == "XX":
        if gate.get("qubits") == [0, 1] and gate.get("angle") == 0.25:
            return 4

    if gate_type == "Rphi":
        qubit = gate.get("qubit")
        axis = gate.get("axis")
        angle = gate.get("angle")

        if angle != 0.5:
            raise ValueError(f"Unsupported Rphi angle: {angle}")

        if qubit == 0 and axis == 0:
            return 0
        if qubit == 0 and axis == 0.5:
            return 1
        if qubit == 1 and axis == 0:
            return 2
        if qubit == 1 and axis == 0.5:
            return 3

    raise ValueError(f"Unknown gate: {gate}")


def load_circuits(
    source: str | Path | list[dict[str, Any]] | dict[str, Any],
    default_shots: int,
) -> tuple[list[list[int]], list[int]]:
    if isinstance(source, Path):
        with source.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(source, str):
        path = Path(source)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(source)
    else:
        data = source

    # Формат A:
    # [
    #   {"repetitions": 5000, "sequence": [gate_dict, ...]},
    #   ...
    # ]
    if isinstance(data, list):
        circuits = []
        shots = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Circuit #{i} must be a dict")

            raw_seq = item.get("sequence")
            reps = item.get("repetitions")

            if raw_seq is None:
                raise ValueError(f"Circuit #{i} missing 'sequence'")
            if reps is None:
                raise ValueError(f"Circuit #{i} missing 'repetitions'")
            if not isinstance(raw_seq, list):
                raise ValueError(f"Circuit #{i} sequence must be a list")

            seq = [gate_to_symbol(g) for g in raw_seq]

            circuits.append(seq)
            shots.append(int(reps))

        return circuits, shots

    # Формат B:
    # {"sequences": [[0,1,2], [4,0], ...]}
    if isinstance(data, dict) and "sequences" in data:
        raw_sequences = data["sequences"]

        if not isinstance(raw_sequences, list):
            raise ValueError("'sequences' must be a list")

        circuits = []
        for i, seq in enumerate(raw_sequences):
            if not isinstance(seq, list):
                raise ValueError(f"Sequence #{i} must be a list")

            parsed = []
            for j, x in enumerate(seq):
                if x not in (0, 1, 2, 3, 4):
                    raise ValueError(
                        f"Sequence #{i}, position #{j}: expected 0..4, got {x}"
                    )
                parsed.append(int(x))

            circuits.append(parsed)

        shots = [default_shots] * len(circuits)
        return circuits, shots

    raise ValueError("Unsupported circuits format")