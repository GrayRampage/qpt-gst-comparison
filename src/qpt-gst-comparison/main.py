import torch as t
import numpy as np
from pathlib import Path

from backend_config import load_config
from sim_backend import SimBackend
import qpt_protocol_generator as qpt_gen


cfg_backend = load_config("./configs/backend_config.toml")

backend = SimBackend(cfg_backend)

result = backend.run("./src/qpt-gst-comparison/seq_1q.json")

# ЗАФИКСИРОВАТЬ ЗЕРНО!!!
#print(result.counts)

schemes = qpt_gen.generate_qpt_protocol_2q(shots=1000, process_ops=[4])

circuits = qpt_gen.extract_full_circuits(schemes)

result_2q = backend.run(circuits)

circuits = [
    # prep = []
    [],
    [1,1,1],
    [0],

    # prep = [0]
    [0],
    [0, 1,1,1],
    [0, 0],

    # prep = [1]
    [1],
    [1, 1,1,1],
    [1, 0],

    # prep = [0,0]
    [0,0],
    [0,0, 1,1,1],
    [0,0, 0],
]

result = backend.run(circuits)

print(result.probs)