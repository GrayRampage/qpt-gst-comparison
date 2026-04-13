import torch as t
import numpy as np
from pathlib import Path

from backend_config import load_config
from sim_backend import SimBackend


cfg_backend = load_config("./configs/backend_config.toml")

backend = SimBackend(cfg_backend)

result = backend.run("./src/qpt-gst-comparison/seq_1q.json")

# ЗАФИКСИРОВАТЬ ЗЕРНО!!!
print(result.counts)

print("Hello, quantum world!")