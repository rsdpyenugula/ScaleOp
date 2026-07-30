"""Shared utilities: determinism, device selection, and experiment-run setup.

`make_run` (added at M2, the first real experiment) gives every experiment the
same spine: seed everything from the config, pick a device, and create a
timestamped output dir with the resolved config dumped next to the results.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

RESULTS_DIR = Path(__file__).resolve().parent.parent / "project_docs" / "results"


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed torch / numpy / random and (optionally) request deterministic kernels.

    Determinism on non-CUDA backends (MPS/CPU) is best-effort: some ops lack a
    deterministic kernel. `warn_only` surfaces those as warnings to record in the
    milestone report rather than crashing or silently disabling determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """Pick a compute device. Order: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict[str, Any]:
    """Human-readable device/memory stats for milestone reports."""
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["name"] = props.name
        info["total_memory_gb"] = round(props.total_memory / 1e9, 2)
        info["allocated_gb"] = round(torch.cuda.memory_allocated(device) / 1e9, 3)
    else:
        info["name"] = "Apple MPS (unified memory)" if device.type == "mps" else "CPU"
    return info


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment's YAML config into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class Run:
    """One experiment run: its config + a timestamped output dir + device.

    Use `run.path("figure.png")` to get an output path (parents created), and
    `run.save_json("metrics.json", obj)` to write results next to the config.
    """

    name: str
    config: dict[str, Any]
    out_dir: Path
    device: torch.device

    def path(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return p


def make_run(name: str, config: dict[str, Any]) -> Run:
    """Seed everything, pick a device, and create project_docs/results/<name>/<ts>/.

    Dumps the resolved config next to the outputs so every result is reproducible.
    """
    set_all_seeds(int(config.get("seed", 0)), deterministic=config.get("deterministic", True))
    device = select_device()
    out_dir = RESULTS_DIR / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump({**config, "_device": str(device)}, f)
    return Run(name, config, out_dir, device)
