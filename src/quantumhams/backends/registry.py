from __future__ import annotations

from quantumhams.backends.auto_backend import AutoBackend
from quantumhams.backends.base import SimulationBackend
from quantumhams.backends.cudaq_backend import CudaqBackend, is_available
from quantumhams.backends.scipy_backend import ScipyBackend


def backend_status() -> dict[str, bool]:
    return {"scipy": True, "cudaq": is_available()}


def get_backend(name: str = "auto") -> SimulationBackend:
    normalized = name.lower()
    if normalized == "auto":
        return AutoBackend()
    if normalized == "scipy":
        return ScipyBackend()
    if normalized == "cudaq":
        return CudaqBackend()
    raise ValueError(f"Unknown backend {name!r}. Available choices: auto, scipy, cudaq.")
