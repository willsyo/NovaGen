from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class BackendInfo:
    name: str
    xp: object
    ndimage: Optional[object] = None
    cuda_available: bool = False


def get_backend(name: str = "auto") -> BackendInfo:
    requested = name.lower()
    if requested in {"auto", "cuda"}:
        try:
            import cupy as cp  # type: ignore
            import cupyx.scipy.ndimage as cndimage  # type: ignore
            return BackendInfo("cuda", cp, cndimage, cuda_available=True)
        except Exception:
            if requested == "cuda":
                warnings.warn("CUDA backend requested but CuPy/cupyx is unavailable. Falling back to CPU.")
    import numpy as np
    import scipy.ndimage as snd
    return BackendInfo("cpu", np, snd, cuda_available=False)
