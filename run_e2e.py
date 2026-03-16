from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snvis.config import load_config
from snvis.pipeline import run_pipeline

if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "remnant_quick.yaml")
    result = run_pipeline(cfg)
    print("E2E completed")
    print(result)
