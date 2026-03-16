from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snvis.config import load_config
from snvis.pipeline import run_pipeline


def test_quick_e2e(tmp_path: Path):
    cfg = load_config(ROOT / "configs" / "remnant_quick.yaml")
    cfg.scene.grid_size = 40
    cfg.render.image_size = 96
    cfg.render.frames = 4
    cfg.export.output_dir = str(tmp_path / "out")
    cfg.export.write_vdb_if_available = False

    result = run_pipeline(cfg)
    out = Path(result["output_dir"])
    assert out.exists()
    assert Path(result["npz_cache"]).exists()
    assert Path(result["manifest"]).exists()
    assert Path(result["summary"]).exists()
    assert Path(result["gif"]).exists()
    assert len(result["frames"]) == 4
    for frame in result["frames"]:
        assert Path(frame).exists()

    metrics = result["metrics"]
    assert len(metrics) == 4
    mean_lumas = [m["mean_luma"] for m in metrics]
    std_lumas = [m["std_luma"] for m in metrics]
    max_lumas = [m["max_luma"] for m in metrics]
    assert min(mean_lumas) > 0.005
    assert max(std_lumas) > 0.02
    assert max(max_lumas) > 0.25

    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["backend_used"] in {"cpu", "cuda"}
