from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from .appearance import generate_remnant_appearance
from .backend import get_backend
from .config import PipelineConfig
from .export import save_npz_cache, write_manifest, try_write_vdb
from .render import render_sequence


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(cfg.render.backend)
    volume = generate_remnant_appearance(cfg.scene, backend=backend)
    volume.metadata["tilt_deg"] = cfg.scene.tilt_deg
    volume.metadata["backend_used"] = backend.name

    results: Dict[str, Any] = {"output_dir": str(output_dir), "backend_used": backend.name}
    if cfg.export.write_npz_cache:
        results["npz_cache"] = str(save_npz_cache(volume, output_dir))
    if cfg.export.write_manifest or cfg.export.ue_channel_manifest:
        results["manifest"] = str(write_manifest(volume, cfg, output_dir))
    if cfg.export.write_vdb_if_available:
        main_vdb, aux_vdb = try_write_vdb(volume, output_dir)
        if main_vdb is not None:
            results["main_vdb"] = str(main_vdb)
        if aux_vdb is not None:
            results["aux_vdb"] = str(aux_vdb)

    frame_paths, metrics = render_sequence(volume, cfg.render, output_dir=output_dir, seed=cfg.scene.seed)
    results["frame_count"] = len(frame_paths)
    results["frames"] = [str(p) for p in frame_paths]
    if (output_dir / "preview.gif").exists():
        results["gif"] = str(output_dir / "preview.gif")
    results["metrics"] = [m.__dict__ for m in metrics]

    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    results["summary"] = str(summary_path)
    return results
