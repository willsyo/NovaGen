from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import json

import numpy as np

from .appearance import AppearanceVolume
from .config import PipelineConfig


def save_npz_cache(volume: AppearanceVolume, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "appearance_cache.npz"
    np.savez_compressed(
        path,
        emissive_rgb=volume.emissive_rgb.astype(np.float32),
        extinction=volume.extinction.astype(np.float32),
        albedo=volume.albedo.astype(np.float32),
        shock=volume.shock.astype(np.float32),
        filament=volume.filament.astype(np.float32),
        dust=volume.dust.astype(np.float32),
    )
    return path


def write_manifest(volume: AppearanceVolume, cfg: PipelineConfig, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **volume.metadata,
        "render": cfg.render.__dict__,
        "export": {"cache_format": "npz", "vdb_optional": True},
        "ue_integration": {
            "target_import": "Sparse Volume Textures",
            "target_actor": "Heterogeneous Volume",
            "main_channels": {
                "AttributesA.R": "emissive_r",
                "AttributesA.G": "emissive_g",
                "AttributesA.B": "emissive_b",
                "AttributesA.A": "extinction",
            },
            "aux_channels": {
                "AttributesB.R": "albedo",
                "AttributesB.G": "shock",
                "AttributesB.B": "filament",
                "AttributesB.A": "dust",
            },
        },
    }
    path = output_dir / "appearance_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def _import_openvdb_module():
    try:
        import openvdb as vdb  # type: ignore
        return vdb
    except Exception:
        try:
            import pyopenvdb as vdb  # type: ignore
            return vdb
        except Exception:
            return None


def _float_grid_from_array(vdb, name: str, arr: np.ndarray, voxel_size: float = 1.0):
    grid = vdb.FloatGrid()
    grid.name = name
    if hasattr(vdb, "createLinearTransform"):
        grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)
    if hasattr(grid, "copyFromArray"):
        grid.copyFromArray(arr.astype(np.float32))
    else:
        it = np.nditer(arr, flags=["multi_index"])
        for x in it:
            if float(x) != 0.0:
                grid.setValueOn(it.multi_index, float(x))
    if hasattr(vdb, "GridClass"):
        try:
            grid.gridClass = vdb.GridClass.FOG_VOLUME
        except Exception:
            pass
    return grid


def try_write_vdb(volume: AppearanceVolume, output_dir: str | Path) -> Tuple[Optional[Path], Optional[Path]]:
    vdb = _import_openvdb_module()
    if vdb is None:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "appearance_main.vdb"
    aux_path = output_dir / "appearance_aux.vdb"

    main_grids = [
        _float_grid_from_array(vdb, "emissive_r", volume.emissive_rgb[..., 0]),
        _float_grid_from_array(vdb, "emissive_g", volume.emissive_rgb[..., 1]),
        _float_grid_from_array(vdb, "emissive_b", volume.emissive_rgb[..., 2]),
        _float_grid_from_array(vdb, "extinction", volume.extinction),
    ]
    aux_grids = [
        _float_grid_from_array(vdb, "albedo", volume.albedo),
        _float_grid_from_array(vdb, "shock", volume.shock),
        _float_grid_from_array(vdb, "filament", volume.filament),
        _float_grid_from_array(vdb, "dust", volume.dust),
    ]
    vdb.write(str(main_path), grids=main_grids, metadata={"schema": "SNV-APPEARANCE-1.0"})
    vdb.write(str(aux_path), grids=aux_grids, metadata={"schema": "SNV-APPEARANCE-1.0"})
    return main_path, aux_path
