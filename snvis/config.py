from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict
import copy
import yaml


@dataclass
class SceneConfig:
    phase: str = "remnant_shell"
    grid_size: int = 96
    seed: int = 42
    physical_extent_ly: float = 12.0
    age_years: float = 340.0
    shell_radius: float = 0.62
    shell_thickness: float = 0.07
    asymmetry: float = 0.18
    filament_strength: float = 1.2
    knot_count: int = 64
    synchrotron_strength: float = 0.28
    dust_strength: float = 0.20
    observational_palette: str = "hubble_like"
    brightness_scale: float = 1.0
    tilt_deg: float = 24.0


@dataclass
class RenderConfig:
    image_size: int = 320
    frames: int = 12
    orbit_degrees: float = 360.0
    backend: str = "auto"
    ray_step_scale: float = 1.0
    exposure: float = 1.55
    gamma: float = 2.2
    bloom_sigma_px: float = 1.35
    bloom_strength: float = 0.10
    detail_sigma_px: float = 3.0
    detail_strength: float = 0.55
    background: str = "space"


@dataclass
class ExportConfig:
    output_dir: str = "outputs/remnant_quick"
    write_npz_cache: bool = True
    write_vdb_if_available: bool = True
    write_manifest: bool = True
    write_png_frames: bool = True
    write_gif: bool = True
    gif_fps: int = 12
    ue_channel_manifest: bool = True


@dataclass
class PipelineConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = PipelineConfig().to_dict()
    merged = _deep_update(copy.deepcopy(defaults), raw)
    return PipelineConfig(
        scene=SceneConfig(**merged.get("scene", {})),
        render=RenderConfig(**merged.get("render", {})),
        export=ExportConfig(**merged.get("export", {})),
    )
