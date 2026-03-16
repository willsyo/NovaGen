from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import json

import imageio.v2 as imageio
import numpy as np
from numba import njit
from scipy.ndimage import rotate, gaussian_filter, zoom

from .appearance import AppearanceVolume
from .config import RenderConfig


@dataclass
class RenderMetrics:
    frame_index: int
    mean_luma: float
    std_luma: float
    max_luma: float


@njit(cache=True)
def _integrate_emission_absorption_numba(emissive_rgb: np.ndarray, extinction: np.ndarray, ds: float) -> np.ndarray:
    depth, height, width, _ = emissive_rgb.shape
    out = np.zeros((height, width, 3), dtype=np.float32)
    trans = np.ones((height, width), dtype=np.float32)
    for k in range(depth):
        sigma = extinction[k]
        atten = np.exp(-sigma * ds)
        for c in range(3):
            out[:, :, c] += trans * emissive_rgb[k, :, :, c] * ds
        trans *= atten
    return out


def _rotate_scalar(vol: np.ndarray, az_deg: float, tilt_deg: float) -> np.ndarray:
    tmp = rotate(vol, angle=az_deg, axes=(0, 1), reshape=False, order=1, mode="constant", cval=0.0)
    return rotate(tmp, angle=tilt_deg, axes=(0, 2), reshape=False, order=1, mode="constant", cval=0.0)


def _rotate_rgb(vol: np.ndarray, az_deg: float, tilt_deg: float) -> np.ndarray:
    return np.stack([_rotate_scalar(vol[..., c], az_deg, tilt_deg) for c in range(vol.shape[-1])], axis=-1)


def _space_background(h: int, w: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg += np.array([0.002, 0.004, 0.008], dtype=np.float32)
    star_count = max(16, (h * w) // 3000)
    ys = rng.integers(0, h, size=star_count)
    xs = rng.integers(0, w, size=star_count)
    vals = rng.uniform(0.2, 0.9, size=star_count)
    for y, x, v in zip(ys, xs, vals):
        bg[y, x] = v
    return gaussian_filter(bg, sigma=(0.8, 0.8, 0.0))


def _postprocess(image: np.ndarray, cfg: RenderConfig, seed: int) -> np.ndarray:
    img = np.maximum(image, 0.0)
    if cfg.detail_strength > 0.0 and cfg.detail_sigma_px > 0.0:
        blurred = gaussian_filter(img, sigma=(cfg.detail_sigma_px, cfg.detail_sigma_px, 0.0))
        img = np.clip(img + cfg.detail_strength * (img - blurred), 0.0, None)
    if cfg.bloom_strength > 0.0 and cfg.bloom_sigma_px > 0.0:
        bloom = gaussian_filter(img, sigma=(cfg.bloom_sigma_px, cfg.bloom_sigma_px, 0.0))
        img = img + cfg.bloom_strength * bloom
    img = 1.0 - np.exp(-cfg.exposure * img)
    if cfg.background == "space":
        img = np.clip(img + _space_background(img.shape[0], img.shape[1], seed=seed), 0.0, 1.0)
    img = np.clip(img, 0.0, 1.0)
    return img ** (1.0 / max(cfg.gamma, 1e-6))


def render_frame(volume: AppearanceVolume, cfg: RenderConfig, az_deg: float, tilt_deg: float, out_size: Optional[int] = None, seed: int = 0) -> np.ndarray:
    emissive = _rotate_rgb(volume.emissive_rgb, az_deg=az_deg, tilt_deg=tilt_deg)
    extinction = _rotate_scalar(volume.extinction, az_deg=az_deg, tilt_deg=tilt_deg)
    ds = 2.0 / (volume.extinction.shape[0] * max(cfg.ray_step_scale, 1e-6))
    img = _integrate_emission_absorption_numba(emissive.astype(np.float32), extinction.astype(np.float32), np.float32(ds))
    if out_size is None:
        out_size = cfg.image_size
    if img.shape[0] != out_size:
        scale = out_size / img.shape[0]
        img = zoom(img, zoom=(scale, scale, 1.0), order=1)
    return _postprocess(img, cfg, seed=seed)


def _luma(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _get_mpi_rank_world() -> Tuple[int, int, object | None]:
    try:
        from mpi4py import MPI  # type: ignore
        comm = MPI.COMM_WORLD
        return int(comm.Get_rank()), int(comm.Get_size()), comm
    except Exception:
        return 0, 1, None


def render_sequence(volume: AppearanceVolume, cfg: RenderConfig, output_dir: str | Path, seed: int = 0) -> Tuple[List[Path], List[RenderMetrics]]:
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rank, world, comm = _get_mpi_rank_world()
    frame_indices = [i for i in range(cfg.frames) if i % world == rank]

    local_metrics: List[RenderMetrics] = []
    for i in frame_indices:
        az_deg = (i / max(cfg.frames, 1)) * cfg.orbit_degrees
        img = render_frame(volume, cfg, az_deg=az_deg, tilt_deg=volume.metadata.get("tilt_deg", 24.0), seed=seed + i)
        frame_path = frames_dir / f"frame_{i:04d}.png"
        imageio.imwrite(frame_path, (np.clip(img, 0, 1) * 255).astype(np.uint8))
        y = _luma(img)
        local_metrics.append(RenderMetrics(i, float(y.mean()), float(y.std()), float(y.max())))

    if comm is not None:
        comm.Barrier()
        gathered = comm.gather([m.__dict__ for m in local_metrics], root=0)
    else:
        gathered = [[m.__dict__ for m in local_metrics]]

    if rank == 0:
        all_metrics = []
        for chunk in gathered:
            all_metrics.extend(chunk)
        all_metrics = sorted(all_metrics, key=lambda x: x["frame_index"])
        all_paths = [frames_dir / f"frame_{i:04d}.png" for i in range(cfg.frames)]
        if all(p.exists() for p in all_paths):
            gif_path = output_dir / "preview.gif"
            images = [imageio.imread(p) for p in all_paths]
            duration_ms = int(round(1000 / max(cfg.frames, 1)))
            imageio.mimsave(gif_path, images, duration=duration_ms)
        with (output_dir / "render_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        return all_paths, [RenderMetrics(**m) for m in all_metrics]
    return [], []
