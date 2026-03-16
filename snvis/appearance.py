from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math

import numpy as np
from scipy.ndimage import gaussian_filter as cpu_gaussian_filter, gaussian_gradient_magnitude

from .config import SceneConfig
from .backend import BackendInfo
from .phases import apply_phase_preset


@dataclass
class AppearanceVolume:
    emissive_rgb: np.ndarray
    extinction: np.ndarray
    albedo: np.ndarray
    shock: np.ndarray
    filament: np.ndarray
    dust: np.ndarray
    metadata: Dict[str, Any]


PALETTES = {
    "hubble_like": {
        "hydrogen": np.array([1.00, 0.42, 0.16], dtype=np.float32),
        "oxygen": np.array([0.22, 0.78, 1.00], dtype=np.float32),
        "sulfur": np.array([0.95, 0.20, 0.46], dtype=np.float32),
        "halo": np.array([0.45, 0.62, 1.00], dtype=np.float32),
    },
    "muted_trueish": {
        "hydrogen": np.array([0.94, 0.48, 0.30], dtype=np.float32),
        "oxygen": np.array([0.36, 0.62, 0.84], dtype=np.float32),
        "sulfur": np.array([0.77, 0.34, 0.48], dtype=np.float32),
        "halo": np.array([0.52, 0.60, 0.92], dtype=np.float32),
    },
}


def _xp_rng(backend: BackendInfo, seed: int):
    if backend.name == "cuda":
        return backend.xp.random.default_rng(seed)
    return np.random.default_rng(seed)


def _to_numpy(arr):
    try:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr)


def _normalize(arr, xp):
    arr_min = arr.min()
    arr_max = arr.max()
    denom = xp.maximum(arr_max - arr_min, xp.asarray(1e-8, dtype=arr.dtype))
    return (arr - arr_min) / denom


def _gaussian_filter(arr, sigma: float, backend: BackendInfo):
    if backend.name == "cuda" and backend.ndimage is not None:
        return backend.ndimage.gaussian_filter(arr, sigma=sigma, mode="wrap")
    return cpu_gaussian_filter(_to_numpy(arr), sigma=sigma, mode="wrap")


def _build_fbm_noise(shape, sigmas, weights, rng, backend: BackendInfo):
    xp = backend.xp
    acc = xp.zeros(shape, dtype=xp.float32)
    for sigma, weight in zip(sigmas, weights):
        noise = rng.standard_normal(shape, dtype=xp.float32)
        smooth = _gaussian_filter(noise, sigma=float(sigma), backend=backend)
        smooth = xp.asarray(smooth, dtype=xp.float32)
        smooth = _normalize(smooth, xp)
        acc = acc + xp.asarray(weight, dtype=xp.float32) * smooth
    return _normalize(acc, xp)


def _create_coords(n: int, backend: BackendInfo):
    xp = backend.xp
    axis = xp.linspace(-1.0, 1.0, n, dtype=xp.float32)
    return xp.meshgrid(axis, axis, axis, indexing="ij")


def _build_knots(shape, shell_radius, shell_thickness, knot_count, rng, backend: BackendInfo):
    xp = backend.xp
    x, y, z = _create_coords(shape[0], backend)
    knots = xp.zeros(shape, dtype=xp.float32)
    for _ in range(int(knot_count)):
        theta = float(rng.uniform(0.0, math.pi))
        phi = float(rng.uniform(-math.pi, math.pi))
        rr = float(rng.normal(loc=shell_radius, scale=shell_thickness * 0.25))
        cx = rr * math.sin(theta) * math.cos(phi)
        cy = rr * math.sin(theta) * math.sin(phi)
        cz = rr * math.cos(theta)
        sigma = float(rng.uniform(shell_thickness * 0.25, shell_thickness * 0.85))
        amp = float(rng.uniform(0.35, 1.0))
        dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        knots = knots + xp.asarray(amp, dtype=xp.float32) * xp.exp(-dist2 / (2.0 * sigma * sigma))
    return _normalize(knots, xp)


def generate_remnant_appearance(scene: SceneConfig, backend: BackendInfo) -> AppearanceVolume:
    scene = SceneConfig(**apply_phase_preset(scene.__dict__.copy()))
    xp = backend.xp
    rng = _xp_rng(backend, scene.seed)

    n = int(scene.grid_size)
    x, y, z = _create_coords(n, backend)

    ax = 1.0 + 0.6 * scene.asymmetry
    ay = 1.0 - 0.35 * scene.asymmetry
    az = 1.0 + 0.15 * scene.asymmetry
    r_ell = xp.sqrt((x / ax) ** 2 + (y / ay) ** 2 + (z / az) ** 2 + 1e-8)
    theta = xp.arccos(xp.clip(z / (r_ell + 1e-8), -1.0, 1.0))
    phi = xp.arctan2(y, x)

    angular_mod = (
        1.0
        + 0.14 * scene.asymmetry * xp.sin(2.0 * phi) * xp.sin(theta) ** 2
        + 0.10 * scene.asymmetry * xp.cos(3.0 * phi + 0.35) * xp.sin(theta) ** 3
        + 0.08 * scene.asymmetry * xp.cos(theta)
    )
    target_radius = scene.shell_radius * angular_mod

    shell_base = xp.exp(-((r_ell - target_radius) / scene.shell_thickness) ** 2)
    shell_skin = xp.exp(-((r_ell - target_radius) / (scene.shell_thickness * 0.52)) ** 2)
    inner_volume = xp.clip(1.0 - (r_ell / (scene.shell_radius * 1.02)), 0.0, 1.0)

    large_noise = _build_fbm_noise((n, n, n), sigmas=[9.0, 5.0], weights=[0.7, 0.3], rng=rng, backend=backend)
    fine_noise = _build_fbm_noise((n, n, n), sigmas=[3.0, 1.4], weights=[0.6, 0.4], rng=rng, backend=backend)
    dust_noise = _build_fbm_noise((n, n, n), sigmas=[7.0, 3.5], weights=[0.75, 0.25], rng=rng, backend=backend)

    filament_raw = xp.clip(0.62 * fine_noise + 0.38 * large_noise - (0.60 - 0.08 * scene.filament_strength), 0.0, 1.0)
    ridge = gaussian_gradient_magnitude(_to_numpy(large_noise).astype(np.float32), sigma=1.0)
    ridge = ridge / max(float(ridge.max()), 1e-6)
    ridge = xp.asarray(ridge, dtype=xp.float32)
    sheets = shell_skin * xp.clip(ridge - 0.28, 0.0, 1.0) ** 1.8
    filament = shell_skin * filament_raw ** 4.4
    filament = _normalize(0.62 * filament + 0.38 * sheets, xp)

    knots = _build_knots((n, n, n), scene.shell_radius, scene.shell_thickness, scene.knot_count, rng, backend)
    knots = shell_skin * knots
    knots = _normalize(knots, xp)

    shell_clump = shell_skin * xp.clip(large_noise - 0.42, 0.0, 1.0) ** 1.6
    shell_clump = _normalize(shell_clump, xp)

    shock = shell_skin * (0.05 + 1.25 * filament + 0.85 * knots + 0.45 * shell_clump)
    shock = xp.clip(shock, 0.0, 1.0)

    halo = xp.exp(-((r_ell - scene.shell_radius * 1.03) / (scene.shell_thickness * 2.7)) ** 2)
    halo *= (0.35 + 0.65 * large_noise)
    halo *= scene.synchrotron_strength
    halo = xp.clip(halo, 0.0, 1.0)

    dust = inner_volume * xp.clip(dust_noise - 0.58, 0.0, 1.0) ** 1.7
    dust *= scene.dust_strength
    dust = xp.clip(dust, 0.0, 1.0)

    palette = PALETTES.get(scene.observational_palette, PALETTES["hubble_like"])

    hydrogen_component = (0.015 * shell_base + 0.12 * shell_clump + 0.18 * shock)[:, :, :, None] * palette["hydrogen"]
    oxygen_component = (1.35 * filament + 0.12 * halo)[:, :, :, None] * palette["oxygen"]
    sulfur_component = (1.15 * knots + 0.18 * shell_clump)[:, :, :, None] * palette["sulfur"]
    halo_component = halo[:, :, :, None] * palette["halo"]

    emissive_rgb = (hydrogen_component + oxygen_component + sulfur_component + halo_component)
    emissive_rgb *= scene.brightness_scale
    emissive_rgb = xp.clip(emissive_rgb, 0.0, None)

    extinction = 0.03 * shell_base + 0.06 * shell_clump + 0.65 * dust + 0.04 * halo + 0.05 * filament
    extinction = xp.clip(extinction, 0.0, 1.0)
    albedo = xp.clip(0.10 + 0.42 * filament + 0.22 * shell_clump + 0.12 * halo - 0.10 * dust, 0.0, 1.0)

    active_bbox_mask = _to_numpy((shell_base + halo + dust + filament + knots) > 0.05)
    active_indices = np.argwhere(active_bbox_mask)
    if len(active_indices):
        min_idx = active_indices.min(axis=0).tolist()
        max_idx = active_indices.max(axis=0).tolist()
    else:
        min_idx = [0, 0, 0]
        max_idx = [n - 1, n - 1, n - 1]

    metadata = {
        "schema_version": "SNV-APPEARANCE-1.0",
        "phase": scene.phase,
        "grid_size": n,
        "age_years": scene.age_years,
        "physical_extent_ly": scene.physical_extent_ly,
        "palette": scene.observational_palette,
        "active_bbox_index": {"min": min_idx, "max": max_idx},
        "channel_semantics": {
            "emissive_rgb": "line-inspired visible/composite emissivity",
            "extinction": "appearance-only absorption/extinction",
            "albedo": "single-scatter artistic proxy",
            "shock": "shock-edge and brightness mask",
            "filament": "thin-structure emission mask",
            "dust": "internal attenuation mask",
        },
        "ue_svt_packing_hint": {
            "AttributesA.R": "emissive_r",
            "AttributesA.G": "emissive_g",
            "AttributesA.B": "emissive_b",
            "AttributesA.A": "extinction",
            "AttributesB.R": "albedo",
            "AttributesB.G": "shock",
            "AttributesB.B": "filament",
            "AttributesB.A": "dust",
        },
    }

    return AppearanceVolume(
        emissive_rgb=_to_numpy(emissive_rgb).astype(np.float32),
        extinction=_to_numpy(extinction).astype(np.float32),
        albedo=_to_numpy(albedo).astype(np.float32),
        shock=_to_numpy(shock).astype(np.float32),
        filament=_to_numpy(filament).astype(np.float32),
        dust=_to_numpy(dust).astype(np.float32),
        metadata=metadata,
    )
