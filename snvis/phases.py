from __future__ import annotations

from typing import Dict, Any

PHASE_PRESETS: Dict[str, Dict[str, Any]] = {
    "remnant_shell": {
        "shell_radius": 0.62,
        "shell_thickness": 0.07,
        "asymmetry": 0.18,
        "filament_strength": 1.2,
        "knot_count": 64,
        "synchrotron_strength": 0.28,
        "dust_strength": 0.20,
        "brightness_scale": 1.0,
    },
    "remnant_shell_hero": {
        "shell_radius": 0.61,
        "shell_thickness": 0.055,
        "asymmetry": 0.23,
        "filament_strength": 1.45,
        "knot_count": 120,
        "synchrotron_strength": 0.34,
        "dust_strength": 0.24,
        "brightness_scale": 1.12,
    },
}


def apply_phase_preset(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    phase = scene_dict.get("phase", "remnant_shell")
    preset = PHASE_PRESETS.get(phase, {})
    return {**preset, **scene_dict}
