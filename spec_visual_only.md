# Visual-Only Supernova / Remnant Pipeline Spec (v1)

## Goal

Produce a visually credible supernova-remnant volume for offline rendering and Unreal Engine ingestion **without** carrying a full scientific state vector. The cache only preserves what the renderer needs.

This spec is tuned for the visually dramatic, filament-rich remnant / nebula phase rather than the immediate supernova flash.

## Design shift from the physics-heavy pipeline

The old design preserved density, temperature, composition tracers, and RT-calibrated proxies.

The refined design preserves only these appearance channels:

- `emissive_r`
- `emissive_g`
- `emissive_b`
- `extinction`
- `albedo`
- `shock`
- `filament`
- `dust`

These are enough to:

- ray-march a remnant in a reference renderer,
- pack the result into UE Sparse Volume Textures later,
- drive volume-domain materials in UE Heterogeneous Volumes,
- preserve the morphology and color structure that makes remnants visually compelling.

## Implemented first slice

1. `remnant_shell` appearance model
2. appearance cache export (`npz` + manifest)
3. reference volume renderer (emission + absorption)
4. quick E2E test

## Deferred next slice

1. direct UE import automation
2. full VDB/SVT importer validation in UE
3. early photospheric supernova phase
4. RT calibration from external physical simulations
