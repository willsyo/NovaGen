"""Visual-only supernova remnant pipeline.

This package builds appearance-first sparse volume caches and renders a
lightweight reference preview using a simple emission-absorption model.
"""

from .config import load_config, PipelineConfig
from .pipeline import run_pipeline

__all__ = ["load_config", "PipelineConfig", "run_pipeline"]
