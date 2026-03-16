from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual-only supernova remnant pipeline")
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(run_pipeline(cfg), indent=2))


if __name__ == "__main__":
    main()
