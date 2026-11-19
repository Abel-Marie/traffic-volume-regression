"""Command-line entry point: ``traffic-volume <stage> [--fast]``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .artifacts import RunContext, to_jsonable
from .config import apply_fast_mode, load_config
from .pipeline import STAGES, run_all, run_drift, run_latency, run_stage


def find_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "configs" / "config.yaml").exists():
            return candidate
    raise SystemExit("Run inside the project (no configs/config.yaml found above the current folder).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="traffic-volume",
                                     description="Hourly I-94 traffic volume: the ML process, stage by stage.")
    parser.add_argument("stage", choices=[*STAGES, "all", "monitor"], help="stage to run, 'all', or 'monitor'")
    parser.add_argument("what", nargs="?", choices=["latency", "drift"], help="for 'monitor': latency or drift")
    parser.add_argument("--fast", action="store_true", help="tiny budgets for smoke tests (never for results)")
    parser.add_argument("--root", type=Path, default=None, help="project root (default: found automatically)")
    parser.add_argument("--config", type=Path, default=None, help="config file (default: configs/config.yaml)")
    parser.add_argument("--url", default=None, help="for 'monitor latency': base URL of the running API")
    args = parser.parse_args(argv)

    root = (args.root or find_root(Path.cwd())).resolve()
    cfg = load_config(args.config or root / "configs" / "config.yaml")
    fast = args.fast or os.environ.get("ML_FAST") == "1"
    if fast:
        cfg = apply_fast_mode(cfg)
    # A fast run writes everything under build/fast/, so it can never overwrite the reported results.
    ctx = RunContext(root / "build" / "fast" if fast else root, repo=root).ensure()
    if fast:
        print(f"FAST MODE: tiny budgets; outputs go to {ctx.root.relative_to(root).as_posix()}/ (not the reported results).")

    if args.stage == "all":
        run_all(cfg, ctx)
    elif args.stage == "monitor":
        if args.what == "latency":
            if not args.url:
                parser.error("monitor latency needs --url")
            result = run_latency(cfg, ctx, args.url)
            result.pop("client_latencies_ms", None)
        elif args.what == "drift":
            result = run_drift(cfg, ctx)
        else:
            parser.error("monitor needs 'latency' or 'drift'")
        print(json.dumps(to_jsonable(result), indent=2))
    else:
        print(json.dumps(to_jsonable(run_stage(args.stage, cfg, ctx)), indent=2, default=str))


if __name__ == "__main__":
    main()
