"""Command-line entry point for notebook-equivalent workflow stages."""

from __future__ import annotations
import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="turbopdz")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Execute one parameterized workflow stage")
    run.add_argument("stage", choices=["prepare", "train-zml", "oof", "train-sigmaml", "evaluate", "diagnostics", "bootstrap", "applicability"])
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir", required=True)
    export = commands.add_parser("export", help="Export only deployable model/support artifacts")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        from .workflow import run_stage
        result = run_stage(args.stage, args.config, args.run_dir)
    else:
        from .workflow import export_run
        result = export_run(args.run_dir, args.destination)
    print(result)


if __name__ == "__main__":
    main()
