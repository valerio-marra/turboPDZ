"""Validate exported bundles against local source predictions without catalog exports."""

import argparse
import json
from pathlib import Path
from turbopdz.artifacts import write_json
from turbopdz.recipes.hsc_validation import validate_legacy_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--artifacts", default="artifacts/hsc-pdr3")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--report", default="artifacts/hsc-pdr3/validation.json")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("paper-runs.json").read_text())
    results = []
    for spec in manifest:
        name = f"{spec['cat']}-{spec['layer']}"
        print(f"Validating {name}", flush=True)
        result = validate_legacy_pair(args.project_root, spec, Path(args.artifacts) / name,
                                      sample_size=args.sample_size)
        results.append(result)
        print(result, flush=True)
    write_json(args.report, {"validation": results, "catalog_rows_exported": False})


if __name__ == "__main__":
    main()
