"""Export trusted HSC reference runs; never overwrite an existing bundle."""

import argparse
import json
from pathlib import Path
from turbopdz.recipes.hsc import export_legacy_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--destination", default="artifacts/hsc-pdr3")
    parser.add_argument("--trust-legacy-pickle", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("paper-runs.json").read_text())
    for spec in manifest:
        name = f"{spec['cat']}-{spec['layer']}"
        print(f"Exporting {name}", flush=True)
        export_legacy_pair(args.project_root, spec, Path(args.destination) / name,
                           trust_pickle=args.trust_legacy_pickle)
    print("Exported six frozen pipeline bundles. No models were retrained.", flush=True)


if __name__ == "__main__":
    main()
