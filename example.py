"""Run the fusion subset on synthetic calibrated supports only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from fusion import fuse_supports


EXAMPLES = (
    ("high supports", (0.95, 0.90, 0.92)),
    ("low supports", (0.10, 0.15, 0.12)),
    ("ambiguous supports", (0.50, 0.50, 0.50)),
    ("one low support", (0.95, 0.90, 0.10)),
    ("all supports at one", (1.00, 1.00, 1.00)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supports", nargs=3, type=float, metavar=("E_D", "E_Q", "E_H"),
        help="three finite calibrated supports in [0, 1]",
    )
    parser.add_argument("--json", action="store_true", help="print structured JSON")
    args = parser.parse_args()
    examples = (("custom", tuple(args.supports)),) if args.supports else EXAMPLES
    records = []
    for name, supports in examples:
        try:
            result = fuse_supports(supports)
        except ValueError as exc:
            parser.error(str(exc))
        records.append({"name": name, "supports": supports, **asdict(result)})

    if args.json:
        print(json.dumps({"fusion_only": True, "cases": records}, indent=2, allow_nan=False))
        return
    print("NCTM: fusion component with synthetic calibrated supports.")
    print(f"{'Example':<22} {'(e_D, e_Q, e_H)':<21} {'T':>8} {'U':>8} {'C':>8}")
    for record in records:
        supports = ", ".join(f"{value:.2f}" for value in record["supports"])
        print(
            f"{record['name']:<22} {supports:<21} "
            f"{record['trust']:8.4f} {record['uncertainty']:8.4f} {record['conflict']:8.4f}"
        )


if __name__ == "__main__":
    main()
