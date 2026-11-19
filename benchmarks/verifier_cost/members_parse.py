"""`members_parse.sh`'s inputs: per N, DIR/N.json, `{"program", "partition"}`: a population of 10·N instances of one
template (`population.node`) and its partition object under `Q_template_instance`, with N blocks of ten units each, the
i-th `[10 i, 10 i + 10)`.

  python members_parse.py DIR N...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from population import node
from verity.circuits import codec
from verity.circuits import partition_object as PO
from verity.circuits.population import population_program


def case(n: int) -> dict:
    t = node(0, 0, 1)
    program = population_program(t, 10 * n)
    blocks = [(10 * i, 10 * i + 10) for i in range(n)]
    return {"program": codec.encode_program(program),
            "partition": PO.build(program, PO.template_instance_query(t), blocks)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("ns", type=int, nargs="+")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for n in a.ns:
        (a.out / f"{n}.json").write_text(json.dumps(case(n), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
