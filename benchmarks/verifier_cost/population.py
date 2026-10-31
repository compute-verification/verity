"""`population.sh`'s inputs: per N, DIR/pop-N.json, `{"descriptor", "query", "population"}` for a template-instance
population of about N units over a layered call hierarchy (a served window's shape): `--roots` root batches cycling over
`--templates` distinct templates, each template the top of `--levels` levels of composites, every composite calling
three of the level below (at most `--width` per level) and XOR-ing their results, under `Q_template_instances` over the
templates.

  python population.py DIR [--roots R] [--templates T] [--levels L] [--width W] N...
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.types import Array, Value

XOR8 = PrimitiveDefinition("BenchPopulationXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)


def _signature(S):
    return (("x", Array(2, Value(8))),), Value(8)


def _body(B, S, x):
    if S.level == 0:
        return B.call(XOR8, x[0], x[1])
    a, b, c = (B.call(node(S.level - 1, (3 * S.i + j) % S.width, S.width), x) for j in range(3))
    return B.call(XOR8, B.call(XOR8, a, b), c)


NODE = CompositeDefinition("BenchPopulationNode", 0, ("level", "i", "width"), _signature, _body, register=False)


@functools.cache
def node(level: int, i: int, width: int):
    return bind(NODE, level=level, i=i, width=width)


def case(n: int, roots: int, templates: int, levels: int, width: int) -> dict:
    tops = [node(levels - 1, t % width, width) for t in range(templates)]
    each, extra = divmod(n, roots)
    members = [(tops[r % templates], each + (1 if r < extra else 0)) for r in range(roots)]
    program = P.mixed_population_program([(t, k) for t, k in members if k > 0])
    query = P.template_instances_query(list(dict.fromkeys(t.id for t, k in members if k > 0)))
    desc = codec.encode_program(program)
    return {"descriptor": desc, "query": query, "population": P.units(program, query)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("ns", type=int, nargs="+")
    ap.add_argument("--roots", type=int, default=234)
    ap.add_argument("--templates", type=int, default=234)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--width", type=int, default=1800)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for n in a.ns:
        c = case(n, a.roots, a.templates, a.levels, a.width)
        (a.out / f"pop-{n}.json").write_text(json.dumps(c, sort_keys=True, separators=(",", ":")))
        print(json.dumps({"n": n, "population": c["population"], "definitions": len(c["descriptor"]["definitions"]),
                          "root_nodes": len(c["descriptor"]["definitions"][c["descriptor"]["root"]]["body"]["nodes"])}))


if __name__ == "__main__":
    main()
