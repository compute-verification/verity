"""A templated statement (`verity/flock-circuit/tmpl`, PRF-133) staged from a flat one: the fixture `pod/90-tmpl-unit.sh`
proves and hands to the Lean verifier.

``template`` composes the flat hidden-output statement of ``N`` RoPE d64 instances (``flat.txt``), bound to the
`verity/partition/v1` object `Q_template_instance` over `population_program` of the set's template at ``N`` (instance i is
unit i), which the verifier holds (``partition.json``, with its program, ``program.json``), and writes its unit as a
`flock-ir-tmpl/v2` template (``unit.tmpl``, :func:`verity_flock.tmpl.emit`). ``stage`` then writes the templated statement
(``circuit.txt``): head ``flock-circuit/tmpl NAME``, META's ``format`` and ``statement`` the templated statement's, its
``template`` the template's lines and its ``layout`` the layout mode, and every section but ``CIRCUIT unit``, which the
verifier reads from the template. The unit's range stays where the flat statement put it, so ``stage`` refuses a layout
whose committed unit (``flock-tmpl-unit``'s (``tests/lean/TmplUnit.lean``) ``TMPL`` line, ``--tmpl``) needs a larger slot. It then writes the
prover's ``inst-N.bin`` and the verifier's ``pub-N.bin`` against that text (:func:`verity_flock.circuit.write`).

    python verity/ml/flock/tests/tmpl_statement.py template --out DIR
    python verity/ml/flock/tests/tmpl_statement.py stage --out DIR --tmpl TMPL.json [--layout iv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

N = 8
SEED = 20261022
FORMAT = "flock-circuit/tmpl"
STATEMENT = "verity/flock-circuit/tmpl"


def _set(out: Path):
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    if not (out / "set.json").exists():
        (out / "set.json").write_text(json.dumps({"path": str(generate(TM.rope_head(64), N, SEED, out / "set")["path"])}))
    return InputSet.open(json.loads((out / "set.json").read_text())["path"])


def sections(text: str) -> tuple[str, dict, list[tuple[str, str]]]:
    """The head, META and each ``CIRCUIT name n`` section's lines."""
    lines = text.split("\n")
    head, meta, out, at = lines[0], json.loads(lines[1][5:]), [], 2
    while at < len(lines) and lines[at]:
        _, name, n = lines[at].split()
        out.append((name, "\n".join(lines[at + 1:at + 1 + int(n)]) + "\n"))
        at += 1 + int(n)
    return head, meta, out


def template(out: Path) -> dict:
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity_flock import circuit as FC
    from verity_flock import tmpl as T

    out.mkdir(parents=True, exist_ok=True)
    s = _set(out)
    low = FC.lowering_for_set(s)
    definition = s.subcircuit.definition(**FC.set_statics(s))
    program = P.population_program(definition, N)
    part = P.partition(program, P.template_instance_query(definition))
    text = FC.compose(low, s.subcircuit, partition=P.digest(part), program=part["program"], statics=FC.set_statics(s))
    (out / "partition.json").write_text(json.dumps(part))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    (out / "flat.txt").write_text(text)
    unit = dict(sections(text)[2])["unit"]
    t, stats = T.emit(T.parse_flat(unit.rstrip("\n").split("\n")))
    if not stats["order"]["ok"]:
        raise ValueError(f"the unit's template is not ordered: {stats['order']}")
    (out / "unit.tmpl").write_text(T.write(t))
    return {"unit": low.name, "flat_rows": stats["flat_rows"], "defs": stats["defs"], "unit_log": json.loads(text.split("\n", 2)[1][5:])["unit_log"]}


def stage(out: Path, tmpl: dict, layout: str = "iv") -> dict:
    from verity_flock import circuit as FC

    head, meta, secs = sections((out / "flat.txt").read_text())
    if not tmpl.get("ok") or tmpl["unit_log"] != meta["unit_log"]:
        raise ValueError(f"the committed unit ({tmpl}) is not in the flat unit's 2^{meta['unit_log']} slot")
    meta.update(format=FORMAT, statement=STATEMENT, template=(out / "unit.tmpl").read_text().rstrip("\n").split("\n"), layout=layout)
    text = (f"{FORMAT} {head.split(' ', 1)[1]}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"
            + "".join(f"CIRCUIT {nm} {body.count(chr(10))}\n{body}" for nm, body in secs if nm != "unit"))
    (out / "circuit.txt").write_text(text)
    FC.write(text, _set(out), 0, N, out / f"inst-{N}.bin", out / f"pub-{N}.bin")
    return {"n": N, "layout": layout, "pin": FC.digest(text), "class": FC.class_digest(text), "unit_log": meta["unit_log"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["template", "stage"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmpl", type=Path, help="flock-tmpl-unit's TMPL line's JSON (stage)")
    ap.add_argument("--layout", default="iv")
    a = ap.parse_args(argv)
    r = template(a.out) if a.step == "template" else stage(a.out, json.loads(a.tmpl.read_text()), a.layout)
    print(json.dumps(r, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
