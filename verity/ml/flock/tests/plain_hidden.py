"""A plain hidden-output session's statement (PROTOCOL.md §16.15): the fixture of `test_lean_plain_hidden.py`, staged by
`pod/91-plain-hidden.sh`, which proves it in one loopback session without `--zk` under a subset sample.

The program (`program`) is `N` RoPE d64 template instances, instance `i` reading row `i` of `x` and of `cs`, under
`Q_template_instance` v0. `stage` writes, under `out`, the hidden-output circuit bound to that program and its partition
(`circuit.txt`, `program.json`, `partition.json`), the MUFU tables if the lowering cuts any (`tables/`), and the prover's
`inst-N.bin` and the verifier's `pub-N.bin`, one row per instance and port with no shared rows and no registered value: the
form the plain verifier covers at hidden outputs (`FailClosed.SoundExecH` at its subset sample design, one table).

    python verity/ml/flock/tests/plain_hidden.py --out DIR [--seed S]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array
from verity.core.protocols.sampled_proofs.one_stage import partition as P

N = 32
SEED = 20261025


def program() -> tuple[Program, dict]:
    """The verifier's own program and query."""
    from verity_numerical.bench import templates as TM

    template = TM.rope_head(64).definition()
    (_, xt), (_, ct) = template.signature()[0]
    ret = template.signature()[1]

    def sig(S):
        return (("x", Array(N, xt)), ("cs", Array(N, ct))), Array(N, ret)

    def body(B, S, x, cs):
        return coll_concat(sig(S)[1], [B.call(template, x[i], cs[i]) for i in range(N)])

    return (Program(bind(CompositeDefinition("PlainHiddenRoPE", 0, (), sig, body, register=False))),
            P.template_instance_query(template))


def stage(out: Path, seed: int = SEED) -> dict:
    from verity_flock import circuit as FC
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    out.mkdir(parents=True, exist_ok=True)
    prog, query = program()
    part = P.partition(prog, query)
    s = InputSet.open(generate(TM.rope_head(64), N, seed, out / "set")["path"])
    low = FC.lowering_for_set(s)
    text = FC.compose(low, s.subcircuit, partition=P.digest(part), program=P.program_digest(prog), statics=FC.set_statics(s))
    (out / "circuit.txt").write_text(text)
    (out / "program.json").write_bytes(canonical_json(encode_program(prog)))
    (out / "partition.json").write_text(json.dumps(part))
    (out / "tables").mkdir(exist_ok=True)
    if low.units.cut:
        from verity_flock import ir_lower as IL

        IL.write_tables(out / "tables")
    FC.write(text, s, 0, N, out / f"inst-{N}.bin", out / f"pub-{N}.bin", None, list(range(N)))
    return {"n": N, "program": P.program_digest(prog), "partition": P.digest(part), "pin": FC.digest(text)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print(json.dumps(stage(Path(a.out), a.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
