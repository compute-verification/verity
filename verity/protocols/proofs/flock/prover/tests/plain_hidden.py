"""A plain hidden-output session's statement (`Flock.ProvedScope`): the fixture of `test_lean_plain_tree.py`, staged by
`pod/91-plain-hidden.sh`, which proves it in one loopback session without `--zk` under a subset sample.

The program (`program`) is `N` RoPE d64 template instances, instance `i` reading row `i` of `x` and of `cs`, under
`Q_template_instance` v0. `stage` writes, under `out`, the hidden-output circuit bound to that program and its partition
(`circuit.txt`, `program.json`, `partition.json`), the MUFU tables if the lowering cuts any (`tables/`), and the prover's
`inst-N.bin` and the verifier's `pub-N.bin`, one row per instance and port with no shared rows and no registered value: the
form the plain verifier covers at hidden outputs (`FailClosed.SoundExecH` at its subset sample design, one table).

``--n`` stages a population of another size, its instance ``i`` the committed set's instance ``i mod N``, and
``--vus-per-block`` the circuit's VUs a block (`circuit.compose`'s); `pod/95-population-past-shape.sh` stages a population
past the spec's shape this way.

    python verity/protocols/proofs/flock/prover/tests/plain_hidden.py --out DIR [--seed S] [--n N] [--vus-per-block G]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array
from verity.protocols.proofs.spot.verifier.one_stage import partition as P

N = 32
SEED = 20261025


def program(n: int = N) -> tuple[Program, dict]:
    """The verifier's own program and query, at ``n`` instances."""
    from verity_flock import subcircuit_templates as TM

    template = TM.rope_head(64).definition()
    (_, xt), (_, ct) = template.signature()[0]
    ret = template.signature()[1]

    def sig(S):
        return (("x", Array(n, xt)), ("cs", Array(n, ct))), Array(n, ret)

    def body(B, S, x, cs):
        return coll_concat(sig(S)[1], [B.call(template, x[i], cs[i]) for i in range(n)])

    return (Program(bind(CompositeDefinition("PlainHiddenRoPE", 0, (), sig, body, register=False))),
            P.template_instance_query(template))


def stage(out: Path, seed: int = SEED, n: int = N, vus_per_block: int | None = None) -> dict:
    import numpy as np
    from verity_flock import circuit as FC
    from verity_flock import subcircuit_templates as TM
    from synthetic_sets import synthetic_set
    from verity_flock.input_sets import InputSet

    out.mkdir(parents=True, exist_ok=True)
    prog, query = program(n)
    part = P.partition(prog, query)
    s = InputSet.open(synthetic_set(TM.rope_head(64), N, seed, out / "set")["path"])
    low = FC.lowering_for_set(s)
    text = FC.compose(low, s.subcircuit, vus_per_block=vus_per_block, partition=P.digest(part), program=P.program_digest(prog),
                      statics=FC.set_statics(s))
    (out / "circuit.txt").write_text(text)
    (out / "program.json").write_bytes(canonical_json(encode_program(prog)))
    (out / "partition.json").write_text(json.dumps(part))
    (out / "tables").mkdir(exist_ok=True)
    if low.units.cut:
        from verity_flock import ir_lower as IL

        IL.write_tables(out / "tables")
    rows = outs = None
    if n != N:
        def tiled(ports):
            words = {p.name: np.asarray(s.port(p.name, 0, N), dtype=np.uint16).reshape(N, -1) for p in ports}
            return {k: np.resize(w, (n, w.shape[1])) for k, w in words.items()}
        rows, outs = tiled(s.subcircuit.inputs), tiled(s.subcircuit.outputs)
    FC.write(text, s, 0, n, out / f"inst-{n}.bin", out / f"pub-{n}.bin", None, list(range(n)), rows=rows, outs=outs)
    return {"n": n, "program": P.program_digest(prog), "partition": P.digest(part), "pin": FC.digest(text)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--vus-per-block", type=int, default=None)
    a = ap.parse_args(argv)
    print(json.dumps(stage(Path(a.out), a.seed, a.n, a.vus_per_block)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
