"""The first recursive session's inner side (hiding model architectures, lane rec-v0; ``80-rec-inner.sh``): the inner
statement staged as M0's ``verity/flock-circuit`` over RoPE d64, proved by the unmodified prover against
``verity_flock.rec_live``'s proxy and coin server.

    python rec_inner.py stage --n N --sets DIR --out DIR [--plain-leaves] [--sparse SHAPE --stacked FLOCK_CIRCUIT [--forge cell]]
                                                             circuit.txt, inst-N.bin, pub-N.bin, and the verifier's partition.json,
                                                             program.json and draw.json (its full draw, rec_live.full_draw);
                                                             prints the statement's shape
                                                             (--plain-leaves: the trees' leaves plain SHA-512, circuit.LEAF_SCHEME_PLAIN;
                                                             --stacked: R registered at staging, rec_session.stage_stacked: the
                                                             circuit registration META binds, circuit-registration.json, and the
                                                             prover's R.bin, ROOT.json and circuit-registration-private.json)
    python rec_inner.py forge --what challenge|query --inner DIR [--record session|prover]
                                                             flips one coin of rep 0 in DIR's session.json (session/'s, or
                                                             prover/'s: candidate 3's, challenge only) and public.json,
                                                             written in the coin server's own format (rec_live.record_bytes)
    python rec_inner.py forge --what nodraw|partial --inner DIR [--record session|prover]
                                                             the record without its sample, or with its last sampled proof block dropped
    python rec_inner.py forge --what top|unsalted --inner DIR
                                                             top: flips one byte of rep 0's level-1 top in private.json;
                                                             unsalted: public.json's tops as their clear nodes (no salts)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verity_flock import circuit as FC
from verity_flock import rec_live as RL
from verity_flock import rec_vstar as V
from verity_flock.record_text import session_text

SEED = 20261004
STREAM = "circuit/rep0"


def stage(n: int, sets: Path, out: Path, plain_leaves: bool = False, sparse=None, stacked: Path | None = None,
          forge: str | None = None) -> dict:
    """The inner statement, bound to the `verity/partition/v1` object `Q_template_instance` v0 over `population_program` of
    the set's template at ``n`` (instance i is unit i), which every verifier of the inner statement holds (``--partition``,
    ``--rec-inner-partition``), written beside it as ``partition.json`` with its program, ``program.json``, and the draw the
    coin server answers ``Register`` with, every unit (`rec_live.full_draw`), ``draw.json``. ``stacked``
    (flock-circuit, with ``sparse``, the bucket's `rec_sparse.Shape`): the stacked table R registered with it here
    (`rec_session.stage_stacked`, ``forge`` ``cell`` its forgery), the circuit registration whose digest META binds."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    s = InputSet.open(generate(TM.rope_head(64), n, SEED, sets)["path"])
    low = FC.lowering_for_set(s)
    template = s.subcircuit.definition(**FC.set_statics(s))
    program = P.population_program(template, n)
    part = P.partition(program, P.template_instance_query(template))
    FC.stage(low, s, 0, n, out, partition=P.digest(part), program=part["program"],
             leaf_scheme=FC.LEAF_SCHEME_PLAIN if plain_leaves else None)
    (out / "partition.json").write_text(json.dumps(part))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    (out / "draw.json").write_text(json.dumps(RL.full_draw(program, part["query"])))
    if stacked is not None:
        from verity_flock import rec_session as RS

        RS.stage_stacked(out, stacked, sparse, program, part, forge)
    text = (out / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    G = meta["vus_per_block"]
    blocks = -(-n // G)
    return {"subcircuit": s.subcircuit.id, "unit": low.name, "n": n, "vus_per_block": G, "k_log": meta["k_log"],
            "units_per_vu": meta["units_per_vu"], "unit_log": meta["unit_log"], "unit_rows": low.layout.useful,
            "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "blocks": blocks,
            "m": meta["k_log"] + max(3, (blocks - 1).bit_length()), "leaf_scheme": meta["leaf_scheme"]["id"],
            "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text), "partition_digest": P.digest(part),
            "program_sha512": part["program"], **({"registration": meta["registration"]} if "registration" in meta else {})}


def forge(what: str, inner: Path, level: int = 0, record: str = "session") -> dict:
    """Flip the low bit of one coin of rep 0, in the session record (``record``: ``session/``'s, or ``prover/``'s, which
    candidate 3's relay writes and whose rounds after z_partial are numbered apart, so only ``challenge``) and in the coin
    server's public record alike: the first coin after the binding round (``challenge``) or the first query coin of Ligerito
    level ``level`` (``query``). ``nodraw`` and ``partial`` forge the record's sample (none, or its last unit dropped);
    ``top`` and ``unsalted`` forge the tree tops instead (the module docstring)."""
    if what in ("nodraw", "partial"):
        f = inner / record / "session.json"
        sess = json.loads(f.read_text())
        if what == "nodraw":
            sess["unit_draw"] = None
        else:
            sess["unit_draw"]["units"] = sess["unit_draw"]["units"][:-1]
        f.write_text(session_text(sess))
        return {"forged": what, "record": record}
    if what == "query" and record != "session":
        raise ValueError("query: the prover's record numbers its rounds after z_partial apart from public.json's")
    pub, priv, proofs = V.load(inner)
    if what == "top":
        p = next(p for p in priv["rounds"] if p["stream"] == STREAM and p["k"] > 0 and p.get("tops"))
        nodes = p["tops"][0]["nodes"]
        nodes[0] = format(int(nodes[0][:2], 16) ^ 1, "02x") + nodes[0][2:]
        (inner / "private.json").write_text(json.dumps(priv))
        return {"forged": what, "stream": STREAM, "k": p["k"], "top": 0}
    if what == "unsalted":
        clear = {(p["stream"], p["k"]): [o["nodes"] for o in p.get("tops", [])] for p in priv["rounds"]}
        for r in pub["rounds"]:
            r["tops"] = clear.get((r["stream"], r["k"]), [])
        pub["link"]["roots"] = {t: v["nodes"] for t, v in priv["roots"].items()}
        del pub["tops"]
        (inner / "public.json").write_text(json.dumps(pub))
        return {"forged": what}
    if what == "challenge":
        k = next(r["k"] for r in V._stream_rounds(pub, STREAM) if r["k"] > 0 and r["n"])
    else:
        k = V.query_rounds(pub, STREAM, [len(lv.rows) for lv in V.decode(proofs[0])["levels"]])[level]["k"]
    sess = json.loads((inner / record / "session.json").read_text())
    for rd in (next(r for r in next(s for s in sess["streams"] if s["stream"] == STREAM)["rounds"] if r["k"] == k),
               next(r for r in pub["rounds"] if r["stream"] == STREAM and r["k"] == k)):
        c = rd["coins"][0]
        rd["coins"][0] = c[:-1] + format(int(c[-1], 16) ^ 1, "x")
    (inner / record / "session.json").write_text(json.dumps(sess))
    (inner / "public.json").write_text(RL.record_bytes(pub))
    return {"forged": what, "record": record, "stream": STREAM, "k": k, "coin": 0, **({"level": level} if what == "query" else {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    st = sp.add_parser("stage")
    st.add_argument("--n", type=int, required=True)
    st.add_argument("--sets", required=True)
    st.add_argument("--out", required=True)
    st.add_argument("--plain-leaves", action="store_true")
    st.add_argument("--sparse", metavar="NU,ELL,S_LOG,T_LOG", help="--stacked: the bucket's RecSparse shape, R's")
    st.add_argument("--stacked", metavar="FLOCK_CIRCUIT", help="register the stacked table R with the statement, by this build")
    st.add_argument("--forge", choices=("cell",), help="--stacked: R registered with cell 0 flipped")
    fg = sp.add_parser("forge")
    fg.add_argument("--what", choices=("challenge", "query", "nodraw", "partial", "top", "unsalted"), required=True)
    fg.add_argument("--inner", required=True)
    fg.add_argument("--level", type=int, default=0)
    fg.add_argument("--record", choices=("session", "prover"), default="session")
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        if bool(a.stacked) != bool(a.sparse):
            ap.error("--stacked and --sparse come together: R is the bucket's table")
        sparse = None
        if a.sparse:
            from verity_flock.rec_sparse import Shape
            sparse = Shape(*(int(v) for v in a.sparse.split(",")))
        res = stage(a.n, Path(a.sets), Path(a.out), a.plain_leaves, sparse, a.stacked and Path(a.stacked), a.forge)
    else:
        res = forge(a.what, Path(a.inner), a.level, a.record)
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
