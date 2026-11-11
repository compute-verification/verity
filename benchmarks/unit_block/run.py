"""A word primitive's Boolean version as one Flock proof block, built (`verity_flock.class_statement.UnitStager`): the select
alone, the unit's slot of 2^`unit_log` witness bits, and the whole block with its rows' hm96 openings, `circuit.compose`'s ranges
in 2^`k_log` bits, each against Flock's proved range of 2^`circuit.K_MAX` (27).

    uv run benchmarks/unit_block/run.py WORD [--first V,V,...] [--stage N] --scratch DIR --out DIR

WORD is a word primitive's id, a lazy family's member included (`GatherF32x50304_v1`). The unit is its Boolean version
(`verity.definitions.boolean.boolean_version`) with each parameter one ``hm96-sha512/row/v2`` bit row and its return one
more. `circuit.compose` refuses a block past 2^`K_MAX`; this lifts that bound while it builds, so such a block is measured, not
refused, and its record says by how much it is over. ``--stage N`` stages N instances on the built circuit into ``--scratch``,
each checked against the word primitive (`UnitStager.stage`): the first parameter's value in each is ``--first``'s (an index
in range, at its ends and past them), the rest random. Writes ``DIR/unit-block-WORD.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

LIFTED_K_MAX = 40


def _peak_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _rows(definition, firsts: list[int], n: int, seed: int) -> dict:
    """Each parameter's row ``(n, words)`` u16: the first parameter's bits the values ``firsts`` (two's complement, cycled), the
    others' random."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = {}
    for i, (name, t) in enumerate(definition.signature()[0]):
        bits = rng.integers(0, 2, (n, t.leaves), dtype=np.uint8)
        if i == 0 and firsts:
            v = [firsts[j % len(firsts)] % (1 << t.leaves) for j in range(n)]
            bits = np.array([[(x >> b) & 1 for b in range(t.leaves)] for x in v], dtype=np.uint8)
        out[name] = np.packbits(bits, axis=1, bitorder="little").view("<u2")
    return out


def build(word_id: str, firsts: list[int], stage: int, scratch: Path) -> dict:
    from verity.circuits.codec import descriptor_id, lookup_primitive
    from verity.circuits.defs import REGISTRY, CompositeDefinition, bind
    from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity.definitions import boolean as BB
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS

    BB.load()  # every registry module, so a word only a module registers (`ref_prims`' F32ExpRn_v1) resolves
    word = lookup_primitive(REGISTRY, word_id)
    d = BB.boolean_version(word)
    d = bind(d) if isinstance(d, CompositeDefinition) else d
    params = d.signature()[0]
    model = FC.unit_witness(d)
    t0 = time.time()
    k_max, FC.K_MAX = FC.K_MAX, LIFTED_K_MAX
    try:
        st = CS.UnitStager(d, f"unit-block/{d.name}", [(p, V2, t.leaves, [p]) for p, t in params], [(V2, (0,))])
    finally:
        FC.K_MAX = k_max
    built_s = round(time.time() - t0, 1)
    m, lay = st.meta, st.low.layout
    end = max((r["pos0"] + r["count"]) << r["slot_log"] for r in m["ranges"])
    rec = {
        "word": word_id, "boolean": descriptor_id(d), "k_max": k_max,
        "model": model,
        "select": {"ands": lay.counts["ands"], "rows": lay.useful, "unit_log": m["unit_log"],
                   "share_of_k_max": round((1 << m["unit_log"]) / (1 << k_max), 6), "rows_share_of_k_max": round(lay.useful / (1 << k_max), 6)},
        "block": {"k_log": m["k_log"], "fits": m["k_log"] <= k_max, "occupied_bits": end, "share_of_k_max": round(end / (1 << k_max), 6),
                  "ranges": m["ranges"], "per_vu": m["per_vu"], "vus_per_block": m["vus_per_block"],
                  "rows": [{"port": p, "words": w, "compressions": c} for p, w, c in m["ports"]]},
        "built_s": built_s, "peak_rss_gb": _peak_gb(), "circuit_bytes": len(st.text), "circuit_sha512": st.pin,
    }
    if m["k_log"] > k_max:
        rec["block"]["refused"] = f"{st.low.name}: an instance needs a block of 2^{m['k_log']} bits (at most 2^{k_max})"
    if stage:
        t0 = time.time()
        rows = _rows(d, firsts, stage, 0)
        out = scratch / f"stage-{d.name}"
        got = st.stage(rows, out, hashlib.sha512(b"benchmarks/unit_block").digest())
        rec["staged"] = {"instances": stage, "firsts": firsts,
                         "outputs": [hex(sum(int(w) << 16 * k for k, w in enumerate(r))) for r in got["outs"]["o0"]],
                         "inst_bin_bytes": (out / "inst.bin").stat().st_size, "pub_bin_bytes": (out / "pub.bin").stat().st_size,
                         "stage_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_gb()}
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("word", help="a word primitive's id, e.g. GatherF32x50304_v1")
    ap.add_argument("--first", default="", help="the first parameter's value in each staged instance, comma-separated")
    ap.add_argument("--stage", type=int, default=0, help="stage this many instances on the built circuit")
    ap.add_argument("--scratch", type=Path, required=True, help="where staged instances go (not kept)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    firsts = [int(x, 0) for x in a.first.split(",") if x]
    rec = build(a.word, firsts, a.stage, a.scratch)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"unit-block-{a.word}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    b, s = rec["block"], rec["select"]
    print(f"{rec['boolean']}: select 2^{s['unit_log']} ({s['share_of_k_max']:.3f} of 2^{rec['k_max']}); with its rows' openings "
          f"2^{b['k_log']} ({b['share_of_k_max']:.3f} of 2^{rec['k_max']}), {'fits' if b['fits'] else 'does not fit'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
