"""[E8 NIGHT 2026-09-12, D40 T-3] OFFLINE test, before any GPU: the PROVER's committed row set for the 32x16 configuration EQUALS the VERIFIER's
required set -- identities and per-call / per-chunk counts -- so n2's ``INTERIOR_WIDTH_MISMATCH`` (3590 vs 3530 words at ``t=0/layers.0/pre/w=32``)
cannot recur.

Both sides are the SHIPPED functions, not copies:
* verifier side: ``verifier_v3.steps.required_source_live(inst, pop, sel, c2, retained)`` -> ``src(req, step)`` = E2's ``LazyInteriorLayout`` (the
  live driver's step-4 ``required=`` source, exactly as ``interior_layout_check`` uses it);
* prover side: the gates + widths the fixed ``record_v5.interior_required_v2`` writes into ``required.json`` (derived here through the layout's
  PUBLIC leaves, ``iter_leaves()`` -> ``LeafSpec.gates`` / ``.width``; with ``--required-v2`` the real module is run into a scratch values dir and
  its ``required_v2_gates.bin`` is compared byte-for-byte), framed as ``commit_ru`` frames them: ``_read_required`` -> ``InteriorLayout.build(tree,
  calls, g, w, prefix)`` with the record's ``logs/interior-calls.json`` entry.

Compared per RU: leaf ids (in order), width, words, and the gate identities of every leaf; the totals; and the named leaf
``.../call=t=0/layers.0/pre/w=32/chunk=0``.  E1's inventory count for that call (3590 in n2's log) is quoted for the difference.

    python -m e2e.demo_core.night.required_set_test --record <n2 record dir> --calls <record>/logs/interior-calls.json --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="record dir with protocol_instance.json, population.v5.json, adapter/selective/commitments.json")
    ap.add_argument("--calls", required=True, help="the record's logs/interior-calls.json (--interior-calls of the prover)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--named-call", default="t=0/layers.0/pre")
    ap.add_argument("--named-width", type=int, default=32)
    ap.add_argument("--e1-count", type=int, default=3590, help="E1's inventory words for the named call/width (n2 log)")
    ap.add_argument("--required-v2", action="store_true", help="also run the real record_v5.interior_required_v2 into a scratch values dir and compare")
    ap.add_argument("--limit", type=int, default=0, help="only the first N RUs (0 = all)")
    a = ap.parse_args()
    rec = Path(a.record)
    from e2e.interior import tree as T
    from e2e.interior.layout import InteriorLayout
    from e2e.verifier_v3 import steps

    inst = json.loads((rec / "protocol_instance.json").read_text())
    pop = json.loads((rec / "population.v5.json").read_text())
    sel = rec / "adapter" / "selective"
    c2 = json.loads((sel / "commitments.json").read_text())
    calls_doc = json.loads(Path(a.calls).read_text())
    retained = list((inst.get("epoch") or {}).get("request_ids") or [])
    t0 = time.perf_counter()
    src, label, facts = steps.required_source_live(inst, pop, sel, c2, retained)
    t_src = time.perf_counter() - t0
    out: dict = {"schema": "veritor/e8/required-set-test/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "record": str(rec),
                 "instance": (inst.get("epoch") or {}).get("epoch_id") or inst.get("instance_id"), "required_source": label, "source_seconds": t_src,
                 "source_facts": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)[:200]) for k, v in (facts or {}).items()},
                 "rus": {}, "verdict": None}
    if src is None:
        out["verdict"] = f"REFUSED: verifier required source did not build: {label}"
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(out["verdict"])
        return 3
    rus = sorted(calls_doc)
    if a.limit:
        rus = rus[: a.limit]
    all_equal = True
    tot_v = tot_p = 0
    named = {}
    for ru in rus:
        req, _, step = ru.partition("/")
        t1 = time.perf_counter()
        lay = src(req, step)
        if lay is None:
            out["rus"][ru] = {"verifier": "no interior region (not a retained interior RU)"}
            continue
        v_leaves = list(lay.iter_leaves()) if hasattr(lay, "iter_leaves") else list(lay.leaves)
        # the prover's required.json content (what interior_required_v2 writes): gates in layout order + one width per gate
        g = np.concatenate([np.asarray(lf.gates, dtype=np.int64) for lf in v_leaves]) if v_leaves else np.zeros(0, dtype=np.int64)
        w = np.concatenate([np.full(len(lf.gates), int(lf.width), dtype=np.int64) for lf in v_leaves]) if v_leaves else np.zeros(0, dtype=np.int64)
        # the prover's framing (commit_ru): _read_required sorts by gate, InteriorLayout.build over the record's pinned call order
        order = np.argsort(g, kind="stable")
        calls, prefix = T._read_calls(Path("/nonexistent"), calls_doc[ru])  # noqa: SLF001 -- the shipped reader over the --interior-calls entry
        p_layout = InteriorLayout.build(f"interior/{ru}", calls, g[order], w[order], prefix=prefix)
        p_leaves = list(p_layout.leaves)
        vs = [(lf.canonical_id, int(lf.width), int(len(lf.gates))) for lf in v_leaves]
        ps = [(lf.canonical_id, int(lf.width), int(len(lf.gates))) for lf in p_leaves]
        ids_equal = vs == ps
        gates_equal = ids_equal and all(np.array_equal(np.asarray(x.gates), np.asarray(y.gates)) for x, y in zip(v_leaves, p_leaves))
        diffs = []
        if not ids_equal:
            pm = {x[0]: x for x in ps}
            for x in vs:
                if pm.get(x[0]) != x:
                    diffs.append({"leaf": x[0], "verifier": x[1:], "prover": (pm.get(x[0]) or (None, None))[1:]})
                if len(diffs) >= 8:
                    break
        vw, pw = int(sum(x[2] for x in vs)), int(sum(x[2] for x in ps))
        tot_v += vw
        tot_p += pw
        all_equal &= gates_equal
        rec_ru = {"leaves_verifier": len(vs), "leaves_prover": len(ps), "words_verifier": vw, "words_prover": pw, "ids_widths_counts_equal": ids_equal,
                  "gate_identities_equal": gates_equal, "seconds": time.perf_counter() - t1, "first_diffs": diffs}
        for lf in v_leaves:
            if lf.call == a.named_call and int(lf.width) == a.named_width and int(lf.chunk) == 0:
                pl = next((y for y in p_leaves if y.canonical_id == lf.canonical_id), None)
                named[ru] = {"leaf": lf.canonical_id, "verifier_words": int(len(lf.gates)), "prover_words": (int(len(pl.gates)) if pl else None),
                             "e1_inventory_words_n2_log": a.e1_count}
                rec_ru["named_leaf"] = named[ru]
        out["rus"][ru] = rec_ru
        print(f"{ru}: verifier {len(vs)} leaves / {vw:,} words; prover {len(ps)} / {pw:,}; equal={gates_equal} ({rec_ru['seconds']:.1f}s)" + (f"; {a.named_call} w={a.named_width} chunk 0: {named[ru]['verifier_words']} (verifier) = {named[ru]['prover_words']} (prover); E1 {a.e1_count}" if ru in named else ""))
        sys.stdout.flush()
    if a.required_v2:
        # the REAL module, into a scratch values dir with the RU step dirs present (empty)
        from record_v5 import interior_required_v2 as IR
        with tempfile.TemporaryDirectory(prefix="e8-reqv2-") as td:
            vals = Path(td)
            for ru in rus:
                (vals / ru).mkdir(parents=True)
            rc = IR.main(["--record", str(rec), "--values", str(vals)])
            cmp = {}
            for ru in rus:
                rj = vals / ru / "required.json"
                if not rj.exists():
                    cmp[ru] = "no required.json written"
                    continue
                doc = json.loads(rj.read_text())
                g2 = np.fromfile(vals / ru / doc["gates_bin"], dtype=doc["dtype"]).astype(np.int64)
                lay = src(*ru.partition("/")[::2])
                g1 = np.concatenate([np.asarray(lf.gates, dtype=np.int64) for lf in lay.iter_leaves()])
                cmp[ru] = {"module_words": int(g2.shape[0]), "public_leaves_words": int(g1.shape[0]), "identical_order_and_identities": bool(np.array_equal(np.sort(g1), np.sort(g2)))}
            out["interior_required_v2"] = {"rc": rc, "per_ru": cmp}
            all_equal &= rc == 0 and all(isinstance(v, dict) and v["identical_order_and_identities"] for v in cmp.values())
    out["totals"] = {"words_verifier": tot_v, "words_prover": tot_p, "rus": len(rus)}
    out["named_leaf"] = named
    out["verdict"] = ("EQUAL: the prover's committed set (required.json from the verifier's layout, framed by commit_ru) == the verifier's required set: "
                      f"{len(rus)} RU(s), {tot_v:,} words, every leaf id/width/count and every gate identity equal" if all_equal else "NOT EQUAL (see rus[*].first_diffs)")
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(out["verdict"])
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
