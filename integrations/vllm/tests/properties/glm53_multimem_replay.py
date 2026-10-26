"""GLM-5.3 TP8's recorded multimem all-reduces (torch's `multimem_all_reduce_` on 8 H200s, the capture of record
`tp_allreduce.CAPTURE_RUNS`) replayed on `AllReduceMultimem_v1` as it is stated now, a batch of `AllReduceMultimemVector_v1` (one
16-byte vector each) per run of one tree, on the CPU.

A run directory holds the capture's record (`glm53_tp8_allreduce_capture.json`, at the top or under ``out/``) and the rank-0 words
it kept (``words/``: every probe, and the served multimem all-reduces at the probes' M). For each kept launch the replay remakes its
operands from the recorded seed (`glm53_tp8_allreduce_capture_gpu.make_operands`), takes the route `tp_allreduce.covered` names at
the launch's setting (from its set of rank processes' custom all-reduce and environment), hidden size and M, which must be
`AllReduceMultimem_v1` at the record's pinned trees and FTZ, and holds the kept words to it three ways: the word Definition at the
launch's SIZE, the Boolean vector `AllReduceMultimemVector_v2` bit-sliced on every vector of each run of one tree, and
`tp_allreduce_check.multimem_rows`. The chain 0..7 at every owner is counted beside them, so the record shows the words tell trees
apart; a probe's words are decoded again (`decode_probe`, `decide_ftz`) to the pinned trees and FTZ. The capture's own verdict on
the launch (its words every rank's, no mismatch) is read from the record. It writes ``<out>/glm53_multimem_replay.json`` and
prints ``GLM53-MULTIMEM-REPLAY GO`` or ``NO-GO``:

    python tests/properties/glm53_multimem_replay.py --run DIR [--run DIR ...] --out DIR [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import glm53_tp8_allreduce_capture_gpu as X
from verity.ml.definitions.registry import tp_allreduce as T
from verity.ml.definitions.registry import tp_allreduce_check as K

OUT = "glm53_multimem_replay.json"
#: vectors per Boolean evaluation, so a vector's every wire over them stays a few hundred MB
BOOL_VECTORS = 1 << 14
ORDERS = ("boolean", "word", "model", "chain")
HELD = ("boolean", "word", "model")


def _tree(t):
    return t if isinstance(t, int) else tuple(_tree(x) for x in t)


def record_of(run: Path) -> tuple[Path, dict[str, Any]]:
    for d in (run, run / "out"):
        if (d / X.RECORD).is_file():
            return d, json.loads((d / X.RECORD).read_text())
    raise FileNotFoundError(f"{run}: no {X.RECORD} (nor out/{X.RECORD})")


def launches(run: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Every launch whose rank-0 words `run` kept, with its route's statics and the record's pins; and what does not hold of
    them (no case, off the multimem route, trees or FTZ not the record's, the capture's own verdict not GO)."""
    d, doc = record_of(run)
    by_file = {f"words/{c['key'].replace('/', '_')}.npy": c for c in doc.get("case_list", [])}
    out, problems = [], []
    for w in doc.get("words", []):
        c = by_file.get(w)
        if c is None:
            problems.append(f"{run}: {w} is no case's words")
            continue
        spawn = doc["spawns"][c["spawn"]]
        M, H = int(c["M"]), int(c.get("H", spawn.get("H", X.HIDDEN)))
        setting, route = T.served_route(T.settings_of(not spawn["custom_all_reduce"], spawn["env"]), M, H)
        cov = T.covered(setting, H, route, M) if setting else None
        pins = doc["h7168"] if c["phase"].startswith("h7168") else doc["probes"]
        verdict = doc["cases"].get(c["key"], {})
        totals = verdict.get("totals", {})
        why = []
        if cov is None or cov.definition != "AllReduceMultimem_v1":
            why.append(f"route {setting!r} {route!r} at N={H} M={M} covers {cov and cov.definition!r}, not AllReduceMultimem_v1")
        elif (T.multimem_trees(cov.statics["TREES"], X.WORLD) != T.multimem_trees(_tree(pins["trees"]), X.WORLD)
              or bool(cov.statics["FTZ"]) != pins["ftz"]):
            why.append(f"CAPTURED's trees and FTZ are not the record's pinned {pins['trees']} {pins['ftz']}")
        if c.get("comm") not in (None, route) or c.get("setting") not in (None, setting):
            why.append(f"the record ran it at {c.get('setting')!r} on {c.get('comm')!r}, not {setting!r} {route!r}")
        if not (verdict.get("ok") is True and totals.get("words") == M * H and totals.get("mismatch") == 0
                and totals.get("rank_mismatch") == 0):
            why.append(f"the capture's verdict is not GO: ok={verdict.get('ok')} totals={totals}")
        problems += [f"{run} {c['key']}: {x}" for x in why]
        if cov is not None and cov.definition == "AllReduceMultimem_v1":
            out.append({"run": str(run), "record": doc.get("digest"), "key": c["key"], "phase": c["phase"], "seed": c["seed"],
                        "M": M, "H": H, "setting": setting, "route": route, "words": str(d / w),
                        "statics": T.route_binding(cov, X.WORLD, M, H)[1]})
    return out, problems


def boolean_rows(parts: np.ndarray, trees, ftz: bool) -> np.ndarray:
    """(WORLD, SIZE) words through `AllReduceMultimemVector_v2`, one instance per vector of each run of one tree."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import bits as BS
    from verity.ml.definitions.registry import boolean_tp_allreduce as BA

    W, size = parts.shape
    V = T.MULTIMEM_VECTOR
    out = np.empty(size, np.uint16)
    for lo, hi, tree in T.multimem_runs(W, size, trees):
        vec = bind(BA.AllReduceMultimemVector, WORLD=W, TREE=tree, FTZ=bool(ftz))
        cols = parts[:, lo:hi].reshape(W, -1, V).transpose(1, 0, 2)
        for a in range(0, cols.shape[0], BOOL_VECTORS):
            got = np.asarray(BS.evaluate_words(vec, np.ascontiguousarray(cols[a:a + BOOL_VECTORS])), np.uint64)
            out[lo + a * V:lo + a * V + got.size] = got.reshape(-1).astype(np.uint16)
    return out


def replay_one(launch: dict[str, Any]) -> dict[str, Any]:
    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.defs import bind

    st = launch["statics"]
    trees, ftz, size = st["TREES"], bool(st["FTZ"]), int(st["SIZE"])
    with tempfile.TemporaryDirectory() as tmp:
        ops = X.make_operands(Path(tmp), {"phase": launch["phase"], "case": launch["key"].split("/")[1], "key": launch["key"],
                                          "seed": launch["seed"], "M": launch["M"], "H": launch["H"]})
        parts = np.load(ops["ops"]["parts"]).reshape(X.WORLD, -1)
    words = np.load(launch["words"]).reshape(-1).astype(np.uint16)
    spec = bind(T.AllReduceMultimem, **st)
    got = {"boolean": boolean_rows(parts, trees, ftz),
           "word": np.asarray(E.evaluate_batch(spec, parts[None]), np.uint64).reshape(-1).astype(np.uint16),
           "model": K.multimem_rows(parts, trees, ftz),
           "chain": K.multimem_rows(parts, (T.chain(X.WORLD),) * X.WORLD, ftz)}
    rec = {k: launch[k] for k in ("run", "key", "M", "H", "setting", "route")}
    rec.update(spec=spec.id.split("{")[0], size=size, elements=int(words.size), vectors=size // T.MULTIMEM_VECTOR,
               runs=len(T.multimem_runs(X.WORLD, size, trees)),
               differ={o: int((words != g).sum()) if words.size == size else size for o, g in got.items()})
    if launch["phase"] in X.PROBE_PHASES:
        dec = X.decode_probe(words, size)["trees"]
        rec["decoded"] = {"trees": dec is not None and T.multimem_trees(dec, X.WORLD) == T.multimem_trees(trees, X.WORLD),
                          "ftz": X.decide_ftz(words, parts)["ftz"] == ftz}
    return rec


def replay(a: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    todo, problems, records = [], [], {}
    for run in map(Path, a.run):
        got, why = launches(run)
        todo += got
        problems += why
        records[str(run)] = record_of(run)[1].get("digest")
    if a.jobs <= 1:
        results = [replay_one(x) for x in todo]
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(replay_one, todo))
    for r in results:
        if r["elements"] != r["size"]:
            problems.append(f"{r['run']} {r['key']}: {r['elements']} words kept, not SIZE {r['size']}")
        if "decoded" in r and not all(r["decoded"].values()):
            problems.append(f"{r['run']} {r['key']}: the probe's words decode to other trees or FTZ ({r['decoded']})")
    doc = {"runs": records, "launches": results, "problems": problems, "calls": len(results),
           "elements": sum(r["size"] for r in results), "rank0_words": sum(r["elements"] for r in results),
           "vectors": sum(r["vectors"] for r in results), "specs": sorted({r["spec"] for r in results}),
           "differ": {o: sum(r["differ"][o] for r in results) for o in ORDERS},
           "calls_equal": {o: sum(r["differ"][o] == 0 for r in results) for o in ORDERS}}
    doc["ok"] = (not problems and doc["calls"] > 0 and all(doc["differ"][o] == 0 for o in HELD)
                 and doc["calls_equal"]["chain"] < doc["calls"])
    (outdir / OUT).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"GLM53-MULTIMEM-REPLAY {'GO' if doc['ok'] else 'NO-GO'} calls={doc['calls']} runs={len(records)} "
          f"elements={doc['elements']} vectors={doc['vectors']} differ={json.dumps(doc['differ'])} "
          f"calls_equal={json.dumps(doc['calls_equal'])} problems={problems}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3 TP8's recorded multimem all-reduces on AllReduceMultimem_v1's vectors")
    ap.add_argument("--run", action="append", required=True, help=f"a capture's directory ({X.RECORD} and words/, or out/ holding them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    return 0 if replay(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
