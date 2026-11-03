"""V-EX's honest remedy on a retained Pearl-C pass (`pouw_pearl_c_device`'s), before its sampling coin: every tile's debit
replayed word by word from the pass's own files, the rows that leave a tile or a credited row over its cap given up
(`clear_work.volunteer_words`), and the manifest's records rewritten with them.  CPU only.

  python benchmarks/pouw/pearl_c_vllm/volunteer_pass.py DIR/manifest.json --coins COINS.json [--jobs J]

- **Why after the pass.** A row's debit is its words' (`pearl_c_debit`): it reads the tile's A′ and B̃ codes, which the
  salt's noise forms, so the screen's per-row rules on the activations can't decide it.  The prover knows the salt when it
  forms and commits, and replays every tile as the verifier will, before the challenger opens the sampling coin.
- **What changes** is the records alone: each call's ``voluntary_a`` and each weight's ``voluntary_b`` (V-EX's ⌈m/64⌉ and
  ⌈n/64⌉, `clear_work.check_records`) and the ``records_root`` over them.  The transcript (the tile leaves and tree, A's
  roots, every committed byte) is the pass's.  A pass already given up is left as it is.
- **The replay** runs on `fast_tile`'s batch forms over J processes: each tile opened from the pass's files
  (`verify_run.Opened`), encoded under the epoch's salt as `clear_work.check_drawn` encodes it, and its words' counts taken
  (`fast_tile.word_debits`).

Writes ``volunteer.json`` beside the manifest (the rows over a cap before, the rows given up, what the count caps leave over,
the records' root before and after, and the credit kept) and exits 0 only when no tile is left over a cap.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "clear"))
import fast_tile as FT  # noqa: E402
import verify_run as VR  # noqa: E402
import clear_work as CW  # noqa: E402
from verity.core.protocols.pouw.audit import opened_value  # noqa: E402
from verity.core.protocols.pouw.coins import CoinCommitment  # noqa: E402
from verity.core.protocols.pouw.protocol import Context  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c_work as PW  # noqa: E402

_JOB: dict = {}


def _words(tid: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Tile `tid` of the pass in `_JOB`: (tid, flagged atoms, later +0 promotions), a word each."""
    p = _JOB["pass"]
    ref = p.verifier.refs[tid]
    u = p.epoch.workload.matmuls[ref.matmul]
    opened = p.prover.open(p.epoch, tid)
    h = p.scheme.commitment_hash
    a, b = [opened_value(h, o.value) for o in opened.a], [opened_value(h, o.value) for o in opened.b]
    ctx = Context(p.epoch.salt, u, p.transcript.activation_roots[ref.matmul], p.epoch.weight_roots[u.weight])
    ops = p.scheme.encode(ctx, p.scheme.noise(ctx), a, b, ref.tile)
    real = [x for x, i in enumerate(ref.tile.rows) if i < p.records[ref.matmul].real_rows]
    atoms, zeros = (np.zeros((len(ref.tile.rows), len(ref.tile.cols)), np.uint16) for _ in range(2))
    if real:
        # filler rows earn no credit, so their words' counts stay zero
        got = FT.word_debits([ops.a.codes[x] for x in real], ops.b.codes, p.scheme.device)
        atoms[real], zeros[real] = got
    return tid, atoms, zeros


def words(p: VR.Pass, jobs: int) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Every tile's words' counts, keyed (record position, the tile's place in its matmul's `scheme.tiles`)."""
    if p.scheme.forming != "v1" or p.scheme.device.g is None:
        raise ValueError(f"{p.scheme.name}: the replay takes Pearl-C v1 on a promoted device")
    if max(u.shape.k for u in p.epoch.workload.matmuls) // p.scheme.device.atom.k >= 1 << 16:
        raise ValueError("a word's atoms don't fit its count's 16 bits")
    _JOB.clear()
    _JOB.update({"pass": p})
    undo = FT.install(p.scheme)
    try:
        todo = range(len(p.verifier.refs))
        if jobs > 1 and len(todo) > 1:
            # fork: the workers unpickle `_words` from this module, which callers load from its path
            with multiprocessing.get_context("fork").Pool(min(jobs, len(todo))) as pool:
                got = pool.map(_words, todo, chunksize=max(1, len(todo) // (8 * jobs)))
        else:
            got = [_words(t) for t in todo]
    finally:
        undo()
        _JOB.clear()
    first = _first(p)
    return {(p.verifier.refs[t].matmul, t - first[p.verifier.refs[t].matmul]): (a, z) for t, a, z in got}


def _first(p: VR.Pass) -> dict[int, int]:
    """Each record position's first tile id (`verifier.refs`' numbering, the verdict's)."""
    first: dict[int, int] = {}
    for ref in p.verifier.refs:
        first.setdefault(ref.matmul, ref.id)
    return first


def _over(p: VR.Pass, records, counts) -> list[dict]:
    first = _first(p)
    return [{"record": x["record"], "tile": x["tile"], "tile_id": first[x["record"]] + x["tile"], "debit": float(x["debit"]),
             "cap": float(x["cap"]), "rows": [[i, float(d), float(c)] for i, d, c in x["rows"]]}
            for x in CW.over_words(p.scheme, records, counts)]


def volunteer(path: Path, coins: CoinCommitment, jobs: int) -> dict:
    """Give up the pass's rows over a cap, rewriting its manifest at `path` (records and their root) when any are new."""
    p = VR.load(path, coins)
    counts = words(p, jobs)
    va, vb = CW.volunteer_words(p.scheme, p.records, counts)
    man = json.loads(path.read_text())
    for u in man["matmuls"]:
        u.pop("voluntary_a", None)
        if va.get(u["index"]):
            u["voluntary_a"] = va[u["index"]]
    for w, wt in man["weights"].items():
        wt.pop("voluntary_b", None)
        if vb.get(int(w)):
            wt["voluntary_b"] = vb[int(w)]
    records = VR.records_of(man, p.epoch.workload.matmuls)
    splits = {w: PW.Split(man["weights"][str(w)]["k"]) for w in p.epoch.workload.weights()}
    reason = CW.check_records(p.epoch, splits, records)
    if reason is not None:
        raise ValueError(f"the records given up are refused: {reason}")
    before, man["records_root"] = man["records_root"], PW.records_root(p.epoch.salt, p.epoch.weights_root, splits, records).hex()
    if man["records_root"] != before:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(man, indent=1) + "\n")
        os.replace(tmp, path)
    left = _over(p, records, counts)
    credit = (CW.work(p.scheme, p.records)[1], CW.work(p.scheme, records)[1])
    out = {"manifest": str(path), "tiles": len(counts), "over": _over(p, p.records, counts),
           "voluntary_a": {str(i): rows for i, rows in va.items()}, "voluntary_b": {str(w): rows for w, rows in vb.items()},
           "left": left, "records_root": {"before": before, "after": man["records_root"]},
           "credit_kept": float(credit[1] / credit[0]) if credit[0] else None, "ok": not left}
    out["line"] = (f"{'PASS' if out['ok'] else 'FAIL'}: {len(counts)} tiles replayed, {len(out['over'])} over a cap; gave up "
                   f"{sum(map(len, va.values()))} A rows ({', '.join(f'call {i}: {r}' for i, r in va.items()) or 'none'}) and "
                   f"{sum(map(len, vb.values()))} B rows ({', '.join(f'weight {w}: {len(r)}' for w, r in vb.items()) or 'none'}); "
                   f"{len(left)} tiles left over a cap; credit kept {out['credit_kept']:.6f}; records root "
                   f"{before[:16]} -> {man['records_root'][:16]}; transcript root {man['commitment']['root'][:16]} unchanged")
    (path.parent / "volunteer.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--coins", type=Path, required=True, help="the epoch's published coin commitment (coins.json)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    out = volunteer(a.manifest, CoinCommitment.from_json(json.loads(a.coins.read_text())), a.jobs)
    print(out["line"], flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
