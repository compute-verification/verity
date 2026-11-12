"""V-EX's honest remedy on a retained Pearl-C pass (`pouw_pearl_c_device`'s), before its sampling coin: every tile's inert
work replayed word by word from the pass's own files, the rows that leave a tile or a credited row over its inert-work
bound given up (`clear_work.volunteer_words`), and the manifest's records rewritten with them.

  python benchmarks/pouw/pearl_c_vllm/volunteer_pass.py DIR/manifest.json --coins COINS.json [--jobs J]
                                                         [--device CUBIN [--cross-check N]]

- **When it runs.** Exactly while the verdict rejects a run on a sampled tile whose credited inert work, or a credited
  row's, is over its inert-work bound, its rule on such a tile ``reject`` (`served_passes.over_cap_rule`): the
  pipelines ask ``served_passes.py volunteer-needed`` before they serve and refuse to skip the give-up then.  Under
  ``uncredited``, where the verdict credits such a row nothing instead, the give-up is off the prover's path, and this
  module stays for whoever runs it by hand.
- **Why after the pass.** A row's inert work is its words' (`pearl_c_debit`): it reads the tile's A′ and B̃ codes, which the
  salt's noise forms, so the screen's per-row rules on the activations can't decide it.  The prover knows the salt when it
  forms and commits, and replays every tile as the verifier will, before the challenger opens the sampling coin.
- **What changes** is the records alone: each call's ``voluntary_a`` and each weight's ``voluntary_b`` (V-EX's ⌈m/64⌉ and
  ⌈n/64⌉, `clear_work.check_records`) and the ``records_root`` over them.  The transcript (the tile leaves and tree, A's
  roots, every committed byte) is the pass's.  A pass already given up is left as it is.
- **The replay** runs on `fast_tile`'s batch forms over J processes: each tile opened from the pass's files
  (`verify_run.Opened`), encoded under the epoch's salt as `clear_work.check_drawn` encodes it, and its words' counts taken
  (`fast_tile.word_debits`).
- **On the device** (``--device``, an H100 or H200 and pearl_c_sm90's debit_sm90.cubin): each call's A′ (its real rows) and
  each weight's B̃ formed once on the J processes, as `encode` forms them (the scheme's `_side`, on fast_tile's `form_v1`;
  B̃ kept across the passes of one salt), every word's counts taken by `debit.Debit` and cut into the tiles in
  `scheme.tiles`' order, filler rows zero.  A tile with a word the device marks non-finite is replayed (the reference
  decides it), and so is every Nth tile (``--cross-check``, default `CROSS_CHECK`): any difference from the device's
  counts stops the step.

Writes ``volunteer.json`` beside the manifest (the tiles over their inert-work bound before, the rows given up, the tiles
still over it after, the records' root before and after, the credit kept, and on the device its counts' source and
seconds) and exits 0 only when no tile is left over its inert-work bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "verity" / "protocols" / "pouw" / "clear"))
import fast_tile as FT  # noqa: E402
import verify_run as VR  # noqa: E402
import clear_work as CW  # noqa: E402
from verity.protocols.pouw.verifier.audit import opened_value  # noqa: E402
from verity.protocols.pouw.verifier.coins import CoinCommitment  # noqa: E402
from verity.protocols.pouw.verifier.protocol import Context  # noqa: E402
from verity.protocols.pouw.prover.schemes import pearl_c as C  # noqa: E402
from verity.protocols.pouw.prover.schemes import pearl_c_work as PW  # noqa: E402

#: the device's counts are held to the CPU replay's on every Nth tile id
CROSS_CHECK = 1024
#: the rows a forming job takes
FORM_ROWS = 64
_JOB: dict = {}
#: B̃ by (salt, the weight's root): a weight's codes are the same in every pass of one salt
_B_CODES: dict[tuple[bytes, bytes], np.ndarray] = {}
_DEBIT: dict[str, object] = {}


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


def _ctx(p: VR.Pass, pos: int) -> Context:
    u = p.epoch.workload.matmuls[pos]
    return Context(p.epoch.salt, u, p.transcript.activation_roots[pos], p.epoch.weight_roots[u.weight])


def _form(job: tuple[str, int, int, int]) -> tuple[tuple[str, int, int, int], np.ndarray]:
    """Rows lo..hi of A′ (side "a", record position `at`) or of B̃ (side "b", the weight of record position `at`) of the pass
    in `_JOB`, formed as `encode` forms a tile's: (job, codes)."""
    side, at, lo, hi = job
    p = _JOB["pass"]
    u, ctx = p.epoch.workload.matmuls[at], _ctx(p, at)
    k, s, idx = u.shape.k, p.scheme, list(range(lo, hi))
    noise = s.noise(ctx)
    if side == "a":
        data = p.prover.files(p.man["matmuls"][at]["rows"])
        rows = [VR.f32_row(data, k, i) for i in idx]
        e = [C.sample_line(x, 0, 0, 0) for x in s.row_seeds(ctx, rows, idx)] if s.hashing in C.ROW_SEEDED else None
        formed, _ = s._side(rows, idx, 0, noise.seed_a, noise.seed_b, k, e)
    else:
        data = p.prover.files(p.man["weights"][str(u.weight)]["rows"])
        formed, _ = s._side([VR.f32_row(data, k, j) for j in idx], idx, 1, noise.seed_b, noise.seed_b, k)
    return job, np.asarray(formed.codes, np.uint8).reshape(len(idx), k)


def _debit(cubin: Path):
    """The device's `debit.Debit` on `cubin`, one a process."""
    key = str(Path(cubin).resolve())
    if key not in _DEBIT:
        sys.path.insert(0, str(HERE.parent / "pearl_c_sm90"))
        import debit

        _DEBIT[key] = debit.Debit(Path(cubin).read_bytes())
    return _DEBIT[key]


def words_device(p: VR.Pass, jobs: int, cubin: Path, cross_check: int = CROSS_CHECK
                 ) -> tuple[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], dict]:
    """`words` from the device's counts (module docstring), with what it replayed and its seconds."""
    if p.scheme.forming != "v1" or p.scheme.device.g is None:
        raise ValueError(f"{p.scheme.name}: the device's counts take Pearl-C v1 on a promoted device")
    matmuls = p.epoch.workload.matmuls
    if max(u.shape.k for u in matmuls) // p.scheme.device.atom.k >= 1 << 16:
        raise ValueError("a word's atoms don't fit its count's 16 bits")
    t0 = time.perf_counter()
    weight_at = {}
    for pos, u in enumerate(matmuls):
        weight_at.setdefault(u.weight, pos)
    b_key = {w: (p.epoch.salt, p.epoch.weight_roots[w]) for w in weight_at}
    todo = [("b", pos, lo, min(lo + FORM_ROWS, matmuls[pos].shape.n)) for w, pos in weight_at.items() if b_key[w] not in _B_CODES
            for lo in range(0, matmuls[pos].shape.n, FORM_ROWS)]
    todo += [("a", pos, lo, min(lo + FORM_ROWS, p.records[pos].real_rows)) for pos in range(len(matmuls))
             for lo in range(0, p.records[pos].real_rows, FORM_ROWS)]
    _JOB.clear()
    _JOB.update({"pass": p})
    undo = FT.install(p.scheme)
    pool = multiprocessing.get_context("fork").Pool(max(1, jobs)) if jobs > 1 else None
    try:
        got = pool.imap_unordered(_form, todo, chunksize=1) if pool else map(_form, todo)
        codes = {("a", pos): np.zeros((p.records[pos].real_rows, u.shape.k), np.uint8) for pos, u in enumerate(matmuls)}
        for w, pos in weight_at.items():
            if b_key[w] not in _B_CODES:
                codes[("b", pos)] = np.zeros((matmuls[pos].shape.n, matmuls[pos].shape.k), np.uint8)
        for (side, pos, lo, hi), rows in got:
            codes[(side, pos)][lo:hi] = rows
        for w, pos in weight_at.items():
            _B_CODES.setdefault(b_key[w], codes.pop(("b", pos), None))
        t1 = time.perf_counter()
        debit, counts, bad = _debit(cubin), {}, set()
        first = _first(p)
        for pos, u in enumerate(matmuls):
            atoms, zeros, off = (np.zeros((u.shape.m, u.shape.n), np.uint16) for _ in range(3))
            real = p.records[pos].real_rows
            atoms[:real], zeros[:real], off[:real] = debit.words(codes.pop(("a", pos)), _B_CODES[b_key[u.weight]])
            for t, tile in enumerate(p.scheme.tiles(u.shape)):
                r, c = slice(tile.rows[0], tile.rows[-1] + 1), slice(tile.cols[0], tile.cols[-1] + 1)
                counts[(pos, t)] = (atoms[r, c], zeros[r, c])
                if off[r, c].any():
                    bad.add(first[pos] + t)
        t2 = time.perf_counter()
        checked = sorted(bad | set(range(0, len(p.verifier.refs), max(1, cross_check))))
        replayed = pool.map(_words, checked, chunksize=max(1, len(checked) // (8 * jobs))) if pool else map(_words, checked)
        differ = []
        for tid, a, z in replayed:
            key = (p.verifier.refs[tid].matmul, tid - first[p.verifier.refs[tid].matmul])
            if tid not in bad and not (np.array_equal(counts[key][0], a) and np.array_equal(counts[key][1], z)):
                differ.append(tid)
            counts[key] = (a, z)
        if differ:
            raise RuntimeError(f"the device's counts differ from the CPU replay's on {len(differ)} of {len(checked)} tiles "
                               f"replayed (tile ids {differ[:8]}): the device's counts are not the reference's")
    finally:
        if pool:
            pool.close()
            pool.join()
        undo()
        _JOB.clear()
    return counts, {"cubin_sha256": hashlib.sha256(Path(cubin).read_bytes()).hexdigest(), "cross_check": cross_check,
                    "replayed": len(checked), "bad_tiles": len(bad), "form_s": t1 - t0, "device_s": t2 - t1,
                    "replay_s": time.perf_counter() - t2}


def _over(p: VR.Pass, records, counts) -> list[dict]:
    first = _first(p)
    return [{"record": x["record"], "tile": x["tile"], "tile_id": first[x["record"]] + x["tile"], "debit": float(x["debit"]),
             "cap": float(x["cap"]), "rows": [[i, float(d), float(c)] for i, d, c in x["rows"]]}
            for x in CW.over_words(p.scheme, records, counts)]


def volunteer(path: Path, coins: CoinCommitment, jobs: int, device: Path | None = None, cross_check: int = CROSS_CHECK) -> dict:
    """Give up the pass's rows over their inert-work bound, rewriting its manifest at `path` (records and their root)
    when any are new; with `device` (debit_sm90.cubin) the words' counts are the device's (`words_device`)."""
    p = VR.load(path, coins)
    counts, on = words_device(p, jobs, device, cross_check) if device is not None else (words(p, jobs), None)
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
    if on is not None:
        out["device"] = on
    how = (f"counted on the device ({on['replayed']} replayed on the CPU too, {on['bad_tiles']} of them for a non-finite word)"
           if on is not None else "replayed")
    out["line"] = (f"{'PASS' if out['ok'] else 'FAIL'}: {len(counts)} tiles {how}, {len(out['over'])} over a cap; gave up "
                   f"{sum(map(len, va.values()))} A rows ({', '.join(f'call {i}: {r}' for i, r in va.items()) or 'none'}) and "
                   f"{sum(map(len, vb.values()))} B rows ({', '.join(f'weight {w}: {len(r)}' for w, r in vb.items()) or 'none'}); "
                   f"{len(left)} tiles left over a cap; credit kept {out['credit_kept']:.6f}; records root "
                   f"{before[:16]} -> {man['records_root'][:16]}; transcript root {man['commitment']['root'][:16]} unchanged")
    (path.parent / "volunteer.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--coins", type=Path, required=True,
                    help="the epoch's published coin commitment (coins.json, a live session's or an in-process challenger's)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--device", type=Path, help="debit_sm90.cubin: the words' counts on this machine's H100 or H200")
    ap.add_argument("--cross-check", type=int, default=CROSS_CHECK, help="with --device, replay every Nth tile on the CPU too")
    a = ap.parse_args(argv)
    out = volunteer(a.manifest, VR.epoch_coins(a.coins), a.jobs, a.device, a.cross_check)
    print(out["line"], flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
