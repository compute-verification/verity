"""The CPU reference's check of a pass `full.py --receipts --keep KEEP` made on the GPU, after the GPU is released:

- **y**: every unit's receipt from the pass's σ (`p2.Sketcher.receipt`, as serve.py's `unit_receipts` takes it), against
  the GPU's;
- **root**: `receipts.tree(seed, Plan.whole(units), those receipts, the pass's salts)`'s root, against the GPU's;
- **σ**: sampled units' σ from C*'s bytes (`p2.Sketcher.block` on each of the unit's four blocks, from the encoded
  store's segment files, read only), against the GPU's: every unit a tensor's first or last block lies in, each
  segment's first and last unit, and `--sample` more sampled by `--seed`, in a process pool.

A pass whose units aren't the ones the store's tensors lie in (0 .. ⌈e / 4⌉ − 1, e the end of the last tensor's
blocks; the padding past the head is in no receipt) is refused before anything is checked.

    python bw_ref.py --keep KEEP --layout LAYOUT --weights M --encoded ENC --source HEX --commitment HEX --out OUT
                     [--sample 100000] [--seed 12] [--workers N]

Writes `OUT/bw_ref.json` and prints one verdict line; exits 1 if anything differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L

from verity.core.protocols.remote_bandwidth import p2
from verity.core.protocols.remote_bandwidth import receipts as R

PER = p2.UNIT_BLOCKS
_JOB: dict = {}


def _init(job: dict) -> None:
    _JOB.update(job)
    _JOB["sk"] = p2.Sketcher(bytes.fromhex(job["seed"]), R.SKETCH_ROWS)


def _sigmas(units: list[int]) -> list[tuple[int, bytes]]:
    sk, out = _JOB["sk"], []
    for g in units:
        rows = L.segment_rows(Path(_JOB["encoded"]), _JOB["spec"], PER * g, PER)
        out.append((g, sk.total([sk.block(PER * g + b, rows[L.ROW * b:L.ROW * (b + 1)]) for b in range(PER)])))
    return out


def sample_units(lay: L.StoreLayout, units: int, sample: int, seed: int) -> list[int]:
    """The units bw_ref recomputes σ for, sorted: each tensor's first and last block's, each segment's first and last,
    and `sample` drawn."""
    bps = lay.params.blocks_per_segment
    picks = {t.first_block // PER for t in lay.tensors} | {(t.block_range.stop - 1) // PER for t in lay.tensors}
    for s in range(lay.params.segments):
        picks |= {s * bps // PER, ((s + 1) * bps - 1) // PER}
    picks = {g for g in picks if g < units}
    rest = [g for g in random.Random(seed).sample(range(units), min(units, sample + len(picks))) if g not in picks]
    return sorted(picks | set(rest[:sample]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keep", type=Path, required=True, help="full.py --keep's directory")
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--encoded", type=Path, required=True, help="the encoded store's directory, read only")
    ap.add_argument("--source", required=True)
    ap.add_argument("--commitment", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    args = ap.parse_args(argv)
    lay, _ = L.open_store(args.layout, json.loads(args.weights.read_text()), args.encoded, args.source, args.commitment,
                          None)
    rec = json.loads((args.keep / "pass.json").read_text())
    seed, units, root = bytes.fromhex(rec["seed"]), rec["units"], bytes.fromhex(rec["root"])
    store_units = lay.params.blocks_per_segment * lay.params.segments // PER
    tensor_units = -(-max(t.block_range.stop for t in lay.tensors) // PER)
    if units != tensor_units:
        raise SystemExit(f"the pass covers {units} units, not the {tensor_units} the store's tensors lie in")
    if R.seed(bytes.fromhex(rec["challenge"]), rec["pass"], bytes.fromhex(rec["previous"])) != seed:
        raise SystemExit("the pass's seed is not receipts.seed(challenge, pass, previous)")
    sigma = (args.keep / "sigma.bin").read_bytes()
    y = (args.keep / "y.bin").read_bytes()
    salts_raw = (args.keep / "salts.bin").read_bytes()
    S, Y = 4 * R.SKETCH_ROWS, R.DIGEST_BYTES
    if len(sigma) != S * units or len(y) != Y * units or len(salts_raw) != R.SALT_BYTES * R.leaves(units):
        raise SystemExit("the kept pass's files are not its units' σ, y and salts")
    out: dict = {"units": units, "store_units": store_units, "seed": rec["seed"], "root": rec["root"],
                 "files_sha512": {"sigma": hashlib.sha512(sigma).hexdigest() == rec["sigma_sha512"],
                                  "y": hashlib.sha512(y).hexdigest() == rec["y_sha512"],
                                  "salts": hashlib.sha512(salts_raw).hexdigest() == rec["salts_sha512"]}}
    t0 = time.perf_counter()
    sk = p2.Sketcher(seed, R.SKETCH_ROWS)
    ys = [sk.receipt(g, sigma[S * g:S * (g + 1)]) for g in range(units)]
    bad_y = [g for g in range(units) if ys[g] != y[Y * g:Y * (g + 1)]] if b"".join(ys) != y else []
    out["y"] = {"units": units, "mismatches": len(bad_y), "first": bad_y[:8], "s": round(time.perf_counter() - t0, 1)}
    t0 = time.perf_counter()
    salts = [salts_raw[R.SALT_BYTES * c:R.SALT_BYTES * (c + 1)] for c in range(R.leaves(units))]
    want = R.tree(seed, R.Plan.whole(units), ys, salts).commitment.root
    out["root_cpu"] = {"root": want.hex(), "match": want == root, "s": round(time.perf_counter() - t0, 1)}
    del ys, salts
    t0 = time.perf_counter()
    picks = sample_units(lay, units, args.sample, args.seed)
    job = {"seed": rec["seed"], "encoded": str(args.encoded), "spec": lay.spec}
    step = max(1, len(picks) // (8 * args.workers))
    bad_s: list[int] = []
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(job,)) as pool:
        for part in pool.imap(_sigmas, [picks[i:i + step] for i in range(0, len(picks), step)]):
            bad_s += [g for g, s in part if s != sigma[S * g:S * (g + 1)]]
    out["sigma"] = {"units": len(picks), "mismatches": len(bad_s), "first": bad_s[:8], "s": round(time.perf_counter() - t0, 1)}
    out["ok"] = bool(all(out["files_sha512"].values()) and not bad_y and out["root_cpu"]["match"] and not bad_s)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bw_ref.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    print(f"bw_ref {'PASS' if out['ok'] else 'FAIL'}: the tensors' {units} units of the store's {store_units} "
          f"({store_units - units} padding units in no receipt); y {units - len(bad_y)}/{units} from the GPU's sigma, root "
          f"{'match' if out['root_cpu']['match'] else 'DIFFERS'} ({rec['root'][:16]}...), sigma "
          f"{len(picks) - len(bad_s)}/{len(picks)} sampled units from C*")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
