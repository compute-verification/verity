"""J10-R's receipts-only units (h1u.cu's kh_seeds_j10r and kh_units_multi_j10r) on the host: the call's seed data and every
unit's receipt, bit for bit the receipt of J10R's `h1tile.j10r_reference` (`PouwH1Tile_v5` / `_v6`, at
559bc0aa69), vectorized as served_h1/receipts.py's `tile_receipts` over each unit's J10-R noise.

A call is A [T, K] (its slot: T rows, the rows past M zero) and B [N, K]; unit u = ri (N / 32) + cj is at position base + u,
and `base`, the call's first unit position, is J10-R's ``call``. seed_c = seeds.call_seed(salt, base, roots); unit (ri, cj)'s
noise is rows 32 ri .. and columns 32 cj .. of seed_c's stream at call scale (`h1tile.j10r_blocks(T, K, (32, 32), ri, cj)`:
row i from the unit's first row tile plus i RB, column j from its first column tile plus j CB), its fold keys σ_R's at
base + u. The call's stream is hashed once and each unit gathers its blocks from it.

On a tree without J10R's Definition (main before it lands) `call_seed`, `blocks` and `counters` are restated from it
(`CALL_SEED_TAG` under `seeds._tagged`, `Layout(T, 32, K).row / col`); the anchor, run on J10R's tree, checks the
restatements against the Definition and these receipts against `h1tile.j10r_reference`, word for word:

    python j10r_units.py anchor --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "served_h1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import receipts as RC

from verity.ml.protocols.pouw.circuit import h1tile as HT
from verity.ml.protocols.pouw.circuit import seeds as SD

CALL_SEED_TAG = b"verity/pouw/call-seed/v1".ljust(24, b"\0")
TILE = (RC.R, RC.C_)
#: whether this tree has J10R's Definition (else the restatements below stand in for it)
DEFINITION = all(hasattr(m, f) for m, f in ((SD, "call_seed"), (HT, "j10r_blocks"), (HT, "j10r_unit_blocks"),
                                            (HT, "j10r_reference")))


def _call_seed_local(salt: bytes, call: int, root_a: bytes, root_b: bytes) -> bytes:
    return SD._tagged(CALL_SEED_TAG, salt, call, root_a, root_b)


def _blocks_local(t: int, k: int, ri: int, cj: int) -> tuple[int, int]:
    lay = HT.Layout(t, TILE[1], k)
    return lay.row(TILE[0] * ri), lay.col(TILE[1] * cj)


def _counters_local(k: int, blk: tuple[int, int]) -> list[int]:
    lay = RC.layout(k)
    return ([blk[0] + i * lay.rb + t for i in range(lay.r) for t in range(lay.rb)]
            + [blk[1] + j * lay.cb + t for j in range(lay.c) for t in range(lay.cb)])


def call_seed(salt: bytes, call: int, root_a: bytes, root_b: bytes) -> bytes:
    return SD.call_seed(salt, call, root_a, root_b) if DEFINITION else _call_seed_local(salt, call, root_a, root_b)


def blocks(t: int, k: int, ri: int, cj: int) -> tuple[int, int]:
    """Unit (ri, cj)'s first row tile and first column tile in its call's stream, for a call of t rows."""
    return HT.j10r_blocks(t, k, TILE, ri, cj) if DEFINITION else _blocks_local(t, k, ri, cj)


def counters(k: int, blk: tuple[int, int]) -> list[int]:
    """The counters of seed_c's blocks a unit reads, its rows' then its columns', in `tile_receipts`' noise order."""
    return HT.j10r_unit_blocks(RC.layout(k), blk) if DEFINITION else _counters_local(k, blk)


def seed_words(salt: bytes, root_a: bytes, root_b: bytes) -> list[int]:
    """kh_seeds_j10r's data: CALL_SEED_TAG, RECEIPT_TAG, salt, root_a, root_b as big-endian 64-bit words."""
    b = CALL_SEED_TAG + SD.RECEIPT_TAG + salt + root_a + root_b
    return [int.from_bytes(b[i:i + 8], "big") for i in range(0, len(b), 8)]


def _kernel_seed(data: list[int], kind: int, pos: int) -> bytes:
    """h1u.cu's h1_seeds_c message from kh_seeds_j10r's `data` (`seed_words`): kind 0 seed_c, 1 σ_R, at position pos."""
    words = [pos] + [(data[3 * kind + w] if w < 3 else data[3 + w]) ^ pos for w in range(27)]
    return hashlib.sha512(b"".join(w.to_bytes(8, "big") for w in words)).digest()


def lean_vectors() -> dict:
    """J10R's `J10RChecks.lean` theorems (the Lean kernel's own evaluation of J10R.lean) against this module's seeds:
    seed_c by the restatement, by `seeds.call_seed` and by h1_seeds_c's message; σ_R by `seeds.receipt_seed` and by
    h1_seeds_c's; and a block of seed_c's stream by `_stream`."""
    path = Path(SD.__file__).resolve().parents[5] / "Security" / "Proofs" / "Accounting" / "TS" / "J10RChecks.lean"
    lean = {m[1]: m[2].strip() for m in re.finditer(r"^theorem (\w+_vec) :.*= (\[.*\]|\(.*\)) := by decide \+kernel$",
                                                    path.read_text(), re.MULTILINE)}
    salt = bytes(range(64))
    ra, rb = bytes((7 * i + 3) % 256 for i in range(64)), bytes((5 * i + 1) % 256 for i in range(64))
    data, sc = seed_words(salt, ra, rb), _call_seed_local(salt, 40, ra, rb)
    got = {"callSeed_vec": {"restated": sc, "seeds.call_seed": SD.call_seed(salt, 40, ra, rb),
                            "h1_seeds_c": _kernel_seed(data, 0, 40)},
           "receiptSeed_vec": {"seeds.receipt_seed": SD.receipt_seed(salt, 46, ra, rb),
                               "h1_seeds_c": _kernel_seed(data, 1, 46)},
           "noiseBlock_vec": {"_stream": _stream(sc, np.array([24]))[0].astype(">u4").tobytes()}}
    doc: dict = {"file": str(path.relative_to(path.parents[4])), "bad": 0}
    for name, by in got.items():
        want = lean.get(name)
        row = {how: "[" + ", ".join(map(str, v[:8])) + "]" == want for how, v in by.items()}
        doc["bad"] += sum(not ok for ok in row.values())
        doc[name] = {"lean": want, **row}
    return doc


def _stream(seed: bytes, ts: np.ndarray) -> np.ndarray:
    head = SD.NOISE_TAG + bytes(seed)
    out = b"".join(hashlib.sha512(head + int(t).to_bytes(8, "big")).digest() for t in ts)
    return np.frombuffer(out, dtype=">u4").astype(np.int64).reshape(len(ts), 16)


def call_receipts(a: torch.Tensor, b: torch.Tensor, salt: bytes, base: int, root_a: bytes, root_b: bytes,
                  only: list[int] | None = None, chunk: int | None = None) -> np.ndarray:
    """The call's J10-R receipts (units, 2 sets) uint32 for A [T, K] and B [N, K] s8 (unit u at position base + u, `only`
    those units in that order): each unit's noise gathered from seed_c's call-scale stream, hashed once a block."""
    t, k = a.shape
    n = b.shape[0]
    ncb = n // TILE[1]
    us = [(u // ncb, u % ncb) for u in (range((t // TILE[0]) * ncb) if only is None else only)]
    pick = [ri * ncb + cj for ri, cj in us]
    lay = RC.layout(k)
    cs = np.array([counters(k, blocks(t, k, ri, cj)) for ri, cj in us], dtype=np.int64)
    need = np.unique(cs)
    stream = _stream(call_seed(salt, base, root_a, root_b), need)
    dev = a.device
    chunk = chunk or max(16, (1 << 20) // k)
    ar = torch.arange(TILE[0], device=dev)
    out = np.empty((len(us), 2 * lay.sets), dtype=np.uint32)
    for at in range(0, len(us), chunk):
        part = slice(at, at + chunk)
        noise = torch.from_numpy(stream[np.searchsorted(need, cs[part])]).to(dev)
        sigmas = [SD.receipt_seed(salt, base + u, root_a, root_b) for u in pick[part]]
        keys = RC.noise_blocks_cpu(sigmas, RC.LANES).to(dev)
        ri = torch.tensor([x[0] for x in us[part]], device=dev)
        cj = torch.tensor([x[1] for x in us[part]], device=dev)
        at_ = a[ri[:, None] * TILE[0] + ar].to(torch.int64)
        bt_ = b[cj[:, None] * TILE[1] + ar].to(torch.int64)
        out[part] = RC.tile_receipts(at_, bt_, noise, keys).cpu().numpy().astype(np.uint32)
    return out


def anchor(out: Path, seed: int = 0x10A) -> dict:
    """On J10R's tree: the restated seed and blocks against the Definition's, and `call_receipts` against
    `h1tile.j10r_reference`'s receipt on every word, at small calls (both row tiles, several column tiles) and at K = 4096
    and 14336 (m = 2 and 3) at the served slot T = 64; and the seeds against J10RChecks.lean's vectors (`lean_vectors`)."""
    if not DEFINITION:
        raise SystemExit("j10r_units anchor: this tree has no J10-R Definition (h1tile.j10r_reference); run it on J10R's")
    rng = random.Random(seed)
    cases = [(64, 96, 512, None), (64, 64, 1024, None), (64, 96, 4096, [0, 2, 4]), (64, 64, 14336, [1, 3])]
    doc: dict = {"schema": "j10r-units/anchor/v1", "cases": [], "seed_bad": 0, "blocks_bad": 0}
    for t, n, k, only in cases:
        salt, ra, rb = rng.randbytes(64), rng.randbytes(64), rng.randbytes(64)
        base = rng.getrandbits(40)
        g = np.random.default_rng(rng.getrandbits(32))
        a = torch.from_numpy(g.integers(-128, 128, size=(t, k), dtype=np.int8))
        b = torch.from_numpy(g.integers(-128, 128, size=(n, k), dtype=np.int8))
        units = list(range((t // 32) * (n // 32))) if only is None else only
        doc["seed_bad"] += int(_call_seed_local(salt, base, ra, rb) != SD.call_seed(salt, base, ra, rb))
        ncb = n // 32
        for u in range((t // 32) * ncb):
            ri, cj = divmod(u, ncb)
            blk = HT.j10r_blocks(t, k, TILE, ri, cj)
            doc["blocks_bad"] += int(_blocks_local(t, k, ri, cj) != blk)
            doc["blocks_bad"] += int(_counters_local(k, blk) != HT.j10r_unit_blocks(RC.layout(k), blk))
        t0 = time.time()
        fast = call_receipts(a, b, salt, base, ra, rb, only=units)
        fast_s = time.time() - t0
        t0 = time.time()
        bad = 0
        for i, u in enumerate(units):
            ri, cj = divmod(u, ncb)
            at_ = (a[32 * ri:32 * ri + 32].to(torch.int64) & 0xFF).tolist()
            bt_ = (b[32 * cj:32 * cj + 32].to(torch.int64) & 0xFF).tolist()
            _, rec = HT.j10r_reference(at_, bt_, salt, base + u, (ra, rb), base, HT.j10r_blocks(t, k, TILE, ri, cj))
            bad += int((np.array(rec, dtype=np.uint32) != fast[i]).sum())
        c = {"t": t, "n": n, "k": k, "units": units, "words": int(fast.shape[1]), "bad_words": bad,
             "fast_s": round(fast_s, 2), "definition_s": round(time.time() - t0, 2)}
        doc["cases"].append(c)
        print(json.dumps(c), flush=True)
    doc["lean_vectors"] = lean_vectors()
    print(json.dumps(doc["lean_vectors"]), flush=True)
    doc["ok"] = (doc["seed_bad"] == 0 and doc["blocks_bad"] == 0 and all(c["bad_words"] == 0 for c in doc["cases"])
                 and doc["lean_vectors"]["bad"] == 0)
    out.mkdir(parents=True, exist_ok=True)
    (out / "j10r-anchor.json").write_text(json.dumps(doc, indent=1))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["anchor"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc = anchor(Path(args.out))
    print(json.dumps({k: v for k, v in doc.items() if k != "cases"}), flush=True)
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
