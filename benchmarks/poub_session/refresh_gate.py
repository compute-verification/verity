"""The refresh path's gate on a card, against vectors the Python reference pins with no card: a store of CHUNKS chunks
of CHUNK blocks from fixed payloads, its chunk 1 re-keyed out of place to index N under an append's coin
(`DeviceStore.rekey`), then passes over refresh plans whose blocks' store indices are not their positions
(`rs_pass_at`), one with its padding, a chained one after the append (`rs_chain_head`), and openings across the
pieces (`rs_subtrees_at`). Every device buffer is 0xA5 before it is written (`DeviceStore.poison`); a pass that
launches nothing must miss the root, and the re-key writes its range of the spare area and nothing else.

  python refresh_gate.py --reference refresh_gate.json            # no card: the pinned vectors
  python refresh_gate.py --lib session.so --kv kv.so [--out FILE]  # on the card, against them

Prints one line a check; exits 1 naming the first that fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "verity/protocols/resource_accounting/memory/poub/prover/poub_session"))

from verity.protocols.resource_accounting.memory.poub.prover import audit_prover  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, receipts as rb  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.store import PAYLOAD_LABEL, store_salt  # noqa: E402

VECTORS = HERE / "refresh_gate.json"
TAG = b"verity/poub/refresh-gate/v1"
CHUNK, CHUNKS, ROWS = 256, 4, 16
N = CHUNK * CHUNKS
ORIGIN, FIRST = CHUNK, N
BLOCK = codec.BLOCK_BYTES


def _h(label: str, n: int = 64) -> bytes:
    return hashlib.sha512(TAG + label.encode()).digest()[:n]


def payload(j: int) -> bytes:
    """Block j's payload: SHA-512 in counter mode, each field word's top bit clear, as the store's filler is."""
    raw = bytearray(b"".join(hashlib.sha512(TAG + b"payload" + j.to_bytes(8, "big") + c.to_bytes(4, "big")).digest()
                             for c in range(BLOCK // 64)))
    for w in range(127, BLOCK, 128):
        raw[w] &= 0x7F
    return bytes(raw)


def setting() -> dict:
    """Everything both sides derive the same way."""
    payloads = [payload(j) for j in range(N)]
    node, salt_key, payload_key = _h("node"), _h("salt", 32), _h("payload-key", 32)
    salts = [store_salt(payload_key, PAYLOAD_LABEL, j) for j in range(N)]
    tree = audit.payload_tree(node, payloads, salts)
    reg = audit.Registration(node, N, tree.commitment.root, _h("codec", codec.SEED_BYTES), codec.DEFAULT, rb.DEFAULT,
                             False, rb.LAYOUT, ROWS)
    rekey = audit_prover.rekey_root(reg.digest, FIRST, CHUNK, ("registered", ORIGIN))
    plans = {"pieces": rb.Plan(((0, 2 * CHUNK), (3 * CHUNK, 2 * CHUNK)), N + CHUNK),
             "padded": rb.Plan(((0, CHUNK), (2 * CHUNK, 2 * CHUNK)), N + CHUNK), "whole": rb.Plan.whole(N)}
    prev = _h("previous")
    return {"payloads": payloads, "tree": tree, "reg": reg, "salt_key": salt_key, "payload_key": payload_key,
            "coin": _h("coin", codec.COIN_BYTES), "seed": _h("seed"), "challenge": _h("challenge", rb.CHALLENGE_BYTES),
            "previous": prev, "head": audit.append_head(None, FIRST, CHUNK, rekey), "plans": plans}


def positions() -> list[int]:
    """Openings at every piece's edges and inside each."""
    c = CHUNK
    return sorted({0, 1, c + 17, 2 * c - 1, 2 * c, 2 * c + 33, 3 * c - 1, 3 * c, 3 * c + 100, 4 * c - 1})


def opening_digest(fields: list[tuple]) -> str:
    h = hashlib.sha512()
    for f in fields:
        blk, leaf, salt, path, psalt, ppath, *rest = f
        h.update(blk + leaf + salt + b"".join(path) + psalt + b"".join(ppath))
        if rest:
            d, origin = rest
            h.update(d + (-1 if origin is None else origin).to_bytes(8, "big", signed=True))
    return h.hexdigest()


def _encode(args) -> bytes:
    m, j, seed, bound, coin = args
    k = codec.key(seed)
    return codec.encode(m, j, k, bound=bound, coin=coin) if bound else codec.encode(m, j, k)


def reference(out: Path) -> dict:
    """The vectors from the codec, the receipter and the receipt tree in Python, no card."""
    s = setting()
    reg, payloads = s["reg"], s["payloads"]
    jobs = [(payloads[j], j, reg.codec_seed, False, None) for j in range(N)]
    jobs += [(payloads[ORIGIN + u], FIRST + u, reg.codec_seed, True, s["coin"]) for u in range(CHUNK)]
    with ProcessPoolExecutor() as pool:
        blocks = list(pool.map(_encode, jobs, chunksize=16))

    def tree(seed: bytes, plan: rb.Plan):
        r = rb.receipter(reg.receipt, seed, reg.rows)
        ys = [r(j, blocks[j]) for j in plan]
        salts = [store_salt(s["salt_key"], seed, c) for c in range(rb.leaves(plan.length))]
        return ys, rb.tree(seed, plan, ys, salts)

    v = {"chunk": CHUNK, "chunks": CHUNKS, "encoded": hashlib.sha512(b"".join(blocks[:N])).hexdigest(),
         "rekeyed": hashlib.sha512(b"".join(blocks[N:])).hexdigest(), "payload_root": s["tree"].commitment.root.hex()}
    for name, plan in s["plans"].items():
        ys, t = tree(s["seed"], plan)
        v[f"root_{name}"] = t.commitment.root.hex()
        if name == "pieces":
            fields = []
            for p in positions():
                c, j = p // rb.PER_LEAF, plan.block(p)
                o = j if j < N else ORIGIN + j - FIRST
                fields.append((blocks[j], b"".join(ys[rb.PER_LEAF * c:rb.PER_LEAF * (c + 1)]),
                               store_salt(s["salt_key"], s["seed"], c), t.open(c).path,
                               store_salt(s["payload_key"], PAYLOAD_LABEL, o), s["tree"].open(o).path,
                               b"" if j < N else codec.digest(payloads[o]), None if j < N else o))
            v["openings"] = opening_digest(fields)
    seed0 = rb.seed(s["challenge"], 0, audit.chained(s["previous"], s["head"]))
    v["chained_seed"] = seed0.hex()
    v["root_chained"] = tree(seed0, s["plans"]["pieces"])[1].commitment.root.hex()
    out.write_text(json.dumps(v, indent=1, sort_keys=True) + "\n")
    return v


def card(lib_path: str, kv_path: str, vectors: dict) -> dict:
    """The same store on the card, every check against `vectors`."""
    import torch

    import device as dv

    s = setting()
    reg = s["reg"]
    out: dict = {}

    def check(name: str, ok: bool) -> None:
        out[name] = bool(ok)
        print(f"{name}: {'pass' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit(f"refresh gate: {name} failed")

    lib = dv.Lib(lib_path, torch)
    copy = tempfile.NamedTemporaryFile(suffix=".so")
    shutil.copyfile(lib_path, copy.name)
    lib_open, kv = dv.Lib(copy.name, torch), dv.Kv(kv_path, torch)
    raw = b"".join(s["payloads"])
    hbm = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to("cuda")
    store = dv.DeviceStore(torch, lib, lib_open, kv, N, s["salt_key"], 0, chunk=CHUNK, hbm=[hbm], rows=ROWS)
    store.payload_key = s["payload_key"]
    check("payload_root", store.payload_tree(reg.node) == bytes.fromhex(vectors["payload_root"]))
    store.encode(reg.key)
    registered = b"".join(store.blocks_at(0, N))
    check("encoded", hashlib.sha512(registered).hexdigest() == vectors["encoded"])
    store.refresh_area(2 * CHUNK)
    store.ledger = ledger = dv.Ledger(reg, dv.payload_digests(raw), store.landed)
    a = ledger.rekey(("registered", ORIGIN), CHUNK, 0)
    store.spare.fill_(0xA5)
    store.poison = True
    store.rekey(ORIGIN, a.first, CHUNK, dv.bound_offsets(reg.key, a.first, a.digests, s["coin"]))
    store.sync()
    check("rekey_landed", store.landed(a.first))
    spare = store.spare.cpu().numpy().tobytes()
    check("rekeyed", hashlib.sha512(spare[:CHUNK * BLOCK]).hexdigest() == vectors["rekeyed"])
    check("rekey_writes_its_range_only", spare[CHUNK * BLOCK:] == b"\xa5" * (len(spare) - CHUNK * BLOCK))
    check("rekey_out_of_place", b"".join(store.blocks_at(0, N)) == registered)
    handles = {}
    for name, plan in s["plans"].items():
        root, handles[name] = store.wait(store.launch(0, None, seed=s["seed"], plan=plan))
        check(f"root_{name}", root == bytes.fromhex(vectors[f"root_{name}"]))
    kept = dv.DeviceStore.ready(handles["pieces"])[2]
    rows = [x[:store.kept_levels(q.n)[store.hc + 1]].view(torch.uint8) for x, q in zip(kept, handles["pieces"].layout[1])]
    check("kept_levels_written", not any(bool((r == 0xA5).all(dim=-1).any()) for r in rows))
    control, _ = store.wait(store.launch(0, None, seed=s["seed"], plan=s["plans"]["pieces"], write=False))
    check("no_write_control_refused", control != bytes.fromhex(vectors["root_pieces"]))
    fields = store.open_window([(handles["pieces"], positions())])[0]
    check("openings", opening_digest(fields) == vectors["openings"])
    store.start_chain(s["challenge"], s["previous"])
    root, h = store.wait(store.launch(0, None, plan=s["plans"]["pieces"], appends=s["head"]))
    check("chained_seed", dv.DeviceStore.ready(h)[0] == bytes.fromhex(vectors["chained_seed"]))
    check("root_chained", root == bytes.fromhex(vectors["root_chained"]))
    store.close()
    out["ok"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", type=Path, help="write the pinned vectors here, no card")
    ap.add_argument("--lib", help="session.so")
    ap.add_argument("--kv", help="kv.so at p1024/k3/masked")
    ap.add_argument("--vectors", type=Path, default=VECTORS)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.reference:
        reference(a.reference)
        return 0
    if not a.lib or not a.kv:
        ap.error("the card's gate takes --lib and --kv")
    got = card(a.lib, a.kv, json.loads(a.vectors.read_text()))
    if a.out:
        a.out.write_text(json.dumps(got, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
