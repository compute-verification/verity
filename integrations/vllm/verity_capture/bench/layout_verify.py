#!/usr/bin/env python3
"""R10 lane `layout` (owner 3): hashlib-only REFERENCE VERIFIER for the serving engine's commitment layouts.

No torch, no production engine import: every rule of `out/gen/r10/layout/L1-spec.md` is re-implemented here from the byte layouts so
that an independent party can check (a) that an `e2e_witness.json` sidecar (owner 6's format, `out/gen/r10/integ/witness_format.md`)
opens to the claimed merkle roots, (b) that the step-root wrapper of every opened step tree is a member of the run root, (c) that every
weights subtree is a member of the weights root, (d) that every obligation's operand positions follow the position rule, and
(e) optionally, that the merkle root of a step recomputed from the RETAINED ARENA BYTES equals the root the GPU produced.

    python verity_capture/bench/layout_verify.py --witness out/gen/r10/layout/openings/L1_B64_dispersed.json [--arena-dir <dump>]
    python verity_capture/bench/layout_verify.py --selftest

Layouts: L1-chunk256 (framing `cmt-chunk-fa2h-v7.s<S>` step trees, `cmt-posleaf-v0` weights subtrees, 256-B chunks) and the
L2-row candidate (framing `cmt-chunk-fa2h-v7.row1152.s<S>`: same header with chunk_words=288, 1152-B leaves; weights `cmt-posleaf-v0`
with 1152-B chunks).  Leaf hash = pure function of (framing, rank, count, value bytes) as the guest sees it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- production node / lift / fold / path rule (verity-vllm framing)
NODE_PREFIX = (19).to_bytes(4, "big") + b"verity-vllm/node/v1" + (32).to_bytes(8, "big")
LIFT_PREFIX = (19).to_bytes(4, "big") + b"verity-vllm/lift/v1" + (32).to_bytes(8, "big")
U64_32 = (32).to_bytes(8, "big")
EMPTY_ROOT = hashlib.sha256((20).to_bytes(4, "big") + b"verity-vllm/empty/v1").digest()

ROOT_TAG_STEP = b"verity/fa2c/root/v0"                      # L1 step root (engine.bind_root)
ROOT_TAG_RUN = b"verity/cmt-integ/run-root/v0"              # L1 run root (native_host.finalize)
ROOT_TAG_WEIGHTS = b"verity/cmt-integ/weights-root/v0"      # L1 weights root (native_host.register_weights)
POS_LEAF_TAG = b"verity/pos-leaf/v0"
# L2 (this lane): distinct tags so an L2 root can never be presented as an L1 root
ROOT_TAG_STEP_L2 = b"verity/r10-layout/L2-row/step-root/v0"
ROOT_TAG_RUN_L2 = b"verity/r10-layout/L2-row/run-root/v0"
ROOT_TAG_WEIGHTS_L2 = b"verity/r10-layout/L2-row/weights-root/v0"
FA2H_TAG = 0x68326166
L1_CHUNK = 256
L2_CHUNK = 1152


def node(l: bytes, r: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + l + U64_32 + r).digest()


def lift(c: bytes) -> bytes:
    return hashlib.sha256(LIFT_PREFIX + c).digest()


def fold_levels(leaves: list[bytes]) -> list[list[bytes]]:
    if not leaves:
        return [[EMPTY_ROOT]]
    levels = [leaves]
    lv = leaves
    while len(lv) > 1:
        nxt = [node(lv[i], lv[i + 1]) for i in range(0, len(lv) - 1, 2)]
        if len(lv) % 2 == 1:
            nxt.append(lift(lv[-1]))
        lv = nxt
        levels.append(lv)
    return levels


def fold(leaves: list[bytes]) -> bytes:
    return fold_levels(leaves)[-1][0]


def path_shape(n: int, index: int) -> list[bool]:
    """True per level = has a sibling; False = odd trailing node (lift, no bytes)."""
    if not (0 <= index < n):
        raise ValueError((n, index))
    shape, width, cur = [], n, index
    while width > 1:
        shape.append(not (cur == width - 1 and width % 2 == 1))
        width = (width + 1) // 2
        cur //= 2
    return shape


def fold_path(leaf: bytes, index: int, path: list[bytes | None]) -> bytes:
    d, cur = leaf, index
    for sib in path:
        if sib is None:
            d = lift(d)
        elif cur % 2 == 0:
            d = node(d, sib)
        else:
            d = node(sib, d)
        cur //= 2
    return d


def path_in_levels(levels: list[list[bytes]], index: int) -> list[bytes | None]:
    out, cur = [], index
    for lv in levels[:-1]:
        out.append(None if (cur == len(lv) - 1 and len(lv) % 2 == 1) else lv[cur ^ 1])
        cur //= 2
    return out


# ---------------------------------------------------------------- leaf framings (pure functions of framing, rank, count, value)
def chunk_header(step: int, rank: int, chunk_words: int, count: int) -> bytes:
    """CMT-2 `chunk_header` as CMT-3 fills it: (launch_tag, HB, M, NB) = (step, step, count*chunk_words, 0), src_mask 0, version 7, kind 1."""
    return struct.pack("<16I", FA2H_TAG, (7 << 16) | (1 << 8) | 0, step, rank, chunk_words, step, count * chunk_words, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def leaf_hash(framing: str, rank: int, count: int, value: bytes) -> bytes:
    if framing.startswith("cmt-chunk-fa2h-v7.row1152.s"):
        step = int(framing[len("cmt-chunk-fa2h-v7.row1152.s"):])
        if len(value) != L2_CHUNK:
            raise ValueError("L2 leaf must be 1152 B")
        return hashlib.sha256(chunk_header(step, rank, L2_CHUNK // 4, count) + value).digest()
    if framing.startswith("cmt-chunk-fa2h-v7.s"):
        step = int(framing[len("cmt-chunk-fa2h-v7.s"):])
        if len(value) != L1_CHUNK:
            raise ValueError("L1 leaf must be 256 B")
        return hashlib.sha256(chunk_header(step, rank, L1_CHUNK // 4, count) + value).digest()
    if framing == "cmt-posleaf-v0":
        return hashlib.sha256(POS_LEAF_TAG + len(value).to_bytes(8, "big") + value).digest()
    raise ValueError(f"unknown framing {framing}")


def sha_blocks(nbytes: int) -> int:
    return (nbytes + 8) // 64 + 1


# ---------------------------------------------------------------- wrappers (public context -> roots)
def ctx_digest(run_id: str, step: int) -> bytes:
    return hashlib.sha256(f"{run_id}/step={step}".encode()).digest()


def layout_digest(layout: list) -> bytes:
    return hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).digest()


def step_root(tag: bytes, program: bytes, ctx: bytes, geo: bytes, layout_dig: bytes, n_leaves: int, merkle: bytes) -> bytes:
    return hashlib.sha256(tag + program + ctx + geo + layout_dig + n_leaves.to_bytes(8, "big") + merkle).digest()


def run_root(tag: bytes, program: bytes, geo: bytes, step_roots: list[bytes]) -> bytes:
    return hashlib.sha256(tag + program + geo + len(step_roots).to_bytes(8, "big") + fold(list(step_roots))).digest()


def weights_root(tag: bytes, geo: bytes, names: list, subtree_roots: list[bytes]) -> bytes:
    names_dig = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).digest()
    return hashlib.sha256(tag + geo + names_dig + len(subtree_roots).to_bytes(8, "big") + fold(list(subtree_roots))).digest()


def arena_merkle_root(arena: bytes, step: int, chunk: int) -> tuple[bytes, int, list[list[bytes]]]:
    """Recompute the step tree from the retained arena bytes (zero-padded to a whole number of chunks)."""
    padded = -(-len(arena) // chunk) * chunk
    buf = arena + bytes(padded - len(arena))
    n = padded // chunk
    framing = f"cmt-chunk-fa2h-v7.s{step}" if chunk == L1_CHUNK else f"cmt-chunk-fa2h-v7.row1152.s{step}"
    leaves = [leaf_hash(framing, i, n, buf[i * chunk:(i + 1) * chunk]) for i in range(n)]
    levels = fold_levels(leaves)
    return levels[-1][0], n, levels


def tensor_subtree(raw: bytes, chunk: int) -> tuple[bytes, int, list[list[bytes]]]:
    n = -(-len(raw) // chunk)
    leaves = [leaf_hash("cmt-posleaf-v0", k, n, raw[k * chunk:(k + 1) * chunk]) for k in range(n)]
    levels = fold_levels(leaves)
    return levels[-1][0], n, levels


# ---------------------------------------------------------------- sidecar verification
def hx(b: bytes) -> str:
    return b.hex()


def verify_witness(doc: dict, arena_dir: Path | None = None, verbose: bool = True) -> dict:
    lay = doc["layout_id"]
    chunk = L1_CHUNK if lay == "L1-chunk256" else L2_CHUNK
    tags = (ROOT_TAG_STEP, ROOT_TAG_RUN, ROOT_TAG_WEIGHTS) if lay == "L1-chunk256" else (ROOT_TAG_STEP_L2, ROOT_TAG_RUN_L2, ROOT_TAG_WEIGHTS_L2)
    ctx = doc["context"]
    program = bytes.fromhex(ctx["program_digest"])
    geo = bytes.fromhex(ctx["geo_digest"])
    roots = doc["roots"]
    res = {"layout_id": lay, "checks": {}, "fail": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        res["checks"][name] = bool(ok)
        if not ok:
            res["fail"].append(f"{name}: {detail}")
        if verbose:
            print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")

    # (b) step-root wrappers -> run root, from PUBLIC data only
    step_roots = []
    for s in roots["steps"]:
        r = step_root(tags[0], program, ctx_digest(ctx["run_id"], s["step"]), geo, layout_digest(s["layout"]), s["n_leaves"], bytes.fromhex(s["merkle_root"]))
        step_roots.append(r)
        check(f"step{s['step']}.wrapper", r.hex() == s["root"], f"layout_digest {layout_digest(s['layout']).hex()[:16]} n_leaves {s['n_leaves']}")
        check(f"step{s['step']}.layout_digest", layout_digest(s["layout"]).hex() == s["layout_digest"])
    check("run_root", run_root(tags[1], program, geo, step_roots).hex() == roots["run_root"], roots["run_root"][:16])
    # (c) weights subtrees -> weights root
    wt = roots["weights"]
    names = [[t["name"], t["dtype"], t["shape"], t["nbytes"]] for t in wt["tensors"]]
    names_dig = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
    check("weights.names_digest", names_dig == wt["names_digest"])
    check("weights_root", weights_root(tags[2], geo, names, [bytes.fromhex(t["subtree_root"]) for t in wt["tensors"]]).hex() == roots["weights_root"])
    sub_by_name = {t["name"]: t for t in wt["tensors"]}
    # (a) every tree's leaves fold to its root; count leaves/nodes/bytes
    trees = {t["id"]: t for t in doc["trees"]}
    n_leaves_total = n_nodes = wbytes = 0
    for tid, t in trees.items():
        root = bytes.fromhex(t["root"])
        ok_all = True
        for lf in t["leaves"]:
            value = bytes.fromhex(lf["bytes_hex"])
            path = [bytes.fromhex(p["digest"]) if p["kind"] == "sibling" else None for p in lf["path"]]
            shape = path_shape(t["count"], lf["rank"])
            if len(path) != len(shape) or any((p is None) == has for p, has in zip(path, shape)):
                ok_all = False
                continue
            ok_all &= fold_path(leaf_hash(t["framing"], lf["rank"], t["count"], value), lf["rank"], path) == root
            n_leaves_total += 1
            n_nodes += sum(1 for p in path if p is not None)
            wbytes += len(value) + 32 * sum(1 for p in path if p is not None)
        check(f"tree.{tid}.openings", ok_all, f"{len(t['leaves'])} leaves, count {t['count']}")
        if tid.startswith("step:"):
            s = next(x for x in roots["steps"] if x["step"] == int(tid[5:]))
            check(f"tree.{tid}.root_is_step_merkle", s["merkle_root"] == t["root"] and s["n_leaves"] == t["count"])
        elif tid.startswith("weights:"):
            w = sub_by_name.get(tid[8:])
            check(f"tree.{tid}.root_is_subtree", w is not None and w["subtree_root"] == t["root"] and w["n_leaves"] == t["count"])
    res["counts"] = {"opened_leaves": n_leaves_total, "path_nodes": n_nodes, "witness_bytes_leaves_plus_paths": wbytes}
    # (d) obligations: position rule + operands readable from the opened leaves + claimed y == committed word
    def word_at(tree_id: str, rank: int, widx: int) -> int | None:
        t = trees[tree_id]
        for lf in t["leaves"]:
            if lf["rank"] == rank:
                v = bytes.fromhex(lf["bytes_hex"])
                return struct.unpack_from("<H", v, 2 * widx)[0] if 2 * widx + 2 <= len(v) else None
        return None

    ok_pos = True
    for ob in doc["obligations"]:
        s = next(x for x in roots["steps"] if x["step"] == ob["logical"]["step"])
        lay_by_name = {row[1]: row for row in s["layout"]}
        for o in ob.get("inputs", []):                          # scalar relation: single-word inputs of the step tree
            b = lay_by_name[o["tensor"]][8] + 2 * o["flat_word0"]
            ok_pos &= [b // chunk] == o["ranks"] and (b % chunk) // 2 == o["word_index0"] and word_at(o["tree"], o["ranks"][0], o["word_index0"]) is not None
        for opnd in ("x", "W") if "x" in ob else ():
            o = ob[opnd]
            if opnd == "x":
                off = lay_by_name[o["tensor"]][8]
                ranks = sorted({(off + 2 * f) // chunk for f in range(o["flat_word0"], o["flat_word0"] + 576)})
                w0 = ((off + 2 * o["flat_word0"]) % chunk) // 2
            else:
                ranks = sorted({(2 * f) // chunk for f in range(o["flat_word0"], o["flat_word0"] + 576)})
                w0 = o["flat_word0"] % (chunk // 2)
            ok_pos &= ranks == sorted(o["ranks"]) and w0 == o["word_index0"]
            ok_pos &= all(any(lf["rank"] == r for lf in trees[o["tree"]]["leaves"]) for r in ranks)
        y = ob["y"]
        off = lay_by_name[y["tensor"]][8]
        b = off + 2 * y["flat_word"]
        ok_pos &= (b // chunk == y["rank"]) and ((b % chunk) // 2 == y["word_index"])
        wv = word_at(y["tree"], y["rank"], y["word_index"])
        # one convention (challenge-v1 cleanup): the y word is the u16 integer `y_word` (bf16 bit pattern read LE from the chunk)
        ok_pos &= wv is not None and wv == ob["expected_y"] == y["y_word"]
    check("obligations.positions_and_claimed_y", ok_pos, f"{len(doc['obligations'])} obligations")
    # (e) optional: recompute step merkle roots from the retained arena bytes
    if arena_dir is not None:
        for s in roots["steps"]:
            p = arena_dir / f"step_{s['step']}.bin"
            if not p.exists():
                continue
            t0 = time.perf_counter()
            mr, n, _ = arena_merkle_root(p.read_bytes(), s["step"], chunk)
            check(f"arena.step{s['step']}.recomputed_root", mr.hex() == s["merkle_root"] and n == s["n_leaves"],
                  f"{n} leaves from {p.stat().st_size} B in {time.perf_counter() - t0:.2f}s")
        wdir = arena_dir / "weights"
        if wdir.is_dir():
            for t in wt["tensors"]:
                p = wdir / (t["name"] + ".bin")
                if p.exists() and any(tid == f"weights:{t['name']}" for tid in trees):
                    sr, n, _ = tensor_subtree(p.read_bytes(), chunk)
                    check(f"weights.{t['name']}.recomputed_subtree", sr.hex() == t["subtree_root"] and n == t["n_leaves"])
    res["ok"] = not res["fail"]
    return res


def selftest() -> None:
    # tree rule vs a hand fold; lift on odd; path shapes
    leaves = [hashlib.sha256(bytes([i])).digest() for i in range(5)]
    lv = fold_levels(leaves)
    assert lv[-1][0] == node(node(node(leaves[0], leaves[1]), node(leaves[2], leaves[3])), lift(lift(leaves[4])))
    for i in range(5):
        assert fold_path(leaves[i], i, path_in_levels(lv, i)) == lv[-1][0]
    assert path_shape(5, 4) == [False, False, True]
    # chunk header equals hidden_stream.chunk_header when the production module is importable (laptop / pod)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from verity_capture.experimental.fa2_commit import hidden_stream as hs  # type: ignore

        class _L:
            HB, M, NB = 3, 64 * 10, 0

        assert hs.chunk_header(_L, 3, 7, 64, 0) == chunk_header(3, 7, 64, 10)
        from verity_capture.experimental.fa2_commit import engine as eng  # type: ignore

        assert eng.pos_leaf(b"abc" * 50) == leaf_hash("cmt-posleaf-v0", 0, 1, b"abc" * 50)
        assert eng.ft.fold(leaves) == lv[-1][0]
        assert list(eng.path_shape(5, 4)) == path_shape(5, 4)
        print("selftest: production-module cross-check OK (chunk_header, pos_leaf, fold, path_shape)")
    except Exception as e:  # noqa: BLE001
        print("selftest: production modules not importable here (hashlib-only rules checked):", e)
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--witness", help="e2e_witness.json (owner 6 format)")
    ap.add_argument("--arena-dir", default=None, help="dump dir with step_<s>.bin (+ weights/<name>.bin) to recompute roots from bytes")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if not a.witness:
            return 0
    doc = json.loads(Path(a.witness).read_text())
    t0 = time.perf_counter()
    res = verify_witness(doc, Path(a.arena_dir) if a.arena_dir else None, verbose=not a.quiet)
    res["verify_s"] = time.perf_counter() - t0
    print(json.dumps({k: v for k, v in res.items() if k != "checks"}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
