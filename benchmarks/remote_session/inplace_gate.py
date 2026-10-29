"""The in-place pass's gate and timing on one GPU (`inplace.py`; `inplace.sh` builds session.cu and runs it).

The gate, before anything is timed, is over a store registered as vLLM's binder registers one (`verity_vllm.protocols.remote_bandwidth`'s
`Store`, `TorchBacking`, `PagedBacking` and `page_layout`): weight tensors (bf16 with an unregistered tail, int8 held
transposed, a view 48 bytes into a fused tensor, a plain bf16 matrix) and a KV pool of three layers, two in
FlashAttention's layout (2, pages, slots, heads, dim) sharing one storage and one page-major (pages, 2, ...). Each
check is against the driver's reference over each chunk as the binder reads it (`backing.read`):

    plans      every receipt and the root (`receipts.tree`, salts `device.store_salt`) of the whole store, of a mixed
               plan of weight runs and live pages, of the same chunks out of store order, and of one- and two-leaf
               plans; the fused pass's kept levels equal to the tree hashed after its receipts (`unfused_kept`), at the
               gate's h0 and at h0 = 0 and FUSED_H0
    openings   every plan's (`receipts.verify_leaf`, each chunk as stored), paths through kept levels and subtrees
    appended   chunks appended at the store's end at their slots' bytes, a page's and single chunks (`InPlace.append`)
    written    a pass after weights and KV are written in place, nothing re-registered
    contiguous the mapped receipts of a weight tensor equal to `rs_sketch16_receipts`'s over its bytes
    refusals   a misaligned region, a gap, a plan past the store, a plane of part chunks, a CPU tensor, pages past
               their storage

Timing is at Llama-3.1-8B's weights as vLLM holds them in bf16 (the embedding; per layer qkv, o, gate_up, down and two
norms; the final norm and lm_head: 195 tensors, 16.06 GB), and then with a KV pool beside them (`--pages` pages a layer,
FlashAttention's layout): the read alone through the map, the receipts alone and the whole pass (map, constants and
signs, receipts and the tree behind them; at the weights also the constants alone and the fused kernels alone,
`InPlace.fused`), beside the pass with its tree hashed after its receipts (`unfused_pass`)
and the read and receipts of the contiguous kernels over a copy of the weights in store order. The in-place root is
checked equal to the unfused pass's and to the contiguous one's over the copy, and sampled receipts to the
reference's. Writes OUT/inplace.json.

    python inplace_gate.py --lib session.so --out OUT [--reps 10 --pages 2048]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "integrations" / "vllm"))

from device import BLOCK, PER_LEAF, Lib, digests, rb, store_salt  # noqa: E402
from inplace import FUSED_H0, InPlace, Region, level_at  # noqa: E402
from verity_vllm.protocols.remote_bandwidth.binder import PagedBacking, TorchBacking, page_layout, storage_bytes  # noqa: E402
from verity_vllm.protocols.remote_bandwidth.interface import IDENTITY, PageLayout  # noqa: E402
from verity_vllm.protocols.remote_bandwidth.registry import RemoteBandwidthReceipt, Store  # noqa: E402

ROWS = rb.SKETCH_ROWS


def register(torch, weights: list, kv: list, pages: int, slots: int):
    """A binder's store over `weights` ((name, tensor)) and the KV layers `kv`: the store and each region's backing."""
    store, backs, receipt = Store(), [], RemoteBandwidthReceipt()
    for name, t in weights:
        b = TorchBacking.of_tensor(t, BLOCK)
        store.register(name, "hbm", "weights", b, block_bytes=BLOCK, codec=IDENTITY, receipt=receipt,
                       source=f"param:{name}", encode=False)
        backs.append(b)
    if kv:
        layers, lay = [], None
        for t in kv:
            phys = page_layout(t, pages, slots)
            layers.append(PagedBacking(storage_bytes(t), t.storage_offset() * t.element_size(), phys))
            lay = PageLayout(pages, slots, phys.slot_bytes, phys.planes, slot_major=phys.slot_major)
        store.register_kv_pool("kv", layers, lay, block_bytes=BLOCK, codec=IDENTITY, receipt=receipt, encode=False)
        backs += layers
    return store, backs


def held(store, backs) -> list[bytes]:
    """Every store chunk as the binder reads it."""
    out = []
    for info, b in zip(store.layout().regions, backs):
        raw = b.read(0, info.blocks * BLOCK)
        out += [raw[k * BLOCK:(k + 1) * BLOCK] for k in range(info.blocks)]
    return out


def reference(seed: bytes, plan: rb.Plan, blocks: list[bytes], key: bytes):
    r = rb.receipter(rb.SKETCH_SHA512, seed, ROWS)
    ys = [r(j, blocks[j]) for j in plan]
    salts = [store_salt(key, seed, c) for c in range(rb.leaves(plan.length))]
    return ys, rb.tree(seed, plan, ys, salts).commitment.root


def refused(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def unfused_kept(ip: InPlace, ys, handle) -> bool:
    """Whether a pass's kept levels equal those of the tree hashed after its receipts ys (`InPlace.tree`: reader.cu's
    leaves, padding and levels), under the pass's constants as set."""
    width = 1 << handle.H
    nodes = ip.tree(ys, 0, rb.leaves(handle.plan.length), width)
    return bool(ip.torch.equal(nodes[level_at(width, 0, handle.h0):], handle.kept))


def unfused_pass(ip: InPlace, seed: bytes, plan: rb.Plan) -> bytes:
    """A pass with its tree hashed after its receipts (`InPlace.receipts`, then `InPlace.tree`): its root."""
    n, leaves = plan.length, rb.leaves(plan.length)
    addrs, js = ip.map(plan)
    nodes = ip.tree(ip.receipts(addrs, js, ip.set_pass(seed, plan), 0, n), 0, leaves, 1 << (leaves - 1).bit_length())
    return digests(nodes[-1:].cpu())[0]


def gate(torch, lib: Lib, rng, log) -> dict:
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(1 << 62)))
    pages, slots, heads, dim = 40, 16, 4, 64
    w0 = torch.empty((37, 1000), dtype=torch.bfloat16, device=dev)
    w1 = torch.empty((8192, 24), dtype=torch.int8, device=dev).t()
    fused = torch.empty(10 * BLOCK + 48, dtype=torch.uint8, device=dev)
    w3 = torch.empty((128, 4096), dtype=torch.bfloat16, device=dev)
    per = 2 * pages * slots * heads * dim
    shared = torch.empty(2 * per + 2048, dtype=torch.bfloat16, device=dev)
    kv = [shared[:per].view(2, pages, slots, heads, dim), shared[per + 2048:].view(2, pages, slots, heads, dim),
          torch.empty((pages, 2, slots, heads, dim), dtype=torch.bfloat16, device=dev)]
    for t in (w0, w1, fused, w3, shared, kv[2]):
        storage_bytes(t).random_(0, 256, generator=gen)
    store, backs = register(torch, [("w0", w0), ("w1", w1), ("w2", fused[48:]), ("w3", w3)], kv, pages, slots)
    infos = {r.name: r for r in store.layout().regions}
    N = store.layout().blocks
    key = rng.bytes(32)
    ip = InPlace(torch, lib, [Region.of_backing(r.first, b) for r, b in zip(store.layout().regions, backs)], key,
                 kept_from=2)
    blocks = held(store, backs)
    page_blocks = 2 * 2   # planes · plane_blocks: a page's chunks in a layer's region
    out = {"blocks": N, "regions": [(r.name, r.kind, r.first, r.blocks) for r in store.layout().regions]}

    def check(name: str, plan: rb.Plan, draws: int = 0) -> bool:
        seed = rng.bytes(64)
        ys, want = reference(seed, plan, blocks, key)
        mapped = ip.pass_receipts(seed, plan)
        got = digests(mapped.cpu())
        root, handle = ip.run_pass(seed, plan)
        r = {"positions": plan.length, "runs": len(plan.runs), "receipts": got == ys, "root": root == want,
             "kept": unfused_kept(ip, mapped, handle)}
        for h in (0, FUSED_H0):
            other = InPlace(torch, lib, ip.regions, key, kept_from=h)
            root_h, handle_h = other.run_pass(seed, plan)
            r[f"kept_from_{h}"] = root_h == want and unfused_kept(other, mapped, handle_h)
        if draws:
            ps = sorted({0, plan.length - 1, *map(int, rng.integers(0, plan.length, draws))})
            ok = []
            for p, (blk, leaf, salt, path) in zip(ps, ip.open(handle, ps)):
                ok.append(blk == blocks[plan.block(p)] and rb.leaf_receipt(leaf, p, plan.length) == ys[p]
                          and rb.verify_leaf(seed, plan, root, p, leaf, salt, path))
            r["openings"], r["opened"] = all(ok), len(ok)
        out[name] = r
        log(f"gate {name}: {r}")
        return bool(r["receipts"] and r["root"] and r["kept"] and r["kept_from_0"] and r[f"kept_from_{FUSED_H0}"]
                    and r.get("openings", True))

    ok = check("whole", rb.Plan.whole(N), 24)
    kv_firsts = [infos[f"kv/{li}"].first for li in range(3)]
    live = {li: sorted(map(int, rng.choice(pages, 13, replace=False))) for li in range(3)}
    weights = [(infos["w0"].first + 3, 9), (infos["w2"].first, 10), (infos["w3"].first + 17, 100)]
    chosen = [j for f, c in weights for j in range(f, f + c)]
    kv_blocks = {li: [kv_firsts[li] + page_blocks * p + k for p in live[li] for k in range(page_blocks)] for li in live}
    mixed = sorted(chosen + [j for li in live for j in kv_blocks[li]])
    if len(mixed) % PER_LEAF == 0:
        mixed.pop()
    ok &= check("mixed", rb.Plan.of(mixed, N), 48)
    taken = set(mixed)
    order = ([j for p in reversed(live[2]) for j in kv_blocks[2][page_blocks * live[2].index(p):][:page_blocks]]
             + chosen + kv_blocks[0] + kv_blocks[1])
    ok &= check("reordered", rb.Plan.of([j for j in order if j in taken], N), 48)
    ok &= check("one_leaf", rb.Plan(((infos["w3"].first + 5, 5),), N), 4)
    ok &= check("two_leaves", rb.Plan(((kv_firsts[0] + 8, 9), (infos["w1"].first, 8)), N), 4)

    slots = ([kv_firsts[1] + page_blocks * 7 + k for k in range(page_blocks)]
             + [kv_firsts[0] + page_blocks * 3 + 1, infos["w0"].first + 2, kv_firsts[2] + 5])
    had = len(ip.regions)
    first = ip.append(slots)
    out["appended_regions"] = len(ip.regions) - had
    blocks += [blocks[s] for s in slots]
    grown = ([infos["w3"].first + k for k in range(30)] + list(range(first, first + len(slots)))
             + [kv_firsts[2] + k for k in range(8)])
    ok &= first == N and check("appended", rb.Plan.of(grown, ip.blocks), 8)

    before, _ = ip.run_pass(seed := rng.bytes(64), plan := rb.Plan.whole(ip.blocks))
    w3[5].fill_(1.5)
    kv[0][1, 3, 2].fill_(-2.0)
    kv[2][9, 0, 11].fill_(0.25)
    blocks = held(store, backs)
    blocks += [blocks[s] for s in slots]
    after, _ = ip.run_pass(seed, plan)
    out["written_changes_root"] = after != before
    ok &= out["written_changes_root"] and check("written", rb.Plan.whole(ip.blocks), 16)

    seed, w3_plan = rng.bytes(64), rb.Plan(((infos["w3"].first, infos["w3"].blocks),), ip.blocks)
    mapped = ip.pass_receipts(seed, w3_plan)
    pk = ip.set_pass(seed, w3_plan)
    flat = torch.empty_like(mapped)
    lib.call("rs_sketch16_receipts", backs[3].flat.data_ptr(), infos["w3"].first, infos["w3"].blocks, pk.data_ptr(),
             flat.data_ptr(), 0)
    out["contiguous_equal"] = bool(torch.equal(mapped, flat))
    ok &= out["contiguous_equal"]

    a0 = backs[0].flat.data_ptr()
    phys2 = backs[-1].physical
    out["refusals"] = {
        "misaligned": refused(lambda: InPlace(torch, lib, [Region(0, 1, a0 + 8)], key)),
        "kept_from_past_fused": refused(lambda: InPlace(torch, lib, [Region(0, 1, a0)], key, kept_from=FUSED_H0 + 1)),
        "gap": refused(lambda: InPlace(torch, lib, [Region(0, 4, a0), Region(5, 4, a0)], key)),
        "past_store": refused(lambda: ip.map(rb.Plan(((0, ip.blocks + 1),), ip.blocks + 1))),
        "part_block_plane": refused(lambda: Region.of_backing(0, PagedBacking(
            backs[-1].flat, backs[-1].base, dataclasses.replace(phys2, slots=6, slot_bytes=1000)))),
        "slot_past_store": refused(lambda: ip.append([ip.blocks])),
        "cpu_tensor": refused(lambda: Region.of_backing(0, TorchBacking.of_tensor(torch.empty(2 * BLOCK,
                                                                                             dtype=torch.uint8),
                                                                                 BLOCK))),
        "pages_past_storage": refused(lambda: Region.of_backing(0, PagedBacking(
            backs[-1].flat, backs[-1].base, dataclasses.replace(phys2, pages=pages + 1)))),
    }
    ok &= all(out["refusals"].values())
    out["ok"] = bool(ok)
    return out


def llama8b(torch, dev, gen) -> list:
    """Llama-3.1-8B's parameters as vLLM's model holds them (qkv and gate_up fused, lm_head untied), bf16, random."""
    vocab, h, layers, kv_heads, hd, inter = 128256, 4096, 32, 8, 128, 14336
    shapes = [("model.embed_tokens.weight", (vocab, h))]
    for li in range(layers):
        p = f"model.layers.{li}"
        shapes += [(f"{p}.self_attn.qkv_proj.weight", (h + 2 * kv_heads * hd, h)),
                   (f"{p}.self_attn.o_proj.weight", (h, h)),
                   (f"{p}.mlp.gate_up_proj.weight", (2 * inter, h)), (f"{p}.mlp.down_proj.weight", (h, inter)),
                   (f"{p}.input_layernorm.weight", (h,)), (f"{p}.post_attention_layernorm.weight", (h,))]
    shapes += [("model.norm.weight", (h,)), ("lm_head.weight", (vocab, h))]
    out = []
    for name, shape in shapes:
        t = torch.empty(shape, dtype=torch.bfloat16, device=dev)
        t.view(torch.uint8).random_(0, 256, generator=gen)
        out.append((name, t))
    return out


def timed(torch, fn, reps: int) -> dict:
    """fn's GPU time, ms: each of `reps` runs after one untimed, between two events on the current stream."""
    fn()
    ms = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    ms.sort()
    return {"median": ms[len(ms) // 2], "min": ms[0], "max": ms[-1]}


def timing(torch, lib: Lib, rng, reps: int, pages: int, log) -> dict:
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(1 << 62)))
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    weights = llama8b(torch, dev, gen)
    store, backs = register(torch, weights, [], 0, 0)
    regions = [Region.of_backing(r.first, b) for r, b in zip(store.layout().regions, backs)]
    N, key = store.layout().blocks, rng.bytes(32)
    ip = InPlace(torch, lib, regions, key)
    nbytes = N * BLOCK
    out = {"model": "Llama-3.1-8B bf16", "tensors": len(weights), "registered_bytes": nbytes, "blocks": N,
           "param_bytes": sum(t.numel() * t.element_size() for _, t in weights),
           "device": torch.cuda.get_device_name(dev)}
    copy = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    for r, b in zip(regions, backs):
        copy[r.first * BLOCK:(r.first + r.blocks) * BLOCK].copy_(b.flat[:r.blocks * BLOCK])
    xr = torch.zeros(1, dtype=torch.int64, device=dev)
    seed, plan = rng.bytes(64), rb.Plan.whole(N)
    addrs, js = ip.map(plan)
    pk = ip.set_pass(seed, plan)
    ys = torch.empty((N, 8), dtype=torch.int64, device=dev)
    ys2 = torch.empty_like(ys)
    t = {"read_contiguous": timed(torch, lambda: lib.call("rb_read", copy.data_ptr(), nbytes, xr.data_ptr(), 16 * sms),
                                  reps),
         "read_mapped": timed(torch, lambda: lib.call("rs_mapped_read", addrs.data_ptr(), N, xr.data_ptr(), 16 * sms),
                              reps),
         "map": timed(torch, lambda: ip.map(plan), reps),
         "receipts_contiguous": timed(torch, lambda: lib.call("rs_sketch16_receipts", copy.data_ptr(), 0, N,
                                                              pk.data_ptr(), ys2.data_ptr(), 0), reps),
         "receipts_mapped": timed(torch, lambda: lib.call("rs_mapped_receipts", addrs.data_ptr(), js.data_ptr(), N,
                                                          pk.data_ptr(), ys.data_ptr(), 0), reps)}
    out["receipts_equal"] = bool(torch.equal(ys, ys2))
    leaves = rb.leaves(N)
    width = 1 << (leaves - 1).bit_length()
    t["tree"] = timed(torch, lambda: ip.tree(ys, 0, leaves, width), reps)
    roots = {}
    t["pass_unfused"] = timed(torch, lambda: roots.update(unfused=unfused_pass(ip, seed, plan)), reps)
    walls = []

    def whole():
        w0 = time.perf_counter_ns()
        roots["fused"] = ip.run_pass(seed, plan)[0]
        walls.append((time.perf_counter_ns() - w0) / 1e6)

    t["pass_inplace"] = timed(torch, whole, reps)
    t["pass_inplace_wall"] = sorted(walls)[len(walls) // 2]
    t["set_pass"] = timed(torch, lambda: ip.set_pass(seed, plan), reps)
    pk = ip.set_pass(seed, plan)
    t["pass_kernels"] = timed(torch, lambda: roots.update(kernels=ip.fused(addrs, js, pk, N)[2]), reps)
    plain = ip.tree(ys2, 0, leaves, width)
    out["root_equal_contiguous"] = roots["fused"] == digests(plain[-1:].cpu())[0]
    out["root_equal_unfused"] = roots["fused"] == roots["unfused"] == digests(roots.pop("kernels")[-1:].cpu())[0]
    r = rb.receipter(rb.SKETCH_SHA512, seed, ROWS)
    host = ys.cpu()
    sample = sorted(map(int, rng.integers(0, N, 64)))
    out["sampled_receipts_reference"] = all(
        digests(host[p:p + 1])[0] == r(p, copy[p * BLOCK:(p + 1) * BLOCK].cpu().numpy().tobytes()) for p in sample)
    out["tau_inplace"] = t["pass_inplace"]["median"] / t["read_contiguous"]["median"]
    out["tau_unfused"] = t["pass_unfused"]["median"] / t["read_contiguous"]["median"]
    out["read_mapped_over_contiguous"] = t["read_mapped"]["median"] / t["read_contiguous"]["median"]
    out["read_GBps"] = {k: nbytes / (t[k]["median"] / 1e3) / 1e9 for k in ("read_contiguous", "read_mapped")}
    out["ms"] = t
    log(f"timing weights: {out}")
    copy = ys2 = plain = None

    slots, kv_heads, hd = 16, 8, 128
    kv = [torch.empty((2, pages, slots, kv_heads, hd), dtype=torch.bfloat16, device=dev) for _ in range(32)]
    for x in kv:
        x.view(torch.uint8).random_(0, 256, generator=gen)
    store2, backs2 = register(torch, weights, kv, pages, slots)
    ip2 = InPlace(torch, lib, [Region.of_backing(r.first, b) for r, b in zip(store2.layout().regions, backs2)], key)
    N2 = store2.layout().blocks
    plan2 = rb.Plan.whole(N2)
    addrs2, js2 = ip2.map(plan2)
    pk2 = ip2.set_pass(seed, plan2)
    ys = torch.empty((N2, 8), dtype=torch.int64, device=dev)
    t2 = {"read_mapped": timed(torch, lambda: lib.call("rs_mapped_read", addrs2.data_ptr(), N2, xr.data_ptr(),
                                                       16 * sms), reps),
          "receipts_mapped": timed(torch, lambda: lib.call("rs_mapped_receipts", addrs2.data_ptr(), js2.data_ptr(), N2,
                                                           pk2.data_ptr(), ys.data_ptr(), 0), reps),
          "pass_unfused": timed(torch, lambda: roots.update(kv_unfused=unfused_pass(ip2, seed, plan2)), reps),
          "pass_inplace": timed(torch, lambda: roots.update(kv_fused=ip2.run_pass(seed, plan2)[0]), reps)}
    blocks = {}
    kv0 = next(r for r in store2.layout().regions if r.kind == "kv").first
    sample = sorted(map(int, rng.integers(kv0, N2, 48)))
    for (info, b) in zip(store2.layout().regions, backs2):
        for p in sample:
            if info.first <= p < info.first + info.blocks:
                blocks[p] = b.read((p - info.first) * BLOCK, BLOCK)
    host = ys.cpu()
    kvr = {"pages": pages, "kv_bytes": (N2 - N) * BLOCK, "blocks": N2,
           "sampled_kv_receipts_reference": all(digests(host[p:p + 1])[0] == r(p, blocks[p]) for p in sample),
           "read_GBps": N2 * BLOCK / (t2["read_mapped"]["median"] / 1e3) / 1e9,
           "tau_over_mapped_read": t2["pass_inplace"]["median"] / t2["read_mapped"]["median"],
           "root_equal_unfused": roots["kv_fused"] == roots["kv_unfused"], "ms": t2}
    log(f"timing weights+kv: {kvr}")
    out["with_kv"] = kvr
    out["ok"] = bool(out["receipts_equal"] and out["root_equal_contiguous"] and out["root_equal_unfused"]
                     and out["sampled_receipts_reference"] and kvr["sampled_kv_receipts_reference"]
                     and kvr["root_equal_unfused"])
    return out


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="session.so")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--pages", type=int, default=2048, help="the 8B timing's KV pages a layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lib = Lib(a.lib, torch)
    res = {"gate": gate(torch, lib, rng, print)}
    if res["gate"]["ok"] and not a.no_timing:
        torch.cuda.empty_cache()
        res["timing"] = timing(torch, lib, rng, a.reps, a.pages, print)
    res["ok"] = bool(res["gate"]["ok"] and (a.no_timing or res["timing"]["ok"]))
    (a.out / "inplace.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"ok": res["ok"], "gate_ok": res["gate"]["ok"]}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
