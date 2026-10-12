"""One arm of the encoded-weights decode benchmark: Llama-3.1-8B or 70B served by vLLM on one GPU (in process, FP8 online
quantization, full CUDA graphs for decode), timed at each decode batch and context length the KV cache holds.

Arms:
  stock     vLLM's own FP8 linears (CUTLASS on sm_120), same engine settings
  plain     every linear through `ve.cu`'s fused GEMM over plain FP8 fragments (k = 0): the native kernel of this harness
  enc       every linear through the fused GEMM over codec words only (k rounds of Sloth at p = 2^61 - 1); the plaintext
            weight is freed before vLLM profiles memory, so decode runs with no plaintext copy of any linear in HBM
  wide      the same with ra-codec-wide's fused GEMM (`wide.cu`, ``--wide-lib``: Sloth over p = 2^127 - 1, 2^255 - 19,
            2^521 - 1, 2^256 - 189 or 2^512 - 569, k rounds, k = 0 its plain payloads), with ``--tag`` also tagging every
            block of words it reads (ro-blake3/v1, the bandwidth audit's tag) as decode runs, or with ``--sketch 16``
            sketching the encoded words its CTAs hold and hashing the sketches (ro-sketch-blake3/v0, `wide_sketch`), and
            with ``--commit`` also committing each decode step as it goes: each linear's tags in one slot of a two-slot
            ring that L2 holds, their leaves hashed (and folded) beside the next linear's GEMM, the tree after the last
  tensor    the same with ra-codec-tensor's fused GEMM (`wt.cu`, ``--tensor-lib``: Sloth over its field, decoder warps
            decoding in shared memory ahead of one MMA warp, k >= 1 rounds), with ``--tag`` its tag warps tagging the
            store's 4,088-byte blocks (ro-blake3/v1) as decode runs, or with ``--sketch K_S`` (k = 1) its decoder warps
            sketching the stored words they decode on the tensor cores and its MMA warp tagging the sketches
            (ro-sketch-blake3/v0), and with
            ``--commit`` also each decode step's commit: the step's tags in one buffer, their leaves and the tree's
            root, after the step's last linear

Each arm runs the engine in compilation mode 0 with full CUDA graphs for decode, or with ``--compiled`` as vLLM serves by
default (torch.compile, piecewise and full graphs; the fused linear is the opaque op ``ve_enc::linear``, and vLLM's activation
quantization before it is compiled as in the stock arm).  vLLM's default compile tunes inductor's kernels by timing in
each process, so two processes can compile different reductions; ``--no-autotune`` compiles without any choice made by
timing (inductor's deterministic mode), the same in every process.

The plain and encoded arms install at the end of the model runner's `load_model` (before memory profiling, compilation and
graph capture), through `verity_vllm.engine.hooks`.  Before anything is timed, each linear passes its gates: every GEMM form
gives the same partial sums bit for bit (encoded against plain at every capture size), the plain sums agree with a float
product, the encoded store decodes to the payloads on the device and in `codec.py` (`wide.py`, `codec_tensor/wide.py`),
sampled tags equal the bandwidth audit's reference (`verity.core.protocols.bandwidth.tags`), a committed arm's step at
each capture size gives the reference's leaves and root over its tags, and (encoded arms) no FP8 or
plain fragment tensor of a linear's size is left anywhere in the process.  Each arm then generates fixed prompts greedily
(`gate`), which `compare.py` requires to be bit-identical between plain and encoded arms (token ids and log-probs).

A decode step's time is the interval between two consecutive scheduler updates that both scheduled exactly B decode tokens
for B requests (`update_from_output`), the median over the point's steady decode; prefill is not timed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kernels as K
import layout
import numpy as np
import torch

CAPTURE = [1, 4, 8, 16, 32]
ENCODED = ("enc", "wide", "tensor")
SEED = hashlib.sha512(b"ra-vllm-encoded/key/v0").digest()[:32]
TAG_SEED = hashlib.sha256(b"ra-vllm-encoded/tag-seed/v0").digest()
TAG = "ro-blake3/v1"
GATE_PROMPTS = [
    "The capital of France is",
    "Write a haiku about memory bandwidth.",
    "def fibonacci(n):",
    "Explain why the sky is blue in one sentence.",
]


def log(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def unwrap(m):
    while not isinstance(m, torch.nn.Module) and hasattr(m, "runnable"):
        m = m.runnable
    return m


def fp8_linears(model) -> list[tuple[str, torch.nn.Module]]:
    from vllm.model_executor.layers.linear import LinearBase
    out = []
    for name, m in model.named_modules():
        if isinstance(m, LinearBase) and getattr(m, "weight", None) is not None and m.weight.dtype == torch.float8_e4m3fn:
            out.append((name, m))
    return out


def rand_fp8(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    return (torch.randn((m, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


class Install:
    """The plain or encoded arm's linears, built inside the runner's `load_model`."""

    def __init__(self, args) -> None:
        self.args = args
        self.record: dict = {}
        self.linears: list[tuple[str, K.Linear]] = []
        self.hooks = None

    def __call__(self, runner) -> None:
        from verity_vllm.engine.hooks import Hooks
        args = self.args
        t0 = time.perf_counter()
        model = unwrap(runner.model)
        layers = fp8_linears(model)
        if not layers:
            raise RuntimeError("no FP8 linear layers to install over")
        lib = K.Lib(args.lib)
        wide, tensor = args.arm == "wide", args.arm == "tensor"
        k = args.k if args.arm in ENCODED else 0
        wlib = K.WideLib(args.wide_lib) if wide else None
        tlib = K.TensorLib(args.tensor_lib) if tensor else None
        if args.sketch and not ((tensor and k == 1) or (wide and k >= 1 and args.sketch == 16)):
            raise ValueError("--sketch is the tensor arm's at k = 1, or the wide arm's at k_s = 16")
        if args.commit and not args.sketch:
            raise ValueError("--commit commits the sketch tags (--sketch)")
        if tensor:
            tlib.set_key(SEED, k)
            if args.sketch:
                tlib.set_sketch(TAG_SEED, args.sketch, args.sketch_t)
            elif args.tag:
                tlib.set_tag(TAG_SEED)
            if args.sketch or args.tag:
                tlib.tag_buffer(max(tlib.tag_vectors(m.output_size_per_partition, m.input_size_per_partition)
                                    for _, m in layers))
        if wide:
            if k:
                wlib.set_key(SEED, k)
            if args.sketch:
                wlib.set_sketch(TAG_SEED)
                wlib.tag_buffer(max(wlib.vectors(m.output_size_per_partition, m.input_size_per_partition)
                                    for _, m in layers))
            elif args.tag:
                wlib.set_tag(TAG_SEED)
                wlib.tag_buffer(max(layout.wide_store_words(m.output_size_per_partition, m.input_size_per_partition,
                                                            wlib.payload) // wlib.lane_words for _, m in layers))
        elif k:
            lib.set_key(SEED, k)
        torch.cuda.synchronize()
        alloc0 = torch.cuda.memory_allocated()
        gen = torch.Generator(device="cuda")
        cands: dict[tuple, list[K.Config]] = {}
        gates = {"layers": 0, "configs_checked": 0, "reps": args.gate_reps, "python_words": 0, "tags_checked": 0,
                 "max_rel_err": 0.0}
        base, tag_base, plain_bytes, store_bytes, encode_s = 0, 0, 0, 0, 0.0
        self.hooks = Hooks(tag="ra-vllm-encoded")
        for idx, (name, layer) in enumerate(layers):
            w = layer.weight.t()
            n, kd = w.shape
            if not w.is_contiguous() or (n, kd) != (layer.output_size_per_partition, layer.input_size_per_partition):
                raise RuntimeError(f"{name}: weight {tuple(layer.weight.shape)} is not the (K, N) view of an N x K matrix")
            ws = layer.weight_scale
            if ws.numel() != 1:
                raise RuntimeError(f"{name}: weight scale {tuple(ws.shape)} is not per tensor")
            plain = K.PlainLinear(lib, w, ws)
            if k or wide:
                torch.cuda.synchronize()
                te = time.perf_counter()
                if tensor:
                    lin = K.TensorLinear(lib, tlib, w, ws, k, base, tag_base)
                elif wide:
                    lin = K.WideLinear(lib, wlib, w, ws, k, base, tag_base=tag_base if args.sketch else None)
                else:
                    lin = K.EncodedLinear(lib, w, ws, k, base)
                torch.cuda.synchronize()
                encode_s += time.perf_counter() - te
            else:
                lin = plain
            nch = kd // layout.TASK_COLS
            gen.manual_seed(1000 + idx)
            bad = []
            for m in CAPTURE:
                x8 = rand_fp8(m, kd, gen)
                ref = plain.gemm(x8, K.Config(0))
                for op in (plain, lin) if lin is not plain else (plain,):
                    for cfg in cands.setdefault((type(op).__name__, op.k, m, nch), op.candidates(m)):
                        reps = args.gate_reps if op is lin else 1
                        if not all(bits_equal(op.gemm(x8, cfg), ref) for _ in range(reps)):
                            bad.append({"op": type(op).__name__, "config": cfg.name, "m": m,
                                        **diagnose(op, x8, cfg, ref, w, ws)})
                        gates["configs_checked"] += 1
                if m == 4:
                    exact = x8.float() @ w.float().t()
                    err = (ref.sum(0) - exact).abs().max().item() / max(exact.abs().max().item(), 1e-30)
                    gates["max_rel_err"] = max(gates["max_rel_err"], err)
                    if err > 1e-3:
                        raise RuntimeError(f"gate: {name}: fused sums are {err:.2e} off the float product")
            if bad:
                raise RuntimeError(f"gate: {name} k={lin.k} (word base {lin.base}): partial sums differ from plain simple: "
                                   + json.dumps(bad))
            if args.sketch:
                gates["tags_checked"] += (check_wide_sketch if wide else check_sketch)(lin, name, args.python_tags)
            elif args.tag:
                gates["tags_checked"] += check_tags(lin, name, args.python_tags)
            if k or wide:
                if not torch.equal(lin.decoded(), lin.payloads):
                    raise RuntimeError(f"gate: {name}: the store does not decode to its payloads")
                if not torch.equal(lin.dense().view(torch.uint8), w.view(torch.uint8)):
                    raise RuntimeError(f"gate: {name}: the decoded weight is not the weight")
                if k:
                    check_words(lin, name, idx, args.python_words)
                    gates["python_words"] += args.python_words
                lin.drop_payloads()
                base += lin.words if wide or tensor else lin.store.numel()
                tag_base += lin.blocks if tensor or (wide and args.sketch) else 0
                del plain
            elif not torch.equal(lin.dense().view(torch.uint8), w.view(torch.uint8)):
                raise RuntimeError(f"gate: {name}: the plain fragments are not the weight")
            plain_bytes += w.numel()
            store_bytes += lin.nbytes
            lin.name = name
            self.linears.append((name, lin))
            del w
            layer._parameters["weight"] = torch.nn.Parameter(
                torch.empty(0, dtype=torch.float8_e4m3fn, device=layer.weight_scale.device), requires_grad=False)
            K.LINEARS.append(lin)
            fn = apply(len(K.LINEARS) - 1, layer.quant_method.fp8_linear.quant_fp8)
            self.hooks.wrap(layer.quant_method, "apply", lambda orig, fn=fn: fn)
            gates["layers"] += 1
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc1 = torch.cuda.memory_allocated()
        selection = self.select()
        if args.commit:
            lins = [lin for _, lin in self.linears]
            lins[0].first = lins[-1].last = True
            if wide:
                wlib.commit_every_step([lin.blocks for lin in lins])
                gates["commit"] = check_wide_commit(self.linears, 1)
            else:
                tlib.commit_every_step(max(sum(tlib.vectors(lin.n, lin.kd,
                                                            tlib.steps(m, lin.n, lin.kd, lin.configs[m])[0])
                                               for lin in lins) for m in CAPTURE))
                gates["commit"] = {m: check_commit(self.linears, m, 1) for m in CAPTURE}
        scan = plaintext_scan(min(lin.n * lin.kd for _, lin in self.linears))
        if k and (scan["fp8"] or scan["uint8"]):
            raise RuntimeError(f"gate: plaintext left in the process: {scan}")
        if tensor:
            codec = {"field": tlib.field.name, "word_bits": tlib.bits, "payload_bits": 8 * tlib.payload_bytes,
                     "limbs": tlib.limbs, "word_bytes": tlib.word_bytes, "task_words": tlib.task_words,
                     "tensor_lib": Path(args.tensor_lib).name, "kmax": tlib.kmax, "tag_blocks": tag_base,
                     "tag": tlib.sketch["name"] if tlib.sketch else TAG if tlib.tag_seed is not None else None}
            if tlib.sketch:
                vecs = {m: sum(tlib.so.wt_sketch_vectors(lin.n, lin.kd, tlib.steps(m, lin.n, lin.kd, lin.configs[m])[0],
                                                         tlib.sketch["t"]) for _, lin in self.linears) for m in CAPTURE}
                codec |= {"sketch_k": tlib.sketch["k"], "sketch_t": tlib.sketch["t"], "tag_blocks": vecs[1],
                          "tag_vectors": vecs, "tag_index_space": tag_base, "commit": bool(args.commit)}
                if tlib.step:
                    codec["commit_buffer_bytes"] = sum(x.numel() * x.element_size()
                                                       for x in (tlib.step["tags"], tlib.step["nodes"]))
        elif wide:
            codec = {"field": wlib.field.name, "word_bits": wlib.bits, "payload_bits": wlib.payload, "limbs": wlib.limbs,
                     "lane_words": wlib.lane_words, "wide_lib": Path(args.wide_lib).name, "kmax": wlib.kmax,
                     "constants": wlib.constants, "tag": TAG if wlib.tag_seed is not None else None}
            if wlib.sketch:
                codec |= {"tag": wlib.sketch["name"], "sketch_k": wlib.sketch["k"], "sketch_t": None,
                          "sketch_tile_row_blocks": wlib.sketch["t"], "sketch_vector_bytes": wlib.sketch["n"],
                          "tag_blocks": tag_base, "commit": bool(args.commit), "commit_ring": bool(args.commit)}
                if wlib.step:
                    codec |= {"commit_fold": wlib.step["fold"], "commit_fold_levels": wlib.sub,
                              "commit_buffer_bytes": sum(x.numel() * x.element_size()
                                                         for x in (wlib.step["nodes"], *wlib.step["ring"]))}
        else:
            codec = {"field": "p61", "word_bits": 61, "payload_bits": lib.bits, "tag": None}
        self.record = {
            "arm": args.arm, "k": k, "lib": Path(args.lib).name, "bits": codec["payload_bits"], "kmax": lib.kmax,
            "codec": codec, "layers": len(layers),
            "plain_bytes": plain_bytes, "store_bytes": store_bytes, "store_over_plain": store_bytes / plain_bytes,
            "words": base, "encode_s": encode_s, "allocated_before": alloc0, "allocated_after": alloc1,
            "allocated_delta": alloc1 - alloc0, "plaintext_scan": scan, "gates": gates, "selection": selection,
            "fp8_kernel": type(layers[0][1].quant_method.fp8_linear).__name__,
            "activation_quant": str(layers[0][1].quant_method.activation_quant_key),
            "install_s": time.perf_counter() - t0,
        }
        log("installed", json.dumps({k_: v for k_, v in self.record.items() if k_ != "selection"}))

    def select(self) -> list[dict]:
        """Each shape's GEMM config at each capture size: the fastest candidate over all of the shape's layers in turn (so
        the weights stream from HBM, not L2)."""
        groups: dict[tuple[int, int], list[K.Linear]] = {}
        for _, lin in self.linears:
            groups.setdefault((lin.n, lin.kd), []).append(lin)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(7)
        out = []
        for (n, kd), lins in groups.items():
            for m in CAPTURE:
                x8 = rand_fp8(m, kd, gen)
                times = {}
                for cfg in lins[0].candidates(m):
                    times[cfg] = time_config(lins, x8, cfg, self.args.select_trials)
                best = min(times, key=times.get)
                for lin in lins:
                    lin.configs[m] = best
                out.append({"n": n, "k": kd, "m": m, "layers": len(lins), "best": best.name, "groups": best.groups,
                            "us": {c.name: round(t, 2) for c, t in times.items()}})
                log(f"select n={n} k={kd} m={m}: {best.name} {times[best]:.1f} us", {c.name: round(t, 1) for c, t in times.items()})
        torch.cuda.empty_cache()
        return out


def diagnose(lin: K.Linear, x8: torch.Tensor, cfg: K.Config, ref: torch.Tensor, w: torch.Tensor,
             ws: torch.Tensor) -> dict:
    """A failing GEMM gate's sums: how many differ and in which chunks and 16-row blocks, whether two more launches
    give the same sums, whether the store decodes to its payloads, and (tensor) whether the same payloads encoded at
    word index 0 pass."""
    got = lin.gemm(x8, cfg)
    diff = got.view(torch.int32) != ref.view(torch.int32)
    out = {"differ": int(diff.sum()), "of": diff.numel(),
           "chunks": diff.any(2).any(1).nonzero().flatten()[:8].tolist(),
           "row_blocks": torch.unique(diff.any(1).any(0).nonzero().flatten() // 16)[:8].tolist(),
           "reruns_equal": bits_equal(lin.gemm(x8, cfg), got) and bits_equal(lin.gemm(x8, cfg), got)}
    if getattr(lin, "payloads", None) is not None:
        out["decodes"] = torch.equal(lin.decoded(), lin.payloads)
    if isinstance(lin, K.TensorLinear):
        z = K.TensorLinear(lin.lib, lin.tlib, w, ws, lin.k, 0, lin.tag_base)
        out["at_base_0"] = bits_equal(z.gemm(x8, cfg), ref)
        del z
    return out


def check_words(lin: K.Linear, name: str, idx: int, n: int) -> None:
    """``n`` sampled words of the store decode to their payloads in the Python reference (`codec.py`, `wide.py` or
    `codec_tensor/wide.py`)."""
    rng = random.Random(idx)
    if isinstance(lin, K.TensorLinear):
        ct, t = K._ct(), lin.tlib
        for j in [rng.randrange(lin.words) for _ in range(n)]:
            y, want = layout.tensor_word(lin.bytes, j, t.word_bytes), layout.tensor_limb_word(lin.payloads, j, t.limbs)
            if ct.decode_word(y, lin.base + j, t.key) != want:
                raise RuntimeError(f"gate: {name}: codec_tensor/wide.py decodes word {lin.base + j} otherwise")
        return
    if isinstance(lin, K.WideLinear):
        wide, w = K._wide(), lin.wlib
        for j in [rng.randrange(lin.words) for _ in range(n)]:
            y, want = layout.wide_word(lin.store, j, w.limbs), layout.wide_word(lin.payloads, j, w.limbs)
            if wide.decode_word(y, lin.base + j, w.key, w.payload) != want:
                raise RuntimeError(f"gate: {name}: wide.py decodes word {lin.base + j} otherwise")
        return
    codec, store = K._codec(), lin.store.view(torch.int64)
    for j in [rng.randrange(store.numel()) for _ in range(n)]:
        if codec.decode_word(int(store[j]), lin.base + j, lin.lib.key, lin.lib.bits) != int(lin.payloads[j]):
            raise RuntimeError(f"gate: {name}: codec.py decodes word {lin.base + j} otherwise")


def check_tags(lin: K.WideLinear | K.TensorLinear, name: str, n: int) -> int:
    """The last GEMM's tags of ``n`` sampled blocks and the last block equal the bandwidth audit's ro-blake3/v1 of the
    block's encoded words at its index (a wide lane's L words, or 4,088 bytes of a tensor store, the last block short)."""
    from verity.core.protocols.bandwidth import tags
    tensor = isinstance(lin, K.TensorLinear)
    w = lin.tlib if tensor else lin.wlib
    torch.cuda.synchronize()
    keyed = tags.get(TAG).keyed(w.tag_seed)
    rng = random.Random(lin.base)
    js = sorted({lin.blocks - 1, *(rng.randrange(lin.blocks) for _ in range(n))})
    got = w.tags[:8 * lin.blocks].view(-1, 8)[js].cpu().numpy()
    first = lin.tag_base if tensor else lin.tag_block_base
    for j, t in zip(js, got):
        if tensor:
            block = lin.bytes[w.tag_block * j:w.tag_block * (j + 1)].cpu().numpy().tobytes()
        else:
            block = layout.wide_block(lin.store, j, w.lane_words, w.limbs)
        if t.tobytes() != keyed(first + j, block):
            raise RuntimeError(f"gate: {name}: block {first + j}'s tag is not {TAG}'s")
    return len(js)


def check_sketch(lin: K.TensorLinear, name: str, n: int, at: int | None = None) -> int:
    """The last GEMM's tags of ``n`` sampled vectors and the last vector equal ro-sketch-blake3/v0's: each vector's
    sketch exactly (`sketching.sketch`, float64 over its stored bytes as `sketching.vectors` lays them out at the launch's
    CTA groups), then its keyed BLAKE3 at index tag_base + v (`tags.py`); with ``at``, the GEMM ran inside a committed
    step, its vector v at index at + v of the step's tags."""
    skt, t = K._sketching(), lin.tlib
    torch.cuda.synchronize()
    nvec = t.vectors(lin.n, lin.kd, lin.groups)
    if not 0 < nvec <= lin.blocks:
        raise RuntimeError(f"gate: {name}: {nvec} sketch vectors at {lin.groups} groups, not within (0, {lin.blocks}]")
    rng = random.Random(lin.base)
    vs = sorted({nvec - 1, *(rng.randrange(nvec) for _ in range(n))})
    words = skt.sketch_words(skt.sketch(lin.sketch_vectors(vs, lin.groups), t.sketch["bits"]))
    if at is None:
        first, got = lin.tag_base, t.tags[:8 * nvec]
    else:
        first, got = at, t.step["tags"][8 * at:8 * (at + nvec)]
    got = got.view(-1, 8)[vs].cpu().numpy()
    keyed = skt.tag().keyed(t.tag_seed)
    for v, w, g in zip(vs, words, got):
        if g.tobytes() != keyed(first + v, w.tobytes()):
            raise RuntimeError(f"gate: {name}: vector {first + v}'s tag is not {t.sketch['name']}'s")
    return len(vs)


def check_commit(linears: list[tuple[str, K.TensorLinear]], m: int, n: int) -> dict:
    """One decode step's linears at ``m`` rows through the committed forward, as the step's CUDA graph runs them: ``n``
    sampled tags of each and its last at their step indices (`check_sketch`), then every leaf and the root equal
    `tags.py`'s leaves over the step's tags and RFC 6962's tree over them (`codec_tensor/tagging.py`)."""
    skt, t = K._sketching(), linears[0][1].tlib
    tg = skt.TG.tags_module()
    if tg.tags_per_leaf(skt.tag()) != 16:
        raise RuntimeError(f"{t.sketch['name']}'s leaves are not wt_tag_leaves' 16 tags")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2000 + m)
    xs = torch.ones(1, dtype=torch.float32, device="cuda")
    for _, lin in linears:
        lin.forward(rand_fp8(m, lin.kd, gen), xs)
    torch.cuda.synchronize()
    nvec, ri = t.step["roots"][m]
    at, checked = 0, 0
    for name, lin in linears:
        checked += check_sketch(lin, name, n, at)
        at += t.vectors(lin.n, lin.kd, lin.groups)
    if at != nvec:
        raise RuntimeError(f"gate: the step at m = {m} committed {nvec} tags, its linears wrote {at}")
    tags = t.step["tags"][:8 * nvec].cpu().numpy().tobytes()
    head, nleaves = tg.leaf_head(skt.tag(), t.tag_seed), -(-nvec // 16)
    want = [tg.leaf_hash(head, c, [tags[512 * c:512 * (c + 1)]]) for c in range(nleaves)]
    nodes = t.step["nodes"].cpu().numpy().astype(">i8")  # SHA-512's state words, the digest's bytes big-endian
    got = nodes[:8 * nleaves].tobytes()
    if got != b"".join(want):
        bad = next(c for c in range(nleaves) if got[64 * c:64 * (c + 1)] != want[c])
        raise RuntimeError(f"gate: the step at m = {m}: leaf {bad} of {nleaves} is not tags.py's")
    root = nodes[8 * ri:8 * ri + 8].tobytes()
    if root != skt.TG.root(want):
        raise RuntimeError(f"gate: the step at m = {m}: the device's root is not RFC 6962's over the leaves")
    log(f"commit gate m={m}: {nvec} tags, {nleaves} leaves, {checked} tags checked, root {root.hex()[:16]}")
    return {"tags": nvec, "leaves": nleaves, "tags_checked": checked, "root": root.hex()}


def wide_vectors(lin: K.WideLinear, vs: list[int]) -> np.ndarray:
    """Vectors ``vs`` of a sketched wide GEMM (vector 16 (G·chunks + kc) + r: row r of tile (G, kc)), each laid out by
    `wide_sketch.vectors` from its tile's T records (row blocks T·G … T·G + T − 1 at chunk kc) of the device store."""
    ws, w = K._wide_sketch(), lin.wlib
    t, rec, nbs = w.sketch["t"], w.lane_words * w.limbs * layout.LANES, lin.n // 16
    out = np.zeros((len(vs), w.sketch["n"]), np.int8)
    for o, v in enumerate(vs):
        q, r = divmod(v, 16)
        g, kc = divmod(q, lin.nchunks)
        recs = [lin.store[(nb * lin.nchunks + kc) * rec:(nb * lin.nchunks + kc + 1) * rec].cpu().numpy() if nb < nbs
                else np.zeros(rec, np.int32) for nb in range(t * g, t * g + t)]
        out[o] = ws.vectors(np.concatenate(recs).astype("<i4").tobytes(), 16 * t, layout.TASK_COLS, w.field)[r]
    return out


def check_wide_sketch(lin: K.WideLinear, name: str, n: int, got: torch.Tensor | None = None,
                      first: int | None = None) -> int:
    """The last GEMM's tags of ``n`` sampled vectors and the last vector equal ro-sketch-blake3/v0's over the wide store
    (`wide_sketch`): each vector's sketch exactly, then its keyed BLAKE3 at index tag_base + v; with ``got`` (the
    linear's tags, 8 int32 each) and ``first``, the same at a step's indices first + v."""
    ws, w = K._wide_sketch(), lin.wlib
    torch.cuda.synchronize()
    rng = random.Random(lin.base)
    vs = sorted({lin.blocks - 1, *(rng.randrange(lin.blocks) for _ in range(n))})
    if got is None:
        got, first = w.tags[:8 * lin.blocks], lin.tag_base
    got = got.view(-1, 8)[vs].cpu().numpy()
    sk = ws.sketch(wide_vectors(lin, vs), w.sign_matrix)
    if np.abs(sk).max(initial=0) >= 1 << 31:
        raise RuntimeError(f"gate: {name}: a sketch word is past an int32")
    keyed = ws.tag_scheme().keyed(w.tag_seed)
    for v, s, g in zip(vs, sk.astype("<i4"), got):
        if g.tobytes() != keyed(first + v, s.tobytes()):
            raise RuntimeError(f"gate: {name}: vector {first + v}'s tag is not {w.sketch['name']}'s")
    return len(vs)


def check_wide_commit(linears: list[tuple[str, K.WideLinear]], n: int) -> dict:
    """The wide kernel's commit of a decode step.  First a reference step: each linear's GEMM writes its tags at its step
    indices into one commit workspace and `ww_sketch_commit` commits them; ``n`` sampled tags of each linear and its last
    equal `wide_sketch`'s (`check_wide_sketch`), every leaf equals `tags.py`'s over the step's tags, and the root is
    RFC 6962's over the leaves (PoUS's tree).  Then at each capture size, the committed forward as the step's CUDA graph
    runs it (the ring, each linear's leaves beside the next GEMM, the fold, the tree after the last) leaves the same
    leaves and root on the device."""
    import ctypes

    from verity.core.protocols.bandwidth import tags as tg
    from verity.core.protocols.pous.verifier import MerkleTree
    ws_mod, w = K._wide_sketch(), linears[0][1].wlib
    st, scheme = w.step, ws_mod.tag_scheme()
    total, nleaves = st["vectors"], -(-st["vectors"] // 16)
    if tg.tags_per_leaf(scheme) != 16:
        raise RuntimeError(f"{w.sketch['name']}'s leaves are not ww_sketch_leaves' 16 tags")
    wsp = torch.empty(w.so.ww_workspace_bytes(total), dtype=torch.uint8, device="cuda")
    tags0, area = w.so.ww_workspace_tags(wsp.data_ptr(), total), -(-32 * total // 256) * 256
    if tags0 != wsp.data_ptr():
        raise RuntimeError("the commit workspace's tags are not at its start")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2000)
    at = 0
    for _, lin in linears:
        x8 = rand_fp8(1, lin.kd, gen)
        part = torch.empty((lin.nchunks, 1, lin.n), dtype=torch.float32, device="cuda")
        lin.sketched(x8, part, True, lin.config(1), tags0 + 32 * at, at)
        at += lin.blocks
    if at != total:
        raise RuntimeError(f"gate: the step's linears write {at} tags, its commit holds {total}")
    torch.cuda.synchronize()
    tags_i32 = wsp[:32 * total].view(torch.int32)
    at, checked = 0, 0
    for name, lin in linears:
        checked += check_wide_sketch(lin, name, n, tags_i32[8 * at:8 * (at + lin.blocks)], at)
        at += lin.blocks
    ref = ctypes.create_string_buffer(64)
    w.call("ww_sketch_commit", wsp.data_ptr(), total, w._seed, ref, K.stream())
    torch.cuda.synchronize()
    raw = wsp[:32 * total].cpu().numpy().tobytes()
    head = tg.leaf_head(scheme, w.tag_seed)
    want = [tg.leaf_hash(head, c, [raw[512 * c:512 * (c + 1)]]) for c in range(nleaves)]
    del raw
    leaves = wsp[area:area + 64 * nleaves].view(torch.int64)
    got = leaves.cpu().numpy().astype(">i8").tobytes()
    if got != b"".join(want):
        bad = next(c for c in range(nleaves) if got[64 * c:64 * (c + 1)] != want[c])
        raise RuntimeError(f"gate: the reference step's leaf {bad} of {nleaves} is not tags.py's")
    root = MerkleTree(want).root
    if root != ref.raw:
        raise RuntimeError("gate: the reference step's root is not RFC 6962's over its leaves")
    ref_leaves = leaves.clone()
    del want, leaves, tags_i32, wsp
    torch.cuda.empty_cache()
    xs = torch.ones(1, dtype=torch.float32, device="cuda")
    for m in CAPTURE:
        st["nodes"].zero_()
        st["roots"].pop(m, None)
        for _, lin in linears:
            lin.forward(rand_fp8(m, lin.kd, gen), xs)
        torch.cuda.synchronize()
        if st["roots"].get(m) != total:
            raise RuntimeError(f"gate: the committed step at m = {m} committed {st['roots'].get(m)} tags, not {total}")
        if not torch.equal(st["nodes"][:8 * nleaves], ref_leaves):
            raise RuntimeError(f"gate: the committed step at m = {m} left other leaves than the reference step's")
        if w.step_root() != root:
            raise RuntimeError(f"gate: the committed step at m = {m} left another root than the reference step's")
    del ref_leaves
    torch.cuda.empty_cache()
    log(f"commit gate (wide): {total} tags, {nleaves} leaves, {checked} tags checked, root {root.hex()[:16]}, the same "
        f"at m = {CAPTURE}, fold {st['fold']}")
    return {"tags": total, "leaves": nleaves, "tags_checked": checked, "root": root.hex(), "sizes": CAPTURE,
            "fold": st["fold"]}


def apply(idx: int, quant):
    """A layer's FP8 apply: vLLM's own activation quantization (traced and fused by torch.compile when the engine compiles),
    then the linear through the op."""
    def fn(layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"activations are {x.dtype}, the fused linear writes BF16")
        x8, xs = quant(x.reshape(-1, x.shape[-1]), None, None)
        y = torch.ops.ve_enc.linear(x8, xs, idx)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])
    return fn


def time_config(lins: list[K.Linear], x8: torch.Tensor, cfg: K.Config, trials: int) -> float:
    """Microseconds a layer: the shape's layers in turn, captured in one CUDA graph as decode runs them (one partials
    buffer, as the graph pool reuses a freed one), the fastest of ``trials`` timings of two replays each."""
    part = torch.empty((lins[0].nchunks, x8.shape[0], lins[0].n), dtype=torch.float32, device=x8.device)
    for lin in lins:
        lin.gemm(x8, cfg, part=part)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for lin in lins:
            lin.gemm(x8, cfg, part=part)
    g.replay()
    best = math.inf
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 2)
    del g
    return best * 1e3 / len(lins)


def graph_ms(fn, trials: int) -> float:
    """Milliseconds of ``fn``'s kernels captured in one CUDA graph: the fastest of ``trials`` timings of two replays."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    best = math.inf
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 2)
    del g
    torch.cuda.empty_cache()
    return best


def native_pass(layers: list[tuple[str, torch.nn.Module]], ms: list[int], trials: int) -> list[dict]:
    """One native pass of the weights at each batch m: every FP8 linear in model order through vLLM's own apply (its
    activation quantization and GEMM, eager) and through ``torch._scaled_mm`` alone on FP8 activations (per-tensor
    scales), each in one CUDA graph.  The rate audit's d is a step's time in such passes."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    out = []
    for m in ms:
        xs = [torch.randn((m, layer.input_size_per_partition), generator=gen, device="cuda", dtype=torch.bfloat16)
              for _, layer in layers]
        x8 = [x.to(torch.float8_e4m3fn) for x in xs]

        def run_apply(xs=xs):
            for (_, layer), x in zip(layers, xs):
                layer.quant_method.apply(layer, x)

        def run_mm(x8=x8):
            for (_, layer), x in zip(layers, x8):
                torch._scaled_mm(x, layer.weight, scale_a=one, scale_b=layer.weight_scale.reshape(()),
                                 out_dtype=torch.bfloat16)

        point = {"batch": m, "apply_ms": graph_ms(run_apply, trials)}
        try:
            point["scaled_mm_ms"] = graph_ms(run_mm, trials)
        except RuntimeError as e:
            point["scaled_mm_error"] = str(e)[:300]
        point["pass_ms"] = min(point["apply_ms"], point.get("scaled_mm_ms", math.inf))
        out.append(point)
        log("native pass", json.dumps(point))
    return out


def plaintext_scan(min_elems: int) -> dict:
    """CUDA tensors in the process at least one linear's size, by dtype: FP8 (a weight) and uint8 (plain fragments)."""
    gc.collect()
    seen, out = set(), {"fp8": 0, "uint8": 0, "fp8_bytes": 0, "uint8_bytes": 0, "held_by": []}
    for o in gc.get_objects():
        if not isinstance(o, torch.Tensor) or o.is_meta or not o.is_cuda or o.numel() < min_elems:
            continue
        key = (o.untyped_storage().data_ptr(), o.dtype)
        if key in seen or o.dtype not in (torch.float8_e4m3fn, torch.uint8):
            continue
        seen.add(key)
        kind = "fp8" if o.dtype == torch.float8_e4m3fn else "uint8"
        out[kind] += 1
        out[kind + "_bytes"] += o.untyped_storage().nbytes()
        if len(out["held_by"]) < 4:
            out["held_by"].append([kind, list(o.shape), sorted({type(r).__name__ for r in gc.get_referrers(o)})[:6]])
    return out


def hook_runner(hooks, install: Install) -> None:
    import importlib
    for mod in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
        try:
            cls = importlib.import_module(mod).GPUModelRunner
        except (ImportError, AttributeError):
            continue

        def make(orig):
            def load_model(self, *a, **kw):
                r = orig(self, *a, **kw)
                install(self)
                return r
            return load_model
        hooks.wrap(cls, "load_model", make)


class Window:
    """cudaProfilerStart/Stop around `steps` steady decode steps of each point (for `ncu --profile-from-start off`), each
    window opened by torch's ``spin_kernel`` (`torch.cuda._sleep`) so the profile splits by point."""

    def __init__(self, steps: int) -> None:
        self.steps, self.point, self.run, self.opened, self.on = steps, None, 0, False, False
        self.windows: list[dict] = []

    def begin(self, b: int, ctx: int) -> None:
        self.point, self.run, self.opened = {"batch": b, "context": ctx}, 0, False

    def __call__(self, n: int, r: int) -> None:
        if not self.steps or self.point is None:
            return
        b = self.point["batch"]
        self.run = self.run + 1 if n == r == b else 0
        if not self.opened and self.run == 4:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            torch.cuda._sleep(1000)
            self.opened = self.on = True
        elif self.on and (self.run == 0 or self.run >= 4 + self.steps):
            self.end()

    def end(self) -> None:
        if self.on:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            self.on = False
            self.windows.append(self.point)
        self.point = None


def recorder(hooks, sched, rec: list, window: Window) -> None:
    def make(orig):
        def update_from_output(so, mro):
            rec.append((time.perf_counter(), so.total_num_scheduled_tokens, len(so.num_scheduled_tokens)))
            window(so.total_num_scheduled_tokens, len(so.num_scheduled_tokens))
            return orig(so, mro)
        return update_from_output
    hooks.wrap(sched, "update_from_output", make)


def steady(rec: list, b: int) -> dict:
    """The point's decode steps: intervals between consecutive updates that each scheduled b tokens for b requests."""
    iv = [t1 - t0 for (t0, n0, r0), (t1, n1, r1) in itertools.pairwise(rec) if n0 == r0 == b and n1 == r1 == b]
    if len(iv) < 4:
        return {"steps": len(iv)}
    iv = iv[1:]
    med = statistics.median(iv)
    q = statistics.quantiles(iv, n=10)
    return {"steps": len(iv), "step_ms": med * 1e3, "mean_ms": statistics.fmean(iv) * 1e3, "p10_ms": q[0] * 1e3,
            "p90_ms": q[-1] * 1e3, "tokens_per_s": b / med}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("stock", "plain", *ENCODED), required=True)
    ap.add_argument("--compiled", action="store_true",
                    help="vLLM's default compilation (torch.compile, piecewise and full CUDA graphs) instead of mode 0")
    ap.add_argument("--no-autotune", action="store_true",
                    help="compiled without any choice made by timing: no max-autotune or coordinate-descent tuning (vLLM "
                         "turns both on for each capture size), no combo-kernel benchmark, inductor's deterministic mode; "
                         "the compiled step is the same in every process (the compiled gate)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lib", help="a build of ve.cu (every arm but stock; the wide and tensor arms reduce with it)")
    ap.add_argument("--wide-lib", help="a build of wide.cu (wide arm)")
    ap.add_argument("--tensor-lib", help="a build of codec_tensor's wt.cu (tensor arm)")
    ap.add_argument("--tag", action="store_true",
                    help="wide and tensor arms: tag every block of the store the GEMMs read (ro-blake3/v1)")
    ap.add_argument("--sketch", type=int, default=0, choices=(0, 16, 64),
                    help="tensor arm at k = 1 (k_s 16 or 64) or wide arm (k_s 16): tag the store under ro-sketch-blake3/v0 "
                         "at this k_s instead")
    ap.add_argument("--sketch-t", type=int, default=8, help="tasks a sketch vector (codec_tensor's --sketch-t)")
    ap.add_argument("--commit", action="store_true",
                    help="with --sketch: commit each decode step's tags (leaves and root): the tensor arm's after its "
                         "last linear, the wide arm's as the step goes")
    ap.add_argument("--python-tags", type=int, default=3, help="sampled tags a layer checked against the reference")
    ap.add_argument("--gate-reps", type=int, default=1,
                    help="launches of each of the arm's GEMM configs a capture size in the gates, each bit for bit")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--contexts", default="512,2048,8192")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--mem", type=float, default=0.75)
    ap.add_argument("--kv-cache-dtype", default="auto", help="vLLM's kv_cache_dtype (fp8: twice the tokens a GB)")
    ap.add_argument("--max-batched-tokens", type=int, default=16384)
    ap.add_argument("--select-trials", type=int, default=5)
    ap.add_argument("--native-trials", type=int, default=10,
                    help="stock arm: timings of one native pass of the weights at each batch (0: none)")
    ap.add_argument("--python-words", type=int, default=16)
    ap.add_argument("--gate-tokens", type=int, default=16)
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="open a profiler window (cudaProfilerStart/Stop) over this many steady decode steps a point")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    if max(batches) > max(CAPTURE):
        ap.error(f"a decode batch is at most the largest capture size, {max(CAPTURE)}")
    if args.compiled:
        # the cache key doesn't cover the installed linears: a cached stock graph would bypass them
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        if args.no_autotune:
            os.environ["VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE"] = "0"
            os.environ["VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING"] = "0"
    elif args.no_autotune:
        ap.error("--no-autotune is a compiled mode")

    import vllm
    from verity_vllm.engine.engine_profile import get_model_runner, get_scheduler
    from verity_vllm.engine.hooks import Hooks
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    hooks = Hooks(tag="ra-vllm-encoded")
    install = Install(args)
    if args.arm != "stock":
        if not args.lib or (args.arm == "wide") != bool(args.wide_lib) or (args.arm == "tensor") != bool(args.tensor_lib):
            ap.error("--lib is required for every arm but stock, --wide-lib for the wide arm only and --tensor-lib for the "
                     "tensor arm only")
        if args.tag and args.arm not in ("wide", "tensor"):
            ap.error("--tag is the wide and tensor arms'")
        hook_runner(hooks, install)
    max_tokens_cap = args.steps + math.ceil(max(batches) * max(contexts) / args.max_batched_tokens) + 8
    cc = {"cudagraph_capture_sizes": CAPTURE}
    if not args.compiled:
        cc.update({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    elif args.no_autotune:
        # vLLM's combo kernels, without the benchmark that picks them; inductor's deterministic mode skips every other
        # on-device benchmark that affects numerics (reduction configs among them)
        cc["inductor_compile_config"] = {"deterministic": True, "combo_kernels": True, "benchmark_combo_kernel": False}
    # every arm's engine holds the largest capture size's requests: the gate prompts are scheduled the same in an arm timing
    # fewer batches, and the warmup's dummy batch (max_num_batched_tokens) fits max_num_seqs · max_model_len
    kw = {"model": args.model, "quantization": "fp8", "max_num_seqs": max(CAPTURE), "enable_prefix_caching": False, "seed": 0,
          "max_model_len": max(contexts) + max_tokens_cap + 64, "max_num_batched_tokens": args.max_batched_tokens,
          "gpu_memory_utilization": args.mem, "kv_cache_dtype": args.kv_cache_dtype, "enforce_eager": False,
          "compilation_config": cc}
    if args.revision:
        kw["revision"] = args.revision
    t0 = time.perf_counter()
    llm = LLM(**kw)
    startup_s = time.perf_counter() - t0
    log(f"engine up in {startup_s:.1f} s")
    runner = get_model_runner(llm)
    model = unwrap(runner.model)
    hf = llm.llm_engine.vllm_config.model_config.hf_config
    cache = llm.llm_engine.vllm_config.cache_config
    kv_tokens = int(cache.num_gpu_blocks) * int(cache.block_size)
    attn = sorted({type(getattr(m, "attn_backend", None)).__name__ for _, m in model.named_modules()
                   if getattr(m, "attn_backend", None) is not None})
    rec: list = []
    window = Window(args.profile_steps)
    recorder(hooks, get_scheduler(llm), rec, window)

    props = torch.cuda.get_device_properties(0)
    result = {
        "arm": args.arm, "compiled": args.compiled, "autotune": args.compiled and not args.no_autotune,
        "k": args.k if args.arm in ENCODED else 0,
        "codec_tag": TAG if args.tag else None, "model": args.model,
        "revision": args.revision,
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": props.name,
        "sms": props.multi_processor_count, "startup_s": startup_s, "kv_tokens": kv_tokens,
        "kv_cache_dtype": str(cache.cache_dtype), "attention": attn, "compilation": cc,
        "max_batched_tokens": args.max_batched_tokens, "install": install.record,
        "model_shape": {"layers": hf.num_hidden_layers, "hidden": hf.hidden_size, "heads": hf.num_attention_heads,
                        "kv_heads": hf.num_key_value_heads, "head_dim": getattr(hf, "head_dim", None) or
                        hf.hidden_size // hf.num_attention_heads, "intermediate": hf.intermediate_size,
                        "vocab": hf.vocab_size, "tied": bool(getattr(hf, "tie_word_embeddings", False))},
        "gate": [], "points": [], "profile_steps": args.profile_steps, "profiled": window.windows,
    }
    if args.arm == "stock":
        layers = fp8_linears(model)
        result["install"] = {"arm": args.arm, "layers": len(layers), "plain_bytes": sum(l.weight.numel() for _, l in layers),
                             "fp8_kernel": type(layers[0][1].quant_method.fp8_linear).__name__,
                             "activation_quant": str(layers[0][1].quant_method.activation_quant_key)}
    out = Path(args.out)

    def save() -> None:
        out.write_text(json.dumps(result, indent=1))

    sp = SamplingParams(temperature=0.0, max_tokens=args.gate_tokens, logprobs=1, seed=0)
    for o in llm.generate(GATE_PROMPTS, sp, use_tqdm=False):
        c = o.outputs[0]
        result["gate"].append({"prompt": o.prompt, "token_ids": list(c.token_ids),
                               "logprobs": [float(lp[t].logprob).hex() for t, lp in zip(c.token_ids, c.logprobs)]})
    save()
    log("gate generation done")

    rng = random.Random(0)
    warm = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(512)])]
    llm.generate(warm, SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, detokenize=False), use_tqdm=False)
    for ctx in contexts:
        for b in batches:
            need = b * (ctx + args.steps + math.ceil(b * ctx / args.max_batched_tokens) + 8)
            point = {"batch": b, "context": ctx, "kv_needed": need}
            if need > kv_tokens:
                point["skipped"] = f"KV cache holds {kv_tokens} tokens, the point needs {need}"
                result["points"].append(point)
                continue
            prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(ctx)]) for _ in range(b)]
            mt = args.steps + math.ceil(b * ctx / args.max_batched_tokens) + 8
            rec.clear()
            window.begin(b, ctx)
            t = time.perf_counter()
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True, detokenize=False),
                         use_tqdm=False)
            window.end()
            point.update(steady(rec, b), wall_s=time.perf_counter() - t, max_tokens=mt, updates=len(rec))
            result["points"].append(point)
            save()
            log("point", json.dumps(point))
    if args.arm == "stock" and args.native_trials:
        result["native_pass"] = native_pass(fp8_linears(model), batches, args.native_trials)
    hooks.uninstall()
    if install.hooks is not None:
        install.hooks.uninstall()
    save()


if __name__ == "__main__":
    main()
