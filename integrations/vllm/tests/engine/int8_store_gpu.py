"""The int8 row's store pieces on one GPU, bit for bit: the credited cache read through paged.cu's pg_window
(`engine.int8_window`), weights decoded at use (`store.decode_at_use`) and the int8 cache as the store lays it out
(`store.int8_kv`).  One process writes `<out>/int8_store_gpu.json` and exits 0 only when every check holds.

  kernel   a synthetic [pages][slots][KV heads][K ‖ V] cache, prefills and decode rows: `int8_ops.paged_attention` on the plain
           cache against `int8_ops.credited_attention` as a copy; encoded with every slot flagged; encoded under the seed's key;
           encoded under per-slot bound offsets with half the slots plaintext; and with the step's writes left plaintext and
           unflagged (the read flags them from slot_mapping).  Without the flags, or under the wrong key, it differs.
  layout   every layer of the int8 Llama-3.1-8B's cache as 4 KiB token-layer blocks (`token_layer_layout`), and the
           session's KV check (`kv_refusal`)
  head     the LM head's GEMM over row pieces of its weight (the pieces `DecodeAtUse` decodes into its scratch) against the
           whole GEMM, at M = 1 .. 256 rows
  engine   greedy generations of the plain engine (token ids, top-5 logprobs, the final norm's output hashed every call)
           against: the credited read as a copy; encoded with every slot flagged; encoded for real (each step's written slots
           encoded in place under their bound offsets and unflagged before the next step, as the session does when an
           append's coin arrives); every weight encoded in place and decoded at use, the embedding's rows and the LM head's
           pieces through a scratch the size of the largest layer module; both together

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared \\
        -DKV_FIELD=1024 -DKV_MASK=1 -DKV_FOLD6=1 -DKV_QHEADS=32 -o /tmp/paged.so ../../benchmarks/vllm_encoded/paged.cu
    $PY tests/engine/int8_store_gpu.py --paged-so /tmp/paged.so --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import torch

from verity_vllm import config
from verity_vllm.engine import int8_ops
from verity_vllm.engine.int8_window import CreditedRead, LayerKeys
from verity_vllm.protocols.poub.remote import session_module

RECORD = "int8_store_gpu.json"
ROUNDS = 3
PROMPTS = ["The capital of France is", "One, two, three,", "def fibonacci(n):"]
HEAD_ROWS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256)


def bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


class Codec:
    """paged.cu's build at P1024 under the seed's key: its shape, token-layer words and in-place encoders."""

    def __init__(self, so: str) -> None:
        self.P = session_module("paged", config.POUB_SESSION.parent / "vllm_encoded")
        self.lib = self.P.PagedLib("p1024", so)
        self.P.KB.set_key(self.lib, self.P.KB.lib_key(self.lib, ROUNDS))
        self.F = self.lib.field
        self.W, self.NL = self.P.K.token_layer_words(self.F), self.F.limbs
        self.G, self.D = self.lib.shape["kv_heads"], self.lib.shape["head_dim"]
        self.TL = 2 * self.G * self.D * 2

    def encode_slots(self, cache: torch.Tensor, slots, store_base: int, bj: torch.Tensor | None) -> None:
        """Each slot's token-layer encoded in place at its codec index (physical order: word order 1)."""
        base, st = cache.data_ptr(), self.P.stream(torch)
        for s in slots:
            p = base + s * self.TL
            if bj is None:
                self.lib.call("kv_encode", p, p, self.W, store_base + s * self.W, ROUNDS, 0, st)
            else:
                self.lib.call("kv_encode_bound", p, p, self.W, store_base + s * self.W, ROUNDS, bj.data_ptr() + s * self.NL * 4, 0, st)

    def offsets(self, rng: random.Random, n: int) -> torch.Tensor:
        return self.P.offsets(torch, [rng.randrange(self.F.p) for _ in range(n)], self.F)


def kernel(c: Codec) -> dict[str, Any]:
    rng = random.Random(5)
    torch.manual_seed(5)
    G, D = c.G, c.D
    pages, bs, H, base = 96, 16, 32, 4099 * c.W
    nslots = pages * bs
    phys = (torch.randn(pages, bs, G, 2 * D, device="cuda") * 2).to(torch.bfloat16)
    lens = [1, 15, 16, 17, 64, 65, 300, 511, 7, 33]
    rows = [n if i % 3 == 0 else 1 for i, n in enumerate(lens)]
    perm, bt, at = rng.sample(range(pages), pages), torch.zeros(len(lens), 40, dtype=torch.int32), 0
    for i, n in enumerate(lens):
        for b in range(-(-n // bs)):
            bt[i, b], at = perm[at], at + 1
    bt = bt.cuda()
    ks = (torch.rand(G, device="cuda") * 0.02 + 0.01).float()
    q = torch.randn(sum(rows), H, D, device="cuda").to(torch.bfloat16)
    qsl = torch.tensor([0, *torch.tensor(rows).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    sl = torch.tensor(lens, dtype=torch.int32, device="cuda")

    def plain_out(view):
        out = torch.full_like(q, float("nan"))
        kc, vc = view.transpose(1, 2).split(D, dim=-1)
        int8_ops.paged_attention(q, kc, vc, qsl, sl, bt, ks, D ** -0.5, out)
        return out

    def credited(view, keys, encoded=True, slot_mapping=None):
        r = CreditedRead(c.P, c.lib, lambda layer: keys, encoded=encoded, rounds=ROUNDS)
        out = torch.full_like(q, float("nan"))
        md = types.SimpleNamespace(query_start_loc=qsl, seq_lens=sl, block_table=bt, slot_mapping=slot_mapping)
        int8_ops.credited_attention(r, q, view, 0, md, G, ks, D ** -0.5, out)
        torch.cuda.synchronize()
        return out

    view = phys.transpose(1, 2)
    want = plain_out(view)
    out: dict[str, Any] = {"copy": bits(credited(view, LayerKeys(base), encoded=False), want)}
    out["flagged_all"] = bits(credited(view, LayerKeys(base, plain=torch.ones(nslots, dtype=torch.uint8, device="cuda"))), want)
    seed_enc = phys.clone()
    c.encode_slots(seed_enc, range(nslots), base, None)
    out["seed_key"] = bits(credited(seed_enc.transpose(1, 2), LayerKeys(base)), want)
    out["seed_key_encoded_differs"] = not bits(plain_out(seed_enc.transpose(1, 2)), want)
    bj = c.offsets(rng, nslots)
    flags = torch.tensor([rng.random() < 0.5 for _ in range(nslots)], dtype=torch.uint8, device="cuda")
    mix = phys.clone()
    c.encode_slots(mix, [s for s in range(nslots) if not flags[s]], base, bj)
    out["bound_mixed"] = bits(credited(mix.transpose(1, 2), LayerKeys(base, bj=bj, plain=flags.clone())), want)
    out["bound_unflagged_differs"] = not bits(credited(mix.transpose(1, 2), LayerKeys(base, bj=bj)), want)
    out["seed_key_on_bound_differs"] = not bits(credited(mix.transpose(1, 2), LayerKeys(base, plain=flags.clone())), want)
    last = [int(bt[i, (n - 1) // bs]) * bs + (n - 1) % bs for i, n in enumerate(lens)]
    wrote, fl = mix.clone(), flags.clone()
    flat_w, flat_p = wrote.view(nslots, G, 2 * D), phys.view(nslots, G, 2 * D)
    for s in last:
        flat_w[s], fl[s] = flat_p[s], 0
    sm = torch.tensor(last + [-1, -1], dtype=torch.int64, device="cuda")
    out["writes_flagged"] = (bits(credited(wrote.transpose(1, 2), LayerKeys(base, bj=bj, plain=fl), slot_mapping=sm), want)
                             and all(int(fl[s]) == 1 for s in last))
    return {"lens": lens, "rows": rows, "checks": out, "ok": all(out.values())}


def head_pieces(weight: torch.Tensor, piece_rows: int) -> dict[str, Any]:
    """F.linear over row pieces of the LM head's weight against the whole, at each M of HEAD_ROWS."""
    g = torch.Generator(device="cuda").manual_seed(13)
    V, out = weight.shape[0], {}
    for m in HEAD_ROWS:
        x = torch.randn(m, weight.shape[1], device="cuda", generator=g).to(torch.bfloat16)
        whole = torch.nn.functional.linear(x, weight)
        parts = torch.cat([torch.nn.functional.linear(x, weight[a:min(V, a + piece_rows)]) for a in range(0, V, piece_rows)], -1)
        out[m] = bits(parts, whole)
    return {"piece_rows": piece_rows, "equal": out, "ok": all(out.values())}


def engine(c: Codec) -> dict[str, Any]:
    from vllm import SamplingParams

    from tests.engine.int8_exact_gpu import scratch_checkpoints
    from verity_vllm.config import checkpoint_entries
    from verity_vllm.engine.build import build_engine
    from verity_vllm.engine.int8_attention_vllm import install_credited_read
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocols.poub.decode_at_use import DecodeAtUse, encoded_weights, kv_decoder
    from verity_vllm.protocols.poub.int8_kv import kv_refusal, layer_keys, token_layer_layout
    from verity_vllm.protocols.poub.interface import Layout, RegionInfo

    entry = next(e for e in checkpoint_entries("manifests/checkpoints.json") if e.get("role") == "LLAMA31_8B_INT8")
    target = TargetProfile.from_device().with_overrides(int8_exact="all", async_scheduling=False)
    with tempfile.TemporaryDirectory() as scratch:
        eng = build_engine(scratch_checkpoints("manifests/checkpoints.json", entry, Path(scratch)), case="LLAMA31_8B_INT8",
                           target=target, max_num_seqs=4, engine_args={"num_gpu_blocks_override": 288, "enable_prefix_caching": False})
    r, model, W = eng.runner, eng.model, c.W
    caches = list(r.kv_caches)
    pages, bs = caches[0].shape[0], int(r.cache_config.block_size)
    nslots, L = pages * bs, len(caches)
    out: dict[str, Any] = {"layers": L, "pages": pages, "slots": bs, "shape": list(caches[0].shape), "stride": list(caches[0].stride())}
    lays = [token_layer_layout(t, pages, bs, c.D) for t in caches]
    out["store_layout"] = {"slot_bytes": lays[0].slot_bytes, "planes": lays[0].planes, "slot_major": lays[0].slot_major,
                           "all_layers": all(x == lays[0] for x in lays)}
    out["kv_refusal"] = kv_refusal(eng.kwargs, caches)

    cap: list[str] = []
    model.model.norm.register_forward_hook(
        lambda m, a, o: cap.append(hashlib.sha256((o[0] if isinstance(o, tuple) else o).contiguous().view(torch.uint8).cpu().numpy()
                                                  .tobytes()).hexdigest()[:16]))

    def gen() -> dict[str, Any]:
        cap.clear()
        outs = eng.llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=24, logprobs=5))
        return {"tokens": [list(o.outputs[0].token_ids) for o in outs],
                "logprobs": [[{int(k): float(v.logprob) for k, v in d.items()} for d in o.outputs[0].logprobs] for o in outs],
                "norm": list(cap), "text": [o.outputs[0].text for o in outs]}

    def same(a: dict, b: dict) -> dict[str, Any]:
        return {"tokens": a["tokens"] == b["tokens"], "logprobs": a["logprobs"] == b["logprobs"], "norm": a["norm"] == b["norm"],
                "norm_calls": len(b["norm"])}

    plain = gen()
    out["plain_text"] = plain["text"]
    rng = random.Random(11)
    bases = {li: (7 + li * nslots) * W for li in range(L)}
    modes: dict[str, Any] = {}
    install_credited_read(model, CreditedRead(c.P, c.lib, layer_keys(bases), encoded=False, rounds=ROUNDS))
    modes["credited_copy"] = same(plain, gen())
    ones = [torch.ones(nslots, dtype=torch.uint8, device="cuda") for _ in range(L)]
    install_credited_read(model, CreditedRead(c.P, c.lib, layer_keys(bases, plain=lambda li: ones[li]), rounds=ROUNDS))
    modes["credited_flagged_all"] = same(plain, gen())
    bjs = [c.offsets(rng, nslots) for _ in range(L)]
    flags = [torch.zeros(nslots, dtype=torch.uint8, device="cuda") for _ in range(L)]

    class Settling(CreditedRead):
        """Before each layer's window, every slot flagged (an earlier step's write) encoded in place under its bound offset and
        unflagged, but for a slot this step's write refilled: the session's encode once an append's coin arrives."""
        settled = 0

        def read(self, kv_cache, li, block_table, slot_mapping, *rest):
            due = flags[li].clone()
            if slot_mapping is not None:
                due[slot_mapping[slot_mapping >= 0].long()] = 0
            slots = due.nonzero().flatten().tolist()
            c.encode_slots(caches[li], slots, bases[li], bjs[li])
            flags[li].zero_()
            self.settled += len(slots)
            super().read(kv_cache, li, block_table, slot_mapping, *rest)

    settling = Settling(c.P, c.lib, layer_keys(bases, bj=lambda li: bjs[li], plain=lambda li: flags[li]), rounds=ROUNDS)
    install_credited_read(model, settling)
    modes["credited_encoded"] = {**same(plain, gen()), "slots_encoded": settling.settled, "steps": settling.steps}
    install_credited_read(model, None)

    regions, first = [], 0
    for name, p in model.named_parameters():
        blocks = p.numel() * p.element_size() // 4096
        if blocks:
            regions.append(RegionInfo(name, "hbm", "weights", first // W, blocks, 4096, first, c.F.word_bytes, c.F.name, "none"))
            first += blocks * W
    weights = encoded_weights(Layout(tuple(regions), first // W, {"hbm": first * c.F.word_bytes}), model)
    hooks = DecodeAtUse(model, weights, kv_decoder(c.lib, ROUNDS, c.F.word_bytes))
    hw = model.lm_head.weight
    out["head"] = head_pieces(hw.data, hooks.scratch.numel() // (hw.shape[1] * hw.element_size()))
    for w in weights:
        p = w.param
        before = hashlib.sha256(p.data.view(-1).view(torch.uint8)[:4096].cpu().numpy().tobytes()).hexdigest()
        c.lib.call("kv_encode", p.data_ptr(), p.data_ptr(), w.blocks * W, w.first_word, ROUNDS, 0, c.P.stream(torch))
        torch.cuda.synchronize()
        if before == hashlib.sha256(p.data.view(-1).view(torch.uint8)[:4096].cpu().numpy().tobytes()).hexdigest():
            raise RuntimeError(f"{w.name}: encoding left its first block as it was")
    hooks.install()
    out["weights"] = {"registered": len(weights), "params": sum(1 for _ in model.parameters()), "blocks": first // W,
                      "modules": len(hooks.modules), "scratch_bytes": hooks.scratch.numel(),
                      "pieces": sorted(k for k, _, _ in hooks.pieces.values()), "vocabulary_bytes": hw.numel() * hw.element_size(),
                      "int8": sum(1 for w in weights if w.param.dtype == torch.int8)}
    modes["weights_decoded_at_use"] = {**same(plain, gen()), "module_calls": hooks.calls}
    for li in range(L):
        flags[li].zero_()
    settling2 = Settling(c.P, c.lib, layer_keys(bases, bj=lambda li: bjs[li], plain=lambda li: flags[li]), rounds=ROUNDS)
    install_credited_read(model, settling2)
    modes["weights_and_kv_encoded"] = {**same(plain, gen()), "slots_encoded": settling2.settled}
    install_credited_read(model, None)
    hooks.remove()
    out["modes"] = modes
    out["ok"] = (out["kv_refusal"] is None and out["weights"]["pieces"] == ["head", "rows"] and out["head"]["ok"]
                 and out["weights"]["scratch_bytes"] < out["weights"]["vocabulary_bytes"]
                 and out["store_layout"] == {"slot_bytes": 4096, "planes": 1, "slot_major": True, "all_layers": True}
                 and all(m["tokens"] and m["logprobs"] and m["norm"] for m in modes.values()))
    return out


def run(a: argparse.Namespace) -> dict[str, Any]:
    c = Codec(a.paged_so)
    res: dict[str, Any] = {"lib": c.lib.shape, "token_layer_words": c.W, "limbs": c.NL}
    res["kernel"] = kernel(c)
    print("INT8_STORE kernel " + json.dumps(res["kernel"]), flush=True)
    res["engine"] = engine(c)
    print("INT8_STORE engine " + json.dumps(res["engine"]), flush=True)
    res["ok"] = res["kernel"]["ok"] and res["engine"]["ok"]
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the int8 row's store pieces (credited read, decode at use, layout) on this GPU, bit for bit")
    ap.add_argument("--paged-so", required=True, help="benchmarks/vllm_encoded/paged.cu built at KV_FIELD=1024 KV_MASK=1 KV_QHEADS=32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, RECORD).write_text(json.dumps(res, indent=1))
    print(f"INT8_STORE {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
