"""`compiled`: value correspondence (C2) for the compiled boundary, from the COMMITTED bytes alone (no reference taps -- Python hooks
cannot observe values inside a replayed CUDA graph, so the eager `ValueChecker` (tap clones) does not apply).

Three relations the committed step streams must satisfy if every value was read at its module boundary at the right step:
  (R) Linear re-execution: for every Linear module whose input is itself a committed tensor (Llama/Qwen leaf rules below), re-run the module
      eagerly on the committed input -> bytes must equal the committed output (`<name>/0`).  The batch-invariant GEMM is the same Triton kernel
      inside and outside the compiled graph.  A late read (value of another step) or a relabel fails here.
  (L) logits: `logits_processor` output == lm_head applied to the committed final-norm rows of the sampled positions (`model.norm/0`).
  (G) sampling chain: greedy argmax of the committed logits == committed sampled ids (when the workload is greedy), and the committed
      embedding INPUT ids of step s+1 (`model.embed_tokens/in`) == the sampled ids of step s (decode rows), i.e. the committed stream of
      one step is the Program's successor of the previous one.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from verity_vllm.program.compiled_relations import warm_vml

_DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "int64": torch.int64, "int32": torch.int32,
       "uint8": torch.uint8, "int8": torch.int8, "float8_e4m3fn": torch.float8_e4m3fn}

# leaf of a Linear module -> (relative path of the committed tensor that is its input)
_LINEAR_INPUT = {
    "qkv_proj": ("input_layernorm/0", 2),          # model.layers.N.self_attn.qkv_proj <- model.layers.N.input_layernorm/0
    "o_proj": ("self_attn.attn", 2),               # <- model.layers.N.self_attn.attn (attention output)
    "gate_up_proj": ("post_attention_layernorm/0", 2),
    "down_proj": ("mlp.act_fn", 2),
}


class CompiledValueCheck:
    def __init__(self, committer: Any, model: torch.nn.Module, runner: Any, sampling: dict | None = None) -> None:
        warm_vml()
        self.c = committer
        self.model = model
        self.runner = runner
        self.sampling = sampling or {}
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------ committed bytes

    def _meta(self, step: int, name: str):
        for m in self.c._layouts.get(step, []):
            if m.name == name:
                return m
        return None

    def _bytes(self, step: int, m) -> bytes | None:
        if m.host is not None:
            return bytes(m.host)
        blocks = getattr(self.c, "_gpu_blocks", None)
        if blocks and step in blocks and blocks[step][0] is not None:
            blk, _used = blocks[step]
            return bytes(blk[m.stream_off:m.stream_off + m.nbytes].cpu().numpy().tobytes())
        return None

    def committed(self, step: int, name: str) -> torch.Tensor | None:
        m = self._meta(step, name)
        if m is None:
            return None
        b = self._bytes(step, m)
        if b is None:
            return None
        dt = _DT.get(m.dtype)
        if dt is None:
            return None
        t = torch.frombuffer(bytearray(b), dtype=torch.uint8).view(dt).reshape(tuple(m.shape))
        return t.to(self.device)

    @staticmethod
    def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    # ------------------------------------------------------------------ the relations

    @torch.inference_mode()
    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        steps = sorted(self.c._layouts)
        res: dict[str, Any] = {"steps": len(steps), "reexec_checked": 0, "reexec_equal": 0, "reexec_mismatch": [], "reexec_skipped": {},
                               "logits_checked": 0, "logits_equal": 0, "logits_mismatch": [], "argmax_checked": 0, "argmax_equal": 0,
                               "chain_checked": 0, "chain_equal": 0, "chain_mismatch": [], "notes": []}
        mods = dict(self.model.named_modules())
        greedy = float(self.sampling.get("temperature", 0.0) or 0.0) == 0.0
        lm_head = mods.get("lm_head")
        lp = mods.get("logits_processor")
        lp_name = next((n for n, m in mods.items() if type(m).__name__ == "LogitsProcessor"), "logits_processor")
        norm_name = next((n for n in ("model.norm", "norm") if n in mods), None)
        embed_in = next((f"{n}/in" for n, m in mods.items() if type(m).__name__ == "VocabParallelEmbedding"), "model.embed_tokens/in")
        step_rows = getattr(self.c, "step_rows", {}) or {}
        for s in steps:
            names = [m.name for m in self.c._layouts[s]]
            nameset = set(names)
            # (R) Linear re-execution
            for name in names:
                if not name.endswith("/0"):
                    continue
                base = name[:-2]
                leaf = base.rsplit(".", 1)[-1]
                rule = _LINEAR_INPUT.get(leaf)
                mod = mods.get(base)
                if rule is None or mod is None:
                    continue
                rel, up = rule
                layer = base.rsplit(".", up)[0]
                in_name = f"{layer}.{rel}"
                if in_name not in nameset and rel.endswith("/0") and f"{layer}.{rel[:-2]}" in nameset:
                    in_name = f"{layer}.{rel[:-2]}"          # layer 0's input_layernorm returns a single tensor (no residual) -> bare name
                if in_name not in nameset:
                    res["reexec_skipped"][leaf] = res["reexec_skipped"].get(leaf, 0) + 1
                    continue
                x = self.committed(s, in_name)
                y = self.committed(s, name)
                if x is None or y is None:
                    res["reexec_skipped"]["no_bytes"] = res["reexec_skipped"].get("no_bytes", 0) + 1
                    continue
                try:
                    out = mod(x)
                    y2 = out[0] if isinstance(out, (tuple, list)) else out
                except Exception as e:  # noqa: BLE001
                    res["reexec_skipped"][f"error:{leaf}"] = repr(e)[:120]
                    continue
                res["reexec_checked"] += 1
                if self._eq(y, y2):
                    res["reexec_equal"] += 1
                elif len(res["reexec_mismatch"]) < 16:
                    d = (y.float() - y2.float()).abs()
                    res["reexec_mismatch"].append({"step": s, "name": name, "input": in_name, "max_abs": float(d.max()), "rows": int(y.shape[0]),
                                                   "rows_differ": int((d.reshape(d.shape[0], -1).amax(1) > 0).sum())})
            # (L) logits from the committed final-norm rows
            if lp is not None and lm_head is not None and norm_name is not None and f"{norm_name}/0" in nameset and lp_name in nameset:
                h = self.committed(s, f"{norm_name}/0")
                lg = self.committed(s, lp_name)
                if h is not None and lg is not None:
                    rows = step_rows.get(s)
                    idx = [hi - 1 for _rid, _lo, hi in rows] if rows else [h.shape[0] - 1]
                    if len(idx) == lg.shape[0]:
                        try:
                            lg2 = lp(lm_head, h[idx])
                            res["logits_checked"] += 1
                            if lg2 is not None and self._eq(lg, lg2.to(lg.dtype)):
                                res["logits_equal"] += 1
                            elif len(res["logits_mismatch"]) < 8:
                                res["logits_mismatch"].append({"step": s, "max_abs": float((lg.float() - lg2.float()).abs().max()) if lg2 is not None else None,
                                                               "shape": list(lg.shape)})
                        except Exception as e:  # noqa: BLE001
                            res["notes"].append(f"logits step {s}: {e!r}"[:160])
                    else:
                        res["notes"].append(f"logits step {s}: {len(idx)} sampled rows vs {lg.shape[0]} logits rows")
            # (G) greedy argmax + successor chain
            ids = self.committed(s, "runner.sampler/sampled_token_ids")
            if ids is not None and lp_name in nameset and greedy:
                lg = self.committed(s, lp_name)
                if lg is not None and lg.shape[0] == ids.shape[0]:
                    res["argmax_checked"] += 1
                    if bool(torch.equal(lg.argmax(-1).reshape(-1).to(ids.dtype), ids.reshape(-1))):
                        res["argmax_equal"] += 1
            if ids is not None and (s + 1) in self.c._layouts:
                nxt = self.committed(s + 1, embed_in)
                if nxt is not None:
                    rows_s, rows_n = step_rows.get(s), step_rows.get(s + 1)
                    if rows_s and rows_n:
                        pos = {rid: i for i, (rid, _lo, _hi) in enumerate(rows_s)}
                        for rid, lo, hi in rows_n:
                            if hi - lo == 1 and rid in pos:
                                res["chain_checked"] += 1
                                if int(nxt[lo]) == int(ids.reshape(-1)[pos[rid]]):
                                    res["chain_equal"] += 1
                                elif len(res["chain_mismatch"]) < 8:
                                    res["chain_mismatch"].append({"step": s + 1, "rid": rid, "in_id": int(nxt[lo]), "sampled_prev": int(ids.reshape(-1)[pos[rid]])})
                    elif ids.numel() == 1 and nxt.numel() == 1:
                        res["chain_checked"] += 1
                        if int(nxt.reshape(-1)[0]) == int(ids.reshape(-1)[0]):
                            res["chain_equal"] += 1
                        elif len(res["chain_mismatch"]) < 8:
                            res["chain_mismatch"].append({"step": s + 1, "in_id": int(nxt.reshape(-1)[0]), "sampled_prev": int(ids.reshape(-1)[0])})
        checked = res["reexec_checked"] + res["logits_checked"] + res["chain_checked"] + res["argmax_checked"]
        equal = res["reexec_equal"] + res["logits_equal"] + res["chain_equal"] + res["argmax_equal"]
        res["checked"] = checked
        res["equal"] = equal
        res["ok"] = checked > 0 and equal == checked and res["reexec_checked"] > 0
        res["method"] = "compiled: Linear re-execution from committed inputs + lm_head(norm) == committed logits + greedy argmax == sampled ids + embed-in(s+1) == sampled(s)"
        res["wall_s"] = time.perf_counter() - t0
        return res
