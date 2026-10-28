"""C2 of record for `Execution=compiled` rows (stratified sampled EXACT VU replay + ordered boundary linkage),
from the committed bytes alone.

VU = one Inductor kernel launch of P_compiled (compiled_relations.relation); its input boundaries are committed under
`c<piece>.<kernel>/<role>` (compiled_kernel_source.py), its registered weights come from the model.  For every launch of the sampled
steps the relation is evaluated from the committed inputs and compared BIT-FOR-BIT with the committed outputs.  Strata: the prefill
step, the first decode step, a middle and the last step, plus `n_random` steps drawn from a seed derived from the run root (fixed
before sampling) -- every kernel kind x every piece of those steps.  LINKAGE (ordered boundary): the embedding ids committed at step s
must equal the token the engine emitted at step s-1 (input prescribed by the previous step's designated output); greedy argmax of the
eager lm_head over the committed final hidden state at step s must equal the token emitted at step s.  A one-step-late drain keeps
every relation consistent and FAILS the linkage (post-overwrite read); an omitted kernel fails coverage (C1) and shows here as
`missing`.
"""
from __future__ import annotations

import re
import time
from typing import Any

import torch

from verity_vllm.check import compiled_autotune
from verity_vllm.commit import challenge as CH
from verity_vllm.program.compiled_relations import OUTPUT_ROLES, relation, roles_of, rms_eps_of, warm_vml

_DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "int64": torch.int64, "int32": torch.int32,
       "uint8": torch.uint8, "int8": torch.int8}
NORM_LABELS = ("r0=2048 warps=8 chunk=8 sep", "r0=2048 warps=8 chunk=8 fma")     # the order exact on every S2 launch (S1-S2.md)


_SAFE = re.compile(r"^[\d\s+\-*/()%,s]*$")


def _sym(expr: str, n: int) -> int:
    """A wrapper shape expression in s72 (`s72`, `2048`, `3072`, `32*((63 + s72) // 64)`) at s72 = n; integers only."""
    e = expr.replace("s72", str(n)).replace("s80", str(n))
    if not _SAFE.match(e.replace("min", "").replace("max", "")):
        raise ValueError(f"unsafe shape expression {expr!r}")
    return int(eval(e, {"__builtins__": {}}, {"min": min, "max": max}))  # noqa: S307 -- integers, s72 and arithmetic only (checked)


def apply_view(t: torch.Tensor, view: dict[str, Any] | None, n: int) -> torch.Tensor:
    """`reinterpret_tensor(buf, dims, strides, offset)` of a committed buffer (its logical contiguous layout) at s72 = n."""
    if not view:
        return t
    dims = [_sym(d, n) for d in view["dims"]]
    strides = [_sym(d, n) for d in view["strides"]]
    off = _sym(view["offset"], n)
    return t.contiguous().flatten().as_strided(dims, strides, off)


def check_edges(dataflow: dict[str, Any], committed, n_rows: int) -> dict[str, Any]:
    """The manifest's dataflow at one committed step: for every edge whose source is a launch output (same piece or across the eager
    attention boundary through the split graph), the committed INPUT bytes of the consuming launch == the committed OUTPUT bytes of the
    producing launch (with the wrapper's views applied) -- launch-to-launch and piece-to-piece linkage of the committed copies.
    `committed(piece, kernel, role)` -> tensor | None."""
    out = {"checked": 0, "equal": 0, "missing": 0, "external": 0, "mismatch": []}
    for piece, rows in enumerate(dataflow.get("edges") or []):
        for r in rows:
            for e in r["edges"]:
                src = e["src"]
                if src.get("kind") != "launch":
                    out["external"] += 1
                    continue
                a = committed(piece, r["kernel"], e["role"])
                b = committed(int(src["piece"]), src["kernel"], src["role"])
                if a is None or b is None:
                    out["missing"] += 1
                    continue
                out["checked"] += 1
                try:
                    bv = apply_view(b, src.get("view_ret"), n_rows)
                    bv = apply_view(bv, src.get("view"), n_rows) if src.get("view") and bv.numel() != a.numel() else bv
                    ok = bv.numel() == a.numel() and bool(torch.equal(bv.reshape(-1), a.reshape(-1).to(bv.dtype)))
                except Exception as ex:  # noqa: BLE001
                    ok = False
                    if len(out["mismatch"]) < 12:
                        out["mismatch"].append({"piece": piece, "kernel": r["kernel"], "role": e["role"], "error": repr(ex)[:160]})
                    continue
                if ok:
                    out["equal"] += 1
                elif len(out["mismatch"]) < 12:
                    d = (bv.reshape(-1) != a.reshape(-1).to(bv.dtype)) if bv.numel() == a.numel() else None
                    out["mismatch"].append({"piece": piece, "kernel": r["kernel"], "role": e["role"], "from": [src["piece"], src["kernel"], src["role"]],
                                            "numel": [int(a.numel()), int(bv.numel())], "n_diff": int(d.sum()) if d is not None else None})
    out["ok"] = out["checked"] > 0 and out["equal"] == out["checked"] and not out["mismatch"]
    return out


class CompiledKernelCheck:
    def __init__(self, committer: Any, model: torch.nn.Module, tokens: list[int] | dict[str, list[int]] | None = None, *, seed: int,
                 seed_source: str | None,
                 n_random: int = 4, pieces: int | None = None, request_rows: dict[int, list[tuple[str, int, int]]] | None = None,
                 arrivals: dict[str, int] | None = None, prompts: dict[str, list[int]] | None = None) -> None:
        warm_vml()
        self.c = committer
        self.model = model
        # [B>=2] tokens per request (`r<i>`), the engine's per-step row order (request, row lo, row hi) and the arrivals: the linkage ties
        # each request's rows of the batched step buffers to ITS tokens (ids(s) == token(k-1), argmax(final_s) == token(k), k = s - arrive);
        # at a request's prefill step its ids rows must equal the DECLARED prompt (workload manifest)
        if isinstance(tokens, dict):
            self.tokens_by_req: dict[str, list[int]] | None = {str(k): list(v) for k, v in tokens.items()} or None
        else:
            self.tokens_by_req = {"r0": list(tokens)} if tokens else None
        self.tokens = list(tokens) if isinstance(tokens, (list, tuple)) and tokens else (next(iter(self.tokens_by_req.values())) if self.tokens_by_req else None)
        self.request_rows = request_rows
        self._norm_labels: dict[tuple, tuple] = {}                            # (kernel, row width) -> the reduction orders found exact (learned once, then pinned)
        self.eps, self.eps_source = rms_eps_of(model)                          # the norm kernels' baked epsilon (Llama 1e-5, Qwen 1e-6), from the model
        self.arrivals = {str(k): int(v) for k, v in (arrivals or {}).items()}
        self.prompts = {str(k): list(v) for k, v in (prompts or {}).items()}
        self.seed = seed
        # the seed's provenance as a FACT the finalizer recomputes from this record's own run root
        self.seed_source = seed_source
        self.n_random = n_random
        self.device = next(model.parameters()).device
        self.mods = dict(model.named_modules())
        self.n_layers = len(self.model_layers())
        self.pieces = pieces or (self.n_layers + 1)

    def model_layers(self) -> list[torch.nn.Module]:
        for n, m in self.mods.items():
            if n.endswith("model.layers") or n == "model.layers" or n == "layers":
                return list(m)
        return [m for n, m in self.mods.items() if n.count(".layers.") == 1 and n.split(".layers.")[1].isdigit()]

    # ------------------------------------------------------------------ committed bytes

    def _metas(self, step: int) -> dict[str, Any]:
        return {m.name: m for m in self.c._layouts.get(step, [])}

    def _bytes(self, step: int, m) -> bytes | None:
        if m.host is not None:
            return bytes(m.host)
        blocks = getattr(self.c, "_gpu_blocks", None)
        if blocks and step in blocks and blocks[step][0] is not None:
            blk, _used = blocks[step]
            return bytes(blk[m.stream_off:m.stream_off + m.nbytes].cpu().numpy().tobytes())
        return None

    def committed(self, step: int, metas: dict[str, Any], name: str) -> torch.Tensor | None:
        m = metas.get(name)
        if m is None:
            return None
        b = self._bytes(step, m)
        dt = _DT.get(m.dtype)
        if b is None or dt is None:
            return None
        return torch.frombuffer(bytearray(b), dtype=torch.uint8).view(dt).reshape(tuple(m.shape)).to(self.device)

    # ------------------------------------------------------------------ registered weights of a launch (by piece + kernel)

    def _weights(self, piece: int, kernel: str, wargs=None) -> dict[int, torch.Tensor]:
        layers = self.model_layers()
        L = layers[piece - 1] if piece >= 1 else None
        Lnext = layers[piece] if piece < len(layers) else None
        w: dict[int, torch.Tensor] = {}
        if kernel.startswith("matmul_kernel_persistent"):
            suffix = kernel.rsplit("_", 1)[1]
            if piece == 0:
                lin = layers[0].self_attn.qkv_proj
            else:
                lin = {"0": L.self_attn.o_proj, "1": L.mlp.gate_up_proj, "2": L.mlp.down_proj, "3": Lnext.self_attn.qkv_proj if Lnext is not None else None}[suffix]
            w[1] = lin.weight.t()                                            # the wrapper passes W^T (reinterpret_tensor(arg, (K, N), (1, K)))
        elif "embedding_rms_norm" in kernel:
            emb = self.mods.get("model.embed_tokens") or next(m for n, m in self.mods.items() if n.endswith("embed_tokens"))
            w[1] = emb.weight
            w[2] = layers[0].input_layernorm.weight
        elif kernel.startswith("triton_red_fused_fused_add_rms_norm_t_0"):
            w[2] = L.post_attention_layernorm.weight
        elif kernel.startswith("triton_red_fused_fused_add_rms_norm_t_2"):
            w[3] = Lnext.input_layernorm.weight
        elif kernel.startswith("triton_red_fused_fused_add_rms_norm_2"):
            norm = self.mods.get("model.norm") or next(m for n, m in self.mods.items() if n.endswith("model.norm"))
            w[3] = norm.weight
        elif kernel.startswith("triton_poi_fused_mul_silu_slice"):
            pass                                                             # SiLU*up: no registered input
        elif kernel.startswith("triton_poi_fused_"):                         # RoPE (q, k of the NEXT layer's attention; piece 0: layer 0)
            attn = (layers[0] if piece == 0 else Lnext).self_attn
            wa = sorted(wargs) if wargs else [2]
            if len(wa) >= 2:                                                 # [Qwen2.5] RoPE(k) + QKV bias: bias first, cos|sin table second
                w[wa[0]] = attn.qkv_proj.bias
                w[wa[-1]] = attn.rotary_emb.cos_sin_cache
            else:
                w[wa[0]] = attn.rotary_emb.cos_sin_cache
        return w

    # ------------------------------------------------------------------ the check

    @torch.inference_mode()
    def run(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        steps = sorted(self.c._layouts)
        res: dict[str, Any] = {"method": "sampled-exact-replay (P_compiled kernel relations from committed bytes) + ordered boundary linkage",
                               "rms_eps": {"value": self.eps, "source": self.eps_source},
                               "population_steps": len(steps), "sampled_steps": [], "seed": self.seed, "seed_source": self.seed_source, "strata": "prefill, first/middle/last decode, "
                               f"{self.n_random} seeded random steps; every kernel kind x every piece of a sampled step",
                               "launches_checked": 0, "launches_exact": 0, "launches_missing": 0, "outputs_checked": 0, "outputs_exact": 0,
                               "by_kind": {}, "mismatch": [], "missing": [], "linkage": {"ids_checked": 0, "ids_equal": 0, "argmax_checked": 0, "argmax_equal": 0,
                                                                                         "mismatch": []}, "errors": []}
        if not steps:
            res["ok"] = False
            res["error"] = "no committed steps"
            return res
        sample = {steps[0]}
        if len(steps) > 1:
            sample |= {steps[1], steps[len(steps) // 2], steps[-1]}
        rng = CH.stream(self.seed)
        pool = [s for s in steps if s not in sample]
        sample |= set(rng.sample(pool, min(self.n_random, len(pool))))
        res["sampled_steps"] = sorted(sample)
        launch_table = None
        dataflow = (manifest or {}).get("program", {}).get("dataflow") if manifest else None
        if dataflow is not None and not dataflow.get("edges"):
            dataflow = None
        res["edges"] = {"declared": dataflow is not None, "wiring_declared": bool(dataflow and dataflow.get("wiring_declared")),
                        "n_edges": (dataflow or {}).get("n_edges"), "n_launch_edges": (dataflow or {}).get("n_launch_edges"),
                        "checked": 0, "equal": 0, "missing": 0, "external": 0, "steps": 0, "mismatch": []}
        role_maps = None
        if manifest:
            lt = manifest["program"]["launch_table"]
            launch_table = [lt["first"]] + [lt["middle"]] * (self.pieces - 2) + [lt["last"]]
            # the manifest's resolved role maps (by piece kind x launch site, from the generated kernels' signatures)
            # take precedence over the name-keyed table: a kernel name shared by two argument lists (Llama RoPE / Qwen RoPE+bias) is not re-roled
            rm = manifest["program"].get("role_maps")
            if rm:
                role_maps = [rm["first"]] + [rm["middle"]] * (self.pieces - 2) + [rm["last"]]
        res["role_maps_from_manifest"] = role_maps is not None

        def _roles_for(piece: int, kernel: str) -> tuple[dict[int, str], set[int]]:
            if role_maps is not None:
                e = next((x for x in role_maps[piece] if x.get("kernel") == kernel), None)
                if e is not None:
                    return {int(k): v for k, v in (e.get("roles") or {}).items()}, set(int(i) for i in (e.get("weight_args") or []))
            return roles_of(kernel)

        lm_head = self.mods.get("lm_head")
        for s in sorted(sample):
            metas = self._metas(s)
            for piece in range(self.pieces):
                launches = launch_table[piece] if launch_table else self._launches_from_names(metas, piece)
                for L in launches:
                    kernel = L["kernel"]
                    roles, wargs = _roles_for(piece, kernel)
                    if roles:
                        self._check_launch(s, piece, metas, kernel, roles, wargs, res)
            # dataflow linkage: the committed copies along every declared edge of P_compiled are equal at this step
            if dataflow is not None:
                memo: dict[tuple, Any] = {}

                def _got(piece: int, kernel: str, role: str, _s=s, _m=metas):
                    key = (piece, kernel, role)
                    if key not in memo:
                        memo[key] = self.committed(_s, _m, f"c{piece:02d}.{kernel}/{role}")
                    return memo[key]
                ids0 = _got(0, self._embed_kernel(metas), "ids") if self._embed_kernel(metas) else None
                n_rows = int(ids0.numel()) if ids0 is not None else 0
                er = check_edges(dataflow, _got, n_rows)
                eg = res["edges"]
                for key in ("checked", "equal", "missing", "external"):
                    eg[key] += er[key]
                eg["steps"] += 1
                for mm in er["mismatch"]:
                    if len(eg["mismatch"]) < 12:
                        eg["mismatch"].append({"step": s, **mm})
            # ordered boundary linkage (per request row range of the step's batched buffers)
            if self.tokens_by_req is not None and lm_head is not None:
                self._link_step(s, metas, lm_head, res)
        # the linkage runs over EVERY committed step, not only the sampled ones: at B>=2 an equal-shape
        # request swap relabels a whole step's rows together and the sampled replay passes by construction, so the per-(step, request)
        # attribution must be exhaustive -- every request's rows at every step tied to ITS tokens (ids in, argmax of the final hidden out).
        if self.tokens_by_req is not None and lm_head is not None:
            for s in steps:
                if s in sample:
                    continue
                try:
                    self._link_step(s, self._metas(s), lm_head, res)
                except Exception as e:                                        # noqa: BLE001 -- named, never silent
                    res["errors"].append({"step": s, "where": "linkage", "error": repr(e)[:300]})
                    if len(res["errors"]) > 32:
                        break
        lk = res["linkage"]
        lk["ok"] = ((lk["ids_checked"] > 0 or lk["argmax_checked"] > 0) and lk["ids_equal"] == lk["ids_checked"] and lk["argmax_equal"] == lk["argmax_checked"]
                    and lk.get("pos_equal", 0) == lk.get("pos_checked", 0) and not lk.get("unmapped_rows"))
        lk["attribution"] = {"steps_checked": lk.get("steps_checked", 0), "population_steps": len(steps),
                             "pairs_checked": lk.get("pairs_checked", 0), "pairs_total": lk.get("pairs_total", 0),
                             "coincident_pairs": int(lk.get("coincident_pairs", 0)), "coincident_sample": lk.get("coincident_sample", []),
                             "coincident_pairs_tokens": int(lk.get("coincident_pairs_tokens", 0)), "coincident_tokens_sample": lk.get("coincident_tokens_sample", []),
                             "position_gate": {"buffer": lk.get("position_gate"), "checked": lk.get("pos_checked", 0), "equal": lk.get("pos_equal", 0)},
                             "complete": bool(self.tokens_by_req is not None and lm_head is not None and lk.get("steps_checked", 0) == len(steps)
                                              and lk.get("pairs_checked", 0) == lk.get("pairs_total", 0) and lk.get("pairs_total", 0) > 0
                                              and int(lk.get("coincident_pairs", 0)) == 0
                                              and lk["ok"] and not any(e.get("where") == "linkage" for e in res["errors"])),
                             "method": "every committed engine step x every request row range of the step (engine row order): ids(s) == token(k-1) / declared prompt at prefill, "
                                       "argmax lm_head(final_norm_out(s)) == token(k); positions(s)[rows] == prompt_len + k - 1 (decode) / 0..len-1 (prefill); "
                                       "tokens = the engine's own per-request outputs (== control arm's); coincident_pairs = rows of one step with equal "
                                       "(token(k-1), token(k), position) -- told apart by no gate, must be 0; coincident_pairs_tokens = the token-only count"}
        eg = res["edges"]
        eg["ok"] = (not eg["declared"]) or (eg["checked"] > 0 and eg["equal"] == eg["checked"] and eg["missing"] == 0 and not eg["mismatch"])
        res["ok"] = (res["launches_checked"] > 0 and res["launches_exact"] == res["launches_checked"] and res["launches_missing"] == 0
                     and not res["errors"] and (lk["ok"] if self.tokens_by_req is not None else True) and eg["ok"])
        res["linkage"]["requests"] = sorted(self.tokens_by_req) if self.tokens_by_req else []
        res["linkage"]["prompt_tied"] = bool(self.prompts)
        res["seconds"] = round(time.perf_counter() - t0, 3)
        return res

    def _check_launch(self, s: int, piece: int, metas: dict[str, Any], kernel: str, roles: dict[int, str], wargs: set[int],
                      res: dict[str, Any]) -> None:
        """One launch of a sampled step: its relation over the committed inputs and registered weights against the committed
        outputs, bit for bit.  A missing role, a mismatch (with where it differs) or an error is counted in `res`."""
        kind = None
        weights = self._weights(piece, kernel, wargs)
        in_idx = set(wargs) | {i for i, r in roles.items() if r not in OUTPUT_ROLES} | set(weights)
        ins: list = [None] * (max(in_idx) + 1)                   # relation() takes the launch's INPUT arguments in signature order
        for i, t in weights.items():
            ins[i] = t
        missing = []
        outs: dict[str, torch.Tensor] = {}
        for i, role in roles.items():
            if i in wargs:
                continue                                          # a weight argument is registered, never committed: _weights() supplies it (Qwen 'bias')
            t = self.committed(s, metas, f"c{piece:02d}.{kernel}/{role}")
            if t is None:
                missing.append(role)
                continue
            if role in OUTPUT_ROLES:
                outs[role] = t
            else:
                ins[i] = t
        if missing:
            res["launches_missing"] += 1
            if len(res["missing"]) < 12:
                res["missing"].append({"step": s, "piece": piece, "kernel": kernel, "roles": missing})
            return
        out_roles = [r for _i, r in sorted(roles.items()) if r in OUTPUT_ROLES]
        outs_meta = [{"shape": list(outs[r].shape), "role": r} for r in out_roles]
        try:
            # the norm reduction order is learned per (kernel, row width) from the first launch checked (every
            # candidate order of the source structure is tried) and then pinned for the rest: Llama's 2048-wide rows keep
            # the S1-S2.md labels as the first guess; any other width starts from the full candidate set.
            width = next((int(t.shape[1]) for t in ins if t is not None and t.dim() == 2), None)
            lkey = (kernel, width)
            if "rms_norm" in kernel:
                labels = self._norm_labels.get(lkey) or NORM_LABELS          # R0_BLOCK 2048 x 8 warps along the row for any row <= 2048 (Llama 2048, Qwen 1536: masked tail); a miss -> full search
            else:
                labels = None
            rel = relation(kernel, [x for x in ins], outs_meta, eps=self.eps, labels=labels)
            if "rms_norm" in kernel and labels is not None and not any(
                    all(c is not None and j < len(c) and c[j].shape == outs[r].reshape(c[j].shape).shape and torch.equal(c[j].to(outs[r].dtype), outs[r].reshape(c[j].shape))
                        for j, r in enumerate([rr for _i, rr in sorted(roles.items()) if rr in OUTPUT_ROLES])) for c in rel.get("variants", {}).values()):
                rel = relation(kernel, [x for x in ins], outs_meta, eps=self.eps, labels=None)     # the pinned/guessed orders miss: full search, once
            kind = rel["kind"]
            labelled = list(rel["variants"].items()) if "variants" in rel else [("single", rel["outputs"])]
            labelled = [(lb, c) for lb, c in labelled if c is not None]
            cands = [c for _lb, c in labelled]
            exact_any = False
            per_out_ok = [False] * len(out_roles)
            matched_labels = []
            for lb, cand in labelled:
                ok = []
                for j, r in enumerate(out_roles):
                    pred = cand[j] if j < len(cand) else None
                    got = outs[r]
                    o = pred is not None and pred.shape == got.reshape(pred.shape).shape and torch.equal(pred.to(got.dtype), got.reshape(pred.shape))
                    ok.append(bool(o))
                per_out_ok = [a or b for a, b in zip(per_out_ok, ok)]
                if all(ok):
                    exact_any = True
                    matched_labels.append(lb)
            bk = res["by_kind"].setdefault(kind, {"launches": 0, "exact": 0, "outputs": 0, "outputs_exact": 0, "variants_exact": {}})
            for lb in matched_labels:
                bk.setdefault("variants_exact", {})[lb] = bk["variants_exact"].get(lb, 0) + 1
            if "rms_norm" in kernel and matched_labels and lkey not in self._norm_labels:
                self._norm_labels[lkey] = tuple(matched_labels)
                res.setdefault("norm_orders", {})[f"{kernel}@{width}"] = list(matched_labels)
            bk["launches"] += 1
            bk["outputs"] += len(out_roles)
            bk["outputs_exact"] += sum(per_out_ok)
            res["outputs_checked"] += len(out_roles)
            res["outputs_exact"] += sum(per_out_ok)
            res["launches_checked"] += 1
            # [the compiled-kernel autotune rule] the kernel's own compiled source under EVERY candidate launch config
            # on these committed inputs == the committed outputs (bit for bit), the executed config included -- shown per kernel
            # on up to MAX_LAUNCHES_PER_KERNEL sampled launches; the merge lifts a cross-arm config difference only for kernels shown
            self._autotune_equivalence(kernel, ins, outs, roles, res)
            if exact_any:
                res["launches_exact"] += 1
                bk["exact"] += 1
            elif len(res["mismatch"]) < 12:
                # diagnostics: WHERE the committed output differs from the relation (late read / overwritten buffer = whole
                # rows differ; numerics = scattered elements) -- against the first candidate
                diag = []
                for j, r in enumerate(out_roles):
                    pred = cands[0][j] if cands and j < len(cands[0]) else None
                    got = outs[r]
                    if pred is None or pred.numel() != got.numel():
                        diag.append({"role": r, "shape_pred": None if pred is None else list(pred.shape), "shape_got": list(got.shape)})
                        continue
                    g = got.reshape(pred.shape); d = pred.to(g.dtype) != g
                    rows_bad = d.reshape(d.shape[0], -1).any(dim=1).nonzero().flatten().tolist() if d.dim() >= 1 else []
                    fd = (pred.float() - g.float()).abs()
                    diag.append({"role": r, "shape": list(pred.shape), "n_diff": int(d.sum()), "rows_diff": len(rows_bad),
                                 "rows_first_last": [rows_bad[0], rows_bad[-1]] if rows_bad else [],
                                 "max_abs_diff": float(fd.max()), "got_finite": bool(torch.isfinite(g.float()).all())})
                res["mismatch"].append({"step": s, "piece": piece, "kernel": kernel, "kind": kind, "outputs_ok": per_out_ok,
                                        "n_rows": int(next(t.shape[0] for t in ins if t is not None and t.dim() >= 1)) if any(t is not None for t in ins) else None, "diag": diag})
        except Exception as e:  # noqa: BLE001
            res["launches_checked"] += 1
            if len(res["errors"]) < 12:
                res["errors"].append(f"step {s} piece {piece} {kernel}: {e!r}"[:300])

    def _link_step(self, s: int, metas: dict[str, Any], lm_head: Any, res: dict[str, Any]) -> None:
        """Ordered boundary linkage of ONE engine step, per request row range of the step's batched buffers (the
        per-(step, request) attribution of the committed rows).  ids(s)[rows of r] == r's token(k-1) (its prefill rows == the DECLARED prompt);
        argmax lm_head(final_norm_out(s)[last row of r]) == r's token(k); the tokens are the engine's own per-request outputs (equal to the
        uninstrumented control arm's).  Runs over EVERY committed step (attribution), the sampled steps included."""
        lk = res["linkage"]
        ids = self.committed(s, metas, f"c00.{self._embed_kernel(metas)}/ids") if self._embed_kernel(metas) else None
        fk = self._final_kernel(metas)
        hid = self.committed(s, metas, f"c{self.pieces - 1:02d}.{fk}/out") if fk else None
        # third gate: the committed POSITIONS buffer (the RoPE launches' `pos` input, one int64 per row,
        # in the required population at every step) -- a request's decode row sits at position prompt_len + k - 1, its prefill rows at
        # 0..prompt_len-1.  Two rows of one step with equal (token(k-1), token(k)) are still told apart when their positions differ, so the
        # coincidence key is (prev, next, position); rows with equal tokens AND equal positions remain coincident (named).
        pn = self._pos_name(metas)
        pos = self.committed(s, metas, pn) if pn else None
        pos = pos.reshape(-1) if pos is not None else None
        rows = (self.request_rows or {}).get(s)
        n_checked = 0                                                 # request rows of this step tied by at least one gate (ids or argmax)
        gate_keys: list[tuple] = []                                   # (prev id, next id, position) per tied row: rows with the same key are NOT told apart by the gates
        token_keys: list[tuple] = []                                  # (prev id, next id) -- the token-only coincidence, kept in the record
        if rows is not None and lk.get("rows_sample") is None:
            lk["rows_sample"] = {"step": s, "rows": [(str(r)[:24], lo, hi) for r, lo, hi in rows[:4]]}
        if rows is not None and len(self.tokens_by_req) == 1 and not any(r in self.tokens_by_req for r, _lo, _hi in rows):
            only = next(iter(self.tokens_by_req))                     # single request: the engine's row id names it whatever it names it
            rows = [(only, lo, hi) for _r, lo, hi in rows]
        if rows is None:                                              # single request, no row record: every row of the step is r0's
            n_rows = int(ids.numel()) if ids is not None else (int(hid.reshape(-1, hid.shape[-1]).shape[0]) if hid is not None else 0)
            rows = [("r0", 0, n_rows)] if n_rows else []
        for rid, lo, hi in rows:
            toks = self.tokens_by_req.get(rid)
            if toks is None:
                lk["unmapped_rows"] = lk.get("unmapped_rows", 0) + 1     # an engine row id the workload does not name (B>=2: named, not silent)
                continue
            k = s - self.arrivals.get(rid, 0)                       # the request's k-th sampling step
            c0 = lk["ids_checked"] + lk["argmax_checked"]
            if ids is not None and ids.numel() >= hi:
                got = ids.reshape(-1)[lo:hi]
                if k == 0 and self.prompts.get(rid):                # prefill rows == the DECLARED prompt (workload manifest)
                    lk["ids_checked"] += 1
                    want = torch.tensor(self.prompts[rid], dtype=got.dtype, device=got.device)
                    if got.numel() == want.numel() and bool(torch.equal(got, want)):
                        lk["ids_equal"] += 1
                    elif len(lk["mismatch"]) < 8:
                        lk["mismatch"].append({"step": s, "request": rid, "prefill_rows": int(got.numel()), "prompt": int(want.numel()),
                                               "first_diff": int((got != want).nonzero()[0].item()) if got.numel() == want.numel() else None})
                elif k >= 1 and hi - lo == 1 and (k - 1) < len(toks):
                    lk["ids_checked"] += 1
                    if int(got[-1]) == int(toks[k - 1]):
                        lk["ids_equal"] += 1
                    elif len(lk["mismatch"]) < 8:
                        lk["mismatch"].append({"step": s, "request": rid, "ids": int(got[-1]), "token_prev": int(toks[k - 1])})
            if hid is not None and 0 <= k < len(toks):
                h2 = hid.reshape(-1, hid.shape[-1])
                if h2.shape[0] >= hi:
                    logits = torch.nn.functional.linear(h2[hi - 1:hi].to(lm_head.weight.dtype), lm_head.weight)   # eager lm_head (bf16) on the request's last row
                    lk["argmax_checked"] += 1
                    am = int(logits.float().argmax(-1).item())
                    if am == int(toks[k]):
                        lk["argmax_equal"] += 1
                    elif len(lk["mismatch"]) < 8:
                        lk["mismatch"].append({"step": s, "request": rid, "argmax": am, "token": int(toks[k])})
            # position gate: the committed positions of the request's rows == its declared prompt length + k - 1 (decode) / 0..len-1 (prefill)
            row_pos = None
            plen = len(self.prompts[rid]) if self.prompts.get(rid) else None
            if pos is not None and pos.numel() >= hi and plen is not None:
                got_p = pos[lo:hi]
                if k == 0:
                    want_p = torch.arange(hi - lo, dtype=got_p.dtype, device=got_p.device)
                elif hi - lo == 1:
                    want_p = torch.tensor([plen + k - 1], dtype=got_p.dtype, device=got_p.device)
                else:
                    want_p = None
                if want_p is not None:
                    lk["pos_checked"] = lk.get("pos_checked", 0) + 1
                    if bool(torch.equal(got_p, want_p)):
                        lk["pos_equal"] = lk.get("pos_equal", 0) + 1
                    elif len(lk["mismatch"]) < 8:
                        lk["mismatch"].append({"step": s, "request": rid, "positions": [int(v) for v in got_p[:4]], "expected": [int(v) for v in want_p[:4]]})
                row_pos = int(got_p[-1])
            n_checked += int(lk["ids_checked"] + lk["argmax_checked"] > c0)
            if k >= 1 and (k - 1) < len(toks) and k < len(toks):
                token_keys.append((int(toks[k - 1]), int(toks[k])))
                gate_keys.append((int(toks[k - 1]), int(toks[k]), row_pos))
        # the named residual: two request rows of the SAME step that no gate tells apart pass under an equal-shape swap --
        # counted per step (pairs); attribution.complete requires 0.  Keys: (token(k-1), token(k), committed position) -- the token-only
        # count (the original definition) is kept beside it; when the positions buffer is not committed the key degrades to tokens.
        def _pairs(keys: list[tuple]) -> tuple[int, dict[tuple, int]]:
            seen: dict[tuple, int] = {}
            for gk in keys:
                seen[gk] = seen.get(gk, 0) + 1
            return sum(n * (n - 1) // 2 for n in seen.values() if n > 1), seen
        coincident, seen = _pairs(gate_keys)
        coincident_tok, seen_tok = _pairs(token_keys)
        if coincident_tok:
            lk["coincident_pairs_tokens"] = lk.get("coincident_pairs_tokens", 0) + coincident_tok
            if len(lk.setdefault("coincident_tokens_sample", [])) < 8:
                lk["coincident_tokens_sample"].append({"step": s, "pairs": [list(gk) for gk, n in seen_tok.items() if n > 1][:4],
                                                       "positions_differ": bool(coincident == 0) if pos is not None else None})
        if coincident:
            lk["coincident_pairs"] = lk.get("coincident_pairs", 0) + coincident
            if len(lk.setdefault("coincident_sample", [])) < 8:
                lk["coincident_sample"].append({"step": s, "pairs": [list(gk) for gk, n in seen.items() if n > 1][:4]})
        lk["position_gate"] = pn or lk.get("position_gate")
        lk["steps_checked"] = lk.get("steps_checked", 0) + 1
        lk["pairs_checked"] = lk.get("pairs_checked", 0) + n_checked
        lk["pairs_total"] = lk.get("pairs_total", 0) + len(rows)

    def _autotune_equivalence(self, kernel: str, ins: list, outs: dict[str, torch.Tensor], roles: dict[int, str], res: dict[str, Any]) -> None:
        """Per kernel: launches shown / not shown under every candidate autotune config (compiled_autotune.replay_under_candidates)."""
        ae = res.setdefault("autotune_equivalence", {})
        k = ae.get(kernel)
        if k is not None and (k.get("launches", 0) >= compiled_autotune.MAX_LAUNCHES_PER_KERNEL or k.get("n_candidates", 0) <= 1 and k.get("why")):
            return
        try:
            r = compiled_autotune.replay_under_candidates(kernel, ins, outs, roles, OUTPUT_ROLES)
        except Exception as e:  # noqa: BLE001
            r = {"shown": False, "why": f"raised: {e!r}"[:200], "candidates": [], "executed": [], "results": []}
        if k is None:
            k = ae[kernel] = {"heuristic": r.get("heuristic"), "structural": bool(r.get("structural")), "candidates": r.get("candidates"), "n_candidates": len(r.get("candidates") or []),
                              "executed": r.get("executed"), "executed_in_candidates": r.get("executed_in_candidates"), "launches": 0, "shown": 0,
                              "not_shown": 0, "why": None, "exact_by_config": {}, "derived_scalars_sample": r.get("derived_scalars")}
        if k["n_candidates"] <= 1:
            k["why"] = r.get("why")
            return
        k["launches"] += 1
        if r.get("shown"):
            k["shown"] += 1
        else:
            k["not_shown"] += 1
            k["why"] = k["why"] or r.get("why")
        for x in r.get("results") or []:
            lb = compiled_autotune.cfg_label(x["config"]) + (" (executed)" if x.get("executed") else "")
            k["exact_by_config"][lb] = k["exact_by_config"].get(lb, 0) + int(bool(x.get("exact")))
            if x.get("error") and len(k.setdefault("errors", [])) < 3:
                k["errors"].append(x["error"])
        k["all_shown"] = bool(k["launches"] > 0 and k["not_shown"] == 0 and k.get("executed_in_candidates"))
        # a lift needs a structural (pointwise) kernel: for a reduction the sampled equality is recorded but does not lift
        k["lift_eligible"] = bool(k["all_shown"] and k.get("structural"))
        if k["all_shown"] and not k.get("structural"):
            k["why"] = "non-pointwise heuristic -- the config sets the reduction order; sampled equality is not a property of the kernel (not lifted)"

    def _pos_name(self, metas: dict[str, Any]) -> str | None:
        """The committed positions buffer of a step: the first piece's RoPE launch `pos` input (`c01.<rope kernel>/pos`), else any `/pos`."""
        for n in sorted(metas):
            if n.endswith("/pos"):
                return n
        return None

    def _embed_kernel(self, metas: dict[str, Any]) -> str | None:
        for n in metas:
            if n.startswith("c00.") and "embedding" in n:
                return n.split(".", 1)[1].rsplit("/", 1)[0]
        return None

    def _final_kernel(self, metas: dict[str, Any]) -> str | None:
        pre = f"c{self.pieces - 1:02d}."
        for n in metas:
            if n.startswith(pre) and n.endswith("rms_norm_2/out"):
                return n.split(".", 1)[1].rsplit("/", 1)[0]
        return None

    def _launches_from_names(self, metas: dict[str, Any], piece: int) -> list[dict[str, str]]:
        pre = f"c{piece:02d}."
        seen = []
        for n in metas:
            if n.startswith(pre):
                k = n.split(".", 1)[1].rsplit("/", 1)[0]
                if k not in seen:
                    seen.append(k)
        return [{"kernel": k} for k in seen]
