"""Sampled exact VU replay (C2 of record) -- positive on a tiny honest Program, targeted negatives named."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.check.replay import (compare as CMP, coverage as COV, driver as SR, evaluate as EV, index as IX, linkage as LK,
                                      opening as OP, population as POP, replay_codes as RC, sample as SM)
from verity.ml.definitions.numerics import rows as RK
from verity_vllm.query.manifest import format as RM
from verity_vllm.query.required import Q_MODULE_BODY_ID

EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
V, H, I, LP, T = 8, 16, 16, 2, 2          # vocab, hidden, intermediate, prompt len, decode steps (steps = LP-prefill + T decode = 1 + T)
IMPL = ("model.layers.0.input_layernorm",)  # an observed implementation path: the Program's wrapper prefix is `model.`



@pytest.fixture(scope="module", autouse=True)
def _under_module_body():
    """These worlds exercise the replay's composition of interior rows (a pre-bias Gemm, the root's embedding scale): Values interior to a
    module body, which only `Q_module_body_v1` has.  Under the query of record every Call boundary is committed and nothing is composed."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IX, "DEFAULT_QUERY", Q_MODULE_BODY_ID)
        yield

def v2_manifest(inst: dict, request_ids=("r0",), *, digest: str = "t" * 64) -> dict:
    """The required-value manifest of record (`verity_vllm.query`) of `inst` for each request id: one request manifest, or (several requests of
    the same Program, arriving together at engine step 0) their global composition, as `query.cli build-global` writes it."""
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.program_view import from_instances
    from verity_vllm.query.compose import compose_global
    from verity_vllm.query.required import request_manifest
    res = {"program": {"digest": digest, "function": "Derived[vLLMRequest[Test]]"}, "wrapper": "request", "LP": LP, "T": T,
           "target": {"declared": {"compute_capability": [8, 9]}}}
    P = from_instances(inst)
    parts = {rid: request_manifest(P, Correspondence(P, implementation_paths=IMPL), res, request_id=rid, query=Q_MODULE_BODY_ID).manifest
             for rid in request_ids}
    if len(parts) == 1:
        return next(iter(parts.values()))
    reqs = [{"request_id": rid, "workload_request_id": rid, "index": k, "prompt_len": LP, "max_tokens": len(parts[rid]["steps"]), "arrive_step": 0}
            for k, rid in enumerate(request_ids)]
    return compose_global({"fixture": "test", "requests": reqs}, parts, reqs, workload_digest=None)


def _bf16(rng, n, scale=1.0):
    return (np.asarray(rng.standard_normal(n) * scale, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


NH, KVH, DH, MAXPOS = 3, 1, 16, 16      # rope=True: attention heads / kv heads / head dim (one k16 step) / rotary cache rows of the qkv + RoPE rows
ABN = 16                                 # attn=True: the attention rows' key block (one k16 P.V step; T <= LP + T = 4 -> one partial block)


def _program(bias: bool = False, rope: bool = False, attn: bool = False, scale: bool = False, registered: bool = False):
    """One request Program: per token row Embedding -> RMSNormTriton -> Gemm(gate_up) -> SiluMul -> Gemm(down) -> RMSNormFused(norm) ;
    per step Gemm(logits, tied embed) -> TokenSelect.  Names / args in the Build's instance-sequence schema.
    `bias`: the gate_up linear is BIASED (Qwen2.5 qkv_proj shape): Gemm -> BiasAdd_v1 under the one module, the module returns post-bias.
    `rope`: the token row also runs Gemm(qkv) -> RoPE_v1 query[k] (member `query`, NHEADS=NH) / key[k] (member `key`,
    NHEADS=KVH) exactly as the real Programs name them (`rotary_emb/rotary_embedding_{s}/query[k]`), arg 1 = ONE row of the `weights`
    field `...rotary_emb.cos_sin_cache` at the token's position (the Program's slice names the position); the rotated heads are leaves
    here (the real consumer is the state-linked Attention row).
    `attn`: (implies rope) the token row ends in `Attention_v3{T=pos+1,NH,KVH,D,BN}` exactly as the real Programs name
    it (`self_attn.attn/unified_attention_with_output_{s}[t=k]`): arg 0 = the query RoPE member, arg 1 = the T committed KEY members of
    positions 0..pos (earlier steps' rows by ref), arg 2 = the T V slices of those positions' qkv rows; the o_proj Gemm reads its output (a leaf).
    `scale`: the ROOT module scales the embedding in its own forward body -- `Bf16MulScalarTensor_v1{N=H}` at the
    UNROOTED path `model/mul_{s}[k]` (arg 0 = the embedding row, arg 1 = the one-word `weights` field `model.normalizer`, a buffer) -- and the
    first norm AND the final norm's residual read the SCALED row: an interior family (no member by the vocabulary) whose output crosses a
    module boundary (the v2 query requires it as the root module's `out`).
    `registered`: `TargetProfile.registered_values` -- the norms are `RMSNormTriton_v3` / `RMSNormFusedCuda_v4` and attention
    `Attention_v10`, each reading its eps / scale as the last arg, a field of the registered `values` parameter (index 1; weights move to 2)."""
    rope = rope or attn
    fields = [("model.model.embed_tokens.weight", V * H, f"Array<{V},Array<{H},Value<16>>>"),
              ("model.model.layers.0.input_layernorm.weight", H, f"Array<{H},Value<16>>"),
              ("model.model.layers.0.mlp.gate_up_proj.weight", 2 * I * H, f"Array<{2 * I},Array<{H},Value<16>>>"),
              ("model.model.layers.0.mlp.down_proj.weight", H * I, f"Array<{H},Array<{I},Value<16>>>"),
              ("model.model.norm.weight", H, f"Array<{H},Value<16>>")]
    if bias:
        fields.insert(3, ("model.model.layers.0.mlp.gate_up_proj.bias", 2 * I, f"Array<{2 * I},Value<16>>"))
    if scale:
        fields.append(("model.normalizer", 1, "Value<16>"))
    QKV = (NH + 2 * KVH) * DH
    if rope:
        fields[2:2] = [("model.model.layers.0.self_attn.qkv_proj.weight", QKV * H, f"Array<{QKV},Array<{H},Value<16>>>"),
                       ("model.model.layers.0.self_attn.rotary_emb.cos_sin_cache", MAXPOS * DH, f"Array<{MAXPOS},Array<{DH},Value<16>>>")]
    if attn:
        fields.append(("model.model.layers.0.self_attn.o_proj.weight", H * NH * DH, f"Array<{H},Array<{NH * DH},Value<16>>>"))
    off, woff = 0, {}
    for n, c, _t in fields:
        woff[n] = (off, off + c)
        off += c
    wtype = "{" + ", ".join(f"{n}: {t}" for n, _c, t in fields) + "}"
    rows, i = [], 0

    def add(spec, name, args):
        nonlocal i
        rows.append({"i": i, "spec": spec, "form": "call", "name": name, "args": args})
        i += 1
        return i - 1

    wp = 2 if registered else 1
    W = lambda n: [["p", wp, woff[n][0], woff[n][1]]]  # noqa: E731
    vfields = [("eps", int(np.float32(float.fromhex(json.loads(EPS)["f64"])).view(np.uint32)))]
    if attn:
        from verity.ml.definitions.registry import b1
        vfields.append((f"scale_log2_d{DH}", b1.scale_log2_bits(DH)))
    VAL = {n: [["p", 1, k, k + 1]] for k, (n, _b) in enumerate(vfields)}
    norm_t, norm_f = (f"RMSNormTriton_v3{{N={H}}}", f"RMSNormFusedCuda_v4{{N={H}}}") if registered else \
        (f"RMSNormTriton_v1{{N={H},EPS={EPS}}}", f"RMSNormFusedCuda_v2{{N={H},EPS={EPS}}}")
    eps_arg = [VAL["eps"]] if registered else []
    prev_norm_out = None                                       # (row, member offset) of the last fused norm (residual stream)
    kv_rows: list[tuple[int, int]] = []                        # attn: (key RoPE row, qkv row) per position, in position order
    for s in range(1 + T):
        toks = list(range(LP)) if s == 0 else [0]
        norm_rows = []
        for k in toks:
            ids = [["p", 0, k, k + 1]] if s == 0 else [["n", sel_prev, 0, 1]]  # noqa: F821  -- sel_prev set after step 0
            e = add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding_{s}[{k}]", [ids, W("model.model.embed_tokens.weight")])
            if scale:                                                  # the root module's own scaling of the embedding: read by the child norms
                e = add(f"Bf16MulScalarTensor_v1{{N={H}}}", f"model/mul_{s}[{k}]", [[["n", e, 0, H]], W("model.normalizer")])
            ln = add(norm_t, f"model.model.layers.0.input_layernorm/triton_launch_{s}[{k}]",
                     [[["n", e, 0, H]], W("model.model.layers.0.input_layernorm.weight"), *eps_arg])
            if rope:
                pos = k if s == 0 else LP + s - 1                              # the token's position = its index in the sequence
                cs_lo = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"][0] + pos * DH
                CS = [["p", wp, cs_lo, cs_lo + DH]]
                qkv = add(f"Gemm_v1{{K={H},N={QKV}}}", f"model.model.layers.0.self_attn.qkv_proj/triton_launch_{s}q[{k}]",
                          [[["n", ln, 0, H]], W("model.model.layers.0.self_attn.qkv_proj.weight")])
                qr = add(f"RoPE_v1{{NHEADS={NH},D={DH}}}", f"model.model.layers.0.self_attn.rotary_emb/rotary_embedding_{s}/query[{k}]",
                         [[["n", qkv, 0, NH * DH]], CS])
                kr = add(f"RoPE_v1{{NHEADS={KVH},D={DH}}}", f"model.model.layers.0.self_attn.rotary_emb/rotary_embedding_{s}/key[{k}]",
                         [[["n", qkv, NH * DH, (NH + KVH) * DH]], CS])
                if attn:
                    kv_rows.append((kr, qkv))
                    at = add(f"Attention_v{10 if registered else 3}{{T={pos + 1},NH={NH},KVH={KVH},D={DH},BN={ABN}}}",
                             f"model.model.layers.0.self_attn.attn/unified_attention_with_output_{s}[t={k}]",
                             [[["n", qr, 0, NH * DH]], [["n", kr_, 0, KVH * DH] for kr_, _q in kv_rows],
                              [["n", q_, (NH + KVH) * DH, QKV] for _k, q_ in kv_rows], *([VAL[f"scale_log2_d{DH}"]] if registered else [])])
                    add(f"Gemm_v1{{K={NH * DH},N={H}}}", f"model.model.layers.0.self_attn.o_proj/triton_launch_{s}o[{k}]",
                        [[["n", at, 0, NH * DH]], W("model.model.layers.0.self_attn.o_proj.weight")])
            gu = add(f"Gemm_v1{{K={H},N={2 * I}}}", f"model.model.layers.0.mlp.gate_up_proj/triton_launch_{s}a[{k}]",
                     [[["n", ln, 0, H]], W("model.model.layers.0.mlp.gate_up_proj.weight")])
            if bias:                                                   # the epilogue row: consumes the Gemm's (interior) output + the bias slice
                gu = add(f"BiasAdd_v1{{N={2 * I}}}", f"model.model.layers.0.mlp.gate_up_proj/bias_add_{s}[{k}]",
                         [[["n", gu, 0, 2 * I]], W("model.model.layers.0.mlp.gate_up_proj.bias")])
            act = add(f"SiluMul_v1{{I={I}}}", f"model.model.layers.0.mlp.act_fn/silu_and_mul_{s}[{k}]", [[["n", gu, 0, 2 * I]]])
            dn = add(f"Gemm_v1{{K={I},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch_{s}b[{k}]",
                     [[["n", act, 0, I]], W("model.model.layers.0.mlp.down_proj.weight")])
            nm = add(norm_f, f"model.model.norm/fused_add_rms_norm_{s}[{k}]",
                     [[["n", dn, 0, H]], [["n", e, 0, H]], W("model.model.norm.weight"), *eps_arg])     # residual = the embedding (one layer)
            norm_rows.append(nm)
        lg = add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_{s}c[0]", [[["n", norm_rows[-1], 0, H]], W("model.model.embed_tokens.weight")])
        sel_prev = add(f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [[["n", lg, 0, V]]])
    params = [["prompt", f"Array<{LP},Value<32>>"], ["weights", wtype]]
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": "t" * 64, "params": params, "rows": rows}
    if registered:
        params.insert(1, ["values", "{" + ", ".join(f"{n}: Value<32>" for n, _b in vfields) + "}"])
        inst["supplied"] = {"registration": {"param": 1, "name": "values", "fields": [{"name": n, "width": 32, "bits": f"0x{b:x}", "offset": k}
                                                                                         for k, (n, b) in enumerate(vfields)]}, "run": []}
    return inst, woff


class Honest:
    """Executes the Program per request with the registry evaluators and commits the module outputs per (engine step, member name):
    request blocks side by side (r0 then r1), the shape the collector's packed members have."""

    def __init__(self, inst, woff, weights, prompts):
        self.inst, self.woff, self.weights, self.prompts = inst, woff, weights, prompts
        self.pi = IX.ProgramIndex(inst)
        self.values: dict[tuple[str, int], np.ndarray] = {}       # (rid, row i) -> flattened output words
        self.blocks: dict[tuple[int, str, str, str], np.ndarray] = {}   # (step, rid, op_path, member) -> block words
        self.dtypes: dict[tuple[int, str, str, str], str] = {}    # (step, rid, op_path, member) -> the member's dtype (bf16 / f32 / i32 / i64)
        for rid, prompt in prompts.items():
            self._run(rid, prompt)

    def wslice(self, lo, hi):
        return self.weights[lo:hi]

    def _run(self, rid, prompt):
        pi = self.pi
        for r in pi.rows:
            d = pi.describe(r)
            fam = d["family"]
            inputs, wsl = [], []
            for group in r["args"]:
                words, w = [], None
                for kind, a, lo, hi in group:
                    if kind == "p" and pi.pnames[a] == "weights":
                        w = self.wslice(lo, hi)
                    elif kind == "p" and a in pi.supplied:
                        words.append(np.asarray([pi.supplied[a][1][lo]["bits"]], dtype=np.uint32))
                    elif kind == "p":
                        words.append(np.asarray(prompt[lo:hi], dtype=np.uint32))
                    else:
                        words.append(self.values[(rid, a)][lo:hi])
                inputs.append(words)
                wsl.append(w)
            got = EV.evaluate(fam, d["statics"], inputs, wsl)
            if len(got) > 1:                                            # the relation's ports under their committed members, as the replay reads them
                got = EV.by_member(got, pi.member_of_port(r["i"]))
            if d.get("plane") and d["plane"] not in got and len(got) == 1:  # the act SiluMul's `out` IS its plane
                got = {d["plane"]: next(iter(got.values()))}
            if not d["members"]:                                        # interior row (pre-bias Gemm): value flows on, nothing is committed
                self.values[(rid, r["i"])] = np.asarray(next(iter(got.values()))).reshape(-1).astype(np.uint16)
                continue
            flat = []
            for m in d["members"]:
                v = np.asarray(got[m[0]] if m[0] in got or len(got) != 1 else next(iter(got.values()))).reshape(-1)   # one output: the member
                wdt = np.uint64 if fam == "TokenSelect_v1" else (np.uint32 if str(m[2]) in ("f32", "i32") else np.uint16)
                flat.append(v.astype(wdt))
                key = (d["step"], rid, d["op_path"], m[0])
                self.dtypes[key] = str(m[2])
                self.blocks[key] = np.concatenate([self.blocks[key], flat[-1]]) if key in self.blocks else flat[-1]
            for c in d.get("copies") or []:                             # the same output committed under its second required name (a plane)
                key = (d["step"], rid, c["op_path"], c["member"])
                self.dtypes[key] = str(c["dtype"])
                self.blocks[key] = np.concatenate([self.blocks[key], flat[0]]) if key in self.blocks else flat[0]
            self.values[(rid, r["i"])] = np.concatenate(flat) if len(flat) > 1 else flat[0]
            if fam == "Embedding_v1":                                   # the ids the embedding consumed -> input_token_ids member `in`
                ids = inputs[0][0].astype(np.uint32)
                key = (d["step"], rid, d["op_path"], "in")
                self.blocks[key] = np.concatenate([self.blocks[key], ids]) if key in self.blocks else ids

    def bmap_and_reader(self, order=("r0", "r1"), mutate=None):
        """Binding map entries (request-local element ranges + element_offset) and a committed reader over packed members."""
        packed: dict[tuple[int, str, str], list] = {}
        for (step, rid, op, mem), blk in sorted(self.blocks.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3], order.index(kv[0][1]))):
            packed.setdefault((step, op, mem), []).append((rid, blk))
        entries, store = [], {}
        dtype_of = lambda op, mem: "i64" if op == IX.SAMPLER_OP else ("i32" if mem == "in" else "bf16")  # noqa: E731
        for (step, op, mem), parts in packed.items():
            name = f"{op}/{mem}"
            off, arrs = 0, []
            for rid, blk in parts:
                entries.append({"identity": {"program_digest": "t" * 64, "request_id": rid, "step": step, "invocation": 0, "op_path": op, "output_member": mem,
                                             "element_range": [0, int(blk.size)], "rank": 0}, "name": name, "element_offset": off,
                                "encoding": {"dtype": self.dtypes.get((step, rid, op, mem), dtype_of(op, mem))},
                                "family": "moe_block_stream" if op.startswith("moe/") else "instance_outputs"})
                off += int(blk.size)
                arrs.append(blk)
            store[(step, name)] = np.concatenate(arrs).tobytes()
        if mutate:
            mutate(store, entries)
        self.store = store

        def committed(step, name, lo, hi):
            b = store.get((int(step), name))
            return None if b is None else b[lo:hi]

        return {"schema": "test", "entries": entries}, committed


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)
    inst, woff = _program()
    n_w = max(hi for lo, hi in woff.values())
    weights = _bf16(rng, n_w, 0.5)
    weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80  # 1.0
    weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
    prompts = {"r0": [1, 2], "r1": [5, 3]}
    return Honest(inst, woff, weights, prompts)


@pytest.fixture(scope="module")
def biased_world():
    """The same Program with a Qwen2.5-shaped biased gate_up linear (Gemm -> BiasAdd under one module)."""
    rng = np.random.default_rng(11)
    inst, woff = _program(bias=True)
    n_w = max(hi for lo, hi in woff.values())
    weights = _bf16(rng, n_w, 0.5)
    weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80
    weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
    return Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


@pytest.fixture(scope="module")
def rope_world():
    """The Program with a qkv linear and RoPE_v1 query / key rows over a random bf16 `cos_sin_cache` (|words| <= 1)."""
    rng = np.random.default_rng(17)
    inst, woff = _program(rope=True)
    n_w = max(hi for lo, hi in woff.values())
    weights = _bf16(rng, n_w, 0.5)
    weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80
    weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
    lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
    weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


def _programs(world):
    return {rid: world.inst for rid in world.prompts}


def _weights(world):
    def read(field, lo, hi):
        flo, _fhi = world.woff[field]
        return world.weights[flo + lo: flo + hi]
    return read


def test_struct_parser_declaration_order_and_counts():
    fs = IX.parse_struct("{a: Array<3,Array<2,Value<16>>>, b: Array<4,Value<16>>, c: Value<16>}")
    assert [(n, c, s) for n, c, s, _b in fs] == [("a", 6, [3, 2]), ("b", 4, [4]), ("c", 1, [])]


@pytest.mark.slow
def test_honest_program_replays_bit_exactly_across_all_strata(world):
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3)
    assert rep["result"] is True and rep["grade"] == "complete", json.dumps(rep, indent=1, default=str)[:3000]
    assert rep["sample"]["mismatch_n"] == 0 and rep["sample"]["evaluated"] == rep["sample"]["picked"] > 0
    fams = set(rep["population"]["by_family"])
    assert fams == {"Embedding_v1", "RMSNormTriton_v1", "Gemm_v1", "SiluMul_v1", "RMSNormFusedCuda_v2", "TokenSelect_v1"}
    # strata: every spec x request x phase present is populated and evaluated; the sample is seeded (repeatable)
    assert all(v["evaluated"] >= 1 for v in rep["strata"].values())
    assert {k.split("|")[1] for k in rep["strata"]} == {"r0", "r1"}
    assert {k.split("|")[3] for k in rep["strata"]} == {"prefill", "first-decode", "last"}
    assert {k.split("|")[2] for k in rep["strata"]} == {"rank0"} and rep["strata_keys"]["ranks"][0].startswith("rank 0")
    # the picks by (request, step): [picked, evaluated, mismatch] per request-relative step; sums == the sample's
    pbs = rep["sample"]["picks_by_request_step"]
    assert set(pbs) == {"r0", "r1"} and all(len(v) == 3 for d in pbs.values() for v in d.values())
    assert sum(v[0] for d in pbs.values() for v in d.values()) == rep["sample"]["picked"]
    assert sum(v[1] for d in pbs.values() for v in d.values()) == rep["sample"]["evaluated"] and sum(v[2] for d in pbs.values() for v in d.values()) == 0
    assert all(str(k).isdigit() for d in pbs.values() for k in d) and "picked, evaluated, mismatch" in rep["sample"]["picks_by_request_step_key"]
    # the step is request-relative; the arrival per request sits beside it (engine step = arrival + step), 0 here
    assert "request-relative step" in rep["sample"]["picks_by_request_step_key"] and rep["sample"]["picks_by_request_arrival"] == {"r0": 0, "r1": 0}
    # a per-rank map (every entry identity.rank == 1) labels its strata rank1
    bmap1 = {"schema": "test", "entries": [dict(e, identity=dict(e["identity"], rank=1)) for e in bmap["entries"]]}
    rep1 = SR.sampled_replay(_programs(world), bmap1, committed, _weights(world), per_stratum=1, seed=3, workers=0)
    assert rep1["result"] is True and {k.split("|")[2] for k in rep1["strata"]} == {"rank1"} and rep1["strata_keys"]["ranks"] == ["rank 1"]
    assert rep["weights"]["equal"] is None and "unverified" in rep["weights"]["note"]
    assert rep["sample"]["seed_source"] == "caller"
    rep2 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3)
    assert rep2["strata"] == rep["strata"] and rep2["sample"] == rep["sample"]
    assert rep["evaluators"]["Gemm_v1"].startswith("verity.ml.definitions.numerics.derived_rows.dot_bf16")


def test_the_plan_reads_every_committed_range_the_replay_reads_and_evaluates_nothing(world):
    bmap, committed = world.bmap_and_reader()
    seen = {"plan": set(), "replay": set()}

    def reader(key):
        def read(step, name, lo, hi):
            seen[key].add((step, name, lo, hi))
            return committed(step, name, lo, hi)
        return read

    plan = SR.sampled_replay(_programs(world), bmap, reader("plan"), None, per_stratum=2, seed=3, plan=True)
    rep = SR.sampled_replay(_programs(world), bmap, reader("replay"), _weights(world), per_stratum=2, seed=3, workers=0)
    assert plan["plan"] is True and plan["picked"] == rep["sample"]["picked"] and plan["unresolved_n"] == 0
    assert seen["replay"] and seen["replay"] <= seen["plan"], sorted(seen["replay"] - seen["plan"])[:4]
    by_step: dict = {}
    for req, step, _row, _op in plan["picks"]:
        by_step.setdefault(req, {}).setdefault(str(step), 0)
        by_step[req][str(step)] += 1
    assert len(plan["picks"]) == plan["picked"] and len({tuple(p) for p in plan["picks"]}) == plan["picked"]
    assert by_step == {r: {str(s): v[0] for s, v in st.items()} for r, st in rep["sample"]["picks_by_request_step"].items()}, \
        "the plan's picks are the replay's, request by request and step by step"


def test_a_population_prepared_before_the_seed_draws_and_reads_the_same(world):
    bmap, committed = world.bmap_and_reader()
    progs, seen = _programs(world), {"cold": [], "warm": []}

    def reader(key):
        def read(step, name, lo, hi):
            seen[key].append((step, name, lo, hi))
            return committed(step, name, lo, hi)
        return read

    memo: dict = {}
    SR.populate(progs, [0], {}, {}, None, attention_paths=None, tp=None, program_dirs=None, manifest=None, memo=memo)
    assert any(k[0] == "population" for k in memo)
    cold = SR.sampled_replay(progs, bmap, reader("cold"), None, per_stratum=2, seed=3, plan=True)
    warm = SR.sampled_replay(progs, bmap, reader("warm"), None, per_stratum=2, seed=3, plan=True, memo=memo)
    assert warm["picked"] == cold["picked"] and seen["warm"] == seen["cold"] and seen["cold"]


def test_flipped_committed_output_byte_fails_named_at_the_request_and_row(world):
    def mutate(store, entries):
        k = (0, "model.layers.0.mlp.gate_up_proj/out")
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r1" and x["name"] == k[1] and x["identity"]["step"] == 0)
        pos = (e["element_offset"] + 1 * 2 * I + 5) * 2      # r1, prompt row 1, element 5
        b[pos] ^= 0x01
        store[k] = bytes(b)
    bmap, committed = world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=1)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    m = [x for x in rep["mismatch"] if x["spec"].startswith("Gemm_v1{K=16,N=32}")]
    assert m and all(x["request_id"] == "r1" and x["step"] == 0 for x in m)
    assert any(x["row"] == 1 for x in m) and "F_V(committed inputs) != committed output" in rep["why"]
    # r0 and the un-flipped rows of r1 stay equal: the fault is localised to the flipped value and its immediate consumer
    assert not any(x["request_id"] == "r0" for x in rep["mismatch"])
    assert all(x["row"] == 1 for x in rep["mismatch"])


def test_equal_shape_request_swap_of_an_input_member_fails_named(world):
    def mutate(store, entries):
        k = (0, "model.layers.0.input_layernorm/out")             # swap r0's and r1's blocks of the norm output (same shape)
        b = bytearray(store[k])
        half = len(b) // 2
        store[k] = bytes(b[half:] + b[:half])
    bmap, committed = world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=1)
    assert rep["result"] is False
    specs = {x["spec"].split("{")[0] for x in rep["mismatch"]}
    # the norm rows themselves (inputs = embeddings of the right request) and the gate_up Gemm rows (input = the other request's norm) fail
    assert {"RMSNormTriton_v1", "Gemm_v1"} <= specs
    assert {x["request_id"] for x in rep["mismatch"]} == {"r0", "r1"}
    # EVERY mismatch is counted by name on the record, uncapped (the detail list carries 12): the counts sum to mismatch_n
    mb = rep["mismatch_by"]
    n = rep["sample"]["mismatch_n"]
    assert sum(mb["by_family"].values()) == sum(mb["by_request"].values()) == sum(mb["by_phase"].values()) == sum(mb["strata"].values()) == n
    assert sum(mb["by_family_request_phase"].values()) == sum(mb["by_module"].values()) == n and n >= len(rep["mismatch"])
    assert {"RMSNormTriton_v1", "Gemm_v1"} <= set(mb["by_family"]) and set(mb["by_request"]) == {"r0", "r1"}
    assert all(rep["strata"][k]["mismatch"] == v and rep["strata"][k]["evaluated"] >= v for k, v in mb["strata"].items())
    assert all("mismatch" not in v for k, v in rep["strata"].items() if k not in mb["strata"])


def test_wrong_member_bytes_fail_named(world):
    def mutate(store, entries):
        store[(0, "model.norm/1")] = store[(0, "model.norm/0")]     # the residual member carries the normed rows
    bmap, committed = world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=1)
    assert rep["result"] is False
    fused = [x for x in rep["mismatch"] if x["spec"].startswith("RMSNormFusedCuda")]
    assert fused and all(any(o["member"] == "1" and not o["equal"] for o in x["outputs"]) for x in fused)
    assert all(all(o["equal"] for o in x["outputs"] if o["member"] == "0") for x in fused)


def test_missing_committed_input_is_partial_never_pass(world):
    def mutate(store, entries):
        del store[(1, "model.layers.0.mlp.act_fn/out")]
    bmap, committed = world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=1)
    assert rep["result"] is None and rep["grade"] == "partial"
    assert any("not retained" in x["why"] for x in rep["not_evaluated"])
    assert any(k.split("|")[3] == "first-decode" for k in rep["strata_unevaluated"])


def test_boundary_linkage_positive_and_negatives(world):
    bmap, committed = world.bmap_and_reader()
    reqs = [{"request_id": rid, "arrive_step": 0, "prompt_token_ids": p, "steps": [0, 1, 2]} for rid, p in world.prompts.items()]
    lk = LK.boundary_linkage(bmap, committed, reqs)
    assert lk["result"] is True and lk["pairs"] == 6 and lk["linked"] == 6, lk
    # foreign prompt: the declared prompt of r1 differs from what the embedding consumed
    bad = [dict(r, prompt_token_ids=[5, 4]) if r["request_id"] == "r1" else r for r in reqs]
    lk2 = LK.boundary_linkage(bmap, committed, bad)
    assert lk2["result"] is False and lk2["unlinked"][0]["request_id"] == "r1" and lk2["unlinked"][0]["kind"] == "prompt"
    # broken chain: r0's committed input id at step 2 is not r0's sampled id of step 1

    def mutate(store, entries):
        k = (2, "model.embed_tokens/in")
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r0" and x["name"] == k[1] and x["identity"]["step"] == 2)
        pos = e["element_offset"] * 4
        b[pos] ^= 0x04
        store[k] = bytes(b)
    bmap3, committed3 = world.bmap_and_reader(mutate=mutate)
    lk3 = LK.boundary_linkage(bmap3, committed3, reqs)
    assert lk3["result"] is False
    u = lk3["unlinked"][0]
    assert u["request_id"] == "r0" and u["engine_step"] == 2 and u["kind"] == "chain" and "!= committed sampled id" in u["why"]
    # the replay reads the Embedding VU's input from the Program's dataflow (the previous step's committed sampled id), so a forged `in`
    # member is invisible to it -- the exhaustive linkage is the check that names it; the two are graded separately (R2)
    rep = SR.sampled_replay(_programs(world), bmap3, committed3, _weights(world), per_stratum=8, seed=1)
    assert rep["result"] is True


def _bnd(n_req):
    return {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": n_req}}


def test_verdict_c2_of_record_replay_plus_linkage_replaces_the_oracle_gate_at_b2():
    """At B>=2 a complete sampled replay + linked boundaries PASS without any Match oracle; a replay mismatch or
    an unlinked boundary is a named FAIL at any B; a partial replay alone does not attribute."""
    from verity_vllm.check.verdict import commit_verdict
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True}}
    ok_sr = {"result": True, "grade": "complete", "why": None}
    ok_lk = {"result": True, "pairs": 6, "linked": 6, "unlinked_n": 0}
    ok_wp = {"result": True, "compared": 98, "mismatched": [], "missing": []}
    base["weights_pin"] = ok_wp                                                           # the checkpoint pin is part of the record
    ok_, why_ = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": ok_sr, "boundary_linkage": ok_lk}, True, min_openings=64)
    assert ok_ is True and why_.startswith("PASS (B=2: per-(step,request) attribution not verified on this record"), why_   # PASS, gap named
    # mismatch: named
    bad_sr = {"result": False, "grade": "mismatch", "why": "Gemm_v1 r1 step 0 model.layers.0.mlp.gate_up_proj/0 row 1: F_V(committed inputs) != committed output"}
    ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": bad_sr, "boundary_linkage": ok_lk}, True, min_openings=64)
    assert not ok and why.startswith("C2 sampled replay:") and "gate_up_proj/0 row 1" in why and "r1" in why
    ok, why = commit_verdict({**base, "binding": _bnd(1), "sampled_replay": bad_sr}, True, min_openings=64)
    assert not ok and why.startswith("C2 sampled replay:")
    # unlinked boundary: named
    bad_lk = {"result": False, "pairs": 6, "linked": 5, "unlinked_n": 1, "why": "committed input id 7 != committed sampled id 3 of step 1"}
    ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": ok_sr, "boundary_linkage": bad_lk}, True, min_openings=64)
    assert not ok and why.startswith("C2 boundary linkage:") and "!= committed sampled id 3" in why
    # partial replay (a stratum not evaluated) = not established, at B2 AND at B1; absence = not established; a complete
    # oracle attribution no longer substitutes.  Records without a binding block (unit fixtures of other checks) keep the legacy rule.
    part = {"result": None, "grade": "partial", "why": "2 strata not evaluated: [...]"}
    for b in (1, 2):
        ok, why = commit_verdict({**base, "binding": _bnd(b), "sampled_replay": part, "boundary_linkage": ok_lk}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established") and "partial" in why
        ok, why = commit_verdict({**base, "binding": _bnd(b)}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established") and "not run (no --program-dir" in why
        ok, why = commit_verdict({**base, "binding": _bnd(b), "sampled_replay": ok_sr}, True, min_openings=64)
        assert not ok and "boundary linkage None -- not run" in why
    oc = {"result": None, "compared": 100, "grade": "attribution-complete", "attribution": {"complete": True, "unverified_n": 0, "ambiguous_steps": []}}
    ok, why = commit_verdict({**base, "binding": _bnd(2), "oracle_compare": oc}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established")
    # a weights-at-replay mismatch is a replay FAIL by name
    wsr = {"result": False, "grade": "mismatch", "why": "weights at replay != committed weights root (root differs)"}
    ok, why = commit_verdict({**base, "binding": _bnd(1), "sampled_replay": wsr, "boundary_linkage": ok_lk}, True, min_openings=64)
    assert not ok and "weights at replay != committed weights root" in why
    # legacy record (no binding block): the old rule
    assert commit_verdict(base, True, min_openings=64) == (True, "PASS")


def test_partial_replay_with_named_gaps_is_covered_by_a_complete_form_b_coord_m0721():
    """C2 of record holds when EITHER form is complete: a PARTIAL replay with 0 mismatch whose every
    unevaluated stratum is NAMED (family | module | reason) + a COMPLETE form (B) oracle compare = PASS carrying the named gap; the same
    partial replay without form (B), with a form (B) mismatch / incomplete attribution, with an UNNAMED gap, or with any replay mismatch =
    FAIL (O1 stands: absence of both forms never passes)."""
    from verity_vllm.check.verdict import commit_verdict
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "value_correspondence": {"method": "sampled-exact-replay(partial) + boundary-linkage + oracle", "result": True},
            "boundary_linkage": {"result": True, "pairs": 6, "linked": 6, "unlinked_n": 0},
            "weights_pin": {"result": True, "compared": 98, "mismatched": [], "missing": []}}
    K1, K2 = "RMSNormFusedCuda_v2{N=2048}|r0|rank0|first-decode|model.layers.1.input_layernorm|fresh-window", "SiluMul_v1{N=1024}|r0|rank0|first-decode|model.layers.1.mlp.experts|fresh-window"
    ne = [{"spec": 'RMSNormFusedCuda_v2{N=2048}', "request_id": "r0", "step": 1, "op_path": "model.layers.1.input_layernorm", "row": 0,
           "why": "no evaluator for RMSNormFusedCuda_v2{N=2048} variant (statics outside the registered relation)", "why_class": "no-evaluator"},
          {"spec": 'BiasAdd_v1{N=2048}', "request_id": "r0", "step": 1, "op_path": "model.layers.1.self_attn.qkv_proj", "row": 0,
           "why": "fed by interior producer(s) without an evaluator (Gemm_v2 (model.layers.1.self_attn.qkv_proj)): composition not possible in this replay (MAN-06g, v2)", "why_class": "interior-composed-v2"}]
    # an older version of this test used a PLANE-FED second stratum and expected form (B) to cover it;
    # form (B) has no oracle family for the moe/ planes those VUs commit (oracle_compare.NO_ORACLE_FAMILY_PREFIXES), so that PASS overclaimed
    # -- the second stratum is now an interior-composed one (Qwen2.5 BiasAdd: its instance output IS compared) and the plane-fed case FAILs below
    ne_plane = {"spec": 'SiluMul_v1{N=1024}', "request_id": "r0", "step": 1, "op_path": "model.layers.1.mlp.experts", "row": 0,
                "why": "fed by moe_block_stream planes (SiluMul_v1@experts): row -> plane mapping not in this replay (MAN-06f, v2)", "why_class": "plane-fed"}
    # the rule reads the UNCAPPED per-stratum list, never the capped not_evaluated[:12] sample
    us = [{"stratum": K1, **{k: ne[0][k] for k in ("spec", "request_id", "op_path", "why", "why_class")}, "phase": "first-decode", "n": 2, "classes": ["no-evaluator"]},
          {"stratum": K2, **{k: ne[1][k] for k in ("spec", "request_id", "op_path", "why", "why_class")}, "phase": "first-decode", "n": 2, "classes": ["interior-composed-v2"]}]
    K3 = "SiluMul_v1{N=1024}|r0|rank0|first-decode|model.layers.1.mlp.experts|fresh-window"
    us_plane = {"stratum": K3, **{k: ne_plane[k] for k in ("spec", "request_id", "op_path", "why", "why_class")}, "phase": "first-decode", "n": 2, "classes": ["plane-fed"]}
    part = {"result": None, "grade": "partial", "why": "2 strata not evaluated: [...]",
            "sample": {"evaluated": 598, "equal": 598, "mismatch_n": 0, "not_evaluated_n": 224},
            "strata_unevaluated": [K1, K2], "unevaluated_strata": us, "not_evaluated": ne}
    oc_ok = {"result": True, "grade": "complete", "compared": 3, "equal": 3, "mismatch_n": 0, "complete": True,
             "compared_by_family": {"instance_outputs": 2, "input_token_ids": 1}, "not_compared_by_family": {},
             "attribution": {"complete": True, "pairs": 3, "unverified_n": 0, "ambiguous_steps": []}}
    for b in (1, 2):
        ok, why = commit_verdict({**base, "binding": _bnd(b), "sampled_replay": part, "oracle_compare": oc_ok}, True, min_openings=64)
        assert ok and why.startswith("PASS (C2 of record by form (B)") and "2 strata named not evaluated (interior-composed-v2, no-evaluator): BiasAdd_v1, RMSNormFusedCuda_v2" in why \
            and "interior producer" in why and "instance outputs 2" in why, why
        # a PLANE-FED stratum beside them: FAIL by name even with the same complete form (B) -- neither form checks the planes
        p3 = {**part, "strata_unevaluated": [K1, K2, K3], "unevaluated_strata": us + [us_plane], "not_evaluated": ne + [ne_plane]}
        ok, why = commit_verdict({**base, "binding": _bnd(b), "sampled_replay": p3, "oracle_compare": oc_ok}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established: 1 plane-fed strata (2 VUs) have committed moe/ plane members value-checked by NEITHER form") \
            and "no oracle family for moe/ planes" in why and "SiluMul_v1" in why and "moe_block_stream" in why and "3 strata named not evaluated" in why, why
        # the oracle under value_correspondence.oracle_compare is read too
        vc = {**base["value_correspondence"], "oracle_compare": oc_ok}
        ok, why = commit_verdict({**base, "value_correspondence": vc, "binding": _bnd(b), "sampled_replay": part}, True, min_openings=64)
        assert ok and "form (B)" in why, why
        # no form (B): FAIL (O1)
        ok, why = commit_verdict({**base, "binding": _bnd(b), "sampled_replay": part}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established") and "partial" in why
    # form (B) incomplete (attribution-complete only / result None) or a mismatch: FAIL
    for oc in ({**oc_ok, "result": None, "grade": "attribution-complete", "complete": False},
               {**oc_ok, "attribution": {"complete": False, "pairs": 3, "unverified_n": 1, "ambiguous_steps": [1]}},
               {**oc_ok, "compared": 0, "equal": 0}):
        ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": part, "oracle_compare": oc}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established"), why
    oc_bad = {**oc_ok, "result": False, "grade": "mismatch", "mismatch_n": 1, "why": "C2 oracle mismatch: 1 identities differ from the Match oracle"}
    ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": part, "oracle_compare": oc_bad}, True, min_openings=64)
    assert not ok and "oracle mismatch" in why
    # an UNNAMED gap never qualifies: no strata; a stratum without a per-stratum entry (only the capped sample names it, G1); an entry
    # without a reason / op_path; a stratum list longer than the entries
    for p in ({**part, "strata_unevaluated": [], "unevaluated_strata": []},
              {**part, "unevaluated_strata": []},                                              # 128 strata, 12 named samples: NOT named
              {**part, "unevaluated_strata": [{**us[0], "why": None}, us[1]]},
              {**part, "unevaluated_strata": [{**us[0], "op_path": None}, us[1]]},
              {**part, "strata_unevaluated": [K1, K2, "Gemm_v1{K=1,N=1}|r1|rank0|mid|m|fresh-window"]}):
        ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": p, "oracle_compare": oc_ok}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established"), why
    # [G2] a FAULT class never qualifies, whatever the text says: evaluator error, mapping fault ('arg slice ... not inside one committed
    # member', as measured), missing committed input, device fault, the JIT race -- with or without a stamped class
    for w, c in (("evaluator error OSError librms_triton_model.so: file too short", "evaluator-error"),
                 ("arg slice ['n', 148006, 0, 2048) is not inside one committed member of row 148006", "mapping"),
                 ("committed identity (r0, 3, model.layers.1.mlp.down_proj, 0) not available", "missing-input"),
                 ("CUDA error: initialization error (forked child)", "device")):
        for stamped in (True, False):
            u0 = {**us[0], "why": w, **({"why_class": c, "classes": [c]} if stamped else {"why_class": None, "classes": None})}
            ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": {**part, "unevaluated_strata": [u0, us[1]]}, "oracle_compare": oc_ok}, True, min_openings=64)
            assert not ok and why.startswith("C2 of record not established"), (w, stamped, why)
    # [G3] form (B) must have compared the module outputs the gaps produce: none compared, or some left not compared -> FAIL
    for oc in ({**oc_ok, "compared_by_family": {"input_token_ids": 3}, "not_compared_by_family": {}},
               {**oc_ok, "compared_by_family": {"instance_outputs": 2}, "not_compared_by_family": {"instance_outputs": 5}}):
        ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": part, "oracle_compare": oc}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established"), why
    # a replay MISMATCH inside a partial replay always fails, whatever form (B) says
    pm = {**part, "sample": {**part["sample"], "mismatch_n": 1}}
    ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": pm, "oracle_compare": oc_ok}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established"), why
    # a COMPLETE replay still passes plainly (no note)
    assert commit_verdict({**base, "binding": _bnd(2), "sampled_replay": {"result": True, "grade": "complete"}, "oracle_compare": oc_ok},
                          True, min_openings=64) == (True, "PASS")


def test_token_select_first_max_tie_rule():
    logits = np.asarray([0x3F80, 0x4000, 0x4000, 0x3F00], dtype=np.uint16)   # 1.0, 2.0, 2.0, 0.5 -> index 1
    assert RK.token_select_first_max(logits) == 1
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity.ml.definitions.registry import b1

    nan = [[0x7FC0, 0x3F80], [0x3F80, 0x7FC0, 0x4000], [0xFF80, 0x7FC0]]   # NaN never wins a comparison, so a NaN logits[0] is the token
    assert [RK.token_select_first_max(np.asarray(r, dtype=np.uint16)) for r in nan] == [0, 2, 0]
    assert [evaluate(bind(b1.TokenSelect, V=len(r)), r)[0] for r in nan] == [0, 2, 0]


def test_catch_probability_is_evaluated_over_population_per_stratum(world):
    """The record states the honest catch probability of a single-VU fault per phase = evaluated/population of its stratum:
    1.0 where the stratum is fully evaluated (prefill / first-decode / last of this tiny Program), < 1 only where a stratum holds more VUs
    than per_stratum picks; the note names the seed-source caveat."""
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0)
    cp = rep["catch_probability"]["single_vu_fault"]
    assert cp["overall"]["strata"] == len(rep["strata"]) and 0 < cp["overall"]["min"] <= cp["overall"]["median"] <= 1.0
    for k, v in rep["strata"].items():
        ph = k.split("|")[3]
        assert cp["by_phase"][ph]["min"] <= min(1.0, v["evaluated"] / v["population"]) <= 1.0
    full = [k for k, v in rep["strata"].items() if v["evaluated"] >= v["population"]]
    assert cp["overall"]["strata_fully_evaluated"] == len(full)
    assert "seed_source" in rep["catch_probability"]["note"]


@pytest.mark.slow
def test_module_strata_cap_floor_and_targets(world):
    """(1) the stratum key carries the module op_path: one stratum per (spec, request, rank, phase, module) so every module of every
    request/phase is sampled; (2) the cap never truncates below one VU per stratum (round-robin order); (3) `--replay-target` pins every VU
    of a module (at a step) outside the cap and the record says so; (4) a flipped mid-phase identity the stratified sample misses is caught
    when targeted."""
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, max_evaluate=1)
    assert rep["result"] is True and rep["sample"]["cap_effective"] == len(rep["strata"]) and rep["strata_unevaluated"] == []
    assert rep["strata_keys"]["modules"] >= 6 and all(k.count("|") == 5 for k in rep["strata"])
    # lifetime axis (window recycle): unwindowed here -> one value, stated; with a ring of 2 slots the steps >= 2 are recycled-window strata
    assert set(rep["strata_keys"]["lifetime"]["values"]) == {"n/a (unwindowed storage)"}
    rep_w = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=3, ring_slots=2)
    assert rep_w["result"] is True and set(rep_w["strata_keys"]["lifetime"]["values"]) == {"fresh-window", "recycled-window"}
    assert rep_w["strata_keys"]["lifetime"]["sampled"]["recycled-window"] >= 1 and rep_w["strata_keys"]["lifetime"]["ring_slots"] == 2
    assert rep["sample"]["stratified_picked"] == len(rep["strata"]) and all(v["evaluated"] == 1 for v in rep["strata"].values())
    assert rep["timing"]["wall_s"] >= 0 and set(rep["timing"]["per_family_n"]) == set(rep["population"]["by_family"])
    # a flipped decode-row of the down_proj at r0's FIRST decode step (a one-VU stratum) is sampled deterministically under the module key
    def mutate(store, entries):
        k = (1, "model.layers.0.mlp.down_proj/out")
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r0" and x["name"] == k[1] and x["identity"]["step"] == 1)
        b[e["element_offset"] * 2 + 3] ^= 0x40
        store[k] = bytes(b)
    bmap2, committed2 = world.bmap_and_reader(mutate=mutate)
    rep2 = SR.sampled_replay(_programs(world), bmap2, committed2, _weights(world), per_stratum=1, seed=3)
    assert rep2["result"] is False and any(x["op_path"].endswith("mlp.down_proj") and x["request_id"] == "r0" and x["step"] == 1 for x in rep2["mismatch"])
    # targets: every VU of a module at a step joins the sample outside the cap, recorded by name with its count
    rep3 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=3, max_evaluate=1,
                             targets=["model.layers.0.mlp.gate_up_proj@0", "no.such.module"])
    assert rep3["result"] is True
    assert rep3["sample"]["targeted"]["model.layers.0.mlp.gate_up_proj@0"] == 2 * LP and rep3["sample"]["targeted"]["no.such.module"] == 0
    assert rep3["sample"]["picked"] == rep3["sample"]["stratified_picked"] + rep3["sample"]["targeted_n"]   # targeted_n = VUs ADDED (hits minus overlap)
    assert rep3["sample"]["targeted_n"] == 2 * LP - sum(1 for k in rep3["strata"] if "mlp.gate_up_proj" in k and "prefill" in k)


def test_biased_linear_member_is_the_post_bias_tensor_produced_by_biasadd(world, biased_world):
    """Qwen2.5 qkv_proj: the module's member 0 is the POST-bias tensor; its producer row is BiasAdd_v1 and the Gemm is
    interior.  (1) manifest: same identity population as the unbiased Program, the biased module's row carries spec BiasAdd_v1 -- the
    oracle reads the fold's BiasAdd instance, never the pre-bias Gemm view; (2) the replay composes the interior Gemm from its committed
    inputs and matches the committed post-bias bytes bit-for-bit; (3) committing the PRE-bias bytes as member 0 fails named."""
    mb, mu = v2_manifest(biased_world.inst), v2_manifest(world.inst)
    assert mb["complete"] is True and mb["unmodelled"] == {}
    key = lambda r: (r["op_path"], r["output_member"], r["step"], tuple(r["element_range"]))  # noqa: E731
    outs = lambda m: sorted(key(r) for r in m["identities"] if r["family"] == "instance_outputs")  # noqa: E731
    assert outs(mb) == outs(mu)                                            # the value population is unchanged: one member per module per row
    gu_b = {r["spec"].split("{")[0] for r in mb["identities"] if r["op_path"] == "model.layers.0.mlp.gate_up_proj"}
    gu_u = {r["spec"].split("{")[0] for r in mu["identities"] if r["op_path"] == "model.layers.0.mlp.gate_up_proj"}
    assert gu_b == {"BiasAdd_v1"} and gu_u == {"Gemm_v1"}
    # (2) population: the interior Gemm is counted by name (not a VU, not 'not evaluable'), the BiasAdd rows are VUs; replay is exact
    pi = IX.ProgramIndex(biased_world.inst)
    vus, not_ev, not_ev_fam = POP.population(pi, "r0")
    assert not_ev_fam["Gemm_v1 (interior: composed into its epilogue's VU / a hinted row of the module is its returned tensor)"] == LP + T
    assert sum(1 for v in vus if v["family"] == "BiasAdd_v1") == LP + T and not not_ev
    bmap, committed = biased_world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(biased_world), bmap, committed, _weights(biased_world), per_stratum=8, seed=1, workers=0)
    assert rep["result"] is True and rep["sample"]["mismatch_n"] == 0 and rep["sample"]["not_evaluated_n"] == 0
    assert rep["timing"]["per_family_n"]["BiasAdd_v1"] == 2 * (LP + T)          # every biased-linear row of both requests: exact
    assert any(u["family"].startswith("Gemm_v1 (interior") and u["n"] == 2 * (LP + T) and "composed" in u["why"] for u in rep["unreplayable"])
    # the replay of one BiasAdd VU composes the interior Gemm (recorded as an `interior` input with its own committed inputs)
    pi0 = IX.ProgramIndex(biased_world.inst)
    ba = next(r for r in pi0.rows if r["spec"].startswith("BiasAdd_v1"))
    one = EV.replay_vu(pi0, OP.CommittedStore(bmap, committed), _weights(biased_world), {"i": ba["i"], "phase": "prefill"} | {"request_id": "r0"}, arrive=0)
    assert one["result"] is True and any(i["kind"] == "interior" and i["family"] == "Gemm_v1" for i in one["inputs"])
    # (3) the pre-bias shape in reverse: pre-bias bytes committed as member out -> mismatch named at the module with the producer spec
    def mutate(store, entries):
        pi2 = IX.ProgramIndex(biased_world.inst)
        gem = [r for r in pi2.rows if r["spec"].startswith("Gemm_v1") and "gate_up_proj" in r["name"] and pi2.step_of(r["i"]) == 0]
        pre = np.concatenate([biased_world.values[("r0", r["i"])] for r in gem])
        post = np.concatenate([biased_world.values[("r1", r["i"] + 1)] for r in gem])
        store[(0, "model.layers.0.mlp.gate_up_proj/out")] = np.concatenate([pre, post]).tobytes()
    bmap2, committed2 = biased_world.bmap_and_reader(mutate=mutate)
    rep2 = SR.sampled_replay(_programs(biased_world), bmap2, committed2, _weights(biased_world), per_stratum=8, seed=1, workers=0)
    assert rep2["result"] is False and rep2["sample"]["mismatch_n"] >= 1
    # named at the biased linear itself (F = Gemm∘BiasAdd on the committed inputs != the pre-bias bytes) and, downstream, at its consumer
    # (SiluMul on the wrong input != its committed output) -- only r0's rows; r1's post-bias bytes are untouched
    assert all(m["request_id"] == "r0" and m["step"] == 0 for m in rep2["mismatch"])
    assert any(m["op_path"] == "model.layers.0.mlp.gate_up_proj" and m["spec"].startswith("BiasAdd_v1") for m in rep2["mismatch"])
    assert {m["op_path"] for m in rep2["mismatch"]} <= {"model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.act_fn"}


def test_biasadd_over_an_interior_gemm_without_evaluator_is_not_evaluable_by_name_hopper_m0748(biased_world, monkeypatch):
    """A biased linear whose INTERIOR pre-bias Gemm has no registered evaluator: BiasAdd_v1 is the producer of record, its input
    is interior, and the BiasAdd VU cannot compose.  It is counted BY NAME as not evaluable (population), never sampled as a stratum that
    stays 'not evaluated' (partial = FAIL closed on every biased-linear row).  Over Gemm_v1 (Ada) the composition is unchanged: the
    BiasAdd rows are VUs and replay exact.  Gemm_v2 (the Hopper GEMM this finding was measured on) now HAS an evaluator -- see the Hopper
    tests below -- so the rule is exercised here with a synthetic GEMM version the query models (its ports are Gemm_v1's) and the
    evaluator table does not know (`Gemm_v9`): the same code path, the same names."""
    from verity_vllm.query import program_view as PV
    inst = json.loads(json.dumps(biased_world.inst))
    pi1 = IX.ProgramIndex(inst)
    vus1, ne1, nef1 = POP.population(pi1, "r0")
    assert sum(1 for v in vus1 if v["family"] == "BiasAdd_v1") == LP + T and not ne1
    assert POP._interior_feeders_without_evaluator(pi1, next(r for r in pi1.rows if r["spec"].startswith("BiasAdd_v1"))) == set()
    # the interior Gemm rows under the biased linear are of a GEMM version without a registered evaluator
    assert "Gemm_v9" not in RK.EVALUATORS and "Gemm_v9" not in RC.NOT_YET
    for r in inst["rows"]:
        if r["spec"].startswith("Gemm_v1") and "gate_up_proj" in r["name"]:
            monkeypatch.setitem(PV._PORTS_CACHE, "Gemm_v9" + r["spec"][len("Gemm_v1"):], PV.definition_ports(r["spec"]))
            r["spec"] = "Gemm_v9" + r["spec"][len("Gemm_v1"):]
    pi2 = IX.ProgramIndex(inst)
    ba = next(r for r in pi2.rows if r["spec"].startswith("BiasAdd_v1"))
    assert POP._interior_feeders_without_evaluator(pi2, ba) == {"Gemm_v9 (no registered evaluator)"}
    vus2, ne2, nef2 = POP.population(pi2, "r0")
    assert sum(1 for v in vus2 if v["family"] == "BiasAdd_v1") == 0                                   # no BiasAdd stratum can stay unevaluated
    assert nef2["BiasAdd_v1 (fed by interior Gemm_v9: composed evaluator, v2)"] == LP + T
    assert ne2["fed by interior producer(s) without an evaluator (Gemm_v9 (no registered evaluator)): composition not possible in this replay (MAN-06g, v2)"] == LP + T
    assert nef2["Gemm_v9 (structure / interior: no identity)"] == LP + T                             # the interior Gemm itself: counted, no VU
    assert {v["stratum"] for v in vus2} == {v["stratum"] for v in vus1 if v["family"] != "BiasAdd_v1"}  # every other stratum untouched
    # the replay of the whole record is COMPLETE with the gap on the record by name (the verdict's O1 is satisfied by form (A) itself)
    progs = {rid: inst for rid in _programs(biased_world)}
    bmap, committed = biased_world.bmap_and_reader()
    rep = SR.sampled_replay(progs, bmap, committed, _weights(biased_world), per_stratum=8, seed=1, workers=0)
    assert rep["result"] is True and rep["grade"] == "complete" and rep["sample"]["not_evaluated_n"] == 0, rep.get("why")
    assert any(u["family"].startswith("BiasAdd_v1 (fed by interior Gemm_v9") and u["n"] == 2 * (LP + T) for u in rep["unreplayable"])
    assert "BiasAdd_v1" not in rep["timing"]["per_family_n"]
    # why classes on the record
    assert rep["why_classes"]["by_name"] == list(RC.WHY_CLASSES_BY_NAME) and rep["unevaluated_strata"] == []
    assert RC.why_class("arg slice ['n', 5, 0, 2048) is not inside one committed member of row 5") == "mapping"
    assert RC.why_class("evaluator error OSError: librms_triton_model.so: file too short") == "evaluator-error"
    assert RC.why_class("fed by moe_block_stream planes (MoeSum_v1): ...") == "plane-fed"
    assert RC.why_class("no derived_rows evaluator for RoPE_v1 yet (v2)") == "no-evaluator"


# ------------------------------------------------------------------------------------------------
# the Hopper GEMM `Gemm_v2{K,N,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}` (targets.gemm_spec on a cc 9.0 step; the spec text of every Hopper row
# of record, e.g. qwen25-15b H100 B1 1024/128) has a registered evaluator: the same VU as Gemm_v1 (one token row x[K] @ W[N,K]^T
# -> [N]) under the wgmma k16 step (one group of 16, 26-bit adder, floor -133).  Positive (honest Hopper world replays exact; the biased
# linear composes), negatives (a flipped committed input byte -> mismatch named; Ampere-committed bytes under a Hopper Program -> the two
# adders are told apart on an adversarial lane; an unregistered / absent DOT -> unresolved BY NAME, never another chain).

HOPPER_DOT = 'DOT={"fn":"HopperBF16WgmmaDot16_v1"}'
AMPERE_DOT = 'DOT={"fn":"AmpereBF16TcDot16_v2"}'


def _hopper_program(inst: dict, dot: str = HOPPER_DOT) -> dict:
    """Every Gemm_v1{K,N} row -> Gemm_v2{K,N,<dot>} (the statics-lifted GEMM a non-(Ampere, FA2) target binds)."""
    out = json.loads(json.dumps(inst))
    for r in out["rows"]:
        if r["spec"].startswith("Gemm_v1{"):
            r["spec"] = "Gemm_v2{" + r["spec"][len("Gemm_v1{"):-1] + "," + dot + "}"
    return out


@pytest.fixture(scope="module")
def hopper_worlds():
    """(plain, biased) honest worlds executed under the Hopper GEMM relation (Honest runs EV.evaluate, so the committed bytes ARE the
    wgmma chain's words), with the same seeds / prompts as `world` / `biased_world`."""
    outs = []
    for bias, seed in ((False, 7), (True, 11)):
        rng = np.random.default_rng(seed)
        inst, woff = _program(bias=bias)
        inst = _hopper_program(inst)
        n_w = max(hi for lo, hi in woff.values())
        weights = _bf16(rng, n_w, 0.5)
        weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80
        weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
        outs.append(Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]}))
    return tuple(outs)


def test_hopper_gemm_v2_is_registered_and_dispatches_on_the_dot_static():
    from verity.ml.definitions.numerics import derived_rows as D
    assert "Gemm_v2" in RK.EVALUATORS and "Gemm_v2" not in RC.NOT_YET and SR._FAMILY_LIBS["Gemm_v2"] == ()
    assert "dot_bf16_for" in RK.EVALUATORS["Gemm_v2"] and "HopperBF16WgmmaDot16_v1" in RK.EVALUATORS["Gemm_v2"]
    assert RK.gemm_dot_prim({"K": 16, "N": 32, "DOT": '{"fn":"HopperBF16WgmmaDot16_v1"}'}) == "HopperBF16WgmmaDot16_v1"
    assert RK.gemm_dot_prim({"DOT": "AmpereBF16TcDot16_v2"}) == "AmpereBF16TcDot16_v2"
    for bad in ({}, {"DOT": "k16"}, {"DOT": '{"fn":"Hopper; drop"}'}, {"DOT": 12}):
        with pytest.raises(EV._Unresolved, match="does not name a registered primitive"):
            RK.gemm_dot_prim(bad)
    # the spec text of a real Hopper row parses to the same statics the Build wrote (codec._spec_id form)
    st = RM.spec_statics('Gemm_v2{K=2048,N=6144,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}')
    assert st == {"K": 2048, "N": 6144, "DOT": '{"fn":"HopperBF16WgmmaDot16_v1"}'} and RK.gemm_dot_prim(st) == "HopperBF16WgmmaDot16_v1"
    assert set(D.TC_DOT16_BY_PRIM) == {"AmpereBF16TcDot16_v2", "HopperBF16WgmmaDot16_v1"}


def test_hopper_gemm_v2_evaluator_is_the_registered_wgmma_chain_not_the_ampere_one():
    """Word for word against the registered scalar primitive (`hopper.HopperBF16WgmmaDot16`, pinned on the H100) on random finite lanes,
    and on an adversarial lane where the 26-bit and 25-bit adders round to DIFFERENT bf16 words: evaluate('Gemm_v2', Hopper DOT) returns
    the wgmma word, evaluate('Gemm_v1') / ('Gemm_v2', Ampere DOT) the mma.sync word."""
    from verity.ml.definitions.numerics import derived_rows as D
    from verity.ml.definitions.registry.hopper import HopperBF16WgmmaDot16
    rng = np.random.default_rng(1701)
    K, N = 64, 8
    x = _bf16(rng, K, 1.0)
    w = _bf16(rng, N * K, 1.0)
    st_h = {"K": K, "N": N, "DOT": '{"fn":"HopperBF16WgmmaDot16_v1"}'}
    got = EV.evaluate("Gemm_v2", st_h, [[x], [w]], [None, w])["0"]
    for n in range(N):                                                                  # the scalar registered chain, coordinate by coordinate
        acc = 0
        for s in range(0, K, 16):
            acc = HopperBF16WgmmaDot16.evaluate(acc, *(int(v) for v in x[s:s + 16]), *(int(v) for v in w[n * K + s: n * K + s + 16]))
        assert int(got[n]) == int(D.f2fp_bf16(np.asarray([acc], dtype=np.uint32))[0]), n
    # an adversarial lane: search random 16-wide operand pairs until the two pipelines' f32 words round to different bf16 words
    found = None
    for _ in range(200):
        a = _bf16(rng, (4096, 16), 1.0)
        b = _bf16(rng, (4096, 16), 1.0)
        z = np.zeros(4096, dtype=np.uint32)
        h, _ = D.tc_dot16_hopper(z, a, b)
        m, _ = D.tc_dot16(z, a, b)
        diff = np.flatnonzero(D.f2fp_bf16(h) != D.f2fp_bf16(m))
        if diff.size:
            found = (a[diff[0]], b[diff[0]], int(D.f2fp_bf16(h)[diff[0]]), int(D.f2fp_bf16(m)[diff[0]]))
            break
    assert found is not None, "no lane where the Hopper and Ampere adders differ after bf16 rounding (search budget exhausted)"
    xa, wb, want_h, want_m = found
    st16 = {"K": 16, "N": 1}
    assert int(EV.evaluate("Gemm_v2", st16 | {"DOT": '{"fn":"HopperBF16WgmmaDot16_v1"}'}, [[xa], [wb]], [None, wb])["0"][0]) == want_h
    assert int(EV.evaluate("Gemm_v2", st16 | {"DOT": '{"fn":"AmpereBF16TcDot16_v2"}'}, [[xa], [wb]], [None, wb])["0"][0]) == want_m
    assert int(EV.evaluate("Gemm_v1", st16, [[xa], [wb]], [None, wb])["0"][0]) == want_m and want_h != want_m
    # an unregistered step / a non-finite operand lane: unresolved BY NAME, never a word under another adder
    with pytest.raises(EV._Unresolved, match="no registered evaluator for Gemm_v2 with DOT=BlackwellBF16TcDot16_v1"):
        EV.evaluate("Gemm_v2", st16 | {"DOT": '{"fn":"BlackwellBF16TcDot16_v1"}'}, [[xa], [wb]], [None, wb])
    xn = xa.copy()
    xn[3] = 0x7F80                                                                       # +inf operand
    with pytest.raises(EV._Unresolved, match="non-finite bf16 operand"):
        EV.evaluate("Gemm_v2", st16 | {"DOT": '{"fn":"HopperBF16WgmmaDot16_v1"}'}, [[xn], [wb]], [None, wb])


def test_hopper_honest_program_replays_bit_exactly_gemm_v2_in_the_population(hopper_worlds):
    plain, _biased = hopper_worlds
    bmap, committed = plain.bmap_and_reader()
    rep = SR.sampled_replay(_programs(plain), bmap, committed, _weights(plain), per_stratum=8, seed=3, workers=0)
    assert rep["result"] is True and rep["grade"] == "complete", json.dumps(rep, indent=1, default=str)[:3000]
    assert rep["sample"]["mismatch_n"] == 0 and rep["sample"]["not_evaluated_n"] == 0 and rep["sample"]["evaluated"] == rep["sample"]["picked"] > 0
    fams = set(rep["population"]["by_family"])
    assert fams == {"Embedding_v1", "RMSNormTriton_v1", "Gemm_v2", "SiluMul_v1", "RMSNormFusedCuda_v2", "TokenSelect_v1"}
    assert "Gemm_v1" not in fams and rep["population"]["not_evaluable"] == {} and rep["unreplayable"] == []
    assert rep["timing"]["per_family_n"]["Gemm_v2"] == 2 * (2 * (LP + T) + (1 + T))        # gate_up + down per token row, logits per step, x2 requests
    assert rep["evaluators"]["Gemm_v2"].startswith("verity.ml.definitions.numerics.derived_rows.dot_bf16_for(DOT)")
    assert "Gemm_v2" not in rep["applicability"]["families_not_yet"]
    # a Gemm_v2 VU's recorded inputs are ONE committed identity (the norm output row) + ONE registered weight slice: explicit, resolvable
    pi = IX.ProgramIndex(plain.inst)
    g = next(r for r in pi.rows if r["spec"].startswith("Gemm_v2{") and "gate_up_proj" in r["name"])
    one = EV.replay_vu(pi, OP.CommittedStore(bmap, committed), _weights(plain), {"i": g["i"], "phase": "prefill", "request_id": "r0"}, arrive=0)
    assert one["result"] is True and [i["kind"] for i in one["inputs"]] == ["committed", "weights"], one
    assert one["inputs"][0]["identity"][2:4] == ["model.layers.0.input_layernorm", "out"] and one["inputs"][1]["field"] == "model.model.layers.0.mlp.gate_up_proj.weight"


def test_hopper_biased_linear_composes_the_interior_gemm_v2(hopper_worlds):
    """The interior-composed gap of record (32,228 'BiasAdd_v1 fed by interior Gemm_v2' VUs on the Qwen2.5 H100 row) closes:
    with the Gemm_v2 evaluator the BiasAdd VU composes the interior wgmma GEMM from ITS committed inputs and replays exact."""
    _plain, biased = hopper_worlds
    pi = IX.ProgramIndex(biased.inst)
    ba = next(r for r in pi.rows if r["spec"].startswith("BiasAdd_v1"))
    assert POP._interior_feeders_without_evaluator(pi, ba) == set()
    vus, not_ev, not_ev_fam = POP.population(pi, "r0")
    assert sum(1 for v in vus if v["family"] == "BiasAdd_v1") == LP + T and not not_ev
    assert not_ev_fam["Gemm_v2 (interior: composed into its epilogue's VU / a hinted row of the module is its returned tensor)"] == LP + T
    assert not any("fed by interior" in k for k in not_ev_fam)
    bmap, committed = biased.bmap_and_reader()
    rep = SR.sampled_replay(_programs(biased), bmap, committed, _weights(biased), per_stratum=8, seed=1, workers=0)
    assert rep["result"] is True and rep["grade"] == "complete" and rep["sample"]["mismatch_n"] == 0 and rep["sample"]["not_evaluated_n"] == 0, rep.get("why")
    assert rep["timing"]["per_family_n"]["BiasAdd_v1"] == 2 * (LP + T)
    one = EV.replay_vu(pi, OP.CommittedStore(bmap, committed), _weights(biased), {"i": ba["i"], "phase": "prefill", "request_id": "r0"}, arrive=0)
    assert one["result"] is True and any(i["kind"] == "interior" and i["family"] == "Gemm_v2" for i in one["inputs"])
    assert rep["why_classes"]["by_name"] == list(RC.WHY_CLASSES_BY_NAME) and not any("interior-composed" in str(u) for u in rep["unreplayable"])


def test_hopper_flipped_committed_input_byte_fails_named_at_the_gemm_v2_row(hopper_worlds):
    """Negative of record: one bit of the committed norm output (the gate_up Gemm_v2's INPUT) of r1's prompt row 1 flips -> the Gemm_v2 VU
    reading it mismatches, named at the request / step / row; r0 and the other rows stay equal."""
    plain, _biased = hopper_worlds

    def mutate(store, entries):
        k = (0, "model.layers.0.input_layernorm/out")
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r1" and x["name"] == k[1] and x["identity"]["step"] == 0)
        pos = (e["element_offset"] + 1 * H + 3) * 2                                       # r1, prompt row 1, element 3 of the norm output
        b[pos] ^= 0x01
        store[k] = bytes(b)
    bmap, committed = plain.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(plain), bmap, committed, _weights(plain), per_stratum=8, seed=1, workers=0)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    m = [x for x in rep["mismatch"] if x["spec"].startswith(f"Gemm_v2{{K={H},N={2 * I},")]
    assert m and all(x["request_id"] == "r1" and x["step"] == 0 and x["row"] == 1 for x in m), rep["mismatch"]
    assert "F_V(committed inputs) != committed output" in rep["why"] and "Gemm_v2" in rep["why"]
    assert not any(x["request_id"] == "r0" for x in rep["mismatch"])
    # the flipped value's PRODUCER (the norm row: F(embedding) != the flipped word) and its CONSUMER (the Gemm_v2 row) are both named, both
    # at r1 row 1 step 0; nothing else moves (r1's other rows read untouched words)
    assert all(x["request_id"] == "r1" and x["row"] == 1 and x["step"] == 0 for x in rep["mismatch"])
    assert {x["spec"].split("{")[0] for x in rep["mismatch"]} <= {"RMSNormTriton_v1", "Gemm_v2", "SiluMul_v1", "RMSNormFusedCuda_v2"}


def test_hopper_program_over_ampere_committed_bytes_is_told_apart_where_the_adders_differ(hopper_worlds, world):
    """Cross-architecture negative: the bytes of `world` were committed under Gemm_v1 (mma.sync chain); read under a Program that declares
    Gemm_v2{..,DOT=Hopper} they must NOT all pass -- the replay compares against the wgmma chain.  At these toy shapes (K=16) the two
    adders round to the same bf16 word on most rows, so the discriminating fact is asserted on the adversarial lane of the evaluator
    test; here the whole-record reading is recorded: result True only if EVERY sampled Gemm coordinate coincided (then the Program's
    declared DOT is checked by the evaluator identity, `rep['evaluators']['Gemm_v2']`), never a silent acceptance of another chain."""
    progs = {rid: _hopper_program(world.inst) for rid in world.prompts}
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=8, seed=1, workers=0)
    assert rep["evaluators"]["Gemm_v2"].startswith("verity.ml.definitions.numerics.derived_rows.dot_bf16_for(DOT)")
    if rep["result"] is False:
        assert rep["grade"] == "mismatch" and all(x["spec"].startswith(("Gemm_v2{", "SiluMul", "RMSNormFusedCuda", "TokenSelect")) for x in rep["mismatch"])
    else:
        assert rep["sample"]["mismatch_n"] == 0                                             # coincidence at K=16: recorded, not a soundness claim


def test_model_weights_provider_reads_buffers_as_well_as_parameters_the_set_the_weights_root_pins():
    """`model_weights_provider` is the replay's registered-input reader on the pods; the rotary table
    `...rotary_emb.cos_sin_cache` is a torch BUFFER (vLLM: `register_buffer(..., persistent=False)`, one shared RotaryEmbedding instance
    named once at layer 0).  A parameters-only reader returned None for it -> every RoPE_v1 row 'registered weight field ... not available
    to the replay' (300/300 strata on the smollm2 canary).  The reader's set must equal the set `register_weights` pins:
    `named_parameters()` + `named_buffers()`; the wrapper prefix rule is unchanged; an unknown field is still None (unresolved by name)."""
    torch = pytest.importorskip("torch")

    class Rot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cos_sin_cache", torch.arange(0, 8, dtype=torch.float32).to(torch.bfloat16), persistent=False)

    class Attn(torch.nn.Module):
        def __init__(self, rot):
            super().__init__()
            self.rotary_emb = rot
            self.qkv_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            rot = Rot()
            self.layers = torch.nn.ModuleList([Attn(rot), Attn(rot)])       # ONE shared rotary instance, as in vLLM

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

    m = Wrapper()
    read = OP.model_weights_provider(m, "model.")
    # the buffer, under the qualname `named_buffers()` gives it (layer 0; the Program names the shared table once)
    cs = read("model.model.layers.0.rotary_emb.cos_sin_cache", 0, 8)
    assert cs is not None and cs.dtype == np.uint16 and cs.shape == (8,)
    want = m.model.layers[0].rotary_emb.cos_sin_cache.view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(cs, want)
    assert read("model.model.layers.0.rotary_emb.cos_sin_cache", 2, 5).tolist() == want[2:5].tolist()      # a slice of the field
    # a parameter still resolves, in its own width, through the same prefix rule
    w = read("model.model.layers.1.qkv_proj.weight", 0, 16)
    assert w is not None and w.dtype == np.uint16 and w.size == 16
    assert np.array_equal(w, m.model.layers[1].qkv_proj.weight.detach().contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1))
    # what the weights root pins == what the reader can serve (names), and a field in neither is None
    pinned = {n for n, _ in list(m.named_parameters()) + list(m.named_buffers())}
    assert "model.layers.0.rotary_emb.cos_sin_cache" in pinned and "model.layers.1.rotary_emb.cos_sin_cache" not in pinned
    for n in pinned:
        assert read("model." + n, 0, 1) is not None
    assert read("model.model.layers.1.rotary_emb.cos_sin_cache", 0, 8) is None                              # not a pinned qualname: unresolved, never inferred
    assert read("model.model.layers.0.rotary_emb.nothing", 0, 1) is None
    assert read("model.model.layers.0.rotary_emb.cos_sin_cache", 0, 9) is None                              # past the field's end


def test_model_weights_provider_serves_an_int32_table_as_its_u32_words():
    """DeepSeek-V4's hash-routed layers register `ffn.gate.tid2eid` as an int32 Parameter [V, 6]; the in-place reader serves its words
    as the deferred replay does (`cpu_replay.WORDS`), so a `GatherI32x{V}` row reads the table either way, and an int64 one stays None."""
    torch = pytest.importorskip("torch")

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tid2eid = torch.nn.Parameter(torch.tensor([[3, -1, 7], [2, 0, 5]], dtype=torch.int32), requires_grad=False)
            self.wide = torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.int64), requires_grad=False)

    read = OP.model_weights_provider(Gate(), "")
    got = read("tid2eid", 0, 6)
    assert got.dtype == np.uint32 and got.tolist() == [3, 0xFFFFFFFF, 7, 2, 0, 5]
    assert read("wide", 0, 2) is None


# --- RoPE_v1: an evaluator, not a NOT_YET label -----------------------------------------------------------------

def test_rope_v1_is_registered_and_the_rows_are_in_the_population_not_named_away(rope_world):
    assert "RoPE_v1" in RK.EVALUATORS and "RoPE_v1" not in RC.NOT_YET and RK.EVALUATORS["RoPE_v1"].startswith("verity.ml.definitions.numerics.derived_rows.rope_rows")
    pi = rope_world.pi
    vus, ne, nef = POP.population(pi, "r0")
    ropes = [v for v in vus if v["spec"].startswith("RoPE_v1")]
    assert len(ropes) == 2 * (LP + T) and not any("RoPE" in k for k in list(ne) + list(nef))
    assert {v["spec"] for v in ropes} == {f"RoPE_v1{{NHEADS={NH},D={DH}}}", f"RoPE_v1{{NHEADS={KVH},D={DH}}}"}
    # the manifest names the member by the row's hint: query -> `0`, key -> `1`; both carry NHEADS*D bf16 words
    q = next(r for r in pi.rows if r["spec"].startswith("RoPE_v1") and "/query" in r["name"])
    k = next(r for r in pi.rows if r["spec"].startswith("RoPE_v1") and "/key" in r["name"])
    assert pi.describe(q)["members"] == [("query", NH * DH, "bf16")] and pi.describe(k)["members"] == [("key", KVH * DH, "bf16")]


def test_rope_v1_honest_program_replays_bit_exactly_with_the_cache_row_read_from_weights(rope_world):
    bmap, committed = rope_world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(rope_world), bmap, committed, _weights(rope_world), per_stratum=8, seed=5)
    assert rep["result"] is True and rep["grade"] == "complete", json.dumps(rep, indent=1, default=str)[:3000]
    assert rep["sample"]["mismatch_n"] == 0 and rep["sample"]["evaluated"] == rep["sample"]["picked"]
    assert {f"RoPE_v1{{NHEADS={NH},D={DH}}}", f"RoPE_v1{{NHEADS={KVH},D={DH}}}"} <= set(rep["population"]["by_family"]) or "RoPE_v1" in rep["population"]["by_family"]
    rs = {k: v for k, v in rep["strata"].items() if k.startswith("RoPE_v1")}
    assert rs and all(v["evaluated"] >= 1 and v["evaluated"] == v["sampled"] for v in rs.values()), rs
    assert not any(k.startswith("RoPE_v1") for k in rep.get("strata_unevaluated") or {}) and not rep["mismatch"]
    assert rep["evaluators"]["RoPE_v1"].startswith("verity.ml.definitions.numerics.derived_rows.rope_rows")


def test_rope_v1_flipped_committed_output_word_fails_named_at_the_rope_row(rope_world):
    def mutate(store, entries):
        k = (0, "model.layers.0.self_attn.rotary_emb/key")                     # the KEY member of step 0
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r1" and x["name"] == k[1] and x["identity"]["step"] == 0)
        b[(e["element_offset"] + 1 * KVH * DH + 3) * 2 + 1] ^= 0x01               # r1, prompt row 1, word 3: one exponent/mantissa bit
        store[k] = bytes(b)
    bmap, committed = rope_world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(rope_world), bmap, committed, _weights(rope_world), per_stratum=8, seed=1)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    m = rep["mismatch"]
    assert m and all(x["spec"] == f"RoPE_v1{{NHEADS={KVH},D={DH}}}" and x["request_id"] == "r1" and x["row"] == 1 and x["step"] == 0 for x in m), m
    assert "F_V(committed inputs) != committed output" in rep["why"]


def test_rope_v1_wrong_cache_row_is_a_mismatch_the_position_is_the_programs_slice_not_inferred(rope_world):
    """The committed rotated heads were produced with cos_sin_cache[pos]; a Program whose RoPE row names ANOTHER cache row (pos+1) for one
    token fails at exactly that row -- the position enters only through the Program's `weights` slice."""
    progs = {}
    for rid in rope_world.prompts:
        inst = json.loads(json.dumps(rope_world.inst))
        if rid == "r0":
            r = next(x for x in inst["rows"] if x["spec"].startswith("RoPE_v1") and x["name"].endswith("rotary_embedding_0/query[1]"))
            r["args"][1][0][2] += DH; r["args"][1][0][3] += DH
        progs[rid] = inst
    bmap, committed = rope_world.bmap_and_reader()
    rep = SR.sampled_replay(progs, bmap, committed, _weights(rope_world), per_stratum=8, seed=1)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    assert rep["mismatch"] and all(x["spec"] == f"RoPE_v1{{NHEADS={NH},D={DH}}}" and x["request_id"] == "r0" and x["row"] == 1 and x["step"] == 0
                                   for x in rep["mismatch"]), rep["mismatch"]


def test_rope_v1_evaluator_equals_the_registered_prims_on_the_replays_own_rows(rope_world):
    """The replay's RoPE_v1 branch == prims.RopeOut / RopeOutAdd per coordinate on the honest Program's own committed q / k words."""
    from verity.ml.definitions.registry import prims as P
    pi = rope_world.pi
    r = next(x for x in pi.rows if x["spec"].startswith("RoPE_v1") and "/query" in x["name"])
    d = pi.describe(r)
    x = rope_world.values[("r0", r["args"][0][0][1])][r["args"][0][0][2]: r["args"][0][0][3]]
    lo, hi = r["args"][1][0][2], r["args"][1][0][3]
    cs = rope_world.weights[lo:hi]
    got = EV.evaluate("RoPE_v1", d["statics"], [[x], []], [None, cs])["0"]
    R = DH // 2
    for h in range(NH):
        for j in range(R):
            xi, yi = int(x[h * DH + j]), int(x[h * DH + j + R])
            assert int(got[h * DH + j]) == P.RopeOut.evaluate(xi, yi, int(cs[j]), int(cs[j + R]))
            assert int(got[h * DH + j + R]) == P.RopeOutAdd.evaluate(xi, yi, int(cs[j]), int(cs[j + R]))
    assert np.array_equal(got, rope_world.values[("r0", r["i"])])


# --- Attention_v3 / Attention_v2 row relation: state-linked resolution over the committed K/V history ------------------

@pytest.fixture(scope="module")
def attn_world():
    """The rope Program with an `Attention_v3{T=pos+1,...,BN=2}` row per token, reading the KEY members / V slices of every earlier position
    (state-linked: the decode rows read step 0's and the previous decode steps' committed words by ref)."""
    rng = np.random.default_rng(23)
    inst, woff = _program(attn=True)
    n_w = max(hi for lo, hi in woff.values())
    weights = _bf16(rng, n_w, 0.5)
    weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80
    weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
    lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
    weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


def test_attention_rows_are_registered_units_of_the_population_and_the_interior_mechanism_is_named(attn_world):
    """Attention_v3 / Attention_v2 are EVALUATORS (the row is the Q_module_v1 unit), not NOT_YET labels: every attention row
    is a VU of the population; the record names the AttnBlock-interior mechanism (Q_fine_v1 units) the replay does not read, with the count."""
    assert all(f in RK.EVALUATORS and f not in RC.NOT_YET for f in ("Attention_v3", "Attention_v2"))
    assert RK.EVALUATORS["Attention_v2"].startswith("verity.ml.definitions.numerics.derived_rows.attention_rows") and "Fa3InvSum_v1" in RK.EVALUATORS["Attention_v2"]
    assert "MAN-09c" in RK.EVALUATORS["RoPE_v1"]                                    # the cos_sin bound travels with the row
    n_att = sum(1 for r in attn_world.pi.rows if r["spec"].startswith("Attention_v3"))
    assert n_att == LP + T
    bmap, committed = attn_world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(attn_world), bmap, committed, _weights(attn_world), per_stratum=8, seed=5)
    assert "Attention_v3" not in rep["population"]["not_evaluable_by_family"]
    assert not any("K/V history" in k or "state-linked" in k for k in rep["population"]["not_evaluable"]), rep["population"]["not_evaluable"]
    assert rep["population"]["by_family"].get("Attention_v3") == 2 * n_att
    um = rep["uncovered_mechanisms"]
    assert um and um[0]["rows"] == 2 * n_att and um[0]["query_of_record"] == "Q_module_v1" and "M1 export stream" in um[0]["why"]
    assert rep["applicability"]["families_not_yet"] == {}                          # nothing of this Program is a label


def _attn_world(registered: bool) -> Honest:
    rng = np.random.default_rng(23)
    inst, woff = _program(attn=True, registered=registered)
    n_w = max(hi for lo, hi in woff.values())
    weights = _bf16(rng, n_w, 0.5)
    weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80
    weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
    lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
    weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


def test_registered_values_program_replays_like_its_constants_twin(attn_world):
    """`TargetProfile.registered_values`: the norms' eps and the attention scale are fields of the registered `values` parameter, which is
    neither the prompt nor a weight.  The honest run commits the same words as the constants Program's, and the replay evaluates every unit
    of the same population, reading each value from the build's supplied layout."""
    reg = _attn_world(True)
    assert reg.pi.ids_params() == {0} and set(reg.pi.supplied) == {1}
    assert reg.blocks.keys() == attn_world.blocks.keys()
    assert all(np.array_equal(reg.blocks[k], attn_world.blocks[k]) for k in reg.blocks)
    reps = {}
    for name, w in (("const", attn_world), ("registered", reg)):
        bmap, committed = w.bmap_and_reader()
        reps[name] = SR.sampled_replay(_programs(w), bmap, committed, _weights(w), per_stratum=64, seed=5, workers=0)
    rc, rr = reps["const"], reps["registered"]
    assert rr["result"] is True and rr["grade"] == "complete", json.dumps(rr, indent=1, default=str)[:3000]
    assert rr["sample"]["mismatch_n"] == 0 and rr["sample"]["evaluated"] == rr["sample"]["picked"] == rc["sample"]["picked"]
    rename = {"RMSNormTriton_v1": "RMSNormTriton_v3", "RMSNormFusedCuda_v2": "RMSNormFusedCuda_v4", "Attention_v3": "Attention_v10"}
    assert rr["population"]["by_family"] == {rename.get(f, f): n for f, n in rc["population"]["by_family"].items()}
    assert rr["population"]["not_evaluable"] == rc["population"]["not_evaluable"] == {}
    vu = next(r for r in reg.pi.rows if r["spec"].startswith("Attention_v10"))
    rec: list = []
    EV._evaluate_row(reg.pi, OP.CommittedStore(*reg.bmap_and_reader()), _weights(reg), vu, "r0", 0, rec, depth=0)
    assert rec[-1] == {"kind": "registered", "param": "values", "field": f"scale_log2_d{DH}", "bits": reg.inst["supplied"]["registration"]["fields"][1]["bits"]}
    # the query names no supplied value: registration binds them, the run commits the same identities as the constants Program's
    key = lambda m: sorted((e["family"], e["op_path"], e["output_member"], str(e["step"]), tuple(e["element_range"])) for e in m["identities"])  # noqa: E731
    man = v2_manifest(reg.inst)
    assert key(man) == key(v2_manifest(attn_world.inst))
    bmap, committed = reg.bmap_and_reader()
    rm = SR.sampled_replay(_programs(reg), bmap, committed, _weights(reg), per_stratum=64, seed=5, workers=0, manifest=man)
    assert rm["result"] is True and rm["sample"]["mismatch_n"] == 0, json.dumps(rm, indent=1, default=str)[:3000]


def test_attention_v3_honest_program_replays_bit_exactly_from_the_committed_kv_history(attn_world):
    bmap, committed = attn_world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(attn_world), bmap, committed, _weights(attn_world), per_stratum=8, seed=5)
    assert rep["result"] is True and rep["grade"] == "complete", json.dumps(rep, indent=1, default=str)[:3000]
    assert rep["sample"]["mismatch_n"] == 0 and rep["sample"]["evaluated"] == rep["sample"]["picked"]
    at = {k: v for k, v in rep["strata"].items() if k.startswith("Attention_v3")}
    assert at and all(v["evaluated"] >= 1 and v["evaluated"] == v["sampled"] for v in at.values()), at
    # the decode rows (T = 3, 4) DID resolve: their K/V inputs are step-0 / step-1 committed members of the request.  The stratum key
    # carries NB = ceil(T / BN) + last=full|partial in place of T (one partial block here: T <= 4 < BN), the phase axis keeps the decode
    # rows apart from the prefill rows
    assert any(k.startswith("Attention_v3{NB=1,last=partial,") and "|last|" in k for k in at), sorted(at)
    assert not any("{T=" in k for k in at) and {k.split("|")[3] for k in at} == {"prefill", "first-decode", "last"}, sorted(at)
    assert not any(k.startswith("Attention_v3") for k in rep.get("strata_unevaluated") or {}) and not rep["mismatch"]
    assert rep["evaluators"]["Attention_v3"].startswith("verity.ml.definitions.numerics.derived_rows.attention_rows")


def test_attention_stratum_key_keeps_the_block_count_and_the_full_or_partial_last_block_apart_and_nothing_else():
    """NB alone mixes T = 128 (full last block: no masked lane) with T = 127 (masked lanes -> exact zeros through exp2 and the
    lane sums); the key carries `last=full|partial` so the full-block chain is a stratum of its own again (<= +1 stratum per formerly mixed
    stratum).  The VU keeps its spec (T); only the stratum key changes.  Every other family's spec is its own key."""
    spec = "Attention_v3{T=%d,NH=32,KVH=8,D=64,BN=128}"
    key = lambda t, bn=128: POP._stratum_spec("Attention_v3", spec % t, {"T": t, "NH": 32, "KVH": 8, "D": 64, "BN": bn})
    assert key(128) == "Attention_v3{NB=1,last=full,NH=32,KVH=8,D=64,BN=128}"
    assert key(127) == "Attention_v3{NB=1,last=partial,NH=32,KVH=8,D=64,BN=128}"
    assert key(1) == key(127) and key(129) == "Attention_v3{NB=2,last=partial,NH=32,KVH=8,D=64,BN=128}"
    assert key(256) == "Attention_v3{NB=2,last=full,NH=32,KVH=8,D=64,BN=128}" and key(384, 192) == "Attention_v3{NB=2,last=full,NH=32,KVH=8,D=64,BN=128}"
    assert key(128) != key(127) != key(129) and len({key(t) for t in range(1, 1025)}) == 16   # 8 block counts x {full, partial}
    assert POP._stratum_spec("Attention_v2", "Attention_v2{T=5,NH=12,KVH=2,D=128,BN=128}", {"T": 5, "BN": 128}) == "Attention_v2{NB=1,last=partial,NH=12,KVH=2,D=128,BN=128}"
    assert POP._stratum_spec("Attention_v3", spec % 7, {"NH": 32}) == spec % 7                    # no T / BN static -> the spec itself
    assert POP._stratum_spec("Attention_v3", spec % 7, {"T": "seven", "BN": 128}) == spec % 7    # unparsable -> the spec itself, never a guess
    assert POP._stratum_spec("Gemm_v2", "Gemm_v2{K=1536,N=2048}", {"T": 7, "BN": 128}) == "Gemm_v2{K=1536,N=2048}"


def test_attention_v3_flipped_committed_output_word_fails_named_at_the_attention_row(attn_world):
    def mutate(store, entries):
        k = (T, "model.layers.0.self_attn.attn/out")                               # the last decode step's attention output
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r1" and x["name"] == k[1] and x["identity"]["step"] == T)
        b[(e["element_offset"] + 5) * 2 + 1] ^= 0x01                              # r1, head 0 word 5: one exponent/mantissa bit
        store[k] = bytes(b)
    bmap, committed = attn_world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(attn_world), bmap, committed, _weights(attn_world), per_stratum=8, seed=1)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    m = rep["mismatch"]
    assert m and all(x["spec"].startswith(f"Attention_v3{{T={LP + T},") and x["request_id"] == "r1" and x["step"] == T for x in m), m
    assert "F_V(committed inputs) != committed output" in rep["why"]


def test_attention_v3_flipped_key_word_of_an_earlier_step_is_named_at_the_rows_that_read_it(attn_world):
    """The state-linked negative: a fault in the committed KEY history (step 0, prompt row 1) is a RoPE_v1 mismatch at step 0 AND an
    Attention_v3 mismatch at every later row that reads that key (step 1, step 2) -- the history is read by ref, never trusted."""
    def mutate(store, entries):
        k = (0, "model.layers.0.self_attn.rotary_emb/key")
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r0" and x["name"] == k[1] and x["identity"]["step"] == 0)
        b[(e["element_offset"] + 1 * KVH * DH + 2) * 2] ^= 0x40                    # r0, prompt row 1, word 2: a mantissa bit
        store[k] = bytes(b)
    bmap, committed = attn_world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(attn_world), bmap, committed, _weights(attn_world), per_stratum=8, seed=1)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    m = rep["mismatch"]
    fams = {(x["spec"].split("{")[0], x["step"]) for x in m}
    assert all(x["request_id"] == "r0" for x in m), m
    assert ("RoPE_v1", 0) in fams, fams                                            # the key row itself: F(qkv, cache) != flipped word
    assert {("Attention_v3", 1), ("Attention_v3", 2)} <= fams, fams                # every later attention row that consumed the flipped key
    # the prompt row 1 attention at step 0 reads key 1 too (T = 2): named as well
    assert ("Attention_v3", 0) in fams, fams


def test_attention_v3_flipped_value_word_of_the_prefill_is_named_at_every_row_that_reads_it(attn_world):
    """A flipped committed V word (the qkv member of step 0, prompt row 0, the V slice) propagates: named at the
    qkv Gemm row itself (F(x, W) != committed) AND at every attention row of the request that reads position 0's V (all four)."""
    def mutate(store, entries):
        k = (0, "model.layers.0.self_attn.qkv_proj/out")
        b = bytearray(store[k])
        e = next(x for x in entries if x["identity"]["request_id"] == "r1" and x["name"] == k[1] and x["identity"]["step"] == 0)
        b[(e["element_offset"] + 0 * (NH + 2 * KVH) * DH + (NH + KVH) * DH + 4) * 2 + 1] ^= 0x01   # r1, prompt row 0, V word 4
        store[k] = bytes(b)
    bmap, committed = attn_world.bmap_and_reader(mutate=mutate)
    rep = SR.sampled_replay(_programs(attn_world), bmap, committed, _weights(attn_world), per_stratum=8, seed=3)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    m = rep["mismatch"]
    assert all(x["request_id"] == "r1" for x in m), m
    fams = {(x["spec"].split("{")[0], x["step"]) for x in m}
    assert ("Gemm_v1", 0) in fams, fams
    assert {("Attention_v3", 0), ("Attention_v3", 1), ("Attention_v3", 2)} <= fams, fams


@pytest.mark.slow
def test_attention_v3_program_naming_the_wrong_position_for_a_key_is_a_mismatch_and_a_short_history_is_refused_by_name(attn_world):
    """The history is the Program's refs, never inferred: a Program whose last attention row names the key of
    position 2 in position 1's slot (a duplicated ref) evaluates F over THAT history != the committed output -> mismatch at exactly that
    row; a row whose refs do not add up to T (one key missing) is unresolved BY NAME -- never evaluated over a padded / guessed history."""
    progs = {}
    for rid in attn_world.prompts:
        inst = json.loads(json.dumps(attn_world.inst))
        if rid == "r0":
            r = next(x for x in inst["rows"] if x["spec"].startswith(f"Attention_v3{{T={LP + T},"))
            r["args"][1][1] = list(r["args"][1][2])                                # key slot 1 <- the ref of position 2
        progs[rid] = inst
    bmap, committed = attn_world.bmap_and_reader()
    rep = SR.sampled_replay(progs, bmap, committed, _weights(attn_world), per_stratum=8, seed=1)
    assert rep["result"] is False and rep["grade"] == "mismatch"
    assert rep["mismatch"] and all(x["spec"].startswith(f"Attention_v3{{T={LP + T},") and x["request_id"] == "r0" and x["step"] == T
                                   for x in rep["mismatch"]), rep["mismatch"]
    progs = {}
    for rid in attn_world.prompts:
        inst = json.loads(json.dumps(attn_world.inst))
        if rid == "r0":
            r = next(x for x in inst["rows"] if x["spec"].startswith(f"Attention_v3{{T={LP + T},"))
            del r["args"][1][1]                                                    # T = 4 keys declared, 3 named
        progs[rid] = inst
    rep = SR.sampled_replay(progs, bmap, committed, _weights(attn_world), per_stratum=8, seed=1)
    assert rep["result"] is None and rep["grade"] == "partial", (rep["grade"], rep["why"])
    ne = [x for x in rep["not_evaluated"] if str(x.get("spec", "")).startswith(f"Attention_v3{{T={LP + T},")]
    assert ne and all("K history" in str(x["why"]) and "!= T*KVH*D" in str(x["why"]) for x in ne), ne
    assert not rep["mismatch"]


def test_attention_v2_statics_resolve_the_hopper_twins_by_name_and_an_unknown_id_is_unresolved(attn_world):
    """Attention_v2{..., DOT, INV}: the Hopper ids resolve to the wgmma step + Fa3InvSum (FA3); the Ampere ids under the lifted name
    reproduce Attention_v3's words; an unregistered id is `_Unresolved` BY NAME, never evaluated under another adder."""
    pi = attn_world.pi
    r = next(x for x in pi.rows if x["spec"].startswith(f"Attention_v3{{T={LP + T},"))
    d = pi.describe(r)
    inputs = [[attn_world.values[("r0", a)][lo:hi] for _k, a, lo, hi in g] for g in r["args"]]
    v3 = EV.evaluate("Attention_v3", d["statics"], inputs, [None] * 3)["out"]
    assert np.array_equal(v3, attn_world.values[("r0", r["i"])])
    st = dict(d["statics"], DOT='{"fn":"AmpereBF16TcDot16_v2"}', INV='{"fn":"Fa2InvSum_v1"}')
    assert np.array_equal(EV.evaluate("Attention_v2", st, inputs, [None] * 3)["out"], v3)
    from verity.ml.definitions.numerics import derived_rows as D
    st = dict(d["statics"], DOT='{"fn":"HopperBF16WgmmaDot16_v1"}', INV='{"fn":"Fa3InvSum_v1"}')
    got = EV.evaluate("Attention_v2", st, inputs, [None] * 3)["out"]
    q = np.asarray(inputs[0][0], dtype=np.uint16).reshape(NH, DH)
    kc = np.concatenate(inputs[1]).reshape(LP + T, KVH, DH)
    vc = np.concatenate(inputs[2]).reshape(LP + T, KVH, DH)
    ref, _nf = D.attention_rows(q, kc, vc, ABN, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
    assert np.array_equal(got, ref.reshape(-1))
    with pytest.raises(EV._Unresolved, match="does not name a registered primitive"):
        EV.evaluate("Attention_v2", dict(d["statics"], DOT="wgmma", INV='{"fn":"Fa3InvSum_v1"}'), inputs, [None] * 3)
    with pytest.raises(EV._Unresolved, match="no registered evaluator for Attention_v2"):
        EV.evaluate("Attention_v2", dict(d["statics"], DOT='{"fn":"BlackwellDot16_v9"}', INV='{"fn":"Fa3InvSum_v1"}'), inputs, [None] * 3)
    # the row's DECLARED signature drives the evaluator: the Definition's arity is 3 (q, K history, V history) -- a row wiring a
    # 4th arg (a mask / scale the Definition does not declare) or only two is unresolved BY NAME, never evaluated over a guessed subset
    with pytest.raises(EV._Unresolved, match="wires 4 args, the registered Definition takes 3"):
        EV.evaluate("Attention_v3", d["statics"], inputs + [[np.zeros(4, dtype=np.uint16)]], [None] * 4)
    with pytest.raises(EV._Unresolved, match="wires 2 args"):
        EV.evaluate("Attention_v3", d["statics"], inputs[:2], [None] * 2)
    assert "DECLARED signature" in COV.ATTENTION_INTERIOR_MECHANISM and "hand list" in COV.ATTENTION_INTERIOR_MECHANISM
    with pytest.raises(EV._Unresolved, match="K history"):
        EV.evaluate("Attention_v3", d["statics"], [inputs[0], inputs[1][:-1], inputs[2]], [None] * 3)


def test_prewarm_reads_the_composed_interior_producers_weights_in_the_parent_exec_m0749(biased_world):
    """Measured on the Qwen2.5 L40S B1 record: a BiasAdd VU's own args name only the bias slice; the composed interior
    Gemm's qkv weight was first read inside a forked child -> AcceleratorError -> every BiasAdd stratum 'not evaluated'.  The parent
    prewarm must descend into the interior producers: after it, no weight field the sample needs is read for the first time in a child."""
    reads: list[str] = []
    base = _weights(biased_world)
    cache: dict = {}

    def weights(field, lo, hi):
        if (field, lo, hi) not in cache:
            reads.append(field)
            cache[(field, lo, hi)] = base(field, lo, hi)
        return cache[(field, lo, hi)]

    pi = IX.ProgramIndex(biased_world.inst)
    ba = [{"i": r["i"], "request_id": "r0", "spec": r["spec"], "family": "BiasAdd_v1"} for r in pi.rows if r["spec"].startswith("BiasAdd_v1")][:2]
    SR._prewarm_weights({"r0": pi}, ba, weights)
    assert "model.model.layers.0.mlp.gate_up_proj.bias" in reads, reads
    assert "model.model.layers.0.mlp.gate_up_proj.weight" in reads, reads           # the interior Gemm's weight: read in the parent
    # the interior Gemm of a BiasAdd composes its OWN 'n' input (the norm output) from a committed member: nothing else to prewarm
    n0 = len(reads)
    SR._prewarm_weights({"r0": pi}, ba, weights)
    assert len(reads) == n0                                                       # idempotent (cache)


def test_challenge_seed_binds_the_verifier_challenge_to_the_run_root():
    """The v2 seed source: the same challenge on two roots (or two challenges on one root) selects different samples; the
    recorded source names the challenge and the binding."""
    r1, r2 = bytes(range(32)), bytes(range(1, 33))
    s1, src = SM.challenge_seed("c0ffee", r1)
    s2, _ = SM.challenge_seed("c0ffee", r2)
    s3, _ = SM.challenge_seed("deadbeef", r1)
    assert len({s1, s2, s3}) == 3 and SM.challenge_seed("c0ffee", r1)[0] == s1
    assert "verifier challenge" in src and "post-commitment" in src


def test_device_retained_store_is_replayed_serially_and_a_child_device_fault_falls_back(world):
    """A DEVICE-retained committed store cannot be read by forked children (CUDA after fork): (a) the caller's
    retention='device' runs the replay serial in the parent (named in the record); (b) without the hint, a child that hits the device is a
    POOL failure -> serial fallback in the parent -- the honest row PASSES either way instead of failing closed on 'not evaluated'."""
    import os
    bmap, committed = world.bmap_and_reader()
    parent = os.getpid()
    calls = {"child": 0, "parent": 0}

    def device_reader(step, name, lo, hi):
        # the collector's device blocks: readable only in the process that owns the CUDA context (the parent)
        if os.getpid() != parent:
            raise RuntimeError("CUDA error: initialization error (forked child)")
        calls["parent"] += 1
        return committed(step, name, lo, hi)

    rep = SR.sampled_replay(_programs(world), bmap, device_reader, _weights(world), per_stratum=2, seed=3, workers=4, retention="device")
    assert rep["result"] is True and rep["grade"] == "complete", rep.get("why")
    assert rep["evaluator"] == {"retention": "device", "workers_requested": 4, "workers": 1, "mode": "serial (parent process)",
                                "note": "device-retained committed store: forked evaluator disabled, serial replay in the parent's CUDA context",
                                "prewarmed_libraries": {}, "fork": {}}
    assert rep["timing"]["workers"] == 1 and calls["parent"] > 0
    # (b) no retention hint, a forked pool: the children's device fault is a pool failure, the parent replays serially and PASSES
    rep2 = SR.sampled_replay(_programs(world), bmap, device_reader, _weights(world), per_stratum=2, seed=3, workers=2, retention=None)
    assert rep2["result"] is True and rep2["sample"]["evaluated"] == rep["sample"]["evaluated"], rep2.get("why")
    assert rep2["evaluator"]["workers"] == 1 and rep2["evaluator"]["workers_requested"] == 2
    assert rep2["evaluator"]["note"].startswith("parallel evaluator unavailable (_DeviceInChild") and "hit the device" in rep2["evaluator"]["note"]
    assert rep2["evaluator"]["fork"] == {}                                   # the pool failed: no forked record stands on a serial replay
    # a host-retained store still uses the forked pool
    rep3 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=2, retention="host")
    assert rep3["result"] is True and rep3["evaluator"]["mode"] == "forked pool" and rep3["evaluator"]["workers"] == 2 and rep3["evaluator"]["note"] is None
    assert rep3["sample"] == rep["sample"]
    # the forked pool froze the parent's heap before the fork and says so on the record
    fk = rep3["evaluator"]["fork"]
    assert fk["gc_freeze"] is True and fk["workers"] == 2 and fk["picked"] == rep3["sample"]["evaluated"] and fk["gc_frozen_objects"] > 0
    # the record of the VUs (inputs, outputs, verdict per row) is the same under the forked pool and the serial parent: gc.freeze is semantics-free
    ser = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=1, retention="host")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("timing", "evaluator", "wall_s")}  # noqa: E731
    assert strip(ser) == strip(rep3)


# ------------------------------------------------------------------------------------------------------------- fused-MoE planes
E_MOE, TOPK, VPT = 16, 2, 8               # experts, routed slots, values per router thread (E/VPT = 2 threads -> a real xor butterfly)


def _moe_program():
    """One request Program with an OLMoE-shaped fused-MoE layer per token row: Embedding -> RMSNormTriton -> gate Gemm (router logits) ->
    MoeRouterTopK -> TOPK x [MoeExpertGemm (w13) -> SiluMul under `.experts` -> MoeExpertGemmW (w2, routed weight)] -> MoeSum -> fused norm
    fed by the sum (residual = the embedding); per step logits Gemm -> TokenSelect.  Row order, names and args are the real derive_step
    shape (OLMoE B1 256/32 Program): pair-major p = t*TOPK + k, `[p]` / `[t]` name indices, the up row reads router id slot k, the down row
    reads act row p + id slot k + weight slot k, the sum row reads the TOPK down rows in slot order."""
    fields = [("model.model.embed_tokens.weight", V * H, f"Array<{V},Array<{H},Value<16>>>"),
              ("model.model.layers.0.input_layernorm.weight", H, f"Array<{H},Value<16>>"),
              ("model.model.layers.0.mlp.gate.weight", E_MOE * H, f"Array<{E_MOE},Array<{H},Value<16>>>"),
              ("model.model.layers.0.mlp.experts.routed_experts.w13_weight", E_MOE * 2 * I * H, f"Array<{E_MOE},Array<{2 * I},Array<{H},Value<16>>>>"),
              ("model.model.layers.0.mlp.experts.routed_experts.w2_weight", E_MOE * H * I, f"Array<{E_MOE},Array<{H},Array<{I},Value<16>>>>"),
              ("model.model.norm.weight", H, f"Array<{H},Value<16>>")]
    off, woff = 0, {}
    for n, c, _t in fields:
        woff[n] = (off, off + c)
        off += c
    wtype = "{" + ", ".join(f"{n}: {t}" for n, _c, t in fields) + "}"
    rows, i = [], 0

    def add(spec, name, args):
        nonlocal i
        rows.append({"i": i, "spec": spec, "form": "call", "name": name, "args": args})
        i += 1
        return i - 1

    W = lambda n: [["p", 1, woff[n][0], woff[n][1]]]  # noqa: E731
    X = "model.model.layers.0.mlp.experts"
    for s in range(1 + T):
        toks = list(range(LP)) if s == 0 else [0]
        norm_rows, sum_rows, dn_rows = [], [], []
        for t in toks:
            ids = [["p", 0, t, t + 1]] if s == 0 else [["n", sel_prev, 0, 1]]  # noqa: F821
            e = add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding_{s}[{t}]", [ids, W("model.model.embed_tokens.weight")])
            ln = add(f"RMSNormTriton_v1{{N={H},EPS={EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch_{s}[{t}]",
                     [[["n", e, 0, H]], W("model.model.layers.0.input_layernorm.weight")])
            lg = add(f"Gemm_v1{{K={H},N={E_MOE}}}", f"model.model.layers.0.mlp.gate/triton_launch_{s}g[{t}]", [[["n", ln, 0, H]], W("model.model.layers.0.mlp.gate.weight")])
            rt = add(f"MoeRouterTopK_v1{{E={E_MOE},TOPK={TOPK},VPT={VPT}}}", f"{X}/topk_softmax_{s}[{t}]", [[["n", lg, 0, E_MOE]]])
            ups = [add(f"MoeExpertGemm_v1{{E={E_MOE},K={H},N={2 * I}}}", f"{X}/triton_launch_{s}u[{t * TOPK + k}]",
                       [[["n", ln, 0, H]], [["n", rt, TOPK + k, TOPK + k + 1]], W("model.model.layers.0.mlp.experts.routed_experts.w13_weight")]) for k in range(TOPK)]
            acts = [add(f"SiluMul_v1{{I={I}}}", f"{X}/silu_and_mul_{s}[{t * TOPK + k}]", [[["n", ups[k], 0, 2 * I]]]) for k in range(TOPK)]
            dns = [add(f"MoeExpertGemmW_v1{{E={E_MOE},K={I},N={H}}}", f"{X}/triton_launch_{s}d[{t * TOPK + k}]",
                       [[["n", acts[k], 0, I]], [["n", rt, TOPK + k, TOPK + k + 1]], [["n", rt, k, k + 1]], W("model.model.layers.0.mlp.experts.routed_experts.w2_weight")])
                   for k in range(TOPK)]
            sm = add(f"MoeSum_v1{{TOPK={TOPK},H={H}}}", f"{X}/moe_sum_{s}[{t}]", [[["n", d, 0, H] for d in dns]])
            nm = add(f"RMSNormFusedCuda_v2{{N={H},EPS={EPS}}}", f"model.model.norm/fused_add_rms_norm_{s}[{t}]",
                     [[["n", sm, 0, H]], [["n", e, 0, H]], W("model.model.norm.weight")])
            norm_rows.append(nm)
        lgt = add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_{s}c[0]", [[["n", norm_rows[-1], 0, H]], W("model.model.embed_tokens.weight")])
        sel_prev = add(f"TokenSelect_v1{{V={V}}}", f"argmax_{s}[0]", [[["n", lgt, 0, V]]])
    return {"schema": "cb-a/instance-sequence/v1", "digest": "m" * 64, "params": [["prompt", f"Array<{LP},Value<32>>"], ["weights", wtype]], "rows": rows}, woff


@pytest.fixture(scope="module")
def moe_world():
    rng = np.random.default_rng(23)
    inst, woff = _moe_program()
    n_w = max(hi for lo, hi in woff.values())
    weights = _bf16(rng, n_w, 0.5)
    weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80
    weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
    return Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


MOE_FAMS = {"MoeRouterTopK_v1", "MoeExpertGemm_v1", "MoeExpertGemmW_v1", "MoeSum_v1"}


def test_moe_plane_rows_are_addressed_at_their_planes_and_the_honest_program_replays_bit_exactly(moe_world):
    """The fused-MoE block rows are VUs of the replay: `describe` addresses them at their moe_block_stream
    plane (op_path moe/L<k>, member = the plane, row = the pair / token ordinal), their consumers (the act SiluMul reading the up plane, the
    fused norm reading the sum plane) resolve their inputs through the same mapping, and the plane producers are evaluated by the registry's
    relations (router / expert GEMMs / sum).  On the honest world every stratum -- dense and MoE -- replays bit-exactly; nothing is
    'plane-fed' / not evaluable by the MoE name any more."""
    w = moe_world
    pi = w.pi
    rt = next(r for r in pi.rows if r["spec"].startswith("MoeRouterTopK"))
    d = pi.describe(rt)
    assert d["op_path"] == "moe/L0" and d["members"] == [("router_w", TOPK, "f32"), ("router_ids", TOPK, "i32")] and d["row"] == 0 and d["plane_kind"] == "router"
    ups = [r for r in pi.rows if r["spec"].startswith("MoeExpertGemm_v1") and pi.describe(r)["step"] == 0]
    assert [pi.describe(r)["row"] for r in ups] == list(range(LP * TOPK))          # pair-major p = t*TOPK + k
    assert pi.describe(ups[3])["members"] == [("up", 2 * I, "bf16")]
    act = next(r for r in pi.rows if r["spec"].startswith("SiluMul") and "experts" in r["name"])
    assert pi.describe(act)["members"] == [("act", I, "bf16")] and pi.describe(act)["op_path"] == "moe/L0"
    assert pi.member_at(int(act["args"][0][0][1]), 0, 2 * I)[1:] == ("up", 0, 2 * I, "bf16")
    sm = pi.describe(next(r for r in pi.rows if r["spec"].startswith("MoeSum")))                # The module's returned tensor
    assert sm["op_path"] == "model.layers.0.mlp.experts" and sm["members"] == [("out", H, "bf16")]
    assert [(c["op_path"], c["member"], c["of"]) for c in sm["copies"]] == [("moe/L0", "sum", "out")]      # ... and its plane copy
    nm = next(r for r in pi.rows if r["spec"].startswith("RMSNormFusedCuda"))
    assert pi.member_at(int(nm["args"][0][0][1]), 0, H)[1:] == ("out", 0, H, "bf16")
    dn = next(r for r in pi.rows if r["spec"].startswith("MoeExpertGemmW"))
    assert pi.member_at(int(dn["args"][1][0][1]), TOPK, TOPK + 1)[1:] == ("router_ids", 0, 1, "i32")
    assert pi.member_at(int(dn["args"][2][0][1]), 0, 1)[1:] == ("router_w", 0, 1, "f32")
    vus, ne, nef = POP.population(pi, "r0")
    assert not any("moe_block_stream" in k or "plane" in k for k in list(ne) + list(nef)), (ne, nef)
    assert {v["family"] for v in vus} >= MOE_FAMS | {"SiluMul_v1", "RMSNormFusedCuda_v2"}
    assert {v["op_path"] for v in vus if v["family"] in MOE_FAMS - {"MoeSum_v1"} or v["family"] == "SiluMul_v1"} == {"moe/L0"}
    bmap, committed = w.bmap_and_reader()
    rep = SR.sampled_replay(_programs(w), bmap, committed, _weights(w), per_stratum=2, seed=5)
    assert rep["result"] is True and rep["grade"] == "complete", json.dumps({k: rep[k] for k in ("why", "mismatch", "not_evaluated", "population")}, indent=1, default=str)[:4000]
    assert rep["sample"]["mismatch_n"] == 0 and rep["sample"]["evaluated"] == rep["sample"]["picked"] > 0
    assert set(rep["population"]["by_family"]) >= MOE_FAMS and all(v["evaluated"] >= 1 for v in rep["strata"].values())
    assert {k.split("|")[0].split("{")[0] for k in rep["strata"] if k.split("|")[4] == "moe/L0"} == (MOE_FAMS - {"MoeSum_v1"}) | {"SiluMul_v1"}
    assert rep["population"]["not_evaluable"] == {} and all(f in rep["evaluators"] for f in MOE_FAMS)


def _moe_evaluate_all(w, mutate, *, targets=("moe/L0@0",)):
    bmap, committed = w.bmap_and_reader(mutate=mutate)
    return SR.sampled_replay(_programs(w), bmap, committed, _weights(w), per_stratum=1, seed=5, targets=list(targets), workers=0)


def test_the_plan_of_every_moe_vu_reads_what_its_replay_reads_its_sum_plane_copy_too(moe_world):
    """A MoeSum row's output is read back at its `moe/L<k>/sum` plane copy too (`describe`'s `copies`): a slim bundle planned without it
    fails those picks as missing-input (OLMoE B8 slim: 17 of 460)."""
    w = moe_world
    bmap, committed = w.bmap_and_reader()
    vus, _, _ = POP.population(w.pi, "r0")
    assert any(v["family"] == "MoeSum_v1" for v in vus)
    for vu in vus:
        seen: dict[str, set] = {"plan": set(), "replay": set()}

        def reader(key):
            def read(step, name, lo, hi):
                seen[key].add((step, name, lo, hi))
                return committed(step, name, lo, hi)
            return read

        assert EV.plan_vu(w.pi, OP.CommittedStore.of_record(bmap, reader("plan"), None), vu, arrive=0) is None
        EV.replay_vu(w.pi, OP.CommittedStore.of_record(bmap, reader("replay"), None), _weights(w), vu, arrive=0)
        assert seen["replay"] and seen["replay"] <= seen["plan"], (vu["family"], sorted(seen["replay"] - seen["plan"])[:4])


def _plane_word(store, entries, rid, step, plane, elem, flip=0x0080, op_path="moe/L0"):
    """Flip one committed word of request `rid`'s plane slice (request-local element `elem`) in the packed step tensor."""
    e = next(e for e in entries if e["identity"]["request_id"] == rid and e["identity"]["step"] == step and e["identity"]["op_path"] == op_path and e["identity"]["output_member"] == plane)
    isz = 4 if e["encoding"]["dtype"] in ("f32", "i32") else 2
    key = (step, e["name"])
    b = bytearray(store[key])
    o = (int(e["element_offset"]) + elem) * isz
    b[o] ^= (flip & 0xFF) if isz == 2 else 0x01
    if isz == 2:
        b[o + 1] ^= (flip >> 8) & 0xFF
    store[key] = bytes(b)


def test_moe_plane_faults_fail_named_at_the_pair_or_token(moe_world):
    """A flipped committed word of one plane is caught BY NAME at its pair / token: an act word (the SiluMul@experts VU
    of that pair and the down row that reads it), a sum word (the MoeSum VU of that token and the fused norm that reads it), an up word (the
    act VU whose input it is and the up VU that produced it), a router id (the router VU and every expert row of that slot); a flipped word
    of r1's plane never fails an r0 VU (the per-request plane split).  The MoeSum is the experts module's returned tensor `out` with the sum
    plane its checked copy: a flipped plane word fails the MoeSum VU (copy), a flipped `out` word also the norm reading it."""
    w = moe_world
    p = 3                                                                          # pair 3 = token 1, slot 1 (LP = 2, TOPK = 2)
    rep = _moe_evaluate_all(w, lambda st, en: _plane_word(st, en, "r1", 0, "act", p * I + 5))
    assert rep["result"] is False
    bad = {(m["spec"].split("{")[0], m["request_id"], m["step"], m["op_path"], m["row"]) for m in rep["mismatch"]}
    assert ("SiluMul_v1", "r1", 0, "moe/L0", p) in bad and ("MoeExpertGemmW_v1", "r1", 0, "moe/L0", p) in bad, bad
    assert all(m["request_id"] == "r1" for m in rep["mismatch"]) and "moe/L0/act row 3" in " ".join(m["why"] for m in rep["mismatch"])
    exp = "model.layers.0.mlp.experts"
    rep = _moe_evaluate_all(w, lambda st, en: _plane_word(st, en, "r0", 0, "sum", 1 * H + 2), targets=(f"{exp}@0", "model.norm@0"))
    bad = {(m["spec"].split("{")[0], m["request_id"], m["step"], m["op_path"], m["row"]) for m in rep["mismatch"]}
    assert rep["result"] is False and ("MoeSum_v1", "r0", 0, exp, 1) in bad and "moe/L0/sum (copy of out)" in " ".join(m.get("why", "") for m in rep["mismatch"]), bad
    rep = _moe_evaluate_all(w, lambda st, en: _plane_word(st, en, "r0", 0, "out", 1 * H + 2, op_path=exp), targets=(f"{exp}@0", "model.norm@0"))
    bad = {(m["spec"].split("{")[0], m["request_id"], m["step"], m["op_path"], m["row"]) for m in rep["mismatch"]}
    assert rep["result"] is False and ("MoeSum_v1", "r0", 0, exp, 1) in bad and ("RMSNormFusedCuda_v2", "r0", 0, "model.norm", 1) in bad, bad
    rep = _moe_evaluate_all(w, lambda st, en: _plane_word(st, en, "r0", 0, "up", p * 2 * I + 7))
    bad = {(m["spec"].split("{")[0], m["request_id"], m["step"], m["op_path"], m["row"]) for m in rep["mismatch"]}
    assert rep["result"] is False and ("MoeExpertGemm_v1", "r0", 0, "moe/L0", p) in bad and ("SiluMul_v1", "r0", 0, "moe/L0", p) in bad, bad
    rep = _moe_evaluate_all(w, lambda st, en: _plane_word(st, en, "r1", 0, "router_ids", 1 * TOPK + 0))    # token 1 slot 0 -> another expert
    bad = {(m["spec"].split("{")[0], m["request_id"], m["step"], m["op_path"], m["row"]) for m in rep["mismatch"]}
    assert rep["result"] is False and ("MoeRouterTopK_v1", "r1", 0, "moe/L0", 1) in bad and ("MoeExpertGemm_v1", "r1", 0, "moe/L0", 2) in bad, bad
    assert not any(m["request_id"] == "r0" for m in rep["mismatch"])


def _alias_up_to_down(store, entries, rid, step):
    """Replay the pre-FLUSH-03 defect on one request's slice: the committed `up` words of (rid, step) become the `down` words of the same
    layer (the collector read the staged up view after the block's w2 launch had overwritten the shared workspace).  Widths differ in this
    fixture (2I vs H): the common prefix is what the aliased base pointer would have produced."""
    def _slice(plane):
        e = next(e for e in entries if e["identity"]["request_id"] == rid and e["identity"]["step"] == step and e["identity"]["op_path"] == "moe/L0" and e["identity"]["output_member"] == plane)
        o = int(e["element_offset"]) * 2
        n = int(e["identity"]["element_range"][1]) * 2
        return e, o, n
    eu, ou, nu = _slice("up")
    ed, od, nd = _slice("down")
    n = min(nu, nd)
    down = store[(step, ed["name"])][od:od + n]
    b = bytearray(store[(step, eu["name"])])
    b[ou:ou + n] = down
    store[(step, eu["name"])] = bytes(b)


def test_moe_plane_alias_check_is_a_fact_of_the_record_honest_true_aliased_false_dense_none(moe_world, world):
    """The replay record states, per (request, step, MoE layer), that the committed up plane is
    not the committed down plane's bytes: honest world -> result True over every triple with both planes; the pre-FLUSH-03 alias replayed on
    one request's step -> result False naming that (request, step, layer) while the other request's triple still differs; a dense Program
    (no moe/L<k> planes) -> None with 0 compared.  The aliased plane is also caught by the plane relation (mismatch by name) -- the check
    is the direct fact beside it, not a second gate."""
    w = moe_world
    bmap, committed = w.bmap_and_reader()
    ac = POP.moe_plane_alias_check(OP.CommittedStore(bmap, committed))
    assert ac["result"] is True and ac["identical_n"] == 0 and ac["compared"] >= 2 and ac["per_layer"] == {"L0": {"compared": ac["compared"], "identical": 0}}, ac
    assert ac["plane_words"] == {"up": sorted({2 * I * LP * TOPK, 2 * I * TOPK}), "down": sorted({H * LP * TOPK, H * TOPK})}, ac["plane_words"]   # prefill / decode rows
    rep = SR.sampled_replay(_programs(w), bmap, committed, _weights(w), per_stratum=1, seed=5, workers=0)
    assert rep["moe_planes"]["alias_check"]["result"] is True and rep["result"] is True
    bmap, committed = w.bmap_and_reader(mutate=lambda st, en: _alias_up_to_down(st, en, "r1", 0))
    ac = POP.moe_plane_alias_check(OP.CommittedStore(bmap, committed))
    assert ac["result"] is False and ac["identical_n"] == 1 and ac["per_layer"]["L0"]["identical"] == 1, ac
    assert ac["identical"][0]["request_id"] == "r1" and ac["identical"][0]["step"] == 0 and ac["identical"][0]["layer"] == 0 and ac["identical"][0]["bytes_compared"] > 0
    assert "r1 step 0 L0" in ac["why"] and "alias" in ac["why"]
    rep = _moe_evaluate_all(w, lambda st, en: _alias_up_to_down(st, en, "r1", 0))
    assert rep["result"] is False and rep["moe_planes"]["alias_check"]["result"] is False       # the relation fails it too; the fact is stated beside it
    assert any(m["request_id"] == "r1" and m["spec"].startswith("MoeExpertGemm") for m in rep["mismatch"]), rep["mismatch"][:3]
    bmap, committed = world.bmap_and_reader()
    ac = POP.moe_plane_alias_check(OP.CommittedStore(bmap, committed))
    assert ac["result"] is None and ac["compared"] == 0 and ac["per_layer"] == {} and ac["why"] is None


def test_moe_swapped_request_plane_split_fails_named(moe_world):
    """The binding's per-request plane split is what the mapping reads through: r1's `up` entry
    pointed at r0's rows (the older 'whole plane to every request' shape) makes r1's act VUs read another request's up words ->
    F_V != committed act, named at r1's pairs; r0 stays exact."""
    w = moe_world

    def swap(store, entries):
        ups = [e for e in entries if e["identity"]["step"] == 0 and e["identity"]["op_path"] == "moe/L0" and e["identity"]["output_member"] == "up"]
        r0 = next(e for e in ups if e["identity"]["request_id"] == "r0")
        r1 = next(e for e in ups if e["identity"]["request_id"] == "r1")
        r1["element_offset"] = r0["element_offset"]
    rep = _moe_evaluate_all(w, swap)
    assert rep["result"] is False
    assert all(m["request_id"] == "r1" and m["op_path"] == "moe/L0" for m in rep["mismatch"]), rep["mismatch"][:2]
    assert {m["spec"].split("{")[0] for m in rep["mismatch"]} >= {"SiluMul_v1", "MoeExpertGemm_v1"}


def _r0_rows_cover_r1(planes):
    """The binding a B=2 step got before binding.py refused a MoE plane whose rows are not a multiple of the step's tokens: the committed
    plane carries r0's rows only, and r1's entry is that whole plane at offset 0 (request-local range [0, r0's words])."""
    def mutate(store, entries):
        for plane in planes:
            r0, r1 = (next(e for e in entries if e["identity"]["request_id"] == rid and e["identity"]["step"] == 0 and e["identity"]["op_path"] == "moe/L0"
                           and e["identity"]["output_member"] == plane) for rid in ("r0", "r1"))
            n0 = int(r0["identity"]["element_range"][1])
            store[(0, r0["name"])] = store[(0, r0["name"])][:n0 * (4 if r0["encoding"]["dtype"] in ("f32", "i32") else 2)]
            r1["identity"]["element_range"], r1["element_offset"] = [0, n0], 0
    return mutate


@pytest.mark.parametrize("planes, failing", [
    (("router_w", "router_ids", "up", "act", "down", "sum"), {"MoeRouterTopK_v1", "MoeExpertGemm_v1", "MoeSum_v1"}),
    (("sum",), {"MoeSum_v1"}), (("router_w",), {"MoeRouterTopK_v1", "MoeExpertGemmW_v1"}),
    (("router_ids",), {"MoeRouterTopK_v1", "MoeExpertGemm_v1", "MoeExpertGemmW_v1"}), (("up",), {"MoeExpertGemm_v1", "SiluMul_v1"}),
    (("act",), {"SiluMul_v1", "MoeExpertGemmW_v1"}), (("down",), {"MoeExpertGemmW_v1", "MoeSum_v1"})],
    ids=["every_plane", "sum", "router_w", "router_ids", "up", "act", "down"])
def test_moe_planes_carrying_r0_rows_only_bound_whole_to_r1_fail_the_replay(moe_world, planes, failing):
    """Red-team claim 3's MoE case run through the replay (the red team checked coverage only): r1's VUs read r0's plane rows, so every
    per-request stratum sampling of record (one pick per request x phase x op) fails the replay by name at r1 -- a plane alone at the VU that
    produces it and the one that reads it; every plane cut at once, at the rows whose inputs come from outside the planes (the router reads
    the gate logits, the up GEMM the normed input, the MoeSum's `out` copy), while r1's act and down VUs stay consistent with r0's words."""
    w = moe_world
    bmap, committed = w.bmap_and_reader(mutate=_r0_rows_cover_r1(planes))
    rep = SR.sampled_replay(_programs(w), bmap, committed, _weights(w), per_stratum=1, seed=5, workers=0)
    assert rep["result"] is False and rep["grade"] == "mismatch" and not rep["not_evaluated"], (rep["grade"], rep["not_evaluated"][:3])
    assert {(m["request_id"], m["step"]) for m in rep["mismatch"]} == {("r1", 0)}, rep["mismatch"][:3]
    assert {m["spec"].split("{")[0] for m in rep["mismatch"]} == failing


def test_moe_plane_mapping_refuses_a_program_that_is_not_pair_major(moe_world):
    """The row -> plane mapping is checked against the Program, never guessed: an up row whose `[p]` name disagrees with its
    Program-order ordinal, or whose args do not read router slot p % TOPK, is NAMED as a mapping failure (class `mapping`, never carried on
    a PASS) and removed from the evaluable population -- no plane word of another pair is ever read.  A dense Program has no MoE rows and
    its population is untouched."""
    inst = moe_world.inst
    rows = [dict(r) for r in inst["rows"]]
    ups = [k for k, r in enumerate(rows) if r["spec"].startswith("MoeExpertGemm_v1")][:2]
    rows[ups[0]]["name"], rows[ups[1]]["name"] = rows[ups[1]]["name"], rows[ups[0]]["name"]
    pi = IX.ProgramIndex(dict(inst, rows=rows))
    vus, ne, nef = POP.population(pi, "r0")
    keys = [k for k in ne if k.startswith("MoE plane row -> plane slice mapping failed")]
    assert sum(ne[k] for k in keys) == 2 and all("!= its Program name index" in k for k in keys) and nef["MoeExpertGemm_v1 (plane mapping failed)"] == 2
    assert all(RC.why_class(k) == "mapping" for k in keys)
    rows = [dict(r) for r in inst["rows"]]
    rows[ups[0]]["args"], rows[ups[1]]["args"] = rows[ups[1]]["args"], rows[ups[0]]["args"]
    pi = IX.ProgramIndex(dict(inst, rows=rows))
    vus2, ne2, _nef2 = POP.population(pi, "r0")
    keys = [k for k in ne2 if k.startswith("MoE plane row -> plane slice mapping failed")]
    assert sum(ne2[k] for k in keys) == 2 and all("not pair-major" in k for k in keys)
    assert len(vus2) == len(vus) and len(vus) + 2 == len(POP.population(IX.ProgramIndex(inst), "r0")[0])
    assert IX.moe_plane_kind("SiluMul_v1", "model.model.layers.0.mlp.act_fn") is None and IX.moe_plane_kind("SiluMul_v1", "model.model.layers.0.mlp.experts") == "act"


def test_tensor_core_evaluators_name_the_measured_total_family_of_their_k16_step():
    """The evaluator provenance a record carries names the total family (`silicon_total.TOTAL_FAMILY_VERSION` /
    `HOPPER_WGMMA_FAMILY_VERSION`, restated by `registry.tc_total`) of the k16 step it replays through, so two Definitions of one
    name with different non-finite content are told apart on the record itself.  The strings are literals (EVALUATORS must import
    without verity.ml.circuits); this test holds them equal to the constants the step is evaluated with."""
    from verity.ml.circuits.silicon import total as TC   # the step's definition of record; the record literal keeps its historical `registry.tc_total` prefix

    ampere = f"registry.tc_total {TC.AMPERE_BF16_DOT16_TOTAL.name} total family v{TC.TOTAL_FAMILY_VERSION}"
    hopper = f"registry.tc_total {TC.HOPPER_BF16_WGMMA_K16.name} family v{TC.HOPPER_WGMMA_FAMILY_VERSION}"
    assert TC.TOTAL_FAMILY_VERSION == "3" and TC.HOPPER_WGMMA_FAMILY_VERSION == "1"
    for name in ("Gemm_v1", "Attention_v3"):
        assert ampere in RK.EVALUATORS[name] and "hopper" not in RK.EVALUATORS[name].split("k16 step = ")[1], name
    for name in ("Gemm_v2", "Attention_v2"):
        assert hopper in RK.EVALUATORS[name] and ampere.split(" ", 1)[1] in RK.EVALUATORS[name], name


def test_moe_evaluators_equal_the_registered_definitions():
    """The numpy / scalar-primitive evaluators of the MoE families are the registered `verity.ml.definitions.registry.moe`
    Definitions (the bodies moe_kernel_xcheck checked against the real kernels): bit-equal to the reference evaluator on random
    words -- router (renormalize False / True, a tie), sum, up and down expert GEMMs (the gather by the committed expert id)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.program import Program
    from verity.ml.definitions.registry import moe as R
    from verity.ml.definitions.numerics.serve3_reference import materialize

    def ref(fn, inputs):
        P = Program(fn)
        tab = materialize(P, inputs)
        return [int(tab[g]) for g in P.circuit.root_outputs()]

    rng = np.random.default_rng(3)
    E, K_, V_ = 64, 8, 8
    for trial in range(3):
        lg = _bf16(rng, E, 2.0)
        if trial == 2:
            lg[5] = lg[7]
        w, ids = RK.moe_router_topk(lg, E=E, TOPK=K_, VPT=V_)
        assert [int(x) for x in w] + [int(x) for x in ids] == ref(bind(R.MoeRouterTopK, E=E, TOPK=K_, VPT=V_), [[int(v) for v in lg]])
        wn, idn = RK.moe_router_topk(lg, E=E, TOPK=K_, VPT=V_, renormalize=True)
        assert [int(x) for x in wn] + [int(x) for x in idn] == ref(bind(R.MoeRouterTopKNorm, E=E, TOPK=K_, VPT=V_), [[int(v) for v in lg]])
    y = _bf16(rng, K_ * 16, 3.0)
    assert [int(v) for v in EV.evaluate("MoeSum_v1", {"TOPK": K_, "H": 16}, [[y]], [None])["sum"]] == ref(bind(R.MoeSum, TOPK=K_, H=16), [[int(v) for v in y]])
    Es, K, N = 4, 32, 4
    x, slab, e = _bf16(rng, K), _bf16(rng, Es * N * K, 0.5), 2
    g = np.float32(0.37).view(np.uint32).item()
    up = EV.evaluate("MoeExpertGemm_v1", {"E": Es, "K": K, "N": N}, [[x], [np.asarray([e], dtype=np.uint32)], [slab]], [None, None, slab])["up"]
    assert [int(v) for v in up] == ref(bind(R.MoeExpertGemm, E=Es, K=K, N=N), [[int(v) for v in x], [e], [int(v) for v in slab]])
    dn = EV.evaluate("MoeExpertGemmW_v1", {"E": Es, "K": K, "N": N}, [[x], [np.asarray([e], dtype=np.uint32)], [np.asarray([g], dtype=np.uint32)], [slab]], [None, None, None, slab])["down"]
    assert [int(v) for v in dn] == ref(bind(R.MoeExpertGemmW, E=Es, K=K, N=N), [[int(v) for v in x], [e], [g], [int(v) for v in slab]])
    other = EV.evaluate("MoeExpertGemm_v1", {"E": Es, "K": K, "N": N}, [[x], [np.asarray([1], dtype=np.uint32)], [slab]], [None, None, slab])["up"]
    assert not np.array_equal(other, up)                                           # the gather is by the value operand e
    with pytest.raises(ValueError):
        EV.evaluate("MoeExpertGemm_v1", {"E": Es, "K": K, "N": N}, [[x], [np.asarray([Es], dtype=np.uint32)], [slab]], [None, None, slab])


def test_jit_twin_library_build_is_atomic_and_serialised_across_concurrent_first_users(tmp_path, monkeypatch):
    """Sibling first users of a twin .so (forked replay evaluators after a fresh install) used to
    dlopen a half-written library ('file too short').  `_jit.atomic_build` compiles to a private temp path, publishes by os.replace
    and serialises builders on a lock: N concurrent builders yield exactly one compile, every observer sees a complete file, and a
    fresh library is not rebuilt.  The parent prewarms the evaluators' libraries before forking (`_prewarm_evaluators`).  The race
    is staged: the first builder stops half-way through its write until five more have found the library missing and reached its lock."""
    import contextlib
    import multiprocessing as mp
    from verity.ml.definitions.numerics import _jit

    lib = tmp_path / "libtwin.so"
    src = tmp_path / "twin.cpp"
    src.write_text("int f(){return 1;}\n")
    counter = tmp_path / "compiles"
    payload = b"\x7fELF" + bytes(range(256)) * 64
    half = len(payload) // 2
    ctx = mp.get_context("fork")
    half_written, release, at_lock, seen = ctx.Event(), ctx.Event(), ctx.SimpleQueue(), ctx.SimpleQueue()

    def compile_to(out):
        with open(counter, "ab") as fh:
            fh.write(b"1")
        with open(out, "wb") as fh:
            fh.write(payload[:half]); fh.flush()
            half_written.set(); release.wait()
            fh.write(payload[half:])

    real_lock = _jit.build_lock

    @contextlib.contextmanager
    def announced_lock(path):
        at_lock.put(os.getpid())                         # past the unlocked staleness check, which found no library
        with real_lock(path):
            yield

    monkeypatch.setattr(_jit, "build_lock", announced_lock)

    def builder():
        seen.put(len(Path(_jit.atomic_build(lib, [src], compile_to)).read_bytes()))   # a reader after publish sees the whole library

    builders = [ctx.Process(target=builder) for _ in range(6)]
    builders[0].start()
    half_written.wait()
    assert not lib.exists() and [t.read_bytes() for t in tmp_path.glob("libtwin.so.*.tmp*")] == [payload[:half]], \
        "the partial file is only ever the TEMP path"
    for b in builders[1:]:
        b.start()
    assert len({at_lock.get() for _ in builders}) == 6   # every builder is at the lock the first one holds
    release.set()
    assert [seen.get() for _ in builders] == [len(payload)] * 6
    for b in builders:
        b.join()
        assert b.exitcode == 0
    assert lib.read_bytes() == payload
    assert counter.read_bytes() == b"1", "exactly one compile across 6 concurrent builders"
    assert not list(tmp_path.glob("libtwin.so.*.tmp*")), "no temp file left behind"
    # fresh: no rebuild; stale source: rebuilt once, atomically
    os.utime(src, (1_600_000_000, 1_600_000_000)); os.utime(lib, (1_600_000_100, 1_600_000_100))
    _jit.atomic_build(lib, [src], compile_to)
    assert counter.read_bytes() == b"1"
    src.write_text("int f(){return 2;}\n"); os.utime(src, (1_600_000_200, 1_600_000_200))
    _jit.atomic_build(lib, [src], compile_to)
    assert counter.read_bytes() == b"11" and lib.read_bytes() == payload
    # the parent prewarm names the rms library for RMSNorm families and nothing for a Gemm-only sample
    assert SR._prewarm_evaluators([{"spec": "Gemm_v1{M=1}"}]) == {}
    pw = SR._prewarm_evaluators([{"spec": 'RMSNormFusedCuda_v2{N=2048}'}])
    assert "rms_triton_model" in pw


@pytest.mark.skipif(not os.environ.get("VERITY_DENSE_PROGRAM_DIRS") and not os.environ.get("VERITY_MOE_PROGRAM_DIRS"),
                    reason="real Programs: set VERITY_DENSE_PROGRAM_DIRS / VERITY_MOE_PROGRAM_DIRS (':'-separated Program dirs)")
def test_moe_plane_naming_on_real_programs_dense_zero_moe_nonzero_rev_cb_m0733():
    """The named-not-evaluable MoE set is derived from Program rows alone, so it must be EMPTY on
    every dense Program and non-empty on every MoE Program: pins that the naming cannot grow silently onto dense rows.  Opt-in on real
    Programs (laptop: the Llama-3.2-1B B8 components; the OLMoE B1 Program)."""
    from verity_vllm.query.manifest import format as RM
    dense = [d for d in os.environ.get("VERITY_DENSE_PROGRAM_DIRS", "").split(":") if d]
    moe = [d for d in os.environ.get("VERITY_MOE_PROGRAM_DIRS", "").split(":") if d]
    for d in dense:
        pi = IX.ProgramIndex(RM.load_program(d)["instances"])
        vus, ne, nef = POP.population(pi, "r0")
        assert not any("moe_block_stream" in k for k in list(ne) + list(nef)), (d, sorted(nef))
        assert len(vus) > 0
    for d in moe:
        pi = IX.ProgramIndex(RM.load_program(d)["instances"])
        vus, ne, nef = POP.population(pi, "r0")
        named = {k: v for k, v in nef.items() if "moe_block_stream" in k}
        assert named and sum(named.values()) > 0, (d, sorted(nef))
        assert not any(k.startswith("SiluMul_v1") and "moe_block_stream" not in k for k in nef if k.startswith("SiluMul_v1 (fed"))


def test_a_fault_on_some_picks_inside_an_evaluated_stratum_is_partial_named_never_complete_f_r16_13(world):
    """A run-time fault (here: committed bytes not retained for ONE row of one member) that hits some picks of a
    stratum whose other picks evaluated used to leave grade 'complete' (result True) with sample.not_evaluated_n > 0 -- the checker PASSed
    on 'sampled replay' without naming the faulted picks.  'complete' = every picked VU evaluated: any not-evaluated pick -> partial, the
    first fault named; the finalizer's evaluated == picked gate already refused it, now the checker agrees."""
    bmap, committed = world.bmap_and_reader()
    calls = []

    def flaky(step, name, lo, hi):
        if step == 0 and name == "model.layers.0.mlp.act_fn/out" and lo == 0:      # r0's row 0 of the act plane: not retained
            calls.append((step, name, lo, hi))
            return None
        return committed(step, name, lo, hi)

    rep = SR.sampled_replay(_programs(world), bmap, flaky, _weights(world), per_stratum=64, seed=1, workers=1)   # serial: the fault counter is ours
    assert calls, "the fault was never exercised"
    assert rep["sample"]["not_evaluated_n"] > 0 and rep["sample"]["evaluated"] > 0 and rep["sample"]["mismatch_n"] == 0
    assert rep["strata_unevaluated"] == []                                       # no stratum is FULLY unevaluated: the old rule graded this complete
    assert rep["result"] is None and rep["grade"] == "partial"
    assert rep["why"].startswith(f"{rep['sample']['not_evaluated_n']} pick(s) not evaluated inside evaluated strata (missing-input x {rep['sample']['not_evaluated_n']}) -- run-time fault, not a gap on the record: ")
    assert "not retained" in rep["why"]
    assert RC.why_class("committed bytes not retained for (r0, step 0, x/out) [0, 16)") == "missing-input" and RC.why_class("ambiguous binding: 2 entries for (...)") == "missing-input"
    assert RC.why_class("committed bytes not retained ...") in RC.WHY_CLASSES_FAULT and "missing-input" not in RC.WHY_CLASSES_BY_NAME
    assert all(x["why_class"] == "missing-input" for x in rep["not_evaluated"])
    # the row verdict: fail closed on the partial replay, the fault in the text (no named-gap lift: no fully unevaluated stratum)
    from verity_vllm.check.verdict import commit_verdict
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "value_correspondence": {"method": "sampled-exact-replay(partial) + boundary-linkage", "result": None},
            "boundary_linkage": {"result": True, "pairs": 6, "linked": 6, "unlinked_n": 0},
            "weights_pin": {"result": True, "compared": 98, "mismatched": [], "missing": []},
            "oracle_compare": {"result": True, "grade": "complete", "compared": 3, "equal": 3, "mismatch_n": 0, "complete": True,
                               "compared_by_family": {"instance_outputs": 2, "input_token_ids": 1}, "not_compared_by_family": {},
                               "attribution": {"complete": True, "pairs": 3, "unverified_n": 0, "ambiguous_steps": []}}}
    ok, why = commit_verdict({**base, "binding": _bnd(2), "sampled_replay": rep}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: sampled replay partial") and "run-time fault" in why
    # the honest replay of the same world is still complete
    rep_ok = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=64, seed=1)
    assert rep_ok["result"] is True and rep_ok["grade"] == "complete" and rep_ok["sample"]["not_evaluated_n"] == 0


def test_padded_moe_block_twin_equals_the_moe01_composition_and_names_every_member():
    """`padded_moe_block` (the evaluator of MoeFfnFixedL_v1 / MoeFfnFixed_v1) is
    value-equal to the MOE-01 composition the record's five kernels are replayed by -- MoeRouterTopK on the logits, per SELECTED slot
    MoeExpertGemm (up) -> SiluMul (act) -> MoeExpertGemmW (down, routed weight in the accumulator) -> MoeSum in id order -- and evaluates the
    UNSELECTED slots too (g = 0.0f: the construction computes them; their down words are f2fp(acc * 0) = +/-0).  Every member Q_module_v1
    (`out`) or Q_module_pad_v1 (slot units + router) may name is returned by ONE evaluation; the memo makes a second pick on the row free.
    (The twin's bit-equality to the `MoeFfnFixedL_v1` Definition itself -- 10 cases + the slot bodies vs MoeExpertBody_v1 -- is the
    evidence record out/gen/sweep/evidence/reqval/padded_moe/difftest_moe_pad.json.)"""
    rng = np.random.default_rng(3)

    def bf16(shape, scale=1.0):
        f = (rng.standard_normal(shape) * scale).astype(np.float32)
        return (f.view(np.uint32) >> 16).astype(np.uint16)

    E, TOPK, VPT, H, I = 8, 2, 4, 32, 16
    st = {"E": E, "TOPK": TOPK, "VPT": VPT, "H": H, "I": I}
    x, logits, w13, w2 = bf16((H,)), bf16((E,), 2.0), bf16((E, 2 * I, H), 0.3), bf16((E, H, I), 0.3)
    RK._PADDED_BLOCK_MEMO.clear()
    got = RK.padded_moe_block(st, [x, logits, w13, w2], router_gemm=False)
    assert set(got) == {"out", "router_ids", "router_w"} | {f"slot{e}/{m}" for e in range(E) for m in ("up", "act", "down", "g")}
    # the MOE-01 composition through the SAME evaluate() entry the record's pair rows use
    r = EV.evaluate("MoeRouterTopK_v1", {"E": E, "TOPK": TOPK, "VPT": VPT}, [[logits]], [None])
    assert got["router_ids"].tolist() == r["router_ids"].tolist() and got["router_w"].tolist() == r["router_w"].tolist()
    ids, alpha = [int(v) for v in r["router_ids"]], [int(v) for v in r["router_w"]]
    downs = []
    for j in range(TOPK):
        e = ids[j]
        up = EV.evaluate("MoeExpertGemm_v1", {"E": E, "K": H, "N": 2 * I}, [[x], [np.asarray([e], dtype=np.uint32)], [w13.reshape(-1)]], [None, None, None])["up"]
        act = EV.evaluate("SiluMul_v1", {"I": I}, [[up]], [None])["out"]
        down = EV.evaluate("MoeExpertGemmW_v1", {"E": E, "K": I, "N": H},
                           [[act], [np.asarray([e], dtype=np.uint32)], [np.asarray([alpha[j]], dtype=np.uint32)], [w2.reshape(-1)]], [None] * 4)["down"]
        assert got[f"slot{e}/up"].tolist() == up.tolist() and got[f"slot{e}/act"].tolist() == act.tolist() and got[f"slot{e}/down"].tolist() == down.tolist()
        assert int(got[f"slot{e}/g"][0]) == alpha[j]
        downs.append(down)
    y = EV.evaluate("MoeSum_v1", {"TOPK": TOPK, "H": H}, [[np.concatenate(downs)]], [None])["sum"]
    assert got["out"].tolist() == y.tolist()
    # the unselected slots: evaluated with g = 0.0f -> down = f2fp(acc * 0) is a signed zero word, the up / act words are the body's own
    for e in range(E):
        if e in ids:
            continue
        assert int(got[f"slot{e}/g"][0]) == 0 and set(int(v) & 0x7FFF for v in got[f"slot{e}/down"]) == {0}
        assert got[f"slot{e}/up"].any()
    # the memo: the same (statics, x, logits) is one evaluation; a different logits word is another
    assert RK.padded_moe_block(st, [x, logits, w13, w2], router_gemm=False) is got
    logits2 = logits.copy(); logits2[0] ^= 0x0080
    assert RK.padded_moe_block(st, [x, logits2, w13, w2], router_gemm=False) is not got
    # MoeFfnFixed_v1: the router Gemm inside = Gemm_v1 on (x, wg) then the same block
    wg = bf16((E, H), 0.5)
    lg = EV.evaluate("Gemm_v1", {"K": H, "N": E}, [[x], [wg.reshape(-1)]], [None, None])["0"]
    assert RK.padded_moe_block(st, [x, wg, w13, w2], router_gemm=True)["out"].tolist() == RK.padded_moe_block(st, [x, lg, w13, w2], router_gemm=False)["out"].tolist()
    assert "MoeFfnFixedL_v1" in RK.EVALUATORS and "MoeFfnFixed_v1" in RK.EVALUATORS


def test_lifted_slot_compare_bot_under_active_and_value_under_disabled_both_fail_by_name_and_exact_bot_passes():
    """The padded slot units are LIFTED words (l17: u32, bf16 payload, bot 0x1_0000; l33: u64, bot 0x1_0000_0000).
    Under a_e = 1 every word must be Active(twin value) = the payload exactly; under a_e = 0 every word must be enc(bot) exactly.  A bot word
    under Active, a value word under disabled, and a bot with a nonzero payload / INV each FAIL with the class named; the exact
    bot plane and the exact Active plane PASS.  The enable is derived from the row's COMMITTED router_ids, never the twin's."""
    payload = np.asarray([0x3F80, 0xBF00, 0x0000, 0x4049], dtype=np.uint16)
    bot17 = 1 << 16
    eq, nd, first, v = CMP.lifted_compare(payload.astype(np.uint32), payload, True, "l17")
    assert eq and nd == 0 and first is None and v.startswith("active")
    eq, nd, first, v = CMP.lifted_compare(np.full(4, bot17, dtype=np.uint32), payload, False, "l17")
    assert eq and v.startswith("disabled")
    have = payload.astype(np.uint32); have[2] = bot17                                # a bot word where the slot is Active
    eq, nd, first, v = CMP.lifted_compare(have, payload, True, "l17")
    assert not eq and nd == 1 and first == 2 and "bot word under an Active enable" in v
    have = np.full(4, bot17, dtype=np.uint32); have[1] = 0x3F80                      # a value word where the slot is disabled
    eq, nd, first, v = CMP.lifted_compare(have, payload, False, "l17")
    assert not eq and nd == 1 and first == 1 and "value word under a disabled enable" in v
    have = np.full(4, bot17, dtype=np.uint32); have[3] = bot17 | 0x0001              # bot with a nonzero payload (or INV): never accepted as bot
    eq, nd, first, v = CMP.lifted_compare(have, payload, False, "l17")
    assert not eq and "nonzero payload" in v
    # an INV word (bot bit + nonzero payload) under an ACTIVE enable is named INV, never 'payload differs'
    have = payload.astype(np.uint32).copy(); have[3] |= bot17                          # payload[2] is 0x0000: bot bit there IS bot, so use [3]
    eq, nd, first, v = CMP.lifted_compare(have, payload, True, "l17")
    assert not eq and nd == 1 and first == 3 and "INV word" in v and "Active" in v
    eq, nd, first, v = CMP.lifted_compare(payload.astype(np.uint32)[:3], payload, True, "l17")
    assert not eq and nd == -1 and "width" in v
    # l33: the g word and the router members (u64)
    g = np.asarray([0x3F2C4635], dtype=np.uint32)
    assert CMP.lifted_compare(np.asarray([0x3F2C4635], dtype=np.uint64), g, True, "l33")[0]
    assert CMP.lifted_compare(np.asarray([1 << 32], dtype=np.uint64), g, False, "l33")[0]
    assert not CMP.lifted_compare(np.asarray([1 << 32], dtype=np.uint64), g, True, "l33")[0]
    # the enable from the COMMITTED router_ids of the row (a store stub): e in ids -> Active; e not in ids -> disabled; a bot id -> not evaluable

    class _Store:
        def __init__(self, ids):
            self.ids = ids

        def words(self, rid, step, op_path, member, elo, ehi):
            assert member == "router_ids"
            return (np.asarray(self.ids[elo:ehi], dtype=np.uint64), "uint64") if self.ids is not None else (None, "no committed identity")

    st = _Store([3, 1, 7, 0])
    assert CMP.padded_slot_enable("slot3/up", {}, st, "r0", 5, "model.layers.1.mlp.experts", 0, 2) == (True, "")
    assert CMP.padded_slot_enable("slot7/down", {}, st, "r0", 5, "model.layers.1.mlp.experts", 0, 2) == (False, "")   # row 0 = ids[0:2] = {3, 1}
    assert CMP.padded_slot_enable("slot7/down", {}, st, "r0", 5, "model.layers.1.mlp.experts", 1, 2) == (True, "")    # row 1 = ids[2:4] = {7, 0}
    assert CMP.padded_slot_enable("router_w", {}, st, "r0", 5, "model.layers.1.mlp.experts", 0, 2) == (True, "")
    a, why = CMP.padded_slot_enable("slot3/up", {}, _Store(None), "r0", 5, "model.layers.1.mlp.experts", 0, 2)
    assert a is None and "committed router_ids are not available" in why
    a, why = CMP.padded_slot_enable("slot3/up", {}, _Store([1 << 32, 1]), "r0", 5, "model.layers.1.mlp.experts", 0, 2)
    assert a is None and "is bot" in why
    assert OP._ITEMSIZE["l17"] == 4 and OP._ITEMSIZE["l33"] == 8


def test_population_stamps_on_every_reason_the_code_its_text_decodes_to(moe_world, biased_world, monkeypatch):
    """Every not-evaluable reason population() counts carries its structured code (`population.not_evaluable_codes`, what the verdict
    reads), and on every site that code is the one a record written before the codes decodes its text to (`replay_codes`): a new record
    and a record of record grade alike."""
    from collections import Counter

    from verity_vllm.check.replay import replay_codes as RC

    monkeypatch.delitem(RK.EVALUATORS, "RMSNormFusedCuda_v2")                  # direct no-evaluator gaps
    monkeypatch.delitem(RK.EVALUATORS, "MoeRouterTopK_v1")                     # a plane producer without an evaluator (plane-fed)
    monkeypatch.delitem(RK.EVALUATORS, "Gemm_v1")                              # BiasAdd over an interior Gemm (interior-composed) + direct Gemm gaps
    monkeypatch.setitem(RC.NOT_YET, "TokenSelect_v1", "not replayed by this version (test label)")     # the stated domain
    rows = [dict(r) for r in moe_world.inst["rows"]]
    ups = [k for k, r in enumerate(rows) if r["spec"].startswith("MoeExpertGemm_v1")][:2]
    rows[ups[0]]["name"], rows[ups[1]]["name"] = rows[ups[1]]["name"], rows[ups[0]]["name"]            # a plane mapping failure
    ne, codes = Counter(), {}
    for inst in (dict(moe_world.inst, rows=rows), biased_world.inst):
        ne.update(POP.population(IX.ProgramIndex(inst), "r0", codes=codes)[1])
    assert set(codes) == set(ne)
    assert {"no-evaluator", "plane-fed", "interior-composed-v2", "mapping"} <= {c["class"] for c in codes.values()}
    assert [k for k, c in codes.items() if c["domain"]] == ["not replayed by this version (test label)"]
    decoded = RC.not_evaluable_codes({"not_evaluable": dict(ne)})
    for k, got in codes.items():
        want = decoded[k]
        assert (got["class"], got["domain"], got["tags"]) == (want["class"], want["domain"], want["tags"]), k
        if got["class"] == "no-evaluator":
            assert got["family"] == want["family"], k
