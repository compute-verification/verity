"""Transfer test: a SECOND
model family, not used to develop the importer's rules, through the importer FROZEN at tag `derive-freeze-1`,
unchanged. Every case is reported as-is: derived (then evaluated against torch CPU on random weights and checked
structurally) or `Unsupported(reason, node, module_path)` with the rules that fired and the gap class
(structural rule vs new numerical kind).

Second family (disjoint from the importer's Qwen2/Llama-style development models and B0/SmolLM2 dims):

  * GPT-2 style decoder (`Conv1D` `x @ W + b` with `W[in, out]` — not `nn.Linear`; `nn.LayerNorm` with bias;
    GELU(tanh); learned positional embeddings `wpe`; merged `c_attn`; explicit softmax attention in f32; tied
    `wte`/`lm_head`), random f32 weights at tiny dims, plus a bf16 variant and a bounded greedy request with KV
    carried as values (`torch.cat`, no host cache).
  * A non-decoder residual MLP with a shared block applied three times and one parameter used in two roles.

Results are written as JSON lines to `$XFER_OUT` (default `/tmp/xfer/transfer-<importer>.jsonl`) for the transfer
report. Needs torch; runs on the pod.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits import Program, program_digest  # noqa: E402
from verity.ml.circuits.evaluation import MissingValue, evaluate_region  # noqa: E402
from verity_vllm.program.frontend.provenance import registry_version  # noqa: E402

from tests.program.test_derive_negative import IMPORTER_NAME, Outcome, _DERIVE, _input_gate_ranges, run_importer  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::verity_vllm.program.frontend.torch_frontend.DeadGateWarning")

nn, F = torch.nn, torch.nn.functional

# ---------------------------------------------------------------------------------------------------------
# the second family
# ---------------------------------------------------------------------------------------------------------

CFG = {"V": 32, "H": 16, "NH": 2, "D": 8, "I": 32, "NL": 2, "MAXPOS": 8, "EPS": 1e-5}


class Conv1D(nn.Module):
    """GPT-2's projection: `x @ W + b` with `W[in, out]` (HF `Conv1D`), the transposed layout of `nn.Linear`."""

    def __init__(self, nin: int, nout: int, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nin, nout, dtype=dtype) * 0.05)
        self.bias = nn.Parameter(torch.zeros(nout, dtype=dtype))

    def forward(self, x):
        return x @ self.weight + self.bias


class Conv1DAddmm(Conv1D):
    """HF's actual `Conv1D.forward`: `torch.addmm(bias, x.view(-1, K), W).view(size_out)` (added after the freeze run,
    00:55, to check DERIVE's support-matrix row 8 against an independently written module)."""

    def forward(self, x):
        size_out = x.size()[:-1] + (self.weight.shape[1],)
        return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(size_out)


class GPT2MLP(nn.Module):
    def __init__(self, h=CFG["H"], i=CFG["I"], dtype=torch.float32, conv=Conv1D):
        super().__init__()
        self.c_fc = conv(h, i, dtype)
        self.c_proj = conv(i, h, dtype)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class GPT2Attention(nn.Module):
    """Causal multi-head attention over T positions: merged qkv `Conv1D`, softmax in f32, `Conv1D` output."""

    def __init__(self, h=CFG["H"], nh=CFG["NH"], dtype=torch.float32, conv=Conv1D):
        super().__init__()
        self.nh, self.d = nh, h // nh
        self.c_attn = conv(h, 3 * h, dtype)
        self.c_proj = conv(h, h, dtype)

    def forward(self, x):  # x: [T, H]
        T = x.shape[0]
        qkv = self.c_attn(x)
        q, k, v = qkv.split(x.shape[-1], dim=-1)
        q = q.view(T, self.nh, self.d).transpose(0, 1)  # [nh, T, d]
        k = k.view(T, self.nh, self.d).transpose(0, 1)
        v = v.view(T, self.nh, self.d).transpose(0, 1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)  # [nh, T, T]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s.float(), dim=-1).to(x.dtype)
        y = (p @ v).transpose(0, 1).reshape(T, self.nh * self.d)
        return self.c_proj(y)


class GPT2Block(nn.Module):
    def __init__(self, dtype=torch.float32, conv=Conv1D):
        super().__init__()
        self.ln_1 = nn.LayerNorm(CFG["H"], eps=CFG["EPS"], dtype=dtype)
        self.attn = GPT2Attention(dtype=dtype, conv=conv)
        self.ln_2 = nn.LayerNorm(CFG["H"], eps=CFG["EPS"], dtype=dtype)
        self.mlp = GPT2MLP(dtype=dtype, conv=conv)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2TinyLM(nn.Module):
    """wte + wpe -> NL blocks -> ln_f -> tied lm_head; returns logits [T, V]."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.wte = nn.Embedding(CFG["V"], CFG["H"], dtype=dtype)
        self.wpe = nn.Embedding(CFG["MAXPOS"], CFG["H"], dtype=dtype)
        self.h = nn.ModuleList([GPT2Block(dtype) for _ in range(CFG["NL"])])
        self.ln_f = nn.LayerNorm(CFG["H"], eps=CFG["EPS"], dtype=dtype)
        self.lm_head = nn.Linear(CFG["H"], CFG["V"], bias=False, dtype=dtype)
        self.lm_head.weight = self.wte.weight  # tied

    def forward(self, tokens):  # [T] int64
        T = tokens.shape[0]
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        for blk in self.h:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


class GPT2Greedy(nn.Module):
    """Bounded request: prefill LP tokens, then STEPS greedy decode steps, KV carried as values (no host cache,
    no early exit). Returns the STEPS+1 sampled tokens. Attention at each step recomputes over the growing
    prefix (the KV dependency is explicit in the dataflow: every step's K/V rows are this request's values)."""

    def __init__(self, LP=2, STEPS=2, dtype=torch.float32):
        super().__init__()
        self.LP, self.STEPS = LP, STEPS
        self.lm = GPT2TinyLM(dtype)

    def forward(self, prompt):  # [LP]
        toks = prompt
        out = []
        for _ in range(self.STEPS + 1):
            logits = self.lm(toks)
            nxt = torch.argmax(logits[-1], dim=-1)
            out.append(nxt)
            toks = torch.cat([toks, nxt.view(1)])
        return torch.stack(out)


class ResidualMLPShared(nn.Module):
    """Non-decoder: one `Linear` block applied three times with residuals (shared weights by reuse) and a `scale`
    parameter used in two roles (pre-scale of the input and post-scale of the output)."""

    def __init__(self, h=CFG["H"], dtype=torch.float32):
        super().__init__()
        self.block = nn.Linear(h, h, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(h, dtype=dtype) * 0.5)

    def forward(self, x):
        x = x * self.scale
        for _ in range(3):
            x = x + torch.relu(self.block(x))
        return x * self.scale


def _tok(*shape, hi=CFG["V"]):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, hi, shape, generator=g, dtype=torch.int64)


@dataclass
class Case:
    id: str
    make: Any
    inputs: tuple
    dtype: str
    expect: str  # the expected outcome, for the report
    notes: str = ""
    importer_kw: dict | None = None  # e.g. {"mode": "profile", "profile": "b1-eager"}; default reference mode


torch.manual_seed(1234)  # the case inputs below are drawn once at import; seeded so every run evaluates the same words

CASES = [
    Case("mlp_shared_f32", lambda: ResidualMLPShared(), (torch.randn(CFG["H"]),), "f32",
         "derives in reference mode (Linear/relu/add/mul); ONE Input for `block.*` and `scale` each; three block calls"),
    Case("gpt2_conv1d_f32", lambda: Conv1D(CFG["H"], CFG["I"]), (torch.randn(CFG["H"]),), "f32",
         "x @ W + b with W[in,out]: a Gemm rule keyed on aten.linear/[out,in] must either handle aten.mm/[in,out] or refuse"),
    Case("gpt2_mlp_f32", lambda: GPT2MLP(), (torch.randn(CFG["H"]),), "f32",
         "GELU(tanh) has no registered kind: numerics_unregistered unless NUMERIC's reference kinds landed and DERIVE bound them"),
    Case("gpt2_layernorm_f32", lambda: nn.LayerNorm(CFG["H"], eps=CFG["EPS"]), (torch.randn(CFG["H"]),), "f32",
         "LayerNorm (mean/var/rsqrt/bias) has no registered kind: numerics_unregistered unless a reference kind exists"),
    Case("gpt2_attention_f32", lambda: GPT2Attention(), (torch.randn(4, CFG["H"]),), "f32",
         "softmax attention over T=4: numerics_unregistered (softmax) unless a reference kind; masked_fill/tril are structural"),
    Case("gpt2_block_f32", lambda: GPT2Block(), (torch.randn(4, CFG["H"]),), "f32",
         "the whole block: first gap in ln_1 or attn"),
    Case("gpt2_lm_f32", lambda: GPT2TinyLM(), (_tok(4),), "f32",
         "tied wte/lm_head must stay ONE Input; wpe gather by arange positions is static"),
    Case("gpt2_greedy_f32", lambda: GPT2Greedy(LP=2, STEPS=2), (_tok(2),), "f32",
         "bounded request: 3 argmax selections, KV as values; must not shrink on observed tokens"),
    Case("gpt2_lm_bf16", lambda: GPT2TinyLM(dtype=torch.bfloat16), (_tok(4),), "bf16",
         "reference mode on a bf16 module: casts/kinds are f32-only in the reference vocabulary -> refusal expected"),
    # added AFTER the freeze run (00:40): the bf16 probe as it was meant, in profile mode with the B1 eager vocabulary
    Case("gpt2_lm_bf16_profile", lambda: GPT2TinyLM(dtype=torch.bfloat16), (_tok(4),), "bf16",
         "profile-mode probe: bf16 x@W layout vs Gemm_v1{K,N}[N][K]; LayerNorm/GELU/softmax unregistered in the B0/B1 profile",
         importer_kw={"mode": "profile", "profile": "b1-eager"}),
    Case("gpt2_lm_bf16_profile_i32", lambda: GPT2TinyLM(dtype=torch.bfloat16), (_tok(4).to(torch.int32),), "bf16",
         "as above with int32 token ids (the profile's Embedding_v1 takes I32): how far does the B1 vocabulary carry GPT-2?",
         importer_kw={"mode": "profile", "profile": "b1-eager"}),
    # HF's addmm form of Conv1D, T=3 rows -- support-matrix row 8 on this file's own modules
    Case("gpt2_conv1d_addmm_f32", lambda: Conv1DAddmm(CFG["H"], CFG["I"]), (torch.randn(3, CFG["H"]),), "f32",
         "HF Conv1D via addmm: matrix row 8 says LinearRef with the weight as a transposed view"),
    Case("gpt2_mlp_addmm_f32", lambda: GPT2MLP(conv=Conv1DAddmm), (torch.randn(3, CFG["H"]),), "f32",
         "matrix row 8 (LayerNorm-less MLP: Conv1D -> gelu(tanh) -> Conv1D) on an independently written module"),
    Case("gpt2_block_addmm_f32", lambda: GPT2Block(conv=Conv1DAddmm), (torch.randn(3, CFG["H"]),), "f32",
         "full block with addmm Conv1D: attention's q@k^T / softmax / causal-mask constant are the expected next gaps"),
]


# ---------------------------------------------------------------------------------------------------------
# evaluation of a derived program against torch CPU
# ---------------------------------------------------------------------------------------------------------


def _bits(t: torch.Tensor, w: int) -> np.ndarray:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().astype(np.int64) & 0xFFFF
    if t.dtype == torch.float16:
        return t.view(torch.int16).numpy().astype(np.int64) & 0xFFFF
    if t.dtype == torch.float32:
        return t.view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF
    if t.dtype == torch.float64:
        return t.view(torch.int64).numpy().astype(object)
    if t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool, torch.uint8):
        v = t.to(torch.int64).numpy().astype(object)
        return np.array([int(x) & ((1 << w) - 1) for x in v.ravel()], dtype=object).reshape(v.shape)
    raise TypeError(t.dtype)


def _decode(word: int, w: int, like: torch.dtype) -> float | int:
    if like in (torch.float32,) or (w == 32 and like.is_floating_point):
        return float(np.array([word], dtype=np.uint32).view(np.float32)[0])
    if like in (torch.bfloat16,) or (w == 16 and like.is_floating_point):
        return float(np.array([word << 16], dtype=np.uint32).view(np.float32)[0])
    if like == torch.float64:
        return float(np.array([word], dtype=np.uint64).view(np.float64)[0])
    if word >= 1 << (w - 1):
        word -= 1 << w
    return int(word)


def _input_map(out: Outcome, module: nn.Module, inputs: tuple) -> list[tuple[str, torch.Tensor]] | None:
    """(name, tensor) per prescribed root parameter, in the program's parameter order. Prefers a map the importer
    reports; falls back to the shipped route's convention (positional args, then the weights record)."""
    rep = out.report
    for attr in ("input_map", "inputs", "input_bindings", "bindings"):
        m = getattr(rep, attr, None) if rep is not None else None
        if isinstance(m, dict) and m:
            return list(m.items())
    P = out.program
    params, _ = P.fn.signature()
    if len(params) == len(inputs) + 1:
        all_named = {**dict(module.named_parameters(remove_duplicate=False)), **dict(module.named_buffers(remove_duplicate=False))}
        weights = {**dict(module.named_parameters()), **dict(module.named_buffers())}
        wl = provenance_of(out).get("weights")
        if isinstance(wl, list) and wl and all(isinstance(w, dict) and "name" in w for w in wl):
            weights = {w["name"]: all_named[w["name"]] for w in wl}  # the importer's canonical fqn per distinct tensor
        return [(n, t) for (n, _), t in zip(params[:-1], inputs)] + [(params[-1][0], weights)]  # type: ignore[list-item]
    if len(params) == len(inputs):
        return [(n, t) for (n, _), t in zip(params, inputs)]
    return None


def _flatten_like(t: Any, T) -> list[tuple[torch.Tensor, int]]:
    """Leaf tensors of a (possibly record) argument, paired with the width of the corresponding type leaves."""
    from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Value
    if isinstance(t, torch.Tensor):
        tt = T
        while isinstance(tt, Array):
            tt = tt.elem
        assert isinstance(tt, Value), (T, t.shape)
        return [(t, tt.w)]
    if isinstance(t, dict):
        assert isinstance(T, Record), T
        return [leaf for (k, ft) in T.fields for leaf in _flatten_like(t[k], ft)]
    if isinstance(t, (tuple, list)):
        fields = T.fields if isinstance(T, TupleT) else [T.elem] * len(t)
        return [leaf for x, ft in zip(t, fields) for leaf in _flatten_like(x, ft)]
    raise TypeError(type(t))


def evaluate_against_torch(out: Outcome, module: nn.Module, inputs: tuple) -> dict:
    """Bind the module's actual weights + inputs to the Input gates, evaluate every gate, compare the root outputs
    with the module's CPU output bit for bit and numerically."""
    P = out.program
    C = P.circuit
    imap = _input_map(out, module, inputs)
    if imap is None:
        return {"evaluated": False, "why": "no input map (parameter count does not match inputs + weights)"}
    params, ret = P.fn.signature()
    transcript: dict[int, int] = {}
    for pi, ((pname, ptype), (_, tensor)) in enumerate(zip(params, imap)):
        coll = P.input_coll(pi)
        leaves = _flatten_like(tensor, ptype)
        words = np.concatenate([_bits(t, w).ravel() for t, w in leaves]) if leaves else np.array([], dtype=object)
        refs = coll.refs
        if refs.count != len(words):
            return {"evaluated": False, "why": f"parameter {pname}: {refs.count} gates vs {len(words)} tensor leaves"}
        for j in range(refs.count):
            _, _, g = refs[j]
            transcript[g] = int(words[j])
    try:
        evaluate_region(C, C[0:C.size], transcript)
    except MissingValue as e:
        return {"evaluated": False, "why": f"MissingValue: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"evaluated": False, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    with torch.no_grad():
        want = module(*[t.clone() for t in inputs])
    want_flat = torch.cat([w.reshape(-1) for w in torch.utils._pytree.tree_leaves(want)])
    outs = C.root_outputs()
    if len(outs) != want_flat.numel():
        return {"evaluated": True, "match": False, "why": f"{len(outs)} root outputs vs {want_flat.numel()} torch outputs"}
    got = []
    for g, like in zip(outs, [want_flat.dtype] * len(outs)):
        w = C.gate(g).prim.ret.w
        got.append(_decode(transcript[g], w, like))
    got_t = torch.tensor(got, dtype=torch.float64)
    want_d = want_flat.to(torch.float64)
    exact = int((got_t == want_d).sum())
    finite = torch.isfinite(want_d)
    max_abs = float((got_t - want_d)[finite].abs().max()) if finite.any() else float("nan")
    denom = want_d[finite].abs().clamp_min(1e-6)
    max_rel = float(((got_t - want_d)[finite].abs() / denom).max()) if finite.any() else float("nan")
    tol = 1e-4 if want_flat.dtype == torch.float32 else (1e-2 if want_flat.dtype == torch.bfloat16 else 0.0)
    close = bool(torch.allclose(got_t, want_d, rtol=tol, atol=tol)) if want_flat.is_floating_point() else bool((got_t == want_d).all())
    return {"evaluated": True, "match": close, "bit_exact": exact == len(outs), "exact_words": exact, "words": len(outs),
            "max_abs": max_abs, "max_rel": max_rel, "gates": C.size}


# ---------------------------------------------------------------------------------------------------------
# structural expectations
# ---------------------------------------------------------------------------------------------------------


def _distinct_param_leaves(m: nn.Module) -> int:
    seen = {id(p): p for p in list(m.parameters()) + list(m.buffers())}
    return sum(p.numel() for p in seen.values())


def structural_checks(case: Case, module: nn.Module, inputs: tuple, out: Outcome) -> dict:
    P = out.program
    n_in = sum(t.numel() for t in inputs)
    res = {"input_gates": P.input_gates, "distinct_param_leaves": _distinct_param_leaves(module), "input_leaves": n_in}
    res["one_input_per_distinct_leaf"] = P.input_gates == _distinct_param_leaves(module) + n_in
    naive = sum(p.numel() for _, p in module.named_parameters(remove_duplicate=False)) + sum(b.numel() for b in module.buffers()) + n_in
    res["naive_duplicated_count"] = naive
    res["sharing_lost"] = P.input_gates == naive and naive != res["distinct_param_leaves"] + n_in
    if case.id.startswith("gpt2_greedy"):
        ro = P.circuit.root_outputs()
        res["root_outputs"] = len(ro)
        res["steps_preserved"] = len(ro) == module.STEPS + 1
    return res


def provenance_of(out: Outcome) -> dict:
    rep = out.report
    prov = getattr(rep, "provenance", None) if rep is not None else None
    if isinstance(rep, dict):
        prov = rep.get("provenance", rep)
    if not isinstance(prov, dict) and out.program is not None:
        prov = out.program.annotations.get("derive") or out.program.annotations.get("provenance") or {}
    return prov if isinstance(prov, dict) else {}


def rules_fired(out: Outcome) -> list[str]:
    """Rule ids that fired, from provenance (`rules_applied`, else the `applications` / `rules` records)."""
    prov = provenance_of(out)
    if prov.get("rules_applied"):
        return sorted(prov["rules_applied"])
    rules = prov.get("applications") or prov.get("rules") or prov.get("rule_applications") or []
    names: list[str] = []
    for r in rules:
        if isinstance(r, dict):
            names.append(r.get("rule") or r.get("id") or json.dumps(r, default=str)[:60])
        else:
            names.append(getattr(r, "rule_id", None) or getattr(r, "id", None) or str(r)[:60])
    return sorted(set(names))


def kinds_emitted(out: Outcome) -> list[str]:
    prov = provenance_of(out)
    ks = {a.get("kind") or a.get("emitted") for a in prov.get("applications", []) if isinstance(a, dict)}
    ks.discard(None)
    return sorted(str(k) for k in ks)


# ---------------------------------------------------------------------------------------------------------
# the run: every case, reported as-is
# ---------------------------------------------------------------------------------------------------------

OUT_PATH = os.environ.get("XFER_OUT") or f"/tmp/xfer/transfer-{'derive' if _DERIVE else 'shipped'}.jsonl"
def _git_head() -> str:
    if os.environ.get("XFER_HEAD"):
        return os.environ["XFER_HEAD"]
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


GIT_HEAD = _git_head()


def _record(rec: dict) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


@pytest.mark.parametrize("case", [pytest.param(c, marks=pytest.mark.slow) if c.id == "gpt2_greedy_f32" else c for c in CASES], ids=[c.id for c in CASES])
def test_transfer_case(case: Case):
    torch.manual_seed(0)
    module = case.make().eval()
    rec: dict[str, Any] = {"case": case.id, "dtype": case.dtype, "importer": IMPORTER_NAME, "head": GIT_HEAD,
                           "expect": case.expect, "torch": torch.__version__, "registry": registry_version()["digest"][:16]}
    rec["importer_kw"] = case.importer_kw or {"mode": "reference"}
    out = run_importer(module, tuple(t.to("meta") if isinstance(t, torch.Tensor) else t for t in case.inputs), **(case.importer_kw or {}))
    rec["outcome"] = out.kind
    rec["reason"] = out.reason
    rec["detail"] = out.detail[:600]
    rec["rules_fired"] = rules_fired(out)
    rec["kinds_emitted"] = kinds_emitted(out)
    prov = provenance_of(out)
    if prov.get("unsupported"):
        rec["unsupported"] = prov["unsupported"]
    if prov.get("vocabulary"):
        rec["vocabulary"] = prov["vocabulary"] if isinstance(prov["vocabulary"], str) else json.dumps(prov["vocabulary"], default=str)[:200]
    if prov.get("families_not_applied") or prov.get("families_not_matched") or prov.get("family"):
        rec["family"] = {k: prov.get(k) for k in ("family", "families_not_applied", "families_not_matched") if prov.get(k)}
    if out.kind == "program":
        rec["program"] = out.program.fn.id
        rec["digest"] = program_digest(out.program)
        rec["gates"] = out.program.gates
        rec["structure"] = structural_checks(case, module, case.inputs, out)
        rec["evaluation"] = evaluate_against_torch(out, module, case.inputs)
    _record(rec)
    print("\nXFER", json.dumps(rec, default=str)[:1500])
    # The test itself passes when the importer behaved lawfully: a complete Program that matches torch and
    # preserves sharing, or an explicit refusal. A crash (unclassified failure) or a Program that is wrong /
    # loses sharing / shrinks the bounded request is a failure.
    if out.kind == "crash":
        pytest.fail(f"CRASH-NOT-REFUSAL {case.id}: {out.reason}: {out.detail[:300]}")
    if out.kind == "program":
        s = rec["structure"]
        assert not s["sharing_lost"], f"{case.id}: tied/shared parameters exported as separate inputs ({s})"
        assert s["one_input_per_distinct_leaf"], f"{case.id}: input gates {s['input_gates']} != distinct leaves {s['distinct_param_leaves']} + inputs {s['input_leaves']}"
        if "steps_preserved" in s:
            assert s["steps_preserved"], f"{case.id}: bounded request shrank to {s['root_outputs']} outputs"
        ev = rec["evaluation"]
        assert ev.get("evaluated"), f"{case.id}: derived program could not be evaluated: {ev.get('why')}"
        assert ev["match"], f"{case.id}: derived program disagrees with torch CPU: {ev}"


class _ViewsOnly(nn.Module):
    """Pure reference rearrangement (no gates): the shipped route accepts it, so the evaluator's binding /
    decoding path is exercised end to end (self-test of `evaluate_against_torch`, not a transfer case)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(8))

    def forward(self, x):
        y = torch.cat([x.view(2, 4).t().reshape(-1), self.w])
        return y, x[3]


def test_evaluator_selftest_views_only():
    torch.manual_seed(1)
    m = _ViewsOnly()
    x = torch.randn(8)
    out = run_importer(m, (x.to("meta"),))
    assert out.kind == "program", out
    ev = evaluate_against_torch(out, m, (x,))
    assert ev["evaluated"] and ev["match"] and ev["bit_exact"], ev
    s = structural_checks(Case("selftest", None, (x,), "f32", ""), m, (x,), out)
    assert s["one_input_per_distinct_leaf"] and not s["sharing_lost"]


# ---------------------------------------------------------------------------------------------------------
# support-matrix audit probes (heavy: ~8 s / 0.5 GB; run with XFER_HEAVY=1, on the pod)
# ---------------------------------------------------------------------------------------------------------

CERTIFIED_B0_DIGEST = "d4f4b1f1b4d592d697eef5b6c9c803108bdf02d0ef24c945eace6e705832e1b8"  # INTPAD2-B0-20260909T013519Z/certificate.json


@pytest.mark.skipif(not os.environ.get("XFER_HEAVY"), reason="heavy audit probe; set XFER_HEAVY=1")
def test_audit_s3_derived_b0_serve_digest_and_kv_ownership():
    """Independent reproduction of the importer's S3 claim: `GreedyServe(B0, steps=15)` on a 32-token prompt derives, the
    family rule claims `Serve_v1{B0,LP=32,STEPS=15}`, the digest equals the certificate of record's, and this file's
    name-independent KV-ownership oracle holds on the derived program (30 layers x 47 positions)."""
    if not _DERIVE:
        pytest.skip("derive() not available")
    from verity_vllm.program.frontend import derive_examples as dex
    from verity_vllm.program.frontend.derive import derive_report
    from verity.ml.definitions.registry.b1 import B0_CONFIG
    from tests.program.test_derive_negative import check_cache_ownership, check_cache_ownership_by_kind
    rep = derive_report(dex.GreedyServe(B0_CONFIG, 15), (torch.zeros(32, dtype=torch.int32),), mode="profile", profile="b1-eager",
                        config=B0_CONFIG)
    assert rep.ok, rep.unsupported
    fam = rep.family or {}
    rec = {"case": "audit_s3_b0_serve", "root": rep.root[:40], "digest": rep.digest, "certified": rep.digest == CERTIFIED_B0_DIGEST,
           "family_matched": bool(fam.get("matched")), "gates": rep.gates, "flat_digest": rep.flat_digest,
           "head": GIT_HEAD, "importer": IMPORTER_NAME}
    rec["ownership_by_name"] = check_cache_ownership(rep.program)[:5]
    rec["ownership_by_kind"] = check_cache_ownership_by_kind(rep.program, B0_CONFIG["NL"], 32, 15)[:5]
    _record(rec)
    assert fam.get("matched"), fam
    assert rep.digest == CERTIFIED_B0_DIGEST, rec
    assert rec["ownership_by_name"] == [] and rec["ownership_by_kind"] == [], rec
