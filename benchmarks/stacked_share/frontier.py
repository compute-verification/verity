"""What honest inference needs from the circuit for frontier open models, and where their dynamic memory comes from: one
decode token-step and one prefill token of each model of `frontier_models.json`, at its reference node and workload, in
nonlinear units by class and by role, with the stacked share f of `share.py`. Nonlinear units are the gate model's
(`Accounting.Gates`): only linear maps with coefficients fixed before the coins are free (a weight times an activation),
every product of two run-dependent values is a data product, a unit, and every other nonlinear operation a unit as
property/v1 (`verity.ml.circuits.stacking`) counts it.

A model is composed from atoms: the exact-int8 profile's Definitions (vLLM's registry: `int8_exact`, `moe`, `b1`, `act`) bound
at the model's statics, whose property/v1 units `atoms.py` counts (measured), a few operations the registry has no Definition
for (linear-attention state updates, a depthwise conv, bias adds, gate products, combine adds, DSA's indexer scores, top-k and
row selection: ``model:`` atoms, one rounding, compare-select or gather unit an operation, modeled), and the data products
property/v1 left free as s8 lanes (attention's QKᵀ and PV, DSA's indexer dots, linear attention's chunked prefill products:
one unit each, from the shapes; a product an fp term already charges, as SwiGLU's or MoE's combine, isn't charged twice). DSA's served circuit
(`verity.ml.definitions.registry.dsa_topk`: a Waksman network over the whole history and a top-k check of N f32 scores in one
primitive) is refused by property/v1 (`Xor3x16`, `DsaTopKOrderCheck` unlisted), so its row selection is costed as gathers,
each selected row's words and its index bits. Each term of a composition is one of them times its instances a token, under a
role; `compose` builds them, `evaluate` sums them, and `check_anchor` holds the composition of Llama-3.1-8B to the units the
int8 census measured on its whole Program.

The reference profile is exact int8: every linear an `Int8RowQuant` (or the row's shared quantization) and a `ScaledMmS8`, its
dots the s8 family's (free: a weight times an activation; attention's, two run values, are data products), the LM head an int8
linear, sampling greedy (`TokenSelect`), the KV stored as built (bf16, K' and
V' quantized at every read: `Int8KvQuant` over the window) or at write (int8 codes under static scales: `Int8StaticQuant` once
a token). One logical program: tensor and expert parallelism change who computes a term, not the terms.

The serving model (`serve`) is a roofline: a decode step reads the weights (the experts a batch touches) and every request's
dynamic memory (a DSA latent only at its top-k rows), at eta_memory of the node's bandwidth, or computes its products at
eta_compute of its int8 rate, whichever is slower; the pool holds as many requests at context S as fit; a prefill is
compute-bound. R_cap is the larger of the two phases' unit rates, and

    f = (128 R_cap L / E + R_in L) / M_dyn

at E a word in nonlinear units (`E_WORD`), with f again for the decode rate alone, for the softmax's work a key at
property/v1's floor (`softmax_floor`), for that with every requantization and linear epilogue at one rounding an element too
(`exact_floor`; no floor removes a data product), and before data products (property/v1's units without them, at
`E_BEFORE`). `sweep` gives f again as the pool's context takes the workload's ``sweep`` values.

    python frontier.py --atoms-needed > atoms-needed.json            # the atoms atoms.py counts
    python frontier.py --atoms atoms.json --out frontier.json          # the table (and frontier.md beside it)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
CLASSES = ("roundings", "compare_select", "gathers", "folds", "products")
#: E a word in nonlinear units, at p1024, k = 3, S = 1,021:
#: conjectured, `SquaringFloor` and `ChainSum` at 2^-40 (k S (2L - 1 + L - d), L = 128); best found, the seam square's 16,544
#: nonlinear units a squaring times a word's k S squarings
E_WORD = {"conjectured": 1_145_562, "best found": 16_544 * 3 * 1021}
#: the conjectured E in property/v1's budgeted units (`EncodeChainHardness`, since falsified), for the "before" column:
#: f as this census had it, without data products
E_BEFORE = {"conjectured": 364_497}
LIFETIMES = {"10 min": 600, "1 h": 3600}
R_IN = 100_000
KV_MODES = ("as built", "at write")
PHASES = ("decode", "prefill")

#: the atoms' run dependence by family: which parameters depend on a run input (a scale, a weight or a bias is a constant)
DYN: dict[str, list[bool]] = {
    "Int8RowQuant_v1": [True],
    "Int8StaticQuant_v1": [True, False],
    "ScaledMmS8_v1": [True, True, False, False],
    "RMSNormFusedCuda_v2": [True, True, False],
    "RMSNormTriton_v1": [True, False],
    "SiluMul_v2": [True],
    "RoPE_v1": [True, False],
    "Embedding_v1": [True, False],
    "TokenSelect_v1": [True],
    "MoeRouterSigmoidBias_v1": [True, False],
    "MoeRouterTopKOrderedNorm_v1": [True],
    "MoeExpertRow_v1": [True, False],
    "NvSigmoid_v1": [True],
    "NvExpf_v1": [True],
    "Int8AttnOut_v1": [True, True, True, True, True],
    "Int8LutAttentionRow_v1": [True, True, False, True, True, True],
    "Int8KvQuant_v1": [True, True, False],
}
#: the families whose units grow with S, counted at `FIT_S` and written const + slope S (held to the last point)
FITTED = ("Int8LutAttentionRow_v1", "Int8KvQuant_v1")
FIT_S = (64, 128, 256)
#: the anchor's embedding: its table bound as a run input, as the census's Program has it
EMB_DYN_TABLE = "@table-dynamic"
#: the modeled atoms: one operation on a run-dependent value
MODELED: dict[str, dict[str, int]] = {
    "model:fp": {"roundings": 1},
    "model:cs": {"compare_select": 1},
    "model:gather": {"gathers": 1},
    "model:product": {"products": 1},
}
#: a modeled MoE dispatch reads one expert index a slot: ceil(log2(E + 1)) gather bits (property/v1's ``range`` kind), when the
#: registry's `MoeExpertRow_v1` isn't counted
FP, CS, GA = "model:fp", "model:cs", "model:gather"
#: a data product: a lane whose two multiplicands both depend on run data, one nonlinear unit (``dataProducts``),
#: counted from the shapes (the attention row's 2 S D a head is `products_pass.py`'s measured 16,384 at D = 128, S = 64)
PRODUCT = "model:product"


def key(family: str, **statics: Any) -> str:
    return family + "{" + ",".join(f"{k}={v}" for k, v in statics.items()) + "}"


def parse(k: str) -> tuple[str, dict[str, Any]]:
    """An atom key's family and statics (numbers as numbers)."""
    fam, _, rest = k.split("@")[0].partition("{")
    st: dict[str, Any] = {}
    for kv in filter(None, rest.rstrip("}").split(",")):
        name, v = kv.split("=", 1)
        try:
            st[name] = int(v)
        except ValueError:
            try:
                st[name] = float(v)
            except ValueError:
                st[name] = v
    return fam, st


def RQ(K): return key("Int8RowQuant_v1", K=K)
def SQ(K): return key("Int8StaticQuant_v1", K=K)
def MM(K, N): return key("ScaledMmS8_v1", K=K, N=N)
def NF(N, eps): return key("RMSNormFusedCuda_v2", N=N, EPS=eps)
def NT(N, eps): return key("RMSNormTriton_v1", N=N, EPS=eps)
def SILU(I): return key("SiluMul_v2", I=I)
def ROPE(n, d): return key("RoPE_v1", NHEADS=n, D=d)
def EMB(V, H): return key("Embedding_v1", V=V, H=H)
def TS(V): return key("TokenSelect_v1", V=V)
def RSIG(E, k, scale): return key("MoeRouterSigmoidBias_v1", E=E, TOPK=k, SCALE=scale, KERNEL="block" if E >= 256 else "warp")
def RSOFT(E, k): return key("MoeRouterTopKOrderedNorm_v1", E=E, TOPK=k, VPT=8)
def XROW(E): return key("MoeExpertRow_v1", E=E, K=32)
def ROW(D): return key("Int8LutAttentionRow_v1", D=D, SCALE=1 / math.sqrt(D))
def KVQ(D): return key("Int8KvQuant_v1", D=D)
SIG = key("NvSigmoid_v1")
EXP = key("NvExpf_v1")
ATTN_OUT = key("Int8AttnOut_v1")


@dataclass(frozen=True)
class Term:
    """``n`` instances a token of `atom` under `role`; a fitted atom at S = ``s``."""

    role: str
    atom: str
    n: Fraction
    s: Fraction | None = None


class Build:
    """A composition's terms, its constant-weight MACs and its data-by-data products a token (``dd``, by kind; ``fp_kinds``
    those its fp terms already charge); `mult` scales what is added (a layer kind's instances)."""

    def __init__(self) -> None:
        self.terms: list[Term] = []
        self.macs = Fraction(0)
        self.dd: Counter = Counter()
        self.fp_kinds: set[str] = set()
        self.mult = Fraction(1)

    def add(self, role: str, atom: str, n: Any = 1, s: Any = None) -> None:
        if n:
            self.terms.append(Term(role, atom, Fraction(n) * self.mult, None if s is None else Fraction(s)))

    def linear(self, K: int, N: int, n: Any = 1, quant: bool = True) -> None:
        """An int8 linear: its row's quantization (unless shared with the linear before it) and its epilogue."""
        if quant:
            self.add("requantization", RQ(K), n)
        self.add("linear epilogue", MM(K, N), n)
        self.macs += Fraction(n) * K * N * self.mult

    def products(self, kind: str, n: Any, role: str | None = None) -> None:
        """n data-by-data products a token: under `role` a data product a unit (property/v1 left them free: s8 lanes); without
        one, products an fp term of the composition already charges (an f32 product of two activations is one rounding)."""
        self.dd[kind] += Fraction(n) * self.mult
        if role is None:
            self.fp_kinds.add(kind)
        else:
            self.add(role, PRODUCT, n)


# ---- the architectures -----------------------------------------------------------------------------------------------------


def layer_kinds(m: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """(attention kind, MLP kind, layers) of the model: ``full`` / ``linear`` attention, ``dense`` / ``moe`` MLP."""
    L, mlp = m["layers"], m["mlp"]
    every = m.get("full_every")
    dense = mlp.get("first_dense", 0) if "experts" in mlp else L
    out: Counter = Counter()
    for i in range(L):
        att = "full" if every is None or (i + 1) % every == 0 else "linear"
        out[att, "dense" if i < dense else "moe"] += 1
    return [(a, b, n) for (a, b), n in sorted(out.items())]


def attention_gqa(b: Build, m: Mapping[str, Any], phase: str, kv: str, S: int, P: int, dyn: list) -> None:
    H, Hq, G, D, rot, eps = m["H"], m["heads"], m["kv_heads"], m["head_dim"], m["rope_dim"], m["eps"]
    gate = bool(m.get("gate"))
    keys = S if phase == "decode" else Fraction(P + 1, 2)
    b.add("norm", NF(H, eps))
    b.linear(H, (Hq * (2 if gate else 1) + 2 * G) * D)
    if m.get("bias"):
        b.add("bias", FP, (Hq + 2 * G) * D)
    if m.get("qk_norm") == "head":
        b.add("norm", NT(D, eps), Hq + G)
    elif m.get("qk_norm") == "layer":
        b.add("norm", NT(Hq * D, eps))
        b.add("norm", NT(G * D, eps))
    b.add("rope", ROPE(Hq, rot))
    b.add("rope", ROPE(G, rot))
    b.add("attention", ROW(D), Hq, s=keys)
    if kv == "as built":
        b.add("KV requant at read", KVQ(D), G if phase == "decode" else Fraction(G, P), s=S if phase == "decode" else P)
    else:
        b.add("KV write", SQ(D), 2 * G)
    b.products("QK^T", Hq * D * keys, "QK^T and PV")
    b.products("PV", Hq * D * keys, "QK^T and PV")
    if gate:
        b.add("gate", SIG, Hq * D)
        b.add("gate", FP, Hq * D)
        b.products("gate", Hq * D)
    b.linear(Hq * D, H)
    stored = 2 * G * D
    dyn.append({"kind": "KV: K and V of G heads", "layers": int(b.mult), "values_per_token_layer": stored,
                "bytes_per_token_layer": {"as built": 2 * stored, "at write": stored}, "state_bytes_per_request_layer": 0,
                "write_path": (f"residual ({H}) -> RMSNorm -> qkv linear ({H} -> {(Hq * (2 if gate else 1) + 2 * G) * D})"
                               + (" + bias" if m.get("bias") else "") + (" -> k RMSNorm" if m.get("qk_norm") else "")
                               + f" -> RoPE on {rot} of {D} -> K, V of {G} heads ({stored} values)"
                               + " -> bf16 (as built) / Int8StaticQuant (at write)"),
                "cut_bits": {"residual": 16 * H, "write inputs": 16 * stored, "stored, at write": 8 * stored}})


def indexer_layers(m: Mapping[str, Any]) -> int:
    """The layers that build a DSA indexer (vLLM's DeepseekV32 attention: layer l skips when max(l - offset + 1, 0) % freq != 0
    and reuses the last indexer's top-k); 0 without DSA."""
    d = m.get("dsa")
    if not d:
        return 0
    freq, off = d.get("topk_freq", 1), d.get("skip_offset", 2)
    return sum(max(i - off + 1, 0) % freq == 0 for i in range(m["layers"]))


def dsa_keys(m: Mapping[str, Any], phase: str, S: int, P: int) -> dict[str, Fraction]:
    """A token's DSA counts, the mean over a prompt's positions in prefill: ``soft`` the keys its softmax reads (min(n, top-k)
    of n), ``rows`` the rows it selects (top-k where n > top-k), ``bits`` a selected row's index bits (property/v1's ``range``
    kind over n rows), ``scored`` the keys the indexer scores (n) and ``ranked`` those it ranks (n where n > top-k)."""
    K = m["dsa"]["topk"]
    ns = [S] if phase == "decode" else range(1, P + 1)
    c = Fraction(1, len(ns))
    return {"soft": c * sum(min(n, K) for n in ns), "rows": c * sum(K for n in ns if n > K),
            "bits": c * sum(K * n.bit_length() for n in ns if n > K), "scored": c * sum(ns), "ranked": c * sum(n for n in ns if n > K)}


def attention_dsa_indexer(b: Build, m: Mapping[str, Any], phase: str, S: int, P: int, dyn: list) -> None:
    """The lightning indexer of a DSA layer, in the exact-int8 profile: index q from q_c, index k and the head weights from the
    residual (their rows' quantizations shared), k's LayerNorm and both RoPEs, int8 q and k (k stored), then a key's score
    sum_h w_h ReLU(q_h k) (the dots the s8 family's, a ReLU, a weight product and an accumulate a head-key, the key's scale a
    key: modeled) and the top-k past top-k keys (modeled: a threshold compare and its tie rule a key). Served, q and k are
    FP8 (`DsaMqaLogits`), which property/v1 refuses past B = 128."""
    H, ql, rope = m["H"], m["q_lora"], m["rope"]
    Hi, di = m["dsa"]["heads"], m["dsa"]["dim"]
    r = Fraction(indexer_layers(m), m["layers"])
    k = dsa_keys(m, phase, S, P)
    keep, b.mult = b.mult, b.mult * r
    b.linear(ql, Hi * di, quant=False)
    b.linear(H, di, quant=False)
    b.linear(H, Hi, quant=False)
    b.add("norm", NT(di, 1e-06))
    b.add("rope", ROPE(Hi, rope))
    b.add("rope", ROPE(1, rope))
    b.add("requantization", RQ(di), Hi)
    b.add("KV write", RQ(di))
    b.add("DSA indexer", CS, Hi * k["scored"])
    b.add("DSA indexer", FP, 2 * Hi * k["scored"] + k["scored"])
    b.add("DSA top-k", CS, 2 * k["ranked"])
    b.products("indexer q k^T", Hi * di * k["scored"], "DSA indexer")
    b.products("indexer weights", Hi * k["scored"])
    dyn.append({"kind": "DSA indexer key", "layers": b.mult, "values_per_token_layer": di,
                "bytes_per_token_layer": {"as built": di + 4, "at write": di + 4}, "state_bytes_per_request_layer": 0,
                "write_path": (f"residual ({H}) -> wk ({H} -> {di}) -> LayerNorm -> RoPE on {rope} of {di} -> int8 under its "
                               f"f32 scale (served: e4m3 under its f32 scale) ({di} values + 4 B)"),
                "cut_bits": {"residual": 16 * H, "write inputs": 16 * di, "stored, at write": 8 * di + 32}})
    b.mult = keep


def attention_mla(b: Build, m: Mapping[str, Any], phase: str, kv: str, S: int, P: int, dyn: list) -> None:
    H, Hq, ql, kl, nope, rope, vh, eps = (m[k] for k in ("H", "heads", "q_lora", "kv_lora", "nope", "rope", "v_head", "eps"))
    dsa = "dsa" in m
    keys = S if phase == "decode" else Fraction(P + 1, 2)
    b.add("norm", NF(H, eps))
    b.linear(H, ql + kl + rope)
    b.add("norm", NT(ql, eps))
    b.add("norm", NT(kl, eps))
    b.linear(ql, Hq * (nope + rope))
    b.add("rope", ROPE(Hq, rope))
    b.add("rope", ROPE(1, rope))
    if dsa:
        attention_dsa_indexer(b, m, phase, S, P, dyn)
        k = dsa_keys(m, phase, S, P)
        keys = k["soft"]
        b.add("DSA selection", GA, k["rows"] * (kl + rope) + k["bits"])
    if phase == "decode" or dsa:
        b.linear(nope, kl, n=Hq)
        b.add("attention", ROW(kl + rope), Hq, s=keys)
        b.add("attention", ATTN_OUT, -Hq * rope)
        if kv == "as built":
            read, w = (keys, 1) if phase == "decode" else (P, Fraction(1, P))
            b.add("KV requant at read", KVQ(kl), w, s=read)
            b.add("KV requant at read", SQ(rope), w * read)
        else:
            b.add("KV write", SQ(kl + rope), 1)
        b.linear(kl, vh, n=Hq)
        b.products("QK^T", Hq * (kl + rope) * keys, "QK^T and PV")
        b.products("PV", Hq * kl * keys, "QK^T and PV")
    else:
        b.linear(kl, Hq * (nope + vh))
        b.add("attention", ROW(nope + rope), Hq, s=keys)
        b.add("attention", ATTN_OUT, -Hq * (nope + rope - vh))
        if kv == "as built":
            b.add("KV requant at read", KVQ(vh), Fraction(Hq, P), s=P)
            b.add("KV requant at read", SQ(nope + rope - vh), Hq)
        else:
            b.add("KV write", SQ(kl + rope), 1)
            b.add("requantization", SQ(nope + rope), Hq)
            b.add("requantization", SQ(vh), Hq)
        b.products("QK^T", Hq * (nope + rope) * keys, "QK^T and PV")
        b.products("PV", Hq * vh * keys, "QK^T and PV")
    b.linear(Hq * vh, H)
    stored = kl + rope
    dyn.append({"kind": "MLA latent c_KV and the RoPE key", "layers": int(b.mult), "values_per_token_layer": stored,
                "bytes_per_token_layer": {"as built": 2 * stored, "at write": stored}, "state_bytes_per_request_layer": 0,
                **({"read_tokens": m["dsa"]["topk"]} if dsa else {}),
                "write_path": (f"residual ({H}) -> RMSNorm -> kv_a ({H} -> {kl} + {rope}, fused with q_a) -> c_KV RMSNorm ({kl}); "
                               f"k_rope RoPE ({rope}) -> {stored} values -> bf16 (as built) / Int8StaticQuant (at write); "
                               f"upstream of the latent: the {H}-wide residual and W_DKV"),
                "cut_bits": {"residual": 16 * H, "write inputs": 16 * stored, "stored, at write": 8 * stored}})


def attention_deltanet(b: Build, m: Mapping[str, Any], phase: str, kv: str, S: int, P: int, dyn: list) -> None:
    H, eps, lin = m["H"], m["eps"], m["linear"]
    Hk, Hv, dk, dv, conv = lin["key_heads"], lin["value_heads"], lin["dk"], lin["dv"], lin["conv"]
    nqkv, nz = 2 * Hk * dk + Hv * dv, Hv * dv
    b.add("norm", NF(H, eps))
    b.linear(H, nqkv + nz)
    b.linear(H, 2 * Hv, quant=False)
    b.add("conv", FP, conv * nqkv)
    b.macs += conv * nqkv * b.mult
    b.add("activation", FP, nqkv)
    b.add("norm", NT(dk, eps), 2 * Hk)
    b.add("gate", SIG, Hv)
    b.add("gate", EXP, 2 * Hv)
    b.add("gate", FP, 4 * Hv)
    if phase == "decode":
        b.add("state update", FP, 4 * Hv * dk * dv + 2 * Hv * dv)
        b.products("state (kS, k err^T, S^T q)", 3 * Hv * dk * dv)
        b.products("state decay", Hv * dk * dv)
    else:
        # the chunked form prefill runs (chunks of C): the state is rewritten once a chunk, and a chunk's tokens meet each
        # other through C x C matrices (beta K K^T and its decay, the WY inverse T, Q K^T)
        C = lin["chunk"]
        b.add("gate", EXP, Fraction(Hv * C, 2))
        b.add("state update", FP, Hv * (Fraction(3 * dk * dv, C) + 3 * C + 2 * dk + 9 * dv))
        b.products("state (K^T U, q S, W S)", 3 * Hv * dk * dv, "state products")
        b.products("state decay", Fraction(Hv * dk * dv, C))
        b.products("intra-chunk (A, T, W, U, QK^T, PV)", Hv * (Fraction(3 * C * dk, 2) + C * dv + Fraction(C * C, 6)),
                   "state products")
    b.add("norm", NT(dv, eps), Hv)
    b.add("activation", FP, 2 * nz)
    b.products("gate", nz)
    b.linear(Hv * dv, H)
    state, cst = Hv * dk * dv * 4, (conv - 1) * nqkv * 2
    write_in = Hk * dk + Hv * dv + 2 * Hv
    dyn.append({"kind": "Gated DeltaNet state (f32) and conv state", "layers": int(b.mult), "values_per_token_layer": 0,
                "bytes_per_token_layer": {"as built": 0, "at write": 0}, "state_bytes_per_request_layer": state + cst,
                "state_rewritten_bytes_per_step": state + 2 * nqkv,
                "write_path": (f"residual ({H}) -> RMSNorm -> in_proj ({H} -> {nqkv + nz} + {2 * Hv}) -> conv{conv} + SiLU -> "
                               f"L2 norm of q, k; beta = sigmoid(b), g = -exp(A) softplus(a + dt) -> S <- S (exp(g) - beta k k^T) "
                               f"+ beta v k^T, {Hv} heads of {dk}x{dv} f32 rewritten every decode token (once a chunk of "
                               f"{lin['chunk']} in prefill)"),
                "cut_bits": {"residual": 16 * H, "write inputs": 16 * write_in, "stored, at write": 8 * (state + cst)}})


def attention_lightning(b: Build, m: Mapping[str, Any], phase: str, kv: str, S: int, P: int, dyn: list) -> None:
    H, eps, lin = m["H"], m["eps"], m["linear"]
    Hh, dk, dv = lin["heads"], lin["dk"], lin["dv"]
    b.add("norm", NF(H, eps))
    b.linear(H, 3 * Hh * dk)
    b.add("activation", FP, 3 * Hh * dk)
    if phase == "decode":
        b.add("state update", FP, 3 * Hh * dk * dv)
        b.products("state (k^T v, q S)", 2 * Hh * dk * dv)
    else:
        # the blocked form prefill runs (blocks of C): the state is rewritten once a block, and a block's tokens meet each
        # other through their decay-masked Q K^T
        C = lin["chunk"]
        b.add("state update", FP, Hh * (Fraction(3 * dk * dv, C) + C + 2 * dk + 3 * dv))
        b.products("state (k^T v, q S)", 2 * Hh * dk * dv, "state products")
        b.products("intra-block (QK^T, PV)", Hh * Fraction(C * (dk + dv), 2), "state products")
    b.add("norm", NT(Hh * dv, eps))
    b.linear(H, Hh * dv, quant=False)
    b.add("gate", SIG, Hh * dv)
    b.add("gate", FP, Hh * dv)
    b.products("gate", Hh * dv)
    b.linear(Hh * dv, H)
    state = Hh * dk * dv * 4
    dyn.append({"kind": "lightning-attention state (f32)", "layers": int(b.mult), "values_per_token_layer": 0,
                "bytes_per_token_layer": {"as built": 0, "at write": 0}, "state_bytes_per_request_layer": state,
                "state_rewritten_bytes_per_step": state,
                "write_path": (f"residual ({H}) -> RMSNorm -> qkv ({H} -> {3 * Hh * dk}) -> SiLU -> S <- lambda_h S + k^T v, "
                               f"{Hh} heads of {dk}x{dv} f32 (lambda_h a constant decay) rewritten every decode token "
                               f"(once a block of {lin['chunk']} in prefill)"),
                "cut_bits": {"residual": 16 * H, "write inputs": 16 * Hh * (dk + dv), "stored, at write": 8 * state}})


ATTENTION = {"gqa": attention_gqa, "mla": attention_mla}
LINEAR = {"hybrid-deltanet": attention_deltanet, "hybrid-lightning": attention_lightning}


def mlp_dense(b: Build, m: Mapping[str, Any], I: int) -> None:
    H = m["H"]
    b.add("norm", NF(H, m["eps"]))
    b.linear(H, 2 * I)
    b.add("activation", SILU(I))
    b.products("SwiGLU", I)
    b.linear(I, H)


def mlp_moe(b: Build, m: Mapping[str, Any]) -> None:
    H, mlp = m["H"], m["mlp"]
    E, k, I = mlp["experts"], mlp["topk"], mlp["expert_I"]
    ns, Is = mlp.get("shared", 0), mlp.get("shared_I", 0)
    b.add("norm", NF(H, m["eps"]))
    b.add("requantization", RQ(H))
    b.linear(H, E, quant=False)
    if mlp["router"] == "sigmoid":
        b.add("MoE routing", RSIG(E, k, mlp["scale"]))
        g, kg = mlp.get("n_group", 1), mlp.get("topk_group", 1)
        if g > 1:
            b.add("MoE routing", CS, 5 * E + 2 * g * kg)
            b.add("MoE routing", FP, g)
    else:
        b.add("MoE routing", RSOFT(E, k))
    b.add("MoE dispatch", XROW(E), k)
    b.linear(H, 2 * I, n=k, quant=False)
    b.add("activation", SILU(I), k)
    b.products("SwiGLU", k * I)
    b.linear(I, H, n=k)
    b.add("MoE combine", FP, k * H)
    b.products("MoE combine", k * H)
    if ns:
        b.linear(H, 2 * ns * Is, quant=False)
        b.add("activation", SILU(ns * Is))
        b.products("SwiGLU", ns * Is)
        b.linear(ns * Is, H)
        if mlp.get("shared_gate"):
            b.linear(H, 1, quant=False)
            b.add("gate", SIG, 1)
            b.add("gate", FP, H)
            b.products("gate", H)
    b.add("MoE combine", FP, (k - 1 + (1 if ns else 0)) * H)


def compose(m: Mapping[str, Any], phase: str, kv: str, S: int, P: int, *, anchor: bool = False) -> dict:
    """One token's terms (a decode token at context S, or a prefill token of a prompt of P from position 0), its constant-weight
    MACs and data-by-data products, and its dynamic memory, KV as built or at write. `anchor`: the census Program's own form
    (its embedding's table a run input, no LM head or sampling: those are a sample's)."""
    b, dyn = Build(), []
    H, V, eps = m["H"], m["vocab"], m["eps"]
    for att, mlp, n in layer_kinds(m):
        b.mult = Fraction(n)
        (ATTENTION[m["arch"]] if att == "full" and m["arch"] in ATTENTION else
         attention_gqa if att == "full" else LINEAR[m["arch"]])(b, m, phase, kv, S, P, dyn)
        if m.get("postnorm"):
            b.add("residual scale", FP, 2 * H)
        mlp_dense(b, m, m["mlp"]["dense"]) if mlp == "dense" else mlp_moe(b, m)
    b.mult = Fraction(1)
    b.add("embedding", EMB(V, H) + (EMB_DYN_TABLE if anchor else ""))
    b.add("norm", NT(H, eps))
    if not anchor:
        w = 1 if phase == "decode" else Fraction(1, P)
        b.linear(H, V, n=w)
        b.add("sampling", TS(V), w)
    merged: dict[str, dict] = {}
    for d in dyn:
        if d["kind"] in merged:
            merged[d["kind"]]["layers"] += d["layers"]
        else:
            merged[d["kind"]] = dict(d)
    for d in merged.values():
        if Fraction(d["layers"]).denominator != 1:
            raise AssertionError(f"{d['kind']}: {d['layers']} layers")
        d["layers"] = int(d["layers"])
    return {"terms": b.terms, "macs": b.macs, "dd": dict(b.dd), "dd_fp": sorted(b.fp_kinds), "dyn": list(merged.values())}


# ---- the atoms -------------------------------------------------------------------------------------------------------------


def needed(models: Mapping[str, Mapping[str, Any]], S: int, P: int) -> list[str]:
    """Every atom key the compositions read, the anchor's among them (fitted families without S: atoms.py fits them)."""
    out: dict[str, None] = {}
    for name, m in models.items():
        for phase in PHASES:
            for kv in KV_MODES:
                for t in compose(m, phase, kv, S, P)["terms"]:
                    if not t.atom.startswith("model:"):
                        out[t.atom] = None
        if name == "Llama-3.1-8B":
            out[EMB(m["vocab"], m["H"]) + EMB_DYN_TABLE] = None
    return list(out)


def atom_units(atoms: Mapping[str, Any], atom: str, s: Fraction | None, what: str = "units") -> dict[str, Fraction]:
    """An atom's units by class (`what` ``primitives``: by ``primitive|class``), at S = `s` for a fitted one."""
    if atom.startswith("model:"):
        return {f"{atom}|{c}" if what == "primitives" else c: Fraction(v) for c, v in MODELED[atom].items()}
    a = atoms[atom]
    if "slope" in a:
        if s is None:
            raise ValueError(f"{atom} is fitted in S: a term reads it at an S")
        c, sl = a["const"][what], a["slope"][what]
        return {k: Fraction(c.get(k, 0)) + Fraction(sl.get(k, 0)) * s for k in set(c) | set(sl)}
    return {k: Fraction(v) for k, v in a[what].items()}


def evaluate(terms: Iterable[Term], atoms: Mapping[str, Any], primitives: bool = True) -> dict:
    """The terms' nonlinear units a token: by class, by role and class, by source (measured atoms, modeled operations, data
    products from the shapes), and by ``primitive|class``."""
    cls: Counter = Counter()
    role: dict[str, Counter] = {}
    src: Counter = Counter()
    prim: Counter = Counter()
    for t in terms:
        u = atom_units(atoms, t.atom, t.s)
        for c, v in u.items():
            cls[c] += t.n * v
            role.setdefault(t.role, Counter())[c] += t.n * v
            src["shapes" if t.atom == PRODUCT else "modeled" if t.atom.startswith("model:") else "measured"] += t.n * v
        if primitives:
            for k, v in atom_units(atoms, t.atom, t.s, "primitives").items():
                prim[f"{t.role}|{k}"] += t.n * v
    return {"by_class": {c: cls[c] for c in CLASSES if cls[c]}, "total": sum(cls.values(), Fraction(0)),
            "by_role": {r: dict(v) for r, v in role.items()}, "by_source": dict(src), "by_role_primitive": dict(prim)}


#: an operation kind by primitive id, within its class (the first pattern that matches)
KINDS = {
    "roundings": (("division", r"Div|Rcp|InvSum"), ("rsqrt / sqrt", r"Rsqrt|Sqrt"),
                  ("exp, log, tanh, SiLU", r"Ex2|Exp|Log|Lg2|Tanh|Silu|Gelu|Erf|Sigmoid"),
                  ("requantization: round or convert", r"Rint|ToS8|ToF32|ToBf16|F2fp|LowByte|ToU32|LutIndex|ToE4m3|ToE2m1|Rne"),
                  ("other FP arithmetic", r".")),
    "compare_select": (("max, min, clamp", r"max|min|Clamp|Sat|Max|Min"), ("select", r"Select|Sel"),
                       ("compare", r"Gt|Le|Eq|Ge|Lt|IsFinite"), ("logic", r"Bit|And|Xor|Not|Or"), ("other", r".")),
    "gathers": (("exp table", r"Exp2Lut|TableRead"), ("index bits", r".")),
    "folds": (("folds", r"."),),
    "products": (("data products", r"."),),
}


#: the roles a token's units are grouped by in the markdown's shares (the last takes every other role)
SHARES = (("softmax attention", ("attention",)), ("QKᵀ and PV (data products)", ("QK^T and PV",)),
          ("requantization and linear epilogues", ("requantization", "linear epilogue")),
          ("KV quantization", ("KV requant at read", "KV write")),
          ("linear-attention state", ("state update", "conv", "state products")),
          ("sparse attention: indexer, top-k, selection", ("DSA indexer", "DSA top-k", "DSA selection")),
          ("MoE routing and dispatch", ("MoE routing", "MoE dispatch")), ("other", ()))


def shares(by_role: Mapping[str, Fraction]) -> dict[str, Fraction]:
    out = {name: Fraction(0) for name, _ in SHARES}
    for role, v in by_role.items():
        out[next((name for name, roles in SHARES if role in roles), "other")] += v
    return out


#: the softmax's own roundings a key (the score less the row's max, its LUT index, the exp's low byte)
SOFTMAX_ROUNDINGS = r"LowByte|LutIndex|I32Sub"


def kinds(prim: Mapping[str, Fraction]) -> dict[str, dict[str, Fraction]]:
    """Units by class and operation kind, from ``role|primitive|class``; the top-k's (MoE routing's, DSA's) and the attention's
    compare-selects apart (the softmax's max), and the softmax's own roundings apart from the requantizations."""
    out: dict[str, Counter] = {}
    for k, v in prim.items():
        role, p, c = k.split("|")
        kind = next(name for name, pat in KINDS[c] if re.search(pat, p))
        if c == "roundings" and role == "attention" and re.search(SOFTMAX_ROUNDINGS, p):
            kind = "softmax: max subtract, LUT index, low byte"
        elif c == "compare_select" and role in ("MoE routing", "DSA top-k"):
            kind = "top-k (MoE routing, DSA)"
        elif c == "compare_select" and role == "attention" and kind in ("select", "compare"):
            kind = "softmax max"
        elif c == "compare_select" and role == "sampling":
            kind = "sampling (argmax)"
        out.setdefault(c, Counter())[kind] += v
    return {c: dict(v) for c, v in out.items()}


def check_anchor(models: Mapping[str, Any], atoms: Mapping[str, Any], anchor: Mapping[str, Any], S: int, P: int) -> dict:
    """Llama-3.1-8B composed from the atoms in the census Program's form, held to the units the census measured on the whole
    Program: a token row's (`compose` with ``anchor``), a sampled row's TokenSelect (its LM head is the bf16 Gemm_v2, 257 units
    an output), and the attention window's coefficients (H rows of `ROW` and G calls of `KVQ` a request-step-layer). Raises on
    any difference."""
    m = models[anchor["model"]]
    tok = evaluate(compose(m, "decode", "as built", S, P, anchor=True)["terms"], atoms, primitives=False)
    att_roles = ("attention", "KV requant at read", "QK^T and PV")
    mine = Counter()
    for r, u in tok["by_role"].items():
        if r not in att_roles:
            mine.update(u)
    want = {c: Fraction(v) for c, v in anchor["token"].items()}
    if {c: v for c, v in mine.items() if v} != want:
        raise AssertionError(f"the composed token row {dict(mine)} is not the census's {want}")
    ts = atom_units(atoms, TS(m["vocab"]), None)
    smp = {c: Fraction(v) for c, v in anchor["sample"].items()}
    smp["roundings"] -= anchor["lm_head_bf16_units_per_output"] * m["vocab"]
    if {c: v for c, v in ts.items() if v} != smp:
        raise AssertionError(f"TokenSelect's {ts} is not the census's sampled row less its LM head {smp}")
    G, Hh, D = anchor["attention"]["G"], anchor["attention"]["H"], m["head_dim"]
    row, kvq = atoms[ROW(D)], atoms[KVQ(D)]
    co = anchor["attention"]["coefficients"]
    for c in co:
        got = {"r0": Hh * Fraction(row["const"]["units"].get(c, 0)), "b_q": Hh * Fraction(row["slope"]["units"].get(c, 0)),
               "b_kv": G * Fraction(kvq["slope"]["units"].get(c, 0)), "w0": G * Fraction(kvq["const"]["units"].get(c, 0))}
        if any(got[k] != co[c][k] for k in got):
            raise AssertionError(f"the attention's {c} coefficients {got} are not the census's {co[c]}")
    return {"token": want, "sample_token_select": smp, "attention": co, "held": True}


# ---- parameters and serving ------------------------------------------------------------------------------------------------


def params(m: Mapping[str, Any]) -> dict[str, int]:
    """Weights (one a parameter, int8) by part: ``total``, ``experts`` (the routed experts') and ``lm_head``; norms and
    biases are left out."""
    H, V, mlp = m["H"], m["vocab"], m["mlp"]
    total = experts = 0
    for att, kind, n in layer_kinds(m):
        if att == "full" and m["arch"] == "mla":
            a = H * (m["q_lora"] + m["kv_lora"] + m["rope"]) + m["q_lora"] * m["heads"] * (m["nope"] + m["rope"]) \
                + m["kv_lora"] * m["heads"] * (m["nope"] + m["v_head"]) + m["heads"] * m["v_head"] * H
        elif att == "full":
            Hq, G, D = m["heads"], m["kv_heads"], m["head_dim"]
            a = H * (Hq * (2 if m.get("gate") else 1) + 2 * G) * D + Hq * D * H
        elif m["arch"] == "hybrid-deltanet":
            lin = m["linear"]
            nqkv, nz = 2 * lin["key_heads"] * lin["dk"] + lin["value_heads"] * lin["dv"], lin["value_heads"] * lin["dv"]
            a = H * (nqkv + nz + 2 * lin["value_heads"]) + lin["conv"] * nqkv + nz * H
        else:
            lin = m["linear"]
            a = H * 3 * lin["heads"] * lin["dk"] + 2 * H * lin["heads"] * lin["dv"]
        if kind == "dense":
            f = 3 * H * mlp["dense"]
            e = 0
        else:
            e = mlp["experts"] * 3 * H * mlp["expert_I"]
            f = H * mlp["experts"] + e + 3 * H * mlp.get("shared", 0) * mlp.get("shared_I", 0)
        total += n * (a + f)
        experts += n * e
    if m.get("dsa"):
        d = m["dsa"]
        total += indexer_layers(m) * (m["q_lora"] * d["heads"] * d["dim"] + H * d["dim"] + H * d["heads"])
    head = 0 if m.get("tied") else V * H
    return {"total": total + V * H + head, "experts": experts, "lm_head": V * H, "embedding": V * H}


def dynamic_bytes(dyn: Sequence[Mapping[str, Any]], kv: str, S: int) -> dict[str, int]:
    """A request's dynamic memory at context S: ``per_token`` (bytes a token, every layer), ``state`` (fixed, a request),
    ``request``, the state a step rewrites (``rewritten``), and the per-token bytes a decode step reads (``read``: an entry's
    tokens up to its ``read_tokens``, DSA's latent its top-k rows)."""
    tok = sum(d["bytes_per_token_layer"][kv] * d["layers"] for d in dyn)
    state = sum(d["state_bytes_per_request_layer"] * d["layers"] for d in dyn)
    rew = sum(d.get("state_rewritten_bytes_per_step", 0) * d["layers"] for d in dyn)
    read = sum(d["bytes_per_token_layer"][kv] * d["layers"] * min(S, d.get("read_tokens", S)) for d in dyn)
    return {"per_token": tok, "state": state, "request": tok * S + state, "rewritten": rew, "read": read}


def serve(m: Mapping[str, Any], node: Mapping[str, Any], dec: Mapping[str, Any], pre: Mapping[str, Any], kv: str, S: int,
          P: int, e_word: Mapping[str, int] = E_WORD) -> dict:
    """The roofline at the reference node (module docstring): the pool, the batch, the decode step and the prefill rate, R_cap,
    the pool's turnover, the honest work a byte written (at R_cap, ``W_byte``, and at the decode rate), and f at `e_word`."""
    gpus, tp = node["gpus"], m["tp"]
    engines = gpus // tp
    pw = params(m)
    W = pw["total"]
    usable = gpus * (node["usable"] * node["hbm_bytes"] - node["reserve_bytes"])
    M = usable - engines * W
    if M <= 0:
        raise ValueError(f"{m.get('hf')}: {engines} engines of {W / 1e9:.0f} GB of weights don't fit the node")
    db = dynamic_bytes(dec["dyn"], kv, S)
    B = int(M // db["request"])
    Be = B / engines
    touched = 1.0
    if "experts" in m["mlp"]:
        touched = 1 - (1 - m["mlp"]["topk"] / m["mlp"]["experts"]) ** Be
    W_read = engines * ((W - pw["experts"]) + pw["experts"] * touched)
    kv_read = B * (db["read"] + db["rewritten"] * 2 + (db["state"] - db["rewritten"]))
    t_mem = (W_read + kv_read) / (node["eta_memory"] * node["hbm_bytes_per_s"] * gpus)
    rate = node["eta_compute"] * node["int8_ops_per_s"] * gpus
    flops_dec = 2 * float(dec["macs"] + sum(dec["dd"].values()))
    t_cmp = B * flops_dec / rate
    t_step = max(t_mem, t_cmp)
    u_dec = float(dec["units"]["total"])
    R_dec = B * u_dec / t_step
    flops_pre = 2 * float(pre["macs"] + sum(pre["dd"].values()))
    tps_pre = rate / flops_pre
    R_pre = tps_pre * float(pre["units"]["total"])
    R_cap = max(R_dec, R_pre)
    tps_dec = B / t_step
    write = tps_dec * db["per_token"] + B * db["rewritten"] / t_step
    T_turn = M / write
    out = {"engines": engines, "weights_bytes": W, "params": pw, "M_dyn": M, "request_bytes": db, "B": B,
           "experts_touched": touched, "t_mem_s": t_mem, "t_cmp_s": t_cmp, "t_step_s": t_step,
           "bound": "memory" if t_mem >= t_cmp else "compute", "tokens_per_s_decode": tps_dec, "tokens_per_s_prefill": tps_pre,
           "R_decode": R_dec, "R_prefill": R_pre, "R_cap": R_cap, "R_cap_phase": "decode" if R_dec >= R_pre else "prefill",
           "u_B_decode": B * u_dec / M, "T_turn_s": T_turn, "W_byte": R_cap * T_turn / M, "W_byte_decode": R_dec * T_turn / M,
           "f": {lt: {e: (128 * R_cap * L / E + R_IN * L) / M for e, E in e_word.items()} for lt, L in LIFETIMES.items()},
           "f_decode": {lt: {e: (128 * R_dec * L / E + R_IN * L) / M for e, E in e_word.items()} for lt, L in LIFETIMES.items()}}
    return out


def per_key_units(m: Mapping[str, Any], atoms: Mapping[str, Any], kv: str, S: int, P: int) -> dict:
    """A decode token's units and data products a key of context, and a key's bytes: the slopes from S to S + 1, over the
    bytes a token adds."""
    c0, c1 = compose(m, "decode", kv, S, P), compose(m, "decode", kv, S + 1, P)
    a, b = (evaluate(c["terms"], atoms, primitives=False)["total"] for c in (c0, c1))
    dd = sum(c1["dd"].values()) - sum(c0["dd"].values())
    tok = dynamic_bytes(c0["dyn"], kv, S)["per_token"]
    return {"units_per_key": b - a, "products_per_key": dd, "bytes_per_key": tok, "units_per_byte": (b - a) / tok if tok else None}


def softmax_floor(atoms: Mapping[str, Any]) -> dict:
    """The atoms with the softmax's work a key (the attention row's slope in S: its max, the exp table's index bits, the LUT
    index, the low byte) at property/v1's floor, one rounding a query head a key: the exp a softmax can't do without."""
    one = {"roundings": 1}
    return {k: ({**a, "slope": {"units": one, "primitives": {"(floor)|roundings": 1}}} if k.startswith("Int8LutAttentionRow_v1")
                else a) for k, a in atoms.items()}


#: a requantization's or an epilogue's one rounding an element: the round to int8, or the output's round to bf16
ESSENTIAL_ROUNDING = {"Int8RowQuant_v1": "F32Rint_v1", "Int8StaticQuant_v1": "F32Rint_v1", "Int8KvQuant_v1": "F32Rint_v1",
                      "ScaledMmS8_v1": "F2fpBf16_v1", "Int8AttnOut_v1": "F32ToBf16Rn_v1"}


def exact_floor(atoms: Mapping[str, Any]) -> dict:
    """`softmax_floor`'s atoms with every requantization and linear epilogue also at its floor: one rounding an element, the
    count of its `ESSENTIAL_ROUNDING` primitive as measured (the amax, the clamp, the divide, the cast and the scale products
    gone), the least an exact int8 profile can be charged for them."""
    def floor(part: Mapping[str, Any], prim: str) -> dict:
        n = part.get("primitives", {}).get(f"{prim}|roundings", 0)
        return {"units": {"roundings": n} if n else {}, "primitives": {f"(floor) {prim}|roundings": n} if n else {}}
    out = {}
    for k, a in softmax_floor(atoms).items():
        prim = ESSENTIAL_ROUNDING.get(parse(k)[0])
        if prim is None:
            out[k] = a
        elif "slope" in a:
            out[k] = {**a, "const": floor(a["const"], prim), "slope": floor(a["slope"], prim)}
        else:
            out[k] = {**a, **floor(a, prim)}
    return out


def table(spec: Mapping[str, Any], atoms: Mapping[str, Any]) -> dict:
    node, wl = spec["node"], spec["workload"]
    S, P = wl["context"], wl["prompt"]
    out: dict[str, Any] = {"schema": "verity-stacked-share/frontier/v2", "units": "nonlinear", "node": node,
                           "workload": wl, "E_word": E_WORD, "E_before": E_BEFORE, "lifetimes_s": LIFETIMES, "R_in": R_IN,
                           "anchor": check_anchor(spec["models"], atoms, spec["anchor"], S, P), "models": {}}
    floors = {"softmax_floor": softmax_floor(atoms), "exact_floor": exact_floor(atoms)}
    for name, m in spec["models"].items():
        row: dict[str, Any] = {"hf": m.get("hf"), "revision": m.get("revision"), "source": m.get("source"),
                               "weights_as_published": m.get("weights"), "arch": m["arch"], "layers": layer_kinds(m)}
        for kv in KV_MODES:
            ph, fl, bf = {}, {f: {} for f in floors}, {}
            for phase in PHASES:
                c = compose(m, phase, kv, S, P)
                u = evaluate(c["terms"], atoms)
                ph[phase] = {"units": u, "kinds": kinds(u["by_role_primitive"]), "macs": c["macs"], "dd": c["dd"],
                             "dd_fp": c["dd_fp"], "dyn": c["dyn"]}
                for f, at in floors.items():
                    fl[f][phase] = {**ph[phase], "units": evaluate(c["terms"], at, primitives=False)}
                bf[phase] = {**ph[phase], "units": evaluate([t for t in c["terms"] if t.atom != PRODUCT], atoms, primitives=False)}
            sv = serve(m, node, ph["decode"], ph["prefill"], kv, S, P)
            row[kv] = {"decode": ph["decode"], "prefill": ph["prefill"], "serving": sv, "per_key": per_key_units(m, atoms, kv, S, P)}
            for f in floors:
                sf = serve(m, node, fl[f]["decode"], fl[f]["prefill"], kv, S, P)
                row[kv][f] = {"units": {p: fl[f][p]["units"]["total"] for p in PHASES},
                              **{k: sf[k] for k in ("R_decode", "R_prefill", "R_cap", "R_cap_phase", "f", "f_decode")}}
            sb = serve(m, node, bf["decode"], bf["prefill"], kv, S, P, E_BEFORE)
            row[kv]["before"] = {"units": {p: bf[p]["units"]["total"] for p in PHASES},
                                 **{k: sb[k] for k in ("R_decode", "R_prefill", "R_cap", "R_cap_phase", "f", "f_decode")}}
        out["models"][name] = row
    out["sweep"] = sweep(spec, atoms, floors)
    return out


def sweep(spec: Mapping[str, Any], atoms: Mapping[str, Any], floors: Mapping[str, Any]) -> dict:
    """f with the KV at write as the pool's context S takes each of the workload's ``sweep`` values, with prefills of
    min(P, S), at full cost and at each floor; a model past its ``max_context`` is left out."""
    node, P0 = spec["node"], spec["workload"]["prompt"]
    costs = {"full": atoms, **floors}
    out: dict[str, Any] = {}
    for name, m in spec["models"].items():
        out[name] = {}
        for S in spec["workload"]["sweep"]:
            if S > m["max_context"]:
                continue
            P = min(P0, S)
            out[name][S] = {}
            for cost, at in costs.items():
                ph = {}
                for phase in PHASES:
                    c = compose(m, phase, "at write", S, P)
                    ph[phase] = {"units": evaluate(c["terms"], at, primitives=False), "macs": c["macs"], "dd": c["dd"], "dyn": c["dyn"]}
                sv = serve(m, node, ph["decode"], ph["prefill"], "at write", S, P)
                out[name][S][cost] = {k: sv[k] for k in ("B", "M_dyn", "R_decode", "R_prefill", "R_cap_phase", "f", "f_decode")}
    return out


# ---- output ----------------------------------------------------------------------------------------------------------------


def jsonable(o: Any) -> Any:
    if isinstance(o, Fraction):
        return int(o) if o.denominator == 1 else float(o)
    if isinstance(o, Mapping):
        return {str(k): jsonable(v) for k, v in o.items() if k != "by_role_primitive"}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, Term):
        return [o.role, o.atom, jsonable(o.n), jsonable(o.s)]
    return o


def g(x: Any) -> str:
    x = float(x)
    if x == 0:
        return "0"
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.2e}"


def pct(x: float) -> str:
    return f"{x:.1%}" if x < 1 else f"{x:.3g}x"


def human(n: float, unit: str = "B") -> str:
    for p, s in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= p:
            return f"{n / p:.3g} {s}{unit}"
    return f"{n:.3g} {unit}"


#: the data products' groups in the markdown, by dd kind (the first pattern that matches)
PRODUCT_GROUPS = (("QKᵀ, PV", r"^(QK\^T|PV)$"), ("DSA indexer", r"^indexer"), ("linear-attention state and chunks", r"^(state|intra)"),
                  ("SwiGLU", r"^SwiGLU$"), ("MoE combine", r"^MoE combine$"), ("gates", r"."))


def product_groups(dd: Mapping[str, Fraction]) -> dict[str, Fraction]:
    out = {name: Fraction(0) for name, _ in PRODUCT_GROUPS}
    for kind, n in dd.items():
        out[next(name for name, pat in PRODUCT_GROUPS if re.search(pat, kind))] += Fraction(n)
    return out


def row_md(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def header_md(cols: Sequence[str]) -> list[str]:
    return [row_md(cols), "|" + "---|" * len(cols)]


def markdown(t: Mapping[str, Any]) -> str:
    wl, node = t["workload"], t["node"]
    E = ", ".join(f"{e} {v:,}" for e, v in E_WORD.items())
    lines = [f"Reference: {node['name']}, a pool full of requests decoding at context S = {wl['context']}, prefills of "
             f"P = {wl['prompt']}; exact-int8 profile; nonlinear units (property/v1's budgeted units plus one a data "
             f"product, a product of two run-dependent values); E a word {E}.", ""]
    lines += ["**The census, KV at write** (decode at S; f at L = 10 min and 1 h; before: property/v1's units without data "
              f"products, E {E_BEFORE['conjectured']:,}).", ""]
    lines += header_md(["model", "decode units a token", "of which data products", "data products a context key",
                        "units a byte written (decode / R_cap)", "R_cap (phase)", "f 10 min, E conjectured / best found",
                        "f 1 h, E conjectured / best found", "f 1 h, E conjectured, decode rate alone",
                        "before: f 1 h, E conjectured (decode alone)"])
    for name, r in t["models"].items():
        x = r["at write"]
        sv, u, bf = x["serving"], x["decode"]["units"], x["before"]
        lines.append(row_md([
            name, g(u["total"]), f"{g(u['by_class'].get('products', 0))} ({float(u['by_class'].get('products', 0) / u['total']):.0%})",
            g(x["per_key"]["products_per_key"]), f"{g(sv['W_byte_decode'])} / {g(sv['W_byte'])}",
            f"{g(sv['R_cap'])} ({sv['R_cap_phase']})",
            " / ".join(pct(sv["f"]["10 min"][e]) for e in E_WORD), " / ".join(pct(sv["f"]["1 h"][e]) for e in E_WORD),
            pct(sv["f_decode"]["1 h"]["conjectured"]),
            f"{pct(bf['f']['1 h']['conjectured'])} ({pct(bf['f_decode']['1 h']['conjectured'])})"]))
    lines += ["", "**Data products a token, KV at write** (products of two run-dependent values, by kind; charged: those "
              "property/v1 left free as s8 lanes, now a unit each; the rest its fp terms already charge as roundings).", ""]
    lines += header_md(["model", "phase", *(n for n, _ in PRODUCT_GROUPS), "all", "charged (new)", "share of the nonlinear units"])
    for name, r in t["models"].items():
        for phase in PHASES:
            x = r["at write"][phase]
            pg, u = product_groups(x["dd"]), x["units"]
            new = u["by_class"].get("products", 0)
            lines.append(row_md([name, phase, *(g(pg[n]) for n, _ in PRODUCT_GROUPS), g(sum(pg.values())), g(new),
                                 f"{float(new / u['total']):.1%}"]))
    lines.append("")
    for kv in KV_MODES:
        lines += [f"**KV {kv}.**", ""]
        lines += header_md(["model", "dynamic memory", "B/token", "state/request", "M_dyn", "batch", "step", "R_cap (phase)",
                            "units/B/step (decode)", "units/key-byte", "T_turn", "units a byte written (R_cap)",
                            *(f"f {lt}, E {e}" for lt in LIFETIMES for e in E_WORD), "f 1 h, E conjectured, R of decode alone",
                            "f 10 min / 1 h, E conjectured, softmax at 1 unit a head-key",
                            "f 10 min / 1 h, E conjectured / best found, softmax and requantization at their floors",
                            "before: f 10 min / 1 h, E conjectured"])
        for name, r in t["models"].items():
            x = r[kv]
            sv, dyn = x["serving"], x["decode"]["dyn"]
            kinds_ = "; ".join(f"{d['kind']} ({d['layers']} layers)" for d in dyn)
            pk = x["per_key"]["units_per_byte"]
            lines.append(row_md([
                name, kinds_, human(sv["request_bytes"]["per_token"]), human(sv["request_bytes"]["state"]), human(sv["M_dyn"]),
                sv["B"], f"{sv['t_step_s'] * 1e3:.3g} ms ({sv['bound']})", f"{g(sv['R_cap'])} ({sv['R_cap_phase']})",
                g(sv["u_B_decode"]), g(pk) if pk is not None else "-", f"{g(sv['T_turn_s'])} s", g(sv["W_byte"]),
                *(pct(sv["f"][lt][e]) for lt in LIFETIMES for e in E_WORD), pct(sv["f_decode"]["1 h"]["conjectured"]),
                " / ".join(pct(x["softmax_floor"]["f"][lt]["conjectured"]) for lt in LIFETIMES),
                " / ".join(pct(x["exact_floor"]["f"][lt][e]) for lt in LIFETIMES for e in E_WORD),
                " / ".join(pct(x["before"]["f"][lt]["conjectured"]) for lt in LIFETIMES)]))
        lines.append("")
    lines += ["**Where the units go, KV at write** (share of a token's nonlinear units by role).", "",
              "| model | phase | " + " | ".join(name for name, _ in SHARES) + " |", "|---|---|" + "---|" * len(SHARES)]
    for name, r in t["models"].items():
        for phase in PHASES:
            u = r["at write"][phase]["units"]
            by = {role: sum(c.values(), Fraction(0)) for role, c in u["by_role"].items()}
            sh = shares(by)
            lines.append(f"| {name} | {phase} | " + " | ".join(f"{float(sh[n] / u['total']):.1%}" for n, _ in SHARES) + " |")
    lines.append("")
    lines += ["**Honest operations a token, KV at write** (decode at S, prefill at P; MACs and nonlinear units a token).", "",
              "| model | phase | const-weight MACs (free) | data x data products | nonlinear units | data products charged "
              "| roundings (requant) | softmax roundings | division | rsqrt | compare-select | of which top-k (MoE routing, DSA) "
              "| of which softmax max | gathers (exp table / index bits) | modeled share |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in t["models"].items():
        for phase in PHASES:
            x = r["at write"][phase]
            u, k = x["units"], x["kinds"]
            rd, cs, ga = k.get("roundings", {}), k.get("compare_select", {}), k.get("gathers", {})
            mod = float(u["by_source"].get("modeled", 0) / u["total"]) if u["total"] else 0
            lines.append(
                f"| {name} | {phase} | {g(x['macs'])} | {g(sum(x['dd'].values()))} | {g(u['total'])} | "
                f"{g(u['by_class'].get('products', 0))} | "
                f"{g(u['by_class'].get('roundings', 0))} ({g(rd.get('requantization: round or convert', 0))}) | "
                f"{g(rd.get('softmax: max subtract, LUT index, low byte', 0))} | {g(rd.get('division', 0))} | {g(rd.get('rsqrt / sqrt', 0))} | {g(u['by_class'].get('compare_select', 0))} | "
                f"{g(cs.get('top-k (MoE routing, DSA)', 0))} | {g(cs.get('softmax max', 0))} | "
                f"{g(ga.get('exp table', 0))} / {g(ga.get('index bits', 0))} | {mod:.1%} |")
    lines += ["", "**Writes and cuts** (bits a token-layer; the narrowest cut in front of each write).", "",
              "| model | dynamic memory | layers | write path | residual | write inputs | stored at write | narrowest |",
              "|---|---|---|---|---|---|---|---|"]
    for name, r in t["models"].items():
        for d in r["at write"]["decode"]["dyn"]:
            cb = d["cut_bits"]
            lines.append(f"| {name} | {d['kind']} | {d['layers']} | {d['write_path']} | {cb['residual']:,} | "
                         f"{cb['write inputs']:,} | {cb['stored, at write']:,} | {min(cb.values()):,} |")
    sw = t["sweep"]
    ctx = t["workload"]["sweep"]
    lines += ["", f"**f as the pool's context S varies** (KV at write, L = 10 min, E conjectured, prefills of min(P, S); a model "
              "past its max_context left out). Each cell is f with R_cap / with the decode rate alone, at full cost, the softmax "
              "floor and the exactness floor.", "", "| model | " + " | ".join(f"S = {s}" for s in ctx) + " |",
              "|---|" + "---|" * len(ctx)]
    for name, r in sw.items():
        cells = []
        for s in ctx:
            x = r.get(s)
            cells.append("-" if x is None else " / ".join(pct(x[c][f]["10 min"]["conjectured"]) for c, f in
                                                          (("full", "f"), ("full", "f_decode"), ("softmax_floor", "f_decode"),
                                                           ("exact_floor", "f_decode"))))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", type=Path, default=HERE / "frontier_models.json")
    ap.add_argument("--atoms", type=Path)
    ap.add_argument("--atoms-needed", action="store_true", help="print the atom keys atoms.py counts and stop")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    spec = json.loads(a.models.read_text())
    S, P = spec["workload"]["context"], spec["workload"]["prompt"]
    if a.atoms_needed:
        print(json.dumps(needed(spec["models"], S, P), indent=1))
        return 0
    doc = json.loads(a.atoms.read_text())
    t = table(spec, doc.get("atoms", doc))
    md = markdown(t)
    if a.out:
        a.out.write_text(json.dumps(jsonable(t), indent=1))
        a.out.with_suffix(".md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
