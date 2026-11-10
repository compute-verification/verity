"""Composites of the speculative-decoding (two-model draft/target) typed serving program.

The B1 composites (`b1.py`) are reused unchanged for both models' scans; this module adds the *acceptance*
and the *state transition* of one speculative step as represented computations, and two programs:

* `Accept_v1{K}`: greedy verification of `K` draft candidates against the target's `K+1` argmaxes — per-slot
  verdicts (prefix of agreements), the accepted length, and the `K+1` appended-token slots with dead slots
  holding `-1`.  Liveness of slot `j` is *derived* from the verdicts (`live[0] = 1`, `live[j] = verdict[j-1]`):
  the bounded-slot answer to "a variable number of live positions" in a static-shape circuit.
* `Advance_v1{K}`: the per-request state transition — next committed length, per-slot KV liveness (which of
  the `K+1` slot rows are committed, which discarded), and their counts.
* `VerifyBranch_v1{C, LP, K}`: the target model's half of one step as a standalone program: prefill of the
  *branch* sequence (committed tape + the `K` candidates), `Final` at each of the `K+1` verification slots
  (rejected slots included), then `Accept`.  This is what correspondence runs on the pod: each recorded step
  is one specialisation, the branch sequence is its prompt.
* `ServeSpec_v1{C, CD, LP, K, A}`: the whole two-model program under a *static schedule* `A` (the accepted
  length of every decode step).  The dynamic accepted length is a value (`Accept`'s output); the shapes it
  would have to drive (how many slot rows are committed, where the next block starts) are fixed by `A`, and
  the program's `accepted` outputs are checked against `A` (an equality obligation of the record).  A program
  whose *shapes* depend on `Accept`'s value is not expressible as a circuit — that is the finding, not a
  construct built here.

Naming: static parameters in CAPITALS; `Coll` arguments in lower case; composite names `Name_vN`.
"""

from __future__ import annotations

from verity.ml.circuits.defs import Coll, bind, composite
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import (Attention, DotBf16, Embedding, Final, LayerPost, LayerPre, ZERO16, ZERO32, _lane, scale_log2_bits, const, kv_type,
                 weights_type)
from verity.ml.definitions.registry.prims import A16, BF16, F32, I32

# Qwen/Qwen2.5-0.5B as the draft model: the same registered functions as B1 with a different static configuration.
DRAFT_CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B",
    "revision": None,
    "H": 896, "NL": 24, "NH": 14, "KVH": 2, "D": 64, "I": 4864, "V": 151936,
    "QKV": 1152, "GU": 9728, "EPS": 1e-6, "ROPE_THETA": 1_000_000.0, "MAX_POS": 32768, "BIAS": True,
    "tie_word_embeddings": True,
    # FA2's head-dim-64 splitkv trait visits 128-key blocks (kBlockN = 128; hdim 128: 64).  Measured (an A100, sd-count draft
    # steps 2-6): with 64-key blocks every draft Attention at a position with more than 64
    # visible keys mismatches in a few words, every one with <= 64 keys is exact.
    "FA2_BN": 128,
    "profile": {"attention": "FA2 flash_fwd_splitkv_kernel<64,64,128,4> (head dim 64: 128-key blocks; trait inferred from the t>=64 mismatch boundary)",
                "linear": "vllm batch_invariant matmul_kernel_persistent (Triton), bias as a separate aten.add",
                "rmsnorm": "layers.0.input_layernorm: batch_invariant Triton _rms_norm_kernel; the other norms: CUDA _C.fused_add_rms_norm",
                "rope": "vllm _C.rotary_embedding (NeoX), bf16 cos/sin cache",
                "act": "vllm _C.silu_and_mul", "sampler": "greedy argmax over bf16 logits (draft proposals)"},
}

ONE32 = const(32, 1)
ZERO32 = const(32, 0)
MINUS_ONE32 = const(32, 0xFFFFFFFF)


# ---------------------------------------------------------------------------------------------------------
# acceptance and state transition
# ---------------------------------------------------------------------------------------------------------


@composite("Accept", 1, ["K"],
           lambda S: ((("cand", Array(S.K, I32)), ("argmax", Array(S.K + 1, I32))),
                      Tuple(Array(S.K, I32), I32, Array(S.K + 1, I32))),
           conformance="exact-model-tested")
def Accept(B, S, cand, argmax):
    """Greedy speculative verification (vLLM `RejectionSampler`, temperature 0): candidate `p` is accepted
    iff every earlier candidate was accepted and `cand[p] == argmax[p]`.  Returns
    (verdict[K] in {0,1}, accepted = sum(verdict), appended[K+1]) where `appended[j]` is the target's
    argmax at slot `j` for the live slots `j <= accepted` and `-1` for the dead slots: the bonus token is
    `argmax[accepted]`, the accepted candidates equal `argmax[p]` by construction."""
    K = S.K
    one, zero, minus_one = B.call(ONE32), B.call(ZERO32), B.call(MINUS_ONE32)
    verdict: list[Coll] = []
    prev = one  # "all earlier candidates accepted", as an i32 0/1
    for p in range(K):
        eq = B.call(P.I32Eq, cand[p], argmax[p])
        v = B.call(P.SelectI32, eq, prev, zero)
        verdict.append(v)
        prev = v
    acc = verdict[0]
    for p in range(1, K):
        acc = B.call(P.I32Add, acc, verdict[p])
    # slot 0 is always live; every appended slot is a fresh gate (not an alias of an argmax gate) so that the
    # `appended` leaf and the per-slot `sampler.select` leaves have disjoint value identities
    live0 = B.call(P.I32Eq, one, one)
    appended: list[Coll] = [B.call(P.SelectI32, live0, argmax[0], minus_one)]
    for j in range(1, K + 1):
        live = B.call(P.I32Eq, verdict[j - 1], one)
        appended.append(B.call(P.SelectI32, live, argmax[j], minus_one))
    return tuple_of(array_of(verdict), acc, array_of(appended))


@composite("Advance", 1, ["K"],
           lambda S: ((("pos", I32), ("verdict", Array(S.K, I32))), Tuple(I32, Array(S.K + 1, I32), I32, I32)),
           conformance="exact-model-tested")
def Advance(B, S, pos, verdict):
    """Per-request state transition after `Accept`: (next committed length = pos + accepted + 1,
    kv_live[K+1] — slot `j`'s KV row is committed iff `j == 0` or `verdict[j-1]` — , committed count,
    discarded count).  The committed rows are the live slots' rows; the rest are freed (vLLM: the
    `num_rejected` trailing slots of the request's block table are released)."""
    K = S.K
    one, zero = B.call(ONE32), B.call(ZERO32)
    live = [B.call(P.SelectI32, B.call(P.I32Eq, one, one), one, zero)]  # slot 0: always committed (fresh gate)
    live += [B.call(P.SelectI32, B.call(P.I32Eq, verdict[j - 1], one), one, zero) for j in range(1, K + 1)]
    committed = live[0]
    for j in range(1, K + 1):
        committed = B.call(P.I32Add, committed, live[j])
    discarded = zero
    for j in range(1, K + 1):
        dead = B.call(P.I32Eq, live[j], zero)
        discarded = B.call(P.I32Add, discarded, B.call(P.SelectI32, dead, one, zero))
    nxt = B.call(P.I32Add, pos, committed)
    return tuple_of(nxt, array_of(live), committed, discarded)


# ---------------------------------------------------------------------------------------------------------
# FA2 attention with a static key-block size (b1's `AttnBlock`/`AttentionHead`/`Attention` fix BN = 64, the
# head-dim-128 trait; the draft's head dim 64 visits 128-key blocks).  Same gate structure otherwise.
# ---------------------------------------------------------------------------------------------------------


@composite("AttnBlockN", 1, ["D", "BN", "NVIS", "FIRST"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="model-untested")
def AttnBlockN(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`b1.AttnBlock` with the key-block size `BN` static (64: identical to b1's; 128: FA2's head-dim-64 trait)."""
    D, BN, NVIS, FIRST = S.D, S.BN, S.NVIS, S.FIRST
    dot = bind(DotBf16, K=D)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    scale_log2 = B.call(const(32, scale_log2_bits(D)))
    if not FIRST:
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]
    lanes: list[Coll] = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else B.call(ZERO32)
            rest = cols[1:]
        else:
            first = p[cols[0]] if cols else B.call(ZERO32)
            acc = B.call(P.F32FmaFtz, scores_scale, l_prev[ln], first)
            rest = cols[1:]
        for c in rest:
            acc = B.call(P.F32AddFtz, acc, p[c])
        lanes.append(acc)
    pb = [B.call(P.F2fpBf16, x) for x in p]
    zero16 = B.call(ZERO16) if NVIS < BN else None
    o = [B.call(ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        prow = _Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, A16), vt_refs)
        o = list(B.batch(P.AmpereBF16TcDot16, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHeadN", 1, ["T", "D", "BN"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="model-untested")
def AttentionHeadN(B, S, q, k, v):
    """`b1.AttentionHead` over `BN`-key blocks (visited last -> first)."""
    T, D, BN = S.T, S.D, S.BN
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:
            if zero_row is None:
                zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = _Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = _Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = b == nb - 1
        blk = bind(AttnBlockN, D=D, BN=BN, NVIS=nvis, FIRST=first)
        if first:
            z = B.call(ZERO32)
            st = B.call(blk, q, kb, vb, z, _Coll(Array(4, F32), concat([z.refs] * 4)), _Coll(Array(D, F32), concat([z.refs] * D)))
        else:
            st = B.call(blk, q, kb, vb, m, l, o)
        m, l, o = st[0], st[1], st[2]
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, l[0], l[2]), B.call(P.F32AddFtz, l[1], l[3]))
    inv = B.call(P.Fa2InvSum, total)
    y = B.batch(P.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite("AttentionN", 1, ["T", "NH", "KVH", "D", "BN"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="model-untested")
def AttentionN(B, S, q, kc, vc):
    """`b1.Attention` (GQA over `[T][KVH][D]` caches) with the key-block size static."""
    T, NH, KVH, D, BN = S.T, S.NH, S.KVH, S.D, S.BN
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(bind(AttentionHeadN, T=T, D=D, BN=BN), q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


def _attention(C: dict, T: int):
    """b1's `Attention` (64-key blocks) unless the configuration declares another FA2 block size."""
    bn = C.get("FA2_BN", 64)
    if bn == 64:
        return bind(Attention, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"])
    return bind(AttentionN, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], BN=bn)


# ---------------------------------------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------------------------------------


def _fwd(B, C: dict, toks: Coll, weights: Coll, T0: int, n_pos: int, caches, prefix: str = ""):
    """`b1._forward_positions` with a name prefix, so the target's and the draft's scans can share one scope, and the
    attention composite chosen by the configuration's FA2 block size."""
    NL, H = C["NL"], C["H"]
    hs = [B.call(bind(Embedding, V=C["V"], H=H), toks[i], weights["embed"], name=f"{prefix}t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [B.call(bind(LayerPre, C=C, L0=(L == 0)), hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][T0 + i],
                      name=f"{prefix}t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = _Coll(kv_type(C, n_pos), concat([p[2].refs for p in pre]))
        v_rows = _Coll(kv_type(C, n_pos), concat([p[3].refs for p in pre]))
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][0].refs, k_rows.refs]))
            vc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][1].refs, v_rows.refs]))
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = B.call(_attention(C, T), pre[i][1], kc[0:T], vc[0:T], name=f"{prefix}t={T - 1}/layers.{L}/attn")
            post = B.call(bind(LayerPost, C=C), attn, pre[i][0], w, name=f"{prefix}t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post[0], post[1]
    return new_kv, list(zip(hs, res))


def _grow(C: dict, caches, new_kv, T0: int, n: int):
    """Append the first `n` of the new slot rows (the committed ones) to every layer's cache."""
    return [(_Coll(kv_type(C, T0 + n), concat([caches[L][0].refs, new_kv[L][0][0:n].refs])),
             _Coll(kv_type(C, T0 + n), concat([caches[L][1].refs, new_kv[L][1][0:n].refs]))) for L in range(C["NL"])]


@composite("VerifyBranch", 1, ["C", "LP", "K"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C))),
                      Tuple(Array(S.K + 1, I32), Array(S.K, I32), I32, Array(S.K + 1, I32))),
           conformance="model-untested")
def VerifyBranch(B, S, prompt, weights):
    """The target model's half of one speculative step, as a prefill of the branch sequence: `prompt` is
    the committed tape up to and including the block's first input token followed by the `K` candidates
    (`prompt[LP-K:]`).  `Final` (norm, LM head, greedy argmax) at each of the `K+1` verification slots —
    positions LP-1-K .. LP-1, rejected ones included — then `Accept`.  Returns (argmax[K+1], verdict[K],
    accepted, appended[K+1])."""
    C, LP, K = S.C, S.LP, S.K
    _, finals = _fwd(B, C, prompt, weights, 0, LP, None)
    arg = [B.call(bind(Final, C=C), finals[t][0], finals[t][1], weights["norm"], weights["embed"], name=f"t={t}/final")
           for t in range(LP - 1 - K, LP)]
    argmax = array_of(arg)
    acc = B.call(bind(Accept, K=K), prompt[LP - K:LP], argmax, name="accept")
    return tuple_of(argmax, acc[0], acc[1], acc[2])


@composite("ServeDraft", 1, ["C", "LP", "STEPS"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C))), Array(S.STEPS + 1, I32)),
           conformance="model-untested")
def ServeDraft(B, S, prompt, weights):
    """`b1.Serve` (greedy prefill of LP tokens then STEPS decode steps) built on `_fwd`, so the configuration's FA2
    block size applies: the draft's half of one speculative step as a standalone program (prompt = the tape up to
    its bonus row, STEPS = K-1 generated rows; returns the K candidates)."""
    C, LP, STEPS = S.C, S.LP, S.STEPS
    caches, finals = _fwd(B, C, prompt, weights, 0, LP, None)
    tok = B.call(bind(Final, C=C), finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = _fwd(B, C, _Coll(Array(1, I32), tok.refs), weights, T0, 1, caches)
        caches = _grow(C, caches, new_kv, T0, 1)
        tok = B.call(bind(Final, C=C), fin[0][0], fin[0][1], weights["norm"], weights["embed"], name=f"t={T0}/final")
        toks.append(tok)
    return _Coll(Array(STEPS + 1, I32), concat([t.refs for t in toks]))


@composite("ServeSpec", 1, ["C", "CD", "LP", "K", "A"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C)), ("draft_weights", weights_type(S.CD))),
                      Tuple(Array(len(S.A), Tuple(I32, Array(S.K + 1, I32))), Array(S.K, I32))),
           conformance="model-untested")
def ServeSpec(B, S, prompt, weights, draft_weights):
    """Two-model greedy speculative serving of one request under the static schedule `A` (tuple: the accepted
    length of decode step s, 0 <= A[s] <= K).  Step 0: target prefill (LP rows, `Final` at LP-1), draft
    prefill of the prompt plus that token (LP+1 rows) and K-1 further draft rows -> K candidates.  Decode
    step s: the target runs the K+1 slot rows (first input token, then the candidates) attending to its
    committed cache, `Final` at every slot, `Accept`; then `Advance`; the first A[s]+1 slot rows are
    committed, the rest discarded (static: chosen by A[s]); the draft re-runs the A[s]+1 appended tokens as
    committed rows plus the bonus row and K-1 generated rows -> the next K candidates.  Returns, per decode
    step, (accepted, appended[K+1]) and the final K candidates.  `accepted` must equal A[s]: this equality is
    an obligation of any record built on this program, not something the program can enforce."""
    C, CD, LP, K, A = S.C, S.CD, S.LP, S.K, tuple(S.A)
    for a in A:
        assert 0 <= a <= K, (a, K)
    empty = [(_Coll(kv_type(CD, 0), concat([])), _Coll(kv_type(CD, 0), concat([])))] * CD["NL"]
    # step 0: target prefill; draft prefill of prompt + first token (LP+1 rows: LP committed, the bonus row not)
    t_caches, fin = _fwd(B, C, prompt, weights, 0, LP, None, prefix="target/")
    tok0 = B.call(bind(Final, C=C), fin[-1][0], fin[-1][1], weights["norm"], weights["embed"], name=f"target/t={LP - 1}/final")
    d_toks = _Coll(Array(LP + 1, I32), concat([prompt.refs, tok0.refs]))
    d_new, dfin = _fwd(B, CD, d_toks, draft_weights, 0, LP + 1, None, prefix="draft/step=0/")
    d_caches = _grow(CD, empty, d_new, 0, LP)
    cands = _draft_generate(B, CD, draft_weights, dfin[-1], _grow(CD, empty, d_new, 0, LP + 1), LP, K, prefix="draft/step=0/")
    committed, first = LP, tok0
    outs: list[Coll] = []
    for s, a in enumerate(A):
        T0 = committed
        pre = f"target/step={s + 1}/"
        # target: the K+1 slot rows (first input token, then the candidates) against the committed cache
        slot_toks = _Coll(Array(K + 1, I32), concat([first.refs, cands.refs]))
        new_kv, fin = _fwd(B, C, slot_toks, weights, T0, K + 1, t_caches, prefix=pre)
        argmax = array_of([B.call(bind(Final, C=C), fin[j][0], fin[j][1], weights["norm"], weights["embed"], name=f"{pre}t={T0 + j}/final")
                           for j in range(K + 1)])
        acc = B.call(bind(Accept, K=K), cands, argmax, name=f"{pre}accept")
        B.call(bind(Advance, K=K), B.call(const(32, T0)), acc[0], name=f"{pre}advance")  # values of the record; the shapes follow A[s]
        t_caches = _grow(C, t_caches, new_kv, T0, a + 1)  # slots 0..a committed, a+1..K discarded
        appended = acc[2]
        outs.append(tuple_of(acc[1], appended))
        # draft: rows T0..T0+a re-run the committed tokens (first, appended[0..a-1]) — vLLM's draft-model proposer
        # recomputes them rather than keeping its own speculative rows — then the bonus row T0+a+1 (appended[a])
        d_in = _Coll(Array(a + 2, I32), concat([first.refs, appended[0:a + 1].refs]))
        dpre = f"draft/step={s + 1}/"
        d_new, dfin = _fwd(B, CD, d_in, draft_weights, T0, a + 2, d_caches, prefix=dpre)
        gen_caches = _grow(CD, d_caches, d_new, T0, a + 2)
        d_caches = _grow(CD, d_caches, d_new, T0, a + 1)
        cands = _draft_generate(B, CD, draft_weights, dfin[-1], gen_caches, T0 + a + 1, K, prefix=dpre)
        committed, first = T0 + a + 1, appended[a]
    return tuple_of(array_of(outs), cands)


def _draft_generate(B, CD: dict, dw: Coll, bonus_fin, caches, T_bonus: int, K: int, prefix: str) -> Coll:
    """From the draft's bonus row (its final (h, res) at position `T_bonus`; `caches` holds every row up to and
    including it): candidate 0 = Final(bonus); K-1 generated rows at T_bonus+1.. each fed the previous
    candidate -> candidates 1..K-1.  The generated rows' (and the bonus row's) KV is not kept: the draft
    recomputes the accepted tokens as committed rows next step."""
    c0 = B.call(bind(Final, C=CD), bonus_fin[0], bonus_fin[1], dw["norm"], dw["embed"], name=f"{prefix}t={T_bonus}/final")
    cands = [c0]
    tok = c0
    for p in range(1, K):
        T0 = T_bonus + p
        new_kv, fin = _fwd(B, CD, _Coll(Array(1, I32), tok.refs), dw, T0, 1, caches, prefix=prefix)
        caches = _grow(CD, caches, new_kv, T0, 1)
        tok = B.call(bind(Final, C=CD), fin[0][0], fin[0][1], dw["norm"], dw["embed"], name=f"{prefix}t={T0}/final")
        cands.append(tok)
    return array_of(cands)


ALL_COMPOSITES = [Accept, Advance, AttnBlockN, AttentionHeadN, AttentionN, VerifyBranch, ServeDraft, ServeSpec]
