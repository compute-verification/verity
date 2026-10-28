"""C2 of a served-compiled row: the served Build's Program (`acquire.served`) evaluated on the committed values, word for word.

Every value the Build names is committed at every step it runs, and each is checked from the committed values its producer
reads and the served weights, by the twin of the Definition the Build bound (`kernels.inductor_rows`, `bf16_gemm_check`,
`rows.token_select_first_max`):

  * a call of a piece (`program.calls`): the Embedding, Inductor's norms, SiLU-and-mul, RoPE and q/k norm-RoPE on every row; a GEMM (``mm``) is
    the Definition the step's plan names at (K, N) for the step's M, on `ROWS` sampled rows of each request;
  * a fused-MoE block's Calls (`acquire.served_moe`): its gate a GEMM as above, its router (`moe_router_rows`) and expert GEMMs (the
    Definition the step's plan names, by the k16 chain of its step, each (token, slot) pair against its routed expert's weight) on
    the sampled rows, its SiLU-and-mul and slot sum on every row;
  * an attention op: the device's attention kernel over the request's cached keys and values, the committed k and v rows of every
    step it ran so far, by the Definition the plan names for each sampled query row (FA3's `Attention_v4` or `Fa3AttentionSplit_v1`,
    FA2's `Attention_v5` or `Fa2AttentionSplit_v1`, with the split count);
  * the logits: the plan's GEMM on each request's last row of the backbone's output; the sampled id: `TokenSelect_v1` on the
    committed logits;
  * the inputs: each request's positions are its token positions (klen - tokens .. klen - 1), and a decode step's token id is the
    id the request sampled at its previous step (the step's input is the previous step's output).

The step's batch must be the plan's (its requests with their token counts), its committed names exactly the Build's, each value
of its width.  A check that cannot evaluate a value (no twin for its Definition, an operand missing) fails by name; an attention
row or an expert pair whose dot read a non-finite operand is never claimed exact.

With a seed (the Commit's, sampled after the values are committed) a request's sampled rows at each step are a uniform draw of
`ROWS` from that seed's generator, one sample a request a step shared by its GEMMs and its attention ops, and the report carries
the seed and every sample; without one they are the fixed spread `sample`.

`population` reconciles the report against the Build's identities one by one (what the check reached, what it noted outside
them, what it never reached), the record's C2 population scope.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from verity.ml.circuits.ops.prims import AmpereBF16TcDot16
from verity_vllm.commit import challenge as CH
from verity.ml.definitions.numerics import dense_rows as DRW
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import fa2_split_twins as FT
from verity.ml.definitions.numerics import inductor_rows as TW
from verity.ml.definitions.numerics.act_rows import silu_mul_expf_rows
from verity.ml.definitions.numerics.moe_router_rows import moe_router_topk
from verity.ml.definitions.numerics.rows import token_select_first_max
from verity.ml.definitions.registry import act, b1, cublas_bf16, cublas_gemv
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity.ml.definitions.registry import fa2_check_inf as F2
from verity.ml.definitions.registry import fa2_split as W2
from verity.ml.definitions.registry import fa3_check_inf as F3
from verity.ml.definitions.registry import inductor as IR
from verity.ml.definitions.registry import moe as MOE
from verity_vllm.query.manifest.bind import split_name
from verity_vllm.query.manifest.format import SERVED_FAMILY, SERVED_IDS as IDS, SERVED_LOGITS as LOGITS
from verity_vllm.query.manifest.format import SERVED_POSITIONS as POSITIONS, SERVED_SAMPLED as SAMPLED

#: the rows of a request a GEMM or an attention op is evaluated on: all of a request's rows up to this many, else its first, its
#: last and evenly spread ones
ROWS = 16


@dataclass(frozen=True)
class Step:
    """One engine step as committed: `rows` the batch, (request_id, first row, end row) in batch order; `values` the committed
    value of each name (a bf16 value's words as uint16, token rows first; the ids and positions one integer a token; the logits
    and the sampled ids one row a request, in batch order)."""
    step: int
    rows: list[tuple[str, int, int]]
    values: Mapping[str, np.ndarray]


def sample(n: int, k: int = ROWS) -> np.ndarray:
    return np.arange(n) if n <= k else np.unique(np.r_[0, n - 1, np.linspace(0, n - 1, k - 2).round().astype(int)])


def _norm(rs: Callable[..., np.ndarray]) -> Callable[..., list[np.ndarray]]:
    return lambda S, *xs: [TW.norm_rows(rs(*xs[:-1]), xs[-1], **S)]


def _add2(S, x, r1, r2, w):
    r = TW.residual_rows(x, r1, r2)
    return [DR.f2fp_bf16(r), TW.norm_rows(r, w, **S)]


def _moe_sum(S, y):
    """`MoeSum_v1{TOPK,H}` on rows of TOPK x H bf16 words: the slots added in ascending order from +0 in f32, one RNE at the end."""
    y = y.reshape(len(y), S["TOPK"], S["H"])
    acc = np.zeros((len(y), S["H"]), np.uint32)
    for k in range(S["TOPK"]):
        acc = DR.add_rn(acc, DR.bf16_to_f32_words(y[:, k]))
    return [DR.f2fp_bf16(acc).astype(np.uint16)]


#: each call Definition's twin by family: twin(statics, *operands) -> the values it writes, rows first
TWINS: dict[str, Callable[..., list[np.ndarray]]] = {
    b1.Embedding.id: lambda S, ids, table: [table[ids.astype(np.int64)]],
    IR.InductorRMSNorm.id: _norm(TW.residual_rows),
    IR.InductorAddRMSNorm.id: _norm(TW.residual_rows),
    IR.InductorAdd2RMSNorm.id: _add2,
    IR.InductorAdd2RMSNormFinal.id: _norm(TW.residual_rows),
    IR.InductorSiluMul.id: lambda S, gu: [TW.silu_mul_rows(gu)],
    IR.InductorRoPE.id: lambda S, x, cs: [TW.rope_rows(x.reshape(len(x), S["NHEADS"], S["D"]), cs).reshape(len(x), -1)],
    IR.InductorQKNormRoPE.id: lambda S, x, w, cs: [TW.qk_norm_rope_rows(x.reshape(len(x), S["NHEADS"], S["D"]), w, cs, **S)
                                                   .reshape(len(x), -1)],
    act.SiluMul.id: lambda S, gu: [silu_mul_expf_rows(gu.reshape(-1, 2 * S["I"]))["out"].reshape(len(gu), -1)],
    MOE.MoeSum.id: _moe_sum,
}
#: each fused-MoE router Definition's `renormalize` (its twin `moe_router_topk`, one row at a time: on the sampled rows)
ROUTERS: dict[str, bool] = {MOE.MoeRouterTopK.id: False, MOE.MoeRouterTopKNorm.id: True}
#: each expert GEMM Definition's MUL_ROUTED_WEIGHT (the w2 launch's: its routed weight into the f32 accumulator)
EXPERTS: dict[str, bool] = {MOE.MoeExpertGemm.id: False, MOE.MoeExpertGemmV2.id: False, MOE.MoeExpertGemmW.id: True,
                            MOE.MoeExpertGemmWV2.id: True}
#: each GEMM Definition's twin by family: twin(statics, x (M, K), w (N, K)) -> (M, N)
GEMMS: dict[str, Callable[..., np.ndarray]] = {
    b1.GemmV2.id: lambda S, x, w: BC.linear_rows(x, w),
    cublas_bf16.CublasSplitKGemmWord.id: lambda S, x, w: BC.cublas_splitk_rows(x, w, S["SPLIT_K"], out="bf16"),
    cublas_bf16.CublasSplitKGemmBf16PartialsWord.id: lambda S, x, w: BC.cublas_splitk_rows(x, w, S["SPLIT_K"], partial="bf16", out="bf16"),
    cublas_bf16.CublasSerialSplitKGemmWord.id: lambda S, x, w: BC.cublas_serial_splitk_rows(x, w, S["SPLIT_K"]),
    cublas_gemv.CublasGemvWord.id: lambda S, x, w: BC.cublas_gemv_rows(x, w, S["LANES"]),
}
#: each attention row Definition's twin (q, kc, vc, NS, BN, masked_from, dot_prim, inv_prim) -> (out, nonfinite) and split count
ATTENTION: dict[str, tuple[Callable[..., tuple[np.ndarray, np.ndarray]], Callable[[dict[str, Any]], int]]] = {
    F3.AttentionV4.id: (TW.fa3_attention_rows, lambda S: 1), IR.Fa3AttentionSplit.id: (TW.fa3_attention_rows, lambda S: S["NS"]),
    F2.AttentionV5.id: (FT.attention_rows, lambda S: 1), W2.Fa2AttentionSplit.id: (FT.attention_rows, lambda S: S["NS"])}


def _verdict(want: np.ndarray, got: np.ndarray, **info: Any) -> dict[str, Any]:
    want, got = np.asarray(want), np.asarray(got)
    if want.shape != got.shape:
        return {"ok": False, "why": f"evaluated {list(want.shape)}, committed {list(got.shape)}", **info}
    bad = int(np.count_nonzero(want != got))
    return {"ok": bad == 0, "bad": bad, "n": int(want.size), **info}


class _Unresolved(Exception):
    pass


class ServedCheck:
    """The check over one run's steps, in engine-step order (each attention op's cache is the committed k and v of every step
    before)."""

    def __init__(self, build: dict[str, Any], weight: Callable[[str], np.ndarray], *, seed: int | None, rows: int = ROWS,
                 prompts: Mapping[str, list[int]] | None = None, log: Callable[[str], None] | None = None) -> None:
        self.build, self.weight, self.k, self.prompts, self.log = build, weight, rows, prompts or {}, log
        self.seed, self.rng = seed, (None if seed is None else CH.generator(seed))
        self.picks: dict[int, dict[str, list[int]]] = {}
        self.program = build["program"]
        self.plans = {p["step"]: p for p in build["steps_plan"]}
        self.names = set(self.program["values"]) | {LOGITS, SAMPLED}
        self.width = {n: v["width"] for n, v in self.program["values"].items()}
        self.cache: dict[tuple[str, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
        self.sampled: dict[str, int] = {}
        self.seen: set[int] = set()
        self.tally: dict[str, list[int]] = {}
        self.steps: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------------------------------------------------- one step
    def step(self, st: Step) -> dict[str, Any]:
        plan = self.plans.get(st.step)
        out: dict[str, Any] = {}
        if self.steps and st.step <= max(self.steps):
            raise ValueError(f"engine step {st.step} after step {max(self.steps)}: the steps are checked in order")
        self.steps[st.step] = out
        if plan is None:
            return self._note(out, "batch", {"ok": False, "why": f"engine step {st.step} is in no step plan of the Build"})
        n = sum(hi - lo for _r, lo, hi in st.rows)
        batch = {r: hi - lo for r, lo, hi in st.rows}
        want = {r["request_id"]: r["tokens"] for r in plan["requests"]}
        if batch != want or n != plan["tokens"]:
            return self._note(out, "batch", {"ok": False, "why": f"batch {batch}, the plan's {want}", "rows": batch})
        self._note(out, "batch", {"ok": True, "rows": batch})
        names = set(st.values)
        V: dict[str, np.ndarray] = {}
        bad_shape = []
        for name in sorted(self.names & names):
            a = np.asarray(st.values[name])
            rows = len(st.rows) if name in (LOGITS, SAMPLED) else n
            V[name] = a.reshape(rows, -1) if a.size else a.reshape(rows, 0)
            if name in self.width and V[name].shape[1] != self.width[name]:
                bad_shape.append(f"{name} {list(a.shape)}")
        self._note(out, "coverage", {"ok": names == self.names and not bad_shape, "missing": sorted(self.names - names)[:8],
                                     "extra": sorted(names - self.names)[:8], "width": bad_shape[:8]})
        req = {r["request_id"]: r for r in plan["requests"]}
        spans = [(r, lo, hi) for r, lo, hi in st.rows]
        every = np.arange(n)
        local = {r: self._draw(hi - lo) for r, lo, hi in spans}
        self.picks[st.step] = {r: [int(i) for i in v] for r, v in local.items()}
        some = np.concatenate([lo + local[r] for r, lo, _hi in spans]) if spans else every
        self._inputs(out, V, spans, req)
        for c in self.program["calls"]:
            name = c["writes"][0]
            try:
                if c["op"] == "mm":
                    d = plan["gemms"][f"{c['gemm'][0]},{c['gemm'][1]}"]
                    x, w = (self._operand(r, V, some) for r in c["reads"])
                    v = _verdict(GEMMS[d["base"]](d["statics"], x, w), V[name][some], definition=d["definition"], rows=len(some))
                    self._note(out, name, v, d["base"])
                    continue
                if "moe" in c:
                    self._experts(out, c, plan, V, some)
                    continue
                if c["base"] in ROUTERS:
                    self._router(out, c, V, some)
                    continue
                twin = TWINS.get(c["base"])
                if twin is None:
                    raise _Unresolved(f"no twin of {c['definition']}")
                got = twin(c["statics"], *(self._operand(r, V, every) for r in c["reads"]))
                for w_, g in zip(c["writes"], got):
                    self._note(out, w_, _verdict(g, V[w_][every]), c["base"])
            except (_Unresolved, KeyError) as e:
                for w_ in c["writes"]:
                    self._note(out, w_, {"ok": False, "why": f"{type(e).__name__}: {e}"}, c.get("base") or c["op"])
        for a in self.program["attention"]:
            self._attention(out, a, V, spans, req, local)
        self._logits(out, plan, V, spans)
        self.seen.add(st.step)
        if self.log:
            bad = [k for k, v in out.items() if not v.get("ok")]
            self.log(f"step {st.step} {plan['mode']} n={n}: {len(out) - len(bad)}/{len(out)} exact" + (f", FAIL {bad[:6]}" if bad else ""))
        return out

    def _draw(self, n: int) -> np.ndarray:
        if self.rng is None or n <= self.k:
            return sample(n, self.k)
        return np.sort(self.rng.choice(n, self.k, replace=False))

    def _note(self, out: dict[str, Any], name: str, v: dict[str, Any], kind: str | None = None) -> dict[str, Any]:
        out[name] = v
        t = self.tally.setdefault(kind or name, [0, 0])
        t[0] += bool(v.get("ok"))
        t[1] += 1
        return out

    def _operand(self, ref: dict[str, Any], V: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
        if "value" in ref:
            if ref["value"] not in V:
                raise _Unresolved(f"{ref['value']} is not committed")
            a = V[ref["value"]][rows]
            if "cols" in ref:
                a = a[:, ref["cols"][0]:ref["cols"][1]]
            return a[:, 0] if ref.get("width") == 1 else a
        if "weight" in ref:
            w = self.weight(ref["weight"])
            if list(w.shape) != list(ref["shape"]):
                raise _Unresolved(f"{ref['weight']} {list(w.shape)}, the Program's {ref['shape']}")
            if "row_of" in ref:
                return w[self._operand(ref["row_of"], V, rows).astype(np.int64)]
            return w
        raise _Unresolved(f"operand {ref}")

    def _router(self, out: dict[str, Any], c: dict[str, Any], V: dict[str, np.ndarray], rows: np.ndarray) -> None:
        S = c["statics"]
        got = [moe_router_topk(r, E=S["E"], TOPK=S["TOPK"], VPT=S["VPT"], renormalize=ROUTERS[c["base"]])
               for r in self._operand(c["reads"][0], V, rows)]
        for k, w_ in enumerate(c["writes"]):
            want = np.array([g[k] for g in got], np.uint32).reshape(len(rows), S["TOPK"])
            self._note(out, w_, _verdict(want, V[w_][rows].view(np.uint32), definition=c["definition"], rows=len(rows)), c["base"])

    def _experts(self, out: dict[str, Any], c: dict[str, Any], plan: dict[str, Any], V: dict[str, np.ndarray], rows: np.ndarray) -> None:
        """A fused_moe_kernel launch on `rows`, the expert GEMM the step's plan names under the call's key: each (token, slot) pair's
        row (the token's hidden states for w13, its slot's `act` for w2) against its routed expert's weight, by the k16 chain of the
        Definition's step, the w2 launch's routed weight multiplied into the f32 accumulator before the one bf16 conversion."""
        d = plan["moe"][c["moe"]]
        if d["base"] not in EXPERTS:
            raise _Unresolved(f"no twin of {d['definition']}")
        E, K, N, routed = d["statics"]["E"], d["statics"]["K"], d["statics"]["N"], EXPERTS[d["base"]]
        name, wref = c["writes"][0], c["reads"][-1]
        x = self._operand(c["reads"][0], V, rows)
        ids = self._operand(c["reads"][1], V, rows).astype(np.int64).reshape(len(rows), -1)
        w = self.weight(wref["weight"])
        if list(w.shape) != list(wref["shape"]):
            raise _Unresolved(f"{wref['weight']} {list(w.shape)}, the Program's {wref['shape']}")
        e = ids.reshape(-1)
        if ((e < 0) | (e >= E)).any():
            raise _Unresolved(f"router ids {sorted(set(e[(e < 0) | (e >= E)].tolist()))[:4]} outside the {E} experts")
        X = x.reshape(len(rows), -1, K)
        X = np.repeat(X, ids.shape[1], axis=1).reshape(-1, K) if X.shape[1] == 1 else X.reshape(-1, K)
        g = self._operand(c["reads"][2], V, rows).reshape(-1).astype(np.uint32) if routed else None
        dot = d["statics"].get("DOT", AmpereBF16TcDot16.id)
        want, nonfin = np.empty((len(e), N), np.uint16), np.zeros(len(e), bool)
        for ex in np.unique(e):
            at = np.flatnonzero(e == ex)
            acc, nf = DRW.chain_rows(X[at], w[ex], dot)
            if g is not None:
                acc = DR.mul_rn(acc, np.repeat(g[at, None], N, axis=1))
            want[at], nonfin[at] = DR.f2fp_bf16(acc).astype(np.uint16), nf.any(axis=1)
        v = _verdict(want, V[name][rows].reshape(-1, N), definition=d["definition"], rows=len(rows), pairs=len(e))
        self._note(out, name, {**v, "ok": v["ok"] and not nonfin.any(), "nonfinite": int(nonfin.sum())}, d["base"])

    def _inputs(self, out: dict[str, Any], V: dict[str, np.ndarray], spans, req) -> None:
        """Positions are each request's token positions; a request's first step reads its prompt (when the check has the
        prompts), and a decode row's id is the id its request sampled at its previous step."""
        if POSITIONS in V:
            pos = V[POSITIONS][:, 0]
            want = np.concatenate([np.arange(req[r]["klen"] - (hi - lo), req[r]["klen"]) for r, lo, hi in spans])
            self._note(out, POSITIONS, _verdict(want, pos))
        if IDS in V:
            ids = V[IDS][:, 0].astype(np.int64)
            want, rows, why, checked = [], [], [], []
            for r, lo, hi in spans:
                if r in self.sampled:
                    if hi - lo != 1:
                        why.append(f"{r} runs {hi - lo} rows after its first step")
                    want.append(self.sampled[r])
                    rows.append(lo)
                    checked.append(r)
                elif r in self.prompts:
                    if len(self.prompts[r]) != hi - lo:
                        why.append(f"{r}: a first step of {hi - lo} rows, a prompt of {len(self.prompts[r])}")
                        continue
                    want += [int(t) for t in self.prompts[r]]
                    rows += list(range(lo, hi))
                    checked.append(r)
            v = _verdict(np.array(want, np.int64), ids[rows], rows=len(rows), requests=checked)
            self._note(out, IDS, {**v, "ok": v["ok"] and not why, **({"why": why[:4]} if why else {})})

    def _attention(self, out: dict[str, Any], a: dict[str, Any], V: dict[str, np.ndarray], spans, req,
                   local: Mapping[str, np.ndarray]) -> None:
        name, NH, KVH, D = a["name"], a["NH"], a["KVH"], a["D"]
        try:
            q = self._operand(a["q"], V, np.arange(sum(hi - lo for _r, lo, hi in spans)))
            k = self._operand(a["k"], V, np.arange(len(q)))
            v = self._operand(a["v"], V, np.arange(len(q)))
        except _Unresolved as e:
            self._note(out, name, {"ok": False, "why": str(e)}, "attention")
            return
        got = V[name]
        bad = rows_n = 0
        why, nonfinite, defs = [], 0, {}
        for r, lo, hi in spans:
            kh, vh = self.cache.setdefault((r, a["piece"]), ([], []))
            kh.append(k[lo:hi].reshape(-1, KVH, D))
            vh.append(v[lo:hi].reshape(-1, KVH, D))
            kc, vc = np.concatenate(kh), np.concatenate(vh)
            plan = req[r]
            if len(kc) != plan["klen"]:
                why.append(f"{r}: {len(kc)} cached keys, the plan's klen {plan['klen']}")
                continue
            for i in local[r]:
                d = plan["attention"][i]
                S = d["statics"]
                if d["base"] not in ATTENTION:
                    why.append(f"no twin of {d['definition']}")
                    continue
                twin, ns = ATTENTION[d["base"]]
                o, nf = twin(q[lo + i].reshape(NH, D), kc[:S["T"]], vc[:S["T"]], NS=ns(S), BN=S["BN"], masked_from=S["MASKED_FROM"],
                             dot_prim=S["DOT"], inv_prim=S["INV"])
                nonfinite += int(nf.any())
                bad += int(np.count_nonzero(o.reshape(-1) != got[lo + i])) + int(nf.any())
                rows_n += 1
                defs[d["base"]] = defs.get(d["base"], 0) + 1
        self._note(out, name, {"ok": not why and bad == 0, "bad": bad, "rows": rows_n, "nonfinite": nonfinite, "definitions": defs,
                               **({"why": why[:4]} if why else {})}, "attention")

    def _logits(self, out: dict[str, Any], plan: dict[str, Any], V: dict[str, np.ndarray], spans) -> None:
        lg, sm = self.program["logits"], self.program["sampled"]
        last = np.array([hi - 1 for _r, _lo, hi in spans])
        if LOGITS in V:
            d = plan["logits"]
            try:
                x, w = self._operand(lg["x"], V, last), self._operand(lg["w"], V, last)
                self._note(out, LOGITS, _verdict(GEMMS[d["base"]](d["statics"], x, w), V[LOGITS], definition=d["definition"]),
                           d["base"])
            except (_Unresolved, KeyError, TypeError) as e:
                self._note(out, LOGITS, {"ok": False, "why": f"{type(e).__name__}: {e}"}, "logits")
        if SAMPLED in V and LOGITS in V and sm["base"] == b1.TokenSelect.id:
            want = np.array([token_select_first_max(row) for row in V[LOGITS]], np.int64)
            got = V[SAMPLED][:, 0].astype(np.int64)
            self._note(out, SAMPLED, _verdict(want, got, definition=sm["definition"]), sm["base"])
            for (r, _lo, _hi), s in zip(spans, got):
                self.sampled[r] = int(s)
        elif SAMPLED in V:
            self._note(out, SAMPLED, {"ok": False, "why": f"no twin of {sm['definition']}"}, "sampled")

    # ------------------------------------------------------------------------------------------------------------- the run
    def report(self) -> dict[str, Any]:
        unrun = sorted(set(self.plans) - self.seen)
        ok = bool(self.steps) and not unrun and all(v.get("ok") for s in self.steps.values() for v in s.values())
        return {"ok": ok, "program_digest": self.build["program_digest"], "steps_checked": sorted(self.seen), "steps_unrun": unrun,
                "rows_per_request": self.k, "seed": self.seed, "picks": {str(s): p for s, p in sorted(self.picks.items())},
                "tally": dict(sorted(self.tally.items())),
                "failures": [{"step": s, "value": k, **v} for s, vs in sorted(self.steps.items()) for k, v in vs.items() if not v.get("ok")][:50],
                "steps": {str(s): v for s, v in sorted(self.steps.items())}}


def check(build: dict[str, Any], steps: Iterable[Step], weight: Callable[[str], np.ndarray], *, seed: int | None, rows: int = ROWS,
          prompts: Mapping[str, list[int]] | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The check over every step of the run, given in engine-step order."""
    c = ServedCheck(build, weight, rows=rows, prompts=prompts, log=log, seed=seed)
    for st in steps:
        c.step(st)
    return c.report()


def population(build: dict[str, Any], rep: dict[str, Any], *, ranks: list[int], first: int = 6) -> dict[str, Any]:
    """The check's reach (`report`) reconciled against the Build's identities, its Q(P), identity by identity: an identity
    (engine step, request, value) is reached when the report notes its value at that step for that request (a request of the
    step's committed batch; the ids only of a request whose prompt or previous sampled id they were compared to).  A noted value
    of a request the Build has no identity for is outside the query, an identity the report never reaches is without rows, and an
    identity's rows (element_range / width) must be the request's committed rows at the step (one for the logits and the sampled
    id).  The fields `c2_rules._query_population_scope` grades: `query_total`, `query_checks` (``ok`` False on any finding), `scope`."""
    prog = build["program"]
    width = {n: v["width"] for n, v in prog["values"].items()} | {LOGITS: prog["logits"]["gemm"][1], SAMPLED: 1}
    want: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    fam: Counter = Counter()
    for i in build["identities"]:
        fam[i["family"]] += 1
        want[(int(i["engine_step"]), str(i["request_id"]), str(i["op_path"]), str(i["output_member"]))] = i
    reached: Counter = Counter()
    outside, disagree, seen = [], [], set()
    for s, notes in rep["steps"].items():
        batch = (notes.get("batch") or {}).get("rows") or {}
        for name, v in notes.items():
            if name in ("batch", "coverage"):
                continue
            op, member = split_name(name)
            for r in ((v.get("requests") or []) if name == IDS else batch):
                at = {"request_id": r, "step": int(s), "op_path": op, "member": member}
                i = want.get(key := (int(s), r, op, member))
                if i is None:
                    outside.append(at)
                    continue
                seen.add(key)
                reached[i["family"]] += 1
                lo, hi = i["element_range"]
                rows = 1 if name in (LOGITS, SAMPLED) else batch[r]
                if hi - lo != rows * width[name]:
                    disagree.append({**at, "manifest_height": (hi - lo) / width[name], "program_rows": rows})
    without = [{"request_id": k[1], "step": k[0], "op_path": k[2], "member": k[3]} for k in want if k not in seen]
    manifest_ranks = sorted({int(i["rank"]) for i in build["identities"]})
    total, n_reached = len(build["identities"]), sum(reached.values())
    checks = {"vus_outside_query": {"n": len(outside), "first": outside[:first]},
              "identities_without_rows": {"n": len(without), "first": without[:first]},
              "rows_disagree": {"n": len(disagree), "first": disagree[:first]},
              "ranks": {"replay": sorted(ranks), "manifest": manifest_ranks, "equal": manifest_ranks == sorted(ranks)},
              "accounted": n_reached + len(without) == total,
              "unstated_families": sorted(set(fam) - {SERVED_FAMILY})}
    checks["ok"] = not (outside or without or disagree or checks["unstated_families"]) and checks["ranks"]["equal"] and checks["accounted"]
    qid = build["kind"]
    sampled = "GEMM, fused-MoE router and expert GEMM" if any("moe" in c for c in prog["calls"]) else "GEMM"
    scope = (f"{qid}(P) = {total} identities ({', '.join(f'{k} {v}' for k, v in sorted(fam.items()))}): one per (engine step, request, "
             f"value) of the served Program; the exact replay reaches {n_reached} ({100.0 * n_reached / max(total, 1):.1f}%): every call "
             f"on every row, each {sampled} and attention op on up to {rep.get('rows_per_request')} drawn rows of each request a step, "
             "the logits and the sampled id of every request")
    return {"query_total": {"source": "the served Build of record (acquire.served)", "query_id": qid, "identities": total,
                            "by_family": dict(sorted(fam.items())), "manifest_digest": build["manifest_digest"],
                            "program_digest": build["program_digest"]},
            "query_checks": checks, "scope": scope}
