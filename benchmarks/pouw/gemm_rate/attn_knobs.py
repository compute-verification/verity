"""FA3's own knobs for the served prefill attention on the H200, each timed against today's call in interleaved windows
and checked byte for byte against it.

    python attn_knobs.py --stamp-cubin stamp.cubin [--prompt 8x1024 ...] [--arms served,sched,contig,...]
                         [--fa3-so NS=PATH ...] [--window-ms 2000 --lead-ms 500 --rounds 3] [--table table.md] > knobs.jsonl

A --prompt is BxL (B sequences of L tokens, one engine step's batch), from the tree's Llama-3.2-1B int8 profile (32 query
heads, 8 KV heads, head dim 64). q, k and v are views of one qkv row block, as vLLM's split leaves them (row stride
(Hq + 2 Hkv) D), and the cache is vLLM's FlashAttention cache, [blocks, Hkv, 16, 2 D] with K and V interleaved by head, the
step's tokens at their slots. Every arm is one layer's flash_attn_varlen_func-equivalent call (causal, scale D^-0.5, window
(-1, -1), softcap 0, bf16 out into a fresh [T, Hq, D]):

- `served`: as FlashAttentionImpl calls it under batch invariance: the paged cache, block table, seqused_k, num_splits 1,
  no scheduler metadata, fa_version 3;
- `kattn`: the same call over attn_rate.py's cache ([blocks, 16, Hkv, D] per K and V), to tie this run to attn_rate.py's;
- `sched`: `served` with scheduler metadata computed ahead (get_scheduler_metadata at num_splits 1, page size 16);
- `contig`: k and v read from the qkv views with cu_seqlens_k = cu_seqlens_q and no block table, FA3's TMA path, as a
  step of fresh prefills allows (every key is a token of this step);
- `contig-sched`: `contig` with its scheduler metadata ahead;
- `page128`: the cache in 128-token pages (FA3's TMA paged path; a different cache layout, for information);
- `splits0`: `served` at num_splits 0 (FA3's split heuristic, not batch-invariant; for information);
- `so:NS`, per --fa3-so NS=PATH: `served` and `contig` through a rebuilt FA3 extension's `torch.ops.NS.fwd` (fa3_tile.sh).

Each arm is checked before it's timed: out and lse equal to `served`'s bytes (else the max abs difference and the most
bf16 ulps apart), equal on a rerun, and equal with another sequence ahead of the step's in the batch. Arms are graphed
(attn_rate.py's `graphed`), and `served-eager` launches `served` from Python as the engine does. Rounds interleave
every arm, each window --window-ms long after --lead-ms of calls; a window's MACs (attn_rate.py's useful causal ones) a
SM clock against bf16's 2,048, theta at 1,830 and 1,980 MHz and the SM clock come from torch_rate.py's stamps.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attn_rate as ar  # noqa: E402
import torch_rate as tr  # noqa: E402

PROMPTS = ("8x1024", "16x1024", "32x1024")
ARMS = ("served", "served-eager", "kattn", "sched", "contig", "contig-sched", "page128", "splits0")
BLOCK = 16


def i32(xs, device="cuda") -> torch.Tensor:
    return torch.tensor(list(xs), dtype=torch.int32, device=device)


def paged(k: torch.Tensor, v: torch.Tensor, lens: list[int], block: int, interleaved: bool):
    """(k cache view, v cache view, block table) holding each sequence's rows in its own run of blocks: vLLM's FlashAttention
    cache [blocks, Hkv, block, 2 D] split into K and V views when interleaved, else separate [blocks, block, Hkv, D] caches."""
    hkv, d = k.shape[1], k.shape[2]
    nbs = [-(-n // block) for n in lens]
    nb = sum(nbs)
    if interleaved:
        kv = torch.zeros((nb, hkv, block, 2 * d), dtype=k.dtype, device=k.device)
        kc, vc = kv.transpose(1, 2).split(d, dim=-1)
    else:
        kc = torch.zeros((nb, block, hkv, d), dtype=k.dtype, device=k.device)
        vc = torch.zeros_like(kc)
    rows, b0, o = [], 0, 0
    for n, nbi in zip(lens, nbs):
        for j in range(nbi):
            r = min(block, n - j * block)
            kc[b0 + j, :r] = k[o + j * block:o + j * block + r]
            vc[b0 + j, :r] = v[o + j * block:o + j * block + r]
        rows.append(list(range(b0, b0 + nbi)) + [0] * (max(nbs) - nbi))
        b0, o = b0 + nbi, o + n
    return kc, vc, i32(sum(rows, []), k.device).view(len(lens), -1)


class Step:
    """One engine step's prefill: q, k, v as views of a qkv block, and every cache an arm reads."""

    def __init__(self, m: dict, lens: list[int], seed: int, lead: int = 0) -> None:
        g = torch.Generator(device="cuda").manual_seed(seed)
        hq, hkv, d = m["hq"], m["hkv"], m["d"]
        self.lens, self.t, self.hq, self.d = lens, sum(lens), hq, d
        self.qkv = torch.randn(self.t, (hq + 2 * hkv) * d, generator=g, device="cuda").to(torch.bfloat16)
        q, k, v = self.qkv.split([hq * d, hkv * d, hkv * d], dim=-1)
        self.q, self.k, self.v = q.view(-1, hq, d), k.view(-1, hkv, d), v.view(-1, hkv, d)
        self.cu = i32([0] + [sum(lens[:i + 1]) for i in range(len(lens))])
        self.used = i32(lens)
        self.max = max(lens)
        self.caches = {"served": paged(self.k, self.v, lens, BLOCK, True), "kattn": paged(self.k, self.v, lens, BLOCK, False),
                       "page128": paged(self.k, self.v, lens, 128, True)}
        self.scale = d ** -0.5
        self.lead = lead

    def out(self) -> torch.Tensor:
        return torch.empty(self.t, self.hq, self.d, dtype=torch.bfloat16, device="cuda")


def fwd(op, q, k, v, out, cu_q, cu_k, used, max_q, max_k, bt, scale, sched, splits):
    """The FA3 op's positional call exactly as vllm_flash_attn's flash_attn_varlen_func makes it at fa_version 3."""
    return op(q, k, v, None, None, None, out, cu_q, cu_k, None, None, used, max_q, max_k, bt, None, None,
              None, None, None, None, None, None, scale, True, -1, -1, 0.0, True, sched, splits, None, 0, None, 1, 0, None)


class Arms:
    """Each arm's call as a thunk over a Step, returning (out, lse)."""

    def __init__(self, sos: dict) -> None:
        from vllm.vllm_flash_attn import flash_attn_interface as fai

        self.fai = fai
        self.ops = {"wheel": torch.ops._vllm_fa3_C.fwd}
        for ns, path in sos.items():
            torch.ops.load_library(path)
            self.ops[ns] = getattr(torch.ops, Path(path).stem).fwd
        self.sched_cache: dict = {}

    def sched(self, s: Step, paged_: bool) -> torch.Tensor:
        key = (id(s), paged_)
        if key not in self.sched_cache:
            self.sched_cache[key] = self.fai.get_scheduler_metadata(
                len(s.lens), s.max, s.max, s.hq, s.k.shape[1], s.d, s.used, qkv_dtype=torch.bfloat16, cu_seqlens_q=s.cu,
                page_size=BLOCK if paged_ else None, causal=True, window_size=(-1, -1), num_splits=1)
        return self.sched_cache[key]

    def call(self, arm: str, s: Step, out: torch.Tensor | None = None):
        op = self.ops["wheel"]
        if arm.startswith("so:"):
            ns, _, form = arm[3:].partition(":")
            op, arm = self.ops[ns], form or "served"
        out = s.out() if out is None else out
        if arm in ("served", "served-eager", "splits0", "sched", "kattn", "page128"):
            kc, vc, bt = s.caches["kattn" if arm == "kattn" else "page128" if arm == "page128" else "served"]
            sched = self.sched(s, True) if arm == "sched" else None
            splits = 0 if arm == "splits0" else 1
            return lambda: fwd(op, s.q, kc, vc, out, s.cu, None, s.used, s.max, s.max, bt, s.scale, sched, splits)[:2]
        if arm in ("contig", "contig-sched"):
            sched = self.sched(s, False) if arm == "contig-sched" else None
            return lambda: fwd(op, s.q, s.k, s.v, out, s.cu, s.cu, None, s.max, s.max, None, s.scale, sched, 1)[:2]
        raise SystemExit(f"unknown arm {arm}")


def ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    """The most bf16 ulps apart two tensors' elements are (on the sign-magnitude integer line)."""
    def line(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return int((line(a) - line(b)).abs().max())


def compare(got, ref) -> dict:
    (o, lse), (ro, rlse) = got, ref
    eq = torch.equal(o.view(torch.int16), ro.view(torch.int16)) and torch.equal(lse.view(torch.int32), rlse.view(torch.int32))
    out = {"eq": eq}
    if not eq:
        out |= {"out_eq": torch.equal(o.view(torch.int16), ro.view(torch.int16)), "max_abs": float((o.float() - ro.float()).abs().max()),
                "max_ulps": ulps(o, ro), "lse_max_abs": float((lse - rlse).abs().max())}
    return out


def snap(r):
    o, lse = r
    return o.clone(), lse.clone()


def poisoned(s: Step) -> torch.Tensor:
    return s.out().view(torch.int16).fill_(0x7FC1).view(torch.bfloat16)


def checks(arms: Arms, arm: str, s: Step, bat: Step, ref) -> dict:
    """arm's out and lse against `served`'s (ref), on a rerun, and with bat's extra sequence ahead of s's in the batch, each
    call into its own NaN-poisoned output."""
    a = snap(arms.call(arm, s, poisoned(s))())
    b = snap(arms.call(arm, s, poisoned(s))())
    c = arms.call(arm, bat, poisoned(bat))()
    torch.cuda.synchronize()
    x = bat.lead
    lead = (c[0][x:], c[1][:, x:])
    return {"vs_served": compare(a, ref), "repeat": compare(b, a), "batch": compare(snap(lead), a),
            "finite": bool(torch.isfinite(a[0].float()).all())}


class Guard:
    """Copies of every tensor a step's calls read, so a call that writes past its own buffers is named by the first check
    after it rather than by a later arm's illegal address."""

    def __init__(self, *steps: Step) -> None:
        self.t = {f"{i}.{k}": x for i, s in enumerate(steps) for k, x in self.reads(s).items()}
        self.ref = {k: x.clone() for k, x in self.t.items()}

    @staticmethod
    def reads(s: Step) -> dict:
        out = {"qkv": s.qkv, "cu": s.cu, "used": s.used}
        for name, (kc, vc, bt) in s.caches.items():
            out |= {f"{name}.k": kc, f"{name}.v": vc, f"{name}.bt": bt}
        return out

    def changed(self) -> list[str]:
        torch.cuda.synchronize()
        return [k for k, x in self.t.items() if not torch.equal(x, self.ref[k])]


def batched(m: dict, s: Step, seed: int) -> Step:
    """s's step with one more sequence (half its length) ahead of its own, the same q, k, v rows for s's sequences."""
    x = max(BLOCK, s.max // 2)
    b = Step(m, [x] + s.lens, seed, lead=x)
    b.qkv[x:] = s.qkv
    b.caches = {"served": paged(b.k, b.v, b.lens, BLOCK, True), "kattn": paged(b.k, b.v, b.lens, BLOCK, False),
                "page128": paged(b.k, b.v, b.lens, 128, True)}
    return b


def table(rows: list[dict]) -> str:
    by: dict = {}
    for r in rows:
        if r.get("kind") == "window" and r.get("sms"):
            by.setdefault((r["prompt"], r["arm"]), []).append(r)
    chk = {(r["prompt"], r["arm"]): r for r in rows if r.get("kind") == "check"}
    out = ["| prompt | arm | of bf16 a clock | θ @1830 | SM MHz | µs a layer | vs served | rounds |", "|---|---|---|---|---|---|---|---|"]
    for (prompt, arm), rs in sorted(by.items()):
        med = {f: statistics.median(x[f] for x in rs) for f in ("of_peak", "theta_1830", "sm_mhz", "us_call")}
        c = chk.get((prompt, arm.removesuffix("-eager")), {})
        v = c.get("vs_served", {})
        same = "—" if not c else "bytes equal" if v.get("eq") and c["repeat"]["eq"] and c["batch"]["eq"] else \
            f"differs: {v.get('max_ulps', 0)} ulp, {v.get('max_abs', 0):.3g} abs" if not v.get("eq") else "not invariant"
        out.append(f"| {prompt} | {arm} | {med['of_peak']:.3f} | {med['theta_1830']:.3f} | {med['sm_mhz']:.0f} | "
                   f"{med['us_call']:.1f} | {same} | {len(rs)} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stamp-cubin", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ar.CONFIG)
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--fa3-so", action="append", default=[], help="NS=PATH: a rebuilt FA3 extension, arms so:NS and so:NS:contig")
    ap.add_argument("--window-ms", type=float, default=2000)
    ap.add_argument("--lead-ms", type=float, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--table", type=Path)
    args = ap.parse_args()
    sos = dict(x.split("=", 1) for x in args.fa3_so)
    names = args.arms.split(",") + [a for ns in sos for a in (f"so:{ns}", f"so:{ns}:contig")]
    m = ar.model(args.config)
    nv = tr.Nvml()
    st = tr.Stamps(args.stamp_cubin)
    st.ck("cuFuncSetAttribute", st.f, ar.CARVEOUT, 100)
    arms = Arms(sos)
    rows: list[dict] = []

    def emit(d: dict) -> None:
        rows.append(d)
        print(json.dumps(d), flush=True)

    emit({"kind": "device", "uuid": nv.uuid, "name": torch.cuda.get_device_name(0), "sms": st.nsm, "torch": torch.__version__,
          "cuda": torch.version.cuda, "arms": names, "sos": sos, **nv.sample()})
    for pi, prompt in enumerate(args.prompt or PROMPTS):
        lens = ar.parse_prompt(prompt)
        arms.sched_cache.clear()
        s = Step(m, lens, args.seed + pi)
        bat = batched(m, s, args.seed + 100 + pi)
        ref = snap(arms.call("served", s, poisoned(s))())
        kc, vc, bt = s.caches["served"]
        via = arms.fai.flash_attn_varlen_func(
            q=s.q, k=kc, v=vc, out=poisoned(s), cu_seqlens_q=s.cu, max_seqlen_q=s.max, seqused_k=s.used, max_seqlen_k=s.max,
            softmax_scale=s.scale, causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0, scheduler_metadata=None,
            fa_version=3, num_splits=1, return_softmax_lse=True)
        emit({"kind": "check", "prompt": prompt, "arm": "flash_attn_varlen_func", "vs_served": compare(snap(via), ref)})
        guard = Guard(s, bat)

        def intact(after: str, guard: Guard = guard) -> None:
            if bad := guard.changed():
                emit({"kind": "corrupt", "prompt": prompt, "after": after, "changed": bad})
                raise SystemExit(f"{after} wrote into {bad}")

        macs = ar.attn_macs(m, lens)
        ready = []
        for a in names:
            if a != "served-eager":
                emit({"kind": "check", "prompt": prompt, "arm": a, **checks(arms, a, s, bat, ref)})
                intact(f"{a} check")
            f = arms.call(a.removesuffix("-eager"), s)
            emit({"kind": "prepare", "prompt": prompt, "arm": a})
            it = ar.prepare(emit, a, "bf16", macs, f, not a.endswith("-eager"), {"prompt": prompt, "t": s.t, "form": "whole"})
            it["call"] = f  # its out buffer, which the graph writes: torch.cuda.graph's empty_cache would free it otherwise
            for _ in range(3):
                it["fn"]()
            intact(f"{a} prepare")
            ready.append(it)
        for r in range(args.rounds):
            for it in ready:
                emit({"kind": "start", "prompt": prompt, "arm": it["arm"], "round": r})
                emit(ar.window(args, nv, st, it, r))
                intact(f"{it['arm']} window")
        del ready, s, bat, guard
        torch.cuda.empty_cache()
    if args.table:
        args.table.write_text(table(rows))


if __name__ == "__main__":
    main()
