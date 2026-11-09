"""MACs per SM clock of the served stack's prefill attention on the H200, against bf16's 2,048, with the served int8
linears beside it, over sustained windows between stamps of every SM (torch_rate.py's windows and stamps).

    python attn_rate.py --stamp-cubin stamp.cubin [--config HF_CONFIG.json] [--prompt 16384 --prompt 8x1024 ...]
                        [--chunk 8192] [--arms fa3,fa3-eager,fa2,bf16,lin] [--window-ms 2000 --lead-ms 500 --rounds 3]
                        [--table table.md] > attn.jsonl

The model's attention and linears come from its HF config (default the tree's Llama-3.2-1B int8 profile): Hq query heads,
Hkv KV heads, head dim D, hidden and MLP widths. A --prompt is L (one sequence of L tokens) or BxL (B sequences of L
tokens, one engine step's batch), each prefilled causally in one call (`whole`) or, for one sequence longer than --chunk C,
also in vLLM's chunked-prefill steps of C query rows over every key so far (`chunked`; C is the engine's
max_num_batched_tokens). An arm's call is one layer of that prompt:

- `fa3`, `fa2`: vLLM's `flash_attn_varlen_func` as its FlashAttentionImpl calls it for a prefill (bf16 q, a paged KV cache
  of 16-token blocks holding the prompt's keys, block table, cu_seqlens_q, seqused_k, causal, scale D^-0.5, window
  (-1, -1), num_splits 1 as batch-invariant serving pins it, no scheduler metadata) at fa_version 3 (vLLM's choice on
  cc 9.x) or 2; replayed from a CUDA graph of the calls, so the window holds the kernel's own time and no launch gaps;
- `fa3-eager`: the same calls launched from Python one by one, as the served engine (enforce_eager) launches them;
- `bf16`: attn_bf16's qk then pv used as a prefill, BF16_ROWS query rows of one sequence a launch pair (S buffered, no
  softmax between: the two products only), graphed;
- `lin`: one layer's four linears as the served `verity_int8::row_quant` and `verity_int8::scaled_mm` run them
  (`int8_fused.row_quant`, `int8_fused.product`) at M = the step's rows, graphed.

MACs are the useful ones: attention's Hq D L (L + 1) a sequence and layer (QK^T and PV, each query row against the keys
at or before it), a linear's M N K. A window's numbers (`point`): of_peak, MACs over its SM clocks over the SM's dense rate
a clock (PEAK: the H200 card's rate.tensor, 4,096 s8 MACs an SM clock, at bf16's weight 2 for attention); theta, MACs a
second over the card's peak at 1,830 and 1,980 MHz (132 SMs); the SM clock, from clock64. The stamp kernel runs at the
largest shared-memory carveout. Before timing, `fa_checks` records whether each FA call is bit-identical on a repeat and
with another sequence ahead of the prompt's in the batch (num_splits 1's batch invariance), and whether the chunked
prefill equals the whole one; `Bf16.check` whether attn_bf16's fast path equals its reference path word for word on the
first prompt's first rows. `--table` writes each arm's median over rounds, and attention's share of one layer's tensor
MACs and time beside the linears at the same prompt (norms, RoPE, the activation and sampling are outside both).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import torch_rate as tr  # noqa: E402

CONFIG = ROOT / "integrations/vllm/verity_vllm/engine/profiles/hf_configs/LLAMA32_1B_INT8.config.json"
PROMPTS = ("1024", "4096", "8192", "16384", "8x1024", "16x1024", "32x1024")
PEAK = {"bf16": 2048, "s8": 4096}
SMS = 132
MHZ = (1830, 1980)
CARVEOUT = 9  # CUfunction_attribute PREFERRED_SHARED_MEMORY_CARVEOUT, in percent of the largest
BLOCK = 16
BF16_ROWS = 1024
GRAPH_MACS = 2e11
FA_ARMS = {"fa3": 3, "fa3-eager": 3, "fa2": 2}


def model(path: Path) -> dict:
    c = json.loads(Path(path).read_text())
    hq, hkv, hidden, ffn = c["num_attention_heads"], c["num_key_value_heads"], c["hidden_size"], c["intermediate_size"]
    d = c.get("head_dim") or hidden // hq
    return {"hq": hq, "hkv": hkv, "d": d, "hidden": hidden, "ffn": ffn, "layers": c["num_hidden_layers"],
            "linears": {"qkv": ((hq + 2 * hkv) * d, hidden), "o": (hidden, hq * d), "gate_up": (2 * ffn, hidden),
                        "down": (hidden, ffn)}}


def parse_prompt(text: str) -> list[int]:
    b, _, n = text.rpartition("x")
    return [int(n)] * int(b or 1)


def attn_macs(m: dict, lens: list[int]) -> int:
    return sum(m["hq"] * m["d"] * n * (n + 1) for n in lens)


def lin_macs(m: dict, rows: int) -> int:
    return rows * sum(n * k for n, k in m["linears"].values())


def chunks(t: int, c: int) -> list[tuple[int, int]]:
    """(first row, rows) of each prefill step of one t-token sequence: one step without a chunk size."""
    c = c or t
    return [(i, min(c, t - i)) for i in range(0, t, c)]


def point(w: dict, macs_call: int, kind: str) -> dict:
    macs = w["calls"] * macs_call
    s = w["span_ns"] * 1e-9
    out = {"macs": macs, "s": s, "us_call": 1e6 * s / w["calls"], "of_peak": macs / w["sm_clocks"] / PEAK[kind],
           "sm_mhz": w["sm_clocks"] / w["sms"] / w["span_ns"] * 1e3}
    for f in MHZ:
        out[f"theta_{f}"] = macs / s / (SMS * PEAK[kind] * f * 1e6)
    return out


def graphed(fn, reps: int):
    """fn's launches, reps times over, as one CUDA graph's replay (after two warm-up calls on a side stream)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    return g.replay


def i32(xs) -> torch.Tensor:
    return torch.tensor(list(xs), dtype=torch.int32, device="cuda")


class Prompt:
    """A step's sequences: q, k, v (bf16 normal) row-stacked, their paged cache, and one more sequence for the batch check."""

    def __init__(self, m: dict, lens: list[int], seed: int) -> None:
        g = torch.Generator(device="cuda").manual_seed(seed)
        hq, hkv, d = m["hq"], m["hkv"], m["d"]
        self.lens, self.t, self.xl = lens, sum(lens), max(BLOCK, lens[0] // 2)
        rn = lambda *s: torch.randn(*s, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)  # noqa: E731
        self.q, self.k, self.v = rn(self.t, hq, d), rn(self.t, hkv, d), rn(self.t, hkv, d)
        self.qx, kx, vx = rn(self.xl, hq, d), rn(self.xl, hkv, d), rn(self.xl, hkv, d)
        nbs = [-(-n // BLOCK) for n in lens + [self.xl]]
        self.kc = torch.zeros((sum(nbs), BLOCK, hkv, d), dtype=torch.bfloat16, device="cuda")
        self.vc = torch.zeros_like(self.kc)
        rows, b0, self.off, self.first = [], 0, [], []
        for i, (n, nb) in enumerate(zip(lens + [self.xl], nbs)):
            o = sum(lens[:i])
            src_k, src_v = (self.k[o:o + n], self.v[o:o + n]) if i < len(lens) else (kx, vx)
            self.kc[b0:b0 + nb].view(-1, hkv, d)[:n] = src_k
            self.vc[b0:b0 + nb].view(-1, hkv, d)[:n] = src_v
            rows.append(list(range(b0, b0 + nb)) + [0] * (max(nbs) - nb))
            self.off.append(o)
            b0 += nb
        self.bt = i32(sum(rows[:-1], [])).view(len(lens), -1)
        self.btx = i32(sum([rows[-1]] + rows[:-1], [])).view(len(lens) + 1, -1)
        self.cu = i32([0] + [sum(lens[:i + 1]) for i in range(len(lens))])
        self.used = i32(lens)
        self.out = torch.empty_like(self.q)
        self.scale = d ** -0.5


def fa(fai, fav: int, q, kc, vc, out, cu, max_q: int, used, max_k: int, bt) -> None:
    fai.flash_attn_varlen_func(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=max_q, seqused_k=used,
                               max_seqlen_k=max_k, softmax_scale=q.shape[-1] ** -0.5, causal=True, window_size=(-1, -1),
                               block_table=bt, softcap=0.0, scheduler_metadata=None, fa_version=fav, num_splits=1)


def fa_calls(fai, p: Prompt, fav: int, c: int, out=None) -> list:
    """The prefill's flash_attn_varlen_func calls, one a step, as thunks writing out (default p.out): the whole batch in one
    call, or one sequence's chunks of c query rows."""
    out = p.out if out is None else out
    if not c or len(p.lens) > 1 or c >= p.t:
        return [lambda: fa(fai, fav, p.q, p.kc, p.vc, out, p.cu, max(p.lens), p.used, max(p.lens), p.bt)]
    calls = []
    for i0, r in chunks(p.t, c):
        cu, used, q, o = i32([0, r]), i32([i0 + r]), p.q[i0:i0 + r], out[i0:i0 + r]
        calls.append(lambda q=q, o=o, cu=cu, used=used, r=r, kl=i0 + r: fa(fai, fav, q, p.kc, p.vc, o, cu, r, used, kl, p.bt))
    return calls


def run_all(calls) -> None:
    for f in calls:
        f()


def fa_checks(fai, p: Prompt, fav: int, c: int) -> dict:
    a, b, ch = torch.empty_like(p.q), torch.empty_like(p.q), torch.empty_like(p.q)
    run_all(fa_calls(fai, p, fav, 0, a))
    run_all(fa_calls(fai, p, fav, 0, b))
    chunked = bool(c) and len(p.lens) == 1 and c < p.t
    if chunked:
        run_all(fa_calls(fai, p, fav, c, ch))
    q = torch.cat([p.qx, p.q])
    bat = torch.empty_like(q)
    fa(fai, fav, q, p.kc, p.vc, bat, torch.cat([i32([0]), p.cu + p.xl]), max(p.lens + [p.xl]),
       torch.cat([i32([p.xl]), p.used]), max(p.lens + [p.xl]), p.btx)
    torch.cuda.synchronize()
    return {"repeat_eq": torch.equal(a, b), "batch_eq": torch.equal(a, bat[p.xl:]),
            "chunked_eq_whole": torch.equal(a, ch) if chunked else None, "finite": bool(torch.isfinite(a.float()).all())}


class Bf16:
    """attn_bf16's qk and pv as a prefill of BF16_ROWS query rows of one sequence a launch pair, over buffers made once."""

    def __init__(self, m: dict, p: Prompt) -> None:
        sys.path.insert(0, str(ROOT / "benchmarks/torch_gpu/attn_bf16"))
        import attn

        self.attn = attn
        self.s = torch.empty(min(BF16_ROWS, max(p.lens)) * m["hq"] * max(p.lens), dtype=torch.bfloat16, device="cuda")
        self.o = torch.empty_like(p.q)
        self.steps = []
        for o, n in zip(p.off, p.lens):
            kt = p.k[o:o + n].permute(1, 0, 2).contiguous()
            vt = p.v[o:o + n].permute(1, 0, 2).contiguous()
            for i0, rr in chunks(n, BF16_ROWS):
                tc = i0 + rr
                self.steps.append((p.q[o + i0:o + i0 + rr], kt[None, :, :tc].expand(rr, -1, -1, -1),
                                   vt[None, :, :tc].expand(rr, -1, -1, -1), i32(range(i0 + 1, i0 + rr + 1)),
                                   self.s[: rr * m["hq"] * tc].view(rr, m["hq"], tc), self.o[o + i0:o + i0 + rr]))

    def call(self, path: str = "fast", steps=None) -> None:
        for q, k, v, lens, s, o in steps or self.steps:
            self.attn.qk(q, k, lens, out=s, check_lens=False, path=path)
            self.attn.pv(s, v, lens, out=o, check_lens=False, path=path)

    def check(self) -> dict:
        q, k, v, lens, s, o = self.steps[0]
        got = []
        for path in ("fast", "ref"):
            s2, o2 = torch.empty_like(s), torch.empty_like(o)
            self.call(path, [(q, k, v, lens, s2, o2)])
            torch.cuda.synchronize()
            got.append((s2.view(torch.int16), o2.view(torch.int16)))
        (sf, of), (sr, orf) = got
        return {"rows": q.shape[0], "s_eq_ref": torch.equal(sf, sr), "o_eq_ref": torch.equal(of, orf)}


class Lin:
    """One layer's four served int8 linears at M rows: bf16 x, int8 w [N, K] and f32 s_w [N]."""

    def __init__(self, m: dict, rows: int, seed: int) -> None:
        from verity_vllm.engine import int8_fused

        self.f = int8_fused
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.ops = []
        for n, k in m["linears"].values():
            x = torch.randn(rows, k, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda", generator=g)
            sw = torch.rand(n, generator=g, device="cuda", dtype=torch.float32).mul_(1e-3).add_(1e-4)
            self.ops.append((x, w, sw))

    def call(self) -> None:
        for x, w, sw in self.ops:
            xq, sa = self.f.row_quant(x)
            self.f.product(xq, sa, w, sw)


def prepare(emit, arm: str, kind: str, macs_call: int, fn, graph: bool, extra: dict) -> dict:
    """An arm's timed unit: fn, or a graph of enough calls of it to hold GRAPH_MACS, and its time a unit."""
    reps = max(1, math.ceil(GRAPH_MACS / macs_call)) if graph else 1
    rep = fn
    if graph:
        try:
            rep = graphed(fn, reps)
        except Exception as e:  # noqa: BLE001
            emit({"kind": "graph_refused", "arm": arm, **extra, "error": repr(e)[:400]})
            graph, reps, rep = False, 1, fn
    return {"arm": arm, "kind": kind, "macs_call": macs_call, "fn": rep, "graph": graph, "reps": reps, "extra": extra,
            "per": tr.per_call_s(rep)}


def window(args, nv, st, it: dict, r: int) -> dict:
    w = tr.window(nv, st, it["fn"], it["per"], args.window_ms, args.lead_ms)
    row = {"kind": "window", "arm": it["arm"], "type": it["kind"], "round": r, "graph": it["graph"],
           "calls_a_replay": it["reps"], "macs_call": it["macs_call"], **it["extra"], **w}
    if w["sms"]:
        row.update(point(w, it["macs_call"] * it["reps"], it["kind"]))
        row["us_call"] /= it["reps"]
    return row


def table(rows: list[dict], m: dict, chunk: int) -> str:
    """Each arm's median over rounds; attention's share of one layer beside the linears at the same prompt."""
    by: dict = {}
    for r in rows:
        if r.get("kind") == "window" and r.get("sms"):
            by.setdefault((r["arm"], r["prompt"], r["form"]), []).append(r)
    fields = ("of_peak", "theta_1830", "theta_1980", "sm_mhz", "us_call")
    med = {key: {f: statistics.median(x[f] for x in rs) for f in fields} | {"n": len(rs), "t": rs[0]["t"]}
           for key, rs in by.items()}
    out = ["| arm | prompt | form | of peak | θ @1830 | θ @1980 | SM MHz | µs a layer | rounds |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (arm, prompt, form), v in sorted(med.items(), key=lambda kv: (kv[1]["t"], kv[0][1], kv[0][2], kv[0][0])):
        out.append(f"| {arm} | {prompt} | {form} | {v['of_peak']:.3f} | {v['theta_1830']:.3f} | {v['theta_1980']:.3f} | "
                   f"{v['sm_mhz']:.0f} | {v['us_call']:.1f} | {v['n']} |")
    out += ["", "| attention arm | prompt | form | attention's MACs share | attention's time share | attention µs | linears µs |",
            "|---|---|---|---|---|---|---|"]
    for (arm, prompt, form), a in sorted(med.items(), key=lambda kv: (kv[1]["t"], kv[0][1], kv[0][2], kv[0][0])):
        if arm == "lin":
            continue
        lens, t = parse_prompt(prompt), a["t"]
        rows_m = min(chunk, t) if form == "chunked" else t
        lin = med.get(("lin", f"M{rows_m}", "whole"))
        if lin is None:
            continue
        t_lin = lin["us_call"] * t / rows_m
        ma, ml = attn_macs(m, lens), lin_macs(m, t)
        out.append(f"| {arm} | {prompt} | {form} | {ma / (ma + ml):.3f} | {a['us_call'] / (a['us_call'] + t_lin):.3f} | "
                   f"{a['us_call']:.1f} | {t_lin:.1f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stamp-cubin", type=Path, required=True, help="stamp.cu built for the card")
    ap.add_argument("--config", type=Path, default=CONFIG, help="the model's HF config.json")
    ap.add_argument("--prompt", action="append", help=f"L or BxL (default {' '.join(PROMPTS)})")
    ap.add_argument("--chunk", type=int, default=8192, help="max_num_batched_tokens: chunked prefill's rows a step (0: none)")
    ap.add_argument("--arms", default="fa3,fa3-eager,fa2,bf16,lin")
    ap.add_argument("--window-ms", type=float, default=2000)
    ap.add_argument("--lead-ms", type=float, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--table", type=Path)
    args = ap.parse_args()
    arms = args.arms.split(",")
    m = model(args.config)
    prompts = args.prompt or list(PROMPTS)
    nv = tr.Nvml()
    st = tr.Stamps(args.stamp_cubin)
    st.ck("cuFuncSetAttribute", st.f, CARVEOUT, 100)
    rows: list[dict] = []

    def emit(d: dict) -> None:
        rows.append(d)
        print(json.dumps(d), flush=True)

    fai = None
    if any(a in FA_ARMS for a in arms):
        from vllm.vllm_flash_attn import flash_attn_interface as fai
    import importlib.metadata as md

    vers = {pkg: md.version(pkg) for pkg in ("vllm", "torch", "triton")}
    sup = {v: (fai.is_fa_version_supported(v) if fai and hasattr(fai, "is_fa_version_supported") else None) for v in (2, 3)}
    emit({"kind": "device", "uuid": nv.uuid, "name": torch.cuda.get_device_name(0), "sms": st.nsm, "cuda": torch.version.cuda,
          "versions": vers, "fa_supported": sup, "model": {k: v for k, v in m.items() if k != "linears"},
          "linears": m["linears"], "config": str(args.config), "chunk": args.chunk, "arms": arms, "prompts": prompts,
          "nvml_mhz": nv.sample()["nvml_mhz"]})
    if st.nsm != SMS:
        raise SystemExit(f"{st.nsm} SMs, not the card's {SMS}")
    lin_done: set[int] = set()
    for pi, prompt in enumerate(prompts):
        lens = parse_prompt(prompt)
        p = Prompt(m, lens, args.seed + pi)
        chunked = bool(args.chunk) and len(lens) == 1 and args.chunk < p.t
        forms = [("whole", 0)] + ([("chunked", args.chunk)] if chunked else [])
        ma = attn_macs(m, lens)
        items: list[tuple] = []
        for arm in arms:
            if arm in FA_ARMS:
                if arm != "fa3-eager":
                    emit({"kind": "check", "arm": arm, "prompt": prompt, "chunk": args.chunk,
                          **fa_checks(fai, p, FA_ARMS[arm], args.chunk)})
                for form, c in forms:
                    calls = fa_calls(fai, p, FA_ARMS[arm], c)
                    items.append((arm, "bf16", ma, lambda calls=calls: run_all(calls), arm != "fa3-eager",
                                  {"prompt": prompt, "t": p.t, "form": form, "steps": len(calls)}))
            elif arm == "bf16":
                b = Bf16(m, p)
                if pi == 0:
                    emit({"kind": "check", "arm": arm, "prompt": prompt, **b.check()})
                items.append((arm, "bf16", ma, b.call, True, {"prompt": prompt, "t": p.t, "form": "whole",
                                                              "steps": len(b.steps)}))
            elif arm == "lin":
                for r in [p.t] + [c for _, c in forms[1:]]:
                    if r not in lin_done:
                        lin_done.add(r)
                        items.append(("lin", "s8", lin_macs(m, r), Lin(m, r, args.seed).call, True,
                                      {"prompt": f"M{r}", "t": r, "form": "whole", "steps": 4}))
        ready = [prepare(emit, *it) for it in items]
        for r in range(args.rounds):
            for it in ready:
                emit(window(args, nv, st, it, r))
        del ready, items, p
        torch.cuda.empty_cache()
    if args.table:
        args.table.write_text(table(rows, m, args.chunk))


if __name__ == "__main__":
    main()
