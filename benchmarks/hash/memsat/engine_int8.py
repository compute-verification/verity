"""One serving engine of run.sh's SERVE=node at saturation, on the exact-int8 profile: memory accounting's
`benchmarks/remote_session/engine.py` unchanged (its arena, decode at use, handshake and phases), with three things
around it.

~~~text
int8       vllm's exact-int8 profile (`TargetProfile(int8_exact=--int8)`): its engine args (quantization verity_int8, and
           the CUSTOM attention backend for `all`) and the registrations `verity_vllm.engine.build` makes before an engine
           exists. The int8 linears' weights are int8 [N, K], contiguous, so engine.py's arena takes them, the embedding and
           the bf16 LM head as it takes a bf16 model's; their 1-D float32 scales stay outside it
saturation --max-num-seqs caps vLLM's batch below engine.py's --concurrency, so requests always wait in the scheduler,
           and a --step-hz above the step rate makes each step start as soon as the last one ends (with --align-steps,
           at the host clock's next 1/hz)
latency    each request's tokens as they come out of the steps (engine.py's DELTA outputs): time to first token from its
           add and every inter-token gap, each by the phase of the step that made it, and its TPOT ((last - first) /
           (n - 1)) once it finishes, only if its first token and its last came in one phase; and each phase's busy
           fraction (step time over its span, the sum of its runs of consecutive steps) and batch (requests a step).
           --out/engine.<g>.latency.json at stop, beside engine.py's engine.<g>.json
~~~

    VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=3 python engine_int8.py --gpu 3 --dir B --out OUT --kv kv.so \
        --model CHECKPOINT --int8 linears --max-num-seqs 256 --concurrency 384 --step-hz 1000 --align-steps
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

RS = Path(__file__).resolve().parents[2] / "remote_session"
sys.path.insert(0, str(RS))

import engine as E

BIN_MS = 0.1


def percentile(hist: Counter, q: float) -> float | None:
    """The q-quantile of a histogram of BIN_MS bins, as the bin's upper edge (ms)."""
    n = sum(hist.values())
    if not n:
        return None
    k, seen = q * (n - 1), 0
    for b in sorted(hist):
        seen += hist[b]
        if seen > k:
            return round((b + 1) * BIN_MS, 3)
    return None


def described(hist: Counter) -> dict:
    n = sum(hist.values())
    mean = sum((b + 0.5) * BIN_MS * c for b, c in hist.items()) / n if n else None
    return {"n": n, "mean_ms": None if mean is None else round(mean, 3),
            **{f"p{round(q * 100)}_ms": percentile(hist, q) for q in (0.5, 0.9, 0.99)}}


class Phase:
    def __init__(self) -> None:
        self.itl, self.ttft, self.tpot = Counter(), Counter(), Counter()
        self.steps = self.tokens = self.batch = 0
        self.busy_s = self.span_s = 0.0

    def report(self) -> dict:
        span = self.span_s
        return {"steps": self.steps, "tokens": self.tokens, "batch_sum": self.batch, "busy_s": round(self.busy_s, 4),
                "span_s": round(span, 3), "tokens_per_s": round(self.tokens / span, 3) if span else None,
                "busy": round(self.busy_s / span, 4) if span else None,
                "batch_mean": round(self.batch / self.steps, 2) if self.steps else None,
                **{k: described(h) for k, h in (("itl", self.itl), ("ttft", self.ttft), ("tpot", self.tpot))},
                "hist": {k: {str(b): c for b, c in sorted(h.items())}
                         for k, h in (("itl", self.itl), ("ttft", self.ttft), ("tpot", self.tpot))}}


class Saturated(E.Load):
    """engine.Load's steady load with each request's token times recorded (`latency` in the module doc). engine.main
    tags a step with its phase after step() returns, so a step's samples join its phase at the next step."""

    current: Saturated | None = None

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.added, self.first, self.last, self.count, self.first_step = {}, {}, {}, {}, {}
        self.phases: dict[str, Phase] = {}
        self.pending, self.prev = None, None
        Saturated.current = self

    def step(self) -> dict:
        while len(self.live) < self.concurrency:
            prompt = [self.rng.randrange(1000, 120000) for _ in range(self.prompt)]
            rid = f"load-{self.n}"
            self.engine.add_request(rid, {"prompt_token_ids": prompt}, self.params)
            self.added[rid] = time.perf_counter()
            self.live.add(rid)
            self.n += 1
        self.settle()
        s: dict = {}
        t0 = time.perf_counter()
        outs = self.engine.step()
        t1 = time.perf_counter()
        tokens, ttft, itl, tpot = 0, [], [], []
        for o in outs:
            n = sum(len(c.token_ids) for c in o.outputs)
            tokens += n
            rid = o.request_id
            if n:
                if rid not in self.first:
                    self.first[rid], self.first_step[rid] = t1, s
                    if rid in self.added:
                        ttft.append(t1 - self.added[rid])
                else:
                    itl.append(t1 - self.last[rid])
                self.last[rid] = t1
                self.count[rid] = self.count.get(rid, 0) + n
            if o.finished:
                self.live.discard(rid)
                if self.count.get(rid, 0) > 1 and rid in self.first:
                    tpot.append(((self.last[rid] - self.first[rid]) / (self.count[rid] - 1), self.first_step[rid]))
                for d in (self.added, self.first, self.last, self.count, self.first_step):
                    d.pop(rid, None)
        now = time.time()
        if self.align:
            self.next_at = E.aligned(time.perf_counter(), now, self.interval)
        else:
            self.next_at = max(self.next_at + self.interval, t1)
        s.update({"t": now, "ms": (t1 - t0) * 1e3, "tokens": tokens, "running": len(self.live), "batch": len(outs)})
        self.pending = (s, t0, t1, tokens, len(outs), ttft, itl, tpot)
        return s

    def settle(self) -> None:
        """The last step's samples into its phase, now that engine.main has tagged it."""
        if self.pending is None:
            return
        s, t0, t1, tokens, batch, ttft, itl, tpot = self.pending
        self.pending = None
        phase = s.get("phase", "untagged")
        p = self.phases.setdefault(phase, Phase())
        p.steps += 1
        p.tokens += tokens
        p.batch += batch
        p.busy_s += t1 - t0
        # a phase can recur (between, after each session), so its span is the sum of its runs of consecutive steps
        p.span_s += t1 - self.prev[1] if self.prev and self.prev[0] == phase else t1 - t0
        self.prev = (phase, t1)
        whole = [x for x, first in tpot if first.get("phase", "untagged") == phase]
        for hist, xs in ((p.ttft, ttft), (p.itl, itl), (p.tpot, whole)):
            hist.update(int(x * 1e3 / BIN_MS) for x in xs)

    def abort(self) -> None:
        super().abort()
        for d in (self.added, self.first, self.last, self.count, self.first_step):
            d.clear()

    def report(self) -> dict:
        self.settle()
        return {name: p.report() for name, p in self.phases.items()}


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--int8", choices=("linears", "attention", "all"), default="linears")
    ap.add_argument("--max-num-seqs", type=int, default=0, help="vLLM's batch cap (0: engine.py's, its --concurrency)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0, help="vLLM's token budget a step (0: vLLM's)")
    mine, rest = ap.parse_known_args()
    where = argparse.ArgumentParser(add_help=False)
    where.add_argument("--gpu", type=int, required=True)
    where.add_argument("--out", type=Path, required=True)
    at, _ = where.parse_known_args(rest)
    sys.argv = [str(RS / "engine.py"), *rest]

    from verity_vllm.program.frontend.target_profile import TargetProfile
    from vllm.v1.engine.llm_engine import LLMEngine

    profile = TargetProfile(int8_exact=mine.int8)
    if "linears" in profile.int8_parts:
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
    if "attention" in profile.int8_parts:
        from verity_vllm.engine import int8_attention_vllm
        int8_attention_vllm.register()
    extra = profile.engine_args(execution=False)
    if mine.max_num_seqs:
        extra["max_num_seqs"] = mine.max_num_seqs
    if mine.max_num_batched_tokens:
        extra["max_num_batched_tokens"] = mine.max_num_batched_tokens
    build = LLMEngine.from_engine_args.__func__

    def from_engine_args(cls, engine_args, *a, **kw):
        for k, v in extra.items():
            setattr(engine_args, k, v)
        return build(cls, engine_args, *a, **kw)

    LLMEngine.from_engine_args = classmethod(from_engine_args)
    E.Load = Saturated
    summary = E.summary

    def summarized(steps: list[dict]) -> dict:
        load = Saturated.current
        report = {"schema": "verity/hash-memsat/latency/v0", "gpu": at.gpu, "int8": mine.int8, "engine_args": extra,
                  "bin_ms": BIN_MS, "phases": load.report() if load else {}}
        E.write_json(at.out / f"engine.{at.gpu}.latency.json", report)
        return summary(steps)

    E.summary = summarized
    return E.main()


if __name__ == "__main__":
    sys.exit(main())
