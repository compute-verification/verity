"""A training step's work read off its Program: the lf model's step (`adapters/vllm/tests/program/train_step_lf.py`,
loaded by its path) under AdamW, as the training-step frontend writes it (`step_program`), looped `--iterations` times
(`loop`) and read by `circuit_properties.work.read` at `--k`, its point's test (`points.check`) among it.

    python -m training_slowdown.program_reading [--size s] [--batch 2] [--ctx 1024] [--vocab 32000] [--iterations 2]
        [--k 65536] [--out OUT.json]

The defaults are the start `s-t20`'s shape (`starts.toml`): size s (d 512, 6 layers, 8 heads, d_ff 1408), ctx 1024,
vocab 32000, at a batch of 2 rows of ctx + 1 tokens. It prints one line, `program_reading` and a JSON object, and writes the object to `--out`: the shape; the
build's seconds and the resident memory after it; the reading's seconds before the point's pass and the pass's own;
the pass's peak resident memory (from the pass's start, where the system can reset it) and the run's; the pass's
primitive words, its points and word-by-word points, and its claims; the refusal's words or None; and the reading's
N and |C| in gates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LF_PATH = ROOT / "adapters" / "vllm" / "tests" / "program" / "train_step_lf.py"
sys.path.append(str(ROOT / "benchmarks" / "circuit_properties"))
from circuit_properties import points as PT  # noqa: E402
from circuit_properties import work as WK  # noqa: E402

from verity.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.circuits.program import Program  # noqa: E402
from verity.circuits.refs import Coll, concat, tuple_of  # noqa: E402
from verity.circuits.types import Array, Tuple  # noqa: E402
from verity_vllm.program.frontend.train_step import AdamW, step_program  # noqa: E402

#: a Program's nesting is not bounded by Python's default recursion limit, so the reading runs in a thread of this
#: stack, at this limit
STACK, RECURSION = 2**30, 1_000_000


def lf():
    """The lf model's module, loaded by its path."""
    spec = importlib.util.spec_from_file_location("training_slowdown_program_reading_lf", LF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def loop(step: Program, n: int) -> Program:
    """``step`` (`step_program`'s: θ, the two moments, three scalars, the buffers, the batch) as the body of a scan of
    ``n`` iterations."""
    params, ret = step.fn.signature()
    P = len(step.theta)
    state, xs, shared = params[:3 * P], params[3 * P:3 * P + 3] + params[-1:], params[3 * P + 3:-1]
    carry_t, loss_t = Tuple(*(t for _, t in state)), ret.fields[0]

    def body(B, S, carry, *rest):
        x, sh = rest[:len(xs)], rest[len(xs):]
        out = B.call(step.fn, *(carry[i] for i in range(len(state))), *x[:3], *sh, x[3])
        return tuple_of(Coll(carry_t, concat([out[1 + i].refs for i in range(len(state))])), out[0])

    it = CompositeDefinition("Iteration", 1, (), lambda S: ((("carry", carry_t), *xs, *shared), Tuple(carry_t, loss_t)),
                             body, register=False)
    sig = ((*state, *((name, Array(n, t)) for name, t in xs), *shared), Tuple(carry_t, Array(n, loss_t)))

    def run(B, S, *a):
        init = Coll(carry_t, concat([c.refs for c in a[:len(state)]]))
        return B.scan(bind(it), init, xs=list(a[len(state):len(state) + len(xs)]), shared=list(a[len(state) + len(xs):]))

    return Program(bind(CompositeDefinition("Loop", 1, (), lambda S: sig, run, register=False)), theta=step.theta)


def _resident_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize() / 2**30


def _peak_gib() -> float | None:
    """The resident peak since the last reset (`_reset_peak`), or None where the status has no `VmHWM` line."""
    with open("/proc/self/status") as f:
        hwm = next((int(x.split()[1]) for x in f if x.startswith("VmHWM:")), None)
    return None if hwm is None else hwm / 2**20


def _reset_peak() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def reading(size: str = "s", batch: int = 2, ctx: int = 1024, vocab: int = 32000, iterations: int = 2,
            k: float = 65536.0) -> dict:
    """The figures the command prints, for the step of that shape looped ``iterations`` times, read at ``k``."""
    LF = lf()
    out: dict = {"size": size, "batch": batch, "ctx": ctx, "vocab": vocab, "iterations": iterations, "k": k}
    t0 = time.monotonic()
    model, x = LF.meta_step(size, batch=batch, ctx=ctx, vocab=vocab)
    program = loop(step_program(model, LF.loss_fn, AdamW(**LF.ADAMW), x), iterations)
    out["build_s"], out["resident_after_build_gib"] = round(time.monotonic() - t0, 1), round(_resident_gib(), 2)
    real, t1 = PT.check, time.monotonic()

    def check(*a):
        t2 = time.monotonic()
        out["read_before_pass_s"] = round(t2 - t1, 1)
        reset = _reset_peak()
        try:
            t = real(*a)
            out.update(pass_words=t.words, points=t.points, word_points=t.word_points, claims=t.claims)
            return t
        finally:
            peak = _peak_gib() if reset else None
            out.update(pass_s=round(time.monotonic() - t2, 1), pass_peak_gib=None if peak is None else round(peak, 2))

    PT.check = check
    try:
        w = WK.read(program, k)
        out.update(refused=None, N=w.N, gates=w.gates)
    except WK.RepeatedRows as e:
        out.update(refused=str(e), N=None, gates=None)
    finally:
        PT.check = real
    out["read_s"] = round(time.monotonic() - t1, 1)
    out["peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training_slowdown.program_reading", description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", default="s")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--k", type=float, default=65536.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    got: dict = {}

    def run():
        got["out"] = reading(a.size, a.batch, a.ctx, a.vocab, a.iterations, a.k)

    sys.setrecursionlimit(RECURSION)
    threading.stack_size(STACK)
    th = threading.Thread(target=run)
    th.start()
    th.join()
    if "out" not in got:
        return 1
    line = json.dumps(got["out"])
    if a.out is not None:
        a.out.write_text(line + "\n")
    print("program_reading", line, flush=True)
    return 0 if got["out"]["refused"] is None else 2


if __name__ == "__main__":
    sys.exit(main())
