"""Each Inductor kernel's autotune in an engine's process, in order of first launch: its name, its source hash, the
config it ran and, where it had more than one, the timings it was chosen by (`CachingAutotuner.benchmark_all_configs`,
which picks the fastest).  With PICK `first` or `last` the choice is made untimed (the first or last config), so two
processes compiling the same kernel run the same config whatever their timings; a kernel with one config, or one read
from the autotune cache, is never timed and its entry has no `timings_ms`.  The log never raises into Inductor: an entry
it can't read records the error.

    log = picks.install(os.environ.get("SBENCH_PICK"))   # before the engine is built, in its process
"""
from __future__ import annotations

import hashlib

PICKS = ("first", "last")


def install(pick: str | None) -> list[dict]:
    if pick not in (None, "", *PICKS):
        raise ValueError(f"SBENCH_PICK: one of {PICKS} or unset, not {pick!r}")
    from torch._inductor.runtime import triton_heuristics as th
    log: list[dict] = []
    seen: dict[int, tuple] = {}   # id -> (autotuner, entry): the autotuner kept, so its id is never reused
    orig_run, orig_bench = th.CachingAutotuner.run, th.CachingAutotuner.benchmark_all_configs

    def entry(self_) -> dict:
        got = seen.get(id(self_))
        if got is None:
            try:
                fn = getattr(self_, "fn", None)
                src = getattr(fn, "src", "") or ""
                e = {"kernel": (getattr(self_, "inductor_meta", None) or {}).get("kernel_name") or getattr(fn, "__name__", "?"),
                     "src": hashlib.sha1(src.encode()).hexdigest()[:10]}
            except Exception as x:  # noqa: BLE001
                e = {"kernel": "?", "error": repr(x)}
            got = seen[id(self_)] = (self_, e)
            log.append(e)
        return got[1]

    def bench(self_, *args, **kw):
        launchers = list(getattr(self_, "launchers", None) or ())
        if pick in PICKS and launchers:
            n = len(launchers)
            t = {ln: float(i if pick == "first" else n - i) for i, ln in enumerate(launchers)}
        else:
            t = orig_bench(self_, *args, **kw)
        try:
            e = entry(self_)
            e["timings_ms"] = [[str(ln.config), round(float(v), 5)] for ln, v in t.items()]
            e["ran"] = str(min(t.items(), key=lambda kv: kv[1])[0].config) if t else None
        except Exception as x:  # noqa: BLE001
            entry(self_)["error"] = repr(x)
        return t

    def run(self_, *args, **kw):
        out = orig_run(self_, *args, **kw)
        try:
            e = entry(self_)
            launchers = getattr(self_, "launchers", None) or ()
            if "ran" not in e and len(launchers) == 1:
                e["ran"] = str(launchers[0].config)
        except Exception as x:  # noqa: BLE001
            entry(self_)["error"] = repr(x)
        return out

    th.CachingAutotuner.benchmark_all_configs = bench
    th.CachingAutotuner.run = run
    return log


def summary(log: list[dict]) -> dict:
    """The log's kernels that chose among configs, and a digest of every (kernel, src, ran) to compare processes by."""
    timed = [e for e in log if len(e.get("timings_ms") or ()) > 1]
    key = "\n".join(f"{e.get('kernel')} {e.get('src')} {e.get('ran')}" for e in log)
    return {"kernels": len(log), "timed": len(timed), "errors": sum("error" in e for e in log),
            "digest": hashlib.sha256(key.encode()).hexdigest()[:16]}
