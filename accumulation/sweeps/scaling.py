"""Extraction-time / memory scaling probes, each in a guarded subprocess (see :mod:`accumulation.sweeps.guard`).

    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.scaling                 # default probe list -> results/extract_scaling.json
    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.scaling --max-rss-mb 2048 --timeout 300

Every probe is ``build + extract`` of one ``(alg, model, workload)`` in a child process (``--child``), polled for
RSS every 0.5 s; the parent records ``ok | timeout | oom-guard | error`` with elapsed time, peak RSS and op count.
Probes outside :data:`accumulation.sweeps.guard.SAFE_LIMITS` are recorded as ``skipped`` unless ``--unsafe``
is given.  Results are appended to ``results/extract_scaling.json`` (deduplicated by probe key), so the list
doubles as the regression set for the extractor: every ``timeout`` / ``oom-guard`` row is a case that must
become ``ok``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from accumulation.configs import MODELS, Workload
from accumulation.sweeps.guard import SAFE_LIMITS, check_limits, run_guarded, sequences_per_forward
from accumulation.sweeps.util import RESULTS_DIR, write_json

REPO = Path(__file__).resolve().parents[2]

#: (alg, model, tokens, seq, extra) -- ordered so that within a group the cheap case comes first
DEFAULT_PROBES: list[tuple] = [
    # one sequence per forward at increasing model size (the regime the sweeps use)
    ("pretrain-dense", "llama3-1b", 4096, 4096, {}),
    ("pretrain-dense", "llama3-8b", 4096, 4096, {}),
    ("pretrain-dense", "llama3-70b", 4096, 4096, {}),
    ("pretrain-dense", "llama3-405b", 4096, 4096, {}),
    ("pretrain-moe", "mixtral-8x7b", 4096, 4096, {}),
    ("pretrain-moe", "deepseek-v3", 4096, 4096, {}),
    ("pretrain-moe", "qwen3-235b-a22b", 4096, 4096, {}),
    # sequences per forward at fixed small seq (8B): 1, 2, 4, 8, 16
    ("pretrain-dense", "llama3-8b", 512, 512, {}),
    ("pretrain-dense", "llama3-8b", 1024, 512, {}),
    ("pretrain-dense", "llama3-8b", 2048, 512, {}),
    ("pretrain-dense", "llama3-8b", 4096, 512, {}),
    ("pretrain-dense", "llama3-8b", 8192, 512, {}),
    # sequences per forward at seq 1024 / 2048 (8B)
    ("pretrain-dense", "llama3-8b", 1024, 1024, {}),
    ("pretrain-dense", "llama3-8b", 2048, 1024, {}),
    ("pretrain-dense", "llama3-8b", 4096, 1024, {}),
    ("pretrain-dense", "llama3-8b", 2048, 2048, {}),
    ("pretrain-dense", "llama3-8b", 4096, 2048, {}),
    # the cases that blew up before the extractor fix (regression set): 2 x 4096 at 8B (and 1B), forward-only too
    ("pretrain-dense", "llama3-1b", 8192, 4096, {}),
    ("inference-dense", "llama3-8b", 8192, 4096, {}),
    ("pretrain-dense", "llama3-8b", 8192, 4096, {}),
    # ES population (members are separate forwards, one sequence each) and chained local steps
    ("es", "llama3-8b", 8192, 4096, {"population": 2}),
    ("es", "llama3-8b", 8192, 1024, {"population": 8}),
    ("local-sgd", "llama3-8b", 8192, 4096, {"local_steps": 2}),
    # post-fix Q scaling at 8B (seq 4096 and 8192) up to the Llama-3 8B batch, then 405B / 70B-LoRA / MoE / RL
    ("pretrain-dense", "llama3-8b", 65536, 4096, {}),
    ("pretrain-dense", "llama3-8b", 1 << 20, 4096, {}),
    ("pretrain-dense", "llama3-8b", 1 << 20, 8192, {}),
    ("pretrain-dense", "llama3-8b", 4_194_304, 8192, {}),
    ("pretrain-dense", "llama3-8b", 4_194_304, 4096, {}),
    ("pretrain-dense", "llama3-405b", 4_194_304, 8192, {}),
    ("lora", "llama3-70b", 1 << 20, 4096, {"rank": 16}),
    ("pretrain-moe", "deepseek-v3", 65536, 4096, {}),
    ("pretrain-moe", "mixtral-8x7b", 1 << 20, 4096, {}),
    ("rl-policy-gradient", "llama3-8b", 65536, 4096, {"gen_tokens": 2048}),
    ("local-sgd", "llama3-8b", 65536, 4096, {"local_steps": 4}),
    ("es", "llama3-8b", 64 * 4096, 4096, {"population": 64}),
    ("es", "llama3-8b", 1024 * 4096, 4096, {"population": 1024}),
    ("pretrain-dense", "llama3-8b", 16_777_216, 8192, {}),
]


def probe_key(p: dict) -> tuple:
    return (p["alg"], p["model"], p["tokens"], p["seq"], p.get("population", 1), p.get("local_steps", 1), p.get("rank", 16),
            p.get("gen_tokens"))


def child_main(alg: str, model: str, tokens: int, seq: int, extra: dict) -> None:
    """``--child`` mode: build + extract once, print one JSON line."""
    from accumulation.algorithms.registry import build
    from accumulation.graph import extract
    t0 = time.perf_counter()
    bp = build(alg, MODELS[model], Workload(tokens=tokens, seq=seq, **extra))
    t1 = time.perf_counter()
    g = extract(bp)
    t2 = time.perf_counter()
    print(json.dumps({"t_build": round(t1 - t0, 3), "t_extract": round(t2 - t1, 3), "ops": len(g.ops), "tensors": len(g.tensors)}))


def run_probe(alg: str, model: str, tokens: int, seq: int, extra: dict, *, timeout_s: float, max_rss_mb: float,
              unsafe: bool = False, log=print) -> dict:
    wl = Workload(tokens=tokens, seq=seq, **extra)
    row = {"alg": alg, "model": model, "tokens": tokens, "seq": seq, **extra, "sequences_per_forward": sequences_per_forward(alg, wl),
           "status": None, "t_extract": None, "ops": None, "peak_rss_mb": None, "elapsed_s": None, "detail": "", "ts": time.time()}
    why = check_limits(alg, model, wl)
    if why and not unsafe:
        row.update(status="skipped", detail=f"outside SAFE_LIMITS: {why}")
        log(f"[scaling] {alg}/{model} Q={tokens} S={seq} {extra}: skipped ({why})")
        return row
    cmd = [sys.executable, "-m", "accumulation.sweeps.scaling", "--child", alg, model, str(tokens), str(seq)] + \
          [f"{k}={v}" for k, v in extra.items()]
    res = run_guarded(cmd, timeout_s=timeout_s, max_rss_mb=max_rss_mb, cwd=REPO, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}, log=log)
    row.update(status=res.status, peak_rss_mb=round(res.peak_rss_mb, 1), elapsed_s=round(res.elapsed_s, 2), detail=res.detail,
               rss_trace=res.rss_trace)
    if res.status == "ok":
        try:
            row.update(json.loads(res.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            row.update(status="error", detail=f"no JSON from child: {res.stderr[-300:]}")
    log(f"[scaling] {alg}/{model} Q={tokens} S={seq} {extra}: {row['status']} extract={row['t_extract']} ops={row['ops']} "
        f"peak={row['peak_rss_mb']} MB elapsed={row['elapsed_s']}s {row['detail']}")
    return row


def run_all(probes=DEFAULT_PROBES, *, out_path: Path = RESULTS_DIR / "extract_scaling.json", timeout_s: float = 300.0,
            max_rss_mb: float = 3072.0, unsafe: bool = False, log=print) -> list[dict]:
    rows: dict[tuple, dict] = {}
    if Path(out_path).exists():
        for r in json.loads(Path(out_path).read_text()):
            rows[probe_key(r)] = r
    for alg, model, tokens, seq, extra in probes:
        r = run_probe(alg, model, tokens, seq, extra, timeout_s=timeout_s, max_rss_mb=max_rss_mb, unsafe=unsafe, log=log)
        rows[probe_key(r)] = r
        write_json(out_path, list(rows.values()))
    return list(rows.values())


def regression_cases(rows: list[dict]) -> list[dict]:
    """Probes that hit the timeout or the RSS guard: the extractor fix must turn these into ``ok``."""
    return [r for r in rows if r.get("status") in ("timeout", "oom-guard")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs="+", metavar="ARG", help="internal: alg model tokens seq [k=v ...]")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-rss-mb", type=float, default=3072.0)
    ap.add_argument("--unsafe", action="store_true", help="also run probes outside SAFE_LIMITS (still guarded)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "extract_scaling.json")
    a = ap.parse_args(argv)
    if a.child:
        alg, model, tokens, seq, *kv = a.child
        child_main(alg, model, int(tokens), int(seq), {k: int(v) for k, v in (x.split("=") for x in kv)})
        return 0
    rows = run_all(out_path=a.out, timeout_s=a.timeout, max_rss_mb=a.max_rss_mb, unsafe=a.unsafe)
    bad = regression_cases(rows)
    print(f"[scaling] {len(rows)} probes; {len(bad)} regression case(s) (timeout / oom-guard):")
    for r in bad:
        print("   ", {k: r[k] for k in ("alg", "model", "tokens", "seq", "population", "local_steps", "status", "detail") if k in r})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
