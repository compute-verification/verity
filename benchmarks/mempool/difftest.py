"""The RAM pool (`pool_ram`) against vLLM's (the replay's `BlockPool`) over the encodings' traces and synthetic ones, its memory
check on each run's log, and the fingerprint on an honest log and on tampered ones. Prints each run's operation counts.

    python3 difftest.py [--traces DIR] [--seeds N] [--json OUT]
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import random
from collections import Counter
from pathlib import Path

import pool_ram as PR
from pool_ram import RP

FIXTURES = Path(__file__).resolve().parents[1] / "network_e2e" / "encodings" / "tests" / "fixtures"


def load(path: Path) -> tuple[RP.Engine, list[RP.RequestIO], list[int]]:
    """A step trace, or a derived-format one (`*.json.gz`), as replay input (`cache_pool_vectors.replay_input`'s)."""
    if path.suffix != ".gz":
        tr = RP.read_any(path)
        return tr.engine, list(tr.requests), list(tr.before_step)
    t = json.loads(gzip.open(path).read())
    c = t["config"]
    eng = RP.Engine(max_num_seqs=c["max_num_seqs"], max_num_batched_tokens=c["max_num_batched_tokens"],
                    max_model_len=c["max_model_len"], num_gpu_blocks=c["num_gpu_blocks"], block_size=c["block_size"],
                    long_prefill_token_threshold=c["long_prefill_token_threshold"],
                    enable_prefix_caching=c["enable_prefix_caching"], async_scheduling=c["scheduler"] == "async",
                    watermark=c["watermark"])
    first: dict[int, int] = {}
    for k, s in enumerate(t["steps"]):
        for x in s["rows"]:
            first.setdefault(x["request"], k)
    ends = [s["t_end_s"] for s in t["steps"]]
    reqs, before = [], []
    for i, r in enumerate(t["requests"]):
        out = r["output_ids"]
        reqs.append(RP.RequestIO(prompt_len=len(r["prompt_ids"]), max_tokens=r["max_tokens"], output_len=len(out),
                                 eos=r["finish"] == "eos", tokens=tuple(r["prompt_ids"]) + tuple(out)))
        before.append(min(bisect.bisect_left(ends, r["arrival_s"]) + 1, first[i]))
    return eng, reqs, before


def synthetic(seed: int, *, n: int = 120, NB: int = 48, B: int = 16, R: int = 8, T: int = 128, asy: bool = True,
              wm: float = 0.0, prefixes: int = 4) -> tuple[RP.Engine, list[RP.RequestIO], list[int]]:
    """Requests sharing a few system prompts and earlier conversations, on a pool small enough to evict and preempt."""
    rng = random.Random(seed)
    L = min(1024, (NB - 1) * B // 2)
    tok = lambda k: tuple(rng.randrange(1, 40) for _ in range(k))  # noqa: E731
    sys_prompts = [tok(rng.randrange(4, 3 * B)) for _ in range(prefixes)]
    reqs, admit, seen, step = [], [], [], 0
    for _ in range(n):
        u = rng.random()
        base = rng.choice(seen) if seen and u < 0.3 else rng.choice(sys_prompts) if u < 0.85 else ()
        prompt = (base + tok(rng.randrange(1, 4 * B)))[: L - 40]
        out_len = rng.randrange(1, 30)
        eos = rng.random() < 0.6
        reqs.append(RP.RequestIO(len(prompt), out_len + (rng.randrange(0, 8) if eos else 0), out_len, eos,
                                 prompt + tok(out_len)))
        seen.append(prompt + reqs[-1].tokens[len(prompt):])
        step += rng.choice((0, 0, 1, 2, 4))
        admit.append(step)
    eng = RP.Engine(max_num_seqs=R, max_num_batched_tokens=T, max_model_len=L, num_gpu_blocks=NB, block_size=B,
                    enable_prefix_caching=True, async_scheduling=asy, watermark=wm)
    return eng, reqs, admit


def fingerprints(ram: PR.Ram, rng: random.Random, *, full: bool = True) -> dict[str, bool]:
    """Whether the fingerprints agree on the honest log, and on logs with one read's value or time changed, or one
    access dropped. Without `full`, over the two multisets' differences (the common records' factors cancel)."""
    alpha, gamma = rng.getrandbits(128), rng.getrandbits(128)

    def agree(r: PR.Ram) -> bool:
        rs, ws = r.multisets()
        if not full:
            rs, ws = rs - ws, ws - rs
        return PR.fingerprint(rs, alpha, gamma) == PR.fingerprint(ws, alpha, gamma)

    out = {"honest": agree(ram)}
    j = rng.randrange(len(ram.log))
    a, v, t, w, now = ram.log[j]
    bad = PR.Ram(ram.initial)
    bad.cells, bad.first = ram.cells, ram.first
    if isinstance(v, int):
        v2 = v ^ 1
    elif isinstance(v, tuple) and hasattr(v, "_replace"):
        v2 = v._replace(ref=v.ref + 1)
    else:
        v2 = (v[0] ^ 1, v[1])
    bad.log = ram.log[:j] + [(a, v2, t, w, now)] + ram.log[j + 1:]
    out["read value changed"] = agree(bad)
    bad.log = ram.log[:j] + [(a, v, max(t - 1, 0) if t else 1, w, now)] + ram.log[j + 1:]
    out["read time changed"] = agree(bad)
    bad.log = ram.log[:j] + ram.log[j + 1:]
    out["access dropped"] = agree(bad)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", type=Path, default=FIXTURES)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs: list[tuple[str, tuple, bool]] = []
    for name in ("async-prefix-cache.jsonl", "async-crush.json.gz", "sync-crush.json.gz"):
        p = a.traces / name
        if p.exists():
            runs.append((name, load(p), False))
    for s in range(a.seeds):
        rng = random.Random(1000 + s)
        kw = {"NB": rng.choice((24, 48, 96)), "B": rng.choice((4, 16)), "R": rng.choice((4, 8, 16)),
              "T": rng.choice((32, 128, 256)), "asy": rng.random() < 0.5, "wm": rng.choice((0.0, 0.0, 0.05))}
        runs.append((f"synthetic {s} {kw}", synthetic(s, **kw), True))
    report = []
    total = Counter()
    for name, (eng, reqs, admit), arrivals in runs:
        how = {"decide": lambda i, k, o, admit=admit: admit[i] <= k} if arrivals else {"admit": admit}
        m = PR.run(eng, reqs, **how)
        fp = fingerprints(m.pool.ram, random.Random(7), full=not report) if m.pool else {}
        assert fp.get("honest", True) and not any(v for k, v in fp.items() if k != "honest"), (name, fp)
        ops = m.pool.ops if m.pool else Counter()
        per_step = max((sum(s.values()) for s in m.steps), default=0)
        row = {"run": name, "caching": eng.enable_prefix_caching, "NB": eng.num_gpu_blocks, "B": eng.block_size,
               **m.counts, "ops": dict(ops), "accesses": len(m.pool.ram.log) if m.pool else 0,
               "max_ops_per_step": per_step, "fingerprint": fp}
        report.append(row)
        total.update(ops)
        total["runs"] += 1
        total["accesses"] += row["accesses"]
        total["steps"] += m.counts["steps"]
        total["tokens"] += m.counts["tokens"]
        print(f"{name}: {m.counts['steps']} steps, {m.counts['tokens']} tokens, ops {dict(ops)}, "
              f"{row['accesses']} accesses, max {per_step} ops in a step; agrees, memory check and fingerprints pass")
    print("total", dict(total))
    if a.json:
        a.json.write_text(json.dumps({"runs": report, "total": dict(total)}, indent=1, default=str))


if __name__ == "__main__":
    main()
