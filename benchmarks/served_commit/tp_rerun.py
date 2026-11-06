"""A served window replayed in fresh engine processes, under a TP engine in eager mode (GLM-5.3 TP8, or DeepSeek-V4-Pro with
``--case``): whether serving the same window again, with the same arrivals, in a new engine on the same node reproduces it bit for
bit. It compares each forward step's composition (the schedule), every request's tokens, and on every rank, per token row, every
module's values as `tp_replay` fingerprints them (GLM-5.3's DSA top-k indices and attention core outputs, and the logits rows,
among them). Measurement only.

    bash benchmarks/served_commit/tp_rerun.sh      (its header has the knobs)

``run`` serves `tp_replay`'s arms (``--arms``, default ``served`` and ``served-2``: the window, then the same window again in the
same engine) in one engine process with `Rerun` on every rank, after one run of fillers of the window's shapes, each arm from an
empty prefix cache, as `tp_replay` does. Per arm, `arms.jsonl` has its tokens, its schedule (`tp_replay.composition` of each
forward step: [request, first position, scheduled tokens]), the top-1 - top-2 logit gap of every row the sampler read, its wall
time and `tp_replay`'s comparison with ``served`` in the same engine for every request of the arm. Every rank's fingerprints go to
``--dump`` (`<arm>.rank<r>.npz`), with the request and module names their codes index, so another process's codes don't matter.

``compare RUN RUN...`` compares each arm of each later run with the same arm of the first: tokens (with `tp_replay.flips` for a
request whose tokens differ: how many flipped, the first, and the logit gap there in both runs), schedule, and on each rank the
entries (request, position, module group) both runs have, how many are equal, those only one has, and the first that differs by
step, then module in the order the first run's forward first called it, then request and position. ``bit_exact`` is equal tokens
and schedules and, on every rank, entries compared, none differing and none one-sided. ``run`` exits 3 when the instrument failed:
`tp_replay`'s conditions, ranks that recorded different schedules, or an arm that read the prefix cache.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from served_commit import tp_replay as R

ARMS = "served,served-2"
#: a row of no step in the first run's schedule sorts after every step
NO_STEP = np.iinfo(np.int64).max


class Rerun(R.Interior):
    """`tp_replay.Interior` that writes an arm's fingerprints with the names their codes index, and its schedule on every rank."""

    def dump(self, path: Path) -> dict[str, Any]:
        self.flush()
        keys, fp, _dups = R.flatten(self.steps)
        names = sorted(self.req_code, key=self.req_code.__getitem__)
        np.savez(path, keys=keys, fp=fp, requests=np.array(names, dtype=str), groups=np.array(self.groups, dtype=str))
        return {"rank": self.rank, "entries": int(len(keys)), "schedule": [[list(x) for x in step] for step in self.schedule]}


def install(worker: Any) -> dict[str, Any]:
    """On a rank (`LLM.collective_rpc` with a callable): `Rerun` where `tp_replay.install` puts its `Interior`."""
    worker._tp_replay = Rerun(worker)
    return worker._tp_replay.install()


def end(worker: Any, *, arm: str, sampled: list[str], reference: str, dump: str) -> dict[str, Any]:
    """On a rank: the arm's fingerprints written to ``dump`` and its schedule, then `tp_replay`'s comparison with ``reference``."""
    me = worker._tp_replay
    made = me.dump(Path(dump) / f"{arm}.rank{me.rank}.npz")
    return {**me.end(arm, sampled, reference), **made}


def record(arm: str, reqs: list[dict[str, Any]], result: dict[str, Any], cached: dict[str, int], ranks: list[dict[str, Any]],
           world: int) -> tuple[dict[str, Any], list[str]]:
    """An arm's `arms.jsonl` row, from its requests, its `run_workload` result, each request's cached prompt tokens and its ranks'
    `end`, and the instrument failures they show."""
    ranks = sorted(ranks, key=lambda x: x["rank"])
    schedules = [x.pop("schedule") for x in ranks]
    gaps = next((x.pop("gaps") for x in ranks if "gaps" in x), {})
    broken = [f"{arm}: rank {x['rank']}'s schedule differs from rank {ranks[0]['rank']}'s" for x, s in zip(ranks, schedules) if s != schedules[0]]
    broken += R.cache_misuse(arm, cached)
    if arm == R.REFERENCE:
        broken += [f"rank {x['rank']}: {d} compared nothing" for x in ranks for d, v in x["per_request"].items() if not v["compared"]]
    row = {"arm": arm, "wall_s": result["wall_s"], "gpu_s": world * result["wall_s"], "steps": result["iterations"],
           "step_wall_s": result["step_wall_s"], "tokens": result["tokens"], "finish_reasons": result["finish_reasons"],
           "prompts": {q["request_id"]: len(q["prompt_token_ids"]) for q in reqs}, "cached_tokens": cached,
           "schedule": schedules[0] if schedules else [], "gaps": gaps, "ranks": ranks}
    return row, broken


def run(a: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    out, dump = Path(a.out), Path(a.dump)
    out.mkdir(parents=True, exist_ok=True)
    dump.mkdir(parents=True, exist_ok=True)
    from served_commit.tp_decode import open_engine

    E = open_engine(a, out, max_num_seqs=R.MAX_SEQS, prefix_caching=True)
    from verity_vllm.pipeline.commit import run_workload

    llm, eng, wl, world = E.llm, E.eng, E.wl, E.world
    sampling = wl["sampling"]
    vocab = int(llm.llm_engine.vllm_config.model_config.hf_config.vocab_size)
    plan = R.arms([int(t) for t in wl["requests"][0]["prompt_token_ids"]], vocab, a.long)
    names = list(plan) if a.arms == "all" else a.arms.split(",")
    if set(names) - set(plan):
        raise SystemExit(f"unknown arms {sorted(set(names) - set(plan))}: tp_replay's are {list(plan)}")
    longest = max(len(r["prompt_token_ids"]) + int(r["max_tokens"]) for n in names for r in plan[n])
    if longest > int(E.kw.get("max_model_len") or 0):
        raise SystemExit(f"max_model_len {E.kw.get('max_model_len')} is below the longest request's {longest} tokens")
    layout = {n: [[r["request_id"], len(r["prompt_token_ids"]), r["max_tokens"], r["arrive_step"]] for r in plan[n]] for n in names}
    (out / "versions.json").write_text(json.dumps({**E.versions, "arms": layout, "dump": str(dump)}, indent=1, default=str) + "\n")
    rng = np.random.default_rng(R.FILLER_SEED + 1)
    first = run_workload(eng, [R.filler(r, i, vocab, rng) for i, r in enumerate(plan[R.REFERENCE])], sampling)
    made = llm.collective_rpc(install, timeout=a.rpc_timeout)
    print(f"[tp-rerun] engine built in {E.build_s:.0f}s; fillers {first['wall_s']:.1f}s; interior on {len(made)} ranks: {made[0]}", flush=True)
    broken: list[str] = []
    with (out / "arms.jsonl").open("w") as f:
        for arm in names:
            R.cold(llm)
            llm.collective_rpc(R.begin, timeout=a.rpc_timeout)
            st = R.Steps()
            r = run_workload(eng, plan[arm], sampling, step=st)
            kw = {"arm": arm, "sampled": sorted({q["request_id"] for q in plan[arm]}), "reference": R.REFERENCE, "dump": str(dump)}
            row, bad = record(arm, plan[arm], r, st.cached, llm.collective_rpc(end, timeout=a.rpc_timeout, kwargs=kw), world)
            broken += bad
            f.write(json.dumps(row) + "\n")
            f.flush()
            differ = sum(v["differ"] + v["only_run"] for x in row["ranks"] for v in x["per_request"].values())
            print(f"[tp-rerun] {arm}: {r['wall_s']:.1f}s, {len(row['schedule'])} forwards, {sum(x['entries'] for x in row['ranks'])} "
                  f"entries; against {R.REFERENCE} in this engine: {differ} entries differ or only here", flush=True)
    left = llm.collective_rpc(R.uninstall, timeout=a.rpc_timeout)
    broken += [f"rank {x['rank']}: {x['hooks_left']} hooks, {x['patches_left']} patches left" for x in left if x["hooks_left"] or x["patches_left"]]
    wall = time.perf_counter() - t0
    doc = {"world": world, "build_s": E.build_s, "fillers_s": first["wall_s"], "wall_s": wall, "gpu_s": world * wall, "arms": names,
           "instrument_failures": broken}
    (out / "run.json").write_text(json.dumps(doc, indent=1) + "\n")
    if broken:
        print(f"[tp-rerun] instrument failed: {broken[:4]}", flush=True)
        return 3
    return 0


def index(*lists: list[str]) -> dict[str, int]:
    """Each name's index: the first list's order, then each later list's new names."""
    out: dict[str, int] = {}
    for names in lists:
        for n in names:
            out.setdefault(n, len(out))
    return out


def canon(keys: np.ndarray, fp: np.ndarray, requests: list[str], groups: list[str], req_index: dict[str, int],
          group_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """A run's keys (request, position, group in its own codes) in the shared indices, sorted, with their fingerprints."""
    rq = np.array([req_index[n] for n in requests], dtype=np.int64)
    gp = np.array([group_index[n] for n in groups], dtype=np.int64)
    k = (rq[keys >> (2 * R.FIELD)] << (2 * R.FIELD)) | (keys & (R.MASK << R.FIELD)) | gp[keys & R.MASK] if len(keys) else keys
    o = np.argsort(k, kind="stable")
    return k[o], fp[o]


def steps_of(schedule: list[list[list[Any]]], req_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Every row a schedule runs as (request << FIELD | position), sorted, and the step that ran it."""
    rows, steps = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for s, step in enumerate(schedule):
        for name, p0, n in step:
            rows.append((req_index[name] << R.FIELD) | (p0 + np.arange(n, dtype=np.int64)))
            steps.append(np.full(n, s, dtype=np.int64))
    rp, st = np.concatenate(rows), np.concatenate(steps)
    o = np.argsort(rp, kind="stable")
    return rp[o], st[o]


def compare_rank(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], rows: tuple[np.ndarray, np.ndarray],
                 req_names: list[str], group_names: list[str]) -> dict[str, Any]:
    """One rank's entries of one arm in two runs, in shared indices: counts, the first difference (by step in the first run's
    schedule, then group, request and position) and the group kinds that differ."""
    (ka, fa), (kb, fb) = a, b
    both, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    eq = (fa[ia] == fb[ib]).all(axis=1)
    d = both[~eq]
    rp, g = d >> R.FIELD, d & R.MASK
    srp, sst = rows
    step = np.full(len(rp), NO_STEP, dtype=np.int64)
    if len(srp) and len(rp):
        i = np.minimum(np.searchsorted(srp, rp), len(srp) - 1)
        hit = srp[i] == rp
        step[hit] = sst[i[hit]]
    first = None
    if len(d):
        j = np.lexsort((rp & R.MASK, rp >> R.FIELD, g, step))[0]
        first = {"step": None if step[j] == NO_STEP else int(step[j]), "request": req_names[int(rp[j] >> R.FIELD)],
                 "pos": int(rp[j] & R.MASK), "group": group_names[int(g[j])]}
    kinds: Counter = Counter()
    for grp, n in zip(*np.unique(g, return_counts=True)):
        kinds[R.kind_of(group_names[int(grp)])] += int(n)
    return {"compared": int(len(both)), "equal": int(eq.sum()), "differ": int(len(d)), "only_first": int(len(ka) - len(both)),
            "only_this": int(len(kb) - len(both)), "positions_differ": int(len(np.unique(rp))),
            "requests_differ": sorted({req_names[int(q)] for q in np.unique(rp >> R.FIELD)}), "first_differ": first,
            "kinds_differ": dict(kinds.most_common(12))}


def tokens_differ(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, int]:
    """Each request whose tokens differ, with its first differing output position."""
    out = {}
    for rid in sorted(set(a) | set(b)):
        x, y = a.get(rid) or [], b.get(rid) or []
        if rid not in a or rid not in b or x != y:
            out[rid] = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    return out


def schedule_differs(a: list[Any], b: list[Any]) -> int | None:
    """The first step at which two schedules differ, None if they don't."""
    s = next((s for s, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return s if s is not None or len(a) == len(b) else min(len(a), len(b))


def read_run(run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    rows = [json.loads(ln) for ln in (run / "arms.jsonl").read_text().splitlines() if ln.strip()]
    versions = json.loads((run / "versions.json").read_text())
    return json.loads((run / "run.json").read_text()), {r["arm"]: r for r in rows}, Path(versions["dump"])


def load(dump: Path, arm: str, rank: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    with np.load(dump / f"{arm}.rank{rank}.npz") as z:
        return z["keys"], z["fp"], [str(x) for x in z["requests"]], [str(x) for x in z["groups"]]


def compare_arm(a: dict[str, Any], dump_a: Path, b: dict[str, Any], dump_b: Path) -> dict[str, Any]:
    """One arm in two runs: tokens, schedule, and every rank both ran."""
    ranks = sorted({x["rank"] for x in a["ranks"]} & {x["rank"] for x in b["ranks"]})
    toks = tokens_differ(a["tokens"], b["tokens"])
    gaps = [{rid: {int(p): g for p, g in v.items()} for rid, v in x["gaps"].items()} for x in (a, b)]
    flips = {rid: v for rid, v in R.flips(a["tokens"], b["tokens"], *gaps, a["prompts"]).items() if not v["tokens_equal"]}
    sched = schedule_differs(a["schedule"], b["schedule"])
    per: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        A, B = load(dump_a, a["arm"], rank), load(dump_b, b["arm"], rank)
        names = [x[0] for s in (a["schedule"], b["schedule"]) for step in s for x in step]
        req_index, group_index = index(A[2], B[2], names, sorted(a["tokens"]), sorted(b["tokens"])), index(A[3], B[3])
        req_names, group_names = list(req_index), list(group_index)
        per[rank] = compare_rank(canon(*A, req_index, group_index), canon(*B, req_index, group_index), steps_of(a["schedule"], req_index),
                                 req_names, group_names)
    firsts = [(r, v["first_differ"]) for r, v in per.items() if v["first_differ"]]
    order = lambda rf: (NO_STEP if rf[1]["step"] is None else rf[1]["step"], rf[0])  # noqa: E731
    first = min(firsts, key=order) if firsts else None
    total = {k: sum(v[k] for v in per.values()) for k in ("compared", "equal", "differ", "only_first", "only_this")}
    kinds: Counter = Counter()
    for v in per.values():
        kinds.update(v["kinds_differ"])
    exact = (not toks and sched is None and len(ranks) == len(a["ranks"]) == len(b["ranks"])
             and all(v["compared"] and not (v["differ"] or v["only_first"] or v["only_this"]) for v in per.values()))
    return {"bit_exact": bool(exact and ranks), "tokens_differ": toks, "flips": flips, "requests": len(set(a["tokens"]) | set(b["tokens"])),
            "schedule_first_differ": sched, "forwards": [len(a["schedule"]), len(b["schedule"])], "ranks": len(ranks), **total,
            "ranks_differ": sum(1 for v in per.values() if v["differ"] or v["only_first"] or v["only_this"]),
            "first_differ": None if first is None else {"rank": first[0], **first[1]}, "kinds_differ": dict(kinds.most_common(12)),
            "per_rank": {str(r): v for r, v in per.items()}}


def in_engine(row: dict[str, Any], ref: dict[str, Any] | None) -> dict[str, Any]:
    """`tp_replay`'s comparison of an arm with ``served`` in the same engine, summed over ranks and requests."""
    s = {k: sum(v[k] for x in row["ranks"] for v in x["per_request"].values()) for k in ("compared", "equal", "differ", "only_ref", "only_run")}
    toks = tokens_differ({k: v for k, v in ref["tokens"].items() if k in row["tokens"]}, row["tokens"]) if ref else None
    return {**s, "tokens_differ": toks, "bit_exact": bool(ref and not toks and s["compared"] and not (s["differ"] or s["only_run"]))}


def compare(runs: list[Path], out: Path | None) -> int:
    meta, arms, dumps = zip(*(read_run(r) for r in runs))
    doc: dict[str, Any] = {"runs": [str(r) for r in runs], "instrument_failures": {str(r): m["instrument_failures"] for r, m in zip(runs, meta)},
                           "gpu_s": {str(r): {"total": m["gpu_s"], "build_s": m["build_s"], **{k: v["gpu_s"] for k, v in a.items()}}
                                     for r, m, a in zip(runs, meta, arms)},
                           "in_engine": {str(r): {k: in_engine(v, a.get(R.REFERENCE)) for k, v in a.items() if k != R.REFERENCE}
                                         for r, a in zip(runs, arms)},
                           "across": {}}
    for r, a, dump in zip(runs[1:], arms[1:], dumps[1:]):
        doc["across"][str(r)] = {k: compare_arm(arms[0][k], dumps[0], v, dump) for k, v in a.items() if k in arms[0]}
    for r, res in doc["across"].items():
        for arm, v in res.items():
            fl = {rid: f"{x['flipped']} from output {x['first_flip']}, gap {x['gap_ref']} vs {x['gap_run']}" for rid, x in v["flips"].items()}
            print(f"[tp-rerun] {r} vs {runs[0]}, {arm}: bit-exact {v['bit_exact']} (tokens differ {fl or v['tokens_differ'] or 'none'}; schedule "
                  f"{'equal' if v['schedule_first_differ'] is None else 'differs at step ' + str(v['schedule_first_differ'])}, "
                  f"{v['forwards'][0]} forwards; {v['differ']} of {v['compared']} entries differ on {v['ranks_differ']} of {v['ranks']} ranks, "
                  f"{v['only_first'] + v['only_this']} one-sided; first {v['first_differ']})", flush=True)
    for r, res in doc["in_engine"].items():
        for arm, v in res.items():
            print(f"[tp-rerun] {r}, {arm} vs {R.REFERENCE} in one engine: bit-exact {v['bit_exact']} ({v['differ']} of {v['compared']} differ, "
                  f"{v['only_run']} only in {arm}; tokens differ {v['tokens_differ'] or 'none'})", flush=True)
    if out is not None:
        out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="a served window replayed in fresh engine processes, compared bit for bit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="serve tp_replay's arms in this engine process and write their tokens, schedules and fingerprints")
    r.add_argument("--workload", required=True, help="the row (its first request's prompt is every request's tile)")
    r.add_argument("--out", required=True)
    r.add_argument("--dump", required=True, help="where every rank's fingerprints go (about 100 MB a rank and arm for GLM-5.3)")
    r.add_argument("--arms", default=ARMS, help=f"tp_replay's arms to serve, comma-separated, or all (default {ARMS})")
    r.add_argument("--case", default=None)
    r.add_argument("--checkpoints", default="manifests/checkpoints.json")
    r.add_argument("--tp", type=int, default=8)
    r.add_argument("--long", type=int, default=R.LONG, help="d-long's prompt length")
    r.add_argument("--engine-arg", action="append", default=[])
    r.add_argument("--rpc-timeout", type=float, default=3600.0)
    c = sub.add_parser("compare", help="compare each later run's arms with the first run's")
    c.add_argument("runs", nargs="+", type=Path)
    c.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    raise SystemExit(run(a) if a.cmd == "run" else compare(a.runs, a.out))


if __name__ == "__main__":
    from served_commit.tp_rerun import main as _main

    _main()
