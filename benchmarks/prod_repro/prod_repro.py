"""vLLM at its serving settings, replayed: what one served job shows, as the settings it ran with and three lines (README.md).

    prod_repro.py arms
    prod_repro.py report OUT --stock ARM --ours ARM [--again ARM] --load NAME [--rep 1] [--replay DIR]
                         [--target PERCENT] [--json]

``report`` reads a finished job's out/ as the serving cost benchmark leaves it (`benchmarks/serve_tax`: a directory an
arm, and its replay's), prints the report and writes it to OUT/prod_repro.json and OUT/prod_repro.txt:

- the settings each of the two arms ran with, read from its launch line, its engine's own configuration (the step
  trace's header) and its counters: a setting the serve did not turn on is ``off``, and one its files can't confirm
  says so;
- ``tokens``: the served window the replay re-ran (`replay.window`), how many of its requests and served tokens the
  replay's kept tokens were compared with (`replay.token_diff`) and how many differ, with the steps that differ and
  the requests of the window that began from a prefix-cache hit;
- ``cost``: what serving with our changes (``--ours``) costs over stock (``--stock``) on the load, a share of stock's
  serving time with its 95% interval, beside stock served a second time (``--again``), the noise floor: by vLLM's
  client the benchmark's paired ratio at the rung where stock is saturated (`serve_tax.serving_cost`), by the corpus
  client its tax over reps (`serve_tax.tax`), which is a cost only where stock did not keep up with the arrivals;
- ``advice``: the bits a request by class, with each class's events.

A line is measured from the job's files, or it says ``not measured`` and why. Nothing here measures: the metrics, the
cost and its interval and the comparison are the benchmark's functions. It exits 1 where the replay differs from what
was served, naming each differing request and its first position, and 0 otherwise: a line that is not measured is not a
failure, and the last line counts the lines that are measured.

``arms`` prints the three arms a job serves by default: the arms at vLLM's published serving settings where the
benchmark has them (`PROD`), else stock compiled vLLM, stock again and the step record (`TODAY`).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/serve_tax"))
import replay as R  # noqa: E402 (benchmarks/serve_tax, put on sys.path above)
import serve_tax as S  # noqa: E402

SCHEMA = "verity-bench/prod-repro/v0"
#: the arms a job serves by default, stock, stock again and ours: the arms at vLLM's published serving settings where
#: `serve_tax.ARMS` has them
PROD = ("prod", "prod-again", "prod-record")
TODAY = ("compiled", "compiled-again", "record")
#: the most serving with our changes may cost over stock, in percent of stock's serving time
TARGET = 1.0
#: the keys of a checkpoint's config.json that bound its context, the smallest of which is the model's own limit
CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length",
                "max_seq_length", "seq_len")
#: how the replayed window was chosen: no audit draws it yet
WINDOW = "every request of the load's rep, a window the command names: no audit's coins drew it"
NO_ADVICE = ("nothing in this job computes it: the advice code (verity_vllm.pipeline.advice_code) codes a serve under the "
             "network certifier's record and the certified tick, and this command runs neither it nor a split of its "
             "bits by class")
NO_CIRCUIT = "no: nothing in this job compiles the settings' circuit"
ON, OFF, UNCONFIRMED = "on", "off", "not confirmed"
#: the cost's verdict where stock and ours did not serve at the same settings: the figure is then not our changes' cost
UNLIKE = "the arms are not at the same settings"


def default_arms() -> tuple[str, str, str]:
    return PROD if all(a in S.ARMS for a in PROD) else TODAY


def not_measured(why: str, **more: object) -> dict:
    return {"measured": False, "why": why, **more}


# ------------------------------------------------------------------------------------------------------- the settings

def flag(argv: list[str], name: str) -> str | None:
    """The value a launch line gives the flag ``name`` (``--name V`` or ``--name=V``), the last one; None without it."""
    out = None
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            out = argv[i + 1]
        elif a.startswith(name + "="):
            out = a.partition("=")[2]
    return out


def flags(argv: list[str], prefixes: tuple[str, ...]) -> str:
    """Every flag of a launch line that begins with one of ``prefixes``, each with the value the line gives it."""
    out = []
    for i, a in enumerate(argv):
        if a.startswith(prefixes):
            value = "=" not in a and i + 1 < len(argv) and not argv[i + 1].startswith("--")
            out.append(f"{a} {argv[i + 1]}" if value else a)
    return " ".join(out)


def trace_header(arm_dir: Path) -> dict:
    """The engine's own configuration, from the header of an arm's step trace; empty for an arm that kept none."""
    traces = S.step_traces(arm_dir)
    if not traces:
        return {}
    with (gzip.open if traces[0].suffix == ".gz" else open)(traces[0], "rt", encoding="utf-8") as f:
        first = json.loads(f.readline() or "{}")
    return first.get("config") or {} if first.get("kind") == "header" else {}


def model_limit(checkpoint: Path) -> dict | None:
    """The model's own context limit: the smallest of `CONTEXT_KEYS` in the checkpoint's config.json, with its key.
    None where this machine has no such file or it names none."""
    f = checkpoint / "config.json"
    if not f.is_file():
        return None
    cfg = json.loads(f.read_text())
    found = [(int(cfg[k]), k) for k in CONTEXT_KEYS if isinstance(cfg.get(k), int)]
    return {"tokens": min(found)[0], "key": min(found)[1]} if found else None


def reps_of(arm: dict, load: str) -> list[dict]:
    """An arm's reps of ``load``, by either client (vLLM's names a pass by its load and rung)."""
    return [r for name, reps in arm["loads"].items() if load in (name, name.rpartition("@")[0]) for r in reps]


def settings(arm_dir: Path, arm: dict, load: str) -> dict:
    """What one arm's serve ran with: each setting's state and what says so."""
    meta = json.loads((arm_dir / "meta.json").read_text())
    argv, header = meta["server_argv"], trace_header(arm_dir)
    shares = [r["prompt_cached_share"] for r in reps_of(arm, load) if r.get("prompt_cached_share") is not None]
    share = sum(shares) / len(shares) if shares else None
    served = "" if share is None else f"; the cache served {100 * share:.1f}% of the load's prompt tokens"
    if "--no-enable-prefix-caching" in argv or header.get("enable_prefix_caching") is False:
        cache = (OFF, "--no-enable-prefix-caching" if "--no-enable-prefix-caching" in argv else "the engine's configuration")
    elif header.get("enable_prefix_caching"):
        cache = (ON, "the engine's configuration" + served)
    elif share:
        cache = (ON, served[2:])
    else:
        cache = (UNCONFIRMED, "the launch line does not turn it off, and the arm kept no step trace and counted no hit")
    spec, kv = flags(argv, ("--speculative", "--num-speculative")), flag(argv, "--kv-cache-dtype")
    limit = header.get("max_model_len") or flag(argv, "--max-model-len")
    own = model_limit(Path(meta["checkpoint"])) if meta.get("checkpoint") else None
    return {"prefix_caching": {"state": cache[0], "by": cache[1], "cached_share": share},
            "speculative_decoding": {"state": ON if spec else OFF, "by": spec or "no --speculative-config in the launch line"},
            "kv_cache": {"state": kv or "the engine's default",
                         "by": "--kv-cache-dtype" if kv else "no --kv-cache-dtype in the launch line"},
            "context_limit": {"tokens": int(limit) if limit is not None else None, "model": own,
                              "models_own": None if limit is None or own is None else int(limit) >= own["tokens"]},
            "compiled": {"state": OFF if "--enforce-eager" in argv else ON,
                         "by": "--enforce-eager" if "--enforce-eager" in argv else "no --enforce-eager in the launch line"}}


def settings_differ(a: dict, b: dict) -> list[str]:
    """The settings two arms' serves differ in."""
    names = {"prefix_caching": "prefix caching", "speculative_decoding": "speculative decoding",
             "kv_cache": "the KV cache's type", "compiled": "torch.compile and CUDA graphs"}
    out = [name for k, name in names.items() if a[k]["state"] != b[k]["state"]]
    return out + ["the context limit"] * (a["context_limit"]["tokens"] != b["context_limit"]["tokens"])


# --------------------------------------------------------------------------------------------------------- the tokens

def tokens(out: Path, ours: str, load: str, rep: int, replay_dir: Path | None) -> dict:
    """The tokens line: the served window of ``ours``'s ``load`` rep ``rep`` against the tokens its replay kept. Not
    measured without a replay, where the replay is of another window, or where the window can't be read back."""
    arm_dir = out / ours
    meta = json.loads((arm_dir / "meta.json").read_text())
    if replay_dir is None or not (replay_dir / "replay.json").is_file():
        return not_measured("no replay ran" + (": a pass sent by vLLM's client keeps neither its prompts' ids nor the "
                                               "served ids, so the replay has no window to rebuild"
                                               if meta.get("client") == S.BENCH else f" (no replay.json in {replay_dir})"))
    rec, run = json.loads((replay_dir / "replay.json").read_text()), meta.get("run_id")
    of = (Path(rec["arm_dir"]).name, rec["load"], rec["rep"], rec.get("served_run_id"))
    if of != (ours, load, rep, run):
        return not_measured(f"the replay in {replay_dir} is of another window (arm {of[0]}, load {of[1]}, rep {of[2]}, "
                            f"run {of[3]}), not of {ours}'s {load} rep {rep} of run {run}")
    try:
        w = R.window(arm_dir, load, rep)
        got = R.read_tokens(replay_dir)
    except (SystemExit, OSError, KeyError, ValueError) as e:
        return not_measured(f"the served window or the replay's tokens can't be read back: {type(e).__name__}: {e}")
    served = {r["engine_id"]: r["served"] for r in w["requests"]}
    diff = R.token_diff(served, got)
    hits = {rid: int(k) for s in w["steps"] for rid, k in (s.get("cache_hit") or {}).items()}
    n = {"requests": len(served), "tokens_served": sum(map(len, served.values())),
         "tokens_replayed": sum(len(got.get(rid, ())) for rid in served),
         "requests_differing": len(diff), "tokens_differing": sum(d["positions"] for d in diff)}
    if not n["tokens_served"]:
        return not_measured("the window holds no served token", **n)
    disagree = (rec["requests_tokens_differing"], rec["token_positions_differing"]) != (n["requests_differing"], n["tokens_differing"])
    return {"measured": True, **n, "differing": diff[:R.DIFFERING_MAX],
            "steps_served": rec["steps_served"], "steps_differing": rec["steps_differing"],
            "first_differing_step": rec["first_differing_step"],
            "record_disagrees": disagree,
            "reproduces": not diff and not rec["steps_differing"] and not disagree,
            "cache_hits": {"requests": len(hits), "tokens": sum(hits.values()), "longest": max(hits.values(), default=0)},
            "window": {"arm": ours, "load": load, "rep": rep, "run": run, "chosen": WINDOW},
            "served_mode": rec["served_mode"], "replay_mode": rec["mode"], "replay_run": rec.get("run_id"),
            "served_window_s": rec["served_window_s"], "replay_wall_s": rec["replay_wall_s"],
            "engine_ready_s": rec["engine_ready_s"]}


# ----------------------------------------------------------------------------------------------------------- the cost

def kept_up(arm_dir: Path, arm: dict, load: str) -> dict:
    """Whether an arm the corpus client sent ``load`` to kept up with its arrivals, as the benchmark says it of a load
    sent as it arrives (`serve_tax.KEPT_UP`): it served every request, and each rep ended within that share of the
    time its requests arrived over after the last one arrived. ``tail_s`` is the longest such end."""
    tails, all_served = [], True
    for r in arm["loads"][load]:
        arrived = json.loads((arm_dir / load / f"rep-{r['rep']}" / "meta.json").read_text())["arrivals_span_s"]
        tails.append((r["span_s"] - arrived, arrived))
        all_served &= r["ok"] == r["sent"]
    return {"kept_up": all_served and all(t <= S.KEPT_UP * a for t, a in tails), "tail_s": max(t for t, _ in tails),
            "arrivals_span_s": max(a for _, a in tails)}


def cost(out: Path, by: dict[str, dict], stock: str, ours: str, again: str | None, load: str, target: float) -> dict:
    """The cost line: ``ours`` over ``stock`` on ``load``, a fraction of stock's serving time, with its interval, the
    noise floor (``again`` over ``stock``, measured the same way) and whether it is under ``target`` percent
    (`serve_tax.verdict`, the interval widened by the floor). Not measured where an arm did not serve the load, where
    an arm left requests unserved (the arms then served different tokens), or where the corpus client's arrivals did
    not saturate stock."""
    missing = [a for a in (stock, ours) if a not in by or not reps_of(by[a], load)]
    if missing:
        return not_measured(f"{', '.join(missing)} did not serve load {load}")
    ref, mine, twin = by[stock], by[ours], by.get(again) if again else None
    unserved = {a["arm"]: sum(r["sent"] - r["ok"] for r in reps_of(a, load)) for a in (ref, mine)}
    if any(unserved.values()):
        return not_measured("the arms did not serve the same tokens: " + ", ".join(
            f"{a} left {n} requests unserved" for a, n in unserved.items() if n))
    if load in S.bench_loads(ref):
        c = S.serving_cost(ref, mine, load)
        if c is None:
            return not_measured(f"{stock} and {ours} share no rung or no rep of load {load}")
        noise = S.serving_cost(ref, twin, load, c["rung"]) if twin else None
        how = {"client": S.BENCH, "rung": c["rung"], "reps": c["reps"], "saturated": c["saturated"],
               "by": "the paired ratio of output tokens a second over reps at one rung (serve_tax.serving_cost)"}
        c = {k: c[k] for k in ("value", "ci95") if k in c}
    else:
        def tok(a: dict) -> dict:
            return S.summary([r.get("output_tok_per_s") for r in a["loads"][load]])

        c = S.tax(tok(ref), tok(mine))
        if c is None:
            return not_measured(f"{stock} or {ours} served no token of load {load}")
        noise = S.tax(tok(ref), tok(twin)) if twin and load in twin["loads"] else None
        kept = kept_up(out / stock, ref, load)
        how = {"client": S.CORPUS, "rate": ref["loads"][load][0]["rate"], "reps": c["n"], **kept,
               "by": "stock's mean output tokens a second over ours, less one, over reps (serve_tax.tax)"}
        if kept["kept_up"]:
            return not_measured(f"{stock} kept up with the arrivals (its longest rep ended {kept['tail_s']:.1f} s after "
                                f"the last of {kept['arrivals_span_s']:.1f} s of arrivals), so its tokens a second "
                                "follow the arrivals and are not a cost: send faster", reading=c, how=how)
    floor = abs(noise["value"]) if noise and noise.get("value") is not None else 0.0
    return {"measured": c["value"] is not None, "value": c["value"], "ci95": c.get("ci95"), "how": how,
            **({} if c["value"] is not None else {"why": "an arm served no token in some rep"}),
            "noise_floor": {"arm": again, "value": noise["value"], "ci95": noise.get("ci95")} if noise else None,
            "target": target, "verdict": S.verdict(c, target / 100, floor)}


# --------------------------------------------------------------------------------------------------------- the report

def report(out: Path, stock: str, ours: str, again: str | None, load: str, rep: int = 1, replay_dir: Path | None = None,
           target: float = TARGET) -> dict:
    dirs = {a: out / a for a in (stock, again, ours) if a and (out / a / "meta.json").is_file()}
    by = {a: S.arm_metrics(d) for a, d in dirs.items()}
    sets = {a: settings(dirs[a], by[a], load) for a in (stock, ours) if a in by}
    meta = json.loads((dirs[ours] / "meta.json").read_text()) if ours in dirs else {}
    line = cost(out, by, stock, ours, again, load, target)
    if len(sets) == 2 and (differ := settings_differ(sets[stock], sets[ours])):
        line["settings_differ"] = differ
        if line["measured"]:
            line["verdict"] = UNLIKE
    return {"schema": SCHEMA, "model": meta.get("model"), "checkpoint": meta.get("checkpoint"),
            "run": meta.get("run_id"), "commit": meta.get("commit"),
            "load": load, "arms": {"stock": stock, "again": again, "ours": ours, "served": sorted(by)},
            "settings": sets, "circuit_ahead_of_serving": NO_CIRCUIT,
            "tokens": tokens(out, ours, load, rep, replay_dir) if ours in dirs else not_measured(f"{ours} was not served"),
            "cost": line, "advice": not_measured(NO_ADVICE)}


def failed(r: dict) -> list[str]:
    """Why the job failed: each request whose replayed tokens differ from the served ones, with its first position,
    the steps that differ, and a replay whose record and kept tokens disagree. Empty where nothing differs."""
    t = r["tokens"]
    if not t["measured"] or t["reproduces"]:
        return []
    out = [f"request {d['request']} position {d['first_position']} (served {d['tokens']} tokens, replayed "
           f"{d['tokens_replayed']}; {d['positions']} positions differ)" for d in t["differing"]]
    if t["requests_differing"] > len(t["differing"]):
        out.append(f"and {t['requests_differing'] - len(t['differing'])} more requests")
    if t["steps_differing"]:
        out.append(f"{t['steps_differing']} of {t['steps_served']} steps' rows differ from the served schedule, the "
                   f"first step {t['first_differing_step']}")
    if t["record_disagrees"]:
        out.append("the replay's record (replay.json) and the tokens it kept disagree on what differs")
    return out


def _pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def _interval(c: dict) -> str:
    return f" [{_pct(c['ci95'][0])}, {_pct(c['ci95'][1])}]" if c.get("ci95") else " (no interval: too few reps)"


def _setting(name: str, key: str, sets: dict, stock: str, ours: str) -> str:
    mine = sets.get(ours)
    if mine is None:
        return f"  {name}: not read ({ours} was not served)"
    s = f"  {name}: {mine[key]['state']} ({mine[key]['by']})"
    theirs = sets.get(stock)
    if theirs is None:
        return s + f"; {stock} was not served"
    return s + ("" if theirs[key]["state"] == mine[key]["state"] else "; DIFFERS") + f"; {stock}: {theirs[key]['state']} ({theirs[key]['by']})"


def _limit(sets: dict, stock: str, ours: str) -> str:
    def one(c: dict) -> str:
        if c["tokens"] is None:
            return "not read"
        own = c["model"]
        return f"{c['tokens']:,} tokens, " + (
            "the model's own limit not read (no config.json at the checkpoint on this machine)" if own is None else
            f"{'the' if c['models_own'] else 'BELOW the'} model's own {own['tokens']:,} (config.json {own['key']})")

    mine, theirs = sets.get(ours), sets.get(stock)
    return "  context limit: " + (one(mine["context_limit"]) if mine else f"not read ({ours} was not served)") + (
        "" if not mine or not theirs or theirs["context_limit"]["tokens"] == mine["context_limit"]["tokens"]
        else f"; DIFFERS; {stock}: {one(theirs['context_limit'])}")


def text(r: dict) -> str:
    a, sets, t, c = r["arms"], r["settings"], r["tokens"], r["cost"]
    stock, ours = a["stock"], a["ours"]
    lines = [f"prod_repro: load {r['load']}; stock {stock}, stock again {a['again']}, ours {ours}; run {r['run']}, "
             f"commit {r['commit']}",
             f"model: {r['model'] or 'not read'}, as the benchmark names it; the checkpoint served: {r['checkpoint'] or 'not read'}",
             f"settings of {ours}, with {stock}'s beside:",
             _setting("prefix caching", "prefix_caching", sets, stock, ours),
             _setting("speculative decoding", "speculative_decoding", sets, stock, ours),
             _limit(sets, stock, ours),
             _setting("KV cache", "kv_cache", sets, stock, ours),
             _setting("torch.compile and CUDA graphs", "compiled", sets, stock, ours),
             f"  circuit compiled ahead of serving: {r['circuit_ahead_of_serving']}"]
    if not t["measured"]:
        lines.append(f"tokens: not measured: {t['why']}")
    else:
        h = t["cache_hits"]
        lines += [
            f"tokens: {'reproduced' if t['reproduces'] else 'FAILED'}: {t['requests']:,} requests replayed, "
            f"{t['tokens_served']:,} served tokens compared with {t['tokens_replayed']:,} replayed, "
            f"{t['tokens_differing']:,} differ in {t['requests_differing']:,} requests",
            *(f"  {x}" for x in failed(r)),
            f"  {t['steps_served']:,} engine steps, {t['steps_differing']:,} with other rows than the served schedule; "
            + (f"{h['requests']:,} requests began from a prefix-cache hit, the longest {h['longest']:,} tokens"
               if h["requests"] else "no request of the window began from a prefix-cache hit"),
            f"  served {t['served_mode']}, replayed {t['replay_mode']}; the window: {t['window']['chosen']}"]
    if not c["measured"]:
        lines.append(f"cost: not measured: {c['why']}")
    else:
        how, noise = c["how"], c["noise_floor"]
        bench = how["client"] == S.BENCH
        at = (f"at concurrency {how['rung']}, where stock's rungs {'show' if how['saturated'] else 'do NOT show'} it saturated"
              if bench else f"at {how['rate']:g} requests a second, which stock did not keep up with")
        reps = f"{how['reps']} paired reps" if bench else f"{how['reps'][0]} and {how['reps'][1]} reps"
        lines += [
            f"cost: {ours} over {stock} {_pct(c['value'])} of stock's serving time{_interval(c)}; under "
            f"{c['target']:g}%: " + {"meets": "yes", "misses": "no", UNLIKE: f"not said, {UNLIKE}"}.get(c["verdict"], "cannot tell"),
            f"  95% interval over {reps}, {at}: {how['by']}",
            "  noise floor: " + (f"{a['again']} over {stock} {_pct(noise['value'])}{_interval(noise)}"
                                 if noise and noise["value"] is not None else
                                 "not measured (stock was not served a second time)")]
    if c.get("settings_differ"):
        lines.append(f"  NOT at the same settings: {ours} and {stock} differ in {', '.join(c['settings_differ'])}, so the "
                     "figure is not our changes' cost")
    lines.append(f"advice: not measured: {r['advice']['why']}")
    done = sum(bool(r[k]["measured"]) for k in ("tokens", "cost", "advice"))
    lines.append(f"prod_repro: {done} of 3 lines measured" + ("; FAILED: the replay differs from what was served" if failed(r) else ""))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("arms", help="the arms a job serves by default: stock, stock again, ours")
    p = sub.add_parser("report", help="the settings and the three lines of a finished job's out/")
    p.add_argument("out", type=Path)
    stock, again, ours = default_arms()
    p.add_argument("--stock", default=stock)
    p.add_argument("--again", default=again, help="stock in a second server lifetime, the noise floor ('' for none)")
    p.add_argument("--ours", default=ours)
    p.add_argument("--load", required=True)
    p.add_argument("--rep", type=int, default=1, help="the rep of the load the replay re-ran")
    p.add_argument("--replay", type=Path, help="the replay's directory (default: OUT/replay-compiled)")
    p.add_argument("--target", type=float, default=TARGET, metavar="PERCENT")
    p.add_argument("--json", action="store_true", help="print the JSON object alone")
    a = ap.parse_args(argv)
    if a.cmd == "arms":
        print(" ".join(default_arms()))
        return 0
    r = report(a.out, a.stock, a.ours, a.again or None, a.load, a.rep, a.replay or a.out / "replay-compiled", a.target)
    r["failed"] = failed(r)
    (a.out / "prod_repro.json").write_text(json.dumps(r, indent=1))
    (a.out / "prod_repro.txt").write_text(text(r))
    print(json.dumps(r) if a.json else text(r), end="\n" if a.json else "")
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
