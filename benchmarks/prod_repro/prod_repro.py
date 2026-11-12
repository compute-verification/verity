"""vLLM at its serving settings, replayed: what one served job shows, as the settings it ran with and three lines (README.md).

    prod_repro.py arms [--model ID] [--out OUT]
    prod_repro.py replay-env --ours ARM --kept DIR
    prod_repro.py replay-ran --ours ARM --kept DIR [--private DIR] --out OUT
    prod_repro.py report OUT [--stock ARM] [--ours ARM] [--again ARM] --load NAME [--replayed PASS] [--rep 1]
                         [--replay DIR] [--expect-replay] [--advice-trace DIR [--advice-stops ID,...]]
                         [--target PERCENT] [--json]

``report`` reads a finished job's out/ as the serving cost benchmark leaves it (`benchmarks/serve_tax`: a directory an
arm, and its replay's), prints the report and writes it to OUT/prod_repro.json and OUT/prod_repro.txt:

- how far each arm's serve and the replay got, from their own records: an arm that failed with its error, one that
  stopped early, a replay that did not end with the last line of its log;
- the settings each of the two arms ran with, read from its launch line, its engine's own configuration (the step
  trace's header) and its counters: a setting the serve did not turn on is ``off``, and one its files can't confirm
  says so;
- ``tokens``: the served window the replay re-ran (`replay.window`), how many of its requests and served tokens the
  replay's kept tokens were compared with (`replay.token_diff`) and how many differ, with the steps that differ and
  the requests of the window that began from a prefix-cache hit. Where they differ: the first token that does (its
  request, position, served and replayed ids and the step that emitted it) and the first step whose rows do;
- ``cost``: what serving with our changes (``--ours``) costs over stock (``--stock``) on the load, a share of stock's
  serving time with its 95% interval, beside stock served a second time (``--again``), the noise floor: by vLLM's
  client the benchmark's paired ratio at the rung where stock is saturated (`serve_tax.serving_cost`), by the corpus
  client its tax over reps (`serve_tax.tax`), which is a cost only where stock did not keep up with the arrivals;
- ``advice``: the bits a request by class, with each class's events, of the certified serve whose trace directory
  ``--advice-trace`` names, by the tree's reader of it (`BY_CLASS`: its ``report``), under the registered stop ids
  (``--advice-stops``, EOS first; else the ``eos_token_id`` of the generation_config.json beside the trace's model).

A line is measured from the job's files, or it says ``not measured`` and why; a line whose files can't be read says
so and the rest of the report stands. Nothing here measures: the metrics, the cost and its interval, the comparison
and the advice's split are the benchmark's and the adapter's functions. It exits 1 where the replay differs from what
was served, naming each differing request and its first position, and where a replay was asked for
(``--expect-replay``) and the tokens line is not measured; 0 otherwise: any other line that is not measured is not a
failure, and the last line counts the lines that are measured.

``arms`` prints the three arms a job serves by default (`default_arms`): the arms at vLLM's published serving settings
(`PROD`) for a model whose recipe the benchmark holds, else stock compiled vLLM, stock again and the step record
(`TODAY`); with ``--out``, the three an earlier job's out/ is read as, which ``report`` reads it as too where none is
named.

``replay-env`` prints, a ``NAME=VALUE`` a line, what the replay of ``--ours``'s window runs under beside the job's
own environment so that its engine is the served one's (`replay_env`: the arm's own vLLM general plugins and
environment, the canonical DSA top-k's), with the plugins' site and the files they keep under ``--kept``; nothing
for an arm with neither. ``replay-ran`` then writes OUT/replay_engine.json (`replay_ran`): those plugins and that
environment, and the canonical top-k's counters as the replay's processes kept them and, where the job kept the
arm's own (``--private``, the benchmark's PRIVATE), as serving's did.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/serve_tax"))
import replay as R  # noqa: E402 (benchmarks/serve_tax, put on sys.path above)
import serve_tax as S  # noqa: E402

SCHEMA = "verity-bench/prod-repro/v0"
#: the arms a job serves by default, stock, stock again and ours: the arms at vLLM's published serving settings
#: (`default_arms`), else `TODAY`
PROD = ("prod", "prod-again", "prod-record")
TODAY = ("compiled", "compiled-again", "record")
#: the most serving with our changes may cost over stock, in percent of stock's serving time
TARGET = 1.0
#: the keys of a checkpoint's config.json that bound its context, the smallest of which is the model's own limit
CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length",
                "max_seq_length", "seq_len")
#: how the replayed window was chosen: no audit draws it yet
WINDOW = "every request of the load's rep, a window the command names: no audit's coins drew it"
#: the reader of a certified serve's advice by class, where the tree has it: its ``report(label, DIR, stop_ids)``
#: gives a trace's bits by class with each class's events (its ``OF``), or its refusal
BY_CLASS = ROOT / "benchmarks/network_e2e/encodings/by_class.py"
NO_ADVICE = ("no trace directory of a certified serve was given (ADVICE_TRACE=DIR): the advice code codes a serve under "
             "the network certifier's record and the certified tick, and this job's own arms run under neither")
NO_CIRCUIT = "no: nothing in this job compiles the settings' circuit"
ON, OFF, UNCONFIRMED = "on", "off", "not confirmed"
#: the cost's verdict where stock and ours did not serve at the same settings: the figure is then not our changes' cost
UNLIKE = "the arms are not at the same settings"
#: the rows of a differing step and the characters of a log's line the report prints
ROWS_MAX, LINE_MAX = 3, 400
#: vLLM's own line for a context limit left to it (``--max-model-len auto``): the limit it fitted to the cards' memory,
#: or the model's own where that fits; and the lines of a server's log read for it
AUTO_FIT = re.compile(r"Auto-fit max_model_len: (?:reduced from (\d+) to (\d+)|full model context length (\d+) fits)")
LOG_LINES = 20000
#: what the report says of a setting the launch line turns on and the step trace's header does not hold
NOT_IN_HEADER = "the step trace's header holds none, and the replay's engine is built from the header"
#: the key of a checkpoint's config.json for the keys the DSA indexer's top-k keeps: a request of more ids than that
#: reads the keys the top-k selected
INDEX_TOPK = "index_topk"
#: the canonical DSA top-k's two variables (`benchmarks/dsa_topk`): its switch, and the directory each process keeps
#: its counters in
TOPK, TOPK_LOG = "VERITY_DSA_TOPK", "VERITY_DSA_TOPK_LOG"
#: what a job's replay step leaves in OUT of the engine its replay ran (`replay_ran`)
REPLAY_ENGINE = "replay_engine.json"


def default_arms(model_id: str | None = None, out: Path | None = None) -> tuple[str, str, str]:
    """The arms of a job that names none: `PROD` where the benchmark has them and holds vLLM's recipe for the model (it
    refuses an arm at the published settings for a model with none), else `TODAY`. An earlier job's ``out`` is read as
    `PROD` where it holds that set's arm of ours."""
    if out is not None:
        return PROD if (out / PROD[2]).is_dir() else TODAY
    model = S.MODELS.get(model_id)
    return PROD if all(a in S.ARMS for a in PROD) and model is not None and getattr(model, "published", ()) else TODAY


def not_measured(why: str, **more: object) -> dict:
    return {"measured": False, "why": why, **more}


def read(what: str, f: Callable[[], dict]) -> dict:
    """One part of the report, or why it can't be read: a refusal of the benchmark's readers or a file of the job that
    is missing or not in their form ends that part, and the rest of the report stands."""
    try:
        return f()
    except (SystemExit, Exception) as e:  # noqa: BLE001 -- the report says why the part can't be read
        return not_measured(f"{what} can't be read from the job's files: {type(e).__name__}: {e}", unread=True)


def of_load(name: str, load: str) -> bool:
    """Whether a pass named ``name`` is of ``load``, by either client (vLLM's names a pass by its load and rung)."""
    return load in (name, name.rpartition("@")[0])


# ----------------------------------------------------------------------------------------------- how far the job got

def arm_state(arm_dir: Path, load: str) -> dict:
    """How far one arm's serve got, from its own record (meta.json): whether it started, the error it stopped on, the
    pass it stopped before (the benchmark's STOP_AT), and the passes of ``load`` it served with their requests and the
    ones left unfinished."""
    f = arm_dir / "meta.json"
    if not f.is_file():
        return {"started": False, "ok": False}
    meta = json.loads(f.read_text())
    passes = [x for x in meta.get("loads") or [] if of_load(x["load"], load)]
    return {"started": True, "error": meta.get("error"), "stopped": meta.get("stopped"), "startup_s": meta.get("startup_s"),
            "passes": len(passes), "requests": sum(x.get("requests") or 0 for x in passes),
            "unfinished": sum(x.get("unfinished") or 0 for x in passes),
            "ok": bool(passes) and not meta.get("error") and not meta.get("stopped")}


def last_line(log: Path) -> str | None:
    """The last line of a log that says anything, cut to `LINE_MAX` characters; None without the log."""
    if not log.is_file():
        return None
    lines = [x.strip() for x in log.read_text(errors="replace").splitlines() if x.strip()]
    return lines[-1][:LINE_MAX] if lines else ""


def replay_state(replay_dir: Path, expected: bool) -> dict:
    """Whether the replay ended (it wrote replay.json), and where it did not, the last line of its log, which the
    benchmark's runner keeps beside its directory; and what its engine ran under, where the job's replay step left
    that there (`REPLAY_ENGINE`)."""
    log, engine = replay_dir.with_name(replay_dir.name + ".log"), replay_dir.with_name(REPLAY_ENGINE)
    return {"expected": expected, "ended": (replay_dir / "replay.json").is_file(), "log": str(log) if log.is_file() else None,
            "last_line": last_line(log), "engine": json.loads(engine.read_text()) if engine.is_file() else None}


# ------------------------------------------------------------------------------------------- the replay's engine

def arm_env(arm: str, private: Path) -> dict[str, str]:
    """An arm's own environment (`serve_tax.Arm.env`), the files it keeps on the node under ``private``; empty for an
    arm the benchmark does not define."""
    a = S.ARMS.get(arm)
    return {k: v.replace(S.PRIVATE, str(private)) for k, v in a.env.items()} if a else {}


def replay_env(arm: str, kept: Path, path: str = "") -> dict[str, str]:
    """What a replay of ``arm``'s window runs under, beside the job's own environment, so that its engine is the
    served one's: the arm's own vLLM general plugins (`serve_tax.Arm.general`: the canonical DSA top-k's), declared in
    the site ``kept``/site and on the path with their directories ahead of ``path``, and the arm's environment, the
    files it keeps under ``kept``/private. The benchmark's own plugin is not among them: the replay records its own
    steps. Empty for an arm with neither, whose replay runs in the job's environment as it is."""
    a = S.ARMS.get(arm)
    if a is None or not (a.general or a.env):
        return {}
    env = arm_env(arm, kept / "private")
    if a.general:
        site = kept / "site"
        info = site / "verity_prod_repro-0.0.dist-info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-prod-repro\nVersion: 0.0\n")
        (info / "entry_points.txt").write_text("[vllm.general_plugins]\n" + "".join(f"{e}\n" for e, _ in a.general))
        env["PYTHONPATH"] = os.pathsep.join([str(site), *(str(ROOT / d) for _, d in a.general), *([path] if path else [])])
    return env


def plugins_load(arm: str, env: dict[str, str]) -> str | None:
    """Why one of ``arm``'s own general plugins doesn't load in this interpreter under ``env``, as the benchmark checks
    an arm's before its server starts (vLLM logs a plugin whose import fails and runs on without it); None where each
    loads."""
    names = [e.split("=")[0].strip() for e, _ in S.ARMS[arm].general]
    code = ("import sys\nfrom importlib.metadata import entry_points\n"
            "found = {e.name: e for e in entry_points(group='vllm.general_plugins')}\n"
            "for name in sys.argv[1:]:\n    found[name].load()\n")
    p = subprocess.run([sys.executable, "-c", code, *names], env={**os.environ, **env}, capture_output=True, text=True)
    return (p.stderr.strip().splitlines() or ["no output"])[-1] if p.returncode else None


def topk_counters(d: Path | None) -> dict | None:
    """The canonical top-k's counters over the processes that kept theirs in ``d`` (`dsa_topk.log_to`: a file a
    process, written every few thousand calls and at its exit): the processes, the calls of the indexer's top-k, the
    canonicalizing launches, and the rows rewritten, those with a tie at the k-th key and those bisected. None where
    no process kept any."""
    recs = [json.loads(f.read_text()) for f in sorted(d.glob("dsa_topk.*.json"))] if d and d.is_dir() else []
    if not recs:
        return None
    rows = Counter()
    for r in recs:
        for dev in r["rows"].values():
            rows.update(dev)
    return {"processes": len(recs), "calls": sum(sum(r["calls"].values()) for r in recs),
            "launches": sum(sum(r["launches"].values()) for r in recs), "rows": dict(rows)}


def replay_ran(arm: str, kept: Path, private: Path | None) -> dict:
    """What the replay's engine ran under (`replay_env`): the arm's own plugins and environment, and where that turns
    the canonical top-k on, its counters as the replay's processes kept them and as serving's did, where the job kept
    the arm's files (``private``, the benchmark's PRIVATE: an arm's are under its name)."""
    a = S.ARMS.get(arm)
    env = arm_env(arm, kept / "private")
    served = arm_env(arm, private / arm) if private else {}
    return {"arm": arm, "plugins": [e for e, _ in a.general] if a else [], "env": env,
            "topk": {"replay": topk_counters(Path(env[TOPK_LOG])),
                     "served": topk_counters(Path(served[TOPK_LOG])) if served else None,
                     "served_kept": bool(private)} if env.get(TOPK) and env.get(TOPK_LOG) else None}


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


def fitted_limit(arm_dir: Path) -> dict | None:
    """The context limit vLLM fitted for an arm whose launch line left it to vLLM, from the server's own log (`AUTO_FIT`,
    in its first `LOG_LINES` lines): the tokens and the log's words. None where the log has no such line."""
    log = arm_dir / "server" / "server.log"
    if not log.is_file():
        return None
    with open(log, encoding="utf-8", errors="replace") as f:
        for _, line in zip(range(LOG_LINES), f):
            m = AUTO_FIT.search(line)
            if m:
                return {"tokens": int(m.group(2) or m.group(3)), "said": m.group(0)}
    return None


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
    """An arm's reps of ``load``, by either client."""
    return [r for name, reps in arm["loads"].items() if of_load(name, load) for r in reps]


def _mean(reps: list[dict], key: str) -> float | None:
    xs = [r[key] for r in reps if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else None


def settings(arm_dir: Path, arm: dict, load: str) -> dict:
    """What one arm's serve ran with: each setting's state and what says so. Speculative decoding and the KV cache's
    type are the launch line's, with the engine's own configuration beside where the arm kept a step trace, and a
    note where the line turns one on that the trace's header does not hold (`NOT_IN_HEADER`). The context limit is
    the engine's own where the arm kept a step trace; else, for a line that leaves it to vLLM (``auto``), the one
    vLLM's log says it fitted (`fitted_limit`); else the launch line's number."""
    meta = json.loads((arm_dir / "meta.json").read_text())
    argv, header, reps = meta["server_argv"], trace_header(arm_dir), reps_of(arm, load)
    share = _mean(reps, "prompt_cached_share")
    served = "" if share is None else f"; the cache served {100 * share:.1f}% of the load's prompt tokens"
    if "--no-enable-prefix-caching" in argv or header.get("enable_prefix_caching") is False:
        cache = (OFF, "--no-enable-prefix-caching" if "--no-enable-prefix-caching" in argv else "the engine's configuration")
    elif header.get("enable_prefix_caching"):
        cache = (ON, "the engine's configuration" + served)
    elif share:
        cache = (ON, served[2:])
    else:
        cache = (UNCONFIRMED, "the launch line does not turn it off, and the arm kept no step trace and counted no hit")
    def said(line: str, engine: str, none: str, on: bool = True) -> str:
        """What says a setting: the launch line's flags, the engine's configuration, and where the line turns it on
        (``on``) a header that lacks it."""
        both = "; ".join(x for x in (line, f"the engine's configuration: {engine}" if engine else "") if x)
        return (both + (f"; {NOT_IN_HEADER}" if line and on and header and not engine else "")) or none

    spec, kv = flags(argv, ("--speculative", "--num-speculative")), flag(argv, "--kv-cache-dtype")
    spec_engine, kv_engine = header.get("speculative"), header.get("kv_cache_dtype")
    if isinstance(spec_engine, dict):
        spec_engine = ", ".join(f"{k} {v}" for k, v in sorted(spec_engine.items()))
    accepted, each = _mean(reps, "draft_accepted_share"), _mean(reps, "accepted_per_draft")
    drafts = "" if each is None else f"; {100 * (accepted or 0.0):.1f}% of its draft tokens accepted, {each:.2f} a draft"
    given, engine = flag(argv, "--max-model-len"), header.get("max_model_len")
    fitted = fitted_limit(arm_dir) if engine is None and given == "auto" else None
    if engine is not None:
        limit, by = int(engine), "the engine took it"
    elif fitted:
        limit, by = fitted["tokens"], f"its server's log: {fitted['said']}"
    else:
        limit, by = (int(given), None) if given is not None and given.isdigit() else (None, None)
    own = model_limit(Path(meta["checkpoint"])) if meta.get("checkpoint") else None
    return {"prefix_caching": {"state": cache[0], "by": cache[1], "cached_share": share},
            "speculative_decoding": {"state": ON if spec or spec_engine else OFF,
                                     "by": said(spec, str(spec_engine or ""), "no --speculative-config in the launch line")
                                     + (drafts if spec or spec_engine else ""),
                                     "draft_accepted_share": accepted, "accepted_per_draft": each},
            "kv_cache": {"state": kv or kv_engine or "the engine's default",
                         "by": said("--kv-cache-dtype" if kv else "", str(kv_engine or ""),
                                    "no --kv-cache-dtype in the launch line", on=kv != "auto")},
            "context_limit": {"tokens": limit, "given": given, "by": by, "model": own,
                              "models_own": None if limit is None or own is None else limit >= own["tokens"]},
            "compiled": {"state": OFF if "--enforce-eager" in argv else ON,
                         "by": "--enforce-eager" if "--enforce-eager" in argv else "no --enforce-eager in the launch line"}}


def settings_differ(a: dict, b: dict) -> list[str]:
    """The settings two arms' serves differ in. Their context limits differ where their launch lines name different
    ones, or where both are resolved to different numbers."""
    names = {"prefix_caching": "prefix caching", "speculative_decoding": "speculative decoding",
             "kv_cache": "the KV cache's type", "compiled": "torch.compile and CUDA graphs"}
    out = [name for k, name in names.items() if a[k]["state"] != b[k]["state"]]
    x, y = a["context_limit"], b["context_limit"]
    limits = x["given"] != y["given"] if None in (x["tokens"], y["tokens"]) else x["tokens"] != y["tokens"]
    return out + ["the context limit"] * limits


# --------------------------------------------------------------------------------------------------------- the tokens

def replayed_steps(replay_dir: Path) -> dict[int, list] | None:
    """The rows of each step the replay ran, from the steps it kept; None where it kept none."""
    f = next((p for p in (replay_dir / "replay_steps.jsonl", replay_dir / "replay_steps.jsonl.gz") if p.is_file()), None)
    if f is None:
        return None
    with (gzip.open if f.suffix == ".gz" else open)(f, "rt", encoding="utf-8") as lines:
        return {s["step"]: s["rows"] for s in map(json.loads, filter(str.strip, lines))}


def first_step(w: dict, replay_dir: Path, step: int | None) -> dict | None:
    """The first step whose rows differ: the rows serving ran that the replay did not and the ones the replay ran
    instead, `ROWS_MAX` of each (a row is its request, first position and tokens)."""
    if step is None:
        return None
    def only(a: list, b: list | None) -> list:
        return [r for r in a if r not in (b or [])][:ROWS_MAX]

    served = next((s["rows"] for s in w["steps"] if s["step"] == step), None)
    kept = replayed_steps(replay_dir)
    mine = None if kept is None else kept.get(step)
    return {"step": step, "steps": len(w["steps"]), "served": None if served is None else only(served, mine),
            "replayed": None if mine is None else only(mine, served), "replay_kept_steps": kept is not None}


def past_topk(w: dict, checkpoint: str | None, differing: list[dict]) -> dict | None:
    """The requests of a window whose prompt and served ids together pass the keys the model's DSA indexer's top-k
    keeps (`INDEX_TOPK` of the checkpoint's config.json), and how many of the differing ones are among them: past
    that many keys attention reads the ones the top-k selected. None for a model with no such top-k, or where this
    machine has no config.json of the checkpoint."""
    f = Path(checkpoint) / "config.json" if checkpoint else None
    k = json.loads(f.read_text()).get(INDEX_TOPK) if f and f.is_file() else None
    if not isinstance(k, int):
        return None
    past = {r["engine_id"] for r in w["requests"] if len(r["prompt_ids"]) + len(r["served"]) > k}
    return {"keys": k, "requests": len(past), "differing": sum(d["request"] in past for d in differing)}


def tokens(out: Path, ours: str, load: str, rep: int, replay_dir: Path, state: dict) -> dict:
    """The tokens line: the served window of ``ours``'s pass ``load`` rep ``rep`` against the tokens its replay kept.
    Not measured without a replay, where the replay is of another window, or where the window can't be read back.
    Each differing request has its first differing position, the ids served and replayed there and the step that
    emitted it while serving, the earliest first."""
    arm_dir = out / ours
    meta = json.loads((arm_dir / "meta.json").read_text())
    if not state["ended"]:
        if state["expected"]:
            return not_measured(f"the replay did not end (no replay.json in {replay_dir})" + (
                f"; its log ends: {state['last_line']} ({state['log']})" if state["log"] else "; it left no log"))
        return not_measured("no replay ran" + (": none was asked for (--replay PASS names the pass of vLLM's client to "
                                               "re-run, from an arm that keeps each request's ids)"
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
    diff, at = R.token_diff(served, got), R.emitted_at(w["steps"])
    for d in diff:
        a, b, p = served[d["request"]], got.get(d["request"], []), d["first_position"]
        steps = at.get(d["request"], [])
        d |= {"served_id": a[p] if p < len(a) else None, "replayed_id": b[p] if p < len(b) else None,
              "step": steps[p] if p < len(steps) else None}
    diff.sort(key=lambda d: (d["step"] is None, d["step"] or 0))
    hits = {rid: int(k) for s in w["steps"] for rid, k in (s.get("cache_hit") or {}).items()}
    sampling = Counter((r["sampling"].get("temperature"), r["sampling"].get("top_p"), r["sampling"].get("seed") is not None)
                       for r in w["requests"])
    n = {"requests": len(served), "tokens_served": sum(map(len, served.values())),
         "tokens_replayed": sum(len(got.get(rid, ())) for rid in served),
         "requests_differing": len(diff), "tokens_differing": sum(d["positions"] for d in diff)}
    if not n["tokens_served"]:
        return not_measured("the window holds no served token", **n)
    disagree = (rec["requests_tokens_differing"], rec["token_positions_differing"]) != (n["requests_differing"], n["tokens_differing"])
    return {"measured": True, **n, "differing": diff[:R.DIFFERING_MAX],
            "steps_served": rec["steps_served"], "steps_differing": rec["steps_differing"],
            "first_differing_step": rec["first_differing_step"],
            "first_step": first_step(w, replay_dir, rec["first_differing_step"]),
            "record_disagrees": disagree,
            "reproduces": not diff and not rec["steps_differing"] and not disagree,
            "cache_hits": {"requests": len(hits), "tokens": sum(hits.values()), "longest": max(hits.values(), default=0)},
            "sampling": [{"temperature": t, "top_p": p, "seeded": seeded, "requests": k}
                         for (t, p, seeded), k in sorted(sampling.items(), key=lambda x: -x[1])],
            "past_topk": past_topk(w, meta.get("checkpoint"), diff),
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
               "sessions": bool(getattr(S.LOADS.get(load), "sessions", False)),
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


# --------------------------------------------------------------------------------------------------------- the advice

def stop_ids(checkpoint: str | None, given: str | None) -> tuple[list[int] | None, str]:
    """The registered stop ids the advice is read under, EOS first, and what says so: ``given`` (ID,...), else the
    ``eos_token_id`` of the generation_config.json of the checkpoint this job served."""
    if given:
        return [int(x) for x in given.split(",")], "ADVICE_STOPS"
    f = Path(checkpoint) / "generation_config.json" if checkpoint else None
    if f is None or not f.is_file():
        return None, "this machine has no generation_config.json of the checkpoint the job served"
    eos = json.loads(f.read_text()).get("eos_token_id")
    ids = [eos] if isinstance(eos, int) else list(eos or [])
    return (ids or None), f"eos_token_id of {f}"


def by_class(reader: Path) -> object:
    """The tree's reader of a certified serve's advice by class (`BY_CLASS`), loaded from its file with the adapter it
    reads on the path, as the benchmark's replay puts it there."""
    adapter = str(ROOT / "adapters/vllm")
    if adapter not in sys.path:
        sys.path.insert(0, adapter)
    spec = importlib.util.spec_from_file_location("prod_repro_by_class", reader)
    if spec is None or spec.loader is None:
        raise ImportError(f"{reader} is not a module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def advice(out: Path, trace: Path | None, stops: str | None, checkpoint: str | None) -> dict:
    """The advice line: the bits a request by class of the certified serve whose trace directory is ``trace``, each
    class with its standing and its events, as the tree's reader gives them (`BY_CLASS`'s ``report``), kept whole in
    OUT/advice_by_class.json. Not measured without the directory, where the tree has no reader, where the stop ids are
    not known, where the reader refuses the trace, or where the advice does not decode back to it."""
    reader = BY_CLASS
    if trace is None:
        return not_measured(NO_ADVICE)
    if not trace.is_dir():
        return not_measured(f"ADVICE_TRACE={trace} is not a directory on this machine")
    if not reader.is_file():
        return not_measured(f"this tree has no reader of a certified serve's advice by class (no {reader}); the trace is "
                            f"{trace}")
    ids, by = stop_ids(checkpoint, stops)
    if ids is None:
        return not_measured(f"the registered stop ids are not known ({by}): ADVICE_STOPS=ID,..., EOS first")
    mod = by_class(reader)
    row = mod.report("advice", str(trace), ids)
    (out / "advice_by_class.json").write_text(json.dumps({"trace": str(trace), "stops": ids, "row": row}, indent=1))
    if "error" in row:
        return not_measured(f"the reader refuses the trace {trace}: {row['error']}", aborts=row.get("aborts"))
    if not row["round_trip"]:
        d = row.get("first_step_that_differs")
        return not_measured(f"the advice of {trace} does not decode back to it" + (
            f": the decoder refuses it: {row['refused']}" if row.get("refused") else "") + (
            f"; its replay's schedule differs from the engine's from step {d['step']}" if d else ""))
    total, n = row["total"], row["requests"]
    classes = {}
    for name, c in row["classes"].items():
        bits, events = total["classes"].get(name), {e: total["events"][e] for e in mod.OF.get(name, ())}
        classes[name] = {"standing": c["standing"], "bits": bits, "bits_a_request": bits / n if bits is not None and n else None,
                         "events": events, "exercised": bool(bits) or any(events.values())}
    return {"measured": True, "bits": total["charge"], "requests": n, "bits_a_request": total["charge"] / n if n else None,
            "classes": classes, "round_trip": True, "tick": bool(row.get("tick")), "windows": row["windows"], "T": row["T"],
            "prefix_caching": row.get("prefix_caching"), "trace": str(trace), "stops": ids, "stops_by": by}


# --------------------------------------------------------------------------------------------------------- the report

def report(out: Path, stock: str, ours: str, again: str | None, load: str, rep: int = 1, replay_dir: Path | None = None,
           target: float = TARGET, replayed: str | None = None, expect_replay: bool = False,
           advice_trace: Path | None = None, advice_stops: str | None = None) -> dict:
    replay_dir = replay_dir or out / "replay-compiled"
    names = [a for a in (stock, again, ours) if a]
    states = {a: read(f"how far {a} got", lambda a=a: arm_state(out / a, load)) for a in names}
    dirs = {a: out / a for a in names if (out / a / "meta.json").is_file()}
    by, sets, unread = {}, {}, {}
    for a, d in dirs.items():
        m = read(f"{a}'s metrics", lambda d=d: S.arm_metrics(d))
        if m.get("unread"):
            unread[a] = m["why"]
            continue
        by[a] = m
        if a in (stock, ours):
            s = read(f"{a}'s settings", lambda a=a, d=d, m=m: settings(d, m, load))
            if s.get("unread"):
                unread[a] = s["why"]
            else:
                sets[a] = s
    meta = read(f"{ours}'s record", lambda: json.loads((dirs[ours] / "meta.json").read_text())) if ours in dirs else {}
    line = read("the cost", lambda: cost(out, by, stock, ours, again, load, target))
    if len(sets) == 2 and (differ := settings_differ(sets[stock], sets[ours])):
        line["settings_differ"] = differ
        if line["measured"]:
            line["verdict"] = UNLIKE
    state = read("how far the replay got", lambda: replay_state(replay_dir, expect_replay))
    if ours not in dirs:
        toks = not_measured(f"{ours} was not served")
    elif state.get("unread"):
        toks = not_measured(state["why"])
    else:
        toks = read("the tokens", lambda: tokens(out, ours, replayed or load, rep, replay_dir, state))
    return {"schema": SCHEMA, "model": meta.get("model"), "checkpoint": meta.get("checkpoint"),
            "run": meta.get("run_id"), "commit": meta.get("commit"),
            "load": load, "arms": {"stock": stock, "again": again, "ours": ours, "served": sorted(by)},
            "ran": {"arms": states, "unread": unread, "replay": state},
            "settings": sets, "circuit_ahead_of_serving": NO_CIRCUIT,
            "tokens": toks, "cost": line,
            "advice": read("the advice", lambda: advice(out, advice_trace, advice_stops, meta.get("checkpoint")))}


def failed(r: dict) -> list[str]:
    """Why the job failed: each request whose replayed tokens differ from the served ones, with its first position,
    the steps that differ, and a replay whose record and kept tokens disagree; or a replay that was asked for and
    left the tokens line not measured. Empty where nothing differs."""
    t = r["tokens"]
    if not t["measured"]:
        return ["a replay was asked for and the tokens line is not measured"] if r["ran"]["replay"].get("expected") else []
    if t["reproduces"]:
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


def where(t: dict) -> list[str]:
    """Where a replay that differs first does: the first token (the earliest step's) and the first step."""
    out = []
    if t["differing"]:
        d = t["differing"][0]
        ids = (f"served id {d['served_id']}" if d["served_id"] is not None else "serving ended before it") + ", " + (
            f"replayed id {d['replayed_id']}" if d["replayed_id"] is not None else "the replay ended before it")
        out.append(f"the first token that differs: request {d['request']} position {d['first_position']}: {ids}"
                   + ("" if d["step"] is None else f"; serving emitted it in step {d['step']} of the window"))
    s = t.get("first_step")
    if s:
        def rows(x: list) -> str:
            return json.dumps(x, separators=(",", ":"))

        mine = ("the replay kept no steps to say what it ran" if not s["replay_kept_steps"] else
                "the replay ran no such step" if s["replayed"] is None else f"the replay ran {rows(s['replayed'])}")
        out.append(f"the first step that differs: step {s['step']} of the window's {s['steps']}: "
                   + ("serving ran no such step" if s["served"] is None else f"serving ran {rows(s['served'])}")
                   + f" where {mine} (a row: request, first position, tokens; the first {ROWS_MAX} of each side's own)")
    return out


def _pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def _interval(c: dict) -> str:
    return f" [{_pct(c['ci95'][0])}, {_pct(c['ci95'][1])}]" if c.get("ci95") else " (no interval: too few reps)"


def _absent(r: dict, arm: str) -> str:
    """Why the report holds no settings of ``arm``."""
    return f"{arm}'s files can't be read" if arm in r["ran"]["unread"] else f"{arm} was not served"


def _setting(name: str, key: str, r: dict) -> str:
    sets, stock, ours = r["settings"], r["arms"]["stock"], r["arms"]["ours"]
    mine = sets.get(ours)
    if mine is None:
        return f"  {name}: not read ({_absent(r, ours)})"
    s = f"  {name}: {mine[key]['state']} ({mine[key]['by']})"
    theirs = sets.get(stock)
    if theirs is None:
        return s + f"; {_absent(r, stock)}"
    return s + ("" if theirs[key]["state"] == mine[key]["state"] else "; DIFFERS") + f"; {stock}: {theirs[key]['state']} ({theirs[key]['by']})"


def _limit(r: dict) -> str:
    def one(c: dict) -> str:
        line = f"--max-model-len {c['given']}" if c["given"] is not None else "no --max-model-len in the launch line"
        if c["tokens"] is None:
            return f"not resolved ({line}; the arm kept no step trace and its server's log does not say what its engine took)"
        own = c["model"]
        return f"{c['tokens']:,} tokens ({c['by'] + '; ' if c['by'] else ''}{line}), " + (
            "the model's own limit not read (no config.json at the checkpoint on this machine)" if own is None else
            f"{'the' if c['models_own'] else 'BELOW the'} model's own {own['tokens']:,} (config.json {own['key']})")

    sets, stock, ours = r["settings"], r["arms"]["stock"], r["arms"]["ours"]
    mine, theirs = sets.get(ours), sets.get(stock)
    if mine is None:
        return f"  context limit: not read ({_absent(r, ours)})"
    s = "  context limit: " + one(mine["context_limit"])
    if theirs is None:
        return s + f"; {_absent(r, stock)}"
    differ = "the context limit" in settings_differ(theirs, mine)
    return s + f"; {'DIFFERS; ' if differ else ''}{stock}: {one(theirs['context_limit'])}"


def _arm(name: str, s: dict) -> str:
    if s.get("unread"):
        return f"  {name}: {s['why']}"
    if not s["started"]:
        return f"  {name}: NOT SERVED: the job holds no record of it (no {name}/meta.json): it did not start"
    up = "" if s["startup_s"] is None else f"up in {s['startup_s']:g} s; "
    served = "NOTHING of the load was served" if not s["passes"] else (
        f"the load served {s['passes']} time{'' if s['passes'] == 1 else 's'}, {s['requests']:,} requests, "
        + ("every one finished" if not s["unfinished"] else f"{s['unfinished']:,} UNFINISHED"))
    if s["error"]:
        return f"  {name}: FAILED: {s['error']} (its record's error; its server's log: {name}/server/server.log); {up}{served}"
    if s["stopped"]:
        st = s["stopped"]
        return (f"  {name}: STOPPED before {st.get('before')} rep {st.get('rep')} (--stop-at {st.get('stop_at')}: the pass "
                f"needed {st.get('need_s')} s); {up}{served}")
    return f"  {name}: {up}{served}"


def _sampling(t: dict) -> str:
    def one(s: dict) -> str:
        if not s["temperature"]:
            return f"{s['requests']:,} greedy"
        top = "" if s["top_p"] in (None, 1.0) else f" and top-p {s['top_p']:g}"
        return f"{s['requests']:,} at temperature {s['temperature']:g}{top}, {'each with a seed' if s['seeded'] else 'with NO seed'}"

    return ", ".join(one(s) for s in t["sampling"])


def _advice(v: dict) -> list[str]:
    def one(name: str, c: dict) -> str:
        events = ", ".join(f"{e.replace('_', ' ')} {k:,}" for e, k in c["events"].items())
        if c["bits"] is None:
            said = "no field in the code"
        elif c["exercised"]:
            said = f"{c['bits']:,} bits, {c['bits_a_request']:.3f} a request"
        else:
            said = "0 bits, not exercised: none of its events arose"
        more = "; ".join(x for x in (c["standing"], events) if x and x != "no field")
        return f"  {name}: {said}" + (f" ({more})" if more else "")

    tick = "under the certified tick" if v["tick"] else "admitted on arrival, NOT under the certified tick"
    return [f"advice: {v['bits_a_request']:.3f} bits a request: {v['bits']:,} bits over {v['requests']:,} requests, {tick}; "
            f"{v['windows']} window{'' if v['windows'] == 1 else 's'} of T = {v['T']} buckets; it decodes back to the trace",
            *(one(name, c) for name, c in v["classes"].items()),
            f"  the trace: {v['trace']}; stop ids {','.join(map(str, v['stops']))} ({v['stops_by']})"]


def _engine(e: dict | None) -> list[str]:
    """What the replay's engine ran under, and where that is the canonical top-k, what its counters say of the
    replay's processes and of serving's: a lower bound, since a process keeps them every few thousand calls and at
    its exit."""
    if e is None:
        return []
    if not e["plugins"] and not e["env"]:
        return [f"  the replay's engine: the job's own environment ({e['arm']} turns nothing on in its engine beside its record)"]

    def one(c: dict) -> str:
        rows = c["rows"]
        return (f"{c['processes']} process{'' if c['processes'] == 1 else 'es'} kept counters: at least {c['calls']:,} "
                f"calls of the indexer's top-k, {rows.get('rows_long', 0):,} rows past its keys rewritten "
                f"({rows.get('rows_tied', 0):,} with a tie at the last key kept, {rows.get('rows_bisected', 0):,} bisected)")

    names = ", ".join(x.split("=")[0].strip() for x in e["plugins"])
    env = ", ".join(f"{k}={v}" for k, v in e["env"].items() if k != TOPK_LOG)
    lines = [f"  the replay's engine: under {e['arm']}'s own changes to the engine: "
             + "; ".join(x for x in (f"the vLLM plugin{'' if len(e['plugins']) == 1 else 's'} {names}" if names else "", env) if x)]
    t = e.get("topk")
    if t:
        lines += ["  the canonical top-k in the replay: "
                  + (one(t["replay"]) if t["replay"] else "NO process kept its counters: nothing shows it ran there"),
                  "  the canonical top-k while serving: "
                  + (one(t["served"]) if t["served"] else "NO process kept its counters: nothing shows it ran there"
                     if t["served_kept"] else "its counters were not kept (the job named no PRIVATE directory)")]
    return lines


def text(r: dict) -> str:
    a, t, c, ran = r["arms"], r["tokens"], r["cost"], r["ran"]
    stock, ours = a["stock"], a["ours"]
    lines = [f"prod_repro: load {r['load']}; stock {stock}, stock again {a['again'] or 'not served'}, ours {ours}; "
             f"run {r['run']}, commit {r['commit']}",
             f"model: {r['model'] or 'not read'}, as the benchmark names it; the checkpoint served: {r['checkpoint'] or 'not read'}",
             "what ran:", *(_arm(name, s) for name, s in ran["arms"].items()),
             *(f"  {why}" for why in ran["unread"].values())]
    rp = ran["replay"]
    if rp.get("unread"):
        lines.append(f"  the replay: {rp['why']}")
    elif rp["ended"]:
        lines.append("  the replay: ended" + ("" if not t["measured"] else
                                              f" in {t['replay_wall_s']:.0f} s, its engine ready in {t['engine_ready_s']:.0f} s; "
                                              f"the served window took {t['served_window_s']:.0f} s"))
    elif rp["expected"]:
        lines.append("  the replay: DID NOT END" + (f"; its log ends: {rp['last_line']} ({rp['log']})" if rp["log"] else
                                                    ": it left no log, so it did not start"))
    else:
        lines.append("  the replay: none was asked for")
    lines += _engine(rp.get("engine"))
    lines += [f"settings of {ours}, with {stock}'s beside:",
              _setting("prefix caching", "prefix_caching", r),
              _setting("speculative decoding", "speculative_decoding", r),
              _limit(r),
              _setting("KV cache", "kv_cache", r),
              _setting("torch.compile and CUDA graphs", "compiled", r),
              f"  circuit compiled ahead of serving: {r['circuit_ahead_of_serving']}"]
    if not t["measured"]:
        lines.append(f"tokens: not measured: {t['why']}")
    else:
        h = t["cache_hits"]
        lines += [
            f"tokens: {'reproduced' if t['reproduces'] else 'FAILED'}: {t['requests']:,} requests replayed, "
            f"{t['tokens_served']:,} served tokens compared with {t['tokens_replayed']:,} replayed, "
            f"{t['tokens_differing']:,} differ in {t['requests_differing']:,} requests",
            *(f"  {x}" for x in (*where(t), *failed(r))),
            f"  {t['steps_served']:,} engine steps, {t['steps_differing']:,} with other rows than the served schedule; "
            + (f"{h['requests']:,} requests began from a prefix-cache hit, the longest {h['longest']:,} tokens"
               if h["requests"] else "no request of the window began from a prefix-cache hit"),
            f"  served {t['served_mode']}, replayed {t['replay_mode']}; the window's requests: {_sampling(t)}",
            *([f"  {k['requests']:,} of the window's requests passed {k['keys']:,} ids of prompt and output, past which "
               "attention reads the keys the DSA indexer's top-k selects"
               + (f"; {k['differing']:,} of the {t['requests_differing']:,} differing requests are among them"
                  if t["requests_differing"] else "")] if (k := t.get("past_topk")) else []),
            f"  the window: {t['window']['chosen']}"]
    if not c["measured"]:
        lines.append(f"cost: not measured: {c['why']}")
    else:
        how, noise = c["how"], c["noise_floor"]
        bench = how["client"] == S.BENCH
        at = (f"at {how['rung']} users" if how.get("sessions") else f"at concurrency {how['rung']}") if bench else ""
        at = (f"{at}, where stock's rungs {'show' if how['saturated'] else 'do NOT show'} it saturated"
              if bench else f"at {how['rate']:g} requests a second, which stock did not keep up with")
        reps = f"{how['reps']} paired rep{'' if how['reps'] == 1 else 's'}" if bench else f"{how['reps'][0]} and {how['reps'][1]} reps"
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
    lines += _advice(r["advice"]) if r["advice"]["measured"] else [f"advice: not measured: {r['advice']['why']}"]
    done = sum(bool(r[k]["measured"]) for k in ("tokens", "cost", "advice"))
    early = [name for name, s in ran["arms"].items() if not s.get("ok")]
    why = failed(r)
    lines.append(f"prod_repro: {done} of 3 lines measured"
                 + ("" if not why else "; FAILED: the replay differs from what was served" if t["measured"] else
                    "; FAILED: a replay was asked for and the tokens line is not measured")
                 + (f"; not served whole: {', '.join(early)}" if early else ""))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("arms", help="the arms a job serves by default: stock, stock again, ours")
    q.add_argument("--model", help="the model the job serves, as the benchmark names it")
    q.add_argument("--out", type=Path, help="an earlier job's out/, read again: the arms it is read as")
    for name, what in (("replay-env", "what the replay of an arm's window runs under beside the job's environment, NAME=VALUE a line"),
                       ("replay-ran", "write OUT/replay_engine.json: what the replay's engine ran under")):
        q = sub.add_parser(name, help=what)
        q.add_argument("--ours", required=True)
        q.add_argument("--kept", type=Path, required=True, help="the replay step's directory on the node, outside the run's")
    q.add_argument("--private", type=Path, help="where the job's arms kept their own files (the benchmark's PRIVATE)")
    q.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report", help="the settings and the three lines of a finished job's out/")
    p.add_argument("out", type=Path)
    p.add_argument("--stock", help="the arms default to the three OUT is read as (default_arms)")
    p.add_argument("--again", help="stock in a second server lifetime, the noise floor ('' for none)")
    p.add_argument("--ours")
    p.add_argument("--load", required=True)
    p.add_argument("--replayed", metavar="PASS",
                   help="the pass of ours the replay re-ran, as the benchmark names it (default: the load; by vLLM's "
                        "client a pass is LOAD@RUNG)")
    p.add_argument("--rep", type=int, default=1, help="the rep of the pass the replay re-ran")
    p.add_argument("--replay", type=Path, help="the replay's directory (default: OUT/replay-compiled)")
    p.add_argument("--expect-replay", action="store_true",
                   help="the job asked for a replay: fail where the tokens line is not measured")
    p.add_argument("--advice-trace", type=Path, metavar="DIR", help="the trace directory of the certified serve")
    p.add_argument("--advice-stops", metavar="ID,...", help="the registered stop ids the advice is read under, EOS first")
    p.add_argument("--target", type=float, default=TARGET, metavar="PERCENT")
    p.add_argument("--json", action="store_true", help="print the JSON object alone")
    a = ap.parse_args(argv)
    if a.cmd == "arms":
        print(" ".join(default_arms(a.model, a.out)))
        return 0
    if a.cmd == "replay-env":
        env = replay_env(a.ours, a.kept, os.environ.get("PYTHONPATH", ""))
        why = plugins_load(a.ours, env) if env else None
        if why:
            print(f"prod_repro: a vLLM general plugin of {a.ours} doesn't load in the replay's environment: {why}", file=sys.stderr)
            return 2
        print("".join(f"{k}={v}\n" for k, v in env.items()), end="")
        return 0
    if a.cmd == "replay-ran":
        (a.out / REPLAY_ENGINE).write_text(json.dumps(replay_ran(a.ours, a.kept, a.private), indent=1))
        return 0
    stock, again, ours = default_arms(out=a.out)
    r = report(a.out, a.stock or stock, a.ours or ours, (again if a.again is None else a.again) or None, a.load, a.rep,
               a.replay, a.target, a.replayed, a.expect_replay, a.advice_trace, a.advice_stops)
    r["failed"] = failed(r)
    (a.out / "prod_repro.json").write_text(json.dumps(r, indent=1))
    (a.out / "prod_repro.txt").write_text(text(r))
    print(json.dumps(r) if a.json else text(r), end="\n" if a.json else "")
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
