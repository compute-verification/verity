"""vLLM at its serving settings, replayed: what one served job shows, as the settings it ran with and three lines (README.md).

    prod_repro.py arms [--model ID] [--out OUT]
    prod_repro.py limit --model ID
    prod_repro.py report OUT [--stock ARM] [--ours ARM] [--again ARM] [--beside ARM ...] --load NAME
                         [--replayed PASS] [--rep 1] [--replay DIR] [--expect-replay] [--private DIR]
                         [--target PERCENT] [--json]

``report`` reads a finished job's out/ as the serving cost benchmark leaves it (`benchmarks/serve_tax`: a directory an
arm, and its replay's), prints the report and writes it to OUT/prod_repro.json and OUT/prod_repro.txt:

- first, the context limit the arms' launch lines name, and for a model `LIMITS` says so of, that vLLM does not
  start with the limit left to it;
- how far each arm's serve and the replay got, from their own records: an arm that failed with its error, one that
  stopped early, a replay that did not end with the last line of its log; and what the replay's engine ran under, as
  the replay's own record says (the served arm's other vLLM general plugins and its environment: the canonical DSA
  top-k's), with the top-k's counters of the replay's processes and of serving's where the job kept them
  (``--private``, the benchmark's PRIVATE; `topk_kept`);
- each pass of the load every arm served, a line a pass, by the counters the benchmark read off its server around
  it (`passes`): its engine steps, its drafts and the share of their tokens accepted, its output tokens and its
  seconds; then how each arm's requests were sampled and whether its client sent their seeds, and for each
  directory whether its server was on the canonical top-k and what the top-k's processes counted there. It is in no
  verdict. The arms served beside the three (``--beside``, the runner's ``--arms``: arms that differ from stock or
  from ours in one thing) are in what ran and in these lines, and in no line of the three;
- the settings each of the two arms ran with, read from its launch line, its engine's own configuration (the step
  trace's header) and its counters: a setting the serve did not turn on is ``off``, and one its files can't confirm
  says so; and the adapter's record of the value settings in force in each server's environment and in the replay's
  engine's, where the benchmark kept it (`value_settings`): how many variables are set, the record's digest, and by
  name the ones ours's differs from stock's in and the replay's from ours's;
- ``tokens``: the served window the replay re-ran (`replay.window`), how many of its requests and served tokens the
  replay's kept tokens were compared with (`replay.token_diff`) and how many differ, with the steps that differ and
  the requests of the window that began from a prefix-cache hit. Where they differ: the first token that does (its
  request, position, served and replayed ids and the step that emitted it) and the first step whose rows do. The
  window is reproduced where every served token equals its replayed one and every served step is in the replay with
  equal rows, but for the steps after a request's end (`after_end`), which are counted and printed and are no
  difference; a replayed step serving did not run, a missing step that emitted a token or holds a request that had
  not ended, and rows that differ all fail;
- ``cost``: what serving with our changes (``--ours``) costs over stock (``--stock``) on the load, a share of stock's
  serving time. On a sessions load it is the benchmark's own cost (`serve_tax.session_cost`): the mean over the
  server lifetimes an arm (the job's ``<arm>.K`` directories, `arm_dirs`) of each one's cost, with its 95% interval
  over the lifetimes, and it is under the target only where the benchmark says that interval's upper end is, which
  a job of one lifetime an arm can't say; beside it, each lifetime's cost, the spread, the cost over the passes
  taken as independent and the noise floor (stock served a second time, ``--again``, else stock's own lifetimes).
  On another load it is the cost over the passes taken as independent, by vLLM's client the benchmark's paired
  ratio at the rung where stock is saturated (`serve_tax.serving_cost`), by the corpus client its tax over reps
  (`serve_tax.tax`), which is a cost only where stock did not keep up with the arrivals; that has no verdict;
- ``advice``: what the schedule of the pass the replay re-ran costs as advice, in bits a request, by the scheduler
  replay's loader (`ENCODINGS`: its ``draft_table`` of that pass's records): the arrivals, which a serve under the
  certified tick does not send, and the count of drafts each row kept, charged at the bits of its code, which the
  loader builds and decodes back, with the figure where the audited block derives the count beside it; and how the
  pass's sampled requests were seeded.

A line is measured from the job's files, or it says ``not measured`` and why; a line whose files can't be read says
so and the rest of the report stands. Nothing here measures: the metrics, the cost and its interval, the comparison
and the advice's split are the benchmark's and the adapter's functions. It exits 1 where the replay differs from what
was served, naming each differing request and its first position, and where a replay was asked for
(``--expect-replay``) and the tokens line is not measured; 0 otherwise: any other line that is not measured is not a
failure, and the last line counts the lines that are measured.

Under research run, where OUT is the run's own out/, the run's result.json names what the report rests on among its
artifacts (`declare`: the report's two files and the replay's record).

``arms`` prints the three arms a job serves by default (`default_arms`): the arms at vLLM's published serving settings
(`PROD`) for a model whose recipe the benchmark holds, else stock compiled vLLM, stock again and the step record
(`TODAY`); with ``--out``, the three an earlier job's out/ is read as, which ``report`` reads it as too where none is
named.

``limit`` prints the context limit a job serves the model at where it names none (`LIMITS`), or nothing.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/serve_tax"))
import replay as R  # noqa: E402 (benchmarks/serve_tax, put on sys.path above)
import serve_tax as S  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402 (the tree's verity/configuration/config, which serve_tax put on sys.path)

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
#: how the replayed window was chosen: no audit samples it yet
WINDOW = "every request of the load's rep, a window the command names: no audit's coins sampled it"
#: the scheduler replay's loader, which reads a served pass's schedule as advice: a step trace's records for one
#: pass (``span``), as a trace (``load_step_trace``), and the bits its two classes of advice cost (``draft_table``)
ENCODINGS = ROOT / "benchmarks/network_e2e/encodings/replay.py"
#: what the kept count costs where it is not sent, and what that rests on
DERIVED = "the kept count is 0 where the audited block derives it, and no Definition holds the accept test yet"
NO_CIRCUIT = "no: nothing in this job compiles the settings' circuit"
ON, OFF, UNCONFIRMED = "on", "off", "not confirmed"
#: why a cost is not shown under its target where stock and ours did not serve at the same settings: the figure is
#: then not our changes' cost
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
#: what the report keeps in OUT of the canonical top-k's counters, which its processes keep outside it (`topk_kept`)
TOPK_KEPT = "topk_counters.json"
#: by model, as the benchmark names it: what is known of leaving the context limit to vLLM (``auto``: whether vLLM
#: then starts, at which settings, why not, and the run that showed it)
LIMITS = HERE / "limits.json"
#: the cost's verdict against its target, and the benchmark's words for each on a sessions load
#: (`serve_tax.session_cost`: by the interval over server lifetimes, "meets" where its upper end is under the target,
#: "misses" where its lower end is over it)
UNDER, OVER, NOT_SHOWN = "under", "over", "not shown"
VERDICTS = {"meets": UNDER, "misses": OVER, "cannot tell": NOT_SHOWN}
#: why a job of one server lifetime an arm has no verdict
ONE_LIFETIME = "each arm has one server lifetime, and the verdict reads the interval over lifetime pairs"
#: why a job with an arm served in more lifetimes than one has none either, where all but one pair of them is without
#: a pass both arms counted (an arm stopped before its lifetime's first counted pass, or not served a lifetime the
#: other was: the benchmark's STOP_AT)
ONE_PAIR = ("only one pair of the arms' lifetimes has a pass both arms counted, and the verdict reads the interval "
            "over lifetime pairs")
#: why a cost with such an interval is not shown to be under its target
NOT_UNDER = "the upper end of the interval over lifetime pairs is not under it"
#: why a load that is not a sessions load has no verdict, however many lifetimes served it
NOT_SESSIONS = ("the interval over lifetime pairs is the benchmark's for a sessions load (serve_tax.session_cost), and "
                "this load is not one")
#: why a cost on a sessions load is not shown whatever its interval: the benchmark's own cost has no verdict there
#: (`serve_tax.session_cost`'s ``same_work``)
UNEQUAL = ("the arms did not do the same work (one left a request unserved, they served another count of output "
           "tokens, or their prefix caches served different shares of the prompts in some rep)")
#: the hex digits printed of a record's SHA-512 (the report's JSON holds it whole)
DIGEST_SHOWN = 12
#: the line where the benchmark's own cost has none on the requests that went on past an end-of-text token
#: (`serve_tax.past_end_text`): an arm's passes hold that count only where its server kept the served ids and its
#: record names the model's end-of-text ids
NOT_COUNTED = "  past an end-of-text token: not counted, no arm's passes hold a count of the requests that went on past one"


def default_arms(model_id: str | None = None, out: Path | None = None) -> tuple[str, str, str]:
    """The arms of a job that names none: `PROD` where the benchmark has them and holds vLLM's recipe for the model (it
    refuses an arm at the published settings for a model with none), else `TODAY`. An earlier job's ``out`` is read as
    `PROD` where it holds that set's arm of ours."""
    if out is not None:
        return PROD if arm_dirs(out, PROD[2])[0].is_dir() else TODAY
    model = S.MODELS.get(model_id)
    return PROD if all(a in S.ARMS for a in PROD) and model is not None and getattr(model, "published", ()) else TODAY


def arm_dirs(out: Path, arm: str) -> list[Path]:
    """An arm's directories in a job's out/: its own, or, where the benchmark served the job's arms in server
    lifetimes (its LIFETIMES), each lifetime's (``<arm>.K``), in the lifetimes' order. The arm's own where it has
    neither."""
    if (out / arm).is_dir():
        return [out / arm]
    k = len(arm) + 1
    found = {int(d.name[k:]): d for d in out.glob(f"{arm}.*") if d.name[k:].isdigit() and d.is_dir()}
    return [found[n] for n in sorted(found)] or [out / arm]


def limits(model_id: str | None) -> dict:
    """What `LIMITS` holds for a model; empty for one it does not hold."""
    return json.loads(LIMITS.read_text()).get(model_id or "", {})


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
            "log_error": first_error(arm_dir / "server" / "server.log") if meta.get("error") else None,
            "passes": len(passes), "requests": sum(x.get("requests") or 0 for x in passes),
            "unfinished": sum(x.get("unfinished") or 0 for x in passes),
            "ok": bool(passes) and not meta.get("error") and not meta.get("stopped")}


def first_error(log: Path) -> str | None:
    """The first line of a log that is an exception's own (``SomeError: what``), from the exception's name on and cut
    to `LINE_MAX` characters: under a server that failed, the first cause its log names, where the record's error is
    the last thing that failed for it. None without the log, "" where it names none."""
    if not log.is_file():
        return None
    for x in log.read_text(errors="replace").splitlines():
        m = re.search(r"\b[\w.]+(?:Error|Exception)\b: .*", x)
        if m:
            return m.group(0).strip()[:LINE_MAX]
    return ""


def last_line(log: Path) -> str | None:
    """The last line of a log that says anything, cut to `LINE_MAX` characters; None without the log."""
    if not log.is_file():
        return None
    lines = [x.strip() for x in log.read_text(errors="replace").splitlines() if x.strip()]
    return lines[-1][:LINE_MAX] if lines else ""


def replay_state(replay_dir: Path, expected: bool) -> dict:
    """Whether the replay ended (it wrote replay.json), and where it did not, the last line of its log, which the
    benchmark's runner keeps beside its directory."""
    log = replay_dir.with_name(replay_dir.name + ".log")
    return {"expected": expected, "ended": (replay_dir / "replay.json").is_file(), "log": str(log) if log.is_file() else None,
            "last_line": last_line(log)}


# ------------------------------------------------------------------------------------------- the replay's engine

def topk_counters(dirs: list[Path]) -> dict | None:
    """The canonical top-k's counters over the processes that kept theirs in ``dirs`` (`dsa_topk.log_to`: a file a
    process, written every few thousand calls and at its exit, so each figure is a lower bound): the processes, the
    calls of the indexer's top-k and the rows of more than its keys that were rewritten. None where no process kept
    any."""
    recs = [json.loads(f.read_text()) for d in dirs if d.is_dir() for f in sorted(d.glob("dsa_topk.*.json"))]
    if not recs:
        return None
    return {"processes": len(recs), "calls": sum(sum(r["calls"].values()) for r in recs),
            "rows_long": sum(dev.get("rows_long", 0) for r in recs for dev in r["rows"].values())}


def topk_kept(out: Path, private: Path | None, ours: str, replay: str) -> dict:
    """The canonical top-k's counters of the replay's processes and of serving's (`topk_counters`). The processes keep
    them on the node, outside the run's directory, under the job's private one (the benchmark's PRIVATE: an arm's
    under its directory's name, each of its lifetimes' under its own, and a replay's under its directory's). With
    ``private`` they are read there and kept in OUT
    (`TOPK_KEPT`); without, they are the ones an earlier report of the job kept. Serving's are also the kept ones where
    the private directory holds none for the arm: the arm's processes ended with it, in this job or in the earlier one
    whose arm this job replays (run.sh --replay-of, which carries that job's). ``kept`` says whether either holds."""
    f = out / TOPK_KEPT
    before = json.loads(f.read_text()) if f.is_file() else {}
    if private is None:
        return {"kept": True, **before} if before else {"kept": False, "replay": None, "served": None}
    log = S.ARMS[ours].env.get(TOPK_LOG) if ours in S.ARMS else None
    kept = {name: topk_counters([Path(log.replace(S.PRIVATE, str(private / d))) for d in ds]) if log else None
            for name, ds in (("replay", [replay]), ("served", [d.name for d in arm_dirs(out, ours)]))}
    kept["served"] = kept["served"] or before.get("served")
    f.write_text(json.dumps(kept | {k: before[k] for k in ("dirs",) if k in before}, indent=1))
    return {"kept": True, **kept}


def replay_engine(out: Path, ours: str, replay_dir: Path, private: Path | None) -> dict | None:
    """What the replay's engine ran under, as the replay's own record says (replay.json's ``served_under``: the served
    arm's other vLLM general plugins and its environment, which `replay.served_under` loads and sets before it builds
    the engine; none for an arm with no such plugin), and where that is the canonical top-k, its counters
    (`topk_kept`). None where the replay did not end or its record does not say."""
    f = replay_dir / "replay.json"
    rec = json.loads(f.read_text()) if f.is_file() else {}
    if "served_under" not in rec:
        return None
    under = rec["served_under"]
    return {"arm": ours, "under": under,
            "topk": topk_kept(out, private, ours, replay_dir.name) if under and under["env"].get(TOPK) else None}


# ----------------------------------------------------------------------------------------------------- every pass

def topk_by_dir(out: Path, private: Path | None, dirs: list[Path]) -> dict | None:
    """The canonical top-k's counters of each directory of ``dirs`` whose server was started under it (its record's
    environment names where its processes keep them, `TOPK_LOG`), by the directory's name (`topk_counters`; None for
    one whose processes kept none). With ``private`` they are read there, each directory's under its name, and kept
    in OUT beside the replay's (`TOPK_KEPT`, under ``dirs``); without, they are the ones the job's own report kept,
    None where it kept none. A directory's are also the kept ones where the private directory holds none for it, as
    serving's are in `topk_kept`."""
    f = out / TOPK_KEPT
    before = json.loads(f.read_text()) if f.is_file() else {}
    if private is None:
        return before.get("dirs")
    logs = {d.name: log for d in dirs if (log := (json.loads((d / "meta.json").read_text()).get("env") or {}).get(TOPK_LOG))}
    kept = {name: topk_counters([Path(log.replace(S.PRIVATE, str(private / name)))]) or (before.get("dirs") or {}).get(name)
            for name, log in logs.items()}
    if kept:
        f.write_text(json.dumps({**before, "dirs": kept}, indent=1))
    return kept


def passes(out: Path, dirs: dict[str, list[Path]], per: dict[str, list[dict]], load: str, private: Path | None) -> dict:
    """Every pass of ``load`` the job's arms served, in the order they were served where every pass's record says
    when it began (else the arms' order), each with the counters the benchmark read off its server before and after
    it (the pass's own record: `serve_tax.SCHEDULE`): its engine steps and, where the engine drafted, its drafts,
    the tokens they proposed and the tokens of them the model accepted; beside them its output tokens and its
    seconds as the cost counts them, and whether it is one its lifetime leaves out.
    With them, by arm, whether its requests were sampled and whether its client sent their seeds (its record), and by
    directory, whether its server was started on the canonical top-k (its record's environment) and what the top-k's
    processes counted there (`topk_by_dir`). It is in no verdict: it is what an arm served beside stock and ours (the
    runner's --arms) is read by."""
    rows, sampling, on = [], {}, {}
    for a, ms in per.items():
        for d, m in zip(dirs[a], ms):
            rec = json.loads((d / "meta.json").read_text())
            sampling[a] = {"sampled": bool(rec.get("sampled")), "seeded": bool(rec.get("sampled")) and bool(m["seeded"])}
            on[d.name] = bool((rec.get("env") or {}).get(TOPK))
            for name, reps in m["loads"].items():
                for r in reps if of_load(name, load) else ():
                    began = json.loads((d / name / f"rep-{r['rep']}" / "meta.json").read_text())
                    c = began.get("schedule") or {}
                    rows.append({"dir": d.name, "arm": a, "pass": name, "rep": r["rep"], "left_out": r["rep"] <= m["left_out"],
                                 **{k: c.get(k) for k in ("steps", "drafts", "draft_tokens", "accepted_tokens")},
                                 "output_tokens": r.get("output_tokens"), "seconds": r.get("span_s"),
                                 "t0_unix": began.get("t0_unix")})
    if all(x["t0_unix"] is not None for x in rows):
        rows.sort(key=lambda x: x["t0_unix"])
    return {"measured": True, "passes": rows, "sampling": sampling,
            "topk": {"on": on, "kept": topk_by_dir(out, private, [d for a in per for d in dirs[a]])}}


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


def value_settings(dirs: dict[str, list[Path]], stock: str | None, ours: str, replay_dir: Path) -> dict:
    """The adapter's record of the value settings in force (`serve_tax.value_settings`: each variable set that can
    change what an engine computes, a credential by name only, and their SHA-512), as the benchmark kept it of the
    environment each server was started with (``settings`` in a lifetime's meta.json) and of the replay's engine's
    (in replay.json): each directory's record whole (``arms``, ``replay``; None for one that holds none, as a job
    served before the benchmark kept it), the arms whose lifetimes do not hold one record (``lifetimes_differ``),
    and the variables, by name, that one side alone sets or sets to another value (``differ``: ours's first
    lifetime against stock's, the replay against it; None where a side has no record). It is beside the settings
    and in no verdict: ours's server is given variables stock's is not."""
    def kept(f: Path) -> dict | None:
        return json.loads(f.read_text()).get("settings") if f.is_file() else None

    def differ(a: dict | None, b: dict | None) -> list[str] | None:
        if a is None or b is None:
            return None
        return sorted(k for k in a["values"].keys() | b["values"].keys() if a["values"].get(k) != b["values"].get(k))

    arms = {a: {d.name: kept(d / "meta.json") for d in dirs.get(a, [])} for a in (stock, ours) if a}
    first = {a: next(iter(recs.values()), None) for a, recs in arms.items()}
    replay = kept(replay_dir / "replay.json")
    return {"arms": arms, "replay": replay,
            "lifetimes_differ": sorted(a for a, recs in arms.items() if len({r and r["digest"] for r in recs.values()}) > 1),
            "differ": {"ours_from_stock": differ(first.get(stock), first.get(ours)),
                       "replay_from_ours": differ(first.get(ours), replay)}}


# --------------------------------------------------------------------------------------------------------- the tokens

def replayed_steps(replay_dir: Path) -> dict[int, list] | None:
    """The rows of each step the replay ran, from the steps it kept; None where it kept none."""
    f = next((p for p in (replay_dir / "replay_steps.jsonl", replay_dir / "replay_steps.jsonl.gz") if p.is_file()), None)
    if f is None:
        return None
    with (gzip.open if f.suffix == ".gz" else open)(f, "rt", encoding="utf-8") as lines:
        return {s["step"]: s["rows"] for s in map(json.loads, filter(str.strip, lines))}


def after_end(w: dict, kept: dict[int, list] | None) -> list[int]:
    """The served steps of a window that are after a request's end: the replay ran no such step, the step emitted no
    token, and each of its rows is of a request that finished in an earlier step. The scheduler had planned such a step
    before it learned the request had ended, and serving dropped what it computed. None where the replay kept no steps
    to say which it ran."""
    done: set[str] = set()
    out = []
    for s in w["steps"]:
        if kept is not None and s["step"] not in kept and not s.get("emitted") and s["rows"] and all(
                r[0] in done for r in s["rows"]):
            out.append(s["step"])
        done |= set(s.get("finished") or ())
    return out


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


def tokens(dirs: list[Path], ours: str, load: str, rep: int, replay_dir: Path, state: dict) -> dict:
    """The tokens line: the served window of ``ours``'s pass ``load`` rep ``rep`` against the tokens its replay kept.
    ``dirs`` are the arm's directories (`arm_dirs`): of an arm served in lifetimes, the window is the pass of the
    lifetime the replay's record names. Not measured without a replay, where the replay is of another window, or
    where the window can't be read back. Each differing request has its first differing position, the ids served and
    replayed there and the step that emitted it while serving, the earliest first."""
    meta = json.loads((dirs[0] / "meta.json").read_text())
    if not state["ended"]:
        if state["expected"]:
            return not_measured(f"the replay did not end (no replay.json in {replay_dir})" + (
                f"; its log ends: {state['last_line']} ({state['log']})" if state["log"] else "; it left no log"))
        return not_measured("no replay ran" + (": none was asked for (--replay PASS names the pass of vLLM's client to "
                                               "re-run, from an arm that keeps each request's ids)"
                                               if meta.get("client") == S.BENCH else f" (no replay.json in {replay_dir})"))
    rec, run = json.loads((replay_dir / "replay.json").read_text()), meta.get("run_id")
    of = (Path(rec["arm_dir"]).name, rec["load"], rec["rep"], rec.get("served_run_id"))
    arm_dir = replayed_dir(dirs, replay_dir)
    if arm_dir is None or of[1:] != (load, rep, run):
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
    # the steps that differ are the record's (replay.compare), less the served ones after a request's end, which the
    # kept steps name; a record that counts other steps than the kept ones show disagrees with them
    kept, rows = replayed_steps(replay_dir), {s["step"]: s["rows"] for s in w["steps"]}
    ended = after_end(w, kept)
    differ = None if kept is None else [k for k in sorted(set(rows) | set(kept)) if rows.get(k) != kept.get(k)]
    if differ is None:
        steps, first = rec["steps_differing"], rec["first_differing_step"]
    else:
        disagree = disagree or (len(differ), differ[0] if differ else None) != (rec["steps_differing"], rec["first_differing_step"])
        counted = [k for k in differ if k not in ended]
        steps, first = len(counted), counted[0] if counted else None
    return {"measured": True, **n, "differing": diff[:R.DIFFERING_MAX],
            "steps_served": rec["steps_served"], "steps_differing": steps, "first_differing_step": first,
            "steps_after_end": ended, "first_step": first_step(w, replay_dir, first),
            "record_disagrees": disagree,
            "reproduces": not diff and not steps and not disagree,
            "cache_hits": {"requests": len(hits), "tokens": sum(hits.values()), "longest": max(hits.values(), default=0)},
            "sampling": [{"temperature": t, "top_p": p, "seeded": seeded, "requests": k}
                         for (t, p, seeded), k in sorted(sampling.items(), key=lambda x: -x[1])],
            "past_topk": past_topk(w, meta.get("checkpoint"), diff),
            "window": {"arm": ours, "lifetime": json.loads((arm_dir / "meta.json").read_text()).get("lifetime"),
                       "load": load, "rep": rep, "run": run, "chosen": WINDOW},
            "served_mode": rec["served_mode"], "replay_mode": rec["mode"], "replay_run": rec.get("run_id"),
            "served_window_s": rec["served_window_s"], "replay_wall_s": rec["replay_wall_s"],
            "engine_ready_s": rec["engine_ready_s"]}


# ----------------------------------------------------------------------------------------------------------- the cost

def rep_dir(dirs: list[Path], r: dict, load: str) -> Path:
    """The directory of an arm's rep of ``load``: of an arm served in lifetimes (`arm_dirs`), the pass's in its
    lifetime's directory (`serve_tax.lifetimes` numbers such an arm's reps over its lifetimes)."""
    d = dirs[0] if "lifetime" not in r else next(d for d in dirs if d.name.rpartition(".")[2] == str(r["lifetime"]))
    return d / load / f"rep-{r.get('pass', r['rep'])}"


def kept_up(dirs: list[Path], arm: dict, load: str) -> dict:
    """Whether an arm the corpus client sent ``load`` to kept up with its arrivals, as the benchmark says it of a load
    sent as it arrives (`serve_tax.KEPT_UP`): it served every request, and each rep ended within that share of the
    time its requests arrived over after the last one arrived. ``tail_s`` is the longest such end."""
    tails, all_served = [], True
    for r in arm["loads"][load]:
        arrived = json.loads((rep_dir(dirs, r, load) / "meta.json").read_text())["arrivals_span_s"]
        tails.append((r["span_s"] - arrived, arrived))
        all_served &= r["ok"] == r["sent"]
    return {"kept_up": all_served and all(t <= S.KEPT_UP * a for t, a in tails), "tail_s": max(t for t, _ in tails),
            "arrivals_span_s": max(a for _, a in tails)}


#: what the benchmark's own cost says of each side beside the cost, which the cost line prints and adjusts nothing
#: for: the engine steps a pass, the share of its draft tokens accepted (percent) and the tokens a draft gave
SIDE = ("steps", "draft_accepted_share", "tokens_per_draft")


def _frac(x: float | None) -> float | None:
    """A figure of the benchmark's own cost, which is percent, as a fraction."""
    return None if x is None else x / 100


def cost(out: Path, by: dict[str, dict], stock: str, ours: str, again: str | None, load: str, target: float) -> dict:
    """The cost line: ``ours`` over ``stock`` on ``load``, a fraction of stock's serving time, and whether it is under
    ``target`` percent. On a sessions load every figure is the benchmark's own cost's (`serve_tax.session_cost`): the
    cost is the mean over the server lifetimes an arm of each one's cost, its interval is over the lifetimes, and the
    verdict is the benchmark's, which reads that interval alone (`VERDICTS`). So a job of one lifetime an arm, which
    has no such interval, shows nothing, nor does one with one pair of lifetimes that has a pass both arms counted
    (``lifetimes`` holds the pairs counted, ``pairs``, and the most lifetimes an arm was served in, ``served``), nor
    one whose arms did not do the same work. Beside it and in no
    verdict: the cost over the passes taken as independent with its interval (``passes``), each lifetime's cost, the
    spread between lifetimes and between a lifetime's passes, the same cost under the benchmark's other readings of
    a pass's rate of work (``readings``), each side's engine steps and accepted drafts (``sides``: `SIDE`) and its
    requests that went on past an end-of-text token (``past_end``: each side's count, None for one whose server kept
    no ids, and the benchmark's own line of them, `serve_tax.past_end_text`), all
    in ``lifetimes``, and the noise floor, a lifetime
    pair its unit too: stock again over stock where a twin of stock was served, else stock's own lifetimes, each
    second one over the one before it. On another load the cost is the one over the passes taken as independent, the
    noise floor is ``again`` over ``stock`` measured the same way, and there is no verdict. ``leaves_out_zero`` says
    whether the floor's own interval leaves out 0: stock then differs from stock. Not measured where an arm did not
    serve the load, where an arm left requests unserved (the arms then served different tokens), where the
    benchmark's own cost refuses the arms, or where the corpus client's arrivals did not saturate stock."""
    missing = [a for a in (stock, ours) if a not in by or not reps_of(by[a], load)]
    if missing:
        return not_measured(f"{', '.join(missing)} did not serve load {load}")
    ref, mine, twin, same = by[stock], by[ours], by.get(again) if again else None, None
    unserved = {a["arm"]: sum(r["sent"] - r["ok"] for r in reps_of(a, load)) for a in (ref, mine)}
    if any(unserved.values()):
        return not_measured("the arms did not serve the same tokens: " + ", ".join(
            f"{a} left {n} requests unserved" for a, n in unserved.items() if n))
    served = max(len(a.get("lifetimes") or [1]) for a in (ref, mine))
    lives = {"pairs": len(set(ref.get("lifetimes") or [1]) & set(mine.get("lifetimes") or [1])), "served": served, "ci95": None}
    verdict = NOT_SHOWN
    if load in S.bench_loads(ref):
        c = S.serving_cost(ref, mine, load)
        if c is None:
            return not_measured(f"{stock} and {ours} share no rung or no rep of load {load}")
        floor = S.serving_cost(ref, twin, load, c["rung"]) if twin else None
        how = {"client": S.BENCH, "rung": c["rung"], "reps": c["reps"], "saturated": c["saturated"],
               "sessions": bool(getattr(S.LOADS.get(load), "sessions", False)),
               "by": "the paired ratio of output tokens a second over reps at one rung (serve_tax.serving_cost)"}
        noise = floor and {"arm": again, "over": "passes", "value": floor["value"], "ci95": floor.get("ci95")}
        value = c["value"]
        if how["sessions"]:
            try:
                own = S.session_cost([ref, mine, *([twin] if twin else [])], ours, stock, target)
            except SystemExit as e:
                return not_measured(f"the benchmark's own cost on a sessions load refuses these arms: {e}", how=how)
            if own is None or own["load"] != load:
                return not_measured(f"the benchmark's own cost on a sessions load is of {own['load'] if own else 'none'}, "
                                    f"not of {load}: a job serves one sessions load", how=how)
            same, value, verdict, n = own["same_work"], _frac(own["value"]), VERDICTS[own["verdict"]], own["noise_floor"]
            both = dict(zip((stock, ours), (own["sides"]["reference"], own["sides"]["arm"])))
            lives = {"pairs": own["lifetimes"], "served": served, "ci95": own["ci95"] and [x / 100 for x in own["ci95"]],
                     "each": own["by_lifetime"] and [{**x, "value": _frac(x["value"])} for x in own["by_lifetime"]],
                     "spread_points": own["spread"], "left_out": own["left_out"],
                     "readings": {k: _frac(v) for k, v in (own.get("readings") or {}).items()},
                     "sides": {a: {k: x.get(k) for k in SIDE} for a, x in both.items()},
                     "past_end": {"said": S.past_end_text(own), "sides": {a: x.get("past_end") for a, x in both.items()}}}
            noise = n and {"arm": n["arm"], "over": "lifetimes", "value": _frac(n["value"]),
                           "ci95": n["ci95"] and [x / 100 for x in n["ci95"]], "pairs": n["pairs"],
                           "own": "lifetimes" in n}
    else:
        def tok(a: dict) -> dict:
            return S.summary([r.get("output_tok_per_s") for r in a["loads"][load]])

        c = S.tax(tok(ref), tok(mine))
        if c is None:
            return not_measured(f"{stock} or {ours} served no token of load {load}")
        floor = S.tax(tok(ref), tok(twin)) if twin and load in twin["loads"] else None
        noise = floor and {"arm": again, "over": "passes", "value": floor["value"], "ci95": floor.get("ci95")}
        value = c["value"]
        kept = kept_up(arm_dirs(out, stock), ref, load)
        how = {"client": S.CORPUS, "rate": ref["loads"][load][0]["rate"], "reps": c["n"], **kept,
               "by": "stock's mean output tokens a second over ours, less one, over reps (serve_tax.tax)"}
        if kept["kept_up"]:
            return not_measured(f"{stock} kept up with the arrivals (its longest rep ended {kept['tail_s']:.1f} s after "
                                f"the last of {kept['arrivals_span_s']:.1f} s of arrivals), so its tokens a second "
                                "follow the arrivals and are not a cost: send faster", reading=c, how=how)
    if noise:
        ends = noise["ci95"]
        noise["leaves_out_zero"] = bool(ends) and not ends[0] <= 0 <= ends[1]
    return {"measured": value is not None, "value": value, "passes": {"value": c["value"], "ci95": c.get("ci95")},
            "how": how, **({} if value is not None else {"why": "an arm served no token in some rep"}),
            "lifetimes": lives, "noise_floor": noise or None, "target": target, "same_work": same,
            "verdict": NOT_SHOWN if same is False else verdict}


# --------------------------------------------------------------------------------------------------------- the advice

def encodings() -> object:
    """The scheduler replay's loader (`ENCODINGS`), loaded from its file under a name of its own, since it is a second
    replay.py and the one on the path is the benchmark's, with the adapter it reads on the path."""
    name = "prod_repro_encodings"
    if name in sys.modules:
        return sys.modules[name]
    adapter = str(ROOT / "adapters/vllm")
    if adapter not in sys.path:
        sys.path.insert(0, adapter)
    spec = importlib.util.spec_from_file_location(name, ENCODINGS)
    if spec is None or spec.loader is None:
        raise ImportError(f"{ENCODINGS} is not a module")
    sys.modules[name] = mod = importlib.util.module_from_spec(spec)  # its dataclasses look their module up by name
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def replayed_dir(dirs: list[Path], replay_dir: Path) -> Path | None:
    """The directory of ours whose pass the replay re-ran: the one its record names, which of an arm served in
    lifetimes is that lifetime's (`arm_dirs`); the first where the replay left no record, the one the runner hands
    it; None where the record names a directory that is none of ours's."""
    f = replay_dir / "replay.json"
    if not f.is_file():
        return dirs[0]
    name = Path(json.loads(f.read_text())["arm_dir"]).name
    return next((d for d in dirs if d.name == name), None)


def seeds(w: dict, meta: dict) -> dict:
    """How a window's sampled requests were seeded, from each one's own arrival (its ``sampling``): the requests at a
    temperature above 0, those of them with no seed in their arrival, and whether the arm's record says the client
    sent each seed (``seeded``). What a sampled request with no seed there was sampled with is in no record."""
    sampled = [r for r in w["requests"] if r["sampling"].get("temperature")]
    return {"sampled": len(sampled), "unseeded": sum(r["sampling"].get("seed") is None for r in sampled),
            "by_client": bool(meta.get("seeded"))}


def advice(dirs: list[Path], ours: str, load: str, rep: int, replay_dir: Path) -> dict:
    """The advice line: what the schedule of ``ours``'s pass ``load`` rep ``rep``, the pass the replay re-ran and the
    tokens line compares (`replayed_dir`), costs as advice, in bits over the pass, as the scheduler replay's loader
    gives it (`ENCODINGS`: ``draft_table``): ``arrivals``, the residual advice of when each request arrived, which a
    serve under the certified tick does not send, and ``kept_count``, the drafts each row kept, as the loader's dict
    has it (``charged``, the bits of the code it builds and decodes back; ``alone``, the counts' entropy over the rows
    that verified drafts, which is not sent; ``derived`` where the audited block computes it). ``bits`` is what is
    charged: the arrivals, unless the serve is under the tick, and the kept count's ``charged``. Not measured where ours holds no such pass or it can't be read back, or where the loader
    refuses it: a schedule its replay does not reproduce, which it names the step of, or a serve with no draft."""
    arm_dir = replayed_dir(dirs, replay_dir)
    if arm_dir is None or not (arm_dir / load / f"rep-{rep}" / "meta.json").is_file():
        return not_measured(f"{ours} holds no pass {load} rep {rep}, the one the replay re-runs: the advice is that "
                            "pass's schedule's (--replay PASS names it)")
    meta, record = json.loads((arm_dir / "meta.json").read_text()), replay_dir / "replay.json"
    if record.is_file() and (of := json.loads(record.read_text())) and (of["load"], of["rep"]) != (load, rep):
        return not_measured(f"the replay in {replay_dir} is of another pass ({of['load']} rep {of['rep']}), not of "
                            f"{ours}'s {load} rep {rep}")
    traces = S.step_traces(arm_dir)
    if len(traces) != 1:
        return not_measured(f"{arm_dir.name} holds {len(traces)} step traces, and the pass's schedule is one engine core's")
    try:
        w, trace = R.window(arm_dir, load, rep), traces[0]
    except (SystemExit, OSError, KeyError, ValueError) as e:
        return not_measured(f"the pass can't be read back from {ours}'s step trace and ids: {type(e).__name__}: {e}")
    E = encodings()
    try:
        recs = E.span(S.trace_records(trace), lambda rid: f"-{w['tag']}-" in rid)
        d = E.draft_table(E.load_step_trace(recs, tokens={r["engine_id"]: [*r["prompt_ids"], *r["served"]]
                                                          for r in w["requests"]}))
    except (ValueError, RuntimeError, AssertionError) as e:
        return not_measured(f"the scheduler replay refuses the pass: {e}")
    sent = 0 if d["tick"] else d["arrivals"]["plain"]
    bits = sent + d["kept_count"]["charged"]
    return {"measured": True, **d, "bits": bits, "bits_a_request": bits / d["requests"], "arrivals_sent": sent,
            "seeds": seeds(w, meta),
            "window": {"arm": ours, "lifetime": meta.get("lifetime"), "load": load, "rep": rep, "run": meta.get("run_id"),
                       "replayed": record.is_file()}}


# --------------------------------------------------------------------------------------------------------- the report

def report(out: Path, stock: str | None, ours: str, again: str | None, load: str, rep: int = 1, replay_dir: Path | None = None,
           target: float = TARGET, replayed: str | None = None, expect_replay: bool = False,
           private: Path | None = None, beside: Sequence[str] = ()) -> dict:
    replay_dir = replay_dir or out / "replay-compiled"
    names = [a for a in (stock, again, ours) if a]
    names += [a for a in beside if a not in names]
    held = {a: arm_dirs(out, a) for a in names}
    states = {d.name: read(f"how far {d.name} got", lambda d=d: arm_state(d, load)) for a in names for d in held[a]}
    dirs = {a: ds for a in names if (ds := [d for d in held[a] if (d / "meta.json").is_file()])}
    per, by, sets, unread = {}, {}, {}, {}
    for a, ds in dirs.items():
        ms = [read(f"{d.name}'s metrics", lambda d=d: S.arm_metrics(d)) for d in ds]
        if bad := next((m for m in ms if m.get("unread")), None):
            unread[a] = bad["why"]
        else:
            per[a] = ms
    # an arm's lifetimes are one arm's, and every arm's reps are numbered in one call (serve_tax.lifetimes), so that
    # a cost pairs the same pass of the same lifetime
    one = read("the arms' lifetimes", lambda: {m["arm"]: m for m in S.lifetimes([m for ms in per.values() for m in ms])})
    for a in per:
        if one.get("unread") or a not in one:
            unread[a] = one["why"] if one.get("unread") else f"{a}'s record names another arm"
            continue
        by[a] = one[a]
        if a in (stock, ours):
            s = read(f"{a}'s settings", lambda a=a: settings(dirs[a][0], by[a], load))
            if s.get("unread"):
                unread[a] = s["why"]
            else:
                sets[a] = s
    meta = read(f"{ours}'s record", lambda: json.loads((dirs[ours][0] / "meta.json").read_text())) if ours in dirs else {}
    line = read("the cost", lambda: cost(out, by, stock, ours, again, load, target)) if stock else not_measured(
        f"no stock arm was served: the job serves {ours} alone")
    if len(sets) == 2 and (differ := settings_differ(sets[stock], sets[ours])):
        line["settings_differ"] = differ
        if line["measured"]:
            line["verdict"] = NOT_SHOWN
    state = read("how far the replay got", lambda: replay_state(replay_dir, expect_replay))
    engine = read("the replay's engine", lambda: {"engine": replay_engine(out, ours, replay_dir, private)})
    if ours not in dirs:
        toks = not_measured(f"{ours} was not served")
    elif state.get("unread"):
        toks = not_measured(state["why"])
    else:
        toks = read("the tokens", lambda: tokens(dirs[ours], ours, replayed or load, rep, replay_dir, state))
    return {"schema": SCHEMA, "model": meta.get("model"), "checkpoint": meta.get("checkpoint"),
            "run": meta.get("run_id"), "commit": meta.get("commit"), "limits": limits(meta.get("model_id")),
            "load": load, "arms": {"stock": stock, "again": again, "ours": ours, "beside": list(beside), "served": sorted(by)},
            "ran": {"arms": states, "unread": unread, "replay": state, "replay_engine": engine,
                    "passes": read("the passes", lambda: passes(out, dirs, per, load, private))},
            "settings": sets, "circuit_ahead_of_serving": NO_CIRCUIT,
            "value_settings": read("the adapter's record of them", lambda: value_settings(dirs, stock, ours, replay_dir)),
            "tokens": toks, "cost": line,
            "advice": read("the advice", lambda: advice(dirs[ours], ours, replayed or load, rep, replay_dir))
            if ours in dirs else not_measured(f"{ours} was not served")}


#: the schema of a run's result.json, whose ``artifacts`` and ``measurement_files`` the store keeps as the run's files
RESULT = "research/result/v0.1"


def outputs(out: Path, replay_dir: Path) -> list[str]:
    """What a reader of a job looks for first, as paths from the run's directory (``out``'s parent): the report's two
    files and the replay's record, where they are there. The rest of out/ (the replayed pass's ids and step trace
    among it) is in the run's record whole, and is not named."""
    files = [out / "prod_repro.txt", out / "prod_repro.json", replay_dir / "replay.json"]
    return [str(f.relative_to(out.parent)) for f in files if f.is_file()]


def declare(out: Path, r: dict, replay_dir: Path) -> Path | None:
    """Under research run, where ``out`` is the run's own out/: the run's result.json names `outputs` among its
    artifacts, so a reader finds them as the run's files without listing its record. A result the benchmark's cost
    wrote stays as it is with them added; a job that has none, one that served ours alone, gets one whose validation
    is the tokens line's. None for an out/ that is not the run's, which is another job's read again."""
    run_dir = SHARED.RESEARCH_RUN_DIR
    if not run_dir or out.resolve() != (Path(run_dir) / "out").resolve():
        return None
    f, t = Path(run_dir) / "result.json", r["tokens"]
    doc = json.loads(f.read_text()) if f.is_file() else {
        "schema": RESULT, "run_id": SHARED.RESEARCH_RUN_ID or Path(run_dir).name,
        "workload_fingerprint": {"benchmark": "prod_repro", "model": r["model"], "arm": r["arms"]["ours"], "load": r["load"]},
        "validation": {"status": "not_run" if not t["measured"] else "passed" if t["reproduces"] else "failed",
                       "checker": "verity bench prod_repro", "evidence": ["out/prod_repro.json"],
                       "detail": next(x for x in text(r).splitlines() if x.startswith("tokens:"))},
        "measurements": [], "measurement_files": [], "artifacts": []}
    doc["artifacts"] = sorted({*(doc.get("artifacts") or ()), *outputs(out, replay_dir)})
    tmp = f.with_name(f"{f.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, f)
    return f


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
    return f", 95% interval [{_pct(c['ci95'][0])}, {_pct(c['ci95'][1])}]" if c.get("ci95") else " (no interval: too few reps)"


def _first(r: dict) -> str:
    """The report's first line: the context limit the launch lines of ours and stock name, and for a model `LIMITS`
    says so of, that vLLM does not start with the limit left to it, at which settings, and the run that showed it."""
    sets, a = r["settings"], r["arms"]
    given = {n: sets[n]["context_limit"]["given"] for n in (a["ours"], a["stock"]) if n in sets}
    who, said = " and ".join(given), set(given.values())
    if not given:
        s = "context limit: not read (no record of ours or of stock)"
    elif len(said) > 1:
        s = "context limit: " + ", ".join(f"{n} {'--max-model-len ' + v if v else 'names none'}" for n, v in given.items())
    elif (v := said.pop()) is None:
        s = f"context limit: the launch line of {who} names none"
    elif v.isdigit():
        s = f"context limit {int(v):,} tokens, given outright to {who}"
    else:
        s = f"context limit left to vLLM by {who} (--max-model-len {v})"
    auto = r["limits"].get("auto")
    if auto and not auto["starts"]:
        s += (f"; with the limit left to it (--max-model-len auto), at {auto['at']}, vLLM does not start this model on "
              f"these cards at the pin: {auto['why']} (run {auto['run']})")
    return "prod_repro: " + s


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
    if stock is None:
        return s
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
    if stock is None:
        return s
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
        first = s.get("log_error")
        log = (f"its server's log: {name}/server/server.log" if first is None else
               f"its server's log names no exception: {name}/server/server.log" if not first else
               f"the first exception its server's log names: {first} ({name}/server/server.log)")
        return f"  {name}: FAILED: {s['error']} (its record's error; {log}); {up}{served}"
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


def _number(x: object) -> str:
    return f"{x:,}" if isinstance(x, int) else f"{x:,.3f}" if isinstance(x, float) else str(x)


def _advice(v: dict) -> list[str]:
    """The advice line: the charged bits a request first, then its two classes, the kept count with its bits a row
    and, where the loader gives it, what the counts would come to taken each alone; under it the loader's own figures
    of the kept count as it gives them, what the count costs where it is derived, and the seeds."""
    n, k, s = v["requests"], v["kept_count"], v["seeds"]
    tick = ("under the certified tick, which sends none" if v["tick"] else
            "0 under the certified tick, which this serve is not under")
    if not s["sampled"]:
        seeded = "no request was sampled, so no seed is read"
    elif s["unseeded"]:
        seeded = (f"{s['unseeded']:,} of the {s['sampled']:,} sampled requests carry NO seed in their arrival: what each "
                  "was sampled with is in no record, so its kept count is not derived and its tokens cannot be replayed")
    else:
        seeded = (f"{s['sampled']:,} sampled requests, each "
                  + ("seed the client's" if s["by_client"] else "with a seed in its arrival"))
    life = f"lifetime {v['window']['lifetime']} of " if v["window"].get("lifetime") else ""
    which = "the one the replay re-ran" if v["window"]["replayed"] else "the one a replay re-runs, and none ran"
    rows = f"{k['rows']:,} rows at {k['charged'] / k['rows']:.3f} bits" if k.get("rows") else "no row verified a draft"
    alone = f", its code decoded back; each count alone {k['alone'] / n:.1f}, not sent" if "alone" in k else ""
    return [f"advice: {v['bits_a_request']:.1f} bits a request charged ({n:,} requests): the arrivals "
            f"{v['arrivals_sent'] / n:.1f} ({tick}), the kept count {k['charged'] / n:.1f} ({rows}{alone})",
            "  the kept count over the pass, as the scheduler replay's loader gives it: "
            + ", ".join(f"{name} {_number(x)}" for name, x in k.items()),
            f"  {DERIVED}",
            f"  {seeded}",
            f"  the pass: {life}{v['window']['arm']}'s {v['window']['load']} rep {v['window']['rep']}, {which}; "
            f"{v['steps']:,} steps, each reproduced by the scheduler replay"]


def _engine(e: dict) -> list[str]:
    """What the replay's engine ran under, and where that is the canonical top-k, what its counters say of the
    replay's processes and of serving's."""
    if e.get("unread"):
        return [f"  the replay's engine: {e['why']}"]
    e = e["engine"]
    if e is None:
        return []
    under = e["under"]
    if not under:
        return [f"  the replay's engine: vLLM's own ({e['arm']} serves with no other vLLM plugin of ours than its record)"]

    def one(c: dict | None, none: str) -> str:
        if c is None:
            return none
        return (f"{c['processes']} process{'' if c['processes'] == 1 else 'es'} kept counters: at least {c['calls']:,} "
                f"calls of the indexer's top-k, {c['rows_long']:,} rows of more than its keys rewritten")

    names = ", ".join(x.split("=")[0].strip() for x in under["plugins"])
    env = ", ".join(f"{k}={v}" for k, v in under["env"].items())
    by = f" (set by {under['by']})" if under.get("by") else ""
    lines = [f"  the replay's engine: under what {e['arm']} served with beside its record{by}: the vLLM plugin"
             f"{'' if len(under['plugins']) == 1 else 's'} {names}; {env}"]
    t = e["topk"]
    if t and not t["kept"]:
        lines.append("  the canonical top-k's counters were not kept (the job named no PRIVATE directory): nothing here "
                     "shows how many rows it rewrote")
    elif t:
        lines += ["  the canonical top-k in the replay: " + one(t["replay"], "NO process kept counters: nothing shows it ran there"),
                  "  the canonical top-k while serving: " + one(t["served"], "NO process kept counters: nothing shows it ran there")]
    return lines


def _names(xs: Sequence[str]) -> str:
    return xs[0] if len(xs) == 1 else f"{', '.join(xs[:-1])} and {xs[-1]}"


def _passes(v: dict) -> list[str]:
    """Each pass's counters, a line a pass; then how each arm's requests were sampled, and for each directory whether
    its server was on the canonical top-k and what its processes counted there."""
    if not v["measured"]:
        return [f"each pass: {v['why']}"]
    if not v["passes"]:
        return []
    lines = ["each pass, by the engine's counters read before and after it (in no verdict):"]
    for x in v["passes"]:
        if x["steps"] is None:
            counted = "its record holds no counters of the engine's"
        else:
            counted = f"{x['steps']:,.0f} engine steps" + (
                f", {x['drafts']:,.0f} drafts, {x['accepted_tokens'] / x['draft_tokens']:.2%} of their tokens accepted"
                if x["drafts"] and x["draft_tokens"] else ", no drafts")
        lines.append(f"  {x['dir']} {x['pass']} rep {x['rep']}{', left out' if x['left_out'] else ''}: {counted}; "
                     f"{x['output_tokens']:,.0f} output tokens in {x['seconds']:.1f} s")
    how = {"greedy": [a for a, x in v["sampling"].items() if not x["sampled"]],
           "sampled with no seed sent": [a for a, x in v["sampling"].items() if x["sampled"] and not x["seeded"]],
           "sampled, each under a seed its client sent": [a for a, x in v["sampling"].items() if x["seeded"]]}
    lines.append("  the requests: " + "; ".join(f"{what} ({_names(arms)})" for what, arms in how.items() if arms))
    t = v["topk"]
    off = [d for d, on in t["on"].items() if not on]

    def one(d: str) -> str:
        if t["kept"] is None or d not in t["kept"]:
            return f"{d} on, its counters not kept (" + (
                "the job named no PRIVATE directory)" if t["kept"] is None else "its record names no place for them)")
        if (c := t["kept"][d]) is None:
            return f"{d} on, and NO process kept counters: nothing shows it ran there"
        return (f"{d} {c['processes']} process{'' if c['processes'] == 1 else 'es'} kept counters: at least {c['calls']:,} "
                f"calls of the indexer's top-k, {c['rows_long']:,} rows of more than its keys rewritten")

    lines.append("  the canonical top-k while serving: " + "; ".join(
        [one(d) for d, on in t["on"].items() if on]
        + [f"not on for {_names(off)} (no {TOPK} in the environment {'its record names' if len(off) == 1 else 'their records name'})"]
        * bool(off)))
    return lines


def _values(v: dict, stock: str | None, ours: str) -> list[str]:
    """The adapter's record of the value settings in force (`value_settings`): for each arm and the replay how many
    variables are set and the record's digest, and by name the ones ours's differs from stock's in and the replay's
    from ours's; not recorded for a side whose directory holds none."""
    if v.get("unread"):
        return [f"  value settings in force: {v['why']}"]
    first = {a: next(iter(recs.values()), None) for a, recs in v["arms"].items()}
    if not any(first.values()) and v["replay"] is None:
        return ["  value settings in force: not recorded (no arm's record and no replay's holds the adapter's record of them)"]

    def one(name: str, rec: dict | None, differ: list[str] | None, other: str) -> str:
        if rec is None:
            return f"{name} not recorded"
        against = ("" if differ is None else f", the same as {other}" if not differ else
                   f", differing from {other} in {', '.join(differ)}")
        return f"{name} {len(rec['values']):,} set, SHA-512 {rec['digest'][:DIGEST_SHOWN]}{against}"

    d = v["differ"]
    sides = [*([one(stock, first.get(stock), None, "")] if stock else []),
             one(ours, first.get(ours), d["ours_from_stock"], f"{stock}'s"),
             one("the replay's engine", v["replay"], d["replay_from_ours"], f"{ours}'s")]
    return ["  value settings in force, the adapter's record of each server's environment (serve_tax.value_settings): "
            + "; ".join(sides),
            *(f"  NOT ONE RECORD: {a}'s lifetimes hold different value settings; the line above reads its first lifetime's"
              for a in v["lifetimes_differ"])]


def _verdict(c: dict) -> str:
    """A cost's verdict in words: yes, no, or not shown and why."""
    if c["verdict"] == UNDER:
        return "yes"
    if c["verdict"] == OVER:
        return "NO, the lower end of the interval over lifetime pairs is over it"
    life = c["lifetimes"]
    why = (UNLIKE if c.get("settings_differ") else UNEQUAL if c.get("same_work") is False else
           (ONE_LIFETIME if life["served"] < 2 else ONE_PAIR) if life["pairs"] < 2 else
           NOT_UNDER if life["ci95"] else NOT_SESSIONS)
    return f"{NOT_SHOWN}, {why}"


def _lifetimes(c: dict) -> list[str]:
    """A sessions load's cost by lifetime: how the cost is taken, with the benchmark's own words on each request's
    length (`serve_tax.TO_LENGTH`) and its line on the requests that went on past an end-of-text token (`NOT_COUNTED`
    where it has none), each lifetime's own and the spread."""
    life, points = c["lifetimes"], c["lifetimes"]["spread_points"]

    def spread(x: float | None) -> str:
        return "not measured" if x is None else f"{x:.2f} points"

    def side(name: str, x: dict) -> str:
        steps = "steps not read" if x.get("steps") is None else f"{x['steps']:,.0f} engine steps a pass"
        drafts = "" if x.get("draft_accepted_share") is None else (
            f", {x['draft_accepted_share']:.1f}% of its draft tokens accepted"
            + ("" if x.get("tokens_per_draft") is None else f", {x['tokens_per_draft']:.2f} tokens a draft"))
        return f"{name} {steps}{drafts}"

    left, other, sides = life.get("left_out"), life.get("readings") or {}, life.get("sides") or {}
    return ["  the cost is the mean of the lifetimes' costs, a lifetime's being ours's seconds for the passes both arms "
            "counted in it over stock's, less one, a pass's seconds from its users' first send to the last one's last "
            "answer (serve_tax.session_cost)"
            + (f"; the first {left['passes']} of each lifetime's passes, its warm-up, served and left out" if left else "")
            + f"; {S.TO_LENGTH}",
            *((life.get("past_end") or {}).get("said") or [NOT_COUNTED]),
            *(["  by lifetime: " + ", ".join(f"{_pct(x['value'])} ({x['reps']} pass{'es' * (x['reps'] != 1)})"
                                             for x in life["each"])] if life.get("each") else []),
            f"  the cost's spread (standard deviations): {spread(points['between'])} between lifetimes, "
            f"{spread(points['within'])} between a lifetime's passes",
            *(["  other readings of the same passes, in no verdict: " + ", ".join(
                f"{_pct(v)} by {S.READINGS.get(k, k)}" for k, v in other.items() if v is not None)]
              if any(v is not None for v in other.values()) else []),
            *(["  beside the cost, which is adjusted for neither: " + "; ".join(side(a, x) for a, x in sides.items())]
              if any(v is not None for x in sides.values() for v in x.values()) else [])]


def _noise(n: dict | None, stock: str) -> str:
    if not n or n["value"] is None:
        return "not measured (stock was not served a second time)"
    if n["over"] == "passes":
        return f"{n['arm']} over {stock} {_pct(n['value'])}{_interval(n)}, passes taken as independent"
    who = f"{stock}'s own lifetimes, each second one over the one before it," if n["own"] else f"{n['arm']} over {stock}"
    ends = f", 95% interval [{_pct(n['ci95'][0])}, {_pct(n['ci95'][1])}]" if n["ci95"] else " (no interval: one pair)"
    return f"{who} {_pct(n['value'])}{ends}, over {n['pairs']} pair{'s' * (n['pairs'] != 1)} of server lifetimes"


def text(r: dict) -> str:
    a, t, c, ran = r["arms"], r["tokens"], r["cost"], r["ran"]
    stock, ours = a["stock"], a["ours"]
    lines = [_first(r),
             f"prod_repro: load {r['load']}; stock {stock or 'not served'}, stock again {a['again'] or 'not served'}, ours {ours}; "
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
    lines += _engine(ran["replay_engine"])
    lines += _passes(ran["passes"])
    lines += [f"settings of {ours}" + (f", with {stock}'s beside:" if stock else ":"),
              _setting("prefix caching", "prefix_caching", r),
              _setting("speculative decoding", "speculative_decoding", r),
              _limit(r),
              _setting("KV cache", "kv_cache", r),
              _setting("torch.compile and CUDA graphs", "compiled", r),
              f"  circuit compiled ahead of serving: {r['circuit_ahead_of_serving']}",
              *_values(r["value_settings"], stock, ours)]
    if not t["measured"]:
        lines.append(f"tokens: not measured: {t['why']}")
    else:
        h = t["cache_hits"]
        lines += [
            f"tokens: {'reproduced' if t['reproduces'] else 'FAILED'}: {t['requests']:,} requests replayed, "
            f"{t['tokens_served']:,} served tokens compared with {t['tokens_replayed']:,} replayed, "
            f"{t['tokens_differing']:,} differ in {t['requests_differing']:,} requests",
            *(f"  {x}" for x in (*where(t), *failed(r))),
            *([f"  {len(e):,} step{'s' * (len(e) > 1)} after a request's end, no token, not replayed: "
               f"step{'s' * (len(e) > 1)} {', '.join(map(str, e[:ROWS_MAX]))}{' and more' * (len(e) > ROWS_MAX)} of the "
               f"window's {t['steps_served']:,} (serving had planned each before it learned its requests had ended, and "
               "dropped what it computed: counted here, and no difference)"] if (e := t["steps_after_end"]) else []),
            f"  {t['steps_served']:,} engine steps, {t['steps_differing']:,} with other rows than the served schedule; "
            + (f"{h['requests']:,} requests began from a prefix-cache hit, the longest {h['longest']:,} tokens"
               if h["requests"] else "no request of the window began from a prefix-cache hit"),
            f"  served {t['served_mode']}, replayed {t['replay_mode']}; the window's requests: {_sampling(t)}",
            *([f"  {k['requests']:,} of the window's requests passed {k['keys']:,} ids of prompt and output, past which "
               "attention reads the keys the DSA indexer's top-k selects"
               + (f"; {k['differing']:,} of the {t['requests_differing']:,} differing requests are among them"
                  if t["requests_differing"] else "")] if (k := t.get("past_topk")) else []),
            f"  the window: {'of lifetime ' + str(lt) + ' of ' + ours + '; ' if (lt := t['window'].get('lifetime')) else ''}"
            f"{t['window']['chosen']}"]
    if not c["measured"]:
        lines.append(f"cost: not measured: {c['why']}")
    else:
        how, noise, life = c["how"], c["noise_floor"], c["lifetimes"]
        bench = how["client"] == S.BENCH
        at = (f"at {how['rung']} users" if how.get("sessions") else f"at concurrency {how['rung']}") if bench else ""
        at = (f"{at}, where stock's rungs {'show' if how['saturated'] else 'do NOT show'} it saturated"
              if bench else f"at {how['rate']:g} requests a second, which stock did not keep up with")
        reps = f"{how['reps']} paired rep{'' if how['reps'] == 1 else 's'}" if bench else f"{how['reps'][0]} and {how['reps'][1]} reps"
        over = (f", 95% interval [{_pct(life['ci95'][0])}, {_pct(life['ci95'][1])}] over {life['pairs']} server lifetimes "
                "an arm" if life["ci95"] else "")
        own = how.get("sessions")
        lines += [
            f"cost: {ours} over {stock} {_pct(c['value'])} of stock's serving time{over}; under {c['target']:g}%: {_verdict(c)}",
            *(_lifetimes(c) if own else []),
            f"  passes taken as independent{' ' + _pct(c['passes']['value']) if own else ''}{_interval(c['passes'])}: over "
            f"{reps}, {at}: {how['by']}{'; in no verdict' if own else ''}",
            "  noise floor, beside and not added: " + _noise(noise, stock)]
        if noise and noise.get("leaves_out_zero"):
            lines.append(f"  STOCK DIFFERS FROM STOCK: the noise floor's interval leaves out 0, so two servers of {stock} "
                         + ("differ by more than their passes do" if noise["over"] == "passes" else
                            "did not serve at the same rate"))
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
    q = sub.add_parser("served", help="exit 0 where an arm of a job's out/ served the load without an error")
    q.add_argument("out", type=Path)
    q.add_argument("arm")
    q.add_argument("--load", required=True)
    p = sub.add_parser("report", help="the settings and the three lines of a finished job's out/")
    p.add_argument("out", type=Path)
    p.add_argument("--stock", help="the arms default to the three OUT is read as (default_arms); '' for a job that "
                                   "served ours alone")
    p.add_argument("--again", help="stock in a second server lifetime, the noise floor ('' for none)")
    p.add_argument("--ours")
    p.add_argument("--beside", nargs="*", default=(), metavar="ARM",
                   help="the arms the job served beside those three (the runner's --arms): in what ran and among the "
                        "passes, and in no line of the three")
    p.add_argument("--load", required=True)
    p.add_argument("--replayed", metavar="PASS",
                   help="the pass of ours the replay re-ran, as the benchmark names it (default: the load; by vLLM's "
                        "client a pass is LOAD@RUNG)")
    p.add_argument("--rep", type=int, default=1, help="the rep of the pass the replay re-ran")
    p.add_argument("--replay", type=Path, help="the replay's directory (default: OUT/replay-compiled)")
    p.add_argument("--expect-replay", action="store_true",
                   help="the job asked for a replay: fail where the tokens line is not measured")
    p.add_argument("--private", type=Path, metavar="DIR",
                   help="the job's private directory on the node (the benchmark's PRIVATE), where the canonical top-k's "
                        "processes kept their counters: read there and kept in OUT")
    p.add_argument("--target", type=float, default=TARGET, metavar="PERCENT")
    p.add_argument("--json", action="store_true", help="print the JSON object alone")
    a = ap.parse_args(argv)
    if a.cmd == "arms":
        print(" ".join(default_arms(a.model, a.out)))
        return 0
    if a.cmd == "served":
        ss = [arm_state(d, a.load) for d in arm_dirs(a.out, a.arm)]
        return 0 if all(s["started"] and not s.get("error") and s.get("passes") for s in ss) else 1
    stock, again, ours = default_arms(out=a.out)
    r = report(a.out, (stock if a.stock is None else a.stock) or None, a.ours or ours,
               (again if a.again is None else a.again) or None, a.load, a.rep,
               a.replay, a.target, a.replayed, a.expect_replay, a.private, a.beside)
    r["failed"] = failed(r)
    (a.out / "prod_repro.json").write_text(json.dumps(r, indent=1))
    (a.out / "prod_repro.txt").write_text(text(r))
    declare(a.out, r, a.replay or a.out / "replay-compiled")
    print(json.dumps(r) if a.json else text(r), end="\n" if a.json else "")
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
