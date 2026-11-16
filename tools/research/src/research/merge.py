"""What the verified queue (`research queue vqueue`) and `check.py` read of a recorded check, and the steps after a landing.

A recorded check: an attempt of the tool (default `check`) passes for a commit only as its own record says (`refused`, `passing`):
`state` done, rc 0, that clean commit; a run of the command the target's declaration of the tool names (`Tool.command`, in a
check slot or not, followed by nothing but check.py's own flags: `unlike`); and its own result, of a kind that declaration
produces (`Tool.produces`: `check/v1` for check) and not in `REFUSED_KINDS` (the Lean fast path's `lean-fast/v1`, which skips
the kernel replay), with the verdict passed and no step listed twice.  A sharded check (`check.py --shard K/N`, one Attempt
per shard, its result a `check-shard/v1`) passes only through the Attempt of `check.py --combine RUN...` on the same commit,
and only when its shards, read here (`shard_set`), are a passing check of that clean commit (one plan, K = 1..N once each,
the commit's own `tools/check/`, every part once, every job its `ci.toml` puts on the merge gate, no step failed) and the
combine's steps theirs; a shard alone passes nothing.  A job off the gate (policy.toml's `off_gate`) is required of no check,
whole or sharded: it runs in the after-landing run, whose red names its culprit.  The steps a change requires are read from
the target's declaration (`required_steps`).  A commit of the lock records a check regenerated is vouched for by that check
when its records are byte for byte the ones it regenerated (`regenerated_parent`, reading them from the store or the run's
node: `regenerated_records`, `live_records`).

After a landing (`caught_up`, from the queue's after-landing job), what landed on origin's main since AFTER_LAND_REF there
gets, once: main pushed to vy-nebius-cpu-4's bare repository, whose main its builds check out (`vy-lean-sandbox checkout`), a
plain push over the ssh `research run --on` uses, bounded by MIRROR_TIMEOUT_S, which the next landing repeats when it fails
($RESEARCH_MERGE_MIRROR names another machine, and empty skips it: `push_mirror`); what the tool's declaration runs after a
landing (`Tool.after_land`), started detached on the landed commit (`after_land`); the Lean cache publish when a Lean package
changed (`lean_publish`); one line per landing in the merge thread, what it merged and its checked run (`report_landings`); a
`landed` label on each run a landing names (`label_landed`); and, for a change to what a guarantee says (a
`lean-audit.json`'s guarantees or the definitions they read), the spec-change DM (`spec_alert`), whose own marker
SPEC_ALERTED_REF moves only when it was sent, so the next landing's steps send a failed one again.  Then the sweep closes
every open pull request whose head is on main, with a comment naming the commit it landed in, retargets onto main every one
whose base branch is on it, and deletes the head branches of those that landed or closed (`sweep`).  Nothing raises: a
landing that happened stands.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .gates import git_env, worktrees_shared


class Refused(Exception):
    pass


def _git(repo: Path, *args: str, check: bool = True, lazy: bool = True) -> subprocess.CompletedProcess:
    """``git ARGS`` in ``repo``; with ``lazy`` False, a partial clone fetches no object it lacks (`GIT_NO_LAZY_FETCH`).  A
    fetch holds the clone's worktree lock shared (`worktrees_shared`), as the queue's do: the queue's clone, where its
    steps after a landing fetch, has its worktrees added beside them."""
    env = git_env() if lazy else git_env(GIT_NO_LAZY_FETCH="1")
    with worktrees_shared(repo, env) if args[:1] == ("fetch",) else contextlib.nullcontext():
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, errors="surrogateescape", env=env)
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r


#: result kinds that pass for no commit, whatever `--tool` recorded them and whatever its declaration says it produces:
#: `lean-fast/v1` is the Lean fast path's (`tools/verity/lean/lean_changed.py --records`), an audit without the kernel replay,
#: which every merge check keeps
REFUSED_KINDS = frozenset({"lean-fast/v1"})
#: what `check.py --record --on` puts before the declared command (its `slot_wrap`): one of the machine's check slots, then
#: slot.py's options (`SLOT_FLAGS`, `SLOT_OPTS`), each at most once and in any order, then `--`
SLOT_WRAP = ("python3", "tools/check/slot.py")
#: slot.py's options a recorded check may carry: which slot it waits for, and how long, never what it runs (not `--budget-gb`,
#: which changes how much CMD starts at once)
SLOT_FLAGS, SLOT_OPTS = ("--priority",), ("--rank", "--by", "--twin", "--wait-max", "--run-max")
#: check.py's own flags, as its `main` reads them in a recorded run: these anywhere, `--verdicts-in PACK` once, and then nothing
#: or `--cores N`; `--no-cache --circuit-cache` reuses less than a cached check does
CHECK_FLAGS = ("--no-cache", "--circuit-cache", "--keep-going")


def invocation(argv, command) -> tuple[str, object]:
    """What an attempt's ``argv`` ran of the target's declared ``command``: ("check", None), ``command`` itself, in a check slot
    (`SLOT_WRAP`) or not, followed by nothing but check.py's own flags (`CHECK_FLAGS`); ("shard", (K, N)), the same with
    `--shard K/N` once (`tools/check/check.py`'s sharding); ("combine", [run ids]), ``command`` with `--combine RUN...` and
    nothing else; or ("refused", why) for anything else."""
    if not command:
        return "refused", "the target's declaration of the tool names no command to hold it to"
    if not (isinstance(argv, list) and all(isinstance(x, str) for x in argv)):
        return "refused", "its record has no argv"
    run = argv
    if tuple(argv[:2]) == SLOT_WRAP:
        opts, seen = argv[2:], set()
        while opts[:1] != ["--"] and opts[:1] and opts[0] not in seen and (
                opts[0] in SLOT_FLAGS or (opts[0] in SLOT_OPTS and len(opts) > 1)):
            seen.add(opts[0])
            opts = opts[1:] if opts[0] in SLOT_FLAGS else opts[2:]
        run = opts[1:] if opts[:1] == ["--"] else argv
    if run[:len(command)] != list(command):
        return "refused", f"its command `{shlex.join(argv)}` isn't the target's declared `{shlex.join(command)}`"
    tail = run[len(command):]
    if tail[:1] == ["--combine"]:
        ids = tail[1:]
        return ("combine", ids) if ids and not any(x.startswith("-") for x in ids) else (
            "refused", f"its command gives the declared one `{shlex.join(tail)}`, which check.py doesn't take")
    rest = [x for x in tail if x not in CHECK_FLAGS]
    if "--verdicts-in" in rest[:-1]:
        i = rest.index("--verdicts-in")
        rest = rest[:i] + rest[i + 2:]
    mode: tuple[str, object] = ("check", None)
    if "--shard" in rest[:-1] and (m := re.fullmatch(r"(?:auto:)?([1-9]\d*)/([1-9]\d*)", rest[rest.index("--shard") + 1])):
        i = rest.index("--shard")
        mode, rest = ("shard", (int(m.group(1)), int(m.group(2)))), rest[:i] + rest[i + 2:]
    if rest and not (rest[0] == "--cores" and len(rest) == 2 and rest[1].isdigit()):
        return "refused", f"its command gives the declared one `{shlex.join(rest)}`, which check.py doesn't take"
    return mode


def unlike(argv, command) -> str | None:
    """Why an attempt's ``argv`` isn't a whole run of the target's declared ``command``, or None (`invocation`'s "check"): a
    shard passes no gate alone, and a combine only through `refused`'s reading of its shards."""
    mode, arg = invocation(argv, command)
    if mode == "shard":
        return (f"it is shard {arg[0]}/{arg[1]} of a sharded check, which passes no gate alone: `check.py --record --combine` "
                "its set")
    if mode == "combine":
        return "it combines shards, which only `refused`'s own reading of them admits"
    return arg if mode == "refused" else None


#: a sharded check's shard result (`tools/check/check.py --shard`): never a kind the tool produces, read only by `shard_set`
SHARD_KIND = "check-shard/v1"


def _ci_toml(repo: Path, commit: str) -> dict | None:
    """``commit``'s own `tools/check/ci.toml`, parsed, or None when it can't be read."""
    import tomllib

    from .store import tools_registry

    path = tools_registry.resolve("tools/check/ci.toml",
                                  lambda p: _git(repo, "cat-file", "-e", f"{commit}:{p}", check=False).returncode == 0)
    r = _git(repo, "show", f"{commit}:{path or 'tools/check/ci.toml'}", check=False)
    try:
        return tomllib.loads(r.stdout) if r.returncode == 0 else None
    except (tomllib.TOMLDecodeError, TypeError):
        return None


def declared(repo: Path, commit: str) -> list[dict] | None:
    """The `[[job]]` tables of ``commit``'s own `tools/check/ci.toml`, in its order (each a name, an owner, a group, and
    `after` when it has one; `gate` where it sets one), or None when it can't be read."""
    jobs = doc.get("job", []) if (doc := _ci_toml(repo, commit)) is not None else None
    if not isinstance(jobs, list) or not all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in jobs):
        return None
    return jobs


def shard_plans(repo: Path, commit: str) -> dict[int, list[list[str]]] | None:
    """The `[[plan]]` tables of ``commit``'s own `tools/check/ci.toml` by their number of shards, as check.py's
    `--shard K/N` takes them: {N: each shard's parts}, or None when it can't be read."""
    plans = doc.get("plan", []) if (doc := _ci_toml(repo, commit)) is not None else None
    if not isinstance(plans, list):
        return None
    out = {}
    for p in plans:
        shards = p.get("shards") if isinstance(p, dict) else None
        if not (isinstance(shards, list) and all(isinstance(s, list) and all(isinstance(x, str) for x in s) for s in shards)):
            return None
        out.setdefault(len(shards), shards)
    return out


def declared_jobs(repo: Path, commit: str) -> list[str] | str:
    """The names of the jobs ``commit``'s own `tools/check/ci.toml` declares that are on the merge gate (every job but
    those ``commit``'s policy, `policy.toml`, names in `off_gate`, or its ci.toml marks `gate = false`), or why they
    can't be read.  A job off the gate runs after landing, in the after-landing run, whose red names its culprit; no check
    needs it to pass a merge, and a recorded check doesn't run it."""
    from . import policy

    pol = policy.at(repo, commit)
    if isinstance(pol, str):
        return pol
    jobs = [t["name"] for t in declared(repo, commit) or ()
            if t.get("gate", True) is not False and t["name"] not in pol.off_gate]
    if not jobs:
        return f"{commit[:12]}'s tools/check/ci.toml can't be read for the jobs a check runs"
    return jobs


def _read(store, run_id: str, tool: str | None, commit: str | None):
    """(record, result id, result manifest or None) of the attempt, or why it can't pass whatever it ran: its record can't be
    read, or doesn't say it is a passing ``tool`` attempt of the clean ``commit``, or its result can't be read."""
    try:
        doc = store.get_attempt(run_id)
    except Exception as e:  # noqa: BLE001  (a guard never fails open)
        return f"its attempt can't be read ({type(e).__name__}: {e})"
    if not isinstance(doc, dict):
        return "its attempt can't be read"
    st, src = doc.get("status") or {}, doc.get("source") or {}
    said = {"tool": (doc.get("tool") or {}).get("name"), "commit": src.get("commit"), "dirty": src.get("dirty"),
            "state": st.get("state"), "rc": st.get("rc")}
    want = {"tool": tool, "commit": commit, "dirty": False, "state": "done", "rc": 0}
    if off := [f"{k} {said[k]}" for k, v in want.items() if v is not None and said[k] != v]:
        return f"its record says {', '.join(off)}, not the index's passing check"
    art = (doc.get("outputs") or {}).get("result")
    try:
        return doc, art, store.get_manifest(art) if art else None
    except Exception as e:  # noqa: BLE001  (a guard never fails open)
        return f"its result can't be read ({type(e).__name__}: {e})"


def shard_set(store, repo: Path, commit: str, ids: list[str], tool: str = "check",
              command: tuple[str, ...] = ()) -> tuple[list[dict] | None, str | None]:
    """(the steps of the one check the shard attempts ``ids`` make, None) when together they are a passing check of the clean
    ``commit``, else (None, why).  Each must be a passing ``tool`` attempt of ``commit`` whose command is the declared
    ``command`` with `--shard K/N` or `--shard auto:K/N` (`invocation`), whose own result is a passed `SHARD_KIND` naming that
    shard, ``commit`` and its config (``commit``'s tree of `tools/check/`), with no step but passed or skipped and its
    steps' parts its shard's parts; together, one N with K = 1..N once each, one list of every part and one auto plan's digest
    (or none), whose parts they make exactly once and which covers every job ``commit``'s `ci.toml` puts on the gate
    (`declared_jobs`: not circuit-check or ratchet-gate, which run after landing), and the same steps outside their parts (the
    preflight), passed in each, but for the builds a shard runs beside its parts (an auto plan's, a hand plan's lean-build
    beside lean-suites' parts), whose verdict is their own shard's.  The steps: those once, then each part as its shard's step
    ran it."""
    if not ids or len(set(ids)) != len(ids):
        return None, "a set of shards names each shard's attempt once" if ids else "no shard is named"
    config = _git(repo, "rev-parse", "--verify", "-q", f"{commit}:tools/check", check=False).stdout.strip()
    if not config:
        return None, f"{commit[:12]} has no tools/check/ to hold the shards' config to"
    jobs = declared_jobs(repo, commit)
    if isinstance(jobs, str):
        return None, jobs
    shards: dict[int, tuple[str, dict, list[dict]]] = {}
    plans = set()
    for rid in ids:
        got = _read(store, rid, tool, commit)
        if isinstance(got, str):
            return None, f"{rid}: {got}"
        doc, art, result = got
        mode, arg = invocation(doc.get("argv"), command)
        if mode != "shard":
            return None, f"{rid}: {arg if mode == 'refused' else 'it is not a shard (`check.py --shard K/N`)'}"
        kind = result.kind if result else None
        if kind != SHARD_KIND:
            return None, f"{rid}: its result is {f'a {kind}' if kind else 'missing'}, not a {SHARD_KIND}"
        meta = result.meta if isinstance(result.meta, dict) else {}
        sh = meta.get("shard") if isinstance(meta.get("shard"), dict) else {}
        steps = meta.get("steps", [])
        if meta.get("run_id") != rid:
            return None, f"{rid}: its result names run {meta.get('run_id')}"
        if (meta.get("validation") or {}).get("status") != "passed":
            return None, f"{rid}: its result's verdict is {(meta.get('validation') or {}).get('status')}, not passed"
        if why := twice(steps):
            return None, f"{rid}: {why}"
        if (sh.get("k"), sh.get("n")) != arg:
            return None, f"{rid}: its result is shard {sh.get('k')}/{sh.get('n')}, its command {arg[0]}/{arg[1]}"
        if sh.get("commit") != commit or sh.get("config") != config:
            return None, (f"{rid}: its result checked commit {sh.get('commit')} with config {sh.get('config')}, not {commit[:12]}'s "
                          f"tools/check/ ({config[:12]})")
        if not all(isinstance(sh.get(f), list) and all(isinstance(p, str) for p in sh[f]) for f in ("parts", "all")):
            return None, f"{rid}: its result doesn't list its parts and every part"
        if bad := [s["step"] for s in steps if s.get("status") not in ("passed", "skipped")]:
            return None, f"{rid}: {', '.join(bad)} didn't pass"
        ran = [p for s in steps for p in (s.get("parts") or [])]
        if not all(isinstance(p, str) for p in ran) or sorted(ran) != sorted(sh["parts"]):
            return None, f"{rid}: its steps ran {', '.join(map(str, ran)) or 'nothing'}, not its shard's parts"
        if arg[0] in shards:
            return None, f"shard {arg[0]}/{arg[1]} is named twice ({shards[arg[0]][0]}, {rid})"
        shards[arg[0]] = (rid, sh, steps)
        plans.add((arg[1], json.dumps(sh["all"]), (sh.get("plan") or {}).get("digest") if isinstance(sh.get("plan"), dict) else None))
    if len(plans) != 1:
        return None, ("the shards are of different plans (their N, every part they say an unsharded check covers, or their auto "
                      "plan's digest)")
    (n, _, _), = plans
    if missing := [k for k in range(1, n + 1) if k not in shards]:
        return None, f"shard {', '.join(f'{k}/{n}' for k in missing)} is not among them"
    every = shards[1][1]["all"]
    covered = [p for k in sorted(shards) for p in shards[k][1]["parts"]]
    if len(set(every)) != len(every):
        return None, "their list of every part names one twice"
    if sorted(covered) != sorted(every):
        again = sorted({p for p in covered if covered.count(p) > 1})
        return None, (f"their parts aren't every part once (twice: {', '.join(again) or 'none'}; in no shard: "
                      f"{', '.join(sorted(set(every) - set(covered))) or 'none'}; not a part: "
                      f"{', '.join(sorted(set(covered) - set(every))) or 'none'})")
    if uncovered := [j for j in jobs if not any(p == j or p.startswith(j + ":") for p in every)]:
        return None, f"no shard runs {', '.join(uncovered)}, which {commit[:12]}'s ci.toml puts on the gate"
    names = {p.partition(":")[0] for p in every}
    # a shard runs the builds its parts need beside them when another shard holds them (`tools/check/shards.py`,
    # `declaration.py`): a job's step outside its parts, passed or skipped like every step of a shard, whose part another holds
    common = [[s["step"] for s in shards[k][2] if not s.get("parts") and s["step"] not in names] for k in sorted(shards)]
    if any(sorted(c) != sorted(common[0]) for c in common):
        return None, "the shards ran different steps outside their parts"
    out = [{"step": s, "status": "passed" if all(x.get("status") == "passed" for k in shards for x in shards[k][2] if x["step"] == s)
            else "skipped", "shards": [shards[k][0] for k in sorted(shards)]} for s in common[0]]
    part = {p: {"step": p, "status": s.get("status"), "shard": k, "run_id": shards[k][0], "seconds": s.get("seconds")}
            for k in sorted(shards) for s in shards[k][2] for p in (s.get("parts") or [])}
    return out + [part[p] for p in every], None


def declaration(repo: Path, tool: str, at: str | None = None):
    """The target's own declaration of ``tool``, read from the checkout (not from the commit being merged), or with ``at`` from
    that commit of ``repo``: a checkout left behind `main` declares an earlier layout's paths.  None when it has none."""
    from .store import tools_registry

    try:
        if at is None:
            return tools_registry.from_tree(tool, repo)
        module = tools_registry.REGISTRY.get(tool, tool).rpartition(":")[0]
        with tempfile.TemporaryDirectory(prefix="declaration-") as d:
            for m in (module, *tools_registry.former_modules(module)):
                rel = Path(*m.split("."))
                got = subprocess.run(["git", "archive", "--format=tar", at, "--", str(rel.parent if len(rel.parts) > 1 else rel)],
                                     cwd=repo, capture_output=True, env=git_env())
                if got.returncode == 0:
                    tarfile.open(fileobj=io.BytesIO(got.stdout)).extractall(d, filter="data")
            return tools_registry.from_tree(tool, Path(d))
    except Exception:  # noqa: BLE001  (a declaration that can't be read is none)
        return None


def produced(repo: Path, tool: str) -> frozenset[str]:
    """The result kinds the target's declaration of ``tool`` produces (`Tool.produces`); none when it has no declaration, so
    that no attempt passes."""
    return frozenset(getattr(declaration(repo, tool), "produces", None) or ())


def declared_command(repo: Path, tool: str) -> tuple[str, ...]:
    """The command the target's declaration of ``tool`` runs (`Tool.command`); none when it has no declaration, so that no
    attempt passes."""
    return tuple(getattr(declaration(repo, tool), "command", None) or ())


def refused(store, run_id: str, kinds: frozenset[str], tool: str | None = None, commit: str | None = None,
            command: tuple[str, ...] | None = None, repo: Path | None = None) -> str | None:
    """Why the attempt can't pass for ``commit`` whatever the index says of it, or None: its record can't be read, or doesn't say
    what the index row did (a passing ``tool`` attempt of the clean ``commit``), or ran something other than ``command`` (the
    tool's `declared_command`, `unlike`; with none, nothing passes), or its result can't be read, is in `REFUSED_KINDS`, is not
    of a kind in ``kinds`` (the tool's `produced`; an attempt that published no result has none), is another run's, has a
    verdict other than passed, or lists a step twice (it would read two ways).  A `--combine` of shards passes only
    when, read in ``repo`` (the target's checkout), its shards are a passing check of ``commit`` (`shard_set`) and its steps are
    the ones they make."""
    got = _read(store, run_id, tool, commit)
    if isinstance(got, str):
        return got
    doc, art, result = got
    mode, ids = invocation(doc.get("argv"), command)
    if mode != "combine" and (why := unlike(doc.get("argv"), command)):
        return why
    kind = result.kind if result else None
    if kind in REFUSED_KINDS:
        return f"its result is a {kind}, which no merge accepts"
    if kind not in kinds:
        return ((f"its result is a {kind}" if kind else f"its result {art} has no kind" if art else "it published no result") +
                f", and the target's declaration of the tool produces {', '.join(sorted(kinds)) or 'nothing'}")
    meta = result.meta if isinstance(result.meta, dict) else {}
    if meta.get("run_id") != run_id:
        return f"its result names run {meta.get('run_id')}, not this one"
    validation = meta.get("validation")
    verdict = validation.get("status") if isinstance(validation, dict) else None
    if verdict != "passed":
        return f"its result's verdict is {verdict}, not passed"
    if why := twice(meta.get("steps", [])):
        return why
    if mode == "combine":
        if repo is None or commit is None:
            return "it combines shards, which this gate reads only in the target's checkout, for a commit"
        steps, why = shard_set(store, repo, commit, ids, tool or "check", command or ())
        if why:
            return f"its shards aren't a passing check: {why}"
        seen = sorted((s["step"], s.get("status")) for s in meta.get("steps", []))
        if seen != sorted((s["step"], s["status"]) for s in steps):
            return "its steps aren't the ones its shards make"
    return None


def twice(steps) -> str | None:
    """Why a result's ``steps`` can't be read one way, or None: each is an object naming its step, and no step is listed twice
    (`step_status` would read the first, a reader of the steps' list the last)."""
    if not (isinstance(steps, list) and all(isinstance(s, dict) and isinstance(s.get("step"), str) for s in steps)):
        return "its result has steps this gate can't read"
    names = [s["step"] for s in steps]
    if again := next((n for i, n in enumerate(names) if n in names[:i]), None):
        return f"its result lists step {again} more than once"
    return None


def passing(store, commit: str, tool: str = "check", *, kinds: frozenset[str], command: tuple[str, ...], tree: str | None = None,
            maps=None, repo: Path | None = None) -> list[str]:
    """Ids of the attempts of ``tool`` that passed on the clean tree of ``commit``, newest first, less those `refused` names
    (``kinds`` and ``command``: the result kinds that may pass and the command that must have run, the target's `produced` and
    `declared_command`).  After a history rewrite (`commitmap`), an attempt of an old commit counts for its image when the map
    says the rewrite kept the tree and, where the attempt recorded its tree, that tree is ``tree``."""
    rows = store.index.sql("SELECT id FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 AND state = 'done' "
                           "AND rc = 0 ORDER BY start_utc DESC", (tool, commit))
    found = [(r[0], commit) for r in rows]
    for old in (maps.equivalent(commit)[1:] if maps else []):
        rows = store.index.sql("SELECT id, manifest_json FROM attempts WHERE tool = ? AND source_commit = ? AND source_dirty = 0 "
                               "AND state = 'done' AND rc = 0 ORDER BY start_utc DESC", (tool, old))
        for rid, doc in rows:
            recorded = (json.loads(doc).get("source") or {}).get("tree") if doc else None
            if tree is None or recorded in (None, tree):
                found.append((rid, old))
    return [rid for rid, c in found if refused(store, rid, kinds, tool, c, command, repo) is None]


#: a Lake package's lock, which a check may regenerate (`check.py --regenerate`, the report's `regenerated`)
LOCK = "lean-audit.json"


def _run_files(store, run_id: str) -> str | None:
    out = (store.get_attempt(run_id) or {}).get("outputs") or {}
    return out.get("run_files") or out.get("run_record")


def _under_run(path: str) -> str:
    """A path the Lean check's report names, as a path among the run's files (`lean-audit/...`)."""
    if "/lean-audit/" in path:
        return "lean-audit/" + path.split("/lean-audit/", 1)[1]
    return path if path.startswith("lean-audit/") else f"lean-audit/{path}"


#: the Lean check's report among a check's run files (`lean_audit.py`): its `regenerated` names each package's regenerated lock
AUDIT = "lean-audit/audit.json"
#: the files beside the locks that the queue's lock run repairs (`lean_audit.py --bookkeeping warn`: its `UNREAD_CAPS` and
#: `CAPS_TEST`), which the report's `repaired` names by their repository paths and the queue commits with the locks
REPAIRED = frozenset({"tools/verity/lean/unread-caps.json", "tests/test_lean_packages.py"})


class Unreadable(Exception):
    """A check's Lean report says it regenerated lock records, or can't be read to say whether it did, and they can't be read:
    whoever would land its tree holds it, or the tree would land without the locks its check regenerated."""


def regenerated_records(store, run_id: str, dest: Path | None = None) -> dict[str, tuple[str, Path | None]] | None:
    """{package: (sha256 of its regenerated lock, the fetched file under ``dest`` or None)} of a check that passed its Lean
    check by regenerating records (the report's `regenerated`), and the same of each file of `REPAIRED` its lock run repaired
    (`repaired`) under its repository path, from its files in the store; empty when it regenerated none,
    None when the store holds no report of its Lean check (its files aren't there yet: a check still running).  `Unreadable`
    when the report is listed but can't be read, or names a record its files lack."""
    if not (art := _run_files(store, run_id)):
        return None
    try:
        rows = {f["path"]: f["sha256"] for f in store.listing(art)}
    except Exception:  # noqa: BLE001  (no listing: the caller reads the live run, else holds)
        return None
    if AUDIT not in rows:
        return None
    with tempfile.TemporaryDirectory() as d:
        try:
            audit = json.loads((store.fetch(art, Path(d), paths=[AUDIT]) / AUDIT).read_text())
        except (OSError, ValueError) as e:
            raise Unreadable(f"{run_id}'s {AUDIT} in the store can't be read ({type(e).__name__})") from e
    want = _regenerated_in(run_id, audit)
    if missing := sorted(pkg for pkg, p in want.items() if p not in rows):
        raise Unreadable(f"{run_id}'s {AUDIT} says it regenerated {', '.join(missing)}'s lock, which its files in the store lack")
    got = store.fetch(art, dest, paths=sorted(want.values())) if dest is not None and want else None
    return {pkg: (rows[p], (got / p) if got is not None else None) for pkg, p in want.items()}


def _regenerated_in(run_id: str, audit) -> dict[str, str]:
    """{package: its regenerated lock's path among the run's files} of a Lean check's report, and {a file of `REPAIRED`: its
    repaired copy's path there} of its `repaired`; `Unreadable` for a `repaired` naming any other file."""
    regen = audit.get("regenerated") if isinstance(audit, dict) else None
    fixed = audit.get("repaired") if isinstance(audit, dict) else None
    if not isinstance(regen or {}, dict) or not isinstance(fixed or {}, dict):
        raise Unreadable(f"{run_id}'s {AUDIT} has a `regenerated` or `repaired` that isn't an object")
    if odd := sorted(set(fixed or {}) - REPAIRED):
        raise Unreadable(f"{run_id}'s {AUDIT} says it repaired {', '.join(odd)}, which no lock run repairs (`REPAIRED`)")
    return {str(pkg): _under_run(str(p)) for pkg, p in (regen or {}).items()} | {
        str(f): _under_run(str(p)) for f, p in (fixed or {}).items()}


def fetch_run(rid: str, names: list[str], cwd: Path | None = None, refresh: bool = False) -> tuple[Path, dict] | None:
    """(run ``rid``'s local dir, its `research status --json` row) with ``names``, paths under its run dir, fetched from its node
    (`research status RID --json --also NAME...`; `landing.read_run` reads one so in the loop's process), or None when it
    can't be read.
    ``refresh`` fetches them for a run that has ended too (`--refresh`), which a plain read answers from what is kept here."""
    also = [x for n in names for x in ("--also", n)]
    r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--json", *(["--refresh"] if refresh else []), *also],
                       cwd=cwd, capture_output=True, text=True)
    try:
        row = json.loads(r.stdout)["runs"][0]
        return Path(row["dir"]), row
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def live_records(rid: str, fetch=fetch_run) -> dict[str, tuple[str, Path | None]] | None:
    """`regenerated_records` of a check whose files aren't in the store yet (a lock run the queue commits the records of: `vqueue_records`),
    read from its run dir on its node with ``fetch`` (`fetch_run`), each record's sha256 taken of the bytes fetched; None when
    its run can't be read or holds no report of its Lean check.  `Unreadable` when the report can't be parsed, or names a
    record the run dir doesn't hold."""
    if (got := fetch(rid, [AUDIT])) is None:
        return None
    try:
        audit = json.loads((got[0] / AUDIT).read_text())
    except OSError:
        return None
    except ValueError as e:
        raise Unreadable(f"{rid}'s {AUDIT} on its node can't be parsed ({e})") from e
    if not (want := _regenerated_in(rid, audit)):
        return {}
    got = fetch(rid, sorted(set(want.values())))
    if missing := sorted(pkg for pkg, p in want.items() if got is None or not (got[0] / p).is_file()):
        raise Unreadable(f"{rid}'s {AUDIT} says it regenerated {', '.join(missing)}'s lock, which can't be fetched from its "
                         "run dir on its node")
    return {pkg: (hashlib.sha256((got[0] / p).read_bytes()).hexdigest(), got[0] / p) for pkg, p in want.items()}


def lean_report(store, rid: str, fetch=fetch_run):
    """Run ``rid``'s Lean report (`AUDIT`, parsed), read from its files in the store, else from its run dir on its node
    (``fetch``); None when neither has it to read.  Raises when it is listed and can't be read or parsed."""
    if (art := _run_files(store, rid)) and AUDIT in {f["path"] for f in store.listing(art)}:
        with tempfile.TemporaryDirectory() as d:
            return json.loads((store.fetch(art, Path(d), paths=[AUDIT]) / AUDIT).read_text())
    if (got := fetch(rid, [AUDIT])) is not None:
        return json.loads((got[0] / AUDIT).read_text())
    return None


def bookkeeping_warnings(store, rid: str, fetch=fetch_run) -> list[str]:
    """The warnings of lock run ``rid``'s bookkeeping repairs (its Lean report's `bookkeeping`, `lean_report`); empty when it
    has none or its report can't be read: the queue tells them and puts them in the lock commit, and nothing waits on them."""
    try:
        audit = lean_report(store, rid, fetch)
        warned = audit.get("bookkeeping") if isinstance(audit, dict) else None
        return [str(w) for w in warned] if isinstance(warned, list) else []
    except Exception as e:  # noqa: BLE001  (warnings only: no report, no warnings)
        print(f"research queue: lock run {rid}'s bookkeeping warnings unread ({type(e).__name__}: {e}); its lock commit "
              f"lists none", file=sys.stderr)
        return []


def _blob_sha256(repo: Path, commit: str, path: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else ""


def _ran_on(store, rid: str, commit: str, fetch) -> bool:
    """Whether run ``rid`` checked ``commit``: its attempt's clean source in the store, else its run record's `tree_sha`."""
    rows = store.index.sql("SELECT source_commit, source_dirty FROM attempts WHERE id = ?", (rid,))
    if rows:
        return rows[0][0] == commit and not rows[0][1]
    got = fetch(rid, []) if fetch is not None else None
    return got is not None and got[1].get("tree_sha") == commit


def regenerated_parent(repo: Path, commit: str, store, *, tool: str, kinds: frozenset[str], command: tuple[str, ...],
                       maps=None, deferred: str | None = None, fetch=None) -> str | None:
    """The passing ``tool`` attempt of ``commit``'s one parent that vouches for ``commit``, or None: ``commit`` changes only Lake
    packages' locks, names exactly those in its `Regenerated:` trailer, and they are byte for byte the locks that attempt
    regenerated, one for each package it regenerated, so the tree that lands is the one that check audited.  A file of
    `REPAIRED` it changes is held the same way, to the copy the lock run repaired (`--bookkeeping warn`).

    ``deferred`` is the lock run the queue commits ``commit`` for on the parent (`vqueue_records`), a check of lean-audit
    alone, so no passing attempt of the whole check: it vouches when ``commit``'s `Checked:` trailer names it, it ran on the
    parent, and its records match, read from the store, else from its run dir on its node (``fetch``: `live_records`), hashed
    here, never digests a caller hands in.  The queue passes it, having read that verdict itself."""
    parents = _git(repo, "rev-list", "--parents", "-n", "1", commit).stdout.split()[1:]
    if len(parents) != 1:
        return None
    parent = parents[0]
    files = _git(repo, "diff", "--name-only", "--no-renames", parent, commit).stdout.split()
    said = {f.strip() for f in _git(repo, "log", "-1", "--format=%(trailers:key=Regenerated,valueonly,separator=%x2C)",
                                     commit).stdout.replace("\n", ",").split(",") if f.strip()}
    if not files or set(files) != said or not all(os.path.basename(f) == LOCK or f in REPAIRED for f in files):
        return None
    if deferred is not None:
        checked = re.findall(r"^Checked: (\S+)", _git(repo, "log", "-1", "--format=%B", commit).stdout, re.M)
        ids = [deferred] if checked == [deferred] and _ran_on(store, deferred, parent, fetch) else []
    else:
        tree = _git(repo, "rev-parse", f"{parent}^{{tree}}").stdout.strip()
        ids = passing(store, parent, tool, kinds=kinds, command=command, tree=tree, maps=maps, repo=repo)
    lock = {f if f in REPAIRED else os.path.dirname(f) or ".": _blob_sha256(repo, commit, f) for f in files}
    for rid in ids:
        try:
            recs = regenerated_records(store, rid)
            if recs is None and fetch is not None:
                recs = live_records(rid, fetch)
        except Exception:  # noqa: BLE001  (`Unreadable`, or the store failing: nothing to vouch with)
            continue
        if recs and {(pkg or "."): sha for pkg, (sha, _) in recs.items()} == lock:
            return rid
    return None


def required_steps(repo: Path, base: str, commit: str, tool: str, at: str | None = None) -> dict[str, list[str]]:
    """The steps the target's own declaration of ``tool`` (`Tool.merge_requires`, read from the checkout, or from commit ``at``
    when given; never from the commit being merged) requires for the change ``base``..``commit``: {step: the prefixes it
    touches}."""
    rules = getattr(declaration(repo, tool, at), "merge_requires", None)
    if not rules:
        return {}
    changed = [p for p in _git(repo, "diff", "--no-renames", "--name-only", "-z", base, commit).stdout.split("\0") if p]   # a rename's source too
    out = {}
    for step, rule in rules.items():
        if hit := rule_touches(rule, changed):
            out[step] = hit
    return out


def rule_touches(rule: list[str], changed: list[str]) -> list[str]:
    """The prefixes of a `Tool.merge_requires` rule that ``changed`` touches.  A rule is path prefixes and `!` patterns: a
    changed file counts when it is under a prefix and matches no pattern (one without a `/` is matched against the file's name,
    one with a `/` against its path), so `["verity/protocols/proofs/flock/prover/", "verity/protocols/proofs/flock/verifier/", "!README.md"]` leaves out every
    README.md under either."""
    import fnmatch

    prefixes = [r for r in rule if not r.startswith("!")]
    ignored = [r[1:] for r in rule if r.startswith("!")]

    def counts(c: str) -> bool:
        return not any(fnmatch.fnmatchcase(c if "/" in g else c.rsplit("/", 1)[-1], g) for g in ignored)

    return [p for p in prefixes if any((c == p.rstrip("/") or c.startswith(p.rstrip("/") + "/")) and counts(c) for c in changed)]


def step_status(store, run_id: str, step: str) -> str:
    """The status ``step`` has in the attempt's result.json ``steps`` ("absent" when it has none).  A job a shard plan splits
    (lean-audit, by Lake package): in a `--combine` check, its parts' (`STEP:PART`), passed only when each is; in a shard
    (`SHARD_KIND`, or a deferred run's shard: the after-landing run's shards), "partial" unless that shard ran every part of it,
    since a shard alone checks only its own."""
    try:
        art = (store.get_attempt(run_id) or {}).get("outputs", {}).get("result")
        doc = store.get_manifest(art).meta if art else {}       # a result artifact's document is its manifest's meta
    except Exception as e:  # noqa: BLE001  (an unreadable result admits nothing)
        return f"unreadable ({type(e).__name__})"
    steps = [s for s in doc.get("steps", []) if isinstance(s, dict)]
    mine = [s for s in steps if s.get("step") == step]
    sh = next((x for x in (doc.get("shard"), doc.get("deferred")) if isinstance(x, dict) and x.get("all")), None)
    if sh and any(s.get("parts") for s in mine):
        every = [p for p in sh["all"] if isinstance(p, str) and p.partition(":")[0] == step]
        if sorted(p for s in mine for p in s.get("parts") or ()) != sorted(every):
            return "partial"
    if mine:
        return mine[0].get("status", "absent")
    parts = [s.get("status") or "absent" for s in steps if str(s.get("step", "")).startswith(step + ":")]
    return next((x for x in parts if x != "passed"), "passed") if parts else "absent"


#: the subject of a pinned pull request's merge in main's log
MERGED_PR = re.compile(r"Merge #(\d+) \(([0-9a-f]{7,40})\)")


def adds_nothing(repo: Path, commit: str, head: str, lazy: bool = True) -> bool:
    """Whether merging ``head`` into ``commit`` leaves ``commit``'s own tree: what ``head`` has, ``commit`` has already (a merge of
    `main` into a pull request the tip already holds); with ``lazy`` False, not when that needs an object the clone lacks."""
    r = _git(repo, "merge-tree", "--write-tree", commit, head, check=False, lazy=lazy)
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == _git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()


#: gh has no timeout of its own: a call on a dead connection can hold a landing's steps for minutes
GH_TIMEOUT_S = 120


def _gh_cli(repo: Path, *args: str) -> str:
    try:
        r = subprocess.run(["gh", *args], cwd=repo, capture_output=True, text=True, timeout=GH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Refused(f"gh {' '.join(args[:2])}: no answer in {GH_TIMEOUT_S} s") from None
    if r.returncode:
        raise Refused(f"gh {' '.join(args[:2])}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def landed_in(repo: Path, commit: str, tip: str) -> str:
    """The commit on ``tip``'s first-parent line that brought ``commit`` in: ``commit`` itself when it is on that line, otherwise
    the oldest commit of the line that contains it (the merge that landed it)."""
    after = set(_git(repo, "rev-list", "--ancestry-path", f"{commit}..{tip}").stdout.split())
    landed = commit
    for c in _git(repo, "rev-list", "--first-parent", "-n", str(len(after) + 1), tip).stdout.split():
        if c not in after:
            return commit if c == commit else landed
        landed = c
    return landed


#: the merged and closed PRs most recently updated (a merge or a close updates one), which GitHub reads consistently, as a
#: search doesn't
ENDED_QUERY = ("query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) { pullRequests(states: [MERGED, "
               "CLOSED], first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { number state headRefName headRefOid "
               "isCrossRepository } } } }")


def sweep(repo: Path, into: str = "main", remote: str = "origin", dry_run: bool = False, gh=None, log=print,
          keep: Iterable[str] = ()) -> tuple[list[int], list[int]]:
    """Close every open pull request whose head is on ``remote``'s ``into``, or whose pinned head landed and whose head since
    adds nothing to it (`adds_nothing`; its branch is kept), with a comment naming the commit it landed in, and
    retarget onto ``into`` every open pull request whose base branch's tip is on it, deleting each closed one's head branch that no
    open pull request uses.  Then the head branch of each pull request GitHub has as merged (its head on ``into``) or closed
    unmerged goes too, while it is still at that PR's head: never a fork's, one of `KEEP_BRANCHES`, one an open pull request
    uses as its head or base, a closed one's that is a lead's run branch (`RUN_BRANCH_RE`), or ``keep`` (the heads of PRs
    the queue closed as stale, which stay to be reopened: `research.stale`); GitHub
    keeps each one's commits under
    `refs/pull/N/head`.  Idempotent: what it closed is no longer open, and what it retargeted is based on ``into``.  Returns
    (closed, retargeted) PR numbers, ascending."""
    gh = gh or (lambda *a: _gh_cli(repo, *a))
    _git(repo, "fetch", "-q", remote, f"+refs/heads/{into}:refs/remotes/{remote}/{into}")
    tip = _git(repo, "rev-parse", f"refs/remotes/{remote}/{into}").stdout.strip()

    # a head the clone lacks once `into` is fetched isn't on it, so a partial clone fetches none to say so: a lazy fetch, with no
    # negotiation, brings that head's whole history, and git's auto maintenance never repacks such packs
    def on(sha: str) -> bool:
        return _git(repo, "merge-base", "--is-ancestor", sha, tip, check=False, lazy=False).returncode == 0

    local = _git(repo, "rev-parse", "--verify", "-q", f"refs/heads/{into}", check=False).stdout.strip()
    if local and not on(local):
        log(f"research queue: {into} has commits {remote}/{into} lacks; the sweep closes only what is on {remote}/{into}")
    prs = sorted(json.loads(gh("pr", "list", "--state", "open", "--limit", "1000", "--json",
                               "number,headRefName,headRefOid,baseRefName")), key=lambda p: p["number"])
    done = [p for p in prs if p["headRefName"] != into and on(p["headRefOid"])]
    rest = [p for p in prs if p not in done]
    bases = sorted({p["baseRefName"] for p in rest if p["baseRefName"] != into})
    tips = {}
    if bases:
        out = _git(repo, "ls-remote", remote, *[f"refs/heads/{b}" for b in bases]).stdout
        tips = {ref.removeprefix("refs/heads/"): sha for sha, ref in (line.split() for line in out.splitlines())}
    moved = [p for p in rest if p["baseRefName"] in tips and on(tips[p["baseRefName"]])]
    # a PR whose pinned head landed and whose head since adds nothing to `into` (it merged `main` in)
    pinned = {int(m[1]): parents.split()[1] for parents, _, subject in (line.partition("\0") for line in _git(
        repo, "log", "-n", "5000", "--format=%P%x00%s", "--grep", r"^Merge #[0-9]", tip).stdout.splitlines())
        if (m := MERGED_PR.fullmatch(subject)) and len(parents.split()) == 2}
    done += [p for p in rest if p not in moved and p["number"] in pinned and adds_nothing(repo, tip, p["headRefOid"], lazy=False)]
    rest = [p for p in rest if p not in done]
    verb = "would " if dry_run else ""
    for p in moved:
        log(f"research queue: {verb}retarget #{p['number']} from {p['baseRefName']} (on {remote}/{into}) onto {into}")
        if not dry_run:
            gh("pr", "edit", str(p["number"]), "--base", into)
    # a closed PR's head branch goes too, unless an open PR still uses it (as its head, or as a base the retargets left)
    used = {p["headRefName"] for p in rest} | {p["baseRefName"] for p in rest if p not in moved}
    for p in done:
        later = not on(p["headRefOid"])
        landed = landed_in(repo, pinned[p["number"]] if later else p["headRefOid"], tip)
        subject = _git(repo, "log", "-1", "--format=%s", landed).stdout.strip()
        note = f"Merged into {into} at {landed[:12]} ({subject})." + (
            f" Its head since, {p['headRefOid'][:12]}, adds nothing to {into}." if later else "")
        log(f"research queue: {verb}close #{p['number']} ({p['headRefName']}): {note}")
        if not dry_run:
            gh("pr", "close", str(p["number"]), "--comment", note)
        head = p["headRefName"]
        if head in used or later:
            continue
        log(f"research queue: {verb}delete {remote}'s branch {head} (#{p['number']}'s head, on {into})")
        if not dry_run:
            # the lease keeps a branch that moved past the PR's landed head: a commit pushed after the landing isn't on `into`
            r = _git(repo, "push", "-q", f"--force-with-lease=refs/heads/{head}:{p['headRefOid']}", remote, f":refs/heads/{head}",
                     check=False)
            if r.returncode:
                log(f"research queue: kept {head}: {(r.stderr or r.stdout).strip()[-200:]}")
    # a PR GitHub marked merged itself (its head reached `into` by the queue's landing) leaves its branch too (most branches
    # are already on main), and so does one closed unmerged; the most recently updated, not created, first (`gh pr list`'s
    # newest 200 by number leave out PRs that waited in the queue, whose branches would stay)
    ended = json.loads(gh("api", "graphql", "-F", "owner={owner}", "-F", "name={repo}", "-f", f"query={ENDED_QUERY}",
                          "--jq", ".data.repository.pullRequests.nodes"))
    # only a branch still at the PR's head: a lease on a branch already gone fails as "stale info"
    heads = {ref.removeprefix("refs/heads/"): sha for sha, ref in (
        line.split() for line in _git(repo, "ls-remote", "--heads", remote).stdout.splitlines())}
    keep = set(keep)
    for p in ended:
        head, closed = p["headRefName"], p.get("state") == "CLOSED"
        if (head in used or head in KEEP_BRANCHES or p.get("isCrossRepository") or heads.get(head) != p["headRefOid"]
                or (closed and (head in keep or RUN_BRANCH_RE.fullmatch(head))) or not (closed or on(p["headRefOid"]))):
            continue
        log(f"research queue: {verb}delete {remote}'s branch {head} "
            + (f"(closed #{p['number']}'s head, unmerged)" if closed else f"(merged #{p['number']}'s head, on {into})"))
        if not dry_run:
            r = _git(repo, "push", "-q", f"--force-with-lease=refs/heads/{head}:{p['headRefOid']}", remote, f":refs/heads/{head}",
                     check=False)
            if r.returncode and "remote ref does not exist" not in (r.stderr or ""):
                log(f"research queue: kept {head}: {(r.stderr or r.stdout).strip()[-200:]}")
    if not done and not moved:
        log(f"research queue: no open pull request is on {remote}/{into} ({tip[:12]}) or based on a branch that is")
    return [p["number"] for p in done], [p["number"] for p in moved]


#: the old path's landing branch, frozen since the verified queue took over, which the queue reads as landed
CHECKED = "checked"
#: branches no sweep touches: the trunk and the old path's frozen branches
KEEP_BRANCHES = frozenset({"main", CHECKED, "next", "queue-shadow"})
#: a lead's run branch (`cursor/verity-<lead>-XXXX`), which its agent pushes to again
RUN_BRANCH_RE = re.compile(r"cursor/verity-[a-z0-9-]+-[0-9a-f]{4}")


def _sweep_after(repo: Path, into: str, remote: str, keep: Iterable[str] = ()) -> None:
    """The sweep after a merge landed (``keep``: `sweep`'s): skipped without the remote; a failure is reported and leaves the
    merge's result alone."""
    if _git(repo, "remote", "get-url", remote, check=False).returncode:
        print(f"research queue: no remote {remote!r}; pull request sweep skipped")
        return
    try:
        sweep(repo, into, remote, keep=keep)
    except (Refused, ValueError, KeyError) as e:
        print(f"research queue: pull request sweep failed ({e}); the next landing's sweep closes what this one left",
              file=sys.stderr)


MIRROR, MIRROR_ENV, MIRROR_TIMEOUT_S = "vy-nebius-cpu-4", "RESEARCH_MERGE_MIRROR", 120


def mirror_name() -> str | None:
    name = os.environ.get(MIRROR_ENV, MIRROR).strip()
    return None if name.lower() in ("", "none", "off") else name


def _mirror_target(name: str) -> tuple[list[str], str]:
    """``name``'s ssh argv, as `research deploy` and `research run --on` reach it, and its bare repository."""
    from . import remote

    m = remote.resolve_machine(name, "verity")
    t = remote.ssh_host_target(m)
    return remote.ssh_argv(t["host"], t["port"], user=t["user"]), f"{m['root']}/{remote.POD_GIT_REPO}"


def push_mirror(repo: Path, sha: str, name: str, target=None, run=subprocess.run) -> bool:
    """``sha`` to the main of ``name``'s bare repository: a plain push, so only a fast-forward, past this checkout's own pre-push
    guard, which judges pushes that move origin's main (a hand landing has no check attempt of its tree, so the guard would refuse
    node 1's copy of it). False, after one line naming the failure and the command that reruns it,
    when it doesn't land."""
    from .remote import git_push_url

    try:
        ssh, bare = (target or _mirror_target)(name)
        r = run(["git", "-c", "protocol.ext.allow=always", "push", "-q", "--no-verify", git_push_url(ssh, bare), f"{sha}:refs/heads/main"],
                cwd=repo, capture_output=True, text=True, timeout=MIRROR_TIMEOUT_S)
        why = None if r.returncode == 0 else f"git push exited {r.returncode}: {' '.join((r.stderr or r.stdout).split())[-300:]}"
    except subprocess.TimeoutExpired:
        why = f"no answer within {MIRROR_TIMEOUT_S} s"
    except Exception as e:                                      # noqa: BLE001: a machine that can't be reached fails only this push
        why = f"{type(e).__name__}: {e}"
    if why:
        print(f"research queue: main ({sha[:12]}) not pushed to {name} ({why}); the next landing pushes its tip again", file=sys.stderr)
        return False
    print(f"research queue: pushed main ({sha[:12]}) to {name}:{bare}")
    return True


#: the Lean cache publisher on the control pod: `SCRIPT SHA` fast-forwards vy-nebius-cpu-4's bare main to SHA and starts
#: vy-lean-cache-daily there (a no-op while one runs), exits 0 once started and logs to LEAN_PUBLISH_LOG; $RESEARCH_LEAN_PUBLISH
#: names another script, and empty skips it
LEAN_PUBLISH, LEAN_PUBLISH_ENV, LEAN_PUBLISH_LOG, LEAN_PUBLISH_TIMEOUT_S = (
    "/data/infra/lean_publish.sh", "RESEARCH_LEAN_PUBLISH", "/data/drive/lean-publish.log", 120)


def lean_publish(repo: Path, start: str, tip: str, log=print, run=subprocess.run, clock=time.monotonic) -> bool:
    """Start the Lean cache's publish of ``tip`` when what landed since ``start`` touches a Lake package's lock or Lean sources
    (`queue.lean_change`), with one line in ``log`` saying whether it started, why not, and the seconds the script took to start.
    It waits only for the script's start, bounded by LEAN_PUBLISH_TIMEOUT_S, and nothing raises: the landing stands either way."""
    from .queue import lean_change

    script, what = os.environ.get(LEAN_PUBLISH_ENV, LEAN_PUBLISH).strip(), f"research queue: the Lean cache publish of {tip[:12]}"
    if not script:
        log(f"{what} is off (${LEAN_PUBLISH_ENV} is empty)")
        return False
    try:
        touched = lean_change(repo, tip, base=start)
    except Exception as e:                                      # noqa: BLE001: a diff that can't be read skips only the publish
        log(f"{what} not started: {start[:12]}..{tip[:12]} can't be diffed ({type(e).__name__}: {e})")
        return False
    if not touched:
        log(f"{what} not started: {start[:12]}..{tip[:12]} touches no Lean package")
        return False
    if not Path(script).is_file():
        log(f"{what} not started: {script} is absent")
        return False
    t0 = clock()
    try:
        r = run([script, tip], cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=LEAN_PUBLISH_TIMEOUT_S)
        why = None if r.returncode == 0 else f"exited {r.returncode}: {' '.join((r.stderr or r.stdout or '').split())[-300:]}"
    except subprocess.TimeoutExpired:
        why = f"no exit within {LEAN_PUBLISH_TIMEOUT_S} s"
    except Exception as e:                                      # noqa: BLE001: a script that can't run fails only the publish
        why = f"{type(e).__name__}: {e}"
    took = clock() - t0
    if why:
        log(f"{what} not started: {script} {why} ({took:.1f} s); its log is {LEAN_PUBLISH_LOG}; rerun: {script} {tip}")
        return False
    log(f"{what} started by {script} in {took:.1f} s; its minutes and bytes are in {LEAN_PUBLISH_LOG}")
    return True


#: `sh -c` around the after-landing command: run it (``$2``...) in its worktree ``$0``, then remove that from ``$1``'s repository
AFTER_LAND_THEN_REMOVE = 'r=$1; shift; "$@"; s=$?; cd / && git -C "$r" worktree remove --force "$0"; exit $s'


def _own_tree(repo: Path, commit: str) -> Path | None:
    """A detached worktree of ``commit`` for the after-landing command alone, since the caller's checkout moves on under it
    (the queue's clone is reset and cleaned every tick, which would take a running `post_train.py`'s `.venv`).  It goes
    under TRAIN_TREES when set, as train.sh's trees do, else in the temporary directory: the control pod's TMPDIR is a network
    volume, where each worktree add and remove takes minutes.  It is added under `queue.worktrees_locked`, as the queue adds
    its own in that clone.  None when ``repo`` can't make one."""
    import shutil

    from .queue import worktrees_locked
    from .settings import train_trees

    if _git(repo, "rev-parse", "--git-common-dir", check=False).returncode:
        return None
    if root := train_trees():
        os.makedirs(root, exist_ok=True)
    tree = Path(tempfile.mkdtemp(prefix=f"research-after-land-{commit[:12]}-", dir=root))
    with worktrees_locked(repo):
        added = _git(repo, "worktree", "add", "-q", "--detach", str(tree), commit, check=False).returncode == 0
    if not added:
        shutil.rmtree(tree, ignore_errors=True)
        return None
    return tree


def after_land(repo: Path, tool: str, commit: str, log=print, spawn=subprocess.Popen, attempt: str = "") -> list[str] | None:
    """Start what the target's own declaration of ``tool`` runs after a landing (`Tool.after_land`, read from the checkout as
    `required_steps` reads `merge_requires`) on ``commit``, in a session of its own, so it outlives this process and nothing it
    does changes the merge's result, and in a worktree of ``commit`` of its own (`_own_tree`) that goes when it ends.  With
    ``attempt``, the check the landing was admitted on, it gets `--attempt RID` after the commit: the queue's landings carry no
    `Check:` trailer.  Returns the argv started, None when the declaration names none."""
    from .store import tools_registry

    try:
        argv = [*tools_registry.from_tree(tool, repo).after_land]
    except Exception:  # noqa: BLE001  (a tool the checkout does not declare runs nothing after a landing)
        return None
    if not argv:
        return None
    argv += [commit, "--attempt", attempt] if attempt else [commit]
    out = Path(tempfile.gettempdir()) / f"research-after-land-{commit[:12]}.log"
    tree = _own_tree(repo, commit)
    run = argv if tree is None else ["sh", "-c", AFTER_LAND_THEN_REMOVE, str(tree), str(repo), *argv]
    try:
        with open(out, "ab") as f:
            spawn(run, cwd=tree or repo, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        if tree is not None:
            _git(repo, "worktree", "remove", "--force", str(tree), check=False)
        log(f"research queue: after the landing, `{shlex.join(argv)}` did not start ({e}); run it by hand")
        return argv
    log(f"research queue: after the landing, started `{shlex.join(argv)}` in {tree or repo} (log {out})")
    return argv


#: the remote's ref holding the main commit the after-landing steps last ran for
AFTER_LAND_REF = "refs/research/after-land"
#: the remote's ref holding the main commit the spec-change DM last went up to: it moves only when the DM went, so one
#: that failed is sent again by the next landing's steps
SPEC_ALERTED_REF = "refs/research/spec-alerted"


def _remote_refs(repo: Path, remote: str, *refs: str) -> dict[str, str] | None:
    r = _git(repo, "ls-remote", remote, *refs, check=False)
    return None if r.returncode else {ref: sha for sha, ref in (line.split() for line in r.stdout.splitlines())}


def _move(repo: Path, remote: str, ref: str, was: str, tip: str, log=print) -> bool:
    """Push ``tip`` to ``remote``'s ``ref``, leased on ``was`` (the one read) so two landers don't both advance it."""
    p = _git(repo, "push", "-q", f"--force-with-lease={ref}:{was}", remote, f"{tip}:{ref}", check=False)
    if p.returncode:
        log(f"research queue: {remote}'s {ref} wasn't moved to {tip[:12]} ({(p.stderr or p.stdout).strip()[-200:]}); "
            "another lander may have caught up first")
    return not p.returncode


def caught_up(repo: Path, remote: str, tool: str = "check", since: str = "", log=print,
              runs: dict[str, str] | None = None) -> str | None:
    """Run the after-landing steps once for what landed on ``remote``'s main since AFTER_LAND_REF there (or ``since``, main before
    the caller's own push, when the remote has no marker yet): push main's tip to node 1's mirror, start ``tool``'s `after_land`
    on the tip, with the attempt ``runs`` names for it (a later landing's `post_train.py` tests the ones before it too), start the
    Lean cache publish when what landed touches a Lean package (`lean_publish`), post each landing to the merge thread
    (`report_landings`) and label the runs each one names (`label_landed`), then move the marker to the tip.  The spec DM
    (`spec_alert`) goes over SPEC_ALERTED_REF..tip (AFTER_LAND_REF's start when that marker is new) and moves its marker only
    when it was sent, so a failed DM is retried by every later run until it goes.  Without a marker or ``since`` it only sets
    the markers.  Nothing raises, since a landing that happened stands.  Returns the tip the steps ran for."""
    from .spec_alert import alert

    refs = _remote_refs(repo, remote, "refs/heads/main", AFTER_LAND_REF, SPEC_ALERTED_REF)
    if refs is None:
        log(f"research queue: {remote}'s main can't be read; the steps after a landing wait for the next landing")
        return None
    tip, mark, alerted = refs.get("refs/heads/main", ""), refs.get(AFTER_LAND_REF, ""), refs.get(SPEC_ALERTED_REF, "")
    start = mark or since
    spec_from = alerted or start
    if not tip or (tip == mark and spec_from in ("", tip)):
        return None
    if start or spec_from:
        fetched = _git(repo, "fetch", "-q", remote, "refs/heads/main", *(r for r, s in ((AFTER_LAND_REF, mark),
                                                                                         (SPEC_ALERTED_REF, alerted)) if s),
                       check=False)
        if fetched.returncode:
            log(f"research queue: {remote}'s main or its markers can't be fetched; the steps after a landing wait for the next "
                "landing")
            return None
    if spec_from and spec_from != tip:
        if _git(repo, "merge-base", "--is-ancestor", spec_from, tip, check=False).returncode:
            log(f"research queue: the spec-change DM last went up to {spec_from[:12]}, which isn't in {remote}'s main "
                f"({tip[:12]}); no DM was sent: send it by hand, then `git push {remote} {tip}:{SPEC_ALERTED_REF}`")
        elif alert(repo, spec_from, tip, remote=remote, log=log):
            _move(repo, remote, SPEC_ALERTED_REF, alerted, tip, log)
        else:
            log(f"research queue: the spec DM over {spec_from[:12]}..{tip[:12]} wasn't sent; {SPEC_ALERTED_REF} stays at "
                f"{spec_from[:12]}, so the next landing's steps send it again")
    elif not alerted:
        _move(repo, remote, SPEC_ALERTED_REF, "", tip, log)
    if tip == mark:
        return None
    if start:
        if _git(repo, "merge-base", "--is-ancestor", start, tip, check=False).returncode:
            log(f"research queue: the steps after a landing last ran for {start[:12]}, which isn't in {remote}'s main "
                f"({tip[:12]}); nothing ran: run them by hand, then `git push {remote} {tip}:{AFTER_LAND_REF}`")
            return None
        if mark and mark != since:
            n = len(_git(repo, "rev-list", "--first-parent", f"{mark}..{tip}").stdout.split())
            log(f"research queue: the steps after a landing last ran for {mark[:12]}; catching up {n} commit(s) of main to {tip[:12]}")
        if remote == "origin" and (name := mirror_name()):
            push_mirror(repo, tip, name)
        after_land(repo, tool, tip, log=log, attempt=(runs or {}).get(tip, ""))
        if remote == "origin":
            lean_publish(repo, start, tip, log=log)
            report_landings(repo, start, tip, log=log, runs=runs)
            label_landed(repo, start, tip, log=log, runs=runs)
    if _move(repo, remote, AFTER_LAND_REF, mark, tip, log) and not start:
        log(f"research queue: {remote} had no {AFTER_LAND_REF}; set it to main ({tip[:12]}): the steps after a landing are "
            "caught up from here, and nothing before it was looked at")
    return tip if start else None


#: what a landing merged, in its own subject or, for an old landing, its train's: a pull request, a branch of origin, or a commit
MERGED_REF = re.compile(r"Merge (#\d+|origin/\S+|[0-9a-f]{12,40}) \([0-9a-f]{7,40}\)|^Queue: (#\d+) ")
CHECK_TRAILER = re.compile(r"^Check(?:ed)?: (.*)$", re.M)
#: a run id, or the lander's form with `*` for a launch time it didn't copy
RUN_ID = re.compile(r"r20\d{6}-(?:\d{6}|\*)-[0-9a-f]{4}")


def _merged_name(ref: str, prs: dict[str, int]) -> str:
    if ref.startswith("#"):
        return ref
    if (branch := ref.removeprefix("origin/")) != ref:
        return f"#{prs[branch]}" if branch in prs else branch
    sha = next((k for k in prs if len(k) == 40 and k.startswith(ref)), None)
    return f"#{prs[sha]}" if sha else ref[:12]


def landing_lines(repo: Path, start: str, tip: str, prs: dict[str, int] | None = None,
                  runs: dict[str, str] | None = None) -> list[str]:
    """One line per commit on ``tip``'s first-parent line after ``start``: what it merged (pull requests, or the branches and
    commits ``prs``, head branch or head sha to number, doesn't map to one) and the runs its `Check:` or `Checked:` trailer
    names, else the run ``runs`` gives for it (a commit the queue fast-forwarded has no trailer).  Read from main's log, since a pull request's state can't stand in: a stacked member can read closed on
    GitHub without having landed on its own."""
    prs = prs or {}
    lines = []
    for c in _git(repo, "rev-list", "--first-parent", "--reverse", f"{start}..{tip}").stdout.split():
        parents = _git(repo, "rev-list", "--parents", "-n", "1", c).stdout.split()[1:]
        subject, body = _git(repo, "log", "-1", "--format=%s%x00%b", c).stdout.split("\0", 1)
        subjects = [subject]
        if len(parents) == 2:
            subjects += _git(repo, "log", "--first-parent", "--format=%s", f"{parents[0]}..{parents[1]}").stdout.splitlines()
        merged = []
        for s in subjects:
            for m in MERGED_REF.finditer(s):
                if m[2] is None and len(parents) == 2 and parents[1].startswith(m[1]):
                    continue                            # an old train's own tip, whose merges name what it holds
                if (ref := _merged_name(m[1] or m[2], prs)) not in merged:
                    merged.append(ref)
        what = " + ".join(reversed(merged)) if merged else f"“{subject}”"
        found = [r for t in CHECK_TRAILER.findall(body) for r in RUN_ID.findall(t)] or RUN_ID.findall((runs or {}).get(c, ""))
        lines.append(f"landed on main: {c[:12]} {what}; checked by {', '.join(dict.fromkeys(found)) or 'no run (no Check: trailer)'}")
    return lines


#: who writes the `landed` labels (`label_landed`)
LANDED_BY = "research-merge"


def landed_runs(repo: Path, start: str, tip: str, runs: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
    """(run, commit, commit time as UTC `...Z`) for each run the `Check:` or `Checked:` trailer of a commit on ``tip``'s first-parent
    line after ``start`` (the whole line when ``start`` is empty) names, else the run ``runs`` gives for it (as `landing_lines`),
    oldest first; the lander's `*` ids as written."""
    out = []
    for c in _git(repo, "rev-list", "--first-parent", "--reverse", f"{start}..{tip}" if start else tip).stdout.split():
        body, ct = _git(repo, "log", "-1", "--format=%b%x00%ct", c).stdout.rsplit("\0", 1)
        ts = datetime.fromtimestamp(int(ct), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        found = [r for t in CHECK_TRAILER.findall(body) for r in RUN_ID.findall(t)] or RUN_ID.findall((runs or {}).get(c, ""))
        out += [(r, c, ts) for r in dict.fromkeys(found)]
    return out


def _resolve_run(remote, run: str) -> str | None:
    """The one run on ``remote`` a lander's `r<date>-*-<suffix>` id can name, or None (no remote, none, or several)."""
    if "*" not in run:
        return run
    if remote is None:
        return None
    date, _, suffix = run.split("-")
    hits = [k.rsplit("/", 1)[-1].removesuffix(".json") for k in remote.list(f"attempts/{date}-") if k.endswith(f"-{suffix}.json")]
    return hits[0] if len(hits) == 1 else None


def label_landed(repo: Path, start: str, tip: str, *, store=None, log=print, runs: dict[str, str] | None = None) -> bool:
    """Write a `landed` label on each check run a landing over ``start``..``tip`` names (`landed_runs`): its value main's merge
    commit, its ts the later of the commit's time and the run's start (its id's), by LANDED_BY, so a rerun anywhere writes the
    same assertions (the store README's `landed` label); then push them.  True when every label reached the remote; nothing raises, since the landings stand."""
    try:
        from .store import labels_sync as LS
        from .store.local import LocalStore

        found = landed_runs(repo, start, tip, runs)
        if not found:
            return True
        st = store if store is not None else LocalStore(None)
        done, unresolved = [], []
        for run, commit, ts in found:
            if (rid := _resolve_run(st.remote, run)) is None:
                unresolved.append(run)
                continue
            d = rid[1:16]
            ts = max(ts, f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[9:11]}:{d[11:13]}:{d[13:15]}Z")    # the lander's commits carry a fixed date
            if not any(lab.key == "landed" and lab.value == commit and lab.by == LANDED_BY for lab in st.labels(rid)):
                st.add_label(rid, "landed", commit, LANDED_BY, ts=ts)
            done.append(rid)
        if unresolved:
            log(f"research queue: no `landed` label on {', '.join(unresolved)}: the id names no single run on the remote")
        if st.remote is None:
            log(f"research queue: {len(done)} `landed` label(s) stay in {st.root} (no remote): `research data push --pending` uploads them")
            return False
        failed = LS.push_targets(st, sorted(set(done)))["failed"] if done else []
        log(f"research queue: {len(done)} `landed` label(s) on the check runs of {start[:12] or 'the root'}..{tip[:12]}"
            + (f"; {len(failed)} not pushed: {failed[0]}" if failed else ""))
        return not failed and not unresolved
    except Exception as e:                              # noqa: BLE001: a label that fails leaves the landings as they are
        print(f"research queue: the `landed` labels over {start[:12]}..{tip[:12]} weren't written: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False


#: a `git revert` commit's own line naming what it reverted
REVERTS = re.compile(r"^This reverts commit ([0-9a-f]{40})", re.M)


def report_landings(repo: Path, start: str, tip: str, *, api=None, reg: dict | None = None, gh=None, log=print,
                    runs: dict[str, str] | None = None) -> bool:
    """Post `landing_lines` over ``start``..``tip`` to the merge thread, one line per landing, so every landing is reported
    whoever pushed it.  True when posted or there was nothing to
    post; a failure is printed and returns False, and nothing raises, since the landings stand."""
    from .monitors import MERGE_THREAD
    from .slack import FRONT_DOOR, QUIET, Slack, _footer, load_registry, registry_path

    try:
        gh = gh or (lambda *a: _gh_cli(repo, *a))
        try:
            prs = {k: p["number"] for p in sorted(json.loads(gh("pr", "list", "--state", "all", "--limit", "300", "--json",
                                                                 "number,headRefName,headRefOid")), key=lambda p: p["number"])
                   for k in (p["headRefName"], p["headRefOid"])}
        except Exception:                               # noqa: BLE001: without GitHub a branch is named by its branch
            prs = {}
        lines = landing_lines(repo, start, tip, prs, runs)
        if not lines:
            return True
        reg = reg if reg is not None else load_registry(registry_path())
        api = api if api is not None else Slack(os.environ.get("SLACK_BOT_TOKEN"))
        api.call("chat.postMessage", channel=reg["channels"][FRONT_DOOR]["id"], thread_ts=MERGE_THREAD,
                 text="\n".join(lines) + _footer((QUIET,)), username="research queue", unfurl_links=False)
        log(f"research queue: posted {len(lines)} landing(s) to the merge thread")
        return True
    except Exception as e:                              # noqa: BLE001: a report that fails leaves the landings as they are
        print(f"research queue: the landings {start[:12]}..{tip[:12]} weren't posted to the merge thread: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False


#: set where the verified queue runs live with the verity-drive App's token (`pods/nebius/vqueue/loop.sh`'s live mode, as on
#: the control pod): GitHub's rulesets "main is append-only" and "only drive moves main and next" let only that App move `main`
DRIVE_ENV = "RESEARCH_DRIVE"
DRIVE_ONLY = ("main", "next", CHECKED)


def on_drive() -> bool:
    return os.environ.get(DRIVE_ENV) == "1"


def off_drive(ref: str, verb: str) -> str:
    """Why a push of ``ref`` from here is refused: only the verity-drive App may move `main`, `checked` and `next`."""
    return (f"`{ref}` is moved only by the verity-drive App, and this isn't the live queue (no {DRIVE_ENV}=1): `{verb}` runs "
            "where the verified queue runs live")


