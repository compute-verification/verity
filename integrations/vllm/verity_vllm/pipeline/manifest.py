"""`verity-vllm manifest` -- the required-value manifest under the query of record (`Q_word_v1`), beside v1's `required_manifest`.

    build         --program <derive dir> [--workload wl.json] [--request-id r0] [--jobs N] --out manifest.json
    build-global  --fixture wl.json --programs-root <row dir> [--workload-digest WD | -] [--jobs N] --out manifest.json
                  TP row: --programs-root <rank r dir> --rank r [--workload-digest WD_r], once per rank, --ranks W
                  `-`: WD is read from stdin once the components are built, so they build while the workload Program composes
    compare       --v1 manifest.json --v2 manifest.json [--out report.json]

`build` / `build-global` take the same inputs as v1's (a derive_step output dir holds `instances.json.gz` + `result.json` [+ `artifact.json`];
a row dir holds one such dir per request shape, `*LP<prompt_len>_T<max_tokens-1>`), and write a manifest of v1's schema whose identities
are the bridge's (`required.request_manifest` / `compose.compose_global`; a TP row's rank manifests merged by `merge_ranks`).  `--no-return-slots`
builds the pure-dataflow population (no RETURN-SLOT policy) for the record.  The header names the Program's partition
(`verity/partition/v1`, Q_word v1 {16, 32}: `required.partition_of`) and the Build fails by name on any violation of it (`--word-check`,
default 16/32: widths, recomputes, a Call boundary not committed); the check that passed is recorded in the header (`query.word_check`,
`word_record`; never in the identities, so the rows digest is the same) and the Commit refuses a Q_word manifest without it
(`commit.partition.word_check_refusal`).  `--query Q_module_body_v1` (with `--word-check off`) is for A/B only.
Four tap policies are on by default, each with a `--no-...` for A/B only: `--norm-scales` (`query.norm_scales`), `--guarded-max`
(`query.guarded_max`: the query header states it, and every attention hidden-stream identity gains the MS class), `--router-softmax` and
`--vocab-range` (`query.router_softmax`, `query.vocab_range`); the scale-products policy (`query.scale_products`) applies to every manifest
and adds nothing to a Program without a shared-scale FP8 GEMM.  `--cross-call-check` (opt-in) fails the Build when a value is computed in
two Calls (`query.cross_call`).  `compare` classifies every identity difference (`compare.summarize`).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from verity_vllm.config import option
from verity_vllm.query.compare import compare, summarize
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.compose import compose_global, merge_ranks
from verity_vllm.query.required import QUERIES, QUERY_OF_RECORD, Q_WORD_ID, request_manifest
from verity_vllm.query import cross_call
from verity_vllm.query import guarded_max as GM
from verity_vllm.query import norm_scales as NS
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import scale_products as SP
from verity_vllm.query import vocab_range as VR
from verity_vllm.query import word as word_query

__all__ = ["TAPS", "DEFAULT_TAPS", "taps_of", "word_record", "cpus", "prefetch_word_rules", "build_components", "build", "build_global",
           "build_global_ranks", "cross_call_check", "main"]

#: `--jobs`'s variable
JOBS_ENV = "VERITY_MANIFEST_JOBS"

#: the tap policies whose identities are Call-interior values, by name (`request_manifest(taps=...)`)
TAPS = {RS.FAMILY: RS.POLICY, VR.FAMILY: VR.POLICY}
#: the tap policies every manifest is built under (each adds nothing to a Program without its Calls)
DEFAULT_TAPS = (SP.POLICY,)


def taps_of(router_softmax: bool = False, vocab_range: bool = False) -> tuple[str, ...]:
    """The tap policy names a manifest is built under, in `TAPS` order."""
    return tuple(name for name, on in ((RS.FAMILY, router_softmax), (VR.FAMILY, vocab_range)) if on)


def _load_json(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


@dataclass(frozen=True)
class _Committed:
    """The committed values of one request Program under the query of record, as `word.check_query` reads them."""
    required: Any
    literals: frozenset[int]
    program_path: Any


def word_check(P: Any, corr: Correspondence, required: Any, word: str, *, strict: bool = True, acquired: dict[str, int] | None = None,
               max_gates: str | None = None, log=None) -> dict[str, Any]:
    """`Q_word_v1{X,W}` (`word`: "16/32[+extras]") over `P`, committed = the manifest's required values and the interior words a tap
    acquires (`acquired`, `word.check_calls`; `max_gates`, the run's gate limits: `word.gate_limits`); `strict` raises
    `word.QueryRuleViolation` naming every violation.  The manifest is not touched."""
    from verity_vllm.query.module_body import literal_calls
    X, W, extras = word_query.parse(word)
    r = word_query.check_query(P, _Committed(required, literal_calls(P), corr.module_of), X, W, extras, strict=strict, acquired=acquired,
                                max_gates=max_gates)
    if log:
        log(f"  {r['query']}: {r['calls']} calls -> {r['units']} units over {r['gates']} gates, {r['committed_interior_words']} committed interior words"
            + (f" ({r['acquired_interior_words']} acquired by a tap)" if acquired is not None else ""))
    return r


def word_record(r: dict[str, Any], *, strict: bool, max_gates: str | None = None) -> dict[str, Any]:
    """The header's record of a word check (`query.word_check`): its query, whether it was strict and passed, its totals and the gate
    limits it ran under; the per-Definition groups stay in the Build's manifest.log."""
    return {"query": r["query"], "strict": bool(strict), "ok": bool(r["ok"]), "violations": len(r["violations"]),
            **{k: r[k] for k in ("calls", "units", "gates", "committed_interior_words", "acquired_interior_words") if k in r},
            "max_gates": word_query.gate_limits(max_gates)}


def recorded_program_digest(program_dir: str, res: dict[str, Any]) -> str | None:
    """The Program's digest (`verity.ml.circuits.codec.program_digest`): the Build's record (`result.json` program.digest), else
    recomputed from its `descriptor.json.gz`; None when the Build has neither.  A Build recorded before the SHA-512 re-baseline
    (ad58900df) names its Program by SHA-256 under `digest` and by today's digest under `sha512`; its `digest` is not read."""
    prog = res.get("program") or {}
    s = prog.get("sha512") or (prog.get("digest") if re.fullmatch(r"[0-9a-f]{128}", str(prog.get("digest"))) else None)
    if s:
        return s
    p = os.path.join(program_dir, "descriptor.json.gz")
    if not os.path.exists(p):
        return None
    import gzip
    from verity.ml.circuits.codec import program_digest
    with gzip.open(p, "rt") as f:
        return program_digest(json.load(f))


def cross_call_check(program_dir: str, *, strict: bool = True, log=None) -> dict[str, Any]:
    """Cross-Call recomputes (`cross_call.check_program`) over the Build's instance sequence; `strict` raises
    `cross_call.RecomputeViolation` naming every one.  The manifest is not touched."""
    return cross_call.check_program(os.path.join(program_dir, "instances.json.gz"), strict=strict, log=log)


def build(program_dir: str, *, workload: dict[str, Any] | None = None, request_id: str | None = None, rank: int = 0, ranks: int = 1,
          profile: str | None = None, return_slots: bool = True, word: str | None = None, norm_scales: bool = False, guarded_max: bool = False,
          taps: tuple[str, ...] = (), cross_call: bool = False, query: str = QUERY_OF_RECORD, word_max_gates: str | None = None,
          jobs: int = 1, log=None) -> dict[str, Any]:
    """The manifest of one request Program; `jobs` > 1 cuts the word check's Definitions in that many processes first."""
    prefetch_word_rules([program_dir], word, word_max_gates=word_max_gates, jobs=jobs, log=log)
    t0 = time.time()
    P = from_instances(os.path.join(program_dir, "instances.json.gz"))
    res = _load_json(os.path.join(program_dir, "result.json"))
    art_p = os.path.join(program_dir, "artifact.json")
    art = _load_json(art_p) if os.path.exists(art_p) else None
    corr = Correspondence.of(P, program_dir, implementation_paths=Correspondence.implementation_paths_of(res))   # the record when the Build carries it, else v1 annotations
    r = request_manifest(P, corr, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir, rank=rank, ranks=ranks,
                         profile=profile, return_slots=return_slots, norm_scales=NS.POLICY if norm_scales else None,
                         taps=tuple(TAPS[t] for t in taps) + DEFAULT_TAPS, query=query,
                         program_digest=recorded_program_digest(program_dir, res) if query == Q_WORD_ID else None)
    if word and word != "off":
        sp = SP.policy_of(r.manifest)
        acquired = {**(NS.TAP_WORDS if norm_scales else {}), **{f: w for t in taps for f, w in TAPS[t].tap_words.items()}, **(SP.TAP_WORDS if sp else {})}
        wc = word_check(P, corr, r.required, word, acquired=acquired if (norm_scales or taps or sp) else None, max_gates=word_max_gates, log=log)
        r.manifest["query"]["word_check"] = word_record(wc, strict=True, max_gates=word_max_gates)
    if guarded_max:
        r.manifest["query"] = GM.header(r.manifest["query"], GM.attention_words(P), GM.max_scaled_words(P))
        r.manifest.update(GM.with_ms(r.manifest))
    if cross_call:
        cross_call_check(program_dir, log=log)
    if log:
        pop = r.manifest["populations"]
        log(f"  {os.path.basename(program_dir)}: {len(P.calls)} calls -> {pop['identities']} identities ({pop['steps']} steps, {pop['units']} units, "
            f"{pop['boundary_values']} boundary values) complete={r.manifest['complete']} correspondence={corr.source} in {time.time() - t0:.1f}s")
    return r.manifest


def cpus() -> int:
    """The CPUs this process may use: its affinity and its cgroup's quota (a pod's `os.cpu_count()` is the host's)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def _headroom() -> int | None:
    """The bytes this process's memory limit leaves (the cgroup's limit less its usage, v2 then v1), else MemAvailable; None when
    neither is readable."""
    for lim_f, cur_f in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(lim_f) as a, open(cur_f) as b:
                lim, cur = a.read().strip(), int(b.read().strip())
        except (OSError, ValueError):
            continue
        if lim.isdigit() and int(lim) < (1 << 60):
            return max(0, int(lim) - cur)
    try:
        with open("/proc/meminfo") as f:
            return next(int(ln.split()[1]) * 1024 for ln in f if ln.split(":")[0] == "MemAvailable")
    except (OSError, StopIteration, ValueError, IndexError):
        return None


#: a component build's peak memory, from its Call count (`spec_histogram`): the TP2 MoE rows' builds peak at 2.6 KB per Call (2.0 GB
#: at 659K Calls) to 3.0 KB (9.0 GB at 2.99M), over a 0.3 GB interpreter; with headroom
PEAK_PER_CALL = 4 << 10
PEAK_BASE = 512 << 20
#: the share of the headroom the component builds may take together
HEADROOM_SHARE = 0.8


def _peak(program_dir: str) -> int | None:
    from verity_vllm.pipeline.global_program import component_spec_histogram
    h = component_spec_histogram(program_dir)
    return PEAK_BASE + PEAK_PER_CALL * sum(int(v) for v in h.values()) if h else None


def prefetch_word_rules(program_dirs: list[str], word: str | None, *, word_max_gates: str | None = None, jobs: int = 1, log=None) -> None:
    """The word check's unit rules for every Definition the Builds under `program_dirs` call (`spec_histogram`), `jobs` processes at a
    time, before their manifests are built (`word.prefetch`); nothing for a Build without the histogram, which checks as it did."""
    if not word or word == "off" or jobs <= 1:
        return
    from verity_vllm.pipeline.global_program import component_spec_histogram
    X, W, extras = word_query.parse(word)
    t0 = time.time()
    specs = [s for d in program_dirs for s in (component_spec_histogram(d) or {})]
    n = word_query.prefetch(specs, X, W, extras, max_gates=word_max_gates, jobs=jobs)
    if log:
        log(f"  word check: {n} unit rules of {len(set(specs))} Definitions cut by {jobs} processes in {time.time() - t0:.1f}s")


def _send(conn: Any, program_dir: str, kw: dict[str, Any]) -> None:
    try:
        out: tuple[str, Any] = ("ok", build(program_dir, **kw))
    except BaseException as e:  # noqa: BLE001  -- re-raised by the parent, in declaration order
        out = ("error", e)
    try:
        conn.send(out)
    except Exception as e:  # noqa: BLE001  -- an exception that does not pickle
        conn.send(("error", RuntimeError(f"{os.path.basename(program_dir)}: {type(out[1]).__name__}: {out[1]}")) if out[0] == "error"
                  else ("error", RuntimeError(f"{os.path.basename(program_dir)}: the manifest did not pickle: {e}")))
    conn.close()


def build_components(tasks: list[tuple[str, dict[str, Any]]], *, jobs: int = 1, after: Callable[[int, dict[str, Any]], None] | None = None,
                     budget_gb: float | None = None, log=None) -> list[dict[str, Any]]:
    """`build(program_dir, **kw)` of every `(program_dir, kw)`, each followed by `after(i, manifest)`, in order.  With `jobs` > 1 (and
    `fork`, and a readable memory headroom) each build runs in a process of its own, largest first, while their estimated peaks
    (`_peak`) fit `HEADROOM_SHARE` of the headroom; a Build without a histogram runs alone.  A build is a function of its Build dir, so
    the manifests are those one process builds, and the first build or `after` to fail in `tasks` order raises, as one after another."""
    import multiprocessing
    after = after or (lambda _i, _m: None)
    if jobs <= 1 or len(tasks) < 2 or "fork" not in multiprocessing.get_all_start_methods():
        return _build_in_turn(tasks, after)
    own = _own_budget(budget_gb)
    if own is not None:
        return _build_forked(tasks, jobs, after, own, log)
    t0 = time.time()
    with _pool_lock():
        if log and time.time() - t0 >= 1:
            log(f"  components: waited {time.time() - t0:.1f}s for the machine's component pool lock")
        head = _headroom()
        if head is None:
            return _build_in_turn(tasks, after)
        return _build_forked(tasks, jobs, after, head, log)


CGROUP_MEMORY_MAX = "/sys/fs/cgroup/memory.max"
MEMINFO = "/proc/meminfo"


def _own_budget(budget_gb: float | None = None) -> int | None:
    """The memory this process's builds may budget without the machine's pool lock: `budget_gb` (capped by the headroom), else the
    headroom of a cgroup whose `memory.max` is below the host's RAM; None when the headroom is the host's, shared."""
    head = _headroom()
    if budget_gb:
        b = int(budget_gb * 2**30)
        return b if head is None else min(b, head)
    try:
        with open(CGROUP_MEMORY_MAX) as f:
            lim = f.read().strip()
        with open(MEMINFO) as f:
            total = next(int(ln.split()[1]) * 1024 for ln in f if ln.split(":")[0] == "MemTotal")
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    return head if lim.isdigit() and int(lim) < total else None


def _build_in_turn(tasks: list[tuple[str, dict[str, Any]]], after: Callable[[int, dict[str, Any]], None]) -> list[dict[str, Any]]:
    out = []
    for i, (d, kw) in enumerate(tasks):
        out.append(build(d, **kw))
        after(i, out[-1])
    return out


POOL_LOCK = "verity-manifest-build-components.lock"


@contextmanager
def _pool_lock(path: str | None = None):
    """One process pool of component builds per machine at a time, whichever user runs them (two budgeting one headroom would take it
    twice: the vLLM suite's xdist workers run both stored TP2 rows at once).  `path`: the lock file, default `POOL_LOCK` in the temp dir."""
    import fcntl
    import tempfile
    fd = _open_shared_lock(path or os.path.join(tempfile.gettempdir(), POOL_LOCK))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _open_shared_lock(path: str, attempts: int = 4) -> int:
    """A read-only descriptor of the lock file `path` that every user of the machine opens.  An existing file is opened without O_CREAT:
    under `fs.protected_regular` an O_CREAT open of another user's file in a sticky world-writable directory (/tmp) is refused whatever
    its mode, and flock needs no write access.  A missing one is created 0o666 whatever the umask; losing that race to another creator
    opens theirs.  A file this user cannot read raises, naming its owner: a lock of its own would not exclude the other users' builds."""
    for _ in range(attempts):
        try:
            return os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            pass
        except PermissionError as e:
            raise PermissionError(e.errno, f"the machine's component pool lock {path} {_owner(path)} is not readable by uid {os.getuid()}: "
                                           "every user's builds share it, so its owner must make it readable (chmod a+r) or remove it", path) from e
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        try:
            os.fchmod(fd, 0o666)
        except BaseException:
            os.close(fd)
            raise
        return fd
    raise RuntimeError(f"the machine's component pool lock {path} kept appearing and vanishing over {attempts} attempts to open it")


def _owner(path: str) -> str:
    import pwd
    try:
        st = os.stat(path)
    except OSError as e:
        return f"(stat failed: {e.strerror})"
    try:
        name = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        name = "?"
    return f"(owner {name}, uid {st.st_uid}, mode {st.st_mode & 0o7777:o})"


def _build_forked(tasks: list[tuple[str, dict[str, Any]]], jobs: int, after: Callable[[int, dict[str, Any]], None], head: int,
                  log) -> list[dict[str, Any]]:
    import multiprocessing
    from multiprocessing.connection import wait
    ctx = multiprocessing.get_context("fork")
    budget = int(head * HEADROOM_SHARE)
    need = [e if (e := _peak(d)) is not None else budget + 1 for d, _kw in tasks]
    pending = sorted(range(len(tasks)), key=lambda i: (-need[i], i))
    results: dict[int, tuple[str, Any]] = {}
    running: dict[Any, tuple[int, Any]] = {}
    used, first_bad = 0, len(tasks)
    if log:
        log(f"  components: {len(tasks)} builds, {jobs} processes, memory budget {budget / 2**30:.1f} GiB")
    while pending or running:
        while pending and len(running) < jobs:
            i = next((i for i in pending if used + need[i] <= budget), None if running else pending[0])
            if i is None:
                break
            pending.remove(i)
            recv, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_send, args=(send, *tasks[i]), daemon=True)
            p.start()
            send.close()
            running[recv] = (i, p)
            used += need[i]
        for conn in wait(list(running)):
            i, p = running.pop(conn)
            try:
                results[i] = conn.recv()
            except EOFError:
                results[i] = ("error", RuntimeError(f"{os.path.basename(tasks[i][0])}: the build process died"))
            conn.close()
            p.join()
            if p.exitcode not in (0, None) and results[i][0] == "ok":
                results[i] = ("error", RuntimeError(f"{os.path.basename(tasks[i][0])}: the build process exited {p.exitcode}"))
            used -= need[i]
            if results[i][0] == "error" and i < first_bad:
                first_bad = i
                pending = [j for j in pending if j < i]
    for i in range(len(tasks)):
        kind, value = results[i]
        if kind == "error":
            raise value
        after(i, value)
    return [results[i][1] for i in range(len(tasks))]


def _program_dirs_for(fx: dict[str, Any], programs_root: str) -> dict[str, str]:
    """v1's lookup (`required_manifest._program_dirs_for`): `*LP<prompt_len>_T<max_tokens-1>` under the root, else a `build_request*`
    whose result.json declares exactly (LP, T)."""
    progs: dict[str, str] = {}
    for r in fx["requests"]:
        lp, t = int(r["prompt_len"]), int(r["max_tokens"]) - 1
        hits = sorted(glob.glob(os.path.join(programs_root, f"*LP{lp}_T{t}")))
        if not hits:
            for cand in sorted(glob.glob(os.path.join(programs_root, "build_request*"))):
                rj = os.path.join(cand, "result.json")
                if os.path.isdir(cand) and os.path.exists(rj):
                    try:
                        res = _load_json(rj)
                    except Exception:  # noqa: BLE001
                        continue
                    if res.get("ok") and int(res.get("LP", -1)) == lp and int(res.get("T", -1)) == t:
                        hits = [cand]
                        break
        if not hits:
            raise SystemExit(f"no request Program for {r['request_id']} (LP={lp} T={t}) under {programs_root}")
        progs[str(r["request_id"])] = hits[0]
    return progs


def _declared(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """The declared requests in (arrive_step, index) order, each labelled `r<index>`."""
    declared = [dict(r, index=int(r.get("index", i))) for i, r in enumerate(fixture["requests"])]
    if len({r["index"] for r in declared}) != len(declared):
        raise ValueError("workload declaration: duplicate request index")
    return [dict(r, workload_request_id=r.get("request_id"), request_id=f"r{r['index']}")
            for r in sorted(declared, key=lambda r: (int(r.get("arrive_step", 0)), r["index"]))]


def stdin_digest() -> str:
    """`--workload-digest -`: the next line of stdin.  An empty line or none (the workload Program was not composed) fails the manifest."""
    wd = sys.stdin.readline().strip()
    if not wd:
        raise SystemExit("build-global --workload-digest -: no workload digest on stdin (the workload Program was not composed)")
    return wd


def _global(fixture: dict[str, Any], roots: list[str], ranks: list[int], world: int, wds: list[str | None], *, guarded_max: bool, jobs: int,
            log, budget_gb: float | None = None, **kw: Any) -> list[dict[str, Any]]:
    """The global manifest of each rank's Program set (`roots[i]` is rank `ranks[i]`'s): every component built by `build_components`,
    checked and composed in declaration order."""
    reqs = _declared(fixture)
    wl = [{"sweep": {"arrive_steps": [int(r.get("arrive_step", 0))], "row_id": fixture.get("fixture"), "batch": len(reqs)}} for r in reqs]
    tasks = []
    for root, rank in zip(roots, ranks):
        dirs = _program_dirs_for(fixture, root)
        tasks += [(dirs[str(r["workload_request_id"])], dict(kw, workload=w, request_id=r["request_id"], rank=rank, ranks=world,
                                                             guarded_max=guarded_max, log=log)) for r, w in zip(reqs, wl)]
    prefetch_word_rules([d for d, _kw in tasks], kw.get("word"), word_max_gates=kw.get("word_max_gates"), jobs=jobs, log=log)
    parts: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []

    def after(i: int, man: dict[str, Any]) -> None:
        k, r = divmod(i, len(reqs))
        rid = reqs[r]["request_id"]
        if len(man["steps"]) != int(reqs[r]["max_tokens"]):
            raise ValueError(f"request {rid}: Program has {len(man['steps'])} steps, the declaration caps {reqs[r]['max_tokens']} tokens: wrong component")
        parts[rid] = man
        if r == len(reqs) - 1:
            g = compose_global(fixture, parts, reqs, workload_digest=stdin_digest() if wds[k] == "-" else wds[k], rank=ranks[k], ranks=world)
            if guarded_max:
                g = GM.with_ms(dict(g, query=GM.merged(g["query"], [p["query"] for p in parts.values()])))
            out.append(g)
            parts.clear()

    build_components(tasks, jobs=jobs, after=after, budget_gb=budget_gb, log=log)
    return out


def build_global(fixture: dict[str, Any], programs_root: str, *, workload_digest: str | None = None, rank: int = 0, ranks: int = 1,
                 profile: str | None = None, return_slots: bool = True, word: str | None = None, norm_scales: bool = False,
                 guarded_max: bool = False, taps: tuple[str, ...] = (), cross_call: bool = False, query: str = QUERY_OF_RECORD,
                 word_max_gates: str | None = None, jobs: int = 1,
                 log=None) -> dict[str, Any]:
    return _global(fixture, [programs_root], [rank], ranks, [workload_digest], profile=profile, return_slots=return_slots, word=word,
                   norm_scales=norm_scales, guarded_max=guarded_max, taps=taps, cross_call=cross_call, query=query,
                   word_max_gates=word_max_gates, jobs=jobs, log=log)[0]


def build_global_ranks(fixture: dict[str, Any], roots: list[str], ranks: list[int], world: int, *, workload_digests: list[str | None] | None = None,
                       profile: str | None = None, return_slots: bool = True, word: str | None = None, norm_scales: bool = False,
                       guarded_max: bool = False, taps: tuple[str, ...] = (), cross_call: bool = False, query: str = QUERY_OF_RECORD,
                       word_max_gates: str | None = None, jobs: int = 1, budget_gb: float | None = None,
                       log=None) -> dict[str, Any]:
    """One global manifest per rank Program set (`roots[i]` is rank `ranks[i]`'s), merged by `merge_ranks` when the row is TP; `jobs`
    processes build the components (`build_components`)."""
    wds = list(workload_digests or [None] * len(roots))
    if len(ranks) != len(roots) or len(wds) != len(roots):
        raise ValueError(f"build-global: {len(roots)} --programs-root, {len(ranks)} --rank, {len(wds)} --workload-digest: give one per rank")
    if world > 1 and sorted(ranks) != list(range(world)):
        raise ValueError(f"build-global: TP world {world} needs one --programs-root per rank 0..{world - 1} (got ranks {ranks})")
    parts = _global(fixture, roots, ranks, world, wds, profile=profile, return_slots=return_slots, word=word, norm_scales=norm_scales,
                    guarded_max=guarded_max, taps=taps, cross_call=cross_call, query=query, word_max_gates=word_max_gates, jobs=jobs,
                    budget_gb=budget_gb, log=log)
    return merge_ranks(parts) if world > 1 else parts[0]


def check_correspondence(program_dir: str, *, workload: dict[str, Any] | None = None, request_id: str | None = None, manifest: bool = True, log=None) -> dict[str, Any]:
    """`Q_module_body_v1` under the record vs under the v1-annotation fallback on ONE Build dir that carries the record: the owning
    module per Call, the Call names the bridge reads, the partition (unit ids), the boundary Values and -- `manifest` -- the bridge's
    manifest digest, each equal or listed.  A Build without the record has nothing to check against (`applicable: False`)."""
    from verity_vllm.query.module_body import Q_module_body_v1, literal_calls, module_body_boundary
    t0 = time.time()
    P = from_instances(os.path.join(program_dir, "instances.json.gz"))
    res = _load_json(os.path.join(program_dir, "result.json"))
    impl = Correspondence.implementation_paths_of(res)
    canon = Correspondence.of(P, program_dir, implementation_paths=impl)
    out: dict[str, Any] = {"schema": "vllm-v2/query-correspondence-check/v1", "program_dir": program_dir, "wrapper": res.get("wrapper"), "calls": len(P.calls),
                           "program_digest": (res.get("program") or {}).get("digest"), "correspondence_digest": canon.digest, "source": canon.source, "problem": canon.problem}
    if canon.source != "correspondence":
        out.update(applicable=False, why="the Build carries no runtime-correspondence/v1 record: only the fallback reads it")
        return out
    fallback = Correspondence(P, implementation_paths=impl)
    module_diffs = [{"call": c.id, "record": canon.module_of(c.id), "v1_annotations": fallback.module_of(c.id), "name": c.name} for c in P.calls if canon.module_of(c.id) != fallback.module_of(c.id)]
    name_diffs = [{"call": c.id, "record": str(canon.name(c.id)), "v1_annotations": str(fallback.name(c.id)), "name": c.name} for c in P.calls if canon.name(c.id) != fallback.name(c.id)]
    Qa, Qb = Q_module_body_v1(P, canon), Q_module_body_v1(P, fallback)
    units_a = {u.id: u.calls for u in Qa(P)}
    units_b = {u.id: u.calls for u in Qb(P)}
    literals = literal_calls(P)
    sa, sb = set(module_body_boundary(P, canon, literals).values), set(module_body_boundary(P, fallback, literals).values)
    out.update(applicable=True, module_paths={"record": len(canon.modules), "v1_annotations": len(fallback.modules)}, implementation_prefix={"record": canon.prefix, "v1_annotations": fallback.prefix},
               module_diffs_n=len(module_diffs), module_diffs=module_diffs[:50], name_diffs_n=len(name_diffs), name_diffs=name_diffs[:50],
               partition_equal=(units_a == units_b), units={"record": len(units_a), "v1_annotations": len(units_b)},
               boundary_values={"record": len(sa), "v1_annotations": len(sb), "equal": sa == sb, "only_record": len(sa - sb), "only_v1_annotations": len(sb - sa)})
    if manifest:
        try:
            art_p = os.path.join(program_dir, "artifact.json")
            art = _load_json(art_p) if os.path.exists(art_p) else None
            tp = res.get("tp") if isinstance(res.get("tp"), dict) else {}
            rk = dict(rank=int(tp.get("rank") or 0), ranks=int(tp.get("world") or 1))          # a rank Program's manifest is its rank's
            ma = request_manifest(P, canon, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir, **rk).manifest
            mb = request_manifest(P, fallback, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir, **rk).manifest
            out["manifest"] = {"record": {"digest": ma["manifest_digest"], "identities": ma["populations"]["identities"], "complete": ma["complete"], "correspondence": ma["correspondence"]},
                               "v1_annotations": {"digest": mb["manifest_digest"], "identities": mb["populations"]["identities"], "complete": mb["complete"], "correspondence": mb["correspondence"]},
                               "digest_equal": ma["manifest_digest"] == mb["manifest_digest"], "wrapper_prefix": [ma["wrapper_prefix"], mb["wrapper_prefix"]]}
        except Exception as e:  # noqa: BLE001  -- a step wrapper is not a request Program; the partition comparison above still stands
            out["manifest"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    out["identical"] = not module_diffs and not name_diffs and out["partition_equal"] and out["boundary_values"]["equal"] and (out.get("manifest", {}).get("digest_equal", True) is True)
    out["seconds"] = round(time.time() - t0, 1)
    if log:
        log(f"  {os.path.basename(program_dir)} ({res.get('wrapper')}): {len(P.calls)} calls, module diffs {len(module_diffs)}, name diffs {len(name_diffs)}, partition_equal {out['partition_equal']}, "
            f"boundary equal {out['boundary_values']['equal']}, manifest {out.get('manifest', {}).get('digest_equal', out.get('manifest'))} in {out['seconds']}s")
    return out


def _write(man: dict[str, Any], out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(man, f, separators=(",", ":"))


ROUTER_SOFTMAX_HELP = ("on by default, --no-router-softmax for A/B only: the router-softmax policy (verity_vllm.query.router_softmax) -- "
                       "one router_softmax identity per (step, MoE layer) of the selection-order router, `moe/L<k>/router_softmax`; the "
                       "Commit then needs --router-tap-so")
VOCAB_RANGE_HELP = ("on by default, --no-vocab-range for A/B only: the vocabulary-range policy (verity_vllm.query.vocab_range) -- one "
                    "vocab_range identity per (step, TP embedding site), `<embedding module>/vocab_range`; the Commit then needs --vocab-range")
NORM_SCALES_HELP = ("on by default, --no-norm-scales for A/B only: the norm-scale policy (verity_vllm.query.norm_scales) -- one norm_scales "
                    "identity per norm Call row, `<norm module>/norm_scale`; the Commit then needs --norm-tap-so")
WORD_CHECK_HELP = ("the partition of record's check: Q_word_v1{X,W} (\"16/32[+extras]\", the default) over the Program -- widths, recomputes, "
                   "every Call boundary committed; fails the Build naming every violation.  \"off\" skips it (A/B only)")
QUERY_HELP = f"the query the manifest is built under: {QUERY_OF_RECORD} (the record) or Q_module_body_v1 (A/B only; pair it with --word-check off)"
CROSS_CALL_HELP = "check that no value is computed in two Calls (verity_vllm.query.cross_call); fail the Build naming every recompute"
WORD_MAX_GATES_HELP = ("--word-check's gate limits (verity_vllm.query.word.gate_limits): a Definition over its limit is a too-large violation, "
                       "not cut.  N for every Definition or FAMILY=N for one, comma-separated (default 24000000; about 0.6 KB of host memory "
                       "per gate cut); the query id is the same under any")
GUARDED_MAX_HELP = ("on by default, --no-guarded-max for A/B only: the guarded-max policy (verity_vllm.query.guarded_max) -- the attention "
                    "stream's ROW word 3 carries GuardNegInfZero(m) after a row's first key block and its MS class every block's max * "
                    "scale; the Commit then needs the guarded-max tap build")
UNIT_RULE_CACHE_HELP = ("a directory the word check's unit rules are shared through, across processes and runs (word.use_rule_cache); "
                        "a rule read there is the one the check computes")
JOBS_HELP = ("processes for the word check's cuts (word.prefetch) and, in build-global, the component builds (build_components, within "
             "the memory headroom); the manifest is the same under any.  0 (the default): the CPUs this process may use; 1: one process")


@dataclass(frozen=True, kw_only=True)
class BuildOptions:
    program: str = option("--program")
    workload: str | None = option("--workload", default=None)
    request_id: str | None = option("--request-id", default=None)
    profile: str | None = option("--profile", default=None)
    rank: int = option("--rank", type=int, default=0)
    ranks: int = option("--ranks", type=int, default=1)
    no_return_slots: bool = option("--no-return-slots", default=False, action="store_true")
    word_check: str = option("--word-check", env="VERITY_WORD_CHECK", default="16/32", help=WORD_CHECK_HELP)
    word_max_gates: str | None = option("--word-max-gates", env=word_query.MAX_GATES_ENV, default=None, help=WORD_MAX_GATES_HELP)
    unit_rule_cache: str | None = option("--unit-rule-cache", env=word_query.RULE_CACHE_ENV, default=None, help=UNIT_RULE_CACHE_HELP)
    query: str = option("--query", default=QUERY_OF_RECORD, choices=list(QUERIES), help=QUERY_HELP)
    norm_scales: bool = option("--norm-scales", default=True, action="boolean_optional", help=NORM_SCALES_HELP)
    guarded_max: bool = option("--guarded-max", default=True, action="boolean_optional", help=GUARDED_MAX_HELP)
    router_softmax: bool = option("--router-softmax", default=True, action="boolean_optional", help=ROUTER_SOFTMAX_HELP)
    vocab_range: bool = option("--vocab-range", default=True, action="boolean_optional", help=VOCAB_RANGE_HELP)
    cross_call_check: bool = option("--cross-call-check", default=False, action="store_true", help=CROSS_CALL_HELP)
    jobs: int = option("--jobs", type=int, env=JOBS_ENV, default=0, help=JOBS_HELP)
    out: str = option("--out")
    cmd: str = field(default="build", init=False)


@dataclass(frozen=True, kw_only=True)
class BuildGlobalOptions:
    fixture: str = option("--fixture")
    programs_root: list[str] = option("--programs-root", action="append",
                                      help="TP row: repeat per rank, with --rank r at the same position")
    workload_digest: list[str] | None = option("--workload-digest", action="append", default=None,
                                               help="TP row: repeat per rank; `-`: read from stdin, one line per such rank in --programs-root "
                                                    "order, once that rank's components are built")
    profile: str | None = option("--profile", default=None)
    rank: list[int] | None = option("--rank", type=int, action="append", default=None,
                                    help="the TP rank of the --programs-root at the same position (default 0)")
    ranks: int = option("--ranks", type=int, default=1)
    no_return_slots: bool = option("--no-return-slots", default=False, action="store_true")
    word_check: str = option("--word-check", env="VERITY_WORD_CHECK", default="16/32", help=WORD_CHECK_HELP)
    word_max_gates: str | None = option("--word-max-gates", env=word_query.MAX_GATES_ENV, default=None, help=WORD_MAX_GATES_HELP)
    unit_rule_cache: str | None = option("--unit-rule-cache", env=word_query.RULE_CACHE_ENV, default=None, help=UNIT_RULE_CACHE_HELP)
    query: str = option("--query", default=QUERY_OF_RECORD, choices=list(QUERIES), help=QUERY_HELP)
    norm_scales: bool = option("--norm-scales", default=True, action="boolean_optional", help=NORM_SCALES_HELP)
    guarded_max: bool = option("--guarded-max", default=True, action="boolean_optional", help=GUARDED_MAX_HELP)
    router_softmax: bool = option("--router-softmax", default=True, action="boolean_optional", help=ROUTER_SOFTMAX_HELP)
    vocab_range: bool = option("--vocab-range", default=True, action="boolean_optional", help=VOCAB_RANGE_HELP)
    cross_call_check: bool = option("--cross-call-check", default=False, action="store_true", help=CROSS_CALL_HELP)
    jobs: int = option("--jobs", type=int, env=JOBS_ENV, default=0, help=JOBS_HELP)
    build_ram_budget_gb: float | None = option("--build-ram-budget-gb", env="BUILD_RAM_BUDGET_GB", type=float, default=None,
                                               help="the component builds' own memory budget (capped by the headroom): set, they size "
                                                    "their pool from it without the machine's component pool lock")
    out: str = option("--out")
    cmd: str = field(default="build-global", init=False)


@dataclass(frozen=True, kw_only=True)
class CompareOptions:
    v1: str = option("--v1")
    v2: str = option("--v2")
    ignore_program_digest: bool = option("--ignore-program-digest", default=False, action="store_true")
    out: str | None = option("--out", default=None)
    cmd: str = field(default="compare", init=False)


@dataclass(frozen=True, kw_only=True)
class CheckCorrespondenceOptions:
    """Q_module_body_v1 under the runtime-correspondence/v1 record vs under the v1-annotation fallback on one Build dir"""

    program: str = option("--program")
    workload: str | None = option("--workload", default=None)
    request_id: str | None = option("--request-id", default=None)
    no_manifest: bool = option("--no-manifest", default=False, action="store_true")
    out: str | None = option("--out", default=None)
    cmd: str = field(default="check-correspondence", init=False)


def incomplete_rc(man: dict[str, Any]) -> int:
    """0 for a complete manifest; 4 for an incomplete one, named on stderr (the file is still written, for diagnosis): the Commit refuses
    an incomplete manifest, so the row's Build fails here instead of passing with it."""
    if man.get("complete") is not False:
        return 0
    um = man.get("unmodelled") or {}
    print(f"manifest INCOMPLETE: {len(um)} unmodelled key(s), first {sorted(um)[:3]}; tp_peer_binding n_unbound "
          f"{(man.get('tp_peer_binding') or {}).get('n_unbound')}", file=sys.stderr)
    return 4


COMMANDS = {"build": BuildOptions, "build-global": BuildGlobalOptions, "compare": CompareOptions, "check-correspondence": CheckCorrespondenceOptions}


def main(a: Any) -> int:
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    if a.cmd == "check-correspondence":
        wl = _load_json(a.workload) if a.workload else None
        r = check_correspondence(a.program, workload=wl, request_id=a.request_id, manifest=not a.no_manifest, log=log)
        if a.out:
            _write(r, a.out)
        print(json.dumps({kk: r.get(kk) for kk in ("wrapper", "calls", "source", "applicable", "module_diffs_n", "name_diffs_n", "partition_equal", "boundary_values", "identical", "correspondence_digest")}
                         | {"manifest_digest_equal": (r.get("manifest") or {}).get("digest_equal"), "manifest_digests": [(r.get("manifest") or {}).get(s, {}).get("digest") for s in ("record", "v1_annotations")]}))
        return 0 if (r.get("identical") or not r.get("applicable")) else 1
    if a.cmd in ("build", "build-global"):
        from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level cyclic-GC thresholds; content-neutral (VERITY_GC_THRESHOLDS)
        _gc_tune()
        word_query.use_rule_cache(a.unit_rule_cache)
    if a.cmd == "build":
        wl = _load_json(a.workload) if a.workload else None
        man = build(a.program, workload=wl, request_id=a.request_id, rank=a.rank, ranks=a.ranks, profile=a.profile, return_slots=not a.no_return_slots,
                    word=a.word_check, norm_scales=a.norm_scales, guarded_max=a.guarded_max, taps=taps_of(a.router_softmax, a.vocab_range),
                    cross_call=a.cross_call_check, query=a.query, word_max_gates=a.word_max_gates, jobs=a.jobs or cpus(), log=log)
        _write(man, a.out)
        print(json.dumps({"identities": man["populations"]["identities"], "manifest_digest": man["manifest_digest"], "complete": man["complete"], "out": a.out}))
        return incomplete_rc(man)
    if a.cmd == "build-global":
        fx = _load_json(a.fixture)
        man = build_global_ranks(fx, a.programs_root, a.rank or [0] * len(a.programs_root), a.ranks, workload_digests=a.workload_digest,
                                 profile=a.profile, return_slots=not a.no_return_slots, word=a.word_check, norm_scales=a.norm_scales,
                                 guarded_max=a.guarded_max, taps=taps_of(a.router_softmax, a.vocab_range),
                                 cross_call=a.cross_call_check, query=a.query, word_max_gates=a.word_max_gates, jobs=a.jobs or cpus(),
                                 budget_gb=a.build_ram_budget_gb, log=log)
        _write(man, a.out)
        print(json.dumps({"identities": man["populations"]["identities"], "manifest_digest": man["manifest_digest"], "program_digest": man["program_digest"],
                          "complete": man["complete"], "by_family": man["populations"]["by_family"],
                          "tp_peer_binding_n_unbound": (man.get("tp_peer_binding") or {}).get("n_unbound"), "out": a.out}))
        return incomplete_rc(man)
    if a.cmd == "compare":
        m1, m2 = _load_json(a.v1), _load_json(a.v2)
        cmp = compare(m1["identities"], m2["identities"], ignore_program_digest=a.ignore_program_digest)
        s = summarize(cmp)
        s["v1_manifest_digest"], s["v2_manifest_digest"] = m1.get("manifest_digest"), m2.get("manifest_digest")
        s["v1_program_digest"], s["v2_program_digest"] = m1.get("program_digest"), m2.get("program_digest")
        if a.out:
            _write(s, a.out)
        print(json.dumps({k: s[k] for k in ("v1_identities", "v2_identities", "common", "v1_rows", "v2_rows", "multiplicity_diff_n", "only_v1_n", "only_v2_n")}))
        for k in ("member_rename", "range", "only_v1", "only_v2"):
            for line in s[k]:
                print(f"  {k:14s} n={line['n']:6d} {line.get('op_path')}/{line.get('member')} spec={line.get('spec')} :: {line['reason'][:120]}")
        for m in s["multiplicity"][:10]:
            print(f"  multiplicity   v1_rows={m['v1_rows']} v2_rows={m['v2_rows']} {m['key'][:160]}")
        return 0
    return 2
