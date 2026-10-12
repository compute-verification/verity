"""Run custody on the remote (cloud migration step f): every file of a run dir -- stdout.log, stderr.log, launcher.log, job.json,
status.json, the outputs -- goes to the store's remote as one `run-record/v1` tree that the attempt names (`outputs.run_record`),
published on the machine that ran it with a minted key, so custody no longer depends on the launcher's disk or on it staying up.

    research run --on M --custody-r2 [--custody-ttl 6h] ...
        launcher (`mint_for_machine`): an action-scoped temporary key (`credentials.mint`; no DeleteObject / CopyObject) and a
        secret-free store.toml, staged under <root>/requests/<run-id>/custody/ (mode 600; never in request.json, remote.json or
        the workload's environment).  Minting needs the parent key here (the Cursor secrets, or ~/.config/verity/r2.env sourced):
        without it the launch is refused, never silently downgraded.
        runner (`runner_publish`, after the workload exits): publishes the attempt with its run record, pushes and verifies it
        (`preserved.check`, head mode; up to PUSH_TRIES), deletes the key, and writes <run>/.custody.
    research data custody RUN [--publish] [--runs-dir D] [--json]        verify a run's custody on the remote and (re)write .custody;
                                                                          --publish first publishes it with a run record and pushes,
                                                                          and leaves a failed verdict in .custody when nothing vouches
    research data custody --triage [--runs-dir D ...] [--verify] [--older-min N] [--json]
                                                                          every run under the runs roots by custody state (F2); with
                                                                          --older-min, the ended runs not on the remote after N minutes

The record uploads the run's own files, everything under out/ (the workload's $OUT), the files matching its `--declared-output`
globs, and undeclared files up to UNDECLARED_CAP_BYTES; every other file is listed (path, bytes, sha256) without bytes (`select`; how jobs declare outputs:
.agents/skills/writing-runs/SKILL.md).  The listing is the record's `meta.trimmed` when it fits in TRIMMED_INLINE_BYTES, else the
file TRIMMED_LISTING in the record's own tree, named by `meta.trimmed_listing` (`trimmed_of`).  A credential a run may be sent
(CREDENTIALS) is deleted from the run dir before its record is taken, whatever the run's tool and however it ended, so no record
holds one; a refused launch deletes it from the request dir (remote.py).

`.custody` (research/run-custody/v0.1) is written only when the attempt is PRESERVED on the remote, names a run record, and every
file now in the run dir is in that record with the same size and sha256 (hashed after `verified_from`, the clock read before
hashing).  A run dir has custody when its `.fetched` is valid (a verified copy on the launcher, `research fetch --all`) or its
`.custody` is: every file in the dir is listed and none changed at or after `verified_from` (`marker_problem`; the pod guard,
pods/sh/pod_guard.sh, applies the same rule without hashing).  Once the marker is written the runner writes nothing more into the
run dir, so launcher.log in the record is the whole of it.

A run's custody is a remote fact, so any machine can check it: `research data preserved <run id>` (head mode) and `research data
custody RUN` answer the same everywhere.  Waivers and never-ran verdicts for the triage are labels on the run id:
`research data label <run id> custody waived|never-ran --by <who>`; a never-ran holds only while the run dir shows no sign that
its workload started (`ran_evidence`), and a later `custody needed` withdraws either.
"""
from __future__ import annotations

import calendar
import contextlib
import fnmatch
import json
import os
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from ..env import DIRTY_CAPTURE
from ..telemetry.job import write_json_atomic
from ..telemetry.run import OUT_DIR
from .attempt import TERMINAL as TERMINAL_STATES
from .evict import EVICTED_FILE

MARKER = ".custody"
SCHEMA = "research/run-custody/v0.1"
KIND = "run-record/v1"
TRIMMED_LISTING = ".custody-trimmed.json"
EXCLUDED = frozenset({".fetched", ".fetched.tmp", MARKER, MARKER + ".tmp", "preserved.json", EVICTED_FILE,   # markers about the dir, not of it
                      TRIMMED_LISTING})
DIR_NAME = "custody"                          # <root>/requests/<run-id>/custody/ on the machine
CRED_FILE = "cred.json"
STORE_TOML = "store.toml"
DEFAULT_TTL = "6h"
ACTIONS = ("HeadObject", "GetObject", "GetBucketLocation", "ListObjectsV1", "ListObjectsV2", "ListMultipartUploads", "ListParts",
           "PutObject", "CreateMultipartUpload", "UploadPart", "CompleteMultipartUpload", "AbortMultipartUpload")
PUSH_TRIES = 3
PUSH_RETRY_S = 20.0
TRIAGE_STATES = ("fetched", "r2", "forced-unverified", "attempt-no-record", "attempt-unpreserved", "conflict", "no-custody", "running",
                 "never-ran", "waived")


# ---- the run dir ------------------------------------------------------------------------------------------------------------------------

def run_files(run_dir: Path) -> list[str]:
    """Every regular file under the run dir (relative posix paths, sorted), markers excluded, and the `.custody.tmp.<pid>` a publish
    killed between write_json_atomic's write and its rename leaves; symlinks are not followed (as `fetch --all`)."""
    d = Path(run_dir)
    out = []
    for base, dirs, files in os.walk(d):
        dirs.sort()
        for f in files:
            p = Path(base) / f
            rel = p.relative_to(d).as_posix()
            if rel in EXCLUDED or rel.startswith(MARKER + ".tmp.") or p.is_symlink() or not p.is_file():
                continue
            out.append(rel)
    return sorted(out)


def run_id_of(run_dir: Path) -> str:
    """The attempt id `attempt.publish` gives the run dir: job.json's run id, else the directory name."""
    job = _read(Path(run_dir) / "job.json") or {}
    return (job.get("run") or {}).get("id") or (job.get("ownership") or {}).get("job") or Path(run_dir).name


def _sha256(p: Path) -> str:
    import hashlib

    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


#: The runner's own files, always uploaded; everything else is kept only when it matches a declared output (`declared_globs`).
#: DIRTY_CAPTURE is a dirty source's bytes (env.py): never trimmed, so the record, not the machine's tree, is their durable copy.
RUN_FILES = frozenset({"stdout.log", "stderr.log", "launcher.log", "sampler.log", "job.json", "status.json", "events.jsonl", "failure.json",
                       "result.json", "launch.json", "remote.json", "refused.json", "slot.json", DIRTY_CAPTURE})
#: The runner's own files under stamped names (fnmatch on the run-dir-relative path): the stall watch's py-spy dumps (telemetry.run).
RUN_FILE_GLOBS = ("logs/stall-*.txt",)
#: The launcher's record of a remote run (remote.py): only on the launching machine, rewritten by every `status` and `fetch`, and never
#: in the record the run's machine takes, so custody of the launcher's copy neither checks it nor lets eviction remove it.
LAUNCHER_RECORD = "remote.json"
#: Undeclared files are uploaded (in path order) while their total stays within this; the rest are trimmed.  0 = never upload undeclared.
UNDECLARED_CAP_BYTES = 256 << 20
#: A trimmed listing bigger than this goes in the record's tree: a manifest's meta is capped at 256 KiB (model.META_MAX_BYTES), and a
#: `check` run lists thousands of files (r20261001-091202-7aa6: 3719 files, a 561 KB meta, so its passing attempt never published).
TRIMMED_INLINE_BYTES = 64 << 10
SKILL = ".agents/skills/writing-runs/SKILL.md"


def is_run_file(rel: str) -> bool:
    return rel in RUN_FILES or any(fnmatch.fnmatchcase(rel, g) for g in RUN_FILE_GLOBS)


def declared_globs(run_dir: Path) -> list[str]:
    """`out/*` (the workload's $OUT), always, then the run's own `--declared-output` globs."""
    job = _read(Path(run_dir) / "job.json") or {}
    return [f"{OUT_DIR}/*", *(str(g) for g in (job.get("execution") or {}).get("declared_outputs") or [])]


def select(run_dir: Path, cap: int | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """(files the run record uploads, [{path, bytes, sha256}] of the files it only lists): the run's own files, every file matching a
    declared glob (`declared_globs`; fnmatch on the path relative to the run dir, so `*` crosses `/`: `out/*` is all of out/, `*` is
    everything), and undeclared files up to `cap` bytes in total."""
    d = Path(run_dir)
    cap = UNDECLARED_CAP_BYTES if cap is None else cap
    globs = declared_globs(d)
    kept, trimmed, used = [], [], 0
    for rel in run_files(d):
        size = (d / rel).stat().st_size
        if is_run_file(rel) or any(fnmatch.fnmatchcase(rel, g) for g in globs):
            kept.append(rel)
        elif used + size <= cap:
            used += size
            kept.append(rel)
        else:
            trimmed.append({"path": rel, "bytes": size, "sha256": _sha256(d / rel)})
    return kept, trimmed


def trim_warning(trimmed: list[dict[str, Any]]) -> str | None:
    if not trimmed:
        return None
    return (f"custody trimmed {len(trimmed)} undeclared file(s), {sum(t['bytes'] for t in trimmed) / (1 << 20):.1f} MiB (listed with size "
            f"and sha256, no bytes), e.g. {trimmed[0]['path']}. Write what to keep under $OUT, or declare it with --declared-output "
            f"GLOB: undeclared uploads will be dropped once lanes declare their outputs. See {SKILL}")


def stage(run_dir: Path, into: Path, rels: list[str] | None = None) -> list[str]:
    """Hard-link (else copy) the run files `rels` (default all) under `into`; the store copies what it ingests, so a link never aliases a blob."""
    rels = run_files(run_dir) if rels is None else rels
    for rel in rels:
        src, dst = Path(run_dir) / rel, Path(into) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return rels


#: What a run may be sent that grants access until it expires, as globs relative to its run dir and its request dir: check's presigned
#: URLs of the Lean dependency bundles (`tools/verity/check/check.py` LEAN_DEPS_URLS), which a run of anything else never takes out of inputs/,
#: and the temporary name a shipped tree's preflight writes that file under before replacing it
CREDENTIALS = ("inputs/lean-deps-urls.json", "inputs/.lean-deps-urls.json.*")


def drop_credentials(d: Path) -> None:
    """Delete each file (or symlink) under ``d`` that CREDENTIALS names; whatever else a workload left at those paths, a directory or a
    file where inputs/ should be, stays, and a symlinked inputs/ is not followed, as `run_files` doesn't follow it."""
    for pattern in CREDENTIALS:
        parent, name = pattern.rsplit("/", 1)
        where = Path(d) / parent
        if where.is_symlink() or not where.is_dir():
            continue
        with os.scandir(where) as entries:
            found = [e.path for e in entries if fnmatch.fnmatchcase(e.name, name) and not e.is_dir(follow_symlinks=False)]
        for p in found:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(p)


def put_run_record(st: Any, run_dir: Path, run_id: str) -> str | None:
    """The `run-record/v1` artifact of the run dir (None when it holds no file): the files `select` keeps, and the listing of the trimmed
    ones (module docstring).  CREDENTIALS are deleted from the run dir first: `verify` needs the record to hold every file left there."""
    drop_credentials(run_dir)
    root = Path(getattr(st, "root", None) or Path(run_dir).parent) / "scratch"
    tmp = root / f".custody-{run_id}-{secrets.token_hex(4)}"
    tmp.mkdir(parents=True)
    try:
        kept, trimmed = select(Path(run_dir))
        rels = stage(Path(run_dir), tmp, kept)
        if not rels:
            return None
        size = sum((Path(run_dir) / r).stat().st_size for r in rels)
        meta: dict[str, Any] = {"run_id": run_id, "files": len(rels), "bytes": size}
        if trimmed:
            listing = json.dumps(trimmed, separators=(",", ":"))
            if len(listing) <= TRIMMED_INLINE_BYTES:
                meta["trimmed"] = trimmed
            else:
                (tmp / TRIMMED_LISTING).write_text(listing)
                meta.update(trimmed_listing=TRIMMED_LISTING, trimmed_files=len(trimmed), trimmed_bytes=sum(t["bytes"] for t in trimmed))
            print(f"research: {trim_warning(trimmed)}", file=sys.stderr)
        return st.put_artifact(KIND, meta, tree=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def trimmed_of(st: Any, record: str, man: Any = None) -> list[dict[str, Any]]:
    """The run record's listing of the files it holds no bytes of: `meta.trimmed`, or its TRIMMED_LISTING file (fetched when not local)."""
    man = man or st.get_manifest(record)
    meta = man.meta or {}
    if not meta.get("trimmed_listing"):
        return list(meta.get("trimmed") or [])
    tree = st.fetch(record, paths=[meta["trimmed_listing"]])
    return json.loads((Path(tree) / meta["trimmed_listing"]).read_text())


def marker_problem(run_dir: Path) -> str | None:
    """None when `.custody` vouches for the run dir as it is now; else why not (module docstring)."""
    try:
        m = json.loads((Path(run_dir) / MARKER).read_text())
    except FileNotFoundError:
        return "no .custody"
    except (OSError, ValueError):
        return "unreadable .custody"
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        return f".custody is not {SCHEMA}"
    if not m.get("preserved"):
        return f".custody: not preserved ({m.get('error') or '?'})"
    try:
        t0, listed = float(m["verified_from"]), set(m["files"])
    except (KeyError, TypeError, ValueError):
        return "unreadable .custody"
    for rel in run_files(Path(run_dir)):
        if rel not in listed:
            return f"{rel} is not in the run record (added after custody)"
        try:
            if os.lstat(Path(run_dir) / rel).st_mtime >= t0:
                return f"{rel} changed after custody"
        except OSError:
            continue
    return None


def has_custody(run_dir: Path) -> tuple[bool, str]:
    """(True, how) when `preserved.json` (a verified copy here) or a valid `.custody` covers the run dir."""
    d = Path(run_dir)
    if (d / "preserved.json").exists():
        return True, "fetched (preserved.json)"
    why = marker_problem(d)
    return (True, "on the remote (.custody)") if why is None else (False, why)


# ---- verify / push / mark -----------------------------------------------------------------------------------------------------------

def push_run(st: Any, run_id: str) -> list[str]:
    """Upload an attempt's artifacts (verified), its manifest and the assertions about it; the errors (empty = all went up)."""
    from . import labels_sync as LS
    from .preserved import attempt_artifacts

    doc = st.get_attempt(run_id)
    if doc is None:
        return [f"attempt {run_id} is not in the store"]
    errors = []
    for name, art in attempt_artifacts(doc).items():
        rep = st.push(art, preserve=True)
        if not rep.get("preserved"):
            errors.append(f"{name} {art[:16]}: {(rep.get('errors') or ['not preserved'])[0]}")
    try:
        st.push_attempt(run_id)
        lrep = st.push_labels(LS.attempt_targets(doc))
        errors += [f"label {x}" for x in lrep.get("failed") or []]
    except Exception as e:  # noqa: BLE001
        errors.append(f"{type(e).__name__}: {e}")
    return errors


def verify(st: Any, run_dir: Path, run_id: str | None = None, *, mark: bool = True) -> dict[str, Any]:
    """The custody predicate for one run dir (module docstring); with `mark`, writes `.custody` when it holds."""
    from . import preserved

    d = Path(run_dir)
    run_id = run_id or d.name
    rep: dict[str, Any] = {"schema": SCHEMA, "run_id": run_id, "attempt": run_id, "run_record": None, "preserved": False, "error": None}
    try:
        doc = st.get_attempt(run_id)
    except Exception as e:  # noqa: BLE001
        doc, rep["error"] = None, f"{type(e).__name__}: {e}"
    if doc is None:
        rep["error"] = rep["error"] or f"attempt {run_id} is not in the store"
        return rep
    rep["run_record"] = rr = (doc.get("outputs") or {}).get("run_record")
    if not rr:
        rep["error"] = "the attempt names no run record (published without custody): only `research fetch --all` gives this run custody"
        return rep
    chk = preserved.check(st, [run_id], mode="head")
    rep["remote"] = chk.get("remote")
    if not chk["preserved"]:
        why = [e for x in chk["not_preserved"] for e in x.get("errors") or []] or [chk.get("reason") or "?"]
        rep["error"] = f"not preserved on the remote: {why[0]}"
        return rep
    man = st.get_manifest(rr)
    want = {f.path: (f.bytes, f.sha256) for f in (man.payload.files or []) if f.path != TRIMMED_LISTING}
    want.update({t["path"]: (t["bytes"], t["sha256"]) for t in trimmed_of(st, rr, man)})
    t0 = time.time()
    have: dict[str, list[Any]] = {}
    for rel in run_files(d):
        if rel == LAUNCHER_RECORD:
            continue
        p = d / rel
        have[rel] = [p.stat().st_size, _sha256(p)]
    extra = sorted(set(have) - set(want))
    changed = sorted(r for r in set(have) & set(want) if tuple(have[r]) != want[r])
    if extra or changed:
        rep["error"] = (f"{len(extra)} file(s) not in the run record, {len(changed)} changed since it was taken: "
                        + ", ".join((extra + changed)[:4]))
        return rep
    rep.update(preserved=True, verified_from=t0, verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files=have,
               gone=sorted(set(want) - set(have)), host=os.uname().nodename)
    if mark:
        write_json_atomic(d / MARKER, rep)
    return rep


def mark_failed(run_dir: Path, rep: dict[str, Any]) -> None:
    """Leave a failed verdict in .custody (as `runner_publish` does), for triage and whoever looks next: `preserved` false, `error`."""
    rep.setdefault("verified_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rep.setdefault("host", os.uname().nodename)
    write_json_atomic(Path(run_dir) / MARKER, rep)


# ---- the launcher: a minted, delete-free key for the machine ---------------------------------------------------------------------------

def mint_for_machine(ttl: str | int = DEFAULT_TTL, *, environ: dict[str, str] | None = None) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    """(public facts for request.json, [(relative path, bytes)] to stage under <requests>/<id>/custody/).  MintError when this machine
    cannot mint (no parent key, a temporary credential in the environment, no endpoint/bucket)."""
    from . import credentials as C

    cred = C.mint(via="auto", actions=list(ACTIONS), ttl_s=C.parse_ttl(ttl), environ=environ)
    toml = ("# written by `research run --custody-r2`; the credential is in cred.json beside it (mode 600)\n[remote]\ntype = \"s3\"\n"
            f"bucket = {json.dumps(cred['bucket'])}\nendpoint = {json.dumps(cred['endpoint'])}\nregion = \"auto\"\n"
            "access_key_id_env = \"AWS_ACCESS_KEY_ID\"\nsecret_access_key_env = \"AWS_SECRET_ACCESS_KEY\"\nsession_token_env = \"AWS_SESSION_TOKEN\"\n")
    secret = {"AWS_ACCESS_KEY_ID": cred["accessKeyId"], "AWS_SECRET_ACCESS_KEY": cred["secretAccessKey"], "AWS_SESSION_TOKEN": cred["sessionToken"],
              "expires_utc": cred["expires_utc"]}
    public = {"schema": SCHEMA, "expires_utc": cred["expires_utc"], "ttl_s": cred["ttl_s"], "via": cred["via"], "actions": list(ACTIONS)}
    return public, [(f"{DIR_NAME}/{STORE_TOML}", toml.encode()), (f"{DIR_NAME}/{CRED_FILE}", (json.dumps(secret) + "\n").encode())]


@contextlib.contextmanager
def custody_env(custody_dir: Path) -> Iterator[None]:
    """The store's remote for the duration: RESEARCH_STORE_CONFIG -> the staged store.toml, AWS_* from cred.json; restored after."""
    d = Path(custody_dir)
    cred = json.loads((d / CRED_FILE).read_text())
    want = {"RESEARCH_STORE_CONFIG": str(d / STORE_TOML), **{k: cred[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")}}
    saved = {k: os.environ.get(k) for k in want}
    os.environ.update(want)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def forget_credential(custody_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (Path(custody_dir) / CRED_FILE).unlink()


PUBLISH_LOG = "publish.log"


def divert_output(run_dir: Path, custody_dir: Path) -> bool:
    """When this process's stderr is the run dir's launcher.log (the machine-side runner), point fds 1 and 2 at <custody dir>/publish.log
    for the rest of the process.  The run record is taken next, and any line printed into launcher.log after it (an S3 retry notice, the
    failure line) makes the run dir differ from its record, so custody could never be verified, not even after a repair push."""
    try:
        if not os.path.samestat(os.fstat(2), os.stat(Path(run_dir) / "launcher.log")):
            return False
        fd = os.open(Path(custody_dir) / PUBLISH_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError:
        return False
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    return True


def _key_expiry(custody_dir: Path) -> str | None:
    try:
        return json.loads((Path(custody_dir) / CRED_FILE).read_text()).get("expires_utc")
    except (OSError, ValueError, AttributeError):
        return None


def _expired(expires_utc: str | None, now: float | None = None) -> bool:
    try:
        exp = calendar.timegm(time.strptime(expires_utc or "", "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, OverflowError):
        return False
    return (time.time() if now is None else now) >= exp


# ---- the runner -------------------------------------------------------------------------------------------------------------------------

#: A run bigger than this uploads without resume (a failed upload starts over), so custody warns (``size_warning``) before pushing it.
SIZE_WARN_BYTES = 4 << 30


def size_warning(run_dir: Path) -> str | None:
    """A warning when the run's files exceed :data:`SIZE_WARN_BYTES`, or ``None``."""
    d = Path(run_dir)
    total = sum((d / r).stat().st_size for r in run_files(d))
    if total <= SIZE_WARN_BYTES:
        return None
    return (f"run files are {total / (1 << 30):.1f} GiB (over {SIZE_WARN_BYTES >> 30} GiB): the custody upload has no resume, so a dropped "
            f"connection restarts it; keep big outputs out of the run dir or give --custody-ttl room")


def runner_publish(run_dir: Path, custody_dir: Path, *, tool: Any = None, conditions: dict[str, Any] | None = None,
                   store_root: str | None = None, sleep: Callable[[float], None] = time.sleep, tries: int = PUSH_TRIES) -> dict[str, Any]:
    """What `research run --custody-dir` does after the workload: publish the attempt with its run record, push and verify it (retrying),
    delete the key, and leave the verdict in <run>/.custody.  Never raises.  Prints only on failure, and on the machine into
    <custody dir>/publish.log (`divert_output`): once the record is taken nothing may touch the run dir, launcher.log included."""
    from . import attempt as A
    from .local import LocalStore

    d = Path(run_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    divert_output(d, custody_dir)
    expires = _key_expiry(custody_dir)
    rep: dict[str, Any] = {"schema": SCHEMA, "run_id": d.name, "preserved": False, "error": None}
    warn = size_warning(d)
    if warn:
        print(f"research: custody: {warn}", file=sys.stderr)
    try:
        with custody_env(custody_dir):
            st = LocalStore(A.store_root(store_root))
            pub = A.publish(d, tool=tool, conditions=conditions, store=st, remote_push=False, run_record=True)
            if not pub.get("published"):
                rep["error"] = f"attempt not published: {pub.get('error')}"
            else:
                for n in range(max(1, tries)):
                    errors = push_run(st, d.name)
                    rep = verify(st, d, d.name, mark=False)
                    if rep["preserved"]:
                        break
                    rep["error"] = f"{rep['error']}" + (f"; push: {errors[0]}" if errors else "")
                    if n + 1 < tries:
                        sleep(PUSH_RETRY_S * (n + 1))
    except Exception as e:  # noqa: BLE001  the run's exit status is the workload's, never custody's
        rep.update(preserved=False, error=f"{type(e).__name__}: {e}")
    finally:
        forget_credential(custody_dir)
    if not rep.get("preserved") and _expired(expires):
        rep["error"] = (f"the custody key expired at {expires}: the run outlived --custody-ttl (launch the next one with a longer ttl; repair "
                        f"this one with a fresh key); {rep['error']}")
    if warn:
        rep["size_warning"] = warn
    if rep.get("preserved"):
        write_json_atomic(d / MARKER, rep)                  # verified_from precedes it; nothing below writes into the run dir
    else:
        mark_failed(d, rep)
        print(f"research: custody NOT on the remote: {rep['error']}  (`research data custody {d.name} --publish` with credentials, or "
              f"`research fetch --all {d.name}` from the launcher)", file=sys.stderr)
    return rep


# ---- triage (F2) ------------------------------------------------------------------------------------------------------------------------

def _label_verdict(st: Any, run_id: str) -> str | None:
    """The run's latest `custody` label when it is a verdict (waived, never-ran); `needed` withdraws an earlier one."""
    from .vocab import CUSTODY

    try:
        labs = st.labels(run_id, key=CUSTODY)
    except Exception:  # noqa: BLE001
        return None
    return str(labs[-1].value) if labs and str(labs[-1].value) in ("waived", "never-ran") else None


#: What `tele run` writes only once it starts the workload: `started`, then the workload's pid in status.json, the sampler
#: (`sampler_start`, then resources.jsonl's readings of the workload's processes) and `span_start`.
_RAN_EVENTS = ("started", "span_start", "sampler_start")


def ran_evidence(run_dir: Path) -> str | None:
    """The first record in the run dir showing that its workload started, or None: a pid in status.json, a started, span_start or
    sampler_start event, a sampler reading, output in stdout.log or stderr.log.  A `custody never-ran` label holds only while
    there is none (the vocabulary's rule)."""
    d = Path(run_dir)
    pid, _ = _runner_pid(_read(d / "status.json") or {})
    if pid is not None:
        return f"status.json names pid {pid}"
    with contextlib.suppress(OSError), (d / "events.jsonl").open(errors="replace") as f:
        for line in f:
            with contextlib.suppress(ValueError, AttributeError):
                if (kind := json.loads(line).get("kind")) in _RAN_EVENTS:
                    return f"events.jsonl has {kind}"
    for name, what in (("resources.jsonl", "sampler readings"), ("stdout.log", "output"), ("stderr.log", "output")):
        with contextlib.suppress(OSError):
            if (d / name).stat().st_size:
                return f"{name} holds {what}"
    return None


TRIAGE_SCHEMA = "research/custody-triage/v1"
NEEDS_ACTION = ("no-custody", "attempt-unpreserved", "attempt-no-record", "forced-unverified", "conflict")
#: A run whose runner this machine can't see (it ran in another pid namespace: a Kubernetes pod on node 1) has ended once none of
#: its records changed for this long: a pod preempted or killed before its runner recorded an end leaves its status `running`.
STALE_S = 24 * 3600
_RECORDS = ("status.json", "job.json", "stdout.log", "stderr.log", "launcher.log", "sampler.log", "events.jsonl", "resources.jsonl")


def this_host() -> tuple[str, str | None]:
    """(hostname, boot id): a runner's pid means something here only when its job.json names both."""
    import socket

    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = None
    return socket.gethostname(), boot


def _changed(d: Path) -> float | None:
    """When the run's own records last changed (its end, for an ended run); markers about the dir don't count."""
    ts = []
    for name in _RECORDS:
        with contextlib.suppress(OSError):
            ts.append((d / name).stat().st_mtime)
    return max(ts) if ts else None


def _runner_pid(status: dict[str, Any]) -> tuple[Any, Any]:
    """(pid, starttime_ticks) of the run's runner: status.json's own, which `set_status` keeps when a later `running` transition
    carries none (`tele set-timeout`'s), else the last transition that has one."""
    for t in (status, *reversed(status.get("transitions") or [])):
        if isinstance(t, dict) and t.get("pid") is not None:
            return t.get("pid"), t.get("starttime_ticks")
    return None, None


def _alive(pid: Any, ticks: Any) -> bool | None:
    """Whether process `pid` with that start time runs here; None when `pid` doesn't read as one, which is never taken for dead."""
    try:
        if int(pid) <= 0:
            return None
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError):
        return None
    try:
        return ticks is None or int(fields[19]) == int(ticks)
    except (ValueError, IndexError, TypeError):
        return None


def _runner(d: Path, status: dict[str, Any], now: float, host: tuple[str, str | None]) -> tuple[bool, str, bool | None]:
    """(ended, how, alive) for a run dir without a launcher's remote.json, i.e. on the machine that ran it, whose status isn't
    terminal: its runner's pid with the same start time when job.json says it ran on this host since this boot, dead when it ran on
    this host before a reboot; else STALE_S. `alive` is what this host saw of the runner: False gone, True running here, None it
    can't see it (a Kubernetes pod's runner, in another pid namespace, or another machine's)."""
    env = (_read(d / "job.json") or {}).get("environment") or {}
    if status and env.get("hostname") == host[0] and host[1] and env.get("boot_id") and env["boot_id"] != host[1]:
        return True, f"dead: this host rebooted since its runner started (boot {env['boot_id']}, now {host[1]})", False
    local = bool(status) and (env.get("hostname"), env.get("boot_id")) == host and bool(host[1])
    pid, ticks = _runner_pid(status)
    alive = _alive(pid, ticks) if local else None
    if alive is False:
        return True, f"dead: its runner (pid {pid}) is gone without recording an end", False
    if alive:
        return False, "running here", True
    changed = _changed(d)
    if changed is not None and now - changed > STALE_S:
        return True, f"stale: status {status.get('state') or 'absent'}, and no record changed for {(now - changed) / 3600:.0f} h", None
    if not status:
        return False, "starting (no status.json yet)", None
    return (False, "running here (its status names no pid that reads)" if local else
            "running elsewhere (its runner is not visible from here)", None)


def force_refusal(run_dir: Path, *, runner_gone: bool = False, now: float | None = None,
                  host: tuple[str, str | None] | None = None) -> str | None:
    """Why `research data custody RUN --publish --force` must not publish the run as its runner last left it, or None. A status
    that isn't terminal is the runner's last word only once the runner is gone. This host can see that of a runner that ran here
    (its pid, its boot); of any other (a Kubernetes pod, another machine) only the run's own host can, and the caller says it did
    with `runner_gone` (n1_custody.sh: `kubectl get pod` says NotFound; n2_custody.sh: the pid is gone on node 2). Records quiet
    for STALE_S never are: a pod whose workload prints nothing for a day may still run."""
    d = Path(run_dir)
    status = _read(d / "status.json") or {}
    if status.get("state") in TERMINAL_STATES:
        return None
    _, how, alive = _runner(d, status, time.time() if now is None else now, host or this_host())
    if alive is False:
        return None
    if alive:
        return f"its runner is {how}"
    if runner_gone:
        return None
    where = ((_read(d / "job.json") or {}).get("environment") or {}).get("hostname") or "?"
    return (f"{how}; this host can't see its runner (job.json names host {where}): pass --runner-gone once that host says it is gone "
            f"(a pod: `kubectl get pod {where}` NotFound)")


def triage_row(st: Any, run_dir: Path, *, verify_remote: bool = False, now: float | None = None,
               host: tuple[str, str | None] | None = None) -> dict[str, Any]:
    """One run dir's custody state (TRIAGE_STATES) with the reason, the next step and when its records last changed (`ended_utc`).

    A `custody` label waived|never-ran decides, the latest one winning (`needed` withdraws either), but a never-ran is ignored while
    `ran_evidence` shows the workload started.  On the machine that ran it, a run still marked running has ended when its runner's
    pid is gone (same host and boot), or, when the runner isn't visible there (a Kubernetes pod), when none of its records changed
    for 24 h.  A valid `.custody` counts as `r2` where the catalog can't say (node 2's runs are published on node 1).  An attempt
    published `--force` is `r2` only while a valid `.custody` shows nothing in the run dir changed after its run record was taken,
    else `forced-unverified`: its write-once attempt can't take the run's real end."""
    from . import preserved

    d = Path(run_dir)
    rid = d.name
    now = time.time() if now is None else now
    changed = _changed(d)
    row: dict[str, Any] = {"run": rid, "state": None, "why": None, "next": None, "files": None, "bytes": None,
                           "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(changed)) if changed else None,
                           "age_min": round((now - changed) / 60, 1) if changed else None}
    rec = _read(d / "remote.json") or {}
    status = _read(d / "status.json") or {}
    if (d / "preserved.json").exists():
        row.update(state="fetched", why="preserved.json (every file verified from the machine)")
        return row
    verdict = _label_verdict(st, rid)
    ran = ran_evidence(d) if verdict == "never-ran" else None
    if verdict and not ran:
        row.update(state="waived" if verdict == "waived" else "never-ran", why=f"label custody={verdict}")
        return row
    from ..notes import NEVER_RAN

    if rec and (rec.get("phase") in NEVER_RAN or (d / "refused.json").exists()):
        row.update(state="never-ran", why=f"launch phase {rec.get('phase')}" + (" (refused)" if (d / "refused.json").exists() else ""))
        return row
    ended, force = None, " --force --runner-gone (once its host says its runner is gone)"
    if not rec and status.get("state") not in TERMINAL_STATES:
        is_ended, how, alive = _runner(d, status, now, host or this_host())
        if not is_ended:
            row.update(state="running", why=how)
            return row
        ended = how
        if alive is False:
            force = " --force"
    try:
        doc = st.get_attempt(rid)
    except Exception:  # noqa: BLE001
        doc = None
    if doc is None:
        live = status.get("state") not in TERMINAL_STATES and rec.get("phase") == "launched"
        row.update(state="running" if live and status else "no-custody",
                   why=("still running as last observed" if live and status else
                        "no attempt in the store" + ("" if status or not rec else ", and its records were never fetched")),
                   next=(f"research fetch {rid}" if live and status else
                         f"research fetch --all {rid} (pod alive), else research data custody {rid} --publish"
                         + (force if ended else "") + ", or label custody waived"))
    else:
        rr = (doc.get("outputs") or {}).get("run_record")
        chk = preserved.check(st, [rid], mode="head" if verify_remote else "recorded", head_fallback=verify_remote)
        why = [e for x in chk["not_preserved"] for e in x.get("errors") or []] or [chk.get("reason") or "?"]
        # an attempt published --force froze the run as its runner last left it; it is the run's end only while nothing in the run
        # dir changed after its run record was taken (a pod forced while it still ran ends later, and its write-once attempt can't)
        forced = (doc.get("status") or {}).get("state") not in TERMINAL_STATES
        unvouched = marker_problem(d) if forced and rr else None
        if not chk["preserved"]:
            row.update(state="attempt-unpreserved", why=f"attempt not preserved: {why[0]}", next=f"research data push {rid}")
        elif not rr:
            row.update(state="attempt-no-record", why="attempt preserved, but its logs are only here (published without a run record)",
                       next=f"research fetch --all {rid} (pod alive), else keep the local dir or label custody waived")
        elif unvouched:
            row.update(state="forced-unverified",
                       why=(f"attempt and run record preserved, published --force with status {(doc.get('status') or {}).get('state')}, "
                            f"and no .custody shows the record holds the run dir as it is now: {unvouched}"),
                       next=(f"research data custody {rid} (it writes .custody while the run record still holds the run dir)"
                             if unvouched in ("no .custody", "unreadable .custody") else
                             "the attempt is write-once, so what the run wrote after its run record is only here: keep the run dir, or put "
                             "it in the store (research data put --kind K --tree DIR --preserve), then label custody waived"))
        else:
            row.update(state="r2", why="attempt and run record preserved on the remote" + (
                " (published --force, and .custody shows nothing in the run dir changed since)" if forced else ""))
        # only a head-mode check sees the attempt document itself: --verify's own, else the last `custody --publish`'s verdict in .custody
        m = {} if verify_remote else _read(d / MARKER) or {}
        seen = why if verify_remote else [str(m.get("error") or "")] if m.get("preserved") is False else []
        if differs := next((e for e in seen if preserved.DIFFERS in e), None):
            row.update(state="conflict", why=differs if verify_remote else f"{differs} (.custody, {m.get('verified_utc') or '?'})",
                       next=f"research data reconcile-attempts {rid} --apply (the remote's attempt is the first one ever pushed, and "
                            f"canonical), then research data custody {rid} --publish; or label custody waived")
    # the marker is written only after a head-mode check of the remote, on whichever machine published the run (node 2's runs are
    # published on node 1); --verify trusts only what it asks the remote itself
    if row["state"] in ("no-custody", "attempt-unpreserved") and not (verify_remote and doc is not None) and marker_problem(d) is None:
        m = _read(d / MARKER) or {}
        row.update(state="r2", why=f"on the remote (.custody, verified {m.get('verified_utc') or '?'} on {m.get('host') or '?'})", next=None)
    if ended and row["state"] != "r2":
        row["why"] = f"{ended}; {row['why']}"
    if ran:
        row["why"] = f"label custody=never-ran ignored, since {ran}; {row['why']}"
    if row["state"] in NEEDS_ACTION:
        rels = run_files(d)
        row.update(files=len(rels), bytes=sum((d / r).stat().st_size for r in rels))
    return row


def _read(p: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def triage(st: Any, runs_dir: Path, *, verify_remote: bool = False, now: float | None = None,
           host: tuple[str, str | None] | None = None) -> list[dict[str, Any]]:
    root = Path(runs_dir)
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
                  and ((p / "job.json").exists() or (p / "remote.json").exists())) if root.is_dir() else []
    host = host or this_host()
    return [triage_row(st, d, verify_remote=verify_remote, now=now, host=host) for d in dirs]


def triage_report(st: Any, roots: list[Path], *, older_min: float | None = None, verify_remote: bool = False, now: float | None = None,
                  host: tuple[str, str | None] | None = None) -> dict[str, Any]:
    """`research data custody --triage` (TRIAGE_SCHEMA): every run under `roots`; a run needing action whose records last changed
    under `older_min` minutes ago is `settling` and doesn't count, so `needs_action` is the invariant "ended runs not on the remote
    after N minutes"."""
    now = time.time() if now is None else now
    rows = []
    for root in roots:
        for r in triage(st, root, verify_remote=verify_remote, now=now, host=host):
            r["root"] = str(root)
            r["settling"] = bool(r["state"] in NEEDS_ACTION and older_min is not None and r["age_min"] is not None
                                 and r["age_min"] < older_min)
            rows.append(r)
    todo = [r for r in rows if r["state"] in NEEDS_ACTION and not r["settling"]]
    return {"schema": TRIAGE_SCHEMA, "host": (host or this_host())[0], "now_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "runs_roots": [str(r) for r in roots], "runs_root": str(roots[0]) if roots else None, "older_min": older_min,
            "counts": {s: sum(1 for r in rows if r["state"] == s) for s in TRIAGE_STATES},
            "rows": rows, "needs_action": len(todo), "settling": sum(1 for r in rows if r["settling"])}


__all__ = ["ACTIONS", "CREDENTIALS", "EXCLUDED", "KIND", "MARKER", "PUBLISH_LOG", "SCHEMA", "custody_env", "divert_output",
           "drop_credentials", "force_refusal", "has_custody", "mark_failed", "marker_problem", "mint_for_machine", "push_run",
           "put_run_record", "ran_evidence", "run_files", "runner_publish", "trimmed_of", "triage", "triage_row", "verify"]
