"""The store's garbage collection (README §7.4): blobs that only uncited artifacts past their kind's TTL hold leave the remote.

What stays (the keep set):

* **cited**: an `art:` id (full, or the abbreviated `art:<8+ hex>` reports print) or a run id named by any label on the remote
  (its target, ref or value), by a file of a citing tree (`--cite DIR`: the notes repository, a checkout of the code with its
  fixture registry) or by Linear (`--cite-linear`: every issue's description and comments, projects, project updates and
  documents); a cited run keeps every `art:` id its attempt names (outputs, inputs, telemetry).
* **young**: an artifact whose manifest is younger than its kind's TTL (`TTL_DAYS`, by the manifest's LastModified).
* **untimed**: every artifact of a kind `TTL_DAYS` doesn't name.

What a kept manifest's `refs` name joins the keep set, to a fixed point; what a cited one's `meta` names (art ids and runs) does
too, but not what a young or untimed one's `meta` mentions: a result naming its own run is no citation of that run.  A blob is a
candidate when an expired artifact (one outside the keep set) holds it and no kept artifact does.  Manifests, attempts and labels
are never deleted: an expired id still resolves, and `fetch` reports its blob absent.

Deletion is two-phase, for uploads in flight.  An upload skips a blob the remote already holds (`S3Remote.put` HEADs first), so a
new artifact can come to hold an old blob before its manifest is listed.  `apply` therefore deletes only a blob that has been a
candidate since a pass at least `GRACE_S` ago (`state`, the marks) and still is in this one.  It deletes `BATCH` blobs at a
time, each batch after a fresh listing of `manifests/`, so a blob a new manifest holds is kept unless that manifest appears while
its own batch is being deleted; the listing after the last batch reports any such blob that is now absent as `lost` (the pass
exits 1).  `GRACE_S` covers a custody upload (its key lives at most 24 h, README §10.1).

Fails closed: a manifest, label or attempt it can't read, or a citing tree it can't scan, stops the pass before anything is
deleted.  Remote objects are immutable, so `cache` keeps every manifest, label and attempt it read, by key.  Memory stays small on
a bucket of millions of blobs: each manifest is parsed as it is read, and its blobs go to a SQLite file in `work` that one query
groups by digest.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

TTL_DAYS: dict[str, float] = {
    "run-record/v1": 7,
    "vllm-replay-bundle/v1": 7,
    "vllm-replay-slim/v1": 7,         # clear values, which no stage publishes
    "vllm-rowdir-tar/v1": 3,
    "run-files/v1": 14,
    "vllm-row-meta/v1": 14,
    "telemetry/v1": 30,
}
GRACE_S = 24 * 3600
STRONG = ("cited", "named", "cited run")
JOBS = 32
BATCH = 1000                   # deletes between two listings of the manifests (apply)
ART_TEXT = re.compile(r"art:([0-9a-f]{8,64})(?![0-9a-f])")
RUN_TEXT = re.compile(r"(?<![0-9A-Za-z])(r\d{8}-\d{6}-[0-9a-f]{4})(?![0-9a-f])")
SCAN_MAX_BYTES = 8 << 20
PARSE_ERRORS = (ValueError, KeyError, TypeError, AttributeError)
MANIFEST_KEY = re.compile(r"manifests/([0-9a-f]{64})\.json")
ATTEMPT_KEY = re.compile(r"attempts/(r\d{8}-\d{6}-[0-9a-f]{4})\.json")
"""The records' keys exactly: a multipart sidecar (`<key>.parts.json`) beside a large one is no record."""


class GcError(Exception):
    """Something the pass could not establish; nothing was deleted."""


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def named(text: str) -> tuple[set[str], set[str]]:
    """The art hex digits (full or abbreviated) and run ids `text` names."""
    return set(ART_TEXT.findall(text)), set(RUN_TEXT.findall(text))


def scan_tree(root: Path) -> tuple[set[str], set[str]]:
    """Every art hex and run id the files of `root` name: its tracked files when it is a git work tree (`git grep`), else every
    regular file under `SCAN_MAX_BYTES`."""
    root = Path(root)
    if not root.is_dir():
        raise GcError(f"citing tree {root} is not a directory")
    if (root / ".git").exists():
        pat = r"art:[0-9a-f]{8,64}|r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}"
        r = subprocess.run(["git", "-C", str(root), "grep", "-I", "-h", "-o", "-E", pat], capture_output=True, text=True)
        if r.returncode not in (0, 1):
            raise GcError(f"git grep in {root} failed (rc {r.returncode}): {r.stderr.strip()[-300:]}")
        return named(r.stdout)
    arts: set[str] = set()
    runs: set[str] = set()
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != ".git"]
        for n in fns:
            p = Path(dp) / n
            try:
                if not p.is_file() or p.stat().st_size > SCAN_MAX_BYTES:
                    continue
                a, r_ = named(p.read_text(errors="replace"))
            except OSError as e:
                raise GcError(f"cannot read {p}: {e}") from None
            arts |= a
            runs |= r_
    return arts, runs


LINEAR_PAGE = 50
LINEAR_ISSUES = """query($after: String) { issues(first: %d, after: $after, includeArchived: true) {
  nodes { id identifier description comments(first: %d) { nodes { body } pageInfo { hasNextPage endCursor } } }
  pageInfo { hasNextPage endCursor } } }""" % (LINEAR_PAGE, LINEAR_PAGE)
LINEAR_COMMENTS = """query($id: String!, $after: String) { issue(id: $id) { comments(first: %d, after: $after) {
  nodes { body } pageInfo { hasNextPage endCursor } } } }""" % LINEAR_PAGE
LINEAR_OTHERS = {
    "projects": "nodes { description content }",
    "projectUpdates": "nodes { body }",
    "documents": "nodes { content }",
}


def scan_linear(gql: Callable[[str, dict], dict]) -> tuple[set[str], set[str], int]:
    """Every art hex and run id Linear's text names (top, 26 Oct: lanes cite mostly there): each issue's description and comments,
    archived ones too, every project's description and content, project updates and documents; and how many pages it read.
    `gql(query, variables)` is `research.linear.Linear.gql`. Any error is a GcError: a citation it couldn't read could lose a blob."""
    arts: set[str] = set()
    runs: set[str] = set()
    pages = 0

    def take(text: Any) -> None:
        a, r = named(json.dumps(text))
        arts.update(a)
        runs.update(r)

    def pull(query: str, variables: dict) -> dict:
        nonlocal pages
        pages += 1
        try:
            return gql(query, variables)
        except Exception as e:  # noqa: BLE001
            raise GcError(f"Linear: {type(e).__name__}: {e}") from None

    after = None
    while True:
        conn = pull(LINEAR_ISSUES, {"after": after})["issues"]
        for issue in conn["nodes"]:
            take([issue.get("identifier"), issue.get("description"), issue["comments"]["nodes"]])
            more = issue["comments"]["pageInfo"]
            while more["hasNextPage"]:
                c = pull(LINEAR_COMMENTS, {"id": issue["id"], "after": more["endCursor"]})["issue"]["comments"]
                take(c["nodes"])
                more = c["pageInfo"]
        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]
    for field, sel in LINEAR_OTHERS.items():
        after = None
        while True:
            q = "query($after: String) { %s(first: %d, after: $after) { %s pageInfo { hasNextPage endCursor } } }" % (
                field, LINEAR_PAGE, sel)
            conn = pull(q, {"after": after})[field]
            take(conn["nodes"])
            if not conn["pageInfo"]["hasNextPage"]:
                break
            after = conn["pageInfo"]["endCursor"]
    return arts, runs, pages


class Reader:
    """GETs immutable keys through a cache directory, `jobs` at a time."""

    def __init__(self, remote: Any, cache: Path, jobs: int = JOBS):
        self.remote, self.cache, self.jobs = remote, Path(cache), jobs

    def get(self, key: str) -> bytes:
        p = self.cache / key
        try:
            return p.read_bytes()
        except FileNotFoundError:
            pass
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f".tmp-{p.name}-{os.getpid()}-{id(p)}"
        try:
            self.remote.get(key, tmp)
        except KeyError:
            raise GcError(f"{key} is listed but can't be read") from None
        os.replace(tmp, p)
        return p.read_bytes()

    def each(self, keys: list[str], fn: Callable[[bytes], Any]) -> Iterator[tuple[str, Any]]:
        """(key, fn(its bytes)) for every key, in order, reading `jobs` ahead; the bytes are dropped once `fn` has them.  Any key
        unreadable, or that `fn` can't parse, is a GcError once every key has been tried."""
        def one(k: str) -> tuple[str, Any, str | None]:
            try:
                return k, fn(self.get(k)), None
            except GcError as e:
                return k, None, str(e)
            except PARSE_ERRORS as e:
                return k, None, f"{k} is unparseable ({type(e).__name__}: {e})"
            except Exception as e:  # noqa: BLE001  reported below, all at once
                return k, None, f"{k}: {type(e).__name__}: {e}"
        bad: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, min(self.jobs, len(keys) or 1))) as ex:
            for i in range(0, len(keys), 4096):
                for k, v, err in ex.map(one, keys[i:i + 4096]):
                    if err:
                        bad.append(err)
                    else:
                        yield k, v
        if bad:
            raise GcError(f"{len(bad)} key(s) unreadable, e.g. {bad[0]}")


@dataclass
class Art:
    idx: int
    kind: str
    age_s: float
    bytes: int
    refs: frozenset[str]
    mentions: frozenset[str]
    runs: frozenset[str]


@dataclass
class Plan:
    """`candidates`: blob hex -> bytes; `expired`: art id -> kind; `by_kind`: kind -> {artifacts, bytes, expired, expired_bytes};
    `kept_by`: why the keep set holds what it holds (artifact counts)."""

    listed: set[str]
    candidates: dict[str, int] = field(default_factory=dict)
    expired: dict[str, str] = field(default_factory=dict)
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    kept_by: dict[str, int] = field(default_factory=dict)
    cited_runs: int = 0

    def summary(self) -> dict[str, Any]:
        return {"artifacts": len(self.listed), "expired_artifacts": len(self.expired), "candidate_blobs": len(self.candidates),
                "candidate_bytes": sum(self.candidates.values()), "cited_runs": self.cited_runs, "kept_by": self.kept_by,
                "by_kind": self.by_kind}


def _parse_manifest(raw: bytes) -> tuple[str, int, frozenset[str], frozenset[str], frozenset[str], list[tuple[bytes, int]]]:
    """(kind, payload bytes, refs' art hex, meta's art hex, meta's runs, [(blob digest, bytes)])."""
    doc = json.loads(raw)
    payload = doc.get("payload") or {}
    if payload.get("type") == "file":
        blobs = [(payload["sha256"], payload["bytes"])]
    else:
        blobs = [(f["sha256"], f["bytes"]) for f in payload.get("files") or []]
    refs, _ = named(json.dumps(doc.get("refs")))
    mentions, runs = named(json.dumps(doc.get("meta")))
    return (str(doc["kind"]), int(payload.get("bytes") or 0), frozenset(refs), frozenset(mentions), frozenset(runs),
            [(bytes.fromhex(b), int(n)) for b, n in blobs])


def _resolve(hexes: set[str], ordered: list[str]) -> set[str]:
    """Full ids for `hexes`: a full one if listed, an abbreviated one as every listed id it begins."""
    out: set[str] = set()
    for h in hexes:
        if len(h) == 64:
            out.add(h)
            continue
        i = bisect.bisect_left(ordered, h)
        while i < len(ordered) and ordered[i].startswith(h):
            out.add(ordered[i])
            i += 1
    return out


def _label_target(key: str) -> str:
    d = key.split("/")[1] if key.count("/") >= 2 else ""
    return d.replace("art_", "art:", 1) if d.startswith("art_") else d


def plan(remote: Any, cache: Path, cite: list[Path], *, now: float, ttl: dict[str, float] | None = None, jobs: int = JOBS,
         work: Path | None = None, linear: Callable[[str, dict], dict] | None = None,
         log: Callable[[str], None] | None = None) -> Plan:
    say = log or (lambda s: None)
    ttl = TTL_DAYS if ttl is None else ttl
    rd = Reader(remote, cache, jobs)
    if not cite:
        raise GcError("no citing tree given (--cite: the notes repository and a checkout of the code)")
    cited_hex: set[str] = set()
    cited_runs: set[str] = set()
    for root in cite:
        a, r = scan_tree(Path(root))
        say(f"{root}: names {len(a)} art ids, {len(r)} runs")
        cited_hex |= a
        cited_runs |= r
    if linear is not None:
        a, r, pages = scan_linear(linear)
        say(f"Linear ({pages} pages): names {len(a)} art ids, {len(r)} runs")
        cited_hex |= a
        cited_runs |= r
    listing = {m.group(1): _epoch(t) for k, t in remote.list_times("manifests/") if (m := MANIFEST_KEY.fullmatch(k))}
    say(f"{len(listing)} manifests listed")
    with tempfile.TemporaryDirectory(prefix="store-gc-", dir=work) as tmp:
        db = sqlite3.connect(Path(tmp) / "blobs.sqlite")
        db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE b(art INTEGER, digest BLOB, size INTEGER);")
        arts: dict[str, Art] = {}
        keys = [f"manifests/{h}.json" for h in sorted(listing)]
        for i, (k, (kind, n, refs, mentions, runs, blobs)) in enumerate(rd.each(keys, _parse_manifest)):
            h = k[len("manifests/"):-len(".json")]
            arts[h] = Art(idx=i, kind=kind, age_s=now - listing[h], bytes=n, refs=refs, mentions=mentions, runs=runs)
            db.executemany("INSERT INTO b VALUES (?,?,?)", ((i, d, s) for d, s in blobs))
            if i % 20000 == 0 and i:
                say(f"{i} manifests read")
        ordered = sorted(arts)
        label_keys = [k for k in remote.list("labels/") if k.endswith(".json") and not k.endswith(".parts.json")]
        for k, (a, r) in rd.each(label_keys, lambda body: named(body.decode(errors="replace"))):
            ta, tr = named(_label_target(k))
            cited_hex |= a | ta
            cited_runs |= r | tr
        say(f"{len(label_keys)} labels read; cited: {len(cited_hex)} art ids, {len(cited_runs)} runs")
        attempts = {m.group(1) for k in remote.list("attempts/") if (m := ATTEMPT_KEY.fullmatch(k))}
        p = Plan(listed=set(arts))
        keep = _keep(arts, ordered, cited_hex, cited_runs, attempts, ttl, rd, p)
        for why in keep.values():
            p.kept_by[why] = p.kept_by.get(why, 0) + 1
        db.execute("CREATE TABLE k(art INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO k VALUES (?)", ((arts[h].idx,) for h in keep))
        for h, a in arts.items():
            row = p.by_kind.setdefault(a.kind, {"artifacts": 0, "bytes": 0, "expired": 0, "expired_bytes": 0})
            row["artifacts"] += 1
            row["bytes"] += a.bytes
            if h not in keep:
                p.expired[f"art:{h}"] = a.kind
                row["expired"] += 1
                row["expired_bytes"] += a.bytes
        q = "SELECT b.digest, MAX(b.size) FROM b LEFT JOIN k ON k.art = b.art GROUP BY b.digest HAVING COUNT(k.art) = 0"
        p.candidates = {d.hex(): int(s) for d, s in db.execute(q)}
        db.close()
    say(f"{len(p.expired)} artifacts expired; {len(p.candidates)} candidate blobs, {sum(p.candidates.values()) / 2**30:.1f} GiB")
    return p


def _keep(arts: dict[str, Art], ordered: list[str], cited_hex: set[str], cited_runs: set[str], attempts: set[str],
          ttl: dict[str, float], rd: Reader, p: Plan) -> dict[str, str]:
    """{art hex: why it stays} (module docstring)."""
    keep: dict[str, str] = {}
    stack: list[str] = []

    def hold(h: str, why: str) -> None:
        """Keep `h`; a citation outranks a weak reason (young, untimed, a weak artifact's ref), and is followed again."""
        if h not in arts or keep.get(h) in STRONG or (h in keep and why not in STRONG):
            return
        keep[h] = why
        stack.append(h)

    for h in _resolve(cited_hex, ordered):
        hold(h, "cited")
    for h, a in arts.items():
        if a.kind not in ttl:
            hold(h, "untimed")
        elif a.age_s < ttl[a.kind] * 86400:
            hold(h, "young")
    seen_runs: set[str] = set()
    todo_runs = set(cited_runs)
    while stack or todo_runs:
        while stack:
            h = stack.pop()
            a, strong = arts[h], keep[h] in STRONG
            for r in _resolve(set(a.refs), ordered):
                hold(r, "named" if strong else "ref")
            if strong:
                for r in _resolve(set(a.mentions), ordered):
                    hold(r, "named")
                todo_runs |= a.runs - seen_runs
        runs = sorted((todo_runs - seen_runs) & attempts)
        seen_runs |= todo_runs
        todo_runs = set()
        for _, a_hex in rd.each([f"attempts/{r}.json" for r in runs], lambda body: named(body.decode(errors="replace"))[0]):
            for h in _resolve(a_hex, ordered):
                hold(h, "cited run")
    p.cited_runs = len(seen_runs & attempts)
    return keep


def load_marks(path: Path) -> dict[str, float]:
    try:
        doc = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except ValueError as e:
        raise GcError(f"{path}: unreadable marks ({e}); move it aside to start marking afresh") from None
    return {str(k): float(v) for k, v in (doc.get("marks") or {}).items()}


def save_marks(path: Path, marks: dict[str, float], now: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps({"schema": "research/store-gc-marks/v0.1", "written": now, "marks": marks}) + "\n")
    os.replace(tmp, path)


def list_manifests(remote: Any) -> list[str]:
    """Every key under `manifests/`, its 16 hex prefixes listed at once: one listing of the whole prefix took 90 s (26 Oct)."""
    with ThreadPoolExecutor(max_workers=16) as ex:
        return [k for keys in ex.map(remote.list, [f"manifests/{x:x}" for x in range(16)]) for k in keys]


def apply(remote: Any, p: Plan, cache: Path, state: Path, *, now: float, grace_s: float = GRACE_S, jobs: int = JOBS,
          batch: int = BATCH, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Delete the blobs `p` names that were marked at least `grace_s` before `now` and that no manifest listed since `p` holds; mark
    the rest.  The deletes go `batch` at a time, each after a fresh listing of the manifests, so a put that reuses a blob while
    the deletes run rescues it unless it lands inside one batch's deletes.  A last listing follows the last batch; a blob already
    deleted that a newly listed manifest holds, and that its put didn't upload again, is `lost`: the artifact must be put again.
    Returns {deleted, deleted_bytes, marked, rescued, lost, lost_count, listings, errors, error_count}."""
    say = log or (lambda s: None)
    marks = load_marks(state)
    due = sorted(b for b, t in marks.items() if b in p.candidates and now - t >= grace_s)
    rd = Reader(remote, cache, jobs)
    seen = set(p.listed)
    rescued: set[str] = set()
    lost: set[str] = set()
    deleted: list[str] = []
    done: set[str] = set()
    errors: list[str] = []
    listings = 0

    def fresh() -> set[str]:
        """The blobs every manifest listed for the first time now holds."""
        nonlocal listings
        new = [k for k in list_manifests(remote) if (m := MANIFEST_KEY.fullmatch(k)) and m.group(1) not in seen]
        listings += 1
        held: set[str] = set()
        for k, parsed in rd.each(new, _parse_manifest):
            seen.add(k[len("manifests/"):-len(".json")])
            held |= {d.hex() for d, _ in parsed[5]}
        return held

    def gone(b: str) -> tuple[str, BaseException | None]:
        try:
            remote.delete(f"objects/sha256/{b}")
            remote.delete(f"objects/sha256/{b}.parts.json")
            return b, None
        except Exception as e:  # noqa: BLE001  one failure must not stop the rest; it stays marked
            return b, e

    try:
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, batch))) as ex:
            i = 0
            while True:
                held = fresh()
                lost |= {b for b in held & done if remote.head(f"objects/sha256/{b}") is None}
                rescued |= (held & p.candidates.keys()) - done
                if i >= len(due):
                    break
                chunk = [b for b in due[i:i + batch] if b not in rescued]
                i += batch
                for b, e in ex.map(gone, chunk):
                    if e:
                        errors.append(f"{b}: {type(e).__name__}: {e}")
                    else:
                        deleted.append(b)
                        done.add(b)
    finally:
        save_marks(state, {b: marks.get(b, now) for b in p.candidates if b not in done and b not in rescued}, now)
    say(f"{len(seen) - len(p.listed)} manifests listed since the plan ({listings} listings); {len(rescued)} candidates they hold "
        f"are kept; {len(deleted)} of {len(due)} due blobs deleted")
    if lost:
        say(f"LOST: {len(lost)} deleted blob(s) a manifest put during the deletes holds, e.g. {sorted(lost)[0]}")
    return {"deleted": len(deleted), "deleted_bytes": sum(p.candidates[b] for b in deleted),
            "marked": len(p.candidates) - len(done) - len(rescued), "rescued": len(rescued),
            "lost": sorted(lost)[:20], "lost_count": len(lost), "listings": listings,
            "errors": errors[:20], "error_count": len(errors)}


def clock() -> float:
    return time.time()
