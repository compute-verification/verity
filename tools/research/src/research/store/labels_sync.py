"""Labels on the remote (README §7/§8): every assertion file is pushed under a CONTENT key and pulled back by union.

    labels/<target dirname>/<sha256 of the canonical assertion>.json

An assertion is immutable, so its canonical bytes name it: two laptops (or a laptop and a pod) that assert the same thing at the
same second produce the same key and the second `put` is a no-op (`skipped`); two different assertions never share a key, so
nothing is ever overwritten -- there is no per-target index file to race on.  Locally the file keeps the store's own layout
(`labels/<target>/<ts>-<by>[~N].json`, O_EXCL), so `LocalStore.labels()` history order is unchanged and a pulled assertion looks
exactly like one written here.

`sync()` lists the remote once (one paginated listing, ~10^3 keys per page), hashes the local files, and pushes the difference
one way and pulls it the other; it is idempotent and needs no state in the disposable catalog.  `import_dir()` unions any
directory of assertion files (e.g. an unpacked `labels-backup/v1` tarball, or another store's `labels/`); `import_backup()` does
that straight from a `labels-backup/v1` artifact, so the interim backups stay readable.

Per target (the r20 lesson: labels asserted on pods lived only in the pod's store; `push` / `publish` moved manifests and blobs and
nothing under `labels/`, and teardown had to tar `labels/` by hand): a preserved Attempt is not preserved until its assertions are
durable too.  `attempt_targets(doc)` is the closure an attempt's durability covers -- the attempt itself, its Derivation and every
artifact it names -- and `push_targets()` pushes exactly those targets' assertions (called by `research data push RUN|ART`,
`attempt publish`, and `LocalStore.push_labels`); `verify_targets()` is the labels leg of the durability gate
(`preserved.check`): every local assertion of every target must be listed on the remote with its canonical size; `compare()` is
the local-vs-remote view behind `research data labels --remote`.  Conflicting assertions coexist on both sides (they are
assertions, not fields); the catalog shows the latest per (key, by) and `--history` everything.
"""
from __future__ import annotations

import json
import os
import re
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

from . import canon, ids
from .model import Label

if TYPE_CHECKING:
    from .local import LocalStore
    from .remote import Remote

PREFIX = "labels/"
KEY_RE = re.compile(r"^labels/([^/]+)/([0-9a-f]{64})\.json$")
SYNC_JOBS = 16
LABELS_BACKUP_KIND = "labels-backup/v1"


def label_hash(label: Label) -> str:
    """The content name of an assertion: sha256 of its canonical document."""
    return canon.sha256(label.to_doc())


def label_key(label: Label, hex_: str | None = None) -> str:
    return f"{PREFIX}{ids.target_dirname(label.target)}/{hex_ or label_hash(label)}.json"


def parse_label_bytes(data: bytes) -> Label:
    """A validated Label from an assertion file's bytes (ValueError / KeyError / JSONDecodeError on a bad file)."""
    return Label.from_doc(json.loads(data)).validate()


def _read_label_files(paths: list[Path], warn: Callable[[str], None] | None) -> dict[str, tuple[Path, Label]]:
    out: dict[str, tuple[Path, Label]] = {}
    for p in paths:
        try:
            lab = parse_label_bytes(p.read_bytes())
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            if warn:
                warn(f"skipping unreadable label {p}: {e}")
            continue
        out.setdefault(label_hash(lab), (p, lab))
    return out


def local_labels(store: LocalStore, warn: Callable[[str], None] | None = None) -> dict[str, tuple[Path, Label]]:
    """`{content hash: (path, label)}` for every readable file under labels/.  A duplicate (same assertion in two files, e.g. after an
    old-style copy) is reported once and kept once."""
    return _read_label_files(sorted((store.root / "labels").glob("*/*.json")), warn)


def local_labels_for(store: LocalStore, target: str, warn: Callable[[str], None] | None = None) -> dict[str, tuple[Path, Label]]:
    """`{content hash: (path, label)}` for one target's local assertions."""
    d = store.labels_dir(target)
    return _read_label_files(sorted(d.glob("*.json")) if d.is_dir() else [], warn)


def _remote_listing(r: Remote, prefix: str) -> dict[str, str]:
    out = {}
    for k in r.list(prefix):
        m = KEY_RE.match(k)
        if m:
            out[m.group(2)] = k
    return out


def remote_labels(r: Remote) -> dict[str, str]:
    """`{content hash: key}` for every well-formed label key on the remote (one listing)."""
    return _remote_listing(r, PREFIX)


def remote_labels_for(r: Remote, target: str) -> dict[str, str]:
    """`{content hash: key}` for one target's assertions on the remote (one listing of its prefix)."""
    return _remote_listing(r, f"{PREFIX}{ids.target_dirname(target)}/")


def attempt_targets(doc: dict) -> list[str]:
    """The label targets an attempt's durability covers: the attempt, its derivation, every artifact it names (outputs + telemetry)."""
    out: list[str] = [doc["id"]]
    if ids.is_drv(doc.get("derivation")):
        out.append(doc["derivation"])
    for sec in ("outputs", "telemetry"):
        for v in (doc.get(sec) or {}).values():
            if ids.is_art(v) and v not in out:
                out.append(v)
    return out


def push_targets(store: LocalStore, targets: list[str], *, jobs: int = SYNC_JOBS, log: Callable[[str], None] | None = None) -> dict:
    """Push every local assertion of `targets` the remote lacks, under its content key.  Never raises for one assertion: the report
    carries `failed` (`"<key>: <error>"`) per target and overall; `pushed` lists content hashes, `already` counts those the remote had.
        {"targets": {t: {"local", "pushed", "already", "failed"}}, "pushed": [...], "already": n, "failed": [...], "local": n}"""
    r = store._require_remote()  # noqa: SLF001
    warn = log or (lambda s: None)
    rep: dict = {"targets": {}, "pushed": [], "already": 0, "failed": [], "local": 0}
    scratch = store.root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def push_one(item: tuple[str, str, Label, Path]) -> tuple[str, str, str | None]:
        t, h, lab, p = item
        data = canon.dumps(lab.to_doc())
        src, tmp = p, None
        try:
            if p.read_bytes() != data:                              # a hand-formatted file goes up as the canonical bytes its name hashes
                tmp = scratch / f".label-{h}-{os.getpid()}-{threading.get_ident()}"
                tmp.write_bytes(data)
                src = tmp
            r.put(label_key(lab, h), src)
            return t, h, None
        except Exception as e:  # noqa: BLE001  one assertion's failure is reported, never raised: the target ends up NOT durable
            return t, h, f"{label_key(lab, h)}: {type(e).__name__}: {e}"
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)

    todo: list[tuple[str, str, Label, Path]] = []
    for t in dict.fromkeys(targets):
        mine = local_labels_for(store, t, warn)
        if not mine:                                                # nothing asserted here: no round trip
            rep["targets"][t] = {"local": 0, "pushed": [], "already": 0, "failed": []}
            continue
        try:
            theirs = remote_labels_for(r, t)
        except Exception as e:  # noqa: BLE001
            rep["targets"][t] = {"local": len(mine), "pushed": [], "already": 0, "failed": [f"{PREFIX}{ids.target_dirname(t)}/: list failed: {type(e).__name__}: {e}"]}
            rep["failed"] += rep["targets"][t]["failed"]
            rep["local"] += len(mine)
            continue
        have = set(mine) & set(theirs)
        rep["targets"][t] = {"local": len(mine), "pushed": [], "already": len(have), "failed": []}
        rep["already"] += len(have)
        rep["local"] += len(mine)
        todo += [(t, h, lab, p) for h, (p, lab) in sorted(mine.items()) if h not in theirs]
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(todo)))) as ex:
            for t, h, err in ex.map(push_one, todo):
                if err:
                    rep["targets"][t]["failed"].append(err)
                    rep["failed"].append(err)
                    warn(f"labels: {err}")
                else:
                    rep["targets"][t]["pushed"].append(h)
                    rep["pushed"].append(h)
    return rep


def pull_targets(store: LocalStore, targets: list[str]) -> dict:
    """Fetch the assertions about `targets` that the remote has and this store lacks (one listing per target), the reverse of
    `push_targets`.  A remote file whose content doesn't hash to its name or doesn't validate is left alone and reported in `bad`.
        {"pulled": [content hash, ...], "bad": ["<key>: <why>", ...]}"""
    r = store._require_remote()  # noqa: SLF001
    return _pull(store, r, ((t, remote_labels_for(r, t)) for t in dict.fromkeys(targets)))


def pull_prefix(store: LocalStore, prefix: str) -> dict:
    """`pull_targets` for every target whose id starts with `prefix` and has assertions on the remote, from one listing of their
    common key prefix (`pr:12@`: each head of pull request 12).
        {"pulled": [content hash, ...], "bad": ["<key>: <why>", ...], "targets": [target, ...]}"""
    r = store._require_remote()  # noqa: SLF001
    d = ids.target_dirname(prefix)
    listing: dict[str, dict[str, str]] = {}
    for k in r.list(f"{PREFIX}{d}"):
        if m := KEY_RE.match(k):
            listing.setdefault(prefix + m.group(1)[len(d):], {})[m.group(2)] = k
    return _pull(store, r, sorted(listing.items())) | {"targets": sorted(listing)}


def _pull(store: LocalStore, r: Remote, listings: Iterable[tuple[str, dict[str, str]]]) -> dict:
    """Fetch each assertion of `listings` ((target, {content hash: key}) pairs from the remote) that this store lacks."""
    rep: dict = {"pulled": [], "bad": []}
    scratch = store.root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for t, theirs in listings:
        mine = local_labels_for(store, t)
        for h, key in sorted(theirs.items()):
            if h in mine:
                continue
            tmp = scratch / f".label-pull-{h}-{os.getpid()}"
            try:
                r.get(key, tmp)
                lab = parse_label_bytes(tmp.read_bytes())
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                rep["bad"].append(f"{key}: {type(e).__name__}: {e}")
                continue
            finally:
                tmp.unlink(missing_ok=True)
            if label_hash(lab) != h or lab.target != t:
                rep["bad"].append(f"{key}: content hashes to {label_hash(lab)} for {lab.target}, not its name")
                continue
            write_label_file(store, lab)
            rep["pulled"].append(h)
    return rep


def read_through(store: LocalStore, targets: list[str], warn: Callable[[str], None], strict: bool = False) -> dict | None:
    """Before an answer from the local labels of `targets` (`research data labels`, the merge queue's grants): `pull_targets` from
    the remote.  The report, or None with a warning when the pull failed or the configured remote is unusable here, and None with
    no remote configured.  `strict` (the merge queue): no pull raises `StoreError` saying why, since the local labels alone may
    be stale."""
    from .local import StoreError

    r = store.remote
    if r is None:
        if strict:
            store._require_remote()  # noqa: SLF001  (raises why there is no remote here)
        if why := getattr(store, "remote_error", None):
            warn(f"labels of {', '.join(targets)}: the remote is unusable here ({why}); answering from the local labels")
        return None
    try:
        rep = pull_targets(store, targets)
    except Exception as e:  # noqa: BLE001  (an unreachable remote: the local labels still answer)
        if strict:
            raise StoreError(f"pulling from {r.describe()} failed ({type(e).__name__}: {e})") from e
        warn(f"labels of {', '.join(targets)}: pulling from {r.describe()} failed ({type(e).__name__}: {e}); answering from the local labels")
        return None
    for bad in rep["bad"]:
        warn(f"labels: skipped a bad remote assertion: {bad}")
    return rep


def verify_targets(store: LocalStore, targets: list[str]) -> dict:
    """The labels leg of the durability gate: for each target, every local assertion must be listed on the remote and HEAD back with
    its canonical size.  Nothing is uploaded.  Assertions that exist only on the remote count as `remote_only` (not a failure: another
    machine asserted them).
        {"durable": bool, "targets": {t: {"local", "remote", "missing": [key...], "remote_only": n, "errors": [...]}}, "missing": n}"""
    r = store._require_remote()  # noqa: SLF001
    rep: dict = {"durable": True, "targets": {}, "missing": 0}
    for t in dict.fromkeys(targets):
        mine = local_labels_for(store, t)
        tr: dict = {"local": len(mine), "remote": 0, "missing": [], "remote_only": 0, "errors": []}
        try:
            theirs = remote_labels_for(r, t)
        except Exception as e:  # noqa: BLE001
            tr["errors"].append(f"{PREFIX}{ids.target_dirname(t)}/: list failed: {type(e).__name__}: {e}")
            tr["missing"] = [label_key(lab, h) for h, (_, lab) in sorted(mine.items())]
        else:
            tr["remote"] = len(theirs)
            tr["remote_only"] = len(set(theirs) - set(mine))
            for h, (_, lab) in sorted(mine.items()):
                key = label_key(lab, h)
                if h not in theirs:
                    tr["missing"].append(key)
                    continue
                head = r.head(key)
                size = len(canon.dumps(lab.to_doc()))
                if head is None:
                    tr["missing"].append(key)
                elif head["bytes"] != size:
                    tr["errors"].append(f"{key}: remote size {head['bytes']} != {size}")
        rep["targets"][t] = tr
        rep["missing"] += len(tr["missing"])
        if tr["missing"] or tr["errors"]:
            rep["durable"] = False
    return rep


def compare(store: LocalStore, target: str) -> list[dict]:
    """Every assertion about `target` on either side, oldest first: `{"where": "both"|"local"|"remote", "hash", "label": doc}`.  Remote-only
    assertions are downloaded to be shown (and refused when their content does not hash to their name); nothing is written locally."""
    r = store._require_remote()  # noqa: SLF001
    mine = local_labels_for(store, target)
    theirs = remote_labels_for(r, target)
    rows: list[dict] = []
    for h, (_, lab) in mine.items():
        rows.append({"where": "both" if h in theirs else "local", "hash": h, "label": lab.to_doc()})
    scratch = store.root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for h in sorted(set(theirs) - set(mine)):
        tmp = scratch / f".label-cmp-{h}-{os.getpid()}"
        try:
            r.get(theirs[h], tmp)
            lab = parse_label_bytes(tmp.read_bytes())
            row = {"where": "remote", "hash": h, "label": lab.to_doc()}
            if label_hash(lab) != h:
                row["error"] = f"content hashes to {label_hash(lab)}, not its name"
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            row = {"where": "remote", "hash": h, "label": None, "error": f"{type(e).__name__}: {e}"}
        finally:
            tmp.unlink(missing_ok=True)
        rows.append(row)
    return sorted(rows, key=lambda x: ((x["label"] or {}).get("ts") or "", x["hash"]))


def write_label_file(store: LocalStore, label: Label) -> Path:
    """Land an assertion that already exists elsewhere (remote, backup, another store) in this store's own layout, and index it.
    The bytes written are the canonical document, so the file re-hashes to the same content name."""
    d = store.labels_dir(label.target)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{re.sub(r'[^0-9TZ]', '', label.ts)}-{re.sub(r'[^A-Za-z0-9._-]', '_', label.by)}"
    data = canon.dumps(label.to_doc())
    for n in range(1, 10000):
        p = d / (f"{stem}.json" if n == 1 else f"{stem}~{n}.json")
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        store.index.index_label(label, p)
        return p
    raise RuntimeError(f"could not allocate a label file under {d}")


def sync(store: LocalStore, *, push: bool = True, pull: bool = True, jobs: int = SYNC_JOBS, log: Callable[[str], None] | None = None) -> dict:
    """Union the local labels/ and the remote `labels/` prefix.  Returns `{"remote", "local", "remote_before", "pushed", "pulled",
    "already", "bad": [...]}` where `pushed` / `pulled` list the content hashes moved and `bad` the remote keys whose content did not
    hash to their name or did not validate (left alone, never written locally)."""
    r = store._require_remote()  # noqa: SLF001
    warn = log or (lambda s: None)
    mine = local_labels(store, warn)
    theirs = remote_labels(r)
    rep: dict = {"remote": r.describe(), "local": len(mine), "remote_before": len(theirs), "pushed": [], "pulled": [], "already": 0, "bad": []}
    rep["already"] = len(set(mine) & set(theirs))
    scratch = store.root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def push_one(h: str) -> str | None:
        p, lab = mine[h]
        data = canon.dumps(lab.to_doc())
        src = p
        tmp = None
        if p.read_bytes() != data:                                  # a hand-formatted file: what goes up is the canonical bytes its name hashes
            tmp = scratch / f".label-{h}-{os.getpid()}"
            tmp.write_bytes(data)
            src = tmp
        try:
            r.put(label_key(lab, h), src)
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)
        return h

    def pull_one(h: str) -> tuple[str, Label | None, str | None]:
        """Download + parse only (threads); the file write and the catalog row happen on the calling thread (sqlite is per-thread)."""
        key = theirs[h]
        tmp = scratch / f".label-pull-{h}-{os.getpid()}"
        try:
            r.get(key, tmp)
            lab = parse_label_bytes(tmp.read_bytes())
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return h, None, f"{key}: {type(e).__name__}: {e}"
        finally:
            tmp.unlink(missing_ok=True)
        if label_hash(lab) != h:
            return h, None, f"{key}: content hashes to {label_hash(lab)}, not its name"
        return h, lab, None

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        if push:
            todo = sorted(set(mine) - set(theirs))
            rep["pushed"] = [h for h in ex.map(push_one, todo) if h]
        if pull:
            todo = sorted(set(theirs) - set(mine))
            for h, lab, err in list(ex.map(pull_one, todo)):
                if err:
                    rep["bad"].append(err)
                    warn(f"labels-sync: {err}")
                else:
                    write_label_file(store, lab)
                    rep["pulled"].append(h)
    return rep


def import_dir(store: LocalStore, src: Path, *, warn: Callable[[str], None] | None = None) -> dict:
    """Union every assertion file under `src` (any depth) into this store: files whose content is already here are skipped, the rest
    land in the store's layout.  Returns {"seen", "added", "already", "bad"}."""
    mine = set(local_labels(store, warn))
    rep = {"seen": 0, "added": 0, "already": 0, "bad": 0}
    for p in sorted(Path(src).rglob("*.json")):
        rep["seen"] += 1
        try:
            lab = parse_label_bytes(p.read_bytes())
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            rep["bad"] += 1
            if warn:
                warn(f"skipping {p}: {e}")
            continue
        h = label_hash(lab)
        if h in mine:
            rep["already"] += 1
            continue
        write_label_file(store, lab)
        mine.add(h)
        rep["added"] += 1
    return rep


def import_backup(store: LocalStore, art_id: str, *, warn: Callable[[str], None] | None = None) -> dict:
    """Union the assertions of a `labels-backup/v1` artifact (a tarball of a labels/ directory) into this store."""
    m = store.get_manifest(art_id)
    if m.kind != LABELS_BACKUP_KIND:
        raise ValueError(f"{art_id} is {m.kind}, not {LABELS_BACKUP_KIND}")
    tgz = store.fetch(art_id)
    mine = set(local_labels(store, warn))
    rep = {"seen": 0, "added": 0, "already": 0, "bad": 0}
    with tarfile.open(tgz, "r:*") as tf:
        for mem in tf:
            if not mem.isfile() or not mem.name.endswith(".json"):
                continue
            rep["seen"] += 1
            f = tf.extractfile(mem)
            try:
                lab = parse_label_bytes(f.read() if f else b"")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                rep["bad"] += 1
                if warn:
                    warn(f"skipping {mem.name}: {e}")
                continue
            h = label_hash(lab)
            if h in mine:
                rep["already"] += 1
                continue
            write_label_file(store, lab)
            mine.add(h)
            rep["added"] += 1
    return rep
