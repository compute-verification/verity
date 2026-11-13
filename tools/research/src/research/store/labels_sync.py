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

The pack is every assertion of `labels/` as one object at a fixed key, so a machine that lacks most of them needs no request per
assertion: `PACK_KEY`, a tarball of `labels/` as the remote lays it out (one member per assertion, named by its key), beside
`PACK_INFO_KEY`, its sha256, size, count and build time.  `publish_pack()` rebuilds and replaces both, the only keys here a writer
overwrites, and a `sync()` that would pull more than `PACK_MIN_MISSING` assertions takes the pack first (`pull_pack()`) and pulls
only what it still lacks.  The pack is a cache of `labels/`, never its source: a member lands only when the remote lists its
name and its content hashes to it, the check a pulled assertion passes, and a pack that is absent, fails its sha256 or cannot be
read costs only the time it would have saved.

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

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, TYPE_CHECKING

from . import canon, ids
from .model import Label, utc_now

if TYPE_CHECKING:
    from .local import LocalStore
    from .remote import Remote

PREFIX = "labels/"
KEY_RE = re.compile(r"^labels/([^/]+)/([0-9a-f]{64})\.json$")
SYNC_JOBS = 16
LABELS_BACKUP_KIND = "labels-backup/v1"
PACK_KEY = "packs/labels/latest.tar.gz"
PACK_INFO_KEY = "packs/labels/latest.json"
PACK_SCHEMA = "research/labels-pack/v1"
PACK_MIN_MISSING = 5000         # a sync that lacks more of the remote's assertions than this takes the pack before it pulls


def label_hash(label: Label) -> str:
    """The content name of an assertion: sha256 of its canonical document."""
    return canon.sha256(label.to_doc())


def label_key(label: Label, hex_: str | None = None) -> str:
    return f"{PREFIX}{ids.target_dirname(label.target)}/{hex_ or label_hash(label)}.json"


def parse_label_bytes(data: bytes) -> Label:
    """A validated Label from an assertion file's bytes (ValueError / KeyError / JSONDecodeError on a bad file)."""
    return Label.from_doc(json.loads(data)).validate()


def named_label(data: bytes, hex_: str) -> Label:
    """`parse_label_bytes` of what the remote holds under the content name `hex_`: ValueError when it hashes to another."""
    lab = parse_label_bytes(data)
    if label_hash(lab) != hex_:
        raise ValueError(f"content hashes to {label_hash(lab)}, not its name")
    return lab


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


def pull_targets(store: LocalStore, targets: list[str], *, jobs: int = 1) -> dict:
    """Fetch the assertions about `targets` that the remote has and this store lacks (one listing per target, and the GETs, `jobs`
    at once), the reverse of `push_targets`.  A remote file whose content doesn't hash to its name or doesn't validate is left
    alone and reported in `bad`.
        {"pulled": [content hash, ...], "bad": ["<key>: <why>", ...]}"""
    r = store._require_remote()  # noqa: SLF001
    todo = list(dict.fromkeys(targets))
    if jobs > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            return _pull(store, r, list(zip(todo, ex.map(lambda t: remote_labels_for(r, t), todo))), jobs=jobs)
    return _pull(store, r, ((t, remote_labels_for(r, t)) for t in todo), jobs=jobs)


def pull_prefix(store: LocalStore, prefix: str) -> dict:
    """`pull_prefixes` of one prefix."""
    return pull_prefixes(store, [prefix])


def pull_prefixes(store: LocalStore, prefixes: list[str], *, jobs: int = 1) -> dict:
    """`pull_targets` for every target whose id starts with one of `prefixes` and has assertions on the remote, from one listing of
    each prefix's common key prefix (`pr:12@`: each head of pull request 12), and the GETs, `jobs` at once.
        {"pulled": [content hash, ...], "bad": ["<key>: <why>", ...], "targets": [target, ...]}"""
    r = store._require_remote()  # noqa: SLF001

    def listed(prefix: str) -> dict[str, dict[str, str]]:
        d, out = ids.target_dirname(prefix), {}
        for k in r.list(f"{PREFIX}{d}"):
            if m := KEY_RE.match(k):
                out.setdefault(prefix + m.group(1)[len(d):], {})[m.group(2)] = k
        return out

    todo = list(dict.fromkeys(prefixes))
    listing: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(todo)))) as ex:
        for got in ex.map(listed, todo):
            listing.update(got)
    return _pull(store, r, sorted(listing.items()), jobs=jobs) | {"targets": sorted(listing)}


def _pull(store: LocalStore, r: Remote, listings: Iterable[tuple[str, dict[str, str]]], jobs: int = 1) -> dict:
    """Fetch each assertion of `listings` ((target, {content hash: key}) pairs from the remote) that this store lacks, `jobs`
    GETs at once; the calling thread checks and writes each."""
    rep: dict = {"pulled": [], "bad": []}
    scratch = store.root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[str, str, str]] = []
    for t, theirs in listings:
        mine = local_labels_for(store, t)
        todo += [(t, h, key) for h, key in sorted(theirs.items()) if h not in mine]

    def get(item: tuple[str, str, str]) -> bytes | Exception:
        tmp = scratch / f".label-pull-{item[1]}-{os.getpid()}-{threading.get_ident()}"
        try:
            r.get(item[2], tmp)
            return tmp.read_bytes()
        except (KeyError, ValueError) as e:                  # reported as bad, as a bad file is; anything else raises
            return e
        finally:
            tmp.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(todo)))) as ex:
        for (t, h, key), data in zip(todo, ex.map(get, todo)):
            try:
                if isinstance(data, Exception):
                    raise data
                lab = parse_label_bytes(data)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                rep["bad"].append(f"{key}: {type(e).__name__}: {e}")
                continue
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


class PackUnusable(Exception):
    """The remote's labels pack can't be read, and why.  A sync then pulls each assertion, as it does with no pack."""


def write_pack(labels: dict[str, Label], dst: Path) -> dict:
    """Write the pack of `labels` (`{content hash: assertion}`) at `dst` and return its info document: a gzipped tarball with one
    member per assertion, named by its remote key and holding its canonical bytes, in key order.  The same assertions give the
    same bytes."""
    members = sorted((label_key(lab, h), h) for h, lab in labels.items())
    with open(dst, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        for key, h in members:
            data = canon.dumps(labels[h].to_doc())
            mem = tarfile.TarInfo(key)
            mem.size = len(data)
            tf.addfile(mem, io.BytesIO(data))
    with open(dst, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    return {"schema": PACK_SCHEMA, "sha256": sha, "bytes": dst.stat().st_size, "labels": len(members), "built_utc": utc_now()}


def publish_pack(store: LocalStore, *, jobs: int = SYNC_JOBS, log: Callable[[str], None] | None = None) -> dict:
    """Rebuild the labels pack and replace the remote's: pull what the remote lists and this store lacks, pack every assertion
    the remote lists, and upload the tarball and then its info, so the info never names a tarball that isn't there yet.  Nothing
    under `labels/` is written.  Returns the info document with `pulled` and `bad`, the pull's counts."""
    r = store._require_remote()
    replace = getattr(r, "replace", None)
    if replace is None:
        raise NotImplementedError(f"{r.describe()} cannot replace a key, which publishing the labels pack needs")
    theirs = remote_labels(r)
    rep = sync(store, push=False, jobs=jobs, log=log, theirs=theirs)
    labels = {h: lab for h, (_, lab) in local_labels(store).items() if h in theirs}
    scratch = store.root / "scratch"
    tmp = scratch / f".labels-pack-build-{os.getpid()}"
    try:
        info = write_pack(labels, tmp)
        replace(PACK_KEY, tmp)
        tmp.write_bytes(canon.dumps(info))
        replace(PACK_INFO_KEY, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return info | {"pulled": len(rep["pulled"]), "bad": len(rep["bad"])}


def pack_labels(r: Remote, tmp: Path, want: dict[str, str], warn: Callable[[str], None]) -> Iterator[tuple[str, Label]]:
    """Each (content hash, assertion) of the remote's labels pack that `want` (`{content hash: key}`) names, downloaded to `tmp`.
    A member counts only under a key `want` holds and with content that hashes to its name, the check a pulled assertion passes
    (`named_label`); the others are skipped, and those whose content is not their name are counted in one line to `warn`.
    Nothing with no pack on the remote; `PackUnusable` for one that fails its sha256 or can't be read."""
    try:
        if r.head(PACK_INFO_KEY) is None:
            return
        r.get(PACK_INFO_KEY, tmp)
        want_sha = json.loads(tmp.read_bytes())["sha256"]
        r.get(PACK_KEY, tmp)
        with open(tmp, "rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        if sha != want_sha:
            raise ValueError(f"its sha256 is {sha}, not the {want_sha} that {PACK_INFO_KEY} names")
        seen: set[str] = set()
        bad = 0
        with tarfile.open(tmp, "r:gz") as tf:
            for mem in tf:
                m = KEY_RE.match(mem.name)
                if not m or not mem.isfile() or want.get(m.group(2)) != mem.name or m.group(2) in seen:
                    continue
                f = tf.extractfile(mem)
                try:
                    lab = named_label(f.read() if f else b"", m.group(2))
                except (KeyError, ValueError, json.JSONDecodeError):
                    bad += 1
                    continue
                seen.add(m.group(2))
                yield m.group(2), lab
        if bad:
            warn(f"labels-sync: {bad} member(s) of the labels pack {PACK_KEY} hold content that is not their name: not written")
    except Exception as e:                                          # the pack is a cache: whatever is wrong with it, the pull still brings every assertion
        raise PackUnusable(f"{type(e).__name__}: {e}") from e


def pull_pack(store: LocalStore, want: dict[str, str], *, log: Callable[[str], None] | None = None) -> list[str]:
    """Land the assertions of `want` (`{content hash: key}`: what the remote lists and this store lacks) that the remote's labels
    pack holds (`pack_labels`), and return their content hashes.  A pack that can't be used is ignored with one line to `log`;
    what it gave before failing stays."""
    warn = log or (lambda s: None)
    scratch = store.root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / f".labels-pack-{os.getpid()}"
    got: list[str] = []
    try:
        for h, lab in pack_labels(store._require_remote(), tmp, want, warn):
            write_label_file(store, lab)
            got.append(h)
    except PackUnusable as e:
        warn(f"labels-sync: ignoring the labels pack {PACK_KEY} ({e}); pulling each assertion it would have given")
    finally:
        tmp.unlink(missing_ok=True)
    return got


def sync(store: LocalStore, *, push: bool = True, pull: bool = True, jobs: int = SYNC_JOBS, log: Callable[[str], None] | None = None,
         theirs: dict[str, str] | None = None) -> dict:
    """Union the local labels/ and the remote `labels/` prefix (`theirs`: its listing, `remote_labels`, when the caller has it).
    Returns `{"remote", "local", "remote_before", "pushed", "pulled", "pack", "already", "bad": [...]}` where `pushed` / `pulled`
    list the content hashes moved, `pack` counts those of `pulled` that the labels pack gave (a pull of more than
    `PACK_MIN_MISSING` takes it first, `pull_pack`) and `bad` lists the remote keys whose content did not hash to their name or
    did not validate (left alone, never written locally)."""
    r = store._require_remote()  # noqa: SLF001
    warn = log or (lambda s: None)
    mine = local_labels(store, warn)
    theirs = remote_labels(r) if theirs is None else theirs
    rep: dict = {"remote": r.describe(), "local": len(mine), "remote_before": len(theirs), "pushed": [], "pulled": [], "pack": 0, "already": 0,
                 "bad": []}
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
            return h, named_label(tmp.read_bytes(), h), None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return h, None, f"{key}: {type(e).__name__}: {e}"
        finally:
            tmp.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        if push:
            todo = sorted(set(mine) - set(theirs))
            rep["pushed"] = [h for h in ex.map(push_one, todo) if h]
        if pull:
            todo = sorted(set(theirs) - set(mine))
            if len(todo) > PACK_MIN_MISSING:
                rep["pulled"] = pull_pack(store, {h: theirs[h] for h in todo}, log=log)
                rep["pack"] = len(rep["pulled"])
                todo = sorted(set(todo) - set(rep["pulled"]))
            for h, lab, err in ex.map(pull_one, todo):                  # written as they arrive: a sync cut short keeps what it pulled
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
