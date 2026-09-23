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
"""
from __future__ import annotations

import json
import os
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TYPE_CHECKING

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


def local_labels(store: LocalStore, warn: Callable[[str], None] | None = None) -> dict[str, tuple[Path, Label]]:
    """`{content hash: (path, label)}` for every readable file under labels/.  A duplicate (same assertion in two files, e.g. after an
    old-style copy) is reported once and kept once."""
    out: dict[str, tuple[Path, Label]] = {}
    for p in sorted((store.root / "labels").glob("*/*.json")):
        try:
            lab = parse_label_bytes(p.read_bytes())
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            if warn:
                warn(f"skipping unreadable label {p}: {e}")
            continue
        out.setdefault(label_hash(lab), (p, lab))
    return out


def remote_labels(r: Remote) -> dict[str, str]:
    """`{content hash: key}` for every well-formed label key on the remote (one listing)."""
    out = {}
    for k in r.list(PREFIX):
        m = KEY_RE.match(k)
        if m:
            out[m.group(2)] = k
    return out


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
