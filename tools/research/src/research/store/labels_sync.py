"""Labels on the remote: every assertion is pushed under a content key and pulled back by union."""
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
PACK_MIN_MISSING = 5000


def label_hash(label: Label) -> str:
    return canon.sha256(label.to_doc())


def label_key(label: Label, hex_: str | None = None) -> str:
    return f"{PREFIX}{ids.target_dirname(label.target)}/{hex_ or label_hash(label)}.json"


def parse_label_bytes(data: bytes) -> Label:
    return Label.from_doc(json.loads(data)).validate()


def named_label(data: bytes, hex_: str) -> Label:
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
    return _read_label_files(sorted((store.root / "labels").glob("*/*.json")), warn)


def local_labels_for(store: LocalStore, target: str, warn: Callable[[str], None] | None = None) -> dict[str, tuple[Path, Label]]:
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
    return _remote_listing(r, PREFIX)


def remote_labels_for(r: Remote, target: str) -> dict[str, str]:
    return _remote_listing(r, f"{PREFIX}{ids.target_dirname(target)}/")


def attempt_targets(doc: dict) -> list[str]:
    out: list[str] = [doc["id"]]
    if ids.is_drv(doc.get("derivation")):
        out.append(doc["derivation"])
    for sec in ("outputs", "telemetry"):
        for v in (doc.get(sec) or {}).values():
            if ids.is_art(v) and v not in out:
                out.append(v)
    return out


def push_targets(store: LocalStore, targets: list[str], *, jobs: int = SYNC_JOBS, log: Callable[[str], None] | None = None) -> dict:
    """Never raises for one assertion: the report carries `failed` per target."""
    r = store._require_remote()
    warn = log or (lambda s: None)
    rep: dict = {"targets": {}, "pushed": [], "already": 0, "failed": [], "local": 0}
    scratch = store.root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def push_one(item: tuple[str, str, Label, Path]) -> tuple[str, str, str | None]:
        t, h, lab, p = item
        data = canon.dumps(lab.to_doc())
        src, tmp = p, None
        try:
            if p.read_bytes() != data:
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
        if not mine:
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
    r = store._require_remote()
    todo = list(dict.fromkeys(targets))
    if jobs > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            return _pull(store, r, list(zip(todo, ex.map(lambda t: remote_labels_for(r, t), todo))), jobs=jobs)
    return _pull(store, r, ((t, remote_labels_for(r, t)) for t in todo), jobs=jobs)


def pull_prefix(store: LocalStore, prefix: str) -> dict:
    return pull_prefixes(store, [prefix])


def pull_prefixes(store: LocalStore, prefixes: list[str], *, jobs: int = 1) -> dict:
    r = store._require_remote()

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
    from .local import StoreError

    r = store.remote
    if r is None:
        if strict:
            store._require_remote()  # raises why there is no remote here
        if why := getattr(store, "remote_error", None):
            warn(f"labels of {', '.join(targets)}: the remote is unusable here ({why}); answering from the local labels")
        return None
    try:
        rep = pull_targets(store, targets)
    except Exception as e:
        if strict:
            raise StoreError(f"pulling from {r.describe()} failed ({type(e).__name__}: {e})") from e
        warn(f"labels of {', '.join(targets)}: pulling from {r.describe()} failed ({type(e).__name__}: {e}); answering from the local labels")
        return None
    for bad in rep["bad"]:
        warn(f"labels: skipped a bad remote assertion: {bad}")
    return rep


def verify_targets(store: LocalStore, targets: list[str]) -> dict:
    """Nothing is uploaded."""
    r = store._require_remote()
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
    r = store._require_remote()
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
    except Exception as e:
        raise PackUnusable(f"{type(e).__name__}: {e}") from e


def pull_pack(store: LocalStore, want: dict[str, str], *, log: Callable[[str], None] | None = None) -> list[str]:
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
    r = store._require_remote()
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
        if p.read_bytes() != data:
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
            for h, lab, err in ex.map(pull_one, todo):
                if err:
                    rep["bad"].append(err)
                    warn(f"labels-sync: {err}")
                else:
                    write_label_file(store, lab)
                    rep["pulled"].append(h)
    return rep


def import_dir(store: LocalStore, src: Path, *, warn: Callable[[str], None] | None = None) -> dict:
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
