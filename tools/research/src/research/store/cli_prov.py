"""`research data lookup | pull | attempt` -- the provenance commands of the store CLI (README §9).

    research data lookup --tool NAME [--param k=v]... [--input name=art:...]... [--condition k=v]... [--source-tree SHA] [--any-source] [-- ARGV...]
    research data pull --from MACHINE RUN --project P            copy an attempt + its outputs' manifests and blobs from a machine's store (ssh)
    research data attempt publish RUN_DIR|RUN_ID [--tool SPEC] [--force] | --all [--runs-dir D]
    research data custody RUN_DIR|RUN_ID [--publish [--force [--runner-gone]]] | --triage [--verify] [--older-min N] [--runs-dir D ...]   run custody on the remote (store.custody)

`research.store.cli` (lane store-core) owns the `research data` dispatcher and merges COMMANDS into its table; `main` here is the
same dispatch for tests and for `python -m research.store.cli_prov`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable

from .. import env as E
from .. import remote
from .. import runs as R
from . import attempt as A
from . import derivation as D
from . import tools_registry


def _kv(items: list[str], what: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"{what} wants KEY=VALUE, got {it!r}")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def _lockfiles_at(source_root: Path, tree: str) -> dict[str, str]:
    """sha256 of each lockfile AS IT IS IN `tree` (git object), so --source-tree names exactly what a run at that tree recorded."""
    out = {}
    for name in E.LOCKFILES:
        try:
            r = subprocess.run(["git", "cat-file", "-p", f"{tree}:{name}"], cwd=source_root, capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            out[name] = hashlib.sha256(r.stdout).hexdigest()
    return out


def _fmt_attempt(row: dict[str, Any]) -> str:
    st = row.get("status") or {}
    ex = row.get("execution") or {}
    outs = row.get("outputs") or {}
    return (f"{row.get('id', '?'):<24} {str(st.get('state')):<9} rc={str(st.get('rc')):<3} val={str(st.get('validation')):<8} "
            f"{str(ex.get('start_utc'))[:19]:<19} {str(ex.get('host')):<16} result={outs.get('result') or '-'}")


def _rows(rows: Any) -> list[dict[str, Any]]:
    out = []
    for r in rows or []:
        if isinstance(r, dict) and "manifest_json" in r and isinstance(r["manifest_json"], str):
            try:
                out.append(json.loads(r["manifest_json"]))
                continue
            except ValueError:
                pass
        out.append(r if isinstance(r, dict) else {"id": str(r)})
    return out


def lookup(tool: Any, root: Path, *, argv=None, params=None, inputs=None, conditions=None, source_tree=None, any_source=False, store=None) -> dict[str, Any]:
    """`research data lookup --json` as a call on a loaded `tool`: its attempts oldest first, `params` overriding what `argv` parses to and
    missing key conditions from this machine.  ValueError when `root`'s tree is dirty and neither `source_tree` nor `any_source` is given."""
    parsed, key = tool.resolve_params(argv or [], root) if (argv or tool.parse) else ({}, {})
    if params:
        key = tool.resolve_key_params({**parsed, **params}, root)
    conds = dict(conditions or {})
    missing = [k for k in tool.key_conditions if k not in conds]
    if missing:
        here = E.conditions(source_root=root)
        here.update(tool.tool_conditions(here))          # the Tool's own probe (`tool.<name>.*`), as the runner records it
        conds.update({k: here.get(k) for k in missing})
    drv = D.derivation_id(tool.identity(), inputs or {}, key, conds, tool.key_conditions)
    if any_source:
        rk = None
    elif source_tree:
        rk = D.reuse_key(drv, {"tree": source_tree, "dirty": False, "lockfiles": _lockfiles_at(root, source_tree)})
    else:
        src = E.source_identity(E.source_conditions(root))
        if src.get("dirty") is not False or not src.get("tree"):
            raise ValueError(f"{root} is {'dirty' if src.get('dirty') else 'not a clean git tree'}: no reuse_key; give --source-tree SHA or --any-source")
        rk = D.reuse_key(drv, src)
    index = A.open_store(store).index
    rows = _rows(index.attempts_by_derivation(drv) if any_source else index.attempts_by_reuse_key(rk))
    return {"derivation": drv, "reuse_key": rk, "key_params": key, "key_conditions": {k: conds.get(k) for k in tool.key_conditions},
            "inputs": inputs or {}, "attempts": rows}


def cmd_lookup(argv: list[str]) -> int:
    tail: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="research data lookup", description="attempts of one request: same tool, inputs, key params, key conditions and (unless --any-source) the same source tree")
    ap.add_argument("--tool", required=True, help="registry name or dotted.module:NAME")
    ap.add_argument("--param", action="append", default=[], metavar="K=V", help="key param (JSON value or string); overrides what ARGV after -- parses to")
    ap.add_argument("--input", action="append", default=[], metavar="NAME=art:...")
    ap.add_argument("--condition", action="append", default=[], metavar="K=V", help="key condition; missing key conditions come from this machine's conditions()")
    ap.add_argument("--source-tree", default=None, metavar="SHA", help="git tree sha (default: the current tree in --source-root)")
    ap.add_argument("--any-source", action="store_true", help="match on the derivation id across source revisions")
    ap.add_argument("--source-root", default=None, help="repo root for path params and the current tree (default: cwd)")
    ap.add_argument("--store", default=None, help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.source_root or os.getcwd())
    try:
        tool = tools_registry.load(ns.tool, root)
    except (ImportError, LookupError, TypeError) as e:
        print(f"research data lookup: --tool {ns.tool}: {e}", file=sys.stderr)
        return 2
    try:
        found = lookup(tool, root, argv=tail, params=_kv(ns.param, "--param"), inputs=_kv(ns.input, "--input"),
                       conditions=_kv(ns.condition, "--condition"), source_tree=ns.source_tree, any_source=ns.any_source, store=A.open_store(None, ns.store))
    except ValueError as e:
        print(f"research data lookup: {e}", file=sys.stderr)
        return 1
    if ns.json:
        print(json.dumps(found, indent=2, sort_keys=True, default=str))
        return 0
    rows = found["attempts"]
    print(f"derivation {found['derivation']}")
    print(f"reuse_key  {found['reuse_key'] or '- (any source)'}")
    print(f"key params {json.dumps(found['key_params'], sort_keys=True, default=str)}   key conditions {json.dumps(found['key_conditions'], sort_keys=True)}")
    if not rows:
        print("no attempts")
        return 0
    for row in rows:
        print(_fmt_attempt(row))
    print(f"{len(rows)} attempt(s)")
    return 0


# ---- pull -------------------------------------------------------------------------------------------------------------------------------------

def _cat(ssh: list[str], path: str, *, timeout: float = 600) -> bytes:
    r = remote.ssh_run(ssh, ["cat", path], timeout=timeout)
    if r.returncode != 0:
        raise remote.Refused(f"cannot read {path}: {r.stderr.decode(errors='replace').strip()[-300:] or 'rc ' + str(r.returncode)}")
    return r.stdout


#: One tar stream carries at most this many blobs / this many bytes (by the manifest's sizes): a run-files tree of 10^4 per-proof
#: files is a handful of round trips instead of one ssh argv of 10^4 paths (which exceeds the remote shell's argument limit).
PULL_BATCH_FILES = 4000
PULL_BATCH_BYTES = 512 * 2**20


def _batches(hexes: list[str], sizes: dict[str, int] | None) -> list[list[str]]:
    out: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for h in hexes:
        n = (sizes or {}).get(h, 0)
        if cur and (len(cur) >= PULL_BATCH_FILES or cur_bytes + n > PULL_BATCH_BYTES):
            out.append(cur)
            cur, cur_bytes = [], 0
        cur.append(h)
        cur_bytes += n
    if cur:
        out.append(cur)
    return out


def _fetch_blobs(ssh: list[str], store_root: str, hexes: list[str], into: Path, *, sizes: dict[str, int] | None = None,
                 have: Callable[[str], Path | None] | None = None) -> dict[str, int]:
    """Blobs from the machine's store into `into/<hex>`, each verified by sha256 on arrival.  The paths travel on tar's stdin (`-T -`),
    never in the ssh argv, and in batches of `PULL_BATCH_FILES` / `PULL_BATCH_BYTES` (`_batches`).  `have(hex)` -> a local path for a blob
    this store already holds: those are linked/copied locally and not fetched.  Returns {"fetched", "reused", "batches"}."""
    stats = {"fetched": 0, "reused": 0, "batches": 0}
    todo = []
    for h in hexes:
        p = have(h) if have else None
        if p is not None and p.is_file():
            try:
                os.link(p, into / h)
            except OSError:
                shutil.copyfile(p, into / h)
            stats["reused"] += 1
        else:
            todo.append(h)
    for batch in _batches(todo, sizes):
        stats["batches"] += 1
        listing = "".join(f"objects/sha256/{h}\n" for h in batch).encode()
        r = remote.ssh_run(ssh, ["tar", "-cf", "-", "-C", store_root, "-T", "-"], stdin=listing, timeout=3600)
        if r.returncode != 0:
            raise remote.Refused(f"fetching {len(batch)} blob(s) failed: {r.stderr.decode(errors='replace').strip()[-300:]}")
        wanted = set(batch)
        with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r|") as tf:
            for m in tf:
                h = Path(m.name).name
                if not m.isfile() or h not in wanted or not m.name.lstrip("./").startswith("objects/sha256/"):
                    continue                               # (bsdtar's ._* AppleDouble entries and anything else unrequested)
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                if hashlib.sha256(data).hexdigest() != h:
                    raise remote.Refused(f"blob {h} arrived with a different sha256; refusing to store it")
                (into / h).write_bytes(data)
                stats["fetched"] += 1
    missing = [h for h in hexes if not (into / h).exists()]
    if missing:
        raise remote.Refused(f"blob(s) not on the machine: {', '.join(missing[:5])}" + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))
    return stats


def pull_artifact(ssh: list[str], store_root: str, art_id: str, st: Any) -> dict[str, Any]:
    """manifests/<hex>.json + every blob it references -> the local store through put_artifact; the recomputed id must equal art_id."""
    hexid = art_id.removeprefix("art:")
    mbytes = _cat(ssh, f"{store_root}/manifests/{hexid}.json")
    if hashlib.sha256(mbytes).hexdigest() != hexid:
        raise remote.Refused(f"manifest {art_id} arrived with a different sha256")
    manifest = json.loads(mbytes)
    payload = manifest.get("payload")
    obj = getattr(st, "object_path", None)                 # a blob this store already holds is not fetched again (FakeStore has no objects/)
    have = (lambda h: obj(h)) if callable(obj) else None
    stats = {"fetched": 0, "reused": 0, "batches": 0}
    scratch = Path(st.root) / "scratch" if getattr(st, "root", None) and (Path(st.root) / "scratch").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="research-pull-", dir=scratch) as tmp:   # the store's scratch/: same filesystem as objects/, so links, not copies
        blobs = Path(tmp) / "blobs"
        blobs.mkdir()
        stage = Path(tmp) / "stage"
        stage.mkdir()
        kw: dict[str, Any] = {}
        if payload is None:
            n = 0
        elif payload.get("type") == "file":
            stats = _fetch_blobs(ssh, store_root, [payload["sha256"]], blobs, sizes={payload["sha256"]: payload.get("bytes") or 0}, have=have)
            dst = stage / (payload.get("name") or payload["sha256"])
            shutil.move(str(blobs / payload["sha256"]), dst)
            kw["file"] = dst
            n = 1
        elif payload.get("type") == "tree":
            files = payload.get("files") or []
            stats = _fetch_blobs(ssh, store_root, sorted({f["sha256"] for f in files}), blobs, sizes={f["sha256"]: f.get("bytes") or 0 for f in files}, have=have)
            for f in files:
                rel = Path(f["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise remote.Refused(f"tree path {f['path']!r} escapes the tree")
                dst = stage / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(blobs / f["sha256"], dst)      # the staged tree is the artifact's own layout; a link per file, a copy only across filesystems
                except OSError:
                    shutil.copyfile(blobs / f["sha256"], dst)
            kw["tree"] = stage
            n = len(files)
        else:
            raise remote.Refused(f"unknown payload type {payload.get('type')!r} in {art_id}")
        new_id = st.put_artifact(manifest["kind"], manifest.get("meta") or {}, manifest.get("refs") or None, **kw)
    if new_id != art_id:
        raise remote.Refused(f"{art_id} re-derived here as {new_id}: the two stores disagree on the manifest; nothing was recorded under {art_id}")
    return {"id": art_id, "kind": manifest["kind"], "blobs": n, **stats}


def pull_labels(ssh: list[str], store_root: str, targets: list[str], st: Any) -> dict[str, Any]:
    """The machine's `labels/<target>/` directories for `targets`, in one tar (the same `tar -cf - -C ROOT -T -` form as the blobs, paths on
    stdin), unioned into this store (`labels_sync.import_dir`: content names, nothing overwritten).  A target the machine never labelled
    is a directory tar cannot stat -- a warning and a non-zero rc with a complete archive on stdout, so the archive's parse, not the rc,
    decides: an unparseable tar is a transport failure.  A store without a labels layout (a test double) is skipped."""
    from . import ids
    from . import labels_sync as LS

    if not hasattr(st, "labels_dir"):
        return {"seen": 0, "added": 0, "already": 0, "bad": 0, "skipped": "store has no labels layout"}
    listing = "".join(f"labels/{ids.target_dirname(t)}\n" for t in dict.fromkeys(targets)).encode()
    r = remote.ssh_run(ssh, ["tar", "-cf", "-", "-C", store_root, "-T", "-"], stdin=listing, timeout=600)
    with tempfile.TemporaryDirectory(prefix="research-pull-labels-") as tmp:
        if r.stdout.strip(b"\0"):
            try:
                with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:") as tf:
                    members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".json") and not Path(m.name).name.startswith("._")   # (bsdtar AppleDouble)
                               and ".." not in Path(m.name).parts and not Path(m.name).is_absolute()]
                    tf.extractall(tmp, members=members, filter="data")
            except tarfile.TarError as e:
                raise remote.Refused(f"fetching labels failed (rc {r.returncode}): {e}: {r.stderr.decode(errors='replace').strip()[-300:]}") from None
        elif r.returncode not in (0, 1, 2):
            raise remote.Refused(f"fetching labels failed (rc {r.returncode}): {r.stderr.decode(errors='replace').strip()[-300:]}")
        return LS.import_dir(st, Path(tmp), warn=lambda s: print(f"research data pull: {s}", file=sys.stderr))


def cmd_pull(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research data pull", description="copy an attempt, its outputs and the assertions about them from a machine's local store into this one")
    ap.add_argument("run", metavar="RUN")
    ap.add_argument("--from", dest="machine", required=True, metavar="MACHINE")
    ap.add_argument("--project", default=os.environ.get("RESEARCH_PROJECT"), help="the project the machine is bound to (default $RESEARCH_PROJECT)")
    ap.add_argument("--store", default=None, help="local store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    try:
        m = remote.resolve_machine(ns.machine, ns.project)
        tgt = remote.pod_target(m)
        ssh = remote.ssh_argv(tgt["host"], tgt["port"], user=tgt.get("user", "root"))
        store_root = f"{m['root'].rstrip('/')}/store"
        st = A.open_store(None, ns.store)
        doc = json.loads(_cat(ssh, f"{store_root}/attempts/{ns.run}.json"))
        pulled = []
        for name, art in sorted({**(doc.get("outputs") or {}), **(doc.get("telemetry") or {})}.items()):
            if art:
                pulled.append({"name": name, **pull_artifact(ssh, store_root, art, st)})
        existing = st.get_attempt(doc["id"])
        if existing is None:
            st.put_attempt(doc)
        elif existing != doc:
            raise remote.Refused(f"attempt {doc['id']} already exists here with different content; not overwritten")
        from . import labels_sync as LS                             # the assertions the pod made about the attempt, its derivation, its outputs

        labels = pull_labels(ssh, store_root, LS.attempt_targets(doc), st)
    except (remote.Refused, ValueError, KeyError, OSError) as e:
        print(f"research data pull: {e}", file=sys.stderr)
        return 1
    if ns.json:
        print(json.dumps({"attempt": doc["id"], "machine": ns.machine, "pulled": pulled, "labels": labels}, indent=2, sort_keys=True))
    else:
        print(f"pulled attempt {doc['id']} from {ns.machine}: " + (", ".join(f"{p['name']}={p['id']} ({p['blobs']} blob{'s' if p['blobs'] != 1 else ''})" for p in pulled) or "no outputs")
              + f"; labels: {labels['added']} added, {labels['already']} already here" + (f", {labels['bad']} unreadable" if labels["bad"] else ""))
    return 0


# ---- attempt ----------------------------------------------------------------------------------------------------------------------------------

def _publish_line(out: dict[str, Any], where: str) -> str:
    if out.get("published"):
        return f"{where}: published {out['n_outputs']} output(s) drv={out['derivation']} reuse={out['reuse_key'] or '-'}" + ("  (already)" if out.get("already") else "")
    return f"{where}: NOT published: {out.get('error')}"


def cmd_attempt(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research data attempt")
    sub = ap.add_subparsers(dest="verb", required=True)
    pub = sub.add_parser("publish", help="what `research run` calls at the end; usable after the fact on any finished run dir")
    pub.add_argument("run", nargs="?", metavar="RUN_DIR|RUN_ID", help="run directory, id, or unique prefix under --runs-dir")
    pub.add_argument("--all", action="store_true", help="every finished run under --runs-dir not yet in the store")
    pub.add_argument("--tool", default=None, metavar="NAME|module:NAME", help="declaration to publish under (default: job.json's `store.tool`, else argv[0])")
    pub.add_argument("--runs-dir", default=None)
    pub.add_argument("--store", default=None)
    pub.add_argument("--force", action="store_true", help="publish even without a terminal status.json")
    pub.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if ns.verb != "publish":
        return 2
    if not ns.run and not ns.all:
        pub.error("RUN_DIR or --all")
    root = R.runs_root(ns.runs_dir)
    try:
        st = A.open_store(None, ns.store)
    except ImportError as e:
        print(f"research data attempt publish: the local store is not available ({e})", file=sys.stderr)
        return 1
    if ns.tool:
        try:
            tool: Any = tools_registry.load(ns.tool)
        except (ImportError, LookupError, TypeError) as e:
            print(f"research data attempt publish: --tool {ns.tool}: {e}", file=sys.stderr)
            return 2
    else:
        tool = None
    dirs: list[Path] = []
    if ns.run:
        try:
            dirs.append(R.resolve(root, ns.run))
        except (FileNotFoundError, LookupError) as e:
            print(f"research data attempt publish: {e}", file=sys.stderr)
            return 1
    if ns.all:
        dirs += [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "job.json").exists() and p not in dirs] if root.is_dir() else []
    results = []
    worst = 0
    for d in dirs:
        out = A.publish(d, tool=tool, store=st, force=ns.force)
        results.append({"run_dir": str(d), **out})
        if not out.get("published"):
            worst = 1
        if not ns.json:
            print(_publish_line(out, d.name))
    if ns.json:
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
    if ns.all and not ns.json:
        c = publish_counts(results)
        print(f"{len(results)} run dir(s): {c['published']} published, {c['already']} already present, {c['refused']} refused (not terminal), {c['errors']} error(s)")
        for cls, n in sorted(c["error_classes"].items(), key=lambda kv: -kv[1]):
            print(f"  {n} x {cls}")
    return worst if not ns.all else 0


def publish_counts(results: list[dict[str, Any]]) -> dict[str, Any]:
    """published / already present / refused (run not terminal) / errors, with the error text per class (run ids elided)."""
    c: dict[str, Any] = {"published": 0, "already": 0, "refused": 0, "errors": 0, "error_classes": {}}
    for r in results:
        if r.get("published"):
            c["already" if r.get("already") else "published"] += 1
            continue
        err = str(r.get("error") or "")
        if "the store only sees completed or failed attempts" in err:
            c["refused"] += 1
            continue
        c["errors"] += 1
        cls = re.sub(r"r\d{8}-\d{6}-[0-9a-f]{4}", "<run>", err)
        cls = re.sub(r"/[^\s:]+/(runs|store)/", "…/\\1/", cls)
        c["error_classes"][cls] = c["error_classes"].get(cls, 0) + 1
    return c


def cmd_custody(argv: list[str]) -> int:
    """`research data custody RUN [--publish]` | `--triage` (store.custody)."""
    from . import custody as CU

    ap = argparse.ArgumentParser(prog="research data custody", description="a run's custody on the store's remote: the attempt and every run file "
                                                                           "(logs included) preserved there; answers the same on every machine")
    ap.add_argument("run", nargs="?", metavar="RUN_DIR|RUN_ID", help="run directory, id, or unique prefix under --runs-dir")
    ap.add_argument("--publish", action="store_true", help="publish the attempt with its run record first (when not yet in the store) and push it; "
                                                         "once custody holds, the run dir's files of 1 MiB and more are removed (store.evict.evict_custody)")
    ap.add_argument("--keep-local", action="store_true", help="--publish: keep every local run file")
    ap.add_argument("--force", action="store_true", help="--publish: publish a run whose status.json is not terminal (its runner was "
                                                       "killed before it recorded an end); the attempt keeps the state it last wrote. "
                                                       "Refused (exit 3) while this host can't see that its runner is gone")
    ap.add_argument("--runner-gone", action="store_true", help="--force: the run's own host says its runner is gone (a pod: `kubectl get "
                                                             "pod` NotFound; another machine: its pid gone there), which this host can "
                                                             "see only of a runner that ran here; quiet records are no such evidence")
    ap.add_argument("--triage", action="store_true", help="every run under --runs-dir by custody state; exit 1 while any needs action, "
                                                        "3 when triage itself fails")
    ap.add_argument("--verify", action="store_true", help="--triage: check preservation on the remote (default: the catalog's recorded replicas, "
                                                        "else the run dir's .custody)")
    ap.add_argument("--older-min", type=float, default=None, metavar="N",
                    help="--triage: a run needs action only once its records last changed N minutes ago; younger ones are listed as settling")
    ap.add_argument("--runs-dir", action="append", default=None, help="a runs root; --triage takes several (repeat it)")
    ap.add_argument("--store", default=None)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if bool(ns.run) == bool(ns.triage):
        ap.error("RUN or --triage")
    if ns.run and len(ns.runs_dir or []) > 1:
        ap.error("RUN takes one --runs-dir")
    if ns.older_min is not None and (not ns.triage or ns.older_min < 0):
        ap.error("--older-min N (N >= 0) is for --triage")
    if ns.force and not ns.publish:
        ap.error("--force is for --publish")
    if ns.runner_gone and not ns.force:
        ap.error("--runner-gone is for --force")
    roots = [R.runs_root(x) for x in ns.runs_dir] if ns.runs_dir else [R.runs_root(None)]
    if ns.triage:
        missing = [str(r) for r in roots if not r.is_dir()]
        if missing:
            print(f"research data custody --triage: no runs root at {', '.join(missing)}", file=sys.stderr)
            return 2
        try:
            rep = CU.triage_report(A.open_store(None, ns.store), roots, older_min=ns.older_min, verify_remote=ns.verify)
        except Exception as e:  # noqa: BLE001  exit 1 is a finding, so a monitor must tell a crash from one
            print(f"research data custody --triage: crashed, nothing triaged: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        todo = [r for r in rep["rows"] if r["state"] in CU.NEEDS_ACTION and not r["settling"]]
        if ns.json:
            print(json.dumps(rep, indent=2, sort_keys=True))
        else:
            print(f"{len(rep['rows'])} run(s) under {', '.join(rep['runs_roots'])}: "
                  + ", ".join(f"{n} {s}" for s, n in rep["counts"].items() if n))
            for r in todo:
                print(f"  {r['run']}  {r['state']:<20} {r['bytes'] / 2**20:8.1f} MB  {r['why']}  -> {r['next']}")
            if rep["settling"]:
                print(f"{rep['settling']} more ended under {ns.older_min:g} min ago (settling: their custody may still be running)")
            print(f"{len(todo)} need action (publish, fetch, or `research data label RUN custody waived|never-ran --by WHO`)" if todo
                  else "nothing needs action")
        return 1 if todo else 0
    st = A.open_store(None, ns.store)
    root = roots[0]
    try:
        d = R.resolve(root, ns.run) if not Path(ns.run).is_dir() else Path(ns.run)
    except (FileNotFoundError, LookupError) as e:
        print(f"research data custody: {e}", file=sys.stderr)
        return 1
    pushed: list[str] | None = None
    if ns.publish:
        if ns.force and (why := CU.force_refusal(d, runner_gone=ns.runner_gone)):
            print(f"research data custody: {d.name}: --force refused: {why}", file=sys.stderr)
            return 3
        out = A.publish(d, store=st, remote_push=False, run_record=True, force=ns.force)
        if not out.get("published"):
            print(f"research data custody: {d.name}: attempt not published: {out.get('error')}", file=sys.stderr)
            return 1
        pushed = CU.push_run(st, CU.run_id_of(d))
    rep = CU.verify(st, d)
    if pushed:
        rep["push_errors"] = pushed
    if ns.publish and not rep["preserved"] and CU.marker_problem(d) is not None:
        CU.mark_failed(d, rep)
    if ns.publish and rep["preserved"] and not ns.keep_local:
        from .evict import evict_custody

        rep["evicted"] = evict_custody(d, rep)
    if ns.json:
        print(json.dumps(rep, indent=2, sort_keys=True, default=str))
    elif rep["preserved"]:
        print(f"{rep['run_id']}: custody on {rep.get('remote')}: attempt + run record {rep['run_record']} ({len(rep['files'])} file(s)); .custody written")
        if (ev := rep.get("evicted")) and ev["files"]:
            print(f"{rep['run_id']}: removed {len(ev['files'])} local file(s), {ev['evicted_bytes'] / 1e6:.1f} MB, preserved in {ev['artifact']} "
                  f"(`research data fetch {ev['artifact']} --to {d}` restores them; --keep-local keeps them)")
    else:
        print(f"{rep['run_id']}: NO custody on the remote: {rep['error']}" + (f"; push: {pushed[0]}" if pushed else ""))
    return 0 if rep["preserved"] else 1


COMMANDS = {"lookup": cmd_lookup, "pull": cmd_pull, "attempt": cmd_attempt, "custody": cmd_custody}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
