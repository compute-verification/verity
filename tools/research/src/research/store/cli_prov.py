"""`research data lookup | pull | attempt` -- the provenance commands of the store CLI (README §9).

    research data lookup --tool NAME [--param k=v]... [--input name=art:...]... [--condition k=v]... [--source-tree SHA] [--any-source] [-- ARGV...]
    research data pull --from MACHINE RUN --project P            copy an attempt + its outputs' manifests and blobs from a machine's store (ssh)
    research data attempt publish RUN_DIR|RUN_ID [--tool SPEC] [--force] | --all [--runs-dir D]

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
from typing import Any

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
    params, key = tool.resolve_params(tail, root) if (tail or tool.parse) else ({}, {})
    overrides = _kv(ns.param, "--param")
    if overrides:
        merged = {**params, **overrides}
        key = tool.resolve_key_params(merged, root)
    inputs = _kv(ns.input, "--input")
    conds = _kv(ns.condition, "--condition")
    missing = [k for k in tool.key_conditions if k not in conds]
    if missing:
        here = E.conditions(source_root=root)
        conds.update({k: here.get(k) for k in missing})
    drv = D.derivation_id(tool.identity(), inputs, key, conds, tool.key_conditions)
    st = A.open_store(None, ns.store)
    rk = None
    if ns.any_source:
        rows = _rows(st.index.attempts_by_derivation(drv))
    else:
        if ns.source_tree:
            src = {"tree": ns.source_tree, "dirty": False, "lockfiles": _lockfiles_at(root, ns.source_tree)}
        else:
            src = E.source_identity(E.source_conditions(root))
            if src.get("dirty") is not False or not src.get("tree"):
                print(f"research data lookup: {root} is {'dirty' if src.get('dirty') else 'not a clean git tree'}: no reuse_key; give --source-tree SHA or --any-source",
                      file=sys.stderr)
                return 1
        rk = D.reuse_key(drv, src)
        rows = _rows(st.index.attempts_by_reuse_key(rk))
    if ns.json:
        print(json.dumps({"derivation": drv, "reuse_key": rk, "key_params": key, "key_conditions": {k: conds.get(k) for k in tool.key_conditions},
                          "inputs": inputs, "attempts": rows}, indent=2, sort_keys=True, default=str))
        return 0
    print(f"derivation {drv}")
    print(f"reuse_key  {rk or '- (any source)'}")
    print(f"key params {json.dumps(key, sort_keys=True, default=str)}   key conditions {json.dumps({k: conds.get(k) for k in tool.key_conditions}, sort_keys=True)}")
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


def _fetch_blobs(ssh: list[str], store_root: str, hexes: list[str], into: Path) -> None:
    """One ssh+tar for all blobs; each verified by sha256 on arrival."""
    if not hexes:
        return
    r = remote.ssh_run(ssh, ["tar", "-cf", "-", "-C", store_root, *[f"objects/sha256/{h}" for h in hexes]], timeout=3600)
    if r.returncode != 0:
        raise remote.Refused(f"fetching {len(hexes)} blob(s) failed: {r.stderr.decode(errors='replace').strip()[-300:]}")
    wanted = set(hexes)
    with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:") as tf:
        for m in tf.getmembers():
            h = Path(m.name).name
            if not m.isfile() or h not in wanted or not m.name.startswith("objects/sha256/"):
                continue                                   # (bsdtar's ._* AppleDouble entries and anything else unrequested)
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            if hashlib.sha256(data).hexdigest() != h:
                raise remote.Refused(f"blob {h} arrived with a different sha256; refusing to store it")
            (into / h).write_bytes(data)
    missing = [h for h in hexes if not (into / h).exists()]
    if missing:
        raise remote.Refused(f"blob(s) not on the machine: {', '.join(missing)}")


def pull_artifact(ssh: list[str], store_root: str, art_id: str, st: Any) -> dict[str, Any]:
    """manifests/<hex>.json + every blob it references -> the local store through put_artifact; the recomputed id must equal art_id."""
    hexid = art_id.removeprefix("art:")
    mbytes = _cat(ssh, f"{store_root}/manifests/{hexid}.json")
    if hashlib.sha256(mbytes).hexdigest() != hexid:
        raise remote.Refused(f"manifest {art_id} arrived with a different sha256")
    manifest = json.loads(mbytes)
    payload = manifest.get("payload")
    with tempfile.TemporaryDirectory(prefix="research-pull-") as tmp:
        blobs = Path(tmp) / "blobs"
        blobs.mkdir()
        stage = Path(tmp) / "stage"
        stage.mkdir()
        kw: dict[str, Any] = {}
        if payload is None:
            n = 0
        elif payload.get("type") == "file":
            _fetch_blobs(ssh, store_root, [payload["sha256"]], blobs)
            dst = stage / (payload.get("name") or payload["sha256"])
            shutil.move(str(blobs / payload["sha256"]), dst)
            kw["file"] = dst
            n = 1
        elif payload.get("type") == "tree":
            files = payload.get("files") or []
            _fetch_blobs(ssh, store_root, sorted({f["sha256"] for f in files}), blobs)
            for f in files:
                rel = Path(f["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise remote.Refused(f"tree path {f['path']!r} escapes the tree")
                dst = stage / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(blobs / f["sha256"], dst)
            kw["tree"] = stage
            n = len(files)
        else:
            raise remote.Refused(f"unknown payload type {payload.get('type')!r} in {art_id}")
        new_id = st.put_artifact(manifest["kind"], manifest.get("meta") or {}, manifest.get("refs") or None, **kw)
    if new_id != art_id:
        raise remote.Refused(f"{art_id} re-derived here as {new_id}: the two stores disagree on the manifest; nothing was recorded under {art_id}")
    return {"id": art_id, "kind": manifest["kind"], "blobs": n}


def cmd_pull(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research data pull", description="copy an attempt and its outputs from a machine's local store into this one")
    ap.add_argument("run", metavar="RUN")
    ap.add_argument("--from", dest="machine", required=True, metavar="MACHINE")
    ap.add_argument("--project", default=os.environ.get("RESEARCH_PROJECT"), help="the project the machine is bound to (default $RESEARCH_PROJECT)")
    ap.add_argument("--store", default=None, help="local store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    try:
        m = remote.resolve_machine(ns.machine, ns.project)
        tgt = remote.pod_target(m)
        ssh = remote.ssh_argv(tgt["host"], tgt["port"])
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
    except (remote.Refused, ValueError, KeyError, OSError) as e:
        print(f"research data pull: {e}", file=sys.stderr)
        return 1
    if ns.json:
        print(json.dumps({"attempt": doc["id"], "machine": ns.machine, "pulled": pulled}, indent=2, sort_keys=True))
    else:
        print(f"pulled attempt {doc['id']} from {ns.machine}: " + (", ".join(f"{p['name']}={p['id']} ({p['blobs']} blob{'s' if p['blobs'] != 1 else ''})" for p in pulled) or "no outputs"))
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


COMMANDS = {"lookup": cmd_lookup, "pull": cmd_pull, "attempt": cmd_attempt}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
