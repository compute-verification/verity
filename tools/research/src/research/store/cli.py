"""`research data …` (README §9): the evidence store from the command line.

  research data put  --kind K [--meta JSON|@file] [--ref name=art:...]... [--file PATH | --tree DIR] [--preserve]   -> art id
  research data show ART|ATTEMPT|DRV [--json]      manifest / attempt / attempts with that derivation
  research data where ART [--json]                 replicas: local / repo / remote (+ verified_utc)
  research data push [ART|RUN ...] [--pending] [--skip-record-less] [--no-labels] [--jobs N] [--verify head|readback]
                                                   preserve on the remote: blobs uploaded concurrently and verified, then the manifest (PRESERVED);
                                                   a run pushes its attempt + outputs + the assertions about them (attempt, derivation, outputs);
                                                   --pending pushes everything local the remote lacks, assertions included;
                                                   --skip-record-less pushes no attempt that names no run record
  research data labels-sync [--push-only|--pull-only|--from-backup ART|--from-dir DIR] [--jobs N] [--json]
                                                   union labels/ with the remote's labels/<target>/<content sha256>.json (nothing overwritten, ever);
                                                   --from-backup unions a labels-backup/v1 tarball, --from-dir another store's labels/
  research data fetch ART [--to DIR] [--path GLOB]... [--list [--json]]
                                                   materialise the payload (file or tree, or just the files matching --path); prints the path;
                                                   --list prints the tree's files (path, bytes, local/remote) and fetches nothing
  research data evict --target-free-gb N [--dry-run] [--runs [--runs-dir D]] [--min-mb M] [--keep-trees] [--keep-registry FILE]...
                     [--src-dir D [--src-origin URL]] [--json]
                                                   free disk: evict blobs (and trees/) whose artifacts are PRESERVED on the remote, largest first;
                                                   --runs also evicts preserved big files of finished run dirs (oldest first); exit 1 if the target is not reached;
                                                   --src-dir deletes the shipped source trees on origin that nothing uses, whatever the target
                                                   --keep-registry keeps every blob of the registry's artifacts (fixtures/artifacts.json)
  research data verify ART [--json]                PRESERVED check against the remote (exit 1 when not preserved)
  research data preserved (RUN|ART)... [--mode head|readback|recorded] [--jobs N] [--json]
                                                   the durability gate: exit 0 iff the attempt manifest, every output artifact AND every local
                                                   assertion about them are verifiably on the remote; otherwise names the failing legs (preserved.py);
                                                   N targets (default 8) checked at once, each printed as it finishes
  research data select [--kind K] [--where meta.key=value]... [--tool T] [--derivation D] [--attempt R] [--label k=v]... [--json|--ids]
                                                   stderr names the source (the index and its as-of time, or the catalog alone);
                                                   exit 1 when there is neither an index nor an artifact in the catalog
  research data snapshot --name NAME (--members art:... | --select <select flags>) [--note TEXT]   -> art id
  research data labels TARGET [--history] [--key K] [--by PAT] [--remote] [--json]
                                                   first pulls TARGET's assertions the remote has and this store lacks (unreachable: a warning, the local ones);
                                                   --remote: each assertion marked both / LOCAL ONLY (not durable) / REMOTE ONLY; exit 1 if any local only
  research data label  TARGET KEY VALUE --by WHO [--ref REF] [--value-json] [--off-vocab]
                                                   writes the canonical key/spelling in the vocabulary's form (zk true -> a real boolean, batch -> B,
                                                   axis -> sweep, candidate A -> A-GKR); REFUSES (exit 2) an unknown key, a measurement, a ref, an
                                                   off-enum value, a word in a bool/int key, or `abandoned` without --ref note:<lane>/<note>
                                                   -- --off-vocab writes it anyway
  research data vocab [--json]                     the label vocabulary (vocab.py): key, asserter group, value form, enum values
  research data vocab-check [--json] [--strict]    read-only audit of labels/: unknown keys, old spellings, strings in bool/int keys, off-enum values,
                                                   refs written as labels; counts, asserters, example targets
  research data refresh-fixtures [--preserve] [--dry-run] [--repo DIR]
                                                   re-register every fixtures/artifacts.json entry whose bytes changed; old id kept under `previous`
  research data fetch-fixtures [PATH ...] [--repo DIR] [--check] [--stubs] [--public-url URL] [--no-private] [--jobs N] [--json]
                                                   make the registry's fetched entries under PATH (default: all) present and exact, from the
                                                   private remote when reachable, else the public export; every byte checked; exit 1 if any is missing
  research data export [--repo DIR] [--dry-run] [--no-previous] [--blobs FILE] [--id ART]... [--jobs N] [--json]
  research data export --verify-url URL [--repo DIR] [--no-previous] [--blobs FILE] [--jobs N] [--json]
                                                   copy every registry entry (and its `previous` ids, and each --id) to the public export bucket; --verify-url
                                                   fetches them all back through a public front end into a scratch store, checking every byte
  research data doctor [--clean] [--runs-dir D]    stale store_publish_error.json files (run is an attempt here) vs real failures; --clean deletes the stale
  research data gc --cite DIR... --cache DIR [--apply --state FILE] [--report FILE] [--work DIR] [--jobs N]
                                                   the remote's blobs only uncited artifacts past their kind's TTL hold (gc.py, README §7.4);
                                                   --apply marks them and deletes those still candidates a grace after their mark
  research data withdraw ART... --reason T --ref ISSUE --as HANDLE --cache DIR [--apply]
                                                   delete the blobs of artifacts holding values that must never be shared, cited or
                                                   not, and label each `withdrawn` (withdraw.py, README §7.4); a blob another
                                                   artifact holds is kept and named
  research data reindex [--remote]                 rebuild the catalog (built aside, swapped in complete); --remote first pulls what the remote has
  research data index [--db F] [--from DIR] [--public [--public-db F]] [--full] [--jobs N] [--json]
                                                   the derived index (bucket_index.py): list the bucket, read the records the index lacks, write
                                                   records/fields/edges/listings; --public then rebuilds the public index from the export bucket
  research data refresh [--jobs N] [--json] [--payloads KIND ...] [--payload-max-bytes N]
                                                   incremental: fetch and index only the manifests/attempts/labels the remote has and we lack;
                                                   --payloads also makes local the small file payloads of those kinds (what a render reads)
  research data pending [--json]                   the write-through queue: what this store holds that the remote lacks (exit 1 if any)
  research data reconcile-attempts (RUN... | --all) [--apply] [--json]
                                                   local attempts that differ from the remote's write-once copy; --apply supersedes them (audited)
  research data parity --fresh DIR [--json-out FILE]
                                                   content parity against a store built from the remote alone (parity.py); exit 0 iff explained
  research data catalog [--json]                   catalog schema version vs the code's (older catalogs migrate forward on open; a newer one is
                                                   refused, exit 3, unless RESEARCH_CATALOG_IGNORE_VERSION=1), row counts per table
  research data catalog --compact OUT              a copy without the whole manifests or replica records, for readers that query (notes mcp)
  research data sql "SELECT ..." [--json]          read-only connection
  research data mint-credential [--ttl 6h] [--permission object-read-write] [--prefix P]... [--actions A,B] [--via auto|local|api] [--env]
                                                   a short-lived bucket-scoped R2 credential for a pod (JSON, or --env export lines)
  (lookup / pull / attempt are provided by cli_prov when present)

Every command takes `--store ROOT` (default `$RESEARCH_STORE` or `~/.research/store`) and accepts abbreviated artifact ids
(`art:cc59294a`, >= 8 hex digits, unique in the local catalog, else on the remote) wherever it takes an id.  With write-through
on (README §7.2) `label` and `put` upload in the same command; a write that stays local says `local only` / `NOT preserved`.  `COMMANDS` maps a name to
`fn(argv) -> int`; `main(argv)` dispatches.  Structured output has `--json` everywhere.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

from ..secret_hint import lost
from . import ids, vocab
from .index import IGNORE_VERSION_ENV, CatalogVersionError, compact
from .local import REFRESH_PAYLOAD_MAX, LocalStore, StoreError, store_root
from .remote import ConfigError, RemoteConflict


def _ap(name: str, **kw) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"research data {name}", allow_abbrev=False, **kw)
    ap.add_argument("--store", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    return ap


def _kv(items: list[str], flag: str) -> dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise ValueError(f"{flag} wants KEY=VALUE, got {it!r}")
        out[k] = v
    return out


def _meta(arg: str | None) -> dict:
    if not arg:
        return {}
    text = Path(arg[1:]).read_text(encoding="utf-8") if arg.startswith("@") else arg
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError("--meta must be a JSON object")
    return doc


def _refs(items: list[str]) -> dict:
    """`name=art:...` (repeat a name, or comma-separate, for a list)."""
    out: dict = {}
    for it in items:
        name, sep, v = it.partition("=")
        if not sep or not name:
            raise ValueError(f"--ref wants name=art:..., got {it!r}")
        vals = [x for x in v.split(",") if x]
        if name in out:
            prev = out[name] if isinstance(out[name], list) else [out[name]]
            out[name] = prev + vals
        else:
            out[name] = vals[0] if len(vals) == 1 else vals
    return out


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _select_parser() -> argparse.ArgumentParser:
    ap = _ap("select")
    ap.add_argument("--kind")
    ap.add_argument("--where", action="append", default=[], metavar="meta.key=value")
    ap.add_argument("--tool")
    ap.add_argument("--derivation")
    ap.add_argument("--attempt")
    ap.add_argument("--label", action="append", default=[], metavar="k=v", help="repeatable; every given label must be asserted (conjunction)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ids", action="store_true", help="print ids only")
    return ap


def _select(st: LocalStore, ns: argparse.Namespace, idx=None) -> tuple[list[dict], dict]:
    """The catalog's matches, or with ``idx`` (`bucket_index.reader`) the index's plus the catalog's the index lacks (pending here)."""
    from . import bucket_index as BI  # noqa: PLC0415

    labels = [tuple(next(iter(_kv([x], "--label").items()))) for x in ns.label]
    query = {k: v for k, v in (("kind", ns.kind), ("where", _kv(ns.where, "--where") or None), ("tool", ns.tool), ("derivation", ns.derivation),
                               ("attempt", ns.attempt), ("label", [list(x) for x in labels] or None)) if v}
    rows = st.index.select_artifacts(kind=ns.kind, where=_kv(ns.where, "--where"), label=labels or None, attempt=ns.attempt, tool=ns.tool, derivation=ns.derivation)
    if idx is None:
        return rows, query
    here = {r["id"]: r for r in rows}
    found = BI.select_artifacts(idx, kind=ns.kind, where=_kv(ns.where, "--where"), labels=labels, attempt=ns.attempt, tool=ns.tool,
                                derivation=ns.derivation)
    out = [here.get(i) or {"id": i, "kind": d.get("kind"), "bytes": (d.get("payload") or {}).get("bytes", 0),
                           "payload_type": (d.get("payload") or {}).get("type"), "manifest": d, "local": False, "remote": True,
                           "repo_path": None}
           for i, d in found]
    pend = BI.not_indexed(idx, [i for i in here if i not in {r["id"] for r in out}])
    return sorted(out + [{**here[i], "pending": True} for i in pend], key=lambda r: r["id"]), query


def _payload_line(doc: dict) -> str:
    p = doc.get("payload")
    if not p:
        return "none"
    if p["type"] == "file":
        return f"file {p['name']} ({p['bytes']} bytes, sha256 {p['sha256'][:12]}…)"
    return f"tree {len(p['files'])} files ({p['bytes']} bytes, sha256 {p['sha256'][:12]}…)"


# ---- commands ---------------------------------------------------------------------------------------------------------------------------

def cmd_put(argv: list[str]) -> int:
    ap = _ap("put")
    ap.add_argument("--kind", required=True)
    ap.add_argument("--meta", default=None, help="JSON object or @file")
    ap.add_argument("--ref", action="append", default=[], metavar="name=art:...")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--file", default=None)
    g.add_argument("--tree", default=None)
    ap.add_argument("--preserve", action="store_true", help="push to the remote and verify (exit 1 unless PRESERVED)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    art = st.put_artifact(ns.kind, _meta(ns.meta), _refs(ns.ref), file=Path(ns.file) if ns.file else None, tree=Path(ns.tree) if ns.tree else None)
    rep, why, gone = None, None, ""
    if ns.preserve:
        rep = st.push(art, preserve=True)
    elif st.write_through:                                          # write-through (README §7.2): the local write stands whatever the upload does
        try:
            rep = st.push(art, preserve=True)
        except Exception as e:  # noqa: BLE001
            why = str(e)
    else:
        gone = lost("R2_ENDPOINT", "R2_BUCKET") or st.remote_lost()
        why = gone or "write-through off"
    if ns.json:
        _dump({"id": art, "preserve": rep, **({"pending": why} if why or (rep and not rep["preserved"]) else {})})
    else:
        print(art)
        if rep is not None and rep["preserved"]:
            print(f"research data put: PRESERVED on {rep['remote']}", file=sys.stderr)
        else:
            why = why or "; ".join(rep["errors"])
            print(f"research data put: {'local only' if why == 'write-through off' else 'NOT preserved'} ({why}); pending: "
                  "`research data push --pending` uploads it", file=sys.stderr)
    return 1 if (ns.preserve and not rep["preserved"]) or gone else 0


def cmd_show(argv: list[str]) -> int:
    ap = _ap("show")
    ap.add_argument("ref", help="art:… | drv:… | run id")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    k = ids.kind_of_id(ns.ref)
    if k == "art":
        doc = st.get_manifest(ns.ref).to_doc()
        producers = st.index.attempts_by_output(ns.ref)
        if ns.json:
            _dump({"id": ns.ref, "manifest": doc, "produced_by": producers})
            return 0
        print(f"id        {ns.ref}\nkind      {doc['kind']}\npayload   {_payload_line(doc)}")
        for name, v in sorted(doc["refs"].items()):
            print(f"ref       {name} = {v if isinstance(v, str) else f'[{len(v)}] ' + ', '.join(v[:3]) + (', …' if len(v) > 3 else '')}")
        print(f"meta      {json.dumps(doc['meta'], sort_keys=True)}")
        if producers:
            print(f"produced  {', '.join(producers)}")
        return 0
    if k == "drv":
        rows = st.index.attempts_by_derivation(ns.ref)
        if ns.json:
            _dump(rows)
            return 0
        print(f"derivation {ns.ref}: {len(rows)} attempt(s)")
        for a in rows:
            print(f"  {a['id']}  {a['tool'].get('name')}@{a['tool'].get('version')}  {(a.get('status') or {}).get('state')} rc={(a.get('status') or {}).get('rc')}"
                  f"  reuse_key={a.get('reuse_key')}  outputs={json.dumps(a.get('outputs'), sort_keys=True)}")
        return 0
    if k == "run":
        doc = st.get_attempt(ns.ref)
        if doc is None:
            print(f"research data show: no attempt {ns.ref} in the store", file=sys.stderr)
            return 1
        if ns.json:
            _dump(doc)
            return 0
        st_ = doc.get("status") or {}
        print(f"attempt    {doc['id']}\ntool       {doc['tool'].get('name')}@{doc['tool'].get('version')}\nderivation {doc['derivation']}\n"
              f"reuse_key  {doc.get('reuse_key')}\nstatus     {st_.get('state')} rc={st_.get('rc')} validation={st_.get('validation')}\n"
              f"source     {json.dumps(doc.get('source'), sort_keys=True)}\ninputs     {json.dumps(doc.get('inputs'), sort_keys=True)}\n"
              f"outputs    {json.dumps(doc.get('outputs'), sort_keys=True)}\nexecution  {json.dumps(doc.get('execution'), sort_keys=True)}")
        return 0
    print(f"research data show: {ns.ref!r} is not an art:, drv: or run id (an abbreviated art: id needs >= {ids.SHORT_ART_MIN_HEX} hex digits)",
          file=sys.stderr)
    return 2


def cmd_where(argv: list[str]) -> int:
    ap = _ap("where")
    ap.add_argument("art")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    rows = LocalStore(ns.store).where(ns.art)
    if ns.json:
        _dump(rows)
        return 0
    for r in rows:
        if r.get("configured") is False:
            print("remote    not configured (the local store is the only replica)")
            continue
        if r.get("error"):
            print(f"remote    UNUSABLE  {r['error']}")
            continue
        print(f"{r['location']:<9} {'present' if r['present'] else 'ABSENT ':<8} verified={r.get('verified_utc') or '-':<21} "
              f"{r.get('path') or r.get('remote')}" + (f"  blobs {r['blobs']}" if r.get("blobs") else ""))
    return 0


def cmd_fetch(argv: list[str]) -> int:
    """Materialise a payload, or with `--path GLOB` (repeatable) just the matching subtree of a tree artifact (`rep1/*`, `*.proof`,
    a bare directory name selects its subtree); the on-disk layout under `--to` is the artifact's own.  `--list` prints the files
    instead (path, bytes, local | remote | absent) and fetches nothing: `local` = the blob is in objects/, `remote` = the catalog
    flags the artifact as on the remote (no per-file HEAD)."""
    ap = _ap("fetch")
    ap.add_argument("art")
    ap.add_argument("--to", default=None, metavar="DIR", help="materialise under this directory (default: the store's own cache)")
    ap.add_argument("--path", action="append", default=[], metavar="GLOB", help="only the files matching this glob (repeatable); tree artifacts")
    ap.add_argument("--list", action="store_true", help="list the payload files (path, bytes, local/remote) and fetch nothing")
    ap.add_argument("--json", action="store_true", help="with --list: the rows as JSON")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    if ns.list:
        rows = st.listing(ns.art, paths=ns.path or None)
        if ns.json:
            _dump(rows)
            return 0
        for r in rows:
            print(f"{r['path']:<60} {r['bytes']:>12}  {'local' if r['local'] else ('remote' if r['remote'] else 'ABSENT')}")
        n_local = sum(1 for r in rows if r["local"])
        print(f"{len(rows)} file(s), {sum(r['bytes'] for r in rows)} bytes; {n_local} local, {len(rows) - n_local} not local"
              + ("" if not rows or rows[0]["remote"] else " (artifact not flagged on the remote)"), file=sys.stderr)
        return 0
    to = None
    if ns.to:
        to = Path(ns.to)
        to.mkdir(parents=True, exist_ok=True)
    print(st.fetch(ns.art, to, paths=ns.path or None))
    return 0


def cmd_evict(argv: list[str]) -> int:
    """Free local disk by evicting copies whose bytes are PRESERVED on the remote (`evict.py`): store blobs held only by artifacts with a
    verified remote replica (largest first) and the `trees/` cache, and with `--runs` the big files of finished run directories whose
    `run_files` artifact is verified (oldest first).  Stops when the free space reaches `--target-free-gb`.  Never touches pending
    objects, manifests, attempts, labels or the catalog.  `--dry-run` reports without removing anything."""
    from . import evict as EV  # noqa: PLC0415

    ap = _ap("evict")
    ap.add_argument("--target-free-gb", required=True, type=float, metavar="N", help="stop once this much is free on the store's filesystem")
    ap.add_argument("--dry-run", action="store_true", help="report the plan; remove nothing")
    ap.add_argument("--min-mb", type=float, default=EV.MIN_BYTES / 2**20, metavar="M", help="only blobs / files of at least this size (default 1)")
    ap.add_argument("--keep-trees", action="store_true", help="leave the trees/ cache alone")
    ap.add_argument("--keep-registry", action="append", default=[], metavar="FILE",
                    help="keep every blob an artifact of this registry holds (fixtures/artifacts.json's form: its `art:` keys); repeatable")
    ap.add_argument("--runs", action="store_true", help="also evict preserved big files from finished run directories (writes run-files.evicted.json)")
    ap.add_argument("--runs-dir", default=None, metavar="DIR", help="runs root for --runs (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--quiet-min", type=float, default=EV.QUIET_S / 60, metavar="M", help="--runs: skip run dirs touched in the last M minutes (default 30)")
    ap.add_argument("--src-dir", default=None, metavar="DIR",
                    help="also delete the `research run` source trees under DIR (<root>/src) recoverable on the node that nothing uses, "
                         "whatever the target (evict.py `evict_src`; needs root or CAP_SYS_PTRACE to read /proc)")
    ap.add_argument("--src-origin", default=None, metavar="URL",
                    help="--src-dir: the repository whose refs make a tree recoverable (default <root>/git/verity.git, the node's own)")
    ap.add_argument("--src-keep-h", type=float, default=EV.SRC_KEEP_S / 3600, metavar="H",
                    help="--src-dir: keep a tree changed, or named by a run recorded, within H hours (default 2); a node short of "
                         "inodes under many shipped commits takes less")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    target = EV.parse_gb(str(ns.target_free_gb))
    min_bytes = int(ns.min_mb * 2**20)
    say = None if ns.json else (lambda s: print(s, flush=True))
    keep = frozenset(k for f in ns.keep_registry for k in json.loads(Path(f).read_text()) if k.startswith("art:"))
    rep: dict = {"store": EV.evict_store(st, target_free_bytes=target, dry_run=ns.dry_run, min_bytes=min_bytes, keep_trees=ns.keep_trees,
                                         keep=keep, log=say)}
    if ns.runs:
        from ..runs import runs_root  # noqa: PLC0415

        rep["runs"] = EV.evict_runs(st, runs_root(ns.runs_dir), target_free_bytes=target, dry_run=ns.dry_run, min_bytes=min_bytes,
                                    quiet_s=ns.quiet_min * 60, log=say)
    if ns.src_dir:
        rep["src"] = EV.evict_src(Path(ns.src_dir), origin=ns.src_origin, dry_run=ns.dry_run, keep_s=ns.src_keep_h * 3600, log=say)
    if ns.json:
        _dump(rep)
    last = rep.get("runs") or rep["store"]
    return 0 if last["reached_target"] else 1


def cmd_verify(argv: list[str]) -> int:
    ap = _ap("verify")
    ap.add_argument("art")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    rep = LocalStore(ns.store).verify(ns.art)
    if ns.json:
        _dump(rep)
    elif rep["preserved"]:
        print(f"{ns.art} PRESERVED on {rep['remote']} at {rep['verified_utc']} ({len(rep['checked'])} objects read back)")
    else:
        print(f"{ns.art} NOT preserved: {rep.get('reason') or '; '.join(rep['errors'])}")
    return 0 if rep["preserved"] else 1


def cmd_preserved(argv: list[str]) -> int:
    """The durability gate (F7, `preserved.py`), three legs: the attempt manifest, every artifact it names (outputs + telemetry),
    and every local assertion about the attempt / its derivation / those artifacts must be verifiably on the remote.  Exit 0 iff so
    for every target; else 1, the list of what is not, and which legs fail.  Exit 2 on a usage error (no targets).  Targets are checked
    `--jobs` at a time and each one's lines are printed as it finishes (completion order); `--json` waits and keeps the given order."""
    from . import preserved as P

    ap = _ap("preserved")
    ap.add_argument("targets", nargs="*", help="run ids and/or art:<hex>")
    ap.add_argument("--mode", choices=list(P.MODES), default="head",
                    help="head: size + MD5 ETag per blob (read-back when unavailable), manifest read back (default); "
                         "readback: every object read back; recorded: trust the local catalog's verified remote replica (offline)")
    ap.add_argument("--jobs", type=int, default=8, metavar="N", help="targets checked at once (default 8)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if not ns.targets:
        print("research data preserved: give at least one run id or art: id", file=sys.stderr)
        return 2

    def say(t: dict) -> None:
        print("\n".join(P.format_target(t)), flush=True)

    rep = P.check(LocalStore(ns.store), ns.targets, mode=ns.mode, jobs=ns.jobs, on_target=None if ns.json else say)
    if ns.json:
        _dump(rep)
    else:
        for line in P.format_summary(rep):
            print(line)
    return 0 if rep["preserved"] else 1


def cmd_select(argv: list[str]) -> int:
    """Artifacts by facet, from the index (`bucket_index.reader`: Neon with $STORE_PRIVATE_READER_URL, else this machine's
    `research data index` build) plus this store's matches the index lacks; from the catalog where there is no index.  Every form
    (text, `--ids`, `--json`) ends with one stderr line naming the source and, for the index, the time it is complete as of: a
    stale build answers only for what it had listed.  With no index that ever listed the bucket and a catalog holding no artifact,
    an empty answer would mean nothing, so it exits 1 naming what is missing (a `research run` job's store, 19 Oct)."""
    from . import bucket_index as BI  # noqa: PLC0415

    ns = _select_parser().parse_args(argv)
    st = LocalStore(ns.store)
    idx = BI.reader(st.root)
    try:
        as_of = BI.as_of(idx) if idx is not None else None
        catalog = st.root / "catalog.sqlite"
        no_index = (f"the index {idx.describe()} never listed the bucket" if idx is not None else
                    f"${BI.READER_ENV} is unset and there is no `research data index` build at {BI.default_path(st.root)}")
        if as_of is None and st.index.db.execute("SELECT 1 FROM artifacts LIMIT 1").fetchone() is None:
            print(f"research data select: no index to select from ({no_index}), and this store's catalog ({catalog}) holds no "
                  f"artifacts, so an empty answer would mean nothing. Set ${BI.READER_ENV} (store_private's read-only URL), or fill "
                  "this store from the remote first (`research data refresh`, or `research data index`); a `research run` job's "
                  "store reads the remote only when the run is launched with `--store-read`", file=sys.stderr)
            return 1
        rows, _ = _select(st, ns, idx)
        source = (f"the index ({idx.describe()}, complete as of {as_of or 'never listed'})" if idx is not None else
                  f"the catalog alone (this store's, {catalog}; no index: {no_index})")
    finally:
        if idx is not None:
            idx.close()
    if ns.json:
        _dump(rows)
    elif ns.ids:
        for r in rows:
            print(r["id"])
    else:
        for r in rows:
            print(f"{r['id']}  {r['kind']:<24} {r['bytes']:>12} {r['payload_type'] or 'none':<5} {'local' if r['local'] else '-':<5} "
                  f"{'remote' if r['remote'] else '-':<6} {r['repo_path'] or ''}{'  (pending: not in the index)' if r.get('pending') else ''}")
    n = sum(1 for r in rows if r.get("pending"))
    print(f"{len(rows)} artifact(s) from {source}" + (f", {n} of them only here" if n else ""), file=sys.stderr)
    return 0


def cmd_snapshot(argv: list[str]) -> int:
    sel_argv: list[str] | None = None
    if "--select" in argv:
        i = argv.index("--select")
        argv, sel_argv = argv[:i], argv[i + 1:]
    ap = _ap("snapshot")
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", action="append", default=[], metavar="art:...", help="repeat or comma-separate; or `--select <select flags>`")
    ap.add_argument("--note", default=None)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    members = [m for item in ns.members for m in item.split(",") if m]
    query = None
    if sel_argv is not None:
        rows, query = _select(st, _select_parser().parse_args(sel_argv))
        members += [r["id"] for r in rows]
    if not members:
        print("research data snapshot: no members (--members or --select matched nothing)", file=sys.stderr)
        return 1
    art = st.snapshot(ns.name, members, note=ns.note, query=query)
    _dump({"id": art, "members": sorted(set(members))}) if ns.json else print(art)
    return 0


def _fmt_label_value(v) -> str:
    return v if isinstance(v, str) else json.dumps(v)


def cmd_labels(argv: list[str]) -> int:
    """TARGET's assertions, after pulling the ones the remote has and this store lacks (`labels_sync.read_through`; unreachable:
    a warning, and the local ones).  `--remote` is the durability view (`labels_sync.compare`): every assertion about TARGET on
    either side, marked `both`, `LOCAL ONLY` (not durable: `research data push` repairs) or `REMOTE ONLY` (asserted elsewhere;
    `labels-sync` / `reindex --remote` brings it here).  Exit 1 when any local assertion is not on the remote."""
    ap = _ap("labels")
    ap.add_argument("target")
    ap.add_argument("--history", action="store_true", help="every assertion in ts order (default: the latest per key and asserter)")
    ap.add_argument("--key")
    ap.add_argument("--by", default=None, metavar="PATTERN", help="fnmatch on the asserter, e.g. 'red-team-*'")
    ap.add_argument("--remote", action="store_true", help="compare with the remote: both / LOCAL ONLY / REMOTE ONLY per assertion (exit 1 if any is local only)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if ns.remote:
        from . import labels_sync as LS  # noqa: PLC0415

        st = LocalStore(ns.store)
        rows = [x for x in LS.compare(st, ns.target)
                if (ns.key is None or (x["label"] or {}).get("key") == ns.key) and (ns.by is None or fnmatch.fnmatchcase(str((x["label"] or {}).get("by")), ns.by))]
        n = {"both": 0, "local": 0, "remote": 0}
        for x in rows:
            n[x["where"]] += 1
        if ns.json:
            _dump({"target": ns.target, "remote": st.remote.describe(), "counts": n, "labels": rows})
        else:
            for x in rows:
                lab = x["label"] or {}
                where = {"both": "both       ", "local": "LOCAL ONLY ", "remote": "REMOTE ONLY"}[x["where"]]
                desc = (f"{lab.get('ts')}  {lab.get('key')}={_fmt_label_value(lab.get('value'))}  by {lab.get('by')}" + (f"  ref {lab['ref']}" if lab.get("ref") else "")) if lab else "?"
                print(f"{where} {desc}" + (f"  [{x['error']}]" if x.get("error") else ""))
            print(f"{ns.target}: {n['both']} on both, {n['local']} local only, {n['remote']} remote only ({st.remote.describe()})"
                  + ("  -- NOT durable: `research data push` repairs" if n["local"] else ""), file=sys.stderr)
        return 1 if n["local"] else 0
    from . import labels_sync as LS  # noqa: PLC0415

    st = LocalStore(ns.store)
    LS.read_through(st, [ns.target], lambda s: print(f"research data labels: {s}", file=sys.stderr))
    labs = st.labels(ns.target, key=ns.key, by_glob=ns.by)
    if not ns.history:
        latest = {}
        for lab in labs:
            latest[(lab.key, lab.by)] = lab
        labs = sorted(latest.values(), key=lambda l: (l.ts, l.key, l.by))
    if ns.json:
        _dump([lab.to_doc() for lab in labs])
        return 0
    for lab in labs:
        print(f"{lab.ts}  {lab.key}={json.dumps(lab.value) if not isinstance(lab.value, str) else lab.value}  by {lab.by}" + (f"  ref {lab.ref}" if lab.ref else ""))
    if not labs:
        print(f"no labels on {ns.target}", file=sys.stderr)
    return 0


def cmd_label(argv: list[str]) -> int:
    ap = _ap("label")
    ap.add_argument("target")
    ap.add_argument("key")
    ap.add_argument("value")
    ap.add_argument("--by", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--value-json", action="store_true", help="parse VALUE as JSON instead of a string")
    ap.add_argument("--off-vocab", action="store_true",
                    help="write a key or value the vocabulary cannot place (a concept it has not learnt yet); add it to vocab.py next")
    ns = ap.parse_args(argv)
    value = json.loads(ns.value) if ns.value_json else ns.value
    key = vocab.canonical_key(ns.key)                      # hygiene (§8.1): the canonical key and spelling go in the file, the form the vocabulary
    value = vocab.coerce(key, vocab.canonical_value(key, value))   # declares (a real bool for zk / model_agrees, an int for B), never the word
    st = LocalStore(ns.store)
    why = vocab.check(key, value) or vocab.check_ref(key, ns.ref)
    if not why and key == vocab.MEMORY_LINE_OF_RECORD:
        why = vocab.check_line_of_record(value, _run_file(st, ns.ref, vocab.MEMORY_LINE_FILE))
    if why and not ns.off_vocab:
        print(f"research data label: refused: {why}.  Nothing written (--off-vocab writes it anyway)", file=sys.stderr)
        return 2
    for note in (f"key {ns.key!r} written as {key!r}" if key != ns.key else None,
                 f"value {ns.value!r} written as {json.dumps(value)}" if value != (json.loads(ns.value) if ns.value_json else ns.value) else None):
        if note:
            print(f"research data label: {note}", file=sys.stderr)
    if why:
        print(f"research data label: warning: {why}; stored anyway (--off-vocab)", file=sys.stderr)
    print(st.add_label(ns.target, key, value, ns.by, ref=ns.ref))
    said, lost = _label_write_through(st, ns.target)
    print(f"research data label: {said}", file=sys.stderr)
    return 1 if lost else 0


def _run_file(st: LocalStore, run_id: str, rel: str) -> object:
    """The JSON file `rel` of run `run_id`'s run record (its Attempt's `outputs.run_record`), or None where the store has none."""
    try:
        art = ((st.get_attempt(run_id) or {}).get("outputs") or {}).get("run_record")
        return json.loads((st.fetch(art, paths=[rel]) / rel).read_text()) if art else None
    except (StoreError, OSError, ValueError):
        return None


def _label_write_through(st: LocalStore, target: str) -> tuple[str, bool]:
    """Upload the target's assertions the remote lacks, in the command that wrote one (README §7.2); the local write stands either way,
    and a write that stays local says so. Also whether write-through was on and the upload failed (the caller exits 1)."""
    from . import labels_sync as LS  # noqa: PLC0415

    pending = "pending: `research data push --pending` uploads it"
    if not st.write_through:
        if hint := lost("R2_ENDPOINT", "R2_BUCKET") or st.remote_lost():
            return f"NOT preserved ({hint}); {pending}", True
        return f"local only (write-through off); {pending}", False
    if st.remote is None:
        return f"NOT preserved ({getattr(st, 'remote_error', None) or 'no remote configured'}); {pending}", True
    try:
        failed = LS.push_targets(st, [target])["failed"]
    except Exception as e:  # noqa: BLE001
        failed = [f"{type(e).__name__}: {e}"]
    return (f"NOT preserved ({failed[0]}); {pending}", True) if failed else (f"on {st.remote.describe()}", False)


def cmd_vocab_check(argv: list[str]) -> int:
    """Read-only audit of labels/ against the vocabulary (`vocab_check.py`): unknown keys, old spellings of keys and values, bool / int /
    number keys holding strings, enum values off the set, refs written as labels, unreadable files -- each with counts, asserters and
    example targets.  Exit 0 always (it is a report); `--strict` exits 1 when anything is found."""
    from . import vocab_check as VC  # noqa: PLC0415

    ap = _ap("vocab-check")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 when there is any finding")
    ns = ap.parse_args(argv)
    rep = VC.audit(LocalStore(ns.store))
    _dump(rep) if ns.json else print(VC.render(rep))
    return 1 if ns.strict and rep["findings"] else 0


def cmd_labels_sync(argv: list[str]) -> int:
    """Union the local labels/ with the remote's `labels/` prefix (`labels_sync.py`): every assertion goes up under its content hash
    (`labels/<target>/<sha256>.json`, immutable, so two machines never overwrite each other), and every remote assertion we lack comes
    down into the store's own layout.  `--from-backup ART` unions a `labels-backup/v1` tarball instead; `--from-dir DIR` any directory of
    assertion files (another store's labels/).  Read-only for the remote with `--pull-only`; nothing is ever deleted or rewritten."""
    from . import labels_sync as LS  # noqa: PLC0415

    ap = _ap("labels-sync")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--push-only", action="store_true", help="only upload local assertions the remote lacks")
    g.add_argument("--pull-only", action="store_true", help="only download remote assertions this store lacks")
    g.add_argument("--from-backup", metavar="ART", help="union a labels-backup/v1 artifact's tarball into labels/ (no remote needed)")
    g.add_argument("--from-dir", metavar="DIR", help="union a directory of assertion files into labels/ (no remote needed)")
    ap.add_argument("--jobs", type=int, default=LS.SYNC_JOBS, metavar="N")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    warn = None if ns.json else (lambda s: print(s, file=sys.stderr))
    if ns.from_backup or ns.from_dir:
        rep = LS.import_backup(st, ns.from_backup, warn=warn) if ns.from_backup else LS.import_dir(st, Path(ns.from_dir), warn=warn)
        _dump(rep) if ns.json else print(f"labels: {rep['seen']} in {ns.from_backup or ns.from_dir}: {rep['added']} added, {rep['already']} already here, {rep['bad']} unreadable")
        return 0
    rep = LS.sync(st, push=not ns.pull_only, pull=not ns.push_only, jobs=max(1, ns.jobs), log=warn)
    if ns.json:
        _dump(rep)
    else:
        print(f"labels: {rep['local']} local, {rep['remote_before']} on {rep['remote']}: {len(rep['pushed'])} pushed, {len(rep['pulled'])} pulled, "
              f"{rep['already']} already on both" + (f", {len(rep['bad'])} bad remote key(s) skipped" if rep["bad"] else ""))
    return 1 if rep["bad"] else 0


def cmd_vocab(argv: list[str]) -> int:
    ap = _ap("vocab")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if ns.json:
        _dump({k.name: {"group": k.group, "form": k.form, "values": list(k.values), "status": k.status, "doc": k.doc} for k in vocab.VOCAB.values()})
    else:
        print(vocab.render())
    return 0


def cmd_reindex(argv: list[str]) -> int:
    ap = _ap("reindex")
    ap.add_argument("--remote", action="store_true", help="also list the remote; fetch manifests/attempts/labels it has and we lack")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    pulled = None
    if ns.remote and st.remote is not None:                         # labels first: the rebuild below then indexes them with everything else
        from . import labels_sync as LS  # noqa: PLC0415

        pulled = len(LS.sync(st, push=False, pull=True, log=None if ns.json else (lambda s: print(s, file=sys.stderr)))["pulled"])
    n = st.reindex(remote=ns.remote)
    if pulled is not None:
        n["labels_pulled"] = pulled
    _dump(n) if ns.json else print(f"reindexed {st.root}: {n['artifacts']} artifacts, {n['attempts']} attempts, {n['labels']} labels"
                                   + (f" ({pulled} pulled from the remote)" if pulled is not None else ""))
    return 0


def cmd_index(argv: list[str]) -> int:
    """The derived index (README §5.1).  Exit 1 when a record failed its check (it is left out and read again next pass)."""
    from . import bucket_index as BI  # noqa: PLC0415
    from .remote import FsRemote, load_export  # noqa: PLC0415

    ap = _ap("index")
    ap.add_argument("--db", type=Path, help="the private index (default <store>/index.sqlite)")
    ap.add_argument("--from", dest="source", type=Path, metavar="DIR", help="index a directory laid out like the bucket instead of the remote")
    ap.add_argument("--public", action="store_true", help="then rebuild the public index from the export bucket's manifests")
    ap.add_argument("--public-db", type=Path, help="the public index (default <store>/index-public.sqlite)")
    ap.add_argument("--full", action="store_true", help="list every run, not only those after the newest indexed one")
    ap.add_argument("--postgres", action="store_true",
                    help="write the schemas store_private and store_public of the database $STORE_INDEXER_DATABASE_URL names, not SQLite files")
    ap.add_argument("--jobs", type=int, default=32, metavar="N")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = store_root(ns.store)
    if ns.postgres and not os.environ.get("STORE_INDEXER_DATABASE_URL"):
        print("research data index: --postgres needs STORE_INDEXER_DATABASE_URL in the environment", file=sys.stderr)
        return 2
    if ns.source:
        remote = FsRemote(ns.source)
    else:
        st = LocalStore(ns.store)
        if st.remote is None:
            print(f"research data index: no usable remote ({st.remote_error or 'none configured'}); --from DIR indexes a directory", file=sys.stderr)
            return 2
        remote = st.remote
    log = None if ns.json else (lambda m: print(m, file=sys.stderr))
    export = None
    if ns.public and (export := load_export()) is None:
        print("research data index: --public needs the export bucket (`research data export`'s config)", file=sys.stderr)
        return 2
    if ns.postgres:
        url = os.environ["STORE_INDEXER_DATABASE_URL"]
        db, public_db = BI.PostgresIndex(url, "store_private"), (BI.PostgresIndex(url, "store_public") if ns.public else None)
    else:
        db, public_db = ns.db or BI.default_path(root), ns.public_db or BI.default_path(root, BI.PUBLIC)
    reps = [BI.index(remote, db, jobs=ns.jobs, full=ns.full, log=log)]
    if ns.public:
        reps.append(BI.index(export, public_db, policy=BI.PUBLIC, private=db, jobs=ns.jobs))
    for rep, where in zip(reps, (db, public_db)):
        idx = where if isinstance(where, BI.PostgresIndex) else BI.SqliteIndex(where)
        rep["db"], rep["size"] = (f"postgres {where.schema}" if isinstance(where, BI.PostgresIndex) else str(where)), idx.size()
        idx.close()
    if ns.json:
        _dump(reps)
    else:
        for rep in reps:
            for e in rep["errors"][:20]:
                print(f"research data index: {e}", file=sys.stderr)
            s = rep["size"]
            print(f"{rep['policy']} index {rep['db']}: listed " + ", ".join(f"{p} {n}" for p, n in rep["listed"].items())
                  + "; added " + ", ".join(f"{n} {t}s" for t, n in rep["added"].items()) + f" in {rep['seconds']}s"
                  + (f"; {len(rep['errors'])} error(s)" if rep["errors"] else "")
                  + f"; {s['records']['rows']} records ({s['records']['doc_bytes'] / 1e6:.1f} MB of doc), {s['fields']['rows']} fields, "
                  f"{s['edges']['rows']} edges, {(s['file_bytes'] or 0) / 1e6:.1f} MB on disk")
    return 1 if any(r["errors"] for r in reps) else 0


def cmd_pending(argv: list[str]) -> int:
    """The write-through queue (README §7.2): artifacts, attempts and assertions this store holds that the remote does not
    (`LocalStore.pending`).  `research data push --pending` drains it.  Exit 1 when anything is pending."""
    ap = _ap("pending")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    rep = LocalStore(ns.store).pending()
    if ns.json:
        _dump(rep)
    else:
        for a in rep["artifacts"]:
            print(f"artifact  {a}")
        for rid in rep["attempts"]:
            print(f"attempt   {rid}")
        for lab in rep["labels"]:
            print(f"label     {lab['target']}  {lab['key']}  by {lab['by']}  ({lab['hash'][:12]})")
        n = len(rep["artifacts"]) + len(rep["attempts"]) + len(rep["labels"])
        print(f"{len(rep['artifacts'])} artifacts, {len(rep['attempts'])} attempts, {len(rep['labels'])} labels pending for {rep['remote']}"
              + ("; `research data push --pending` uploads them" if n else ""), file=sys.stderr)
    return 1 if rep["artifacts"] or rep["attempts"] or rep["labels"] else 0


def cmd_refresh(argv: list[str]) -> int:
    """Incremental catch-up from the remote (`LocalStore.refresh`): fetch the manifests, attempts and assertions this store lacks and
    index only those; nothing is wiped.  On a fresh machine this builds the catalog from the remote alone.  Exit 1 on any bad key."""
    ap = _ap("refresh")
    ap.add_argument("--jobs", type=int, default=32, metavar="N")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--payloads", nargs="+", default=[], metavar="KIND", help="also fetch the file payloads of artifacts of these kinds")
    ap.add_argument("--payload-max-bytes", type=int, default=REFRESH_PAYLOAD_MAX, metavar="N", help="skip payloads larger than this")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    rep = st.refresh(jobs=max(1, ns.jobs), log=None if ns.json else (lambda m: print(m, file=sys.stderr)),
                     payload_kinds=ns.payloads, payload_max_bytes=ns.payload_max_bytes)
    if ns.json:
        _dump(rep)
    else:
        for e in rep["errors"][:20]:
            print(f"research data refresh: {e}", file=sys.stderr)
        print(f"refreshed {st.root} from {rep['remote']}: +{rep['manifests']} manifests, +{rep['attempts']} attempts, +{rep['labels']} labels"
              + (f", +{rep['payloads']} payloads" if "payloads" in rep else "") + " "
              f"in {rep['seconds']}s" + (f"; {len(rep['errors'])} error(s)" if rep["errors"] else ""))
    return 1 if rep["errors"] else 0


def cmd_parity(argv: list[str]) -> int:
    """Content parity of this store against a store built from the remote alone (`parity.py`): artifacts, attempts, labels, replica
    rows.  Read-only for the store under test.  Exit 0 iff every difference is explained."""
    from . import parity as PA  # noqa: PLC0415

    ap = _ap("parity")
    ap.add_argument("--fresh", required=True, metavar="DIR", help="where to build the store from the remote alone (a new directory)")
    ap.add_argument("--jobs", type=int, default=32, metavar="N")
    ap.add_argument("--json-out", metavar="FILE", help="write the full report (every difference and its cause) here")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    rep = PA.run(store_root(ns.store), Path(ns.fresh), jobs=max(1, ns.jobs), log=None if ns.json else (lambda m: print(m, file=sys.stderr)))
    if ns.json_out:
        Path(ns.json_out).write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    if ns.json:
        _dump(rep)
    else:
        for line in PA.format_report(rep):
            print(line)
    return 0 if rep["ok"] else 1


def cmd_reconcile_attempts(argv: list[str]) -> int:
    """Local attempt documents that differ from the remote's copy of the same write-once key (`reconcile.py`): listed (dry run), or
    with `--apply` superseded by the remote's copy, the local one kept under attempts/.superseded/ and each change logged in
    <store>/reconcile.log.  Exit 1 when divergent attempts remain (dry run) or one could not be reconciled."""
    from . import reconcile as RC  # noqa: PLC0415

    ap = _ap("reconcile-attempts")
    ap.add_argument("runs", nargs="*", help="run ids (default with --all: every local attempt the remote also holds)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--apply", action="store_true", help="replace each divergent local copy with the remote's (audited)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if not ns.runs and not ns.all:
        print("research data reconcile-attempts: give run ids or --all", file=sys.stderr)
        return 2
    st = LocalStore(ns.store)
    found = RC.diverged(st, ns.runs or None)
    done = RC.apply(st, found) if ns.apply else []
    if ns.json:
        _dump({"divergent": [{k: v for k, v in f.items() if not k.startswith("_")} for f in found], "applied": done})
    else:
        for f in found:
            if f.get("error"):
                print(f"{f['run']}  ERROR {f['error']}")
                continue
            print(f"{f['run']}  local {f['local_sha256'][:12]} != remote {f['remote_sha256'][:12]} (remote written {f['remote_last_modified']})  "
                  f"differs: {', '.join(f['differs'][:6]) or '-'}; remote only: {', '.join(f['remote_only'][:6]) or '-'}; "
                  f"local only: {', '.join(f['local_only'][:6]) or '-'}" + ("  -> SUPERSEDED by the remote's copy" if ns.apply else ""))
        print(f"{len(found)} divergent attempt(s)" + (f"; {len(done)} reconciled, logged in {st.root / RC.LOG}" if ns.apply
                                                      else ("; `--apply` replaces the local copies (kept under attempts/.superseded/)" if found else "")),
              file=sys.stderr)
    return 1 if any(f.get("error") for f in found) or (found and not ns.apply) else 0


def cmd_catalog(argv: list[str]) -> int:
    """The catalog's schema version against this code's, what opening it just migrated (older catalogs are brought forward on every
    open, in one transaction), and row counts per table.  Exit 0 when current, 3 (before printing) when the catalog is NEWER than
    the code -- see `CatalogVersionError`."""
    ap = _ap("catalog")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--compact", type=Path, metavar="OUT", help="write a copy without manifest_json or the replica records (index.compact)")
    ns = ap.parse_args(argv)
    if ns.compact:
        rep = compact(LocalStore(ns.store).index.path, ns.compact)
        if ns.json:
            _dump(rep)
        else:
            print(f"{rep['path']}: {rep['bytes'] / 1e6:.1f} MB; " + ", ".join(f"{t} {n}" for t, n in rep["tables"].items()))
        return 0
    info = LocalStore(ns.store).index.info()
    if ns.json:
        _dump(info)
        return 0
    mig = info["migration"]
    did = (f"migrated {' , '.join(mig['steps'])} on this open" if mig["steps"] else
           ("opened under " + IGNORE_VERSION_ENV if mig.get("ignored") else "current"))
    print(f"{info['path']}: schema v{info['schema_version']} (code v{info['code_version']}); {did}; {info['bytes'] / 1e6:.1f} MB")
    for t, n in info["tables"].items():
        print(f"  {t:<20} {n:>8} rows")
    return 0


def cmd_sql(argv: list[str]) -> int:
    ap = _ap("sql")
    ap.add_argument("query")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--index", action="store_true", help="query the index's records, fields, edges and listings (README §5.1): Neon's "
                    "store_private with $STORE_PRIVATE_READER_URL, else this machine's `research data index` build; not the catalog")
    ns = ap.parse_args(argv)
    if ns.index:
        return _sql_index(ns)
    idx = LocalStore(ns.store).index
    try:
        cols, rows = idx.sql_with_columns(ns.query)
    except sqlite3.OperationalError as e:
        if not str(e).startswith("no such"):
            raise
        print(f"research data sql: {e}", file=sys.stderr)
        for (t,) in idx.sql_with_columns("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")[1]:
            print(f"  {t}({', '.join(r[1] for r in idx.sql_with_columns(f'PRAGMA table_info({t})')[1])})", file=sys.stderr)
        print("  (custody: `research data preserved ART|RUN...`, not hand-written SQL)", file=sys.stderr)
        return 2
    if ns.json:
        _dump([dict(zip(cols, r)) for r in rows])
    else:
        if cols:
            print("\t".join(cols))
        for r in rows:
            print("\t".join("" if v is None else str(v) for v in r))
    return 0


def _sql_index(ns: argparse.Namespace) -> int:
    from . import bucket_index as BI  # noqa: PLC0415

    st = LocalStore(ns.store)
    idx = BI.reader(st.root)
    if idx is None:
        print(f"research data sql --index: no index here: set {BI.READER_ENV} (store_private_reader's URL) or build one with "
              "`research data index`", file=sys.stderr)
        return 2
    try:
        try:
            cols, rows = idx.query(ns.query)
        except (sqlite3.Error, RuntimeError) as e:
            print(f"research data sql --index: {e}\n  tables: records(id, type, kind, time, doc), fields(id, path, value), "
                  "edges(src, name, dst), listings(prefix, listed_at, keys)", file=sys.stderr)
            return 2
        if ns.json:
            _dump([dict(zip(cols, r)) for r in rows])
        else:
            if cols:
                print("\t".join(cols))
            for r in rows:
                print("\t".join("" if v is None else str(v) for v in r))
        n = len(BI.pending(st, idx))
        print(f"research data sql --index: {idx.describe()}, complete as of {BI.as_of(idx) or 'never listed'}"
              + (f"; {n} record(s) here are not in it (`research data pending`)" if n else ""), file=sys.stderr)
        return 0
    finally:
        idx.close()


def _attempt_artifacts(doc: dict) -> list[str]:
    out = []
    for sec in ("outputs", "telemetry"):
        for v in (doc.get(sec) or {}).values():
            if isinstance(v, str) and v.startswith("art:"):
                out.append(v)
    return out


def _push_line(x: dict) -> str:
    """One line per artifact: outcome, then objects / uploaded / skipped, MB, seconds, objects/s and the verification done."""
    head = f"{x['id']}  {x['kind']}  {'PRESERVED' if x['preserved'] else 'NOT preserved: ' + '; '.join(x.get('errors') or ['absent'])}"
    if not x.get("objects"):
        return head
    s = x.get("seconds") or 0.0
    rate = f"{x['objects'] / s:.1f} obj/s" if s > 0 else "-"
    return (f"{head}  {x['objects']} objects ({x.get('uploaded', 0)} uploaded, {x.get('skipped', 0)} already there) "
            f"{x.get('bytes', 0) / 1e6:.1f} MB in {s:.1f}s  {rate}  verified {x.get('verified') or '-'}")


def cmd_push(argv: list[str]) -> int:
    """Preserve on the remote (README §7): an artifact's blobs go up concurrently (`--jobs`), each verified (`--verify head`: size +
    MD5 ETag, the default; `--verify readback`: download + sha256), then its manifest; a run id pushes the attempt manifest, every
    artifact it references, and the assertions about the attempt, its derivation and those artifacts (`labels_sync.push_targets`; an
    explicit art: id pushes its own assertions) -- an attempt whose assertions did not all land is reported NOT preserved.  `--pending`
    pushes every local artifact without a verified remote replica, every attempt the remote lacks, and every local assertion the remote
    lacks.  One object's failure is reported, the batch continues; exit 1 if anything is not PRESERVED."""
    ap = _ap("push")
    ap.add_argument("targets", nargs="*", help="art:<hex> or run ids")
    ap.add_argument("--pending", action="store_true", help="everything local that the remote does not verifiably hold")
    ap.add_argument("--jobs", type=int, default=32, metavar="N", help="concurrent uploads per artifact (default 32)")
    ap.add_argument("--verify", choices=["head", "readback"], default="head",
                    help="per-object check: head = Content-Length + single-part ETag == MD5 (default); readback = download + sha256")
    ap.add_argument("--no-labels", action="store_true", help="do not push assertions (a run's: about it, its derivation and its outputs; --pending: all local ones)")
    ap.add_argument("--skip-record-less", action="store_true",
                    help="push no attempt that names no run record (--pending: one the remote lacks keeps its logs only here)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if ns.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    st = LocalStore(ns.store)
    from .local import attempt_key  # noqa: PLC0415

    r = st._require_remote()  # noqa: SLF001  (names the missing credentials when the remote is configured but unusable here)
    arts: list[str] = []
    runs: list[str] = []
    for t in ns.targets:
        (arts if t.startswith("art:") else runs).append(t)
    if ns.pending:                                                 # the queue `research data pending` shows
        queue = st.pending()
        arts += queue["artifacts"]
        runs += queue["attempts"]
    reports: list[dict] = []
    record_less: list[str] = []
    label_targets: dict[str, list[str]] = {}                       # report id -> the label targets its durability covers
    from . import labels_sync as LS  # noqa: PLC0415

    for rid in dict.fromkeys(runs):
        if not ids.is_run_id(rid):
            raise ValueError(f"not an artifact or run id: {rid!r}")
        doc = st.get_attempt(rid)
        if doc is None:
            raise StoreError(f"attempt {rid} is not in the store")
        if ns.skip_record_less and not (doc.get("outputs") or {}).get("run_record"):
            record_less.append(rid)
            continue
        arts += _attempt_artifacts(doc)
        label_targets[rid] = LS.attempt_targets(doc)
        try:                                                       # one bad upload must not abandon the batch: record, continue
            st.push_attempt(rid)
            reports.append({"id": rid, "kind": "attempt", "preserved": r.head(attempt_key(rid)) is not None})
        except Exception as e:  # noqa: BLE001
            reports.append({"id": rid, "kind": "attempt", "preserved": False, "errors": [f"{type(e).__name__}: {e}"]})
    if record_less:
        print(f"research data push: {len(record_less)} attempt(s) without a run record not pushed (--skip-record-less): "
              + " ".join(record_less[:5]) + (" …" if len(record_less) > 5 else ""), file=sys.stderr)
    for a in arts:                                                 # an artifact's own assertions count against it as well (explicit or via a run)
        label_targets.setdefault(a, [a])
    def heartbeat(p: dict) -> None:                                # long artifacts (10^5 objects): a running count on stderr, no --json noise
        if not ns.json and sys.stderr.isatty():
            print(f"\r  {p['id'][:16]}… {p['done']}/{p['objects'] - 1} blobs  {p['done'] / max(p['seconds'], 1e-9):.1f} obj/s", end="", file=sys.stderr, flush=True)

    notes_seen: set[str] = set()
    for a in dict.fromkeys(arts):
        try:
            rep = st.push(a, preserve=True, jobs=ns.jobs, verify=ns.verify, progress=heartbeat)
            reports.append({"id": a, "kind": st.get_manifest(a).kind, **{k: rep.get(k) for k in ("preserved", "errors", "verified", "objects", "uploaded", "skipped", "bytes", "seconds", "notes")}})
        except Exception as e:  # noqa: BLE001
            reports.append({"id": a, "kind": "?", "preserved": False, "errors": [f"{type(e).__name__}: {e}"]})
        if not ns.json:
            x = reports[-1]
            if sys.stderr.isatty() and x.get("objects"):
                print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
            print(_push_line(x), flush=True)
            for note in x.get("notes") or []:
                if note not in notes_seen:                             # e.g. the remote switching conditional PUT off: said once
                    notes_seen.add(note)
                    print(f"research data push: note: {note}", file=sys.stderr)
    labels_rep = None
    if not ns.no_labels:                                            # labels are assertions about what was just pushed: they travel with it
        try:
            if ns.pending:                                          # everything local the remote lacks (subsumes the targets' assertions)
                labels_rep = LS.sync(st, push=True, pull=False, jobs=ns.jobs)
            elif label_targets:                                     # the attempts' closures (attempt, derivation, outputs) and the named artifacts
                labels_rep = LS.push_targets(st, [t for ts in label_targets.values() for t in ts], jobs=ns.jobs)
        except Exception as e:  # noqa: BLE001
            labels_rep = {"error": f"{type(e).__name__}: {e}", "pushed": [], "local": None, "already": None, "failed": []}
    if labels_rep and labels_rep.get("targets"):                    # an attempt / artifact whose assertions did not all land is NOT preserved
        for x in reports:
            failed = [e for t in label_targets.get(x["id"], []) for e in labels_rep["targets"].get(t, {}).get("failed", [])]
            if failed:
                x["preserved"] = False
                x["errors"] = [*(x.get("errors") or []), f"labels: {len(failed)} assertion(s) not pushed ({failed[0]})"]
    bad = [x for x in reports if not x["preserved"]]
    if ns.json:
        _dump({"remote": r.describe(), "pushed": reports, "not_preserved": len(bad), "labels": labels_rep})
    else:
        for x in reports:
            if x["kind"] == "attempt":
                print(f"{x['id']}  attempt  {'PRESERVED' if x['preserved'] else 'NOT preserved: ' + '; '.join(x.get('errors') or ['absent'])}")
        print(f"{len(reports) - len(bad)}/{len(reports)} preserved on {r.describe()}")
        if labels_rep is not None and (ns.pending or labels_rep.get("local") or labels_rep.get("failed") or "error" in labels_rep):
            if "error" in labels_rep:
                print(f"labels: NOT pushed: {labels_rep['error']}")
            else:
                print(f"labels: {len(labels_rep['pushed'])} pushed, {labels_rep['already']} already there"
                      + (f", {len(labels_rep['failed'])} NOT pushed: {labels_rep['failed'][0]}" if labels_rep.get("failed") else ""))
    return 1 if bad or (labels_rep and ("error" in labels_rep or labels_rep.get("failed"))) else 0


def cmd_mint_credential(argv: list[str]) -> int:
    """A short-lived, bucket-scoped R2 credential for a pod (`credentials.py`): signed locally with the parent key from the environment
    (the env vars store.toml names, default AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), or via the Cloudflare API with
    CLOUDFLARE_API_TOKEN.  Prints JSON, or with `--env` the three `export` lines for a bootstrap; never writes a file."""
    from . import credentials as C  # noqa: PLC0415

    ap = _ap("mint-credential")
    ap.add_argument("--ttl", default="1h", help="lifetime: <n>[s|m|h|d], max 7d (default 1h)")
    ap.add_argument("--permission", default="object-read-write", choices=C.SCOPES, help="scope preset (default object-read-write)")
    ap.add_argument("--prefix", action="append", default=[], metavar="P", help="restrict to keys under this prefix (repeatable)")
    ap.add_argument("--object", action="append", default=[], metavar="KEY", help="restrict to this exact key (repeatable)")
    ap.add_argument("--actions", default=None, metavar="A,B", help="explicit S3 actions, e.g. PutObject,HeadObject,GetObject (local signing only)")
    ap.add_argument("--via", default="auto", choices=["auto", "local", "api"], help="local = sign a JWT with the parent secret; api = Cloudflare API")
    ap.add_argument("--endpoint", default=None, help="override store.toml / R2_ENDPOINT")
    ap.add_argument("--bucket", default=None, help="override store.toml / R2_BUCKET")
    ap.add_argument("--account-id", default=None, help="override R2_ACCOUNT_ID / the endpoint's <account> host part")
    ap.add_argument("--env", action="store_true", help="print `export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...`")
    ap.add_argument("--json", action="store_true", help="(default output) the credential with bucket, scope, expiry, how it was minted")
    ns = ap.parse_args(argv)
    actions = [a for a in (ns.actions or "").split(",") if a] or None
    cred = C.mint(via=ns.via, scope=ns.permission, ttl_s=C.parse_ttl(ns.ttl), prefixes=ns.prefix or None, objects=ns.object or None, actions=actions,
                  endpoint=ns.endpoint, bucket=ns.bucket, account_id=ns.account_id)
    if ns.env:
        print(C.env_lines(cred))
        print(f"# {cred['scope'] or 'actions-scoped'} on s3://{cred['bucket']} until {cred['expires_utc']} (minted {cred['via']}"
              + (f", prefixes {cred['prefixes']}" if cred["prefixes"] else "") + (f", actions {cred['actions']}" if cred["actions"] else "") + ")")
    else:
        _dump({k: v for k, v in cred.items() if k != "jwt"})
    return 0


def cmd_refresh_fixtures(argv: list[str]) -> int:
    """Re-register every `fixtures/artifacts.json` entry whose bytes changed (`fixtures.refresh`): put under the new id (content-only
    identity, as the entries were registered), `--preserve` pushes it, the entry is rewritten and the old id kept under `previous`.
    Prints `old -> new  path` per changed entry.  Exit 1 when an entry's path is missing, conflicts, or was not PRESERVED."""
    from . import fixtures as F  # noqa: PLC0415

    ap = _ap("refresh-fixtures")
    ap.add_argument("--repo", default=None, metavar="DIR", help="repository root (default $RESEARCH_REPO or the nearest ancestor of cwd with fixtures/artifacts.json)")
    ap.add_argument("--preserve", action="store_true", help="push each re-registered fixture to the remote and verify (as `put --preserve`)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; put nothing, write nothing")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store, repo=Path(ns.repo).expanduser().resolve() if ns.repo else None)
    repo = st.repo
    if repo is None or not (repo / F.REPLICA_MAP).is_file():
        print(f"research data refresh-fixtures: no {F.REPLICA_MAP} found (give --repo DIR or set RESEARCH_REPO)", file=sys.stderr)
        return 1
    rep = F.refresh(st, repo, preserve=ns.preserve, dry_run=ns.dry_run)
    bad = [e for e in rep["entries"] if e["status"] in ("missing", "conflict") or (ns.preserve and e["status"] == "changed" and not e.get("preserved"))]
    if ns.json:
        _dump(rep)
        return 1 if bad else 0
    for e in rep["entries"]:
        if e["status"] == "changed":
            older = [x for x in e["previous"] if x != e["id"]]
            print(f"{e['id']} -> {e['new_id']}  {e['path']}" + (f"  (previously also {', '.join(older)})" if older else "")
                  + ("" if not ns.preserve or ns.dry_run else ("  PRESERVED" if e["preserved"] else "  NOT preserved: " + "; ".join(e["errors"]))))
        elif e["status"] == "missing":
            print(f"{e['id']}  {e['path']}  MISSING: not in the working tree")
        elif e["status"] == "conflict":
            print(f"{e['id']}  {e['path']}  CONFLICT: " + "; ".join(e["errors"]))
    what = "dry run, nothing written" if ns.dry_run else ("map rewritten" if rep["written"] else "map unchanged")
    print(f"{len(rep['entries'])} fixture(s) in {rep['map']}: {rep['changed']} changed, {rep['unchanged']} unchanged, {rep['missing']} missing; {what}",
          file=sys.stderr)
    return 1 if bad else 0


def _fixture_repo(st: LocalStore, cmd: str, repo: str | None) -> Path | None:
    from . import fixtures as F  # noqa: PLC0415

    root = Path(repo).expanduser().resolve() if repo else st.repo
    if root is None or not (root / F.REPLICA_MAP).is_file():
        print(f"research data {cmd}: no {F.REPLICA_MAP} found (give --repo DIR or set RESEARCH_REPO)", file=sys.stderr)
        return None
    return root


def _repo_rel(root: Path, p: str) -> str:
    """A PATH argument as a repo-relative path: absolute, or relative to cwd when that lands inside the repo, else as given."""
    q = Path(p).expanduser()
    q = q if q.is_absolute() else Path.cwd() / q
    try:
        return q.resolve().relative_to(root).as_posix() if q.resolve() != root else "."
    except ValueError:
        return p.rstrip("/")


def _blob_list(path: str | None) -> list[tuple[str, int]]:
    """`--blobs FILE`: one `<sha256> <bytes>` per line (the history rewrite's stubbed blobs)."""
    if not path:
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            h, n = line.split()[:2]
            if not ids.is_sha256(h):
                raise ValueError(f"--blobs {path}: {h!r} is not a sha256")
            out.append((h, int(n)))
    return out


def cmd_fetch_fixtures(argv: list[str]) -> int:
    """Make the fetched entries of `fixtures/artifacts.json` under each PATH present and exactly their ids (`fixtures.fetch_registry`):
    manifests and blobs come from the local store, else the private remote when this machine can reach it, else the public export
    (`--public-url`, `$RESEARCH_PUBLIC_URL`, else the built-in URL), `--jobs` at a time, every byte checked.  A tracked entry is
    left alone.  `--check` only reports (exit 1 if any is missing or differs).  `--stubs` also restores the history stubs in
    tracked files (an old commit's checkout)."""
    from . import fetchset, fixtures as F  # noqa: PLC0415

    ap = _ap("fetch-fixtures")
    ap.add_argument("paths", nargs="*", metavar="PATH", help="only entries under these paths (default: all)")
    ap.add_argument("--repo", default=None, metavar="DIR", help="repository root (default $RESEARCH_REPO or the nearest ancestor of cwd with fixtures/artifacts.json)")
    ap.add_argument("--check", action="store_true", help="report what is missing or different; fetch and write nothing")
    ap.add_argument("--stubs", action="store_true", help="also restore history stubs (verity-fixture/v1) in tracked files")
    ap.add_argument("--public-url", default=None, metavar="URL", help=f"the public export (default ${fetchset.PUBLIC_URL_ENV}, else the built-in one; '' for none)")
    ap.add_argument("--no-private", action="store_true", help="skip the private remote even where it is reachable (test the public path)")
    ap.add_argument("--jobs", type=int, default=fetchset.JOBS)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store, repo=Path(ns.repo).expanduser().resolve() if ns.repo else None)
    root = _fixture_repo(st, "fetch-fixtures", ns.repo)
    if root is None:
        return 1
    srcs = fetchset.sources(st, public=ns.public_url, private=not ns.no_private)
    paths = [_repo_rel(root, p) for p in ns.paths]
    rep = F.fetch_registry(st, root, paths, srcs=srcs, jobs=ns.jobs, check_only=ns.check)
    if ns.stubs and not ns.check:
        rep["stubs"] = F.hydrate_stubs(st, root, paths, srcs=srcs, jobs=ns.jobs)
        rep["ok"] = rep["ok"] and rep["stubs"]["ok"]
    if ns.json:
        _dump(rep)
        return 0 if rep["ok"] else 1
    n = {}
    for e in rep["entries"]:
        n[e["status"]] = n.get(e["status"], 0) + 1
        if e["status"] in ("missing", "stale"):
            print(f"{e['status'].upper():8} {e['path']}  {e['id']}" + (f"  differs: {', '.join(e['differs'])}" if e.get("differs") else ""))
    for x in rep["fetch"]["missing"]:
        print(f"         {x.get('id') or x.get('sha256')}: " + "; ".join(x["tried"]), file=sys.stderr)
    fb = rep["fetch"]["blobs"]
    got = f"; {fb['fetched']} blob(s), {fb['bytes'] / 1e6:.1f} MB fetched" + (f" from {', '.join(rep['fetch']['by_source'])}" if rep["fetch"]["by_source"] else "") \
        if fb["fetched"] else ""
    stubs = f"; {len(rep['stubs']['hydrated'])}/{rep['stubs']['stubs']} stub(s) restored" if "stubs" in rep else ""
    print(f"fetch-fixtures: {len(rep['entries'])} entr{'y' if len(rep['entries']) == 1 else 'ies'}: "
          + ", ".join(f"{v} {k}" for k, v in sorted(n.items())) + got + stubs
          + ("" if rep["sources"] else "; no source configured (no reachable private remote, no public URL)"), file=sys.stderr)
    if not rep["ok"] and not ns.check:
        print("fetch-fixtures: some fixtures could not be fetched; a test reading them fails until they are", file=sys.stderr)
    return 0 if rep["ok"] else 1


def cmd_export(argv: list[str]) -> int:
    """Copy every `fixtures/artifacts.json` entry, with the ids under its `previous` block, each `--id` and any `--blobs`, to the
    public export (`remote.load_export`: the `[export]` table of store.toml, `$RESEARCH_EXPORT_DIR`, or `$RESEARCH_EXPORT_BUCKET`), taking the
    bytes from the repo, the local store or the private remote.  `--verify-url URL` exports nothing: it fetches all of it back
    through URL into a scratch store and checks every byte against its id (the end-to-end test of a public front end)."""
    from . import fetchset, fixtures as F  # noqa: PLC0415
    from .remote import load_export  # noqa: PLC0415

    ap = _ap("export")
    ap.add_argument("--repo", default=None, metavar="DIR", help="repository root (default $RESEARCH_REPO or the nearest ancestor of cwd with fixtures/artifacts.json)")
    ap.add_argument("--dry-run", action="store_true", help="HEAD the export and report what would be uploaded")
    ap.add_argument("--no-previous", action="store_true", help="leave out the ids under the map's `previous` block")
    ap.add_argument("--blobs", default=None, metavar="FILE", help="also these blobs: one `<sha256> <bytes>` per line (history stubs)")
    ap.add_argument("--id", action="append", default=[], metavar="ART", help="also this artifact, which the registry doesn't hold; repeatable")
    ap.add_argument("--verify-url", default=None, metavar="URL", help="fetch everything back through URL and check it; export nothing")
    ap.add_argument("--jobs", type=int, default=fetchset.JOBS)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store, repo=Path(ns.repo).expanduser().resolve() if ns.repo else None)
    root = _fixture_repo(st, "export", ns.repo)
    if root is None:
        return 1
    blobs = _blob_list(ns.blobs)
    if ns.verify_url:
        rep = F.verify_url(root, ns.verify_url, include_previous=not ns.no_previous, blobs=blobs, jobs=ns.jobs)
        if ns.json:
            _dump(rep)
        else:
            for x in rep["fetch"]["missing"]:
                name = x.get("id") or x.get("sha256")
                print(f"{'WARN   ' if name in rep['warnings'] else 'MISSING'}  {name}: " + "; ".join(x["tried"]))
            f = rep["fetch"]
            print(f"export --verify-url {ns.verify_url}: {rep['manifests']}/{rep['ids']} manifest(s) and {f['blobs']['fetched']} blob(s) "
                  f"({f['blobs']['bytes'] / 1e6:.1f} MB) fetched and checked; {len(rep['missing'])} missing, "
                  f"{len(rep['warnings'])} superseded id(s) nobody holds", file=sys.stderr)
        return 0 if rep["ok"] else 1
    dest = load_export(st._config_path)
    if dest is None:
        print("research data export: no export configured ([export] in store.toml, RESEARCH_EXPORT_DIR or RESEARCH_EXPORT_BUCKET)", file=sys.stderr)
        return 1
    rep = F.export(st, root, dest, srcs=fetchset.sources(st, public=""), include_previous=not ns.no_previous, blobs=blobs,
                   arts=ns.id, jobs=ns.jobs, dry_run=ns.dry_run)
    if ns.json:
        _dump(rep)
    else:
        for e in rep["errors"]:
            print(f"ERROR  {e}")
        for w in rep["warnings"]:
            print(f"WARN   {w}")
        verb = "would upload" if ns.dry_run else "uploaded"
        print(f"export to {rep['dest']}: {rep['ids']} id(s), {rep['blobs']} blob(s); {verb} {rep['uploaded']} object(s) "
              f"({rep['bytes_uploaded'] / 1e6:.1f} MB), {rep['skipped']} already there; {len(rep['errors'])} error(s)", file=sys.stderr)
    return 0 if rep["ok"] else 1


def _json_object(p: Path) -> dict:
    """A run record as a dict; an unreadable or non-object file reads as `{"error": ...}` (a doctor reports, never trips)."""
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"error": f"unreadable: {e}"}
    return doc if isinstance(doc, dict) else {"error": "not a JSON object"}


def cmd_doctor(argv: list[str]) -> int:
    """Run dirs with a `store_publish_error.json` (README §10) whose run *is* now an attempt in this store -- a stale record of a publish
    that later succeeded (a back-fill, a retry) -- versus those whose run is not (a real failure, or a run still going).  Read-only;
    `--clean` deletes the stale files and nothing else.  Local attempts only: `reindex --remote` first if an attempt may live only on the vault."""
    from ..runs import runs_root  # noqa: PLC0415  (the run registry is the runner's half of the package)

    ap = _ap("doctor")
    ap.add_argument("--runs-dir", default=None, metavar="DIR", help="runs root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--clean", action="store_true", help="delete the stale store_publish_error.json files (never anything else)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    root = runs_root(ns.runs_dir)
    st = LocalStore(ns.store)
    stale: list[dict] = []
    real: list[dict] = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        p = d / "store_publish_error.json"
        if not d.is_dir() or not p.is_file():
            continue
        err = _json_object(p)
        status = _json_object(d / "status.json") if (d / "status.json").is_file() else {}
        row = {"run": d.name, "path": str(p), "error": str(err.get("error") or "?").splitlines()[0][:200], "at_utc": err.get("at_utc"),
               "state": status.get("state"), "deleted": False}
        (stale if ids.is_run_id(d.name) and st.attempt_path(d.name).exists() else real).append(row)
    if ns.clean:
        for row in stale:
            Path(row["path"]).unlink()
            row["deleted"] = True
    if ns.json:
        _dump({"runs_dir": str(root), "store": str(st.root), "stale": stale, "real": real, "cleaned": ns.clean})
        return 0
    for row in stale:
        print(f"{row['run']}  stale  {'deleted' if row['deleted'] else 'attempt in the store'}  {row['error']}")
    for row in real:
        print(f"{row['run']}  REAL   state={row['state'] or '?'}  {row['error']}")
    n = len(stale) + len(real)
    print(f"{n} store_publish_error.json file(s) under {root}: {len(stale)} stale (run is an attempt in {st.root}), {len(real)} real (not in the store)"
          + (f"; deleted {len(stale)} stale file(s), nothing else touched" if ns.clean else "; read-only (--clean deletes the stale ones)"), file=sys.stderr)
    return 0


def cmd_gc(argv: list[str]) -> int:
    """README §7.4 (`gc.py`): which blobs only uncited artifacts past their kind's TTL hold; `--apply` marks them and deletes those
    marked at least `gc.GRACE_S` ago that still are.  Read-only without `--apply`."""
    from . import gc  # noqa: PLC0415

    ap = _ap("gc")
    ap.add_argument("--cite", action="append", default=[], metavar="DIR", required=True,
                    help="a tree whose files cite art ids and runs (the notes repository; a checkout of the code); repeatable")
    ap.add_argument("--cache", required=True, metavar="DIR", help="where manifests, labels and attempts read from the remote are kept")
    ap.add_argument("--state", metavar="FILE", help="the marks (required with --apply)")
    ap.add_argument("--apply", action="store_true", help="mark this pass's candidates and delete the ones marked a grace ago")
    ap.add_argument("--report", metavar="FILE", help="write the plan's summary and the expired ids here (JSON)")
    ap.add_argument("--work", metavar="DIR", help="where the pass's blob table goes (a local disk; default the temp dir)")
    ap.add_argument("--cite-linear", action="store_true", help="Linear's issues, comments, projects, project updates and "
                    "documents cite too (research.linear's credentials; required with --apply)")
    ap.add_argument("--jobs", type=int, default=gc.JOBS)
    ns = ap.parse_args(argv)
    if ns.apply and not (ns.state and ns.cite_linear):
        ap.error("--apply needs --state and --cite-linear (lanes cite in Linear most: top, 26 Oct)")
    st = LocalStore(ns.store)
    if st.remote is None:
        print("research data gc: no remote configured", file=sys.stderr)
        return 2
    say = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    try:
        linear = None
        if ns.cite_linear:
            from ..linear import Linear  # noqa: PLC0415
            linear = Linear().gql
        p = gc.plan(st.remote, Path(ns.cache), [Path(c) for c in ns.cite], now=gc.clock(), jobs=ns.jobs,
                    work=Path(ns.work) if ns.work else None, linear=linear, log=say)
        out: dict[str, Any] = {"remote": st.remote.describe(), "ttl_days": gc.TTL_DAYS, **p.summary()}
        if ns.apply:
            out["applied"] = gc.apply(st.remote, p, Path(ns.cache), Path(ns.state), now=gc.clock(), jobs=ns.jobs, log=say)
    except gc.GcError as e:
        print(f"research data gc: nothing deleted: {e}", file=sys.stderr)
        return 1
    if ns.report:
        Path(ns.report).write_text(json.dumps({**out, "expired": p.expired}, sort_keys=True) + "\n")
    _dump(out)
    applied = out.get("applied") or {}
    return 1 if applied.get("error_count") or applied.get("lost_count") else 0


def cmd_withdraw(argv: list[str]) -> int:
    """README §7.4 (`withdraw.py`): delete the blobs of artifacts holding values that must never be shared, cited or not, and label
    each `withdrawn` with the reason.  A blob another artifact also holds is kept and named.  Read-only without `--apply`."""
    from . import gc, withdraw

    ap = _ap("withdraw")
    ap.add_argument("arts", nargs="+", metavar="ART", help="full art: ids")
    ap.add_argument("--reason", required=True, help="why, in a few words: the `withdrawn` label's value")
    ap.add_argument("--ref", required=True, help="the issue or ruling that asked for it (INF-333)")
    ap.add_argument("--as", dest="by", required=True, metavar="HANDLE", help="the handle withdrawing them: the labels' `by`")
    ap.add_argument("--cache", required=True, metavar="DIR", help="where manifests read from the remote are kept (gc's cache works)")
    ap.add_argument("--work", metavar="DIR", help="where the pass's blob table goes (default the temp dir)")
    ap.add_argument("--path", dest="paths", action="append", metavar="PREFIX",
                    help="withdraw only the files under PREFIX in the arts' trees (repeatable; a run's out/sampled-proofs/served, say)")
    ap.add_argument("--apply", action="store_true", help="delete the blobs and write the labels")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    if st.remote is None:
        print("research data withdraw: no remote configured", file=sys.stderr)
        return 2
    say = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    try:
        p = withdraw.plan(st.remote, Path(ns.cache), ns.arts, work=Path(ns.work) if ns.work else None, log=say, paths=ns.paths)
        out: dict[str, Any] = {"remote": st.remote.describe(), **p.summary()}
        if p.missing:
            _dump(out)
            print("research data withdraw: no manifest for some ids; nothing deleted", file=sys.stderr)
            return 1
        if ns.apply:
            out["applied"] = withdraw.apply(st.remote, p, log=say)
            lost = False
            for art in p.arts:
                value = f"{ns.reason} (only {', '.join(ns.paths)})" if ns.paths else ns.reason
                st.add_label(art, vocab.WITHDRAWN, value, ns.by, ref=ns.ref)
                said, bad = _label_write_through(st, art)
                say(f"{art}: withdrawn label {said}")
                lost |= bad
            out["labels_lost"] = lost
    except gc.GcError as e:
        print(f"research data withdraw: nothing deleted: {e}", file=sys.stderr)
        return 1
    _dump(out)
    return 1 if out.get("labels_lost") else 0


_VERBATIM_AFTER = ("--meta", "--note")      # free text / JSON values: never rewritten
_VERBATIM_COMMANDS = ("sql",)


def _store_root_arg(argv: list[str]) -> str | None:
    for i, t in enumerate(argv):
        if t == "--store" and i + 1 < len(argv):
            return argv[i + 1]
        if t.startswith("--store="):
            return t.split("=", 1)[1]
    return None


def _rewrite_token(tok: str, resolve: Callable[[str], str]) -> str:
    """The token itself, a comma list (`--members a,b`), or `name=<comma list>` (`--ref proof=art:…`, `--where meta.k=art:…`),
    with every abbreviated id replaced; any other token comes back unchanged."""
    if tok.startswith("--") and "=" in tok:
        flag, _, val = tok.partition("=")
        return tok if flag in _VERBATIM_AFTER else f"{flag}={_rewrite_token(val, resolve)}"
    head, sep, body = tok.partition("=") if "=" in tok else ("", "", tok)
    parts = body.split(",")
    if any(ids.is_short_art(p) for p in parts) and all(ids.is_art(p) or ids.is_short_art(p) for p in parts):
        return head + sep + ",".join(resolve(p) if ids.is_short_art(p) else p for p in parts)
    return tok


def expand_short_ids(cmd: str, argv: list[str]) -> list[str]:
    """Abbreviated artifact ids (`art:cc59294a`, >= 8 hex digits: the form notes and reports print) resolved against the local
    catalog, git-style: exactly one catalogued id must start with it.  None -> the catalog may simply not know a remote-only
    artifact (`reindex --remote`); several -> ambiguous.  Both are a ValueError naming what to do.  The catalog is opened only
    when some token needs it."""
    if cmd in _VERBATIM_COMMANDS:
        return argv
    st: LocalStore | None = None

    def resolve(short: str) -> str:
        nonlocal st
        st = st or LocalStore(_store_root_arg(argv))
        hits = st.resolve_art(short, limit=6)
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise ValueError(f"{short} matches no artifact in the local catalog ({st.root})" + (" or on the remote" if st.remote else "")
                             + "; give the full id, or run `research data refresh` to catalogue what the remote holds")
        shown = ", ".join(h[:20] + "…" for h in hits[:5]) + (", …" if len(hits) > 5 else "")
        raise ValueError(f"{short} is ambiguous ({shown}); give more digits")

    out: list[str] = []
    for i, tok in enumerate(argv):
        out.append(tok if i and argv[i - 1] in _VERBATIM_AFTER else _rewrite_token(tok, resolve))
    return out


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "put": cmd_put, "show": cmd_show, "where": cmd_where, "fetch": cmd_fetch, "verify": cmd_verify, "preserved": cmd_preserved, "select": cmd_select,
    "snapshot": cmd_snapshot,
    "labels": cmd_labels, "label": cmd_label, "reindex": cmd_reindex, "index": cmd_index, "sql": cmd_sql, "push": cmd_push, "mint-credential": cmd_mint_credential,
    "vocab": cmd_vocab, "refresh-fixtures": cmd_refresh_fixtures, "fetch-fixtures": cmd_fetch_fixtures, "export": cmd_export,
    "doctor": cmd_doctor, "evict": cmd_evict, "labels-sync": cmd_labels_sync,
    "catalog": cmd_catalog, "vocab-check": cmd_vocab_check, "pending": cmd_pending, "refresh": cmd_refresh,
    "parity": cmd_parity, "reconcile-attempts": cmd_reconcile_attempts, "gc": cmd_gc, "withdraw": cmd_withdraw,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    cmd, rest = argv[0], argv[1:]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"research data: unknown command {cmd!r} (have: {', '.join(sorted(COMMANDS))})", file=sys.stderr)
        return 2
    try:
        return fn(expand_short_ids(cmd, rest))
    except CatalogVersionError as e:                                  # the worktree is behind the catalog: say so, exit 3, no traceback
        print(f"research data {cmd}: {e}", file=sys.stderr)
        return 3
    except (StoreError, ConfigError, RemoteConflict, ValueError, OSError, NotImplementedError, sqlite3.Error) as e:
        print(f"research data {cmd}: {e}", file=sys.stderr)
        return 1


try:  # the provenance half (lookup / pull / attempt) plugs in when present
    from .cli_prov import COMMANDS as _PROV

    COMMANDS.update(_PROV)
except ImportError:
    pass


if __name__ == "__main__":
    sys.exit(main())
