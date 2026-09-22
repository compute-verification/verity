"""`research data …` (README §9): the evidence store from the command line.

  research data put  --kind K [--meta JSON|@file] [--ref name=art:...]... [--file PATH | --tree DIR] [--preserve]   -> art id
  research data show ART|ATTEMPT|DRV [--json]      manifest / attempt / attempts with that derivation
  research data where ART [--json]                 replicas: local / repo / remote (+ verified_utc)
  research data push [ART|RUN ...] [--pending] [--jobs N] [--verify head|readback]
                                                   preserve on the remote: blobs uploaded concurrently and verified, then the manifest (PRESERVED);
                                                   a run pushes its attempt + outputs
  research data fetch ART [--to DIR]               materialise the payload (file or tree); prints the path
  research data verify ART [--json]                PRESERVED check against the remote (exit 1 when not preserved)
  research data select [--kind K] [--where meta.key=value]... [--tool T] [--derivation D] [--attempt R] [--label k=v]... [--json]
  research data snapshot --name NAME (--members art:... | --select <select flags>) [--note TEXT]   -> art id
  research data labels TARGET [--history] [--key K] [--by PAT] [--json]
  research data label  TARGET KEY VALUE --by WHO [--ref REF] [--value-json]
  research data reindex [--remote]
  research data sql "SELECT ..." [--json]          read-only connection
  research data mint-credential [--ttl 6h] [--permission object-read-write] [--prefix P]... [--actions A,B] [--via auto|local|api] [--env]
                                                   a short-lived bucket-scoped R2 credential for a pod (JSON, or --env export lines)
  (lookup / pull / attempt are provided by cli_prov when present)

Every command takes `--store ROOT` (default `$RESEARCH_STORE` or `~/.research/store`).  `COMMANDS` maps a name to
`fn(argv) -> int`; `main(argv)` dispatches.  Structured output has `--json` everywhere.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Callable

from . import ids
from .local import LocalStore, StoreError
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


def _select(st: LocalStore, ns: argparse.Namespace) -> tuple[list[dict], dict]:
    labels = [tuple(next(iter(_kv([x], "--label").items()))) for x in ns.label]
    query = {k: v for k, v in (("kind", ns.kind), ("where", _kv(ns.where, "--where") or None), ("tool", ns.tool), ("derivation", ns.derivation),
                               ("attempt", ns.attempt), ("label", [list(x) for x in labels] or None)) if v}
    rows = st.index.select_artifacts(kind=ns.kind, where=_kv(ns.where, "--where"), label=labels or None, attempt=ns.attempt, tool=ns.tool, derivation=ns.derivation)
    return rows, query


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
    rep = st.push(art, preserve=True) if ns.preserve else None
    if ns.json:
        _dump({"id": art, "preserve": rep})
    else:
        print(art)
        if rep is not None:
            print(f"research data put: {'PRESERVED' if rep['preserved'] else 'NOT preserved: ' + '; '.join(rep['errors'])} on {rep['remote']}", file=sys.stderr)
    return 0 if rep is None or rep["preserved"] else 1


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
    print(f"research data show: {ns.ref!r} is not an art:, drv: or run id", file=sys.stderr)
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
    ap = _ap("fetch")
    ap.add_argument("art")
    ap.add_argument("--to", default=None, metavar="DIR", help="materialise under this directory (default: the store's own cache)")
    ns = ap.parse_args(argv)
    to = None
    if ns.to:
        to = Path(ns.to)
        to.mkdir(parents=True, exist_ok=True)
    print(LocalStore(ns.store).fetch(ns.art, to))
    return 0


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


def cmd_select(argv: list[str]) -> int:
    ns = _select_parser().parse_args(argv)
    rows, _ = _select(LocalStore(ns.store), ns)
    if ns.json:
        _dump(rows)
    elif ns.ids:
        for r in rows:
            print(r["id"])
    else:
        for r in rows:
            print(f"{r['id']}  {r['kind']:<24} {r['bytes']:>12} {r['payload_type'] or 'none':<5} {'local' if r['local'] else '-':<5} "
                  f"{'remote' if r['remote'] else '-':<6} {r['repo_path'] or ''}")
        print(f"{len(rows)} artifact(s)", file=sys.stderr)
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


def cmd_labels(argv: list[str]) -> int:
    ap = _ap("labels")
    ap.add_argument("target")
    ap.add_argument("--history", action="store_true", help="every assertion in ts order (default: the latest per key and asserter)")
    ap.add_argument("--key")
    ap.add_argument("--by", default=None, metavar="PATTERN", help="fnmatch on the asserter, e.g. 'red-team-*'")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    labs = LocalStore(ns.store).labels(ns.target, key=ns.key, by_glob=ns.by)
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
    ns = ap.parse_args(argv)
    value = json.loads(ns.value) if ns.value_json else ns.value
    print(LocalStore(ns.store).add_label(ns.target, ns.key, value, ns.by, ref=ns.ref))
    return 0


def cmd_reindex(argv: list[str]) -> int:
    ap = _ap("reindex")
    ap.add_argument("--remote", action="store_true", help="also list the remote; fetch manifests/attempts it has and we lack")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    st = LocalStore(ns.store)
    n = st.reindex(remote=ns.remote)
    _dump(n) if ns.json else print(f"reindexed {st.root}: {n['artifacts']} artifacts, {n['attempts']} attempts, {n['labels']} labels")
    return 0


def cmd_sql(argv: list[str]) -> int:
    ap = _ap("sql")
    ap.add_argument("query")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    cols, rows = LocalStore(ns.store).index.sql_with_columns(ns.query)
    if ns.json:
        _dump([dict(zip(cols, r)) for r in rows])
    else:
        if cols:
            print("\t".join(cols))
        for r in rows:
            print("\t".join("" if v is None else str(v) for v in r))
    return 0


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
    MD5 ETag, the default; `--verify readback`: download + sha256), then its manifest; a run id pushes the attempt manifest and every
    artifact it references.  `--pending` pushes every local artifact without a verified remote replica and every attempt the remote
    lacks.  One artifact's failure is reported, the batch continues; exit 1 if anything is not PRESERVED."""
    ap = _ap("push")
    ap.add_argument("targets", nargs="*", help="art:<hex> or run ids")
    ap.add_argument("--pending", action="store_true", help="everything local that the remote does not verifiably hold")
    ap.add_argument("--jobs", type=int, default=32, metavar="N", help="concurrent uploads per artifact (default 32)")
    ap.add_argument("--verify", choices=["head", "readback"], default="head",
                    help="per-object check: head = Content-Length + single-part ETag == MD5 (default); readback = download + sha256")
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
    if ns.pending:
        verified = {row[0] for row in st.index.db.execute("SELECT id FROM replicas WHERE location='remote' AND verified_utc IS NOT NULL")}
        arts += [a for a in st.artifact_ids() if a not in verified]
        runs += [rid for rid in st.attempt_ids() if r.head(attempt_key(rid)) is None]
    reports: list[dict] = []
    for rid in dict.fromkeys(runs):
        if not ids.is_run_id(rid):
            raise ValueError(f"not an artifact or run id: {rid!r}")
        doc = st.get_attempt(rid)
        if doc is None:
            raise StoreError(f"attempt {rid} is not in the store")
        arts += _attempt_artifacts(doc)
        try:                                                       # one bad upload must not abandon the batch: record, continue
            st.push_attempt(rid)
            reports.append({"id": rid, "kind": "attempt", "preserved": r.head(attempt_key(rid)) is not None})
        except Exception as e:  # noqa: BLE001
            reports.append({"id": rid, "kind": "attempt", "preserved": False, "errors": [f"{type(e).__name__}: {e}"]})
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
    bad = [x for x in reports if not x["preserved"]]
    if ns.json:
        _dump({"remote": r.describe(), "pushed": reports, "not_preserved": len(bad)})
    else:
        for x in reports:
            if x["kind"] == "attempt":
                print(f"{x['id']}  attempt  {'PRESERVED' if x['preserved'] else 'NOT preserved: ' + '; '.join(x.get('errors') or ['absent'])}")
        print(f"{len(reports) - len(bad)}/{len(reports)} preserved on {r.describe()}")
    return 1 if bad else 0


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


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "put": cmd_put, "show": cmd_show, "where": cmd_where, "fetch": cmd_fetch, "verify": cmd_verify, "select": cmd_select, "snapshot": cmd_snapshot,
    "labels": cmd_labels, "label": cmd_label, "reindex": cmd_reindex, "sql": cmd_sql, "push": cmd_push, "mint-credential": cmd_mint_credential,
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
        return fn(rest)
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
