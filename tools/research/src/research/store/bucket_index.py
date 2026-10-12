"""The store's derived index (README §5.1): four generic tables built from the bucket's records by `index()`, their only writer.

records(id, type, kind, time, doc)   type artifact | run | label; kind = an artifact's kind, a run's tool, a label's key;
                                     time = when an artifact's manifest reached the bucket, a run's start, a label's ts;
                                     doc = the record less `payload.files` and `closure`
fields(id, path, value)              every scalar leaf of doc as `index.flatten` writes it (dotted path, text value); a value
                                     longer than FIELD_MAX characters is left to doc (Postgres cannot index it)
edges(src, name, dst)                refs.<name>, inputs.<name>, outputs.<name>, telemetry.<name>, and a label's `target`;
                                     dst need not be a record (a drv: id, pr:<n>@<sha>)
listings(prefix, listed_at, keys)    what the last listing of each prefix saw: the index is complete as of `listed_at`

The index never names a field inside a record, so a new kind or field needs no migration: a schema change is a rebuild. It
never lists `objects/` or opens a payload. A label's id is `label:` + the sha256 of its canonical assertion (its remote key);
every record read is checked against its key, and one that fails is reported and left out.

`index(remote, db)` lists the bucket, reads the records the index lacks and writes their rows (idempotent). Runs are listed from
the newest indexed run id (`start-after`), and in full once `FULL_RUNS_EVERY` has passed, for a run whose custody published it
late. `index(public_remote, db, policy=PUBLIC, private=private_db)` rebuilds the public tables from the private index: the
artifacts whose manifests are in the public bucket, each run that produced one (its doc cut to `PUBLIC_RUN_FIELDS`) and the
`PUBLIC_LABEL_KEYS` labels on those targets (cut to `PUBLIC_LABEL_FIELDS`). Both are whitelists.

`db` is a SQLite file (`SqliteIndex`) or a schema of a Postgres database (`PostgresIndex`, written through `psql` with the URL
in libpq's environment). The public rebuild reads the private index one record at a time, cheap in SQLite only.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import canon, ids
from .index import flatten
from .model import Label

PRIVATE, PUBLIC = "private", "public"
MANIFESTS, ATTEMPTS, LABELS = "manifests/", "attempts/", "labels/"
RUNS_TAIL = "attempts/>"                                # the listings row of the start-after listing of runs
FULL_RUNS_EVERY = timedelta(hours=1)
FIELD_MAX = 1024
ENVELOPE_ONLY = frozenset({"redteam-findings/v1"})     # meta (the findings' text) stays in the bucket
EDGE_SECTIONS = {"artifact": ("refs",), "run": ("inputs", "outputs", "telemetry")}
PUBLIC_LABEL_KEYS = ("verified", "verifier", "proof_class")
PUBLIC_LABEL_FIELDS = ("schema", "target", "key", "value", "by", "ts")
PUBLIC_RUN_FIELDS = (("schema",), ("id",), ("tool", "name"), ("status", "state"), ("execution", "start_utc"), ("execution", "wall_s"),
                     ("conditions", "hw.sku"), ("conditions", "hw.count"), ("conditions", "rt.driver"), ("conditions", "rt.cuda"),
                     ("source", "commit"))
HEX_RE = re.compile(ids.HEX64)

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, type TEXT NOT NULL, kind TEXT, time TEXT, doc TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS records_type_kind_time ON records(type, kind, time);
CREATE TABLE IF NOT EXISTS fields(id TEXT NOT NULL, path TEXT NOT NULL, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS fields_id ON fields(id);
CREATE INDEX IF NOT EXISTS fields_path_value ON fields(path, value);
CREATE TABLE IF NOT EXISTS edges(src TEXT NOT NULL, name TEXT NOT NULL, dst TEXT NOT NULL, PRIMARY KEY(src, name, dst));
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst, name);
CREATE TABLE IF NOT EXISTS listings(prefix TEXT PRIMARY KEY, listed_at TEXT NOT NULL, keys INTEGER NOT NULL);
"""

POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.records(id text PRIMARY KEY, type text NOT NULL, kind text, time timestamptz, doc jsonb NOT NULL);
CREATE INDEX IF NOT EXISTS records_type_kind_time ON {schema}.records(type, kind, time DESC);
CREATE TABLE IF NOT EXISTS {schema}.fields(id text NOT NULL, path text NOT NULL, value text NOT NULL);
CREATE INDEX IF NOT EXISTS fields_id ON {schema}.fields(id);
CREATE INDEX IF NOT EXISTS fields_path_value ON {schema}.fields(path, value);
CREATE TABLE IF NOT EXISTS {schema}.edges(src text NOT NULL, name text NOT NULL, dst text NOT NULL, PRIMARY KEY(src, name, dst));
CREATE INDEX IF NOT EXISTS edges_dst ON {schema}.edges(dst, name);
CREATE TABLE IF NOT EXISTS {schema}.listings(prefix text PRIMARY KEY, listed_at timestamptz NOT NULL, keys bigint NOT NULL);
"""


def iso(text: Any) -> str | None:
    """`YYYY-MM-DDTHH:MM:SSZ` in UTC, or None for anything that is not a time."""
    if not isinstance(text, str) or not text:
        return None
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Row:
    id: str
    type: str
    kind: str | None
    time: str | None
    doc: dict

    def record(self) -> tuple:
        return (self.id, self.type, self.kind, self.time, canon.dumps(self.doc).decode())

    def fields(self) -> list[tuple[str, str, str]]:
        return [(self.id, p, v) for p, v in flatten(self.doc) if len(v) <= FIELD_MAX]

    def edges(self) -> list[tuple[str, str, str]]:
        if self.type == "label":
            return [(self.id, "target", self.doc["target"])]
        out = set()
        for sec in EDGE_SECTIONS[self.type]:
            for name, v in (self.doc.get(sec) or {}).items():
                for dst in (v if isinstance(v, list) else [v]):
                    if isinstance(dst, str) and dst:
                        out.add((self.id, f"{sec}.{name}", dst))
        return sorted(out)


def record_id(key: str) -> str | None:
    """The id a key names, or None when only the content can say (a label under a non-content key, as a local store keeps them)."""
    name = key.rsplit("/", 1)[-1].removesuffix(".json")
    if key.startswith(MANIFESTS):
        return f"art:{name}"
    if key.startswith(ATTEMPTS):
        return name
    return f"label:{name}" if HEX_RE.fullmatch(name) else None


def parse(key: str, data: bytes, modified: str | None) -> Row:
    """The row of one record read from `key`, checked against the key; ValueError when it does not match."""
    doc = json.loads(data)
    if not isinstance(doc, dict):
        raise ValueError(f"{key}: not a JSON object")
    if key.startswith(MANIFESTS):
        rid = "art:" + canon.sha256(doc)
        if rid != record_id(key):
            raise ValueError(f"{key}: manifest hashes to {rid}")
        doc = dict(doc)
        if isinstance(doc.get("payload"), dict):
            doc["payload"] = {k: v for k, v in doc["payload"].items() if k != "files"}
        if doc.get("kind") in ENVELOPE_ONLY:
            doc.pop("meta", None)
        return Row(rid, "artifact", doc.get("kind"), iso(modified), doc)
    if key.startswith(ATTEMPTS):
        if doc.get("id") != record_id(key):
            raise ValueError(f"{key}: holds run {doc.get('id')!r}")
        doc = {k: v for k, v in doc.items() if k != "closure"}
        return Row(doc["id"], "run", (doc.get("tool") or {}).get("name"), iso((doc.get("execution") or {}).get("start_utc")), doc)
    if key.startswith(LABELS):
        lab = Label.from_doc(doc)
        lab.validate()
        rid = "label:" + canon.sha256(lab.to_doc())
        if record_id(key) not in (None, rid):
            raise ValueError(f"{key}: assertion hashes to {rid}")
        return Row(rid, "label", lab.key, iso(lab.ts), lab.to_doc())
    raise ValueError(f"{key}: not a record key")


def public_run(row: Row) -> Row:
    doc: dict = {}
    for path in PUBLIC_RUN_FIELDS:
        src, dst = row.doc, doc
        for k in path[:-1]:
            src = src.get(k) if isinstance(src, dict) else None
            dst = dst.setdefault(k, {})
        if isinstance(src, dict) and path[-1] in src:
            dst[path[-1]] = src[path[-1]]
    return Row(row.id, row.type, row.kind, row.time, doc)


def public_label(row: Row) -> Row:
    return Row(row.id, row.type, row.kind, row.time, {k: row.doc.get(k) for k in PUBLIC_LABEL_FIELDS})


class SqliteIndex:
    """The four tables in one SQLite file (a schema of its own: the private and the public index are two files)."""

    def __init__(self, db: Path | str | sqlite3.Connection):
        if isinstance(db, sqlite3.Connection):
            self.db, self.path = db, None
        else:
            self.path = Path(db)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        self.db.executescript(SQLITE_DDL)

    def close(self) -> None:
        if self.path is not None:
            self.db.close()

    def ids(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT id FROM records")}

    def newest_run(self) -> str | None:
        r = self.db.execute("SELECT max(id) FROM records WHERE type='run'").fetchone()
        return r[0] if r else None

    def listed_at(self, prefix: str) -> str | None:
        r = self.db.execute("SELECT listed_at FROM listings WHERE prefix=?", (prefix,)).fetchone()
        return r[0] if r else None

    def row(self, rid: str) -> Row | None:
        r = self.db.execute("SELECT id, type, kind, time, doc FROM records WHERE id=?", (rid,)).fetchone()
        return Row(r[0], r[1], r[2], r[3], json.loads(r[4])) if r else None

    def producers(self, art: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT src FROM edges WHERE dst=? AND name LIKE 'outputs.%' ORDER BY src", (art,))]

    def labels_on(self, target: str, keys: Iterable[str]) -> list[str]:
        keys = list(keys)
        return [r[0] for r in self.db.execute(
            "SELECT e.src FROM edges e JOIN records r ON r.id = e.src WHERE e.dst=? AND e.name='target' "
            f"AND r.kind IN ({','.join('?' * len(keys))}) ORDER BY e.src", (target, *keys))]

    def edges_of(self, rid: str) -> list[tuple[str, str, str]]:
        return list(self.db.execute("SELECT src, name, dst FROM edges WHERE src=? ORDER BY name, dst", (rid,)))

    def add(self, rows: list[Row], edges: Callable[[Row], list[tuple]] = Row.edges) -> int:
        """Insert the rows not already present, in one transaction.  How many were new."""
        n = 0
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for row in rows:
                if self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)", row.record()).rowcount:
                    n += 1
                    self.db.executemany("INSERT INTO fields VALUES (?,?,?)", row.fields())
                    self.db.executemany("INSERT OR IGNORE INTO edges VALUES (?,?,?)", edges(row))
        return n

    def replace(self, groups: list[tuple[list[Row], Callable[[Row], list[tuple]]]], listings: dict[str, tuple[str, int]]) -> None:
        """Every row replaced by `groups` and `listings`, in one transaction (the public rebuild)."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for t in ("records", "fields", "edges", "listings"):
                self.db.execute(f"DELETE FROM {t}")
            for rows, edges in groups:
                for row in rows:
                    if self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)", row.record()).rowcount:
                        self.db.executemany("INSERT INTO fields VALUES (?,?,?)", row.fields())
                        self.db.executemany("INSERT OR IGNORE INTO edges VALUES (?,?,?)", edges(row))
            self.db.executemany("INSERT INTO listings VALUES (?,?,?)", [(p, t, n) for p, (t, n) in listings.items()])

    def set_listing(self, prefix: str, listed_at: str, keys: int) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO listings VALUES (?,?,?)", (prefix, listed_at, keys))

    def size(self) -> dict:
        """Rows per table and the bytes the rows hold (doc text, field and edge text), for the measurement step 1 asks for."""
        one = lambda q: self.db.execute(q).fetchone()  # noqa: E731
        rec = one("SELECT count(*), coalesce(sum(length(CAST(doc AS BLOB))), 0) FROM records")
        fld = one("SELECT count(*), coalesce(sum(length(CAST(id AS BLOB)) + length(CAST(path AS BLOB)) + length(CAST(value AS BLOB))), 0) FROM fields")
        edg = one("SELECT count(*), coalesce(sum(length(CAST(src AS BLOB)) + length(CAST(name AS BLOB)) + length(CAST(dst AS BLOB))), 0) FROM edges")
        by_type = {t: {"rows": n, "doc_bytes": b} for t, n, b in
                   self.db.execute("SELECT type, count(*), sum(length(CAST(doc AS BLOB))) FROM records GROUP BY type ORDER BY type")}
        return {"records": {"rows": rec[0], "doc_bytes": rec[1], "by_type": by_type}, "fields": {"rows": fld[0], "bytes": fld[1]},
                "edges": {"rows": edg[0], "bytes": edg[1]},
                "file_bytes": self.path.stat().st_size if self.path is not None and self.path.exists() else None}


def pg_env(url: str) -> dict[str, str]:
    """libpq's environment for a `postgresql://user:password@host/db?sslmode=...` URL, so the URL never sits on a command line."""
    u = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(u.query)
    if u.scheme not in ("postgres", "postgresql") or not (u.hostname or query.get("host")):
        raise ValueError(f"not a postgres URL with a host (scheme {u.scheme!r})")
    env = {"PGHOST": u.hostname or "", "PGPORT": str(u.port or 5432), "PGDATABASE": urllib.parse.unquote(u.path.lstrip("/")),
           "PGUSER": urllib.parse.unquote(u.username or ""), "PGPASSWORD": urllib.parse.unquote(u.password or "")}
    for key, var in (("host", "PGHOST"), ("sslmode", "PGSSLMODE"), ("channel_binding", "PGCHANNELBINDING"), ("options", "PGOPTIONS"),
                     ("connect_timeout", "PGCONNECT_TIMEOUT")):
        if v := query.get(key):
            env[var] = v[-1]
    return {k: v for k, v in env.items() if v}


def _lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _values(rows: Iterable[tuple]) -> str:
    """Rows as SQL VALUES literals.  Postgres text and jsonb cannot hold NUL, so it becomes U+FFFD here only."""
    return ",\n".join("(" + ",".join("NULL" if v is None else _lit(str(v).replace("\x00", "\ufffd")) for v in r) + ")" for r in rows)


_UNCOPY = {"\\": "\\", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}


def _uncopy(field: str) -> str | None:
    return None if field == r"\N" else re.sub(r"\\(.)", lambda m: _UNCOPY.get(m.group(1), m.group(1)), field)


_UTC = "to_char({col} AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"')"


class PostgresIndex:
    """The four tables in one schema of a Postgres database (Neon's `store`: `store_private`, `store_public`), written through
    `psql` (stdlib only: no driver).  The schema is the database owner's; this creates the tables in it when they are missing and
    otherwise only inserts and deletes rows, so the readers' grants on the schema survive a rebuild."""

    def __init__(self, url: str, schema: str, psql: str = "psql", *, create: bool = True):
        """`create=False` for a reader: Postgres checks CREATE on the schema even when every table exists."""
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", schema):
            raise ValueError(f"bad schema name {schema!r}")
        self.env, self.schema, self.psql, self.path = pg_env(url), schema, psql, None
        if create:
            self.run(POSTGRES_DDL.format(schema=schema))

    def run(self, script: str) -> list[list[str]]:
        r = subprocess.run([self.psql, "-X", "-q", "-A", "-t", "-F", "\t", "-v", "ON_ERROR_STOP=1", "-f", "-"], input=script,
                           env={**os.environ, **self.env}, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"psql ({self.schema}): {r.stderr.strip()[-2000:]}")
        return [ln.split("\t") for ln in r.stdout.splitlines() if ln]

    def select(self, query: str) -> list[tuple]:
        """The rows of one SELECT, read through COPY's text format, so values may hold tabs and newlines; NULL is None."""
        return [tuple(_uncopy(f) for f in row) for row in self.run(f"COPY ({query}) TO STDOUT;")]

    def close(self) -> None:
        pass

    def ids(self) -> set[str]:
        return {r[0] for r in self.select(f"SELECT id FROM {self.schema}.records")}

    def newest_run(self) -> str | None:
        r = self.select(f"SELECT max(id) FROM {self.schema}.records WHERE type='run'")
        return r[0][0] if r else None

    def listed_at(self, prefix: str) -> str | None:
        r = self.select(f"SELECT {_UTC.format(col='listed_at')} FROM {self.schema}.listings WHERE prefix={_lit(prefix)}")
        return r[0][0] if r else None

    def row(self, rid: str) -> Row | None:
        r = self.select(f"SELECT id, type, kind, {_UTC.format(col='time')}, doc::text FROM {self.schema}.records WHERE id={_lit(rid)}")
        return Row(*r[0][:4], json.loads(r[0][4])) if r else None

    def producers(self, art: str) -> list[str]:
        return [r[0] for r in self.select(f"SELECT DISTINCT src FROM {self.schema}.edges WHERE dst={_lit(art)} AND name LIKE 'outputs.%' ORDER BY src")]

    def labels_on(self, target: str, keys: Iterable[str]) -> list[str]:
        return [r[0] for r in self.select(
            f"SELECT e.src FROM {self.schema}.edges e JOIN {self.schema}.records r ON r.id = e.src WHERE e.dst={_lit(target)} "
            f"AND e.name='target' AND r.kind IN ({','.join(_lit(k) for k in keys)}) ORDER BY e.src")]

    def edges_of(self, rid: str) -> list[tuple[str, str, str]]:
        return self.select(f"SELECT src, name, dst FROM {self.schema}.edges WHERE src={_lit(rid)} ORDER BY name, dst")

    def _insert(self, rows: list[Row], edges: Callable[[Row], list[tuple]]) -> str:
        """One statement inserting the records not yet present and the fields and edges of those only; it returns how many were new.
        Plain VALUES rather than COPY: COPY cannot skip a present key, and the indexer may not create temporary tables."""
        s = self.schema
        fields = [x for row in rows for x in row.fields()]
        links = [x for row in rows for x in edges(row)]
        sql = (f"WITH new AS (INSERT INTO {s}.records SELECT id, type, kind, time::timestamptz, doc::jsonb FROM (VALUES\n"
               f"{_values(row.record() for row in rows)}) v(id, type, kind, time, doc) ON CONFLICT (id) DO NOTHING RETURNING id)")
        if fields:
            sql += (f",\nf AS (INSERT INTO {s}.fields SELECT v.* FROM (VALUES\n{_values(fields)}) v(id, path, value) JOIN new USING (id) "
                    "RETURNING 1)")
        if links:
            sql += (f",\ne AS (INSERT INTO {s}.edges SELECT v.* FROM (VALUES\n{_values(links)}) v(src, name, dst) "
                    "JOIN new ON new.id = v.src ON CONFLICT DO NOTHING RETURNING 1)")
        return sql + "\nSELECT count(*) FROM new;\n"

    def add(self, rows: list[Row], edges: Callable[[Row], list[tuple]] = Row.edges) -> int:
        """Insert the rows not already present, in one transaction.  How many were new."""
        if not rows:
            return 0
        return int(self.run(self._insert(rows, edges))[-1][0])

    def replace(self, groups: list[tuple[list[Row], Callable[[Row], list[tuple]]]], listings: dict[str, tuple[str, int]]) -> None:
        """Every row replaced by `groups` and `listings`, in one transaction (the public rebuild); tables are kept, so grants are."""
        s = self.schema
        self.run("BEGIN;\n" + "".join(f"DELETE FROM {s}.{t};\n" for t in ("records", "fields", "edges", "listings"))
                 + "".join(self._insert(rows, edges) for rows, edges in groups if rows)
                 + "".join(f"INSERT INTO {s}.listings VALUES ({_lit(p)}, {_lit(t)}, {int(n)});\n" for p, (t, n) in listings.items())
                 + "COMMIT;\n")

    def set_listing(self, prefix: str, listed_at: str, keys: int) -> None:
        self.run(f"INSERT INTO {self.schema}.listings VALUES ({_lit(prefix)}, {_lit(listed_at)}, {int(keys)}) "
                 "ON CONFLICT (prefix) DO UPDATE SET listed_at = EXCLUDED.listed_at, keys = EXCLUDED.keys;")

    def size(self) -> dict:
        """SqliteIndex.size's shape; `bytes` is each table's with its indexes and `file_bytes` their sum."""
        s = self.schema
        out = {t: {"rows": int(n), "bytes": int(b)} for t, n, b in self.run("".join(
            f"SELECT '{t}', (SELECT count(*) FROM {s}.{t}), pg_total_relation_size('{s}.{t}');\n"
            for t in ("records", "fields", "edges", "listings")))}
        out["records"]["doc_bytes"] = int(self.run(f"SELECT coalesce(sum(pg_column_size(doc)), 0) FROM {s}.records;")[0][0])
        return {**out, "schema": s, "file_bytes": sum(t["bytes"] for t in out.values())}


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(remote, keys: list[tuple[str, str | None]], jobs: int) -> tuple[list[Row], list[str]]:
    """Read and check each key, `jobs` at a time.  (rows in key order, errors)."""
    lock, rows, errors = threading.Lock(), {}, []
    with tempfile.TemporaryDirectory(prefix="bucket-index-") as tmp:
        def one(i_kv):
            i, (key, modified) = i_kv
            path = Path(tmp) / f"{i}.json"
            try:
                remote.get(key, path)
                row = parse(key, path.read_bytes(), modified)
            except Exception as e:  # noqa: BLE001 -- one bad record is reported and left out; the next pass reads it again
                with lock:
                    errors.append(f"{key}: {type(e).__name__}: {e}")
                return
            finally:
                path.unlink(missing_ok=True)
            with lock:
                rows[i] = row
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            list(ex.map(one, enumerate(keys)))
    return [rows[i] for i in sorted(rows)], sorted(errors)


def _private(remote, idx, *, jobs: int, full: bool, now: datetime, batch: int, log) -> dict:
    stamp = _stamp(now)
    have = idx.ids()
    newest = idx.newest_run()
    last_full = idx.listed_at(ATTEMPTS)
    runs_full = full or newest is None or last_full is None or now - datetime.fromisoformat(last_full.replace("Z", "+00:00")) >= FULL_RUNS_EVERY
    listings = {MANIFESTS: remote.list_times(MANIFESTS), LABELS: remote.list_times(LABELS)}
    if runs_full:
        listings[ATTEMPTS] = remote.list_times(ATTEMPTS)
    else:
        listings[RUNS_TAIL] = remote.list_times(ATTEMPTS, start_after=f"{ATTEMPTS}{newest}.json")
    want = [(k, m) for keys in listings.values() for k, m in keys if k.endswith(".json") and record_id(k) not in have]
    rep = {"listed": {p: len(v) for p, v in listings.items()}, "runs_listing": "full" if runs_full else "tail", "read": 0,
           "added": {"artifact": 0, "run": 0, "label": 0}, "errors": []}
    for i in range(0, len(want), batch):
        rows, errors = _read(remote, want[i:i + batch], jobs)
        rep["read"] += len(rows)
        rep["errors"] += errors
        new = [r for r in rows if r.id not in have]
        idx.add(new)
        for r in new:
            have.add(r.id)
            rep["added"][r.type] += 1
        if log:
            log(f"index: {min(i + batch, len(want))}/{len(want)} keys read, {sum(rep['added'].values())} records added")
    for prefix, keys in listings.items():
        idx.set_listing(prefix, stamp, len(keys))
    return rep


def _public(public_remote, idx, private, *, jobs: int, now: datetime) -> dict:
    listed = public_remote.list_times(MANIFESTS)
    arts, missing = {}, []
    for key, modified in listed:
        rid = record_id(key)
        row = private.row(rid)
        if row is None:
            missing.append((key, modified))
        else:
            arts[rid] = row
    fetched, errors = _read(public_remote, missing, jobs)
    arts.update({r.id: r for r in fetched})
    runs = {}
    for art in sorted(arts):
        for rid in private.producers(art):
            if rid not in runs and (row := private.row(rid)) is not None:
                runs[rid] = public_run(row)
    labels = {}
    for target in sorted(arts) + sorted(runs):
        for lid in private.labels_on(target, PUBLIC_LABEL_KEYS):
            if (row := private.row(lid)) is not None:
                labels[lid] = public_label(row)
    shown = set(arts) | set(runs)
    idx.replace([(list(arts.values()), Row.edges),
                 (list(runs.values()), lambda r: [e for e in private.edges_of(r.id) if e[2] in shown]),
                 (list(labels.values()), Row.edges)], {MANIFESTS: (_stamp(now), len(listed))})
    return {"listed": {MANIFESTS: len(listed)}, "read": len(fetched), "added": {"artifact": len(arts), "run": len(runs), "label": len(labels)},
            "errors": errors}


def index(remote, db: Path | str | sqlite3.Connection | SqliteIndex | PostgresIndex, *, policy: str = PRIVATE,
          private: Path | str | SqliteIndex | PostgresIndex | None = None,
          jobs: int = 32, full: bool = False, now: datetime | None = None, batch: int = 2000,
          log: Callable[[str], None] | None = None) -> dict:
    """Bring the index at `db` up to what `remote` holds (module docstring).  `remote` needs `list_times` and `get`.  PRIVATE reads only
    the records `db` lacks; PUBLIC rebuilds `db` from the private index `private` and the public bucket `remote`'s manifest listing.
    {"policy", "listed": {prefix: keys}, "read", "added": {type: n}, "errors", "seconds"}; a record that fails its check is in
    `errors` and is read again next pass."""
    t0 = time.monotonic()
    idx = db if isinstance(db, (SqliteIndex, PostgresIndex)) else SqliteIndex(db)
    try:
        if policy == PRIVATE:
            rep = _private(remote, idx, jobs=jobs, full=full, now=_now(now), batch=max(1, batch), log=log)
        elif policy == PUBLIC:
            if private is None:
                raise ValueError("index(policy=PUBLIC) derives from the private index: pass private=")
            priv = private if isinstance(private, (SqliteIndex, PostgresIndex)) else SqliteIndex(private)
            try:
                rep = _public(remote, idx, priv, jobs=jobs, now=_now(now))
            finally:
                if priv is not private:
                    priv.close()
        else:
            raise ValueError(f"policy must be {PRIVATE!r} or {PUBLIC!r}, got {policy!r}")
    finally:
        if idx is not db:
            idx.close()
    return {"policy": policy, **rep, "seconds": round(time.monotonic() - t0, 3)}


def default_path(store_root: Path | str, policy: str = PRIVATE) -> Path:
    return Path(store_root) / ("index.sqlite" if policy == PRIVATE else "index-public.sqlite")


def dump(db: Path | str | SqliteIndex | PostgresIndex) -> dict[str, list[tuple]]:
    """Every row of records, fields and edges, sorted: what two builds of one bucket must agree on (listings carry their time)."""
    if isinstance(db, PostgresIndex):
        s = db.schema
        recs = [(*r[:4], canon.dumps(json.loads(r[4])).decode())
                for r in db.select(f"SELECT id, type, kind, {_UTC.format(col='time')}, doc::text FROM {s}.records")]
        return {"records": sorted(recs), "fields": sorted(db.select(f"SELECT * FROM {s}.fields")),
                "edges": sorted(db.select(f"SELECT * FROM {s}.edges"))}
    idx = db if isinstance(db, SqliteIndex) else SqliteIndex(db)
    try:
        return {t: sorted(idx.db.execute(f"SELECT * FROM {t}")) for t in ("records", "fields", "edges")}
    finally:
        if idx is not db:
            idx.close()

