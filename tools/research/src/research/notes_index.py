"""`research notes index`: a notes clone as one SQLite file, searched in full text and joined to the evidence store's catalog;
`research notes mcp` (notes_mcp.py) serves it to agents.

    research notes index [--root DIR] [--index PATH] [--catalog PATH] [--search TEXT] [--evidence ID]

    notes(nid, path, lane, id, kind, status, created, time, sender, recipient, campaign, title, bytes, mtime_ns)
    notes_fts(title, body)       FTS5 with rowid = notes.nid: `notes_fts MATCH 'words'`, ranked by bm25
    cites(path, target, kind)    each `art:<hex>` (as written) and run id a note names; kind art | run

`id` is the front matter's `id:`, else the file name without `.md` (what `note:<id>` names); `time` is the file name's stamp,
else `created:`, else the mtime, as `YYYY-MM-DDTHH:MMZ`; `sender` / `recipient` are `from:` / `to:`; a note under
`lanes/<lane>/` (its month folders too) has that lane.  Every `*.md` under the root is indexed, dot-directories skipped.  A
refresh re-reads only the notes whose size or mtime changed and drops the ones gone, so the server runs one before every call.
The index is disposable (`~/.research/notes-index/<hash of the root>.sqlite` by default): another `VERSION` is rebuilt.

The catalog is attached read-only as `ev`: `--catalog PATH`, else the local store's `catalog.sqlite` when there is one.  A copy
made by `research data catalog --compact` has the same tables with `manifest_json` empty, so a machine without the store's
catalog can join notes to evidence from that one file.  A run counts as in the catalog once its attempt or a label on it is
there (a waived custody label accounts for it), an `art:` prefix once it names an artifact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import sqlite3
import sys
from pathlib import Path

from .notes import ART_RE, _split_front_matter
from .notes_sync import CREATED_RE, STAMP_RE
from .store.ids import RUN_RE
from .store.local import store_root

VERSION = 1
SCHEMA = """
CREATE TABLE notes(nid INTEGER PRIMARY KEY, path TEXT UNIQUE, lane TEXT, id TEXT, kind TEXT, status TEXT, created TEXT, time TEXT,
                   sender TEXT, recipient TEXT, campaign TEXT, title TEXT, bytes INTEGER, mtime_ns INTEGER);
CREATE INDEX notes_id ON notes(id);
CREATE INDEX notes_time ON notes(time);
CREATE INDEX notes_lane ON notes(lane, time);
CREATE VIRTUAL TABLE notes_fts USING fts5(title, body, tokenize='porter unicode61');
CREATE TABLE cites(path TEXT, target TEXT, kind TEXT);
CREATE INDEX cites_target ON cites(target);
CREATE INDEX cites_path ON cites(path);
"""
COLUMNS = ("path", "lane", "id", "kind", "status", "created", "time", "sender", "recipient", "campaign", "title", "bytes", "mtime_ns")
FIELDS = {"id": "id", "kind": "kind", "status": "status", "from": "sender", "to": "recipient", "campaign": "campaign", "lane": "lane"}
FIELD_RE = re.compile(r"^([A-Za-z_][\w-]*):[ \t]*(.*?)[ \t]*$")
RUN_CITE_RE = re.compile(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b")
MAX_ROWS = 200
SQL_BUDGET = 100_000        # progress-handler ticks of 1000 SQLite instructions each: a runaway query stops, deterministically
READ_MAX = 60_000


def default_index(root: Path) -> Path:
    return Path.home() / ".research" / "notes-index" / f"{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]}.sqlite"


def default_catalog() -> Path | None:
    p = store_root() / "catalog.sqlite"
    return p if p.is_file() else None


def _uri(p: Path, ro: bool = False) -> str:
    return p.resolve().as_uri() + ("?mode=ro" if ro else "")


def _utc(m: re.Match | None) -> str | None:
    """`created:` (CREATED_RE) as `YYYY-MM-DDTHH:MMZ`; an offset is applied, none means UTC."""
    if m is None:
        return None
    try:
        t = dt.datetime.fromisoformat(f"{m[2]}T{m[3][:5]}")
    except ValueError:
        return None
    tz = (m[4] or "Z").upper()
    if tz != "Z":
        t -= (1 if tz[0] == "+" else -1) * dt.timedelta(hours=int(tz[1:3]), minutes=int(tz[-2:]))
    return f"{t:%Y-%m-%dT%H:%M}Z"


def _files(root: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            if f.endswith(".md") and not f.startswith("."):
                try:
                    st = os.stat(os.path.join(d, f))
                except OSError:
                    continue
                out[Path(d, f).relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


def read_note(root: Path, rel: str, size: int, mtime_ns: int) -> tuple[dict, str, set[tuple[str, str]]]:
    """(the `notes` row, the body, {(target, kind)}) of one file (module docstring)."""
    text = (root / rel).read_text(errors="replace")
    front, body = _split_front_matter(text)
    fm: dict[str, str | None] = {}
    for line in front.splitlines()[1:-1]:
        if (m := FIELD_RE.match(line)) and (col := FIELDS.get(m[1].lower())) and col not in fm:
            fm[col] = m[2].strip("\"'") or None
    parts = rel.split("/")
    if len(parts) > 2 and parts[0] == "lanes":
        fm["lane"] = parts[1]
    created = _utc(CREATED_RE.search(front))
    s = m[1] if (m := STAMP_RE.match(parts[-1])) else None
    t = (f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}Z" if s else
         created or f"{dt.datetime.fromtimestamp(mtime_ns / 1e9, dt.timezone.utc):%Y-%m-%dT%H:%M}Z")
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    title = next((ln for ln in lines if ln.startswith("#")), lines[0] if lines else "").lstrip("#").strip()[:200]
    row = {**fm, "path": rel, "id": fm.get("id") or parts[-1][:-3], "created": created, "time": t, "title": title,
           "bytes": size, "mtime_ns": mtime_ns}
    cites = {(f"art:{h}", "art") for h in ART_RE.findall(text)} | {(r, "run") for r in RUN_CITE_RE.findall(text)}
    return row, body, cites


def _clip(v: object, n: int = 200) -> str:
    s = " ".join(str(v).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _quoted(query: str) -> str:
    """Each word as an FTS5 string, for text that is not FTS5 syntax (`sm_90`, `art:cc59`, `r20260928-042319-9aac`)."""
    return " ".join('"' + w.replace('"', '""') + '"' for w in query.split())


def _deny_attach(action: int, *_args) -> int:
    return sqlite3.SQLITE_DENY if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH) else sqlite3.SQLITE_OK


class NotesIndex:
    """Writes go through a short-lived connection without the catalog: a write transaction would hold the attached catalog's
    lock and stall the store's own writers.  Queries go through `db`, which has the catalog attached as `ev`."""

    def __init__(self, root: Path, path: Path | None = None, catalog: Path | None = None):
        self.root = Path(root)
        self.path = Path(path) if path else default_index(self.root)
        self.catalog = Path(catalog) if catalog else default_catalog()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write(self._create)
        self.db = sqlite3.connect(_uri(self.path), uri=True, timeout=30, isolation_level=None)
        if self.catalog:
            self.db.execute("ATTACH DATABASE ? AS ev", (_uri(self.catalog, ro=True),))

    def _write(self, fn):
        w = sqlite3.connect(_uri(self.path), uri=True, timeout=30, isolation_level=None)
        try:
            w.execute("PRAGMA journal_mode=WAL")
            w.execute("BEGIN IMMEDIATE")
            try:
                out = fn(w)
                w.execute("COMMIT")
            except BaseException:
                w.execute("ROLLBACK")
                raise
            return out
        finally:
            w.close()

    @staticmethod
    def _create(w: sqlite3.Connection) -> None:
        if w.execute("PRAGMA user_version").fetchone()[0] == VERSION:
            return
        for (t,) in w.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('notes', 'notes_fts', 'cites')").fetchall():
            w.execute(f"DROP TABLE {t}")
        for stmt in SCHEMA.split(";"):
            if stmt.strip():
                w.execute(stmt)
        w.execute(f"PRAGMA user_version = {VERSION}")

    def close(self) -> None:
        self.db.close()

    def refresh(self) -> dict:
        """Catch up with the files (module docstring), in one write transaction: {notes, read, dropped}."""
        return self._write(self._refresh)

    def _refresh(self, w: sqlite3.Connection) -> dict:
        have = {p: (b, m, nid) for nid, p, b, m in w.execute("SELECT nid, path, bytes, mtime_ns FROM notes")}
        now = _files(self.root)
        gone = [p for p in have if p not in now]
        todo = [p for p, st in now.items() if have.get(p, (None, None, None))[:2] != st]
        for p in gone + [p for p in todo if p in have]:
            w.execute("DELETE FROM notes WHERE nid=?", (have[p][2],))
            w.execute("DELETE FROM notes_fts WHERE rowid=?", (have[p][2],))
            w.execute("DELETE FROM cites WHERE path=?", (p,))
        for p in todo:
            try:
                row, body, cites = read_note(self.root, p, *now[p])
            except OSError:
                continue
            nid = w.execute(f"INSERT INTO notes({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                            [row.get(c) for c in COLUMNS]).lastrowid
            w.execute("INSERT INTO notes_fts(rowid, title, body) VALUES (?, ?, ?)", (nid, row["title"], body))
            w.executemany("INSERT INTO cites VALUES (?, ?, ?)", [(p, t, k) for t, k in sorted(cites)])
        return {"notes": len(now), "read": len(todo), "dropped": len(gone)}

    # ---- queries (what the MCP tools print) -----------------------------------------------------------------------------------

    def in_catalog(self, target: str, kind: str) -> bool | None:
        """None without a catalog (module docstring for what counts)."""
        if not self.catalog:
            return None
        if kind == "run":
            q = "SELECT EXISTS(SELECT 1 FROM ev.attempts WHERE id=?) OR EXISTS(SELECT 1 FROM ev.labels WHERE target=?)"
            return bool(self.db.execute(q, (target, target)).fetchone()[0])
        return self.db.execute("SELECT 1 FROM ev.artifacts WHERE id >= ? AND id < ? LIMIT 1", (target, target + "g")).fetchone() is not None

    def _cited(self, path: str) -> str:
        cites = self.db.execute("SELECT target, kind FROM cites WHERE path=?", (path,)).fetchall()
        if not cites:
            return ""
        if not self.catalog:
            return f"cites {len(cites)}"
        return f"cites {len(cites)}, {sum(bool(self.in_catalog(t, k)) for t, k in cites)} in the catalog"

    def search(self, query: str = "", lane: str = "", kind: str = "", since: str = "", limit: int = 20) -> str:
        limit = max(1, min(int(limit), MAX_ROWS))
        where = [(c, v) for c, v in (("n.lane = ?", lane), ("n.kind = ?", kind), ("n.time >= ?", since)) if v]
        cond, args = "".join(f" AND {c}" for c, _ in where), [v for _, v in where]
        if query.strip():
            sql = ("SELECT n.time, n.path, n.kind, n.title, snippet(notes_fts, 1, '[', ']', ' ... ', 16) FROM notes_fts "
                   f"JOIN notes n ON n.nid = notes_fts.rowid WHERE notes_fts MATCH ?{cond} ORDER BY bm25(notes_fts, 4.0, 1.0) LIMIT ?")
            try:
                rows = self.db.execute(sql, [query, *args, limit]).fetchall()
            except sqlite3.OperationalError:
                rows = self.db.execute(sql, [_quoted(query), *args, limit]).fetchall()
        else:
            rows = self.db.execute(f"SELECT n.time, n.path, n.kind, n.title, '' FROM notes n WHERE 1{cond} "
                                   "ORDER BY n.time DESC, n.path LIMIT ?", [*args, limit]).fetchall()
        if not rows:
            return "no notes match"
        out = []
        for t, path, k, title, snip in rows:
            out.append("  ".join(x for x in (t, path, f"[{k}]" if k else "", title, self._cited(path)) if x))
            if snip:
                out.append("    " + " ".join(snip.split()))
        return "\n".join(out)

    def read(self, ref: str) -> str:
        ref = ref.strip().removeprefix("note:")
        rows = self.db.execute("SELECT path FROM notes WHERE path = ? OR id = ? ORDER BY time DESC", (ref, ref)).fetchall()
        if not rows:
            raise LookupError(f"no note {ref!r}: give its path under the notes root, or its id (the file name without .md)")
        if len(rows) > 1:
            return f"{len(rows)} notes have the id {ref!r}; read one by path:\n" + "\n".join(p for (p,) in rows)
        path = rows[0][0]
        text = (self.root / path).read_text(errors="replace")
        if len(text) > READ_MAX:
            text = text[:READ_MAX] + f"\n[... {len(text) - READ_MAX} more characters]"
        cites = self.db.execute("SELECT target, kind FROM cites WHERE path=? ORDER BY kind, target", (path,)).fetchall()
        return "\n".join([f"# {path}", text, *([f"\n## Cited evidence ({len(cites)})"] if cites else []),
                          *(self._summary(t, k) for t, k in cites)])

    def _summary(self, target: str, kind: str) -> str:
        """One line: what the catalog holds for a cited id."""
        if not self.catalog:
            return f"- {target}: no catalog attached"
        if kind == "run":
            r = self.db.execute("SELECT tool, state, rc, start_utc, host FROM ev.attempts WHERE id=?", (target,)).fetchone()
            labels = [f"{k}={_clip(v, 60)}" for k, v in self.db.execute("SELECT key, value FROM ev.labels WHERE target=? ORDER BY ts", (target,))]
            if r is None:
                return f"- {target}: no attempt in the catalog" + (f"; labels {', '.join(labels)}" if labels else "")
            return f"- {target}: {r[0]} {r[1]} rc={r[2]} {(r[3] or '')[:16]} on {r[4]}" + (f"; labels {', '.join(labels)}" if labels else "")
        arts = self.db.execute("SELECT id, kind, bytes FROM ev.artifacts WHERE id >= ? AND id < ? LIMIT 3", (target, target + "g")).fetchall()
        if not arts:
            return f"- {target}: not in the catalog"
        if len(arts) > 1:
            return f"- {target}: ambiguous ({len(arts)}+ artifacts)"
        return f"- {target}: {arts[0][1]}, {arts[0][2]} bytes"

    def _citing(self, target: str, kind: str, full: str | None = None) -> list[str]:
        if kind == "run":
            q, args = "c.target = ?", [target]
        elif full:
            q, args = "c.kind = 'art' AND c.target = substr(?, 1, length(c.target))", [full]
        else:
            q, args = "c.kind = 'art' AND (substr(c.target, 1, length(?)) = ? OR c.target = substr(?, 1, length(c.target)))", [target] * 3
        rows = self.db.execute(f"SELECT DISTINCT n.time, n.path, n.title FROM cites c JOIN notes n ON n.path = c.path WHERE {q} "
                               "ORDER BY n.time", args).fetchall()
        return [f"  {t}  {p}  {title}" for t, p, title in rows]

    def evidence(self, ident: str) -> str:
        ident = ident.strip()
        out: list[str] = []
        full = None
        if ident.startswith("note:"):
            ref = ident[5:]
            notes = self.db.execute("SELECT path FROM notes WHERE id = ?", (ref,)).fetchall()
            out.append(f"{ident}: " + (", ".join(p for (p,) in notes) if notes else "no such note in the index"))
            if self.catalog:
                labels = self.db.execute("SELECT target, key, value, by, ts FROM ev.labels WHERE ref = ? ORDER BY ts", (ident,)).fetchall()
                out.append(f"labels whose ref is {ident} ({len(labels)}):")
                out += [f"  {t}  {k} = {_clip(v)}  ({by}, {ts})" for t, k, v, by, ts in labels]
        elif RUN_RE.match(ident):
            if self.catalog:
                r = self.db.execute("SELECT tool, state, rc, validation, start_utc, wall_s, host, project, campaign, source_commit "
                                    "FROM ev.attempts WHERE id=?", (ident,)).fetchone()
                if r is None:
                    out.append(f"{ident}: no attempt in the catalog")
                else:
                    keys = ("rc", "validation", "start", "wall_s", "host", "project", "campaign", "source")
                    vals = (r[2], r[3], r[4], None if r[5] is None else round(r[5]), r[6], r[7], r[8], (r[9] or "")[:12])
                    out.append(f"{ident}: {r[0]} {r[1]} " + " ".join(f"{k}={v}" for k, v in zip(keys, vals) if v not in (None, "")))
                    for name, art in self.db.execute("SELECT o.name, o.artifact FROM ev.attempt_outputs o WHERE o.attempt=? ORDER BY o.name", (ident,)):
                        k = self.db.execute("SELECT kind FROM ev.artifacts WHERE id=?", (art,)).fetchone()
                        out.append(f"  output {name}: {art[:12]} ({k[0] if k else 'not in the catalog'})")
                out += self._labels(ident)
        elif ART_RE.fullmatch(ident):
            if self.catalog:
                arts = self.db.execute("SELECT id, kind, bytes, payload_type FROM ev.artifacts WHERE id >= ? AND id < ? LIMIT 11",
                                       (ident, ident + "g")).fetchall()
                if not arts:
                    out.append(f"{ident}: not in the catalog")
                elif len(arts) > 1:
                    out.append(f"{ident}: ambiguous, {'more than 10' if len(arts) > 10 else len(arts)} artifacts:")
                    out += [f"  {a[0]}  {a[1]}" for a in arts[:10]]
                else:
                    full = arts[0][0]
                    out.append(f"{full}: {arts[0][1]}, {arts[0][2]} bytes" + (f", payload {arts[0][3]}" if arts[0][3] else ""))
                    for (att,) in self.db.execute("SELECT attempt FROM ev.attempt_outputs WHERE artifact=? ORDER BY attempt", (full,)):
                        out.append(f"  made by {att}")
                    out += self._labels(full)
        else:
            raise ValueError(f"{ident!r}: an id is `art:<hex>` (8 or more digits), a run id (rYYYYMMDD-HHMMSS-xxxx) or `note:<id>`")
        if not self.catalog:
            out.insert(0, "no catalog attached: `research data refresh` builds the local store's, or pass --catalog with a compact copy")
        if not ident.startswith("note:"):
            citing = self._citing(ident, "run" if RUN_RE.match(ident) else "art", full)
            out.append(f"notes citing it ({len(citing)}):")
            out += citing
        return "\n".join(out)

    def _labels(self, target: str) -> list[str]:
        rows = self.db.execute("SELECT key, value, by, ts, ref FROM ev.labels WHERE target=? ORDER BY ts, path", (target,)).fetchall()
        return [f"labels ({len(rows)}):" if rows else "labels: none"] + [
            f"  {k} = {_clip(v)}  ({by}, {ts}{', ref ' + _clip(ref, 80) if ref else ''})" for k, v, by, ts, ref in rows]

    def sql(self, query: str) -> str:
        """`query` on a read-only connection to the index, with the catalog attached as `ev`: no writes, no ATTACH, at most
        MAX_ROWS rows and SQL_BUDGET progress ticks."""
        ro = sqlite3.connect(_uri(self.path, ro=True), uri=True, timeout=30)
        try:
            if self.catalog:
                ro.execute("ATTACH DATABASE ? AS ev", (_uri(self.catalog, ro=True),))
            ro.execute("PRAGMA query_only = ON")
            ro.set_authorizer(_deny_attach)
            ticks = [0]

            def tick() -> int:
                ticks[0] += 1
                return int(ticks[0] > SQL_BUDGET)

            ro.set_progress_handler(tick, 1000)
            try:
                cur = ro.execute(query)
                rows = cur.fetchmany(MAX_ROWS + 1)
            except sqlite3.OperationalError as e:
                if str(e) == "interrupted":
                    raise ValueError(f"the query took more than {SQL_BUDGET * 1000:,} SQLite steps; narrow it") from None
                raise
            cols = [d[0] for d in cur.description or []]
        finally:
            ro.close()
        lines = ["\t".join(cols)] if cols else []
        lines += ["\t".join("" if v is None else str(v)[:300] for v in r) for r in rows[:MAX_ROWS]]
        if len(rows) > MAX_ROWS:
            lines.append(f"(the first {MAX_ROWS} rows)")
        return "\n".join(lines) or "(no rows)"


def main(argv: list[str], default_root: Path) -> int:
    ap = argparse.ArgumentParser(prog="research notes index", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="the notes clone (default $RESEARCH_NOTES or ~/.research/notes)")
    ap.add_argument("--index", type=Path, default=None, help="the index file (default ~/.research/notes-index/<hash>.sqlite)")
    ap.add_argument("--catalog", type=Path, default=None, help="the evidence catalog (default: the local store's catalog.sqlite)")
    ap.add_argument("--search", metavar="TEXT", help="print the notes matching TEXT (search_notes)")
    ap.add_argument("--evidence", metavar="ID", help="print what the catalog holds for ID and the notes citing it")
    ns = ap.parse_args(argv)
    root = (ns.root or default_root).expanduser()
    if not root.is_dir():
        print(f"research notes index: no notes root at {root}", file=sys.stderr)
        return 2
    idx = NotesIndex(root, ns.index, ns.catalog)
    try:
        rep = idx.refresh()
        if ns.search is not None:
            print(idx.search(ns.search))
        elif ns.evidence:
            print(idx.evidence(ns.evidence))
        else:
            print(f"research notes index: {rep['notes']} notes ({rep['read']} read, {rep['dropped']} dropped) in {idx.path}; "
                  f"catalog {idx.catalog or 'none'}")
    except (ValueError, LookupError) as e:
        print(f"research notes index: {e}", file=sys.stderr)
        return 2
    finally:
        idx.close()
    return 0
