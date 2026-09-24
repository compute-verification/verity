"""`research chats`: a durable archive and a full-text index of Cursor chat transcripts and lane notes.

  research chats ingest MIRROR [--transcripts DIR] [--notes DIR] [--meta FILE] [--index PATH] [--host NAME] [--no-push] [--json]
  research chats search QUERY [--project P] [--since DATE] [--limit N] [--in all|messages|notes] [--sort rank|date] [--refresh] [--json]
  research chats pull [--art ART]                 # materialise the newest (or the given) chat-index/v1 at the index path
  research chats stats [--json]                   # row counts, metadata coverage, redactions by pattern, source artifacts

MIRROR is an rsynced copy of the laptop's sources (`~/.research/bin/chats-sync.sh` writes it nightly, see the README):

  MIRROR/transcripts/<project dir>/agent-transcripts/<chat>.jsonl | <chat>/<chat>.jsonl | <chat>/subagents/<sub>.jsonl
  MIRROR/notes/...                     ~/.research/notes, text only (rsync applies the notes repo's .gitignore and a 1 MB cap)
  MIRROR/meta/composer_meta.jsonl      one object per chat read from Cursor's state.vscdb (composerData:<id> + a bubbleId:<id>: key count)

`ingest` pushes the transcripts and the notes as trees and the metadata as a file, each PRESERVED (read back from the remote), with
meta {kind, host, date}; refreshes the index incrementally (a transcript is re-parsed only when its size or mtime changed; notes,
chats and metadata are rebuilt, they are small); pushes a compacted, gzipped copy of the index (chat-index/v1, refs to the three
sources); then labels the fixed, payload-free anchor artifact (ANCHOR_META) `superseded_by` the new index.  The anchor's id is a
function of its content, so `search`/`pull` on a machine without an index list that one label prefix on the remote to find the newest.
An index written by another SCHEMA_VERSION is rebuilt from empty: bump it whenever parsing or redaction changes.

A transcript's rows are one contiguous `messages.id` range in line order, so a hit's context is `WHERE id BETWEEN id-5 AND id+5`
(`search --json` gives the id; `files.first_id`/`last_id` bound the file).

The archive is verbatim.  The index is redacted (REDACTIONS: `[REDACTED:<pattern>]`) and keeps the count per source and pattern in
`redactions`; nothing prints a value.  The index also drops the boilerplate blocks Cursor attaches to user messages (NOISE_TAGS)
and renders a tool call as `key=value` pairs, each value clipped at TOOL_VALUE_MAX characters.  Tool results are not in the
transcripts at all.  A message's `ts` is its own `<timestamp>` tag (`ts_source` tag), else the last tag above it in the file
(carry), else the chat's createdAt from the metadata (created).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import re
import shutil
import socket
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator

from .store import kinds as _kinds

SCHEMA_VERSION = 1
INDEX_ENV = "RESEARCH_CHATS_INDEX"
DEFAULT_INDEX = Path.home() / ".research" / "chats" / "index.sqlite"
KIND_TRANSCRIPTS = "chat-transcripts/v1"
KIND_NOTES = "lane-notes/v1"
KIND_META = "chat-metadata/v1"
KIND_INDEX = "chat-index/v1"
KIND_ANCHOR = "chat-index-anchor/v1"
ANCHOR_META = {"name": "research chats index", "doc": "the newest superseded_by label names the current chat-index/v1 (research.chats)"}
LABEL_BY = "research-chats"
INDEX_PAYLOAD_NAME = "chats-index.sqlite.gz"
TOOL_VALUE_MAX = 400
TOOL_TEXT_MAX = 1500
NOTE_MAX_BYTES = 1 << 20
NOISE_TAGS = ("dynamic_tool_catalog", "available_subagent_types", "available_subagent_models", "mcp_server_catalog",
              "manually_attached_skills", "hooks_context", "cursor_commands", "agent_skills", "user_rules", "rules", "user_info",
              "git_status", "agent_transcripts", "terminal_files_information", "system_reminder")

for _k, _doc in {
    KIND_TRANSCRIPTS: "a copy of ~/.cursor/projects/*/agent-transcripts (research chats); payload tree; meta {kind, host, date, source, files}",
    KIND_NOTES: "a copy of ~/.research/notes, text only (research chats); payload tree; meta {kind, host, date, source, files}",
    KIND_META: "per-chat metadata read from Cursor's state.vscdb (research chats); payload composer_meta.jsonl; meta {kind, host, date, chats}",
    KIND_INDEX: "the chats SQLite/FTS5 index, redacted (research chats); payload chats-index.sqlite; refs transcripts, notes, metadata",
    KIND_ANCHOR: "fixed pointer for research chats: labels superseded_by = the newest chat-index/v1; payload null",
}.items():
    _kinds.KINDS.setdefault(_k, _doc)

# ---- redaction -------------------------------------------------------------------------------------------------------------------------

#: (name, pattern, group replaced: 0 = the whole match).  Applied in order, so a JWT inside a bearer header counts once, as `jwt`.
REDACTIONS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), 0),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{20,})"), 1),
    ("cloudflare", re.compile(r"\bcf[a-z]{2}_[A-Za-z0-9_-]{20,}"), 0),
    ("github", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})"), 0),
    ("runpod", re.compile(r"\brpa_[A-Za-z0-9]{20,}"), 0),
    ("sk", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), 0),
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), 0),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), 0),
    # NAME=value / "name": "value" where NAME says secret; the value must mix letters and digits (not an env var name, not a word)
    ("secret_assignment", re.compile(r"(?i)\b[A-Z0-9_.-]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY(?:_?ID)?)[A-Z0-9_]*"
                                     r"[\"']?\s*[=:]\s*[\"']?([A-Za-z0-9/+=._~-]{16,})"), 1),
)


def redact(text: str, counts: dict[str, int] | None = None) -> str:
    """`text` with every REDACTIONS match replaced by `[REDACTED:<name>]`; `counts[name]` += matches."""
    for name, rx, group in REDACTIONS:
        def sub(m: re.Match[str], name: str = name, group: int = group) -> str:
            val = m.group(group)
            if name == "secret_assignment" and not (re.search(r"\d", val) and re.search(r"[A-Za-z]", val)):
                return m.group(0)
            if counts is not None:
                counts[name] = counts.get(name, 0) + 1
            if group == 0:
                return f"[REDACTED:{name}]"
            whole, base = m.group(0), m.start()
            return whole[: m.start(group) - base] + f"[REDACTED:{name}]" + whole[m.end(group) - base:]
        text = rx.sub(sub, text)
    return text

# ---- transcripts -----------------------------------------------------------------------------------------------------------------------

TS_TAG = re.compile(r"<timestamp>\s*(.*?)\s*</timestamp>", re.S)
_TZ = re.compile(r"\((?:UTC|GMT)\s*(?:([+-])(\d{1,2})(?::?(\d{2}))?)?\)")
_TS_FORMATS = ("%A, %b %d, %Y, %I:%M %p", "%A, %B %d, %Y, %I:%M %p", "%A, %b %d, %Y, %I:%M:%S %p", "%b %d, %Y, %I:%M %p",
               "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")
_NOISE = re.compile(r"<(" + "|".join(NOISE_TAGS) + r")(?:\s[^>]*)?>.*?</\1>", re.S)


def utc_iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ms_iso(ms: Any) -> str | None:
    try:
        return utc_iso(dt.datetime.fromtimestamp(float(ms) / 1000, dt.timezone.utc)) if ms else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def parse_ts(raw: str) -> str | None:
    """`Wednesday, Sep 2, 2026, 8:16 AM (UTC-7)` (the tag Cursor writes) and a few ISO spellings -> `2026-09-02T15:16:00Z`."""
    s, tz = raw.strip(), dt.timezone.utc
    m = _TZ.search(s)
    if m:
        if m.group(1):
            sign = -1 if m.group(1) == "-" else 1
            tz = dt.timezone(sign * dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0)))
        s = s[: m.start()].strip()
    for fmt in _TS_FORMATS:
        try:
            return utc_iso(dt.datetime.strptime(s, fmt).replace(tzinfo=tz))
        except ValueError:
            continue
    return None


def project_name(dirname: str) -> str:
    """`Users-danielreuter-projects-veritor` -> `veritor`; `Users-danielreuter-projects` -> `projects`; anything else as is."""
    for rx in (r"^Users-[^-]+-projects-(.+)$", r"^Users-[^-]+-(.+)$"):
        m = re.match(rx, dirname)
        if m:
            return m.group(1)
    return dirname


def workspace_project(fs_path: str | None) -> str | None:
    """A workspace folder as Cursor names its project dir (`/Users/x/projects/veritor` -> `Users-x-projects-veritor`), shortened."""
    return project_name(re.sub(r"[^A-Za-z0-9]+", "-", fs_path.strip("/"))) if fs_path else None


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else f"{s[:n]}…[+{len(s) - n}]"


def render_tool(inp: Any) -> str:
    """A tool call's input as `key=value` pairs, each value clipped at TOOL_VALUE_MAX, the whole at TOOL_TEXT_MAX."""
    if not isinstance(inp, dict):
        return clip(json.dumps(inp, ensure_ascii=False, default=str), TOOL_TEXT_MAX)
    parts = [f"{k}={clip(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str), TOOL_VALUE_MAX)}" for k, v in inp.items()]
    return clip(" ".join(parts), TOOL_TEXT_MAX)


def transcript_files(root: Path) -> Iterator[tuple[Path, dict]]:
    """Every transcript under `root` (a copy of ~/.cursor/projects) with {rel, chat_id, parent, project}; a subagent's parent is
    the chat whose `subagents/` directory holds it."""
    for p in sorted(root.glob("*/agent-transcripts/**/*.jsonl")):
        parts = p.relative_to(root).parts
        parent = parts[-3] if len(parts) >= 5 and parts[-2] == "subagents" else None
        yield p, {"rel": p.relative_to(root).as_posix(), "chat_id": p.stem, "parent": parent, "project": project_name(parts[0])}


def parse_transcript(path: Path, info: dict, created: str | None = None, counts: dict[str, int] | None = None) -> tuple[list[dict], int]:
    """One row per content item (text / tool_use) of every line, plus `turn_ended` events that are not successes.  Returns
    (rows, unparseable lines).  Text is cleaned (NOISE_TAGS) and redacted here; `counts` collects the redactions."""
    rows: list[dict] = []
    bad, last_ts = 0, None
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, 1):
            try:
                obj = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                bad += 1
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue
            role = obj.get("role")
            if role is None:
                if obj.get("type") and obj.get("status") not in (None, "success"):
                    items = [{"type": "event", "text": " ".join(str(obj.get(k)) for k in ("type", "status", "error") if obj.get(k))}]
                    role = "event"
                else:
                    continue
            else:
                msg = obj.get("message")
                items = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(items, str):
                    items = [{"type": "text", "text": items}]
                if not isinstance(items, list):
                    continue
            for seq, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                kind, tool, ts_source = str(item.get("type") or "?"), None, None
                if kind == "tool_use":
                    tool, text = str(item.get("name") or "?"), render_tool(item.get("input"))
                elif isinstance(item.get("text"), str):
                    text = item["text"]
                    m = TS_TAG.search(text) if role == "user" else None
                    if m and (t := parse_ts(m.group(1))):
                        last_ts, ts_source = t, "tag"
                    if role == "user":
                        text = _NOISE.sub(lambda mm: f"<{mm.group(1)}/>", text)
                else:
                    text = clip(json.dumps({k: v for k, v in item.items() if k != "type"}, ensure_ascii=False, default=str), TOOL_TEXT_MAX)
                if ts_source is None:
                    ts_source = "carry" if last_ts else ("created" if created else None)
                rows.append({"chat_id": info["chat_id"], "parent_chat_id": info.get("parent"), "project": info.get("project"),
                             "file": info["rel"], "line": lineno, "seq": seq, "ts": last_ts or created, "ts_source": ts_source,
                             "role": str(role), "kind": kind, "tool_name": tool, "text": redact(text, counts)})
    return rows, bad

# ---- the index -------------------------------------------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, chat_id TEXT, parent_chat_id TEXT, project TEXT, size INTEGER, mtime_ns INTEGER,
                                 rows INTEGER, bad_lines INTEGER, indexed_utc TEXT, first_id INTEGER, last_id INTEGER);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, chat_id TEXT NOT NULL, parent_chat_id TEXT, project TEXT, file TEXT NOT NULL,
                                    line INTEGER, seq INTEGER, ts TEXT, ts_source TEXT, role TEXT, kind TEXT, tool_name TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, project TEXT, title TEXT, subtitle TEXT, created_at TEXT, updated_at TEXT, model TEXT,
                                 parent TEXT, root_parent TEXT, subagent_type TEXT, workspace TEXT, message_count INTEGER, transcript_rows INTEGER,
                                 first_ts TEXT, last_ts TEXT, in_meta INTEGER, archived INTEGER);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime TEXT, bytes INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS redactions(source TEXT, pattern TEXT, n INTEGER, PRIMARY KEY(source, pattern));
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, content='messages', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(path, text, content='notes', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF text ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, path, text) VALUES (new.id, new.path, new.text); END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, path, text) VALUES ('delete', old.id, old.path, old.text); END;
"""
_MSG_COLS = ("chat_id", "parent_chat_id", "project", "file", "line", "seq", "ts", "ts_source", "role", "kind", "tool_name", "text")


def index_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    return Path(explicit or os.environ.get(INDEX_ENV) or DEFAULT_INDEX).expanduser()


def open_index(path: Path, rebuild: bool = False) -> sqlite3.Connection:
    """Open (or create) the index; one written by another SCHEMA_VERSION, or `rebuild`, starts from empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not rebuild:
        old = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rebuild = old.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() != (str(SCHEMA_VERSION),)
        except sqlite3.Error:
            rebuild = True
        finally:
            old.close()
    if rebuild:
        path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    return conn


def load_meta(path: Path | None) -> dict[str, dict]:
    """composer_meta.jsonl -> {chat id: object}; a missing file is an empty map (the index then has no titles)."""
    out: dict[str, dict] = {}
    if path is None or not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("id"), str):
            out[doc["id"]] = doc
    return out


def _set_redactions(conn: sqlite3.Connection, source: str, counts: dict[str, int]) -> None:
    conn.execute("DELETE FROM redactions WHERE source=?", (source,))
    conn.executemany("INSERT INTO redactions VALUES (?,?,?)", [(source, k, v) for k, v in sorted(counts.items()) if v])


def _drop_file_rows(conn: sqlite3.Connection, rel: str, span: tuple[int | None, int | None]) -> None:
    """A file's rows are one contiguous id range (inserted together), so no index on `messages.file` is needed to delete them."""
    if span[0] is not None:
        conn.execute("DELETE FROM messages WHERE id BETWEEN ? AND ? AND file=?", (span[0], span[1], rel))


def refresh_transcripts(conn: sqlite3.Connection, root: Path, meta: dict[str, dict], log=None) -> dict:
    """Re-parse every transcript whose (size, mtime) differs from the last pass, drop the rows of files that are gone."""
    seen, parsed, skipped, rows_in, bad = set(), 0, 0, 0, 0
    have = {r[0]: (r[1], r[2], r[3], r[4]) for r in conn.execute("SELECT path, size, mtime_ns, first_id, last_id FROM files")}
    now = utc_iso(dt.datetime.now(dt.timezone.utc))
    for p, info in transcript_files(root):
        st, rel = p.stat(), info["rel"]
        seen.add(rel)
        old = have.get(rel)
        if old and old[:2] == (st.st_size, st.st_mtime_ns):
            skipped += 1
            continue
        counts: dict[str, int] = {}
        rows, nbad = parse_transcript(p, info, ms_iso((meta.get(info["chat_id"]) or {}).get("created_at")), counts)
        if old:
            _drop_file_rows(conn, rel, old[2:])
        first = (conn.execute("SELECT max(id) FROM messages").fetchone()[0] or 0) + 1
        conn.executemany(f"INSERT INTO messages(id,{','.join(_MSG_COLS)}) VALUES (?,{','.join('?' * len(_MSG_COLS))})",
                         [(first + i, *(r[c] for c in _MSG_COLS)) for i, r in enumerate(rows)])
        _set_redactions(conn, rel, counts)
        conn.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (rel, info["chat_id"], info["parent"], info["project"], st.st_size, st.st_mtime_ns, len(rows), nbad, now,
                      first if rows else None, first + len(rows) - 1 if rows else None))
        parsed, rows_in, bad = parsed + 1, rows_in + len(rows), bad + nbad
        if log and parsed % 200 == 0:
            log(f"research chats: parsed {parsed} transcripts ({rows_in} rows)")
            conn.commit()
    gone = [p for p in have if p not in seen]
    for p in gone:
        _drop_file_rows(conn, p, have[p][2:])
        conn.execute("DELETE FROM files WHERE path=?", (p,))
        conn.execute("DELETE FROM redactions WHERE source=?", (p,))
    conn.commit()
    return {"parsed": parsed, "unchanged": skipped, "removed": len(gone), "rows_inserted": rows_in, "bad_lines": bad}


def refresh_chats(conn: sqlite3.Connection, meta: dict[str, dict]) -> dict:
    """Rebuild `chats` from the metadata plus every chat id the transcripts name; fill parents the transcript layout could not."""
    conn.execute("DELETE FROM chats")
    counts: dict[str, int] = {}
    conn.executemany("INSERT INTO chats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,1,?)", [
        (cid, workspace_project(d.get("workspace")), redact(d["title"], counts) if isinstance(d.get("title"), str) else None,
         redact(d["subtitle"], counts) if isinstance(d.get("subtitle"), str) else None, ms_iso(d.get("created_at")),
         ms_iso(d.get("updated_at") or d.get("checkpoint_updated_at")), d.get("model"), d.get("parent"), d.get("root_parent"),
         d.get("subagent_type"), d.get("workspace"), d.get("bubbles"), 1 if d.get("archived") else 0)
        for cid, d in meta.items()])
    _set_redactions(conn, "chats", counts)
    conn.execute("INSERT OR IGNORE INTO chats(id, project, parent, in_meta) SELECT chat_id, min(project), max(parent_chat_id), 0 FROM files GROUP BY chat_id")
    conn.execute("UPDATE chats SET project = (SELECT min(f.project) FROM files f WHERE f.chat_id = chats.id) "
                 "WHERE EXISTS (SELECT 1 FROM files f WHERE f.chat_id = chats.id)")
    conn.execute("UPDATE chats SET transcript_rows = s.n, first_ts = s.a, last_ts = s.b FROM "
                 "(SELECT chat_id, count(*) AS n, min(ts) AS a, max(ts) AS b FROM messages GROUP BY chat_id) AS s WHERE s.chat_id = chats.id")
    conn.execute("UPDATE messages SET parent_chat_id = (SELECT parent FROM chats WHERE chats.id = messages.chat_id) "
                 "WHERE parent_chat_id IS NULL AND chat_id IN (SELECT id FROM chats WHERE parent IS NOT NULL)")
    conn.commit()
    return coverage(conn)


def coverage(conn: sqlite3.Connection) -> dict:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {"transcript_chats": q("SELECT count(DISTINCT chat_id) FROM files"),
            "with_metadata": q("SELECT count(*) FROM chats WHERE in_meta=1 AND id IN (SELECT chat_id FROM files)"),
            "with_title": q("SELECT count(*) FROM chats WHERE title IS NOT NULL AND id IN (SELECT chat_id FROM files)"),
            "metadata_chats": q("SELECT count(*) FROM chats WHERE in_meta=1"),
            "metadata_without_transcript": q("SELECT count(*) FROM chats WHERE in_meta=1 AND id NOT IN (SELECT chat_id FROM files)")}


def refresh_notes(conn: sqlite3.Connection, root: Path | None) -> dict:
    """Rebuild `notes` from every UTF-8 text file (<= NOTE_MAX_BYTES, no NUL) under `root`, `.git` excluded."""
    conn.execute("DELETE FROM notes")
    conn.execute("DELETE FROM redactions WHERE source LIKE 'notes:%'")
    n = skipped = 0
    if root is not None and root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != ".git")
            for name in sorted(filenames):
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                    if not p.is_file() or st.st_size > NOTE_MAX_BYTES:
                        skipped += 1
                        continue
                    data = p.read_bytes()
                    text = data.decode("utf-8") if b"\0" not in data[:8192] else None
                except (OSError, UnicodeDecodeError):
                    text = None
                if text is None:
                    skipped += 1
                    continue
                rel, counts = p.relative_to(root).as_posix(), {}
                conn.execute("INSERT INTO notes(path, mtime, bytes, text) VALUES (?,?,?,?)",
                             (rel, utc_iso(dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)), st.st_size, redact(text, counts)))
                _set_redactions(conn, f"notes:{rel}", counts)
                n += 1
    conn.commit()
    return {"notes": n, "skipped": skipped}


def refresh(conn: sqlite3.Connection, transcripts: Path, notes: Path | None, meta: dict[str, dict], log=None) -> dict:
    t = refresh_transcripts(conn, transcripts, meta, log)
    nt = refresh_notes(conn, notes)
    cov = refresh_chats(conn, meta)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('refreshed_utc', ?)", (utc_iso(dt.datetime.now(dt.timezone.utc)),))
    conn.commit()
    return {"transcripts": t, "notes": nt, "coverage": cov}


def stats(conn: sqlite3.Connection) -> dict:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    red = {"messages": {}, "notes": {}, "chats": {}}
    for src, pat, n in conn.execute("SELECT CASE WHEN source='chats' THEN 'chats' WHEN source LIKE 'notes:%' THEN 'notes' ELSE 'messages' END, "
                                    "pattern, sum(n) FROM redactions GROUP BY 1, 2 ORDER BY 1, 3 DESC"):
        red[src][pat] = n
    return {"rows": {t: q(f"SELECT count(*) FROM {t}") for t in ("files", "messages", "chats", "notes")},
            "coverage": coverage(conn), "redactions": red,
            "ts_source": dict(conn.execute("SELECT coalesce(ts_source, 'none'), count(*) FROM messages GROUP BY 1").fetchall()),
            "meta": dict(conn.execute("SELECT key, value FROM meta").fetchall())}

# ---- search ----------------------------------------------------------------------------------------------------------------------------


def _quoted(query: str) -> str:
    return " ".join('"' + t.replace('"', '""') + '"' for t in query.split())


def _match(conn: sqlite3.Connection, sql: str, args: list, query: str) -> list[tuple]:
    """Run with the query as FTS5 syntax; when FTS5 cannot parse it (`F-r19-int-25`, `a.b`), as quoted phrases instead."""
    try:
        return conn.execute(sql, [query, *args]).fetchall()
    except sqlite3.OperationalError as e:
        if "fts5" not in str(e) and "no such column" not in str(e) and "syntax error" not in str(e):
            raise
        return conn.execute(sql, [_quoted(query), *args]).fetchall()


def search(conn: sqlite3.Connection, query: str, *, project: str | None = None, since: str | None = None, limit: int = 20,
           where: str = "all", sort: str = "rank") -> dict:
    """Message hits (deduplicated: a subagent's transcript is often stored twice) and note hits; `project` searches messages only."""
    out: dict[str, list[dict]] = {"messages": [], "notes": []}
    if where in ("all", "messages"):
        sql = ("SELECT m.id, m.ts, m.project, m.role, m.kind, m.tool_name, m.chat_id, m.parent_chat_id, m.file, m.line, "
               "snippet(messages_fts, 0, '«', '»', '…', 16), c.title, pc.title "
               "FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid LEFT JOIN chats c ON c.id = m.chat_id "
               "LEFT JOIN chats pc ON pc.id = m.parent_chat_id WHERE messages_fts MATCH ?")
        args: list = []
        if project:
            sql += " AND m.project LIKE ?"
            args.append(f"%{project}%")
        if since:
            sql += " AND m.ts >= ?"
            args.append(since)
        sql += (" ORDER BY m.ts DESC" if sort == "date" else " ORDER BY rank") + " LIMIT ?"
        seen = set()
        for r in _match(conn, sql, [*args, limit * 4], query):
            h = dict(zip(("id", "ts", "project", "role", "kind", "tool_name", "chat_id", "parent_chat_id", "file", "line", "snippet", "title",
                          "parent_title"), r))
            if (h["chat_id"], h["role"], h["snippet"]) in seen:
                continue
            seen.add((h["chat_id"], h["role"], h["snippet"]))
            out["messages"].append(h)
            if len(out["messages"]) >= limit:
                break
    if where in ("all", "notes") and not project:
        sql = "SELECT n.path, n.mtime, snippet(notes_fts, 1, '«', '»', '…', 16) FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid WHERE notes_fts MATCH ?"
        args = []
        if since:
            sql += " AND n.mtime >= ?"
            args.append(since)
        sql += (" ORDER BY n.mtime DESC" if sort == "date" else " ORDER BY rank") + " LIMIT ?"
        out["notes"] = [dict(zip(("path", "mtime", "snippet"), r)) for r in _match(conn, sql, [*args, limit], query)]
    return out


def _flat(s: str | None, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def format_hits(hits: dict) -> str:
    lines = []
    for h in hits["messages"]:
        title = h["title"] or (f"{h['parent_title']} › sub" if h["parent_title"] else "(untitled)")
        role = f"tool:{h['tool_name']}" if h["kind"] == "tool_use" else h["role"]
        lines.append(f"{(h['ts'] or '????-??-??T??:??')[:16].replace('T', ' ')}  {_flat(h['project'], 14):14s}  {_flat(title, 40):40s}  "
                     f"{_flat(role, 16):16s}  {_flat(h['snippet'], 150)}  {h['chat_id']}")
    if hits["notes"]:
        lines.append("-- notes")
        for h in hits["notes"]:
            lines.append(f"{h['mtime'][:16].replace('T', ' ')}  {_flat(h['path'], 60):60s}  {_flat(h['snippet'], 150)}")
    return "\n".join(lines) if lines else "(no hits)"

# ---- store: archive, index artifact, anchor --------------------------------------------------------------------------------------------


def anchor_manifest():
    from .store.model import ArtifactManifest  # noqa: PLC0415

    return ArtifactManifest(kind=KIND_ANCHOR, meta=dict(ANCHOR_META), refs={}, payload=None).validate()


def anchor_id() -> str:
    return anchor_manifest().id


def _preserve(st, art: str, what: str, log) -> dict:
    rep = st.push(art, preserve=True)
    state = "PRESERVED" if rep["preserved"] else "NOT preserved: " + "; ".join(rep["errors"][:3])
    log(f"research chats: {what} {art} {state} ({rep['uploaded']} uploaded, {rep['skipped']} already there, {rep['seconds']} s)")
    return {"id": art, "preserved": bool(rep["preserved"]), "uploaded": rep["uploaded"], "errors": rep["errors"][:3]}


def latest_index_art(st) -> str | None:
    """The newest chat-index/v1: the anchor's newest `superseded_by` label on the remote, else the local catalog's newest by built_utc."""
    from .store import labels_sync as LS  # noqa: PLC0415

    r, best = st.remote, None
    if r is not None:
        anchor = anchor_id()
        for key in LS.remote_labels_for(r, anchor).values():
            tmp = st.root / f".tmp-chats-label-{os.getpid()}"
            try:
                r.get(key, tmp)
                lab = LS.parse_label_bytes(tmp.read_bytes())
            except (KeyError, ValueError, OSError):
                continue
            finally:
                tmp.unlink(missing_ok=True)
            if lab.key == "superseded_by" and isinstance(lab.value, str) and (best is None or lab.ts > best[0]):
                best = (lab.ts, lab.value)
        if best:
            return best[1]
    local = [(st.get_manifest(row["id"]).meta.get("built_utc") or "", row["id"]) for row in st.index.select_artifacts(kind=KIND_INDEX)]
    return max(local)[1] if local else None


def pull_index(dest: Path, art: str | None = None, store_root: str | None = None, log=None) -> tuple[Path, str]:
    """Materialise `art` (default: the newest chat-index/v1) at `dest`, temp + rename; a `.gz` payload is decompressed."""
    from .store.local import LocalStore  # noqa: PLC0415

    st = LocalStore(store_root)
    art = art or latest_index_art(st)
    if not art:
        raise SystemExit("research chats: no chat-index/v1 in the local catalog or behind the anchor on the remote (run `research chats ingest`)")
    if log:
        log(f"research chats: pulling {art} -> {dest}")
    blob = st.fetch(art)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    with (gzip.open(blob, "rb") if (st.get_manifest(art).payload.name or "").endswith(".gz") else open(blob, "rb")) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    os.replace(tmp, dest)
    return dest, art


def ingest(mirror: Path, *, transcripts: Path | None = None, notes: Path | None = None, meta: Path | None = None, index: Path | None = None,
           host: str = "laptop", date: str | None = None, push: bool = True, store_root: str | None = None, evict_free_gb: float | None = 3.0,
           rebuild: bool = False, log=None) -> dict:
    say = log or (lambda s: print(s, file=sys.stderr))
    transcripts = transcripts or mirror / "transcripts"
    notes = notes if notes is not None else mirror / "notes"
    meta = meta if meta is not None else mirror / "meta" / "composer_meta.jsonl"
    index = index_path(index)
    now = dt.datetime.now(dt.timezone.utc)
    date = date or now.strftime("%Y-%m-%d")
    if not transcripts.is_dir():
        raise SystemExit(f"research chats ingest: {transcripts} is not a directory")
    rep: dict[str, Any] = {"date": date, "index": str(index), "artifacts": {}}
    st = None
    if push:
        from .store.local import LocalStore  # noqa: PLC0415

        st = LocalStore(store_root)
        if st.remote is None:
            raise SystemExit(f"research chats ingest: no usable remote ({getattr(st, 'remote_error', None) or 'no [remote] in the store config'}); "
                             "--no-push refreshes the index only")
        src = {"host": host, "date": date}
        n_tr = sum(1 for _ in transcripts.rglob("*") if _.is_file())
        tr = st.put_artifact(KIND_TRANSCRIPTS, {"kind": "chat-transcripts", **src, "source": "~/.cursor/projects/*/agent-transcripts", "files": n_tr},
                             tree=transcripts)
        rep["artifacts"]["transcripts"] = _preserve(st, tr, "transcripts", say)
        if notes.is_dir():
            n_nt = sum(1 for _ in notes.rglob("*") if _.is_file())
            nt = st.put_artifact(KIND_NOTES, {"kind": "lane-notes", **src, "source": "~/.research/notes", "files": n_nt}, tree=notes)
            rep["artifacts"]["notes"] = _preserve(st, nt, "notes", say)
        if meta.is_file():
            md = st.put_artifact(KIND_META, {"kind": "chat-metadata", **src, "chats": len(load_meta(meta))}, file=meta)
            rep["artifacts"]["metadata"] = _preserve(st, md, "metadata", say)
    conn = open_index(index, rebuild)
    meta_map = load_meta(meta)
    rep["refresh"] = refresh(conn, transcripts, notes if notes.is_dir() else None, meta_map, say)
    for k in ("transcripts", "notes", "metadata"):
        if k in rep["artifacts"]:
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"source_{k}", rep["artifacts"][k]["id"]))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('source_date', ?)", (date,))
    conn.commit()
    rep["stats"] = stats(conn)
    if st is not None:
        snap_dir = index.parent / f".push-{os.getpid()}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap, packed = snap_dir / INDEX_PAYLOAD_NAME[: -len(".gz")], snap_dir / INDEX_PAYLOAD_NAME
        try:
            conn.execute("VACUUM INTO ?", (str(snap),))
            conn.close()
            with open(snap, "rb") as src, gzip.GzipFile(packed, "wb", compresslevel=6, mtime=0) as out:
                shutil.copyfileobj(src, out, 1 << 20)
            s = rep["stats"]
            idx_meta = {"kind": "chat-index", "host": socket.gethostname(), "source_host": host, "date": date, "built_utc": utc_iso(now),
                        "schema": SCHEMA_VERSION, "rows": s["rows"], "coverage": s["coverage"], "redactions": s["redactions"],
                        "sqlite_bytes": snap.stat().st_size}
            snap.unlink()
            refs = {k: v["id"] for k, v in rep["artifacts"].items()}
            idx = st.put_artifact(KIND_INDEX, idx_meta, refs, file=packed)
            rep["artifacts"]["index"] = _preserve(st, idx, "index", say)
        finally:
            shutil.rmtree(snap_dir, ignore_errors=True)
        if rep["artifacts"]["index"]["preserved"]:
            anchor = st.put_manifest(anchor_manifest())
            rep["artifacts"]["anchor"] = _preserve(st, anchor, "anchor", say)
            st.add_label(anchor, "superseded_by", idx, by=LABEL_BY, ref=idx)
            lr = st.push_labels([anchor])
            rep["anchor_label"] = {"pushed": len(lr.get("pushed", [])), "failed": lr.get("failed", [])[:3]}
        if evict_free_gb:
            from .store.evict import evict_store  # noqa: PLC0415

            rep["evicted_bytes"] = evict_store(st, target_free_bytes=int(evict_free_gb * 1e9))["evicted_bytes"]
    else:
        conn.close()
    rep["preserved"] = all(v["preserved"] for v in rep["artifacts"].values()) if rep["artifacts"] else None
    return rep

# ---- command line ----------------------------------------------------------------------------------------------------------------------


def _ingest_lines(rep: dict) -> list[str]:
    s, r = rep["stats"], rep["refresh"]
    out = [f"{k:12s} {v['id']}  {'PRESERVED' if v['preserved'] else 'NOT PRESERVED'}" for k, v in rep["artifacts"].items()]
    t = r["transcripts"]
    out.append(f"transcripts  parsed {t['parsed']}, unchanged {t['unchanged']}, removed {t['removed']}, rows +{t['rows_inserted']}, bad lines {t['bad_lines']}")
    out.append("rows         " + ", ".join(f"{k} {v}" for k, v in s["rows"].items()))
    c = s["coverage"]
    out.append(f"coverage     {c['with_metadata']}/{c['transcript_chats']} transcript chats have metadata ({c['with_title']} titled); "
               f"{c['metadata_without_transcript']} metadata chats have no transcript")
    for src, d in s["redactions"].items():
        out.append(f"redactions   {src}: " + (", ".join(f"{k} {v}" for k, v in d.items()) or "none"))
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="research chats", description=__doc__.split("\n\n")[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ig = sub.add_parser("ingest", help="archive the mirror (preserved), refresh the index, push it")
    ig.add_argument("mirror", type=Path)
    ig.add_argument("--transcripts", type=Path, default=None, help="default MIRROR/transcripts")
    ig.add_argument("--notes", type=Path, default=None, help="default MIRROR/notes")
    ig.add_argument("--meta", type=Path, default=None, help="default MIRROR/meta/composer_meta.jsonl")
    ig.add_argument("--host", default="laptop", help="where the sources live (meta.host of the archive artifacts)")
    ig.add_argument("--date", default=None, help="meta.date (default: today, UTC)")
    ig.add_argument("--no-push", action="store_true", help="refresh the local index only; nothing goes to the store")
    ig.add_argument("--evict-free-gb", type=float, default=3.0, help="afterwards evict PRESERVED blobs until this much disk is free (0: never)")
    ig.add_argument("--rebuild", action="store_true", help="start the index from empty (also automatic when SCHEMA_VERSION changed)")
    se = sub.add_parser("search", help="FTS5 query over message text and notes")
    se.add_argument("query")
    se.add_argument("--project", default=None, help="substring of the project (veritor, os, ...); messages only")
    se.add_argument("--since", default=None, metavar="DATE", help="YYYY-MM-DD[THH:MM] (UTC)")
    se.add_argument("--limit", type=int, default=20)
    se.add_argument("--in", dest="where", choices=("all", "messages", "notes"), default="all")
    se.add_argument("--sort", choices=("rank", "date"), default="rank")
    se.add_argument("--refresh", action="store_true", help="pull the newest index from the store first")
    pu = sub.add_parser("pull", help="materialise the newest (or --art) chat-index/v1 at the index path")
    pu.add_argument("--art", default=None)
    stp = sub.add_parser("stats", help="row counts, metadata coverage, redactions by pattern")
    for p in (ig, se, pu, stp):
        p.add_argument("--index", type=Path, default=None, help=f"index file (default ${INDEX_ENV} or {DEFAULT_INDEX})")
        p.add_argument("--store", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
        p.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    say = lambda s: print(s, file=sys.stderr)  # noqa: E731
    if ns.cmd == "ingest":
        rep = ingest(ns.mirror, transcripts=ns.transcripts, notes=ns.notes, meta=ns.meta, index=ns.index, host=ns.host, date=ns.date,
                     push=not ns.no_push, store_root=ns.store, evict_free_gb=ns.evict_free_gb or None, rebuild=ns.rebuild, log=say)
        print(json.dumps(rep, indent=2, sort_keys=True, default=str) if ns.json else "\n".join(_ingest_lines(rep)))
        return 0 if rep["preserved"] in (True, None) else 1
    path = index_path(ns.index)
    if ns.cmd == "pull" or (ns.cmd == "search" and (ns.refresh or not path.exists())):
        _, art = pull_index(path, getattr(ns, "art", None), ns.store, say)
        if ns.cmd == "pull":
            print(json.dumps({"index": str(path), "art": art}) if ns.json else f"{path}  {art}")
            return 0
    if not path.exists():
        print(f"research chats: no index at {path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    if ns.cmd == "stats":
        s = stats(conn)
        s["index_bytes"] = path.stat().st_size
        print(json.dumps(s, indent=2, sort_keys=True))
        return 0
    hits = search(conn, ns.query, project=ns.project, since=ns.since, limit=ns.limit, where=ns.where, sort=ns.sort)
    print(json.dumps(hits, indent=2, ensure_ascii=False) if ns.json else format_hits(hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
