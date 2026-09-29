"""`research notes mcp`: the notes index and the evidence catalog it joins (notes_index.py) as a Model Context Protocol server on
stdin/stdout, for an agent's MCP client (README, "Notes and evidence over MCP").  JSON-RPC 2.0, one message per line; it hosts
nothing and reads no credential.  Before each tool call the index catches up with the clone (only changed notes are re-read).

    research notes mcp [--root DIR] [--index PATH] [--catalog PATH]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, BinaryIO

from .notes_index import MAX_ROWS, NotesIndex

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SQL_TABLES = ("notes(path, lane, id, kind, status, created, time, sender, recipient, campaign, title, bytes); notes_fts(title, body) "
              "with rowid = notes.nid; cites(path, target, kind: art | run); and with a catalog, ev.attempts(id, tool, state, rc, "
              "validation, start_utc, wall_s, host, project, campaign, source_commit), ev.artifacts(id, kind, bytes, payload_type), "
              "ev.labels(target, key, value, by, ts, ref), ev.attempt_outputs(attempt, name, artifact), ev.attempt_params(attempt, key, "
              "value), ev.artifact_meta(id, key, value)")
TOOLS: list[dict[str, Any]] = [
    {"name": "search_notes",
     "description": "Search the research notes in full text, best match first, or list the newest when `query` is empty. Each hit: "
                    "time, path, kind, title, how many ids it cites and how many of those the evidence catalog holds, and a snippet.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "words (all must match), \"a phrase\", OR, NOT, prefix*; plain text also works"},
         "lane": {"type": "string", "description": "only this lane's notes"},
         "kind": {"type": "string", "description": "front matter kind: handoff, report, finding, answer, ..."},
         "since": {"type": "string", "description": "only notes at or after this UTC time, e.g. 2026-09-28 or 2026-09-28T06:00Z"},
         "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS, "default": 20}}}},
    {"name": "read_note",
     "description": "One note's full text, by its path under the notes root or its id (the file name without .md, or `note:<id>`), "
                    "followed by each id it cites and what the evidence catalog holds for it. Any form notes cite by works: "
                    "`note:<lane>/<name>`, `note:<campaign>/<lane>/<name>`, or a path from before the note moved into a month folder.",
     "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}},
    {"name": "evidence",
     "description": "What the evidence catalog holds for an id, and the notes that cite it. A run id (rYYYYMMDD-HHMMSS-xxxx): its "
                    "attempt (tool, state, rc, when, where), outputs and labels. `art:<hex>` (8+ digits): kind, bytes, the attempt that "
                    "made it, labels. `note:<id>`: the note, and the labels whose ref names it (in any of the forms read_note takes), "
                    "e.g. `abandoned` on the evidence behind it.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "sql",
     "description": f"Read-only SQLite over the notes index and the evidence catalog; at most {MAX_ROWS} rows, tab-separated. Tables: "
                    + SQL_TABLES + ". An `art:` id in notes is a prefix: match it with `id >= target AND id < target || 'g'`.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


class UnknownTool(Exception):
    pass


def _error(mid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


class Server:
    def __init__(self, index: NotesIndex):
        self.index = index

    def handle(self, msg: Any) -> dict | None:
        """The response to one JSON-RPC message; None for a notification or a response (this server sends no requests)."""
        if not isinstance(msg, dict) or not isinstance(msg.get("method"), str):
            if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                return None
            return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
        if "id" not in msg:
            return None
        mid, method, params = msg["id"], msg["method"], msg.get("params") or {}
        if method == "initialize":
            v = params.get("protocolVersion")
            result: dict = {"protocolVersion": v if v in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "research-notes", "version": "1"},
                            "instructions": f"Research notes at {self.index.root}; evidence catalog "
                                            f"{self.index.catalog or 'none (evidence lookups say so)'}. Search and read notes, and "
                                            "look up the runs and art: ids they cite."}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                result = self.call(params.get("name"), params.get("arguments") or {})
            except UnknownTool as e:
                return _error(mid, -32602, str(e))
        else:
            return _error(mid, -32601, f"Method not found: {method}")
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def call(self, name: Any, args: dict) -> dict:
        idx = self.index
        tools = {"search_notes": lambda a: idx.search(**a), "read_note": lambda a: idx.read(a["ref"]),
                 "evidence": lambda a: idx.evidence(a["id"]), "sql": lambda a: idx.sql(a["query"])}
        if name not in tools:
            raise UnknownTool(f"Unknown tool: {name}")
        try:
            idx.refresh()
            text = tools[name](args)
        except (TypeError, ValueError, LookupError, sqlite3.Error, OSError) as e:
            return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
        return {"content": [{"type": "text", "text": text}], "isError": False}


def serve(server: Server, stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Answer each line of `stdin` on `stdout` until end of input."""
    for raw in iter(stdin.readline, b""):
        if not raw.strip():
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            out: Any = _error(None, -32700, "Parse error")
        else:
            out = [r for m in msg if (r := server.handle(m)) is not None] or None if isinstance(msg, list) else server.handle(msg)
        if out is not None:
            stdout.write(json.dumps(out, ensure_ascii=False).encode() + b"\n")
            stdout.flush()
    return 0


def main(argv: list[str], default_root: Path) -> int:
    ap = argparse.ArgumentParser(prog="research notes mcp", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="the notes clone (default $RESEARCH_NOTES or ~/.research/notes)")
    ap.add_argument("--index", type=Path, default=None, help="the index file (default ~/.research/notes-index/<hash>.sqlite)")
    ap.add_argument("--catalog", type=Path, default=None, help="the evidence catalog (default: the local store's catalog.sqlite)")
    ns = ap.parse_args(argv)
    root = (ns.root or default_root).expanduser()
    if not root.is_dir():
        print(f"research notes mcp: no notes root at {root}", file=sys.stderr)
        return 2
    idx = NotesIndex(root, ns.index, ns.catalog)
    try:
        rep = idx.refresh()
        print(f"research notes mcp: {rep['notes']} notes under {root}; catalog {idx.catalog or 'none'}", file=sys.stderr)
        return serve(Server(idx), sys.stdin.buffer, sys.stdout.buffer)
    finally:
        idx.close()
