"""`research chats`: transcript parsing, redaction, the FTS index, and ingest -> anchor -> pull through an fs remote."""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from research import chats as C

CHAT = "11111111-1111-4111-8111-111111111111"
SUB = "22222222-2222-4222-8222-222222222222"
ONLY_META = "33333333-3333-4333-8333-333333333333"
SECRET = "abcd1234efgh5678ijkl9012mnop3456"


def _line(role, *items):
    return json.dumps({"role": role, "message": {"content": list(items)}})


def _text(t):
    return {"type": "text", "text": t}


@pytest.fixture
def mirror(tmp_path):
    root = tmp_path / "mirror"
    chat_dir = root / "transcripts" / "Users-x-projects-veritor" / "agent-transcripts" / CHAT
    (chat_dir / "subagents").mkdir(parents=True)
    (chat_dir / f"{CHAT}.jsonl").write_text("\n".join([
        _line("user", _text("<timestamp>Wednesday, Sep 2, 2026, 8:16 AM (UTC-7)</timestamp>\n<user_query>\nWhy did F-r19-int-25 fail? "
                            "Use the global_match check.\n</user_query>\n<dynamic_tool_catalog>BIGNOISE catalogword</dynamic_tool_catalog>")),
        _line("assistant", _text("Looking at module_body.py."), {"type": "tool_use", "name": "Read", "input": {"path": "/x/module_body.py"}},
              {"type": "tool_use", "name": "Task", "input": {"description": "probe", "prompt": "p" * 1000}}),
        json.dumps({"type": "turn_ended", "status": "success"}),
        "{not json",
        json.dumps({"type": "turn_ended", "status": "error", "error": "[invalid_argument] Error"}),
        _line("user", _text(f"<timestamp>Thursday, Sep 3, 2026, 11:05 PM (UTC-7)</timestamp>\n<user_query>AWS_SECRET_ACCESS_KEY={SECRET}</user_query>")),
    ]) + "\n")
    (chat_dir / "subagents" / f"{SUB}.jsonl").write_text("\n".join([
        _line("user", _text("<timestamp>Wednesday, Sep 2, 2026, 8:20 AM (UTC-7)</timestamp>\nInvestigate Gemma decode.")),
        _line("assistant", _text("Gemma uses sk-abcdefghijklmnopqrstuvwxyz0123 in the example.")),
    ]) + "\n")
    (root / "notes" / "lanes" / "demo").mkdir(parents=True)
    (root / "notes" / "lanes" / "demo" / "state.md").write_text("# demo\nDecided: Gemma rows use the fp8 path because of global_match.\n")
    (root / "notes" / "blob.bin").write_bytes(b"\0\1\2")
    (root / "meta").mkdir()
    (root / "meta" / "composer_meta.jsonl").write_text("\n".join(json.dumps(d) for d in [
        {"id": CHAT, "title": "Debug F-r19", "created_at": 1788361000000, "model": "m-1", "bubbles": 7, "workspace": "/Users/x/projects/veritor"},
        {"id": SUB, "title": None, "parent": CHAT, "subagent_type": "generalPurpose", "bubbles": 2},
        {"id": ONLY_META, "title": "No transcript", "created_at": 1788000000000, "bubbles": 3, "workspace": "/Users/x/projects/os"},
    ]) + "\n")
    return root


def test_parser_reads_text_tool_use_subagents_and_timestamps(mirror):
    files = {info["chat_id"]: (p, info) for p, info in C.transcript_files(mirror / "transcripts")}
    assert set(files) == {CHAT, SUB}
    assert files[SUB][1]["parent"] == CHAT and files[CHAT][1]["parent"] is None
    assert files[CHAT][1]["project"] == "veritor"

    rows, bad = C.parse_transcript(*files[CHAT])
    assert bad == 1
    assert [(r["line"], r["seq"], r["role"], r["kind"], r["tool_name"]) for r in rows] == [
        (1, 0, "user", "text", None), (2, 0, "assistant", "text", None), (2, 1, "assistant", "tool_use", "Read"),
        (2, 2, "assistant", "tool_use", "Task"), (5, 0, "event", "event", None), (6, 0, "user", "text", None)]
    assert rows[0]["ts"] == "2026-09-02T15:16:00Z" and rows[0]["ts_source"] == "tag"
    assert rows[1]["ts"] == "2026-09-02T15:16:00Z" and rows[1]["ts_source"] == "carry"
    assert rows[5]["ts"] == "2026-09-04T06:05:00Z"                     # 11:05 PM at UTC-7 is the next day in UTC
    assert "<dynamic_tool_catalog/>" in rows[0]["text"] and "BIGNOISE" not in rows[0]["text"]
    assert rows[2]["text"] == "path=/x/module_body.py"
    assert rows[3]["text"].startswith("description=probe prompt=ppp") and "…[+600]" in rows[3]["text"]
    assert rows[4]["text"] == "turn_ended error [invalid_argument] Error"

    sub_rows, _ = C.parse_transcript(*files[SUB])
    assert {r["parent_chat_id"] for r in sub_rows} == {CHAT} and sub_rows[0]["ts"] == "2026-09-02T15:20:00Z"
    assert C.parse_ts("garbage") is None
    assert C.project_name("Users-danielreuter-projects") == "projects" and C.project_name("empty-window") == "empty-window"


def test_redaction_counts_patterns_and_leaves_hashes_alone():
    secrets = {
        "cloudflare": "cfat_" + "A1b2" * 8,
        "sk": "sk-proj-" + "a1" * 15,
        "runpod": "rpa_" + "B1" * 15,
        "aws_access_key_id": "AKIA" + "ABCDEFGHIJKLMNOP",
        "github": "ghp_" + "x1" * 18,
        "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.abcdefghijk",
    }
    text = " | ".join(f"{k}: {v}" for k, v in secrets.items())
    text += f" | AWS_SECRET_ACCESS_KEY={SECRET} | \"secret_access_key\": \"{'f00d' * 16}\" | Authorization: Bearer {'Zz9' * 10}"
    text += " | github_pat_" + "Y1" * 20
    keep = ["art:" + "ab" * 32, 'session_token_env = "AWS_SESSION_TOKEN"', "risk-free-rate", "the task-xyz", "max_tokens: 4096"]
    counts: dict[str, int] = {}
    out = C.redact(text + " | " + " | ".join(keep), counts)
    assert counts == {"cloudflare": 1, "sk": 1, "runpod": 1, "aws_access_key_id": 1, "github": 2, "jwt": 1, "secret_assignment": 2, "bearer": 1}
    for v in [*secrets.values(), SECRET, "f00d" * 16, "Zz9" * 10]:
        assert v not in out
    for k in keep:
        assert k in out
    assert "AWS_SECRET_ACCESS_KEY=[REDACTED:secret_assignment]" in out and "Bearer [REDACTED:bearer]" in out


def test_fts_round_trip_titles_filters_and_incremental_refresh(mirror, tmp_path):
    conn = C.open_index(tmp_path / "index.sqlite")
    rep = C.refresh(conn, mirror / "transcripts", mirror / "notes", C.load_meta(mirror / "meta" / "composer_meta.jsonl"))
    assert rep["transcripts"]["parsed"] == 2 and rep["notes"] == {"notes": 1, "skipped": 1}
    assert rep["coverage"] == {"transcript_chats": 2, "with_metadata": 2, "with_title": 1, "metadata_chats": 3, "metadata_without_transcript": 1}

    hits = C.search(conn, "F-r19-int-25")                                # not FTS5 syntax: retried as a quoted phrase
    assert [(h["chat_id"], h["title"], h["ts"][:10], h["project"]) for h in hits["messages"]] == [(CHAT, "Debug F-r19", "2026-09-02", "veritor")]
    assert C.search(conn, "global_match")["messages"][0]["role"] == "user"
    assert {(h["kind"], h["tool_name"]) for h in C.search(conn, "module_body.py", where="messages")["messages"]} == {("text", None), ("tool_use", "Read")}
    g = C.search(conn, "Gemma")
    assert {h["chat_id"] for h in g["messages"]} == {SUB} and g["messages"][0]["parent_title"] == "Debug F-r19"
    assert [n["path"] for n in g["notes"]] == ["lanes/demo/state.md"]
    assert "Debug F-r19 › sub" in C.format_hits(g)
    assert C.search(conn, "BIGNOISE") == {"messages": [], "notes": []}
    assert C.search(conn, SECRET)["messages"] == [] and C.search(conn, "REDACTED")["messages"]
    assert C.search(conn, "Gemma", since="2026-09-05")["messages"] == [] and C.search(conn, "Gemma", since="2026-09-02T15:20")["messages"]
    assert C.search(conn, "Gemma", project="nothere") == {"messages": [], "notes": []}

    s = C.stats(conn)
    assert s["redactions"]["messages"] == {"secret_assignment": 1, "sk": 1}
    assert conn.execute("SELECT title, project, message_count, transcript_rows FROM chats WHERE id=?", (CHAT,)).fetchone() == ("Debug F-r19", "veritor", 7, 6)
    assert conn.execute("SELECT project FROM chats WHERE id=?", (ONLY_META,)).fetchone() == ("os",)

    again = C.refresh(conn, mirror / "transcripts", mirror / "notes", C.load_meta(mirror / "meta" / "composer_meta.jsonl"))
    assert (again["transcripts"]["parsed"], again["transcripts"]["unchanged"]) == (0, 2)
    sub = next(p for p, i in C.transcript_files(mirror / "transcripts") if i["chat_id"] == SUB)
    with open(sub, "a") as f:
        f.write(_line("assistant", _text("Gemma follow-up: zebrafish")) + "\n")
    os.utime(sub, ns=(sub.stat().st_atime_ns, sub.stat().st_mtime_ns + 10**9))
    third = C.refresh(conn, mirror / "transcripts", mirror / "notes", C.load_meta(mirror / "meta" / "composer_meta.jsonl"))
    assert (third["transcripts"]["parsed"], third["transcripts"]["unchanged"]) == (1, 1)
    assert len(C.search(conn, "zebrafish")["messages"]) == 1 and C.stats(conn)["rows"]["messages"] == 9


def test_ingest_preserves_sources_and_search_pulls_the_index_through_the_anchor(mirror, tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "remote"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    rep = C.ingest(mirror, index=tmp_path / "built.sqlite", evict_free_gb=None, log=lambda s: None)
    assert rep["preserved"] is True
    assert set(rep["artifacts"]) == {"transcripts", "notes", "metadata", "index", "anchor"}
    assert rep["artifacts"]["anchor"]["id"] == C.anchor_id() and rep["anchor_label"]["pushed"] == 1

    from research.store.local import LocalStore

    st = LocalStore()
    tr = st.get_manifest(rep["artifacts"]["transcripts"]["id"])
    assert tr.kind == C.KIND_TRANSCRIPTS and tr.meta["kind"] == "chat-transcripts" and tr.meta["host"] == "laptop" and tr.meta["date"] == rep["date"]
    idx = st.get_manifest(rep["artifacts"]["index"]["id"])
    assert idx.refs == {k: rep["artifacts"][k]["id"] for k in ("transcripts", "notes", "metadata")}
    assert idx.meta["coverage"]["with_metadata"] == 2

    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "fresh-store"))      # another machine: empty store, same remote
    fresh = tmp_path / "fresh" / "index.sqlite"
    assert C.main(["search", "Gemma", "--index", str(fresh)]) == 0
    out = capsys.readouterr().out
    assert SUB in out and "lanes/demo/state.md" in out
    conn = sqlite3.connect(fresh)
    assert conn.execute("SELECT value FROM meta WHERE key='source_transcripts'").fetchone() == (rep["artifacts"]["transcripts"]["id"],)
