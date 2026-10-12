"""The derived index (`research.store.bucket_index`, README §5.1): built from the bucket's records alone, the same rows whether
built in one pass or pass by pass, every record checked against its key, and the public index a whitelist of the private one."""
from __future__ import annotations

import http.server
import json
import os
import threading
import urllib.parse
from datetime import datetime, timedelta, timezone

import pytest

from research.store import bucket_index as BI
from research.store import canon, ids
from research.store.model import Label
from research.store.remote import FsRemote
from research.store.remote_s3 import S3Remote

T0 = datetime(2026, 10, 11, 18, 0, tzinfo=timezone.utc)


def _put(root, key: str, doc) -> None:
    p = root / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(canon.dumps(doc))
    os.utime(p, (1_791_700_000, 1_791_700_000))


def artifact(root, kind: str = "bench-result/v1", meta=None, refs=None, payload=None) -> str:
    doc = {"schema": "research/artifact/v0.1", "kind": kind, "meta": meta or {}, "refs": refs or {}, "payload": payload}
    aid = ids.art_id(doc)
    _put(root, f"manifests/{ids.hex_of(aid)}.json", doc)
    return aid


def run(root, rid: str, *, outputs=None, inputs=None, tool: str = "bench_vu", start: str = "2026-10-11T17:00:00.123456+00:00") -> str:
    doc = {"schema": "research/attempt/v0.1", "id": rid, "tool": {"name": tool, "version": "1"}, "derivation": "drv:" + "d" * 64,
           "reuse_key": None, "argv": ["python", "secret-arg"], "params": {"argv": ["python", "secret-arg"]}, "key_params": [],
           "inputs": inputs or {}, "conditions": {"hw.sku": "NVIDIA H100 80GB HBM3", "hw.count": 8, "hw.hostname": "vy-g5",
                                                  "rt.driver": "570.1", "rt.cuda": "12.8"},
           "key_conditions": [], "source": {"commit": "c" * 40, "tree": "t" * 40, "dirty": False},
           "closure": {"a.py": "0" * 64}, "execution": {"start_utc": start, "wall_s": 12.5, "host": "vy-g5", "run_dir": "/x"},
           "cost": {"usd": 1.0}, "status": {"state": "done", "rc": 0}, "outputs": outputs or {},
           "telemetry": {"events": None, "resources": None}}
    _put(root, f"attempts/{rid}.json", doc)
    return rid


def label(root, target: str, key: str, value, by: str = "lane", ts: str = "2026-10-11T17:30:00Z", ref=None, local_layout=False) -> str:
    lab = Label(target=target, key=key, value=value, by=by, ts=ts, ref=ref)
    h = canon.sha256(lab.to_doc())
    name = f"{ts.replace('-', '').replace(':', '')}-{by}" if local_layout else h
    _put(root, f"labels/{ids.target_dirname(target)}/{name}.json", lab.to_doc())
    return f"label:{h}"


def test_one_pass_and_pass_by_pass_give_the_same_rows_including_a_run_published_late(tmp_path):
    """The rebuild test carried over (design invariant 4).  A run whose id sorts before the newest indexed one (custody published it
    late) is missed by the start-after listing and caught by the hourly full listing; after that the two builds agree."""
    b = tmp_path / "bucket"
    remote = FsRemote(b)
    a1 = artifact(b, meta={"x": 1})
    run(b, "r20261011-170000-0001", outputs={"result": a1})
    label(b, a1, "verified", "accepted", by="red-team-zk")
    inc = tmp_path / "inc.sqlite"
    assert BI.index(remote, inc, now=T0)["runs_listing"] == "full"

    a2 = artifact(b, kind="tc-evidence/v1", refs={"inputs": [a1]}, payload={"type": "tree", "sha256": "e" * 64, "bytes": 3,
                                                                              "files": [{"path": "x", "sha256": "f" * 64, "bytes": 3}]})
    run(b, "r20261011-180000-0002", outputs={"evidence": a2}, inputs={"golden": a1})
    rep = BI.index(remote, inc, now=T0 + timedelta(minutes=5))
    assert rep["runs_listing"] == "tail" and rep["listed"][BI.RUNS_TAIL] == 1 and rep["added"] == {"artifact": 1, "run": 1, "label": 0}

    run(b, "r20261011-175000-0003")                          # sorts before the newest indexed run
    label(b, "pr:1782@" + "a" * 40, "ready", True, by="infra")
    rep = BI.index(remote, inc, now=T0 + timedelta(minutes=10))
    assert rep["runs_listing"] == "tail" and rep["added"] == {"artifact": 0, "run": 0, "label": 1}
    early = tmp_path / "early.sqlite"
    BI.index(remote, early, now=T0)
    assert BI.dump(inc) != BI.dump(early)

    rep = BI.index(remote, inc, now=T0 + timedelta(minutes=61))
    assert rep["runs_listing"] == "full" and rep["added"]["run"] == 1
    assert BI.index(remote, inc, now=T0 + timedelta(minutes=66))["read"] == 0          # nothing new: nothing read

    fresh = tmp_path / "fresh.sqlite"
    BI.index(remote, fresh, now=T0 + timedelta(days=3))
    assert BI.dump(inc) == BI.dump(fresh)
    idx = BI.SqliteIndex(inc)
    assert dict(idx.db.execute("SELECT prefix, keys FROM listings")) == {"manifests/": 2, "labels/": 2, "attempts/": 3, "attempts/>": 0}
    idx.close()


def test_rows_drop_bulk_and_findings_text_and_name_every_id_a_record_names(tmp_path):
    b = tmp_path / "bucket"
    tree = {"type": "tree", "sha256": "e" * 64, "bytes": 3, "files": [{"path": "x", "sha256": "f" * 64, "bytes": 3}]}
    art = artifact(b, kind="run-record/v1", meta={"long": "y" * (BI.FIELD_MAX + 1), "short": [True, 1.5, None]}, payload=tree)
    finding = artifact(b, kind="redteam-findings/v1", meta={"findings": ["the exploit"]}, refs={"target": art})
    rid = run(b, "r20261011-170000-0001", outputs={"run_record": art}, inputs={"golden": finding})
    lid = label(b, "drv:" + "d" * 64, "note", "hello")
    db = tmp_path / "i.sqlite"
    assert BI.index(FsRemote(b), db, now=T0)["errors"] == []
    idx = BI.SqliteIndex(db)
    a = idx.row(art)
    assert a.type == "artifact" and a.kind == "run-record/v1" and a.time == "2026-10-11T06:26:40Z"
    assert a.doc["payload"] == {"type": "tree", "sha256": "e" * 64, "bytes": 3}
    assert idx.row(finding).doc == {"schema": "research/artifact/v0.1", "kind": "redteam-findings/v1", "refs": {"target": art}, "payload": None}
    r = idx.row(rid)
    assert r.kind == "bench_vu" and r.time == "2026-10-11T17:00:00Z" and "closure" not in r.doc and r.doc["argv"] == ["python", "secret-arg"]
    assert idx.row(lid).kind == "note" and idx.row(lid).time == "2026-10-11T17:30:00Z"
    fields = dict(idx.db.execute("SELECT path, value FROM fields WHERE id=?", (art,)))
    assert fields["meta.short.0"] == "true" and fields["meta.short.1"] == "1.5" and fields["meta.short.2"] == "null" and "meta.long" not in fields
    assert dict(idx.db.execute("SELECT path, value FROM fields WHERE id=?", (rid,)))["conditions.hw.sku"] == "NVIDIA H100 80GB HBM3"
    assert idx.edges_of(rid) == [(rid, "inputs.golden", finding), (rid, "outputs.run_record", art)]
    assert idx.edges_of(finding) == [(finding, "refs.target", art)]
    assert idx.edges_of(lid) == [(lid, "target", "drv:" + "d" * 64)]
    idx.close()


def test_a_record_that_does_not_match_its_key_is_reported_and_left_out(tmp_path):
    b = tmp_path / "bucket"
    good = artifact(b)
    _put(b, f"manifests/{'0' * 64}.json", {"schema": "research/artifact/v0.1", "kind": "x/v1", "meta": {}, "refs": {}, "payload": None})
    _put(b, "attempts/r20261011-170000-0001.json", {"id": "r20261011-170000-0002"})
    lab = Label(target=good, key="verified", value="accepted", by="v", ts="2026-10-11T17:30:00Z").to_doc()
    _put(b, f"labels/{ids.target_dirname(good)}/{'1' * 64}.json", lab)
    local = label(b, good, "verifier", "lean", local_layout=True)
    db = tmp_path / "i.sqlite"
    rep = BI.index(FsRemote(b), db, now=T0)
    assert len(rep["errors"]) == 3 and rep["added"] == {"artifact": 1, "run": 0, "label": 1}
    assert {r[0] for r in BI.dump(db)["records"]} == {good, local}
    assert BI.index(FsRemote(b), db, now=T0 + timedelta(minutes=5))["added"] == {"artifact": 0, "run": 0, "label": 0}


def test_the_public_index_is_the_published_artifacts_their_runs_summarised_and_whitelisted_labels(tmp_path):
    b, pub = tmp_path / "bucket", tmp_path / "public"
    fixture = artifact(b, kind="fixture/v1")
    private_art = artifact(b, kind="proof/v1")
    run(b, "r20261011-170000-0001", outputs={"result": fixture, "proof": private_art}, inputs={"golden": private_art})
    run(b, "r20261011-170000-0002", outputs={"proof": private_art})
    on_fixture = label(b, fixture, "verified", "accepted", ref="art:" + "9" * 64)
    label(b, fixture, "note", "private words")
    label(b, private_art, "verified", "accepted")
    on_run = label(b, "r20261011-170000-0001", "proof_class", "COMPONENT", by="red-team-zk")
    only_public = {"schema": "research/artifact/v0.1", "kind": "fixture/v1", "meta": {"old": True}, "refs": {}, "payload": None}
    old = ids.art_id(only_public)
    _put(pub, f"manifests/{ids.hex_of(fixture)}.json", json.loads((b / f"manifests/{ids.hex_of(fixture)}.json").read_bytes()))
    _put(pub, f"manifests/{ids.hex_of(old)}.json", only_public)
    priv, public = tmp_path / "i.sqlite", tmp_path / "p.sqlite"
    BI.index(FsRemote(b), priv, now=T0)
    rep = BI.index(FsRemote(pub), public, policy=BI.PUBLIC, private=priv, now=T0)
    assert rep["errors"] == [] and rep["read"] == 1 and rep["added"] == {"artifact": 2, "run": 1, "label": 2}
    rows = BI.dump(public)
    assert {r[0] for r in rows["records"]} == {fixture, old, "r20261011-170000-0001", on_fixture, on_run}
    idx = BI.SqliteIndex(public)
    assert idx.row("r20261011-170000-0001").doc == {
        "schema": "research/attempt/v0.1", "id": "r20261011-170000-0001", "tool": {"name": "bench_vu"}, "status": {"state": "done"},
        "execution": {"start_utc": "2026-10-11T17:00:00.123456+00:00", "wall_s": 12.5},
        "conditions": {"hw.sku": "NVIDIA H100 80GB HBM3", "hw.count": 8, "rt.driver": "570.1", "rt.cuda": "12.8"}, "source": {"commit": "c" * 40}}
    assert idx.edges_of("r20261011-170000-0001") == [("r20261011-170000-0001", "outputs.result", fixture)]
    assert "ref" not in idx.row(on_fixture).doc and idx.row(on_fixture).doc["by"] == "lane"
    assert dict(idx.db.execute("SELECT prefix, keys FROM listings")) == {"manifests/": 2}
    idx.close()
    BI.index(FsRemote(pub), public, policy=BI.PUBLIC, private=priv, now=T0 + timedelta(minutes=5))
    assert BI.dump(public) == rows


def test_the_public_pass_reads_the_private_index_in_batches_and_builds_what_one_batch_builds(tmp_path, monkeypatch):
    b, pub = tmp_path / "bucket", tmp_path / "public"
    arts = [artifact(b, kind="fixture/v1", meta={"i": i}) for i in range(5)]
    for i, a in enumerate(arts):
        run(b, f"r20261011-17000{i}-0001", outputs={"result": a}, inputs={"golden": arts[(i + 1) % 5]})
        label(b, a, "verified", "accepted")
        _put(pub, f"manifests/{ids.hex_of(a)}.json", json.loads((b / f"manifests/{ids.hex_of(a)}.json").read_bytes()))
    priv = tmp_path / "i.sqlite"
    BI.index(FsRemote(b), priv, now=T0)
    BI.index(FsRemote(pub), tmp_path / "one.sqlite", policy=BI.PUBLIC, private=priv, now=T0)
    monkeypatch.setattr(BI, "IN_BATCH", 2)
    rep = BI.index(FsRemote(pub), tmp_path / "two.sqlite", policy=BI.PUBLIC, private=priv, now=T0)
    assert rep["read"] == 0 and rep["added"] == {"artifact": 5, "run": 5, "label": 5}
    assert BI.dump(tmp_path / "two.sqlite") == BI.dump(tmp_path / "one.sqlite")


@pytest.fixture
def s3(tmp_path):
    objects = {"attempts/r1.json": b"1", "attempts/r2.json": b"2", "attempts/r3.json": b"3", "labels/x.json": b"x"}
    queries: list[dict] = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D102
            pass

        def do_GET(self):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
            queries.append(q)
            keys = sorted(k for k in objects if k.startswith(q.get("prefix", "")) and k > q.get("start-after", ""))
            start = int(q.get("continuation-token", "0"))
            page, more = keys[start:start + 1], start + 1 < len(keys)
            xml = ('<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                   + "".join(f"<Contents><Key>{k}</Key><LastModified>2026-10-11T17:0{i}:00.000Z</LastModified></Contents>" for i, k in enumerate(page))
                   + f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
                   + (f"<NextContinuationToken>{start + 1}</NextContinuationToken>" if more else "") + "</ListBucketResult>").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(xml)))
            self.end_headers()
            self.wfile.write(xml)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield S3Remote(f"http://127.0.0.1:{srv.server_address[1]}", "bkt", "ak", "sk"), queries
    srv.shutdown()
    srv.server_close()


def test_s3_list_times_pages_and_starts_after_a_key(s3):
    remote, queries = s3
    assert remote.list_times("attempts/") == [(f"attempts/r{i}.json", "2026-10-11T17:00:00.000Z") for i in (1, 2, 3)]
    queries.clear()
    assert remote.list_times("attempts/", start_after="attempts/r2.json") == [("attempts/r3.json", "2026-10-11T17:00:00.000Z")]
    assert [q.get("start-after") for q in queries] == ["attempts/r2.json"]


def test_the_cli_builds_the_index_from_a_directory(tmp_path, capsys):
    from research.store.cli import main

    b = tmp_path / "bucket"
    a = artifact(b)
    label(b, a, "verified", "accepted")
    db = tmp_path / "i.sqlite"
    assert main(["index", "--store", str(tmp_path / "store"), "--from", str(b), "--db", str(db), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)[0]
    assert rep["added"] == {"artifact": 1, "run": 0, "label": 1} and rep["size"]["records"]["rows"] == 2


def test_pg_env_takes_the_url_apart_and_values_and_copy_text_round_trip():
    url = "postgresql://store_indexer:p%40ss%2Fw@ep-x.eu-central-1.aws.neon.tech/store?sslmode=require&channel_binding=require"
    assert BI.pg_env(url) == {"PGHOST": "ep-x.eu-central-1.aws.neon.tech", "PGPORT": "5432", "PGDATABASE": "store", "PGUSER": "store_indexer",
                              "PGPASSWORD": "p@ss/w", "PGSSLMODE": "require", "PGCHANNELBINDING": "require"}
    assert BI.pg_env("postgresql://u@/store?host=/run/pg")["PGHOST"] == "/run/pg"
    with pytest.raises(ValueError, match="not a postgres URL"):
        BI.pg_env("sqlite:///x.db")
    assert BI._values([(None, 1.5, "it's\x00")]) == "(NULL,'1.5','it''s\ufffd')"
    assert BI._uncopy(r"\N") is None and BI._uncopy(r"a\tb\nc\\d") == "a\tb\nc\\d"


def test_postgres_index_hands_psql_the_url_only_in_its_environment_and_names_the_schema_on_every_table(tmp_path):
    """The SQL PostgresIndex writes, read off a stand-in `psql` that records what it is given."""
    seen = tmp_path / "seen"
    fake = tmp_path / "psql"
    fake.write_text(f"#!{__import__('sys').executable}\nimport json, os, sys\n"
                    f"open({str(seen)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'pg': {{k: os.environ[k] for k in os.environ if k.startswith('PG')}},"
                    " 'sql': sys.stdin.read()}) + '\\n')\nprint('1')\n")
    fake.chmod(0o755)
    url = "postgresql://store_indexer:sekrit@db.example/store?sslmode=require"
    idx = BI.PostgresIndex(url, "store_private", psql=str(fake))
    b = tmp_path / "bucket"
    a = artifact(b, meta={"note": "tab\there"})
    rows, errors = BI._read(FsRemote(b), [(f"manifests/{ids.hex_of(a)}.json", None)], 1)
    assert errors == [] and idx.add(rows) == 1
    calls = [json.loads(ln) for ln in seen.read_text().splitlines()]
    assert all("sekrit" not in " ".join(c["argv"]) and c["pg"]["PGPASSWORD"] == "sekrit" and c["pg"]["PGHOST"] == "db.example" for c in calls)
    ddl, insert = calls[0]["sql"], calls[1]["sql"]
    assert "CREATE TABLE IF NOT EXISTS store_private.records" in ddl and "DROP" not in ddl and "SCHEMA" not in ddl
    assert "INSERT INTO store_private.records" in insert and "TEMP" not in insert
    assert f"('{a}','meta.note','tab\there')" in insert
    with pytest.raises(ValueError, match="bad schema name"):
        BI.PostgresIndex(url, "store_private; DROP", psql=str(fake))


@pytest.fixture
def postgres(tmp_path):
    """A throwaway cluster, owned like Neon's `store`: the schemas are another role's, the indexer may only use and create in them."""
    if not os.environ.get("VERITY_RESEARCH_POSTGRES"):
        pytest.skip("starts a Postgres server (a few seconds; needs initdb): set VERITY_RESEARCH_POSTGRES=1")
    import glob
    import subprocess
    import tempfile

    bins = sorted(glob.glob("/usr/lib/postgresql/*/bin/initdb"))
    assert bins, "VERITY_RESEARCH_POSTGRES=1 needs initdb (apt install postgresql)"
    bindir = os.path.dirname(bins[-1])
    sock = tempfile.mkdtemp(prefix="pg", dir=tmp_path if len(str(tmp_path)) < 80 else None)
    data = tmp_path / "pgdata"
    subprocess.run([f"{bindir}/initdb", "-D", str(data), "-A", "trust", "-U", "admin"], check=True, capture_output=True)
    subprocess.run([f"{bindir}/pg_ctl", "-D", str(data), "-w", "-l", str(tmp_path / "pg.log"), "-o", f"-k {sock} -c listen_addresses=''", "start"],
                   check=True, capture_output=True)
    try:
        sql = ("REVOKE TEMP ON DATABASE store FROM PUBLIC; CREATE ROLE store_indexer LOGIN; CREATE ROLE store_private_reader LOGIN; CREATE SCHEMA store_private; CREATE SCHEMA store_public;"
               " GRANT USAGE, CREATE ON SCHEMA store_private, store_public TO store_indexer; GRANT USAGE ON SCHEMA store_private TO store_private_reader;"
               " ALTER DEFAULT PRIVILEGES FOR ROLE store_indexer IN SCHEMA store_private GRANT SELECT ON TABLES TO store_private_reader;"
               " ALTER ROLE store_private_reader SET search_path = store_private;"
               " ALTER ROLE store_private_reader SET default_transaction_read_only = on;")
        subprocess.run(["psql", "-X", "-q", "-h", sock, "-U", "admin", "-d", "postgres", "-c", "CREATE DATABASE store;"], check=True)
        subprocess.run(["psql", "-X", "-q", "-h", sock, "-U", "admin", "-d", "store", "-c", sql], check=True)
        yield f"postgresql://store_indexer@/store?host={sock}", f"postgresql://store_private_reader@/store?host={sock}"
    finally:
        subprocess.run([f"{bindir}/pg_ctl", "-D", str(data), "-w", "-m", "fast", "stop"], check=True, capture_output=True)


def test_postgres_builds_the_rows_sqlite_builds_and_a_rebuild_keeps_the_readers_grants(tmp_path, postgres, monkeypatch):
    url, reader = postgres
    monkeypatch.setattr(BI, "IN_BATCH", 1)
    b, pub = tmp_path / "bucket", tmp_path / "public"
    fixture = artifact(b, kind="fixture/v1", meta={"weird": "a\tb\nc\\d", "n": [1, 2.5, None]})
    proof = artifact(b, kind="proof/v1", refs={"inputs": [fixture]})
    run(b, "r20261011-170000-0001", outputs={"result": fixture, "proof": proof})
    label(b, fixture, "verified", "accepted", ref="art:" + "9" * 64)
    label(b, "r20261011-170000-0001", "note", "words")
    _put(pub, f"manifests/{ids.hex_of(fixture)}.json", json.loads((b / f"manifests/{ids.hex_of(fixture)}.json").read_bytes()))
    lite, lite_pub = tmp_path / "i.sqlite", tmp_path / "p.sqlite"
    pg, pg_pub = BI.PostgresIndex(url, "store_private"), BI.PostgresIndex(url, "store_public")
    calls = []
    monkeypatch.setattr(pg, "run", lambda script, run=pg.run: calls.append(script) or run(script))
    for _ in range(2):
        assert BI.index(FsRemote(b), lite, now=T0)["errors"] == [] and BI.index(FsRemote(b), pg, now=T0)["errors"] == []
        BI.index(FsRemote(pub), lite_pub, policy=BI.PUBLIC, private=lite, now=T0)
        calls.clear()
        BI.index(FsRemote(pub), pg_pub, policy=BI.PUBLIC, private=pg, now=T0)
        assert len(calls) <= 6
    assert BI.dump(pg) == BI.dump(lite) and BI.dump(pg_pub) == BI.dump(lite_pub)
    assert pg.listed_at(BI.MANIFESTS) == "2026-10-11T18:00:00Z" and pg.newest_run() == "r20261011-170000-0001"
    assert pg.size()["records"]["rows"] == 5
    assert BI.PostgresIndex(reader, "store_private", create=False).run("SELECT count(*) FROM records;") == [["5"]]


def _readers_world(tmp_path):
    """A bucket of one run, its output and a label on it, indexed where `reader` looks, and a store holding one artifact and one
    label the bucket lacks."""
    from research.store.local import LocalStore

    b, root = tmp_path / "bucket", tmp_path / "store"
    a1 = artifact(b, meta={"x": 1})
    run(b, "r20261011-170000-0001", outputs={"result": a1})
    label(b, a1, "verified", "accepted")
    st = LocalStore(root)
    (tmp_path / "f.txt").write_text("here only\n")
    mine = st.put_artifact("bench-result/v1", {"x": 1}, {}, file=tmp_path / "f.txt")
    st.add_label(a1, "verified", "rejected", "infra", ts="2026-10-11T19:00:00Z")
    return b, st, a1, mine


def test_displays_read_the_index_with_what_this_store_holds_beside_it(tmp_path, monkeypatch, capsys):
    """Step 5 of the store design: `select` and `sql --index` read the index (Neon's store_private with its reader's URL, else a local
    build), complete as of its listing, and name this store's records the index lacks rather than leave them out."""
    from research.store import labels_sync as LS
    from research.store.cli import main

    b, st, a1, mine = _readers_world(tmp_path)
    assert BI.reader(st.root) is None, "no index here: displays read the catalog"
    BI.index(FsRemote(b), BI.default_path(st.root), now=T0)
    idx = BI.reader(st.root)
    assert isinstance(idx, BI.SqliteIndex) and BI.as_of(idx) == "2026-10-11T18:00:00Z"
    for facets in ({"kind": "bench-result/v1"}, {"where": {"x": "1"}}, {"where": {"meta.x": "1"}}, {"labels": [("verified", "accepted")]},
                   {"tool": "bench_vu"}, {"attempt": "r20261011-170000-0001"}, {"derivation": "drv:" + "d" * 64}):
        assert [i for i, _ in BI.select_artifacts(idx, **facets)] == [a1], facets
    for facets in ({"kind": "proof/v1"}, {"where": {"x": "2"}}, {"labels": [("verified", "rejected")]}, {"tool": "check"}):
        assert BI.select_artifacts(idx, **facets) == [], facets
    (h,) = LS.local_labels(st)
    assert set(BI.pending(st, idx)) == {mine, f"label:{h}"}
    idx.close()

    capsys.readouterr()
    assert main(["select", "--store", str(st.root), "--where", "x=1"]) == 0
    out, err = capsys.readouterr()
    assert a1 in out and f"{mine}" in out and "(pending: not in the index)" in out.splitlines()[[mine in ln for ln in out.splitlines()].index(True)]
    assert "2 artifact(s) from the index" in err and "complete as of 2026-10-11T18:00:00Z), 1 of them only here" in err
    assert main(["sql", "--index", "--store", str(st.root), "SELECT type, count(*) AS n FROM records GROUP BY type ORDER BY type"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == ["type\tn", "artifact\t1", "label\t1", "run\t1"]
    assert "complete as of 2026-10-11T18:00:00Z; 2 record(s) here are not in it" in err
    assert main(["sql", "--index", "--store", str(st.root), "SELECT * FROM attempts"]) == 2
    assert "tables: records(id, type, kind, time, doc)" in capsys.readouterr().err
    monkeypatch.setenv(BI.READER_ENV, "postgresql://store_private_reader:pw@db.example/store?sslmode=require")
    pg = BI.reader(st.root)
    assert isinstance(pg, BI.PostgresIndex) and pg.schema == "store_private" and pg.env["PGPASSWORD"] == "pw"


def test_the_postgres_reader_reads_read_only_with_column_names_and_selects_what_sqlite_selects(tmp_path, postgres):
    url, reader = postgres
    b, st, a1, _mine = _readers_world(tmp_path)
    lite = BI.SqliteIndex(tmp_path / "i.sqlite")
    for db in (lite, BI.PostgresIndex(url, "store_private")):
        BI.index(FsRemote(b), db, now=T0)
    pg = BI.PostgresIndex(reader, "store_private", create=False)
    assert pg.query("SELECT kind, id FROM records WHERE type = 'run'") == (["kind", "id"], [("bench_vu", "r20261011-170000-0001")])
    assert BI.as_of(pg) == BI.as_of(lite) == "2026-10-11T18:00:00Z"
    for facets in ({"where": {"x": "1"}}, {"labels": [("verified", "accepted")]}, {"tool": "bench_vu"}, {"kind": "proof/v1"}):
        assert BI.select_artifacts(pg, **facets) == BI.select_artifacts(lite, **facets), facets
    assert BI.not_indexed(pg, [a1, "art:" + "0" * 64]) == BI.not_indexed(lite, [a1, "art:" + "0" * 64]) == ["art:" + "0" * 64]
    with pytest.raises(RuntimeError, match="read-only"):
        pg.query("DELETE FROM records RETURNING id")
