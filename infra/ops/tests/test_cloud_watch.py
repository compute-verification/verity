"""The cloud watch (INF-73, infra/control/cloud_watch.py) posts an anomaly once when it opens and once when it clears, reads each
control-pod loop's success marker, sweeps R2 a bounded number of pages a pass, and compares spend with budgets.toml's lines."""
import calendar
import copy
import importlib.util
import json
import os
import re
import sys
import tomllib
import urllib.parse
import xml.sax.saxutils as su
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "infra" / "control"
_spec = importlib.util.spec_from_file_location("cloud_watch", CONTROL / "cloud_watch.py")
W = sys.modules["cloud_watch"] = importlib.util.module_from_spec(_spec)     # dataclasses look their module up there
_spec.loader.exec_module(W)
REAL = tomllib.loads((CONTROL / "cloud_watch.toml").read_text())
T0 = float(calendar.timegm((2026, 10, 18, 12, 0, 0)))


def config(tmp_path, **over) -> dict:
    """The checked-in config with every path in tmp_path and no loops or endpoints, unless a test adds them."""
    cfg = copy.deepcopy(REAL)
    cfg.update(loop=[], http=[], state=str(tmp_path / "state.json"))
    cfg["spend"].update(budgets=[str(tmp_path / "budgets.toml")], guard_state=str(tmp_path / "guard.json"),
                        steward_log=str(tmp_path / "watch.log"), gap=[])
    cfg["nebius"]["machines_d"] = [str(tmp_path / "machines.d")]
    cfg.update(over)
    return cfg


class Posts:
    def __init__(self, refuse: int = 0):
        self.sent, self.refuse = [], refuse

    def __call__(self, cfg, env, text, thread, dry, out=None):
        if self.refuse:
            self.refuse -= 1
            return None
        self.sent.append((text, thread))
        return f"{len(self.sent)}.0"


@pytest.fixture
def posts(monkeypatch):
    p = Posts()
    monkeypatch.setattr(W, "post", p)
    return p


def ctx(cfg, state, now, env=None, **kw):
    return W.Ctx(cfg, env or {}, state, now, **kw)


def log_line(path: Path, t: float, rest: str = "pass rc=0 abc") -> None:
    with open(path, "a") as f:
        f.write(f"{W.iso(t)} {rest}\n")


def store_index_loop(tmp_path) -> dict:
    return {"name": "Store index", "started_by": "/data/infra/store_index.sh", "file": str(tmp_path / "store-index.log"),
            "pattern": r"^(\S+Z) pass rc=0 ", "period_s": 300}


def test_an_anomaly_posts_once_when_it_opens_and_once_when_it_clears(tmp_path, posts):
    cfg, state = config(tmp_path, loop=[store_index_loop(tmp_path)]), {}
    log = tmp_path / "store-index.log"
    log_line(log, T0 - 100)
    log_line(log, T0 - 50, "pass rc=1 abc")
    W.one_pass(ctx(cfg, state, T0), out=open(os.devnull, "w"))
    assert posts.sent == [], "100 s since its last success is inside twice its period"
    W.one_pass(ctx(cfg, state, T0 + 600), out=open(os.devnull, "w"))
    (text, thread), = posts.sent
    assert thread is None and f"<!subteam^{cfg['mention']}>" in text and "*Store index* has not succeeded" in text
    assert "last success 2026-10-18T11:58:20Z" in text
    W.one_pass(ctx(cfg, state, T0 + 1500), out=open(os.devnull, "w"))
    assert len(posts.sent) == 1, "an open anomaly posts once"
    log_line(log, T0 + 1600)
    W.one_pass(ctx(cfg, state, T0 + 1700), out=open(os.devnull, "w"))
    assert len(posts.sent) == 1, "gone one pass is not yet cleared (clear_passes 2)"
    W.one_pass(ctx(cfg, state, T0 + 1800), out=open(os.devnull, "w"))
    text, thread = posts.sent[1]
    assert thread == "1.0" and text.startswith("*Cloud watch, cleared* after 20 min: *Store index*")
    assert state["open"] == {}
    W.one_pass(ctx(cfg, state, T0 + 1900), out=open(os.devnull, "w"))
    assert len(posts.sent) == 2


def test_a_post_slack_refuses_is_tried_again_next_pass(tmp_path, monkeypatch):
    posts = Posts(refuse=1)
    monkeypatch.setattr(W, "post", posts)
    cfg, state = config(tmp_path, loop=[store_index_loop(tmp_path)]), {}
    log_line(tmp_path / "store-index.log", T0 - 1000)
    W.one_pass(ctx(cfg, state, T0))
    assert posts.sent == [] and not state["open"]["loop|Store index"]["posted"]
    W.one_pass(ctx(cfg, state, T0 + 60))
    assert len(posts.sent) == 1 and state["open"]["loop|Store index"]["ts"] == "1.0"


def test_each_kind_of_success_marker(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"t": W.iso(T0 - 30)}))
    (tmp_path / "guard.json").write_text(json.dumps({"last_ok_poll": T0 - 40.5}))
    (tmp_path / "seen.json").write_text("{}")
    os.utime(tmp_path / "seen.json", (T0 - 50, T0 - 50))
    (tmp_path / "steward.log").write_text("23:58Z pass 2656: 3 open, 1 stale\n23:59Z ALIVE hash (via=report)\n")
    (tmp_path / "console.log").write_text("starting\n")
    (tmp_path / "console.log.1").write_text(f'{{"at": "{W.iso(T0 - 70)}", "group": "store", "seconds": 1150.0}}\nTraceback\n')
    mark = lambda **kw: W.marker_time(kw, T0)  # noqa: E731
    assert mark(file=str(tmp_path / "status.json"), json="t") == T0 - 30
    assert mark(file=str(tmp_path / "guard.json"), json="last_ok_poll") == T0 - 40.5
    assert mark(file=str(tmp_path / "seen.json")) == T0 - 50
    assert mark(file=str(tmp_path / "steward.log"), pattern=r"^(\d\d:\d\dZ) pass \d+:") == T0 - 12 * 3600 - 2 * 60, \
        "a time of day later than now is yesterday's"
    assert mark(file=str(tmp_path / "console.log"), rotated=[str(tmp_path / "console.log.1")],
                pattern=r'^\{"at": "(\S+Z)", "group": "store"') == T0 - 70, "a just-rotated log's success is in its .1"


def test_a_missing_or_silent_marker_is_an_anomaly_naming_its_file(tmp_path):
    loop = store_index_loop(tmp_path)
    rep = W.Report()
    W.check_loops(ctx(config(tmp_path, loop=[loop]), {}, T0), rep)
    (f,) = rep.findings
    assert f.key == "loop|Store index" and "store-index.log` is missing" in f.text
    (tmp_path / "store-index.log").write_text("2026-10-18T06:45:00Z pass rc=1 abc\n")
    rep = W.Report()
    W.check_loops(ctx(config(tmp_path, loop=[loop]), {}, T0), rep)
    assert "no success line in its log" in rep.findings[0].text


def test_tail_match_reads_back_across_chunks(tmp_path):
    p = tmp_path / "big.log"
    p.write_text("2026-10-18T06:45:00Z pass rc=0 a\n" + "noise line\n" * 5000 + "2026-10-18T11:00:00Z pass rc=1 a\n")
    assert W.tail_match(p, re.compile(r"^(\S+Z) pass rc=0 "), chunk=1000) == "2026-10-18T06:45:00Z"


class FakeBucket:
    """A ListObjectsV2 endpoint over `objects` (key -> size) that answers at most `page` keys after `start-after`; `modified`
    gives a key's LastModified."""

    def __init__(self, objects: dict[str, int], page: int = 2, modified: dict[str, float] | None = None):
        self.objects, self.page, self.calls, self.modified = objects, page, [], modified or {}

    def __call__(self, method, url, headers=None, body=None, timeout=30):
        assert method == "GET" and headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=dummy-id/")
        self.calls.append(url)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.split("?", 1)[1]).items()}
        keys = sorted(k for k in self.objects if k > q.get("start-after", ""))
        got, more = keys[:self.page], len(keys) > self.page
        items = "".join(f"<Contents><Key>{su.escape(k)}</Key><Size>{self.objects[k]}</Size>"
                        + (f"<LastModified>{W.iso(self.modified[k])[:-1]}.000Z</LastModified>" if k in self.modified else "")
                        + "</Contents>" for k in got)
        return 200, (f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{items}'
                     f"<IsTruncated>{'true' if more else 'false'}</IsTruncated></ListBucketResult>").encode()


R2_ENV = {"R2_BUCKET": "dummy-bucket", "R2_ENDPOINT": "https://dummy.r2.example", "AWS_ACCESS_KEY_ID": "dummy-id",
          "AWS_SECRET_ACCESS_KEY": "dummy"}


def r2_pass(tmp_path, state, now, bucket, page_cap=2):
    cfg = config(tmp_path)
    cfg["r2"] = {**cfg["r2"], "page_cap": page_cap, "bucket": cfg["r2"]["bucket"][:1]}
    rep = W.Report()
    W.check_r2(ctx(cfg, state, now, R2_ENV, http=bucket, clock=lambda: now), rep)
    return rep


def test_r2_sweeps_a_bounded_number_of_pages_a_pass_and_names_the_biggest_writer(tmp_path):
    gib = W.GIB
    bucket = FakeBucket({"attempts/a": 1, "attempts/b": 2, "objects/x": gib, "objects/y": gib, "labels/l": 5})
    state = {}
    rep = r2_pass(tmp_path, state, T0, bucket)
    assert len(bucket.calls) == 2 and "first sweep in progress, 4 objects listed so far" in rep.lines[0]
    assert "start-after=attempts%2Fb" in bucket.calls[1]
    rep = r2_pass(tmp_path, state, T0 + 900, bucket)
    (sweep,) = state["r2"]["store"]["sweeps"]
    assert sweep["prefixes"] == {"attempts/": [3, 2], "labels/": [5, 1], "objects/": [2 * gib, 2]}
    assert "2.0 GiB in 5 objects" in rep.lines[0] and not rep.findings
    bucket.objects.update({"objects/z": 300 * gib, "attempts/c": gib})
    for i in range(3):
        rep = r2_pass(tmp_path, state, T0 + 86400 + 900 * i, bucket)
    assert len(state["r2"]["store"]["sweeps"]) == 2
    (f,) = rep.findings
    assert f.key == "r2|store|growth" and "over 200 GiB/day" in f.text and "Biggest writer: `objects/`" in f.text
    assert state["r2"]["store"]["list_ops"]["2026-10"] == 2 + 1 + 2 + 2 + 2, "one class-A op a page"


def test_a_bucket_with_no_new_object_within_its_stale_after_h_is_an_anomaly(tmp_path):
    cfg = config(tmp_path)
    cfg["r2"] = {**cfg["r2"], "bucket": [b for b in cfg["r2"]["bucket"] if b["label"] == "lean cache"]}
    env = {**R2_ENV, "LEAN_CACHE_BUCKET": "dummy-lean"}
    bucket = FakeBucket({"a.sha256.json": 10, "b.tar": 20}, page=1000, modified={"a.sha256.json": T0 - 12 * 86400, "b.tar": T0 - 13 * 86400})
    state, rep = {}, W.Report()
    W.check_r2(ctx(cfg, state, T0, env, http=bucket, clock=lambda: T0), rep)
    (f,) = rep.findings
    assert f.key == "r2|lean cache|stale" and "no new object for 12.0 d" in f.text and "`vy-lean-cache-daily`" in f.text
    assert "newest write 2026-10-06T12:00:00Z" in rep.lines[0]
    bucket.objects["c.sha256.json"], bucket.modified["c.sha256.json"] = 5, T0 + 600
    rep = W.Report()
    W.check_r2(ctx(cfg, state, T0 + 900, env, http=bucket, clock=lambda: T0 + 900), rep)
    assert not [f for f in rep.findings if f.key.endswith("|stale")]


def test_jobs_queued_past_a_day_are_one_anomaly_by_kind(tmp_path):
    jobs = [{"id": 4, "kind": "audit-main", "enqueued_at": W.iso(T0 - 3600), "not_before": None},
            {"id": 3, "kind": "check", "enqueued_at": W.iso(T0 - 3 * 86400), "not_before": W.iso(T0 + 3600)},
            {"id": 2, "kind": "audit-main", "enqueued_at": W.iso(T0 - 2 * 86400), "not_before": W.iso(T0 - 2 * 86400)},
            {"id": 1, "kind": "audit-main", "enqueued_at": W.iso(T0 - 3 * 86400), "not_before": None}]
    seen = []

    def http(method, url, headers=None, body=None, timeout=30):
        seen.append((url, headers["Authorization"]))
        return 200, json.dumps({"jobs": jobs}).encode()

    rep = W.Report()
    W.check_jobs(ctx(config(tmp_path), {}, T0, {"JOBS_TOKEN": "dummy"}, http=http), rep)
    assert seen == [("https://website-docs-sage.vercel.app/api/jobs?state=queued&limit=200", "Bearer dummy")]
    (f,) = rep.findings
    assert f.key == "jobs|stale" and f.text.startswith("*2 jobs queued over 24 h") and "(2 `audit-main`); oldest #1 since 2026-10-15T12:00:00Z" in f.text
    assert rep.lines == ["Job queue: 4 queued, oldest waiting 72.0 h"]
    rep = W.Report()
    W.check_jobs(ctx(config(tmp_path), {}, T0), rep)
    assert rep.gaps == ["needs JOBS_TOKEN: the Job queue's queued jobs no worker claims"] and not rep.findings


def test_neon_reads_through_psql_with_the_url_only_in_the_environment(tmp_path, monkeypatch):
    seen = {}

    def run(argv, env, stdin=""):
        seen.update(argv=argv, env=env, sql=stdin)
        return 0, f"{9 * W.GIB}\t90\t100\t{T0 - 7200}\t0\n", ""

    monkeypatch.setattr(W.shutil, "which", lambda name, path=None: "/usr/bin/psql")
    url = "postgresql://indexer:" + "pw-dummy" + "@ep-x.neon.example/store?sslmode=require&channel_binding=require"
    rep = W.Report()
    W.check_neon(ctx(config(tmp_path), {}, T0, {"STORE_INDEXER_DATABASE_URL": url}, run=run), rep)
    assert url not in " ".join(seen["argv"]) and "pw-dummy" not in " ".join(seen["argv"])
    assert seen["env"]["PGPASSWORD"] == "pw-dummy" and seen["env"]["PGCHANNELBINDING"] == "require"
    assert "store_private.listings" in seen["sql"]
    assert sorted(f.key for f in rep.findings) == ["neon|connections", "neon|size", "neon|store_private|stale"]
    assert "schemaname = 'store_public'" in seen["sql"]
    assert rep.lines == [("Neon: 9.00 GiB, 90 of 100 connections; store index last write `store_private` 2.0 h ago; "
                          "`store_public` has no tables (never built)")]


def test_a_neon_failure_names_no_host_address_or_user(tmp_path, monkeypatch):
    err = ('psql: error: connection to server at "ep-x.neon.example" (34.205.173.201), port 5432 failed: '
           "ERROR:  password authentication failed for user 'indexer'\n")
    monkeypatch.setattr(W.shutil, "which", lambda name, path=None: "/usr/bin/psql")
    url = "postgresql://indexer:" + "pw-dummy" + "@ep-x.neon.example/store?sslmode=require"
    with pytest.raises(W.Unreadable) as e:
        W.check_neon(ctx(config(tmp_path), {}, T0, {"STORE_INDEXER_DATABASE_URL": url}, run=lambda *a: (2, "", err)), W.Report())
    msg = str(e.value)
    assert "password authentication failed" in msg
    assert not any(s in msg for s in ("ep-x.neon.example", "34.205", "indexer", "pw-dummy")), msg


BUDGETS = """
[budgets]
"vy-cpu-" = { cap_usd = 100, expires = "2026-10-12T17:00Z" }
"vy-dc-" = { cap_usd = 100, expires = "2026-10-19T06:00Z" }
"vy-nebius-cpu-" = { cap_usd_per_day = 500, expires = "2026-10-21T21:00Z" }
"vy-nebius-" = { cap_usd = 8500, expires = "2026-10-18T10:00Z" }
"vy-glm53-" = { cap_usd = 1000, expires = "2026-10-24T21:00Z" }
"""


def test_runpod_lines_near_their_cap_and_a_short_runway_are_anomalies(tmp_path):
    (tmp_path / "budgets.toml").write_text(BUDGETS)
    month = float(calendar.timegm((2026, 10, 1, 0, 0, 0)))
    pods = {"a": {"first_seen": month - 10 * 3600, "last_seen": month + 10 * 3600, "spent": 20.0},     # half of it in October
            "b": {"first_seen": month + 3600, "last_seen": month + 7200, "spent": 3.0},
            "c": {"first_seen": month - 7200, "last_seen": month - 3600, "spent": 50.0}}               # all of it in September
    (tmp_path / "guard.json").write_text(json.dumps({
        "balance": 30.0, "rate_now": 0.0, "budgets": {"balance_floor": 25}, "pods": pods,
        "lines": {"vy-dc-": {"spent": 85.0}, "vy-cpu-": {"spent": 99.0}, "vy-glm53-": {"spent": 1.0}}}))
    (tmp_path / "watch.log").write_text("19:31Z ACCOUNT $30.00 at $0.25/h, runway 120.0h; lane pods $0.00/h (0)\n19:31Z pass 1\n")
    rep = W.Report()
    W.spend_runpod(ctx(config(tmp_path), {}, T0), rep, tomllib.loads(BUDGETS)["budgets"])
    assert sorted(f.key for f in rep.findings) == ["spend|runpod|runway", "spend|runpod|vy-dc-"], "vy-cpu-'s line has expired"
    hours = (T0 - month) / 3600
    assert f"this month about {W.usd(10 + 3 + 0.25 * hours)} ($13.00 of guarded pods" in rep.lines[0]
    assert "the top-up failed*, and the balance reaches the guard's $25.00 floor in 20 h at $0.25/h" in next(
        f.text for f in rep.findings if f.key == "spend|runpod|runway")


def test_a_short_runway_above_the_auto_top_up_is_not_an_anomaly(tmp_path):
    (tmp_path / "budgets.toml").write_text(BUDGETS)
    (tmp_path / "guard.json").write_text(json.dumps({"balance": 125.0, "rate_now": 42.0, "budgets": {"balance_floor": 25}}))
    (tmp_path / "watch.log").write_text("16:41Z ACCOUNT $125.76 at $42.83/h, runway 2.9h; lane pods $0.00/h (0)\n")
    rep = W.Report()
    W.spend_runpod(ctx(config(tmp_path), {}, T0), rep, tomllib.loads(BUDGETS)["budgets"])
    assert not rep.findings, "RunPod tops the balance up at $100, so 2 h of runway at $125 is no failure"


def test_a_nebius_vm_on_no_line_past_its_line_or_over_a_daily_cap_is_an_anomaly(tmp_path):
    (tmp_path / "budgets.toml").write_text(BUDGETS)
    md = tmp_path / "machines.d"
    md.mkdir()
    for name, rate in {"vy-nebius-1": 14.4, "vy-nebius-cpu-1": 12.0, "vy-nebius-cpu-2": 10.0, "vy-glm53-eu": 43.2}.items():
        (md / f"{name}.toml").write_text(f"rate_usd_per_hour = {rate}\n")
    vms = [{"name": n, "state": s} for n, s in [("vy-nebius-1", "RUNNING"), ("vy-nebius-cpu-1", "RUNNING"),
                                                 ("vy-nebius-cpu-2", "RUNNING"), ("vy-glm53-eu", "RUNNING"),
                                                 ("vy-glm53-h8", "STOPPED"), ("stray", "RUNNING")]]
    env = {k: "dummy" for k in REAL["nebius"]["key_env"]}
    rep = W.Report()
    run = lambda argv, env, stdin="": (0, json.dumps(vms), "")  # noqa: E731
    W.spend_nebius(ctx(config(tmp_path), {}, T0, env, run=run), rep, tomllib.loads(BUDGETS)["budgets"])
    assert sorted(f.key for f in rep.findings) == ["spend|nebius|stray|uncovered", "spend|nebius|vy-glm53-|cap",
                                                   "spend|nebius|vy-nebius-1|expired", "spend|nebius|vy-nebius-cpu-|day"]
    assert rep.lines[0].startswith("Nebius: 5 VMs running, $79.60/h known") and "no rate registered for `stray`" in rep.lines[0]


def test_a_missing_credential_is_a_gap_in_the_digest_and_never_an_anomaly(tmp_path, posts):
    cfg, state = config(tmp_path), {}
    W.one_pass(ctx(cfg, state, T0 + 4 * 3600), out=open(os.devnull, "w"))
    (digest, thread), = posts.sent
    assert thread is None and digest.startswith("*Cloud health and spend, 2026-10-18T16:00Z*") and "Nothing open." in digest
    gaps = [ln for ln in digest.split("*Unmonitored* (")[1].splitlines()[1:] if ln.startswith("• needs ")]
    assert len(gaps) == len(set(gaps))
    for need in ("needs R2_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_ENDPOINT: R2 bucket `store`",
                 "needs STORE_INDEXER_DATABASE_URL", "needs GH_TOKEN", "needs NEBIUS_SA_ID, NEBIUS_SA_PUBLIC_KEY_ID, NEBIUS_SA_PRIVATE_KEY_B64", "needs budgets.toml"):
        assert any(need in g for g in gaps), need
    assert state["open"] == {}


def test_the_digest_posts_once_a_day_from_its_hour(tmp_path, posts):
    cfg, state = config(tmp_path), {}
    for h in (14, 15, 16, 23, 24 + 14, 24 + 15):
        W.one_pass(ctx(cfg, state, T0 + (h - 12) * 3600), out=open(os.devnull, "w"))
    assert [t.split("\n")[0] for t, _ in posts.sent] == [
        "*Cloud health and spend, 2026-10-18T15:00Z* (`infra/control/cloud_watch.py`, INF-73)",
        "*Cloud health and spend, 2026-10-19T15:00Z* (`infra/control/cloud_watch.py`, INF-73)"]


def test_a_failing_check_opens_unreadable_after_its_passes_and_keeps_what_it_had_open(tmp_path, posts, monkeypatch):
    cfg, state = config(tmp_path), {"digest_day": "2026-10-18"}
    state["open"] = {"github|core": {"since": T0, "text": "low", "ts": "9.0", "posted": True}}

    def down(ctx, rep):
        raise W.Unreadable("GitHub rate_limit answered HTTP 502")

    monkeypatch.setitem(W.CHECKS, "github", down)
    for i in range(3):
        W.one_pass(ctx(cfg, state, T0 + 900 * i), out=open(os.devnull, "w"))
    (text, thread), = posts.sent
    assert "can't read `github`* for 3 passes: GitHub rate_limit answered HTTP 502" in text
    assert "github|core" in state["open"], "what a check couldn't read hasn't cleared"


def test_dry_run_prints_and_posts_nothing(tmp_path, capsys):
    W.post(REAL, {"SLACK_BOT_TOKEN": "dummy"}, "*Cloud watch:* hello", None, True)
    assert capsys.readouterr().out == "would post:\n*Cloud watch:* hello\n\n"


def test_sigv4_matches_the_aws_list_objects_example():
    """docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html, "GET Bucket (List Objects)"."""
    h = W.sign("GET", "examplebucket.s3.amazonaws.com", "/", {"max-keys": "2", "prefix": "J"}, "AKIAIOSFODNN7EXAMPLE",
               "wJalrXUtnFEMI/K7MDENG/" "bPxRfiCYEXAMPLEKEY", "20130524T000000Z", region="us-east-1")
    assert h["Authorization"].endswith("Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")


def test_the_config_names_the_registry_channel_and_every_loop_post_start_starts():
    reg = json.loads((ROOT / ".agents/skills/using-slack/registry.json").read_text())
    assert REAL["channel"] == reg["channels"]["agent-alerts"]["id"]
    assert REAL["mention"] == next(h["usergroup"] for h in reg["handles"] if h["handle"] == "infra")
    started = re.findall(r"setsid -f (/data/\S+)", (CONTROL / "post_start.sh").read_text())
    watched = " ".join(loop["started_by"] for loop in REAL["loop"])
    unwatched = [s for s in started if s not in watched and s != "/data/infra/cloud_watch.sh"]   # it can't watch itself
    assert unwatched == [] and "research pods guard" in watched, f"add a [[loop]] to cloud_watch.toml for {unwatched}"
