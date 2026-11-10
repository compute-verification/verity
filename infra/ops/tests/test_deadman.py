"""The control pod's dead-man's switch (infra/control/deadman): a stale heartbeat, or a stale cloud watch beside a fresh one, is
posted on the first cron call past its limit and again every REPEAT_S, never on a public read, and never for an unreadable Neon."""
import calendar
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("deadman", Path(__file__).resolve().parents[3] / "infra/control/deadman/api/deadman.py")
D = sys.modules["deadman"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
NOW = float(calendar.timegm((2026, 10, 19, 20, 0, 0)))
ENV = {"STORE_PRIVATE_READER_URL": "dummy", "SLACK_BOT_TOKEN": "dummy", "CRON_SECRET": "s"}


def row(age_s, cw_age_s=60):
    t = D.time.strftime("%Y-%m-%d %H:%M:%S+00", D.time.gmtime(NOW - age_s))
    return {"t": t, "body": json.dumps({"cloud_watch_t": D.time.strftime("%Y-%m-%dT%H:%M:%SZ", D.time.gmtime(NOW - cw_age_s))})}


def run(r, cron=True, now=NOW):
    sent = []
    code, out = D.check(ENV, now, cron, read=lambda url, host: r, send=lambda token, text: sent.append(text))
    return code, out, sent


def test_a_fresh_heartbeat_is_ok_and_posts_nothing():
    code, out, sent = run(row(60))
    assert (code, out["ok"], out["age_s"], sent) == (200, True, 60, [])


def test_a_stale_heartbeat_posts_on_the_first_call_past_its_limit_then_every_repeat_s():
    posted = [len(run(row(D.STALE_S + k * D.EVERY_S))[2]) for k in range(0, 40)]
    per_repeat = D.REPEAT_S // D.EVERY_S
    assert posted[0] == 1 and sum(posted[1:per_repeat]) == 0 and posted[per_repeat] == 1
    code, out, sent = run(row(D.STALE_S))
    assert code == 503 and "<!subteam^S0C5APDSYF9>" in sent[0] and "15 min old" in sent[0]


def test_a_public_read_says_stale_but_posts_nothing():
    code, out, sent = run(row(3600), cron=False)
    assert code == 503 and sent == [] and "posted" not in out


def test_a_stale_cloud_watch_beside_a_fresh_heartbeat_is_its_own_finding():
    code, out, sent = run(row(60, cw_age_s=D.CLOUD_WATCH_STALE_S))
    assert code == 503 and "cloud watch" in sent[0]


def test_no_row_is_stale_but_never_posted_and_an_unreadable_neon_is_502():
    assert run(None)[0] == 503 and run(None)[2] == []

    def boom(url, host):
        raise OSError("down")

    code, out = D.check(ENV, NOW, True, read=boom, send=lambda *a: (_ for _ in ()).throw(AssertionError("posted")))
    assert code == 502 and not out["ok"]


def test_only_the_cron_secret_authorizes_the_cron():
    assert D.authorized("Bearer s", "s") and not D.authorized("Bearer t", "s") and not D.authorized(None, "s")
    assert not D.authorized("Bearer ", "")


def test_the_cron_is_scheduled_every_every_s():
    v = json.loads((Path(D.__file__).parents[1] / "vercel.json").read_text())
    assert v["crons"] == [{"path": "/api/deadman", "schedule": f"*/{D.EVERY_S // 60} * * * *"}]
