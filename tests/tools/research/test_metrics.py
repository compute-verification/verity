"""`research metrics show` against a fake route: each declared metric's line, one no run has recorded, a refused row and
why the site couldn't read something; filtering by id and owner; an undeclared id or a refusal as one line; which
credential goes where: a site key only to the console's route, an older token never, the identity otherwise. And `research
metrics send`, with a stand-in for the site: a reading's shape, the panels key read from its file and left out of a dry run,
sent only over https to the site's own host, a reading the door would refuse sent not at all, and a failure on the network
one line, never raised."""
from __future__ import annotations

import http.client
import io
import json
import urllib.error
import urllib.request

import pytest

from research import metrics, slack
from research.jobs import client
from research.slack import RouteError

COST = {"id": "vllm/serving-cost-glm53", "title": "GLM-5.3's serving cost over stock vLLM", "owner": "vllm",
        "unit": "percent of stock's serving compute", "better": "lower", "command": "bash benchmarks/serve_tax/run.sh",
        "every": "7d", "floor": None, "pastFloor": None, "stale": False, "ageMs": 2 * 86_400_000,
        "change7d": -0.4, "change30d": None,
        "latest": {"value": 2.0, "ci95": [1.8, 2.2], "verdict": "meets", "target": 2.05, "run": "r20261109-000000-abcd",
                   "commit": "d" * 40}}
LAND = {"id": "ci/mark-to-main", "title": "Ready mark to main", "owner": "ci", "unit": "min", "better": "lower",
        "command": "research run -- python tools/ci/landing.py", "every": "12h", "floor": 90, "pastFloor": None,
        "stale": False, "ageMs": None, "change7d": None, "change30d": None, "latest": None}


class Fake:
    def __init__(self, report: dict | None = None, refuse: RouteError | None = None):
        self.report, self.refuse = report, refuse

    def call(self) -> dict:
        if self.refuse:
            raise self.refuse
        return json.loads(json.dumps(self.report))


REPORT = {"sha": "s1", "metrics": [COST, LAND], "refused": [{"id": "ci/a", "why": "better is neither lower nor higher"}],
          "why": None}


def test_show_prints_each_metric_its_standing_and_what_was_refused(capsys):
    assert metrics.main(["show"], Fake(REPORT)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "vllm/serving-cost-glm53  @vllm  GLM-5.3's serving cost over stock vLLM",
        "  2 [1.8, 2.2] percent of stock's serving compute  7d -0.4  30d —  age 2 d (every 7d)  meets (target 2.05)",
        "  run r20261109-000000-abcd at dddddddddddd",
        "ci/mark-to-main  @ci  Ready mark to main",
        "  not recorded yet; measured by: research run -- python tools/ci/landing.py",
        "refused: ci/a: better is neither lower nor higher",
    ]


def test_show_filters_by_id_and_owner_and_prints_json(capsys):
    assert metrics.main(["show", "--owner", "ci", "--json"], Fake(REPORT)) == 0
    assert [m["id"] for m in json.loads(capsys.readouterr().out)["metrics"]] == ["ci/mark-to-main"]
    assert metrics.main(["show", "vllm/serving-cost-glm53"], Fake(REPORT)) == 0
    assert capsys.readouterr().out.startswith("vllm/serving-cost-glm53  @vllm")


def test_an_undeclared_id_a_refusal_and_the_sites_note_are_one_line_each(capsys):
    assert metrics.main(["show", "vllm/nope"], Fake(REPORT)) == 1
    assert "not declared in any metrics.toml on main: vllm/nope" in capsys.readouterr().out
    assert metrics.main(["show"], Fake(refuse=RouteError(f"GET {metrics.METRICS}: 401 invalid_identity", 401))) == 1
    assert capsys.readouterr().out == f"research metrics: GET {metrics.METRICS}: 401 invalid_identity\n"
    assert metrics.main(["show"], Fake({"sha": None, "metrics": [], "refused": [], "why": "not on main yet"})) == 0
    assert capsys.readouterr().out.splitlines() == ["no metric is declared yet", "note: not on main yet"]


def test_a_site_key_is_sent_as_is_and_no_identity_is_minted(monkeypatch):
    sent = []
    monkeypatch.setattr(metrics, "http_json", lambda method, url, data, headers: sent.append((method, url, headers)) or REPORT)
    monkeypatch.setattr(metrics, "_Identity", lambda aud: (_ for _ in ()).throw(AssertionError("minted an identity")))
    assert metrics.Route(key="vsk_test").call()["sha"] == "s1"
    assert sent == [("GET", metrics.METRICS, {"Authorization": "Bearer vsk_test"})]


def test_only_a_site_key_is_taken_and_an_older_token_leaves_the_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "TOKEN_FILE", tmp_path / "token")
    monkeypatch.delenv(client.TOKEN_ENV, raising=False)
    assert metrics._key() is None
    monkeypatch.setenv(client.TOKEN_ENV, "a1b2c3d4older")
    assert metrics._key() is None
    monkeypatch.setenv(client.TOKEN_ENV, "vsk_live")
    assert metrics._key() == "vsk_live"
    monkeypatch.delenv(client.TOKEN_ENV)
    (tmp_path / "token").write_text("vsk_file\n")
    (tmp_path / "token").chmod(0o644)
    with pytest.raises(SystemExit, match="readable by others"):
        metrics._key()
    (tmp_path / "token").chmod(0o600)
    assert metrics._key() == "vsk_file"


class _Identity:
    def __init__(self, audience: str) -> None:
        self.audience = audience

    def send(self, get):
        return get("h.p.s")


def test_another_url_gets_the_identity_and_never_the_key(monkeypatch):
    sent = []
    monkeypatch.setattr(metrics, "http_json", lambda method, url, data, headers: sent.append((url, headers)) or REPORT)
    monkeypatch.setattr(metrics, "_Identity", _Identity)
    assert metrics.Route("http://example.invalid/collect", key="vsk_test").call()["sha"] == "s1"
    assert sent == [("http://example.invalid/collect", {"Authorization": "Bearer h.p.s"})]


def test_a_refused_key_is_named_as_the_key_and_the_url_as_the_one_called(monkeypatch):
    def refuse(method, url, data, headers):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", None, io.BytesIO(b'{"error": "invalid_key"}'))
    monkeypatch.setattr(metrics, "http_json", refuse)
    with pytest.raises(RouteError, match=r"rejected the site key in \$JOBS_TOKEN .*\(HTTP 401 invalid_key\)"):
        metrics.Route(key="vsk_dead").call()
    monkeypatch.setattr(metrics, "_Identity", _Identity)
    with pytest.raises(RouteError, match=r"^GET http://example.invalid/collect: the route rejected this agent's Cursor identity"):
        metrics.Route("http://example.invalid/collect").call()


T = "2026-11-14T05:00:00Z"


def test_a_reading_goes_as_one_post_naming_its_command_and_commit_with_the_key_from_its_file(tmp_path):
    key = tmp_path / "panels.key"
    key.write_text("k-not-a-real-key\n")
    sent = []
    r = metrics.reading("infra/x", 133.56, time=T, target=150, member=None)
    assert r == {"metric": "infra/x", "value": 133.56, "mark": "measured", "time": T, "target": 150}
    cmd = "infra/control/loop_metrics.py"
    assert metrics.send([r], command=cmd, commit="c" * 40, key=key, post=lambda *a: sent.append(a)) is None
    assert sent == [("POST", metrics.METRICS, {"readings": [{**r, "command": cmd, "commit": "c" * 40}]},
                     "k-not-a-real-key", f"POST {metrics.METRICS}")]
    assert metrics.send([r], command=cmd, commit=None, key=key, post=lambda *a: "POST x: HTTP 401 invalid_key") == (
        "POST x: HTTP 401 invalid_key")


def test_a_reading_the_door_would_refuse_and_a_missing_key_or_plain_http_send_nothing(tmp_path):
    def post(*a):
        raise AssertionError("nothing is sent")
    key, what = tmp_path / "panels.key", f"POST {metrics.METRICS}"
    ok = metrics.reading("infra/x", 1, time=T)
    assert metrics.send([ok], command="l", commit=None, key=key, post=post) == (
        f"{what}: no panels key at {key} (FileNotFoundError): nothing sent")
    key.write_text("k")
    assert metrics.send([], command="l", commit=None, key=key, post=post) == f"{what}: no reading to send"
    bad = [{**ok, "value": None}, {**ok, "mark": "guessed"}, {**ok, "value": "3"}, {**ok, "time": ""}, {**ok, "colour": 1}]
    assert metrics.send(bad, command="l", commit=None, key=key, post=post) == (
        f"{what}: nothing sent: infra/x: a null value needs its why; infra/x: mark is 'guessed', not one of measured, "
        "modeled, partial; infra/x: value is '3', not a number or null; infra/x: a reading needs its metric and the time its "
        "number was read; infra/x: unknown key colour, which the door refuses")


#: the keys the door takes: the site's lib/console/metrics/readings.ts, KEYS
DOOR_KEYS = {"metric", "time", "value", "mark", "unit", "ci95", "target", "verdict", "why", "period", "over", "member", "run",
             "commit", "command"}


def test_a_reading_holds_only_the_keys_the_door_takes_and_names_its_command_or_run(tmp_path):
    """The door refuses a whole request for one reading with a key it doesn't know (`loop` among them), one naming neither its
    run nor its command, or a commit that isn't hex: the sender refuses each first, with nothing sent."""
    def post(*a):
        raise AssertionError("nothing is sent")
    assert set(metrics.KEYS) == DOOR_KEYS
    key, what = tmp_path / "panels.key", f"POST {metrics.METRICS}"
    key.write_text("k")
    ok = metrics.reading("infra/x", 1, time=T)
    assert metrics.send([{**ok, "loop": "loop-metrics"}], command="c", commit=None, key=key, post=post) == (
        f"{what}: nothing sent: infra/x: unknown key loop, which the door refuses")
    assert metrics.send([ok], command=None, commit=None, key=key, post=post) == (
        f"{what}: nothing sent: infra/x: names neither the run that recorded it nor the command that computed it")
    assert metrics.send([ok], command="c", commit="not-hex", key=key, post=post) == (
        f"{what}: nothing sent: infra/x: commit is 'not-hex', not a commit's hex")
    sent = []
    assert metrics.send([{**ok, "run": "r1"}], command=None, commit="0123abc", key=key, post=lambda *a: sent.append(a)) is None
    assert set(sent[0][2]["readings"][0]) <= DOOR_KEYS


def test_the_dry_run_prints_the_request_without_the_key_and_sends_nothing(tmp_path, capsys):
    key = tmp_path / "panels.key"
    key.write_text("k-not-a-real-key\n")
    r = metrics.reading("infra/x", None, why="the guard's poll is 12 min old", time=T, target=150)
    assert metrics.send([r], command="infra/control/loop_metrics.py", commit="c1d2e3f", key=key, dry_run=True,
                        post=lambda *a: (_ for _ in ()).throw(AssertionError("a dry run sends nothing"))) is None
    out = capsys.readouterr().out
    assert "k-not-a-real-key" not in out
    head, body = out.split("\n\n", 1)
    assert head.splitlines() == [f"POST {metrics.METRICS}", f"Authorization: Bearer <the panels key in {key}>",
                                 "Content-Type: application/json"]
    assert json.loads(body) == {"readings": [{**r, "command": "infra/control/loop_metrics.py", "commit": "c1d2e3f"}]}


def test_the_command_sends_one_reading_from_its_arguments_or_the_readings_on_stdin(tmp_path, monkeypatch, capsys):
    key = tmp_path / "panels.key"
    key.write_text("k")
    monkeypatch.setattr(metrics, "PANELS_KEY", key)
    sent = []
    monkeypatch.setattr(metrics, "send_json", lambda method, url, body, k, what: sent.append(body) and None)
    assert metrics.main(["send", "--command", "infra/control/store_index.sh", "--commit", "c1d2e3f", "--metric", "infra/s",
                         "--value", "365", "--time", T]) == 0
    assert capsys.readouterr().out == f"research metrics: sent 1 reading(s) to {metrics.METRICS}\n"
    assert sent.pop() == {"readings": [{"metric": "infra/s", "value": 365.0, "mark": "measured", "time": T,
                                        "command": "infra/control/store_index.sh", "commit": "c1d2e3f"}]}
    two = [{"metric": "infra/a", "value": 1, "mark": "measured", "time": T},
           {"metric": "infra/b", "value": None, "why": "failed", "mark": "measured", "time": T}]
    assert metrics.main(["send", "--command", "l"], stdin=io.StringIO(json.dumps(two))) == 0
    assert sent.pop() == {"readings": [{**r, "command": "l"} for r in two]}
    capsys.readouterr()
    assert metrics.main(["send", "--command", "l", "--metric", "infra/s", "--time", T]) == 1
    assert capsys.readouterr().out == f"research metrics: POST {metrics.METRICS}: nothing sent: infra/s: a null value needs its why\n"
    assert metrics.main(["send", "--command", "l"], stdin=io.StringIO("{nope")) == 1
    assert capsys.readouterr().out.startswith("research metrics: stdin is not JSON (")
    assert not sent


def _no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("nothing is sent")
    monkeypatch.setattr(urllib.request, "build_opener", refuse)
    monkeypatch.setattr(metrics, "http_open", refuse, raising=False)


def test_the_panels_key_goes_only_to_the_sites_own_host(tmp_path, monkeypatch, capsys):
    """Whatever `--url` or $FLEET_PANELS_URL says: another host, or the site's over http, is refused with nothing sent."""
    _no_network(monkeypatch)
    key = tmp_path / "panels.key"
    key.write_text("k")
    ok = metrics.reading("infra/x", 1, time=T)
    for url in ("https://elsewhere.invalid/api/metrics", f"http://{metrics.SITE_HOST}/api/metrics",
                f"https://{metrics.SITE_HOST}.elsewhere.invalid/api/metrics"):
        why = (f"{url!r} is not on the site (https://{metrics.SITE_HOST}), the one host the panels key goes to: "
               "nothing sent")
        assert metrics.send([ok], command="l", commit=None, url=url, key=key) == f"POST {url}: {why}"
        assert metrics.send([ok], command="l", commit=None, url=url, key=key, dry_run=True) == f"POST {url}: {why}"
        assert metrics.send_json("PUT", url, {}, "k", "x") == f"x: {why}"
    assert metrics.main(["--url", "https://elsewhere.invalid/api/metrics", "send", "--command", "l", "--metric", "infra/x",
                         "--value", "1", "--time", T]) == 1
    assert "nothing sent" in capsys.readouterr().out


@pytest.mark.parametrize("raised", [
    http.client.IncompleteRead(b""), http.client.BadStatusLine("x"), http.client.LineTooLong("header line"),
    http.client.InvalidURL("bad"), ConnectionResetError(104, "reset"), TimeoutError("timed out"),
    urllib.error.URLError("redirected; refusing to forward the bearer token"),
    urllib.error.HTTPError(metrics.METRICS, 503, "down", None, io.BytesIO(b'{"error": "unavailable"}'))])
def test_the_site_client_never_raises_on_the_network(monkeypatch, raised):
    _no_network(monkeypatch)

    def http_open(*a):
        raise raised
    monkeypatch.setattr(metrics, "http_open", http_open)
    why = metrics.send_json("POST", metrics.METRICS, {}, "k", "w")
    assert why.startswith("w: ") and "\n" not in why
    if isinstance(raised, urllib.error.HTTPError):
        assert why == "w: HTTP 503 unavailable"


def test_a_failure_reading_a_refusals_body_is_one_line_too(monkeypatch):
    _no_network(monkeypatch)

    class Cut(io.BytesIO):
        def read(self, *a):
            raise http.client.IncompleteRead(b"")

    def http_open(*a):
        raise urllib.error.HTTPError(metrics.METRICS, 502, "the site failed", None, Cut())
    monkeypatch.setattr(metrics, "http_open", http_open)
    assert metrics.send_json("POST", metrics.METRICS, {}, "k", "w") == "w: IncompleteRead: IncompleteRead(0 bytes read)"


def test_the_site_client_is_slacks_with_the_bearer_and_its_answer_unread(monkeypatch):
    """One client: `slack.http_open`, which `http_json` reads with and which follows no redirect. The answer isn't read, so a
    2xx with any body or none is taken."""
    assert metrics.http_open is slack.http_open and "http_open" in slack.http_json.__code__.co_names
    _no_network(monkeypatch)
    calls = []

    def http_open(method, url, data, headers):
        calls.append((method, url, json.loads(data), headers))
        return io.BytesIO(b"not json")
    monkeypatch.setattr(metrics, "http_open", http_open)
    assert metrics.send_json("PUT", metrics.METRICS, {"a": 1}, "k", "x") is None
    assert calls == [("PUT", metrics.METRICS, {"a": 1}, {"Authorization": "Bearer k", "Content-Type": "application/json"})]
