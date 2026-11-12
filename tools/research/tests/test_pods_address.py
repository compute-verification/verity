"""How `research` reaches a RunPod machine without asking RunPod at every call (pods/runpod.py, remote.pod_target): the address
last read is kept and used while it still takes a connection; a read asks GraphQL when REST gives no answer; and every call sent
to RunPod is one line of the call log, with the counts RunPod says are left. No network: `urlopen` answers from a queue, the
connection probe is a function the test sets, and the clock is the test's."""
from __future__ import annotations

import email.message
import io
import json
import urllib.error

import pytest

from research.pods import guard, runpod

POD = "abc123def45678"
#: the headers as RunPod sends them: REST's one `ratelimit` header, GraphQL's one header a window
REST_LEFT = {"ratelimit": '"minute";r=21;t=51, "hour";r=404;t=2511, "day";r=5129;t=78111',
             "ratelimit-policy": '"minute";q=22;w=60, "hour";q=450;w=3600, "day";q=5400;w=86400'}
REST_SPENT = {"ratelimit": '"minute";r=20;t=50, "hour";r=403;t=2510, "day";r=0;t=78110', "retry-after": "78110"}
GRAPHQL_LEFT = {"x-ratelimit-remaining-minute": "179", "x-ratelimit-remaining-hour": "3574", "x-ratelimit-remaining-day": "43019"}
REFUSED = (429, {"error": "rate limit exceeded for the day window", "status": 429}, REST_SPENT)
REST_POD = {"id": POD, "name": "vy-x-veritor-campaign", "desiredStatus": "RUNNING", "costPerHr": 0.16, "publicIp": "203.0.113.7",
            "portMappings": {"22": 40123}, "machine": {}, "machineId": "m1", "createdAt": "2026-10-30 13:58:56.573 +0000 UTC",
            "env": {"PUBLIC_KEY": "ssh-ed25519 dummy"}}
GRAPHQL_POD = {"id": POD, "name": "vy-x-veritor-campaign", "desiredStatus": "RUNNING", "costPerHr": 0.16, "machineId": "m1",
               "createdAt": "2026-10-30T13:58:56.573Z", "machine": {"gpuTypeId": "unknown"},
               "runtime": {"ports": [{"ip": "100.65.0.9", "isIpPublic": False, "privatePort": 19123, "publicPort": 60302, "type": "http"},
                                     {"ip": "203.0.113.7", "isIpPublic": True, "privatePort": 22, "publicPort": 40123, "type": "tcp"}]}}
GRAPHQL_STOPPED = {"id": "zzz999yyy88877", "name": "vy-old-veritor-campaign", "desiredStatus": "EXITED", "costPerHr": 0.03,
                   "machineId": "m2", "createdAt": "2026-09-21T17:53:28.637Z", "machine": {"gpuTypeId": "unknown"}, "runtime": None}
AS_REST = {"id": POD, "name": "vy-x-veritor-campaign", "desiredStatus": "RUNNING", "costPerHr": 0.16, "machineId": "m1",
           "createdAt": "2026-10-30T13:58:56.573Z", "machine": {"gpuTypeId": "unknown"}, "publicIp": "203.0.113.7",
           "portMappings": {"22": 40123}}
T0 = 1794449889.0       # 2026-11-12T02:18:09Z


class Answer:
    def __init__(self, status: int, body, headers: dict):
        self.status, self.headers, self._raw = status, _headers(headers), json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._raw


def _headers(values: dict) -> email.message.Message:
    m = email.message.Message()
    for k, v in values.items():
        m[k] = v
    return m


class Wire:
    """`urlopen`'s stand-in. REST and GraphQL each answer from their own queue: (status, body, headers), or an exception to raise.
    `sent` is every request: (api, method, path) for REST, (api, method, query) for GraphQL."""

    def __init__(self):
        self.rest: list = []
        self.graphql: list = []
        self.sent: list[tuple[str, str, str]] = []

    def __call__(self, request, timeout=0):
        api = "graphql" if request.full_url.startswith("https://api.invalid/graphql") else "rest"
        what = json.loads(request.data)["query"] if api == "graphql" else request.full_url.removeprefix(runpod.REST)
        self.sent.append((api, request.get_method(), what))
        queue = self.graphql if api == "graphql" else self.rest
        assert queue, f"an unexpected {api} call: {what}"
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, body, headers = answer
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "x", _headers(headers), io.BytesIO(json.dumps(body).encode()))
        return Answer(status, body, headers)

    def apis(self) -> list[str]:
        return [api for api, _, _ in self.sent]


@pytest.fixture
def wire(monkeypatch, tmp_path):
    """A machine that holds the account's key, with RunPod played by a `Wire`, the clock at T0, and `research pods ssh` as argv."""
    (tmp_path / "config.toml").write_text('apikey = "dummy"\napiurl = "https://api.invalid/graphql"\n')
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "config.toml")
    w = Wire()
    monkeypatch.setattr(runpod.urllib.request, "urlopen", w)
    monkeypatch.setattr(runpod, "clock", lambda: T0)
    monkeypatch.setattr("sys.argv", ["/opt/venv/bin/research", "pods", "ssh", "vy-x", "--", "nvidia-smi"])
    monkeypatch.delenv("CURSOR_CONVERSATION_ID", raising=False)
    return w


def log_lines() -> list[str]:
    return (runpod._env_dir() / runpod.CALL_LOG).read_text().splitlines()


# ---- the call log ---------------------------------------------------------------------------------------------------------------------------

def test_every_call_sent_to_runpod_is_one_log_line_with_the_counts_left(wire, monkeypatch):
    wire.rest += [(200, REST_POD, REST_LEFT), REFUSED, urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))]
    wire.graphql += [(200, {"data": {"myself": {"clientBalance": 242.0}}}, GRAPHQL_LEFT)]
    assert runpod._request("GET", f"/pods/{POD}") == REST_POD
    with pytest.raises(runpod.PodError, match="GET /pods -> HTTP 429: .*rate limit exceeded for the day window") as refused:
        runpod._request("GET", "/pods")
    assert refused.value.code == 429
    monkeypatch.setenv("RESEARCH_AGENT", "circuits")
    monkeypatch.setattr("sys.argv", ["/data/src/research/__main__.py", "pods", "guard", "--supervise", "--budgets", "/notes"])
    assert runpod._graphql("{ myself { clientBalance } }")["data"]["myself"]["clientBalance"] == 242.0
    with pytest.raises(runpod.PodError, match="DELETE /pods/abc123def45678: URLError"):
        runpod._request("DELETE", f"/pods/{POD}")
    pid = runpod.os.getpid()
    assert log_lines() == [
        f"2026-11-12T02:18:09Z rest GET /pods/{POD} 200 day=5129 hour=404 minute=21 by=research,pods,ssh pid={pid}",
        f"2026-11-12T02:18:09Z rest GET /pods 429 day=0 hour=403 minute=20 by=research,pods,ssh pid={pid}",
        f"2026-11-12T02:18:09Z graphql POST myself.clientBalance 200 day=43019 hour=3574 minute=179 by=research,pods,guard pid={pid} "
        "agent=circuits",
        f"2026-11-12T02:18:09Z rest DELETE /pods/{POD} none by=research,pods,guard pid={pid} agent=circuits"]
    assert "dummy" not in (runpod._env_dir() / runpod.CALL_LOG).read_text(), "the key is in neither the call nor the caller"


def test_the_log_starts_again_past_its_size_and_a_log_that_cannot_be_written_fails_no_call(wire, monkeypatch, tmp_path):
    wire.rest += [(200, [], REST_LEFT)] * 4
    monkeypatch.setattr(runpod, "CALL_LOG_MAX", 150)
    for _ in range(3):
        assert runpod.pods() == []
    older = runpod._env_dir() / f"{runpod.CALL_LOG}.1"
    assert len(older.read_text().splitlines()) == 2 and len(log_lines()) == 1, "two lines pass 150 bytes: the third starts a new log"

    (tmp_path / "a-file").write_text("")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "a-file"))
    assert runpod.pods() == [] and wire.rest == [], "no directory to log in: the call is made and answered all the same"


# ---- GraphQL when REST gives no answer -------------------------------------------------------------------------------------------------------

def test_one_pod_is_read_from_graphql_when_rest_refuses_and_comes_back_in_rests_shape(wire):
    wire.rest += [REFUSED]
    wire.graphql += [(200, {"data": {"pod": GRAPHQL_POD}}, GRAPHQL_LEFT)]
    pod = runpod.get(POD)
    assert pod == AS_REST and runpod.ssh_target(pod) == runpod.ssh_target(REST_POD) == ("203.0.113.7", 40123)
    assert wire.sent == [("rest", "GET", f"/pods/{POD}"),
                         ("graphql", "POST", f'{{ pod(input: {{podId: "{POD}"}}) {{ {runpod.POD_FIELDS} }} }}')]
    assert [ln.split()[1:5] for ln in log_lines()] == [["rest", "GET", f"/pods/{POD}", "429"], ["graphql", "POST", "pod.input", "200"]]


def test_the_pods_are_listed_from_graphql_when_rest_refuses(wire):
    wire.rest += [REFUSED, (503, {"error": "upstream"}, {}), urllib.error.URLError(TimeoutError("timed out"))]
    wire.graphql += [(200, {"data": {"myself": {"pods": [GRAPHQL_POD, GRAPHQL_STOPPED]}}}, GRAPHQL_LEFT)] * 3
    stopped = {"id": "zzz999yyy88877", "name": "vy-old-veritor-campaign", "desiredStatus": "EXITED", "costPerHr": 0.03, "machineId": "m2",
               "createdAt": "2026-09-21T17:53:28.637Z", "machine": {"gpuTypeId": "unknown"}, "publicIp": "", "portMappings": {}}
    for _ in range(3):                      # its rate limit, a server error, no answer at all
        assert runpod.pods() == [AS_REST, stopped]
    assert runpod.ssh_target(stopped) is None and wire.apis() == ["rest", "graphql"] * 3
    assert wire.sent[1][2] == f"{{ myself {{ pods {{ {runpod.POD_FIELDS} }} }} }}"
    assert guard.parse_time(AS_REST["createdAt"]) == guard.parse_time(REST_POD["createdAt"]), "the guard reads a pod's age from either"


def test_an_answer_from_rest_stands_and_a_pod_graphql_does_not_know_is_gone(wire):
    wire.rest += [(404, {"error": "pod not found"}, REST_LEFT), (401, {"error": "unauthorized"}, {}), REFUSED]
    wire.graphql += [(200, {"data": {"pod": None}}, GRAPHQL_LEFT)]
    with pytest.raises(runpod.PodError, match="HTTP 404: not found") as gone:
        runpod.get(POD)
    with pytest.raises(runpod.PodError, match="GET /pods -> HTTP 401") as denied:
        runpod.pods()
    assert (gone.value.code, denied.value.code) == (404, 401) and wire.apis() == ["rest", "rest"], "REST answered: GraphQL is not asked"
    with pytest.raises(runpod.PodError, match=f"GraphQL pod {POD}: not found") as also_gone:
        runpod.get(POD)
    assert also_gone.value.code == 404 and also_gone.value.pod_id == POD


def test_when_neither_answers_the_error_names_both_and_keeps_rests_status(wire):
    wire.rest += [REFUSED, REFUSED, REFUSED]
    wire.graphql += [(200, {"errors": [{"message": "Cannot query field"}]}, {}), (429, {"error": "too many requests"}, {}),
                     urllib.error.URLError(TimeoutError("timed out"))]
    for said in ("GraphQL: .*Cannot query field", "GraphQL -> HTTP 429: .*too many requests", "GraphQL: URLError: timed out"):
        with pytest.raises(runpod.PodError, match=f"GET /pods/{POD} -> HTTP 429: .*day window.*; {said}") as both:
            runpod.get(POD)
        assert both.value.code == 429 and both.value.pod_id == POD
    wire.rest += [REFUSED]
    with pytest.raises(runpod.PodError, match="GET /pods/no-such-id -> HTTP 429: .*; GraphQL: 'no-such-id' is not a pod id"):
        runpod.get("no-such-id")
    assert wire.apis().count("graphql") == 3, "an id that is not one is put in no query"


def test_a_machine_with_no_key_asked_the_door_and_its_failure_stands(wire, monkeypatch, tmp_path):
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "absent.toml")
    monkeypatch.setenv("RESEARCH_GUARD_HOST", "local")
    with pytest.raises(runpod.PodError, match="no RunPod API config at .* and RUNPOD_API_KEY is not set$"):
        runpod.get(POD)
    assert wire.sent == []
