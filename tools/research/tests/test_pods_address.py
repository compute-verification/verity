"""How `research` reaches a RunPod machine without asking RunPod at every call (pods/runpod.py, remote.pod_target): the address
last read is kept and used only where the host there proves the host key recorded for it, by ssh commands that check that key; a
read asks GraphQL when REST gives no answer; and every call sent to RunPod is one line of the call log, with the counts RunPod
says are left. No network and no ssh: `urlopen` answers from a queue, OpenSSH's client is a function that says what ssh says of a
command built here, and the clock is the test's."""
from __future__ import annotations

import email.message
import http.client
import io
import json
import re
import subprocess
import urllib.error
from pathlib import Path

import pytest

from research import remote
from research.pods import connect, guard, runpod

POD = "abc123def45678"
#: the headers as RunPod sends them: REST's one `ratelimit` header, GraphQL's one header a window
REST_LEFT = {"ratelimit": '"minute";r=21;t=30, "hour";r=400;t=1800, "day";r=5000;t=43200',
             "ratelimit-policy": '"minute";q=22;w=60, "hour";q=450;w=3600, "day";q=5400;w=86400'}
REST_SPENT = {"ratelimit": '"minute";r=20;t=29, "hour";r=399;t=1799, "day";r=0;t=43199', "retry-after": "43199"}
GRAPHQL_LEFT = {"x-ratelimit-remaining-minute": "170", "x-ratelimit-remaining-hour": "3500", "x-ratelimit-remaining-day": "43000"}
REFUSED = (429, {"error": "rate limit exceeded for the day window", "status": 429}, REST_SPENT)
REST_POD = {"id": POD, "name": "vy-x-veritor-campaign", "desiredStatus": "RUNNING", "costPerHr": 0.16, "publicIp": "203.0.113.7",
            "portMappings": {"22": 40123}, "machine": {}, "machineId": "m1", "createdAt": "2025-01-02 03:04:05.678 +0000 UTC",
            "env": {"PUBLIC_KEY": "ssh-ed25519 dummy"}}
GRAPHQL_POD = {"id": POD, "name": "vy-x-veritor-campaign", "desiredStatus": "RUNNING", "costPerHr": 0.16, "machineId": "m1",
               "createdAt": "2025-01-02T03:04:05.678Z", "machine": {"gpuTypeId": "unknown"},
               "runtime": {"ports": [{"ip": "100.65.0.9", "isIpPublic": False, "privatePort": 19123, "publicPort": 60302, "type": "http"},
                                     {"ip": "203.0.113.7", "isIpPublic": True, "privatePort": 22, "publicPort": 40123, "type": "tcp"}]}}
GRAPHQL_STOPPED = {"id": "zzz999yyy88877", "name": "vy-old-veritor-campaign", "desiredStatus": "EXITED", "costPerHr": 0.03,
                   "machineId": "m2", "createdAt": "2024-12-01T01:02:03.456Z", "machine": {"gpuTypeId": "unknown"}, "runtime": None}
AS_REST = {"id": POD, "name": "vy-x-veritor-campaign", "desiredStatus": "RUNNING", "costPerHr": 0.16, "machineId": "m1",
           "createdAt": "2025-01-02T03:04:05.678Z", "machine": {"gpuTypeId": "unknown"}, "publicIp": "203.0.113.7",
           "portMappings": {"22": 40123}}
T0 = 1700000000.0       # 2023-11-14T22:13:20Z
KEY = "rpa_DUMMY0123456789dummy"


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
    (tmp_path / "config.toml").write_text(f'apikey = "{KEY}"\napiurl = "https://api.invalid/graphql"\n')
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "config.toml")
    w = Wire()
    monkeypatch.setattr(runpod.urllib.request, "urlopen", w)
    monkeypatch.setattr(runpod, "clock", lambda: T0)
    monkeypatch.setattr("sys.argv", ["/opt/venv/bin/research", "pods", "ssh", "vy-x", "--", "nvidia-smi"])
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
    monkeypatch.setattr("sys.argv", ["/data/src/research/__main__.py", "pods", "guard", "--supervise", "--budgets", "/notes"])
    assert runpod._graphql("{ myself { clientBalance } }")["data"]["myself"]["clientBalance"] == 242.0
    with pytest.raises(runpod.PodError, match="DELETE /pods/abc123def45678: URLError"):
        runpod._request("DELETE", f"/pods/{POD}")
    pid = runpod.os.getpid()
    assert log_lines() == [
        f"2023-11-14T22:13:20Z rest GET /pods/{POD} 200 day=5000 hour=400 minute=21 by=research,pods,ssh pid={pid}",
        f"2023-11-14T22:13:20Z rest GET /pods 429 day=0 hour=399 minute=20 by=research,pods,ssh pid={pid}",
        f"2023-11-14T22:13:20Z graphql POST myself.clientBalance 200 day=43000 hour=3500 minute=170 by=research,pods,guard pid={pid}",
        f"2023-11-14T22:13:20Z rest DELETE /pods/{POD} none by=research,pods,guard pid={pid}"]


def test_the_api_key_is_in_no_log_line_no_message_and_no_error(wire, monkeypatch, capsys):
    """REST's header and the GraphQL request's URL hold the key. It is written nowhere: not when RunPod's answer repeats the
    request, not when a failure's own text holds the URL or the header, and not when the command line holds the key."""
    url = f"https://api.invalid/graphql?api_key={KEY}"
    echo = {"error": f"refused {url} with Authorization: Bearer {KEY}"}
    struck = json.dumps(echo).replace(KEY, "<the API key>")
    monkeypatch.setattr("sys.argv", ["/opt/venv/bin/research", KEY, "pods"])
    invalid = http.client.InvalidURL(f"URL can't contain control characters. '/graphql?api_key={KEY} ' (found at least ' ')")
    no_ascii = UnicodeEncodeError("ascii", f"POST /graphql?api_key={KEY}\u00e9 HTTP/1.1", 39, 40, "ordinal not in range(128)")
    said = []
    for failure in ((429, echo, {}), (200, {"errors": [echo]}, GRAPHQL_LEFT), invalid, no_ascii,
                    http.client.IncompleteRead(KEY.encode()), urllib.error.URLError(f"unreachable: {url}")):
        wire.rest += [(429, echo, REST_SPENT)]
        wire.graphql += [failure]
        with pytest.raises(runpod.PodError) as both:
            runpod.get(POD)
        said.append(str(both.value).removeprefix(f"GET /pods/{POD} -> HTTP 429: {struck}; "))
    no_request = "the RunPod config's API URL or key holds a character no request can carry"
    assert said == [f"GraphQL -> HTTP 429: {struck}", f"GraphQL: [{struck}]", f"GraphQL: {no_request}", f"GraphQL: {no_request}",
                    "GraphQL: IncompleteRead", "GraphQL: URLError: unreachable: https://api.invalid/graphql?api_key=<the API key>"]
    wire.graphql += [invalid]
    with pytest.raises(runpod.PodError) as balance:
        runpod._graphql("{ myself { clientBalance } }")         # the balance read's callers print what it raises
    wire.rest += [ValueError(f"Invalid header value b'Bearer {KEY}\\n'")]
    with pytest.raises(runpod.PodError) as header:
        runpod._request("GET", "/pods")
    assert [str(balance.value), str(header.value)] == [
        f"GraphQL: {no_request}", "GET /pods: the RunPod config's API key holds a character no request can carry"]
    written = "\n".join([*said, str(balance.value), str(header.value), *log_lines(), *capsys.readouterr()])
    assert KEY not in written and len(log_lines()) == 14 and {ln.split()[-2] for ln in log_lines()} == {"by=research"}


@pytest.mark.parametrize("key", ["", KEY], ids=["empty-key", "key"])
def test_only_a_key_the_config_holds_is_struck_from_what_runpod_answers(wire, tmp_path, key):
    """An empty key is between every two characters of any text, so striking it would leave RunPod's answer unreadable: with no
    key to strike, REST's answer and GraphQL's are quoted as they came."""
    (tmp_path / "config.toml").write_text(f'apikey = "{key}"\napiurl = "https://api.invalid/graphql"\n')
    echo = {"error": f"rate limit exceeded for Bearer {KEY}"}
    answer = json.dumps(echo).replace(KEY, "<the API key>") if key else json.dumps(echo)
    wire.rest += [(429, echo, REST_SPENT)] * 2
    wire.graphql += [(429, echo, {}), (200, {"errors": [echo]}, GRAPHQL_LEFT)]
    for graphql in (f"GraphQL -> HTTP 429: {answer}", f"GraphQL: [{answer}]"):
        with pytest.raises(runpod.PodError) as both:
            runpod.get(POD)
        assert str(both.value) == f"GET /pods/{POD} -> HTTP 429: {answer}; {graphql}"


def test_the_log_starts_again_past_its_size_and_a_log_that_cannot_be_written_fails_no_call(wire, monkeypatch, tmp_path, capsys):
    wire.rest += [(200, [], REST_LEFT)] * 5
    monkeypatch.setattr(runpod, "CALL_LOG_MAX", 150)
    for _ in range(3):
        assert runpod.pods() == []
    older = runpod._env_dir() / f"{runpod.CALL_LOG}.1"
    assert len(older.read_text().splitlines()) == 2 and len(log_lines()) == 1, "two lines pass 150 bytes: the third starts a new log"

    (tmp_path / "a-file").write_text("")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "a-file"))
    assert runpod.pods() == [] and runpod.pods() == [] and wire.rest == [], "no directory to log in: the calls are made and answered"
    said, = capsys.readouterr().err.splitlines()
    assert said.startswith("research: RunPod's call log can't be written, so its calls are in none: NotADirectoryError: ")


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
               "createdAt": "2024-12-01T01:02:03.456Z", "machine": {"gpuTypeId": "unknown"}, "publicIp": "", "portMappings": {}}
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


# ---- the address kept between calls ----------------------------------------------------------------------------------------------------------

A, B = ("203.0.113.7", 40123), ("203.0.113.9", 41000)
#: what OpenSSH's client says, told to check a host's key, of a host it has no key for and of one that proves another
NO_KEY_RECORDED = ("No ED25519 host key is known for [{}]:{} and you have requested strict checking.\r\n"
                   "Host key verification failed.\r\n")
ANOTHER_KEY = ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\r\n"
               "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\r\n"
               "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\r\n"
               "IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!\r\n"
               "Offending ED25519 key in /home/u/.local/state/research/runpod/known_hosts:1\r\n"
               "Host key for [{}]:{} has changed and you have requested strict checking.\r\n"
               "Host key verification failed.\r\n")
DOWN = runpod.PodError(f"GET /pods/{POD} -> HTTP 429: rate limit exceeded for the day window; GraphQL: URLError: timed out",
                       code=429, pod_id=POD)
ELSEWHERE = {**REST_POD, "publicIp": B[0], "portMappings": {"22": B[1]}}


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """RunPod as `runpod.get` sees it, and the hosts as ssh does. `answer` is what RunPod gives next (a pod, or an error to raise)
    and `asked` counts the calls. `hosts` is the host key of whoever is at each address, `recorded` the key known_hosts holds for
    each, and `ran` every command `runpod` ran about a host's key. `ssh` is OpenSSH's client on a command built here: what it says
    on stderr, having recorded a new host's key unless told to be strict, and refused a changed one unless told to check none.
    `writable` is whether known_hosts can be written: where it can't, ssh records no key and goes on as it would have."""
    class Pod:
        answer: object = dict(REST_POD)
        asked = 0
        clock = T0
        hosts: dict = {}
        recorded: dict = {}
        ran: list = []
        keygen_exits = 0
        writable = True

        @staticmethod
        def ssh(argv: list[str]) -> str:
            host = next(a for a in argv if re.fullmatch(r"root@[\d.]+", a)).removeprefix("root@")
            at, checking = (host, int(argv[argv.index("-p") + 1])), checks(argv)
            if at not in Pod.hosts:
                return f"ssh: connect to host {host} port {at[1]}: Connection refused\r\n"
            if checking != "no" and Pod.recorded.get(at, Pod.hosts[at]) != Pod.hosts[at]:
                return ANOTHER_KEY.format(*at)
            if checking == "yes" and at not in Pod.recorded:
                return NO_KEY_RECORDED.format(*at)
            if Pod.writable:
                Pod.recorded.setdefault(at, Pod.hosts[at])  # checking none, a changed key is warned of and left as recorded
            return f"root@{host}: Permission denied (publickey).\r\n" if "PreferredAuthentications=none" in argv else ""

        @staticmethod
        def logs_in(target: dict) -> bool:
            """Whether the launcher's ssh for the target gets a shell on the host there."""
            return Pod.ssh(remote.ssh_argv(target["host"], target["port"], user=target["user"])) == ""

        @staticmethod
        def handshakes() -> list[tuple[str, str]]:
            """(the address, how the key was checked) of each ssh `runpod` ran, and ("ssh-keygen", the name dropped) of each drop."""
            return [("ssh-keygen", c[2]) if c[0] == "ssh-keygen" else (f"{c[-2].removeprefix('root@')}:{c[c.index('-p') + 1]}", checks(c))
                    for c in Pod.ran]

    def get(pod_id):
        Pod.asked += 1
        if isinstance(Pod.answer, Exception):
            raise Pod.answer
        return Pod.answer

    def run(argv, **_kwargs):
        Pod.ran.append(list(argv))
        if argv[0] == "ssh":
            return subprocess.CompletedProcess(argv, 255, "", Pod.ssh(argv))
        assert argv[:2] == ["ssh-keygen", "-R"] and argv[3:] == ["-f", str(runpod.known_hosts())]
        host, port = re.fullmatch(r"\[(.+)\]:(\d+)", argv[2]).groups()
        if not Pod.keygen_exits:
            Pod.recorded.pop((host, int(port)), None)
        return subprocess.CompletedProcess(argv, Pod.keygen_exits, "", "")

    monkeypatch.setattr(runpod, "get", get)
    monkeypatch.setattr(runpod, "run", run)
    # known_hosts() is then a file of the test's own state directory, made here for a record to be dropped from (`drop_key`)
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "absent" / "known_hosts")
    runpod.known_hosts().touch()
    monkeypatch.setattr(runpod, "clock", lambda: Pod.clock)
    monkeypatch.setattr(remote, "utc_now_ms", lambda: f"read-{Pod.asked}")      # a target says which read of RunPod it is
    monkeypatch.setattr(remote, "_TOLD_KEPT", set())
    monkeypatch.setattr(remote, "POD_TARGET_RETRY_S", 0.0)
    return Pod


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """The registry names the pod `vy-x`."""
    (tmp_path / "machines.d").mkdir()
    (tmp_path / "machines.d" / "vy-x.toml").write_text(f'provider = "runpod"\npod_id = "{POD}"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(tmp_path / "machines.d"))
    monkeypatch.setenv("RESEARCH_NOTES", str(tmp_path / "no-notes"))
    return tmp_path


MACHINE = {"provider": "runpod", "pod_id": POD, "name": "vy-x"}
BUILDERS = ("runpod.ssh_command", "remote.ssh_argv", "remote._compressed", "remote.git_push_url", "connect.ssh_for",
            "connect.SshTransport.argv", "connect.SshTransport.rsync_args")
AT_A, AT_B = "203.0.113.7:40123", "203.0.113.9:41000"


def checks(command) -> str:
    """How an ssh command, an rsync `-e` or a git `ext::` URL checks the host's key: its one StrictHostKeyChecking option."""
    option, = re.findall(r"StrictHostKeyChecking=([\w-]+)", command if isinstance(command, str) else " ".join(command))
    return option


def commands(target: dict) -> dict[str, object]:
    """Every ssh command `research` builds for a target, by its builder: ssh's own argv, rsync's `-e` and git's `ext::` URL."""
    host, port = target["host"], target["port"]
    launcher, pods = remote.ssh_argv(host, port, user=target["user"]), connect.ssh_for(host, port, user=target["user"])
    built = {"runpod.ssh_command": runpod.ssh_command(host, port, user=target["user"]), "remote.ssh_argv": launcher,
             "remote._compressed": remote._compressed(launcher), "remote.git_push_url": remote.git_push_url(launcher, "/workspace/src.git"),
             "connect.ssh_for": pods, "connect.SshTransport.argv": connect.SshTransport(pods).argv("true"),
             "connect.SshTransport.rsync_args": connect.SshTransport(pods).rsync_args("/workspace/src")}
    assert tuple(built) == BUILDERS
    return built


def test_a_kept_address_is_used_for_an_hour_where_its_host_proves_the_key_it_held_when_runpod_gave_it(pod):
    pod.hosts = {A: "the pod's key"}
    first = remote.pod_target(MACHINE)
    assert first == {"host": "203.0.113.7", "port": 40123, "user": "root", "pod_name": "vy-x-veritor-campaign", "cost_per_hr": 0.16,
                     "rate_observed_utc": "read-1", "gpu": None} and tuple(first) == runpod.ADDRESS_KEYS and pod.asked == 1
    assert pod.handshakes() == [(AT_A, "accept-new")] and pod.recorded == {A: "the pod's key"}, "the host RunPod named proved a key"
    assert checks(runpod.ssh_command(*A)) == "no", "RunPod's word: dialled as ever"

    pod.clock = T0 + 3599
    for _ in range(5):
        assert remote.pod_target(MACHINE) == {**first, "kept": True}, "the address, the rate and the time the rate was read, as kept"
    assert pod.asked == 1 and pod.handshakes()[1:] == [(AT_A, "yes")] * 5 and checks(runpod.ssh_command(*A)) == "yes" and pod.logs_in(first)

    pod.answer, pod.hosts = ELSEWHERE, {B: "the pod's next key"}            # the pod restarted elsewhere: nothing is at A
    moved = remote.pod_target(MACHINE)
    assert (moved["host"], moved["port"]) == B and "kept" not in moved and pod.asked == 2 and checks(runpod.ssh_command(*B)) == "no"
    assert remote.pod_target(MACHINE) == {**moved, "kept": True} and pod.asked == 2
    assert pod.handshakes()[6:] == [(AT_A, "yes"), (AT_B, "accept-new"), (AT_B, "yes")], "a silent host has no key dropped"

    pod.clock += runpod.ADDRESS_TTL_S
    again = remote.pod_target(MACHINE)
    assert (again["host"], again["port"]) == B and "kept" not in again and pod.asked == 3 and pod.handshakes()[9:] == [(AT_B, "accept-new")], \
        "an hour old: RunPod is asked, whatever the host would prove"
    assert checks(runpod.ssh_command(*B)) == "no", "RunPod's fresh word stands as it always did"


def test_every_ssh_command_for_a_kept_address_checks_the_host_key_and_for_a_fresh_one_connects_as_before(pod, registry):
    pod.hosts = {A: "the pod's key"}
    fresh = remote.pod_target(MACHINE)
    assert "kept" not in fresh and {name: checks(c) for name, c in commands(fresh).items()} == dict.fromkeys(BUILDERS, "no")

    kept = remote.pod_target(MACHINE)
    assert kept["kept"] is True and {name: checks(c) for name, c in commands(kept).items()} == dict.fromkeys(BUILDERS, "yes")
    by_name = connect.ssh_base("vy-x")
    assert checks(by_name) == "yes" and by_name[-1] == "root@203.0.113.7" and pod.asked == 1
    assert commands(kept)["connect.SshTransport.rsync_args"][0] == "-e" and commands(kept)["remote.git_push_url"].startswith("ext::ssh ")
    assert checks(runpod.ssh_command(*B)) == "no", "another address is not this one"

    src = Path(runpod.__file__).parents[1]
    writers = sorted(str(p.relative_to(src)) for p in src.rglob("*.py") if "StrictHostKeyChecking=" in p.read_text())
    assert writers == ["pods/nebius/fleet.py", "pods/runpod.py"], \
        "a pod's ssh options are written in one place, runpod.ssh_command, which every builder above starts from (fleet.py's are a Nebius VM's)"


def test_an_address_is_kept_only_if_its_host_proves_a_key_when_runpod_gives_it(pod):
    first = remote.pod_target(MACHINE)                  # no host answers there yet
    assert "kept" not in first and runpod.kept_address(POD) is None and pod.handshakes() == [(AT_A, "accept-new")]
    assert remote.pod_target(MACHINE)["rate_observed_utc"] == "read-2" and pod.asked == 2, "not kept: RunPod is asked at each call"

    pod.hosts = {A: "another tenant's key"}             # a command that still holds RunPod's answer dials later, as it always could,
    assert pod.logs_in(first) and pod.recorded == {A: "another tenant's key"}       # and records the key of whoever is there by then
    assert runpod.kept_address(POD) is None, "a key first recorded after RunPod's word makes no address kept"
    pod.answer, pod.hosts[B] = ELSEWHERE, "the pod's key"
    assert remote.pod_target(MACHINE)["host"] == B[0] and pod.asked == 3 and runpod.kept_address(POD)[0]["host"] == B[0]


def test_a_kept_address_with_no_recorded_key_asks_runpod(pod):
    pod.hosts = {A: "the pod's key"}
    remote.pod_target(MACHINE)
    pod.recorded.clear()                                # the record is lost
    target = remote.pod_target(MACHINE)
    assert "kept" not in target and target["rate_observed_utc"] == "read-2" and pod.asked == 2 and checks(runpod.ssh_command(*A)) == "no"
    assert pod.handshakes() == [(AT_A, "accept-new"), (AT_A, "yes"), (AT_A, "accept-new")] and pod.recorded == {A: "the pod's key"}
    assert pod.ran[1] == ["ssh", "-i", str(runpod.ssh_key()), "-p", "40123", "-o", "StrictHostKeyChecking=yes",
                          "-o", f"UserKnownHostsFile={runpod.known_hosts()}", "-o", "ServerAliveInterval=30",
                          "-o", "BatchMode=yes", "-o", "PreferredAuthentications=none", "-o", "ControlPath=none",
                          "-o", "ConnectTimeout=5", "root@203.0.113.7", "true"], "one handshake of its own that asks for no login"


def test_a_host_refused_for_its_key_is_not_dialled_and_the_address_runpod_gives_is(pod, capsys):
    pod.hosts = {A: "the pod's key"}
    remote.pod_target(MACHINE)
    assert remote.pod_target(MACHINE)["kept"] is True and pod.asked == 1

    pod.answer, pod.hosts = ELSEWHERE, {A: "another tenant's key", B: "the pod's next key"}     # its old address is someone else's
    del pod.ran[:]
    moved = remote.pod_target(MACHINE)
    assert (moved["host"], moved["port"]) == B and "kept" not in moved and pod.asked == 2
    assert pod.handshakes() == [(AT_A, "yes"), (AT_B, "accept-new")]
    assert pod.recorded == {A: "the pod's key", B: "the pod's next key"}, "no record is dropped for an address RunPod did not just give"
    assert json.loads((runpod._env_dir() / f"address-{POD}.json").read_text())["target"]["host"] == B[0]
    assert checks(runpod.ssh_command(*A)) == "yes", "no command built here dials the refused host unchecked"
    assert checks(runpod.ssh_command(*B)) == "no" and remote.pod_target(MACHINE) == {**moved, "kept": True}

    pod.answer, pod.hosts = DOWN, {B: "another tenant's key"}               # and when RunPod can't be asked, the command is refused
    with pytest.raises(remote.Refused, match=rf"pod {POD} \(machine vy-x\): runpod API unavailable: GET /pods/{POD} -> HTTP 429"):
        remote.pod_target(MACHINE)
    assert runpod.kept_address(POD) is None and checks(runpod.ssh_command(*B)) == "yes" and capsys.readouterr().err == ""
    with pytest.raises(remote.Refused, match="runpod API unavailable"):
        remote.pod_target(MACHINE)
    assert pod.handshakes()[2:] == [(AT_B, "yes"), (AT_B, "yes")], "with no kept address no host is asked for a key"


def test_a_pod_back_at_its_address_with_a_new_host_key_is_kept_again_once_runpod_gives_the_address(pod, capsys):
    pod.hosts = {A: "the pod's key"}
    remote.pod_target(MACHINE)
    assert remote.pod_target(MACHINE)["kept"] is True
    pod.hosts = {A: "the pod's key after its restart"}
    del pod.ran[:]
    again = remote.pod_target(MACHINE)
    assert "kept" not in again and pod.asked == 2 and checks(runpod.ssh_command(*A)) == "no", "RunPod has just given this address"
    assert pod.handshakes() == [(AT_A, "yes"), (AT_A, "accept-new"), ("ssh-keygen", "[203.0.113.7]:40123"), (AT_A, "accept-new")], \
        "refused as kept; RunPod asked; its word is fresher than the record, which is dropped and made again"
    assert pod.recorded == {A: "the pod's key after its restart"}
    assert remote.pod_target(MACHINE) == {**again, "kept": True} and pod.asked == 2

    pod.hosts, pod.keygen_exits = {A: "a third key"}, 1         # a record that can't be dropped: not kept, RunPod asked each time
    for asked in (3, 4):
        assert "kept" not in remote.pod_target(MACHINE) and pod.asked == asked and checks(runpod.ssh_command(*A)) == "no"
    said, = capsys.readouterr().err.splitlines()
    assert runpod.kept_address(POD) is None and said == (
        "research: the host key recorded for [203.0.113.7]:40123 can't be removed (ssh-keygen exits 1): until it is, RunPod is asked "
        "for that address at each call")


def test_when_runpod_cannot_be_asked_the_kept_address_stands_only_where_its_host_proves_the_recorded_key(pod, capsys):
    pod.hosts = {A: "the pod's key"}
    first = remote.pod_target(MACHINE)
    pod.clock += 2 * runpod.ADDRESS_TTL_S
    pod.answer = DOWN
    for _ in range(2):
        assert remote.pod_target(MACHINE) == {**first, "kept": True}
    assert pod.asked == 3 and {name: checks(c) for name, c in commands(first).items()} == dict.fromkeys(BUILDERS, "yes")
    said, = capsys.readouterr().err.splitlines()
    assert said.startswith(f"research: pod {POD} (machine vy-x): runpod API unavailable (GET /pods/{POD} -> HTTP 429")
    assert said.endswith("using its address as read read-1, where the host holds the key recorded for it")

    pod.hosts = {}                                  # nothing answers there: refused, and the address kept for when the pod is back
    with pytest.raises(remote.Refused, match=rf"pod {POD} \(machine vy-x\): runpod API unavailable: GET /pods/{POD} -> HTTP 429"):
        remote.pod_target(MACHINE)
    assert runpod.kept_address(POD) is not None

    pod.hosts, pod.recorded = {A: "the pod's key"}, {}      # a host, and no key recorded for the address: refused, and forgotten
    with pytest.raises(remote.Refused, match="runpod API unavailable"):
        remote.pod_target(MACHINE)
    assert runpod.kept_address(POD) is None and checks(runpod.ssh_command(*A)) == "yes"
    assert pod.recorded == {} and pod.handshakes() == [(AT_A, "accept-new")] + [(AT_A, "yes")] * 4, "no key is recorded without RunPod's word"


def test_a_kept_address_under_an_hour_old_whose_host_is_silent_is_refused_when_runpod_cannot_be_asked(pod):
    pod.hosts = {A: "the pod's key"}
    remote.pod_target(MACHINE)
    pod.hosts, pod.answer = {}, DOWN
    with pytest.raises(remote.Refused, match="runpod API unavailable"):
        remote.pod_target(MACHINE)
    assert pod.asked == 2 and pod.handshakes() == [(AT_A, "accept-new"), (AT_A, "yes")], "asked for its key once, not again after RunPod failed"


def test_the_mark_is_never_stored_so_a_later_process_asks_the_host_for_its_key_too(pod, monkeypatch):
    pod.hosts = {A: "the pod's key"}
    fresh = remote.pod_target(MACHINE)
    path = runpod._env_dir() / f"address-{POD}.json"
    assert remote.pod_target(MACHINE) == {**fresh, "kept": True} and json.loads(path.read_text()) == {"target": fresh, "read_at": T0}
    runpod.keep_address(POD, {**fresh, "kept": True})
    assert json.loads(path.read_text()) == {"target": fresh, "read_at": T0}, "a marked target is stored without its mark"

    monkeypatch.setattr(runpod, "_KEPT", set())                    # a later process, and by then another tenant has the address
    pod.hosts, pod.answer = {A: "another tenant's key"}, DOWN
    assert checks(runpod.ssh_command(*A)) == "no", "it knows the address only through pod_target, which asks the host first"
    with pytest.raises(remote.Refused, match="runpod API unavailable"):
        remote.pod_target(MACHINE)
    assert pod.asked == 2 and not path.exists()

    path.write_text(json.dumps({"target": {**fresh, "kept": True}, "read_at": T0}))
    assert runpod.kept_address(POD) is None and not path.exists(), "a file that carries the mark is no address"


@pytest.mark.parametrize(("exits", "stderr", "verdict"), [
    (255, "root@203.0.113.7: Permission denied (publickey).\r\n", "held"),
    (255, "Warning: Permanently added '[203.0.113.7]:40123' (ED25519) to the list of known hosts.\r\n"
          "root@203.0.113.7: Permission denied (publickey,password).\r\n", "held"),
    (0, "", "held"),
    (255, NO_KEY_RECORDED.format(*A), "refused"),
    (255, ANOTHER_KEY.format(*A), "refused"),
    (255, "ssh: connect to host 203.0.113.7 port 40123: Connection refused\r\n", "silent"),
    (255, "ssh: connect to host 203.0.113.7 port 40123: Connection timed out\r\n", "silent"),
    (255, "kex_exchange_identification: Connection closed by remote host\r\nConnection closed by 203.0.113.7 port 40123\r\n", "silent"),
    (255, 'Load key "/home/u/.runpod/ssh/key": Permission denied\r\nConnection reset by 203.0.113.7 port 40123\r\n', "silent"),
    (255, "", "silent"),
    (subprocess.TimeoutExpired("ssh", 20.0), "", "silent"),
    (FileNotFoundError(2, "No such file or directory: 'ssh'"), "", "silent"),
], ids=["denied-login", "denied-login-after-recording", "let-in", "no-key-recorded", "another-key", "connection-refused",
        "connection-timed-out", "closed-before-keys", "reset-before-keys", "said-nothing", "ssh-timed-out", "no-ssh"])
def test_a_host_holds_its_recorded_key_only_when_ssh_checked_the_key_and_got_as_far_as_login(monkeypatch, exits, stderr, verdict):
    ran = []

    def run(argv, **kwargs):
        ran.append((argv, kwargs))
        if isinstance(exits, Exception):
            raise exits
        return subprocess.CompletedProcess(argv, exits, "", stderr)

    monkeypatch.setattr(runpod, "run", run)
    assert runpod.host_key(*A) == verdict and runpod.host_key(*A, fresh=True) == verdict
    assert [checks(argv) for argv, _ in ran] == ["yes", "accept-new"], "a kept address's host; a host RunPod has just named"
    for argv, kwargs in ran:
        assert argv[-2:] == ["root@203.0.113.7", "true"] and argv[argv.index("-p") + 1] == "40123"
        assert kwargs == {"stdin": subprocess.DEVNULL, "capture_output": True, "text": True, "timeout": runpod.HOST_KEY_S}


def test_known_hosts_is_the_configured_file_where_its_directory_exists_and_else_a_file_of_the_state_directory(monkeypatch, tmp_path):
    """`known_hosts()` names the file ssh is to use and writes none itself: ssh and ssh-keygen do."""
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "absent" / "known_hosts")
    state = runpod._env_dir()
    assert not state.exists()
    assert runpod.known_hosts() == state / "known_hosts" and list(state.iterdir()) == [] and not (tmp_path / "absent").exists()
    assert f"UserKnownHostsFile={state / 'known_hosts'}" in runpod.ssh_command(*A)

    (tmp_path / "ssh").mkdir()
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "ssh" / "known_hosts")
    assert runpod.known_hosts() == tmp_path / "ssh" / "known_hosts" and list((tmp_path / "ssh").iterdir()) == []
    assert f"UserKnownHostsFile={tmp_path / 'ssh' / 'known_hosts'}" in runpod.ssh_command(*A)


def test_where_known_hosts_cannot_be_written_no_kept_address_stands_and_runpod_is_asked_at_each_call(pod, monkeypatch, tmp_path, capsys):
    """The configured file's directory exists and nothing can be written there. `research` writes nothing to it, so nothing
    here raises: ssh records no key and connects on RunPod's word as ever. With no key recorded no kept address stands, so
    RunPod is asked at each call, as it was before addresses were kept, and no host is dialled on a kept address."""
    (tmp_path / "ssh").mkdir()
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "ssh" / "known_hosts")
    pod.hosts, pod.writable = {A: "the pod's key"}, False
    for asked in (1, 2, 3):
        target = remote.pod_target(MACHINE)
        assert "kept" not in target and pod.asked == asked and checks(runpod.ssh_command(*A)) == "no" and pod.logs_in(target)
    assert pod.handshakes() == [(AT_A, "accept-new"), (AT_A, "yes"), (AT_A, "accept-new"), (AT_A, "yes"), (AT_A, "accept-new")]
    assert all(f"UserKnownHostsFile={tmp_path / 'ssh' / 'known_hosts'}" in command for command in pod.ran)
    assert pod.recorded == {} and list((tmp_path / "ssh").iterdir()) == [] and capsys.readouterr().err == ""


def test_a_pod_that_is_gone_or_stopped_is_refused_and_its_address_forgotten(pod):
    pod.hosts = {A: "the pod's key"}
    remote.pod_target(MACHINE)
    pod.clock += runpod.ADDRESS_TTL_S
    pod.answer = {**REST_POD, "desiredStatus": "EXITED", "publicIp": "", "portMappings": None}
    with pytest.raises(remote.Refused, match="is EXITED; no implicit provisioning"):
        remote.pod_target(MACHINE)
    assert runpod.kept_address(POD) is None

    pod.answer = dict(REST_POD)
    remote.pod_target(MACHINE)
    pod.clock += runpod.ADDRESS_TTL_S
    pod.answer = runpod.PodError(f"GET /pods/{POD} -> HTTP 404: not found (terminated, or never existed)", code=404, pod_id=POD)
    with pytest.raises(remote.Refused, match=r"no longer exists on runpod \(HTTP 404"):
        remote.pod_target(MACHINE)
    assert runpod.kept_address(POD) is None and not list(runpod._env_dir().glob("address-*"))


def test_a_kept_file_that_is_not_a_target_is_removed_and_reads_as_none(pod):
    pod.hosts = {A: "the pod's key"}
    fresh = remote.pod_target(MACHINE)
    path = runpod._env_dir() / f"address-{POD}.json"
    assert runpod.kept_address(POD) == ({**fresh, "kept": True}, 0.0) and json.loads(path.read_text()) == {"target": fresh, "read_at": T0}
    for text in ("", "{", "[]", json.dumps({"target": {"host": "203.0.113.7"}, "read_at": T0}),
                 json.dumps({"target": {**fresh, "port": "40123"}, "read_at": T0}), json.dumps({"target": fresh, "read_at": None})):
        path.write_text(text)
        assert runpod.kept_address(POD) is None and not path.exists()
    assert runpod.kept_address("../outside") is None
    runpod.keep_address("../outside", fresh)
    runpod.forget_address("../outside")
    assert [p.name for p in runpod._env_dir().iterdir()] == ["known_hosts"]


# ---- the callers ---------------------------------------------------------------------------------------------------------------------------------

def test_an_ssh_entry_that_names_the_pod_overrides_the_registrys_entry_and_asks_runpod_and_no_host_nothing(pod, registry):
    """machines.toml may give a pod's address itself: an `ssh` entry carrying the pod's id binds the name as the registry does, so
    the two merge, and the address is the entry's, dialled as an ssh machine's always was."""
    (registry / "machines.toml").write_text(f'[machines.vy-x]\nprovider = "ssh"\nhost = "198.51.100.4"\nport = 12250\nuser = "root"\n'
                                            f'pod_id = "{POD}"\nrate_usd_per_hour = 0.16\nproject = "verity"\n')
    machine = remote.resolve_machine("vy-x", "verity")
    target = remote.pod_target(machine)
    assert (target["host"], target["port"], target["cost_per_hr"], machine["pod_id"]) == ("198.51.100.4", 12250, 0.16, POD)
    assert "kept" not in target and {name: checks(c) for name, c in commands(target).items()} == dict.fromkeys(BUILDERS, "no")
    by_name = connect.ssh_base("vy-x")
    assert by_name[-1] == "root@198.51.100.4" and checks(by_name) == "no"
    assert pod.asked == 0 and pod.ran == [] and not list(runpod._env_dir().glob("address-*"))

    (registry / "machines.toml").unlink()
    assert remote.pod_target(remote.resolve_machine("vy-x", "verity"))["host"] == "203.0.113.7" and pod.asked == 1, \
        "without the entry the registry's is read, and RunPod asked once"


def test_the_guard_lists_its_pods_through_graphql_while_rest_refuses(wire):
    wire.rest += [(200, [REST_POD], REST_LEFT), REFUSED]
    wire.graphql += [(200, {"data": {"myself": {"pods": [GRAPHQL_POD, GRAPHQL_STOPPED]}}}, GRAPHQL_LEFT)]
    api = guard.RunPodAPI()
    assert api.pods() == [REST_POD] and wire.apis() == ["rest"], "REST answers: it is the list, and GraphQL is not asked"
    listed = api.pods()
    assert [(p["id"], p["name"], p["desiredStatus"], p["costPerHr"]) for p in listed] == \
        [(POD, "vy-x-veritor-campaign", "RUNNING", 0.16), ("zzz999yyy88877", "vy-old-veritor-campaign", "EXITED", 0.03)]
    assert guard.parse_time(listed[0]["createdAt"]) == guard.parse_time(REST_POD["createdAt"])
