"""`research metrics`: the console's Metrics tab (/console/metrics) as text, from the site's GET /api/metrics, and a
loop's readings sent to it with POST /api/metrics.

  research metrics show [ID ...] [--owner HANDLE] [--json]   each metric a metrics.toml on main declares (one beside
                                                            each benchmark's script: benchmarks/README.md's
                                                            "Metrics"): its latest value, changes over 7 and 30
                                                            days, age and standing
  research metrics send [--command SCRIPT] [--commit SHA] [--dry-run] --metric ID (--value N | --why TEXT) --time ISO
                        [--mark M] [--target N] [--period P] [--over O] [--member M] [--run R]
                                                            one reading, or with no --metric the readings on stdin
                                                            (a JSON object or list), sent with the control pod's
                                                            panels key ($FLEET_PANELS_KEY), only to the site's own
                                                            host; --dry-run prints the request without the key and
                                                            sends nothing

The page and this command read one door. It answers a site key with events:read (`$JOBS_TOKEN`, else
~/.research/jobs/token, as `research jobs` reads it, when it starts with vsk_) or else this agent's Cursor identity
minted for the route. A site key goes only to the console's own route: --url anywhere else carries the identity.

A reading is `metric`, `value` (or null with `why`), `mark` and `time` (when the number was read, not when it is sent),
and names the `run` that recorded it or the `command` (a loop's script) that computed it, with `commit` beside either; it
may carry `unit`, `ci95`, `target`, `verdict`, `period`, `over` and `member`. The door refuses a request whose reading
holds any other key, and `send` refuses it first. The control pod's loops send theirs with infra/control/loop_metrics.py.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import urllib.error
from pathlib import Path
from urllib.parse import urlsplit

from .jobs.client import DEFAULT_URL, TOKEN_ENV, TOKEN_FILE, kept_token
from .pods.nebius.settings import PANELS_KEY
from .slack import RouteError, SlackError, _code, _Identity, http_json, http_open, route_error

METRICS = f"{DEFAULT_URL}/api/metrics"   # the route, and its OIDC token's audience
SITE_HOST = urlsplit(DEFAULT_URL).netloc   # the one host the panels key goes to
KEY_PREFIX = "vsk_"   # the site reads a bearer with it as a site key, and any other as a Cursor identity
MARKS = ("measured", "modeled", "partial")   # kit.MARKS
#: every key the door takes (the site's lib/console/metrics/readings.ts, KEYS): it refuses a reading with any other
KEYS = ("metric", "time", "value", "mark", "unit", "ci95", "target", "verdict", "why", "period", "over", "member", "run",
        "commit", "command")
COMMIT = re.compile(r"[0-9a-f]{7,40}")


def off_site(url: str) -> str | None:
    """Why ``url`` may not be sent the panels key: anything but https to the site's own host. None when it may."""
    u = urlsplit(url)
    if u.scheme == "https" and u.netloc == SITE_HOST:
        return None
    return f"{url!r} is not on the site (https://{SITE_HOST}), the one host the panels key goes to: nothing sent"


def send_json(method: str, url: str, body: object, key: str, what: str) -> str | None:
    """One JSON request to the site with ``key`` as its bearer, through `slack.http_open`, the client `http_json` reads with (no
    redirect followed, 30 s): None when the site took it, else why not, in one line; it never raises. Its answer isn't read.
    The key goes only to the site's own host."""
    if why := off_site(url):
        return f"{what}: {why}"
    try:
        try:
            with http_open(method, url, json.dumps(body).encode(),
                           {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}):
                return None
        except urllib.error.HTTPError as e:
            return f"{what}: HTTP {e.code}{_code(e)}"
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
        return f"{what}: {type(e).__name__}: {e}"


def reading(metric: str, value: float | None, *, time: str, why: str | None = None, mark: str = "measured",
            **optional: object) -> dict:
    """One reading of the door's shape; an optional field that is None is left out."""
    r = {"metric": metric, "value": value, **({"why": why} if why else {}), "mark": mark, "time": time}
    return {**r, **{k: v for k, v in optional.items() if v is not None}}


def _refused(r: object) -> str | None:
    if not isinstance(r, dict):
        return "a reading is not a JSON object"
    m = r.get("metric", "?")
    if unknown := sorted(set(r) - set(KEYS)):
        return f"{m}: unknown key {', '.join(unknown)}, which the door refuses"
    if not all(isinstance(r.get(k), str) and r[k] for k in ("metric", "time")):
        return f"{m}: a reading needs its metric and the time its number was read"
    if r.get("mark") not in MARKS:
        return f"{m}: mark is {r.get('mark')!r}, not one of {', '.join(MARKS)}"
    v = r.get("value")
    if v is None and not r.get("why"):
        return f"{m}: a null value needs its why"
    if v is not None and (isinstance(v, bool) or not isinstance(v, int | float)):
        return f"{m}: value is {v!r}, not a number or null"
    if not (r.get("run") or r.get("command")):
        return f"{m}: names neither the run that recorded it nor the command that computed it"
    if r.get("commit") is not None and not (isinstance(r["commit"], str) and COMMIT.fullmatch(r["commit"])):
        return f"{m}: commit is {r['commit']!r}, not a commit's hex"
    return None


def send(readings: list[dict], *, command: str | None, commit: str | None, url: str | None = None, key: Path | None = None,
         dry_run: bool = False, post=None, out=print) -> str | None:
    """POST ``readings`` to the door, each naming ``command`` and ``commit`` where given, with the panels key: None when the
    site took them (or, with ``dry_run``, when the request is printed with the key left out and nothing sent), else why not,
    in one line. Never raises on the network or a missing key, so a loop goes on. ``url`` may only be on the site's own
    host."""
    url, key = url or METRICS, key or PANELS_KEY
    body = {"readings": [{**r, **({"command": command} if command else {}), **({"commit": commit} if commit else {})}
                         for r in readings]}
    what = f"POST {url}"
    if not body["readings"]:
        return f"{what}: no reading to send"
    if bad := [w for r in body["readings"] if (w := _refused(r))]:
        return f"{what}: nothing sent: {'; '.join(bad)}"
    if why := off_site(url):
        return f"{what}: {why}"
    if dry_run:
        held = "the panels key in" if key.is_file() else "no key: a send would refuse, none at"
        out(f"{what}\nAuthorization: Bearer <{held} {key}>\nContent-Type: application/json\n\n{json.dumps(body, indent=1)}")
        return None
    try:
        secret = key.read_text().strip()
    except OSError as e:
        return f"{what}: no panels key at {key} ({type(e).__name__}): nothing sent"
    return (post or send_json)("POST", url, body, secret, what)


def _given(ns: argparse.Namespace, stdin) -> list[dict]:
    if ns.metric:
        return [reading(ns.metric, ns.value, time=ns.time, why=ns.why, mark=ns.mark, target=ns.target, period=ns.period,
                        over=ns.over, member=ns.member, run=ns.run)]
    try:
        got = json.load(stdin)
    except ValueError as e:
        raise SlackError(f"stdin is not JSON ({e}): give --metric, or a reading or a list of them on stdin") from None
    return [got] if isinstance(got, dict) else got if isinstance(got, list) else [got]


class Route:
    def __init__(self, url: str = METRICS, key: str | None = None) -> None:
        self.base = url
        self.key = key if url == METRICS else None
        self._oidc = None if self.key else _Identity(METRICS)

    def call(self) -> dict:
        what = f"GET {self.base}"

        def get(token: str) -> object:
            return http_json("GET", self.base, None, {"Authorization": f"Bearer {token}"})

        try:
            out = get(self.key) if self.key else self._oidc.send(get)
        except urllib.error.HTTPError as e:
            if self.key and e.code == 401:
                raise RouteError(f"{what}: the route rejected the site key in ${TOKEN_ENV} or {TOKEN_FILE} "
                                 f"(HTTP 401{_code(e)}): it takes a live key with events:read", 401) from None
            raise RouteError(route_error(what, e), e.code) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SlackError(f"{what}: the route is unreachable: {getattr(e, 'reason', e)}") from None
        if not isinstance(out, dict) or not isinstance(out.get("metrics"), list):
            raise SlackError(f"{what}: the answer has no metrics list")
        return out


def _key() -> str | None:
    """`research jobs`' token when it is a site key; an older token, without KEY_PREFIX, isn't sent."""
    key = kept_token()
    return key if key and key.startswith(KEY_PREFIX) else None


def _num(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3g}" if v and (abs(v) < 0.01 or abs(v) >= 1e6) else f"{v:,.2f}".rstrip("0").rstrip(".")


def _change(v: float | None) -> str:
    return "—" if v is None else ("+" if v > 0 else "") + _num(v)


def _age(ms: float | None) -> str:
    if ms is None:
        return "—"
    h = ms / 3_600_000
    return f"{h:.0f} h" if h < 48 else f"{h / 24:.0f} d"


def _standing(m: dict) -> str:
    if m.get("pastFloor") is not None:
        return f"past its floor {_num(m['floor'])}" if m["pastFloor"] else "within its floor"
    latest = m.get("latest") or {}
    if not latest.get("verdict"):
        return ""
    return latest["verdict"] + (f" (target {_num(latest['target'])})" if latest.get("target") is not None else "")


def show(report: dict) -> str:
    lines = []
    for m in report["metrics"]:
        latest = m.get("latest")
        head = f"{m['id']}  @{m['owner']}  {m['title']}"
        if not latest:
            lines += [head, f"  not recorded yet; measured by: {m['command']}"]
            continue
        ci = latest.get("ci95")
        value = _num(latest["value"]) + (f" [{_num(ci[0])}, {_num(ci[1])}]" if ci else "") + f" {m['unit']}"
        stale = ", stale" if m.get("stale") else ""
        lines += [head,
                  f"  {value}  7d {_change(m.get('change7d'))}  30d {_change(m.get('change30d'))}  age {_age(m.get('ageMs'))}"
                  f" (every {m['every']}{stale})  {_standing(m)}".rstrip(),
                  f"  run {latest['run']}" + (f" at {latest['commit'][:12]}" if latest.get("commit") else "")]
    if not report["metrics"]:
        lines.append("no metric is declared yet")
    lines += [f"refused: {r['id']}: {r['why']}" for r in report.get("refused") or []]
    if report.get("why"):
        lines.append(f"note: {report['why']}")
    return "\n".join(lines)


def main(argv: list[str], route: Route | None = None, stdin=None) -> int:
    p = argparse.ArgumentParser(prog="research metrics", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=METRICS, help="the route (default: the console's); only the console's gets a site key, "
                                                  "and `send` refuses any host but the site's")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("ids", nargs="*", metavar="ID")
    s.add_argument("--owner", action="append", default=[], metavar="HANDLE", help="only this lead's; repeat for more")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("send", help="send readings with the panels key; with no --metric, a JSON reading or list on stdin")
    s.add_argument("--command", help="the script that computed them, as its metrics.toml row names it")
    s.add_argument("--commit", help="the commit of the copy that runs (its COMMIT file)")
    s.add_argument("--dry-run", action="store_true", help="print the request without the key and send nothing")
    s.add_argument("--metric", metavar="ID")
    s.add_argument("--value", type=float)
    s.add_argument("--why", help="why the value is null")
    s.add_argument("--time", help="when the number was read, ISO 8601 UTC")
    s.add_argument("--mark", default="measured", choices=MARKS)
    s.add_argument("--target", type=float)
    for k in ("period", "over", "member", "run"):
        s.add_argument(f"--{k}")
    ns = p.parse_args(argv)
    if ns.cmd == "send":
        try:
            readings = _given(ns, stdin or sys.stdin)
        except SlackError as e:
            print(f"research metrics: {e}")
            return 1
        if why := send(readings, command=ns.command, commit=ns.commit, url=ns.url, dry_run=ns.dry_run):
            print(f"research metrics: {why}")
            return 1
        if not ns.dry_run:
            print(f"research metrics: sent {len(readings)} reading(s) to {ns.url}")
        return 0
    try:
        report = (route or Route(ns.url, _key())).call()
        missing = [i for i in ns.ids if i not in {m["id"] for m in report["metrics"]}]
        if missing:
            raise SlackError(f"not declared in any metrics.toml on main: {', '.join(missing)} (research metrics show lists "
                             "them; benchmarks/README.md's \"Metrics\" says where a row goes)")
        report["metrics"] = [m for m in report["metrics"]
                             if (not ns.ids or m["id"] in ns.ids) and (not ns.owner or m["owner"] in ns.owner)]
        print(json.dumps(report, indent=1, ensure_ascii=False) if ns.json else show(report))
        return 0
    except (RouteError, SlackError) as e:
        print(f"research metrics: {e}")
        return 1
