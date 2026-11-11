"""`research metrics`: the console's Metrics tab (/console/metrics) as text, from the site's GET /api/metrics.

  research metrics show [ID ...] [--owner HANDLE] [--json]   each metric benchmarks/metrics.toml declares: its latest
                                                            value, changes over 7 and 30 days, age and standing

The page and this command read one door. It answers a site key with events:read (`$JOBS_TOKEN`, else
~/.research/jobs/token, as `research jobs` reads it) or else this agent's Cursor identity minted for the route.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error

from .jobs.client import TOKEN_ENV, TOKEN_FILE, load_token
from .slack import RouteError, SlackError, _Identity, http_json, route_error

METRICS = "https://website-docs-sage.vercel.app/api/metrics"   # the route, and its OIDC token's audience


class Route:
    def __init__(self, url: str = METRICS, key: str | None = None) -> None:
        self.base, self.key = url, key
        self._oidc = None if key else _Identity(METRICS)

    def call(self) -> dict:
        what = f"GET {METRICS}"

        def get(token: str) -> object:
            return http_json("GET", self.base, None, {"Authorization": f"Bearer {token}"})

        try:
            out = get(self.key) if self.key else self._oidc.send(get)
        except urllib.error.HTTPError as e:
            raise RouteError(route_error(what, e), e.code) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SlackError(f"{what}: the route is unreachable: {getattr(e, 'reason', e)}") from None
        if not isinstance(out, dict) or not isinstance(out.get("metrics"), list):
            raise SlackError(f"{what}: the answer has no metrics list")
        return out


def _key() -> str | None:
    return load_token() if os.environ.get(TOKEN_ENV) or TOKEN_FILE.exists() else None


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


def main(argv: list[str], route: Route | None = None) -> int:
    p = argparse.ArgumentParser(prog="research metrics", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=METRICS, help="the route (default: the console's)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("ids", nargs="*", metavar="ID")
    s.add_argument("--owner", action="append", default=[], metavar="HANDLE", help="only this lead's; repeat for more")
    s.add_argument("--json", action="store_true")
    ns = p.parse_args(argv)
    try:
        report = (route or Route(ns.url, _key())).call()
        missing = [i for i in ns.ids if i not in {m["id"] for m in report["metrics"]}]
        if missing:
            raise SlackError(f"not declared in benchmarks/metrics.toml: {', '.join(missing)} (research metrics show lists them)")
        report["metrics"] = [m for m in report["metrics"]
                             if (not ns.ids or m["id"] in ns.ids) and (not ns.owner or m["owner"] in ns.owner)]
        print(json.dumps(report, indent=1, ensure_ascii=False) if ns.json else show(report))
        return 0
    except (RouteError, SlackError) as e:
        print(f"research metrics: {e}")
        return 1
