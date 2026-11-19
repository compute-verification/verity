#!/usr/bin/env python3
"""Verity's panels for the site's live console (`/admin/live`), recomputed from their live sources and sent in the site's panel
format (`apps/docs/lib/panels/format.ts` in the website repository).

    verity_console.py --group store [--store DIR] [--research-src DIR] [--out DIR] [--key FILE] [--no-refresh]

- `store`, on the control pod: the evidence store's `check` Attempts of the last day and their `landed` labels. The rest of the
  store's results are the site's Results view, which reads the store's index.

Every panel is checked against the format (fields, lengths, cell types, 64 KiB) and written to OUT/<name>.json. When the key file
exists (default ~/.config/verity/panels.key) each valid panel is also PUT to the site; without it the run is a dry run. A panel
whose source fails this run is neither written nor sent, so the site keeps the last good one. Prints one JSON line per run; exit 1
when a panel failed, was invalid, or wasn't accepted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://website-docs-sage.vercel.app"
FIELDS = ("id", "title", "kind", "unit", "y_scale", "target", "target_label", "stacked", "note", "updated_at", "source", "columns", "rows")
REQUIRED = {"id", "title", "kind", "updated_at", "source", "columns", "rows"}
MAX_BODY = 64 * 1024
ID = re.compile(r"[a-z0-9-]+/[a-z0-9-]+")
ISO = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d(:\d\d(\.\d+)?)?(Z|[+-]\d\d:\d\d)")


def iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def isoz(s: str | None) -> str | None:
    """A store timestamp (`…T12:54:39.764Z`) in the format's form, to the second."""
    if not s:
        return None
    try:
        return iso(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError as e:
        print(f"verity_console: store time {s[:40]!r} does not parse, so it is shown as none ({e})", file=sys.stderr)
        return None


def sig(v, n: int = 4):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(f"{v:.{n}g}")


def panel(name: str, title: str, kind: str, columns: list, rows: list, *, source: str, updated_at: str, unit: str | None = None,
          note: str | None = None, y_scale: str | None = None, producer: str = "verity", target: float | None = None,
          target_label: str | None = None, stacked: bool | None = None) -> dict:
    p = {"id": f"{producer}/{name}", "title": title, "kind": kind, "unit": unit, "y_scale": y_scale, "target": target,
         "target_label": target_label, "stacked": stacked, "note": note,
         "updated_at": updated_at, "source": source, "columns": columns, "rows": rows}
    return {k: p[k] for k in FIELDS if p[k] is not None}


def problems(p: dict) -> list[str]:
    """Everything the site's validator would refuse (panel-format.md §1)."""
    out = [f"unknown field `{k}`" for k in p if k not in FIELDS]
    out += [f"missing `{k}`" for k in sorted(REQUIRED - set(p))]
    if out:
        return out

    def text(k: str, hi: int) -> None:
        if k in p and not (isinstance(p[k], str) and 1 <= len(p[k]) <= hi):
            out.append(f"`{k}` must be 1 to {hi} characters")

    if not (isinstance(p["id"], str) and ID.fullmatch(p["id"])):
        out.append("`id` must be producer/name in lowercase letters, digits and dashes")
    text("title", 120), text("unit", 20), text("note", 500), text("source", 300)
    if p["kind"] not in ("table", "line", "bar"):
        out.append("`kind` must be table, line or bar")
    if "y_scale" in p and (p["y_scale"] not in ("linear", "log") or p["kind"] == "table"):
        out.append("`y_scale` must be linear or log, on a line or bar chart")
    if "target" in p and (p["kind"] == "table" or isinstance(p["target"], bool) or not isinstance(p["target"], (int, float))):
        out.append("`target` must be a number, on a line or bar chart")
    if "target_label" in p and ("target" not in p or not (isinstance(p["target_label"], str) and 1 <= len(p["target_label"]) <= 40)):
        out.append("`target_label` must be 1 to 40 characters, with a target")
    if "stacked" in p and (p["kind"] != "bar" or not isinstance(p["stacked"], bool)):
        out.append("`stacked` must be true or false, on a bar chart")
    if not (isinstance(p["updated_at"], str) and ISO.fullmatch(p["updated_at"])):
        out.append("`updated_at` must be ISO 8601 with a zone")
    cols, rows = p["columns"], p["rows"]
    if not (isinstance(cols, list) and 1 <= len(cols) <= 20 and all(isinstance(c, str) and 1 <= len(c) <= 60 for c in cols)
            and len(set(cols)) == len(cols)):
        out.append("`columns` must be 1 to 20 distinct names of up to 60 characters")
    elif p["kind"] != "table" and len(cols) < 2:
        out.append("a chart needs at least 2 columns")
    if not isinstance(rows, list) or len(rows) > 1000:
        out.append("`rows` must be a list of at most 1000 rows")
    else:
        for i, r in enumerate(rows):
            if not isinstance(r, list) or len(r) != len(cols):
                out.append(f"row {i} has {len(r) if isinstance(r, list) else '?'} cells, not {len(cols)}")
                continue
            for j, c in enumerate(r):
                ok = c is None or (isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)) or (
                    isinstance(c, str) and len(c) <= 200 and (j == 0 or p["kind"] == "table"))
                if not ok:
                    out.append(f"row {i} column {j}: {'a series cell must be a number or null' if isinstance(c, str) else 'bad cell'}")
    size = len(json.dumps(p).encode())
    if size > MAX_BODY:
        out.append(f"{size} bytes, over 64 KiB")
    return out


def fit(p: dict) -> dict:
    """A table over 64 KiB: its long strings cut shorter, then its rows cut, saying so in `note`."""
    if p["kind"] != "table":
        return p
    for width in (200, 140, 100, 70):
        if len(json.dumps(p).encode()) <= MAX_BODY:
            return p
        p["rows"] = [[c[:width - 1] + "…" if isinstance(c, str) and len(c) > width else c for c in r] for r in p["rows"]]
    n, note = len(p["rows"]), p.get("note", "")
    while len(json.dumps(p).encode()) > MAX_BODY and p["rows"]:
        p["rows"] = p["rows"][:int(len(p["rows"]) * 0.9)]
        p["note"] = (f"The first {len(p['rows'])} of {n} rows (the site's 64 KiB limit). " + note)[:500]
    return p


def publish(p: dict, url: str, key: str) -> str:
    if (scheme := urllib.parse.urlsplit(url).scheme) not in ("http", "https"):
        return f"refused: --url is not an http or https URL (scheme {scheme!r}): nothing sent"
    req = urllib.request.Request(  # noqa: S310 (http or https, checked above)
        f"{url}/api/panels/{p['id']}", data=json.dumps(p).encode(), method="PUT",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (http or https, checked above)
                return str(r.status)
        except urllib.error.HTTPError as e:
            if e.code != 503 or attempt == 2:
                return f"{e.code} {e.read()[:300].decode(errors='replace')}"
        except (urllib.error.URLError, OSError) as e:
            if attempt == 2:
                return f"unreachable: {e}"
        time.sleep(5)
    return "unreachable"


def write(out: Path, p: dict) -> None:
    dest = out / f"{p['id'].split('/', 1)[1]}.json"
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_text(json.dumps(p, indent=1))
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------------------------------------------- store group

def store_panels(ns, now: datetime, errors: list) -> tuple[list[dict], dict]:
    sys.path[:0] = [str(ns.research_src)]
    from research.store.local import LocalStore

    store = LocalStore(ns.store) if ns.store else LocalStore()
    info = {"refresh": "skipped (--no-refresh)"}
    if not ns.no_refresh:
        try:
            rep = store.refresh(jobs=8)
            info["refresh"] = f"+{rep['attempts']} attempts, +{rep['labels']} labels, {len(rep['errors'])} errors, {rep['seconds']:.0f} s"
        except Exception as e:  # noqa: BLE001  a stale catalog still renders; the run line says so
            info["refresh"] = f"failed: {type(e).__name__}: {e}"
    try:
        return [check_panel(store.index, now)], info
    except Exception as e:  # noqa: BLE001  the run line names it
        errors.append(f"merge-checks: {type(e).__name__}: {e}")
        return [], info


def machine_of(doc: dict) -> str | None:
    ex = doc.get("execution") or {}
    return ex.get("machine") or (ex.get("remote") or {}).get("machine") or ex.get("host")


def check_panel(idx, now: datetime) -> dict:
    """Every `check` Attempt of the last 24 hours, with the main commit that landed it from its `landed` label (`merge.label_landed`)."""
    since = iso(now - timedelta(hours=24))
    landed = dict(idx.sql("SELECT target, value FROM labels WHERE key = 'landed' AND by = 'research-merge' ORDER BY ts"))
    rows, ends = [], []
    for run, start, wall, rc, state, commit, mj in idx.sql(
            "SELECT id, start_utc, wall_s, rc, state, source_commit, manifest_json FROM attempts WHERE tool = 'check' AND start_utc >= ? "
            "ORDER BY start_utc DESC", (since,)):
        result = "pass" if state == "done" and rc == 0 else state if state in ("running", "cancelled") else f"fail (rc {rc})"
        rows.append([run, isoz(start), sig(wall / 60, 3) if wall else None, result, (commit or "")[:8] or None,
                     (landed.get(run) or "")[:9] or None, machine_of(json.loads(mj)) if mj else None])
        if start and wall:
            ends.append(datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(seconds=wall))
    return panel("merge-checks", "Every `check` in the last 24 hours", "table",
                 ["Run", "Started (UTC)", "Minutes", "Result", "Commit", "Landed as", "Machine"], rows,
                 updated_at=iso(max(ends)) if ends else iso(now), source="evidence store: tool=check Attempts and their landed labels",
                 note="Landed as is main's commit whose Check: trailer names the run (its `landed` label); the others are checks of "
                      "branches or of candidates that didn't land. Attempts are published when a check ends, so one in progress isn't listed.")


# ---------------------------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", choices=("store",), required=True)
    ap.add_argument("--out", type=Path, default=HERE / "outbox", help="where each panel's JSON is written")
    ap.add_argument("--key", type=Path, default=Path("~/.config/verity/panels.key").expanduser(),
                    help="the panels:write key; absent means a dry run")
    ap.add_argument("--url", default=os.environ.get("PANELS_URL", URL))
    ap.add_argument("--store", type=Path, help="the evidence store's local root (default: $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--research-src", type=Path, default=HERE / "research-src", help="a directory holding the `research` package")
    ap.add_argument("--no-refresh", action="store_true", help="read the store's catalog as it is")
    ns = ap.parse_args(argv)
    t0, now, errors = time.time(), datetime.now(UTC).replace(microsecond=0), []
    ns.out.mkdir(parents=True, exist_ok=True)
    try:
        panels, info = store_panels(ns, now, errors)
    except Exception as e:  # noqa: BLE001  nothing is sent; the site keeps the last good panels
        panels, info = [], {}
        errors.append(f"{ns.group}: {type(e).__name__}: {e} | {traceback.format_exc(limit=3).splitlines()[-3:]}")
    key = ns.key.read_text().strip() if ns.key.is_file() else None
    sent: dict[str, str] = {}
    for p in panels:
        bad = problems(p)
        if bad:
            errors.append(f"{p.get('id')}: invalid: {'; '.join(bad)}")
            continue
        write(ns.out, p)
        if key:
            sent[p["id"]] = publish(p, ns.url, key)
    refused = {k: v for k, v in sent.items() if v not in ("200", "201")}
    line = {"at": iso(now), "group": ns.group, "seconds": round(time.time() - t0, 1), "panels": len(panels),
            "published": (f"{len(sent) - len(refused)} of {len(sent)}" if key else "dry run (no key)"), **info,
            "refused": refused or None, "errors": errors or None}
    (ns.out / f"_last-run-{ns.group}.json").write_text(json.dumps(line, indent=1))
    print(json.dumps(line), flush=True)
    return 1 if errors or refused else 0


if __name__ == "__main__":
    sys.exit(main())
