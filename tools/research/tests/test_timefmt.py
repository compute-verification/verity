from __future__ import annotations

import datetime as dt
import subprocess
import sys

import pytest

from research.timefmt import both, slack_dates

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 30, 20, 41, tzinfo=UTC)


@pytest.mark.parametrize("t, want", [
    (NOW, "1:41 PM PDT (20:41Z)"),
    ("2026-09-30T20:41Z", "1:41 PM PDT (20:41Z)"),
    ("2026-09-30T20:41:05.123Z", "1:41 PM PDT (20:41Z)"),
    (dt.datetime(2026, 9, 30, 20, 41), "1:41 PM PDT (20:41Z)"),                         # naive is UTC
    (NOW.timestamp(), "1:41 PM PDT (20:41Z)"),
    (f"{NOW.timestamp():.6f}", "1:41 PM PDT (20:41Z)"),                                   # a Slack ts
    ("2026-11-01T08:30Z", "1:30 AM PDT (08:30Z)"),                                        # 2026-11-01: the last PDT hour
    ("2026-11-01T09:30Z", "1:30 AM PST (09:30Z)"),                                        # the same wall time, an hour later
    ("2026-11-01T20:00Z", "12:00 PM PST (20:00Z)"),
    ("2026-10-01T06:05Z", "11:05 PM PDT (06:05Z)"),                                       # the Pacific day is the day before
])
def test_both_is_pacific_then_utc_across_the_dst_boundary(t, want):
    assert both(t) == want


def _token(iso: str) -> str:
    u = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return f"<!date^{int(u.timestamp())}^{{date_short_pretty}} {{time}}|{both(u)}>"


def test_slack_dates_rewrites_both_forms_with_hh_mm_on_today_utc():
    text = "freeze at 22:00Z, thaw 2026-11-01T09:30Z, recheck 2026-11-01T08:30:15Z; by 9:05Z."
    assert slack_dates(text, NOW) == (f"freeze at {_token('2026-09-30T22:00Z')}, thaw {_token('2026-11-01T09:30Z')}, "
                                      f"recheck {_token('2026-11-01T08:30:15Z')}; by {_token('2026-09-30T09:05Z')}.")
    assert slack_dates("22:00Z", dt.datetime(2026, 11, 1, 23, 0, tzinfo=UTC)).endswith("|2:00 PM PST (22:00Z)>")


@pytest.mark.parametrize("text", [
    "note:20260930T2020Z-handoff-infra and lanes/x/20260930T2020Z-report-x.md",
    "run `research slack read --since 22:00Z` first",
    "```\nat 22:00Z\n2026-09-30T20:41Z\n```",
    "see https://example.com/log?at=2026-09-30T20:41Z#22:00Z",
    "<https://example.com/22:00Z|the 22:00Z log>",
    "<!date^1790000000^{date_short_pretty} {time}|1:41 PM PDT (20:41Z)>",
    "2026-09-30T20:41:05.123Z 12:00:00Z 24:00Z 20:41Z-report 20:41Zulu",
])
def test_slack_dates_leaves_ids_code_links_and_non_times_alone(text):
    assert slack_dates(text, NOW) == text


def test_slack_dates_rewrites_around_protected_spans():
    assert slack_dates("`22:00Z` then 22:00Z", NOW) == f"`22:00Z` then {_token('2026-09-30T22:00Z')}"


def test_research_imports_without_a_tz_database_and_prints_utc_alone():
    """`research run` wraps every Nebius job, whose image has neither /usr/share/zoneinfo nor the tzdata package."""
    code = ("import sys, zoneinfo; sys.modules['tzdata'] = None; zoneinfo.reset_tzpath(()); "
            "import research.cli; from research.timefmt import both; print(both('2026-09-30T20:41Z'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out == "20:41Z\n"
