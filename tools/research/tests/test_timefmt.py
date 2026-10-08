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


def _token(iso: str, fallback: str | None = None) -> str:
    u = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return f"<!date^{int(u.timestamp())}^{{date_short_pretty}} {{time}}|{fallback or both(u)}>"


def test_slack_dates_rewrites_iso_times():
    text = "thaw 2026-11-01T09:30Z, recheck 2026-11-01T08:30:15Z."
    assert slack_dates(text, NOW) == f"thaw {_token('2026-11-01T09:30Z')}, recheck {_token('2026-11-01T08:30:15Z')}."


@pytest.mark.parametrize("text", [
    "freeze at 22:00Z; by 9:05Z.",
    "leases are clamped to 7:55 AM PDT (14:55Z), today",            # a time with no date names no day
])
def test_slack_dates_leaves_a_time_without_a_date_as_written(text):
    assert slack_dates(text, NOW) == text


@pytest.mark.parametrize("now, text, want", [
    (NOW, "on 14 Oct 14:55Z", f"on {_token('2026-10-14T14:55Z', '14 Oct 7:55 AM PDT (14:55Z)')}"),
    (NOW, "on Oct 14 at 14:55Z", f"on {_token('2026-10-14T14:55Z', '14 Oct 7:55 AM PDT (14:55Z)')}"),
    (NOW, "Oct 14th, 14:55Z", _token("2026-10-14T14:55Z", "14 Oct 7:55 AM PDT (14:55Z)")),
    (NOW, "Oct 14, 2027 14:55Z", _token("2027-10-14T14:55Z", "14 Oct 7:55 AM PDT (14:55Z)")),
    (NOW, "you mean 14 Oct 7:55 AM PDT (14:55Z)",
     f"you mean {_token('2026-10-14T14:55Z', '14 Oct 7:55 AM PDT (14:55Z)')}"),
    (NOW, "30 Sep 11:05 PM PDT (06:05Z)",                           # the date before a Pacific time is the Pacific date
     _token("2026-10-01T06:05Z", "30 Sep 11:05 PM PDT (06:05Z)")),
    (dt.datetime(2026, 12, 31, 23, 0, tzinfo=UTC), "3 Jan 01:00Z",  # the year nearest now
     _token("2027-01-03T01:00Z", "2 Jan 5:00 PM PST (01:00Z)")),
])
def test_slack_dates_uses_the_date_written_before_the_time(now, text, want):
    assert slack_dates(text, now) == want


@pytest.mark.parametrize("text", [
    "14 Oct 7:55 AM PDT (15:55Z)",                                  # the two zones disagree
    "31 Feb 10:00Z",
])
def test_slack_dates_leaves_an_impossible_dated_time_as_written(text):
    assert slack_dates(text, NOW) == text


@pytest.mark.parametrize("text", [
    "note:20260930T2020Z-handoff-infra and lanes/x/20260930T2020Z-report-x.md",
    "run `research slack read --since 22:00Z` first",
    "```\nat 14 Oct 22:00Z\n2026-09-30T20:41Z\n```",
    "see https://example.com/log?at=2026-09-30T20:41Z#22:00Z",
    "<https://example.com/22:00Z|the 14 Oct 22:00Z log>",
    "<!date^1790000000^{date_short_pretty} {time}|1:41 PM PDT (20:41Z)>",
    "2026-09-30T20:41:05.123Z 12:00:00Z 24:00Z 20:41Z-report 20:41Zulu",
])
def test_slack_dates_leaves_ids_code_links_and_non_times_alone(text):
    assert slack_dates(text, NOW) == text


def test_slack_dates_rewrites_around_protected_spans():
    assert (slack_dates("`2026-09-30T22:00Z` then 2026-09-30T22:00Z", NOW)
            == f"`2026-09-30T22:00Z` then {_token('2026-09-30T22:00Z')}")


def test_research_imports_without_a_tz_database_and_prints_utc_alone():
    """`research run` wraps every Nebius job, whose image has neither /usr/share/zoneinfo nor the tzdata package."""
    code = ("import sys, zoneinfo; sys.modules['tzdata'] = None; zoneinfo.reset_tzpath(()); "
            "import research.cli; from research.timefmt import both; print(both('2026-09-30T20:41Z'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out == "20:41Z\n"
