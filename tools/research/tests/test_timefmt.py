from __future__ import annotations

import datetime as dt
import subprocess
import sys

import pytest

from research.timefmt import both, slack_dates, stamp_ahead

UTC = dt.UTC
NOW = dt.datetime(2026, 9, 30, 20, 41, tzinfo=UTC)


@pytest.mark.parametrize("t, want", [
    (NOW, "1:41 PM PDT (20:41Z)"),
    ("2026-09-30T20:41Z", "1:41 PM PDT (20:41Z)"),
    ("2026-09-30T20:41:05.123Z", "1:41 PM PDT (20:41Z)"),
    (dt.datetime(2026, 9, 30, 20, 41), "1:41 PM PDT (20:41Z)"),                         # naive is UTC
    (NOW.timestamp(), "1:41 PM PDT (20:41Z)"),
    (f"{NOW.timestamp():.6f}", "1:41 PM PDT (20:41Z)"),                                   # a Slack ts
    ("2026-11-01T08:30Z", "1:30 AM PDT (08:30Z)"),                                        # the day DST ends: the last PDT hour
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
    (NOW, "due Tue 3 Nov 02:00Z", f"due {_token('2026-11-03T02:00Z', 'Mon 2 Nov 6:00 PM PST (02:00Z)')}"),  # across midnight
    (NOW, "Wednesday, Oct 14, 14:55Z", _token("2026-10-14T14:55Z", "Wed 14 Oct 7:55 AM PDT (14:55Z)")),
])
def test_slack_dates_uses_the_date_written_before_the_time(now, text, want):
    assert slack_dates(text, now) == want


@pytest.mark.parametrize("text", [
    "14 Oct 7:55 AM PDT (15:55Z)",                                  # the two zones disagree
    "31 Feb 10:00Z",
    "Tue 22:00Z",                                                    # a weekday is no date
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


SUN_0225 = dt.datetime(2026, 11, 15, 10, 25, 30, tzinfo=UTC)                           # Sun 15 Nov, 2:25:30 AM PST
SAT_2358 = dt.datetime(2026, 11, 15, 7, 58, 10, tzinfo=UTC)                            # Sat 14 Nov, 11:58 PM PST
SUN_0001 = dt.datetime(2026, 11, 15, 8, 1, 10, tzinfo=UTC)                             # Sun 15 Nov, 12:01 AM PST
COMING = "leave the time out, or, if it is a time to come, put `by`, `at` or `ETA` before it"
PT_FIX = "write the time `TZ=America/Los_Angeles date` gives now, " + COMING
Z_FIX = "write the time `date -u` gives now, " + COMING
SUN_0820 = dt.datetime(2026, 11, 15, 16, 20, 5, tzinfo=UTC)                            # Sun 15 Nov, 8:20 AM PST: comms' probes


@pytest.mark.parametrize("now, text, says", [
    (SUN_0225, "lean (lead), Sun 15 Nov 2:28 AM PST: the run is up",
     "the first line says Sun 15 Nov 2:28 AM PST and it is Sun 15 Nov 2:25 AM PST: " + PT_FIX),
    (SUN_0225, "lean/AREAS, 2:26 AM PST: one minute ahead", "the first line says 2:26 AM PST and it is 2:25 AM PST: " + PT_FIX),
    (SUN_0225, "lean (lead), 2:28 AM PST (10:28Z): x", "the first line says 2:28 AM PST and it is 2:25 AM PST: " + PT_FIX),
    (SUN_0225, "ci 10:26Z: posts", "the first line says 10:26Z and it is 10:25Z: " + Z_FIX),
    (SUN_0225, "ci, 15 Nov 10:40 UTC: x", "the first line says 15 Nov 10:40 UTC and it is 15 Nov 10:25Z: " + Z_FIX),
    (SUN_0225, "lean, Mon 16 Nov 1:00 AM PST: x",
     "the first line says Mon 16 Nov 1:00 AM PST and it is Sun 15 Nov 2:25 AM PST: " + PT_FIX),
    (SAT_2358, "lean (lead), 12:01 AM PST: x", "the first line says 12:01 AM PST and it is 11:58 PM PST: " + PT_FIX),
    (dt.datetime(2026, 11, 15, 23, 58, 10, tzinfo=UTC), "ci 00:01Z: x",
     "the first line says 00:01Z and it is 23:58Z: " + Z_FIX),
    (dt.datetime(2026, 11, 15, 10, 25, 20, tzinfo=UTC), "@top lean (lead), Sun 15 Nov 2:28 AM PST, to ci's 2:25 AM PST.",
     "the first line says Sun 15 Nov 2:28 AM PST and it is Sun 15 Nov 2:25 AM PST: " + PT_FIX),
    (dt.datetime(2026, 11, 15, 10, 7, 40, tzinfo=UTC), "@top @lean ci, Sun 15 Nov 2:24 AM PST, to your 1:55 AM item 2.",
     "the first line says Sun 15 Nov 2:24 AM PST and it is Sun 15 Nov 2:07 AM PST: " + PT_FIX),
    (dt.datetime(2026, 11, 15, 7, 18, 30, tzinfo=UTC), "lean's worker AREAS, Sat 14 Nov 11:19 PM PST",
     "the first line says Sat 14 Nov 11:19 PM PST and it is Sat 14 Nov 11:18 PM PST: " + PT_FIX),
    (dt.datetime(2026, 11, 15, 9, 55, 10, tzinfo=UTC), "Coordinator to TRAIN, Sun 15 Nov 2:05 AM PST.",
     "the first line says Sun 15 Nov 2:05 AM PST and it is Sun 15 Nov 1:55 AM PST: " + PT_FIX),
    (dt.datetime(2026, 11, 15, 21, 0, tzinfo=UTC), "node 2 reboot, 22:00Z: drain it now",   # a header time still can be to come
     "the first line says 22:00Z and it is 21:00Z: " + Z_FIX),
])
def test_a_first_line_stamped_ahead_of_the_clock_is_refused_with_the_stamp_the_clock_and_the_fix(now, text, says):
    """The lean lane's stamps ran up to 17 minutes ahead of the clock, kept right only by hand: one minute ahead is refused."""
    assert stamp_ahead(text, now) == says


@pytest.mark.parametrize("now, text", [
    (SUN_0225, "lean (lead), 2:25 AM PST: the clock's own minute"),
    (SUN_0225, "lean (lead), Sun 15 Nov 2:24 AM PST: behind"),
    (SUN_0225, "lean, 1:55 AM PST (09:55Z): behind"),
    (SUN_0225, "ci 10:25Z: the clock's own minute"),
    (SUN_0225, "lean/STAMP: the PR is up, no time"),
    (SUN_0225, "lean 2:2x AM PST: minutes not known"),
    (SUN_0225, "lean, 2:2x AM PST: the run ends 3:00 AM PST"),            # the first clock time can't be read: no later one
    (SUN_0225, "the run finishes by 3:00 AM PST"), (SUN_0225, "lean: ETA 4:00 AM PST"), (SUN_0225, "node 2 reboots at 22:00Z"),
    (SUN_0225, "lean/STAMP, the PR is up\n3:00 AM PST on the second line"),
    (SUN_0225, "lean 13:05 PM PST: no such hour"), (SUN_0225, "ci 25:00Z: no such hour"), (SUN_0225, "ci 10:61Z: no such minute"),
    (SUN_0225, "lean, 31 Nov 3:00 AM PST: no such day"), (SUN_0225, "ci, 15 Nov 9999 23:00Z: past the calendar"),
    (SUN_0225, "freeze 2026-11-15T12:00Z"), (SUN_0225, "lean, Sat 14 Nov 11:59 PM PST: yesterday"),
    (SUN_0001, "lean (lead), 11:59 PM PST: last night, across midnight"),
    (dt.datetime(2026, 11, 16, 0, 1, 10, tzinfo=UTC), "ci 23:59Z: across UTC midnight"),
    (SUN_0820, "the review is due 8:30 AM PST today"), (SUN_0820, "the window closes Sun 15 Nov 10:30 AM PST"),
    (SUN_0820, "check starts 20:00Z"), (SUN_0820, "the window closes by Sun 15 Nov 10:30 AM PST"),
    (dt.datetime(2026, 11, 15, 21, 0, tzinfo=UTC), "node 2 reboots at 22:00Z: drain it now"),
    (dt.datetime(2026, 11, 15, 19, 0, tzinfo=UTC), "<!subteam^S1|@lean> *Agent watch:* ROWS "
     "(<https://cursor.com/agents/bc-1|bc-1>) stopped: \u201cwait before the 20:30Z row\u201d"),
    (dt.datetime(2026, 11, 14, 23, 0, tzinfo=UTC), "lean: an ask from ci has waited over 4 h with no answer: "
     "\u201cnode A booked Sun 00:30Z, take B\u201d"),
    (dt.datetime(2026, 11, 15, 4, 50, tzinfo=UTC), "<!subteam^S2|@infra> *Agent watch:* G2-1 "
     "(<https://cursor.com/agents/bc-2|bc-2>) went idle: \u201chold the node before the 05:10Z go\u201d"),
    (dt.datetime(2026, 11, 15, 16, 0, tzinfo=UTC), "@lean ci, held for your 8:30 AM PST stop"),
    (dt.datetime(2026, 11, 15, 16, 0, tzinfo=UTC), "the merge train is held for your 8:30 AM PST stop."),
])
def test_a_stamp_at_or_behind_the_clock_no_stamp_and_a_doubt_all_pass(now, text):
    assert stamp_ahead(text, now) is None


SUN_1257 = dt.datetime(2026, 11, 15, 20, 57, 30, tzinfo=UTC)                           # Sun 15 Nov, 12:57 PM PST
HEADINGS = [f"{b}{h}{b}{s}" for h in ("FOR THE COORDINATOR", "FOR THE LEAN LANE'S COORDINATOR")
            for b in ("", "**") for s in (" \u2014", " \u2013", " -", ":")] + ["**FOR THE COORDINATOR:**"]


@pytest.mark.parametrize("heading", HEADINGS)
@pytest.mark.parametrize("name, rest", [("AREAS", ", the probe is up"), ("NAME", "."),
                                        ("lean's worker AREAS (lead) again", ", the probe is up")])
def test_a_heading_before_the_name_still_leaves_the_stamp_in_the_header(heading, name, rest):
    """Workers' briefs head a report `**FOR THE COORDINATOR** — NAME, <stamp>`: the dash or colon ended the header before
    the stamp, so 12:59 PM went out at 12:57. Up to 5 tokens on each side of the separator; any later time still passes."""
    text = f"{heading} {name}, Sun 15 Nov 12:59 PM PST{rest}"
    assert stamp_ahead(text, SUN_1257) == "the first line says Sun 15 Nov 12:59 PM PST and it is Sun 15 Nov 12:57 PM PST: " + PT_FIX
    assert stamp_ahead(text.replace("12:59", "12:57"), SUN_1257) is None


@pytest.mark.parametrize("text, stamp", [("FOR THE COORDINATOR \u2014 Sun 15 Nov 12:59 PM PST, the probe is up", "Sun 15 Nov 12:59 PM PST"),
                                         ("**FOR THE COORDINATOR:** 12:59 PM PST, the probe is up", "12:59 PM PST")])
def test_a_heading_then_the_stamp_with_no_name_between_is_a_header(text, stamp):
    assert stamp_ahead(text, SUN_1257).startswith(f"the first line says {stamp} and it is ")


@pytest.mark.parametrize("text", [
    "FOR THE LEAN LANE'S OWN COORDINATOR \u2014 AREAS, Sun 15 Nov 12:59 PM PST.",          # 6 heading tokens: no header
    "FOR THE COORDINATOR \u2014 lean's worker AREAS (lead) once again, Sun 15 Nov 12:59 PM PST.",   # 6 name tokens
    "the window \u2014 closes by Sun 15 Nov 12:59 PM PST, then the lock", "ETA: 12:59 PM PST, then the lock",
    "*Agent watch:* run r20261115-144403-6f08 (r20261115-144403-6f08) of worker ROWS: rows \u2014 booked Sun 00:30Z, "
    "then rebook (<https://cursor.com/agents/bc-1|bc-1>) ended rc=0 after 5 min on vy-nebius-cpu-1",
    "<!subteam^S2|@infra> *Agent watch:* G2-1 \u2014 went idle: \u201chold the node before the 1:10 PM PST go\u201d",
    "lean: an ask from ci \u2014 waited over 4 h with no answer: \u201cnode A booked 1:30 PM PST, take B\u201d",
])
def test_a_dash_or_colon_past_the_header_or_with_no_name_after_it_makes_no_stamp(text):
    assert stamp_ahead(text, SUN_1257) is None


def test_a_heading_then_words_with_no_comma_before_a_time_is_no_header():
    """A correction quotes the earlier comment's stamp after its heading; that time is not this post's."""
    text = "RTCODE: my comment above is stamped 10:22 AM PST (18:22Z); `date -u` read 10:21 AM PST (18:21Z) when it was written."
    assert stamp_ahead(text, dt.datetime(2026, 11, 14, 18, 21, 30, tzinfo=UTC)) is None


def test_research_imports_without_a_tz_database_and_prints_utc_alone():
    """`research run` wraps every Nebius job, whose image has neither /usr/share/zoneinfo nor the tzdata package."""
    code = ("import sys, zoneinfo; sys.modules['tzdata'] = None; zoneinfo.reset_tzpath(()); "
            "import research.cli; from research.timefmt import both; print(both('2026-09-30T20:41Z'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out == "20:41Z\n"
