"""Times people read: Pacific first, then UTC (`1:41 PM PDT (20:41Z)`), so nobody converts by hand. Machine output (JSON, file
names, ids, front matter, log lines) stays UTC. `slack_dates` rewrites the dated UTC times in a Slack post into Slack's date token,
which shows each reader their own zone and falls back to `both`. Stdlib only."""
from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    PACIFIC: ZoneInfo | None = ZoneInfo("America/Los_Angeles")
except ZoneInfoNotFoundError:  # a host with no tz database (the Nebius job images): `both` prints UTC alone
    PACIFIC = None
UTC = dt.timezone.utc

#: left alone by `slack_dates`: code blocks and spans, Slack's <...> tokens (links, mentions, dates) and bare URLs
_PROTECTED = re.compile(r"```.*?```|`[^`\n]*`|<[^<>\n]*>|\bhttps?://\S+", re.S)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MON = r"(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = (rf"(?:(?P<d1>\d{{1,2}})\s+(?P<m1>{_MON})|(?P<m2>{_MON})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?)"
         rf"(?:,?\s+(?P<y>\d{{4}}))?,?\s+(?:at\s+)?")
_PAC = r"(?P<ph>\d{1,2}):(?P<pm>\d{2})\s*(?P<ap>[AP]M)\s+P[DS]T\s+\("
_UTC_TIME = re.compile(rf"(?<![\w:.-])(?:(?P<iso>(?P<Y>\d{{4}})-(?P<M>\d{{2}})-(?P<D>\d{{2}})T(?P<H>\d{{2}}):(?P<N>\d{{2}})(?::(?P<S>\d{{2}}))?Z)"
                       rf"|(?:{_DATE})?(?:{_PAC}(?P<zh>\d{{1,2}}):(?P<zm>\d{{2}})Z\)|(?P<bh>\d{{1,2}}):(?P<bm>\d{{2}})Z))(?![\w-])")


def _utc(t: dt.datetime | str | float) -> dt.datetime:
    if isinstance(t, (int, float)):
        return dt.datetime.fromtimestamp(t, UTC)
    if isinstance(t, str):
        s = t.strip()
        t = dt.datetime.fromtimestamp(float(s), UTC) if re.fullmatch(r"\d+(?:\.\d+)?", s) else dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=UTC)).astimezone(UTC)


def both(t: dt.datetime | str | float) -> str:
    """`1:41 PM PDT (20:41Z)` for an aware datetime (naive is UTC), an ISO string, or a Unix time (a number, or a Slack ts);
    `20:41Z` where there is no tz database."""
    u = _utc(t)
    if PACIFIC is None:
        return f"{u:%H:%M}Z"
    p = u.astimezone(PACIFIC)
    return f"{p:%I:%M %p}".lstrip("0") + f" {p.tzname()} ({u:%H:%M}Z)"


def _day(u: dt.datetime) -> str:
    d = u.astimezone(PACIFIC) if PACIFIC else u
    return f"{d.day} {d:%b}"


def slack_dates(text: str, now: dt.datetime | None = None) -> str:
    """Each dated UTC time outside code, links and URLs, as `<!date^UNIX^{date_short_pretty} {time}|FALLBACK>`:
    `YYYY-MM-DDTHH:MM[:SS]Z`, or `HH:MMZ` (or `both`'s `7:55 AM PDT (14:55Z)`) right after a date such as `14 Oct`,
    `Oct 14` or `Oct 14, 2026` (a date without a year is the one nearest `now`; before `both`'s form it is the Pacific date).
    A time with no date stays as written: guessing its day put a 14 Oct deadline on today."""
    t0 = _utc(now or dt.datetime.now(UTC))

    def token(m: re.Match) -> str:
        g = m.groupdict()
        mon = g["m1"] or g["m2"]
        if not g["iso"] and not mon:
            return m[0]
        try:
            if g["iso"]:
                u = dt.datetime(int(g["Y"]), int(g["M"]), int(g["D"]), int(g["H"]), int(g["N"]), int(g["S"] or 0), tzinfo=UTC)
            else:
                month, day = _MONTHS.index(mon[:3].lower()) + 1, int(g["d1"] or g["d2"])
                pac = g["ph"] is not None and PACIFIC is not None
                if pac:
                    hour, minute, zone = int(g["ph"]) % 12 + (12 if g["ap"].upper() == "PM" else 0), int(g["pm"]), PACIFIC
                else:
                    hour, minute, zone = int(g["zh"] or g["bh"]), int(g["zm"] or g["bm"]), UTC
                years = [int(g["y"])] if g["y"] else [t0.year - 1, t0.year, t0.year + 1]
                u = min((dt.datetime(y, month, day, hour, minute, tzinfo=zone).astimezone(UTC) for y in years),
                        key=lambda c: abs(c - t0))
                if pac and (u.hour, u.minute) != (int(g["zh"]), int(g["zm"])):
                    return m[0]
        except ValueError:
            return m[0]
        return f"<!date^{int(u.timestamp())}^{{date_short_pretty}} {{time}}|{'' if g['iso'] else _day(u) + ' '}{both(u)}>"

    out, i = [], 0
    for m in _PROTECTED.finditer(text):
        out += [_UTC_TIME.sub(token, text[i:m.start()]), m[0]]
        i = m.end()
    return "".join(out) + _UTC_TIME.sub(token, text[i:])
