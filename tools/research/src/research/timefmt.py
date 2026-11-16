"""Times people read: Pacific first, then UTC (`1:41 PM PDT (20:41Z)`), so nobody converts by hand. Machine output (JSON, file
names, ids, front matter, log lines) stays UTC. `slack_dates` rewrites the dated UTC times in a Slack post into Slack's date token,
which shows each reader their own zone and falls back to `both`. Stdlib only."""
from __future__ import annotations

import calendar
import datetime as dt
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    PACIFIC: ZoneInfo | None = ZoneInfo("America/Los_Angeles")
except ZoneInfoNotFoundError:  # a host with no tz database (the Nebius job images): `both` prints UTC alone
    PACIFIC = None
UTC = dt.UTC

#: left alone by `slack_dates`: code blocks and spans, Slack's <...> tokens (links, mentions, dates) and bare URLs
_PROTECTED = re.compile(r"```.*?```|`[^`\n]*`|<[^<>\n]*>|\bhttps?://\S+", re.S)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MON = r"(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_WEEKDAY = r"(?i:mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\.?"
_DATE = (rf"(?:(?P<wd>{_WEEKDAY}),?\s+)?(?:(?P<d1>\d{{1,2}})\s+(?P<m1>{_MON})|(?P<m2>{_MON})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?)"
         rf"(?:,?\s+(?P<y>\d{{4}}))?,?\s+(?:at\s+)?")
_PAC = r"(?P<ph>\d{1,2}):(?P<pm>\d{2})\s*(?P<ap>[AP]M)\s+P[DS]T\s+\("
_UTC_TIME = re.compile(rf"(?<![\w:.-])(?:(?P<iso>(?P<Y>\d{{4}})-(?P<M>\d{{2}})-(?P<D>\d{{2}})T(?P<H>\d{{2}}):(?P<N>\d{{2}})(?::(?P<S>\d{{2}}))?Z)"
                       rf"|(?:{_DATE})?(?:{_PAC}(?P<zh>\d{{1,2}}):(?P<zm>\d{{2}})Z\)|(?P<bh>\d{{1,2}}):(?P<bm>\d{{2}})Z))(?![\w-])")
#: a word just before a time that makes it a deadline (`by 20:00Z`, `ETA 4:00 AM PST`), not when the line was written
DEADLINE_WORDS = {"by", "until", "till", "before", "after", "at", "from", "eta", "since", "to", "due"}
#: a header's name token: a Slack token (`<!subteam^S1|@ci>`) or a word of letters, digits and `@*_().,'’/-`, never a `:`
_NAME = r"(?:<[^>\s]+>|[@*_\w().,'\u2019/-]+)"
#: the header position (`stamp_ahead`): up to 5 name tokens, an optional date, then `2:28 AM PST`, `10:28Z` or `10:28 UTC`
_HEADER = re.compile(rf"^\W{{0,4}}(?P<names>(?:{_NAME} ){{0,5}}?)(?:{_DATE})?(?P<t>(?P<ph>\d{{1,2}}):(?P<pm>\d{{2}})\s*"
                     rf"(?P<ap>[AP]M)\s+P[DS]?T|(?P<zh>\d{{1,2}}):(?P<zm>\d{{2}})(?:Z|\s+UTC))(?![\w-])")
#: what follows a header time with no comma before it (`ci 13:07Z: ...`, `*infra 11:45Z* (quiet)`), as `HEADER_TIME_RE` reads
_HEADER_END = re.compile(r"\*?(?:[:,)]| \(| [-\u2014])")


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


def _day(u: dt.datetime, weekday: bool = False) -> str:
    d = u.astimezone(PACIFIC) if PACIFIC else u
    return (f"{d:%a} " if weekday else "") + f"{d.day} {d:%b}"


def slack_dates(text: str, now: dt.datetime | None = None) -> str:
    """Each dated UTC time outside code, links and URLs, as `<!date^UNIX^{date_short_pretty} {time}|FALLBACK>`:
    `YYYY-MM-DDTHH:MM[:SS]Z`, or `HH:MMZ` (or `both`'s `7:55 AM PDT (14:55Z)`) right after a date such as `DD Mon`,
    `Mon DD` or `Mon DD, YYYY` (a date without a year is the one nearest `now`; before `both`'s form it is the Pacific date).
    A weekday written before the date goes into the token, and the fallback names the Pacific one: the writer's was the UTC
    day's, which a time across Pacific midnight contradicts.
    A time with no date stays as written: guessing its day can put another day's deadline on today."""
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
        return f"<!date^{int(u.timestamp())}^{{date_short_pretty}} {{time}}|{'' if g['iso'] else _day(u, bool(g['wd'])) + ' '}{both(u)}>"

    out, i = [], 0
    for m in _PROTECTED.finditer(text):
        out += [_UTC_TIME.sub(token, text[i:m.start()]), m[0]]
        i = m.end()
    return "".join(out) + _UTC_TIME.sub(token, text[i:])


def stamp_ahead(text: str, now: dt.datetime) -> str | None:
    """The refusal of a post whose first line is stamped later than `now`'s minute, else None.

    Only a time in the header position is a stamp. The header is the first line's start: at most 4 non-word characters,
    then at most 5 name tokens, each a Slack token (`<!subteam^S1|@ci>`) or a word of letters, digits and `@*_().,'’/-`
    (no `:`, so `ci: ...` ends a header), one space after each, then an optional date (`Sun 15 Nov`, `Nov 15, 2026`), then
    the time, Pacific (`2:28 AM PST`) or UTC (`10:28Z`, `10:28 UTC`). It is a header only when the last name token ends in a
    comma (`lean (lead), Sun 15 Nov 2:28 AM PST`, `lean's worker AREAS, 11:19 PM PST`) or the time is followed by `:`,
    `,`, `)`, ` (`, ` -` or ` —` (`ci 10:28Z: ...`), and not when that last token is a deadline word (`by 20:00Z:`). So a time
    later in the line, as an alert quoting `before the 20:30Z row` or `check starts 20:00Z`, is no stamp.

    The stamp is read on the date written before it, else on the day nearest `now` (so a stamp across midnight is read on
    its own day). A line with no header time, or one it can't read (`10:5x PM PST`, an hour past 12), passes: a doubt
    never blocks a post."""
    line = (text or "").lstrip().split("\n", 1)[0]
    m = _HEADER.match(line)
    names = m["names"].split() if m else []
    if not m or names and names[-1].strip("*_,()").lower() in DEADLINE_WORDS or not (
            names and names[-1].endswith(",") or _HEADER_END.match(line, m.end("t"))):
        return None
    g, u0 = m.groupdict(), _utc(now).replace(second=0, microsecond=0)
    pac = g["ph"] is not None
    if pac and PACIFIC is None:
        return None
    zone = PACIFIC if pac else UTC
    hour, minute = int(g["ph"] if pac else g["zh"]), int(g["pm"] if pac else g["zm"])
    if minute > 59 or not (1 <= hour <= 12 if pac else hour <= 23):
        return None
    if pac:
        hour = hour % 12 + (12 if g["ap"].upper() == "PM" else 0)
    mon = g["m1"] or g["m2"]
    if mon:
        month, day = _MONTHS.index(mon[:3].lower()) + 1, int(g["d1"] or g["d2"])
        days = [dt.date(y, month, day) for y in ([int(g["y"])] if g["y"] else [u0.year - 1, u0.year, u0.year + 1])
                if 1 < y < dt.MAXYEAR and 1 <= day <= calendar.monthrange(y, month)[1]]
    else:
        today = u0.astimezone(zone).date()
        days = [today + dt.timedelta(days=k) for k in (-1, 0, 1)]
    if not days:
        return None
    stamp = min((dt.datetime(d.year, d.month, d.day, hour, minute, tzinfo=zone).astimezone(UTC) for d in days),
                key=lambda c: abs(c - u0))
    if stamp <= u0:
        return None
    local = u0.astimezone(zone)
    clock = ((f"{local:%a} " if g["wd"] else "") + f"{local.day} {local:%b} " if mon else "") + (
        f"{local:%I:%M %p}".lstrip("0") + f" {local.tzname()}" if pac else f"{local:%H:%M}Z")
    told = "TZ=America/Los_Angeles date" if pac else "date -u"
    return (f"the first line says {line[m.end('names'):m.end('t')].strip()} and it is {clock}: write the time `{told}` gives now, leave the time out, "
            "or, if it is a time to come, put `by`, `at` or `ETA` before it")
