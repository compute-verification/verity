"""Times people read: Pacific first, then UTC (`1:41 PM PDT (20:41Z)`), so nobody converts by hand. Machine output (JSON, file
names, ids, front matter, log lines) stays UTC. `slack_dates` rewrites the UTC times in a Slack post into Slack's date token,
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
_UTC_TIME = re.compile(r"(?<![\w:.-])(?:(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?|(\d{1,2}):(\d{2}))Z(?![\w-])")


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


def slack_dates(text: str, now: dt.datetime | None = None) -> str:
    """Each `HH:MMZ` (the one nearest `now`: `00:00Z` written at 16:25Z is the coming midnight, not the one 16 h ago) and
    `YYYY-MM-DDTHH:MM[:SS]Z` outside code, links and URLs, as `<!date^UNIX^{date_short_pretty} {time}|FALLBACK>` with `both`
    as the fallback."""
    t0 = _utc(now or dt.datetime.now(UTC))

    def token(m: re.Match) -> str:
        y, mo, d, hh, mm, ss, h2, m2 = m.groups()
        try:
            if y:
                u = dt.datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0), tzinfo=UTC)
            else:
                at = dt.datetime.combine(t0.date(), dt.time(int(h2), int(m2)), tzinfo=UTC)
                u = min((at + dt.timedelta(days=k) for k in (-1, 0, 1)), key=lambda c: abs(c - t0))
        except ValueError:
            return m[0]
        return f"<!date^{int(u.timestamp())}^{{date_short_pretty}} {{time}}|{both(u)}>"

    out, i = [], 0
    for m in _PROTECTED.finditer(text):
        out += [_UTC_TIME.sub(token, text[i:m.start()]), m[0]]
        i = m.end()
    return "".join(out) + _UTC_TIME.sub(token, text[i:])
