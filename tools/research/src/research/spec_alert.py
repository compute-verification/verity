"""After `research merge --push` lands a change to what a guarantee says, Daniel gets a Slack direct message showing it.

Daniel's ruling (3 Oct, 6:46 PM PDT): such a change no longer waits on a statement reviewer; nobody signs off, and he sees each
one as it lands. A change is a difference in a `lean-audit.json`'s `pins` or `reads` between `main` before and after the push
(the sections the old `statement-reviewer` rule read). One that an `audit.py --update` report on its pull request printed as the
same statements (only moved: `research.queue.reviewed`) sends nothing. The message names the pull requests that changed the
file, then the report's `review` lines, or without a report the names that were added, removed or changed.

A message that can't be sent is never silent: `research merge` prints it, and it goes to #agent-alerts instead.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

SECTIONS = ("pins", "reads")
ALERTS = "agent-alerts"
MAX_TEXT = 3500                       # characters; past it the message points at `git diff` for the rest
MAX_NAMES = 12                        # names listed per kind of change before "and N more"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def _section(repo: Path, rev: str, path: str, section: str):
    from .queue import _section as read

    return read(repo, rev, path, section)


def changes(repo: Path, old: str, new: str) -> dict[str, dict[str, dict]]:
    """{lean-audit.json path: {section: {"old": value, "new": value}}} for each `pins` or `reads` that differs from ``old`` to
    ``new``."""
    paths = [p for p in _git(repo, "diff", "--name-only", old, new).splitlines()
             if p == "lean-audit.json" or p.endswith("/lean-audit.json")]
    out = {}
    for p in paths:
        secs = {}
        for s in SECTIONS:
            a, b = _section(repo, old, p, s), _section(repo, new, p, s)
            if a != b:
                secs[s] = {"old": a, "new": b}
        if secs:
            out[p] = secs
    return out


def names(old, new) -> dict[str, list[str]]:
    """The keys added, removed and changed from ``old`` to ``new`` (a section's value; an unparseable one is all "changed")."""
    if not all(isinstance(x, (dict, type(None))) for x in (old, new)):
        return {"added": [], "removed": [], "changed": ["(the file doesn't parse)"]}
    a, b = old or {}, new or {}
    return {"added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)),
            "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k])}


def pulls(repo: Path, old: str, new: str, path: str, gh: Callable[..., str]) -> list[dict]:
    """The pull requests ({number, url, head}) of the commits in ``old``..``new`` that changed ``path``, ascending."""
    found: dict[int, dict] = {}
    for sha in _git(repo, "log", "--no-merges", "--format=%H", f"{old}..{new}", "--", path).split():
        try:
            for p in json.loads(gh("api", f"repos/{{owner}}/{{repo}}/commits/{sha}/pulls")):
                found.setdefault(int(p["number"]), {"number": int(p["number"]), "url": p.get("html_url", ""),
                                                    "head": (p.get("head") or {}).get("sha", "")})
        except (subprocess.CalledProcessError, ValueError, KeyError, TypeError):
            continue
    return [found[n] for n in sorted(found)]


def _shown(kind: str, keys: list[str]) -> str:
    more = f" and {len(keys) - MAX_NAMES} more" if len(keys) > MAX_NAMES else ""
    return f"{kind} " + ", ".join(f"`{k}`" for k in keys[:MAX_NAMES]) + more


def message(repo: Path, old: str, new: str, gh: Callable[..., str], reviews: Callable[[int, str], list[dict]]) -> str | None:
    """The direct message for ``old`` -> ``new``, or None when no guarantee's statement changed."""
    from .queue import reviewed

    parts = []
    for path, secs in changes(repo, old, new).items():
        prs = pulls(repo, old, new, path, gh)
        olds = {s: (secs[s]["old"] if s in secs else _section(repo, old, path, s)) for s in SECTIONS}
        news = {s: (secs[s]["new"] if s in secs else _section(repo, new, path, s)) for s in SECTIONS}
        reports = [r for p in prs if p["head"] for r in reviews(p["number"], p["head"])]
        if any(reviewed(path, olds, news, r) for r in reports):
            continue                                      # printed whole as the same statements: only moved
        who = ", ".join(f"<{p['url']}|#{p['number']}>" if p["url"] else f"#{p['number']}" for p in prs) or "no pull request found"
        lines = [f"*{path}* ({who})"]
        pkg = path.rpartition("/")[0]
        notes = [n for r in reports if r.get("statement_changes") and str(r.get("package", "")).rstrip("/").endswith(pkg)
                 for n in r.get("review", [])]
        if notes:
            lines += [f"> {n}" for n in notes]
        else:
            for s in SECTIONS:
                if s in secs:
                    n = names(secs[s]["old"], secs[s]["new"])
                    lines.append(f"• {s}: " + "; ".join(_shown(k, n[k]) for k in ("added", "removed", "changed") if n[k]))
        parts.append("\n".join(lines))
    if not parts:
        return None
    text = (f"A guarantee's statement changed on main: `{old[:12]}` → `{new[:12]}` (no sign-off; your 3 Oct ruling). "
            f"`git diff {old[:12]} {new[:12]} -- '*lean-audit.json'` shows all of it.\n\n" + "\n\n".join(parts))
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT].rsplit("\n", 1)[0] + "\n… (cut; the `git diff` above has the rest)"


def _store_reviews(store) -> Callable[[int, str], list[dict]]:
    from .queue import QueueError, StoreGrants

    grants = StoreGrants(store, lambda s: print(s.replace("research queue:", "research merge:", 1), file=sys.stderr))

    def reviews(pr: int, head: str) -> list[dict]:
        try:
            return grants.reviews(pr, head)
        except QueueError as e:                           # no report read: the change is shown, never skipped
            print(f"research merge: {e}", file=sys.stderr)
            return []
    return reviews


def _slack_send(text: str) -> None:
    from . import slack

    reg = slack.load_registry(slack.registry_path())
    daniel = slack.human_entry(reg, "daniel")
    if not daniel:
        raise slack.SlackError("the registry has no `humans` entry for Daniel with a slack_user")
    slack.Slack(os.environ.get("SLACK_BOT_TOKEN")).call("chat.postMessage", channel=daniel["slack_user"], text=text,
                                                         unfurl_links=False)


def _slack_alert(text: str) -> None:
    from . import slack

    reg = slack.load_registry(slack.registry_path())
    cid = reg["channels"][ALERTS]["id"]
    slack.Slack(os.environ.get("SLACK_BOT_TOKEN")).call("chat.postMessage", channel=cid, text=text, unfurl_links=False)


def alert(repo: Path, old: str, new: str, *, gh: Callable[..., str], store=None,
          reviews: Callable[[int, str], list[dict]] | None = None,
          send: Callable[[str], None] = _slack_send, fallback: Callable[[str], None] = _slack_alert, log=print) -> int:
    """DM Daniel the guarantee changes ``old`` -> ``new``. 0 when there were none or the DM went; 1 when it didn't, after
    printing it and trying #agent-alerts."""
    if reviews is None:
        def reviews(pr: int, head: str, _cache: list = []) -> list[dict]:  # noqa: B006  (the store opens only if needed)
            if not _cache:
                from .store.local import LocalStore

                _cache.append(_store_reviews(store if store is not None else LocalStore(None)))
            return _cache[0](pr, head)
    try:
        text = message(repo, old, new, gh, reviews)
    except Exception as e:  # noqa: BLE001  (an alert that can't be built must still be loud)
        text = (f"A guarantee change may have landed on main `{old[:12]}` → `{new[:12]}`, but `research merge` couldn't build "
                f"its message ({type(e).__name__}: {e}); see `git diff {old[:12]} {new[:12]} -- '*lean-audit.json'`.")
    if text is None:
        return 0
    try:
        send(text)
        log(f"research merge: sent Daniel the guarantee change on {new[:12]}")
        return 0
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
    print(f"research merge: GUARANTEE CHANGE NOT SENT TO DANIEL ({why}). The message:\n{text}", file=sys.stderr)
    try:
        fallback(f"`research merge` couldn't DM Daniel a guarantee change ({why}). The message:\n\n{text}")
        print(f"research merge: posted it to #{ALERTS} instead", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"research merge: and #{ALERTS} failed too ({type(e).__name__}: {e}); send it to Daniel by hand", file=sys.stderr)
    return 1
