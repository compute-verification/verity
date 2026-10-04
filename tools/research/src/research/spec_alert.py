"""The DM a spec change sends Daniel when it lands on main, by `research merge --push` or `research queue land` (his ruling,
3 Oct: a change to what a guarantee says waits on nobody, and he is shown it).

The spec is what each package's `lean-audit.json` records: every guarantee's statement (its printed signature and type
hash) and every definition the statements read (its hash, under `reads`). Between the main the push replaced and the one it
pushed, `changes` lists the guarantees added, removed or restated, and the definitions whose hash changed under the same
name. A guarantee that moved (removed and added under another name with the same type hash), one that only prints
differently, and a definition that moved send nothing. A DM that fails is printed and posted to #agent-alerts, so a spec
change never lands unseen.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

RECORD = "lean-audit.json"
#: characters of one DM; past it the message ends with the compare link, which holds the rest
LIMIT = 12000
#: lines of one signature shown; a longer one ends with a line saying how many more there are
SIGNATURE_LINES = 40
READERS_SHOWN = 4


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def _record(repo: Path, rev: str, path: str) -> dict:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True)
    if r.returncode:
        return {}
    doc = json.loads(r.stdout)
    return doc if isinstance(doc, dict) else {}


def _guarantees(doc: dict) -> dict:
    """The record's guarantees; one written before they were called that (a push's old side, at the rename) has `pins`."""
    return doc.get("guarantees") or doc.get("pins") or {}


def _definitions(doc: dict) -> dict[str, tuple[str | None, list[str]]]:
    """{definition: (hash, the guarantees that read it)} over every module under `reads`."""
    out: dict[str, tuple[str | None, list[str]]] = {}
    for group in (doc.get("reads") or {}).values():
        defs = group.get("definitions", {})
        defs = defs if isinstance(defs, dict) else dict.fromkeys(defs)
        for d, h in defs.items():
            out[d] = (h, sorted(group.get("guarantees") or group.get("pins") or []))
    return out


def _signature(rec: dict) -> str:
    lines = rec.get("signature") or ["(no signature recorded)"]
    lines = [lines] if isinstance(lines, str) else list(lines)
    if len(lines) > SIGNATURE_LINES:
        lines = lines[:SIGNATURE_LINES] + [f"… {len(lines) - SIGNATURE_LINES} more lines"]
    return "```\n" + "\n".join(lines).replace("```", "'''") + "\n```"


def _restated(a: dict, b: dict) -> bool:
    ha, hb = a.get("type_hash") or "", b.get("type_hash") or ""
    if len(ha) != len(hb):                      # one record rehashed (32-bit to SHA-256): only the printed statement compares
        return a.get("signature") != b.get("signature")
    return ha != hb


def changes(old: dict, new: dict) -> list[str]:
    """What changed in one package's spec, from its record ``old`` to ``new``, as Slack text: one entry per guarantee or
    definition, empty when every change is a move or a reprint."""
    a, b = _guarantees(old), _guarantees(new)
    gone = {n: r for n, r in a.items() if n not in b}
    came = {n: r for n, r in b.items() if n not in a}
    moved = {r.get("type_hash") for r in gone.values()} & {r.get("type_hash") for r in came.values()} - {None, ""}
    out = [f"• new guarantee `{n}`\n{_signature(r)}" for n, r in sorted(came.items()) if r.get("type_hash") not in moved]
    out += [f"• removed guarantee `{n}`" for n, r in sorted(gone.items()) if r.get("type_hash") not in moved]
    out += [f"• changed guarantee `{n}`, before:\n{_signature(a[n])}\nafter:\n{_signature(b[n])}"
            for n in sorted(set(a) & set(b)) if _restated(a[n], b[n])]
    da, db = _definitions(old), _definitions(new)
    for d in sorted(set(da) & set(db)):
        (h0, _), (h1, readers) = da[d], db[d]
        if h0 is not None and h1 is not None and len(h0) == len(h1) and h0 != h1:
            shown = ", ".join(f"`{r}`" for r in readers[:READERS_SHOWN])
            more = f" and {len(readers) - READERS_SHOWN} more" if len(readers) > READERS_SHOWN else ""
            out.append(f"• definition `{d}` changed; read by {shown or 'no guarantee'}{more}")
    return out


def web(repo: Path, remote: str) -> str | None:
    """The remote's GitHub URL (`https://github.com/OWNER/REPO`), or None when it isn't on GitHub."""
    r = subprocess.run(["git", "-C", str(repo), "remote", "get-url", remote], capture_output=True, text=True)
    m = re.search(r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", r.stdout.strip()) if not r.returncode else None
    return f"https://github.com/{m[1]}/{m[2]}" if m else None


def message(repo: Path, old: str, new: str, url: str | None = None, prs: list[int] | None = None) -> str | None:
    """The DM for main moving from ``old`` to ``new``, or None when no guarantee says anything new.  ``prs``, the pull requests
    it landed, default to the numbers in its first-parent subjects (`Merge #12 #34 (…)`)."""
    paths = [p for p in _git(repo, "diff", "--name-only", old, new).splitlines()
             if p == RECORD or p.endswith("/" + RECORD)]
    blocks = []
    for p in paths:
        if lines := changes(_record(repo, old, p), _record(repo, new, p)):
            blocks.append(f"*{p.rpartition('/')[0] or '.'}*\n" + "\n".join(lines))
    if not blocks:
        return None
    compare = f"<{url}/compare/{old[:12]}...{new[:12]}|{old[:9]}..{new[:9]}>" if url else f"{old[:9]}..{new[:9]}"
    if prs is None:
        prs = sorted({int(n) for n in re.findall(r"#(\d+)", _git(repo, "log", "--first-parent", "--format=%s", f"{old}..{new}"))})
    landed = ", ".join(f"<{url}/pull/{n}|#{n}>" if url else f"#{n}" for n in prs)
    text = f"A spec change landed on main ({compare})" + (f" from {landed}" if landed else "") + ".\n\n" + "\n\n".join(blocks)
    if len(text) > LIMIT:
        cut = text.rfind("\n", 0, LIMIT)
        text = text[:cut] + ("\n```" if text[:cut].count("```") % 2 else "") + f"\n… the rest is in {compare}."
    return text


def alert(repo: Path, old: str, new: str, *, remote: str = "origin", prs: list[int] | None = None, api=None,
          reg: dict | None = None, log=print) -> bool:
    """DM Daniel what the push from ``old`` to ``new`` changed in the spec; True when it was sent or there was nothing to
    send. A failure is printed and posted to #agent-alerts, and returns False; nothing raises, since the push has happened
    and the caller still has its own steps after a landing to run."""
    from .slack import Slack, SlackError, human_entry, load_registry, registry_path

    try:
        text = message(repo, old, new, web(repo, remote), prs)
    except Exception as e:
        text = (f"A change landed on main ({old[:9]}..{new[:9]}), and reading its `lean-audit.json` records failed: "
                f"{type(e).__name__}: {e}")
    if text is None:
        return True
    try:
        reg = reg if reg is not None else load_registry(registry_path())
        who = human_entry(reg, "daniel")
        if not who:
            raise SlackError("the Slack registry names no Daniel with a slack_user")
        api = api if api is not None else Slack(os.environ.get("SLACK_BOT_TOKEN"))
        api.call("chat.postMessage", channel=who["slack_user"], text=text, username="research merge", unfurl_links=False)
        log(f"research merge: the spec change went to Daniel as a DM ({len(text)} characters)")
        return True
    except Exception as e:
        why = (f"research merge: a spec change landed on main ({old[:12]}..{new[:12]}), and the DM to Daniel failed: "
               + (str(e) if isinstance(e, SlackError) else f"{type(e).__name__}: {e}"))
        print(why, file=sys.stderr)
        try:
            api = api if api is not None else Slack(os.environ.get("SLACK_BOT_TOKEN"))
            api.call("chat.postMessage", channel=((reg or {}).get("channels") or {})["agent-alerts"]["id"],
                     text=why + "\n\n" + text[:LIMIT // 2], username="research merge", unfurl_links=False)
        except Exception as e2:
            print(f"research merge: posting that to #agent-alerts failed too: {type(e2).__name__}: {e2}", file=sys.stderr)
        return False


def remote_head(repo: Path, remote: str, branch: str, auth: tuple[str, ...] = ()) -> str:
    """``branch`` as ``remote`` has it now (`git ls-remote`), so a stale tracking ref can't make a DM repeat what an earlier
    landing sent; the tracking ref when the remote can't be asked, and "" when there is neither."""
    r = subprocess.run(["git", "-C", str(repo), *auth, "ls-remote", remote, f"refs/heads/{branch}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.split()[0] if r.stdout.strip() else ""
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{branch}^{{commit}}"],
                          capture_output=True, text=True).stdout.strip()
