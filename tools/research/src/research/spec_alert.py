"""The DM a spec change sends Daniel when it lands on main, by `research merge --push` or drive's landing: a change to what a
guarantee says waits on nobody, and he is shown it.

The spec is what each package's `lean-audit.json` records: every guarantee's statement (its printed signature and type
hash), every definition the statements read (its hash, under `reads`), every named assumption no `reads` record holds
(its `digest` under `assumptions`, of it and the definitions it reads: one that gains a digest is shown as recorded), what
the shape rule exempts (`shape.frozen`:
an entry added there is shown with its reason), and the compiled code allowed on a verifier's path (`verifier.allow`: an
entry added there is shown). Between the main the push replaced and the one it
pushed, the guarantees added, removed or restated, and the definitions whose hash changed under the same name, are found
across every record that changed, not record by record: a guarantee or a definition that moved from one package's record
to another's under its name is no change, nor is a guarantee renamed with the same type hash, one that only prints
differently, or a record that moved with its package, and a push of only these sends nothing. Nor is a definition whose
hash changed in a push that changed no Lean input (a `.lean` file, a lakefile, a manifest or a toolchain, outside the
audit's own `tools/verity/lean/`, or a dependency bundle newly pinned in its `lean-deps.json`): what a definition says
comes from those alone, so its record was only rehashed or regrouped by the audit, and it is tallied as `rehashed`. The DM opens with each
package's tally (with the guarantees that moved into it), then lists what existing guarantees and definitions now say
before the new guarantees. A DM that fails is
printed and posted to #agent-alerts, so a spec change never lands unseen.
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
#: the files a definition's meaning is elaborated from, by suffix or name; the audit's own directory writes records, says nothing
LEAN_INPUTS = (".lean", "lakefile.toml", "lakefile.lean", "lake-manifest.json", "lean-toolchain")
AUDIT_DIR = "tools/verity/lean/"
#: the dependency bundles every package with fetched dependencies is built against, pinned by content (`sha256`)
DEPS_PINS = AUDIT_DIR + "lean-deps.json"


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


def _assumptions(doc: dict) -> dict[str, str | None]:
    """{named assumption: its record's `digest`, None without one} for the entries no `reads` record holds as a definition:
    a premise whose statement and what it reads only its own digest records (one under `reads` shows as a definition)."""
    a = doc.get("assumptions") or {}
    held = _definitions(doc)
    return {n: (r or {}).get("digest") for n, r in a.items() if n not in held} if isinstance(a, dict) else {}


def _frozen(doc: dict) -> dict[str, str]:
    """{name: why}: the record's `shape.frozen`, what the shape rule exempts until it fits."""
    f = (doc.get("shape") or {}).get("frozen") or {}
    return f if isinstance(f, dict) else {}


def _conjectures(doc: dict) -> dict[str, dict]:
    """{name: {status, test, why}}: the record's `shape.conjectures`, the named conjectures properties may rest on."""
    c = (doc.get("shape") or {}).get("conjectures") or {}
    return {n: e for n, e in c.items() if isinstance(e, dict)} if isinstance(c, dict) else {}


def _allowed(doc: dict) -> list[str]:
    """The record's `verifier.allow`: the compiled code no proof sees that its verifier's path may reach."""
    a = (doc.get("verifier") or {}).get("allow") or []
    return [x for x in a if isinstance(x, str)] if isinstance(a, list) else []


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


def _package(path: str) -> str:
    return path.rpartition("/")[0] or "."


def _across(olds: dict[str, dict], news: dict[str, dict], sources: bool = True) -> list[tuple[str, str, str]]:
    """(package, kind, text) of every change from the records ``olds`` to ``news`` ({record path: record}), matched by name
    and type hash across all of them, so a guarantee or definition that moved from one record to another is no change.
    The kinds are `changed`, `removed`, `definition`, `assumption` (an assumption's digest changed under its name),
    `recorded` (an assumption listed before, with a digest now), `frozen` (an entry added to `shape.frozen`, the guarantees and
    statements exempt from the shape rule, matched by name across records like a guarantee), `conjecture` (an entry of
    `shape.conjectures` added or with a new status), `compiled` (an entry added to `verifier.allow`) and `new`, in that order (what an existing guarantee says comes first,
    so a DM cut at `LIMIT` loses new guarantees' signatures before anything else), and `moved` (text ""): a guarantee now in
    another package's record under its name, counted in that package's tally. A guarantee under another name with its type
    hash is a rename, and no change. Without ``sources`` (the push changed no Lean input), a definition's or an
    assumption's changed hash is `rehashed` (text ""), counted in its package's tally; a digest an assumption gains is
    shown all the same, since it records what nothing recorded before."""
    a = {n: (_package(p), r) for p, d in olds.items() for n, r in _guarantees(d).items()}
    b = {n: (_package(p), r) for p, d in news.items() for n, r in _guarantees(d).items()}
    gone = {n: r for n, (_, r) in a.items() if n not in b}
    came = {n: r for n, (_, r) in b.items() if n not in a}
    renamed = {r.get("type_hash") for r in gone.values()} & {r.get("type_hash") for r in came.values()} - {None, ""}
    out = [(b[n][0], "changed", f"• changed guarantee `{n}`, before:\n{_signature(a[n][1])}\nafter:\n{_signature(b[n][1])}")
           for n in sorted(set(a) & set(b)) if _restated(a[n][1], b[n][1])]
    out += [(a[n][0], "removed", f"• removed guarantee `{n}`") for n, r in sorted(gone.items()) if r.get("type_hash") not in renamed]
    da: dict[str, set[str]] = {}
    for d in olds.values():
        for name, (h, _) in _definitions(d).items():
            da.setdefault(name, set()).add(h)
    for p, d in news.items():
        for name, (h1, readers) in sorted(_definitions(d).items()):
            before = {h for h in da.get(name, ()) if h is not None}
            if h1 is not None and before and h1 not in before and all(len(h) == len(h1) for h in before):
                if not sources:
                    out.append((_package(p), "rehashed", ""))
                    continue
                shown = ", ".join(f"`{r}`" for r in readers[:READERS_SHOWN])
                more = f" and {len(readers) - READERS_SHOWN} more" if len(readers) > READERS_SHOWN else ""
                out.append((_package(p), "definition", f"• definition `{name}` changed; read by {shown or 'no guarantee'}{more}"))
    aa: dict[str, set] = {}
    for d in olds.values():
        for name, h in _assumptions(d).items():
            aa.setdefault(name, set()).add(h)
    for p, d in news.items():
        for name, h1 in sorted(_assumptions(d).items()):
            before = {h for h in aa.get(name, ()) if h is not None}
            if h1 is None or name not in aa or h1 in before:
                continue
            if not before:
                out.append((_package(p), "recorded", f"• assumption `{name}` recorded: its statement and the definitions it "
                                                     f"reads are now digested in its record"))
            elif not sources:
                out.append((_package(p), "rehashed", ""))
            else:
                out.append((_package(p), "assumption", f"• assumption `{name}` changed: its statement or a definition it reads"))
    fa = {n for d in olds.values() for n in _frozen(d)}
    out += [(_package(p), "frozen", f"• frozen `{n}` (doesn't have the shape yet): {why}")
            for p, d in news.items() for n, why in sorted(_frozen(d).items()) if n not in fa]
    ca = {n: c.get("status") for d in olds.values() for n, c in _conjectures(d).items()}
    out += [(_package(p), "conjecture", f"• conjecture `{n}` {c.get('status')}"
             + (f" (was {ca[n]})" if n in ca else "") + f": {c.get('why', '')}")
            for p, d in news.items() for n, c in sorted(_conjectures(d).items()) if ca.get(n, "") != c.get("status")]
    va = {x for d in olds.values() for x in _allowed(d)}
    out += [(_package(p), "compiled", f"• compiled code allowed on the verifier's path, which no proof sees: `{x}`")
            for p, d in news.items() for x in _allowed(d) if x not in va]
    out += [(b[n][0], "new", f"• new guarantee `{n}`\n{_signature(r)}")
            for n, r in sorted(came.items()) if r.get("type_hash") not in renamed]
    out += [(b[n][0], "moved", "") for n in sorted(set(a) & set(b)) if a[n][0] != b[n][0] and not _restated(a[n][1], b[n][1])]
    return out


def _entries(old: dict, new: dict) -> list[tuple[str, str]]:
    """``changes``' entries for one record, with their kinds."""
    return [(kind, text) for _, kind, text in _across({RECORD: old}, {RECORD: new}) if text]


def changes(old: dict, new: dict) -> list[str]:
    """What changed in one package's spec, from its record ``old`` to ``new``, as Slack text: one entry per guarantee or
    definition, empty when every change is a move or a reprint."""
    return [text for _, text in _entries(old, new)]


def _tally(kinds: list[str]) -> str:
    """`guarantees: 1 changed, 19 new, 1691 moved here; definitions: 2 changed` for one package's entries."""
    g = ", ".join(f"{kinds.count(k)} {k}" for k in ("changed", "removed", "new") if k in kinds)
    g = ", ".join(([g] if g else []) + ([f"{kinds.count('moved')} moved here"] if "moved" in kinds else []))
    d = ", ".join(f"{kinds.count(k)} {w}" for k, w in (("definition", "changed"), ("rehashed", "rehashed")) if k in kinds)
    a = ", ".join(f"{kinds.count(k)} {w}" for k, w in (("assumption", "changed"), ("recorded", "recorded")) if k in kinds)
    f = ", ".join(f"{kinds.count(k)} {w}" for k, w in (("frozen", "frozen"), ("conjecture", "conjectures marked"))
                  if k in kinds)
    f = f"shape: {f}" if f else ""
    v = f"verifier: {kinds.count('compiled')} allowed" if "compiled" in kinds else ""
    return "; ".join(([f"guarantees: {g}"] if g else []) + ([f"definitions: {d}"] if d else [])
                     + ([f"assumptions: {a}"] if a else []) + ([f] if f else []) + ([v] if v else []))


def web(repo: Path, remote: str) -> str | None:
    """The remote's GitHub URL (`https://github.com/OWNER/REPO`), or None when it isn't on GitHub."""
    r = subprocess.run(["git", "-C", str(repo), "remote", "get-url", remote], capture_output=True, text=True)
    m = re.search(r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", r.stdout.strip()) if not r.returncode else None
    return f"https://github.com/{m[1]}/{m[2]}" if m else None


def _is_record(path: str) -> bool:
    return path == RECORD or path.endswith("/" + RECORD)


def _records_changed(repo: Path, old: str, new: str) -> list[tuple[str, str]]:
    """Each record that differs between ``old`` and ``new`` as (its path in ``old``, its path in ``new``): a record git
    sees renamed (`-M`), such as one that moved with its package, is read from its old path, so a move says nothing new."""
    out = []
    for line in _git(repo, "diff", "--name-status", "-M", old, new).splitlines():
        status, *paths = line.split("\t")
        if status.startswith("R") and all(map(_is_record, paths)):
            out.append((paths[0], paths[1]))
        else:
            out += [(p, p) for p in paths if _is_record(p)]
    return out


def _sources(repo: Path, old: str, new: str) -> bool:
    """Whether the push from ``old`` to ``new`` changed a Lean input: a file under `LEAN_INPUTS` outside the audit's own
    directory, or a dependency bundle pinned that ``old`` didn't pin (re-keying or dropping a pin builds against no new tree)."""
    if any(p.endswith(LEAN_INPUTS) and not p.startswith(AUDIT_DIR) for p in _git(repo, "diff", "--name-only", old, new).splitlines()):
        return True
    bundles = [{d.get("sha256") for d in (_record(repo, rev, DEPS_PINS).get("deps") or {}).values()} for rev in (old, new)]
    return bool(bundles[1] - bundles[0])


def message(repo: Path, old: str, new: str, url: str | None = None, prs: list[int] | None = None) -> str | None:
    """The DM for main moving from ``old`` to ``new``, or None when no guarantee says anything new.  ``prs``, the pull requests
    it landed, default to the numbers in its first-parent subjects (`Merge #12 #34 (…)`)."""
    pairs = _records_changed(repo, old, new)
    entries = _across({b: _record(repo, old, b) for b, _ in pairs}, {p: _record(repo, new, p) for _, p in pairs},
                     _sources(repo, old, new))
    if not any(text for _, _, text in entries):
        return None
    tallies, blocks = [], []
    for pkg in dict.fromkeys(pkg for pkg, _, _ in entries):
        mine = [(kind, text) for p, kind, text in entries if p == pkg]
        tallies.append(f"• `{pkg}`: {_tally([k for k, _ in mine])}")
        if texts := [text for _, text in mine if text]:
            blocks.append(f"*{pkg}*\n" + "\n".join(texts))
    compare = f"<{url}/compare/{old[:12]}...{new[:12]}|{old[:9]}..{new[:9]}>" if url else f"{old[:9]}..{new[:9]}"
    if prs is None:
        prs = sorted({int(n) for n in re.findall(r"#(\d+)", _git(repo, "log", "--first-parent", "--format=%s", f"{old}..{new}"))})
    landed = ", ".join(f"<{url}/pull/{n}|#{n}>" if url else f"#{n}" for n in prs)
    text = (f"A spec change landed on main ({compare})" + (f" from {landed}" if landed else "") + ".\n" + "\n".join(tallies)
            + "\n\n" + "\n\n".join(blocks))
    if len(text) > LIMIT:
        tail = f"\n… the rest is in {compare}."
        cut = text.rfind("\n", 0, LIMIT - len(tail) - len("\n```"))
        text = text[:cut] + ("\n```" if text[:cut].count("```") % 2 else "") + tail
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
