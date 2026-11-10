"""The DM a spec change sends the person who reads the spec (the Slack registry's `daniel` entry) when it lands on main, by
`research merge --push` or drive's landing: a change to what a guarantee says waits on nobody, and he is shown it.

The spec is what each package's `lean-audit.json` records: every guarantee's statement (its printed signature and type
hash; for one stated as a definition, that statement unfolded once, `unfolded_hash`, so a restated definition shows its
guarantee changed; and the premises it rests on, its `premises`: a premise it gains or drops is shown), every definition
the statements read (its hash, under `reads`), every listed premise no `reads` record holds (its `digest` under `assumptions`, `domain`
or `open`, of it and the definitions it reads: one that gains a digest is shown as recorded), the named assumptions
(`assumptions`: one added or removed is shown, and so is a premise that becomes one), the domain hypotheses (`domain`:
one listed, with its `why`, one no longer listed, and a new `why` are shown) and the open premises (`open`: one listed,
one discharged, a new issue or due date, and a `why` gained or lost, by which the guarantees that read the premise
without taking it rest on it, are shown; an assumption that becomes open is tallied), what
the shape rule exempts (`shape.frozen`:
an entry added there is shown with its reason), the compiled code allowed on a verifier's path (`verifier.allow`: an
entry added there is shown), and the trust (`deployed`, the guarantees that state the deployed system; `trust`, the
premises they may rest on; `trust_gap`, those outside it each still rests on, which an update only shrinks: a gap grown,
which only a hand edit does, is shown first, ahead of everything else, and a guarantee deployed or no longer, a premise
trusted or no longer, and a gap shrunk are shown too). Between the main the push replaced and the one it
pushed, the guarantees added, removed or restated, and the definitions whose hash changed under the same name, are found
across every record that changed, not record by record: a guarantee or a definition that moved from one package's record
to another's under its name is no change, nor is a guarantee renamed with the same type hash (the premises it gains or
drops under its new name are shown all the same), one that only prints
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
#: the files a definition's meaning is elaborated from, by suffix or name; the audit's own directory writes records, says nothing
LEAN_INPUTS = (".lean", "lakefile.toml", "lakefile.lean", "lake-manifest.json", "lean-toolchain")
AUDIT_DIR = "tools/verity/lean/"
#: the dependency bundles every package with fetched dependencies is built against, pinned by content (`sha256`)
DEPS_PINS = AUDIT_DIR + "lean-deps.json"
#: who an open premise whose entry gives a `why` counts for besides the statements that take it
READERS = "the guarantees that read it without taking it"


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


def _definitions(doc: dict) -> dict[str, tuple[str | None, str]]:
    """{definition: (hash, its module)} over every module under `reads`. Which guarantees read one is the build's, and the
    record doesn't list it: the review `check.py --update` prints does."""
    out: dict[str, tuple[str | None, str]] = {}
    for m, group in (doc.get("reads") or {}).items():
        defs = group.get("definitions", {})
        defs = defs if isinstance(defs, dict) else dict.fromkeys(defs)
        for d, h in defs.items():
            out[d] = (h, m)
    return out


def _listed(doc: dict) -> dict[str, tuple[str, dict]]:
    """{premise: (its section, `assumptions`, `domain` or `open`, its entry)} over the record's listed premises."""
    out: dict[str, tuple[str, dict]] = {}
    for k in ("open", "domain", "assumptions"):
        section = doc.get(k) or {}
        if isinstance(section, dict):
            out |= {n: (k, r if isinstance(r, dict) else {}) for n, r in section.items()}
    return out


def _assumptions(doc: dict) -> dict[str, str | None]:
    """{listed premise: its record's `digest`, None without one} for the entries no `reads` record holds as a definition:
    a premise whose statement and what it reads only its own digest records (one under `reads` shows as a definition)."""
    held = _definitions(doc)
    return {n: r.get("digest") for n, (_, r) in _listed(doc).items() if n not in held}


def _premises(rec: dict) -> list[str] | None:
    """The premises a guarantee's record says its statement rests on, named assumptions, domain hypotheses and open
    premises alike (one that moves between them is no change to what the statement takes); None in a record written
    before records held them."""
    p = rec.get("premises")
    p = {"assumptions": p} if isinstance(p, list) else p
    if not isinstance(p, dict):
        return None
    return sorted({x for k in ("assumptions", "domain", "open") for x in (p.get(k) or []) if isinstance(x, str)})


def _ticket(r: dict) -> str:
    return f"{r.get('issue') or 'no issue'}, due {r.get('due')}"


def _scope(r: dict) -> str:
    return f"{r.get('why')}" + (f", citing `{r['cites']}`" if r.get("cites") else "")


def _on_read(entry: tuple[str, dict]) -> bool:
    """Whether a listed premise's entry is an open one that gives a `why`: `READERS` rest on it, and each one's record
    lists it under `premises.open`."""
    return entry[0] == "open" and "why" in entry[1]


def _was(entry: tuple[str, dict]) -> str:
    kind, r = entry
    return {"assumptions": "a named assumption", "domain": f"a domain hypothesis: {_scope(r)}"}.get(
        kind, f"open: {_ticket(r)}")


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


def _trust(doc: dict) -> tuple[list[str], dict[str, str], dict[str, dict[str, str]], dict[str, list[str]]] | None:
    """The record's trust: (`deployed`, the guarantees that state the deployed system; `trust.base`, {premise: the trust
    assumption it states}; `trust.extra`, {deployed guarantee: {premise it also trusts: the ruling's reason}}; `trust_gap`,
    {deployed guarantee: the premises outside its trust it rests on}), or None for a record written before it held them."""
    if not any(k in doc for k in ("deployed", "trust", "trust_gap")):
        return None
    t = doc.get("trust") if isinstance(doc.get("trust"), dict) else {}
    base = t.get("base") if isinstance(t.get("base"), dict) else {}
    extra = t.get("extra") if isinstance(t.get("extra"), dict) else {}
    gap = doc.get("trust_gap") if isinstance(doc.get("trust_gap"), dict) else {}
    return ([x for x in doc.get("deployed") or [] if isinstance(x, str)], {p: str(w) for p, w in base.items()},
            {g: {p: str(w) for p, w in ps.items()} for g, ps in extra.items() if isinstance(ps, dict)},
            {g: sorted(x for x in ps if isinstance(x, str)) for g, ps in gap.items() if isinstance(ps, list)})


def _renames(olds: dict[str, dict], news: dict[str, dict], guarantees: dict[str, str]) -> dict[str, str]:
    """{old name: new name} of a rename: ``guarantees``' (those gone and come with one type hash), and each premise or
    definition gone under one name and come under another with the one digest or hash only it has."""
    def hashes(docs: dict[str, dict]) -> dict[str, str]:
        out: dict[str, str] = {}
        for d in docs.values():
            out |= {n: h for n, (h, _) in _definitions(d).items() if h} | {n: h for n, h in _assumptions(d).items() if h}
        return out

    pa, pb = hashes(olds), hashes(news)
    gone = {n: h for n, h in pa.items() if n not in pb}
    came = {n: h for n, h in pb.items() if n not in pa}
    one = {h for h in gone.values() if list(gone.values()).count(h) == 1 and list(came.values()).count(h) == 1}
    by_hash = {h: n for n, h in came.items() if h in one}
    return {n: by_hash[h] for n, h in gone.items() if h in one} | guarantees


def _trusted(docs: dict[str, dict], ren: dict[str, str]) -> tuple[dict, dict, dict, dict] | None:
    """Every record's trust (`_trust`) across ``docs``, each name through ``ren`` (an old name's new one), each entry with
    its package: ({deployed guarantee: package}, {base premise: (package, what it states)}, {(guarantee, premise): (package,
    reason)}, {guarantee: (package, gap premises)}); None when no record holds a trust."""
    got = [(_package(p), t) for p, d in docs.items() if (t := _trust(d)) is not None]
    if not got:
        return None
    dep, base, extra, gap = {}, {}, {}, {}
    for pkg, (ds, bs, xs, gs) in got:
        dep |= {ren.get(g, g): pkg for g in ds}
        base |= {ren.get(p, p): (pkg, w) for p, w in bs.items()}
        extra |= {(ren.get(g, g), ren.get(p, p)): (pkg, w) for g, ps in xs.items() for p, w in ps.items()}
        gap |= {ren.get(g, g): (pkg, sorted(ren.get(p, p) for p in ps)) for g, ps in gs.items()}
    return dep, base, extra, gap


def _known(docs: dict[str, dict], ren: dict[str, str]) -> set[str]:
    """Every name ``docs`` hold, through ``ren``: their guarantees, listed premises and the definitions they read."""
    return {ren.get(n, n) for d in docs.values() for n in (*_guarantees(d), *_listed(d), *_definitions(d))}


def _unpaired(added: list[str], dropped: list[str], fresh: set[str], gone: set[str]) -> tuple[list[str], list[str]]:
    """``added`` and ``dropped`` less the renames among them: as many added names the old records never held (``fresh``)
    as dropped ones the new records no longer hold (``gone``), a name changed under a rename whose text changed with it
    (one whose hash or digest stayed is matched by `_renames` already)."""
    k = min(len(new := [p for p in added if p in fresh]), len(old := [p for p in dropped if p in gone]))
    return [p for p in added if p not in new[:k]], [p for p in dropped if p not in old[:k]]


def _trust_changes(olds: dict[str, dict], news: dict[str, dict], ren: dict[str, str]) -> list[tuple[str, str, str]]:
    """(package, kind, text) of what changed in the records' trust (`_trust`), a rename none (``ren``, else a premise
    replaced by one no old record held, or listed in the trust under another name with the same text): `grown` (a
    deployed guarantee rests outside its trust on a premise its `trust_gap` didn't list: an update only shrinks the gap,
    so it was grown by hand), `deployed` and `undeployed` (a guarantee listed in `deployed`, or no longer), `trusted` and
    `untrusted` (a premise listed in the trust base or a deployed guarantee's extra trust, or with what it states or its
    reason changed; or no longer listed) and `shrunk` (a premise the gap no longer lists), in that order; `trust` (text "")
    for a package whose record holds a trust for the first time."""
    a, b = _trusted(olds, ren), _trusted(news, {})
    if b is None:
        return []
    if a is None:
        return [(pkg, "trust", "") for pkg in dict.fromkeys(_package(p) for p, d in news.items() if _trust(d) is not None)]
    (da, ba, xa, ga), (db, bb, xb, gb) = a, b
    known_old, known_new = _known(olds, ren), _known(news, {})
    fresh, gone = known_new - known_old, known_old - known_new
    more, less = {}, {}
    for g in sorted(set(ga) | set(gb)):
        was, now = (ga.get(g) or ("", []))[1], (gb.get(g) or ("", []))[1]
        more[g], less[g] = _unpaired([p for p in now if p not in was], [p for p in was if p not in now], fresh, gone)
    pkg_of = {g: (gb.get(g) or ga.get(g))[0] for g in set(ga) | set(gb)}
    out = [(pkg_of[g], "grown", f"• trust gap grown, by hand: deployed guarantee `{g}` now also rests outside its trust on "
                                f"{', '.join(f'`{p}`' for p in ps)}") for g, ps in more.items() if ps]
    out += [(pkg, "deployed", f"• deployed guarantee `{g}` listed: it states the deployed system")
            for g, pkg in sorted(db.items()) if g not in da]
    out += [(pkg, "undeployed", f"• `{g}` no longer listed as deployed") for g, pkg in sorted(da.items()) if g not in db]
    same_base = {(p, q) for p, (_, w) in ba.items() if p not in bb for q, (_, v) in bb.items() if q not in ba and v == w}
    same_extra = {(k, j) for k, (_, w) in xa.items() if k not in xb for j, (_, v) in xb.items()
                  if j not in xa and j[0] == k[0] and v == w}
    for p, (pkg, w) in sorted(bb.items()):
        if (p not in ba or ba[p][1] != w) and not any(q == p for _, q in same_base):
            out.append((pkg, "trusted", f"• trust base `{p}`: {w}" + (f" (was: {ba[p][1]})" if p in ba else ", listed")))
    for (g, p), (pkg, w) in sorted(xb.items()):
        if ((g, p) not in xa or xa[g, p][1] != w) and not any(j == (g, p) for _, j in same_extra):
            out.append((pkg, "trusted", f"• deployed guarantee `{g}` also trusts `{p}`: {w}"
                        + (f" (was: {xa[g, p][1]})" if (g, p) in xa else "")))
    out += [(pkg, "untrusted", f"• trust base `{p}` no longer listed (it stated: {w})")
            for p, (pkg, w) in sorted(ba.items()) if p not in bb and not any(o == p for o, _ in same_base)]
    out += [(pkg, "untrusted", f"• deployed guarantee `{g}` no longer trusts `{p}`")
            for (g, p), (pkg, _) in sorted(xa.items()) if (g, p) not in xb and not any(k == (g, p) for k, _ in same_extra)]
    out += [(pkg_of[g], "shrunk", f"• trust gap shrunk: `{g}` no longer rests outside its trust on "
                                  f"{', '.join(f'`{p}`' for p in ps)}") for g, ps in less.items() if ps]
    return out


def _signature(rec: dict) -> str:
    lines = rec.get("signature") or ["(no signature recorded)"]
    lines = [lines] if isinstance(lines, str) else list(lines)
    if len(lines) > SIGNATURE_LINES:
        lines = lines[:SIGNATURE_LINES] + [f"… {len(lines) - SIGNATURE_LINES} more lines"]
    return "```\n" + "\n".join(lines).replace("```", "'''") + "\n```"


def _restated(a: dict, b: dict, sources: bool = True) -> bool:
    """Whether a guarantee's record says something else: its type hash or, where both records hold one and a Lean input
    changed, the hash of its statement with its definition unfolded once (`unfolded_hash`), which a restated definition
    moves under the same signature."""
    ha, hb = a.get("type_hash") or "", b.get("type_hash") or ""
    if len(ha) != len(hb):                      # one record rehashed (32-bit to SHA-256): only the printed statement compares
        return a.get("signature") != b.get("signature")
    ua, ub = a.get("unfolded_hash"), b.get("unfolded_hash")
    return ha != hb or (sources and bool(ua and ub) and ua != ub)


def _changed(n: str, a: dict, b: dict) -> str:
    if a.get("type_hash") == b.get("type_hash"):
        return (f"• changed guarantee `{n}`: its definition, unfolded once, says something else (the `reads` record holds "
                f"its new text's hash):\n{_signature(b)}")
    return f"• changed guarantee `{n}`, before:\n{_signature(a)}\nafter:\n{_signature(b)}"


def _package(path: str) -> str:
    return path.rpartition("/")[0] or "."


def _across(olds: dict[str, dict], news: dict[str, dict], sources: bool = True) -> list[tuple[str, str, str]]:
    """(package, kind, text) of every change from the records ``olds`` to ``news`` ({record path: record}), matched by name
    and type hash across all of them, so a guarantee or definition that moved from one record to another is no change.
    A trust gap grown by hand (`_trust_changes`' `grown`) comes first, before anything else, and the rest of what changed
    in the records' trust (`deployed`, `undeployed`, `trusted`, `untrusted`, `shrunk`, and `trust` for one held for the
    first time) right after `changed` and `removed`, through any rename (`_renames`).
    The other kinds are `changed`, `removed`, `premises` (the premises a guarantee's record says it rests on changed,
    under its name or through the same renames, its own and a premise's;
    `premised`, text "", for a record that lists them for the first time), `assumed` (a named assumption
    added, or another premise now one), `unassumed` (a named assumption no longer listed, saying so when statements still read
    it), `domain` (a domain hypothesis
    listed, or another premise now one, with its `why`), `undomained` (a domain hypothesis no longer listed), `scope` (a
    domain hypothesis's `why` or `cites` changed), `open` (an open premise listed, or a domain hypothesis now one),
    `opened` (text "": a named assumption now open), `discharged` (an open premise no longer listed), `ticket` (an open
    premise's issue or due date changed), `on_read` and `off_read` (an open premise's entry gains or loses its `why`, by
    which `READERS` rest on it; one listed or discharged with a `why` says so in its own line), `definition`,
    `assumption` (a listed premise's digest changed under its name),
    `recorded` (a premise listed before, with a digest now), `frozen` (an entry added to `shape.frozen`, the guarantees and
    statements exempt from the shape rule, matched by name across records like a guarantee), `conjecture` (an entry of
    `shape.conjectures` added or with a new status), `compiled` (an entry added to `verifier.allow`) and `new`, in that order (what an existing guarantee says comes first, and
    the premises it rests on before the hashes of what it reads, which a rehash moves by the thousand; so a DM cut at
    `LIMIT` loses new guarantees' signatures before anything else), and `moved` (text ""): a guarantee now in
    another package's record under its name, counted in that package's tally. A guarantee under another name with its type
    hash is a rename, and no change. Without ``sources`` (the push changed no Lean input), a definition's or an
    assumption's changed hash is `rehashed` (text ""), counted in its package's tally; a digest an assumption gains is
    shown all the same, since it records what nothing recorded before."""
    a = {n: (_package(p), r) for p, d in olds.items() for n, r in _guarantees(d).items()}
    b = {n: (_package(p), r) for p, d in news.items() for n, r in _guarantees(d).items()}
    gone = {n: r for n, (_, r) in a.items() if n not in b}
    came = {n: r for n, (_, r) in b.items() if n not in a}
    renamed = {r.get("type_hash") for r in gone.values()} & {r.get("type_hash") for r in came.values()} - {None, ""}
    out = [(b[n][0], "changed", _changed(n, a[n][1], b[n][1]))
           for n in sorted(set(a) & set(b)) if _restated(a[n][1], b[n][1], sources)]
    out += [(a[n][0], "removed", f"• removed guarantee `{n}`") for n, r in sorted(gone.items()) if r.get("type_hash") not in renamed]
    hg, hc = [r.get("type_hash") for r in gone.values()], [r.get("type_hash") for r in came.values()]
    moved_to = {n: m for n, r in gone.items() for m, s in came.items()
                if r.get("type_hash") and r.get("type_hash") == s.get("type_hash") and hg.count(r["type_hash"]) == 1
                and hc.count(r["type_hash"]) == 1}
    ren = _renames(olds, news, moved_to)
    trust = _trust_changes(olds, news, ren)
    out = [e for e in trust if e[1] == "grown"] + out + [e for e in trust if e[1] != "grown"]
    for n, m in sorted(({n: n for n in set(a) & set(b)} | moved_to).items(), key=lambda pair: pair[1]):
        was, now = _premises(a[n][1]), _premises(b[m][1])
        was = was and sorted({ren.get(x, x) for x in was})
        if now is None or was == now:
            continue
        if was is None:
            out.append((b[m][0], "premised", ""))
            continue
        more, less = [f"`{x}`" for x in now if x not in was], [f"`{x}`" for x in was if x not in now]
        said = ([f"now also on {', '.join(more)}"] if more else []) + ([f"no longer on {', '.join(less)}"] if less else [])
        out.append((b[m][0], "premises", f"• guarantee `{m}`" + (f" (renamed from `{n}`)" if m != n else "")
                    + f" rests {'; '.join(said)}"))
    hashed: list[tuple[str, str, str]] = []
    da: dict[str, set[str]] = {}
    for d in olds.values():
        for name, (h, _) in _definitions(d).items():
            da.setdefault(name, set()).add(h)
    for p, d in news.items():
        for name, (h1, module) in sorted(_definitions(d).items()):
            before = {h for h in da.get(name, ()) if h is not None}
            if h1 is not None and before and h1 not in before and all(len(h) == len(h1) for h in before):
                if not sources:
                    hashed.append((_package(p), "rehashed", ""))
                    continue
                hashed.append((_package(p), "definition", f"• definition `{name}` (`{module}`) changed"))
    aa: dict[str, set] = {}
    for d in olds.values():
        for name, h in _assumptions(d).items():
            aa.setdefault(name, set()).add(h)
    for p, d in news.items():
        sections = _listed(d)
        for name, h1 in sorted(_assumptions(d).items()):
            before = {h for h in aa.get(name, ()) if h is not None}
            if h1 is None or name not in aa or h1 in before:
                continue
            what = {"open": "open premise", "domain": "domain hypothesis"}.get(sections[name][0], "assumption")
            if not before:
                hashed.append((_package(p), "recorded", f"• {what} `{name}` recorded: its statement and the definitions it "
                                                        f"reads are now digested in its record"))
            elif not sources:
                hashed.append((_package(p), "rehashed", ""))
            else:
                hashed.append((_package(p), "assumption", f"• {what} `{name}` changed: its statement or a definition it reads"))
    la = {n: (_package(p), e) for p, d in olds.items() for n, e in _listed(d).items()}
    lb = {n: (_package(p), e) for p, d in news.items() for n, e in _listed(d).items()}
    for n, (p, (kind, r)) in sorted(lb.items()):
        was = la[n][1] if n in la else None
        if kind == "assumptions" and (was is None or was[0] != "assumptions"):
            out.append((p, "assumed", f"• named assumption `{n}` listed" + (f" (was {_was(was)})" if was else "")))
        elif kind == "domain" and (was is None or was[0] != "domain"):
            out.append((p, "domain", f"• domain hypothesis `{n}` listed: {_scope(r)}"
                        + (f" (was {_was(was)})" if was else "")))
        elif kind == "domain" and _scope(r) != _scope(was[1]):
            out.append((p, "scope", f"• domain hypothesis `{n}`: {_scope(r)} (was {_scope(was[1])})"))
        elif kind == "open" and (was is None or was[0] == "domain"):
            out.append((p, "open", f"• open premise `{n}` listed ({_ticket(r)})"
                        + (f"; {READERS} rest on it ({r['why']})" if "why" in r else "")
                        + (f" (was {_was(was)})" if was else "")))
        elif kind == "open" and was[0] == "assumptions":
            out.append((p, "opened", ""))
        elif kind == "open" and _ticket(r) != _ticket(was[1]):
            out.append((p, "ticket", f"• open premise `{n}`: {_ticket(r)} (was {_ticket(was[1])})"))
        if kind == "open" and was is not None and was[0] != "domain" and ("why" in r) != _on_read(was):
            out.append((p, "on_read", f"• open premise `{n}`: {READERS} now rest on it ({r['why']})") if "why" in r else
                       (p, "off_read", f"• open premise `{n}`: {READERS} no longer rest on it (its `why` was: {was[1]['why']})"))
    read = {name for d in news.values() for name in _definitions(d)}
    for n, (p, (kind, r)) in sorted(la.items()):
        if n not in lb:
            out.append((p, "unassumed", f"• named assumption `{n}` no longer listed"
                        + (": statements only read it, and its `reads` record holds it" if n in read else ""))
                       if kind == "assumptions" else
                       (p, "undomained", f"• domain hypothesis `{n}` no longer listed ({_scope(r)})") if kind == "domain"
                       else (p, "discharged", f"• open premise `{n}` discharged: no longer listed ({_ticket(r)})"
                             + (f"; {READERS} rested on it ({r['why']})" if "why" in r else "")))
    out += hashed
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
    out += [(b[n][0], "moved", "") for n in sorted(set(a) & set(b)) if a[n][0] != b[n][0] and not _restated(a[n][1], b[n][1], sources)]
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
    g = ", ".join([f"{kinds.count(k)} {k}" for k in ("changed", "removed", "new") if k in kinds]
                  + [f"{kinds.count(k)} {w}" for k, w in (("premises", "with new premises"), ("premised", "premises recorded"))
                     if k in kinds])
    g = ", ".join(([g] if g else []) + ([f"{kinds.count('moved')} moved here"] if "moved" in kinds else []))
    d = ", ".join(f"{kinds.count(k)} {w}" for k, w in (("definition", "changed"), ("rehashed", "rehashed")) if k in kinds)
    a = ", ".join(f"{kinds.count(k)} {w}" for k, w in (("assumption", "changed"), ("recorded", "recorded"),
                                                       ("assumed", "named"), ("unassumed", "no longer named"),
                                                       ("domain", "domain listed"), ("undomained", "domain dropped"),
                                                       ("scope", "domain restated"),
                                                       ("open", "open listed"), ("opened", "now open"),
                                                       ("discharged", "discharged"), ("ticket", "re-ticketed"),
                                                       ("on_read", "open where read"),
                                                       ("off_read", "no longer open where read"))
                  if k in kinds)
    f = ", ".join(f"{kinds.count(k)} {w}" for k, w in (("frozen", "frozen"), ("conjecture", "conjectures marked"))
                  if k in kinds)
    f = f"shape: {f}" if f else ""
    v = f"verifier: {kinds.count('compiled')} allowed" if "compiled" in kinds else ""
    t = ", ".join(f"{kinds.count(k)} {w}" for k, w in (("grown", "gap grown by hand"), ("deployed", "deployed"),
                                                       ("undeployed", "no longer deployed"), ("trusted", "trusted"),
                                                       ("untrusted", "no longer trusted"), ("shrunk", "gap shrunk"),
                                                       ("trust", "recorded"))
                  if k in kinds)
    return "; ".join(([f"trust: {t}"] if t else []) + ([f"guarantees: {g}"] if g else []) + ([f"definitions: {d}"] if d else [])
                     + ([f"premises: {a}"] if a else []) + ([f] if f else []) + ([v] if v else []))


def web(repo: Path, remote: str) -> str | None:
    """The remote's GitHub URL (`https://github.com/OWNER/REPO`), or None when it isn't on GitHub."""
    r = subprocess.run(["git", "-C", str(repo), "remote", "get-url", remote], capture_output=True, text=True)
    m = re.search(r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", r.stdout.strip()) if not r.returncode else None
    return f"https://github.com/{m[1]}/{m[2]}" if m else None


def _is_record(path: str) -> bool:
    """Whether ``path`` is a package's record. The audit's own directory holds none: a `lean-audit.json` there is a fixture
    of its tests, in whatever form the test needs, and a layout move that renames one says nothing about the spec."""
    return (path == RECORD or path.endswith("/" + RECORD)) and not path.startswith(AUDIT_DIR)


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
    """DM the spec's reader (`human_entry(reg, "daniel")`) what the push from ``old`` to ``new`` changed in the spec; True when
    it was sent or there was nothing to send. A failure is printed and posted to #agent-alerts, and returns False; nothing raises, since the push has happened
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
