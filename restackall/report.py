#!/usr/bin/env python3
"""Scratch, lands nowhere: the tables of a restackall measurement.

    report.py --prs PRS.json [--out REPORT.md] [--json ALL.json] RUNDIR...

RUNDIR is a fetched run (`~/.research/runs/RUN`): its `out/heads/N.json` are `run.py`'s records. PRS.json is the
`gh pr list --json number,headRefOid,headRefName,author,body,isDraft,title` the heads were taken from.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

STOP_OWN = "stops at `git merge main`"
ORDER = ("clean", "clean with the drivers only", STOP_OWN, "stops at the layout", "restack cannot run", "not measured")


def outcome(r: dict) -> tuple[str, str]:
    """(class, detail) of one head's record."""
    s = r.get("status")
    if s in ("clean", "up-to-date"):
        if r.get("without_drivers"):
            at = set(r["without_drivers"])
            return "clean with the drivers only", ("at `git merge main`" if at == {"main"} else
                                                    "at the layout" if "main" not in at else "at both")
        return "clean", ""
    if s == "in-main":
        return "clean", "main already holds it"
    if s == "conflicted":
        if r.get("step") == "main":
            return STOP_OWN, ""
        return "stops at the layout", f"{r.get('step')} of {str(r.get('at_move'))[:12]}"
    if s in ("locks-stale", "pins-stale"):
        return "stops at the layout", s
    if s == "script-failed" and "with no row in the map" in (r.get("error") or ""):
        r["files"] = [x.strip() for x in r["error"].split("\n")[1:] if x.strip()]
        r["kinds"] = {f: "no row in the map" for f in r["files"]}
        return "stops at the layout", f"script of {str(r.get('at_move'))[:12]}: a new file with no row in the map"
    if s in ("script-failed", "based-after", "refused", "no-drivers", "no-moves"):
        return "restack cannot run", s
    return "not measured", str(s)


def load(dirs: list[str]) -> dict[int, dict]:
    rows = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "out", "heads", "*.json")):
            r = json.load(open(p))
            r["run"] = os.path.basename(os.path.normpath(d))
            rows[r["n"]] = r
    return rows


def pin_names(p: dict | None) -> set[str]:
    """The pins `pins.py --stale` named, by path."""
    rep = (p or {}).get("report") or []
    if isinstance(rep, str):
        rep = rep.splitlines()
    return {x.split(" is stale")[0].strip() for x in rep if " is stale" in x}


def staged_pins(dirs: list[str]) -> dict | None:
    for d in dirs:
        for p in glob.glob(os.path.join(d, "out", "summary-*.json")):
            got = json.load(open(p)).get("staged_stale_pins")
            if got:
                return got
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--prs", required=True)
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--brief", help="write the short table by owner here")
    ap.add_argument("--intro", help="a text to start the report with; @LEAD@ and @FINDINGS@ are filled in")
    ap.add_argument("--findings")
    a = ap.parse_args(argv)
    prs = {p["number"]: p for p in json.load(open(a.prs))}
    rows = load(a.dirs)
    control = rows.pop(0, None)
    for n, p in prs.items():
        m = re.search(r"^Lane[: ]+([A-Za-z0-9_-]+)", p.get("body") or "", re.M)
        r = rows.setdefault(n, {"n": n, "status": "no record", "head": p["headRefOid"]})
        r.update(owner=m.group(1) if m else "unknown", branch=p["headRefName"], draft=p.get("isDraft"), title=p.get("title"))
        r["class"], r["detail"] = outcome(r)
    rows = {n: r for n, r in rows.items() if n in prs}
    L: list[str] = []
    say = L.append
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r["owner"]].append(r)
    tot = collections.Counter(r["class"] for r in rows.values())
    if a.intro:
        drv = collections.Counter(r["detail"] for r in rows.values() if r["class"] == ORDER[1])
        lead = (f"**Of {len(rows)} open pull requests against `main`, {tot['clean'] + tot[ORDER[1]]} restack onto the staged tree "
                f"by themselves ({tot[ORDER[1]]} of them only with the merge drivers registered), {tot[STOP_OWN]} stop at "
                f"`git merge main` before any move, and {tot[ORDER[3]]} stop at the layout"
                + (f"; {tot[ORDER[4]] + tot[ORDER[5]]} are not measured" if tot[ORDER[4]] + tot[ORDER[5]] else "") + ".** [run by me]"
                + f" The drivers are needed {', '.join(f'{k}: {v}' for k, v in drv.most_common())}.")
        say(open(a.intro).read().replace("@LEAD@", lead)
            .replace("@FINDINGS@", open(a.findings).read().strip() if a.findings else ""))
    say("## The table by owner")
    say("")
    say(f"heads: {len(rows)}; " + "; ".join(f"{k}: {tot[k]}" for k in ORDER if tot[k]))
    if control:
        say(f"control (main itself across the staging): {control.get('status')}"
            f"{' ' + str(control.get('result'))[:12] if control.get('result') else ''} {control.get('error') or ''}")
    say("")
    say("| owner | heads | clean | clean, drivers only | stops at `git merge main` | stops at the layout | cannot run | not measured |")
    say("|---|---|---|---|---|---|---|---|")

    def cell(rs: list[dict], k: str) -> str:
        hit = sorted(r["n"] for r in rs if r["class"] == k)
        return str(len(hit))

    for owner, rs in sorted(by.items(), key=lambda kv: (-sum(r["class"] in ORDER[2:] for r in kv[1]), -len(kv[1]), kv[0])):
        say(f"| {owner} | {len(rs)} | " + " | ".join(cell(rs, k) for k in ORDER) + " |")
    say(f"| all | {len(rows)} | " + " | ".join(str(tot[k]) for k in ORDER) + " |")
    say("")
    left = [r for r in rows.values() if r.get("left_behind")]
    say(f"clean, with files of its own left in a directory the layout emptied: {len(left)}: "
        + ", ".join(f"#{r['n']} ({r['owner']}, {len(r['left_behind'])})" for r in sorted(left, key=lambda r: r["n"])))
    say("")
    base_pins = staged_pins(a.dirs)
    if base_pins:
        say(f"the staged tree's own stale pins (`pins.py --stale` exits {base_pins.get('exit')}): "
            + ", ".join(f"`{x}`" for x in sorted(pin_names(base_pins))))
        own = {r["n"]: sorted(pin_names(r.get("stale_pins")) - pin_names(base_pins)) for r in rows.values()
               if r.get("stale_pins")}
        more = {n: v for n, v in own.items() if v}
        say(f"restacked without the pins step: {len(own)} heads; with a stale pin beyond the staged tree's: {len(more)}"
            + "".join(f"\n- #{n} ({rows[n]['owner']}): {', '.join('`%s`' % x for x in v[:6])}{' ...' if len(v) > 6 else ''}"
                      for n, v in sorted(more.items())))
        say("")
    say("## Files most heads stop on")
    say("")
    for title, keep in (("at the layout (restack's own steps)", lambda r: r["class"] == "stops at the layout"),
                        ("at `git merge main` (the branch's own, before any move)", lambda r: r["class"] == STOP_OWN)):
        cnt, kinds, who = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(list)
        alone: collections.Counter = collections.Counter()
        stopped = [r for r in rows.values() if keep(r)]
        for r in rows.values():
            if keep(r):
                if len(r.get("files") or []) == 1:
                    alone[r["files"][0]] += 1
                for f in r.get("files") or []:
                    cnt[f] += 1
                    kinds[f][(r.get("kinds") or {}).get(f, "?")] += 1
                    who[f].append(r["n"])
        say(f"### {title}")
        say("")
        say(f"{len(stopped)} heads, {len(cnt)} files. \"alone\" counts the heads this file is the only stop of at that"
            " step (a later step may still stop them: a restack ends at its first conflict).")
        say("")
        say("| file | heads | alone | kinds | pull requests |")
        say("|---|---|---|---|---|")
        for f, c in cnt.most_common(20):
            ns = sorted(who[f])
            say(f"| `{f}` | {c} | {alone[f]} | {', '.join(f'{k} {v}' for k, v in kinds[f].most_common())} | "
                f"{', '.join('#%d' % x for x in ns[:12])}{' ...' if len(ns) > 12 else ''} |")
        say("")
        groups = {
            "a new file under a retired path takes its directory's row (`no row in the map`)":
                lambda f, k: k == "no row in the map",
            "every `pyproject.toml`": lambda f, k: f.endswith("pyproject.toml"),
            "every `lean-audit.json`, `unread-caps.json` and lakefile": lambda f, k: f.endswith(
                ("lean-audit.json", "unread-caps.json", "lakefile.toml", "lakefile.lean", "lake-manifest.json")),
            "every file under `tools/move/`": lambda f, k: f.startswith("tools/move/"),
            "every allowlist under `integrations/vllm/tests/` or `adapters/vllm/tests/`": lambda f, k: f.startswith(
                ("integrations/vllm/tests/", "adapters/vllm/tests/")) and f.endswith((".txt", ".json", ".toml")),
            "every `.md`": lambda f, k: f.endswith(".md"),
        }
        freed = []
        for name, hit in groups.items():
            ns = sorted(r["n"] for r in stopped if r.get("files")
                        and all(hit(f, (r.get("kinds") or {}).get(f, "?")) for f in r["files"]))
            if ns:
                freed.append((len(ns), name, ns))
        for f in cnt:
            if alone[f] > 1:
                freed.append((alone[f], f"`{f}` alone", sorted(r["n"] for r in stopped if r.get("files") == [f])))
        if freed:
            say("One hand rule, by the heads it would take past this step:")
            say("")
            for c, name, ns in sorted(freed, key=lambda x: -x[0])[:8]:
                say(f"- {c}: {name} ({', '.join('#%d' % x for x in ns[:14])}{' ...' if len(ns) > 14 else ''})")
            say("")
    say("## Each head that is not plainly clean")
    say("")
    for owner, rs in sorted(by.items()):
        bad = [r for r in rs if r["class"] != "clean" or r.get("left_behind")]
        if not bad:
            continue
        say(f"### {owner} ({len(bad)} of {len(rs)})")
        say("")
        for r in sorted(bad, key=lambda r: (ORDER.index(r["class"]), r["n"])):
            fs = r.get("files") or []
            kinds = r.get("kinds") or {}
            more = f", and {len(fs) - 8} more" if len(fs) > 8 else ""
            what = ", ".join(f"`{f}` ({kinds.get(f, '?')})" for f in fs[:8]) + more
            if r["class"] == "clean with the drivers only":
                what = "; ".join(f"{k}: {', '.join('`%s`' % x for x in v[:5])}{' ...' if len(v) > 5 else ''}"
                                 for k, v in r["without_drivers"].items())
            if r.get("left_behind"):
                what += f" left behind: {', '.join('`%s`' % x for x in r['left_behind'][:6])}" \
                        f"{' ...' if len(r['left_behind']) > 6 else ''}"
            if r["class"] in ("restack cannot run", "not measured"):
                what = (r.get("error") or "")[-400:].replace("\n", " ")
            say(f"- #{r['n']} `{str(r.get('head'))[:12]}`{' [draft]' if r.get('draft') else ''} {r['class']}"
                f"{' (' + r['detail'] + ')' if r['detail'] else ''}: {what}")
        say("")
    say("## Each clean head's restacked commit")
    say("")
    for r in sorted(rows.values(), key=lambda r: r["n"]):
        if r.get("result"):
            say(f"- #{r['n']} {r['owner']} `{r['head'][:12]}` -> `{r['result']}` ({r['run']})")
    if a.brief:
        B = ["| owner | heads | clean | drivers only | stops: merge main | stops: layout | not measured |", "|---|---|---|---|---|---|---|"]
        rest = []
        for owner, rs in sorted(by.items(), key=lambda kv: (-sum(r["class"] in ORDER[2:] for r in kv[1]), -len(kv[1]), kv[0])):
            c = collections.Counter(r["class"] for r in rs)
            if len(rs) <= 6:
                rest.append(f"{owner} {len(rs)}/{c['clean'] + c[ORDER[1]]}/{c[STOP_OWN]}/{c[ORDER[3]]}/{c[ORDER[4]] + c[ORDER[5]]}")
                continue
            B.append(f"| {owner} | {len(rs)} | {c['clean']} | {c[ORDER[1]]} | {c[STOP_OWN]} | {c[ORDER[3]]} | {c[ORDER[4]] + c[ORDER[5]]} |")
        B.append(f"| all | {len(rows)} | {tot['clean']} | {tot[ORDER[1]]} | {tot[STOP_OWN]} | {tot[ORDER[3]]} | {tot[ORDER[4]] + tot[ORDER[5]]} |")
        B.append("smaller owners (heads/clean/merge main/layout/not measured): " + "; ".join(rest))
        with open(a.brief, "w") as f:
            f.write("\n".join(B) + "\n")
    text = "\n".join(L) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"control": control, "rows": sorted(rows.values(), key=lambda r: r["n"])}, f, indent=1)
    print(text[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
