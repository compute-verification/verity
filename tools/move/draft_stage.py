#!/usr/bin/env python3
"""DRAFT, a test of the move tools and not of record: stage draft layout maps as one chain, going on past a refusal, timed.

Run at the root of a clone (a node's `research run --cwd clone`), with a git identity in the environment:

    python3 tools/move/draft_stage.py --out "$OUT" --branch NAME [--no-lock] MAP...

For each map, in order, on the commit the one before left:

1. `layout.py --map MAP --dry-run --report-splits --report FILE`, timed; a refusal skips the map;
2. the references to a split directory its dry run names are written to the map's `[split]` and committed (not a move);
3. `stage.py`'s own `Stage.move`: the starting commit's `layout.py`, the commit, `move_check.py`, the second run, each script
   call timed. With `--token-extras`, `move_check.py` is given each `[tokens]` row of the map as `--map OLD=NEW` (the path
   and, under `verity/`, the dotted name): a reference to a directory the map splits is otherwise no part of a pure move. A refusal after the commit keeps the commit, recorded as not certified, so that the next map still runs.

`OUT/results.json` is rewritten after every map; at the end `OUT/cert.json` holds the certified moves' reports and
`OUT/move.bundle` the branch over the starting commit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage", HERE / "stage.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)

LAYOUT = "tools/move/layout.py"
SPLIT_NOTE = "references to a split directory [split] doesn't acknowledge"
REASON = "draft: not reviewed"


def git(cwd: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, check=check, capture_output=True,
                          text=True).stdout.strip()


def say(*words) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *words, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--token-extras", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    start = git(repo, "rev-parse", "HEAD")
    out = a.out.resolve()
    for d in ("dry", "stage", "check", "rerun"):
        (out / d).mkdir(parents=True, exist_ok=True)
    s = ST.Stage(repo, start, extra=[], no_lock=a.no_lock, rerun=True, log=lambda x: say("  |", str(x).replace("\n", "\n  | ")))
    calls: list[dict] = []
    name = ""
    inner = s.run

    def run(script: str, *args: str):
        t = time.monotonic()
        r = inner(script, *args)
        n = len(calls)
        calls.append({"script": script, "args": [x for x in args if not x.startswith("/")], "seconds": round(time.monotonic() - t, 1),
                      "rc": r.returncode})
        say(f"  {script} {' '.join(calls[-1]['args'])}: rc {r.returncode}, {calls[-1]['seconds']} s")
        if script == ST.MOVE_CHECK:
            (out / "check" / f"{name}.json").write_text(r.stdout)
            (out / "check" / f"{name}.err").write_text(r.stderr)
        else:
            (out / "stage" / f"{name}.{n}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        return r

    s.run = run
    results: list[dict] = []
    certified: list[dict] = []
    parent = start

    def save() -> None:
        (out / "results.json").write_text(json.dumps({"start": start, "head": parent, "maps": results}, indent=1) + "\n")

    try:
        for m in a.maps:
            name = Path(m).stem
            rec: dict = {"map": m, "parent": parent}
            results.append(rec)
            say(f"== {m} on {parent[:12]}")
            calls.clear()
            try:
                report = out / "dry" / f"{name}.json"
                r = run(LAYOUT, "--map", m, "--dry-run", "--no-lock", "--report-splits", "--report", str(report), "--root", str(s.wt))
                rec["dry"] = {"seconds": calls[-1]["seconds"], "rc": r.returncode}
                if r.returncode:
                    rec["status"] = "refused at the dry run"
                    rec["refusal"] = (r.stderr.strip() or r.stdout.strip())[-6000:]
                    say("  refused at the dry run:", rec["refusal"][:1500])
                    continue
                doc = json.loads(report.read_text())
                splits = sorted({x.split(" (", 1)[0] for x in doc["notes"].get(SPLIT_NOTE, [])})
                rec["dry"].update(changes=doc["changes"], files_rewritten=len(doc["files"]), moves=len(doc["moves"]),
                                  notes={k: len(v) for k, v in doc["notes"].items()}, split_rows=len(splits))
                if splits:
                    f = s.wt / m
                    rows = "".join(f"{json.dumps(k)} = {json.dumps(REASON)}\n" for k in splits)
                    text = f.read_text()
                    if "\n[split]\n" in text:           # the map has the table: the rows go under its header
                        text = text.replace("\n[split]\n", "\n[split]\n" + rows, 1)
                    else:
                        text = text.rstrip("\n") + "\n\n[split]\n" + rows
                    f.write_text(text)
                    git(s.wt, "add", "--", m)
                    git(s.wt, "commit", "-q", "--no-verify", "-m",
                        f"draft: {m}'s [split] rows, from its dry run\n\n{len(splits)} references to a directory the map splits, "
                        "none of them reviewed.\n")
                    parent = git(s.wt, "rev-parse", "HEAD")
                    rec["split_commit"] = parent
                first = len(calls)
                tokens = []
                for o, n in tomllib.loads((s.wt / m).read_text()).get("tokens", {}).items():
                    tokens.append(f"--map={o}={n}")
                    if o.startswith("verity/") and n.startswith("verity/"):
                        tokens.append(f"--map={o.replace('/', '.')}={n.replace('/', '.')}")
                s.extra = list(tokens) if a.token_extras else []
                rec["extras"] = len(s.extra)
                try:
                    mv = s.move(m, parent)
                    rec.update(status="certified", commit=mv["commit"], summary=mv["summary"])
                    certified.append(mv)
                except ST.Refused as e:
                    head = git(s.wt, "rev-parse", "HEAD")
                    rec["refusal"] = str(e)[-8000:]
                    say("  refused:", rec["refusal"][:2500])
                    status = git(s.wt, "status", "--porcelain").splitlines()
                    if head != parent:
                        after_check = len(calls) - first >= 3
                        rec.update(commit=head, status="certified, but the second run refused" if after_check
                                   else "committed, not certified")
                        if status:
                            rec["rerun_changed"] = {"paths": len(status), "first": status[:40]}
                            (out / "rerun" / f"{name}.diff").write_text(git(s.wt, "diff", "HEAD", "--stat=200", check=False)[-20000:]
                                                                        + "\n\n" + git(s.wt, "diff", "HEAD", check=False)[:200000])
                    else:
                        rec["status"] = "refused, no commit"
                    if head != parent and not after_check and tokens and not a.token_extras:
                        # what the map's own [tokens] rows would explain, were they given to move_check.py as --map
                        keep = name
                        name = f"{keep}.with-tokens"
                        r = run(ST.MOVE_CHECK, parent, head, *tokens, "--json")
                        name = keep
                        try:
                            rec["with_tokens"] = {"rc": r.returncode, "refusals": len(json.loads(r.stdout).get("refusals", []))}
                        except ValueError:
                            rec["with_tokens"] = {"rc": r.returncode, "tail": (r.stderr or r.stdout)[-600:]}
                    if head != parent and not after_check:
                        # stage.py stops at the refusal; the second run is still worth knowing
                        git(s.wt, "reset", "-q", "--hard", check=False)
                        git(s.wt, "clean", "-fdq", check=False)
                        script, *args = s.command("paths", m)
                        r = run(script, *ST.RS.BRANCH_ARGS.get(script, ()), *args, "--root", str(s.wt))
                        again = git(s.wt, "status", "--porcelain").splitlines()
                        rec["second_run"] = {"rc": r.returncode, "paths_changed": len(again), "first": again[:40],
                                             "tail": (r.stderr.strip() or r.stdout.strip())[-1500:] if r.returncode else ""}
                        if again:
                            (out / "rerun" / f"{name}.diff").write_text(git(s.wt, "diff", "HEAD", "--stat=200", check=False)[-20000:]
                                                                        + "\n\n" + git(s.wt, "diff", "HEAD", check=False)[:200000])
                        say(f"  second run on the uncertified commit: rc {r.returncode}, {len(again)} paths changed")
            except Exception:  # noqa: BLE001  the chain goes on, and the record says what stopped this map
                rec["status"] = "error"
                rec["error"] = traceback.format_exc()[-6000:]
                say("  error:", rec["error"])
            finally:
                git(s.wt, "reset", "-q", "--hard", check=False)
                git(s.wt, "clean", "-fdq", check=False)
                parent = rec.get("commit") or git(s.wt, "rev-parse", "HEAD")
                rec["calls"] = list(calls)
                rec["seconds"] = round(sum(c["seconds"] for c in calls), 1)
                say(f"  -> {rec.get('status')} in {rec['seconds']} s; head {parent[:12]}")
                save()
    finally:
        s.close()
        if parent != start:
            git(repo, "branch", "-f", a.branch, parent)
            git(repo, "bundle", "create", "-q", str(out / "move.bundle"), f"refs/heads/{a.branch}", f"^{start}")
        (out / "cert.json").write_text(json.dumps({"base": start, "head": parent, "moves": certified}, indent=1) + "\n")
        save()
    say(f"done: {sum(r.get('status') == 'certified' for r in results)} of {len(results)} certified; head {parent[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
