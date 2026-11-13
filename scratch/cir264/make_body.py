"""Writes /tmp/cir264/body/new.md from old.md (PR #9693's body as read from GitHub) and the final run's records:
python3 make_body.py HEAD SUITES_RUN CC_RUN.  Every replaced passage must be found exactly once."""
import json
import os
import re
import sys
from collections import Counter

HEAD, SUITES, CC = sys.argv[1:4]
H12 = HEAD[:12]
runs = os.path.expanduser("~/.research/runs/")
old = open("/tmp/cir264/body/old.md").read()
body = old


def swap(a: str, b: str) -> None:
    global body
    assert body.count(a) == 1, (body.count(a), a[:80])
    body = body.replace(a, b)


def slot(run: str) -> tuple[str, str, str]:
    s = json.load(open(runs + run + "/slot.json"))
    st = json.load(open(runs + run + "/status.json"))
    from datetime import datetime
    f = lambda t: datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")
    wait = (f(s["took"]) - f(s["since"])).total_seconds()
    held = (f(st["ended_utc"]) - f(s["took"])).total_seconds()
    fmt = lambda x: f"{x:.0f} s" if x < 90 else f"{x / 60:.1f} min"
    return ("no wait" if wait < 5 else f"waited {fmt(wait)}"), f"held {fmt(held)}", json.load(open(runs + run + "/job.json"))["run"]["remote"]["machine"]


# --- the case outside the domain
swap("[run `r20261113-144638-5f15`, below; each is exact over the uniform's 2^32 words given the rows, and its error is the "
     "spread over requests]. Counted directly, 0 of the 460,800 requests met it [run]. These are two kinds of synthetic rows, "
     "not a served model's. The lead carries the case as that stated rate, with no mend to vLLM's kernel and no statement of "
     "what the card stores after a split, and has asked vllm's and proofs' word by Fri 13 Nov 3 PM PST [the lead's decision, "
     "read on CIR-264, Fri 13 Nov 7:29 AM PST].",
     "[run `r20261113-144638-5f15`, below; both over synthetic rows at K = 5, V = 154,880, TILE = 8192; each is exact over the "
     "uniform's 2^32 words given the rows, and its error is the spread over requests]. Counted directly, 0 of the 460,800 "
     "requests met it [run]. These are two kinds of synthetic rows, not a served model's: the share of a served model's rows "
     "whose two sums differ is not measured, and **if every row's two sums differed the rate would be about 1.4e-6 a request** "
     "[computed from that run's report, on the first kind of rows: 9.35e-11 a reached position over the 2.31e-4 of them whose "
     "sums differ, times 3.55 reached positions a request]. **When a step in the case is sampled, an honest server's proof of "
     "it can fail, since what the card stores after the split need not be lane 0's arithmetic: there is no retry, and the "
     "verifier cannot tell that failure from a dishonest server's.** The module's docstring now states the rate, what it rests "
     "on and that sentence (without the run's id, which the tree does not cite). The case is carried as that stated rate, with "
     "no mend to vLLM's kernel and no statement of what the card stores after a split, and both answers the lead asked for are "
     "in [the lead's decision and both answers, read on CIR-264]. vllm (Fri 13 Nov 7:33 AM PST): it keeps the target's rows "
     "after temperature and top-p from no served run with a draft, so no served share replaces the synthetic one, and nothing "
     "of vllm's stands against this. proofs (Fri 13 Nov 10:12 AM PST): no statement of theirs names the term; it comes back to "
     "them only if someone writes a statement that bounds an honest server's refusals over a monitoring period, or one that "
     "says an honest server is always accepted.")

# --- circuit-check
cw, ch, cm = slot(CC)
swap("in a clone of `66f957e940d7` (run `r20261113-042405-da93`): **31 targets",
     f"in a clone of the head `{H12}` (run `{CC}`): **31 targets")
swap("Each target, as the run printed it (`u` units, `b` committed bits):",
     "Each target, as the run printed it, without its timings (`u` units, `b` committed bits); every line is the one the first "
     "head `66f957e940d7` gave before the carry (run `r20261113-042405-da93`) [compared by script]:")

# --- the head
r = json.load(open(runs + SUITES + "/out/suites.json"))
rr = json.load(open(runs + SUITES + "/out/repository/suites.json"))
assert set(e["status"] for e in r) <= {"passed", "cached"}, Counter(e["status"] for e in r)
assert rr[0]["status"] == "passed" and len(rr) == 1
tot = Counter()
for e in r:
    tot.update(e["counts"])
assert tot["failed"] == 0 and tot["error"] == 0
assert not any(e.get("outside") or e.get("child_outside") or e.get("failed_tests") for e in r + rr)
by = {e["name"]: e["counts"] for e in r}
cached = [e["name"] for e in r if e["status"] == "cached"]
ran = len(r) - len(cached)
logs = runs + SUITES + "/out/"
for log in ("repository/logs/repository.log", "logs/adapters_vllm.log"):
    assert "sampled_accept" not in open(logs + log, errors="replace").read(), log
gate = open(runs + SUITES + "/out/ratchet_gate.txt").read().strip().splitlines()
m = re.match(r"ratchet gate (\w+)\.\.(\w+): (\d+) change\(s\), (\d+) failed", gate[0])
assert m and m.group(2) == H12 and m.group(4) == "0", gate[0]
pins = open(runs + SUITES + "/out/pins_stale.txt").read()
assert pins == "", pins
sw, sh, sm = slot(SUITES)

swap("The head is `66f957e940d7428d0904626920d265a4f38e93c7`. The capture and its replay ran at its parent `903c545eb3d3`; the "
     "head differs from it only in `evidence.json`'s entry for those two runs and in the conformance text that names them. "
     "Times are UTC.",
     f"The head is `{HEAD}`. It holds the pull request's first head `66f957e940d7428d0904626920d265a4f38e93c7` carried across "
     "the 29 layout moves that landed on `main` (the lead's job `r20261113-155454-6b20`, `tools/move/restack.py`, to "
     "`7495e5e85729`) [the lead's report; I read that the head holds both and that the carry changed four lines of this "
     "change: `sampled_accept.py` imports its primitives from `verity.circuits.ops`, where it imported them from "
     "`verity.definitions.ops`], a plain merge of `main` at `af9b13440693` with no conflict, and two commits: `f1d2ecc7050a` "
     "(two import blocks sorted and one string join parenthesized under vLLM's rules, in the three files under "
     "`adapters/vllm/tests/`) and `8e96f9e2a3d3` (the two sentences on the case outside the domain, in the module's "
     "docstring). Against `main` it is the same 7 files: 1,652 insertions, 1 deletion [git]. Each run below names the commit "
     "it ran at. The capture and its replay ran before the carry at `903c545eb3d3`, the first head's parent, and were not run "
     "again; since that commit the capture script differs by the parentheses around one string join of its report, and the "
     "vectors module by the order of two imports [git]. The checks ran at the head. Times are UTC.")

swap("- **`r20261113-034537-8871`, the capture on a card** (node A,", "- **`r20261113-034537-8871`, the capture on a card, at `903c545eb3d3`** (node A,")
swap("- **`r20261113-034958-1553`, the capture replayed off the card** (waited", "- **`r20261113-034958-1553`, the capture replayed off the card, at `903c545eb3d3`** (waited")

a = body.index("- **`r20261113-042405-da93`, circuit-check at the head**")
b = body.index("- **`r20261112-135317-f990`, the kept PTX**")
vllm, defs, circ, repo = by["verity-vllm"], by["verity/definitions"], by["verity/circuits"], by["repository"]
checks = (
    f"- **`{CC}`, circuit-check at the head `{H12}`** ({cw}, {ch}): the report above. **It wrote no pin, and the docstring's "
    "two sentences moved nothing it reads:** its report is the one `r20261113-180332-58cd` gave at `f1d2ecc7050a`, before "
    "those sentences, field for field but for timings and memory [compared by script].\n"
    f"- **`{SUITES}`, the suites this change reaches, `repository` whole, the ratchet gate and the pins, at the head `{H12}`** "
    f"({sw}, {sh}; 64 GB). `suites.py --changed af9b13440693 --quick`, the base named by the commit of `main` that the head "
    f"holds: **{len(r)} suites: {len(r)} passed ({len(cached)} cached), 0 failed**; {tot['passed']:,} tests passed and "
    f"{tot['skipped']:,} skipped over the {ran} that ran in this run [summed by me from `out/suites.json`], with no read "
    f"outside a suite's inputs. The {len(cached)} cached are passes the node kept from another commit's run, reused because "
    f"every file that run read is as it was ({', '.join('`' + c + '`' for c in cached)}); none holds a file of this change. "
    f"Among the {ran}: `verity-vllm` {vllm['passed']:,} passed, {vllm['skipped']:,} skipped (it ran `test_sampled_accept.py` "
    f"whole, slow tests included); `verity/definitions` {defs['passed']:,} passed (its `test_catalog.py` holds the registered "
    f"ids to `catalog_ids.json`); `verity/circuits` {circ['passed']:,} passed; `repository` {repo['passed']:,} passed. Then "
    f"`repository` whole: **{rr[0]['counts']['passed']:,} passed, 0 failed**. Then `adapters/vllm/tools/ratchet_gate.py "
    f"af9b13440693 HEAD`: **{m.group(3)} change, 0 failed** (no new environment read, silent except, lane-text site or "
    "retired-term site). Then `tools/verity/check/pins.py --stale`: exit 0, it names no pin. No line of the `repository` log "
    "or of `verity-vllm`'s names a file of this change [read: both logs "
    "searched]. `out/suites.json` and each suite's log are in the run.\n"
    "- **The same checks one commit earlier, at `f1d2ecc7050a`** (before the docstring's two sentences): "
    "`r20261113-173733-65a2`, `pins.py --stale`, exit 0, no pin named (waited 22.5 min for a slot, held it under a second); "
    "`r20261113-180332-58cd`, circuit-check, the same report (waited 22 s, held 71 s); `r20261113-180800-47af`, the same "
    "three steps (no wait, held 57.8 min): 62 suites passed (0 cached), 0 failed, 39,156 tests passed and 570 skipped; `repository` "
    "whole 361 passed; the ratchet gate 1 change, 0 failed.\n"
    "- **Before the carry, at the first head `66f957e940d7`:** `r20261113-042405-da93` (circuit-check, the same report), "
    "`r20261113-044914-2cc1` (`suites.py --changed 3498d4ad65 --quick`: 59 suites passed, 0 failed) and "
    "`r20261113-065139-a23b` (`repository` whole 352 passed; the ratchet gate 1 change, 0 failed).\n")
body = body[:a] + checks + body[b:]

swap("which is cut from this head, adds only files under `scratch/cir264/` and is not for landing.",
     "which is cut from the first head `66f957e940d7`, adds only files under `scratch/cir264/` and is not for landing.")
swap("- Locally at the head: the retired-terms and lane-text lints (`tests/lint/test_repo_retired_terms.py`, "
     "`test_repo_lane_text.py`): 17 passed, 2 bookkeeping warnings, neither naming a file of this change.",
     "- Locally at the head: ruff under the root's rules passes on the four files the two commits touch, and under the "
     "integration's (`adapters/vllm/ruff.toml`) and vLLM's (`adapters/vllm/tests/lint/ruff_vllm.toml`) on the three of them "
     "under `adapters/vllm/tests/`; the lane-text and retired-terms lints' own counters (`sites` of `tests/lint/test_repo_lane_text.py` and `test_repo_retired_terms.py`) "
     "find 0 sites in `sampled_accept.py` [run locally]. Both lints' tests are in `repository`, above.")

swap("- A merge of `main`: every run above is at this branch's own head, cut from `3498d4ad65`.",
     f"- The capture on a card and its replay at the carried head: both ran before the carry, at `903c545eb3d3`. What ran at "
     f"the head `{H12}` (the carried head `7495e5e85729`, `main` at `af9b13440693` merged in, and the two commits above): "
     "circuit-check on the sidecar, the quick tier of the 62 suites the change reaches, `repository` whole, the ratchet gate "
     "and `pins.py --stale`. `main` has moved on since that merge; no run is at a merge with a later `main`.")

assert "Three style warnings" not in body and "3 PM PST" not in body
assert body.split("\n")[:3] == ["Lane circuits", "Refs CIR-264", "Goal vllm/production-vllm-reproducible"]
open("/tmp/cir264/body/new.md", "w").write(body)
print(len(old), "->", len(body), "chars;", len(body.encode()), "bytes")
