[2026-11-12T08:55Z] 6cd038a7e3cd
CROSSTICK, 12:56 AM PST Thu 12 Nov. The count's pull request is open and not marked.

**[#9216](https://github.com/compute-verification/verity/pull/9216)**, branch `cursor/na-span-count-e7c1`, head `e9f5475a56cddd5fe6b6f9563d4f966b79b223da`, on `main`, no label [ran: `research queue open`, then `gh pr view`]. Stacked on #9154 (`5dcf9585f5d3`) and #9188 (`a8d956c8dd14`); the registered read's call is its own commit (`57dd66f8585d`) above the merge of #9188, and one test-only commit is above that.

**Since my 12:23 AM line.**
- One more commit, tests only (`e9f5475a56cd`): a change to the registered hardware takes its forms in turn by the seed, and a Lean test replays every form on both sides (6 links, 5 keys, the two lone keys, 23 malformed sites and heads). The sampled difftest had reached only 3 of the 23 malformed forms, so I made it exhaustive. On the link and key refusals Lean and the mirror say the same words; the test asserts it.
- Your ruling 2 is one line of the body, under the checks.
- Lean's tip had moved again when I fetched: `a88eee998fa2`, past the `1ab0a942376e` you named. The merge check and the scratch merge are against `a88eee998fa2`.

**Runs at the head `e9f5475a56cd`, each launched once [ran].**
- The span difftest, `r20261112-083457-8f2c`: "passed verity/protocols/certifier/verifier/network 11 passed in 50 s"; "1 suites: 1 passed (0 cached), 0 failed". 600 sampled cases agree: 165 accepted, 435 rejected, each of the 46 changes seen.
- The package's Lean check, no `--update`, `r20261112-083524-b4d1`: "AUDIT: PASS", 1,771 declarations in 12 modules; axioms propext, Classical.choice, Quot.sound; 4 guarantees. Its note: 82 definitions read by no guarantee, 48 of them the count's.
- The root suite, `r20261112-083551-9e7b`: "passed repository 313 passed in 400 s"; "1 suites: 1 passed (0 cached), 0 failed".
- The scratch merge into lean's tip `a88eee998fa2` (local commit, not pushed), `r20261112-083616-7b58`, through `suites.py` under its read guard as you asked: the whole network suite without slow tests, "88 passed, 14 deselected"; then the span test module whole, "11 passed in 40 s".
- The changed suites (`--changed d6075b75d289 --quick --skip repository`), `r20261112-083603-e557`: still running at 12:55 AM, 21 suites passed and none failed so far, of 75. I do not wait on it; its limit is 240 minutes.
- `verity lean try --siblings` on `Span.lean`, `r20261112-081515-43c4`, at `57dd66f8585d`: "lean exit 0 in 26 s".

**Checked without a run [ran].** `git merge-tree` against lean's tip `a88eee998fa2` and against `main`: names no file. Ratchet gate from `main`'s tip to the head: retired terms 0 at the base, 7 at the head, the same seven sites, named by file in the body with your words; nothing else gained.

**No lock record moves.** The package's lock has two hand-written entries of mine (`exempt["NetworkCheck.SpanMain"]`, `layers["NetworkCheck.Span"]`, whose imports now hold `NetworkCheck.Registered`). No `--update` anywhere. The model lines' branch stays at `f2f78fd25cad`, with no lock entry, as you ruled.

**Two things I want you to see, both in the body under "Not checked".**
1. The registered read is opt-in. Parameters without `site` and `head` are read as before, so nothing in the count makes a caller hand the site over, and nothing holds the site to an accepted log.
2. An answer does not say whether its parameters held the site: an honest span's answer with the site is the one without it. The profile and the public output are as you ruled them. If a reader of an answer alone should be able to tell, the profile would need the head, and that is a change for you and soundness to rule on, not one I made.

**Not mine to fix, for you.** `research queue open` warned that NET-355 has no label naming a live goal.

**Not checked.** The whole `check`. The changed suites to their end, and their slow tests. The merge with lean's branch at any tip after `a88eee998fa2`. JSON strings with a lone surrogate escape between Lean and Python. The model lines under the package's whole Lean check.

[2026-11-12T09:00Z] a410addfa617
SCANLEAN to network-accounting (lead), Thu 12 Nov 1:30 AM PST. Part 2 alone does not pass your rule for a mark on `main`, so I have NOT opened a pull request. It is built and pushed on `main`'s side for you to see, and I am building it where it merges, on lean's landing tip. Say if you want the pull request on `main` opened anyway.

- **Built on `main`** [ran]: branch `cursor/network-accounting-scan-pad-4e7a`, head `fba526ef3840`, one commit, two files. The set is the template's alone (`set_of`: `unit-scan-<template>`), every member carries it from the host's read, and `stage` refuses a member whose staged `o0` frame domain is not the set's over its units. The test stages a group of two execution phases and holds four rounds (honest, chain, tampered, unread) to one set of domains computed once; it passes here, with the group test beside it (2 passed, 2.6 s, pytest on the one file; not yet through `suites.py`).
- **Your first check fails** [ran: `git merge-tree --write-tree --name-only`, my head against lean's tip `f3837f57d113`]: it names `benchmarks/network_traces/scan_zk.py` and `tests/test_scan_zk.py`, both mine. `main` alone against that tip names twelve files and neither of these.
- **Why, and it is not a matter of moving lines** [read]: lean's tip carries another `scan_zk.py` (610 lines differ). There the set name is made in `member()`, `f"unit-scan-{tp}-{name}-{mb.name}"`, and `stage()` only takes `mr.set_name`; on `main` it is made in `stage()`. So the two sides change the same thing in two places. A change on `main` placed to merge without a conflict would be silently undone by lean's `member()`.
- **On lean's tip the fix is one line** in `member()` and the same test, better there: that file commits each member's own rows host-side (`Frame`, `frame_binding`), so the test holds the registration's own `o0` domains, not a formula. I cut `cursor/network-accounting-okbit-constant-4e7a` from `f3837f57d113` now, part 2 as its first commit and part 3 on top (part 3's constant needs part 2's name anyway). Not opened.
- **Part 1 on lean's tip:** the same finding as my 12:55 AM PST line; `first` and `last` are one unit each there too [read].

Next, in this order: part 2 on lean's tip with its run; (2) read on lean's tip's files, since they are what the second way will change, by 9 AM PST; then part 3.

[2026-11-12T09:00Z] de31bd27f234
SCANLEAN to network-accounting (lead): a correction of a time only. My second line, stamped 1:30 AM PST, was sent at 1:00 AM PST Thu 12 Nov (`date`); the first, stamped 12:55 AM PST, at 12:52 AM PST. Nothing else in them changes.

[2026-11-12T09:15Z] b5385c07f6a6
SCANLEAN to network-accounting (coordinator), Thu 12 Nov 1:15 AM PST. Job 5 item (1), the line you asked for.

**#9228 is open, unmarked: part 2 alone (the frame domain's set name), head `c4a83b814a6f`, on `main` `19fb19a21e40`. Do not mark it as it stands: it meets lean's landing tip in both of its files. The same change on the landing tip is pushed, `cursor/network-accounting-okbit-constant-4e7a` at `4a07fca408fd`.**

- **Number, head:** https://github.com/compute-verification/verity/pull/9228, `c4a83b814a6f` (two commits: the change, then a rewording, since the ratchet gate failed the first on five uses of a retired name in my added lines; it passes now [ran]).
- **Runs, all `research run --on auto`, all running at 1:15 AM PST, none waited for a slot:**
  - `r20261112-091233-3c62`: the suites #9228 touches, fresh, at its head.
  - `r20261112-090554-e3a6`: the scan proved at the test class with two execution phases a registration, at #9228's first commit (same code). It shows whether Lean accepts two members sharing a frame domain.
  - `r20261112-091349-288a`: the changed suites on the landing-tip branch's head, under the read guard.
  - `r20261112-090552-0208`: the same scan run on the landing-tip branch's first commit (same code).
  - Two more, at the first commits before the rewording: `r20261112-090553-5ca8` (#9228's suites), `r20261112-090552-4c2c` (the landing tip's).
- **The stop list:** part 1 is not in it (stopped, your 12:57 AM PST comment has it). Part 2 touches nothing on the list: no Lean, the period check and its mirror read no frame domain of a member, no statement's text changes, and no kept record or fixture holds a scan member's port or set name [ran: `rg` over the tree's json, jsonl and toml].
- **Before a mark [ran]:** `git merge-tree` of `c4a83b814a6f` against the landing tip `f3837f57d113` (fetched 1:12 AM PST) names `benchmarks/network_traces/scan_zk.py` and `benchmarks/network_traces/tests/test_scan_zk.py`, both mine, beside twelve that are lean's own. It cannot be otherwise on `main`: there the set's name is made in `stage`, on the landing tip in `member()`. The body says so. Your call: land #9228 and lean resolves two files to my landing-tip branch's, or close it and the landing-tip commits ride with part 3.
- **The Goal line:** NET-357 carries no goal label, so `open` wrote none and warned. I wrote `Goal network/period-bound-checked`, which NET-352's description names; it is live.
- **One thing to know of part 2:** with the set naming no member, two members of one template and one unit count share a frame domain (`ingress-0` and `ingress-1`, `egress-0` and `egress-1`, the unread round's unit and `last`). No rule I read asks for distinct domains; the two scan runs are the proof either way.
- **Item (2):** started. I hold Thu 12 Nov 9 AM PST.

