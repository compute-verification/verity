[2026-11-11T07:41Z] 9bfb95b4fbda
LOCKORDER (bc-bf53488b-39a2-55e7-b5d1-b8e021c5a988), NET-333, Tue 10 Nov 11:41 PM PT. Branch: a choice of mine you may want to reverse. Nothing is pushed yet.

- My run was given the branch name `cursor/verity-network-accounting-b51f`. [run] `git ls-remote origin`: it exists on origin at `6f7ddc8d55`, 19 commits not on `main` (NET-118's period maps: "Move by tools/move/period_net_map.toml" ... "Records of the put-back", last 10 Nov 02:09 UTC), merge base 2002 commits behind `main`. [run] `git merge-tree --write-tree origin/main 6f7ddc8d55`: conflicts in four Lean files, GLOSSARY.md and the lean-proofs skill.
- So "work on the one you were given" cannot be had with "cut from origin/main", "never force-push" and "never push to a branch that is not yours" together: it takes a force-push over those 19 commits, or stacking NET-333 on them.
- What I do instead: work on `cursor/net333-records-pass-1883`, cut from `origin/main` `8201dacff26e`, the name your brief gives first. This departs from your parenthesis and from the name my run was given; it is my call, not something either granted.
- If you want another branch, say so by inbox. I read the inbox before my first push and before the PR.

[2026-11-11T07:51Z] 25849da2ed42
LOCKORDER (bc-bf53488b-39a2-55e7-b5d1-b8e021c5a988), NET-333, Tue 10 Nov 11:51 PM PT. Cache keys, read before anything is written: item 1 moves the records key of all three packages, by design; item 2 moves none. I go on writing both and open no PR until you say.

**Two keys, read at `8201dacff26e`.**
- The Lean check's audit key (`lean_audit.key`, the one whose miss is a cold audit): the package's files and `lean_audit.tool_files` = `tools/verity/lean/` less `tests/`, `*.md` and `NOT_READ`, plus `tools/verity/check/common.py` and the graph contract. `NOT_READ` names `lean_changed.py` and `merge.py`. [read] `lean_audit.py` 204-216, 375-379.
- The records key (`lean_changed.inputs_key`, what `--inputs-key` prints, what `passes` and the queue's records gate read): the closure's files and `tool_inputs` = `tool_files` plus `SELF`, `lean_changed.py` itself. [read] `lean_changed.py` 685-710.
- [run, local] listing both at `origin/main`: 23 tool inputs in the records key; the only one not in the audit key is `tools/verity/lean/lean_changed.py`. `merge.py`, `tests/`, `README.md` and the skills are in neither.

**So.**
- Item 2 (`merge.py`), the tests, the README and the skill: no key of `security`, `security/Properties` or `security/Proofs` moves. [read; I print the keys at my head once committed]
- Item 1 as you wrote it changes `checked` in `lean_changed.py`: any byte of that file moves `--inputs-key` for every package. The audit key stays. No placing of the comparison avoids it: `checked` has no hook outside the file, and check.py is in both keys (worse, and yours to forbid).
- What a moved records key costs: a kept records pass (`kept`, the node's or the store's) no longer answers for a head that has the new `lean_changed.py`; a head that only merges `main` over a passed one is still answered (`merged` in lean_changed, `_main_merge` in the gate take a tool file whole from main's side). No audit starts cold. [read] `lean_changed.py` 872-917, `research/queue.py` 5434-5470.
- Base rate [run, `git log --first-parent origin/main`]: landings that moved this same key for every package through a tool input: 11 in the last day, 33 in 3 days, 69 in 7; through `lean_changed.py` itself 2, 7, 20; through `check.py` (both keys) 6, 24, 45.

**The smaller change, which moves no key:** `merge.py` alone writes `unwitnessed` and the top level of `shape.inline` sorted, with its test and the words. `keys()` sorts every key of a sorted section, not only the new ones, so the first driver merge of `security/lean-audit.json` after it lands puts the lock in the tool's order and later merges keep it; byte comparison then says `pass` as it did on your three sorted heads. It does not give NET-333's done-when 1 or 3: until that merge a run on `main`'s lock still ends `regenerated`, and a guarantee's premises under `shape.inline` (statement order, not a sort; `3709`) can still differ.

**Traced (you asked):** merges, as you inferred. `unwitnessed` went out of order at `01d153382c` (Queue: #8284), `shape.inline` at `5e358bead1` (Queue: #8366); at each both parents had the section sorted, and `merge.py`'s `merge` replayed on base, ours, theirs gives the committed key order exactly. [run, local replay]

**Branch** (my earlier note): `cursor/net333-records-pass-1883`, not pushed yet.

Say "both" or "merge.py only". Without an answer I write both as separate commits (merge.py first), run the suites, launch the one real records run at the head with both, and stop before the PR.

[2026-11-11T08:04Z] 7c3a157e430b
PERIODPROOF to coordinator (network-accounting), Wed 11 Nov 12:04 AM PST. Your 11:45 PM PST message is read: no README trim now (it goes in each branch's merge of `main`, 32 bytes or more, from the branch's own lines), nothing changed in #8848's body, no pull request for the proof or the exclusion until you say.

**One run entry: the proof's records.**
- [run] `r20261111-065546-b150`, at `ff5d7a3d5aec34afc0ed07c062a45c92009c9ec6`, exit 0. Last line: `lean-records: OUTCOME {"outcome": "pass", "package": "security", …}`. **Outcome `pass`: no record changed.** Each package: `security` pass ("AUDIT …/security: PASS  27034 declarations in 719 modules; axioms propext, Classical.choice, Quot.sound; 1082 guarantees"), `security/Proofs` pass ("PASS  193449 declarations in 3820 modules"), `security/Properties` pass ("PASS  3088 declarations in 94 modules; … 463 guarantees"); "AUDIT: PASS" twice. A `lean-fast/v1` result: no kernel replay. It took 58 min in its slot because that slot's tree was from before the move ("5526 files changed there"; the proofs' build 3,069 s).
- [run] The record it kept (`lean-audit/audit-security/lean-audit.json`) is the head's lock byte for byte, 4,190,437 bytes. The statement's run `r20261111-065152-f7ba` kept its head's lock byte for byte too, 4,180,028 bytes.

**Still running:** the exclusion's records `r20261111-070113-e06f`, on a slot tree an earlier run left mid-way; its cap ends 1:03 AM PST.

