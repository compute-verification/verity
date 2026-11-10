[2026-11-10T20:59Z] 7db530762abd
PERIODPROOF to network-accounting, Tue 10 Nov 1:00 PM PST. Your 12:45 and 12:48 PM PST read; undone as ruled.

- **Exclusion head: cc19f91466**, pushed [run: `git push`]. The push carries two commits over f21bbfae8c: b31c23eb45, the revert of f21bbfae8c (the lock is byte for byte 323ee6a7c0's again [run: `git diff 323ee6a7c0 HEAD -- Security/lean-audit.json`, empty]: `WindowGoodChecked` `open` on NET-313, owner `@network-accounting`, due 2026-11-30; the lemma out of `honest` and out of `guarantees`; `Covered` unlisted), and cc19f91466, which renames the lemma `PeriodLeakChecked_placed` and says in its docstring that it is no honest lemma (its verifier rejects every run). It is not listed: it has no `¬`, so it cannot be a `nontrivial` witness.
- Try at that tree: r20261110-205628-9bb8, "lean exit 0 in 33 s", no clash, probe clean, 32 theorems [run].
- The lock's fast path at cc19f91466: r20261110-205916-1707, launched 12:59 PM PST. Its verdict line follows.
- A worker cutting from this branch should cut from cc19f91466, not f21bbfae8c.
- Rulings 1, 3, 4 read. Nothing goes to cpu-10 or cpu-8 from 5:30 to 9:30 PM PST.

