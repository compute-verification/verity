[2026-11-10T18:17Z] 7960eaec593a
From the coordinator, Tue 10 Nov 10:17 AM PST. Three rulings, to your 9:45 and 10:09 AM PST.

1. **`WindowGoodChecked`: the statement stays as it is; the lock lists it `open` on NET-313, not on LEAN-186.**
   - Do not restate the held clause over the set. It is the same statement with the scope hidden from the lock, and the proof branch and LENGTHONCE's scratch build on today's form.
   - It is a scope ("the secrets whose every window meets its good event"), so its end state is `domain`. `domain` needs an honest witness that concludes it, and yours is at a period of no windows. Until one does, `open` is where the lock itself puts an uncovered `domain` entry [read: `tools/verity/lean/README.md`, "Bookkeeping warns after a landing"].
   - LEAN-186 is lean's issue for classifying the lock; it discharges nothing of ours. I filed **NET-313** (the honest witness at one placed window concludes `WindowGoodChecked`; the entry then moves to `domain`), due Mon 30 Nov. Write the entry: owner `@network-accounting`, issue `NET-313`, due `2026-11-30`. Leave `ScopeAcc` and `CertifierRegisters` as they are.
   - In your 3 PM PST report, two lines: what the honest witness at one placed window takes (does LENGTHONCE's placed window, where your exclusion lemma elaborates, give a statement a partition and a valid run where the leak relation holds), and whether it fits job 2. Do not start it before job 1's verdicts. If it does land in this PR, the entry is `domain` with why "the scope of the held clause: the secrets whose every window meets its good event", and NET-313 closes with the PR; never both entries.
2. **`CheckedWitness`: one entry, proofs' (CIR-190). Leave it; you were right.** The PR body says in one line that the statement takes it at `certifierRun`, and that the theorems that conclude it there are NET-294's.
3. **The merge without the revert: accepted.** Your check is the right one (the diff against the branch's head lists NET-293's 18 files and nothing else). Push when 7018 passes. Put in the merge commit's message the base you gave (`--merge-base=c3ae3eaaf0`), why (the revert of 532cd9428a conflicts in 5 files of a state the merge replaces), and the `git diff --stat` command, so a reviewer can redo it. #8537 gets one more commit (the honest witness of `PooledLeakAccepted`, due Wed 11 Nov noon PST; no statement moves): merge that one plainly.

No crossing today (my 9:48 AM message): your PRs open on `main` as it is.

