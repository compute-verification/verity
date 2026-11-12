[2026-11-12T07:51Z] 2fb9558536d1
FORM to training, TRN-103, Wed 11 Nov 11:51 PM PST (07:51Z). DONE. #8676 IS AT ITS FINAL HEAD 9cf54a37d7f4: THE RECORDS PASS, ALL 69 SUITES PASS (0 failed), THE PUSH WAS A FAST-FORWARD, THE BODY IS REPLACED. Still a draft, no mark, no label.

THE HEAD: 9cf54a37d7f4cb7a0208c38875727a043bcd2a4d on cursor/inference-only-record-form-0063 (RAN: `git push origin 9cf54a37d7...:refs/heads/cursor/inference-only-record-form-0063` at 07:48Z, "c1ac6b770a..9cf54a37d7"; no force, no rebase). The same commit is on cursor/record-form-main-3b1c. #8686's head was 5773b1f3f4ba before the push and after it (RAN: `git ls-remote`, 07:47Z and 07:48Z), and it is in this head. No file of #8686 is edited. `form_complies` and the four statements keep their text; no lock record moved (the records run regenerated nothing).

THE RUNS AT 9cf54a37d7 (READ: each run's job.json names the commit; each log read by me to its last line):
1. r20261112-030840-0094, the records: outcome pass for security/Proofs, security, security/Properties (sent 04:43Z).
2. r20261112-044349-43ec (cpu-9), the eleven large suites by name: "11 suites: 11 passed (0 cached), 0 failed; 97.7 min of test time".
3. r20261112-044433-e691 (cpu-11), verity-vllm: "14726 passed, 500 skipped in 4015 s"; "1 suites: 1 passed (0 cached), 0 failed".
4. r20261112-060949-b264 (cpu-2), the other 57 of the 69 that `suites.py --changed origin/main` names: "57 suites: 57 passed (5 cached), 0 failed; 207.0 min of test time". The five cached (verity-datacenter, benchmarks/hash/dither, benchmarks/torch_gpu/j10r_sm90, verity/definitions/private_circuits, verity/protocols/tests) reuse passes that node held from 7581e0708a (TRN-133's branch), each "traced: every file its run read is as it was".
No run was started twice. Slot waits: 16 min (43ec), about 12 min (e691), 2.5 min (b264).

THE BODY (RAN: `research queue edit 8676 --body-file ... --by training`, dry run first; read back equal to what I sent): 64,888 characters. It has the merge paragraph naming main's 9725922ca6, the Lake manifests' spelling, the zero-width-leaf bullet, `keyedAsInput` under what is left, and a row for each run at this head.
One sentence I corrected against the last body: "no file of this diff holds more retired terms than the list allows" is false at this head. Two files are over, both NET-301's as at #8686's head: `NetworkCheck/Period.lean` 250 sites against 185 and `period_case.py` 166 against 129 (RAN: the lint's own counter at this head, at #8686's head and at main be9a3d566905). main holds them over already (247 and 165; the lint warns there on 636 files). The umbrella `Nci.lean` COUNTS named (16 against 14) is main's own bytes and entry, not in this diff. I changed no allowlist count: it warns, it does not fail (`repository` passed, 311 tests, in b264).

NOT CHECKED:
- Today's main. The head holds main at be9a3d566905; main is now bcaf58a45d53, 70 first-parent commits on. GitHub shows #8676 CONFLICTING. The files both sides changed are three: the two locks under security/ and the retired-terms allowlist, each with a registered merge driver; a dry merge with the drivers has no conflict (RAN: `git merge-tree`, tree 9065861aba). Nothing built or tested on it. main changed 35 Lean files since; of what these proofs read, `Circuits/Program.lean` gained `readJson`, `decodeOf` and `decode_eq_of` after `decode`, which is unchanged (READ: the diff). Whether `DecodeKeys` still builds over it: not run. Per your order I did not merge main alone.
- The proofs package's kernel replay (the records run is the fast path).
- The six packages' check (0faa) ran at 7bc9188630, the head less the mend; the mend is in the proofs package alone.

ONE THING THAT HAPPENED: my VM was replaced at about 07:46Z (uptime 1 min at 07:47Z); the worktree and /tmp were gone. Nothing was lost: the head was pushed, and the body builder and rows were in note 20261112T0613Z, from which I rebuilt the body. b264 ended at 06:58Z, 48 min before my timer (set for three hours of suites; it took 49 min), so the push came at 07:48Z instead of about 07:00Z.

Notes: 20261112T0613Z (the builder), 20261112T0642Z, and the final one synced now. I am done unless you send more.

