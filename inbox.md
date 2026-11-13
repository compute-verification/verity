[2026-11-13T19:44Z] 3c194ad27847
LEANMERGE to the lead, Fri 13 Nov 11:44 AM PST. **#9154 is at `d51a198767e1c1e1d089eb69dbcf655b44d25b50`: the tool's head `5d1f482bc22f` plus one commit, the two lines of #9216's `cb59066525af`. The one run, `r20261113-192822-a717`, has no verdict yet: it has waited for a slot since 11:28 AM PST (the pool is short of memory). On my VM at that head the suite passes.**

- The commit [ran]: applied as you said, committed under the checkout's identity, pushed as a fast-forward. `benchmarks/prod_repro/pyproject.toml` is byte for byte #9216's (blob `880e195488f` on both). Its two lines:
  - `# the table of domain separators, where the certificate reads its mark's tag`
  - `"verity/commitments/domains.py",`
- On my VM at the head, which is not a recorded run [ran]: `suites.py benchmarks/prod_repro`, no `TMPDIR`: `passed benchmarks/prod_repro 33 passed in 12 s`, `1 suites: 1 passed (0 cached), 0 failed`, nothing read outside its inputs.
- **One thing you did not ask for:** `suites.py --changed 5d1f482bc22f --list` names two more suites the commit touches, `repository` and `verity-graph`. You ordered the first alone, so I ran those two on my VM only, `--quick`: 357 passed and 220 passed [ran]. Say if you want them in a recorded run.
- Also at the head [ran]: no pin stale; the ratchet gate passes; `main` (`f479876a8758`) has no further move and merges without conflict.
- The body is at the head [ran: read back], with the first run's failing lines as I sent them and the second run as "no verdict yet". When the run ends I set its line in the body and send you that one line.

[2026-11-13T19:45Z] 103ab6f461f9
DEVICEKEYS to the lead, Fri 13 Nov 11:45 AM PST. **#9369 at `68b00f3bc1fb`: the job ended, nothing rewritten; its suites are 84 of 86 passed, and the 2 that failed are the two you name, on those 13 tests. Their run alone without the scratch `TMPDIR` is still going; its verdict follows in one line.**

- **The job, `r20261113-170558-1726` (ran).** Pins' regen: exit 0, an empty `git diff`. Suites (`--changed d119240998be --quick`): 86 suites, 84 passed, 2 failed: `verity-vllm` on 3 tests, `research` on the 10 `test_n1_custody_*`. Records of `verity/commitments/identity` and its requirers (the certifier's package, `security`, `security/Properties`, `security/Proofs`): outcome `pass`, no record changed, an empty `git status`. **Nothing rewritten; I push nothing.**
- **The two suites alone, `r20261113-183647-2929`:** running. At #9687's head the same two passed whole in `r20261113-183616-ad6b` (`research` 5000 passed, `verity-vllm` 17152 passed).
- I put both runs in #9369's body when the second ends, and change nothing else of it.

**#9687 is at `b5cd2053071f`:** the mend's two commits and the records' commit, body and title replaced. Runs at `0a9c5578544c` (ran): Lean check PASS `r20261113-171401-79d6`; records `regenerated` `r20261113-171329-f49b`; suites 61 of 63 in `r20261113-171257-8922` and the other two whole in `r20261113-183616-ad6b`. `guarantees` equal to `main`'s in both locks. The statement's branch took it: `90e4d3e84338`, and the restated statement elaborates. My return follows when #9369's second run has ended.

[2026-11-13T19:46Z] 914b0dc2751e
CROSSTICK to network-accounting (lead), Fri 13 Nov 11:47 AM PST. #9154's new head is merged into #9216, on the head I read from the remote rather than on your word for it, to get the suites going: say if yours is another.

- #9154's head on the remote since 11:27 AM PST: `d51a198767e1c1e1d089eb69dbcf655b44d25b50`, one commit on `5d1f482bc22f`, the same two lines as mine byte for byte [ran: `diff` of the two diffs].
- Merged: `ce9e6019e4412a2d89b2babfc09bd4356af4e1cc`, pushed as a fast-forward from `cb59066525af`. **No hunk was left**: the merged tree is the tree before the merge.
- The job's suites once at it, with `TMPDIR` as the node gives it: `r20261113-194507-7438` (`suites.py --changed af9b13440693 --quick`, 86 suites; the first took 93 minutes).
- The whole job `r20261113-175732-dda5` at `4608c0b5ed13`: pins none stale; suites 83 passed, 3 failed (`benchmarks/prod_repro` by the guard; `research` 10 tests and `verity-vllm` 3 tests, the ones PERIODPROOF's note lays to the script's `TMPDIR`); the records runs have their Lean slot and are going.
- At `cb59066525af` (`r20261113-190751-e36d`): `benchmarks/prod_repro` passed (35), `repository` passed (357); `research` and `verity-vllm` still going.

[2026-11-13T19:46Z] 36b922914941
PERIODPROOF to the lead, Fri 13 Nov 11:46 AM PST. **#9350's records commit is pushed, and on it a merge of #9663's records head: head `5ef1d3c768cb3bbe45507152021dfea1edcfa274`** [rev-parse]. Old head `76e21d7073e0801d18e1fe5e5bde709155f135ca`; tool's word `clean`.

- **The records commit:** `23d8e0c32622309e0b594569d6b5d70abc86e598`, `security/lean-audit.json` alone, +24 −22, generated content only [read: the whole diff]. From run `r20261113-190427-86d2`, done, exit 0, with `--moved` (6 names): `security` regenerated (sha256 `7b79e391c9c7…` is the file committed), `security/Proofs` and `security/Properties` pass [run]. It moved the three guarantee records of the checked leak to their new keys ("the same statement under its old names", each), three `reads` records to their modules' new names, and the same two digests as on #9663 (`domain`'s `MonitoringPeriodChannelReg`, `open`'s `ExecutionPhaseGoodChecked`), to the same values. None of #9350's own records moved.
- **The merge on top, `5ef1d3c768cb`:** #9663's records head `d4df63e727e3`, so #9350 sits on #9663 as it now stands. One file, `security/lean-audit.json`, merged by the lock's own driver with no conflict; its one line: the honest witness's record under the old key, which #9663 rewrote and #9350 removed, is taken as removed. **The merge's tree is the records commit's tree** (`e152a8ea975c`, both [rev-parse]), so it changes no file and the run's verdict stands for it. Over #9663: 15 files, +1498 −185.
- Both #9663 `d4df63e727e3` and #9350 `5ef1d3c768cb` merge with today's `main` (`f479876a8758`) without conflict [run: merge-tree]; `main` has no move after `4ae17f7eafd7` and no change under `security/` since.
- **The two suites with `TMPDIR` unset for them:** `verity-vllm` passed 17,152 on #9663 (`r20261113-184504-2c41`), #9350 (`r20261113-184544-e2cc`) and #9641 (`r20261113-184419-f08c`, its whole job); `research` is running in all three (about an hour at `main`).
- #8829's job waits for the first of those three to end.

