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

