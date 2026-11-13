[2026-11-13T19:44Z] 3c194ad27847
LEANMERGE to the lead, Fri 13 Nov 11:44 AM PST. **#9154 is at `d51a198767e1c1e1d089eb69dbcf655b44d25b50`: the tool's head `5d1f482bc22f` plus one commit, the two lines of #9216's `cb59066525af`. The one run, `r20261113-192822-a717`, has no verdict yet: it has waited for a slot since 11:28 AM PST (the pool is short of memory). On my VM at that head the suite passes.**

- The commit [ran]: applied as you said, committed under the checkout's identity, pushed as a fast-forward. `benchmarks/prod_repro/pyproject.toml` is byte for byte #9216's (blob `880e195488f` on both). Its two lines:
  - `# the table of domain separators, where the certificate reads its mark's tag`
  - `"verity/commitments/domains.py",`
- On my VM at the head, which is not a recorded run [ran]: `suites.py benchmarks/prod_repro`, no `TMPDIR`: `passed benchmarks/prod_repro 33 passed in 12 s`, `1 suites: 1 passed (0 cached), 0 failed`, nothing read outside its inputs.
- **One thing you did not ask for:** `suites.py --changed 5d1f482bc22f --list` names two more suites the commit touches, `repository` and `verity-graph`. You ordered the first alone, so I ran those two on my VM only, `--quick`: 357 passed and 220 passed [ran]. Say if you want them in a recorded run.
- Also at the head [ran]: no pin stale; the ratchet gate passes; `main` (`f479876a8758`) has no further move and merges without conflict.
- The body is at the head [ran: read back], with the first run's failing lines as I sent them and the second run as "no verdict yet". When the run ends I set its line in the body and send you that one line.

