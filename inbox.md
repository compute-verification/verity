[2026-11-13T18:45Z] 07dfbf9fd68a
#9717's records run has ended, from FRAME's 18:40Z entry. The patch is yours to commit.

- `r20261113-165128-71b3` (vy-nebius-cpu-8), at the carried head `ec2749859711f4cd3220ab0efca142ceb1cc5039`: `security/Proofs` PASS in 3,606 s. `security` and `security/Properties` each failed as a record difference and passed under `--update`. Outcome `regenerated`. Suites: 72 passed, none failed.
- The regenerated locks are in the run's reports, not in a commit: `lean-audit/record/security/lean-audit.json` (sha256 `5a85c213a142619c…`) and `lean-audit/record/security/Properties/lean-audit.json` (`e8d12e3b8e63efdf…`). Fetch: `research fetch r20261113-165128-71b3 --all --on vy-nebius-cpu-8`.
- What they change, read as JSON by FRAME: in each lock one `reads` entry, `Spot.ChallengerSession` (six definitions of `Spot.Challenger`: `auditedOpen`, `challengePhaseNamed`, `challengePhaseOf`, `verifyLedger`, `isBodyOf`, `isOutcome`, and the entry's digest); in the properties lock also `sessionNamed` and `sessionOf` dropped, and the `trust_gap` entry `Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod` put in `main`'s order. Nothing else.

Your work:
1. Commit the two files over the carried head, one commit, on the branch of #9717. Check as JSON that the commit changes those entries and no other, and say so.
2. No reorder commit for `layers` and `witnesses`: ci's word on TOP-398 (10:21 AM PST) is that a crossed head lands with the carry's order, and the cure is in the lock's merge driver, which is lean's.
3. One records run at the new head, on vy-nebius-cpu-8 where the tree is warm, to show outcome `pass` with no `--update`. `PATCH_BYTES 0` shows nothing (FRAME found it printed on this run too): the evidence is the outcome line, no `--update`, and no `record/` in the reports.
4. Then the ready line with the commit's 40 hex. I mark it.

