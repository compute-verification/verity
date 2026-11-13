[2026-11-13T14:39Z] bb733fa86b15
From proofs (coordinator), Fri 13 Nov 14:39Z. Your 14:25Z second line is read; the cause is on PRF-615 (14:26Z) and PRF-490. The fresh run `r20261113-142134-4eb4` stands as launched: your reading of my bound is the right one.
Soundness's word (PRF-490, stamped 6:35 AM PST): "if the fresh session at `main` is accepted at 7:20 AM, `RecursiveAuditC3`'s citation hold lifts, and the stale fixture is remade or deleted, not kept as passing for the wrong reason."
So the mend, as commits on a branch, no pull request before "carry":
1. The kept data is REMADE from the fresh run's `out/data` (an `art:` id in `fixtures/artifacts.json`, by `research data put --preserve`), if the session is accepted part for part. If it cannot be remade today, the two tests and the stale `recursive_data.json` are deleted, and PRF-615 stays open for the remake.
2. `test_a_flipped_inner_coin_is_refused_where_flock_verify_refuses_it` fails for the right reason on the new data, or goes.
3. The tests go in the opt-in heavy tier that the nightly runs.
One line when the session's verdicts are in (about 15:20Z), with the count of parts accepted; the entry by 16:00Z.

