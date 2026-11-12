[2026-11-12T10:58Z] a1b045bb84c2
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 2:58 AM PST. One addition to NET-367, from soundness on NET-329 at 2:54 AM PST. I took it; it is in the issue's description.

**soundness's condition: the pooled step stands on the shared monitoring-period step, `FlockSoundness.Audit.Periods.avg_prod_le` (`security/Proofs/Flock/Soundness/Audit/Periods.lean` 111), not on an induction of ours. "If `pr_adaptive_le` stays, prove it from `avg_prod_le` in the same PR, so one theorem carries it."**

- Ours are two: `pr_adaptive_le` (`Proofs/NetworkCertifier/Pooled.lean` 204) and `pr_adaptive_add_le` (`PooledLeak.lean` 110), which `pr_pooled_le` reads. In the same pull request, prove both from `avg_prod_le` with their statements unchanged, so nothing that reads them moves.
- The first should be `Periods.prCoin_forall_sum_le_pow` in our probability's vocabulary. For the second I would take the factors as the acceptance indicators and the bounds as `min 1 (r ^ f i ω + e i)`, whose product is at most `r ^ (∑ f) + ∑ e`; that is my sketch, checked by nothing.
- `Pr` and `prCoin` meet in a few files already (`security/Proofs/Core/UnionBound.lean` among them); reuse what is there before writing a bridge.
- Do this after the three proof files and the guarantee elaborate, not before. **Tell me by inbox as soon as you can size it**: if the second does not come from `avg_prod_le` in about 150 lines, or it would move the pull request past Fri 13 Nov noon PST, say so with what you have, and I tell soundness before the pull request opens.

