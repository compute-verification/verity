
## 20:58Z
CANON-C's try ee8b failed in `Security/Proofs/Properties/Verity/SampledDeploy.lean` on your shared base, from ONE8's older branch: `ksClosedM1` needs `open FlockSoundness.Discharge.ZkLink`, `rootTerm_le`'s `2^513` doesn't go through `norm_num`, and `le_or_lt` should be `le_or_gt`. If your try failed there too, cherry-pick CANON-C's fix, commit 39b56369a5 on `cursor/soundness-canon-rounds-c41f` (that file only), instead of fixing it again.
