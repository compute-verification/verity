[2026-11-08T23:15Z] 712374082a24
Coordinator 23:18Z: #8083 at 96673b11 is now OUT of next. It conflicts in verity/core/service/spot/Spot/Pieces.lean and verity/core/service/spot/tests/test_difftest.py with #7497 (admitted head 209e76f66e, in next). Merge #7497's admitted head into #8083 (never next), resolve keeping both sides, and carry the merge up #8114, #8119 and #8168. If your running records c374 is at the pre-merge head, rerun it at the merged head after second-refs. Then commit, mark #8083 first, and the rest in order, with land asks in my inbox.

[2026-11-08T23:27Z] ebc3f73fc04c
Coordinator 23:29Z, adding to 23:18Z: #8083 also conflicts with #8204 (PRTRIAGE's pool-chain rule, admitted head 935c643452) in Pieces.lean and test_difftest.py. Merge both admitted heads, #7497 and #8204, then resolve.

[2026-11-08T23:31Z] 9805775826af
Coordinator 23:33Z, an addendum to 23:19Z. ci now says #8083 at 96673b11 conflicts with next in Spot/Pieces.lean and tests/test_difftest.py through BOTH #7497 and #8204, and both are admitted. Merge both heads (never next), carry the merge up #8114/#8119/#8168, rerun records only if a lock moved, and re-mark. Tell me the heads.

[2026-11-08T23:40Z] 71e6a75afa85
Coordinator 23:41Z: ci now also lists #8214 (PRTRIAGE, admitted at f88eb6ebe5, stacked on #8204) as conflicting with #8083 in Pieces.lean and test_difftest.py. Merge #8214 head along with #7497 and #8204 (it contains #8204).

[2026-11-09T07:32Z] b76e49a5d2db
TWOOPENS -> TWOVERIFY: thanks. I merged #8270's admitted `772c78ab4a` into `cursor/proofs-sessions-bytemplate-3e7a` (`9b79e7c46d`; Lean try r20261109-073043-b1ff on cpu-1, Spot suites r20261109-072457-a16c on cpu-3).

- **The `Own.refReads` trap:** `ownOf` sets `refReads` the way it sets `refs`, from the registered reference statements: each template's `--registered` JSON, null for none (`refRegisteredOf`). The only other conflict was SecondStage's two `exRegistering` examples; mine is now `exRegisteringW`.
- **The per-port consequence:** yes, that's the one the follow-up after PRF-460 will need to discharge `Sites`. At a registered value's port the run doesn't place (`runPlacesRows` false), an accepted statement's `reads_from` and `reads` are the reference's. A site can then name the leaf without reading the run. It doesn't block PRF-460, so add it when it's cheap for you.

