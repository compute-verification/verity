# To the proofs coordinator

## TRUSTV5, 16:40Z Sat: plan for V5's three Bridge premises

Branch `cursor/proofs-trustv5-eaeb` (from checked `4e724154a3`).

**All three can go. One can't be done the way the brief says:** `NotBernoulli` can't be proved from #7037's refusal
alone. #7037 refuses Bernoulli only at 0 < p < 1. The verifier still accepts p = 0 and p = 1, where `drawnLaw`
returns `.bernoulli` and `NotBernoulli` is false. So item 3 adds the refusal where the verifier draws.

1. **`RootFixesDomainL` becomes one ecr/sha-512 term** (PR A, Lean only, medium: about 250 new lines and 60 changed).
   First PR, marked by 23:00Z.
   - New `Bridge/LeafRoots.lean` holds #4412's finder at a leaf map, because V5 reads roots at `LeafMap.ofPublic`,
     where shared-row ports can have different leaf counts. It has `topInL`, `root_topL`, `topInL_ne`,
     `rootsOfL_collides`, `rootsCollL`, a coinless `rootsFinderL` costing `rootsCostL` SHA-512 calls, and
     `le_add_ecrBoundL`.
   - Two top frames of one kind are told apart by #4412's `frame_head_inj` and `step_schema`. Frames of different
     kinds (pad, leaf, node) differ at the tag byte.
   - V5 is restated with no `_hroots`, and its bound gains `+ Claims.ecrBound sha512Msg (rootsFinderL D.unroll Lm)
     (rootsCostL D.unroll Lm)`.
   - The current statement stays as a Proofs lemma (`Bridge.SampledKSLiveV5R`) with the current assembly, and
     `le_add_ecrBoundL` gives V5 from it. `SampledKSLiveV5Accepted` derives from that lemma and is unchanged.
   - Lock: V5's record is restated, and `RootFixesDomainL` leaves `open`.
2. **`DrawsUnder` becomes `domain`, backed by a lemma** (PR B, small). Also marked by 23:00Z.
   - #7569 is on main (09:14Z). It argued `DrawsUnder` is the verifier's own computation, but the lock still lists it
     `open`, so #7790's gap counts it.
   - It isn't a fact about the world: it defines V5's `law` and `kd` as what the verifier derives from the
     registration. The lock's category for that is `domain` (part of what the guarantee says), which #7790 doesn't
     count. Inline equations can't replace it: the inline rule refuses new ones.
   - Lean: `Bridge.drawsUnder_of_drawnFrom`. At every session the one-stage verifier draws (`OneStage.drawnFrom … =
     .ok`), `DrawsUnder` holds at the law it drew under, and the sample is `drawUnits` under that law. It follows
     from #7569's `drawnFrom_drawnLaw`, in about 30 lines.
   - Lock: `DrawsUnder` moves from `open` to `domain`, with the why "the law V5 is about, the one the verifier derives
     from the registration and draws under (met by `Bridge.drawsUnder_of_drawnFrom`)". The statement doesn't change.
   - **Flag:** this is a reclassification backed by a lemma, not a statement change. The alternative is restating V5
     over the verifier's `Registration.Verifier`, which is much bigger, and I don't recommend it. Say if you or Daniel
     want that instead.
3. **`NotBernoulli` is proved from a verifier refusal** (PR C, medium, Sun). It goes on top of B, since both touch
   `DrawnLaw.lean`.
   - PARTCHECK's #7335 refuses every Bernoulli law at the parameters and at register. It is open, was marked Nov 4 at
     `0a55478cba`, and never landed. Even landed, it doesn't refuse in the draw `drawnLaw` mirrors.
   - Plan: `OneStage.drawnFrom`, the draw the one-stage verifier's coin check runs on every live session, refuses a
     Bernoulli law, and `Bridge.drawnLaw` mirrors it (`drawnFrom_drawnLaw` re-proved).
   - Then `drawnLaw … = .ok law → NotBernoulli law` holds, `drawnLaw_cases` loses its Bernoulli case, and V5 drops
     `hb`.
   - Python: the difftest's `_coins_problem` mirrors the refusal. `SampledDraw.drawnFrom_ok` is adjusted.
   - Sizes: verifier about 6 lines, Bridge about 40 changed, Python about 4, and records for `sampled_proofs`,
     Security and Properties.
   - **Ask:** should #7335 land too? It closes the stream path and Python's service. I can restack it on checked if
     PARTCHECK is gone. PR C doesn't depend on it.

**Runs:** each PR gets a remote `verity lean try`, then `lean_changed.py --records Security/Proofs --update` (about 1 h),
then the quick suites.

**Gap:** V5 goes from 8 premises to 7 after A, 6 after B and 5 after C. The five left are TRUST6's. I expect no file
overlap with TRUST6 beyond the locks, which the merge driver handles.
