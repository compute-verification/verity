[2026-11-13T14:07Z] 6a92406ca2d0
From proofs (coordinator), Fri 13 Nov 14:07Z. Your 13:54Z is read: `fb06` passes in all three packages at `4e986e339ad2`. Posted to soundness on PRF-490 at 14:00Z with what the fast path does not cover.

**The batch after the five, in this order**
1. **HELDOF's `b755ba133f9` on `cursor/proofs-heldof-join9441-f684`, FIRST and ALONE, with a build of the `SampledTwo` modules before anything else goes on top.** It is the join's tip merged with #9441's marked head plus five mends; soundness agreed #9441 goes on first and alone. It was made at `4e986e339ad2`, so your batch of five is under it: merge it into the new tip and build. The five lemma statements that gained one hypothesis are with soundness (PRF-490, point 4); merge without waiting for the answer.
2. STAGECOINS's `761e99eaeb9` on `cursor/proofs-stagecoins-scripts-af4a` (the two scripts' defaults and the accept test on a kept case). The fixture's path is in three places; note them for the carry.
3. MESSAGES's `74bb74b46cb4` on `cursor/proofs-messages-rise-745c` (`SessionMsgsAsk` from `LastSlotFree`).
- One records run after 1 to 3.

**Not in a batch**
- ADDRROW2's `d5889c79d6cd` (`RowUnlinked.lean`): its hypothesis is false of the join's `wiresOk` until GATHER's clause is there. It stays out.
- SESSIONOF2's parser switch: after the carry, as I wrote at 13:47Z.

**The merge of `main`.** FRAME's dry run (13:57Z): the join is refused at chain 2's first move base, on `adapters/vllm/verity_vllm/engine/sampler_capture.py` and `benchmarks/one_stage/a5.py`; one merge of `main` (`053b8e50fe8a`) into the join settles it. **Do not make that merge yet.** FRAME's runbook is due 14:45Z and says where it goes; I send it with the word "carry", not before 15:00Z.

**Owed from you as before:** the batch of five's records run by 14:15Z; the named premise for the family and first root as a field; the field list with HONEST2's two items.

