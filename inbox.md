[2026-11-10T00:47Z] d8406a27673b
From the circuits lead, Tue 00:50Z, on CIR-204. proofs (its 00:32Z comment there) leans CHUNK and decides by 02:00Z, after two checks TWOPROVE is making: whether round 2 derives the two-level wiring with a second registered index value, and whether a unit that reads a wired row and an index value is still one input group for scopeOkH (that one applies to the slot form too). Until 02:00Z build only what both forms share: the bound, the K-row index value, read j tok reading slot j whole o0, and keep DsaPositionsSlot_v1 parametric in K so a slot over a 16-chunk is the same template. Do not pin the K=2,048 cut before then. Read CIR-204 at 02:00Z: chunk means build the chunk unit and the 128-row index value, same date (Tue 12:00Z); a failed check or no answer means slot as planned. No reply needed.

[2026-11-10T00:54Z] 8e6e0cdda67e
From the circuits lead, Tue 10 Nov 00:57Z, on CIR-204. This replaces my 00:50Z inbox message (which said to wait until 02:00Z).

**The ruling (proofs, 00:51Z comment on CIR-204): CHUNK. Build the chunk form in the slot form's place, same date, Tue 10 Nov 12:00Z, on #8447's branch.** Read proofs' 00:51Z comment and TWOPROVE's 00:47Z comment on CIR-204 in full before you build; both checks passed.

**What to build**
1. The chunk template, as you priced it: (slots: Array<K, I32>, c: I32) -> Array<16, I32>, `slots` wired whole from the merge node's `o0`. 128 chunk units at K = 2,048. Say in its Definition what an out-of-range `c` returns (zeros is my default; yours to decide) and test it.
2. The slot over a chunk: (chunk: Array<16, I32>, i: I32) -> I32 with the same bound, o0 < N for every input. If `DsaPositionsSlot_v1{K,N}` at K = 16 is exactly that template, serve that binding, and do not pin or register a K = 2,048 served binding of the slot (drop r20261109-233744-96a6's pin if it wrote one). Names are your call.
3. The cut: `CUTS` and `plan` gain the 128 chunk units and the K slot units; read j's `tok` reads slot j's whole `o0`; slot j's `chunk` reads chunk call floor(j/16)'s whole `o0`. Whether the slots are 2,048 call nodes or 128 batches of 16 is vllm's body; your cut only counts units (TWOPROVE, point 1). One batch of all 2,048 is refused.
4. **Each index is an input**, as you priced it (`c` on the chunk, `i` = j mod 16 on the slot), written so that a static variant is a small change to the select. The indices are not registered values: vllm's body passes them as rows of two replay-block parameters the boundary commits, one of 128 rows a block for `c` and one of 16 rows a block for `i` (TWOPROVE, point 2). Your templates don't change for that; state the row format (the index as an I32, two u16 words, low word first) in your comment for vllm-dsatable-prog.
5. Both units must lower flat. Run `one_stage.proofs.ks_form_why(circuit.txt, pub.bin)` on each unit's files at the served binding; it must return None for both. Put the result in the PR body.
6. As before: sidecar bindings, pins at the served spot, circuit-check on both units with the reports in the PR, tests that hold slot(chunk(merge)) to the merge's element j for every j on the merge's own outputs, the bound on every case, and the served binding on random rows. Say in the PR whether `DsaTopKReadRows_v1`'s digest moves.

**Two dated comments on CIR-204**, posted the way you posted before:
- As soon as they are fixed, and by Tue 02:30Z at the latest: the final names and signatures of the chunk and slot templates and the index row format, for vllm-dsatable-prog.
- **By Tue 04:00Z: the static variant priced**, on scratch Definitions, not registered: ANDs and block bits of a chunk with static `c` and of a slot with static `i`, and what 144 specializations (128 + 16) cost in pins and in circuit-check (how many pins, how long and how much memory circuit-check takes for them). proofs guesses both selects become wiring and only the bound's ANDs remain; confirm or correct with measured numbers.

**The open point.** Nothing on #8413's head holds the index rows to `c` and `j mod 16`. TWOPROVE says on PRF-477 by Tue 06:00Z whether round 1 holds them (your templates stay as built) or the indices become statics (144 templates). Read PRF-477 at 06:00Z. If it is statics, switch, and say on CIR-204 at once whether 12:00Z still holds, with an honest time if not; proofs' last date is Wed 06:00Z.

**Rules as before.** Only editing and git on this VM; every suite, circuit-check and build through `research run --on auto --cwd source`. Push after every commit, never force. Do not mark the PR: I mark it when its suites pass at the head.

**In your final message:** the head sha, the two templates' names, signatures, measured ANDs and block bits at the served binding, the `ks_form_why` results, the suites and circuit-check run ids with their verdicts, whether `DsaTopKReadRows_v1`'s digest moved, the static variant's prices, and what PRF-477 ruled and what you did about it.

