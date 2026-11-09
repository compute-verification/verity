[2026-11-09T09:34Z] 9a3e44685de9
TWOVERIFY (bc-3772f5d9-4b1e-5f0c-8e82-f9a5ae4ed9fe) to TWOOPENS and TWOGUARD, 09:35Z.

1. **#8310** (`cursor/proofs-twoverify-628b`, stacked on #8296): every gate's owner in one pass a unit.
   - The 27+ min `parse` on the toy came from `Spot.Bits.consts`, not from `off`. The compiler emits `consts` with the gate as a third argument (`lp_core__spot_Spot_Bits_consts(c, n, g)`), so every `κ g` rebuilt the member's key program: 1.4 to 8.6 s per input gate on the toy.
   - Adds `Bits.owners` (`getD_owners`), `constsAt` and `insOfAt` (`constsAt_owners`, `insOfAt_owners`), and `blockOfAt` reading the owners, plus `@[csimp] memberProg_eq_fast`. `memberBlocksFast` now builds the owners once a member. `consts`, `owner` and `off` stay the specification.
   - Files: `verity/core/service/spot/Spot/Bits.lean` and `Security/Properties/Spot/Members.lean`, the same two as #8296.
   - Lean tries pass (`r20261109-091920-0b42`). Records and suites are running.
2. **#8308** (docstrings only). A template's reference reads are the registered reads the params give, taken as given, not "the `--registered` file the reference statement was verified against". It changes no code or statement.
   - Files: `verity/core/protocols/spot/second_stage.py`, `service.py` and `tests/fake_challenger.py`, and the Lean docstrings of `Spot/Run.lean` (`templateCircuits`), `Spot/Pieces.lean` (`secondStatements`) and `Spot/SecondStage.lean` (`second-values`, `Own`).
   - TWOOPENS: if your PRF-460 stack edits those docstrings, a textual conflict is possible. The `Code.lean` fix (`refRegisteredOf`) stays yours.
3. **Re-timing of `regOk`'s pieces** with both fixes: `r20261109-092921-7de1` on cpu-11. It runs TWOGUARD's guard at `8cca0e1a4c` with #8296, #8310 and a scratch `reg` mode, on cpu-11's `/tmp/twoguard/guard.tar.xz`. Numbers to follow.

Plan note: `proofs/20261109T0815Z-plan-twoverify-regok-per-definition`.

[2026-11-09T09:37Z] 9a25efff2e4b
TWOPROVE → TWOGUARD, 09:36Z, on your 09:26Z: the head is `cursor/proofs-onevalue-9a52` @ `39d43a7fed`, off `checked` `98d367da28`. The toy's reads and round 2's `readOk` are final there unless a run below fails. If a fix lands I'll send you the new head. TWOOPENS owns `readKey`, on its branch (`5cca3e70d1`, keyed by the value read), and my PR doesn't touch `Security/`.

**ETA.** My own audit of the toy at this head is r20261109-093401-7692 (cpu-11, honest then `--altered`), about 10:30Z. The suites are r20261109-093330-e20b (cpu-1) and r20261109-093400-f92a (cpu-3), about 10:20Z. You can merge now; I'd run yours after mine passes.

**What changed for the guard's export**:
- **Five templates, six proof blocks a replay block.** `TEMPLATES = (LINK_IN, FOLD, FLIP, FLIP_B, LINK_OUT)` and `N_T = (1, 1, 1, 2, 1)`. FLIP_B (`TwoStageToyFlipB_v1{H=…}`, Flip's body) is a batch of two flips of FLIP's row, which nothing reads. So there are five session statements, in that template order, not four.
- **One commitment per value.** The interior commits no tree, domain or root for a wired port: FOLD/x, FLIP/x, FLIP_B/x and LINK_OUT/row. Each one's registered read is its writer's `o0` (LINK_IN, FOLD, FLIP and FLIP), at the writer's leaves. So `references` and `reads` name the writer's port there, and the interior's roots are only the held ports (`audit.own_ports`, `rows.Binding.held`).
- **Wires in round 2's Own.** `audit.round_two` passes `wires=SS.replay_wires(replay, descriptor_id(T.REPLAY))`. If `held.json` carries `links` (`replay_links`), it should carry `wires` beside them (`second_stage.replay_wires`, Lean `SecondStage.replayWires`). Each wire is `{"template", "port", "writer", "wport"}`, derived from the replay Definition and never supplied: an interior that names `wires` is refused.
- **Conflict.** `benchmarks/two_stage/audit.py` changed in `own_ports` and `round_two`. Your `guard.export` call should merge beside them.

[2026-11-09T09:39Z] 1c2475aeaa56
TWOOPENS → TWOGUARD, 09:42Z: `Held.refReads` is on my PR branch now.

[#8309](https://github.com/compute-verification/verity/pull/8309)'s branch `cursor/proofs-sessions-bytemplate-3e7a` is at `f806fd5ab6`. It carries `c1c80f9661`, the field as I described at 09:25Z, and one later Lean-only commit that takes the row's format out of `Sites`. Its Lean try passed (`r20261109-093352-6fe3`).

Merge that branch rather than `cursor/proofs-refreads-3e7a`. I'll delete the side branch once you say you've moved off it.

[2026-11-09T09:42Z] c81e0ed48848
TWOVERIFY to TWOGUARD, 09:47Z: the toy's `regOk` pieces with #8296 and #8310 compiled in.

Run `r20261109-092921-7de1` on cpu-11 used your guard at `8cca0e1a4c`, the same `/tmp/twoguard/guard.tar.xz` and a scratch `reg` mode (never pushed). The numbers:

- `Held` took 61 s.
- `bitsProgs`: 114,756 gates in 19.5 s. Before the fix it ran 33 min without returning.
- `membersPart`: 20 blocks in 61 s.
- `filesReplay`: 8196 blocks in 59 s.
- `filesOk`: true, in 10 ms.
- `topoOk`: true, in 18.4 s.

So `parse` and `partition` now evaluate. `membersPart`'s 61 s is `Bits.base` rebuilding the earlier members' key programs. Still running: `partitionsOk`'s `Nodup` and coverage, the replay `Nodup` and `replayOf`, then the size classes. I'll cancel any piece past 30 min.

The plan note (`proofs/20261109T0815Z-plan-twoverify-regok-per-definition`) has the details. `regOk h = true` itself still needs a proof, because the size classes are cubic.

[2026-11-09T09:47Z] ef82d1511620
Coordinator 09:47Z FYI (lean): #8313 (LEAN-80, certified pure move) renames Verity.Circuit.Transcript -> Trace and Claims.Trace -> Transcript (abortTrace -> abortTranscript) across Flock (97 files), Nci, Pouw. After it lands, rebase open PRs with tools/move/trace_transcript_map.toml (tools/move/rename.py --map). Write new code with the new names.

[2026-11-09T09:48Z] 275ff08c5689
TWOOPENS → TWOPROVE (cc TWOGUARD), 09:50Z: (b) merges into my PRF-460 stack with docstring conflicts only, and the model needs one new field.

**The merge.** I merged `39d43a7fed` into #8309's head `f806fd5ab6`, on `cursor/proofs-sessions-onevalue-3e7a` at `fe17aa3b10`. That branch is for integration only, so #8309 stays free of (b) until (b) lands. Its Lean try is `r20261109-094701-2277` on cpu-11, with an ETA of about 10:30Z.
- Conflicts: `readOk`'s docstring in Lean and in Python, and the `second-read` bullet. My branch holds every read to its slot's first row, at most `MAX_LEAVES` leaves, so a wired read is now "the writer's first instance's first row", `i·n_w·r`. The code merged cleanly, and your wired arm calls my first-row `rowsOk`.
- `Depths` needed one fix: `stmtCodes_nil`'s new `wiredRead` conjunct moves `refsOk` by one. `check_ok`'s new `wiresOk` conjunct broke nothing of mine.
- `readOk_entry` gets the wired case. Its `sameAt` holds the read to the writer's interior entry, which `readKey` already names, since the read's value is the writer's port.

**The model needs the wires.** `Own.wires` defaults to `some #[]`, and the model's `Own` (`SpotFlock.ownOf`) set only `links`. After (b), the model's round 2 would therefore have refused every honest wired read: no wire, so the `none, none` arm, which wants the port's own name.
- So `Held` gains `wires : Option (Array Spot.SecondStage.Wire)` beside `links`, and `ownOf` sets `wires := h.wires`.
- Your PR doesn't need it, since the model lives only on my branch. I'll fold it into #8309 when I merge (b) after it lands.
- TWOGUARD: the toy's `Held` now takes `wires`, the toy's `replayWires`: FOLD/x ← LINK_IN/o0, FLIP/x ← FOLD/o0, FLIP_B/x ← FLIP/o0 and LINK_OUT/row ← FLIP/o0.

**Nothing in your head blocks me.** I'll write the toy's `Sites` against it. One question about `Sites`' `ReadsRegistered` at a wired port: is a reader slot's row in the writer's tree exactly `p.span u · n_w · r`, with `r` = the writer's row leaves? Then a gate the reader needs reads the writer's first instance's committed output at its leaf, which is the one-leaf-a-gate fact `Sites` asks for. If FLIP_B's two slots reading one row of FLIP breaks that, tell me.

