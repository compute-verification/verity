
[2026-11-09T05:53Z] c4ded86ffd56
TWOOPENS -> SLOTSAT: the branch is pushed. `cursor/proofs-sessions-bytemplate-3e7a` is at `1899d21f6c`.

What's on it since `eb6b0d6f57`:
- C and E (`3e6e86f096`): a slot's leaf comes from the roots, and `SlotsAt` is restated by the slot's fields. This is the shape I sent you at 04:12Z, now in `FrameIsAt.lean` and `FrameAt.lean`.
- HEADERROOTS merged, and `HeaderRootsAt` is gone: `interior_entry` calls `header_roots` directly.
- `SpotFlock/Depths.lean`: `frameIs_of (hf : h.frame = frameAt slot c) (hS : SlotsAt h slot c L) (h64 : 64 ≤ depth c.own) : FrameIs h L`. Your `SlotsAt` is the last fact `FrameIs` needs; `h64` waits on the coordinator's call.
- P1+P2 merged.
- `EntriesAt.ins_fields` is renamed `ins_reads_public`. It clashed with `Placement.ins_fields`, so the root didn't build at `eb6b0d6f57`.

Lean status:
- Try r20261109-044624-2bdf (cpu-13), at the head before the last commit, built 12398 of 12399 modules. Every SpotFlock module passed except `Depths`, whose three errors `1899d21f6c` fixes.
- Try r20261109-054903-897f (cpu-13) is checking the root with siblings at `1899d21f6c`. It is on a cold slot, ETA about 06:30Z.

TWOVERIFY's finding G, the coordinator's option (a), is on #8270 (`772c78ab4a`), stacked on #8119. Its `check_ok_values` gives `(s.template, r) ∈ own.refReads`, with a registered value's reads held to the reference's. I merge it once it is admitted.

[2026-11-09T06:18Z] 96186fb78723
TWOOPENS -> SLOTSAT, 06:18Z. My head is pushed: `cursor/proofs-sessions-bytemplate-3e7a` at `1899d21f6ceec046a224aa16927345fa6ed331d2` (on origin since 05:52Z).

What it adds since `eb6b0d6f57`:
- A and F: round 2's `sameAt` holds a read to at most `MAX_LEAVES` = 2^64 leaves; `rowsOk` holds each read to its unit's first row, `pos = b*(leaves/units)`. `populationOk` also holds `instances` to at most `MAX_LEAVES`.
- C and E: `irootsOf` states the replay block's place and count; a slot computes its leaf from the roots; `SlotsAt` is restated by the slot's fields; `templateOk` requires v2 rows (`port.v2 && port.segs.isNone`).
- HEADERROOTS merged; `EntriesAt`'s `HeaderRootsAt` is gone (`interior_entry` calls `header_roots`).
- `SpotFlock/Depths.lean`: `rootedDepthsAt`, and `frameIs_of (hf : h.frame = frameAt slot c) (hS : SlotsAt h slot c L) (h64 : 64 <= depth c.own) : FrameIs h L`. Your `depth_revealCode` gives h64.
- P1+P2 merged. `EntriesAt.ins_fields` is renamed `ins_reads_public`: it clashed with `Placement.ins_fields` at the root.

Build: try r20261109-044624-2bdf (cpu-13) built every module but `Depths` (12398 of 12399), among them FrameAt, FrameIsAt, LayoutAt, EntriesAt, HeaderRoots and Placement. `1899d21f6c` fixes Depths' three errors; the root with siblings is try r20261109-054903-897f (cpu-13), ETA about 07:00Z.

Once your restated `slotsAt_slotMap` is on a branch, I merge it and assemble `FrameIs h (layoutFor h)` from `frameIs_of`, `slotsAt_slotMap` and `depth_revealCode`, at a frame with `h.frame = frameAt (slotMap h) (revealCode h (siteMap h) c0)`.

[2026-11-09T06:31Z] b0c6ea8bf2cd
TWOOPENS -> SLOTSAT, 06:31Z: a heads-up before you restate. `checked` now has #8229, which changes `FrameIs`'s opening.
- The opening is no longer `∃ o ∈ h.frame.reveal s u v wF, o.2.pos = g ∧ o.1 = roots ∧ …`. It is the program's: `let o := Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u v wF g` (`SpotFlock/Reveals.lean`).
- `Reveal.frameAt f … g r` is `f.reveal`'s FIRST opening with `pos = g` and roots `r`. So the position and the roots come for free (`rootsAt_eq`), but a frame must make its first such opening the right one.
- I'm merging #8230's head `f2fedfc8eb` (which carries #8229) and `checked` into my branch now. I'll restate my `SlotsAt` the same way: `let o := Reveal.openAt … (Reveal.frameAt c) … g`, then the same field-by-field conjuncts about `o`. The `∃ o ∈ c.reveal`, `o.2.pos = g` and `o.1 = …` conjuncts go.
- Your field-by-field work carries over. What changes is that `revealCode`'s reveal must give, as its first opening at `(g, rootsAt g)`, the opening your sites describe. I'll send the merged head, ETA about 07:30Z.
[2026-11-09T06:33Z] fd2a68bc405e
TWOOPENS -> SLOTSAT: my merged head `7206dc0ce4` is in a Lean try now (r20261109-063041-18dd, cpu-1). I'll push it when it's green, ETA about 07:30Z. It carries #8230's head and `checked` `204e59da60`. Two things in it you'll need:

- **`SlotsAt` states the code's opening, as `FrameIs` now does.** The opening part is `let o := Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt c) s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF g`, then `o.2.val = bitAt …`, the row, the salt and both branches as before. `∃ o ∈ c.reveal`, `o.2.pos = g` and the roots conjunct are gone: `Reveal.openAt_mem` and `Reveal.rootsAt_eq` give them. So `slotsAt_slotMap` has to show that `revealCode`'s FIRST opening with `pos = g` and `roots = rootsAt … g` carries the value, row, salt and path. `frameIs_frameAt` takes it as is, since `Reveal.frameAt (frameAt slot c)` is `Reveal.frameAt c` by `rfl`.
- **`checked`'s round 2 reads `Own.verified`** (`second-verified`, `second-tables`), which `ownOf` leaves empty. Left alone, round 2 refuses every statement and the model is vacuous. `codesOk` now runs `stmtCodes` at `{ own with verified := S.checked.toArray }`: the code checks each session itself, so it is the verifier of record. `stmtCodes_nil` gained two leading conjuncts. If you merge `checked` before my head, take both changes.

[2026-11-09T06:40Z] d0b25fc6bc99
TWOVERIFY (proofs), 06:40Z. #8270 (`second-values`, with finding G) is marked and admitted at `772c78ab4ae5ecfcff80fcede5f9f5f748dcf8a8` (branch `cursor/proofs-twoverify-7300`). Its suites and Lean try passed, and its land ask is with the coordinator. #8119 (`6a0dd4f0c475`) has already landed on `checked`.

The statement of `check_ok_values` is the one I sent at 05:01Z, unchanged:

```lean
theorem check_ok_values {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∀ i < ss.size, ∃ s r, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧ (s.template, r) ∈ own.refReads ∧
      jeq (blankReads own s.template s.registered) (blankReads own s.template r) = true
```

One trap when you merge `checked` after #8270 lands, the same one TWOOPENS hit with `Own.verified`:

- `Own` gains `refReads : Array (String × Json) := #[]`, each template's registered reference statement's `--registered` file (or `.null` when it reads none), as the auditor's params give it.
- A model `ownOf` that leaves it at that default makes `valuesOk` false for every statement, since a template with no reference reads passes no statement. Round 2 then refuses everything, and anything proved from a passing check holds vacuously.
- Set it the way you set `refs`: from the registered reference statements, which are fixed before the run.

If a per-port consequence of `check_ok_values` would help (for example, "a registered value read at port p is read at the reference's rows"), tell me which one. It needs `Flock.canon` injectivity, so it isn't free, but I can add it.

