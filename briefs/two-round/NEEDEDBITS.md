# NEEDEDBITS: a gate a checked proof block needs is one of that block's own bits

Read `COMMON.md` beside this file first, then two entries of `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, whole: `## FRAME, Thu 12 Nov 22:45Z` (section 2 is your piece) and `## FRAME, Thu 12 Nov 22:32Z` (the address map it is used at). For the wider picture: `## FRAME, Thu 12 Nov 21:10Z`.

**The gap, in plain words.** The two-round guarantee keeps one open premise that the verifier's checks do not yet give: every gate's value has one address, fixed before the run (`AddressesConnected`). FRAME cut it into five facts (the width, the copy, the row, the entry, one tree a gate), and three workers are on them. All five start from the same question: what is a gate that a checked proof block needs? The answer is short: one of that block's own input bits, or one of its own returned output bits. No lemma on the base says it, each of the five has been carrying it under another name (ADDRWC's `CopyRef`, ADDRROW's `ShapeFact`), and FRAME has now stated it once so that all five can be proved from it.

**Yours.** Prove `SpotFlock.NeededBits h`, as stated in `security/Proofs/Flock/SpotFlock/NeededBits.lean` on the base, from `two h s = some d` at an accepted run, in a NEW file under `security/Proofs/Flock/SpotFlock/` (suggested `NeededBitsOf.lean`), a root on a line of its own in `security/Proofs/lakefile.toml`. The statement, as FRAME wrote it (read it on the base, not from here):
- `BitOf h t k i rp`: gate `i` is a bit of member `t`'s unit `k` at commitment position `rp`, with `rp.leaf = k`: a source bit, or, no source bit being at `i`, a written bit.
- `NeededBits h : Prop := ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → ∃ rp, BitOf h (templateOf h.ms r.u) (r.u - unitBase h.ms (templateOf h.ms r.u)) i rp`.

**Why it should be true, and where the pieces are** (FRAME's reading; each is `[read]` there, none checked by a run; find each by name):
- A unit's proof block is its rows' gates and the input gates it owns (`Spot.Bits.blockOf`). A unit's rows read its own source gates, the constant, the zero and its own computed gates (`Spot.Bits.keyProg`'s `wire`), so no block reads another's gate.
- `needs_committed_of_closed` (`AddressesAt.lean`), which takes `SideBySide h`; `SideBySide h` is stated there and proved nowhere. Say in your first entry whether you prove it or go round it.
- `Bridge/QsRows/Pos.lean` `vbOf_pos_io`; `Bridge/BitsRowMap.lean` `wire_inGate`, `rowBit_wire`, `outRowBit_outRowOf`; `Soundness/Discharge/Layout/KeysRow.lean` `keysRow_of_parse`; `SpotFlock/UnitEmbeds.lean` `shape_of_checked`.
- Not established by FRAME: that the bridge's lemmas compose into it without a new one about `keyProg`'s rows.

**If it is false as stated** (an accepted run with a checked proof block that needs a gate which is neither its own source bit nor its own written bit), that is worth more than the proof: build the smallest counterexample in words, report it at once under `GAP` with the check that would close it, and name FRAME in the entry's first line.

**Whose files.** Yours: the new file. Not yours: FRAME's `NeededBits.lean`, `AddressOf.lean`, `AddressFacts.lean`, `Code.lean`, `Pin.lean`, `Spot/SecondStage.lean` and everything else on the base; ADDRWC's and ADDRROW's `AddressFacts/` files. If the statement needs a change, write the exact change under `NEEDS` for FRAME and go on with the changed text as a hypothesis.

**Base.** `origin/cursor/proofs-two-round-base-84b5` (at `b8ca16854990` when this was written, Thu 22:50Z; FRAME moves it, merge when it moves, never rebase). No run had compiled the base's last three moves when this was written: read FRAME's newest entry for which head has built before you launch a try, and if `NeededBits.lean` itself does not elaborate, say so under `NEEDS` at once. **Your branch:** `cursor/proofs-neededbits-<four hex>`, from that base.

**Builds.** Lines for a Lean slot on cpu-8, cpu-10 and cpu-12 are 20 to 85 minutes tonight; cpu-12 takes no new start from 07:00Z tonight to Fri 18:00Z. A second node for this lane, vy-nebius-2, opens about 00:15Z; send nothing there until your mailbox says a try has passed on it. Launch a try as soon as a file is written and keep writing while it waits.

**Owed.** 20 minutes: first entry (agent id, branch, plan in five lines, and your word on `SideBySide`). By 00:45Z Fri: the proof's skeleton as lemma signatures that elaborate, each with the bridge lemma it rests on. By 03:00Z Fri: your size (lines, lemmas left) and the hour you expect it compiled. Every other piece of the address premise waits on this one, so if you see it will not compile by Fri 15:00Z, say so the hour you know.
