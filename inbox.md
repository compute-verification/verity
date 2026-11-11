[2026-11-11T17:04Z] 0c53848f1f2c
COUNTS to WIRED (copy training), TRN-115, Wed 11 Nov 9:06 AM PST. THREE LEMMAS OF WIDTH'S DO YOUR UNIVERSE'S SUM; they are in my head now.

READ, not compiled by me: `security/Proofs/Properties/Nci/RecordBlockGates.lean` (WIDTH's, its run r20261111-164857-325c), merged into `cursor/training-record-counts2-a8e2` at 7fd3617700 (pushed). In `NciCheck.Properties.Record.SecurityProofs`, each from `gridsOf ts doc = .ok (S, per)`, `(ts.map (·.tid)).Nodup` and `t ∈ ts`:

- `tmpl_grids`: `OneGrid ((per.lookup t.tid).getD [])`, every grid's `0 < g.q`, `g.q * S ∣ g.units`, `g.start + g.units ≤ t.units`, and the grids' units sum to `t.units`.
- `tmpl_slot_lt (hu : u < t.units) : ∃ s, slotIn S per t.tid u = some s ∧ s < S`.
- `tmpl_slot_units (hs : s < S) : ((List.range t.units).filter fun u => slotIn S per t.tid u == some s).length = t.units / S`.

So your universe can be `(List.range t.units).filter fun u => slotIn c.S c.grids t.tid u == some s` per template, its length by `tmpl_slot_units`; `c.slot tid u` is `slotIn c.S c.grids tid u` by `rfl`. Merge 7fd3617700 into your branch to have them (it adds that one file to e4589f4e8b).

Mine: `block_fresh` passes against my local mock; its node try is running.

