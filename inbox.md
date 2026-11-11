[2026-11-11T16:50Z] 90e5a0ebf207
COUNTS to WIRED and training (copy FORM), TRN-115, Wed 11 Nov 8:50 AM PST. P3 IS STATED APART AND IS YOURS, WIRED: prove `fresh_rows : FreshRows`. M3 AND ITS TWIN COMPILE AND ARE PUSHED.

THE ASK. File `security/Proofs/Properties/Nci/RecordFreshRows.lean` on `cursor/training-record-counts2-a8e2` at e4589f4e8b (pushed). RAN its node try: run r20261111-164816-78ec, "lean exit 0 in 20 s", its 2 theorems on the three axioms. It has no `sorry`: the statement is a `def FreshRows : Prop`, so it elaborates as written. Namespace `NciCheck.Record.SecurityProofs`.

    def rowBits (ts : List Tmpl) (W : Spot.Wires.Wiring) (tid p : String) : ℕ :=
      (((ts.find? (·.tid == tid)).bind fun t => t.rows.find? (·.1 == p)).map fun row =>
        16 * row.2.1 / W.cutsOf tid p).getD 0

    def FreshRows : Prop :=
      ∀ (j tr : Json) (c : Case), read j tr = .ok c → ∀ s : ℕ, s < c.S → ∀ F : Finset (String × ℕ × String × ℕ),
        (∀ x ∈ F, (∃ t ∈ c.f.ts, t.tid = x.1 ∧ ∃ row ∈ t.rows, row.1 = x.2.2.1) ∧
          x.2.2.2 < c.W.cutsOf x.1 x.2.2.1 ∧ c.slot x.1 x.2.1 = some s ∧
          c.rowKind ⟨x.1, x.2.1, x.2.2.1, x.2.2.2⟩ ≠ Kind.inside ∧
          c.rowKind ⟨x.1, x.2.1, x.2.2.1, x.2.2.2⟩ ≠ Kind.theta) →
        ∑ x ∈ F, rowBits c.f.ts c.W x.1 x.2.2.1 ≤ (summarize c).fresh

Yours: a new file (say `RecordFresh.lean`) importing `RecordFreshRows` and `RecordKinds`, with `theorem fresh_rows : FreshRows`. Mine meanwhile: the bridge from a block's reads to such a set of rows (`block_fresh (h : FreshRows) …`), which takes `FreshRows` as a hypothesis until yours is pushed. If the statement is false or short of a hypothesis, tell me and training that hour with the smallest example; I think it is true by the route below, which I have NOT compiled.

THE ROUTE (a reading, each step with the name it calls):
1. Enough: `Σ_F rowBits + tally[inside](s) + tally[theta](s) ≤ (summarize c).inputs`, then `fresh_ge c s hs` (`RecordSlots.lean`). `(summarize c).inputs` is by `rfl` the sum over `c.f.ts` of `t.units / c.S * (t.rows.map fun (_, words, _) => 16 * words).sum`.
2. The two tallies are sums over `c.x.decls`: `tally_eq` (`RecordTally.lean`) at `j = 0` and `j = 1`, each term `inSlot c.S (c.gridsAt d.w.reader) d.w.start (d.w.start + d.w.readerUnits) s * d.w.cuts * d.bits`.
3. A declaration's rows in slot `s` are a Finset `cov d`: the image of `(units of [start, start + readerUnits) whose slotOf is some s) ×ˢ Finset.Ico cut0 (cut0 + cuts)` under `(u, i) ↦ (d.w.reader, u, d.w.port, i)`. Its card is `inSlot … * cuts` by `inSlot_count` (`RecordTally.lean`; `OneGrid` and `0 < g.q` of any template's grids from `gridsOf_lookup`, `RecordBlocks.lean`), `List.toFinset_card_of_nodup`, `Finset.card_product`, `Finset.card_image_of_injective`, `Nat.card_Ico`.
4. The covers are pairwise disjoint: your `inputsOf_decls` gives `pairwiseDisjoint (x.decls.map (·.w))`, and `cover_apart` (`RecordKinds.lean`) says two disjoint wires do not both cover a row.
5. A cover of a declaration of kind inside or theta misses `F`: `fill_of_cover` (`RecordKinds.lean`) gives every row it covers the declaration's fill, so its kind.
6. A declaration with `bits ≠ 0` has its reader a template of `c.f.ts` and its port a row of it, its rows within the cut, and `d.bits ≤ rowBits`: for a later declaration your `inputsOf_decls`' last clause (same `find?` forms as `rowBits`, equality); for a wire's, `bits = 16 * rowWords`, so `rowWords ≠ 0` gives the template and the row by `lookup` (a `lookup`-to-`find?` lemma is needed), `16 * (words / k) ≤ 16 * words / k`, and `w.cut0 + w.cuts ≤ c.W.cutsOf …` from `wire_shape` and `shape_spec` (`RecordWire.lean`). `text.cutsOf = c.W.cutsOf` by `finish_text`'s `c.W.cuts = text.cuts`.
7. The universe: the list `L` of every row of slot `s` (templates, their units `u < t.units` with `slotOf … u = some s`, their rows, `i < cutsOf`). `Σ over L of rowBits ≤ inputs`: `rowBits_eq` (in the ask file; ids distinct and row names distinct by `tmpls_named`, `RecordTemplates.lean`), `k * (16 * words / k) ≤ 16 * words`, and a template has `t.units / c.S` units in a slot: `slot_units` (`RecordBlockSize.lean`) with `tiles_ordered`, `tiles_sum`, `gridsOf_ok`, `gridsOf_period`, `eq_of_nodup_map`, `mem_of_lookup` (`RecordBlocks.lean`). A unit with a slot is below its template's units (its grid lies inside them, `tiles_ordered`), which puts `F` and the covers inside `L`.
8. The abstract step, Mathlib only: for `R : Finset ρ`, `b : ρ → ℕ`, a list `ds` with pairwise disjoint `cov d` and weights `wt d`, where every `d` with `wt d ≠ 0` has `cov d ⊆ R`, `Disjoint F (cov d)` and `wt d ≤ b r` on its cover, and `F ⊆ R`: `Σ_F b + (ds.map fun d => (cov d).card * wt d).sum ≤ Σ_R b`. Induction on `ds`, adding `cov d` to `F`. With `wt d := if kind is inside or theta then d.bits else 0` and `R := L.toFinset` (`Σ over L.toFinset ≤ (L.map b).sum` by induction).

A LOCAL LOOP, seconds a try (my note 20261111T1649Z-checkpoint-trn115-p2-kinds-pushed-p3-stated has the command and a generator script): the VM has Mathlib's oleans and the toolchain's `lean`, so a scratch file outside the repository that imports only Mathlib compiles at once; step 8 and the list lemmas can be finished there, and the rest against a mock of the check-side definitions with the imported statements as axioms. `RecordKinds.lean` was written that way and passed its first node try.

ALSO PUSHED: `RecordKinds.lean` at 75cf031812, run r20261111-164546-b144 (lean exit 0, 4 theorems, the three axioms): M3 `theta_held (hc : finish f pin = .ok c) r (hk : c.rowKind r = Kind.theta) : (c.heldAt r).isSome = true`; its twin `inside_writer (h : read j tr = .ok c) r (hk : c.rowKind r = Kind.inside)`: the row has a writer, and the writer's slot is the row's; `cover_apart`; `fill_of_cover`.

FORM: two more modules for the umbrella, `Proofs.Properties.Nci.RecordKinds` and `Proofs.Properties.Nci.RecordFreshRows`.

training: `form_counts` is then three calls: my `block_fresh`, WIRED's `fresh_rows`, WIDTH's `block_gates`. My reading for `block_fresh` compiled: about 10 AM PST today; `form_counts` the hour after the later of WIRED's and WIDTH's pushes.

Names compiled, mine: K1, K2, M4, `field_leaf`, P2, `text_digits`, M3, `inside_writer`. Names left, mine: `block_fresh`, `form_counts`. WIRED's: `fresh_rows`.

