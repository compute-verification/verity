[2026-11-11T15:59Z] d3fef124f555
WIDTH to COUNTS, copy training, TRN-125, Wed 11 Nov 7:59 AM PST. `widths_size` exists, read at cf9b216bd9: `theorem widths_size (t : Ty) : t.widths.size = t.leaves` in `security/Proofs/Properties/Nci/RecordRowBits.lean`, namespace `NciCheck.Properties.Record.SecurityProofs`. Do not prove it again. `field_leaf` as you state it is what I read; I import `RecordFields`, `RecordWire` and `RecordWide` from fc1ffb9dca.

[2026-11-11T16:00Z] e4593f7ad412
FORM to WIDTH, copy training and COUNTS, TRN-103, 16:00Z Wed 11 Nov. THE ASK FILE FOR THE CIRCUIT, and your 15:43Z answered.

1. YES to `templatesOf_tmplOf`, in my file, exactly as you wrote it:
   theorem templatesOf_tmplOf {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) : ∀ t ∈ ts, ∃ r, tmplOf P r = .ok t
   With it, same file: `tmpls_named (hts : templatesOf P doc = .ok ts) : (ts.map (·.tid)).Nodup ∧ ∀ t ∈ ts, (t.rows.map (·.1) ++ t.outs.map (·.1)).Nodup`; `templatesOf_spec` and `unit_tmpl` as my 15:42Z; `mapM_getElem?`; `evaluate_congr`; `templates_of_read (h : read j tr = .ok c) : ∃ rs, evaluate (ofProg c.f.P) (c.W.templates.map (·.1)).toArray = .ok rs ∧ Templates c.f.P rs c.W`.
   WRITTEN, NOT YET COMPILED: `RecordTemplates.lean` at c11fa199fc on `cursor/record-prim-operands-3b1c` (my scratch branch; the integration branch moves when it compiles). Its node try is r20261111-155816-b278, running. Namespace `NciCheck.Properties.Record.SecurityProofs`; imports `Proofs.Properties.Nci.Record`, `RecordBlocks`, `RecordProgram`. I send the head the minute it passes; take the statements as hypotheses until then.

2. THE HAND-OVER the coordinator named for you after K3 and K4: two statements, `RecordCircuitAsk.lean` on `cursor/record-property-asks-3b1c` at 5d797e2d28 (statements with `sorry`, never for landing; its elaboration try runs when a slot of mine frees and I send the run). Your file: `RecordCircuit.lean`.

   C1  theorem names_divides {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
         {D : Verity.Circuit} {at_ : Fin D.N → ℕ} (hL : LaysOut P rs W S D at_) (hO : OwnRows P rs W)
         (hG : GatesOwn P rs) (held : Row → Bool) (slot : String → ℕ → Option ℕ) (n : ℕ)
         (hslot : ∀ tid u sc, UnitAt P rs tid u sc → ∃ s, slot tid u = some s ∧ s < n) :
         ∃ θ : Finset (Fin D.N), (∀ g ∈ θ, D.isInput g = true) ∧ ∃ U : Verity.Partition D n,
           Names P rs W held at_ θ ∧ Divides P rs slot at_ U

   C2  theorem slot_of_unit {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c)
         {ts : Array String} {rs : Array TRange}
         (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P) ts = .ok rs) {tid : String} {u : ℕ}
         {sc : Scope} (hu : UnitAt c.f.P rs tid u sc) : ∃ s, c.slot tid u = some s ∧ s < c.S

   How I read each to hold (READ, not compiled):
   C1. θ: the classical filter of `Finset.univ` by `∃ r l lo hi, held r = true ∧ RowLeaf P rs W r l (at_ x) lo hi`, so `Names` is its membership. Each is an input: `hO.1` gives `P.gate (at_ x) = .ok (p, _, _)` with an input id for key, `hL`'s third conjunct gives the same gate with `D.isInput x = isInputId (P.defs[p]!).key`. U: for `D.isInput x = false`, `hL` gives the gate and `at_ x < gates P` (second conjunct, left to right at `⟨x, rfl⟩`), `hG` at it gives `tid u sc'` with `UnitAt` and the range in `Divides`' own form, `hslot` its slot `s < n`; choose them (`Classical.choice`), `unit x := if D.isInput x then none else some ⟨s, _⟩`. The policy check's proof has the same construction for its own circuit: `NciCheck.Properties.SecurityProofs.Decodes` (read by `Proofs/Properties/Remote.lean`'s `policyHeld`).
   C2. `read_front`, `finish_text` (`gridsOf c.f.ts c.f.doc = .ok (c.S, c.grids)`), `front_text`; `unitAt_iff` gives `t ∈ ts` with `t.tid = tid` and `u < t.units`; `gridsOf_covers` (with `tmpls_named`.1 or `templatesOf_ranges` for the `Nodup`) gives a grid of `(c.grids.lookup tid).getD []` holding `u`, so `slotIn`'s `find?` is `some`; the slot is `… % c.S` and `1 ≤ c.S` is `gridsOf_ok`.1.

   If either statement fails for you, tell me that hour and I change it; I change neither without telling you first.

3. The bounds on `ret_passed_on`: yours, in your file, as you say.

[2026-11-11T16:02Z] 4fcfe98edaa9
From WIRED (TRN-116), 8:02 AM PST. To COUNTS, copy the coordinator and FORM.

Your statements of M1 and M2 have not reached me. So that you lose no time: both are written in MY form, in `RecordInputs.lean` at `a619b35892` on `cursor/training-record-rows-05f2` (committed, not pushed until its node try passes; RAN locally: `lake env lean` on one file, exit 0, no warning). They are yours to change: send the form K1 and K2 read and I change the statement that hour. If these are what you read, say so.

M1, one walk of `inputsOf` with I1 read off it:

```lean
theorem inputsOf_decls {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) :
    Spot.Wires.pairwiseDisjoint (x.decls.map (·.w)) = true ∧
      ∃ rest, x.decls = (W.wires.zipIdx.map fun (w, i) =>
          ({ w, bits := 16 * rowWords ts W w, fill := .wire i } : Decl)) ++ rest ∧
        ∀ d ∈ rest, (∀ i, d.fill ≠ .wire i) ∧ ∃ t, ts.find? (·.tid == d.w.reader) = some t ∧
          d.w.start + d.w.readerUnits ≤ t.units ∧ d.w.cut0 + d.w.cuts ≤ W.cutsOf d.w.reader d.w.port ∧
          ((t.rows.find? (·.1 == d.w.port) = none ∧ d.bits = 0) ∨
            ∃ row, t.rows.find? (·.1 == d.w.port) = some row ∧
              d.bits = 16 * row.2.1 / W.cutsOf d.w.reader d.w.port)
```

What it says beyond the three facts, and what it does not:
- A later declaration is no wire's (`∀ i, d.fill ≠ .wire i`).
- Bits 0 comes WITH "its port is no row of the template": that is a registered read of an output. Every other later declaration has a row and that row's `16 * words / k`.
- `start + readerUnits ≤ t.units` holds for a rectangle at any cut: its `readerUnits` is `units * k / k`, which is `units` where `0 < k` and 0 where `k = 0`. It does NOT say `1 ≤ readerUnits`, which fails at `k = 0`; `inputsOf` alone does not make the text's cuts positive (`Spot.Wires.check` on `c.W` does, in `finish`). Say if you need `0 < k → 1 ≤ readerUnits`, or the rectangle itself (`d.w = rect …`).
- It does not say which template row `rowWords` read for a wire's own declaration: that is `Spot.Wires.check`'s.

M2:

```lean
theorem finish_case {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) :
    Spot.Wires.check c.W = .ok c.wired ∧ unwhole f.ts f.leaves c.wired.wires = none ∧
      rolesOf f.P f.ts pin c.x = .ok c.pins ∧
      c.inside = (c.wired.wires.map fun w => sameSlot c.S (c.gridsAt w.reader) (c.gridsAt w.writer) w).toArray ∧
      c.theta = c.x.params.map (fun (p, leaves) => (p, isTheta f.T leaves p)) ∧
      c.held = (c.x.openings.map (openingHeld (fun tid => (c.pins.lookup tid).isSome) c.isTheta c.x)).toArray
```

With `finish_text` (`RecordRead.lean`) it is everything `finish` puts in the case. I2 `wired_of_read` is now its first conjunct.

