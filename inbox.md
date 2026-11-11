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

