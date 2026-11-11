module

public import Properties.Nci.Record
public import Properties.Verity.Game
public import Mathlib.Tactic

/-!
# A record of wide gates does not lay out the composite's declaration

The composite's declaration (`Verity.Vocab.declaration`) is the parsed circuit file's circuit, every gate one bit
wide (`declaration_w`), with each computed gate its own inference block (`Verity.RegisteredCircuit.ofCircuit`). The
record property's circuit (`Served`) is the Program's: each gate has its Program gate's width (`LaysOut`), and a
computed gate's inference block is its unit's slot (`Divides`). Two clauses of `Served` therefore fail at the
declaration:

* the width clause, where the Program has a computed gate wider than one bit (`not_served_declaration_of_wide`;
  for any circuit of one-bit gates and any gate no wire covers, `not_laysOut_of_wide`);
* the division, where every unit has the same slot and the declaration has two computed gates
  (`not_served_declaration_of_one_slot`): at such a declaration a computed gate's unit has the gate's own index as
  its slot (`divides_singletons_slot`).

Nothing here says which circuit the record property should name.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)

/-! ## The declaration's gates are one bit wide -/

private theorem parsed_circuit_w (o : Option Flock.Bridge.Parsed) (x : Fin (Flock.Bridge.Parsed.circuit o).N) :
    (Flock.Bridge.Parsed.circuit o).w x = 1 := by
  cases o with
  | none => exact x.elim0
  | some r => rfl

/-- Every gate of the composite's declaration is one bit wide: its circuit is the parsed circuit file's, a Boolean
circuit, or the empty circuit where the files don't parse. -/
theorem declaration_w {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Verity.Vocab d α)
    (θ : ByteArray) (Fs : Array ByteArray) (x : Fin (V.declaration θ Fs).D.N) : (V.declaration θ Fs).D.w x = 1 :=
  parsed_circuit_w (Flock.Bridge.parsedOf (V.read θ Fs).tags (V.read θ Fs).circuitFile (V.read θ Fs).tables
    (V.read θ Fs).publicFile) x

/-! ## The width clause -/

private theorem gate_lt {P : Prog} {g p : ℕ} {ops : Array ℕ} {sc : Scope} (h : P.gate g = .ok (p, ops, sc)) :
    g < gates P := by
  by_contra hlt
  have hge : g ≥ (P.defs[P.root]!).gates := not_lt.1 hlt
  unfold Flock.Program.Prog.gate at h
  simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- Where every unit's rows are its own input gates, no wire covers a computed gate. -/
private theorem not_wired_of_computed {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring}
    {S : Spot.Wires.Summary} (hO : OwnRows P rs W) {g p : ℕ} {ops : Array ℕ} {sc : Scope}
    (hgate : P.gate g = .ok (p, ops, sc)) (hcomp : Flock.Program.isInputId (P.defs[p]!).key = false) :
    ¬ Wired P rs W S g := by
  rintro ⟨r, l, lo, hi, -, hleaf⟩
  obtain ⟨p', ops', sc', hgate', hin⟩ := hO.1 r l g lo hi hleaf
  have hp : p' = p := (Prod.mk.inj (Except.ok.inj (hgate'.symm.trans hgate))).1
  rw [hp, hcomp] at hin
  exact absurd hin (by decide)

/-- **A circuit of one-bit gates lays out no Program with a wide gate that no wire covers**: `LaysOut` gives the
gate a gate of the circuit of its own width. -/
theorem not_laysOut_of_wide {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {D : Verity.Circuit} (hD : ∀ x : Fin D.N, D.w x = 1) {g p w : ℕ} {ops : Array ℕ} {sc : Scope}
    (hg : g < gates P) (hW : ¬ Wired P rs W S g) (hgate : P.gate g = .ok (p, ops, sc))
    (hret : (P.defs[p]!).ret = .value w) (hw : w ≠ 1) (at_ : Fin D.N → ℕ) : ¬ LaysOut P rs W S D at_ := by
  rintro ⟨-, honto, hgates⟩
  obtain ⟨x, hx⟩ := (honto g).2 ⟨hg, hW⟩
  obtain ⟨p', ops', sc', hgate', hret', -, -⟩ := hgates x
  rw [hx] at hgate'
  have hp : p' = p := (Prod.mk.inj (Except.ok.inj (hgate'.symm.trans hgate))).1
  rw [hp, hret, hD x] at hret'
  injection hret' with hw1
  exact hw hw1

/-- **The width clause fails at the composite's declaration**: it is the circuit of no record whose Program has a
computed gate wider than one bit. -/
theorem not_served_declaration_of_wide {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]
    (V : Verity.Vocab d α) (θ : ByteArray) (Fs : Array ByteArray) {F : Form} {c : F.Case} {P : Prog}
    (hP : Flock.Program.decode (F.inst c).program.toUTF8 = .ok P) {g p w : ℕ} {ops : Array ℕ} {sc : Scope}
    (hgate : P.gate g = .ok (p, ops, sc)) (hcomp : Flock.Program.isInputId (P.defs[p]!).key = false)
    (hret : (P.defs[p]!).ret = .value w) (hw : w ≠ 1) : ¬ Served F c (V.declaration θ Fs) := by
  rintro ⟨P', doc, W, S, rs, at_, hP', -, -, -, -, -, -, -, -, -, hO, -, -, -, hL, -, -⟩
  have hPP : P' = P := Except.ok.inj (hP'.symm.trans hP)
  rw [hPP] at hO hL
  exact not_laysOut_of_wide (D := (V.declaration θ Fs).D) (declaration_w V θ Fs) (gate_lt hgate)
    (not_wired_of_computed hO hgate hcomp) hgate hret hw at_ hL

/-! ## The division -/

/-- **At a declaration whose computed gates are each their own block, `Divides` gives a computed gate's unit the
gate's own index as its slot.** -/
theorem divides_singletons_slot {P : Prog} {rs : Array TRange} {slot : String → ℕ → Option ℕ} {C : Verity.Circuit}
    {at_ : Fin C.N → ℕ} (h : Divides P rs slot at_ (Verity.Partition.singletons C)) {x : Fin C.N}
    (hx : C.isInput x = false) :
    ∃ tid u sc, UnitAt P rs tid u sc ∧ sc.off ≤ at_ x ∧ at_ x < sc.off + (P.defs[sc.d]!).gates ∧
      slot tid u = some x.val := by
  obtain ⟨tid, u, sc, s, hu, hlo, hhi, hs, hmap⟩ := h x hx
  have hunit : (Verity.Partition.singletons C).unit x = some x := by simp [Verity.Partition.singletons, hx]
  rw [hunit] at hmap
  have hsx : x.val = s := Option.some.inj hmap
  exact ⟨tid, u, sc, hu, hlo, hhi, by rw [hs, hsx]⟩

/-- Where every unit has the same slot, a declaration whose computed gates are each their own block has at most one
computed gate. -/
theorem divides_singletons_one_slot {P : Prog} {rs : Array TRange} {slot : String → ℕ → Option ℕ}
    {C : Verity.Circuit} {at_ : Fin C.N → ℕ} (h : Divides P rs slot at_ (Verity.Partition.singletons C)) {s₀ : ℕ}
    (hone : ∀ tid u s, slot tid u = some s → s = s₀) {x y : Fin C.N} (hx : C.isInput x = false)
    (hy : C.isInput y = false) : x = y := by
  obtain ⟨tid, u, _, -, -, -, hsx⟩ := divides_singletons_slot h hx
  obtain ⟨tid', u', _, -, -, -, hsy⟩ := divides_singletons_slot h hy
  exact Fin.ext ((hone tid u _ hsx).trans (hone tid' u' _ hsy).symm)

/-- **The division fails at a circuit registered with each computed gate its own block**: it is the circuit of no
record whose units all have one slot, once it has two computed gates. -/
theorem not_served_ofCircuit_of_one_slot {F : Form} {c : F.Case} {C : Verity.Circuit} {θ : Finset (Fin C.N)}
    {hθ : ∀ g ∈ θ, C.isInput g = true} {s₀ : ℕ} (hone : ∀ tid u s, F.slot c tid u = some s → s = s₀)
    {x y : Fin C.N} (hx : C.isInput x = false) (hy : C.isInput y = false) (hxy : x ≠ y) :
    ¬ Served F c (Verity.RegisteredCircuit.ofCircuit C θ hθ) := by
  rintro ⟨P, doc, W, S, rs, at_, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hDiv⟩
  have hDiv' : Divides P rs (F.slot c) (D := C) at_ (Verity.Partition.singletons C) := hDiv
  exact hxy (divides_singletons_one_slot hDiv' hone hx hy)

/-- **The division fails at the composite's declaration**: it is the circuit of no record whose units all have one
slot, once the committed files' circuit has two computed gates. -/
theorem not_served_declaration_of_one_slot {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]
    (V : Verity.Vocab d α) (θ : ByteArray) (Fs : Array ByteArray) {F : Form} {c : F.Case} {s₀ : ℕ}
    (hone : ∀ tid u s, F.slot c tid u = some s → s = s₀) {x y : Fin (V.circuitOf θ Fs).N}
    (hx : (V.circuitOf θ Fs).isInput x = false) (hy : (V.circuitOf θ Fs).isInput y = false) (hxy : x ≠ y) :
    ¬ Served F c (V.declaration θ Fs) :=
  not_served_ofCircuit_of_one_slot (C := V.circuitOf θ Fs) (θ := V.theta θ Fs) (hθ := V.theta_input θ Fs) hone hx hy
    hxy

end NciCheck.Properties.Record.SecurityProofs
