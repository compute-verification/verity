module

public import Proofs.Properties.Nci.RecordTakes
public import Mathlib.Tactic

/-!
# The record's wires lay out a circuit

`laysOut_exists`: where the wires rank the units, the Program's gates that no wire covers are the gates of a
circuit (`LaysOut`), each with its Program gate's width and kind and reading where its operands land.

A circuit's gates are in an order in which every gate reads only earlier gates. The order here is by a key: input
gates first, then the least rank of a unit that holds the gate (`gateRank`), then the gate's index. A computed gate
lies in a unit, and each of its operands is an earlier gate of that unit or a leaf of one of the unit's reader rows.
A read of an earlier gate of the unit lands on that gate (`lands_inside`), which ranks no higher and has a smaller
index. A read of a row's leaf lands on an input gate or, through the wires, on a gate of a unit that ranks below the
row's (`lands_rank`): a row's writer ranks below the row's unit, and what the writer returns is a gate of its own or
a leaf of one of its own rows, followed in turn.

The construction reads no width beyond each gate's own, gives every gate a constant operation and names no output
gate: `LaysOut` leaves operations and outputs free.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open RecordText (Row)

/-! ## A finite set of gates, numbered by a key -/

/-- A finite set of naturals is numbered in the order of a key: by `κ` first, then by the natural itself. -/
private theorem numbered (U : Finset ℕ) (κ : ℕ → ℕ) :
    ∃ (N : ℕ) (at_ : Fin N → ℕ), Function.Injective at_ ∧ (∀ g, (∃ x, at_ x = g) ↔ g ∈ U) ∧
      ∀ x y, x < y ↔ κ (at_ x) < κ (at_ y) ∨ κ (at_ x) = κ (at_ y) ∧ at_ x < at_ y := by
  classical
  let key : ℕ → ℕ ×ₗ ℕ := fun g => toLex (κ g, g)
  let K : Finset (ℕ ×ₗ ℕ) := U.image key
  let e : Fin K.card ↪o ℕ ×ₗ ℕ := K.orderEmbOfFin rfl
  have hat : ∀ x, (ofLex (e x)).2 ∈ U ∧ e x = key (ofLex (e x)).2 := fun x => by
    obtain ⟨g, hg, hk⟩ := Finset.mem_image.1 (K.orderEmbOfFin_mem rfl x)
    have h2 : (ofLex (e x)).2 = g := congrArg (fun p => (ofLex p).2) hk.symm
    rw [h2]
    exact ⟨hg, hk.symm⟩
  refine ⟨K.card, fun x => (ofLex (e x)).2, ?_, ?_, ?_⟩
  · intro x y h
    apply e.injective
    rw [(hat x).2, (hat y).2]
    exact congrArg key h
  · intro g
    constructor
    · rintro ⟨x, rfl⟩
      exact (hat x).1
    · intro hg
      have hmem : key g ∈ Set.range e := by
        rw [K.range_orderEmbOfFin rfl]
        exact Finset.mem_coe.2 (Finset.mem_image_of_mem key hg)
      obtain ⟨x, hx⟩ := hmem
      exact ⟨x, congrArg (fun p => (ofLex p).2) hx⟩
  · intro x y
    rw [← e.lt_iff_lt, (hat x).2, (hat y).2]
    exact Prod.Lex.toLex_lt_toLex

/-- A circuit on a finite set of naturals: where every read goes back in the order of a key, the set's members,
numbered in that order, are the gates of a circuit, each of the width and kind given and reading the gates `R`
relates it to. Its operations are constant and it has no output gate. -/
private theorem circuit_of_key (U : Finset ℕ) (κ w : ℕ → ℕ) (inp : ℕ → Bool) (R : ℕ → ℕ → Prop)
    (hin : ∀ g ∈ U, inp g = true → ∀ a, ¬ R g a)
    (hlt : ∀ g ∈ U, ∀ a ∈ U, R g a → κ a < κ g ∨ κ a = κ g ∧ a < g) :
    ∃ (D : Verity.Circuit) (at_ : Fin D.N → ℕ), Function.Injective at_ ∧ (∀ g, (∃ x, at_ x = g) ↔ g ∈ U) ∧
      ∀ x, D.w x = w (at_ x) ∧ D.isInput x = inp (at_ x) ∧ ∀ a, a ∈ D.reads x ↔ R (at_ x) (at_ a) := by
  classical
  obtain ⟨N, at_, hinj, hon, hord⟩ := numbered U κ
  have hU : ∀ x, at_ x ∈ U := fun x => (hon _).1 ⟨x, rfl⟩
  refine ⟨{ N := N
            w := fun x => w (at_ x)
            isInput := fun x => inp (at_ x)
            reads := fun x => Finset.univ.filter fun a => R (at_ x) (at_ a)
            reads_lt := fun x a ha => (hord a x).2 (hlt _ (hU x) _ (hU a) (Finset.mem_filter.1 ha).2)
            input_reads := fun x hx => Finset.filter_eq_empty_iff.2 fun a _ => hin _ (hU x) hx _
            op := fun _ _ => 0
            op_congr := fun _ _ _ _ => rfl
            outputs := ∅ }, at_, hinj, hon, fun x => ⟨rfl, rfl, fun a => ?_⟩⟩
  exact ⟨fun ha => (Finset.mem_filter.1 ha).2, fun ha => Finset.mem_filter.2 ⟨Finset.mem_univ _, ha⟩⟩

/-! ## A Program gate's width, kind, reads and rank -/

/-- The width of Program gate `g`'s value: 0 where the gate is not located or its primitive returns no one value. -/
private def widthAt (P : Prog) (g : ℕ) : ℕ :=
  match P.gate g with
  | .ok (p, _, _) =>
    match (P.defs[p]!).ret with
    | .value w => w
    | _ => 0
  | .error _ => 0

/-- Whether Program gate `g` is an input gate. -/
private def inputAt (P : Prog) (g : ℕ) : Bool :=
  match P.gate g with
  | .ok (p, _, _) => Flock.Program.isInputId (P.defs[p]!).key
  | .error _ => false

/-- A read of an operand of Program gate `g` lands on Program gate `a`. -/
private def OperandLands (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    (g a : ℕ) : Prop :=
  ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ ∃ o ∈ ops, Lands P rs W S o a

/-- The rank of a Program gate under a rank `ρ` of the units: 0 where no unit holds the gate, else one more than
the least rank of a unit that holds it. -/
private noncomputable def gateRank (P : Prog) (rs : Array TRange) (ρ : String × ℤ → ℕ) (g : ℕ) : ℕ :=
  sInf {n | ∃ tid u sc, UnitAt P rs tid u sc ∧ sc.off ≤ g ∧ g < sc.off + (P.defs[sc.d]!).gates ∧
    n = ρ (tid, (u : ℤ)) + 1}

/-- The width of a located gate whose primitive returns one value. -/
private theorem widthAt_eq {P : Prog} {g p w : ℕ} {ops : Array ℕ} {sc : Scope}
    (h : P.gate g = .ok (p, ops, sc)) (hw : (P.defs[p]!).ret = .value w) : widthAt P g = w := by
  simp only [widthAt, h, hw]

/-- The kind of a located gate. -/
private theorem inputAt_eq {P : Prog} {g p : ℕ} {ops : Array ℕ} {sc : Scope}
    (h : P.gate g = .ok (p, ops, sc)) : inputAt P g = Flock.Program.isInputId (P.defs[p]!).key := by
  simp only [inputAt, h]

/-- A gate ranks at most one more than a unit that holds it. -/
private theorem gateRank_le {P : Prog} {rs : Array TRange} (ρ : String × ℤ → ℕ) {tid : String} {u g : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (h1 : sc.off ≤ g) (h2 : g < sc.off + (P.defs[sc.d]!).gates) :
    gateRank P rs ρ g ≤ ρ (tid, (u : ℤ)) + 1 :=
  Nat.sInf_le ⟨tid, u, sc, hu, h1, h2, rfl⟩

/-- A gate some unit holds ranks one more than a unit that holds it. -/
private theorem gateRank_mem {P : Prog} {rs : Array TRange} (ρ : String × ℤ → ℕ) {tid : String} {u g : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (h1 : sc.off ≤ g) (h2 : g < sc.off + (P.defs[sc.d]!).gates) :
    ∃ tid' u' sc', UnitAt P rs tid' u' sc' ∧ sc'.off ≤ g ∧ g < sc'.off + (P.defs[sc'.d]!).gates ∧
      gateRank P rs ρ g = ρ (tid', (u' : ℤ)) + 1 :=
  Nat.sInf_mem (s := {n | ∃ tid u sc, UnitAt P rs tid u sc ∧ sc.off ≤ g ∧
    g < sc.off + (P.defs[sc.d]!).gates ∧ n = ρ (tid, (u : ℤ)) + 1}) ⟨_, tid, u, sc, hu, h1, h2, rfl⟩

/-- An input gate ranks 0: no unit holds it. -/
private theorem gateRank_input {P : Prog} {rs : Array TRange} (ρ : String × ℤ → ℕ) (hU : UnitsComputed P rs)
    {g p : ℕ} {ops : Array ℕ} {sc : Scope} (h : P.gate g = .ok (p, ops, sc))
    (hin : Flock.Program.isInputId (P.defs[p]!).key = true) : gateRank P rs ρ g = 0 := by
  by_contra hne
  obtain ⟨n, tid, u, sc', hu, h1, h2, -⟩ := Nat.nonempty_of_pos_sInf (Nat.pos_of_ne_zero hne)
  have hc := hU tid u sc' g p ops sc hu h1 h2 h
  rw [hin] at hc
  exact Bool.noConfusion hc

/-! ## Where a read lands -/

/-- A read of a gate inside a unit lands on the gate itself: the gate is computed, so it is no leaf of a reader
row and no wire covers it. -/
private theorem lands_inside {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    (hO : OwnRows P rs W) (hU : UnitsComputed P rs) {tid : String} {u g a : ℕ} {sc : Scope}
    (hu : UnitAt P rs tid u sc) (h1 : sc.off ≤ g) (h2 : g < sc.off + (P.defs[sc.d]!).gates)
    (hl : Lands P rs W S g a) : a = g := by
  cases hl with
  | here _ _ => rfl
  | wire _ g' _ hf _ =>
    obtain ⟨r, l, l', lo, hi, lo', hi', hr, -⟩ := hf
    obtain ⟨p, ops, sc', hg, hin⟩ := hO.1 r l g lo hi hr
    have hc := hU tid u sc g p ops sc' hu h1 h2 hg
    rw [hin] at hc
    exact Bool.noConfusion hc

/-- A read of a leaf of a reader row lands on a gate that ranks no higher than the row's unit: on the leaf, an
input gate, or through a wire on a gate of the writer's unit, which ranks below the row's, or on where a read of
a leaf of one of the writer's own rows lands. -/
private theorem lands_rank {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    (hO : OwnRows P rs W) (hR : ReturnsOwn P rs W) (hU : UnitsComputed P rs) {ρ : String × ℤ → ℕ}
    (hρ : Ranked S ρ) {o a : ℕ} (hl : Lands P rs W S o a) :
    ∀ (r : Row) (l lo hi : ℕ), RowLeaf P rs W r l o lo hi →
      gateRank P rs ρ a ≤ ρ (r.tid, (r.unit : ℤ)) := by
  induction hl with
  | here o _ =>
    intro r l lo hi hr
    obtain ⟨p, ops, sc, hg, hin⟩ := hO.1 r l o lo hi hr
    rw [gateRank_input ρ hU hg hin]
    exact Nat.zero_le _
  | wire o g a hf hl ih =>
    intro r l lo hi hr
    obtain ⟨r₁, l₁, l', lo₁, hi₁, lo', hi', hr₁, hw, -, -⟩ := hf
    obtain ⟨rfl, -⟩ := hO.2 r l lo hi r₁ l₁ lo₁ hi₁ o hr hr₁
    obtain ⟨wt, wu, out, sc', hwr, h0, hu, hout⟩ := hw
    have hlt := hρ r wt wu out hwr
    have hwu : ((wu.toNat : ℕ) : ℤ) = wu := Int.toNat_of_nonneg h0
    rcases hR wt wu.toNat sc' out l' g lo' hi' hu hout with ⟨h1, h2⟩ | ⟨r', l'', lo'', hi'', ht, hun, hr'⟩
    · have hag := lands_inside hO hU hu h1 h2 hl
      have hle := gateRank_le ρ hu h1 h2
      rw [hwu] at hle
      rw [hag]
      omega
    · have hle := ih r' l'' lo'' hi'' hr'
      rw [ht, hun, hwu] at hle
      omega

/-! ## The wired circuit -/

/-- **The record's wires lay out a circuit.** Where the wires rank the units, every unit's rows are its own input
gates, what a unit returns and what its gates read are its own, a gate inside a unit is computed and every computed
gate lies in a unit, the Program's gates that no wire covers are the gates of a circuit: in the order of input
gates first, then the least rank of a unit holding the gate, then the gate's index, every gate reads only earlier
gates. -/
theorem laysOut_exists (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    (hO : OwnRows P rs W) (hR : ReturnsOwn P rs W) (hU : UnitsComputed P rs) (hOp : OperandsOwn P rs W)
    (hG : GatesOwn P rs) (ρ : String × ℤ → ℕ) (hρ : Ranked S ρ) :
    ∃ (D : Verity.Circuit) (at_ : Fin D.N → ℕ), LaysOut P rs W S D at_ := by
  classical
  have hmemU : ∀ g, g ∈ (Finset.range (gates P)).filter (fun g => ¬ Wired P rs W S g) ↔
      g < gates P ∧ ¬ Wired P rs W S g := fun g => by
    rw [Finset.mem_filter, Finset.mem_range]
  have hin : ∀ g ∈ (Finset.range (gates P)).filter (fun g => ¬ Wired P rs W S g), inputAt P g = true →
      ∀ a, ¬ OperandLands P rs W S g a := by
    rintro g hgU hin a ⟨p, ops, sc, hg, o, ho, -⟩
    obtain ⟨p', ops', sc', w, hg', -, hops, -⟩ := hG g ((hmemU g).1 hgU).1
    rw [hg] at hg'
    cases hg'
    rw [inputAt_eq hg] at hin
    rw [hops hin] at ho
    exact absurd ho (Array.not_mem_empty o)
  have hlt : ∀ g ∈ (Finset.range (gates P)).filter (fun g => ¬ Wired P rs W S g),
      ∀ a ∈ (Finset.range (gates P)).filter (fun g => ¬ Wired P rs W S g), OperandLands P rs W S g a →
        gateRank P rs ρ a < gateRank P rs ρ g ∨ gateRank P rs ρ a = gateRank P rs ρ g ∧ a < g := by
    rintro g hgU a - ⟨p, ops, sc, hg, o, ho, hl⟩
    obtain ⟨p', ops', sc', w, hg', -, hops, hunit⟩ := hG g ((hmemU g).1 hgU).1
    rw [hg] at hg'
    cases hg'
    cases hi : Flock.Program.isInputId (P.defs[p]!).key with
    | true =>
      rw [hops hi] at ho
      exact absurd ho (Array.not_mem_empty o)
    | false =>
      obtain ⟨tid₀, u₀, sc₀, hu₀, h1₀, h2₀⟩ := hunit hi
      obtain ⟨tid, u, scu, hu, h1, h2, hrank⟩ := gateRank_mem ρ hu₀ h1₀ h2₀
      rcases hOp tid u scu g p ops sc hu h1 h2 hg o ho with ⟨ho1, ho2⟩ | ⟨r, l, lo, hi', ht, hun, hr⟩
      · have ho3 : o < scu.off + (P.defs[scu.d]!).gates := by omega
        have hao := lands_inside hO hU hu ho1 ho3 hl
        have hle := gateRank_le ρ hu ho1 ho3
        rw [hao]
        omega
      · have hle := lands_rank hO hR hU hρ hl r l lo hi' hr
        rw [ht, hun] at hle
        omega
  obtain ⟨D, at_, hinj, hon, hD⟩ :=
    circuit_of_key ((Finset.range (gates P)).filter fun g => ¬ Wired P rs W S g) (gateRank P rs ρ)
      (widthAt P) (inputAt P) (OperandLands P rs W S) hin hlt
  refine ⟨D, at_, hinj, fun g => (hon g).trans (hmemU g), fun x => ?_⟩
  obtain ⟨p, ops, sc, w, hg, hw, -, -⟩ := hG (at_ x) ((hmemU _).1 ((hon _).1 ⟨x, rfl⟩)).1
  obtain ⟨hDw, hDi, hDr⟩ := hD x
  refine ⟨p, ops, sc, hg, ?_, ?_, fun a => ?_⟩
  · rw [hDw, widthAt_eq hg hw]
    exact hw
  · rw [hDi, inputAt_eq hg]
  · rw [hDr a]
    constructor
    · rintro ⟨p', ops', sc', hg', h⟩
      rw [hg] at hg'
      cases hg'
      exact h
    · exact fun h => ⟨p, ops, sc, hg, h⟩

end NciCheck.Properties.Record.SecurityProofs
