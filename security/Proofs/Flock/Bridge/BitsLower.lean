module

public import Proofs.Flock.Bridge.Defs
public import Properties.Spot.Bits

/-!
# The bridge's lowering is Bits' (`Spot.Bits.prog`, `posOf`, `blockOf`)

`LaysOut kp C`: the Bits key program `kp` lays out the circuit `C`. It has `C`'s gates, gate by gate an input where `kp`
has no row and a row reading the same gates where it has one, and `C`'s outputs, ascending. At such a pair Bits'
lowering is the bridge's, gate for gate, under `bitsGate`:

* `gateSize_eq`, `off_eq`, `posOf_eq`: each gate lowers to as many gates, after as many, at the same position.
* `lower_eq`, `progOf_eq`: at constants that agree, each gate lowers to the same gates, so `progOf C κ'` is
  `bitsProg kp κ`.

`UnitsOut kp P`: the units agree too, `kp`'s `units` being `P`'s and each gate in the same unit. Then each unit's
lowered gates (`unitGates_eq`), each gate's owner (`owner_eq`), each unit's owned inputs (`insOf_eq`) and its proof
block (`blockOf_eq`) are Bits'.

Nothing here reads how `kp` or `C` is built: the key program's `LaysOut` and `UnitsOut` are proved where it is.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit
open Properties.Spot (bitsGate bitsProg)

/-- **The Bits key program `kp` lays out the circuit `C`**: as many gates; an input where `kp` has no row, and a row
reading the same gates where it has one; and `C`'s outputs, in gate order. -/
structure LaysOut (kp : Spot.Bits.KeyProg) (C : Verity.Boolean.Circuit) : Prop where
  size : kp.size = C.N
  input : ∀ g : Fin C.N, kp.row? g.val = none → C.op g = .input
  row : ∀ g : Fin C.N, ∀ a b, kp.row? g.val = some (a, b) →
    ∃ A B, C.op g = .row A B ∧ a = A.map Fin.val ∧ b = B.map Fin.val
  outs : kp.outs = ((List.finRange C.N).filter fun g => decide (g ∈ C.outputs)).map Fin.val

variable {kp : Spot.Bits.KeyProg} {C : Verity.Boolean.Circuit}

private theorem finRange_map_val (n : ℕ) : (List.finRange n).map Fin.val = List.range n :=
  List.ext_getElem (by simp) fun i _ _ => by simp

theorem LaysOut.gateSize_eq (h : LaysOut kp C) (g : Fin C.N) :
    gateSize (C.op g) = Spot.Bits.gateSize kp g.val := by
  unfold Spot.Bits.gateSize
  rcases hr : kp.row? g.val with _ | ⟨a, b⟩
  · rw [h.input g hr]
    rfl
  · obtain ⟨A, B, hop, -, -⟩ := h.row g a b hr
    rw [hop]
    rfl

theorem LaysOut.off_eq (h : LaysOut kp C) (g : ℕ) : off C g = Spot.Bits.off kp g := by
  unfold off Spot.Bits.off
  rw [h.size, ← finRange_map_val, List.filter_map, List.map_map]
  exact congrArg List.sum (List.map_congr_left fun x _ => h.gateSize_eq x)

theorem LaysOut.posOf_eq (h : LaysOut kp C) (g : Fin C.N) : posOf C g = Spot.Bits.posOf kp g.val := by
  unfold posOf Spot.Bits.posOf
  rw [h.off_eq, h.gateSize_eq]

theorem LaysOut.posOf_comp (h : LaysOut kp C) : (Spot.Bits.posOf kp ∘ Fin.val : Fin C.N → ℕ) = posOf C :=
  funext fun x => (h.posOf_eq x).symm

/-- **Gate `g` lowers to the same gates**, at constants that agree. -/
theorem LaysOut.lower_eq (h : LaysOut kp C) {κ' : Fin C.N → Option Bool} {κ : ℕ → Option Bool}
    (hκ : ∀ g, κ' g = κ g.val) (g : Fin C.N) :
    lower C κ' g = (Spot.Bits.lower kp κ g.val).map bitsGate := by
  unfold lower Spot.Bits.lower
  rcases hr : kp.row? g.val with _ | ⟨a, b⟩
  · rw [h.input g hr, hκ g]
    cases κ g.val <;> rfl
  · obtain ⟨A, B, hop, rfl, rfl⟩ := h.row g _ _ hr
    rw [hop]
    simp only [List.map_cons, List.map_nil, bitsGate, List.map_map, h.posOf_comp, h.off_eq]

/-- **The bridge's program is Bits'**, at constants that agree. -/
theorem LaysOut.progOf_eq (h : LaysOut kp C) {κ' : Fin C.N → Option Bool} {κ : ℕ → Option Bool}
    (hκ : ∀ g, κ' g = κ g.val) : progOf C κ' = bitsProg kp κ := by
  unfold progOf bitsProg Spot.Bits.prog
  congr 1
  · rw [List.map_flatMap, h.size, ← finRange_map_val, List.flatMap_map]
    exact List.flatMap_congr fun g _ => h.lower_eq hκ g
  · rw [h.outs, List.map_map, h.posOf_comp]

/-! ## Units and proof blocks -/

/-- **`kp`'s units are `P`'s**: as many, and each gate in the same unit. -/
structure UnitsOut (kp : Spot.Bits.KeyProg) {n : ℕ} (P : Partition C n) : Prop where
  units : kp.units = n
  unit : ∀ g : Fin C.N, Spot.Bits.unitOf kp g.val = (P.unit g).map Fin.val

variable {n : ℕ} {P : Partition C n}

private theorem map_val_eq_some {m : ℕ} {o : Option (Fin m)} {u : Fin m} :
    o.map Fin.val = some u.val ↔ o = some u := by
  cases o <;> simp [Fin.ext_iff]

private theorem decide_eq_map_beq {m : ℕ} (o : Option (Fin m)) (u : Fin m) :
    decide (o = some u) = (o.map Fin.val == some u.val) := by
  rw [Bool.eq_iff_iff]
  cases o <;> simp [Fin.ext_iff]

theorem LaysOut.args_eq (h : LaysOut kp C) (g : Fin C.N) :
    Spot.Bits.args kp g.val = (C.op g).args.map Fin.val := by
  unfold Spot.Bits.args
  rcases hr : kp.row? g.val with _ | ⟨a, b⟩
  · rw [h.input g hr]
    rfl
  · obtain ⟨A, B, hop, rfl, rfl⟩ := h.row g _ _ hr
    rw [hop, Verity.Boolean.Op.args, List.map_append]

/-- **Unit `u`'s lowered gates are Bits'.** -/
theorem unitGates_eq (h : LaysOut kp C) (hu : UnitsOut kp P) (u : Fin n) :
    unitGates P u = Spot.Bits.unitGates kp u.val := by
  unfold unitGates Spot.Bits.unitGates
  rw [h.size, ← finRange_map_val, List.flatMap_map]
  refine List.flatMap_congr fun g _ => ?_
  rw [hu.unit g, h.gateSize_eq, h.off_eq]
  by_cases hg : P.unit g = some u
  · simp [hg]
  · have : ¬ (P.unit g).map Fin.val = some u.val := fun e => hg (map_val_eq_some.1 e)
    simp [hg, this]

/-- **Gate `g`'s owner is Bits'.** -/
theorem owner_eq (h : LaysOut kp C) (hu : UnitsOut kp P) (g : Fin C.N) :
    Spot.Bits.owner kp g.val = (owner P g).map Fin.val := by
  unfold Spot.Bits.owner owner
  rw [hu.units, ← finRange_map_val n, List.find?_map]
  congr 1
  refine List.find?_congr fun u _ => ?_
  rw [Bool.eq_iff_iff]
  simp only [Function.comp_apply, List.any_eq_true, Bool.and_eq_true, beq_iff_eq, List.contains_iff_mem,
    decide_eq_true_eq]
  constructor
  · rintro ⟨x, hx, hux, hgx⟩
    have hxN : x < C.N := h.size ▸ List.mem_range.1 hx
    refine ⟨⟨x, hxN⟩, map_val_eq_some.1 ((hu.unit ⟨x, hxN⟩).symm.trans hux), ?_⟩
    rw [h.args_eq ⟨x, hxN⟩] at hgx
    obtain ⟨g', hg', hgg⟩ := List.mem_map.1 hgx
    rwa [← Fin.ext hgg]
  · rintro ⟨x, hux, hgx⟩
    refine ⟨x.val, List.mem_range.2 (by rw [h.size]; exact x.isLt), (hu.unit x).trans (by rw [hux]; rfl), ?_⟩
    rw [h.args_eq x]
    exact List.mem_map_of_mem hgx

theorem LaysOut.row_isNone (h : LaysOut kp C) (g : Fin C.N) : (kp.row? g.val).isNone = (C.op g).isInput := by
  rcases hr : kp.row? g.val with _ | ⟨a, b⟩
  · rw [h.input g hr]
    rfl
  · obtain ⟨A, B, hop, -, -⟩ := h.row g a b hr
    rw [hop]
    rfl

/-- **Unit `u`'s owned inputs are Bits'**, at constants that agree. -/
theorem insOf_eq (h : LaysOut kp C) (hu : UnitsOut kp P) {κ' : Fin C.N → Option Bool} {κ : ℕ → Option Bool}
    (hκ : ∀ g, κ' g = κ g.val) (u : Fin n) :
    (insOf P κ' u).map Fin.val = Spot.Bits.insOf kp κ u.val := by
  unfold insOf Spot.Bits.insOf
  rw [h.size, ← finRange_map_val, List.filter_map]
  congr 1
  refine List.filter_congr fun g _ => ?_
  simp only [Function.comp, h.row_isNone, hκ, owner_eq h hu, decide_eq_map_beq]

/-- **Unit `u`'s proof block is Bits'**, at constants that agree. -/
theorem blockOf_eq (h : LaysOut kp C) (hu : UnitsOut kp P) {κ' : Fin C.N → Option Bool} {κ : ℕ → Option Bool}
    (hκ : ∀ g, κ' g = κ g.val) (u : Fin n) :
    blockOf P κ' u = Spot.Bits.blockOf kp κ u.val := by
  unfold blockOf Spot.Bits.blockOf
  rw [unitGates_eq h hu, ← insOf_eq h hu hκ, List.map_map, h.posOf_comp]

end Flock.Bridge
