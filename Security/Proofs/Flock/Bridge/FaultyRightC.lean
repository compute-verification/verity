module

public import Proofs.Flock.Bridge.Canon
public import Proofs.Flock.Claims.Bridge.WrongSet
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Bridge.Lowers
import Proofs.Flock.SamplingBoundary
import Std.Tactic.BVDecide.Normalize.BitVec

/-!
# A unit right on its canonical openings is not faulty

`faultyRightC`, the canonical form of `FaultyRightTarget` (`Bridge.Targets`), proves `FaultyRightTargetC`
(`Bridge.Canon`). `canon g v` is any predicate on the value of an opening at wire `g`; the bridge's is `canonRow`: the
wire has a row, and the value is RowShaped at its row's width. A unit `u` is not faulty in `witDep C Cm H roots Xr`
when its committed wires each carry, in some canonical opening of their registered commit string, their value in `C`'s
evaluation on `Xr`'s inputs (`RightRegZc`'s second conjunct), and each of its canonical registered openings opens its
wire's position under `Cm` and `roots`. It reads the transport only at the right openings, so at
`canon := fun _ _ => True` it is `FaultyRightTarget`.

* `reads_lower_args`: a lowered gate reads the positions of its wire's arguments, or a row's AND its own two XORs.
* `need_posOf`: every gate of `need` that the unit reads from outside or commits is the position of a wire in `P.io u`
  (`boundary_inner` excludes a row's two XORs).
* `witDep_val`: `witDep` opens a wire's position at its value wherever some datum passes there (`posOf_injective`).
* Slot `u` is unit `u`'s proof block with the inputs it owns, which leave its values alone (`ref_blockOf`), so it is
  correct when the unit's gates are and each owned input opens (`correct_blockOf`). `correct_blockOf_of`: at any wire
  values `V` that apply the unit's gates, carried by the committed values at its committed wires, `lowers_of` gives the
  unit's gates correct whatever the interior, and each input the unit owns is one of its committed wires
  (`mem_io_of_owner`), so it opens. `faultyRightC` takes `V` the evaluation on `Xr`; `correct_of_local` takes `u`'s
  local evaluation on values `X` that make `u` correct (`Partition.Correct`), which is `X` on `u`'s committed wires.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

variable {C : Verity.Boolean.Circuit} {κ : Fin C.N → Option Bool}

theorem mem_io_of_read {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n} {a h : Fin C.N}
    (hh : P.unit h = some u) (ha : a ∈ (C.op h).args) (hne : P.unit h ≠ P.unit a) : a ∈ P.io u := by
  simp only [FlockSoundness.Audit.Partition.io, FlockSoundness.Audit.Partition.committed, Finset.mem_filter,
    Finset.mem_univ, true_and]
  exact ⟨Or.inr (Or.inr ⟨h, ha, hne⟩), Or.inr ⟨h, hh, ha⟩⟩

/-- **An input a unit owns is one of its committed wires**: an input is committed, and its owner reads it. -/
theorem mem_io_of_owner {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n} {a : Fin C.N}
    (ha : a ∈ insOf P κ u) : a ∈ P.io u := by
  obtain ⟨hin, -, ho⟩ := mem_insOf.1 ha
  obtain ⟨h, hh, hr⟩ := reads_of_owner ho
  refine mem_io_of_read hh hr ?_
  rw [hh, (P.input_iff a).2 (by simp [hin, Verity.Boolean.Op.isInput])]
  simp

/-- **A position the unit commits is a wire of `P.io u`**: a gate of `unitGates P u` that `Boundary` commits is the
position of one of `u`'s wires, and that wire is committed. -/
theorem posOf_of_boundary {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n} {i : ℕ}
    (hi : i ∈ unitGates P u) (hB : (progOf C κ).Boundary (unitGates P u) i = true) :
    ∃ a ∈ P.io u, i = posOf C a := by
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hi
  by_cases hjl : j + 1 < gateSize (C.op g)
  · rw [boundary_inner hu hjl] at hB
    exact absurd hB (by simp)
  have hpos : off C g.val + j = posOf C g := by rw [posOf_eq]; omega
  rw [hpos] at hB ⊢
  refine ⟨g, ?_, rfl⟩
  have hcom : g ∈ P.committed := by
    simp only [Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true,
      List.mem_range, Bool.and_eq_true, decide_not, Bool.not_eq_true', decide_eq_false_iff_not] at hB
    simp only [FlockSoundness.Audit.Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and]
    rcases hB with (hout | hin) | ⟨g', -, hg'u, hrd⟩
    · simp only [progOf, List.mem_map, List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq] at hout
      obtain ⟨a', ha', he⟩ := hout
      rw [posOf_injective he] at ha'
      exact Or.inr (Or.inl ha')
    · have h1 := one_le_gateSize (C.op g)
      rw [posOf_eq, gates_off g (by omega)] at hin
      exfalso
      cases hop : C.op g with
      | input => exact absurd ((P.input_iff g).2 (by simp [hop, Verity.Boolean.Op.isInput])) (by simp [hu])
      | and a b => simp [lower, hop, gateSize] at hin
      | xor a b => simp [lower, hop, gateSize] at hin
      | not a => simp [lower, hop, gateSize] at hin
      | row a b => simp [lower, hop, gateSize] at hin
    · cases hx : (progOf C κ).gates[g']? with
      | none => simp [hx] at hrd
      | some x =>
        simp only [hx, Option.map_some, Option.getD_some] at hrd
        obtain ⟨h, k, hk, rfl, hxk⟩ := exists_block hx
        have hhu : P.unit h ≠ some u := fun hh => hg'u (mem_unitGates.2 ⟨h, hh, k, hk, rfl⟩)
        rcases reads_lower_args hxk hrd with ⟨a, ha, he⟩ | ⟨rfl, hi⟩
        · rw [posOf_injective he] at hu ⊢
          exact Or.inr (Or.inr ⟨h, ha, fun e => hhu (e.trans hu)⟩)
        · exfalso
          rcases hi with hi | hi
          · exact off_ne_posOf (g := h) (a := g) (j := 0) (by omega) (by rw [Nat.add_zero]; exact hi.symm)
          · exact off_ne_posOf (g := h) (a := g) (j := 1) (by omega) hi.symm
  simp only [FlockSoundness.Audit.Partition.io, Finset.mem_filter]
  exact ⟨hcom, Or.inl hu⟩

/-- **What the unit needs, outside its interior, is its committed wires' positions**: a gate of `need` of
`unitGates P u` outside the unit, or one the unit commits, is the position of a wire in `P.io u`. -/
theorem need_posOf {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n} {i : ℕ}
    (hi : i ∈ FlockSoundness.Discharge.ZkLink.WrongSet.need (progOf C κ) (unitGates P u))
    (hB : i ∉ unitGates P u ∨ (progOf C κ).Boundary (unitGates P u) i = true) :
    ∃ a ∈ P.io u, i = posOf C a := by
  by_cases hin : i ∈ unitGates P u
  · exact posOf_of_boundary hin (hB.resolve_left (not_not.2 hin))
  simp only [FlockSoundness.Discharge.ZkLink.WrongSet.need, List.mem_append, List.mem_flatMap] at hi
  rcases hi with hi | ⟨p, hp, hrd⟩
  · exact absurd hi hin
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hp
  rw [gates_off g hj] at hrd
  cases hx : (lower C κ g)[j]? with
  | none => simp [hx] at hrd
  | some x =>
    simp only [hx, Option.map_some, Option.getD_some] at hrd
    rcases reads_lower_args hx hrd with ⟨a, ha, rfl⟩ | ⟨rfl, hi⟩
    · have hau : P.unit a ≠ some u := fun h => hin (posOf_mem_unitGates.2 h)
      exact ⟨a, mem_io_of_read hu ha (fun e => hau (e.symm.trans hu)), rfl⟩
    · exfalso
      rcases hi with rfl | rfl
      · exact hin (mem_unitGates.2 ⟨g, hu, 0, by omega, by rw [Nat.add_zero]⟩)
      · exact hin (mem_unitGates.2 ⟨g, hu, 1, by omega, rfl⟩)

theorem witDep_mem {Cm : CommitmentScheme Bool} {H : Claims.Msg → Claims.Msg} {roots : List Claims.Msg}
    {xv : Fin C.N → Bool} {o : Opening Bool} (h : o ∈ (witDep C Cm H roots xv).vals) :
    ∃ g, o.pos = posOf C g ∧ o.val = C.eval xv g := by
  simp only [witDep, List.mem_map, List.mem_finRange, true_and] at h
  obtain ⟨g, hg⟩ := h
  refine ⟨g, ?_⟩
  subst hg
  split <;> exact ⟨rfl, rfl⟩

/-- **`witDep` opens a wire's position at its value** wherever some datum passes there. -/
theorem witDep_val {Cm : CommitmentScheme Bool} {H : Claims.Msg → Claims.Msg} {roots : List Claims.Msg}
    {xv : Fin C.N → Bool} (s : Claims.Sampling.AuditStatement) (hs : s.roots = roots) (a : Fin C.N)
    (hex : ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C a, C.eval xv a, d⟩ = true) :
    (witDep C Cm H roots xv).val H Cm s (posOf C a) = some (C.eval xv a) := by
  unfold Wit.val
  rw [hs]
  have hfa : ∃ o ∈ (witDep C Cm H roots xv).vals, (o.pos == posOf C a && Cm.Opens H roots o) = true := by
    refine ⟨⟨posOf C a, C.eval xv a, Classical.choose hex⟩, ?_, ?_⟩
    · simp only [witDep, List.mem_map, List.mem_finRange, true_and]
      exact ⟨a, by simp [hex]⟩
    · simp only [beq_self_eq_true, Bool.true_and]
      exact Classical.choose_spec hex
  obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 (List.find?_isSome.2 hfa)
  rw [ho, Option.map_some]
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨g', hg'p, hg'v⟩ := witDep_mem (List.mem_of_find?_eq_some ho)
  rw [posOf_injective (hg'p.symm.trans hpo.1)] at hg'v
  rw [hg'v]

/-- **Slot `u` is correct on committed values that carry wire values `V` applying the unit's gates**: its proof block
`blockOf P κ u` is correct on committed values `val` carrying `V` at each committed wire of `u` that `κ` doesn't fix,
whatever its interior values `ι`, at constants that fix only inputs and that are `V`'s at `u`'s committed wires
(`lowers_of` at `wireAt`, and each input `u` owns is one of its committed wires). -/
theorem correct_blockOf_of {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (u : Fin n)
    (hκin : ∀ g b, κ g = some b → (C.op g).isInput = true)
    (V : Fin C.N → Bool) (hV : ∀ g, P.unit g = some u → V g = (C.op g).apply V)
    (hκ : ∀ g ∈ P.io u, ∀ b, κ g = some b → V g = b)
    (val : ℕ → Option Bool) (hval : ∀ g ∈ P.io u, κ g = none → val (posOf C g) = some (V g)) (ι : ℕ → Bool) :
    (progOf C κ).Correct ((progOf C κ).refIn (blockOf P κ u) val ι) (blockOf P κ u) := by
  have hκ' : ∀ a b, C.op a = .input → κ a = some b → (∃ h, P.unit h = some u ∧ a ∈ (C.op h).args) → V a = b := by
    rintro a b hin hk ⟨h, hh, ha⟩
    refine hκ a (mem_io_of_read hh ha ?_) b hk
    rw [hh, (P.input_iff a).2 (by simp [hin, Verity.Boolean.Op.isInput])]
    simp
  rw [ref_blockOf, correct_blockOf]
  by_cases hN : Nonempty (Fin C.N)
  · obtain ⟨d⟩ := hN
    refine ⟨lowers_of P u (wireAt C d) (wireAt_posOf d) V hV hκ' _ fun i hi hc hB => ?_, fun a ha => ?_⟩
    · obtain ⟨a, ha, rfl⟩ := need_posOf hi hB
      have hnone : κ a = none := by
        cases hk : κ a with
        | none => rfl
        | some b => exact absurd (gates_posOf_const (input_of_isInput (hκin a b hk)) hk) (hc b)
      have hv : (progOf C κ).refIn (unitGates P u) val ι (posOf C a) = val (posOf C a) := by
        rcases hB with hB | hB
        · exact Boundary.ref_out _ hB
        · exact Boundary.ref_boundary _ hB
      rw [hv, wireAt_posOf, hval a ha hnone]
    · obtain ⟨hin, hk, -⟩ := mem_insOf.1 ha
      rw [Boundary.ref_input _ (gates_posOf_input hin hk), hval a (mem_io_of_owner ha) hk]
      simp
  · refine ⟨?_, fun a _ => absurd ⟨a⟩ hN⟩
    intro g hg
    obtain ⟨w, -⟩ := mem_unitGates.1 hg
    exact absurd ⟨w⟩ hN

/-- **A unit correct on its own values is correct at its slot**: where `Xu` makes `u` correct (`Partition.Correct`: its
committed outputs are its gates applied to its committed inputs, all read from `Xu`) and is `κ`'s constant at each of
`u`'s committed wires `κ` fixes, slot `u`'s proof block is correct on committed values that carry `Xu` at each committed
wire of `u` that `κ` doesn't fix, whatever its interior values (`correct_blockOf_of` at `u`'s local evaluation on `Xu`,
which is `Xu` on `u`'s committed wires). -/
theorem correct_of_local {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (u : Fin n)
    (hκin : ∀ g b, κ g = some b → (C.op g).isInput = true)
    (Xu : Fin C.N → Bool) (hX : P.Correct Xu u) (hκ : ∀ g ∈ P.io u, ∀ b, κ g = some b → Xu g = b)
    (val : ℕ → Option Bool) (hval : ∀ g ∈ P.io u, κ g = none → val (posOf C g) = some (Xu g)) (ι : ℕ → Bool) :
    (progOf C κ).Correct ((progOf C κ).refIn (blockOf P κ u) val ι) (blockOf P κ u) := by
  have hloc : ∀ g ∈ P.io u, P.localEval Xu u g = Xu g := by
    intro g hg
    by_cases hu : P.unit g = some u
    · simp only [FlockSoundness.Audit.Partition.io, Finset.mem_filter] at hg
      exact (hX g hg.1 hu).symm
    · exact P.localEval_of_not_mem Xu hu
  exact correct_blockOf_of P u hκin (P.localEval Xu u) (fun _ hu => P.localEval_of_mem Xu hu)
    (fun g hg b hk => (hloc g hg).trans (hκ g hg b hk)) val
    (fun g hg hk => (hval g hg hk).trans (congrArg some (hloc g hg).symm)) ι

/-- **A unit right on its canonical openings is not faulty** (`FaultyRightTarget` at canonical openings), at constants
`κ` that fix only inputs: a constant the unit reads is `.const`, which carries its value when `hκ` holds there, and any
other wire it reads is transported (`correct_blockOf_of` at `C`'s evaluation on `Xr`). -/
theorem faultyRightC {n : ℕ} (P : FlockSoundness.Audit.Partition C n) {Pos Val Opn Com : Type}
    (pos : Fin C.N → Pos) (bit : Val → Fin C.N → Bool) (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com)
    (canon : Fin C.N → Val → Prop) (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (roots : List Claims.Msg)
    (Xr : Fin C.N → Bool) (κ : Fin C.N → Option Bool) (u : Fin n)
    (hκin : ∀ g b, κ g = some b → (C.op g).isInput = true)
    (hright : ∃ op : Pos → Opn, ∀ g ∈ P.io u,
      com (op (pos g)) = cs (pos g) ∧ bit (val (op (pos g))) g = C.eval Xr g ∧ canon g (val (op (pos g))))
    (htr : ∀ g ∈ P.io u, κ g = none → ∀ o : Opn, com o = cs (pos g) → canon g (val o) →
      ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C g, bit (val o) g, d⟩ = true)
    (hκ : ∀ g ∈ P.io u, ∀ b, κ g = some b → C.eval Xr g = b)
    {ns : List ℕ} (hns : ns.sum = n) (r : OneStage) (desc : Claims.Msg) (q : Query) :
    u.val ∉ faulty H Cm (stmtOf P κ r desc q roots) (partOf P κ ns) (witDep C Cm H roots Xr) := by
  obtain ⟨op, hop⟩ := hright
  have hval : ∀ a ∈ P.io u, κ a = none →
      (witDep C Cm H roots Xr).val H Cm (stmtOf P κ r desc q roots) (posOf C a) = some (C.eval Xr a) := by
    intro a ha hk
    obtain ⟨h1, h3, h2⟩ := hop a ha
    obtain ⟨d, hd⟩ := htr a ha hk _ h1 h2
    exact witDep_val _ rfl a ⟨d, h3 ▸ hd⟩
  intro hmem
  simp only [faulty, Finset.mem_filter] at hmem
  apply hmem.2
  refine ⟨blockOf P κ u, blocks_partOf_getElem? P hns u, fun _ => false, ?_⟩
  change (progOf C κ).Correct ((progOf C κ).refIn (blockOf P κ u)
    ((witDep C Cm H roots Xr).val H Cm (stmtOf P κ r desc q roots)) (fun _ => false)) (blockOf P κ u)
  exact correct_blockOf_of P u hκin (C.eval Xr) (fun _ hu => C.eval_op Xr (isInput_of_unit hu)) hκ _ hval _

/-- **`FaultyRightTargetC`**, at any `canon` (`Bridge.Canon`): `faultyRightC`. -/
theorem faultyRightTargetC : FaultyRightTargetC := by
  intro C n P Pos Val Opn Com pos bit val com cs canon H Cm roots Xr κ u hκin hright htr hκ ns r desc q hns
  exact faultyRightC P pos bit val com cs canon H Cm roots Xr κ u hκin hright htr hκ hns r desc q

end Flock.Bridge
