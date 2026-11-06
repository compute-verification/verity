module

public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.SamplingGeneric.TwoCount
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
import Mathlib.Topology.Sheaves.Init

/-!
# The served-output cap on two-stage exfiltration

A set `T` of incorrect proof blocks changes only the served outputs that read its committed values, so what it serves
is bounded request by request. Split the served outputs into requests by any map `req`; request `r` serves `|r|` bits
(`servedBits`), and `T_r` is the proof blocks of `T` with a committed value one of `r`'s served outputs reads
(`blocksTo`), so that `T` touches `r` when `T_r` is nonempty. On fixed inputs, `T` serves `r` in at most
`2^{min(b(T_r), |r|)}` ways, and the influence set has at most `Σ_{T admissible} 2^{Σ_{r touched by T} min(b(T_r), |r|)}`
members. The split is the lemma's to take: the bound holds at every one, so it trusts none.

* `Prog.Reads`: gate `g` reads gate `j`, itself or through the wires it reads.
* `Prog.runIn_congr_reads`, `boundary_eq_reads`, `outs_eq_reads`: **the determinism lemma at one returned gate**: two
  traces that make every proof block outside `T` correct and agree at the inputs and at the committed values of `T`
  that a returned gate reads agree at it.
* `served_eq_on`, `card_image_servedAt_le`, `card_servedAt_le_served`: **one set `T`**: request `r`'s served outputs are
  fixed by `T_r`'s committed bits, and number at most `2^{|r|}`; the outputs `T` serves are at most the product over the
  requests.
* `card_influenceTwo_le_served`, `card_influenceTwo_le_served_min`: **the influence set** under the served-output cap,
  and under the smaller of it and `CompiledTwo.Exfil`'s `2^{b(T)}` for each `T` (a proof block read by several requests
  counts in each, so neither bound is always the smaller).
* `CompiledTwo.exfil_served`: **the corollary of `CompiledTwo.Exfil`**: KS at `RInflTwo`, with the influence set under
  the served-output cap.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

/-! ## What a gate reads -/

/-- **Gate `g` reads gate `j`**: `j` is `g`, or a gate that one of the wires `g` reads reads. -/
inductive Prog.Reads (C : Prog) : ℕ → ℕ → Prop
  | refl (g : ℕ) : Prog.Reads C g g
  | step {g i j : ℕ} {x : BGate} : C.gates[g]? = some x → i ∈ x.reads → Prog.Reads C i j → Prog.Reads C g j

theorem Prog.Reads.trans {C : Prog} {g i j : ℕ} (h₁ : C.Reads g i) (h₂ : C.Reads i j) : C.Reads g j := by
  revert h₂
  induction h₁ with
  | refl => exact id
  | step hx hi _ ih => exact fun h₂ => .step hx hi (ih h₂)

theorem Prog.Reads.read {C : Prog} {g i : ℕ} {x : BGate} (hx : C.gates[g]? = some x) (hi : i ∈ x.reads) :
    C.Reads g i :=
  .step hx hi (.refl i)

/-- In a topologically ordered program a gate reads only itself and earlier gates. -/
theorem Prog.Reads.le {C : Prog} (hC : C.Topological) {g j : ℕ} (h : C.Reads g j) : j ≤ g := by
  induction h with
  | refl => exact le_rfl
  | step hx hi _ ih => exact ih.trans (hC.1 _ _ hx _ hi).le

/-! ## The determinism lemma at one returned gate -/

/-- **`runIn` at `g` reads the inputs and outside reads of the gates `g` reads.** -/
theorem Prog.runIn_congr_reads (C : Prog) {ref₁ ref₂ : ℕ → Option Bool} {u : List ℕ} :
    ∀ g ∈ u, (∀ g' ∈ u, C.Reads g g' → C.gates[g']? = some .input → ref₁ g' = ref₂ g') →
      (∀ g' ∈ u, C.Reads g g' → ∀ x : BGate, C.gates[g']? = some x → ∀ i ∈ x.reads, i ∉ u →
        C.read ref₁ i = C.read ref₂ i) →
      C.runIn ref₁ u g = C.runIn ref₂ u g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hg hin hrd
  rw [Prog.runIn, Prog.runIn]
  cases hx : C.gates[g]? with
  | none => rfl
  | some x =>
    refine BGate.val_congr (fun i hi => ?_) fun h => hin g hg (.refl g) (hx.trans (congrArg some h))
    by_cases hiu : i ∈ u
    · rw [ite_eq_left hiu, ite_eq_left hiu]
      by_cases hig : i < g
      · rw [dite_eq_left hig, dite_eq_left hig]
        exact ih i hig hiu (fun g' hg' h' => hin g' hg' ((Prog.Reads.read hx hi).trans h'))
          fun g' hg' h' => hrd g' hg' ((Prog.Reads.read hx hi).trans h')
      · rw [dite_eq_right hig, dite_eq_right hig]
    · rw [ite_eq_right hiu, ite_eq_right hiu]
      exact hrd g hg (.refl g) x hx i hi hiu

/-- **The determinism lemma at the gates `o` reads**: two traces that make every proof block outside `T` correct, with
the same values at the input gates and the same bits at those committed values of `T`'s proof blocks that `o` reads,
have the same bits at every committed value `o` reads. -/
theorem boundary_eq_reads {C : Prog} {P : Partition} (hC : C.Topological) (hP : P.Partitions C)
    {X Y : ℕ → Option Bool} {T : Finset ℕ} {o : ℕ}
    (hX : ∀ v < P.blocks.length, v ∉ T → C.Correct X (P.blocks.getD v []))
    (hY : ∀ v < P.blocks.length, v ∉ T → C.Correct Y (P.blocks.getD v []))
    (hin : ∀ g, C.gates[g]? = some .input → X g = Y g)
    (hT : ∀ v ∈ T, ∀ g ∈ P.blocks.getD v [], C.Boundary (P.blocks.getD v []) g = true → C.Reads o g →
      (X g).getD false = (Y g).getD false) :
    ∀ g, C.Reads o g → ∀ v < P.blocks.length, g ∈ P.blocks.getD v [] → C.Boundary (P.blocks.getD v []) g = true →
      (X g).getD false = (Y g).getD false := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hog v hv hg hb
  by_cases hvT : v ∈ T
  · exact hT v hvT g hg hb hog
  have hvb : P.blocks[v]? = some (P.blocks.getD v []) := getElem?_getD hv
  obtain ⟨b, hXb, hXr⟩ := (hX v hv hvT g hg).2 hb
  obtain ⟨b', hYb, hYr⟩ := (hY v hv hvT g hg).2 hb
  have hrun : C.runIn X (P.blocks.getD v []) g = C.runIn Y (P.blocks.getD v []) g := by
    refine C.runIn_congr_reads g hg (fun g' _ _ h' => hin g' h') fun g' hg' hgg' x hx i hi hiu => ?_
    have hig : i < g := (hC.1 g' x hx i hi).trans_le (hgg'.le hC)
    have hoi : C.Reads o i := hog.trans (hgg'.trans (Prog.Reads.read hx hi))
    have hXn := (hX v hv hvT g' hg').1 x hx i hi hiu
    have hYn := (hY v hv hvT g' hg').1 x hx i hi hiu
    by_cases hc : ∃ c, C.gates[i]? = some (.const c)
    · obtain ⟨c, hc⟩ := hc
      simp only [Prog.read, hc]
    have hc' : ∀ c, C.gates[i]? ≠ some (.const c) := fun c h => hc ⟨c, h⟩
    rw [C.read_of_not_const X hc'] at hXn ⊢
    rw [C.read_of_not_const Y hc'] at hYn ⊢
    have hg'lt : g' < C.gates.length := hP.2.1 g' (mem_gates hvb hg')
    have hiP : i ∈ P.gates := hP.2.2.1 i ((hC.1 g' x hx i hi).trans hg'lt) hc'
      fun hu => hu.2.2 x (List.mem_of_getElem? hx) hi
    obtain ⟨v', hv', hiv'⟩ := exists_block hiP
    have hg'v' : g' ∉ P.blocks.getD v' [] := fun h' => by
      have := eq_of_mem_of_nodup hP.1 hvb (getElem?_getD hv') hg' h'
      subst this
      exact hiu hiv'
    exact eq_of_getD_eq hXn hYn (ih i hig hoi v' hv' hiv' (Prog.boundary_of_read hg'v' hg'lt hx hi))
  have hbb : b = b' := Option.some.inj (hXr.symm.trans (hrun.trans hYr))
  rw [hXb, hYb, hbb]

/-- **A returned gate's bits** agree under the determinism lemma's hypotheses at the gates it reads. -/
theorem outs_eq_reads {C : Prog} {P : Partition} (hC : C.Topological) (hP : P.Partitions C)
    {X Y : ℕ → Option Bool} {T : Finset ℕ} {o : ℕ} (ho : o ∈ C.outs)
    (hX : ∀ v < P.blocks.length, v ∉ T → C.Correct X (P.blocks.getD v []))
    (hY : ∀ v < P.blocks.length, v ∉ T → C.Correct Y (P.blocks.getD v []))
    (hin : ∀ g, C.gates[g]? = some .input → X g = Y g)
    (hT : ∀ v ∈ T, ∀ g ∈ P.blocks.getD v [], C.Boundary (P.blocks.getD v []) g = true → C.Reads o g →
      (X g).getD false = (Y g).getD false) :
    (C.read X o).getD false = (C.read Y o).getD false := by
  by_cases hc : ∃ c, C.gates[o]? = some (.const c)
  · obtain ⟨c, hc⟩ := hc
    simp only [Prog.read, hc]
  have hc' : ∀ c, C.gates[o]? ≠ some (.const c) := fun c h => hc ⟨c, h⟩
  rw [C.read_of_not_const X hc', C.read_of_not_const Y hc']
  obtain ⟨v, hv, hov⟩ := exists_block (hP.2.2.1 o (hC.2 o ho) hc' fun hu => hu.2.1 ho)
  exact boundary_eq_reads hC hP hX hY hin hT o (.refl o) v hv hov (Prog.boundary_of_out _ ho)

/-! ## One set of incorrect proof blocks, request by request -/

/-- **Request `r`'s served bits** `|r|` under a split `req` of the served outputs into requests: the served outputs it
holds. -/
def servedBits {R : Type} [DecidableEq R] {n : ℕ} (req : Fin n → R) (r : R) : ℕ :=
  (univ.filter fun i => req i = r).card

open Classical in
/-- **The proof blocks of `T` that request `r` reads** `T_r`: those with a committed value that one of `r`'s served
outputs reads. `T` touches `r` when `T_r` is nonempty. -/
noncomputable def blocksTo {R : Type} (C : Prog) (Pt : Partition) (req : Fin C.outs.length → R) (T : Finset ℕ)
    (r : R) : Finset ℕ :=
  T.filter fun v => ∃ g ∈ Pt.blocks.getD v [], C.Boundary (Pt.blocks.getD v []) g = true ∧
    ∃ i, req i = r ∧ C.Reads C.outs[i] g

/-- Request `r`'s served outputs of `y`. -/
def onReq {R : Type} {n : ℕ} (req : Fin n → R) (r : R) (y : Fin n → Bool) : {i // req i = r} → Bool :=
  fun i => y i.1

/-- Two boundary traces at interiors with the incorrect set `T`, the same inputs and the same bits at `T_r`'s committed
values serve request `r` the same outputs. -/
theorem served_eq_on (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) {R : Type} (req : Fin s.prog.outs.length → R) (r : R)
    {w w' : Wit} {ι ι' : ℕ → ℕ → Bool} {T : Finset ℕ}
    (hw : wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) = T)
    (hw' : wrongOf s.prog Pt (glue s.prog Pρ (w'.val H Cm s) ι') = T)
    (ha : inputsOf H Cm s w = inputsOf H Cm s w')
    (hT : ∀ g ∈ exits s.prog Pt (blocksTo s.prog Pt req T r),
      (glue s.prog Pρ (w.val H Cm s) ι g).getD false = (glue s.prog Pρ (w'.val H Cm s) ι' g).getD false) :
    ∀ i, req i = r → served H Cm s w i = served H Cm s w' i := by
  classical
  have hcor : ∀ Z : ℕ → Option Bool, wrongOf s.prog Pt Z = T →
      ∀ v < Pt.blocks.length, v ∉ T → s.prog.Correct Z (Pt.blocks.getD v []) := by
    intro Z hZ v hv hvT
    by_contra hc
    refine hvT (hZ ▸ ?_)
    unfold wrongOf
    exact mem_filter.2 ⟨mem_range.2 hv, hc⟩
  have hin : ∀ g, s.prog.gates[g]? = some .input →
      glue s.prog Pρ (w.val H Cm s) ι g = glue s.prog Pρ (w'.val H Cm s) ι' g := by
    intro g hg
    have h1 := congrFun ha g
    unfold inputsOf at h1
    rw [ite_eq_left hg, ite_eq_left hg] at h1
    rw [glue_of_boundary fun r => Prog.boundary_of_input r hg, glue_of_boundary fun r => Prog.boundary_of_input r hg]
    exact h1
  intro i hi
  have ho : s.prog.outs[i] ∈ s.prog.outs := List.getElem_mem i.isLt
  have hglue : ∀ (u : Wit) (κ : ℕ → ℕ → Bool), s.prog.read (u.val H Cm s) s.prog.outs[i] =
      s.prog.read (glue s.prog Pρ (u.val H Cm s) κ) s.prog.outs[i] := fun u κ =>
    s.prog.read_congr (glue_of_boundary fun r => Prog.boundary_of_out r ho).symm
  show (s.prog.read (w.val H Cm s) s.prog.outs[i]).getD false = (s.prog.read (w'.val H Cm s) s.prog.outs[i]).getD false
  rw [hglue w ι, hglue w' ι']
  refine outs_eq_reads hC hP ho (hcor _ hw) (hcor _ hw') hin fun v hv g hg hb hR => hT g ?_
  refine mem_biUnion.2 ⟨v, ?_, List.mem_toFinset.2 (List.mem_filter.2 ⟨hg, hb⟩)⟩
  unfold blocksTo
  exact mem_filter.2 ⟨hv, g, hg, hb, i, hi, hR⟩

/-- **A set `T` of incorrect proof blocks serves request `r` in at most `2^{min(b(T_r), |r|)}` ways** on fixed inputs:
`r`'s served outputs are fixed by `T_r`'s committed bits, and there are `|r|` of them. -/
theorem card_image_servedAt_le (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) (a : ℕ → Option Bool) (T : Finset ℕ) {R : Type}
    [DecidableEq R] (req : Fin s.prog.outs.length → R) (r : R) :
    ((servedAt H Cm s Pt Pρ a T).image (onReq req r)).card ≤
      2 ^ min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := by
  classical
  have hA : ((servedAt H Cm s Pt Pρ a T).image (onReq req r)).card ≤
      2 ^ (exits s.prog Pt (blocksTo s.prog Pt req T r)).card := by
    let f : (Fin s.prog.outs.length → Bool) → (exits s.prog Pt (blocksTo s.prog Pt req T r) → Bool) := fun y =>
      if h : ∃ (w : Wit) (ι : ℕ → ℕ → Bool), inputsOf H Cm s w = a ∧
          wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) = T ∧ served H Cm s w = y then
        fun g => (glue s.prog Pρ ((Classical.choose h).val H Cm s) (Classical.choose (Classical.choose_spec h)) g).getD
          false
      else fun _ => false
    let F : ({i // req i = r} → Bool) → (exits s.prog Pt (blocksTo s.prog Pt req T r) → Bool) := fun z =>
      if h : ∃ y ∈ servedAt H Cm s Pt Pρ a T, onReq req r y = z then f (Classical.choose h) else fun _ => false
    calc ((servedAt H Cm s Pt Pρ a T).image (onReq req r)).card
        ≤ (univ : Finset (exits s.prog Pt (blocksTo s.prog Pt req T r) → Bool)).card := by
          refine card_le_card_of_injOn F (fun _ _ => mem_coe.2 (mem_univ _)) ?_
          intro z hz z' hz' hF
          have h := mem_image.1 hz
          have h' := mem_image.1 hz'
          simp only [F, h, h', ↓reduceDIte] at hF
          obtain ⟨hy, hyz⟩ := Classical.choose_spec h
          obtain ⟨hy', hyz'⟩ := Classical.choose_spec h'
          have hs := (mem_filter.1 hy).2
          have hs' := (mem_filter.1 hy').2
          simp only [f, hs, hs', ↓reduceDIte] at hF
          obtain ⟨ha, hw, hsw⟩ := Classical.choose_spec (Classical.choose_spec hs)
          obtain ⟨ha', hw', hsw'⟩ := Classical.choose_spec (Classical.choose_spec hs')
          rw [← hyz, ← hyz']
          funext j
          simp only [onReq]
          rw [← hsw, ← hsw']
          exact served_eq_on H Cm hC hP req r hw hw' (ha.trans ha'.symm) (fun g hg => congrFun hF ⟨g, hg⟩) j.1 j.2
      _ = 2 ^ (exits s.prog Pt (blocksTo s.prog Pt req T r)).card := by
          rw [card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]
  have hc : Fintype.card {i // req i = r} = servedBits req r :=
    Fintype.card_of_subtype (univ.filter fun i => req i = r) fun i => by simp
  have hB : ((servedAt H Cm s Pt Pρ a T).image (onReq req r)).card ≤ 2 ^ servedBits req r :=
    (card_le_univ _).trans (le_of_eq (by rw [Fintype.card_fun, Fintype.card_bool, hc]))
  rcases le_total (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) with h | h
  · rw [min_eq_left h]
    exact hA.trans (Nat.pow_le_pow_right (by norm_num) (card_exits_le s Pt _))
  · rw [min_eq_right h]
    exact hB

/-- **A set `T` of incorrect proof blocks serves at most `2^{Σ_r min(b(T_r), |r|)}` outputs** on fixed inputs, at any
split `req` of the served outputs into requests: an output is its requests' served outputs. -/
theorem card_servedAt_le_served (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) (a : ℕ → Option Bool) (T : Finset ℕ) {R : Type}
    [Fintype R] [DecidableEq R] (req : Fin s.prog.outs.length → R) :
    (servedAt H Cm s Pt Pρ a T).card ≤
      2 ^ ∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := by
  calc (servedAt H Cm s Pt Pρ a T).card
      ≤ (Fintype.piFinset fun r => (servedAt H Cm s Pt Pρ a T).image (onReq req r)).card := by
        refine card_le_card_of_injOn (fun y r => onReq req r y)
          (fun y hy => mem_coe.2 (Fintype.mem_piFinset.2 fun r => mem_image_of_mem _ hy)) ?_
        intro y _ y' _ h
        funext i
        exact congrFun (congrFun h (req i)) ⟨i, rfl⟩
    _ = ∏ r, ((servedAt H Cm s Pt Pρ a T).image (onReq req r)).card := Fintype.card_piFinset _
    _ ≤ ∏ r, 2 ^ min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) :=
        prod_le_prod fun r _ => card_image_servedAt_le H Cm hC hP a T req r
    _ = 2 ^ ∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := prod_pow_eq_pow_sum _ _ _

/-- A request `T` doesn't touch adds nothing: `T_r` is empty. -/
theorem sum_touched {R : Type} [Fintype R] [DecidableEq R] (s : AuditStatement) (Pt : Partition)
    (req : Fin s.prog.outs.length → R) (T : Finset ℕ) :
    ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
        min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) =
      ∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := by
  refine sum_filter_of_ne fun r _ h => ?_
  by_contra hne
  rw [not_nonempty_iff_eq_empty] at hne
  rw [hne] at h
  simp [bitsOf] at h

/-! ## The influence set -/

/-- **The served-output cap**: on a statement in scope, at any input values and any split `req` of the served outputs
into requests, the boundary traces `RTwo` takes at `δ` serve at most `Σ_{T admissible} 2^{Σ_{r touched by T}
min(b(T_r), |r|)}` outputs: each request touched by a set of incorrect proof blocks is charged the smaller of the bits
those of them it reads commit and its own served bits. -/
theorem card_influenceTwo_le_served (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {R : Type} [Fintype R] [DecidableEq R]
    (req : Fin s.prog.outs.length → R) :
    (influenceTwo H Cm s Pt d δ a).card ≤
      ∑ T ∈ admissibleTwo Pt d δ, 2 ^ ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
        min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := by
  classical
  have hsub : influenceTwo H Cm s Pt d δ a ⊆ (admissibleTwo Pt d δ).biUnion (servedAt H Cm s Pt d.part a) := by
    intro y hy
    unfold influenceTwo at hy
    obtain ⟨w, ha, ⟨ι, hM⟩, hwy⟩ := (mem_filter.1 hy).2
    refine mem_biUnion.2 ⟨_, wrongOf_mem_admissible H Cm hs.2.1 hs.2.2.1 hM, ?_⟩
    unfold servedAt
    exact mem_filter.2 ⟨mem_univ _, w, ι, ha, rfl, hwy⟩
  calc (influenceTwo H Cm s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, (servedAt H Cm s Pt d.part a T).card :=
        (card_le_card hsub).trans card_biUnion_le
    _ ≤ _ := sum_le_sum fun T _ => by
        rw [sum_touched s Pt req T]
        exact card_servedAt_le_served H Cm hs.1 hs.2.1 a T req

/-- **The served-output cap beside the committed bits**: for each admissible `T`, the smaller of `b(T)` and
`Σ_{r touched by T} min(b(T_r), |r|)`. -/
theorem card_influenceTwo_le_served_min (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) :
    (influenceTwo H Cm s Pt d δ a).card ≤
      ∑ T ∈ admissibleTwo Pt d δ, 2 ^ min (bitsOf s Pt T)
        (∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
          min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)) := by
  classical
  have hsub : influenceTwo H Cm s Pt d δ a ⊆ (admissibleTwo Pt d δ).biUnion (servedAt H Cm s Pt d.part a) := by
    intro y hy
    unfold influenceTwo at hy
    obtain ⟨w, ha, ⟨ι, hM⟩, hwy⟩ := (mem_filter.1 hy).2
    refine mem_biUnion.2 ⟨_, wrongOf_mem_admissible H Cm hs.2.1 hs.2.2.1 hM, ?_⟩
    unfold servedAt
    exact mem_filter.2 ⟨mem_univ _, w, ι, ha, rfl, hwy⟩
  calc (influenceTwo H Cm s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, (servedAt H Cm s Pt d.part a T).card :=
        (card_le_card hsub).trans card_biUnion_le
    _ ≤ _ := sum_le_sum fun T _ => by
        rw [sum_touched s Pt req T]
        rcases le_total (bitsOf s Pt T)
            (∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)) with h | h
        · rw [min_eq_left h]
          exact (card_servedAt_le H Cm hs.1 hs.2.1 a T).trans
            (Nat.pow_le_pow_right (by norm_num) (card_exits_le s Pt T))
        · rw [min_eq_right h]
          exact card_servedAt_le_served H Cm hs.1 hs.2.1 a T req

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (InScopeTwo RInflTwo influenceTwo admissibleTwo bitsOf blocksTo servedBits KSLinOn Bext
  card_influenceTwo_le_served)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The served-output cap, a corollary of `Exfil`**: at every `δ ≥ 0`, KS at `RInflTwo` at `Exfil`'s extractor and
errors, and on a statement in scope, at any input values and any split `req` of its served outputs into requests, the
influence set has at most `Σ_{T admissible} 2^{Σ_{r touched by T} min(b(T_r), |r|)}` members. -/
theorem exfil_served {H : Msg → Msg} {L : CompiledTwo Coin A} (hL : L.Exfil H) (δ : ℝ) (hδ : 0 ≤ δ) :
    KSLinOn H L.V L.Scope (RInflTwo H L.commit L.parse L.partition L.two δ) L.E (L.κ δ) L.γ (fun _ => 1)
        (Bext H L.V L.E) ∧
      ∀ (x : Msg) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
        (d : Claims.Sampling.Design.Two) (a : ℕ → Option Bool) (R : Type) [Fintype R] [DecidableEq R]
        (req : Fin s.prog.outs.length → R),
        L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
        (influenceTwo H L.commit s Pt d δ a).card ≤
          ∑ T ∈ admissibleTwo Pt d δ, 2 ^ ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
            min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := by
  refine ⟨(hL δ hδ).1, ?_⟩
  intro x s Pt d a R _ _ req _ _ _ hin
  exact card_influenceTwo_le_served H L.commit hin δ a req

end CompiledTwo

end Claims.SamplingGeneric
