module

public import Definitions.Claims.SamplingTwo
public import Definitions.Claims.Guarantees.SamplingTwo
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
members. It is a set of served outputs, so it also has at most `2^{|outs|}`, and the bound to charge is the smaller.
The split is the lemma's to take: the bound holds at every one, so it trusts none.

Counting by sets charges each set of a request's proof blocks on its own, so a wholly incorrect request costs which of
its blocks are incorrect as well as its `|r|` bits. Counting by patterns removes that. Sets that give `r` one `T_r`
serve it alike, whatever else is incorrect, so the traces of the admissible sets of one pattern `r ↦ cls r (T_r)`
serve at most `∏_r min(Σ_{U of class p r} 2^{b(U)}, 2^{|r|})` outputs, at any classes `cls`. With `lightOf`'s
classes, a request whose `T_r` commits at least `|r|` bits costs `|r|` and its blocks are forgotten, and the count is
never above the sets'.

* `Prog.Reads.trans`, `Prog.Reads.read`, `Prog.Reads.le`: gate `g` reads gate `j` (`Prog.Reads`, with `servedBits`,
  `blocksTo`, `lightOf` and `lightBits` in `Definitions.Claims.SamplingServed`), itself or through the wires it reads.
* `Prog.runIn_congr_reads`, `boundary_eq_reads`, `outs_eq_reads`: **the determinism lemma at one returned gate**: two
  traces that make every proof block outside `T` correct and agree at the inputs and at the committed values of `T`
  that a returned gate reads agree at it.
* `served_eq_on`, `blocksTo_union`, `served_eq_on_of_mem`, `card_image_onReq_le`: **request by request**: request
  `r`'s served outputs are fixed by the committed bits of `S_r`, for any `S` holding both traces' incorrect sets, so
  the traces of sets that all give `r` one `U` serve it in at most `2^{b(U)}` ways.
* `card_le_prod_onReq`, `card_image_onReq_le_served`, `card_image_servedAt_le`, `card_servedAt_le_served`: **one set
  `T`**: it serves `r` in at most `2^{min(b(T_r), |r|)}` ways, and the outputs it serves are at most the product over
  the requests.
* `card_influenceTwo_le_served`, `card_influenceTwo_le_served_min`: **the influence set** under the served-output cap,
  and under the smaller of it and `CompiledTwo.Exfil`'s `2^{b(T)}` for each `T` (a proof block read by several requests
  counts in each, so neither bound is always the smaller).
* `sum_servedBits`, `card_influenceTwo_le_two_pow_outs`, `card_influenceTwo_le_min`: **never above the served
  outputs**: the influence set is a set of served outputs, so it has at most `2^{|outs|}` members, `|outs| = Σ_r |r|`,
  and the bound to charge is the smaller of that and the served-output cap.
* `card_influenceTwo_le_classes`, `card_influenceTwo_le_touched`: **patterns**: the influence set counted by the
  admissible sets' patterns under any classes, and by which requests they touch.
* `lightBits_lightOf`, `card_influenceTwo_le_patterns`, `sum_patterns_le_served`: **the patterns' form** at `lightOf`'s
  classes: a request whose `T_r` commits at least `|r|` bits costs `|r|`, with its blocks forgotten; never above the
  sets' count. The lock lists it as `Claims.SecurityProofs.InfluenceTwoPatterns`.
* `CompiledTwo.exfil_served`: **the corollary of `CompiledTwo.Exfil`**: KS at `RInflTwo`, with the influence set under
  the smaller of the served-output cap and `2^{|outs|}`.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

/-! ## What a gate reads -/

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

/-- Request `r`'s served outputs of `y`. -/
def onReq {R : Type} {n : ℕ} (req : Fin n → R) (r : R) (y : Fin n → Bool) : {i // req i = r} → Bool :=
  fun i => y i.1

/-- Two boundary traces at interiors whose incorrect sets lie in `S`, with the same inputs and the same bits at `S_r`'s
committed values, serve request `r` the same outputs. -/
theorem served_eq_on (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) {R : Type} (req : Fin s.prog.outs.length → R) (r : R)
    {w w' : Wit} {ι ι' : ℕ → ℕ → Bool} {S : Finset ℕ}
    (hw : wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) ⊆ S)
    (hw' : wrongOf s.prog Pt (glue s.prog Pρ (w'.val H Cm s) ι') ⊆ S)
    (ha : inputsOf H Cm s w = inputsOf H Cm s w')
    (hT : ∀ g ∈ exits s.prog Pt (blocksTo s.prog Pt req S r),
      (glue s.prog Pρ (w.val H Cm s) ι g).getD false = (glue s.prog Pρ (w'.val H Cm s) ι' g).getD false) :
    ∀ i, req i = r → served H Cm s w i = served H Cm s w' i := by
  classical
  have hcor : ∀ Z : ℕ → Option Bool, wrongOf s.prog Pt Z ⊆ S →
      ∀ v < Pt.blocks.length, v ∉ S → s.prog.Correct Z (Pt.blocks.getD v []) := by
    intro Z hZ v hv hvS
    by_contra hc
    refine hvS (hZ ?_)
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

/-- The proof blocks of `T ∪ T'` that request `r` reads are `T_r ∪ T'_r`. -/
theorem blocksTo_union {R : Type} (C : Prog) (Pt : Partition) (req : Fin C.outs.length → R) (T T' : Finset ℕ)
    (r : R) : blocksTo C Pt req (T ∪ T') r = blocksTo C Pt req T r ∪ blocksTo C Pt req T' r := by
  ext v
  unfold blocksTo
  simp only [mem_filter, mem_union, or_and_right]

/-- Two boundary traces whose incorrect sets are in `𝒯`, each with `T_r = U`, the same inputs and the same bits at
`U`'s committed values serve request `r` the same outputs: both are correct outside the union of their sets. -/
theorem served_eq_on_of_mem (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) {R : Type} (req : Fin s.prog.outs.length → R) (r : R)
    {𝒯 : Finset (Finset ℕ)} {U : Finset ℕ} (hU : ∀ T ∈ 𝒯, blocksTo s.prog Pt req T r = U)
    {w w' : Wit} {ι ι' : ℕ → ℕ → Bool}
    (hw : wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) ∈ 𝒯)
    (hw' : wrongOf s.prog Pt (glue s.prog Pρ (w'.val H Cm s) ι') ∈ 𝒯)
    (ha : inputsOf H Cm s w = inputsOf H Cm s w')
    (hT : ∀ g ∈ exits s.prog Pt U,
      (glue s.prog Pρ (w.val H Cm s) ι g).getD false = (glue s.prog Pρ (w'.val H Cm s) ι' g).getD false) :
    ∀ i, req i = r → served H Cm s w i = served H Cm s w' i := by
  have hS := blocksTo_union s.prog Pt req (wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι))
    (wrongOf s.prog Pt (glue s.prog Pρ (w'.val H Cm s) ι')) r
  rw [hU _ hw, hU _ hw', union_self] at hS
  refine served_eq_on H Cm hC hP req r subset_union_left subset_union_right ha fun g hg => hT g ?_
  rwa [hS] at hg

/-- A set of served outputs has at most the product over the requests of its requests' outputs. -/
theorem card_le_prod_onReq {R : Type} [Fintype R] [DecidableEq R] {n : ℕ} (req : Fin n → R)
    (Y : Finset (Fin n → Bool)) : Y.card ≤ ∏ r, (Y.image (onReq req r)).card :=
  calc Y.card ≤ (Fintype.piFinset fun r => Y.image (onReq req r)).card := by
        refine card_le_card_of_injOn (fun y r => onReq req r y)
          (fun y hy => mem_coe.2 (Fintype.mem_piFinset.2 fun r => mem_image_of_mem _ hy)) ?_
        intro y _ y' _ h
        funext i
        exact congrFun (congrFun h (req i)) ⟨i, rfl⟩
    _ = ∏ r, (Y.image (onReq req r)).card := Fintype.card_piFinset _

/-- Request `r` has at most `2^{|r|}` served outputs. -/
theorem card_image_onReq_le_served {R : Type} [DecidableEq R] {n : ℕ} (req : Fin n → R) (r : R)
    (Y : Finset (Fin n → Bool)) : (Y.image (onReq req r)).card ≤ 2 ^ servedBits req r := by
  have hc : Fintype.card {i // req i = r} = servedBits req r :=
    Fintype.card_of_subtype (univ.filter fun i => req i = r) fun i => by simp
  exact (card_le_univ _).trans (le_of_eq (by rw [Fintype.card_fun, Fintype.card_bool, hc]))

/-- **Sets that request `r` reads alike serve it alike**: on fixed inputs, the traces whose incorrect set is one of `𝒯`,
each with `T_r = U`, serve request `r` at most `2^{b(U)}` outputs. Two of them are correct outside the union of their
sets, which `r` reads only at `U`. -/
theorem card_image_onReq_le (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) (a : ℕ → Option Bool) {R : Type} [DecidableEq R]
    (req : Fin s.prog.outs.length → R) (r : R) (𝒯 : Finset (Finset ℕ)) (U : Finset ℕ)
    (hU : ∀ T ∈ 𝒯, blocksTo s.prog Pt req T r = U) :
    ((𝒯.biUnion (servedAt H Cm s Pt Pρ a)).image (onReq req r)).card ≤ 2 ^ bitsOf s Pt U := by
  classical
  have hspec : ∀ y ∈ 𝒯.biUnion (servedAt H Cm s Pt Pρ a), ∃ (w : Wit) (ι : ℕ → ℕ → Bool), inputsOf H Cm s w = a ∧
      wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) ∈ 𝒯 ∧ served H Cm s w = y := by
    intro y hy
    obtain ⟨T, hT, hyT⟩ := mem_biUnion.1 hy
    obtain ⟨w, ι, ha, hw, hwy⟩ := (mem_filter.1 hyT).2
    exact ⟨w, ι, ha, hw ▸ hT, hwy⟩
  let f : (Fin s.prog.outs.length → Bool) → (exits s.prog Pt U → Bool) := fun y =>
    if h : ∃ (w : Wit) (ι : ℕ → ℕ → Bool), inputsOf H Cm s w = a ∧
        wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) ∈ 𝒯 ∧ served H Cm s w = y then
      fun g => (glue s.prog Pρ ((Classical.choose h).val H Cm s) (Classical.choose (Classical.choose_spec h)) g).getD
        false
    else fun _ => false
  let F : ({i // req i = r} → Bool) → (exits s.prog Pt U → Bool) := fun z =>
    if h : ∃ y ∈ 𝒯.biUnion (servedAt H Cm s Pt Pρ a), onReq req r y = z then f (Classical.choose h)
    else fun _ => false
  calc ((𝒯.biUnion (servedAt H Cm s Pt Pρ a)).image (onReq req r)).card
      ≤ (univ : Finset (exits s.prog Pt U → Bool)).card := by
        refine card_le_card_of_injOn F (fun _ _ => mem_coe.2 (mem_univ _)) ?_
        intro z hz z' hz' hF
        have h := mem_image.1 hz
        have h' := mem_image.1 hz'
        simp only [F, h, h', ↓reduceDIte] at hF
        obtain ⟨hy, hyz⟩ := Classical.choose_spec h
        obtain ⟨hy', hyz'⟩ := Classical.choose_spec h'
        have hs := hspec _ hy
        have hs' := hspec _ hy'
        simp only [f, hs, hs', ↓reduceDIte] at hF
        obtain ⟨ha, hw, hsw⟩ := Classical.choose_spec (Classical.choose_spec hs)
        obtain ⟨ha', hw', hsw'⟩ := Classical.choose_spec (Classical.choose_spec hs')
        rw [← hyz, ← hyz']
        funext j
        simp only [onReq]
        rw [← hsw, ← hsw']
        exact served_eq_on_of_mem H Cm hC hP req r hU hw hw' (ha.trans ha'.symm) (fun g hg => congrFun hF ⟨g, hg⟩)
          j.1 j.2
    _ = 2 ^ (exits s.prog Pt U).card := by rw [card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]
    _ ≤ 2 ^ bitsOf s Pt U := Nat.pow_le_pow_right (by norm_num) (card_exits_le s Pt U)

/-- **A set `T` of incorrect proof blocks serves request `r` in at most `2^{min(b(T_r), |r|)}` ways** on fixed inputs:
`r`'s served outputs are fixed by `T_r`'s committed bits, and there are `|r|` of them. -/
theorem card_image_servedAt_le (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) (a : ℕ → Option Bool) (T : Finset ℕ) {R : Type}
    [DecidableEq R] (req : Fin s.prog.outs.length → R) (r : R) :
    ((servedAt H Cm s Pt Pρ a T).image (onReq req r)).card ≤
      2 ^ min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := by
  rcases le_total (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) with h | h
  · rw [min_eq_left h]
    have hT := card_image_onReq_le (Pρ := Pρ) H Cm hC hP a req r {T} (blocksTo s.prog Pt req T r) fun T' hT' => by
      rw [mem_singleton.1 hT']
    rwa [singleton_biUnion] at hT
  · rw [min_eq_right h]
    exact card_image_onReq_le_served req r _

/-- **A set `T` of incorrect proof blocks serves at most `2^{Σ_r min(b(T_r), |r|)}` outputs** on fixed inputs, at any
split `req` of the served outputs into requests: an output is its requests' served outputs. -/
theorem card_servedAt_le_served (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) (a : ℕ → Option Bool) (T : Finset ℕ) {R : Type}
    [Fintype R] [DecidableEq R] (req : Fin s.prog.outs.length → R) :
    (servedAt H Cm s Pt Pρ a T).card ≤
      2 ^ ∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) :=
  calc (servedAt H Cm s Pt Pρ a T).card ≤ ∏ r, ((servedAt H Cm s Pt Pρ a T).image (onReq req r)).card :=
        card_le_prod_onReq req _
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

/-- **The requests' served bits add up to the served outputs**: each served output is in exactly one request. -/
theorem sum_servedBits {R : Type} [Fintype R] [DecidableEq R] {n : ℕ} (req : Fin n → R) :
    ∑ r, servedBits req r = n := by
  have h := card_eq_sum_card_fiberwise (f := req) (s := (univ : Finset (Fin n))) (t := (univ : Finset R))
    fun _ _ => mem_coe.2 (mem_univ _)
  rw [card_univ, Fintype.card_fin] at h
  exact h.symm

/-- **The influence set is a set of served outputs**, so it has at most `2^{|outs|}` members, on any statement, at any
`δ` and any input values. -/
theorem card_influenceTwo_le_two_pow_outs (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement)
    (Pt : Partition) (d : Design.Two) (δ : ℝ) (a : ℕ → Option Bool) :
    (influenceTwo H Cm s Pt d δ a).card ≤ 2 ^ s.prog.outs.length :=
  (card_le_univ _).trans (le_of_eq (by rw [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]))

/-- **The served-output cap, never above the served outputs**: at any split `req`, the influence set has at most the
smaller of `Σ_{T admissible} 2^{min(b(T), Σ_{r touched by T} min(b(T_r), |r|))}` and `2^{Σ_r |r|} = 2^{|outs|}`. -/
theorem card_influenceTwo_le_min (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {R : Type} [Fintype R] [DecidableEq R]
    (req : Fin s.prog.outs.length → R) :
    (influenceTwo H Cm s Pt d δ a).card ≤
      min (∑ T ∈ admissibleTwo Pt d δ, 2 ^ min (bitsOf s Pt T)
          (∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
            min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)))
        (2 ^ ∑ r, servedBits req r) := by
  rw [sum_servedBits]
  exact le_min (card_influenceTwo_le_served_min H Cm hs δ a req) (card_influenceTwo_le_two_pow_outs H Cm s Pt d δ a)

/-! ## Patterns: what each request is charged, counted once -/

/-- **The influence set by patterns**: on a statement in scope, at any input values, any split `req` of the served
outputs into requests and any classes `cls r` of the sets of proof blocks request `r` reads, the influence set has at
most `Σ_p ∏_r min(Σ_{U ∈ 𝒰_r, cls r U = p r} 2^{b(U)}, 2^{|r|})` members. `p` ranges over the admissible sets' patterns
`r ↦ cls r (T_r)`, and `𝒰_r` over the sets `T_r` they give `r`: the traces of a pattern serve `r` only what the sets of
its class serve it, and no more than `|r|` bits. -/
theorem card_influenceTwo_le_classes (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) {K : Type} [DecidableEq K] (cls : R → Finset ℕ → K) :
    (influenceTwo H Cm s Pt d δ a).card ≤
      ∑ p ∈ (admissibleTwo Pt d δ).image (fun T r => cls r (blocksTo s.prog Pt req T r)),
        ∏ r, min (∑ U ∈ ((admissibleTwo Pt d δ).image fun T => blocksTo s.prog Pt req T r).filter
            (fun U => cls r U = p r), 2 ^ bitsOf s Pt U) (2 ^ servedBits req r) := by
  classical
  have hsub : influenceTwo H Cm s Pt d δ a ⊆
      ((admissibleTwo Pt d δ).image (fun T r => cls r (blocksTo s.prog Pt req T r))).biUnion fun p =>
        ((admissibleTwo Pt d δ).filter fun T => (fun r => cls r (blocksTo s.prog Pt req T r)) = p).biUnion
          (servedAt H Cm s Pt d.part a) := by
    intro y hy
    unfold influenceTwo at hy
    obtain ⟨w, ha, ⟨ι, hM⟩, hwy⟩ := (mem_filter.1 hy).2
    have hT := wrongOf_mem_admissible H Cm hs.2.1 hs.2.2.1 hM
    refine mem_biUnion.2 ⟨_, mem_image_of_mem _ hT, mem_biUnion.2 ⟨_, mem_filter.2 ⟨hT, rfl⟩, ?_⟩⟩
    unfold servedAt
    exact mem_filter.2 ⟨mem_univ _, w, ι, ha, rfl, hwy⟩
  refine (card_le_card hsub).trans (card_biUnion_le.trans (sum_le_sum fun p _ => ?_))
  refine (card_le_prod_onReq req _).trans (prod_le_prod fun r _ => le_min ?_ (card_image_onReq_le_served req r _))
  refine (card_le_card (t := (((admissibleTwo Pt d δ).image fun T => blocksTo s.prog Pt req T r).filter
      fun U => cls r U = p r).biUnion fun U => (((admissibleTwo Pt d δ).filter fun T =>
        blocksTo s.prog Pt req T r = U).biUnion (servedAt H Cm s Pt d.part a)).image (onReq req r)) ?_).trans
    (card_biUnion_le.trans (sum_le_sum fun U _ =>
      card_image_onReq_le H Cm hs.1 hs.2.1 a req r _ U fun T hT => (mem_filter.1 hT).2))
  intro z hz
  obtain ⟨y, hy, rfl⟩ := mem_image.1 hz
  obtain ⟨T, hT, hyT⟩ := mem_biUnion.1 hy
  obtain ⟨hTA, hTp⟩ := mem_filter.1 hT
  exact mem_biUnion.2 ⟨blocksTo s.prog Pt req T r, mem_filter.2 ⟨mem_image_of_mem _ hTA, congrFun hTp r⟩,
    mem_image_of_mem _ (mem_biUnion.2 ⟨T, mem_filter.2 ⟨hTA, rfl⟩, hyT⟩)⟩

/-- **Count which requests are touched, not which proof blocks**: at any split `req`, the influence set has at most
`Σ_Q ∏_{r ∈ Q} min(Σ_{U ∈ 𝒰_r, U ≠ ∅} 2^{b(U)}, 2^{|r|})` members, `Q` over the sets of requests the admissible sets
touch and `𝒰_r` over the sets `T_r` they give `r`. A request a set doesn't touch serves its honest outputs. -/
theorem card_influenceTwo_le_touched (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) :
    (influenceTwo H Cm s Pt d δ a).card ≤
      ∑ Q ∈ (admissibleTwo Pt d δ).image (fun T => univ.filter fun r => (blocksTo s.prog Pt req T r).Nonempty),
        ∏ r ∈ Q, min (∑ U ∈ ((admissibleTwo Pt d δ).image fun T => blocksTo s.prog Pt req T r).filter
            (fun U => U.Nonempty), 2 ^ bitsOf s Pt U) (2 ^ servedBits req r) := by
  classical
  refine (card_influenceTwo_le_classes H Cm hs δ a req fun _ U => decide U.Nonempty).trans ?_
  have himg : ((admissibleTwo Pt d δ).image fun T r => decide (blocksTo s.prog Pt req T r).Nonempty) =
      ((admissibleTwo Pt d δ).image fun T => univ.filter fun r => (blocksTo s.prog Pt req T r).Nonempty).image
        fun Q r => decide (r ∈ Q) := by
    rw [image_image]
    congr 1
    funext T r
    simp
  rw [himg, sum_image]
  · refine sum_le_sum fun Q _ => ?_
    beta_reduce
    calc _ ≤ ∏ r, if r ∈ Q then min (∑ U ∈ ((admissibleTwo Pt d δ).image fun T => blocksTo s.prog Pt req T r).filter
              (fun U => U.Nonempty), 2 ^ bitsOf s Pt U) (2 ^ servedBits req r) else 1 := by
          refine prod_le_prod fun r _ => ?_
          by_cases hr : r ∈ Q
          · rw [ite_eq_left hr]
            refine le_min ((min_le_left _ _).trans (le_of_eq (sum_congr (filter_congr fun U _ => ?_) fun _ _ => rfl)))
              (min_le_right _ _)
            simp [hr]
          · rw [ite_eq_right hr]
            refine (min_le_left _ _).trans ((sum_le_sum_of_subset (t := {∅}) fun U hU => ?_).trans (by simp [bitsOf]))
            have h : U = ∅ := by simpa [hr] using (mem_filter.1 hU).2
            exact mem_singleton.2 h
      _ = _ := by rw [prod_ite_mem, univ_inter]
  · intro Q _ Q' _ h
    ext r
    simpa using congrFun h r

theorem lightOf_eq_none {s : AuditStatement} {Pt : Partition} {n : ℕ} {U : Finset ℕ} (h : lightOf s Pt n U = none) :
    U = ∅ := by
  unfold lightOf at h
  split_ifs at h with h₁
  exact h₁

theorem lightOf_eq_some_some {s : AuditStatement} {Pt : Partition} {n : ℕ} {U U₀ : Finset ℕ}
    (h : lightOf s Pt n U = some (some U₀)) : U = U₀ := by
  unfold lightOf at h
  split_ifs at h with h₁ h₂ <;> first | exact Option.some.inj (Option.some.inj h) | cases h

/-- A class's bits are `min(b(U), n)`. -/
theorem lightBits_lightOf (s : AuditStatement) (Pt : Partition) (n : ℕ) (U : Finset ℕ) :
    lightBits s Pt n (lightOf s Pt n U) = min (bitsOf s Pt U) n := by
  unfold lightOf
  split_ifs with h₁ h₂
  · subst h₁
    simp [lightBits, bitsOf]
  · simp [lightBits, min_eq_right h₂]
  · simp [lightBits, min_eq_left (le_of_lt (not_le.1 h₂))]

/-- **The patterns' form**: at any split `req`, the influence set has at most `Σ_p 2^{Σ_r bits(p r)}` members, `p` over
the admissible sets' patterns `r ↦ lightOf |r| T_r`. A request `T` doesn't touch costs nothing. One whose `T_r` commits
at least `|r|` bits costs `|r|`, and its pattern forgets which of its blocks are incorrect. Any other costs `b(T_r)`.
The exponent is the capped sets' `Σ_{r touched by T} min(b(T_r), |r|)` at every `T` of the pattern
(`lightBits_lightOf`), so the bound is never above `card_influenceTwo_le_served`'s (`sum_patterns_le_served`). -/
theorem card_influenceTwo_le_patterns (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) :
    (influenceTwo H Cm s Pt d δ a).card ≤
      ∑ p ∈ (admissibleTwo Pt d δ).image (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
        2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) := by
  classical
  refine (card_influenceTwo_le_classes H Cm hs δ a req fun r U => lightOf s Pt (servedBits req r) U).trans ?_
  refine sum_le_sum fun p _ => ?_
  rw [← prod_pow_eq_pow_sum]
  refine prod_le_prod fun r _ => ?_
  match hp : p r with
  | none =>
    refine (min_le_left _ _).trans
      ((sum_le_sum_of_subset (t := {∅}) fun U hU => ?_).trans (by simp [bitsOf, lightBits]))
    exact mem_singleton.2 (lightOf_eq_none (mem_filter.1 hU).2)
  | some none => exact min_le_right _ _
  | some (some U₀) =>
    refine (min_le_left _ _).trans ((sum_le_sum_of_subset (t := {U₀}) fun U hU => ?_).trans (by simp [lightBits]))
    exact mem_singleton.2 (lightOf_eq_some_some (mem_filter.1 hU).2)

/-- **The patterns' form is never above the capped sets'**: a pattern's exponent is that of each of its sets. -/
theorem sum_patterns_le_served {R : Type} [Fintype R] [DecidableEq R] (s : AuditStatement) (Pt : Partition)
    (d : Design.Two) (δ : ℝ) (req : Fin s.prog.outs.length → R) :
    ∑ p ∈ (admissibleTwo Pt d δ).image (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
        2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) ≤
      ∑ T ∈ admissibleTwo Pt d δ, 2 ^ ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
        min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) := by
  refine (sum_image_le_of_nonneg fun _ _ => Nat.zero_le _).trans (le_of_eq (sum_congr rfl fun T _ => ?_))
  beta_reduce
  rw [sum_touched s Pt req T]
  exact congrArg (2 ^ ·) (sum_congr rfl fun r _ => lightBits_lightOf s Pt _ _)

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (InScopeTwo RInflTwo influenceTwo admissibleTwo bitsOf blocksTo servedBits KSLinOn Bext
  card_influenceTwo_le_min)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The served-output cap, a corollary of `Exfil`**: at every `δ ≥ 0`, KS at `RInflTwo` at `Exfil`'s extractor and
errors, and on a statement in scope, at any input values and any split `req` of its served outputs into requests, the
influence set has at most the smaller of `Σ_{T admissible} 2^{min(b(T), Σ_{r touched by T} min(b(T_r), |r|))}` and
`2^{Σ_r |r|} = 2^{|outs|}` members (`card_influenceTwo_le_min`). -/
theorem exfil_served {H : Msg → Msg} {L : CompiledTwo Coin A} (hL : L.Exfil H) (δ : ℝ) (hδ : 0 ≤ δ) :
    KSLinOn H L.V L.Scope (RInflTwo H L.commit L.parse L.partition L.two δ) L.E (L.κ δ) L.γ (fun _ => 1)
        (Bext H L.V L.E) ∧
      ∀ (x : Msg) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
        (d : Claims.Sampling.Design.Two) (a : ℕ → Option Bool) (R : Type) [Fintype R] [DecidableEq R]
        (req : Fin s.prog.outs.length → R),
        L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
        (influenceTwo H L.commit s Pt d δ a).card ≤
          min (∑ T ∈ admissibleTwo Pt d δ, 2 ^ min (bitsOf s Pt T)
              (∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
                min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)))
            (2 ^ ∑ r, servedBits req r) := by
  refine ⟨(hL δ hδ).1, ?_⟩
  intro x s Pt d a R _ _ req _ _ _ hin
  exact card_influenceTwo_le_min H L.commit hin δ a req

end CompiledTwo

end Claims.SamplingGeneric

/-- **The patterns' form** (`card_influenceTwo_le_patterns`). -/
theorem Claims.SecurityProofs.InfluenceTwoPatterns : Claims.Guarantees.InfluenceTwoPatterns :=
  fun H Cm _ _ _ hs δ a _ _ _ req => Claims.Sampling.card_influenceTwo_le_patterns H Cm hs δ a req
