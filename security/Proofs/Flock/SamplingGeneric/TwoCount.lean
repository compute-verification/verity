module

public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.Claims.Sampling.ProgReads
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
import Proofs.Flock.Claims.Prob

/-!
# What the incorrect proof blocks of a two-stage trace can serve

The exfiltration bound at the footprint: on fixed inputs, the boundary traces `RTwo` takes serve at most
`Σ_{T admissible} 2^{b(T)}` outputs (`card_influenceTwo_le`), and `CompilerExfilTwo` from `CompilerKSTwo`.

* `BGate.val_congr`, `Prog.runIn_congr`, `Prog.correct_congr` (`Claims.Sampling.ProgReads`): a proof block's
  correctness reads a trace only at its gates and the gates they read.
* `eq_of_mem_of_nodup`, `gateBlock`, `glue`, `glue_correct_iff`: **one trace at interiors `ι`**. Replay blocks are
  disjoint, so each gate's values at interiors are those of the one replay block that holds it (`glue`), and a proof block
  of a cell is correct on it exactly when it is on its replay block's values (`Prog.refIn`). Its incorrect proof blocks
  (`wrongOf`) have the footprint's counts (`counts_wrongOf`), so they are admissible when the footprint is in `M̄_δ`.
* `boundary_eq`, `outs_eq`: **the determinism lemma**, by induction along the topological order: two traces that make
  every proof block outside `T` correct and agree at the inputs and at `T`'s committed values (`exits`) agree at every
  committed value, so at every returned gate.
* `card_servedAt_le`, `card_influenceTwo_le`: **the count**. A set `T` of incorrect proof blocks serves at most
  `2^{|exits T|} ≤ 2^{b(T)}` outputs; the influence set is their union over the admissible sets.
* `card_admissibleTwo_le`, `card_influenceTwo_le_uniform`, `bitsOf_le_widths`, `card_decoded_le`: **the corollaries**.
  With every admissible set at most `K` proof blocks and `H` bits, `|influenceTwo| ≤ 2^H·Σ_{j≤K} C(n, j)`; `H` from
  per-stratum widths; a code that decodes each of `M` messages from served outputs in the influence set has
  `|M| ≤ |influenceTwo|`, so exporting `β` bits needs `2^β ≤ 2^H·Σ_{j≤K} C(n, j)`.
* `Design.Two.Pa_zero`, `footprint_congr`, `rTwo_congr`: **what `RTwo` reads**. At the zero footprint every cell
  escapes, so the acceptance probability is the first stage's at the empty set. The footprint, and so `RTwo`, reads no
  replay block's interior: two boundary traces that agree off the interiors are in `RTwo` together, so it says nothing
  of whether the interiors were ever computed (`Witnesses/SamplingTwoWork.lean`: compute credit is not certified with
  deferred interiors).
* `CompiledTwo.exfil_of_ks`, `compilerExfilTwo`: **`CompilerExfilTwo`**: `RTwo` implies `RInflTwo`.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

theorem eq_of_getD_eq {a b : Option Bool} (ha : a ≠ none) (hb : b ≠ none) (h : a.getD false = b.getD false) :
    a = b := by
  cases a <;> cases b <;> simp_all

/-! ## Disjoint blocks -/

/-- **Blocks of a duplicate-free flattening are disjoint**: a gate in blocks `i` and `j` has `i = j`. -/
theorem eq_of_mem_of_nodup {L : List (List ℕ)} (h : L.flatten.Nodup) {i j : ℕ} {a b : List ℕ} {x : ℕ}
    (hi : L[i]? = some a) (hj : L[j]? = some b) (ha : x ∈ a) (hb : x ∈ b) : i = j := by
  have h' : List.Pairwise (· ≠ ·) L.flatten := h
  have hp := List.pairwise_iff_getElem.1 (List.pairwise_flatten.1 h').2
  obtain ⟨hi', rfl⟩ := List.getElem?_eq_some_iff.1 hi
  obtain ⟨hj', rfl⟩ := List.getElem?_eq_some_iff.1 hj
  rcases Nat.lt_trichotomy i j with hij | hij | hij
  · exact absurd rfl (hp i j hi' hj' hij x ha x hb)
  · exact hij
  · exact absurd rfl (hp j i hj' hi' hij x hb x ha)

theorem getElem?_getD {L : List (List ℕ)} {v : ℕ} (hv : v < L.length) : L[v]? = some (L.getD v []) := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hv]
  rfl

theorem getD_of_getElem? {L : List (List ℕ)} {v : ℕ} {u : List ℕ} (h : L[v]? = some u) : L.getD v [] = u := by
  rw [List.getD_eq_getElem?_getD, h]
  rfl

/-- A gate of a partition's gates is in one of its blocks. -/
theorem exists_block {P : Partition} {g : ℕ} (h : g ∈ P.gates) :
    ∃ v < P.blocks.length, g ∈ P.blocks.getD v [] := by
  obtain ⟨u, hu, hg⟩ := List.mem_flatten.1 h
  obtain ⟨v, hv⟩ := List.mem_iff_getElem?.1 hu
  exact ⟨v, (List.getElem?_eq_some_iff.1 hv).1, by rw [getD_of_getElem? hv]; exact hg⟩

theorem mem_gates {P : Partition} {v g : ℕ} {u : List ℕ} (hu : P.blocks[v]? = some u) (hg : g ∈ u) : g ∈ P.gates :=
  List.mem_flatten.2 ⟨u, List.mem_of_getElem? hu, hg⟩

/-! ## One trace at interiors -/

/-- The replay block that holds gate `g`: the first block of `Pρ` that does. -/
def gateBlock (Pρ : Partition) (g : ℕ) : Option ℕ :=
  (List.range Pρ.blocks.length).find? fun ρ => (Pρ.blocks.getD ρ []).contains g

theorem gateBlock_some {Pρ : Partition} {g ρ : ℕ} (h : gateBlock Pρ g = some ρ) :
    Pρ.blocks[ρ]? = some (Pρ.blocks.getD ρ []) ∧ g ∈ Pρ.blocks.getD ρ [] := by
  obtain ⟨hp, hmem, _⟩ := List.find?_range_eq_some.1 h
  exact ⟨getElem?_getD (List.mem_range.1 hmem), by simpa using hp⟩

/-- With disjoint replay blocks, a gate's replay block is the one that holds it. -/
theorem gateBlock_of_mem {Pρ : Partition} (hn : Pρ.gates.Nodup) {ρ : ℕ} {r : List ℕ} {g : ℕ}
    (hr : Pρ.blocks[ρ]? = some r) (hg : g ∈ r) : gateBlock Pρ g = some ρ := by
  cases h : gateBlock Pρ g with
  | none =>
    have hρ : ρ < Pρ.blocks.length := (List.getElem?_eq_some_iff.1 hr).1
    have := (List.find?_range_eq_none.1 h) ρ hρ
    simp only [getD_of_getElem? hr] at this
    simp [hg] at this
  | some ρ' =>
    obtain ⟨hr', hg'⟩ := gateBlock_some h
    exact congrArg some (eq_of_mem_of_nodup hn hr' hr hg' hg)

/-- **The trace at interiors `ι`**, `ι ρ` replay block `ρ`'s: a gate of a replay block has its values there
(`Prog.refIn`: an interior gate's from `ι ρ`, a boundary gate's committed), every other gate its committed value. -/
def glue (C : Prog) (Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) (g : ℕ) : Option Bool :=
  match gateBlock Pρ g with
  | some ρ => C.refIn (Pρ.blocks.getD ρ []) val (ι ρ) g
  | none => val g

theorem glue_of_gateBlock {C : Prog} {Pρ : Partition} {val : ℕ → Option Bool} {ι : ℕ → ℕ → Bool} {g ρ : ℕ}
    (h : gateBlock Pρ g = some ρ) : glue C Pρ val ι g = C.refIn (Pρ.blocks.getD ρ []) val (ι ρ) g := by
  simp only [glue, h]

/-- A gate on the boundary of every set (an input, a returned gate) has its committed value. -/
theorem glue_of_boundary {C : Prog} {Pρ : Partition} {val : ℕ → Option Bool} {ι : ℕ → ℕ → Bool} {g : ℕ}
    (h : ∀ r, C.Boundary r g = true) : glue C Pρ val ι g = val g := by
  unfold glue
  split
  · exact Prog.refIn_of_boundary (h _)
  · rfl

/-- **The gluing lemma**: a proof block within replay block `ρ` is correct on the trace at interiors `ι` exactly when it
is on `ρ`'s values at its interior `ι ρ`. -/
theorem glue_correct_iff {C : Prog} {Pt Pρ : Partition} (hP : Pt.Partitions C) (hn : Pρ.gates.Nodup)
    {val : ℕ → Option Bool} {ι : ℕ → ℕ → Bool} {v ρ : ℕ} {u r : List ℕ} (hu : Pt.blocks[v]? = some u)
    (hr : Pρ.blocks[ρ]? = some r) (hur : ∀ g ∈ u, g ∈ r) :
    C.Correct (glue C Pρ val ι) u ↔ C.Correct (C.refIn r val (ι ρ)) u := by
  have hin : ∀ g ∈ r, glue C Pρ val ι g = C.refIn r val (ι ρ) g := fun g hg => by
    rw [glue_of_gateBlock (gateBlock_of_mem hn hr hg), getD_of_getElem? hr]
  refine C.correct_congr (fun g hg => hin g (hur g hg)) fun g hg x hx i hi _ => ?_
  by_cases hir : i ∈ r
  · exact hin i hir
  rw [Prog.refIn_of_not_mem hir]
  cases hgi : gateBlock Pρ i with
  | none => simp only [glue, hgi]
  | some ρ' =>
    obtain ⟨hr', hi'⟩ := gateBlock_some hgi
    have hgr' : g ∉ Pρ.blocks.getD ρ' [] := fun hg' => by
      have := eq_of_mem_of_nodup hn hr' hr hg' (hur g hg)
      subst this
      exact hir (by rw [getD_of_getElem? hr] at hi'; exact hi')
    rw [glue_of_gateBlock hgi, Prog.refIn_of_boundary (Prog.boundary_of_read hgr'
      (hP.2.1 g (mem_gates hu hg)) hx hi)]

open Classical in
/-- **The incorrect proof blocks of a trace** `X`: those it doesn't make correct. -/
noncomputable def wrongOf (C : Prog) (Pt : Partition) (X : ℕ → Option Bool) : Finset ℕ :=
  (range Pt.blocks.length).filter fun v => ¬ C.Correct X (Pt.blocks.getD v [])

/-- A cell's proof block lies within its replay block. -/
theorem cell_blocks {Pt Pρ : Partition} {ρ t v : ℕ} (h : v ∈ cell Pt Pρ ρ t) :
    ∃ u r, Pt.blocks[v]? = some u ∧ Pρ.blocks[ρ]? = some r ∧ ∀ g ∈ u, g ∈ r := by
  have hρ : replayOf Pt Pρ v = some ρ := (mem_filter.1 h).2.1
  unfold replayOf at hρ
  have hp := (List.find?_range_eq_some.1 hρ).1
  cases hu : Pt.blocks[v]? with
  | none => simp [hu] at hp
  | some u =>
    cases hr : Pρ.blocks[ρ]? with
    | none => simp [hu, hr] at hp
    | some r =>
      simp only [hu, hr, List.all_eq_true] at hp
      exact ⟨u, r, rfl, rfl, fun g hg => by simpa using hp g hg⟩

/-- **The incorrect proof blocks at interiors `ι` are the footprint's, cell by cell.** -/
theorem inter_cell_wrongOf (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hP : Pt.Partitions s.prog) (hn : Pρ.gates.Nodup) (w : Wit) (ι : ℕ → ℕ → Bool) (ρ t : ℕ) :
    wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) ∩ cell Pt Pρ ρ t = wrongIn H Cm s Pt Pρ w ρ t (ι ρ) := by
  classical
  ext v
  unfold wrongOf wrongIn
  simp only [mem_inter, mem_filter]
  constructor
  · rintro ⟨⟨_, hnc⟩, hc⟩
    refine ⟨hc, ?_⟩
    rintro ⟨u, r, hu, hr, hcor⟩
    obtain ⟨u', r', hu', hr', hur⟩ := cell_blocks hc
    obtain rfl := Option.some.inj (hu'.symm.trans hu)
    obtain rfl := Option.some.inj (hr'.symm.trans hr)
    exact hnc (by rw [getD_of_getElem? hu']; exact (glue_correct_iff hP hn hu' hr' hur).2 hcor)
  · rintro ⟨hc, hnc⟩
    refine ⟨⟨cell_subset Pt Pρ ρ t hc, fun hcor => hnc ?_⟩, hc⟩
    obtain ⟨u, r, hu, hr, hur⟩ := cell_blocks hc
    rw [getD_of_getElem? hu] at hcor
    exact ⟨u, r, hu, hr, (glue_correct_iff hP hn hu hr hur).1 hcor⟩

theorem counts_wrongOf (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hP : Pt.Partitions s.prog) (hn : Pρ.gates.Nodup) (w : Wit) (ι : ℕ → ℕ → Bool) :
    counts Pt Pρ (wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι)) = footprint H Cm s Pt Pρ w ι := by
  funext ρ t
  exact congrArg card (inter_cell_wrongOf H Cm hP hn w ι ρ t)

/-- **The incorrect set at interiors whose footprint is in `M̄_δ` is admissible.** -/
theorem wrongOf_mem_admissible (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hP : Pt.Partitions s.prog) (hn : d.part.gates.Nodup) {w : Wit} {ι : ℕ → ℕ → Bool} {δ : ℝ}
    (hM : footprint H Cm s Pt d.part w ι ∈ d.M Pt δ) :
    wrongOf s.prog Pt (glue s.prog d.part (w.val H Cm s) ι) ∈ admissibleTwo Pt d δ := by
  classical
  unfold admissibleTwo
  refine mem_filter.2 ⟨mem_powerset.2 fun v hv => ?_, ?_⟩
  · unfold wrongOf at hv
    exact (mem_filter.1 hv).1
  · rw [counts_wrongOf H Cm hP hn w ι]
    exact hM

/-! ## The determinism lemma -/

/-- **The determinism lemma**: two traces that make every proof block outside `T` correct, with the same values at
the input gates and the same bits at the committed values of `T`'s proof blocks, have the same bits at every committed
value of every proof block. By induction along the topological order: a correct proof block computes its committed
values from its inputs and the committed values it reads, each of an earlier gate. -/
theorem boundary_eq {C : Prog} {P : Partition} (hC : C.Topological) (hP : P.Partitions C)
    {X Y : ℕ → Option Bool} {T : Finset ℕ}
    (hX : ∀ v < P.blocks.length, v ∉ T → C.Correct X (P.blocks.getD v []))
    (hY : ∀ v < P.blocks.length, v ∉ T → C.Correct Y (P.blocks.getD v []))
    (hin : ∀ g, C.gates[g]? = some .input → X g = Y g)
    (hT : ∀ v ∈ T, ∀ g ∈ P.blocks.getD v [], C.Boundary (P.blocks.getD v []) g = true →
      (X g).getD false = (Y g).getD false) :
    ∀ g, ∀ v < P.blocks.length, g ∈ P.blocks.getD v [] → C.Boundary (P.blocks.getD v []) g = true →
      (X g).getD false = (Y g).getD false := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro v hv hg hb
  by_cases hvT : v ∈ T
  · exact hT v hvT g hg hb
  have hvb : P.blocks[v]? = some (P.blocks.getD v []) := getElem?_getD hv
  obtain ⟨b, hXb, hXr⟩ := (hX v hv hvT g hg).2 hb
  obtain ⟨b', hYb, hYr⟩ := (hY v hv hvT g hg).2 hb
  have hrun : C.runIn X (P.blocks.getD v []) g = C.runIn Y (P.blocks.getD v []) g := by
    refine C.runIn_congr g hg (fun g' _ _ h' => hin g' h') fun g' hg' hle x hx i hi hiu => ?_
    have hig : i < g := (hC.1 g' x hx i hi).trans_le hle
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
    exact eq_of_getD_eq hXn hYn (ih i hig v' hv' hiv' (Prog.boundary_of_read hg'v' hg'lt hx hi))
  have hbb : b = b' := Option.some.inj (hXr.symm.trans (hrun.trans hYr))
  rw [hXb, hYb, hbb]

/-- **The returned gates' bits** agree under the determinism lemma's hypotheses: a returned gate is a constant or
committed by its proof block. -/
theorem outs_eq {C : Prog} {P : Partition} (hC : C.Topological) (hP : P.Partitions C)
    {X Y : ℕ → Option Bool} {T : Finset ℕ}
    (hX : ∀ v < P.blocks.length, v ∉ T → C.Correct X (P.blocks.getD v []))
    (hY : ∀ v < P.blocks.length, v ∉ T → C.Correct Y (P.blocks.getD v []))
    (hin : ∀ g, C.gates[g]? = some .input → X g = Y g)
    (hT : ∀ v ∈ T, ∀ g ∈ P.blocks.getD v [], C.Boundary (P.blocks.getD v []) g = true →
      (X g).getD false = (Y g).getD false) :
    ∀ o ∈ C.outs, (C.read X o).getD false = (C.read Y o).getD false := by
  intro o ho
  by_cases hc : ∃ c, C.gates[o]? = some (.const c)
  · obtain ⟨c, hc⟩ := hc
    simp only [Prog.read, hc]
  have hc' : ∀ c, C.gates[o]? ≠ some (.const c) := fun c h => hc ⟨c, h⟩
  rw [C.read_of_not_const X hc', C.read_of_not_const Y hc']
  obtain ⟨v, hv, hov⟩ := exists_block (hP.2.2.1 o (hC.2 o ho) hc' fun hu => hu.2.1 ho)
  exact boundary_eq hC hP hX hY hin hT o v hv hov (Prog.boundary_of_out _ ho)

/-! ## The count -/

open Classical in
/-- What the boundary traces with inputs `a` whose incorrect set at some interiors is `T` serve. -/
noncomputable def servedAt (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition)
    (a : ℕ → Option Bool) (T : Finset ℕ) : Finset (Fin s.prog.outs.length → Bool) :=
  univ.filter fun y => ∃ (w : Wit) (ι : ℕ → ℕ → Bool), inputsOf H Cm s w = a ∧
    wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) = T ∧ served H Cm s w = y

/-- **`T`'s exits**: the committed values of its proof blocks. -/
def exits (C : Prog) (Pt : Partition) (T : Finset ℕ) : Finset ℕ :=
  T.biUnion fun v => ((Pt.blocks.getD v []).filter fun g => C.Boundary (Pt.blocks.getD v []) g).toFinset

theorem card_exits_le (s : AuditStatement) (Pt : Partition) (T : Finset ℕ) :
    (exits s.prog Pt T).card ≤ bitsOf s Pt T :=
  card_biUnion_le.trans (sum_le_sum fun _ _ => List.toFinset_card_le _)

/-- Two boundary traces at interiors with the incorrect set `T`, the same inputs and the same bits at `T`'s exits serve
the same outputs. -/
theorem served_eq_of (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) {w w' : Wit} {ι ι' : ℕ → ℕ → Bool} {T : Finset ℕ}
    (hw : wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) = T)
    (hw' : wrongOf s.prog Pt (glue s.prog Pρ (w'.val H Cm s) ι') = T)
    (ha : inputsOf H Cm s w = inputsOf H Cm s w')
    (hT : ∀ g ∈ exits s.prog Pt T,
      (glue s.prog Pρ (w.val H Cm s) ι g).getD false = (glue s.prog Pρ (w'.val H Cm s) ι' g).getD false) :
    served H Cm s w = served H Cm s w' := by
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
    rw [if_pos hg, if_pos hg] at h1
    rw [glue_of_boundary fun r => Prog.boundary_of_input r hg, glue_of_boundary fun r => Prog.boundary_of_input r hg]
    exact h1
  have hTb : ∀ v ∈ T, ∀ g ∈ Pt.blocks.getD v [], s.prog.Boundary (Pt.blocks.getD v []) g = true →
      (glue s.prog Pρ (w.val H Cm s) ι g).getD false = (glue s.prog Pρ (w'.val H Cm s) ι' g).getD false :=
    fun v hv g hg hb => hT g (mem_biUnion.2 ⟨v, hv, List.mem_toFinset.2 (List.mem_filter.2 ⟨hg, hb⟩)⟩)
  funext i
  have ho : s.prog.outs[i] ∈ s.prog.outs := List.getElem_mem i.isLt
  have hglue : ∀ (u : Wit) (κ : ℕ → ℕ → Bool), s.prog.read (u.val H Cm s) s.prog.outs[i] =
      s.prog.read (glue s.prog Pρ (u.val H Cm s) κ) s.prog.outs[i] := fun u κ =>
    s.prog.read_congr (glue_of_boundary fun r => Prog.boundary_of_out r ho).symm
  show (s.prog.read (w.val H Cm s) s.prog.outs[i]).getD false = (s.prog.read (w'.val H Cm s) s.prog.outs[i]).getD false
  rw [hglue w ι, hglue w' ι']
  exact outs_eq hC hP (hcor _ hw) (hcor _ hw') hin hTb _ ho

/-- **A set `T` of incorrect proof blocks serves at most `2^{|exits T|}` outputs** on fixed inputs. -/
theorem card_servedAt_le (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hC : s.prog.Topological) (hP : Pt.Partitions s.prog) (a : ℕ → Option Bool) (T : Finset ℕ) :
    (servedAt H Cm s Pt Pρ a T).card ≤ 2 ^ (exits s.prog Pt T).card := by
  classical
  have hspec : ∀ y ∈ servedAt H Cm s Pt Pρ a T, ∃ (w : Wit) (ι : ℕ → ℕ → Bool), inputsOf H Cm s w = a ∧
      wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) = T ∧ served H Cm s w = y :=
    fun y hy => (mem_filter.1 hy).2
  let f : (Fin s.prog.outs.length → Bool) → (exits s.prog Pt T → Bool) := fun y =>
    if h : ∃ (w : Wit) (ι : ℕ → ℕ → Bool), inputsOf H Cm s w = a ∧
        wrongOf s.prog Pt (glue s.prog Pρ (w.val H Cm s) ι) = T ∧ served H Cm s w = y then
      fun g => (glue s.prog Pρ ((Classical.choose h).val H Cm s) (Classical.choose (Classical.choose_spec h)) g).getD false
    else fun _ => false
  calc (servedAt H Cm s Pt Pρ a T).card ≤ (univ : Finset (exits s.prog Pt T → Bool)).card := by
        refine card_le_card_of_injOn f (fun _ _ => mem_coe.2 (mem_univ _)) ?_
        intro y hy y' hy' hf
        have h := hspec y hy
        have h' := hspec y' hy'
        simp only [f, h, h', ↓reduceDIte] at hf
        obtain ⟨ha, hw, hsw⟩ := Classical.choose_spec (Classical.choose_spec h)
        obtain ⟨ha', hw', hsw'⟩ := Classical.choose_spec (Classical.choose_spec h')
        rw [← hsw, ← hsw']
        exact served_eq_of H Cm hC hP hw hw' (ha.trans ha'.symm) fun g hg => congrFun hf ⟨g, hg⟩
    _ = 2 ^ (exits s.prog Pt T).card := by rw [card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]

/-- **The exfiltration count at the footprint**: on a statement in scope, at any input values, the boundary traces
`RTwo` takes at `δ` serve at most `Σ_{T admissible} 2^{b(T)}` outputs. -/
theorem card_influenceTwo_le (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) :
    (influenceTwo H Cm s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T := by
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
    _ ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T :=
        sum_le_sum fun T _ => (card_servedAt_le H Cm hs.1 hs.2.1 a T).trans
          (Nat.pow_le_pow_right (by norm_num) (card_exits_le s Pt T))

/-! ## The corollaries -/

/-- **The location term**: when every admissible set holds at most `K` proof blocks, there are at most
`Σ_{j≤K} C(n, j)` of them. -/
theorem card_admissibleTwo_le {Pt : Partition} {d : Design.Two} {δ : ℝ} {K : ℕ}
    (hK : ∀ T ∈ admissibleTwo Pt d δ, T.card ≤ K) :
    (admissibleTwo Pt d δ).card ≤ ∑ j ∈ range (K + 1), Pt.blocks.length.choose j := by
  classical
  have hsub : admissibleTwo Pt d δ ⊆ (range (K + 1)).biUnion fun j => (range Pt.blocks.length).powersetCard j := by
    intro T hT
    have hTs : T ⊆ range Pt.blocks.length := by
      unfold admissibleTwo at hT
      exact mem_powerset.1 (mem_filter.1 hT).1
    exact mem_biUnion.2 ⟨T.card, mem_range.2 (Nat.lt_succ_of_le (hK T hT)), mem_powersetCard.2 ⟨hTs, rfl⟩⟩
  refine (card_le_card hsub).trans (card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => ?_)))
  rw [card_powersetCard, card_range]

/-- **The uniform bound**: with every admissible set at most `K` proof blocks and `H` committed bits,
`|influenceTwo| ≤ 2^H·Σ_{j≤K} C(n, j)`. -/
theorem card_influenceTwo_le_uniform (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {Hb K : ℕ}
    (hH : ∀ T ∈ admissibleTwo Pt d δ, bitsOf s Pt T ≤ Hb) (hK : ∀ T ∈ admissibleTwo Pt d δ, T.card ≤ K) :
    (influenceTwo H Cm s Pt d δ a).card ≤ 2 ^ Hb * ∑ j ∈ range (K + 1), Pt.blocks.length.choose j := by
  refine (card_influenceTwo_le H Cm hs δ a).trans ?_
  calc ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ Hb :=
        sum_le_sum fun T hT => Nat.pow_le_pow_right (by norm_num) (hH T hT)
    _ = 2 ^ Hb * (admissibleTwo Pt d δ).card := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ 2 ^ Hb * ∑ j ∈ range (K + 1), Pt.blocks.length.choose j := Nat.mul_le_mul_left _ (card_admissibleTwo_le hK)

/-- **Per-stratum widths**: with every proof block of `T` committing at most its stratum's width `b`,
`b(T) ≤ Σ_{v∈T} b(t(v))`, so `H` may be taken as the largest such sum over the admissible sets. -/
theorem bitsOf_le_widths (s : AuditStatement) (Pt : Partition) (b : ℕ → ℕ) {T : Finset ℕ}
    (hb : ∀ v ∈ T, committed s.prog (Pt.blocks.getD v []) ≤ b (Pt.stratumOf v)) :
    bitsOf s Pt T ≤ ∑ v ∈ T, b (Pt.stratumOf v) :=
  sum_le_sum hb

/-- **Exporting bits**: an encoding of the messages `M` into served outputs, each in the influence set, that a
receiver decodes (`g`) has `|M| ≤ 2^H·Σ_{j≤K} C(n, j)`: exporting `β` bits needs `β ≤ H + log₂ Σ_{j≤K} C(n, j)`. -/
theorem card_decoded_le {M : Type} [Fintype M] (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) (δ : ℝ) (a : ℕ → Option Bool) {Hb K : ℕ}
    (hH : ∀ T ∈ admissibleTwo Pt d δ, bitsOf s Pt T ≤ Hb) (hK : ∀ T ∈ admissibleTwo Pt d δ, T.card ≤ K)
    (y : M → Fin s.prog.outs.length → Bool) (g : (Fin s.prog.outs.length → Bool) → M)
    (hy : ∀ m, y m ∈ influenceTwo H Cm s Pt d δ a) (hg : ∀ m, g (y m) = m) :
    Fintype.card M ≤ 2 ^ Hb * ∑ j ∈ range (K + 1), Pt.blocks.length.choose j := by
  classical
  have hinj : Function.Injective y := fun m m' h => by rw [← hg m, h, hg m']
  calc Fintype.card M = (univ.image y).card := (card_image_of_injective _ hinj).symm
    _ ≤ (influenceTwo H Cm s Pt d δ a).card := card_le_card fun z hz => by
        obtain ⟨m, -, rfl⟩ := mem_image.1 hz
        exact hy m
    _ ≤ _ := card_influenceTwo_le_uniform H Cm hs δ a hH hK

/-! ## What `RTwo` reads -/

/-- **The acceptance probability at the zero footprint** is the first stage's at the empty set: every cell escapes. -/
theorem Design.Two.Pa_zero (d : Design.Two) (Pt : Partition) : d.Pa Pt (fun _ _ => 0) = d.replay.Pa d.part ∅ := by
  have he : ∀ ρ, d.esc Pt (fun _ _ => 0) ρ = 1 := fun ρ => by
    unfold Design.Two.esc
    exact prod_eq_one fun t _ => escCell_zero _ _
  unfold Design.Two.Pa
  rw [sum_eq_single ∅]
  · simp [he]
  · intro T _ hT
    obtain ⟨ρ, hρ⟩ := nonempty_iff_ne_empty.2 hT
    have h0 : ∏ ρ ∈ T, (1 - d.esc Pt (fun _ _ => 0) ρ) = 0 := prod_eq_zero hρ (by simp [he])
    rw [h0, zero_mul, mul_zero]
  · intro h
    exact absurd (empty_mem_powerset _) h

/-- At a gate of replay block `ρ`, its values are the same under two committed traces that agree at every gate that
is no replay block's interior. -/
theorem refIn_congr_off {C : Prog} {Pρ : Partition} (hn : Pρ.gates.Nodup) {val val' : ℕ → Option Bool}
    (hw : ∀ g, (∀ (ρ : ℕ) (r : List ℕ), Pρ.blocks[ρ]? = some r → g ∈ r → C.Boundary r g = true) → val g = val' g)
    {ρ : ℕ} {r : List ℕ} (hr : Pρ.blocks[ρ]? = some r) (ι : ℕ → Bool) {g : ℕ} (hg : g ∈ r) :
    C.refIn r val ι g = C.refIn r val' ι g := by
  by_cases hb : C.Boundary r g = true
  · rw [Prog.refIn_of_boundary hb, Prog.refIn_of_boundary hb]
    refine hw g fun ρ' r' hr' hg' => ?_
    have hρ := eq_of_mem_of_nodup hn hr' hr hg' hg
    subst hρ
    rw [Option.some.inj (hr'.symm.trans hr)]
    exact hb
  · unfold Prog.refIn
    simp [hg, hb]

/-- **The footprint reads no replay block's interior**: two boundary traces that open the same values at every gate
that is no replay block's interior have the same footprint at every interior values. -/
theorem footprint_congr (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt Pρ : Partition}
    (hP : Pt.Partitions s.prog) (hn : Pρ.gates.Nodup) {w w' : Wit}
    (hw : ∀ g, (∀ (ρ : ℕ) (r : List ℕ), Pρ.blocks[ρ]? = some r → g ∈ r → s.prog.Boundary r g = true) →
      w.val H Cm s g = w'.val H Cm s g) (ι : ℕ → ℕ → Bool) :
    footprint H Cm s Pt Pρ w ι = footprint H Cm s Pt Pρ w' ι := by
  classical
  funext ρ t
  unfold footprint wrongIn
  refine congrArg card (filter_congr fun v hv => not_congr ?_)
  obtain ⟨u, r, hu, hr, hur⟩ := cell_blocks hv
  have hin : ∀ g ∈ r, s.prog.refIn r (w.val H Cm s) (ι ρ) g = s.prog.refIn r (w'.val H Cm s) (ι ρ) g :=
    fun g hg => refIn_congr_off hn hw hr (ι ρ) hg
  have hiff : s.prog.Correct (s.prog.refIn r (w.val H Cm s) (ι ρ)) u ↔
      s.prog.Correct (s.prog.refIn r (w'.val H Cm s) (ι ρ)) u := by
    refine s.prog.correct_congr (fun g hg => hin g (hur g hg)) fun g hg x hx i hi _ => ?_
    by_cases hir : i ∈ r
    · exact hin i hir
    rw [Prog.refIn_of_not_mem hir, Prog.refIn_of_not_mem hir]
    refine hw i fun ρ' r' hr' hi' => ?_
    have hgr' : g ∉ r' := fun hg' => by
      have hρ := eq_of_mem_of_nodup hn hr' hr hg' (hur g hg)
      subst hρ
      rw [Option.some.inj (hr'.symm.trans hr)] at hi'
      exact hir hi'
    exact Prog.boundary_of_read hgr' (hP.2.1 g (mem_gates hu hg)) hx hi
  constructor
  · rintro ⟨u', r', hu', hr', hc⟩
    rw [hu, Option.some.injEq] at hu'
    rw [hr, Option.some.injEq] at hr'
    subst hu' hr'
    exact ⟨_, _, hu, hr, hiff.1 hc⟩
  · rintro ⟨u', r', hu', hr', hc⟩
    rw [hu, Option.some.injEq] at hu'
    rw [hr, Option.some.injEq] at hr'
    subst hu' hr'
    exact ⟨_, _, hu, hr, hiff.2 hc⟩

/-- **`RTwo` is blind to the interiors**: on a statement in scope, two boundary traces that open the same values at
every gate that is no replay block's interior are in `RTwo` together. What it takes holds whether or not the interiors
were ever computed. -/
theorem rTwo_congr (H : Msg → Msg) (Cm : CommitmentScheme Bool) {parse : Msg → Option AuditStatement}
    {partition : AuditStatement → Option Partition} {two : AuditStatement → Option Design.Two} {δ : ℝ} {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hs : parse x = some s) (hPt : partition s = some Pt)
    (hd : two s = some d) (hin : InScopeTwo Pt d s) {w w' : Wit}
    (hw : ∀ g, (∀ (ρ : ℕ) (r : List ℕ), d.part.blocks[ρ]? = some r → g ∈ r → s.prog.Boundary r g = true) →
      w.val H Cm s g = w'.val H Cm s g) :
    RTwo H Cm parse partition two δ x w ↔ RTwo H Cm parse partition two δ x w' := by
  have key : ∀ ι, footprint H Cm s Pt d.part w ι = footprint H Cm s Pt d.part w' ι :=
    footprint_congr H Cm hin.2.1 hin.2.2.1 hw
  constructor
  · rintro ⟨s', Pt', d', hs', hPt', hd', ι, hM⟩
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    rw [hPt, Option.some.injEq] at hPt'
    rw [hd, Option.some.injEq] at hd'
    subst hPt' hd'
    exact ⟨_, _, _, hs, hPt, hd, ι, by rw [← key ι]; exact hM⟩
  · rintro ⟨s', Pt', d', hs', hPt', hd', ι, hM⟩
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    rw [hPt, Option.some.injEq] at hPt'
    rw [hd, Option.some.injEq] at hd'
    subst hPt' hd'
    exact ⟨_, _, _, hs, hPt, hd, ι, by rw [key ι]; exact hM⟩

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (RTwo RInflTwo card_influenceTwo_le)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- A boundary trace in `RTwo` is in `RInflTwo`: its served outputs are in the influence set at its own inputs. -/
theorem rInflTwo_of_rTwo {H : Msg → Msg} {Cm : Claims.Sampling.CommitmentScheme Bool}
    {parse : Msg → Option Claims.Sampling.AuditStatement}
    {partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition}
    {two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two} {δ : ℝ} {x : Msg}
    {w : Claims.Sampling.Wit} (h : RTwo H Cm parse partition two δ x w) : RInflTwo H Cm parse partition two δ x w := by
  classical
  obtain ⟨s, Pt, d, hs, hPt, hd, hM⟩ := h
  refine ⟨s, Pt, d, hs, hPt, hd, ?_⟩
  unfold Claims.Sampling.influenceTwo
  exact mem_filter.2 ⟨mem_univ _, w, rfl, hM, rfl⟩

/-- **`Exfil` from `KS`**: `RTwo` implies `RInflTwo`, and the count is `card_influenceTwo_le`. -/
theorem exfil_of_ks {H : Msg → Msg} {L : CompiledTwo Coin A} (h : L.KS H) : L.Exfil H := by
  intro δ hδ
  refine ⟨?_, fun x s Pt d a _ _ _ hin => card_influenceTwo_le H L.commit hin δ a⟩
  intro x hx Ω _ _ P
  have hk := h δ hδ x hx Ω P
  unfold PrBad at hk ⊢
  refine le_trans (Pr_mono fun s hs => ⟨hs.1, fun hR => hs.2 ?_⟩) hk
  revert hR
  cases (L.E.eval H P x s.1 s.2.1 s.2.2).1 with
  | none => exact id
  | some w => exact rInflTwo_of_rTwo

end CompiledTwo

theorem compilerExfilTwo_of (h : CompilerKSTwo) : CompilerExfilTwo := by
  intro Coin _ _ _ A L H hA hB hS
  exact CompiledTwo.exfil_of_ks (h Coin A L H hA hB hS)

/-- **`CompilerExfilTwo`**: the two-stage compiler bounds exfiltration at the footprint. -/
theorem compilerExfilTwo : CompilerExfilTwo := compilerExfilTwo_of compilerKSTwo

end Claims.SamplingGeneric
