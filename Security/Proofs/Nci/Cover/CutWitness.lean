import Proofs.Nci.Cover.Cut
import Proofs.Nci.Cover.Witness
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum

/-!
# The cover theorem: the cut width holds, and what it cannot give

`cut_witness`: a circuit with a counted product that satisfies `ConditionalUpdateHardness`, disjoint upstreams at width
`f = 2` (so the cut width), the upstream certificate, `k < b D`, and has a cover by inference blocks.
`cut_witness_width`: there every inference block of every size that computes the counted product whole is handed at
least two gates of its activation region, and `cut_witness_tight` shows one handed exactly two.

Two limits. `width_unbounded_fails`: on the same circuit `Width` as `Assumptions` states it fails at `s = 17`, since a
block handed six bits outside `θ` computes half of the upstream and is handed one gate of the region; only blocks
handed fewer than `b D` bits outside `θ`, what `WidthOfCut` covers, pay the width. `shared_upstream`: a circuit in which
two activation gates the product reads are each reached, avoiding the other, from upstream products over `D`
parameters (so two vertex-disjoint paths enter the activation, each from a gate with a deep upstream of its own), yet an
inference block computing the product whole is handed one gate of the region: both bits are computed from one gate
`h`, and the block is handed `h`. The cut width and disjoint upstreams fail there (`shared_upstream_not_cutWidth`), as
`WidthOfCut` says they must.

The circuit (`CutWitness.tC`, 27 gates, `θ = {0}`): gate 0 is the registered token and gates 1–9 the updated
parameters' bits; the upstream products `(0, l)`, `l = 1, …, 8`, are gates `9 + l`, each the token times parameter `l`;
gates 18–21 add them in pairs and 22, 23 add those in pairs, two partial sums of four products each; the activation gates
24 and 25 each read both partial sums; the counted product `(1, 9)` is gate 26, reading both activation gates and
parameter 9. Use 1's region is gates 10–25. The two disjoint upstreams are the two halves, each partial sum's tree with
one activation gate, at depth `D = 4`: a block handed one gate of the region leaves one half whole behind it.
-/

namespace Nci.Cover.SecurityProofs

local instance cutDecSubcircuit (C : Circuit) (I : Finset (Fin C.N)) : Decidable (C.Subcircuit I) :=
  inferInstanceAs (Decidable (I.Nonempty ∧ ∀ g ∈ I, C.pred g ⊆ I ∨ Disjoint (C.pred g) I))

local instance cutDecInferenceBlock (C : Circuit) (s k : ℕ) (I : Finset (Fin C.N)) :
    Decidable (C.InferenceBlock s k I) :=
  inferInstanceAs (Decidable (C.Subcircuit I ∧ I.card = s ∧ (C.inputs I \ C.θ).card ≤ k))

local instance cutDecCovers (C : Circuit) (𝓘 : Finset (Finset (Fin C.N))) : Decidable (C.Covers 𝓘) :=
  inferInstanceAs (Decidable (∀ g, (C.pred g).Nonempty → ∃ I ∈ 𝓘, g ∈ C.computed I))

namespace CutWitness

/-- The wiring: the token 0 and the parameter bits 1–9 are input gates; `9 + l` is upstream product `l`; 18–21 and
22, 23 are the two partial sums' trees; 24 and 25 are the activation gates and 26 the counted product. -/
def tPred : Fin 27 → Finset (Fin 27) :=
  ![∅, ∅, ∅, ∅, ∅, ∅, ∅, ∅, ∅, ∅, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8},
    {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}, {22, 23}, {9, 24, 25}]

abbrev tC : Circuit where
  N := 27
  pred := tPred
  pred_lt := by decide
  θ := {0}
  θ_input := by decide

def tm : ℕ × ℕ → Finset (Fin 27)
  | (0, 1) => {10}
  | (0, 2) => {11}
  | (0, 3) => {12}
  | (0, 4) => {13}
  | (0, 5) => {14}
  | (0, 6) => {15}
  | (0, 7) => {16}
  | (0, 8) => {17}
  | (1, 9) => {26}
  | _ => ∅

def to' : ℕ × ℕ → Fin 27
  | (0, 1) => 10
  | (0, 2) => 11
  | (0, 3) => 12
  | (0, 4) => 13
  | (0, 5) => 14
  | (0, 6) => 15
  | (0, 7) => 16
  | (0, 8) => 17
  | _ => 26

/-- The products: eight upstream ones, `(0, l)`, and the counted `(1, 9)`. -/
abbrev tM : MulBlocks tC ℕ ℕ where
  Pall := {(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 9)}
  P := {(1, 9)}
  P_subset := by decide
  m := tm
  m_subcircuit := by decide
  m_computed := by decide
  m_disjoint := by
    have h : ∀ p ∈ ({(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 9)} : Finset (ℕ × ℕ)),
        ∀ q ∈ ({(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 9)} : Finset (ℕ × ℕ)),
        p ≠ q → Disjoint (tm p) (tm q) := by
      decide
    exact fun p hp q hq => h p hp q hq
  o := to'
  m_outputs := by decide

/-- Use 1's activation region: the upstream products, the two trees and the activation gates. -/
def tA : ℕ → Finset (Fin 27) := fun j =>
  if j = 1 then {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25} else ∅

def tF : ℕ → Finset (Fin 27)
  | 1 => {1}
  | 2 => {2}
  | 3 => {3}
  | 4 => {4}
  | 5 => {5}
  | 6 => {6}
  | 7 => {7}
  | 8 => {8}
  | 9 => {9}
  | _ => ∅

/-- The two halves: each partial sum's tree with one activation gate. -/
def tB : Fin 2 → Finset (Fin 27) := ![{10, 11, 12, 13, 18, 19, 22, 24}, {14, 15, 16, 17, 20, 21, 23, 25}]

/-- The cover: a block for each upstream product, for each pair sum, for each partial sum, one for the activation gates
and one for the counted product, each of four gates. -/
def tI : Finset (Finset (Fin 27)) :=
  {{0, 1, 10, 9}, {0, 2, 11, 9}, {0, 3, 12, 9}, {0, 4, 13, 9}, {0, 5, 14, 9}, {0, 6, 15, 9}, {0, 7, 16, 9},
    {0, 8, 17, 9}, {10, 11, 18, 9}, {12, 13, 19, 9}, {14, 15, 20, 9}, {16, 17, 21, 9}, {18, 19, 22, 9},
    {20, 21, 23, 9}, {22, 23, 24, 25}, {24, 25, 9, 26}}

theorem tM_P_nonempty : tM.P.Nonempty := ⟨(1, 9), by decide⟩

theorem tM_conditional (s : ℕ) : Assumptions.ConditionalUpdateHardness tM tA s 1 := by
  have hcard : ∀ p ∈ tM.Pall, (tF p.2).card = 1 := by decide
  refine Witness.conditionalUpdateHardness_of_feeds tM tA tF s (by decide) (by decide) (by decide) fun p hp => ?_
  rw [hcard p hp, Nat.cast_one]

theorem tM_hardness (s : ℕ) : Assumptions.UpdateHardness tM s 1 :=
  updateHardness_of_conditional tM tA (tM_conditional s)

theorem path4 {a b c d : Fin 27} (h : [a, b, c, d].IsChain fun u v => u ∈ tC.pred v) : tC.IsPath [a, b, c, d] a d :=
  ⟨rfl, rfl, h⟩

theorem tM_disjointUpstreams : tM.DisjointUpstreams tA 4 2 := by
  intro p hp
  obtain rfl : p = (1, 9) := Finset.mem_singleton.1 hp
  refine ⟨tB, by decide, fun i => ?_⟩
  fin_cases i
  · refine ⟨by decide, {(0, 1), (0, 2), (0, 3), (0, 4)}, by decide, by decide, 26, by decide, 24, by decide, ?_⟩
    have hU : ∀ q ∈ ({(0, 1), (0, 2), (0, 3), (0, 4)} : Finset (ℕ × ℕ)), tM.m q ⊆ {10, 11, 12, 13} := by decide
    intro q hq x hx
    have hx' := hU q hq hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx'
    rcases hx' with rfl | rfl | rfl | rfl
    · exact ⟨[10, 18, 22, 24], path4 (by decide), by decide⟩
    · exact ⟨[11, 18, 22, 24], path4 (by decide), by decide⟩
    · exact ⟨[12, 19, 22, 24], path4 (by decide), by decide⟩
    · exact ⟨[13, 19, 22, 24], path4 (by decide), by decide⟩
  · refine ⟨by decide, {(0, 5), (0, 6), (0, 7), (0, 8)}, by decide, by decide, 26, by decide, 25, by decide, ?_⟩
    have hU : ∀ q ∈ ({(0, 5), (0, 6), (0, 7), (0, 8)} : Finset (ℕ × ℕ)), tM.m q ⊆ {14, 15, 16, 17} := by decide
    intro q hq x hx
    have hx' := hU q hq hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx'
    rcases hx' with rfl | rfl | rfl | rfl
    · exact ⟨[14, 20, 23, 25], path4 (by decide), by decide⟩
    · exact ⟨[15, 20, 23, 25], path4 (by decide), by decide⟩
    · exact ⟨[16, 21, 23, 25], path4 (by decide), by decide⟩
    · exact ⟨[17, 21, 23, 25], path4 (by decide), by decide⟩

theorem tM_cutWidth : tM.CutWidth tA 4 2 := CutOfDisjointUpstreams tC tM tA 4 2 tM_disjointUpstreams

theorem tM_uses : tM.uses = {1} := by decide

theorem tM_upstream : tM.Upstream tA 8 where
  disjoint := by
    rw [tM_uses, Finset.coe_singleton]
    exact Set.pairwiseDisjoint_singleton _ _
  disjoint_θ := by decide
  upstream := by
    intro j hj
    rw [tM_uses, Finset.mem_singleton] at hj
    subst hj
    refine ⟨{(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)}, by decide, by decide, by decide, ?_⟩
    intro p hp _
    obtain rfl : p = (1, 9) := Finset.mem_singleton.1 hp
    refine ⟨26, by decide, 24, by decide, ?_⟩
    have hU : ∀ q ∈ ({(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)} : Finset (ℕ × ℕ)),
        tM.m q ⊆ {10, 11, 12, 13, 14, 15, 16, 17} := by decide
    intro q hq x hx
    have hx' := hU q hq hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx'
    rcases hx' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact ⟨[10, 18, 22, 24], path4 (by decide), by decide⟩
    · exact ⟨[11, 18, 22, 24], path4 (by decide), by decide⟩
    · exact ⟨[12, 19, 22, 24], path4 (by decide), by decide⟩
    · exact ⟨[13, 19, 22, 24], path4 (by decide), by decide⟩
    · exact ⟨[14, 20, 23, 24], path4 (by decide), by decide⟩
    · exact ⟨[15, 20, 23, 24], path4 (by decide), by decide⟩
    · exact ⟨[16, 21, 23, 24], path4 (by decide), by decide⟩
    · exact ⟨[17, 21, 23, 24], path4 (by decide), by decide⟩

theorem tI_blocks : ∀ I ∈ tI, tC.InferenceBlock 4 3 I := by
  intro I hI
  simp only [tI, Finset.mem_insert, Finset.mem_singleton] at hI
  rcases hI with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    decide

theorem tI_covers : tC.Covers tI := by decide

/-- The block handed `H1` (gate 22) and the four parameters of the other half: it computes that half, both activation
gates and the counted product. -/
def tS : Finset (Fin 27) := {0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26}

end CutWitness

open CutWitness in
/-- **Non-vacuity of the cut width**: some circuit with a counted product satisfies `ConditionalUpdateHardness`,
disjoint upstreams and the cut width at `f = 2`, the upstream certificate, and `k < b D`, and has a cover by inference
blocks (`CutWitness.tC`: `s = 4`, `k = 3`, `D = 4`, `b = 1`, sixteen blocks). -/
theorem cut_witness :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (A : ℕ → Finset (Fin C.N)) (s k D D₀ f : ℕ) (b : ℝ)
      (𝓘 : Finset (Finset (Fin C.N))),
      M.P.Nonempty ∧ 0 < b ∧ 2 ≤ f ∧ Assumptions.ConditionalUpdateHardness M A s b ∧ M.DisjointUpstreams A D f ∧
      M.CutWidth A D f ∧ M.Upstream A D₀ ∧ (k : ℝ) < b * D ∧ (∀ I ∈ 𝓘, C.InferenceBlock s k I) ∧ C.Covers 𝓘 :=
  ⟨tC, tM, tA, 4, 3, 4, 8, 2, 1, tI, tM_P_nonempty, one_pos, le_rfl, tM_conditional 4, tM_disjointUpstreams,
    tM_cutWidth, tM_upstream, by norm_num, tI_blocks, tI_covers⟩

open CutWitness in
/-- **The derived width on the witness**: at every block size, an inference block handed at most three bits outside `θ`
that computes the counted product whole is handed at least two gates of its activation region. -/
theorem cut_witness_width (s : ℕ) :
    ∀ I, tC.InferenceBlock s 3 I → ∀ p ∈ tM.P, tM.m p ⊆ tC.computed I → 2 ≤ (tC.inputs I ∩ tA p.1).card :=
  WidthOfCut tC tM tA s 3 4 2 1 one_pos (tM_hardness s) tM_cutWidth (by norm_num)

open CutWitness in
/-- **Tightness on the witness**: the block of the counted product, handed both activation gates and its parameter, is
handed exactly two gates of the region, and the cut width there is at most two (`cutWidth_le_reads`). -/
theorem cut_witness_tight :
    tC.InferenceBlock 4 3 {24, 25, 9, 26} ∧ tM.m (1, 9) ⊆ tC.computed {24, 25, 9, 26} ∧
      (tC.inputs {24, 25, 9, 26} ∩ tA 1).card = 2 ∧ ¬ tM.CutWidth tA 1 3 := by
  refine ⟨by decide, by decide, by decide, fun hW => ?_⟩
  have h := cutWidth_le_reads tM tA hW one_pos (p := (1, 9)) (by decide)
  have h2 : ((tM.m (1, 9)).biUnion tC.pred ∩ tA (1, 9).1).card = 2 := by decide
  omega

open CutWitness in
/-- **`Width` for unbounded blocks fails on the witness**: at `s = 17`, the block `CutWitness.tS`, handed gate 22 of the
region and six bits outside `θ` in all, computes the counted product whole while handed one gate of the region. What
the wiring gives is the width of blocks handed fewer than `b D` bits outside `θ` (`cut_witness_width`), not `Width`. -/
theorem width_unbounded_fails : ¬ Assumptions.Width tM tA 17 2 := by
  intro hW
  have h := hW tS (by decide) (by decide) (1, 9) (by decide) (by decide) (by decide)
  have h1 : (tC.inputs tS ∩ tA (1, 9).1).card = 1 := by decide
  omega

namespace SharedUpstream

/-- The wiring: the token 0 and the parameter bits 1–5 are input gates; `5 + l` is upstream product `l`; gate 10 adds
all four, the one gate both activation gates 11 and 12 read; 13 is the counted product. -/
def sPred : Fin 14 → Finset (Fin 14) :=
  ![∅, ∅, ∅, ∅, ∅, ∅, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {6, 7, 8, 9}, {10}, {10}, {5, 11, 12}]

abbrev sC : Circuit where
  N := 14
  pred := sPred
  pred_lt := by decide
  θ := {0}
  θ_input := by decide

def sm : ℕ × ℕ → Finset (Fin 14)
  | (0, 1) => {6}
  | (0, 2) => {7}
  | (0, 3) => {8}
  | (0, 4) => {9}
  | (1, 5) => {13}
  | _ => ∅

def so : ℕ × ℕ → Fin 14
  | (0, 1) => 6
  | (0, 2) => 7
  | (0, 3) => 8
  | (0, 4) => 9
  | _ => 13

abbrev sM : MulBlocks sC ℕ ℕ where
  Pall := {(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)}
  P := {(1, 5)}
  P_subset := by decide
  m := sm
  m_subcircuit := by decide
  m_computed := by decide
  m_disjoint := by
    have h : ∀ p ∈ ({(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)} : Finset (ℕ × ℕ)),
        ∀ q ∈ ({(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)} : Finset (ℕ × ℕ)), p ≠ q → Disjoint (sm p) (sm q) := by
      decide
    exact fun p hp q hq => h p hp q hq
  o := so
  m_outputs := by decide

def sA : ℕ → Finset (Fin 14) := fun j => if j = 1 then {6, 7, 8, 9, 10, 11, 12} else ∅

def sF : ℕ → Finset (Fin 14)
  | 1 => {1}
  | 2 => {2}
  | 3 => {3}
  | 4 => {4}
  | 5 => {5}
  | _ => ∅

theorem sM_hardness (s : ℕ) : Assumptions.UpdateHardness sM s 1 := by
  have hcard : ∀ p ∈ sM.Pall, (sF p.2).card = 1 := by decide
  refine updateHardness_of_conditional sM sA
    (Witness.conditionalUpdateHardness_of_feeds sM sA sF s (by decide) (by decide) (by decide) fun p hp => ?_)
  rw [hcard p hp, Nat.cast_one]

theorem sM_uses : sM.uses = {1} := by decide

theorem path3 {a b c : Fin 14} (h : [a, b, c].IsChain fun u v => u ∈ sC.pred v) : sC.IsPath [a, b, c] a c :=
  ⟨rfl, rfl, h⟩

/-- Every upstream product reaches activation gate `a` through gate 10, inside any region holding 6–10 and `a`. -/
theorem sM_reach (a : Fin 14) (ha : a = 11 ∨ a = 12) (R : Finset (Fin 14)) (hR : {6, 7, 8, 9, 10, a} ⊆ R) :
    ∀ q ∈ ({(0, 1), (0, 2), (0, 3), (0, 4)} : Finset (ℕ × ℕ)), ∀ x ∈ sM.m q, sC.ReachesIn R x a := by
  have hU : ∀ q ∈ ({(0, 1), (0, 2), (0, 3), (0, 4)} : Finset (ℕ × ℕ)), sM.m q ⊆ {6, 7, 8, 9} := by decide
  have hRv : ∀ v ∈ ({6, 7, 8, 9, 10, a} : Finset (Fin 14)), v ∈ R := fun v hv => hR hv
  simp only [Finset.mem_insert, Finset.mem_singleton, forall_eq_or_imp, forall_eq] at hRv
  obtain ⟨h6, h7, h8, h9, h10, haR⟩ := hRv
  intro q hq x hx
  have hx' := hU q hq hx
  simp only [Finset.mem_insert, Finset.mem_singleton] at hx'
  have hmem : ∀ y : Fin 14, (y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9) → ∀ v ∈ [y, 10, a], v ∈ R := by
    rintro y hy v hv
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl
    · rcases hy with rfl | rfl | rfl | rfl <;> assumption
    · exact h10
    · exact haR
  rcases ha with rfl | rfl <;> rcases hx' with rfl | rfl | rfl | rfl <;>
    exact ⟨_, path3 (by decide), hmem _ (by decide)⟩

theorem sM_upstream : sM.Upstream sA 4 where
  disjoint := by
    rw [sM_uses, Finset.coe_singleton]
    exact Set.pairwiseDisjoint_singleton _ _
  disjoint_θ := by decide
  upstream := by
    intro j hj
    rw [sM_uses, Finset.mem_singleton] at hj
    subst hj
    refine ⟨{(0, 1), (0, 2), (0, 3), (0, 4)}, by decide, by decide, by decide, ?_⟩
    intro p hp _
    obtain rfl : p = (1, 5) := Finset.mem_singleton.1 hp
    exact ⟨13, by decide, 11, by decide, sM_reach 11 (Or.inl rfl) _ (by decide)⟩

/-- The block handed gate 10 and the counted product's parameter: it computes both activation gates and the product. -/
def sS : Finset (Fin 14) := {5, 10, 11, 12, 13}

end SharedUpstream

open SharedUpstream in
/-- **The shared upstream**: two vertex-disjoint paths inside the region enter activation gates the counted product
reads, each from a gate whose own upstream inside the region minus the other path is over `D` updated parameters, under
`UpdateHardness`, the upstream certificate and `k < b D`; yet an inference block that computes the product whole is
handed one gate of the region. Both activation gates read one gate, which the block is handed. So neither `f`
vertex-disjoint paths into the activation (Menger) nor such paths with deep upstreams of their own give width `f`: what
counts is the upstream behind a cut (`MulBlocks.CutWidth`), and here one gate cuts all of it. -/
theorem shared_upstream :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (A : ℕ → Finset (Fin C.N)) (s k D : ℕ) (b : ℝ),
      0 < b ∧ Assumptions.UpdateHardness M s b ∧ M.Upstream A D ∧ (k : ℝ) < b * D ∧
      (∀ p ∈ M.P, ∃ π₁ π₂ : List (Fin C.N), ∃ d₁ d₂ : Fin C.N, ∃ g₁ ∈ M.m p, ∃ a₁ ∈ C.pred g₁,
        ∃ g₂ ∈ M.m p, ∃ a₂ ∈ C.pred g₂,
        C.IsPath π₁ d₁ a₁ ∧ C.IsPath π₂ d₂ a₂ ∧ (∀ v ∈ π₁, v ∈ A p.1) ∧ (∀ v ∈ π₂, v ∈ A p.1) ∧
        (∀ v ∈ π₁, v ∉ π₂) ∧
        (∃ U ⊆ M.Pall, D ≤ (U.image Prod.snd).card ∧ ∀ q ∈ U, ∀ x ∈ M.m q, C.ReachesIn (A p.1 \ π₂.toFinset) x d₁) ∧
        (∃ U ⊆ M.Pall, D ≤ (U.image Prod.snd).card ∧ ∀ q ∈ U, ∀ x ∈ M.m q, C.ReachesIn (A p.1 \ π₁.toFinset) x d₂)) ∧
      ∃ I, C.InferenceBlock s k I ∧ ∃ p ∈ M.P, M.m p ⊆ C.computed I ∧ (C.inputs I ∩ A p.1).card < 2 := by
  refine ⟨sC, sM, sA, 5, 3, 4, 1, one_pos, sM_hardness 5, sM_upstream, by norm_num, ?_, sS, by decide, (1, 5),
    by decide, by decide, by decide⟩
  intro p hp
  obtain rfl : p = (1, 5) := Finset.mem_singleton.1 hp
  refine ⟨[11], [12], 11, 12, 13, by decide, 11, by decide, 13, by decide, 12, by decide, ⟨rfl, rfl, by decide⟩,
    ⟨rfl, rfl, by decide⟩, by decide, by decide, by decide,
    ⟨{(0, 1), (0, 2), (0, 3), (0, 4)}, by decide, by decide, sM_reach 11 (Or.inl rfl) _ (by decide)⟩,
    ⟨{(0, 1), (0, 2), (0, 3), (0, 4)}, by decide, by decide, sM_reach 12 (Or.inr rfl) _ (by decide)⟩⟩

open SharedUpstream in
/-- On the shared-upstream circuit the cut width and disjoint upstreams at `f = 2` fail at depth 4, as `WidthOfCut`
requires: the block `SharedUpstream.sS` would otherwise be handed two gates of the region. -/
theorem shared_upstream_not_cutWidth : ¬ sM.CutWidth sA 4 2 ∧ ¬ sM.DisjointUpstreams sA 4 2 := by
  have hnot : ¬ sM.CutWidth sA 4 2 := fun hW => by
    have h := WidthOfCut sC sM sA 5 3 4 2 1 one_pos (sM_hardness 5) hW (by norm_num) sS (by decide) (1, 5)
      (by decide) (by decide)
    have h1 : (sC.inputs sS ∩ sA (1, 5).1).card = 1 := by decide
    omega
  exact ⟨hnot, fun hT => hnot (CutOfDisjointUpstreams sC sM sA 4 2 hT)⟩

end Nci.Cover.SecurityProofs
