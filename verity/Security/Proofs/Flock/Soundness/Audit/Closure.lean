import Proofs.Flock.Soundness.Audit.Work

/-!
# Closure draws

A closure draw proves each drawn unit together with its closure `cl t`, the units its credit depends on: for a PoUW tile,
the X and Y strips it reads and the node units on its X strip's path. The closure map is the verifier's own, from its
program.

* `Law.closure L cl`: `L` widened by `cl` on `L`'s coins (`Law.widen`). `mem_closure_draw`: a unit is proved when it is
  drawn or lies in a drawn unit's closure.
* `closure_escape`: **the closure draw's escape is the base law's escape of `B` with the units it makes unsound**,
  `(L.closure cl).escape B = L.escape (B ∪ unsoundTiles cl B)`, where `unsoundTiles cl B` are the units whose closure
  meets `B`. So a wrong unit that is never drawn on its own is charged to every unit above it.
* `unsoundWork σ w cl B`, the work of `B ∪ unsoundTiles cl B`: the wrong units and the units they make unsound, the set
  `closure_escape` measures. A closure map need not list a unit in its own closure (a tile's is its strips and path
  nodes), so the wrong units' own work is counted explicitly: `workOf_le_unsoundWork`.
* `harm σ w cl u`, the work of `u` and of the units whose closure holds it. `harm_le_unsoundWork`: **a wrong unit's whole
  harm is in the unsound work**, whether or not a draw reaches it. `unsoundWork_le_harm`: the unsound work is at most the
  wrong units' harm (coverage), and `unsoundWork_singleton`: exactly, for one wrong unit.
* `audit_work_closure`, `extraction_audit_work_closure`: under the work law's closure draw, at any floors `f_s ≥ 1`, the
  audit accepts while the unsound units do at least `T` of the work with probability at most
  `((W − T)/W)^K + ε_ks + δ_link`. `audit_work_closure_harm`: likewise while some wrong unit's harm is at least `T`.
  `audit_work_closure_of_record`: at `K = 27,713`, unsound work of at least 0.1% with at most `2^-40 + ε_ks + δ_link`.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- The units made unsound by the wrong-unit set `B`: those whose closure meets it. -/
def unsoundTiles (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) : Finset (Fin n) :=
  univ.filter fun t => ¬ Disjoint (cl t) B

/-- **Closure draws**: `L`'s draw, each drawn unit proved together with its closure. -/
def closure (L : Law n) (cl : Fin n → Finset (Fin n)) : Law n :=
  L.widen fun S => S ∪ S.biUnion cl

/-- **What a closure draw proves**: a unit is proved when it is drawn or lies in a drawn unit's closure. -/
theorem mem_closure_draw (L : Law n) (cl : Fin n → Finset (Fin n)) (ω : (L.closure cl).Ω) (u : Fin n) :
    u ∈ (L.closure cl).draw ω ↔ u ∈ L.draw ω ∨ ∃ t ∈ L.draw ω, u ∈ cl t := by
  show u ∈ L.draw ω ∪ (L.draw ω).biUnion cl ↔ _
  rw [mem_union, mem_biUnion]

/-- **The closure draw's escape** is the base law's escape of `B` with the units it makes unsound. -/
theorem closure_escape (L : Law n) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) :
    (L.closure cl).escape B = L.escape (B ∪ unsoundTiles cl B) := by
  show prCoin (fun ω : L.Ω => Disjoint (L.draw ω ∪ (L.draw ω).biUnion cl) B) =
    prCoin (fun ω : L.Ω => Disjoint (L.draw ω) (B ∪ unsoundTiles cl B))
  congr 1
  funext ω
  apply propext
  rw [disjoint_union_left, disjoint_biUnion_left, disjoint_union_right]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨h1, disjoint_left.2 fun t ht htU => (mem_filter.1 htU).2 (h2 t ht)⟩
  · rintro ⟨h1, h2⟩
    refine ⟨h1, fun t ht => ?_⟩
    by_contra hc
    exact disjoint_left.1 h2 ht (mem_filter.2 ⟨mem_univ t, hc⟩)

/-- The work of the wrong units `B` and of the units they make unsound. -/
abbrev unsoundWork (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) : ℕ :=
  workOf σ w (B ∪ unsoundTiles cl B)

/-- **The wrong units' own work is in the unsound work**, whatever the closure map. -/
theorem workOf_le_unsoundWork (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) :
    workOf σ w B ≤ unsoundWork σ w cl B :=
  sum_le_sum_of_subset subset_union_left

/-- **The harm of a unit**: its own work and the work of every unit whose closure holds it, what it spoils when wrong. -/
def harm (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (u : Fin n) : ℕ :=
  workOf σ w (insert u (univ.filter fun t => u ∈ cl t))

/-- **A wrong unit's whole harm is in the unsound work**, whether or not a draw reaches it: it is wrong itself, and every
unit whose closure holds it is unsound. -/
theorem harm_le_unsoundWork (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) {B : Finset (Fin n)}
    {u : Fin n} (hu : u ∈ B) : harm σ w cl u ≤ unsoundWork σ w cl B := by
  unfold harm unsoundWork unsoundTiles workOf
  refine sum_le_sum_of_subset fun t ht => ?_
  rcases mem_insert.1 ht with rfl | ht
  · exact mem_union_left _ hu
  · exact mem_union_right _ (mem_filter.2 ⟨mem_univ t, not_disjoint_iff.2 ⟨u, (mem_filter.1 ht).2, hu⟩⟩)

/-- **Coverage**: the unsound work is at most the wrong units' harm. Each wrong unit is charged to itself and each unsound
unit to a wrong unit of its closure; overlaps only overcount. -/
theorem unsoundWork_le_harm (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) :
    unsoundWork σ w cl B ≤ ∑ u ∈ B, harm σ w cl u := by
  classical
  have e : ∀ S : Finset (Fin n), workOf σ w S = ∑ t, if t ∈ S then w (σ t) else 0 := fun S => by
    rw [sum_ite_mem, univ_inter, workOf]
  unfold unsoundWork harm
  simp only [e]
  rw [sum_comm]
  refine sum_le_sum fun t _ => ?_
  split_ifs with h
  · obtain ⟨u, hu, htu⟩ : ∃ u ∈ B, t ∈ insert u (univ.filter fun t => u ∈ cl t) := by
      rcases mem_union.1 h with htB | htU
      · exact ⟨t, htB, mem_insert_self _ _⟩
      · obtain ⟨u, hu, huB⟩ := not_disjoint_iff.1 (mem_filter.1 htU).2
        exact ⟨u, huB, mem_insert_of_mem (mem_filter.2 ⟨mem_univ t, hu⟩)⟩
    calc w (σ t) = (if t ∈ insert u (univ.filter fun t => u ∈ cl t) then w (σ t) else 0) := by simp only [htu, ite_true]
      _ ≤ ∑ u ∈ B, if t ∈ insert u (univ.filter fun t => u ∈ cl t) then w (σ t) else 0 :=
          single_le_sum (f := fun u => if t ∈ insert u (univ.filter fun t => u ∈ cl t) then w (σ t) else 0)
            (fun _ _ => Nat.zero_le _) hu
  · exact Nat.zero_le _

/-- **The charging is exact for one wrong unit.** -/
theorem unsoundWork_singleton (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (u : Fin n) :
    unsoundWork σ w cl {u} = harm σ w cl u := by
  unfold unsoundWork unsoundTiles harm
  congr 1
  ext t
  simp [disjoint_singleton_right]

/-- Under the work law's closure draw, a set whose unsound work is at least `T` escapes with at most `((W − T)/W)^K`. -/
theorem closure_escape_le_work (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (cl : Fin n → Finset (Fin n))
    (hf : ∀ s, 1 ≤ f s) (hW : 0 < totalWork σ w) {B : Finset (Fin n)} {T : ℕ} (hT : T ≤ unsoundWork σ w cl B) :
    ((work σ w f K).closure cl).escape B ≤ (((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  rw [closure_escape]
  exact work_escape_le_of_le σ w f K hf hW hT

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ} {cl : Fin n → Finset (Fin n)}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- **The work audit with closure draws** (oracle layer): the audit accepts while its wrong units and the units they make
unsound do at least `T` of the work with probability at most `((W − T)/W)^K + ε_ks + δ_link`. -/
theorem audit_work_closure (A : Analysis ((Law.work σ w f K).closure cl) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hf : ∀ s, 1 ≤ f s) (hW : 0 < Law.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit ((Law.work σ w f K).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')))
        (audit ((Law.work σ w f K).closure cl) Reg session) σ' ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => T ≤ Law.unsoundWork σ w cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.closure_escape_le_work σ w f K cl hf hW hB

/-- **The work audit with closure draws** (compiled layer, randomized extractor): at most
`((W − T)/W)^K + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_work_closure (A : ExtractionAnalysis ((Law.work σ w f K).closure cl) Reg session)
    {εks δlink : (R : Reg) → Cont ((Law.work σ w f K).closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hf : ∀ s, 1 ≤ f s) (hW : 0 < Law.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit ((Law.work σ w f K).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')))
        (audit ((Law.work σ w f K).closure cl) Reg session) σ' ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ Law.unsoundWork σ w cl (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (Law.closure_escape_le_work σ w f K cl hf hW hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **Each wrong unit is charged its harm** (oracle layer): the audit accepts while some wrong unit, drawn or not, has
harm at least `T` with probability at most `((W − T)/W)^K + ε_ks + δ_link`. -/
theorem audit_work_closure_harm (A : Analysis ((Law.work σ w f K).closure cl) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hf : ∀ s, 1 ≤ f s) (hW : 0 < Law.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit ((Law.work σ w f K).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ ∃ u ∈ A.wrong (A.committedOf σ'), T ≤ Law.harm σ w cl u)
        (audit ((Law.work σ w f K).closure cl) Reg session) σ' ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K + εks + δlink := by
  refine le_trans (prob_mono (fun o ho => ⟨ho.1, ?_⟩) _ _) (audit_work_closure A hks hlink hf hW T σ')
  obtain ⟨u, hu, hT⟩ := ho.2
  exact hT.trans (Law.harm_le_unsoundWork σ w cl hu)

/-- **The closure audit at the sizing of record**: with `K = 27,713`, the audit accepts while its wrong units and the units
they make unsound do at least 0.1% of the work with probability at most `2^-40 + ε_ks + δ_link`. -/
theorem audit_work_closure_of_record (A : Analysis ((Law.work σ w f 27713).closure cl) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hf : ∀ s, 1 ≤ f s) (hW : 0 < Law.totalWork σ w)
    (σ' : Strategy (audit ((Law.work σ w f 27713).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')))
        (audit ((Law.work σ w f 27713).closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl B) σ').trans ?_
  gcongr
  refine iSup₂_le fun B hB => ?_
  rw [Law.closure_escape]
  exact Law.work_escape_of_record σ w f hf hW hB

end Audits

end FlockSoundness.Audit
