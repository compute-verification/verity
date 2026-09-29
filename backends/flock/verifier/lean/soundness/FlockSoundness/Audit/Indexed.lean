import FlockSoundness.Audit.Window

/-!
# The audit with a receipt-indexed law

The live window draw depends on the registration: the verifier derives its key from a fresh uniform source and the
receipt, `D(source, digest(R))`. As a game that is the one-stage audit with the law indexed by the registration,
`.send Reg fun R => .coin (L R).Ω fun ω => session ((L R).draw ω) R` (`auditIndexed`).

* `audit_indexed_le`, `audit_indexed_profile`: `audit_le` and `audit_profile` for `L : Reg → Law n`, bounded by the
  registered law's escapes. The committed transcript is still fixed before the draw, now for each registration.
* `extraction_audit_indexed_le`: the same with a randomized extractor.
* `audit_indexed_window_of_le_slack`, `audit_indexed_window_split_of_record_of_le_slack` and
  `extraction_audit_indexed_window_of_le_slack`: the window audits for `hL : ∀ R B, (L R).escape B ≤ stratified.escape B
  + η`, the form a keyed-stream assumption gives with one `η` for every receipt.

The uniformity is in the hypotheses: one `η` for every receipt, and at the oracle layer one `ε_ks` and `δ_link` across
registrations. **The fixed-law audit is the constant case**, `L R = L` (`auditIndexed_const`, and the examples below),
so the fixed-law pins are unchanged.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **The one-stage audit with a receipt-indexed law.** The prover registers `R : Reg`; the verifier draws `S` by
`L R`; the session on the drawn units runs against `R`. Output: the verdict and the draw. -/
def auditIndexed (L : Reg → Law n) (session : Finset (Fin n) → Reg → Game Bool) : Game (Bool × Finset (Fin n)) :=
  .send Reg fun R => .coin (L R).Ω fun ω => (session ((L R).draw ω) R).map fun ok => (ok, (L R).draw ω)

/-- The fixed-law audit is the constant case. -/
theorem auditIndexed_const (L : Law n) (session : Finset (Fin n) → Reg → Game Bool) :
    auditIndexed (fun _ : Reg => L) session = audit L Reg session := rfl

variable {L : Reg → Law n} {session : Finset (Fin n) → Reg → Game Bool}

/-- A prover's state after registering `R`: its session strategy for every draw of `L R`. -/
abbrev ContIndexed (L : Reg → Law n) (session : Finset (Fin n) → Reg → Game Bool) (R : Reg) :=
  ∀ ω : (L R).Ω, Strategy (session ((L R).draw ω) R)

/-- The registration of an audit strategy. -/
def regIndexed (σ : Strategy (auditIndexed L session)) : Reg := σ.1

/-- The prover's state after registration. -/
def contIndexed (σ : Strategy (auditIndexed L session)) : ContIndexed L session (regIndexed σ) :=
  fun ω => Strategy.ofMap _ _ (σ.2 ω)

/-- What the analysis of an indexed audit fixes, as `Analysis`: the committed transcript is a function of the
prover's state at registration. -/
structure AnalysisIndexed (L : Reg → Law n) (session : Finset (Fin n) → Reg → Game Bool) where
  Tr : Type
  wrong : Tr → Finset (Fin n)
  committed : (R : Reg) → ContIndexed L session R → Tr
  ksFail : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Prop
  linkFail : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Tr → Prop
  cover : ∀ S R (τ : Strategy (session S R)) u X, u ∈ wrong X → ksFail S R τ u ∨ linkFail S R τ u X

namespace AnalysisIndexed

variable (A : AnalysisIndexed L session)

/-- The committed transcript of an audit strategy. -/
def committedOf (σ : Strategy (auditIndexed L session)) : A.Tr := A.committed (regIndexed σ) (contIndexed σ)

/-- **Knowledge soundness, per unit** (named hypothesis `ε_ks`), as `Analysis.KnowledgeSound`, over each registration's
law: one `ε_ks` for every registration. -/
def KnowledgeSound (εks : ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : ContIndexed L session R) (tgt : (L R).Ω → Option (Fin n)),
    avg (fun ω => prob (fun ok => ok = true ∧
      ∃ u, tgt ω = some u ∧ u ∈ (L R).draw ω ∧ A.ksFail ((L R).draw ω) R (τ ω) u)
      (session ((L R).draw ω) R) (τ ω)) ≤ εks

/-- **The link term** (named hypothesis `δ_link`), as `Analysis.LinkSound`, over each registration's law. -/
def LinkSound (δlink : ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : ContIndexed L session R) (tgt : (L R).Ω → Option (Fin n)),
    avg (fun ω => prob (fun ok => ok = true ∧
      ∃ u, tgt ω = some u ∧ u ∈ (L R).draw ω ∧ A.linkFail ((L R).draw ω) R (τ ω) u (A.committed R τ))
      (session ((L R).draw ω) R) (τ ω)) ≤ δlink

end AnalysisIndexed

/-- A fixed-law analysis is an analysis of the constant indexed law. -/
def Analysis.toIndexed {L : Law n} (A : Analysis L Reg session) : AnalysisIndexed (fun _ : Reg => L) session where
  Tr := A.Tr
  wrong := A.wrong
  committed := A.committed
  ksFail := A.ksFail
  linkFail := A.linkFail
  cover := A.cover

/-- **The one-stage audit bound, with a receipt-indexed law**: `audit_le` over the registered law `L (reg σ)`. -/
theorem audit_indexed_le (A : AnalysisIndexed L session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) (E : A.Tr → Finset (Fin n) → Prop) (σ : Strategy (auditIndexed L session)) :
    prob (fun o => o.1 = true ∧ E (A.committedOf σ) o.2) (auditIndexed L session) σ ≤
      prCoin (fun ω : (L (regIndexed σ)).Ω => E (A.committedOf σ) ((L (regIndexed σ)).draw ω) ∧
        Disjoint ((L (regIndexed σ)).draw ω) (A.wrong (A.committedOf σ))) + εks + δlink := by
  classical
  obtain ⟨R, τ'⟩ := σ
  set τ : ContIndexed L session R := contIndexed (L := L) ⟨R, τ'⟩
  set X := A.committed R τ with hX
  show prob _ (auditIndexed L session) ⟨R, τ'⟩ ≤
    prCoin (fun ω => E X ((L R).draw ω) ∧ Disjoint ((L R).draw ω) (A.wrong X)) + εks + δlink
  let tgt : (L R).Ω → Option (Fin n) := fun ω =>
    if h : ∃ u, u ∈ (L R).draw ω ∧ u ∈ A.wrong X then some (Classical.choose h) else none
  let ksE : (L R).Ω → Bool → Prop := fun ω ok => ok = true ∧
    ∃ u, tgt ω = some u ∧ u ∈ (L R).draw ω ∧ A.ksFail ((L R).draw ω) R (τ ω) u
  let lkE : (L R).Ω → Bool → Prop := fun ω ok => ok = true ∧
    ∃ u, tgt ω = some u ∧ u ∈ (L R).draw ω ∧ A.linkFail ((L R).draw ω) R (τ ω) u X
  have step : ∀ ω, prob (fun ok => ok = true ∧ E X ((L R).draw ω)) (session ((L R).draw ω) R) (τ ω) ≤
      (if E X ((L R).draw ω) ∧ Disjoint ((L R).draw ω) (A.wrong X) then 1 else 0) +
        prob (ksE ω) (session ((L R).draw ω) R) (τ ω) + prob (lkE ω) (session ((L R).draw ω) R) (τ ω) := by
    intro ω
    by_cases hd : Disjoint ((L R).draw ω) (A.wrong X)
    · rw [prob_and_const]
      by_cases hE : E X ((L R).draw ω)
      · simp only [hE, hd, and_self, ite_true]
        exact (prob_le_one _ _ _).trans (le_add_right le_self_add)
      · simp only [hE, ite_false, false_and, zero_le]
    · have h : ∃ u, u ∈ (L R).draw ω ∧ u ∈ A.wrong X := by
        obtain ⟨u, hu, hu'⟩ := Finset.not_disjoint_iff.1 hd
        exact ⟨u, hu, hu'⟩
      have ht : tgt ω = some (Classical.choose h) := by simp only [tgt, h, dite_true]
      obtain ⟨hmem, hw⟩ := Classical.choose_spec h
      have hc := A.cover ((L R).draw ω) R (τ ω) _ X hw
      calc prob (fun ok => ok = true ∧ E X ((L R).draw ω)) (session ((L R).draw ω) R) (τ ω)
          ≤ prob (fun ok => ksE ω ok ∨ lkE ω ok) (session ((L R).draw ω) R) (τ ω) :=
            prob_mono (fun ok hok => by
              rcases hc with hk | hl
              · exact Or.inl ⟨hok.1, _, ht, hmem, hk⟩
              · exact Or.inr ⟨hok.1, _, ht, hmem, hl⟩) _ _
        _ ≤ prob (ksE ω) _ _ + prob (lkE ω) _ _ := prob_or_le _ _ _ _
        _ ≤ _ := by rw [add_assoc]; exact le_add_left le_rfl
  calc prob (fun o => o.1 = true ∧ E X o.2) (auditIndexed L session) ⟨R, τ'⟩
      = avg fun ω => prob (fun ok => ok = true ∧ E X ((L R).draw ω)) (session ((L R).draw ω) R) (τ ω) := by
        exact congrArg avg (funext fun ω => prob_map _ _ _ _)
    _ ≤ avg fun ω => (if E X ((L R).draw ω) ∧ Disjoint ((L R).draw ω) (A.wrong X) then 1 else 0) +
        prob (ksE ω) (session ((L R).draw ω) R) (τ ω) + prob (lkE ω) (session ((L R).draw ω) R) (τ ω) :=
        avg_mono step
    _ = _ := avg_add₃ _ _ _
    _ ≤ _ := add_le_add (add_le_add (avg_ite_eq_prCoin _).le (hks R τ tgt)) (hlink R τ tgt)

/-- **The integrity profile, with a receipt-indexed law**: `Pr[accept ∧ wrong(X) ∈ 𝓑] ≤
sup_{B ∈ 𝓑} escape_{L R} B + ε_ks + δ_link`, `R` the strategy's registration. -/
theorem audit_indexed_profile (A : AnalysisIndexed L session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) (𝓑 : Finset (Fin n) → Prop) (σ : Strategy (auditIndexed L session)) :
    prob (fun o => o.1 = true ∧ 𝓑 (A.wrong (A.committedOf σ))) (auditIndexed L session) σ ≤
      (⨆ (B : Finset (Fin n)) (_ : 𝓑 B), (L (regIndexed σ)).escape B) + εks + δlink := by
  refine (audit_indexed_le A hks hlink (fun X _ => 𝓑 (A.wrong X)) σ).trans ?_
  gcongr
  by_cases hB : 𝓑 (A.wrong (A.committedOf σ))
  · exact (prCoin_mono fun ω h => h.2).trans
      (le_iSup₂_of_le (f := fun B (_ : 𝓑 B) => (L (regIndexed σ)).escape B) _ hB le_rfl)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => absurd h.1 hB).trans (prCoin_false.le.trans zero_le)

/-- The constant case of `audit_indexed_profile` is `audit_profile`. -/
example {L : Law n} (A : Analysis L Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) (𝓑 : Finset (Fin n) → Prop) (σ : Strategy (audit L Reg session)) :
    prob (fun o => o.1 = true ∧ 𝓑 (A.wrong (A.committedOf σ))) (audit L Reg session) σ ≤
      (⨆ (B : Finset (Fin n)) (_ : 𝓑 B), L.escape B) + εks + δlink :=
  audit_indexed_profile A.toIndexed hks hlink 𝓑 σ

/-! ## With a randomized extractor -/

/-- What the analysis of an indexed audit fixes when extraction is randomized, as `ExtractionAnalysis`. -/
structure ExtractionAnalysisIndexed (L : Reg → Law n) (session : Finset (Fin n) → Reg → Game Bool) where
  Tr : Type
  wrong : Tr → Finset (Fin n)
  committed : (R : Reg) → ContIndexed L session R → Tr
  /-- `Pr[the session accepts ∧ extraction of u fails]` -/
  ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞
  /-- `Pr[it accepts ∧ extraction of u succeeds but differs from X on u's wires]` -/
  link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Tr → ℝ≥0∞
  cover : ∀ S R (τ : Strategy (session S R)) u X, u ∈ S → u ∈ wrong X →
    prob (· = true) (session S R) τ ≤ ks S R τ u + link S R τ u X

namespace ExtractionAnalysisIndexed

variable (A : ExtractionAnalysisIndexed L session)

/-- The committed transcript of an audit strategy. -/
def committedOf (σ : Strategy (auditIndexed L session)) : A.Tr := A.committed (regIndexed σ) (contIndexed σ)

/-- **Knowledge soundness, per unit** (named hypothesis `ε_ks`, which may depend on the prover's state), over each
registration's law. -/
def KnowledgeSound (εks : (R : Reg) → ContIndexed L session R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : ContIndexed L session R) (tgt : (L R).Ω → Option (Fin n)),
    avg (fun ω => atTgt ((L R).draw ω) (tgt ω) (A.ks ((L R).draw ω) R (τ ω))) ≤ εks R τ

/-- **The link term** (named hypothesis `δ_link`, which may depend on the prover's state), over each registration's
law. -/
def LinkSound (δlink : (R : Reg) → ContIndexed L session R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : ContIndexed L session R) (tgt : (L R).Ω → Option (Fin n)),
    avg (fun ω => atTgt ((L R).draw ω) (tgt ω) fun u => A.link ((L R).draw ω) R (τ ω) u (A.committed R τ)) ≤
      δlink R τ

end ExtractionAnalysisIndexed

/-- **The one-stage audit bound with a randomized extractor and a receipt-indexed law**: `extraction_audit_le` over
the registered law. -/
theorem extraction_audit_indexed_le (A : ExtractionAnalysisIndexed L session)
    {εks δlink : (R : Reg) → ContIndexed L session R → ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) (E : A.Tr → Finset (Fin n) → Prop) (σ : Strategy (auditIndexed L session)) :
    prob (fun o => o.1 = true ∧ E (A.committedOf σ) o.2) (auditIndexed L session) σ ≤
      prCoin (fun ω : (L (regIndexed σ)).Ω => E (A.committedOf σ) ((L (regIndexed σ)).draw ω) ∧
        Disjoint ((L (regIndexed σ)).draw ω) (A.wrong (A.committedOf σ))) +
      εks (regIndexed σ) (contIndexed σ) + δlink (regIndexed σ) (contIndexed σ) := by
  classical
  obtain ⟨R, τ'⟩ := σ
  set τ : ContIndexed L session R := contIndexed (L := L) ⟨R, τ'⟩
  set X := A.committed R τ with hX
  show prob _ (auditIndexed L session) ⟨R, τ'⟩ ≤
    prCoin (fun ω => E X ((L R).draw ω) ∧ Disjoint ((L R).draw ω) (A.wrong X)) + εks R τ + δlink R τ
  let tgt : (L R).Ω → Option (Fin n) := fun ω =>
    if h : ∃ u, u ∈ (L R).draw ω ∧ u ∈ A.wrong X then some (Classical.choose h) else none
  let ks : (L R).Ω → ℝ≥0∞ := fun ω => atTgt ((L R).draw ω) (tgt ω) (A.ks ((L R).draw ω) R (τ ω))
  let lk : (L R).Ω → ℝ≥0∞ := fun ω => atTgt ((L R).draw ω) (tgt ω) fun u => A.link ((L R).draw ω) R (τ ω) u X
  have step : ∀ ω, prob (fun ok => ok = true ∧ E X ((L R).draw ω)) (session ((L R).draw ω) R) (τ ω) ≤
      (if E X ((L R).draw ω) ∧ Disjoint ((L R).draw ω) (A.wrong X) then 1 else 0) + ks ω + lk ω := by
    intro ω
    by_cases hd : Disjoint ((L R).draw ω) (A.wrong X)
    · rw [prob_and_const]
      by_cases hE : E X ((L R).draw ω)
      · simp only [hE, hd, and_self, ite_true]
        exact (prob_le_one _ _ _).trans (le_add_right le_self_add)
      · simp only [hE, ite_false, false_and, zero_le]
    · have h : ∃ u, u ∈ (L R).draw ω ∧ u ∈ A.wrong X := by
        obtain ⟨u, hu, hu'⟩ := Finset.not_disjoint_iff.1 hd
        exact ⟨u, hu, hu'⟩
      have ht : tgt ω = some (Classical.choose h) := by simp only [tgt, h, dite_true]
      obtain ⟨hmem, hw⟩ := Classical.choose_spec h
      have hc := A.cover ((L R).draw ω) R (τ ω) _ X hmem hw
      have hks : ks ω = A.ks ((L R).draw ω) R (τ ω) (Classical.choose h) := by
        simp [ks, atTgt, ht, hmem]
      have hlk : lk ω = A.link ((L R).draw ω) R (τ ω) (Classical.choose h) X := by
        simp [lk, atTgt, ht, hmem]
      calc prob (fun ok => ok = true ∧ E X ((L R).draw ω)) (session ((L R).draw ω) R) (τ ω)
          ≤ prob (· = true) (session ((L R).draw ω) R) (τ ω) := prob_mono (fun ok h => h.1) _ _
        _ ≤ ks ω + lk ω := by rw [hks, hlk]; exact hc
        _ ≤ _ := by rw [add_assoc]; exact le_add_left le_rfl
  calc prob (fun o => o.1 = true ∧ E X o.2) (auditIndexed L session) ⟨R, τ'⟩
      = avg fun ω => prob (fun ok => ok = true ∧ E X ((L R).draw ω)) (session ((L R).draw ω) R) (τ ω) :=
        congrArg avg (funext fun ω => prob_map _ _ _ _)
    _ ≤ avg fun ω => (if E X ((L R).draw ω) ∧ Disjoint ((L R).draw ω) (A.wrong X) then 1 else 0) +
        ks ω + lk ω := avg_mono step
    _ = _ := avg_add₃ _ _ _
    _ ≤ _ := add_le_add (add_le_add (avg_ite_eq_prCoin _).le (hks R τ tgt)) (hlink R τ tgt)

/-! ## Windows -/

section Windows

variable {m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (Law.stratum σ s).card}
  {cl : Fin n → Finset (Fin n)}

/-- **The window audit with a receipt-indexed law** (oracle layer): every registration's law within `η` of the product
law, drawn with closures. At most `max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + η + ε_ks + δ_link`. -/
theorem audit_indexed_window_of_le_slack {η : ℝ≥0∞}
    (hL : ∀ R B, (L R).escape B ≤ (Law.stratified σ k hk).escape B + η)
    (A : AnalysisIndexed (fun R => (L R).closure cl) session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K)
    (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (auditIndexed (fun R => (L R).closure cl) session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (auditIndexed (fun R => (L R).closure cl) session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + η + εks + δlink := by
  refine (audit_indexed_profile A hks hlink
    (fun B => T ≤ Law.unsoundWork σ w cl B ∨ Ty ≤ Law.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.closure_escape_le_window_slack σ k hk (hL (regIndexed σ')) cl hW hc hN hcy hB

/-- The constant case of `audit_indexed_window_of_le_slack` is `audit_window_of_le_slack`. -/
example {L₀ : Law n} {η : ℝ≥0∞} (hL : ∀ B, L₀.escape B ≤ (Law.stratified σ k hk).escape B + η)
    (A : Analysis (L₀.closure cl) Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K)
    (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (audit (L₀.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L₀.closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + η + εks + δlink :=
  audit_indexed_window_of_le_slack (L := fun _ => L₀) (fun _ => hL) A.toIndexed hks hlink hW hc hN hcy T Ty σ'

/-- **The per-call split with y's floors, with a receipt-indexed law, at the sizing of record**: every registration's
law within `η` of the per-call split's product law, at most `2^-40 + η + ε_ks + δ_link`. -/
theorem audit_indexed_window_split_of_record_of_le_slack {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ}
    {η : ℝ≥0∞}
    (hL : ∀ R B, (L R).escape B ≤
      (Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).escape B + η)
    (A : AnalysisIndexed (fun R => (L R).closure cl) session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) (hW : 0 < Law.totalWork σ w)
    (hone : ∀ s, 0 < w s → Law.callWork σ call w (call s) = w s * (Law.stratum σ s).card)
    (hN : 0 < Law.totalWork σ v) (hfy : ∀ s, 27713 * v s * (Law.stratum σ s).card ≤ f s * Law.totalWork σ v)
    (σ' : Strategy (auditIndexed (fun R => (L R).closure cl) session)) :
    prob (fun o => o.1 = true ∧ (Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (auditIndexed (fun R => (L R).closure cl) session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks + δlink := by
  have hc := Law.covers_window σ call w f 27713 hW hone
  have hcy := Law.covers_of_floor σ _ f v 27713 (Law.windowK_le σ call w f 27713)
    (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy
  refine (audit_indexed_profile A hks hlink (fun B => Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl B ∨
    Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  refine iSup₂_le fun B hB => ?_
  show ((L (regIndexed σ')).closure cl).escape B ≤ _
  rw [Law.closure_escape]
  refine (hL _ _).trans (add_le_add ?_ le_rfl)
  rcases hB with hB | hB
  · exact (Law.stratified_escape_le_of_covers σ _ _ w 27713 hW hc _).trans (Law.pow_record_le hW hB)
  · exact (Law.stratified_escape_le_of_covers σ _ _ v 27713 hN hcy _).trans (Law.pow_record_le hN hB)

/-- **The window audit with a receipt-indexed law** (compiled layer, randomized extractor): at most
`max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + η + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_indexed_window_of_le_slack {η : ℝ≥0∞}
    (hL : ∀ R B, (L R).escape B ≤ (Law.stratified σ k hk).escape B + η)
    (A : ExtractionAnalysisIndexed (fun R => (L R).closure cl) session)
    {εks δlink : (R : Reg) → ContIndexed (fun R => (L R).closure cl) session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K) (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (auditIndexed (fun R => (L R).closure cl) session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (auditIndexed (fun R => (L R).closure cl) session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + η +
        εks (regIndexed σ') (contIndexed σ') + δlink (regIndexed σ') (contIndexed σ') := by
  refine (extraction_audit_indexed_le A hks hlink (fun X _ => T ≤ Law.unsoundWork σ w cl (A.wrong X) ∨
    Ty ≤ Law.unsoundWork σ v cl (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
      Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans
      (Law.closure_escape_le_window_slack σ k hk (hL (regIndexed σ')) cl hW hc hN hcy hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

end Windows

end FlockSoundness.Audit
