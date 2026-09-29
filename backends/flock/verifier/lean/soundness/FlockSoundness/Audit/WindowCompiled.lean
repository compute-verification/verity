import FlockSoundness.Audit.Window

/-!
# The window audit for a law within `η` of the product law, at the compiled layer

`Audit/Window.lean`'s slack forms (`audit_window_of_le_slack`, `audit_window_split_of_record_of_le_slack`, #421) are at the
oracle layer. These are their compiled-layer forms, for a randomized extractor (`ExtractionAnalysis`), with the same
proofs through `extraction_audit_le` in place of `audit_profile`. The PoUW keyed draw's claim of record uses them
(`KeyedDraw.lean`).
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (Law.stratum σ s).card}
  {cl : Fin n → Finset (Fin n)} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- **The window audit for a law within `η` of the product law** (compiled layer, randomized extractor): at most
`max(((W − T)/W)^K, ((N − T_y)/N)^(K_y)) + η + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_window_of_le_slack {L : Law n} {η : ℝ≥0∞}
    (hL : ∀ B, L.escape B ≤ (Law.stratified σ k hk).escape B + η) (A : ExtractionAnalysis (L.closure cl) Reg session)
    {εks δlink : (R : Reg) → Cont (L.closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < Law.totalWork σ w) (hc : Law.Covers σ k w K) (hN : 0 < Law.totalWork σ v) (hcy : Law.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((Law.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (Law.totalWork σ v : ℝ≥0∞)) ^ Ky) + η +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ Law.unsoundWork σ w cl (A.wrong X) ∨
    Ty ≤ Law.unsoundWork σ v cl (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
      Ty ≤ Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (Law.closure_escape_le_window_slack σ k hk hL cl hW hc hN hcy hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **The per-call split with y's floors, for a law within `η` of it, at the sizing of record** (compiled layer): at most
`2^-40 + η + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_window_split_of_record_of_le_slack {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ}
    {L : Law n} {η : ℝ≥0∞}
    (hL : ∀ B, L.escape B ≤
      (Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).escape B + η)
    (A : ExtractionAnalysis (L.closure cl) Reg session)
    {εks δlink : (R : Reg) → Cont (L.closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hW : 0 < Law.totalWork σ w)
    (hone : ∀ s, 0 < w s → Law.callWork σ call w (call s) = w s * (Law.stratum σ s).card)
    (hN : 0 < Law.totalWork σ v) (hfy : ∀ s, 27713 * v s * (Law.stratum σ s).card ≤ f s * Law.totalWork σ v)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  have hc := Law.covers_window σ call w f 27713 hW hone
  have hcy := Law.covers_of_floor σ _ f v 27713 (Law.windowK_le σ call w f 27713)
    (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy
  refine (extraction_audit_le A hks hlink (fun X _ => Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong X) ∨
    Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
      Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))
  · refine (prCoin_mono fun ω h => h.2).trans ?_
    change (L.closure cl).escape _ ≤ _
    rw [Law.closure_escape]
    refine (hL _).trans (add_le_add ?_ le_rfl)
    rcases hB with hB | hB
    · exact (Law.stratified_escape_le_of_covers σ _ _ w 27713 hW hc _).trans (Law.pow_record_le hW hB)
    · exact (Law.stratified_escape_le_of_covers σ _ _ v 27713 hN hcy _).trans (Law.pow_record_le hN hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

end Audits

end FlockSoundness.Audit
