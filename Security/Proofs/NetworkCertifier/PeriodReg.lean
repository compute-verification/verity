import Definitions.NetworkCertifier.Guarantees.Period
import Proofs.NetworkCertifier.Period
import Proofs.NetworkCertifier.SampledLeakReg

/-!
# The leak through the network certifier over a monitoring period, at the registration's values

On the audit's scope, the sampling relation at a trace that agrees with the network certifier wherever both open a
transfer gate, and with the registration at every registered input it opens, gives the leak relation at the network
certifier's rows with the registered inputs pinned at the registration's values (`rLeakTapReg_of_r`): the trace
filled in with the network certifier's values carries its records, so the circuit accepts at that trace's inputs, and
an input outside the faulty units is opened, at the registration's value where it is registered. A trace that
disagrees with either makes a collision with its openings (`TapProof.tapColl_of_disagree`, `regColl_of_disagree`), so
each window's failure is at most `δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]` (`sampledLeak_tapReg`). A window that
meets `WindowGoodReg` has its observations among the sequences `AccSeqsSampled` counts at its registration's values
(`mem_accSeqsSampled_of_windowGoodReg`), and `encard_period_le` counts them. Like `periodLeak`, the first part goes
through `R` and `KS` directly, as `TapProof.sampledLeak_tap` does, not through `SampledAuditAccepted`.
-/

namespace NetTiming

open Claims Claims.Sampling NetTiming.SampledAuditProof Finset

section Deploy

variable {Coin Col T : Type}

/-- **At the network certifier's rows and the registration's values**: on the audit's scope at `δ ≥ 0`, the sampling
relation at a trace that agrees with the network certifier wherever both open a gate, and with the registration's
values `regVal (ops s)` at every registered input it opens, gives the leak relation at the network certifier's rows
with the registered inputs pinned at the registration's values. -/
theorem rLeakTapReg_of_r (H : Msg → Msg) (D : Deploy Coin Col T) {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (reg : AuditStatement → Finset ℕ)
    (ops : AuditStatement → List (Opening Bool)) {Dj : ℕ} (tops : AuditStatement → List (Opening Bool)) {x : Msg}
    (hx : ScopeAcc D lay k x) {w : Wit} (hR : R H D.commit D.parse D.partition δ x w)
    (hagree : ∀ s, D.parse x = some s → (∀ q ∈ tops s, D.commit.Opens H s.roots q = true) →
      ∀ g, (∃ q ∈ tops s, q.pos = g) → ∀ b, w.val H D.commit s g = some b → b = tapVal (tops s) g)
    (hreg : ∀ s, D.parse x = some s → ∀ g ∈ reg s, ∀ b, w.val H D.commit s g = some b → b = regVal (ops s) g) :
    RLeakTap H D.commit D.parse D.partition L K S lay k δ
      (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops x w := by
  obtain ⟨s, P, hs, hP, hM⟩ := hR
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hsc⟩, s₂, hs₂, hfit⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  have hF := faulty_mem_faultySets hM (disjoint_allSlots hδ hM)
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H D.commit s P w
  refine ⟨s, P, hs, hP, fun op o htap hcov hl hD => ?_⟩
  set fl := TapProof.fill (w.val H D.commit s) (tapVal (tops s)) with hfl
  have hext : ∀ g b, w.val H D.commit s g = some b → fl g = some b := TapProof.fill_ext
  have hacc := accepted_of hsc.1 hsc.2.1 hfit ι
    (fun v u hu hv => TapProof.correct_mono (TapProof.refIn_mono hext) (hι v u hu hv))
    (TapProof.carries_fill hfit htap (hagree s hs htap.1)) hcov
  set tv := traceVal s.prog fl with htv
  set G := freeInputs s.prog (lay s) (k s) with hG
  set free := heldAdv s.prog P (faulty H D.commit s P w) fl with hfree
  let a' : ℕ → Bool := fun g => if g ∈ G then (if g ∈ reg s then regVal (ops s) g else tv g) else false
  have hpt : ∀ i, s.prog.gates[i]? = some .input → free i = none →
      (fun g => if g ∈ G then tv g else false) i = a' i := by
    intro i hi hfi
    show (if i ∈ G then tv i else false) = if i ∈ G then (if i ∈ reg s then regVal (ops s) i else tv i) else false
    by_cases hiG : i ∈ G
    · rw [ite_eq_left hiG, ite_eq_left hiG]
      by_cases hir : i ∈ reg s
      · rw [ite_eq_left hir]
        have hnf : ¬ FaultyGate s.prog P (faulty H D.commit s P w) i := fun hfg => by
          rw [hfree, heldAdv_of hfg] at hfi
          cases hfi
        obtain ⟨b, hb⟩ := val_some_of_input hsc.2.1 hi hnf
        rw [htv, traceVal_of_val hi (hext i b hb)]
        exact hreg s hs i hir b hb
      · rw [ite_eq_right hir]
    · rw [ite_eq_right hiG, ite_eq_right hiG]
  refine ⟨hl, hD, _, hF, a', ⟨fun g hg hgf => ?_, fun g hg => ite_eq_right hg⟩, free,
    fun g b h => faultyGate_of_heldAdv h, fun t ht => ?_⟩
  · show (if g ∈ G then (if g ∈ reg s then regVal (ops s) g else tv g) else false) = regVal (ops s) g
    rw [ite_eq_left hgf, ite_eq_left hg]
  · obtain ⟨h, hr, hst⟩ := hacc t ht
    refine ⟨h, ?_, hst⟩
    have hv : valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) tv) free =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) a') free :=
      funext (valOver_congr_held s.prog free fun i hi hfi =>
        (inputOf_restrict s.prog (lay s) (k s) _ tv hi).trans
          (inputOf_congr_at (lay s) (k s) _ (hpt i hi hfi)))
    rw [hr]
    simp only [circuitProgram, hv]

theorem mem_accSeqsSampled_of_windowGoodReg {H : Msg → Msg} {D : Deploy Coin Col T} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ}
    {ops : AuditStatement → List (Opening Bool)} {Dj : ℕ} {s₀ : AuditStatement} {P₀ : Partition}
    (hP₀ : D.partition s₀ = some P₀) {ρ : ℕ → Bool} {tops : AuditStatement → List (Opening Bool)}
    {op : ℕ → List (Obs L K S) → List Bool} {x : Msg} {w : Wit} {o : List (Obs L K S)}
    (htap : ∀ s, D.parse x = some s → TapOpens H D.commit (lay s) (k s) op s (tops s) o)
    (h : WindowGoodReg H D L K S lay k δ reg ops Dj s₀ ρ tops x w o) :
    (lay s₀).Fits s₀.prog (k s₀) ∧
      o ∈ AccSeqsSampled s₀.prog P₀ s₀.design (lay s₀) (k s₀) (AsReg s₀.prog (lay s₀) (k s₀) (reg s₀) ρ) δ op Dj := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hl⟩, ⟨-, s₂, hs₂, hfits⟩, s, hs, ⟨hprog, hdes, hlay, hk, hreg, hpart⟩, hρ, hcov,
    hlen, hj⟩ := h
  rw [Option.some_injective _ (hs₁.symm.trans hs)] at hP₁ hl
  rw [Option.some_injective _ (hs₂.symm.trans hs)] at hfits
  rw [Option.some_injective _ (hP₁.symm.trans (hpart.trans hP₀))] at hl
  have hmem := hl op o (htap s hs) hcov hlen hj
  beta_reduce at hmem
  rw [asReg_congr _ _ _ _ hρ] at hmem
  rw [hprog, hdes, hlay, hk, hreg] at hmem
  rw [hprog, hlay, hk] at hfits
  exact ⟨hfits, hmem⟩

variable [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The leak bound at the network certifier's rows and the registration's values**, at any deployment `D` of the
sampled verifier where `KS` holds and the trace's commitment binds. On the statements in the audit's scope whose
registration opens their registered inputs: `Pr[acc ∧ ¬RLeakTap] ≤ δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]`, with the
registered inputs pinned at the registration's values. -/
theorem sampledLeak_tapReg (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T)
    (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ)
    (tops : AuditStatement → List (Opening Bool)) (x : Msg) (hx : ScopeAcc D lay k x)
    (hreg : ∀ st, D.parse x = some st → RegOpens H D.commit st (reg st) (ops st)) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    PrBad H D.V (RLeakTap H D.commit D.parse D.partition L K S lay k δ
        (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops) (E D decode) x P ≤
      κ D δ x + Real.sqrt (γ D x * PrB H D.V P x) + (Bext H D.V (E D decode) Ω P x).Pr H +
        (Btap H D.V (E D decode) D.commit D.parse tops Ω P x).Pr H +
        (Breg H D.V (E D decode) D.commit D.parse ops Ω P x).Pr H := by
  have hks' := hks δ decode hδ hdec x hx.1 Ω P
  rw [one_mul] at hks'
  unfold PrBad at hks' ⊢
  calc Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops).ext x
            ((E D decode).eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
          (((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (R H D.commit D.parse D.partition δ).ext x ((E D decode).eval H P x s.1 s.2.1 s.2.2).1) ∨
            TapProof.TapColl H D decode tops P x s) ∨ RegColl H D decode ops P x s) := by
        refine Pr_mono fun s ⟨hac, hR⟩ => ?_
        rcases hw : ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 with _ | w
        · exact Or.inl (Or.inl ⟨hac, by simp [Rel.ext]⟩)
        · rw [hw] at hR
          by_cases hA : R H D.commit D.parse D.partition δ x w
          · by_contra hnc
            refine hR (rLeakTapReg_of_r H D lay k hδ reg ops tops hx hA (fun st hst hpass g hq b hwb => ?_)
              fun st hst g hg b hwb => ?_)
            · by_contra hne
              exact hnc (Or.inl (Or.inr
                (TapProof.tapColl_of_disagree H D hb decode tops P x s hst hpass hac hw hq hwb hne)))
            · by_contra hne
              exact hnc (Or.inr (regColl_of_disagree H D hb decode reg ops P x s hst (hreg st hst) hac hw hg hwb hne))
          · exact Or.inl (Or.inl ⟨hac, hA⟩)
    _ ≤ _ := Pr_union_le _ _
    _ ≤ _ := add_le_add (Pr_union_le _ _) le_rfl
    _ ≤ _ := add_le_add (add_le_add hks' (TapProof.Pr_tapColl_le_Btap H D decode tops P x))
        (Pr_regColl_le_Breg H D decode ops P x)

end Deploy

/-- **The leak through the network certifier over a monitoring period, at the registration's values**:
`Guarantees.PeriodLeakRegistered`, from `sampledLeak_tapReg` and `encard_period_le`. -/
theorem periodLeak_registered : Guarantees.PeriodLeakRegistered := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay k δ decode hδ hdec reg ops Dj n sw Pw hPw ρ op β hβ Conn Cell Srv Dev Z _
    arr f tap run htrust cap Ss hcap hfit Zs schedOf srvOf hs dec hdz x wit tops hb
  exact ⟨fun m hm hreg Ω _ _ P =>
      sampledLeak_tapReg H D hks hb L K S lay k δ decode hδ hdec reg ops Dj tops m hm hreg Ω P,
    fun htap => encard_period_le L K S lay k δ reg Dj sw Pw ρ op β hβ arr f tap run htrust cap Ss hfit Zs schedOf
      srvOf hs dec hdz _ fun z hz hg i => mem_accSeqsSampled_of_windowGoodReg (hPw i) (htap z hz i) (hg i)⟩

end NetTiming
