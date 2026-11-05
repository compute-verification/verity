import Definitions.NetworkCertifier.Guarantees.Period
import Definitions.NetworkCertifier.RecordRows
import Proofs.NetworkCertifier.Period
import Proofs.NetworkCertifier.SampledLeakReg
import Proofs.NetworkCertifier.ReadFree

/-!
# The leak through the network certifier over a monitoring period, at the registration's values

On the audit's scope, sampling at the incorrect set (`RWired`, at the layout's wiring) at a trace that agrees with the
network certifier wherever both open a transfer gate, and with the registration at every registered input it opens,
gives the leak relation at the network certifier's rows with the registered inputs pinned at the registration's values
(`rLeakTapReg_of_r`): the trace filled in with the network certifier's values carries its records, so the circuit
accepts at that trace's inputs, and an input outside the incorrect proof blocks is opened, at the registration's value
where it is registered. A trace that disagrees with either makes a collision with its openings
(`TapProof.tapColl_of_disagree`, `regColl_of_disagree`), so each window's failure is at most
`δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]` (`sampledLeak_tapReg`). A window that meets `WindowGoodReg` has its
observations among the sequences `AccSeqsSampled` counts at its registration's values
(`mem_accSeqsSampled_of_windowGoodReg`), and the rows format's read-back from the record and `(r, D)` counts the
secrets (`RowsFormat.encard_periodRead_le`). Like `periodLeak`, the first part goes through `RWired` and `KSBy` at the
wired check (`ksBy_wiredOk`) directly, as `TapProof.sampledLeak_tap` does, not through `SampledAuditAccepted`.

`PeriodLeakRegisteredWf` states it at every rows format that reads back, taking `RowsFormat.Wf` as a hypothesis: the
listed guarantee, `Properties.NetworkPeriodLeak.PeriodLeakRegistered`, is it at the network certifier's own format,
where the period check's acceptance gives `Wf` (`Proofs.Properties.NetworkPeriodLeak`).
-/

namespace NetTiming

open Claims Claims.Sampling NetTiming.SampledAuditProof Finset

section Deploy

variable {Coin Col T : Type}

/-- **At the network certifier's rows and the registration's values**: on the audit's scope at `δ ≥ 0`, sampling at
the incorrect set (`RWired`, at the layout's wiring) at a trace that agrees with the network certifier wherever both open
a gate, and with the registration's values `regVal (ops s)` at every registered input it opens, gives the leak relation
at the network certifier's rows with the registered inputs pinned at the registration's values. -/
theorem rLeakTapReg_of_r (H : Msg → Msg) (D : Deploy Coin Col T) {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (reg : AuditStatement → Finset ℕ)
    (ops : AuditStatement → List (Opening Bool)) {Dj : ℕ} (tops : AuditStatement → List (Opening Bool)) {x : Msg}
    (hx : ScopeAcc D lay k x) {w : Wit} (hR : RWired H D.commit D.parse D.partition (fun s => (lay s).wires) δ x w)
    (hagree : ∀ s, D.parse x = some s → (∀ q ∈ tops s, D.commit.Opens H s.roots q = true) →
      ∀ g, (∃ q ∈ tops s, q.pos = g) → ∀ b, w.val H D.commit s g = some b → b = tapVal (tops s) g)
    (hreg : ∀ s, D.parse x = some s → ∀ g ∈ reg s, ∀ b, w.val H D.commit s g = some b → b = regVal (ops s) g) :
    RLeakTap H D.commit D.parse D.partition L K S lay k δ
      (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops x w := by
  obtain ⟨s, P, hs, hP, hM⟩ := hR
  beta_reduce at hM
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hsc⟩, s₂, hs₂, hfit⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  have hF := incorrect_mem_faultySets hM (disjoint_allSlots hδ hM)
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H D.commit s P w
  refine ⟨s, P, hs, hP, fun op o htap hcov hl hD => ?_⟩
  set fl := TapProof.fill (w.val H D.commit s) (tapVal (tops s)) with hfl
  have hext : ∀ g b, w.val H D.commit s g = some b → fl g = some b := TapProof.fill_ext
  have hacc := accepted_of (F := incorrect H D.commit s P (lay s).wires w) hsc.1 hsc.2.1 hfit ι
    (fun v u hu hv => TapProof.correct_mono (TapProof.refIn_mono hext) (hι v u hu fun h => hv (Finset.mem_union_left _ h)))
    (fun g hg hnf w' hw' => by
      obtain ⟨hc, b, h1, h2⟩ := wireRight_of_not_faultyGate H D.commit s hsc.2.1 (lay s).wires w hg hnf w' hw'
      exact ⟨hc, b, hext g b h1, hext w' b h2⟩)
    (TapProof.carries_fill hfit htap (hagree s hs htap.1)) hcov
  set tv := traceVal s.prog fl with htv
  set G := freeInputs s.prog (lay s) (k s) with hG
  set free := heldAdv s.prog P (incorrect H D.commit s P (lay s).wires w) fl with hfree
  let a' : ℕ → Bool := fun g => if g ∈ G then (if g ∈ reg s then regVal (ops s) g else tv g) else false
  have hpt : ∀ i, s.prog.gates[i]? = some .input → free i = none →
      (fun g => if g ∈ G then tv g else false) i = a' i := by
    intro i hi hfi
    show (if i ∈ G then tv i else false) = if i ∈ G then (if i ∈ reg s then regVal (ops s) i else tv i) else false
    by_cases hiG : i ∈ G
    · rw [ite_eq_left hiG, ite_eq_left hiG]
      by_cases hir : i ∈ reg s
      · rw [ite_eq_left hir]
        have hnf : ¬ FaultyGate s.prog P (faulty H D.commit s P w) i := fun ⟨v, hv, hr⟩ => by
          rw [hfree, heldAdv_of (F := incorrect H D.commit s P (lay s).wires w)
            ⟨v, Finset.mem_union_left _ hv, hr⟩] at hfi
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
    have hv : valOver s.prog (resolveWires s.prog (lay s).wires ((lay s).inputOf (k s) (ingressUpTo op o t) tv) free
          s.prog.gates.length) free =
        valOver s.prog (resolveWires s.prog (lay s).wires ((lay s).inputOf (k s) (ingressUpTo op o t) a') free
          s.prog.gates.length) free :=
      funext (valOver_congr_held s.prog free (resolveWires_congr_held s.prog (lay s).wires free
        (fun i hi hfi hw => (inputOf_restrict s.prog (lay s) (k s) _ tv hi hw).trans
          (inputOf_congr_at (lay s) (k s) _ (hpt i hi hfi))) _))
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
sampled verifier where sampling at the incorrect set holds at the layout's wiring (`KSBy` at the wired check) and the
trace's commitment binds. On the statements in the audit's scope whose registration opens their registered inputs:
`Pr[acc ∧ ¬RLeakTap] ≤ δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]`, with the registered inputs pinned at the
registration's values. -/
theorem sampledLeak_tapReg (H : Msg → Msg) (D : Deploy Coin Col T) (hb : D.commit.Binds H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (hks : KSBy H D (wiredOk fun s => (lay s).wires)) (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T)
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
  have hks' := ksBy_wiredOk.1 hks δ decode hδ hdec x hx.1 Ω P
  rw [one_mul] at hks'
  unfold PrBad at hks' ⊢
  calc Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops).ext x
            ((E D decode).eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
          (((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (RWired H D.commit D.parse D.partition (fun s => (lay s).wires) δ).ext x
              ((E D decode).eval H P x s.1 s.2.1 s.2.2).1) ∨
            TapProof.TapColl H D decode tops P x s) ∨ RegColl H D decode ops P x s) := by
        refine Pr_mono fun s ⟨hac, hR⟩ => ?_
        rcases hw : ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 with _ | w
        · exact Or.inl (Or.inl ⟨hac, by simp [Rel.ext]⟩)
        · rw [hw] at hR
          by_cases hA : RWired H D.commit D.parse D.partition (fun s => (lay s).wires) δ x w
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

set_option linter.unusedVariables false in
/-- **The leak through the network certifier over a monitoring period, at the registration's values, at every
well-formed rows format.** As `Guarantees.PeriodLeakAccepted`, with the registered inputs pinned at the values the
registration opens under each statement's roots (`regVal (ops s)`, `RegOpens`) rather than at the trace's, so no
window's good event reads the trace at a registered input (`WindowGoodReg`), and with a network certifier whose code
records in a rows format `F` that reads back (`RowsFormat.Wf`), on schedules whose `(r, D)` is in the registered set `Θ`
and which fit `F`: each window's audit, at a statement in its scope whose registration opens its registered inputs
(`RegOpensAt`), has `Pr[acc ∧ ¬RLeakTap δ] ≤ δ + ε_ks + ε_bind + Pr[B_tap] + Pr[B_reg]`, and where the network
certifier registers its observations (`CertifiersOpen`), a covert sender's secrets whose every window meets
`WindowGoodReg` number at most `#Θ · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`. The client
sees what crosses the wire (`laneWire`), padding included, and the rows format reads it back from the record alone, so
the count charges the schedule only for its `(r, D)` among `Θ`. `Properties.NetworkPeriodLeak.PeriodLeakRegistered`
is this at the network certifier's own format (`cellRows`), whose `Wf` the period check's acceptance gives. -/
def PeriodLeakRegisteredWf : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (hks : KSBy H D (wiredOk fun s => (lay s).wires))
    (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
    (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition)
    (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D δ sw Pw β)
    {Conn Cell Srv Dev Z : Type} (arr : Srv → Conn → Arrivals Cell) (F : RowsFormat Conn Cell L K) (hF : F.Wf)
    (cert : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost cert run fun p =>
      (laneWire p.1 (wardenCode p.1 (arr p.2)), F.record S n p.1 (arr p.2) (wardenCode p.1 (arr p.2))))
    (Θ : Set (ℕ × ℕ)) (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv)
    (hcov : SchedulesCovered {s | (s.r, s.D) ∈ Θ ∧ F.Fits S n s} Zs schedOf)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : SenderDecodes run cert Zs schedOf srvOf dec)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool)) (hb : D.commit.Binds H),
    (∀ m, ScopeAcc D lay k m → RegOpensAt H D reg ops m →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops)
          (E D decode) m P ≤
        κ D δ m + Real.sqrt (γ D m * Claims.PrB H D.V P m) + (Bext H D.V (E D decode) Ω P m).Pr H +
          (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H +
          (Breg H D.V (E D decode) D.commit D.parse ops Ω P m).Pr H) ∧
      (CertifiersOpen H D lay k op tops Zs x (fun z => (run cert (schedOf z, srvOf z)).2) →
        {z | z ∈ Zs ∧ ∀ i, WindowGoodReg H D L K S lay k δ reg ops Dj (sw i) (ρ i) tops (x z i) (wit z i)
            ((run cert (schedOf z, srvOf z)).2 i)}.encard ≤
          Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞))

/-- **The leak through the network certifier over a monitoring period, at the registration's values**:
`PeriodLeakRegisteredWf`, from `sampledLeak_tapReg` and the rows format's count from the read-back
(`RowsFormat.encard_periodRead_le`) at each window's `AccSeqsSampled` (`encard_accSeqsSampled_le`). -/
theorem periodLeak_registered : PeriodLeakRegisteredWf := by
  intro Coin Col T _ _ _ _ _ H D L K S lay hks k δ decode hδ hdec reg ops Dj n sw Pw ρ op β ⟨hPw, hβ⟩ Conn Cell Srv
    Dev Sec arr F hF cert run htrust Θ Zs schedOf srvOf hcov dec hdz x wit tops hb
  refine ⟨fun m hm hreg Ω _ _ P =>
      sampledLeak_tapReg H D hb L K S lay hks k δ decode hδ hdec reg ops Dj tops m hm hreg Ω P, fun htap => ?_⟩
  have hwin := fun z (hz : z ∈ Zs) (hg : ∀ i, WindowGoodReg H D L K S lay k δ reg ops Dj (sw i) (ρ i) tops (x z i)
      (wit z i) ((run cert (schedOf z, srvOf z)).2 i)) (i : Fin n) =>
    mem_accSeqsSampled_of_windowGoodReg (hPw i) (htap z hz i) (hg i)
  exact F.encard_periodRead_le hF arr cert run htrust Θ Zs schedOf srvOf hcov dec hdz _
    (fun i => AccSeqsSampled (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) δ (op i) Dj)
    (fun i => 2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
      (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i))
    (fun z hz hg i => (hwin z hz hg i).2)
    fun ⟨z₀, hz₀⟩ i => encard_accSeqsSampled_le _ _ _ _ _ (hwin z₀ hz₀.1 hz₀.2 i).1 δ _ _
      (encard_asReg_le _ _ _ _ _) (β i) (hβ i) (op i) Dj

end NetTiming
