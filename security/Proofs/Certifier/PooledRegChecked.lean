module

public import Definitions.Certifier.Guarantees.PooledRegChecked
public import Proofs.Certifier.PooledReg
public import Proofs.Certifier.Carried
import Proofs.Certifier.MonitoringPeriodSum

/-!
# The checked bound pooled over a monitoring period, proved

`Guarantees.PooledLeakRegisteredChecked`: `PooledLeakRegistered`'s bound with each phase's check holding on the run at
its trace with the registered openings, and its checked inputs left out of the count. The steps:

1. `ingress_of_registers`: where the network attester registers a record, the operator's moves over the horizon are
   the registered openings' values at the ingress gates.
2. `leakAtChecked_of_trace`: `leakAt_of_trace` with the incorrect set off the classes sampled `all` and the check
   holding on the trace's run.
3. `leakAtChecked_regR`, the bridge: `leakAt_regR` with a valid run, the run being `regRunR`. It needs no extracted
   witness: the run is fixed before the phase's coins.
4. `pr_miss_meets_all`, `pr_miss_rate_le`, `pr_phase_le`: the sample's miss of a set meeting a class sampled `all`,
   its miss at the design's rate, and one phase's per-history bound lifted to the period's coins.
5. `pr_phaseHit_le`, `phasesMissNoHit`, `phaseLeaksChecked_regR`: a phase accepting with a sampled block of `WregR`
   (at most `e i`), accepting without one (at most the miss), and the checked leak step (at most `εchk i`).
6. `pooledChecked_of_event`: the probability bound, each `e i` paid once.
7. `encard_pooledRegChecked_le`: the count, `encard_pooledJoint_view_le` at the empty pool's carried count.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem ingress_of_registers {Lk K S : ℕ} {H : Msg → Msg} {Cm : CommitmentScheme Bool} {lay : BoundaryGates Lk S}
    {k : ℕ} {op : ℕ → List (Obs Lk K S) → List Bool} {s : AuditStatement} {ops : List (Opening Bool)}
    {o : List (Obs Lk K S)} (h : CertifierRegisters H Cm lay k op s ops o) :
    (List.range k).map (fun t => op t (o.take t)) =
      (List.range k).map fun t => (lay.ingress t).map (certifierVal ops) := by
  refine List.map_congr_left fun t ht => ?_
  obtain ⟨-, -, -, -, h4⟩ := h.2 t (List.mem_range.1 ht)
  exact h4

theorem leakAtChecked_of_trace {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) (s : AuditStatement)
    (Pt : Partition) (hin : InScope Pt s) (hfit : (lay s).Fits s.prog (k s)) (tr : ℕ → Option Bool) (F : Finset ℕ)
    (ι : ℕ → ℕ → Bool) (hι : ∀ v u, Pt.blocks[v]? = some u → v ∉ F → s.prog.Correct (s.prog.refIn u tr (ι v)) u)
    (ρ : ℕ → Bool) (hρ : ∀ g ∈ reg s, g ∈ freeInputs s.prog (lay s) (k s) → traceVal s.prog tr g = ρ g)
    (op : ℕ → List (Obs Lk K S) → List Bool) (o : List (Obs Lk K S))
    (hcar : Carries (lay s) s.prog tr op (k s) o) (hcov : Covered (k s) o) (hlen : o.length = k s)
    (hj : ∀ y ∈ o, y.jitter < Dj) (hdisj : Disjoint F (allSlots s.design Pt))
    (hv : valid s (valOver s.prog ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i))
      (traceVal s.prog tr)) (heldAdv s.prog Pt F tr))) :
    LeakAtChecked lay k reg valid Dj s Pt ρ op F o := by
  have hacc := NetTiming.SampledAuditProof.accepted_of (P := Pt) (F := F) hin.1 hin.2.1 hfit ι hι hcar hcov
  refine ⟨hfit, hdisj, hlen, hj, fun g => if g ∈ freeInputs s.prog (lay s) (k s) then traceVal s.prog tr g else false,
    ⟨fun g hg hgf => ?_, fun g hg => ?_⟩,
    heldAdv s.prog Pt F tr, fun g b h => NetTiming.SampledAuditProof.faultyGate_of_heldAdv h, ?_, ?_⟩
  · simp only [hgf, ite_true]
    exact hρ g hg hgf
  · simp only [hg, ite_false]
  · have hvr : valOver s.prog ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i))
          fun g => if g ∈ freeInputs s.prog (lay s) (k s) then traceVal s.prog tr g else false)
          (heldAdv s.prog Pt F tr) =
        valOver s.prog ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i))
          (traceVal s.prog tr)) (heldAdv s.prog Pt F tr) :=
      funext (NetTiming.valOver_congr_input s.prog _ fun i hi =>
        (NetTiming.inputOf_restrict s.prog (lay s) (k s) _ (traceVal s.prog tr) hi).symm)
    rw [hvr]
    exact hv
  · intro t ht
    obtain ⟨h, hr, hst⟩ := hacc t ht
    refine ⟨h, ?_, hst⟩
    have hv' : valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) (traceVal s.prog tr))
          (heldAdv s.prog Pt F tr) =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t)
          fun g => if g ∈ freeInputs s.prog (lay s) (k s) then traceVal s.prog tr g else false)
          (heldAdv s.prog Pt F tr) :=
      funext (NetTiming.valOver_congr_input s.prog _ fun i hi =>
        NetTiming.inputOf_restrict s.prog (lay s) (k s) _ (traceVal s.prog tr) hi)
    rw [hr]
    simp only [circuitProgram, hv']

/-- **The bridge**: `leakAt_regR` with a valid run. Where `FregR` holds no proof block of a class sampled `all` and the
check holds on the run at the trace with the registered openings (`regRunR`), the observations meet the checked leak
relation at `FregR`, with no probability. -/
theorem leakAtChecked_regR {H : Msg → Msg} {D : Compiled Coin A} {Lk K S : ℕ}
    (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) {x : Msg} {s : AuditStatement}
    {Pt : Partition} (hs : D.parse x = some s) (hPt : D.partition s = some Pt) (hin : InScope Pt s)
    (hfit : (lay s).Fits s.prog (k s)) {Ω : Type} [Fintype Ω] (P : Prover Ω)
    (ops : AuditStatement → List (Opening Bool)) (op : ℕ → List (Obs Lk K S) → List Bool) (o : List (Obs Lk K S))
    (hcert : CertifierRegisters H D.commit (lay s) (k s) op s (ops s) o) (hcov : Covered (k s) o)
    (hlen : o.length = k s) (hj : ∀ y ∈ o, y.jitter < Dj) (hregd : ∀ g ∈ reg s, ∃ q ∈ ops s, q.pos = g)
    (hdisj : Disjoint (FregR H D P x ops) (allSlots s.design Pt))
    (hvalid : valid s (regRunR H D P x ops lay k s Pt)) :
    LeakAtChecked lay k reg valid Dj s Pt (certifierVal (ops s)) op (FregR H D P x ops) o := by
  classical
  have hpart : D.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  have hag : ∀ g, (∃ q ∈ ops s, q.pos = g) → ∀ b, regValR H D P x (ops s) g = some b →
      b = certifierVal (ops s) g := by
    intro g hg b hb
    have hany : (ops s).any (fun o => o.pos == g) = true := by
      obtain ⟨q, hq, hp⟩ := hg
      exact List.any_eq_true.2 ⟨q, hq, by simp [hp]⟩
    rw [regValR, ite_eq_left hany] at hb
    exact (Option.some.inj hb).symm
  have hno : ∀ v, ∃ ι : ℕ → Bool, ∀ u, Pt.blocks[v]? = some u → v ∉ FregR H D P x ops →
      s.prog.Correct (s.prog.refIn u (regValR H D P x (ops s)) ι) u := by
    intro v
    by_cases hex : ∃ u, Pt.blocks[v]? = some u ∧ v ∉ FregR H D P x ops
    · obtain ⟨u, hu, hv⟩ := hex
      obtain ⟨hvlt, -⟩ := List.getElem?_eq_some_iff.1 hu
      simp only [FregR, hs, mem_filter, mem_range, hpart, not_and, not_not] at hv
      obtain ⟨b, hb, val, ι, hval, hc⟩ := hv hvlt
      rw [hu] at hb
      have hbu : u = b := Option.some.inj hb
      rw [← hbu] at hc
      refine ⟨ι, fun u' hu' _ => ?_⟩
      rw [hu] at hu'
      have huu : u = u' := Option.some.inj hu'
      subst huu
      exact (correct_refIn_congr hin.1 hin.2.1 hu u hval fun _ _ _ => rfl).1 hc
    · exact ⟨fun _ => false, fun u hu hv => absurd ⟨u, hu, hv⟩ hex⟩
  choose ι hι using hno
  refine leakAtChecked_of_trace lay k reg valid Dj s Pt hin hfit
    (fillTrace (regValR H D P x (ops s)) (certifierVal (ops s))) (FregR H D P x ops) ι
    (fun v u hu hv => Prog.correct_mono (Prog.refIn_mono NetTiming.TapProof.fillTrace_ext) (hι v u hu hv))
    (certifierVal (ops s)) (fun g hg hgf => ?_) op o
    (NetTiming.TapProof.carries_fillTrace hfit hcert hag) hcov hlen hj hdisj ?_
  · have hinp : s.prog.gates[g]? = some .input := by
      simp only [freeInputs, mem_filter] at hgf
      exact hgf.2.1
    exact NetTiming.TapProof.traceVal_fillTrace (fun b hb => by rw [hinp] at hb; cases hb) (hag g (hregd g hg))
  · rw [ingress_of_registers hcert]
    exact hvalid

/-- The checked leak relation reads a statement through its program, design, layout, steps, registered inputs and
check only. -/
theorem leakAtChecked_congr {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) {V V' : AuditStatement → (ℕ → Bool) → Prop} (Dj : ℕ)
    {s s₀ : AuditStatement} {Pt : Partition} {ρ ρ' : ℕ → Bool} {op : ℕ → List (Obs Lk K S) → List Bool}
    {F : Finset ℕ} {o : List (Obs Lk K S)} (hprog : s.prog = s₀.prog) (hdes : s.design = s₀.design)
    (hlay : lay s = lay s₀) (hk : k s = k s₀) (hreg : reg s = reg s₀) (hV : V s = V' s₀)
    (hρ : ∀ g ∈ reg s, ρ g = ρ' g) (h : LeakAtChecked lay k reg V Dj s Pt ρ op F o) :
    LeakAtChecked lay k reg V' Dj s₀ Pt ρ' op F o := by
  unfold LeakAtChecked AcceptedChecked at h ⊢
  rw [← hprog, ← hdes, ← hlay, ← hk, ← hreg, ← hV]
  obtain ⟨h1, h2, h3, h4, a, ha, hrest⟩ := h
  exact ⟨h1, h2, h3, h4, a, ⟨fun g hg hgf => (ha.1 g hg hgf).trans (hρ g hg), ha.2⟩, hrest⟩

/-- **A set meeting a class sampled `all` is never missed**: the sample holds one of its blocks always, so the coins
whose sample holds none of the set have probability `0`. -/
theorem pr_miss_meets_all {H : Msg → Msg} {D : Compiled Coin A} (hS : D.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Partition} (hs : D.parse x = some s) (hPt : D.partition s = some Pt) (hin : InScope Pt s)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (F : Finset ℕ) (hF : ∀ v ∈ F, v < D.nPU x)
    (hd : ¬Disjoint F (allSlots s.design Pt)) :
    Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => ∀ u ∈ D.checked x p.2, u ∉ F) ≤ 0 := by
  have hpart : D.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  have hz : s.design.Pa (D.part s) F = 0 := by
    rw [hpart]
    exact NetTiming.SampledAuditProof.classesPa_eq_zero s.design.rules Pt.classes 0 F hd
  have hmiss : Pr (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F) ≤ 0 := by
    refine (Claims.Pr_mono fun r hr => ⟨fun hM => ?_, hr⟩).trans
      (Compiled.pr_miss_le hS hs hPt hin F hF (δ := 0) le_rfl)
    simp only [Design.M, Set.mem_ofPred_eq] at hM
    rw [hz] at hM
    exact lt_irrefl _ hM
  exact (Claims.SamplingGeneric.Pr_snd (α := Ω)
    (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F)).trans_le hmiss

/-- **A sampled set's miss at the design's rate**: the coins whose sample holds none of `F` have probability at most
`(1 - q)^{|F|}`. -/
theorem pr_miss_rate_le {H : Msg → Msg} {D : Compiled Coin A} (hS : D.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Partition} (hs : D.parse x = some s) (hPt : D.partition s = some Pt) (hin : InScope Pt s)
    {q : ℝ} (hq1 : q ≤ 1) (hq : RateAt s Pt q) (Ω : Type) [Fintype Ω] [Nonempty Ω] (F : Finset ℕ)
    (hF : ∀ v ∈ F, v < D.nPU x) :
    Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => ∀ u ∈ D.checked x p.2, u ∉ F) ≤ (1 - q) ^ F.card := by
  have hpart : D.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  have hsub : F ⊆ range Pt.blocks.length := fun v hv =>
    mem_range.2 (by rw [← Compiled.nPU_eq hs hPt]; exact hF v hv)
  have hpow0 : 0 ≤ (1 - q) ^ F.card := pow_nonneg (by linarith) _
  have hPa : s.design.Pa (D.part s) F ≤ (1 - q) ^ F.card := by
    rw [hpart]
    exact hq _ hsub
  have hmiss : Pr (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F) ≤ (1 - q) ^ F.card := by
    have h := Compiled.pr_miss_le hS hs hPt hin F hF (δ := max 0 (s.design.Pa (D.part s) F)) (le_max_left _ _)
    refine (Claims.Pr_mono fun r hr => ⟨fun hM => ?_, hr⟩).trans (h.trans (max_le hpow0 hPa))
    simp only [Design.M, Set.mem_ofPred_eq] at hM
    exact absurd hM (not_lt.2 (le_max_right _ _))
  exact (Claims.SamplingGeneric.Pr_snd (α := Ω)
    (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F)).trans_le hmiss

/-- **One phase's bound at every history lifts to the period's coins**: `pr_adaptive_add_le` at the rate `0`, with
phase `i` counted once and every other phase's event always true. -/
theorem pr_phase_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} (i : Fin m) (E : (Fin m → Ω) → Ω → Prop)
    (hE : ∀ ω ω', (∀ j, j < i → ω j = ω' j) → E ω = E ω') {c : ℝ} (h : ∀ ω, Pr (E ω) ≤ c) :
    Pr (fun ω => E ω (ω i)) ≤ c := by
  classical
  have hc0 : 0 ≤ c := (Claims.Pr_nonneg _).trans (h fun _ => Classical.arbitrary Ω)
  have key := pr_adaptive_add_le (Ω := Ω) (r := 0) le_rfl zero_le_one m
    (fun j ω a => j = i → E ω a) (fun j _ => if j = i then 1 else 0) (fun j => if j = i then c else 0)
    (fun j ω ω' hω => by
      by_cases hj : j = i
      · subst hj
        rw [hE ω ω' hω]
      · funext a
        simp only [hj, false_implies])
    (fun _ _ _ _ => rfl)
    (fun j => by split_ifs <;> first | exact hc0 | exact le_rfl)
    (fun j ω => by
      by_cases hj : j = i
      · subst hj
        simpa using h ω
      · simpa [hj] using Claims.Pr_le_one (fun _ : Ω => True))
    1
  refine le_trans (Claims.Pr_mono fun ω hω => ⟨fun j hj => by subst hj; exact hω, by simp⟩)
    (key.trans_eq (by simp))

section Phases

variable {m kV : ℕ} {Tp Sec : Type}

/-- **Execution phase `i`'s sample holds a block of its incorrect set** `WregR`, at the phase's own coins. -/
noncomputable def phaseHit [Fintype Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)) :
    Sec → Fin m → (Fin m → Tp × (Fin kV → Coin)) → Tp × (Fin kV → Coin) → Prop :=
  fun z i ω c => ∃ u ∈ (D (x z ω i)).checked (x z ω i) (fun j => c.2 (Fin.castLE (hk z ω i) j)),
    u ∈ WregR H D x Pv ops z ω i

/-- A phase's sample meets its incorrect set reading only the earlier phases' coins where its statement, its prover's
strategy and its registered openings do. -/
theorem phaseHit_prefix [Fintype Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (had : PhasesAdaptive x Pv ops) :
    ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → phaseHit H D x Pv hk ops z i ω = phaseHit H D x Pv hk ops z i ω' := by
  intro z i ω ω' h
  have key : ∀ (x₁ x₂ : Msg) (F₁ F₂ : Finset ℕ) (h₁ : (D x₁).V.k x₁ ≤ kV) (h₂ : (D x₂).V.k x₂ ≤ kV), x₁ = x₂ →
      F₁ = F₂ →
      (fun c : Tp × (Fin kV → Coin) => ∃ u ∈ (D x₁).checked x₁ (fun j => c.2 (Fin.castLE h₁ j)), u ∈ F₁) =
        fun c : Tp × (Fin kV → Coin) => ∃ u ∈ (D x₂).checked x₂ (fun j => c.2 (Fin.castLE h₂ j)), u ∈ F₂ := by
    intro x₁ x₂ F₁ F₂ h₁ h₂ hxx hFF
    subst hxx
    subst hFF
    rfl
  exact key _ _ _ _ _ _ (had.1 z i ω ω' h) (wreg_prefix H D had z i ω ω' h)

/-- **A phase accepts with a sampled block of its incorrect set** with probability at most `e i`, over the period's
coins: the event `e i` bounds at every history, lifted (`pr_phase_le`). -/
theorem pr_phaseHit_le [Fintype Tp] [Nonempty Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A}
    (hS : ∀ y, (D y).Sessions H) (Zs : Set Sec) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (had : PhasesAdaptive x Pv ops) {e : Fin m → ℝ} (he0 : ∀ i, 0 ≤ e i)
    (hev : ∀ z ∈ Zs, ∀ ω i, (D (x z ω i)).Scope (x z ω i) →
      Pr (fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) p.1 p.2).out = .acc ∧
          ∃ u ∈ (D (x z ω i)).checked (x z ω i) p.2,
            u ∈ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) ≤ e i) :
    ∀ z ∈ Zs, ∀ i, Pr (fun ω => phaseAcc H D x Pv hk z i ω (ω i) ∧ phaseHit H D x Pv hk ops z i ω (ω i)) ≤ e i := by
  intro z hz i
  refine pr_phase_le i (fun ω c => phaseAcc H D x Pv hk z i ω c ∧ phaseHit H D x Pv hk ops z i ω c)
    (fun ω ω' hω => by
      rw [phaseAcc_prefix H D x Pv hk had.1 had.2.1 z i ω ω' hω, phaseHit_prefix H D x Pv hk ops had z i ω ω' hω])
    (fun ω => ?_)
  by_cases hsc : (D (x z ω i)).Scope (x z ω i)
  · exact (pr_castLE (hk z ω i) (fun (t : Tp) (r : Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
      (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) t r).out = .acc ∧
        ∃ u ∈ (D (x z ω i)).checked (x z ω i) r,
          u ∈ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i))).trans_le (hev z hz ω i hsc)
  · refine (Claims.Pr_mono (q := fun _ => False) fun c hc => hsc ?_).trans
      (by rw [Claims.Pr_false]; exact he0 i)
    exact (hS (x z ω i)).refuses _ _ _ _ _ hc.1

/-- **A phase accepts with no sampled block of its incorrect set** with probability at most `(1 - q)^{|WregR|}` at
every history: the sample misses `WregR`. -/
theorem phasesMissNoHit [Fintype Tp] [Nonempty Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A}
    (hS : ∀ y, (D y).Sessions H) (Zs : Set Sec) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    {q : ℝ} (hq1 : q ≤ 1)
    (hrate : ∀ z ∈ Zs, ∀ ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q) :
    ∀ z ∈ Zs, ∀ i ω, Pr (fun c => phaseAcc H D x Pv hk z i ω c ∧ ¬phaseHit H D x Pv hk ops z i ω c) ≤
      (1 - q) ^ (WregR H D x Pv ops z ω i).card + 0 := by
  intro z hz i ω
  rw [add_zero]
  by_cases hsc : (D (x z ω i)).Scope (x z ω i)
  · obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
    have hc := pr_castLE (hk z ω i) (fun (t : Tp) (r : Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
      (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) t r).out = .acc ∧
        ¬∃ u ∈ (D (x z ω i)).checked (x z ω i) r,
          u ∈ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i))
    refine le_trans (Claims.Pr_mono fun c hc' => ?_) (hc.trans_le ?_)
    · exact hc'
    refine le_trans (Claims.Pr_mono (q := fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
      ∀ u ∈ (D (x z ω i)).checked (x z ω i) p.2,
        u ∉ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) fun p hp u hu hu' => hp.2 ⟨u, hu, hu'⟩) ?_
    exact pr_miss_rate_le (hS (x z ω i)) hs hPt hin hq1 (hrate z hz ω i s Pt hs hPt) Tp _
      (fregR_lt H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i))
  · refine (Claims.Pr_mono (q := fun _ => False) fun c hc => hsc ?_).trans
      (by rw [Claims.Pr_false]; exact pow_nonneg (by linarith) _)
    exact (hS (x z ω i)).refuses _ _ _ _ _ hc.1

/-- **Each phase accepted with no sampled block of `WregR` meets its checked leak relation there**, except with
probability `εchk i`: where `FregR` meets a class sampled `all`, the sample always holds one of its blocks, so this
happens with probability `0` (`pr_miss_meets_all`); where the check fails on `regRunR`, it is the miss `εchk i` bounds
(`PhaseChecks`); otherwise the bridge (`leakAtChecked_regR`) gives the relation with no probability. -/
theorem phaseLeaksChecked_regR [Fintype Tp] [Nonempty Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A}
    (hS : ∀ y, (D y).Sessions H) {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ)
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool) (Zs : Set Sec)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S))
    (R : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prop) (had : PhasesAdaptive x Pv ops)
    {εchk : Fin m → ℝ} (hck : PhaseChecks H D lay k valid sw x Pv ops εchk)
    (hyp : ∀ z ∈ Zs, ∀ ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      R z ω i →
      SameRegistration (D (x z ω i)) lay k reg (sw i) s ∧ (D (x z ω i)).partition s = some (Pw i) ∧
        (lay s).Fits s.prog (k s) ∧
        CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
          (obs z ω i) ∧
        Covered (k s) (obs z ω i) ∧ (obs z ω i).length = k s ∧ (∀ y ∈ obs z ω i, y.jitter < Dj) ∧
        ∀ g ∈ reg s, (∃ o ∈ ops z ω i s, o.pos = g) ∧ certifierVal (ops z ω i s) g = ρ i g) :
    ∀ z ∈ Zs, ∀ i, Pr (fun ω => (phaseAcc H D x Pv hk z i ω (ω i) ∧ ¬phaseHit H D x Pv hk ops z i ω (ω i)) ∧
      ¬(R z ω i → LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i))
        (WregR H D x Pv ops z ω i) (obs z ω i))) ≤ εchk i := by
  intro z hz i
  let Bad1 : (Fin m → Tp × (Fin kV → Coin)) → Tp × (Fin kV → Coin) → Prop := fun ω c =>
    (phaseAcc H D x Pv hk z i ω c ∧ ¬phaseHit H D x Pv hk ops z i ω c) ∧ ∃ s Pt,
      (D (x z ω i)).parse (x z ω i) = some s ∧ (D (x z ω i)).partition s = some Pt ∧
        ¬Disjoint (WregR H D x Pv ops z ω i) (allSlots s.design Pt)
  let Bad2 : (Fin m → Tp × (Fin kV → Coin)) → Tp × (Fin kV → Coin) → Prop := fun ω c =>
    phaseAcc H D x Pv hk z i ω c ∧ ∃ s Pt, (D (x z ω i)).parse (x z ω i) = some s ∧
      (D (x z ω i)).partition s = some Pt ∧ Disjoint (WregR H D x Pv ops z ω i) (allSlots s.design Pt) ∧
      ¬valid (sw i) (regRunR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i) lay k s Pt)
  have h1 : Pr (fun ω => Bad1 ω (ω i)) ≤ 0 := by
    refine pr_phase_le i Bad1 (fun ω ω' hω => ?_) (fun ω => ?_)
    · simp only [Bad1]
      rw [phaseAcc_prefix H D x Pv hk had.1 had.2.1 z i ω ω' hω, phaseHit_prefix H D x Pv hk ops had z i ω ω' hω,
        wreg_prefix H D had z i ω ω' hω, had.1 z i ω ω' hω]
    · by_cases hsc : (D (x z ω i)).Scope (x z ω i)
      · obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
        by_cases hd : Disjoint (WregR H D x Pv ops z ω i) (allSlots s.design Pt)
        · refine (Claims.Pr_mono (q := fun _ => False) fun c hc => ?_).trans_eq Claims.Pr_false
          obtain ⟨-, s', Pt', hs', hPt', hd'⟩ := hc
          obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
          obtain rfl : Pt' = Pt := Option.some.inj (hPt'.symm.trans hPt)
          exact hd' hd
        · have hc := pr_castLE (hk z ω i) (fun (t : Tp) (r : Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
            ∀ u ∈ (D (x z ω i)).checked (x z ω i) r,
              u ∉ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i))
          refine le_trans (Claims.Pr_mono fun c hc' => ?_) (hc.trans_le ?_)
          · exact fun u hu hu' => hc'.1.2 ⟨u, hu, hu'⟩
          · exact pr_miss_meets_all (hS (x z ω i)) hs hPt hin Tp _
              (fregR_lt H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) hd
      · refine (Claims.Pr_mono (q := fun _ => False) fun c hc => hsc ?_).trans_eq Claims.Pr_false
        exact (hS (x z ω i)).refuses _ _ _ _ _ hc.1.1
  have h2 : Pr (fun ω => Bad2 ω (ω i)) ≤ εchk i := by
    refine pr_phase_le i Bad2 (fun ω ω' hω => ?_) (fun ω => ?_)
    · simp only [Bad2]
      rw [phaseAcc_prefix H D x Pv hk had.1 had.2.1 z i ω ω' hω, wreg_prefix H D had z i ω ω' hω,
        had.1 z i ω ω' hω, had.2.1 z i ω ω' hω, had.2.2 z i ω ω' hω]
    · exact (pr_castLE (hk z ω i) (fun (t : Tp) (r : Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) t r).out = .acc ∧
          ∃ s Pt, (D (x z ω i)).parse (x z ω i) = some s ∧ (D (x z ω i)).partition s = some Pt ∧
            Disjoint (FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) (allSlots s.design Pt) ∧
            ¬valid (sw i) (regRunR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i) lay k s Pt))).trans_le
        (hck z ω i)
  refine (Claims.Sampling.Pr_or_le (fun ω hω => ?_) h1 h2).trans_eq (zero_add _)
  obtain ⟨⟨hacc, hnh⟩, hnot⟩ := hω
  by_contra hc
  simp only [not_or] at hc
  refine hnot fun hR => ?_
  have hacc' : (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) (ω i).1
      (fun j => (ω i).2 (Fin.castLE (hk z ω i) j))).out = .acc := hacc
  obtain ⟨s, Pt, hs, hPt, hin⟩ := (hS (x z ω i)).refuses _ _ _ _ _ hacc'
  obtain ⟨hsame, hPw, hfit, hcert, hcov, hlen, hj, hreg⟩ := hyp z hz ω i s hs hacc hR
  have hdisj : Disjoint (WregR H D x Pv ops z ω i) (allSlots s.design Pt) :=
    by_contra fun hd => hc.1 ⟨⟨hacc, hnh⟩, s, Pt, hs, hPt, hd⟩
  have hval : valid (sw i) (regRunR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i) lay k s Pt) :=
    by_contra fun hv => hc.2 ⟨hacc, s, Pt, hs, hPt, hdisj, hv⟩
  have hPtw : Pt = Pw i := Option.some.inj (hPt.symm.trans hPw)
  subst hPtw
  have hl := leakAtChecked_regR lay k reg (fun _ => valid (sw i)) Dj hs hPt hin hfit (Pv z ω i) (ops z ω i)
    (op i (runsBefore (obs z ω) i)) (obs z ω i) hcert hcov hlen hj (fun g hg => (hreg g hg).1) hdisj hval
  exact leakAtChecked_congr lay k reg Dj hsame.1 hsame.2.1 hsame.2.2.1 hsame.2.2.2.1 hsame.2.2.2.2.1 rfl
    (fun g hg => (hreg g hg).2) hl

/-- **The checked pooled bound from any bound on each phase's event at `WregR`**: at a secret of `Zs`, every phase is
accepted (and meets `R`) while the period's checked good event fails with probability at most
`(1 - q)^Fmax + Σ_i e i + Σ_i εchk i`. Some phase accepts with a sampled block of its `WregR` with probability at most
`Σ_i e i` (`pr_phaseHit_le`); otherwise every phase accepts with its `WregR` missed, which `pr_pooled_le` bounds at
the miss `(1 - q)^{|WregR|}` alone and the checked leak step `phaseLeaksChecked_regR`. Each `e i` is paid once. -/
theorem pooledChecked_of_event [Fintype Tp] [Nonempty Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A}
    (hS : ∀ y, (D y).Sessions H) {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool) (Zs : Set Sec)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S))
    (R : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prop) (had : PhasesAdaptive x Pv ops)
    {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hrate : ∀ z ∈ Zs, ∀ ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
    {e εchk : Fin m → ℝ} (he0 : ∀ i, 0 ≤ e i)
    (hev : ∀ z ∈ Zs, ∀ ω i, (D (x z ω i)).Scope (x z ω i) →
      Pr (fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) p.1 p.2).out = .acc ∧
          ∃ u ∈ (D (x z ω i)).checked (x z ω i) p.2,
            u ∈ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) ≤ e i)
    (hck : PhaseChecks H D lay k valid sw x Pv ops εchk)
    (hyp : ∀ z ∈ Zs, ∀ ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      R z ω i →
      SameRegistration (D (x z ω i)) lay k reg (sw i) s ∧ (D (x z ω i)).partition s = some (Pw i) ∧
        (lay s).Fits s.prog (k s) ∧
        CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
          (obs z ω i) ∧
        Covered (k s) (obs z ω i) ∧ (obs z ω i).length = k s ∧ (∀ y ∈ obs z ω i, y.jitter < Dj) ∧
        ∀ g ∈ reg s, (∃ o ∈ ops z ω i s, o.pos = g) ∧ certifierVal (ops z ω i s) g = ρ i g)
    (Fmax : ℕ) :
    ∀ z ∈ Zs, Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i) ∧ R z ω i) ∧
        ¬PooledGoodChecked lay k reg valid Dj sw Pw ρ (fun i => op i (runsBefore (obs z ω) i)) Fmax
          (WregR H D x Pv ops z ω) (obs z ω)) ≤
      (1 - q) ^ Fmax + ∑ i, e i + ∑ i, εchk i := by
  intro z hz
  have hmain := NetTiming.pr_pooled_le
    (fun i ω c => phaseAcc H D x Pv hk z i ω c ∧ ¬phaseHit H D x Pv hk ops z i ω c)
    (fun i ω ω' hω => by
      rw [phaseAcc_prefix H D x Pv hk had.1 had.2.1 z i ω ω' hω, phaseHit_prefix H D x Pv hk ops had z i ω ω' hω])
    (WregR H D x Pv ops z) (fun i ω ω' hω => wreg_prefix H D had z i ω ω' hω) (fun _ => 0) εchk (fun _ => le_rfl)
    q hq0 hq1 (fun i ω => phasesMissNoHit hS Zs x Pv hk ops hq1 hrate z hz i ω)
    (fun i ω => R z ω i → LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i))
      (WregR H D x Pv ops z ω i) (obs z ω i))
    (phaseLeaksChecked_regR hS lay k reg valid Dj sw Pw ρ op Zs x Pv hk ops obs R had hck hyp z hz) Fmax
  simp only [Finset.sum_const_zero, add_zero] at hmain
  have hhit : Pr (fun ω => ∃ i ∈ (univ : Finset (Fin m)),
      phaseAcc H D x Pv hk z i ω (ω i) ∧ phaseHit H D x Pv hk ops z i ω (ω i)) ≤ ∑ i, e i :=
    (Pr_exists_mem_le_sum univ _).trans
      (sum_le_sum fun i _ => pr_phaseHit_le hS Zs x Pv hk ops had he0 hev z hz i)
  refine (Claims.Sampling.Pr_or_le (fun ω hω => ?_) hhit hmain).trans (le_of_eq (by ring))
  by_cases hh : ∃ i ∈ (univ : Finset (Fin m)),
      phaseAcc H D x Pv hk z i ω (ω i) ∧ phaseHit H D x Pv hk ops z i ω (ω i)
  · exact Or.inl hh
  · exact Or.inr ⟨fun i => ⟨(hω.1 i).1, fun hp => hh ⟨i, mem_univ i, (hω.1 i).1, hp⟩⟩,
      fun hg => hω.2 ⟨hg.1, fun i => hg.2 i (hω.1 i).2⟩⟩

end Phases

/-- **The checked pooled count, read under a registered parameter**: at one sample of the coins, the secrets of `Zs`
meeting `Reg` (where each phase's incorrect set holds only its partition's proof blocks) and the period's checked good
event number at most `#Θ · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ (reg_i ∪ chk_i)) · Dj^k_i`. It is
`encard_pooledJoint_view_le` at the carried count of the empty pool (`encard_carried_le` through
`leakAtCarried_of_leakAtChecked`), each view read back under a parameter of `Θ` (`RecordFormat.ReadsBy`). -/
theorem encard_pooledRegChecked_le {Lk K S m : ℕ} (lay : AuditStatement → BoundaryGates Lk S)
    (k : AuditStatement → ℕ) (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
    {Conn Cell Srv Dev Par Zt : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs Lk K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs Lk K S)))
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin m → List (Obs Lk K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows) (Θ : Set Par) (Zs : Set Zt)
    (sched : Zt → CertifierSchedule Conn) (srv : Zt → Srv) (dec : (Conn → List (ℕ × CellView Cell)) → Zt)
    (hch : MonitoringPeriodChannelReg f allows Θ Zs sched srv dec fun p => (run tap p).1)
    (W : Zt → Fin m → Finset ℕ) (Reg : Zt → Prop)
    (hWn : ∀ z ∈ Zs, Reg z → ∀ i, W z i ⊆ range (Pw i).blocks.length) (Fmax : ℕ) :
    {z | z ∈ Zs ∧ Reg z ∧ PooledGoodChecked lay k reg valid Dj sw Pw ρ
        (fun i => op i (runsBefore (run tap (sched z, srv z)).2 i)) Fmax (W z) (run tap (sched z, srv z)).2}.encard ≤
      Θ.encard * ((((∑ i, (Pw i).blocks.length) + 1) ^ Fmax *
        2 ^ ((univ.sup fun i => (range (Pw i).blocks.length).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
        ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card *
          Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  classical
  let W' : Zt → Fin m → Finset ℕ := fun z i => (W z i).filter (· < (Pw i).blocks.length)
  have hW' : ∀ z i, W' z i ⊆ range (Pw i).blocks.length := fun z i v hv => mem_range.2 (mem_filter.1 hv).2
  have hWeq : ∀ z ∈ Zs, Reg z → W' z = W z := fun z hz hr => funext fun i =>
    Finset.filter_true_of_mem fun v hv => mem_range.1 (hWn z hz hr i hv)
  let Good : (Fin m → Finset ℕ) → (Fin m → List (Obs Lk K S)) → Prop := fun F o =>
    ∀ i, LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i (runsBefore o i)) (F i) (o i)
  have hsub : {z | z ∈ Zs ∧ Reg z ∧ PooledGoodChecked lay k reg valid Dj sw Pw ρ
        (fun i => op i (runsBefore (run tap (sched z, srv z)).2 i)) Fmax (W z) (run tap (sched z, srv z)).2} ⊆
      {z | z ∈ Zs ∧ (∑ i, (W' z i).card < Fmax ∧ Good (W' z) (run tap (sched z, srv z)).2)} := by
    rintro z ⟨hz, hr, hg⟩
    have hw := hWeq z hz hr
    refine ⟨hz, ?_, ?_⟩
    · rw [hw]
      exact hg.1
    · rw [hw]
      exact hg.2
  have hcnt : ∀ F : Fin m → Finset ℕ, (∀ i, F i ⊆ range (Pw i).blocks.length) →
      {o | Good F o}.encard ≤ ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
        (reg (sw i) ∪ chk (sw i))).card * (2 ^ faultyBits (sw i).prog (Pw i) (F i) * Dj ^ k (sw i))) : ℕ) : ℕ∞) := by
    intro F _
    refine (Set.encard_le_encard fun o ho => ?_).trans ((encard_carried_le lay k reg chk valid (fun _ => [])
      (fun _ => []) Dj sw Pw ρ op hchk (fun i => fun _ _ _ _ _ _ _ _ _ => rfl) (fun _ => false) (fun _ => []) F).trans_eq
      (by simp only [List.toFinset_nil, Finset.union_empty]))
    exact ⟨fun _ => [], fun _ => [], fun i => by simp [poolOpen],
      fun i => leakAtCarried_of_leakAtChecked (ho i)⟩
  calc _ ≤ _ := Set.encard_le_encard hsub
    _ ≤ _ := encard_pooledJoint_view_le Good (fun i => (Pw i).blocks.length)
        (univ.sup fun i => (range (Pw i).blocks.length).sup fun v => faultyBits (sw i).prog (Pw i) {v})
        (fun i => (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card)
        (fun i => Dj ^ k (sw i)) (fun i F => faultyBits (sw i).prog (Pw i) F)
        (fun i F hF => faultyBits_le_card_mul _ _ hF fun v hv =>
          (Finset.le_sup (f := fun v => faultyBits (sw i).prog (Pw i) {v}) (mem_range.2 hv)).trans
            (Finset.le_sup (f := fun i => (range (Pw i).blocks.length).sup fun v => faultyBits (sw i).prog (Pw i) {v})
              (mem_univ i)))
        hcnt rd Θ Zs (fun z => (run tap (sched z, srv z)).1) (fun z => (run tap (sched z, srv z)).2) dec hch.2 W' hW'
        Fmax (fun z hz _ _ => by
          obtain ⟨hfit, θ, hθ, hal⟩ := hch.1 z hz
          refine ⟨θ, hθ, ?_⟩
          rw [htrust (sched z, srv z)]
          exact (hread θ _ hal hfit (arr (srv z))).symm)

/-- **The leak through the network attester over a monitoring period, at the trace with the registered openings, with
the checked inputs left out, given `PhaseChecks`**: `Guarantees.PooledLeakRegisteredChecked`, conditional on that open
premise. The bound is `pooledChecked_of_event` at
`PooledLeakRegistered`'s event bound (`Compiled.ksRegR_of_sessionKS`) and its registration facts; the count is
`encard_pooledRegChecked_le` at each sample of the coins, a registered phase's `WregR` holding only its partition's
proof blocks (`fregR_lt`). -/
theorem pooledLeakRegisteredChecked : Guarantees.PooledLeakRegisteredChecked := by
  intro Coin _ _ _ A H D hA hb hS Lk K S lay k reg chk valid Dj m kV Tp Sec _ _ sw Pw ρ hchk op x Pv hk ops had hops
    q hq0 hq1 hrate e he εchk hck hfit Conn Cell Srv Dev Par arr f tap run htrust rd allows hread Θ Zs schedOf srvOf
    dec hch hcert Fmax
  refine ⟨fun z _ => ?_, fun ω => ?_⟩
  · have he0 : ∀ i, 0 ≤ e i := fun i =>
      le_trans' (he z (fun _ => (Classical.arbitrary Tp, fun _ => default)) i)
        (add_nonneg (Compiled.εregX_nonneg H _ _ _) (Claims.Pr_nonneg _))
    refine pooledChecked_of_event hS lay k reg valid Dj sw Pw ρ op Set.univ x Pv hk ops
      (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2)
      (fun z ω i => PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧
        RecordComplete k Dj sw (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) z ω i) had hq0 hq1
      (fun z _ => hrate z) he0
      (fun z _ ω i hsc =>
        (Compiled.ksRegR_of_sessionKS (hb (x z ω i)) (hS (x z ω i))
          (Claims.SamplingGeneric.sessionKSOf Coin A (D (x z ω i)) H hA (hS (x z ω i))) (ops z ω i) hsc
          (fun s hs => (hops z ω i s hs).1) Tp (Pv z ω i)).trans (he z ω i))
      hck (fun z _ ω i s hs hacc hR => ?_) Fmax z (Set.mem_univ z)
    obtain ⟨⟨s', hs', hsame, hPw, hval⟩, hcov, hlen, hj⟩ := hR
    have hss : s' = s := Option.some.inj (hs'.symm.trans hs)
    rw [hss] at hsame hPw hval
    have hks : k s = k (sw i) := hsame.2.2.2.1
    have hfit' : (lay s).Fits s.prog (k s) := by
      rw [hsame.1, hsame.2.2.1, hks]
      exact hfit i
    have hcov' : Covered (k s) ((run tap (schedOf z ω, srvOf z ω)).2 i) := by
      rw [hks]
      exact hcov
    exact ⟨hsame, hPw, hfit', hcert z ω i s hs hacc, hcov', hlen.trans hks.symm, hj,
      fun g hg => ⟨(hops z ω i s hs).2 g hg, hval g hg⟩⟩
  · exact encard_pooledRegChecked_le lay k reg chk valid Dj sw Pw ρ hchk op arr f tap run htrust rd allows hread Θ Zs
      (fun z => schedOf z ω) (fun z => srvOf z ω) (dec ω) (hch ω) (fun z => WregR H D x Pv ops z ω)
      (fun z => ∀ i, PhaseRegistered D lay k reg sw Pw ρ x ops z ω i)
      (fun z _ hr i v hv => by
        obtain ⟨s, hs, -, hPw, -⟩ := hr i
        exact mem_range.2 (by
          rw [← Compiled.nPU_eq hs hPw]
          exact fregR_lt H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i) v hv))
      Fmax

/-- **The checked good event is at least as strong as the unchecked one**: `LeakAtChecked` is `LeakAt` with the
incorrect set off the classes sampled `all` and a valid run, so a period's checked good event gives its good event. -/
theorem pooledGood_of_pooledGoodChecked {L K S m : ℕ} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop}
    {Dj : ℕ} {sw : Fin m → AuditStatement} {Pw : Fin m → Partition} {ρ : Fin m → ℕ → Bool}
    {op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool} {Fmax : ℕ} {W : Fin m → Finset ℕ}
    {o : Fin m → List (Obs L K S)}
    (h : PooledGoodChecked lay k reg valid Dj sw Pw ρ (fun i => op i (runsBefore o i)) Fmax W o) :
    PooledGood lay k reg Dj sw Pw ρ op Fmax W o := by
  refine ⟨h.1, fun i => ?_⟩
  obtain ⟨hfit, -, hlen, hj, a, ha, free, hfree, -, hacc⟩ := h.2 i
  exact ⟨hfit, hlen, hj, a, ha, free, hfree, hacc⟩

/-- **The discount is real**: where some phase's check covers a free input off its registered ones and `Dj > 0`, the
count's product over the phases with the checked inputs left out is strictly below the product with only the
registered inputs left out (`PooledLeakRegistered`'s inputs, `chk = ∅`). -/
theorem pooledCount_checked_lt {L S m : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) {Dj : ℕ} (hDj : 0 < Dj) (sw : Fin m → AuditStatement) (i₀ : Fin m)
    {g : ℕ} (hg : g ∈ chk (sw i₀)) (hgf : g ∈ freeInputs (sw i₀).prog (lay (sw i₀)) (k (sw i₀)))
    (hgr : g ∉ reg (sw i₀)) :
    ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card * Dj ^ k (sw i)) <
      ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i)) := by
  classical
  have hle : ∀ i, (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card ≤
      (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card := fun i =>
    Finset.card_le_card (Finset.sdiff_subset_sdiff (subset_refl _) Finset.subset_union_left)
  have hlt : (freeInputs (sw i₀).prog (lay (sw i₀)) (k (sw i₀)) \ (reg (sw i₀) ∪ chk (sw i₀))).card <
      (freeInputs (sw i₀).prog (lay (sw i₀)) (k (sw i₀)) \ reg (sw i₀)).card := by
    refine Finset.card_lt_card ⟨Finset.sdiff_subset_sdiff (subset_refl _) Finset.subset_union_left, fun h => ?_⟩
    have hgm : g ∈ freeInputs (sw i₀).prog (lay (sw i₀)) (k (sw i₀)) \ reg (sw i₀) := Finset.mem_sdiff.2 ⟨hgf, hgr⟩
    exact (Finset.mem_sdiff.1 (h hgm)).2 (Finset.mem_union_right _ hg)
  have hD : ∀ i, 0 < Dj ^ k (sw i) := fun i => pow_pos hDj _
  refine Finset.prod_lt_prod₀ (fun i _ => Nat.mul_pos (pow_pos two_pos _) (hD i))
    (fun i _ => Nat.mul_le_mul_right _ (Nat.pow_le_pow_right two_pos (hle i))) ⟨i₀, mem_univ _, ?_⟩
  exact Nat.mul_lt_mul_of_pos_right (Nat.pow_lt_pow_right (by norm_num) hlt) (hD i₀)

end NetTiming

namespace NetTiming.SecurityProofs

theorem PooledLeakRegisteredChecked : NetTiming.Guarantees.PooledLeakRegisteredChecked :=
  NetTiming.pooledLeakRegisteredChecked

end NetTiming.SecurityProofs

#print axioms NetTiming.SecurityProofs.PooledLeakRegisteredChecked
#print axioms NetTiming.pooledGood_of_pooledGoodChecked
#print axioms NetTiming.pooledCount_checked_lt
