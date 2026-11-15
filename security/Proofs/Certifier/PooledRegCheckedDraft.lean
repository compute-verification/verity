module

public import Definitions.Certifier.Guarantees.PooledRegChecked
public import Proofs.Certifier.PooledReg
public import Proofs.Certifier.Carried
public import Proofs.Certifier.MonitoringPeriodChecked

/-!
# DRAFT (scratch branch, in no root): the lemma chain for `PooledLeakRegisteredChecked`

Steps 1 to 7 are proved here; 8 to 10 are not yet written.

1. `ingress_of_registers` (new, routine): where the network certifier registers a record, the operator's moves over the
   horizon are the registered openings' values at the ingress gates.
2. `leakAtChecked_of_trace` (new, routine): `leakAt_of_trace` with the incorrect set off the classes sampled `all` and
   the check holding on the trace's run.
3. `leakAtChecked_regR` (new, **the bridge**): `leakAt_regR` with a valid run, the run being `regRunR`.
4. `leakAtChecked_congr` (new, routine): the relation read through the registration.
5. `pr_acc_meets_all_le` (new, routine): a set meeting a class sampled `all` is missed with probability `0`, so
   acceptance with it is at most the event's bound.
6. `pr_phase_le` (new, routine): a per-history bound on one phase lifts to the period's coins.
7. `phaseLeaksChecked_regR` (new): each phase's checked leak relation at `WregR` fails with probability at most
   `e i + εchk i`.
8. `pooledChecked_of_event` (new, routine): `pooled_of_event` with `phaseLeaksChecked_regR` as the leak step and the
   disjointness bound counted at `e i` a second time.
9. `encard_pooledReg_le` (new): the pooled count in the `#Θ` form, after `encard_pooledJoint_le` takes `Θ`.
10. The guarantee: 8 for the first conjunct, 9 for the second.
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

/-- **A set meeting a class sampled `all`**: the sample holds one of its blocks always, so the run accepts with the set
meeting such a class with probability at most the bound on its accepting with a sampled block of the set. -/
theorem pr_acc_meets_all_le {H : Msg → Msg} {D : Compiled Coin A} (hS : D.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Partition} (hs : D.parse x = some s) (hPt : D.partition s = some Pt) (hin : InScope Pt s)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (F : Finset ℕ) (hF : ∀ v ∈ F, v < D.nPU x) {c : ℝ}
    (h1 : Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ D.checked x p.2, u ∈ F) ≤ c) :
    Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc ∧
        ¬Disjoint F (allSlots s.design Pt)) ≤ c := by
  by_cases hd : Disjoint F (allSlots s.design Pt)
  · refine (Claims.Pr_mono (q := fun _ => False) fun p hp => hp.2 hd).trans ?_
    rw [Claims.Pr_false]
    exact (Claims.Pr_nonneg _).trans h1
  · have hpart : D.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
    have hz : s.design.Pa (D.part s) F = 0 := by
      rw [hpart]
      exact NetTiming.SampledAuditProof.classesPa_eq_zero s.design.rules Pt.classes 0 F hd
    have hmiss : Pr (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F) ≤ 0 := by
      refine (Claims.Pr_mono fun r hr => ⟨fun hM => ?_, hr⟩).trans
        (Compiled.pr_miss_le hS hs hPt hin F hF (δ := 0) le_rfl)
      simp only [Design.M, Set.mem_ofPred_eq] at hM
      rw [hz] at hM
      exact lt_irrefl _ hM
    have hsnd : Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => ∀ u ∈ D.checked x p.2, u ∉ F) ≤ 0 :=
      (Claims.SamplingGeneric.Pr_snd (α := Ω)
        (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F)).trans_le hmiss
    have h2 := Claims.Sampling.Pr_or_le
      (p := fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc ∧
        ¬Disjoint F (allSlots s.design Pt))
      (fun p hp => by
        by_cases h : ∃ u ∈ D.checked x p.2, u ∈ F
        · exact Or.inl ⟨hp.1, h⟩
        · exact Or.inr fun u hu hu' => h ⟨u, hu, hu'⟩) h1 hsnd
    linarith

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

/-- **Each accepted phase's record meets its checked leak relation at `WregR`**, except with probability
`e i + εchk i`: where `FregR` meets a class sampled `all`, acceptance is the event `e i` bounds
(`pr_acc_meets_all_le`); where the check fails on `regRunR`, it is the miss `εchk i` bounds (`PhaseChecks`); otherwise
the bridge (`leakAtChecked_regR`) gives the relation with no probability. -/
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
        ∀ g ∈ reg s, (∃ o ∈ ops z ω i s, o.pos = g) ∧ certifierVal (ops z ω i s) g = ρ i g) :
    ∀ z ∈ Zs, ∀ i, Pr (fun ω => phaseAcc H D x Pv hk z i ω (ω i) ∧
      ¬(R z ω i → LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i))
        (WregR H D x Pv ops z ω i) (obs z ω i))) ≤ e i + εchk i := by
  intro z hz i
  let Bad1 : (Fin m → Tp × (Fin kV → Coin)) → Tp × (Fin kV → Coin) → Prop := fun ω c =>
    phaseAcc H D x Pv hk z i ω c ∧ ∃ s Pt, (D (x z ω i)).parse (x z ω i) = some s ∧
      (D (x z ω i)).partition s = some Pt ∧ ¬Disjoint (WregR H D x Pv ops z ω i) (allSlots s.design Pt)
  let Bad2 : (Fin m → Tp × (Fin kV → Coin)) → Tp × (Fin kV → Coin) → Prop := fun ω c =>
    phaseAcc H D x Pv hk z i ω c ∧ ∃ s Pt, (D (x z ω i)).parse (x z ω i) = some s ∧
      (D (x z ω i)).partition s = some Pt ∧ Disjoint (WregR H D x Pv ops z ω i) (allSlots s.design Pt) ∧
      ¬valid (sw i) (regRunR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i) lay k s Pt)
  have h1 : Pr (fun ω => Bad1 ω (ω i)) ≤ e i := by
    refine pr_phase_le i Bad1 (fun ω ω' hω => ?_) (fun ω => ?_)
    · simp only [Bad1]
      rw [phaseAcc_prefix H D x Pv hk had.1 had.2.1 z i ω ω' hω, wreg_prefix H D had z i ω ω' hω,
        had.1 z i ω ω' hω]
    · by_cases hsc : (D (x z ω i)).Scope (x z ω i)
      · obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
        have hc := pr_castLE (hk z ω i) (fun (t : Tp) (r : Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
          (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) t r).out = .acc ∧
            ¬Disjoint (FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) (allSlots s.design Pt))
        refine le_trans (Claims.Pr_mono fun c hc' => ?_) (hc.trans_le
          (pr_acc_meets_all_le (hS (x z ω i)) hs hPt hin Tp (Pv z ω i) _
            (fregR_lt H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) (hev z hz ω i ⟨s, Pt, hs, hPt, hin⟩)))
        obtain ⟨ha, s', Pt', hs', hPt', hd⟩ := hc'
        obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
        obtain rfl : Pt' = Pt := Option.some.inj (hPt'.symm.trans hPt)
        exact ⟨ha, hd⟩
      · refine (Claims.Pr_mono (q := fun _ => False) fun c hc' => hsc ?_).trans
          (by rw [Claims.Pr_false]; exact he0 i)
        exact (hS (x z ω i)).refuses _ _ _ _ _ hc'.1
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
  refine Claims.Sampling.Pr_or_le (fun ω hω => ?_) h1 h2
  obtain ⟨hacc, hnot⟩ := hω
  by_contra hc
  simp only [not_or] at hc
  refine hnot fun hR => ?_
  have hacc' : (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) (ω i).1
      (fun j => (ω i).2 (Fin.castLE (hk z ω i) j))).out = .acc := hacc
  obtain ⟨s, Pt, hs, hPt, hin⟩ := (hS (x z ω i)).refuses _ _ _ _ _ hacc'
  obtain ⟨hsame, hPw, hfit, hcert, hcov, hlen, hj, hreg⟩ := hyp z hz ω i s hs hacc hR
  have hdisj : Disjoint (WregR H D x Pv ops z ω i) (allSlots s.design Pt) :=
    by_contra fun hd => hc.1 ⟨hacc, s, Pt, hs, hPt, hd⟩
  have hval : valid (sw i) (regRunR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i) lay k s Pt) :=
    by_contra fun hv => hc.2 ⟨hacc, s, Pt, hs, hPt, hdisj, hv⟩
  have hPtw : Pt = Pw i := Option.some.inj (hPt.symm.trans hPw)
  subst hPtw
  have hl := leakAtChecked_regR lay k reg (fun _ => valid (sw i)) Dj hs hPt hin hfit (Pv z ω i) (ops z ω i)
    (op i (runsBefore (obs z ω) i)) (obs z ω i) hcert hcov hlen hj (fun g hg => (hreg g hg).1) hdisj hval
  exact leakAtChecked_congr lay k reg Dj hsame.1 hsame.2.1 hsame.2.2.1 hsame.2.2.2.1 hsame.2.2.2.2.1 rfl
    (fun g hg => (hreg g hg).2) hl

end Phases

end NetTiming

#print axioms NetTiming.leakAtChecked_regR
#print axioms NetTiming.pr_acc_meets_all_le
#print axioms NetTiming.pr_phase_le
#print axioms NetTiming.phaseLeaksChecked_regR
