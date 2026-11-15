module

public import Definitions.Certifier.Guarantees.PooledRegChecked
public import Proofs.Certifier.PooledReg
public import Proofs.Certifier.Carried
public import Proofs.Certifier.MonitoringPeriodChecked

/-!
# DRAFT (scratch branch, in no root): the lemma chain for `PooledLeakRegisteredChecked`

Each `sorry` here is a named gap of the draft, not a proof.

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
8. `encard_pooledReg_le` (new, routine): the pooled count in the `#Θ` form.
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
  sorry

/-- **One phase's bound at every history lifts to the period's coins.** -/
theorem pr_phase_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} (i : Fin m) (E : (Fin m → Ω) → Ω → Prop)
    (hE : ∀ ω ω', (∀ j, j < i → ω j = ω' j) → E ω = E ω') {c : ℝ} (h : ∀ ω, Pr (E ω) ≤ c) :
    Pr (fun ω => E ω (ω i)) ≤ c := by
  sorry

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
    {e εchk : Fin m → ℝ}
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
  sorry

end Phases

end NetTiming
