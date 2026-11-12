module

public import Definitions.NetworkCertifier.PooledReg
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Sampling.ProgReads
public import Proofs.Flock.SamplingGeneric.TwoReg
public import Proofs.NetworkCertifier.SampledAuditAccepted
public import Proofs.NetworkCertifier.SampledLeak
public import Proofs.NetworkCertifier.TapRows

/-!
# The pooled bound at the trace with the registered openings: the leak relation

The leak relation at an execution phase's incorrect set holds with no probability once the set is the one of a trace
that carries the network certifier's record.

* `leakAt_of_trace`: at any trace whose transfer gates carry the observations, and any set that holds its incorrect
  proof blocks, the observations meet the leak relation at that set (`LeakAt`). `leakAt_congr`: the relation reads a
  statement through its program, layout, steps and registered inputs only.
* `leakAt_regR`: the trace read at the statement with the registered openings (`regValR`), filled at the network
  certifier's values, is such a trace where the network certifier registers the observations among the registered
  openings, so the observations meet the leak relation at `FregR`.
* `phaseLeaks_regR`: over a monitoring period, an execution phase is accepted while its record fails the leak
  relation at its `WregR` with probability `0`, the form `NetTiming.pr_pooled_le` takes each phase's leak relation in.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The leak relation at a trace's incorrect set -/

/-- **The leak relation at a trace, at any set that holds its incorrect proof blocks**: where every proof block
outside `F` is correct at the trace `tr` at some interior, the trace's transfer gates carry the observations `o`, every
status is complete and the trace holds `ρ` at the registered free inputs, `o` meets the leak relation at `F`. -/
theorem leakAt_of_trace {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (s : AuditStatement) (Pt : Partition) (hin : InScope Pt s)
    (hfit : (lay s).Fits s.prog (k s)) (tr : ℕ → Option Bool) (F : Finset ℕ) (ι : ℕ → ℕ → Bool)
    (hι : ∀ v u, Pt.blocks[v]? = some u → v ∉ F → s.prog.Correct (s.prog.refIn u tr (ι v)) u)
    (ρ : ℕ → Bool) (hρ : ∀ g ∈ reg s, g ∈ freeInputs s.prog (lay s) (k s) → traceVal s.prog tr g = ρ g)
    (op : ℕ → List (Obs Lk K S) → List Bool) (o : List (Obs Lk K S))
    (hcar : Carries (lay s) s.prog tr op (k s) o) (hcov : Covered (k s) o) (hlen : o.length = k s)
    (hj : ∀ y ∈ o, y.jitter < Dj) :
    LeakAt lay k reg Dj s Pt ρ op F o := by
  have hacc := NetTiming.SampledAuditProof.accepted_of (P := Pt) (F := F) hin.1 hin.2.1 hfit ι hι hcar hcov
  refine ⟨hfit, hlen, hj, fun g => if g ∈ freeInputs s.prog (lay s) (k s) then traceVal s.prog tr g else false,
    ⟨fun g hg hgf => ?_, fun g hg => ?_⟩,
    heldAdv s.prog Pt F tr, fun g b h => NetTiming.SampledAuditProof.faultyGate_of_heldAdv h, ?_⟩
  · simp only [hgf, ite_true]
    exact hρ g hg hgf
  · simp only [hg, ite_false]
  intro t ht
  obtain ⟨h, hr, hst⟩ := hacc t ht
  refine ⟨h, ?_, hst⟩
  have hv : valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) (traceVal s.prog tr))
        (heldAdv s.prog Pt F tr) =
      valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t)
        fun g => if g ∈ freeInputs s.prog (lay s) (k s) then traceVal s.prog tr g else false)
        (heldAdv s.prog Pt F tr) :=
    funext (NetTiming.valOver_congr_input s.prog _ fun i hi =>
      NetTiming.inputOf_restrict s.prog (lay s) (k s) _ (traceVal s.prog tr) hi)
  rw [hr]
  simp only [circuitProgram, hv]

/-- The leak relation reads a statement through its program, layout, steps and registered inputs, and the registered
values at the registered inputs only. -/
theorem leakAt_congr {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {s s₀ : AuditStatement} {Pt : Partition} {ρ ρ' : ℕ → Bool}
    {op : ℕ → List (Obs Lk K S) → List Bool} {F : Finset ℕ} {o : List (Obs Lk K S)}
    (hprog : s.prog = s₀.prog) (hlay : lay s = lay s₀) (hk : k s = k s₀) (hreg : reg s = reg s₀)
    (hρ : ∀ g ∈ reg s, ρ g = ρ' g) (h : LeakAt lay k reg Dj s Pt ρ op F o) :
    LeakAt lay k reg Dj s₀ Pt ρ' op F o := by
  unfold LeakAt at h ⊢
  rw [← hprog, ← hlay, ← hk, ← hreg]
  obtain ⟨h1, h2, h3, a, ha, hrest⟩ := h
  exact ⟨h1, h2, h3, a, ⟨fun g hg hgf => (ha.1 g hg hgf).trans (hρ g hg), ha.2⟩, hrest⟩

/-! ## At the trace with the registered openings -/

/-- **The leak relation at `FregR`, no probability**: at a statement in the audit's scope whose layout fits, where the
network certifier registers the observations `o` among the registered openings and every registered input has one,
`o` meets the leak relation at `FregR`, at the registered openings' values. -/
theorem leakAt_regR {H : Msg → Msg} {D : Compiled Coin A} {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {x : Msg} {s : AuditStatement}
    {Pt : Partition} (hs : D.parse x = some s) (hPt : D.partition s = some Pt) (hin : InScope Pt s)
    (hfit : (lay s).Fits s.prog (k s)) {Ω : Type} [Fintype Ω] (P : Prover Ω)
    (ops : AuditStatement → List (Opening Bool)) (op : ℕ → List (Obs Lk K S) → List Bool) (o : List (Obs Lk K S))
    (hcert : CertifierRegisters H D.commit (lay s) (k s) op s (ops s) o) (hcov : Covered (k s) o)
    (hlen : o.length = k s) (hj : ∀ y ∈ o, y.jitter < Dj) (hregd : ∀ g ∈ reg s, ∃ q ∈ ops s, q.pos = g) :
    LeakAt lay k reg Dj s Pt (certifierVal (ops s)) op (FregR H D P x ops) o := by
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
  refine leakAt_of_trace lay k reg Dj s Pt hin hfit
    (fillTrace (regValR H D P x (ops s)) (certifierVal (ops s))) (FregR H D P x ops) ι
    (fun v u hu hv => Prog.correct_mono (Prog.refIn_mono NetTiming.TapProof.fillTrace_ext) (hι v u hu hv))
    (certifierVal (ops s)) (fun g hg hgf => ?_) op o
    (NetTiming.TapProof.carries_fillTrace hfit hcert hag) hcov hlen hj
  have hinp : s.prog.gates[g]? = some .input := by
    simp only [freeInputs, mem_filter] at hgf
    exact hgf.2.1
  exact NetTiming.TapProof.traceVal_fillTrace (fun b hb => by rw [hinp] at hb; cases hb) (hag g (hregd g hg))

/-! ## The monitoring period -/

section Phases

variable {m kV : ℕ} {Tp Sec : Type}

/-- **Each accepted execution phase's record meets its leak relation at `WregR`**: over a monitoring period whose
every accepted phase's statement is registered as `sw i` with the partition `Pw i`, with a layout that fits, and at
which the network certifier registers its record among the phase's registered openings `ops z ω i`, which open every
registered input at the registered value `ρ i`, a phase is accepted while its record fails the leak relation at its
incorrect set with probability at most `0`. Its conclusion is the form `NetTiming.pr_pooled_le` takes each phase's
leak relation in. -/
theorem phaseLeaks_regR [Fintype Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A} (hS : ∀ y, (D y).Sessions H)
    {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
    (Zs : Set Sec) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S))
    (hyp : ∀ z ∈ Zs, ∀ ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      SameRegistration (D (x z ω i)) lay k reg (sw i) s ∧ (D (x z ω i)).partition s = some (Pw i) ∧
        (lay s).Fits s.prog (k s) ∧
        CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
          (obs z ω i) ∧
        Covered (k s) (obs z ω i) ∧ (obs z ω i).length = k s ∧ (∀ y ∈ obs z ω i, y.jitter < Dj) ∧
        ∀ g ∈ reg s, (∃ o ∈ ops z ω i s, o.pos = g) ∧ certifierVal (ops z ω i s) g = ρ i g) :
    ∀ z ∈ Zs, ∀ i, Pr (fun ω => phaseAcc H D x Pv hk z i ω (ω i) ∧
      ¬LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i)) (WregR H D x Pv ops z ω i)
        (obs z ω i)) ≤ 0 := by
  intro z hz i
  refine (Claims.Pr_mono (q := fun _ => False) fun ω hω => ?_).trans_eq Claims.Pr_false
  obtain ⟨hacc, hnot⟩ := hω
  have hacc' : (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) (ω i).1
      (fun j => (ω i).2 (Fin.castLE (hk z ω i) j))).out = .acc := hacc
  obtain ⟨s, Pt, hs, hPt, hin⟩ := (hS (x z ω i)).refuses _ _ _ _ _ hacc'
  obtain ⟨hsame, hPw, hfit, hcert, hcov, hlen, hj, hreg⟩ := hyp z hz ω i s hs hacc
  have hPtw : Pt = Pw i := Option.some.inj (hPt.symm.trans hPw)
  subst hPtw
  have hl := leakAt_regR lay k reg Dj hs hPt hin hfit (Pv z ω i) (ops z ω i) _ _ hcert hcov hlen hj
    (fun g hg => (hreg g hg).1)
  exact hnot (leakAt_congr lay k reg Dj hsame.1 hsame.2.2.1 hsame.2.2.2.1 hsame.2.2.2.2.1
    (fun g hg => (hreg g hg).2) hl)

end Phases

end NetTiming
