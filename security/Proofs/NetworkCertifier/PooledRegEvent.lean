module

public import Definitions.NetworkCertifier.PooledReg
public import Proofs.Flock.Claims.Finder
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Sampling.Binding
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.Claims.Sampling.Wait
public import Proofs.Flock.SamplingGeneric.Assemble
public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.Fails
public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Flock.SamplingGeneric.RegTrace
public import Proofs.Flock.SamplingGeneric.RegX
public import Proofs.Flock.SamplingGeneric.TwoFirstHitAt
public import Proofs.Flock.SamplingGeneric.TwoOffClassAt
public import Proofs.NetworkCertifier.TapRows

/-!
# The pooled bound at the trace with the registered openings: the event

At one execution phase, the event the sample's term is paid against: the run accepts with a sampled proof block of
`FregR`, the incorrect set of the trace read at the statement with the registered openings.

* `ksRegR_of_masses`: on that event the first such block's session is unfound at its first slot, or the run's
  extractions disagree with themselves, or open a gate off the trace read at the statement, or open a gate a
  registered opening names off that opening (`OffOps`). So the event is at most the sessions' bound plus any bounds on
  those three masses. It is the event's bound at a table of the statement (`Compiled.event_le_of_off`) at `regValR`
  (`FregR_eq_FregAt`), whose gates opened off the table are the last two cases.
* `pr_offOps_le`: the fourth mass is at most `Pr[B_ops,X]`, by the commitment's binding.
* `ksRegR_of_sessionKS`: the event is at most `ε_reg,X + Pr[B_ops,X]`, its binding terms in the square-root form;
  `εregX_nonneg`: `ε_reg,X` is not negative.
* `ksRegR_of_costs`: the event with its binding terms from costs, with no square root of a collision term.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement InScope Opening ownLog)
open NetTiming (regValR FregR BopsX)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The registered openings' binding term -/

/-- A run's extractions open a gate a registered opening names at another value than that opening's. -/
def OffOps (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ops : List (Opening Bool))
    (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : Prop :=
  ∃ g, (∃ q ∈ ops, q.pos = g) ∧ ∃ b, L.opensAt H P x ω r c g = some b ∧ b ≠ NetTiming.certifierVal ops g

/-- An event of a tape and a run at it that gives a collision `BopsX`'s output is at most `Pr[BopsX]`. -/
theorem le_bopsX (H : Msg → Msg) (L : Compiled Coin A) (ops : AuditStatement → List (Opening Bool)) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {Q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) → Prop}
    (hQ : ∀ q, Q q → ∃ u v, findColl (ownLog H (run H P L.V x q.1)
        (L.runExt x s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2) ++
      NetTiming.certifierLog H L.commit s.roots (ops s)) = some (u, v) ∧ Collision H u v) :
    Pr Q ≤ (BopsX H L ops Ω P x).Pr H := by
  simp only [BopsX, hs]
  exact Pr_mono hQ

/-- **A run's extraction off a registered opening**: at most `Pr[BopsX]`. The extraction's opening and the registered
one pass under the statement's roots at one gate with two values, a collision among the extraction's own calls and the
checks of the registered openings (`CommitmentScheme.Binds`). -/
theorem pr_offOps_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H)
    (ops : AuditStatement → List (Opening Bool)) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
    (hpass : ∀ o ∈ ops s, L.commit.Opens H s.roots o = true) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffOps H P x (ops s) ω rc.1 rc.2) ≤
      (BopsX H L ops Ω P x).Pr H := by
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffOps H P x (ops s) ω rc.1 rc.2)
      = Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffOps H P x (ops s) q.1 q.2.1 q.2.2) :=
        (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          L.OffOps H P x (ops s) q.1 q.2.1 q.2.2)).symm
    _ ≤ (BopsX H L ops Ω P x).Pr H :=
        le_bopsX H L ops hs P fun q hq => by
          obtain ⟨g, hg, b, hb1, hne⟩ := hq
          rw [L.opensAt_eq H P x q.1 hs] at hb1
          obtain ⟨a, ha, har, hap, hav, hao⟩ := val_some_mem hb1
          obtain ⟨o', ho', ho'p, ho'v⟩ := NetTiming.TapProof.certifierVal_mem hg
          obtain ⟨-, hl⟩ :=
            L.passes_runExt H x (run H P L.V x q.1) s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2 a ha
          rw [har] at hl
          have hc : Sampling.LogColl (ownLog H (run H P L.V x q.1)
              (L.runExt x s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2) ++
              NetTiming.certifierLog H L.commit s.roots (ops s)) :=
            Sampling.disagree_collision hbind (hap.trans ho'p.symm) (by rw [hav, ho'v]; exact hne) hao
              (hpass o' ho') (fun t ht => List.mem_append_left _ (hl t ht))
              (fun t ht => List.mem_append_right _ (List.mem_flatMap.2 ⟨o', ho', ht⟩))
          obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (Sampling.findColl_isSome _ hc)
          refine ⟨u, v, huv, findColl_spec H _ (fun p hp => ?_) u v huv⟩
          rcases List.mem_append.1 hp with hp | hp
          · exact Sampling.ownLog_valid H _ _ p hp
          · exact NetTiming.TapProof.certifierLog_valid H L.commit s.roots (ops s) p hp

/-- **The incorrect set with the registered openings is the set at a table**: `FregR` is `Encodings.FregAt` at the
table that gives a statement the trace read at the statement with that statement's registered openings. -/
theorem FregR_eq_FregAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (ops : AuditStatement → List (Opening Bool)) :
    FregR H L P x ops = Encodings.FregAt L x (fun s => regValR H L P x (ops s)) := by
  ext v
  cases hs : L.parse x with
  | none => simp only [FregR, Encodings.FregAt, hs]
  | some s => simp only [FregR, Encodings.FregAt, hs, mem_filter, mem_range]

set_option synthInstance.maxSize 1024 in
/-- **The committed form at the trace with the registered openings, from the three masses**: the event's bound at a
table of the statement (`event_le_of_off`) at `regValR`. A gate a run's extractions open off that table is a gate a
registered opening names, opened off that opening (`OffOps`), or a gate opened off the trace read at the statement.
So the event is at most the sessions' bound plus any bounds `cI`, `cO`, `cP` on one run's own disagreement and those
two masses. It takes no binding: a caller bounds each mass its own way. -/
theorem ksRegR_of_masses {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) (hSK : L.SessionKS H)
    (ops : AuditStatement → List (Opening Bool)) {x : Msg} (hsc : L.Scope x)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) {cI cO cP : ℝ}
    (hI : ∀ s, L.parse x = some s →
      Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) ≤ cI)
    (hO : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2) ≤ cO)
    (hOp : ∀ s, L.parse x = some s →
      Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffOps H P x (ops s) ω rc.1 rc.2) ≤
        cP) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ FregR H L P x ops) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + cI + cO + cP := by
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  have hoff : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      ∃ g < L.nG x, ∃ b, L.opensAt H P x q.1 q.2.1 q.2.2 g = some b ∧ regValR H L P x (ops s) g ≠ some b) ≤
      cO + cP := by
    refine Sampling.Pr_or_le (fun q hq => ?_)
      ((Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.OffRegX H P x q.1 q.2.1 q.2.2)).trans_le hO)
      ((Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.OffOps H P x (ops s) q.1 q.2.1 q.2.2)).trans_le (hOp s hs))
    obtain ⟨-, g, hg, b, hb1, hne⟩ := hq
    by_cases hany : (ops s).any (fun o => o.pos == g) = true
    · refine Or.inr ⟨g, ?_, b, hb1, fun hbe => hne ?_⟩
      · obtain ⟨o, ho, hp⟩ := List.any_eq_true.1 hany
        exact ⟨o, ho, by simpa using hp⟩
      · rw [regValR, ite_eq_left hany, hbe]
    · refine Or.inl ⟨g, hg, b, hb1, ?_⟩
      rw [regValR, ite_eq_right hany] at hne
      exact hne
  simp only [FregR_eq_FregAt]
  exact (event_le_of_off hS hSK hs hPt hin P (fun _ s' => regValR H L P x (ops s')) (hI s hs) hoff).trans_eq
    (add_assoc _ _ _).symm

/-- **The committed form at the trace with the registered openings, in its square-root form**: `ksRegR_of_masses` at
`pr_inConfX_le`, `ex_offRegX_le` and `pr_offOps_le`, so at most `ε_reg,X + Pr[B_ops,X]`. -/
theorem ksRegR_of_sessionKS {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) (ops : AuditStatement → List (Opening Bool)) {x : Msg} (hsc : L.Scope x)
    (hops : ∀ s, L.parse x = some s → ∀ o ∈ ops s, L.commit.Opens H s.roots o = true)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ FregR H L P x ops) ≤ L.εregX H x P + (BopsX H L ops Ω P x).Pr H := by
  have ⟨s, _, hs, _, _⟩ := hsc
  have h := ksRegR_of_masses hS hSK ops hsc Ω P (fun s' hs' => pr_inConfX_le hbind hs' P)
    (ex_offRegX_le hbind hs P) (fun s' hs' => pr_offOps_le hbind ops hs' (hops s' hs') P)
  unfold Compiled.εregX
  linarith

/-- The committed form's error at the trace read at the statement is not negative. -/
theorem εregX_nonneg (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : 0 ≤ L.εregX H x P := by
  unfold Compiled.εregX
  exact add_nonneg (add_nonneg (add_nonneg (L.κs_nonneg x) (Real.sqrt_nonneg _))
    (mul_nonneg (Nat.cast_nonneg _) (Real.sqrt_nonneg _))) (Pr_nonneg _)

/-! ## The binding terms from costs -/

/-- **The committed form at the trace with the registered openings, with no square root of a collision term**:
`ksRegR_of_masses` at `bconfX_le_at` and `pr_offRegX_le_cls_at`. Where a run's extractions open a class of gates whole
or not at all (`hallX`), every run calls `H` at most `T` times, and `B_conf,X` and the by-class finders meet the
linear expected-time bound, an accepted run samples a proof block of `FregR` with probability at most
`κ_s + √(k·γ_s·Pr[B]) + 2·runCost/2^256 + 2·firstHitCost/2^256 + Pr[B_ops,X]`. -/
theorem ksRegR_of_costs {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) (hopen : L.OpenCallsLe H) (ops : AuditStatement → List (Opening Bool)) {x : Msg}
    (hsc : L.Scope x) (hops : ∀ s, L.parse x = some s → ∀ o ∈ ops s, L.commit.Opens H s.roots o = true)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ s, L.parse x = some s → ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (BconfX H L Ω P x))
    (hX : ∀ s, L.parse x = some s → ∀ K : ℕ, ExpectedCR H (L.finderXCls H P x s needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ FregR H L P x ops) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 * L.runCost x T Ns / 2 ^ (256 : ℝ) +
        2 * L.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) + (BopsX H L ops Ω P x).Pr H := by
  have ⟨s, _, hs, _, _⟩ := hsc
  exact ksRegR_of_masses hS hSK ops hsc Ω P
    (fun s' hs' => (pr_inConfX_le hbind hs' P).trans (bconfX_le_at hS.toCostContract hopen x P T hT hNs hB))
    ((Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.OffRegX H P x q.1 q.2.1 q.2.2)).symm.trans_le
      (pr_offRegX_le_cls_at hbind hS.toCostContract hopen hs P needs (hpos s hs) cls hallX T hT hNs hKs hKm
        (hX s hs)))
    (fun s' hs' => pr_offOps_le hbind ops hs' (hops s' hs') P)

end Compiled

end Claims.SamplingGeneric
