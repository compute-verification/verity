module

public import Definitions.Claims.Encodings
public import Definitions.Claims.SamplingGenericX
public import Proofs.Flock.SamplingGeneric.Reg
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.SamplingGeneric.Assemble

/-!
# The committed form's event, at a table of the statement

The first half of the committed form, proved once, at the incorrect set of any table of the statement
(`Encodings.FregAt`): a trace for each statement, which may read the prover's tape.

* `event_le_of_off`: an accepted run samples a proof block that no values agreeing with the table on the program's
  gates make correct with probability at most `κ_s + √(k·γ_s·Pr[B]) + b_I + b_O`, where `b_I` bounds one run's own
  disagreement and `b_O` the accepted runs whose extractions open a gate off the table. It is the deterministic half
  at a trace (`trace_split`), the sessions' bound at the first sampled block of the set (`pr_unfound_le` at
  `leadSel (selOf …)`) and a union bound. Which table it is, only the two bounds know, and they are its hypotheses.
* `Freg_eq_FregAt`, `FregX_eq_FregAt`: the committed trace's incorrect set, and the incorrect set of the trace read at
  the statement, are `FregAt` at those traces. So the per-tape form (`ksReg_of_sessionKS`, below), the form at the
  trace read at the statement (`ksRegX_of_sessionKS`, `SamplingGeneric/RegX.lean`) and the form that counts each bad
  run at one gate (`pr_regHit_le_canon_on`, `SamplingGeneric/RegCanon.lean`) are `event_le_of_off` at their tables,
  each with its own two bounds.
* `ksReg_of_sessionKS`, `compilerKSReg_of`: **`KSReg`**: the event at the committed trace, with `B_conf` for one run's
  disagreement and the rewinding lemma for the gates opened off the trace; and on the sample, `miss_of_event` at
  `Freg`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement InScope)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The incorrect sets, as the set at a table -/

/-- **The committed trace's incorrect set is the set at a table**: `Freg` at the tape `ω` is `Encodings.FregAt` at
the table that gives every statement the committed trace at `ω`. -/
theorem Freg_eq_FregAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    L.Freg H P x ω = Encodings.FregAt L x (fun _ => L.regVal H P x ω) := by
  ext v
  cases hs : L.parse x with
  | none => simp only [Freg, Encodings.FregAt, hs]
  | some s => simp only [Freg, Encodings.FregAt, hs, mem_filter, mem_range]

/-- **The incorrect set of the trace read at the statement is the set at a table**: `FregX` is `Encodings.FregAt` at
the table that gives every statement the trace read at the statement. -/
theorem FregX_eq_FregAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) :
    L.FregX H P x = Encodings.FregAt L x (fun _ => L.regValX H P x) := by
  ext v
  cases hs : L.parse x with
  | none => simp only [FregX, Encodings.FregAt, hs]
  | some s => simp only [FregX, Encodings.FregAt, hs, mem_filter, mem_range]

/-! ## The event's bound from bounds on its two binding terms -/

set_option synthInstance.maxSize 1024 in
/-- **The event's bound from bounds on its two binding terms**, at a table `T` of the statement that may read the
tape: an accepted run samples a proof block incorrect at the table (`Encodings.FregAt`) with probability at most
`κ_s + √(k·γ_s·Pr[B]) + b_I + b_O`, where `b_I` bounds one run's own disagreement and `b_O` the accepted runs whose
extractions open a gate off the table. At such a run the first sampled block of the set has its session unfound at its
first slot (the sessions' bound at `leadSel (selOf …)`), or the run's extractions disagree with themselves or open a
gate off the table (`trace_split`). The contract's clauses it reads are `slots`, `shared` and `opens`. -/
theorem event_le_of_off {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) (hSK : L.SessionKS H) {x : Msg}
    {s : AuditStatement} {Pt : Sampling.Partition} (hs : L.parse x = some s) (hPt : L.partition s = some Pt)
    (hin : InScope Pt s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (T : Ω → AuditStatement → ℕ → Option Bool) {bI bO : ℝ}
    (hI : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) ≤ bI)
    (hO : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      ∃ g < L.nG x, ∃ b, L.opensAt H P x q.1 q.2.1 q.2.2 g = some b ∧ T q.1 s g ≠ some b) ≤ bO) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ Encodings.FregAt L x (T p.1)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + bI + bO := by
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hpart : L.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  have hFlt : ∀ ω, ∀ v ∈ Encodings.FregAt L x (T ω), v < L.nPU x := by
    intro ω v hv
    simp only [Encodings.FregAt, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1
  have hFno : ∀ ω, ∀ v ∈ Encodings.FregAt L x (T ω), ¬ ∃ b, Pt.blocks[v]? = some b ∧
      ∃ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ g < s.prog.gates.length, val g = T ω s g) ∧ s.prog.Correct (s.prog.refIn b val ι) b := by
    intro ω v hv
    simp only [Encodings.FregAt, hs, mem_filter, mem_range] at hv
    rw [hpart] at hv
    exact hv.2
  have hper : ∀ ω, Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ Encodings.FregAt L x (T ω)) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (Encodings.FregAt L x (T ω))) (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) +
        Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
          ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ T ω s g ≠ some b)) := by
    intro ω
    have hIO := Sampling.Pr_union_le
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ T ω s g ≠ some b)
    refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ Encodings.FregAt L x (T ω))).symm.trans_le
        (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl hIO)
    obtain ⟨hacc, v, hvl, hvF⟩ := hrc
    obtain ⟨i, hi, u, hiu, huF⟩ := L.selOf_some x (hS.slots x rc.1) (hFlt ω) hvl hvF
    obtain ⟨i₀, u₀, hlead, hu₀⟩ := leadOf_some hiu u.isLt
    rcases trace_split hS hs hPt hin P ω rc.1 rc.2 hacc i u hiu (T ω s) (hFno ω _ huF) i₀ u₀ hlead hu₀
      with h | h | h
    · exact Or.inl ⟨hacc, i₀, by rw [leadSel, hi, Option.bind_some, hlead], u₀, hu₀, h⟩
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr ⟨hacc, h⟩)
  have hT := L.pr_unfound_le H P β hβ0 hβ (fun ω => L.leadSel x (L.selOf x (Encodings.FregAt L x (T ω))))
    fun ω r => hκ ω r (L.leadSel x (L.selOf x (Encodings.FregAt L x (T ω))))
  have hO' : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ T ω s g ≠ some b) ≤ bO :=
    (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.opensAt H P x q.1 q.2.1 q.2.2 g = some b ∧ T q.1 s g ≠ some b)).symm.trans_le hO
  refine (Sampling.Pr_prod_eq_Ex _).trans_le ((Ex_mono hper).trans ?_)
  rw [Ex_add, Ex_add]
  linarith

/-! ## The committed form -/

set_option synthInstance.maxSize 1024 in
/-- **`KSReg` from the sessions' bound**: `event_le_of_off` at the committed trace of each tape, where one run's own
disagreement is `B_conf`'s collision and a run's extractions open a gate off the committed trace with probability at
most `N·√Pr[conflict]` at each tape (the rewinding lemma), at most `N·√Pr[B_conf]` on average (Jensen's inequality);
and a set refused at `δ` is missed with probability at most `P_a ≤ δ` (`miss_of_event`). -/
theorem ksReg_of_sessionKS {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) : L.KSReg H := by
  intro x hsc Ω _ _ P
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  have hFlt : ∀ ω, ∀ v ∈ L.Freg H P x ω, v < L.nPU x := by
    intro ω v hv
    simp only [Freg, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1
  have hC : Ex (fun ω : Ω => Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
      ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) ≤ Real.sqrt ((Bconf H L Ω P x).Pr H) :=
    (Ex_sqrt_le _ fun _ => Pr_nonneg _).trans (Real.sqrt_le_sqrt (pr_conflict_le hbind hs P))
  have hO : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffReg H P x q.1 q.2.1 q.2.2) ≤ L.nG x * Real.sqrt ((Bconf H L Ω P x).Pr H) :=
    calc Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
          L.OffReg H P x q.1 q.2.1 q.2.2)
        ≤ Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) :=
          Pr_mono fun _ h => h.2
      _ = Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) :=
          Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            L.OffReg H P x q.1 q.2.1 q.2.2)
      _ ≤ Ex (fun ω : Ω => L.nG x * Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
            ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) :=
          Ex_mono fun ω => L.pr_offReg_le H P x ω
      _ = L.nG x * Ex (fun ω : Ω => Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
            ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) := Ex_mul_left _ _
      _ ≤ L.nG x * Real.sqrt ((Bconf H L Ω P x).Pr H) := mul_le_mul_of_nonneg_left hC (Nat.cast_nonneg _)
  have h1 : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤ L.εreg H x P := by
    simp only [Freg_eq_FregAt]
    refine (event_le_of_off hS hSK hs hPt hin P (fun ω _ => L.regVal H P x ω) (pr_inConf_le hbind hs P) hO).trans ?_
    unfold Compiled.εreg
    linarith
  refine ⟨h1, fun s' hs' δ hδ => ?_⟩
  rw [show s' = s from Option.some.inj (hs'.symm.trans hs)]
  exact miss_of_event hS hs hPt hin P (L.Freg H P x) hFlt h1 hδ

end Compiled

/-- **The committed form** from the sessions' bound. -/
theorem compilerKSReg_of (hsess : SessionKSOf) : CompilerKSReg := by
  intro Coin _ _ _ A L H hA hB hS
  exact Compiled.ksReg_of_sessionKS hB hS (hsess Coin A L H hA hS)

end Claims.SamplingGeneric
