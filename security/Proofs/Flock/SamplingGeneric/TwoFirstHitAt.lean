module

public import Proofs.Flock.SamplingGeneric.TwoRegFirstHit
public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.TwoKSAt
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.Claims.Sampling.Extract

/-!
# The two-stage first-hit bound at a budget a run, as one closed right-hand side

`eventTwoX_le_firstHit_at` bounds the event of `KSRegTwoX` by `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X]` and
`4·E[c + Σ_{g ∈ sample} c_g]/2^256`, where `c` is a run's SHA-512 calls with every session of its sample extracted
(`Compiled.callsAt`) and `c_g` its calls with only the sessions that have a proof block needing gate `g` extracted
(`Compiled.callsBy`). This file bounds that expectation by a closed expression, `Compiled.firstHitCost`, where each run
calls SHA-512 at most `T` times, both parties':

`E[c + Σ_{g ∈ sample} c_g] ≤ T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls`.

* `N_s` bounds the expected number of sessions in a run's sample (`Compiled.firstSlots`; `n_sess` is one such bound,
  `ex_firstSlots_le`), `K_s` the expected number of gates the sample needs (`Compiled.sampleNeeds`), and `K_m` the
  expected number of gates its sessions need, session by session (`Compiled.sessCount`); the three are hypotheses on
  the coins alone.
* A run's calls are at most `T`, its extractions' own calls and `T` for each of their reruns (`callsAt_le_at`,
  `callsBy_le_at`, from `Sampling.calls_le_own`); the own calls are at most `n_PU·openCalls` (`runExt_own_le`).
* At the gates of the sample, a session's reruns count once for each gate its proof blocks need
  (`sum_runExtAt_runs_le`). A session's expected reruns are at most `e_s` over its own rounds, on which the sample
  does not depend (`RunsFixed.fixed`), so a weight of the sample stays outside the expectation
  (`slot_exp_le_weighted`, `ex_weighted_runs_le`): the gates the session needs at a further run, and `1` at the first
  run.

The gates a finder pays for at a further run need not be every gate the sample needs: `ex_calls_le_at_in` is the same
bound at any set of gates of the sample, with `K_s` the expected size of the set and `K_m` the expected number of its
gates the sample's sessions need, session by session (`Compiled.sessCountIn`, `sum_runExtAt_runs_le_in`), and
`ex_calls_le_at_cls` reads it at the representatives of the classes of the gates the sample needs, the set a by-class
first-hit finder pays for.

`CompiledTwo.eventTwoX_le_at` is the event's bound at that right-hand side, and `eventTwoX_le_at_cap` the same at one
cap on each finder's waits, with the caps' terms. `B_conf,X`'s cost is its two runs' calls, so under the linear
expected-time bound at it `Pr[B_conf,X] ≤ 2·(T·(1 + e_s·N_s) + n_PU·openCalls)/2^256` (`Compiled.runCost`,
`Compiled.bconfX_le_at`), and `eventTwoX_le_at_budget` reads the event's bound with that term at the budget too.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling

/-! ## Sums over the parts of a list that filters keep -/

/-- Over the part of a list a filter keeps, a sum is at most the sum over the list. -/
theorem sum_map_filter_le {β : Type} (keep : β → Bool) (F : β → ℕ) (l : List β) :
    ((l.filter keep).map F).sum ≤ (l.map F).sum := by
  induction l with
  | nil => exact le_rfl
  | cons b bs ih =>
    by_cases hb : keep b = true
    · rw [List.filter_cons_of_pos hb, List.map_cons, List.map_cons, List.sum_cons, List.sum_cons]
      exact Nat.add_le_add_left ih _
    · rw [List.filter_cons_of_neg hb, List.map_cons, List.sum_cons]
      exact ih.trans (Nat.le_add_left _ _)

/-- **Sums over the parts of a list that a set of filters keeps**: summed over the filters, they weigh each element
of the list once for each filter that keeps it. -/
theorem sum_filter_map_card {G β : Type} (S : Finset G) (keep : G → β → Bool) (F : β → ℕ) (l : List β) :
    ∑ g ∈ S, ((l.filter (keep g)).map F).sum = (l.map fun b => (S.filter fun g => keep g b).card * F b).sum := by
  induction l with
  | nil => simp
  | cons b bs ih =>
    have hb : ∀ g, (((b :: bs).filter (keep g)).map F).sum =
        (if keep g b = true then F b else 0) + ((bs.filter (keep g)).map F).sum := by
      intro g
      by_cases hg : keep g b = true
      · rw [List.filter_cons_of_pos hg, List.map_cons, List.sum_cons, ite_eq_left hg]
      · rw [List.filter_cons_of_neg hg, ite_eq_right hg, Nat.zero_add]
    simp only [hb]
    rw [Finset.sum_add_distrib, ih, List.map_cons, List.sum_cons, ← Finset.sum_filter, Finset.sum_const,
      smul_eq_mul]

/-- An average over a tape, coins and further coins of a function of the coins is the average over the coins. -/
theorem Ex_coins {Ω R C : Type} [Fintype Ω] [Nonempty Ω] [Fintype R] [Fintype C] [Nonempty C] (f : R → ℝ) :
    Ex (fun q : Ω × (R × C) => f q.2.1) = Ex f := by
  rw [Verity.FirstHit.ex_tape fun (_ : Ω) (rc : R × C) => f rc.1, Ex_const,
    Verity.FirstHit.ex_tape fun (r : R) (_ : C) => f r]
  exact congrArg Ex (funext fun r => Ex_const (f r))

/-- Two independent draws: the average of a function at the first plus the function at the second is twice the
function's average. -/
theorem Ex_two {β : Type} [Fintype β] [Nonempty β] (F : β → ℝ) :
    Ex (fun q : β × β => F q.1 + F q.2) = 2 * Ex F := by
  have h1 : Ex (fun q : β × β => F q.1) = Ex F :=
    (Ex_prod fun q : β × β => F q.1).trans (congrArg Ex (funext fun a => Ex_const (α := β) (F a)))
  have h2 : Ex (fun q : β × β => F q.2) = Ex F :=
    (Ex_prod fun q : β × β => F q.2).trans (Ex_const (α := β) (Ex F))
  rw [Ex_add, h1, h2, two_mul]

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The counts the bound reads -/

/-- **The gates a sample needs**: the program's gates that some proof block of the sample `l` needs. -/
def sampleNeeds (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ) (l : List ℕ) : Finset (Fin (L.nG x)) :=
  Finset.univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ l, g.val ∈ needs u)

/-- **The gates a session needs**: the program's gates that some proof block of slot `i`'s session in the sample `l`
needs. -/
def sessNeeds (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ) (l : List ℕ) (i : ℕ) : Finset (Fin (L.nG x)) :=
  Finset.univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u)

/-- **A sample's sessions**, each at its first slot: the slots of the sample `l` that are their session's first. -/
def firstSlots (L : Compiled Coin A) (x : Msg) (l : List ℕ) : Finset (Fin (L.nPU x)) :=
  Finset.univ.filter fun i : Fin (L.nPU x) => (L.sessSlots x l i.val).head? = some i.val

/-- **The gates a sample's sessions need, session by session**: over the sessions of the sample `l`, each at its
first slot, the number of gates the session needs (`sessNeeds`). A gate two sessions need counts twice. -/
def sessCount (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ) (l : List ℕ) : ℕ :=
  ∑ i ∈ L.firstSlots x l, (L.sessNeeds x needs l i.val).card

/-- **The gates of a set a sample's sessions need, session by session**: over the sessions of the sample `l`, each at
its first slot, the number of gates of the set `G l` that the session's proof blocks need. A gate two sessions need
counts twice. -/
def sessCountIn (L : Compiled Coin A) (x : Msg) (needs : ℕ → Finset ℕ) (G : List ℕ → Finset (Fin (L.nG x)))
    (l : List ℕ) : ℕ :=
  ∑ i ∈ L.firstSlots x l,
    ((G l).filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.sessBlocks x l i.val, g.val ∈ needs u)).card

/-- **A run's expected calls at a budget `T`**, in closed form: `T·(1 + e_s·N_s) + n_PU·openCalls`, the run with
every session of its sample extracted, where `N_s` bounds the expected number of sessions in a run's sample. -/
noncomputable def runCost (L : Compiled Coin A) (x : Msg) (T : ℕ) (Ns : ℝ) : ℝ :=
  T * (1 + L.es x * Ns) + (L.nPU x : ℝ) * L.openCalls x

/-- **The first-hit bound's calls at a budget `T` a run**, in closed form:
`T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls`, where `N_s` bounds the expected number of sessions in a
run's sample, `K_s` the expected number of gates the sample needs and `K_m` the expected number its sessions need,
session by session. -/
noncomputable def firstHitCost (L : Compiled Coin A) (x : Msg) (T : ℕ) (Ns Ks Km : ℝ) : ℝ :=
  T * (1 + L.es x * Ns + Ks + L.es x * Km) + (1 + Ks) * ((L.nPU x : ℝ) * L.openCalls x)

/-! ## A run's calls at a budget -/

/-- At a gate, a run's extractions call `H` themselves at most as often as its extractions at every gate. -/
theorem runExtAt_own_le (H : Msg → Msg) (L : Compiled Coin A) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (s : AuditStatement) (needs : ℕ → Finset ℕ) (g : ℕ)
    (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) :
    (ownLog H run' (L.runExtAt x s needs g r v c)).length ≤ (ownLog H run' (L.runExt x s r v c)).length := by
  unfold runExtAt runExt
  rw [ownLog_flatMapRP, ownLog_flatMapRP]
  exact sum_map_filter_le _ _ _

/-- **A run's calls at a budget**: where each run calls `H` at most `T` times, both parties', a run's calls with
every session of its sample extracted are at most `T`, its extractions' own calls and `T` for each of their
reruns. -/
theorem callsAt_le_at (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (T : ℕ) (ω : Ω) (hT : ∀ r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    L.callsAt H P x s ω r c ≤
      T + (ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length +
        T * ((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 := by
  have h := calls_le_own H (run H P L.V x ω) T hT (L.runExt x s r (run H P L.V x ω r).seen c)
  have h' := hT r
  unfold callsAt
  omega

/-- **A run's calls at one gate, at a budget**: at most `T`, its extractions' own calls at every gate, and `T` for
each rerun of its extractions at the gate. -/
theorem callsBy_le_at (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (needs : ℕ → Finset ℕ) (T : ℕ) (ω : Ω)
    (hT : ∀ r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) (g : ℕ)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    L.callsBy H P x s needs g ω r c ≤
      T + (ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length +
        T * ((L.runExtAt x s needs g r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 := by
  have h := calls_le_own H (run H P L.V x ω) T hT (L.runExtAt x s needs g r (run H P L.V x ω r).seen c)
  have h' := hT r
  have h'' := L.runExtAt_own_le H x (run H P L.V x ω) s needs g r (run H P L.V x ω r).seen c
  unfold callsBy
  omega

/-! ## The reruns at the gates of a set -/

/-- **The reruns of a run's extractions at the gates of a set, summed**: each slot's reruns (`slotRuns`), once for
each gate of the set that its session's proof blocks need, so at most as often as its session needs gates
(`sessNeeds`). -/
theorem sum_runExtAt_runs_le (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (needs : ℕ → Finset ℕ) (S : Finset (Fin (L.nG x)))
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    ((∑ g ∈ S, ((L.runExtAt x s needs g.val r (run' r).seen c).eval H run').2.2 : ℕ) : ℝ) ≤
      ∑ i : Fin (L.nPU x),
        ((L.sessNeeds x needs (L.checked x r) i.val).card : ℝ) * L.slotRuns H x s run' i r (c i) := by
  have hmid := sum_filter_map_card S
    (fun (g : Fin (L.nG x)) (iu : Fin (L.nPU x) × ℕ) =>
      decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u))
    (fun iu : Fin (L.nPU x) × ℕ =>
      ((if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
          L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)
        else .ret []).eval H run').2.2)
    ((List.finRange (L.nPU x)).zip (L.checked x r))
  have hsum : ∑ g ∈ S, ((L.runExtAt x s needs g.val r (run' r).seen c).eval H run').2.2 =
      (((List.finRange (L.nPU x)).zip (L.checked x r)).map fun iu : Fin (L.nPU x) × ℕ =>
        (S.filter fun g : Fin (L.nG x) =>
            decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u)).card *
          ((if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
              L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)
            else .ret []).eval H run').2.2).sum := by
    refine Eq.trans (Finset.sum_congr rfl fun g _ => ?_) hmid
    unfold runExtAt
    exact flatMapRP_runs H run' _ _
  rw [hsum, Nat.cast_list_sum, List.map_map]
  refine le_of_eq_of_le (Eq.trans ?_ (sum_zip_finRange (L.nPU x) (fun iu : Fin (L.nPU x) × ℕ =>
    (((S.filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u)).card : ℕ) : ℝ) *
      if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
        (((L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval H
          run').2.2 : ℝ)
      else 0) (L.checked x r))) (Finset.sum_le_sum fun i _ => ?_)
  · refine congrArg List.sum (List.map_congr_left fun iu _ => ?_)
    simp only [Function.comp_apply, Nat.cast_mul]
    congr 1
    split
    · rfl
    · exact Nat.cast_zero
  · unfold slotRuns sessNeeds
    cases hl : (L.checked x r)[i.val]? with
    | none => simp
    | some u =>
      simp only [Option.map_some, Option.getD_some]
      refine mul_le_mul_of_nonneg_right (Nat.cast_le.2 (Finset.card_le_card
        (Finset.filter_subset_filter _ (Finset.subset_univ _)))) ?_
      split
      · exact Nat.cast_nonneg _
      · exact le_rfl

/-- **The reruns of a run's extractions at the gates of a set, summed, counted in the set**: each slot's reruns
(`slotRuns`), once for each gate of the set that its session's proof blocks need. `sum_runExtAt_runs_le` counts every
gate the session needs, in the set or not. -/
theorem sum_runExtAt_runs_le_in (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (needs : ℕ → Finset ℕ) (S : Finset (Fin (L.nG x)))
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    ((∑ g ∈ S, ((L.runExtAt x s needs g.val r (run' r).seen c).eval H run').2.2 : ℕ) : ℝ) ≤
      ∑ i : Fin (L.nPU x),
        (((S.filter fun g : Fin (L.nG x) =>
            decide (∃ u ∈ L.sessBlocks x (L.checked x r) i.val, g.val ∈ needs u)).card : ℕ) : ℝ) *
          L.slotRuns H x s run' i r (c i) := by
  have hmid := sum_filter_map_card S
    (fun (g : Fin (L.nG x)) (iu : Fin (L.nPU x) × ℕ) =>
      decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u))
    (fun iu : Fin (L.nPU x) × ℕ =>
      ((if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
          L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)
        else .ret []).eval H run').2.2)
    ((List.finRange (L.nPU x)).zip (L.checked x r))
  have hsum : ∑ g ∈ S, ((L.runExtAt x s needs g.val r (run' r).seen c).eval H run').2.2 =
      (((List.finRange (L.nPU x)).zip (L.checked x r)).map fun iu : Fin (L.nPU x) × ℕ =>
        (S.filter fun g : Fin (L.nG x) =>
            decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u)).card *
          ((if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
              L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)
            else .ret []).eval H run').2.2).sum := by
    refine Eq.trans (Finset.sum_congr rfl fun g _ => ?_) hmid
    unfold runExtAt
    exact flatMapRP_runs H run' _ _
  rw [hsum, Nat.cast_list_sum, List.map_map]
  refine le_of_eq_of_le (Eq.trans ?_ (sum_zip_finRange (L.nPU x) (fun iu : Fin (L.nPU x) × ℕ =>
    (((S.filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g.val ∈ needs u)).card : ℕ) : ℝ) *
      if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
        (((L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run' r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval H
          run').2.2 : ℝ)
      else 0) (L.checked x r))) (Finset.sum_le_sum fun i _ => ?_)
  · refine congrArg List.sum (List.map_congr_left fun iu _ => ?_)
    simp only [Function.comp_apply, Nat.cast_mul]
    congr 1
    split
    · rfl
    · exact Nat.cast_zero
  · unfold slotRuns
    cases hl : (L.checked x r)[i.val]? with
    | none => simp
    | some u =>
      simp only [Option.map_some, Option.getD_some]
      refine mul_le_mul_of_nonneg_right le_rfl ?_
      split
      · exact Nat.cast_nonneg _
      · exact le_rfl

/-! ## A session's expected reruns, weighted by the sample -/

variable {H : Msg → Msg} {L : Compiled Coin A}

/-- **Slot `i`'s expected reruns, weighted by the sample**: `slot_exp_le` with a weight `w` of the sample, on both
sides, in place of acceptance. Resampling the slot's session's rounds keeps the sample (`RunsFixed.fixed`), so the
weight is the same at every resampled run and stays outside the session's expectation, which is at most `e_s`. -/
theorem slot_exp_le_weighted (hS : L.CostContract H) {x : Msg} (s : AuditStatement) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (i : Fin (L.nPU x)) (w : List ℕ → ℝ) (hw : ∀ l, 0 ≤ w l) (r : Fin (L.V.k x) → Coin) :
    Ex (fun d : Fin (L.V.k x) → Coin => w (L.checked x (forkOn (L.stage2 x) r d)) *
        Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ) ≤
      Ex fun d : Fin (L.V.k x) → Coin => w (L.checked x (forkOn (L.stage2 x) r d)) *
        if (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val then L.es x else 0 := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hres := Ex_resample (fun d => L.selRounds x (L.firstSel x i) (forkOn (L.stage2 x) r d))
    (selRounds_resample hS.toRunsFixed (L.firstSel x i) r) fun d =>
      w (L.checked x (forkOn (L.stage2 x) r d)) *
        Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.stage2 x) r d) ρ
  beta_reduce at hres
  rw [← hres]
  refine Ex_mono fun d => ?_
  cases h : L.selSlot x (L.firstSel x i) (forkOn (L.stage2 x) r d) with
  | none =>
    rw [selRounds_none h]
    simp only [forkOn_empty, slotRuns_none H L x s (run H P L.V x ω) i h, Ex_const, mul_zero]
    refine mul_nonneg (hw _) ?_
    split_ifs
    · exact he0
    · exact le_rfl
  | some iu =>
    obtain ⟨i', u⟩ := iu
    obtain ⟨hsel, hm⟩ := selSlot_eq_some.1 h
    have hfi : (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val ∧ i' = i := by
      unfold firstSel at hsel
      by_cases hf : (L.sessSlots x (L.checked x (forkOn (L.stage2 x) r d)) i).head? = some i.val
      · rw [ite_eq_left hf] at hsel
        exact ⟨hf, (Option.some.inj hsel).symm⟩
      · rw [ite_eq_right hf] at hsel
        cases hsel
    rw [hfi.2] at h hm
    rw [ite_eq_left hfi.1]
    have hmem : u.val ∈ L.checked x (forkOn (L.stage2 x) r d) := List.mem_of_getElem? hm
    rw [selRounds_some h]
    simp only [forkOn_sub (hS.fixed x _ u hmem).1]
    simp only [(hS.fixed x _ u hmem).2]
    rw [Ex_mul_left]
    refine mul_le_mul_of_nonneg_left ?_ (hw _)
    obtain ⟨xS, P', hshape, -, -, hcard, -, hview⟩ := hS.runs x Ω P ω (forkOn (L.stage2 x) r d) u hmem
    have hag : ∀ σ : Fin (L.V.k x) → Coin, ∀ t ∉ L.rounds x u,
        forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ t = forkOn (L.stage2 x) r d t :=
      fun σ t ht => by simp [forkOn, ht]
    have hunit : ∀ σ (ρ : L.BlockEC x),
        L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ) ρ =
          ((A.E.eval H P' xS () (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) (ρ u))).2.2 :
            ℝ) := by
      intro σ ρ
      have hcσ := (hS.fixed x _ u hmem).2 σ
      have hmσ : (L.checked x (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ))[i.val]? = some u.val := by
        rw [hcσ]
        exact hm
      have hfσ : (L.sessSlots x (L.checked x (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ)) i).head? =
          some i.val := by
        rw [hcσ]
        exact hfi.1
      rw [slotRuns_some H L x s _ hmσ hfσ ρ, L.sessExt_runs_eq H P ω P' (forkOn (L.stage2 x) r d) hshape hcard
        (fun r' h => (hview r' h).1) (hag σ) s _ (ρ u), L.coinsOf_forkOn hcard]
    calc Ex (fun σ : Fin (L.V.k x) → Coin => Ex fun ρ : L.BlockEC x =>
            L.slotRuns H x s (run H P L.V x ω) i (forkOn (L.rounds x u) (forkOn (L.stage2 x) r d) σ) ρ)
        = Ex (fun σ : Fin (L.V.k x) → Coin => Ex fun c : A.EC (L.sh x u) =>
            ((A.E.eval H P' xS () (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) c)).2.2 :
              ℝ)) := by
          refine congrArg Ex (funext fun σ => ?_)
          simp only [hunit σ]
          exact Ex_pi_eval_dep (E := fun v : Fin (L.nPU x) => A.EC (L.sh x v)) u fun c : A.EC (L.sh x u) =>
            ((A.E.eval H P' xS () (fun t => σ (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) c)).2.2 : ℝ)
      _ = Ex (fun p : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) => ((A.E.eval H P' xS ()
            (fun t => p.1 (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) p.2)).2.2 : ℝ)) :=
          (Ex_prod fun p : (Fin (L.V.k x) → Coin) × A.EC (L.sh x u) => ((A.E.eval H P' xS ()
            (fun t => p.1 (L.emb x u xS hcard t)) (cast (congrArg A.EC hshape.symm) p.2)).2.2 : ℝ)).symm
      _ = Ex (fun p : (Fin (A.V.k xS) → Coin) × A.EC (A.shape xS) => ((A.E.eval H P' xS () p.1 p.2).2.2 : ℝ)) :=
          Ex_restrict_shape (L.emb x u xS hcard) (L.emb x u xS hcard).injective hshape
            fun ρ' c' => ((A.E.eval H P' xS () ρ' c').2.2 : ℝ)
      _ ≤ 1 + A.waits (A.shape xS) * A.k (A.shape xS) := A.rerunsExp_le H xS Unit P' ()
      _ = 1 + A.waits (L.sh x u) * A.k (L.sh x u) := by rw [hshape]
      _ ≤ L.es x := le_maxBelow (fun v => 1 + (A.waits (L.sh x v) : ℝ) * A.k (L.sh x v)) u.isLt

/-- **Expected reruns of a run's extractions, each slot weighted by the sample**: over the run's coins and the
extractor's, at most `e_s` times the expected weight of the sample's first slots (`slot_exp_le_weighted`, the second
stage's coins fresh at uniform coins, `Ex_forkOn`). At the weight `1` it bounds a run's expected reruns by `e_s`
times the expected number of its sample's sessions. -/
theorem ex_weighted_runs_le (hS : L.CostContract H) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) (s : AuditStatement)
    (w : Fin (L.nPU x) → List ℕ → ℝ) (hw : ∀ i l, 0 ≤ w i l) :
    Ex (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        ∑ i, w i (L.checked x p.1) * L.slotRuns H x s (run H P L.V x ω) i p.1 (p.2 i)) ≤
      L.es x * Ex (fun r : Fin (L.V.k x) → Coin =>
        ∑ i ∈ Finset.univ.filter (fun i : Fin (L.nPU x) => (L.sessSlots x (L.checked x r) i.val).head? = some i.val),
          w i (L.checked x r)) := by
  have hslot : ∀ i : Fin (L.nPU x), Ex (fun r : Fin (L.V.k x) → Coin =>
      w i (L.checked x r) * Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ) ≤
        Ex fun r : Fin (L.V.k x) → Coin =>
          w i (L.checked x r) * if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0 := by
    intro i
    rw [← Ex_forkOn (L.stage2 x) (fun r => w i (L.checked x r) *
        Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ),
      ← Ex_forkOn (L.stage2 x) (fun r => w i (L.checked x r) *
        if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0), Ex_prod, Ex_prod]
    exact Ex_mono fun r => slot_exp_le_weighted hS s P ω i (w i) (hw i) r
  calc Ex (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          ∑ i, w i (L.checked x p.1) * L.slotRuns H x s (run H P L.V x ω) i p.1 (p.2 i))
      = Ex (fun r : Fin (L.V.k x) → Coin =>
          ∑ i, w i (L.checked x r) * Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ) := by
        rw [Ex_prod]
        refine congrArg Ex (funext fun r => ?_)
        show Ex (fun c : L.SlotEC x =>
          ∑ i, w i (L.checked x r) * L.slotRuns H x s (run H P L.V x ω) i r (c i)) = _
        rw [Ex_finsum]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [Ex_mul_left]
        exact congrArg (fun z => w i (L.checked x r) * z)
          (Ex_pi_eval i fun ρ => L.slotRuns H x s (run H P L.V x ω) i r ρ)
    _ = ∑ i, Ex (fun r : Fin (L.V.k x) → Coin =>
          w i (L.checked x r) * Ex fun ρ : L.BlockEC x => L.slotRuns H x s (run H P L.V x ω) i r ρ) := by
        rw [Ex_finsum]
    _ ≤ ∑ i : Fin (L.nPU x), Ex (fun r : Fin (L.V.k x) → Coin =>
          w i (L.checked x r) * if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0) :=
        Finset.sum_le_sum fun i _ => hslot i
    _ = Ex (fun r : Fin (L.V.k x) → Coin => ∑ i : Fin (L.nPU x),
          w i (L.checked x r) * if (L.sessSlots x (L.checked x r) i).head? = some i.val then L.es x else 0) := by
        rw [Ex_finsum]
    _ = Ex (fun r : Fin (L.V.k x) → Coin => L.es x *
          ∑ i ∈ Finset.univ.filter (fun i : Fin (L.nPU x) =>
            (L.sessSlots x (L.checked x r) i.val).head? = some i.val), w i (L.checked x r)) := by
        refine congrArg Ex (funext fun r => ?_)
        rw [Finset.mul_sum, Finset.sum_filter]
        refine Finset.sum_congr rfl fun i _ => ?_
        split_ifs
        · exact mul_comm _ _
        · exact mul_zero _
    _ = _ := Ex_mul_left _ _

/-! ## The bound's expectation in closed form -/

/-- **A sample has at most `n_sess` sessions in expectation**, as on every coin (`card_first_le`): `n_sess` is one
`N_s` the bound may read. -/
theorem ex_firstSlots_le (hS : L.CostContract H) (x : Msg) :
    Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ L.nSess x :=
  (Ex_mono fun r => Nat.cast_le.2 (card_first_le hS.keys x r)).trans_eq (Ex_const _)

/-- **Expected reruns of a run's extractions**: over the tape, the run's coins and the extractor's, at most `e_s`
times the expected number of its sample's sessions (`ex_weighted_runs_le` at the weight `1`). -/
theorem ex_runExt_runs_le (hS : L.CostContract H) {x : Msg} (s : AuditStatement) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) {Ns : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((L.runExt x s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2).eval H (run H P L.V x q.1)).2.2 : ℝ)) ≤
      L.es x * Ns := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  refine (Verity.FirstHit.ex_tape fun (ω : Ω) (p : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
    (((L.runExt x s p.1 (run H P L.V x ω p.1).seen p.2).eval H (run H P L.V x ω)).2.2 : ℝ)).trans_le ?_
  refine (Ex_mono fun ω => ?_).trans_eq (Ex_const (L.es x * Ns))
  refine le_trans (le_of_eq_of_le ?_ (ex_weighted_runs_le hS P ω s (fun _ _ => (1 : ℝ))
    fun _ _ => zero_le_one)) (mul_le_mul_of_nonneg_left (le_of_eq_of_le ?_ hNs) he0)
  · refine congrArg Ex (funext fun p => ?_)
    rw [runExt_runs_eq H L x s (run H P L.V x ω) p.1 p.2]
    exact Finset.sum_congr rfl fun i _ => (one_mul _).symm
  · refine congrArg Ex (funext fun r => ?_)
    unfold firstSlots
    rw [Finset.sum_const, nsmul_eq_mul, mul_one]

/-- **A run's expected calls at a budget `T`**, with every session of its sample extracted: at most `runCost`,
`T·(1 + e_s·N_s) + n_PU·openCalls` (`callsAt_le_at`, `runExt_own_le`, `ex_runExt_runs_le`). -/
theorem ex_callsAt_le_at (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ)) ≤
      L.runCost x T Ns := by
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg _
  have hpt : ∀ q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x), (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ) ≤
      ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) + (T : ℝ) *
        (((L.runExt x s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2).eval H (run H P L.V x q.1)).2.2 : ℝ) := by
    rintro ⟨ω, r, c⟩
    have h1 : (L.callsAt H P x s ω r c : ℝ) ≤ (T : ℝ) +
        ((ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length : ℝ) +
        (T : ℝ) * (((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℝ) := by
      exact_mod_cast L.callsAt_le_at H P x s T ω (hT ω) r c
    have hownr : ((ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length : ℝ) ≤
        (L.nPU x : ℝ) * L.openCalls x := by exact_mod_cast runExt_own_le hS hopen hs P ω r c
    show (L.callsAt H P x s ω r c : ℝ) ≤ ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) + (T : ℝ) *
      (((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℝ)
    linarith
  refine (Ex_mono hpt).trans ?_
  rw [Ex_add, Ex_const, Ex_mul_left]
  refine (add_le_add (le_refl ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x))
    (mul_le_mul_of_nonneg_left (ex_runExt_runs_le hS s P hNs) hT0)).trans (le_of_eq ?_)
  unfold runCost
  ring

/-- **`B_conf,X`'s expected cost at a budget `T` a run**: its two runs' calls, each run's at most `runCost` in
expectation (`ex_callsAt_le_at`), so at most `2·(T·(1 + e_s·N_s) + n_PU·openCalls)`. -/
theorem ex_bconfX_cost_le (hS : L.CostContract H) (hopen : L.OpenCallsLe H) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns) :
    Ex (fun q => ((BconfX H L Ω P x).cost q : ℝ)) ≤ 2 * L.runCost x T Ns := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hNs0 : 0 ≤ Ns := le_trans (Ex_nonneg fun _ => Nat.cast_nonneg _) hNs
  have hrc0 : 0 ≤ L.runCost x T Ns := by
    unfold runCost
    exact add_nonneg (mul_nonneg (Nat.cast_nonneg _) (add_nonneg zero_le_one (mul_nonneg he0 hNs0)))
      (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  delta BconfX
  rcases hs : L.parse x with _ | s
  · show Ex (fun _ : Unit => ((0 : ℕ) : ℝ)) ≤ _
    rw [Nat.cast_zero, Ex_const]
    exact mul_nonneg zero_le_two hrc0
  · refine (Ex_mono (g := fun q : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
          (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        (L.callsAt H P x s q.1.1 q.1.2.1 q.1.2.2 : ℝ) + (L.callsAt H P x s q.2.1 q.2.2.1 q.2.2.2 : ℝ))
      fun q => le_of_eq ?_).trans ?_
    · simp only [callsAt]
      push_cast
      ring
    · exact (Ex_two fun q' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (L.callsAt H P x s q'.1 q'.2.1 q'.2.2 : ℝ)).trans_le
          (mul_le_mul_of_nonneg_left (ex_callsAt_le_at hS hopen hs P T hT hNs) zero_le_two)

/-- **`B_conf,X` at a budget `T` a run**: under the linear expected-time bound at `B_conf,X`, it finds a collision
with probability at most `2·(T·(1 + e_s·N_s) + n_PU·openCalls)/2^256` (`ex_bconfX_cost_le`). -/
theorem bconfX_le_at (hS : L.CostContract H) (hopen : L.OpenCallsLe H) (x : Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hB : ExpectedCR H (BconfX H L Ω P x)) :
    (BconfX H L Ω P x).Pr H ≤ 2 * L.runCost x T Ns / 2 ^ (256 : ℝ) := by
  unfold ExpectedCR at hB
  exact hB.trans (div_le_div_of_nonneg_right (ex_bconfX_cost_le hS hopen x P T hT hNs) (by positivity))

/-- **The first-hit bound's expected calls, at a budget `T` a run, in closed form.** Where each run calls `H` at most
`T` times, both parties', the expectation `eventTwoX_le_firstHit_at` reads, of a run's calls with every session
extracted and, for each gate its sample needs, its calls with only the sessions needing the gate extracted, is at
most `firstHitCost`: `T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls`, at any `N_s` at least the
expected number of sessions in a run's sample, any `K_s` at least the expected number of gates the sample needs and
any `K_m` at least the expected number its sessions need, session by session. -/
theorem ex_calls_le_at (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.sampleNeeds x needs (L.checked x r)).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin => (L.sessCount x needs (L.checked x r) : ℝ)) ≤ Km) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ g ∈ Finset.univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤ L.firstHitCost x T Ns Ks Km := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg _
  have hN0 : (0 : ℝ) ≤ (L.nPU x : ℝ) * L.openCalls x := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hpt : ∀ q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x),
      ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
        ∑ g ∈ Finset.univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
          L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ) ≤
        ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
          (T : ℝ) *
            (((L.runExt x s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2).eval H (run H P L.V x q.1)).2.2 : ℝ) +
          ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) * ((L.sampleNeeds x needs (L.checked x q.2.1)).card : ℝ) +
          (T : ℝ) * ∑ i : Fin (L.nPU x), ((L.sessNeeds x needs (L.checked x q.2.1) i.val).card : ℝ) *
            L.slotRuns H x s (run H P L.V x q.1) i q.2.1 (q.2.2 i) := by
    rintro ⟨ω, r, c⟩
    have h1 : (L.callsAt H P x s ω r c : ℝ) ≤ (T : ℝ) +
        ((ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length : ℝ) +
        (T : ℝ) * (((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℝ) := by
      exact_mod_cast L.callsAt_le_at H P x s T ω (hT ω) r c
    have hown := runExt_own_le hS hopen hs P ω r c
    have hownr : ((ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length : ℝ) ≤
        (L.nPU x : ℝ) * L.openCalls x := by exact_mod_cast hown
    have h4 : ∑ g ∈ L.sampleNeeds x needs (L.checked x r), L.callsBy H P x s needs g.val ω r c ≤
        (L.sampleNeeds x needs (L.checked x r)).card * (T + L.nPU x * L.openCalls x) +
          T * ∑ g ∈ L.sampleNeeds x needs (L.checked x r),
            ((L.runExtAt x s needs g.val r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 := by
      refine (Finset.sum_le_sum fun g _ => (L.callsBy_le_at H P x s needs T ω (hT ω) g.val r c).trans
        (Nat.add_le_add_right (Nat.add_le_add_left hown T) _)).trans_eq ?_
      rw [Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul, Finset.mul_sum]
    have h4r : ((∑ g ∈ L.sampleNeeds x needs (L.checked x r), L.callsBy H P x s needs g.val ω r c : ℕ) : ℝ) ≤
        ((L.sampleNeeds x needs (L.checked x r)).card : ℝ) * ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
          (T : ℝ) * ((∑ g ∈ L.sampleNeeds x needs (L.checked x r),
            ((L.runExtAt x s needs g.val r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℕ) : ℝ) := by
      exact_mod_cast h4
    have h3 := mul_le_mul_of_nonneg_left (L.sum_runExtAt_runs_le H x s (run H P L.V x ω) needs
      (L.sampleNeeds x needs (L.checked x r)) r c) hT0
    show ((L.callsAt H P x s ω r c +
        ∑ g ∈ L.sampleNeeds x needs (L.checked x r), L.callsBy H P x s needs g.val ω r c : ℕ) : ℝ) ≤
      ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
        (T : ℝ) * (((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℝ) +
        ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) * ((L.sampleNeeds x needs (L.checked x r)).card : ℝ) +
        (T : ℝ) * ∑ i : Fin (L.nPU x), ((L.sessNeeds x needs (L.checked x r) i.val).card : ℝ) *
          L.slotRuns H x s (run H P L.V x ω) i r (c i)
    rw [Nat.cast_add]
    linarith
  have hB := ex_runExt_runs_le hS s P hNs
  have hC : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ((L.sampleNeeds x needs (L.checked x q.2.1)).card : ℝ)) ≤ Ks :=
    (Ex_coins fun r : Fin (L.V.k x) → Coin => ((L.sampleNeeds x needs (L.checked x r)).card : ℝ)).trans_le hKs
  have hD : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ∑ i : Fin (L.nPU x), ((L.sessNeeds x needs (L.checked x q.2.1) i.val).card : ℝ) *
        L.slotRuns H x s (run H P L.V x q.1) i q.2.1 (q.2.2 i)) ≤ L.es x * Km := by
    refine (Verity.FirstHit.ex_tape fun (ω : Ω) (p : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ∑ i : Fin (L.nPU x), ((L.sessNeeds x needs (L.checked x p.1) i.val).card : ℝ) *
        L.slotRuns H x s (run H P L.V x ω) i p.1 (p.2 i)).trans_le ?_
    refine (Ex_mono fun ω => ex_weighted_runs_le hS P ω s
      (fun i l => ((L.sessNeeds x needs l i.val).card : ℝ)) fun _ _ => Nat.cast_nonneg _).trans ?_
    rw [Ex_const]
    refine mul_le_mul_of_nonneg_left (le_of_eq_of_le ?_ hKm) he0
    refine congrArg Ex (funext fun r => ?_)
    unfold sessCount firstSlots
    rw [Nat.cast_sum]
  refine (Ex_mono hpt).trans ?_
  rw [Ex_add, Ex_add, Ex_add, Ex_const, Ex_mul_left, Ex_mul_left, Ex_mul_left]
  refine (add_le_add (add_le_add (add_le_add (le_refl ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x))
    (mul_le_mul_of_nonneg_left hB hT0)) (mul_le_mul_of_nonneg_left hC (add_nonneg hT0 hN0)))
    (mul_le_mul_of_nonneg_left hD hT0)).trans (le_of_eq ?_)
  unfold firstHitCost
  ring

/-- **The first-hit bound's expected calls at any set of gates of the sample, at a budget `T` a run, in closed
form.** `ex_calls_le_at` with the gates a finder pays for at a further run a set `G` of the run's sample, where that
lemma has every gate the sample needs: the expectation of a run's calls with every session extracted and, for each
gate of `G`, its calls with only the sessions needing the gate extracted, is at most `firstHitCost` at any `N_s` at
least the expected number of sessions in a run's sample, any `K_s` at least the expected number of gates in `G` and
any `K_m` at least the expected number of them its sessions need, session by session (`sessCountIn`). -/
theorem ex_calls_le_at_in (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (G : List ℕ → Finset (Fin (L.nG x))) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin => ((G (L.checked x r)).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin => (L.sessCountIn x needs G (L.checked x r) : ℝ)) ≤ Km) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ g ∈ G (L.checked x q.2.1),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤ L.firstHitCost x T Ns Ks Km := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg _
  have hN0 : (0 : ℝ) ≤ (L.nPU x : ℝ) * L.openCalls x := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hpt : ∀ q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x),
      ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
        ∑ g ∈ G (L.checked x q.2.1),
          L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ) ≤
        ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
          (T : ℝ) *
            (((L.runExt x s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2).eval H (run H P L.V x q.1)).2.2 : ℝ) +
          ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) * ((G (L.checked x q.2.1)).card : ℝ) +
          (T : ℝ) * ∑ i : Fin (L.nPU x), (((G (L.checked x q.2.1)).filter fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.sessBlocks x (L.checked x q.2.1) i.val, g.val ∈ needs u)).card : ℝ) *
            L.slotRuns H x s (run H P L.V x q.1) i q.2.1 (q.2.2 i) := by
    rintro ⟨ω, r, c⟩
    have h1 : (L.callsAt H P x s ω r c : ℝ) ≤ (T : ℝ) +
        ((ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length : ℝ) +
        (T : ℝ) * (((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℝ) := by
      exact_mod_cast L.callsAt_le_at H P x s T ω (hT ω) r c
    have hown := runExt_own_le hS hopen hs P ω r c
    have hownr : ((ownLog H (run H P L.V x ω) (L.runExt x s r (run H P L.V x ω r).seen c)).length : ℝ) ≤
        (L.nPU x : ℝ) * L.openCalls x := by exact_mod_cast hown
    have h4 : ∑ g ∈ G (L.checked x r), L.callsBy H P x s needs g.val ω r c ≤
        (G (L.checked x r)).card * (T + L.nPU x * L.openCalls x) +
          T * ∑ g ∈ G (L.checked x r),
            ((L.runExtAt x s needs g.val r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 := by
      refine (Finset.sum_le_sum fun g _ => (L.callsBy_le_at H P x s needs T ω (hT ω) g.val r c).trans
        (Nat.add_le_add_right (Nat.add_le_add_left hown T) _)).trans_eq ?_
      rw [Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul, Finset.mul_sum]
    have h4r : ((∑ g ∈ G (L.checked x r), L.callsBy H P x s needs g.val ω r c : ℕ) : ℝ) ≤
        ((G (L.checked x r)).card : ℝ) * ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
          (T : ℝ) * ((∑ g ∈ G (L.checked x r),
            ((L.runExtAt x s needs g.val r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℕ) : ℝ) := by
      exact_mod_cast h4
    have h3 := mul_le_mul_of_nonneg_left (L.sum_runExtAt_runs_le_in H x s (run H P L.V x ω) needs
      (G (L.checked x r)) r c) hT0
    show ((L.callsAt H P x s ω r c +
        ∑ g ∈ G (L.checked x r), L.callsBy H P x s needs g.val ω r c : ℕ) : ℝ) ≤
      ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) +
        (T : ℝ) * (((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.2 : ℝ) +
        ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x) * ((G (L.checked x r)).card : ℝ) +
        (T : ℝ) * ∑ i : Fin (L.nPU x), (((G (L.checked x r)).filter fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.sessBlocks x (L.checked x r) i.val, g.val ∈ needs u)).card : ℝ) *
          L.slotRuns H x s (run H P L.V x ω) i r (c i)
    rw [Nat.cast_add]
    linarith
  have hB := ex_runExt_runs_le hS s P hNs
  have hC : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ((G (L.checked x q.2.1)).card : ℝ)) ≤ Ks :=
    (Ex_coins fun r : Fin (L.V.k x) → Coin => ((G (L.checked x r)).card : ℝ)).trans_le hKs
  have hD : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ∑ i : Fin (L.nPU x), (((G (L.checked x q.2.1)).filter fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.sessBlocks x (L.checked x q.2.1) i.val, g.val ∈ needs u)).card : ℝ) *
        L.slotRuns H x s (run H P L.V x q.1) i q.2.1 (q.2.2 i)) ≤ L.es x * Km := by
    refine (Verity.FirstHit.ex_tape fun (ω : Ω) (p : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ∑ i : Fin (L.nPU x), (((G (L.checked x p.1)).filter fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.sessBlocks x (L.checked x p.1) i.val, g.val ∈ needs u)).card : ℝ) *
        L.slotRuns H x s (run H P L.V x ω) i p.1 (p.2 i)).trans_le ?_
    refine (Ex_mono fun ω => ex_weighted_runs_le hS P ω s
      (fun i l => (((G l).filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.sessBlocks x l i.val, g.val ∈ needs u)).card : ℝ)) fun _ _ => Nat.cast_nonneg _).trans ?_
    rw [Ex_const]
    refine mul_le_mul_of_nonneg_left (le_of_eq_of_le ?_ hKm) he0
    refine congrArg Ex (funext fun r => ?_)
    unfold sessCountIn firstSlots
    rw [Nat.cast_sum]
  refine (Ex_mono hpt).trans ?_
  rw [Ex_add, Ex_add, Ex_add, Ex_const, Ex_mul_left, Ex_mul_left, Ex_mul_left]
  refine (add_le_add (add_le_add (add_le_add (le_refl ((T : ℝ) + (L.nPU x : ℝ) * L.openCalls x))
    (mul_le_mul_of_nonneg_left hB hT0)) (mul_le_mul_of_nonneg_left hC (add_nonneg hT0 hN0)))
    (mul_le_mul_of_nonneg_left hD hT0)).trans (le_of_eq ?_)
  unfold firstHitCost
  ring

/-- **The first-hit bound's expected calls by class of gates, at a budget `T` a run, in closed form**:
`ex_calls_le_at_in` at the representatives of the classes of the gates a run's sample needs, the set a by-class
first-hit finder pays for at a further run. `K_s` is at least the expected number of classes of the gates the sample
needs, and `K_m` at least the expected number of their representatives its sessions need, session by session. -/
theorem ex_calls_le_at_cls (hS : L.CostContract H) (hopen : L.OpenCallsLe H) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.sessCountIn x needs (fun l => (L.sampleNeeds x needs l).image cls) (L.checked x r) : ℝ)) ≤ Km) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ c ∈ (Finset.univ.filter (fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
            L.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤ L.firstHitCost x T Ns Ks Km :=
  ex_calls_le_at_in hS hopen hs P needs (fun l => (L.sampleNeeds x needs l).image cls) T hT hNs hKs hKm

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The event of `KSRegTwoX` at a budget `T` a run, with one closed right-hand side, at one cap on each finder's
waits.** On a statement in scope, where a reveal for proof block `u` opens only gates of `needs u` and each run calls
`H` at most `T` times, both parties', an accepted run samples a proof block of `FregTwoX` with probability at most

`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls)/2^256`
`+ N/(KX + 1) + N/(KI + 1)`,

under the linear expected-time bound at the two by-position finders of the caps `KX` and `KI`: `N_s` at least the
expected number of sessions in a run's sample, `K_s` at least the expected number of gates the sample needs, `K_m`
at least the expected number its sessions need, session by session, `N` the program's gates
(`eventTwoX_le_firstHit_at_cap`, `Compiled.ex_calls_le_at`). -/
theorem eventTwoX_le_at_cap {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      ((L.toCompiled.sampleNeeds x needs (L.checked x r)).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin => (L.toCompiled.sessCount x needs (L.checked x r) : ℝ)) ≤ Km)
    (KX KI : ℕ) (hX : ExpectedCR H (L.toCompiled.finderXAt H P x s needs KX))
    (hI : ExpectedCR H (L.finderIAt H P x s d.part needs KI)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * L.toCompiled.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) +
        ((L.nG x : ℝ) / ((KX : ℝ) + 1) + (L.nG x : ℝ) / ((KI : ℝ) + 1)) := by
  have h := eventTwoX_le_firstHit_at_cap hA hbind hS hs hPt hd hin P needs hpos KX KI hX hI
  have hc := Compiled.ex_calls_le_at hS.toCostContract hopen hs P needs T hT hNs hKs hKm
  have hdiv := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 4))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ))
  exact h.trans (add_le_add (add_le_add (le_refl _) hdiv) (le_refl _))

/-- **The event of `KSRegTwoX` at a budget `T` a run, with one closed right-hand side.** On a statement in scope,
where a reveal for proof block `u` opens only gates of `needs u` and each run calls `H` at most `T` times, both
parties', an accepted run samples a proof block of `FregTwoX` with probability at most

`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·n_PU·openCalls)/2^256`,

under the linear expected-time bound at the two by-position finders of every cap: `N_s` at least the expected number
of sessions in a run's sample, `K_s` at least the expected number of gates the sample needs, `K_m` at least the
expected number its sessions need, session by session (`eventTwoX_le_firstHit_at`, `Compiled.ex_calls_le_at`). -/
theorem eventTwoX_le_at {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      ((L.toCompiled.sampleNeeds x needs (L.checked x r)).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin => (L.toCompiled.sessCount x needs (L.checked x r) : ℝ)) ≤ Km)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXAt H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderIAt H P x s d.part needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * L.toCompiled.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) := by
  have h := eventTwoX_le_firstHit_at hA hbind hS hs hPt hd hin P needs hpos hX hI
  have hc := Compiled.ex_calls_le_at hS.toCostContract hopen hs P needs T hT hNs hKs hKm
  have hdiv := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 4))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ))
  exact h.trans (add_le_add (le_refl _) hdiv)

/-- **The event of `KSRegTwoX` at a budget `T` a run, `B_conf,X` at the budget too.** `eventTwoX_le_at` with
`Pr[B_conf,X]` read at the budget (`Compiled.bconfX_le_at`), under the linear expected-time bound at `B_conf,X` and
at the two by-position finders of every cap: an accepted run samples a proof block of `FregTwoX` with probability at
most

`κ_s + √(k·γ_s·Pr[B]) + (2·(T·(1 + e_s·N_s) + n_PU·openCalls) + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m)`
`+ (1 + K_s)·n_PU·openCalls))/2^256`. -/
theorem eventTwoX_le_at_budget {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      ((L.toCompiled.sampleNeeds x needs (L.checked x r)).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin => (L.toCompiled.sessCount x needs (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXAt H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderIAt H P x s d.part needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * L.toCompiled.runCost x T Ns + 4 * L.toCompiled.firstHitCost x T Ns Ks Km) / 2 ^ (256 : ℝ) := by
  have h := eventTwoX_le_at hA hbind hS hopen hs hPt hd hin P needs hpos T hT hNs hKs hKm hX hI
  have hb := Compiled.bconfX_le_at hS.toCostContract hopen x P T hT hNs hB
  rw [add_div]
  exact h.trans ((add_le_add (add_le_add (le_refl _) hb) (le_refl _)).trans (le_of_eq (add_assoc _ _ _)))

end CompiledTwo

end Claims.SamplingGeneric
