module

public import Definitions.Pouw.Guarantees.Deadline
public import Proofs.Pouw.Lemmas.Deadline
public import Proofs.Pouw.Game
public import Proofs.Pouw.Break
public import Proofs.Pouw.PearlC.DeviceInertBoundRev1Gap
public import Proofs.Pouw.PearlC.ServedMixedRev1Gap
public import Proofs.Pouw.PearlC.TTOutW1
public import Proofs.Core.Challenger
public import Definitions.Pouw.Witness.Honest
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.DeviceInertBoundRev1Gap
import Proofs.Pouw.Prob

/-!
# Proofs of the deadline's guarantees

The verifier's rule passes the clock's check (`clockDrift_sound`: an auditor's device of the key, in a host whose drift
`q` is below 1) and finds `0 ≤ ra.elapsed rb ≤ Δ` between the salt's entry and the docket's. Under
`clock-within-drift`, `ClockWithin.real_le` gives the real order of the two entries from the first inequality, and the
clock not slow puts the docket's entry within `Δ / (1 - q)` of the salt's (`InTime.deadline`). There the honest clock
names a program that outputs the adversary's trace, after the same preprocessing, at a cost within the run's spending
in the real window. That program is within the budget `EndToEnd` (or `GGapSampled`) reads, and its accepted transcripts
include every one the verifier accepts by the deadline.

The clock's witnesses (`Witness.Clock`): every Site's devices meet `clock-within-drift` with real time equal to the
reading (`clockWithin_real`), the witnesses' Site passes the check at drift `0` (`clockDrift_site`), so its real window
is `Δ` itself, and its entries are in time when their readings are `Δ` or less apart (`Witness.Clock.inTime`).

The sm_120 instances (`Proofs.Pouw.Lemmas.Deadline`) are stated at W1: each is `PearlCSampledByDeadline` at
`CM := W1.w1Priced dv salt`, with its sampled g from the old reduction at that cost model (`Proofs.Pouw.PearlC.TTOutW1`).

The honest clock is stated once over `Adversaries`; at each game's `CostModel.adversaries` it unfolds to that game's own
reading (`CostModel.honestClock_iff`, by `Iff.rfl`; `PearlC.CostModel.honestClock_iff`, whose adversaries hash nothing).
-/

@[expose] public section

open scoped NNRat

namespace Pouw

variable {Q R S : Type}

/-- The honest clock at the integer game's adversaries of `CM` on `U`, unfolded. -/
theorem CostModel.honestClock_iff {CM : CostModel Q R S} {U : Workload} {A : Adversary CM} {k : String}
    {dk : Docket S Trace k} {st : Stamps Q R S k} {run : TimedRun Q R S} {d : ℝ} :
    Assumptions.HonestClock (CM.adversaries U) A dk st run d ↔
      ∃ A' : Adversary CM, A'.pre = A.pre ∧ ∀ (H : Q → R) (s : S),
        (dk.digest (A.trace U H s) ∈ (st.docket H s).shown → st.tr H s ≤ st.ts s + d →
          A'.trace U H s = A.trace U H s ∧ ∀ m ∈ A'.hashed U H s, m ∈ A.hashed U H s) ∧
        CM.cost A'.online H U A'.act (A'.pre.eval H) s ≤ run.spentIn H s (st.ts s) (st.ts s + d) ∧
        CM.calls A'.online H U A'.act (A'.pre.eval H) s ≤ CM.calls A.online H U A.act (A.pre.eval H) s :=
  Iff.rfl

/-- The honest clock at Pearl-C's adversaries of `CM` on `U`, unfolded. -/
theorem PearlC.CostModel.honestClock_iff {CM : PearlC.CostModel Q R S} {U : PearlC.Workload} {A : PearlC.Adversary CM}
    {k : String} {dk : Docket S PearlC.Trace k} {st : Stamps Q R S k} {run : TimedRun Q R S} {d : ℝ} :
    Assumptions.HonestClock (CM.adversaries U) A dk st run d ↔
      ∃ A' : PearlC.Adversary CM, A'.pre = A.pre ∧ ∀ (H : Q → R) (s : S),
        (dk.digest (A.trace U H s) ∈ (st.docket H s).shown → st.tr H s ≤ st.ts s + d →
          A'.trace U H s = A.trace U H s) ∧
        CM.cost A'.online H U A'.act (A'.pre.eval H) s ≤ run.spentIn H s (st.ts s) (st.ts s + d) ∧
        CM.calls A'.online H U A'.act (A'.pre.eval H) s ≤ CM.calls A.online H U A.act (A.pre.eval H) s :=
  ⟨fun ⟨A', hpre, h⟩ => ⟨A', hpre, fun H s => ⟨fun hd ht => ((h H s).1 hd ht).1, (h H s).2⟩⟩,
    fun ⟨A', hpre, h⟩ => ⟨A', hpre, fun H s => ⟨fun hd ht => ⟨(h H s).1 hd ht, fun _ hm => hm⟩, (h H s).2⟩⟩⟩

open Verity.Challenger Verity.Assumptions in
/-- **The clock's check, sound**: where it returns the drift `q`, the Site has an auditor's device registered under
`k`, in a host that registers `q`, and `q < 1`. -/
theorem clockDrift_sound {site : Verity.Site} {k : String} {q : ℚ≥0} (h : clockDrift site k = .ok q) :
    ∃ x ∈ site.locs, x.kind.auditor = true ∧ x.key = k ∧ ∃ p ∈ x.parent, ∃ l ∈ site.loc? p, q ∈ l.drift ∧ q < 1 := by
  unfold clockDrift at h
  split at h
  · cases h
  · rename_i x hx
    have hmem := List.mem_of_find?_eq_some hx
    have hp := List.find?_some hx
    simp only [Bool.and_eq_true, beq_iff_eq] at hp
    split at h
    · cases h
    · rename_i q' hq'
      split_ifs at h with hlt
      cases h
      obtain ⟨l, hl, hd⟩ := Option.bind_eq_some_iff.1 hq'
      obtain ⟨p, hpp, hpl⟩ := Option.bind_eq_some_iff.1 hl
      exact ⟨x, hmem, hp.1, hp.2, p, hpp, l, hpl, hd, hlt⟩

open Verity.Challenger Verity.Assumptions in
/-- **A docket in time is signed within the real window**: under `clock-within-drift`, where the Site passes the
clock's check and the docket is in time at `Δ`, the docket's entry `b` signs the trace's digest and its real time is at
most `realWindow site k Δ` after the salt's entry `a`. Its real order comes from `0 ≤ ra.elapsed rb` by
`ClockWithin.real_le`, and its distance from the clock not slow. It says nothing of when any other entry was made. -/
theorem InTime.deadline {Out : Type} {site : Verity.Site} {k : String} {dk : Docket S Out k} {st : Stamps Q R S k}
    {Δ : ℕ} {H : Q → R} {s : S} {e : Out} (hc : ClockWithin site k st.real)
    (hsite : ∃ q, clockDrift site k = .ok q) (h : InTime dk st Δ H s e) :
    dk.digest e ∈ (st.docket H s).shown ∧ st.tr H s ≤ st.ts s + realWindow site k Δ := by
  obtain ⟨q, hq⟩ := hsite
  obtain ⟨x, hx, hau, hk, p, hp, l, hl, hd, hq1⟩ := clockDrift_sound hq
  obtain ⟨-, ra, rb, ⟨⟨-, hdig⟩, hra, hrb⟩, he0, heΔ⟩ := h
  refine ⟨hdig, ?_⟩
  have hle : st.real (st.salt s) ≤ st.real (st.docket H s) :=
    ClockWithin.real_le hc hx hau hk hp hl hd hq1 hra hrb he0
  have hs := (hc x hx hau hk p hp l hl q hd (st.salt s) (st.docket H s) ra hra rb hrb hle).1
  unfold NotSlow at hs
  have hq' : (q : ℝ) < 1 := by exact_mod_cast hq1
  have heΔ' : ((ra.elapsed rb : ℤ) : ℝ) ≤ (Δ : ℝ) := by exact_mod_cast heΔ
  have hw : realWindow site k Δ = Δ / (1 - (q : ℝ)) := by simp only [realWindow, hq]
  rw [hw, Stamps.tr, Stamps.ts, ← sub_le_iff_le_add', le_div_iff₀ (by linarith)]
  linarith

namespace Witness.Clock

open Verity.Challenger Verity.Assumptions

/-- Every Site's devices meet `clock-within-drift` when real time is the reading. -/
theorem clockWithin_real (site : Verity.Site) (k : String) : ClockWithin site k real := by
  intro x _ _ _ p _ l _ q _ a b ra hra rb hrb hle
  have ha : real a = (ra.ns : ℝ) := by
    have h' : a.reading = some ra := hra
    simp only [real, h']
  have hb : real b = (rb.ns : ℝ) := by
    have h' : b.reading = some rb := hrb
    simp only [real, h']
  have he : ((ra.elapsed rb : ℤ) : ℝ) = (rb.ns : ℝ) - ra.ns := by
    simp only [Reading.elapsed]
    push_cast
    ring
  have hq : (0 : ℝ) ≤ q := by positivity
  rw [ha, hb] at hle ⊢
  rw [he]
  have hd : (0 : ℝ) ≤ (rb.ns : ℝ) - ra.ns := by linarith
  exact ⟨by unfold NotSlow; nlinarith, by unfold NotFast; nlinarith⟩

/-- The witnesses' Site passes the clock's check at drift `0`. -/
theorem clockDrift_site : clockDrift site key = .ok 0 := by
  rfl

/-- The witnesses' real window is the deadline itself. -/
theorem realWindow_site (Δ : ℕ) : realWindow site key Δ = Δ := by
  simp [realWindow, clockDrift_site]

/-- The witnesses' entries, at readings `a ≤ b ≤ a + Δ`, are in time at `Δ` for every trace. -/
theorem inTime {Out : Type} (a b Δ : ℕ) (hab : a ≤ b) (hbΔ : b ≤ a + Δ) (H : Q → R) (s : S) (e : Out) :
    InTime (docket S Out) (stamps Q R S a b) Δ H s e :=
  ⟨List.mem_singleton_self _, ⟨a⟩, ⟨b⟩, ⟨⟨trivial, List.mem_singleton_self _⟩, rfl, rfl⟩,
    by unfold Reading.elapsed; omega, by unfold Reading.elapsed; omega⟩

theorem stamps_ts (a b : ℕ) (s : S) : (stamps Q R S a b).ts s = a := by
  simp [Stamps.ts, stamps, stamp, real]

theorem stamps_tr (a b : ℕ) (H : Q → R) (s : S) : (stamps Q R S a b).tr H s = b := by
  simp [Stamps.tr, stamps, stamp, real]

/-- The witnesses' Site passes the clock's check. -/
theorem site_ok : ∃ q, clockDrift site key = .ok q := ⟨0, clockDrift_site⟩

end Witness.Clock

end Pouw

namespace Pouw.SecurityProofs

theorem WorkBetweenSaltAndDeadline : Pouw.Guarantees.WorkBetweenSaltAndDeadline := by
  intro Q R S _ _ _ _ CM P D g₀ Ωmax ε hg₀ hΩ hW hTT U hU A site k dk st run Δ hcw hclock T hT q hq hB t εs hεs
  obtain ⟨A', hpre, hA'⟩ := CostModel.honestClock_iff.1 hclock
  have hb : A'.Bounded U T q := by
    intro H s
    refine ⟨(hA' H s).2.1.trans (hB H s).1, ?_⟩
    have h1 := (hA' H s).2.2
    have h2 := (hB H s).2
    rw [hpre] at h1 ⊢
    omega
  refine le_trans (pr_mono fun x hx => ?_) (EndToEnd Q R S CM P D g₀ Ωmax ε hg₀ hΩ hW hTT U hU T hT q hq A' hb t εs hεs)
  obtain ⟨⟨hsite, hin, hacc⟩, hlt, hbr⟩ := hx
  obtain ⟨hdig, htr⟩ := InTime.deadline hcw hsite hin
  obtain ⟨heq, hsub⟩ := (hA' x.1 x.2.1).1 hdig htr
  rw [heq]
  exact ⟨hacc, hlt, fun h => hbr (Pouw.Proofs.collisionIn_mono hsub h)⟩

theorem SpareCapacityAtDeadline : Pouw.Guarantees.SpareCapacityAtDeadline := by
  intro Q R S _ _ _ _ CM P D g₀ Ωmax ε hg₀ hΩ hW hTT U hU A site k dk st run Δ hcw hclock C Y hY q hq hB t εs hεs
  have hb : A.BoundedBy U st run (realWindow site k Δ) (C * realWindow site k Δ - Y) q := by
    intro H s
    exact ⟨by linarith [(hB H s).1], (hB H s).2⟩
  refine le_trans (pr_mono fun x hx => ⟨hx.1, by linarith [hx.2.1], hx.2.2⟩)
    (WorkBetweenSaltAndDeadline Q R S CM P D g₀ Ωmax ε hg₀ hΩ hW hTT U hU A site k dk st run Δ hcw hclock
      (C * realWindow site k Δ - Y) (by linarith) q hq hb t εs hεs)

theorem PearlCSampledByDeadline : Pouw.Guarantees.PearlCSampledByDeadline := by
  intro Q R S _ _ _ _ CM P TR D g δ hG U hU A site k dk st run Δ hcw hclock T hT q hq hB w κ hκ hw t εs hεs
  obtain ⟨A', hpre, hA'⟩ := PearlC.CostModel.honestClock_iff.1 hclock
  have hb : A'.Bounded U T q := by
    intro H s
    refine ⟨(hA' H s).2.1.trans (hB H s).1, ?_⟩
    have h1 := (hA' H s).2.2
    have h2 := (hB H s).2
    rw [hpre] at h1 ⊢
    omega
  refine le_trans (pr_mono fun x hx => ?_) (hG U hU T hT q hq A' hb w κ hκ hw t εs hεs)
  obtain ⟨⟨hsite, hin, hacc⟩, hlt⟩ := hx
  obtain ⟨hdig, htr⟩ := InTime.deadline hcw hsite hin
  rw [(hA' x.1 x.2.1).1 hdig htr]
  exact ⟨hacc, hlt⟩

theorem PearlCSm120Rev1InertShare1000ByDeadline_8192 : Pouw.Guarantees.PearlCSm120Rev1InertShare1000ByDeadline_8192 :=
  fun Q R S _ _ _ _ sha fetch salt sem hTT =>
    PearlCSampledByDeadline Q R S (PearlC.W1.w1Priced (PearlC.W1.W1Device.sm120Loop sha fetch) salt) _ _ _ _ _
      (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_8192 _ sem (PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT))

theorem PearlCSm120Rev1InertShare1000ByDeadline_16384 : Pouw.Guarantees.PearlCSm120Rev1InertShare1000ByDeadline_16384 :=
  fun Q R S _ _ _ _ sha fetch salt sem hTT =>
    PearlCSampledByDeadline Q R S (PearlC.W1.w1Priced (PearlC.W1.W1Device.sm120Loop sha fetch) salt) _ _ _ _ _
      (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_16384 _ sem (PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT))

theorem PearlCSm120Rev1InertShare1000ByDeadline_qwen3_8b : Pouw.Guarantees.PearlCSm120Rev1InertShare1000ByDeadline_qwen3_8b :=
  fun Q R S _ _ _ _ sha fetch salt sem hTT =>
    PearlCSampledByDeadline Q R S (PearlC.W1.w1Priced (PearlC.W1.W1Device.sm120Loop sha fetch) salt) _ _ _ _ _
      (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_qwen3_8b _ sem
        (PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT))

theorem PearlCSm120Rev1InertShare1000ByDeadline_llama31_8b : Pouw.Guarantees.PearlCSm120Rev1InertShare1000ByDeadline_llama31_8b :=
  fun Q R S _ _ _ _ sha fetch salt sem hTT =>
    PearlCSampledByDeadline Q R S (PearlC.W1.w1Priced (PearlC.W1.W1Device.sm120Loop sha fetch) salt) _ _ _ _ _
      (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_llama31_8b _ sem
        (PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT))

end Pouw.SecurityProofs
