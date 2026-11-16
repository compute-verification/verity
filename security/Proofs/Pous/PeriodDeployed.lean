module

public import Proofs.Pous.Lemmas.PeriodDeployed
public import Proofs.Pous.Period
public import Proofs.Pous.P2.RingTrunc

/-!
# Proofs of PoUS's per-period guarantee at the deployed ring budget, and of its honest and nontrivial lemmas

`periodHeldDeployed` is `periodHeld` at `(RING_D, RING_Q, RING_L)`. The budget sits outside both regions where
`P2RingAuditSecure` is refuted: `RING_D` is one round short of a root (`rootProg_not_ringBounded`), so recomputation
(two roots, `32,894` rounds) does not fit it, and the truncation family refutes `k = 105` only from `L = 2^169`
(`not_p2RingAuditSecure_trunc`, at `t = 168`, `m = 501,819`: `(501819/2^19)^105 ≈ 0.01006`), far above `RING_L = 2^64`.
At `t = 167` its `m = 501,788` gives `(501788/2^19)^105 ≈ 0.009996 < 1%`, so `2^169` is where that family starts.

The honest period (`periodHeldHonest`): the honest strategy is the lookup strategy (`periodHeld_honest`), which keeps
its state and runs a `ret` every round (`seqRounds_ret`), so each answer takes no round or query, and its state is
`C`'s `FULL` bits, which the capacity account holds at tick `0` of the window; the code's sample never fails at a
32-byte id (`coinView_some`, `uniformOn_split`), and a lookup returns the challenged block; `63,000,000 · 105 + 2^30 + 437,248,000 = 8,125,989,824 ≤
S₀ = 8,169,621,180`; `⌊854,000 · 349 / 36,155⌋ = 8,243 ≤ 16,446`; and `16,447 · 36,155 / 349 ≥ 1.995 · 854,000`.
The empty period has probability 1 of the accept-all event, and the honest one held `105 · FULL` bit-ticks
(`periodHeldNontrivial`).
-/

@[expose] public section

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties Pous.P2.Assumptions
open scoped ENNReal

set_option exponentiation.threshold 20000

theorem periodHeldDeployed : PeriodHeldDeployed := by
  intro hP
  exact periodHeld RING_D RING_Q RING_L hP

/-! ## The budget -/

theorem limit_ns : LIMIT_NS = 854000 := rfl

theorem deadline_rounds : DEADLINE_ROUNDS = 8243 := by
  unfold DEADLINE_ROUNDS
  rw [limit_ns, Nat.floor_eq_iff (by norm_num [ROUND_NS])]
  norm_num [ROUND_NS]


/-! ## The honest and the empty period -/

theorem S_eq : Pous.EndToEnd.S = 8169621180 := PousP2Ring.stateBound_p2ring_b19 _ _

theorem held_periodRun (m : ℕ) (hm : m ≤ FULL)
    (ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) :
    (periodRun m hm).held ω = K * m := by
  simp [ResourceAccounting.Run.held, ResourceAccounting.Schedule.held, ResourceAccounting.Window.ticks, periodRun]

theorem periodHeldHonest : PeriodHeldHonest := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro P hs
    have h0 : (0 : ℕ) ∈ (honestRun.win .capacity).ticks := by
      simp [honestRun, periodRun, ResourceAccounting.Window.ticks, K]
    exact periodHeld_honest honestRun P RING_D RING_Q RING_L 1 hs fun _ => ⟨0, h0, le_rfl⟩
  · show (0 : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ)
    norm_num [HONEST_S, K]
  · rw [S_eq]
    show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
      ((8169621180 : ℕ) : ℝ)
    norm_num [HONEST_S, K]
  · rw [deadline_rounds]
    show 8243 ≤ 16446
    omega
  · rw [limit_ns]
    norm_num [RING_D, ROUND_NS]


theorem periodHeldNontrivial : PeriodHeldNontrivial := by
  refine ⟨fun P o hb h => ?_, fun P o x ⟨_, _, hh⟩ => ?_⟩
  · have h1 : Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
        True ∧ ¬ P.Break (o.1, o.2, x) ∧
          (emptyRun.held (o.1, o.2, x) : ℝ) < HONEST_S * (emptyRun.win .capacity).len) = 1 := by
      refine Pouw.pr_eq_one fun x => ⟨trivial, hb x, ?_⟩
      have he : emptyRun.held (o.1, o.2, x) = K * 0 := held_periodRun 0 _ _
      rw [he]
      show ((K * 0 : ℕ) : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ)
      norm_num [HONEST_S, K]
    have h2 := h1.symm.le.trans h
    rw [εMax_toReal] at h2
    norm_num [Params.δ] at h2
  · have he : honestRun.held (o.1, o.2, x) = K * FULL := held_periodRun FULL le_rfl _
    rw [he] at hh
    change ((K * FULL : ℕ) : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ) at hh
    norm_num [HONEST_S, K, FULL] at hh

end Pous.Period.SecurityProofs
