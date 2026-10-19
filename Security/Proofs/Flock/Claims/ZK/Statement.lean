import Proofs.Flock.Claims.Prob

/-!
# Zero knowledge against a malicious verifier: what the proofs add to the spec's `ZKmalR`

The claim is the spec's `Claims.ZKmalR` (the real run lasts the protocol's `k(x)` rounds, and the bound adds `γ` times
the probability that `Fork`, two real runs of the honest prover on one nonce, finds a collision among the prover's
SHA-512 calls), and Flock's is `Claims.Flock.ZK`, at the masked verifier's rounds `D.V.k` plus the `Hello` that commits
every coin. This module adds:

* **ZK bits** (`ZKmalR.bits`): under `StrictBirthday` at `Fork` the collision term is at most `γ·q²/2^513`.
* **The worst-case cost** (`SimCostLe`), beside the spec's expected `SimCost`, which it implies (`SimCostLe.simCost`).
* What `findColl` finds (`findColl_ne_none_iff`, `findColl_collision`).
-/

namespace Claims.ZKMal

open Claims

/-- **ZK bits, proposed**: `ZKmalR` with its collision term bounded by the strict birthday bound at `Fork`, for every
verifier whose two runs with the honest prover make at most `q` SHA-512 calls. -/
def ZKmalBits {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (k : Msg → ℕ)
    (P : W → Prover (N × Ω)) (R : Rel W) (S : Simulator) (εzk γ : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ q : ℝ,
    (∀ s, ((Fork H (P w) Vs θ x).cost s : ℝ) ≤ q) → StrictBirthday H (Fork H (P w) Vs θ x) →
    ∀ (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : N × Ω => Ev (runStar H (P w) Vs x ω θ).msgs) - Pr (fun c : S.C x => Ev (S.out H Vs θ x c))|
      ≤ εzk x + γ x * (q ^ 2 / 2 ^ (513 : ℝ))

/-- **`ZKmalR` gives ZK bits** at a nonnegative `γ`: the collision term is `Fork`'s success, which the strict birthday
bound caps. -/
theorem ZKmalR.bits {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : Msg → ℕ}
    {P : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x)
    (h : ZKmalR H k P R S εzk γ) : ZKmalBits H k P R S εzk γ := by
  intro x w hR Θ Vs θ hk q hq hSB Ev
  exact (h x w hR Θ Vs θ hk Ev).trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left (hSB q hq) (hγ x)))

/-- **The simulator's cost, worst case**: at most `bound(x)` calls (hash and ask) on every coin, against every
verifier. -/
def SimCostLe (H : Msg → Msg) (S : Simulator) (bound : Msg → ℕ) : Prop :=
  ∀ x (c : S.C x) (ask : List Msg → Out), ((S.prog x c).eval H ask).2 ≤ bound x

/-- A worst-case bound is an expected one. -/
theorem SimCostLe.simCost {H : Msg → Msg} {S : Simulator} {bound : Msg → ℕ} (h : SimCostLe H S bound) :
    SimCost H S fun x => (bound x : ℝ) := by
  intro x ask
  unfold Ex
  rw [div_le_iff₀ (by exact_mod_cast Fintype.card_pos)]
  calc (∑ c : S.C x, (((S.prog x c).eval H ask).2 : ℝ)) ≤ ∑ _c : S.C x, (bound x : ℝ) :=
        Finset.sum_le_sum fun c _ => by exact_mod_cast h x c ask
    _ = (bound x : ℝ) * Fintype.card (S.C x) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm]

/-! ## What `findColl` finds -/

/-- `findColl` finds a collision exactly when two calls in the log have different inputs and one output. -/
theorem findColl_ne_none_iff : ∀ l : List (Msg × Msg),
    findColl l ≠ none ↔ ∃ p ∈ l, ∃ q ∈ l, p.1 ≠ q.1 ∧ p.2 = q.2
  | [] => by simp [findColl]
  | (u, a) :: l => by
      have ih := findColl_ne_none_iff l
      unfold findColl
      cases hf : l.find? (fun p => decide (p.1 ≠ u) && decide (p.2 = a)) with
      | some p =>
          have hp := List.find?_some hf
          have hm := List.mem_of_find?_eq_some hf
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
          simp only [ne_eq, reduceCtorEq, not_false_eq_true, true_iff]
          exact ⟨(u, a), List.mem_cons_self .., p, List.mem_cons_of_mem _ hm, fun h => hp.1 h.symm, hp.2.symm⟩
      | none =>
          simp only
          rw [ih]
          have hn : ∀ p ∈ l, ¬ (p.1 ≠ u ∧ p.2 = a) := by
            intro p hp h
            have := List.find?_eq_none.mp hf p hp
            simp only [Bool.and_eq_true, decide_eq_true_eq, not_and] at this
            exact this h.1 h.2
          constructor
          · rintro ⟨p, hp, q, hq, h⟩
            exact ⟨p, List.mem_cons_of_mem _ hp, q, List.mem_cons_of_mem _ hq, h⟩
          · rintro ⟨p, hp, q, hq, hne, heq⟩
            rcases List.mem_cons.mp hp with rfl | hp' <;> rcases List.mem_cons.mp hq with rfl | hq'
            · exact absurd rfl hne
            · exact absurd ⟨fun h => hne h.symm, heq.symm⟩ (hn q hq')
            · exact absurd ⟨hne, heq⟩ (hn p hp')
            · exact ⟨p, hp', q, hq', hne, heq⟩

/-- A collision in a log of `H` calls is a collision of `H`. -/
theorem findColl_collision (H : Msg → Msg) : ∀ l : List (Msg × Msg), (∀ p ∈ l, p.2 = H p.1) →
    ∀ u v, findColl l = some (u, v) → Collision H u v
  | [], _, _, _, h => by simp [findColl] at h
  | (q, a) :: l, hl, u, v, h => by
      unfold findColl at h
      cases hf : l.find? (fun p => decide (p.1 ≠ q) && decide (p.2 = a)) with
      | some p =>
          rw [hf] at h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          have hp := List.find?_some hf
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
          have ha := hl (q, a) (List.mem_cons_self ..)
          have hb := hl p (List.mem_cons_of_mem _ (List.mem_of_find?_eq_some hf))
          exact ⟨fun h => hp.1 h.symm, by simp only at ha; rw [← ha, ← hb, hp.2]⟩
      | none =>
          rw [hf] at h
          exact findColl_collision H l (fun p hp => hl p (List.mem_cons_of_mem _ hp)) u v h

end Claims.ZKMal

namespace Claims.Flock

/-- **Flock's `ZK` at a given prover and simulator**: what `Claims.Flock.ZK` says of a masked record, at the record's
reference prover and simulator (`ZK_iff_at`), stated at any. The proofs prove it at a committed-coin protocol's prover,
which is a masked record's reference prover where the protocol's messages are the record's (`ZKMal.Proto.Honest`,
`ZKMal.zk_zkOf`). -/
def ZKAt {Coin Col T N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (D : Deploy Coin Col T) (P : Wit → Prover (N × Ω)) (S : Simulator) (εzk γ : Msg → ℝ) : Prop :=
  D.zk = true ∧ ZKmalR H (fun x => D.V.k x + 1) P (R D.parse) S εzk γ

theorem ZK_iff_at {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : ZkDeploy Coin Col T) (εzk γ : Msg → ℝ) :
    ZK H D εzk γ ↔ ZKAt H D.toDeploy D.prover D.S εzk γ := Iff.rfl

end Claims.Flock
