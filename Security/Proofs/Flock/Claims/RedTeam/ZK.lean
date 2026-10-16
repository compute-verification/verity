import Proofs.Flock.Claims.Prob

/-!
# Red team: `ZKmalR` holds at `ε_zk = 0` for every prover

`ZKmalR`'s bound adds the probability that the simulator's own SHA-512 calls hold a collision (`simLog`). The simulator
chooses those calls, so one that hashes a colliding pair makes that probability `1`, and the bound holds at `ε_zk = 0`
whatever the prover, the relation and the rounds: the failure of the form first frozen (`ZKmal`, which held only at
`ε_zk ≥ 1`), moved into the collision term. Every hash of bounded output length has a collision, SHA-512 among them, so
the colliding pair is a Lean term (`Classical.choice`) though nobody can print it.

* `exists_collision`: a hash whose outputs are at most `n` bytes has a collision (pigeonhole).
* `collSim u v`: the simulator that hashes `u`, then `v`, and outputs the empty transcript; two calls (`simCost_collSim`).
* `zkmalR_collSim`: at a collision `(u, v)`, `ZKmalR H k P R (collSim u v) 0` for every `k`, `P` and `R`.
* `flockZK_collSim`: Flock's `ZK` at any masked record and any hash with a collision; at SHA-512 and the deployed
  `--zk` record, `RedTeam.Deployed.flockZK_deployZk`.
-/

namespace Claims.RedTeam

/-- **A hash of bounded output length has a collision.** -/
theorem exists_collision {H : Msg → Msg} {n : ℕ} (h : ∀ u, (H u).length ≤ n) : ∃ u v, Collision H u v := by
  have : Finite {l : Msg | l.length ≤ n} := (List.finite_length_le UInt8 n).to_subtype
  obtain ⟨u, v, hne, heq⟩ :=
    Finite.exists_ne_map_eq_of_infinite fun u : Msg => (⟨H u, h u⟩ : {l : Msg | l.length ≤ n})
  exact ⟨u, v, hne, congrArg Subtype.val heq⟩

/-- **The colliding simulator**: it hashes `u`, then `v`, asks the verifier nothing and outputs the empty transcript. -/
def collSim (u v : Msg) : Simulator where
  C _ := Unit
  prog _ _ := .hash u fun _ => .hash v fun _ => .ret []

theorem findColl_collSim {H : Msg → Msg} {u v : Msg} (h : Collision H u v) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (x : Msg) (c : Unit) : findColl (simLog H (collSim u v) Vs θ x c) = some (u, v) := by
  have hvu : v ≠ u := fun e => h.1 e.symm
  simp [simLog, spLog, collSim, findColl, hvu, h.2]

theorem Pr_true {α : Type} [Fintype α] [Nonempty α] : Pr (fun _ : α => True) = 1 := by
  unfold Pr; simp

theorem Pr_le_one' {α : Type} [Fintype α] (p : α → Prop) : Pr p ≤ 1 := by
  classical
  unfold Pr
  exact div_le_one_of_le₀ (by exact_mod_cast Finset.card_filter_le _ _) (Nat.cast_nonneg _)

/-- **`ZKmalR` at `ε_zk = 0`, for every prover**, at the colliding simulator of any collision. -/
theorem zkmalR_collSim {W Ω : Type} [Fintype Ω] (H : Msg → Msg) (k : Msg → ℕ) (P : W → Prover Ω) (R : Rel W)
    {u v : Msg} (h : Collision H u v) : ZKmalR H k P R (collSim u v) 0 := by
  intro x w _ Θ Vs θ _ Ev
  have h1 : Pr (fun c : (collSim u v).C x => findColl (simLog H (collSim u v) Vs θ x c) ≠ none) = 1 := by
    rw [Pr_congr (q := fun _ => True) fun c => by simp [findColl_collSim h Vs θ x c]]
    exact Pr_true
  rw [h1, Pi.zero_apply, zero_add, abs_sub_le_iff]
  have := Pr_nonneg (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs)
  have := Pr_le_one' (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs)
  have := Pr_nonneg (fun c : (collSim u v).C x => Ev ((collSim u v).out H Vs θ x c))
  have := Pr_le_one' (fun c : (collSim u v).C x => Ev ((collSim u v).out H Vs θ x c))
  constructor <;> linarith

/-- The colliding simulator makes two calls on every coin, against every verifier. -/
theorem simCost_collSim (H : Msg → Msg) (u v : Msg) : SimCost H (collSim u v) fun _ => 2 := by
  intro x ask
  have e : ∀ c, ((((collSim u v).prog x c).eval H ask).2 : ℝ) = 2 := fun _ => by
    show ((2 : ℕ) : ℝ) = 2
    norm_num
  simp only [e, Ex_const, le_refl]

/-- **Flock's `ZK` holds at `ε_zk = 0`** at any masked record and any hash with a collision, for every honest prover. -/
theorem flockZK_collSim {Coin Col T Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Fintype Ω] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (hzk : D.zk = true) (P : Flock.Wit → Prover Ω)
    {u v : Msg} (h : Collision H u v) : Flock.ZK H D P (collSim u v) 0 :=
  ⟨hzk, zkmalR_collSim H D.V.k P _ h⟩

end Claims.RedTeam
