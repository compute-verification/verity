import Definitions.Accounting.TS.ProofLayer
import Proofs.Accounting.TS.Fork

/-!
RT3-LOGIC-1 witness (red team, not for landing): binding's fork form (`ProofLayer.Forked`) resamples round `w`'s coin
and compares round `w`'s openings, so an honest proof layer whose round-`w` openings read that round's coin forks
with probability `1 − 1/|Ω|` (`½` on `Bool`). `coinLayer n` is such a layer over `n + 1` rounds: every proof accepted,
proof `0` of round `w` opening `coinAt x w`, the coin `x` drew at round `w`, as a real round's proof carries the
receipt's keys and the draw (`LiUnits.k ω`, `window.instance_ts`). It has every round's relation `True`, so it is
sound and pinned at any `δ_ZK`. Its advantage form (`ProofLayer.Moved`, the premise of #2699's four statements) is
`½` too (`coinLayer_moved_half`): the committed value is one fixed coin, the opening the drawn one.

`game_accepts_last`: whatever the layer, the committed values of the last round are a function of the coins before
it, so on a path the play accepts whose committed values satisfy every relation, `Of` makes the ideal game accept
every last-round coin. Builds on main.
-/

namespace Accounting.SecurityProofs.TS.RtLogic3

open Finset Pouw.RelativeSecurity Accounting.TS

/-- The coin a path drew at round `w` (`false` past its end). -/
def coinAt : {n : ℕ} → Paths Bool n → ℕ → Bool
  | 0, _, _ => false
  | _ + 1, x, 0 => x.1
  | _ + 1, x, w + 1 => coinAt x.2 w

/-- An honest proof layer whose round-`w` opening is round `w`'s coin. -/
def coinLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted := fun _ _ _ => True
  opened := fun x w _ => coinAt x w
  rel := fun _ _ _ => True
  right := fun _ => True

theorem coinLayer_forked_zero (n : ℕ) (x z : Paths Bool (n + 1)) :
    (coinLayer n).Forked 0 x z ↔ x.1 ≠ z.1 := by
  simp [ProofLayer.Forked, ProofLayer.Proved, coinLayer, splice, coinAt]

/-- The all-`false` path. -/
def falsePath : (n : ℕ) → Paths Bool n
  | 0 => ()
  | n + 1 => (false, falsePath n)

/-- Flipping the first path's round-0 coin: an involution on pairs of paths. -/
def flip1 (n : ℕ) : Paths Bool (n + 1) × Paths Bool (n + 1) ≃ Paths Bool (n + 1) × Paths Bool (n + 1) where
  toFun p := ((!p.1.1, p.1.2), p.2)
  invFun p := ((!p.1.1, p.1.2), p.2)
  left_inv p := by simp
  right_inv p := by simp

theorem card_ne_twice (n : ℕ) [DecidablePred fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1]
    [DecidablePred fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => ¬ p.1.1 ≠ p.2.1] :
    (univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card * 2 =
      Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by
  have hsw : (univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card =
      (univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => ¬ p.1.1 ≠ p.2.1).card := by
    refine Finset.card_bij (fun p _ => flip1 n p) ?_ ?_ ?_
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, flip1, Equiv.coe_fn_mk] at hp ⊢
      cases h1 : p.1.1 <;> cases h2 : p.2.1 <;> simp_all
    · intro a _ b _ h
      exact (flip1 n).injective h
    · intro b hb
      refine ⟨flip1 n b, ?_, ?_⟩
      · simp only [Finset.mem_filter, Finset.mem_univ, true_and, flip1, Equiv.coe_fn_mk] at hb ⊢
        cases h1 : b.1.1 <;> cases h2 : b.2.1 <;> simp_all
      · simp [flip1]
  have hsum := Finset.card_filter_add_card_filter_not
    (s := (univ : Finset (Paths Bool (n + 1) × Paths Bool (n + 1))))
    (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1)
  rw [Finset.card_univ] at hsum
  omega

/-- Two paths' round-0 coins differ with probability `½`. -/
theorem prob_ne_half (n : ℕ) :
    prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1) = 1 / 2 := by
  classical
  have hpos : (0 : ℝ) < Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by
    exact_mod_cast Fintype.card_pos_iff.mpr ⟨(falsePath (n + 1), falsePath (n + 1))⟩
  have hN : ((univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card : ℝ) =
      1 / 2 * Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by
    have h := card_ne_twice n
    have h' : ((univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card : ℝ) * 2 =
        Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by exact_mod_cast h
    linarith
  unfold prob
  rw [div_eq_iff hpos.ne']
  convert hN

/-- **The honest coin layer forks at round 0 with probability `½`.** -/
theorem coinLayer_forked_half (n : ℕ) :
    prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => (coinLayer n).Forked 0 p.1 p.2) = 1 / 2 := by
  rw [← prob_ne_half n]
  congr 1
  funext p
  exact propext (coinLayer_forked_zero n p.1 p.2)

/-- Binding's fork form fails for the honest coin layer at any `δ_fork < ½` (`BindingForkAt`'s body, at round 0). -/
theorem coinLayer_not_fork_le (n : ℕ) (δ : ℝ) (hδ : δ < 1 / 2) :
    ¬ ∀ w, prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => (coinLayer n).Forked w p.1 p.2) ≤ δ := by
  intro h
  have := h 0
  rw [coinLayer_forked_half] at this
  linarith

/-! ## The advantage form (`Moved`), #2699's premise -/

/-- Flipping a path's round-0 coin. -/
def flip0 (n : ℕ) : Paths Bool (n + 1) ≃ Paths Bool (n + 1) where
  toFun x := (!x.1, x.2)
  invFun x := (!x.1, x.2)
  left_inv x := by simp
  right_inv x := by simp

theorem card_first_ne_twice (n : ℕ) (b : Bool) :
    (univ.filter fun x : Paths Bool (n + 1) => x.1 ≠ b).card * 2 = Fintype.card (Paths Bool (n + 1)) := by
  have hsw : (univ.filter fun x : Paths Bool (n + 1) => x.1 ≠ b).card =
      (univ.filter fun x : Paths Bool (n + 1) => ¬ x.1 ≠ b).card := by
    refine Finset.card_bij (fun p _ => flip0 n p) ?_ ?_ ?_
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, flip0, Equiv.coe_fn_mk] at hp ⊢
      cases h1 : p.1 <;> cases b <;> simp_all
    · intro a _ c _ h
      exact (flip0 n).injective h
    · intro c hc
      refine ⟨flip0 n c, ?_, ?_⟩
      · simp only [Finset.mem_filter, Finset.mem_univ, true_and, flip0, Equiv.coe_fn_mk] at hc ⊢
        cases h1 : c.1 <;> cases b <;> simp_all
      · simp [flip0]
  have hsum := Finset.card_filter_add_card_filter_not
    (s := (univ : Finset (Paths Bool (n + 1)))) (fun x : Paths Bool (n + 1) => x.1 ≠ b)
  rw [Finset.card_univ] at hsum
  omega

/-- A path's round-0 coin differs from a fixed one with probability `½`. -/
theorem prob_first_ne_half (n : ℕ) (b : Bool) : prob (fun x : Paths Bool (n + 1) => x.1 ≠ b) = 1 / 2 := by
  classical
  have hpos : (0 : ℝ) < Fintype.card (Paths Bool (n + 1)) := by
    exact_mod_cast Fintype.card_pos_iff.mpr ⟨falsePath (n + 1)⟩
  have hN : ((univ.filter fun x : Paths Bool (n + 1) => x.1 ≠ b).card : ℝ) =
      1 / 2 * Fintype.card (Paths Bool (n + 1)) := by
    have h := card_first_ne_twice n b
    have h' : ((univ.filter fun x : Paths Bool (n + 1) => x.1 ≠ b).card : ℝ) * 2 =
        Fintype.card (Paths Bool (n + 1)) := by exact_mod_cast h
    linarith
  unfold prob
  rw [div_eq_iff hpos.ne']
  convert hN

theorem coinLayer_provedAt_zero (n : ℕ) (x : Paths Bool (n + 1)) : ((coinLayer n).provedAt x 0).Nonempty :=
  ⟨x, by simp [ProofLayer.provedAt, ProofLayer.Proved, coinLayer]⟩

/-- Round 0's reference continuation is the same for every path: its continuations are every path. -/
theorem coinLayer_ref_zero (n : ℕ) (x : Paths Bool (n + 1)) :
    (coinLayer n).ref x 0 = (coinLayer n).ref (falsePath (n + 1)) 0 := by
  have h := Accounting.SecurityProofs.TS.ProofLayer.ref_splice (coinLayer n) 0 (falsePath (n + 1)) x
    (coinLayer_provedAt_zero n _)
  have hs : splice ((0 : Fin (n + 1)) : ℕ) (falsePath (n + 1)) x = x := by simp [splice]
  rwa [hs] at h

theorem coinLayer_moved_zero (n : ℕ) (x : Paths Bool (n + 1)) :
    (coinLayer n).Moved x 0 ↔ x.1 ≠ ((coinLayer n).ref (falsePath (n + 1)) 0).1 := by
  have hc : ∀ (y : Paths Bool (n + 1)) i, (coinLayer n).opened y 0 i = y.1 := fun y i => by
    simp [coinLayer, coinAt]
  simp only [ProofLayer.Moved, ProofLayer.committed, coinLayer_ref_zero n x, hc]
  exact ⟨fun h => h.2.elim fun _ hi => hi, fun h => ⟨fun _ => trivial, 0, h⟩⟩

/-- **The honest coin layer moves at round 0 with probability `½`.** -/
theorem coinLayer_moved_half (n : ℕ) :
    prob (fun x : Paths Bool (n + 1) => (coinLayer n).Moved x 0) = 1 / 2 := by
  rw [← prob_first_ne_half n ((coinLayer n).ref (falsePath (n + 1)) 0).1]
  congr 1
  funext x
  exact propext (coinLayer_moved_zero n x)

theorem coinLayer_not_moved_le (n : ℕ) (δ : ℝ) (hδ : δ < 1 / 2) :
    ¬ ∀ w, prob (fun x : Paths Bool (n + 1) => (coinLayer n).Moved x w) ≤ δ := by
  intro h
  have := h 0
  rw [coinLayer_moved_half] at this
  linarith

/-- #2699's binding premise, `∀ w, prob (fun x => pf.Moved x w) ≤ δbind`, as a definition over any `S`. -/
def MovedAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) (δbind : ℝ) : Prop :=
  ∀ w, prob (fun x => pf.Moved x w) ≤ δbind

/-- A budget `s·(δ_bind + n_π·δ_ZK) ≤ 2^-41` over `s ≥ 1` rounds and the binding premise fail together for the coin
layer; #2699's is `s = 48,000`, `n = 47,999` (`coinLayer_moved_budget_unsat_dc`). -/
theorem coinLayer_moved_budget_unsat (n : ℕ) (s δbind δzk : ℝ) (hs : 1 ≤ s) (hzk : 0 ≤ δzk)
    (hB : MovedAt (coinLayer n) δbind) : ¬ s * (δbind + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 := by
  intro hb
  have hδ : 1 / 2 ≤ δbind := by
    by_contra h
    exact coinLayer_not_moved_le n δbind (not_le.mp h) hB
  have h1 : (1 / 2 : ℝ) ≤ δbind + ((1 : ℕ) : ℝ) * δzk := by push_cast; linarith
  have h2 := mul_le_mul_of_nonneg_right hs (show (0 : ℝ) ≤ δbind + ((1 : ℕ) : ℝ) * δzk by linarith)
  have h3 : ((1 : ℝ) / 2) ^ 41 < 1 / 2 := by norm_num
  linarith

theorem coinLayer_moved_budget_unsat_dc (δbind δzk : ℝ) (hzk : 0 ≤ δzk) (hB : MovedAt (coinLayer 47999) δbind) :
    ¬ (48000 : ℝ) * (δbind + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 :=
  coinLayer_moved_budget_unsat 47999 48000 δbind δzk (by norm_num) hzk hB

/-! ## `Of` and the last round -/

theorem splice_self {Ω : Type} : ∀ {n : ℕ} (w : ℕ) (x : Paths Ω n), splice w x x = x
  | 0, _, _ => rfl
  | _ + 1, 0, _ => rfl
  | _ + 1, w + 1, x => by simp only [splice, splice_self w x.2, Prod.mk.eta]

theorem splice_absorb {Ω : Type} : ∀ {n : ℕ} (w v : ℕ), w ≤ v → ∀ (x z : Paths Ω n),
    splice w x (splice v x z) = splice v x z
  | 0, _, _, _, _, _ => rfl
  | _ + 1, 0, _, _, _, _ => rfl
  | _ + 1, w + 1, 0, h, _, _ => absurd h (by omega)
  | _ + 1, w + 1, v + 1, h, x, z => by simp only [splice, splice_absorb w v (by omega) x.2 z.2]

/-- Round `w`'s committed values are those of every continuation of `x` from a round `v ≥ w`. -/
theorem committed_splice_of_le {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) (w : Fin S)
    (v : ℕ) (hwv : (w : ℕ) ≤ v) (x z : Paths Ω S) (h : (pf.provedAt x w).Nonempty) (i : Fin nπ) :
    pf.committed (splice v x z) w i = pf.committed x w i := by
  have hr : pf.ref (splice v x z) w = pf.ref x w := by
    have := Accounting.SecurityProofs.TS.ProofLayer.ref_splice pf w x (splice v x z) h
    rwa [splice_absorb (w : ℕ) v hwv x z] at this
  simp only [ProofLayer.committed, hr]

/-- **`Of` accepts every last coin**: on a path the play accepts whose committed values satisfy every relation, the
ideal game accepts every continuation from the last round (round `S − 1`'s coin resampled), whatever the layer. -/
theorem game_accepts_last {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V K T L U : Type} (pf : ProofLayer Ω S nπ V)
    (P : Play Ω S K T L U) (hOf : pf.Of P) (x : Paths Ω S) (hx : P.accepts x)
    (hrel : ∀ w i, pf.rel w i (pf.committed x w i)) (z : Paths Ω S) : P.game.accepts (splice (S - 1) x z) := by
  apply hOf.2
  intro w i
  have hne : (pf.provedAt x w).Nonempty := ⟨x, by
    simp only [ProofLayer.provedAt, Finset.mem_filter, Finset.mem_univ, true_and, splice_self]
    exact hOf.1 x hx w⟩
  have hw : (w : ℕ) ≤ S - 1 := by have := w.isLt; omega
  rw [committed_splice_of_le pf w (S - 1) hw x z hne i]
  exact hrel w i

end Accounting.SecurityProofs.TS.RtLogic3
