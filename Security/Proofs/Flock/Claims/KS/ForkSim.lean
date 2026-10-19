import Proofs.Flock.Claims.Bridge.Live
import Proofs.Flock.Claims.Bridge.Dead

/-!
# Two runs forked after a number of coins, through a simulation

`forkC F j g s` runs `s` in `g` to its `j`-th coin and then continues it twice independently, `F` read on the two
outcomes, in expectation; a run that ends first is its own pair. It is the game-side law of two runs sharing their
first `j` coins.

* `forkC_send`, `forkC_nonneg`, `forkC_le_expect`, `forkC_eqRec`: it follows the strategy at a send, is nonnegative,
  is at most the expectation of a bound on its first argument, and survives a cast.
* `Sim.expect_le`, `Sim.expect₂_le`, `Sim.forkC_le`: through a simulation (`Refine.Sim`) whose returns relate the live
  and model outcomes by `R` and whose dead branches reach no outcome in `Q`, an expectation, a two-run expectation and
  a fork after `j` coins of a model function at most an indicator of `Q` are at most the live ones of a function that
  bounds it on related outcomes: the simulation matches the coins one to one.
* `NoCoin`, `Fresh`, `forkX_fresh`, `fresh_sessionK`, `forkX_sessionK`: on a game whose every coin answers its own
  send (`Fresh`), as the live session's do, `B`'s fork at round `j` (`forkX`) is the fork after `j` coins.
-/

namespace FlockSoundness.Game

variable {α : Type}

/-- One step of `forkC`: follow the strategy to the next coin, then `next` at each of its values. -/
noncomputable def forkStep (F : α → α → ℝ) (next : (g : Game α) → Strategy g → ℝ) : (g : Game α) → Strategy g → ℝ
  | .ret a, _ => F a a
  | .send _ k, s => forkStep F next (k s.1) s.2
  | @Game.coin _ C iF _ k, s => (∑ c : C, next (k c) (s c)) / (@Fintype.card C iF)

/-- **Two runs forked after `j` coins**: one run to its `j`-th coin, then two independent continuations, `F` read on
their two outcomes, in expectation. A run that ends before is its own pair. -/
noncomputable def forkC (F : α → α → ℝ) : ℕ → (g : Game α) → Strategy g → ℝ
  | 0 => expect₂ F
  | j + 1 => forkStep F (forkC F j)

theorem forkC_zero (F : α → α → ℝ) (g : Game α) (s : Strategy g) : forkC F 0 g s = expect₂ F g s := rfl

theorem forkC_send (F : α → α → ℝ) (j : ℕ) {M : Type} {k : M → Game α} (s : Strategy (.send M k)) :
    forkC F j (.send M k) s = forkC F j (k s.1) s.2 := by
  cases j <;> rfl

theorem forkC_ret (F : α → α → ℝ) (j : ℕ) (a : α) (s : Strategy (.ret a)) : forkC F j (.ret a) s = F a a := by
  cases j <;> rfl

theorem forkC_coin (F : α → α → ℝ) (j : ℕ) {C : Type} [Fintype C] [Nonempty C] {k : C → Game α}
    (s : Strategy (.coin C k)) : forkC F (j + 1) (.coin C k) s = (∑ c : C, forkC F j (k c) (s c)) / Fintype.card C :=
  rfl

theorem forkC_nonneg {F : α → α → ℝ} (hF : ∀ a a', 0 ≤ F a a') :
    ∀ (j : ℕ) (g : Game α) (s : Strategy g), 0 ≤ forkC F j g s
  | 0, g, s => expect_nonneg (fun a => expect_nonneg (hF a) g s) g s
  | _ + 1, .ret a, _ => hF a a
  | j + 1, .send _ k, s => by rw [forkC_send]; exact forkC_nonneg hF (j + 1) (k s.1) s.2
  | j + 1, @Game.coin _ C _ _ k, s =>
      div_nonneg (Finset.sum_nonneg fun c _ => forkC_nonneg hF j (k c) (s c)) (Nat.cast_nonneg _)

/-- A fork is at most the expectation of a bound on its first run. -/
theorem forkC_le_expect {F : α → α → ℝ} {f : α → ℝ} (hF : ∀ a a', F a a' ≤ f a) :
    ∀ (j : ℕ) (g : Game α) (s : Strategy g), forkC F j g s ≤ expect f g s
  | 0, g, s => expect_mono (fun a => (expect_mono (hF a) g s).trans_eq (expect_const (f a) g s)) g s
  | _ + 1, .ret a, _ => hF a a
  | j + 1, .send _ k, s => by rw [forkC_send]; exact forkC_le_expect hF (j + 1) (k s.1) s.2
  | j + 1, @Game.coin _ C _ _ k, s =>
      div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => forkC_le_expect hF j (k c) (s c)) (Nat.cast_nonneg _)

theorem forkC_eqRec {g g' : Game α} (h : g = g') (F : α → α → ℝ) (j : ℕ) (s : Strategy g) :
    forkC F j g' (h ▸ s) = forkC F j g s := by
  subst h
  rfl

theorem forkC_cast {g g' : Game α} (h : g = g') (F : α → α → ℝ) (j : ℕ) (s : Strategy g) :
    forkC F j g' (cast (congrArg Strategy h) s) = forkC F j g s := by
  subst h
  rfl

end FlockSoundness.Game

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game

section Transfer

variable {β α : Type} {pL : β → Prop} {pg : α → Prop}

/-- **An expectation through a simulation**: a model function at most a nonnegative live one on related outcomes,
and nonpositive on the dead branches, has its expectation against the simulated strategy at most the live one. -/
theorem Sim.expect_le {R : β → α → Prop} {D : (g : Game α) → Strategy g → Prop} {f : α → ℝ} {f' : β → ℝ}
    (hR : ∀ b a, R b a → f a ≤ f' b) (hf' : ∀ b, 0 ≤ f' b) (hD : ∀ g σ, D g σ → expect f g σ ≤ 0) :
    ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g), Sim.All R D d → ∀ s : Strategy L,
      expect f g (d.strategy s) ≤ expect f' L s
  | _, _, @Sim.ret _ _ _ _ b a _, h, _ => hR b a h
  | _, _, .model _ d, h, s => by simp only [expect, Sim.strategy]; exact Sim.expect_le hR hf' hD d h s
  | _, _, .live k, h, s => by simp only [expect, Sim.strategy]; exact Sim.expect_le hR hf' hD (k s.1) (h s.1) s.2
  | _, _, @Sim.coin _ _ _ _ C C' iF iN iF' iN' k k' e f, h, s => by
      simp only [expect, Sim.strategy]
      rw [← e.sum_comp, @Fintype.card_congr C' C iF' iF e]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun c' _ => Sim.expect_le hR hf' hD (f c') (h c') (s (e c'))) (Nat.cast_nonneg _)
  | _, _, .dead _ σ, h, s => (hD _ σ h).trans (expect_nonneg hf' _ s)

open Classical in
theorem expect_le_zero_of_prob {Q : α → Prop} {f : α → ℝ} (hf : ∀ a, f a ≤ ind Q a) {g : Game α} {σ : Strategy g}
    (h : prob Q g σ = 0) : expect f g σ ≤ 0 := by
  have e := prob_toReal Q g σ
  rw [h, ENNReal.toReal_zero] at e
  exact (expect_mono hf g σ).trans e.symm.le

open Classical in
/-- **Two independent runs through a simulation**: a model function at most an indicator of `Q` in each argument and
at most a nonnegative live one on related outcomes, where the dead branches reach no outcome in `Q`. -/
theorem Sim.expect₂_le {R : β → α → Prop} {Q : α → Prop} {F : α → α → ℝ} {G : β → β → ℝ}
    (hFQ : ∀ a a', F a a' ≤ ind Q a) (hFQ' : ∀ a a', F a a' ≤ ind Q a') (hG : ∀ b b', 0 ≤ G b b')
    (hRG : ∀ b a b' a', R b a → R b' a' → F a a' ≤ G b b') {L : Game β} {g : Game α} (d : Sim pL pg L g)
    (h : Sim.All R (fun g σ => prob Q g σ = 0) d) (s : Strategy L) :
    expect₂ F g (d.strategy s) ≤ expect₂ G L s := by
  unfold expect₂
  refine Sim.expect_le (f := fun a => expect (F a) g (d.strategy s)) (f' := fun b => expect (G b) L s)
    (fun b a hba => ?_) (fun b => expect_nonneg (hG b) L s) (fun g' σ' hσ' => ?_) d h s
  · exact Sim.expect_le (f := F a) (f' := G b) (fun b' a' h' => hRG b a b' a' hba h') (hG b)
      (fun _ _ hσ => expect_le_zero_of_prob (hFQ' a) hσ) d h s
  · exact expect_le_zero_of_prob (fun a => (expect_mono (fun a' => hFQ a a') g _).trans_eq
      (expect_const _ g _)) hσ'

open Classical in
/-- **A fork after `j` coins through a simulation**: as `Sim.expect₂_le`, since the simulation matches the model's
coins with the live ones one to one. -/
theorem Sim.forkC_le {R : β → α → Prop} {Q : α → Prop} {F : α → α → ℝ} {G : β → β → ℝ}
    (hFQ : ∀ a a', F a a' ≤ ind Q a) (hFQ' : ∀ a a', F a a' ≤ ind Q a') (hG : ∀ b b', 0 ≤ G b b')
    (hRG : ∀ b a b' a', R b a → R b' a' → F a a' ≤ G b b') :
    ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g), Sim.All R (fun g σ => prob Q g σ = 0) d →
      ∀ (j : ℕ) (s : Strategy L), forkC F j g (d.strategy s) ≤ forkC G j L s
  | _, _, @Sim.ret _ _ _ _ b a _, h, j, s => by
      rw [forkC_ret, forkC_ret]
      exact hRG b a b a h h
  | _, _, .model _ d, h, j, s => (forkC_send F j _).trans_le (Sim.forkC_le hFQ hFQ' hG hRG d h j s)
  | _, _, .live k, h, j, s =>
      (Sim.forkC_le hFQ hFQ' hG hRG (k s.1) (h s.1) j s.2).trans_eq (forkC_send G j s).symm
  | _, _, @Sim.coin _ _ _ _ C C' iF iN iF' iN' k k' e f, h, 0, s =>
      Sim.expect₂_le hFQ hFQ' hG hRG (Sim.coin e f) h s
  | _, _, @Sim.coin _ _ _ _ C C' iF iN iF' iN' k k' e f, h, j + 1, s => by
      rw [forkC_coin, forkC_coin]
      simp only [Sim.strategy]
      rw [← e.sum_comp, @Fintype.card_congr C' C iF' iF e]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun c' _ => Sim.forkC_le hFQ hFQ' hG hRG (f c') (h c') j (s (e c'))) (Nat.cast_nonneg _)
  | _, _, .dead _ σ, h, j, s =>
      ((forkC_le_expect hFQ j _ σ).trans (expect_le_zero_of_prob (fun _ => le_rfl) h)).trans
        (forkC_nonneg hG j _ s)

end Transfer

/-! ## `B`'s fork is the fork after as many coins -/

section Fresh

variable {β : Type}

/-- A game with no coin. -/
def NoCoin : Game β → Prop
  | .ret _ => True
  | .send _ k => ∀ m, NoCoin (k m)
  | @Game.coin _ _ _ _ _ => False

/-- **Every coin answers its own send**: from a position with no round open (`true`), the game sends before each coin;
right after a send (`false`), it draws that send's coin or draws none again. -/
def Fresh : Bool → Game β → Prop
  | _, .ret _ => True
  | true, .send _ k => ∀ m, Fresh false (k m)
  | true, @Game.coin _ _ _ _ _ => False
  | false, .send _ k => ∀ m, NoCoin (k m)
  | false, @Game.coin _ _ _ _ k => ∀ c, Fresh true (k c)

theorem noCoin_send {M : Type} {k : M → Game β} (h : NoCoin (.send M k)) (m : M) : NoCoin (k m) := by
  unfold NoCoin at h
  exact h m

theorem fresh_true_send {M : Type} {k : M → Game β} (h : Fresh true (.send M k)) (m : M) : Fresh false (k m) := by
  unfold Fresh at h
  exact h m

theorem fresh_false_send {M : Type} {k : M → Game β} (h : Fresh false (.send M k)) : NoCoin (.send M k) := by
  unfold Fresh at h
  unfold NoCoin
  exact h

theorem fresh_false_coin {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} (h : Fresh false (.coin C k))
    (c : C) : Fresh true (k c) := by
  unfold Fresh at h
  exact h c

theorem forkC_noCoin (F : β → β → ℝ) : ∀ (g : Game β) (s : Strategy g), NoCoin g → ∀ j j' : ℕ,
    forkC F j g s = forkC F j' g s
  | .ret _, _, _, j, j' => by rw [forkC_ret, forkC_ret]
  | .send _ k, s, h, j, j' => by
      rw [forkC_send, forkC_send]
      exact forkC_noCoin F (k s.1) s.2 (noCoin_send h s.1) j j'
  | @Game.coin _ _ _ _ _, _, h, _, _ => by unfold NoCoin at h; exact h.elim

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

theorem forkX_noCoin (F : β → β → ℝ) : ∀ (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I), NoCoin g →
    ∀ (j : ℕ) (p : Option Coin), forkX F j g I s p = forkC F 0 g (toStrat g I s)
  | .ret _, _, _, _, 0, _ => rfl
  | .ret _, _, _, _, _ + 1, _ => rfl
  | .send _ _, _, _, _, 0, _ => rfl
  | .send _ k, I, s, h, j + 1, _ => by
      rw [forkX, forkC_send]
      simp only [toStrat]
      rw [show (fun u : Coin => forkX F j (k (I.1 s.1)) (I.2 s.1) s.2 (some u)) =
        fun _ => forkC F 0 (k (I.1 s.1)) (toStrat _ (I.2 s.1) s.2) from
          funext fun u => forkX_noCoin F _ _ _ (noCoin_send h _) j (some u)]
      exact Ex_const _
  | @Game.coin _ _ _ _ _, _, _, h, _, _ => by unfold NoCoin at h; exact h.elim

/-- **`B`'s fork is the fork after as many coins** where every coin answers its own send: at a position with no
round open, round `j`'s fork is the fork after `j` coins; right after a send, the pending coin averaged, after `j + 1`.
-/
theorem forkX_fresh (F : β → β → ℝ) : ∀ (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I),
    (Fresh true g → ∀ (j : ℕ) (p : Option Coin), forkX F j g I s p = forkC F j g (toStrat g I s)) ∧
    (Fresh false g → ∀ j : ℕ, Ex (fun u : Coin => forkX F j g I s (some u)) = forkC F (j + 1) g (toStrat g I s))
  | .ret _, _, _ => ⟨fun _ j _ => by cases j <;> rfl, fun _ j => by cases j <;> exact Ex_const _⟩
  | .send _ k, I, s => by
    refine ⟨fun h j p => ?_, fun h j => ?_⟩
    · cases j with
      | zero => rfl
      | succ j =>
        rw [forkX, forkC_send]
        exact (forkX_fresh F _ _ _).2 (fresh_true_send h _) j
    · rw [show (fun u : Coin => forkX F j (.send _ k) I s (some u)) = fun _ => forkC F 0 (.send _ k) (toStrat _ I s)
        from funext fun u => forkX_noCoin F _ I s (fresh_false_send h) j (some u), Ex_const]
      exact forkC_noCoin F _ _ (fresh_false_send h) 0 (j + 1)
  | @Game.coin _ C _ _ k, I, s => by
    refine ⟨fun h => by unfold Fresh at h; exact h.elim, fun h j => ?_⟩
    cases j with
    | zero =>
      simp only [forkX, expectP, forkC]
      exact I.1.unif (fun c => expect₂ F (k c) (toStrat _ (I.2 c) (s c)))
    | succ j =>
      simp only [forkX]
      rw [show (fun u : Coin => forkX F (j + 1) (k (I.1.proj u)) (I.2 _) (s _) none) =
        fun u => forkC F (j + 1) (k (I.1.proj u)) (toStrat _ (I.2 _) (s _)) from
          funext fun u => (forkX_fresh F _ _ _).1 (fresh_false_coin h _) (j + 1) none,
        I.1.unif (fun c => forkC F (j + 1) (k c) (toStrat _ (I.2 c) (s c)))]
      rfl

end Fresh

namespace Live

open FlockSoundness.Discharge.Tables

theorem fresh_KLJ {J mPts : ℕ} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → FlockLevel3.GF128)
    (rs : Fin J → Fin 2 → List Flock.Round) (p : Bool) : Fresh p (KLJ rootB pts rs) := by
  cases p <;> simp [KLJ, Fresh, NoCoin]

theorem fresh_roundsK (dg : ByteArray → ByteArray) {J mPts : ℕ} (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round), Fresh true (roundsK dg rootB pts N rs)
  | 0, rs => fresh_KLJ rootB pts rs true
  | N + 1, rs => by
    simp only [roundsK, Fresh]
    intro o
    cases o with
    | none => exact fresh_KLJ rootB pts rs false
    | some s =>
      simp only [branch, Fresh]
      exact fun c => fresh_roundsK dg rootB pts N _

theorem fresh_sessionK (dg : ByteArray → ByteArray) (J mPts N : ℕ) : Fresh true (sessionK dg J mPts N) := by
  simp only [sessionK, Fresh]
  exact fun _ pts => fresh_roundsK dg _ pts N _

/-- **`B`'s fork at the live session is the fork after as many coins**. -/
theorem forkX_sessionK {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (N : ℕ)
    (F : LiveOutJ J mPts → LiveOutJ J mPts → ℝ) (j : ℕ)
    (s : RStrat (LCoin nMax) (sessionK dg J mPts N) (sessionIf w dg N)) :
    forkX F j (sessionK dg J mPts N) (sessionIf w dg N) s none =
      forkC F j (sessionK dg J mPts N) (toStrat _ (sessionIf w dg N) s) :=
  (forkX_fresh F _ _ s).1 (fresh_sessionK dg J mPts N) j none

end Live

end Claims.Bridge
