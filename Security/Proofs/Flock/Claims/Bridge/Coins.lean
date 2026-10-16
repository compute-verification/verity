import Proofs.Flock.Claims.Prob
import Proofs.Flock.Soundness.Game.Expect

/-!
# The coin law: a game played on `Claims`' coin sequence

The live games (`FlockSoundness.Game`) draw a coin of their own type at each `coin` node; `Claims.run` gives the
verifier one coin `r_j : Coin` per round, answering the prover's message `a_j`. This file plays a game on the round
coins and proves that the result has the game's law.

* `Iface Coin g`: how the game's moves and coins sit in `Claims`' messages: at each `send` node a total decoder of the
  prover's bytes (`dec`), at each `coin` node a uniform projection of the round coin onto the game's coin (`CoinIf`),
  with the bytes of the verifier's reply that carries it. Only decoded moves need an interface below them.
* `RStrat g I`: a prover that sends bytes at every `send` node; `toStrat` reads it as a `Strategy` of the game.
* `playOn g I s p r`: the game played on the round coins `r`, one per round. A round opens with the prover's message:
  at a `send` node it is the move, and the round's coin `r_j` stays pending; a `coin` node right after reads the pending
  coin (its projection), any other `coin` node opens a round of its own, whose message the game ignores.
* `RoundsLe g I s p n`: the interaction ends within `n` rounds, counting the round that carries the decision.
* `ex_playOn` (**the coin law**): `Ex_r f(playOn … r) = expect f g (toStrat s)` over `r : Fin n → Coin` uniform: each
  game coin is uniform, independent of the other coins, and drawn after the message it answers.
* `forkX`, `ex_fork`: the same for two plays sharing their first `j` round coins (`Claims.forkCoins j`), the law of
  `B`'s two runs: one play to round `j`, then two independent continuations.
-/

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game

/-- How a round coin gives one game coin: a uniform projection, and the bytes of the verifier's reply that carries the
game coin, with their decoding. -/
structure CoinIf (Coin C : Type) [Fintype Coin] [Fintype C] where
  proj : Coin → C
  unif : ∀ h : C → ℝ, Ex (fun u => h (proj u)) = Ex h
  enc : C → Msg
  dec : Msg → C
  dec_enc : ∀ c, dec (enc c) = c

section Iface

variable {β : Type} (Coin : Type) [Fintype Coin]

/-- **The interface of a game with `Claims`' messages and coins**: a total decoder of the prover's bytes at each `send`
node (only decoded moves need an interface below), a uniform projection of the round coin at each `coin` node. -/
def Iface : Game β → Type
  | .ret _ => PUnit
  | .send M k => (dec : Msg → M) × ((a : Msg) → Iface (k (dec a)))
  | @Game.coin _ C iF _ k => @CoinIf Coin C _ iF × ((c : C) → Iface (k c))

/-- A prover that sends bytes at every `send` node, as a function of the history (its position in the tree). -/
def RStrat : (g : Game β) → Iface Coin g → Type
  | .ret _, _ => PUnit
  | .send _ k, I => (a : Msg) × RStrat (k (I.1 a)) (I.2 a)
  | @Game.coin _ C _ _ k, I => (c : C) → RStrat (k c) (I.2 c)

end Iface

variable {β Coin : Type} [Fintype Coin]

/-- The bytes prover as a strategy of the game: its moves are its bytes, decoded. -/
def toStrat : (g : Game β) → (I : Iface Coin g) → RStrat Coin g I → Strategy g
  | .ret _, _, _ => PUnit.unit
  | .send _ k, I, s => ⟨I.1 s.1, toStrat (k (I.1 s.1)) (I.2 s.1) s.2⟩
  | @Game.coin _ _ _ _ k, I, s => fun c => toStrat (k c) (I.2 c) (s c)

/-- **The game played on the round coins** `r`, from a position whose round coin `p` is pending (`some`: the prover's
message of the current round was a move, and its coin is not read yet). A `send` node takes the next round; a `coin`
node reads the pending coin, or takes the next round when none is pending. -/
def playOn : (g : Game β) → (I : Iface Coin g) → RStrat Coin g I → Option Coin → (ℕ → Coin) → β
  | .ret b, _, _, _, _ => b
  | .send _ k, I, s, _, r => playOn (k (I.1 s.1)) (I.2 s.1) s.2 (some (r 0)) fun i => r (i + 1)
  | @Game.coin _ _ _ _ k, I, s, some u, r => playOn (k (I.1.proj u)) (I.2 _) (s _) none r
  | @Game.coin _ _ _ _ k, I, s, none, r => playOn (k (I.1.proj (r 0))) (I.2 _) (s _) none fun i => r (i + 1)

/-- **The interaction ends within `n` rounds** from a position (`p`: a round is open, its message read), counting the
round whose reply is the decision: a `ret` with no round open takes one more. -/
def RoundsLe : (g : Game β) → (I : Iface Coin g) → RStrat Coin g I → Bool → ℕ → Prop
  | .ret _, _, _, true, _ => True
  | .ret _, _, _, false, n => 0 < n
  | .send _ _, _, _, _, 0 => False
  | .send _ k, I, s, _, n + 1 => RoundsLe (k (I.1 s.1)) (I.2 s.1) s.2 true n
  | @Game.coin _ _ _ _ k, I, s, true, n => ∀ c, RoundsLe (k c) (I.2 c) (s c) false n
  | @Game.coin _ _ _ _ _, _, _, false, 0 => False
  | @Game.coin _ _ _ _ k, I, s, false, n + 1 => ∀ c, RoundsLe (k c) (I.2 c) (s c) false n

/-- The expectation of `f` from a position with the pending coin `p`: the game's own `expect`, a pending coin read by a
`coin` node. -/
noncomputable def expectP (f : β → ℝ) : (g : Game β) → (I : Iface Coin g) → RStrat Coin g I → Option Coin → ℝ
  | .ret b, _, _, _ => f b
  | .send M k, I, s, _ => expect f (.send M k) (toStrat _ I s)
  | @Game.coin _ _ _ _ k, I, s, some u => expect f (k (I.1.proj u)) (toStrat _ _ (s (I.1.proj u)))
  | @Game.coin _ C iF iN k, I, s, none => expect f (@Game.coin _ C iF iN k) (toStrat _ I s)

/-- **Two plays forked at round `j`**: the plays share their first `j` round coins and then continue independently;
`F` read on the two outcomes, in expectation. -/
noncomputable def forkX (F : β → β → ℝ) : ℕ → (g : Game β) → (I : Iface Coin g) → RStrat Coin g I → Option Coin → ℝ
  | 0, g, I, s, p => expectP (fun b => expectP (F b) g I s p) g I s p
  | _ + 1, .ret b, _, _, _ => F b b
  | j + 1, .send _ k, I, s, _ => Ex fun u : Coin => forkX F j (k (I.1 s.1)) (I.2 s.1) s.2 (some u)
  | j + 1, @Game.coin _ _ _ _ k, I, s, some u => forkX F (j + 1) (k (I.1.proj u)) (I.2 _) (s _) none
  | j + 1, @Game.coin _ _ _ _ k, I, s, none => Ex fun u : Coin => forkX F j (k (I.1.proj u)) (I.2 _) (s _) none

/-! ## Uniform averages over round coins -/

theorem Ex_cons {n : ℕ} (f : (Fin (n + 1) → Coin) → ℝ) :
    Ex f = Ex fun c : Coin => Ex fun r : Fin n → Coin => f (Fin.cons c r) := by
  classical
  unfold Ex
  rw [← Fintype.sum_equiv (Fin.consEquiv fun _ => Coin) (fun p => f (Fin.cons p.1 p.2)) f (fun _ => rfl),
    Fintype.sum_prod_type, Fintype.card_congr (Fin.consEquiv fun _ : Fin (n + 1) => Coin).symm,
    Fintype.card_prod, Finset.sum_div, Finset.sum_div]
  refine Finset.sum_congr rfl fun c _ => ?_
  push_cast
  rw [div_div, mul_comm]

theorem Ex_prod {α γ : Type} [Fintype α] [Fintype γ] (f : α × γ → ℝ) :
    Ex f = Ex fun a => Ex fun c => f (a, c) := by
  unfold Ex
  rw [Fintype.sum_prod_type, Fintype.card_prod, Finset.sum_div]
  push_cast
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [div_div, mul_comm]

theorem Ex_comm {α γ : Type} [Fintype α] [Fintype γ] (f : α → γ → ℝ) :
    (Ex fun a => Ex fun c => f a c) = Ex fun c => Ex fun a => f a c := by
  unfold Ex
  simp only [Finset.sum_div, div_div]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun a _ => ?_
  rw [mul_comm]

theorem Ex_const' {α : Type} [Fintype α] [Nonempty α] (c : ℝ) : Ex (fun _ : α => c) = c := Ex_const c

variable [Inhabited Coin]

omit [Fintype Coin] in
theorem coinSeq_cons_zero {n : ℕ} (c : Coin) (r : Fin n → Coin) :
    coinSeq (Fin.cons c r : Fin (n + 1) → Coin) 0 = c := by
  simp [coinSeq]

omit [Fintype Coin] in
theorem coinSeq_cons_succ {n : ℕ} (c : Coin) (r : Fin n → Coin) :
    (fun i => coinSeq (Fin.cons c r : Fin (n + 1) → Coin) (i + 1)) = coinSeq r := by
  funext i
  unfold coinSeq
  by_cases h : i < n
  · have h' : i + 1 < n + 1 := by omega
    simp only [h, h', ↓reduceDIte]
    exact Fin.cons_succ (α := fun _ => Coin) c r ⟨i, h⟩
  · have h' : ¬ i + 1 < n + 1 := by omega
    simp only [h, h', ↓reduceDIte]

omit [Fintype Coin] [Inhabited Coin] in
theorem forkCoins_zero {n : ℕ} (j : Fin n) (hj : j.val = 0) (r r' : Fin n → Coin) : forkCoins j r r' = r' := by
  funext i
  simp [forkCoins, hj]

omit [Fintype Coin] [Inhabited Coin] in
theorem forkCoins_cons {n : ℕ} (j : Fin n) (c c' : Coin) (r r' : Fin n → Coin) :
    forkCoins j.succ (Fin.cons c r : Fin (n + 1) → Coin) (Fin.cons c' r') = Fin.cons c (forkCoins j r r') := by
  funext i
  refine Fin.cases ?_ (fun i => ?_) i
  · simp [forkCoins]
  · simp only [forkCoins, Fin.val_succ, Fin.cons_succ, add_lt_add_iff_right]

theorem playOn_send_cons {M : Type} {k : M → Game β} {n : ℕ} (I : Iface Coin (.send M k))
    (s : RStrat Coin (.send M k) I) (p : Option Coin) (c : Coin) (r : Fin n → Coin) :
    playOn (.send M k) I s p (coinSeq (Fin.cons c r : Fin (n + 1) → Coin)) =
      playOn (k (I.1 s.1)) (I.2 s.1) s.2 (some c) (coinSeq r) := by
  simp only [playOn]
  rw [coinSeq_cons_zero, coinSeq_cons_succ]

theorem playOn_coin_cons {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {n : ℕ} (I : Iface Coin (.coin C k))
    (s : RStrat Coin (.coin C k) I) (c : Coin) (r : Fin n → Coin) :
    playOn (.coin C k) I s none (coinSeq (Fin.cons c r : Fin (n + 1) → Coin)) =
      playOn (k (I.1.proj c)) (I.2 _) (s _) none (coinSeq r) := by
  simp only [playOn]
  rw [coinSeq_cons_zero, coinSeq_cons_succ]

/-! ## The coin law -/

omit [Inhabited Coin] in
theorem expectP_none (f : β → ℝ) (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I) :
    expectP f g I s none = expect f g (toStrat g I s) := by
  cases g <;> rfl

omit [Inhabited Coin] in
/-- A pending coin, averaged, is a coin not yet drawn: the projection is uniform. -/
theorem ex_expectP_some [Nonempty Coin] (f : β → ℝ) (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I) :
    Ex (fun u : Coin => expectP f g I s (some u)) = expectP f g I s none := by
  cases g with
  | ret b => exact Ex_const _
  | send M k => exact Ex_const _
  | coin C k =>
    simp only [expectP]
    rw [I.1.unif (fun c => expect f (k c) (toStrat _ _ (s c)))]
    rfl

/-- **The coin law**: the game played on uniform round coins has the game's law, `expect` against the decoded
strategy, whenever it ends within the rounds. -/
theorem ex_playOn [Nonempty Coin] (f : β → ℝ) : ∀ (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I)
    (p : Option Coin) (n : ℕ), RoundsLe g I s p.isSome n →
      Ex (fun r : Fin n → Coin => f (playOn g I s p (coinSeq r))) = expectP f g I s p := by
  intro g
  induction g with
  | ret b => intro I s p n _; exact Ex_const (α := Fin n → Coin) (f b)
  | send M k ih =>
    intro I s p n hR
    cases n with
    | zero => cases p <;> exact hR.elim
    | succ n =>
      rw [Ex_cons]
      simp only [playOn_send_cons]
      have hR' : RoundsLe (k (I.1 s.1)) (I.2 s.1) s.2 true n := by cases p <;> exact hR
      rw [show (fun c : Coin => Ex fun r : Fin n → Coin => f (playOn (k (I.1 s.1)) (I.2 s.1) s.2 (some c)
          (coinSeq r))) = fun c => expectP f (k (I.1 s.1)) (I.2 s.1) s.2 (some c) from
        funext fun c => ih _ _ _ (some c) n hR', ex_expectP_some, expectP_none]
      rfl
  | coin C k ih =>
    intro I s p n hR
    cases p with
    | some u =>
      simp only [playOn]
      rw [ih _ _ _ none n (hR _), expectP_none]
      rfl
    | none =>
      cases n with
      | zero => exact hR.elim
      | succ n =>
        rw [Ex_cons]
        simp only [playOn_coin_cons]
        rw [show (fun c : Coin => Ex fun r : Fin n → Coin => f (playOn (k (I.1.proj c)) (I.2 _) (s _) none
            (coinSeq r))) = fun c => expect f (k (I.1.proj c)) (toStrat _ _ (s (I.1.proj c))) from
          funext fun c => (ih _ _ _ none n (hR _)).trans (expectP_none _ _ _ _),
          I.1.unif (fun c => expect f (k c) (toStrat _ _ (s c)))]
        rfl

/-- The coin law at the root: no round open. -/
theorem ex_playOn_root [Nonempty Coin] (f : β → ℝ) (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I) (n : ℕ)
    (hR : RoundsLe g I s false n) :
    Ex (fun r : Fin n → Coin => f (playOn g I s none (coinSeq r))) = expect f g (toStrat g I s) :=
  (ex_playOn f g I s none n hR).trans (expectP_none f g I s)

theorem ex_fork_zero [Nonempty Coin] (F : β → β → ℝ) (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I)
    (p : Option Coin) (n : ℕ) (j : Fin n) (hj : j.val = 0) (hR : RoundsLe g I s p.isSome n) :
    Ex (fun rr : (Fin n → Coin) × (Fin n → Coin) =>
      F (playOn g I s p (coinSeq rr.1)) (playOn g I s p (coinSeq (forkCoins j rr.1 rr.2)))) =
      forkX F 0 g I s p := by
  rw [forkX.eq_1, Ex_prod]
  simp only [forkCoins_zero j hj]
  rw [show (fun r : Fin n → Coin => Ex fun r' : Fin n → Coin => F (playOn g I s p (coinSeq r))
      (playOn g I s p (coinSeq r'))) = fun r => expectP (F (playOn g I s p (coinSeq r))) g I s p from
    funext fun r => ex_playOn (F (playOn g I s p (coinSeq r))) g I s p n hR]
  exact ex_playOn (fun b => expectP (F b) g I s p) g I s p n hR

/-- **The fork law**: two plays sharing their first `j` round coins, the rest independent, have the law `forkX`: one
play to round `j`, then two independent continuations. -/
theorem ex_fork [Nonempty Coin] (F : β → β → ℝ) : ∀ (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I)
    (p : Option Coin) (n j : ℕ) (hj : j < n), RoundsLe g I s p.isSome n →
      Ex (fun rr : (Fin n → Coin) × (Fin n → Coin) =>
        F (playOn g I s p (coinSeq rr.1)) (playOn g I s p (coinSeq (forkCoins ⟨j, hj⟩ rr.1 rr.2)))) =
        forkX F j g I s p := by
  intro g
  induction g with
  | ret b =>
    intro I s p n j hj _
    cases j with
    | zero => exact Ex_const (α := (Fin n → Coin) × (Fin n → Coin)) (F b b)
    | succ j => exact Ex_const (α := (Fin n → Coin) × (Fin n → Coin)) (F b b)
  | send M k ih =>
    intro I s p n j hj hR
    cases j with
    | zero => exact ex_fork_zero F _ I s p n _ rfl hR
    | succ j =>
      cases n with
      | zero => omega
      | succ n =>
        have hj' : j < n := by omega
        have hR' : RoundsLe (k (I.1 s.1)) (I.2 s.1) s.2 true n := by cases p <;> exact hR
        rw [show (⟨j + 1, hj⟩ : Fin (n + 1)) = Fin.succ ⟨j, hj'⟩ from rfl, Ex_prod]
        simp only [Ex_cons, playOn_send_cons, forkCoins_cons, Ex_const]
        rw [forkX]
        refine congrArg Ex (funext fun c => ?_)
        rw [← ih (I.1 s.1) (I.2 s.1) s.2 (some c) n j hj' hR', Ex_prod]
  | coin C k ih =>
    intro I s p n j hj hR
    cases j with
    | zero => exact ex_fork_zero F _ I s p n _ rfl hR
    | succ j =>
      cases p with
      | some u =>
        simp only [playOn]
        rw [forkX, ← ih _ _ _ none n (j + 1) hj (hR _)]
      | none =>
        cases n with
        | zero => omega
        | succ n =>
          have hj' : j < n := by omega
          rw [show (⟨j + 1, hj⟩ : Fin (n + 1)) = Fin.succ ⟨j, hj'⟩ from rfl, Ex_prod]
          simp only [Ex_cons, playOn_coin_cons, forkCoins_cons, Ex_const]
          rw [forkX]
          refine congrArg Ex (funext fun c => ?_)
          rw [← ih _ _ _ none n j hj' (hR _), Ex_prod]

end Claims.Bridge
