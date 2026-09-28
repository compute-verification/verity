import Proofs.Flock.Soundness.Refine.Tapes

/-!
# The live session, and simulating it in the model

The transfer (R11) compares two interactions. The **live** one is the challenger's session as the executable sees it:
the prover sends framed bytes, the server draws coins, and at the end the executable checks what the server recorded.
The **model** one is the compiled game the soundness theorems bound.

* `Sim pL pg L g` is a simulation of a live interaction `L` by a model interaction `g`. It is data: how the model answers
  each live move, bijections between their coins, and a strategy for the model wherever no live run can succeed. With it
  come the facts that make a live acceptance (`pL`) a model acceptance (`pg`). `Sim.strategy` is the model strategy it
  plays against a live one, and `Sim.prob_le` compares the two probabilities.
* `liveTable` is one table's live session on the Blake3 unsalted path, whose server retains every round's bytes.
-/

namespace FlockSoundness.Game

open scoped ENNReal

variable {α β : Type}

/-- **A simulation** of a live interaction `L` by a model interaction `g`, for the live verdict `pL` and the model's `pg`.
At every live move it says what the model does:
* `ret`: both end, and a live acceptance is a model acceptance;
* `model`: the model sends `m` first;
* `live`: the live prover sends, and the model answers each message its own way;
* `coin`: both draw a coin, matched by a bijection;
* `dead`: no live run from here is accepted, and the model plays a given strategy. -/
inductive Sim (pL : β → Prop) (pg : α → Prop) : Game β → Game α → Type 1 where
  | ret {b : β} {a : α} : (pL b → pg a) → Sim pL pg (.ret b) (.ret a)
  | model {L : Game β} {M : Type} {k : M → Game α} (m : M) : Sim pL pg L (k m) → Sim pL pg L (.send M k)
  | live {M : Type} {k : M → Game β} {g : Game α} : (∀ m, Sim pL pg (k m) g) → Sim pL pg (.send M k) g
  | coin {C C' : Type} {iF : Fintype C} {iN : Nonempty C} {iF' : Fintype C'} {iN' : Nonempty C'}
      {k : C → Game β} {k' : C' → Game α} (e : C' ≃ C) :
      (∀ c', Sim pL pg (k (e c')) (k' c')) → Sim pL pg (@Game.coin β C iF iN k) (@Game.coin α C' iF' iN' k')
  | dead {L : Game β} {g : Game α} : (∀ s : Strategy L, prob pL L s = 0) → Strategy g → Sim pL pg L g

namespace Sim

variable {pL : β → Prop} {pg : α → Prop}

/-- The model strategy a simulation plays against the live strategy `s`. -/
def strategy : {L : Game β} → {g : Game α} → Sim pL pg L g → Strategy L → Strategy g
  | _, _, .ret _, _ => PUnit.unit
  | _, _, .model m d, s => ⟨m, d.strategy s⟩
  | _, _, .live f, s => (f s.1).strategy s.2
  | _, _, .coin e f, s => fun c' => (f c').strategy (s (e c'))
  | _, _, .dead _ σ, _ => σ

/-- **The model accepts at least as often.** Against any live strategy, the live verdict holds with probability at most
that of the model's verdict against the simulated strategy. -/
theorem prob_le : ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g) (s : Strategy L),
    prob pL L s ≤ prob pg g (d.strategy s)
  | _, _, .ret h, _ => by
      classical
      simp only [prob]
      split_ifs with h₁ h₂ <;> simp_all
  | _, _, .model m d, s => by simp only [prob, strategy]; exact prob_le d s
  | _, _, .live f, s => by simp only [prob, strategy]; exact prob_le (f s.1) s.2
  | _, _, @coin _ _ _ _ C C' iF iN iF' iN' k k' e f, s => by
      simp only [prob, strategy]
      rw [← e.sum_comp, @Fintype.card_congr C' C iF' iF e]
      exact Finset.sum_le_sum fun c' _ => mul_le_mul_right (prob_le (f c') (s (e c'))) _
  | _, _, .dead h _, s => by rw [h s]; exact zero_le

end Sim

/-- `Out g b`: `b` is one of `g`'s outcomes, whatever the messages and coins. -/
inductive Out : {β : Type} → Game β → β → Prop
  | ret {β : Type} (b : β) : Out (.ret b) b
  | send {β M : Type} {k : M → Game β} (m : M) {b : β} : Out (k m) b → Out (.send M k) b
  | coin {β C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β} (c : C) {b : β} :
      Out (k c) b → Out (@Game.coin β C iF iN k) b

/-- An interaction none of whose outcomes satisfies `p` satisfies it with probability `0`. -/
theorem prob_eq_zero_of_out {p : β → Prop} : ∀ {L : Game β}, (∀ b, Out L b → ¬ p b) → ∀ s : Strategy L, prob p L s = 0
  | .ret b, h, _ => by classical simp only [prob]; simp [h b (.ret b)]
  | .send _ k, h, s => by simp only [prob]; exact prob_eq_zero_of_out (fun b hb => h b (.send s.1 hb)) s.2
  | @Game.coin _ C _ _ k, h, s => by
      simp only [prob]
      exact Finset.sum_eq_zero fun c _ => by rw [prob_eq_zero_of_out (fun b hb => h b (.coin c hb)) (s c), mul_zero]

/-- An outcome of `g.bind f` is an outcome of `f a`, for an outcome `a` of `g`. -/
theorem Out.of_bind {γ : Type} {f : β → Game γ} : ∀ {g : Game β} {c : γ}, Out (g.bind f) c → ∃ b, Out g b ∧ Out (f b) c
  | .ret b, _, h => ⟨b, .ret b, h⟩
  | .send _ k, _, .send m h => let ⟨b, hb, hc⟩ := Out.of_bind h; ⟨b, .send m hb, hc⟩
  | @Game.coin _ C _ _ k, _, .coin x h => let ⟨b, hb, hc⟩ := Out.of_bind h; ⟨b, .coin x hb, hc⟩

/-- An outcome of `g.map f` is the image of an outcome of `g`. -/
theorem Out.of_map {γ : Type} {f : β → γ} {g : Game β} {c : γ} (h : Out (g.map f) c) : ∃ b, Out g b ∧ c = f b := by
  obtain ⟨b, hb, hc⟩ := Out.of_bind h
  cases hc
  exact ⟨b, hb, rfl⟩

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game

local notation "𝔽" => FlockLevel3.GF128

/-- The round the server records for framed bytes `b` and their `n` coins: the coin count, the bytes' digest `dg b`, the
coins, and the bytes themselves, which the unsalted path's server retains (PROTOCOL.md §6.2, §7.1). -/
def roundOf (dg : ByteArray → ByteArray) (b : ByteArray) (n : ℕ) (c : Fin n → 𝔽) : Round :=
  { n, sha := dg b, coins := Array.ofFn fun i => (c i).val, bytes := some b }

/-- A round the prover submits to the challenger (PROTOCOL.md §7.1): its stream (rep `0` or `1`), the framed bytes,
and the coin count. -/
structure Submit where
  rep : Fin 2
  bytes : ByteArray
  n : ℕ

/-- Append a round to rep `r`'s stream. -/
def addRound (acc : List Round × List Round) (r : Fin 2) (rd : Round) : List Round × List Round :=
  if r = 0 then (acc.1 ++ [rd], acc.2) else (acc.1, acc.2 ++ [rd])

/-- **The challenger's rounds** (PROTOCOL.md §7.1). The prover submits a round on either rep's stream, or stops, at
most `N` times; the server draws the round's `n` coins uniformly once its bytes are fixed, and appends the round to that
stream. The output is the two streams' rounds, in order. -/
def liveRounds (dg : ByteArray → ByteArray) : ℕ → List Round × List Round → Game (List Round × List Round)
  | 0, acc => .ret acc
  | N + 1, acc => .send (Option Submit) fun
    | none => .ret acc
    | some s => .coin (Fin s.n → 𝔽) fun c => liveRounds dg N (addRound acc s.rep (roundOf dg s.bytes s.n c))

/-- A live table session's outcome: the root `Commit` sent, the link points, each rep's stream, and the proof files. -/
structure LiveOut (mPts : ℕ) where
  rootB : ByteArray
  pts : Fin 2 → Fin mPts → 𝔽
  rep0 : List Round
  rep1 : List Round
  proofs : Array ByteArray

/-- **One table's live session** (PROTOCOL.md §5, §7): `Commit` sends the root `root_B`, the server draws the two link
points, the prover runs its rounds on the two reps' streams, then sends the two proof files. -/
def liveTable (dg : ByteArray → ByteArray) (mPts N : ℕ) : Game (LiveOut mPts) :=
  .send ByteArray fun rootB =>
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
  (liveRounds dg N ([], [])).bind fun rs =>
  .send (Array ByteArray) fun proofs => .ret ⟨rootB, pts, rs.1, rs.2, proofs⟩

theorem roundOf_coins (dg : ByteArray → ByteArray) (b : ByteArray) (c : Array F128) :
    (roundOf dg b c.size fun i => fOf c[i.val]!).coins = c := by
  apply Array.ext (by simp [roundOf])
  intro i h₁ h₂
  simp only [roundOf, Array.getElem_ofFn]
  rw [getElem!_pos c i h₂]
  rfl

/-- A live run's streams extend the rounds they had. -/
theorem liveRounds_out (dg : ByteArray → ByteArray) : ∀ (N : ℕ) (acc : List Round × List Round)
    {rs : List Round × List Round}, Out (liveRounds dg N acc) rs → ∃ l₀ l₁, rs = (acc.1 ++ l₀, acc.2 ++ l₁)
  | 0, acc, _, .ret _ => ⟨[], [], by simp⟩
  | N + 1, acc, _, .send none (.ret _) => ⟨[], [], by simp⟩
  | N + 1, acc, _, .send (some s) (.coin c h) => by
      obtain ⟨l₀, l₁, rfl⟩ := liveRounds_out dg N _ h
      unfold addRound
      split_ifs
      · exact ⟨roundOf dg s.bytes s.n c :: l₀, l₁, by simp⟩
      · exact ⟨l₀, roundOf dg s.bytes s.n c :: l₁, by simp⟩

end FlockSoundness.Refine
