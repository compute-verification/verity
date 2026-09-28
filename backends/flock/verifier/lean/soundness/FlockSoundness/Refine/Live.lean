import FlockSoundness.Refine.Tapes
import FlockSoundness.Game.Interleave

/-!
# The live session, and simulating it in the model

The transfer (R11) compares two interactions. The **live** one is the coin server's session as the executable sees it:
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

theorem Out.of_interleaveWith {g₁ : Game α} {adv : Game β → Game (α × β)} {fin : β → Game (α × β)}
    (hadv : ∀ g₂ c, Out (adv g₂) c → Out g₁ c.1 ∧ Out g₂ c.2) (hfin : ∀ b c, Out (fin b) c → Out g₁ c.1 ∧ c.2 = b) :
    ∀ {g₂ : Game β} {c : α × β}, Out (interleaveWith adv fin g₂) c → Out g₁ c.1 ∧ Out g₂ c.2
  | .ret b, c, h => let ⟨h₁, h₂⟩ := hfin b c h; ⟨h₁, h₂ ▸ .ret b⟩
  | .send _ _, c, .send true h => hadv _ c h
  | .send _ _, _, .send false (.send m h) =>
      let ⟨h₁, h₂⟩ := Out.of_interleaveWith hadv hfin h; ⟨h₁, .send m h₂⟩
  | @Game.coin _ _ _ _ _, c, .send true h => hadv _ c h
  | @Game.coin _ _ _ _ _, _, .send false (.coin x h) =>
      let ⟨h₁, h₂⟩ := Out.of_interleaveWith hadv hfin h; ⟨h₁, .coin x h₂⟩

/-- An outcome of two interactions run concurrently is a pair of their outcomes. -/
theorem Out.of_interleave : ∀ {g₁ : Game α} {g₂ : Game β} {c : α × β}, Out (interleave g₁ g₂) c → Out g₁ c.1 ∧ Out g₂ c.2
  | .ret a, _, _, h => by
      obtain ⟨b, hb, rfl⟩ := Out.of_map h
      exact ⟨.ret a, hb⟩
  | .send M k, _, _, h =>
      Out.of_interleaveWith (adv := fun g₂ => .send M fun m => interleave (k m) g₂)
        (fin := fun b => (Game.send M k).map fun a => (a, b))
        (fun _ _ h => by
          cases h with
          | send m h => exact ⟨.send m (Out.of_interleave h).1, (Out.of_interleave h).2⟩)
        (fun _ _ h => by obtain ⟨a, ha, rfl⟩ := Out.of_map h; exact ⟨ha, rfl⟩) h
  | @Game.coin _ C iF iN k, _, _, h =>
      Out.of_interleaveWith (adv := fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
        (fin := fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))
        (fun _ _ h => by
          cases h with
          | coin x h => exact ⟨.coin x (Out.of_interleave h).1, (Out.of_interleave h).2⟩)
        (fun _ _ h => by obtain ⟨a, ha, rfl⟩ := Out.of_map h; exact ⟨ha, rfl⟩) h

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game

local notation "𝔽" => FlockLevel3.GF128

/-- The round the server records for framed bytes `b` and their `n` coins: the coin count, the bytes' digest `dg b`, the
coins, and the bytes themselves, which the unsalted path's server retains (PROTOCOL.md §6.2, §7.1). -/
def roundOf (dg : ByteArray → ByteArray) (b : ByteArray) (n : ℕ) (c : Fin n → 𝔽) : Round :=
  { n, sha := dg b, coins := Array.ofFn fun i => (c i).val, bytes := some b }

/-- **One stream on the coin server** (PROTOCOL.md §7.1). The prover submits a round's framed bytes and coin count, or
stops, at most `N` times; the server draws the round's coins uniformly once its bytes are fixed. The output is the
stream's rounds, in order. -/
def liveStream (dg : ByteArray → ByteArray) : ℕ → List Round → Game (List Round)
  | 0, acc => .ret acc
  | N + 1, acc => .send (Option (ByteArray × ℕ)) fun
    | none => .ret acc
    | some (b, n) => .coin (Fin n → 𝔽) fun c => liveStream dg N (acc ++ [roundOf dg b n c])

/-- A live table session's outcome: the root `Commit` sent, the link points, each rep's stream, and the proof files. -/
structure LiveOut (mPts : ℕ) where
  rootB : ByteArray
  pts : Fin 2 → Fin mPts → 𝔽
  rep0 : List Round
  rep1 : List Round
  proofs : Array ByteArray

/-- **One table's live session** (PROTOCOL.md §5, §7). `Commit` sends the root `root_B`, then the server draws the two
link points, the prover runs the two reps' streams, then it sends the two proof files. `interleave` lets the prover pick
the stream at every step, as the server does. It also lets a prover leave a submitted round's coin for later, which the
server doesn't: a stronger prover, so a bound here covers the server's. -/
def liveTable (dg : ByteArray → ByteArray) (mPts N : ℕ) : Game (LiveOut mPts) :=
  .send ByteArray fun rootB =>
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
  (interleave (liveStream dg N []) (liveStream dg N [])).bind fun rs =>
  .send (Array ByteArray) fun proofs => .ret ⟨rootB, pts, rs.1, rs.2, proofs⟩

theorem roundOf_coins (dg : ByteArray → ByteArray) (b : ByteArray) (c : Array F128) :
    (roundOf dg b c.size fun i => fOf c[i.val]!).coins = c := by
  apply Array.ext (by simp [roundOf])
  intro i h₁ h₂
  simp only [roundOf, Array.getElem_ofFn]
  rw [getElem!_pos c i h₂]
  rfl

/-- A stream's outcome extends the rounds it had. -/
theorem liveStream_out (dg : ByteArray → ByteArray) : ∀ (N : ℕ) (acc : List Round) {rs : List Round},
    Out (liveStream dg N acc) rs → ∃ l, rs = acc ++ l
  | 0, acc, _, .ret _ => ⟨[], by simp⟩
  | N + 1, acc, _, .send none (.ret _) => ⟨[], by simp⟩
  | N + 1, acc, _, .send (some (b, n)) (.coin c h) =>
      let ⟨l, hl⟩ := liveStream_out dg N _ h; ⟨roundOf dg b n c :: l, by rw [hl]; simp⟩

end FlockSoundness.Refine
