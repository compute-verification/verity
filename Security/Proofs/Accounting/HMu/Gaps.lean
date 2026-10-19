import Proofs.Accounting.HMu.Luck

/-!
# Why the coins don't give the rest

* `readsCoin`: a cheater whose view reads the fresh bit (it holds σ and has computed a word that shares the coin, or a
  bit leaked) skips only when the bit pins the skipped value, and is lucky on every deficient round, which has
  probability `1/2`. So the view's independence from the fresh bits can't be dropped, and no bound relative to the
  deficiency survives a leak.
* `twoCopies`: a step `s + x` on `Z/2` is injective in the state and in the word, yet a pattern with two positions
  before the checkpoint leaves the checkpoint's state unchanged for every skipped value. So Lemma B′ with two words in
  a unit needs a fibre bound at the start state, which injectivity alone doesn't give.
-/

namespace Accounting.HMuProofs

open Accounting.HMu Pouw.RelativeSecurity Finset

/-- One fresh bit as the skipped value, a view that reads it, a skip exactly when it is `1`, and the receipt for `1`
committed. -/
noncomputable def readsCoinRound : SkipRound Bool Bool ℤ ℤ where
  view := id
  skipped := fun e => if e then 1 else 0
  d := fun o => if o then 1 else 0
  committed := fun _ => 1
  honest := fun _ x => x

theorem readsCoin :
    prob readsCoinRound.lucky = prob (fun ω => 0 < readsCoinRound.d (readsCoinRound.view ω)) ∧
      prob (fun ω => 0 < readsCoinRound.d (readsCoinRound.view ω)) = 1 / 2 := by
  have hd : ∀ ω, 0 < readsCoinRound.d (readsCoinRound.view ω) ↔ ω = true := by
    intro ω; cases ω <;> simp [readsCoinRound]
  refine ⟨prob_congr fun ω => ?_, ?_⟩
  · cases ω <;> simp [SkipRound.lucky, readsCoinRound]
  · rw [prob_congr hd, prob_eq_card, show (univ.filter fun ω : Bool => ω = true).card = 1 by decide]
    norm_num

theorem twoCopies :
    (∀ n x, Function.Injective (fun s : ZMod 2 => (fun _ s x => s + x : ℕ → ZMod 2 → ZMod 2 → ZMod 2) n s x)) ∧
      (∀ n s, Function.Injective ((fun _ s x => s + x : ℕ → ZMod 2 → ZMod 2 → ZMod 2) n s)) ∧
      ¬Function.Injective (fun X : ZMod 2 => chain (fun _ s x => s + x) 0 (persist (fun _ => 0) {0, 1} X) 2) := by
  refine ⟨fun n x a b h => add_right_cancel h, fun n s a b h => add_left_cancel h, fun h => ?_⟩
  have := @h 0 1 (by simp [chain, persist]; decide)
  exact absurd this (by decide)

end Accounting.HMuProofs
