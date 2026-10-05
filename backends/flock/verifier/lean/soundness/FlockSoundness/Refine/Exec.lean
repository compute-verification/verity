import FlockSoundness.Refine.Run
import Flock.Verify
import FlockProofs.Transcript
import Mathlib.Data.List.Range
import FlockSoundness.Protocol.Refine.Exec
import FlockSoundness.Game.Basic

/-!
# Runs of the executable's transcript programs

The executable verifier's phases are programs in `TM = StateT Transcript (Except String)` (PROTOCOL.md §6). A run that
succeeds, `x t = .ok (a, t')`, is taken apart one step at a time: each lemma below peels one step off the front of a
`bind` and says what that step did.

* Labels, absorbs and proof-of-work nonces change only the pending bytes: `t` and `t'` have the same rounds and the same
  round counter (`Keeps`).
* A squeeze of `n ≠ 0` coins consumes the next round, which issued `n` coins, and returns that round's coins.
* A check that passes changes nothing.

`Adv t t' cs` records what a stretch of a run consumed: the same rounds, and the coins of the rounds from `t.k` to
`t'.k` are `cs`, in order (`consumed`). It composes (`Adv.trans`), so a phase's coins are its steps' coins, concatenated.
-/

namespace FlockSoundness.Refine

open Flock Transcript

/-- `t'` follows `t` on the same rounds, and the rounds consumed in between have the coins `cs`. -/
def Adv (t t' : Transcript) (cs : List (Array F128)) : Prop :=
  t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧ consumed t.rounds t.k t'.k = cs

theorem Adv.refl (t : Transcript) : Adv t t [] := ⟨rfl, le_rfl, consumed_self _ _⟩

theorem Adv.of_keeps {t t' : Transcript} (hk : t'.k = t.k) (hr : t'.rounds = t.rounds) : Adv t t' [] :=
  ⟨hr, hk.ge, by rw [hk, consumed_self]⟩

theorem Adv.trans {t₁ t₂ t₃ : Transcript} {cs₁ cs₂ : List (Array F128)} (h₁ : Adv t₁ t₂ cs₁) (h₂ : Adv t₂ t₃ cs₂) :
    Adv t₁ t₃ (cs₁ ++ cs₂) := by
  obtain ⟨r₁, k₁, c₁⟩ := h₁
  obtain ⟨r₂, k₂, c₂⟩ := h₂
  refine ⟨r₂.trans r₁, k₁.trans k₂, ?_⟩
  rw [consumed_append _ k₁ k₂, c₁, ← r₁, c₂]

theorem Adv.step {t t' : Transcript} (hr : t'.rounds = t.rounds) (hk : t'.k = t.k + 1) :
    Adv t t' [(t.rounds[t.k]!).coins] :=
  ⟨hr, by omega, by rw [hk, consumed_succ]⟩

/-! ## Peeling one step -/

variable {α β : Type}

theorem Keeps.adv {t t' : Transcript} (h : Keeps t t') : Adv t t' [] := Adv.of_keeps h.1 h.2

/-- `sampleK` then `f`: round `t.k` issued two coins, and `f` continues with them as one `F256`. -/
theorem sampleK_bind {f : F256 → TM β} {t t' : Transcript} {b : β} (h : (sampleK >>= f) t = .ok (b, t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 2 ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧
        f ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩ t₁ = .ok (b, t') := by
  obtain ⟨c, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨hlt, hn', hc, hr, hk⟩ := sampleK_ok h₁
  exact ⟨hlt, hn', t₁, hr, hk, hc ▸ h₂⟩

/-! ## Loops over a range, every step yielding -/

end FlockSoundness.Refine
