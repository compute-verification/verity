import Proofs.Accounting.Checkpoint

/-!
# RT-LUCK's witnesses against R-CP's card and the tile-adapted reading

Proofs only, no guarantee reads them: each theorem is a counterexample or a corollary that a red-team finding of
note:20261019T1052Z-draft-rt-luck cites. `tile` is the one-round game `WholeGuessOfCheckpoints` concludes on (share `1`
in mode `s`, check `same`), and `luckAt_tile_iff` unfolds `RoundGame.LuckAt` on it.

* `suffixSkip` (RT-LUCK-1, -2): a unit that computes segment 0 and skips only segment 1's product and fold. It keeps
  checkpoint 0 always and guesses checkpoint 1, so it meets the per-unit `½` and misses both the per-checkpoint `½`
  (`CheckpointGuessAt`) and the compounded `¼`.
* `cardImpliesRefused` (RT-LUCK-2): `CheckpointGuessAt` at `½` over `n ≥ 64` checkpoints gives the per-unit `2^-64`.
* `selectiveOracle` (RT-LUCK-3): over the salt and a two-point oracle, a cheater that queried the oracle at a guessed
  salt and is in mode whole only when the salt is its guess. The round's coin has point mass `½` at every value, and
  the cheater's luck in mode whole is `1`, so no relative luck below `1` holds.
* `guessedSeed` (RT-LUCK-7): a function of a seed answer keeps its value at a guessed answer on at least `1/|B|` of the
  answers, so no per-query whole-skip bound below the answer's size holds.
* `adaptiveOrder` (RT-LUCK-4): with the order read from the coins, each tile's commitment reads nothing or only an
  earlier tile on each branch, and tile 1's commitment is its own coin.
-/

namespace Accounting.SecurityProofs.RtLuck

open Accounting.TS Accounting.Assumptions.Checkpoint Accounting.HMuProofs
open Pouw.RelativeSecurity (prob)

/-- The one-round game of `tileGame`'s shape: share `1` in mode `s`, check `same`. -/
noncomputable abbrev tile {C M : Type} [DecidableEq M] (mode : C → M) (s : M) (same : C → Prop) : RoundGame C 1 :=
  (ReceiptGame.round (fun c => if mode c = s then 1 else 0) same same same fun _ => .done).game

theorem luckAt_tile_iff {C M : Type} [Fintype C] [DecidableEq M] (mode : C → M) (s : M) (same : C → Prop) (p : ℝ) :
    RoundGame.LuckAt p (tile mode s same) ↔ prob (fun c => same c ∧ mode c = s) ≤ p * prob (fun c => mode c = s) := by
  have hd : ∀ c, (0 < (if mode c = s then (1 : ℝ) else 0)) ↔ mode c = s := fun c => by
    by_cases hc : mode c = s <;> simp [hc]
  show ((∀ c, 0 ≤ (if mode c = s then (1 : ℝ) else 0) ∧ (if mode c = s then (1 : ℝ) else 0) ≤ 1) ∧
      prob (fun c => same c ∧ 0 < (if mode c = s then (1 : ℝ) else 0)) ≤
        p * prob (fun c => 0 < (if mode c = s then (1 : ℝ) else 0)) ∧ ∀ _ : C, True) ↔ _
  rw [prob_congr fun c => and_congr (Iff.refl (same c)) (hd c), prob_congr hd]
  exact ⟨fun h => h.2.1, fun h => ⟨fun c => by by_cases hc : mode c = s <;> simp [hc], h, fun _ => trivial⟩⟩

/-- The unit's mode: whole, on every coin. -/
def always (_ : Bool × Bool) : Unit := ()

/-- Checkpoint `s` kept, for a unit that computes segment 0 and skips segment 1's product and fold, committing `false`
as checkpoint 1. Set 1's fresh coin `c.2` is checkpoint 1's honest value; checkpoint 0 is computed, so it is kept. -/
def suffixE (s : ℕ) (c : Bool × Bool) : Prop := s = 0 ∨ c.2 = false

/-- **A suffix skip is whole and keeps all but its last checkpoint.** The unit is in mode whole on every coin, its check
passes when checkpoint 1 is kept, and its checks force both checkpoint events (first conjunct). It meets the per-unit
guess bound at `½` (ASK-9's value), but not at `(½)^2`, and it misses `CheckpointGuessAt` at `½` over its two
checkpoints, at checkpoint 0, which it keeps with probability `1`. So a card read per checkpoint over every segment
covers only units skipped in every segment, and a unit skipped in its last segment alone is lucky at `q`, not `q^n`. -/
theorem suffixSkip :
    (∀ c, c.2 = false → ∀ r < 2, suffixE r c) ∧ RoundGame.LuckAt (1 / 2) (tile always () fun c => c.2 = false) ∧
      ¬RoundGame.LuckAt ((1 / 2) ^ 2) (tile always () fun c => c.2 = false) ∧
      ¬CheckpointGuessAt (1 / 2) 2 suffixE fun c => always c = () := by
  have hW : prob (fun c => always c = ()) = 1 := prob_eq_one fun _ => rfl
  have hS : prob (fun c : Bool × Bool => c.2 = false ∧ always c = ()) = 1 / 2 := by
    rw [prob_eq_card,
      show (Finset.univ.filter fun c : Bool × Bool => c.2 = false ∧ always c = ()).card = 2 by decide]
    norm_num
  refine ⟨fun c hc _ _ => Or.inr hc, ?_, ?_, fun h => ?_⟩
  · rw [luckAt_tile_iff, hS, hW]
    norm_num
  · rw [luckAt_tile_iff, hS, hW]
    norm_num
  · have h0 := h 0 (by norm_num)
    have e0 : prob (fun c => suffixE 0 c ∧ (∀ r < 0, suffixE r c) ∧ always c = ()) = 1 :=
      prob_eq_one fun _ => ⟨Or.inl rfl, fun r hr => absurd hr (Nat.not_lt_zero r), rfl⟩
    have e1 : prob (fun c => (∀ r < 0, suffixE r c) ∧ always c = ()) = 1 :=
      prob_eq_one fun _ => ⟨fun r hr => absurd hr (Nat.not_lt_zero r), rfl⟩
    rw [e0, e1] at h0
    norm_num at h0

/-- **The card at `½` over 64 or more checkpoints is the per-unit `2^-64`.** `CheckpointGuessAt (1/2) n` with
`n ≥ 64` (every unit with `K ≥ 2048`, `KS = K/32`) gives the one-round game of `tileGame`'s shape (L) at `2^-64`, the
whole-unit value Compose's `GuessLuckAt` docstring records as proposed and not granted. -/
theorem cardImpliesRefused {C M : Type} [Fintype C] [DecidableEq M] (mode : C → M) (s : M) (right same : C → Prop)
    (E : ℕ → C → Prop) (n : ℕ) (hn : 64 ≤ n) (hsame : ∀ c, same c → ∀ r < n, E r c)
    (h : CheckpointGuessAt (1 / 2) n E fun c => mode c = s) :
    RoundGame.LuckAt (1 / 2 ^ 64)
      (ReceiptGame.round (fun c => if mode c = s then 1 else 0) same right same fun _ => .done).game := by
  have h64 := Accounting.SecurityProofs.Checkpoint.WholeGuessOfCheckpoints C M mode s right same E 64 (1 / 2)
    (fun c hc r hr => hsame c hc r (hr.trans_le hn)) fun r hr => h r (hr.trans_le hn)
  rwa [div_pow, one_pow] at h64

/-- The round's coin over the salt `c.1` and a two-point random oracle `H` (`H false = c.2.1`, `H true = c.2.2`): `H`
at the drawn salt. -/
def oracleCoin (c : Bool × Bool × Bool) : Bool := if c.1 then c.2.2 else c.2.1

/-- **Relative luck is not a theorem of the random oracle.** A cheater guesses the salt `true` and queries the oracle
there before the round, so its view holds `H true = c.2.2`. It commits that answer as the unit's checkpoint, and is in
mode whole exactly when the drawn salt is its guess. Over the salt and the oracle, the round's coin `H(salt)` has point
mass `½` at every value (first conjunct). But in mode whole the committed checkpoint is always the honest one (second
conjunct), so the round's game is (L) at no `q < 1`. Its absolute gain is `Pr[salt = guess]`: `½` here, and
`T_q·2^-512` for a 512-bit salt and `T_q` queries, which an additive term can carry and a relative one can't. -/
theorem selectiveOracle :
    (∀ b : Bool, prob (fun c : Bool × Bool × Bool => oracleCoin c = b) = 1 / 2) ∧
      prob (fun c : Bool × Bool × Bool => c.2.2 = oracleCoin c ∧ c.1 = true) =
        prob (fun c : Bool × Bool × Bool => c.1 = true) ∧
      ∀ q < 1, ¬RoundGame.LuckAt q (tile (fun c : Bool × Bool × Bool => c.1) true fun c => c.2.2 = oracleCoin c) := by
  have hW : prob (fun c : Bool × Bool × Bool => c.1 = true) = 1 / 2 := by
    rw [prob_eq_card, show (Finset.univ.filter fun c : Bool × Bool × Bool => c.1 = true).card = 4 by decide]
    norm_num
  have hS : prob (fun c : Bool × Bool × Bool => c.2.2 = oracleCoin c ∧ c.1 = true) = 1 / 2 := by
    rw [prob_eq_card,
      show (Finset.univ.filter fun c : Bool × Bool × Bool => c.2.2 = oracleCoin c ∧ c.1 = true).card = 4 by decide]
    norm_num
  refine ⟨fun b => ?_, hS.trans hW.symm, fun q hq h => ?_⟩
  · rw [prob_eq_card,
      show (Finset.univ.filter fun c : Bool × Bool × Bool => oracleCoin c = b).card = 4 by cases b <;> decide]
    norm_num
  · rw [luckAt_tile_iff, hS, hW] at h
    linarith

/-- **A guessed seed answer caps whole-skip entropy at the answer's size.** In the seeded model a tile's honest
checkpoints are a function `f` of its seed query's answer: its products are a function of the committed operands, and
its noise and receipt keys are oracle answers at inputs the seed names, which the cheater may read. A cheater that
commits `f b0` for a guessed answer `b0`, reading no answer of the query, keeps them on a share at least `1/|B|` of the
answers. So no per-query bound `e^{-h}` below `1/|B|` holds: `h ≤ ln |B|`, which is `1024·ln 2` for a unit's
`seed_j` and `σ_R`. -/
theorem guessedSeed {B Y : Type} [Fintype B] [DecidableEq Y] (f : B → Y) (b0 : B) :
    1 / (Fintype.card B : ℝ) ≤ prob (fun b => f b = f b0) := by
  rw [prob_eq_card]
  have h1 : 1 ≤ (Finset.univ.filter fun b => f b = f b0).card := Finset.card_pos.2 ⟨b0, by simp⟩
  gcongr
  exact_mod_cast h1

/-- Tile 1's commitment: `false` when its coin `c.1` is `false` (order 1, 2: it reads nothing), and `true` otherwise
(order 2, 1: it may read tile 2's coin, and reads none). -/
def adaptiveC1 (c : Bool × Bool) : Bool := if c.1 then true else false

/-- Tile 2's commitment: tile 1's coin when that is `false` (order 1, 2: an earlier tile's), and `true` otherwise
(order 2, 1: it reads nothing). -/
def adaptiveC2 (c : Bool × Bool) : Bool := if c.1 then true else c.1

/-- **An order read from the coins excludes nothing.** Two tiles skipped whole, with one-bit coins `c.1`, `c.2`, each
tile's honest checkpoint its own coin. The order is (1, 2) when `c.1 = false` and (2, 1) when `c.1 = true`. On each
branch every commitment is a constant or reads an earlier tile's coin (first two conjuncts). Still, tile 1's
commitment is its own coin (third), and both tiles keep their checkpoints with probability `½`, not `¼`. -/
theorem adaptiveOrder :
    (∀ c : Bool × Bool, c.1 = false → adaptiveC1 c = false ∧ adaptiveC2 c = c.1) ∧
      (∀ c : Bool × Bool, c.1 = true → adaptiveC2 c = true ∧ adaptiveC1 c = true) ∧
      (∀ c : Bool × Bool, adaptiveC1 c = c.1) ∧
      prob (fun c : Bool × Bool => adaptiveC1 c = c.1 ∧ adaptiveC2 c = c.2) = 1 / 2 := by
  refine ⟨by decide, by decide, by decide, ?_⟩
  rw [prob_eq_card,
    show (Finset.univ.filter fun c : Bool × Bool => adaptiveC1 c = c.1 ∧ adaptiveC2 c = c.2).card = 2 by decide]
  norm_num

end Accounting.SecurityProofs.RtLuck
