import Definitions.Pouw.RelativeSecurity

/-!
# Per-stage coins with sampled proofs (trusted)

STEER-PROOF's route A over PoUW's rounds (`Pouw.RelativeSecurity.RoundGame`, `Paths`, `prob`): a round runs its stages
in order, and stage `c`'s coin is released after stage `c`'s operands are committed (with whole dot products for units),
so what the cheater does at a stage is a function of the coins of that stage and the earlier ones, never a later one.
After the round's last stage, `s` independent draws each name a unit (a whole dot product) by a uniform ticket, so any
draw law with rational weights, and the round passes when every drawn unit's proof passes.

* `CoinStages`: a round's stages: per stage, the units the cheater skips (does not pay for) and the units whose proof
  passes, as functions of the stage's coin, then the rest of the round, which may depend on it. A unit belongs to one
  stage, `stg u`, counted from the round's last (stage `0` is the last).
* `CoinStages.skipped`, `CoinStages.passes`: a unit skipped, and its proof passing, along the round's coins.
* `CoinStages.LuckAt p`: at every stage, whatever the past, a unit of that stage that is skipped passes with
  probability at most `p` over the stage's coin.
* `StageWindow`: a window of rounds of `M` stages and `s` draws each; `StageWindow.LuckAt p` is `CoinStages.LuckAt p` in
  every round.
* `StageWindow.game`: its ideal game: round `w`'s share is the work weights of its skipped units, its check the drawn
  units' proofs.
* `SampledRound`, `SampledWindow`: a sampled round reduced to what its budget reads, the coins `A` before the draws, a
  share on them and, per draw `D`, whether the draw lands on saved work (`hit`) and whether its check passes (`ok`);
  `SampledWindow.game` is its ideal game and `SampledWindow.LuckAt p` the luck of its draws, `Pr[hit ∧ ok] ≤ p·Pr[hit]`
  over `A × D` in every round. Route A's window is one (`hit`: the drawn unit is skipped); so is a one-σ round whose
  closure fails every steered unit drawn (`hit`: the drawn unit saves work, skipped or steered).
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **A round's stages under per-stage coins**: per stage, the units the cheater skips and the units whose proof
passes, as functions of the stage's coin `C` (the stage's outputs are computed after it is released), then the rest of
the round, which may depend on it. -/
inductive CoinStages (C U : Type) : ℕ → Type
  | done : CoinStages C U 0
  | stage {m : ℕ} (skip ok : C → U → Prop) (next : C → CoinStages C U m) : CoinStages C U (m + 1)

namespace CoinStages

variable {C U : Type}

/-- Unit `u` is skipped along the round's coins: the stage it belongs to (`stg u`, stage `m` the one with `m` stages
after it) skips it. -/
def skipped (stg : U → ℕ) : {m : ℕ} → CoinStages C U m → Paths C m → U → Prop
  | _, .done, _, _ => False
  | _, .stage (m := m) skip _ next, x, u => (stg u = m ∧ skip x.1 u) ∨ skipped stg (next x.1) x.2 u

/-- Unit `u`'s proof passes along the round's coins: its stage's check holds. -/
def passes (stg : U → ℕ) : {m : ℕ} → CoinStages C U m → Paths C m → U → Prop
  | _, .done, _, _ => True
  | _, .stage (m := m) _ ok next, x, u => (stg u = m → ok x.1 u) ∧ passes stg (next x.1) x.2 u

/-- **(L) at `p`, per unit at its stage's coin**: at every stage, whatever the past, a unit of the stage that is
skipped passes with probability at most `p` over the stage's coin, `Pr[skip_u ∧ ok_u] ≤ p·Pr[skip_u]`. -/
def LuckAt [Fintype C] (p : ℝ) (stg : U → ℕ) : {m : ℕ} → CoinStages C U m → Prop
  | _, .done => True
  | _, .stage (m := m) skip ok next =>
      (∀ u, stg u = m → prob (fun c => skip c u ∧ ok c u) ≤ p * prob (fun c => skip c u)) ∧
        ∀ c, LuckAt p stg (next c)

end CoinStages

/-- **A window of sampled rounds under per-stage coins**: per round, its `M` stages, then the rest of the window, which
may depend on the round's stage coins and its `s` draws (tickets on `D`). -/
inductive StageWindow (C U D : Type) (M s : ℕ) : ℕ → Type
  | done : StageWindow C U D M s 0
  | round {n : ℕ} (T : CoinStages C U M) (next : Paths C M × (Fin s → D) → StageWindow C U D M s n) :
      StageWindow C U D M s (n + 1)

namespace StageWindow

variable {C U D : Type} {M s : ℕ}

/-- (L) at `p` per unit in every round (`CoinStages.LuckAt`). -/
def LuckAt [Fintype C] (p : ℝ) (stg : U → ℕ) : {n : ℕ} → StageWindow C U D M s n → Prop
  | _, .done => True
  | _, .round T next => T.LuckAt p stg ∧ ∀ a, LuckAt p stg (next a)

/-- **The window's ideal game**: round `w`'s share is the work weights `w` of its skipped units, and its check passes
when the proof of every unit its draws name (`tk`) passes. -/
noncomputable def game [Fintype U] (stg : U → ℕ) (tk : D → U) (w : U → ℝ) :
    {n : ℕ} → StageWindow C U D M s n → RoundGame (Paths C M × (Fin s → D)) n
  | _, .done => .done
  | _, .round T next =>
    .round (fun a => ∑ u, (by classical exact if T.skipped stg a.1 u then w u else 0))
      (fun a => ∀ i, T.passes stg a.1 (tk (a.2 i))) fun a => (next a).game stg tk w

end StageWindow

/-- **A sampled round** reduced to what its budget reads: on the coins `A` before the draws, the round's share and, per
draw on `D`, whether it lands on saved work (`hit`) and whether its check passes (`ok`). -/
structure SampledRound (A D : Type) where
  share : A → ℝ
  hit : A → D → Prop
  ok : A → D → Prop

/-- A window of sampled rounds with `s` draws each: per round, its sampled round, then the rest of the window, which may
depend on the round's coins and draws. -/
inductive SampledWindow (A D : Type) (s : ℕ) : ℕ → Type
  | done : SampledWindow A D s 0
  | round {n : ℕ} (R : SampledRound A D) (next : A × (Fin s → D) → SampledWindow A D s n) :
      SampledWindow A D s (n + 1)

namespace SampledWindow

variable {A D : Type} {s : ℕ}

/-- The window's ideal game: round `w`'s share is its sampled round's, and its check passes when every draw's does. -/
def game : {n : ℕ} → SampledWindow A D s n → RoundGame (A × (Fin s → D)) n
  | _, .done => .done
  | _, .round R next => .round (fun a => R.share a.1) (fun a => ∀ i, R.ok a.1 (a.2 i)) fun a => (next a).game

/-- **The luck of the draws at `p`**: in every round, whatever the past, a draw lands on saved work and passes with
probability at most `p` times that of landing on saved work, `Pr[hit ∧ ok] ≤ p·Pr[hit]` over the coins and one draw. -/
def LuckAt [Fintype A] [Fintype D] (p : ℝ) : {n : ℕ} → SampledWindow A D s n → Prop
  | _, .done => True
  | _, .round R next =>
      prob (fun z : A × D => R.hit z.1 z.2 ∧ R.ok z.1 z.2) ≤ p * prob (fun z : A × D => R.hit z.1 z.2) ∧
        ∀ a, LuckAt p (next a)

/-- **The share is priced by the draws at `r`**: in every round the share is at least 0 and at most `r` times the
probability that one draw lands on saved work. -/
def Priced [Fintype D] (r : ℝ) : {n : ℕ} → SampledWindow A D s n → Prop
  | _, .done => True
  | _, .round R next => (∀ a, 0 ≤ R.share a ∧ R.share a ≤ r * prob (R.hit a)) ∧ ∀ a, Priced r (next a)

end SampledWindow

end Accounting.TS
