module

public import Definitions.Core.Game
public import Definitions.Flock.Game
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Prod
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset

/-! Interactions with committed coins (`TGame`), the verifier's side of Flock ZK's coin tree. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game
open scoped ENNReal

inductive TGame (I Blk α : Type) : Type 1 where
  | ret : α → TGame I Blk α
  | send (M : Type) : (M → TGame I Blk α) → TGame I Blk α
  | coin (i : I) : (Blk → TGame I Blk α) → TGame I Blk α
  | fresh (C : Type) [Fintype C] [Nonempty C] : (C → TGame I Blk α) → TGame I Blk α

namespace TGame

variable {I Blk α β : Type}

def erase [Fintype Blk] [Nonempty Blk] : TGame I Blk α → Game α
  | .ret a => .ret a
  | .send M k => .send M fun m => erase (k m)
  | .coin _ k => .coin Blk fun b => erase (k b)
  | @TGame.fresh _ _ _ C iF iN k => @Game.coin _ C iF iN fun c => erase (k c)

def CStrat (Salt : Type) : TGame I Blk α → Type
  | .ret _ => PUnit
  | .send _ k => (m : _) × CStrat Salt (k m)
  | .coin _ k => (b : Blk) → Salt → CStrat Salt (k b)
  | @TGame.fresh _ _ _ C _ _ k => (c : C) → CStrat Salt (k c)

open Classical in
noncomputable def pr {Salt : Type} (p : α → Prop) : (g : TGame I Blk α) → CStrat Salt g → (I → Blk × Salt) → ℝ≥0∞
  | .ret a, _, _ => if p a then 1 else 0
  | .send _ k, s, ω => pr p (k s.1) s.2 ω
  | .coin i k, s, ω => pr p (k (ω i).1) (s (ω i).1 (ω i).2) ω
  | @TGame.fresh _ _ _ C iF _ k, s, ω => @avg C iF fun c => pr p (k c) (s c) ω

def NoRead (i : I) : TGame I Blk α → Prop
  | .ret _ => True
  | .send _ k => ∀ m, NoRead i (k m)
  | .coin j k => j ≠ i ∧ ∀ b, NoRead i (k b)
  | @TGame.fresh _ _ _ _ _ _ k => ∀ c, NoRead i (k c)

def Distinct : TGame I Blk α → Prop
  | .ret _ => True
  | .send _ k => ∀ m, Distinct (k m)
  | .coin i k => ∀ b, NoRead i (k b) ∧ Distinct (k b)
  | @TGame.fresh _ _ _ _ _ _ k => ∀ c, Distinct (k c)

end TGame

def leaf {I Blk Salt B Cs D : Type} [Add B] (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) (i : I)
    (z : Blk × Salt) : D :=
  H (dig i z.1 + My z.2, cs z.2)

/-- The prover sees every round's leaf before its first message, more than the root and the paths it receives. -/
noncomputable def probC {I Blk Salt B Cs D α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Fintype Salt] [Add B]
    (p : α → Prop) (g : TGame I Blk α) (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D)
    (τ : (I → D) → TGame.CStrat Salt g) : ℝ≥0∞ :=
  avg fun ω : I → Blk × Salt => TGame.pr p g (τ fun i => leaf dig My cs H i (ω i)) ω

end FlockSoundness.Discharge.ZkCoins
