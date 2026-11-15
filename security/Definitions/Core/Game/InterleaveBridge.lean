module

public import Definitions.Core.Game.TwoRun

/-!
# Positions of a play, and the walk to the next stop

A strategy `σ` of a game `g` is simulated on a tape of `g` (`CoinTape`): the simulation holds a position on the tape
and walks `σ`'s play to the next node where one of a family of cuts (`DigestCut`) fixes a message, a stop. There a move
is read off the most likely message at each cut fixed there, and coins are written into the tape at the stop before the
walk goes on.

* `DigestCut.atSend`, `DigestCut.atCoin`, `DigestCut.isHere`, `DigestCut.modeHere`: a cut after the prover's message or
  after a coin, whether it fixes its message at a node, and the most likely message it fixes there (`revealMode`).
* `Pos`: a node of `g` with `σ`'s strategy there, the cuts there, the tape from there and the tapes of the whole play
  through the node (`Pos.back`). `Pos.root` is the play's start, and `Pos.tape` the tape of the whole play.
* `Pos.toStop` (`Pos.walk`): along the tape, through the prover's messages and the tape's coins, to the first node at
  which a cut is fixed, or to the play's end. `Pos.consume` marks the cuts fixed at a stop as past, `Pos.rewrite`
  rewrites the tape from the position, and `Pos.modes` is the most likely message of each cut fixed at the position.

What the walk keeps and resamples is proved in `Proofs.Core.Game.InterleaveBridge`.
-/

@[expose] public section

namespace Verity.Game

variable {α V ι : Type}

namespace DigestCut

/-- **The cut after the prover sends `m`**; a cut fixed at the node is past. -/
def atSend {M : Type} {k : M → Game α} : DigestCut V (.send M k) → (m : M) → DigestCut V (k m)
  | .skip, _ => .skip
  | .here _, _ => .skip
  | .send K, m => K m

/-- **The cut after the coin `c`**; a cut fixed at the node is past. -/
def atCoin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} :
    DigestCut V (@Game.coin _ C iF iN k) → (c : C) → DigestCut V (k c)
  | .skip, _ => .skip
  | .here _, _ => .skip
  | .coin K, c => K c

/-- Whether the cut fixes its message at this node. -/
def isHere {g : Game α} : DigestCut V g → Bool
  | .here _ => true
  | _ => false

/-- **The most likely message the cut fixes at this node**; `none` where it fixes none here. -/
noncomputable def modeHere [Nonempty V] {g : Game α} : DigestCut V g → Strategy g → Option V
  | .here rv, s => some (revealMode rv g s)
  | _, _ => none

end DigestCut

/-- **A position of a play of `σ` in `g`**: a node `g'` with `σ`'s strategy there (`s'`), the cuts there (`Ks`), the tape
from there (`x`), and the tapes of the whole play through the node (`back`: a tape from the node, put back in place). -/
structure Pos (g : Game α) (σ : Strategy g) (V ι : Type) : Type 1 where
  g' : Game α
  s' : Strategy g'
  Ks : ι → DigestCut V g'
  x : CoinTape g' s'
  back : CoinTape g' s' → CoinTape g σ

namespace Pos

variable {g : Game α} {σ : Strategy g}

/-- The play's start on the tape `x`, at the cuts `Ks`. -/
def root (Ks : ι → DigestCut V g) (x : CoinTape g σ) : Pos g σ V ι := ⟨g, σ, Ks, x, id⟩

/-- The tape of the whole play the position holds. -/
def tape (p : Pos g σ V ι) : CoinTape g σ := p.back p.x

open Classical in
/-- **The walk to the next stop**: from the node `g'` along the tape, through the prover's messages and the tape's
coins, to the first node at which one of the cuts is fixed, or to the play's end. -/
noncomputable def walk : (g' : Game α) → (s' : Strategy g') → (ι → DigestCut V g') → CoinTape g' s' →
    (CoinTape g' s' → CoinTape g σ) → Pos g σ V ι
  | .ret a, s', Ks, x, back => ⟨.ret a, s', Ks, x, back⟩
  | .send M k, s', Ks, x, back =>
    if ∃ i, (Ks i).isHere then ⟨.send M k, s', Ks, x, back⟩
    else walk (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) x back
  | @Game.coin _ C iF iN k, s', Ks, x, back =>
    if ∃ i, (Ks i).isHere then ⟨@Game.coin _ C iF iN k, s', Ks, x, back⟩
    else walk (k x.1) (s' x.1) (fun i => (Ks i).atCoin x.1) (x.2 x.1)
      fun y => back (x.1, Function.update x.2 x.1 y)

/-- **The next stop from the position** (`walk`). -/
noncomputable def toStop (p : Pos g σ V ι) : Pos g σ V ι := walk p.g' p.s' p.Ks p.x p.back

/-- **The cuts fixed at the position, marked past.** -/
def consume (p : Pos g σ V ι) : Pos g σ V ι where
  __ := p
  Ks i := if (p.Ks i).isHere then .skip else p.Ks i

/-- **The tape from the position, rewritten by `f`.** -/
def rewrite (p : Pos g σ V ι) (f : CoinTape p.g' p.s' → CoinTape p.g' p.s') : Pos g σ V ι where
  __ := p
  x := f p.x

/-- **The most likely message each cut fixed at the position fixes there**; `none` at every other cut. -/
noncomputable def modes [Nonempty V] (p : Pos g σ V ι) (i : ι) : Option V := (p.Ks i).modeHere p.s'

end Pos

end Verity.Game
