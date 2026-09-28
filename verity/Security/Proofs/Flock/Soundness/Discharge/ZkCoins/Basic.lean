import Proofs.Flock.Soundness.Game.Basic
import Proofs.Flock.Soundness.Assumptions

/-!
# Interactions with committed coins (live/PROTOCOL.md §9, C-Flock ZK)

Under C-Flock ZK the challenger draws every round's block of coins, and a salt for each, at Hello, and sends a
commitment to all of them before the prover's first message: round `i`'s block `b` is chained into a digest `dig i b`,
its leaf is the hm96 leaf `H(dig i b ⊕ M·y ‖ c y)` under the round's salt `y` (`c y = H(sp ‖ y)`), and the leaves go
under a SHA-512 tree. Round `i`'s coins, `T_n`, the salt and the path are opened only when the prover's round-`i`
message arrives.

`TGame I Blk α` is the verifier's side of such an interaction, as Lean defines the protocol: `Game`'s `ret` and `send`;
`coin i k`, which opens committed round `i`'s block `b : Blk` and continues as `k b`; and `fresh C k`, a live coin drawn
when it is reached (a round the commitment does not hold). The blocks and salts `ω : I → Blk × Salt` are uniform and
independent (the server's OS randomness), drawn before anything else.

A committed prover (`CStrat`) sees, before its first message, every round's leaf (`leaf`), which is more than the root
and the paths it receives; at round `i`'s opening it sees the whole block and the salt, more than the `n` coins and
`T_n`. `probC` is the probability that the verifier's output satisfies `p`, over `ω`. `erase` is the same interaction
with every committed round drawn live, as a `Game`, and `Distinct` says that no path opens a round twice (the server
answers each round once).
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game
open scoped ENNReal

/-- **An interaction with committed coins**: `Game` with committed rounds `coin i k` and live coins `fresh C k`. -/
inductive TGame (I Blk α : Type) : Type 1 where
  | ret : α → TGame I Blk α
  | send (M : Type) : (M → TGame I Blk α) → TGame I Blk α
  | coin (i : I) : (Blk → TGame I Blk α) → TGame I Blk α
  | fresh (C : Type) [Fintype C] [Nonempty C] : (C → TGame I Blk α) → TGame I Blk α

namespace TGame

variable {I Blk α β : Type}

/-- **The interaction with every committed round drawn live**: each `coin i k` becomes a uniform block drawn when the
round is reached. -/
def erase [Fintype Blk] [Nonempty Blk] : TGame I Blk α → Game α
  | .ret a => .ret a
  | .send M k => .send M fun m => erase (k m)
  | .coin _ k => .coin Blk fun b => erase (k b)
  | @TGame.fresh _ _ _ C iF iN k => @Game.coin _ C iF iN fun c => erase (k c)

/-- **A committed prover's strategy after the commitment**: a message at every `send` node; at round `i`'s opening it
sees the block and the salt `y : Salt`, and at a live coin the coin. -/
def CStrat (Salt : Type) : TGame I Blk α → Type
  | .ret _ => PUnit
  | .send _ k => (m : _) × CStrat Salt (k m)
  | .coin _ k => (b : Blk) → Salt → CStrat Salt (k b)
  | @TGame.fresh _ _ _ C _ _ k => (c : C) → CStrat Salt (k c)

open Classical in
/-- The probability that the run of `s` on the committed blocks and salts `ω` ends in an output satisfying `p`, over
the live coins. -/
noncomputable def pr {Salt : Type} (p : α → Prop) : (g : TGame I Blk α) → CStrat Salt g → (I → Blk × Salt) → ℝ≥0∞
  | .ret a, _, _ => if p a then 1 else 0
  | .send _ k, s, ω => pr p (k s.1) s.2 ω
  | .coin i k, s, ω => pr p (k (ω i).1) (s (ω i).1 (ω i).2) ω
  | @TGame.fresh _ _ _ C iF _ k, s, ω => @avg C iF fun c => pr p (k c) (s c) ω

/-- No path reads committed round `i`. -/
def NoRead (i : I) : TGame I Blk α → Prop
  | .ret _ => True
  | .send _ k => ∀ m, NoRead i (k m)
  | .coin j k => j ≠ i ∧ ∀ b, NoRead i (k b)
  | @TGame.fresh _ _ _ _ _ _ k => ∀ c, NoRead i (k c)

/-- **No path opens a committed round twice.** -/
def Distinct : TGame I Blk α → Prop
  | .ret _ => True
  | .send _ k => ∀ m, Distinct (k m)
  | .coin i k => ∀ b, NoRead i (k b) ∧ Distinct (k b)
  | @TGame.fresh _ _ _ _ _ _ k => ∀ c, Distinct (k c)

end TGame

/-- **Committed round `i`'s leaf** on block `z.1` and salt `z.2`: the hm96 leaf `H(dig i b + My y, cs y)`. -/
def leaf {I Blk Salt B Cs D : Type} [Add B] (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) (i : I)
    (z : Blk × Salt) : D :=
  H (dig i z.1 + My z.2, cs z.2)

/-- **A committed prover's success probability**: over the uniform blocks and salts `ω`, the prover `τ` sees every
round's leaf and then plays `g`. -/
noncomputable def probC {I Blk Salt B Cs D α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Fintype Salt] [Add B]
    (p : α → Prop) (g : TGame I Blk α) (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D)
    (τ : (I → D) → TGame.CStrat Salt g) : ℝ≥0∞ :=
  avg fun ω : I → Blk × Salt => TGame.pr p g (τ fun i => leaf dig My cs H i (ω i)) ω

end FlockSoundness.Discharge.ZkCoins
