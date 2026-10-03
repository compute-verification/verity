import FlockSoundness.Discharge.ZkCoins
import FlockSoundness.CROnly.Hm96Defs
import FlockSoundness.Refine.Live

/-!
# The tree-coin run against the model's live coins: definitions

Under `--zk` the coin server (`live/src/coin_tree.rs`, `coin-tree/hm96-sha512/v2`) answers the prover's `Hello` nonce `ν`
with a fresh hm96 key `K` and the root of a SHA-512 tree whose leaves commit, under `K`, to every (stream, round)'s
block of coins, each with its own salt. Round `r`'s leaf is opened only in the answer to the prover's round-`r` message.
The model (A6) instead draws each round's coins when the round is reached. This file defines what compares the two.

* `Dec Blk C`: how a committed round's block is read as a coin `c : C`: an equivalence `C × Rest ≃ Blk`, the coin and
  the rest of the block (for the coin tree's blocks, the round's first `n` coins and the `block - n` it does not open).
* `ErD g G`: the committed interaction `g` is the model `G` with some of `G`'s coins read off committed rounds through
  `Dec`s, and the others drawn live (`fresh`). It is `ZkCoins.Er` as data, with the decoder an equivalence.
* `Mix G Hl`: a randomized prover of `G`, a finite mixture of strategies `σ ρ` with weights `w ρ` summing to one, each
  with a value `hello ρ : Hl` of its randomness (the `Hello` it drew for itself). `Mix.pr E` is the probability, over
  the mixture and the run, that the output satisfies `E (hello ρ)`.
* `Hm96TwoVector My c δ`: hm96 PROTOCOL.md §3's two-vector case, the leaf pairs `(x + M·y, c y)` and `(x' + M·y, c y)`
  of one uniform salt `y` are within `δ` on every event, for every two digests `x`, `x'`.
* `TreeProver`, `prTree`: the tree-coin run of a one-stage audit whose session is the committed `g`: the prover
  registers, the verifier draws the units, the prover sends its nonce, the server draws the key, the blocks and the
  salts, and the prover, shown the key and every leaf, plays `g`, seeing each round's block and salt at its opening.
* `coinD`, `liveRoundsC`, `liveTableC`: the coin server's live table session (`Refine.liveTable`) on the coin tree, its
  link points on round 0 of the session stream and the `t`-th round of rep `r` on round `t` of rep `r`'s stream.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins
open scoped ENNReal

/-- **A committed round's block read as a coin**: block `e (c, r)` is coin `c` with the rest `r` of the block. -/
structure Dec (Blk C : Type) : Type 1 where
  Rest : Type
  [fin : Fintype Rest]
  e : C × Rest ≃ Blk

attribute [instance] Dec.fin

/-- **`g` is `G` with some coins read off committed blocks** (`ZkCoins.Er`, as data): a committed round `coin i` stands
for a coin `C` of `G` through a `Dec`, a live coin for the same coin. -/
inductive ErD {I Blk α : Type} : TGame I Blk α → Game α → Type 1 where
  | ret (a : α) : ErD (.ret a) (.ret a)
  | send (M : Type) {k : M → TGame I Blk α} {k' : M → Game α} : ((m : M) → ErD (k m) (k' m)) →
      ErD (.send M k) (.send M k')
  | coin (i : I) {C : Type} [Fintype C] [Nonempty C] (d : Dec Blk C) {k : Blk → TGame I Blk α} {k' : C → Game α} :
      ((c : C) → (r : d.Rest) → ErD (k (d.e (c, r))) (k' c)) → ErD (.coin i k) (.coin C k')
  | fresh (C : Type) [Fintype C] [Nonempty C] {k : C → TGame I Blk α} {k' : C → Game α} :
      ((c : C) → ErD (k c) (k' c)) → ErD (.fresh C k) (.coin C k')

/-- **A randomized prover of `G`**: strategies `σ ρ` mixed with weights `w ρ` summing to one; `hello ρ` is what of the
randomness `ρ` an event may read. -/
structure Mix {α : Type} (G : Game α) (Hl : Type) : Type 1 where
  Ρ : Type
  [fin : Fintype Ρ]
  w : Ρ → ℝ≥0∞
  sum_w : ∑ ρ, w ρ = 1
  σ : Ρ → Strategy G
  hello : Ρ → Hl

attribute [instance] Mix.fin

/-- The probability, over the mixture and the run, that the output satisfies `E (hello ρ)`. -/
noncomputable def Mix.pr {α Hl : Type} {G : Game α} (M : Mix G Hl) (E : Hl → α → Prop) : ℝ≥0∞ :=
  ∑ ρ, M.w ρ * prob (E (M.hello ρ)) G (M.σ ρ)

/-- **hm96's two-vector hiding** (hm96 PROTOCOL.md §3): for every two digests `x`, `x'`, the leaf inputs
`(x + M·y, c y)` and `(x' + M·y, c y)` of a uniform salt `y` are within `δ` on every event. -/
def Hm96TwoVector {Sl B C : Type} [Fintype Sl] [Add B] (My : Sl → B) (c : Sl → C) (δ : ℝ≥0∞) : Prop :=
  ∀ (x x' : B) (P : B × C → Prop), prCoin (fun y => P (x + My y, c y)) ≤ prCoin (fun y => P (x' + My y, c y)) + δ

/-! ## The tree-coin run of a one-stage audit -/

section Run

variable {n : ℕ} {E₂ I Blk Cs D Nonce Reg β γ : Type} [Field E₂] [Fintype E₂] [Fintype I] [DecidableEq I] [Fintype Blk]

/-- An hm96-sha512 session key: the `512 + 1536 - 1` key bits of `M(K)`'s Hankel windows. -/
abbrev Key (E₂ : Type) : Type := Fin (512 + 1536 - 1) → E₂

/-- An hm96-sha512 salt: `1536` bits. -/
abbrev Salt (E₂ : Type) : Type := Fin 1536 → E₂

/-- **A prover of the tree-coin audit**: its registration, and at each draw `ω` its nonce `ν` and, after `Hello` (the
key `K` and, more than the root, every leaf), its committed strategy, which sees each round's block and salt at the
round's opening (more than the round's coins, `T_n`, the salt and the path). -/
def TreeProver (L : Audit.Law n) (Reg Nonce : Type) (E₂ : Type) (D : Type) (g : TGame I Blk β) : Type :=
  Reg × ((ω : L.Ω) → Nonce × (Key E₂ → (I → D) → TGame.CStrat (Salt E₂) g))

/-- Round `i`'s keyed hm96 leaf at nonce `ν` and key `κ`: `H_ν(LEAF_PREFIX(κ) ‖ x ⊕ M(κ)·y ‖ c)` with
`x = dig ν i b` the round digest of the block's hash chain and `c = cs ν y`. -/
def treeLeaf (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D) (ν : Nonce) (κ : Key E₂) (i : I) (z : Blk × Salt E₂) : D :=
  leaf (dig ν) (CROnly.hankel (n := 512) (l := 1536) κ) (cs ν) (H ν κ) i z

/-- **The tree-coin run**: the probability that the audit's outcome (the registration, the draw and the session's
output written out by `out`) satisfies `E`, over the draw `ω`, the key `κ` and every round's block and salt `z`, all
uniform, with the leaves at the prover's nonce for `ω`. -/
noncomputable def prTree (L : Audit.Law n) (g : TGame I Blk β) (out : Finset (Fin n) → Reg → β → γ)
    (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D) (τ : TreeProver L Reg Nonce E₂ D g)
    (E : Reg × Finset (Fin n) × γ → Prop) : ℝ≥0∞ :=
  avg fun ω : L.Ω => avg fun κ : Key E₂ => avg fun z : I → Blk × Salt E₂ =>
    TGame.pr (fun o => E (τ.1, L.draw ω, out (L.draw ω) τ.1 o)) g
      ((τ.2 ω).2 κ fun i => treeLeaf dig cs H (τ.2 ω).1 κ i (z i)) z

end Run

/-! ## The live table session on the coin tree -/

section Live

open Refine Flock

local notation "𝔽" => FlockLevel3.GF128

variable {I Blk α : Type}

/-- A coin of type `C` on committed round `i` when `d` reads it off the block, else a live coin. -/
def coinD (i : I) {C : Type} [Fintype C] [Nonempty C] (d : Option (Dec Blk C)) (k : C → TGame I Blk α) :
    TGame I Blk α :=
  match d with
  | some d => .coin i fun b => k (d.e.symm b).1
  | none => .fresh C k

variable (dg : ByteArray → ByteArray) {nS : ℕ} (rl : ℕ) (i0 i1 : Fin nS)
  (dr : (n : ℕ) → Option (Dec Blk (Fin n → 𝔽)))

/-- **The coin server's rounds on the coin tree** (`Refine.liveRounds`): rep `r`'s `t`-th round reads round `t` of rep
`r`'s stream (`i0`, `i1`) while `t < 2^rl` and `dr` reads its `n` coins off a block; any other round's coins are live. -/
def liveRoundsC : ℕ → ℕ → ℕ → List Round × List Round →
    TGame (Fin nS × Fin (2 ^ rl)) Blk (List Round × List Round)
  | 0, _, _, acc => .ret acc
  | N + 1, t₀, t₁, acc => .send (Option Submit) fun
    | none => .ret acc
    | some s =>
      if h : (if s.rep = 0 then t₀ else t₁) < 2 ^ rl then
        coinD ((if s.rep = 0 then i0 else i1), ⟨_, h⟩) (dr s.n) fun c =>
          liveRoundsC N (if s.rep = 0 then t₀ + 1 else t₀) (if s.rep = 0 then t₁ else t₁ + 1)
            (addRound acc s.rep (roundOf dg s.bytes s.n c))
      else .fresh (Fin s.n → 𝔽) fun c =>
          liveRoundsC N (if s.rep = 0 then t₀ + 1 else t₀) (if s.rep = 0 then t₁ else t₁ + 1)
            (addRound acc s.rep (roundOf dg s.bytes s.n c))

/-- **One table's live session on the coin tree** (`Refine.liveTable`): the link points read round 0 of the session
stream `iS` (through `dl`), and the reps' rounds their own streams' rounds. -/
def liveTableC (iS : Fin nS) {mPts : ℕ} (dl : Option (Dec Blk (Fin 2 → Fin mPts → 𝔽))) (N : ℕ) :
    TGame (Fin nS × Fin (2 ^ rl)) Blk (LiveOut mPts) :=
  .send ByteArray fun rootB =>
  coinD (iS, ⟨0, Nat.two_pow_pos rl⟩) dl fun pts =>
  (liveRoundsC dg rl i0 i1 dr N 0 0 ([], [])).bind fun rs =>
  .send (Array ByteArray) fun proofs => .ret ⟨rootB, pts, rs.1, rs.2, proofs⟩

end Live

end FlockSoundness.Discharge.ZkTree
