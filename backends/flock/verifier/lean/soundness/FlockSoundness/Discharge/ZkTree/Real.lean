import FlockSoundness.Discharge.ZkTree.Defs

/-!
# Every prover of the live coin-tree session is a tree-coin prover

The live session on the coin tree (`live/src/coin_tree.rs`, `coin-tree/hm96-sha512/v2`), from the prover's side: it sends
its `Hello` nonce `ν`; the server draws the key `K`, every round's block and every round's salt, and answers with `K`
and the root; then the session `G` runs, and at each committed round the prover, after it sends the round's message,
receives the round's opening (the round's `n` coins, `T_n`, the salt and the path), which it may check against the root.

* `OStrat Op G`: a deterministic real prover's strategy after `Hello`'s answer: `Strategy G` with, at each coin, the
  coin and the opening view `Op` it arrives with.
* `OpenView`: what the opening of committed round `i` shows, as a function of the round, its decoder `d` (which fixes
  `n`) and the round's block and salt. The live view is one: the coins are the block's first `n` words, `T_n` the hash
  chain of the rest of the block under `ν`, and the path a function of `ν`, `i` and the leaves.
* `prR`, `prReal`: the real run. At committed round `i` the server opens block `(z i).1`, the session's coin is the
  block read by `d`, and the prover sees that coin and `op i d (z i)`. A `fresh` node of `g` is a round the server
  refuses: in `liveTableC` these are exactly the rounds outside the committed schedule (a round past `2^rounds_log`, more
  coins than a block holds, link points that do not fit round 0's block), on which `CoinTree::open` errs and the
  session ends refused; `prR` counts them `0`, so `prReal … E` is the probability of `E` on sessions the server does not
  refuse. A refused session is not accepted (flock-verify's S1 rejects an aborted record), so on acceptance events this
  is the real probability.
* `RealProver.toTree`, `realProver_le_tree`: every real prover is a `TreeProver` that does at least as well on every
  event: shown the key and every leaf it computes the root, and shown the round's block and salt it computes the
  opening view; at a refused round, where the real session ends, it plays on. So the tree-coin model's `fresh` coins
  (gap 3) only give its prover more.

A randomized real prover is a mixture of deterministic ones, and `prReal` is linear in the mixture, so the bound holds
for it at each of its components.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins
open scoped ENNReal

/-- **A real prover's strategy after `Hello`'s answer**: a message at every `send` node; at a coin, the coin and the
opening view it arrives with. -/
def OStrat (Op : Type) {α : Type} : Game α → Type
  | .ret _ => PUnit
  | .send M k => (m : M) × OStrat Op (k m)
  | @Game.coin _ C _ _ k => (c : C) → Op → OStrat Op (k c)

/-- **What the opening of a committed round shows**: at round `i`, read by `d`, of block `b` and salt `y`. -/
abbrev OpenView (I Blk Salt Op : Type) : Type 1 := I → {C : Type} → Dec Blk C → Blk × Salt → Op

open Classical in
/-- **The real run on the committed blocks and salts `z`**: the probability that the output satisfies `p` on a session
the server does not refuse. -/
noncomputable def prR {I Blk Salt Op α : Type} (p : α → Prop) (op : OpenView I Blk Salt Op) :
    {g : TGame I Blk α} → {G : Game α} → ErD g G → OStrat Op G → (I → Blk × Salt) → ℝ≥0∞
  | _, _, .ret a, _, _ => if p a then 1 else 0
  | _, _, .send _ f, s, z => prR p op (f s.1) s.2 z
  | _, _, @ErD.coin _ _ _ i _ _ _ d _ _ f, s, z =>
      prR p op (f (d.e.symm (z i).1).1 (d.e.symm (z i).1).2) (s (d.e.symm (z i).1).1 (op i d (z i))) z
  | _, _, @ErD.fresh _ _ _ _ _ _ _ _ _, _, _ => 0

/-- **A real committed prover's strategy as a tree-coin prover's**: shown the block and the salt, it computes the
opening view; at a refused round it plays on with the default view. -/
def toC {I Blk Salt Op α : Type} [Inhabited Op] (op : OpenView I Blk Salt Op) :
    {g : TGame I Blk α} → {G : Game α} → ErD g G → OStrat Op G → TGame.CStrat Salt g
  | _, _, .ret _, _ => PUnit.unit
  | _, _, .send _ f, s => ⟨s.1, toC op (f s.1) s.2⟩
  | _, _, @ErD.coin _ _ _ i _ _ _ d k _ f, s => fun b y =>
      cast (congrArg (fun b' => TGame.CStrat Salt (k b')) (d.e.apply_symm_apply b))
        (toC op (f (d.e.symm b).1 (d.e.symm b).2) (s (d.e.symm b).1 (op i d (b, y))))
  | _, _, @ErD.fresh _ _ _ _ _ _ _ _ f, s => fun c => toC op (f c) (s c default)

/-- **The run is no better than its tree-coin prover's.** -/
theorem prR_le_pr_toC {I Blk Salt Op α : Type} [Inhabited Op] (p : α → Prop) (op : OpenView I Blk Salt Op) :
    ∀ {g : TGame I Blk α} {G : Game α} (h : ErD g G) (s : OStrat Op G) (z : I → Blk × Salt),
      prR p op h s z ≤ TGame.pr p g (toC op h s) z := by
  sorry

section Run

variable {n : ℕ} {E₂ I Blk Cs D Nonce Reg Root Op β γ : Type} [Field E₂] [Fintype E₂] [Fintype I] [DecidableEq I]
  [Fintype Blk]

/-- **A prover of the live coin-tree audit**: its registration, and at each draw `ω` its nonce and, after `Hello`'s
answer (the key and the root), its strategy in the session. -/
def RealProver (L : Audit.Law n) (Reg Nonce E₂ Root Op : Type) (G : Game β) : Type :=
  Reg × ((ω : L.Ω) → Nonce × (Key E₂ → Root → OStrat Op G))

/-- **The live coin-tree run of a one-stage audit**: the probability that the audit's outcome satisfies `E` on a session
the server does not refuse, over the draw `ω`, the key `κ` and every round's block and salt `z`, all uniform; the
prover sees the key and the root `rt ν κ ℓ` of the leaves `ℓ` at its nonce, and each committed round's opening view
`op ν κ ℓ`. -/
noncomputable def prReal (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (out : Finset (Fin n) → Reg → β → γ) (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D) (rt : Nonce → Key E₂ → (I → D) → Root)
    (op : Nonce → Key E₂ → (I → D) → OpenView I Blk (Salt E₂) Op) (ρ : RealProver L Reg Nonce E₂ Root Op G)
    (E : Reg × Finset (Fin n) × γ → Prop) : ℝ≥0∞ :=
  avg fun ω : L.Ω => avg fun κ : Key E₂ => avg fun z : I → Blk × Salt E₂ =>
    prR (fun o => E (ρ.1, L.draw ω, out (L.draw ω) ρ.1 o))
      (op (ρ.2 ω).1 κ fun i => treeLeaf dig cs H (ρ.2 ω).1 κ i (z i)) hgG
      ((ρ.2 ω).2 κ (rt (ρ.2 ω).1 κ fun i => treeLeaf dig cs H (ρ.2 ω).1 κ i (z i))) z

/-- **A real prover as a tree-coin prover**: the same registration and nonces; shown the key and every leaf, the
real strategy at the root of those leaves, each round's opening view computed from the round's block and salt. -/
def RealProver.toTree [Inhabited Op] (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (rt : Nonce → Key E₂ → (I → D) → Root) (op : Nonce → Key E₂ → (I → D) → OpenView I Blk (Salt E₂) Op)
    (ρ : RealProver L Reg Nonce E₂ Root Op G) : TreeProver L Reg Nonce E₂ D g :=
  (ρ.1, fun ω => ((ρ.2 ω).1, fun κ ℓ => toC (op (ρ.2 ω).1 κ ℓ) hgG ((ρ.2 ω).2 κ (rt (ρ.2 ω).1 κ ℓ))))

/-- **Every real prover is a tree-coin prover, on every event**: the live coin-tree run of `ρ` is at most the tree-coin
run of `ρ.toTree`, which registers what `ρ` registers and sends `ρ`'s nonces. -/
theorem realProver_le_tree [Inhabited Op] (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (out : Finset (Fin n) → Reg → β → γ) (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D) (rt : Nonce → Key E₂ → (I → D) → Root)
    (op : Nonce → Key E₂ → (I → D) → OpenView I Blk (Salt E₂) Op) (ρ : RealProver L Reg Nonce E₂ Root Op G)
    (E : Reg × Finset (Fin n) × γ → Prop) :
    prReal L hgG out dig cs H rt op ρ E ≤ prTree L g out dig cs H (ρ.toTree L hgG rt op) E := by
  sorry

/-- **The live coin-tree run, its outcome written with the session's header**: `prReal` with the outcome written by
`outH (ν, κ, root)`, as the coin server's record carries `coin_tree = {key, nonce, root, spec}` and the prover's
`Hello` with its nonce. -/
noncomputable def prRealH (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (outH : Nonce × Key E₂ × Root → Finset (Fin n) → Reg → β → γ) (dig : Nonce → I → Blk → (Fin 512 → E₂))
    (cs : Nonce → Salt E₂ → Cs) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    (rt : Nonce → Key E₂ → (I → D) → Root) (op : Nonce → Key E₂ → (I → D) → OpenView I Blk (Salt E₂) Op)
    (ρ : RealProver L Reg Nonce E₂ Root Op G) (E : Reg × Finset (Fin n) × γ → Prop) : ℝ≥0∞ :=
  avg fun ω : L.Ω => avg fun κ : Key E₂ => avg fun z : I → Blk × Salt E₂ =>
    prR (fun o => E (ρ.1, L.draw ω, outH ((ρ.2 ω).1, κ, rt (ρ.2 ω).1 κ fun i => treeLeaf dig cs H (ρ.2 ω).1 κ i (z i))
        (L.draw ω) ρ.1 o))
      (op (ρ.2 ω).1 κ fun i => treeLeaf dig cs H (ρ.2 ω).1 κ i (z i)) hgG
      ((ρ.2 ω).2 κ (rt (ρ.2 ω).1 κ fun i => treeLeaf dig cs H (ρ.2 ω).1 κ i (z i))) z

/-- **An event blind to the header** has the probability it has with any fixed writer it is blind against. -/
theorem prRealH_eq (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (outH : Nonce × Key E₂ × Root → Finset (Fin n) → Reg → β → γ) (out : Finset (Fin n) → Reg → β → γ)
    (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    (rt : Nonce → Key E₂ → (I → D) → Root) (op : Nonce → Key E₂ → (I → D) → OpenView I Blk (Salt E₂) Op)
    (ρ : RealProver L Reg Nonce E₂ Root Op G) (E : Reg × Finset (Fin n) × γ → Prop)
    (hE : ∀ h S R o, E (R, S, outH h S R o) ↔ E (R, S, out S R o)) :
    prRealH L hgG outH dig cs H rt op ρ E = prReal L hgG out dig cs H rt op ρ E := by
  sorry

end Run

end FlockSoundness.Discharge.ZkTree
