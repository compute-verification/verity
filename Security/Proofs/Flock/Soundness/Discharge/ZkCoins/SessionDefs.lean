import Proofs.Flock.Soundness.Discharge.ZkCoins.Basic
import Proofs.Flock.Soundness.Discharge.ZkSession.Session

/-!
# The batched `--zk` session on committed coins (live/PROTOCOL.md §9, C-Flock ZK)

`sessionZKC` is `ZkSession.sessionZK` with its coins taken from C-Flock ZK's coin tree: the link points from the session
stream's round 0 (index `none`), and the `t`-th coin draw of table `j`'s rep `r` from round `t` of stream
`circuit/rep r` of table `j` (index `some (j, r, t)`), for `t < R` (C-Flock ZK: `R = 2^8` rounds a stream). A draw of
type `C` reads its value off the round's block through the decoder `dc` gives for `C` (C-Flock ZK: the round's first `n`
coins), which must be `Balanced`, i.e. uniform on uniform blocks. A draw past round `R`, or of a type `dc` has no
decoder for, is a live coin (`fresh`).

The structure follows `sessionZK`: `TGame.interleave` and `TGame.batch` are `Game.interleave` and `Game.batch`, the two
reps of a table run concurrently in any order the prover picks, each on its own stream (`tagRep`).
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Model ZkSession
open scoped ENNReal

namespace TGame

variable {I Blk α β : Type}

/-- Sequencing. -/
def bind : TGame I Blk α → (α → TGame I Blk β) → TGame I Blk β
  | .ret a, f => f a
  | .send M k, f => .send M fun m => bind (k m) f
  | .coin i k, f => .coin i fun b => bind (k b) f
  | @TGame.fresh _ _ _ C iF iN k, f => @TGame.fresh _ _ _ C iF iN fun c => bind (k c) f

/-- Relabel the output. -/
def map (f : α → β) (g : TGame I Blk α) : TGame I Blk β := g.bind fun a => .ret (f a)

/-- `Game.interleaveWith` on committed coins. -/
def interleaveWith (adv₁ : TGame I Blk β → TGame I Blk (α × β)) (fin₁ : β → TGame I Blk (α × β)) :
    TGame I Blk β → TGame I Blk (α × β)
  | .ret b => fin₁ b
  | .send M k => .send Bool fun first =>
      if first then adv₁ (.send M k) else .send M fun m => interleaveWith adv₁ fin₁ (k m)
  | .coin i k => .send Bool fun first =>
      if first then adv₁ (.coin i k) else .coin i fun b => interleaveWith adv₁ fin₁ (k b)
  | @TGame.fresh _ _ _ C iF iN k => .send Bool fun first =>
      if first then adv₁ (@TGame.fresh _ _ _ C iF iN k)
      else @TGame.fresh _ _ _ C iF iN fun c => interleaveWith adv₁ fin₁ (k c)

/-- **`Game.interleave` on committed coins**: two interactions, the prover choosing at every step which advances. -/
def interleave : TGame I Blk α → TGame I Blk β → TGame I Blk (α × β)
  | .ret a => fun g₂ => g₂.map fun b => (a, b)
  | .send M k => interleaveWith (fun g₂ => .send M fun m => interleave (k m) g₂)
      (fun b => (TGame.send M k).map fun a => (a, b))
  | .coin i k => interleaveWith (fun g₂ => .coin i fun c => interleave (k c) g₂)
      (fun b => (TGame.coin i k).map fun a => (a, b))
  | @TGame.fresh _ _ _ C iF iN k => interleaveWith
      (fun g₂ => @TGame.fresh _ _ _ C iF iN fun c => interleave (k c) g₂)
      (fun b => (@TGame.fresh _ _ _ C iF iN k).map fun a => (a, b))

/-- **`Game.batch` on committed coins.** -/
def batch : List (TGame I Blk α) → TGame I Blk (List α)
  | [] => .ret []
  | g :: gs => (g.interleave (batch gs)).map fun x => x.1 :: x.2

end TGame

open Classical in
/-- A decoder `d : Blk → C` is **balanced** when a uniform block decodes to a uniform `C`. -/
def Balanced {Blk C : Type} [Fintype Blk] [Fintype C] (d : Blk → C) : Prop :=
  ∀ c : C, Fintype.card C * (Finset.univ.filter fun b => d b = c).card = Fintype.card Blk

/-- **How a committed round's block is read as a coin draw**: for each coin type, at most one balanced decoder. -/
structure Decoders (Blk : Type) [Fintype Blk] : Type 1 where
  dec : (C : Type) → [Fintype C] → [Nonempty C] → Option (Blk → C)
  balanced : ∀ (C : Type) [Fintype C] [Nonempty C] (d : Blk → C), dec C = some d → Balanced d

variable {I Blk α : Type} [Fintype Blk]

/-- A draw of type `C` from committed round `i` when `dc` decodes `C`, else a live coin. -/
def coinAt (dc : Decoders Blk) (i : I) {C : Type} [Fintype C] [Nonempty C] (k : C → TGame I Blk α) : TGame I Blk α :=
  match dc.dec C with
  | some d => .coin i fun b => k (d b)
  | none => .fresh C k

/-- **One stream's draws on committed rounds**: the `t`-th coin draw of `g` reads round `ix t` while `t < R`. -/
def tagRep (dc : Decoders Blk) {R : ℕ} (ix : Fin R → I) : ℕ → Game α → TGame I Blk α
  | _, .ret a => .ret a
  | t, .send M k => .send M fun m => tagRep dc ix t (k m)
  | t, @Game.coin _ C iF iN k =>
      if h : t < R then
        match @Decoders.dec Blk _ dc C iF iN with
        | some d => .coin (ix ⟨t, h⟩) fun b => tagRep dc ix (t + 1) (k (d b))
        | none => @TGame.fresh _ _ _ C iF iN fun c => tagRep dc ix t (k c)
      else @TGame.fresh _ _ _ C iF iN fun c => tagRep dc ix t (k c)

/-- Committed rounds of a session of `J` tables, `R` rounds a stream: the session stream's round 0 (`none`) and round
`t` of table `j`'s rep `r` (`some (j, r, t)`). -/
abbrev SessIdx (J R : ℕ) : Type := Option (Fin J × Fin 2 × Fin R)

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

/-- Rep `r` of table `T` after its level-0 oracle `C₀` and the link points `pts`, as `tableAfterZK` runs it. -/
noncomputable def repOf (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) (r : ℕ) :
    Game Bool :=
  repZK T.dom T.np T.nP T.q T.pos T.tr
    (repMaskedZK A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r C₀ (extraClaims T.S T.ptLocal mPts pts))

/-- **Table `j` after its commitment, on committed coins**: `TabZK.after` with rep `r` on stream `(j, r)`. -/
noncomputable def afterC (dc : Decoders Blk) (R : ℕ) {J : ℕ} (j : Fin J) (A : Arith F K) (T : TabZK F mPts)
    (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) : TGame (SessIdx J R) Blk (Bool × Oracle F) :=
  (TGame.interleave (tagRep dc (fun t => some (j, 0, t)) 0 (repOf A T C₀ pts 0))
    (tagRep dc (fun t => some (j, 1, t)) 0 (repOf A T C₀ pts 1))).map fun a => (a.1 && a.2, C₀)

/-- **A batched `--zk` session on committed coins**: `sessionZK` with the link points from the session stream's round
0 and every table's reps on their own streams. -/
noncomputable def sessionZKC (dc : Decoders Blk) (R : ℕ) (A : Arith F K) (ts : List (TabZK F mPts)) :
    TGame (SessIdx ts.length R) Blk (List (Bool × Oracle F)) :=
  .send (Fin ts.length → Oracle F) fun Cs =>
    coinAt dc none fun pts : Fin 2 → Fin mPts → F =>
      TGame.batch (List.ofFn fun i => afterC dc R i A (ts.get i) (Cs i) pts)

end FlockSoundness.Discharge.ZkCoins
