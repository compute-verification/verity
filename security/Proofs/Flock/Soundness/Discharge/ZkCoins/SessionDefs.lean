module

public import Proofs.Flock.Soundness.Discharge.ZkCoins.Basic
public import Proofs.Flock.Soundness.Discharge.ZkSession.Session
import Mathlib.Combinatorics.Matroid.Init

/-! The batched `--zk` session on committed coins (`sessionZKC`), its coins from Flock ZK's coin tree. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Model ZkSession
open scoped ENNReal

namespace TGame

variable {I Blk α β : Type}

def bind : TGame I Blk α → (α → TGame I Blk β) → TGame I Blk β
  | .ret a, f => f a
  | .send M k, f => .send M fun m => bind (k m) f
  | .coin i k, f => .coin i fun b => bind (k b) f
  | @TGame.fresh _ _ _ C iF iN k, f => @TGame.fresh _ _ _ C iF iN fun c => bind (k c) f

def map (f : α → β) (g : TGame I Blk α) : TGame I Blk β := g.bind fun a => .ret (f a)

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

def interleave : TGame I Blk α → TGame I Blk β → TGame I Blk (α × β)
  | .ret a => fun g₂ => g₂.map fun b => (a, b)
  | .send M k => interleaveWith (fun g₂ => .send M fun m => interleave (k m) g₂)
      (fun b => (TGame.send M k).map fun a => (a, b))
  | .coin i k => interleaveWith (fun g₂ => .coin i fun c => interleave (k c) g₂)
      (fun b => (TGame.coin i k).map fun a => (a, b))
  | @TGame.fresh _ _ _ C iF iN k => interleaveWith
      (fun g₂ => @TGame.fresh _ _ _ C iF iN fun c => interleave (k c) g₂)
      (fun b => (@TGame.fresh _ _ _ C iF iN k).map fun a => (a, b))

def batch : List (TGame I Blk α) → TGame I Blk (List α)
  | [] => .ret []
  | g :: gs => (g.interleave (batch gs)).map fun x => x.1 :: x.2

end TGame

open Classical in
def Balanced {Blk C : Type} [Fintype Blk] [Fintype C] (d : Blk → C) : Prop :=
  ∀ c : C, Fintype.card C * (Finset.univ.filter fun b => d b = c).card = Fintype.card Blk

structure Decoders (Blk : Type) [Fintype Blk] : Type 1 where
  dec : (C : Type) → [Fintype C] → [Nonempty C] → Option (Blk → C)
  balanced : ∀ (C : Type) [Fintype C] [Nonempty C] (d : Blk → C), dec C = some d → Balanced d

variable {I Blk α : Type} [Fintype Blk]

def coinAt (dc : Decoders Blk) (i : I) {C : Type} [Fintype C] [Nonempty C] (k : C → TGame I Blk α) : TGame I Blk α :=
  match dc.dec C with
  | some d => .coin i fun b => k (d b)
  | none => .fresh C k

def tagRep (dc : Decoders Blk) {R : ℕ} (ix : Fin R → I) : ℕ → Game α → TGame I Blk α
  | _, .ret a => .ret a
  | t, .send M k => .send M fun m => tagRep dc ix t (k m)
  | t, @Game.coin _ C iF iN k =>
      if h : t < R then
        match @Decoders.dec Blk _ dc C iF iN with
        | some d => .coin (ix ⟨t, h⟩) fun b => tagRep dc ix (t + 1) (k (d b))
        | none => @TGame.fresh _ _ _ C iF iN fun c => tagRep dc ix t (k c)
      else @TGame.fresh _ _ _ C iF iN fun c => tagRep dc ix t (k c)

abbrev SessIdx (J R : ℕ) : Type := Option (Fin J × Fin 2 × Fin R)

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

noncomputable def repOf (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) (r : ℕ) :
    Game Bool :=
  repZK T.dom T.np T.nP T.q T.pos T.tr
    (repMaskedZK A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r C₀ (extraClaims T.S T.ptLocal mPts pts))

noncomputable def afterC (dc : Decoders Blk) (R : ℕ) {J : ℕ} (j : Fin J) (A : Arith F K) (T : TabZK F mPts)
    (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) : TGame (SessIdx J R) Blk (Bool × Oracle F) :=
  (TGame.interleave (tagRep dc (fun t => some (j, 0, t)) 0 (repOf A T C₀ pts 0))
    (tagRep dc (fun t => some (j, 1, t)) 0 (repOf A T C₀ pts 1))).map fun a => (a.1 && a.2, C₀)

noncomputable def sessionZKC (dc : Decoders Blk) (R : ℕ) (A : Arith F K) (ts : List (TabZK F mPts)) :
    TGame (SessIdx ts.length R) Blk (List (Bool × Oracle F)) :=
  .send (Fin ts.length → Oracle F) fun Cs =>
    coinAt dc none fun pts : Fin 2 → Fin mPts → F =>
      TGame.batch (List.ofFn fun i => afterC dc R i A (ts.get i) (Cs i) pts)

end FlockSoundness.Discharge.ZkCoins
