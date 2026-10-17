import Proofs.Flock.Claims.Complete.Zk.InnerMerkle
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
import Proofs.Flock.Claims.Complete.Replay.Steps

/-!
# The masked prover's inner proof in closed form

`innerP_go`: the masked prover's inner proof (`innerP`) in the pure session, at the coins of the replies to its four
sends (`ε`, the batching coins, `β`, the query coins), where its two refusals don't fire: the batched constraint holds on
the pads, and `β ≠ 0`. Its proof is `InZ.proof` and its transcript `InZ.tEnd`.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace InZ

/-- The inner proof's coins, each the reply to its send: `ε`, the batching coins, `β` and the query coins. -/
structure Coins where
  ce : ℕ → F128
  cg : ℕ → F128
  cb : ℕ → F128
  cq : ℕ → F128

def coinsAt (resp : ℕ → Claims.Msg) (L : ℕ) : Coins :=
  ⟨coinOf (resp L), coinOf (resp (L + 1)), coinOf (resp (L + 2)), coinOf (resp (L + 3))⟩

variable (lay : Zk.PadLayout) (p : PadsZk) (finalC : F128) (cons : Array (String × Zk.Lin)) (h : ByteArray → ByteArray)
  (ic : InnerCommit) (cs : Coins)

def rho : F128 := cs.ce 0 * p.h[lay.fa]! + p.h[lay.tx]!

def sigma : F128 := p.h[lay.fb]! + p.h[lay.ty]!

/-- Every constraint and the triple's, at the sacrifice coin. -/
def all : Array (String × Zk.Lin) := cons ++ Zk.triple lay (cs.ce 0) (rho lay p cs) (sigma lay p)

/-- The batched constraint. -/
def lin : Zk.Lin := Zk.Lin.combo ((Array.range (all lay p cons cs).size).map cs.cg) ((all lay p cons cs).map (·.2)) lay.len

def tau : F128 := dot (lin lay p cons cs).a p.mu

/-- `τ`'s commitment through `h`. -/
def tcOf (tau : F128) (salt : ByteArray) : ByteArray := Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes Id) tau salt

def y : Array F128 :=
  (Array.range (p.h ++ p.padH).size).map fun i => (p.h ++ p.padH)[i]! + cs.cb 0 * (p.mu ++ p.padMu)[i]!

def qs : Array ℕ := ((Array.range Zk.Q).map cs.cq).map fun w => w.lo.toNat % ic.ch.size

/-- **The masked prover's inner proof.** -/
def proof : Zk.InnerProof :=
  { root := ic.root, rho := rho lay p cs, sigma := sigma lay p, tauCommit := tcOf h (tau lay p cons cs) p.tauSalt,
    tauSalt := p.tauSalt, tau := tau lay p cons cs, y := y p cs,
    opened := (qs ic cs).map fun j => (ic.ch[j]!, ic.cm[j]!), salts := (qs ic cs).map fun j => ic.salts[j]!,
    paths := (qs ic cs).map fun j => (Array.range (Nat.log2 ic.ch.size)).map fun k => (ic.layers[k]!)[(j >>> k) ^^^ 1]! }

def t₁ (t : Transcript) : Transcript := ⟨t.rounds, t.k, pF finalC (pLabel Zk.LABEL_INNER t.pending)⟩

def t₂ (t : Transcript) : Transcript := afterSq (t₁ finalC t) 1 1 cs.ce

def t₃ (t : Transcript) : Transcript :=
  ⟨(t₂ finalC cs t).rounds, (t₂ finalC cs t).k, pSlice #[rho lay p cs, sigma lay p] (t₂ finalC cs t).pending⟩

def t₄ (t : Transcript) : Transcript := afterSq (t₃ lay p finalC cs t) 2 (all lay p cons cs).size cs.cg

def t₅ (t : Transcript) : Transcript :=
  ⟨(t₄ lay p finalC cons cs t).rounds, (t₄ lay p finalC cons cs t).k,
    pBytes (tcOf h (tau lay p cons cs) p.tauSalt) (t₄ lay p finalC cons cs t).pending⟩

def t₆ (t : Transcript) : Transcript := afterSq (t₅ lay p finalC cons h cs t) 1 1 cs.cb

def t₇ (t : Transcript) : Transcript :=
  ⟨(t₆ lay p finalC cons h cs t).rounds, (t₆ lay p finalC cons h cs t).k,
    pSlice (y p cs) (t₆ lay p finalC cons h cs t).pending⟩

/-- **The masked prover's transcript after its inner proof**: the query coins squeezed. -/
def tEnd (t : Transcript) : Transcript := afterSq (t₇ lay p finalC cons h cs t) 2 Zk.Q cs.cq

end InZ

theorem tauCommitmentWith_ofSha512 (h : ByteArray → ByteArray) (tau : F128) (salt : ByteArray) :
    Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt = pure (InZ.tcOf h tau salt) := rfl

/-- **The masked prover's inner proof, in closed form**, at the coins of the replies to its sends, where the batched
constraint holds on the pads and `β ≠ 0`. -/
theorem innerP_go (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) (t : Transcript) (l : Array Claims.Msg)
    (hz : ((InZ.lin lay p cons (InZ.coinsAt resp l.size)).c +
      dot (InZ.lin lay p cons (InZ.coinsAt resp l.size)).a p.h).isZero = true)
    (hb : (coinOf (resp (l.size + 2)) 0).isZero = false) :
    ∃ l', l'.size = l.size + 4 ∧
      ((innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons).run t).go l =
        (.ok (InZ.proof lay p cons h ic (InZ.coinsAt resp l.size),
          InZ.tEnd lay p finalC cons h (InZ.coinsAt resp l.size) t), l') := by
  unfold innerP
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  dsimp only
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleVecP_go resp (by simp [Zk.triple]) _ _)]
  simp only [InZ.lin, InZ.all, InZ.rho, InZ.sigma, InZ.coinsAt] at hz
  simp only [Array.size_push, hz, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  rw [tauCommitmentWith_ofSha512]
  rw [bind_go (x := StateT.lift (pure _)) rfl]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  simp only [Array.size_push, hb, Bool.false_eq_true, ↓reduceIte]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleVecP_go resp (by decide) _ _)]
  simp only [Array.size_push]
  rw [pure_go]
  exact ⟨_, by simp, rfl⟩

end FlockComplete.Replay
