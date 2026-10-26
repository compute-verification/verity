import Proofs.Flock.Claims.ZK.FinishPiop
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
import Proofs.Flock.Soundness.Refine.Exec

/-!
# The firewall's inner proof is the masked session's where no constraint fails on the pads

`Flock.Firewall.Finish`'s pads commitment and inner proof are copies of `FlockComplete.innerCommitP` and
`FlockComplete.innerP` on its own `PadsZk` and `InnerCommit` (the same fields, read through `PadsZk.toC` and
`InnerCommit.toC`), except that the port's `innerP` never checks the batched constraint on the pads: `repFw` stops first
where a constraint the rep hands the inner proof fails there (`failingOn`).

* `innerCommitP_eq`: the pads commitment is the analysis's.
* **`innerP_eq`**: where the batched constraint holds at every sacrifice and batching coin (`Batched`), the inner proof
  is the analysis's.
* `linAt_eq`: `failingOn`'s value of a constraint at the pads is `evalL`'s.
* **`batched_of_failingOn`**: at pads with `h_ab = h_a·h_b` and `w = x·y`, where no constraint fails (`failingOn`
  empty), each on the pads' length, the batched constraint holds at every coin (`FlockComplete.batched_of_holds`, with the
  triple's three at the prover's openings, `FlockComplete.triple_holds`).
-/

namespace Flock.Firewall.Finish

open FlockComplete FlockLevel3 FlockSoundness.Discharge.Composed.ZerocheckMasked
open FlockSoundness.Discharge.Composed.Zerocheck (eq_of_of dot_eq_foldl of_foldl_dot)

/-- The port's pads as the analysis's: the same five fields. -/
def PadsZk.toC (p : PadsZk) : FlockComplete.PadsZk := ⟨p.h, p.mu, p.padH, p.padMu, p.tauSalt⟩

/-- The port's pads commitment as the analysis's: the same five fields. -/
def InnerCommit.toC (ic : InnerCommit) : FlockComplete.InnerCommit := ⟨ic.ch, ic.cm, ic.salts, ic.layers, ic.root⟩

/-- **The batched constraint `FlockComplete.innerP` checks** on the pads `h`, at the sacrifice coin `eps` and the
batching coins `gammas`, with the triple at the prover's openings. -/
def Batched (lay : Zk.PadLayout) (h : Array F128) (cons : Array (String × Zk.Lin)) (eps : F128)
    (gammas : Array F128) : Bool :=
  let l := Zk.Lin.combo gammas ((cons ++ Zk.triple lay eps (eps * h[lay.fa]! + h[lay.tx]!)
    (h[lay.fb]! + h[lay.ty]!)).map (·.2)) lay.len
  (l.c + dot l.a h).isZero

section

variable {M : Type → Type} [Monad M]

theorem innerCommitP_eq [LawfulMonad M] (hs : Hashes M) (p : PadsZk) (salt : ℕ → ByteArray) :
    InnerCommit.toC <$> innerCommitP hs p salt = FlockComplete.innerCommitP hs p.toC salt := by
  simp only [FlockComplete.innerCommitP, innerCommitP, map_bind, map_pure]
  rfl

variable [MonadExceptOf String M] [MonadLiftT V M]

theorem innerP_eq (io : Asks M) (hs : Hashes M) (lay : Zk.PadLayout) (p : PadsZk) (ic : InnerCommit) (finalC : F128)
    (cons : Array (String × Zk.Lin)) (hb : ∀ eps gammas, Batched lay p.h cons eps gammas = true) :
    innerP io hs lay p ic finalC cons = FlockComplete.innerP io.toC hs lay p.toC ic.toC finalC cons := by
  simp only [Batched] at hb
  simp only [FlockComplete.innerP, innerP, PadsZk.toC, InnerCommit.toC, hb, Bool.not_true, Bool.false_eq_true,
    ↓reduceIte]
  rfl

end

theorem linAt_eq (l : Zk.Lin) (h : Array F128) : linAt l h = evalL l h := by
  apply eq_of_of
  rw [linAt, ← Array.foldl_toList, Array.toList_range, evalL, dot_eq_foldl, of_add, of_foldl_dot, of_foldl_dot,
    ← GF128.zero_eq, zero_add]

/-- **Where no constraint fails on the pads**, each on the pads' length, every one vanishes there. -/
theorem consHold_of_failingOn {cons : Array (String × Zk.Lin)} {h : Array F128}
    (hs : ∀ q ∈ cons, q.2.a.size = h.size) (hf : failingOn cons h = #[]) : ConsHold cons h := by
  unfold failingOn at hf
  rw [Array.map_eq_empty_iff, Array.filter_eq_empty_iff] at hf
  intro q hq
  refine ⟨hs q hq, ?_⟩
  rw [← linAt_eq]
  simpa [bne] using hf q hq

/-- **The batched constraint holds where no constraint fails on the pads**, at every sacrifice and batching coin. -/
theorem batched_of_failingOn (lay : Zk.PadLayout) {h : Array F128} {cons : Array (String × Zk.Lin)}
    (hs : h.size = lay.len) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) (htw : h[lay.tw]! = h[lay.tx]! * h[lay.ty]!)
    (hsz : ∀ q ∈ cons, q.2.a.size = h.size) (hf : failingOn cons h = #[]) (eps : F128) (gammas : Array F128) :
    Batched lay h cons eps gammas = true := by
  have e := batched_of_holds ((consHold_of_failingOn hsz hf).append (triple_holds lay h hs hab htw eps)) gammas
  rw [hs] at e
  unfold evalL at e
  simp only [Batched]
  rw [e]
  rfl

end Flock.Firewall.Finish
