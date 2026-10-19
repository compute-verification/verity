import Proofs.Flock.Claims.Complete.Zk.Ligerito

/-!
# The masked prover's inner proof

Live PROTOCOL.md §5 from the prover's side (upstream's `InnerCommit` and `prove_inner`), as `Zk.verifyInnerWith` reads
it.

* **The pads' commitment** (`innerCommitP`): the rows `h ‖ pad_h` and `μ ‖ pad_μ` encoded (`Zk.codeAt` at the
  `n = 8·nextPow2(s + K_PAD)` columns), leaf `j` the `hm96-sha512/v1` leaf of `SHA-512(C_h[j] ‖ C_μ[j])` under its
  salt, SHA-512 nodes up to the root. The root is bound before the zerocheck's first coin (`zerocheckPZk`).
* **The inner proof** (`innerP`): after `LABEL_INNER` and the masked `final_c`, the sacrifice coin `ε` and the triple's
  openings `ρ = ε h_a + x`, `σ = h_b + y`, the batching coins over every constraint and the triple's, then (if the
  batched constraint holds on the pads, as upstream's `SelfCheck::Stop` checks) `τ = ⟨c, μ⟩` committed, `β` (refused
  at `0`, as upstream does), `Y = (h ‖ pad_h) + β (μ ‖ pad_μ)`, and the opened columns at the query coins with their
  salts and paths.
-/

namespace FlockComplete

open Flock Transcript

/-- **The pads' commitment**: the two rows' codewords, the leaves' salts, the tree's levels and its root. -/
structure InnerCommit where
  ch : Array F128
  cm : Array F128
  salts : Array ByteArray
  layers : Array (Array ByteArray)
  root : ByteArray
  deriving Inhabited

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The pads' commitment** of the rep's pads, leaf `j` salted from `salt j` (fitted to 192 bytes). -/
def innerCommitP (hs : Hashes M) (p : PadsZk) (salt : Nat → ByteArray) : M InnerCommit := do
  let hv := p.h ++ p.padH
  let mv := p.mu ++ p.padMu
  let n := hv.size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE
  let ch := (Array.range n).map (Zk.codeAt hv)
  let cm := (Array.range n).map (Zk.codeAt mv)
  let salts := (Array.range n).map fun j => fitBytes 192 (salt j)
  let leaves ← (Array.range n).mapM fun j => do
    Hm96.Default512.leafWith hs (← hs.sha512 (ch[j]!.toBytes ++ cm[j]!.toBytes)) salts[j]!
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:Nat.log2 n] do
    cur ← (Array.range (cur.size / 2)).mapM fun i => (MerkleOps.hm96With hs).node cur[2 * i]! cur[2 * i + 1]!
    layers := layers.push cur
  return ⟨ch, cm, salts, layers, cur[0]!⟩

/-- **The inner proof** that the pads satisfy `cons` and the triple's constraints, `finalC` the masked `final_c`. -/
def innerP (io : Asks M) (hs : Hashes M) (lay : Zk.PadLayout) (p : PadsZk) (ic : InnerCommit) (finalC : F128)
    (cons : Array (String × Zk.Lin)) : TMW M Zk.InnerProof := do
  let n := ic.ch.size
  liftTM (label Zk.LABEL_INNER)
  liftTM (absorbF finalC)
  let eps ← sampleP io
  let rho := eps * p.h[lay.fa]! + p.h[lay.tx]!
  let sigma := p.h[lay.fb]! + p.h[lay.ty]!
  liftTM (absorbSlice #[rho, sigma])
  let cons := cons ++ Zk.triple lay eps rho sigma
  let gammas ← sampleVecP io cons.size
  let l := Zk.Lin.combo gammas (cons.map (·.2)) lay.len
  if !(l.c + dot l.a p.h).isZero then throw "PROVER-STOPPED: the batched constraint fails (⟨c, h⟩ ≠ t)"
  let tau := dot l.a p.mu
  let tauCommit ← StateT.lift (Zk.tauCommitmentWith hs tau p.tauSalt)
  liftTM (absorbBytes tauCommit)
  let beta ← sampleP io
  if beta.isZero then throw "PROVER-STOPPED: degenerate coin β = 0"
  let hv := p.h ++ p.padH
  let mv := p.mu ++ p.padMu
  let y := (Array.range hv.size).map fun i => hv[i]! + beta * mv[i]!
  liftTM (absorbSlice y)
  let qs := (← sampleVecP io Zk.Q).map fun w => w.lo.toNat % n
  return { root := ic.root, rho, sigma, tauCommit, tauSalt := p.tauSalt, tau, y,
           opened := qs.map fun j => (ic.ch[j]!, ic.cm[j]!), salts := qs.map fun j => ic.salts[j]!,
           paths := qs.map fun j => (Array.range (Nat.log2 n)).map fun k => (ic.layers[k]!)[(j >>> k) ^^^ 1]! }

end FlockComplete
