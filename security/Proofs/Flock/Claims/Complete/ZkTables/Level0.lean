module

public import Proofs.Flock.Claims.Complete.Zk.LigeritoProverPos
public import Proofs.Flock.Claims.Complete.Replay.LigeritoCons
public import Proofs.Flock.Claims.Complete.Replay.Steps
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! Level 0's masked Ligerito: the premises the table-level forms leave to it (`ProverRuns`, `ProverShape`, `Lanes`,
`Verifier`). -/

@[expose] public section

namespace FlockComplete.ZkTables.Level0

open Flock Transcript FlockComplete FlockComplete.Replay FlockSoundness.Discharge.ZkView

variable {wideLeaf : Bool}

def LgShaped (sh : Shape) (lg : LigeritoProof) : Prop :=
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  CapOk sh (lvl 0) lg.initialCap ∧ LevelOk sh (lvl 0) (2 ^ (lvl 0).lanes + Zk.EXTRA_LANES) lg.initialProof ∧
    lg.recursiveCaps.size = R ∧ (∀ i (h : i < lg.recursiveCaps.size), CapOk sh (lvl (i + 1)) lg.recursiveCaps[i]) ∧
    lg.recursiveProofs.size = R - 1 ∧
    (∀ i (h : i < lg.recursiveProofs.size), LevelOk sh (lvl (i + 1)) 16 lg.recursiveProofs[i]) ∧
    lg.finalProof.yr.size = 2 ^ (s.ext + 1) ∧
    LevelOk sh (lvl R) 16 ⟨lg.finalProof.rows, lg.finalProof.siblings, lg.finalProof.salts⟩ ∧
    lg.messages.size = s.k0 + 8 + 7 * (R - 1) ∧ lg.queryNonces.size = R + 1 ∧ lg.oodValues.size = 1 + 2 * R ∧
    lg.claimNonces.size = 3 * R + 2 ∧ lg.consistencyNonces.size = R + 1 ∧ NoncesZero lg

def lanePoint (resp : ℕ → Claims.Msg) (n L : ℕ) : Array F128 := (Array.range n).map (coinOf (resp L))

def laneCh (resp : ℕ → Claims.Msg) (k0 L : ℕ) : List F256 := (List.range k0).map fun j => coinKOf (resp (L + 2 + j))

def laneRho (resp : ℕ → Claims.Msg) (k0 L : ℕ) : F256 × F256 :=
  (coinKOf (resp (L + 2 + k0)), coinKOf (resp (L + 3 + k0)))

def laneW0 (resp : ℕ → Claims.Msg) (n L : ℕ) (b : Array F128) : Array F256 :=
  addScaled (embedF b) (embedF (eqTable (lanePoint resp n L))) (coinOf (resp (L + 1)) 0)

def pairLane (extra : ℕ → ℕ → F128) (e s : ℕ) : Array F128 := (Array.range s).map (extra e)

def blinded (f : Array F256) (rho : F256 × F256) (r0 r1 : Array F128) : Array F256 :=
  (Array.range f.size).map fun c => f[c]! + rho.1.smul r0[c]! + rho.2.smul r1[c]!

structure LaneForm (resp : ℕ → Claims.Msg) (sch : Schedule) (T : F128) (tb b : Array F128) (lay : Zk.PadLayout)
    (hv : Array F128) (rep : ℕ) (extra : ℕ → ℕ → F128) (L : ℕ) (P : LigeritoProof) (zl : Zk.Level0Msgs)
    (rc : Zk.L0Rec) : Prop where
  record : rc =
    { target := T, ood := #[(coinOf (resp (L + 1)) 0, P.oodValues[0]!)],
      challenges := (laneCh resp sch.k0 L).toArray, rho := laneRho resp sch.k0 L }
  ood : P.oodValues[0]! = mleF tb (lanePoint resp sch.n L) + hv[lay.lgOod 0]!
  msgs_size : sch.k0 ≤ P.messages.size
  msgs : ∀ j < sch.k0, P.messages[j]! = maskMsg lay hv j
    (roundPoly (Lg.foldKs ((laneCh resp sch.k0 L).take j) (embedF tb))
      (Lg.foldKs ((laneCh resp sch.k0 L).take j) (laneW0 resp sch.n L b)))
  e0 : zl.e0 = dotK (Lg.foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
      (embedF (pairLane extra (2 * rep) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size)) +
    ⟨hv[lay.lgE 0 0]!, hv[lay.lgE 0 1]!⟩
  e1 : zl.e1 = dotK (Lg.foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
      (embedF (pairLane extra (2 * rep + 1) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size)) +
    ⟨hv[lay.lgE 1 0]!, hv[lay.lgE 1 1]!⟩
  tPrime : zl.tPrime = dotK (Lg.foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
    (blinded (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)) (laneRho resp sch.k0 L)
      (pairLane extra (2 * rep) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size)
      (pairLane extra (2 * rep + 1) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size))

/-! ## The premises

Each is at a `--zk` schedule of either statement version (`Zk.schedule wideLeaf m`). -/

/-- Its one refusal: the replies to sends `L + 2 + k_0` and `L + 3 + k_0` dependent over F (`Zk.rhoIndependent`). -/
def ProverRuns : Prop :=
  ∀ (wideLeaf : Bool) (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray)
    (m : ℕ) (v : Schedule × ℕ) (T : F128) (cm0 : Commit)
    (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (t : Transcript) (l : Array Claims.Msg),
    (∀ x, (h x).size = 64) → 25 ≤ m → m ≤ 35 → Zk.schedule wideLeaf m = .ok v →
    Zk.rhoIndependent (laneRho resp v.1.k0 l.size) = true →
    ∃ res l', ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2
      extra lanePad salt nonce).run t).go l = (.ok res, l')

def ProverShape : Prop :=
  ∀ (wideLeaf : Bool) (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray)
    (m : ℕ) (v : Schedule × ℕ) (sh : Shape) (T : F128)
    (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt0 : ℕ → ByteArray) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (t t' : Transcript) (l l' : Array Claims.Msg) (P : LigeritoProof) (zl : Zk.Level0Msgs) (rc : Zk.L0Rec),
    (∀ x, (h x).size = 64) → 25 ≤ m → m ≤ 35 → Zk.schedule wideLeaf m = .ok v → sh.sched = v.1 →
    sh.hashLen = MerkleScheme.hm96Sha512.digestLen →
    sh.saltLen = (if MerkleScheme.hm96Sha512.saltBytes = 0 then none else some MerkleScheme.hm96Sha512.saltBytes) →
    tb.size = 2 ^ v.1.n → b.size = 2 ^ v.1.n →
    commitPZk (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 (v.1.levels.getD 0 default) v.2 tb extra
      lanePad salt0 = .ok cm0 →
    ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra
      lanePad salt nonce).run t).go l = (.ok ((P, zl, rc), t'), l') →
    LgShaped sh P ∧ zl.ybar.size = v.2

def Lanes : Prop :=
  ∀ (wideLeaf : Bool) (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray)
    (m : ℕ) (v : Schedule × ℕ) (T : F128) (cm0 : Commit)
    (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (t t' : Transcript) (l l' : Array Claims.Msg)
    (P : LigeritoProof) (zl : Zk.Level0Msgs) (rc : Zk.L0Rec),
    25 ≤ m → m ≤ 35 → Zk.schedule wideLeaf m = .ok v → tb.size = 2 ^ v.1.n → b.size = 2 ^ v.1.n →
    ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra
      lanePad salt nonce).run t).go l = (.ok ((P, zl, rc), t'), l') →
    LaneForm resp v.1 T tb b lay hv rep extra l.size P zl rc

def Verifier : Prop :=
  ∀ (wideLeaf : Bool) (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray)
    (m : ℕ) (v : Schedule × ℕ) (T : F128) (cm0 : Commit)
    (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt0 : ℕ → ByteArray) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (basisAt : Array F256 → ℕ → Array F256) (L : ℕ) (t t' : Transcript) (l l' : Array Claims.Msg)
    (P : LigeritoProof) (zl : Zk.Level0Msgs) (rc : Zk.L0Rec) (R : Array Round),
    (∀ x, (h x).size = 64) → 25 ≤ m → m ≤ 35 → Zk.schedule wideLeaf m = .ok v → rep < 2 →
    tb.size = 2 ^ v.1.n → b.size = 2 ^ v.1.n → dot tb b = T →
    commitPZk (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 (v.1.levels.getD 0 default) v.2 tb extra
      lanePad salt0 = .ok cm0 →
    (∀ q e y, q.size + e = v.1.n → y < 2 ^ e →
      FlockSoundness.Refine.gOf (basisAt q e)[y]! =
        FlockSoundness.Model.bHat FlockSoundness.execArith v.1.n (fun w => FlockSoundness.Refine.fOf b[w]!)
          (q.toList.map FlockSoundness.Refine.gOf ++ FlockSoundness.Model.bitsK e y)) →
    (∀ q e, (basisAt q e).size = 2 ^ e) →
    t.k = t.rounds.size → l.size = L + t.k →
    ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra
      lanePad salt nonce).run t).go l = (.ok ((P, zl, rc), t'), l') →
    Agrees R t'.rounds t.k →
    (Zk.ligeritoWith (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 v.1 T cm0.cap P basisAt
      ⟨v.2, rep, zl⟩).run ⟨R, t.k, t.pending⟩ = .ok (rc, ⟨R, t'.k, t'.pending⟩) ∧ t'.k = t'.rounds.size

end FlockComplete.ZkTables.Level0
