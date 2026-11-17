module

public import Proofs.Flock.Claims.Recursion.BoundReaders
public import Proofs.Flock.Claims.Recursion.VStarLinks
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

/-! V*'s bound layout off the registered ports, at affine forms (`layoutBDeployed`), at which `HeldB` is false since no part holds round 0. -/

@[expose] public section

namespace Claims.Recursion.VStarB

theorem readForms_eq_readLin (w : Flock.Wit) (fs : List _root_.Flock.HmNets.Lin) :
    readForms w fs = VStar.readLin w fs := rfl

end Claims.Recursion.VStarB

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- The hidden value of column `col` of the slot at in-block column `base`, at instance `n` of a part's one table (block
`n / g`, `Readable.copyCol`'s index for one table, as `rowHidden`). -/
def slotHid (c : Circuit) (n base col : ℕ) : ℕ := n / c.g * 2 ^ c.kLog + (base + col)

/-- The in-block column of port `q`'s hm96 slot at instance `n`: slot `q` of VU `n % g` in the hm96 range. -/
def hmBase (c : Circuit) (q n : ℕ) : ℕ := c.slotCol c.hmNet (n % c.g) q

/-- The hm96 net's `k`-th input port's first column in its slot: `cv`, `pad`, then the salt `y`. -/
def hmIn (c : Circuit) (k : ℕ) : ℕ := ((portCols (c.nets.getD c.hmNet default).2.inGroups)[k]?.map (·.1)).getD 0

/-- The hm96 net's output port's first column in its slot: `b ‖ c`. -/
def hmOutCol (c : Circuit) : ℕ := ((portCols (c.nets.getD c.hmNet default).2.outGroups)[0]?.map (·.1)).getD 0

/-- **Port `q`'s commit string at instance `n`**, `b ‖ c`: its hm96 slot's 1024 output bits, in byte-string order. -/
def hmCommit (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range 1024).map fun t => HmNets.Lin.var (slotHid c n (hmBase c q n) (hmOutCol c + t))

/-- **Port `q`'s salt at instance `n`**: its hm96 slot's 1536 input bits `y`, in byte-string order. -/
def hmSalt (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range 1536).map fun j => HmNets.Lin.var (slotHid c n (hmBase c q n) (hmIn c 2 + j))

/-- **Port `q`'s row digest at instance `n`**, an XOR form: hm96's `b = d ⊕ Φy`, where `Φ[i][j]` is key bit `i + j`
(`HmNets.maskBit`), so bit `i` of `d` is `b`'s bit `i` plus the salt bits `j` at which key bit `i + j` is set. -/
def hmDigest (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range 512).map fun i =>
    ⟨slotHid c n (hmBase c q n) (hmOutCol c + i) ::
      ((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map fun j =>
        slotHid c n (hmBase c q n) (hmIn c 2 + j), false⟩

/-- **Port `q`'s hm96 calls at instance `n`** (port `nm`): the salt's, `SHA-512(p_salt ‖ y) = c`, `c` the commit
string's last 64 bytes; then the row's, `SHA-512(rowIn row) = d`, the row's `sha512/row/v1` prefix, then its bits at
their unit input columns (`rowHidden`). -/
def hmCalls (c : Circuit) (nm : String) (q n : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  [(Hm96.Default512.saltPrefix.data.toList, hmSalt c q n, (hmCommit c q n).drop 512),
    ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
      ((rowHidden c nm n).getD []).map HmNets.Lin.var, hmDigest c q n)]

/-- **What part `s` holds**, off its registered message-round ports, as `partHolds`: for each port under `rec-c<k>` and
each row `rep` it reads, rep `rep`'s `k`-th message round as a round of the rep streams (`msgRound`), the port's name,
its index among the part's circuit's ports, and the instance `rep`. -/
def heldRows (r : XR) (s : ℕ) : List (ℕ × String × ℕ × ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (fun vs => vs[s]?), r.w.parts[s]?.bind partCircuit,
      repSched r.shape with
  | some st, some c, some rs =>
    (RecVStar.ownPorts st).flatMap fun q => match RecVStar.cIndex q.2.1 with
      | some k => q.2.2.toList.filterMap fun row => do
        let rep ← row
        let j ← msgRound rs rep k
        let qi ← c.ports.findIdx? (·.name == q.1)
        pure (j, q.1, qi, rep)
      | none => []
  | _, _, _ => []

/-- Round `j` of the two reps' streams at the verifier's inner shape (rep `j / R`'s round `j % R`, of `R`). -/
def streamRound (x : Msg) (j : ℕ) : Option RecVStar.SchedRound :=
  (readX x).bind fun r => (repSched r.shape).bind fun rs =>
    if j < 2 * rs.size then some (rs.getD (j % rs.size) default) else none

/-- A stream round commits no cap node: no tops and no copy of level 0's top. Its commit strings are then `F_k`'s
alone, one where it absorbs a message and none where it absorbs none. -/
def noTops (sr : RecVStar.SchedRound) : Bool := sr.2.1.isEmpty && sr.2.2.1 == 0

/-- **V\*'s bound layout off the registered ports**, at forms: the deployed plain verifier's inner rounds; round `0`
neither held nor fixed; stream round `j` at round `j + 1`, fixed to the empty message where it carries no commit
string, held where its one string is `F_k`'s by the part whose port reads that rep's row, as that port's hm96 slot's
output `b ‖ c` (`hmCommit`); each held port's two hm96 calls (`hmCalls`); no pieces. -/
def layoutBDeployed : Claims.Recursion.VStarB.LayoutB where
  rounds := layoutOf.rounds
  msgAt x s i := match i with
    | 0 => none
    | j + 1 => (streamRound x j).bind fun sr =>
      if noTops sr && sr.2.2.2 then
        (readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).bind fun c =>
          ((heldRows r s).find? (·.1 == j)).map fun h => [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)]
      else none
  shaAt x s := ((readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).map fun c =>
    (heldRows r s).flatMap fun h => hmCalls c h.2.1 h.2.2.1 h.2.2.2).getD []
  fixedAt x i := match i with
    | 0 => none
    | j + 1 => (streamRound x j).bind fun sr => if noTops sr && !sr.2.2.2 then some [] else none

theorem layoutBDeployed_msgAt_zero (x : Msg) (s : ℕ) : layoutBDeployed.msgAt x s 0 = none := rfl

theorem layoutBDeployed_fixedAt_zero (x : Msg) : layoutBDeployed.fixedAt x 0 = none := rfl

theorem layoutBDeployed_pieceAt (x : Msg) (i : ℕ) : layoutBDeployed.pieceAt x i = [] := rfl

theorem wholeBeforeLast_layoutBDeployed : layoutBDeployed.WholeBeforeLast := fun _ _ _ => rfl

theorem readForms_hmCommit (w : Claims.Flock.Wit) (c : Circuit) (q n : ℕ) :
    Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
      Claims.Recursion.VStar.readAt w ((List.range 1024).map fun t => slotHid c n (hmBase c q n) (hmOutCol c + t)) := by
  have h := Claims.Recursion.VStarB.readForms_pos w
    ((List.range 1024).map fun t => slotHid c n (hmBase c q n) (hmOutCol c + t))
  rw [List.map_map] at h
  exact h

theorem heldB_layoutBDeployed_iff {W : Type} (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W)
    (shape : Claims.Recursion.Form → Prop) :
    Claims.Recursion.VStarB.HeldB D I layoutBDeployed shape ↔
      ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → ¬ shape st.form := by
  constructor
  · intro h x st hst hf
    obtain ⟨hK, hcl, -⟩ := h x st hst hf
    have hpos : 0 < I.V.k (Claims.Recursion.xI D x) := by
      rw [hK]
      exact layoutOf_rounds_pos x
    rcases (hcl 0 hpos).1 with h0 | ⟨s, -, h0⟩ | ⟨h0, -⟩
    · rw [layoutBDeployed_fixedAt_zero] at h0
      simp at h0
    · rw [layoutBDeployed_msgAt_zero] at h0
      simp at h0
    · exact h0 (layoutBDeployed_pieceAt x 0)
  · intro h x st hst hf
    exact absurd hf (h x st hst)

end RecDeploy
