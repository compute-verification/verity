module

public import Proofs.Flock.Claims.Steps.NextSha
public import Proofs.Flock.Claims.Recursion.NextP
public import Proofs.Flock.Claims.Complete.Json.Record
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

/-!
# The record's proof digests written from the proofs' files (`decisionAtS`)

S17 (`verifyJWithM`) holds each proof file to the digest the record's value lists for its rep stream (`proof_{dk}`).
`decisionAtS` is `decisionAtP` at the value with `proof_{dk}` written from the files' digests, as `coinSubJ` writes the
live coins: one entry per rep stream, in `spec.repStreams`' order, its name and its file's digest (`shaSubJ`), the entry
the honest prover writes (`shaArr_honest`). The digests are the oracle's (`dgWith Claims.Hashes.oc`), the ones S17
computes.

Nothing else reads `proof_{dk}`. In the decision only the verdict's decoder does (`Record.decodeJWith`, into
`RunRecord.proofSha`, which S17 alone reads); custody (`liveOfJ`) reads the link and the streams, the sample check
`unit_draw`. Outside it, check 5 (`RecCoins.Inner.load`) and the inner transcript's checks (`RecVStar.innerCommits`,
`RecVStar.innerLink`) read the inner record's streams, link points and sample. The rec-coins file's `record_sha512` binds
the record's whole text, but only on `--rec-inner-coins`' path, which candidate 3's session refuses
(`RecCoins.checkAtC3`), and what it binds (`Inner.load`'s values) reads no `proof_{dk}`.

* **Plain acceptance carries over** (`decisionAtS_canonical`): run against `H`, at a record value whose `proof_{dk}` is
  already the one the files' digests at `H` write (`shaSubJ` fixes it), the verdict is `decisionAtP`'s.
* **`shapeAt` is the verdict's shape** (`repFileShape_shapeAt`): at the link points `pts`, rep `r` of table `i` decodes
  its file at `shapeAt`'s shape for file `2 i + r` (`Flock.verifyRepWith_split`), and each shaped proof's file decodes
  to the proof (`decode_filesAt`), so the verdict's decode of `filesAt`'s files refuses none.
* **Oracle runs** (`EvOk`): a run against `H` that returns `a`; through binds, `mapM` and the record digest.
-/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-! ## Oracle runs that return -/

/-! ## The substitution -/

/-- At the files' digests through `h`, `shaArr` is the honest record's `proof_{dk}` (`FlockComplete.JsonRT.proofObj`). -/
theorem shaArr_honest (spec : SessionSpec) (h : ByteArray → ByteArray) (bs : Array ByteArray) :
    shaArr spec (bs.map (FlockComplete.dgOf spec h)) =
      .arr ((spec.repStreams.zip bs).map (FlockComplete.JsonRT.proofObj spec h)) := by
  unfold shaArr
  rw [Array.zip_map_right, Array.map_map]
  rfl

/-- **Plain acceptance carries over**: run against `H`, at a record value that the substitution at the files' digests
at `H` fixes, `decisionAtS`'s verdict is `decisionAtP`'s. -/
theorem decisionAtS_canonical (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    {stmt : Claims.Msg} {j : Json} {ps : Array Flock.Proof}
    (hfix : shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
      (FlockComplete.dgOf spec (Claims.hOf H))) j = j) :
    ((decisionAtS s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 =
      ((decisionAtP s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 := by
  dsimp only [decisionAtS]
  rw [evOk_bind_fst (evOk_mapM_array (evOk_dg H spec) (filesAt (tabsOf sts) (pointsOf spec c₁) ps)), hfix]

/-! ## The verdict's shape -/

/-- **`shapeAt` is the verdict's shape**: at the link points `pts`, where table `i`'s `fast100` schedule is `sch` and
its extra claims at `pts` are `extras`, rep `r < 2` of table `i` decodes its file at `shapeAt`'s shape for file
`2 i + r` (`Flock.verifyRepWith_split`). -/
theorem repFileShape_shapeAt {tabs : Array Flock.Setup} {pts : Array (Array F128)} {i r : ℕ} (hr : r < 2)
    {sch : Flock.Schedule} (hs : fast100 tabs[i]!.m = some sch) {extras : Array Flock.Claim}
    (he : tabs[i]!.extra pts = .ok extras) :
    Flock.repFileShape tabs[i]! sch extras.size = shapeAt tabs pts (2 * i + r) := by
  have hk : (2 * i + r) / 2 = i := by omega
  simp only [shapeAt, hk, hs, he, Option.getD_some]

/-- **A shaped proof's file decodes to it**: `filesAt`'s file `k` decodes at `shapeAt` to proof `k`, where that proof is
shaped for it (`FlockVBridge.Family.ProofShaped`). -/
theorem decode_filesAt {tabs : Array Flock.Setup} {pts : Array (Array F128)} {ps : Array Flock.Proof} {k : ℕ}
    (hk : k < ps.size) (hsh : FlockVBridge.Family.ProofShaped (shapeAt tabs pts k) ps[k]) :
    decodeProof (shapeAt tabs pts k) ((filesAt tabs pts ps)[k]'(by simpa only [filesAt, Array.size_mapIdx] using hk)) =
      .ok ps[k] := by
  simp only [filesAt, Array.getElem_mapIdx]
  exact FlockVBridge.Family.decodeProof_encodeProof _ _ hsh

end FlockDeploy
