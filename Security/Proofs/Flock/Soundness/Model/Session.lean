import Proofs.Flock.Soundness.Model.Ligerito
import Proofs.Flock.Soundness.Game.Interleave

/-!
# One rep, and one table's session

A table's session (PROTOCOL.md §5, §16): the prover commits the level-0 table (`Commit`, whose
`root_B` is the table's Merkle cap, §14), the verifier draws the two link points, and the prover
then runs the table's two reps, each on its own stream of live coins, in any interleaving it
likes (S15). The verifier accepts the table only if both reps accept (S18: both on the one
committed root).

The link points are drawn once, before either rep (R5), and give the region claims both reps
open (§16.2). `y` is pinned to zero for these statements (S10) and carries no information.
-/

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The point of a region claim at link point `r` (§16.2): free bits from `r`, the other in-block
bits fixed, the block bits from `r[PT_LOCAL …]`. -/
def regionPoint (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (r : Fin mPts → F) :
    Fin S.m → F :=
  fun j =>
    if hj : j.val < S.kLog then
      match R.free.idxOf? ⟨j.val, hj⟩ with
      | some t => if ht : t < mPts then r ⟨t, ht⟩ else 0
      | none => if R.fixedBits ⟨j.val, hj⟩ then 1 else 0
    else
      let i := ptLocal + (j.val - S.kLog)
      if hi : i < mPts then r ⟨i, hi⟩ else 0

/-- The value of a region claim (§16.2):
`Σ_b eq(r[PT_LOCAL …], b)·Σ_i eq(r[… |free|], i)·bit_i(bytes(region, b))`. -/
def regionValue (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (r : Fin mPts → F) : F :=
  ∑ b : Fin (2 ^ S.nbl), (∏ i : Fin S.nbl,
        let c := ptLocal + i.val
        let rc : F := if hc : c < mPts then r ⟨c, hc⟩ else 0
        if b.val.testBit i then rc else 1 + rc)
    * ∑ i ∈ Finset.range (2 ^ R.free.length), (∏ t : Fin R.free.length,
        let rt : F := if ht : t.val < mPts then r ⟨t.val, ht⟩ else 0
        if i.testBit t then rt else 1 + rt) * ((R.bits b i).val : F)

/-- The claim of region `R` at link point `r`: skip weights `eq(p[0…6], ·)`, `x = p[6 … m]`. -/
def regionClaim (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (r : Fin mPts → F) :
    Claim F S.m :=
  let p := regionPoint S R ptLocal mPts r
  ⟨regionValue S R ptLocal mPts r,
    fun s => ∏ j : Fin 6, (if s.val.testBit j then (if h : j.val < S.m then p ⟨j.val, h⟩ else 0)
      else 1 + (if h : j.val < S.m then p ⟨j.val, h⟩ else 0)),
    fun i => if h : i.val + 6 < S.m then p ⟨i.val + 6, h⟩ else 0⟩

/-- The extra claims (§11, §16.2): region-major, point 0 before point 1. -/
def extraClaims (S : Statement) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → F) : List (Claim F S.m) :=
  S.regions.flatMap fun R => [regionClaim S R ptLocal mPts (pts 0), regionClaim S R ptLocal mPts (pts 1)]

/-- **One rep** (§9–§13) on the committed level-0 table `C₀`, opening the region claims
`extra` alongside `ab` and `c`. -/
def rep (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) : Game Bool := do
  match ← zerocheck A S.m with
  | none => pure false
  | some zc =>
    match ← lincheck A S zc with
    | none => pure false
    | some ab =>
      match ← opening A (by omega) (ab :: cClaim A zc :: extra) with
      | none => pure false
      | some op => ligerito A sch op.T op.b C₀

/-- **One table's session.** Returns whether both reps accepted, and the committed table. -/
def table (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) :
    Game (Bool × Oracle F) := do
  let C₀ ← recv (Oracle F)                               -- Commit: the level-0 table (root_B)
  let pts ← draw (Fin 2 → Fin mPts → F)                  -- the link points
  let extra := extraClaims S ptLocal mPts pts
  let accepts ← interleave (rep A S sch hm C₀ extra) (rep A S sch hm C₀ extra)
  pure (accepts.1 && accepts.2, C₀)

end FlockSoundness.Model
