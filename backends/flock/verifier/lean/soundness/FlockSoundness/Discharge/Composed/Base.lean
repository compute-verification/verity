import FlockSoundness.Discharge.ZkLink
import FlockSoundness.Discharge.ZkExec
import FlockSoundness.Discharge.ZkView
import FlockSoundness.Discharge.ZkBind
import FlockSoundness.Protocol.Discharge.Composed.Base

/-!
# A draw's `--zk` table: its coins and sizes

What `coinedOf` (`Composed/Coined.lean`) and `innerOf` (`Composed/Inner.lean`) both read:

* `CoinsZK`: a table's coin vector, the coins `refused_coins` reads (`β`, `ρ`) and the others (`rest`).
* `RestZK`, `RestOf`: the other coins at a draw, every coin `verify --zk` draws for a rep, in its order: level 0's queries,
  the zerocheck's (`rSkip`, `rOuter`, `z`, the residual rounds' `ρ`), the lincheck's (`α`, `β`, its rounds', `r′`), the ring
  switch's (`rpp` per claim, `γ` per claim), level 0's out-of-domain and lane coins, Ligerito's coins after `T'`, and the inner proof's (`ε`, its
  constraints' `γ`, its queries).
* `pointsOf`: the rank check's claim points, from those coins (`abcClaims`; `point_parts`: a claim's point but its first
  coordinate), rep 0's `ab`, `c`, then rep 1's.
* `hiCount`, `padLenOf`, `padNOf`, `HiOf`, `VLOf`: the hidden salts, the pads layout's sizes and the value lists.
* `eqWord`, `maskWordsOf`: the rank check's weights and the mask slot's words.
* `regionPtOf`, `pubRegOf`: the region claims' points and public ring-switch messages at the link points.
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession Refine

local notation "𝔽" => FlockLevel3.GF128

variable {mPts : ℕ}

/-- Column `i` is in the first mask range's word-aligned span, decidably (`Inner/Holds.lean`'s `MaskCol`). -/
def maskColB (st : Stmt) (i : ℕ) : Bool :=
  match st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => false
  | some r => decide (r.start / 128 ≤ i / 128 ∧ i / 128 < r.start / 128 + 2 ^ (r.slotLog - 7))

/-- **Δ avoids the mask slot**: no pair of Δ (`da`, `db`) has a row or a column in the mask range's span. A condition on
the parsed statement (`avoidsMask`), which `HmRow.check` does not imply: `verify --zk` checks it (`Zk.maskFree`). -/
def deltaAvoidsMask (st : Stmt) : Bool :=
  (st.da.toList ++ st.db.toList).all fun p => !maskColB st p.1 && !maskColB st p.2

/-- Every cell of every region (`regionsOf`, the in-block address of each region bit) lies outside the mask range. -/
def regionCellsOff (st : Stmt) : Bool :=
  (regionsOf st).all fun R => (List.range (2 ^ R.free.length)).all fun i => !maskColB st (R.inBlock i)

/-- Every region differs from every mask-slot word at an in-block word bit it fixes: some bit `i + 7 < k_log`, not free,
whose fixed value is not bit `i` of the word. -/
def regionWordsOff (st : Stmt) : Bool :=
  st.regions.toList.all fun reg => (maskWordsOf st).all fun x =>
    (List.range (st.c.kLog - 7)).any fun i => !reg.free.contains (i + 7) &&
      ((reg.mask >>> (i + 7)) % 2 == 1 && (reg.fixed >>> (i + 7)) % 2 == 1) != x.testBit i

/-- **The regions avoid the mask slot**, by cell and by word: `HmRow.check` does not imply it, since `Circuit.outRegion`
and `HmRow.regions`' tail region sit at an output group's column, which `Net.ofRows` bounds by the unit's slot (D11) but
not away from the mask range; `verify --zk` checks it (`Zk.maskFree`). -/
def regionsAvoidMask (st : Stmt) : Bool := regionCellsOff st && regionWordsOff st

/-- **The statement avoids the mask slot**: its Δ (`deltaAvoidsMask`) and its regions (`regionsAvoidMask`). A condition on
the parsed statement that the honest prover doesn't check: its `Stmt::new` (`backends/flock/live/src/circuit.rs`) refuses
only a column that shares a word with a partial region's bits and isn't forced to zero (`region_word_violation`), and
checks no Δ pair or region cell against the mask span. `verify --zk` refuses a statement that fails it (`Zk.shapeOk`'s
`Zk.maskFree`, D12), so it holds at every `--zk` setup (`FailClosed.avoidsMask_of_shapeOk`). The zero-knowledge
statements here take it as a hypothesis (`hdm`); `FailClosed.ViewZ` and `ViewZHJ` take it from the setup. No proof
derives it from D11's bound on output groups and the layout checks (Δ's forced-zero column `ur.slot 0 + useful` can lie
past the unit's slot); the check makes it a fact of every `--zk` setup. -/
def avoidsMask (st : Stmt) : Bool := deltaAvoidsMask st && regionsAvoidMask st

end FlockSoundness.Discharge.Composed
