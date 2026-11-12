/-
`Zk.verify` with a table's two reps' lincheck walks made in one pass over each net (`Zk.verifyOnePass`): `Zk.verify` itself,
called on setups equal to the statements' (`Setup.ofCircuitGiven_eq`), so its verdict on every input is `Zk.verify`'s on
`sts.map Setup.ofCircuit` (`verifyOnePass_eq`). A rep's `replay` asks the fold at a point of that rep's coins, `z` and `ρ`;
the record holds both reps' coins before either is replayed, so both points are read from it first (`foldPoint`) and each
net is walked once for the two (`Stmt.walks2`).
-/
module

public import Flock.Zk
public import Flock.FoldGiven

@[expose] public section

namespace Flock

open Lean (Json)

/-- `Setup.ofCircuit` (`ofCircuitGiven_eq`) with the fold reading the walks `ws`. -/
def Setup.ofCircuitGiven (st : Stmt) (ws : Thunk Stmt.Walks) : Setup :=
  { Setup.ofCircuit st with fold := st.circuitFoldGiven ws }

theorem Setup.ofCircuitGiven_eq (st : Stmt) {ws : Thunk Stmt.Walks} (h : st.WalksOk ws.get) :
    Setup.ofCircuitGiven st ws = Setup.ofCircuit st := by
  unfold Setup.ofCircuitGiven
  rw [st.circuitFoldGiven_eq h]
  rfl

namespace Zk

/-- The coins of each round of stream `name` in the record `j`, read where `Record.decode` reads them and with none of
its checks. -/
def repCoins (j : Json) (name : String) : Option (Array (Array F128)) := do
  let streams ← (Record.arr j "streams").toOption
  let s ← streams.find? fun s => (Record.str s "stream").toOption == some name
  let rounds ← (Record.arr s "rounds").toOption
  rounds.mapM fun r => (Record.arr r "coins" >>= Record.f128s).toOption

/-- The point `replay` asks the fold at, from a rep's coins in round order: `z` is the coin after the two vectors
`r_skip` and `r_outer` (the latter no round when it is empty), and `ρ` the coins of the zerocheck rounds after it, as
many as the fold reads. -/
def foldPoint (m kLog : Nat) (coins : Array (Array F128)) : Array F128 × Array F128 :=
  let k := if m - 13 = 0 then 1 else 2
  (skipWeights ((coins.getD k #[]).getD 0 0),
    (Array.range (min (kLog - 6) (m - 6))).map fun j => (coins.getD (k + 1 + j) #[]).getD 0 0)

/-- Table `i`'s walks at its two reps' points, the points read from the record's coins (`foldPoint`); none where the
record does not give both reps' coins. Nothing rests on a point being the one `replay` reaches: `replay` takes every coin
from its own checked round, and asks the fold at the point it reaches (`Stmt.foldedGiven`). -/
def walksOf (spec : SessionSpec) (st : Stmt) (i : Nat) (record : String) : Stmt.Walks :=
  match JsonParse.parse record with
  | .error _ => #[]
  | .ok j =>
    let t := spec.tables.getD i ""
    match repCoins j s!"{t}/rep0", repCoins j s!"{t}/rep1" with
    | some c0, some c1 => st.walks2 (foldPoint st.m st.c.kLog c0) (foldPoint st.m st.c.kLog c1)
    | _, _ => #[]

theorem walksOf_ok (spec : SessionSpec) (st : Stmt) (i : Nat) (record : String) :
    st.WalksOk (walksOf spec st i record) := by
  unfold walksOf
  split
  · exact fun e he => absurd he (by simp)
  · dsimp only
    split
    · exact st.walks2_ok _ _
    · exact fun e he => absurd he (by simp)

/-- `Zk.verify` on the statements `sts` (`verifyOnePass_eq`), each table's two reps' walks made in one pass, when its
first rep's fold asks for them. -/
def verifyOnePass (spec : SessionSpec) (sts : Array Stmt) (record : String) (proofs : Array ByteArray) : V Unit :=
  verify spec (sts.mapIdx fun i st => Setup.ofCircuitGiven st (Thunk.mk fun _ => walksOf spec st i record)) record proofs

/-- The one-pass verdict is `Zk.verify`'s on the statements' setups, for every session, record and proof. -/
theorem verifyOnePass_eq (spec : SessionSpec) (sts : Array Stmt) (record : String) (proofs : Array ByteArray) :
    verifyOnePass spec sts record proofs = verify spec (sts.map Setup.ofCircuit) record proofs := by
  unfold verifyOnePass
  congr 1
  apply Array.ext
  · simp
  · intro i h1 h2
    simp only [Array.getElem_mapIdx, Array.getElem_map]
    exact Setup.ofCircuitGiven_eq _ (walksOf_ok spec _ i record)

end Zk

end Flock
