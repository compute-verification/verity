/-
# The schedule a firewalled session's coins follow (live PROTOCOL.md §10.4)

The pure part of the session loop (`Session.lean`), at the statement its setup computed (`Hello.Setup.setup`): which
openings leave and in what order (`planOf`, `reqAt`), and the session `Hello` the coin check reads (`helloOf`). Both are
the setup's own Lean values, never a parse of the `Hello` text.

* `repCoins`: a `--zk` rep's rounds, each round's coin count, as `Zk.verifyRep` squeezes them.
* `planOf`: the openings after the session stream's, rep stream by rep stream in `SessionSpec.repStreams` order, by the
  stream's index there, as the coin layer's schedule places them (`Claims.ZKMal.CoinLayer.slot`).
* `streamsOk`: the coin tree's streams are the session's own (`session`, then the rep streams, all distinct, two a table),
  so that a stream's round, which the coin check counts by the stream's index in the tree (`Coins.Release.open`), is its
  own.
* `helloOf`: what the coin check reads of the session's `Hello`, the tree (`SessionSpec.coinTree`), the session stream's
  coin count (`points · m_pts`) and the nonce; none where the tree's streams are not the session's.

The analysis holds them to the coin layer (`Claims/ZK/SessionLean.lean`).
-/
import Flock.Firewall.Hello

namespace Flock.Firewall.Session

/-- The constraints `Zk.verifyInner` batches, one coin each: the zerocheck's two and the lincheck's (`Zk.replay`), the ring
switch's two, level 0's lane phase's two (`Zk.ligeritoConstraints`) and the triple's three. -/
def INNER_CONS : Nat := 3 + 2 + 2 + 3

/-- The masked replay's rounds (`Zk.replay`): the zerocheck's for a table of `2^m` packed elements, then the lincheck's, for
`2^kLog` a block. -/
def replayCoins (m kLog : Nat) : Array Nat :=
  #[6, m - 13, 1] ++ Array.replicate (m - 6) 1 ++ #[1, 1] ++ Array.replicate (kLog - 6) 1 ++ #[1]

/-- The split after `i` folds, from `s`: each fold of four challenges takes three columns. -/
def splitAt (s : Nat) : Nat → Nat
  | 0 => s
  | i + 1 => splitAt s i - 4 + 1

/-- Level `i`'s queries and their `α`. -/
def queryCoins (sch : Flock.Schedule) (i : Nat) : Array Nat :=
  let q := (sch.levels.getD i default).q
  #[q, Nat.log2 (q - 1) + 1, 1]

/-- Ligerito's rounds with level 0's hiding (`Zk.ligerito`): the OOD sample, the lane rounds and the pair's `ρ`, two intro
samples a level, each level's queries and their `α`, four challenges a fold. -/
def ligeritoCoins (sch : Flock.Schedule) : Array Nat :=
  let R := sch.levels.size - 1
  let s := sch.n - sch.k0 + 1
  #[sch.n, 1] ++ Array.replicate sch.k0 2 ++ #[2, 2, s, 1, s, 1] ++ queryCoins sch 0 ++
    (Array.range R).flatMap fun i => Array.replicate 4 2 ++
      if i + 1 < R then #[splitAt s (i + 1), 1, splitAt s (i + 1), 1] ++ queryCoins sch (i + 1) else queryCoins sch R

/-- **A `--zk` rep's rounds**, as `Zk.verifyRep` squeezes them: each round's coin count, in order, for a table of `2^m`
packed elements, `2^kLog` a block, with `claims` claims. The zerocheck's and the lincheck's (`replayCoins`), each claim's
ring switch and their batching (`ringSwitch`), Ligerito (`ligeritoCoins`), and the inner proof's `ε`, `γ`, `β` and queries
(`Zk.verifyInner`). A squeeze of no coins is no round. -/
def repCoins (m kLog claims : Nat) : V (Array Nat) := do
  let (sch, _) ← Zk.schedule m
  return (replayCoins m kLog ++ Array.replicate claims 7 ++ #[claims] ++ ligeritoCoins sch ++ #[1, INNER_CONS, 1, Zk.Q]).filter
    (· != 0)

/-- A rep's rounds at its table's statement `st` (`repCoins`, with the statement's `2 + points · regions` claims); none for a
table with no `--zk` schedule, as `FlockDeploy.repRounds` counts it, whose proof no verifier accepts. -/
def repPlan (spec : SessionSpec) (st : Stmt) : Array Nat :=
  (repCoins st.m st.c.kLog (2 + spec.points * st.regions.size)).toOption.getD #[]

/-- Rep stream `k`'s name (`SessionSpec.repStreams`), `""` past them. -/
def repName (spec : SessionSpec) (k : Nat) : String := (spec.repStreams[k]?.map (·.2.2)).getD ""

/-- **The openings after the session stream's**, in the order the prover asks for them: every round of rep stream `k`, for
each `k` in `SessionSpec.repStreams` order (table by table, rep 0's before rep 1's), at table `k / 2`'s statement
(`repPlan`), as `CoinLayer.slot` places them. -/
def planOf (spec : SessionSpec) (sts : Array Stmt) : Array (String × Nat) :=
  (Array.range spec.repStreams.size).flatMap fun k => (repPlan spec sts[k / 2]!).map (repName spec k, ·)

/-- **Opening `j`'s stream and coin count**, by the schedule: opening `0` is the session stream's (its count `Hello`'s
link points'), opening `j + 1` the plan's `j`-th. -/
def reqAt (plan : Array (String × Nat)) (links j : Nat) : String × Nat :=
  if j = 0 then (Coins.SESSION, links) else plan.getD (j - 1) ("", 0)

/-- **The coin tree's streams are the session's own**: the `session` stream, then each rep stream in `SessionSpec.repStreams`
order, all distinct, two a table statement (`Zk.coinSpec`'s layout). The coin check counts a stream's rounds by its index in
the tree (`Coins.Release.open`), so only then is each stream's round its own. -/
def streamsOk (spec : SessionSpec) (sts : Array Stmt) (ct : CoinTree.Spec) : Bool :=
  ct.streams.toList == Coins.SESSION :: spec.repStreams.toList.map (·.2.2) && decide ct.streams.toList.Nodup &&
    spec.repStreams.size == 2 * sts.size

/-- **What the coin check reads of the session's `Hello`**, from the setup's values: the tree (`SessionSpec.coinTree`), the
session stream's coin count (the link points, `points · m_pts`) and the nonce; none where the session has no tree or its
streams are not the session's (`streamsOk`). -/
def helloOf (spec : SessionSpec) (sts : Array Stmt) (nonce : ByteArray) : Option Coins.Hello :=
  match spec.coinTree with
  | some ct =>
    if streamsOk spec sts ct then some { spec := ct, links := spec.points * spec.mPts, nonce := nonce.data.toList }
    else none
  | none => none

end Flock.Firewall.Session
