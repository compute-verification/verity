import Proofs.Flock.Claims.Recursion.VStarBound
import Proofs.Flock.Claims.Recursion.DeployPremises

/-!
# `KSBound` at the deployment (option (d))

`Claims.Recursion.KSBound` takes the compiled record as a function of the inner verifier its inner view runs and the
bound record (`Claims.Recursion.Bound`). Here both are pinned to the deployment:

* **The compiled record** is `deployRec IV ivScope b` at `IV = IVStar`: its inner view runs `IVStar`
  (`innerView_ksBound`), its inner scope check is the deployed plain parser's, and its readers are V*'s bound layout's
  (`LayoutB.readers`: every round off the parts, `VStarB.HeldB`).
* **The element-level inner Flock `DE`** is the deployed plain record at plain leaves, `deployPlain rel`, with only three
  fields replaced: its verifier's steps (at the deployed plain verifier's rounds, `FlockDeploy.V false`'s `k`), its
  columns and its weight (`deE`). Its parser, scope, tables, code, decoder and numbers are `deployPlain`'s, so the inner
  decoder `KSBound` quantifies over meets the spec where `deployPlain`'s does.
* **The bound record** (`bdDeployed`) answers a round with its coin's bytes as `FlockDeploy.next` does, counts the
  schedule's commit strings (`nAtSched`, off `rec_live.rep_rounds`), and opens at the pinned hm96 key
  (`Hm96.Default512`, as `deployRec`'s). Its rows' digest input, how its last message reads and the element-level
  verifier's steps stay parameters: they are the code side's, which has no element-level verdict yet.

Statements only.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **V\*'s bound layout as the compiled verifier's readers**: the inner messages and each part's calls, off the parts'
witnesses. -/
def _root_.Claims.Recursion.VStarB.LayoutB.readers (L : Claims.Recursion.VStarB.LayoutB) : RecDeploy.Readers :=
  ⟨L.msgs, L.calls⟩

/-! ## The schedule's commit strings -/

/-- A rep's schedule at a table of the session (`rec_live.rep_rounds`, `Flock.RecVStar.repRounds`): at the table's `m`
and `k_log` and the session's claims, `2 + points · regions` as `FlockDeploy.repRounds` counts them; empty off
`fast100`. -/
def repSched (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) : List _root_.Flock.RecVStar.SchedRound :=
  ((_root_.Flock.RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size)).map Array.toList).getD []

/-- A table's level-0 top's nodes, `2^{cap₀}` at `fast100`'s first level (`2^7 = 128` at its 218 queries). -/
def level0Nodes (st : _root_.Flock.Stmt) : ℕ :=
  2 ^ ((_root_.Flock.fast100 st.m).map fun s => (s.levels.getD 0 default).cap).getD 0

/-- **The schedule's commit strings at each inner round before the last** (`Bound.nAt` at the deployment), at an inner
statement the deployed plain verifier sets up. Round 0, the link, carries each table's level-0 top, one string a cap
node. Round `1 + i`, the `i`-th recorded round (each table's two reps in turn, each rep's rounds in order), carries the
salted tops it commits, one a cap node, then one row where it absorbs a message. A rep's copies of level 0's top are
round 0's strings again, not new ones. None at a statement that doesn't set up. -/
def nAtSched (x : Msg) (j : ℕ) : ℕ :=
  match FlockDeploy.setupOf false x with
  | none => 0
  | some (_, spec, sts) =>
    if j = 0 then (sts.toList.map level0Nodes).sum
    else
      match (sts.toList.flatMap fun st => repSched spec st ++ repSched spec st)[j - 1]? with
      | some r => r.2.1.toList.sum + if r.2.2.2 then 1 else 0
      | none => 0

/-! ## The pinned records -/

/-- **The element-level inner Flock, pinned to the deployed plain record**: `deployPlain rel` with only its verifier,
its columns and its weight replaced. The verifier keeps the deployed plain verifier's rounds (`FlockDeploy.V false`'s
`k`); its steps (`nextE`) read element-level messages. -/
noncomputable def deE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  { FlockDeploy.deployPlain rel with V := ⟨(FlockDeploy.V false).k, nextE⟩, cols := colsE, weight := wE }

/-- **The deployed bound record**: `DE` pinned (`deE`), each round before the last answered by its coin's bytes, the
schedule's commit strings (`nAtSched`) and the pinned hm96 key's salts, salt prefix and salt mask. The rows' digest input
(`rowIn`), how the last message reads (`final`), how a round's rows join (`join`) and `γ`'s weight at `IVStar` (`w`)
are the code side's. -/
noncomputable def bdDeployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace where
  DE := deE rel nextE colsE wE
  ans c := c.bytes
  join := join
  nAt := nAtSched
  saltLen := _root_.Flock.Hm96.sha512.saltBytes
  saltPrefix := Claims.toMsg _root_.Flock.Hm96.Default512.saltPrefix
  mask y i := (_root_.Flock.Hm96.Default512.mask ⟨Array.ofFn y⟩).get! i.val
  rowIn := rowIn
  final := final
  weight := w

/-- **The compiled record at an inner verifier, at the deployment**: `deployRec` with the deployed plain parser's
scope check for the inner statement and V*'s bound layout's readers. -/
noncomputable def recBound (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) : Claims.Recursion.Deploy Coin :=
  deployRec IV (FlockDeploy.deployPlain rel).scopeCheck { b with rd := L.readers }

/-- **`KSBound` at the deployment**: at SHA-512, the compiled record `deployRec` whose inner view runs `IVStar` at V*'s
bound layout (`recBound`), the deployed bound record (`bdDeployed`) and the deployed outer `--zk` record
(`deployZkRec`). -/
def KSBoundDeployed (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSBound FlockDeploy.sha512 (recBound rel b L) (bdDeployed rel nextE colsE wE join rowIn final w)
    (deployZkRec relO)

/-- **The private variant at the deployment**, at the same records. -/
def KSPrivBoundDeployed (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSPrivBound FlockDeploy.sha512 (recBound rel b L)
    (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO)

/-- **The compiled record's inner view runs `IVStar`**, at V*'s bound layout's messages and the run's first coins. -/
theorem innerView_ksBound (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128)
      Claims.Flock.PlainTrace) (H : Msg → Msg) (x : Msg) (r : Fin ((recBound rel b L Bd.IVStar).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    (recBound rel b L Bd.IVStar).innerView H x r ws =
      (Claims.run H (Claims.replay (L.msgs x ws)) Bd.IVStar (xIOf x) () fun i => Claims.coinSeq r i.val).seen :=
  rfl

/-- **`DE`'s rounds are the deployed plain verifier's.** -/
theorem deE_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    (deE rel nextE colsE wE).V.k = (FlockDeploy.V false).k :=
  rfl

/-- **The inner decoder meets the spec at the pinned `DE`** wherever it does at the deployed plain record: `decOK` reads
the code (`enc`, `len`, `θ`), which `deE` keeps. -/
theorem decOK_deE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    Claims.Recursion.decOK (deE rel nextE colsE wE) decI ↔ Claims.Recursion.decOK (FlockDeploy.deployPlain rel) decI :=
  Iff.rfl

end RecDeploy
