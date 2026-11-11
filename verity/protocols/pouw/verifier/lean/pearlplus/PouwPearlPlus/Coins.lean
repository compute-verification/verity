module

public import PouwPearlPlus.Run
public import PouwCommit.Py
public import PouwWindow.Rounds

/-!
# The run's coins: the epoch's salt coin and the verifier's sample (`verify_run`, "The run's sample" and "The coin record")

* **The epoch's coins** (`EpochCoins`, `verify_run.epoch_coins`) are an in-process challenger's coin commitment
  (`PouwCommit.CoinCommitment`), or a live session's on the auditor host's challenger when the coins file's scheme is
  `window.LIVE_SCHEME` (`Spot.Challenger.LiveCommitment`). A live pass's salt coin opens as `host.epoch_of` holds
  it (`liveSalt`): the commitment is a Pearl-C salt session's, its salt coin alone (`window.salt_epoch_reason`), and the
  opening is that coin opened after the salt epoch's open root (`salt.epoch_open_root` of the fixed operands' root and the
  manifest's ``pass_index``, `challenger.live_reason`); the salt is `audit.salt_of`'s over the fixed operands' root.
* **The verifier's sample** is the coin record's ``draw``: a challenger sample session's coin ``pouw/draw`` 0, or the node's
  coins. `drawReason` holds a challenger sample as a reader of the record does (`served_verdicts.pearlplus_live_draw`, less
  the auditor's ledger, key and period, which the compose checks): its opening is coin ``pouw/draw`` 0 of its commitment
  opened after the run's commitment C (`challenger.live_reason`), its coin is the record's source S, and its commitment is
  of a sample session's layout (`window.draw_epoch_reason`).
* **Of record** (`--of-record`), the verdict takes only a challenger sample on live epoch coins: `ofRecordReason` refuses
  the node's sample, a v1 record (which names no sample), no record, a source given on the command line, and an in-process
  challenger's epoch coins. Without it they are read as development runs.

Every JSON comparison here is by a match (`PouwCommit.fieldIs`), never `Lean.Json`'s `==`.
-/

@[expose] public section

namespace PouwCheck.PearlPlus

open Lean PouwCommit
open Spot.Challenger (LiveCommitment LiveOpening)

def LIVE_SCHEME : String := PouwWindow.LIVE_SCHEME

/-- `repr` of a JSON value as Python prints the value `json.loads` gives. -/
def pyReprJson : Json → String
  | .null => "None"
  | .bool b => if b then "True" else "False"
  | .str s => pyRepr s
  | .num ⟨m, 0⟩ => toString m
  | j => j.compress

def get? (j : Json) (k : String) : Option Json :=
  match j with
  | .obj kv => kv.get? k
  | _ => none

/-- The epoch's coin commitment as its challenger published it: an in-process challenger's, or a live session's. -/
inductive EpochCoins where
  | inproc (c : CoinCommitment)
  | live (c : LiveCommitment)

instance instCoeCoinCommitmentEpochCoins : Coe CoinCommitment EpochCoins := ⟨.inproc⟩

/-- `verify_run.epoch_coins`: a live session's commitment when the document's scheme is `LIVE_SCHEME`, else an in-process
challenger's. -/
def EpochCoins.ofJson (j : Json) : Except String EpochCoins :=
  if fieldIs (get? j "scheme") LIVE_SCHEME then EpochCoins.live <$> LiveCommitment.ofJson j
  else EpochCoins.inproc <$> CoinCommitment.ofJson j

/-- `LiveCommitment.__eq__`, field by field. -/
def sameLive (a b : LiveCommitment) : Bool :=
  a.scheme == b.scheme && a.epoch == b.epoch && a.families == b.families && a.root == b.root

def EPOCH_OPEN_TAG : ByteArray := "verity/pouw-epoch-open/v1".toUTF8

example : EPOCH_OPEN_TAG = Domains.POUW_EPOCH_OPEN.toUTF8 := rfl

/-- `salt.epoch_open_root`: SHA-512(`EPOCH_OPEN_TAG` ‖ the fixed operands' root ‖ u64be(`k`)). -/
def epochOpenRoot (weights : ByteArray) (k : Nat) : ByteArray :=
  Flock.Sha512.hash (EPOCH_OPEN_TAG ++ weights ++ be8 k)

/-- `host.epoch_of`'s live salt, once every fixed operand is committed: `.ok salt`, or `.error` with its refusal (`TYPE: why`).
`k` is the manifest's ``pass_index`` as it gives it. -/
def liveSalt (c : LiveCommitment) (o : LiveOpening) (weightsRoot : ByteArray) (k : Json) :
    Except String ByteArray := do
  if let some why := PouwWindow.saltEpochReason c then throw s!"DomainError: the salt's {why}"
  let some n := (jsonInt? k).filter (fun i => decide (0 ≤ i) && decide (i < 2 ^ 64))
    | throw s!"DomainError: a salt epoch's number is an integer in [0, 2^64), not {pyReprJson k}"
  match Spot.Challenger.liveReason c o PouwWindow.SALT 0 (epochOpenRoot weightsRoot n.toNat) with
  | .error e => throw s!"CoinError: {e}"
  | .ok (some why) => throw s!"DomainError: the salt's {why}"
  | .ok none => match saltOf ⟨c.epoch, c.root⟩ o.coin weightsRoot with
    | .ok s => pure s
    | .error e => throw s!"ValueError: {e}"

def DRAW_CHALLENGER : String := "challenger"
def DRAW_NODE : String := "node"
def NODE_CLAIM : String := "uniform/io-getrandombytes"

/-- The sample a record names, `{"from": "node", "claim": ...}`, as the verifier writes it for a source it is given. -/
def nodeDraw : Json := Json.mkObj [("from", .str DRAW_NODE), ("claim", .str NODE_CLAIM)]

/-- Why the record's challenger sample is not coin ``pouw/draw`` 0 of a sample session's commitment, opened after the run's
commitment `root`, its coin the record's source; `none` when it is, or when the record names no challenger sample. -/
def drawReason (h : CoinRecord) (root : ByteArray) : Option String := Id.run do
  if !fieldIs (get? h.draw "from") DRAW_CHALLENGER then return none
  let parsed : Except String (LiveCommitment × LiveOpening) := do
    pure (← LiveCommitment.ofJson ((get? h.draw "commitment").getD .null),
      ← LiveOpening.ofJson ((get? h.draw "opening").getD .null))
  let (c, o) ← match parsed with
    | .ok p => pure p
    | .error e => return some s!"the record's draw is malformed: CoinError: {e}"
  let why ← match PouwWindow.drawEpochReason c with
    | some why => pure (some why)
    | none => match Spot.Challenger.liveReason c o PouwWindow.DRAW 0 root with
      | .ok w => pure w
      | .error e => return some s!"the record's draw is malformed: CoinError: {e}"
  if let some why := why then return some s!"the record's draw coin: {why}"
  if o.coin != h.source then return some "the record's source is not its draw session's coin"
  return none

def DRAW_REFUSED : String := "draw"
def NOT_OF_RECORD : String := "of-record"

/-- Why a verdict on these coins is no verdict of record; `none` when the epoch's coins are a live session's and the
record (`held`) names a challenger sample. -/
def ofRecordReason (coins : EpochCoins) (source : Option ByteArray) (held : Option CoinRecord) : Option String :=
  match coins, source, held with
  | .inproc _, _, _ => some ("the epoch's coins are an in-process challenger's, not a live session's on the auditor " ++
      "host's challenger: a development run")
  | _, some _, _ => some "the source is given on the command line, not a challenger coin the verifier's record carries"
  | _, _, none => some "the verifier holds no record of its coin, so no challenger draw pins the source"
  | _, _, some h =>
    if h.schema != COIN_SCHEMA then
      some s!"the verifier's coin record is {pyRepr h.schema}, not {COIN_SCHEMA}, which carries its draw"
    else
      let src := get? h.draw "from"
      if fieldIs src DRAW_CHALLENGER then none
      else if fieldIs src DRAW_NODE then
        some ("the verifier's draw source is drawn on the serving node (verify_run.node_draw), not a challenger coin " ++
          "opened on the run commitment: a development draw no verdict of record takes")
      else some s!"the verifier's coin record's draw is from {pyReprJson (src.getD .null)}, not the challenger's"

end PouwCheck.PearlPlus
