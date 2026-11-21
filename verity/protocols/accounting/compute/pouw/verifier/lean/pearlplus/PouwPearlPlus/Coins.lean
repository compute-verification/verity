module

public import PouwPearlPlus.Run
public import PouwCommit.Py
public import PouwWindow.Rounds
public import Spot.ChallengerSession
public import Spot.ChallengerCoins

/-!
# The run's coins: the epoch's salt coin and the verifier's sample (`verify_run`, "The run's sample" and "The coin record")

* **The epoch's coins** (`EpochCoins`, `verify_run.epoch_coins`) are an in-process challenger's coin commitment
  (`PouwCommit.CoinCommitment`), or a live session's on the auditor device's challenger when the coins file's scheme is
  `window.LIVE_SCHEME` (`Spot.Challenger.LiveCommitment`). A live pass's salt coin opens as `host.epoch_of` holds
  it (`liveSalt`): the commitment is a Pearl-C salt session's, its salt coin alone (`window.salt_epoch_reason`), and the
  opening is that coin opened after the salt epoch's open root (`salt.epoch_open_root` of the fixed operands' root and the
  manifest's ``pass_index``, `challenger.live_reason`); the salt is `audit.salt_of`'s over the fixed operands' root.
* **The verifier's sample** is the coin record's ``draw``: a challenger sample session's coin ``pouw/draw`` 0, or the node's
  coins. `drawReason` holds a challenger sample as `served_verdicts.pearlplus_live_sample` does, less the auditor's ledger,
  key and period: its opening is coin ``pouw/draw`` 0 of its commitment opened after the run's commitment C
  (`challenger.live_reason`), its coin is the record's source S, and its commitment is of a sample session's layout
  (`window.draw_epoch_reason`).
* **The challenger key.** `check` reads no challenger key: an opening is checked only against the commitment it is
  handed. `liveDrawReason` (`pouw-pearlplus --draw-live`) reads the pinned key, the auditor's ledger and the draw
  session's files, and holds a challenger sample to its draw session, the ledger's and the session's signatures verified
  under the key. `liveSaltReason` (`--salt-live`) does the same for each live pass's salt session
  (`served_verdicts.pearlplus_coins`, `core_check.live_salt`): the session the ledger opened on the pass's salt epoch's
  open root, its signed commitment the manifest's coins. The compose side reads copies the node hands it, and no
  composite and no compose-side code verifies a signature.
* **Of record** (`--of-record`), the verdict takes only a challenger sample on live epoch coins: `ofRecordReason` refuses
  the node's sample, a v1 record (which names no sample), no record, a source given on the command line, and an in-process
  challenger's epoch coins. Without it they are read as development runs.

Every JSON comparison here is by a match (`PouwCommit.fieldIs`), never `Lean.Json`'s `==`.
-/

@[expose] public section

namespace PouwVerifier.PearlPlus

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

/-- `challenger.pinned_key`'s pin: why `key` is not the challenger key whose public key's SHA-512 the verifier pins,
`live`'s ``key_sha512`` (`H1_KEY_SHA512`); `none` when it is. The ledger's opens name the key they were signed under, so
without the pin any key the ledger was signed under would hold. -/
def keyPinReason (live : Json) (key : Spot.Challenger.ChallengerKey) : Option String :=
  if fieldIs (get? live "key_sha512") (Flock.hexOf (Flock.Sha512.hash key.pub)) then none
  else some "challenger key is not the one the verifier pins (key_sha512, the SHA-512 of its public key)"

/-- `served_verdicts.pearlplus_draw`: why the record's challenger sample is not its draw session's, read from the auditor's
ledger and the session's own files (`{key, key_sha512, ledger, pins, main, session}`); `none` when it is, or when the
record names no challenger sample. The key is the one `key_sha512` pins (`keyPinReason`). The session is the one the
ledger opens on the run's commitment `root` under that key,
run from a commit the host's pins log records on main's first-parent line, from its files
(`Spot.Challenger.trustedChallengePhaseOf`, which verifies the ledger's and the session's signatures under that key); the
ledger opens it withheld (`withheldOn`); its signed commitment is a sample session's (`drawEpochReason`) and the
record's; and its log shows `root` alone as the root its draw coin followed. The record's opening and source are
`drawReason`'s, against the record's commitment, which this holds to the signed one. -/
def liveDrawReason (h : CoinRecord) (root : ByteArray) (live : Json) : Option String := Id.run do
  if !fieldIs (get? h.draw "from") DRAW_CHALLENGER then return none
  let g := fun k => (get? live k).getD .null
  let k ← match Spot.Challenger.ChallengerKey.ofJson (g "key") with
    | .ok k => pure k
    | .error e => return some s!"the draw session's challenger key is malformed: {e}"
  if let some why := keyPinReason live k then return some s!"the sample session's {why}"
  let window := Json.mkObj [("root", .str (Flock.hexOf root))]
  let (c, log) ← match Spot.Challenger.trustedChallengePhaseOf (g "ledger") window k (g "pins") (g "main") (g "session") with
    | .ok p => pure p
    | .error e => return some s!"the draw session is not trusted: {e}"
  if !Spot.Challenger.withheldOn (g "ledger") (Flock.hexOf root) then
    return some s!"the run's draw session: {Spot.Challenger.WITHHELD}"
  if let some why := PouwWindow.drawEpochReason c then return some s!"the draw session's signed coins: {why}"
  let .ok claimed := LiveCommitment.ofJson ((get? h.draw "commitment").getD .null)
    | return some "the record's draw commitment is malformed"
  if !sameLive claimed c then
    return some "the record's draw commitment is not the one the draw session's challenger signed"
  if ((Spot.Challenger.byFamily c log).lookup PouwWindow.DRAW).getD [] != [root] then
    return some "the draw session's log does not show the run's commitment as the one root its draw coin followed"
  return none

/-- `core_check.NO_RESERVE`, `RESERVE_LAYOUT`, `INCOMPLETE`, `LIVE_NOT_RECEIVED`, `LIVE_NOT_SIGNED` and `LIVE_SESSION`, as
`PouwVerifier.liveSaltCheck` words them. -/
def SALT_NO_RESERVE : String :=
  "the auditor's ledger opens the window's session without reserving it: a window's open root binds the session a " ++
    "signed reserve entry names"
def SALT_RESERVE_LAYOUT : String :=
  "the auditor's ledger reserves the pass's session for another layout than a Pearl-C salt session's, its salt coin " ++
    "alone"
def SALT_INCOMPLETE : String :=
  "the auditor's ledger does not end the pass's session complete, having released its salt coin and nothing else"
def SALT_NOT_RECEIVED : String :=
  "the challenger's log does not show the epoch's root as the one root its salt coin followed"
def SALT_NOT_SIGNED : String := "the pass's coin commitment is not the one its session's challenger signed"
def SALT_SESSION_CARRIED : String :=
  "the pass's session is not the gate's answer, without its secret, for the session the auditor's ledger opened on " ++
    "its epoch's root: its session, seq, root and commitment"

/-- `core_check._carries`: `s`, the session a pass's manifest carries, is the gate's open answer for session `sid`,
opened at ledger entry `seq` on `root` under the commitment whose root is `commitment` (hex), without its `secret`. -/
def carriesSession (s : Json) (sid : String) (seq : Nat) (root : ByteArray) (commitment : String) : Bool :=
  match s with
  | .obj _ => (get? s "secret").isNone && fieldIs (get? s "session") sid &&
      ((get? s "seq").bind jsonInt?) == some (seq : Int) && fieldIs (get? s "root") (Flock.hexOf root) &&
      fieldIs (get? s "commitment") commitment
  | _ => false

/-- `served_verdicts.pearlplus_coins` (`core_check.live_salt`, `PouwVerifier.liveSaltCheck`'s session half): why a live
pass's salt is not its salt session's, read from the auditor's ledger and the session's own files
(`{key, key_sha512, ledger, pins, main, session}`); `none` when it is. `m` is a pass `check` loaded, its salt opened
by `liveSalt` against the manifest's coins after R = `epochOpenRoot weightsRoot k` (k its ``pass_index``). The key is
the one `key_sha512` pins (`keyPinReason`). The session is the one the ledger opens on `{root: R}` under that key, run
from a commit the host's pins log records on main's first-parent line (`pinnedChallengePhase`), from its files
(`challengePhaseNamed`), each signature verified under that key;
its signed commitment is a Pearl-C salt session's (`PouwWindow.saltEpochReason`); the ledger reserved it for that
layout, opened it binding that reserve and ended it complete, having released its salt coin and nothing else; its log
shows R alone as the root its salt coin followed; the manifest's coins are its signed commitment; and the manifest's
session is the gate's answer for it without its secret (`carriesSession`). Since R hashes the fixed operands' root W, the
ledger logged W before the session opened, and the salt coin was released after it. -/
def liveSaltReason (m : Json) (weightsRoot : ByteArray) (live : Json) : Option String := Id.run do
  let g := fun k => (get? live k).getD .null
  let some n := ((get? m "pass_index").bind jsonInt?).filter (fun i => decide (0 ≤ i) && decide (i < 2 ^ 64))
    | return some "the pass's manifest names no epoch: `pass_index` is an integer in [0, 2^64)"
  let r := epochOpenRoot weightsRoot n.toNat
  let key ← match Spot.Challenger.ChallengerKey.ofJson (g "key") with
    | .ok k => pure k
    | .error e => return some s!"the salt session's challenger key is malformed: {e}"
  if let some why := keyPinReason live key then return some s!"the salt session's {why}"
  let window := Json.mkObj [("root", .str (Flock.hexOf r))]
  let a ← match Spot.Challenger.pinnedChallengePhase (g "ledger") window key (g "pins") (g "main") with
    | .ok a => pure a
    | .error e => return some s!"the salt session is not trusted: {e}"
  let (c, log) ← match Spot.Challenger.challengePhaseNamed a key (g "session") with
    | .ok p => pure p
    | .error e => return some s!"the salt session is not trusted: {e}"
  if let some why := PouwWindow.saltEpochReason c then return some s!"the salt session's signed coins: {why}"
  let named : List Spot.Challenger.LedgerEntry :=
    ((match g "ledger" with | .arr ls => ls.toList | _ => []).filterMap Spot.Challenger.ledgerEntry?).filter
      (fun e : Spot.Challenger.LedgerEntry => e.session == a.session)
  let (some reserve, some opened, some ended) :=
      (named.find? (·.kind == "reserve"), named.find? (·.kind == "open"), named.find? (·.kind == "end"))
    | return some SALT_NO_RESERVE
  if !Spot.jeq (Spot.getD reserve.body "families") (Spot.Challenger.familiesJson PouwWindow.SALT_FAMILIES) then
    return some SALT_RESERVE_LAYOUT
  if Spot.int? (Spot.getD opened.body "reserve") != some (reserve.seq : Int) then return some SALT_NO_RESERVE
  let complete := match Spot.getD ended.body "complete" with | .bool true => true | _ => false
  if !(complete && Spot.jeq (Spot.getD ended.body "released") (Json.mkObj [(PouwWindow.SALT, toJson (1 : Nat))])) then
    return some SALT_INCOMPLETE
  if Spot.Challenger.byFamily c log != [(PouwWindow.SALT, [r])] then return some SALT_NOT_RECEIVED
  let .ok claimed := LiveCommitment.ofJson ((get? m "coins").getD .null)
    | return some "the pass's coins are malformed"
  if !sameLive claimed c then return some SALT_NOT_SIGNED
  if !carriesSession ((get? m "session").getD .null) a.session opened.seq r a.commitment then
    return some SALT_SESSION_CARRIED
  return none

def DRAW_REFUSED : String := "draw"
def DRAW_SESSION_REFUSED : String := "draw-session"
def SALT_SESSION_REFUSED : String := "salt-session"
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

end PouwVerifier.PearlPlus
