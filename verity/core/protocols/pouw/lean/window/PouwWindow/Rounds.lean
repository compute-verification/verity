import PouwCommit.Draws
import SampledProofs.Challenger

/-!
# A live window's round coins (`window.py`, "Live windows")

PoUW's live window is the proof system's challenger's first client (`SampledProofs.Challenger`). The check item's
optional `coins` is the window's coin mode (`mode`, Compose's `CoinMode`), seeded when the item names none. A live
epoch's coins are the layout `liveFamilies R` under `LIVE_SCHEME`: its salt coin, its draw coin and `R` round coins
(`pouw/round`, rounds numbered from 0), all committed before the epoch. Round `r`'s coin opens once, after the verifier
recorded round `r`'s root. `liveEpochReason` is `live_epoch_reason`, `roundsReason` is `rounds_reason`, the rule on a
window's round coins, and `roundKey` is `round_key`, round `r`'s noise key from its opened coin and the round's root.
A Pearl-C pass's live session is reserved for its salt coin alone (`SALT_FAMILIES`), which `saltEpochReason`
(`salt_epoch_reason`) checks.
-/

namespace PouwWindow

open SampledProofs.Challenger (LiveCommitment LiveOpening)

def COINS : String := "coins"
def LIVE_SCHEME : String := "verity/pouw/epoch-coins/live/v1"
def SALT : String := "pouw/salt"
def DRAW : String := "pouw/draw"
def ROUND : String := "pouw/round"
def ROUND_NOISE_DOMAIN : String := "verity/pouw/round-noise/v1"

/-- A window's coin mode (Compose's `CoinMode`). -/
inductive CoinMode where
  | live
  | seeded
  deriving DecidableEq, Repr

def CoinMode.name : CoinMode → String
  | .live => "live"
  | .seeded => "seeded"

/-- `MODES`, by their names. -/
def CoinMode.ofName? : String → Option CoinMode
  | "live" => some .live
  | "seeded" => some .seeded
  | _ => none

/-- `mode`: the coin mode a check item names, seeded when it names none. -/
def mode (coins : Option CoinMode) : CoinMode := coins.getD .seeded

/-- `live_families`: a live epoch's layout, its salt coin, its draw coin and `rounds` round coins. -/
def liveFamilies (rounds : Nat) : List (String × Nat) := [(SALT, 1), (DRAW, 1), (ROUND, rounds)]

def NOT_LIVE_EPOCH : String := "the coin commitment's coins are not a live epoch's: one salt, one draw, then the rounds'"

/-- `live_epoch_reason`: why `c` is not a live epoch's coin commitment; `none` when it is. -/
def liveEpochReason (c : LiveCommitment) : Option String :=
  if c.scheme != LIVE_SCHEME then some s!"the coin commitment's scheme is {c.scheme}, not {LIVE_SCHEME}"
  else match c.families with
    | [s, d, (r, _)] => if s == (SALT, 1) && d == (DRAW, 1) && r == ROUND then none else some NOT_LIVE_EPOCH
    | _ => some NOT_LIVE_EPOCH

/-- `SALT_FAMILIES`: a Pearl-C pass's live layout, its salt coin alone, so a session reserved with it is complete exactly
when it released that coin. -/
def SALT_FAMILIES : List (String × Nat) := [(SALT, 1)]

def NOT_SALT_EPOCH : String := "the coin commitment's coins are not a Pearl-C salt session's: its salt coin alone"

/-- `salt_epoch_reason`: why `c` is not a Pearl-C salt session's coin commitment; `none` when it is. -/
def saltEpochReason (c : LiveCommitment) : Option String :=
  if c.scheme != LIVE_SCHEME then some s!"the coin commitment's scheme is {c.scheme}, not {LIVE_SCHEME}"
  else if c.families != SALT_FAMILIES then some NOT_SALT_EPOCH
  else none

/-- `rounds_reason`: why `openings` are not a live window's round coins over the round roots the verifier recorded,
`roots` in order; `none` when they are. -/
def roundsReason (c : LiveCommitment) (roots : List ByteArray) (openings : List LiveOpening) :
    Except String (Option String) := do
  if let some why := liveEpochReason c then return some why
  SampledProofs.Challenger.orderReason c ROUND roots openings

/-- `round_key`: round `r`'s noise key from its opened coin and the round's root. -/
def roundKey (c : LiveCommitment) (coin : ByteArray) (r : Nat) (root : ByteArray) : Except String ByteArray :=
  PouwCommit.key? coin ROUND_NOISE_DOMAIN
    [("coins", .bytes c.root), ("epoch", .int c.epoch), ("root", .bytes root), ("round", .int r)]

end PouwWindow
