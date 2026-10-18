import Flock.Hm96
import PouwCommit.Frame
import PouwCommit.Py
import PouwCommit.Coins
import PouwCommit.Draws

/-!
# Live coins, by label and index, and PoUW's live epoch (`verity.core.protocols.pouw.coins`)

A live coin layout is a scheme, an epoch and its families `(label, count)` in order: coin `(label, i)` sits at its family's
offset plus `i`. Before the epoch the challenger draws every coin and commits to them:

    x_p     = SHA-512(scheme ‖ 0x00 ‖ coin_p)
    value_p = hm96-sha512 tree_leaf(key, commit_string(key, x_p, salt_p))           salt_p 192 bytes, the pinned key
    root    = the frame-v3-sha512 tree over value_0 … value_(K−1): binding tagged(scheme, e, label_0, count_0, …), owner -1,
              leaf schema the scheme

and opens coin `(label, i)` once, after the root the verifier recorded for it. `liveReason` is `live_reason`, the check of
one opening; `orderReason` is `order_reason`, the check of a family's openings against the roots the verifier recorded, in
order: coin `i` opened after root `i`, each once, none before its root is recorded, and every recorded root's coin opened.
Both give Python's reasons in Python's order and words, `none` when the coins check, and throw where Python raises: a coin
or family the layout doesn't hold, and an opening path holding an item of neither 32 nor 64 bytes.

PoUW's live epoch is the layout `liveFamilies R` under `LIVE_SCHEME`: its salt coin, its draw coin and `R` round coins.
`roundsReason` is `rounds_reason`, the check of a window's round coins, and `roundKey` is `round_key`, round `r`'s noise key
from its opened coin and the round's root. `mode` is the epoch's coin mode, Compose's `CoinMode`, which the coin
commitment's scheme says: `SCHEME`'s epochs are seeded, `LIVE_SCHEME`'s live.

The JSON forms are read as `LiveCommitment.from_json` and `LiveOpening.from_json` read them, refusing what they refuse.
`layoutOf` is `layout_problem` over a scheme, an epoch and families as JSON gives them, each family a list (`from_json`
makes tuples of them).
-/

namespace PouwCommit

open Lean Flock

def LIVE_SCHEME : String := "verity/pouw/epoch-coins/live/v1"
def ROUND : String := "round"
def ROUND_NOISE_DOMAIN : String := "verity/pouw/round-noise/v1"
def MAX_COINS : Nat := 2 ^ 20

/-- A scheme or a label: nonempty, of lowercase ASCII letters, digits and '-', and '/' in a scheme. -/
def isName (s : String) (scheme : Bool) : Bool :=
  s != "" && s.toList.all fun c => ('a' ≤ c && c ≤ 'z') || ('0' ≤ c && c ≤ '9') || c == '-' || (scheme && c == '/')

def distinct : List String → Bool
  | [] => true
  | l :: ls => !ls.contains l && distinct ls

def NOT_LAYOUT : String :=
  "a live coin layout is a nonempty sequence of (label, count), a name and a positive integer"

/-- `layout_problem` (module docstring): the layout, or why there is none in `layout_problem`'s words. -/
def layoutOf (scheme epoch families : Json) : Except String (String × Nat × List (String × Nat)) := do
  let .str s := scheme | throw "a live coin scheme is a nonempty name of lowercase letters, digits, '-' and '/'"
  if !isName s true then throw "a live coin scheme is a nonempty name of lowercase letters, digits, '-' and '/'"
  let some e := jsonInt? epoch | throw "an epoch is a nonnegative integer"
  if e < 0 then throw "an epoch is a nonnegative integer"
  let .arr fs := families | throw NOT_LAYOUT
  let some fs := fs.toList.mapM (fun | .arr f => some f.toList | _ => none) | throw NOT_LAYOUT
  if fs.isEmpty then throw NOT_LAYOUT
  let fs ← fs.mapM fun
    | [.str l, n] => match jsonInt? n with
      | some k => if isName l false && 1 ≤ k then pure (l, k.toNat) else throw NOT_LAYOUT
      | none => throw NOT_LAYOUT
    | _ => throw NOT_LAYOUT
  if !distinct (fs.map (·.1)) then throw "a live coin layout's labels are distinct"
  if (fs.map (·.2)).sum > MAX_COINS then throw s!"a live coin layout holds at most {MAX_COINS} coins"
  pure (s, e.toNat, fs)

/-- The layout's coin tree: `CommitmentDomain(tagged(scheme, e, label_0, count_0, …), -1, RangeIndexedDomain(K), "sha512")`. -/
def liveDomain (scheme : String) (epoch : Nat) (families : List (String × Nat)) : Domain :=
  let total := (families.map (·.2)).sum
  ⟨.sha512, total, frameHash .sha512 "domain"
    [tagged scheme (.int epoch :: families.flatMap fun (l, n) => [.str l, .int n]), natBytes 1, rangeIdentity .sha512 total,
     natBytes total]⟩

/-- `live_leaf`: the committed value of one live coin. -/
def liveLeaf (scheme : String) (coin salt : ByteArray) : ByteArray :=
  Hm96.Default512.leaf (Sha512.hash (scheme.toUTF8.push 0 ++ coin)) salt

structure LiveCommitment where
  scheme : String
  epoch : Nat
  families : List (String × Nat)
  root : ByteArray

structure LiveOpening where
  epoch : Int
  label : String
  index : Int
  after : ByteArray
  coin : ByteArray
  salt : ByteArray
  path : List ByteArray

namespace LiveCommitment

def count (c : LiveCommitment) : Nat := (c.families.map (·.2)).sum

/-- How many coins family `label` holds, or `none` when the layout has no such family. -/
def family (c : LiveCommitment) (label : String) : Option Nat := (c.families.find? (·.1 == label)).map (·.2)

/-- Coin (`label`, `index`)'s position in the tree, or `none` when the layout has no such coin. -/
def position (c : LiveCommitment) (label : String) (index : Int) : Option Nat := go c.families 0
where
  go : List (String × Nat) → Nat → Option Nat
    | [], _ => none
    | (name, n) :: rest, offset =>
      if name == label then (if 0 ≤ index && index < n then some (offset + index.toNat) else none) else go rest (offset + n)

/-- `LiveCommitment.from_json`. -/
def ofJson (j : Json) : Except String LiveCommitment := do
  let families ← field j "families"
  match families with
  | .arr fs => if !fs.all (fun | .arr _ => true | _ => false) then throw "a live coin layout is a list of [label, count]"
  | _ => throw "a live coin layout is a list of [label, count]"
  let scheme ← field j "scheme"
  let epoch ← field j "epoch"
  let root ← hexField j "root"
  let (s, e, fs) ← layoutOf scheme epoch families
  if root.size != 64 then throw "a coin commitment's root is 64 bytes"
  pure ⟨s, e, fs, root⟩

end LiveCommitment

/-- `LiveOpening.from_json`. -/
def LiveOpening.ofJson (j : Json) : Except String LiveOpening := do
  let (some epoch, some index) := (jsonInt? (← field j "epoch"), jsonInt? (← field j "index"))
    | throw "an opening's epoch and index are integers"
  let label ← match ← field j "label" with
    | .str s => pure s
    | _ => throw "an opening's label is a string"
  let after ← hexField j "after"
  let coin ← hexField j "coin"
  let salt ← hexField j "salt"
  let path ← match ← field j "path" with
    | .arr ps => ps.toList.mapM fun
      | .str s => pyFromHex s
      | _ => throw "a path item is not a string"
    | _ => throw "an opening's path is an array"
  pure ⟨epoch, label, index, after, coin, salt, path⟩

/-- `live_reason` (module docstring). -/
def liveReason (c : LiveCommitment) (o : LiveOpening) (label : String) (index : Int) (after : ByteArray) :
    Except String (Option String) := do
  let some rank := c.position label index | throw s!"the epoch's coins have no {pyRepr label} coin {index}"
  if o.epoch != c.epoch then return some s!"coin is epoch {o.epoch}'s, not epoch {c.epoch}'s"
  if o.label != label || o.index != index then
    return some s!"coin is the epoch's {pyRepr o.label} coin {o.index}, not its {pyRepr label} coin {index}"
  if o.coin.size != COIN_BYTES || o.salt.size != 192 then return some "coin opening is malformed"
  let value := liveLeaf c.scheme o.coin o.salt
  checkPath o.path
  let d := liveDomain c.scheme c.epoch c.families
  if !(← d.verify c.root rank value o.path fun v => pure (d.leaf rank c.scheme v)) then
    return some "coin does not open against the epoch's coin commitment"
  if o.after != after then return some "coin was not opened after what the verifier recorded"
  return none

/-- `order_reason` (module docstring): `roots` the family's roots in the order the verifier recorded them. -/
def orderReason (c : LiveCommitment) (label : String) (roots : List ByteArray) (openings : List LiveOpening) :
    Except String (Option String) := do
  let some count := c.family label | throw s!"the epoch's coins have no {pyRepr label} family"
  if roots.length > count then
    return some s!"{roots.length} roots recorded, more than the epoch's {count} {pyRepr label} coins"
  let mut seen : List (String × Int) := []
  for (o, i) in openings.zipIdx do
    if seen.contains (o.label, o.index) then
      return some s!"the epoch's {pyRepr o.label} coin {o.index} is opened twice: a retry never redraws"
    seen := (o.label, o.index) :: seen
    let some root := roots[i]? | return some s!"{pyRepr label} coin {i} was opened before its root was recorded"
    match ← liveReason c o label i root with
    | some why => return some s!"{pyRepr label} coin {i}: {why}"
    | none => pure ()
  if openings.length < roots.length then return some s!"{pyRepr label} coin {openings.length} was never opened"
  return none

/-- An epoch's coin mode (Compose's `CoinMode`). -/
inductive CoinMode where
  | live
  | seeded
  deriving BEq, Repr

/-- `mode`: the coin mode of an epoch whose coin commitment is of `scheme`. -/
def mode (scheme : String) : Except String CoinMode :=
  if scheme == SCHEME then pure .seeded
  else if scheme == LIVE_SCHEME then pure .live
  else throw s!"a PoUW epoch's coin scheme is {pyRepr SCHEME} (seeded) or {pyRepr LIVE_SCHEME} (live)"

/-- `live_families`: a live epoch's layout, its salt coin, its draw coin and `rounds` round coins. -/
def liveFamilies (rounds : Nat) : List (String × Nat) := [("salt", 1), ("draw", 1), (ROUND, rounds)]

/-- `live_epoch_reason`: why `c` is not a live PoUW epoch's coin commitment; `none` when it is. -/
def liveEpochReason (c : LiveCommitment) : Option String :=
  if c.scheme != LIVE_SCHEME then some s!"the coin commitment's scheme is {pyRepr c.scheme}, not {pyRepr LIVE_SCHEME}"
  else match c.families with
    | [s, d, (r, _)] =>
      if s == ("salt", 1) && d == ("draw", 1) && r == ROUND then none
      else some "the coin commitment's coins are not a live epoch's: one salt, one draw, then the rounds'"
    | _ => some "the coin commitment's coins are not a live epoch's: one salt, one draw, then the rounds'"

/-- `rounds_reason`: why `openings` are not a window's round coins over its round roots `roots`, in order. -/
def roundsReason (c : LiveCommitment) (roots : List ByteArray) (openings : List LiveOpening) :
    Except String (Option String) := do
  if let some r := liveEpochReason c then return some r
  orderReason c ROUND roots openings

/-- `round_key`: round `r`'s noise key from its opened coin and the round's root. -/
def roundKey (c : LiveCommitment) (coin : ByteArray) (r : Nat) (root : ByteArray) : Except String ByteArray :=
  key? coin ROUND_NOISE_DOMAIN [("coins", .bytes c.root), ("epoch", .int c.epoch), ("root", .bytes root), ("round", .int r)]

end PouwCommit
