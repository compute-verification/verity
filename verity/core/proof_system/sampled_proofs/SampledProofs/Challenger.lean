/-
The proof service's challenger: its live coins, the one live-coin source of every protocol, as the verifier checks them
(`verity/core/protocols/sampled_proofs/challenger.py`, which `tests/test_challenger_difftest.py` holds to this, the
verifier of record).

Coins are indexed by label and index and named by no protocol. A layout is a scheme, an epoch and its families
`(label, count)` in order: coin `(label, i)` sits at its family's offset plus `i`, at most `MAX_COINS` coins. Before the
epoch the challenger draws every coin and its hm96 salt and publishes the coin commitment:

    x_p     = SHA-512(scheme ‖ 0x00 ‖ coin_p)                                       coin_p 64 bytes (`COIN_BYTES`)
    value_p = hm96-sha512 tree_leaf(key, commit_string(key, x_p, salt_p))           salt_p 192 bytes, the pinned key
    root    = the frame-v3-sha512 tree over value_0 … value_(K−1) at positions 0 … K−1, leaf schema the scheme, domain
              binding identity_digest_sha512("verity/challenger/coins/v1", {epoch, families, scheme}), owner -1

and opens coin `(label, i)` once, after the root the verifier recorded for it. There is no beacon and no Fiat–Shamir.
`after` binds an opening to that root but can't show the coin was released after it: the release order is a trust
assumption on the auditor-trusted challenger (`trusted-host`), not something this checks.
`liveReason` is `live_reason`, the check of one opening; `orderReason` is `order_reason`, the check of a family's
openings against the roots the verifier recorded, in order: coin `i` opened after root `i`, each once, none before its
root is recorded, and every recorded root's coin opened. Both give Python's reasons in Python's words and order, `none`
when the coins check, and throw where Python raises (`CoinError`): a coin or a family the layout doesn't hold.

`LiveCommitment.ofJson` and `LiveOpening.ofJson` read the JSON forms as `from_json` reads them, refusing what it refuses
with its messages; `layoutOf` is `layout_problem` over a layout as JSON gives it.

The difftest's pieces (`Pieces.table`): `challenger_commitment`, `challenger_opening`, `challenger_reason` and
`challenger_order`, each `{"refused": why}` where the challenger refuses, so the reasons are compared too.
-/
import Flock.Hm96
import Flock.Net
import Flock.Public
import Flock.Registered
import SampledProofs.Json

namespace SampledProofs.Challenger

open Lean (Json)
open Flock (V)

def BINDING : String := "verity/challenger/coins/v1"
def COIN_BYTES : Nat := 64
def SALT_BYTES : Nat := 192
def MAX_COINS : Nat := 2 ^ 20
def COMMITMENT_FORM : String :=
  "a live coin commitment is {epoch, families, root, scheme}: families a list of [label, count], root lowercase hex"
def OPENING_FORM : String :=
  "a live coin opening is {after, coin, epoch, index, label, path, salt}: integers, a string, lowercase hex and a path of SHA-512 digests"
def NOT_SCHEME : String := "a live coin scheme is a nonempty name of lowercase letters, digits, '-' and '/'"
def NOT_EPOCH : String := "an epoch is a nonnegative integer"
def NOT_LAYOUT : String := "a live coin layout is a nonempty sequence of (label, count), a name and a positive integer"

/-- A scheme or a label: nonempty, of lowercase ASCII letters, digits, '-' and '/'. -/
def isName (s : String) : Bool :=
  !s.isEmpty && s.all fun c => ('a' ≤ c && c ≤ 'z') || ('0' ≤ c && c ≤ '9') || c == '-' || c == '/'

def distinct : List String → Bool
  | [] => true
  | l :: ls => !ls.contains l && distinct ls

abbrev Families := List (String × Nat)

def total (fs : Families) : Nat := (fs.map (·.2)).sum

/-- `layout_problem` (module docstring) over a layout as JSON gives it, each family a list: the layout, or why there is
none. -/
def layoutOf (scheme epoch families : Json) : V (String × Nat × Families) := do
  let some s := (match scheme with | .str s => some s | _ => none).filter isName | throw NOT_SCHEME
  let some e := nat? epoch | throw NOT_EPOCH
  let .arr fs := families | throw NOT_LAYOUT
  if fs.isEmpty then throw NOT_LAYOUT
  let fs ← fs.toList.mapM fun
    | .arr #[.str l, n] => match nat? n with
      | some k => if isName l && 1 ≤ k then pure (l, k) else throw NOT_LAYOUT
      | none => throw NOT_LAYOUT
    | _ => throw NOT_LAYOUT
  unless distinct (fs.map (·.1)) do throw "a live coin layout's labels are distinct"
  if total fs > MAX_COINS then throw s!"a live coin layout holds at most {MAX_COINS} coins"
  pure (s, e, fs)

/-- `tagged_sha512(tag, canonical_json(manifest))`. -/
def taggedSha512 (tag : String) (manifest : Json) : V ByteArray := do
  let t := tag.toUTF8
  let m := (← Flock.canon manifest).toUTF8
  return Flock.Sha512.hash
    ("verity/tagged-sha512/v1".toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 m.size.toUInt64 ++ m)

def familiesJson (fs : Families) : Json := Json.arr (fs.toArray.map fun (l, n) => Json.arr #[Json.str l, Lean.toJson n])

/-- `live_domain(scheme, epoch, families).domain_id`: `CommitmentDomain(binding, -1, RangeIndexedDomain(K), "sha512")`. -/
def liveDomain (scheme : String) (epoch : Nat) (fs : Families) : V ByteArray := do
  let binding ← taggedSha512 BINDING
    (Json.mkObj [("epoch", Lean.toJson epoch), ("families", familiesJson fs), ("scheme", Json.str scheme)])
  let positions ← taggedSha512 "veritor/indexed-domain/range/v1"
    (Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson (total fs))])
  return Flock.FV3.hash "domain" [binding, Flock.FV3.uint 1, positions, Flock.FV3.uint (total fs)] Flock.Sha512.hash

/-- `live_leaf`: the committed value of one coin. -/
def liveLeaf (scheme : String) (coin salt : ByteArray) : ByteArray :=
  Flock.Hm96.Default512.leaf (Flock.Sha512.hash (scheme.toUTF8.push 0 ++ coin)) salt

/-- What the challenger publishes before the epoch. -/
structure LiveCommitment where
  scheme : String
  epoch : Nat
  families : Families
  root : ByteArray

/-- Coin (`label`, `index`) of epoch `epoch`, opened after the verifier recorded `after`. -/
structure LiveOpening where
  epoch : Int
  label : String
  index : Int
  after : ByteArray
  coin : ByteArray
  salt : ByteArray
  path : List ByteArray

private def hexAt (j : Json) (k : String) : Option ByteArray :=
  match getD j k with
  | .str s => (Flock.unhex s).toOption
  | _ => none

namespace LiveCommitment

def count (c : LiveCommitment) : Nat := total c.families

/-- How many coins family `label` holds, or `none` when the layout has no such family. -/
def family (c : LiveCommitment) (label : String) : Option Nat := (c.families.find? (·.1 == label)).map (·.2)

/-- Coin (`label`, `index`)'s position in the tree, or `none` when the layout has no such coin. -/
def position (c : LiveCommitment) (label : String) (index : Int) : Option Nat := go c.families 0
where
  go : Families → Nat → Option Nat
    | [], _ => none
    | (name, n) :: rest, offset =>
      if name == label then (if 0 ≤ index && index < n then some (offset + index.toNat) else none)
      else go rest (offset + n)

def toJson (c : LiveCommitment) : Json :=
  Json.mkObj [("epoch", Lean.toJson c.epoch), ("families", familiesJson c.families), ("root", Json.str (Flock.hexOf c.root)),
    ("scheme", Json.str c.scheme)]

/-- `LiveCommitment.from_json`. -/
def ofJson (j : Json) : V LiveCommitment := do
  unless keys? j == some #["epoch", "families", "root", "scheme"] do throw COMMITMENT_FORM
  let families := getD j "families"
  let lists := match families with
    | .arr fs => fs.all fun | .arr _ => true | _ => false
    | _ => false
  let some root := hexAt j "root" | throw COMMITMENT_FORM
  unless lists do throw COMMITMENT_FORM
  let (s, e, fs) ← layoutOf (getD j "scheme") (getD j "epoch") families
  unless root.size == 64 do throw "a coin commitment's root is 64 bytes"
  pure ⟨s, e, fs, root⟩

end LiveCommitment

namespace LiveOpening

def toJson (o : LiveOpening) : Json :=
  Json.mkObj [("after", Json.str (Flock.hexOf o.after)), ("coin", Json.str (Flock.hexOf o.coin)),
    ("epoch", Lean.toJson o.epoch), ("index", Lean.toJson o.index), ("label", Json.str o.label),
    ("path", Json.arr (o.path.toArray.map fun p => Json.str (Flock.hexOf p))), ("salt", Json.str (Flock.hexOf o.salt))]

/-- `LiveOpening.from_json`. -/
def ofJson (j : Json) : V LiveOpening := do
  unless keys? j == some #["after", "coin", "epoch", "index", "label", "path", "salt"] do throw OPENING_FORM
  let path : Option (List ByteArray) := match getD j "path" with
    | .arr ps => ps.toList.mapM fun
      | .str s => (Flock.unhex s).toOption.filter (·.size == 64)
      | _ => none
    | _ => none
  match int? (getD j "epoch"), int? (getD j "index"), getD j "label", hexAt j "after", hexAt j "coin", hexAt j "salt",
      path with
  | some epoch, some index, .str label, some after, some coin, some salt, some path =>
    pure ⟨epoch, label, index, after, coin, salt, path⟩
  | _, _, _, _, _, _, _ => throw OPENING_FORM

end LiveOpening

/-- `live_reason` (module docstring): why `o` is not coin (`label`, `index`) of `c`, opened after `after`; `none` when it
is. -/
def liveReason (c : LiveCommitment) (o : LiveOpening) (label : String) (index : Int) (after : ByteArray) :
    V (Option String) := do
  let some pos := c.position label index | throw s!"the epoch's coins have no {label} coin {index}"
  if o.epoch != (c.epoch : Int) then return some s!"coin is epoch {o.epoch}'s, not epoch {c.epoch}'s"
  if o.label != label || o.index != index then
    return some s!"coin is the epoch's {o.label} coin {o.index}, not its {label} coin {index}"
  if o.coin.size != COIN_BYTES || o.salt.size != SALT_BYTES then return some "coin opening is malformed"
  let dom ← liveDomain c.scheme c.epoch c.families
  let leaf := Flock.FV3.leaf dom pos c.scheme (liveLeaf c.scheme o.coin o.salt) Flock.Sha512.hash
  unless o.path.length == Flock.log2ceil c.count && o.path.all (·.size == 64) &&
      Flock.Registered.climbL dom leaf pos 0 o.path == c.root do
    return some "coin does not open against the epoch's coin commitment"
  if o.after != after then return some "coin was not opened after what the verifier recorded"
  return none

/-- `order_reason` (module docstring): `roots` the family's roots in the order the verifier recorded them. -/
def orderReason (c : LiveCommitment) (label : String) (roots : List ByteArray) (openings : List LiveOpening) :
    V (Option String) := do
  let some count := c.family label | throw s!"the epoch's coins have no {label} family"
  if roots.length > count then
    return some s!"{roots.length} roots recorded, more than the epoch's {count} {label} coins"
  let mut seen : List (String × Int) := []
  for (o, i) in openings.zipIdx do
    if seen.contains (o.label, o.index) then
      return some s!"the epoch's {o.label} coin {o.index} is opened twice: a retry never redraws"
    seen := (o.label, o.index) :: seen
    let some root := roots[i]? | return some s!"{label} coin {i} was opened before its root was recorded"
    match ← liveReason c o label i root with
    | some why => return some s!"{label} coin {i}: {why}"
    | none => pure ()
  if openings.length < roots.length then return some s!"{label} coin {openings.length} was never opened"
  return none

/-! The difftest's pieces. -/

private def refusing (f : V Json) : Json :=
  match f with
  | .ok j => j
  | .error why => Json.mkObj [("refused", Json.str why)]

private def reasonJson (r : Option String) : Json := Json.mkObj [("reason", r.elim Json.null Json.str)]

private def request (j : Json) (ks : Array String) : V Unit :=
  unless keys? j == some ks do throw s!"a request's keys are {", ".intercalate ks.toList}"

private def lowerHex (j : Json) (what : String) : V ByteArray := do
  let .str s := j | throw s!"{what} is not lowercase hex"
  Flock.unhex s

/-- A commitment's JSON to its own `toJson` with its coin count and its tree's domain. -/
def commitmentPiece (j : Json) : V Json := pure <| refusing do
  let c ← LiveCommitment.ofJson j
  let dom ← liveDomain c.scheme c.epoch c.families
  pure (c.toJson.setObjVal! "count" (Lean.toJson c.count) |>.setObjVal! "domain" (Json.str (Flock.hexOf dom)))

/-- An opening's JSON to its own `toJson`. -/
def openingPiece (j : Json) : V Json := pure <| refusing do pure (← LiveOpening.ofJson j).toJson

/-- `{"after", "commitment", "index", "label", "opening"}` to `live_reason`'s. -/
def reasonPiece (j : Json) : V Json := do
  request j #["after", "commitment", "index", "label", "opening"]
  let .str label := getD j "label" | throw "label is not a string"
  let some index := int? (getD j "index") | throw "index is not an integer"
  let after ← lowerHex (getD j "after") "after"
  pure <| refusing do
    return reasonJson (← liveReason (← LiveCommitment.ofJson (getD j "commitment"))
      (← LiveOpening.ofJson (getD j "opening")) label index after)

/-- `{"commitment", "label", "openings", "roots"}` to `order_reason`'s. -/
def orderPiece (j : Json) : V Json := do
  request j #["commitment", "label", "openings", "roots"]
  let .str label := getD j "label" | throw "label is not a string"
  let .arr roots := getD j "roots" | throw "roots is not a list"
  let roots ← roots.toList.mapM (lowerHex · "a root")
  let .arr openings := getD j "openings" | throw "openings is not a list"
  pure <| refusing do
    return reasonJson (← orderReason (← LiveCommitment.ofJson (getD j "commitment")) label roots
      (← openings.toList.mapM LiveOpening.ofJson))

end SampledProofs.Challenger
