import Flock.Hm96
import PouwCommit.Frame
import PouwCommit.Py

/-!
# The epoch coins' check (`verity.core.protocols.pouw.coins`)

The challenger commits to epoch `e`'s coins, one 32-byte coin per label of `LABELS`, before the epoch:

    x_l     = SHA-512("verity/pouw/epoch-coin/v1\0" ‖ coin_l)
    value_l = hm96-sha512 tree_leaf(key, commit_string(key, x_l, salt_l))           salt_l 192 bytes, the pinned key
    root    = the frame-v3-sha512 tree over (value_salt, value_draw) at positions 0, 1: binding
              tagged("verity/pouw/epoch-coins/v1", e), owner -1, leaf schema "verity/pouw/epoch-coins/v1"

`coinReason` is `coin_reason`: why an opening is not epoch `e`'s `label` coin, opened after what the verifier recorded and
opening against the root, in Python's order and words; `none` when it is. It throws where Python raises: a label not in
`LABELS`, and an opening path holding an item of neither 32 nor 64 bytes (`Opening`'s constructor). The JSON forms are read
as `CoinCommitment.from_json` and `CoinOpening.from_json` read them, refusing what they refuse. Lean also refuses, where
Python would compare it, an opening whose epoch is not a JSON integer or whose label is not a string.
-/

namespace PouwCommit

open Lean Flock

def SCHEME : String := "verity/pouw/epoch-coins/v1"
def COIN_TAG : ByteArray := "verity/pouw/epoch-coin/v1".toUTF8.push 0
def LABELS : List String := ["salt", "draw"]
def COIN_BYTES : Nat := 32

/-- The epoch's coin tree: `CommitmentDomain(tagged(SCHEME, e), -1, RangeIndexedDomain(2), "sha512")`. -/
def coinDomain (epoch : Nat) : Domain :=
  ⟨.sha512, 2, frameHash .sha512 "domain"
    [tagged SCHEME [.int epoch], natBytes 1, rangeIdentity .sha512 2, natBytes 2]⟩

/-- `leaf_value`: the committed value of one coin. -/
def leafValue (coin salt : ByteArray) : ByteArray := Hm96.Default512.leaf (Sha512.hash (COIN_TAG ++ coin)) salt

structure CoinCommitment where
  epoch : Nat
  root : ByteArray

structure CoinOpening where
  epoch : Int
  label : String
  after : ByteArray
  coin : ByteArray
  salt : ByteArray
  path : List ByteArray

/-- `CoinCommitment.from_json`. -/
def CoinCommitment.ofJson (j : Json) : Except String CoinCommitment := do
  let scheme := match j with
    | .obj kv => kv.get? "scheme"
    | _ => none
  if scheme != some (.str SCHEME) then throw s!"a coin commitment's scheme is {pyRepr SCHEME}"
  let epoch ← match jsonInt? (← field j "epoch") with
    | some e => if e < 0 then throw "an epoch is a nonnegative integer" else pure e.toNat
    | none => throw "an epoch is a nonnegative integer"
  let root ← hexField j "root"
  if root.size != 64 then throw "a coin commitment's root is 64 bytes"
  pure ⟨epoch, root⟩

/-- `CoinOpening.from_json`. -/
def CoinOpening.ofJson (j : Json) : Except String CoinOpening := do
  let epoch ← match jsonInt? (← field j "epoch") with
    | some e => pure e
    | none => throw "an opening's epoch is an integer"
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
  pure ⟨epoch, label, after, coin, salt, path⟩

/-- `coin_reason` (module docstring). -/
def coinReason (c : CoinCommitment) (o : CoinOpening) (label : String) (after : ByteArray) :
    Except String (Option String) := do
  let some rank := LABELS.idxOf? label | throw "an epoch's coins are ('salt', 'draw')"
  if o.epoch != c.epoch then return some s!"coin is epoch {o.epoch}'s, not epoch {c.epoch}'s"
  if o.label != label then return some s!"coin is the epoch's {pyRepr o.label} coin, not its {pyRepr label} coin"
  if o.coin.size != COIN_BYTES || o.salt.size != 192 then return some "coin opening is malformed"
  let value := leafValue o.coin o.salt
  checkPath o.path
  let d := coinDomain c.epoch
  if !(← d.verify c.root rank value o.path fun v => pure (d.leaf rank SCHEME v)) then
    return some "coin does not open against the epoch's coin commitment"
  if o.after != after then return some "coin was not opened after what the verifier recorded"
  return none

end PouwCommit
