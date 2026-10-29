/-
Supplied values (`one_stage/registered.py`'s `supplied_root`): a registered value the verifier fills itself, from what it
holds before the draw, so the prover never chooses it.

* `sources`: the verifier's `{value: [[source, ...], ...]}`, each source `{"port": P}` (the record's `roots` entry of port
  P), `{"value": V}` (its `registered` entry of value V, never a supplied value's), `{"bytes": hex}` (the verifier's own
  bytes), `{"position": "K"}` (K, a decimal below the record's population, after every unit of the log's records,
  `logBase`, as a u64 big-endian) or `{"window": hex}` (`windowSalt`: SHA-512 of `WINDOW_SALT`, the coin, the verifier's
  binding, its stream's digest and its prover's identity as hex, and the record's window as canonical JSON, NUL between
  them, a window that is an object of the canonical domain; none without a binding);
* a row is its sources' bytes in order (a root the record names, its last entry, as R3 and R7 read a name twice; hex
  non-empty lowercase of whole bytes), read as the parameter's leaves of `w` bits (`bytesLeaves`: each big-endian
  32-bit half, its `32 / w` leaves least significant first, so a digest is 16 leaves of 32 bits or 512 of one);
* `root`: the rows, as many as the verifier's own entry's `leaves`, each of its `words` leaves, committed as any registered
  value's rows (`Registered.commit`: `sha512/row/v1` of its u16 words, hm96-sha512 under the default key, the frame-v3-sha512
  tree under the value's domain) with row `i` salted by `salt`: SHAKE256 of `verity/one-stage/supplied-salt/v0`, the
  program's SHA-512 hex, the value's name and `i` (u32 little-endian), NUL between them, 192 bytes.
-/
import Flock.Canon
import Flock.Firewall
import Flock.Keccak
import Flock.Program
import Flock.Registered
import SampledProofs.Registered

namespace SampledProofs.Supplied

open Lean (Json)
open Flock (V)

def SUPPLIED_SALT : String := "verity/one-stage/supplied-salt/v0"
def WINDOW_SALT : String := "verity/one-stage/window-salt/v1"
def SOURCES : List String := ["port", "value", "bytes", "position", "window"]
def SALT_BYTES : Nat := 192

/-- A supplied value's rows, each its sources `(kind, name)` in order. -/
abbrev Rows := Array (Array (String × String))

/-- `registered.supplied_salts`' row `i`. -/
def salt (program value : String) (i : Nat) : ByteArray :=
  Flock.Keccak.shake256
    ((((SUPPLIED_SALT.toUTF8.push 0 ++ program.toUTF8).push 0 ++ value.toUTF8).push 0) ++ Flock.le32 i.toUInt32) SALT_BYTES

private def source (v : String) : Json → V (String × String)
  | .obj o => match o.toArray with
    | #[(k, .str n)] => if SOURCES.contains k then pure (k, n) else throw s!"supplied value {v}: a source is {k}"
    | _ => throw s!"supplied value {v}: a source is not one of port, value, bytes, position or window naming a string"
  | _ => throw s!"supplied value {v}: a source is not an object"

/-- `registered.sources`: the verifier's supplied values, in key order; an error on any other form. -/
def sources (j : Json) : V (Array (String × Rows)) := do
  let .obj kvs := j | throw "supplied values are not an object of rows"
  kvs.toArray.mapM fun (v, rows) => do
    let .arr rows := rows | throw s!"supplied value {v}: its rows are not a list"
    let rows ← rows.mapM fun r => match r with
      | .arr ss => ss.mapM (source v)
      | _ => throw s!"supplied value {v}: a row is not a list of sources"
    pure (v, rows)

/-- `registered.bytes_leaves`: `data` as leaves of `bits` bits, each big-endian 32-bit half's `32 / bits` least significant
first. -/
def bytesLeaves (data : ByteArray) (bits : Nat) : V (Array Nat) := do
  if !(0 < bits && bits ≤ 32 && 32 % bits == 0) then throw s!"leaves of {bits} bits hold no whole 32-bit halves"
  if data.size % 4 != 0 then throw s!"{data.size} bytes are no whole 32-bit halves"
  return (Array.range (data.size / 4)).flatMap fun i =>
    let h := (List.range 4).foldl (fun acc k => acc * 256 + (data.get! (4 * i + k)).toNat) 0
    (Array.range (32 / bits)).map fun j => h / 2 ^ (bits * j) % 2 ^ bits

/-- `registered.log_base`: the first position after the log's records, the sum of their populations. -/
def logBase (log : Array Json) : V Nat :=
  log.foldlM (fun base e => match int? (getD e "population") with
    | some n => if n < 0 then throw "a record of the log has no population" else pure (base + n.toNat)
    | none => throw "a record of the log has no population") 0

/-- `registered.window_salt`: the salt of the record's window under the verifier's coin and its binding
`(stream, prover)`. -/
def windowSalt (coin : ByteArray) (binding : Option (String × String)) (window : Json) : V ByteArray := do
  let some (stream, prover) := binding
    | throw "a window salt is bound to a stream and a prover: the verifier holds no binding"
  if !(window matches .obj _) || (Flock.Program.domainError window).isSome then
    throw "the record's window is no canonical JSON object"
  let head := ((WINDOW_SALT.toUTF8.push 0 ++ coin).push 0 ++ stream.toUTF8).push 0 ++ prover.toUTF8
  return Flock.Sha512.hash (head.push 0 ++ (← Flock.canon window).toUTF8)

private def decimal? (s : String) : Option Nat :=
  if s == "0" || (!s.isEmpty && s.toList.all Char.isDigit && s.front != '0') then some s.toNat! else none

/-- A source's bytes: the root the record names for it, the verifier's own, a position after the log's, or the verifier's
coin bound to its binding and the record's window; an error where there is none. -/
def sourceBytes (rec : Json) (log : Array Json) (binding : Option (String × String)) (supplied : Array String)
    (value : String) : String × String → V ByteArray
  | (kind, name) => do
    if kind == "position" then
      let some k := decimal? name | throw s!"supplied value {value}: position {name} is no decimal"
      let some n := int? (getD rec "population") | throw s!"supplied value {value}: the record has no population"
      if n ≤ (k : Int) then throw s!"supplied value {value}: position {name} is no unit of the record's"
      let p := (← logBase log) + k
      if p ≥ 2 ^ 64 then throw s!"supplied value {value}: position {p} is no u64"
      return Flock.be64 p.toUInt64
    if kind == "window" then
      if name.isEmpty then throw s!"supplied value {value}: window {name} is no bytes"
      return (← windowSalt (← Flock.unhex name) binding (getD rec "window"))
    let text ← if kind == "bytes" then pure (Json.str name) else do
      if kind == "value" && supplied.contains name then
        throw s!"supplied value {value} reads supplied value {name}'s root"
      let (field, key) := if kind == "port" then ("roots", "port") else ("registered", "value")
      let es := match getD rec field with | .arr es => es | _ => #[]
      let some e := es.reverse.find? (fun e => jeq (getD e key) (Json.str name))
        | throw s!"supplied value {value}: the record names no {key} {name}"
      pure (getD e "root")
    let .str s := text | throw s!"supplied value {value}: {kind} {name} is not hex"
    if s.isEmpty then throw s!"supplied value {value}: {kind} {name} is no bytes"
    Flock.unhex s

/-- `registered.supplied_rows`: each row its sources' bytes as the parameter's leaves. -/
def rows (rec : Json) (log : Array Json) (binding : Option (String × String)) (ps : Array Registered.Param)
    (supplied : Array String) (value : String) (rs : Rows) : V (Array (Array Nat)) := do
  let some p := ps.find? (·.name == value) | throw s!"{value} is not a parameter of the program"
  let [w] := Registered.widthSet p.ty | throw s!"parameter {value}: its leaves are not one width"
  rs.mapM fun r => do
    let data ← r.foldlM (fun acc s => return acc ++ (← sourceBytes rec log binding supplied value s)) ByteArray.empty
    bytesLeaves data w

/-- A registered row's u16 words: each leaf `wordBits / 16` words, low first, zero-padded to whole SHA-512 blocks. -/
def rowWords (row : Array Nat) (wordBits : Nat) : Array Nat :=
  let ws := row.flatMap fun x => (Array.range (wordBits / 16)).map fun j => x / 2 ^ (16 * j) % 2 ^ 16
  ws ++ Array.replicate (Flock.Registered.rowWords row.size wordBits - ws.size) 0

/-- `registered.supplied_root`: the root R7 holds a supplied value's entry to (hex), the rows of `rs` committed under
`salt` at the verifier's own entry `want`; an error where a source names nothing or the rows are not its shape. -/
def root (rec : Json) (log : Array Json) (binding : Option (String × String)) (ps : Array Registered.Param)
    (program : String) (supplied : Array String) (value : String) (rs : Rows) (want : Registered.Entry) : V String := do
  let got ← rows rec log binding ps supplied value rs
  if got.size != want.leaves || got.any (·.size != want.words) then
    throw s!"supplied value {value}: rows of {got.map (·.size)} leaves, not {want.leaves} of {want.words}"
  let dom ← Registered.domain program value want.words want.leaves want.wordBits
  let leaves := got.mapIdx fun i r =>
    Flock.Registered.rowLeaf dom i (Flock.Hm96.Default512.treeLeaf
      (Flock.Hm96.Default512.commitString (Flock.Firewall.rowDigest (rowWords r want.wordBits)) (salt program value i)))
  return Flock.hexOf (Flock.FV3.root dom leaves Flock.Sha512.hash)

end SampledProofs.Supplied
