module

public import Flock.ZkServer
public import Commitments.SessionCoins
public import Lms.Verify
public import Spot.Registered
meta import Flock.ZkServer
meta import Spot.Registered

/-!
Supplied values (`one_stage/registered.py`'s `supplied_root`): a registered value the verifier fills itself, from what it
holds before the sample, so the prover never chooses it.

* `sources`: the verifier's `{value: [[source, ...], ...]}`, each source `{"port": P}` (the record's `roots` entry of port
  P), `{"value": V}` (its `registered` entry of value V, never a supplied value's), `{"domain": V}` (that entry's
  domain id), `{"bytes": hex}` (the verifier's own
  bytes), `{"position": "K"}` (K, a decimal below the record's population, after every unit of the log's records,
  `logBase`, as a u64 big-endian) or `{"window": hex}` (`runSalt`: SHA-512 of `RUN_SALT`, the coin, the verifier's
  binding, its stream's digest and its prover's identity as hex, and the record's window as canonical JSON, NUL between
  them, a window that is an object of the canonical domain; none without a binding);
* a row is its sources' bytes in order (a root the record names, its last entry, as R3 and R7 read a name twice; hex
  non-empty lowercase of whole bytes), read as the parameter's leaves of `w` bits (`bytesLeaves`: each big-endian
  32-bit half, its `32 / w` leaves least significant first, so a digest is 16 leaves of 32 bits or 512 of one; a row of
  two bytes is one 16-bit half, its `16 / w` leaves: `halfLeaves`, as `SecondStage.VERDICT_ONE_LEAF`), except
  where the parameter's leaves are whole `Digest`s (`digests`: `Array<8, Array<64, Value<1>>>` or an array of them),
  read as a circuit's `Digest` holds them (`digestLeaves`: each big-endian 64-bit word's 64 bits least significant
  first);
* `root`: the rows, as many as the verifier's own entry's `leaves`, each the parameter's leaves of one row of its `words`
  leaves in the value's row format (`schemas`, `hm96-sha512/row/v1` where they name none; `Registered.rowLeaves`),
  committed as any registered value's rows in that format (`Registered.commit`: `sha512/row/v1` of its u16 words, or
  under `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1` the bit row of its leaves packed into `wordBits`-bit words,
  `pack`; hm96-sha512 under the default key, the frame-v3-sha512 tree under the value's domain, its leaves framed by the
  format) with row `i` salted by `salt`: 192 bytes of SHA-512 in counter mode over `verity/one-stage/supplied-salt/v1`,
  the program's SHA-512 hex, the value's name and `i` (u32 little-endian), NUL between them.

**Scope.** Nothing reads the storage session's three bodies (`pous.session.round_coin/v1`, `.header/v1`, `.context/v1`)
through the door yet: they are definitions with their refusals, held to the Python's by `spot-difftest`. The verifier holds
a session in two entries by when each of its parts exists. `commitments` holds what exists before the first coin: the
header's fields (the session id, the coin root, the hash of the coin key, the blocks, the rounds `k`, the limit, the answer's
size, the binding), the attester's registered key and the `place` the verifier registered it at. `coins` holds what the
session releases after it: the openings and the coin key itself (a round's coin), how it ended and its number of rounds (the
header's). The order rule, `order`, counts `coins` alone as coins and holds the `place` of each held commitment before the
place of every coin read beside it. A round's coin is held to the coin root of the commitment the verifier holds, never to
one the coin carries, and to the coin key whose hash that commitment holds; a session that did not end complete with all its
rounds has no statement.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Supplied

open Lean (Json)
open Flock (V)

def SUPPLIED_SALT : String := "verity/one-stage/supplied-salt/v1"
def RUN_SALT : String := "verity/one-stage/window-salt/v1"
def SOURCES : List String := ["port", "value", "domain", "bytes", "position", "window"]
def SALT_BYTES : Nat := 192

/-- A supplied value's rows, each its sources `(kind, name)` in order. -/
abbrev Rows := Array (Array (String × String))

/-- `registered_value.supplied_salts`' row `i`: SHA-512 in counter mode, `SHA-512(msg ‖ u32be c)` for `c < SALT_BYTES / 64`,
each output whole. -/
def salt (program value : String) (i : Nat) : ByteArray :=
  let msg := (((SUPPLIED_SALT.toUTF8.push 0 ++ program.toUTF8).push 0 ++ value.toUTF8).push 0) ++ Flock.le32 i.toUInt32
  (List.range (SALT_BYTES / 64)).foldl (fun acc c => acc ++ Flock.Sha512.hash (msg ++ Flock.be32 c.toUInt32))
    ByteArray.empty

private def source (v : String) : Json → V (String × String)
  | .obj o => match o.toArray with
    | #[(k, .str n)] => if SOURCES.contains k then pure (k, n) else throw s!"supplied value {v}: a source is {k}"
    | _ => throw s!"supplied value {v}: a source is not one of port, value, domain, bytes, position or window naming a string"
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

/-- `registered.half_leaves`: two bytes, one big-endian 16-bit half, as its `16 / bits` leaves of `bits` bits least
significant first: the row `SecondStage.VERDICT_ONE_LEAF` fills, one 16-bit leaf. -/
def halfLeaves (data : ByteArray) (bits : Nat) : V (Array Nat) := do
  if !(0 < bits && bits ≤ 16 && 16 % bits == 0) then throw s!"leaves of {bits} bits hold no whole 16-bit half"
  if data.size != 2 then throw s!"{data.size} bytes are not one 16-bit half"
  let h := (data.get! 0).toNat * 256 + (data.get! 1).toNat
  return (Array.range (16 / bits)).map fun j => h / 2 ^ (bits * j) % 2 ^ bits

/-- A row of a supplied input whose layout is `le` leads with `(LE, "")` (`ofMark`): the rest's bytes read by `leLeaves`.
No scheme names it. -/
def LE : String := "le"

/-- `registered.le_leaves`, §5.4's `le`: `data` as one little-endian bit string (bit `t` is bit `t mod 8` of byte
`t / 8`), leaf `i` its bits `[i·bits, (i+1)·bits)`. -/
def leLeaves (data : ByteArray) (bits : Nat) : V (Array Nat) := do
  if !([1, 8, 16, 32, 64].contains bits && (8 * data.size) % bits == 0) then
    throw s!"{data.size} bytes are no whole leaves of {bits} bits"
  let x := data.toList.reverse.foldl (fun acc b => acc * 256 + b.toNat) 0
  return (Array.range (8 * data.size / bits)).map fun i => x / 2 ^ (bits * i) % 2 ^ bits

/-- A row's bytes as the parameter's leaves of `bits` bits: one 16-bit half's (`halfLeaves`) where it is two bytes,
else its 32-bit halves' (`bytesLeaves`). -/
def leavesOf (data : ByteArray) (bits : Nat) : V (Array Nat) :=
  if data.size == 2 then halfLeaves data bits else bytesLeaves data bits

/-- `registered.digests`: a parameter whose leaves are whole `Digest`s, eight SHA-512 words of 64 one-bit leaves each
(`gates.rowleaf.digest_bits`'s order), or an array of them. -/
def digests : Flock.Program.Ty → Bool
  | .array 8 (.array 64 (.value 1)) => true
  | .array _ (.array 8 (.array 64 (.value 1))) => true
  | _ => false

/-- `registered.digest_leaves`: `data` as whole `Digest`s' leaves, each big-endian 64-bit word's 64 bits least
significant first. -/
def digestLeaves (data : ByteArray) : V (Array Nat) := do
  if data.size % 64 != 0 then throw s!"{data.size} bytes are no whole digests"
  return (Array.range (data.size / 8)).flatMap fun i =>
    let h := (List.range 8).foldl (fun acc k => acc * 256 + (data.get! (8 * i + k)).toNat) 0
    (Array.range 64).map fun j => h / 2 ^ j % 2

/-- `registered.log_base`: the first position after the log's records, the sum of their populations. -/
def logBase (log : Array Json) : V Nat :=
  log.foldlM (fun base e => match int? (getD e "population") with
    | some n => if n < 0 then throw "a record of the log has no population" else pure (base + n.toNat)
    | none => throw "a record of the log has no population") 0

/-- `registered.window_salt`: the salt of the record's window under the verifier's coin and its binding
`(stream, prover)`. -/
def runSalt (coin : ByteArray) (binding : Option (String × String)) (window : Json) : V ByteArray := do
  let some (stream, prover) := binding
    | throw "a window salt is bound to a stream and a prover: the verifier holds no binding"
  if !(window matches .obj _) || (Flock.Program.domainError window).isSome then
    throw "the record's window is no canonical JSON object"
  let head := ((RUN_SALT.toUTF8.push 0 ++ coin).push 0 ++ stream.toUTF8).push 0 ++ prover.toUTF8
  return Flock.Sha512.hash (head.push 0 ++ (← Flock.canon window).toUTF8)

private def decimal? (s : String) : Option Nat :=
  if s == "0" || (!s.isEmpty && s.toList.all Char.isDigit && s.front != '0') then some s.toNat! else none

/-- The sources only a mark's body names (`body`): a table's root and domain id as the verifier holds them, never the
record's. -/
def HELD_ROOT : String := "held-root"
def HELD_DOMAIN : String := "held-domain"

/-- A source's bytes: the root the record names for it, the verifier's own, a position after the log's, the verifier's
coin bound to its binding and the record's window, or a held table's root or domain id (`held`: `(value, root, domain)`
of each table the verifier holds, its accepted pins); an error where there is none. -/
def sourceBytes (rec : Json) (log : Array Json) (binding : Option (String × String))
    (held : Array (String × String × String)) (supplied : Array String)
    (value : String) : String × String → V ByteArray
  | (kind, name) => do
    if kind == HELD_ROOT || kind == HELD_DOMAIN then
      let some (_, root, domain) := held.find? (·.1 == name)
        | throw s!"supplied value {value}: the verifier holds no value {name}"
      return (← Flock.unhex (if kind == HELD_ROOT then root else domain))
    if kind == "position" then
      let some k := decimal? name | throw s!"supplied value {value}: position {name} is no decimal"
      let some n := int? (getD rec "population") | throw s!"supplied value {value}: the record has no population"
      if n ≤ (k : Int) then throw s!"supplied value {value}: position {name} is no unit of the record's"
      let p := (← logBase log) + k
      if p ≥ 2 ^ 64 then throw s!"supplied value {value}: position {p} is no u64"
      return Flock.be64 p.toUInt64
    if kind == "window" then
      if name.isEmpty then throw s!"supplied value {value}: window {name} is no bytes"
      return (← runSalt (← Flock.unhex name) binding (getD rec "window"))
    let text ← if kind == "bytes" then pure (Json.str name) else do
      if (kind == "value" || kind == "domain") && supplied.contains name then
        throw s!"supplied value {value} reads supplied value {name}'s root"
      let (field, key) := if kind == "port" then ("roots", "port") else ("registered", "value")
      let es := match getD rec field with | .arr es => es | _ => #[]
      let some e := es.reverse.find? (fun e => jeq (getD e key) (Json.str name))
        | throw s!"supplied value {value}: the record names no {key} {name}"
      pure (getD e (if kind == "domain" then "domain" else "root"))
    let .str s := text | throw s!"supplied value {value}: {kind} {name} is not hex"
    if s.isEmpty then throw s!"supplied value {value}: {kind} {name} is no bytes"
    Flock.unhex s

/-- `registered.supplied_rows`: each row its sources' bytes as the parameter's leaves, `Digest`s' where its leaves are
whole digests (`digests`). -/
def rows (rec : Json) (log : Array Json) (binding : Option (String × String)) (held : Array (String × String × String))
    (ps : Array Registered.Param) (supplied : Array String) (value : String) (rs : Rows) : V (Array (Array Nat)) := do
  let some p := ps.find? (·.name == value) | throw s!"{value} is not a parameter of the program"
  let [w] := Registered.widthSet p.ty | throw s!"parameter {value}: its leaves are not one width"
  rs.mapM fun r => do
    let le := r[0]?.map (·.1) == some LE
    let data ← (if le then r.extract 1 r.size else r).foldlM
      (fun acc s => return acc ++ (← sourceBytes rec log binding held supplied value s)) ByteArray.empty
    if le then leLeaves data w else if digests p.ty then digestLeaves data else leavesOf data w

/-- A registered row's u16 words: each leaf `wordBits / 16` words, low first, zero-padded to whole SHA-512 blocks. -/
def rowWords (row : Array Nat) (wordBits : Nat) : Array Nat :=
  let ws := row.flatMap fun x => (Array.range (wordBits / 16)).map fun j => x / 2 ^ (16 * j) % 2 ^ 16
  ws ++ Array.replicate (Flock.Registered.rowWords row.size wordBits - ws.size) 0

/-- `registered.bit_words`: a row of `bits`-bit leaves as a bit row's `wordBits`-bit leaves, its bits in order, each
leaf's least significant first. -/
def pack (bits wordBits : Nat) (row : Array Nat) : Array Nat :=
  let bit (i : Nat) : Nat := row.getD (i / bits) 0 / 2 ^ (i % bits) % 2
  (Array.range (row.size * bits / wordBits)).map fun j =>
    (List.range wordBits).foldl (fun acc t => acc + bit (j * wordBits + t) * 2 ^ t) 0

/-- `registered_value.row_digest`: a registered row's `x` in `schema`, `sha512/row/v1` of its u16 words (`rowWords`), else the
bit row of its u16 words, little-endian, under `sha512/row/v2` or `sha512/row-seg/v1`. -/
def rowDigest (schema : String) (row : Array Nat) (wordBits : Nat) : ByteArray :=
  if schema == Registered.SCHEMA then Flock.ZkServer.rowDigest (rowWords row wordBits) else
  let ws := row.flatMap fun x => (Array.range (wordBits / 16)).map fun j => x / 2 ^ (16 * j) % 2 ^ 16
  let bytes := ws.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8) ByteArray.empty
  if schema == Flock.HmRow.ROW_SCHEMA_V2 then Flock.Sha512.hash (Flock.HmRow.rowPrefixV2 (16 * ws.size) ++ bytes)
  else Flock.HmRow.segDigest (16 * ws.size) bytes

/-- `registered.cut`: `got`'s one row of `n * k` leaves as `n` rows of `k` in order, where `n > 1`; else `got`. -/
def cut (got : Array (Array Nat)) (n k : Nat) : Array (Array Nat) :=
  match got with
  | #[r] => if n > 1 && r.size == n * k then (Array.range n).map fun i => r.extract (i * k) (i * k + k) else got
  | _ => got

#guard cut #[#[1, 2, 3]] 3 1 == #[#[1], #[2], #[3]]
#guard cut #[#[1, 2, 3]] 2 1 == #[#[1, 2, 3]]
#guard cut #[#[1, 2], #[3, 4]] 4 1 == #[#[1, 2], #[3, 4]]

/-- `registered.supplied_root`: the root R7 holds a supplied value's entry to (hex), the rows of `rs` in the value's row
format of `schemas` committed under `salt` at the verifier's own entry `want`, a value given as one row of all their
leaves (a mark's, `ofMark`) cut into them (`cut`); an error where a source names nothing or the rows are not its
shape. -/
def root (rec : Json) (log : Array Json) (binding : Option (String × String)) (held : Array (String × String × String))
    (ps : Array Registered.Param) (program : String) (supplied : Array String) (schemas : Array (String × String))
    (value : String) (rs : Rows) (want : Registered.Entry) : V String := do
  let given ← rows rec log binding held ps supplied value rs
  let schema := (schemas.find? (·.1 == value)).elim Registered.SCHEMA (·.2)
  let some p := ps.find? (·.name == value) | throw s!"{value} is not a parameter of the program"
  let k ← Registered.rowLeaves p schema want.words want.wordBits
  let got := cut given want.leaves k
  if got.size != want.leaves || got.any (·.size != k) then
    throw s!"supplied value {value}: rows of {got.map (·.size)} leaves, not {want.leaves} of {k}"
  let [w] := Registered.widthSet p.ty | throw s!"parameter {value}: its leaves are not one width"
  let packed := if schema == Registered.SCHEMA then got else got.map (pack w want.wordBits)
  let dom ← Registered.domain program value want.words want.leaves want.wordBits
  let leaves := packed.mapIdx fun i r =>
    Flock.Registered.leafAs schema dom i (Flock.Hm96.Default512.treeLeaf
      (Flock.Hm96.Default512.commitString (rowDigest schema r want.wordBits) (salt program value i)))
  return Flock.hexOf (Flock.FV3.root dom leaves Flock.Sha512.hash)

/-! The registry (`verity/circuits/PROTOCOL.md` §5.4): every function a supplied input of C may name, and the bodies the
verifier evaluates. C names the function in its input's mark, under its digest, so no scheme passes one. -/

/-- Every function a mark may name, the format's closed list (`verity.circuits.supplied.FUNCTIONS` names the same). -/
def FUNCTIONS : List String :=
  ["pous.session.round_coin/v1", "pous.session.header/v1", "pous.session.context/v1", "pous.session.message/v1",
   "pous.setup.salt/v1",
   "poub.read.pass_seed/v1", "poub.read.pass_root/v1", "window.salt/v1", "pouw.pearlplus.salt/v1", "registered.root/v1",
   "registered.domain/v1", "record.root/v1", "log.base/v1", "const.u16/v1"]
def LAYOUTS : List String := ["le", "halves", "digest"]
/-- A supplied value is short: at most this many bits, whatever the population; what grows with it is computed in C. -/
def SHORT_BITS : Nat := 65536
def SUPPLIED_PREFIX : String := "SuppliedInput"

/-- An input's mark: its function, its arguments, its layout and its value's bits. -/
structure Mark where
  fn : String
  args : Array (String × Json)
  layout : String
  bits : Nat
deriving Inhabited

/-- `registered.mark_of`: a top-level input's mark `{"supplied", "args", "layout", "bits"}`; an error on any other form,
on a function the registry lacks and on a value that isn't short. -/
def markOf (name : String) (j : Json) : V Mark := do
  let .obj kvs := j | throw s!"supplied input {name}: its mark is not an object"
  let ks := kvs.toArray.map (·.1)
  if ks.qsort (· < ·) != #["args", "bits", "layout", "supplied"] then
    throw s!"supplied input {name}: its mark is \{supplied, args, layout, bits}, not {ks.toList}"
  let .str fn := getD j "supplied" | throw s!"supplied input {name}: its function is not a string"
  let .obj args := getD j "args" | throw s!"supplied input {name}: its args are not an object"
  let .str layout := getD j "layout" | throw s!"supplied input {name}: its layout is not a string"
  let some bits := nat? (getD j "bits") | throw s!"supplied input {name}: its bits are not a natural number"
  if !FUNCTIONS.contains fn then throw s!"supplied input {name}: {fn} is no supplied function"
  if !LAYOUTS.contains layout then throw s!"supplied input {name}: {layout} is no layout"
  if bits == 0 || bits > SHORT_BITS then
    throw s!"supplied input {name}: {bits} bits is no short value (at most {SHORT_BITS}): a value that grows with the population is computed in C"
  return { fn, args := args.toArray, layout, bits }

private def callee (callees : Array Json) (n : Json) : String :=
  match nat? (getD n "fn") with
  | some i => match callees[i]? with | some (.str s) => s | _ => ""
  | none => ""

/-- The marks of the program descriptor `desc`'s top-level `inputs` (`[name, nodes]`, a supplied one's
`[name, nodes, mark]`), by input, each marked input's root nodes all `SuppliedInput<w>_v1` and no other input's one; none
where it has no `inputs`. -/
def marks (desc : Json) : V (Array (String × Mark)) := do
  let .arr xs := getD desc "inputs" | return #[]
  let .str root := getD desc "root" | throw "the program names no root"
  let .arr nodes := getD (getD (getD (getD desc "definitions") root) "body") "nodes"
    | throw "the program's root has no nodes"
  let .arr callees := getD desc "callees" | throw "the program has no callees"
  let mut k := 0
  let mut out := #[]
  for x in xs do
    let .arr e := x | throw "an input is not [name, nodes] or [name, nodes, mark]"
    let some (Json.str name) := e[0]? | throw "an input is not [name, nodes] or [name, nodes, mark]"
    let some n := e[1]?.bind nat? | throw s!"input {name}: its nodes are not a natural number"
    let ids := (nodes.extract k (k + n)).map (callee callees)
    if ids.size != n then throw s!"input {name}: the root has no nodes {k}..{k + n - 1}"
    let supplied := ids.map (·.startsWith SUPPLIED_PREFIX)
    match e[2]?, e.size with
    | none, 2 =>
      if supplied.any id then throw s!"input {name}: supplied input nodes and no mark naming their function"
    | some m, 3 =>
      if !supplied.all id then throw s!"input {name}: marked supplied, and its nodes are not all supplied inputs"
      out := out.push (name, ← markOf name m)
    | _, _ => throw s!"input {name}: an input is [name, nodes] or [name, nodes, mark]"
    k := k + n
  return out

/-- `supplied.registered_values`: the values a record of the program registers, `values` as given, then each table a
mark of an anchor's names (`registered.root/v1` or `registered.domain/v1`, its `value`), in the marks' order, once and
not where it is among `values`: a table is there only because a mark in C names it. An error where `values` names a value
twice. -/
def registeredValues (values : Array String) (marks : Array (String × Mark)) : V (Array String) := do
  let _ ← values.foldlM (init := (#[] : Array String)) fun seen v =>
    if seen.contains v then throw s!"values: {v} is there twice: each once" else pure (seen.push v)
  return marks.foldl (init := values) fun out (_, m) =>
    match ["registered.root/v1", "registered.domain/v1"].contains m.fn, m.args.find? (·.1 == "value") with
    | true, some (_, .str t) => if out.contains t then out else out.push t
    | _, _ => out

private def anchorMark (fn t : String) : String × Mark :=
  (s!"{t}/{fn}", { fn := s!"registered.{fn}/v1", args := #[("value", .str t)], layout := "digest", bits := 512 })

#guard (registeredValues #["v", "u"] #[]).toOption == some #["v", "u"]
#guard (registeredValues #["v"] #[anchorMark "domain" "t", anchorMark "root" "t"]).toOption == some #["v", "t"]
#guard (registeredValues #["v", "v"] #[]).toOption.isNone

private def argNat (v : String) (m : Mark) (a : String) : V Nat := do
  if m.args.map (·.1) != #[a] then throw s!"supplied input {v}: {m.fn} takes the args [{a}]"
  let some x := (m.args.find? (·.1 == a)).bind (nat? ·.2) | throw s!"supplied input {v}: {m.fn}'s {a} is no natural"
  return x

private def argStr (v : String) (m : Mark) (a : String) : V String := do
  if m.args.map (·.1) != #[a] then throw s!"supplied input {v}: {m.fn} takes the args [{a}]"
  let some (_, .str x) := m.args.find? (·.1 == a) | throw s!"supplied input {v}: {m.fn}'s {a} is no string"
  if x.isEmpty then throw s!"supplied input {v}: {m.fn}'s {a} is no string"
  return x

private def coin (coins : Array (String × ByteArray)) (v : String) (m : Mark) : V ByteArray := do
  if !m.args.isEmpty then throw s!"supplied input {v}: {m.fn} takes no args"
  let some (_, c) := coins.find? (·.1 == m.fn) | throw s!"supplied input {v}: no statement: the verifier holds no coin for {m.fn}"
  return c

private def noStatement (v f why : String) : String := s!"supplied input {v}: no statement: {f}'s {why}"

private def hexOpt : Json → Option ByteArray
  | .str s => (Flock.unhex s).toOption
  | _ => none

private def numberOk (t : Array UInt8) : Bool :=
  t == #[48] || (t.size > 0 && 49 ≤ t[0]!.toNat && t[0]!.toNat ≤ 57 && t.all fun c => 48 ≤ c.toNat && c.toNat ≤ 57)

/-- `registered.canonical_numbers`: every number in the JSON text `b` (read as JSON already) is `0` or digits with no
leading zero: no sign, fraction or exponent, which Lean's reader takes as an integer where Python's reads a float or reads
`-0` as 0. A number is a run of `0-9 + - . e E` that starts outside a string at one of `0-9 + - .`. -/
private def canonicalNumbers (b : ByteArray) : Bool := Id.run do
  let mut inStr := false
  let mut esc := false
  let mut tok : Array UInt8 := #[]
  for c in b.toList do
    if inStr then
      if esc then esc := false
      else if c == 92 then esc := true
      else if c == 34 then inStr := false
    else
      let start := (48 ≤ c.toNat && c.toNat ≤ 57) || c == 43 || c == 45 || c == 46
      if tok.size > 0 && (start || c == 101 || c == 69) then tok := tok.push c
      else
        if tok.size > 0 then
          if !numberOk tok then return false
          tok := #[]
        if c == 34 then inStr := true
        else if start then tok := #[c]
  return tok.isEmpty || numberOk tok

/-- What the verifier holds under function `f` in `tbl` (`kind`, `"commitment"` or `"coin"`), as a JSON object of
canonical numbers; "no statement" for what it holds none of, or that is no such object. -/
private def heldObject (tbl : Array (String × ByteArray)) (kind v f : String) : V Json := do
  let some (_, c) := tbl.find? (·.1 == f) | throw s!"supplied input {v}: no statement: the verifier holds no {kind} for {f}"
  let some s := String.fromUTF8? c | throw (noStatement v f s!"{kind} is no JSON")
  let .ok j := Flock.JsonParse.parse s | throw (noStatement v f s!"{kind} is no JSON")
  if !canonicalNumbers c then throw (noStatement v f s!"{kind} holds a number that is no canonical natural")
  let .obj _ := j | throw (noStatement v f s!"{kind} is no object")
  return j

/-- The held commitment's fields `hexes` (lowercase hex of whole bytes, the session id 32 bytes) and `nats` (naturals below
`2 ^ bits`), in that order of refusal. -/
private def sessionFields (v f : String) (rel : Json) (hexes : List String) (nats : List (String × Nat)) :
    V (Array (String × ByteArray) × Array (String × Nat)) := do
  let .obj c := getD rel "commitment" | throw (noStatement v f "commitment holds no commitment object")
  let cj := Json.obj c
  let hs ← hexes.toArray.mapM fun k => match hexOpt (getD cj k) with
    | some b =>
      if k == "session" && b.size != 32 then throw (noStatement v f s!"commitment's session is {b.size} bytes, not 32")
      else pure (k, b)
    | none => throw (noStatement v f s!"commitment's {k} is no lowercase hex")
  let ns ← nats.toArray.mapM fun (k, bits) => match nat? (getD cj k) with
    | some n => if n < 2 ^ bits then pure (k, n) else throw (noStatement v f s!"commitment's {k} is no u{bits}")
    | none => throw (noStatement v f s!"commitment's {k} is no u{bits}")
  return (hs, ns)

private def hexIn (xs : Array (String × ByteArray)) (k : String) : ByteArray := (xs.find? (·.1 == k)).elim ByteArray.empty (·.2)
private def natIn (xs : Array (String × Nat)) (k : String) : Nat := (xs.find? (·.1 == k)).elim 0 (·.2)

/-- The storage session's round coin: round `round`'s index, the last opening of the round the coin holds, opened under the
coin root of the commitment the verifier holds (never one the coin carries) with the coin key the coin releases, which
hashes to the commitment's `coin_key_sha512`. -/
private def roundCoin (coins commitments : Array (String × ByteArray)) (v : String) (m : Mark) :
    V (Array (String × String) × Nat) := do
  let f := m.fn
  let j ← argNat v m "round"
  let rel ← heldObject commitments "commitment" v f
  let (hs, ns) ← sessionFields v f rel ["session", "coin_root", "coin_key_sha512"] [("blocks", 64), ("k", 32)]
  let k := natIn ns "k"
  let blocks := natIn ns "blocks"
  if j ≥ 2 ^ 32 then throw (noStatement v f s!"round {j} is no u32")
  if j ≥ k then throw (noStatement v f s!"round {j} is no round of the session's {k}")
  let coin ← heldObject coins "coin" v f
  if (entries coin).any (fun (n, _) => n != "openings" && n != "coin_key") then
    throw (noStatement v f "coin holds more than its openings and its key: the coin root is the verifier's own")
  let some key := hexOpt (getD coin "coin_key") | throw (noStatement v f "coin's coin_key is no lowercase hex")
  if Flock.Sha512.hash key != hexIn hs "coin_key_sha512" then
    throw (noStatement v f "coin's key is not the one the commitment holds the hash of")
  let .arr ops := getD coin "openings" | throw (noStatement v f "coin holds no openings list")
  let some o := ops.findRev? (fun o => nat? (getD o "j") == some j)
    | throw (noStatement v f s!"coin holds no opening of round {j}")
  let bad := noStatement v f s!"opening of round {j} is malformed"
  let some index := nat? (getD o "index") | throw bad
  let some salt := hexOpt (getD o "salt") | throw bad
  let .arr pj := getD o "path" | throw bad
  let some path := pj.mapM hexOpt | throw bad
  if index ≥ 2 ^ 64 then throw (noStatement v f s!"round {j}'s index {index} is no u64")
  if index ≥ blocks then throw (noStatement v f s!"round {j}'s index {index} is no block of the session's {blocks}")
  if !Flock.SessionCoins.verify (hexIn hs "session") (hexIn hs "coin_root") key blocks k j index salt path then
    throw (noStatement v f s!"round {j}'s index {index} is not its committed coin")
  return (#[("bytes", Flock.hexOf (Flock.be32 j.toUInt32 ++ Flock.be64 index.toUInt64))], 12)

/-- The storage session's signed header: the held commitment's fields (the key by its hash) with `"ended": "complete"` and
`"rounds": k`, which the `ended` and `rounds` the session's release holds (the coin held under the function) are compared
with, so a late or short session has no statement. -/
private def header (coins commitments : Array (String × ByteArray)) (v : String) (m : Mark) :
    V (Array (String × String) × Nat) := do
  let f := m.fn
  let want ← argNat v m "head"
  let rel ← heldObject commitments "commitment" v f
  let (hs, ns) ← sessionFields v f rel ["session", "coin_root", "coin_key_sha512", "binding"]
    [("blocks", 64), ("k", 32), ("limit_ns", 64), ("block_bytes", 64)]
  let k := natIn ns "k"
  let released ← heldObject coins "coin" v f
  let .str ended := getD released "ended" | throw (noStatement v f "coin holds no ended string")
  let some rounds := nat? (getD released "rounds") | throw (noStatement v f "coin's rounds is no u32")
  if rounds ≥ 2 ^ 32 then throw (noStatement v f "coin's rounds is no u32")
  if ended != Flock.SessionCoins.COMPLETE then throw (noStatement v f "session did not end complete")
  if rounds != k then throw (noStatement v f s!"session has {rounds} rounds, not its {k}")
  let head := (Flock.SessionCoins.header (hexIn hs "session") (hexIn hs "coin_root") (hexIn hs "coin_key_sha512")
    (hexIn hs "binding") (natIn ns "blocks") k (natIn ns "limit_ns") (natIn ns "block_bytes")
    Flock.SessionCoins.COMPLETE k).toUTF8
  if head.size != want then throw (noStatement v f s!"header is {head.size} bytes, not {want}")
  let row := if head.size % 2 == 1 then head.push 0 else head
  return (#[("bytes", Flock.hexOf row)], row.size)

/-- The storage session's signed context: the held commitment's session id and the first 16 bytes of the held key. -/
def UNPINNED : String := "key is not one the configuration pins for the memory attester"
def BAD_SIGNATURE : String := "signature does not verify over the message under the pinned key"

/-- The held commitment's LMS public key, refused unless its SHA-512 is one of `pinned`, the configuration's pins of the
memory attester's keys: the session's own object never names the key it is checked under. -/
private def pinnedKey (rel : Json) (v f : String) (pinned : Array ByteArray) : V ByteArray := do
  let some key := hexOpt (getD rel "key") | throw (noStatement v f "commitment's key is no lowercase hex")
  if !pinned.contains (Flock.Sha512.hash key) then throw (noStatement v f UNPINNED)
  return key

private def context (commitments : Array (String × ByteArray)) (v : String) (m : Mark) (pinned : Array ByteArray) :
    V (Array (String × String) × Nat) := do
  let f := m.fn
  if !m.args.isEmpty then throw s!"supplied input {v}: {f} takes no args"
  let rel ← heldObject commitments "commitment" v f
  let (hs, _) ← sessionFields v f rel ["session"] []
  let session := hexIn hs "session"
  let ident := (← pinnedKey rel v f pinned).extract 0 16
  if ident.size != 16 then throw (noStatement v f s!"key is {ident.size} bytes, not an LMS identifier's 16")
  let ctx := Flock.SessionCoins.signedContext ident session
  return (#[("bytes", Flock.hexOf ctx)], ctx.size)

/-- The storage session's signed message: the `message` the session releases (the coin held under the function,
`{message, signature}`), the 64-byte salted commitment the memory attester signed, released in the clear with its
`signature`, which verifies over it under the pinned key the commitment holds, at the deployed parameters (`Lms.deployed`). -/
private def message (coins commitments : Array (String × ByteArray)) (v : String) (m : Mark) (pinned : Array ByteArray) :
    V (Array (String × String) × Nat) := do
  let f := m.fn
  if !m.args.isEmpty then throw s!"supplied input {v}: {f} takes no args"
  let rel ← heldObject commitments "commitment" v f
  let (hs, _) ← sessionFields v f rel ["session"] []
  let session := hexIn hs "session"
  let key ← pinnedKey rel v f pinned
  let released ← heldObject coins "coin" v f
  let some msg := hexOpt (getD released "message") | throw (noStatement v f "coin's message is no lowercase hex")
  if msg.size != 64 then throw (noStatement v f s!"message is {msg.size} bytes, not 64")
  let some sig := hexOpt (getD released "signature") | throw (noStatement v f "coin's signature is no lowercase hex")
  let signed := Flock.Sha512.hash (Flock.SessionCoins.signedContext (key.extract 0 16) session ++
    Flock.be64 msg.size.toUInt64 ++ msg)
  let ok : Bool := key.size == 16 + Lms.deployed.N && match Lms.Signature.decode Lms.deployed sig with
    | .ok s => (Lms.verify Lms.deployed key signed s).toOption == some true
    | .error _ => false
  if !ok then throw (noStatement v f BAD_SIGNATURE)
  return (#[("bytes", Flock.hexOf msg)], 64)

/-- A function's body: its value's sources (`sourceBytes`) and its length in bytes. `const.u16/v1`'s `value` as a u16
little-endian; `pous.setup.salt/v1` the coin the verifier holds for it, 24 bytes (`coins`: the memory attester's salt,
drawn after W's commitment); `pouw.pearlplus.salt/v1` the epoch's salt of Pearl+, the coin it holds for it, 32 bytes;
`window.salt/v1` the window's salt under the coin it holds for it (`runSalt`, 64 bytes);
`registered.root/v1` the root of the table its `value` names as the verifier holds it (`HELD_ROOT`, never the record's
entry), `registered.domain/v1` that held table's domain id (`HELD_DOMAIN`), and
`record.root/v1` the root the record names for its
`port` (64 bytes each); `log.base/v1` the first position after the log's records (8 bytes). The storage session's four functions read the
session in two entries by when each part exists: the commitment, held before the first coin, `{commitment, key, place}`
(the header's fields with the coin key by its hash, the LMS public key and the place it was registered at), and the coin,
released after the session, `{openings, coin_key}` for the round coin, `{ended, rounds}` for the header and
`{message, signature}` for the message:
`pous.session.round_coin/v1` is `u32be(round) ‖ u64be(index)` (12 bytes) of the round's last opening, which opens its
committed coin below the commitment's blocks; `pous.session.header/v1` the header of a session that ended complete with
all its rounds, in `head` bytes (whole u16 words: a zero byte after an odd one); `pous.session.context/v1` the 106 bytes
the signature's message hashes before the record; `pous.session.message/v1` the 64-byte message m the signature signs.
The context and the message read the commitment's key only once its SHA-512 is one of `pinned`, the configuration's pins of
the memory attester's keys (`UNPINNED`): no body takes the key the session's own object names. The message is read only
once the coin's `signature` verifies over it under that key (`BAD_SIGNATURE`): m and the signature are public, and the signature's leaf q shows how many signatures the attester has
made. "no statement" for any other function, whose body isn't written. -/
def body (coins commitments : Array (String × ByteArray)) (v : String) (m : Mark) (pinned : Array ByteArray := #[]) :
    V (Array (String × String) × Nat) := do
  match m.fn with
  | "const.u16/v1" =>
    let x ← argNat v m "value"
    if x ≥ 2 ^ 16 then throw s!"supplied input {v}: no statement: {x} is no u16"
    return (#[("bytes", Flock.hexOf ⟨#[(x % 256).toUInt8, (x / 256).toUInt8]⟩)], 2)
  | "pous.setup.salt/v1" =>
    let c ← coin coins v m
    if c.size != 24 then throw s!"supplied input {v}: no statement: {m.fn}'s coin is {c.size} bytes, not 24"
    return (#[("bytes", Flock.hexOf c)], 24)
  | "pouw.pearlplus.salt/v1" =>
    let c ← coin coins v m
    if c.size != 32 then throw s!"supplied input {v}: no statement: {m.fn}'s coin is {c.size} bytes, not 32"
    return (#[("bytes", Flock.hexOf c)], 32)
  | "window.salt/v1" =>
    let c ← coin coins v m
    if c.isEmpty then throw s!"supplied input {v}: no statement: {m.fn}'s coin is no bytes"
    return (#[("window", Flock.hexOf c)], 64)
  | "registered.root/v1" => return (#[(HELD_ROOT, ← argStr v m "value")], 64)
  | "registered.domain/v1" => return (#[(HELD_DOMAIN, ← argStr v m "value")], 64)
  | "record.root/v1" => return (#[("port", ← argStr v m "port")], 64)
  | "log.base/v1" =>
    if !m.args.isEmpty then throw s!"supplied input {v}: {m.fn} takes no args"
    return (#[("position", "0")], 8)
  | "pous.session.round_coin/v1" => roundCoin coins commitments v m
  | "pous.session.header/v1" => header coins commitments v m
  | "pous.session.context/v1" => context commitments v m pinned
  | "pous.session.message/v1" => message coins commitments v m pinned
  | f => throw s!"supplied input {v}: no statement: {f}'s body is not the verifier's yet"

/-- The verifier's `attester_keys`, the SHA-512s (hex) of the memory attester's keys its configuration pins; none when
absent. -/
def attesterKeys (j : Json) : V (Array ByteArray) := match getD j "attester_keys" with
  | .null => pure #[]
  | .arr ks => ks.mapM fun
    | .str s => Flock.unhex s
    | _ => throw "attester_keys: a key's hash is not hex"
  | _ => throw "attester_keys is a list of hex"

/-- A supplied input's one row of sources (`Rows`): its body's, led by `(LE, "")` where its layout is `le`, its value's
length the mark's bits. -/
def ofMark (coins commitments : Array (String × ByteArray)) (v : String) (m : Mark) (pinned : Array ByteArray := #[]) :
    V Rows := do
  let (ss, n) ← body coins commitments v m pinned
  if 8 * n != m.bits then
    throw s!"supplied input {v}: {m.fn}'s value is {8 * n} bits, and the mark says {m.bits}"
  return #[(if m.layout == "le" then #[(LE, "")] else #[]) ++ ss]

private def hexRep (b : String) : Json := .str (String.join (List.replicate 64 b))
private def domainRec (domain : Option String) : Json :=
  Json.mkObj [("registered", Json.arr #[Json.mkObj ([("value", .str "w"), ("root", hexRep "11")] ++
    (domain.map fun d => [("domain", hexRep d)]).getD [])])]
private def anchorRow (fn : String) (rec : Json) (held : Array (String × String × String)) : V (Array Nat) := do
  let rs ← ofMark #[] #[] "x" { fn, args := #[("value", .str "w")], layout := "digest", bits := 512 }
  let some r := rs[0]? | throw "no row"
  digestLeaves (← r.foldlM (fun acc s => return acc ++ (← sourceBytes rec #[] none held #[] "x" s)) ByteArray.empty)
private def digestOf (b : UInt8) : Option (Array Nat) := (digestLeaves ⟨(List.replicate 64 b).toArray⟩).toOption
private def HELD_W : Array (String × String × String) :=
  #[("w", String.join (List.replicate 64 "44"), String.join (List.replicate 64 "55"))]

-- a table input reads the held table's domain id and root, whatever the record's entry of it says
#guard (anchorRow "registered.domain/v1" (domainRec (some "33")) HELD_W).toOption == digestOf 0x55
#guard (anchorRow "registered.root/v1" (domainRec (some "33")) HELD_W).toOption == digestOf 0x44
-- and with no held value it has no row: the record's entry is never its source
#guard (anchorRow "registered.domain/v1" (domainRec (some "33")) #[]).toOption.isNone
#guard (anchorRow "registered.root/v1" (domainRec (some "33")) #[]).toOption.isNone

/-- The place a held commitment was registered at: its JSON's `place`, a natural; none for a text that is no JSON object of
one. -/
private def commitmentPlace (b : ByteArray) : Option Nat := do
  let s ← String.fromUTF8? b
  let .ok j := Flock.JsonParse.parse s | none
  nat? (getD j "place")

/-- A Python list of strings' repr, as `supplied.check_order` names the late commitments. -/
private def pyNames (xs : List String) : String := "[" ++ ", ".intercalate (xs.map fun x => "'" ++ x ++ "'") ++ "]"

/-- `registered.order`, §5.4's order rule at registration: every commitment C's marks read is registered before every
coin they read, `E_SUPPLIED_ORDER` naming the late ones. A mark reads a coin where its function's value is one the
verifier holds (`coins`), at its place in `released` (`{function: n}`: the coin was released after n records, this one
the `records`-th, its commitments before any coin it reads); its rows' sources read the commitments: a `value`'s or
`port`'s root this record's, at place `records`, and `position` the log's, at `records - 1`. Refused where `released`
is not an object of a place by supplied function, or names no place of a coin read beside a commitment. A held commitment
(`commitments`) is no coin, but a coin follows it: the `place` its JSON carries (where the verifier registered it) is
before the place of every coin a mark reads, `E_SUPPLIED_ORDER` naming the commitments at or after it, and refused where a
commitment has no place or the coin has none in `released`. -/
def order (marked : Array (String × Mark)) (rows : Array (String × Rows)) (coins commitments : Array (String × ByteArray))
    (released : Json) (records : Nat) : V Unit := do
  let places ← match released with
    | .null => pure #[]
    | .obj kvs => kvs.toArray.mapM fun (f, n) => match nat? n with
      | some n => if FUNCTIONS.contains f then pure (f, n) else throw "released is an object of a place by supplied function"
      | none => throw "released is an object of a place by supplied function"
    | _ => throw "released is an object of a place by supplied function"
  let reads := rows.foldl (fun acc (_, rs) => rs.foldl (fun acc r => r.foldl (fun acc (k, n) =>
      let place? := if k == "value" || k == "port" then some records
        else if k == "position" && records > 0 then some (records - 1) else none
      match place? with
      | none => acc
      | some place =>
        let name := if k == "position" then "log" else n
        match acc.find? (·.1 == name) with
        | some (_, b) => (acc.filter (·.1 != name)).push (name, max place b)
        | none => acc.push (name, place)) acc) acc) (#[] : Array (String × Nat))
  let fns := (marked.foldl (fun l (_, m) => if coins.any (·.1 == m.fn) && !l.contains m.fn then l ++ [m.fn] else l)
    ([] : List String)).mergeSort (· ≤ ·)
  let cms := (marked.foldl (fun l (_, m) => if commitments.any (·.1 == m.fn) && !l.contains m.fn then l ++ [m.fn] else l)
    ([] : List String)).mergeSort (· ≤ ·)
  if !cms.isEmpty && !fns.isEmpty then
    let placed ← cms.mapM fun c => do
      let some p := (commitments.find? (·.1 == c)).bind (commitmentPlace ·.2)
        | throw s!"released: the verifier names no place of {c}'s commitment, which a coin it holds follows"
      pure (c, p)
    for f in fns do
      let some (_, coinAt) := places.find? (·.1 == f)
        | throw s!"released: the verifier names no place of {f}'s coin, which follows a commitment it holds"
      let late := (placed.filter (·.2 ≥ coinAt)).map fun (c, _) => "commitment " ++ c
      unless late.isEmpty do
        throw s!"E_SUPPLIED_ORDER: {pyNames late} registered after {f}, which a supplied value reads"
  for f in fns do
    if reads.isEmpty then return
    let some (_, coinAt) := places.find? (·.1 == f)
      | throw s!"released: the verifier names no place of {f}'s coin, which a supplied value reads beside commitments"
    let late := ((reads.filter (·.2 ≥ coinAt)).map (·.1)).toList.mergeSort (· ≤ ·)
    unless late.isEmpty do
      throw s!"E_SUPPLIED_ORDER: {pyNames late} registered after {f}, which a supplied value reads"

/-- The difftest's `supplied_marks`: on `{program, coins, commitments, attester_keys, widths, record, log, binding, held}`
(`held`: `{table: {root, domain}}`, the tables the verifier holds; `coins` and
`commitments`: `{function: hex}`, what the verifier holds of each function as a coin and as a released commitment, which
the order rule counts the first of alone;
`widths`: `{input: w}`; `binding`: `{stream, prover}` or null), each supplied input's rows as leaves of its width
(`marks`, `ofMark`, held to the order rule at `released`, `order`, then `rows`' reading), cut where `cuts` names it
(`{input: [n, k]}`, `cut`), or the refusal. -/
def piece (j : Json) : Except String Json := do
  let ms ← marks (getD j "program")
  let coins ← match getD j "coins" with
    | .obj kvs => kvs.toArray.mapM fun (f, h) => match h with
      | .str s => do pure (f, ← Flock.unhex s)
      | _ => throw "coins: a coin is not hex"
    | _ => throw "coins is not an object"
  let commitments ← match getD j "commitments" with
    | .null => pure #[]
    | .obj kvs => kvs.toArray.mapM fun (f, h) => match h with
      | .str s => do pure (f, ← Flock.unhex s)
      | _ => throw "commitments: a commitment is not hex"
    | _ => throw "commitments is not an object"
  let widths := getD j "widths"
  let .obj _ := widths | throw "widths is not an object"
  let .arr log := getD j "log" | throw "log is not a list"
  let binding ← match getD j "binding" with
    | .null => pure none
    | b => match getD b "stream", getD b "prover" with
      | .str s, .str p => pure (some (s, p))
      | _, _ => throw "binding is not {stream, prover}"
  let rec_ := getD j "record"
  let held ← match getD j "held" with
    | .null => pure #[]
    | .obj kvs => kvs.toArray.mapM fun (v, e) => match getD e "root", getD e "domain" with
      | .str r, .str d => pure (v, r, d)
      | _, _ => throw s!"held: {v} has no root and domain"
    | _ => throw "held is not an object"
  let pinned ← attesterKeys j
  let rows ← ms.mapM fun (v, m) => do pure (v, ← ofMark coins commitments v m pinned)
  order ms rows coins commitments (getD j "released") log.size
  let out ← ms.mapM fun (v, m) => do
    let some w := nat? (getD widths v) | throw s!"widths: no width of {v}"
    let rs ← ofMark coins commitments v m pinned
    let ls ← rs.mapM fun r => do
      let le := r[0]?.map (·.1) == some LE
      let data ← (if le then r.extract 1 r.size else r).foldlM
        (fun acc s => return acc ++ (← sourceBytes rec_ log binding held (ms.map (·.1)) v s)) ByteArray.empty
      if le then leLeaves data w else if m.layout == "digest" then digestLeaves data else leavesOf data w
    let ls ← match getD (getD j "cuts") v with
      | .null => pure ls
      | .arr #[n, k] => match nat? n, nat? k with
        | some n, some k => pure (cut ls n k)
        | _, _ => throw s!"cuts: {v}'s is not [n, k]"
      | _ => throw s!"cuts: {v}'s is not [n, k]"
    pure (v, Json.arr (ls.map fun l => Json.arr (l.map fun (x : Nat) => Json.num (Int.ofNat x))))
  return Json.mkObj out.toList

/-- The difftest's `registered_values`: on `{program, values}`, the values a record of the program registers
(`registeredValues` of `values` and the program's `marks`), or the refusal. -/
def registeredValuesPiece (j : Json) : Except String Json := do
  let .arr vs := getD j "values" | throw "values is not a list"
  let values ← vs.mapM fun v => match v with
    | .str s => pure s
    | _ => throw "values: a value is not a string"
  return Json.arr ((← registeredValues values (← marks (getD j "program"))).map Json.str)

end Spot.Supplied
