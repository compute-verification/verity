module

public import NetworkVerifier.Check
public import Flock.Registered

/-!
# The check of a load's registrations

A workload's own requests reach its host through the network attester, which commits each value of the weights as
it crosses and signs its head once a session (`transfer_rows.py`'s module docstring is the head's and the
registration's spec). `loadCheck` is the verifier's read of the registrations a run kept of them (``load.json``,
`load_relay.load_doc`), under the deployment's key and session: the document names a shape, whose manifest
(`manifest`) the verifier computes itself, and holds one registration of each of the manifest's values, in order,
which `registrationCheck` accepts for the transfer the verifier expects: the document's link, ingress, window 0, kind
``load``, the value's name, the document's program and the value's rows, of 16-bit leaves in `hm96-sha512/row/v2`
rows.

`registrationCheck` reads one registration for the transfer `Expected` states, in this order, and refuses by name:

* `kind`: the expected transfer is not a load, ingress;
* `leaf-width`, `row-schema`: its leaves are not 16 or 32 bits, or its rows not `hm96-sha512/row/v2` or
  `hm96-sha512/row-seg/v1` rows;
* `program`: its program digest is not 128 lowercase hex digits;
* `domain`: no domain can be derived for it (`Flock.Registered.domain`);
* `not-object`: the registration is not a JSON object;
* `field:<key>`: the registration's `format`, `link`, `direction`, `window`, `kind`, `value`, `program`, `words`,
  `rows`, `word_bits`, `row_schema` or `domain`, the first in that order that is not the verifier's own: a string is the
  verifier's string and a number the verifier's natural number, and nothing else is either;
* `keys`: the registration holds another key set than those and `root` and `certificate`;
* `root`: its root is not 128 lowercase hex digits;
* `certificate`: its certificate is not an object of exactly `key`, `digest` and `signature`, each lowercase hex;
* `encode`: the head does not encode (a name of 65,536 UTF-8 bytes or more, a count past its width);
* `head`: the certificate's signed digest is not the head's digest (`headDigest`);
* `signature`: the certificate is not the key's, or its LMS signature doesn't verify over its message in the session
  for the expected link, direction and window (`Check.Certificate.signs`).

`loadCheck` refuses the document, by name, before it reads a registration:

* `not-object`: the document is not a JSON object;
* `shape`: it names no shape of `shapes`;
* `format`: its format is not `loadFormat`, its window not the number 0, or its link not a string;
* `manifest`: the shape's manifest has a value that is not whole rows (no shape of `shapes` has one);
* `values`: its values are not a list of as many registrations as the manifest has values.

Past those it has two problems of the document's own, each beside what its registrations give:

* `load-keys`: it holds another key set than `loadKeys`, or its `link_format` is not a string;
* `key-session`: its key and session are not the deployment's, the ones the check is made under.

Then every value's registration is read, and each refusal is a problem naming its value. The load is accepted when
there is no problem; `registered` counts the values whose registration is accepted, whatever the document's own. The check reads the manifest, of 5 + 14 · layers values, and the registrations it is handed, each
once.

`loadFn` and `registrationFn` are the `load` and `transfer-registration` functions of `network-records verdicts`
(`Difftest.serve`): `relay_check.check_load` calls `load` for a run's verdict on its load, and
`tests/verity/protocols/accounting/network/verifier/network/test_network_load_agreement.py` holds both to
the Python reference (`transfer_rows.check_registration`). Code: it imports only Lean, v4's certificate and its one
LMS check (`NetworkVerifier.Check`) and Flock's registered value's domain (`Flock.Registered`).
-/

@[expose] public section

namespace NetworkVerifier.Load

open Lean (Json toJson)
open Flock (V)
open NetworkVerifier.Check (Certificate)

/-- `load_check.FORMAT`. -/
def loadFormat : String := "verity-network-certifier/load/v1"
/-- `transfer_rows.TRANSFER_TAG`. -/
def transferTag : String := "verity-network-certifier/transfer/v1"
/-- `transfer_rows.REGISTRATION`. -/
def registrationFormat : String := "verity-network-certifier/transfer-registration/v1"
/-- `transfer_rows.KINDS`: the one kind, a load, ingress. -/
def loadKind : String := "load"
/-- The load's execution phase, the session's first (`load_check.WINDOW`). -/
def loadWindow : Nat := 0
/-- A load value's leaves (`load_check.Value.word_bits`). -/
def loadWordBits : Nat := 16
/-- A load value's rows (`load_check.SCHEMA`). -/
def loadSchema : String := Flock.HmRow.ROW_SCHEMA_V2
/-- `transfer_rows.SCHEMAS`: the bit-row formats a proof block's port carries. -/
def schemas : List String := [Flock.HmRow.ROW_SCHEMA_V2, Flock.HmRow.ROW_SCHEMA_SEG]

/-- The keys a registration holds (`transfer_rows.registration`). -/
def registrationKeys : List String :=
  ["certificate", "direction", "domain", "format", "kind", "link", "program", "root", "row_schema", "rows", "value",
    "window", "word_bits", "words"]

/-- The keys a load's document holds (`load_relay.load_doc`). -/
def loadKeys : List String :=
  ["format", "key", "link", "link_format", "program", "session", "shape", "values", "window"]

/-! ## Reading -/

/-- A refusal: its name and its words. -/
structure Refusal where
  name : String
  why : String

abbrev R := Except Refusal

def refuse {α} (name why : String) : R α := .error ⟨name, why⟩

/-- An object's keys, in order; none for anything else. -/
def keysOf : Json → Option (List String)
  | .obj kvs => some kvs.keys
  | _ => none

/-- The string at `k`; none where `j` has no `k` or holds another type there. -/
def strAt (j : Json) (k : String) : Option String :=
  match j.getObjVal? k with
  | .ok (.str s) => some s
  | _ => none

/-- The natural number at `k`; none where `j` has no `k` or holds anything else there (a fraction, a negative, a
boolean). -/
def natAt (j : Json) (k : String) : Option Nat :=
  match j.getObjVal? k with
  | .ok v => v.getNat?.toOption
  | .error _ => none

/-- The list of natural numbers at `k`. -/
def natsAt (j : Json) (k : String) : Option (List Nat) :=
  match j.getObjVal? k with
  | .ok (.arr xs) => xs.toList.mapM fun x => x.getNat?.toOption
  | _ => none

/-- The bytes at `k`, a string of lowercase hex. -/
def hexAt (j : Json) (k : String) : Option ByteArray := (strAt j k).bind fun s => (Flock.unhex s).toOption

/-- Whether `s` is a SHA-512 digest in lowercase hex. -/
def isDigest (s : String) : Bool := s.length == 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

/-! ## The key -/

/-- What the check is made under: the network attester's LMS key and its session, the deployment's. -/
structure Key where
  lms : Lms.Params
  pub : ByteArray
  session : ByteArray

/-- The key as a deployment writes it (`record_files.public_json`): the LMS parameters `[N, W, H]`, the public key
`I ‖ T[1]` in lowercase hex, and its identifier `I`. -/
def Key.json (k : Key) : Json :=
  Json.mkObj [("key_id", .str (Flock.hexOf (k.pub.extract 0 16))), ("params", toJson [k.lms.N, k.lms.W, k.lms.H]),
    ("public", .str (Flock.hexOf k.pub))]

/-- Whether `j` is the key as a deployment writes it (`Key.json`), key by key. -/
def Key.written (k : Key) (j : Json) : Bool :=
  keysOf j == some ["key_id", "params", "public"] && strAt j "key_id" == some (Flock.hexOf (k.pub.extract 0 16)) &&
    natsAt j "params" == some [k.lms.N, k.lms.W, k.lms.H] && strAt j "public" == some (Flock.hexOf k.pub)

/-- The key and session a case names: the deployment's ``key`` and ``session``. -/
def keyOf (j : Json) (session : String) : V Key := do
  unless keysOf j == some ["key_id", "params", "public"] do throw "key: the keys must be [key_id, params, public]"
  let some [n, w, h] := natsAt j "params" | throw "key.params: not [N, W, H]"
  let lms ← Lms.Params.make n w h false
  let some pub := hexAt j "public" | throw "key.public: not lowercase hex"
  unless pub.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {pub.size}"
  let key : Key := { lms, pub, session := ← Flock.unhex session }
  unless key.written j do throw "key.key_id: not the key's identifier, its first 16 bytes"
  pure key

/-! ## A registration -/

/-- The transfer the verifier expects (`transfer_rows.check_registration`'s arguments): its link and execution phase,
its kind, the value, the circuit's program digest, and the value's rows of `words` leaves of `wordBits` bits in
`schema` rows. -/
structure Expected where
  link : String
  direction : Direction
  window : Nat
  kind : String
  value : String
  program : String
  words : Nat
  rows : Nat
  wordBits : Nat
  schema : String

/-- `TransferHead.digest`: `tagged_sha512(TRANSFER_TAG, H)` of the head the network attester signs for a transfer
value, at the value's domain and root. -/
def headDigest (e : Expected) (program dom root : ByteArray) : V ByteArray := do
  unless program.size == digestBytes && dom.size == digestBytes && root.size == digestBytes do
    throw "the program digest, the domain and the root are 64 bytes"
  Identity.taggedSha512 transferTag ((← text e.link "link") ++ (← uBE 1 e.direction.code "direction") ++
    (← uBE 8 e.window "window") ++ (← text e.kind "kind") ++ (← text e.value "value") ++ program ++
    (← uBE 4 e.words "words") ++ (← uBE 4 e.rows "rows") ++ (← uBE 1 e.wordBits "word_bits") ++
    (← text e.schema "schema") ++ dom ++ root)

/-- The registration's string at `k` is the verifier's `v`. -/
def holdsStr (r : Json) (k v : String) : R Unit :=
  if strAt r k == some v then pure ()
  else refuse s!"field:{k}" s!"the transfer registration's {k} is not the verifier's {v}"

/-- The registration's natural number at `k` is the verifier's `v`. -/
def holdsNat (r : Json) (k : String) (v : Nat) : R Unit :=
  if natAt r k == some v then pure ()
  else refuse s!"field:{k}" s!"the transfer registration's {k} is not the verifier's {v}"

/-- `transfer_rows.check_registration`: the registration `r` read for the transfer `e` under `key` (the module
docstring's order and names). -/
def registrationCheck (key : Key) (e : Expected) (r : Json) : R Unit := do
  unless e.kind == loadKind && e.direction == .ingress do
    refuse "kind" s!"a transfer is a load ingress, not {e.kind} {e.direction.value}"
  unless Flock.Registered.WIDTHS.contains e.wordBits do
    refuse "leaf-width" s!"a transfer value's leaves are 16 or 32 bits, not {e.wordBits}"
  unless schemas.contains e.schema do
    refuse "row-schema" s!"a transfer value's rows are {" or ".intercalate schemas} rows, not {e.schema}"
  unless isDigest e.program do refuse "program" "a program digest is lowercase hex of 64 bytes"
  let dom ← match Flock.Registered.domain e.program e.value e.words e.rows e.wordBits with
    | .ok dom => pure dom
    | .error why => refuse "domain" s!"the transfer has no domain: {why}"
  unless (keysOf r).isSome do refuse "not-object" "a transfer registration is a JSON object"
  holdsStr r "format" registrationFormat
  holdsStr r "link" e.link
  holdsStr r "direction" e.direction.value
  holdsNat r "window" e.window
  holdsStr r "kind" e.kind
  holdsStr r "value" e.value
  holdsStr r "program" e.program
  holdsNat r "words" e.words
  holdsNat r "rows" e.rows
  holdsNat r "word_bits" e.wordBits
  holdsStr r "row_schema" e.schema
  holdsStr r "domain" (Flock.hexOf dom)
  unless keysOf r == some registrationKeys do
    refuse "keys" s!"a transfer registration holds the keys {registrationKeys} and no other"
  let some root := (strAt r "root").filter isDigest
    | refuse "root" "the transfer registration's root is lowercase hex of 64 bytes"
  let c := r.getObjValD "certificate"
  unless keysOf c == some ["digest", "key", "signature"] do
    refuse "certificate" "the transfer registration's certificate is {key, digest, signature}"
  let some keyId := hexAt c "key" | refuse "certificate" "the certificate's key is lowercase hex"
  let some digest := hexAt c "digest" | refuse "certificate" "the certificate's digest is lowercase hex"
  let some signature := hexAt c "signature" | refuse "certificate" "the certificate's signature is lowercase hex"
  let head ← match (do headDigest e (← Flock.unhex e.program) dom (← Flock.unhex root)) with
    | .ok head => pure head
    | .error why => refuse "encode" s!"the transfer's head doesn't encode: {why}"
  unless digest == head do refuse "head" "the transfer registration's certificate doesn't sign its head"
  unless ({ keyId, link := e.link, direction := e.direction, window := e.window, digest, signature } :
      Certificate).signs key.lms key.pub key.session do
    refuse "signature" s!"the transfer registration's certificate doesn't verify under the network attester's key in \
      its session for {e.link} {e.direction.value} window {e.window}"

/-! ## The manifest -/

/-- Interface 1's shape (`load_check.Shape`). -/
structure Shape where
  name : String
  width : Nat
  layers : Nat
  heads : Nat
  headDim : Nat
  kvHeads : Nat
  mlp : Nat
  vocab : Nat
  context : Nat
  rowCodes : Nat

/-- `load_check.H200`. -/
def h200 : Shape := ⟨"cmp92-h200-mqa-w16384-l52", 16384, 52, 256, 64, 1, 65536, 256, 32, 8192⟩
/-- `load_check.TINY`. -/
def tiny : Shape := ⟨"tiny-h32-l1", 32, 1, 2, 16, 1, 64, 256, 8, 16⟩
/-- The shapes a load names (`load_check.SHAPES`). -/
def shapes : List Shape := [h200, tiny]

/-- One value of the load as the verifier reads its registration (`load_check.Value`): interface 1's tensor `name`,
its `nbytes` bytes of the weights stream, in rows of `words` u16 words. -/
structure Value where
  name : String
  nbytes : Nat
  words : Nat

/-- A row's bytes. -/
def Value.rowBytes (v : Value) : Nat := 2 * v.words

/-- The value's rows (`load_check.Value.leaves`). -/
def Value.leaves (v : Value) : Nat := v.nbytes / v.rowBytes

/-- A linear's two values (`load_check.manifest`'s `linear`): its `n` s8 rows of `k` codes, `rowCodes` codes a row,
then its f32 scale, two words a row. -/
def linear (s : Shape) (name : String) (n k : Nat) : List Value :=
  [⟨name, n * k, s.rowCodes / 2⟩, ⟨name ++ ".scale", 4 * n, 2⟩]

/-- A layer's values, in stream order. -/
def layer (s : Shape) (i : Nat) : List Value :=
  let p := s!"l{i}."
  let (q, kv) := (s.heads * s.headDim, s.kvHeads * s.headDim)
  [⟨p ++ "attn_norm", 2 * s.width, 1⟩] ++ linear s (p ++ "wq") q s.width ++ linear s (p ++ "wk") kv s.width ++
    linear s (p ++ "wv") kv s.width ++ linear s (p ++ "wo") s.width q ++ [⟨p ++ "mlp_norm", 2 * s.width, 1⟩] ++
    linear s (p ++ "wup") s.mlp s.width ++ linear s (p ++ "wdown") s.width s.mlp

/-- `load_check.manifest`: interface 1's tensors in stream order, each a value. -/
def manifest (s : Shape) : List Value :=
  [⟨"tok", 2 * s.width * s.vocab, s.vocab⟩, ⟨"pos", 2 * s.width * s.context, 1⟩] ++
    (List.range s.layers).flatMap (layer s) ++ [⟨"final_norm", 2 * s.width, 1⟩] ++ linear s "head" s.vocab s.width

/-- Every value is whole rows of at least one word (`load_check.manifest`'s `add` raises on a value that isn't). -/
def whole (vs : List Value) : Bool := vs.all fun v => 1 ≤ v.words && v.nbytes % v.rowBytes == 0

-- each shape's values, bytes and rows are `load_check.manifest`'s, every value whole rows
#guard shapes.all fun s => whole (manifest s)
#guard (manifest tiny).length == 19 && (manifest tiny).foldl (· + ·.nbytes) 0 == 34240 &&
  (manifest tiny).foldl (· + ·.leaves) 0 == 1792
#guard (manifest h200).length == 733 && (manifest h200).foldl (· + ·.nbytes) 0 == 139736443904 &&
  (manifest h200).foldl (· + ·.leaves) 0 == 25284864

/-- The transfer the verifier expects for a load's value `v` on `link` under `program`. -/
def expected (link program : String) (v : Value) : Expected :=
  { link, direction := .ingress, window := loadWindow, kind := loadKind, value := v.name, program, words := v.words,
    rows := v.leaves, wordBits := loadWordBits, schema := loadSchema }

/-! ## The load -/

/-- A problem of the load: a refusal, of the value's registration where it names one, else of the document. -/
structure Problem where
  value : Option String
  refusal : Refusal

/-- The check's summary: the manifest's values, how many have a registration the check
accepts, and the problems, none on an accepted load. -/
structure Summary where
  values : Nat
  registered : Nat
  problems : Array Problem

/-- The document read before its registrations: its link and program, its shape's values each beside its
registration, and the document's own problems that leave its registrations readable (`load-keys`, `key-session`); or
its refusal, with the manifest's values where the shape is read. -/
def documentOf (key : Key) (doc : Json) :
    Except (Nat × Refusal) (String × String × List (Value × Json) × Array Refusal) := do
  unless (keysOf doc).isSome do throw (0, ⟨"not-object", "a load's registrations are a JSON object"⟩)
  let some shape := (strAt doc "shape").bind fun n => shapes.find? (·.name == n)
    | throw (0, ⟨"shape", s!"no such shape: one of {shapes.map (·.name)}"⟩)
  let some link := strAt doc "link" |
    throw (0, ⟨"format", s!"not a load's registrations ({loadFormat}, window {loadWindow})"⟩)
  unless strAt doc "format" == some loadFormat && natAt doc "window" == some loadWindow do
    throw (0, ⟨"format", s!"not a load's registrations ({loadFormat}, window {loadWindow})"⟩)
  let values := manifest shape
  let n := values.length
  unless whole values do throw (n, ⟨"manifest", s!"{shape.name}: a value that is not whole rows of its words"⟩)
  let regs ← match doc.getObjValD "values" with
    | .arr regs => pure regs
    | _ => throw (n, ⟨"values", s!"not one registration of each of {n} values"⟩)
  unless regs.size == n do throw (n, ⟨"values", s!"not one registration of each of {n} values"⟩)
  let keys : Array Refusal :=
    if keysOf doc == some loadKeys && (strAt doc "link_format").isSome then #[]
    else #[⟨"load-keys", s!"a load's registrations hold the keys {loadKeys} and no other, link_format a string"⟩]
  let under : Array Refusal :=
    if key.written (doc.getObjValD "key") && strAt doc "session" == some (Flock.hexOf key.session) then #[]
    else #[⟨"key-session", "not signed under the deployment's key and session"⟩]
  pure (link, (strAt doc "program").getD "", values.zip regs.toList, keys ++ under)

/-- The check of a load's document under the deployment's key and session (the module docstring). -/
def loadCheck (key : Key) (doc : Json) : Summary :=
  match documentOf key doc with
  | .error (n, r) => ⟨n, 0, #[⟨none, r⟩]⟩
  | .ok (link, program, read, own) =>
    let problems := read.toArray.filterMap fun (v, reg) =>
      match registrationCheck key (expected link program v) reg with
      | .ok () => none
      | .error r => some ⟨some v.name, r⟩
    ⟨read.length, read.length - problems.size, own.map (⟨none, ·⟩) ++ problems⟩

def Refusal.fields (r : Refusal) : List (String × Json) := [("refusal", .str r.name), ("why", .str r.why)]

def Summary.json (o : Summary) : Json :=
  Json.mkObj [("values", toJson o.values), ("registered", toJson o.registered),
    ("problems", .arr (o.problems.map fun p =>
      Json.mkObj (p.refusal.fields ++ match p.value with
        | some v => [("value", .str v)]
        | none => [])))]

/-! ## The functions -/

/-- A case's object holds exactly `want`. -/
def exactly (j : Json) (want : List String) (ctx : String) : V Unit :=
  if keysOf j == some want then pure () else throw s!"{ctx}: the keys must be {want}"

/-- `network-records verdicts`' `load`: the outcome of a load's document under the deployment's key and session,

    {"key": {"key_id", "params": [N, W, H], "public"}, "session": hex, "load": the document}

as `{"values", "registered", "problems": [{"refusal", "why", "value"?}, …]}`; or the refusal of a case whose key or
session it can't read. -/
def loadFn (j : Json) : Except String Json := do
  exactly j ["key", "load", "session"] "the case"
  let some session := strAt j "session" | throw "session: not a string"
  let key ← keyOf (j.getObjValD "key") session
  return (loadCheck key (j.getObjValD "load")).json

/-- What a number a case expects is below. A registration's number is read by its value (`natAt`), and the reference
by its type too: the two read one registration alike while every number the verifier expects is below this, since the
caller writes a fraction of such a value with its point (`16.0`), which is no natural number, and only one at or past
it in exponent form (`1e+16`), which is one. A load's are: its window is 0, its leaves 16 bits, and its words and rows
those of `shapes`. -/
def numberBound : Nat := 10 ^ 15

/-- The transfer a case expects, `{"link", "direction", "window", "kind", "value", "program", "words", "rows",
"word_bits", "row_schema"}`, each number below `numberBound`. -/
def expectedOf (j : Json) : V Expected := do
  exactly j ["direction", "kind", "link", "program", "row_schema", "rows", "value", "window", "word_bits", "words"]
    "expected"
  let need {α} (o : Option α) (k : String) : V α := match o with
    | some a => pure a
    | none => throw s!"expected.{k}: not the type the transfer states it in"
  let nat (k : String) : V Nat := do
    let n ← need (natAt j k) k
    unless n < numberBound do throw s!"expected.{k}: {numberBound} or more"
    pure n
  pure { link := ← need (strAt j "link") "link",
         direction := ← Direction.ofValue (← need (strAt j "direction") "direction"),
         window := ← nat "window", kind := ← need (strAt j "kind") "kind",
         value := ← need (strAt j "value") "value", program := ← need (strAt j "program") "program",
         words := ← nat "words", rows := ← nat "rows", wordBits := ← nat "word_bits",
         schema := ← need (strAt j "row_schema") "row_schema" }

/-- `network-records verdicts`' `transfer-registration`: `registrationCheck` of one registration for the transfer a
case expects, `{"key", "session", "expected", "registration"}`, as `{"refusal": null}` or `{"refusal", "why"}`. -/
def registrationFn (j : Json) : Except String Json := do
  exactly j ["expected", "key", "registration", "session"] "the case"
  let some session := strAt j "session" | throw "session: not a string"
  let key ← keyOf (j.getObjValD "key") session
  let e ← expectedOf (j.getObjValD "expected")
  return match registrationCheck key e (j.getObjValD "registration") with
    | .ok () => Json.mkObj [("refusal", .null)]
    | .error r => Json.mkObj r.fields

end NetworkVerifier.Load
