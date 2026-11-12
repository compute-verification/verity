/-
The ZK server's contract, inner phase (live PROTOCOL.md §10.3): what the ZK server may send the challenger's live-coin part, as a
program. `rec_live` relays a live session through it (`flock-firewall live`, `liveCmd`), and its Python mirror is held to
it (`flock-firewall`, `verity/ml/flock/tests/test_firewall_agreement.py`).
-/
module

public import Lean.Data.Json
public import Flock.Hm96
public import Flock.JsonParse
public import Flock.CanonForm

@[expose] public section

namespace Flock.Firewall

open Lean (Json JsonNumber)

/-! ## The contract

Before release, every value the ZK server sends the auditor is one of three things:

* **salted**: a commitment, `Hm96.Default512.commitString`, under a fresh salt the workers never see;
* **masked**: the value plus a fresh pad the workers never see, released only after every relation the verifier will
  check holds of the value (the outer phase, §10.1; not in this file);
* **fixed**: uniquely fixed by what is already public (the pinned schedule) or committed.

Everything else a worker sends (the statement digest, labels, the domain, nonces, the hidden statement's Σ) is
**dropped**, and a request that leaves the schedule stops the session, which the auditor sees as `end: "stop"`.

For the inner phase that is the type `Sent`: each message to the challenger's live-coin part is the schedule's own value, a count the
schedule fixes, zeros, or a list of salted commitments. Nothing a worker sends reaches it except under a salt from the
ZK server's stream. `run` is the ZK server: the pinned schedule, the worker's requests, the ZK server's salts in, the challenger's
live-coin part's record out. -/

/-- A message from the ZK server to the challenger's live-coin part. -/
inductive Sent where
  /-- the schedule's `Hello` (fixed), never the worker's text -/
  | hello (h : Json)
  /-- each table's level-0 top, node by node, salted -/
  | commit (roots : List (String × List ByteArray))
  /-- `Link`'s `y`: the schedule's count of zero elements (fixed) -/
  | link (zeros : Nat)
  /-- a stream's name, parent and position, the schedule's (fixed), sent just before the stream's first round, never where
  the worker's `Open` came -/
  | openStream (stream parent : String) (pos : Nat)
  /-- a round: its stream, its coin count and its copies of a level-0 top committed at `Commit` (fixed), the nodes of each
  top it commits and its one message row, salted -/
  | round (stream : String) (tops : List (List ByteArray)) (level0 : Nat) (rows : List ByteArray) (n : Nat)
  /-- how the session ended, `finish` (true) or `stop` (false), never why -/
  | finish (ok : Bool)

/-! ## The wire (`live/src/lib.rs` `Req`; `rec_live.decode_req`) -/

inductive Req where
  | hello (text : String)
  | link (y : ByteArray)
  | ping
  | openStream (stream parent : String) (label : ByteArray) (pos : Nat)
  | round (stream : String) (content : ByteArray) (n : Nat)
  | proof
  | finish
  | register
  | commit (rootF : ByteArray) (roots publics : List (String × ByteArray))
  | prime

abbrev Rd := StateT Nat V

def need (b : ByteArray) (n : Nat) : Rd Unit := do
  if (← get) + n > b.size then throw "short frame"

def rdU32 (b : ByteArray) : Rd Nat := do
  need b 4
  let pos0 ← get
  set (pos0 + 4)
  return (List.range 4).foldl (fun x i => x + (b.get! (pos0 + i)).toNat * 256 ^ i) 0

def rdU64 (b : ByteArray) : Rd Nat := do
  need b 8
  let pos0 ← get
  set (pos0 + 8)
  return (List.range 8).foldl (fun x i => x + (b.get! (pos0 + i)).toNat * 256 ^ i) 0

def rdBytes (b : ByteArray) : Rd ByteArray := do
  let n ← rdU32 b
  need b n
  let pos0 ← get
  set (pos0 + n)
  return b.extract pos0 (pos0 + n)

def rdStr (b : ByteArray) : Rd String := do
  match String.fromUTF8? (← rdBytes b) with
  | some s => return s
  | none => throw "not UTF-8"

def rdList (b : ByteArray) : Rd (List (String × ByteArray)) := do
  let n ← rdU32 b
  let mut out := #[]
  for _ in [0:n] do
    let k ← rdStr b
    out := out.push (k, ← rdBytes b)
  return out.toList

/-- `decode_req`: the request in a frame (trailing bytes ignored, as `rec_live` ignores them). -/
def decode (b : ByteArray) : V Req := do
  if b.size == 0 then throw "empty frame"
  let r : Rd Req := match b.get! 0 with
    | 1 => do return .hello (← rdStr b)
    | 2 => do return .link (← rdBytes b)
    | 9 => do let _ ← rdBytes b; return .ping
    | 3 => do return .openStream (← rdStr b) (← rdStr b) (← rdBytes b) (← rdU64 b)
    | 4 => do return .round (← rdStr b) (← rdBytes b) (← rdU32 b)
    | 5 => do let _ ← rdStr b; let _ ← rdU32 b; let _ ← rdBytes b; return .proof
    | 6 => return .finish
    | 10 => return .register
    | 7 => do return .commit (← rdBytes b) (← rdList b) (← rdList b)
    | 8 => do let _ ← rdBytes b; let _ ← rdU32 b; return .prime
    | t => throw s!"unknown request tag {t}"
  return (← r.run 1).1

/-! ## The honest worker's wire (`Req::encode`, `rec_live.encode_req`): `decode`'s inverse where each length fits its
word -/

/-- `n`'s low `k` bytes, little-endian. -/
def putLE (k n : Nat) : ByteArray := ⟨((List.range k).map fun i => (n / 256 ^ i).toUInt8).toArray⟩

def putBytes (b : ByteArray) : ByteArray := putLE 4 b.size ++ b

def putStr (s : String) : ByteArray := putBytes s.toUTF8

def putPairs : List (String × ByteArray) → ByteArray
  | [] => .empty
  | (k, v) :: l => putStr k ++ putBytes v ++ putPairs l

def putList (l : List (String × ByteArray)) : ByteArray := putLE 4 l.length ++ putPairs l

def putReq : Req → ByteArray
  | .hello text => ⟨#[1]⟩ ++ putStr text
  | .link y => ⟨#[2]⟩ ++ putBytes y
  | .ping => ⟨#[9]⟩ ++ putBytes .empty
  | .openStream stream parent label pos =>
    ⟨#[3]⟩ ++ (putStr stream ++ (putStr parent ++ (putBytes label ++ putLE 8 pos)))
  | .round stream content n => ⟨#[4]⟩ ++ (putStr stream ++ (putBytes content ++ putLE 4 n))
  | .proof => ⟨#[5]⟩ ++ (putStr "" ++ (putLE 4 0 ++ putBytes .empty))
  | .finish => ⟨#[6]⟩
  | .register => ⟨#[10]⟩
  | .commit rootF roots publics => ⟨#[7]⟩ ++ (putBytes rootF ++ (putList roots ++ putList publics))
  | .prime => ⟨#[8]⟩ ++ (putBytes .empty ++ putLE 4 0)

/-! ## A round's content (`rec_live.frames`, `split_round`) -/

def OP_DOMAIN : Nat := 1
def OP_LABEL : Nat := 2
def OP_OBSERVE : Nat := 3
def OP_SQUEEZE : Nat := 4
def OP_BYTES : Nat := 5
def LABEL_R1CS : ByteArray := "flock-r1cs-v0".toUTF8
def NONCE_BYTES : Nat := 8
def NODE_BYTES : Nat := 64
def ROW_WORDS : Nat := 64

def le (b : ByteArray) (off n : Nat) : Nat := (List.range n).foldl (fun x i => x + (b.get! (off + i)).toNat * 256 ^ i) 0

/-- The transcript-v2 frames of a round's content: `(op, kind, n, payload)`. -/
def frames (c : ByteArray) : V (List (Nat × Nat × Nat × ByteArray)) := do
  let mut out := #[]
  let mut pos0 := 0
  -- each frame takes at least its 16-byte header, so `c.size / 16 + 1` steps reach the end
  for _ in [0:c.size / 16 + 1] do
    if pos0 < c.size then
      if pos0 + 16 > c.size then throw "a truncated frame header"
      let op := (c.get! pos0).toNat
      let kind := (c.get! (pos0 + 1)).toNat
      let n := le c (pos0 + 8) 8
      pos0 := pos0 + 16
      let size := if op == OP_OBSERVE then 16 * n else if op == OP_SQUEEZE then 0 else n
      if !(op ∈ [OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES]) then throw s!"unknown frame op {op}"
      if pos0 + size > c.size then throw "a frame overruns its round"
      out := out.push (op, kind, n, c.extract pos0 (pos0 + size))
      pos0 := pos0 + (size + 15) / 16 * 16
  return out.toList

structure Split where
  caps : Array ByteArray := #[]
  free : Array Nat := #[]
  n : Nat := 0

def isCap (p : ByteArray) : Bool := p.size > 0 && p.size % NODE_BYTES == 0

/-- `split_round`: a round's caps, its free message and its coin count. `first`: the round binds the statement (a rep
stream's first round: `LABEL_R1CS`, the statement digest, then the level-0 cap). -/
def split (c : ByteArray) (first : Bool) : V Split := do
  let fr ← frames c
  let some (lastOp, _, lastN, _) := fr.getLast? | throw "a round without its squeeze"
  if lastOp != OP_SQUEEZE then throw "a round without its squeeze"
  let mut s : Split := {}
  -- the binding's progress: none, after the label (0), after the digest (1), after the level-0 cap (2)
  let mut binding : Option Nat := none
  for (op, kind, n, p) in fr do
    if op == OP_LABEL then
      if first && p == LABEL_R1CS && binding.isNone then binding := some 0
    else if op == OP_BYTES then
      if binding == some 0 then binding := some 1
      else if binding == some 1 then
        if !isCap p then throw s!"a level-0 cap of {p.size} bytes"
        s := { s with caps := s.caps.push p }
        binding := some 2
      else if p.size == NONCE_BYTES then pure ()
      else if isCap p then s := { s with caps := s.caps.push p }
      else throw s!"a BYTES frame of {p.size} bytes is neither a cap nor a nonce"
    else if op == OP_OBSERVE then
      if (kind == 1 && n != 1) || !(kind == 1 || kind == 2) then throw s!"an observe of kind {kind} and length {n}"
      s := { s with free := s.free ++ (Array.range n).map fun i => le p (16 * i) 16 }
  if first && binding != some 2 then throw "a rep's first round binds no statement digest and level-0 cap"
  return { s with n := lastN }

/-- A frame as the worker writes it (`push_header`, `push_padded`; `frames`' inverse): its 16-byte header (op, kind, six
zero bytes, `n` as a little-endian u64), then its payload and zeros up to whole blocks of 16 bytes. -/
def putFrame : Nat × Nat × Nat × ByteArray → ByteArray
  | (op, kind, n, p) => ⟨#[op.toUInt8, kind.toUInt8, 0, 0, 0, 0, 0, 0]⟩ ++ putLE 8 n ++ p ++
      ⟨Array.replicate ((p.size + 15) / 16 * 16 - p.size) 0⟩

def putFrames : List (Nat × Nat × Nat × ByteArray) → ByteArray
  | [] => .empty
  | f :: fs => putFrame f ++ putFrames fs

/-- What the honest worker's challenger absorbs in a round, call by call (`framed_challenger!`). -/
inductive Absorb where
  | domain (d : ByteArray)
  | label (l : ByteArray)
  /-- the statement's binding: `LABEL_R1CS`, the statement digest, the level-0 cap -/
  | bind (digest cap : ByteArray)
  | cap (c : ByteArray)
  | nonce (x : ByteArray)
  | observe (kind : Nat) (elems : List Nat)
  /-- a squeeze inside the round, of no coins -/
  | squeeze (kind : Nat)

def Absorb.frames : Absorb → List (Nat × Nat × Nat × ByteArray)
  | .domain d => [(OP_DOMAIN, 0, d.size, d)]
  | .label l => [(OP_LABEL, 0, l.size, l)]
  | .bind d c => [(OP_LABEL, 0, LABEL_R1CS.size, LABEL_R1CS), (OP_BYTES, 0, d.size, d), (OP_BYTES, 0, c.size, c)]
  | .cap c => [(OP_BYTES, 0, c.size, c)]
  | .nonce x => [(OP_BYTES, 0, x.size, x)]
  | .observe kind es => [(OP_OBSERVE, kind, es.length, es.foldr (fun e p => putLE 16 e ++ p) .empty)]
  | .squeeze kind => [(OP_SQUEEZE, kind, 0, .empty)]

/-- A round's content: its items, then the squeeze of its `n` coins. -/
def putRound (items : List Absorb) (kind n : Nat) : ByteArray :=
  putFrames (items.flatMap Absorb.frames ++ [(OP_SQUEEZE, kind, n, .empty)])

/-! ## Salted commitments (`rec_live.message_rows`, `top_rows`, `commit_rows`) -/

/-- `sha512/row/v1`'s digest of a row of `ROW_WORDS` u16 words, role x (`HmRow.rowPrefix`;
`verity.primitives.commitments.rowleaf.sha512_row_digest`). -/
def rowDigest (row : Array Nat) : ByteArray :=
  let pre := Hm96.pad ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16] ++ be32 row.size.toUInt32) 128
  Sha512.hash (row.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8) pre)

/-- `F_k` as its round's one row (none for an empty message): each element 8 u16 words, lo then hi, then zero words up to
whole blocks of `ROW_WORDS`. -/
def messageRows (free : Array Nat) : List (Array Nat) :=
  let words := free.foldl (fun ws v => ws ++ (Array.range 8).map fun j => v / 65536 ^ j % 65536) #[]
  if words.isEmpty then [] else [words ++ Array.replicate ((ROW_WORDS - words.size % ROW_WORDS) % ROW_WORDS) 0]

/-- A tree top's nodes as rows: each 64-byte node as 32 little-endian u16 words, then 32 zero words. -/
def topRows (top : ByteArray) : List (Array Nat) :=
  (List.range (top.size / NODE_BYTES)).map fun i =>
    (Array.range 32).map (fun j => (top.get! (64 * i + 2 * j)).toNat + 256 * (top.get! (64 * i + 2 * j + 1)).toNat) ++
      Array.replicate 32 0

def nodes (top : ByteArray) : V Nat := do
  if !isCap top then throw s!"a tree top of {top.size} bytes"
  return top.size / NODE_BYTES

/-! ## The schedule (`rec_live.check_schedule`) -/

def natOf? : Json → Option Nat
  | .num ⟨m, 0⟩ => if m ≥ 0 then some m.toNat else none
  | _ => none

def field (j : Json) (k : String) : V Json := j.getObjVal? k

def nat (j : Json) (k : String) : V Nat := do
  match natOf? (← field j k) with
  | some n => return n
  | none => throw s!"{k}: not a natural number"

def str (j : Json) (k : String) : V String := do
  match ← field j k with
  | .str s => return s
  | _ => throw s!"{k}: not a string"

def obj (j : Json) : V (List (String × Json)) := do
  match j with
  | .obj kvs => return kvs.toArray.toList
  | _ => throw "not an object"

def arr (j : Json) : V (Array Json) := do
  match j with
  | .arr a => return a
  | _ => throw "not an array"

structure Schedule where
  hello : Json
  roots : Json
  y : Nat
  m : Nat
  streams : List (String × String × Nat × Nat)
  rounds : Array Json

/-- The schedule's tables in its order: the names of its `roots`, as its JSON object holds them, ascending. Commit names
them in this order and no other, so the order carries nothing of the worker's. -/
def Schedule.tables (s : Schedule) : List String :=
  match s.roots with
  | .obj kvs => kvs.toArray.toList.map (·.1)
  | _ => []

def jnat (n : Nat) : Json := .num ⟨Int.ofNat n, 0⟩

/-- A round's schedule entry read field by field: its stream, `k`, `n`, each fresh top's node count, its copies of a
level-0 top and its message rows (`none` for an entry `schedule` refuses). What the ZK server sends for a round is pinned
to these, beside the canonical compare, so that what it sends is the schedule's value. -/
def roundSpec (j : Json) : Option (String × Nat × Nat × List Nat × Nat × Nat) :=
  Except.toOption (ε := String) do
    let tops ← (← arr (← field j "tops")).toList.mapM fun c => match natOf? c with
      | some v => pure v
      | none => throw "tops"
    return (← str j "stream", ← nat j "k", ← nat j "n", tops, ← nat j "level0", ← nat j "commits")

/-- A round's entry of just the six fields `roundSpec` reads. -/
def specJson : String × Nat × Nat × List Nat × Nat × Nat → Json
  | (stream, k, n, tops, level0, commits) => Json.mkObj [("stream", .str stream), ("k", jnat k), ("n", jnat n),
    ("tops", .arr (tops.map jnat).toArray), ("level0", jnat level0), ("commits", jnat commits)]

/-- A stream's table: its name without its last `/` part (`rec_live`'s `stream.rsplit("/", 1)[0]`). -/
def tableOf (stream : String) : String :=
  let parts := stream.splitOn "/"
  if parts.length ≤ 1 then stream else "/".intercalate parts.dropLast

/-- How many caps of the schedule's round `i` copy level 0's top: its first `level0`. Copies are by position, never by
value: every later cap is a fresh top, whatever its value. -/
def copiesOf (s : Schedule) (i : Nat) : Nat := ((roundSpec (s.rounds.getD i .null)).map (·.2.2.2.2.1)).getD 0

/-! ## Canonical forms (`rec_live.canonical`) -/

/-- A string `JsonParse` reads back unescaped: no `"`, `\` or control character. -/
def plainStr (s : String) : Bool := s.all fun c => c != '"' && c != '\\' && 0x20 ≤ c.val

/-- An object's keys ascend in its tree's order, as in every tree `JsonParse` builds. -/
def ascending (t : Std.DTreeMap.Internal.Impl String (fun _ => Json)) : Bool :=
  decide (t.toListModel.Pairwise fun a b => compare a.1 b.1 = .lt)

mutual
/-- A value that prints back as it parses: its numbers naturals, its strings and keys plain (`plainStr`), each object's
keys ascending (`ascending`). -/
def plain : Json → Bool
  | .null => true
  | .bool _ => true
  | .num ⟨m, e⟩ => e == 0 && decide (0 ≤ m)
  | .str s => plainStr s
  | .arr ⟨a⟩ => plainList a
  | .obj ⟨⟨t⟩⟩ => plainObj t && ascending t
def plainList : List Json → Bool
  | [] => true
  | j :: js => plain j && plainList js
def plainObj : Std.DTreeMap.Internal.Impl String (fun _ => Json) → Bool
  | .leaf => true
  | .inner _ k v l r => plainObj l && plainStr k && plain v && plainObj r
end

mutual
/-- A value's text as the honest worker writes it (`serde_json`'s `to_string`): no whitespace, each object's members in
its tree's order, each number its mantissa in decimal. A plain value's text parses back to it (`plain`). -/
def jsonChars : Json → List Char
  | .null => ['n', 'u', 'l', 'l']
  | .bool true => ['t', 'r', 'u', 'e']
  | .bool false => ['f', 'a', 'l', 's', 'e']
  | .num ⟨m, _⟩ => Nat.toDigits 10 m.toNat
  | .str s => '"' :: s.toList ++ ['"']
  | .arr ⟨a⟩ => '[' :: [','].intercalate (jsonCharsList a) ++ [']']
  | .obj ⟨⟨t⟩⟩ => '{' :: [','].intercalate (jsonCharsObj t) ++ ['}']
def jsonCharsList : List Json → List (List Char)
  | [] => []
  | j :: js => jsonChars j :: jsonCharsList js
def jsonCharsObj : Std.DTreeMap.Internal.Impl String (fun _ => Json) → List (List Char)
  | .leaf => []
  | .inner _ k v l r => jsonCharsObj l ++ [('"' :: k.toList ++ ['"', ':']) ++ jsonChars v] ++ jsonCharsObj r
end

def printJson (j : Json) : String := String.ofList (jsonChars j)

/-- **The schedules the ZK server holds a session to**, those an honest worker's session can finish; `schedule` refuses
every other:
* Hello is plain (`plain`), so it prints back as it parses, names no `coins`, and is an object whose `link` is an object
  without `sigma`;
* the roots are an object of naturals, its keys ascending (`ascending`), and the streams' names ascend, as in every
  tree `JsonParse` builds;
* each round's entry is its six fields and nothing more (`specJson`), and its stream is one of the schedule's;
* each stream has a round, and its rounds come in the schedule's order with `k` = 0, 1, …, as many as its `rounds`;
* a stream without a parent, and a stream with a round that copies level 0's top, is of a table with a root. -/
def Schedule.canonical (s : Schedule) : Bool :=
  let rooted := fun (stream : String) => (s.roots.getObjVal? (tableOf stream)).toOption.isSome
  plain s.hello && (s.hello.getObjVal? "coins").toOption.isNone &&
    (match s.hello.getObjVal? "link" with
      | .ok l@(.obj _) => (l.getObjVal? "sigma").toOption.isNone
      | _ => false) &&
    (match s.roots with
      | .obj ⟨⟨t⟩⟩ => ascending t && t.toListModel.all fun e => (natOf? e.2).isSome
      | _ => false) &&
    decide ((s.streams.map (·.1)).Pairwise fun a b => compare a b = .lt) &&
    s.rounds.all (fun r => match roundSpec r with
      | some sp => canonForm r == canonForm (specJson sp) && (sp.2.2.2.2.1 == 0 || rooted sp.1) &&
          s.streams.any (·.1 == sp.1)
      | none => false) &&
    s.streams.all fun (name, parent, _, count) =>
      0 < count && (parent != "" || rooted name) &&
        s.rounds.toList.filterMap (fun r => (roundSpec r).bind fun sp => if sp.1 == name then some sp.2.1 else none) ==
          List.range count

def schedule (s : Json) : V Schedule := do
  let e : V Schedule := do
    if (← str s "kind") != "rec/inner-transcript/v1/schedule" then throw "kind"
    let h ← field s "hello"
    let _ ← obj h
    let l ← field h "link"
    let _ ← obj l
    if (l.getObjVal? "sigma").toOption.isSome then throw "sigma"
    let m ← nat l "m"
    let _ ← nat l "points"
    let y ← nat (← field s "link") "y"
    let roots ← field s "roots"
    for (_, v) in ← obj roots do
      if (natOf? v).isNone then throw "roots"
    let mut streams := #[]
    for (n, v) in ← obj (← field s "streams") do
      streams := streams.push (n, ← str v "parent", ← nat v "at", ← nat v "rounds")
    let rounds ← arr (← field s "rounds")
    for r in rounds do
      let st ← str r "stream"
      if !(streams.any (·.1 == st)) then throw "a round's stream"
      let _ ← nat r "k"
      let _ ← nat r "n"
      for c in ← arr (← field r "tops") do
        if (natOf? c).isNone then throw "tops"
      let _ ← nat r "level0"
      let _ ← nat r "commits"
    let sc : Schedule := { hello := h, roots, y, m, streams := streams.toList, rounds }
    if !sc.canonical then throw "canonical"
    return sc
  match e with
  | .ok v => return v
  | .error _ => throw "not a rec/inner-transcript/v1/schedule"

/-! ## The ZK server (`rec_live.Proxy`) -/

structure State where
  /-- what the challenger's live-coin part has received, in order -/
  sent : Array Sent := #[]
  hello : Bool := false
  committed : Bool := false
  /-- each table's level-0 top, as `Commit` gave it -/
  roots : List (String × ByteArray) := []
  linked : Bool := false
  /-- each opened stream: its parent and its rounds so far -/
  streams : List (String × String × Nat) := []
  rounds : Nat := 0
  stopped : Bool := false
  finished : Bool := false
  /-- the ZK server's salts not yet sampled -/
  salts : List ByteArray

/-- What the ZK server got equals the pinned schedule's, compared as `rec_live` compares them, by canonical form. -/
def pinned (what : String) (got want : Json) : V Unit := do
  if canonForm got != canonForm want then
    throw s!"FIREWALL-REFUSED: {what}: {got.compress} where the pinned schedule has {want.compress}"

abbrev Fw := StateT State V

def salt : Fw ByteArray := do
  let st ← get
  match st.salts with
  | y :: rest => set { st with salts := rest }; return y
  | [] => throw "the firewall's salts ran out"

def salted (row : Array Nat) : Fw ByteArray := do
  return Hm96.Default512.commitString (rowDigest row) (← salt)

def saltedTop (top : ByteArray) : Fw (List ByteArray) := (topRows top).mapM salted

def send (m : Sent) : Fw Unit := modify fun st => { st with sent := st.sent.push m }

/-- The text has a number with a fraction or an exponent, outside strings. `rec_live` compares a `Hello` as Python's
canonical JSON, where `2.0` and `2` differ, and Lean's parser keeps no spelling, so the ZK server refuses any such number. -/
def fractional (text : String) : Bool := Id.run do
  let mut inStr := false
  let mut esc := false
  let mut inNum := false
  for c in text.toList do
    if inStr then
      if esc then esc := false
      else if c == '\\' then esc := true
      else if c == '"' then inStr := false
    else if c == '"' then
      inStr := true
      inNum := false
    else if c.isDigit || c == '-' then inNum := true
    else if inNum && (c == '.' || c == 'e' || c == 'E') then return true
    else inNum := false
  return false

def withField (j : Json) (k : String) (x : Json) : Json :=
  match j with
  | .obj kvs => Json.mkObj ((kvs.toArray.toList.filter (·.1 != k)) ++ [(k, x)])
  | _ => j

def without (j : Json) (k : String) : Json :=
  match j with
  | .obj kvs => Json.mkObj (kvs.toArray.toList.filter (·.1 != k))
  | _ => j

/-- One request, as `Proxy._handle`: a throw stops the session. -/
def step (s : Schedule) : Req → Fw Unit
  | .ping => pure ()
  | .register => do if (← get).finished then throw "session finished"
  | .proof => do if (← get).finished then throw "session finished"
  | .prime => throw "request tag 8 has no place in an inner session"
  | .hello text => do
    if (← get).finished then throw "session finished"
    if (← get).hello then throw "a second Hello"
    let v ← match JsonParse.parse text with
      | .ok v => pure v
      | .error e => throw e
    let ok := match v.getObjVal? "link" with
      | .ok l@(.obj _) => match l.getObjVal? "sigma" with | .ok (.str _) => true | _ => false
      | _ => false
    if !ok then throw "FIREWALL-REFUSED: Hello: not an object whose link names Σ"
    if (v.getObjVal? "coins").toOption.isSome then throw "the inner session runs on live OS coins"
    if fractional text then throw "FIREWALL-REFUSED: Hello: a number that is not an integer"
    let link := without ((v.getObjVal? "link").toOption.getD .null) "sigma"
    pinned "Hello" (withField v "link" link) s.hello
    modify fun st => { st with hello := true }
    send (.hello s.hello)
  | .commit _ roots publics => do
    if (← get).finished then throw "session finished"
    if publics.any (·.2.size > 0) then throw "public words in an inner session"
    if (roots.map (·.1)).eraseDups.length != roots.length then throw "FIREWALL-REFUSED: Commit names a table twice"
    let counts ← roots.mapM fun (n, v) => do return (n, jnat (← nodes v))
    pinned "Commit's roots" (Json.mkObj counts) s.roots
    if roots.any fun (n, v) => natOf? (s.roots.getObjValD n) != some (v.size / NODE_BYTES) then
      throw "FIREWALL-REFUSED: Commit's roots: a table's node count is not the pinned schedule's"
    if roots.map (·.1) != s.tables then
      throw "FIREWALL-REFUSED: Commit's roots: the tables are not in the pinned schedule's order"
    let tops ← roots.mapM fun (n, v) => do return (n, ← saltedTop v)
    let st ← get
    if !st.hello || st.committed then throw "the link points are drawn once, after Hello"
    set { st with committed := true, roots }
    send (.commit tops)
  | .link y => do
    if (← get).finished then throw "session finished"
    if y.size != 16 * s.y then throw s!"FIREWALL-REFUSED: Link's y: {y.size} bytes"
    if y.toList.any (· != 0) then throw "FIREWALL-REFUSED: Link's y is not zero"
    let st ← get
    if !st.committed || st.linked then throw "y follows the link points, once"
    set { st with linked := true }
    send (.link s.y)
  | .openStream stream parent _ pos => do
    if (← get).finished then throw "session finished"
    let some (_, wp, wa, _) := s.streams.find? (·.1 == stream) | throw s!"FIREWALL-REFUSED: Open {stream}"
    if parent != wp || pos != wa then throw s!"FIREWALL-REFUSED: Open {stream}"
    let st ← get
    if st.streams.any (·.1 == stream) then throw s!"stream {stream} opened twice"
    set { st with streams := st.streams ++ [(stream, parent, 0)] }
  | .round stream content n => do
    if (← get).finished then throw "session finished"
    let some (_, parent, k) := (← get).streams.find? (·.1 == stream) | throw s!"a round on stream {stream}, never opened"
    let sp ← split content (k == 0 && parent == "")
    if sp.n != n then throw s!"the round squeezes {sp.n} coins, the prover asked for {n}"
    let root := ((← get).roots.find? (·.1 == tableOf stream)).map (·.2)
    if k == 0 && parent == "" then
      match root, sp.caps[0]? with
      | some rootB, some c => if c != rootB then throw s!"{stream}: the binding round's cap is not the committed root_B"
      | _, _ => throw s!"{stream}: the binding round's cap is not the committed root_B"
    let i := (← get).rounds
    let copies := copiesOf s i
    if sp.caps.size < copies || (sp.caps.toList.take copies).any (some · != root) then
      throw s!"FIREWALL-REFUSED: round {i}: a copy of level 0's top is not {tableOf stream}'s committed top"
    let fresh := sp.caps.toList.drop copies
    let tops ← fresh.mapM saltedTop
    let rows := messageRows sp.free
    let topNodes ← fresh.mapM fun c => do return jnat (← nodes c)
    let got := Json.mkObj [("stream", .str stream), ("k", jnat k), ("n", jnat n), ("tops", .arr topNodes.toArray),
      ("level0", jnat copies), ("commits", jnat rows.length)]
    pinned s!"round {i}" got (s.rounds.getD i .null)
    if roundSpec (s.rounds.getD i .null) ≠ some (stream, k, n, fresh.map (·.size / NODE_BYTES), copies, rows.length) then
      throw s!"FIREWALL-REFUSED: round {i}: not the pinned schedule's"
    let bcs ← rows.mapM salted
    if !(← get).linked then throw "a coin before the link exchange"
    if k == 0 then
      let some (_, wp, pos, _) := s.streams.find? (·.1 == stream) | throw s!"FIREWALL-REFUSED: Open {stream}"
      send (.openStream stream wp pos)
    modify fun st => { st with rounds := st.rounds + 1,
                               streams := st.streams.map fun e => if e.1 == stream then (e.1, e.2.1, e.2.2 + 1) else e }
    send (.round stream tops copies bcs n)
  | .finish => do
    let st ← get
    if st.finished then throw "session finished"
    if !st.committed || !st.linked then throw "Finish before the link exchange"
    pinned "Finish's rounds" (jnat st.rounds) (jnat s.rounds.size)
    if st.rounds ≠ s.rounds.size then throw "FIREWALL-REFUSED: Finish's rounds: not the pinned schedule's"
    let got := Json.mkObj (st.streams.map fun (n, _, k) => (n, jnat k))
    let want := Json.mkObj (s.streams.map fun (n, _, _, k) => (n, jnat k))
    pinned "Finish's streams" got want
    set { st with finished := true }
    send (.finish true)

/-- The ZK server over a session: each request in turn, stopping at the first refusal; a session that ends without
`Finish` stops. A request after `Finish` (other than `Ping`) stops it too, as `rec_live` records. -/
def run (s : Schedule) (reqs : List ByteArray) (salts : List ByteArray) : Array Sent × Bool := Id.run do
  let mut st : State := { salts }
  let mut stopped := false
  for b in reqs do
    if stopped then
      continue
    match (show Fw Unit from do step s (← decode b)).run st with
    | .ok ((), st') => st := st'
    | .error _ => stopped := true
  return (st.sent, st.finished && !stopped)

/-! ## The challenger's live-coin part's record (`rec_live.CoinServer.record`, without the coins it drew) -/

def hexs (l : List ByteArray) : Json := .arr (l.map (Json.str ∘ hexOf)).toArray

def record (sent : Array Sent) (ok : Bool) : Json := Id.run do
  let mut hello := Json.null
  let mut link := Json.null
  let mut streams : List (String × Json) := []
  let mut rounds : Array Json := #[]
  let mut counts : List (String × Nat) := []
  let mut rows := 0
  for m in sent do
    match m with
    | .hello h => hello := h
    | .commit roots =>
      link := Json.mkObj [("roots", Json.mkObj (roots.map fun (n, ns) => (n, hexs ns))), ("y", .null)]
    | .link z => link := withField link "y" (.arr (Array.replicate z (.str ("".pushn '0' 32))))
    | .openStream st p pos => streams := streams ++ [(st, Json.mkObj [("parent", .str p), ("at", jnat pos)])]
    | .round st tops level0 bcs n =>
      let k := (counts.find? (·.1 == st)).map (·.2) |>.getD 0
      counts := (st, k + 1) :: counts.filter (·.1 != st)
      rows := rows + bcs.length
      rounds := rounds.push (Json.mkObj [("stream", .str st), ("k", jnat k), ("tops", .arr (tops.map hexs).toArray),
        ("level0", jnat level0), ("commits", hexs bcs), ("n", jnat n)])
    | .finish _ => pure ()
  let named := streams.filterMap fun (n, v) =>
    (counts.find? (·.1 == n)).map fun (_, k) => (n, withField v "rounds" (jnat k))
  return Json.mkObj [("kind", .str "rec/inner-transcript/v1"), ("tops", .str "hm96-sha512/v1/node-row"), ("hello", hello),
    ("link", link), ("streams", Json.mkObj named), ("rounds", .arr rounds), ("message_rows", jnat rows),
    ("row_words", jnat ROW_WORDS), ("end", .str (if ok then "finish" else "stop"))]

/-- The challenger's live-coin part's calls, in order, as `rec_live.CoinServer`'s methods: `start`, `commit`, `y`, `open` with its stream,
parent and position, `round` with its stream and coin count. -/
def calls (sent : Array Sent) : Json :=
  .arr (sent.filterMap fun
    | .hello _ => some (.arr #[.str "start"])
    | .commit _ => some (.arr #[.str "commit"])
    | .link _ => some (.arr #[.str "y"])
    | .openStream st p pos => some (.arr #[.str "open", .str st, .str p, jnat pos])
    | .round st _ _ _ n => some (.arr #[.str "round", .str st, jnat n])
    | .finish _ => none)

/-! ## A session on the challenger's coin tree (`rec_live --coin-tree`)

The prover's `Hello` names the challenger's coin tree under `coins`: the tree's spec, and a nonce of the prover's own, which
the session never uses (the challenger's nonce keys the tree). The challenger's live-coin part is sent the schedule's `Hello` either way,
so the ZK server's messages are the live-coin session's. `treeHello` reads such a request as `step` reads one on live coins:
a `Hello` whose `coins` without its nonce is the pinned spec, as that `Hello` without `coins`; any other `Hello` stops the
session, and every other request is unchanged. -/

def u32le (n : Nat) : ByteArray := ⟨(List.range 4).toArray.map fun i => (n / 256 ^ i % 256).toUInt8⟩

def treeHello (spec : Json) (b : ByteArray) : V ByteArray := do
  let .hello text ← decode b | return b
  if fractional text then throw "FIREWALL-REFUSED: Hello: a number that is not an integer"
  let v ← JsonParse.parse text
  let some c := (v.getObjVal? "coins").toOption
    | throw s!"FIREWALL-REFUSED: Hello: no coins where the challenger's coin tree is {spec.compress}"
  if (without c "nonce").compress != spec.compress then
    throw s!"FIREWALL-REFUSED: Hello: coins {c.compress} where the challenger's coin tree is {spec.compress}"
  let t := (without v "coins").compress.toUTF8
  return ByteArray.mk #[1] ++ u32le t.size ++ t

/-- The requests as `step` reads them: under a pinned coin tree (`some spec`), through `treeHello`. -/
def onTree : Option Json → ByteArray → V ByteArray
  | none, b => pure b
  | some spec, b => treeHello spec b

/-- One case of `flock-firewall`'s input, `{"schedule", "requests": [hex], "salts": [hex]}`, and `"coin_tree": spec` for a
session on the challenger's coin tree: the record; with `"calls": true`, `{"record", "calls"}`. Under a coin tree the
requests are read through `treeHello`, and the first it refuses ends them, as `run` stops at a refusal. -/
def case (j : Json) : V Json := do
  let s ← schedule (← field j "schedule")
  let hexList := fun (k : String) => do
    let a ← arr (← field j k)
    a.toList.mapM fun x => match x with
      | .str h => unhex h
      | _ => throw s!"{k}: not hex"
  let tree := (j.getObjVal? "coin_tree").toOption
  let reqs := ((← hexList "requests").map (onTree tree)).takeWhile (·.toOption.isSome) |>.filterMap (·.toOption)
  let reqs := if reqs.length < (← hexList "requests").length then reqs ++ [ByteArray.empty] else reqs
  let (sent, ok) := run s reqs (← hexList "salts")
  if j.getObjValD "calls" == .bool true then
    return Json.mkObj [("record", record sent ok), ("calls", calls sent)]
  return record sent ok

/-- `flock-firewall`: one case a line on stdin, one record a line on stdout (`{"error": …}` for a case it cannot read). -/
def cmd (_ : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    if line.all Char.isWhitespace then continue
    let out := match JsonParse.parse line >>= case with
      | .ok r => r
      | .error e => Json.mkObj [("error", .str e)]
    stdout.putStrLn out.compress
    stdout.flush
  return 0

/-! ## `flock-firewall live`: the ZK server as a live session's send check (`rec_live --lean`) -/

/-- A salt's bytes, `hm96-sha512/v1`'s. -/
def SALT_BYTES : Nat := 192

/-- A message to the challenger's live-coin part, as `rec_live` relays it to `CoinServer`'s methods. -/
def sentJson : Sent → Json
  | .hello h => Json.mkObj [("hello", h)]
  | .commit roots => Json.mkObj [("commit", .arr (roots.map fun (n, t) => Json.arr #[.str n, hexs t]).toArray)]
  | .link z => Json.mkObj [("link", jnat z)]
  | .openStream st p pos => Json.mkObj [("open", .arr #[.str st, .str p, jnat pos])]
  | .round st tops level0 rows n => Json.mkObj [("round", Json.mkObj [("stream", .str st),
      ("tops", .arr (tops.map hexs).toArray), ("level0", jnat level0), ("commits", hexs rows), ("n", jnat n)])]
  | .finish ok => Json.mkObj [("finish", .bool ok)]

/-- One request of a live session at the state `st`, its salts those `st` holds unsampled then `fresh`: what it sends, the
salts it drew in order and the state after it, holding the salts it didn't sample, or the refusal that stops the session.
On a session's requests in turn from `{ salts := [] }`, each with the fresh salts read for it, this is `run`'s loop at
those salts in order (`FirewallLive.run_live`). -/
def liveStep (s : Schedule) (st : State) (b : ByteArray) (fresh : List ByteArray) :
    Except String (List Sent × List ByteArray × State) :=
  let pool := st.salts ++ fresh
  match (show Fw Unit from do step s (← decode b)).run { st with salts := pool, sent := #[] } with
  | .ok ((), st') => .ok (st'.sent.toList, pool.take (pool.length - st'.salts.length), { st' with sent := #[] })
  | .error e => .error e

/-- `flock-firewall live` at the byte source `get`: one session. Its first line is `{"schedule": …}`, with `"coin_tree":
spec` for a session on the challenger's coin tree (each request then read through `treeHello`), answered
`{"ok": true}`; each later line is one request in hex, answered with what the ZK server sends the challenger's live-coin part for it and
the salts it drew for them, `{"sent": [sentJson…], "salts": [hex…]}`, or with the refusal that stops the session,
`{"stop": why}`, which every later line is answered with too. Every line is a request, an empty one an empty frame. A
request's fresh salts are read from `get` before it runs, as many as it could sample (a salted row reads at least one node
of 64 request bytes, a round's message row aside), and those it doesn't sample are the next request's first: the salts
sampled are the first of the salts read, in order. -/
def liveWith (get : USize → IO ByteArray) (_ : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let out := fun (j : Json) => do
    stdout.putStrLn j.compress
    stdout.flush
  let first ← stdin.getLine
  let some s := (JsonParse.parse first >>= (field · "schedule") >>= schedule).toOption
    | out (Json.mkObj [("error", .str "the first line is not {\"schedule\": a rec/inner-transcript/v1/schedule}")])
      return 1
  let tree := (JsonParse.parse first >>= (·.getObjVal? "coin_tree")).toOption
  out (Json.mkObj [("ok", .bool true)])
  let mut st : State := { salts := [] }
  let mut stop : Option String := none
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    if let some why := stop then
      out (Json.mkObj [("stop", .str why)])
      continue
    match unhex line.trim >>= onTree tree with
    | .error e =>
      stop := some e
      out (Json.mkObj [("stop", .str e)])
    | .ok b =>
      let need := b.size / NODE_BYTES + 1
      let pool ← get (SALT_BYTES * need).toUSize
      let fresh := (List.range need).map fun i => pool.extract (SALT_BYTES * i) (SALT_BYTES * (i + 1))
      match liveStep s st b fresh with
      | .ok (sent, used, st') =>
        st := st'
        out (Json.mkObj [("sent", .arr (sent.map sentJson).toArray), ("salts", hexs used)])
      | .error e =>
        stop := some e
        out (Json.mkObj [("stop", .str e)])
  return 0

/-- `flock-firewall live` at the operating system's bytes. -/
def liveCmd : List String → IO UInt32 := liveWith IO.getRandomBytes

end Flock.Firewall
