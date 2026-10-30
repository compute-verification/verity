/-
The firewall's contract, inner phase (live PROTOCOL.md §10.3): what the firewall may send the challenger's coin server, as a
program, so that `rec_live`'s proxy is tested against it (`flock-firewall`, `backends/flock/tests/test_firewall_agreement.py`).
-/
import Lean.Data.Json
import Flock.Hm96

namespace Flock.Firewall

open Lean (Json JsonNumber)

/-! ## The contract

Before release, every value the firewall sends the auditor is one of three things:

* **salted**: a commitment, `Hm96.Default512.commitString`, under a fresh salt the workers never see;
* **masked**: the value plus a fresh pad the workers never see, released only after every relation the verifier will
  check holds of the value (the outer phase, §10.1; not in this file);
* **fixed**: uniquely fixed by what is already public (the pinned schedule) or committed.

Everything else a worker sends (the statement digest, labels, the domain, nonces, the hidden statement's Σ) is
**dropped**, and a request that leaves the schedule stops the session, which the auditor sees as `end: "stop"`.

For the inner phase that is the type `Sent`: each message to the coin server is the schedule's own value, a count the
schedule fixes, zeros, or a list of salted commitments. Nothing a worker sends reaches it except under a salt from the
firewall's stream. `run` is the firewall: the pinned schedule, the worker's requests, the firewall's salts in, the coin
server's record out. -/

/-- A message from the firewall to the coin server. -/
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
    return { hello := h, roots, y, m, streams := streams.toList, rounds }
  match e with
  | .ok v => return v
  | .error _ => throw "not a rec/inner-transcript/v1/schedule"

/-! ## The firewall (`rec_live.Proxy`) -/

structure State where
  /-- what the coin server has received, in order -/
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
  /-- the firewall's salts not yet drawn -/
  salts : List ByteArray

def jnat (n : Nat) : Json := .num ⟨Int.ofNat n, 0⟩

def pinned (what : String) (got want : Json) : V Unit := do
  if got.compress != want.compress then
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
canonical JSON, where `2.0` and `2` differ, and Lean's parser keeps no spelling, so the firewall refuses any such number. -/
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
    let v ← match Json.parse text with
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
    let roots := (← get).roots
    let level0 := fun (c : ByteArray) => roots.any (·.2 == c)
    if k == 0 && parent == "" then
      let parts := stream.splitOn "/"
      let table := if parts.length ≤ 1 then stream else "/".intercalate parts.dropLast
      match roots.find? (·.1 == table), sp.caps[0]? with
      | some (_, rootB), some c => if c != rootB then throw s!"{stream}: the binding round's cap is not the committed root_B"
      | _, _ => throw s!"{stream}: the binding round's cap is not the committed root_B"
    let fresh := sp.caps.toList.filter (!level0 ·)
    let copies := (sp.caps.toList.filter level0).length
    let tops ← fresh.mapM saltedTop
    let rows := messageRows sp.free
    let i := (← get).rounds
    let topNodes ← fresh.mapM fun c => do return jnat (← nodes c)
    let got := Json.mkObj [("stream", .str stream), ("k", jnat k), ("n", jnat n), ("tops", .arr topNodes.toArray),
      ("level0", jnat copies), ("commits", jnat rows.length)]
    pinned s!"round {i}" got (s.rounds.getD i .null)
    let bcs ← rows.mapM salted
    if !(← get).linked then throw "a coin before the link exchange"
    if k == 0 then
      let some (_, _, pos, _) := s.streams.find? (·.1 == stream) | throw s!"FIREWALL-REFUSED: Open {stream}"
      send (.openStream stream parent pos)
    modify fun st => { st with rounds := st.rounds + 1,
                               streams := st.streams.map fun e => if e.1 == stream then (e.1, e.2.1, e.2.2 + 1) else e }
    send (.round stream tops copies bcs n)
  | .finish => do
    let st ← get
    if st.finished then throw "session finished"
    if !st.committed || !st.linked then throw "Finish before the link exchange"
    pinned "Finish's rounds" (jnat st.rounds) (jnat s.rounds.size)
    let got := Json.mkObj (st.streams.map fun (n, _, k) => (n, jnat k))
    let want := Json.mkObj (s.streams.map fun (n, _, _, k) => (n, jnat k))
    pinned "Finish's streams" got want
    set { st with finished := true }
    send (.finish true)

/-- The firewall over a session: each request in turn, stopping at the first refusal; a session that ends without
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

/-! ## The coin server's record (`rec_live.CoinServer.record`, without the coins it drew) -/

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

/-- The coin server's calls, in order, as `rec_live.CoinServer`'s methods: `start`, `commit`, `y`, `open` with its stream,
parent and position, `round` with its stream and coin count. -/
def calls (sent : Array Sent) : Json :=
  .arr (sent.filterMap fun
    | .hello _ => some (.arr #[.str "start"])
    | .commit _ => some (.arr #[.str "commit"])
    | .link _ => some (.arr #[.str "y"])
    | .openStream st p pos => some (.arr #[.str "open", .str st, .str p, jnat pos])
    | .round st _ _ _ n => some (.arr #[.str "round", .str st, jnat n])
    | .finish _ => none)

/-- One case of `flock-firewall`'s input, `{"schedule", "requests": [hex], "salts": [hex]}`: the record; with
`"calls": true`, `{"record", "calls"}`. -/
def case (j : Json) : V Json := do
  let s ← schedule (← field j "schedule")
  let hexList := fun (k : String) => do
    let a ← arr (← field j k)
    a.toList.mapM fun x => match x with
      | .str h => unhex h
      | _ => throw s!"{k}: not hex"
  let (sent, ok) := run s (← hexList "requests") (← hexList "salts")
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
    let out := match Json.parse line >>= case with
      | .ok r => r
      | .error e => Json.mkObj [("error", .str e)]
    stdout.putStrLn out.compress
    stdout.flush
  return 0

end Flock.Firewall
