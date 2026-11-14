import Spot.ChallengerSession

/-!
`ledger-replay-time S REQUEST EXPECT [REPEAT]`: one replay of an auditor's ledger of S sessions by the checkout's
verifier, timed. REQUEST is the `challenger_ledger` piece's request, `{"key", "ledger", "window"}`, and EXPECT what the
piece's reference answers on it (`ledger_replay.py` writes both). One `PHASE {json}` line each on stdout, its
microseconds (`us`) the least of REPEAT timings (default 3):

* `request`: the two files' read and parse, once: the harness's work, no part of a replay;
* `piece`: the difftest's piece (`Challenger.ledgerPiece`: `auditedChallengePhase`, one replay and the window's lookup);
* `checked`: the replay alone (`Challenger.checked`: each line's parse, then `verifyLedger`);
* `entries`: each line's reading (`ledgerEntry?`), as `checked` makes it, and four parts of it: `parse`
  (`Flock.JsonParse.parse`), `canonical` (`PyJson.canonical`, held to the line), `canon` (`Flock.canon`) and `record`
  (`Core.Record.ofJson`); what `entries` takes beyond them is the line's length, its scan for an escaped surrogate, its
  keys' order and its body's form;
* `chain`: each line's SHA-512 in hex, which the next entry's `prev` is held to;
* `messages`: what each entry's signature signs (`ledgerMessage`);
* `signatures`: each signature's decode and its LMS verification under the key;
* `searches`: the searches of `verifyLedger`'s loop over the windows, reserves, record leaves and entry leaves it has
  read so far (`searches` below).

`entries` to `signatures` are the replay's work that is one pass over the entries, made here as the replay makes it;
what `checked` takes beyond them holds its searches and each entry's checks of its body. `searches` is a copy of the
loop's searches and not the loop: the verifier's own loop is not timed in parts, which would take a change to it. Exit
1 if the piece's answer is not EXPECT's, or a signature of a ledger the replay accepts doesn't verify here.
`ledger_replay.sh` builds it against the checkout's verifier and runs it; it is a measurement, not a verifier.
-/

open Lean Spot Spot.Challenger

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

/-- `f`'s value and the least of `n` timings of it, in microseconds; `f` reads the clock's reading and `size` reads its
value before the clock stops, so the compiler can't move it out of the timing. -/
def timed {α : Type} (n : Nat) (f : Nat → α) (size : α → Nat) : IO (α × Nat) := do
  let t0 ← IO.monoNanosNow
  let mut a := f (t0 + 1)
  if size a == 1000000007 then IO.eprintln "unreachable"
  let mut best := ((← IO.monoNanosNow) - t0) / 1000
  for _ in [1:n] do
    let t0 ← IO.monoNanosNow
    a := f (t0 + 1)
    if size a == 1000000007 then IO.eprintln "unreachable"
    best := min best (((← IO.monoNanosNow) - t0) / 1000)
  return (a, best)

def row (phase : String) (sessions entries us : Nat) (extra : List (String × Json)) : IO Unit := do
  let hwm ← statusKb "VmHWM:"
  IO.println s!"PHASE {(Json.mkObj ([("phase", Json.str phase), ("sessions", toJson sessions),
    ("entries", toJson entries), ("us", toJson us), ("hwm_mb", toJson (hwm / 1024))] ++ extra)).compress}"
  (← IO.getStdout).flush

/-- Whether `sig` signs `m` under `key`, as the replay's own check of an entry reads it. -/
def signs (key : ChallengerKey) (m : Flock.V ByteArray) (sig : ByteArray) : Bool :=
  match m, Lms.Signature.decode key.params sig with
  | .ok m, .ok s => (Lms.verify key.params key.pub m s).toOption == some true
  | _, _ => false

/-- `verifyLedger`'s searches on a ledger it accepts, in its loop's order and on the arrays its loop builds, with none
of its checks: at a reserve, the reserves and the windows for its session; at an open, the windows for its root and its
session and the reserves for its session; at an end, the windows for its session and the record leaves for its own; at
a period's close, the windows for each session the period holds (those reserved since its open, a ledger of one lane);
and after the loop, the entries' leaves for each record leaf. The count of what they found, so that each is made. -/
def searches (es : Array LedgerEntry) : Nat := Id.run do
  let mut wins : Array Win := #[]
  let mut reserves : Array Reserve := #[]
  let mut records : Array (Nat × Nat) := #[]
  let mut leaves : Array Nat := #[]
  let mut joined : Array String := #[]
  let mut found := 0
  for (e, i) in es.zipIdx do
    let s := e.session
    match e.kind with
    | "reserve" =>
      if reserves.any (·.session == s) || wins.any (·.opened.session == s) then found := found + 1
      reserves := reserves.push { session := s, entry := e, period := none }
      joined := joined.push s
    | "open" =>
      let root := match getD (getD e.body "window") "root" with | .str r => r | _ => ""
      if wins.any (·.root == root) || wins.any (·.opened.session == s) then found := found + 1
      if (reserves.find? (·.session == s)).isSome then found := found + 1
      wins := wins.push { root, opened := e, leaf := leafOf e.sig }
    | "end" =>
      if let some k := wins.findIdx? (·.opened.session == s) then
        if let some r := (getD e.body "record_leaf").getNat?.toOption then
          if records.any (·.1 == r) then found := found + 1
          records := records.push (r, i)
        wins := wins.modify k fun w => { w with ended := some e }
        found := found + 1
    | "period-open" => joined := #[]
    | "period-close" => found := found + (joined.filterMap fun ses => wins.find? (·.opened.session == ses)).size
    | _ => pure ()
    leaves := leaves.push (leafOf e.sig)
  return found + (records.filter fun (r, _) => leaves.contains r).size

def main (args : List String) : IO UInt32 := do
  let (s, path, expect, n) ← match args with
    | [s, p, e] => pure (s.toNat?.getD 0, p, e, 3)
    | [s, p, e, n] => pure (s.toNat?.getD 0, p, e, n.toNat?.getD 3)
    | _ => do IO.eprintln "usage: ledger-replay-time S REQUEST EXPECT [REPEAT]"; return 2
  let t0 ← IO.monoNanosNow
  let j ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  let want ← IO.ofExcept (Json.parse (← IO.FS.readFile expect))
  let key ← IO.ofExcept (ChallengerKey.ofJson (getD j "key"))
  let ledger := getD j "ledger"
  let lines := match ledger with | .arr ls => ls | _ => #[]
  let size := lines.size
  row "request" s size (((← IO.monoNanosNow) - t0) / 1000) []
  let (got, us) ← timed n (fun t => if t == 0 then .error "unreachable" else ledgerPiece j)
    fun | .ok a => a.compress.length | .error e => e.length
  let same := match got with | .ok a => a.compress == want.compress | .error _ => false
  row "piece" s size us [("same", toJson same)]
  let (r, us) ← timed n (fun t => if t == 0 then .error "unreachable" else checked ledger key)
    fun | .ok (w, p) => w.size + p.size | .error e => e.length
  let (ok, windows) := match r with | .ok (w, _) => (true, w.size) | .error _ => (false, 0)
  row "checked" s size us [("ok", toJson ok), ("windows", toJson windows)]
  let (es, us) ← timed n (fun t => if t == 0 then #[] else lines.filterMap ledgerEntry?) (·.size)
  row "entries" s size us [("read", toJson es.size)]
  let texts := lines.filterMap fun | .str x => some x | _ => none
  let (parsed, us) ← timed n
    (fun t => if t == 0 then #[] else texts.filterMap fun x => (Flock.JsonParse.parse x).toOption) (·.size)
  row "parse" s size us [("read", toJson parsed.size)]
  let (own, us) ← timed n (fun t => if t == 0 then 0 else
    (texts.zip parsed).foldl (fun acc (x, e) => if (PyJson.canonical e).toOption == some x then acc + 1 else acc) 0) id
  row "canonical" s size us [("read", toJson own)]
  let (whole, us) ← timed n (fun t => if t == 0 then 0 else
    parsed.foldl (fun acc e => if (Flock.canon e).toOption.isSome then acc + 1 else acc) 0) id
  row "canon" s size us [("read", toJson whole)]
  let (records, us) ← timed n
    (fun t => if t == 0 then #[] else parsed.filterMap fun e => (Core.Record.ofJson e).toOption) (·.size)
  row "record" s size us [("read", toJson records.size)]
  let (_, us) ← timed n
    (fun t => es.foldl (fun acc e => acc + (Flock.hexOf (Flock.Sha512.hash e.text.toUTF8)).length) (t % 2)) id
  row "chain" s size us []
  let I := key.pub.extract 0 16
  let (ms, us) ← timed n (fun t => if t == 0 then #[] else es.map fun e => ledgerMessage I e.entry) (·.size)
  row "messages" s size us []
  let (verified, us) ← timed n
    (fun t => if t == 0 then 0 else (es.zip ms).foldl (fun acc (e, m) => if signs key m e.sig then acc + 1 else acc) 0) id
  row "signatures" s size us [("verified", toJson verified)]
  let (found, us) ← timed n (fun t => if t == 0 then 0 else searches es) id
  row "searches" s size us [("found", toJson found)]
  -- the request is read once more here, so that it is not freed inside a timing
  if j.isNull || !same || (ok && verified != size) then
    IO.eprintln s!"the piece answered as its reference: {same}; the replay accepted: {ok}; {verified} of {size} signatures verify"
    return 1
  return 0
