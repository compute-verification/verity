import Flock.TmplFold

/-!
`flock-tmpl layout TMPL`: `Tmpl.layout`, option (iv) lenient and strict, its size and the time it takes (O(description)),
then the reference column map `Tmpl.posMap` and its digest (`h ← h · 1000003 + pos` over the columns, mod 2^64), which
`tmpl_layouts.py --digest` computes on its own map. `flock-tmpl fold TMPL [--ref]`: `Tmpl.fold` on the lenient layout at a
fixed point, its work and time against the template's row entries; with `--ref`, then the reference `Tmpl.foldRef`
(O(unrolled)), which it must equal.
`flock-tmpl read TMPL`, `flock-tmpl unroll TMPL` and `flock-tmpl diff TMPL CIRCUIT`: the templated unit's read
(`Flock.Tmpl.parse`, its text's read, line split and form checks, O(description)), the read then `Tmpl.unroll` (the flat
net, through `Net.ofRows`'s matrices and checks: what a verifier that unrolls would hold), and the difftest of its unroll against the circuit file's flat unit:
`Tmpl.unroll`, then each line `Tmpl.flatLines` would print compared byte for byte with the circuit's block of the same
net name, read a line at a time. One `PHASE {json}` line per phase on stdout: milliseconds, RSS and peak RSS in MB.
`flock_tmpl.sh` builds it against the checkout's verifier and runs it; it is a measurement and a difftest, not a verifier.
-/

open Flock Lean

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

def report (phase : String) (t0 : Nat) (extra : List (String × Json) := []) : IO Unit := do
  let t1 ← IO.monoNanosNow
  let rss ← statusKb "VmRSS:"
  let hwm ← statusKb "VmHWM:"
  let j := Json.mkObj ([("phase", Json.str phase), ("ms", toJson ((t1 - t0) / 1000000)),
    ("rss_mb", toJson (rss / 1024)), ("hwm_mb", toJson (hwm / 1024))] ++ extra)
  IO.println s!"PHASE {j.compress}"
  (← IO.getStdout).flush

def readTmpl (path : String) : IO (Option Tmpl.Tmpl) := do
  let t0 ← IO.monoNanosNow
  let text ← IO.FS.readFile path
  report "read" t0 [("bytes", toJson text.utf8ByteSize)]
  let t0 ← IO.monoNanosNow
  let lines0 := (text.splitOn "\n").toArray
  let lines := if lines0.back? == some "" then lines0.pop else lines0
  report "split" t0 [("lines", toJson lines.size)]
  let t0 ← IO.monoNanosNow
  match Tmpl.parse lines with
  | .error e => report "parse" t0 [("error", Json.str e)]; return none
  | .ok t =>
    report "parse" t0 [("defs", toJson t.defs.size), ("rows", toJson (t.defs.foldl (· + ·.rows.size) 0)),
      ("calls", toJson (t.defs.foldl (· + ·.calls.size) 0)),
      ("segs", toJson (t.defs.foldl (fun n d => d.calls.foldl (· + ·.segs.size) n) 0)),
      ("cols", toJson (t.defs.getD t.root default).cols)]
    return some t

def clip (s : String) : Json := Json.str (s.take 300).toString

/-- The circuit's block holding the net `t` names, compared line by line with the unroll's lines. -/
def diff (t : Tmpl.Tmpl) (n : Net) (circuit : String) : IO Bool := do
  let t0 ← IO.monoNanosNow
  let h ← IO.FS.Handle.mk circuit .read
  let _ ← h.getLine
  let _ ← h.getLine
  let mut blocks := 0
  let mut equal := true
  let mut first : Option (Nat × String × String) := none
  repeat
    let bl ← h.getLine
    if bl.isEmpty then break
    let some cnt := ((tokens bl).getD 2 "").toNat? | equal := false; break
    let l0 ← h.getLine
    let ht := tokens l0
    if ht[0]? != some "flock-ir-unit/v2" || ht[1]? != some t.name then
      for _ in [1:cnt] do let _ ← h.getLine
      continue
    blocks := blocks + 1
    let want := 1 + n.useful + (if t.leaves.isSome then 1 else 0)
    if cnt != want then
      equal := false
      first := first <|> some (0, s!"{want} lines", s!"{cnt} lines")
    let same := fun (got want : String) => got == want ++ "\n" || got == want
    if !same l0 (Tmpl.headerLine n) then
      equal := false
      first := first <|> some (0, Tmpl.headerLine n, l0)
    for i in [0:min n.useful (cnt - 1)] do
      let l ← h.getLine
      let w := Tmpl.rowLine n i
      if !same l w then
        equal := false
        if first.isNone then first := some (1 + i, w, l)
    for r in [0:cnt - 1 - min n.useful (cnt - 1)] do
      let l ← h.getLine
      if let some lv := t.leaves then
        if r == 0 && cnt == want && !same l lv then
          equal := false
          first := first <|> some (cnt - 1, lv, l)
  let ok := equal && blocks > 0
  report "diff" t0 ([("equal", toJson ok), ("blocks", toJson blocks)] ++
    match first with
    | some (i, w, g) => [("first_difference", toJson i), ("unrolled", clip w), ("flat", clip g)]
    | none => [])
  return ok

def main (args : List String) : IO UInt32 := do
  match args with
  | ["read", tp] => return if (← readTmpl tp).isSome then 0 else 1
  | ["unroll", tp] =>
    let some t ← readTmpl tp | return 1
    let t0 ← IO.monoNanosNow
    match Tmpl.unroll t with
    | .error e => report "unroll" t0 [("error", Json.str e)]; return 1
    | .ok n =>
      report "unroll" t0 [("useful", toJson n.useful), ("nnz", toJson ((n.a.data.size + n.b.data.size) / 4)),
        ("unit_log", toJson n.unitLog)]
      return 0
  | ["diff", tp, circuit] =>
    let some t ← readTmpl tp | return 1
    let t0 ← IO.monoNanosNow
    match Tmpl.unroll t with
    | .error e => report "unroll" t0 [("error", Json.str e)]; return 1
    | .ok n =>
      report "unroll" t0 [("useful", toJson n.useful), ("nnz", toJson ((n.a.data.size + n.b.data.size) / 4)),
        ("unit_log", toJson n.unitLog)]
      return if ← diff t n circuit then 0 else 1
  | ["layout", tp] =>
    let some t ← readTmpl tp | return 1
    let mut ok := true
    for strict in [false, true] do
      let name := if strict then "iv/strict" else "iv"
      let t0 ← IO.monoNanosNow
      match Tmpl.layout t strict with
      | .error e => report name t0 [("error", Json.str e)]; ok := false
      | .ok lays =>
        let L := lays.getD t.root default
        report name t0 [("useful", toJson L.size), ("slot_log", toJson (max 7 (log2ceil L.size))),
          ("align", toJson L.align)]
        let t0 ← IO.monoNanosNow
        let m := Tmpl.posMap t lays
        let h := m.foldl (fun h p => h * 1000003 + p.toUInt64) (0 : UInt64)
        report (name ++ "/map") t0 [("digest", Json.str s!"{h.toNat}"), ("columns", toJson m.size)]
    return if ok then 0 else 1
  | "fold" :: tp :: opt =>
    let some t ← readTmpl tp | return 1
    let t0 ← IO.monoNanosNow
    let .ok lays := Tmpl.layout t false | report "fold" t0 [("error", Json.str "layout refused")]; return 1
    let sl := max 7 (log2ceil (lays.getD t.root default).size)
    let pt := fun (i : Nat) => (⟨(0x9e3779b97f4a7c15 * (i.toUInt64 + 1)) ^^^ 0x5bd1e995,
      0xc2b2ae3d27d4eb4f * (i.toUInt64 + 7)⟩ : F128)
    let lw := (Array.range 64).map fun i => pt (100 + i)
    let rho := (Array.range (sl - 6)).map fun i => pt (200 + i)
    let r := (Array.range (sl - 6)).map fun i => pt (300 + i)
    let t0 ← IO.monoNanosNow
    match Tmpl.fold t lays (pt 7) lw rho r with
    | .error e => report "fold" t0 [("error", Json.str e)]; return 1
    | .ok (p, work) =>
      let entries := t.defs.foldl (fun n d => d.rows.foldl (fun n r => n + r.2.1.size + r.2.2.size) n) 0
      report "fold" t0 [("work", toJson work), ("entries", toJson entries), ("slot_log", toJson sl),
        ("p0", Json.str (p.getD 0 0).toHex)]
      if opt != ["--ref"] then return 0
      let t0 ← IO.monoNanosNow
      match Tmpl.foldRef t lays (pt 7) lw rho r with
      | .error e => report "fold_ref" t0 [("error", Json.str e)]; return 1
      | .ok q =>
        report "fold_ref" t0 [("equal", toJson (p == q))]
        return if p == q then 0 else 1
  | _ =>
    IO.eprintln ("usage: flock-tmpl read TMPL | flock-tmpl unroll TMPL | flock-tmpl diff TMPL CIRCUIT | " ++
      "flock-tmpl layout TMPL | flock-tmpl fold TMPL [--ref]")
    return 2
