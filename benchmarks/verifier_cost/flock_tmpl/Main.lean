import Flock.TmplSplit
import Flock.NetAt
import Flock.TmplAt
import Flock.ProvedScope
import Flock.KnownRead

/-!
`flock-tmpl layout TMPL [MODE…]`: `Tmpl.layoutM`, option (iv) under each `MODE` (`iv`, `iv/strict`, `iv/hybrid`,
`iv/hybrid/T`; default `iv iv/strict`), its size and the time it takes (O(description)), then the reference column map
`Tmpl.posMap` and its digest (`h ← h · 1000003 + pos` over the columns, mod 2^64), which `tmpl_layouts.py --digest`
computes on its own map. `flock-tmpl fold TMPL [--mode MODE] [--split [--plain]] [--ref]`: `Tmpl.fold` on the layout
(default `iv`) at a fixed point, its work and time against the template's row entries, or with `--split`
`Tmpl.foldSplit`, its work, tables and the work they took (with `--plain`, then `Tmpl.fold`, which it must equal); with
`--ref`, then the reference `Tmpl.foldRef` (O(unrolled)), which it must equal.
`flock-tmpl read TMPL`, `flock-tmpl unroll TMPL` and `flock-tmpl diff TMPL CIRCUIT`: the templated unit's read
(`Flock.Tmpl.parse`, its text's read, line split and form checks, O(description)), the read then `Tmpl.unroll` (the flat
net, through `Net.ofRows`'s matrices and checks: what a verifier that unrolls would hold), and the difftest of its unroll against the circuit file's flat unit:
`Tmpl.unroll`, then each line `Tmpl.flatLines` would print compared byte for byte with the circuit's block of the same
net name, read a line at a time. One `PHASE {json}` line per phase on stdout: milliseconds, RSS and peak RSS in MB.
`flock-tmpl unrollat TMPL [--mode MODE] [--phases]`: `Tmpl.unrollAt` at the layout (default `iv/hybrid/4`), what the
templated statement's read runs, timed whole or a stage at a time (`rows`, `posMap`, `Net.ofRows`, `invOf`, `posOk`,
`placed`, `Sparse.ofRows`); `flock-tmpl flat CIRCUIT NAME`: `Net.parse` on the circuit file's block of net `NAME`, what
the flat statement's read runs in its place. `flock-tmpl checks TMPL`: the description checks (`Tmpl.wellFormed`,
`Tmpl.ordered`, `Tmpl.inputsT`, `Tmpl.apart`, `Tmpl.groupsOk`: together `Tmpl.unroll`'s success, `unroll_ok`) timed one at a
time, then the unrolled input-row checks they stand for (`Net.ofRows`' input rows and `Net.inputsOk` on `Tmpl.rows`), which
must agree. `flock-tmpl checks-at TMPL [--mode MODE]`: the layout's checks (`Tmpl.layApart` over the Definitions,
`Tmpl.identOk`, `Tmpl.outsAt`) and all of `Tmpl.unrollOk`, `Tmpl.unrollAt`'s success, at `layoutM`'s layouts (default
`iv/hybrid/4`), timed one at a time.
`flock-tmpl stmt CIRCUIT`: a templated statement's circuit read as `HmOut.parseTmpl` reads it, a stage at a time (META off
its own line, `HmOut.metaOf`; its template lines, `Tmpl.parse`, `layoutM`, `unrollOk`, `TmplUnit.read`, `HmRow.parse`, and
`HmRow.parse`'s own work timed apart: its SHA-512, UTF-8 check, line split and META parse, the file's sections' `Net.parse`
and `HmNets.check`'s comparison with the verifier's `sha512x3` and `hm96` nets, which `HmRow.parse` built (what is left of
it is their build); then `checkRows`), then what decides
which of the scope check's walks run (`ProvedScope.wiresFeed`), the input columns the description surely reads
(`KnownRead.sureReads`, timed, and whether it dispatched), the unit's rows built (`Tmpl.unitNet`'s, which the verifier
never reads), the columns they read (`RecResiduals.readIns`), which must include every surely read one (`sure_not_read`
0, else exit 1), and the walks (`portReadsB`, `outReads`, `wiresAcyclic`) on the unit timed.
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

def modeOf : String → Option Tmpl.Mode
  | "iv" => some .lenient
  | "iv/strict" => some .strict
  | "iv/hybrid" => some (.hybrid 0)
  | s => if s.startsWith "iv/hybrid/" then ((s.drop 10).toString.toNat?).map .hybrid else none

def entries (rows : Array (Array Nat)) : Nat := rows.foldl (· + ·.size) 0

/-- The lines of the circuit file's block holding net `name`, as `Net.parse` reads them. -/
def flatBlock (circuit name : String) : IO (Option (Array String)) := do
  let h ← IO.FS.Handle.mk circuit .read
  let _ ← h.getLine
  let _ ← h.getLine
  repeat
    let bl ← h.getLine
    if bl.isEmpty then break
    let some cnt := ((tokens bl).getD 2 "").toNat? | return none
    let mut ls := #[]
    for _ in [0:cnt] do ls := ls.push ((← h.getLine).trimAsciiEnd.copy)
    let ht := tokens (ls.getD 0 "")
    if ht[0]? == some "flock-ir-unit/v2" && ht[1]? == some name then return some ls
  return none

/-- `Tmpl.unrollAt`, whole or (`phases`) a stage at a time as it runs them. -/
def unrollAtCmd (t : Tmpl.Tmpl) (mode : Tmpl.Mode) (phases : Bool) : IO UInt32 := do
  let t0 ← IO.monoNanosNow
  let .ok lays := Tmpl.layoutM t mode | report "layout" t0 [("error", Json.str "layout refused")]; return 1
  report "layout" t0 [("useful", toJson (lays.getD t.root default).size)]
  if !phases then
    let t0 ← IO.monoNanosNow
    match Tmpl.unrollAt t lays with
    | .error e => report "unroll_at" t0 [("error", Json.str e)]; return 1
    | .ok n =>
      report "unroll_at" t0 [("useful", toJson n.useful), ("nnz", toJson ((n.a.data.size + n.b.data.size) / 4)),
        ("unit_log", toJson n.unitLog)]
      return 0
  let t0 ← IO.monoNanosNow
  let acc ← match t.rows with
    | .error e => report "rows" t0 [("error", Json.str e)]; return 1
    | .ok acc => report "rows" t0 [("cols", toJson acc.ra.size), ("nnz", toJson (entries acc.ra + entries acc.rb))]; pure acc
  let t0 ← IO.monoNanosNow
  let pos := Tmpl.posMap t lays
  report "pos_map" t0 [("columns", toJson pos.size), ("last", toJson (pos.back?.getD 0))]
  let cols := (t.defs.getD t.root default).cols
  let t0 ← IO.monoNanosNow
  let n ← match Net.ofRows t.name cols t.constPos t.inGroups t.outGroups acc.ra acc.rb with
    | .error e => report "of_rows" t0 [("error", Json.str e)]; return 1
    | .ok n => report "of_rows" t0 [("nnz", toJson ((n.a.data.size + n.b.data.size) / 4))]; pure n
  let size := pos.getD t.constPos 0 + 1
  let t0 ← IO.monoNanosNow
  let inv := Net.invOf pos size
  report "inv_of" t0 [("size", toJson inv.size)]
  let t0 ← IO.monoNanosNow
  let ok := Net.posOk cols (n.inWords * WORD) size pos inv && Net.outsOk pos t.outGroups
  report "pos_ok" t0 [("ok", toJson ok)]
  let t0 ← IO.monoNanosNow
  let pa := Net.placed pos inv acc.ra
  let pb := Net.placed pos inv acc.rb
  report "placed" t0 [("nnz", toJson (entries pa + entries pb))]
  let t0 ← IO.monoNanosNow
  match Sparse.ofRows pa (2 ^ max 7 (log2ceil size)), Sparse.ofRows pb (2 ^ max 7 (log2ceil size)) with
  | .ok a, .ok b => report "sparse" t0 [("nnz", toJson ((a.data.size + b.data.size) / 4))]; return 0
  | _, _ => report "sparse" t0 [("error", Json.str "refused")]; return 1

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
  | "unrollat" :: tp :: opt =>
    let name := match opt.dropWhile (· != "--mode") with | _ :: m :: _ => m | _ => "iv/hybrid/4"
    let some mode := modeOf name | IO.eprintln s!"unknown layout {name}"; return 2
    let some t ← readTmpl tp | return 1
    unrollAtCmd t mode (opt.contains "--phases")
  | ["flat", circuit, name] =>
    let t0 ← IO.monoNanosNow
    let some ls ← flatBlock circuit name | report "flat_read" t0 [("error", Json.str s!"no net {name}")]; return 1
    report "flat_read" t0 [("lines", toJson ls.size)]
    let t0 ← IO.monoNanosNow
    match Net.parse ls with
    | .error e => report "flat_parse" t0 [("error", Json.str e)]; return 1
    | .ok n =>
      report "flat_parse" t0 [("useful", toJson n.useful), ("nnz", toJson ((n.a.data.size + n.b.data.size) / 4))]
      return 0
  | ["checks", tp] =>
    let some t ← readTmpl tp | return 1
    let mut ok := true
    for (name, check) in [("well_formed", Tmpl.Tmpl.wellFormed), ("ordered", Tmpl.Tmpl.ordered),
        ("inputs_t", Tmpl.Tmpl.inputsT), ("apart", Tmpl.Tmpl.apart), ("groups_ok", Tmpl.Tmpl.groupsOk)] do
      let t0 ← IO.monoNanosNow
      let v := check t
      report name t0 [("ok", toJson v)]
      ok := ok && v
    let t0 ← IO.monoNanosNow
    match t.rows with
    | .error e => report "unrolled_inputs" t0 [("error", Json.str e)]; return 1
    | .ok acc =>
      let rowsOk := (List.range t.inCols).all fun i =>
        (acc.ra.getD i #[] == #[i] && acc.rb.getD i #[] == #[i]) || ((acc.ra.getD i #[]).isEmpty && (acc.rb.getD i #[]).isEmpty)
      let v := rowsOk && Net.inputsOk t.inGroups acc.ra acc.rb
      report "unrolled_inputs" t0 [("ok", toJson v), ("in_cols", toJson t.inCols)]
      return if ok && v then 0 else 1
  | "checks-at" :: tp :: opt =>
    let name := match opt.dropWhile (· != "--mode") with | _ :: m :: _ => m | _ => "iv/hybrid/4"
    let some mode := modeOf name | IO.eprintln s!"unknown layout {name}"; return 2
    let some t ← readTmpl tp | return 1
    let t0 ← IO.monoNanosNow
    let .ok lays := Tmpl.layoutM t mode | report "layout" t0 [("error", Json.str "layout refused")]; return 1
    report "layout" t0 [("size", toJson (lays.getD t.root default).size)]
    let mut ok := true
    for (name, check) in [("lay_apart", fun t lays => (List.range t.defs.size).all (Tmpl.layApart t lays)),
        ("ident_ok", Tmpl.identOk), ("outs_at", Tmpl.outsAt), ("unroll_ok", Tmpl.unrollOk)] do
      let t0 ← IO.monoNanosNow
      let v := check t lays
      report name t0 [("ok", toJson v)]
      ok := ok && v
    return if ok then 0 else 1
  | "layout" :: tp :: names =>
    let some t ← readTmpl tp | return 1
    let mut ok := true
    for name in (if names.isEmpty then ["iv", "iv/strict"] else names) do
      let some mode := modeOf name | IO.eprintln s!"unknown layout {name}"; return 2
      let t0 ← IO.monoNanosNow
      match Tmpl.layoutM t mode with
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
    let name := match opt.dropWhile (· != "--mode") with | _ :: m :: _ => m | _ => "iv"
    let some mode := modeOf name | IO.eprintln s!"unknown layout {name}"; return 2
    let t0 ← IO.monoNanosNow
    let .ok lays := Tmpl.layoutM t mode | report "fold" t0 [("error", Json.str "layout refused")]; return 1
    let sl := max 7 (log2ceil (lays.getD t.root default).size)
    let pt := fun (i : Nat) => (⟨(0x9e3779b97f4a7c15 * (i.toUInt64 + 1)) ^^^ 0x5bd1e995,
      0xc2b2ae3d27d4eb4f * (i.toUInt64 + 7)⟩ : F128)
    let lw := (Array.range 64).map fun i => pt (100 + i)
    let rho := (Array.range (sl - 6)).map fun i => pt (200 + i)
    let r := (Array.range (sl - 6)).map fun i => pt (300 + i)
    let entries := t.defs.foldl (fun n d => d.rows.foldl (fun n r => n + r.2.1.size + r.2.2.size) n) 0
    let plain := fun (phase : String) => do
      let t0 ← IO.monoNanosNow
      match Tmpl.fold t lays (pt 7) lw rho r with
      | .error e => report phase t0 [("error", Json.str e)]; return none
      | .ok (p, work) =>
        report phase t0 [("mode", Json.str name), ("work", toJson work), ("entries", toJson entries),
          ("slot_log", toJson sl), ("p0", Json.str (p.getD 0 0).toHex)]
        return some p
    let some p ← (if opt.contains "--split" then do
        let t0 ← IO.monoNanosNow
        match Tmpl.foldSplit t lays (pt 7) lw rho r with
        | .error e => report "fold_split" t0 [("error", Json.str e)]; return none
        | .ok (p, work, tables, tw) =>
          report "fold_split" t0 [("mode", Json.str name), ("work", toJson work), ("tables", toJson tables),
            ("table_work", toJson tw), ("entries", toJson entries), ("slot_log", toJson sl),
            ("p0", Json.str (p.getD 0 0).toHex)]
          if !opt.contains "--plain" then return some p
          let some q ← plain "fold" | return none
          if p != q then report "fold" t0 [("equal", toJson false)]; return none
          return some p
      else plain "fold") | return 1
    if !opt.contains "--ref" then return 0
    let t0 ← IO.monoNanosNow
    match Tmpl.foldRef t lays (pt 7) lw rho r with
    | .error e => report "fold_ref" t0 [("error", Json.str e)]; return 1
    | .ok q =>
      report "fold_ref" t0 [("equal", toJson (p == q))]
      return if p == q then 0 else 1
  | ["stmt", circuit] =>
    let some tags := Tags.templated.head? | return 2
    let t0 ← IO.monoNanosNow
    let bytes ← IO.FS.readBinFile circuit
    report "read" t0 [("bytes", toJson bytes.size)]
    let t0 ← IO.monoNanosNow
    let .ok mj := HmOut.metaOf bytes | report "meta" t0 [("error", Json.str "refused")]; return 1
    report "meta" t0
    let t0 ← IO.monoNanosNow
    let .ok lines := (do (← J.arr mj "template" "META").mapM fun x => J.asStr x "template line" : V (Array String))
      | report "template_lines" t0 [("error", Json.str "refused")]; return 1
    report "template_lines" t0 [("lines", toJson lines.size)]
    let some mode := (J.str mj "layout" "META").toOption.bind Tmpl.modeOf | IO.eprintln "META: no layout"; return 1
    let t0 ← IO.monoNanosNow
    let .ok t := Tmpl.parse lines | report "parse" t0 [("error", Json.str "refused")]; return 1
    report "parse" t0 [("defs", toJson t.defs.size)]
    let t0 ← IO.monoNanosNow
    let .ok lays := Tmpl.layoutM t mode | report "layout" t0 [("error", Json.str "refused")]; return 1
    report "layout" t0
    let t0 ← IO.monoNanosNow
    report "unroll_ok" t0 [("ok", toJson (Tmpl.unrollOk t lays))]
    let t0 ← IO.monoNanosNow
    let .ok tu := TmplUnit.read t mode | report "unit_read" t0 [("error", Json.str "refused")]; return 1
    report "unit_read" t0 [("useful", toJson tu.net.useful), ("unit_log", toJson tu.net.unitLog)]
    let t0 ← IO.monoNanosNow
    let .ok c := HmRow.parse tags bytes [] (pre := #[("unit", tu.net)]) | report "hmrow_parse" t0 [("error", Json.str "refused")]; return 1
    report "hmrow_parse" t0
    let t0 ← IO.monoNanosNow
    report "hmrow_sha" t0 [("digest", toJson (Sha512.hash bytes).size)]
    let t0 ← IO.monoNanosNow
    let some text := String.fromUTF8? bytes | report "hmrow_utf8" t0 [("error", Json.str "refused")]; return 1
    report "hmrow_utf8" t0 [("bytes", toJson text.utf8ByteSize)]
    let t0 ← IO.monoNanosNow
    let lines := (text.splitOn "\n").toArray
    report "hmrow_lines" t0 [("lines", toJson lines.size)]
    let t0 ← IO.monoNanosNow
    let mj := Json.parse ((lines.getD 1 "").trimAsciiEnd.toString.drop 5).toString
    report "hmrow_meta" t0 [("ok", toJson (mj matches .ok _))]
    let t0 ← IO.monoNanosNow
    let mut at_ := 2
    let mut parsed := 0
    while at_ < lines.size do
      let n := ((tokens lines[at_]!).getD 2 "").toNat?.getD 0
      if Net.parse (lines.extract (at_ + 1) (at_ + 1 + n)) matches .ok _ then parsed := parsed + 1
      at_ := at_ + 1 + n
    report "hmrow_sections" t0 [("nets", toJson parsed)]
    let t0 ← IO.monoNanosNow
    report "hmnets_check" t0 [("ok", toJson (HmNets.check c matches .ok _))]
    let t0 ← IO.monoNanosNow
    let .ok () := HmOut.checkRows c | report "check_rows" t0 [("error", Json.str "refused")]; return 1
    report "check_rows" t0
    let c := { c with tmpl := some tu }
    report "scope" t0 [("in_groups", toJson c.unit.inGroups.size), ("out_is_unit", toJson (c.outNet == c.unitNet)),
      ("wires", toJson c.wires.size), ("wires_feed", toJson (ProvedScope.wiresFeed c)), ("nets", toJson c.nets.size),
      ("unit_log", toJson c.unit.unitLog), ("ranges", toJson c.ranges.size)]
    let t0 ← IO.monoNanosNow
    let sure := tu.t.readsSure
    report "sure_reads" t0 [("columns", toJson sure.size), ("read", toJson (sure.filter id).size),
      ("dispatched", toJson (KnownRead.sureReads c).isSome)]
    let t0 ← IO.monoNanosNow
    report "rows" t0 [("nnz", toJson (c.unit.a.data.size / 4 + c.unit.b.data.size / 4))]
    let t0 ← IO.monoNanosNow
    let rd := RecResiduals.readIns c.unit
    let mask := ProvedScope.portMask c.unit.inGroups (c.unit.inWords * WORD)
    let cols := List.range (max rd.size sure.size)
    let extra := cols.filter fun y => sure.getD y false && !rd.getD y false
    let missed := cols.filter fun y => mask.getD y false && rd.getD y false && !sure.getD y false
    report "read_ins" t0 [("columns", toJson rd.size), ("read", toJson (rd.filter id).size),
      ("port_bits_read", toJson (cols.filter fun y => mask.getD y false && rd.getD y false).length),
      ("sure_not_read", toJson extra.length), ("read_not_sure", toJson missed.length),
      ("first_missed", toJson (missed.take 8))]
    if !extra.isEmpty then return 1
    let t0 ← IO.monoNanosNow
    report "port_reads" t0 [("ok", toJson (ProvedScope.portReadsB c.unit))]
    let t0 ← IO.monoNanosNow
    report "out_reads" t0 [("ok", toJson (ProvedScope.outReads c.unit).2)]
    let t0 ← IO.monoNanosNow
    report "wires_acyclic" t0 [("ok", toJson (ProvedScope.wiresAcyclic c))]
    return 0
  | _ =>
    IO.eprintln ("usage: flock-tmpl read TMPL | flock-tmpl unroll TMPL | flock-tmpl diff TMPL CIRCUIT | " ++
      "flock-tmpl layout TMPL [MODE…] | flock-tmpl fold TMPL [--mode MODE] [--split [--plain]] [--ref] | " ++
      "flock-tmpl unrollat TMPL [--mode MODE] [--phases] | flock-tmpl flat CIRCUIT NAME | flock-tmpl checks TMPL | " ++
      "flock-tmpl checks-at TMPL [--mode MODE] | flock-tmpl stmt CIRCUIT")
    return 2
