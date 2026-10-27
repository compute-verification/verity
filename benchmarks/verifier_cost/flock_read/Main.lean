import Flock

/-!
`flock-read CIRCUIT PUBLIC [STATEMENT]`: the phases of `flock-verify verify --zk`'s read of a circuit file, each timed,
one `PHASE {json}` line per phase on stdout: the file's read, its SHA-512 (the pin), its UTF-8 decode and line split,
`Zk.maskLogOf` (a second decode and split, which `verifyCmd` runs for the statement's tags), each section's `Net.parse`
split into its tokenizer and `Net.ofRows` (and that into `Sparse.ofRows`, `checkOrder`, `inputsOk`), the class digest,
`HmOut.parse` end to end, `Stmt.setupHiddenOf` on the parsed circuit, and one lincheck fold (`Stmt.folded`, the `--zk`
lincheck's) at seeded coins. `flock_read.sh` builds it against the checkout's verifier and runs it; it is a measurement,
not a verifier, and runs nothing the verifier doesn't.
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

/-- A seeded xorshift64 stream of field elements. -/
def coins (seed : UInt64) (n : Nat) : Array F128 := Id.run do
  let mut x := seed
  let mut out := #[]
  for _ in [0:n] do
    x := x ^^^ (x <<< 13); x := x ^^^ (x >>> 7); x := x ^^^ (x <<< 17)
    let lo := x
    x := x ^^^ (x <<< 13); x := x ^^^ (x >>> 7); x := x ^^^ (x <<< 17)
    out := out.push ⟨lo, x⟩
  return out

def nnz (n : Net) : Nat := (n.a.data.size + n.b.data.size) / 4

/-- `Net.parse`'s tokenizer alone: each useful row's A and B columns. -/
def rowsOf (lines : Array String) (useful : Nat) : Except String (Array (Array Nat) × Array (Array Nat)) := do
  let mut ra := #[]
  let mut rb := #[]
  for i in [0:useful] do
    let v ← (tokens (lines.getD (1 + i) "")).mapM fun t => match t.toNat? with
      | some n => pure n
      | none => throw s!"row {i} is not numbers"
    let na := v.getD 0 0
    let nb := v.getD (1 + na) 0
    ra := ra.push (v.extract 1 (1 + na))
    rb := rb.push (v.extract (2 + na) (2 + na + nb))
  return (ra, rb)

def sectionPhases (lines : Array String) (at_ n : Nat) (name : String) : IO Unit := do
  let block := lines.extract (at_ + 1) (at_ + 1 + n)
  let t0 ← IO.monoNanosNow
  match Net.parse block with
  | .error e => report s!"net.parse {name}" t0 [("error", Json.str e)]
  | .ok net =>
    report s!"net.parse {name}" t0 [("useful", toJson net.useful), ("nnz", toJson (nnz net)), ("unit_log", toJson net.unitLog)]
    if net.useful < 100000 then return
    -- the big section's parts, each on its own
    let t1 ← IO.monoNanosNow
    match rowsOf block net.useful with
    | .error e => report "unit.tokens" t1 [("error", Json.str e)]
    | .ok (ra, rb) =>
      report "unit.tokens" t1 [("rows", toJson ra.size)]
      let t2 ← IO.monoNanosNow
      match Sparse.ofRows ra (2 ^ net.unitLog), Sparse.ofRows rb (2 ^ net.unitLog) with
      | .ok a, .ok b => report "unit.sparse" t2 [("bytes", toJson (a.data.size + b.data.size))]
      | _, _ => report "unit.sparse" t2 [("error", Json.str "refused")]
      let t3 ← IO.monoNanosNow
      match Net.checkOrder "probe" (net.inWords * WORD) net.constPos ra rb with
      | .ok () => report "unit.checkOrder" t3
      | .error e => report "unit.checkOrder" t3 [("error", Json.str e)]
      let t4 ← IO.monoNanosNow
      if Net.inputsOk net.inGroups ra rb then report "unit.inputsOk" t4 else report "unit.inputsOk" t4 [("error", Json.str "false")]
      let t5 ← IO.monoNanosNow
      match Net.ofRows name net.useful net.constPos net.inGroups net.outGroups ra rb with
      | .ok n2 => report "unit.ofRows" t5 [("nnz", toJson (nnz n2))]
      | .error e => report "unit.ofRows" t5 [("error", Json.str e)]

def main (args : List String) : IO UInt32 := do
  let circuit :: pubPath :: rest := args | IO.eprintln "usage: flock-read CIRCUIT PUBLIC [STATEMENT]"; return 2
  let some tags0 := Tags.find? (rest.headD Tags.circuit.name) | IO.eprintln "unknown statement"; return 2
  let t0 ← IO.monoNanosNow
  let bytes ← IO.FS.readBinFile circuit
  report "read" t0 [("bytes", toJson bytes.size)]
  let pubB ← IO.FS.readBinFile pubPath
  let t0 ← IO.monoNanosNow
  let pin := Sha512.hash bytes
  if pin.size == 64 then report "sha512" t0 [("pin", Json.str ((hexOf pin).take 16).toString)] else report "sha512" t0
  let t0 ← IO.monoNanosNow
  let ml := Zk.maskLogOf bytes
  if ml > 0 then report "maskLogOf" t0 [("mask_log", toJson ml)] else report "maskLogOf" t0
  let count := 1
  let coinsV := Coins.tree (Zk.coinSpec (tags0.tableNames count))
  let tags := Zk.tags (tags0.withCoins coinsV) ml count
  -- the decode and line split, then each section's parse, as `HmRow.parse` reads them
  let t0 ← IO.monoNanosNow
  match String.fromUTF8? bytes with
  | none => report "utf8+split" t0 [("error", Json.str "not UTF-8")]; return 1
  | some text =>
    let lines0 := (text.splitOn "\n").toArray
    let lines := if lines0.back? == some "" then lines0.pop else lines0
    report "utf8+split" t0 [("lines", toJson lines.size)]
    let mut at_ := 2
    while at_ < lines.size do
      let h := tokens lines[at_]!
      let some n := (h.getD 2 "").toNat? | break
      sectionPhases lines at_ n (h.getD 1 "")
      at_ := at_ + 1 + n
    let t0 ← IO.monoNanosNow
    let head := (lines.getD 0 "").trimAsciiEnd.toString
    match Json.parse ((lines.getD 1 "").trimAsciiEnd.toString.drop 5).toString with
    | .error e => report "classOf" t0 [("error", Json.str e)]
    | .ok mj => match HmRow.classOf bytes head mj with
      | .ok d => report "classOf" t0 [("class", Json.str ((hexOf d).take 16).toString)]
      | .error e => report "classOf" t0 [("error", Json.str e)]
  let t0 ← IO.monoNanosNow
  match HmOut.parse tags bytes [] with
  | .error e => report "HmOut.parse" t0 [("error", Json.str e)]; return 1
  | .ok c =>
    report "HmOut.parse" t0 [("nets", toJson (c.nets.map fun (nm, n) => Json.mkObj [("name", Json.str nm),
      ("useful", toJson n.useful), ("nnz", toJson (nnz n)), ("unit_log", toJson n.unitLog)])), ("k_log", toJson c.kLog),
      ("vus_per_block", toJson c.g), ("ranges", toJson c.ranges.size)]
    let t0 ← IO.monoNanosNow
    match Stmt.setupHiddenOf (.ok c) tags pubB coinsV with
    | .error e => report "setupHiddenOf" t0 [("error", Json.str e)]; return 1
    | .ok st =>
      report "setupHiddenOf" t0 [("delta", toJson (st.da.size / 2)), ("regions", toJson st.regions.size), ("m", toJson st.m),
        ("structured", toJson st.structured)]
      let kl := st.c.kLog
      let r := coins 0x9e3779b97f4a7c15 (2 + 64 + 2 * (kl - 6))
      let t0 ← IO.monoNanosNow
      match st.folded r[0]! r[1]! (r.extract 2 66) (r.extract 66 (66 + kl - 6)) (r.extract (66 + kl - 6) r.size) with
      | .ok v => report "fold" t0 [("width", toJson v.size), ("v0", Json.str (toString v[0]!.lo))]
      | .error e => report "fold" t0 [("error", Json.str e)]
      return 0
