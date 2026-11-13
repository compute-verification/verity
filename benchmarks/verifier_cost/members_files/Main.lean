import Spot

/-!
`members-files-time PARAMS COMMITMENT BLOBS N|all... [--ref-max R]`: the per-member form's pass over every member's
files (`OneStageZk.membersFiles`, as the compiled check runs it) against the pass as it is stated, `Member.files` a
member, each reading the blobs and parsing its circuit file itself (`membersFiles_eq_kept` proves them equal), on one
per-member case's parameters, registration record and files, as the executable's `check` takes them; BLOBS is a file
of the blobs' paths, a line each. A row is the case's first N members of its first member's circuit file, in the
parameters' order, so N members of one circuit file; `all` is every member of the case. Each pass is timed once the
case is parsed and the blobs are read and hashed; one `PHASE {json}` line each on stdout. The stated pass runs up to R
members. The case's own line carries its program's size (its bytes and its root's nodes) and the microseconds of one
evaluation of the strata of its partition's query on the program (`Partition.strata`), which the stated pass makes
once a member and the compiled pass once for the members that share the parameters' partition.

The guard, without which no figure counts: both return every member's statement with its files held, and the two
agree on every check's result of every statement; exit 1 otherwise. `members_files.sh` builds it against the
checkout's verifier and runs it; it is a measurement, not a verifier.
-/

open Lean Spot Spot.OneStageZk

abbrev Files := Flock.V (List (String × StepAt))

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

/-- The pass as stated: a member's files by `Member.files`, which shares nothing between members. -/
def filesRef (slice : Member → Json) (bs : Core.Blobs) (ms : List Member) : Files :=
  ms.mapM fun m => (m.files (slice m) bs).map fun a => (m.name, a)

def why (r : Flock.V Unit) : String :=
  match r with
  | .ok _ => "ok"
  | .error e => e

/-- What a pass returned, a line a statement: its member and each check's result, or the refusal. -/
def shown (r : Files) : String :=
  match r with
  | .error e => s!"refused: {e}"
  | .ok as => "\n".intercalate (as.map fun x =>
      s!"{x.1}|{why x.2.scope}|{why x.2.ksFiles}|{why x.2.ksForm}|{why x.2.population}|{x.2.g.getD 0}|" ++
        s!"{x.2.programSha.getD "none"}|{x.2.want.compress}")

/-- `f`'s result and its microseconds; `f` reads the clock's reading so the compiler can't hoist it out of the
timing. -/
def timed (f : List Member → Files) (ms : List Member) : IO (Files × Nat) := do
  let t0 ← IO.monoNanosNow
  let ms := if t0 == 0 then [] else ms
  let r := f ms
  let size := match r with
    | .ok as => as.length
    | .error e => e.length
  if size == 1000000007 then IO.eprintln "unreachable"
  let t1 ← IO.monoNanosNow
  return (r, (t1 - t0) / 1000)

/-- One evaluation of the strata of the partition's query `query` on the program `prog`: how many it gave and its
microseconds; `query` reads the clock's reading so the compiler can't hoist the evaluation out of the timing. -/
def strataTimed (prog : Program) (query : Json) : IO (Nat × Nat) := do
  let t0 ← IO.monoNanosNow
  let query := if t0 == 0 then Json.null else query
  let n := match Partition.strata prog query with
    | .ok s => s.size
    | .error e => e.length
  if n == 1000000007 then IO.eprintln "unreachable"
  let t1 ← IO.monoNanosNow
  return (n, (t1 - t0) / 1000)

def usage : String := "usage: members-files-time PARAMS COMMITMENT BLOBS N|all... [--ref-max R]"

def main (args : List String) : IO UInt32 := do
  let params :: commitment :: blobs :: rest := args
    | do IO.eprintln usage; return 2
  let (sizes, refMax) := match rest.span (· != "--ref-max") with
    | (sizes, [_, r]) => (sizes, r.toNat?)
    | (sizes, []) => (sizes, none)
    | _ => ([], none)
  if sizes.isEmpty || sizes.any (fun s => s != "all" && s.toNat?.isNone) then
    IO.eprintln usage
    return 2
  let p ← IO.ofExcept (Core.canonicalJson (← IO.FS.readBinFile params) >>= paramsEitherOfJson)
  let c ← IO.ofExcept (OneStage.commitmentOf (← IO.FS.readBinFile commitment))
  let some ms := p.members
    | do IO.eprintln "the parameters name no members: the case is not of the per-member form"; return 2
  let some m := ms[0]?
    | do IO.eprintln "the parameters' members are none"; return 2
  let roots ← IO.ofExcept (rootsOf ms c.record)
  let prog ← IO.ofExcept (Program.parse p.own.program)
  let rootNodes := match prog.prog.defs[prog.prog.root]? with
    | some d => d.body.nodes.size
    | none => 0
  let (strata, strataUs) ← strataTimed prog (getD p.own.partition "query")
  let paths := (← IO.FS.lines blobs).filter (· != "")
  let files ← paths.mapM fun (f : String) => IO.FS.readBinFile f
  let bs := Core.Blobs.ofArray files
  let slice := fun (x : Member) => x.slice c.record roots
  let same := ms.toList.filter (·.own.circuit == m.own.circuit)
  let circuit := match bs.get m.own.circuit "circuit" with
    | .ok b => b.size
    | .error _ => 0
  let head := Json.mkObj [("fn", Json.str "case"), ("members", toJson ms.size), ("circuit_members", toJson same.length),
    ("blobs", toJson files.size), ("blob_bytes", toJson (files.foldl (· + ·.size) 0)),
    ("circuit_bytes", toJson circuit), ("program_bytes", toJson p.own.program.size), ("root_nodes", toJson rootNodes),
    ("strata", toJson strata), ("strata_us", toJson strataUs)]
  IO.println s!"PHASE {head.compress}"
  let mut bad := false
  for size in sizes do
    if size.toNat?.any (· > same.length) then
      IO.eprintln s!"{size} members asked of a circuit file the case holds {same.length} members of"
      return 2
    let handed := match size.toNat? with
      | some n => same.take n
      | none => ms.toList
    let n := handed.length
    let mut got : Array String := #[]
    for (name, f) in [("membersFiles", membersFiles slice bs), ("Member.files", filesRef slice bs)] do
      if name == "Member.files" && refMax.any (n > ·) then continue
      let (r, us) ← timed f handed
      let held := match r with
        | .ok as => as.length == n
        | .error _ => false
      let text := shown r
      got := got.push text
      bad := bad || !held
      let hwm ← statusKb "VmHWM:"
      let row := Json.mkObj [("fn", Json.str name), ("members", toJson n),
        ("circuit_files", toJson (handed.map (·.own.circuit)).eraseDups.length), ("us", toJson us),
        ("hwm_mb", toJson (hwm / 1024)), ("held", Json.bool held),
        ("result_sha512", Json.str (Flock.hexOf (Flock.Sha512.hash text.toUTF8)))]
      IO.println s!"PHASE {row.compress}"
      (← IO.getStdout).flush
      if !held then IO.eprintln s!"{name} at {n} members: {(text.take 600).toString}"
    if got.any (· != got[0]!) then
      IO.eprintln s!"membersFiles and Member.files disagree at {n} members"
      bad := true
  return if bad then 1 else 0
