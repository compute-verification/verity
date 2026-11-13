import Spot

/-!
`members-files-time PARAMS COMMITMENT BLOBS N... [--ref-max R]`: the per-member form's pass over every member's
files (`OneStageZk.membersFiles`, as the compiled check runs it) against the pass as it is stated, `Member.files` a
member, each reading the blobs and parsing its circuit file itself (`membersFiles_eq_kept` proves them equal), on one
per-member case's parameters, registration record and files, as the executable's `check` takes them; BLOBS is a file
of the blobs' paths, a line each. The case's first member is handed in N times over, so a row is N members of one
circuit file. Each is timed once the case is parsed and the blobs are read and hashed; one `PHASE {json}` line each
on stdout. The stated pass runs up to R members.

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

def usage : String := "usage: members-files-time PARAMS COMMITMENT BLOBS N... [--ref-max R]"

def main (args : List String) : IO UInt32 := do
  let params :: commitment :: blobs :: rest := args
    | do IO.eprintln usage; return 2
  let (ns, refMax) := match rest.span (· != "--ref-max") with
    | (ns, [_, r]) => (ns.filterMap String.toNat?, r.toNat?)
    | (ns, _) => (ns.filterMap String.toNat?, none)
  if ns.isEmpty || ns.length != (rest.span (· != "--ref-max")).1.length then
    IO.eprintln usage
    return 2
  let p ← IO.ofExcept (Core.canonicalJson (← IO.FS.readBinFile params) >>= paramsEitherOfJson)
  let c ← IO.ofExcept (OneStage.commitmentOf (← IO.FS.readBinFile commitment))
  let some ms := p.members
    | do IO.eprintln "the parameters name no members: the case is not of the per-member form"; return 2
  let some m := ms[0]?
    | do IO.eprintln "the parameters' members are none"; return 2
  let roots ← IO.ofExcept (rootsOf ms c.record)
  let paths := (← IO.FS.lines blobs).filter (· != "")
  let files ← paths.mapM fun (f : String) => IO.FS.readBinFile f
  let bs := Core.Blobs.ofArray files
  let slice := fun (x : Member) => x.slice c.record roots
  let circuit := match bs.get m.own.circuit "circuit" with
    | .ok b => b.size
    | .error _ => 0
  let head := Json.mkObj [("fn", Json.str "case"), ("member", Json.str m.name), ("case_members", toJson ms.size),
    ("blobs", toJson files.size), ("blob_bytes", toJson (files.foldl (· + ·.size) 0)),
    ("circuit_bytes", toJson circuit)]
  IO.println s!"PHASE {head.compress}"
  let mut bad := false
  for n in ns do
    let handed := List.replicate n m
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
      let row := Json.mkObj [("fn", Json.str name), ("members", toJson n), ("us", toJson us),
        ("hwm_mb", toJson (hwm / 1024)), ("held", Json.bool held),
        ("result_sha512", Json.str (Flock.hexOf (Flock.Sha512.hash text.toUTF8)))]
      IO.println s!"PHASE {row.compress}"
      (← IO.getStdout).flush
      if !held then IO.eprintln s!"{name} at {n} members: {(text.take 600).toString}"
    if got.any (· != got[0]!) then
      IO.eprintln s!"membersFiles and Member.files disagree at {n} members"
      bad := true
  return if bad then 1 else 0
