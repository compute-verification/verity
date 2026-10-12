import Grader.Registry

/-!
# The grader

`pous-grader FILE TARGET IMPORTS`. `grade.sh` has compiled the submission in a sandbox and put its `.olean`, alone,
first on `LEAN_PATH`; every other module resolves to the trusted tree. FILE is the submission's path (for the report),
TARGET a pinned statement, IMPORTS the submission's imports, comma-separated. It prints one JSON line and exits 0 iff
PASS. With `POUS_GRADE_MEMORY_MB` set, as `grade.sh` sets it, that is Lean's resident-memory limit (`lean -M`), which
the kernel checks during the replay.

1. The trusted environment is the submission's imports without the submission. TARGET must be in `pinnedTargets`
   and be a definition of type `Prop` in namespace `Pous.Guarantees` there.
2. Every constant the submission adds is replayed through the kernel on top of the trusted environment, so a
   declaration that skipped kernel checking (the kernel-skipping debug option, environment tampering) is rejected here.
3. In the replayed environment, `solution` must be a universe-monomorphic theorem whose type is the constant TARGET
   itself, as `theorem solution : Pous.Guarantees.NAME` elaborates. The comparison is syntactic: deciding definitional
   equality against a near-miss statement can keep the kernel unfolding without bound.
4. Every axiom `solution` depends on must be in `allowedAxioms`. The axioms are collected from the replayed
   environment with no environment extensions loaded, so nothing the submission's own compilation recorded is
   trusted.
-/

open Lean

/-- Lean's resident-memory limit in bytes, as `lean -M` sets it. -/
@[extern "lean_internal_set_max_memory"] opaque setMaxMemory (max : USize) : BaseIO Unit

def solutionName : Name := `solution
def pinnedPrefix : Name := `Pous.Guarantees

def report (file target : String) (pass : Bool) (reason : String) (axioms : Array Name := #[]) : IO UInt32 := do
  IO.println <| Json.compress <| Json.mkObj
    [("file", file), ("target", target), ("verdict", if pass then "PASS" else "FAIL"), ("reason", reason),
     ("axioms", Json.arr (axioms.map fun a => Json.str a.toString))]
  return if pass then 0 else 1

def main (args : List String) : IO UInt32 := do
  let [file, target, importsArg] := args
    | IO.eprintln "usage: pous-grader FILE TARGET IMPORTS"; return 2
  if let some mb := (← IO.getEnv "POUS_GRADE_MEMORY_MB").bind String.toNat? then
    setMaxMemory (mb * 1024 * 1024).toUSize
  let imports := (importsArg.splitOn ",").filter (· ≠ "") |>.map fun m => ({ module := m.toName } : Import)
  initSearchPath (← findSysroot)
  let targetName := target.toName
  unless pinnedTargets.contains targetName do
    return ← report file target false s!"{target} is not a pinned statement"
  let base ← importModules imports.toArray {}
  let env ← importModules #[{ module := `PousSubmission }] {}
  -- 1. the target is a pinned statement of the trusted environment
  let some tinfo := base.find? targetName
    | report file target false s!"{target} is not a constant of the trusted environment"
  unless pinnedPrefix.isPrefixOf targetName && tinfo.isDefinition && tinfo.type == .sort .zero &&
      tinfo.levelParams.isEmpty do
    return ← report file target false s!"{target} is not a pinned statement"
  -- 2. kernel replay of everything the submission adds; no constant may make the kernel run compiled code
  let newConsts := env.constants.fold (init := ({} : Std.HashMap Name ConstantInfo)) fun m n ci =>
    if base.contains n then m else m.insert n ci
  let evaluated := [``Lean.reduceBool, ``Lean.reduceNat, ``Lean.ofReduceBool, ``Lean.ofReduceNat]
  for (n, ci) in newConsts.toList do
    if (ci.type.getUsedConstants ++ (ci.value?.map (·.getUsedConstants)).getD #[]).any evaluated.contains then
      return ← report file target false s!"{n} asks the kernel to evaluate compiled code"
  let kenv ← try base.toKernelEnv.replay newConsts
    catch e => return ← report file target false s!"the kernel rejects the submission: {e}"
  let checked := Environment.ofKernelEnv kenv
  -- 3. `solution` proves exactly the target
  let some sinfo := kenv.find? solutionName
    | report file target false "no checked theorem named solution"
  unless sinfo matches .thmInfo _ do
    return ← report file target false "solution is not a theorem"
  unless sinfo.levelParams.isEmpty do
    return ← report file target false "solution must not be universe polymorphic"
  unless sinfo.type == mkConst targetName do
    return ← report file target false s!"solution does not have the pinned type {target}"
  -- 4. axiom audit
  let axioms ← (collectAxioms solutionName : CoreM (Array Name)).toIO'
    { fileName := "<pous-grader>", fileMap := default } { env := checked }
  let unregistered := axioms.filter (!allowedAxioms.contains ·)
  if unregistered.isEmpty then
    report file target true s!"proves {target}; all axioms allowed" axioms
  else
    report file target false s!"unregistered axioms: {unregistered.toList}" axioms
