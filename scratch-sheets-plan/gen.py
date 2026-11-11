"""Writes the scratch reads-walker: for every statement the lock lists, the definitions it reads (as Facts.lean's readsWhere walks them)."""
import json, re, sys
lock, scope, imp, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(lock))
stmts = []
for k, v in d["guarantees"].items():
    s = " ".join(x.strip() for x in v["signature"])
    m = re.fullmatch(r"(\S+?)(\.\{[^}]*\})? : (\S+)", s)
    stmts.append((k, m.group(3) if m else ""))
names = ",\n  ".join(f'("{k}", "{s}")' for k, s in stmts)
TARGETS = ["Sheet", "Headline", "Accel", "H200", "H100", "B200", "Sm120", "Sm90", "Sm100", "Device", "Prices", "W1",
           "DitherInstance", "StreamedWipe", "JointBridge", "Dimension", "TTOut", "Deadline", "Cap", "Rev1", "Wref",
           "NaiveStrassen", "Physical", "CostModel", "Capacity", "Site", "TS.Machine", "Witness"]
targets = ", ".join(f'"{t}"' for t in TARGETS)
open(out, "w").write(f'''import {imp}
import Lean
open Lean Elab Command

namespace SheetPlanScratch

def stmts : Array (String × String) := #[
  {names}]

def targets : Array String := #[{targets}]

def usesOf : ConstantInfo → Array Name
  | .thmInfo _ | .quotInfo _ => #[]
  | .axiomInfo v => v.type.getUsedConstants
  | .defnInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray
  | .ctorInfo v => v.type.getUsedConstants
  | .recInfo v => v.type.getUsedConstants

def inScope (m : Name) : Bool := m.getRoot == `{scope}

def litsOf (ci : ConstantInfo) : IO (Array Nat) := do
  let r ← IO.mkRef (#[] : Array Nat)
  let visit (e : Expr) : IO Unit := e.forEach' fun s => do
    if let .lit (.natVal n) := s then
      if n ≥ 100 then r.modify fun a => if a.contains n then a else a.push n
    pure true
  match ci with
  | .defnInfo v => visit v.value
  | .opaqueInfo v => visit v.value
  | _ => pure ()
  let a := (← r.get).qsort (· > ·)
  return a.extract 0 8

run_cmd do
  let env ← getEnv
  let modOf (c : Name) : Option Name := (env.getModuleIdxFor? c).map fun i => env.header.moduleNames[i.toNat]!
  let mut modIdx : Std.HashMap Name Nat := {{}}
  let mut defIdx : Std.HashMap Name Nat := {{}}
  let mut flagged : Std.HashMap Name Bool := {{}}
  let mut lines : Array String := #[]
  for (g, s) in stmts do
    let root := s.toName
    if s.isEmpty || (env.find? root).isNone then
      lines := lines.push s!"SP|X|{{g}}|{{s}}"
      continue
    let mut seen : NameSet := {{}}
    let mut stack : Array Name := #[root]
    let mut mods : Array Nat := #[]
    let mut defs : Array Nat := #[]
    let mut outside : NameSet := {{}}
    repeat
      let some c := stack.back? | break
      stack := stack.pop
      if seen.contains c then continue
      seen := seen.insert c
      let some ci := env.find? c | continue
      let some m := modOf c | continue
      if ci matches .thmInfo _ then continue
      if !inScope m then
        outside := outside.insert m.getRoot
        continue
      let mi ← match modIdx[m]? with
        | some i => pure i
        | none => do
          let i := modIdx.size
          modIdx := modIdx.insert m i
          lines := lines.push s!"SP|M|{{i}}|{{m}}"
          pure i
      unless mods.contains mi do mods := mods.push mi
      let fl ← match flagged[c]? with
        | some b => pure b
        | none => do
          let ms := m.toString
          let lits ← litsOf ci
          let b := targets.any (fun t => (ms.splitOn t).length > 1) || !lits.isEmpty
          flagged := flagged.insert c b
          if b then
            let i := defIdx.size
            defIdx := defIdx.insert c i
            lines := lines.push s!"SP|D|{{i}}|{{mi}}|{{c}}|{{lits}}"
          pure b
      if fl then defs := defs.push defIdx[c]!
      stack := stack ++ usesOf ci
    lines := lines.push s!"SP|G|{{g}}|{{mods.qsort (· < ·)}}|{{defs.qsort (· < ·)}}|{{outside.toList}}"
  logInfo ("\\n".intercalate lines.toList)

end SheetPlanScratch
''')
