"""Scratch reads-walker, second form: roots are every constant a listed statement's printed signature names."""
import json, re, sys
lock, scope_mode, imports, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(lock))
SKIP = re.compile(r"^(inst[A-Z]|Mathlib\.|Lean\.|Nat\.|Int\.|Real\.|ENNReal\.|Finset\.|List\.|Set\.|Multiset\.|Fin\.|Rat\.|NNRat\.|String\.|Bool\.|Option\.|Prod\.|Sum\.|Function\.|Fintype\.|SetLike\.|Membership\.|HAdd\.|HMul\.|Unit\.|Except\.|ByteArray\.|Array\.|UInt\d+\.|Char\.|BitVec\.|Std\.)")
rows = []
for k, v in d["guarantees"].items():
    s = " ".join(x.strip() for x in v["signature"])
    toks = []
    for t in re.split(r"[\s()\[\]{},:→↔∀∃λ⟨⟩=≤<≥>+*/^¬∧∨|]+", s):
        t = t.strip(".")
        if not t or not re.search(r"[A-Za-zα-ωΑ-Ω]", t) or t == k or SKIP.match(t) or "." not in t:
            continue
        if '"' in t or "\\" in t:
            continue
        if t not in toks:
            toks.append(t)
    rows.append(k + "\t" + " ".join(toks))
blob = "\\n".join(rows)
TARGETS = ["Sheet", "Headline", "Accel", "H200", "H100", "B200", "Sm120", "Sm90", "Sm100", "Device", "Prices", "W1",
           "DitherInstance", "StreamedWipe", "JointBridge", "Dimension", "TTOut", "Deadline", "Cap", "Rev1", "Wref",
           "NaiveStrassen", "Physical", "CostModel", "Capacity", "TS.Machine", "Witness", "PeelExact", "Fp8Atom", "Fp4",
           "Honest", "Period", "Credit", "Served"]
targets = ", ".join(f'"{t}"' for t in TARGETS)
if scope_mode == "definitions":
    scope = "m.getRoot == `Definitions"
else:
    scope = "!(#[`Init, `Lean, `Std, `Mathlib, `Batteries, `Aesop, `Qq, `ProofWidgets, `ImportGraph, `LeanSearchClient, `Plausible, `Cli].contains m.getRoot)"
imps = "\n".join(f"import {i}" for i in imports.split(","))
open(out, "w").write(f'''{imps}
import Lean
open Lean Elab Command

namespace SheetPlanScratch

def blob : String := "{blob}"

def targets : Array String := #[{targets}]

def usesOf : ConstantInfo → Array Name
  | .thmInfo _ | .quotInfo _ => #[]
  | .axiomInfo v => v.type.getUsedConstants
  | .defnInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray
  | .ctorInfo v => v.type.getUsedConstants
  | .recInfo v => v.type.getUsedConstants

def inScope (m : Name) : Bool := {scope}

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
  for row in blob.splitOn "\\n" do
    let parts := row.splitOn "\\t"
    let g := parts[0]!
    let toks := (parts.getD 1 "").splitOn " "
    let roots : Array Name := toks.toArray.filterMap fun t =>
      let n := t.toName
      if t.isEmpty then none else if env.contains n then some n else none
    let mut seen : NameSet := {{}}
    let mut stack : Array Name := roots
    let mut mods : Array Nat := #[]
    let mut defs : Array Nat := #[]
    repeat
      let some c := stack.back? | break
      stack := stack.pop
      if seen.contains c then continue
      seen := seen.insert c
      let some ci := env.find? c | continue
      let some m := modOf c | continue
      if ci matches .thmInfo _ then continue
      if !inScope m then continue
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
    lines := lines.push s!"SP|G|{{g}}|{{mods.qsort (· < ·)}}|{{defs.qsort (· < ·)}}|{{roots.size}}"
  logInfo ("\\n".intercalate lines.toList)

end SheetPlanScratch
''')
print(len(blob))
