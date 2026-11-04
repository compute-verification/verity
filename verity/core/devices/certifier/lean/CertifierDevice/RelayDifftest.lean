import CertifierDevice.Relay

/-!
# The relay difftest: `Relay.run` against `warden.Warden`, on generated connections

`scripts/relay_vectors.py` writes instances (the relay's parameters as `NetworkCheck.Period.scheduleOf` reads them, its
windows, and each connection in arrival order as `[ready, last]`) and what the checkout's real `warden.Warden`, driven
tick by tick, does with each connection: `"refused"`, `"cut"`, or `[admitted, cells]`.
`CertifierDevice/RelayDifftestMain.lean` reads both files and fails unless `evaluate`, which parses the parameters with
`scheduleOf` and runs `Relay.run` over `windows · window` ticks, gives every decision exactly, compared as `Decision`s.
The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

namespace CertifierDevice.RelayDifftest

open Lean
open CertifierDevice.Relay

/-- A decision as the Python side writes it. -/
def decisionOf (j : Json) : Except String Decision := do
  if let .ok "refused" := j.getStr? then return .refused
  if let .ok "cut" := j.getStr? then return .cut
  let #[adm, cells] ← j.getArr? | throw s!"not a decision: {j.compress}"
  pure (.admitted { adm := ← adm.getNat?, cells := ← cells.getNat? })

/-- The decisions `Relay.run` makes on an instance's connections. -/
def evaluate (j : Json) : Except String (List Decision) := do
  let s ← NetworkCheck.Period.scheduleOf (← j.getObjVal? "schedule")
  let windows ← j.getObjValAs? Nat "windows"
  let conns ← (← j.getObjValAs? (Array Json) "conns").toList.mapM fun c => do
    let #[ready, last] ← c.getArr? | throw s!"not [ready, last]: {c.compress}"
    pure ({ ready := ← ready.getNat?, last := ← last.getNat? } : Conn)
  pure (Relay.run s (windows * s.window) conns)

/-- How the Lean side's decisions on instance line `i` differ from Python's, one line per connection that differs. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst >>= evaluate,
      Json.parse python >>= (·.getObjValAs? (Array Json) "decisions") >>= (·.toList.mapM decisionOf) with
  | .error e, _ => [s!"instance {i}: the Lean side rejects it: {e}"]
  | _, .error e => [s!"instance {i}: Python's decisions don't parse: {e}"]
  | .ok lean, .ok py =>
    if lean.length != py.length then [s!"instance {i}: {lean.length} decisions, Python {py.length}"] else
      (lean.zip py).zipIdx.filterMap fun ((x, y), k) =>
        if decide (x = y) then none else some s!"instance {i} connection {k}: Lean {repr x}, Python {repr y}"

/-- `certifier-relay-difftest INSTANCES PYTHON N`: every instance line of `INSTANCES` evaluated, each compared with the
same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: certifier-relay-difftest INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"relay difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let found := (a.zip b).toList.zipIdx.flatMap fun ((x, y), i) => differences i x y
  for d in found.take 50 do IO.eprintln d
  IO.println s!"relay difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end CertifierDevice.RelayDifftest
