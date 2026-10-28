import NciCheck.Desc
import NciCheck.DescDifftest

/-!
# The liveness cases of the description-sized check

`scripts/desc_liveness_vectors.py` writes Program shapes on which `NciCheck.Desc` once walked a scan's iterations or a
batch's members one by one (PRF-151: a stack overflow, a time and a memory linear in them), each at a small `n` and at an
`n` of `2^22` to `2^52`. `NciCheck/DescLivenessMain.lean` fails unless, on every case, `NciCheck.Desc.check` gives on the
small instance what `NciCheck.check` gives on it expanded gate by gate (`expand`), and on the big one a verdict that is
the small one's (`same`) or has its novel reads per IU (`novel`, where the gates grow with `n`). That it finishes is the
point: the big instances are past anything `NciCheck.Desc` walks one by one (`NciCheck.Desc.WALK`). The Lean audit runs it
(`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

namespace NciCheck.DescLiveness

open Lean (Json toJson)

/-- `j`, an instance of `NciCheck.Desc` (θ and the owners per root node), expanded gate by gate as `NciCheck.check`
reads it: θ its nodes' gates, and each gate's owner its root node's. -/
def expand (j : Json) : Except String Json := do
  let i ← parse j
  let P ← match Flock.Program.decode i.program.toUTF8 with
    | .ok P => pure P
    | .error e => throw e
  let offs := (P.defs[P.root]!).body.offsets
  let gates (k : Nat) : Array Nat := (Array.range (offs[k + 1]! - offs[k]!)).map (offs[k]! + ·)
  let theta := i.theta.flatMap fun t => ((natOf? t).map fun k => (gates k).map toJson).getD #[]
  let owner := (Array.range i.owner.size).flatMap fun k => (gates k).map fun _ => i.owner[k]!
  return j.setObjVal! "theta" (Json.arr theta) |>.setObjVal! "owner" (Json.arr owner)

/-- Per IU its novel reads, from a verdict. -/
def novel (v : Json) : Json :=
  match v.getObjValD "ius" with
  | .arr us => Json.arr (us.map fun u => match u with | .arr #[_, x] => x | _ => Json.null)
  | _ => Json.null

/-- How case `c` fails, one line per difference. -/
def differences (c : Json) : List String :=
  let id := (c.getObjValD "id").compress
  let small := c.getObjValD "small"
  let big := c.getObjValD "big"
  let d := Desc.check small
  let unrolled := match expand small with
    | .ok x => DescDifftest.compare 0 "NciCheck.check" d (check x)
    | .error e => [s!"the expansion fails: {e}"]
  let verdict := if (d.getObjValD "error").isNull then [] else [s!"small: refused, {d.compress}"]
  let b := Desc.check big
  let same := match c.getObjValD "mode" with
    | .str "same" => DescDifftest.compare 0 "small" b d
    | .str "novel" =>
      (if (b.getObjValD "error").isNull then [] else [s!"big: refused, {b.compress}"]) ++
        (if novel b == novel d then [] else [s!"novel reads: big {(novel b).compress}, small {(novel d).compress}"])
    | m => [s!"mode {m.compress}"]
  (unrolled ++ verdict ++ same).map (s!"case {id}: " ++ ·)

/-- `lake env lean --run NciCheck/DescLivenessMain.lean CASES N`: fails unless the file has `N` cases and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [path, n] := args | IO.eprintln "usage: NciCheck/DescLivenessMain.lean CASES N"; return 2
  let lines := ((← IO.FS.readFile path).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  if n.toNat? != some lines.size then
    IO.eprintln s!"desc liveness: {lines.size} cases, not {n}"
    return 1
  let found := lines.toList.flatMap fun l => match Json.parse l with
    | .ok c => differences c
    | .error e => [s!"not JSON: {e}"]
  for d in found do IO.eprintln d
  IO.println s!"desc liveness: {lines.size} cases checked, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NciCheck.DescLiveness
