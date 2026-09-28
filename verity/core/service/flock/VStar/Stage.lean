import VStar.Parts

/-!
# V*'s statements over an inner session (`verity_flock.rec_vstage.stage`)

For an inner session of shape `sh` (its table's `m` and `k_log`, its claims), V* is one statement per Ligerito level of
`fast100(m)`, that level's openings of both reps (`rec_outer.levels`: `RecOpen{LANES, H}` over `2 q` instances, `LANES`
the level's row of `2^lanes` elements and `H = d − cap` the siblings below its cap), then one per part of the rep's algebra
(`rec_residuals.parts(spec(sh))`: `InnerRepCheck{S}` over the two reps). Each is written under the directory `stage` writes
it to: `L<level>` and `alg/p<j>`.
-/

namespace VStar

/-- One of V*'s statements. -/
inductive Stmt where
  | recOpen (level LANES H n : Nat)
  | inner (j : Nat) (name : String) (part : Part)
  deriving Inhabited

def recOpenSub (LANES H : Nat) : Sub :=
  { id := s!"rec-open/l{LANES}-h{H}", template := "rec-open", inputs := ["row", "aux", "acc", "coef", "dirs"],
    outputs := ["out", "acc_out"] }

def innerSub (S : RepS) (name : String) : Sub :=
  { id := name, template := "rec-residuals", inputs := S.ports.map (·.1) ++ ["v"], outputs := ["out"] }

def Stmt.name : Stmt → String
  | .recOpen _ L H _ => s!"rec-open/l{L}-h{H}"
  | .inner _ name _ => name

def Stmt.instances : Stmt → Nat
  | .recOpen _ _ _ n => n
  | .inner .. => 2

def Stmt.dir : Stmt → String
  | .recOpen level .. => s!"L{level}"
  | .inner j .. => s!"alg/p{j}"

def Stmt.sub : Stmt → Sub
  | .recOpen _ L H _ => recOpenSub L H
  | .inner _ name p => innerSub p.S name

def Stmt.unit : Stmt → Except String UnitC
  | .recOpen _ L H _ => recOpenUnit L H
  | .inner _ name p => innerUnit p.S name

/-- V*'s statements over an inner session of shape `sh`, in `stage`'s order. -/
def statements (sh : Algebra.Shape) : Except String (Array Stmt) := do
  let s ← sh.sched
  let mut out : Array Stmt := #[]
  for l in [0:s.levels.size] do
    let lv := s.levels[l]!
    out := out.push (.recOpen l (2 ^ lv.lanes) (lv.d - lv.cap) (2 * lv.q))
  let ps ← parts (← Algebra.spec sh)
  for j in [0:ps.size] do
    out := out.push (.inner j s!"rec-residuals/m{sh.m}-k{sh.kLog}-c{sh.claims}-p{j}" ps[j]!)
  pure out

end VStar
