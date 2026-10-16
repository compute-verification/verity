import VStar.Parts

/-!
# V*'s statements over an inner session (`verity_flock.rec_vstage.stage`)

For an inner session of shape `sh` (its table's `m` and `k_log`, its claims), V* is one statement per Ligerito level of
`fast100(m)`, that level's openings of both reps (`rec_outer.levels`: `RecOpen{LANES, H}` over `2 q` instances, `LANES`
the level's row of `2^lanes` elements and `H = d − cap` the siblings below its cap), then one per part of the rep's algebra
(`rec_residuals.parts(spec(sh))`: `InnerRepCheck{S}` over the two reps). Each is written under the directory `stage` writes
it to: `L<level>` and `alg/p<j>`.

Candidate 3's session adds, under `stage --sparse` (`rec_vstage.stage_sparse`, `stage_stacked`): the sparse phase's
`RecSparse_v1{shape}` over both reps (`sparse`), each level of the stacked table R's openings of its one stream
(`RecOpen{LANES, H}` over `q` instances, named `rec-open-stacked/l<LANES>-h<H>` so that no name is both one of R's levels
and one of the table's: `stacked/L<level>`), and each part of R's registered opening (`InnerRepCheck{S}` at R's
opening-only shape, one instance, `rec-opening/m<m>-c<claims>-p<j>`: `stacked/open/p<j>`).
-/

namespace VStar

/-- One of V*'s statements. `sparse`'s `hRound` is the table's round whose message is its port `h`; `opening`'s `ports`
are its part's row ports, `[name, elements]`, and `nv` its v elements. -/
inductive Stmt where
  | recOpen (level LANES H n : Nat)
  | inner (j : Nat) (name : String) (part : Part)
  | recOpenStacked (level LANES H n : Nat)
  | sparse (sh : Flock.RecSparse.Shape) (hRound : Nat)
  | opening (m c j : Nat) (S : Flock.RecOpen.GfS) (ports : List (String × Nat)) (nv : Nat)
  deriving Inhabited

def recOpenSub (LANES H : Nat) : Sub :=
  { id := s!"rec-open/l{LANES}-h{H}", template := "rec-open", inputs := ["row", "aux", "acc", "coef", "dirs"],
    outputs := ["out", "acc_out"] }

def innerSub (S : RepS) (name : String) : Sub :=
  { id := name, template := "rec-residuals", inputs := S.ports.map (·.1) ++ ["v"], outputs := ["out"] }

/-- R's level's unit name (`rec_outer.stacked_name`): never a name of the circuit table's levels, `rec-open/…`. -/
def recOpenStackedName (LANES H : Nat) : String := s!"rec-open-stacked/l{LANES}-h{H}"

/-- `rec_sparse.subcircuit`: its ports in `ports`' order, one output `out`. -/
def sparseSub (sh : Flock.RecSparse.Shape) : Sub :=
  { id := Flock.RecSparse.unitName sh, template := "rec-sparse", inputs := (Flock.RecSparse.ports sh).map (·.1),
    outputs := ["out"] }

/-- `rec_sparse.lowering`'s unit: `rec_sparse.unit_circuit` (`Flock.RecSparse.unitNet'`), one 16-bit port per word. -/
def sparseUnit (sh : Flock.RecSparse.Shape) : Except String UnitC := do
  if !sh.valid then throw s!"{Flock.RecSparse.unitName sh}: rec-sparse's NU is at least 1, ELL at least 6 and T_LOG at most 16"
  let r := (Flock.RecSparse.unitNet' sh).run Flock.HmNets.St.empty
  pure { name := Flock.RecSparse.unitName sh, inPorts := (Flock.RecSparse.ports sh).map (·.2),
         outPorts := [Flock.RecSparse.outWords sh], st := r.2, outs := r.1.toArray }

/-- `rec_residuals.lowering(S, name)`'s unit at a part of R's opening (`Flock.RecResiduals.unitNet'`). -/
def openingUnit (name : String) (S : Flock.RecOpen.GfS) (ports : List (String × Nat)) (nv : Nat) : UnitC :=
  let ns := ports.map (·.2)
  let r := (Flock.RecResiduals.unitNet' S ns nv).run Flock.HmNets.St.empty
  { name, inPorts := Flock.RecResiduals.portWords ns nv, outPorts := [Flock.RecResiduals.RES_OUT_WORDS], st := r.2,
    outs := r.1.toArray }

def Stmt.name : Stmt → String
  | .recOpen _ L H _ => s!"rec-open/l{L}-h{H}"
  | .inner _ name _ => name
  | .recOpenStacked _ L H _ => recOpenStackedName L H
  | .sparse sh _ => Flock.RecSparse.unitName sh
  | .opening m c j .. => Flock.RecOpening.unitName m c j

def Stmt.instances : Stmt → Nat
  | .recOpen _ _ _ n => n
  | .inner .. => 2
  | .recOpenStacked _ _ _ n => n
  | .sparse .. => 2
  | .opening .. => 1

def Stmt.dir : Stmt → String
  | .recOpen level .. => s!"L{level}"
  | .inner j .. => s!"alg/p{j}"
  | .recOpenStacked level .. => s!"stacked/L{level}"
  | .sparse .. => "sparse"
  | .opening _ _ j .. => s!"stacked/open/p{j}"

def Stmt.sub : Stmt → Sub
  | .recOpen _ L H _ => recOpenSub L H
  | .inner _ name p => innerSub p.S name
  | .recOpenStacked _ L H _ => { recOpenSub L H with id := recOpenStackedName L H }
  | .sparse sh _ => sparseSub sh
  | s@(.opening _ _ _ _ ports _) =>
    { id := s.name, template := "rec-residuals", inputs := ports.map (·.1) ++ ["v"], outputs := ["out"] }

def Stmt.unit : Stmt → Except String UnitC
  | .recOpen _ L H _ => recOpenUnit L H
  | .inner _ name p => innerUnit p.S name
  | .recOpenStacked _ L H _ => do pure { ← recOpenUnit L H with name := recOpenStackedName L H }
  | .sparse sh _ => sparseUnit sh
  | s@(.opening _ _ _ S ports nv) => pure (openingUnit s.name S ports nv)

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
