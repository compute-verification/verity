/-
**A V\* session's registered reads, capped.** Under the verifier's inner shape, `verify`'s setup refuses a V\* session whose
`--registered` file names more than `MAX_PORTS` ports or a port whose tree is deeper than `MAX_DEPTH` levels (`regs`,
before any read is opened), and a table of more than `MAX_INSTANCES` instances (`table`, beside `RecSetup.oneFull`). Each
opening costs `2 + log2ceil leaves` SHA-512 inputs, one per instance and port, so the setup's registered reads are at most
`MAX_INSTANCES · MAX_PORTS · (2 + MAX_DEPTH)`, the bound the recursion's call count (`setupCalls`) is proved at; without
these refusals that count is the prover's to inflate. The caps sit far above every deployed V\* session: a level's table
holds its `2 q` openings (`q < 256`, `fast100`), a later level's RecCap both reps' `2 · 2^cap ≤ 256` tops and RecCap
`2^cap ≤ 128`, a part its two reps; a part registers its row ports (at most `RecResiduals.MAX_REGIONS = 64`) with `v` and
`out`; and the deepest registered tree is `rec-acc`'s, the chain of every opening of both reps and R's, fewer than `2^12`
rows. It also refuses an inner shape at which V\*'s list (`RecVStar.statements`) holds more than `MAX_STATEMENTS`
statements (`statements`), so the recursion's V\* parts, one a statement, are at most that many; a deployed shape's list
holds about ten (`fast100`'s levels, the algebra's parts, then RecCap). And it refuses one at which the list records more
than `MAX_RECORDED` hm96 calls (`recorded`): the calls the V\* parts' witnesses record (two an hm96 slot, the salt's and
the row's, at each registered row a part's layout reads), counted off each statement's description (`recordedAt`), a bound
the recursion's call bound `w` is proved at. `verity_flock.rec_budget` is the stager's mirror.

The inner statement's setup hashes every row of its public file (two SHA-512 inputs a row, then the row tree), and a
shared-row file's tables are as long as its header says, which only the file's length bounds. So where the verifier
holds the inner statement's files, `RecSetup.checkShape` refuses one whose shared-row table holds more than
`MAX_INNER_ROWS` rows (`innerRows`), as many as a statement's instances can be (`m ≤ 35`, slots of at least `2^7`).
-/
module

public import Flock.RecVStar

@[expose] public section

namespace Flock
namespace RecBudget

/-- The most instances a V\* table holds. -/
def MAX_INSTANCES : Nat := 65536
/-- The most ports a V\* session's `--registered` file names. -/
def MAX_PORTS : Nat := 4096
/-- The deepest registered tree a V\* session reads: `log2ceil leaves` levels. -/
def MAX_DEPTH : Nat := 64
/-- The most statements V\*'s list holds at the verifier's inner shape: the recursion's V\* parts. -/
def MAX_STATEMENTS : Nat := 4096
/-- The most rows a shared-row table of the inner statement's public file holds. -/
def MAX_INNER_ROWS : Nat := 268435456

def tail : String := "the recursion's call bound covers no more"

/-- **A V\* session's registration is within the caps**: at most `MAX_PORTS` ports, each of a tree of at most
`MAX_DEPTH` levels. Under no inner shape, or with no `--registered`, nothing is refused. -/
def regs (inner : Option RecVStar.Inner) (registered : Option (Array Registered.Port)) : V Unit :=
  match inner, registered with
  | some _, some own =>
    if MAX_PORTS < own.size then
      throw s!"refused: a V* session's registered reads name {own.size} ports, more than {MAX_PORTS}: {tail}"
    else own.forM fun p =>
      if MAX_DEPTH < log2ceil p.leaves then
        throw (s!"refused: a V* session's registered port {p.name} has {p.leaves} leaves, a tree of " ++
          s!"{log2ceil p.leaves} levels, more than {MAX_DEPTH}: {tail}")
      else pure ()
  | _, _ => pure ()

/-- **A V\* table holds at most `MAX_INSTANCES` instances.** Under no inner shape, nothing is refused. -/
def table (inner : Option RecVStar.Inner) (sts : Array Stmt) : V Unit :=
  match inner with
  | none => pure ()
  | some _ => sts.forM fun st =>
    if MAX_INSTANCES < st.pub.n then
      throw s!"{st.c.unit.name}: refused: a V* session's table of {st.pub.n} instances, more than {MAX_INSTANCES}: {tail}"
    else pure ()

/-- **V\*'s list of `n` statements is within the cap.** -/
def count (n : Nat) : V Unit :=
  if MAX_STATEMENTS < n then
    throw s!"refused: V*'s list at the verifier's inner shape holds {n} statements, more than {MAX_STATEMENTS}: {tail}"
  else pure ()

/-- The most hm96 calls V\*'s list records at the verifier's inner shape (`recordedCalls`): `2^32`. -/
def MAX_RECORDED : Nat := 4294967296

/-- **The hm96 calls V\*'s statement `st` records at most**, at the inner shape `i`, two at each port and instance it
reads: at each row a message-round port of it reads (`RecVStar.ownPorts`); at an algebra part, at `v` and `out`, then
`acc0` and `acc1`, at each of its two instances; at a level of `n` instances, at each one's `out`, and at the table's
(`.recOpen`) at its `coef` and `dirs`; at a RecCap of `N` nodes, at each one's `top` and `out`; and at each `rec-acc` row
a level's units read (`RecVStar.accReads`), at the unit's `acc` and its `acc_out`. -/
def recordedAt (i : RecVStar.Inner) (st : VStar.Stmt) : Nat :=
  2 * ((RecVStar.ownPorts st).map (·.2.2.size)).sum +
    (match st with
      | .inner .. => 16
      | .recOpen _ _ _ n => 6 * n
      | .recOpenStacked _ _ _ n => 2 * n
      | .cap _ N => 4 * N
      | _ => 0) +
    4 * ((((RecVStar.accReads i st).toOption.bind id).map Array.size).getD 0)

/-- The hm96 calls V\*'s list `ss` records at most at the inner shape `i`: each statement's (`recordedAt`). -/
def recordedCalls (i : RecVStar.Inner) (ss : Array VStar.Stmt) : Nat := (ss.toList.map (recordedAt i)).sum

/-- **V\*'s list records at most `MAX_RECORDED` hm96 calls**, `n` of them. -/
def recorded (n : Nat) : V Unit :=
  if MAX_RECORDED < n then
    throw s!"refused: V*'s list at the verifier's inner shape records {n} hm96 calls, more than {MAX_RECORDED}: {tail}"
  else pure ()

/-- **V\*'s list at the verifier's inner shape holds at most `MAX_STATEMENTS` statements and records at most
`MAX_RECORDED` hm96 calls.** Under no inner shape nothing is refused, and a shape at which `RecVStar.statements` itself
refuses is left to the checks that run it. -/
def statements (inner : Option RecVStar.Inner) : V Unit :=
  match inner with
  | none => pure ()
  | some i => match RecVStar.statements i with
    | .ok ss => do
      count ss.size
      recorded (recordedCalls i ss)
    | .error _ => pure ()

/-- **The inner statement's shared-row tables hold at most `MAX_INNER_ROWS` rows each**, at the statement `st` its files
set up. A per-instance file has no table, and passes. -/
def innerRows (st : Stmt) : V Unit :=
  match st.pub.tables with
  | none => pure ()
  | some t =>
    if t.counts.any (MAX_INNER_ROWS < ·) then
      throw s!"refused: the inner statement's public file has a shared-row table of more than {MAX_INNER_ROWS} rows: {tail}"
    else pure ()

end RecBudget
end Flock
