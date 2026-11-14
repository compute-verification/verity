/-
**Round 2's reader over the joined record**: a two-stage run's second-stage sessions held to the joined circuit's record
(the verifier's own copy, its digest the Definition's `window`) and to its ranks' registrations, beside
`Spot.SecondStage`, which reads an interior registration and stays as it is until a later switch with its proof.

* **Links** are the record's `boundary_reads` (each a reader template, port, units and field, row `i` the field's leaves
  from `base + Σ d_k · step_k`), read at round 1's boundary port for that field (`linksOf?`).
* **Wires** are the record's `wires`, each a reader span and a writer in closed form (`Spot.Wires`): the first
  `Σ wires_r` are the ranks' own, the rest the join's (`wiresOf?`).
* **Rows** are each declaration's pattern at the units the second sample draws, once per declaration and sampled unit,
  never unrolled (`rowsOf?`).
* **Parts** are rank members (`Spot.Ranks.memberOf`): each sampled unit is proved in the session of the rank member that
  holds it, and that session's sample is the sampled block's units in that member.
* **No interior registration.** This reader drops the interior registration and its four checks (`second-interior-order`,
  `second-interior-names`, `second-return`, `second-inputs`) because every row round 2 opens is in a rank window whose
  registration is in the log before the period's challenge, which `ranks-run` checks (`Spot.Ranks`). Without that
  check the interior registration's checks would still be needed.
* **Its checks**: the ranks' (`Spot.Ranks.check`); per session, `Spot.SecondStage`'s `second-header`,
  `second-population`, `second-roots`, `second-sample`, `second-read`, `second-values`, `second-refs`, `second-tables`
  and `second-verified`, read against the rank member in place of an interior part, with the window-domain binding at
  the rank's program, partition digest and run (`portBinding`); and `ranks-wire-tree`: the writer rows a sampled reader
  member reads through one wire lie in one writer member.
* **The form**: it runs behind `Spot.SecondStage.FORM`, the one switch, and is refused last (`two-stage-form`) while
  that is off.

No body is built yet: each check refuses with its code (`Spot.Ranks.NOT_BUILT`), so nothing passes, and no guarantee
reads this module.
-/
module

public import Spot.Ranks
public import Spot.SecondStage
meta import Spot.Ranks
meta import Spot.SecondStage

@[expose] public section

namespace Spot.SecondWindow

open Lean (Json)

/-- The per-session codes of `Spot.SecondStage` this reader reuses, read against a rank member. -/
def SESSION_CODES : List String :=
  ["second-header", "second-population", "second-roots", "second-sample", "second-read", "second-values",
    "second-refs", "second-tables", "second-verified"]

/-- The links of the joined record `rec`, from its `boundary_reads`. Not built: none. -/
def linksOf? (_ : Json) : Option (Array Spot.SecondStage.Link) := none

/-- The wires of the joined record `rec`, from its `wires`. Not built: none. -/
def wiresOf? (_ : Json) : Option (Array Spot.SecondStage.Wire) := none

/-- The rows of the joined record `rec`'s declarations at the sampled units `sample`. Not built: none. -/
def rowsOf? (_ : Json) (_ : Array Nat) : Option (Array Spot.SecondStage.Rows) := none

/-- The rank members that hold the sampled units `sample` of the joined circuit. Not built: none. -/
def partsOf? (_ : Spot.Ranks.Ranks) (_ : Array Spot.Ranks.Registered) (_ : Array Nat) :
    Option (Array Spot.Ranks.Member) := none

/-- `ranks-wire-tree` of the sampled members `ms` over the joined record `rec`. Not built: refused. -/
def wireTreeFailures (_ : Json) (_ : Array Spot.Ranks.Member) : List (String × Json) :=
  [("ranks-wire-tree", Json.str Spot.Ranks.NOT_BUILT)]

/-- The per-session failures of statements `ss` against members `ms`. Not built: refused. -/
def sessionFailures (_ : Json) (_ : Array Spot.Ranks.Member) (_ : List Json) : List (String × Json) :=
  SESSION_CODES.map fun c => (c, Json.str Spot.Ranks.NOT_BUILT)

/-- Every failure of statements `stmts` over the joined record `rec`, its ranks' records `records`, served records
`served` and registrations `regs` at the serve's root `root`, `logged` whether a registration is in the log before the
period's challenge; when there is none and the form isn't allowed, its refusal. -/
def failures (allow : Bool) (rec : Json) (records : Array Json) (served : Array (String × String))
    (regs : Array Spot.Ranks.Registered) (root : String) (logged : String → Bool) (sample : Array Nat)
    (stmts : List Json) : List (String × Json) :=
  let ranks := Spot.Ranks.failures rec records served regs root logged
  let members := (Spot.Ranks.Ranks.ofJson? (getD rec "ranks")).bind fun rs => partsOf? rs regs sample
  let found := match members with
    | none => ranks ++ [("ranks-members", Json.str Spot.Ranks.NOT_BUILT)]
    | some ms => ranks ++ wireTreeFailures rec ms ++ sessionFailures rec ms stmts
  if found.isEmpty && !allow then [("two-stage-form", Json.str Spot.SecondStage.FORM_WHY)] else found

/-- **Round 2's check over the joined record** (module docstring), the form allowed when `allow` (`Spot.SecondStage.FORM`
in the run): each code once, with the detail it first failed with. -/
def check (allow : Bool) (rec : Json) (records : Array Json) (served : Array (String × String))
    (regs : Array Spot.Ranks.Registered) (root : String) (logged : String → Bool) (sample : Array Nat)
    (stmts : List Json) : Check :=
  (failures allow rec records served regs root logged sample stmts).foldl (fun c x => c.fail x.1 x.2) {}

end Spot.SecondWindow
