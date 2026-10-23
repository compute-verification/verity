/-
What `verify` and `statement` check of V*'s statements at setup, beside their own steps: the inner transcript as the
verifier holds it (`commits`, from the files `--rec-inner-transcript` and `--rec-inner-registration` and the inner
circuit's META), candidate 3's check 5 inputs (`c3`), candidate 3's checks before the session's own setup
(`checkInner`), and the V* checks both run on every statement (`checks`). Both call sites call these, so a statement
about them is one about what either accepts. `verify`'s setup also refuses a V* session that is not one table holding
its whole population (`oneFull`). `verify` also refuses, last in each verdict, a recursive session without
the verifier's inner shape, a V* session whose draw is not every unit of its population (`fullDraw`), and a session over
an inner session whose record holds no draw (`innerDrawn`).
-/
import Flock.RecCoins
import Flock.Witness

namespace Flock
namespace RecSetup

open Lean

/-- META's `registration` (`inner_registered.REGISTRATION_KEY`): the digest of the circuit registration it binds
(one-stage `registration.digest`), or none when META names none; and META's `stacked` (`RecVStar.stackedShapeOf`), the
shape staging committed R at. -/
def metaBound (m : Json) : V (Option String × Option RecVStar.StackedShape) := do
  let sh ← RecVStar.stackedShapeOf m
  match m.getObjVal? "registration" with
  | .error _ => pure (none, sh)
  | .ok (.str d) => pure (some d, sh)
  | .ok _ => throw "rec inner circuit: META's registration is not a digest"

/-- The inner transcript `t` (`rec/inner-transcript/v1`) held against the inner session's record `record`, the circuit
registration `r` (`verity/registration/v2`) if the verifier holds one, and what the inner circuit's META binds (`bound`,
`metaBound`): `RecVStar.innerCommits` on the files as read, and at the verifier's inner shape `inner` the transcript on
its schedule (`RecVStar.checkSchedule`). At candidate 3's inner shape (`inner` naming a RecSparse shape) it needs the
registration, which binds R before the coins. -/
def commits (inner : Option RecVStar.Inner) (t : V String) (r : V (Option String))
    (bound : V (Option String × Option RecVStar.StackedShape)) (record : String) : V RecVStar.InnerCommits := do
  let rj ← match ← r with
    | none =>
      if (inner.bind (·.sparse)).isSome then
        throw ("rec inner registration: candidate 3's session needs R's circuit registration " ++
          "(--rec-inner-registration), which binds R before the coins")
      pure none
    | some s => match Json.parse s with
      | .ok j => pure (some j)
      | .error e => throw s!"rec inner registration: not JSON: {e}"
  let (bound, shape) ← bound
  match Json.parse (← t) with
  | .ok j =>
    let cs ← RecVStar.innerCommits j record rj bound shape
    if let some i := inner then RecVStar.checkSchedule i j
    pure cs
  | .error e => throw s!"rec inner transcript: not JSON: {e}"

/-- Candidate 3's check 5 inputs: the verifier's RecSparse shape (`rec-inner --sparse`) and R's, its own inner shape's
or else the shape the inner transcript held R's stream to, and the coin server's record of the inner session. -/
def c3 (inner : Option RecVStar.Inner) (commits : Option RecVStar.InnerCommits) : Option RecCoins.C3 := do
  let i ← inner
  let cs ← commits
  let sp ← i.sparse
  pure { sparse := sp, stacked := i.stacked.or cs.stacked, commits := cs }

/-- What `checkMeta` compares of a META's binding: the registration's digest and R's `m` and claims. -/
def boundKey (b : Option String × Option RecVStar.StackedShape) : Option String × Option (Nat × Nat) :=
  (b.1, b.2.map fun s => (s.m, s.claims))

/-- **The inner circuit's META is its header's**: the inner circuit `cf` (`--rec-inner-circuit`, the bytes check 5
sets the inner statement up from) parses as the inner statement's circuit (`HmOut.parse` under
`RecVStar.innerTags`), and what its META binds (`metaBound`) is `bound`, what the verifier read off the file's header
before it held the inner transcript to it (`RecSetup.commits`). The header is read apart from the file, so nothing
else ties the two. -/
def checkMeta (cf : ByteArray) (bound : V (Option String × Option RecVStar.StackedShape)) : V Unit := do
  let c ← (HmOut.parse RecVStar.innerTags cf).mapError (s!"rec inner circuit: " ++ ·)
  let b ← metaBound c.metaJ
  if boundKey (← bound) ≠ boundKey b then
    throw "rec inner circuit: the registration and R's shape its header's META binds are not its circuit's META's"

/-- **The inner statement is set up as its files set it up**: how Flock's plain verifier is told to set the inner
statement up (`form`) is how `RecCoins.Files.stmt` sets it up from its files, `st`: as a
`Tags.circuitPlainLeaves` statement (`--rec-inner-statement`) on the coin server's coins
(`--rec-inner-statement-coins os`), with no lookup table (`--rec-inner-tables`), at a draw (`--rec-inner-draw`) that
proves every one of `st`'s units in order, so its `m` and regions are `st`'s. A session without `form` is refused. -/
def checkForm (form : Option RecCoins.InnerForm) (st : Stmt) : V Unit := do
  let some fm := form
    | throw ("rec-inner: the verifier holds no setup of the inner statement (--rec-inner-statement, " ++
        "--rec-inner-statement-coins, --rec-inner-draw) to hold to its files' setup")
  if fm.statement != Tags.circuitPlainLeaves.name then
    throw s!"rec-inner: the inner statement is a {fm.statement} statement, not {Tags.circuitPlainLeaves.name}, as its files set it up"
  if !fm.os then
    throw "rec-inner: the inner statement's coins are not the coin server's (--rec-inner-statement-coins os), as its files set it up"
  if !fm.tables.isEmpty then
    throw "rec-inner: the inner statement names lookup tables (--rec-inner-tables), and its files set it up with none"
  let some b := fm.draw | throw "rec-inner: the inner statement has no draw (--rec-inner-draw)"
  let some t := String.fromUTF8? b | throw "rec-inner: the inner statement's draw: not UTF-8"
  let d ← (Json.parse t).mapError (s!"rec-inner: the inner statement's draw: not JSON: " ++ ·)
  let D ← Draw.UnitDraw.ofJson d
  if D.proved != Array.range st.n then
    throw s!"rec-inner: the inner statement's draw proves {D.proved.size} units, not every one of its files' {st.n} in order"

/-- **The verifier's inner shape is the inner statement's**: where it holds the inner statement's files, the `m`,
`k_log`, regions and claims of its inner shape (`--rec-inner`) are the ones the inner statement sets up at from them
(`RecCoins.Files.stmt`; claims two and the session's points per region, as `RecVStar.innerOf` writes them), so the V*
list and the rep schedule the shape fixes are the inner session's; and the inner statement is set up as its files set
it up (`checkForm`). -/
def checkShape (inner : Option RecVStar.Inner) (src : Option RecCoins.Source) : V Unit := do
  let (some i, some (.files f)) := (inner, src) | return
  let (spec, st) ← f.stmt "the inner session"
  let claims := 2 + spec.points * st.regions.size
  if i.m != st.m || i.kLog != st.c.kLog || i.regions != st.regions.size || i.claims != claims then
    throw (s!"rec-inner: the verifier's inner shape (m {i.m}, k_log {i.kLog}, regions {i.regions}, claims {i.claims}) " ++
      s!"is not the inner statement's (m {st.m}, k_log {st.c.kLog}, regions {st.regions.size}, claims {claims}), " ++
      "which its files (--rec-inner-circuit, --rec-inner-public, --rec-inner-partition) set up")
  checkForm f.form st

/-- **Candidate 3's checks before the session's own setup**, on the inner statement's files `src`: check 5's sparse
range on the inner statement (`RecCoins.checkAtC3`, which refuses a session without the files), the inner shape the
inner statement's (`checkShape`), and, where the verifier's inner shape names a RecSparse shape, that the files' inner
circuit's META is the header's it read (`checkMeta`). -/
def checkInner (inner : Option RecVStar.Inner) (src : Option RecCoins.Source)
    (bound : V (Option String × Option RecVStar.StackedShape)) : V Unit := do
  RecCoins.checkAtC3 inner src
  checkShape inner src
  if let (some i, some (.files f)) := (inner, src) then
    if i.sparse.isSome then checkMeta f.circuit bound

/-- The V* checks of setup on the statements `sts`: at candidate 3's inner shape, that the verifier holds the inner
transcript; each statement is one of V*'s at the verifier's inner shape (`RecVStar.checkStmt`); and the inner shape
covers the inner transcript the verifier holds (`RecVStar.checkCovers`). -/
def checks (inner : Option RecVStar.Inner) (commits : Option RecVStar.InnerCommits) (tags : Tags)
    (tables : List (String × ByteArray)) (sts : Array Stmt) (circuitB : ByteArray) (partition : Option Json)
    (program : Option ByteArray) (registered : Option (Array Registered.Port)) (part : Option Nat)
    (bindings : Option VStar.Bindings) (rows : Option (Array Registered.Port)) : V Unit := do
  if (inner.bind (·.sparse)).isSome && commits.isNone then
    throw ("rec inner transcript: candidate 3's session needs the inner transcript (--rec-inner-transcript, " ++
      "--rec-inner-session), which its V* list and rows are checked against")
  sts.forM (RecVStar.checkStmt inner tags tables · circuitB partition program registered part bindings rows)
  if let (some i, some cs) := (inner, commits) then RecVStar.checkCovers i cs

/-- **A V* session sets up one table holding its whole population**: under the verifier's inner shape (`inner`), the
setup's statements `sts` must be one table (as an honest V* session's are), whose instances are its population's
(`RecVStar.population`, which `checkStmt` makes V*'s count): a drawn table short of its population is refused here, at
setup, as `fullDraw` refuses its draw in the verdict. The recursion's proof covers no other form: V*'s slots are read at
the one table's addresses, and RecCap reads every instance's. Without an inner shape it passes. -/
def oneFull (inner : Option RecVStar.Inner) (sts : Array Stmt) : V Unit :=
  match inner with
  | none => pure ()
  | some _ =>
    if sts.size != 1 then
      throw (s!"refused: a V* session of {sts.size} tables: a V* session is one table, " ++
        "the only form the recursion's proof covers")
    else sts.forM fun st =>
      if st.n < RecVStar.population st then
        throw (s!"{st.c.unit.name}: refused: a table of {st.n} of its population's {RecVStar.population st} " ++
          "instances: a V* session's table holds every instance, the only form the recursion's proof covers")
      else pure ()

/-- The refusal of a recursive session without the verifier's own inner shape. -/
def noInner : String :=
  "refused: a recursive session (--rec-part, --rec-bindings or --rec-rows) without the verifier's own inner shape " ++
    "(--rec-inner F): V*'s checks and its full-draw refusal run under that shape, and the recursion's proof covers no " ++
    "session without it"

/-- **A V* session draws every unit of its population**: under the verifier's inner shape (`inner`), the session's
record (`record`, whose draw is `draw`) must hold the `subset` draw of every unit of the draw's population with no
closure (`Witness.fullDraw`, the draw `RecordCustody` reads, `fullDraw_ok`). The draw's population is the registered
population's (`HmRow.drawn` refuses another at setup), and `checkStmt` pins the circuit, so the part's numbers (its
rounds, table error, `N₀` and weight) are the ones its circuit and population fix, never a smaller draw's. A recursive
session (`rec`: the verifier holds `--rec-part`, `--rec-bindings` or `--rec-rows`) without an inner shape is refused
(`noInner`), since the V* checks and this refusal run under that shape. Without a draw (which `ProvedScope.drawLaw`
refuses), or outside a recursive session without an inner shape, it passes. -/
def fullDraw (inner : Option RecVStar.Inner) (rec : Bool) (draw : Option Json) (record : String) : V Unit := do
  if rec && inner.isNone then throw noInner
  let (some _, some d) := (inner, draw) | return
  (Witness.fullDraw (← Draw.UnitDraw.ofJson d).population record).mapError
    (s!"refused: {·}: a V* session draws every unit of its population, the only draw the recursion's proof covers")

/-- The refusal of a session over an inner session whose record holds no draw. -/
def noInnerDraw : String :=
  "refused: the inner session's record (--rec-inner-session) holds no draw (unit_draw): the recursion's inner session " ++
    "is drawn (the coin server's full draw, rec_live.full_draw), the only inner session the recursion's proof covers"

/-- **The recursion's inner session is drawn**: where the verifier holds the inner session's record (`src`, check 5's
source: `--rec-inner-session` with the inner statement's files or with `--rec-inner-coins`), the record holds a draw
(`RecCoins.recordDraw`: a `unit_draw` neither absent nor null), which check 5 decodes the record at
(`RecCoins.drawnSpec`); a session over a record without one is refused (`noInnerDraw`). Without the record it passes. -/
def innerDrawn (src : Option RecCoins.Source) : V Unit := do
  let some s := src | return
  if (RecCoins.recordDraw s.record).isNone then throw noInnerDraw

end RecSetup
end Flock
