/-
What `verify` and `statement` check of V*'s statements at setup, beside their own steps: the inner transcript as the
verifier holds it (`commits`, from the files `--rec-inner-transcript` and `--rec-inner-registration` and the inner
circuit's META), candidate 3's check 5 inputs (`c3`), candidate 3's checks before the session's own setup
(`checkInner`), and the V* checks both run on every statement (`checks`). Both call sites call these, so a statement
about them is one about what either accepts. `verify`'s setup also refuses a V* session that is not one table holding
its whole population (`oneFull`). `verify` also refuses, last in each verdict, a recursive session without
the verifier's inner shape, a V* session whose draw is not every unit of its population (`fullDraw`), a session over
an inner session whose record holds no draw (`innerDrawn`), and a session at a holo or sparse inner shape or a
recursive one without `--zk`, neither of which `RecursiveKS` covers (`denseOnly`, `zkOnly`).
-/
import Flock.RecCoins
import Flock.RecRecord
import Flock.Witness

namespace Flock
namespace RecSetup

open Lean

/-- A round of the record read in its own format, counted as `RecVStar.schedOf` counts one: its coins, its tops' node
counts, its copies of level 0's top, and whether it holds a commit string. -/
def schedOfRound (x : RecRecord.Round) : RecVStar.SchedRound :=
  (x.coins.length, (x.tops.map List.length).toArray, x.level0, !x.commits.isEmpty)

/-- Stream `s`'s salted tops in the record read in its own format, in the record's order, each with its round. -/
def repTops (r : RecRecord.Record) (s : List UInt8) : List (Nat × List (List UInt8)) :=
  (r.rounds.filter (·.stream == s)).flatMap fun x => x.tops.map (x.k, ·)

/-- Stream `s`'s commit strings in the record read in its own format, in the record's order: each round that holds any,
with its round. -/
def repCommits (r : RecRecord.Record) (s : List UInt8) : List (Nat × List (List UInt8)) :=
  (r.rounds.filter (·.stream == s)).filterMap fun x => if x.commits.isEmpty then none else some (x.k, x.commits)

/-- The refusal of a coin server's record that is not in its own format. -/
def notOwnFormat : String :=
  "rec inner transcript: the coin server's record does not read in its own format (RecRecord.readFull: " ++
    "rec_live.record_bytes, its rounds first, then its kind, tops, Hello in its keys' order and the link's roots " ++
    "and points, at json.dumps' indent=1)"

/-- Each round of the coin server's record as its JSON reads it (`tr`, its parse), in the record's round order: its
stream, its round number, and its coins hex-parsed (`f128OfHex`, as `Record.decode` parses the inner session record's). -/
def jsonRounds (tr : Json) : V (List (Option (List UInt8) × Option Nat × List F128)) :=
  ((tr.getObjValAs? (Array Json) "rounds").toOption.getD #[]).toList.mapM fun x => do
    let coins ← (RecVStar.coinsOf x).toList.mapM f128OfHex
    pure ((x.getObjValAs? String "stream").toOption.map (·.toUTF8.toList), (x.getObjValAs? Nat "k").toOption, coins)

/-- **The record read in its own format is the record its JSON readers read**: at the verifier's inner shape `i`, the
inner transcript's bytes `tt` read in the coin server's format (`RecRecord.readFull`, the reader the recursion's
transcript model reads back), and they hold what the JSON readers took of them (`tr`, its parse): Hello's one table, the
table's `root_B` and the link points (`RecVStar.innerLink`), the rounds the schedule's in its order, each with its
stream, round, coin count and counts (`RecVStar.checkSchedule`'s), each rep's salted tops and `c_k` with their
rounds (`innerLink`'s tops and `cs`, `RecVStar.innerCommits`), and every round, with its stream and round, its coins
(`jsonRounds`: the values `innerCommits` holds the inner session record's coins to). The JSON readers run first, so a
record they refuse is refused as before. Bytes that repeat a key read two ways (`Json.parse` keeps the last, `readFull`
the first), and the coins are refused unless both read the same. -/
def checkReads (i : RecVStar.Inner) (tt : String) (tr : Json) (record : String) (cs : RecVStar.InnerCommits) :
    V Unit := do
  let lk ← RecVStar.innerLink tr record
  let some r := RecRecord.readFull tt.toUTF8.toList | throw notOwnFormat
  let t := lk.table
  let tb := t.toUTF8.toList
  let at_ := "rec inner transcript: the coin server's record read in its own format holds "
  if r.hello.tables != [tb] then throw s!"{at_}Hello's tables other than {t} alone, as its JSON reads them"
  if r.roots != [(tb, lk.root.toList.map (·.toList))] then
    throw s!"{at_}a link.roots other than {t}'s root_B alone, as its JSON reads it"
  if r.points != (lk.points.toList.map (·.toList)).flatten then
    throw s!"{at_}link points other than its JSON's"
  let some want := RecVStar.scheduleOf i t | throw s!"{at_}rounds, and the inner shape has no schedule"
  let flat := want.toList.flatMap fun (s, _, _, rs) => rs.toList.mapIdx fun k w => (s.toUTF8.toList, k, w.1, w)
  if r.rounds.map (fun x => (x.stream, x.k, x.n, schedOfRound x)) != flat then
    throw s!"{at_}rounds other than its schedule's (stream, round, coin count, tops, level 0 and commits)"
  for x in [0, 1] do
    let s := s!"{t}/rep{x}".toUTF8.toList
    if repTops r s != (lk.tops.getD x #[]).toList.map fun (k, top) => (k, top.toList.map (·.toList)) then
      throw s!"{at_}rep {x}'s salted tops other than its JSON's"
    if repCommits r s != (cs.reps.getD x #[]).toList.map fun (k, c) => (k, [c.toList]) then
      throw s!"{at_}rep {x}'s c_k other than its JSON's"
  let jr ← (jsonRounds tr).mapError (s!"{at_}a round whose coins its JSON doesn't read as coins: " ++ ·)
  if r.rounds.map (fun x => (some x.stream, some x.k, x.coins)) != jr then
    throw s!"{at_}rounds other than its JSON's (stream, round and coins)"

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
its schedule (`RecVStar.checkSchedule`) and read in its own format as its JSON readers read it (`checkReads`). At
candidate 3's inner shape (`inner` naming a RecSparse shape) it needs the registration, which binds R before the
coins. -/
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
  let tt ← t
  match Json.parse tt with
  | .ok j =>
    let cs ← RecVStar.innerCommits j record rj bound shape
    if let some i := inner then
      RecVStar.checkSchedule i j
      checkReads i tt j record cs
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
  let D ← Draw.UnitDraw.ofJson (Flock.norm d)
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
source: `--rec-inner-session` with the inner statement's files), the record holds a draw
(`RecCoins.recordDraw`: a `unit_draw` neither absent nor null), which check 5 decodes the record at
(`RecCoins.drawnSpec`); a session over a record without one is refused (`noInnerDraw`). Without the record it passes. -/
def innerDrawn (src : Option RecCoins.Source) : V Unit := do
  let some s := src | return
  if (RecCoins.recordDraw s.record).isNone then throw noInnerDraw

/-- The refusal of an inner shape of the form `form`, one the recursion's proof doesn't cover. -/
def notDense (form : String) : String :=
  s!"refused: the verifier's inner shape (--rec-inner) is {form}: RecursiveKS covers the dense form only (no holo, " ++
    "no sparse phase), and the verifier refuses every form its proof doesn't cover"

/-- **The recursion's inner shape is dense**: `RecursiveKS` is proved at the dense inner form only (`Form.Dense`: not
holo, no sparse phase), so under a holo inner shape (`rec-inner --hide`) or a sparse one (candidate 3's RecSparse phase)
the session is refused, until a proof covers that form. Without an inner shape it passes. -/
def denseOnly : Option RecVStar.Inner → V Unit
  | some i =>
    if i.holo then throw (notDense "holo (\"holo\": true)")
    else if i.sparse.isSome then throw (notDense "sparse (\"sparse\", candidate 3's RecSparse phase)")
    else pure ()
  | none => pure ()

/-- The refusal of a recursive session verified without `--zk`. -/
def notZk : String :=
  "refused: a recursive session (a --rec-* flag) without --zk: RecursiveKS covers the masked verifier (verify --zk) " ++
    "only, and the verifier refuses every form its proof doesn't cover"

/-- **A recursive session is masked**: the recursion's proof reads the masked verifier only (`verify --zk`; its model
reads no statement without it), so a session verified with a recursive flag (`rec`) but without `--zk` is refused. -/
def zkOnly (zk rec : Bool) : V Unit := if rec && !zk then throw notZk else pure ()

theorem denseOnly_some {i : RecVStar.Inner} (h : denseOnly (some i) = .ok ()) : i.holo = false ∧ i.sparse = none := by
  cases hh : i.holo <;> cases hs : i.sparse <;>
    simp [denseOnly, hh, hs, throw, throwThe, MonadExceptOf.throw] at h ⊢

end RecSetup
end Flock
