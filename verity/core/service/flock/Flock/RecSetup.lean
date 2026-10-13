/-
What `verify` and `statement` check of V*'s statements at setup, beside their own steps: the inner transcript as the
verifier holds it (`commits`, from the files `--rec-inner-transcript` and `--rec-inner-registration` and the inner
circuit's META), candidate 3's check 5 inputs (`c3`), candidate 3's checks before the session's own setup
(`checkInner`), and the V* checks both run on every statement (`checks`). Both call sites call these, so a statement
about them is one about what either accepts.
-/
import Flock.RecCoins

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
`metaBound`): `RecVStar.innerCommits` on the files as read. At candidate 3's inner shape (`inner` naming a RecSparse
shape) it needs the registration, which binds R before the coins. -/
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
  | .ok j => RecVStar.innerCommits j record rj bound shape
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

/-- **Candidate 3's checks before the session's own setup**, on the inner statement's files `src`: check 5's sparse
range on the inner statement (`RecCoins.checkAtC3`, which refuses a session without the files), and, where the
verifier's inner shape names a RecSparse shape, that the files' inner circuit's META is the header's it read
(`checkMeta`). -/
def checkInner (inner : Option RecVStar.Inner) (src : Option RecCoins.Source)
    (bound : V (Option String × Option RecVStar.StackedShape)) : V Unit := do
  RecCoins.checkAtC3 inner src
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

end RecSetup
end Flock
