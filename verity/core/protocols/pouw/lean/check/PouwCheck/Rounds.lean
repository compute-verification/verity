import PouwCommit.Live
import PouwCheck.Pass

/-!
# A live window's round coins (`core_check.py`, "a live window's round coins")

`roundsCheck` is `core_check.rounds`: under the challenger's live coin commitment, round `r`'s coin opened once, after the
`r`-th round root the verifier recorded, and every recorded round's coin opened (`PouwCommit.roundsReason`). A refusal is
`coin` (the coin rule's, or a malformed commitment or opening) or `rounds` (the roots or the openings not lists of their
kind), with Python's words.
-/

namespace PouwCheck

open Lean (Json)

def LIVE_FORM : String := "a live coin commitment is {epoch, families, root, scheme}: a list of [label, count] and a hex root"
def LIVE_OPENING_FORM : String :=
  "a live coin opening is {after, coin, epoch, index, label, path, salt}: integers, a string, hex, and SHA-512 digests"

/-- `live_coins_of`. -/
def liveCoinsOf (j : Json) : Except String PouwCommit.LiveCommitment := do
  let families := at? j "families"
  let some root := hex? (at? j "root") | throw LIVE_FORM
  let .arr fs := families | throw LIVE_FORM
  unless fs.all (fun | .arr _ => true | _ => false) do throw LIVE_FORM
  let (scheme, epoch, layout) ← PouwCommit.layoutOf (at? j "scheme") (at? j "epoch") families
  if root.size != 64 then throw "a coin commitment's root is 64 bytes"
  pure ⟨scheme, epoch, layout, root⟩

/-- `live_opening_of`. -/
def liveOpeningOf (j : Json) : Except String PouwCommit.LiveOpening := do
  let some epoch := PouwWindow.intOf? (at? j "epoch") | throw LIVE_OPENING_FORM
  let some index := PouwWindow.intOf? (at? j "index") | throw LIVE_OPENING_FORM
  let .str label := at? j "label" | throw LIVE_OPENING_FORM
  let some after := hex? (at? j "after") | throw LIVE_OPENING_FORM
  let some coin := hex? (at? j "coin") | throw LIVE_OPENING_FORM
  let some salt := hex? (at? j "salt") | throw LIVE_OPENING_FORM
  let .arr ps := at? j "path" | throw LIVE_OPENING_FORM
  let some path := ps.toList.mapM fun p => (hex? p).filter (·.size == 64) | throw LIVE_OPENING_FORM
  pure ⟨epoch, label, index, after, coin, salt, path⟩

/-- `core_check.rounds`: none when `openings` are the window's round coins over `roots`, else the refusal. -/
def roundsCheck (coins roots openings : Json) : R Unit := do
  let c ← (liveCoinsOf coins).mapError fun e => ("coin", s!"the verifier's coins are malformed: {e}")
  let some rs := (match roots with
      | .arr rs => rs.toList.mapM fun r => (hex? r).filter (·.size == 64)
      | _ => none) | throw ("rounds", "the round roots are not a list of SHA-512 digests")
  let .arr os := openings | throw ("rounds", "the round coins are not a list")
  let os ← os.toList.zipIdx.mapM fun (o, i) => (liveOpeningOf o).mapError fun e => ("coin", s!"round coin {i}: {e}")
  if let some why := (← port (PouwCommit.roundsReason c rs os)) then coin why

end PouwCheck
