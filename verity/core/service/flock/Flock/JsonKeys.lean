/-
A key named twice in one JSON object. `JsonParse.parse` keeps one value per key, so a verifier input naming a key twice
(a `--registered` file's port, or a port's key) would read as naming it once; `FlockVerify.registeredInput` refuses
such a file by `dupKey`, on the text `JsonParse.parse` accepted.
-/
import Lean.Data.Json
import Flock.JsonParse

namespace Flock.JsonKeys

/-- The innermost object's keys, after a key string at bytes `[s, e)` of `b` (its quotes included), with the key added:
`none` when that object names it already. The key is compared as `JsonParse.parse` decodes it, escapes and all. -/
def addKey (keys : Array String) (b : ByteArray) (s e : Nat) : Option (Array String) :=
  let raw := String.fromUTF8? (b.extract s e) |>.getD ""
  let k := match JsonParse.parse raw with
    | .ok (.str k) => k
    | _ => raw
  if keys.contains k then none else some (keys.push k)

/-- The first key (as written, quotes included) that one object of the JSON text `t` names twice. Every key of every
object is read, a string being a key right after its object's `{` or a `,` in it; `t` is a text `JsonParse.parse` accepts. -/
def dupKey (t : String) : Option String := Id.run do
  let b := t.toUTF8
  -- one frame per open object (its keys so far) or array (`none`)
  let mut stack : Array (Option (Array String)) := #[]
  let mut key := false
  let mut str : Option Nat := none
  let mut esc := false
  for i in [0:b.size] do
    let c := b.get! i
    match str with
    | some s =>
      if esc then esc := false
      else if c == 92 then esc := true
      else if c == 34 then
        str := none
        if key then
          key := false
          match stack.back? with
          | some (some keys) =>
            match addKey keys b s (i + 1) with
            | some keys' => stack := stack.set! (stack.size - 1) (some keys')
            | none => return some ((String.fromUTF8? (b.extract s (i + 1))).getD "")
          | _ => pure ()
    | none =>
      if c == 34 then
        str := some i
      else if c == 123 then
        stack := stack.push (some #[])
        key := true
      else if c == 91 then
        stack := stack.push none
        key := false
      else if c == 125 || c == 93 then
        stack := stack.pop
        key := false
      else if c == 44 then
        key := match stack.back? with
          | some (some _) => true
          | _ => false
  return none

end Flock.JsonKeys
