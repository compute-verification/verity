module

public import NetworkVerifier.Check

/-!
# The period's decoders

The decoders the period's check (`NetworkCheck.Period`) reads, in its namespace: `keysOf`, `exactly`, `hexAt` and
`certificateOf`. Code: it imports only the certificate check (`NetworkCheck.Check`) and what that imports, and nothing
of the inference-only policy's.
-/

@[expose] public section

namespace NetworkVerifier.Period

open Lean (Json)
open Flock (V)
open NetworkVerifier.Check (Certificate)

def keysOf : Json → List String
  | .obj kvs => kvs.keys
  | _ => []

def exactly (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj _ => if keysOf j == want then pure () else throw s!"{ctx}: the keys must be {want}, not {keysOf j}"
  | _ => throw s!"{ctx}: not an object"

def certificateOf (j : Json) (ctx : String) : V Certificate := do
  exactly j ["digest", "direction", "key_id", "link", "signature", "window"] ctx
  let direction ← Direction.ofValue (← Flock.J.str j "direction" ctx)
  pure { keyId := ← Flock.unhex (← Flock.J.str j "key_id" ctx), link := ← Flock.J.str j "link" ctx, direction,
         window := ← Flock.J.nat j "window" ctx, digest := ← Flock.unhex (← Flock.J.str j "digest" ctx),
         signature := ← Flock.unhex (← Flock.J.str j "signature" ctx) }

def hexAt (j : Json) (k ctx : String) : V ByteArray := do Flock.unhex (← Flock.J.str j k ctx)

end NetworkVerifier.Period
