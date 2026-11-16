import NetworkCheck.Registered

/-!
What `NetworkCheck.Registered` answers on a file of vectors, for `test_network_certifier_registered.py`, which holds
`registered.py` to it.

    lake env lean --run RegisteredVectors.lean VECTORS

VECTORS: a JSON list of `{"head", "key", "link", "site"}`. It prints one JSON list, an answer a vector in order:
`{"malformed": why}` where `siteOf` refuses the site, `{"refused": why}` where `registered` refuses the link or the key,
else `{"ok": {"beta", "certifier", "head", "host", "key", "link"}}`.
-/

open Lean (Json)
open NetworkCheck (Registered)

def answer (j : Json) : Flock.V Json := do
  let head ← Flock.J.str j "head" "a vector"
  let link ← Flock.J.str j "link" "a vector"
  let key ← Flock.J.str j "key" "a vector"
  match Registered.siteOf (← Flock.J.get j "site" "a vector") with
  | .error why => pure (Json.mkObj [("malformed", .str why)])
  | .ok S =>
    match Registered.registered S head link key with
    | .error why => pure (Json.mkObj [("refused", .str why)])
    | .ok r => pure (Json.mkObj [("ok", Json.mkObj [("beta", Lean.toJson r.β), ("certifier", .str r.certifier),
        ("head", .str r.head), ("host", .str r.host), ("key", .str r.key), ("link", .str r.link)])])

def answers (text : String) : Flock.V Json := do
  let vectors ← Flock.J.asArr (← Json.parse text) "the vectors"
  pure (.arr (← vectors.mapM answer))

def main (args : List String) : IO UInt32 := do
  let [file] := args
    | IO.eprintln "usage: lean --run RegisteredVectors.lean VECTORS"
      return 2
  match answers (← IO.FS.readFile file) with
  | .error why =>
    IO.eprintln why
    return 2
  | .ok out =>
    IO.println out.compress
    return 0
