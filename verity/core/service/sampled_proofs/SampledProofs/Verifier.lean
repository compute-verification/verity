import Difftest
import SampledProofs.Pieces

/-!
# `sampled-proofs-verifier`: the verifier of record, and the challenger's coins

The verifier's functions (`Pieces.verifierAt`) behind `Difftest.respond`'s line protocol, at the programs the process has
loaded, and two more requests:

* `{"fn": "coins", "input": n}`: `n` bytes of the operating system's randomness (`IO.getRandomBytes`, A3's
  `uniform/io-getrandombytes`), answered `{"output": HEX}`. The proof service's challenger (`service.Stream`, through
  `one_stage.verifier.coins`) reads its coins from here, so the Python never reads them from the operating system itself;
  the draw of record is then this process's `Select.select` on those bytes.
* `{"fn": "program", "input": DESCRIPTOR}`: the process loads the program (`Loaded.Store.load`) and answers its digest,
  `{"output": HEX}`, or refuses it as a piece would. From then on a piece's program may be named by
  `{"annotations": A, "loaded": HEX}` (`Loaded.ref?`), and the piece reads the program loaded rather than parse it again:
  the program it would read from that descriptor (`Loaded.parse_ref`). A process loads a program once and holds it until
  its input ends. A reference to a program the process hasn't loaded is read as a descriptor, which it isn't, so the
  piece refuses it.

`serve` answers one request a line until its input ends; `sampled-proofs-verifier` (`SampledProofs.VerifierMain`) runs it.
-/

namespace SampledProofs.Verifier

open Lean (Json)

/-- The most bytes one `coins` request takes. -/
def COINS_MAX : Nat := 2 ^ 24

/-- The answer to `coins` with input `input`: `n` bytes of `IO.getRandomBytes` in lowercase hex, or why `input` is
refused (not a natural number, or past `COINS_MAX`). -/
def coins (input : Json) : IO String := do
  match input.getNat? with
  | .ok n =>
    if n ≤ COINS_MAX then
      return "{\"output\":" ++ Flock.canonStr (Flock.hexOf (← IO.getRandomBytes n.toUSize)) ++ "}"
    return "{\"reject\":" ++ Flock.canonStr s!"coins: {n} bytes in one request, past {COINS_MAX}" ++ "}"
  | .error _ => return "{\"reject\":" ++ Flock.canonStr "coins: the input is a number of bytes" ++ "}"

/-- The answer to `program` with input `input` at the programs loaded `S`, and the programs loaded after it. -/
def program (S : Loaded.Store) (input : Json) : String × Loaded.Store :=
  match S.load input with
  | .ok (S', h) => ("{\"output\":" ++ Flock.canonStr h ++ "}", S')
  | .error e => ("{\"reject\":" ++ Flock.canonStr e ++ "}", S)

/-- The response line to one request line at the programs loaded `S`, and the programs loaded after it: `coins`,
`program`, or the verifier's (`Difftest.respond` on `Pieces.verifierAt S`). -/
def respondAt (S : Loaded.Store) (request : String) : IO (String × Loaded.Store) :=
  match Json.parse request with
  | .ok req =>
    match req.getObjValAs? String "fn", req.getObjVal? "input" with
    | .ok "coins", .ok input => return (← coins input, S)
    | .ok "program", .ok input => return program S input
    | _, _ => pure (Difftest.respond (Difftest.table (Pieces.verifierAt S)) request, S)
  | .error _ => pure (Difftest.respond (Difftest.table (Pieces.verifierAt S)) request, S)

/-- Answer requests from stdin until it ends, flushing each response, holding the programs loaded. -/
def serve : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let mut S : Loaded.Store := #[]
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    let (out, S') ← respondAt S request
    S := S'
    stdout.putStrLn out
    stdout.flush

end SampledProofs.Verifier
