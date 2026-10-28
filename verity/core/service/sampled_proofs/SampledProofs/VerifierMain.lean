import Difftest
import SampledProofs.Pieces

/-!
# `sampled-proofs-verifier`: the verifier of record, and the challenger's coins

The verifier's functions (`Pieces.verifier`) behind `Difftest.respond`'s line protocol, and one more request,
`{"fn": "coins", "input": n}`: `n` bytes of the operating system's randomness (`IO.getRandomBytes`, A3's
`uniform/io-getrandombytes`), answered `{"output": HEX}`. The proof system's challenger (`service.Stream`, through
`one_stage.verifier.coins`) reads its coins from here, so the Python never reads them from the operating system itself;
the draw of record is then this process's `Select.select` on those bytes.
-/

namespace SampledProofs.VerifierMain

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

/-- The response line to one request line: `coins`, or the verifier's (`Difftest.respond`). -/
def respond (request : String) : IO String :=
  match Json.parse request with
  | .ok req =>
    match req.getObjValAs? String "fn", req.getObjVal? "input" with
    | .ok "coins", .ok input => coins input
    | _, _ => pure (Difftest.respond (Difftest.table Pieces.verifier) request)
  | .error _ => pure (Difftest.respond (Difftest.table Pieces.verifier) request)

end SampledProofs.VerifierMain

/-- Answer requests from stdin until it ends, flushing each response. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    stdout.putStrLn (← SampledProofs.VerifierMain.respond request)
    stdout.flush
