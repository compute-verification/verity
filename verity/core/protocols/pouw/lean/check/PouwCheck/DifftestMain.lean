import PouwCheck.VerifiedDifftest

/-!
# `pouw-check-difftest`: PoUW's checks of record, and the challenger's coins

The port's functions (`PouwCheck.Difftest`'s, then `PouwCheck.CreditDifftest`'s and `PouwCheck.VerifiedDifftest`'s:
`VerifiedDifftest.all`) behind `Difftest.respond`'s line protocol, which `tests/test_difftest.py` drives against
`core_check.py`, `benchmarks/pouw/served_zk/of_record.py` runs as the challenger's checks of record and
`credit._lean_verified` asks for a served window's statements' verdict (`credit-verified`), and one
more request, `{"fn": "coins", "input": n}`: `n` bytes of the operating system's randomness (`IO.getRandomBytes`, A3's
`uniform/io-getrandombytes`), answered `{"output": HEX}`. PoUW's challenger (`pouw.coins.Challenger`,
`pouw.window.live_challenger`, through `pouw.verifier.coins`) reads its entropy from here, so the Python never reads it
from the operating system itself.
-/

namespace PouwCheck.DifftestMain

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

/-- The response line to one request line: `coins`, or the port's (`Difftest.respond`). -/
def respond (request : String) : IO String :=
  match Json.parse request with
  | .ok req =>
    match req.getObjValAs? String "fn", req.getObjVal? "input" with
    | .ok "coins", .ok input => coins input
    | _, _ => pure (Difftest.respond PouwCheck.VerifiedDifftest.all request)
  | .error _ => pure (Difftest.respond PouwCheck.VerifiedDifftest.all request)

end PouwCheck.DifftestMain

/-- Answer requests from stdin until it ends, flushing each response. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    stdout.putStrLn (← PouwCheck.DifftestMain.respond request)
    stdout.flush
