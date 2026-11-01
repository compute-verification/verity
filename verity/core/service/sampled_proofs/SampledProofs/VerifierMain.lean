import Difftest
import SampledProofs.Pieces

/-!
# `sampled-proofs-verifier`: the verifier of record, and the challenger's coins

The verifier's functions behind `Difftest.respond`'s line protocol, at the programs the process has loaded
(`Pieces.At.verifier`): `{"fn": "program", "input": DESCRIPTOR}` loads one (`Loaded.Programs.load`) and answers
its digest, `{"output": HEX}`, or refuses it as a piece would. From then on a piece may name that program by
`{"annotations": A, "loaded": HEX}` (`Loaded.ref?`) and reads the program loaded, the one it would read from that
descriptor (`Loaded.parse_ref`), until the process's input ends; a reference to a program the process hasn't loaded is
refused. One more request,
`{"fn": "coins", "input": n}`: `n` bytes of the operating system's randomness (`IO.getRandomBytes`, A3's
`uniform/io-getrandombytes`), answered `{"output": HEX}`: the entropy a challenger flips its coins from
(`LiveChallenger`, through `one_stage.verifier.coins`), on the challenger's own host, so the Python never reads it from the
operating system itself. The verifier reads no coins from here: its draws read the challenger's coins from its log
(`Run.coinsHold`).
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

/-- The answer to `program` with input `input` at the programs loaded `S`, and the programs loaded after it. -/
def program (S : Loaded.Programs) (input : Json) : String × Loaded.Programs :=
  match S.load input with
  | .ok (S', h) => ("{\"output\":" ++ Flock.canonStr h ++ "}", S')
  | .error e => ("{\"reject\":" ++ Flock.canonStr e ++ "}", S)

/-- The response line to one request line at the programs loaded `S`, and the programs loaded after it: `coins`,
`program`, or the verifier's (`Difftest.respond` on `Pieces.At.verifier S`). -/
def respond (S : Loaded.Programs) (request : String) : IO (String × Loaded.Programs) :=
  match Json.parse request with
  | .ok req =>
    match req.getObjValAs? String "fn", req.getObjVal? "input" with
    | .ok "coins", .ok input => return (← coins input, S)
    | .ok "program", .ok input => return program S input
    | _, _ => pure (Difftest.respond (Difftest.table (Pieces.At.verifier S)) request, S)
  | .error _ => pure (Difftest.respond (Difftest.table (Pieces.At.verifier S)) request, S)

end SampledProofs.VerifierMain

/-- Answer requests from stdin until it ends, flushing each response, holding the programs loaded. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let mut S : SampledProofs.Loaded.Programs := #[]
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    let (out, S') ← SampledProofs.VerifierMain.respond S request
    S := S'
    stdout.putStrLn out
    stdout.flush
