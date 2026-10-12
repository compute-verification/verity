import Difftest
import PousCheck.Run

/-! `pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json SESSION_ID`: `PousCheck.Run.check` on a recorded run,
`TRUSTED.json` the verifier's own list of memory-challenger keys. With the one argument `serve`, `PousCheck.Run.checkAt`
behind `Difftest.serve`, which `tests/test_run.py` drives against `run.check_at`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["serve"] =>
    Difftest.serve (Difftest.table [("run", PousCheck.Run.runFn)])
    return 0
  | [trusted, setup, digests, session, sessionId] => PousCheck.Run.run trusted setup digests session sessionId
  | _ =>
    IO.eprintln "usage: pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json SESSION_ID | pous-run serve"
    return 2
