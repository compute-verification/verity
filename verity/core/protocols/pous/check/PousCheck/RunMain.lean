import Difftest
import PousCheck.Run

/-! `pous-run SETUP.json DIGESTS.bin SESSION.json`: `PousCheck.Run.check` on a recorded run. With the one argument
`serve`, `PousCheck.Run.checkAt` behind `Difftest.serve`, which `tests/test_run.py` drives against `run.check_at`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["serve"] =>
    Difftest.serve (Difftest.table [("run", PousCheck.Run.runFn)])
    return 0
  | [setup, digests, session] => PousCheck.Run.run setup digests session
  | _ =>
    IO.eprintln "usage: pous-run SETUP.json DIGESTS.bin SESSION.json | pous-run serve"
    return 2
