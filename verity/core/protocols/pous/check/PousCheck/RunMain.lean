import Difftest
import PousCheck.RunProtocol

/-! `pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json SESSION_ID`: `PousCheck.Run.check` on a recorded run,
`TRUSTED.json` the verifier's own list of memory-challenger keys. With the one argument `serve`, `PousCheck.Run.checkAt`
(`run`) and the run protocol's certificate at a point (`certify`, `PousCheck.RunProtocol.certifyFn`) behind
`Difftest.serve`, which `tests/test_run.py` drives against `run.check_at` and `tests/test_run_protocol.py` against
`period.certify`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["serve"] =>
    Difftest.serve (Difftest.table [("run", PousCheck.Run.runFn), ("certify", PousCheck.RunProtocol.certifyFn)])
    return 0
  | [trusted, setup, digests, session, sessionId] => PousCheck.Run.run trusted setup digests session sessionId
  | _ =>
    IO.eprintln "usage: pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json SESSION_ID | pous-run serve"
    return 2
