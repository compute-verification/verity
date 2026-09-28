import Difftest
import PousCheck.TimedAudit

/-! `pous-timed-audit READY.json LEAVES.bin AUDIT.json`: `PousCheck.TimedAudit.verdict` on a strict audit's record, the
verdict of record that `memory_challenger.VERDICT = "lean"` reads. With the one argument `serve`, the same verdict behind
`Difftest.serve`, which `timed_audit.py` drives against `memory_challenger.verdict`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | ["serve"] =>
    Difftest.serve (Difftest.table [("verdict", PousCheck.TimedAudit.verdictFn)])
    return 0
  | [ready, leaves, audit] => PousCheck.TimedAudit.run ready leaves audit
  | _ =>
    IO.eprintln "usage: pous-timed-audit READY.json LEAVES.bin AUDIT.json | pous-timed-audit serve"
    return 2
