import FlockAudit

/-- `flock-audit-tape --coin-tape FILE <flock-audit run's arguments>`: a test harness. The auditor's program over its
workers with every coin node's bytes the tape's next ones, so a test can reproduce a recorded session's draw. -/
def main (args : List String) : IO UInt32 := do
  let some f := FlockAudit.Cli.argOf args "--coin-tape" | IO.eprintln "flock-audit-tape: --coin-tape FILE"; return 2
  let tape ← IO.FS.readBinFile f
  let pos ← IO.mkRef 0
  FlockAudit.Cli.run args fun n => do
    let p ← pos.get
    if p + n > tape.size then throw (IO.userError s!"coin tape: {n} bytes asked at {p} of {tape.size}")
    pos.set (p + n)
    return tape.extract p (p + n)
