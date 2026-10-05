/- The `flock-audit` executable: `FlockAudit.Cli.cmd`. -/
import FlockAudit

def main (args : List String) : IO UInt32 := FlockAudit.Cli.cmd args
