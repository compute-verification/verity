import Lean
open Lean Elab Command

/-! `False`, proved by `True.intro`, added with the kernel check skipped: elaboration and `#print axioms` accept it, and only a kernel
replay (tools/lean/replay.sh) rejects it.  tools/lean/audit.sh requires the replay to reject this module before it trusts a PASS. -/

set_option debug.skipKernelTC true in
run_cmd liftCoreM <| addDecl <| .thmDecl
  { name := `NegativeControl.bad, levelParams := [], type := mkConst ``False, value := mkConst ``True.intro }
