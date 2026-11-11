module

public import Flock.Program

/-!
# The text of the Program of no node

A scratch file, never merged: the text the decoder's probes and its closed answer read.
-/

@[expose] public section

namespace Flock.Program.Closed

/-- A descriptor whose root has no node. -/
def programEmpty : String :=
  "{\"annotations\":{},\"callees\":[],\"definitions\":{\"Root_v1\":{\"bindings\":{},\"body\":{\"nodes\"" ++
  ":[],\"params\":[],\"ret\":{\"concat\":[]},\"ret_type\":[\"Tuple\",[]]},\"form\":\"composite\",\"id\"" ++
  ":\"Root_v1\"}},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-ir/descriptor/v1" ++
  "\",\"types\":[]}"

end Flock.Program.Closed
