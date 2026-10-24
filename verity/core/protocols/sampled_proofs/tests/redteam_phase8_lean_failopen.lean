import Flock.Registered
open Lean (Json)

/- Run (not a Lake target; not replayed by the Lean check — it lives in a Python tests dir, not a Lake package):
     cd verity/core/service/flock && lake build flock-verify
     lake env lean ../../protocols/sampled_proofs/tests/redteam_phase8_lean_failopen.lean

   REDTEAM phase 8, target 1: Lean's fail-open in `Flock.Registered.check` when the verifier holds no registered
   reads (`own = none`, i.e. `--registered` absent). The check decides whether there ARE registered reads to enforce
   from the PROVER's public header (`pub.header.registered`), never from the circuit's ports or the verifier's program.
   So the SAME circuit `c` — which has a port — is accepted when the header omits `registered` and refused when it
   names it. A prover whose circuit reads a registered value need only omit `registered` from the public header (and
   commit the rows afresh) to pass. This is what #3597's host guard (`proofs.reads_registered`) compensates for. -/

def port : Flock.Port := { name := "cs", words := 1, chunks := 1, nb := 1 }
def c : Flock.Circuit := { (default : Flock.Circuit) with ports := #[port] }

def pubNoReg : Flock.Public :=
  { header := Json.mkObj [], n := 1, digests := #[], outputs := #[], sha := ByteArray.empty }

def pubNamesReg : Flock.Public :=
  { header := Json.mkObj [("registered", Json.mkObj [("cs", Json.mkObj [])])],
    n := 1, digests := #[], outputs := #[], sha := ByteArray.empty }

-- own = none (no --registered); derived flag is irrelevant (the check short-circuits before it).
#eval Flock.Registered.check c pubNoReg none false      -- EXPECT: Except.ok ()      (FAIL OPEN)
#eval Flock.Registered.check c pubNoReg none true        -- EXPECT: Except.ok ()      (FAIL OPEN, derived too)
#eval Flock.Registered.check c pubNamesReg none false    -- EXPECT: Except.error ...  (refused: header named it)
