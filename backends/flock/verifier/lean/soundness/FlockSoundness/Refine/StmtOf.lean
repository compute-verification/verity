import FlockSoundness.Refine.Realizes
import FlockLevel3.Folded
import FlockSoundness.Protocol.Refine.StmtOf

/-!
# The statement decoding, part 1: the executable's layout is a block R1CS (PROTOCOL.md §10, §16.1)

The executable never holds `A₀` and `B₀`: its lincheck fold (`Stmt.fold`) reads them off the circuit's layout. `stmtOf st`
is the model statement those matrices make. Entry `(i, c)` of `A₀` or `B₀` counts, mod 2:
* the covering range's slot matrix at `(i mod 2^sl, c mod 2^sl)`, when row `i` is in column `c`'s slot. The covering
  range is the last in the layout that covers `c`, since the fold overwrites (`lastCover`). A mask slot is the identity
  on both sides;
* for a lookup slot, on the B side, the product rows its table's words send to `c` (`lkCnt`);
* the pairs `(i, c)` of Δ_A or Δ_B.

`ofCircuit_fold`: the fold `Setup.ofCircuit` hands the verifier realizes `stmtOf st` (`FoldRealizes`, R3), given the
layout facts `StmtWF`: `6 ≤ k_log ≤ m`, the pin inside the block, and every range's slots between 2^6 and 2^k_log. The
regions are a parameter here.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128

/-! ### The matrices -/

/-! ### Counting in characteristic 2 -/

/-! ### One slot -/

/-! ### The statement -/

end FlockSoundness.Refine
