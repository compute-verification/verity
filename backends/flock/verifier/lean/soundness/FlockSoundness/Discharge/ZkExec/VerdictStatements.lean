import FlockSoundness.Discharge.ZkExec.PlaysStatements
import FlockSoundness.Discharge.ZkExec.Coin
import FlockSoundness.Discharge.Exec.Verdict
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictStatements

/-!
# `RunsZC`'s verdict half: the compiled `--zk` table accepts at the two reps' results (statements)

`RunsZC` (`Coin.lean`) ends with `accZ`: the compiled table's verdict, at the cap `root_B` decodes to and the openings
the proof files decode to, holds at the two reps' results. The results are those of runs played with each rep's
messages and its stream's coins; a run has one result (`plays_det`), and the statements hold at every one.

The compiled table is the executable's under `hm96-sha512/v1`:
* `hs512`: SHA-512 nodes, and `hm96` leaves at the executable's key (`Refine.K512`) for the rows (`leaf512`), the pads
  pairs (`encPair`: `a.toBytes ‖ b.toBytes`) and `τ` (`encTau`: `TAU_TAG ‖ τ.toBytes`);
* `opnZ`: the openings the two proof files decode to: every level's rows and salts (`opensOf`, `saltsOf`) and each
  rep's inner-proof openings (`innerOpensOf`);
* `ZkTabLay`: the table's padding constant, pads and inner code are the executable's: `PAD_C`, the pads layout's
  positions, `innerN` columns, column `j` at the point `j + 1` and query position `w.lo mod N`.

The statements:
* `ZkTableVerdict`: when both reps of table `i` pass `Zk.verifyRep` with one cap, their compiled reps, played with
  their proof files' messages and their streams' coins, have results, and `accZ` accepts at every pair of them.
* `ZkRepAccepts`: one rep's half: when a run of `repProg` consumes every round, every result of the compiled rep has a
  Ligerito verdict that accepts the proof's rows, an inner check that holds on its inner openings, and row paths that
  verify (`OpensOKZ`).
* `ZkInnerAccepts`: the inner proof's half of `ZkRepAccepts`: when `Zk.verifyInner` accepts seven constraints, at rows
  that are those constraints in their order (`LinRel`), every result of `innerGameC` is a check that holds on the
  proof's openings.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

/-! ## The hashes -/

/-! ## The openings -/

/-! ## The table -/

/-! ## Statements -/

end FlockSoundness.Discharge.ZkExec
