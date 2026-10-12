import Proofs.Flock.C3.Fork.Sound
import Proofs.Flock.C3.Compose.Bridge
import Proofs.Flock.C3.Compose.Discharge
import Proofs.Flock.C3.Compose.Decoder
import Proofs.Flock.C3.Compose.Custody
import Proofs.Flock.Soundness.Refine.HoloRepConverse
import Proofs.Flock.C3.Sparse.Read
import Proofs.Flock.C3.Sparse.Src
import Proofs.Flock.C3.Sparse.SrcTie
import Proofs.Flock.C3.VStarList.Statements
import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.C3.VStarList.Range
import Proofs.Flock.C3.RegLevels.Top
import Proofs.Flock.C3.Opening.Read
import Proofs.Flock.C3.RegLevels.Read
import Proofs.Flock.C3.PubRows.Shape
import Proofs.Flock.C3.Opening.Rows
import Proofs.Flock.C3.HoloValues.Read
import Proofs.Flock.C3.Opening.Session
import Proofs.Flock.C3.Opening.SrcOpening
import Proofs.Flock.C3.Opening.SrcLayer

/-!
# Candidate 3's recursive guarantee: its open pieces

`Flock.Guarantees.RecursiveAuditC3` (`Proofs.Flock.Recursive.AuditC3`) is proved from piece (a); pieces (b)–(h)
discharge what it takes besides: VBridge at candidate 3's table (`_hbr`) and V*'s statements over the verifier's list
(`_hob`). Each piece has its own directory and namespace, and `README.md` beside this file is the gap list.

* (a) `C3/Fork/`, `FlockC3.Fork`: candidate 3's table is sound in fork form (`holoForkSound_holoAt`), from the runs
  (`holo_runs_sound`), which the oracle count (`holoTable_sound`, a1o, with a1s's `sparse_sound` and a1r's
  `regOpen_sound`) and the coupling (`holoTable_value_le_oracle`, a1l) bound.
* (b) `C3/Compose/`, `FlockC3.Compose`: the family (`C3Family`, `C3Run`), the tape (`holoSteps`,
  `holoTableCL_plays`), VBridge from the stages' obligations (`vbridge_c3`), the honest family (`C3Family.honest`,
  n0), `_hbr` at it from n1 and the stages' read facts (`hbr_honest`), and the fixed decoder at which n6 states the
  read facts (`C3/Compose/Decoder.lean`).
* (c) `C3/HoloValues/`, `FlockC3.HoloValues`: each rep plays `holoRepC` (`HReps`; at the honest family, n2).
* (d) `C3/Sparse/`, `FlockC3.Sparse`: each rep's sparse phase at session level (`HSparse`; n3), and its read fact on
  the source's runs' own steps from their sparse phases' plays (`SrcSparseOK`, `sparseReadAt_of_ok`), which a source
  tied to V*'s accepted RecSparse units gives (`SrcSparseTie`, `srcSparseOK_of_tie`).
* (e) `C3/Opening/`, `FlockC3.Opening`: R's opening on the two claims (`HOpening`; n4), at the honest family from
  its read fact (`OpeningRead`, `hOpening_of_read`), and that fact from the `rec-opening` unit's residuals
  (`residuals_eq_runO`, `openingRead_of_rows`), and from V*'s sessions and the verifier's coins but the claim tie and
  the tape prefix (`openingRead_of_session`).
* (f) `C3/RegLevels/`, `FlockC3.RegLevels`: R's levels against the registered cap (`HRegLevels`; n5) and R's
  registered top from the verifier's hold of the inner transcript (`stackedTop_of_innerCommits`, t0).
* (g) `C3/PubRows/`, `FlockC3.PubRows`: the public rows per value prefix (`HPubRows`) and the tape (`HTape`), at the
  honest family from the verifier's check (n1), and the commitments there (`hCommits_honest`).
* (h) `C3/VStarList/`, `FlockC3.VStarList`: V*'s statements over the verifier's list (`vstarParsesScoped_of_list`), R's
  shape from the verifier's checks (`stacked_of_openingShape`, `claims_of_stackedShape`), and R's `ℓ` and the slots'
  units from check 5 (`ellOk_of_sparseRange`, `units_of_sparseRange`).
-/
