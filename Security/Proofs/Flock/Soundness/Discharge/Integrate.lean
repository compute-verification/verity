import Proofs.Flock.Soundness.Discharge.Integrate.Sites
import Proofs.Flock.Soundness.Discharge.Integrate.Bridge
import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
import Proofs.Flock.Soundness.Discharge.Integrate.Headline

/-!
# The headline at the executable, composed

`Prog.flock_headline_live` (e2e-exec's salted, one-table headline at flock-verify's verdict) at the production program,
with every guarantee the merged branches prove discharged and every other one a named hypothesis:
* `Sites`: the session's table at each draw (`DrawSetup`, `tableAt`: setupH's statement at a draw it sets up, an
  unsatisfiable one elsewhere), each drawn unit's site at its rank in the draw (`sitesAt`, distinct units at distinct
  sites by `sitesAt_inj`), the one circuit file's parse (`sitesAt_parse`), and e2e-layout's site fact at every
  satisfiable site (`sitesAt_bcSite`);
* `Bridge`: the statement `Placed.Accepted` reads (`Layout.model`) is the one the executable's refinement plays
  (`Refine.stmtOf`) for every statement `Stmt.setupH` accepts (`model_eq_stmtOf`), so a drawn table is `stmtOf` of its
  statement (`tableAt_stmtOf`), where `DecodesAll` is stated;
* `Accepted`: a draw flock-verify accepts sets up (`drawSetup_of_accepts`) when the record's draw proves the drawn units
  and its statement's link points are `mPts`, so there the table is `stmtOf` of the accepted statement
  (`tableAt_of_accepts`);
* `Headline`: `flock_headline_exec`, the headline at `Placed.keyProg c hU n` with `pp`, `hConst`, `hZero`, `lay`, `rs`
  and `hHm` (`rsAt_computes`, e2e-hm's `keyedRowsOuts_computes`) built, and `hdec` (e2e-exec), the circuit conditions
  `rs` takes and the assumptions of record taken as hypotheses.
-/
