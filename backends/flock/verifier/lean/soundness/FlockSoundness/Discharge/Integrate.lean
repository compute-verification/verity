import FlockSoundness.Discharge.Integrate.Sites
import FlockSoundness.Discharge.Integrate.Headline

/-!
# The headline at the executable, composed

`Prog.flock_headline_live` (e2e-exec's salted, one-table headline at flock-verify's verdict) at the production program,
with every guarantee the merged branches prove discharged and every other one a named hypothesis:
* `Sites`: the session's table at each draw (`DrawSetup`, `tableAt`: setupH's statement at a draw it sets up, an
  unsatisfiable one elsewhere), each drawn unit's site at its rank in the draw (`sitesAt`, distinct units at distinct
  sites by `sitesAt_inj`), the one circuit file's parse (`sitesAt_parse`), and e2e-layout's site fact at every
  satisfiable site (`sitesAt_bcSite`);
* `Headline`: `flock_headline_exec`, the headline at `Placed.keyProg c hU n` with `pp`, `hConst`, `hZero`, `lay` and
  `rs` built, and `hHm` (e2e-hm), `hdec` (e2e-exec) and the assumptions of record taken as hypotheses.
-/
