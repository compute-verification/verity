import Proofs.Flock.Soundness.Discharge.ZkPubIn.Defs
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Spec
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Tables
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Setup
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Composed
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Runs
import Proofs.Flock.Soundness.Discharge.ZkPubIn.WireP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Flat
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Sites
import Proofs.Flock.Soundness.Discharge.ZkPubIn.HmP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.SiteP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.RowsP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.TablesJ
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Bind
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Count
import Proofs.Flock.Soundness.Discharge.ZkPubIn.ComposedJ

/-!
# One hidden-output `--zk` sampled session with the verifier's public inputs

`flock-verify verify --zk --public-inputs F` over a circuit with public input ports (`HmIn.parse`): `Stmt.setupHidden`
with `own` F's bytes, which checks the public words against the verifier's own (`HmIn.checkOwn`), pins them in the
`in` regions (`HmIn.regions`) and leaves their columns out of the leaf map (`HmIn.delta`).

`Defs`: the acceptance (`AcceptsZKP`, `AcceptsZKRP`), the verifier's own words (`ownWord`, `OwnWords`), the public leaf
map (`LeavesPublic`, `pubLeaf`), the key program with a gate per public column (`keyProgP`), the public bits
(`pubBit`, `pubOnes`, `pubZeros`) and what setupHidden accepts (`SpecP`). `Spec`: `setupHidden_specP`. `Setup`:
HiddenExec's facts about setupHidden and setupTables with `own` (`setupHidden_wfP`, `setupTables_okHP`). `Tables`: the
draw's setups and `--zk` tables (`DrawSetupHP`, `DrawSetupZKHP`, `tablesAtZKHP`, `DrawOkZKHP`). `Composed`: Composed's
and ZkExec's session pieces at any acceptance event. `Runs`: HiddenExec's session runs and openings at `AcceptsZKP`.

These read the event, not the statement's rows. The rows, sites and binding at `keyProgP` are proved at canonical rows
(`Discharge.Canonical`), and the session and its registered reads come after them. `WireP`: `keyProgP`'s wiring
beside `keyProg`'s at the canonical output-row bits, the first of them. `Flat`, `Sites`: the statement and its flat
twin off the public leaves at canonical rows (`rowsFlatCP`), and `keyProgP`'s sites and places
(`SitesHP.progPlaces_keyProgP`), with a drawn table's acceptance at canonical rows (`TableHP.acc`). `HmP`: the row nets' slots at the statement, and
a canonical output row's data bit holds its copy (`msg_copyCP`). `SiteP`: a drawn unit's site, its registered `b ‖ c`
(`bc_site_ofHP`), its input copies (`copy_siteHP`), its returned outputs at canonical rows (`out_copyHCP`), and a
canonical row's `b ‖ c` (`bc_site_commitHCP`). `RowsP`: `lay` and `rs` at `keyProgP`'s places in canonical tables
(`keyLayoutCP`, `keyRowsCP`, `keyRowsCP_computes`), a public gate holding the verifier's own word's bit (`pub_siteCP`,
from `Canonical.public_pinnedC`). `TablesJ`: the session's `J` tables with `own` (`setupOfP_tableJH`,
`DrawSetupZHJP`, `planZHJP`) and their public shadow (`tableShHJP`, `planShHJP`). `Bind`: the canonical session's
sites (`sitesShCP`, `sitesShCP_bcSite`) and `lay`, `rs` at `keyProgP` moved to its `--zk` tables (`layZCP`, `rsZCP`,
`rsZCP_computes`). `Count`: the count curve at the public transcript `Xpub pubOnes pubZeros XplurZC`
(`zk_flock_countC_pub`), the public gates decoding to the verifier's own bits at the canonical places
(`pubDecodesZ_placeZCP`). `ComposedJ`: the `J`-table session's decoder, simulation and headline at any acceptance
event (`DecodesOnZKJA`, `RunsOnZCJA`, `zk_headline_exec_tablesA`).
-/
