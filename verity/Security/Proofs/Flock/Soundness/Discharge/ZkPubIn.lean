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
a canonical output row's data bit holds its copy (`msg_copyCP`).
-/
