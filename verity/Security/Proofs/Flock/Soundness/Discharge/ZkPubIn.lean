import Proofs.Flock.Soundness.Discharge.ZkPubIn.Defs
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Spec
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Tables
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Setup
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Flat
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Sites
import Proofs.Flock.Soundness.Discharge.ZkPubIn.HmP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.SiteP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.KeyWire
import Proofs.Flock.Soundness.Discharge.ZkPubIn.InPin
import Proofs.Flock.Soundness.Discharge.ZkPubIn.KeyP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Bind
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Count
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Composed
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Runs
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Reg
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Sound
import Proofs.Flock.Soundness.Discharge.ZkPubIn.SoundCustody

/-!
# One hidden-output `--zk` sampled session with the verifier's public inputs

`flock-verify verify --zk --public-inputs F` over a circuit with public input ports (`HmIn.parse`): `Stmt.setupHidden`
with `own` F's bytes, which checks the public words against the verifier's own (`HmIn.checkOwn`), pins them in the
`in` regions (`HmIn.regions`) and leaves their columns out of the leaf map (`HmIn.delta`).

`Defs`: the acceptance (`AcceptsZKP`, `AcceptsZKRP`), the verifier's own words (`ownWord`, `OwnWords`), the public leaf
map (`LeavesPublic`, `pubLeaf`), the key program with a gate per public column (`keyProgP`), the public bits
(`pubBit`, `pubOnes`, `pubZeros`) and what setupHidden accepts (`SpecP`). `Spec`: `setupHidden_specP`. `Setup`: HiddenExec's
facts about setupHidden and setupTables with `own` (`setupHidden_wfP`, `setupTables_okHP`). `Tables`: the
draw's setups and `--zk` tables (`DrawSetupHP`, `DrawSetupZKHP`, `tablesAtZKHP`, `DrawOkZKHP`). `Flat`: the statement's layout (`layoutP`) and its flat twin's rows
off the public leaves (`UnitRowsFlatP`, `rowsFlatP`), and the aliasing they give (`aliased_of_copiesHP`). `HmP`: Hm's row chain at the statement with public inputs (`row_bc_flatP`).
`KeyWire`: `keyProgP`'s wiring beside `keyProg`'s. `InPin`: the `in` regions pin a public leaf's bit at a drawn unit's site
to the verifier's own word's (`pub_siteHP`).
`Sites`, `SiteP`, `KeyP`, `Bind`: the sites, the key layout and rows at `keyProgP`, and their `--zk` moves (`placeZHP`, `rsZHP`). `Count`: the
public-input step (`zk_flock_countC_execOS_atHP`, the count curve at `Xpub pubOnes pubZeros XplurZC`). `Composed`:
Composed's simulation at any acceptance (`simZA`, `zk_headline_exec_onA`, `RunsOnZCA`). `Runs`: the record
(`RecordsLiveZKP`) and `runsZC_liveHP`. `Reg`: the registered reads (`OwnReadsHP`, `checked_of_acceptsHP`). `Sound`:
`zk_session_soundHP`, `zk_session_soundHRP`; `SoundCustody`: their `_custody` forms, under `live-verifier`
(`Assumptions.ZkPubIn.RecordCustodyZKP`).
-/
