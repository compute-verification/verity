import FlockSoundness.Discharge.ZkPubIn.Defs
import FlockSoundness.Discharge.ZkPubIn.Spec
import FlockSoundness.Discharge.ZkPubIn.Tables
import FlockSoundness.Discharge.ZkPubIn.Sites
import FlockSoundness.Discharge.ZkPubIn.KeyP
import FlockSoundness.Discharge.ZkPubIn.Bind
import FlockSoundness.Discharge.ZkPubIn.Count
import FlockSoundness.Discharge.ZkPubIn.Composed
import FlockSoundness.Discharge.ZkPubIn.Runs
import FlockSoundness.Discharge.ZkPubIn.Reg
import FlockSoundness.Discharge.ZkPubIn.Sound
import FlockSoundness.Discharge.ZkPubIn.SoundCustody

/-!
# One hidden-output `--zk` sampled session with the verifier's public inputs

`flock-verify verify --zk --public-inputs F` over a circuit with public input ports (`HmIn.parse`): `Stmt.setupHidden`
with `own` F's bytes, which checks the public words against the verifier's own (`HmIn.checkOwn`), pins them in the
`in` regions (`HmIn.regions`) and leaves their columns out of the leaf map (`HmIn.delta`).

`Defs`: the acceptance (`AcceptsZKP`, `AcceptsZKRP`), the verifier's own words (`ownWord`, `OwnWords`), the public leaf
map (`LeavesPublic`, `pubLeaf`), the key program with a gate per public column (`keyProgP`), the public bits
(`pubBit`, `pubOnes`, `pubZeros`) and what setupHidden accepts (`SpecP`). `Spec`: `setupHidden_specP`. `Tables`: the
draw's setups and `--zk` tables (`DrawSetupHP`, `DrawSetupZKHP`, `tablesAtZKHP`, `DrawOkZKHP`). `Sites`, `KeyP`,
`Bind`: the sites, the key layout and rows at `keyProgP`, and their `--zk` moves (`placeZHP`, `rsZHP`). `Count`: the
public-input step (`zk_flock_countC_execOS_atHP`, the count curve at `Xpub pubOnes pubZeros XplurZC`). `Composed`:
Composed's simulation at any acceptance (`simZA`, `zk_headline_exec_onA`, `RunsOnZCA`). `Runs`: the record
(`RecordsLiveZKP`) and `runsZC_liveHP`. `Reg`: the registered reads (`OwnReadsHP`, `checked_of_acceptsHP`). `Sound`:
`zk_session_soundHP`, `zk_session_soundHRP`; `SoundCustody`: their `_custody` forms, under `live-verifier`
(`Assumptions.ZkPubIn.RecordCustodyZKP`).
-/
