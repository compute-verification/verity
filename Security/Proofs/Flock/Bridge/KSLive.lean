import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.OpeningsCanon
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.Rows
import Proofs.Flock.Bridge.Transports

/-!
# The sampled audit's knowledge soundness at the extracted values and under `verify --zk`, proved

`Flock.Bridge.sampledKSLiveXr : Flock.Bridge.SampledKSLiveXr`: `AssemblyTargetC` (`assemblyTargetC`) with canonical
transport (`transportsLiveC`), M1's core (`faultyRightTargetC`), the slot count (`partBlocks`), the count lemma (`count`)
and `EndToEndHiddenC` (`endToEndHiddenC_of`, from `openingsCanon`) plugged in.
`Flock.Bridge.sampledKSLiveV2Z : Flock.Guarantees.SampledKSLiveV2Z`: the claims at the registrant's rows under
`verify --zk`, from the count curve at the rows link (`sampledKSLiveV2Z_of_count`), canonical transport, the count lemma
and canonical openings. `SampledKSLiveV2`, `SampledKSLiveV3` (`RowsReg`) and `SampledKSLivePub` (`Public`) are proved
from these two.
-/

/-- **The sampled audit's knowledge soundness, live, at the extracted values** (`Flock.Bridge.SampledKSLiveXr`). -/
theorem Flock.Bridge.sampledKSLiveXr : Flock.Bridge.SampledKSLiveXr :=
  Flock.Bridge.assemblyTargetC Flock.Bridge.transportsLiveC Flock.Bridge.faultyRightTargetC Flock.Bridge.partBlocks
    Flock.Bridge.count (Flock.Bridge.endToEndHiddenC_of Flock.Bridge.openingsCanon)

/-- **The sampled audit's knowledge soundness, live, at the registrant's rows, under `verify --zk`**
(`Flock.Guarantees.SampledKSLiveV2Z`). -/
theorem Flock.Bridge.sampledKSLiveV2Z : Flock.Guarantees.SampledKSLiveV2Z :=
  Flock.Bridge.sampledKSLiveV2Z_of_count Flock.Bridge.transportsLiveC Flock.Bridge.count Flock.Bridge.openingsCanon
