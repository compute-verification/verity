import FlockSoundness.Discharge.PrivateCircuit.Statements
import FlockSoundness.Discharge.ZkView.Fixed

/-!
# Circuit privacy from protocol 1's zero knowledge

`circuit_private_of_zk`: at every registrant's salt, `ZkWireFixed` at the universal unit's tables moves the session to
`S_shvzk` on the dummy witness, at `zkGap`; what is left reads the program only through its registration, which hides it
up to `ε`. `circuit_indist_of_zk` is two such steps through the simulator. `circuit_private` and `circuit_indist`
discharge `ZkWireFixed` with `zk_wire_fixed`; the `_of_zk` forms take it as a hypothesis, to compose with a later form of
protocol 1's zero knowledge.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView
open scoped ENNReal

theorem circuit_private_of_zk (hzk : ZkWireFixed) : CircuitPrivate := by
  sorry

theorem circuit_indist_of_zk (hzk : ZkWireFixed) : CircuitIndist := by
  sorry

/-- **Circuit privacy, the simulation form**, at protocol 1's proved zero knowledge (`zk_wire_fixed`). -/
theorem circuit_private : CircuitPrivate := circuit_private_of_zk zk_wire_fixed

/-- **Circuit privacy, the indistinguishability form**, at protocol 1's proved zero knowledge (`zk_wire_fixed`). -/
theorem circuit_indist : CircuitIndist := circuit_indist_of_zk zk_wire_fixed

end FlockSoundness.Discharge.PrivateCircuit
