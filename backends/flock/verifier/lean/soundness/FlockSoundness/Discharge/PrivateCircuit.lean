import FlockSoundness.Discharge.PrivateCircuit.Privacy
import FlockSoundness.Discharge.PrivateCircuit.Soundness
import FlockSoundness.Discharge.PrivateCircuit.InstanceCommit
import FlockSoundness.Discharge.PrivateCircuit.InstanceHides
import FlockSoundness.Discharge.PrivateCircuit.InstanceSound
import FlockSoundness.Discharge.PrivateCircuit.InstancePrivate
import FlockSoundness.Discharge.PrivateCircuit.InstanceBridge

/-!
# Protocol 2, the private circuit

The circuit is padded to a public size class and registered as a hidden program of the public universal unit `U cls`;
each session is a protocol-1 `--zk` session of `U cls`. The objects are in `PrivateCircuit/Defs.lean` and the statements
in `PrivateCircuit/Statements.lean`: circuit privacy (`circuit_private`, `circuit_indist`) from protocol 1's zero
knowledge and the registration's hiding, and soundness (`ran_registered`) from protocol 1's soundness of `U cls` and the
program commitment's binding under `cr/sha-512`.
-/
