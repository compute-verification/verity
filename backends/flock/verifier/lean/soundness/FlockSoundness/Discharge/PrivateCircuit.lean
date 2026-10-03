import FlockSoundness.Discharge.PrivateCircuit.InterpPad
import FlockSoundness.Discharge.PrivateCircuit.Privacy
import FlockSoundness.Discharge.PrivateCircuit.Soundness

/-!
# Protocol 2, the private circuit

The circuit is padded to a public size class and registered as a hidden program of the public universal unit `U cls`;
each session is a protocol-1 `--zk` session of `U cls`. The objects are in `PrivateCircuit/Defs.lean` and the statements
in `PrivateCircuit/Statements.lean`: circuit privacy (`circuit_private`, `circuit_indist`) from protocol 1's zero
knowledge and the registration's hiding, and soundness (`ran_registered`) from protocol 1's soundness of `U cls` and the
program commitment's binding under `cr/sha-512`. The universal unit's semantics, `UniversalUnit_v1`'s interpreter and
the padding, are in `PrivateCircuit/Interp.lean`; `pad_correct` (`InterpPad.lean`) is `PadCorrect` at it, and `pad_wrong`
reads "fewer than `K` wrong units" of the padded program as the same of the circuit.
-/
