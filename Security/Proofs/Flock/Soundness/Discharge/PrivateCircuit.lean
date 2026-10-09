import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Privacy
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Soundness
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.ReadsSound
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.ReadsHidden
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.RouteP

/-!
# Protocol 2, the private circuit

The circuit is padded to a public size class and registered as a hidden program of the public universal unit `U cls`;
each session is a protocol-1 `--zk` session of `U cls`. The objects are in `PrivateCircuit/Defs.lean` and the statements
in `PrivateCircuit/Statements.lean`: circuit privacy (`circuit_private`, `circuit_indist`) from protocol 1's zero
knowledge and the registration's hiding, and soundness (`ran_registered`) from protocol 1's soundness of `U cls` and the
program commitment's binding under `cr/sha-512`.

With hidden reads (`PrivateCircuit/Reads*.lean`, on the instance's objects in `PrivateCircuit/InstanceDefs.lean`) each
proof unit runs its class's unit circuit (a template) and the program is a committed topology table: each unit reads its
inputs from a committed row store at positions the table holds, through openings its own circuit checks. `read_binding`
makes a read off the registered store or table a SHA-512 collision, the registered circuit's wrong units are judged
against the store (`store_cone`), and `private_circuit_sound_reads` is `ran_registered` at the class circuit and the
table, with the class circuit entering as a fact about its net (`ClassComputes`). At hidden unit circuits
(`PrivateCircuit/ReadsHidden.lean`) each unit circuit is a committed witness in its topology entry and the class list is
size-only (`SizeList`); `private_circuit_sound_hidden` is the read binding at a committed unit circuit.
Route P (`PrivateCircuit/RouteP.lean`) hides units that all differ: one public table of gate rows, each unit's hidden
wiring checked by a grand product (`wiring_sound`, `wiring_escape`), and `route_p_sound`.
-/
