import FlockSoundness.Discharge.PrivateCircuit.InterpPad
import FlockSoundness.Discharge.PrivateCircuit.Wiring
import FlockSoundness.Discharge.PrivateCircuit.Privacy
import FlockSoundness.Discharge.PrivateCircuit.Soundness
import FlockSoundness.Discharge.PrivateCircuit.InstanceCommit
import FlockSoundness.Discharge.PrivateCircuit.InstanceHides
import FlockSoundness.Discharge.PrivateCircuit.InstanceSound
import FlockSoundness.Discharge.PrivateCircuit.InstancePrivate
import FlockSoundness.Discharge.PrivateCircuit.InstanceTree
import FlockSoundness.Discharge.PrivateCircuit.InstanceJoint
import FlockSoundness.Discharge.PrivateCircuit.InstanceBridge
import FlockSoundness.Discharge.PrivateCircuit.InstanceDrawn
import FlockSoundness.Discharge.PrivateCircuit.InstanceReal
import FlockSoundness.Discharge.PrivateCircuit.InstanceRows
import FlockSoundness.Discharge.PrivateCircuit.InstanceWired
import FlockSoundness.Discharge.PrivateCircuit.InstanceRealH
import FlockSoundness.Discharge.PrivateCircuit.ReadsDraw
import FlockSoundness.Discharge.PrivateCircuit.ReadsSound
import FlockSoundness.Discharge.PrivateCircuit.ReadsNonVacuous
import FlockSoundness.Discharge.PrivateCircuit.ReadsHidden

/-!
# Protocol 2, the private circuit

The circuit is padded to a public size class and registered as a hidden program of the public universal unit `U cls`;
each session is a protocol-1 `--zk` session of `U cls`. The objects are in `PrivateCircuit/Defs.lean` and the statements
in `PrivateCircuit/Statements.lean`: circuit privacy (`circuit_private`, `circuit_indist`) from protocol 1's zero
knowledge and the registration's hiding, and soundness (`ran_registered`) from protocol 1's soundness of `U cls` and the
program commitment's binding under `cr/sha-512`. The universal unit's semantics, `UniversalUnit_v1`'s interpreter and
the padding, are in `PrivateCircuit/Interp.lean`; `pad_correct` (`InterpPad.lean`) is `PadCorrect` at it, and `pad_wrong`
reads "fewer than `K` wrong units" of the padded program as the same of the circuit. At the universal unit,
`private_circuit_sound` (`InstanceSound.lean`) counts `C`'s own wrong units through `pad_wrong`, and `private_circuit`
(`InstanceJoint.lean`) states soundness and privacy over one registration. The wiring between units is a public topology
of the class (`PrivateCircuit/Wiring.lean`): `wired_eval` composes correct units along it into `C` end to end, and
`wired_wrong` reads an owed output off it as a wrong unit.

With hidden reads (`PrivateCircuit/Reads*.lean`) each proof unit runs its class's unit circuit (a template) and the
program is a committed topology table: each unit reads its inputs from a committed row store at positions the table holds, through openings its
own circuit checks. `read_binding` makes a read off the registered store or table a SHA-512 collision, the registered
circuit's wrong units are judged against the store (`store_cone`), the draw's escape is taken at that one set
(`drawEscapes_realR`), and `private_circuit_sound_reads` is `ran_registered` at the class circuit and the table, with the
class circuit entering as a fact about its net (`ClassComputes`; `private_circuit_sound_reads_nonvacuous` meets every
hypothesis at once). `private_circuit_private_reads` and `private_circuit_indist_reads` are `circuit_private` and
`circuit_indist` at a public class list (`ClassList`: template, dimension class and padded count per block): the view is
a function of the class list and the owed outputs. The class list is public: it can tell apart circuits whose class lists
differ, and within one class list only the committed contents (dimensions within a class, read positions) are hidden.
No part of the full circuit is public under recursion: at hidden unit circuits (`PrivateCircuit/ReadsHidden.lean`) each
unit circuit is a committed witness in its topology entry, the class list is size-only (`SizeList`: gate bound, input and
output widths and padded unit count per class), and `private_circuit_sound_hidden` is the read binding at a committed
unit circuit, checked by the outer verifier circuit V*; `private_circuit_private_hidden` and
`private_circuit_indist_hidden` give a view that is a function of the size-only class list and the owed outputs.
-/
