module

public import Circuits.Program
meta import Circuits.Program

/-!
`Program.decode` on PROTOCOL.md §4.4's read at another width: one root over an input node of two 16-bit leaves, its
`ret` reading the node as 8-bit leaves (read recut, `q = 2`), as 3-bit leaves (no divisor), past its `N·q = 4` leaves,
off its whole leaves, and the same read of a node that is no input node. Each verdict is `codec.decode_program`'s on
the same bytes, and the gate layout refuses the recut read as `Program.circuit` does (`recut-gates`): `resolve`,
`operands` and `Prog.gate` each.
-/

namespace Flock.Test.Recut

open Flock.Program (decode resolve operands)

private def recutDesc_legal : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,4]},\"ret_type\":[\"Array\",4,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]],[\"Array\",4,[\"Value\",8]]]}"
private def recutDesc_width : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,2]},\"ret_type\":[\"Array\",2,[\"Value\",3]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]],[\"Array\",2,[\"Value\",3]]]}"
private def recutDesc_overrun : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,6]},\"ret_type\":[\"Array\",6,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]],[\"Array\",6,[\"Value\",8]]]}"
private def recutDesc_boundary : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,1,2]},\"ret_type\":[\"Array\",2,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]],[\"Array\",2,[\"Value\",8]]]}"
private def recutDesc_noninput : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\",\"Id_v1\"],\"definitions\":{\"Id_v1\":{\"bindings\":{},\"body\":{\"nodes\":[],\"params\":[[\"x\",[\"Array\",2,[\"Value\",16]]]],\"ret\":{\"affine\":[\"p\",0,0,2]},\"ret_type\":[\"Array\",2,[\"Value\",16]]},\"form\":\"composite\",\"id\":\"Id_v1\"},\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2]},\"type\":1}],\"fn\":1,\"form\":\"call\",\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,2]},\"ret_type\":[\"Array\",2,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]],[\"Array\",2,[\"Value\",8]]]}"

private def recutVerdict (s : String) : String :=
  match decode s.toUTF8 with
  | .ok P => s!"ok recut={(P.defs[P.root]!).body.recut.toList}"
  | .error e => e

#guard recutVerdict recutDesc_legal == "ok recut=[(0, 2)]"
#guard recutVerdict recutDesc_width ==
  "recut-width: an input node of 16-bit leaves read in 3-bit leaves: the reader's width divides the registered width"
#guard recutVerdict recutDesc_overrun ==
  "recut-overrun: {'affine': ['n', 0, 0, 6]} reads past the 2 16-bit leaves of its input node"
#guard recutVerdict recutDesc_boundary == "recut-boundary: {'affine': ['n', 0, 1, 2]} reads an input node of 16-bit " ++
  "leaves in 8-bit leaves other than whole registered leaves"
#guard recutVerdict recutDesc_noninput == "definition 'Root[P_v1]_v1': E_REINTERPRET: the body's return: leaf 0 is typed " ++
  "Value<8> but names leaf 0 of node 1, a Value<16>: a silent reinterpretation. A word is read only at the width it was " ++
  "committed with; a runtime word is split into limbs only by a registered LimbSplit Definition, which computes the " ++
  "limbs and their carries"
#guard (match decode recutDesc_legal.toUTF8 with
  | .ok P => match resolve P P.rootScope ((P.defs[P.root]!).body.ret.get 0) with
    | .ok g => s!"gate {g}"
    | .error e => e
  | .error e => e) ==
  "recut-gates: input nodes [0] are read in leaves of part of their width, which the gate layout doesn't lay out"
#guard (match decode recutDesc_legal.toUTF8 with
  | .ok P => match operands P P.rootScope ((P.defs[P.root]!).body.nodes[0]!) 0 with
    | .ok ops => s!"operands {ops.toList}"
    | .error e => e
  | .error e => e) ==
  "recut-gates: input nodes [0] are read in leaves of part of their width, which the gate layout doesn't lay out"
#guard (match decode recutDesc_legal.toUTF8 with
  | .ok P => match P.gate 0 with
    | .ok (p, _, _) => s!"gate of {p}"
    | .error e => e
  | .error e => e) ==
  "recut-gates: input nodes [0] are read in leaves of part of their width, which the gate layout doesn't lay out"

end Flock.Test.Recut

