/-
`TemplateQuery.Desc.ofProg`, the template queries' reading of the program `Program.decode` decoded, reads what `Desc.parse`
reads from the same bytes: the root, the program's SHA-512, and at every definition the decoder reached its nodes (form,
member count, callee) and gates, which are the gates the decoder counted; and a template query's units with their gates, or
its refusal. Each program lists its callees in another order than the decoder numbers its definitions: `pops`, two
`TQPair_v0` and three `TQTriple_v0` instances over `Input8_v1` gates (the codec's `mixed_population_program`), and
`Flock.Test.RegProgram`'s `plain` and `reg`, whose roots a template query refuses.
-/
import Flock.Test.RegProgram

namespace Flock.Test.DescOfProg

open Flock.Test.RegProgram (reg plain)
open TemplateQuery (Desc)

def pops : ByteArray := "{\"annotations\":{},\"callees\":[\"TQXor8_v1\",\"Input8_v1\",\"TQPair_v0\",\"TQTriple_v0\"],\"definitions\":{\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Root[TemplatePopulations_v1{templates=('TQPair_v0','TQTriple_v0'),counts=(2,3)}]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":9,\"out\":2},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":3}],\"axes\":[0],\"fn\":2,\"form\":\"batch\",\"n\":2,\"out\":4},{\"args\":[{\"refs\":{\"affine\":[\"n\",1,0,9]},\"type\":5}],\"axes\":[0],\"fn\":3,\"form\":\"batch\",\"n\":3,\"out\":6}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,2]},{\"affine\":[\"n\",3,0,3]}]},\"ret_type\":[\"Tuple\",[[\"Array\",2,[\"Value\",8]],[\"Array\",3,[\"Value\",8]]]]},\"form\":\"composite\",\"id\":\"Root[TemplatePopulations_v1{templates=('TQPair_v0','TQTriple_v0'),counts=(2,3)}]_v1\"},\"TQPair_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,1,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Array\",2,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"TQPair_v0\"},\"TQTriple_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,1,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,2,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Array\",3,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"TQTriple_v0\"},\"TQXor8_v1\":{\"form\":\"primitive\",\"id\":\"TQXor8_v1\",\"params\":[[\"a\",[\"Value\",8]],[\"b\",[\"Value\",8]]],\"ret\":[\"Value\",8]}},\"program_fn\":\"TemplatePopulations_v1{templates=[\\\"TQPair_v0\\\",\\\"TQTriple_v0\\\"],counts=[2,3]}\",\"root\":\"Root[TemplatePopulations_v1{templates=('TQPair_v0','TQTriple_v0'),counts=(2,3)}]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",8],[\"Array\",4,[\"Value\",8]],[\"Array\",9,[\"Value\",8]],[\"Array\",2,[\"Array\",2,[\"Value\",8]]],[\"Array\",2,[\"Value\",8]],[\"Array\",3,[\"Array\",3,[\"Value\",8]]],[\"Array\",3,[\"Value\",8]]]}".toUTF8

def templates : Array String := #["TQPair_v0", "TQTriple_v0"]

/-- What the template queries read at definition `id`: whether it is a primitive, and each node's form, member count and
callee. -/
def readAt (d : Desc) (id : String) : Option (Bool × Array (String × Nat × String)) :=
  (d.def? id).map fun df => (df.prim, df.nodes.map fun x => (x.form, x.n, d.callees.getD x.fn ""))

/-- The template query over `ts`: each unit's root node, template and gates `[lo, hi)`; none when it refuses. -/
def queryUnits (d : Desc) (ts : Array String) : Option (Array (Option (Nat × String) × Option (Nat × Nat))) :=
  (TemplateQuery.evaluate d ts).toOption.map fun rs => (Array.range (TemplateQuery.population rs)).map fun i =>
    ((TemplateQuery.locate rs i).map fun (k, _, t) => (k, t), TemplateQuery.gates d rs i)

def refusal (d : Desc) (ts : Array String) : Option String :=
  match TemplateQuery.evaluate d ts with
  | .ok _ => none
  | .error e => some e

/-- `Desc.parse` of `b` and `Desc.ofProg` of `Program.decode` of `b` read `b` alike, and the template query over `ts` on
each gives the same units or the same refusal. -/
def agree (b : ByteArray) (ts : Array String) : Bool :=
  match TemplateQuery.Desc.parse b, Program.decode b with
  | .ok d, .ok P =>
    let e := Desc.ofProg P
    d.root == e.root && d.sha512 == e.sha512 && queryUnits d ts == queryUnits e ts && refusal d ts == refusal e ts &&
      P.defs.all fun df => readAt d df.key == readAt e df.key && d.gates df.key == df.gates && e.gates df.key == df.gates
  | _, _ => false

/-- The descriptor lists its callees in another order than the decoder numbers its definitions. -/
def reordered (b : ByteArray) : Bool :=
  match TemplateQuery.Desc.parse b, Program.decode b with
  | .ok d, .ok P => (P.defs.map (·.key)).filter d.callees.contains != d.callees
  | _, _ => false

#guard reordered pops && reordered plain && reordered reg

#guard agree pops templates
-- five units after the 13 Input gates: two of `TQPair_v0` (a gate each), three of `TQTriple_v0` (two each)
#guard ((Program.decode pops).toOption.bind fun P => queryUnits (Desc.ofProg P) templates) == some #[
  (some (2, "TQPair_v0"), some (13, 14)), (some (2, "TQPair_v0"), some (14, 15)), (some (3, "TQTriple_v0"), some (15, 17)),
  (some (3, "TQTriple_v0"), some (17, 19)), (some (3, "TQTriple_v0"), some (19, 21))]

-- a root node that is a batch or a scan of a template the query doesn't list is refused alike
#guard agree plain #["TPair_v1"] && agree reg #["TPair_v1"]
#guard agree plain #["TStep_v1"] && agree reg #["TStep_v1"]

end Flock.Test.DescOfProg
