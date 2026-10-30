/-
`Flock.Annot.check`, at the size of the description, against `Flock.Annot.reference`, the same meaning decided gate by
gate (`Flock.AnnotExpand`): equal results on every term and meaning below, and the results themselves on a few. The
programs are `TGemm{K=32,N=6}` (Input batches, a composite batch, nested calls; 248 gates, 224 of them input gates) and
`TSynth{N=4}` (two zip-able batches, a constant, a scan; 21 gates, 4 of them input gates), as `verity.ml.circuits` encodes
them (`verity/core/difftest/tests/test_annot.py` builds them, and holds the sets to `query_ast`'s).
-/
import Flock.AnnotExpand

namespace Flock.Test.Annot

open Lean (Json)
open Flock.Annot

def gemm : String := "{\"annotations\":{},\"callees\":[\"TZero32_v1\",\"TDot16_v1\",\"TDot_v1{K=32}\",\"TCvt16_v1\",\"Input16_v1\",\"TGemmCoordinate_v1{K=32}\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[TGemm_v1{K=32,N=6}]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":4,\"form\":\"batch\",\"n\":32,\"out\":2},{\"args\":[],\"axes\":[],\"fn\":4,\"form\":\"batch\",\"n\":192,\"out\":4},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,32]},\"type\":2},{\"refs\":{\"affine\":[\"n\",1,0,192]},\"type\":5}],\"axes\":[null,0],\"fn\":5,\"form\":\"batch\",\"n\":6,\"out\":6}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,6]},\"ret_type\":[\"Array\",6,[\"Value\",16]]},\"form\":\"composite\",\"id\":\"Root[TGemm_v1{K=32,N=6}]_v1\"},\"TCvt16_v1\":{\"form\":\"primitive\",\"id\":\"TCvt16_v1\",\"params\":[[\"a\",[\"Value\",32]]],\"ret\":[\"Value\",16]},\"TDot16_v1\":{\"form\":\"primitive\",\"id\":\"TDot16_v1\",\"params\":[[\"acc\",[\"Value\",32]],[\"a\",[\"Array\",16,[\"Value\",16]]],[\"b\",[\"Array\",16,[\"Value\",16]]]],\"ret\":[\"Value\",32]},\"TDot_v1{K=32}\":{\"bindings\":{\"K\":32},\"body\":{\"nodes\":[{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,16]},\"type\":1},{\"refs\":{\"affine\":[\"p\",1,0,16]},\"type\":1}],\"fn\":1,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",1,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,16,16]},\"type\":1},{\"refs\":{\"affine\":[\"p\",1,16,16]},\"type\":1}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Array\",32,[\"Value\",16]]],[\"w\",[\"Array\",32,[\"Value\",16]]]],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",32]},\"form\":\"composite\",\"id\":\"TDot_v1\"},\"TGemmCoordinate_v1{K=32}\":{\"bindings\":{\"K\":32},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,32]},\"type\":2},{\"refs\":{\"affine\":[\"p\",1,0,32]},\"type\":2}],\"fn\":2,\"form\":\"call\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0}],\"fn\":3,\"form\":\"primitive\",\"out\":3}],\"params\":[[\"x\",[\"Array\",32,[\"Value\",16]]],[\"wrow\",[\"Array\",32,[\"Value\",16]]]],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",16]},\"form\":\"composite\",\"id\":\"TGemmCoordinate_v1\"},\"TZero32_v1\":{\"form\":\"primitive\",\"id\":\"TZero32_v1\",\"params\":[],\"ret\":[\"Value\",32]}},\"program_fn\":\"TGemm_v1{K=32,N=6}\",\"root\":\"Root[TGemm_v1{K=32,N=6}]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Array\",16,[\"Value\",16]],[\"Array\",32,[\"Value\",16]],[\"Value\",16],[\"Array\",192,[\"Value\",16]],[\"Array\",6,[\"Array\",32,[\"Value\",16]]],[\"Array\",6,[\"Value\",16]]]}"

def synth : String := "{\"annotations\":{},\"callees\":[\"TAdd32_v1\",\"TMul32_v1\",\"Input32_v1\",\"TZero32_v1\",\"TScanStep_v1\"],\"definitions\":{\"Input32_v1\":{\"form\":\"primitive\",\"id\":\"Input32_v1\",\"params\":[],\"ret\":[\"Value\",32]},\"Root[TSynth_v1{N=4}]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":2,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":1},{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":1}],\"axes\":[0,0],\"fn\":1,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",1,0,4]},\"type\":1},{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":1}],\"axes\":[0,0],\"fn\":0,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[],\"fn\":3,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",3,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"n\",2,0,4]},\"type\":1}],\"fn\":4,\"form\":\"scan\",\"n\":4,\"n_xs\":1,\"out\":2}],\"params\":[],\"ret\":{\"affine\":[\"n\",4,0,5]},\"ret_type\":[\"Tuple\",[[\"Value\",32],[\"Array\",4,[\"Value\",32]]]]},\"form\":\"composite\",\"id\":\"Root[TSynth_v1{N=4}]_v1\"},\"TAdd32_v1\":{\"form\":\"primitive\",\"id\":\"TAdd32_v1\",\"params\":[[\"a\",[\"Value\",32]],[\"b\",[\"Value\",32]]],\"ret\":[\"Value\",32]},\"TMul32_v1\":{\"form\":\"primitive\",\"id\":\"TMul32_v1\",\"params\":[[\"a\",[\"Value\",32]],[\"b\",[\"Value\",32]]],\"ret\":[\"Value\",32]},\"TScanStep_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",1,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",1,0,1]},\"type\":0}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"carry\",[\"Value\",32]],[\"x\",[\"Value\",32]]],\"ret\":{\"concat\":[{\"affine\":[\"n\",0,0,1]},{\"affine\":[\"n\",1,0,1]}]},\"ret_type\":[\"Tuple\",[[\"Value\",32],[\"Value\",32]]]},\"form\":\"composite\",\"id\":\"TScanStep_v1\"},\"TZero32_v1\":{\"form\":\"primitive\",\"id\":\"TZero32_v1\",\"params\":[],\"ret\":[\"Value\",32]}},\"program_fn\":\"TSynth_v1{N=4}\",\"root\":\"Root[TSynth_v1{N=4}]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Array\",4,[\"Value\",32]],[\"Tuple\",[[\"Value\",32],[\"Array\",4,[\"Value\",32]]]]]}"


def nest : String := "{\"annotations\":{},\"callees\":[\"TZero32_v1\",\"TDot16_v1\",\"TDot_v1{K=16}\",\"TCvt16_v1\",\"TGemmCoordinate_v1{K=16}\",\"Input16_v1\",\"TGemm_v1{K=16,N=4}\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[TNest_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":5,\"form\":\"batch\",\"n\":48,\"out\":5},{\"args\":[],\"axes\":[],\"fn\":5,\"form\":\"batch\",\"n\":64,\"out\":6},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,48]},\"type\":7},{\"refs\":{\"affine\":[\"n\",1,0,64]},\"type\":3}],\"axes\":[0,null],\"fn\":6,\"form\":\"batch\",\"n\":3,\"out\":8}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,12]},\"ret_type\":[\"Array\",3,[\"Array\",4,[\"Value\",16]]]},\"form\":\"composite\",\"id\":\"Root[TNest_v1]_v1\"},\"TCvt16_v1\":{\"form\":\"primitive\",\"id\":\"TCvt16_v1\",\"params\":[[\"a\",[\"Value\",32]]],\"ret\":[\"Value\",16]},\"TDot16_v1\":{\"form\":\"primitive\",\"id\":\"TDot16_v1\",\"params\":[[\"acc\",[\"Value\",32]],[\"a\",[\"Array\",16,[\"Value\",16]]],[\"b\",[\"Array\",16,[\"Value\",16]]]],\"ret\":[\"Value\",32]},\"TDot_v1{K=16}\":{\"bindings\":{\"K\":16},\"body\":{\"nodes\":[{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,16]},\"type\":1},{\"refs\":{\"affine\":[\"p\",1,0,16]},\"type\":1}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Array\",16,[\"Value\",16]]],[\"w\",[\"Array\",16,[\"Value\",16]]]],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",32]},\"form\":\"composite\",\"id\":\"TDot_v1\"},\"TGemmCoordinate_v1{K=16}\":{\"bindings\":{\"K\":16},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,16]},\"type\":1},{\"refs\":{\"affine\":[\"p\",1,0,16]},\"type\":1}],\"fn\":2,\"form\":\"call\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0}],\"fn\":3,\"form\":\"primitive\",\"out\":2}],\"params\":[[\"x\",[\"Array\",16,[\"Value\",16]]],[\"wrow\",[\"Array\",16,[\"Value\",16]]]],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",16]},\"form\":\"composite\",\"id\":\"TGemmCoordinate_v1\"},\"TGemm_v1{K=16,N=4}\":{\"bindings\":{\"K\":16,\"N\":4},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,16]},\"type\":1},{\"refs\":{\"affine\":[\"p\",1,0,64]},\"type\":3}],\"axes\":[null,0],\"fn\":4,\"form\":\"batch\",\"n\":4,\"out\":4}],\"params\":[[\"x\",[\"Array\",16,[\"Value\",16]]],[\"w\",[\"Array\",4,[\"Array\",16,[\"Value\",16]]]]],\"ret\":{\"affine\":[\"n\",0,0,4]},\"ret_type\":[\"Array\",4,[\"Value\",16]]},\"form\":\"composite\",\"id\":\"TGemm_v1\"},\"TZero32_v1\":{\"form\":\"primitive\",\"id\":\"TZero32_v1\",\"params\":[],\"ret\":[\"Value\",32]}},\"program_fn\":\"TNest_v1\",\"root\":\"Root[TNest_v1]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Array\",16,[\"Value\",16]],[\"Value\",16],[\"Array\",4,[\"Array\",16,[\"Value\",16]]],[\"Array\",4,[\"Value\",16]],[\"Array\",48,[\"Value\",16]],[\"Array\",64,[\"Value\",16]],[\"Array\",3,[\"Array\",16,[\"Value\",16]]],[\"Array\",3,[\"Array\",4,[\"Value\",16]]]]}"
/-- A result as text, or `reject CODE` for a refusal. -/
def run (f : Meaning → Flock.Program.Prog → Desc → V CheckResult) (p a m : String) : String :=
  match (do
      let P ← Flock.Program.decode p.toUTF8
      let D ← Desc.ofJson (← Json.parse a)
      f (← Meaning.ofJson (← Json.parse m)) P D) with
  | .ok v => v.toJson.compress
  | .error e => "reject " ++ (e.splitOn ":").head!

/-- The check's result is the reference's on program `p`, annotation `a`, every meaning `ms`. -/
def same (p a : String) (ms : List String) : Bool := ms.all fun m => run check p a m == run reference p a m

def meanings : List String := ["{\"tiling\":\"partition\"}", "{\"tiling\":\"cover\"}", "{\"max_count\":6}", "{\"max_gates\":4}", "{}", "{\"tiling\":\"partition\",\"max_gates\":12,\"max_count\":3}"]

#guard same gemm "{\"query\":{\"op\":\"default\",\"w\":32}}" meanings
#guard same gemm "{\"query\":{\"op\":\"instances\",\"path\":\"@2\"}}" meanings
#guard same gemm "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"instances\",\"path\":\"@2\"},{\"op\":\"instances\",\"path\":\"//TDot_v1\"}]}}" meanings
#guard same gemm "{\"query\":{\"op\":\"instances\",\"path\":\"//TDot_v1\"}}" meanings
#guard same gemm "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"instances\",\"path\":\"//TDot_v1\"},{\"base\":null,\"op\":\"rest\"}]}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"op\":\"instances\",\"path\":\"//TGemmCoordinate_v1\"},\"inner\":{\"op\":\"concat\",\"parts\":[{\"op\":\"instances\",\"path\":\"@call:TDot_v1\"},{\"base\":null,\"op\":\"rest\"}]},\"op\":\"each\"}}" meanings
#guard same gemm "{\"query\":{\"k\":4,\"op\":\"members\",\"path\":\"//TGemm_v1\",\"sel\":\"@2\"}}" meanings
#guard same gemm "{\"query\":{\"k\":1,\"op\":\"members\",\"path\":\"//TDot_v1\",\"sel\":\"@*\"}}" meanings
#guard same gemm "{\"query\":{\"op\":\"root\"}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"k\":1,\"op\":\"members\",\"path\":\"//TGemmCoordinate_v1\",\"sel\":\"@*\"},\"key\":\"definition\",\"op\":\"group\"}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"k\":1,\"op\":\"members\",\"path\":\"//TGemmCoordinate_v1\",\"sel\":\"@*\"},\"key\":\"name\",\"op\":\"group\"}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"op\":\"instances\",\"path\":\"@2\"},\"map\":{\"affine\":[1,0,4]},\"op\":\"label\"}}" meanings
#guard same gemm "{\"query\":{\"op\":\"table\",\"path\":\".\",\"table\":\"t\"},\"tables\":{\"t\":{\"Root[TGemm_v1{K=32,N=6}]_v1\":[null,null,0]}}}" meanings
#guard same gemm "{\"query\":{\"op\":\"table\",\"path\":\".\",\"table\":\"t\"},\"tables\":{\"t\":{\"Root[TGemm_v1{K=32,N=6}]_v1\":[0,1,2]}}}" meanings
#guard same gemm "{\"query\":{\"op\":\"table\",\"path\":\".\",\"table\":\"t\"},\"tables\":{\"t\":{\"Root[TGemm_v1{K=32,N=6}]_v1\":[null,0,0]}}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"k\":1,\"op\":\"members\",\"path\":\"//TGemmCoordinate_v1\",\"sel\":\"@*\"},\"key\":[\"name\",\"static:K\"],\"op\":\"group\"}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"op\":\"default\",\"w\":16},\"key\":\"static:K\",\"op\":\"group\"}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"k\":1,\"op\":\"members\",\"path\":\".\",\"sel\":\"@*\"},\"key\":[],\"op\":\"group\"}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"op\":\"instances\",\"path\":\"@2\"},\"map\":{\"table\":\"u\"},\"op\":\"label\"},\"tables\":{\"u\":{\"Root[TGemm_v1{K=32,N=6}]_v1\":[null,null,0]}}}" meanings
#guard same gemm "{\"query\":{\"base\":{\"op\":\"default\",\"w\":16},\"map\":{\"table\":\"u\"},\"op\":\"label\"},\"tables\":{\"u\":{\"Root[TGemm_v1{K=32,N=6}]_v1\":[0,1,1]}}}" meanings
#guard same synth "{\"query\":{\"op\":\"default\",\"w\":32}}" meanings
#guard same synth "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"zip\",\"path\":\".\",\"sels\":[\"@1\",\"@2\"]},{\"base\":null,\"op\":\"rest\"}]}}" meanings
#guard same synth "{\"query\":{\"op\":\"zip\",\"path\":\".\",\"sels\":[\"@1\",\"@2\"]}}" meanings
#guard same synth "{\"query\":{\"base\":{\"op\":\"instances\",\"path\":\"//TScanStep_v1\"},\"inner\":{\"op\":\"concat\",\"parts\":[{\"k\":1,\"op\":\"members\",\"path\":\".\",\"sel\":\"@0\"},{\"base\":null,\"op\":\"rest\"}]},\"op\":\"each\"}}" meanings
#guard same synth "{\"query\":{\"k\":3,\"op\":\"members\",\"path\":\".\",\"sel\":\"@*\"}}" meanings
#guard same synth "{\"query\":{\"op\":\"concat\",\"parts\":[{\"k\":1,\"op\":\"members\",\"path\":\".\",\"sel\":\"@1\"},{\"k\":1,\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\"},{\"k\":1,\"op\":\"members\",\"path\":\".\",\"sel\":\"@3\"},{\"op\":\"instances\",\"path\":\"//TScanStep_v1\"}]}}" meanings
#guard same synth "{\"query\":{\"base\":{\"k\":1,\"op\":\"members\",\"path\":\"//TScanStep_v1\",\"sel\":\"@*\"},\"map\":{\"affine\":[1,0,2]},\"op\":\"label\"}}" meanings
#guard same synth "{\"query\":{\"op\":\"table\",\"path\":\".\",\"table\":\"t\"},\"tables\":{\"t\":{\"Root[TSynth_v1{N=4}]_v1\":[null,0,0,1,2]}}}" meanings
#guard same synth "{\"query\":{\"op\":\"table\",\"path\":\".\",\"table\":\"t\"},\"tables\":{\"t\":{\"Root[TSynth_v1{N=4}]_v1\":[null,1,0,1,0]}}}" meanings

#guard same synth "{\"query\":{\"base\":{\"k\":1,\"op\":\"members\",\"path\":\".\",\"sel\":\"@*\"},\"map\":{\"table\":\"u\"},\"op\":\"label\"},\"tables\":{\"u\":{\"Root[TSynth_v1{N=4}]_v1\":[null,0,0,1,1]}}}" meanings
#guard same synth "{\"query\":{\"base\":{\"op\":\"zip\",\"path\":\".\",\"sels\":[\"@1\",\"@2\"]},\"map\":{\"table\":\"u\"},\"op\":\"label\"},\"tables\":{\"u\":{\"Root[TSynth_v1{N=4}]_v1\":[null,0,1,null,null]}}}" meanings
/-! The results: `default(32)` holds the Input nodes' gates, so it is no partition; the six coordinates (4 gates each:
`TDot`'s 3 and `TCvt16`) are one; with `TDot` again inside them they overlap; `TDot` alone leaves `TCvt16` in no set;
grouped by Definition the coordinates' members are `TDot{K=32}` (18 gates) and `TCvt16` (6); four affine labels are more
than three; `synth`'s `rest()` holds its Input batch, and its batches, constant and scan steps partition it. -/
#guard run check gemm "{\"query\":{\"op\":\"default\",\"w\":32}}" "{\"tiling\":\"partition\"}" == "{\"accept\":false,\"code\":\"input-held\",\"count\":230}"
#guard run check gemm "{\"query\":{\"op\":\"instances\",\"path\":\"@2\"}}" "{\"tiling\":\"partition\",\"max_gates\":4}" == "{\"accept\":true,\"count\":6}"
#guard run check gemm "{\"query\":{\"op\":\"instances\",\"path\":\"@2\"}}" "{\"max_gates\":3}" == "{\"accept\":false,\"code\":\"too-large\",\"count\":6}"
#guard run check gemm "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"instances\",\"path\":\"@2\"},{\"op\":\"instances\",\"path\":\"//TDot_v1\"}]}}" "{\"tiling\":\"partition\"}" == "{\"accept\":false,\"code\":\"overlap\",\"count\":12}"
#guard run check gemm "{\"query\":{\"op\":\"instances\",\"path\":\"//TDot_v1\"}}" "{\"tiling\":\"cover\"}" == "{\"accept\":false,\"code\":\"gate-in-no-set\",\"count\":6}"
#guard run check gemm "{\"query\":{\"op\":\"group\",\"key\":\"definition\",\"base\":{\"op\":\"members\",\"path\":\"//TGemmCoordinate_v1\",\"sel\":\"@*\",\"k\":1}}}" "{\"tiling\":\"partition\",\"max_gates\":18}" == "{\"accept\":true,\"count\":2}"
#guard run check gemm "{\"query\":{\"op\":\"group\",\"key\":\"definition\",\"base\":{\"op\":\"members\",\"path\":\"//TGemmCoordinate_v1\",\"sel\":\"@*\",\"k\":1}}}" "{\"max_gates\":17}" == "{\"accept\":false,\"code\":\"too-large\",\"count\":2}"
#guard run check gemm "{\"query\":{\"op\":\"label\",\"map\":{\"affine\":[1,0,4]},\"base\":{\"op\":\"instances\",\"path\":\"@2\"}}}" "{\"tiling\":\"partition\",\"max_count\":3}" == "{\"accept\":false,\"code\":\"too-many\",\"count\":4}"
#guard run check synth "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"zip\",\"path\":\".\",\"sels\":[\"@1\",\"@2\"]},{\"op\":\"rest\",\"base\":null}]}}" "{\"tiling\":\"partition\"}" == "{\"accept\":false,\"code\":\"input-held\",\"count\":13}"
#guard run check synth "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"members\",\"path\":\".\",\"sel\":\"@1\",\"k\":1},{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":1},{\"op\":\"members\",\"path\":\".\",\"sel\":\"@3\",\"k\":1},{\"op\":\"instances\",\"path\":\"//TScanStep_v1\"}]}}" "{\"tiling\":\"partition\",\"max_gates\":2}" == "{\"accept\":true,\"count\":13}"

/-! Labelled by the root's table, the six coordinates are one label of 24 gates; the root's members by `[_, 0, 0, 1, 1]`
are two labels, and a zip of nodes labelled apart is refused as `query_ast` refuses it. -/
#guard run check gemm "{\"query\":{\"op\":\"label\",\"map\":{\"table\":\"u\"},\"base\":{\"op\":\"instances\",\"path\":\"@2\"}},\"tables\":{\"u\":{\"Root[TGemm_v1{K=32,N=6}]_v1\":[null,null,0]}}}" "{\"tiling\":\"partition\",\"max_gates\":24}" == "{\"accept\":true,\"count\":1}"
#guard run check gemm "{\"query\":{\"op\":\"label\",\"map\":{\"table\":\"u\"},\"base\":{\"op\":\"instances\",\"path\":\"@2\"}},\"tables\":{\"u\":{\"Root[TGemm_v1{K=32,N=6}]_v1\":[null,null,0]}}}" "{\"max_gates\":23}" == "{\"accept\":false,\"code\":\"too-large\",\"count\":1}"
#guard run check synth "{\"query\":{\"op\":\"label\",\"map\":{\"table\":\"u\"},\"base\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@*\",\"k\":1}},\"tables\":{\"u\":{\"Root[TSynth_v1{N=4}]_v1\":[null,0,0,1,1]}}}" "{\"max_count\":1}" == "{\"accept\":false,\"code\":\"too-many\",\"count\":2}"
#guard run check synth "{\"query\":{\"op\":\"label\",\"map\":{\"table\":\"u\"},\"base\":{\"op\":\"zip\",\"path\":\".\",\"sels\":[\"@1\",\"@2\"]}},\"tables\":{\"u\":{\"Root[TSynth_v1{N=4}]_v1\":[null,0,1,null,null]}}}" "{}" == "reject plan"

/-! `nest` batches `TGemm{K=16,N=4}` over three rows, so `members(".", "@2", k=(2, 3))` is two by two rectangles: rows
`0, 1` by coordinates `0..2` (18 gates), by coordinate `3` (6), and row `2` by each (9, 3). They partition the computed
gates, the `TDot` calls inside them overlap them, and the Input batches, primitives, have one axis. -/
#guard same nest "{\"query\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[2,3]}}" meanings
#guard same nest "{\"query\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[5,7]}}" meanings
#guard same nest "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[2,2]},{\"op\":\"rest\",\"base\":null}]}}" meanings
#guard same nest "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[2,2]},{\"op\":\"instances\",\"path\":\"//TDot_v1\"}]}}" meanings
#guard same nest "{\"query\":{\"op\":\"group\",\"key\":\"definition\",\"base\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[1,2]}}}" meanings
#guard same nest "{\"query\":{\"op\":\"label\",\"map\":{\"affine\":[1,0,2]},\"base\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[2,3]}}}" meanings
#guard same nest "{\"query\":{\"op\":\"label\",\"map\":{\"table\":\"u\"},\"base\":{\"op\":\"concat\",\"parts\":[{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[3,1]},{\"op\":\"rest\",\"base\":null}]}},\"tables\":{\"u\":{\"Root[TNest_v1]_v1\":[null,1,0]}}}" meanings
#guard run check nest "{\"query\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[2,3]}}" "{\"tiling\":\"partition\",\"max_gates\":18}" == "{\"accept\":true,\"count\":4}"
#guard run check nest "{\"query\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[2,3]}}" "{\"max_gates\":17}" == "{\"accept\":false,\"code\":\"too-large\",\"count\":4}"
#guard run check nest "{\"query\":{\"op\":\"concat\",\"parts\":[{\"op\":\"members\",\"path\":\".\",\"sel\":\"@2\",\"k\":[2,2]},{\"op\":\"instances\",\"path\":\"//TDot_v1\"}]}}" "{\"tiling\":\"partition\"}" == "{\"accept\":false,\"code\":\"overlap\",\"count\":16}"
#guard run check nest "{\"query\":{\"op\":\"members\",\"path\":\".\",\"sel\":\"@*\",\"k\":[2,3]}}" "{}" == "reject plan"

/-- `expand` of the six coordinates: the last input gate in no label, the first and last coordinate gates in labels 0 and 5. -/
def coords : Except String (Array (Array Nat)) := do
  expand (← Flock.Program.decode gemm.toUTF8) (← Desc.ofJson (← Json.parse "{\"query\":{\"op\":\"instances\",\"path\":\"@2\"}}"))

#guard (coords.toOption.map fun e => (e.size, e[223]!, e[224]!, e[227]!, e[228]!, e[247]!)) ==
  some (248, #[], #[0], #[0], #[1], #[5])

end Flock.Test.Annot
