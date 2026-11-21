module

public import Spot.OneStageZk
meta import Spot.OneStageZk

/-!
`Spot.OneStageZk.verdictTreesHeldAt` on a toy C: three units of an `EmbeddingRead_v3` over a `FrameV3Read_v1`, its
verdict field `e/o1`, against records parsed by `Flock.JsonParse.parse`. The roots are `registered_value.public_root`'s
(`tests/verity/protocols/proof/spot/verifier/test_verdict_trees.py` pins the first).
-/

@[expose] public section

namespace Spot.Test.VerdictTrees

open Lean (Json)
open Spot.OneStageZk
open Flock.Program (Prog Node Defn Ty)

def v16 : Ty := .value 16

def call (fn : Nat) : Node := { form := .batch, fn := fn, args := #[], out := v16 }

def composite (id : String) (ret : Ty) (nodes : Array Node) : Defn :=
  { key := id, id := id, prim := false, params := #[("t", v16)], ret := ret, gates := 0, paramLeaves := 1,
    outLeaves := 1, body := { params := #[("t", v16)], retType := ret, nodes := nodes, ret := default, offsets := #[] } }

/-- The input, the reader, `EmbeddingRead_v3` (`(h, ok)`), an unlisted opener `Opens_v1`, and the root: the input node
and a batch of three `EmbeddingRead_v3` units, then `extra`. -/
def toy (extra : Array Node := #[]) : Prog :=
  let input : Defn :=
    { key := "Input16_v1", id := "Input16_v1", prim := true, params := #[], ret := v16, gates := 1, paramLeaves := 0,
      outLeaves := 1, binding := some Flock.Program.RUN }
  let read : Defn :=
    { key := "FrameV3Read_v1", id := "FrameV3Read_v1", prim := true, params := #[("t", v16)], ret := v16, gates := 1,
      paramLeaves := 1, outLeaves := 1 }
  let root : Defn := composite "R" v16
    (#[{ form := .batch, fn := 0, args := #[], out := .array 3 v16, n := 3 },
       { form := .batch, fn := 2, args := #[], out := .array 3 (.tuple #[v16, v16]), n := 3 }] ++ extra)
  { defs := #[input, read, composite "EmbeddingRead_v3" (.tuple #[v16, v16]) #[call 1, call 1],
      composite "Opens_v1" v16 #[call 1], root],
    index := (({} : Std.HashMap String Nat).insert "EmbeddingRead_v3" 2).insert "Opens_v1" 3, root := 4,
    programFn := "", sha512 := .empty }

def rs : Array Flock.TemplateQuery.TRange := #[⟨1, "EmbeddingRead_v3", 0, 3⟩]

def ms : Array Member :=
  #[{ name := "e/c0", units := #[(0, 3)], g := 1,
      own := { programJ := .null, program := .empty, partition := .null, pub := "", registered := none } }]

def PROGRAM : String := String.join (List.replicate 64 "ab")

def ZEROS : String := String.join (List.replicate 64 "00")

/-- The record holding `e/o1` at `root`, over the domain id of 64 zero bytes. -/
def recordAt (root : String) : Json :=
  (Flock.JsonParse.parse s!"\{\"roots\":[\{\"domain\":\"{ZEROS}\",\"leaves\":3,\"port\":\"e/o1\",\"root\":\"{root}\"}]}").toOption.getD .null

/-- `verdictTreesHeldAt` on `P`: none where it accepts, else its refusal. -/
def held (P : Prog) (root : String) : Option String :=
  match verdictTreesHeldAt P PROGRAM rs ms (recordAt root) with | .ok () => none | .error e => some e

/-- Every row 1, under the public salts. -/
def ONES : String :=
  "66ca66f47536e4d509e1b7056ad7f670da8afde628eeb4b57d8b9458e926847dbac5752c21d49774ff7b7714c57a46e0b69cd4249a52429ff12495fd92cfffcd"

/-- Row 1 is 0. -/
def ONE_ZERO : String :=
  "5bf17b863a56c9fed18dd3ca336b6fd560d000f59c69179bd8aed55231046af49d790de0d5d37850db7d25fb33ed5f211d3e2612ba7987b633f92e1e2a1ae050"

#guard verdictRoot PROGRAM "EmbeddingRead/o1" (ByteArray.mk (Array.replicate 64 0)) 3 ==
  "25cabd2e7f8d2d0a8ffd61d69d0fa58ac19d876606f65b72832f5a6fcc23cf3d51c7234eb8f93c1f41f3bc77650fc0a0622895ca1fb36bb5c5ef6cfe60b58a84"
#guard (verdictTrees (toy) rs ms).toOption == some #[("e/o1", 3)]
#guard held (toy) ONES == none
#guard held (toy) ONE_ZERO == some (VERDICT_TREE_WHY "e/o1")
#guard held (toy) ZEROS == some (VERDICT_TREE_WHY "e/o1")
#guard held (toy #[call 3]) ONES == some (UNLISTED_OPENING_WHY "Opens_v1")
#guard opensTree (toy) == #[false, true, true, true, true]

end Spot.Test.VerdictTrees
