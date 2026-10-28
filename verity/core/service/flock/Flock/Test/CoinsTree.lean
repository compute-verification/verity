/-
What a statement keeps without parsing its own output. The keyed coin tree a session reads (`Stmt.coinTree`,
`SessionSpec.coinTree`) is the setup's coins' (`Coins.tree?`), which is what the setup's `Hello` names
(`CoinTree.ofHello (helloOf …)`): the tree's spec for `.tree`, none for the OS's coins or a seed's. The public header is
kept as `Flock.norm` of what `JsonParse.parse` read, which has the same value at every key, the same `unit_draw` once `drawn`
puts the draw in, and the same canonical bytes (`publicDigest` hashes those).
-/
import Flock.HmRow

namespace Flock.Test.CoinsTree

open Lean (Json)

def specs : List CoinTree.Spec :=
  [⟨#["rep0", "rep1"], 3, 64⟩, ⟨#[], 0, 1⟩, ⟨#["a\"b", "u~i", "x"], 20, 65536⟩]

def coinsAll : List Coins := [.os, .seed] ++ specs.map .tree

def helloAt (t : Tags) (c : Coins) (tables : Array String) : String :=
  (helloOf t (ByteArray.mk #[1, 2, 3]) 10 c tables).toOption.getD ""

#guard coinsAll.all fun c => (helloOf Tags.circuitPlainLeaves ByteArray.empty 10 c).toOption.isSome
-- a `Hello` is canonical JSON, so a name outside ASCII without DEL, a rep's or a table's, has none
#guard (helloOf Tags.circuitPlainLeaves ByteArray.empty 10 (.tree ⟨#["ünï"], 3, 64⟩)).toOption.isNone
#guard (helloOf Tags.circuitPlainLeaves ByteArray.empty 10 .os #["dé"]).toOption.isNone
#guard Tags.all.all fun t => coinsAll.all fun c =>
  [#[t.table], #["t0", "t1", "t2"]].all fun tabs => CoinTree.ofHello (helloAt t c tabs) == c.tree?

def headers : List String :=
  ["{\"format\":\"f\",\"statement\":\"s\",\"instances\":4,\"rows\":false,\"units\":{\"indices\":[0,1,2,3]}}",
   "{\"units\":{\"indices\":[3,2]},\"rows\":false,\"instances\":2,\"statement\":\"s\",\"format\":\"f\",\"z\":{\"b\":1,\"a\":[{\"y\":0,\"x\":null}]}}",
   "{\"m\":1,\"c\":2,\"x\":3,\"a\":4,\"q\":5,\"b\":6,\"k\":7,\"d\":8,\"unit_draw\":{\"law\":\"subset\",\"k\":1}}"]

def keys : List String := ["format", "statement", "instances", "rows", "units", "z", "m", "a", "b", "unit_draw", "none"]

def parsed : List Json := headers.filterMap fun h => (JsonParse.parse h).toOption

#guard parsed.length == headers.length
#guard parsed.all fun j => (canon (Flock.norm j)).toOption == (canon j).toOption && (canon j).toOption.isSome
#guard parsed.all fun j => keys.all fun k => ((Flock.norm j).getObjVal? k).toOption == (j.getObjVal? k).toOption
#guard parsed.all fun j =>
  let d := Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson (4 : Nat))]
  (((Flock.norm j).setObjVal! "unit_draw" d).getObjVal? "unit_draw").toOption == some d &&
    (canon ((Flock.norm j).setObjVal! "unit_draw" d)).toOption == (canon (j.setObjVal! "unit_draw" d)).toOption

end Flock.Test.CoinsTree
