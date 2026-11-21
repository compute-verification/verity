module

public import Properties.Spot.PortsHeld
public import Proofs.Properties.Spot
import all Init.Prelude
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Legacy
import all Init.Data.String.Search
import all Init.Data.String.Slice
import all Init.Data.String.Substring
import all Init.Data.ByteArray.Basic
import all Init.Data.Array.Basic
import all Base.Hash
import all Flock.Registered
import all Spot.OneTree
import all Spot.OneStageZk
import all Base.JsonParse
import all Circuits.Descriptor
import all Std.Data.HashMap.Basic

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot.WitnessProbe

set_option maxRecDepth 100000
set_option maxHeartbeats 0

theorem b1 : ("w" == "w") = true := by decide +kernel
theorem b2 : "abc".toUTF8.size = 3 := by decide +kernel
theorem b5 : (Flock.Sha512.hash ByteArray.empty).size = 64 := by decide +kernel
theorem a2 : Flock.hexOf (ByteArray.mk #[1, 255]) = "01ff" := by decide +kernel

def descT : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,2]},\"ret_type\":[\"Array\",2,[\"Value\",16]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]]]}"

theorem c1 : (Flock.JsonParse.parse descT).toBool = true := by decide +kernel
theorem c4 : (((({} : Std.HashMap String Nat).insert "a" 1).get? "a") = some 1) := by simp

end Properties.Spot.WitnessProbe
