import PouwCheck.Protocol

/-!
# A literal PoUW window that PoUW's check accepts

A served window of 4 units, `k = 4` drawn (`subset`), its registration record naming the program of
`difftest_vectors.hand()` (the policy's hand-worked declaration). `pouw_accepts` evaluates PoUW's check on its decoded
values in the kernel. The decoders read bytes through `Lean.Json.parse`, so the bytes are tied to these values by the
`#guard`s below, evaluated and not proved.
-/

namespace Properties.Remote.SecurityProofs.Reach.Pouw

open Lean (Json)
open PouwCheck

/-- The window's registration record, canonical JSON. -/
def recordText : String :=
  "{\"anchors\":[],\"format\":\"verity/registration/v2\",\"law\":{\"k\":4,\"law\":\"subset\"},\"leaf_layer\":null,\"partition\":\"8624a71caafa48bad4c5d72801f15870303f2505d6796f6ede09a10c8fa1fbebc1d6cb0926b5388e8354605ab3b10e3af6f3cb294c2fcf7734d7464f4a647aba\",\"population\":4,\"previous\":null,\"program\":\"fb7bcd5a9ffd6149155dd1e0bfb4ca1fa023c06697ecba274b265e51ae1bddd2ce8811694cddc9631dbbcdb5a931efe79d5487a7708890a994fe91088ebb0429\",\"protocol\":\"verity/sampled-proofs/one-stage/v1\",\"query\":\"0f93d0dce725a5269061b91a1eb39ad294a1369c6f47814880acc74cef30aad734b0067d1cc7f5347bf9db4279a6519eb0abfa000c581759b14fd0ed5a7f9647\",\"roots\":[{\"domain\":\"07af5854237d241484a6773e36f1b6a9919590664dda62d76e2fbdfc6f09114a439d3f6dbadb56b294fdd311d73416cb87030c7aa77b5502d84c0cd038a71d2e\",\"leaves\":6,\"port\":\"entries\",\"root\":\"361de02ae046de888da779d03dc002d004fe04bd9c299e4c31a98a91c2b5729392be60939c461715d52e28c50bff7b08f9c84b96b15d4b4abc1bb33a4437d549\"},{\"domain\":\"c325b4298407cab36efdcafdd48f4517f1ba04dd28b8f90f62fbdd21be2568a9dd818842e8235b0ce7be4f8dc763504da5ef4225cfa376622385510e831c8a31\",\"leaves\":934388,\"port\":\"W\",\"root\":\"f32cd6859e04c9b093e8d7f14e011ffe55a7bde1975f01785247f75e3b0ce99965e53000905adb8ab63059c106ab12c8845a5c2037265086a9bf680250ba477c\"}],\"scheme\":\"frame-v3-sha512/hm96-sha512\",\"window\":{\"context\":\"reach/pouw-window\",\"kind\":\"served\",\"public\":{\"check\":{\"coins\":\"seeded\",\"entries\":6,\"scheme\":\"frame-v3-sha512/hm96-sha512\",\"template\":\"Pc8TileLeafH3_v1{K=4096,DEV='sm120'}\",\"tile_work\":4096},\"length\":{\"max_tokens\":[1],\"served\":[1],\"stop\":\"fixed\"},\"root_W\":\"f32cd6859e04c9b093e8d7f14e011ffe55a7bde1975f01785247f75e3b0ce99965e53000905adb8ab63059c106ab12c8845a5c2037265086a9bf680250ba477c\",\"sampling\":{\"delta\":\"1/10\",\"eps\":\"2/3\",\"grid_bits\":0,\"law\":{\"k\":4,\"law\":\"subset\"}},\"work\":{\"credited\":8192,\"credited_tiles\":2,\"level\":1,\"n\":4}}}}"

/-- `registration.digest` of `recordText`, lowercase hex. -/
def recordHex : String :=
  "1f727f7b54a159b6cb5d2a2090ea3686160091fe9380750777db0e9dfc1f3fbf7ae8778b80525cfb6cd7497dfb4cf007e2e1a52b570f044faad2949fd04cb9cd"

def programHex : String :=
  "fb7bcd5a9ffd6149155dd1e0bfb4ca1fa023c06697ecba274b265e51ae1bddd2ce8811694cddc9631dbbcdb5a931efe79d5487a7708890a994fe91088ebb0429"

def queryHex : String :=
  "0f93d0dce725a5269061b91a1eb39ad294a1369c6f47814880acc74cef30aad734b0067d1cc7f5347bf9db4279a6519eb0abfa000c581759b14fd0ed5a7f9647"

def partitionHex : String :=
  "8624a71caafa48bad4c5d72801f15870303f2505d6796f6ede09a10c8fa1fbebc1d6cb0926b5388e8354605ab3b10e3af6f3cb294c2fcf7734d7464f4a647aba"

def rootW : String :=
  "f32cd6859e04c9b093e8d7f14e011ffe55a7bde1975f01785247f75e3b0ce99965e53000905adb8ab63059c106ab12c8845a5c2037265086a9bf680250ba477c"

def items : Json :=
  Json.mkObj [
    ("check", Json.mkObj [("coins", .str "seeded"), ("entries", .num 6), ("scheme", .str "frame-v3-sha512/hm96-sha512"),
      ("template", .str "Pc8TileLeafH3_v1{K=4096,DEV='sm120'}"), ("tile_work", .num 4096)]),
    ("length", Json.mkObj [("max_tokens", .arr #[.num 1]), ("served", .arr #[.num 1]), ("stop", .str "fixed")]),
    ("root_W", .str rootW),
    ("sampling", Json.mkObj [("delta", .str "1/10"), ("eps", .str "2/3"), ("grid_bits", .num 0),
      ("law", Json.mkObj [("k", .num 4), ("law", .str "subset")])]),
    ("work", Json.mkObj [("credited", .num 8192), ("credited_tiles", .num 2), ("level", .num 1), ("n", .num 4)])]

def params : Params where
  check := { entries := 6, scheme := "frame-v3-sha512/hm96-sha512", template := "Pc8TileLeafH3_v1{K=4096,DEV='sm120'}",
              tileWork := 4096, coins := some "seeded" }
  context := "reach/pouw-window"
  delta := ⟨1, 10⟩
  eps := ⟨2, 3⟩
  gridBits := 0
  query := queryHex
  lo := 1
  hi := 7
  maxTokens := [1]
  live := none

def registration : Registration where
  bytes := recordText.toUTF8
  format := "verity/registration/v2"
  program := programHex
  query := queryHex
  partition := partitionHex
  scheme := "frame-v3-sha512/hm96-sha512"
  population := 4
  law := Json.mkObj [("k", .num 4), ("law", .str "subset")]
  roots := #[
    ⟨"entries",
      "361de02ae046de888da779d03dc002d004fe04bd9c299e4c31a98a91c2b5729392be60939c461715d52e28c50bff7b08f9c84b96b15d4b4abc1bb33a4437d549",
      6,
      "07af5854237d241484a6773e36f1b6a9919590664dda62d76e2fbdfc6f09114a439d3f6dbadb56b294fdd311d73416cb87030c7aa77b5502d84c0cd038a71d2e"⟩,
    ⟨"W", rootW, 934388,
      "c325b4298407cab36efdcafdd48f4517f1ba04dd28b8f90f62fbdd21be2568a9dd818842e8235b0ce7be4f8dc763504da5ef4225cfa376622385510e831c8a31"⟩]
  window := { kind := "served", context := "reach/pouw-window", items, activations := none, registered := none }

def profile : Core.Integrity where
  programs := #[programHex]
  prescribed := #[]
  linked := true
  covered := #[]
  levels := #[{ name := "unit", query := some partitionHex, size := 4, draw := some (.subset 4) }]
  delta := ⟨1, 10⟩
  drawn := #[("unit", 4)]

def evidence : Evidence where
  protocol := "verity/sampled-proofs/one-stage/v1"
  commitment := some recordHex
  said := .accept profile { law := .subset 4, population := 4, units := #[0, 1, 2, 3] }
  salt := none

/-- The verifier's parameters, canonical JSON. -/
def paramsText : String :=
  "{\"check\":{\"coins\":\"seeded\",\"entries\":6,\"scheme\":\"frame-v3-sha512/hm96-sha512\",\"template\":\"Pc8TileLeafH3_v1{K=4096,DEV='sm120'}\",\"tile_work\":4096},\"context\":\"reach/pouw-window\",\"delta\":\"1/10\",\"eps\":\"2/3\",\"grid_bits\":0,\"length\":{\"max_tokens\":[1],\"stop\":\"fixed\"},\"query\":\"0f93d0dce725a5269061b91a1eb39ad294a1369c6f47814880acc74cef30aad734b0067d1cc7f5347bf9db4279a6519eb0abfa000c581759b14fd0ed5a7f9647\",\"range\":[1,7]}"

/-- The service's answer, canonical JSON. -/
def answerText : String :=
  "{\"commitment\":\"1f727f7b54a159b6cb5d2a2090ea3686160091fe9380750777db0e9dfc1f3fbf7ae8778b80525cfb6cd7497dfb4cf007e2e1a52b570f044faad2949fd04cb9cd\",\"profile\":{\"covered\":[],\"delta\":\"1/10\",\"drawn\":{\"unit\":4},\"levels\":[{\"draw\":{\"subset\":4},\"name\":\"unit\",\"query\":\"8624a71caafa48bad4c5d72801f15870303f2505d6796f6ede09a10c8fa1fbebc1d6cb0926b5388e8354605ab3b10e3af6f3cb294c2fcf7734d7464f4a647aba\",\"size\":4}],\"linked\":true,\"prescribed\":[],\"programs\":[\"fb7bcd5a9ffd6149155dd1e0bfb4ca1fa023c06697ecba274b265e51ae1bddd2ce8811694cddc9631dbbcdb5a931efe79d5487a7708890a994fe91088ebb0429\"]},\"protocol\":\"verity/sampled-proofs/one-stage/v1\",\"public\":{\"draw\":{\"k\":4,\"law\":\"subset\",\"population\":4,\"units\":[0,1,2,3]}},\"reason\":null,\"verdict\":\"accept\"}"

#guard Flock.hexOf registration.digest == recordHex
#guard match PouwCheck.registrationOf recordText.toUTF8 with
  | .ok r => r.digest == registration.digest && r.program == registration.program && r.population == 4
  | .error _ => false
#guard match protocol.answer paramsText.toUTF8 recordText.toUTF8 answerText.toUTF8 with
  | { verdict := .accept, digest := some d, .. } => d == registration.digest
  | _ => false

/-- **PoUW's check accepts the literal window**: a draw of `k = 4` of its 4 units. -/
theorem pouw_accepts : protocol.Accepts params registration evidence := by
  have h : (match check params registration evidence with | .accept _ _ => true | _ => false) = true := by
    decide +kernel
  show ∃ pub prof, check params registration evidence = .accept pub prof
  revert h
  generalize check params registration evidence = o
  cases o with
  | accept pub prof => exact fun _ => ⟨pub, prof, rfl⟩
  | reject _ => simp
  | undecided _ => simp

end Properties.Remote.SecurityProofs.Reach.Pouw
