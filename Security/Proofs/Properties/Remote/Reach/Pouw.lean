import PouwCheck.Protocol
import Proofs.Pouw.Period.Accept

/-!
# A literal PoUW window that PoUW's check accepts

A served window on the 0-bit grid at level 1 (2 credited tiles, padded to `N = 4`), at ε = 2/3 and δ = 1/10, so that
`K′ = 4` (`(2/3)^4 > 1/10`, capped at `N`), drawn `subset:4`; its registration record names the program of
`difftest_vectors.hand()` (the policy's hand-worked declaration). `pouw_accepts` is PoUW's check accepting its decoded
values, by `Properties.Pouw.Check.acceptMeans_complete`: the check sorts keys and writes canonical JSON by well-founded
recursion, which the kernel does not unfold, so each conjunct of `RegistrationKeeps` is its own lemma, the sorts by
`Proofs.Pouw.Period.Accept`'s `keys_of` and `sortedKeys_of`. The decoders read bytes through `Lean.Json.parse`, so the
bytes are tied to these values by the `#guard`s below, evaluated and not proved.
-/

namespace Properties.Remote.SecurityProofs.Reach.Pouw

open Lean (Json)
open PouwCheck
open PouwWindow (Work)
open Properties.Pouw.Period.SecurityProofs.Accept (keys_of sortedKeys_of registeredSorted_eq)

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

/-- What the window fixes: level 1 of the 0-bit grid (2 credited tiles, padded to 4), `K′ = 4`, and its `W` root. -/
def fixed : Fixed := ⟨Work.at 0 1, ⟨toRat ⟨2, 3⟩, toRat ⟨1, 10⟩, 0, 4⟩, some rootW⟩

/-- Its registered items: the verifier's own at level 1 and the fixed length's. -/
def items : Json :=
  Json.mkObj [("check", params.check.toJson), ("length", PouwWindow.lengthItem [1]), ("root_W", rootWJson fixed.rootW),
    ("sampling", fixed.sampling.toJson), ("work", fixed.work.itemJson params.check.tileWork)]

def registration : Registration where
  bytes := recordText.toUTF8
  format := "verity/registration/v2"
  program := programHex
  query := queryHex
  partition := partitionHex
  scheme := "frame-v3-sha512/hm96-sha512"
  population := 4
  law := subsetLaw 4
  roots := #[
    ⟨"entries",
      "361de02ae046de888da779d03dc002d004fe04bd9c299e4c31a98a91c2b5729392be60939c461715d52e28c50bff7b08f9c84b96b15d4b4abc1bb33a4437d549",
      6,
      "07af5854237d241484a6773e36f1b6a9919590664dda62d76e2fbdfc6f09114a439d3f6dbadb56b294fdd311d73416cb87030c7aa77b5502d84c0cd038a71d2e"⟩,
    ⟨"W", rootW, 934388,
      "c325b4298407cab36efdcafdd48f4517f1ba04dd28b8f90f62fbdd21be2568a9dd818842e8235b0ce7be4f8dc763504da5ef4225cfa376622385510e831c8a31"⟩]
  window := ⟨"served", "reach/pouw-window", items, none, none⟩

/-- The profile the rule fixes: the registered Program over one level, the partition's 4 units drawn `subset:4`. -/
def profile : Core.Integrity where
  programs := #[programHex]
  prescribed := #[]
  linked := true
  covered := #[]
  levels := #[{ name := unitLevel, query := some partitionHex, size := fixed.work.n,
                draw := some (.subset fixed.sampling.k) }]
  delta := ⟨1, 10⟩
  drawn := #[(unitLevel, fixed.sampling.k)]

/-- The service's draw: all 4 units. -/
def draw : UnitDraw := ⟨.subset fixed.sampling.k, fixed.work.n, #[0, 1, 2, 3]⟩

def evidence : Evidence where
  protocol := oneStage
  commitment := some (Flock.hexOf registration.digest)
  said := .accept profile draw

/-- The verifier's parameters, canonical JSON. -/
def paramsText : String :=
  "{\"check\":{\"coins\":\"seeded\",\"entries\":6,\"scheme\":\"frame-v3-sha512/hm96-sha512\",\"template\":\"Pc8TileLeafH3_v1{K=4096,DEV='sm120'}\",\"tile_work\":4096},\"context\":\"reach/pouw-window\",\"delta\":\"1/10\",\"eps\":\"2/3\",\"grid_bits\":0,\"length\":{\"max_tokens\":[1],\"stop\":\"fixed\"},\"query\":\"0f93d0dce725a5269061b91a1eb39ad294a1369c6f47814880acc74cef30aad734b0067d1cc7f5347bf9db4279a6519eb0abfa000c581759b14fd0ed5a7f9647\",\"range\":[1,7]}"

/-- The service's answer, canonical JSON. -/
def answerText : String :=
  "{\"commitment\":\"1f727f7b54a159b6cb5d2a2090ea3686160091fe9380750777db0e9dfc1f3fbf7ae8778b80525cfb6cd7497dfb4cf007e2e1a52b570f044faad2949fd04cb9cd\",\"profile\":{\"covered\":[],\"delta\":\"1/10\",\"drawn\":{\"unit\":4},\"levels\":[{\"draw\":{\"subset\":4},\"name\":\"unit\",\"query\":\"8624a71caafa48bad4c5d72801f15870303f2505d6796f6ede09a10c8fa1fbebc1d6cb0926b5388e8354605ab3b10e3af6f3cb294c2fcf7734d7464f4a647aba\",\"size\":4}],\"linked\":true,\"prescribed\":[],\"programs\":[\"fb7bcd5a9ffd6149155dd1e0bfb4ca1fa023c06697ecba274b265e51ae1bddd2ce8811694cddc9631dbbcdb5a931efe79d5487a7708890a994fe91088ebb0429\"]},\"protocol\":\"verity/sampled-proofs/one-stage/v1\",\"public\":{\"draw\":{\"k\":4,\"law\":\"subset\",\"population\":4,\"units\":[0,1,2,3]}},\"reason\":null,\"verdict\":\"accept\"}"

/-- What the check publishes on it. -/
def published : Public := ⟨params.check, fixed.work, fixed.sampling, params.maxTokens, fixed.rootW, draw.units⟩

#guard Flock.hexOf registration.digest == recordHex
#guard match PouwCheck.registrationOf recordText.toUTF8 with
  | .ok r => r.digest == registration.digest && r.format == registration.format && r.program == registration.program &&
      r.query == registration.query && r.partition == registration.partition && r.scheme == registration.scheme &&
      r.population == registration.population && cs r.law == cs registration.law && r.roots == registration.roots &&
      r.window.kind == registration.window.kind && r.window.context == registration.window.context &&
      cs r.window.items == cs registration.window.items && r.window.activations.isNone && r.window.registered.isNone
  | .error _ => false
#guard match protocol.answer paramsText.toUTF8 recordText.toUTF8 answerText.toUTF8 with
  | { verdict := .accept, digest := some d, published := some j, profile := some q, .. } =>
    d == registration.digest && j == protocol.publicJson published && q == protocol.profileJson profile
  | _ => false

/-! ## The registration keeps PoUW's rule -/

theorem ok_of_match {α ε : Type} [DecidableEq α] {x : Except ε α} {a : α}
    (h : (match x with | .ok b => decide (b = a) | .error _ => false) = true) : x = .ok a := by
  cases x <;> simp_all

theorem keys_items : keys items = some registeredSorted := by
  rw [registeredSorted_eq]
  exact keys_of (by decide) (by decide +kernel)

theorem length_items : PouwWindow.lengthProblem (get items "length") [1] = none := by
  have hg : get items "length" = PouwWindow.lengthItem [1] := rfl
  have hs : PouwWindow.sortedKeys (PouwWindow.lengthItem [1]) = some PouwWindow.lengthKeys :=
    sortedKeys_of (by decide) (by decide +kernel)
  rw [hg]
  unfold PouwWindow.lengthProblem
  rw [hs]
  decide +kernel

theorem level_items : levelOf items = some 1 := by decide +kernel

theorem grid : 1 + 1 ≤ PouwWindow.level 0 (PouwWindow.floorPoint 0 7) ∧ 1 ≤ PouwWindow.point 0 1 := by decide

/-- **`K′ = 4`**: `p = (2/3)·(2·4096)/(4·4096) = 1/3`, and `(2/3)^4 = 16/81 > 1/10`, so no `K ≤ N = 4` reaches δ and
`K′` is `N`; the search's bisection is not reached. -/
theorem sampling_ok : PouwWindow.sampling (toRat params.eps) (toRat params.delta) fixed.work
    (params.check.tileWork : Int) params.gridBits = .ok fixed.sampling :=
  ok_of_match (by decide +kernel)

theorem rootW_ok : fixed.rootW = (registration.roots.find? (·.port = "W")).map (·.root) := by decide +kernel

theorem cs_check : cs (get items "check") = cs params.check.toJson := by
  first
  | rw [show get items "check" = params.check.toJson from rfl]
  | decide +kernel

theorem cs_work : cs (get items "work") = cs (fixed.work.itemJson params.check.tileWork) := by
  first
  | rw [show get items "work" = fixed.work.itemJson params.check.tileWork from rfl]
  | decide +kernel

theorem cs_rootW : cs (get items "root_W") = cs (rootWJson fixed.rootW) := by
  first
  | rw [show get items "root_W" = rootWJson fixed.rootW from rfl]
  | decide +kernel

theorem cs_sampling : cs (get items "sampling") = cs fixed.sampling.toJson := by
  first
  | rw [show get items "sampling" = fixed.sampling.toJson from rfl]
  | decide +kernel

theorem keeps : Properties.Pouw.Check.RegistrationKeeps params registration fixed := by
  refine ⟨⟨rfl, rfl⟩, congrArg (fun o => o.getD []) keys_items, length_items, level_items, grid.1, grid.2, rfl,
    sampling_ok, rootW_ok, ?_, rfl, rfl, rfl, rfl⟩
  intro kv hkv
  simp only [expected, List.mem_cons] at hkv
  rcases hkv with rfl | rfl | rfl | rfl | h
  · exact cs_check
  · exact cs_rootW
  · exact cs_sampling
  · exact cs_work
  · cases h

theorem serviceAccepted : Properties.Pouw.Check.ServiceAccepted params registration fixed evidence profile draw :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, by decide +kernel, by decide +kernel⟩

theorem saltKept : Properties.Pouw.Check.SaltKept params registration evidence := by decide

/-- **PoUW's check accepts the literal window**, a draw of `K′ = 4` of its 4 units, publishing level 1's work, the
sampling item at `K′ = 4`, its `W` root and the 4 units, with the profile the rule fixes. -/
theorem pouw_check : protocol.check params registration evidence = .accept published profile :=
  Properties.Pouw.Check.acceptMeans_complete _ _ _ fixed _ draw keeps serviceAccepted saltKept

theorem pouw_accepts : protocol.Accepts params registration evidence := ⟨_, _, pouw_check⟩

end Properties.Remote.SecurityProofs.Reach.Pouw
