module

public import Spot.Test.Docket
public import Spot.OneStageZk
public import Spot.ChallengerCoins
public import Spot.ChallengerTrees
meta import Spot.Test.Docket
meta import Spot.OneStageZk
meta import Spot.ChallengerCoins
meta import Spot.ChallengerTrees

/-!
The one-statement audit's, the challenger's and the draw's named refusals, each on the smallest input that reaches it: the
docket's verifier (`Spot.Test.Docket`), and one honest parameter object over the same program changed in one place.
-/

@[expose] public section

namespace Spot.Test.Refusals

open Lean (Json)
open Flock (V)
open Spot.Test.Docket (V3 QE ROOT program)

/-- The refusal `r` gives, `ok` when it gives none. -/
def why? {α : Type} (r : V α) : String := match r with
  | .ok _ => "ok"
  | .error e => e

/-- A challenger key at `W` and `H`, its public key 80 bytes. -/
def key (W H : Nat) : Json :=
  Json.mkObj [("H", H), ("W", W), ("public", String.ofList (List.replicate 160 'a'))]

/-- The one statement's parameters over `E3` under the template query, sampled `law`, the challenger's key `k`. -/
def params (law : Json := Json.mkObj [("law", "subset"), ("k", (1 : Nat))]) (k : Json := key 1 20) : Json :=
  let p := program Spot.OneStageZk.E3
  Json.mkObj [("anchors", Json.arr #[]), ("delta", "1/1000"), ("law", law), ("leaf_layer", Json.null),
    ("log", Json.arr #[]), ("partition", Partition.build (Flock.hexOf p.sha512) QE),
    ("ports", Json.mkObj [("o", Json.mkObj [("domain", "d"), ("leaves", (4 : Nat))])]),
    ("program", (Flock.JsonParse.parse Spot.OneStageZk.E3).toOption.getD Json.null), ("query", QE), ("registered", Json.null),
    ("scheme", "s"), ("work_table", Json.null), ("challenger", Json.mkObj [("key", k), ("label", "draw")]),
    ("window_root", ROOT), ("proofs", Json.mkObj [("public", ROOT), ("registered", Json.null)])]

#guard why? (OneStageZk.paramsOfJson params) == "ok"
#guard why? (OneStageZk.paramsOfJson (Spot.Test.Docket.put params "challenger" none)) == OneStageZk.NO_CHALLENGER
#guard why? (OneStageZk.paramsOfJson (params (k := key 2 5))) == OneStageZk.KEY_PARAMS
#guard why? (OneStageZk.paramsOfJson (params (Json.mkObj [("law", "stratified"), ("k", (1 : Nat))]))) ==
  OneStageZk.SUBSET_ONLY "stratified"

/-- The challenger session's release log after its head: the family `draw`'s root `ab` recorded, a coin tree's round
opened on the link's stream, then the coin released: a coin of the sample released after a round opened. -/
def lateRelease : Json := Json.arr #["head",
  "{\"event\":\"root\",\"index\":0,\"label\":\"draw\",\"received_at\":0,\"root\":\"ab\"}",
  "{\"after\":\"cd\",\"coins\":\"ef\",\"event\":\"tree-open\",\"n\":1,\"round\":0,\"stream\":\"" ++
    Challenger.TREE_SESSION ++ "\",\"tree\":\"01\"}",
  "{\"after\":\"ab\",\"event\":\"release\",\"index\":0,\"label\":\"draw\",\"released_at\":0}"]

/-- Evidence with no session but the challenger session's log `log`. -/
def evidence (log : Json) : OneStageZk.Evidence :=
  { base := { receipt := .null, draw := .null, provedUnits := #[], instances := 0, statement := .null, verdicts := #[],
              sessions := #[], served := none, sessionRoots := none,
              coins := Json.mkObj [("session", Json.mkObj [("log", log)])] },
    sessions := #[], statements := none }

#guard why? (do OneStageZk.orderHeld (← OneStageZk.paramsParsed params) ⟨.null, ⟨#[0xab]⟩⟩ (evidence lateRelease)) ==
  s!"coins: {Challenger.SAMPLE_LINK_FIRST}"

-- no coin opened, so the family's roots are all the log holds of it
#guard why? (Challenger.challengerStream ⟨"s", 0, [], ByteArray.empty⟩
  (Json.mkObj [("after", "ab"), ("label", "draw"), ("openings", Json.arr #[]), ("start", (0 : Nat))]) ⟨#[0xab]⟩
  (some [{ kind := .recorded, label := "draw", index := 0, root := ⟨#[0xcd]⟩, time := 0, pos := 1 }])) ==
  Challenger.FAMILY_ROOT

#guard why? ((Draw.take ByteArray.empty 1).run' 0) == Draw.RAN_OUT

#guard why? (OneStageZk.popHeld V3 (Json.mkObj [("population", (2 : Nat))])) == OneStageZk.POPULATION_ORDER

#guard why? (OneStageZk.heldReadsOf { V3 with query := Json.mkObj [("name", "Q_unit"), ("params", Json.mkObj []),
  ("version", (0 : Nat))] }) == OneStageZk.HELD_QUERY_WHY

/-- A circuit of `upv` units, each returning `leaves` output leaves. -/
def circuit (upv leaves : Nat) : Flock.Circuit :=
  { name := "c", sha := ByteArray.empty, nets := #[], ranges := #[], ports := #[], outPorts := #[], g := 0, kLog := 0,
    upv, unitNet := 0, leavesIn := #[], leavesOut := (List.replicate upv (List.range leaves).toArray).toArray,
    key := ByteArray.empty, dummyDigests := #[], dummyOutputs := #[], leafScheme := "", perVu := #[], wires := #[] }

#guard why? (OneStageZk.templateOk (circuit 1 1)) == "ok"
#guard why? (OneStageZk.templateOk (circuit 0 1)) == OneStageZk.NO_ROW_WHY
-- 16 bits a leaf: 16385 leaves are one past the widest size class
#guard why? (OneStageZk.templateOk (circuit 1 16385)) == OneStageZk.WIDE_WHY

end Spot.Test.Refusals
