/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
components are integrity by sampled proofs' one-stage audit and work by proof of useful work; each is a stub
(`Core.Composition.Stub`) until its port lands. `Core.Composition.product` composes them: both accept or remote doesn't,
and either one's rejection rejects remote whatever the other answers.
-/
import CoreComposition.Cases
import CoreComposition.Stubs

namespace Core.Remote

open Composition

/-- Remote's components, in the order of their ids. -/
def components : List Protocol := [Stub.pouw, Stub.sampledProofs]

def protocol : Protocol := product "verity/remote/v0" components

open Cases in
private def answer (xs : List Stubbed) (ps : List Stubbed := xs) : Answer := Cases.answer protocol xs ps

private def canonical (a : Answer) : String :=
  match a.canonical with
  | .ok s => s
  | .error e => s!"error: {e}"

private def work : String := Stub.pouw.id
private def integrity : String := Stub.sampledProofs.id

#guard ordered components
#guard components.map (·.id) == ["verity/stub/pouw/v0", "verity/stub/sampled-proofs/v0"]
#guard canonical (answer [Cases.accepts work, Cases.accepts integrity]) ==
  "{\"commitment\":\"" ++ Flock.hexOf (Cases.commitmentOf protocol [Cases.accepts work, Cases.accepts integrity]) ++
  "\",\"profile\":{" ++ ",".intercalate ([work, integrity].map fun i =>
    "\"" ++ i ++ "\":{\"covered\":[],\"delta\":\"1/1000000\",\"drawn\":{\"unit\":2},\"levels\":[{\"draw\":{\"subset\":2}," ++
    "\"name\":\"unit\",\"query\":null,\"size\":2}],\"linked\":true,\"prescribed\":[],\"programs\":[\"" ++
    Flock.hexOf (Cases.accepts i).root ++ "\"]}") ++
  "},\"protocol\":\"verity/remote/v0\",\"public\":{\"" ++ work ++ "\":{\"passed\":2},\"" ++ integrity ++
  "\":{\"passed\":2}},\"reason\":null,\"verdict\":\"accept\"}"
-- work rejected while integrity is undecided: remote rejects
#guard canonical (answer [Cases.rejects work, Cases.undecided integrity]) ==
  "{\"commitment\":\"" ++ Flock.hexOf (Cases.commitmentOf protocol [Cases.rejects work, Cases.undecided integrity]) ++
  "\",\"profile\":null,\"protocol\":\"verity/remote/v0\",\"public\":null," ++
  "\"reason\":\"verity/stub/pouw/v0 reject: 1 of 2 drawn units failed\",\"verdict\":\"reject\"}"
#guard (answer [Cases.accepts work, Cases.undecided integrity]).verdict == .undecided
#guard (answer [Cases.accepts work, Cases.rejects integrity]).verdict == .reject
#guard (answer [Cases.accepts work]).reason == some "params: verity/stub/sampled-proofs/v0 is missing"

end Core.Remote
