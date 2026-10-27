/-
The `#guard`s that pin the composition's verdict rule on a product of two stubs, `a` and `b` (`Core.Example.protocol`,
k = 2): every pair of the components' verdicts, reject dominating; a commitment that doesn't bind their digests; and
every way the cases can fail to fit the composition. Then the same two after a registration check, a third stub `r`
(`registered`, on the registration `b` commits to): its refusal refusing the whole, the commitment binding its digest,
and a registration other than the one `b` committed to refused.
-/
import CoreComposition.Cases
import CoreComposition.Stubs

namespace Core.Composition.Guards

open Cases

private def toy : Protocol := product "verity/toy/v0" [Stub.stub "a", Stub.stub "b"]

private def answer (xs : List Stubbed) (ps : List Stubbed := xs) (c : ByteArray := commitmentOf toy xs) : Answer :=
  Cases.answer toy xs ps c

private def verdict (xs : List Stubbed) (ps : List Stubbed := xs) (c : ByteArray := commitmentOf toy xs) : Verdict :=
  (answer xs ps c).verdict

#guard ordered [Stub.stub "a", Stub.stub "b"]
#guard !ordered [Stub.stub "b", Stub.stub "a"] && !ordered [Stub.stub "a", Stub.stub "a"]

-- every pair of verdicts: accept only when both accept, and a rejection dominates an undecided component
#guard verdict [accepts "a", accepts "b"] == .accept
#guard verdict [accepts "a", rejects "b"] == .reject
#guard verdict [rejects "a", accepts "b"] == .reject
#guard verdict [accepts "a", undecided "b"] == .undecided
#guard verdict [undecided "a", accepts "b"] == .undecided
#guard verdict [undecided "a", undecided "b"] == .undecided
#guard verdict [rejects "a", rejects "b"] == .reject
#guard verdict [rejects "a", undecided "b"] == .reject
#guard verdict [undecided "a", rejects "b"] == .reject
#guard (answer [undecided "a", rejects "b"]).reason == some "b reject: 1 of 2 drawn units failed"
#guard (answer [undecided "a", undecided "b"]).reason ==
  some "a undecided: 1 of the 2 units drawn; b undecided: 1 of the 2 units drawn"

-- a component whose own commitment doesn't decode rejects, and so the whole, beside an undecided one
#guard verdict [undecided "a", { accepts "b" with root := ByteArray.empty }] == .reject
#guard (answer [undecided "a", { accepts "b" with root := ByteArray.empty }]).reason ==
  some "b reject: commitment: a root is 32 bytes, not 0"

-- the commitment must bind the components' digests: a wrong one rejects, beside an undecided component too
#guard verdict [accepts "a", accepts "b"] (c := commitmentOf toy [accepts "a", { accepts "b" with root := ⟨#[1]⟩ }])
  == .reject
#guard verdict [accepts "a", undecided "b"] (c := bind "verity/other/v0" [("a", (accepts "a").root), ("b", (accepts "b").root)])
  == .reject
#guard (answer [accepts "a", undecided "b"] (c := ⟨(List.replicate 64 0).toArray⟩)).reason ==
  some "the commitment does not bind the components' digests"
#guard (answer [accepts "a", accepts "b"] (c := ByteArray.empty)).reason ==
  some "commitment: the binding of the components' digests is 64 bytes, not 0"

-- on acceptance the components' public outputs and profiles, keyed by id; otherwise nothing
#guard (answer [accepts "a", accepts "b"]).published.map (·.compress) == some "{\"a\":{\"passed\":2},\"b\":{\"passed\":2}}"
#guard (answer [accepts "a", accepts "b"]).profile.map (fun j => (j.getObjValD "a" == j.getObjValD "b", j.getObjValD "c"))
  == some (true, .null)
#guard (answer [accepts "a", undecided "b"]).published.isNone && (answer [accepts "a", undecided "b"]).profile.isNone
#guard (answer [accepts "a", rejects "b"]).published.isNone && (answer [accepts "a", rejects "b"]).profile.isNone

private def readsBack (a : Answer) : Bool :=
  match Answer.ofJson a.toJson, a.canonical with
  | .ok b, .ok s => b.canonical.toOption == some s
  | _, _ => false

-- every verdict's answer reads back through `Answer.ofJson`, which a protocol consuming the composition's answer reads
#guard [answer [accepts "a", accepts "b"], answer [accepts "a", undecided "b"], answer [undecided "a", rejects "b"],
  answer [accepts "a"] [accepts "a"], answer [accepts "a", accepts "b"] (c := ByteArray.empty)].all readsBack

-- the cases must fit the composition: each component named once, in order, under the composition's params
#guard (answer [accepts "a"] [accepts "a"]).reason == some "params: b is missing"
#guard (answer [accepts "a"] [accepts "a", accepts "b"]).reason == some "evidence: b is missing"
#guard (answer [accepts "a", accepts "b", accepts "c"] [accepts "a", accepts "b"]).reason ==
  some "evidence: c is not a component of this composition"
#guard (answer [accepts "a", accepts "b"] [accepts "b", accepts "a"]).reason ==
  some "params: name each component once, in the order of their ids (a, b)"
#guard (answer [accepts "a", accepts "b"] [accepts "a", accepts "a", accepts "b"]).reason ==
  some "params: name each component once, in the order of their ids (a, b)"
#guard (answer [accepts "a", accepts "b"] [accepts "a", { accepts "b" with k := 1 }]).reason ==
  some "evidence: b's case carries params other than the composition's"
#guard (toy.answer "{\"components\":[]}".toUTF8 (commitmentOf toy []) "{\"components\":[]}".toUTF8).reason ==
  some "params: a is missing"
#guard (toy.answer (paramsOf [accepts "a", accepts "b"]).toUTF8 (commitmentOf toy [accepts "a", accepts "b"])
  "{\"components\": []}".toUTF8).reason == some "evidence: not canonical JSON (compact, keys sorted, integers only)"
#guard (toy.answer "{\"components\":[],\"x\":1}".toUTF8 (commitmentOf toy []) "{\"components\":[]}".toUTF8).reason ==
  some "params: the composition's params: the keys must be [components], not [components, x]"
#guard (toy.answer (paramsOf [accepts "a", accepts "b"]).toUTF8 (commitmentOf toy [accepts "a", accepts "b"])
  "{\"components\":[{\"id\":\"a\"}]}".toUTF8).reason ==
  some "evidence: a component's case: the keys must be [commitment, evidence, id, params], not [id]"

private def registeredToy : Protocol := registered "verity/toy/v1" (Stub.stub "r") "b" [Stub.stub "a", Stub.stub "b"]

private def answerR (r : Stubbed) (xs : List Stubbed) (c : ByteArray := commitmentOf registeredToy xs ++ r.root) :
    Answer :=
  Cases.answerRegistered registeredToy r.case (xs.map (·.case)) c

-- a registered composition runs its registration check first: a registration it refuses refuses the whole, whatever the
-- components answer
#guard (answerR (accepts "r") [accepts "a", accepts "b"]).verdict == .accept
#guard (answerR (rejects "r") [accepts "a", accepts "b"]).reason ==
  some "registration: r reject: 1 of 2 drawn units failed"
#guard (answerR (undecided "r") [accepts "a", accepts "b"]).reason ==
  some "registration: r undecided: 1 of the 2 units drawn"
#guard (answerR (rejects "r") [accepts "a", undecided "b"]).verdict == .reject
#guard (answerR (undecided "r") [rejects "a", undecided "b"]).reason ==
  some "registration: r undecided: 1 of the 2 units drawn"
-- an accepted registration leaves the verdict to the components, as `product`'s
#guard (answerR (accepts "r") [accepts "a", undecided "b"]).verdict == .undecided
#guard (answerR (accepts "r") [undecided "a", rejects "b"]).reason == some "b reject: 1 of 2 drawn units failed"
#guard (answerR (accepts "r") [accepts "a"]).reason == some "params: b is missing"
-- the commitment binds the components' digests and then the registration's
#guard (answerR (accepts "r") [accepts "a", accepts "b"]
  (c := commitmentOf registeredToy [accepts "a", accepts "b"])).reason ==
  some "the commitment does not bind the registration's digest"
#guard (answerR (accepts "r") [accepts "a", accepts "b"]
  (c := commitmentOf registeredToy [accepts "a", accepts "b"] ++ { accepts "r" with root := ⟨#[1]⟩ }.root)).reason ==
  some "the commitment does not bind the registration's digest"
#guard (answerR (accepts "r") [accepts "a", accepts "b"]
  (c := commitmentOf toy [accepts "a", accepts "b"] ++ (accepts "r").root)).reason ==
  some "the commitment does not bind the components' digests"
#guard (answerR (accepts "r") [accepts "a", accepts "b"] (c := ByteArray.empty)).reason ==
  some "commitment: the binding of the components' digests is 64 bytes, then the registration's digest: not 0 bytes"
-- the registration the registration check checked is the one the component `b` committed to: one digest in both, once
-- the cases fit the composition, whatever the components answer
private def other : Stubbed := { accepts "r" with root := ⟨(List.replicate 32 9).toArray⟩ }
private def hex (b : Nat) : String := String.join (List.replicate 32 s!"0{b}")
#guard (answerR other [accepts "a", accepts "b"]).reason ==
  some s!"registration: r checked {hex 9}, b committed {hex 7}"
#guard (answerR other [accepts "a", undecided "b"]).verdict == .reject
#guard (answerR other [rejects "a", accepts "b"]).reason ==
  some s!"registration: r checked {hex 9}, b committed {hex 7}"
#guard (answerR (accepts "r") [accepts "a", { accepts "b" with root := ByteArray.empty }]).reason ==
  some s!"registration: r checked {hex 7}, b committed no digest"
#guard (answerR other [accepts "a"]).reason == some "params: b is missing"
-- the registration is the registration check's, under the composition's params
#guard (answerR (accepts "a") [accepts "a", accepts "b"]).reason == some "params: the registration is r's, not a's"
#guard (registeredToy.answer (paramsOfRegistered (accepts "r").case [(accepts "a").case, (accepts "b").case]).toUTF8
  (commitmentOf registeredToy [accepts "a", accepts "b"] ++ (accepts "r").root)
  (evidenceOfRegistered { accepts "r" with k := 1 }.case [(accepts "a").case, (accepts "b").case]).toUTF8).reason ==
  some "evidence: the registration's case carries params other than the composition's"
#guard (registeredToy.answer (paramsOf [accepts "a", accepts "b"]).toUTF8
  (commitmentOf registeredToy [accepts "a", accepts "b"] ++ (accepts "r").root)
  (evidenceOfRegistered (accepts "r").case [(accepts "a").case, (accepts "b").case]).toUTF8).reason ==
  some "params: the composition's params: the keys must be [components, registration], not [components]"
-- on acceptance only the components' public outputs and profiles, keyed by their ids
#guard (answerR (accepts "r") [accepts "a", accepts "b"]).published.map (·.compress) ==
  some "{\"a\":{\"passed\":2},\"b\":{\"passed\":2}}"

end Core.Composition.Guards
