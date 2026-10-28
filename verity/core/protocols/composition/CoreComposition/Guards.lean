/-
The `#guard`s that pin the composition's verdict rule on a product of two stubs, `a` and `b` (`Core.Example.protocol`,
k = 2): every pair of the components' verdicts, reject dominating; a commitment that doesn't bind their digests; and
every way the cases can fail to fit the composition. Then three stubs `a`, `b` and `c` of which `a` and `c` must commit
to one record (`same`): a pair that didn't refused whatever the components answer. Then the same three with `a` and `b`
also on one store (`tiedSame`): two stores refused unless `same` rejects already.
-/
import CoreComposition.Cases
import CoreComposition.Same
import CoreComposition.Tie
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

private def sameToy : Protocol := same "verity/toy/v2" ["a", "c"] [Stub.stub "a", Stub.stub "b", Stub.stub "c"]

private def answerS (xs : List Stubbed) (c : ByteArray := commitmentOf sameToy xs) : Answer :=
  Cases.answer sameToy xs (c := c)

private def hex (b : Nat) : String := String.join (List.replicate 32 s!"0{b}")

private def stray (x : Stubbed) : Stubbed := { x with root := ⟨(List.replicate 32 9).toArray⟩ }

-- `a` and `c` committed to one record: the verdict is `product`'s, whatever `b` committed to
#guard (answerS [accepts "a", accepts "b", accepts "c"]).verdict == .accept
#guard (answerS [accepts "a", stray (accepts "b"), accepts "c"]).verdict == .accept
#guard (answerS [accepts "a", undecided "b", accepts "c"]).verdict == .undecided
#guard (answerS [rejects "a", accepts "b", accepts "c"]).reason == some "a reject: 1 of 2 drawn units failed"
-- `a` and `c` committed to two: refused once the cases fit, whatever the components answer
#guard (answerS [accepts "a", accepts "b", stray (accepts "c")]).reason ==
  some s!"record: a committed {hex 7}, c committed {hex 9}"
#guard (answerS [accepts "a", undecided "b", stray (undecided "c")]).verdict == .reject
#guard (answerS [rejects "a", accepts "b", stray (accepts "c")]).reason ==
  some s!"record: a committed {hex 7}, c committed {hex 9}"
#guard (answerS [accepts "a", accepts "b", stray (accepts "c")]).published.isNone
#guard (answerS [accepts "a", stray (accepts "c")]).reason == some "params: b is missing"
-- a commitment that doesn't decode rejects as its component
#guard (answerS [accepts "a", accepts "b", { accepts "c" with root := ByteArray.empty }]).reason ==
  some "c reject: commitment: a root is 32 bytes, not 0"

private def reads (s : String) : String × (ByteArray → Option ByteArray) :=
  (s, fun b => ((Stub.stub s).commitment b).toOption.map (Stub.stub s).digest)

private def tiedToy : Protocol :=
  tiedSame "verity/toy/v3" ["a", "c"] [⟨"store", [reads "a", reads "b"]⟩] [Stub.stub "a", Stub.stub "b", Stub.stub "c"]

private def answerT (xs : List Stubbed) (c : ByteArray := commitmentOf tiedToy xs) : Answer :=
  Cases.answer tiedToy xs (c := c)

-- `a` and `b` on one store and `a` and `c` on one record: the verdict is `product`'s
#guard (answerT [accepts "a", accepts "b", accepts "c"]).verdict == .accept
#guard (answerT [accepts "a", undecided "b", accepts "c"]).verdict == .undecided
-- `a` and `b` on two stores: refused once the cases fit, unless `same` rejects already, and nothing is published
#guard (answerT [accepts "a", stray (accepts "b"), accepts "c"]).reason ==
  some s!"store: a committed {hex 7}, b committed {hex 9}"
#guard (answerT [undecided "a", stray (undecided "b"), accepts "c"]).verdict == .reject
#guard (answerT [accepts "a", stray (accepts "b"), accepts "c"]).published.isNone
#guard (answerT [accepts "a", stray (accepts "b"), rejects "c"]).reason == some "c reject: 1 of 2 drawn units failed"
#guard (answerT [accepts "a", stray (accepts "b"), stray (accepts "c")]).reason ==
  some s!"record: a committed {hex 7}, c committed {hex 9}"
#guard (answerT [accepts "a", stray (accepts "b")]).reason == some "params: c is missing"
-- a commitment that doesn't decode rejects as its component
#guard (answerT [accepts "a", { accepts "b" with root := ByteArray.empty }, accepts "c"]).reason ==
  some "b reject: commitment: a root is 32 bytes, not 0"

end Core.Composition.Guards
