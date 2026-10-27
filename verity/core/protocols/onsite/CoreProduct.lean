/-
The verifier's product for a configuration (`verity run CONFIG`): `Core.Composition.productB` over exactly the components
the configuration's `protocols` lists, by their ids, on the caller's blobs. Onsite is its six-component case, so listing
every one of onsite's components is onsite's own protocol (`verity/onsite/v0`, with its registration check); any other
list is `productB "verity/product/v0"` over the listed components, in the order of their ids. A list naming a component onsite doesn't have, naming one twice, or naming none is refused by
name, so a product never runs over a component it can't check, and an unlisted component is never read as accepted.

    product IDS id | check PARAMS COMMITMENT EVIDENCE | serve        IDS: the listed ids, comma-separated

The command after `IDS` is the core CLI's (`CoreProtocol.Cli`).
-/
import CoreOnsite

namespace Core.Product

open Composition

/-- A proper subset's product id: the parameters and the binding name its components. -/
def id : String := "verity/product/v0"

/-- Every component a configuration may list: onsite's, in the order of their ids. -/
def registry : List BlobProtocol := Core.Onsite.components

/-- The product over exactly the components `ids` names, or why the list is refused. -/
def protocolOf (ids : List String) : Except String BlobProtocol :=
  let known := registry.map (·.id)
  match ids.find? (fun i => !known.contains i) with
  | some i => .error s!"{i} is not a component a configuration may list (onsite's are {", ".intercalate known})"
  | none =>
    if ids.isEmpty then .error "the configuration lists no component"
    else match ids.find? (fun i => ids.count i > 1) with
      | some i => .error s!"{i} is listed twice"
      | none =>
        if ids.length == known.length then .ok Core.Onsite.protocol
        else .ok (productB id (registry.filter (ids.contains ·.id)))

/-- `product IDS ...`: the core CLI of `protocolOf IDS`; 64 when `IDS` is refused or missing. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | ids :: rest =>
    match protocolOf ((ids.splitOn ",").filter (· ≠ "")) with
    | .ok P => P.main rest
    | .error e => IO.eprintln s!"product: {e}"; return 64
  | [] => IO.eprintln "usage: product IDS id | check PARAMS COMMITMENT EVIDENCE | serve"; return 64

private def idOf (ids : List String) : Option String := (protocolOf ids).toOption.map (·.id)

#guard idOf (registry.map (·.id)) == some "verity/onsite/v0"
#guard idOf ["verity/sampled-proofs/one-stage/v2"] == some id
#guard idOf ["verity/sampled-proofs/one-stage/v2", "verity/certifier/v0"] == some id
#guard orderedB (registry.filter (["verity/sampled-proofs/one-stage/v2", "verity/certifier/v0"].contains ·.id))
#guard (protocolOf []).toOption.isNone
#guard (protocolOf ["verity/sampled-proofs/one-stage/v1"]).toOption.isNone
#guard (protocolOf ["verity/certifier/v0", "verity/certifier/v0"]).toOption.isNone

-- a listed component whose case its own `answer` refuses rejects the product, and an unlisted one is not a component
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def pair : List BlobProtocol :=
  registry.filter (["verity/sampled-proofs/one-stage/v2", "verity/certifier/v0"].contains ·.id)

private def pairAnswer (xs : List Case) : Answer :=
  Cases.answerCases ((productB id pair).on #[]) xs ⟨(List.replicate 64 0).toArray⟩

#guard (pairAnswer (pair.map refused)).verdict == .reject
#guard (pairAnswer (pair.map refused)).published.isNone
#guard (pairAnswer [refused SampledProofs.OneStageZk.protocol]).reason == some "params: verity/certifier/v0 is missing"

end Core.Product
