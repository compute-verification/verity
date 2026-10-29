/-
The verifier's product for a configuration (`verity run CONFIG`): `Core.Composition.product` over exactly the components
the configuration's `protocols` lists, by their ids. Onsite is its five-component case, so listing every one of
onsite's components is onsite's own protocol (`verity/onsite/v0`); any other list is `product "verity/product/v0"` over
the listed components, in the order of their ids. A list naming a component onsite doesn't have, naming one twice, or naming none is refused by
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
def registry : List Protocol := Core.Onsite.components

/-- The product over exactly the components `ids` names, or why the list is refused. -/
def protocolOf (ids : List String) : Except String Protocol :=
  let known := registry.map (·.id)
  match ids.find? (fun i => !known.contains i) with
  | some i => .error s!"{i} is not a component a configuration may list (onsite's are {", ".intercalate known})"
  | none =>
    if ids.isEmpty then .error "the configuration lists no component"
    else match ids.find? (fun i => ids.count i > 1) with
      | some i => .error s!"{i} is listed twice"
      | none =>
        if ids.length == known.length then .ok Core.Onsite.protocol
        else .ok (product id (registry.filter (ids.contains ·.id)))

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
#guard idOf ["verity/sampled-proofs/one-stage/v1"] == some id
#guard idOf ["verity/sampled-proofs/one-stage/v1", "verity/certifier/v0"] == some id
#guard ordered (registry.filter (["verity/sampled-proofs/one-stage/v1", "verity/certifier/v0"].contains ·.id))
#guard (protocolOf []).toOption.isNone
#guard (protocolOf ["verity/wipe/v0"]).toOption.isNone
#guard (protocolOf ["verity/certifier/v0", "verity/certifier/v0"]).toOption.isNone

-- a listed component whose case its own `answer` refuses rejects the product, and an unlisted one is not a component
private def refused (P : Protocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def pair : List Protocol :=
  registry.filter (["verity/sampled-proofs/one-stage/v1", "verity/certifier/v0"].contains ·.id)

private def pairAnswer (xs : List Case) : Answer :=
  Cases.answerCases (product id pair) xs ⟨(List.replicate 64 0).toArray⟩

#guard (pairAnswer (pair.map refused)).verdict == .reject
#guard (pairAnswer (pair.map refused)).published.isNone
#guard (pairAnswer [refused SampledProofs.OneStage.protocol]).reason == some "params: verity/certifier/v0 is missing"

end Core.Product
