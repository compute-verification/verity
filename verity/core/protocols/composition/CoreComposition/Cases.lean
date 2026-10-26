/-
Inputs for a composition, which the `#guard`s of the composition and of each product build: a stub's case (its
parameter k, the units drawn and passed its evidence records, its root), any component's case as its own `serve` reads
it (`Case`), and the composition's params, evidence and commitment over a list of them.
-/
import CoreComposition.Product

namespace Core.Composition.Cases

/-- A stub's case. -/
structure Stubbed where
  id : String
  k : Nat := 2
  drawn : Nat
  passed : Nat
  root : ByteArray := ⟨(List.replicate 32 7).toArray⟩

def accepts (id : String) : Stubbed := { id, drawn := 2, passed := 2 }
def rejects (id : String) : Stubbed := { id, drawn := 2, passed := 1 }
def undecided (id : String) : Stubbed := { id, drawn := 1, passed := 1 }

def stubParams (x : Stubbed) : String := "{\"k\":" ++ toString x.k ++ "}"
def stubEvidence (x : Stubbed) : String :=
  "{\"drawn\":" ++ toString x.drawn ++ ",\"passed\":" ++ toString x.passed ++ "}"

/-- A stub's case as a component's case: its root is its commitment, and its own digest. -/
def Stubbed.case (x : Stubbed) : Case :=
  { id := x.id, params := stubParams x, commitment := x.root, evidence := (stubEvidence x).toUTF8 }

def paramsOfCases (xs : List Case) : String :=
  "{\"components\":[" ++ ",".intercalate (xs.map fun x =>
    "{\"id\":\"" ++ x.id ++ "\",\"params\":" ++ x.params ++ "}") ++ "]}"

def evidenceOfCases (xs : List Case) : String :=
  "{\"components\":[" ++ ",".intercalate (xs.map fun x =>
    "{\"commitment\":\"" ++ Flock.hexOf x.commitment ++ "\",\"evidence\":\"" ++ Flock.hexOf x.evidence ++
    "\",\"id\":\"" ++ x.id ++ "\",\"params\":" ++ x.params ++ "}") ++ "]}"

def paramsOf (xs : List Stubbed) : String := paramsOfCases (xs.map (·.case))

def evidenceOf (xs : List Stubbed) : String := evidenceOfCases (xs.map (·.case))

/-- The binding the composition `P` expects of these cases. -/
def commitmentOf (P : Protocol) (xs : List Stubbed) : ByteArray := bind P.id (xs.map fun x => (x.id, x.root))

/-- `P`'s answer on the cases `xs`, under the params of `ps` and the commitment `c` (by default those of `xs`). -/
def answer (P : Protocol) (xs : List Stubbed) (ps : List Stubbed := xs) (c : ByteArray := commitmentOf P xs) : Answer :=
  P.answer (paramsOf ps).toUTF8 c (evidenceOf xs).toUTF8

/-- `P`'s answer on any components' cases `xs`, under their own params and the commitment `c`. -/
def answerCases (P : Protocol) (xs : List Case) (c : ByteArray) : Answer :=
  P.answer (paramsOfCases xs).toUTF8 c (evidenceOfCases xs).toUTF8

end Core.Composition.Cases
