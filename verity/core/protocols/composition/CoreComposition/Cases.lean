/-
Inputs for a composition, which the `#guard`s of the composition and of each product build: a component's case (`Part`:
its id, the parameters, commitment and evidence its own `answer` reads, and the digest its commitment decodes to), a
stub's case as one (its parameter k, the units drawn and passed its evidence records, its root), and the composition's
params, evidence and commitment over a list of them.
-/
import CoreComposition.Product

namespace Core.Composition.Cases

/-- A component's case: its parameters as canonical JSON text, its commitment and evidence bytes, and the digest the
component's `answer` gives its commitment. -/
structure Part where
  id : String
  params : String
  commitment : ByteArray
  evidence : ByteArray
  digest : ByteArray

/-- `P`'s case on these inputs, its digest the one `P`'s answer gives (none when the commitment doesn't decode). -/
def Part.of (P : Protocol) (params commitment evidence : String) : Part :=
  ⟨P.id, params, commitment.toUTF8, evidence.toUTF8,
   (P.answer params.toUTF8 commitment.toUTF8 evidence.toUTF8).digest.getD ByteArray.empty⟩

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

/-- A stub's case as a component's: its root is its own digest. -/
def Stubbed.part (x : Stubbed) : Part := ⟨x.id, stubParams x, x.root, (stubEvidence x).toUTF8, x.root⟩

def paramsOfParts (xs : List Part) : String :=
  "{\"components\":[" ++ ",".intercalate (xs.map fun x =>
    "{\"id\":\"" ++ x.id ++ "\",\"params\":" ++ x.params ++ "}") ++ "]}"

def evidenceOfParts (xs : List Part) : String :=
  "{\"components\":[" ++ ",".intercalate (xs.map fun x =>
    "{\"commitment\":\"" ++ Flock.hexOf x.commitment ++ "\",\"evidence\":\"" ++ Flock.hexOf x.evidence ++
    "\",\"id\":\"" ++ x.id ++ "\",\"params\":" ++ x.params ++ "}") ++ "]}"

/-- The binding the composition `P` expects of these cases. -/
def commitmentOfParts (P : Protocol) (xs : List Part) : ByteArray := bind P.id (xs.map fun x => (x.id, x.digest))

/-- `P`'s answer on the cases `xs`, under the params of `ps` and the commitment `c` (by default those of `xs`). -/
def answerOfParts (P : Protocol) (xs : List Part) (ps : List Part := xs) (c : ByteArray := commitmentOfParts P xs) :
    Answer :=
  P.answer (paramsOfParts ps).toUTF8 c (evidenceOfParts xs).toUTF8

def paramsOf (xs : List Stubbed) : String := paramsOfParts (xs.map (·.part))

def evidenceOf (xs : List Stubbed) : String := evidenceOfParts (xs.map (·.part))

def commitmentOf (P : Protocol) (xs : List Stubbed) : ByteArray := commitmentOfParts P (xs.map (·.part))

def answer (P : Protocol) (xs : List Stubbed) (ps : List Stubbed := xs) (c : ByteArray := commitmentOf P xs) : Answer :=
  answerOfParts P (xs.map (·.part)) (ps.map (·.part)) c

end Core.Composition.Cases
