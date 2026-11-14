module

public import Proofs.Properties.Nci.DigestEntries
public import Proofs.Properties.Nci.DigestParts

/-!
# The closure digest pins the decoded Definition

`DigestPins` splits at the canonical entries. One closure digest gives, up to a collision, equivalent canonical entries
for every Definition the key reaches (`entries_agree`); what is left is that such entries pin the decoded Definition
(`EntriesPin`): the decoder reads each field from the entry and each callee's fields from its own Definition, so the two
decoders build the same Definition, callees and all. `digestPins_of_entriesPin` joins the two.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open NciCheck.DefinitionDigest (canonicalEntry)

/-- **Equivalent canonical entries pin the decoded Definition.** Two Program texts whose descriptors give every
Definition `key` reaches equivalent canonical entries decode `key` to the same Definition, callees and all. -/
def EntriesPin : Prop :=
  ∀ (text text' : String) (j j' : Json) (P P' : Prog) (key : String) (k k' : ℕ),
    Flock.Program.decode text.toUTF8 = .ok P → Flock.Program.decode text'.toUTF8 = .ok P' →
    Flock.JsonParse.parse text = .ok j → Flock.JsonParse.parse text' = .ok j' →
    (∀ c, Reaches j key c → Reaches j' key c ∧ ∃ ce ce',
      canonicalEntry (tab j "callees") (tab j "types") (entry j c) = .ok ce ∧
      canonicalEntry (tab j' "callees") (tab j' "types") (entry j' c) = .ok ce' ∧ FlockDeploy.JsonEquiv ce ce') →
    P.index[key]? = some k → P'.index[key]? = some k' → Same P P' k k'

theorem digestPins_of_entriesPin (hE : EntriesPin) : DigestPins := by
  intro text text' j j' P P' key h k k' hP hP' hj hj' H H' hk hk'
  rcases entries_agree (FlockComplete.JsonRT.parse_wf text j hj) (FlockComplete.JsonRT.parse_wf text' j' hj') H H'
    with hC | hA
  · exact .inr hC
  · exact .inl (hE text text' j j' P P' key k k' hP hP' hj hj' hA hk hk')

end NciCheck.Properties.Record.SecurityProofs
