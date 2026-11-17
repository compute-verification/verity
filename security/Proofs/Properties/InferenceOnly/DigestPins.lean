module

public import Proofs.Properties.InferenceOnly.DigestEntries
public import Proofs.Properties.InferenceOnly.DigestParts

/-! The closure digest pins the decoded Definition up to a collision, given `EntriesPin`. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open InferenceOnlyCheck.DefinitionDigest (canonicalEntry)

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

end InferenceOnlyCheck.Properties.Record.SecurityProofs
