module

public import Proofs.Properties.Challenger.Built
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# What the challenger's parties sign exists

Each message a party of a session signs is a hash over a tagged digest of a value's canonical bytes, and the reader
recomputes it. These say the reader's functions return one, for values an honest party writes:

* a layout of families has a domain (`liveDomain_ok`);
* a record and a close have their messages under an identifier of 16 bytes (`recordMessage_ok`, `closeMessage_ok`);
* a ledger entry under the ledger's schema whose body is printable has its message (`ledgerMessage_ok`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Properties.Challenger.Messages

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger
open Claims.Json (Good)
open FlockComplete.JsonRT (good_mkObj)
open CanonRead (distinct_of_keys)
open Built (tagged_ok)
open Entry (unsigned_toJson)

theorem good_families (fs : Families) : Good (familiesJson fs) := by
  refine .arr fun x hx => ?_
  obtain ⟨p, -, rfl⟩ := Array.mem_map.1 hx
  refine .arr fun y hy => ?_
  simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hy
  rcases hy with rfl | rfl
  · exact .str _
  · exact .num _

theorem good_commitment (c : LiveCommitment) : Good c.toJson :=
  good_mkObj (distinct_of_keys (by simp only [List.map_cons, List.map_nil]; decide)) (by
    intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl
    · exact .num _
    · exact good_families _
    · exact .str _
    · exact .str _)

/-- **A layout has a domain**: `liveDomain` returns one for every scheme, epoch and families. -/
theorem liveDomain_ok (scheme : String) (epoch : Nat) (fs : Families) :
    ∃ dom, liveDomain scheme epoch fs = .ok dom := by
  obtain ⟨b, hb⟩ := tagged_ok BINDING
    (j := Json.mkObj [("epoch", Lean.toJson epoch), ("families", familiesJson fs), ("scheme", Json.str scheme)])
    (good_mkObj (distinct_of_keys (by simp only [List.map_cons, List.map_nil]; decide)) (by
      intro p hp
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl | rfl
      · exact .num _
      · exact good_families _
      · exact .str _))
  obtain ⟨q, hq⟩ := tagged_ok "veritor/indexed-domain/range/v1"
    (j := Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)),
      ("stop", Lean.toJson (total fs))])
    (good_mkObj (distinct_of_keys (by simp only [List.map_cons, List.map_nil]; decide)) (by
      intro p hp
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl | rfl <;> exact .num _))
  exact ⟨_, by unfold liveDomain; rw [hb, hq]; rfl⟩

/-- **A record has a message** under an identifier of 16 bytes. -/
theorem recordMessage_ok {I : ByteArray} (hI : I.size = 16) (host : String) (c : LiveCommitment) :
    ∃ m, recordMessage I host c = .ok m := by
  obtain ⟨d, hd⟩ := tagged_ok COMMITMENT_TAG (good_commitment c)
  refine ⟨Flock.Sha512.hash (Spot.Challenger.field RECORD_ROLE.toUTF8 ++ I ++
    Spot.Challenger.field (Spot.Challenger.field host.toUTF8) ++ Spot.Challenger.field d), ?_⟩
  unfold recordMessage
  rw [hd]
  simp only [hI, beq_self_eq_true]
  rfl

/-- **A close has a message** under an identifier of 16 bytes. -/
theorem closeMessage_ok {I : ByteArray} (hI : I.size = 16) (host : String) (c : LiveCommitment) (session : String)
    (lines : Nat) (chain : ByteArray) (ended : Option String) :
    ∃ m, closeMessage I host c session lines chain ended = .ok m := by
  obtain ⟨r, hr⟩ := recordMessage_ok hI host c
  exact ⟨_, by unfold closeMessage; rw [hr]; rfl⟩

/-- **An entry under the ledger's schema has a message** under an identifier of 16 bytes, when its body is
printable. -/
theorem ledgerMessage_ok {I : ByteArray} (hI : I.size = 16) {r : Core.Record} (hs : r.schema = LEDGER_SCHEMA)
    (hb : Good r.body) : ∃ m, ledgerMessage I r.toJson = .ok m := by
  have hg : Good (Core.Record.unsigned r.toJson) := by
    rw [unsigned_toJson]
    refine good_mkObj (distinct_of_keys (by simp only [List.map_cons, List.map_nil]; decide)) ?_
    intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl | rfl | rfl
    · exact hb
    · exact .str _
    · exact .str _
    · exact .str _
    · exact .num _
    · cases r.session with
      | none => exact .null
      | some s => exact .str s
  obtain ⟨d, hd⟩ := tagged_ok "verity/challenger-auditor/entry/v2" hg
  have hsch : getD r.toJson "schema" = .str r.schema := rfl
  refine ⟨Core.Record.message LEDGER_ROLE I d, ?_⟩
  unfold ledgerMessage
  rw [hsch, hs]
  have ht : ledgerTag? LEDGER_SCHEMA = some "verity/challenger-auditor/entry/v2" := rfl
  simp only [hI, beq_self_eq_true, ht]
  rw [hd]
  rfl

end Properties.Challenger.Messages
