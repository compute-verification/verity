module

public import Proofs.Properties.Nci.DigestDecode
public import Proofs.Properties.Nci.DigestEntries
public import Proofs.Properties.Nci.DigestRefs
import all Lean.Data.Json.Basic

/-!
# Equivalent canonical entries hold equivalent flat parts

The canonical entry writes each reference as its flat parts (`written`): the one part, or `{"concat": parts}`. A flat
part the decoder parses holds no concatenation (`PartsRel`), so a lone part is never equivalent to a written
concatenation, and two equivalent written sequences hold the same number of parts, equivalent one by one
(`written_jsonEquiv`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Claims.Json (Wf)
open FlockDeploy (JsonEquiv jsonEquiv_arr_iff)
open NciCheck.DefinitionDigest (written)

theorem concat_mkObj1 (v : Json) : ((Json.mkObj [("concat", v)]).getObjVal? "concat").toOption = some v := by
  simp only [Json.mkObj, Json.getObjVal?]
  rw [Std.TreeMap.Raw.get?_eq_getElem?,
    Std.TreeMap.Raw.getElem?_ofList_of_mem Std.ReflCmp.compare_self (by simp) List.mem_cons_self]
  rfl

theorem written_one {ps : Array Json} (h : ps.size = 1) : written ps = ps[0] := by
  rcases ps with ⟨_ | ⟨p, _ | ⟨q, l⟩⟩⟩ <;> simp at h
  rfl

theorem written_many {ps : Array Json} (h : ¬ ps.size = 1) : written ps = Json.mkObj [("concat", .arr ps)] := by
  rcases ps with ⟨_ | ⟨p, _ | ⟨q, l⟩⟩⟩
  · rfl
  · simp at h
  · rfl

/-- Equivalent flat sequences, as the canonical entry writes them, of parts with no concatenation: the same parts up
to equivalence, one by one. -/
theorem written_jsonEquiv {ps ps' : Array Json} (hw : ∀ p ∈ ps, Wf p) (hw' : ∀ p ∈ ps', Wf p)
    (hc : ∀ p ∈ ps, (p.getObjVal? "concat").toOption = none) (hc' : ∀ p ∈ ps', (p.getObjVal? "concat").toOption = none)
    (h : JsonEquiv (written ps) (written ps')) :
    ps.size = ps'.size ∧ ∀ i (h1 : i < ps.size) (h2 : i < ps'.size), JsonEquiv ps[i] ps'[i] := by
  have one : ∀ (p : Json) (qs : Array Json), Wf p → (∀ q ∈ qs, Wf q) → (p.getObjVal? "concat").toOption = none →
      ¬ JsonEquiv p (Json.mkObj [("concat", .arr qs)]) := by
    intro p qs hp hq hpc he
    have := field_jsonEquiv hp (wf_mkObj1 (.arr hq)) he "concat"
    rw [hpc, concat_mkObj1] at this
    exact this
  by_cases h1 : ps.size = 1 <;> by_cases h1' : ps'.size = 1
  · rw [written_one h1, written_one h1'] at h
    refine ⟨by omega, fun i hi hi' => ?_⟩
    obtain rfl : i = 0 := by omega
    exact h
  · rw [written_one h1, written_many h1'] at h
    exact absurd h (one _ _ (hw _ (Array.getElem_mem _)) hw' (hc _ (Array.getElem_mem _)))
  · rw [written_many h1, written_one h1'] at h
    exact absurd h.symm (one _ _ (hw' _ (Array.getElem_mem _)) hw (hc' _ (Array.getElem_mem _)))
  · rw [written_many h1, written_many h1'] at h
    have := field_jsonEquiv (wf_mkObj1 (.arr hw)) (wf_mkObj1 (.arr hw')) h "concat"
    rw [concat_mkObj1, concat_mkObj1] at this
    exact (jsonEquiv_arr_iff ps ps').1 this

end NciCheck.Properties.Record.SecurityProofs
