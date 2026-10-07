import Proofs.Flock.Soundness.Discharge.DecodeZL.ZL
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# The opening's rounds carry its messages (R11b)

`op_carries`: the ring switches' rounds of `rsFrames`, read by `wOP`, are `opMsgs`, cut at `opShape`'s coins. Each ring
switch's round closes at its `sq 2 7` and carries the switch's 128 values (`carries_switch`; the batch label rides in
front of the first), the rest by induction (`carries_switches`), and the batching coin's round carries nothing.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128

/-- **The opening's wires**: each ring switch's round its 128 values, the batching coin's round nothing. -/
def wOP (n : ℕ) : List (List Wire) := List.replicate n [.v 128] ++ [[]]

theorem readAll_v {fs : List Frame} {a : Array F128} {n : ℕ} (ha : a.size = n) (hv : valsOf fs = a.toList) :
    readAll [.v n] (valsOf fs) (bytesOf fs) = [⟨Fin n → 𝔽, vecOf n a⟩] := by
  have hl : a.toList.take n = a.toList := List.take_of_length_le (by simp [ha])
  rw [hv]
  simp only [readAll, Wire.v, hl, Array.getElem!_toList]
  rfl

/-- **A ring switch's round** carries its 128 values, behind any label-only prefix. -/
theorem carries_switch {pre : List Frame} (hpre : ∀ f ∈ pre, closes f = false) (hpv : valsOf pre = [])
    (hpwf : ∀ f ∈ pre, f.WF) {a : Array F128} (ha : a.size = 128) :
    Carries [[.v 128]] [.msg (Fin 128 → 𝔽), .coin (Fin 7 → 𝔽)]
      ((pre ++ ([.label "flock-ring-switch-v0".toUTF8, .vec a.toList] : List Frame)) ++ [.sq 2 7])
      [⟨Fin 128 → 𝔽, vecOf 128 a⟩] [] := by
  refine carries_one ?_ (by simp [closes]) ?_ (by simp [coinPat, isCoin])
  · intro f hf
    rcases List.mem_append.1 hf with hf | hf
    · exact hpre f hf
    · simp only [List.mem_cons, List.mem_nil_iff, or_false] at hf
      rcases hf with rfl | rfl <;> rfl
  · have hwf : ∀ f ∈ (pre ++ ([.label "flock-ring-switch-v0".toUTF8, .vec a.toList] : List Frame)) ++ [.sq 2 7],
        f.WF := by
      intro f hf
      simp only [List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at hf
      rcases hf with (hf | rfl | rfl) | rfl
      · exact hpwf f hf
      · show ("flock-ring-switch-v0".toUTF8).size < 2 ^ 64
        decide
      · exact wf_vec (by simp [ha])
      · exact wf_sq _ (by norm_num)
    rw [decRound_encs hwf]
    refine readAll_v ha ?_
    unfold valsOf at hpv ⊢
    simp [List.flatMap_append, hpv, Frame.vals]

theorem size_at {rs : Array (Array F128)} (h : ∀ v ∈ rs, v.size = 128) {i : ℕ} (hi : i < rs.size) :
    (rs[i]!).size = 128 := by
  rw [getElem!_pos rs i hi]
  exact h _ (Array.getElem_mem hi)

/-- **The ring switches from `a` on** carry their values, one per round. -/
theorem carries_switches (rs : Array (Array F128)) (h : ∀ v ∈ rs, v.size = 128) : ∀ (k a : ℕ), a + k ≤ rs.size →
    Carries (List.replicate k [.v 128]) (switches 𝔽 k)
      ((List.range' a k).flatMap fun i => [.label "flock-ring-switch-v0".toUTF8, .vec (rs[i]!).toList, .sq 2 7])
      ((List.range' a k).map fun i => ⟨Fin 128 → 𝔽, vecOf 128 rs[i]!⟩) []
  | 0, a, _ => by simpa [switches] using carries_nil
  | k + 1, a, hk => by
      have h₁ := carries_switch (pre := []) (by simp) rfl (by simp) (size_at h (show a < rs.size by omega))
      have := h₁.append (carries_switches rs h k (a + 1) (by omega))
      simpa [List.range'_succ, List.replicate_succ, switches] using this

theorem op_carries : OPCarries wOP := by
  intro st sch nExtra bytes p _ hp
  have hs := decodeProof_sizes _ _ _ hp
  have hsz : p.ringSwitches.size = nExtra + 1 + 1 := by rw [hs.rs]; simp [shapeOf]; omega
  rw [show 2 + nExtra = nExtra + 1 + 1 by omega]
  have h₁ := carries_switch (pre := [.label "flock-pcs-open-batch-v0".toUTF8]) (by simp [closes]) rfl
    (by simp only [List.mem_cons, List.mem_nil_iff, or_false]; rintro f rfl
        show ("flock-pcs-open-batch-v0".toUTF8).size < 2 ^ 64
        decide)
    (size_at hs.rsEach (show 0 < p.ringSwitches.size by omega))
  have h₂ := carries_switches p.ringSwitches hs.rsEach (nExtra + 1) 1 (by omega)
  have h₃ : Carries [[]] [.coin (Fin (nExtra + 1 + 1) → 𝔽)] ([] ++ [.sq 2 (nExtra + 1 + 1)]) [] [] :=
    carries_one (by simp) (by simp [closes]) (decRound_nil _) (by simp [coinPat, isCoin])
  have := h₁.append (h₂.append h₃)
  simpa [wOP, opShape, switches, rsFrames, opMsgs, List.range_eq_range', List.range'_succ, List.replicate_succ]
    using this

end FlockSoundness.Discharge.DecodeZL
