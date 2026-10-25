import Properties.Proofs.Pouw.LiveSalt.Ledger

/-!
# The Pearl-C period credits distinct work within each epoch (`CreditDistinct` of `PouwCheck.pearlCPeriodCredit`)

`PouwCheck.pearlCPeriodCredit` credits the manifests its clauses 0 to 5 return (`PouwCheck.Period.periodChecks`) only
once clause 6 (`PouwCheck.Period.passesDistinct`) and then `PouwCheck.Period.distinctCalls` have passed over them. The
latter reads each pass's epoch and call indices by `PassesDistinct`'s own functions (`namedEpoch_eq`, `callsOf_eq`),
and checks each index below 2^32 and against every earlier pass of the same epoch (`freshCalls_ok`), so the passes it
accepts meet `PassesDistinct` (`passesDistinct_of_distinctCalls`), and so does what the period credits
(`creditDistinct`). Two passes' epochs are compared by `PouwCheck.Period.sameEpoch`, which holds of any two equal epochs
(`sameEpoch_self`), so nothing here reads `BEq Json`, which the kernel can't reduce.
-/

namespace Properties.Pouw.LiveSalt

open Lean (Json)
open PouwCheck.Period (sameEpoch namedEpoch callsOf freshCalls distinctCalls)
open Properties.Pouw.LiveSalt.Ledger (bind_ok)

/-- Clause 6 reads a pass's epoch as `PassesDistinct` does. -/
theorem namedEpoch_eq (m : Json) : namedEpoch m = epochOf m := rfl

/-- Clause 6 reads a pass's call indices as `PassesDistinct` does. -/
theorem callsOf_eq (m : Json) : callsOf m = callIndices m := by
  first
  | rfl
  | (unfold PouwCheck.Period.callsOf callIndices; split <;> simp_all)

theorem sameEpoch_self (e : Json) : sameEpoch e e = true := by
  cases e <;> first | rfl | simp [sameEpoch]

/-- What `freshCalls` accepts for pass `i` of epoch `e` after the passes `seen`: each of its call indices `us` is below
2^32 and in no pass of `seen` of the same epoch. -/
theorem freshCalls_ok {nm : String} {i : Nat} {e : Json} {seen : List (Nat × Json × List Int)} (us : List Int)
    (h : freshCalls nm i e seen us = .ok ()) :
    ∀ u ∈ us, (0 ≤ u ∧ u < 2 ^ 32) ∧ ∀ p ∈ seen, sameEpoch p.2.1 e = true → u ∉ p.2.2 := by
  induction us with
  | nil => simp
  | cons u us ih =>
    simp only [freshCalls] at h
    split at h
    · rename_i hu
      split at h
      · first | cases h | simp [throw, throwThe, MonadExceptOf.throw] at h
      · rename_i hf
        intro u' hu'
        rcases List.mem_cons.1 hu' with rfl | hu'
        · refine ⟨hu, fun p hp hs hm => ?_⟩
          have hn := List.find?_eq_none.1 hf p hp
          simp [hs, hm] at hn
        · exact ih h u' hu'
    · first | cases h | simp [throw, throwThe, MonadExceptOf.throw] at h

/-- What `distinctCalls` accepts of the passes `ms` after the passes `seen`: their call indices are below 2^32, none is
in a pass of `seen` of the same epoch, and of two of them of one epoch the later holds none of the earlier's. -/
theorem distinctCalls_ok {nm : String} (ms : List Json) :
    ∀ {i : Nat} {seen : List (Nat × Json × List Int)}, distinctCalls nm i seen ms = .ok () →
      (∀ m ∈ ms, ∀ u ∈ callIndices m, 0 ≤ u ∧ u < 2 ^ 32) ∧
      (∀ m ∈ ms, ∀ p ∈ seen, sameEpoch p.2.1 (epochOf m) = true → ∀ u ∈ callIndices m, u ∉ p.2.2) ∧
      ms.Pairwise fun m m' => sameEpoch (epochOf m) (epochOf m') = true → ∀ u ∈ callIndices m', u ∉ callIndices m := by
  induction ms with
  | nil => intro _ _ _; exact ⟨by simp, by simp, .nil⟩
  | cons m ms ih =>
    intro i seen h
    simp only [distinctCalls] at h
    obtain ⟨a, h1, h2⟩ := bind_ok.1 h
    cases a
    try dsimp only at h2
    have hf := freshCalls_ok _ h1
    rw [callsOf_eq, namedEpoch_eq] at hf
    obtain ⟨r, s, p⟩ := ih h2
    refine ⟨?_, ?_, ?_⟩
    · intro m' hm' u hu
      rcases List.mem_cons.1 hm' with rfl | hm'
      · exact (hf u hu).1
      · exact r m' hm' u hu
    · intro m' hm' q hq hs u hu
      rcases List.mem_cons.1 hm' with rfl | hm'
      · exact (hf u hu).2 q hq hs
      · exact s m' hm' q (List.mem_append_left _ hq) hs u hu
    · refine List.pairwise_cons.2 ⟨fun m' hm' hs u hu => ?_, p⟩
      have hm := s m' hm' (i, namedEpoch m, callsOf m) (List.mem_append_right _ (List.mem_singleton_self _))
      rw [namedEpoch_eq, callsOf_eq] at hm
      exact hm hs u hu

/-- The passes `distinctCalls` accepts from the first, after none, meet `PassesDistinct`. -/
theorem passesDistinct_of_distinctCalls {nm : String} {ms : List Json} (h : distinctCalls nm 0 [] ms = .ok ()) :
    PassesDistinct ms := by
  obtain ⟨r, -, p⟩ := distinctCalls_ok ms h
  refine ⟨r, List.Pairwise.imp (fun hp he u hu hu' => hp ?_ u hu' hu) p⟩
  rw [he]
  exact sameEpoch_self _

theorem creditDistinct : CreditDistinct PouwCheck.pearlCPeriodCredit := by
  intro key ledger pins main name passes earlier epochs credited h
  unfold PouwCheck.pearlCPeriodCredit at h
  obtain ⟨r, -, h⟩ := bind_ok.1 h
  obtain ⟨a, -, h⟩ := bind_ok.1 h
  cases a
  try dsimp only at h
  obtain ⟨b, hd, h⟩ := bind_ok.1 h
  cases b
  try dsimp only at h
  have hc : r = (epochs, credited) := by
    first
    | (cases h; rfl)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h; exact h)
  subst hc
  try dsimp only at hd
  exact passesDistinct_of_distinctCalls hd

end Properties.Pouw.LiveSalt
