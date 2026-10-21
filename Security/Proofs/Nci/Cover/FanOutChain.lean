import Proofs.Nci.Cover.FanOutWalk

/-!
# The fan-out structure: one spent value per key chain

With `Keyed`'s (I) no two values share a key, so two key paths into one value are one path, the shorter a suffix of
the longer (`Keyed.follows_suffix`). With (F) no value is keyed to a value a block uses, a whole product's activation or
a held root's leg, so a key path from one such value passes into no other. Together: no value a block spends is a key
ancestor of another it spends (`Keyed.spent_eq_of_follows`). A block that cuts a spent value's word, reading it at one
bit, computes the rest from bits that no other spent value's key path holds.
-/

namespace Nci.Cover

namespace Follows

variable {Val : Type*} {key : Val → Option Val}

/-- Of two values that one key path reaches, one reaches the other. -/
theorem total {S : Finset Val} {x y z : Val} (hy : Follows key S x y) (hz : Follows key S x z) :
    Follows key S y z ∨ Follows key S z y := by
  induction hy generalizing z with
  | refl => exact Or.inl hz
  | step hx hk h ih =>
    cases hz with
    | refl => exact Or.inr (Follows.step hx hk h)
    | step _ hk' hz' =>
      rw [Option.some.inj (hk.symm.trans hk')] at ih
      exact ih hz'

/-- A key path that moves ends with a step: a value outside the stops keyed to its end. -/
theorem last_step {S : Finset Val} {a c : Val} (h : Follows key S a c) (hac : a ≠ c) :
    ∃ w, Follows key S a w ∧ w ∉ S ∧ key w = some c := by
  revert hac
  induction h with
  | refl => exact fun hac => absurd rfl hac
  | @step v κ u hv hk h ih =>
    intro _
    by_cases hκ : κ = u
    · subst hκ
      exact ⟨v, Follows.refl v, hv, hk⟩
    · obtain ⟨w, hw, hwS, hwk⟩ := ih hκ
      exact ⟨w, Follows.step hv hk hw, hwS, hwk⟩

end Follows

namespace Keyed

variable {C : Circuit} {J L Tok Val : Type*} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V f : ℕ}

theorem key_mem (K : Keyed M Tok Val d aOut c V f) {v κ : Val} (hv : v ∈ K.vals) (hs : v ∉ K.stops)
    (hk : K.key v = some κ) : κ ∈ K.vals := by
  obtain ⟨κ', hκ', hk', -⟩ := K.keyed v hv hs
  rw [Option.some.inj (hk.symm.trans hk')]
  exact hκ'

/-- A key path from a value, through stop sets containing the stops, stays among the values. -/
theorem follows_mem (K : Keyed M Tok Val d aOut c V f) {T : Finset Val} (hT : K.stops ⊆ T) {a w : Val}
    (ha : a ∈ K.vals) (h : Follows K.key T a w) : w ∈ K.vals := by
  revert ha
  induction h with
  | refl => exact id
  | step hv hk _ ih => exact fun ha => ih (K.key_mem ha (fun hs => hv (hT hs)) hk)

/-- **Two key paths into one value** (I): the start of one lies on the other. -/
theorem follows_suffix (K : Keyed M Tok Val d aOut c V f) {T T' : Finset Val} (hT : K.stops ⊆ T)
    (hT' : K.stops ⊆ T') {a b c : Val} (ha : a ∈ K.vals) (hb : b ∈ K.vals) (hbc : Follows K.key T b c)
    (hac : Follows K.key T' a c) : Follows K.key T b a ∨ Follows K.key T' a b := by
  revert hb hac
  induction hbc with
  | refl => exact fun _ hac => Or.inr hac
  | @step v κ u hv hk h ih =>
    intro hb hac
    rcases ih (K.key_mem hb (fun hs => hv (hT hs)) hk) hac with h1 | h1
    · exact Or.inl (Follows.step hv hk h1)
    · by_cases haκ : a = κ
      · rw [← haκ] at hk
        exact Or.inl (Follows.step hv hk (Follows.refl a))
      · obtain ⟨w, hw, -, hwk⟩ := Follows.last_step h1 haκ
        rw [K.key_inj w (K.follows_mem hT' ha hw) v hb κ hwk hk] at hw
        exact Or.inr hw

/-- **No spent value is a key ancestor of another** ((I) and (F)). -/
theorem spent_eq_of_follows [DecidableEq J] (K : Keyed M Tok Val d aOut c V f) {I : Finset (Fin C.N)} {u u' : Val}
    (hu : K.Spent I u) (hu' : K.Spent I u') (h : Follows K.key K.stops u u') : u = u' := by
  obtain ⟨hur, -⟩ := hu
  obtain ⟨hu'r, v', hv', hf'⟩ := hu'
  have hTs : K.stops ⊆ K.stops ∪ K.readSet I := Finset.subset_union_left
  have huT : u ∈ K.stops ∪ K.readSet I := Finset.mem_union_right _ hur
  have hu'T : u' ∈ K.stops ∪ K.readSet I := Finset.mem_union_right _ hu'r
  have huv : u ∈ K.vals := (Finset.mem_filter.1 hur).1
  have hv'v : v' ∈ K.vals := by
    rcases hv' with ⟨p, hp, rfl⟩ | ⟨r, hr, hq⟩
    · exact K.act_mem p (Finset.mem_filter.1 hp).1
    · exact (K.legs_mem r (Finset.mem_inter.1 hr).1 v' hq).1
  have hstart : ∀ w ∈ K.vals, K.key w ≠ some v' := by
    intro w hw hwk
    obtain ⟨hact, hleg⟩ := K.key_fresh w hw v' hwk
    rcases hv' with ⟨p, hp, hpv⟩ | ⟨r, hr, hq⟩
    · exact hact p (Finset.mem_filter.1 hp).1 hpv
    · exact hleg r (Finset.mem_inter.1 hr).1 hq
  rcases K.follows_suffix hTs subset_rfl huv hv'v hf' h with h1 | h1
  · rcases Follows.total h1 hf' with h2 | h2
    · exact (Follows.eq_of_mem huT h2).symm
    · exact Follows.eq_of_mem hu'T h2
  · by_cases huv' : u = v'
    · subst huv'
      exact (Follows.eq_of_mem huT hf').symm
    · obtain ⟨w, hw, -, hwk⟩ := Follows.last_step h1 huv'
      exact absurd hwk (hstart w (K.follows_mem subset_rfl huv hw))

end Keyed

end Nci.Cover
