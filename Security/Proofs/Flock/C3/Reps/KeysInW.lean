import Proofs.Flock.C3.Reps.WithReads

/-!
# `wKey` from w's names: the Bool check `(keys m).all (wKey sh)` by the keys' construction

`wKey_of_mem`: a key of w's names (`RecAlgebra.wNames`) that is neither a ring switch's transpose nor a comb element
passes `wKey`. The rest place each message element w absorbs outside Ligerito's rounds in w's names, at the shape's
counts: the zerocheck's (`ab`, `c`, `g1`, `ginf`, `fa`, `fb`, at `m`), the lincheck's (`q1`, `qinf`, `zp`, at `k_log`),
`h` at a holo shape, the claims' ring switches (`rs`), and the running sums. So SentKeys' message keys pass the check
at the rep's own shape: the zerocheck's at `m` (`zcKeys_all`), the lincheck's at `k_log` and a holo shape
(`lcKeys_all`), and each claim's ring switch (`rsKeys_all`).
-/

namespace FlockC3.Reps

theorem mem_wNames {sh : Flock.Shape} {r : Array Flock.RecAlgebra.Key} {x : Flock.RecAlgebra.Key}
    (hr : r ∈ Flock.RecAlgebra.rounds sh) (hx : x ∈ r) : x ∈ Flock.RecAlgebra.wNames sh :=
  Array.mem_append_left _ (Array.mem_flatten.2 ⟨r, hr, hx⟩)

/-- **A key of w's names passes `wKey`**, but a transpose or a comb element. -/
theorem wKey_of_mem {sh : Flock.Shape} {x : Flock.RecAlgebra.Key} (h : x ∈ Flock.RecAlgebra.wNames sh)
    (h1 : ∀ k u, x ≠ .rsT k u) (h2 : ∀ l, x ≠ .comb l) : wKey sh x = true := by
  have hs : (Flock.RecAlgebra.wi sh x).isSome = true := by
    unfold Flock.RecAlgebra.wi
    exact Array.isSome_idxOf?.2 h
  cases x with
  | rsT k u => exact (h1 k u rfl).elim
  | comb l => exact (h2 l rfl).elim
  | cons => rfl
  | _ => exact hs

theorem ab_mem (sh : Flock.Shape) {i : ℕ} (hi : i < 64) : Flock.RecAlgebra.Key.ab i ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := (Array.range 64).map .ab ++ (Array.range 64).map .c) (by simp [Flock.RecAlgebra.rounds])
    (by simp [hi])

theorem c_mem (sh : Flock.Shape) {i : ℕ} (hi : i < 64) : Flock.RecAlgebra.Key.c i ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := (Array.range 64).map .ab ++ (Array.range 64).map .c) (by simp [Flock.RecAlgebra.rounds])
    (by simp [hi])

theorem g1_mem {sh : Flock.Shape} {i : ℕ} (hi : i < sh.m - 6) :
    Flock.RecAlgebra.Key.g1 i ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := #[.g1 i, .ginf i]) (by simp [Flock.RecAlgebra.rounds, hi]) (by simp)

theorem ginf_mem {sh : Flock.Shape} {i : ℕ} (hi : i < sh.m - 6) :
    Flock.RecAlgebra.Key.ginf i ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := #[.g1 i, .ginf i]) (by simp [Flock.RecAlgebra.rounds, hi]) (by simp)

theorem fa_mem (sh : Flock.Shape) : Flock.RecAlgebra.Key.fa ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := #[.fa, .fb]) (by simp [Flock.RecAlgebra.rounds]) (by simp)

theorem fb_mem (sh : Flock.Shape) : Flock.RecAlgebra.Key.fb ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := #[.fa, .fb]) (by simp [Flock.RecAlgebra.rounds]) (by simp)

theorem q1_mem {sh : Flock.Shape} {j : ℕ} (hj : j < sh.kLog - 6) :
    Flock.RecAlgebra.Key.q1 j ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := #[.q1 j, .qinf j]) (by simp [Flock.RecAlgebra.rounds, hj]) (by simp)

theorem qinf_mem {sh : Flock.Shape} {j : ℕ} (hj : j < sh.kLog - 6) :
    Flock.RecAlgebra.Key.qinf j ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := #[.q1 j, .qinf j]) (by simp [Flock.RecAlgebra.rounds, hj]) (by simp)

theorem zp_mem (sh : Flock.Shape) {l : ℕ} (hl : l < 64) : Flock.RecAlgebra.Key.zp l ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := (Array.range 64).map .zp) (by simp [Flock.RecAlgebra.rounds]) (by simp [hl])

theorem h_mem {sh : Flock.Shape} (hh : sh.holo = true) {l : ℕ} (hl : l < 64) :
    Flock.RecAlgebra.Key.h l ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := (Array.range 64).map .h) (by simp [Flock.RecAlgebra.rounds, Flock.RecAlgebra.hRound, hh])
    (by simp [hl])

theorem rs_mem {sh : Flock.Shape} {k u : ℕ} (hk : k < sh.claims) (hu : u < 128) :
    Flock.RecAlgebra.Key.rs k u ∈ Flock.RecAlgebra.wNames sh :=
  mem_wNames (r := (Array.range 128).map (.rs k))
    (by
      unfold Flock.RecAlgebra.rounds
      exact Array.mem_append_left _ (Array.mem_append_right _ (Array.mem_map.2 ⟨k, by simpa, rfl⟩)))
    (by simp [hu])

theorem acc0_mem (sh : Flock.Shape) {r : ℕ} (hr : r < 2) :
    Flock.RecAlgebra.Key.acc0 r ∈ Flock.RecAlgebra.wNames sh := by
  unfold Flock.RecAlgebra.wNames
  refine Array.mem_append_right _ ?_
  match r, hr with
  | 0, _ => simp [Flock.RecAlgebra.accNames]
  | 1, _ => simp [Flock.RecAlgebra.accNames]

theorem acc1_mem (sh : Flock.Shape) {r : ℕ} (hr : r < 2) :
    Flock.RecAlgebra.Key.acc1 r ∈ Flock.RecAlgebra.wNames sh := by
  unfold Flock.RecAlgebra.wNames
  refine Array.mem_append_right _ ?_
  match r, hr with
  | 0, _ => simp [Flock.RecAlgebra.accNames]
  | 1, _ => simp [Flock.RecAlgebra.accNames]

/-- **The zerocheck's keys pass `wKey`** at the shape's `m`. -/
theorem zcKeys_all (sh : Flock.Shape) (m : ℕ) : (zcKeys sh.m m).all (wKey sh) = true := by
  refine List.all_eq_true.2 fun x hx => ?_
  rcases m with _ | i
  · rcases List.mem_append.1 hx with hx | hx <;> obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx
    · exact wKey_of_mem (ab_mem sh (List.mem_range.1 hi)) (fun _ _ e => by cases e) fun _ e => by cases e
    · exact wKey_of_mem (c_mem sh (List.mem_range.1 hi)) (fun _ _ e => by cases e) fun _ e => by cases e
  · by_cases hi : i < sh.m - 6
    · rw [zcKeys, ite_eq_left hi] at hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact wKey_of_mem (g1_mem hi) (fun _ _ e => by cases e) fun _ e => by cases e
      · rw [List.mem_singleton.1 hx]
        exact wKey_of_mem (ginf_mem hi) (fun _ _ e => by cases e) fun _ e => by cases e
    · rw [zcKeys, ite_eq_right hi] at hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact wKey_of_mem (fa_mem sh) (fun _ _ e => by cases e) fun _ e => by cases e
      · rw [List.mem_singleton.1 hx]
        exact wKey_of_mem (fb_mem sh) (fun _ _ e => by cases e) fun _ e => by cases e

/-- **The lincheck's keys, with `h`, pass `wKey`** at a holo shape's `k_log`. -/
theorem lcKeys_all {sh : Flock.Shape} (hh : sh.holo = true) (m : ℕ) : (lcKeys sh.kLog m).all (wKey sh) = true := by
  refine List.all_eq_true.2 fun x hx => ?_
  by_cases h1 : m < sh.kLog - 6
  · rw [lcKeys, ite_eq_left h1] at hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact wKey_of_mem (q1_mem h1) (fun _ _ e => by cases e) fun _ e => by cases e
    · rw [List.mem_singleton.1 hx]
      exact wKey_of_mem (qinf_mem h1) (fun _ _ e => by cases e) fun _ e => by cases e
  · by_cases h2 : m = sh.kLog - 6
    · rw [lcKeys, ite_eq_right h1, ite_eq_left h2] at hx
      obtain ⟨l, hl, rfl⟩ := List.mem_map.1 hx
      exact wKey_of_mem (zp_mem sh (List.mem_range.1 hl)) (fun _ _ e => by cases e) fun _ e => by cases e
    · rw [lcKeys, ite_eq_right h1, ite_eq_right h2] at hx
      obtain ⟨l, hl, rfl⟩ := List.mem_map.1 hx
      exact wKey_of_mem (h_mem hh (List.mem_range.1 hl)) (fun _ _ e => by cases e) fun _ e => by cases e

/-- **A claim's ring-switch keys pass `wKey`.** -/
theorem rsKeys_all {sh : Flock.Shape} {k : ℕ} (hk : k < sh.claims) : (rsKeys k).all (wKey sh) = true :=
  List.all_eq_true.2 fun _ hx => by
    obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hx
    exact wKey_of_mem (rs_mem hk (List.mem_range.1 hu)) (fun _ _ e => by cases e) fun _ e => by cases e

end FlockC3.Reps

#print axioms FlockC3.Reps.mem_wNames
#print axioms FlockC3.Reps.wKey_of_mem
#print axioms FlockC3.Reps.ab_mem
#print axioms FlockC3.Reps.c_mem
#print axioms FlockC3.Reps.g1_mem
#print axioms FlockC3.Reps.ginf_mem
#print axioms FlockC3.Reps.fa_mem
#print axioms FlockC3.Reps.fb_mem
#print axioms FlockC3.Reps.q1_mem
#print axioms FlockC3.Reps.qinf_mem
#print axioms FlockC3.Reps.zp_mem
#print axioms FlockC3.Reps.h_mem
#print axioms FlockC3.Reps.rs_mem
#print axioms FlockC3.Reps.acc0_mem
#print axioms FlockC3.Reps.acc1_mem
#print axioms FlockC3.Reps.zcKeys_all
#print axioms FlockC3.Reps.lcKeys_all
#print axioms FlockC3.Reps.rsKeys_all
