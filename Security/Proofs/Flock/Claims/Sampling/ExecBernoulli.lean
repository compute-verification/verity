import Proofs.Flock.Soundness.ExecDrawOS

/-!
# The executable Bernoulli draw escapes no more often than the model's

`Flock.Draw.bernoulli num den n src` is the verifier's executable Bernoulli sampler: for each unit `u < n` in order, one
exact uniform draw `r ∈ [0, den)` from the stream (`Flock.Draw.uniform`, rejection sampling on big-endian words), and
`u` is drawn when `r < num`. `num = 0` draws nothing and `num = den` every unit, without reading the stream.

* `bernoulli_eq`: the loop as a list recursion (`bernLoop`), through `Std.Legacy.Range.forIn_eq_forIn_range'`.
* `bernLoop_streamOf`, `bernL_le`: on a stream of uniform bytes the loop reads, unit by unit, `uniformL` on the rest of
  the stream (`DrawExec.uniformL_le`), so a unit of `B` is missed with probability `(den − num)/den` independently of
  what came before, and `B` with at most `((den − num)/den)^|B|`.
-/

namespace FlockSoundness.Audit.DrawExec

open Flock.Draw Game
open scoped ENNReal

/-- `bernoulli`'s loop as a list recursion: the units drawn so far `out`, and the stream position `pos`. -/
def bernLoop (num den : ℕ) (src : ByteArray) : List ℕ → Array ℕ → ℕ → Option (Array ℕ)
  | [], out, _ => some out
  | u :: us, out, pos => match uniform den src (src.size + 1) pos with
    | none => none
    | some (r, p) => bernLoop num den src us (if r < num then out.push u else out) p

/-- The loop's state: an early return, the units drawn so far and the stream position. -/
abbrev BSt := Option (Option (Array ℕ)) × Array ℕ × ℕ

/-- A `forIn` whose body draws with `uniform` and keeps the unit when the draw is below `num`, followed by a
continuation that returns the early exit or finishes: the loop computes `bernLoop`. -/
theorem forIn_bern (num den : ℕ) (src : ByteArray) (F : ℕ → BSt → Id (ForInStep BSt))
    (POST : BSt → Id (Option (Array ℕ)))
    (hFn : ∀ u s, uniform den src (src.size + 1) s.2.2 = none → F u s = pure (.done (some none, s.2.1, s.2.2)))
    (hFs : ∀ u s r p, uniform den src (src.size + 1) s.2.2 = some (r, p) →
      F u s = pure (.yield (none, if r < num then s.2.1.push u else s.2.1, p)))
    (hPs : ∀ s r, s.1 = some r → POST s = pure r) (hPn : ∀ s, s.1 = none → POST s = pure (some s.2.1)) :
    ∀ (l : List ℕ) (out : Array ℕ) (pos : ℕ),
      (forIn (m := Id) l ((none : Option (Option (Array ℕ))), out, pos) F >>= POST).run = bernLoop num den src l out pos := by
  intro l
  induction l with
  | nil =>
    intro out pos
    simp only [List.forIn_nil, bernLoop]
    rw [pure_bind, hPn _ rfl]
    rfl
  | cons u us ih =>
    intro out pos
    rw [List.forIn_cons]
    cases h : uniform den src (src.size + 1) pos with
    | none =>
      rw [hFn u _ h]
      simp only [bernLoop, h, pure_bind]
      rw [hPs _ none rfl]
      rfl
    | some rp =>
      obtain ⟨r, p⟩ := rp
      rw [hFs u _ r p h]
      simp only [bernLoop, h, pure_bind]
      exact ih _ _

theorem bernoulli_eq (num den n : ℕ) (src : ByteArray) (h0 : num ≠ 0) (h1 : num ≠ den) :
    bernoulli num den n src = bernLoop num den src (List.range' 0 n) #[] 0 := by
  unfold bernoulli
  simp only [beq_iff_eq, h0, h1, ite_false]
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:n] : Std.Legacy.Range).size = n := by
    show (n - 0 + 1 - 1) / 1 = n
    omega
  rw [hsize]
  refine forIn_bern num den src _ _ ?_ ?_ ?_ ?_ (List.range' 0 n) #[] 0
  · intro u s h; simp only [h]
  · intro u s r p h; simp only [h]; split_ifs <;> rfl
  · intro s r h; simp only [h]
  · intro s h; simp only [h]

/-! ## The loop on the rest of the stream -/

/-- `bernLoop` on the rest of the stream, with `uniformL`'s fuel `F`: the units it draws. -/
def bernL (num den F : ℕ) : List ℕ → List UInt8 → Option (List ℕ)
  | [], _ => some []
  | u :: us, bs => match uniformL den F bs with
    | none => none
    | some (r, rest) => (bernL num den F us rest).map fun D => if r < num then u :: D else D

theorem bernLoop_streamOf {L : ℕ} (f : Fin L → Fin 256) (num den : ℕ) :
    ∀ (l : List ℕ) (out : Array ℕ) (pos : ℕ), pos ≤ L → ∀ D, bernLoop num den (Law.streamOf f) l out pos = some D →
      ∃ D', bernL num den (L + 1) l ((bsOf f).drop pos) = some D' ∧ ∀ u, u ∈ D → u ∈ out ∨ u ∈ D' := by
  intro l
  induction l with
  | nil =>
    intro out pos _ D h
    simp only [bernLoop, Option.some.injEq] at h
    subst h
    exact ⟨[], rfl, fun u hu => Or.inl hu⟩
  | cons u us ih =>
    intro out pos hpos D h
    have hU := uniform_streamOf f den (L + 1) pos hpos
    simp only [bernLoop, size_streamOf] at h
    revert hU
    cases hu : uniformL den (L + 1) ((bsOf f).drop pos) with
    | none =>
      intro hU
      simp only at hU
      rw [hU] at h
      exact absurd h (by simp)
    | some rr =>
      obtain ⟨r, rest⟩ := rr
      intro hU
      simp only at hU
      obtain ⟨p, hp, hrest, hU⟩ := hU
      rw [hU] at h
      simp only at h
      obtain ⟨D', hD', hmem⟩ := ih _ p hp D h
      refine ⟨if r < num then u :: D' else D', ?_, fun v hv => ?_⟩
      · simp only [bernL, hu]
        rw [hrest, hD', Option.map_some]
      · rcases hmem v hv with hv | hv
        · split_ifs at hv with hr
          · rcases Array.mem_push.1 hv with hv | rfl
            · exact Or.inl hv
            · exact Or.inr (by simp [hr])
          · exact Or.inl hv
        · right
          split_ifs <;> simp [hv]

open Classical in
/-- **Each unit of `B` is missed independently with probability `(den − num)/den`**: on a uniform stream, `bernL` returns
a draw missing `B` with probability at most `((den − num)/den)` to the number of `l`'s entries in `B`. -/
theorem bernL_le (num den F : ℕ) (hden : 0 < den) (B : Finset ℕ) :
    ∀ (l : List ℕ) ℓ, prCoin (fun g : Fin ℓ → Fin 256 => ∃ D, bernL num den F l (bsOf g) = some D ∧ ∀ u ∈ B, u ∉ D) ≤
      (((den - num : ℕ) : ℝ≥0∞) / den) ^ (l.filter (· ∈ B)).length := by
  intro l
  induction l with
  | nil => intro ℓ; simp only [List.filter_nil, List.length_nil, pow_zero]; exact prCoin_le_one _
  | cons u us ih =>
    intro ℓ
    set q : ℝ≥0∞ := ((den - num : ℕ) : ℝ≥0∞) / den with hq
    set c := (us.filter (· ∈ B)).length with hc
    have hden0 : (den : ℝ≥0∞) ≠ 0 := by exact_mod_cast hden.ne'
    have hmono : ∀ g : Fin ℓ → Fin 256, (∃ D, bernL num den F (u :: us) (bsOf g) = some D ∧ ∀ v ∈ B, v ∉ D) →
        ∃ r rest, uniformL den F (bsOf g) = some (r, rest) ∧ ((r < num → u ∉ B) ∧ ∃ D, bernL num den F us rest = some D ∧
          ∀ v ∈ B, v ∉ D) := by
      intro g ⟨D, hD, hmiss⟩
      simp only [bernL] at hD
      cases hu : uniformL den F (bsOf g) with
      | none => rw [hu] at hD; exact absurd hD (by simp)
      | some rr =>
        obtain ⟨r, rest⟩ := rr
        rw [hu] at hD
        simp only at hD
        cases hD' : bernL num den F us rest with
        | none => rw [hD'] at hD; exact absurd hD (by simp)
        | some D' =>
          rw [hD', Option.map_some, Option.some.injEq] at hD
          refine ⟨r, rest, rfl, fun hr huB => ?_, D', hD', fun v hv hvD => ?_⟩
          · exact hmiss u huB (by rw [← hD]; simp [hr])
          · refine hmiss v hv ?_
            rw [← hD]
            split_ifs
            · exact List.mem_cons_of_mem _ hvD
            · exact hvD
    have hQ : ∀ r < den, ∀ ℓ', prCoin (fun g : Fin ℓ' → Fin 256 => (r < num → u ∉ B) ∧
        ∃ D, bernL num den F us (bsOf g) = some D ∧ ∀ v ∈ B, v ∉ D) ≤ if r < num ∧ u ∈ B then 0 else q ^ c := by
      intro r _ ℓ'
      split_ifs with h
      · refine le_of_eq ?_
        convert prCoin_false (C := Fin ℓ' → Fin 256) using 2 with g
        simp [h.1, h.2]
      · exact (prCoin_mono fun g hg => hg.2).trans (ih ℓ')
    refine (prCoin_mono hmono).trans ((uniformL_le den hden _ _ hQ F ℓ).trans ?_)
    by_cases huB : u ∈ B
    · have hcnt : ((u :: us).filter (· ∈ B)).length = c + 1 := by simp [huB, hc]
      rw [hcnt]
      have hsplit : (∑ r ∈ Finset.range den, if r < num ∧ u ∈ B then (0 : ℝ≥0∞) else q ^ c) =
          ((den - num : ℕ) : ℝ≥0∞) * q ^ c := by
        rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, nsmul_eq_mul]
        congr 2
        have : (Finset.range den).filter (fun r => ¬ (r < num ∧ u ∈ B)) = Finset.Ico num den := by
          ext r; simp [huB]; omega
        rw [this, Nat.card_Ico]
      rw [hsplit, pow_succ, mul_comm (q ^ c), ENNReal.mul_div_right_comm]
    · have hcnt : ((u :: us).filter (· ∈ B)).length = c := by simp [huB, hc]
      rw [hcnt]
      have hsum : (∑ r ∈ Finset.range den, if r < num ∧ u ∈ B then (0 : ℝ≥0∞) else q ^ c) = den * q ^ c := by
        simp [huB]
      rw [hsum]
      exact ((ENNReal.eq_div_iff hden0 (ENNReal.natCast_ne_top _)).mpr rfl).symm.le

end FlockSoundness.Audit.DrawExec
