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
* `Law.bernoulli_exec_escape_le`: over a uniform stream of `L` bytes, the executable returns a draw missing `B` with
  probability at most `Law.bernoulli`'s escape, `((den − num)/den)^|B|` (`Law.bernoulli_escape`).
* `Law.bernoulli_append`: a draw that succeeds is the same on a longer stream, so `Flock.Draw.onWith`'s extend-and-rerun
  never changes it; `Law.drawOS_bernoulli_escape_le`, `Law.execOS_bernoulli_escape_le`: the live draw
  (`Flock.Draw.drawWith`, `Law.runOS`) on a uniform byte source (A3) escapes no more often, as `ExecDrawOS.lean` proves
  for the other three laws.
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
      ∃ D', bernL num den (L + 1) l ((bsOf f).drop pos) = some D' ∧ ∀ u, u ∈ D ↔ u ∈ out ∨ u ∈ D' := by
  intro l
  induction l with
  | nil =>
    intro out pos _ D h
    simp only [bernLoop, Option.some.injEq] at h
    subst h
    exact ⟨[], rfl, fun u => by simp⟩
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
      refine ⟨if r < num then u :: D' else D', ?_, fun v => ?_⟩
      · simp only [bernL, hu]
        rw [hrest, hD', Option.map_some]
      · rw [hmem v]
        split_ifs <;> simp [Array.mem_push, or_assoc, or_left_comm]

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

theorem length_filter_range' (n : ℕ) (B : Finset ℕ) (hB : ∀ u ∈ B, u < n) :
    ((List.range' 0 n).filter (· ∈ B)).length = B.card := by
  classical
  have h1 : ((Finset.range n).filter (· ∈ B)) = B := by
    ext u; simp only [Finset.mem_filter, Finset.mem_range]; exact ⟨fun h => h.2, fun h => ⟨hB u h, h⟩⟩
  rw [← List.range_eq_range']
  conv_rhs => rw [← h1]
  rw [Finset.card_def, Finset.filter_val, Finset.range_val, Multiset.range, Multiset.filter_coe, Multiset.coe_card]

theorem bernLoop_append (num den : ℕ) (a b : ByteArray) : ∀ (l : List ℕ) (out : Array ℕ) (pos : ℕ) D,
    bernLoop num den a l out pos = some D → bernLoop num den (a ++ b) l out pos = some D := by
  intro l
  induction l with
  | nil => intro out pos D h; exact h
  | cons u us ih =>
    intro out pos D h
    simp only [bernLoop] at h ⊢
    cases hu : uniform den a (a.size + 1) pos with
    | none => rw [hu] at h; exact absurd h (by simp)
    | some rp =>
      obtain ⟨r, p⟩ := rp
      rw [hu] at h
      rw [Law.uniform_append den a b _ pos r p hu _ (by rw [ByteArray.size_append]; omega)]
      exact ih _ _ _ h

end FlockSoundness.Audit.DrawExec

namespace FlockSoundness.Audit.Law

open Game Finset
open scoped ENNReal

/-- **A `bernoulli` draw that succeeds is the same on a longer stream**: extending the stream and rerunning changes
nothing once the draw has succeeded. -/
theorem bernoulli_append (num den n : ℕ) (a b : ByteArray) (D : Array ℕ)
    (h : Flock.Draw.bernoulli num den n a = some D) : Flock.Draw.bernoulli num den n (a ++ b) = some D := by
  by_cases h0 : num = 0
  · subst h0; simpa [Flock.Draw.bernoulli] using h
  by_cases h1 : num = den
  · subst h1; simpa [Flock.Draw.bernoulli, h0] using h
  rw [DrawExec.bernoulli_eq num den n _ h0 h1] at h ⊢
  exact DrawExec.bernLoop_append num den a b _ _ _ D h

theorem bernoulli_escape_empty {n num den : ℕ} [NeZero den] (h : num ≤ den) : (bernoulli n num den).escape ∅ = 1 := by
  rw [bernoulli_escape h, Finset.card_empty, pow_zero]

/-- **The executable Bernoulli draw escapes no more often than the model's** (`Law.bernoulli`), on a uniform stream of
`L` bytes. -/
theorem bernoulli_exec_escape_le (n num den L : ℕ) [NeZero den] (h : num ≤ den) (B : Finset (Fin n)) :
    prCoin (fun f : Fin L → Fin 256 =>
        ∃ D, Flock.Draw.bernoulli num den n (streamOf f) = some D ∧ ∀ u ∈ B, u.val ∉ D) ≤
      (bernoulli n num den).escape B := by
  classical
  have hden : 0 < den := Nat.pos_of_ne_zero (NeZero.ne den)
  rcases B.eq_empty_or_nonempty with rfl | ⟨b, hb⟩
  · rw [bernoulli_escape_empty h]; exact DrawExec.prCoin_le_one _
  by_cases h0 : num = 0
  · subst h0
    rw [bernoulli_escape h]
    refine (DrawExec.prCoin_le_one _).trans (le_of_eq ?_)
    rw [bernFactor, Nat.cast_zero, zero_mul, zero_add, Nat.sub_zero,
      ENNReal.div_self (by exact_mod_cast hden.ne') (ENNReal.natCast_ne_top _), one_pow]
  by_cases h1 : num = den
  · subst h1
    refine le_trans (le_of_eq ?_) zero_le
    convert prCoin_false (C := Fin L → Fin 256) using 1
    refine congrArg prCoin (funext fun f => propext ⟨fun ⟨D, hD, hmiss⟩ => ?_, False.elim⟩)
    simp [Flock.Draw.bernoulli, h0] at hD
    subst hD
    exact hmiss b hb (Array.mem_range.2 b.isLt)
  set B' : Finset ℕ := B.image Fin.val with hB'
  have hB'n : ∀ u ∈ B', u < n := fun u hu => by obtain ⟨v, _, rfl⟩ := mem_image.1 hu; exact v.isLt
  have hcard : B'.card = B.card := card_image_of_injective _ Fin.val_injective
  calc prCoin (fun f : Fin L → Fin 256 =>
          ∃ D, Flock.Draw.bernoulli num den n (streamOf f) = some D ∧ ∀ u ∈ B, u.val ∉ D)
      ≤ prCoin (fun f : Fin L → Fin 256 => ∃ D, DrawExec.bernL num den (L + 1) (List.range' 0 n)
          (DrawExec.bsOf f) = some D ∧ ∀ u ∈ B', u ∉ D) := by
        refine prCoin_mono fun f ⟨D, hD, hmiss⟩ => ?_
        rw [DrawExec.bernoulli_eq num den n _ h0 h1] at hD
        obtain ⟨D', hD', hmem⟩ := DrawExec.bernLoop_streamOf f num den _ #[] 0 (Nat.zero_le _) D hD
        rw [List.drop_zero] at hD'
        refine ⟨D', hD', fun u hu huD => ?_⟩
        obtain ⟨v, hv, rfl⟩ := mem_image.1 hu
        exact hmiss v hv ((hmem _).2 (Or.inr huD))
    _ ≤ (((den - num : ℕ) : ℝ≥0∞) / den) ^ ((List.range' 0 n).filter (· ∈ B')).length :=
        DrawExec.bernL_le num den (L + 1) hden B' _ L
    _ = (bernoulli n num den).escape B := by
        rw [DrawExec.length_filter_range' n B' hB'n, hcard, bernoulli_escape h, bernFactor, mul_zero, zero_add]

theorem drawWith_bernoulli_run (get : ℕ → ℕ → StateT (ℕ → ℕ) Option ByteArray) (num den pop : ℕ) (s : ℕ → ℕ) :
    (Flock.Draw.drawWith get (.bernoulli num den) pop).run s =
      ((Flock.Draw.onWith (get 0) (2 * pop * Flock.Draw.width den + 64)
        (Flock.Draw.bernoulli num den pop)).run s).map fun p =>
        (p.1.map fun units => ({ law := .bernoulli num den, population := pop, units } : Flock.Draw.UnitDraw), p.2) := by
  unfold Flock.Draw.drawWith
  rw [StateT.run_bind]
  cases (Flock.Draw.onWith (get 0) (2 * pop * Flock.Draw.width den + 64) (Flock.Draw.bernoulli num den pop)).run s <;>
    rfl

/-- **The live Bernoulli draw escapes no more often than the model's** (`Law.bernoulli`), extend-and-rerun included, on a
uniform byte source (A3). -/
theorem drawOS_bernoulli_escape_le (n num den : ℕ) [NeZero den] (h : num ≤ den) {Ω : Type} [Fintype Ω] (L : Fin 1 → ℕ)
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) (B : Finset (Fin n)) :
    prCoin (fun ω => ∃ d, runOS (t ω) (.bernoulli num den) n = some d ∧ ∀ u ∈ B, u.val ∉ d.units) ≤
      (bernoulli n num den).escape B := by
  classical
  rw [ht fun f => ∃ d, runOS f (.bernoulli num den) n = some d ∧ ∀ u ∈ B, u.val ∉ d.units]
  calc prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 =>
          ∃ d, runOS f (.bernoulli num den) n = some d ∧ ∀ u ∈ B, u.val ∉ d.units)
      ≤ prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => ∀ s ∈ (univ : Finset (Fin 1)),
          (fun _ (g : Fin (L s) → Fin 256) =>
            ∃ D, Flock.Draw.bernoulli num den n (streamOf g) = some D ∧ ∀ u ∈ B, u.val ∉ D) s (f s)) := by
        refine prCoin_mono fun f ⟨d, hd, hmiss⟩ s _ => ?_
        obtain rfl : s = 0 := Subsingleton.elim _ _
        refine ⟨d.units, ?_, hmiss⟩
        unfold runOS at hd
        rw [drawWith_bernoulli_run] at hd
        cases ho : (Flock.Draw.onWith (tapeGet (tapesOf f) 0) (2 * n * Flock.Draw.width den + 64)
            (Flock.Draw.bernoulli num den n)).run (fun _ => 0) with
        | none => rw [ho] at hd; exact absurd hd (by simp)
        | some p =>
          obtain ⟨u, pos'⟩ := p
          rw [ho] at hd
          cases u with
          | none => exact absurd hd (by simp)
          | some D =>
            have hD : D = d.units := by rw [← Option.some.inj hd]
            have h1 := (onWith_tape (tapesOf f) 0 _ (by omega) _ (fun a b u h => bernoulli_append _ _ _ a b u h) _
              pos' _ rfl ho).2 D rfl
            have h2 : tapesOf f 0 = streamOf (f 0) := by simp [tapesOf, Array.getD_eq_getD_getElem?]
            rw [h2] at h1
            rw [← hD]; exact h1
    _ = ∏ s : Fin 1, prCoin (fun g : Fin (L s) → Fin 256 =>
          ∃ D, Flock.Draw.bernoulli num den n (streamOf g) = some D ∧ ∀ u ∈ B, u.val ∉ D) :=
        prCoin_pi_forall (Ω := fun s => Fin (L s) → Fin 256) univ (fun _ (g : Fin (L _) → Fin 256) =>
          ∃ D, Flock.Draw.bernoulli num den n (streamOf g) = some D ∧ ∀ u ∈ B, u.val ∉ D)
    _ ≤ (bernoulli n num den).escape B := by
        rw [Fin.prod_univ_one]
        exact bernoulli_exec_escape_le n num den (L 0) h B

/-- **The live Bernoulli draw, as a law, escapes no more often than `Law.bernoulli`**, on a uniform byte source (A3). -/
theorem execOS_bernoulli_escape_le (n num den : ℕ) [NeZero den] (h : num ≤ den) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (L : Fin 1 → ℕ) (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t)
    (B : Finset (Fin n)) :
    (execOS (n := n) t (.bernoulli num den) n).escape B ≤ (bernoulli n num den).escape B :=
  execOS_escape_le_of t _ n B _ (fun hB => by rw [hB, bernoulli_escape_empty h])
    (drawOS_bernoulli_escape_le n num den h L t ht B)

end FlockSoundness.Audit.Law
