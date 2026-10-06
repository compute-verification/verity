import Proofs.Flock.Soundness.Discharge.DrawWords.Statements

/-!
# The deterministic lemma: accepted first words make the live run succeed

`words_run`: when each stratum's stream holds the `need` bytes `onWith` reads first and the first `k_s` words of each
stream are accepted at the first try, `onWith`'s first run of `subset` succeeds on every stratum, so `stratifiedWith`
returns a draw. The words fit in the first read: word `i < k` ends at `wordOff n (i + 1) ≤ k · width n ≤ need`
(`word_end_le_need`), since `width` is monotone.
-/

namespace FlockSoundness.Discharge.DrawWords

open FlockSoundness.Audit.Law FlockSoundness.Audit.DrawExec Finset

theorem width_mono {a b : ℕ} (h : a ≤ b) : Flock.Draw.width a ≤ Flock.Draw.width b := by
  unfold Flock.Draw.width
  have : Nat.log2 a ≤ Nat.log2 b := by
    rw [Nat.log2_eq_log_two, Nat.log2_eq_log_two]; exact Nat.log_mono_right h
  omega

theorem wordOff_zero (size : ℕ) : wordOff size 0 = 0 := by simp [wordOff]

theorem wordOff_succ (size i : ℕ) : wordOff size (i + 1) = wordOff size i + Flock.Draw.width (size - i) := by
  unfold wordOff; rw [Finset.sum_range_succ]

theorem wordOff_le (size i : ℕ) : wordOff size i ≤ i * Flock.Draw.width size := by
  unfold wordOff
  calc ∑ j ∈ range i, Flock.Draw.width (size - j) ≤ ∑ _j ∈ range i, Flock.Draw.width size :=
        Finset.sum_le_sum fun j _ => width_mono (Nat.sub_le _ _)
    _ = i * Flock.Draw.width size := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

/-- **Each of a stratum's first `k` words ends within the `need` bytes `onWith` reads first.** -/
theorem word_end_le_need (s : Flock.Draw.Stratum) {i : ℕ} (hi : i < s.k) :
    wordOff s.size i + Flock.Draw.width (s.size - i) ≤ s.need := by
  rw [← wordOff_succ]
  have h1 := wordOff_le s.size (i + 1)
  have h2 : (i + 1) * Flock.Draw.width s.size ≤ s.k * Flock.Draw.width s.size := Nat.mul_le_mul_right _ hi
  have h3 : s.k * Flock.Draw.width s.size ≤ 2 * s.k * Flock.Draw.width s.size := by
    rw [Nat.mul_assoc]; exact Nat.le_mul_of_pos_left _ (by norm_num)
  unfold Flock.Draw.Stratum.need
  omega

theorem uniform_accept (m : ℕ) (src : ByteArray) (fuel pos : ℕ) (hw : pos + Flock.Draw.width m ≤ src.size)
    (hx : Flock.Draw.beNat src pos (Flock.Draw.width m) <
      2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m) :
    Flock.Draw.uniform m src (fuel + 1) pos =
      some (Flock.Draw.beNat src pos (Flock.Draw.width m) % m, pos + Flock.Draw.width m) := by
  have hw' : ¬ pos + Flock.Draw.width m > src.size := by omega
  simp only [Flock.Draw.uniform, hw', ↓reduceIte, hx]

theorem loopSw_some (k n : ℕ) (src : ByteArray)
    (h : ∀ i < k, wordOff n i + Flock.Draw.width (n - i) ≤ src.size ∧ WordOK n src i) :
    ∀ c j (sw : Std.HashMap ℕ ℕ), j + c ≤ k → ∃ sw', loopSw n src (List.range' j c) sw (wordOff n j) = some sw' := by
  intro c
  induction c with
  | zero => intro j sw _; exact ⟨sw, rfl⟩
  | succ c ih =>
    intro j sw hjc
    obtain ⟨hw, hok⟩ := h j (by omega)
    rw [List.range'_succ]
    simp only [loopSw]
    rw [uniform_accept (n - j) src src.size (wordOff n j) hw hok, ← wordOff_succ]
    exact ih (j + 1) _ (by omega)

/-- **`subset k n` succeeds when its first `k` words are accepted at the first try.** -/
theorem subset_some (k n : ℕ) (src : ByteArray)
    (h : ∀ i < k, wordOff n i + Flock.Draw.width (n - i) ≤ src.size ∧ WordOK n src i) :
    ∃ D, Flock.Draw.subset k n src = some D := by
  obtain ⟨sw, hsw⟩ := loopSw_some k n src h k 0 ∅ (by omega)
  rw [wordOff_zero] at hsw
  rw [subset_eq, hsw]
  exact ⟨_, rfl⟩

theorem beNat_extract (a : ByteArray) (j off len : ℕ) (hj : j ≤ a.size) (h : off + len ≤ j) :
    Flock.Draw.beNat (a.extract 0 j) off len = Flock.Draw.beNat a off len := by
  conv_rhs => rw [extract_split a j hj]
  rw [beNat_append _ _ _ _ (by rw [ByteArray.size_extract]; omega)]

/-- **`onWith`'s first run, when it succeeds**: its first read and nothing more. -/
theorem onWith_first (T : ℕ → ByteArray) (i need : ℕ) (run : ByteArray → Option (Array ℕ)) (pos : ℕ → ℕ)
    (hle : pos i + need ≤ (T i).size) (u : Array ℕ) (hu : run ((T i).extract (pos i) (pos i + need)) = some u) :
    (Flock.Draw.onWith (tapeGet T i) need run).run pos = some (some u, Function.update pos i (pos i + need)) := by
  unfold Flock.Draw.onWith
  rw [StateT.run_bind]
  have hg : (tapeGet T i need).run pos =
      some ((T i).extract (pos i) (pos i + need), Function.update pos i (pos i + need)) := by
    simp only [tapeGet, StateT.run, hle, ↓reduceIte]
  rw [hg, Option.bind_eq_bind, Option.bind_some]
  dsimp only
  rw [StateT.run_bind]
  rw [show ∀ (b : Option (Option (Array ℕ)) × ByteArray)
      (f : Unit → Option (Option (Array ℕ)) × ByteArray →
        StateT (ℕ → ℕ) Option (ForInStep (Option (Option (Array ℕ)) × ByteArray))),
      (forIn Lean.Loop.mk b f : StateT (ℕ → ℕ) Option _) = Lean.Loop.forIn Lean.Loop.mk b f from fun _ _ => rfl,
    Lean.Loop.forIn_eq_of_monadTail]
  simp only [hu, StateT.run_bind, StateT.run_pure, Option.bind_eq_bind]
  rfl

/-- **`stratifiedWith` succeeds from unread streams when each stratum's first run of `subset` does**, on the tape's first
`need` bytes. -/
theorem stratifiedWith_some (strata : Array Flock.Draw.Stratum) (T : ℕ → ByteArray)
    (h : ∀ i < strata.size, strata[i]!.need ≤ (T i).size ∧
      ∃ u, Flock.Draw.subset strata[i]!.k strata[i]!.size ((T i).extract 0 strata[i]!.need) = some u) :
    ∃ D pos', (Flock.Draw.stratifiedWith (tapeGet T) strata).run (fun _ => 0) = some (some D, pos') := by
  unfold Flock.Draw.stratifiedWith
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl]
  have loop : ∀ (l : List ℕ), l.Nodup → (∀ j ∈ l, j < strata.size) → ∀ (out : Array ℕ) (pos : ℕ → ℕ),
      (∀ j ∈ l, pos j = 0) → ∃ out' pos₁,
      (forIn (m := StateT (ℕ → ℕ) Option) l ((none : Option (Option (Array ℕ))), out) (fun i __s => do
          let __x ← Flock.Draw.onWith (tapeGet T i) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = some ((none, out'), pos₁) := by
    intro l
    induction l with
    | nil => intro _ _ out pos _; exact ⟨out, pos, rfl⟩
    | cons i is ih =>
      intro hnd hlt out pos hpos
      obtain ⟨hle, u, hu⟩ := h i (hlt i List.mem_cons_self)
      have hpi : pos i = 0 := hpos i List.mem_cons_self
      have ho := onWith_first T i strata[i]!.need (Flock.Draw.subset strata[i]!.k strata[i]!.size) pos
        (by rw [hpi, Nat.zero_add]; exact hle) u (by rw [hpi, Nat.zero_add]; exact hu)
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind, ho]
      simp only [Option.bind_eq_bind, Option.bind_some, StateT.run_pure]
      refine ih (List.nodup_cons.1 hnd).2 (fun j hj => hlt j (List.mem_cons_of_mem _ hj)) _ _ fun j hj => ?_
      show Function.update pos i _ j = 0
      rw [Function.update_of_ne (fun (e : j = i) => (List.nodup_cons.1 hnd).1 (e ▸ hj))]
      exact hpos j (List.mem_cons_of_mem _ hj)
  obtain ⟨out', pos₁, h'⟩ := loop (List.range' 0 strata.size) List.nodup_range'
    (fun j hj => by rw [List.mem_range'_1] at hj; omega) #[] (fun _ => 0) (fun _ _ => rfl)
  rw [StateT.run_bind]
  cases hl : (forIn (m := StateT (ℕ → ℕ) Option) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run (fun _ => 0) with
  | none => exact absurd (hl.symm.trans h') (by simp)
  | some y =>
    obtain rfl := Option.some.inj (hl.symm.trans h')
    exact ⟨_, _, rfl⟩

theorem words_run : WordsRun := by
  intro m S K pop L f hneed hok
  unfold runOS
  rw [drawWith_stratified_run]
  obtain ⟨D, pos', hD⟩ := stratifiedWith_some (Array.ofFn S) (tapesOf f) fun i hi => by
    have hi' : i < m := by simpa using hi
    have hS : (Array.ofFn S)[i]! = S ⟨i, hi'⟩ := by simp [hi']
    have hT : tapesOf f i = streamOf (f ⟨i, hi'⟩) := by simp [tapesOf, Array.getD_eq_getD_getElem?, hi']
    rw [hS, hT, size_streamOf]
    refine ⟨hneed _, subset_some _ _ _ fun j hj => ?_⟩
    have hend := word_end_le_need (S ⟨i, hi'⟩) hj
    have hsz : ((streamOf (f ⟨i, hi'⟩)).extract 0 (S ⟨i, hi'⟩).need).size = (S ⟨i, hi'⟩).need := by
      rw [ByteArray.size_extract, size_streamOf]; have := hneed ⟨i, hi'⟩; omega
    refine ⟨by rw [hsz]; exact hend, ?_⟩
    unfold WordOK
    rw [beNat_extract _ _ _ _ (by rw [size_streamOf]; exact hneed _) hend]
    exact hok _ j hj
  rw [hD]
  simp only [Option.map_some]
  exact ⟨_, rfl⟩

end FlockSoundness.Discharge.DrawWords
