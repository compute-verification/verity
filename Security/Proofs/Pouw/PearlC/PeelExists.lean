import Proofs.Pouw.PearlC.RowStats
import Proofs.Pouw.PearlC.HonestWords

/-!
# The peel's rows exist (proofs)

`Pc8EntryOK` asks that U's pinned order form `C̃_ij`'s word and the factor rows `P_A,i`, `P_B,j`
(`PeelExactProofsP.peelRowsP`): that no step, promotion, FMA or BF16 narrowing leaves the finite range. At a step
`⟨[32], w, z⟩` with `w ≥ 24` (sm_120's `sm120E4m3`, width 26) and `k ≤ 2^16` that holds on every input whose codes
are not NaN, by magnitudes:

* **A step** (`step_fin`, `HonestWords.step_abs_le`'s converse): from a finite accumulator of magnitude `x`, 32
  products of codes that are not NaN write a finite word of magnitude at most `x + 32·448² ≤ x + 2^23`.
* **A window and a chain** (`windowWordP_fin`, `chainWordP_fin`): a chain of `n < 2^24` atoms is finite and at most
  `n·2^23`. A promotion rounds a sum of words, on FP32's grid, below `(A + g)·2^23`, an FP32 value, so `rn` keeps the
  bound (`promote_fin`, as `HonestWords.foldl_promote_bound`).
* **The FMA chain** (`accChain_fin`): `n` FMAs of factors at most `B` and `B'` stay at most `(2^n − 1)·(2·B·B' + 1)`,
  each rounding at most doubling its value plus one.
* **The roundings** (`rnBf16_fin`, `addWord_fin`): a finite 32-bit word below `2^127` narrows to BF16 (its exponent
  field is at most 253), and a sum below `2^127` is finite.
* **The rows** (`peelRowsP_isSome`): at `k/32 ≤ 2^11` every chain is at most `2^34` and every FMA chain below `2^76`.
  `pc8EntryOK_of_wide` is `Pc8EntryOK` at such a step from `Noise8OK` alone: `A′`'s and `B̃`'s codes are E4M3 casts,
  never NaN; the bases' codes are `Noise8OK`'s; and `be8`'s words are BF16 words of `E′`'s codes, at most `448`.
-/

namespace Pouw.PearlC.PeelExists

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.PeelExact
  Pouw.PearlC.PeelExactP Matrix

/-! ## A step -/

/-- A term's magnitude: its significand at its exponent. -/
theorem abs_value (t : Term) : |t.value| = (t.mag : ℚ) * 2 ^ (t.exp - 23) := by
  unfold Term.value
  split <;> simp [abs_mul, abs_of_pos (two_zpow_pos _)]

/-- **A step from a bounded finite accumulator is finite**: at `⟨[32], w, z⟩` with `w ≥ 24`, on codes that are not
NaN, the word written is finite and at most `2^23` larger in magnitude than the accumulator. -/
theorem step_fin {w : ℕ} (hw : 24 ≤ w) (z : ℤ) {acc : ℕ} {x : ℚ} (hx : wordQ acc = some x) (a b : List Code)
    (ha : ∀ c ∈ a, c.isNaN = false) (hb : ∀ c ∈ b, c.isNaN = false) (hB : |x| + 2 ^ 23 < 2 ^ 127) :
    ∃ y, wordQ ((⟨[32], w, z⟩ : Pipeline).step acc a b) = some y ∧ |y| ≤ |x| + 2 ^ 23 := by
  obtain ⟨tc, htc, rfl⟩ := PeelExactProofs.wordQ_fin hx
  set L := (a.zip b).take 32 with hL
  have hLok : ∀ p ∈ L, p.1.isNaN = false ∧ p.2.isNaN = false := fun p hp =>
    have h := List.of_mem_zip (List.mem_of_mem_take hp)
    ⟨ha _ h.1, hb _ h.2⟩
  have hps : L.map (fun x => productTotal x.1 x.2) = L.map (fun x => Value.fin (product x.1 x.2)) :=
    List.map_congr_left fun p hp => by simp [productTotal, (hLok p hp).1, (hLok p hp).2]
  set ts : List Term := tc :: L.map (fun x => product x.1 x.2) with hts
  obtain ⟨hcan, habs⟩ := HonestWords.groupSum_le hw z ts
  have hsum : (ts.map fun t => |t.value|).sum ≤ |tc.value| + 2 ^ 23 := by
    rw [hts, List.map_cons, List.sum_cons]
    gcongr
    have hlen : (L.map fun x => product x.1 x.2).length ≤ 32 := by simp [hL]
    have hprod : ∀ t' ∈ L.map (fun x => product x.1 x.2), |t'.value| ≤ 448 ^ 2 := by
      intro t' ht'
      obtain ⟨⟨ca, cb⟩, hp, rfl⟩ := List.mem_map.1 ht'
      obtain ⟨h1, h2⟩ := hLok _ hp
      rw [productExact _ _ _ _ (val_eq h1) (val_eq h2), abs_mul]
      exact (mul_le_mul (val_abs_le h1) (val_abs_le h2) (abs_nonneg _) (by norm_num)).trans (by norm_num)
    generalize L.map (fun x => product x.1 x.2) = ps at hlen hprod
    calc (ps.map fun t => |t.value|).sum ≤ (ps.map fun _ => (448 ^ 2 : ℚ)).sum :=
          List.sum_le_sum fun t' ht' => hprod t' ht'
      _ = ps.length * 448 ^ 2 := by simp
      _ ≤ 32 * 448 ^ 2 := by gcongr; exact_mod_cast hlen
      _ ≤ 2 ^ 23 := by norm_num
  generalize ht : groupSum w z ts = t at hcan habs
  have hexp : ¬ (t.mag ≠ 0 ∧ 127 < t.exp) := by
    rintro ⟨hm, he⟩
    rcases hcan with h0 | ⟨-, -, h23 | h126⟩
    · exact hm h0
    · have h1 : (2 : ℚ) ^ 23 ≤ t.mag := by exact_mod_cast h23
      have h2 : (2 : ℚ) ^ (105 : ℤ) ≤ 2 ^ (t.exp - 23) := zpow_le_zpow_right₀ one_le_two (by omega)
      have h3 : (2 : ℚ) ^ 128 ≤ |t.value| := by
        rw [abs_value]
        calc (2 : ℚ) ^ 128 = 2 ^ 23 * 2 ^ (105 : ℤ) := by norm_num
          _ ≤ _ := mul_le_mul h1 h2 (two_zpow_pos _).le (by positivity)
      linarith [habs.trans hsum]
    · omega
  have hall : ∀ v ∈ (Value.fin tc :: L.map fun x => Value.fin (product x.1 x.2)), ∃ t, v = .fin t := by
    intro v hv
    rcases List.mem_cons.1 hv with rfl | hv
    · exact ⟨tc, rfl⟩
    · obtain ⟨p, -, rfl⟩ := List.mem_map.1 hv
      exact ⟨_, rfl⟩
  have hn : (Value.fin tc :: L.map fun x => Value.fin (product x.1 x.2)).any Value.isNaN = false :=
    List.any_eq_false.2 fun v hv => by obtain ⟨t, rfl⟩ := hall v hv; simp [Value.isNaN]
  have hi : ∀ s, (Value.fin tc :: L.map fun x => Value.fin (product x.1 x.2)).any (Value.isInf s) = false :=
    fun s => List.any_eq_false.2 fun v hv => by obtain ⟨t, rfl⟩ := hall v hv; simp [Value.isInf]
  have hf : (Value.fin tc :: L.map fun x => Value.fin (product x.1 x.2)).filterMap Value.finite? = ts := by
    simp [hts, List.filterMap_map, Function.comp_def, Value.finite?]
  have hstep : (⟨[32], w, z⟩ : Pipeline).step acc a b = packTerm t := by
    have hs : (⟨[32], w, z⟩ : Pipeline).step acc a b =
        packValue (groupStepTotal w z (fp32Value acc) (L.map fun x => productTotal x.1 x.2)) := rfl
    rw [hs, htc, hps]
    unfold groupStepTotal
    simp only [hn, hi, hf, ht, Bool.false_eq_true, ↓reduceIte, Bool.and_self, hexp]
    rfl
  exact ⟨t.value, by rw [hstep]; exact HonestWords.wordQ_packTerm_canon hcan fun hm => by omega, habs.trans hsum⟩

/-! ## A window and a chain -/

/-- No code of the atoms is NaN. -/
def AtomsOK (g : List AtomIn) : Prop := ∀ x ∈ g, ∀ l, (x.1 l).isNaN = false ∧ (x.2 l).isNaN = false

theorem windowFold_fin {w : ℕ} (hw : 24 ≤ w) (z : ℤ) :
    ∀ (g : List AtomIn) (acc : ℕ) (x : ℚ) (n : ℕ), AtomsOK g → wordQ acc = some x → |x| ≤ n * 2 ^ 23 →
      n + g.length < 2 ^ 24 →
      ∃ c cv, g.foldlM (fun v x => finite? ((⟨[32], w, z⟩ : Pipeline).step v (List.ofFn x.1) (List.ofFn x.2)))
          acc = some c ∧ wordQ c = some cv ∧ |cv| ≤ ((n + g.length : ℕ) : ℚ) * 2 ^ 23
  | [], acc, x, n, _, hx, hb, _ => ⟨acc, x, rfl, hx, by simpa using hb⟩
  | a :: g, acc, x, n, hok, hx, hb, hn => by
    simp only [List.length_cons] at hn
    have hn' : (n : ℚ) + 1 ≤ 2 ^ 24 := by exact_mod_cast (show n + 1 ≤ 2 ^ 24 by omega)
    have hB : |x| + 2 ^ 23 < 2 ^ 127 := by nlinarith
    obtain ⟨y, hy, hyb⟩ := step_fin hw z hx (List.ofFn a.1) (List.ofFn a.2)
      (fun c hc => by obtain ⟨l, rfl⟩ := List.mem_ofFn.1 hc; exact (hok a List.mem_cons_self l).1)
      (fun c hc => by obtain ⟨l, rfl⟩ := List.mem_ofFn.1 hc; exact (hok a List.mem_cons_self l).2) hB
    rw [List.foldlM_cons, RowCircuit.finite?_of_wordQ hy]
    obtain ⟨c, cv, hc, hcv, hcb⟩ := windowFold_fin hw z g _ y (n + 1)
      (fun x hx => hok x (List.mem_cons_of_mem _ hx)) hy (by push_cast; linarith) (by omega)
    refine ⟨c, cv, hc, hcv, ?_⟩
    rwa [show n + 1 + g.length = n + (g.length + 1) by omega] at hcb

/-- **A window is finite**: `n < 2^24` atoms whose codes are not NaN, stepped from `+0`, write a finite word of
magnitude at most `n·2^23`. -/
theorem windowWordP_fin {w : ℕ} (hw : 24 ≤ w) (z : ℤ) {g : List AtomIn} (hok : AtomsOK g) (hn : g.length < 2 ^ 24) :
    ∃ c cv, windowWordP? ⟨[32], w, z⟩ g = some c ∧ wordQ c = some cv ∧ |cv| ≤ g.length * 2 ^ 23 := by
  obtain ⟨c, cv, hc, hcv, hcb⟩ := windowFold_fin hw z g 0 0 0 hok Freeze.wordQ_zero (by simp) (by omega)
  exact ⟨c, cv, hc, hcv, by simpa using hcb⟩

/-- **A promotion of bounded words is finite**: words of magnitude at most `A·2^23` and `g·2^23`, `A + g < 2^24`,
promote to a word of magnitude at most `(A + g)·2^23`, since the sum is on FP32's grid and `rn` keeps an FP32 bound. -/
theorem promote_fin {t x : ℕ} {tv xv : ℚ} {A g : ℕ} (ht : wordQ t = some tv) (hx : wordQ x = some xv)
    (htb : |tv| ≤ A * 2 ^ 23) (hxb : |xv| ≤ g * 2 ^ 23) (hAg : A + g < 2 ^ 24) :
    ∃ s sv, promoteWord? t x = some s ∧ wordQ s = some sv ∧ |sv| ≤ ((A + g : ℕ) : ℚ) * 2 ^ 23 := by
  have hs : |tv + xv| ≤ ((A + g : ℕ) : ℚ) * 2 ^ 23 := by push_cast; linarith [abs_add_le tv xv]
  have hAg' : ((A + g : ℕ) : ℚ) < 2 ^ 24 := by exact_mod_cast hAg
  have hlt : |tv + xv| < 2 ^ 127 := by nlinarith
  set nz := decide (tv = 0) && decide (xv = 0) && signBit t && signBit x
  obtain ⟨r, hr⟩ := RowCircuit.rnWord_finite nz hlt
  have hbound : |r| ≤ ((A + g : ℕ) : ℚ) * 2 ^ 23 := by
    have hr' := hr
    unfold rnWord at hr'
    by_cases h0 : tv + xv = 0
    · rw [ite_eq_left h0] at hr'
      have : r = 0 := by
        split_ifs at hr'
        · rw [PeelExactProofs.wordQ_negZero] at hr'; exact (Option.some.inj hr').symm
        · rw [Freeze.wordQ_zero] at hr'; exact (Option.some.inj hr').symm
      rw [this, abs_zero]
      positivity
    · rw [ite_eq_right h0] at hr'
      have hg : ∃ m : ℤ, tv + xv = m * 2 ^ (-149 : ℤ) := by
        obtain ⟨i, hi⟩ := TileCircuit.wordQ_grid ht
        obtain ⟨j, hj⟩ := TileCircuit.wordQ_grid hx
        exact ⟨i + j, by rw [hi, hj]; push_cast; ring⟩
      rw [TileCircuit.f32Word_value hg (RowCircuit.wordQ_exp_ne hr'), Option.some.injEq] at hr'
      rw [← hr']
      have := HonestWords.abs_rn_le (M := A + g) (e := 23) hAg (by rw [zpow_ofNat]; exact hs)
      rwa [zpow_ofNat] at this
  refine ⟨if rnWord (tv + xv) nz = 0x80000000 then 0 else rnWord (tv + xv) nz, r, ?_, ?_, hbound⟩
  · unfold promoteWord? addWord?
    rw [ht, hx]
    exact congrArg (Option.map _) (RowCircuit.finite?_of_wordQ hr)
  · split_ifs with h
    · rw [h, PeelExactProofs.wordQ_negZero] at hr
      rw [Freeze.wordQ_zero, hr]
    · exact hr

theorem chainFold_fin {w : ℕ} (hw : 24 ≤ w) (z : ℤ) :
    ∀ (gs : List (List AtomIn)) (t : ℕ) (tv : ℚ) (A : ℕ), (∀ g ∈ gs, AtomsOK g) → wordQ t = some tv →
      |tv| ≤ A * 2 ^ 23 → A + (gs.map List.length).sum < 2 ^ 24 →
      ∃ c cv, gs.foldlM (fun t g => (windowWordP? ⟨[32], w, z⟩ g).bind (promoteWord? t)) t = some c ∧
        wordQ c = some cv ∧ |cv| ≤ ((A + (gs.map List.length).sum : ℕ) : ℚ) * 2 ^ 23
  | [], t, tv, A, _, ht, hb, _ => ⟨t, tv, rfl, ht, by simpa using hb⟩
  | g :: gs, t, tv, A, hok, ht, hb, hA => by
    simp only [List.map_cons, List.sum_cons] at hA ⊢
    obtain ⟨x, xv, hx, hxv, hxb⟩ := windowWordP_fin hw z (hok g List.mem_cons_self) (by omega)
    obtain ⟨s, sv, hs, hsv, hsb⟩ := promote_fin ht hxv hb hxb (by omega)
    rw [List.foldlM_cons, hx, Option.bind_some, hs]
    obtain ⟨c, cv, hc, hcv, hcb⟩ := chainFold_fin hw z gs s sv (A + g.length)
      (fun g' hg' => hok g' (List.mem_cons_of_mem _ hg')) hsv hsb (by omega)
    refine ⟨c, cv, hc, hcv, ?_⟩
    rwa [show A + g.length + (gs.map List.length).sum = A + (g.length + (gs.map List.length).sum) by omega] at hcb

theorem mem_of_mem_toChunks {α : Type} {G : ℕ} {l g : List α} {x : α} (hg : g ∈ l.toChunks G) (hx : x ∈ g) :
    x ∈ l := by
  rcases Nat.eq_zero_or_pos G with rfl | hG
  · cases l with
    | nil => simp [AccumProofs.toChunks_zero_nil] at hg
    | cons y ys =>
      rw [AccumProofs.toChunks_zero_cons, List.mem_singleton] at hg
      exact hg ▸ hx
  · rw [Freeze.toChunks_eq hG, List.mem_map] at hg
    obtain ⟨i, -, rfl⟩ := hg
    exact List.mem_of_mem_drop (List.mem_of_mem_take hx)

/-- **A chain is finite**: `n < 2^24` atoms whose codes are not NaN, in windows of any `G`, give a finite word of
magnitude at most `n·2^23`. -/
theorem chainWordP_fin {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (G : ℕ) {atoms : List AtomIn} (hok : AtomsOK atoms)
    (hn : atoms.length < 2 ^ 24) :
    ∃ c cv, chainWordP? ⟨[32], w, z⟩ G atoms = some c ∧ wordQ c = some cv ∧ |cv| ≤ atoms.length * 2 ^ 23 := by
  obtain ⟨c, cv, hc, hcv, hcb⟩ := chainFold_fin hw z (atoms.toChunks G) 0 0 0
    (fun g hg x hx => hok x (mem_of_mem_toChunks hg hx)) Freeze.wordQ_zero (by simp)
    (by rw [HonestWords.toChunks_length_sum]; omega)
  refine ⟨c, cv, hc, hcv, ?_⟩
  rwa [HonestWords.toChunks_length_sum, zero_add] at hcb

/-! ## The roundings -/

/-- A finite word below `2^127` has an exponent field of at most `253`. -/
theorem exp_le_253 {w : ℕ} {x : ℚ} (h : wordQ w = some x) (hx : |x| < 2 ^ 127) : (w >>> 23) % 256 ≤ 253 := by
  have he := RowCircuit.wordQ_exp_ne h
  by_contra h254
  have e254 : (w >>> 23) % 256 = 254 := by omega
  rw [PeelExactProofs.wordQ_fields he, Option.some.injEq, e254] at h
  subst h
  have hs : |(if (w >>> 31) % 2 = 1 then (-1 : ℚ) else 1)| = 1 := by split_ifs <;> norm_num
  rw [abs_mul, abs_mul, hs, one_mul, abs_of_nonneg (Nat.cast_nonneg _), abs_of_pos (two_zpow_pos _)] at hx
  have h1 : (2 : ℚ) ^ 23 ≤ (((if 254 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) : ℚ) := by
    have : ((2 ^ 23 : ℕ) : ℚ) ≤ (((if 254 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) : ℚ) := by
      exact_mod_cast (show 2 ^ 23 ≤ (if 254 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 by simp)
    exact_mod_cast this
  have h2 : (2 : ℚ) ^ (((max 254 1 : ℕ) : ℤ) - 150) = 2 ^ 104 := by norm_num
  rw [h2] at hx
  have := mul_le_mul_of_nonneg_right h1 (by positivity : (0 : ℚ) ≤ 2 ^ 104)
  norm_num at this
  linarith

/-- **A finite 32-bit word below `2^127` narrows to BF16**: its exponent field is at most `253`, so the rounding
carries at most into field `254`. -/
theorem rnBf16_fin {w : ℕ} {x : ℚ} (hw : w < 2 ^ 32) (h : wordQ w = some x) (hx : |x| < 2 ^ 127) :
    ∃ b, rnBf16? w = some b := by
  have he := exp_le_253 h hx
  unfold rnBf16?
  rw [RowCircuit.finite?_of_wordQ h, Option.bind_some]
  dsimp only
  split_ifs with hc
  · simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow] at he hc
    have hb : w / 65536 % 2 ≤ 1 := by omega
    generalize w / 65536 % 2 = bit at hb hc
    rcases Nat.lt_or_ge w 2147483648 with hs | hs
    · have h1 : w < 2130706432 := by omega
      generalize hq : (w + 32767 + bit) / 65536 = q at hc
      have : q ≤ 32512 := by omega
      omega
    · obtain ⟨r, rfl⟩ : ∃ r, w = r + 2147483648 := ⟨w - 2147483648, by omega⟩
      have h1 : r < 2130706432 := by omega
      generalize hq : (r + 2147483648 + 32767 + bit) / 65536 = q at hc
      have : 32768 ≤ q ∧ q ≤ 32768 + 32512 := by omega
      omega
  · exact ⟨_, rfl⟩

/-- A rounding at most doubles a value, plus one. -/
theorem rnWord_abs_le {v r : ℚ} {nz : Bool} (h : wordQ (rnWord v nz) = some r) : |r| ≤ 2 * |v| + 1 := by
  have he := PeelExactProofs.rnWord_err h
  have h1 : |r| ≤ |r - v| + |v| := by simpa using abs_add_le (r - v) v
  have h2 : |v| / 2 ^ 24 ≤ |v| := div_le_self (abs_nonneg _) (by norm_num)
  have h3 : (2 : ℚ) ^ (-150 : ℤ) ≤ 1 := by norm_num
  linarith

/-- **An FP32 add below `2^127` is finite.** -/
theorem addWord_fin {a b : ℕ} {x y : ℚ} (ha : wordQ a = some x) (hb : wordQ b = some y) (hxy : |x + y| < 2 ^ 127) :
    ∃ s sv, addWord? a b = some s ∧ wordQ s = some sv ∧ |sv| ≤ 2 * |x + y| + 1 := by
  obtain ⟨r, hr⟩ := RowCircuit.rnWord_finite (decide (x = 0) && decide (y = 0) && signBit a && signBit b) hxy
  refine ⟨_, r, ?_, hr, rnWord_abs_le hr⟩
  unfold addWord?
  rw [ha, hb]
  exact RowCircuit.finite?_of_wordQ hr

/-- **An FMA below `2^127` is finite.** -/
theorem fmaWord_fin {a b c : ℕ} {x y z : ℚ} (ha : wordQ a = some x) (hb : wordQ b = some y) (hc : wordQ c = some z)
    (hv : |x * y + z| < 2 ^ 127) :
    ∃ s sv, fmaWord? a b c = some s ∧ wordQ s = some sv ∧ |sv| ≤ 2 * |x * y + z| + 1 := by
  obtain ⟨r, hr⟩ := RowCircuit.rnWord_finite
    (decide (x * y = 0) && decide (z = 0) && (signBit a != signBit b) && signBit c) hv
  refine ⟨_, r, ?_, hr, rnWord_abs_le hr⟩
  unfold fmaWord?
  rw [ha, hb, hc]
  exact RowCircuit.finite?_of_wordQ hr

/-! ## The FMA chain -/

theorem accFold_fin {B B' : ℚ} (hB : 0 ≤ B) (hB' : 0 ≤ B') :
    ∀ (ps : List (ℕ × ℕ)) (acc : ℕ) (av : ℚ) (i : ℕ),
      (∀ x ∈ ps, (∃ a, wordQ (x.1 <<< 16) = some a ∧ |a| ≤ B) ∧ ∃ b, wordQ x.2 = some b ∧ |b| ≤ B') →
      wordQ acc = some av → |av| ≤ (2 ^ i - 1) * (2 * B * B' + 1) →
      2 ^ (i + ps.length) * (2 * B * B' + 1) < 2 ^ 127 →
      ∃ c cv, ps.foldlM (fun acc x => fmaWord? (x.1 <<< 16) x.2 acc) acc = some c ∧ wordQ c = some cv ∧
        |cv| ≤ (2 ^ (i + ps.length) - 1) * (2 * B * B' + 1)
  | [], acc, av, i, _, h, hb, _ => ⟨acc, av, rfl, h, by simpa using hb⟩
  | p :: ps, acc, av, i, hok, h, hb, hlt => by
    obtain ⟨⟨a, ha, hab⟩, b, hb', hbb⟩ := hok p List.mem_cons_self
    have hab' : |a * b| ≤ B * B' := by rw [abs_mul]; exact mul_le_mul hab hbb (abs_nonneg _) hB
    have hv : |a * b + av| ≤ B * B' + (2 ^ i - 1) * (2 * B * B' + 1) := (abs_add_le _ _).trans (add_le_add hab' hb)
    simp only [List.length_cons] at hlt
    have hD : 0 ≤ 2 * B * B' + 1 := by positivity
    have hpow : (2 : ℚ) ^ i ≤ 2 ^ (i + (ps.length + 1)) := pow_le_pow_right₀ one_le_two (by omega)
    have hlt' : |a * b + av| < 2 ^ 127 := by
      have h1 := mul_le_mul_of_nonneg_right hpow hD
      have h2 : 0 ≤ B * B' := mul_nonneg hB hB'
      linarith
    obtain ⟨s, sv, hs, hsv, hsb⟩ := fmaWord_fin ha hb' h hlt'
    rw [List.foldlM_cons, hs]
    have e : i + 1 + ps.length = i + (ps.length + 1) := by omega
    obtain ⟨c, cv, hc, hcv, hcb⟩ := accFold_fin hB hB' ps s sv (i + 1)
      (fun x hx => hok x (List.mem_cons_of_mem _ hx)) hsv (by rw [pow_succ]; linarith) (by rwa [e])
    refine ⟨c, cv, hc, hcv, ?_⟩
    rwa [e] at hcb

/-- **The FMA chain is finite**: `n` FMAs of BF16 factors at most `B` and words at most `B'` from `+0`, with
`2^n·(2·B·B' + 1) < 2^127`, give a finite word of magnitude at most `2^n·(2·B·B' + 1)`. -/
theorem accChain_fin {B B' : ℚ} (hB : 0 ≤ B) (hB' : 0 ≤ B') {be gq : List ℕ}
    (hok : ∀ x ∈ be.zip gq, (∃ a, wordQ (x.1 <<< 16) = some a ∧ |a| ≤ B) ∧ ∃ b, wordQ x.2 = some b ∧ |b| ≤ B')
    (hlt : 2 ^ (be.zip gq).length * (2 * B * B' + 1) < 2 ^ 127) :
    ∃ c cv, accChain? be gq = some c ∧ wordQ c = some cv ∧ |cv| ≤ 2 ^ (be.zip gq).length * (2 * B * B' + 1) := by
  obtain ⟨c, cv, hc, hcv, hcb⟩ :=
    accFold_fin hB hB' (be.zip gq) 0 0 0 hok Freeze.wordQ_zero (by simp) (by rwa [zero_add])
  refine ⟨c, cv, hc, hcv, hcb.trans ?_⟩
  rw [zero_add]
  have hD : 0 ≤ 2 * B * B' + 1 := by positivity
  linarith

/-! ## The rows -/

theorem mapM_ofFn {β γ : Type} {N : ℕ} (f : β → Option γ) (x : Fin N → β) (g : Fin N → γ)
    (h : ∀ q, f (x q) = some (g q)) : (List.ofFn x).mapM f = some (List.ofFn g) := by
  induction N with
  | zero => rfl
  | succ N ih =>
    rw [List.ofFn_succ, List.ofFn_succ, List.mapM_cons, ih (fun q => x q.succ) (fun q => g q.succ) fun q => h q.succ,
      h 0]
    rfl

theorem mapM_exists {β γ : Type} (f : β → Option γ) :
    ∀ l : List β, (∀ x ∈ l, ∃ y, f x = some y) → ∃ ys, l.mapM f = some ys
  | [], _ => ⟨[], rfl⟩
  | x :: l, h => by
    obtain ⟨y, hy⟩ := h x List.mem_cons_self
    obtain ⟨ys, hys⟩ := mapM_exists f l fun x' hx' => h x' (List.mem_cons_of_mem _ hx')
    exact ⟨y :: ys, by rw [List.mapM_cons, hy, hys]; rfl⟩

theorem getD_ofFn {N : ℕ} (f : Fin N → ℕ) {q : ℕ} (hq : q < N) : (List.ofFn f).getD q 0 = f ⟨q, hq⟩ := by
  simp [List.getD_eq_getElem?_getD, hq]

theorem atomsOK_rowAtoms {k : ℕ} {a b : Fin k → Code} (ha : ∀ l, (a l).isNaN = false)
    (hb : ∀ l, (b l).isNaN = false) : AtomsOK (rowAtoms a b) := by
  intro x hx l
  obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hx
  exact ⟨ha _, hb _⟩

theorem rowAtoms_length {k : ℕ} (a b : Fin k → Code) : (rowAtoms a b).length = k / 32 := List.length_ofFn

/-- **`P_A`'s row exists** where every chain word of `A′·F_B` is a finite word of magnitude at most `2^64`: each narrows
to BF16. -/
theorem paRow_fin (be : List ℕ) {af : Fin 32 → ℕ} (hlt : ∀ p, af p < 2 ^ 32)
    (hf : ∀ p, ∃ x, wordQ (af p) = some x ∧ |x| ≤ 2 ^ 64) : ∃ pa, paRow? be (List.ofFn af) = some pa := by
  obtain ⟨bA, hbA⟩ := mapM_exists rnBf16? (List.ofFn af) fun x hx => by
    obtain ⟨p, rfl⟩ := List.mem_ofFn.1 hx
    obtain ⟨v, hv, hvb⟩ := hf p
    exact rnBf16_fin (hlt p) hv (by linarith)
  exact ⟨_, by unfold paRow?; rw [hbA]; rfl⟩

/-- **`P_B`'s row exists** where row `j`'s scaled noise `be` is BF16 words of magnitude at most `448` and every chain
word of `B̃·F_A` and of `G` is finite and at most `2^64`: each FMA chain of `P_B,j` (at most `2^32·(2·448·2^64 + 1)`),
its difference with `B̃·F_A`'s word and that difference's narrowing are finite. -/
theorem pbRow_fin {be bf : Fin 32 → ℕ} {g : Fin 32 → Fin 32 → ℕ}
    (hbe : ∀ q, ∃ a, wordQ (be q <<< 16) = some a ∧ |a| ≤ 448)
    (hbf : ∀ q, ∃ b, wordQ (bf q) = some b ∧ |b| ≤ 2 ^ 64)
    (hg : ∀ p q, ∃ b, wordQ (g p q) = some b ∧ |b| ≤ 2 ^ 64) :
    ∃ pb, pbRow? (List.ofFn be) (List.ofFn bf) (List.ofFn fun p => List.ofFn (g p)) = some pb := by
  have key : ∀ q ∈ List.range Params.pi.r, ∃ y, (do
      let acc ← accChain? (List.ofFn be) (col (List.ofFn fun p => List.ofFn (g p)) q)
      rnBf16? (← addWord? acc (negWord ((List.ofFn bf).getD q 0)))) = some y := by
    intro q hq
    have hq : q < 32 := List.mem_range.1 hq
    have hcol : col (List.ofFn fun p => List.ofFn (g p)) q = List.ofFn fun p => g p ⟨q, hq⟩ := by
      unfold PeelExact.col
      rw [List.map_ofFn]
      exact congrArg List.ofFn (funext fun p => getD_ofFn (g p) hq)
    have hok : ∀ x ∈ (List.ofFn be).zip (List.ofFn fun p => g p ⟨q, hq⟩),
        (∃ a, wordQ (x.1 <<< 16) = some a ∧ |a| ≤ 448) ∧ ∃ b, wordQ x.2 = some b ∧ |b| ≤ 2 ^ 64 := by
      intro x hx
      obtain ⟨h1, h2⟩ := List.of_mem_zip hx
      obtain ⟨p, hp⟩ := List.mem_ofFn.1 h1
      obtain ⟨p', hp'⟩ := List.mem_ofFn.1 h2
      exact ⟨hp ▸ hbe p, hp' ▸ hg p' ⟨q, hq⟩⟩
    have hlen : ((List.ofFn be).zip (List.ofFn fun p => g p ⟨q, hq⟩)).length = 32 := by simp
    obtain ⟨acc, accv, hacc, haccv, haccb⟩ := accChain_fin (by norm_num) (by norm_num) hok (by rw [hlen]; norm_num)
    rw [hlen] at haccb
    obtain ⟨bv, hbv, hbvb⟩ := hbf ⟨q, hq⟩
    have hbq : wordQ (negWord ((List.ofFn bf).getD q 0)) = some (-bv) := by
      rw [getD_ofFn bf hq, PeelExactProofs.wordQ_negWord, hbv]
      rfl
    have hsum : |accv + -bv| ≤ 2 ^ 32 * (2 * 448 * 2 ^ 64 + 1) + 2 ^ 64 := by
      have := abs_add_le accv (-bv)
      rw [abs_neg] at this
      linarith
    obtain ⟨s, sv, hs, hsv, hsb⟩ := addWord_fin haccv hbq (by linarith)
    obtain ⟨b, hb⟩ := rnBf16_fin (WholeCircuit.addWord?_lt hs) hsv (by linarith)
    refine ⟨b, ?_⟩
    rw [hcol, hacc]
    simp only [Option.bind_eq_bind, Option.bind_some, hs, hb]
  obtain ⟨mid, hmid⟩ := mapM_exists _ _ key
  exact ⟨_, by unfold pbRow?; rw [hmid]; rfl⟩

/-- **The peel's rows exist** at a step `⟨[32], w, z⟩` with `w ≥ 24`, in windows of any `G`, at `k/32 ≤ 2^11`: on
operands and bases whose codes are not NaN, with row `j`'s scaled noise `beB j` BF16 words of magnitude at most `448`,
every chain is finite and at most `2^34`, so `P_A,i` and `P_B,j` exist (`paRow_fin`, `pbRow_fin`). -/
theorem peelRowsP_isSome {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (G : ℕ) {m n k : ℕ} (hk : k / 32 ≤ 2 ^ 11)
    (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code)
    (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ) (i : Fin m) (j : Fin n)
    (hA : ∀ l, (Aq i l).isNaN = false) (hB : ∀ l, (Bq j l).isNaN = false)
    (hFA : ∀ l q, (FA l q).isNaN = false) (hFB : ∀ l q, (FB l q).isNaN = false)
    (heB : ∀ q, ∃ a, wordQ (beB j q <<< 16) = some a ∧ |a| ≤ 448) :
    (PeelExactProofsP.peelRowsP ⟨[32], w, z⟩ G Aq Bq FA FB beA beB i j).isSome := by
  have hch : ∀ {a b : Fin k → Code}, (∀ l, (a l).isNaN = false) → (∀ l, (b l).isNaN = false) →
      ∃ c cv, chainWordP? ⟨[32], w, z⟩ G (rowAtoms a b) = some c ∧ wordQ c = some cv ∧ |cv| ≤ 2 ^ 34 := by
    intro a b ha hb
    have hl : ((rowAtoms a b).length : ℚ) ≤ 2 ^ 11 := by rw [rowAtoms_length]; exact_mod_cast hk
    obtain ⟨c, cv, hc, hcv, hcb⟩ :=
      chainWordP_fin hw z G (atomsOK_rowAtoms ha hb) (by rw [rowAtoms_length]; omega)
    refine ⟨c, cv, hc, hcv, hcb.trans ?_⟩
    calc ((rowAtoms a b).length : ℚ) * 2 ^ 23 ≤ 2 ^ 11 * 2 ^ 23 := by gcongr
      _ = 2 ^ 34 := by norm_num
  obtain ⟨c, -, hc, -, -⟩ := hch hA hB
  choose af afv haf hafv hafb using fun p : Fin 32 => hch (a := Aq i) (b := FBᵀ p) hA fun l => hFB l p
  choose bf bfv hbf hbfv hbfb using fun q : Fin 32 => hch (a := Bq j) (b := FAᵀ q) hB fun l => hFA l q
  choose g gv hg hgv hgb using
    fun p q : Fin 32 => hch (a := FBᵀ p) (b := FAᵀ q) (fun l => hFB l p) fun l => hFA l q
  have hafL : (List.ofFn fun p => chainWordP? ⟨[32], w, z⟩ G (rowAtoms (Aq i) (FBᵀ p))).mapM id =
      some (List.ofFn af) := mapM_ofFn id _ af haf
  have hbfL : (List.ofFn fun q => chainWordP? ⟨[32], w, z⟩ G (rowAtoms (Bq j) (FAᵀ q))).mapM id =
      some (List.ofFn bf) := mapM_ofFn id _ bf hbf
  have hgL : (List.ofFn fun p => (List.ofFn fun q =>
      chainWordP? ⟨[32], w, z⟩ G (rowAtoms (FBᵀ p) (FAᵀ q))).mapM id).mapM id =
      some (List.ofFn fun p => List.ofFn (g p)) := mapM_ofFn id _ _ fun p => mapM_ofFn id _ (g p) (hg p)
  have h64 : ∀ {x : ℚ}, |x| ≤ 2 ^ 34 → |x| ≤ 2 ^ 64 := fun h => h.trans (by norm_num)
  obtain ⟨pa, hpa⟩ := paRow_fin (List.ofFn (beA i)) (fun p => WholeCircuit.chainWordP?_lt (haf p))
    fun p => ⟨afv p, hafv p, h64 (hafb p)⟩
  obtain ⟨pb, hpb⟩ := pbRow_fin heB (fun q => ⟨bfv q, hbfv q, h64 (hbfb q)⟩)
    fun p q => ⟨gv p q, hgv p q, h64 (hgb p q)⟩
  unfold PeelExactProofsP.peelRowsP
  simp only [hc, hafL, hbfL, hgL, hpa, hpb, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
    Option.isSome_some]

/-! ## `pc8`'s entries -/

/-- A BF16 word of an E4M3 code that is not NaN, widened, reads back as the code's value. -/
theorem wordQ_codeBf16 : ∀ c : Code, c.isNaN = false → wordQ (codeBf16 c <<< 16) = some (codeVal c) := by
  decide +kernel

theorem abs_codeVal_le : ∀ c : Code, |codeVal c| ≤ 448 := by decide +kernel

theorem e4m3_isNaN (v : ℚ) : (e4m3 v).isNaN = false := by
  have := RowCircuit.e4m3_notNaN v
  simp [Code.isNaN, this]

/-- **`Pc8EntryOK` at a step `⟨[32], w, z⟩` with `w ≥ 24` and `k ≤ 2^16`, from `Noise8OK` alone**: `A′`'s and `B̃`'s
codes are E4M3 casts, never NaN, the bases' codes are `Noise8OK`'s, and `be8`'s words are BF16 words of `E′`'s codes
(`lineCodes_notNaN`), at most `448`. -/
theorem pc8EntryOK_of_wide {ops : RowCircuit.RowOps} {w : ℕ} (hw : 24 ≤ w) (zf : ℤ) {m n k : ℕ} (hk : k ≤ 2 ^ 16)
    {z : Noise m n k Params.pi.r} (hz : WholeCircuit.Noise8OK z) (act wt : Codes) (i : Fin m) (j : Fin n) :
    WholeCircuit.Pc8EntryOK ops ⟨[32], w, zf⟩ z act wt i j :=
  ⟨peelRowsP_isSome hw zf 4 (by omega) _ _ _ _ _ _ i j (fun _ => e4m3_isNaN _) (fun _ => e4m3_isNaN _)
    (fun l q => hz.2.1 l q) (fun l q => hz.2.2.2 l q)
    fun _ => ⟨_, wordQ_codeBf16 _ (WholeCircuit.lineCodes_notNaN _ _ _), abs_codeVal_le _⟩⟩
end Pouw.PearlC.PeelExists
