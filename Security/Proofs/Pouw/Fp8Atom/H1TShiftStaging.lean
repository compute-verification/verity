import Proofs.Pouw.Fp8Atom.H1TGapStaging
import Proofs.Pouw.Fp8Atom.H1TProofs

/-!
# H-1T: the late words against staircases (staging, unpinned)

`H1TCrossLate` compares a block-2 running word `R_p(j)` with another column's running word `R_q(j′)`, with `p` and `q`
from `T − 1` to `2T − 3`. Fix every salt but slice `T − 1`'s. Then `R_p(j) = chainFrom A (I :: cs)`: `A = R_(T−2)(j)`,
`I` is column `j`'s block-1 word of slice `T − 1`, and `cs` are the block-2 words of slices `0..p − T`, none of which
reads slice `T − 1`'s salt. The same holds for `j′`, with `cs′ = []` at `q = T − 1`.

* `rn_shift2`, `rnAdd_shift`, `chainFrom_shift`: a chain commutes with adding `D` to its start while each step's two
  sums share an ulp `u` and `D ∈ 2uℤ` (`SameCell`). `chainFrom_equivariant` gets that from one binade (`InBinade`).
* `staircase_extend`, `split_of_staircase`: a monotone map that commutes with `+ W` on a window gives a step-`W`
  staircase (`Staircase`) there, `⌊g / W⌋`. So if no two staircases agree on two families of values, no two such maps
  agree either.
* `lateWords_split`: where both columns' sums after `A` stay in one binade of ulp at most `W / 2` for every word in a
  window, the words separate the two chains whenever no two staircases agree on them. `lateWords_of_tagSplit` takes
  that from `TagStaircaseSplit W`, a statement about one slice's two block-1 words that reads neither `A` nor the
  positions. It is not proved; `internal/pouw-fp8/h1t-crosslate-small-delta.md` in the store has its evidence.
* `chainFrom_shift_ok`, `late_shift_ok`, `lateWords_split_on`: the same with each step checked on its own
  (`StepOK`: its two sums in one binade, which may change from step to step, or both exact; `stepOK_of_grid` is the
  exact region below `2^24` grid steps).
* `lateWords_split_cut`, `lateWords_of_cutSplit`: **the straddle lemma**. A chain that crosses one power of two at
  one step passes `+ W` on each side of a cut, so the words separate unless four staircases, one per column and side,
  agree on them; `TagCutSplit W` (not proved, and it implies `TagStaircaseSplit W`) says they never do.
* `lateWords_split_cut2`, `lateWords_of_cutSplit2`: the same with two cuts per column and three staircases each;
  `TagCutSplit2 W` (not proved, and it implies `TagCutSplit W`). `lateWords_split_cut2_on` needs windows only on a
  sub-family, for words that lie in no window.
* `lateWords_of_cutSplit2Off`: the same with windows only off exclusion zones, `z` intervals of width `w` per column
  (`InZones`); `TagCutSplit2Off W z w` (not proved, and it implies `TagCutSplit2 W`) asks the separating tag setting
  to avoid them.
* `chainFrom_err`, `lateWords_margin`: below `2^e` a chain is its exact sum to within `2^(e − 25)` per step, across
  any binades; so two tag settings whose difference `Y′ − Y` moves by more than `(n + n′ + 2)·2^(e − 24)` separate
  the chains outright.
-/

namespace Pouw.Fp8Atom.H1T.Shift

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T

/-! ## Rounding commutes with even shifts -/

theorem roundEven_add_two (m : ℚ) (n : ℤ) : roundEven (m + 2 * n) = roundEven m + 2 * n := by
  have hf : ⌊m + 2 * (n : ℚ)⌋ = ⌊m⌋ + 2 * n := by
    rw [show m + 2 * (n : ℚ) = m + ((2 * n : ℤ) : ℚ) by push_cast; ring, Int.floor_add_intCast]
  have hd : m + 2 * (n : ℚ) - ((⌊m⌋ + 2 * n : ℤ) : ℚ) = m - ⌊m⌋ := by push_cast; ring
  unfold roundEven; dsimp only
  rw [hf, hd]
  split_ifs <;> omega

theorem roundEven_neg (m : ℚ) : roundEven (-m) = -roundEven m := by
  rcases eq_or_ne (m - ⌊m⌋) 0 with h0 | h0
  · obtain ⟨k, rfl⟩ : ∃ k : ℤ, m = k := ⟨⌊m⌋, by linarith⟩
    rw [← Int.cast_neg, roundEven_intCast, roundEven_intCast]
  · have hlt : (⌊m⌋ : ℚ) < m := lt_of_le_of_ne (Int.floor_le m) (fun h => h0 (by rw [h]; ring))
    have h1 := Int.lt_floor_add_one m
    have hf : ⌊-m⌋ = -⌊m⌋ - 1 := by
      rw [Int.floor_eq_iff]; push_cast
      constructor <;> linarith
    have hd : -m - ((-⌊m⌋ - 1 : ℤ) : ℚ) = 1 - (m - ⌊m⌋) := by push_cast; ring
    unfold roundEven; dsimp only
    rw [hf, hd]
    split_ifs <;> first | omega | (exfalso; linarith)

theorem rn_div (x : ℚ) : rn x = (roundEven (x / ulp x) : ℚ) * ulp x := by
  unfold rn
  rcases lt_or_ge x 0 with hx | hx
  · rw [ite_eq_left hx, abs_of_neg hx, neg_div, roundEven_neg]; push_cast; ring
  · rw [ite_eq_right (not_lt.2 hx), abs_of_nonneg hx, one_mul]

/-- Adding an even number of ulps that keeps the ulp moves the rounded value by the same amount. -/
theorem rn_shift2 {y : ℚ} {n : ℤ} (hu : ulp (y + 2 * n * ulp y) = ulp y) :
    rn (y + 2 * n * ulp y) = rn y + 2 * n * ulp y := by
  have h0 := (ulp_pos y).ne'
  rw [rn_div, rn_div y, hu]
  have e : (y + 2 * n * ulp y) / ulp y = y / ulp y + 2 * (n : ℚ) := by
    rw [add_div, mul_div_assoc, div_self h0, mul_one]
  rw [e, roundEven_add_two]; push_cast; ring

/-- `x` and `y` share an ulp, and `D` is an even number of it. -/
def SameCell (D x y : ℚ) : Prop := ulp x = ulp y ∧ ∃ n : ℤ, D = 2 * n * ulp y

theorem rnAdd_shift {a b c D : ℚ} (hab : a = b + D) (h : SameCell D (a + c) (b + c)) :
    rnAdd a c = rnAdd b c + D := by
  obtain ⟨hu, n, hn⟩ := h
  have hx : a + c = b + c + 2 * n * ulp (b + c) := by rw [hab, ← hn]; ring
  rw [hx] at hu
  unfold rnAdd; rw [hx, rn_shift2 hu, ← hn]

/-! ## The chain commutes with shifts inside one binade -/

theorem chainFrom_cons (a c : ℚ) (cs : List ℚ) : chainFrom a (c :: cs) = chainFrom (rnAdd a c) cs := rfl

/-- **The chain shift**: two chains through the same increments, whose starts differ by `D`, still differ by `D` while
each step's two sums are `SameCell D`. -/
theorem chainFrom_shift {D : ℚ} : ∀ (cs : List ℚ) (a b : ℚ), a = b + D →
    (∀ (k : ℕ) (c : ℚ), cs[k]? = some c →
      SameCell D (chainFrom a (cs.take k) + c) (chainFrom b (cs.take k) + c)) →
    chainFrom a cs = chainFrom b cs + D
  | [], _, _, hab, _ => hab
  | c :: cs, a, b, hab, h => by
    have h0 : SameCell D (a + c) (b + c) := by simpa [chainFrom] using h 0 c (by simp)
    have := chainFrom_shift cs (rnAdd a c) (rnAdd b c) (rnAdd_shift hab h0) fun k c' hk => by
      simpa [chainFrom] using h (k + 1) c' (by simpa using hk)
    simpa [chainFrom] using this

theorem chainFrom_mono : ∀ (cs : List ℚ) {a b : ℚ}, a ≤ b → chainFrom a cs ≤ chainFrom b cs
  | [], _, _, h => h
  | c :: cs, a, b, h => chainFrom_mono cs (rnAdd_mono a c b c (by linarith))

/-- Every step's sum of the chain from `a` through `cs` lies in the binade `[2^e, 2^(e+1))` in magnitude. -/
def InBinade (e : ℤ) (a : ℚ) (cs : List ℚ) : Prop :=
  ∀ (k : ℕ) (c : ℚ), cs[k]? = some c →
    2 ^ e ≤ |chainFrom a (cs.take k) + c| ∧ |chainFrom a (cs.take k) + c| < 2 ^ (e + 1)

theorem ulp_of_binade {z : ℚ} {e : ℤ} (h1 : 2 ^ e ≤ |z|) (h2 : |z| < 2 ^ (e + 1)) : ulp z = 2 ^ (e - 23) := by
  unfold ulp; rw [binade_unique (lt_of_lt_of_le (two_zpow_pos e) h1) h1 h2]

theorem chainFrom_equivariant {W : ℚ} {e n : ℤ} (hWn : W = 2 * n * 2 ^ (e - 23)) {a b : ℚ} {cs : List ℚ}
    (hab : a = b + W) (ha : InBinade e a cs) (hb : InBinade e b cs) : chainFrom a cs = chainFrom b cs + W :=
  chainFrom_shift cs a b hab fun k c hk => by
    obtain ⟨a1, a2⟩ := ha k c hk
    obtain ⟨b1, b2⟩ := hb k c hk
    exact ⟨by rw [ulp_of_binade a1 a2, ulp_of_binade b1 b2], n, by rw [ulp_of_binade b1 b2, hWn]⟩

/-! ## Staircases -/

/-- A step-`W` staircase: monotone, and one step up for every `W`. -/
def Staircase (W : ℚ) (N : ℚ → ℤ) : Prop := Monotone N ∧ ∀ x, N (x + W) = N x + 1

/-- A monotone `g` that commutes with `+ W` on `[L, H]` agrees there with a step-`W` staircase, `⌊g / W⌋`. -/
theorem staircase_extend {W L H : ℚ} (hW : 0 < W) (hLH : L + W ≤ H) {g : ℚ → ℚ} (hg : Monotone g)
    (he : ∀ x, L ≤ x → x + W ≤ H → g (x + W) = g x + W) :
    ∃ N, Staircase W N ∧ ∀ x, L ≤ x → x ≤ H → N x = ⌊g x / W⌋ := by
  obtain ⟨k, hk⟩ : ∃ k : ℚ → ℤ, ∀ x, k x = ⌊(x - L) / W⌋ := ⟨_, fun _ => rfl⟩
  obtain ⟨r, hr⟩ : ∃ r : ℚ → ℚ, ∀ x, r x = x - k x * W := ⟨_, fun _ => rfl⟩
  have hk1 : ∀ x, (k x : ℚ) * W ≤ x - L := fun x => by
    have := Int.floor_le ((x - L) / W); rw [← hk, le_div_iff₀ hW] at this; exact this
  have hk2 : ∀ x, x - L < (k x : ℚ) * W + W := fun x => by
    have := Int.lt_floor_add_one ((x - L) / W); rw [← hk, div_lt_iff₀ hW] at this; linarith
  have hr1 : ∀ x, L ≤ r x := fun x => by rw [hr]; linarith [hk1 x]
  have hr2 : ∀ x, r x < L + W := fun x => by rw [hr]; linarith [hk2 x]
  have hkW : ∀ x, k (x + W) = k x + 1 := fun x => by
    rw [hk, hk, show (x + W - L) / W = (x - L) / W + 1 by rw [div_add_one hW.ne']; ring, Int.floor_add_one]
  have hrW : ∀ x, r (x + W) = r x := fun x => by rw [hr, hr, hkW]; push_cast; ring
  have hiter : ∀ (y : ℚ) (j : ℕ), L ≤ y → y + j * W ≤ H → g (y + j * W) = g y + j * W := by
    intro y j hy
    induction j with
    | zero => intro _; simp
    | succ j ih =>
      intro hj
      have hj0 : (0 : ℚ) ≤ j * W := mul_nonneg (Nat.cast_nonneg j) hW.le
      have e : y + ((j + 1 : ℕ) : ℚ) * W = y + j * W + W := by push_cast; ring
      rw [e] at hj ⊢
      rw [he _ (by linarith) hj, ih (by linarith)]; push_cast; ring
  refine ⟨fun x => ⌊g (r x) / W⌋ + k x, ⟨fun x y hxy => ?_, fun x => ?_⟩, fun x hx1 hx2 => ?_⟩
  · show ⌊g (r x) / W⌋ + k x ≤ ⌊g (r y) / W⌋ + k y
    have hkm : k x ≤ k y := by
      rw [hk, hk]; exact Int.floor_mono (div_le_div_of_nonneg_right (by linarith) hW.le)
    rcases hkm.lt_or_eq with hlt | heq
    · have a1 : g (r x) ≤ g L + W := le_trans (hg (hr2 x).le) (he L le_rfl hLH).le
      have a2 : g L ≤ g (r y) := hg (hr1 y)
      have h3 : g (r x) / W ≤ g (r y) / W + 1 := by
        rw [div_add_one hW.ne']; exact div_le_div_of_nonneg_right (by linarith) hW.le
      have := Int.floor_mono h3
      rw [Int.floor_add_one] at this
      omega
    · have hrr : r x ≤ r y := by rw [hr, hr, heq]; linarith
      have := Int.floor_mono (div_le_div_of_nonneg_right (hg hrr) hW.le)
      omega
  · show ⌊g (r (x + W)) / W⌋ + k (x + W) = ⌊g (r x) / W⌋ + k x + 1
    rw [hrW, hkW]; ring
  · show ⌊g (r x) / W⌋ + k x = ⌊g x / W⌋
    have hk0 : 0 ≤ k x := by rw [hk]; exact Int.floor_nonneg.2 (div_nonneg (by linarith) hW.le)
    obtain ⟨m, hm⟩ : ∃ m : ℕ, (m : ℤ) = k x := ⟨(k x).toNat, Int.toNat_of_nonneg hk0⟩
    have hmq : (m : ℚ) = (k x : ℚ) := by exact_mod_cast hm
    have hxm : x = r x + (m : ℚ) * W := by rw [hr, hmq]; ring
    have hg' := hiter (r x) m (hr1 x) (by rw [← hxm]; exact hx2)
    rw [← hxm] at hg'
    rw [hg', hmq, show (g (r x) + (k x : ℚ) * W) / W = g (r x) / W + (k x : ℚ) by
      rw [add_div, mul_div_assoc, div_self hW.ne', mul_one], Int.floor_add_intCast]

/-- **Staircases to maps**: if no two step-`W` staircases agree on `Y` and `Y′`, then no two monotone maps agree on
them that commute with `+ W` on windows at least `W` wide holding `Y` and `Y′`. -/
theorem split_of_staircase {ι : Type*} {W : ℚ} (hW : 0 < W) {Y Y' : ι → ℚ}
    (h : ∀ N N', Staircase W N → Staircase W N' → ∃ i, N (Y i) ≠ N' (Y' i))
    {g g' : ℚ → ℚ} {L H L' H' : ℚ} (hg : Monotone g) (hg' : Monotone g') (hLH : L + W ≤ H)
    (hLH' : L' + W ≤ H') (he : ∀ x, L ≤ x → x + W ≤ H → g (x + W) = g x + W)
    (he' : ∀ x, L' ≤ x → x + W ≤ H' → g' (x + W) = g' x + W)
    (hY : ∀ i, L ≤ Y i ∧ Y i ≤ H) (hY' : ∀ i, L' ≤ Y' i ∧ Y' i ≤ H') : ∃ i, g (Y i) ≠ g' (Y' i) := by
  obtain ⟨N, hN, hNe⟩ := staircase_extend hW hLH hg he
  obtain ⟨N', hN', hNe'⟩ := staircase_extend hW hLH' hg' he'
  obtain ⟨i, hi⟩ := h N N' hN hN'
  refine ⟨i, fun heq => hi ?_⟩
  rw [hNe _ (hY i).1 (hY i).2, hNe' _ (hY' i).1 (hY' i).2, heq]

theorem staircase_double {W : ℚ} {N : ℚ → ℤ} (h : Staircase W N) : Staircase (2 * W) fun x => N x / 2 := by
  refine ⟨fun x y hxy => Int.ediv_le_ediv (by norm_num) (h.1 hxy), fun x => ?_⟩
  show N (x + 2 * W) / 2 = N x / 2 + 1
  rw [show x + 2 * W = x + W + W by ring, h.2, h.2, show N x + 1 + 1 = N x + 1 * 2 by ring,
    Int.add_mul_ediv_right _ _ (by norm_num)]

/-! ## The late words -/

/-- The chain from `A` whose first increment is `x`, then `cs`, sums in binade `e` throughout. -/
def LateInBinade (e : ℤ) (A x : ℚ) (cs : List ℚ) : Prop :=
  (2 ^ e ≤ |A + x| ∧ |A + x| < 2 ^ (e + 1)) ∧ InBinade e (rnAdd A x) cs

theorem late_equivariant {W A L H : ℚ} {cs : List ℚ} {e n : ℤ} (hW : 0 < W) (hWn : W = 2 * n * 2 ^ (e - 23))
    (hB : ∀ x, L ≤ x → x ≤ H → LateInBinade e A x cs) :
    ∀ x, L ≤ x → x + W ≤ H → chainFrom A ((x + W) :: cs) = chainFrom A (x :: cs) + W := by
  intro x hx1 hx2
  obtain ⟨⟨a1, a2⟩, ha⟩ := hB (x + W) (by linarith) hx2
  obtain ⟨⟨b1, b2⟩, hb⟩ := hB x hx1 (by linarith)
  have hstep : rnAdd A (x + W) = rnAdd A x + W := by
    have e1 : rnAdd A (x + W) = rnAdd (x + W) A := by unfold rnAdd; rw [add_comm]
    have e2 : rnAdd A x = rnAdd x A := by unfold rnAdd; rw [add_comm]
    rw [e1, e2]
    refine rnAdd_shift rfl ⟨?_, n, ?_⟩
    · rw [show x + W + A = A + (x + W) by ring, show x + A = A + x by ring, ulp_of_binade a1 a2,
        ulp_of_binade b1 b2]
    · rw [show x + A = A + x by ring, ulp_of_binade b1 b2, hWn]
  rw [chainFrom_cons, chainFrom_cons]
  exact chainFrom_equivariant hWn hstep ha hb

/-- **The late words split**: two chains from `A`, `A′` whose first increments are `Y i`, `Y′ i` and whose later
increments `cs`, `cs′` do not depend on `i`. If each chain's sums stay in one binade (`e`, `e′`) for every first
increment in a window at least `W` wide that holds the words, `W` is an even number of both ulps, and no two step-`W`
staircases agree on the words, then some `i` separates the chains. -/
theorem lateWords_split {ι : Type*} {W : ℚ} (hW : 0 < W) {Y Y' : ι → ℚ}
    (h : ∀ N N', Staircase W N → Staircase W N' → ∃ i, N (Y i) ≠ N' (Y' i))
    {A A' L H L' H' : ℚ} {cs cs' : List ℚ} {e e' n n' : ℤ}
    (hWn : W = 2 * n * 2 ^ (e - 23)) (hWn' : W = 2 * n' * 2 ^ (e' - 23)) (hLH : L + W ≤ H) (hLH' : L' + W ≤ H')
    (hB : ∀ x, L ≤ x → x ≤ H → LateInBinade e A x cs) (hB' : ∀ x, L' ≤ x → x ≤ H' → LateInBinade e' A' x cs')
    (hY : ∀ i, L ≤ Y i ∧ Y i ≤ H) (hY' : ∀ i, L' ≤ Y' i ∧ Y' i ≤ H') :
    ∃ i, chainFrom A (Y i :: cs) ≠ chainFrom A' (Y' i :: cs') :=
  split_of_staircase (g := fun x => chainFrom A (x :: cs)) (g' := fun x => chainFrom A' (x :: cs')) hW h
    (fun x y hxy => chainFrom_mono cs (rnAdd_mono A x A y (by linarith)))
    (fun x y hxy => chainFrom_mono cs' (rnAdd_mono A' x A' y (by linarith))) hLH hLH'
    (late_equivariant hW hWn hB) (late_equivariant hW hWn' hB') hY hY'

/-! ## The one-slice statement -/

/-- Column `j`'s block-1 word of a slice with real codes `xs`, column codes `b`, real signs `pos` and tag bits
`neg`, `man`. -/
def tagWord (xs : Fin 29 → Code) (b : ℕ → Code) (j τ : ℕ) (pos : Fin 29 → Bool) (neg : Fin 3 → Bool)
    (man : Fin 3 → Fin 8) : ℚ :=
  wordVal (hopperStep 0 (formSlice xs ⟨pos, neg, man⟩).1 (colLanes b j τ))

/-- **The tag staircase split** (not proved): for finite codes and two columns with different tuples, whatever the
real signs, no two step-`W` staircases agree on the two columns' block-1 words over the slice's 12 tag bits. -/
def TagStaircaseSplit (W : ℚ) : Prop :=
  ∀ (xs : Fin 29 → Code) (b b' : ℕ → Code) (j j' τ : ℕ) (pos : Fin 29 → Bool), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) → (∀ r, (b' r).isNaN = false) → tuple j ≠ tuple j' →
    ∀ N N' : ℚ → ℤ, Staircase W N → Staircase W N' →
      ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
        N (tagWord xs b j τ pos neg man) ≠ N' (tagWord xs b' j' τ pos neg man)

theorem tagStaircaseSplit_half {W : ℚ} (h : TagStaircaseSplit (2 * W)) : TagStaircaseSplit W := by
  intro xs b b' j j' τ pos hx hb hb' ht N N' hN hN'
  obtain ⟨neg, man, hne⟩ := h xs b b' j j' τ pos hx hb hb' ht _ _ (staircase_double hN) (staircase_double hN')
  exact ⟨neg, man, fun he => hne (by simp only [he])⟩

/-- **`TagStaircaseSplit` separates the late words** of one slice's two columns, under `lateWords_split`'s binade
windows. -/
theorem lateWords_of_tagSplit {W : ℚ} (hW : 0 < W) (hS : TagStaircaseSplit W) {xs : Fin 29 → Code}
    {b b' : ℕ → Code} {j j' τ : ℕ} {pos : Fin 29 → Bool} (hx : ∀ l, (xs l).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (hb' : ∀ r, (b' r).isNaN = false) (ht : tuple j ≠ tuple j')
    {A A' L H L' H' : ℚ} {cs cs' : List ℚ} {e e' n n' : ℤ}
    (hWn : W = 2 * n * 2 ^ (e - 23)) (hWn' : W = 2 * n' * 2 ^ (e' - 23)) (hLH : L + W ≤ H) (hLH' : L' + W ≤ H')
    (hB : ∀ x, L ≤ x → x ≤ H → LateInBinade e A x cs) (hB' : ∀ x, L' ≤ x → x ≤ H' → LateInBinade e' A' x cs')
    (hY : ∀ neg man, L ≤ tagWord xs b j τ pos neg man ∧ tagWord xs b j τ pos neg man ≤ H)
    (hY' : ∀ neg man, L' ≤ tagWord xs b' j' τ pos neg man ∧ tagWord xs b' j' τ pos neg man ≤ H') :
    ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
      chainFrom A (tagWord xs b j τ pos neg man :: cs) ≠ chainFrom A' (tagWord xs b' j' τ pos neg man :: cs') := by
  obtain ⟨⟨neg, man⟩, hne⟩ := lateWords_split (ι := (Fin 3 → Bool) × (Fin 3 → Fin 8))
    (Y := fun s => tagWord xs b j τ pos s.1 s.2) (Y' := fun s => tagWord xs b' j' τ pos s.1 s.2) hW
    (fun N N' hN hN' => by
      obtain ⟨neg, man, h⟩ := hS xs b b' j j' τ pos hx hb hb' ht N N' hN hN'
      exact ⟨(neg, man), h⟩)
    hWn hWn' hLH hLH' hB hB' (fun s => hY s.1 s.2) (fun s => hY' s.1 s.2)
  exact ⟨neg, man, hne⟩

/-! ## Step by step: binades that change between steps, exact steps, and a straddle -/

/-- A step's sums `z + D` and `z` pass the shift `D` through `rn`: they share an ulp of which `D` is an even
number, or both are representable. -/
def StepOK (D z : ℚ) : Prop := SameCell D (z + D) z ∨ (Representable (z + D) ∧ Representable z)

theorem rn_shift_ok {D z : ℚ} (h : StepOK D z) : rn (z + D) = rn z + D := by
  rcases h with ⟨hu, n, hn⟩ | ⟨h1, h2⟩
  · subst hn; exact rn_shift2 hu
  · rw [rn_exact h1, rn_exact h2]

theorem stepOK_of_binade {D z : ℚ} {e n : ℤ} (hD : D = 2 * n * 2 ^ (e - 23)) (h1 : 2 ^ e ≤ |z + D|)
    (h2 : |z + D| < 2 ^ (e + 1)) (h3 : 2 ^ e ≤ |z|) (h4 : |z| < 2 ^ (e + 1)) : StepOK D z :=
  Or.inl ⟨by rw [ulp_of_binade h1 h2, ulp_of_binade h3 h4], n, by rw [ulp_of_binade h3 h4, hD]⟩

/-- A multiple of `2^g` below `2^(24 + g)` in magnitude is representable. -/
theorem representable_of_grid {x : ℚ} {g m : ℤ} (hx : x = m * 2 ^ g) (h : |x| < 2 ^ (24 + g)) :
    Representable x := by
  refine ⟨m, g, ?_, hx⟩
  have hg := two_zpow_pos g
  rw [hx, abs_mul, abs_of_pos hg, zpow_add₀ two_ne_zero] at h
  have hm : |(m : ℚ)| < 2 ^ (24 : ℤ) := lt_of_mul_lt_mul_right h hg.le
  have : |(m : ℚ)| < 2 ^ 24 := by simpa using hm
  exact_mod_cast this

/-- **The exact region**: both sums multiples of `2^g` below `2^(24 + g)` in magnitude. Every word of a slice is a
multiple of `F/8` (its tag products are at least `1024·F`), so below `2^21·F` for the least `F` of the slices read,
every add is exact. -/
theorem stepOK_of_grid {D z : ℚ} {g a b : ℤ} (ha : z + D = a * 2 ^ g) (hb : z = b * 2 ^ g)
    (h1 : |z + D| < 2 ^ (24 + g)) (h2 : |z| < 2 ^ (24 + g)) : StepOK D z :=
  Or.inr ⟨representable_of_grid ha h1, representable_of_grid hb h2⟩

/-- **The chain shift, step by step**: the chain from `b + D` is the chain from `b` plus `D` while every sum `z` of
`b`'s chain has `StepOK D z`, each step in a binade of its own or exact. -/
theorem chainFrom_shift_ok {D : ℚ} : ∀ (cs : List ℚ) (b : ℚ),
    (∀ (k : ℕ) (c : ℚ), cs[k]? = some c → StepOK D (chainFrom b (cs.take k) + c)) →
    chainFrom (b + D) cs = chainFrom b cs + D
  | [], _, _ => rfl
  | c :: cs, b, h => by
    have h0 : StepOK D (b + c) := by simpa [chainFrom] using h 0 c (by simp)
    have e : rnAdd (b + D) c = rnAdd b c + D := by
      unfold rnAdd; rw [show b + D + c = b + c + D by ring]; exact rn_shift_ok h0
    rw [chainFrom_cons, chainFrom_cons, e]
    exact chainFrom_shift_ok cs (rnAdd b c) fun k c' hk => by
      simpa [chainFrom] using h (k + 1) c' (by simpa using hk)

/-- The late word from `A` with first increment `x` passes `+ W` at every step: the sum `A + x`, then every sum of
the chain from `RN(A + x)`. -/
def LateOK (W A x : ℚ) (cs : List ℚ) : Prop :=
  StepOK W (A + x) ∧ ∀ (k : ℕ) (c : ℚ), cs[k]? = some c → StepOK W (chainFrom (rnAdd A x) (cs.take k) + c)

theorem late_shift_ok {W A x : ℚ} {cs : List ℚ} (h : LateOK W A x cs) :
    chainFrom A ((x + W) :: cs) = chainFrom A (x :: cs) + W := by
  have e : rnAdd A (x + W) = rnAdd A x + W := by
    unfold rnAdd; rw [← add_assoc]; exact rn_shift_ok h.1
  rw [chainFrom_cons, chainFrom_cons, e]
  exact chainFrom_shift_ok cs _ h.2

/-- **The late words split on a sub-family**: `lateWords_split` with the steps checked one by one (`LateOK` on each
window) and the staircase split taken only over the indices in `P`, whose words lie in the windows. -/
theorem lateWords_split_on {ι : Type*} {W : ℚ} (hW : 0 < W) {Y Y' : ι → ℚ} {P : ι → Prop}
    (h : ∀ N N', Staircase W N → Staircase W N' → ∃ i, P i ∧ N (Y i) ≠ N' (Y' i))
    {A A' L H L' H' : ℚ} {cs cs' : List ℚ} (hLH : L + W ≤ H) (hLH' : L' + W ≤ H')
    (hB : ∀ x, L ≤ x → x + W ≤ H → LateOK W A x cs) (hB' : ∀ x, L' ≤ x → x + W ≤ H' → LateOK W A' x cs')
    (hY : ∀ i, P i → L ≤ Y i ∧ Y i ≤ H) (hY' : ∀ i, P i → L' ≤ Y' i ∧ Y' i ≤ H') :
    ∃ i, P i ∧ chainFrom A (Y i :: cs) ≠ chainFrom A' (Y' i :: cs') := by
  obtain ⟨⟨i, hi⟩, hne⟩ := split_of_staircase (ι := {i // P i}) (Y := fun i => Y i.1) (Y' := fun i => Y' i.1)
    (g := fun x => chainFrom A (x :: cs)) (g' := fun x => chainFrom A' (x :: cs')) hW
    (fun N N' hN hN' => by obtain ⟨i, hi, h⟩ := h N N' hN hN'; exact ⟨⟨i, hi⟩, h⟩)
    (fun x y hxy => chainFrom_mono cs (rnAdd_mono A x A y (by linarith)))
    (fun x y hxy => chainFrom_mono cs' (rnAdd_mono A' x A' y (by linarith))) hLH hLH'
    (fun x h1 h2 => late_shift_ok (hB x h1 h2)) (fun x h1 h2 => late_shift_ok (hB' x h1 h2))
    (fun i => hY i.1 i.2) (fun i => hY' i.1 i.2)
  exact ⟨i, hi, hne⟩

/-- **The straddle lemma**: each column's chain passes `+ W` on two windows, one holding its words below a cut (`κ`,
`κ′`) and one holding those from the cut on. A chain that crosses one power of two at one step does, with the cut
where that step's sum reaches the power. If no four step-`W` staircases, one per column and side, agree on the words,
some index separates the chains. -/
theorem lateWords_split_cut {ι : Type*} {W : ℚ} (hW : 0 < W) {Y Y' : ι → ℚ} {κ κ' : ℚ}
    (h : ∀ N₁ N₂ N₁' N₂', Staircase W N₁ → Staircase W N₂ → Staircase W N₁' → Staircase W N₂' →
      ∃ i, (if Y i < κ then N₁ else N₂) (Y i) ≠ (if Y' i < κ' then N₁' else N₂') (Y' i))
    {A A' L₁ H₁ L₂ H₂ L₁' H₁' L₂' H₂' : ℚ} {cs cs' : List ℚ}
    (h₁ : L₁ + W ≤ H₁) (h₂ : L₂ + W ≤ H₂) (h₁' : L₁' + W ≤ H₁') (h₂' : L₂' + W ≤ H₂')
    (hB₁ : ∀ x, L₁ ≤ x → x + W ≤ H₁ → LateOK W A x cs) (hB₂ : ∀ x, L₂ ≤ x → x + W ≤ H₂ → LateOK W A x cs)
    (hB₁' : ∀ x, L₁' ≤ x → x + W ≤ H₁' → LateOK W A' x cs')
    (hB₂' : ∀ x, L₂' ≤ x → x + W ≤ H₂' → LateOK W A' x cs')
    (hY : ∀ i, (Y i < κ → L₁ ≤ Y i ∧ Y i ≤ H₁) ∧ (κ ≤ Y i → L₂ ≤ Y i ∧ Y i ≤ H₂))
    (hY' : ∀ i, (Y' i < κ' → L₁' ≤ Y' i ∧ Y' i ≤ H₁') ∧ (κ' ≤ Y' i → L₂' ≤ Y' i ∧ Y' i ≤ H₂')) :
    ∃ i, chainFrom A (Y i :: cs) ≠ chainFrom A' (Y' i :: cs') := by
  have hm : ∀ (A : ℚ) (cs : List ℚ), Monotone fun x => chainFrom A (x :: cs) := fun A cs x y hxy =>
    chainFrom_mono cs (rnAdd_mono A x A y (by linarith))
  obtain ⟨N₁, hN₁, e₁⟩ := staircase_extend hW h₁ (hm A cs) fun x a b => late_shift_ok (hB₁ x a b)
  obtain ⟨N₂, hN₂, e₂⟩ := staircase_extend hW h₂ (hm A cs) fun x a b => late_shift_ok (hB₂ x a b)
  obtain ⟨N₁', hN₁', e₁'⟩ := staircase_extend hW h₁' (hm A' cs') fun x a b => late_shift_ok (hB₁' x a b)
  obtain ⟨N₂', hN₂', e₂'⟩ := staircase_extend hW h₂' (hm A' cs') fun x a b => late_shift_ok (hB₂' x a b)
  obtain ⟨i, hi⟩ := h N₁ N₂ N₁' N₂' hN₁ hN₂ hN₁' hN₂'
  refine ⟨i, fun heq => hi ?_⟩
  have hl : (if Y i < κ then N₁ else N₂) (Y i) = ⌊chainFrom A (Y i :: cs) / W⌋ := by
    by_cases hc : Y i < κ
    · rw [ite_eq_left hc]; exact e₁ _ ((hY i).1 hc).1 ((hY i).1 hc).2
    · rw [ite_eq_right hc]; exact e₂ _ ((hY i).2 (not_lt.1 hc)).1 ((hY i).2 (not_lt.1 hc)).2
  have hr : (if Y' i < κ' then N₁' else N₂') (Y' i) = ⌊chainFrom A' (Y' i :: cs') / W⌋ := by
    by_cases hc : Y' i < κ'
    · rw [ite_eq_left hc]; exact e₁' _ ((hY' i).1 hc).1 ((hY' i).1 hc).2
    · rw [ite_eq_right hc]; exact e₂' _ ((hY' i).2 (not_lt.1 hc)).1 ((hY' i).2 (not_lt.1 hc)).2
  rw [hl, hr, heq]

/-- **The tag cut split** (not proved): `TagStaircaseSplit` with each column's words cut at any value (`κ`, `κ′`)
and each side matched by a staircase of its own. It carries the straddle case of the binade windows
(`lateWords_split_cut`). -/
def TagCutSplit (W : ℚ) : Prop :=
  ∀ (xs : Fin 29 → Code) (b b' : ℕ → Code) (j j' τ : ℕ) (pos : Fin 29 → Bool), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) → (∀ r, (b' r).isNaN = false) → tuple j ≠ tuple j' →
    ∀ (κ κ' : ℚ) (N₁ N₂ N₁' N₂' : ℚ → ℤ), Staircase W N₁ → Staircase W N₂ → Staircase W N₁' → Staircase W N₂' →
      ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
        (if tagWord xs b j τ pos neg man < κ then N₁ else N₂) (tagWord xs b j τ pos neg man) ≠
          (if tagWord xs b' j' τ pos neg man < κ' then N₁' else N₂') (tagWord xs b' j' τ pos neg man)

theorem tagStaircaseSplit_of_cut {W : ℚ} (h : TagCutSplit W) : TagStaircaseSplit W := by
  intro xs b b' j j' τ pos hx hb hb' ht N N' hN hN'
  obtain ⟨neg, man, hne⟩ := h xs b b' j j' τ pos hx hb hb' ht 0 0 N N N' N' hN hN hN' hN'
  exact ⟨neg, man, by simpa only [ite_self] using hne⟩

/-- **`TagCutSplit` separates the late words** of one slice's two columns when each chain crosses at most one power
of two at one step (`lateWords_split_cut`'s windows). -/
theorem lateWords_of_cutSplit {W : ℚ} (hW : 0 < W) (hS : TagCutSplit W) {xs : Fin 29 → Code}
    {b b' : ℕ → Code} {j j' τ : ℕ} {pos : Fin 29 → Bool} (hx : ∀ l, (xs l).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (hb' : ∀ r, (b' r).isNaN = false) (ht : tuple j ≠ tuple j') {κ κ' : ℚ}
    {A A' L₁ H₁ L₂ H₂ L₁' H₁' L₂' H₂' : ℚ} {cs cs' : List ℚ}
    (h₁ : L₁ + W ≤ H₁) (h₂ : L₂ + W ≤ H₂) (h₁' : L₁' + W ≤ H₁') (h₂' : L₂' + W ≤ H₂')
    (hB₁ : ∀ x, L₁ ≤ x → x + W ≤ H₁ → LateOK W A x cs) (hB₂ : ∀ x, L₂ ≤ x → x + W ≤ H₂ → LateOK W A x cs)
    (hB₁' : ∀ x, L₁' ≤ x → x + W ≤ H₁' → LateOK W A' x cs')
    (hB₂' : ∀ x, L₂' ≤ x → x + W ≤ H₂' → LateOK W A' x cs')
    (hY : ∀ neg man, (tagWord xs b j τ pos neg man < κ → L₁ ≤ tagWord xs b j τ pos neg man ∧
        tagWord xs b j τ pos neg man ≤ H₁) ∧
      (κ ≤ tagWord xs b j τ pos neg man → L₂ ≤ tagWord xs b j τ pos neg man ∧ tagWord xs b j τ pos neg man ≤ H₂))
    (hY' : ∀ neg man, (tagWord xs b' j' τ pos neg man < κ' → L₁' ≤ tagWord xs b' j' τ pos neg man ∧
        tagWord xs b' j' τ pos neg man ≤ H₁') ∧
      (κ' ≤ tagWord xs b' j' τ pos neg man → L₂' ≤ tagWord xs b' j' τ pos neg man ∧
        tagWord xs b' j' τ pos neg man ≤ H₂')) :
    ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
      chainFrom A (tagWord xs b j τ pos neg man :: cs) ≠ chainFrom A' (tagWord xs b' j' τ pos neg man :: cs') := by
  obtain ⟨⟨neg, man⟩, hne⟩ := lateWords_split_cut (ι := (Fin 3 → Bool) × (Fin 3 → Fin 8))
    (Y := fun s => tagWord xs b j τ pos s.1 s.2) (Y' := fun s => tagWord xs b' j' τ pos s.1 s.2) hW
    (fun N₁ N₂ N₁' N₂' a b c d => by
      obtain ⟨neg, man, h⟩ := hS xs _ _ j j' τ pos hx hb hb' ht κ κ' N₁ N₂ N₁' N₂' a b c d
      exact ⟨(neg, man), h⟩)
    h₁ h₂ h₁' h₂' hB₁ hB₂ hB₁' hB₂' (fun s => hY s.1 s.2) (fun s => hY' s.1 s.2)
  exact ⟨neg, man, hne⟩

/-! ## Two cuts per column -/

/-- The cell of `y` for the cuts `κ₁`, `κ₂`: `0` below `κ₁`, `1` on `[κ₁, κ₂)`, `2` from `max κ₁ κ₂`. -/
def cell3 (κ₁ κ₂ y : ℚ) : Fin 3 := if y < κ₁ then 0 else if y < κ₂ then 1 else 2

/-- **The straddle lemma with two cuts**: each column's chain passes `+ W` on three windows, holding its words below
`κ₁`, on `[κ₁, κ₂)` and from `κ₂`. A chain whose steps cross powers of two in two bands of the words does, with a cut
at each band. If no six step-`W` staircases, one per column and cell, agree on the words, some index separates the
chains. -/
theorem lateWords_split_cut2 {ι : Type*} {W : ℚ} (hW : 0 < W) {Y Y' : ι → ℚ} {κ₁ κ₂ κ₁' κ₂' : ℚ}
    (h : ∀ N N' : Fin 3 → ℚ → ℤ, (∀ c, Staircase W (N c)) → (∀ c, Staircase W (N' c)) →
      ∃ i, N (cell3 κ₁ κ₂ (Y i)) (Y i) ≠ N' (cell3 κ₁' κ₂' (Y' i)) (Y' i))
    {A A' : ℚ} {cs cs' : List ℚ} {L H L' H' : Fin 3 → ℚ}
    (hLH : ∀ c, L c + W ≤ H c) (hLH' : ∀ c, L' c + W ≤ H' c)
    (hB : ∀ c x, L c ≤ x → x + W ≤ H c → LateOK W A x cs)
    (hB' : ∀ c x, L' c ≤ x → x + W ≤ H' c → LateOK W A' x cs')
    (hY : ∀ i, L (cell3 κ₁ κ₂ (Y i)) ≤ Y i ∧ Y i ≤ H (cell3 κ₁ κ₂ (Y i)))
    (hY' : ∀ i, L' (cell3 κ₁' κ₂' (Y' i)) ≤ Y' i ∧ Y' i ≤ H' (cell3 κ₁' κ₂' (Y' i))) :
    ∃ i, chainFrom A (Y i :: cs) ≠ chainFrom A' (Y' i :: cs') := by
  have hm : ∀ (A : ℚ) (cs : List ℚ), Monotone fun x => chainFrom A (x :: cs) := fun A cs x y hxy =>
    chainFrom_mono cs (rnAdd_mono A x A y (by linarith))
  choose N hN e using fun c => staircase_extend hW (hLH c) (hm A cs) fun x a b => late_shift_ok (hB c x a b)
  choose N' hN' e' using fun c => staircase_extend hW (hLH' c) (hm A' cs') fun x a b => late_shift_ok (hB' c x a b)
  obtain ⟨i, hi⟩ := h N N' hN hN'
  refine ⟨i, fun heq => hi ?_⟩
  have hl : N (cell3 κ₁ κ₂ (Y i)) (Y i) = ⌊chainFrom A (Y i :: cs) / W⌋ := e _ _ (hY i).1 (hY i).2
  have hr : N' (cell3 κ₁' κ₂' (Y' i)) (Y' i) = ⌊chainFrom A' (Y' i :: cs') / W⌋ := e' _ _ (hY' i).1 (hY' i).2
  rw [hl, hr, heq]

/-- **The straddle lemma with two cuts, on a sub-family**: `lateWords_split_cut2` with the windows needed only for the
indices in `P` and the staircase split taken over them. It lets words that lie in no window be dropped. -/
theorem lateWords_split_cut2_on {ι : Type*} {W : ℚ} (hW : 0 < W) {Y Y' : ι → ℚ} {P : ι → Prop}
    {κ₁ κ₂ κ₁' κ₂' : ℚ}
    (h : ∀ N N' : Fin 3 → ℚ → ℤ, (∀ c, Staircase W (N c)) → (∀ c, Staircase W (N' c)) →
      ∃ i, P i ∧ N (cell3 κ₁ κ₂ (Y i)) (Y i) ≠ N' (cell3 κ₁' κ₂' (Y' i)) (Y' i))
    {A A' : ℚ} {cs cs' : List ℚ} {L H L' H' : Fin 3 → ℚ}
    (hLH : ∀ c, L c + W ≤ H c) (hLH' : ∀ c, L' c + W ≤ H' c)
    (hB : ∀ c x, L c ≤ x → x + W ≤ H c → LateOK W A x cs)
    (hB' : ∀ c x, L' c ≤ x → x + W ≤ H' c → LateOK W A' x cs')
    (hY : ∀ i, P i → L (cell3 κ₁ κ₂ (Y i)) ≤ Y i ∧ Y i ≤ H (cell3 κ₁ κ₂ (Y i)))
    (hY' : ∀ i, P i → L' (cell3 κ₁' κ₂' (Y' i)) ≤ Y' i ∧ Y' i ≤ H' (cell3 κ₁' κ₂' (Y' i))) :
    ∃ i, P i ∧ chainFrom A (Y i :: cs) ≠ chainFrom A' (Y' i :: cs') := by
  obtain ⟨⟨i, hi⟩, hne⟩ := lateWords_split_cut2 (ι := {i // P i}) (Y := fun i => Y i.1) (Y' := fun i => Y' i.1)
    hW (fun N N' hN hN' => by obtain ⟨i, hi, h⟩ := h N N' hN hN'; exact ⟨⟨i, hi⟩, h⟩)
    hLH hLH' hB hB' (fun i => hY i.1 i.2) (fun i => hY' i.1 i.2)
  exact ⟨i, hi, hne⟩

/-- **The two-cut tag split** (not proved): `TagCutSplit` with each column's words cut twice and each of the three
cells matched by a staircase of its own. It carries two straddles per column (`lateWords_split_cut2`) and implies
`TagCutSplit W`. -/
def TagCutSplit2 (W : ℚ) : Prop :=
  ∀ (xs : Fin 29 → Code) (b b' : ℕ → Code) (j j' τ : ℕ) (pos : Fin 29 → Bool), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) → (∀ r, (b' r).isNaN = false) → tuple j ≠ tuple j' →
    ∀ (κ₁ κ₂ κ₁' κ₂' : ℚ) (N N' : Fin 3 → ℚ → ℤ), (∀ c, Staircase W (N c)) → (∀ c, Staircase W (N' c)) →
      ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
        N (cell3 κ₁ κ₂ (tagWord xs b j τ pos neg man)) (tagWord xs b j τ pos neg man) ≠
          N' (cell3 κ₁' κ₂' (tagWord xs b' j' τ pos neg man)) (tagWord xs b' j' τ pos neg man)

theorem tagCutSplit_of_cutSplit2 {W : ℚ} (h : TagCutSplit2 W) : TagCutSplit W := by
  intro xs b b' j j' τ pos hx hb hb' ht κ κ' N₁ N₂ N₁' N₂' h₁ h₂ h₁' h₂'
  have h3 : ∀ {M₁ M₂ : ℚ → ℤ}, Staircase W M₁ → Staircase W M₂ → ∀ c, Staircase W ((![M₁, M₂, M₂] : Fin 3 → ℚ → ℤ) c) :=
    fun h₁ h₂ c => by
      rcases c with ⟨_ | _ | _ | n, hc⟩
      · exact h₁
      · exact h₂
      · exact h₂
      · exact absurd hc (by omega)
  obtain ⟨neg, man, hne⟩ := h xs b b' j j' τ pos hx hb hb' ht κ κ κ' κ' ![N₁, N₂, N₂] ![N₁', N₂', N₂']
    (h3 h₁ h₂) (h3 h₁' h₂')
  have e : ∀ (κ y : ℚ) (M₁ M₂ : ℚ → ℤ),
      (![M₁, M₂, M₂] : Fin 3 → ℚ → ℤ) (cell3 κ κ y) y = (if y < κ then M₁ else M₂) y := by
    intro κ y M₁ M₂
    by_cases hy : y < κ
    · have hc : cell3 κ κ y = 0 := ite_eq_left hy
      rw [hc, ite_eq_left hy]; rfl
    · have hc : cell3 κ κ y = 2 := by unfold cell3; rw [ite_eq_right hy, ite_eq_right hy]
      rw [hc, ite_eq_right hy]; rfl
  exact ⟨neg, man, by rw [← e, ← e]; exact hne⟩

/-- **`TagCutSplit2` separates the late words** of one slice's two columns when each chain passes `+ W` on three
windows per column (`lateWords_split_cut2`). -/
theorem lateWords_of_cutSplit2 {W : ℚ} (hW : 0 < W) (hS : TagCutSplit2 W) {xs : Fin 29 → Code}
    {b b' : ℕ → Code} {j j' τ : ℕ} {pos : Fin 29 → Bool} (hx : ∀ l, (xs l).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (hb' : ∀ r, (b' r).isNaN = false) (ht : tuple j ≠ tuple j')
    {κ₁ κ₂ κ₁' κ₂' A A' : ℚ} {cs cs' : List ℚ} {L H L' H' : Fin 3 → ℚ}
    (hLH : ∀ c, L c + W ≤ H c) (hLH' : ∀ c, L' c + W ≤ H' c)
    (hB : ∀ c x, L c ≤ x → x + W ≤ H c → LateOK W A x cs)
    (hB' : ∀ c x, L' c ≤ x → x + W ≤ H' c → LateOK W A' x cs')
    (hY : ∀ neg man, L (cell3 κ₁ κ₂ (tagWord xs b j τ pos neg man)) ≤ tagWord xs b j τ pos neg man ∧
      tagWord xs b j τ pos neg man ≤ H (cell3 κ₁ κ₂ (tagWord xs b j τ pos neg man)))
    (hY' : ∀ neg man, L' (cell3 κ₁' κ₂' (tagWord xs b' j' τ pos neg man)) ≤ tagWord xs b' j' τ pos neg man ∧
      tagWord xs b' j' τ pos neg man ≤ H' (cell3 κ₁' κ₂' (tagWord xs b' j' τ pos neg man))) :
    ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
      chainFrom A (tagWord xs b j τ pos neg man :: cs) ≠ chainFrom A' (tagWord xs b' j' τ pos neg man :: cs') := by
  obtain ⟨⟨neg, man⟩, hne⟩ := lateWords_split_cut2 (ι := (Fin 3 → Bool) × (Fin 3 → Fin 8))
    (Y := fun s => tagWord xs b j τ pos s.1 s.2) (Y' := fun s => tagWord xs b' j' τ pos s.1 s.2) hW
    (fun N N' hN hN' => by
      obtain ⟨neg, man, h⟩ := hS xs _ _ j j' τ pos hx hb hb' ht κ₁ κ₂ κ₁' κ₂' N N' hN hN'
      exact ⟨(neg, man), h⟩)
    hLH hLH' hB hB' (fun s => hY s.1 s.2) (fun s => hY' s.1 s.2)
  exact ⟨neg, man, hne⟩

/-- `y` lies in one of the `z` closed intervals `[e i, e i + w]`. -/
def InZones {z : ℕ} (w : ℚ) (e : Fin z → ℚ) (y : ℚ) : Prop := ∃ i, e i ≤ y ∧ y ≤ e i + w

/-- **The two-cut tag split off exclusion zones** (not proved): `TagCutSplit2` with the separating tag setting's two
words outside given exclusion sets, in each column `z` intervals of width `w` placed anywhere. The words that lie in
no window, near a crossing, form such sets, so the windows are needed only off them (`lateWords_of_cutSplit2Off`).
It implies `TagCutSplit2 W`, and fewer or narrower zones (`tagCutSplit2Off_mono`). -/
def TagCutSplit2Off (W : ℚ) (z : ℕ) (w : ℚ) : Prop :=
  ∀ (xs : Fin 29 → Code) (b b' : ℕ → Code) (j j' τ : ℕ) (pos : Fin 29 → Bool), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) → (∀ r, (b' r).isNaN = false) → tuple j ≠ tuple j' →
    ∀ (e e' : Fin z → ℚ) (κ₁ κ₂ κ₁' κ₂' : ℚ) (N N' : Fin 3 → ℚ → ℤ), (∀ c, Staircase W (N c)) →
      (∀ c, Staircase W (N' c)) →
      ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
        ¬ InZones w e (tagWord xs b j τ pos neg man) ∧ ¬ InZones w e' (tagWord xs b' j' τ pos neg man) ∧
          N (cell3 κ₁ κ₂ (tagWord xs b j τ pos neg man)) (tagWord xs b j τ pos neg man) ≠
            N' (cell3 κ₁' κ₂' (tagWord xs b' j' τ pos neg man)) (tagWord xs b' j' τ pos neg man)

theorem tagCutSplit2_of_off {W w : ℚ} {z : ℕ} (h : TagCutSplit2Off W z w) : TagCutSplit2 W := by
  intro xs b b' j j' τ pos hx hb hb' ht κ₁ κ₂ κ₁' κ₂' N N' hN hN'
  obtain ⟨neg, man, -, -, hne⟩ := h xs b b' j j' τ pos hx hb hb' ht 0 0 κ₁ κ₂ κ₁' κ₂' N N' hN hN'
  exact ⟨neg, man, hne⟩

theorem tagCutSplit2Off_mono {W w w' : ℚ} {z z' : ℕ} (hz : z' ≤ z) (hw : w' ≤ w) (h : TagCutSplit2Off W z w) :
    TagCutSplit2Off W z' w' := by
  intro xs b b' j j' τ pos hx hb hb' ht e e' κ₁ κ₂ κ₁' κ₂' N N' hN hN'
  let pad : (Fin z' → ℚ) → Fin z → ℚ := fun f i => if hi : (i : ℕ) < z' then f ⟨i, hi⟩ else 0
  have hpad : ∀ (f : Fin z' → ℚ) y, InZones w' f y → InZones w (pad f) y := by
    rintro f y ⟨i, h₁, h₂⟩
    refine ⟨⟨i, lt_of_lt_of_le i.2 hz⟩, ?_, ?_⟩ <;> simp only [pad, i.2, ↓reduceDIte] <;> linarith
  obtain ⟨neg, man, h₁, h₂, hne⟩ := h xs b b' j j' τ pos hx hb hb' ht (pad e) (pad e') κ₁ κ₂ κ₁' κ₂' N N' hN hN'
  exact ⟨neg, man, fun h => h₁ (hpad e _ h), fun h => h₂ (hpad e' _ h), hne⟩

/-- **`TagCutSplit2Off` separates the late words** of one slice's two columns when every word outside the exclusion
zones lies in one of three windows per column (`lateWords_split_cut2_on`); the words inside them need no window. -/
theorem lateWords_of_cutSplit2Off {W w : ℚ} {z : ℕ} (hW : 0 < W) (hS : TagCutSplit2Off W z w)
    {xs : Fin 29 → Code} {b b' : ℕ → Code} {j j' τ : ℕ} {pos : Fin 29 → Bool} (hx : ∀ l, (xs l).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (hb' : ∀ r, (b' r).isNaN = false) (ht : tuple j ≠ tuple j')
    {e e' : Fin z → ℚ} {κ₁ κ₂ κ₁' κ₂' A A' : ℚ} {cs cs' : List ℚ} {L H L' H' : Fin 3 → ℚ}
    (hLH : ∀ c, L c + W ≤ H c) (hLH' : ∀ c, L' c + W ≤ H' c)
    (hB : ∀ c x, L c ≤ x → x + W ≤ H c → LateOK W A x cs)
    (hB' : ∀ c x, L' c ≤ x → x + W ≤ H' c → LateOK W A' x cs')
    (hY : ∀ neg man, ¬ InZones w e (tagWord xs b j τ pos neg man) →
      L (cell3 κ₁ κ₂ (tagWord xs b j τ pos neg man)) ≤ tagWord xs b j τ pos neg man ∧
        tagWord xs b j τ pos neg man ≤ H (cell3 κ₁ κ₂ (tagWord xs b j τ pos neg man)))
    (hY' : ∀ neg man, ¬ InZones w e' (tagWord xs b' j' τ pos neg man) →
      L' (cell3 κ₁' κ₂' (tagWord xs b' j' τ pos neg man)) ≤ tagWord xs b' j' τ pos neg man ∧
        tagWord xs b' j' τ pos neg man ≤ H' (cell3 κ₁' κ₂' (tagWord xs b' j' τ pos neg man))) :
    ∃ (neg : Fin 3 → Bool) (man : Fin 3 → Fin 8),
      chainFrom A (tagWord xs b j τ pos neg man :: cs) ≠ chainFrom A' (tagWord xs b' j' τ pos neg man :: cs') := by
  obtain ⟨⟨neg, man⟩, -, hne⟩ := lateWords_split_cut2_on (ι := (Fin 3 → Bool) × (Fin 3 → Fin 8))
    (Y := fun s => tagWord xs b j τ pos s.1 s.2) (Y' := fun s => tagWord xs b' j' τ pos s.1 s.2)
    (P := fun s => ¬ InZones w e (tagWord xs b j τ pos s.1 s.2) ∧ ¬ InZones w e' (tagWord xs b' j' τ pos s.1 s.2)) hW
    (fun N N' hN hN' => by
      obtain ⟨neg, man, h₁, h₂, h⟩ := hS xs _ _ j j' τ pos hx hb hb' ht e e' κ₁ κ₂ κ₁' κ₂' N N' hN hN'
      exact ⟨(neg, man), ⟨h₁, h₂⟩, h⟩)
    hLH hLH' hB hB' (fun s hs => hY s.1 s.2 hs.1) (fun s hs => hY' s.1 s.2 hs.2)
  exact ⟨neg, man, hne⟩

/-! ## Small magnitudes: the rounding margin, across any binades -/

theorem rn_err_pow {y : ℚ} {e : ℤ} (h : |y| < 2 ^ e) : |rn y - y| ≤ 2 ^ (e - 25) := by
  rcases eq_or_ne y 0 with rfl | hy
  · have : rn 0 = 0 := rn_exact ⟨0, 0, by norm_num, by simp⟩
    rw [this, sub_zero, abs_zero]; exact (two_zpow_pos _).le
  · have hb := (binade_spec (abs_pos.2 hy)).1
    have hlt : binade |y| < e := (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℚ) < 2)).1 (lt_of_le_of_lt hb h)
    have hu : ulp y ≤ 2 ^ (e - 24) := by
      unfold ulp; exact zpow_le_zpow_right₀ (by norm_num) (by omega)
    have e2 : (2 : ℚ) ^ (e - 24) = 2 * 2 ^ (e - 25) := by
      rw [show e - 24 = (e - 25) + 1 by ring, zpow_add_one₀ two_ne_zero]; ring
    linarith [Proofs.rn_err y]

/-- Every sum of the chain from `a` through `cs` is below `2^e` in magnitude. -/
def SumsBelow (e : ℤ) (a : ℚ) (cs : List ℚ) : Prop :=
  ∀ (k : ℕ) (c : ℚ), cs[k]? = some c → |chainFrom a (cs.take k) + c| < 2 ^ e

/-- A chain whose sums stay below `2^e` is within `2^(e − 25)` per step of its exact sum. -/
theorem chainFrom_err {e : ℤ} : ∀ (cs : List ℚ) (a : ℚ), SumsBelow e a cs →
    |chainFrom a cs - (a + cs.sum)| ≤ cs.length * 2 ^ (e - 25)
  | [], a, _ => by simp [chainFrom]
  | c :: cs, a, h => by
    have h0 : |a + c| < 2 ^ e := by simpa [chainFrom] using h 0 c (by simp)
    have ih := chainFrom_err cs (rnAdd a c) fun k c' hk => by
      simpa [chainFrom] using h (k + 1) c' (by simpa using hk)
    have e1 : |rnAdd a c - (a + c)| ≤ 2 ^ (e - 25) := rn_err_pow h0
    rw [chainFrom_cons, List.sum_cons, List.length_cons]
    have ea : chainFrom (rnAdd a c) cs - (a + (c + cs.sum)) =
        (chainFrom (rnAdd a c) cs - (rnAdd a c + cs.sum)) + (rnAdd a c - (a + c)) := by ring
    rw [ea]; push_cast
    calc _ ≤ |chainFrom (rnAdd a c) cs - (rnAdd a c + cs.sum)| + |rnAdd a c - (a + c)| := abs_add_le _ _
      _ ≤ cs.length * 2 ^ (e - 25) + 2 ^ (e - 25) := add_le_add ih e1
      _ = (cs.length + 1) * 2 ^ (e - 25) := by ring

/-- **The margin at small magnitudes**: two chains from `A`, `A′` whose first increments are `Y i`, `Y′ i` and whose
later increments do not depend on `i`. If, at two indices `s` and `t`, every sum of both chains stays below `2^e`,
and the words' differences `Y′ − Y` at `s` and `t` differ by more than the rounding can absorb, one of the two
separates the chains. The sums may change binade at every step. -/
theorem lateWords_margin {ι : Type*} {Y Y' : ι → ℚ} {A A' : ℚ} {cs cs' : List ℚ} {e : ℤ} (s t : ι)
    (hs : SumsBelow e A (Y s :: cs)) (ht : SumsBelow e A (Y t :: cs))
    (hs' : SumsBelow e A' (Y' s :: cs')) (ht' : SumsBelow e A' (Y' t :: cs'))
    (hD : ((cs.length : ℚ) + cs'.length + 2) * 2 ^ (e - 24) < |(Y' s - Y s) - (Y' t - Y t)|) :
    chainFrom A (Y s :: cs) ≠ chainFrom A' (Y' s :: cs') ∨
      chainFrom A (Y t :: cs) ≠ chainFrom A' (Y' t :: cs') := by
  by_contra hne
  simp only [not_or, not_not] at hne
  obtain ⟨e1, e2⟩ := hne
  have a1 := chainFrom_err _ _ hs
  have a2 := chainFrom_err _ _ ht
  have a3 := chainFrom_err _ _ hs'
  have a4 := chainFrom_err _ _ ht'
  simp only [List.sum_cons, List.length_cons] at a1 a2 a3 a4
  push_cast at a1 a2 a3 a4
  rw [abs_le] at a1 a2 a3 a4
  have h25 : (2 : ℚ) ^ (e - 24) = 2 * 2 ^ (e - 25) := by
    rw [show e - 24 = (e - 25) + 1 by ring, zpow_add_one₀ two_ne_zero]; ring
  rw [h25] at hD
  have hle : |(Y' s - Y s) - (Y' t - Y t)| ≤ ((cs.length : ℚ) + cs'.length + 2) * (2 * 2 ^ (e - 25)) := by
    rw [abs_le]; constructor <;> nlinarith [a1.1, a1.2, a2.1, a2.2, a3.1, a3.2, a4.1, a4.2]
  linarith

end Pouw.Fp8Atom.H1T.Shift
