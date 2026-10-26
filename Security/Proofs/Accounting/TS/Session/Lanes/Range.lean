import Proofs.Accounting.TS.Session.Lanes.Decode
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Zify

/-!
# The decodes' range lemmas: decoding an encoded in-range word gives it back

A field's decode (`Accounting.TS.Lanes.decLane`, `decComb`) reads its bits through `bitsN` and, for a signed word,
`sgnBits`. Where a reader holds a word's bits (`Holds`: bit `i` of the field is bit `i` of the word):

* `bitsN_eq`: the bits of `x < 2^n` spell `x`, and `bitsN_eq_mod` in general `x mod 2^n`;
* `sgnBits_bitsN`: the bits of a signed word `v` with `-2^(n-1) ≤ v < 2^(n-1)`, in two's complement (`encS n v`), decode
  to `v` (s8 codes, s32 z);
* `bitsN_cast`: an unsigned word `0 ≤ v < 2^n` decodes to `v` (a scale's f32 or a y's bf16 encoding);
* `ofNat_bitsN`: a u32 word's bits decode to it (partials and receipt words).

So the decodes give a leaf back from rows that hold its words' encodings; that the known ports' rows hold their
content is flock-verify's known-read check (`Lanes.KnownRead`). The leaf `LeafLanesSound` opens is the one the data
ports' rows decode to (`Lanes.Committed`), so nothing is asked of those rows.
-/

namespace Accounting.TS.Lanes

/-- **A reader holds `x`'s low `n` bits at `[lo, lo + n)`**: bit `i` of the field is bit `i` of `x`. -/
def Holds (rd : ℕ → Option Bool) (lo n x : ℕ) : Prop := ∀ i < n, rd (lo + i) = some (x.testBit i)

/-- A signed word's `n`-bit two's complement. -/
def encS (n : ℕ) (v : ℤ) : ℕ := (v % 2 ^ n).toNat

/-- `bitsN` one bit longer. -/
theorem bitsN_succ (rd : ℕ → Option Bool) (lo n : ℕ) :
    bitsN rd lo (n + 1) = bitsN rd lo n + (if rd (lo + n) = some true then 2 ^ n else 0) := by
  unfold bitsN
  rw [Finset.sum_range_succ]

/-- **A held word's bits spell it modulo `2^n`.** -/
theorem bitsN_eq_mod {rd : ℕ → Option Bool} {lo x : ℕ} : ∀ {n : ℕ}, Holds rd lo n x → bitsN rd lo n = x % 2 ^ n
  | 0, _ => by simp [bitsN, Nat.mod_one]
  | n + 1, h => by
    rw [bitsN_succ, bitsN_eq_mod (fun i hi => h i (by omega)), h n (by omega), Nat.mod_pow_succ]
    congr 1
    rw [Nat.testBit_eq_decide_div_mod_eq]
    rcases Nat.mod_two_eq_zero_or_one (x / 2 ^ n) with h0 | h1
    · simp [h0]
    · simp [h1]

/-- **A held word below `2^n` decodes to itself.** -/
theorem bitsN_eq {rd : ℕ → Option Bool} {lo n x : ℕ} (h : Holds rd lo n x) (hx : x < 2 ^ n) : bitsN rd lo n = x := by
  rw [bitsN_eq_mod h, Nat.mod_eq_of_lt hx]

/-- **Two's complement round-trips** on `[-2^(n-1), 2^(n-1))`. -/
theorem sgnBits_encS {n : ℕ} {v : ℤ} (hn : 0 < n) (hlo : -2 ^ (n - 1) ≤ v) (hhi : v < 2 ^ (n - 1)) :
    sgnBits n (encS n v) = v := by
  have hp : (2 : ℤ) ^ n = 2 * 2 ^ (n - 1) := by
    rw [← pow_succ']
    congr 1
    omega
  have hpos : (0 : ℤ) < 2 ^ (n - 1) := by positivity
  unfold sgnBits encS
  by_cases hv : 0 ≤ v
  · have e : v % 2 ^ n = v := Int.emod_eq_of_lt hv (by omega)
    have ht : ((v % 2 ^ n).toNat : ℤ) = v := by rw [e]; exact Int.toNat_of_nonneg hv
    have hlt : (v % 2 ^ n).toNat < 2 ^ (n - 1) := by
      zify
      rw [ht]
      exact hhi
    simp only [hlt, ↓reduceIte]
    exact ht
  · have e : v % 2 ^ n = v + 2 ^ n := by
      rw [show v % 2 ^ n = (v + 2 ^ n) % 2 ^ n from (Int.add_emod_right v (2 ^ n)).symm]
      exact Int.emod_eq_of_lt (by omega) (by omega)
    have ht : ((v % 2 ^ n).toNat : ℤ) = v + 2 ^ n := by rw [e]; exact Int.toNat_of_nonneg (by omega)
    have hge : ¬ (v % 2 ^ n).toNat < 2 ^ (n - 1) := by
      zify
      rw [ht]
      omega
    simp only [hge, ↓reduceIte]
    rw [ht]
    omega

/-- A two's complement is below `2^n`. -/
theorem encS_lt (n : ℕ) (v : ℤ) : encS n v < 2 ^ n := by
  unfold encS
  have hpos : (0 : ℤ) < 2 ^ n := by positivity
  have h1 := Int.emod_nonneg v hpos.ne'
  have h2 := Int.emod_lt_of_pos v hpos
  zify
  rw [Int.toNat_of_nonneg h1]
  exact h2

/-- **A held u32 word decodes to itself.** -/
theorem ofNat_bitsN {rd : ℕ → Option Bool} {lo : ℕ} {w : BitVec 32} (h : Holds rd lo 32 w.toNat) :
    BitVec.ofNat 32 (bitsN rd lo 32) = w := by
  rw [bitsN_eq h w.isLt]
  exact BitVec.eq_of_toNat_eq (by rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt w.isLt])

/-- **A held unsigned word below `2^n` decodes to itself**, as an integer. -/
theorem bitsN_cast {rd : ℕ → Option Bool} {lo n : ℕ} {v : ℤ} (h0 : 0 ≤ v) (hv : v < 2 ^ n) (h : Holds rd lo n v.toNat) :
    (bitsN rd lo n : ℤ) = v := by
  have hlt : v.toNat < 2 ^ n := by zify; rw [Int.toNat_of_nonneg h0]; exact hv
  rw [bitsN_eq h hlt, Int.toNat_of_nonneg h0]

/-- **A held signed word in range decodes to itself.** -/
theorem sgnBits_bitsN {rd : ℕ → Option Bool} {lo n : ℕ} {v : ℤ} (hn : 0 < n) (hlo : -2 ^ (n - 1) ≤ v)
    (hhi : v < 2 ^ (n - 1)) (h : Holds rd lo n (encS n v)) : sgnBits n (bitsN rd lo n) = v := by
  rw [bitsN_eq h (encS_lt n v)]
  exact sgnBits_encS hn hlo hhi

end Accounting.TS.Lanes
