module

public import Proofs.Pouw.PearlPlus.Words
public import Proofs.Pouw.PearlPlus.Rho

/-!
# Pearl+'s bridge, part (b), the row statistics: `s5_stats` is `pc8`'s

`Words.s5Stats` folds a row's `max(m, w & 0x7FFFFFFF)`, runs the FMA partials of `x²` over every 8th word into 32
accumulators and their butterfly, and forms `ρ = √(sum / n)`, `α = 448 / max(s + 4ρ, 2⁻³²)` and
`β = ρ·α·2^(−19/2)`. `statsAreTheRow` says these are `pc8`'s `rowRho`, `alpha8` and `beta8` on core's primitives at
the row's values re-encoded.

Each step is the row's primitive on the same words: the add where its sum is finite (`fp32AddIsTheSpec`, the check's
`finite` guarding every sum), the root where ρ is finite (`sqrtIsTheSpec`, `statsRhoIsFinite`), and core's multiply,
divide and FMA as they are. Every word stays sign-clear (`Pos`), so the check's integer `max` is `F32Fmaxf` and its fold
is the row's `I32Le`/`SelectI32` max; `ofInt n` is `f32Word n` up to `n = 2^24`. A raw word and its re-encoding differ
only at `−0`, which none of the formulas tells from `+0` (`WholeCircuit.alpha8_zeroSign`).
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

namespace StatsSim

open Pouw.PearlC.RowCircuit WordsSim

/-! ### Sign-clear words stay sign-clear -/

/-- A decoded word with its sign bit clear: a NaN, `+inf` or a non-negative finite. -/
def Pos : Verity.Fp32.Decoded → Prop
  | .nan => True
  | .inf s => s = false
  | .fin s _ => s = false

theorem decode_pos {u : ℕ} (hu : u < 2 ^ 31) : Pos (Verity.Fp32.decode u) := by
  have hs : ((u >>> 31) &&& 1 == 1) = false := by
    rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt hu]; rfl
  unfold Verity.Fp32.decode
  simp only [hs]
  split_ifs <;> first | exact True.intro | exact rfl

theorem roundRn_lt31 (n d : ℕ) : Verity.Fp32.roundRn false n d < 2 ^ 31 := by
  unfold Verity.Fp32.roundRn
  dsimp only
  rw [show Verity.Fp32.zeroWord false = 0 from rfl, Nat.zero_or]
  have key : ∀ w : ℕ, (if w ≥ 0x7F800000 then 0x7F800000 else w) < 2 ^ 31 := fun w => by split <;> omega
  exact key _

theorem addRn_lt31 {nan : ℕ} (d : ℕ) (hn : nan < 2 ^ 31) {x y : Verity.Fp32.Decoded} (hx : Pos x) (hy : Pos y) :
    x.addRn nan d y < 2 ^ 31 := by
  cases x <;> cases y <;> (try simp only [Pos] at hx hy) <;> subst_vars <;>
    simp only [Verity.Fp32.Decoded.addRn, Verity.Fp32.roundSum, ite_true] <;>
    first | exact hn | decide | exact roundRn_lt31 _ _

theorem pos_mul {x y : Verity.Fp32.Decoded} (hx : Pos x) (hy : Pos y) : Pos (x.mul y) := by
  cases x <;> cases y <;> (try simp only [Pos] at hx hy) <;> subst_vars <;>
    simp only [Verity.Fp32.Decoded.mul] <;> (try split) <;> first | exact True.intro | exact rfl

theorem pos_mul_self (x : Verity.Fp32.Decoded) : Pos (x.mul x) := by
  cases x with
  | nan => exact True.intro
  | inf s => cases s <;> exact rfl
  | fin s m => cases s <;> exact rfl

theorem addV2_lt31 {a b : ℕ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) : Verity.Fp32.f32AddV2 a b < 2 ^ 31 :=
  addRn_lt31 1 (by decide) (decode_pos ha) (decode_pos hb)

theorem mulV2_lt31 {a b : ℕ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) : Verity.Fp32.f32MulV2 a b < 2 ^ 31 := by
  have hp := pos_mul (decode_pos ha) (decode_pos hb)
  unfold Verity.Fp32.f32MulV2
  generalize (Verity.Fp32.decode a).mul (Verity.Fp32.decode b) = d at hp ⊢
  cases d with
  | nan => decide
  | inf s => simp only [Pos] at hp; subst hp; decide
  | fin s m => simp only [Pos] at hp; subst hp; exact roundRn_lt31 _ _

theorem divV2_lt31 {a b : ℕ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) : Verity.Fp32.f32DivV2 a b < 2 ^ 31 := by
  have hx := decode_pos ha
  have hy := decode_pos hb
  unfold Verity.Fp32.f32DivV2
  generalize Verity.Fp32.decode a = da at hx ⊢
  generalize Verity.Fp32.decode b = db at hy ⊢
  cases da <;> cases db <;> (try simp only [Pos] at hx hy) <;> subst_vars <;> dsimp only <;> (try split_ifs) <;>
    first | decide | exact roundRn_lt31 _ _

theorem fmaSq_lt31 (a : ℕ) {c : ℕ} (hc : c < 2 ^ 31) : Verity.Fp32.f32FmaV2 a a c < 2 ^ 31 := by
  unfold Verity.Fp32.f32FmaV2 Verity.Fp32.f32GpuNan
  split
  · decide
  unfold Verity.Fp32.f32FmaBits
  dsimp only
  refine addRn_lt31 _ (by decide) (pos_mul_self _) ?_
  have hp := decode_pos hc
  generalize Verity.Fp32.decode c = dc at hp ⊢
  cases dc with
  | nan => exact True.intro
  | inf s => exact hp
  | fin s m => exact (show s = false from hp)

theorem sqrtWord_lt31 {a : ℕ} (ha : a < 2 ^ 31) : sqrtWord a < 2 ^ 31 := by
  unfold sqrtWord
  dsimp only
  rw [Nat.mod_eq_of_lt (show a < 2 ^ 32 by omega)]
  split_ifs <;> first | omega | (apply f32Word_lt31; positivity)

/-! ### `ofInt` of the row's partial count -/

theorem ofInt_lt31 {n : ℕ} (h0 : 0 < n) (h : n ≤ 2 ^ 24) : Words.ofInt n < 2 ^ 31 := by
  have ha : n ≠ 0 := by omega
  have hl1 := Nat.log2_self_le ha
  have hl2 := Nat.lt_log2_self (n := n)
  have hl3 : n.log2 < 25 := (Nat.log2_lt ha).2 (by omega)
  obtain ⟨e, he⟩ : ∃ e, n.log2 = e := ⟨_, rfl⟩
  rw [he] at hl1 hl2 hl3
  have hd : n - 2 ^ e < 2 ^ e := by rw [Nat.pow_succ] at hl2; omega
  have hB : (n - 2 ^ e) <<< (23 - e) < 2 ^ 31 := by
    rw [Nat.shiftLeft_eq]
    rcases Nat.lt_or_ge e 24 with h24 | h24
    · calc (n - 2 ^ e) * 2 ^ (23 - e) < 2 ^ e * 2 ^ (23 - e) := Nat.mul_lt_mul_of_pos_right hd (Nat.two_pow_pos _)
        _ = 2 ^ 23 := by rw [← Nat.pow_add]; congr 1; omega
        _ < 2 ^ 31 := by norm_num
    · obtain rfl : e = 24 := by omega
      rw [show n - 2 ^ 24 = 0 by omega, Nat.zero_mul]; norm_num
  have hA : (e + 127) <<< 23 < 2 ^ 31 := by rw [Nat.shiftLeft_eq]; omega
  unfold Words.ofInt
  simp only [Int.natAbs_natCast, he, ha, ↓reduceIte, show ¬ (n : ℤ) < 0 by omega, Nat.zero_or]
  exact Nat.or_lt_two_pow hA hB

theorem ofInt_wordQ {n : ℕ} (h0 : 0 < n) (h : n ≤ 2 ^ 24) : wordQ (Words.ofInt n) = some (n : ℚ) := by
  rcases Nat.lt_or_ge n (2 ^ 24) with hlt | hge
  · have hs := (NoiseProofs.ofInt_spec (z := (n : ℤ)) (by omega) (by rw [Int.natAbs_natCast]; exact hlt)).1
    rw [hs]; norm_cast
  · obtain rfl : n = 2 ^ 24 := le_antisymm h hge
    have hw : Words.ofInt ((2 ^ 24 : ℕ) : ℤ) = 0 * 2 ^ 31 + 151 * 2 ^ 23 + 0 := by
      unfold Words.ofInt
      simp only [Int.natAbs_natCast, Nat.log2_two_pow]
      decide
    rw [hw, NoiseProofs.wordQ_word (by norm_num) (by norm_num) (by norm_num) (by norm_num)]
    norm_num

theorem ofInt_eq {n : ℕ} (h0 : 0 < n) (h : n ≤ 2 ^ 24) : Words.ofInt n = f32Word (n : ℚ) := by
  have hq := ofInt_wordQ h0 h
  rw [← f32Word_wordVal (stepWord_of_lt31 (ofInt_lt31 h0 h) hq), wordVal_of_wordQ hq]

/-! ### The row's primitives on core's words -/

theorem eval_add (a b : ℕ) : Prim.eval coreOps .add [a, b] = TileCircuit.f32Add a b % 2 ^ 32 := rfl
theorem eval_mul (a b : ℕ) : Prim.eval coreOps .mul [a, b] = Verity.Fp32.f32MulV2 a b % 2 ^ 32 := rfl
theorem eval_div (a b : ℕ) : Prim.eval coreOps .div [a, b] = Verity.Fp32.f32DivV2 a b % 2 ^ 32 := rfl
theorem eval_fma (a b c : ℕ) : Prim.eval coreOps .fma [a, b, c] = Verity.Fp32.f32FmaV2 a b c % 2 ^ 32 := rfl
theorem eval_sqrt (a : ℕ) : Prim.eval coreOps .sqrt [a] = sqrtWord a % 2 ^ 32 := rfl
theorem eval_fmaxf (a b : ℕ) : Prim.eval coreOps .fmaxf [a, b] = coreOps.f32Fmaxf a b % 2 ^ 32 := rfl
theorem eval_le (a b : ℕ) : Prim.eval coreOps .le [a, b] = coreOps.i32Le a b % 2 := rfl
theorem eval_sel (c a b : ℕ) : Prim.eval coreOps .selI32 [c, a, b] = coreOps.selectI32 c a b % 2 ^ 32 := rfl

/-- A finite sum of sign-clear words is the row's `add`, and sign-clear. -/
theorem add_eval {a b v : ℕ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31)
    (h : Words.finite (Verity.Fp32.f32AddV2 a b) = .ok v) : v = Prim.eval coreOps .add [a, b] ∧ v < 2 ^ 31 := by
  have hs := fp32AddIsTheSpec a b
  rw [show Evaluate.Fp32.f32AddV2 = Verity.Fp32.f32AddV2 from Properties.Evaluator.Fp32.f32AddV2] at hs
  have he := (hs (Rho.wordQ_of_finite h)).2.2
  obtain ⟨rfl, -⟩ := finite_ok h
  have hlt := addV2_lt31 ha hb
  refine ⟨?_, hlt⟩
  rw [eval_add, ← he, Nat.mod_eq_of_lt (show Verity.Fp32.f32AddV2 a b < 2 ^ 32 by omega)]

theorem add00 : Verity.Fp32.f32AddV2 0 0 = 0 := by decide +kernel

/-! ### The running max of `|x|` -/

theorem rowMax_succ (V : ℕ → ℕ) (t : ℕ) (ht : rowMax coreOps .fabs V t < 2 ^ 31) :
    rowMax coreOps .fabs V (t + 1) = max (rowMax coreOps .fabs V t) (V (t + 1) % 2 ^ 31) := by
  simp only [rowMax, Wire.eval, CoreOps.coreOps_fabs]
  generalize rowMax coreOps .fabs V t = r at ht ⊢
  have ha : V (t + 1) % 2 ^ 31 < 2 ^ 31 := Nat.mod_lt _ (by norm_num)
  generalize V (t + 1) % 2 ^ 31 = a at ha ⊢
  rw [eval_sel, CoreOps.coreOps_sel, eval_le, Nat.mod_mod, coreOps_le r a ht ha, Nat.max_def]
  by_cases h : r ≤ a <;> simp [h] <;> omega

theorem rowMax_lt31 (V : ℕ → ℕ) : ∀ t, rowMax coreOps .fabs V t < 2 ^ 31
  | 0 => by simp only [rowMax, Wire.eval, CoreOps.coreOps_fabs]; exact Nat.mod_lt _ (by norm_num)
  | t + 1 => by
    rw [rowMax_succ V t (rowMax_lt31 V t), Nat.max_def]
    have := rowMax_lt31 V t
    have : V (t + 1) % 2 ^ 31 < 2 ^ 31 := Nat.mod_lt _ (by norm_num)
    split <;> omega

/-- The check's fold of `max(m, w & 0x7FFFFFFF)` is the row's running max of `|x|`. -/
theorem max_fold (x : Array ℕ) (hx : 0 < x.size) :
    x.foldl (fun m w => max m (w &&& 0x7FFFFFFF)) 0 = rowMax coreOps .fabs (fun l => x[l]!) (x.size - 1) := by
  have := Array.foldl_induction (as := x)
    (fun i b => (i = 0 ∧ b = 0) ∨ (0 < i ∧ b = rowMax coreOps .fabs (fun l => x[l]!) (i - 1)))
    (Or.inl ⟨rfl, rfl⟩) (f := fun m w => max m (w &&& 0x7FFFFFFF)) (fun ⟨i, hi⟩ b hb => by
      have hm : x[i] &&& 0x7FFFFFFF = x[i]! % 2 ^ 31 := by
        rw [getElem!_pos x i hi]; exact Nat.and_two_pow_sub_one_eq_mod _ 31
      dsimp only at hb ⊢
      simp only [Fin.getElem_fin]
      rw [hm]
      right
      refine ⟨by omega, ?_⟩
      rcases hb with ⟨rfl, rfl⟩ | ⟨h0, rfl⟩
      · simp only [rowMax, Wire.eval, CoreOps.coreOps_fabs, Nat.zero_max, Nat.zero_add, Nat.sub_self]
      · obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
        simp only [Nat.add_sub_cancel]
        rw [rowMax_succ _ _ (rowMax_lt31 _ _)])
  rcases this with ⟨h0, -⟩ | ⟨-, h⟩
  · omega
  · exact h

/-! ### The check's loops -/

/-- A `for` loop over a list that returns `.ok` keeps an invariant its body keeps. -/
theorem forIn_inv {α γ : Type} (P : ℕ → γ → Prop) {body : α → γ → Except String (ForInStep γ)}
    (l : List α) : ∀ (i : ℕ) {x y : γ}, P i x →
      (∀ j (hj : j < l.length) z r, P (i + j) z → body l[j] z = .ok r →
        ∃ z', r = .yield z' ∧ P (i + j + 1) z') →
      forIn l x body = .ok y → P (i + l.length) y := by
  induction l with
  | nil =>
    intro i x y hx _ h
    rw [List.forIn_nil] at h
    obtain rfl := pure_ok h
    exact hx
  | cons a l ih =>
    intro i x y hx hstep h
    rw [List.forIn_cons] at h
    obtain ⟨r, hr, h⟩ := bind_ok h
    obtain ⟨z', rfl, hz⟩ := hstep 0 (by simp) x r hx hr
    have := ih (i + 1) hz (fun j hj z r hz hr => by
      have := hstep (j + 1) (by rw [List.length_cons]; omega) z r
        (by rwa [show i + (j + 1) = i + 1 + j by omega]) hr
      rwa [show i + (j + 1) + 1 = i + 1 + j + 1 by omega] at this) h
    rwa [List.length_cons, show i + (l.length + 1) = i + 1 + l.length by omega]

/-- After the FMA loop's first `j` partials, accumulator `i` holds partial `accLast j i`, or `+0` before its first;
every accumulator is sign-clear. -/
def FmaInv (V : ℕ → ℕ) (j : ℕ) (p : Array ℕ) : Prop :=
  p.size = 32 ∧ ∀ i < 32, p[i]! = (if i < j then fmaAcc coreOps .id V (accLast j i) else 0) ∧ p[i]! < 2 ^ 31

theorem fmaInv_zero (V : ℕ → ℕ) : FmaInv V 0 (Array.replicate 32 0) := by
  refine ⟨Array.size_replicate, fun i hi => ?_⟩
  rw [getElem!_pos _ i (by rw [Array.size_replicate]; exact hi), Array.getElem_replicate]
  exact ⟨by simp, by norm_num⟩

theorem fma_step {x : Array ℕ} {j : ℕ} {p : Array ℕ} {v : ℕ} (hp : FmaInv (fun l => x[l]!) j p)
    (hv : Words.finite (Verity.Fp32.f32FmaV2 x[8 * j]! x[8 * j]! p[j % 32]!) = .ok v) :
    FmaInv (fun l => x[l]!) (j + 1) (p.set! (j % 32) v) := by
  obtain ⟨hs, hp⟩ := hp
  obtain ⟨rfl, -⟩ := finite_ok hv
  have hj32 : j % 32 < 32 := Nat.mod_lt _ (by norm_num)
  refine ⟨by rw [Array.size_set!, hs], fun i hi => ?_⟩
  by_cases hij : i = j % 32
  · subst hij
    rw [Array.getElem!_set!_self _ _ _ (by omega)]
    obtain ⟨hpe, hplt⟩ := hp (j % 32) hj32
    refine ⟨?_, fmaSq_lt31 _ hplt⟩
    have hc : (if j < 32 then 0 else fmaAcc coreOps .id (fun l => x[l]!) (j - 32)) = p[j % 32]! := by
      rw [hpe]
      by_cases h32 : j < 32
      · rw [ite_eq_left h32, ite_eq_right (show ¬ j % 32 < j by omega)]
      · rw [ite_eq_right h32, ite_eq_left (show j % 32 < j by omega),
          show accLast j (j % 32) = j - 32 by unfold accLast; omega]
    rw [ite_eq_left (show j % 32 < j + 1 by omega), show accLast (j + 1) (j % 32) = j by unfold accLast; omega,
      fmaAcc.eq_1]
    simp only [Wire.eval]
    rw [hc, eval_fma, Nat.mod_eq_of_lt (show Verity.Fp32.f32FmaV2 x[8 * j]! x[8 * j]! p[j % 32]! < 2 ^ 32 by
      have := fmaSq_lt31 x[8 * j]! hplt; omega)]
  · rw [Array.getElem!_set!_ne _ _ _ _ (Ne.symm hij)]
    obtain ⟨hpe, hplt⟩ := hp i hi
    refine ⟨?_, hplt⟩
    rw [hpe]
    by_cases hlt : i < j
    · rw [ite_eq_left hlt, ite_eq_left (show i < j + 1 by omega),
        show accLast (j + 1) i = accLast j i by unfold accLast; omega]
    · rw [ite_eq_right hlt, ite_eq_right (show ¬ i < j + 1 by omega)]

theorem fma_loop {x : Array ℕ} {n : ℕ} {body : ℕ → Array ℕ → Except String (ForInStep (Array ℕ))}
    {p : Array ℕ} (hbody : ∀ i r y, body i r = .ok y →
      ∃ v, Words.finite (Verity.Fp32.f32FmaV2 x[8 * i]! x[8 * i]! r[i % 32]!) = .ok v ∧
        y = .yield (r.set! (i % 32) v))
    (h : forIn [:n] (Array.replicate 32 0) body = .ok p) : FmaInv (fun l => x[l]!) n p := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have := forIn_inv (FmaInv (fun l => x[l]!)) _ 0 (fmaInv_zero _) (fun j hj z r hz hr => by
    rw [List.getElem_range'] at hr
    obtain ⟨v, hv, rfl⟩ := hbody _ _ _ hr
    refine ⟨_, rfl, ?_⟩
    simp only [Nat.zero_add, Nat.one_mul] at hz hv ⊢
    exact fma_step hz hv) h
  simpa [Std.Legacy.Range.size] using this

/-- The butterfly's sums after `j` rounds, round `j` adding sum `i + 2^(4-j)` into sum `i`. -/
def lvl (n : ℕ) (F : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0 => F
  | j + 1 => bfly coreOps (2 ^ (4 - j)) n (lvl n F j)

/-- After `j` butterfly rounds, the first `2^(5-j)` sums are `lvl`'s (`+0` past the live ones), and every sum is
sign-clear. -/
def BflyInv (n : ℕ) (F : ℕ → ℕ) (j : ℕ) (p : Array ℕ) : Prop :=
  p.size = 32 ∧ (∀ i < 32, p[i]! < 2 ^ 31) ∧ ∀ i < 2 ^ (5 - j), p[i]! = if i < n then lvl n F j i else 0

theorem bfly_step {n : ℕ} {F : ℕ → ℕ} {j o : ℕ} (hj : j < 5) (ho : o = 2 ^ (4 - j)) {p q : Array ℕ}
    (hp : BflyInv n F j p)
    (hq : (Array.range 32).mapM (fun i => Words.finite (Verity.Fp32.f32AddV2 p[i]! p[i ^^^ o]!)) = .ok q) :
    BflyInv n F (j + 1) q := by
  obtain ⟨-, hlt, hv⟩ := hp
  obtain ⟨hqs, hqv⟩ := mapM_ok hq
  rw [Array.size_range] at hqs
  have hxor : ∀ i < 32, i ^^^ o < 32 := fun i hi => by
    subst ho; interval_cases j <;> exact Nat.xor_lt_two_pow (n := 5) hi (by norm_num)
  have hadd : ∀ i < o, i ^^^ o = i + o := by subst ho; interval_cases j <;> decide
  have ho2 : 2 ^ (5 - j) = 2 * o := by subst ho; interval_cases j <;> rfl
  have ho1 : 2 ^ (5 - (j + 1)) = o := by subst ho; interval_cases j <;> rfl
  have ho16 : o ≤ 16 := by subst ho; interval_cases j <;> norm_num
  have hf : ∀ i < 32, Words.finite (Verity.Fp32.f32AddV2 p[i]! p[i ^^^ o]!) = .ok q[i]! := fun i hi => by
    have := hqv i (by rw [Array.size_range]; exact hi)
    rwa [getElem!_pos (Array.range 32) i (by rw [Array.size_range]; exact hi), Array.getElem_range] at this
  refine ⟨hqs, fun i hi => (add_eval (hlt i hi) (hlt _ (hxor i hi)) (hf i hi)).2, fun i hi => ?_⟩
  rw [ho1] at hi
  have hi32 : i < 32 := by omega
  have h := hf i hi32
  have ha := hlt i hi32
  have hb := hlt (i + o) (by omega)
  rw [hadd i hi] at h
  rw [hv i (by omega)] at h ha
  rw [hv (i + o) (by omega)] at h hb
  rw [show (if i < n then lvl n F (j + 1) i else 0) = if i < n then bfly coreOps o n (lvl n F j) i else 0 by
    subst ho; rfl]
  by_cases hin : i < n
  · simp only [hin, ↓reduceIte] at h ha ⊢
    rw [(add_eval ha hb h).1]
    rfl
  · simp only [hin, ↓reduceIte, show ¬ i + o < n by omega] at h ⊢
    rw [(finite_ok h).1, add00]

theorem bfly_loop {n : ℕ} {F : ℕ → ℕ} {body : ℕ → Array ℕ → Except String (ForInStep (Array ℕ))}
    {p q : Array ℕ} (hbody : ∀ o r y, body o r = .ok y →
      ∃ r', (Array.range 32).mapM (fun j => Words.finite (Verity.Fp32.f32AddV2 r[j]! r[j ^^^ o]!)) = .ok r' ∧
        y = .yield r')
    (h0 : BflyInv n F 0 p) (h : forIn [16, 8, 4, 2, 1] p body = .ok q) : BflyInv n F 5 q := by
  have := forIn_inv (BflyInv n F) [16, 8, 4, 2, 1] 0 h0 (fun j hj z r hz hr => by
    obtain ⟨r', hm, rfl⟩ := hbody _ _ _ hr
    refine ⟨r', rfl, ?_⟩
    rw [Nat.zero_add] at hz ⊢
    have hj5 : j < 5 := hj
    exact bfly_step (o := [16, 8, 4, 2, 1][j]) hj5 (by interval_cases j <;> rfl) hz hm) h
  exact this

/-! ### The statistics on the raw words -/

theorem stats_V {x : Array ℕ} {s ρw αw βw : ℕ} (hk0 : 0 < x.size) (hk8 : 8 ∣ x.size) (hkK : x.size ≤ 2 ^ 27)
    (h : Words.s5Stats x = .ok (s, ρw, αw, βw)) :
    ρw = rowRho coreOps .id (fun l => x[l]!) x.size ∧ αw = alpha8 coreOps (fun l => x[l]!) x.size ∧
      βw = beta8 coreOps (fun l => x[l]!) x.size := by
  have hfin := statsRhoIsFinite x s ρw αw βw h
  unfold Words.s5Stats at h
  obtain ⟨p1, hp1, h⟩ := bind_ok h
  obtain ⟨p2, hp2, h⟩ := bind_ok h
  obtain ⟨t, ht, h⟩ := bind_ok h
  obtain ⟨b, hb, h⟩ := bind_ok h
  have h := pure_ok h
  simp only [Prod.mk.injEq] at h
  obtain ⟨hs, hρ, hα, hβ⟩ := h
  have eMul : Evaluate.Fp32.f32MulV2 = Verity.Fp32.f32MulV2 := Properties.Evaluator.Fp32.f32MulV2
  have eDiv : Evaluate.Fp32.f32DivV2 = Verity.Fp32.f32DivV2 := Properties.Evaluator.Fp32.f32DivV2
  have eAdd : Evaluate.Fp32.f32AddV2 = Verity.Fp32.f32AddV2 := Properties.Evaluator.Fp32.f32AddV2
  have eFma : Evaluate.Fp32.f32FmaV2 = Verity.Fp32.f32FmaV2 := Properties.Evaluator.Fp32.f32FmaV2
  rw [eAdd, eMul, eDiv, show Words.S5_FOUR = 0x40800000 from rfl] at ht
  rw [eDiv] at hρ
  rw [eDiv, show Words.S5_QMAX = 0x43E00000 from rfl, show Words.S5_FLOOR = 0x2F800000 from rfl] at hα
  rw [eMul, eDiv, show Words.S5_QMAX = 0x43E00000 from rfl, show Words.S5_FLOOR = 0x2F800000 from rfl,
    show Words.S5_LINE = 0x3AB504F3 from rfl] at hb
  rw [hρ, hs] at ht
  rw [hρ, hα] at hb
  have hn : (x.size + 7) / 8 = x.size / 8 := by omega
  have hn0 : 0 < x.size / 8 := by omega
  have hn24 : x.size / 8 ≤ 2 ^ 24 := by omega
  rw [hn] at hρ
  have hF : FmaInv (fun l => x[l]!) ((x.size + 7) / 8) p1 := fma_loop (fun i r y hy => by
      dsimp only at hy
      obtain ⟨v, hv, hy⟩ := bind_ok hy
      exact ⟨v, by rwa [eFma] at hv, (pure_ok hy).symm⟩) hp1
  rw [hn] at hF
  have hB : BflyInv (x.size / 8) (fun j => fmaAcc coreOps .id (fun l => x[l]!) (accLast (x.size / 8) j)) 5 p2 :=
    bfly_loop (fun o r y hy => by
      dsimp only at hy
      obtain ⟨r', hm, hy⟩ := bind_ok hy
      exact ⟨r', by rwa [eAdd] at hm, (pure_ok hy).symm⟩)
      ⟨hF.1, fun i hi => (hF.2 i hi).2, fun i hi => (hF.2 i hi).1⟩ hp2
  have hp0 : p2[0]! = sumSq coreOps .id (fun l => x[l]!) (x.size / 8) := by
    rw [hB.2.2 0 (by norm_num), ite_eq_left hn0]; rfl
  have hp031 : p2[0]! < 2 ^ 31 := hB.2.1 0 (by norm_num)
  have hofs := ofInt_eq hn0 hn24
  have hq31 : Verity.Fp32.f32DivV2 p2[0]! (Words.ofInt ↑(x.size / 8)) < 2 ^ 31 :=
    divV2_lt31 hp031 (ofInt_lt31 hn0 hn24)
  have hρw : ρw = sqrtWord (Verity.Fp32.f32DivV2 p2[0]! (Words.ofInt ↑(x.size / 8))) := by
    rw [← hρ]; exact sqrtIsTheSpec _ (by rw [hρ]; exact hfin)
  have hρ31 : ρw < 2 ^ 31 := by rw [hρw]; exact sqrtWord_lt31 hq31
  have hρ' : ρw = rowRho coreOps .id (fun l => x[l]!) x.size := by
    have hsq := sqrtWord_lt31 hq31
    unfold rowRho
    rw [eval_div, eval_sqrt, ← hp0, ← hofs,
      Nat.mod_eq_of_lt (show Verity.Fp32.f32DivV2 p2[0]! (Words.ofInt ↑(x.size / 8)) < 2 ^ 32 by omega),
      Nat.mod_eq_of_lt (show sqrtWord (Verity.Fp32.f32DivV2 p2[0]! (Words.ofInt ↑(x.size / 8))) < 2 ^ 32 by omega),
      hρw]
  have hs' : s = rowMax coreOps .fabs (fun l => x[l]!) (x.size - 1) := by rw [← hs, max_fold x hk0]
  have hs31 : s < 2 ^ 31 := by rw [hs']; exact rowMax_lt31 _ _
  have hm31 : Verity.Fp32.f32MulV2 0x40800000 ρw < 2 ^ 31 := mulV2_lt31 (by norm_num) hρ31
  obtain ⟨htv, ht31⟩ := add_eval hs31 hm31 ht
  have htw : Prim.eval coreOps .add [rowMax coreOps .fabs (fun l => x[l]!) (x.size - 1),
      Prim.eval coreOps .mul [0x40800000, rowRho coreOps .id (fun l => x[l]!) x.size]] = t := by
    rw [← hρ', ← hs', eval_mul, Nat.mod_eq_of_lt (show Verity.Fp32.f32MulV2 0x40800000 ρw < 2 ^ 32 by omega), htv]
  have hFl := NoiseProofs.wordQ_word (sb := 0) (e := 95) (mm := 0) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
  rw [show 0 * 2 ^ 31 + 95 * 2 ^ 23 + 0 = 0x2F800000 by norm_num] at hFl
  obtain ⟨xt, hxt⟩ := Option.isSome_iff_exists.1 (Rho.wordQ_of_finite ht)
  rw [← (finite_ok ht).1] at hxt
  have hmax : coreOps.f32Fmaxf t 0x2F800000 % 2 ^ 32 = max t 0x2F800000 := by
    rw [CoreOps.coreOps_fmaxfAny t _ _ (by omega) (by norm_num) hFl (by norm_num)]
    have hne : t ≠ 0x7F800000 := by
      rintro rfl
      rw [show wordQ 0x7F800000 = none by decide +kernel] at hxt
      cases hxt
    have hc := CoreOps.wordQ_lt_iff (show 0x2F800000 < 2 ^ 31 by norm_num) ht31 hFl hxt
    rw [Nat.max_def]
    by_cases hlt : 0x2F800000 < t
    · rw [ite_eq_left (Or.inr ⟨xt, hxt, hc.2 hlt⟩), ite_eq_right (show ¬ t ≤ 0x2F800000 by omega)]
    · rw [ite_eq_left (show t ≤ 0x2F800000 by omega), ite_eq_right]
      rintro (h | ⟨x', hx', hlt'⟩)
      · exact hne h
      · rw [hxt] at hx'
        cases hx'
        exact hlt (hc.1 hlt')
  have hmx31 : max t 0x2F800000 < 2 ^ 31 := by rw [Nat.max_def]; split <;> omega
  have hα31 : αw < 2 ^ 31 := by rw [← hα]; exact divV2_lt31 (by norm_num) hmx31
  have hα' : αw = alpha8 coreOps (fun l => x[l]!) x.size := by
    unfold alpha8
    rw [htw, eval_fmaxf, hmax, eval_div,
      Nat.mod_eq_of_lt (show Verity.Fp32.f32DivV2 0x43E00000 (max t 0x2F800000) < 2 ^ 32 by rw [hα]; omega), hα]
  have hβ' : βw = beta8 coreOps (fun l => x[l]!) x.size := by
    have hra := mulV2_lt31 hρ31 hα31
    have hrl := mulV2_lt31 hra (show 0x3AB504F3 < 2 ^ 31 by norm_num)
    unfold beta8
    rw [← hρ', ← hα', eval_mul, eval_mul, Nat.mod_eq_of_lt (show Verity.Fp32.f32MulV2 ρw αw < 2 ^ 32 by omega),
      Nat.mod_eq_of_lt (show Verity.Fp32.f32MulV2 (Verity.Fp32.f32MulV2 ρw αw) 0x3AB504F3 < 2 ^ 32 by omega), ← hβ,
      (finite_ok hb).1]
  exact ⟨hρ', hα', hβ'⟩

end StatsSim

open StatsSim RowCircuit in
/-- **`s5_stats` is the row's statistics**: on a row of `k` finite words, `8 ∣ k`, `k ≤ 2^27`, where the check's
statistics are a value, its `ρ`, `α` and `β` are `pc8`'s formulas on the row's values re-encoded. -/
theorem statsAreTheRow : StatsAreTheRow := by
  intro k x s ρw αw βw hk hk0 hk8 hkK hx h
  subst hk
  obtain ⟨h1, h2, h3⟩ := stats_V hk0 hk8 hkK h
  have hK : 8 ≤ x.size := Nat.le_of_dvd hk0 hk8
  have hw : ∀ l < x.size, (wordQ ((fun l => x[l]!) l)).isSome := fun l hl => (hx l hl).2
  have hZ : ∀ l < x.size, WholeCircuit.ZeroSign ((fun l => x[l]!) l) (encRow x.size (rowVals x.size x) l) := by
    intro l hl
    have := WholeCircuit.encRow_zeroSign hw l hl
    rw [Nat.mod_eq_of_lt (hx l hl).1] at this
    exact this
  refine ⟨?_, ?_, ?_⟩
  · rw [h1]
    exact WholeCircuit.rowRho_congrW hK fun l hl c hc =>
      WholeCircuit.fmaSq_zeroSign CoreOps.pc8OpsAgree_coreOps (hZ l hl) hc
  · rw [h2]; exact WholeCircuit.alpha8_zeroSign CoreOps.pc8OpsAgree_coreOps hK hZ
  · rw [h3]; exact WholeCircuit.beta8_zeroSign CoreOps.pc8OpsAgree_coreOps hK hZ

end Pouw.PearlPlus.Bridge
