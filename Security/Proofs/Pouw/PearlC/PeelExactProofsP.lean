import Proofs.Pouw.PearlC.FormingP
import Proofs.Pouw.PearlC.PeelExactProofs
import Proofs.Pouw.PearlC.AccumProofs

/-!
# P2(b) and (c) on `PeelExactP`'s U, at any step width (sm_120 at width 26)

`PeelExactProofs` proves P2(b), the served U within `peelBound` of its exact counterpart, at Π's E4M3 step
(`hopperE4m3K32`, width 14) in windows of `G = 4`. Here the same bound is proved for `PeelExactP.peelExactP p G`, on any
step `p` that has a relative error on a range, and for every `G`, with `G = 0` the pure chain (`AccumProofs`).

* **The step at width `w`** (`pipeStep_rel`): the step of `⟨[32], w, z⟩`, for `2 ≤ w` and a floor `z < w − 127`, on
  `HRangeW w z` (`PeelExactProofs.H8Range` with `w − 127` for `−113`) writes a finite word within
  `epsW w·(|c| + Σ_l |a_l·b_l|)` of `c + Σ_l a_l·b_l`, where
  `epsW w = 33·2^−(w−1) + (1 + 33·2^−(w−1))·2^−(min w 24 − 1)`. It is built on `FormingP.pipeStep_value` and
  `FormingP.pipeStep_err`. `epsW 14 = eps8`; `epsW 26` is `33·2^−25 + (1 + 33·2^−25)·2^−23`, about `2^−19.8`.
* **A chain on any step** (`StepRel p R ε`: the step's relative error `ε` on the range `R`). A window's word is within
  `((1 + ε)^n − 1)·Σ|a·b|` of its `n` atoms' sum (`windowWordP_err`); so is the pure chain (`pureWordP_err`,
  `pureBound`). In windows of `G ≥ 1` the chain is within `chainBoundP ε G`, `PeelExactProofs.chainBound`'s form with
  `epsCP ε G = (1 + 2^−24)·(1 + ε)^G − 1` (`chainWordP_err`). `chainBoundAt ε G` and `ChainRangeAt p R G` take the
  pure form at `G = 0` (`chainWordAt_err`).
* **U** (`peelExactValP_err`): `peelExactValP p G` within `peelBoundP ε G`, `peelBound` with `chainBoundAt ε G` for
  `chainBound`, on `PeelRangeP p R G`.
* **sm_120** (`sm120E4m3`, width 26, floor −133): `peelExactValP_sm120_err` for every `G`, with `G = 4`
  (`peelExactValP_sm120_four`) and `G = 0` (`peelExactValP_sm120_zero`, `peelExactPureValP_sm120_err`).
* **(c) at sm_120** (`served120_salt_dependent_four`, `_zero`): `PeelExactProofs.served_salt_dependent`'s witness
  formed by sm_120's atom-mix (`FormingP.atomMixP sm120E4m3`) and served by `peelExactValP sm120E4m3 G`; the two salts
  give `0` and `10` at `G = 4` and at `G = 0`, by kernel evaluation.
-/

namespace Pouw.PearlC.PeelExactP

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Matrix

variable {m n k : Nat}

theorem windowWordP?_hopper : windowWordP? hopperE4m3K32 = windowWord? := rfl

/-- Π's chain on words (windows of `Params.pi.G = 4`) is the chain twin at `hopperE4m3K32`. -/
theorem chainWordP?_hopper : chainWordP? hopperE4m3K32 4 = chainWord? := rfl

/-- Π's U is U's twin at `hopperE4m3K32` and `G = 4`. -/
theorem peelExactP_hopper : @peelExactP m n k hopperE4m3K32 4 = @peelExact m n k := rfl

/-- Π's U as values is U's twin's at `hopperE4m3K32` and `G = 4`. -/
theorem peelExactValP_hopper : @peelExactValP m n k hopperE4m3K32 4 = @peelExactVal m n k := rfl

end Pouw.PearlC.PeelExactP

namespace Pouw.PearlC.PeelExactProofsP

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.PeelExact
  Pouw.PearlC.PeelExactP Pouw.PearlC.PeelExactProofs Matrix

variable {m n k : Nat}

/-! ## The step at width `w` -/

/-- **The step's range at width `w` and floor `z`** (`H8Range` at `w = 14`, `z = −139`): every code is zero or normal,
and the step's exponent is at most 120, and at least `w − 127` unless every term is zero. -/
def HRangeW (w : Nat) (z : ℤ) (tc : Term) (a b : Fin 32 → Code) : Prop :=
  (∀ l, ZeroOrNormal8 (a l)) ∧ (∀ l, ZeroOrNormal8 (b l)) ∧ FormingP.stepExpP z tc a b ≤ 120 ∧
    ((w : ℤ) - 127 ≤ FormingP.stepExpP z tc a b ∨ (tc.mag = 0 ∧ ∀ l, (product (a l) (b l)).mag = 0))

/-- The step's relative error at width `w` on its range: `33·2^−(w−1) + (1 + 33·2^−(w−1))·2^−(min w 24 − 1)`. -/
def epsW (w : Nat) : ℚ := 33 / 2 ^ (w - 1) + (1 + 33 / 2 ^ (w - 1)) / 2 ^ (min w 24 - 1)

theorem epsW_nonneg (w : Nat) : 0 ≤ epsW w := by unfold epsW; positivity

theorem epsW_14 : epsW 14 = eps8 := by unfold epsW eps8; norm_num

/-- On the range, the step's grid is bounded by its terms: `2^E ≤ |c| + Σ_l |a_l·b_l|`. -/
theorem two_zpow_stepExpW_le {w : Nat} {z : ℤ} (hw2 : 2 ≤ w) (hz : z < (w : ℤ) - 127) {acc : Nat} {tc : Term}
    (hw : fp32Value acc = .fin tc) {a b : Fin 32 → Code} (ha : ∀ l, ZeroOrNormal8 (a l))
    (hb : ∀ l, ZeroOrNormal8 (b l)) (hE1 : (w : ℤ) - 127 ≤ FormingP.stepExpP z tc a b) :
    (2 : ℚ) ^ FormingP.stepExpP z tc a b ≤ |tc.value| + ∑ l, |val (a l) * val (b l)| := by
  have hB : 0 ≤ ∑ l, |val (a l) * val (b l)| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hc := abs_nonneg tc.value
  unfold FormingP.stepExpP at hE1 ⊢
  rcases AtomErr.maxExp_cases z (tc :: List.ofFn fun l => product (a l) (b l)) with e | ⟨t, ht, hm, e⟩
  · rw [e] at hE1; omega
  · rw [e] at hE1 ⊢
    rcases List.mem_cons.1 ht with rfl | ht
    · rcases AtomErr.fp32Value_cases hw with h0 | he | h23
      · exact absurd h0 hm
      · rw [he] at hE1; omega
      · have := two_zpow_term_le h23; linarith
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      have h1 := product_normal8 (ha l) (hb l) hm
      have h2 : |val (a l) * val (b l)| ≤ ∑ l, |val (a l) * val (b l)| :=
        Finset.single_le_sum (f := fun l => |val (a l) * val (b l)|) (fun _ _ => abs_nonneg _)
          (Finset.mem_univ l)
      linarith

/-- The step on zero terms writes `+0`, at any width and floor. -/
theorem pipeStep_zero {w : Nat} {z : ℤ} {acc : Nat} {tc : Term} (hw : fp32Value acc = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) (h0 : tc.mag = 0)
    (hp : ∀ l, (product (a l) (b l)).mag = 0) :
    wordQ ((⟨[32], w, z⟩ : Pipeline).step acc (List.ofFn a) (List.ofFn b)) = some 0 := by
  rw [FormingP.pipeStep_fin hw a b ha hb]
  have hs : ∀ E, alignedSum w E (tc :: List.ofFn fun l => product (a l) (b l)) = 0 := by
    intro E
    unfold alignedSum
    apply List.sum_eq_zero
    intro x hx
    simp only [List.mem_map, List.mem_cons, List.mem_ofFn] at hx
    obtain ⟨t, ht, rfl⟩ := hx
    rcases ht with rfl | ⟨l, rfl⟩
    · simp [aligned, h0]
    · simp [aligned, hp l]
  have hn : ∀ E, normalize w E 0 = Term.zero := fun E => by unfold normalize; rw [ite_eq_left rfl]
  unfold groupSum
  rw [hs, hn]
  show wordQ (packValue (if Term.zero.mag ≠ 0 ∧ 127 < Term.zero.exp then .inf Term.zero.neg else .fin Term.zero)) = _
  rw [ite_eq_right (fun h => absurd rfl h.1)]
  exact Freeze.wordQ_zero

theorem rel_of_errW {w : Nat} (hw1 : 1 ≤ w) {v S B : ℚ} {E : ℤ}
    (he : |v - S| ≤ (((32 : ℕ) : ℚ) + 1) * 2 ^ (E - (w : ℤ) + 1) +
      (|S| + (((32 : ℕ) : ℚ) + 1) * 2 ^ (E - (w : ℤ) + 1)) / 2 ^ (min w 24 - 1))
    (hS : |S| ≤ B) (hB : (2 : ℚ) ^ E ≤ B) : |v - S| ≤ epsW w * B := by
  have e : (2 : ℚ) ^ (E - (w : ℤ) + 1) = 2 ^ E / 2 ^ (w - 1) := by
    rw [← zpow_natCast (2 : ℚ) (w - 1), ← zpow_sub₀ two_ne_zero, Nat.cast_sub hw1]
    congr 1
    push_cast
    ring
  have hg : (2 : ℚ) ^ (E - (w : ℤ) + 1) ≤ B / 2 ^ (w - 1) := by
    rw [e]; exact div_le_div_of_nonneg_right hB (by positivity)
  have e33 : ((32 : ℕ) : ℚ) + 1 = 33 := by norm_num
  rw [e33] at he
  calc _ ≤ _ := he
    _ ≤ 33 * (B / 2 ^ (w - 1)) + (B + 33 * (B / 2 ^ (w - 1))) / 2 ^ (min w 24 - 1) := by gcongr
    _ = epsW w * B := by unfold epsW; ring

/-- **The step's relative error at width `w` on the range** (`hopperStep_rel` at `w = 14`): the word is finite, and
within `epsW w·(|c| + Σ_l |a_l·b_l|)` of `c + Σ_l a_l·b_l`. -/
theorem pipeStep_rel {w : Nat} {z : ℤ} (hw2 : 2 ≤ w) (hz : z < (w : ℤ) - 127) {acc : Nat} {tc : Term}
    (hw : fp32Value acc = .fin tc) {a b : Fin 32 → Code} (hR : HRangeW w z tc a b) :
    ∃ v, wordQ ((⟨[32], w, z⟩ : Pipeline).step acc (List.ofFn a) (List.ofFn b)) = some v ∧
      |v - (tc.value + ∑ l, val (a l) * val (b l))| ≤ epsW w * (|tc.value| + ∑ l, |val (a l) * val (b l)|) := by
  obtain ⟨ha, hb, hE2, hE1 | ⟨h0, hp⟩⟩ := hR
  · have hv := FormingP.pipeStep_value (n := 32) (by norm_num) (by omega) hw a b (fun l => (ha l).1)
      (fun l => (hb l).1) hE1 hE2
    refine ⟨_, hv, ?_⟩
    have he := FormingP.pipeStep_err (n := 32) (by norm_num) (by omega) hw a b (fun l => (ha l).1)
      (fun l => (hb l).1) hE1 hE2
    rw [wordVal_of hv] at he
    have hS : |tc.value + ∑ l, val (a l) * val (b l)| ≤ |tc.value| + ∑ l, |val (a l) * val (b l)| := by
      have h1 := abs_add_le tc.value (∑ l, val (a l) * val (b l))
      have h2 := Finset.abs_sum_le_sum_abs (fun l => val (a l) * val (b l)) Finset.univ
      linarith
    exact rel_of_errW (by omega) he hS (two_zpow_stepExpW_le hw2 hz hw ha hb hE1)
  · refine ⟨0, pipeStep_zero hw a b (fun l => (ha l).1) (fun l => (hb l).1) h0 hp, ?_⟩
    have hc : tc.value = 0 := by simp [Term.value, h0]
    have hp' : ∀ l, val (a l) * val (b l) = 0 := by
      intro l
      rw [← productExact _ _ _ _ (val_eq (ha l).1) (val_eq (hb l).1)]
      simp [Term.value, hp l]
    simp only [hc, hp', Finset.sum_const_zero, add_zero, sub_zero, abs_zero, mul_zero, le_refl]

/-! ## A chain on any step with a relative error -/

/-- **A step's relative error `ε` on a range `R`** of accumulator terms and atoms: on a finite accumulator term `tc`
with `R tc x`, the step writes a finite word within `ε·(|c| + Σ_l |a_l·b_l|)` of `c + Σ_l a_l·b_l`. -/
def StepRel (P : Pipeline) (R : Term → AtomIn → Prop) (ε : ℚ) : Prop :=
  ∀ (acc : Nat) (tc : Term) (x : AtomIn), fp32Value acc = .fin tc → R tc x →
    ∃ v, wordQ (P.step acc (List.ofFn x.1) (List.ofFn x.2)) = some v ∧
      |v - (tc.value + atomQ x)| ≤ ε * (|tc.value| + atomAbs x)

/-- The step `⟨[32], w, z⟩` has relative error `epsW w` on `HRangeW w z`. -/
theorem stepRel_pipe {w : Nat} {z : ℤ} (hw2 : 2 ≤ w) (hz : z < (w : ℤ) - 127) :
    StepRel ⟨[32], w, z⟩ (fun tc x => HRangeW w z tc x.1 x.2) (epsW w) :=
  fun _ _ _ hw hR => pipeStep_rel hw2 hz hw hR

/-- Π's step has relative error `eps8` on `H8Range` (`hopperStep_rel`). -/
theorem hopper_stepRel : StepRel hopperE4m3K32 (fun tc x => H8Range tc x.1 x.2) eps8 :=
  fun _ _ _ hw hR => hopperStep_rel hw hR

/-- One window's step on `P`, the word finite. -/
def winStepP (P : Pipeline) (w : Nat) (x : AtomIn) : Option Nat := finite? (P.step w (List.ofFn x.1) (List.ofFn x.2))

/-- The range along a window's steps from `+0`. -/
def WindowRangeP (P : Pipeline) (R : Term → AtomIn → Prop) (g : List AtomIn) : Prop :=
  FoldRange (winStepP P) (fun c x => ∀ tc, fp32Value c = .fin tc → R tc x) g 0

variable {P : Pipeline} {R : Term → AtomIn → Prop} {ε : ℚ}

/-- **A window's error** (`windowWord_err` on any step): its word is within `((1 + ε)^n − 1)·Σ |a·b|` of its `n`
atoms' exact sum. -/
theorem windowWordP_err (hS : StepRel P R ε) (hε : 0 ≤ ε) {g : List AtomIn} {w : Nat}
    (h : windowWordP? P g = some w) (hR : WindowRangeP P R g) :
    ∃ v, wordQ w = some v ∧ |v - (g.map atomQ).sum| ≤ ((1 + ε) ^ g.length - 1) * (g.map atomAbs).sum := by
  have key := fold_rel (winStepP P) (fun c x => ∀ tc, fp32Value c = .fin tc → R tc x) atomQ atomAbs
    hε (fun x => Finset.abs_sum_le_sum_abs _ _)
    (fun c x c' cv hR hc hs => by
      obtain ⟨tc, htc, htv⟩ := wordQ_fin hc
      obtain ⟨rfl, -⟩ := finite?_eq_some hs
      obtain ⟨v, hv, he⟩ := hS c tc x htc (hR tc htc)
      exact ⟨v, hv, by rw [← htv]; exact he⟩)
  obtain ⟨v, hv, he⟩ := key g 0 w 0 Freeze.wordQ_zero h hR
  refine ⟨v, hv, ?_⟩
  rw [zero_add, abs_zero, zero_add] at he
  exact he

/-- **The pure chain's bound** over `n` atoms: `((1 + ε)^n − 1)·Σ |a·b|`; every step's error compounds. -/
def pureBound (ε : ℚ) (atoms : List AtomIn) : ℚ := ((1 + ε) ^ atoms.length - 1) * (atoms.map atomAbs).sum

/-- **The pure chain's error**: `pureWordP? P`'s word is within `pureBound ε` of the exact chain. -/
theorem pureWordP_err (hS : StepRel P R ε) (hε : 0 ≤ ε) {atoms : List AtomIn} {w : Nat}
    (h : pureWordP? P atoms = some w) (hR : WindowRangeP P R atoms) :
    ∃ v, wordQ w = some v ∧ |v - chainQ atoms| ≤ pureBound ε atoms :=
  windowWordP_err hS hε h hR

/-- The chain's step on `P`: a window's word promoted into the total. -/
def chainStepP (P : Pipeline) (t : Nat) (g : List AtomIn) : Option Nat := (windowWordP? P g).bind (promoteWord? t)

/-- **The chain's range in windows of `G`**: every window's range, along the chain's windows. -/
def ChainRangeP (P : Pipeline) (R : Term → AtomIn → Prop) (G : Nat) (atoms : List AtomIn) : Prop :=
  FoldRange (chainStepP P) (fun _ g => WindowRangeP P R g) (atoms.toChunks G) 0

/-- A promotion's relative error over a window of at most `G` atoms: `(1 + 2^−24)·(1 + ε)^G − 1` (`epsC` at `eps8`,
`G = 4`). -/
def epsCP (ε : ℚ) (G : Nat) : ℚ := (1 + 1 / 2 ^ 24) * (1 + ε) ^ G - 1

theorem epsCP_ge (hε : 0 ≤ ε) (G : Nat) : 1 / 2 ^ 24 ≤ epsCP ε G := by
  unfold epsCP
  have h := one_le_pow₀ (n := G) (by linarith : (1 : ℚ) ≤ 1 + ε)
  nlinarith

theorem chainStepP_err (hS : StepRel P R ε) (hε : 0 ≤ ε) {G c c' : Nat} {g : List AtomIn} {cv : ℚ}
    (hg : g.length ≤ G) (hR : WindowRangeP P R g) (hc : wordQ c = some cv) (hs : chainStepP P c g = some c') :
    ∃ v, wordQ c' = some v ∧
      |v - (cv + (g.map atomQ).sum)| ≤ 1 / 2 ^ 24 * |cv| + (epsCP ε G * (g.map atomAbs).sum + 2 ^ (-150 : ℤ)) := by
  obtain ⟨W, hW, hp⟩ := Option.bind_eq_some_iff.1 hs
  obtain ⟨wv, hwv, hwe⟩ := windowWordP_err hS hε hW hR
  obtain ⟨s', hs', hss⟩ := promoteWord_val hp
  obtain ⟨sv, hsv, hse⟩ := addWord_err hc hwv hs'
  refine ⟨sv, by rw [hss]; exact hsv, ?_⟩
  set A := (g.map atomAbs).sum
  set Wq := (g.map atomQ).sum
  have hWA : |Wq| ≤ A := abs_listSum_le atomQ atomAbs (fun x => Finset.abs_sum_le_sum_abs _ _) g
  have hK : (1 + ε) ^ g.length ≤ (1 + ε) ^ G := pow_le_pow_right₀ (by linarith) hg
  have hK1 : 1 ≤ (1 + ε) ^ g.length := one_le_pow₀ (by linarith)
  have hA : 0 ≤ A := le_trans (abs_nonneg _) hWA
  have hwe' : |wv - Wq| ≤ ((1 + ε) ^ G - 1) * A := le_trans hwe (mul_le_mul_of_nonneg_right (by linarith) hA)
  have hwa : |wv| ≤ (1 + ε) ^ G * A := by
    have := abs_sub_abs_le_abs_sub wv Wq; nlinarith
  have hcw : |cv + wv| ≤ |cv| + (1 + ε) ^ G * A := le_trans (abs_add_le _ _) (by linarith)
  calc |sv - (cv + Wq)| = |(sv - (cv + wv)) + (wv - Wq)| := by ring_nf
    _ ≤ |sv - (cv + wv)| + |wv - Wq| := abs_add_le _ _
    _ ≤ (|cv| + (1 + ε) ^ G * A) / 2 ^ 24 + 2 ^ (-150 : ℤ) + ((1 + ε) ^ G - 1) * A := by
        gcongr; exact le_trans hse (by gcongr)
    _ = _ := by unfold epsCP; ring

/-- **The chain's bound in windows of `G`** over its `W` windows (`chainBound` at `eps8`, `G = 4`):
`((1 + 2^−24)^W − 1)·Σ |a·b| + (1 + 2^−24)^W·(epsCP ε G·Σ |a·b| + W·2^−150)`. -/
def chainBoundP (ε : ℚ) (G : Nat) (atoms : List AtomIn) : ℚ :=
  ((1 + 1 / 2 ^ 24) ^ (atoms.toChunks G).length - 1) * (atoms.map atomAbs).sum +
    (1 + 1 / 2 ^ 24) ^ (atoms.toChunks G).length *
      (epsCP ε G * (atoms.map atomAbs).sum + (atoms.toChunks G).length * 2 ^ (-150 : ℤ))

theorem chainBoundP_hopper : chainBoundP eps8 4 = chainBound := rfl

/-- **The chain's error in windows of `G ≥ 1`** (`chainWord_err` on any step): `chainWordP? P G`'s word is within
`chainBoundP ε G` of the exact chain. -/
theorem chainWordP_err (hS : StepRel P R ε) (hε : 0 ≤ ε) {G : Nat} (hG : 1 ≤ G) {atoms : List AtomIn} {w : Nat}
    (h : chainWordP? P G atoms = some w) (hR : ChainRangeP P R G atoms) :
    ∃ v, wordQ w = some v ∧ |v - chainQ atoms| ≤ chainBoundP ε G atoms := by
  obtain ⟨G', rfl⟩ : ∃ G', G = G' + 1 := ⟨G - 1, by omega⟩
  have hch : atoms.toChunks (G' + 1) = Freeze.chunksOf G' atoms := Freeze.toChunks_succ G' atoms
  have hlen : ∀ g ∈ atoms.toChunks (G' + 1), ∀ _c : Nat, g.length ≤ G' + 1 := by
    intro g hg _; rw [hch] at hg; exact chunksOf_length_le G' atoms g hg
  have hA : ∀ g : List AtomIn, |(g.map atomQ).sum| ≤ (g.map atomAbs).sum :=
    abs_listSum_le atomQ atomAbs (fun x => Finset.abs_sum_le_sum_abs _ _)
  have key := fold_rel2 (chainStepP P) (fun _ g => g.length ≤ G' + 1 ∧ WindowRangeP P R g)
    (fun g => (g.map atomQ).sum) (fun g => (g.map atomAbs).sum)
    (fun g => epsCP ε (G' + 1) * (g.map atomAbs).sum + 2 ^ (-150 : ℤ)) (by positivity) hA
    (fun g => add_nonneg (mul_nonneg (le_trans (by positivity) (epsCP_ge hε _)) (le_trans (abs_nonneg _) (hA g)))
      (by positivity))
    (fun c g c' cv hR hc hs => chainStepP_err hS hε hR.1 hR.2 hc hs)
  obtain ⟨v, hv, he⟩ := key (atoms.toChunks (G' + 1)) 0 w 0 Freeze.wordQ_zero h
    (foldRange_and _ _ _ _ _ hlen hR)
  refine ⟨v, hv, ?_⟩
  rw [zero_add, abs_zero, zero_add] at he
  have e1 : ((atoms.toChunks (G' + 1)).map fun g => (g.map atomQ).sum).sum = chainQ atoms := by
    rw [hch, chunksOf_sum]; rfl
  have e2 : ((atoms.toChunks (G' + 1)).map fun g => (g.map atomAbs).sum).sum = (atoms.map atomAbs).sum := by
    rw [hch, chunksOf_sum]
  have e3 : ((atoms.toChunks (G' + 1)).map fun g => epsCP ε (G' + 1) * (g.map atomAbs).sum + 2 ^ (-150 : ℤ)).sum =
      epsCP ε (G' + 1) * (atoms.map atomAbs).sum + (atoms.toChunks (G' + 1)).length * 2 ^ (-150 : ℤ) := by
    rw [List.sum_map_add, List.map_const', List.sum_replicate, nsmul_eq_mul, List.sum_map_mul_left, e2]
  rw [e1, e2, e3] at he
  exact he

/-- **The chain's range at `G`**: the pure chain's (one window over every atom) at `G = 0`, `ChainRangeP` otherwise. -/
def ChainRangeAt (P : Pipeline) (R : Term → AtomIn → Prop) (G : Nat) (atoms : List AtomIn) : Prop :=
  if G = 0 then WindowRangeP P R atoms else ChainRangeP P R G atoms

/-- **The chain's bound at `G`**: `pureBound ε` at `G = 0`, `chainBoundP ε G` otherwise. -/
def chainBoundAt (ε : ℚ) (G : Nat) (atoms : List AtomIn) : ℚ :=
  if G = 0 then pureBound ε atoms else chainBoundP ε G atoms

theorem chainBoundAt_zero (ε : ℚ) : chainBoundAt ε 0 = pureBound ε := funext fun _ => ite_eq_left rfl

theorem chainBoundAt_succ (ε : ℚ) (G : Nat) : chainBoundAt ε (G + 1) = chainBoundP ε (G + 1) :=
  funext fun _ => ite_eq_right (Nat.succ_ne_zero G)

/-- **The chain's error at any `G`**: `chainWordP? P G`'s word is within `chainBoundAt ε G` of the exact chain. -/
theorem chainWordAt_err (hS : StepRel P R ε) (hε : 0 ≤ ε) (G : Nat) {atoms : List AtomIn} {w : Nat}
    (h : chainWordP? P G atoms = some w) (hR : ChainRangeAt P R G atoms) :
    ∃ v, wordQ w = some v ∧ |v - chainQ atoms| ≤ chainBoundAt ε G atoms := by
  unfold ChainRangeAt at hR
  unfold chainBoundAt
  split_ifs at hR ⊢ with hG
  · subst hG
    rw [AccumProofs.chainWordP?_zero] at h
    exact pureWordP_err hS hε h hR
  · exact chainWordP_err hS hε (by omega) h hR

/-! ## U on the step `P` in windows of `G` -/

/-- `peelRows` on `chainWordP? P G`: `C̃`'s word and the rows `P_A,i`, `P_B,j` before U's four steps. -/
def peelRowsP (P : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)
    (i : Fin m) (j : Fin n) : Option (Nat × List Nat × List Nat) := do
  let c ← chainWordP? P G (rowAtoms (Aq i) (Bq j))
  let af ← (List.ofFn fun p => chainWordP? P G (rowAtoms (Aq i) (FBᵀ p))).mapM id
  let bf ← (List.ofFn fun q => chainWordP? P G (rowAtoms (Bq j) (FAᵀ q))).mapM id
  let g ← (List.ofFn fun p => (List.ofFn fun q => chainWordP? P G (rowAtoms (FBᵀ p) (FAᵀ q))).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  pure (c, pa, pb)

theorem peelExactP_eq_rows (P : Pipeline) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code)
    (Bq : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat)
    (beB : Matrix (Fin n) (Fin 32) Nat) (i : Fin m) (j : Fin n) :
    peelExactP P G Aq Bq FA FB beA beB i j =
      (peelRowsP P G Aq Bq FA FB beA beB i j).bind fun r => uWord? r.1 r.2.1 r.2.2 := by
  unfold peelExactP peelRowsP
  simp only [Option.bind_eq_bind, Option.bind_assoc, Option.pure_def, Option.bind_some]

theorem peelRowsP_spec {G : Nat} {Aq : Matrix (Fin m) (Fin k) Code} {Bq : Matrix (Fin n) (Fin k) Code}
    {FA FB : Matrix (Fin k) (Fin 32) Code} {beA : Matrix (Fin m) (Fin 32) Nat} {beB : Matrix (Fin n) (Fin 32) Nat}
    {i : Fin m} {j : Fin n} {c : Nat} {pa pb : List Nat}
    (h : peelRowsP P G Aq Bq FA FB beA beB i j = some (c, pa, pb)) :
    ∃ af bf g, chainWordP? P G (rowAtoms (Aq i) (Bq j)) = some c ∧
      (List.ofFn fun p => chainWordP? P G (rowAtoms (Aq i) (FBᵀ p))).mapM id = some af ∧
      (List.ofFn fun q => chainWordP? P G (rowAtoms (Bq j) (FAᵀ q))).mapM id = some bf ∧
      (List.ofFn fun p => (List.ofFn fun q => chainWordP? P G (rowAtoms (FBᵀ p) (FAᵀ q))).mapM id).mapM id =
        some g ∧
      paRow? (List.ofFn (beA i)) af = some pa ∧ pbRow? (List.ofFn (beB j)) bf g = some pb := by
  unfold peelRowsP at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨c', hc, af, haf, bf, hbf, g, hg, pa', hpa, pb', hpb, rfl, rfl, rfl⟩ := h
  exact ⟨af, bf, g, hc, haf, hbf, hg, hpa, hpb⟩

/-- **U's steps, `C̃`'s chain and the rows** (`peelExact_err` on `peelExactValP P G`), with `C̃`'s word `c` of value
`cv`. -/
theorem peelExactP_err (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)
    (i : Fin m) (j : Fin n) {c : Nat} {pa pb : List Nat} {v cv : ℚ}
    (hrows : peelRowsP P G Aq Bq FA FB beA beB i j = some (c, pa, pb))
    (hv : peelExactValP P G Aq Bq FA FB beA beB i j = some v) (hcv : wordQ c = some cv)
    (hR : URange pa pb (List.range 4) c) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤
      ((1 + eps16) ^ 4 - 1) * (|cv| + ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)|) +
        |cv - cQ Aq Bq i j| +
        ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0) -
          paQ Aq FB beA i t * pbQ Bq FA FB beB j t| := by
  unfold peelExactValP at hv
  rw [peelExactP_eq_rows, hrows] at hv
  change (uWord? c pa pb).bind wordQ = some v at hv
  obtain ⟨u, hu, huv⟩ := Option.bind_eq_some_iff.1 hv
  obtain ⟨uv, huv', he⟩ := uWord_err pa pb hcv hu hR
  rw [huv] at huv'
  cases huv'
  have hq : uQ Aq Bq FA FB beA beB i j = cQ Aq Bq i j + ∑ t, paQ Aq FB beA i t * pbQ Bq FA FB beB j t := by
    unfold uQ; exact foldl_k16Q _ _ _
  rw [hq]
  set S := ∑ t : Fin 64, bfVal (pa.getD t 0) * bfVal (pb.getD t 0)
  set Sq := ∑ t, paQ Aq FB beA i t * pbQ Bq FA FB beB j t
  have hd : |S - Sq| ≤ ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0) -
      paQ Aq FB beA i t * pbQ Bq FA FB beB j t| := by
    rw [← Finset.sum_sub_distrib]; exact Finset.abs_sum_le_sum_abs _ _
  calc |v - (cQ Aq Bq i j + Sq)| = |(v - (cv + S)) + (cv - cQ Aq Bq i j) + (S - Sq)| := by ring_nf
    _ ≤ |v - (cv + S)| + |cv - cQ Aq Bq i j| + |S - Sq| := abs_add_three _ _ _
    _ ≤ _ := by gcongr

section Bound

variable (P) (R) (ε) (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
  (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)

/-- `P_A,i`'s rounded entry `32 + p` (`paErr` with `chainBoundAt ε G`). -/
def paErrP (i : Fin m) (p : Fin 32) : ℚ :=
  chainBoundAt ε G (rowAtoms (Aq i) (FBᵀ p)) +
    (|afQ Aq FB i p| + chainBoundAt ε G (rowAtoms (Aq i) (FBᵀ p))) / 2 ^ 8 + 2 ^ (-134 : ℤ)

/-- `P_B,j`'s rounded entry `q` (`pbErr` with `chainBoundAt ε G`). -/
def pbErrP (j : Fin n) (q : Fin 32) : ℚ :=
  pbErrOf (beB j) (fun p => gQ FA FB p q) (fun p => chainBoundAt ε G (rowAtoms (FBᵀ p) (FAᵀ q))) (bfQ Bq FA j q)
    (chainBoundAt ε G (rowAtoms (Bq j) (FAᵀ q)))

/-- Position `t`'s row error (`rowErr` with `chainBoundAt ε G`). -/
def rowErrP (i : Fin m) (j : Fin n) (t : Fin 64) : ℚ :=
  if h : t.val < 32 then |paQ Aq FB beA i t| * pbErrP ε G Bq FA FB beB j ⟨t.val, h⟩
  else paErrP ε G Aq FB i ⟨t.val - 32, by have := t.isLt; omega⟩ * |pbQ Bq FA FB beB j t|

/-- **The served entry's bound at step error `ε` and windows of `G`** (`peelBound` with `chainBoundAt ε G`). -/
def peelBoundP (i : Fin m) (j : Fin n) : ℚ :=
  ((1 + eps16) ^ 4 - 1) * (|cQ Aq Bq i j| + chainBoundAt ε G (rowAtoms (Aq i) (Bq j)) +
      ∑ t, (|paQ Aq FB beA i t * pbQ Bq FA FB beB j t| + rowErrP ε G Aq Bq FA FB beA beB i j t)) +
    chainBoundAt ε G (rowAtoms (Aq i) (Bq j)) + ∑ t, rowErrP ε G Aq Bq FA FB beA beB i j t

/-- **The range on the step `P` in windows of `G`**: every E4M3 chain of the entry on `ChainRangeAt P R G`, and U's
steps on the K16 range (`URange`) over the served rows. -/
def PeelRangeP (i : Fin m) (j : Fin n) : Prop :=
  ChainRangeAt P R G (rowAtoms (Aq i) (Bq j)) ∧ (∀ p, ChainRangeAt P R G (rowAtoms (Aq i) (FBᵀ p))) ∧
    (∀ q, ChainRangeAt P R G (rowAtoms (Bq j) (FAᵀ q))) ∧
    (∀ p q, ChainRangeAt P R G (rowAtoms (FBᵀ p) (FAᵀ q))) ∧
    ∀ c pa pb, peelRowsP P G Aq Bq FA FB beA beB i j = some (c, pa, pb) → URange pa pb (List.range 4) c

/-- **(b) on the step `P` in windows of `G`**: when `P`'s step has relative error `ε` on `R`, the served U
`peelExactValP P G` is within `peelBoundP ε G` of its exact counterpart on the range, wherever the entry is finite. -/
theorem peelExactValP_err (hS : StepRel P R ε) (hε : 0 ≤ ε) (i : Fin m) (j : Fin n) {v : ℚ}
    (hv : peelExactValP P G Aq Bq FA FB beA beB i j = some v) (hR : PeelRangeP P R G Aq Bq FA FB beA beB i j) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤ peelBoundP ε G Aq Bq FA FB beA beB i j := by
  have hv' := hv
  unfold peelExactValP at hv'
  rw [peelExactP_eq_rows] at hv'
  obtain ⟨u, hu, -⟩ := Option.bind_eq_some_iff.1 hv'
  obtain ⟨⟨c, pa, pb⟩, hrows, -⟩ := Option.bind_eq_some_iff.1 hu
  obtain ⟨af, bf, g, hc, haf, hbf, hg, hpa, hpb⟩ := peelRowsP_spec hrows
  obtain ⟨hRc, hRa, hRb, hRg, hRu⟩ := hR
  obtain ⟨cv, hcv, hce⟩ := chainWordAt_err hS hε G hc hRc
  have he := peelExactP_err G Aq Bq FA FB beA beB i j hrows hv hcv (hRu c pa pb hrows)
  obtain ⟨hafl, hafw⟩ := mapM_ofFn _ 0 haf
  obtain ⟨-, hbfw⟩ := mapM_ofFn _ 0 hbf
  obtain ⟨hgl, hgw⟩ := mapM_ofFn _ [] hg
  have hgw' : ∀ p q : Fin 32, chainWordP? P G (rowAtoms (FBᵀ p) (FAᵀ q)) = some ((g.getD p []).getD q 0) :=
    fun p q => (mapM_ofFn _ 0 (hgw p)).2 q
  obtain ⟨hpa1, hpa2⟩ := paRow_spec hpa (by simp)
  obtain ⟨hpb1, hpb2⟩ := pbRow_spec hpb (by simp)
  have hrow : ∀ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0) - paQ Aq FB beA i t * pbQ Bq FA FB beB j t| ≤
      rowErrP ε G Aq Bq FA FB beA beB i j t := by
    intro t
    by_cases ht : t.val < 32
    · have e1 : bfVal (pa.getD t 0) = paQ Aq FB beA i t := by
        rw [hpa1 t ht, show (t : Nat) = ((⟨t.val, ht⟩ : Fin 32) : Nat) from rfl, getD_ofFn]
        unfold paQ; rw [dite_eq_left ht]
      obtain ⟨acc, s, hacc, hs, hh⟩ := hpb1 t ht
      have hpbe := pbEntry_err (be := beB j) (G := fun p => gQ FA FB p ⟨t.val, ht⟩)
        (δG := fun p => chainBoundAt ε G (rowAtoms (FBᵀ p) (FAᵀ ⟨t.val, ht⟩))) hgl
        (fun p => chainWordAt_err hS hε G (hgw' p ⟨t.val, ht⟩) (hRg _ _))
        (chainWordAt_err hS hε G (hbfw ⟨t.val, ht⟩) (hRb _)) hacc hs hh
      have e2 : pbQ Bq FA FB beB j t = (∑ p, bfVal (beB j p) * gQ FA FB p ⟨t.val, ht⟩) - bfQ Bq FA j ⟨t.val, ht⟩ := by
        unfold pbQ accQ; rw [dite_eq_left ht]
      unfold rowErrP
      rw [dite_eq_left ht, e1, ← mul_sub, abs_mul, e2]
      exact mul_le_mul_of_nonneg_left hpbe (abs_nonneg _)
    · have hp : t.val - 32 < 32 := by have := t.isLt; omega
      have e1 : bfVal (pb.getD t 0) = pbQ Bq FA FB beB j t := by
        have h1 := hpb2 (t.val - 32) hp
        rw [show 32 + (t.val - 32) = t.val by omega,
          show t.val - 32 = ((⟨t.val - 32, hp⟩ : Fin 32) : Nat) from rfl, getD_ofFn] at h1
        rw [h1, bfVal_xor]
        unfold pbQ; rw [dite_eq_right ht]
      have hpa' : rnBf16? (af.getD (t.val - 32) 0) = some (pa.getD t 0) := by
        have h1 := hpa2 (t.val - 32) (by rw [hafl]; exact hp)
        rwa [show 32 + (t.val - 32) = t.val by omega] at h1
      have hpae := paEntry_err (A := afQ Aq FB i ⟨t.val - 32, hp⟩)
        (δ := chainBoundAt ε G (rowAtoms (Aq i) (FBᵀ ⟨t.val - 32, hp⟩)))
        (chainWordAt_err hS hε G (hafw _) (hRa _)) hpa'
      have e2 : paQ Aq FB beA i t = afQ Aq FB i ⟨t.val - 32, hp⟩ := by unfold paQ; rw [dite_eq_right ht]
      unfold rowErrP
      rw [dite_eq_right ht, e1, ← sub_mul, abs_mul, e2]
      exact mul_le_mul_of_nonneg_right hpae (abs_nonneg _)
  set δC := chainBoundAt ε G (rowAtoms (Aq i) (Bq j))
  have hC : |cv - cQ Aq Bq i j| ≤ δC := hce
  have hCa : |cv| ≤ |cQ Aq Bq i j| + δC := by
    have := abs_sub_abs_le_abs_sub cv (cQ Aq Bq i j); linarith
  have hS' : ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)| ≤
      ∑ t, (|paQ Aq FB beA i t * pbQ Bq FA FB beB j t| + rowErrP ε G Aq Bq FA FB beA beB i j t) := by
    refine Finset.sum_le_sum fun t _ => ?_
    have := abs_sub_abs_le_abs_sub (bfVal (pa.getD t 0) * bfVal (pb.getD t 0))
      (paQ Aq FB beA i t * pbQ Bq FA FB beB j t)
    linarith [hrow t]
  have hK : 0 ≤ (1 + eps16) ^ 4 - 1 := by
    have := one_le_pow₀ (n := 4) (by linarith [eps16_nonneg] : (1 : ℚ) ≤ 1 + eps16); linarith
  calc |v - uQ Aq Bq FA FB beA beB i j| ≤ _ := he
    _ ≤ peelBoundP ε G Aq Bq FA FB beA beB i j := by
        unfold peelBoundP
        gcongr ?_ + ?_ + ?_
        · exact mul_le_mul_of_nonneg_left (by linarith) hK
        · exact Finset.sum_le_sum fun t _ => hrow t

end Bound

/-! ## sm_120: width 26, floor −133 -/

/-- **sm_120's step range**: `HRangeW 26 (−133)` on the atom, the step's exponent in `[−101, 120]` unless every term is
zero. -/
def Sm120Range (tc : Term) (x : AtomIn) : Prop := HRangeW 26 (-133) tc x.1 x.2

/-- sm_120's step has relative error `epsW 26 = 33·2^−25 + (1 + 33·2^−25)·2^−23`, about `2^−19.8`, on its range. -/
theorem sm120_stepRel : StepRel sm120E4m3 Sm120Range (epsW 26) := stepRel_pipe (by norm_num) (by norm_num)

section Sm120

variable (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
  (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)

/-- **(b) at sm_120, any `G`**: the served U `peelExactValP sm120E4m3 G` is within `peelBoundP (epsW 26) G` of its
exact counterpart, on `PeelRangeP sm120E4m3 Sm120Range G`, wherever the entry is finite. -/
theorem peelExactValP_sm120_err (i : Fin m) (j : Fin n) {v : ℚ}
    (hv : peelExactValP sm120E4m3 G Aq Bq FA FB beA beB i j = some v)
    (hR : PeelRangeP sm120E4m3 Sm120Range G Aq Bq FA FB beA beB i j) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤ peelBoundP (epsW 26) G Aq Bq FA FB beA beB i j :=
  peelExactValP_err sm120E4m3 Sm120Range (epsW 26) G Aq Bq FA FB beA beB sm120_stepRel (epsW_nonneg 26) i j hv hR

/-- **(b) at sm_120 in windows of 4** (`pearl-c-sm120-v1`'s chain). -/
theorem peelExactValP_sm120_four (i : Fin m) (j : Fin n) {v : ℚ}
    (hv : peelExactValP sm120E4m3 4 Aq Bq FA FB beA beB i j = some v)
    (hR : PeelRangeP sm120E4m3 Sm120Range 4 Aq Bq FA FB beA beB i j) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤ peelBoundP (epsW 26) 4 Aq Bq FA FB beA beB i j :=
  peelExactValP_sm120_err 4 Aq Bq FA FB beA beB i j hv hR

/-- **(b) at sm_120 with no promotion** (`G = 0`, the pure chains; v2). -/
theorem peelExactValP_sm120_zero (i : Fin m) (j : Fin n) {v : ℚ}
    (hv : peelExactValP sm120E4m3 0 Aq Bq FA FB beA beB i j = some v)
    (hR : PeelRangeP sm120E4m3 Sm120Range 0 Aq Bq FA FB beA beB i j) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤ peelBoundP (epsW 26) 0 Aq Bq FA FB beA beB i j :=
  peelExactValP_sm120_err 0 Aq Bq FA FB beA beB i j hv hR

/-- The same on `peelExactPureValP sm120E4m3`, U with pure chains. -/
theorem peelExactPureValP_sm120_err (i : Fin m) (j : Fin n) {v : ℚ}
    (hv : peelExactPureValP sm120E4m3 Aq Bq FA FB beA beB i j = some v)
    (hR : PeelRangeP sm120E4m3 Sm120Range 0 Aq Bq FA FB beA beB i j) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤ peelBoundP (epsW 26) 0 Aq Bq FA FB beA beB i j := by
  rw [← AccumProofs.peelExactValP_zero_eq_pure] at hv
  exact peelExactValP_sm120_zero Aq Bq FA FB beA beB i j hv hR

end Sm120

/-! ## (c) at sm_120 -/

/-- **`peelExactP` on constant operand and basis rows** (`peelExact_const` on `chainWordP? P G`). -/
theorem peelExactP_const (P : Pipeline) (G : Nat) (a b f g : Code) (beA : Matrix (Fin m) (Fin 32) Nat)
    (beB : Matrix (Fin n) (Fin 32) Nat) (i : Fin m) (j : Fin n) :
    peelExactP (k := k) P G (fun _ _ => a) (fun _ _ => b) (fun _ _ => f) (fun _ _ => g) beA beB i j =
      (do
        let c ← chainWordP? P G (rowAtoms (fun _ : Fin k => a) (fun _ => b))
        let af ← chainWordP? P G (rowAtoms (fun _ : Fin k => a) (fun _ => g))
        let bf ← chainWordP? P G (rowAtoms (fun _ : Fin k => b) (fun _ => f))
        let gw ← chainWordP? P G (rowAtoms (fun _ : Fin k => g) (fun _ => f))
        let pa ← paRow? (List.ofFn (beA i)) (List.replicate 32 af)
        let pb ← pbRow? (List.ofFn (beB j)) (List.replicate 32 bf) (List.replicate 32 (List.replicate 32 gw))
        uWord? c pa pb) := by
  have hT : ∀ c : Code, (fun _ _ => c : Matrix (Fin k) (Fin 32) Code)ᵀ = fun _ _ => c := fun _ => rfl
  unfold peelExactP
  simp only [hT, List.ofFn_const, mapM_id_replicate (show (32 : Nat) ≠ 0 by norm_num), Option.bind_eq_bind,
    Option.bind_map, Function.comp_def]

/-- The witness's `A′` at sm_120: `x = 0` with `β = 1` (`α = 1`), formed by sm_120's atom-mix. -/
def wA120 (z : Noise 1 1 32 32) : Matrix (Fin 1) (Fin 32) Code :=
  (FormingP.atomMixP sm120E4m3).quantA (fun _ => 1) (fun _ => 1) 0 z

/-- The witness's `B̃` at sm_120: `B = 0` with `β = 1/3` (`α = 1`), formed by sm_120's atom-mix. -/
def wB120 (z : Noise 1 1 32 32) : Matrix (Fin 1) (Fin 32) Code :=
  (FormingP.atomMixP sm120E4m3).quantB (fun _ => 1) (fun _ => 1 / 3) 0 z

/-- The served U's entry at the witness under the salt whose noise is `z`, at sm_120 in windows of `G`:
`peelExactValP sm120E4m3 G` on sm_120's atom-mix operands, the bases' codes and v1's noise words. -/
def wServed120 (G : Nat) (z : Noise 1 1 32 32) : Option ℚ :=
  peelExactValP sm120E4m3 G (wA120 z) (wB120 z) (fun l t => z.FA l t) (fun l t => z.FB l t)
    (beV1 (fun _ => 1) z.EA) (beV1 (fun _ => 1 / 3) z.EB) 0 0

theorem wServed120_eq (G : Nat) (a b : ℚ) (cA cB : Code)
    (hA : (FormingP.atomMixP sm120E4m3).elem 1 0 1 ((wNoise a b).EA 0) (fun _ => e4m3 1) = cA)
    (hB : (FormingP.atomMixP sm120E4m3).elem 1 0 (1 / 3) ((wNoise a b).EB 0) (fun _ => e4m3 1) = cB) :
    wServed120 G (wNoise a b) =
      (peelExactP (k := 32) sm120E4m3 G (fun _ _ => cA) (fun _ _ => cB) (fun _ _ => e4m3 1) (fun _ _ => e4m3 1)
        (beV1 (fun _ => 1) (wNoise a b).EA) (beV1 (fun _ => 1 / 3) (wNoise a b).EB) 0 0).bind wordQ := by
  have hA' : wA120 (wNoise a b) = fun _ _ => cA := by rw [← hA]; rfl
  have hB' : wB120 (wNoise a b) = fun _ _ => cB := by rw [← hB]; rfl
  unfold wServed120 peelExactValP
  rw [hA', hB']
  rfl

set_option maxHeartbeats 0 in
/-- **(c) at sm_120 in windows of 4**: at `Peel.lean`'s degenerate inputs, formed by sm_120's atom-mix, two salts give
the served entries `0` (all-ones noise) and `10` (`E_A`'s row starting with `2`, `E_B`'s with `−1`). -/
theorem served120_live_four : wServed120 4 (wNoise 1 1) = some 0 ∧ wServed120 4 (wNoise 2 (-1)) = some 10 := by
  constructor
  · rw [wServed120_eq 4 1 1 (e4m3 32) (e4m3 11) (by decide +kernel) (by decide +kernel), peelExactP_const]
    decide +kernel
  · rw [wServed120_eq 4 2 (-1) (e4m3 32) (e4m3 10) (by decide +kernel) (by decide +kernel), peelExactP_const]
    decide +kernel

set_option maxHeartbeats 0 in
/-- **(c) at sm_120 with no promotion** (`G = 0`): the same two salts give `0` and `10`. -/
theorem served120_live_zero : wServed120 0 (wNoise 1 1) = some 0 ∧ wServed120 0 (wNoise 2 (-1)) = some 10 := by
  constructor
  · rw [wServed120_eq 0 1 1 (e4m3 32) (e4m3 11) (by decide +kernel) (by decide +kernel), peelExactP_const]
    decide +kernel
  · rw [wServed120_eq 0 2 (-1) (e4m3 32) (e4m3 10) (by decide +kernel) (by decide +kernel), peelExactP_const]
    decide +kernel

/-- The two salts' served entries differ at sm_120 in windows of 4. -/
theorem served120_salt_dependent_four : wServed120 4 (wNoise 1 1) ≠ wServed120 4 (wNoise 2 (-1)) := by
  rw [served120_live_four.1, served120_live_four.2]
  decide

/-- The two salts' served entries differ at sm_120 with no promotion. -/
theorem served120_salt_dependent_zero : wServed120 0 (wNoise 1 1) ≠ wServed120 0 (wNoise 2 (-1)) := by
  rw [served120_live_zero.1, served120_live_zero.2]
  decide

end Pouw.PearlC.PeelExactProofsP
