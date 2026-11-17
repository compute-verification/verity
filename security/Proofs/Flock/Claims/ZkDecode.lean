module

public import Proofs.Flock.Claims.Decode
public import Proofs.Flock.Soundness.PadCode

/-! The masked (`--zk`) level-0 code as an `IRS`, its decoder, and the read step. -/

@[expose] public section

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model Polynomial

/-- A masked level-0 message: each of `s` lanes' `2^cols` message symbols and its `t` padding coefficients. -/
abbrev ZkMessage (F : Type) (s cols t : ℕ) := (Fin s → Fin (2 ^ cols) → F) × (Fin s → Fin t → F)

/-- **A masked lane's coefficients** on the first `2^cols + t` novel-basis monomials: `M + c₀·μ` at `[0, t)`, `M` up
to `2^cols`, then `μ` (`lane_codeword`'s `v[i] += c·pad[i]`, `v[L + i] = pad[i]`). -/
def zkCoeffs {F : Type} [Field F] (c₀ : F) {cols t : ℕ} (M : Fin (2 ^ cols) → F) (μ : Fin t → F) :
    Fin (2 ^ cols + t) → F :=
  Fin.append (fun c => M c + if h : c.val < t then c₀ * μ ⟨c.val, h⟩ else 0) μ

theorem zkCoeffs_right {F : Type} [Field F] (c₀ : F) {cols t : ℕ} (M : Fin (2 ^ cols) → F) (μ : Fin t → F)
    (j : Fin t) : zkCoeffs c₀ M μ (Fin.natAdd (2 ^ cols) j) = μ j := by
  simp [zkCoeffs]

theorem zkCoeffs_left {F : Type} [Field F] (c₀ : F) {cols t : ℕ} (M : Fin (2 ^ cols) → F) (μ : Fin t → F)
    (c : Fin (2 ^ cols)) :
    zkCoeffs c₀ M μ (Fin.castAdd t c) = M c + if h : c.val < t then c₀ * μ ⟨c.val, h⟩ else 0 := by
  simp [zkCoeffs]

theorem zkCoeffs_inj {F : Type} [Field F] (c₀ : F) {cols t : ℕ} {M M' : Fin (2 ^ cols) → F} {μ μ' : Fin t → F}
    (h : zkCoeffs c₀ M μ = zkCoeffs c₀ M' μ') : M = M' ∧ μ = μ' := by
  have hμ : μ = μ' := funext fun j => by
    have := congrFun h (Fin.natAdd (2 ^ cols) j)
    rwa [zkCoeffs_right, zkCoeffs_right] at this
  subst hμ
  refine ⟨funext fun c => ?_, rfl⟩
  have := congrFun h (Fin.castAdd t c)
  rw [zkCoeffs_left, zkCoeffs_left] at this
  exact add_right_cancel this

def zkEnc {F K : Type} [Field F] [Field K] (A : Arith F K) (d : ℕ) (c₀ : F) {s cols t : ℕ}
    (M : ZkMessage F s cols t) (q : ℕ) : Fin s → F :=
  fun ℓ => encAt A d (2 ^ cols + t) (zkCoeffs c₀ (M.1 ℓ) (M.2 ℓ)) q

theorem zkEnc_eq_add_pad {F K : Type} [Field F] [Field K] (A : Arith F K) (d : ℕ) (c₀ : F) {s cols t : ℕ}
    (ht : t ≤ 2 ^ cols) (M : ZkMessage F s cols t) (q : ℕ) (ℓ : Fin s) :
    zkEnc A d c₀ M q ℓ = encAt A d (2 ^ cols) (M.1 ℓ) q +
      ∑ j : Fin t, M.2 ℓ j * (c₀ * A.Xhat j.val d (A.omega q) + A.Xhat (2 ^ cols + j.val) d (A.omega q)) := by
  have key : ∑ c : Fin (2 ^ cols), (if h : c.val < t then c₀ * M.2 ℓ ⟨c.val, h⟩ else 0) *
      A.Xhat c.val d (A.omega q) = ∑ j : Fin t, M.2 ℓ j * (c₀ * A.Xhat j.val d (A.omega q)) := by
    let g : ℕ → F := fun c => if h : c < t then c₀ * M.2 ℓ ⟨c, h⟩ * A.Xhat c d (A.omega q) else 0
    have h1 : ∑ c : Fin (2 ^ cols), (if h : c.val < t then c₀ * M.2 ℓ ⟨c.val, h⟩ else 0) *
        A.Xhat c.val d (A.omega q) = ∑ c ∈ Finset.range (2 ^ cols), g c := by
      rw [← Fin.sum_univ_eq_sum_range]
      refine Finset.sum_congr rfl fun c _ => ?_
      simp only [g]
      split_ifs <;> simp
    have h2 : ∑ j : Fin t, M.2 ℓ j * (c₀ * A.Xhat j.val d (A.omega q)) = ∑ c ∈ Finset.range t, g c := by
      rw [← Fin.sum_univ_eq_sum_range]
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [g, j.isLt, ↓reduceDIte]
      ring
    rw [h1, h2]
    refine (Finset.sum_subset (Finset.range_subset_range.2 ht) fun c _ hc => ?_).symm
    simp only [g, Finset.mem_range] at hc ⊢
    simp [hc]
  simp only [zkEnc, encAt, Fin.sum_univ_add, zkCoeffs_left, zkCoeffs_right, Fin.val_castAdd, Fin.val_natAdd,
    add_mul, Finset.sum_add_distrib, key, mul_add]
  ring

theorem encPoly_injective {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) :
    Function.Injective (encPoly A hA d k hd hk) := by
  intro a a' h
  have hsum : ∑ c, (a c - a' c) • xP A hA d hd (Fin.castLE hk c) = 0 := by
    rw [← sub_eq_zero.2 h]
    simp only [encPoly, ← Finset.sum_sub_distrib, sub_smul, Polynomial.smul_eq_C_mul, ← sub_mul]
  funext c
  exact sub_eq_zero.1 (Fintype.linearIndependent_iff.1
    ((xP_linearIndependent A hA d hd).comp _ (Fin.castLE_injective hk)) _ hsum c)

/-- The masked message's lane polynomials. -/
noncomputable def zkRows {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d : ℕ) (c₀ : F) {s cols t : ℕ} (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (M : ZkMessage F s cols t) : Fin s → F[X] :=
  fun ℓ => encPoly A hA d (2 ^ cols + t) hd hk (zkCoeffs c₀ (M.1 ℓ) (M.2 ℓ))

theorem zkRows_injective {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d : ℕ) (c₀ : F) {s cols t : ℕ} (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) : Function.Injective (zkRows A hA d c₀ (s := s) hd hk) := by
  intro M M' h
  have hl : ∀ ℓ, M.1 ℓ = M'.1 ℓ ∧ M.2 ℓ = M'.2 ℓ := fun ℓ =>
    zkCoeffs_inj c₀ (encPoly_injective A hA d _ hd hk (congrFun h ℓ))
  exact Prod.ext (funext fun ℓ => (hl ℓ).1) (funext fun ℓ => (hl ℓ).2)

/-- **The masked level-0 code** as an interleaved Reed–Solomon code: `s` lanes, each of degree below `2^cols + t`,
at the points `ω_q`. -/
noncomputable def zkLevelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) : IRS F (ZkMessage F s cols t) s where
  k := 2 ^ cols + t
  dom := A.omega
  rows := zkRows A hA d c₀ hd hk
  natDegree_rows M ℓ := encPoly_natDegree A hA d _ hd hk (by positivity) _
  ofRows := Function.invFun (zkRows A hA d c₀ hd hk)
  ofRows_rows := Function.leftInverse_invFun (zkRows_injective A hA d c₀ hd hk)

theorem zkLevelZero_enc {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) : (zkLevelZero A hA s cols t d c₀ hd hk).enc = zkEnc A d c₀ := by
  funext M q ℓ
  exact encPoly_eval A hA d _ hd hk _ q

theorem listComplete_zkLevelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) (ha : 0 < agree) :
    ListComplete (zkEnc A d c₀ (s := s) (cols := cols) (t := t)) agree (fun _ => johnsonCount (2 ^ cols + t) agree)
      ((zkLevelZero A hA s cols t d c₀ hd hk).decode agree) := by
  rw [← zkLevelZero_enc A hA s cols t d c₀ hd hk]
  exact (zkLevelZero A hA s cols t d c₀ hd hk).listComplete_decode agree ha

theorem zkLevelZero_points_nodup {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (os : List (ℕ × (Fin s → F))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ 64) :
    (os.map fun p => (zkLevelZero A hA s cols t d c₀ hd hk).dom p.1).Nodup := by
  have hpts : (os.map fun p => A.omega p.1) = (os.map Prod.fst).map A.omega := by rw [List.map_map]; rfl
  change (os.map fun p => A.omega p.1).Nodup
  rw [hpts]
  refine hnd.map_on fun x hx y hy hxy => hA.omega_injective 64 le_rfl ?_ ?_ hxy
  · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hx
    exact hlt p hp
  · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hy
    exact hlt p hp

theorem zkLevelZero_regime {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) (os : List (ℕ × (Fin s → F))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ 64) (ha : agree ≤ 1)
    (hJ : Real.sqrt ((2 ^ cols + t + 1) * os.length) < agree * os.length) :
    (zkLevelZero A hA s cols t d c₀ hd hk).Regime agree os :=
  (zkLevelZero A hA s cols t d c₀ hd hk).regime_of_johnson agree os
    (zkLevelZero_points_nodup A hA s cols t d c₀ hd hk os hnd hlt) ha (by simpa [zkLevelZero] using hJ)

theorem zkLevelZero_decode_eq_nil {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) (os : List (ℕ × (Fin s → F))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ 64) (h : ¬ (zkLevelZero A hA s cols t d c₀ hd hk).Regime agree os) :
    (zkLevelZero A hA s cols t d c₀ hd hk).decode agree os = [] :=
  (zkLevelZero A hA s cols t d c₀ hd hk).decode_eq_nil agree os
    (zkLevelZero_points_nodup A hA s cols t d c₀ hd hk os hnd hlt) h

theorem pad_lt_zkJohnsonCount {cols t : ℕ} {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) :
    t < johnsonCount (2 ^ cols + t) agree := by
  exact lt_of_le_of_lt (Nat.le_add_left t (2 ^ cols)) (Nat.lt_of_succ_lt (lt_johnsonCount ha0 ha1))

/-- The witness lanes' messages: the first `2^lanes` lanes' message symbols; their padding and the `e` extra lanes are
the prover's randomness, not the trace. -/
def zkWitnessMsg {F : Type} {lanes e cols t : ℕ} (M : ZkMessage F (2 ^ lanes + e) cols t) : Message F lanes cols :=
  fun ℓ => M.1 (Fin.castAdd e ℓ)

/-- A level-0 message's trace, `2^m` bits in address order (`Model.witnessOf`: table entry `c·2^lanes + ℓ` packs trace
bits `128·(c·2^lanes + ℓ) + v`). -/
def traceBits {F K : Type} [Field F] [Field K] (A : Arith F K) (m lanes cols : ℕ) (M : Message F lanes cols) :
    List Bool :=
  List.ofFn fun a : Fin (2 ^ m) => decide (witnessOf A m lanes cols M a = 1)

/-- **The read step on a level-0 message**: the trace's bits are the hidden values, hidden value `a` being trace bit
`a`. -/
def readTrace {F K : Type} [Field F] [Field K] (A : Arith F K) (m lanes cols : ℕ) (M : Message F lanes cols) : Wit :=
  ⟨traceBits A m lanes cols M⟩

/-- **The read step on a masked message**: the witness lanes' trace (`readTrace` of `zkWitnessMsg`). -/
def zkRead {F K : Type} [Field F] [Field K] (A : Arith F K) (m lanes e cols t : ℕ)
    (M : ZkMessage F (2 ^ lanes + e) cols t) : Wit :=
  readTrace A m lanes cols (zkWitnessMsg M)

theorem readTrace_hid {F K : Type} [Field F] [Field K] (A : Arith F K) (m lanes cols : ℕ) (M : Message F lanes cols)
    (a : Fin (2 ^ m)) : (readTrace A m lanes cols M).hid.getD a false = decide (witnessOf A m lanes cols M a = 1) := by
  simp [readTrace, traceBits]

theorem readTrace_hid_length {F K : Type} [Field F] [Field K] (A : Arith F K) (m lanes cols : ℕ)
    (M : Message F lanes cols) : (readTrace A m lanes cols M).hid.length = 2 ^ m := by
  simp [readTrace, traceBits]

end Claims.Flock
