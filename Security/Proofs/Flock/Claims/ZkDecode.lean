import Proofs.Flock.Claims.Decode
import Proofs.Flock.Soundness.PadCode

/-!
# The masked (`--zk`) level-0 code as an `IRS`, its decoder, and the read step

In a masked session the first committed codeword is level 0 with VEIL's hiding (`Flock.Zk.ligerito`,
`commit_zk` in `flock-zk-b684b12.patch`): `s` lanes, the witness's `2^lanes` then `e` extra lanes (`Zk.EXTRA_LANES = 4`,
a pair per rep), each the Reed–Solomon encoding of `2^cols` message symbols `M` padded by `(Ŵ_cols + c₀)·μ`, `μ` of
`t` coefficients (`t = 2·q₀`, `c₀ = Zk.PAD_C`). On the novel basis the padded lane's coefficients are `M + c₀·μ` at
`[0, t)`, `M` up to `2^cols`, and `μ` at `[2^cols, 2^cols + t)` (`zkCoeffs`, `lane_codeword`), so each lane is a codeword
of `RS[2^d, 2^cols + t]` at the points `ω_q` (`PadCode.encAt`), and the masked code is an interleaved Reed–Solomon code
(`zkLevelZero`): the plain level-0 code with `e` more lanes and `t` more coefficients per lane, on the same positions.
At `t ≤ 2^cols` a masked lane is the plain lane of its message plus the padding's codeword
`Σ_j μ_j·(c₀·X̂_j + X̂_{2^cols + j})` (`zkEnc_eq_add_pad`).
A message (`ZkMessage`) is every lane's `M` and `μ`, a column every lane's symbol (`zkEnc`). The decoder therefore
meets `ListComplete` for it at the Johnson count `johnsonCount (2^cols + t) agree = ⌊(2^cols + t + 1)/agree²⌋ + 1`, at
every `agree > 0` (`listComplete_zkLevelZero`), and on opened columns at distinct positions below `2^64` it is the
Guruswami–Sudan path once `agree ≤ 1` is above `√((2^cols + t + 1)/N)` (`zkLevelZero_regime`), and `[]` outside that
regime (`zkLevelZero_decode_eq_nil`). One run's two reps open at most `t = 2·q₀` positions of the codeword, fewer than
the count (`pad_lt_zkJohnsonCount`): it is reached only by `E^rep`'s columns, gathered over its accepted reruns.

The read step (`zkRead`) drops the padding and the extra lanes, keeps the witness lanes' messages (`zkWitnessMsg`), and
reads the packed trace (`Model.witnessOf`) as the hidden values, bit `a` of the trace being hidden value `a`
(`readTrace`): the masked statement's commitments are `hm96` rows the circuit computes from trace bits, so their salts
are trace bits too.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model Polynomial

/-! ## The masked code -/

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

/-- A lane's message and padding are its coefficients' to read back. -/
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

/-- **The masked level-0 column** at position `q`: every lane's codeword symbol, its coefficients encoded at `d` bits. -/
def zkEnc {F K : Type} [Field F] [Field K] (A : Arith F K) (d : ℕ) (c₀ : F) {s cols t : ℕ}
    (M : ZkMessage F s cols t) (q : ℕ) : Fin s → F :=
  fun ℓ => encAt A d (2 ^ cols + t) (zkCoeffs c₀ (M.1 ℓ) (M.2 ℓ)) q

/-- **A masked lane is the plain lane plus the padding's codeword**: at `t ≤ 2^cols`, lane `ℓ`'s symbol at `q` is its
`2^cols` message symbols on the first `2^cols` monomials, plus `Σ_j μ_j·(c₀·X̂_j + X̂_{2^cols + j})(ω_q)`, the
padding term the model's level-0 row check corrects by (`Model.padCorr`). -/
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

/-- The lane polynomial determines the coefficients: the novel basis is linearly independent. -/
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

/-- **The masked code's Johnson count** at `agree`: `⌊(2^cols + t + 1)/agree²⌋ + 1`. -/
theorem zkLevelZero_nMin {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) :
    (zkLevelZero A hA s cols t d c₀ hd hk).nMin agree = johnsonCount (2 ^ cols + t) agree := rfl

/-- **The masked level-0 decoder meets `ListComplete`** for the masked columns, at the Johnson count
`⌊(2^cols + t + 1)/agree²⌋ + 1`, at every `agree > 0`. -/
theorem listComplete_zkLevelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) (ha : 0 < agree) :
    ListComplete (zkEnc A d c₀ (s := s) (cols := cols) (t := t)) agree (fun _ => johnsonCount (2 ^ cols + t) agree)
      ((zkLevelZero A hA s cols t d c₀ hd hk).decode agree) := by
  rw [← zkLevelZero_enc A hA s cols t d c₀ hd hk]
  exact (zkLevelZero A hA s cols t d c₀ hd hk).listComplete_decode agree ha

/-- Opened masked columns at distinct positions below `2^64` sit at distinct points. -/
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

/-- **The masked code's Johnson regime**: opened columns at distinct positions below `2^64`, and `agree ≤ 1`
strictly above `√((2^cols + t + 1)/N)`. -/
theorem zkLevelZero_regime {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) (os : List (ℕ × (Fin s → F))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ 64) (ha : agree ≤ 1)
    (hJ : Real.sqrt ((2 ^ cols + t + 1) * os.length) < agree * os.length) :
    (zkLevelZero A hA s cols t d c₀ hd hk).Regime agree os :=
  (zkLevelZero A hA s cols t d c₀ hd hk).regime_of_johnson agree os
    (zkLevelZero_points_nodup A hA s cols t d c₀ hd hk os hnd hlt) ha (by simpa [zkLevelZero] using hJ)

/-- Outside the masked code's Johnson regime, on opened columns at distinct positions below `2^64`, the decoder
returns `[]`. -/
theorem zkLevelZero_decode_eq_nil {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) (os : List (ℕ × (Fin s → F))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ 64) (h : ¬ (zkLevelZero A hA s cols t d c₀ hd hk).Regime agree os) :
    (zkLevelZero A hA s cols t d c₀ hd hk).decode agree os = [] :=
  (zkLevelZero A hA s cols t d c₀ hd hk).decode_eq_nil agree os
    (zkLevelZero_points_nodup A hA s cols t d c₀ hd hk os hnd hlt) h

/-- **One masked run's level-0 openings are fewer than the Johnson count**: at `0 < agree ≤ 1`, the count is above
`2^cols + t + 1`, so above the `t = 2·q₀` positions both reps open. -/
theorem pad_lt_zkJohnsonCount {cols t : ℕ} {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) :
    t < johnsonCount (2 ^ cols + t) agree := by
  exact lt_of_le_of_lt (Nat.le_add_left t (2 ^ cols)) (Nat.lt_of_succ_lt (lt_johnsonCount ha0 ha1))

/-- In the masked code's Johnson regime the decoder is the Guruswami–Sudan path. -/
theorem zkLevelZero_decode_eq_gsPath {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (agree : ℝ) (os : List (ℕ × (Fin s → F))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ 64) (ha : agree ≤ 1)
    (hJ : Real.sqrt ((2 ^ cols + t + 1) * os.length) < agree * os.length) :
    (zkLevelZero A hA s cols t d c₀ hd hk).decode agree os =
      (zkLevelZero A hA s cols t d c₀ hd hk).gsPath agree os
        (zkLevelZero_regime A hA s cols t d c₀ hd hk agree os hnd hlt ha hJ).1 :=
  (zkLevelZero A hA s cols t d c₀ hd hk).decode_eq_gsPath agree os
    (zkLevelZero_regime A hA s cols t d c₀ hd hk agree os hnd hlt ha hJ)

/-! ## The read step -/

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

/-- The masked read sees only the witness lanes' messages: the padding and the extra lanes don't reach the witness. -/
theorem zkRead_eq_of_witnessMsg {F K : Type} [Field F] [Field K] (A : Arith F K) (m lanes e cols t : ℕ)
    {M M' : ZkMessage F (2 ^ lanes + e) cols t} (h : zkWitnessMsg M = zkWitnessMsg M') :
    zkRead A m lanes e cols t M = zkRead A m lanes e cols t M' := by
  simp only [zkRead, h]

end Claims.Flock
