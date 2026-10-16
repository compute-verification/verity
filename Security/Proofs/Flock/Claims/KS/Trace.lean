import Proofs.Flock.Claims.KS.Joint
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.LigeritoPadLevel0

/-!
# The old table at the filled word

`TableBoundθ` (`KS.Joint`) is the old theorems' fixed-table bound at the deployed verifier. The old bound's event is
`¬ Committed A S sch T` at an oracle `T`, the spec's `¬ GoodTable C` at a word of columns `C`. This file maps one to
the other.

* **The word as the old oracle** (`oracleOf`): lane `ℓ` at position `q` is column `q`'s symbol `ℓ`. When every column
  has the code's width, `Close γ` (every lane agreeing on a `1 − γ` fraction of the positions) is whole columns
  agreeing on that fraction (`agree_of_close`).
* **The plain record**: the deployed code is the old `encode` (`plainEncD_encode`), the read step the bits of the old
  `witnessOf` (`plainReadD_hid`), and the deployed schedule the old model's (`fast100_plainSched`), whose level 0 is
  the deployed one (`level₀_plainSched`) with the deployed `θ` as its `radius` (`radius_plain`). So a table of
  full-width columns whose oracle is `Committed` at the deployed schedule extracts (`good_of_committed_plain`), given
  that a satisfying old witness reads as a witness of the parsed statement.
* **The masked record**: a masked message's lanes are coefficient rows of the old padded code (`zkRowsOf`,
  `zkEnc_eq_encAt`), the read step reads the old `padWit` of them (`zkReadD_eq_padWit`), and the deployed `θ` is the
  old `padRadius` (`radius_zk`). A masked column is an opened row's witness lanes, so the old masked bound's
  `CommittedPad` (`padList` with no extra lanes) reads exactly the masked columns: a table of witness-lane columns
  whose oracle is `CommittedPad` at the deployed masked schedule and padding extracts (`good_of_committed_zk`), its
  rows taken back to a masked message (`zkOfRows`).
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model FlockSoundness.Accounting FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## The word as the old oracle -/

/-- **A word of columns as the old oracle**: lane `ℓ` at position `q` is column `q`'s symbol `ℓ`; `0` past the word or
the column. -/
def oracleOf {F : Type} [Zero F] {N : ℕ} (C : Fin N → List F) : ℕ → ℕ → F :=
  fun ℓ q => if h : q < N then (C ⟨q, h⟩).getD ℓ 0 else 0

/-- **`Close` is whole columns agreeing**: when every column has the code's `2^lanes` symbols and is the codeword's
column wherever every lane agrees, a `γ`-close oracle's word agrees with the codeword on a `1 − γ` fraction of the
positions. -/
theorem agree_of_close {F : Type} [Field F] [DecidableEq F] {lanes d N : ℕ} (hN : N = 2 ^ d) {γ : ℝ}
    (C : Fin N → List F) (hw : ∀ q, (C q).length = 2 ^ lanes) (U : Table F lanes d) (f : ℕ → List F)
    (hf : ∀ q : Fin (2 ^ d), f q = List.ofFn fun ℓ => U ℓ q)
    (h : Close γ (fun (ℓ : Fin (2 ^ lanes)) (q : Fin (2 ^ d)) => oracleOf C ℓ q) U) :
    (1 - γ) * 2 ^ d ≤ ((Finset.univ : Finset (Fin (2 ^ d))).filter fun q => f q.val = extTab C q.val).card := by
  subst hN
  unfold Close at h
  refine le_trans (ge_iff_le.1 h) ?_
  refine Nat.cast_le.2 (Finset.card_le_card fun q => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  intro hall
  rw [hf q, extTab, dite_eq_left q.isLt]
  refine List.ext_getElem (by simp [hw q]) fun n h1 h2 => ?_
  have e := hall ⟨n, by simpa using h1⟩
  simp only [oracleOf, q.isLt, ↓reduceDIte, Fin.eta, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2,
    Option.getD_some] at e
  simpa using e.symm

/-! ## The plain record -/

/-- **The deployed plain code is the old `encode`**: a trace's column at a position is every lane's old codeword
symbol there. -/
theorem plainEncD_encode (i : PlainIdx) (M : Message 𝔽 (plainLanes i) (plainCols i)) (q : Fin (2 ^ plainLen i)) :
    plainEncD ⟨i, M⟩ q = List.ofFn fun ℓ => encode execArith (plainLen i) M ℓ q := by
  rw [plainEncD_eq]
  rfl

/-- **The deployed plain read step is the old witness's bits**: hidden value `a` is `witnessOf`'s bit `a`. -/
theorem plainReadD_hid (i : PlainIdx) (M : Message 𝔽 (plainLanes i) (plainCols i)) (a : Fin (2 ^ i.m)) :
    (plainReadD ⟨i, M⟩).hid.getD a false = decide (witnessOf execArith i.m (plainLanes i) (plainCols i) M a = 1) :=
  readTrace_hid _ _ _ _ _ _

theorem plainSched_some (i : PlainIdx) : _root_.Flock.fast100 i.m = some (plainSched i.m) := by
  have h := (plainShape_table i).1
  unfold plainSched
  cases hf : _root_.Flock.fast100 i.m with
  | none => rw [hf] at h; cases h
  | some s => rfl

/-- **The deployed plain schedule is the old model's** (`fast100_schedOf`). -/
theorem fast100_plainSched (i : PlainIdx) : Accounting.fast100 i.m = some (schedOf (plainSched i.m)) := by
  rw [← fast100_schedOf, plainSched_some]
  rfl

/-- The plain level 0's rate is `1/2` (`log 1`), and the old model's level 0 is the deployed one. -/
theorem plainLevel_table : ∀ i : PlainIdx,
    (plainLevel0 i.m).rate = 1 ∧ (schedOf (plainSched i.m)).level₀ = levelOf (plainLevel0 i.m) := by
  decide +kernel

theorem level₀_plainSched (i : PlainIdx) : (schedOf (plainSched i.m)).level₀ = levelOf (plainLevel0 i.m) :=
  (plainLevel_table i).2

theorem k0_plainSched (i : PlainIdx) : (schedOf (plainSched i.m)).k0 = plainLanes i := (plainShape_table i).2.1.symm

theorem logLen_plain (i : PlainIdx) : (levelOf (plainLevel0 i.m)).logLen = plainLen i := by
  show (plainLevel0 i.m).cols + (plainLevel0 i.m).rate = plainLen i
  rw [(plainLevel_table i).1, (plainShape_table i).2.2.2.1]
  rfl

/-- **The deployed plain `θ` is the old level-0 `radius`**, `1 − √(1/2) − 1/200`. -/
theorem radius_plain (M : PlainTrace) : plainThetaD M = (levelOf (plainLevel0 M.1.m)).radius := by
  obtain ⟨i, M⟩ := M
  have hd : plainLen i = plainCols i + 1 := (plainShape_table i).2.2.2.1
  have hr := (plainLevel_table i).1
  unfold plainThetaD Level.radius Level.rate eta
  have e : ((2 : ℝ) ^ plainCols i / 2 ^ plainLen i) = (2⁻¹ : ℝ) ^ (levelOf (plainLevel0 i.m)).logInvRate := by
    show _ = (2⁻¹ : ℝ) ^ (plainLevel0 i.m).rate
    rw [hr, hd]
    rw [pow_succ]
    field_simp
  rw [e]

/-! ## The masked record -/

/-- **A masked message's coefficient rows**: each lane's first `2^cols + t` coefficients (`zkCoeffs`). -/
def zkRowsOf {F : Type} [Field F] (c₀ : F) {s cols t : ℕ} (M : ZkMessage F s cols t) :
    Fin s → Fin (2 ^ cols + t) → F :=
  fun ℓ => zkCoeffs c₀ (M.1 ℓ) (M.2 ℓ)

/-- **A masked lane is the old padded code's** at its coefficient row. -/
theorem zkEnc_eq_encAt {F K : Type} [Field F] [Field K] (A : Arith F K) (d : ℕ) (c₀ : F) {s cols t : ℕ}
    (M : ZkMessage F s cols t) (q : ℕ) (ℓ : Fin s) :
    zkEnc A d c₀ M q ℓ = encAt A d (2 ^ cols + t) (zkRowsOf c₀ M ℓ) q := rfl

/-- **The old padded witness of a masked message's rows is its witness lanes' messages.** -/
theorem padWit_zkRowsOf {F : Type} [Field F] (c₀ : F) {lanes e cols t : ℕ} (M : ZkMessage F (2 ^ lanes + e) cols t) :
    padWit c₀ (zkRowsOf c₀ M) = zkWitnessMsg M := by
  funext ℓ x
  simp only [padWit, padMsgAt, zkRowsOf, zkCoeffs, zkWitnessMsg, Fin.append_left]
  split_ifs with h
  · have : (⟨2 ^ cols + x.val, by omega⟩ : Fin (2 ^ cols + t)) = Fin.natAdd (2 ^ cols) ⟨x.val, h⟩ := rfl
    rw [this, Fin.append_right]
    ring
  · ring

/-- **The deployed masked read step reads the old padded witness** of the message's rows. -/
theorem zkReadD_eq_padWit (i : ZkIdx) (M : ZkMessage 𝔽 (zkWidth i) (zkCols i) (zkPad i)) :
    zkReadD ⟨i, M⟩ = readTrace execArith i.m (zkLanes i) (zkCols i)
      (padWit (e := 0) (FlockLevel3.GF128.of _root_.Flock.Zk.PAD_C)
        (zkRowsOf (FlockLevel3.GF128.of _root_.Flock.Zk.PAD_C) M)) := by
  rw [padWit_zkRowsOf (e := 0)]
  rfl

theorem zkLevel_rate : ∀ i : ZkIdx, (zkLevel0 i.m).rate = 1 := by
  decide +kernel

/-- **The deployed masked `θ` is the old `padRadius`** at the deployed padding. -/
theorem radius_zk (M : ZkTrace) : zkThetaD M = (levelOf (zkLevel0 M.1.m)).padRadius (zkPad M.1) := by
  obtain ⟨i, M⟩ := M
  have hd : zkLen i = zkCols i + 1 := (zkShape_table i).2.2.2.1
  unfold zkThetaD Level.padRadius Level.padRate eta
  have hl : (levelOf (zkLevel0 i.m)).logLen = zkLen i := by
    show (zkLevel0 i.m).cols + (zkLevel0 i.m).rate = zkLen i
    rw [zkLevel_rate i, hd]
    rfl
  rw [hl]
  push_cast
  rfl

/-- **Rows as a masked message**: lane `ℓ`'s message the row's `padMsgAt`, its padding the row's last `t`
coefficients. -/
def zkOfRows {F : Type} [Field F] (c₀ : F) {s cols t : ℕ} (R : Fin s → Fin (2 ^ cols + t) → F) :
    ZkMessage F s cols t :=
  (fun ℓ => padMsgAt c₀ (R ℓ), fun ℓ j => R ℓ (Fin.natAdd (2 ^ cols) j))

theorem zkRowsOf_zkOfRows {F : Type} [Field F] (c₀ : F) {s cols t : ℕ} (R : Fin s → Fin (2 ^ cols + t) → F) :
    zkRowsOf c₀ (zkOfRows c₀ R) = R := by
  funext ℓ c
  refine Fin.addCases (fun x => ?_) (fun j => ?_) c
  · rw [zkRowsOf, zkCoeffs_left]
    simp only [zkOfRows, padMsgAt]
    split_ifs with h
    · have : (⟨2 ^ cols + x.val, by omega⟩ : Fin (2 ^ cols + t)) = Fin.natAdd (2 ^ cols) ⟨x.val, h⟩ := rfl
      rw [this]
      ring
    · ring
  · rw [zkRowsOf, zkCoeffs_right]
    rfl

/-- The masked schedule's level 0 is the deployed one, and its `k0` the witness lanes. -/
theorem zkLevel_table : ∀ i : ZkIdx,
    (schedOf (zkSched i.m).1).level₀ = levelOf (zkLevel0 i.m) ∧ (schedOf (zkSched i.m).1).k0 = zkLanes i := by
  decide +kernel

theorem logLen_zk (i : ZkIdx) : (levelOf (zkLevel0 i.m)).logLen = zkLen i := by
  show (zkLevel0 i.m).cols + (zkLevel0 i.m).rate = zkLen i
  rw [zkLevel_rate i, (zkShape_table i).2.2.2.1]
  rfl

end Claims.Flock

/-! ## A committed table extracts, at the plain record -/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Model FlockSoundness.Refine Claims.Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **A close table extracts**, at the plain record: a table of full-width columns at a statement whose table 0 is at
index `i`, `θ`-close to the old encoding of a message whose read step is a witness. -/
theorem good_of_close_plain (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployPlain rel).parse x = some s) (i : PlainIdx)
    (hN : wordLen false x = 2 ^ plainLen i) (C : Fin (wordLen false x) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ plainLanes i) (M : Message 𝔽 (plainLanes i) (plainCols i))
    (hclose : Close (plainThetaD ⟨i, M⟩)
      (fun (ℓ : Fin (2 ^ plainLanes i)) (q : Fin (2 ^ plainLen i)) => oracleOf C ℓ q) (encode execArith (plainLen i) M))
    (hrel : Claims.Flock.RelS s (plainReadD ⟨i, M⟩)) :
    GoodTable (deployPlain rel) (wordLen false) x C := by
  refine ⟨⟨i, M⟩, s, hs, hrel, hN.ge, ?_⟩
  have := agree_of_close hN C hw _ _ (plainEncD_encode i M) hclose
  show (1 - plainThetaD ⟨i, M⟩) * ((2 ^ plainLen i : ℕ) : ℝ) ≤
    ((Finset.univ : Finset (Fin (2 ^ plainLen i))).filter fun q => plainEncD ⟨i, M⟩ q.val = extTab C q.val).card
  push_cast
  exact this

/-- **A committed table extracts**, at the plain record: a table of full-width columns at a statement whose table 0 is
at index `i`, whose oracle is `Committed` for the old statement `S` at the deployed schedule, when a satisfying old
witness of `S` reads as a witness of the parsed statement. -/
theorem good_of_committed_plain (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployPlain rel).parse x = some s) (i : PlainIdx)
    (hN : wordLen false x = 2 ^ plainLen i) (C : Fin (wordLen false x) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ plainLanes i) (S : Statement)
    (hwit : ∀ M : Message 𝔽 (plainLanes i) (plainCols i),
      S.Satisfies (witnessOf execArith S.m (plainLanes i) (plainCols i) M) → Claims.Flock.RelS s (plainReadD ⟨i, M⟩))
    (hc : Committed execArith S (schedOf (plainSched i.m)) (oracleOf C)) :
    GoodTable (deployPlain rel) (wordLen false) x C := by
  have key : ∀ (k0 cols d : ℕ) (γ : ℝ), k0 = plainLanes i → cols = plainCols i → d = plainLen i →
      γ = (levelOf (plainLevel0 i.m)).radius →
      CommittedSatisfies execArith S k0 cols d γ (fun (ℓ : Fin (2 ^ k0)) (q : Fin (2 ^ d)) => oracleOf C ℓ q) →
      GoodTable (deployPlain rel) (wordLen false) x C := by
    intro k0 cols d γ hk hcols hd hγ h
    subst hk hcols hd hγ
    obtain ⟨M, hclose, hsat⟩ := h
    rw [← radius_plain ⟨i, M⟩] at hclose
    exact good_of_close_plain rel x s hs i hN C hw M hclose (hwit M hsat)
  refine key _ _ _ _ (k0_plainSched i) ?_ ?_ ?_ hc
  · rw [level₀_plainSched]
    rfl
  · rw [level₀_plainSched]
    exact logLen_plain i
  · rw [level₀_plainSched]

/-- **At table 0's index**: the word is `2^d` long and a full-width column has `2^lanes` symbols. -/
theorem plain_at {x : Claims.Msg} {spec : SessionSpec} {st : _root_.Flock.Stmt}
    (h : (tables false x).head? = some (spec, st)) {i : PlainIdx} (hi : i.m = st.m) :
    wordLen false x = 2 ^ plainLen i ∧ ∀ u, wide false x u ↔ u.length = 2 ^ plainLanes i := by
  refine ⟨?_, fun u => ?_⟩
  · unfold wordLen
    rw [h]
    simp [plainLen, hi]
  · unfold wide zeroCol
    rw [h]
    simp [plainLanes, hi]

/-- **An uncommitted old table**, at the plain record: a table of full-width columns that does not extract has an
oracle that is not `Committed` at the deployed schedule, for an old statement whose satisfying witnesses read as
witnesses of the parsed statement. -/
theorem not_committed_plain (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployPlain rel).parse x = some s) {spec : SessionSpec}
    {st : _root_.Flock.Stmt}
    (hst : (tables false x).head? = some (spec, st)) (i : PlainIdx) (hi : i.m = st.m)
    (C : Fin (wordLen false x) → List 𝔽) (hw : ∀ q, wide false x (C q)) (S : Statement)
    (hwit : ∀ M : Message 𝔽 (plainLanes i) (plainCols i),
      S.Satisfies (witnessOf execArith S.m (plainLanes i) (plainCols i) M) → Claims.Flock.RelS s (plainReadD ⟨i, M⟩))
    (hg : ¬ GoodTable (deployPlain rel) (wordLen false) x C) :
    ¬ Committed execArith S (schedOf (plainSched i.m)) (oracleOf C) := fun hc =>
  hg (good_of_committed_plain rel x s hs i (plain_at hst hi).1 C (fun q => ((plain_at hst hi).2 _).1 (hw q)) S hwit
    hc)

/-! ## A committed table extracts, at the masked record -/

/-- `PAD_C`, the masked padding's constant, in GF(2^128). -/
abbrev padC : 𝔽 := FlockLevel3.GF128.of _root_.Flock.Zk.PAD_C

theorem padLane_zero (k0 r : ℕ) (ℓ : Fin (2 ^ k0 + 0)) : padLane k0 0 r ℓ = ℓ.val := by
  have : ℓ = Fin.castAdd 0 (⟨ℓ.val, by omega⟩ : Fin (2 ^ k0)) := Fin.ext rfl
  rw [this, padLane, Fin.addCases_left]
  rfl

/-- **A close table extracts**, at the masked record: a table of witness-lane columns at a statement whose table 0 is
at index `i`, close at the deployed `θ` to the old padded encoding of coefficient rows whose padded witness reads as a
witness. -/
theorem good_of_close_zk (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployZk rel).parse x = some s) (i : ZkIdx)
    (hN : wordLen true x = 2 ^ zkLen i) (C : Fin (wordLen true x) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ zkLanes i) (R : Fin (2 ^ zkLanes i) → Fin (2 ^ zkCols i + zkPad i) → 𝔽)
    (hclose : CloseRows execArith (zkLen i) (2 ^ zkCols i + zkPad i) (zkThetaD ⟨i, zkOfRows padC R⟩)
      (fun (ℓ : Fin (2 ^ zkLanes i)) (q : Fin (2 ^ zkLen i)) => oracleOf C ℓ q) R)
    (hrel : Claims.Flock.RelS s (readTrace execArith i.m (zkLanes i) (zkCols i) (padWit (e := 0) padC R))) :
    GoodTable (deployZk rel) (wordLen true) x C := by
  refine ⟨⟨i, zkOfRows padC R⟩, s, hs, ?_, hN.ge, ?_⟩
  · show Claims.Flock.RelS s (zkReadD ⟨i, zkOfRows padC R⟩)
    rw [zkReadD_eq_padWit, zkRowsOf_zkOfRows]
    exact hrel
  · have hf : ∀ q : Fin (2 ^ zkLen i), zkEncD ⟨i, zkOfRows padC R⟩ q =
        List.ofFn fun ℓ => encAt execArith (zkLen i) (2 ^ zkCols i + zkPad i) (R ℓ) q := by
      intro q
      rw [zkEncD_eq]
      refine congrArg List.ofFn (funext fun ℓ => ?_)
      rw [zkEnc_eq_encAt, zkRowsOf_zkOfRows]
    have hc : Close (zkThetaD ⟨i, zkOfRows padC R⟩)
        (fun (ℓ : Fin (2 ^ zkLanes i)) (q : Fin (2 ^ zkLen i)) => oracleOf C ℓ q)
        (fun ℓ q => encAt execArith (zkLen i) (2 ^ zkCols i + zkPad i) (R ℓ) q.val) := by
      unfold Close
      unfold CloseRows at hclose
      convert hclose using 4
    have := agree_of_close hN C hw _ _ hf hc
    show (1 - zkThetaD ⟨i, zkOfRows padC R⟩) * ((2 ^ zkLen i : ℕ) : ℝ) ≤
      ((Finset.univ : Finset (Fin (2 ^ zkLen i))).filter fun q =>
        zkEncD ⟨i, zkOfRows padC R⟩ q.val = extTab C q.val).card
    push_cast
    exact this

/-- **A committed table extracts**, at the masked record: a table of witness-lane columns at a statement whose table 0
is at index `i`, whose oracle is `CommittedPad` for the old statement `S` at the deployed masked schedule and padding,
when a satisfying old witness of `S` reads as a witness of the parsed statement. -/
theorem good_of_committed_zk (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployZk rel).parse x = some s) (i : ZkIdx)
    (hN : wordLen true x = 2 ^ zkLen i) (C : Fin (wordLen true x) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ zkLanes i) (S : Statement)
    (hwit : ∀ P : Message 𝔽 (zkLanes i) (zkCols i),
      S.Satisfies (witnessOf execArith S.m (zkLanes i) (zkCols i) P) →
        Claims.Flock.RelS s (readTrace execArith i.m (zkLanes i) (zkCols i) P))
    (hc : CommittedPad execArith S (schedOf (zkSched i.m).1) (zkPad i) padC (oracleOf C)) :
    GoodTable (deployZk rel) (wordLen true) x C := by
  have key : ∀ (k0 : ℕ) (l : FlockSoundness.Accounting.Level), k0 = zkLanes i → l = levelOf (zkLevel0 i.m) →
      ∀ R ∈ padList execArith l (zkPad i) k0 0 0 (oracleOf C),
        S.Satisfies (witnessOf execArith S.m k0 l.logCols (padWit padC R)) →
      GoodTable (deployZk rel) (wordLen true) x C := by
    intro k0 l hk hl R hR hsat
    subst hk hl
    simp only [padList, Finset.mem_filter, Finset.mem_univ, true_and] at hR
    refine good_of_close_zk rel x s hs i hN C hw R ?_ (hwit _ hsat)
    rw [logLen_zk, ← radius_zk ⟨i, zkOfRows padC R⟩] at hR
    unfold CloseRows at hR ⊢
    convert hR using 5 with q
    simp only [padRows, padLane_zero]
    exact Iff.rfl
  obtain ⟨P, ⟨R, hR, rfl⟩, hsat⟩ := hc
  exact key _ _ (zkLevel_table i).2 (zkLevel_table i).1 R hR hsat

/-- **At table 0's index**, masked: the word is `2^d` long and a column has the `2^lanes` witness lanes. -/
theorem zk_at {x : Claims.Msg} {spec : SessionSpec} {st : _root_.Flock.Stmt}
    (h : (tables true x).head? = some (spec, st)) {i : ZkIdx} (hi : i.m = st.m) :
    wordLen true x = 2 ^ zkLen i ∧ ∀ u, wide true x u ↔ u.length = 2 ^ zkLanes i := by
  refine ⟨?_, fun u => ?_⟩
  · unfold wordLen
    rw [h]
    simp [zkLen, hi]
  · unfold wide zeroCol
    rw [h]
    simp [zkLanes, hi]

/-- **An uncommitted old table**, at the masked record: a table of witness-lane columns that does not extract has an
oracle that is not `CommittedPad` at the deployed masked schedule, for an old statement whose satisfying witnesses read
as witnesses of the parsed statement. -/
theorem not_committed_zk (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployZk rel).parse x = some s) {spec : SessionSpec}
    {st : _root_.Flock.Stmt}
    (hst : (tables true x).head? = some (spec, st)) (i : ZkIdx) (hi : i.m = st.m)
    (C : Fin (wordLen true x) → List 𝔽) (hw : ∀ q, wide true x (C q)) (S : Statement)
    (hwit : ∀ P : Message 𝔽 (zkLanes i) (zkCols i),
      S.Satisfies (witnessOf execArith S.m (zkLanes i) (zkCols i) P) →
        Claims.Flock.RelS s (readTrace execArith i.m (zkLanes i) (zkCols i) P))
    (hg : ¬ GoodTable (deployZk rel) (wordLen true) x C) :
    ¬ CommittedPad execArith S (schedOf (zkSched i.m).1) (zkPad i) padC (oracleOf C) := fun hc =>
  hg (good_of_committed_zk rel x s hs i (zk_at hst hi).1 C (fun q => ((zk_at hst hi).2 _).1 (hw q)) S hwit hc)

end FlockDeploy.KS
