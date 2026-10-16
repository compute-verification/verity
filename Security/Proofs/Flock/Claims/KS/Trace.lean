import Proofs.Flock.Claims.KS.Joint
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.LigeritoPadLevel0

/-!
# The old table at the filled word

`TableBoundθ` (`KS.Joint`) is the old theorems' fixed-table bound at the deployed verifier. The old bound's event is
`¬ Committed A S sch T` at an oracle `T`, the radius form's `¬ GoodTable C` at a word of columns `C`. This file maps one
to the other.

* **The word as the old oracle** (`oracleOf`): lane `ℓ` at position `q` is column `q`'s symbol `ℓ`. When every column
  has the code's width, `Close γ` (every lane agreeing on a `1 − γ` fraction of the positions) bounds the distance per
  whole column by `γ·N` (`hdist_le_of_close`).
* **The plain record**: the deployed code is the old `encode` (`plainEncD_encode`), the read step the bits of the old
  `witnessOf` (`plainReadD_hid`), and the deployed schedule the old model's (`fast100_plainSched`), whose level 0 is
  the deployed one (`level₀_plainSched`) with the radius form's `θ` as its `radius` (`radius_plain`). So a table of
  full-width columns whose oracle is `Committed` at the deployed schedule extracts (`good_of_committed_plain`), given
  that a satisfying old witness reads as a witness of the parsed statement.
* **The masked record**: a masked message's lanes are coefficient rows of the old padded code (`zkRowsOf`,
  `zkEnc_eq_encAt`), the read step reads the old `padWit` of them (`zkReadD_eq_padWit`), and the radius form's `θ` is
  the old `padRadius` (`radius_zk`). A masked column is an opened row's witness lanes, so the old masked bound's
  `CommittedPad` (`padList` with no extra lanes) reads exactly the masked columns.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model FlockSoundness.Accounting FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## The word as the old oracle -/

/-- **A word of columns as the old oracle**: lane `ℓ` at position `q` is column `q`'s symbol `ℓ`; `0` past the word or
the column. -/
def oracleOf {F : Type} [Zero F] {N : ℕ} (C : Fin N → List F) : ℕ → ℕ → F :=
  fun ℓ q => if h : q < N then (C ⟨q, h⟩).getD ℓ 0 else 0

theorem hdist_ofFn {Col : Type} [DecidableEq Col] {N : ℕ} (C : Fin N → Col) (f : ℕ → Col) :
    hdist (List.ofFn C) f = (Finset.univ.filter fun q : Fin N => C q ≠ f q).card := by
  unfold hdist
  rw [List.length_ofFn, ← Finset.card_map Fin.valEmbedding]
  congr 1
  ext q
  simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_map, Finset.mem_univ, true_and,
    Fin.valEmbedding_apply]
  constructor
  · rintro ⟨hq, h⟩
    refine ⟨⟨q, hq⟩, fun he => h ?_, rfl⟩
    simp [hq, he]
  · rintro ⟨q', h, rfl⟩
    refine ⟨q'.isLt, fun he => h ?_⟩
    simpa [List.getElem?_ofFn] using he

/-- **`Close` bounds the distance per whole column**: when every column has the code's `2^lanes` symbols and is the
codeword's column wherever every lane agrees, a `γ`-close oracle's word is within `γ·2^d` of the codeword. -/
theorem hdist_le_of_close {F : Type} [Field F] [DecidableEq F] {lanes d N : ℕ} (hN : N = 2 ^ d) {γ : ℝ}
    (C : Fin N → List F) (hw : ∀ q, (C q).length = 2 ^ lanes) (U : Table F lanes d) (f : ℕ → List F)
    (hf : ∀ q : Fin (2 ^ d), f q = List.ofFn fun ℓ => U ℓ q)
    (h : Close γ (fun (ℓ : Fin (2 ^ lanes)) (q : Fin (2 ^ d)) => oracleOf C ℓ q) U) :
    (hdist (List.ofFn C) f : ℝ) ≤ γ * 2 ^ d := by
  subst hN
  rw [hdist_ofFn]
  have hsub : (Finset.univ.filter fun q : Fin (2 ^ d) => C q ≠ f q) ⊆
      Finset.univ.filter fun q => ¬ ∀ ℓ : Fin (2 ^ lanes), oracleOf C ℓ q = U ℓ q := by
    intro q
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    intro hne hall
    apply hne
    rw [hf q]
    refine List.ext_getElem (by simp [hw q]) fun n h1 h2 => ?_
    have e := hall ⟨n, by rwa [hw q] at h1⟩
    simp only [oracleOf, q.isLt, ↓reduceDIte, Fin.eta, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1,
      Option.getD_some] at e
    simpa using e
  have hc := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin (2 ^ d))))
    (fun q => ∀ ℓ : Fin (2 ^ lanes), oracleOf C ℓ q = U ℓ q)
  rw [Finset.card_univ, Fintype.card_fin] at hc
  have h1 := Finset.card_le_card hsub
  have h' : ((Finset.univ.filter fun q : Fin (2 ^ d) => ∀ ℓ : Fin (2 ^ lanes), oracleOf C ℓ q = U ℓ q).card : ℝ) ≥
      (1 - γ) * 2 ^ d := by
    unfold Close at h
    convert h
  have hc' : ((Finset.univ.filter fun q : Fin (2 ^ d) => ¬ ∀ ℓ : Fin (2 ^ lanes), oracleOf C ℓ q = U ℓ q).card : ℝ) =
      2 ^ d - ((Finset.univ.filter fun q : Fin (2 ^ d) => ∀ ℓ : Fin (2 ^ lanes), oracleOf C ℓ q = U ℓ q).card : ℝ) := by
    have := congrArg (Nat.cast (R := ℝ)) hc
    push_cast at this
    linarith
  calc (((Finset.univ.filter fun q : Fin (2 ^ d) => C q ≠ f q).card : ℕ) : ℝ)
      ≤ ((Finset.univ.filter fun q : Fin (2 ^ d) => ¬ ∀ ℓ : Fin (2 ^ lanes), oracleOf C ℓ q = U ℓ q).card : ℝ) := by
        exact_mod_cast h1
    _ = _ := hc'
    _ ≤ γ * 2 ^ d := by linarith

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

theorem plainReadD_salts (i : PlainIdx) (M : Message 𝔽 (plainLanes i) (plainCols i)) :
    (plainReadD ⟨i, M⟩).salts = [] := rfl

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

/-- **The radius form's plain `θ` is the old level-0 `radius`**, `1 − √(1/2) − 1/200`. -/
theorem radius_plain (i : PlainIdx) :
    FlockDeploy.KS.radiusOf (2 ^ plainCols i) (2 ^ plainLen i) = (levelOf (plainLevel0 i.m)).radius := by
  have hd : plainLen i = plainCols i + 1 := (plainShape_table i).2.2.2.1
  have hr := (plainLevel_table i).1
  unfold FlockDeploy.KS.radiusOf Level.radius Level.rate eta
  have e : (((2 ^ plainCols i : ℕ) : ℝ) / ((2 ^ plainLen i : ℕ) : ℝ)) = (2⁻¹ : ℝ) ^ (levelOf (plainLevel0 i.m)).logInvRate := by
    show _ = (2⁻¹ : ℝ) ^ (plainLevel0 i.m).rate
    rw [hr, hd]
    push_cast
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

/-- **The radius form's masked `θ` is the old `padRadius`** at the deployed padding. -/
theorem radius_zk (i : ZkIdx) :
    FlockDeploy.KS.radiusOf (2 ^ zkCols i + zkPad i) (2 ^ zkLen i) = (levelOf (zkLevel0 i.m)).padRadius (zkPad i) := by
  have hd : zkLen i = zkCols i + 1 := (zkShape_table i).2.2.2.1
  unfold FlockDeploy.KS.radiusOf Level.padRadius Level.padRate eta
  have hl : (levelOf (zkLevel0 i.m)).logLen = zkLen i := by
    show (zkLevel0 i.m).cols + (zkLevel0 i.m).rate = zkLen i
    rw [zkLevel_rate i, hd]
    rfl
  rw [hl]
  push_cast
  rfl

end Claims.Flock

/-! ## A committed table extracts, at the plain record -/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Model FlockSoundness.Refine Claims.Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **A close table extracts**, at the plain record: a table of full-width columns at a statement whose table 0 is at
index `i`, `γ`-close at the radius form's `θ` to the old encoding of a message whose read step is a witness. -/
theorem good_of_close_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployPlain rel).parse x = some s) (i : PlainIdx)
    (hN : plainRadius.N x = 2 ^ plainLen i) (C : Fin (plainRadius.N x) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ plainLanes i) (M : Message 𝔽 (plainLanes i) (plainCols i))
    (hclose : Close (radiusOf (2 ^ plainCols i) (2 ^ plainLen i))
      (fun (ℓ : Fin (2 ^ plainLanes i)) (q : Fin (2 ^ plainLen i)) => oracleOf C ℓ q) (encode execArith (plainLen i) M))
    (hrel : Claims.Flock.RelS H s (plainReadD ⟨i, M⟩)) :
    GoodTable H (deployPlain rel) plainRadius x C := by
  refine ⟨⟨i, M⟩, s, hs, hrel, hN, ?_⟩
  have := hdist_le_of_close hN C hw _ _ (plainEncD_encode i M) hclose
  show (hdist (List.ofFn C) (plainEncD ⟨i, M⟩) : ℝ) ≤ radiusOf (2 ^ plainCols i) (2 ^ plainLen i) *
    ((2 ^ plainLen i : ℕ) : ℝ)
  push_cast
  exact this

/-- **A committed table extracts**, at the plain record: a table of full-width columns at a statement whose table 0 is
at index `i`, whose oracle is `Committed` for the old statement `S` at the deployed schedule, when a satisfying old
witness of `S` reads as a witness of the parsed statement. -/
theorem good_of_committed_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployPlain rel).parse x = some s) (i : PlainIdx)
    (hN : plainRadius.N x = 2 ^ plainLen i) (C : Fin (plainRadius.N x) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ plainLanes i) (S : Statement)
    (hwit : ∀ M : Message 𝔽 (plainLanes i) (plainCols i),
      S.Satisfies (witnessOf execArith S.m (plainLanes i) (plainCols i) M) → Claims.Flock.RelS H s (plainReadD ⟨i, M⟩))
    (hc : Committed execArith S (schedOf (plainSched i.m)) (oracleOf C)) :
    GoodTable H (deployPlain rel) plainRadius x C := by
  have key : ∀ (k0 cols d : ℕ) (γ : ℝ), k0 = plainLanes i → cols = plainCols i → d = plainLen i →
      γ = radiusOf (2 ^ plainCols i) (2 ^ plainLen i) →
      CommittedSatisfies execArith S k0 cols d γ (fun (ℓ : Fin (2 ^ k0)) (q : Fin (2 ^ d)) => oracleOf C ℓ q) →
      GoodTable H (deployPlain rel) plainRadius x C := by
    intro k0 cols d γ hk hcols hd hγ h
    subst hk hcols hd hγ
    obtain ⟨M, hclose, hsat⟩ := h
    exact good_of_close_plain H rel x s hs i hN C hw M hclose (hwit M hsat)
  refine key _ _ _ _ (k0_plainSched i) ?_ ?_ ?_ hc
  · rw [level₀_plainSched]
    rfl
  · rw [level₀_plainSched]
    exact logLen_plain i
  · rw [level₀_plainSched, radius_plain]

/-- **At table 0's index**: the word is `2^d` long and a full-width column has `2^lanes` symbols. -/
theorem plain_at {x : Claims.Msg} {spec : SessionSpec} {st : _root_.Flock.Stmt} (h : (tables false x).head? = some (spec, st))
    {i : PlainIdx} (hi : i.m = st.m) :
    plainRadius.N x = 2 ^ plainLen i ∧ ∀ u, wide false x u ↔ u.length = 2 ^ plainLanes i := by
  refine ⟨?_, fun u => ?_⟩
  · show wordLen false x = _
    unfold wordLen
    rw [h]
    simp [plainLen, hi]
  · unfold wide zeroCol
    rw [h]
    simp [plainLanes, hi]

/-- **An uncommitted old table**, at the plain record: a table of full-width columns that does not extract has an
oracle that is not `Committed` at the deployed schedule, for an old statement whose satisfying witnesses read as
witnesses of the parsed statement. -/
theorem not_committed_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (x : Claims.Msg) (s : Claims.Flock.Stmt) (hs : (deployPlain rel).parse x = some s) {spec : SessionSpec}
    {st : _root_.Flock.Stmt}
    (hst : (tables false x).head? = some (spec, st)) (i : PlainIdx) (hi : i.m = st.m)
    (C : Fin (plainRadius.N x) → List 𝔽) (hw : ∀ q, wide false x (C q)) (S : Statement)
    (hwit : ∀ M : Message 𝔽 (plainLanes i) (plainCols i),
      S.Satisfies (witnessOf execArith S.m (plainLanes i) (plainCols i) M) → Claims.Flock.RelS H s (plainReadD ⟨i, M⟩))
    (hg : ¬ GoodTable H (deployPlain rel) plainRadius x C) :
    ¬ Committed execArith S (schedOf (plainSched i.m)) (oracleOf C) := fun hc =>
  hg (good_of_committed_plain H rel x s hs i (plain_at hst hi).1 C (fun q => ((plain_at hst hi).2 _).1 (hw q)) S hwit hc)

end FlockDeploy.KS
