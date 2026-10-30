import Proofs.Flock.VBridge.Family.Assembly
import Proofs.Flock.VBridge.Family.Lig
import Proofs.Flock.VBridge.Algebra.Sim
import Proofs.Flock.VBridge.Algebra.LigeritoConverse
import Proofs.Flock.VBridge.Family.Names

/-!
# The inner statement's setup, from the hidden statement

The recursion's inner protocol is the compiled table on the model statement `M.toStatement regs` (`flockInnerCL`); no
executable statement `Stmt` stands behind it, so the setup the table files are read at is the model statement's own:

* `modelSetup S ptLocal mPts`: `m` and `k_log` the statement's, the fold `Σ_i e_i·(α·A₀[i, c] + B₀[i, c])` computed
  from `S`'s matrices (`modelFold`), with `β` at the pin and halved at the round challenges as `Stmt.folded`'s fallback
  (Statement.lean:332-336), the extra claims the model's at the link points (`claimOf`), and plain SHA-512 rows (`flock-leaf/sha512-unsalted`);
* `modelSetup_realizes`: it realizes `S` (`SetupRealizes`) at any `fast100` schedule of `S.m`;
* `foldsOk_modelSetup`: its fold returns 64 elements at the round challenges `residualLins` draws (`ts_size`), so
  `FoldsOk` holds;
* `CoinsAgree`: the rounds `verifierRows` takes have `run_of_frames`' counts (`roundSizes_shaped`), and Ligerito's
  frames close at most the `ligCoins` rounds V*'s Ligerito draws after them (`ligRoundsLe_of`, at a schedule of the
  seams);
* at the family, `innerSetup` is `modelSetup` of each history's hidden statement, and `ShapeAt` (V*'s `m`, `k_log` and
  schedule are the class's) gives `ShapeIs`, `SetupRealizes`, `FoldsOk` and `CoinsAgree` there;
* `vbridgeRead_model`: `vbridgeRead_assembled` at `innerSetup`, without `hne`: with no history and registration,
  `VBridgeRead`'s first two conjuncts are vacuous.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

local notation "𝔽" => FlockLevel3.GF128

/-! ## The model statement's setup -/

/-- The fold of `S`'s matrices at `α`, the skip weights `lw` and the block point `ρ`: entry `c` is
`Σ_i e_i·(α·A₀[i, c] + B₀[i, c])` for `e_i = lw[i mod 64]·eq(ρ, ⌊i/64⌋)` (§10, `FoldRealizes.fold`). -/
noncomputable def modelFoldV (S : Model.Statement) (α : F128) (lw ρ : Array F128) : Array F128 :=
  Array.ofFn fun col : Fin (2 ^ S.kLog) => toF128 (∑ i : Fin (2 ^ S.kLog),
    fOf lw[i.val % 64]! * Model.eqAt (vecOf (S.kLog - 6) ρ) (i.val / 64) *
      (fOf α * ((S.A₀ i col).val : 𝔽) + ((S.B₀ i col).val : 𝔽)))

/-- The model statement's fold: `modelFoldV`, and `folded` it with `β` at the pin, halved at `ts` in turn. -/
noncomputable def modelFold (S : Model.Statement) : CircuitFold where
  kLog := S.kLog
  pin := S.pin.val
  fold α lw ρ := pure (modelFoldV S α lw ρ)
  folded α β lw ρ ts := pure (ts.foldl (fun acc t => halve t acc) ((modelFoldV S α lw ρ).modify S.pin.val (· + β)))

/-- A model claim as the executable's. -/
def claimOf {m : ℕ} (c : Model.Claim 𝔽 m) : Flock.Claim :=
  ⟨toF128 c.value, Array.ofFn fun s => toF128 (c.skip s), Array.ofFn fun i => toF128 (c.x i)⟩

/-- **The model statement's setup**: `S`'s `m`, `k_log` and fold, its extra claims at the link points the model's
(`Model.extraClaims`, read through `ptsOf`), its rows plain SHA-512's
(`flock-leaf/sha512-unsalted`, the inner statement's leaf). -/
noncomputable def modelSetup (S : Model.Statement) (ptLocal mPts : ℕ) : Setup where
  m := S.m
  kLog := S.kLog
  digest := ByteArray.empty
  fold := modelFold S
  extra P := pure ((FlockSoundness.Model.extraClaims S ptLocal mPts (FlockSoundness.Refine.ptsOf mPts P)).toArray.map
    claimOf)
  publics _ := pure ()
  merkle := MerkleScheme.sha512 "flock-leaf/sha512-unsalted"
  openedSalts := true

theorem modelFold_realizes (S : Model.Statement) : FoldRealizes (modelFold S) S where
  kLog := rfl
  pin := rfl
  fold := by
    intro α lw ρ v _ _ hv col
    change Except.ok (modelFoldV S α lw ρ) = Except.ok v at hv
    cases hv
    have hc : col.val < (modelFoldV S α lw ρ).size := by
      simp only [modelFoldV, Array.size_ofFn]
      exact col.isLt
    rw [getElem!_pos _ col.val hc]
    simp only [modelFoldV, Array.getElem_ofFn]
    rfl
  folded := by
    intro α β lw ρ ts w hw
    change Except.ok _ = Except.ok w at hw
    cases hw
    exact ⟨modelFoldV S α lw ρ, rfl, Array.size_ofFn, rfl⟩

theorem claimRel_claimOf {m : ℕ} (c : Model.Claim 𝔽 m) : ClaimRel m (claimOf c) c where
  value := rfl
  skip_size := Array.size_ofFn
  skip s := by
    show c.skip s = fOf (Array.ofFn fun s => toF128 (c.skip s))[s.val]!
    rw [getElem!_pos _ s.val (by simp only [Array.size_ofFn]; exact s.isLt), Array.getElem_ofFn]
    rfl
  x_size := Array.size_ofFn
  x i := by
    show c.x i = fOf (Array.ofFn fun i => toF128 (c.x i))[i.val]!
    rw [getElem!_pos _ i.val (by simp only [Array.size_ofFn]; exact i.isLt), Array.getElem_ofFn]
    rfl

/-- The link points `pts` as the verifier's arrays. -/
noncomputable def pointsArr {mPts : ℕ} (pts : Fin 2 → Fin mPts → 𝔽) : Array (Array F128) :=
  Array.ofFn fun k : Fin 2 => Array.ofFn fun t : Fin mPts => toF128 (pts k t)

theorem ptsOf_pointsArr {mPts : ℕ} (pts : Fin 2 → Fin mPts → 𝔽) :
    FlockSoundness.Refine.ptsOf mPts (pointsArr pts) = pts := by
  funext k t
  simp only [FlockSoundness.Refine.ptsOf, pointsArr]
  rw [getElem!_pos _ k.val (by simp only [Array.size_ofFn]; exact k.isLt), Array.getElem_ofFn,
    getElem!_pos _ t.val (by simp only [Array.size_ofFn]; exact t.isLt), Array.getElem_ofFn]
  rfl

/-- **The model setup's extra claims realize the model's** at every link points, at their arrays `pointsArr`. -/
theorem modelSetup_points (S : Model.Statement) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → 𝔽) :
    PointsOK (modelSetup S ptLocal mPts) S ptLocal mPts pts (pointsArr pts) := by
  have he : (modelSetup S ptLocal mPts).extra (pointsArr pts) =
      .ok ((FlockSoundness.Model.extraClaims S ptLocal mPts pts).toArray.map claimOf) := by
    change Except.ok _ = _
    rw [ptsOf_pointsArr]
  refine ⟨_, he, fun extras hx => ?_, fun j hj => ?_⟩
  · rw [he] at hx
    cases hx
    refine ⟨by simp, fun j hj => ?_⟩
    rw [getElem!_pos _ j (by simpa using hj), Array.getElem_map, List.getElem_toArray]
    exact claimRel_claimOf _
  · rw [getElem!_pos _ j hj, Array.getElem_map]
    exact ⟨Array.size_ofFn, Array.size_ofFn⟩

/-- **The model setup realizes its statement** (`SetupRealizes`) at a `fast100` schedule of its `m`, whose range gives
`14 ≤ m`. -/
theorem modelSetup_realizes (S : Model.Statement) (ptLocal mPts : ℕ) {sch : Flock.Schedule}
    (hf : Flock.fast100 S.m = some sch) : SetupRealizes (modelSetup S ptLocal mPts) S sch ptLocal mPts := by
  refine ⟨rfl, rfl, modelFold_realizes S, rfl, hf, ?_, S.hk, rfl, fun pts => ⟨_, modelSetup_points S ptLocal mPts pts⟩⟩
  show 14 ≤ S.m
  have := (fast100_range hf).1
  omega

/-- The lincheck's round challenges number `k_log − 6` (`lincheck_coins`, over `symbolic`'s `k_log − 6` rounds). -/
theorem ts_size {sh : Shape} {src : Flock.RecAlgebra.CoinSource} {extras : Array Flock.Claim}
    {lins : Array Flock.RecAlgebra.Lin} {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin}
    (h : Flock.RecAlgebra.residualLins sh src extras = .ok (lins, info)) : info.lc.fold.ts.size = sh.kLog - 6 := by
  obtain ⟨j, hj⟩ := Flock.RecAlgebra.residualLins_run h
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, -, h1, -⟩ := Flock.RecAlgebra.run_stages hj
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hts, -⟩ := Flock.RecAlgebra.lincheck_coins h1
  rw [hts]
  simp [Flock.RecAlgebra.symbolic]

/-- **`FoldsOk` at the model setup**: its `folded` always returns, `2^k_log` entries halved at the `k_log − 6` round
challenges `residualLins` draws (`ts_size`), so 64 of them. -/
theorem foldsOk_modelSetup (S : Model.Statement) (ptLocal mPts : ℕ) {sh : Shape} (hk : sh.kLog = S.kLog) :
    FoldsOk (modelSetup S ptLocal mPts) sh := by
  intro rounds extras lins info _ h
  refine ⟨info.lc.fold.ts.foldl (fun acc t => halve t acc)
    ((modelFoldV S info.lc.fold.alpha info.lc.fold.lw info.lc.fold.rhoIn).modify S.pin.val (· + info.lc.fold.beta)),
    rfl, ?_⟩
  have h6 := S.hk6
  obtain ⟨d, hd⟩ : ∃ d, S.kLog = d + 6 := ⟨S.kLog - 6, by omega⟩
  rw [← Array.foldl_toList, FlockLevel3.foldl_halve_size, Array.size_modify, modelFoldV, Array.size_ofFn,
    Array.length_toList, ts_size h, hk, hd, Nat.add_sub_cancel, pow_add, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  rfl

/-! ## The round sizes `verifierRows` takes -/

theorem round_size {rounds : Array (Array F128)} {k c : ℕ} {v : Array F128} (hne : Flock.RecAlgebra.NoEmpty rounds)
    (h : (Flock.RecAlgebra.CoinSource.ofRounds rounds).round k c = .ok v) : k < rounds.size ∧ (rounds[k]!).size = c := by
  obtain ⟨h1, h2⟩ := Flock.RecAlgebra.ofRounds_get? hne h
  obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.mp h1
  exact ⟨hk, by rw [getElem!_pos rounds k hk]; exact h2⟩

theorem size_at {rounds : Array (Array F128)} {a b c : ℕ} (h : a = b) (hs : (rounds[a]!).size = c) :
    (rounds[b]!).size = c := h ▸ hs

/-- **`verifierRows`' rounds have `run_of_frames`' counts** up to Ligerito's: at the rounds `verifierRows` takes
(`roundSizes`), `run` draws the zerocheck's `6`, `m − 13`, `1` and `m − 6` single coins (`zerocheck_coins`), the
lincheck's `α`, `β`, `k_log − 6` round challenges and `r'` (`lincheck_coins`), `7` per claim and the `claims` batching
coins (`ringSwitch_spec`), and Ligerito's `L` rounds after them. -/
theorem roundSizes_shaped {sh : Shape} {S : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf sh = .ok S)
    {n : ℕ} (hcl : sh.claims = 2 + n) (hm : 14 ≤ sh.m) :
    ∃ L, CoinsShaped sh.m sh.kLog n L (Flock.RecAlgebra.zeroRounds (roundSizes sh)) ∧
      (roundSizes sh).length = 3 + (sh.m - 6) + 2 + (sh.kLog - 6) + 1 + (2 + n) + 1 + L := by
  have hsz : (Flock.RecAlgebra.zeroRounds (roundSizes sh)).size = (roundSizes sh).length := by
    simp [Flock.RecAlgebra.zeroRounds]
  have hneZ := Flock.RecAlgebra.noEmpty_zeroRounds (roundSizes_ne hS)
  obtain ⟨x, hx⟩ := (roundSizes_spec hS hneZ (Flock.RecAlgebra.extrasLike sh) #[]).2
    ⟨zeroRounds_sizes _, by simp [Flock.RecAlgebra.extrasLike]; omega⟩
  obtain ⟨v, info, kappa⟩ := x
  obtain ⟨lins, hl⟩ := residualLins_of_verifierRows hx
  obtain ⟨j, hj⟩ := Flock.RecAlgebra.residualLins_run hl
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, h0, h1, h2, -, -⟩ := Flock.RecAlgebra.run_stages hj
  obtain ⟨c₀, c₁, c₂, e0, e1, e2, -, -, -, hk1, ezc⟩ := Flock.RecAlgebra.zerocheck_coins hm h0
  obtain ⟨cα, cβ, cr, eα, eβ, er, -, -, -, -, -, -, -, -, hk2, elc⟩ := Flock.RecAlgebra.lincheck_coins h1
  have hcs : (Flock.RecAlgebra.claimsOf sh.m sh.kLog info.zc info.lc (Flock.RecAlgebra.extrasLike sh)).size =
      2 + n := by
    simp [Flock.RecAlgebra.claimsOf, Flock.RecAlgebra.extrasLike]; omega
  obtain ⟨_, -, -, eg, hk3, -, -, ers⟩ := Flock.RecAlgebra.ringSwitch_spec (by omega) h2
  have hzc : (Flock.RecAlgebra.symbolic sh).zc.size = sh.m - 6 := by simp [Flock.RecAlgebra.symbolic]
  have hlc : (Flock.RecAlgebra.symbolic sh).lc.size = sh.kLog - 6 := by simp [Flock.RecAlgebra.symbolic]
  rw [hzc] at hk1 ezc
  rw [hlc] at hk2 elc er
  rw [hcs] at eg hk3
  have hlt := (round_size hneZ eg).1
  refine ⟨(Flock.RecAlgebra.zeroRounds (roundSizes sh)).size - k3,
    ⟨by omega, size_at (by omega) (round_size hneZ e0).2, size_at (by omega) (round_size hneZ e1).2,
      size_at (by omega) (round_size hneZ e2).2, fun i hi => ?_, fun i hi => ?_, fun i hi => ?_,
      size_at (by omega) (round_size hneZ eg).2⟩, by omega⟩
  · obtain ⟨c, hc, -⟩ := ezc i hi
    exact size_at (by omega) (round_size hneZ hc).2
  · by_cases hi0 : i = 0
    · exact size_at (by omega) (round_size hneZ eα).2
    by_cases hi1 : i = 1
    · exact size_at (by omega) (round_size hneZ eβ).2
    by_cases hil : i = 2 + (sh.kLog - 6)
    · exact size_at (by omega) (round_size hneZ er).2
    obtain ⟨c, hc, -⟩ := elc (i - 2) (by omega)
    exact size_at (by omega) (round_size hneZ hc).2
  · obtain ⟨c, hc, -⟩ := ers i (by omega)
    exact size_at (by omega) (round_size hneZ hc).2

/-- **Ligerito's rounds are at most those `verifierRows` takes after ring switching**: the rounds a Ligerito run's
frames close (`chunks (ligFrames …)`, `.sq` frames of the schedule's counts only, FramesLigerito.lean:130-141, 207-219)
are at most the rounds `run` draws after the ring switch's (`roundSizes`, RecAlgebra.lean:532-539). -/
def LigRoundsLe (sh : Shape) (sch : Flock.Schedule) : Prop :=
  ∀ (T : F128) (cap : Array ByteArray) (lig : LigeritoProof),
    3 + (sh.m - 6) + 2 + (sh.kLog - 6) + 1 + sh.claims + 1 + (chunks (ligFrames sch T cap lig)).1.length ≤
      (roundSizes sh).length

/-- The rounds level `i` of `R` draws after level 0 (RecAlgebra.lean:415–432): four challenges, then two intros and a
query fold below the last level, the queries and the batching coin at it. -/
def levelCoins (R i : ℕ) : ℕ := if i + 1 < R then 11 else 7

/-- The rounds Ligerito draws up to its final check (RecAlgebra.lean:436–450): the level-0 point and its batching
coin, `k0` challenges, two intros, the query fold, then the levels'. -/
def ligCoins (sch : Flock.Schedule) : ℕ :=
  sch.k0 + 9 + ((List.range' 0 (sch.levels.size - 1)).map (levelCoins (sch.levels.size - 1))).sum

theorem levelsLoop_k {c : Array (Array F128)} [Fact (Flock.RecAlgebra.NoEmpty c)] {M : Flock.RecAlgebra.Messages F128}
    {sch : Flock.Schedule} {R : ℕ} (hq : ∀ i ≤ R, (sch.levels.getD i default).q ≠ 0) :
    ∀ (fuel i split : ℕ) (levels : Array (Array F256)) (e : ℕ) (st : Flock.RecAlgebra.LSt F128) (k : ℕ)
      (out : Array (Array F256) × ℕ) (st' : Flock.RecAlgebra.LSt F128) (k' : ℕ), i + fuel ≤ R →
      Flock.RecAlgebra.levelsLoop sch M R fuel i split levels e st k (Flock.RecAlgebra.CoinSource.ofRounds c) =
        .ok ((out, st'), k') →
      k' = k + ((List.range' i fuel).map (levelCoins R)).sum
  | 0, _, _, _, _, st, k, out, st', k', _, h => by
    simp only [Flock.RecAlgebra.levelsLoop] at h
    obtain ⟨-, -, rfl⟩ := Flock.RecAlgebra.vpure h
    simp
  | fuel + 1, i, split, levels, e, st, k, out, st', k', hi, h => by
    simp only [Flock.RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (Flock.RecAlgebra.vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := Flock.RecAlgebra.vbind_ok h
    obtain ⟨e₁, -⟩ := Flock.RecAlgebra.challenges_conv (p := default) 4 0 #[] st st₁ k k₁ chs h₁
    try dsimp only at h
    rw [List.range'_succ, List.map_cons, List.sum_cons]
    by_cases hi' : i + 1 < R
    · rw [ite_eq_left hi'] at h
      obtain ⟨u₂, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨u₃, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨u₄, st₄, k₄, h₄, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨e₂, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
      obtain ⟨e₃, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
      obtain ⟨e₄, -⟩ := Flock.RecAlgebra.vqueryFold (hq (i + 1) (by omega)) h₄
      have e₅ := levelsLoop_k hq fuel (i + 1) _ _ _ st₄ k₄ out st' k' (by omega) h
      simp only [levelCoins, hi', ↓reduceIte]
      omega
    · rw [ite_eq_right hi'] at h
      obtain ⟨r, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨rfl, rfl, -, -, rfl⟩ := Flock.RecAlgebra.vquery (hq R le_rfl) h₂
      try dsimp only at h
      obtain ⟨-, h⟩ := Flock.RecAlgebra.vcoinOne_bind h
      obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨-, rfl⟩ := Flock.RecAlgebra.vmodify_ok h₃
      have e₅ := levelsLoop_k hq fuel (i + 1) _ _ _ st₃ _ out st' k' (by omega) h
      simp only [levelCoins, hi', ↓reduceIte]
      omega

/-- **Ligerito up to its final check draws `ligCoins` rounds** at a schedule of the seams (`DimSeam`, `LevelSeam`). -/
theorem ligeritoRun_k {c : Array (Array F128)} [Fact (Flock.RecAlgebra.NoEmpty c)] {sch : Flock.Schedule}
    {M : Flock.RecAlgebra.Messages F128} {st₀ : Flock.RecAlgebra.LSt F128} {k₀ k' : ℕ}
    {out : Array F256 × Array (Array F256) × ℕ}
    {stV : Flock.RecAlgebra.LSt F128} (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hrun : (Flock.RecAlgebra.ligeritoRun sch M).run st₀ k₀ (Flock.RecAlgebra.CoinSource.ofRounds c) =
      .ok ((out, stV), k')) :
    k' = k₀ + ligCoins sch := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hq : ∀ i ≤ sch.levels.size - 1, (sch.levels.getD i default).q ≠ 0 :=
    fun i hi => (hlv i (by omega)).q_ne_zero
  rw [StateT.run] at hrun
  unfold Flock.RecAlgebra.ligeritoRun at hrun
  obtain ⟨-, h⟩ := Flock.RecAlgebra.vcoinVec_bind (by omega) hrun
  obtain ⟨y0, -, h⟩ := Flock.RecAlgebra.vnextOod_bind h
  obtain ⟨-, h⟩ := Flock.RecAlgebra.vcoinOne_bind h
  obtain ⟨⟨⟩, stZ, kZ, hZ, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨-, rfl⟩ := Flock.RecAlgebra.vmodify_ok hZ
  obtain ⟨m0, -, h⟩ := Flock.RecAlgebra.vnextMsg_bind h
  obtain ⟨⟨⟩, stA, kA, hA, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨-, rfl⟩ := Flock.RecAlgebra.vmodify_ok hA
  obtain ⟨ic, st₁, k₁, h₁, h⟩ := Flock.RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₄, k₄, h₄, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨lv', ex'⟩, st₅, k₅, h₅, h⟩ := Flock.RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨-, -, rfl⟩ := Flock.RecAlgebra.vpure h
  obtain ⟨e₁, -⟩ := Flock.RecAlgebra.challenges_conv (p := default) sch.k0 0 #[] _ _ _ _ _ h₁
  obtain ⟨e₂, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₂
  obtain ⟨e₃, -⟩ := Flock.RecAlgebra.oodIntro_conv (p := default) (by omega) h₃
  obtain ⟨e₄, -⟩ := Flock.RecAlgebra.vqueryFold (hlv 0 (by omega)).q_ne_zero h₄
  have e₅ := levelsLoop_k hq _ _ _ _ _ st₄ _ _ _ _ (by omega) h₅
  unfold ligCoins
  omega

/-- `ligerito` returns only after Ligerito up to its final check, having drawn every recorded round
(RecAlgebra.lean:511–513). -/
theorem ligerito_drawn {sch : Flock.Schedule} {M : Flock.RecAlgebra.Messages F128} {tIn : F128}
    {sws : Array Flock.Switched} {k k' : ℕ} {src : Flock.RecAlgebra.CoinSource}
    {r : Array F128 × Flock.RecAlgebra.LgPts} (h : Flock.RecAlgebra.ligerito sch M tIn sws k src = .ok (r, k')) :
    ∃ st₀ out stV k₁, (Flock.RecAlgebra.ligeritoRun sch M).run st₀ k src = .ok ((out, stV), k₁) ∧
      src.drawnAll k₁ = true := by
  unfold Flock.RecAlgebra.ligerito at h
  obtain ⟨⟨out, stV⟩, k₁, h1, h⟩ := Flock.RecAlgebra.cm_bind h
  obtain ⟨a, k₂, ha, h⟩ := Flock.RecAlgebra.cm_bind h
  cases ha
  obtain ⟨b, k₃, hb, h⟩ := Flock.RecAlgebra.cm_bind h
  cases hb
  by_cases cd : src.drawnAll k₁ = true
  · exact ⟨_, out, stV, k₁, h1, cd⟩
  · rw [ite_eq_left (by simpa using cd)] at h
    obtain ⟨_, _, h0, _⟩ := Flock.RecAlgebra.cm_bind h
    cases h0

theorem chunks_length : ∀ fs : List Frame, (chunks fs).1.length = fs.countP closes
  | [] => rfl
  | f :: fs => by
    rw [List.countP_cons, ← chunks_length fs, chunks]
    by_cases hf : closes f = true
    · simp only [hf, ite_true, List.length_cons]
    · simp only [hf, Bool.false_eq_true, ite_false, Nat.add_zero]
      cases (chunks fs).1 <;> rfl

theorem countP_kFr (m : F256 × F256) : (kFr m).countP closes = 0 := rfl

theorem closes_one (v : F128) : closes (.one v) = false := rfl
theorem closes_bytes (b : ByteArray) : closes (.bytes b) = false := rfl
theorem closes_label (b : ByteArray) : closes (.label b) = false := rfl
theorem closes_capFr (cap : Array ByteArray) : closes (capFr cap) = false := rfl
theorem closes_sq11 : closes (.sq 1 1) = true := rfl
theorem closes_sq22 : closes (.sq 2 2) = true := rfl

theorem countP_ones (l : List F128) : (l.map Frame.one).countP closes = 0 := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.map_cons, List.countP_cons, ih]; rfl

theorem countP_chFr (p : LigeritoProof) : ∀ k msg, (chFr p msg k).countP closes = k
  | 0, _ => rfl
  | k + 1, msg => by
    simp only [chFr, List.countP_cons, List.countP_append, countP_kFr, countP_chFr p k (msg + 1), closes_sq22,
      ↓reduceIte]
    omega

theorem countP_introFr (p : LigeritoProof) (dim ood msg cbn : ℕ) :
    (introFr p dim ood msg cbn).countP closes ≤ 2 := by
  cases dim <;> exact Nat.le_of_ble_eq_true rfl

theorem countP_queryFr (p : LigeritoProof) (lv : Flock.Level) (qn cn : ℕ) :
    (queryFr p lv qn cn).countP closes ≤ 2 := by
  unfold queryFr
  generalize lv.q = q
  cases q <;> exact Nat.le_of_ble_eq_true rfl

theorem countP_levelFr (sch : Flock.Schedule) (p : LigeritoProof) (i : ℕ) :
    (levelFr sch p i).countP closes ≤ levelCoins (sch.levels.size - 1) i := by
  unfold levelFr levelCoins
  by_cases hi : i + 1 < sch.levels.size - 1
  · have := countP_introFr p ((sch.levels.getD (i + 1) default).cols + 1) (3 + 2 * i) (sch.k0 + 8 + 7 * i) (4 + 3 * i)
    have := countP_introFr p ((sch.levels.getD (i + 1) default).cols + 1) (4 + 2 * i) (sch.k0 + 9 + 7 * i) (5 + 3 * i)
    have := countP_queryFr p (sch.levels.getD (i + 1) default) (1 + i) (1 + i)
    simp only [hi, ↓reduceIte, List.countP_append, List.countP_cons, List.countP_nil, countP_chFr, countP_kFr,
      closes_capFr, closes_bytes, closes_sq11, Bool.false_eq_true]
    omega
  · have := countP_queryFr p (sch.levels.getD (sch.levels.size - 1) default) (1 + i) (1 + i)
    simp only [hi, ↓reduceIte, List.countP_append, List.countP_cons, List.countP_nil, countP_chFr, countP_ones,
      closes_bytes, closes_sq11, Bool.false_eq_true]
    omega

theorem countP_levels (sch : Flock.Schedule) (p : LigeritoProof) :
    ∀ a n, ((List.range' a n).flatMap (levelFr sch p)).countP closes ≤
      ((List.range' a n).map (levelCoins (sch.levels.size - 1))).sum
  | a, 0 => by simp
  | a, n + 1 => by
    rw [List.range'_succ, List.flatMap_cons, List.countP_append, List.map_cons, List.sum_cons]
    have := countP_levelFr sch p a
    have := countP_levels sch p (a + 1) n
    omega

/-- **A Ligerito run's frames close at most `ligCoins` rounds**: every `.sq` frame of `ligFrames` but the batching
coins' is a draw of `ligeritoRun`'s, and none closes without one (FramesLigerito.lean:130–141, 207–228). -/
theorem chunks_ligFrames_le (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof) :
    (chunks (ligFrames sch T cap p)).1.length ≤ ligCoins sch := by
  rw [chunks_length]
  have := countP_introFr p (sch.n - sch.k0 + 1) 1 (sch.k0 + 1) 1
  have := countP_introFr p (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2
  have := countP_queryFr p (sch.levels.getD 0 default) 0 0
  have := countP_levels sch p 0 (sch.levels.size - 1)
  unfold ligFrames ligCoins
  simp only [List.countP_append, List.countP_cons, List.countP_nil, countP_chFr, countP_kFr, closes_capFr,
    closes_bytes, closes_label, closes_one, closes_sq11, Bool.false_eq_true, ↓reduceIte]
  split <;> omega

/-- **`LigRoundsLe` at a schedule of the seams**: the rounds `verifierRows` takes are the zerocheck's, the
lincheck's, ring switching's and then exactly `ligCoins` (`ligeritoRun_k`, at the evaluated messages `ligerito_sim`
gives, and `drawnAll`), and Ligerito's frames close at most that many (`chunks_ligFrames_le`). -/
theorem ligRoundsLe_of {sh : Shape} {S : Flock.RecAlgebra.GfS} {sch : Flock.Schedule} {n : ℕ}
    (hS : Flock.RecAlgebra.structureOf sh = .ok S) (hcl : sh.claims = 2 + n) (hm : 14 ≤ sh.m) (hsch : sh.sched = sch)
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) :
    LigRoundsLe sh sch := by
  intro T cap lig
  have hsz : (Flock.RecAlgebra.zeroRounds (roundSizes sh)).size = (roundSizes sh).length := by
    simp [Flock.RecAlgebra.zeroRounds]
  have hneZ := Flock.RecAlgebra.noEmpty_zeroRounds (roundSizes_ne hS)
  obtain ⟨x, hx⟩ := (roundSizes_spec hS hneZ (Flock.RecAlgebra.extrasLike sh) #[]).2
    ⟨zeroRounds_sizes _, by simp [Flock.RecAlgebra.extrasLike]; omega⟩
  obtain ⟨v, info, kappa⟩ := x
  obtain ⟨lins, hl⟩ := residualLins_of_verifierRows hx
  obtain ⟨j, hj⟩ := Flock.RecAlgebra.residualLins_run hl
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, h0, h1, h2, h3, -⟩ := Flock.RecAlgebra.run_stages hj
  obtain ⟨-, -, -, -, -, -, -, -, -, hk1, -⟩ := Flock.RecAlgebra.zerocheck_coins hm h0
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, hk2, -⟩ := Flock.RecAlgebra.lincheck_coins h1
  have hcs : (Flock.RecAlgebra.claimsOf sh.m sh.kLog info.zc info.lc (Flock.RecAlgebra.extrasLike sh)).size =
      2 + n := by
    simp [Flock.RecAlgebra.claimsOf, Flock.RecAlgebra.extrasLike]; omega
  obtain ⟨_, -, -, -, hk3, -⟩ := Flock.RecAlgebra.ringSwitch_spec (by omega) h2
  have hzc : (Flock.RecAlgebra.symbolic sh).zc.size = sh.m - 6 := by simp [Flock.RecAlgebra.symbolic]
  have hlc : (Flock.RecAlgebra.symbolic sh).lc.size = sh.kLog - 6 := by simp [Flock.RecAlgebra.symbolic]
  rw [hzc] at hk1
  rw [hlc] at hk2
  rw [hcs] at hk3
  have h3' := Flock.RecAlgebra.ligerito_sim (Flock.RecAlgebra.Lin.eval_hom fun _ => 0) _ _ _ _ _ _ _ _ h3
  obtain ⟨st₀, out, stV, k₄, hrun, hdr⟩ := ligerito_drawn h3'
  rw [hsch] at hrun
  have : Fact (Flock.RecAlgebra.NoEmpty (Flock.RecAlgebra.zeroRounds (roundSizes sh))) := ⟨hneZ⟩
  have hk := ligeritoRun_k hdim hlv hrun
  have hd : k₄ = (Flock.RecAlgebra.zeroRounds (roundSizes sh)).size := by
    have e : k₄ = ((Flock.RecAlgebra.zeroRounds (roundSizes sh)).filter (·.size ≠ 0)).size := beq_iff_eq.mp hdr
    rwa [Flock.RecAlgebra.filter_noEmpty hneZ] at e
  have := chunks_ligFrames_le sch T cap lig
  omega

/-- **`CoinsAgree`** at a setup of the shape's `m` and fold `k_log`, from `roundSizes_shaped` and `LigRoundsLe`. -/
theorem coinsAgree_of {st : Setup} {sh : Shape} {sch : Flock.Schedule} {n : ℕ} {S : Flock.RecAlgebra.GfS}
    (hS : Flock.RecAlgebra.structureOf sh = .ok S) (hcl : sh.claims = 2 + n) (hm : st.m = sh.m)
    (hk : st.fold.kLog = sh.kLog) (h14 : 14 ≤ sh.m) (hlig : LigRoundsLe sh sch) : CoinsAgree st sh sch n := by
  obtain ⟨L, hL, hlen⟩ := roundSizes_shaped hS hcl h14
  refine ⟨L, by rw [hm, hk]; exact hL, fun T cap lig => ?_, roundSizes_ne hS⟩
  have := hlig T cap lig
  omega

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **The inner setup at each history**: the model setup of the hidden statement `x R ω`. -/
noncomputable def innerSetup (R : Reg) (ω : L.Ω) : Setup :=
  modelSetup ((F.x R ω).toStatement regs) cls.ptLocal cls.mPts

/-- Obligation: discharged by a verifier check of V*'s shape against the inner statement (`vsh`'s `m`, `k_log` and
schedule).
**V*'s shape is the class's** (the verifier's gap `vsh`, with `ClaimsAt`): `vsh`'s `m` and `k_log` are the hidden
class's (`rec_vstage.shape`), and the inner session's schedule is `fast100`'s at that `m`. -/
def ShapeAt : Prop := F.vsh.m = cls.m ∧ F.vsh.kLog = cls.kLog ∧ Flock.fast100 cls.m = some sch

theorem shapeIs_of (h : F.ShapeAt) (R : Reg) (ω : L.Ω) : ShapeIs F.sh (F.innerSetup R ω) sch := by
  obtain ⟨hf, hm, hk, -⟩ := shape?_eq F.hsh
  obtain ⟨hvm, hvk, hs⟩ := h
  refine ⟨hm.trans hvm, hk.trans hvk, ?_⟩
  rw [hvm, hs] at hf
  exact (Option.some.inj hf).symm

theorem setupRealizes_of (h : F.ShapeAt) (R : Reg) (ω : L.Ω) :
    SetupRealizes (F.innerSetup R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts :=
  modelSetup_realizes _ _ _ h.2.2

theorem foldsOk_of (h : F.ShapeAt) (R : Reg) (ω : L.Ω) : FoldsOk (F.innerSetup R ω) F.sh :=
  foldsOk_modelSetup _ _ _ ((F.shapeIs_of h R ω).2.1)

theorem coinsAgree_at (h : F.ShapeAt) {GS : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf F.sh = .ok GS)
    (hcl : F.ClaimsAt) (R : Reg) (ω : L.Ω) :
    CoinsAgree (F.innerSetup R ω) F.sh sch (2 * ((F.x R ω).toStatement regs).regions.length) := by
  obtain ⟨hf, hm, -⟩ := shape?_eq F.hsh
  have h22 := (fast100_range hf).1
  have hs := h.2.2
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level _ _ hs _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hshp := F.shapeIs_of h R ω
  exact coinsAgree_of hS (F.claimsIs_of hcl R ω) hshp.1.symm hshp.2.1.symm (by omega)
    (ligRoundsLe_of hS (F.claimsIs_of hcl R ω) (by omega) hshp.2.2 (fast100_dims _ _ hs).1 hlv)

end VStarFamily

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)
    (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)

/-- **VBridge G's read conjunct at the model setup**: `vbridgeRead_assembled` at `innerSetup`, whose `SetupRealizes`,
`ShapeIs`, `FoldsOk` and `CoinsAgree` follow from `ShapeAt`, and `HOpenName` and `HUnitName` from the circuit files'
emit/parse round trip (`hOpenName_of`, `hUnitName_of`). Without a history and registration
(`¬ Nonempty (Reg × L.Ω × Reg₂)`), `VBridgeRead`'s first two conjuncts quantify over none and its last two are the
lanes' (`vDecodesRead_of`, `vFullRead_of`), so `hne` is not needed. -/
theorem vbridgeRead_model
    (hshape : F.ShapeAt) (hcount : F.HPartCount)
    (hv : (F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).HVCoins dg N k Rw)
    (hlay : F.OneInLayer dg N k Rw) (hrr : F.RowsRegistered dg N k Rw) (hrck : F.RowColKeys dg N k Rw)
    (hsame : F.SameRow dg N k Rw) (hseg : F.ReadsUnsegmented dg N k Rw) (hz : F.HZeroL dg N k Rw)
    (hcoef : (F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).HCoefCols dg N k Rw)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V)
    (hE : (F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).HEnds dg N k Rw V) (hcl : F.ClaimsAt)
    (hcapS : F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (F.innerSetup R ω) sch (cap l))
    (hligS : F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (F.innerSetup R ω) sch (lig l r))
    (hregs : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48)
    (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw)
    (hread : F.MsgsRead dg N k Rw spec F.innerSetup cap lig own sel) (hT : TysIn T sch.ext)
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (F.innerSetup R ω) sch F.sh
        (((F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).tableSrc spec F.innerSetup cap lig R ω t l).proofAt
          r)
        ((F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).rep R ω t l r))
    (hpub : (F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).PubCommits bc) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x
      (wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r (F.decOf dg N k Rw spec F.innerSetup cap lig own sel)) (skOf F.Vs own sel)
      (cmtOf F.Vs bc) (msgOf F.Vs (skOf F.Vs own sel) (F.decOf dg N k Rw spec F.innerSetup cap lig own sel)) := by
  rcases isEmpty_or_nonempty (Reg × L.Ω × Reg₂) with he | hne
  · refine ⟨?_, ?_, (F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).vDecodesRead_of
        (F.decOf dg N k Rw spec F.innerSetup cap lig own sel) rfl rfl,
      (F.assembled dg N k Rw spec F.innerSetup cap lig own sel bc).vFullRead_of⟩
    · intro R ω _ R₂
      exact (he.false (R, ω, R₂)).elim
    · intro _ R ω _ _ _ R₂
      exact (he.false (R, ω, R₂)).elim
  · obtain ⟨hsome, -, -, -⟩ := F.structure_of hne (F.hOpenName_of hne) hcount
    obtain ⟨GS, gs, hst⟩ := F.hStruct_of hne hsome (F.hUnitName_of hne)
    exact F.vbridgeRead_assembled dg N k Rw spec F.innerSetup cap lig own sel bc hne (F.hOpenName_of hne) hcount
      (F.hUnitName_of hne) hv
      hlay hrr hrck hsame hseg hz hcoef V hV hE (F.setupRealizes_of hshape) (F.shapeIs_of hshape) hcl
      (F.foldsOk_of hshape) (F.coinsAgree_at hshape hst.1 hcl) hcapS hligS hregs hCT hleaf hread hT hopen hpub

end VStarFamily

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (w : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → List (List (SMsg T)))

/-- **The cap read from the inner messages `w l`** (`capOf`, Lig.lean). -/
noncomputable def capW : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray := fun l => capOf T sch (w l)

/-- **The Ligerito openings read from V*'s layers and the inner messages `w l`** (`ligOf`, Lig.lean). -/
noncomputable def ligW : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof :=
  fun l r => F.ligOf (w l) l r

variable (spec : SessionSpec) (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)

/-- **`vbridgeRead_model` with the cap and openings read from the inner messages**: `cap := capW w` and
`lig := ligW w`, whose `CapShaped` and `LigShaped` follow from the model setup's scheme (`hCapS_of`, `hLigS_of`). -/
theorem vbridgeRead_model_msgs
    (hshape : F.ShapeAt) (hcount : F.HPartCount)
    (hv : (F.assembled dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel bc).HVCoins dg N k Rw)
    (hlay : F.OneInLayer dg N k Rw) (hrr : F.RowsRegistered dg N k Rw) (hrck : F.RowColKeys dg N k Rw)
    (hsame : F.SameRow dg N k Rw) (hseg : F.ReadsUnsegmented dg N k Rw) (hz : F.HZeroL dg N k Rw)
    (hcoef : (F.assembled dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel bc).HCoefCols dg N k Rw)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V)
    (hE : (F.assembled dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel bc).HEnds dg N k Rw V)
    (hcl : F.ClaimsAt) (hregs : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48)
    (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw)
    (hread : F.MsgsRead dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel) (hT : TysIn T sch.ext)
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (F.innerSetup R ω) sch F.sh
        (((F.assembled dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel bc).tableSrc spec F.innerSetup
          (F.capW w) (F.ligW w) R ω t l).proofAt r)
        ((F.assembled dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel bc).rep R ω t l r))
    (hpub : (F.assembled dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel bc).PubCommits bc) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x
      (wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r (F.decOf dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel))
      (skOf F.Vs own sel) (cmtOf F.Vs bc)
      (msgOf F.Vs (skOf F.Vs own sel) (F.decOf dg N k Rw spec F.innerSetup (F.capW w) (F.ligW w) own sel)) :=
  F.vbridgeRead_model dg N k Rw spec (F.capW w) (F.ligW w) own sel bc hshape hcount hv hlay hrr
    hrck hsame hseg hz hcoef V hV hE hcl
    (F.hCapS_of dg N k Rw F.innerSetup (F.setupRealizes_of hshape) w)
    (F.hLigS_of dg N k Rw F.innerSetup (F.setupRealizes_of hshape) w) hregs hCT hleaf hread hT hopen hpub

end VStarFamily

end family

end FlockVBridge.Family
