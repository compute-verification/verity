import Proofs.Flock.VBridge.Family.Assembly

/-!
# The inner statement's setup, from the hidden statement

The recursion's inner protocol is the compiled table on the model statement `M.toStatement regs` (`flockInnerCL`); no
executable statement `Stmt` stands behind it, so the setup the table files are read at is the model statement's own:

* `modelSetup S ptLocal mPts`: `m` and `k_log` the statement's, the fold `Σ_i e_i·(α·A₀[i, c] + B₀[i, c])` computed
  from `S`'s matrices (`modelFold`), with `β` at the pin and halved at the round challenges as `Stmt.folded`'s fallback
  (Statement.lean:332-336), the extra claims the model's at the link points (`claimOf`), and `hm96-sha512` rows;
* `modelSetup_realizes`: it realizes `S` (`SetupRealizes`) at any `fast100` schedule of `S.m`;
* `foldsOk_modelSetup`: its fold returns 64 elements at the round challenges `residualLins` draws (`ts_size`), so
  `FoldsOk` holds;
* at the family, `innerSetup` is `modelSetup` of each history's hidden statement, and `ShapeAt` (V*'s `m`, `k_log` and
  schedule are the class's) gives `ShapeIs`, `SetupRealizes` and `FoldsOk` there;
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
(`Model.extraClaims`, read through `ptsOf`), its rows `hm96-sha512`'s. -/
noncomputable def modelSetup (S : Model.Statement) (ptLocal mPts : ℕ) : Setup where
  m := S.m
  kLog := S.kLog
  digest := ByteArray.empty
  fold := modelFold S
  extra P := pure ((FlockSoundness.Model.extraClaims S ptLocal mPts (FlockSoundness.Refine.ptsOf mPts P)).toArray.map
    claimOf)
  publics _ := pure ()
  merkle := MerkleScheme.hm96Sha512
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
theorem ts_size {sh : Shape} {src : Flock.RecAlgebra.Coins} {extras : Array Flock.Claim}
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

theorem round_size {rounds : Array (Array F128)} {k c : ℕ} {v : Array F128}
    (h : (Flock.RecAlgebra.Coins.ofRounds rounds).round k c = .ok v) : k < rounds.size ∧ (rounds[k]!).size = c := by
  obtain ⟨h1, h2⟩ := Flock.RecAlgebra.ofRounds_get? h
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
  obtain ⟨x, hx⟩ := (roundSizes_spec hS (Flock.RecAlgebra.zeroRounds (roundSizes sh))
    (Flock.RecAlgebra.extrasLike sh) #[]).2 ⟨zeroRounds_sizes _, by simp [Flock.RecAlgebra.extrasLike]; omega⟩
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
  have hlt := (round_size eg).1
  refine ⟨(Flock.RecAlgebra.zeroRounds (roundSizes sh)).size - k3,
    ⟨by omega, size_at (by omega) (round_size e0).2, size_at (by omega) (round_size e1).2,
      size_at (by omega) (round_size e2).2, fun i hi => ?_, fun i hi => ?_, fun i hi => ?_,
      size_at (by omega) (round_size eg).2⟩, by omega⟩
  · obtain ⟨c, hc, -⟩ := ezc i hi
    exact size_at (by omega) (round_size hc).2
  · by_cases hi0 : i = 0
    · exact size_at (by omega) (round_size eα).2
    by_cases hi1 : i = 1
    · exact size_at (by omega) (round_size eβ).2
    by_cases hil : i = 2 + (sh.kLog - 6)
    · exact size_at (by omega) (round_size er).2
    obtain ⟨c, hc, -⟩ := elc (i - 2) (by omega)
    exact size_at (by omega) (round_size hc).2
  · obtain ⟨c, hc, -⟩ := ers i (by omega)
    exact size_at (by omega) (round_size hc).2

/-- **Ligerito's rounds are at most those `verifierRows` takes after ring switching**: the rounds a Ligerito run's
frames close (`chunks (ligFrames …)`, `.sq` frames of the schedule's counts only, FramesLigerito.lean:130-141, 207-219)
are at most the rounds `run` draws after the ring switch's (`roundSizes`, RecAlgebra.lean:532-539). -/
def LigRoundsLe (sh : Shape) (sch : Flock.Schedule) : Prop :=
  ∀ (T : F128) (cap : Array ByteArray) (lig : LigeritoProof),
    3 + (sh.m - 6) + 2 + (sh.kLog - 6) + 1 + sh.claims + 1 + (chunks (ligFrames sch T cap lig)).1.length ≤
      (roundSizes sh).length

/-- **`CoinsAgree`** at a setup of the shape's `m` and fold `k_log`, from `roundSizes_shaped` and `LigRoundsLe`. -/
theorem coinsAgree_of {st : Setup} {sh : Shape} {sch : Flock.Schedule} {n : ℕ} {S : Flock.RecAlgebra.GfS}
    (hS : Flock.RecAlgebra.structureOf sh = .ok S) (hcl : sh.claims = 2 + n) (hm : st.m = sh.m)
    (hk : st.fold.kLog = sh.kLog) (h14 : 14 ≤ sh.m) (hlig : LigRoundsLe sh sch) : CoinsAgree st sh sch n := by
  obtain ⟨L, hL, hlen⟩ := roundSizes_shaped hS hcl h14
  refine ⟨L, by rw [hm, hk]; exact hL, fun T cap lig => ?_⟩
  have := hlig T cap lig
  omega

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 leaf512 cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **The inner setup at each history**: the model setup of the hidden statement `x R ω`. -/
noncomputable def innerSetup (R : Reg) (ω : L.Ω) : Setup :=
  modelSetup ((F.x R ω).toStatement regs) cls.ptLocal cls.mPts

/-- **V*'s shape is the class's** (the verifier's gap `vsh`, with `ClaimsAt`): `vsh`'s `m` and `k_log` are the hidden
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
    (hcl : F.ClaimsAt) (hlig : LigRoundsLe F.sh sch) (R : Reg) (ω : L.Ω) :
    CoinsAgree (F.innerSetup R ω) F.sh sch (2 * ((F.x R ω).toStatement regs).regions.length) := by
  obtain ⟨hf, hm, -⟩ := shape?_eq F.hsh
  have h22 := (fast100_range hf).1
  exact coinsAgree_of hS (F.claimsIs_of hcl R ω) (F.shapeIs_of h R ω).1.symm (F.shapeIs_of h R ω).2.1.symm
    (by omega) hlig

end VStarFamily

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)
    (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)

/-- **VBridge G's read conjunct at the model setup**: `vbridgeRead_assembled` at `innerSetup`, whose `SetupRealizes`,
`ShapeIs` and `FoldsOk` follow from `ShapeAt`. Without a history and registration (`¬ Nonempty (Reg × L.Ω × Reg₂)`),
`VBridgeRead`'s first two conjuncts quantify over none and its last two are the lanes' (`vDecodesRead_of`,
`vFullRead_of`), so `hne` is not needed. -/
theorem vbridgeRead_model (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hshape : F.ShapeAt) (honame : F.HOpenName) (hcount : F.HPartCount) (hname : F.HUnitName)
    (hseen : F.HRowSeen) (hv : (F.assembled spec F.innerSetup cap lig own sel bc dec).HVCoins dg N k Rw)
    (hlay : F.OneInLayer dg N k Rw) (hrr : F.RowsRegistered dg N k Rw) (hrck : F.RowColKeys dg N k Rw)
    (hsame : F.SameRow dg N k Rw) (hshared : F.RowsShared dg N k Rw) (hz : F.HZeroL dg N k Rw)
    (hcoef : (F.assembled spec F.innerSetup cap lig own sel bc dec).HCoefCols dg N k Rw)
    (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128) (hV : F.HAccRows dg N k Rw V)
    (hE : (F.assembled spec F.innerSetup cap lig own sel bc dec).HEnds dg N k Rw V) (hcl : F.ClaimsAt)
    (hlig : LigRoundsLe F.sh sch)
    (hcapS : F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (F.innerSetup R ω) sch (cap l))
    (hligS : F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (F.innerSetup R ω) sch (lig l r))
    (hregs : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48)
    (htr : F.AtAccepted dg N k Rw fun R ω t l =>
      ((F.assembled spec F.innerSetup cap lig own sel bc dec).tableSrc spec F.innerSetup cap lig R ω t l).TranscriptIs
        T CT cls ((wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r dec l).zip (t.map Prod.snd)))
    (hopen : F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (F.innerSetup R ω) sch F.sh
        (((F.assembled spec F.innerSetup cap lig own sel bc dec).tableSrc spec F.innerSetup cap lig R ω t l).proofAt r)
        ((F.assembled spec F.innerSetup cap lig own sel bc dec).rep R ω t l r))
    (hpub : (F.assembled spec F.innerSetup cap lig own sel bc dec).PubCommits bc) (hsel : SelRounds sel)
    (hS : F.LawsExec) {dj : Fin F.stmts.size → Lean.Json} {d : Fin F.stmts.size → Flock.Draw.UnitDraw}
    (hu4 : F.U4Passes own dj d) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x (wmsgOf F.Vs (skOf F.Vs own sel) (ℐ).r dec) (skOf F.Vs own sel)
      (cmtOf F.Vs bc) (msgOf F.Vs (skOf F.Vs own sel) dec) := by
  rcases isEmpty_or_nonempty (Reg × L.Ω × Reg₂) with he | hne
  · refine ⟨?_, ?_, (F.assembled spec F.innerSetup cap lig own sel bc dec).vDecodesRead_of dec rfl rfl,
      (F.assembled spec F.innerSetup cap lig own sel bc dec).vFullRead_of own sel rfl hsel hS hu4⟩
    · intro R ω _ R₂
      exact (he.false (R, ω, R₂)).elim
    · intro _ R ω _ _ _ R₂
      exact (he.false (R, ω, R₂)).elim
  · obtain ⟨hsome, -, -, -⟩ := F.structure_of hne honame hcount
    obtain ⟨GS, gs, hst⟩ := F.hStruct_of hne hsome hname
    exact F.vbridgeRead_assembled dg N k Rw spec F.innerSetup cap lig own sel bc dec hne honame hcount hname hseen hv
      hlay hrr hrck hsame hshared hz hcoef V hV hE (F.setupRealizes_of hshape) (F.shapeIs_of hshape) hcl
      (F.foldsOk_of hshape) (F.coinsAgree_at hshape hst.1 hcl hlig) hcapS hligS hregs htr hopen hpub hsel hS hu4

end VStarFamily

end family

end FlockVBridge.Family
