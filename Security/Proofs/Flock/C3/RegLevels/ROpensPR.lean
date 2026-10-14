import Proofs.Flock.C3.RegLevels.ROpens
import Proofs.Flock.C3.RegLevels.SrcUnits
import Proofs.Flock.C3.RegLevels.Maps
import Proofs.Flock.C3.Reps.SrcOpens
import Proofs.Flock.C3.Reps.IoCols
import Proofs.Flock.C3.Reps.ReadSib
import Proofs.Flock.C3.Reps.SrcOK
import Proofs.Flock.C3.PubRows.Shape
import Proofs.Flock.C3.VStarList.Statements
import Proofs.Flock.Soundness.Refine.Seam

/-!
# n11: R's openings at `pR`, read from the layers (`ROpensOfLayers`, `RRowsPinned`)

At the proof G's lig lane builds (`pR` over `StackedMaps.ops`), R's openings are `ligOfRead`'s at R's rows and siblings.
Where those are the read of R's map's units (`StackedReads`, by construction) and that read's columns are input columns
a row reads (`StackedOpenCols`, `RepMaps.OpenCols`' form at R's map), `opensReadAt_two_of_layers`' hypotheses both hold.
`StackedOpenCols` is `RepMaps.openCols_of_readCol`'s at R's map (`stackedOpenCols_of`), from its units' parameters
(`StackedParams`, at `RecCoins.stackedParams`), their nets (`StackedNet`, from `RecOpen.check`: `stackedNet_of`) and S's
`OpenReadCol` at their names (`stackedReadCol`, `readCol_unitLaid`'s). `stackedParams_of_check`: a unit `RecOpen.check`
accepted under V*'s name `rec-open-stacked/l<L>-h<H>` has `stackedParams` `(L, H)`, so `stackedParams_of_checked`:
`StackedParams` at a family's list (`listStmt`) from R's levels in it, the verifier's check of each level's statement
and R's shape at `fast100`'s schedule at R's `m` (`shape?_opening`).

* `openRow_ligOfRead_out`, `openRow_ligOfRead_past`: past a level's `q` a `ligOfRead` opening is empty, and past a
  schedule of at least two levels (`DimSeam`) it is the last level's row with no siblings (the default level's `d - cap`
  is 0). So `rRowsPinned_of_read`: any proof whose openings are `ligOfRead`'s has `RRowsPinned`.
* `openRow_stackedRead_mask`: the read is the same in the masked layer (`colOf_mask`, as `RepMaps.openRow_mask`), so
  `rOpensOfLayers_of_read`: `ROpensOfLayers` at `W` every position, `rdR` the read of the masked layers.
* `openRow_pR_stacked`: `pR`'s openings at `StackedMaps.ops` are the read's, at any cap, messages, coins and caps.
  `rOpens_pR` is both hypotheses at `pR`.
* `opensReadAt_two_pR`: `OpensReadAt 2` at the reps' tied source (`repsOfVals` sets only strata 0 and 1,
  `os_tied_repsOfVals_of_ge`), `srcOK_repsOfVals`' `hos₂`; `srcOK_repsOfVals_pR` is `srcOK_repsOfVals` with it, at R's
  shape of `fast100`'s schedule (two levels, `DimSeam`).
* `readMaps`: R's map at its read (`stackedRead`'s rows, `readSibs` its siblings), where `StackedReads` holds by
  construction (`stackedReads_readMaps`) and the rows are `levelLanes` wide (`row_size_readMaps`, `levelTie_of_maps`'
  `hwd`). `opensReadAt_two_read` is `hos₂` there from the units' parameters and nets alone.
* `ligOfVals`: the step source with R's Ligerito steps woven from its messages and rounds along `ligShape` and its
  openings at stratum 2 R's rows (`repsOfVals`' form, at R). `ligSteps_of_plays`: where compiled Ligerito plays on them,
  the woven steps have its kinds, messages and coin steps (`levelTie_of_maps`' `hk`, `hl` and its coins), and no tie
  changes them (`lg_tied_repsOfVals_ligOfVals`); `os_two_tied_repsOfVals_ligOfVals` is its `ho`.
-/

namespace FlockC3.RegLevels

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Family (maskOf colOf ioColAt_of_readCol elems bytesAt LigRead ligOfRead levAt)
open FlockVBridge.OfVStar (ReadCol)
open FlockC3.Reps (UnitVals repsOfVals CapRead TsRead OpensOfLayers)

local notation "𝔽" => FlockLevel3.GF128

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}
  {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-! ## The rows past the ranges -/

theorem openRow_proofR (sch : Flock.Schedule) (cap : Array ByteArray) (M : Flock.RecAlgebra.Messages Flock.F128)
    (x : LigRead) (sch' : Flock.Schedule) (ℓ j : ℕ) :
    openRow (proofR sch cap M x) sch' ℓ j = openRow (ligOfRead 64 none sch x) sch' ℓ j := by
  unfold openRow
  rw [openAt_proofR]

/-- **Past a level's `q`, a `ligOfRead` opening is the same at every read**: no row and no siblings. -/
theorem openRow_ligOfRead_out {sl : Option ℕ} {sch : Flock.Schedule} (x x' : LigRead) {ℓ q : ℕ}
    (hℓ : ℓ < sch.levels.size) (hq : ¬q < (sch.levels.getD ℓ default).q) :
    openRow (ligOfRead 64 sl sch x) sch ℓ q = openRow (ligOfRead 64 sl sch x') sch ℓ q := by
  have hn := Nat.mul_le_mul_right ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap)
    (Nat.le_of_not_lt hq)
  have hr : ∀ y w, (levAt sl sch y ℓ w).rows[q]! = default := fun y w =>
    getElem!_neg _ q (by simp only [levAt, FlockVBridge.Family.levelOf, Array.size_ofFn]; exact hq)
  have hs : ∀ y w, Flock.sibsOf (levAt sl sch y ℓ w).siblings
      ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) q = [] := fun y w => by
    apply List.eq_nil_of_length_eq_zero
    simp only [Flock.sibsOf, Array.length_toList, Array.size_extract, levAt, FlockVBridge.Family.levelOf,
      Array.size_ofFn]
    omega
  obtain ⟨w, -, h⟩ := FlockVBridge.Family.openAt_ligOfRead (sch := sch) sl x hℓ
  obtain ⟨w', -, h'⟩ := FlockVBridge.Family.openAt_ligOfRead (sch := sch) sl x' hℓ
  unfold openRow
  rw [h, h']
  dsimp only
  rw [hr, hr, hs, hs]

/-- **Past the schedule** (of at least two levels), a `ligOfRead` opening is the last level's row with no siblings, so
two reads agreeing at the last level agree there. -/
theorem openRow_ligOfRead_past {sl : Option ℕ} {sch : Flock.Schedule} (x x' : LigRead) {ℓ q : ℕ}
    (h2 : 2 ≤ sch.levels.size) (hℓ : sch.levels.size ≤ ℓ)
    (hR : openRow (ligOfRead 64 sl sch x) sch (sch.levels.size - 1) q =
      openRow (ligOfRead 64 sl sch x') sch (sch.levels.size - 1) q) :
    openRow (ligOfRead 64 sl sch x) sch ℓ q = openRow (ligOfRead 64 sl sch x') sch ℓ q := by
  have e : ∀ y, FlockSoundness.Refine.openAt (ligOfRead 64 sl sch y) (sch.levels.size - 1) ℓ =
      FlockSoundness.Refine.openAt (ligOfRead 64 sl sch y) (sch.levels.size - 1) (sch.levels.size - 1) := fun y => by
    unfold FlockSoundness.Refine.openAt
    rw [ite_eq_right (show ¬ℓ = 0 by omega), ite_eq_right (show ¬ℓ < sch.levels.size - 1 by omega),
      ite_eq_right (show ¬sch.levels.size - 1 = 0 by omega), ite_eq_right (Nat.lt_irrefl (sch.levels.size - 1))]
  have hl : ¬ℓ < sch.levels.size := by omega
  have hd : (sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap = 0 :=
    (congrArg (fun v : Flock.Level => v.d - v.cap)
      ((FlockVBridge.Family.Conss.getD_default sch.levels ℓ).trans (getElem!_neg sch.levels ℓ hl))).trans rfl
  have hs : ∀ s : Array ByteArray, Flock.sibsOf s 0 q = [] := fun s => by
    apply List.eq_nil_of_length_eq_zero
    simp only [Flock.sibsOf, Array.length_toList, Array.size_extract]
    omega
  unfold openRow at hR ⊢
  rw [e x, e x', hd, hs, hs]
  exact Prod.ext (Prod.mk.inj hR).1 rfl

/-- **`RRowsPinned` at any proof whose openings are `ligOfRead`'s**, at a schedule of at least two levels. -/
theorem rRowsPinned_of_read {sl : Option ℕ} {sch : Flock.Schedule} (h2 : 2 ≤ sch.levels.size)
    {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof} (x : ((j : Fin S) → (Vs j).Bits) → LigRead)
    (hpr : ∀ Lb ℓ q, openRow (prR Lb) sch ℓ q = openRow (ligOfRead 64 sl sch (x Lb)) sch ℓ q) :
    RRowsPinned prR sch := by
  intro Lb Lb' h ℓ q
  have hin : ∀ l < sch.levels.size, ∀ j, openRow (ligOfRead 64 sl sch (x Lb)) sch l j =
      openRow (ligOfRead 64 sl sch (x Lb')) sch l j := fun l hl j => by
    by_cases hj : j < (sch.levels.getD l default).q
    · exact (hpr Lb l j).symm.trans ((h l hl j hj).trans (hpr Lb' l j))
    · exact openRow_ligOfRead_out (x Lb) (x Lb') hl hj
  refine (hpr Lb ℓ q).trans (Eq.trans ?_ (hpr Lb' ℓ q).symm)
  by_cases hℓ : ℓ < sch.levels.size
  · exact hin ℓ hℓ q
  · exact openRow_ligOfRead_past (x Lb) (x Lb') h2 (by omega) (hin (sch.levels.size - 1) (by omega) q)

/-! ## R's openings read off the layers -/

/-- Copy 0 of opening `j` of level `ℓ` of R's map's unit, in the layers `l`. -/
def stackedCol (M : FlockC3.Reps.StackedMaps stmt Vs) (l : (j : Fin S) → (Vs j).Bits) (ℓ j : ℕ) : ℕ → Bool :=
  colOf (Vs (M.sess ℓ)) (l (M.sess ℓ)) (M.un ℓ j)

/-- **R's openings read off the layers `l`** (`RepMaps.ligRead`'s at R's map): opening `j` of level `ℓ`'s row is the
first `2^lanes` elements of its unit's columns, its `h`-th sibling the 64 bytes from column `128 · 2^lanes + 512 h`. -/
def stackedRead (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule) (l : (j : Fin S) → (Vs j).Bits) :
    LigRead where
  row ℓ j := elems (stackedCol M l ℓ j) 0 (2 ^ (sch.levels.getD ℓ default).lanes)
  sib ℓ j h := bytesAt (stackedCol M l ℓ j) (128 * 2 ^ (sch.levels.getD ℓ default).lanes + 512 * h) 64
  salt _ _ := ByteArray.empty
  cap _ _ := ByteArray.empty

/-- **Hypothesis: R's openings are the read of its map's units** (by construction at `readMaps`,
`stackedReads_readMaps`): in range, `M.row` is `stackedRead`'s row and `sb` its siblings. -/
def StackedReads (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule)
    (sb : ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray) : Prop :=
  ∀ Lb, ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q, M.row Lb ℓ j = (stackedRead M sch Lb).row ℓ j ∧
    ∀ h < (sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap,
      sb Lb ℓ j h = (stackedRead M sch Lb).sib ℓ j h

/-- **Hypothesis: R's openings' `row` and `aux` columns are input columns a row reads** (`RepMaps.OpenCols` at R's map,
`M.sess` for `sessO`): at each level `ℓ` of `sch`, every column below `128 · 2^lanes + 512 (d − cap)` of its unit. -/
def StackedOpenCols (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule) : Prop :=
  ∀ ℓ < sch.levels.size, ∀ c < 128 * 2 ^ (sch.levels.getD ℓ default).lanes +
      512 * ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap),
    c < (Rows.ofNet (Vs (M.sess ℓ)).hU).nIn ∧ ReadCol (Rows.ofNet (Vs (M.sess ℓ)).hU) c

/-- **Hypothesis: R's map's units' parameters** (`RecCoins.stackedParams` at `M.sess`, the parse
`RecCoins.checkStackedLevel` is reached at): level `ℓ`'s unit is `rec-open-stacked/l<levelLanes>-h<d - cap>`. -/
def StackedParams (M : FlockC3.Reps.StackedMaps stmt Vs) (sh : Flock.Shape) : Prop :=
  ∀ ℓ < sh.sched.levels.size, Flock.RecCoins.stackedParams (Vs (M.sess ℓ)).c.unit.name =
    some (FlockVBridge.AccChain.levelLanes sh ℓ, (sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap)

/-- **R's map's units' nets** (`RecOpen.check_ok_stacked`'s conclusion, `stackedNet_of`): a unit at `stackedParams` `L`,
`H` has a valid `L` and `H` and is the net of `unitLaid L H` under its stacked name. -/
def StackedNet (M : FlockC3.Reps.StackedMaps stmt Vs) : Prop :=
  ∀ ℓ {L H : ℕ}, Flock.RecCoins.stackedParams (Vs (M.sess ℓ)).c.unit.name = some (L, H) →
    Flock.RecOpen.valid L H = true ∧
      Flock.HmNets.netOf (VStar.recOpenStackedName L H) (Flock.RecOpen.unitLaid L H) = .ok (Vs (M.sess ℓ)).c.unit

/-- **`StackedNet` from the verifier's check**: `RecOpen.check` rebuilds a `rec-open-stacked/` unit's net under its own
name and compares it (`RecOpen.check_ok_stacked`; `RecCoins.stackedParams` is `RecOpen.stackedParams` word for word). -/
theorem stackedNet_of (M : FlockC3.Reps.StackedMaps stmt Vs) (hrec : ∀ j, Flock.RecOpen.check (Vs j).c = .ok ()) :
    StackedNet M := fun ℓ _ _ hp => Flock.RecOpen.check_ok_stacked (hrec (M.sess ℓ)) hp

/-- **The name `rec-open-stacked/l<L>-h<H>` determines `L` and `H`** (`FlockVBridge.unitName_inj`'s argument). -/
theorem stackedName_inj {L H L' H' : ℕ} (h : Flock.RecOpen.stackedName L H = Flock.RecOpen.stackedName L' H') :
    L = L' ∧ H = H' := by
  have h := congrArg String.toList h
  simp only [Flock.RecOpen.stackedName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h2⟩ := FlockVBridge.digits_dash (dig L) (dig L') (by simpa using h)
  simp only [List.cons.injEq, true_and] at h2
  exact ⟨FlockVBridge.eq_of_toDigits h1, FlockVBridge.eq_of_toDigits h2⟩

theorem stackedName_startsWith (L H : ℕ) :
    (Flock.RecOpen.stackedName L H).startsWith "rec-open-stacked/" = true := by
  have : Flock.RecOpen.stackedName L H = "rec-open-stacked/" ++ s!"l{L}-h{H}" := by
    simp [Flock.RecOpen.stackedName, ← String.toList_inj, List.append_assoc, FlockVBridge.toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

/-- **A unit the verifier accepted under V*'s name of one of R's levels gives back its `L` and `H`** at
`RecCoins.stackedParams` (`FlockVBridge.params_of_check`'s, at R's names): where `stackedParams` is `none`, so is
`params` (`RecOpen.unitName_ne_stackedName`), and `RecOpen.check` refuses the name. Stated at an accepted unit, as
`params_of_check` is, so it needs no lemma about `String.splitOn`. -/
theorem stackedParams_of_check {c : Flock.Circuit} (h : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hn : c.unit.name = VStar.recOpenStackedName L H) : Flock.RecCoins.stackedParams c.unit.name = some (L, H) := by
  have hn' : c.unit.name = Flock.RecOpen.stackedName L H := hn
  show Flock.RecOpen.stackedParams c.unit.name = some (L, H)
  cases hp : Flock.RecOpen.stackedParams c.unit.name with
  | some p =>
    obtain ⟨L', H'⟩ := p
    obtain ⟨rfl, rfl⟩ := stackedName_inj ((Flock.RecOpen.stackedName_of_params hp).trans hn')
    rfl
  | none =>
    have hu : Flock.RecOpen.params c.unit.name = none := by
      cases hq : Flock.RecOpen.params c.unit.name with
      | none => rfl
      | some q =>
        exact absurd ((Flock.RecOpen.unitName_of_params hq).trans hn') (Flock.RecOpen.unitName_ne_stackedName _ _ _ _)
    unfold Flock.RecOpen.check at h
    rw [hu, hp] at h
    dsimp only at h
    rw [hn', Flock.RecOpen.startsWith_stackedName, stackedName_startsWith] at h
    cases h

/-- **R's shape at the opening's** (`RecOpening.shape?`, where `RecCoins.opening` reads R): its `m` and `fast100`'s
schedule at it. -/
theorem shape?_opening {m c : ℕ} {sh : Flock.Shape} (h : Flock.RecOpening.shape? m c = some sh) :
    sh.m = m ∧ Flock.fast100 sh.m = some sh.sched := by
  unfold Flock.RecOpening.shape? at h
  cases hf : Flock.fast100 m with
  | none => rw [hf] at h; cases h
  | some s =>
    rw [hf] at h
    dsimp only at h
    split at h
    · cases h
    · cases h
      exact ⟨rfl, hf⟩

/-- **V*'s statements as the family's list** (`StackedMaps`' `stmt` at a family): statement `j` is the list's `j`-th
(`vstarCount_eq`). -/
def listStmt {xR : HoloReg GF128 Dig} (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (j : Fin (vstarCount F.ish)) : VStar.Stmt :=
  F.stmts[(j : ℕ)]'(by rw [← FlockC3.VStarList.vstarCount_eq F.hstmts]; exact j.isLt)

/-- **`StackedParams` at the family's list, from the verifier's checks**: at R's shape `shR` (R's `m`, `fast100`'s
schedule at it, as `shape?_opening` gives), the list holds level `ℓ`'s statement at some index (`stacked_of_list`),
which is `M.sess ℓ` (`hsess`); the circuit the verifier checked as statement `M.sess ℓ` (`hck`, `stmt_of_session` at
an accepted session) is named as it (`nameChecked`), `rec-open-stacked/l<levelLanes>-h<d - cap>`, and every unit
passes `RecOpen.check` (`recCheck`), so `stackedParams_of_check`. -/
theorem stackedParams_of_checked {xR : HoloReg GF128 Dig} (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (M : FlockC3.Reps.StackedMaps (listStmt F) F.Vs)
    {r : Flock.RecVStar.StackedShape} (hr : F.ish.stacked = some r) {shR : Flock.Shape} (hm : shR.m = r.m)
    (hf : Flock.fast100 shR.m = some shR.sched)
    (hck : ∀ ℓ < shR.sched.levels.size, ∃ (sj : Flock.Stmt) (tags : _) (tables : _) (circuitB : _) (partition : _)
      (program : _) (own : _) (rows : _), sj.c = (F.Vs (M.sess ℓ)).c ∧
        Flock.RecVStar.checkStmt (some F.ish) tags tables sj circuitB partition program (some own)
          (some (M.sess ℓ : ℕ)) (some (F.bind (M.sess ℓ))) rows = .ok ()) :
    StackedParams M shR := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro ℓ hℓ
  obtain ⟨sch, hsch, hmem⟩ := FlockC3.VStarList.stacked_of_list F.hstmts hr
  obtain rfl : sch = shR.sched := Option.some.inj (hsch.symm.trans (hm ▸ hf))
  obtain ⟨i, hi, hget⟩ := Array.mem_iff_getElem.1 (hmem ℓ hℓ)
  have hi' : i < vstarCount F.ish := by rw [FlockC3.VStarList.vstarCount_eq F.hstmts]; exact hi
  have hj : M.sess ℓ = ⟨i, hi'⟩ := M.hsess ℓ ⟨i, hi'⟩ ⟨_, _, _, hget⟩
  obtain ⟨sj, tags, tables, circuitB, partition, program, own, rows, hc, hpass⟩ := hck ℓ hℓ
  obtain ⟨s, hs, hname⟩ :=
    FlockC3.PubRows.nameChecked F _ sj tags tables circuitB partition program own rows hc hpass
  rw [hj, Array.getElem?_eq_getElem hi, hget] at hs
  cases hs
  exact stackedParams_of_check (FlockC3.Reps.recCheck (F.zs R ω [] R₂ (M.sess ℓ))) hname.symm

/-- **`OpenReadCol` at R's stacked names** (`OpenReadCol` is at `RecOpen.unitName`; `stackedReadCol`). -/
def StackedReadCol (LANES H : ℕ) : Prop :=
  ∀ {net : Flock.Net} (hU : Built net),
    Flock.HmNets.netOf (VStar.recOpenStackedName LANES H) (Flock.RecOpen.unitLaid LANES H) = .ok net →
      Flock.RecOpen.valid LANES H = true → net.unitLog ≤ 32 →
        ∀ c < 128 * LANES + 512 * H, ReadCol (Rows.ofNet hU) c

/-- **`StackedReadCol`**: S's `readCol_unitLaid` at R's stacked name. -/
theorem stackedReadCol (LANES H : ℕ) : StackedReadCol LANES H := fun hU hn hv hu =>
  FlockC3.Reps.readCol_unitLaid LANES H _ hU hn hv hu

/-- **An opening's row and `aux` columns are inputs** at any name of `unitLaid` (`lt_nIn_of_params`' argument). -/
theorem lt_nIn_of_net {Q : VStmt} {LANES H : ℕ} {name : String}
    (hn : Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok Q.c.unit) {c : ℕ}
    (hc : c < 128 * LANES + 512 * H) : c < (Rows.ofNet Q.hU).nIn := by
  rw [Layout.ofNet_nIn_eq, FlockVBridge.inWords_unit hn]
  refine Nat.lt_of_lt_of_le ?_ (Flock.HmNets.le_groupWords _)
  rw [FlockVBridge.foldl_wordPorts, FlockVBridge.unitIn_eq]
  have := Nat.le_max_right 1 ((H + 1) / 2)
  omega

/-- **`StackedOpenCols` from the units' parameters and nets** (`RepMaps.openCols_of_readCol` at R's map): each level's
unit is the net of `unitLaid` (`StackedNet`), so the columns are inputs (`lt_nIn_of_net`) and read (`stackedReadCol`,
with `unitLog ≤ 32` by `unitLog_of_scope`). -/
theorem stackedOpenCols_of (M : FlockC3.Reps.StackedMaps stmt Vs) {sh : Flock.Shape} (hP : StackedParams M sh)
    (hN : StackedNet M) : StackedOpenCols M sh.sched := by
  intro ℓ hℓ c hc
  rw [FlockVBridge.Family.Conss.getD_default] at hc
  have hok := hN ℓ (hP ℓ hℓ)
  exact ⟨lt_nIn_of_net hok.2 hc,
    stackedReadCol _ _ (Vs (M.sess ℓ)).hU hok.2 hok.1 (FlockC3.Reps.unitLog_of_scope (Vs (M.sess ℓ)).hscope) c hc⟩

/-- R's opening at level `ℓ`, query `q`, read off the layers `l`. -/
def rdR (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule) (ℓ q : ℕ) (l : (j : Fin S) → (Vs j).Bits) :
    List 𝔽 × List Dig :=
  openRow (ligOfRead 64 none sch (stackedRead M sch l)) sch ℓ q

theorem stackedCol_mask (M : FlockC3.Reps.StackedMaps stmt Vs) {sch : Flock.Schedule} (hk : StackedOpenCols M sch)
    (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) {ℓ : ℕ} (hℓ : ℓ < sch.levels.size) (j c : ℕ)
    (hc : c < 128 * 2 ^ (sch.levels.getD ℓ default).lanes +
      512 * ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap)) :
    stackedCol M (maskOf Vs (fun _ _ => Finset.univ) i Xs) ℓ j c = stackedCol M Xs ℓ j c :=
  FlockC3.Reps.colOf_mask i Xs (M.sess ℓ) (M.un ℓ j) (hk ℓ hℓ c hc).1
    (ioColAt_of_readCol (M.un ℓ j) (hk ℓ hℓ c hc).1 (hk ℓ hℓ c hc).2)

/-- **R's openings are the same in the masked layer.** -/
theorem openRow_stackedRead_mask (M : FlockC3.Reps.StackedMaps stmt Vs) {sch : Flock.Schedule}
    (hk : StackedOpenCols M sch) (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) (ℓ q : ℕ) :
    openRow (ligOfRead 64 none sch (stackedRead M sch Xs)) sch ℓ q =
      openRow (ligOfRead 64 none sch (stackedRead M sch (maskOf Vs (fun _ _ => Finset.univ) i Xs))) sch ℓ q := by
  refine FlockC3.Reps.openRow_ligOfRead_congr (fun ℓ' w => ?_) ℓ q
  by_cases hℓ : ℓ' < sch.levels.size
  · refine FlockC3.Reps.levelOf_congr (fun j _ => ?_) (fun j _ _h _hh => ?_)
    · dsimp only [stackedRead]
      exact FlockC3.Reps.elems_congr fun c _ _h2 => (stackedCol_mask M hk i Xs hℓ j c (by omega)).symm
    · dsimp only [stackedRead]
      exact FlockC3.Reps.bytesAt_congr fun c _ _h2 => (stackedCol_mask M hk i Xs hℓ j c (by omega)).symm
  · refine FlockC3.Reps.levelOf_congr (fun _ hj => absurd hj ?_) (fun _ hj => absurd hj ?_) <;>
      rw [FlockVBridge.Family.Conss.getD_default, getElem!_neg sch.levels ℓ' hℓ] <;> exact Nat.not_lt_zero _

/-- **`ROpensOfLayers` at any proof whose openings are R's read**, at `W` every position. -/
theorem rOpensOfLayers_of_read (M : FlockC3.Reps.StackedMaps stmt Vs) {sch : Flock.Schedule}
    (hk : StackedOpenCols M sch)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof}
    (hpr : ∀ Lb ℓ q, openRow (prR Lb) sch ℓ q = openRow (ligOfRead 64 none sch (stackedRead M sch Lb)) sch ℓ q) :
    ROpensOfLayers zs dg N k Rw (fun _ _ => Finset.univ) (rdR M sch) prR sch := by
  intro R ω t R₂ τ _o _hA ℓ _hℓ q _hq
  exact (hpr _ ℓ q).trans (openRow_stackedRead_mask M hk 2 _ ℓ q)

/-! ## At `pR` -/

/-- **`pR`'s openings at `StackedMaps.ops` are R's read**, at any cap entries, messages, coins and caps. -/
theorem openRow_pR_stacked (M : FlockC3.Reps.StackedMaps stmt Vs) {sh : Flock.Shape}
    {sb : ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray} (hrd : StackedReads M sh.sched sb)
    (Lb : (j : Fin S) → (Vs j).Bits) (e0 : ℕ → ByteArray) (w : Array Flock.F128)
    (kap : Array (Array (Array Flock.F128))) (cl : Array (Flock.RecAlgebra.AClaim Unit)) (k₀ : ℕ)
    (c : Array (Array Flock.F128)) (e : ℕ → ℕ → ByteArray) (ℓ q : ℕ) :
    openRow (pR sh e0 w (M.ops sh Lb kap cl k₀ c) (sb Lb) e) sh.sched ℓ q =
      openRow (ligOfRead 64 none sh.sched (stackedRead M sh.sched Lb)) sh.sched ℓ q := by
  refine (openRow_proofR sh.sched _ _ (readR (M.ops sh Lb kap cl k₀ c) (sb Lb) e) sh.sched ℓ q).trans ?_
  refine FlockC3.Reps.openRow_ligOfRead_congr (fun ℓ' w' => ?_) ℓ q
  by_cases hℓ : ℓ' < sh.sched.levels.size
  · refine FlockC3.Reps.levelOf_congr (fun j hj => ?_) (fun j hj h hh => (hrd Lb ℓ' hℓ j hj).2 h hh)
    have hj' : j < (sh.sched.levels[ℓ']!).q := by
      rw [← FlockVBridge.Family.Conss.getD_default]
      exact hj
    show ((M.ops sh Lb kap cl k₀ c).rows ℓ')[j]! = _
    rw [M.rows_get sh Lb kap cl k₀ c hj']
    exact (hrd Lb ℓ' hℓ j hj).1
  · refine FlockC3.Reps.levelOf_congr (fun _ hj => absurd hj ?_) (fun _ hj => absurd hj ?_) <;>
      rw [FlockVBridge.Family.Conss.getD_default, getElem!_neg sh.sched.levels ℓ' hℓ] <;> exact Nat.not_lt_zero _

/-- **R's openings' hypotheses at `pR`**: at any `prR` that is `pR` over `StackedMaps.ops` at each layer assignment (at
any cap entries, messages, coins and caps), `ROpensOfLayers` at `W` every position and `RRowsPinned`, at a schedule of
at least two levels (`DimSeam`), where R's openings are the read of its map's units and their columns are read. -/
theorem rOpens_pR (M : FlockC3.Reps.StackedMaps stmt Vs) {sh : Flock.Shape} (h2 : 2 ≤ sh.sched.levels.size)
    (hk : StackedOpenCols M sh.sched) {sb : ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    (hrd : StackedReads M sh.sched sb)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof}
    (hpr : ∀ Lb, ∃ e0 w kap cl k₀ c e, prR Lb = pR sh e0 w (M.ops sh Lb kap cl k₀ c) (sb Lb) e) :
    ROpensOfLayers zs dg N k Rw (fun _ _ => Finset.univ) (rdR M sh.sched) prR sh.sched ∧
      RRowsPinned prR sh.sched := by
  have hrow : ∀ Lb ℓ q, openRow (prR Lb) sh.sched ℓ q =
      openRow (ligOfRead 64 none sh.sched (stackedRead M sh.sched Lb)) sh.sched ℓ q := fun Lb ℓ q => by
    obtain ⟨e0, w, kap, cl, k₀, c, e, h⟩ := hpr Lb
    exact (congrArg (fun p => openRow p sh.sched ℓ q) h).trans (openRow_pR_stacked M hrd Lb e0 w kap cl k₀ c e ℓ q)
  exact ⟨rOpensOfLayers_of_read M hk hrow, rRowsPinned_of_read h2 (stackedRead M sh.sched) hrow⟩

/-! ## At the reps' tied source -/

/-- **Past the reps, the reps' tied source's openings are `stp`'s**: `repsOfVals` sets strata 0 and 1, and the tie
keeps the openings (`tie_os`). -/
theorem os_tied_repsOfVals_of_ge
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) (sh : Flock.Shape)
    (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) {s : ℕ} (hs : ¬s < 2)
    (ℓ q : ℕ) : (tiedSrc (repsOfVals stp sh rep vals) R ω t Lb).os.1 s ℓ q = (stp R ω t Lb).os.1 s ℓ q := by
  show (tie (repsOfVals stp sh rep vals R ω t Lb)).os.1 s ℓ q = _
  rw [tie_os]
  exact dite_eq_right hs

/-- **`OpensReadAt 2` at the reps' tied source** (`srcOK_repsOfVals`' `hos₂`) whose step source's openings at stratum
2 are `prR`'s rows, `prR` at `pR` over R's map at R's shape `shR`. -/
theorem opensReadAt_two_pR (M : FlockC3.Reps.StackedMaps stmt Vs) {shR : Flock.Shape} (h2 : 2 ≤ shR.sched.levels.size)
    (hk : StackedOpenCols M shR.sched) {sb : ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    (hrd : StackedReads M shR.sched sb)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof}
    (hpr : ∀ Lb, ∃ e0 w kap cl k₀ c e, prR Lb = pR shR e0 w (M.ops shR Lb kap cl k₀ c) (sb Lb) e)
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (hsrc : ∀ R ω t Lb ℓ q, (stp R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) shR.sched ℓ q) :
    OpensReadAt zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) 2 := by
  obtain ⟨hv, hpin⟩ := rOpens_pR (zs := zs) (dg := dg) (N := N) (k := k) (Rw := Rw) M h2 hk hrd hpr
  exact opensReadAt_two_of_layers hv hpin fun R ω t Lb ℓ q =>
    (os_tied_repsOfVals_of_ge stp sh rep vals R ω t Lb (Nat.lt_irrefl 2) ℓ q).trans (hsrc R ω t Lb ℓ q)

/-- **`SrcOK` at the reps' normalized tied source, R's openings at `pR`**: `srcOK_repsOfVals` with `hos₂` from
`opensReadAt_two_pR`, at R's shape `shR` of `fast100`'s schedule (two levels, `fast100_dims`), its map's columns read
(`stackedOpenCols_of`), R's openings the read of its map's units (`StackedReads`) and `stp`'s at stratum 2. -/
theorem srcOK_repsOfVals_pR {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hleaf : HLeafC3 zs dg N k Rw)
    (hcoins : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageCoins (CT := CT) (t.map Prod.snd) u)
    (hc : CapRead zs dg N k Rw stp own sel) (hts : TsRead zs dg N k Rw (repsOfVals stp sh rep vals) own sel)
    (hsp : SparseMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hop : OpenMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hlg : LigMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hv : OpensOfLayers zs dg N k Rw W rd vals)
    (M : FlockC3.Reps.StackedMaps stmt Vs) {shR : Flock.Shape} (hf : Flock.fast100 shR.m = some shR.sched)
    (hP : StackedParams M shR) (hN : StackedNet M)
    {sb : ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    (hrd : StackedReads M shR.sched sb) {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof}
    (hpr : ∀ Lb, ∃ e0 w kap cl k₀ c e, prR Lb = pR shR e0 w (M.ops shR Lb kap cl k₀ c) (sb Lb) e)
    (hsrc : ∀ R ω t Lb ℓ q, (stp R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) shR.sched ℓ q)
    (hw : LigWhole zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hcap : ∃ j, T j = Cap Dig) (hbool : ∃ j, T j = Bool) (hty : ∃ j, T j = (Opens GF128 Dig × Salts (List UInt8)))
    (hT : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => ShapeTys T (holoAt st hp) u)
    (hplay : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => StagesPlay (holoAt st hp) u) :
    SrcOK zs dg N k Rw (tiedSrc (normSrc (repsOfVals stp sh rep vals))) own sel :=
  FlockC3.Reps.srcOK_repsOfVals hleaf hcoins hc hts hsp hop hlg hv
    (opensReadAt_two_pR M (FlockSoundness.Refine.fast100_dims _ _ hf).1.2.1 (stackedOpenCols_of M hP hN) hrd hpr hsrc)
    hw hcap hbool hty hT hplay

/-! ## R's map at its read -/

/-- **R's map at its read**: `M`'s sessions and units, its rows the read of its units at `sch` (`stackedRead`). -/
def readMaps (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule) : FlockC3.Reps.StackedMaps stmt Vs :=
  { M with row := fun l => (stackedRead M sch l).row }

/-- The siblings of R's read: `pR`'s at `readMaps`. -/
def readSibs (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule) :
    ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray :=
  fun l => (stackedRead M sch l).sib

/-- **`StackedReads` at R's map at its read**, by construction. -/
theorem stackedReads_readMaps (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule) :
    StackedReads (readMaps M sch) sch (readSibs M sch) :=
  fun _ _ _ _ _ => ⟨rfl, fun _ _ => rfl⟩

/-- **R's map's rows at its read are `levelLanes` wide** (`levelTie_of_maps`' `hwd`). -/
theorem row_size_readMaps (M : FlockC3.Reps.StackedMaps stmt Vs) (sh : Flock.Shape) (l : (j : Fin S) → (Vs j).Bits)
    (ℓ j : ℕ) : ((readMaps M sh.sched).row l ℓ j).size = FlockVBridge.AccChain.levelLanes sh ℓ := by
  show (elems (stackedCol M l ℓ j) 0 (2 ^ (sh.sched.levels.getD ℓ default).lanes)).size =
    2 ^ (sh.sched.levels[ℓ]!).lanes
  rw [FlockVBridge.Family.size_elems, FlockVBridge.Family.Conss.getD_default]

/-- **`OpensReadAt 2` at the reps' tied source, R's map at its read** (`srcOK_repsOfVals`' `hos₂`):
`opensReadAt_two_pR` at `readMaps`, its `StackedReads` by construction (`stackedReads_readMaps`) and its columns read
from its units' parameters and nets (`stackedOpenCols_of`), at R's shape `shR` of `fast100`'s
schedule. -/
theorem opensReadAt_two_read (M : FlockC3.Reps.StackedMaps stmt Vs) {shR : Flock.Shape}
    (hf : Flock.fast100 shR.m = some shR.sched) (hP : StackedParams M shR) (hN : StackedNet M)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof}
    (hpr : ∀ Lb, ∃ e0 w kap cl k₀ c e,
      prR Lb = pR shR e0 w ((readMaps M shR.sched).ops shR Lb kap cl k₀ c) (readSibs M shR.sched Lb) e)
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (hsrc : ∀ R ω t Lb ℓ q, (stp R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) shR.sched ℓ q) :
    OpensReadAt zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) 2 :=
  opensReadAt_two_pR (readMaps M shR.sched) (FlockSoundness.Refine.fast100_dims _ _ hf).1.2.1
    (stackedOpenCols_of (sh := shR) (readMaps M shR.sched) hP hN) (stackedReads_readMaps M shR.sched) hpr hsrc

/-- **A row is the elements its columns hold.** -/
theorem elems_of_cols {f : ℕ → Bool} {base : ℕ} {r : Array Flock.F128}
    (h : FlockVBridge.ColsHold f base (Flock.MerkleScheme.rowBytes r)) : elems f base r.size = r := by
  have hc := FlockVBridge.Family.colsHold_elems f base r.size
  refine Array.ext (FlockVBridge.Family.size_elems f base r.size) fun k hk hk' => ?_
  have e : (elems f base r.size)[k]! = r[k]! := FlockVBridge.AccChain.getElem!_of_bitL fun t ht => by
    have hb : 128 * k + t < 8 * (Flock.MerkleScheme.rowBytes r).size := by rw [FlockVBridge.size_rowBytes]; omega
    have hb' : 128 * k + t < 8 * (Flock.MerkleScheme.rowBytes (elems f base r.size)).size := by
      rw [FlockVBridge.size_rowBytes, FlockVBridge.Family.size_elems]; omega
    exact (hc _ hb').symm.trans (h _ hb)
  rwa [getElem!_pos (elems f base r.size) k hk, getElem!_pos r k hk'] at e

/-- **`OpensAtMaps` at R's map at its read**: an opening's row is the row its copy's own columns hold (`OpensTopAt`,
`elems_of_cols`), which is `readMaps`' row, so any map whose openings are its units' (`OpensAtMaps`) gives
`readMaps`', and `levelTie_of_maps` takes `readMaps` with `row_size_readMaps` as `hwd`. -/
theorem opensAtMaps_readMaps (M : FlockC3.Reps.StackedMaps stmt Vs) {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} {sh : Flock.Shape} {sibs : ℕ → Array ByteArray} {cap : Array ByteArray}
    {p : Flock.LigeritoProof} {lg : Flock.RecAlgebra.LgPts} (h : OpensAtMaps M zs Lb sh sibs cap p lg) :
    OpensAtMaps (readMaps M sh.sched) zs Lb sh sibs cap p lg := by
  intro ℓ hℓ j hj
  obtain ⟨a, cf, A, hT⟩ := h ℓ hℓ j hj
  have e : (readMaps M sh.sched).row Lb ℓ j = M.row Lb ℓ j := by
    obtain ⟨h0, _, _, _, _, _, _, _, _, _, _, _, hr, _, _, _, _, _, _, _, _, hrow, _⟩ := hT
    have hL : 2 ^ (sh.sched.levels.getD ℓ default).lanes = (M.row Lb ℓ j).size := by
      rw [FlockVBridge.Family.Conss.getD_default]
      exact hr.symm
    have hf : stackedCol M Lb ℓ j = _ := FlockVBridge.Family.colOf_eq (Vs (M.sess ℓ)) (Lb (M.sess ℓ)) (M.un ℓ j) h0
    show elems (stackedCol M Lb ℓ j) 0 (2 ^ (sh.sched.levels.getD ℓ default).lanes) = M.row Lb ℓ j
    rw [hL, hf]
    exact elems_of_cols hrow
  refine ⟨a, cf, A, ?_⟩
  rw [e]
  exact hT

/-- **R's map's rows at its read are in its units' columns** (`OpensTopAt`'s row conjunct at `readMaps`, by
construction): copy 0 of opening `j`'s unit holds `readMaps`' row from column 0 (`colsHold_elems`). -/
theorem rowCols_readMaps (M : FlockC3.Reps.StackedMaps stmt Vs) (sch : Flock.Schedule) (Lb : (j : Fin S) → (Vs j).Bits)
    (ℓ j : ℕ) (h0 : 0 < (Vs (M.sess ℓ)).c.upv) :
    FlockVBridge.ColsHold (FlockVBridge.copyVal (FlockVBridge.keyInst (Vs (M.sess ℓ)).hU (Vs (M.sess ℓ)).nv
      (Layout.outsOf (Vs (M.sess ℓ)).c (Vs (M.sess ℓ)).hU (Vs (M.sess ℓ)).nv) (M.un ℓ j)) (Lb (M.sess ℓ)) h0) 0
      (Flock.MerkleScheme.rowBytes ((readMaps M sch).row Lb ℓ j)) := by
  have h := FlockVBridge.Family.colsHold_elems (stackedCol M Lb ℓ j) 0 (2 ^ (sch.levels.getD ℓ default).lanes)
  intro t ht
  rw [← FlockVBridge.Family.colOf_eq (Vs (M.sess ℓ)) (Lb (M.sess ℓ)) (M.un ℓ j) h0]
  exact h t ht

/-- **R's map's columns are input wires of their units** (`OpensTopAt`'s wire conjunct at copy 0, from
`StackedOpenCols` by `ioColAt_of_readCol`). -/
theorem ioCols_of_open (M : FlockC3.Reps.StackedMaps stmt Vs) {sch : Flock.Schedule} (hk : StackedOpenCols M sch)
    {ℓ : ℕ} (hℓ : ℓ < sch.levels.size) (j : ℕ) {col : ℕ}
    (hc : col < 128 * 2 ^ (sch.levels.getD ℓ default).lanes +
      512 * ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap)) :
    col < (Rows.ofNet (Vs (M.sess ℓ)).hU).nIn ∧ FlockVBridge.Family.IoColAt (Vs (M.sess ℓ)) (M.un ℓ j) col :=
  ⟨(hk ℓ hℓ col hc).1, ioColAt_of_readCol (M.un ℓ j) (hk ℓ hℓ col hc).1 (hk ℓ hℓ col hc).2⟩

/-! ## R's Ligerito steps and openings at the source -/

/-- **The step source with R's Ligerito steps and openings read off R's proof** (`repsOfVals`' form, at R): `stp` with
its Ligerito steps the messages `ms` and the coin steps of the rounds `cs` woven along compiled Ligerito's shape at R's
schedule (`ligShape`), and its openings at stratum 2 `opR`; the rest is `stp`'s. -/
def ligOfVals (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (ms : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List FlockSoundness.Game.Msg)
    (cs : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List (Array Flock.F128))
    (opR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → List 𝔽 × List Dig) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp) := fun R ω t Lb =>
  { stp R ω t Lb with
    lg := FlockSoundness.Discharge.Exec.weave
      (FlockSoundness.Discharge.Exec.ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch) (ms R ω t Lb)
      ((cs R ω t Lb).map (FlockSoundness.Discharge.PlaysCoin.coinArr fOf))
    os := (fun q ℓ j => if q = 2 then opR R ω t Lb ℓ j else (stp R ω t Lb).os.1 q ℓ j, (stp R ω t Lb).os.2) }

/-- **Woven Ligerito steps are a play's tape** (`levelTie_of_maps`' `hk` and `hl`, and its Ligerito coins): where
compiled Ligerito at R's schedule plays on `ms` and the rounds `cs`, steps woven from them along its shape are a tape
along it (`tapeAlong_of_plays` at `ligeritoC_spine`), so they have its kinds, the messages `ms` and the coin steps of
`cs` (`Woven.steps`). -/
theorem ligSteps_of_plays {u : C3Run (holoAt st hp)} {ms : List FlockSoundness.Game.Msg}
    {cs : List (Array Flock.F128)} {T₀ : 𝔽} {b : ℕ → 𝔽} {pend : FlockSoundness.Model.PendL 𝔽 Dig}
    (h : FlockSoundness.Game.Plays fOf
      (FlockSoundness.Model.ligeritoC (D := Dig) FlockSoundness.execArith 2 (holoAt st hp).rsch.sch T₀ b) ms cs pend)
    (hu : u.lg = FlockSoundness.Discharge.Exec.weave
      (FlockSoundness.Discharge.Exec.ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch) ms
      (cs.map (FlockSoundness.Discharge.PlaysCoin.coinArr fOf))) :
    u.lg.map FlockSoundness.Discharge.Exec.kindOf <+:
        FlockSoundness.Discharge.Exec.ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch ∧
      u.lg.filterMap Sum.getLeft? = ms ∧
        u.lg.filterMap Sum.getRight? = cs.map (FlockSoundness.Discharge.PlaysCoin.coinArr fOf) :=
  FlockC3.Reps.Woven.steps ⟨hu, FlockC3.Reps.tapeAlong_of_plays
    (FlockSoundness.Discharge.Exec.ligeritoC_spine (D := Dig) FlockSoundness.execArith 2 _ T₀ b) h⟩

/-- **R's Ligerito steps at the reps' tied source over `ligOfVals`** are its weave (`ligSteps_of_plays`' `hu`): no tie
changes the steps. -/
theorem lg_tied_repsOfVals_ligOfVals
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (ms : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List FlockSoundness.Game.Msg)
    (cs : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List (Array Flock.F128))
    (opR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → List 𝔽 × List Dig)
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) :
    (tiedSrc (repsOfVals (ligOfVals stp ms cs opR) sh rep vals) R ω t Lb).lg = FlockSoundness.Discharge.Exec.weave
      (FlockSoundness.Discharge.Exec.ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch) (ms R ω t Lb)
      ((cs R ω t Lb).map (FlockSoundness.Discharge.PlaysCoin.coinArr fOf)) := rfl

/-- **Stratum 2's openings at `ligOfVals` are `opR`'s** (`opensReadAt_two_read`'s `hsrc` where `opR` is `prR`'s rows,
`openRow (prR Lb) shR.sched`). -/
theorem os_two_ligOfVals
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (ms : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List FlockSoundness.Game.Msg)
    (cs : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List (Array Flock.F128))
    (opR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → List 𝔽 × List Dig)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) (ℓ j : ℕ) :
    (ligOfVals stp ms cs opR R ω t Lb).os.1 2 ℓ j = opR R ω t Lb ℓ j := rfl

/-- **`levelTie_of_maps`' `ho` at the reps' tied source over `ligOfVals`**: past the reps the tied source's openings
are `ligOfVals`' (`os_tied_repsOfVals_of_ge`), at stratum 2 `opR`'s (`os_two_ligOfVals`). -/
theorem os_two_tied_repsOfVals_ligOfVals
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (ms : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List FlockSoundness.Game.Msg)
    (cs : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → List (Array Flock.F128))
    (opR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → List 𝔽 × List Dig)
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) (ℓ j : ℕ) :
    (tiedSrc (repsOfVals (ligOfVals stp ms cs opR) sh rep vals) R ω t Lb).os.1 2 ℓ j = opR R ω t Lb ℓ j :=
  (os_tied_repsOfVals_of_ge (ligOfVals stp ms cs opR) sh rep vals R ω t Lb (Nat.lt_irrefl 2) ℓ j).trans
    (os_two_ligOfVals stp ms cs opR R ω t Lb ℓ j)

end

end FlockC3.RegLevels
