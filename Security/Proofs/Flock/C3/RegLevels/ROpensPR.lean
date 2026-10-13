import Proofs.Flock.C3.RegLevels.ROpens
import Proofs.Flock.C3.RegLevels.SrcUnits
import Proofs.Flock.C3.Reps.SrcOpens

/-!
# n11: R's openings at `pR`, read from the layers (`ROpensOfLayers`, `RRowsPinned`)

At the proof G's lig lane builds (`pR` over `StackedMaps.ops`), R's openings are `ligOfRead`'s at R's rows and siblings.
Where those are the read of R's map's units (`StackedReads`, by construction) and that read's columns are input columns
a row reads (`StackedOpenCols`, `RepMaps.OpenCols`' form at R's map), `opensReadAt_two_of_layers`' hypotheses both hold.

* `openRow_ligOfRead_out`, `openRow_ligOfRead_past`: past a level's `q` a `ligOfRead` opening is empty, and past a
  schedule of at least two levels (`DimSeam`) it is the last level's row with no siblings (the default level's `d - cap`
  is 0). So `rRowsPinned_of_read`: any proof whose openings are `ligOfRead`'s has `RRowsPinned`.
* `openRow_stackedRead_mask`: the read is the same in the masked layer (`colOf_mask`, as `RepMaps.openRow_mask`), so
  `rOpensOfLayers_of_read`: `ROpensOfLayers` at `W` every position, `rdR` the read of the masked layers.
* `openRow_pR_stacked`: `pR`'s openings at `StackedMaps.ops` are the read's, at any cap, messages, coins and caps.
  `rOpens_pR` is both hypotheses at `pR`, and `opensReadAt_two_pR` is `OpensReadAt 2` at H's normalized tied source.
-/

namespace FlockC3.RegLevels

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Family (maskOf colOf ioColAt_of_readCol elems bytesAt LigRead ligOfRead levAt)
open FlockVBridge.OfVStar (ReadCol)

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

/-- **Hypothesis: R's openings are the read of its map's units** (G's lig lane's, by construction): in range, `M.row` is
`stackedRead`'s row and `sb` its siblings. -/
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

/-- **`OpensReadAt 2` at H's normalized tied source** whose openings at stratum 2 are `prR`'s rows, `prR` at `pR`. -/
theorem opensReadAt_two_pR (M : FlockC3.Reps.StackedMaps stmt Vs) {sh : Flock.Shape} (h2 : 2 ≤ sh.sched.levels.size)
    (hk : StackedOpenCols M sh.sched) {sb : ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    (hrd : StackedReads M sh.sched sb)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {prR : ((j : Fin S) → (Vs j).Bits) → Flock.LigeritoProof}
    (hpr : ∀ Lb, ∃ e0 w kap cl k₀ c e, prR Lb = pR sh e0 w (M.ops sh Lb kap cl k₀ c) (sb Lb) e)
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hsrc : ∀ R ω t Lb ℓ q, (stp R ω t Lb).os.1 2 ℓ q = openRow (prR Lb) sh.sched ℓ q) :
    OpensReadAt zs dg N k Rw (tiedSrc (normSrc stp)) 2 := by
  obtain ⟨hv, hpin⟩ := rOpens_pR (zs := zs) (dg := dg) (N := N) (k := k) (Rw := Rw) M h2 hk hrd hpr
  exact opensReadAt_two_norm hv hpin hsrc

end

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.opensReadAt_two_pR
