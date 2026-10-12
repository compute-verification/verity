import Proofs.Flock.C3.Reps.Maps
import Proofs.Flock.C3.Reps.OpensRead
import Proofs.Flock.VBridge.Family.Opened
import Proofs.Flock.VBridge.Renumber

/-!
# The reps' openings at G's values, read from the layers' read wires

G's values open rep `r`'s Ligerito rows from V*'s layers (`ligOfRead` over `RepMaps`' `sessO`, `unO`, as
`VStarFamily.ligRead`): level `ℓ`'s opening `j` is copy 0 of unit `unO ℓ r j` of session `sessO ℓ`, its row the first
`2^lanes` elements of its columns, its `h`-th sibling the 64 bytes from column `128 · 2^lanes + 512 h`. Those are
input columns its rows read, so at `W` every position the masked layer has them, and `SrcOpensOfLayers` holds.

* `colOf_mask`: an input column a row reads is the same in the masked layer (`copyVal_input`, `IoColAt`);
  `elems_congr`, `bytesAt_congr`, `levelOf_congr`, `openRow_ligOfRead_congr`: the openings read only those columns, at
  in-range queries and siblings (levels past the schedule open nothing; `openRow` reads no cap and no salt).
* `RepMaps.OpenCols` (the hypothesis, `OpenKeys`' form): every `row` and `aux` column of each level's unit is an input
  column a row reads. `srcOpensOfLayers` and `opensReadAt` follow at any `b` whose openings are `ligOfRead`'s.
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Family (maskOf colOf IoColAt ioColAt_of_readCol elems bytesAt natOfBits LigRead ligOfRead levAt)
open FlockVBridge.OfVStar (ReadCol)

local notation "𝔽" => FlockLevel3.GF128

section cols

theorem natOfBits_congr {f f' : ℕ → Bool} {n : ℕ} (h : ∀ t < n, f t = f' t) : natOfBits f n = natOfBits f' n := by
  refine Nat.eq_of_testBit_eq fun i => ?_
  rw [FlockVBridge.Family.testBit_natOfBits, FlockVBridge.Family.testBit_natOfBits]
  by_cases hi : i < n
  · rw [h i hi]
  · simp [hi]

theorem elems_congr {f f' : ℕ → Bool} {base n : ℕ} (h : ∀ c, base ≤ c → c < base + 128 * n → f c = f' c) :
    elems f base n = elems f' base n := by
  unfold elems
  refine congrArg (Array.ofFn (n := n)) (funext fun i => FlockVBridge.OfVStar.ofBits_congr fun t _ht => ?_)
  exact h _ (by omega) (by have := i.isLt; omega)

theorem bytesAt_congr {f f' : ℕ → Bool} {base N : ℕ} (h : ∀ c, base ≤ c → c < base + 8 * N → f c = f' c) :
    bytesAt f base N = bytesAt f' base N := by
  unfold bytesAt
  apply congrArg ByteArray.mk
  apply congrArg List.toArray
  refine List.map_congr_left fun i hi => ?_
  apply congrArg UInt8.ofNat
  refine natOfBits_congr fun t _ht => ?_
  have := List.mem_range.1 hi
  exact h _ (by omega) (by omega)

theorem levelOf_congr {H : ℕ} {sl : Option ℕ} {q w n : ℕ} {row row' : ℕ → Array F128}
    {sib sib' : ℕ → ℕ → ByteArray} {salt : ℕ → ByteArray} (hr : ∀ j < q, row j = row' j)
    (hs : ∀ j < q, ∀ h < n, sib j h = sib' j h) :
    FlockVBridge.Family.levelOf H sl q w n row sib salt = FlockVBridge.Family.levelOf H sl q w n row' sib' salt := by
  unfold FlockVBridge.Family.levelOf
  congr 1
  · congr 1
    funext j
    rw [hr j.val j.isLt]
  · congr 1
    funext k
    have hk := k.isLt
    have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; omega)
    rw [hs (k.val / n) (Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm n q]; exact hk)) (k.val % n) (Nat.mod_lt _ hn)]

/-- **`openRow` reads only the levels' openings**, not the caps or the nonces. -/
theorem openRow_ligOfRead_congr {sl : Option ℕ} {sch : Flock.Schedule} {x x' : LigRead}
    (h : ∀ ℓ w, levAt sl sch x ℓ w = levAt sl sch x' ℓ w) (ℓ j : ℕ) :
    openRow (ligOfRead 64 sl sch x) sch ℓ j = openRow (ligOfRead 64 sl sch x') sch ℓ j := by
  have e : openAt (ligOfRead 64 sl sch x) (sch.levels.size - 1) ℓ =
      openAt (ligOfRead 64 sl sch x') (sch.levels.size - 1) ℓ := by
    simp only [levAt] at h
    simp only [openAt, ligOfRead, h]
  unfold openRow
  rw [e]

/-- **An input column a row reads is the same in the masked layer** at `W` every position. -/
theorem colOf_mask {S : ℕ} {Vs : Fin S → VStmt} (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) (j : Fin S)
    (u : Fin (Vs j).nv) {c : ℕ} (hc : c < (Rows.ofNet (Vs j).hU).nIn) (hio : IoColAt (Vs j) u c) :
    colOf (Vs j) (maskOf Vs (fun _ _ => Finset.univ) i Xs j) u c = colOf (Vs j) (Xs j) u c := by
  unfold colOf
  by_cases h : 0 < (Vs j).c.upv
  · rw [dite_eq_left h, dite_eq_left h]
    refine (FlockVBridge.copyVal_input _ _ h hc).trans (Eq.trans ?_ (FlockVBridge.copyVal_input _ _ h hc).symm)
    simp only [maskOf]
    exact ite_eq_left ⟨Finset.mem_univ _, u, hio h _⟩
  · rw [dite_eq_right h, dite_eq_right h]

end cols

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}
  {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} {sh : Flock.Shape}

local notation "ℐ" => holoInnerAt T CT st hp

namespace RepMaps

variable (M : RepMaps stmt Vs sh)

/-- Copy 0 of opening `j` of level `ℓ` of rep `r`'s unit, in the layers `l` (`VStarFamily.openCol`). -/
def openCol (l : (j : Fin S) → (Vs j).Bits) (r : Fin 2) (ℓ j : ℕ) : ℕ → Bool :=
  colOf (Vs (M.sessO ℓ)) (l (M.sessO ℓ)) (M.unO ℓ r j)

/-- **Rep `r`'s reads in the layers `l`** at the schedule `sch` (`VStarFamily.ligRead`), its caps `cp`. -/
def ligRead (sch : Flock.Schedule) (cp : ℕ → ℕ → ByteArray) (l : (j : Fin S) → (Vs j).Bits) (r : Fin 2) : LigRead where
  row ℓ j := elems (M.openCol l r ℓ j) 0 (2 ^ (sch.levels.getD ℓ default).lanes)
  sib ℓ j h := bytesAt (M.openCol l r ℓ j) (128 * 2 ^ (sch.levels.getD ℓ default).lanes + 512 * h) 64
  salt _ _ := ByteArray.empty
  cap := cp

/-- **Hypothesis: the openings' `row` and `aux` columns are input columns a row reads** (`OpenKeys`' `ReadCol` form):
at each level `ℓ` of `sch`, every column below `128 · 2^lanes + 512 (d − cap)` of its session's unit. -/
def OpenCols (sch : Flock.Schedule) : Prop :=
  ∀ ℓ < sch.levels.size, ∀ c < 128 * 2 ^ (sch.levels.getD ℓ default).lanes +
      512 * ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap),
    c < (Rows.ofNet (Vs (M.sessO ℓ)).hU).nIn ∧ ReadCol (Rows.ofNet (Vs (M.sessO ℓ)).hU) c

/-- Rep `i`'s opening at level `ℓ`, query `q`, read off the layers `l`. -/
def rdOpen (sch : Flock.Schedule) (i : Fin 2) (ℓ q : ℕ) (l : (j : Fin S) → (Vs j).Bits) : List 𝔽 × List Dig :=
  openRow (ligOfRead 64 none sch (M.ligRead sch (fun _ _ => ByteArray.empty) l i)) sch ℓ q

theorem openCol_mask {sch : Flock.Schedule} (hk : M.OpenCols sch) (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits)
    (r : Fin 2) {ℓ : ℕ} (hℓ : ℓ < sch.levels.size) (j c : ℕ)
    (hc : c < 128 * 2 ^ (sch.levels.getD ℓ default).lanes +
      512 * ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap)) :
    M.openCol (maskOf Vs (fun _ _ => Finset.univ) i Xs) r ℓ j c = M.openCol Xs r ℓ j c :=
  colOf_mask i Xs (M.sessO ℓ) (M.unO ℓ r j) (hk ℓ hℓ c hc).1
    (ioColAt_of_readCol (M.unO ℓ r j) (hk ℓ hℓ c hc).1 (hk ℓ hℓ c hc).2)

/-- **The openings are the same in the masked layer**, at any caps. -/
theorem openRow_mask {sch : Flock.Schedule} (hk : M.OpenCols sch) (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits)
    (r : Fin 2) (cp cp' : ℕ → ℕ → ByteArray) (ℓ q : ℕ) :
    openRow (ligOfRead 64 none sch (M.ligRead sch cp Xs r)) sch ℓ q =
      openRow (ligOfRead 64 none sch (M.ligRead sch cp' (maskOf Vs (fun _ _ => Finset.univ) i Xs) r)) sch ℓ q := by
  refine openRow_ligOfRead_congr (fun ℓ' w => ?_) ℓ q
  by_cases hℓ : ℓ' < sch.levels.size
  · refine levelOf_congr (fun j _ => ?_) (fun j _ _h _hh => ?_)
    · dsimp only [RepMaps.ligRead]
      exact elems_congr fun c _ _h2 => (M.openCol_mask hk i Xs r hℓ j c (by omega)).symm
    · dsimp only [RepMaps.ligRead]
      exact bytesAt_congr fun c _ _h2 => (M.openCol_mask hk i Xs r hℓ j c (by omega)).symm
  · refine levelOf_congr (fun _ hj => absurd hj ?_) (fun _ hj => absurd hj ?_) <;>
      rw [FlockVBridge.Family.Conss.getD_default, getElem!_neg sch.levels ℓ' hℓ] <;> exact Nat.not_lt_zero _

/-- **`SrcOpensOfLayers` at G's values**: at `W` every position, any `b` whose openings are `ligOfRead`'s over the
maps' units (`hb`, at any caps) has them read off the layers by `rdOpen`. -/
theorem srcOpensOfLayers {sch : Flock.Schedule} (hk : M.OpenCols sch)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {cp : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ℕ → ℕ → ByteArray}
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (hb : ∀ Lb i, (b Lb i).sch = sch ∧ (b Lb i).lig = ligOfRead 64 none sch (M.ligRead sch (cp Lb i) Lb i)) :
    SrcOpensOfLayers zs dg N k Rw (fun _ _ => Finset.univ) (M.rdOpen sch) b := by
  intro R ω t R₂ τ _o _hA i ℓ q
  obtain ⟨h1, h2⟩ := hb (layers zs dg N k Rw R ω t R₂ τ) i
  rw [h1, h2]
  exact M.openRow_mask hk i.val _ i _ _ ℓ q

/-- **`OpensReadAt` at reps 0 and 1 at G's values** (`opensReadAt_src` at `srcOpensOfLayers`). -/
theorem opensReadAt {sch : Flock.Schedule} (hk : M.OpenCols sch)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData} {nm : C3Run (holoAt st hp) → C3Run (holoAt st hp)}
    (hnm : ∀ u (i : Fin 2), (nm u).os.1 i = u.os.1 i) {cp : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ℕ → ℕ → ByteArray}
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (hb : ∀ Lb i, (b Lb i).sch = sch ∧ (b Lb i).lig = ligOfRead 64 none sch (M.ligRead sch (cp Lb i) Lb i))
    (i : Fin 2) :
    OpensReadAt zs dg N k Rw
      (tiedSrc fun R ω t Lb => nm (repsOfVals stp sh rep (fun Lb i => (b Lb i).vals sh (rep Lb i)) R ω t Lb)) i :=
  opensReadAt_src hnm (M.srcOpensOfLayers hk (zs := zs) (dg := dg) (N := N) (k := k) (Rw := Rw) hb) i

end RepMaps

end

end FlockC3.Reps
