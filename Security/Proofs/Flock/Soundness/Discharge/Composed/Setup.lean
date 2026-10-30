import Proofs.Flock.Soundness.Discharge.Composed.Coins
import Proofs.Flock.Soundness.Discharge.Composed.Geometry
import Proofs.Flock.Soundness.Discharge.Composed.Pads
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds

/-!
# `coinedOf_setup`, from the refinement's three facts

`coinedOf` fixes the pinned fields of `CoinedOfSetup` (the statement, `RANK_POINTS`, the mask words) and `CoinedOK`'s
coin ties and padding columns by definition, and `coin_facts` gives the rank check, `W^r` and `β ≠ 0` from `Admits`.
What is left reads `bodyOf`, the refinement, and is stated here as three named facts, the third in its own file at every draw:

* `bodyOf_geometry` (proved): C-Flock ZK's level-0 padding geometry from the table's `PadOK`, the opened positions
  distinct and in range by `padPos`.
* `bodyOf_padsOnto` (proved): each rep's pads code reaches every value at its opened positions (`padsOnto_pow` at
  `coinedOf`'s `Cp`, the inner code's padding monomials at the distinct nonzero opened points).
* `innerOf_holds` (`Composed/Inner/Holds.lean`, the inner lane's): the honest prover's pads satisfy the inner proof's constraint at every valid witness whose registered
  values are the public ones (`Table.InnerHolds`; `innerOf`'s `lam`, `theta`, `vals`, `lanes`, `lig`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL FlockLevel3 ZkView

local notation "𝔽" => FlockLevel3.GF128

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

theorem padPos_list (L : ℕ) (qs : List ℕ) :
    let l := (List.range L).filter (fun x => decide (x ∈ qs)) ++ (List.range L).filter (fun x => decide (x ∉ qs))
    l.Nodup ∧ l.length = L ∧ ∀ x ∈ l, x < L := by
  intro l
  refine ⟨?_, ?_, ?_⟩
  · refine List.Nodup.append ((List.nodup_range).filter _) ((List.nodup_range).filter _) ?_
    intro x hx hy
    simp only [List.mem_filter, decide_eq_true_eq] at hx hy
    exact hy.2 hx.2
  · simp only [l, List.length_append]
    have := List.length_eq_length_filter_add (l := List.range L) (fun x => decide (x ∈ qs))
    simpa [decide_not] using this.symm
  · intro x hx
    simp only [l, List.mem_append, List.mem_filter, List.mem_range] at hx
    rcases hx with h | h <;> exact h.1

theorem padPos_injective {L t : ℕ} (qs : List ℕ) (h : t ≤ L) : Function.Injective (padPos L t qs) := by
  obtain ⟨hn, hl, -⟩ := padPos_list L qs
  intro i j hij
  simp only [padPos] at hij
  rw [List.getD_eq_getElem _ _ (by omega), List.getD_eq_getElem _ _ (by omega)] at hij
  exact Fin.ext ((hn.getElem_inj_iff).mp hij)

theorem padPos_lt {L t : ℕ} (qs : List ℕ) (h : t ≤ L) (i : Fin t) : padPos L t qs i < L := by
  obtain ⟨-, hl, hlt⟩ := padPos_list L qs
  simp only [padPos]
  rw [List.getD_eq_getElem _ _ (by omega)]
  exact hlt _ (List.getElem_mem _)

/-- **C-Flock ZK's level-0 padding geometry at the draw**: from the table's `PadOK` (`cols ≤ logLen ≤ 40`, `t ≤ 2^cols`)
and `padPos`. -/
theorem bodyOf_geometry_of (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    C.cols ≤ 40 ∧ C.z.tpad ≤ 2 ^ C.cols ∧ C.z.nS ≤ C.z.tpad ∧
      ∀ c, C.Admits c → Function.Injective (C.pos c) ∧ ∀ q, C.pos c q < 2 ^ (C.cols + 1) := by
  intro C
  have hp := (Tz).hpad
  have hcl : (Tz).sch.level₀.logCols ≤
      (Tz).sch.level₀.logLen :=
    (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).mp (le_trans (Nat.le_add_right _ _) hp.hk)
  have ht2 : (Tz).t ≤ 2 ^ ((Tz).sch.level₀.logCols + 1) :=
    le_trans hp.ht (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ _))
  exact ⟨le_trans hcl hp.hd, hp.ht, le_rfl, fun c _ => ⟨padPos_injective _ ht2, padPos_lt _ ht2⟩⟩

/-- `bodyOf_geometry` at the draw. -/
theorem bodyOf_geometry (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)) :
    let C := coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st
    C.cols ≤ 40 ∧ C.z.tpad ≤ 2 ^ C.cols ∧ C.z.nS ≤ C.z.tpad ∧
      ∀ c, C.Admits c → Function.Injective (C.pos c) ∧ ∀ q, C.pos c q < 2 ^ (C.cols + 1) :=
  bodyOf_geometry_of P _ _

theorem pt_injective {a b : ℕ} (ha : a < 2 ^ 63) (hb : b < 2 ^ 63)
    (h : fOf (⟨(a + 1).toUInt64, 0⟩ : F128) = fOf ⟨(b + 1).toUInt64, 0⟩) : a = b := by
  have h1 : (a + 1).toUInt64 = (b + 1).toUInt64 := congrArg F128.lo h
  have h2 := congrArg UInt64.toNat h1
  norm_num at ha hb
  simp [Nat.toUInt64_eq] at h2
  rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)] at h2
  omega

theorem pt_ne_zero {a : ℕ} (ha : a < 2 ^ 63) : fOf (⟨(a + 1).toUInt64, 0⟩ : F128) ≠ 0 := by
  intro h
  have h1 : (a + 1).toUInt64 = 0 := congrArg F128.lo (h.trans GF128.zero_eq)
  have h2 := congrArg UInt64.toNat h1
  norm_num at ha
  simp [Nat.toUInt64_eq] at h2
  rw [Nat.mod_eq_of_lt (by omega)] at h2
  omega

/-- **The pads onto at the draw**: `padsOnto_pow` at the opened pads points `p + 1`, distinct and nonzero by `padPos`
below `2^63`, `Q ≤ K_PAD`. -/
theorem bodyOf_padsOnto_of (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c, C.Admits c → (C.T c).tab.PadsOnto := by
  intro C c _ r
  let B := bodyOf (Tz) st
  have hQ : Zk.Q ≤ max (min B.padN (2 ^ 63)) Zk.Q := le_max_right _ _
  have hL : max (min B.padN (2 ^ 63)) Zk.Q ≤ 2 ^ 63 := max_le (min_le_right _ _) (by decide)
  have hlt := fun k => lt_of_lt_of_le (padPos_lt (B.padQueries r c.rest) hQ k) hL
  exact padsOnto_pow (F := 𝔽) (nQ := Zk.Q) (nPad := Zk.K_PAD) B.padLen
    (fun k => fOf ⟨(padPos (max (min B.padN (2 ^ 63)) Zk.Q) Zk.Q (B.padQueries r c.rest) k + 1).toUInt64, 0⟩)
    (fun i j h => padPos_injective _ hQ (pt_injective (hlt i) (hlt j) h)) (fun k => pt_ne_zero (hlt k)) (by decide)

/-- `bodyOf_padsOnto` at the draw. -/
theorem bodyOf_padsOnto (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)) :
    let C := coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st
    ∀ c, C.Admits c → (C.T c).tab.PadsOnto :=
  bodyOf_padsOnto_of P _ _

/-- **A `--zk` table of a setup has Lemma B's facts** (`coinedOf_setup` at any table `Tz` over `st`'s statement at its
regions, `st` a `MaskSetup` that avoids the mask slot). -/
theorem coinedOf_setup_of (Tz : TabZK 𝔽 mPts) (st : Stmt) (hwf : StmtWF st) (hS : Tz.S = stmtOf st hwf (regionsOf st))
    (M : MaskSetup st) (hdm : avoidsMask st = true) :
    (coinedOf P Tz st).S = Tz.S ∧ (coinedOf P Tz st).z.nC = Zk.RANK_POINTS ∧
    (coinedOf P Tz st).z.nMask = (maskWordsOf st).length ∧
    (∀ x, (coinedOf P Tz st).maskWords x = (maskWordsOf st).getD x 0) ∧
    Nonempty (coinedOf P Tz st).Rest ∧ CoinedOK execArith (fOf Zk.PAD_C) (coinedOf P Tz st) := by
  obtain ⟨g1, g2, g3, g4⟩ := bodyOf_geometry_of P Tz st
  refine ⟨rfl, rfl, rfl, fun _ => rfl, ⟨(bodyOf _ _).rest0⟩, g1, g2, g3, fun c hc => ?_⟩
  have hβ : ∀ r, ((coinedOf P Tz st).T c).β r = fOf (c.beta r) := fun _ => rfl
  have hρ : ∀ r, ((coinedOf P Tz st).T c).ρ r 0 = GF256.of (c.rho r).1 ∧
      ((coinedOf P Tz st).T c).ρ r 1 = GF256.of (c.rho r).2 :=
    fun _ => ⟨rfl, rfl⟩
  have hw : ∀ k x, ((coinedOf P Tz st).T c).eqw k x =
      fOf (eqWord ((coinedOf P Tz st).points c k) ((coinedOf P Tz st).maskWords x)) :=
    fun _ _ => rfl
  have he : ∀ a : F128, ((coinedOf P Tz st).T c).embed (fOf a) =
      GF256.of ⟨a, 0⟩ := fun _ => rfl
  obtain ⟨hrank, hW, hb⟩ := coin_facts _ c hc hβ hρ hw he
  exact ⟨g4 c hc, ⟨hβ, hρ, hw, he⟩, hrank, fun _ _ => rfl, hW, bodyOf_padsOnto_of P Tz st c hc, hb,
    innerOf_holds_of P Tz st hwf hS M hdm c hc⟩

theorem coinedOf_setup (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (y : DrawSetupZK I mPts S (dj S)), avoidsMask y.x.st = true) :
    CoinedOfSetup (Reg := Reg) P I dj y₀ := by
  intro S R y
  exact coinedOf_setup_of P _ y.x.st y.x.wf.1 (by rw [tablesAtZK_eq dj y₀ S R y]; rfl)
    (maskSetup_of_setupH ((ZkExec.zkTags_typed I).trans ht) y.x.h) (hdm S y)

end FlockSoundness.Discharge.Composed
