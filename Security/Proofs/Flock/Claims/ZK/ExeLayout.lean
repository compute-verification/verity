import Proofs.Flock.Claims.ZK.Round0

/-!
# The masked session's tape at the executable's indices

`flock-circuit --zk`'s tape (`FlockComplete.TapeZk`) is indexed by table, rep, level and position, and some of its
indices depend on the statement: extra lane `e`'s padding is lane `2^lanes + e`'s (`ligeritoPZk`'s `ybar`), and the
pads `h_x, h_y` sit at `n_E + 1, n_E + 2` of the pads layout (`padsOf`, `Composed.padIdx`). So the tape is read
from cells at the executable's own indices, and each statement reads its tables' coordinates at the indices its
sizes give.

* `ExeCell`: one table's cell, each `TapeZk` field the table draws as an array at its indices up to the live tables'
  largest sizes (`zMax`, `schMax`): the mask slot's words, the lane paddings by lane (`2^lanes + 4` lanes), the extra
  lanes, each rep's pads `h`, `μ` and their padding, level 0's salts by tree position, the pads trees' and `τ`'s salts,
  and each rep's levels ≥ 1 salts by level and position; besides them, the coordinates the honest prover never reads
  (the revealed salts `RefSession.baseWith` overwrites, the levels ≥ 1 salts no position takes, the ideal seeds).
* `exeRead`: a statement's reading of a cell as one table's randomness, every coordinate at its executable index: an
  injection of places, so with fibers of one size (`fibersEq_exeRead`; `fibersEq_pair` where two coordinates share one
  array). Levels ≥ 1's salt `k` of rep `r` is the salt at the level and position `hiIdx` puts at `k` (`hiEmbOf`).
* `TapeLayout.exe`, `exeTape`: the layout, and the masked prover's tape read from the cells (`exeTape`), its PoW
  nonces `0` (the live device grinds with nonce `0`, `firewall.rs`'s `grind_pow`).
* `agrees0_exe`: the layout agrees with the executable at round `0` (`Agrees0`), so round `0`'s honest message is
  `enc0`'s (`honest_round0_exe`), and `HonestPreOn` at that layout is its rounds after the first
  (`honestPreOn_exe_of_pos`).
* `Flock.zk_deployZk_masked_exe_gap`: `Flock.zk_deployZk_masked_layout_gap` at this layout, round `0` discharged.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

local notation "𝔽" => FlockLevel3.GF128

/-! ## Two readings of one array -/

/-- **Reading one function at two injections' disjoint points has fibers of one size**: one injection of their sum. -/
theorem fibersEq_pair {A B C Cod : Type} [Finite A] [Finite Cod] (ι₁ : B ↪ A) (ι₂ : C ↪ A)
    (hd : ∀ b c, ι₁ b ≠ ι₂ c) : FibersEq fun f : A → Cod => (f ∘ ι₁, f ∘ ι₂) := by
  let ι : B ⊕ C ↪ A := ⟨Sum.elim ι₁ ι₂, by
    rintro (a | a) (b | b) h
    · exact congrArg Sum.inl (ι₁.injective h)
    · exact absurd h (hd a b)
    · exact absurd h.symm (hd b a)
    · exact congrArg Sum.inr (ι₂.injective h)⟩
  exact (fibersEq_comp_emb (Cod := Cod) ι).comp
    (fibersEq_of_bijective (Equiv.sumArrowEquivProdArrow B C Cod).bijective)

/-- **A restriction then a bijection has fibers of one size.** -/
theorem fibersEq_emb_bij {A D Cod Out : Type} [Finite A] [Finite Cod] (ι : D ↪ A) {e : (D → Cod) → Out}
    (he : Function.Bijective e) : FibersEq fun f : A → Cod => e (f ∘ ι) :=
  (fibersEq_comp_emb (Cod := Cod) ι).comp (fibersEq_of_bijective he)

namespace Masked

/-! ## The cell -/

/-- **A table's base draws at the executable's indices**: the mask slot's `2^35` words; the lane paddings by lane
(`schMax + 4` lanes of `schMax` coefficients) and the extra lanes; each rep's pads `h` by layout index; each rep's
`pad_h`, `pad_μ`, and `μ` by layout index; the revealed salts (level 0's, the pads trees', `y_τ`); each rep's levels ≥ 1
salts by level and position, and the leftover ones. -/
abbrev ExeBase : Type :=
  (Fin (2 ^ 35) → Fin 128 → ZMod 2) ×
    ((Fin (schMax + 4) → Fin schMax → 𝔽) × (Fin 2 → Fin 2 → Fin schMax → 𝔽)) ×
    (Fin 2 → Fin (2 ^ 64 + 4) → 𝔽) ×
    ((Fin 2 → Fin _root_.Flock.Zk.K_PAD → 𝔽) × (Fin 2 → Fin _root_.Flock.Zk.K_PAD → 𝔽) ×
      (Fin 2 → Fin (2 ^ 64 + 4) → 𝔽) ×
      ((Fin schMax → ZkView.Salt) × (Fin 2 → Fin _root_.Flock.Zk.Q → ZkView.Salt) × (Fin 2 → ZkView.Salt) ×
        (Fin 2 → (Fin (schMax + 1) × Fin schMax → ZkView.Salt) × (Fin schMax → ZkView.Salt))))

/-- **A table's cell**: its base draws; level 0's salts by tree position, each rep's pads tree's salts by position and
`τ`'s salt; the ideal seeds. -/
abbrev ExeCell : Type :=
  ExeBase × ((Fin (schMax + schMax) → ZkView.Salt) × (Fin 2 → Fin (_root_.Flock.Zk.Q + 2 ^ 64) → ZkView.Salt) ×
    (Fin 2 → ZkView.Salt)) × (ZkView.LeafPos zMax → ZkView.Bits × ZkView.Salt)

/-! ## A statement's reading of a cell -/

section Read

variable {z : ZkView.Sizes}

theorem Fits.nL' (hz : Fits z zMax) : z.nL ≤ schMax := hz.nL
theorem Fits.nE' (hz : Fits z zMax) : z.nE ≤ 2 ^ 64 := hz.nE

/-- Lane `l < n_L` at its index. -/
def laneEmb (hz : Fits z zMax) : Fin z.nL ↪ Fin (schMax + 4) := castEmb (hz.nL.trans (Nat.le_add_right _ _))

/-- **Extra lane `(r, i)`'s padding at lane `n_L + 2r + i`** (`ligeritoPZk`'s `ybar`, lane `2^lanes + 2·rep + i`). -/
def extEmb (hz : Fits z zMax) : Fin 2 × Fin 2 ↪ Fin (schMax + 4) :=
  ⟨fun q => ⟨z.nL + (2 * q.1.val + q.2.val), by have := hz.nL'; have := q.1.isLt; have := q.2.isLt; omega⟩,
    fun a b h => by
      have h' : 2 * a.1.val + a.2.val = 2 * b.1.val + b.2.val := by
        have := congrArg Fin.val h
        simp only at this
        omega
      exact Prod.ext (Fin.ext (by have := a.2.isLt; have := b.2.isLt; omega))
        (Fin.ext (by have := a.2.isLt; have := b.2.isLt; omega))⟩

theorem laneEmb_ne_extEmb (hz : Fits z zMax) (l : Fin z.nL) (q : Fin 2 × Fin 2) :
    laneEmb hz l ≠ extEmb hz q := fun h => by
  have e : l.val = z.nL + (2 * q.1.val + q.2.val) := congrArg Fin.val h
  have := l.isLt
  omega

/-- **The lanes' paddings and the extra lanes**, from one array of lane paddings and the extra lanes. -/
def readLanes (hz : Fits z zMax) (p : (Fin (schMax + 4) → Fin schMax → 𝔽) × (Fin 2 → Fin 2 → Fin schMax → 𝔽)) :
    (Fin z.nL → Fin z.tpad → 𝔽) × (Fin 2 → Fin 2 → Fin z.L → 𝔽) × (Fin 2 → Fin 2 → Fin z.tpad → 𝔽) :=
  (fun l => cut hz.tpad (p.1 (laneEmb hz l)), fun r i => cut hz.L (p.2 r i),
    fun r i => cut hz.tpad (p.1 (extEmb hz (r, i))))

theorem fibersEq_readLanes (hz : Fits z zMax) : FibersEq (readLanes hz) := by
  have h1 : FibersEq (Prod.map (fun A : Fin (schMax + 4) → Fin schMax → 𝔽 => (A ∘ laneEmb hz, A ∘ extEmb hz))
      (id : (Fin 2 → Fin 2 → Fin schMax → 𝔽) → _)) :=
    (fibersEq_pair _ _ (laneEmb_ne_extEmb hz)).prodMap fibersEq_id
  have h2 : FibersEq (Prod.map
      (Prod.map (fun (a : Fin z.nL → Fin schMax → 𝔽) l => cut hz.tpad (a l))
        (fun (b : Fin 2 × Fin 2 → Fin schMax → 𝔽) q => cut hz.tpad (b q)))
      (fun (e : Fin 2 → Fin 2 → Fin schMax → 𝔽) r i => cut hz.L (e r i))) :=
    ((FibersEq.piConst (fibersEq_cut _)).prodMap (FibersEq.piConst (fibersEq_cut _))).prodMap
      (FibersEq.piConst (FibersEq.piConst (fibersEq_cut _)))
  have h3 : Function.Bijective fun (p : ((Fin z.nL → Fin z.tpad → 𝔽) × (Fin 2 × Fin 2 → Fin z.tpad → 𝔽)) ×
      (Fin 2 → Fin 2 → Fin z.L → 𝔽)) => (p.1.1, p.2, Function.curry p.1.2) :=
    ⟨fun a b h => by
      simp only [Prod.mk.injEq] at h
      obtain ⟨h1, h2, h3⟩ := h
      exact Prod.ext (Prod.ext h1 (by simpa using congrArg Function.uncurry h3)) h2,
     fun q => ⟨((q.1, Function.uncurry q.2.2), q.2.1), by simp⟩⟩
  exact (h1.comp h2).comp (fibersEq_of_bijective h3)

/-- Pad `k < n_E` at its index, `h_x, h_y` at `n_E + 1, n_E + 2` (`Composed.padIdx`). -/
def hEmb (hz : Fits z zMax) : Fin z.nE ⊕ Fin 2 ↪ Fin (2 ^ 64 + 4) :=
  ⟨fun q => match q with
    | .inl k => ⟨k.val, by have := hz.nE'; have := k.isLt; omega⟩
    | .inr b => ⟨z.nE + 1 + b.val, by have := hz.nE'; have := b.isLt; omega⟩,
    by
      rintro (a | a) (b | b) h <;> have h' := congrArg Fin.val h <;> simp only at h'
      · exact congrArg Sum.inl (Fin.ext h')
      · have := a.isLt; omega
      · have := b.isLt; omega
      · exact congrArg Sum.inr (Fin.ext (by omega))⟩

/-- **Each rep's free pads** from its pads by layout index: the exposed ones and `h_x, h_y`. -/
def readH (hz : Fits z zMax) (a : Fin 2 → Fin (2 ^ 64 + 4) → 𝔽) : Fin 2 → (Fin z.nE → 𝔽) × 𝔽 × 𝔽 :=
  fun r => (fun k => a r (hEmb hz (.inl k)), a r (hEmb hz (.inr 0)), a r (hEmb hz (.inr 1)))

theorem fibersEq_readH (hz : Fits z zMax) : FibersEq (readH hz) := by
  have e : Function.Bijective fun (g : Fin z.nE ⊕ Fin 2 → 𝔽) => (g ∘ Sum.inl, g (.inr 0), g (.inr 1)) :=
    ((Equiv.sumArrowEquivProdArrow _ _ _).trans (Equiv.prodCongr (Equiv.refl _) (piFinTwoEquiv fun _ => 𝔽))).bijective
  exact FibersEq.piConst (fibersEq_emb_bij (hEmb hz) e)

/-- A pad of the layout at its index (`Composed.padIdx`). -/
def padEmb' (hz : Fits z zMax) : PadIdx z.nE ↪ Fin (2 ^ 64 + 4) :=
  ⟨fun p => ⟨Composed.padIdx p, by
      have := hz.nE'
      rcases p with k | j
      · simp only [Composed.padIdx]; have := k.isLt; omega
      · simp only [Composed.padIdx]; fin_cases j <;> simp <;> omega⟩,
    by
      rintro (a | a) (b | b) h <;> have h' := congrArg Fin.val h <;> simp only [Composed.padIdx] at h'
      · exact congrArg Sum.inl (Fin.ext h')
      · have := a.isLt; fin_cases b <;> simp at h' <;> omega
      · have := b.isLt; fin_cases a <;> simp at h' <;> omega
      · fin_cases a <;> fin_cases b <;> simp at h' ⊢⟩

/-- **The tables' base at a statement's sizes**, every coordinate at its executable index; levels ≥ 1's salts through
`ιH`. -/
def exeBase (hz : Fits z zMax) {h : ℕ} (ιH : Fin h ↪ (Fin (schMax + 1) × Fin schMax) ⊕ Fin schMax) :
    ExeBase → ZkView.Base z (Fin h → ZkView.Salt) :=
  Prod.map (cut hz.nMask)
    (Prod.map (readLanes hz)
      (Prod.map (readH hz)
        (Prod.map (fun a r => cut hz.nPad (a r))
          (Prod.map (fun a r => cut hz.nPad (a r))
            (Prod.map (fun a r => a r ∘ padEmb' hz)
              (Prod.map (cut hz.nS)
                (Prod.map (fun a r => cut hz.nQ (a r))
                  (Prod.map id (fun a r => Sum.elim (a r).1 (a r).2 ∘ ιH)))))))))

theorem fibersEq_exeBase (hz : Fits z zMax) {h : ℕ} (ιH : Fin h ↪ (Fin (schMax + 1) × Fin schMax) ⊕ Fin schMax) :
    FibersEq (exeBase hz ιH) := by
  have hH : FibersEq fun (a : Fin 2 → (Fin (schMax + 1) × Fin schMax → ZkView.Salt) × (Fin schMax → ZkView.Salt))
      (r : Fin 2) => Sum.elim (a r).1 (a r).2 ∘ ιH :=
    FibersEq.piConst ((fibersEq_of_bijective (Equiv.sumArrowEquivProdArrow _ _ _).symm.bijective).comp
      (fibersEq_comp_emb ιH))
  exact (fibersEq_cut _).prodMap ((fibersEq_readLanes hz).prodMap ((fibersEq_readH hz).prodMap
    ((FibersEq.piConst (fibersEq_cut _)).prodMap ((FibersEq.piConst (fibersEq_cut _)).prodMap
      ((FibersEq.piConst (fibersEq_comp_emb _)).prodMap ((fibersEq_cut _).prodMap
        ((FibersEq.piConst (fibersEq_cut _)).prodMap (fibersEq_id.prodMap hH))))))))

/-- **The pre-final leaves' salts at a statement's sizes**: level 0's by tree position, each rep's pads tree's by
position, and `τ`'s. -/
def exeSalts (hz : Fits z zMax)
    (p : (Fin (schMax + schMax) → ZkView.Salt) × (Fin 2 → Fin (_root_.Flock.Zk.Q + 2 ^ 64) → ZkView.Salt) ×
      (Fin 2 → ZkView.Salt)) : ZkView.LeafPos z → ZkView.Salt :=
  Sum.elim (cut (Nat.add_le_add hz.nS hz.nU) p.1)
    (Sum.elim (fun q => p.2.1 q.1 (Fin.castLE (Nat.add_le_add hz.nQ hz.nPU) q.2)) p.2.2)

theorem fibersEq_exeSalts (hz : Fits z zMax) : FibersEq (exeSalts hz) := by
  have h1 : FibersEq (Prod.map (cut (Cod := ZkView.Salt) (Nat.add_le_add hz.nS hz.nU))
      (Prod.map (fun (b : Fin 2 → Fin (_root_.Flock.Zk.Q + 2 ^ 64) → ZkView.Salt) r =>
        cut (Nat.add_le_add hz.nQ hz.nPU) (b r)) (id : (Fin 2 → ZkView.Salt) → _))) :=
    (fibersEq_cut _).prodMap ((FibersEq.piConst (fibersEq_cut _)).prodMap fibersEq_id)
  have h2 : Function.Bijective fun (p : (Fin (z.nS + z.nU) → ZkView.Salt) ×
      (Fin 2 → Fin (z.nQ + z.nPU) → ZkView.Salt) × (Fin 2 → ZkView.Salt)) =>
      Sum.elim p.1 (Sum.elim (Function.uncurry p.2.1) p.2.2) :=
    ((Equiv.prodCongr (Equiv.refl _) ((Equiv.prodCongr (Equiv.curry _ _ _).symm (Equiv.refl _)).trans
      (Equiv.sumArrowEquivProdArrow _ _ _).symm)).trans (Equiv.sumArrowEquivProdArrow _ _ _).symm).bijective
  exact h1.comp (fibersEq_of_bijective h2)

/-- **A table's randomness at a statement's sizes from a cell**: its base, its leaves' salts, its ideal seeds. -/
def exeRead (hz : Fits z zMax) {h : ℕ} (ιH : Fin h ↪ (Fin (schMax + 1) × Fin schMax) ⊕ Fin schMax) :
    ExeCell → TabRand z (Fin h → ZkView.Salt) :=
  Prod.map (exeBase hz ιH) (Prod.map (exeSalts hz) fun f => f ∘ leafEmb hz)

theorem fibersEq_exeRead (hz : Fits z zMax) {h : ℕ} (ιH : Fin h ↪ (Fin (schMax + 1) × Fin schMax) ⊕ Fin schMax) :
    FibersEq (exeRead hz ιH) :=
  (fibersEq_exeBase hz ιH).prodMap ((fibersEq_exeSalts hz).prodMap (fibersEq_comp_emb _))

end Read

/-! ## Levels ≥ 1's salts by level and position -/

/-- Where level `l ≥ 1`'s salts start among the levels ≥ 1's: the positions of the levels before it. -/
def hiOff (ls : List FlockSoundness.Accounting.Level) (l : ℕ) : ℕ :=
  ((ls.tail.take (l - 1)).map fun lv => 2 ^ lv.logLen).sum

/-- A level `≥ 1` of the schedule and a position of its tree. -/
def HiV (ls : List FlockSoundness.Accounting.Level) (q : Fin (schMax + 1) × Fin schMax) : Prop :=
  1 ≤ q.1.val ∧ q.1.val < ls.length ∧ q.2.val < 2 ^ (ls.getD q.1.val default).logLen

/-- **Level `l`'s salt at position `p`**, levels ≥ 1's salts in level order. -/
def hiIdx (ls : List FlockSoundness.Accounting.Level) (q : Fin (schMax + 1) × Fin schMax) : ℕ :=
  hiOff ls q.1.val + q.2.val

theorem hiOff_succ (ls : List FlockSoundness.Accounting.Level) {l : ℕ} (h1 : 1 ≤ l) (hl : l < ls.length) :
    hiOff ls (l + 1) = hiOff ls l + 2 ^ (ls.getD l default).logLen := by
  unfold hiOff
  have ht : l - 1 < ls.tail.length := by rw [List.length_tail]; omega
  have e : ls.tail[l - 1] = ls.getD l default := by
    rw [List.getElem_tail, List.getD_eq_getElem _ _ hl]
    congr 1
    omega
  rw [show l + 1 - 1 = (l - 1) + 1 by omega, List.take_add_one, List.getElem?_eq_getElem ht, e]
  simp only [Option.toList_some, List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, add_zero]

theorem hiOff_mono (ls : List FlockSoundness.Accounting.Level) {l l' : ℕ} (h : l ≤ l') : hiOff ls l ≤ hiOff ls l' := by
  unfold hiOff
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  have e : l + d - 1 = (l - 1) + (l + d - 1 - (l - 1)) := by omega
  rw [e, List.take_add, List.map_append, List.sum_append]
  exact Nat.le_add_right _ _

theorem hiIdx_injOn (ls : List FlockSoundness.Accounting.Level) {q q' : Fin (schMax + 1) × Fin schMax}
    (hq : HiV ls q) (hq' : HiV ls q') (h : hiIdx ls q = hiIdx ls q') : q = q' := by
  obtain ⟨h1, hl, hp⟩ := hq
  obtain ⟨h1', hl', hp'⟩ := hq'
  unfold hiIdx at h
  have key : ∀ {a b : Fin (schMax + 1) × Fin schMax}, 1 ≤ a.1.val → a.1.val < ls.length →
      a.2.val < 2 ^ (ls.getD a.1.val default).logLen → a.1.val < b.1.val →
      hiOff ls a.1.val + a.2.val < hiOff ls b.1.val + b.2.val := fun {a b} ha hla hpa hab => by
    have := hiOff_succ ls ha hla
    have := hiOff_mono ls (show a.1.val + 1 ≤ b.1.val from hab)
    omega
  rcases Nat.lt_trichotomy q.1.val q'.1.val with hlt | heq | hgt
  · exact absurd h (key h1 hl hp hlt).ne
  · rw [heq] at h
    exact Prod.ext (Fin.ext heq) (Fin.ext (by omega))
  · exact absurd h.symm (key h1' hl' hp' hgt).ne

open Classical in
/-- **Levels ≥ 1's salt `k`'s place**: the level and position `hiIdx` puts at `k`, else its own leftover place. -/
noncomputable def hiEmbOf (ls : List FlockSoundness.Accounting.Level) {n : ℕ} (hn : n ≤ schMax) :
    Fin n ↪ (Fin (schMax + 1) × Fin schMax) ⊕ Fin schMax :=
  ⟨fun k => if h : ∃ q, HiV ls q ∧ hiIdx ls q = k.val then .inl h.choose else .inr (Fin.castLE hn k), by
    intro a b hab
    simp only at hab
    split_ifs at hab with ha hb hb <;> simp only [Sum.inl.injEq, Sum.inr.injEq] at hab
    · exact Fin.ext (ha.choose_spec.2.symm.trans ((congrArg (hiIdx ls) hab).trans hb.choose_spec.2))
    · exact Fin.ext (by simpa using congrArg Fin.val hab)⟩

theorem hiEmbOf_hiIdx (ls : List FlockSoundness.Accounting.Level) {n : ℕ} (hn : n ≤ schMax)
    {q : Fin (schMax + 1) × Fin schMax} (hq : HiV ls q) (hk : hiIdx ls q < n) :
    hiEmbOf ls hn ⟨hiIdx ls q, hk⟩ = .inl q := by
  classical
  have h : ∃ q', HiV ls q' ∧ hiIdx ls q' = hiIdx ls q := ⟨q, hq, rfl⟩
  show (if h : ∃ q', HiV ls q' ∧ hiIdx ls q' = hiIdx ls q then Sum.inl h.choose else _) = _
  rw [dite_eq_left h]
  exact congrArg Sum.inl (hiIdx_injOn ls h.choose_spec.1 hq h.choose_spec.2)

/-! ## The layout and the tape -/

/-- Table `j`'s levels ≥ 1's salts' places at `x`. -/
noncomputable def hiEmb (x : Msg) (j : Fin (jOf x)) :
    Fin (Composed.hiCount (zkTab x j).table) ↪ (Fin (schMax + 1) × Fin schMax) ⊕ Fin schMax :=
  hiEmbOf (zkTab x j).table.sch.levels (hi_le (zkTab x j))

/-- **The executable's layout**: a cell at the executable's indices (`ExeCell`), each statement's table reading every
coordinate at the index its sizes give (`exeRead`); the mask words the cell's slot bits, packed. -/
noncomputable def TapeLayout.exe (P : Composed.LigProver) : TapeLayout P where
  Cell := ExeCell
  read x j := exeRead (fits_zkTab P (zkTab x j)) (hiEmb x j)
  fibers x j := fibersEq_exeRead _ _
  maskAt c k := if hk : k < 2 ^ 35 then wordOfBits (c.1.1 ⟨k, hk⟩) else 0
  mask_read x j c k hk := by
    have hk' : k < 2 ^ 35 := lt_of_lt_of_le hk (fits_zkTab P (zkTab x j)).nMask
    simp only [hk', ↓reduceDIte]
    rfl

/-- An array's entry `k`, `d` past its end. -/
def fAt {α : Type} {n : ℕ} (d : α) (f : Fin n → α) (k : ℕ) : α := if h : k < n then f ⟨k, h⟩ else d

theorem fAt_lt {α : Type} {n : ℕ} (d : α) (f : Fin n → α) {k : ℕ} (h : k < n) : fAt d f k = f ⟨k, h⟩ := dite_eq_left h

/-- Extra lane `e = 2r + i` of `E`, `d` past the four. -/
def ext4 {α : Type} (E : Fin 2 → Fin 2 → α) (d : α) (e : ℕ) : α :=
  if e = 0 then E 0 0 else if e = 1 then E 0 1 else if e = 2 then E 1 0 else if e = 3 then E 1 1 else d

theorem ext4_eq {α : Type} (E : Fin 2 → Fin 2 → α) (d : α) (r i : Fin 2) : ext4 E d (2 * r.val + i.val) = E r i := by
  fin_cases r <;> fin_cases i <;> rfl

/-- Cell `i` of the tape, an arbitrary one past the cap. -/
noncomputable def cellAt (t : Fin Claims.Flock.zkTableCap → ExeCell) (i : ℕ) : ExeCell :=
  if h : i < Claims.Flock.zkTableCap then t ⟨i, h⟩ else Classical.ofNonempty

theorem cellAt_lt (t : Fin Claims.Flock.zkTableCap → ExeCell) {i : ℕ} (h : i < Claims.Flock.zkTableCap) :
    cellAt t i = t ⟨i, h⟩ := dite_eq_left h

/-- **Table `i`'s salts** (`Tape.salt i a l p`): level 0's by tree position (rep `0`), rep `a`'s levels `l ≥ 1` by level
and position. -/
def saltAt (c : ExeCell) (a l p : ℕ) : ByteArray :=
  if l = 0 then (if a = 0 then fAt ByteArray.empty (fun q => ZkView.saltBytes (c.2.1.1 q)) p else ByteArray.empty)
  else fAt ByteArray.empty (fun r => fAt ByteArray.empty (fun l' => fAt ByteArray.empty
    (fun p' => ZkView.saltBytes ((c.1.2.2.2.2.2.2.2.2.2 r).1 (l', p'))) p) l) a

/-- **The masked prover's tape read from the cells**, each field at its executable index: the salts (`saltAt`), PoW
nonces `0`, each rep's pads, `μ` and their padding, `τ`'s and the pads trees' salts, the extra lanes (`2r + i`) and the
lane paddings. Its mask words are the layout's (`TapePlan.ofLayout`). -/
noncomputable def exeTape (t : Fin Claims.Flock.zkTableCap → ExeCell) : FlockComplete.TapeZk where
  salt i a l p := saltAt (cellAt t i) a l p
  nonce _ _ _ _ := 0
  h i r k := fAt 0 (fun r => fAt 0 ((cellAt t i).1.2.2.1 r) k) r
  mu i r k := fAt 0 (fun r => fAt 0 ((cellAt t i).1.2.2.2.2.2.1 r) k) r
  padH i r k := fAt 0 (fun r => fAt 0 ((cellAt t i).1.2.2.2.1 r) k) r
  padMu i r k := fAt 0 (fun r => fAt 0 ((cellAt t i).1.2.2.2.2.1 r) k) r
  tauSalt i r := fAt ByteArray.empty (fun r => ZkView.saltBytes ((cellAt t i).2.1.2.2 r)) r
  innerSalt i r p := fAt ByteArray.empty (fun r => fAt ByteArray.empty
    (fun p => ZkView.saltBytes ((cellAt t i).2.1.2.1 r p)) p) r
  extra i e c := ext4 (fun r i' => fAt 0 ((cellAt t i).1.2.1.2 r i') c) 0 e
  lanePad i l k := fAt 0 (fun l => fAt 0 ((cellAt t i).1.2.1.1 l) k) l
  mask _ _ := 0

/-! ## Round `0` at the executable's layout -/

theorem saltBytes_size (y : ZkView.Salt) : (ZkView.saltBytes y).size = 192 := by
  simp [ZkView.saltBytes, ZkView.bytesOf, ByteArray.size]

theorem fitBytes_of_size {n : ℕ} {b : ByteArray} (h : b.size = n) : FlockComplete.fitBytes n b = b := by
  subst h
  unfold FlockComplete.fitBytes
  rw [ByteArray.extract_zero_size, Nat.min_self, Nat.sub_self]
  exact ByteArray.append_empty

/-- A cell's lane padding at lane `a` (index `l`), coefficient `b`, is `exeTape`'s `lanePad`. -/
theorem lanePad_at (t : Fin Claims.Flock.zkTableCap → ExeCell) {j : ℕ} (hc : j < Claims.Flock.zkTableCap)
    (a : Fin (schMax + 4)) {l : ℕ} (ha : a.val = l) (b : Fin schMax) :
    (t ⟨j, hc⟩).1.2.1.1 a b =
      fAt (0 : _root_.Flock.F128) (fun l' => fAt (0 : _root_.Flock.F128) ((cellAt t j).1.2.1.1 l') b.val) l := by
  subst ha
  simp only [fAt, cellAt, hc, a.isLt, b.isLt, ↓reduceDIte]

/-- A cell's extra lane `(r, i)`, column `b`, is `exeTape`'s `extra` at `2r + i`. -/
theorem extra_at (t : Fin Claims.Flock.zkTableCap → ExeCell) {j : ℕ} (hc : j < Claims.Flock.zkTableCap)
    (r i : Fin 2) (b : Fin schMax) :
    (t ⟨j, hc⟩).1.2.1.2 r i b = ext4 (fun r i' => fAt (0 : _root_.Flock.F128) ((cellAt t j).1.2.1.2 r i') b.val)
      (0 : _root_.Flock.F128) (2 * r.val + i.val) := by
  rw [ext4_eq]
  simp only [fAt, cellAt, hc, b.isLt, ↓reduceDIte]

/-- **The executable's layout agrees with the executable at round `0`**: lane `l`'s padding, extra lane `2r + i` and its
padding at lane `2^lanes + 2r + i`, and level 0's salt at each tree position are the tape's at their indices. -/
theorem agrees0_exe (P : Composed.LigProver) : Agrees0 P (TapeLayout.exe P) exeTape where
  pad x t j l k := by
    obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = t := ⟨t, rfl⟩
    have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
    have hz := fits_zkTab P (zkTab x j)
    change (t ⟨j.val, hc⟩).1.2.1.1 (laneEmb hz l) (Fin.castLE hz.tpad k) =
      fAt (0 : _root_.Flock.F128) (fun l' => fAt (0 : _root_.Flock.F128) ((cellAt t j.val).1.2.1.1 l') k.val) l.val
    exact lanePad_at t hc _ rfl _
  ext x t j r i c := by
    obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = t := ⟨t, rfl⟩
    have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
    have hz := fits_zkTab P (zkTab x j)
    change (t ⟨j.val, hc⟩).1.2.1.2 r i (Fin.castLE hz.L c) =
      ext4 (fun r i' => fAt (0 : _root_.Flock.F128) ((cellAt t j.val).1.2.1.2 r i') c.val) (0 : _root_.Flock.F128)
        (2 * r.val + i.val)
    exact extra_at t hc r i _
  extPad x t j r i k := by
    obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = t := ⟨t, rfl⟩
    have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
    have hz := fits_zkTab P (zkTab x j)
    change (t ⟨j.val, hc⟩).1.2.1.1 (extEmb hz (r, i)) (Fin.castLE hz.tpad k) =
      fAt (0 : _root_.Flock.F128) (fun l' => fAt (0 : _root_.Flock.F128) ((cellAt t j.val).1.2.1.1 l') k.val)
        (2 ^ (lv0 x j).lanes + (2 * r.val + i.val))
    exact lanePad_at t hc _ (congrArg (· + (2 * r.val + i.val)) (tabC_z P x j).1) _
  salt x t j p := by
    obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = t := ⟨t, rfl⟩
    have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
    have hz := fits_zkTab P (zkTab x j)
    have hp : p.val < schMax + schMax := lt_of_lt_of_le p.isLt (Nat.add_le_add hz.nS hz.nU)
    change ZkView.saltBytes ((t (Fin.castLE (jOf_le_cap x) j)).2.1.1 (Fin.castLE (Nat.add_le_add hz.nS hz.nU) p)) =
      FlockComplete.fitBytes 192 (saltAt (cellAt t j.val) 0 0 p.val)
    rw [cellAt_lt t hc]
    unfold saltAt
    rw [ite_eq_left rfl, ite_eq_left rfl, fAt_lt _ _ hp, fitBytes_of_size (saltBytes_size _)]
    rfl

open Classical in
/-- **Round `0`'s honest message is `enc0`'s at the executable's layout** (`honest_round0` at `agrees0_exe`). -/
theorem honest_round0_exe (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) (P : Composed.LigProver)
    {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) :
    enc0 P x cz (cutAt (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
        (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) =
      (((FlockComplete.ZkTables.honestPR w).next x []
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t)).eval H).1 :=
  honest_round0 H hH P (TapeLayout.exe P) exeTape (agrees0_exe P) hx w t cz

/-- **`HonestPreOn` after the first round**: at a pre-final round after at least one opening
(`0 < ops.length < refK x`), `enc` of the cut pre-final view is `honestPR`'s next message. -/
def _root_.Claims.ZKMal.RefSession.HonestPreOnPos (S : RefSession) (A : Msg → Prop) (R : Rel Claims.Flock.Wit)
    (H : Msg → Msg) (coinOf : Msg → Msg) (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w, R x w → ∀ ν t (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
    S.enc x (S.pad (ops.map fun p => S.decOf p.2)) ops.length
        (S.cutOf x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w) t) =
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

open Classical in
/-- **`HonestPreOn` at the executable's layout is its rounds after the first**, at a round plan whose round `0` is
`enc0` and where `hOf H` is SHA-512: round `0` is `honest_round0_exe`. -/
theorem honestPreOn_exe_of_pos (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) (nr : Msg → ℕ → ℕ)
    (P : Composed.LigProver) (Q : RoundPlan (tabC P))
    (h0 : ∀ x c cut, Q.enc x c 0 cut = enc0 P x (Q.cz x c) cut)
    (hpos : (session nr (tableData P Q (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestPreOnPos Live RZk H
      coinOfD fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2) :
    (session nr (tableData P Q (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestPreOn Live RZk H coinOfD
      fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2 := by
  intro x hx w hR ν t ops hops
  rcases ops with _ | ⟨o, ops⟩
  · show Q.enc x _ 0 _ = _
    exact (h0 x _ _).trans (honest_round0_exe H hH P hx w t _)
  · exact hpos x hx w hR ν t _ (Nat.succ_pos _) hops

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout**:
`Flock.zk_deployZk_masked_layout_gap` at `TapeLayout.exe` and `exeTape`, `hOf H` SHA-512, at a round plan whose round
`0` is `enc0`, so `HonestPreOn` is asked after the first round only (`honestPreOn_exe_of_pos`). Left: the round plan's
`Causal`, the rounds after the first, and the last round. -/
theorem Flock.zk_deployZk_masked_exe_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver) (Q : RoundPlan (tabC P))
    (h0 : ∀ x c cut, Q.enc x c 0 cut = enc0 P x (Q.cz x c) cut) (t : ℕ) (ht : 0 < t) (hc : Q.Causal)
    (hpos : (session nr (tableData P Q (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestPreOnPos Live RZk H
      coinOfD fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2)
    (hfin : (session nr (tableData P Q (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestFinOn Live RZk H
      coinOfD fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2)
    (hoff : (session nr (tableData P Q (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestOffOn Live RZk H
      coinOfD fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P Q (TapePlan.ofLayout (TapeLayout.exe P) exeTape)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_layout_gap H nr P Q (TapeLayout.exe P) exeTape t ht hc
    (honestPreOn_exe_of_pos H hH nr P Q h0 hpos) hfin hoff

end Claims.ZKMal
