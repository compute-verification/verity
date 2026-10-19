import Proofs.Flock.Soundness.Discharge.ZkLink.BadZ
import Proofs.Flock.Soundness.Discharge.ZkLink.ChooseZ

/-!
# A cap's openings on accepting runs

A rep's openings at a level's cap are read in table `n` of the session's outputs only where tables `0, …, n` accept
(`AccUpTo`, `opRAccZ`): a run that rejects a table skips the rest of that table's Ligerito, so its later caps are not
at the coin counts an accepting run has them at, and a fork event at a cap that counts accepting runs only is read at
the round the session actually reached. The rewinding at a cap is then over the plurality of the filtered openings
(`plurAccZ`: a given chooser at the pads and `τ` slots, `s < 4`), and since the session's fork probabilities do not
read the chooser (`forkE_lockSessZ_choose`):

* `leafE_off_le_sessZ`: under any chooser whose table at a slot is a plurality of the slot's openings, the mass of runs
  off that table is at most `√(|P|·Q·fork)`, the fork at the slot's node under any chooser at level-0 oracle `default`;
* `ind_capAcc_eq`: on runs where tables `0, …, n` accept, a cap's filtered event is its plain one at table `n`;
* `forkE_capAcc_le`: a cap's filtered fork event is at most its plain one.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-- Tables `0, …, n` of the session's outputs accept. -/
def AccUpTo [Inhabited D] [Inhabited Sl] (n : ℕ) (outs : List (OutZC F D Sl)) : Prop :=
  ∀ i ≤ n, (outs.getD i dfltOutZC).accepted = true

open Classical in
/-- **Rep `r`'s level-`ℓ` cap's openings in table `n`, on accepting runs**: `opRZ` at table `n` where tables
`0, …, n` accept (`AccUpTo`), none elsewhere. -/
noncomputable def opRAccZ [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule) (n r ℓ : ℕ) (cap : Cap D) (outs : List (OutZC F D Sl)) :
    Fin (2 ^ (lvOf sch ℓ).logLen) → Option (Fin (2 ^ (lvOf sch ℓ).lanes) → F) :=
  if AccUpTo n outs then opRZ H E Lf sch r ℓ cap (outs.getD n dfltOutZC) else fun _ => none

section Opens

variable [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
  (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule) (n r ℓ : ℕ) (cap : Cap D)

omit [Fintype F] [DecidableEq F] in
theorem opRAccZ_of_acc {outs : List (OutZC F D Sl)} (h : AccUpTo n outs) :
    opRAccZ H E Lf sch n r ℓ cap outs = opRZ H E Lf sch r ℓ cap (outs.getD n dfltOutZC) := by
  unfold opRAccZ
  simp only [h, ↓reduceIte]

omit [Fintype F] [DecidableEq F] in
theorem acc_of_opRAccZ {outs : List (OutZC F D Sl)} {p : Fin (2 ^ (lvOf sch ℓ).logLen)}
    (h : (opRAccZ H E Lf sch n r ℓ cap outs p).isSome) : AccUpTo n outs := by
  by_contra hn
  unfold opRAccZ at h
  simp [hn] at h

omit [Fintype F] [DecidableEq F] in
/-- **A cap's filtered conflict is a plain one at table `n`**: two runs that open a position at the cap to different
rows, both accepting through table `n`, do so at table `n`. -/
theorem ind_conflictAcc_le (b b' : List (OutZC F D Sl)) :
    ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p,
        Rewinding.ConflictAt (opRAccZ H E Lf sch n r ℓ cap) x p) (b, b') ≤
      ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt
        (fun outs : List (OutZC F D Sl) => opRZ H E Lf sch r ℓ cap (outs.getD n dfltOutZC)) x p) (b, b') := by
  refine ind_mono (fun x hx => ?_) _
  obtain ⟨p, h₁, h₂, hne⟩ := hx
  have e₁ := opRAccZ_of_acc H E Lf sch n r ℓ cap (acc_of_opRAccZ H E Lf sch n r ℓ cap h₁)
  have e₂ := opRAccZ_of_acc H E Lf sch n r ℓ cap (acc_of_opRAccZ H E Lf sch n r ℓ cap h₂)
  rw [e₁] at h₁ hne
  rw [e₂] at h₂ hne
  exact ⟨p, h₁, h₂, hne⟩

end Opens

omit [Fintype F] [DecidableEq F] in
open Classical in
/-- **On accepting runs a cap's filtered event is its plain one**: where tables `0, …, n` accept, a run opens a
position off the table fixed at a cap through the filtered openings iff it does at table `n`. -/
theorem ind_capAcc_eq [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (sch : Schedule) (n r ℓ slot : ℕ)
    (i : IZ F D × IZ F D) {outs : List (OutZC F D Sl)} (h : AccUpTo n outs) :
    ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i slot = some (t, d) ∧
        ∃ p, Rewinding.Off (opRAccZ Hs.H Hs.E Hs.Lf sch n r ℓ d) (rowT (lvOf sch ℓ).lanes (lvOf sch ℓ).logLen t) b p)
        outs =
      ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i slot = some (t, d) ∧
        ∃ p, Rewinding.Off (opRZ Hs.H Hs.E Hs.Lf sch r ℓ d) (rowT (lvOf sch ℓ).lanes (lvOf sch ℓ).logLen t) b p)
        (outs.getD n dfltOutZC) := by
  have e : ∀ d p, opRAccZ Hs.H Hs.E Hs.Lf sch n r ℓ d outs p =
      opRZ Hs.H Hs.E Hs.Lf sch r ℓ d (outs.getD n dfltOutZC) p :=
    fun d p => congrFun (opRAccZ_of_acc Hs.H Hs.E Hs.Lf sch n r ℓ d h) p
  unfold ind
  refine if_congr ?_ rfl rfl
  simp only [Rewinding.Off, e]

/-! ## The chooser -/

open Classical in
/-- **The tables `P̃` sends, a cap's over accepting runs**: `plur`'s at the pads and `τ` slots (`s < 4`), and at rep
`(s - 4) % 2`'s level `(s - 4) / 2` cap the plurality of its openings in table `n` on runs where tables `0, …, n`
accept (`opRAccZ`). -/
noncomputable def plurAccZ [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n : ℕ)
    (plur : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl))) :
    Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl)) :=
  fun s d g σ =>
    if s < 4 then plur s d g σ
    else tabOf (lvOf T.sch ((s - 4) / 2)).lanes (lvOf T.sch ((s - 4) / 2)).logLen
      (Classical.choose (Rewinding.exists_pluralityE
        (opRAccZ Hs.H Hs.E Hs.Lf T.sch n ((s - 4) % 2) ((s - 4) / 2) d) g σ))

theorem plurAccZ_lt [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n : ℕ)
    (plur : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl))) {s : ℕ} (hs : s < 4) (d : Cap D)
    (g : Game (List (OutZC F D Sl))) (σ : Strategy g) :
    plurAccZ Hs T n plur s d g σ = plur s d g σ := by
  unfold plurAccZ
  simp only [hs, ↓reduceIte]

theorem plurAccZ_lev [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n r ℓ : ℕ) (hr : r < 2)
    (plur : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl))) (d : Cap D) (g : Game (List (OutZC F D Sl)))
    (σ : Strategy g) :
    Rewinding.IsPluralityE (opRAccZ Hs.H Hs.E Hs.Lf T.sch n r ℓ d) g σ
      (rowT (lvOf T.sch ℓ).lanes (lvOf T.sch ℓ).logLen (plurAccZ Hs T n plur (slotOf r ℓ + 4) d g σ)) := by
  have key : ∀ s, 4 ≤ s → Rewinding.IsPluralityE (opRAccZ Hs.H Hs.E Hs.Lf T.sch n ((s - 4) % 2) ((s - 4) / 2) d) g σ
      (rowT (lvOf T.sch ((s - 4) / 2)).lanes (lvOf T.sch ((s - 4) / 2)).logLen (plurAccZ Hs T n plur s d g σ)) := by
    intro s hs
    unfold plurAccZ
    simp only [show ¬ s < 4 by omega, ↓reduceIte, rowT_tabOf]
    exact Classical.choose_spec (Rewinding.exists_pluralityE _ g σ)
  have h1 : (slotOf r ℓ + 4 - 4) % 2 = r := by unfold slotOf; omega
  have h2 : (slotOf r ℓ + 4 - 4) / 2 = ℓ := by unfold slotOf; omega
  have := key (slotOf r ℓ + 4) (by omega)
  rw [h1, h2] at this
  exact this

/-! ## Rewinding at a slot of the session -/

section Rewind

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

open Classical in
/-- **Rewinding at a slot of the session**: under a chooser `plur` whose table at `slot` is a plurality of the slot's
openings `op`, the mass of runs off the table fixed at `slot` is at most `√(|P|·Q·fork)`, the fork at the slot's node
under any chooser `plur'` at level-0 oracle `default`. -/
theorem leafE_off_le_sessZ (C : Oracle F) {P V : Type} [Fintype P] [Fintype V] [DecidableEq V]
    (op : Cap D → List (OutZC F D Sl) → P → Option V) (row : Oracle F → P → V) (slot Q : ℕ)
    (hQ : ∀ d b, (Finset.univ.filter fun p => (op d b p).isSome).card ≤ Q)
    (plur plur' : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl)))
    (hplur : ∀ (d : Cap D) (g'' : Game (List (OutZC F D Sl))) (σ'' : Strategy g''),
      Rewinding.IsPluralityE (op d) g'' σ'' (row (plur slot d g'' σ'')))
    (σ' : Strategy (sessAtZ pre post (tabAfterZCP A Hs T cap₀))) :
    Lock.leafE plur (fun (i : IZ F D × IZ F D) (b : List (OutZC F D Sl)) =>
        ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i slot = some (t, d) ∧
          ∃ p, Rewinding.Off (op d) (row t) b p) b) (lockSessZ A Hs T pre post C cap₀) σ' ≤
      Real.sqrt (Fintype.card P * Q * Lock.forkE plur' slot (fun d b b' =>
        ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b'))
        (lockSessZ A Hs T pre post default cap₀) σ') := by
  have h := Lock.leafE_off_le op row slot Q hQ plur hplur (lockSessZ A Hs T pre post C cap₀) σ'
    (getPair_empty_none _)
  rwa [forkE_lockSessZ_choose A Hs T pre post C default cap₀ plur plur' slot] at h

omit [Fintype K] [DecidableEq K] in
/-- **A cap's filtered fork event is at most its plain one**, along any lockstep and chooser. -/
theorem forkE_capAcc_le {I α : Type} {get : I → ℕ → Option (Oracle F × Cap D)} {R : I → α → List (OutZC F D Sl) → Prop}
    (plur : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl))) (slot n r ℓ : ℕ) {i : I} {g : Game α}
    {g' : Game (List (OutZC F D Sl))} (L : Lock (Oracle F) (Cap D) get R i g g') (σ' : Strategy g') :
    Lock.forkE plur slot (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p,
        Rewinding.ConflictAt (opRAccZ Hs.H Hs.E Hs.Lf T.sch n r ℓ d) x p) (b, b')) L σ' ≤
      Lock.forkE plur slot (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p,
        Rewinding.ConflictAt (fun outs : List (OutZC F D Sl) =>
          opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD n dfltOutZC)) x p) (b, b')) L σ' :=
  Lock.forkE_mono plur slot (fun d b b' => ind_conflictAcc_le Hs.H Hs.E Hs.Lf T.sch n r ℓ d b b') L σ'

end Rewind

end FlockSoundness.Discharge.ZkLink
