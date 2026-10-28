import Proofs.Flock.Claims.Complete.ZkTables.Round1
import Proofs.Flock.Claims.ZK.Sent

/-!
# Round `1` of the masked session: the pads trees by tree position

The masked session's second body binds table `0`'s rep `0` pads root (`Masked.liveRun_round1`). On the model's side a
rep's pads leaf at tree position `p` commits `colPAt w b r p`. Here, at every coin vector of a `--zk` table:

* `Composed.coinedOf_sP_val`: each rep's pads tree places its opened positions and the others at their indices, where
  `Q ≤ n ≤ 2^63` for the pads code's length `n`.
* `Composed.coinedOf_colPAt`: so rep `r`'s pads column at tree position `p` is `padColAt` at `X = p + 1`
  (`ZkLink.innerPt`), each pad (`padH` at the exposed `h_fa`, `h_fb`: `coinedOf_ia`, `coinedOf_ib`) at its layout index
  (`padIdx`) and padding symbol `j` at `len + j`, which reads no coin.
* `Masked.padN_bounds`: a live table's pads code has `Q ≤ n ≤ 2^63`.
* `Masked.leafAtP`, `Masked.leafAtP_cutAt`: rep `r`'s pads leaf at tree position `q` of a cut, the leaf at `q` whatever
  the coins' tree; `Masked.digOf_pads`: a pads leaf's digest is the same at every coin vector.

`treeEquiv` is irreducible here: unfolding it decides its `∃ e, …` over the tree's size.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL

local notation "𝔽" => FlockLevel3.GF128

variable (P : LigProver) {mPts : ℕ}

attribute [local irreducible] treeEquiv

/-- **Each rep's pads tree places every position at its index** at every coin vector of a `--zk` table whose pads code
has `Q ≤ n ≤ 2^63` positions: the opened ones at `padPos`, the others at `unPos`. -/
theorem coinedOf_sP_val (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (hQ : Zk.Q ≤ padNOf Tz st) (hN : padNOf Tz st ≤ 2 ^ 63) (r : Fin 2)
    (x : Fin (coinedOf P Tz st).z.nQ ⊕ Fin (coinedOf P Tz st).z.nPU) :
    ((((coinedOf P Tz st).T c).sP r) x).val =
      Sum.elim (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest))
        (fun u => unPos (padNOf Tz st)
          (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val) x := by
  have hL : max (min (padNOf Tz st) (2 ^ 63)) Zk.Q = padNOf Tz st := by
    rw [min_eq_left hN, max_eq_left hQ]
  have hQ' : Zk.Q ≤ max (min (padNOf Tz st) (2 ^ 63)) Zk.Q := le_max_right _ _
  have hsP : ((coinedOf P Tz st).T c).sP r = treeEquiv (a := (coinedOf P Tz st).z.nQ) (b := (coinedOf P Tz st).z.nPU)
      (Sum.elim (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest))
        (fun u => unPos (padNOf Tz st)
          (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val)) := rfl
  rw [hsP]
  exact treeEquiv_val_pos (t := Zk.Q) (N := padNOf Tz st) _ (padPos_injective _ hQ')
    (fun i => lt_of_lt_of_eq (padPos_lt _ hQ' i) hL) hQ x

/-- **A `--zk` table's exposed pad `h_fa`** (`coinedOf`'s `ia`), the same at every coin vector. -/
theorem coinedOf_ia (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ia = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩ :=
  rfl

/-- **A `--zk` table's exposed pad `h_fb`** (`coinedOf`'s `ib`), the same at every coin vector. -/
theorem coinedOf_ib (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ib = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ :=
  rfl

/-- **A rep's pads at its `h`** (`ZK.Table.hFull` at `ia`, `ib`): `h`'s entries, then `h_x`, `h_y`, `h_ia·h_ib`,
`h_x·h_y`. -/
noncomputable def padH {nE : ℕ} (ia ib : Fin nE) (h : (Fin nE → 𝔽) × 𝔽 × 𝔽) : ZK.PadIdx nE → 𝔽
  | .inl i => h.1 i
  | .inr j => ![h.2.1, h.2.2, h.1 ia * h.1 ib, h.2.1 * h.2.2] j

/-- **Rep `r`'s pads column at the code symbol of position `p`**: `(C_h, C_μ)` at `X = p + 1` (`innerPt`), pad `i`
(`padH` at `ia`, `ib`) at `X^(padIdx i)` and padding symbol `j` at `X^(len + j)`. -/
noncomputable def padColAt {Hi : Type} {z : ZkView.Sizes} (ia ib : Fin z.nE) (padLen : ℕ) (b : ZkView.Base z Hi)
    (r : Fin 2) (p : ℕ) : 𝔽 × 𝔽 :=
  (∑ i, innerPt p ^ padIdx i * padH ia ib (b.2.2.1 r) i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.1 r j,
    ∑ i, innerPt p ^ padIdx i * b.2.2.2.2.2.1 r i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.2.1 r j)

/-- **Rep `r`'s pads column at tree position `p` is `padColAt` at `X = p + 1`**, at every coin vector of a `--zk` table
whose pads code has `Q ≤ n ≤ 2^63` positions (`coinedOf_sP_val`): an opened column and a hidden one are both the code
there. -/
theorem coinedOf_colPAt (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (hQ : Zk.Q ≤ padNOf Tz st) (hN : padNOf Tz st ≤ 2 ^ 63)
    (w : Model.Witness (coinedOf P Tz st).S.m) (b : ZkView.Base (coinedOf P Tz st).z (coinedOf P Tz st).Hi) (r : Fin 2)
    (p : Fin ((coinedOf P Tz st).z.nQ + (coinedOf P Tz st).z.nPU)) :
    ((coinedOf P Tz st).T c).colPAt w b r p =
      padColAt (z := (coinedOf P Tz st).z)
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ (padLenOf Tz st) b r p.val := by
  have hs := coinedOf_sP_val P Tz st c hQ hN r
  have hH : ∀ (h : (Fin (coinedOf P Tz st).z.nE → 𝔽) × 𝔽 × 𝔽) i, ((coinedOf P Tz st).T c).tab.hFull h i =
      padH (nE := (coinedOf P Tz st).z.nE)
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
        ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ h i := fun h i => by
    rcases i with i | i <;> rfl
  have hH' := funext (hH (b.2.2.1 r))
  unfold ZkView.Coined.colPAt
  rcases h : (((coinedOf P Tz st).T c).sP r).symm p with k | u
  · have hk : padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k = p.val := by
      have e : ((coinedOf P Tz st).T c).sP r (.inl k) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs (.inl k)
      rw [e] at this
      exact this.symm
    have hC : ∀ i, ((coinedOf P Tz st).T c).Ch r k i = innerPt p.val ^ padIdx i := fun i => by
      show fOf ⟨(padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k + 1).toUInt64,
        0⟩ ^ padIdx i = _
      rw [hk]
      rfl
    have hP : ∀ j, ((coinedOf P Tz st).T c).Cp r k j = innerPt p.val ^ (padLenOf Tz st + j.val) := fun j => by
      show fOf ⟨(padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest) k + 1).toUInt64,
        0⟩ ^ (padLenOf Tz st + j.val) = _
      rw [hk]
      rfl
    refine Prod.ext ?_ ?_
    · show ∑ i, ((coinedOf P Tz st).T c).Ch r k i * ((coinedOf P Tz st).T c).tab.hFull (b.2.2.1 r) i +
          ∑ j, ((coinedOf P Tz st).T c).Cp r k j * b.2.2.2.1 r j = _
      simp only [hC, hP, hH']
      rfl
    · show ∑ i, ((coinedOf P Tz st).T c).Ch r k i * b.2.2.2.2.2.1 r i +
          ∑ j, ((coinedOf P Tz st).T c).Cp r k j * b.2.2.2.2.1 r j = _
      simp only [hC, hP]
      rfl
  · have hu : unPos (padNOf Tz st)
        (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q ((bodyOf Tz st).padQueries r c.rest)) u.val = p.val := by
      have e : ((coinedOf P Tz st).T c).sP r (.inr u) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs (.inr u)
      rw [e] at this
      exact this.symm
    have hC : ∀ i, ((coinedOf P Tz st).T c).ChU r u i = innerPt p.val ^ padIdx i := fun i => by
      show fOf ⟨(unPos (padNOf Tz st) (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q
        ((bodyOf Tz st).padQueries r c.rest)) u.val + 1).toUInt64, 0⟩ ^ padIdx i = _
      rw [hu]
      rfl
    have hP : ∀ j, ((coinedOf P Tz st).T c).CpU r u j = innerPt p.val ^ (padLenOf Tz st + j.val) := fun j => by
      show fOf ⟨(unPos (padNOf Tz st) (padPos (max (min (padNOf Tz st) (2 ^ 63)) Zk.Q) Zk.Q
        ((bodyOf Tz st).padQueries r c.rest)) u.val + 1).toUInt64, 0⟩ ^ (padLenOf Tz st + j.val) = _
      rw [hu]
      rfl
    refine Prod.ext ?_ ?_
    · show ∑ i, ((coinedOf P Tz st).T c).ChU r u i * ((coinedOf P Tz st).T c).tab.hFull (b.2.2.1 r) i +
          ∑ j, ((coinedOf P Tz st).T c).CpU r u j * b.2.2.2.1 r j = _
      simp only [hC, hP, hH']
      rfl
    · show ∑ i, ((coinedOf P Tz st).T c).ChU r u i * b.2.2.2.2.2.1 r i +
          ∑ j, ((coinedOf P Tz st).T c).CpU r u j * b.2.2.2.2.1 r j = _
      simp only [hC, hP]
      rfl

end FlockSoundness.Discharge.Composed

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge

attribute [local irreducible] Composed.treeEquiv

/-- **A live table's pads code has `Q ≤ n ≤ 2^63` positions**: `n = innerN`, a power of two below `2^64` holding the
`len + K_PAD ≥ Q` symbols. -/
theorem padN_bounds (x : Msg) (j : Fin (jOf x)) :
    _root_.Flock.Zk.Q ≤ Composed.padNOf (zkTab x j).table (stAt x j) ∧
      Composed.padNOf (zkTab x j).table (stAt x j) ≤ 2 ^ 63 := by
  have hN : Composed.padNOf (zkTab x j).table (stAt x j) = ZkExec.innerN (ZkLink.layOf (stAt x j) (zkTab x j).sch) := rfl
  have hL := (zkTab x j).hL
  have h64 := (zkTab x j).hN
  rw [hN]
  refine ⟨le_trans (by simp [_root_.Flock.Zk.Q, _root_.Flock.Zk.K_PAD]) hL, ?_⟩
  have hp := FlockComplete.innerCols_pow ((ZkLink.layOf (stAt x j) (zkTab x j).sch).len + _root_.Flock.Zk.K_PAD)
  unfold ZkExec.innerN at h64 ⊢
  rw [hp] at h64 ⊢
  exact Nat.pow_le_pow_right (by norm_num) (by
    have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).1 h64
    omega)

/-- **Table `j`'s rep `r` pads column at tree position `p`** is `padColAt` at `X = p + 1`, at every coin vector. -/
theorem colPAt_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest)
    (w : Model.Witness (tabC P x j).S.m) (b : ZkView.Base (tabC P x j).z (tabC P x j).Hi) (r : Fin 2)
    (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    ((tabC P x j).T c).colPAt w b r p =
      Composed.padColAt (z := (tabC P x j).z)
        ⟨(_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).table.sch.k0).fa,
          Nat.lt_of_succ_lt (Composed.fa_lt (zkTab x j).table (stAt x j))⟩
        ⟨(_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).table.sch.k0).fb,
          Composed.fa_lt (zkTab x j).table (stAt x j)⟩
        (Composed.padLenOf (zkTab x j).table (stAt x j)) b r p.val :=
  Composed.coinedOf_colPAt P (zkTab x j).table (stAt x j) c (padN_bounds x j).1 (padN_bounds x j).2 w b r p

/-- **Table `j`'s rep `r` pads leaf at tree position `q`** of a cut at the coins `cz`: the cut's opened or hidden pads
leaf the coins' tree places at `q` (`Coined.sP`), and the empty string past the tree. -/
noncomputable def leafAtP (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) (j : Fin (jOf x)) (r : Fin 2) (q : ℕ) : ByteArray :=
  if hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU then
    match (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with
    | .inl a => ((cut j).leavesP r).1 a
    | .inr u => ((cut j).leavesP r).2 u
  else ByteArray.empty

/-- **A cut's pads leaf at tree position `q` is the leaf at `q`**, whatever the coins' tree. -/
theorem leafAtP_cutAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC P x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2) {q : ℕ}
    (hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU) :
    leafAtP P x cz (cutAt (tabC P x) cz wt ω lv) j r q = lv ⟨j, .inr (.inl (r, ⟨q, hq⟩))⟩ := by
  unfold leafAtP
  rw [dite_eq_left_of_eq_true (eq_true hq)]
  show (match (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with
    | .inl a => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inl a)))⟩
    | .inr u => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inr u)))⟩) = _
  rcases h : (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with a | u
  · have e : ((tabC P x j).T (cz j)).sP r (.inl a) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]
  · have e : ((tabC P x j).T (cz j)).sP r (.inr u) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]

/-- **A pads leaf's digest is the same at every coin vector** (`colPAt_tabC`). -/
theorem digOf_pads (P : Composed.LigProver) (x : Msg) (cz cz' : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi) (j : Fin (jOf x))
    (r : Fin 2) (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    digOf (tabC P x) cz wt ω ⟨j, .inr (.inl (r, p))⟩ = digOf (tabC P x) cz' wt ω ⟨j, .inr (.inl (r, p))⟩ := by
  unfold digOf
  simp only [ZkView.Coined.content]
  rw [colPAt_tabC, colPAt_tabC]

end Claims.ZKMal.Masked
