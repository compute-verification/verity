import Proofs.Flock.Claims.ZK.Split

/-!
# The commitments from the cut's leaves (`Masked.SimCommit`)

At a live statement, table `j`'s level-0 commitment built from the cut's leaves by tree position (`commit0Cut` on the
deployed source `srcE`) is the masked prover's (`commitPZk`) but its rows and salts (`eraseC`), and each rep's pads
commitment from the cut's pads leaves (`padsTreeCut`) is the masked prover's (`innerCommitP`) but its columns and salts
(`eraseIC`), as the pure session `PS` runs them from any bodies.

* `FlockComplete.commit0Cut_eq`, `FlockComplete.padsTreeCut_eq`: a tree built from the leaves the prover hashes is the
  prover's, since under SHA-512 every leaf is a pure value of `PS` and the two build the same levels above them.
* Level 0's leaves: `leafAt_cutAt` and `exeOf_inl` (Round0) at the executable's layout (`agrees0_exe`).
* The pads' leaves: `padLeafCut_cutAt` and `exeOf_pads`. `exeOf_pads` and what it reads (`colPAt_tabC`, `sum_padCode`
  and their lemmas) are private copies of the Round1 pads modules' (`Round1Pads`, `Round1Root`), which `Split`'s
  closure doesn't hold; once it does, import them and delete the copies.
-/

namespace FlockComplete

open Flock

/-- `hm96-sha512/v1`'s leaf through SHA-512 is a pure value of `PS`. -/
private theorem hm96Ops_leaf (row : Array F128) (salt : ByteArray) :
    (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 Sha512.hash : Hashes PS)).leaf row salt =
      pure (MerkleScheme.hm96Sha512.leaf row salt) := rfl

private theorem ofSha512_sha512 (b : ByteArray) :
    (Hashes.ofSha512 Sha512.hash : Hashes PS).sha512 b = pure (Sha512.hash b) := rfl

private theorem leafWith_ofSha512 (x salt : ByteArray) :
    Hm96.Default512.leafWith (Hashes.ofSha512 Sha512.hash : Hashes PS) x salt = pure (Hm96.Default512.leaf x salt) :=
  rfl

/-- **Level 0's tree from its leaves is the prover's** but its rows and salts: where the source's leaf at each tree
position is `hm96-sha512/v1`'s leaf of row `q` under salt `q`, and its rows and salts are empty. -/
private theorem commit0Cut_eq (lv : Level) (src : TabSrc) (rows : Array (Array F128)) (salt : ℕ → ByteArray)
    (hsz : rows.size = 2 ^ lv.d)
    (hleaf : ∀ q < 2 ^ lv.d, src.leaf0 q = MerkleScheme.hm96Sha512.leaf rows[q]! (fitBytes 192 (salt q)))
    (hrow : ∀ q, src.row0 q = #[]) (hsalt : ∀ q, src.salt0 q = ByteArray.empty) :
    commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes PS) MerkleScheme.hm96Sha512 lv src =
      eraseC <$> commitRowsP (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 lv rows salt := by
  have hL : (Array.range (2 ^ lv.d)).map src.leaf0 = (Array.range (2 ^ lv.d)).map fun q =>
      MerkleScheme.hm96Sha512.leaf rows[q]!
        ((Array.range (2 ^ lv.d)).map fun q => fitBytes MerkleScheme.hm96Sha512.saltBytes (salt q))[q]! :=
    Array.map_inj_left.2 fun q hq => by
      rw [hleaf q (Array.mem_range.1 hq), FlockSoundness.Refine.getElem!_range_map _ (Array.mem_range.1 hq)]
      rfl
  have hR : (Array.range (2 ^ lv.d)).map src.row0 = rows.map fun _ => #[] :=
    Array.ext (by simp [hsz]) fun i _ _ => by simp [hrow]
  have hS : (Array.range (2 ^ lv.d)).map src.salt0 =
      ((Array.range (2 ^ lv.d)).map fun q => fitBytes MerkleScheme.hm96Sha512.saltBytes (salt q)).map
        fun _ => ByteArray.empty :=
    Array.ext (by simp) fun i _ _ => by simp [hsalt]
  unfold commit0Cut commitRowsP
  simp only [hm96Ops_leaf, Array.mapM_pure, pure_bind, map_bind, map_pure, hL, hR, hS]
  rfl

/-- **A rep's pads tree from its leaves is the prover's** but its columns and salts: where the source's pads leaf at
each of the `n` tree positions is the leaf of the codes of `h ‖ pad_h` and `μ ‖ pad_μ` there under salt `j`, and its
columns and salts are zero and empty. -/
private theorem padsTreeCut_eq (src : RepSrc) (p : PadsZk) (salt : ℕ → ByteArray) {n : ℕ}
    (hn : n = (p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)
    (hleaf : ∀ j < n, src.padLeaf j = Hm96.Default512.leaf (Sha512.hash ((Zk.codeAt (p.h ++ p.padH) j).toBytes ++
      (Zk.codeAt (p.mu ++ p.padMu) j).toBytes)) (fitBytes 192 (salt j)))
    (hch : ∀ j, src.ch j = 0) (hcm : ∀ j, src.cm j = 0) (hps : ∀ j, src.padSalt j = ByteArray.empty) :
    padsTreeCut (Hashes.ofSha512 Sha512.hash : Hashes PS) n src =
      eraseIC <$> innerCommitP (Hashes.ofSha512 Sha512.hash) p salt := by
  subst hn
  have hL : (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map src.padLeaf =
      (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map fun j =>
        Hm96.Default512.leaf (Sha512.hash
          (((Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
              (Zk.codeAt (p.h ++ p.padH)))[j]!.toBytes ++
            ((Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
              (Zk.codeAt (p.mu ++ p.padMu)))[j]!.toBytes))
          ((Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
            fun j => fitBytes 192 (salt j))[j]! :=
    Array.map_inj_left.2 fun j hj => by
      have hj' := Array.mem_range.1 hj
      rw [hleaf j hj', FlockSoundness.Refine.getElem!_range_map _ hj', FlockSoundness.Refine.getElem!_range_map _ hj',
        FlockSoundness.Refine.getElem!_range_map _ hj']
  have hC : (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map src.ch =
      ((Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
        (Zk.codeAt (p.h ++ p.padH))).map fun _ => 0 :=
    Array.ext (by simp) fun i _ _ => by simp [hch]
  have hM : (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map src.cm =
      ((Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
        (Zk.codeAt (p.mu ++ p.padMu))).map fun _ => 0 :=
    Array.ext (by simp) fun i _ _ => by simp [hcm]
  have hP : (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map src.padSalt =
      ((Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
        fun j => fitBytes 192 (salt j)).map fun _ => ByteArray.empty :=
    Array.ext (by simp) fun i _ _ => by simp [hps]
  unfold padsTreeCut innerCommitP
  simp only [ofSha512_sha512, leafWith_ofSha512, pure_bind, Array.mapM_pure, map_bind, map_pure, hL, hC, hM, hP]
  rfl

/-- Entry `|a| + k` of `a ++ b` is `b`'s entry `k`. -/
private theorem get_append_right (a b : Array F128) {k : ℕ} (hk : k < b.size) : (a ++ b)[a.size + k]! = b[k]! := by
  rw [getElem!_pos (a ++ b) _ (by simp; omega), getElem!_pos b k hk, Array.getElem_append_right (by omega)]
  simp

end FlockComplete

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL

local notation "𝔽" => FlockLevel3.GF128

variable (P : LigProver) {mPts : ℕ}

attribute [local irreducible] treeEquiv

/-- **Each rep's pads tree places every position at its index** at every coin vector of a `--zk` table whose pads code
has `Q ≤ n ≤ 2^63` positions: the opened ones at `padPos`, the others at `unPos`. -/
private theorem coinedOf_sP_val (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
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
private theorem coinedOf_ia (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ia = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩ :=
  rfl

/-- **A `--zk` table's exposed pad `h_fb`** (`coinedOf`'s `ib`), the same at every coin vector. -/
private theorem coinedOf_ib (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).ib = ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ :=
  rfl

/-- **A rep's pads at its `h`** (`ZK.Table.hFull` at `ia`, `ib`): `h`'s entries, then `h_x`, `h_y`, `h_ia·h_ib`,
`h_x·h_y`. -/
private noncomputable def padH {nE : ℕ} (ia ib : Fin nE) (h : (Fin nE → 𝔽) × 𝔽 × 𝔽) : ZK.PadIdx nE → 𝔽
  | .inl i => h.1 i
  | .inr j => ![h.2.1, h.2.2, h.1 ia * h.1 ib, h.2.1 * h.2.2] j

/-- **Rep `r`'s pads column at the code symbol of position `p`**: `(C_h, C_μ)` at `X = p + 1` (`innerPt`), pad `i`
(`padH` at `ia`, `ib`) at `X^(padIdx i)` and padding symbol `j` at `X^(len + j)`. -/
private noncomputable def padColAt {Hi : Type} {z : ZkView.Sizes} (ia ib : Fin z.nE) (padLen : ℕ) (b : ZkView.Base z Hi)
    (r : Fin 2) (p : ℕ) : 𝔽 × 𝔽 :=
  (∑ i, innerPt p ^ padIdx i * padH ia ib (b.2.2.1 r) i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.1 r j,
    ∑ i, innerPt p ^ padIdx i * b.2.2.2.2.2.1 r i + ∑ j : Fin z.nPad, innerPt p ^ (padLen + j.val) * b.2.2.2.2.1 r j)

/-- **Rep `r`'s pads column at tree position `p` is `padColAt` at `X = p + 1`**, at every coin vector of a `--zk` table
whose pads code has `Q ≤ n ≤ 2^63` positions (`coinedOf_sP_val`): an opened column and a hidden one are both the code
there. -/
private theorem coinedOf_colPAt (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
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

namespace FlockSoundness.Discharge.Composed

open Flock FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-- A pad's layout index is below `n_E + 4`. -/
private theorem padIdx_lt {nE : ℕ} (p : ZK.PadIdx nE) : padIdx p < nE + 4 := by
  rcases p with k | j
  · show k.val < nE + 4
    omega
  · show nE + (![1, 2, 0, 3] : Fin 4 → ℕ) j < nE + 4
    fin_cases j <;> simp

private theorem padIdx_injective {nE : ℕ} : Function.Injective (padIdx (nE := nE)) := by
  rintro (a | a) (b | b) h <;> simp only [padIdx] at h
  · exact congrArg Sum.inl (Fin.ext h)
  · have := a.isLt; fin_cases b <;> simp at h <;> omega
  · have := b.isLt; fin_cases a <;> simp at h <;> omega
  · fin_cases a <;> fin_cases b <;> simp at h ⊢

/-- **The pads layout's indices**: pad `i` at `padIdx i`, a bijection onto the layout's `n_E + 4` positions. -/
private noncomputable def padIdxEquiv (nE : ℕ) : ZK.PadIdx nE ≃ Fin (nE + 4) :=
  Equiv.ofBijective (fun p => ⟨padIdx p, padIdx_lt p⟩)
    ((Fintype.bijective_iff_injective_and_card _).2
      ⟨fun a b h => padIdx_injective (congrArg Fin.val h), by simp [ZK.PadIdx]⟩)

/-- **`padColAt`'s sums are a code's sum over its array** `y` of `n_E + 4 + K` symbols, where pad `i` is `y`'s entry
`padIdx i` and padding symbol `j` is entry `n_E + 4 + j`. -/
private theorem sum_padCode {nE K padLen : ℕ} (hpl : padLen = nE + 4) (y : Array F128) (ξ : 𝔽) (f : ZK.PadIdx nE → 𝔽)
    (g : Fin K → 𝔽) (hf : ∀ i, fOf y[padIdx i]! = f i) (hg : ∀ j : Fin K, fOf y[nE + 4 + j.val]! = g j) :
    ∑ i, ξ ^ padIdx i * f i + ∑ j : Fin K, ξ ^ (padLen + j.val) * g j =
      ∑ i : Fin (nE + 4 + K), fOf y[i.val]! * ξ ^ (i : ℕ) := by
  subst hpl
  rw [Fin.sum_univ_add]
  refine congrArg₂ (· + ·) ?_ (Finset.sum_congr rfl fun j _ => ?_)
  · exact Fintype.sum_equiv (padIdxEquiv nE) _ _ fun p => by
      show ξ ^ padIdx p * f p = fOf y[padIdx p]! * ξ ^ padIdx p
      rw [hf, mul_comm]
  · simp only [Fin.val_natAdd]
    rw [hg, mul_comm]

end FlockSoundness.Discharge.Composed

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge

section

attribute [local irreducible] Composed.treeEquiv

/-- **A live table's pads code has `Q ≤ n ≤ 2^63` positions**: `n = innerN`, a power of two below `2^64` holding the
`len + K_PAD ≥ Q` symbols. -/
private theorem padN_bounds (x : Msg) (j : Fin (jOf x)) :
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
private theorem colPAt_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest)
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

/-- **A cut's pads leaf at tree position `q` is the leaf at `q`** (`padLeafCut`), whatever the coins' tree. -/
private theorem padLeafCut_cutAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC P x j).z) → ByteArray) (j : Fin (jOf x)) (r : Fin 2) {q : ℕ}
    (hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU) :
    padLeafCut P x cz (cutAt (tabC P x) cz wt ω lv) j r q = lv ⟨j, .inr (.inl (r, ⟨q, hq⟩))⟩ := by
  unfold padLeafCut
  rw [dite_eq_left_of_eq_true (eq_true hq)]
  show (match (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with
    | .inl a => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inl a)))⟩
    | .inr u => lv ⟨j, .inr (.inl (r, ((tabC P x j).T (cz j)).sP r (.inr u)))⟩) = _
  rcases h : (((tabC P x j).T (cz j)).sP r).symm ⟨q, hq⟩ with a | u
  · have e : ((tabC P x j).T (cz j)).sP r (.inl a) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]
  · have e : ((tabC P x j).T (cz j)).sP r (.inr u) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]

end

/-- **Table `j`'s rep `r` pads leaf at tree position `p` is the masked prover's**: at the executable's layout, the
executable's leaf of the cut's view (`exeOf`) is `verifyInner`'s leaf of `innerCommitP`'s column `p`, the codes of
`h ‖ pad_h` and `μ ‖ pad_μ` at `X = p + 1`, under the tape's pads salt `p`, at every coin vector. -/
private theorem exeOf_pads (P : Composed.LigProver) {x : Msg} (w : Claims.Flock.Wit)
    (t : Fin Claims.Flock.zkTableCap → ExeCell) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (j : Fin (jOf x))
    (r : Fin 2) (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩ =
      _root_.Flock.Hm96.Default512.leaf (_root_.Flock.Sha512.hash
        ((_root_.Flock.Zk.codeAt ((FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val
            r.val (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).h ++
          (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padH) p.val).toBytes ++
        (_root_.Flock.Zk.codeAt ((FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val
            r.val (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).mu ++
          (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padMu) p.val).toBytes))
        (FlockComplete.fitBytes 192 (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t).innerSalt j.val r.val
          p.val)) := by
  set T := (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t with hT
  set L := _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 with hLy
  set B := ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j with hB
  have hz := fits_zkTab P (zkTab x j)
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hE : (tabC P x j).z.nE = L.hab := by
    show L.len - 4 = L.hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlen : L.len = (tabC P x j).z.nE + 4 := by
    rw [hE]
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
  have hE64 : (tabC P x j).z.nE ≤ 2 ^ 64 := hz.nE
  have hfa : L.fa < L.fb := by simp only [_root_.Flock.Zk.PadLayout.fb]; omega
  have hfb : L.fb < L.hab := Composed.ZerocheckMasked.fb_lt_hab L
  have htx : L.tx = L.hab + 1 := rfl
  have hty : L.ty = L.hab + 2 := rfl
  have htw : L.tw = L.hab + 3 := rfl
  have hhs := FlockComplete.padsOf_size T j.val r.val L
  have hpH : (FlockComplete.padsOf T j.val r.val L).padH.size = _root_.Flock.Zk.K_PAD := by
    simp [FlockComplete.padsOf]
  have hms : (FlockComplete.padsOf T j.val r.val L).mu.size = L.len := by simp [FlockComplete.padsOf]
  have hpM : (FlockComplete.padsOf T j.val r.val L).padMu.size = _root_.Flock.Zk.K_PAD := by
    simp [FlockComplete.padsOf]
  have tH : ∀ k : Fin (2 ^ 64 + 4), fOf (T.h j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.1 r k := fun k => by
    show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.1 r) k.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tM : ∀ k : Fin (2 ^ 64 + 4), fOf (T.mu j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.2.2.1 r k := fun k => by
    show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.2.2.1 r) k.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tPH : ∀ k : Fin _root_.Flock.Zk.K_PAD, fOf (T.padH j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.1 r k :=
    fun k => by
      show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.1 r) k.val) r.val) = _
      rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tPM : ∀ k : Fin _root_.Flock.Zk.K_PAD, fOf (T.padMu j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.2.1 r k :=
    fun k => by
      show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.2.1 r) k.val) r.val) = _
      rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have hpl : Composed.padLenOf (zkTab x j).table (stAt x j) = (tabC P x j).z.nE + 4 := hlen
  set pd := FlockComplete.padsOf T j.val r.val L with hpd
  have hsz : (pd.h ++ pd.padH).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad := by
    rw [Array.size_append, hhs, hpH, hlen]; rfl
  have hszm : (pd.mu ++ pd.padMu).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad := by
    rw [Array.size_append, hms, hpM, hlen]; rfl
  have hget0 : ∀ k, k < L.len → k ≠ L.hab → k ≠ L.tw → pd.h[k]! = T.h j.val r.val k :=
    fun k hk h1 h2 => FlockComplete.padsOf_get T _ _ L k hk h1 h2
  have hget : ∀ k, k < L.len → k ≠ L.hab → k ≠ L.tw → (pd.h ++ pd.padH)[k]! = T.h j.val r.val k :=
    fun k hk h1 h2 => by rw [FlockComplete.get_append_left _ _ (by rw [hhs]; exact hk), hget0 k hk h1 h2]
  have hfaE : L.fa < (tabC P x j).z.nE := by omega
  have hfbE : L.fb < (tabC P x j).z.nE := by omega
  have hEv : ∀ b : Fin 2, (hEmb hz (.inr b)).val = L.hab + 1 + b.val := fun b => by
    show (tabC P x j).z.nE + 1 + b.val = _
    rw [hE]
  have hf : ∀ i : ZK.PadIdx (tabC P x j).z.nE, fOf (pd.h ++ pd.padH)[Composed.padIdx i]! =
      Composed.padH (nE := (tabC P x j).z.nE) ⟨L.fa, hfaE⟩ ⟨L.fb, hfbE⟩ (B.2.2.1 r) i := by
    rintro (k | i)
    · have hk := k.isLt
      show fOf (pd.h ++ pd.padH)[k.val]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl k))
      rw [hget k.val (by omega) (by omega) (by omega)]
      exact tH (hEmb hz (.inl k))
    · fin_cases i
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 1]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0))
        rw [hget ((tabC P x j).z.nE + 1) (by omega) (by omega) (by omega), ← tH, hEv, hE]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 2]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1))
        rw [hget ((tabC P x j).z.nE + 2) (by omega) (by omega) (by omega), ← tH, hEv, hE]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 0]! =
          (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨L.fa, hfaE⟩)) *
            (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨L.fb, hfbE⟩))
        rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), show (tabC P x j).z.nE + 0 = L.hab by omega,
          FlockComplete.padsOf_hab, hget0 L.fa (by omega) (by omega) (by omega),
          hget0 L.fb (by omega) (by omega) (by omega), fOf_mul, tH ⟨L.fa, by omega⟩, tH ⟨L.fb, by omega⟩]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 3]! =
          (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0)) * (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1))
        rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), show (tabC P x j).z.nE + 3 = L.tw by omega,
          FlockComplete.padsOf_tw, hget0 L.tx (by omega) (by omega) (by omega),
          hget0 L.ty (by omega) (by omega) (by omega), fOf_mul, ← tH, ← tH, hEv, hEv]
        rfl
  have hfm : ∀ i : ZK.PadIdx (tabC P x j).z.nE, fOf (pd.mu ++ pd.padMu)[Composed.padIdx i]! = B.2.2.2.2.2.1 r i := by
    intro i
    have hi := Composed.padIdx_lt i
    rw [FlockComplete.get_append_left _ _ (by rw [hms]; omega),
      show pd.mu[Composed.padIdx i]! = T.mu j.val r.val (Composed.padIdx i) from Composed.Zerocheck.get_map_range _ (by omega)]
    exact tM (padEmb' hz i)
  have hg : ∀ k : Fin (tabC P x j).z.nPad, fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.h.size + k.val by rw [hhs, hlen],
      FlockComplete.get_append_right _ _ (by rw [hpH]; exact hk),
      show pd.padH[k.val]! = T.padH j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPH ⟨k.val, hk⟩
  have hgm : ∀ k : Fin (tabC P x j).z.nPad,
      fOf (pd.mu ++ pd.padMu)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.mu.size + k.val by rw [hms, hlen],
      FlockComplete.get_append_right _ _ (by rw [hpM]; exact hk),
      show pd.padMu[k.val]! = T.padMu j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPM ⟨k.val, hk⟩
  have h1 : (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).1 =
      fOf (_root_.Flock.Zk.codeAt (pd.h ++ pd.padH) p.val) := by
    rw [colPAt_tabC, ZkExec.fOf_codeAt _ p.val _ hsz]
    exact Composed.sum_padCode hpl (pd.h ++ pd.padH) (ZkLink.innerPt p.val) _ _ hf hg
  have h2 : (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).2 =
      fOf (_root_.Flock.Zk.codeAt (pd.mu ++ pd.padMu) p.val) := by
    rw [colPAt_tabC, ZkExec.fOf_codeAt _ p.val _ hszm]
    exact Composed.sum_padCode hpl (pd.mu ++ pd.padMu) (ZkLink.innerPt p.val) _ _ hfm hgm
  have hp : p.val < _root_.Flock.Zk.Q + 2 ^ 64 := by
    have h1 := p.isLt
    have h2 : (tabC P x j).z.nQ ≤ _root_.Flock.Zk.Q := hz.nQ
    have h3 : (tabC P x j).z.nPU ≤ 2 ^ 64 := hz.nPU
    omega
  have hs : ZkView.saltBytes (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩) =
      FlockComplete.fitBytes 192 (T.innerSalt j.val r.val p.val) := by
    show ZkView.saltBytes ((t ⟨j.val, hc⟩).2.1.2.1 r (Fin.castLE (Nat.add_le_add hz.nQ hz.nPU) p)) =
      FlockComplete.fitBytes 192 (fAt ByteArray.empty (fun r => fAt ByteArray.empty
        (fun p => ZkView.saltBytes ((cellAt t j.val).2.1.2.1 r p)) p.val) r.val)
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt, fitBytes_of_size (by rw [fAt_lt _ _ hp]; exact saltBytes_size _),
      fAt_lt _ _ hp]
    rfl
  show _root_.Flock.Hm96.Default512.leaf (_root_.Flock.Sha512.hash
      (_root_.Flock.F128.toBytes (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).1 ++
        _root_.Flock.F128.toBytes (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).2))
      (ZkView.saltBytes (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩)) = _
  rw [h1, h2, hs]
  rfl

/-- **`SimCommit`**: at a live statement, table `j`'s level-0 commitment from the cut's leaves is the masked prover's
but its rows and salts, and each rep's pads commitment from the cut's pads leaves is the masked prover's but its
columns and salts, as `PS` runs them from any bodies. -/
theorem simCommit : SimCommit := by
  intro x hx w _ τ c j l
  have hj := lt_of_jOf j
  obtain ⟨s, spec, hs⟩ := setupOf_of_pos (x := x) (by omega)
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hjt : j.val < (FlockDeploy.tabsOf (stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have hget : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val hjt]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  have hms : (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hget]
    exact (hall j.val hj).2.2
  have hleaf0 : ∀ q, (srcE x c w τ j.val).leaf0 q = leafAt PE x (czExe PE x c) (cutE x c w τ) j q := fun q => by
    unfold srcE tabSrcCut
    rw [dite_eq_left j.isLt]
  have hrow0 : ∀ q, (srcE x c w τ j.val).row0 q = #[] := fun q => by
    unfold srcE tabSrcCut
    rw [dite_eq_left j.isLt]
  have hsalt0 : ∀ q, (srcE x c w τ j.val).salt0 q = ByteArray.empty := fun q => by
    unfold srcE tabSrcCut
    rw [dite_eq_left j.isLt]
  refine ⟨?_, fun r => ?_⟩
  · congr 1
    rw [hms]
    unfold FlockComplete.commitPZk
    obtain ⟨-, -, -, -, hd, -⟩ := tabC_z PE x j
    refine FlockComplete.commit0Cut_eq _ _ _ _ (by simp [FlockComplete.encodeRowsZk]) (fun q hq => ?_) hrow0 hsalt0
    have hq' : q < (tabC PE x j).z.nS + (tabC PE x j).z.nU := by rw [hd]; exact hq
    rw [hleaf0]
    refine ((leafAt_cutAt PE x (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1
        (exeOf (tabC PE x) (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1 (UE.tape x τ).2.1) j hq').trans
      (exeOf_inl PE (TapeLayout.exe PE) exeTape (agrees0_exe PE) w τ (czExe PE x c) j ⟨q, hq'⟩)).trans ?_
    rw [FlockComplete.tablesP, FlockSoundness.Refine.getElem!_range_map _ hjt]
    rfl
  · congr 1
    have hrep : (srcE x c w τ j.val).rep r.val = repSrcCut PE x (czExe PE x c) (cutE x c w τ) j r := by
      unfold srcE tabSrcCut
      rw [dite_eq_left j.isLt]
      exact dite_eq_left r.isLt
    have hm : (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = (stAt x j).m := by rw [hget]; rfl
    have hk : (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog = (stAt x j).c.kLog := by rw [hget]; rfl
    rw [hrep, hm, hk]
    have hpH : (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
        (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padH.size =
          _root_.Flock.Zk.K_PAD := by
      simp [FlockComplete.padsOf]
    have hNq : (tabC PE x j).z.nQ + (tabC PE x j).z.nPU = Composed.padNOf (zkTab x j).table (stAt x j) := by
      have e1 : (tabC PE x j).z.nQ = _root_.Flock.Zk.Q := rfl
      have e2 : (tabC PE x j).z.nPU = Composed.padNOf (zkTab x j).table (stAt x j) - _root_.Flock.Zk.Q := rfl
      have := (padN_bounds x j).1
      omega
    refine FlockComplete.padsTreeCut_eq _ _ _ (by rw [Array.size_append, FlockComplete.padsOf_size, hpH])
      (fun q hq => ?_) (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
    have hq' : q < (tabC PE x j).z.nQ + (tabC PE x j).z.nPU := by rw [hNq]; exact hq
    have h1 := padLeafCut_cutAt PE x (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1
      (exeOf (tabC PE x) (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1 (UE.tape x τ).2.1) j r hq'
    have h2 := exeOf_pads PE w τ (czExe PE x c) j r ⟨q, hq'⟩
    simp only [repSrcCut]
    unfold cutE
    rw [h1]
    exact h2

end Claims.ZKMal.Masked
