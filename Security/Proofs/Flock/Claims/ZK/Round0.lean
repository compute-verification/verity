import Proofs.Flock.Claims.ZK.OneTape
import Proofs.Flock.Claims.Complete.ZkTables.Round0

/-!
# Round `0` of the masked session: the level-0 columns by tree position

The masked prover's first message is the link of its tables' level-0 caps (`FlockComplete.honestPR_first`), each cap a
function of the level's leaves (`FlockComplete.commitsPZk_caps`). On the model's side a table's level-0 leaf at tree
position `p` commits `row0 (col0At w b p)`. Here, at every coin vector of a `--zk` table:

* `treeEquiv_val`, `unPos_list`: where the positions fill the tree, each sits at its own index.
* `coinedOf_s0_val`: so at a `--zk` table's sizes (`t_pad ≤ 2^cols`, `cols + 1 ≤ logLen`), level 0's tree places its
  opened positions and the others at their indices, at every coin vector.
* `coinedOf_col0At`: so level 0's column at tree position `p` is `colAt` at `ω_p`, the lanes and extra lanes encoded at
  `ω_p` plus their padding, the same at every coin vector.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL

local notation "𝔽" => FlockLevel3.GF128

/-! ## Each position at its tree index -/

/-- **`treeEquiv` places each position at its index** where those indices are distinct and fill the tree. -/
theorem treeEquiv_val {a b : ℕ} {f : Fin a ⊕ Fin b → ℕ} (hlt : ∀ x, f x < a + b) (hinj : Function.Injective f)
    (x : Fin a ⊕ Fin b) : (treeEquiv f x).val = f x := by
  have h : ∃ e : Fin a ⊕ Fin b ≃ Fin (a + b), ∀ x, (e x).val = f x := by
    let g : Fin a ⊕ Fin b → Fin (a + b) := fun x => ⟨f x, hlt x⟩
    have hg : Function.Injective g := fun x y hxy => hinj (congrArg Fin.val hxy)
    have hb : Function.Bijective g := (Fintype.bijective_iff_injective_and_card g).2 ⟨hg, by simp⟩
    exact ⟨Equiv.ofBijective g hb, fun _ => rfl⟩
  unfold treeEquiv
  split
  · next h' => exact h'.choose_spec x
  · contradiction

/-- **The positions below `L` nobody opens**: where the `t` opened positions are distinct and below `L`, the others
are `L − t` distinct positions below `L`, none opened. -/
theorem unPos_list {t : ℕ} (L : ℕ) (opened : Fin t → ℕ) (hinj : Function.Injective opened)
    (hlt : ∀ i, opened i < L) :
    ((List.range L).filter fun x => decide (x ∉ List.ofFn opened)).Nodup ∧
      ((List.range L).filter fun x => decide (x ∉ List.ofFn opened)).length = L - t ∧
      ∀ x ∈ (List.range L).filter (fun x => decide (x ∉ List.ofFn opened)), x < L ∧ x ∉ List.ofFn opened := by
  have hin : ((List.range L).filter fun x => decide (x ∈ List.ofFn opened)).length = t := by
    have hp : ((List.range L).filter fun x => decide (x ∈ List.ofFn opened)).Perm (List.ofFn opened) := by
      rw [List.perm_ext_iff_of_nodup (List.nodup_range.filter _) (List.nodup_ofFn.2 hinj)]
      intro a
      simp only [List.mem_filter, List.mem_range, decide_eq_true_eq]
      refine ⟨fun h => h.2, fun h => ⟨?_, h⟩⟩
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 h
      exact hlt i
    rw [hp.length_eq, List.length_ofFn]
  have hsum := List.length_eq_length_filter_add (l := List.range L) (fun x => decide (x ∈ List.ofFn opened))
  simp only [List.length_range, hin] at hsum
  refine ⟨List.nodup_range.filter _, by simp only [decide_not]; omega, fun x hx => ?_⟩
  simp only [List.mem_filter, List.mem_range, decide_eq_true_eq] at hx
  exact hx

theorem unPos_spec {t : ℕ} (L : ℕ) (opened : Fin t → ℕ) (hinj : Function.Injective opened)
    (hlt : ∀ i, opened i < L) {u : ℕ} (hu : u < L - t) :
    unPos L opened u < L ∧ unPos L opened u ∉ List.ofFn opened := by
  obtain ⟨-, hl, hx⟩ := unPos_list L opened hinj hlt
  unfold unPos
  rw [List.getD_eq_getElem _ _ (by omega)]
  exact hx _ (List.getElem_mem _)

theorem unPos_injOn {t : ℕ} (L : ℕ) (opened : Fin t → ℕ) (hinj : Function.Injective opened)
    (hlt : ∀ i, opened i < L) {u u' : ℕ} (hu : u < L - t) (hu' : u' < L - t)
    (h : unPos L opened u = unPos L opened u') : u = u' := by
  obtain ⟨hn, hl, -⟩ := unPos_list L opened hinj hlt
  unfold unPos at h
  rw [List.getD_eq_getElem _ _ (by omega), List.getD_eq_getElem _ _ (by omega)] at h
  exact (List.Nodup.getElem_inj_iff hn).1 h

/-- **The opened positions, then the others, each at its index**: `t` distinct opened positions below `N`, then the
`N − t` others in increasing order (`unPos`). -/
theorem treeEquiv_val_pos {t N : ℕ} (pos : Fin t → ℕ) (hinj : Function.Injective pos) (hlt : ∀ i, pos i < N)
    (htN : t ≤ N) (x : Fin t ⊕ Fin (N - t)) :
    (treeEquiv (Sum.elim pos fun u => unPos N pos u.val) x).val = Sum.elim pos (fun u => unPos N pos u.val) x := by
  refine treeEquiv_val (fun y => ?_) (fun y y' hy => ?_) x
  · rcases y with q | u
    · have := hlt q
      simp only [Sum.elim_inl]
      omega
    · have := (unPos_spec N pos hinj hlt u.isLt).1
      simp only [Sum.elim_inr]
      omega
  · rcases y with q | u <;> rcases y' with q' | u'
    · exact congrArg Sum.inl (hinj hy)
    · exact absurd (List.mem_ofFn.2 ⟨q, hy⟩) (unPos_spec N pos hinj hlt u'.isLt).2
    · exact absurd (List.mem_ofFn.2 ⟨q', hy.symm⟩) (unPos_spec N pos hinj hlt u.isLt).2
    · exact congrArg Sum.inr (Fin.ext (unPos_injOn N pos hinj hlt u.isLt u'.isLt hy))

/-! ## Level 0's tree at a `--zk` table -/

variable (P : LigProver) {mPts : ℕ}

/-- **Level 0's tree places every position at its index** at every coin vector of a `--zk` table whose `t_pad`
opened positions fit below `2^(cols+1)` and whose level 0 has at least `2^(cols+1)` positions: the opened ones at
`pos c`, the others at `unPos`. -/
theorem coinedOf_s0_val (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (ht : Tz.t ≤ 2 ^ (Tz.sch.level₀.logCols + 1)) (hL : Tz.sch.level₀.logCols + 1 ≤ Tz.sch.level₀.logLen)
    (x : Fin (coinedOf P Tz st).z.nS ⊕ Fin (coinedOf P Tz st).z.nU) :
    (((coinedOf P Tz st).T c).s0 x).val =
      Sum.elim ((coinedOf P Tz st).pos c) (fun u => unPos (2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) u.val)
        x := by
  have h2 : 2 ^ (Tz.sch.level₀.logCols + 1) ≤ 2 ^ Tz.sch.level₀.logLen := Nat.pow_le_pow_right (by norm_num) hL
  exact treeEquiv_val_pos (N := 2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) (padPos_injective _ ht)
    (fun i => lt_of_lt_of_le (padPos_lt _ ht i) h2) (le_trans ht h2) x

/-- Level 0's message column `c` at the codeword symbol of position `p`: `X̂_c(ω_p)`. -/
noncomputable def eAt (cols p : ℕ) {L : ℕ} (c : Fin L) : 𝔽 := execArith.Xhat c.val (cols + 1) (execArith.omega p)

/-- Level 0's padding column `j` at the codeword symbol of position `p`: `κ·X̂_j(ω_p) + X̂_{2^cols+j}(ω_p)` at
`κ = PAD_C`. -/
noncomputable def gAt (cols p : ℕ) {t : ℕ} (j : Fin t) : 𝔽 :=
  fOf Zk.PAD_C * execArith.Xhat j.val (cols + 1) (execArith.omega p) +
    execArith.Xhat (2 ^ cols + j.val) (cols + 1) (execArith.omega p)

/-- **Level 0's column at the codeword symbol of position `p`**: each witness lane and each extra lane encoded at
`ω_p` (`eAt`), plus its padding coefficients on the padding columns there (`gAt`). -/
noncomputable def colAt {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (cols : ℕ)
    (w : W) (b : ZkView.Base z Hi) (p : ℕ) : (Fin z.nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽) :=
  (fun l => ∑ c, T.lanes w b.1 l c * eAt cols p c + ∑ j, b.2.1.1 l j * gAt cols p j,
    fun r i => ∑ c, b.2.1.2.1 r i c * eAt cols p c + ∑ j, b.2.1.2.2 r i j * gAt cols p j)

/-- **Level 0's column at tree position `p` is `colAt` at `ω_p`**, at every coin vector of a `--zk` table whose sizes
fit (`coinedOf_s0_val`): an opened position's rows and a hidden position's column are both the code at `ω_p`. -/
theorem coinedOf_col0At (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (ht : Tz.t ≤ 2 ^ (Tz.sch.level₀.logCols + 1)) (hL : Tz.sch.level₀.logCols + 1 ≤ Tz.sch.level₀.logLen)
    (w : Model.Witness (coinedOf P Tz st).S.m) (b : ZkView.Base (coinedOf P Tz st).z (coinedOf P Tz st).Hi)
    (p : Fin ((coinedOf P Tz st).z.nS + (coinedOf P Tz st).z.nU)) :
    ((coinedOf P Tz st).T c).col0At w b p = colAt ((coinedOf P Tz st).T c) Tz.sch.level₀.logCols w b p.val := by
  have hs0 := coinedOf_s0_val P Tz st c ht hL
  unfold ZkView.Coined.col0At
  rcases h : ((coinedOf P Tz st).T c).s0.symm p with q | u
  · have hq : (coinedOf P Tz st).pos c q = p.val := by
      have e : ((coinedOf P Tz st).T c).s0 (.inl q) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs0 (.inl q)
      rw [e] at this
      exact this.symm
    have hE : ∀ k, ((coinedOf P Tz st).T c).E q k = eAt Tz.sch.level₀.logCols p.val k := fun k => by
      show execArith.Xhat _ _ (execArith.omega ((coinedOf P Tz st).pos c q)) = _
      rw [hq]
      rfl
    have hG : ∀ j, ((coinedOf P Tz st).T c).Gp q j = gAt Tz.sch.level₀.logCols p.val j := fun j => by
      show fOf Zk.PAD_C * execArith.Xhat _ _ (execArith.omega ((coinedOf P Tz st).pos c q)) +
        execArith.Xhat _ _ (execArith.omega ((coinedOf P Tz st).pos c q)) = _
      rw [hq]
      rfl
    refine Prod.ext (funext fun l => ?_) (funext fun r => funext fun i => ?_)
    · show ∑ k, ((coinedOf P Tz st).T c).lanes w b.1 l k * ((coinedOf P Tz st).T c).E q k +
          ∑ j, b.2.1.1 l j * ((coinedOf P Tz st).T c).Gp q j = _
      simp only [hE, hG]
      rfl
    · show ∑ k, b.2.1.2.1 r i k * ((coinedOf P Tz st).T c).E q k +
          ∑ j, b.2.1.2.2 r i j * ((coinedOf P Tz st).T c).Gp q j = _
      simp only [hE, hG]
      rfl
  · have hu : unPos (2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) u.val = p.val := by
      have e : ((coinedOf P Tz st).T c).s0 (.inr u) = p := by rw [← h, Equiv.apply_symm_apply]
      have := hs0 (.inr u)
      rw [e] at this
      exact this.symm
    have hE : ∀ k, ((coinedOf P Tz st).T c).EU u k = eAt Tz.sch.level₀.logCols p.val k := fun k => by
      show execArith.Xhat _ _ (execArith.omega (unPos (2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) u.val)) = _
      rw [hu]
      rfl
    have hG : ∀ j, ((coinedOf P Tz st).T c).GpU u j = gAt Tz.sch.level₀.logCols p.val j := fun j => by
      show fOf Zk.PAD_C * execArith.Xhat _ _
          (execArith.omega (unPos (2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) u.val)) +
        execArith.Xhat _ _ (execArith.omega (unPos (2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) u.val)) = _
      rw [hu]
      rfl
    refine Prod.ext (funext fun l => ?_) (funext fun r => funext fun i => ?_)
    · show ∑ k, ((coinedOf P Tz st).T c).lanes w b.1 l k * ((coinedOf P Tz st).T c).EU u k +
          ∑ j, b.2.1.1 l j * ((coinedOf P Tz st).T c).GpU u j = _
      simp only [hE, hG]
      rfl
    · show ∑ k, b.2.1.2.1 r i k * ((coinedOf P Tz st).T c).EU u k +
          ∑ j, b.2.1.2.2 r i j * ((coinedOf P Tz st).T c).GpU u j = _
      simp only [hE, hG]
      rfl

end FlockSoundness.Discharge.Composed
