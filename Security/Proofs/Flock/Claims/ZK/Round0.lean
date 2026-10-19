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
* `FlockComplete.row0_eq_encodeRowsZk`: a column whose lanes are the encoded words of `g`, `extra` and `lanePad` has
  `encodeRowsZk`'s row at `q` as its `row0`.
* `Claims.ZKMal.Masked.zkLevel0`: every `--zk` schedule's level 0 has `d = cols + rate`, `rate ≥ 1`, `lanes = k₀` and
  `t_pad ≤ 2^cols`, and is the model schedule's level 0 (one `decide +kernel` per `m`, `round0Facts_all`).
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

/-! ## A masked level-0 row from its column -/

namespace FlockComplete

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Composed

local notation "𝔽" => FlockLevel3.GF128

theorem row0_size {nL : ℕ} (col : (Fin nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽)) : (ZkView.row0 col).size = nL + 4 := by
  show (Array.ofFn col.1 ++ #[col.2 0 0, col.2 0 1, col.2 1 0, col.2 1 1] : Array 𝔽).size = _
  simp

theorem row0_left {nL : ℕ} (col : (Fin nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽)) {l : ℕ} (hl : l < nL) (h) :
    (ZkView.row0 col)[l]'h = col.1 ⟨l, hl⟩ := by
  show (Array.ofFn col.1 ++ #[col.2 0 0, col.2 0 1, col.2 1 0, col.2 1 1] : Array 𝔽)[l]'(by simp; omega) = _
  rw [Array.getElem_append_left (by simpa using hl), Array.getElem_ofFn]

theorem row0_right {nL : ℕ} (col : (Fin nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽)) {e : ℕ} (he : e < 4) (h) :
    (ZkView.row0 col)[nL + e]'h = #[col.2 0 0, col.2 0 1, col.2 1 0, col.2 1 1][e]'(by simpa using he) := by
  show (Array.ofFn col.1 ++ #[col.2 0 0, col.2 0 1, col.2 1 0, col.2 1 1] : Array 𝔽)[nL + e]'(by simp; omega) = _
  rw [Array.getElem_append_right (by simp)]
  simp

/-- **A masked level-0 row is the model's row of its column** (`ZkView.row0`), where the column's witness lanes are
`g`'s words and its extra lanes `extra`'s, each encoded at `ω_q` and padded by `lanePad`'s coefficients. -/
theorem row0_eq_encodeRowsZk {nL L tp : ℕ} (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128)
    {q : ℕ} (hq : q < 2 ^ lv.d) (hnL : nL = 2 ^ lv.lanes) (hL : L = 2 ^ lv.cols) (htp : tp = tPad)
    (ht : tPad ≤ 2 ^ lv.cols) (col : (Fin nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽))
    (h1 : ∀ l : Fin nL, col.1 l = ∑ c : Fin L, fOf g[c.val * 2 ^ lv.lanes + l.val]! * eAt lv.cols q c +
      ∑ j : Fin tp, fOf (lanePad l.val j.val) * gAt lv.cols q j)
    (h2 : ∀ r i : Fin 2, col.2 r i = ∑ c : Fin L, fOf (extra (2 * r.val + i.val) c.val) * eAt lv.cols q c +
      ∑ j : Fin tp, fOf (lanePad (2 ^ lv.lanes + (2 * r.val + i.val)) j.val) * gAt lv.cols q j) :
    ZkView.row0 col = (encodeRowsZk lv tPad g extra lanePad)[q]! := by
  subst nL L tp
  have hsz := encodeRowsZk_size lv tPad g extra lanePad hq
  have hE : Zk.EXTRA_LANES = 4 := rfl
  refine Array.ext (by rw [row0_size, hsz, hE]) fun l h1' h2' => ?_
  have hl : l < 2 ^ lv.lanes + Zk.EXTRA_LANES := by rw [hE]; rwa [row0_size] at h1'
  rw [← getElem!_pos (encodeRowsZk lv tPad g extra lanePad)[q]! l h2']
  show fOf _ = fOf _
  rw [encodeRowsZk_entry lv tPad g extra lanePad hq hl ht, Finset.sum_range]
  by_cases hlt : l < 2 ^ lv.lanes
  · simp only [hlt, ↓reduceIte]
    rw [row0_left col hlt h1']
    exact h1 ⟨l, hlt⟩
  · simp only [hlt, ↓reduceIte]
    obtain ⟨e, he, rfl⟩ : ∃ e, e < 4 ∧ l = 2 ^ lv.lanes + e := ⟨l - 2 ^ lv.lanes, by omega, by omega⟩
    rw [row0_right col he h1']
    simp only [Nat.add_sub_cancel_left]
    interval_cases e
    · exact h2 0 0
    · exact h2 0 1
    · exact h2 1 0
    · exact h2 1 1

end FlockComplete

/-! ## The `--zk` schedule's level 0 -/

namespace Claims.ZKMal.Masked

open FlockSoundness FlockSoundness.Refine

/-- The level-0 facts round `0` reads, as one check per `m`: level 0 exists, its codeword block is `cols + rate`
with `rate ≥ 1`, its lanes are the schedule's `k₀`, and `t_pad ≤ 2^cols`. -/
def round0Facts (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule m with
  | .ok (s, t) =>
    decide (0 < s.levels.size) &&
      decide ((s.levels.getD 0 default).d = (s.levels.getD 0 default).cols + (s.levels.getD 0 default).rate) &&
      decide (1 ≤ (s.levels.getD 0 default).rate) && decide ((s.levels.getD 0 default).lanes = s.k0) &&
      decide (t ≤ 2 ^ (s.levels.getD 0 default).cols)
  | .error _ => true

set_option maxRecDepth 100000 in
theorem round0Facts_all : ∀ m, 25 ≤ m → m ≤ 35 → round0Facts m = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

theorem schedOf_level₀ (s : _root_.Flock.Schedule) (h0 : 0 < s.levels.size) :
    (schedOf s).level₀ = levelOf (s.levels.getD 0 default) := by
  rcases s with ⟨m, n, k0, ⟨l⟩, ext⟩
  cases l with
  | nil => simp at h0
  | cons a l => rfl

/-- **A `--zk` schedule's level 0**: its codeword block is `cols + rate` with `rate ≥ 1`, its lanes are `k₀`, and
`t_pad ≤ 2^cols`; and it is the model schedule's level 0. -/
theorem zkLevel0 {m : ℕ} {s : _root_.Flock.Schedule} {t : ℕ} (h : _root_.Flock.Zk.schedule m = .ok (s, t)) :
    (s.levels.getD 0 default).d = (s.levels.getD 0 default).cols + (s.levels.getD 0 default).rate ∧
      1 ≤ (s.levels.getD 0 default).rate ∧ (s.levels.getD 0 default).lanes = s.k0 ∧
      t ≤ 2 ^ (s.levels.getD 0 default).cols ∧ (schedOf s).level₀ = levelOf (s.levels.getD 0 default) := by
  have hf := round0Facts_all m (FlockSoundness.Discharge.ZkHidden.schedule_range h).1 (FlockSoundness.Discharge.ZkHidden.schedule_range h).2
  unfold round0Facts at hf
  rw [h] at hf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hf
  obtain ⟨⟨⟨⟨h0, hd⟩, hr⟩, hl⟩, ht⟩ := hf
  exact ⟨hd, hr, hl, ht, schedOf_level₀ s h0⟩

end Claims.ZKMal.Masked
