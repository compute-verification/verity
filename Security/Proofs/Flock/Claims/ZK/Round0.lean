import Proofs.Flock.Claims.ZK.OneTape
import Proofs.Flock.Claims.Complete.ZkTables.Round0
import Proofs.Flock.Claims.Complete.ZkTables.Scope

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
* `FlockComplete.wordOf_packed`, `FlockComplete.zOf_traces`: a table's lanes, the model's packed words of its share of
  the witness, are `packed`'s entries of the executable's trace of that share.
* `Claims.ZKMal.Masked.enc0`: round `0`'s encoding, the link message of each table's level-0 cap built from the cut's
  level-0 leaves by tree position (`leafAt`, through the coins' tree `s0`), the shape of `honestPR_first`.
* `Claims.ZKMal.Masked.enc0_causal`: round `0` reads no coin. A cut's leaf at tree position `q` is the leaf at `q`
  whatever the coins' tree (`leafAt_cutAt`), and a level-0 leaf's digest is the same at every coin vector
  (`digOf_inl`, from `coinedOf_col0At` and `coinedOf_lanes`).
* `Claims.ZKMal.Masked.honest_round0`: **round `0`'s honest message is `enc0`'s** at a live statement, at every coin
  vector, where `hOf H` is SHA-512 and the layout agrees with the executable's tape at round `0` (`Agrees0`: the lanes'
  paddings, the extra lanes and their paddings at `2^lanes + e`, level 0's salts). Each table's cap is `layerOf` of the
  same leaves (`exeOf_inl`: the model's level-0 leaf is `hm96Sha512.leaf` of `encodeRowsZk`'s row, its lanes the
  executable's packed table, `lanes_tabC`), at each table's `ZkTab` schedule (`schedsPZk_live`). Where the statement sets
  up no session, both are empty.
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

/-- **A `--zk` table's lanes read no coin**: at every coin vector, lane `l`'s column `col` is the packed word
`l + n_L·col` of the witness with its mask slot (`innerOf`). -/
theorem coinedOf_lanes (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).lanes =
      fun w u l col => wordOf (zWith st w u) (l.val + (coinedOf P Tz st).z.nL * col.val) :=
  rfl

/-- `colAt` reads a table only through its lanes. -/
theorem colAt_congr {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} {T T' : ZkView.Coined K W Hi VL z}
    (h : T.lanes = T'.lanes) (cols : ℕ) (w : W) (b : ZkView.Base z Hi) (p : ℕ) :
    colAt T cols w b p = colAt T' cols w b p := by
  unfold colAt
  rw [h]

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

/-- **The model's packed word is the executable's** (`Composed.wordOf`, `packed`): where the witness reads the trace's
bits, its word `y` is `packed`'s entry `y`. -/
theorem wordOf_packed (tr : Trace) {M : ℕ} (zw : Model.Witness M)
    (hz : ∀ a : Fin (2 ^ M), zw a = Flock.Readable.toZ tr.z[a.val]!) {y : ℕ} (hy : y < 2 ^ (M - 7)) (hM : 7 ≤ M) :
    Composed.wordOf zw y = fOf (packed tr M)[y]! := by
  rw [Replay.Lg.fOf_packed tr M y hy, Composed.wordOf]
  show FlockLevel3.pack _ = _
  rw [Refine.pack_eq_sum, Finset.sum_range]
  refine Finset.sum_congr rfl fun v _ => ?_
  have e : 2 ^ M = 2 ^ 7 * 2 ^ (M - 7) := by rw [← pow_add]; congr 1; omega
  have hlt : 128 * y + v.val < 2 ^ M := by rw [e]; have := v.isLt; norm_num; omega
  simp only [hlt, ↓reduceDIte, hz]
  cases tr.z[128 * y + v.val]! <;> simp [Flock.Readable.toZ, Replay.Lg.bF]

/-- **A table's share of the witness is its trace's bits** (`Readable.zOf`, `ZkTables.traces`). -/
theorem zOf_traces (sts : Array Stmt) (w : Claims.Flock.Wit) {j : ℕ} (hj : j < sts.size) (a : Fin (2 ^ sts[j].m)) :
    Flock.Readable.zOf sts.size w j sts[j].m a = Flock.Readable.toZ (ZkTables.traces sts w)[j]!.z[a.val]! := by
  have hs : sts[j]! = sts[j] := getElem!_pos sts j hj
  rw [ZkTables.traces, Refine.getElem!_range_map _ hj]
  simp only [traceOf, hs]
  rw [Refine.getElem!_range_map _ a.isLt, Nat.zero_add, Array.getD_eq_getD_getElem?, List.getElem?_toArray,
    ← List.getD_eq_getElem?_getD, Flock.Readable.tableWit_getD hj]
  rfl

/-- `layerOf` reads its leaves below `2^d` only. -/
theorem layerOf_congr (nd : ByteArray → ByteArray → ByteArray) (d : ℕ) {f g : ℕ → ByteArray}
    (h : ∀ q < 2 ^ d, f q = g q) : ∀ k, layerOf nd d f k = layerOf nd d g k
  | 0 => by
    show (Array.range (2 ^ d)).map f = (Array.range (2 ^ d)).map g
    refine Array.ext (by simp) fun i h1 _ => ?_
    simp only [Array.getElem_map, Array.getElem_range]
    exact h i (by simpa using h1)
  | k + 1 => by
    simp only [layerOf, layerOf_congr nd d h k]

end FlockComplete

/-! ## The `--zk` schedule's level 0 -/

namespace Claims.ZKMal.Masked

open FlockSoundness FlockSoundness.Refine

/-- The level-0 facts round `0` reads, as one check per `m`: level 0 exists, its codeword block is `cols + rate`
with `rate ≥ 1`, its lanes are the schedule's `k₀`, `t_pad ≤ 2^cols`, and its lanes and columns pack the table's
`2^(m−7)` words. -/
def round0Facts (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule m with
  | .ok (s, t) =>
    decide (0 < s.levels.size) &&
      decide ((s.levels.getD 0 default).d = (s.levels.getD 0 default).cols + (s.levels.getD 0 default).rate) &&
      decide (1 ≤ (s.levels.getD 0 default).rate) && decide ((s.levels.getD 0 default).lanes = s.k0) &&
      decide (t ≤ 2 ^ (s.levels.getD 0 default).cols) && decide (s.k0 + (s.levels.getD 0 default).cols + 7 = m)
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
      t ≤ 2 ^ (s.levels.getD 0 default).cols ∧ (schedOf s).level₀ = levelOf (s.levels.getD 0 default) ∧
      s.k0 + (s.levels.getD 0 default).cols + 7 = m := by
  have hf := round0Facts_all m (FlockSoundness.Discharge.ZkHidden.schedule_range h).1
    (FlockSoundness.Discharge.ZkHidden.schedule_range h).2
  unfold round0Facts at hf
  rw [h] at hf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hf
  obtain ⟨⟨⟨⟨⟨h0, hd⟩, hr⟩, hl⟩, ht⟩, hp⟩ := hf
  exact ⟨hd, hr, hl, ht, schedOf_level₀ s h0, hp⟩

/-! ## Round `0`'s encoding -/

open FlockSoundness.Discharge

/-- Level 0 of table `j`'s `--zk` schedule. -/
noncomputable def lv0 (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Level := (zkTab x j).sch.levels.getD 0 default

/-- **A `--zk` table's level-0 sizes** as `Composed.coinedOf_col0At` reads them: `t_pad ≤ 2^(cols+1)` and
`cols + 1 ≤ logLen`, at `cols` level 0's. -/
theorem tabSizes (x : Msg) (j : Fin (jOf x)) :
    (zkTab x j).table.t ≤ 2 ^ ((zkTab x j).table.sch.level₀.logCols + 1) ∧
      (zkTab x j).table.sch.level₀.logCols + 1 ≤ (zkTab x j).table.sch.level₀.logLen ∧
      (zkTab x j).table.sch.level₀.logCols = (lv0 x j).cols := by
  obtain ⟨-, hr, -, ht, hl, -⟩ := zkLevel0 (zkTab x j).hz
  have hc : (zkTab x j).table.sch.level₀.logCols = (lv0 x j).cols := by
    show (schedOf (zkTab x j).sch).level₀.logCols = _
    rw [hl]
    rfl
  have hL : (zkTab x j).table.sch.level₀.logLen = (lv0 x j).cols + (lv0 x j).rate := by
    show (schedOf (zkTab x j).sch).level₀.logLen = _
    rw [hl]
    rfl
  refine ⟨?_, ?_, hc⟩
  · rw [hc]
    exact le_trans ht (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ _))
  · rw [hc, hL]
    exact Nat.add_le_add_left hr _

/-- **Table `j`'s level-0 column at tree position `p`** is `colAt` at `ω_p`, at every coin vector. -/
theorem col0At_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest)
    (w : Model.Witness (tabC P x j).S.m) (b : ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (p : Fin ((tabC P x j).z.nS + (tabC P x j).z.nU)) :
    ((tabC P x j).T c).col0At w b p = Composed.colAt ((tabC P x j).T c) (lv0 x j).cols w b p.val := by
  obtain ⟨ht, hL, hc⟩ := tabSizes x j
  rw [← hc]
  exact Composed.coinedOf_col0At P (zkTab x j).table (stAt x j) c ht hL w b p

/-- **Table `j`'s sizes from its level 0**: `n_L = 2^lanes` witness lanes of `2^cols` columns, `t_pad` padding columns
(`t_pad ≤ 2^cols`), `2^d` level-0 positions, and the lanes and columns pack the table's `2^(m−7)` words. -/
theorem tabC_z (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) :
    (tabC P x j).z.nL = 2 ^ (lv0 x j).lanes ∧ (tabC P x j).z.L = 2 ^ (lv0 x j).cols ∧
      (tabC P x j).z.tpad = (zkTab x j).tPad ∧ (zkTab x j).tPad ≤ 2 ^ (lv0 x j).cols ∧
      (tabC P x j).z.nS + (tabC P x j).z.nU = 2 ^ (lv0 x j).d ∧
      (lv0 x j).lanes + (lv0 x j).cols + 7 = (tabC P x j).S.m := by
  obtain ⟨hd, hr, hl, ht, h0, hp⟩ := zkLevel0 (zkTab x j).hz
  obtain ⟨-, -, hc⟩ := tabSizes x j
  simp only [lv0] at hc ⊢
  have hL : (zkTab x j).table.sch.level₀.logLen = ((zkTab x j).sch.levels.getD 0 default).d := by
    show (schedOf (zkTab x j).sch).level₀.logLen = _
    rw [h0, hd]
    rfl
  refine ⟨?_, ?_, rfl, ht, ?_, ?_⟩
  · show 2 ^ (schedOf (zkTab x j).sch).k0 = _
    rw [show (schedOf (zkTab x j).sch).k0 = (zkTab x j).sch.k0 from rfl, ← hl]
  · show 2 ^ (zkTab x j).table.sch.level₀.logCols = _
    rw [hc]
  · show (zkTab x j).tPad + (2 ^ (zkTab x j).table.sch.level₀.logLen - (zkTab x j).tPad) = _
    rw [hL]
    have : (zkTab x j).tPad ≤ 2 ^ ((zkTab x j).sch.levels.getD 0 default).d :=
      le_trans ht (Nat.pow_le_pow_right (by norm_num) (by omega))
    omega
  · show _ = (stAt x j).m
    rw [hl]
    exact hp

/-- **Table `j`'s level-0 leaf at tree position `q`** of a cut at the coins `cz`: the cut's opened or hidden leaf the
coins' tree places at `q` (`Coined.s0`), and the empty string past the tree. -/
noncomputable def leafAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) (j : Fin (jOf x)) (q : ℕ) : ByteArray :=
  if hq : q < (tabC P x j).z.nS + (tabC P x j).z.nU then
    match ((tabC P x j).T (cz j)).s0.symm ⟨q, hq⟩ with
    | .inl a => (cut j).leaves0.1 a
    | .inr u => (cut j).leaves0.2 u
  else ByteArray.empty

/-- **Round `0`'s encoding**: the link message of the tables' level-0 caps, each built from the cut's level-0 leaves by
tree position with SHA-512 nodes (`FlockComplete.layerOf`), the shape of `honestPR`'s first message
(`FlockComplete.honestPR_first`); the empty message where the statement sets up no session (`ZkTables.prog`). -/
noncomputable def enc0 (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) : Msg :=
  if (FlockDeploy.setupOf true x).isNone then [] else
  0 :: FlockDeploy.linkBody (List.ofFn fun j => FlockComplete.flat
      (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lv0 x j).d (leafAt P x cz cut j)
        ((lv0 x j).d - (lv0 x j).cap)))
    (List.replicate (jOf x) ByteArray.empty)

/-- **A cut's level-0 leaf at tree position `q` is the leaf at `q`**, whatever the coins' tree. -/
theorem leafAt_cutAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC P x j).z) → ByteArray) (j : Fin (jOf x)) {q : ℕ}
    (hq : q < (tabC P x j).z.nS + (tabC P x j).z.nU) :
    leafAt P x cz (cutAt (tabC P x) cz wt ω lv) j q = lv ⟨j, .inl ⟨q, hq⟩⟩ := by
  unfold leafAt
  rw [dite_eq_left_of_eq_true (eq_true hq)]
  show (match ((tabC P x j).T (cz j)).s0.symm ⟨q, hq⟩ with
    | .inl a => lv ⟨j, .inl (((tabC P x j).T (cz j)).s0 (.inl a))⟩
    | .inr u => lv ⟨j, .inl (((tabC P x j).T (cz j)).s0 (.inr u))⟩) = _
  rcases h : ((tabC P x j).T (cz j)).s0.symm ⟨q, hq⟩ with a | u
  · have e : ((tabC P x j).T (cz j)).s0 (.inl a) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]
  · have e : ((tabC P x j).T (cz j)).s0 (.inr u) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]

/-- **A level-0 leaf's digest is the same at every coin vector** (`col0At_tabC`). -/
theorem digOf_inl (P : Composed.LigProver) (x : Msg) (cz cz' : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi) (j : Fin (jOf x))
    (p : Fin ((tabC P x j).z.nS + (tabC P x j).z.nU)) :
    digOf (tabC P x) cz wt ω ⟨j, .inl p⟩ = digOf (tabC P x) cz' wt ω ⟨j, .inl p⟩ := by
  have hl : ((tabC P x j).T (cz j)).lanes = ((tabC P x j).T (cz' j)).lanes :=
    (Composed.coinedOf_lanes P (zkTab x j).table (stAt x j) (cz j)).trans
      (Composed.coinedOf_lanes P (zkTab x j).table (stAt x j) (cz' j)).symm
  unfold digOf
  simp only [ZkView.Coined.content]
  rw [col0At_tabC, col0At_tabC, Composed.colAt_congr hl]

/-- **Round `0` reads no coin** (`RoundPlan.Causal` at round `0`): its encoding of the cut at any coins, at the
simulator's leaves `H(dig ⊕ b, c)`, is the same at every coin vector. -/
theorem enc0_causal (P : Composed.LigProver) (x : Msg) (cz cz' : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC P x j).z) → ZkView.Bits × ByteArray) :
    enc0 P x cz (cutAt (tabC P x) cz wt ω fun l => ZkView.leafH (digOf (tabC P x) cz wt ω l + (p l).1, (p l).2)) =
      enc0 P x cz' (cutAt (tabC P x) cz' wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz' wt ω l + (p l).1, (p l).2)) := by
  have h : ∀ j, leafAt P x cz (cutAt (tabC P x) cz wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz wt ω l + (p l).1, (p l).2)) j =
      leafAt P x cz' (cutAt (tabC P x) cz' wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz' wt ω l + (p l).1, (p l).2)) j := fun j => funext fun q => by
    by_cases hq : q < (tabC P x j).z.nS + (tabC P x j).z.nU
    · rw [leafAt_cutAt P x cz wt ω _ j hq, leafAt_cutAt P x cz' wt ω _ j hq, digOf_inl P x cz cz' wt ω j]
    · simp only [leafAt, hq, ↓reduceDIte]
  unfold enc0
  simp only [h]

/-! ## Round `0`'s honest message -/

/-- **A layout agrees with the executable at round `0`**: at every statement's table `j`, the masked prover's tape
(`TapePlan.ofLayout`'s `tapeZk`, here `z`'s) holds the lanes' paddings, the extra lanes and their paddings at the
executable's indices (`encodeRowsZk`: extra lane `e`'s padding at `2^lanes + e`), and level 0's salts, that the model's
base and leaves' salts hold. -/
structure Agrees0 (P : Composed.LigProver) (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) : Prop where
  pad : ∀ x t (j : Fin (jOf x)) (l : Fin (tabC P x j).z.nL) (k : Fin (tabC P x j).z.tpad),
    (((TapePlan.ofLayout L z).tape x t).1 j).2.1.1 l k = fOf ((z t).lanePad j.val l.val k.val)
  ext : ∀ x t (j : Fin (jOf x)) (r i : Fin 2) (c : Fin (tabC P x j).z.L),
    (((TapePlan.ofLayout L z).tape x t).1 j).2.1.2.1 r i c = fOf ((z t).extra j.val (2 * r.val + i.val) c.val)
  extPad : ∀ x t (j : Fin (jOf x)) (r i : Fin 2) (k : Fin (tabC P x j).z.tpad),
    (((TapePlan.ofLayout L z).tape x t).1 j).2.1.2.2 r i k =
      fOf ((z t).lanePad j.val (2 ^ (lv0 x j).lanes + (2 * r.val + i.val)) k.val)
  salt : ∀ x t (j : Fin (jOf x)) (p : Fin ((tabC P x j).z.nS + (tabC P x j).z.nU)),
    ZkView.saltBytes (((TapePlan.ofLayout L z).tape x t).2.1 ⟨j, .inl p⟩) =
      FlockComplete.fitBytes 192 ((z t).toTape.salt j.val 0 0 p.val)

/-- **Table `j`'s lanes at a layout's tape are the executable's packed table**: lane `l`'s column `c` is word
`c·2^lanes + l` of `tablesP` at the traces of the witness with its mask slots the tape's (`zOf_ofLayout`). -/
theorem lanes_tabC (P : Composed.LigProver) (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) {x : Msg} (w : Claims.Flock.Wit)
    (t : Fin Claims.Flock.zkTableCap → L.Cell) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest)
    (l : Fin (tabC P x j).z.nL) (col : Fin (tabC P x j).z.L) :
    ((tabC P x j).T c).lanes (wtOf P x w j) (((TapePlan.ofLayout L z).tape x t).1 j).1 l col =
      fOf ((FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
        (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout L z).tapeZk t).mask)))[j.val]!)[col.val *
          2 ^ (lv0 x j).lanes + l.val]! := by
  obtain ⟨hnL, hL, -, -, -, hm⟩ := tabC_z P x j
  have hj := lt_of_jOf j
  rw [show ((tabC P x j).T c).lanes = fun w u l col =>
      Composed.wordOf (Composed.zWith (stAt x j) w u) (l.val + (tabC P x j).z.nL * col.val) from
    Composed.coinedOf_lanes P (zkTab x j).table (stAt x j) c]
  show Composed.wordOf (Composed.zWith (stAt x j) (wtOf P x w j) (((TapePlan.ofLayout L z).tape x t).1 j).1)
    (l.val + (tabC P x j).z.nL * col.val) = _
  rw [← zOf_ofLayout L z x t w j,
    show (tabC P x j).z.nL * col.val = 2 ^ (lv0 x j).lanes * col.val by rw [hnL]]
  have hy : l.val + 2 ^ (lv0 x j).lanes * col.val < 2 ^ ((tabC P x j).S.m - 7) := by
    have h1 : l.val < 2 ^ (lv0 x j).lanes := lt_of_lt_of_eq l.isLt hnL
    have h2 : col.val < 2 ^ (lv0 x j).cols := lt_of_lt_of_eq col.isLt hL
    rw [← hm, show (lv0 x j).lanes + (lv0 x j).cols + 7 - 7 = (lv0 x j).lanes + (lv0 x j).cols by omega, pow_add]
    calc l.val + 2 ^ (lv0 x j).lanes * col.val < 2 ^ (lv0 x j).lanes * (col.val + 1) := by rw [mul_add]; omega
      _ ≤ 2 ^ (lv0 x j).lanes * 2 ^ (lv0 x j).cols := Nat.mul_le_mul_left _ h2
  rw [FlockComplete.wordOf_packed (M := (tabC P x j).S.m) _ (Flock.Readable.zOf (stsOf x).size
    (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout L z).tapeZk t).mask) j.val (tabC P x j).S.m)
    (fun a => FlockComplete.zOf_traces (stsOf x) _ hj a) hy (by omega)]
  rw [FlockComplete.tablesP, FlockSoundness.Refine.getElem!_range_map _ (by simpa [FlockDeploy.tabsOf] using hj)]
  have e : (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = (tabC P x j).S.m := by
    rw [getElem!_pos _ _ (by simpa [FlockDeploy.tabsOf] using hj)]
    simp [FlockDeploy.tabsOf, _root_.Flock.Setup.ofCircuit]
    rfl
  rw [e, Nat.add_comm, Nat.mul_comm]

/-- **Table `j`'s level-0 leaf at tree position `p` is the masked prover's**: at a layout agreeing at round `0`, the
executable's leaf of the cut's view (`exeOf`) is `hm96-sha512/v1`'s leaf of `encodeRowsZk`'s row `p` under the tape's
level-0 salt `p`, at every coin vector. -/
theorem exeOf_inl (P : Composed.LigProver) (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (hA : Agrees0 P L z) {x : Msg}
    (w : Claims.Flock.Wit) (t : Fin Claims.Flock.zkTableCap → L.Cell) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (j : Fin (jOf x)) (p : Fin ((tabC P x j).z.nS + (tabC P x j).z.nU)) :
    exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout L z).tape x t).1 ((TapePlan.ofLayout L z).tape x t).2.1
        ⟨j, .inl p⟩ =
      _root_.Flock.MerkleScheme.hm96Sha512.leaf
        (FlockComplete.encodeRowsZk (lv0 x j) (zkTab x j).tPad
          ((FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
            (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout L z).tapeZk t).mask)))[j.val]!)
          ((z t).extra j.val) ((z t).lanePad j.val))[p.val]!
        (FlockComplete.fitBytes 192 ((z t).toTape.salt j.val 0 0 p.val)) := by
  obtain ⟨hnL, hL, htp, ht, hd, -⟩ := tabC_z P x j
  show _root_.Flock.MerkleScheme.hm96Sha512.leaf (ZkView.row0 (((tabC P x j).T (cz j)).col0At (wtOf P x w j)
      (((TapePlan.ofLayout L z).tape x t).1 j) p)) (ZkView.saltBytes (((TapePlan.ofLayout L z).tape x t).2.1 ⟨j, .inl p⟩)) = _
  rw [col0At_tabC, hA.salt x t j p]
  congr 1
  refine FlockComplete.row0_eq_encodeRowsZk (lv0 x j) _ _ _ _ (by rw [← hd]; exact p.isLt) hnL hL htp ht _
    (fun l => ?_) (fun r i => ?_)
  · show ∑ c, ((tabC P x j).T (cz j)).lanes (wtOf P x w j) (((TapePlan.ofLayout L z).tape x t).1 j).1 l c *
        Composed.eAt (lv0 x j).cols p.val c +
      ∑ k, (((TapePlan.ofLayout L z).tape x t).1 j).2.1.1 l k * Composed.gAt (lv0 x j).cols p.val k = _
    simp only [lanes_tabC]
    exact congrArg (_ + ·) (Finset.sum_congr rfl fun k _ => congrArg (· * _) (hA.pad x t j l k))
  · show ∑ c, (((TapePlan.ofLayout L z).tape x t).1 j).2.1.2.1 r i c * Composed.eAt (lv0 x j).cols p.val c +
      ∑ k, (((TapePlan.ofLayout L z).tape x t).1 j).2.1.2.2 r i k * Composed.gAt (lv0 x j).cols p.val k = _
    exact congrArg₂ (· + ·) (Finset.sum_congr rfl fun c _ => congrArg (· * _) (hA.ext x t j r i c))
      (Finset.sum_congr rfl fun k _ => congrArg (· * _) (hA.extPad x t j r i k))

/-- **Every `--zk` table's schedule at a live statement**: `schedsPZk` returns, table `j`'s the `ZkTab`'s. -/
theorem schedsPZk_live {x : Msg} (hx : Live x) :
    ∃ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs ∧
      ∀ j : Fin (jOf x), schs[j.val]! = ((zkTab x j).sch, (zkTab x j).tPad) := by
  let g : _root_.Flock.Setup → _root_.Flock.Schedule × ℕ := fun st =>
    match _root_.Flock.Zk.schedule st.m with
    | .ok v => v
    | .error _ => default
  have hsch : ∀ i (hi : i < (stsOf x).size),
      _root_.Flock.Zk.schedule (stsOf x)[i].m = Except.ok (g (_root_.Flock.Setup.ofCircuit (stsOf x)[i])) := by
    intro i hi
    obtain ⟨y⟩ := hx.2 i hi
    show _ = Except.ok (match _root_.Flock.Zk.schedule (stsOf x)[i].m with
      | .ok v => v
      | .error _ => default)
    rw [y.hz]
  refine ⟨(FlockDeploy.tabsOf (stsOf x)).map g, ?_, fun j => ?_⟩
  · unfold FlockComplete.schedsPZk
    refine FlockComplete.mapM_ok_mem g _ fun st hst => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hst
    have hi' : i < (stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact hsch i hi'
  · have hj := lt_of_jOf j
    rw [getElem!_pos ((FlockDeploy.tabsOf (stsOf x)).map g) j.val (by simpa [FlockDeploy.tabsOf] using hj)]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    show (match _root_.Flock.Zk.schedule (stAt x j).m with
      | .ok v => v
      | .error _ => default) = _
    rw [(zkTab x j).hz]

open Classical in
/-- **Round `0`'s honest message is `enc0`'s** (`HonestPreOn` before any opening), at a layout agreeing with the
executable at round `0` and where `hOf H` is SHA-512: at a live statement, `honestPR`'s first message, the link message
of the tables' level-0 caps (`FlockComplete.honestPR_first`, `FlockComplete.commitsPZk_caps`), is round `0`'s encoding
of the cut view at the tape, at every coin vector. -/
theorem honest_round0 (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) (P : Composed.LigProver)
    (L : TapeLayout P) (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (hA : Agrees0 P L z)
    {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (t : Fin Claims.Flock.zkTableCap → L.Cell)
    (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) :
    enc0 P x cz (cutAt (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout L z).tape x t).1
        (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout L z).tape x t).1
          ((TapePlan.ofLayout L z).tape x t).2.1)) =
      (((FlockComplete.ZkTables.honestPR w).next x [] ((TapePlan.ofLayout L z).tapeZk t)).eval H).1 := by
  cases hs0 : FlockDeploy.setupOf true x with
  | none =>
    rw [FlockComplete.ZkTables.honestPR_next (by simp [FlockComplete.ZkTables.outerHit])]
    change _ = ((FlockComplete.feed (FlockComplete.ZkTables.prog x (FlockComplete.ZkTables.maskWOf x w
      ((TapePlan.ofLayout L z).tapeZk t).mask) ((TapePlan.ofLayout L z).tapeZk t)) []).eval H).1
    unfold enc0
    simp only [hs0, Option.isNone_none, ↓reduceIte, FlockComplete.ZkTables.prog]
    rfl
  | some v =>
    obtain ⟨s, spec, sts⟩ := v
    have hsts : stsOf x = sts := by unfold stsOf; rw [hs0]
    subst hsts
    obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs0
    have hJ : jOf x = (stsOf x).size := by unfold jOf; exact ite_eq_left_of_eq_true _ _ (eq_true hx)
    obtain ⟨schs, hs, hschs⟩ := schedsPZk_live hx
    have hms : ∀ i < (FlockDeploy.tabsOf (stsOf x)).size,
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
      intro i hi
      have hi' : i < (stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
      rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i hi]
      simp only [FlockDeploy.tabsOf, Array.getElem_map]
      exact (hall i hi').2.2
    obtain ⟨cms, hcm, hsize, hcaps⟩ := FlockComplete.commitsPZk_caps (FlockDeploy.tabsOf (stsOf x)) schs
      (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
        (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout L z).tapeZk t).mask)))
      ((TapePlan.ofLayout L z).tapeZk t) hms
    have hc : (FlockComplete.commitsPZk (_root_.Flock.Hashes.ofSha512 (Claims.hOf H)) (FlockDeploy.tabsOf (stsOf x))
        schs (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
          (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout L z).tapeZk t).mask)))
        ((TapePlan.ofLayout L z).tapeZk t) : _root_.Flock.V _) = .ok cms := by
      rw [hH]
      exact hcm
    rw [FlockComplete.honestPR_first H w x _ hs0 hs hc]
    have hts : (FlockDeploy.tabsOf (stsOf x)).size = jOf x := by rw [hJ]; simp [FlockDeploy.tabsOf]
    unfold enc0
    simp only [hs0, Option.isNone_some, Bool.false_eq_true, ↓reduceIte]
    refine congrArg (0 :: ·) (congrArg₂ FlockDeploy.linkBody ?_ ?_)
    · refine List.ext_getElem (by simp [hsize, hts]) fun n h1 h2 => ?_
      have hn : n < jOf x := by simpa using h1
      have hn' : n < cms.size := by rw [hsize, hts]; exact hn
      simp only [List.getElem_ofFn, Array.toList_map, List.getElem_map, Array.getElem_toList]
      rw [← getElem!_pos cms n hn', hcaps n (by rw [hts]; exact hn), hschs ⟨n, hn⟩]
      refine congrArg FlockComplete.flat (FlockComplete.layerOf_congr _ _ (fun q hq => ?_) _)
      obtain ⟨-, -, -, -, hd, -⟩ := tabC_z P x ⟨n, hn⟩
      have hq' : q < (tabC P x ⟨n, hn⟩).z.nS + (tabC P x ⟨n, hn⟩).z.nU := by rw [hd]; exact hq
      rw [leafAt_cutAt P x cz _ _ _ ⟨n, hn⟩ hq', exeOf_inl P L z hA w t cz ⟨n, hn⟩ ⟨q, hq'⟩]
      rfl
    · simp [hts]

end Claims.ZKMal.Masked
