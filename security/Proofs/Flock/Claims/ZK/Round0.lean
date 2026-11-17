module

public import Proofs.Flock.Claims.Complete.ZkTables.Round0
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.ZK.OneTape
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Replay.LigeritoSum
import Proofs.Flock.Soundness.Discharge.Composed.Setup
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched

/-! Round `0` of the masked session: the level-0 columns by tree position. -/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL

local notation "𝔽" => FlockLevel3.GF128

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

variable (P : LigProver) {mPts : ℕ}

theorem coinedOf_s0_val (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (ht : Tz.t ≤ 2 ^ (Tz.sch.level₀.logCols + 1)) (hL : Tz.sch.level₀.logCols + 1 ≤ Tz.sch.level₀.logLen)
    (x : Fin (coinedOf P Tz st).z.nS ⊕ Fin (coinedOf P Tz st).z.nU) :
    (((coinedOf P Tz st).T c).s0 x).val =
      Sum.elim ((coinedOf P Tz st).pos c) (fun u => unPos (2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) u.val)
        x := by
  have h2 : 2 ^ (Tz.sch.level₀.logCols + 1) ≤ 2 ^ Tz.sch.level₀.logLen := Nat.pow_le_pow_right (by norm_num) hL
  exact treeEquiv_val_pos (N := 2 ^ Tz.sch.level₀.logLen) ((coinedOf P Tz st).pos c) (padPos_injective _ ht)
    (fun i => lt_of_lt_of_le (padPos_lt _ ht i) h2) (le_trans ht h2) x

noncomputable def eAt (cols p : ℕ) {L : ℕ} (c : Fin L) : 𝔽 := execArith.Xhat c.val (cols + 1) (execArith.omega p)

noncomputable def gAt (cols p : ℕ) {t : ℕ} (j : Fin t) : 𝔽 :=
  fOf Zk.PAD_C * execArith.Xhat j.val (cols + 1) (execArith.omega p) +
    execArith.Xhat (2 ^ cols + j.val) (cols + 1) (execArith.omega p)

noncomputable def colAt {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (cols : ℕ)
    (w : W) (b : ZkView.Base z Hi) (p : ℕ) : (Fin z.nL → 𝔽) × (Fin 2 → Fin 2 → 𝔽) :=
  (fun l => ∑ c, T.lanes w b.1 l c * eAt cols p c + ∑ j, b.2.1.1 l j * gAt cols p j,
    fun r i => ∑ c, b.2.1.2.1 r i c * eAt cols p c + ∑ j, b.2.1.2.2 r i j * gAt cols p j)

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

theorem coinedOf_lanes (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest) :
    ((coinedOf P Tz st).T c).lanes =
      fun w u l col => wordOf (zWith st w u) (l.val + (coinedOf P Tz st).z.nL * col.val) :=
  rfl

theorem colAt_congr {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} {T T' : ZkView.Coined K W Hi VL z}
    (h : T.lanes = T'.lanes) (cols : ℕ) (w : W) (b : ZkView.Base z Hi) (p : ℕ) :
    colAt T cols w b p = colAt T' cols w b p := by
  unfold colAt
  rw [h]

end FlockSoundness.Discharge.Composed

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

theorem zOf_traces (sts : Array Stmt) (w : Claims.Flock.Wit) {j : ℕ} (hj : j < sts.size) (a : Fin (2 ^ sts[j].m)) :
    Flock.Readable.zOf sts.size w j sts[j].m a = Flock.Readable.toZ (ZkTables.traces sts w)[j]!.z[a.val]! := by
  have hs : sts[j]! = sts[j] := getElem!_pos sts j hj
  rw [ZkTables.traces, Refine.getElem!_range_map _ hj]
  simp only [traceOf, hs]
  rw [Refine.getElem!_range_map _ a.isLt, Nat.zero_add, Array.getD_eq_getD_getElem?, List.getElem?_toArray,
    ← List.getD_eq_getElem?_getD, Flock.Readable.tableWit_getD hj]
  rfl

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

namespace Claims.ZKMal.Masked

open FlockSoundness FlockSoundness.Refine

def round0Facts (wideLeaf : Bool) (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule wideLeaf m with
  | .ok (s, t) =>
    decide (0 < s.levels.size) &&
      decide ((s.levels.getD 0 default).d = (s.levels.getD 0 default).cols + (s.levels.getD 0 default).rate) &&
      decide (1 ≤ (s.levels.getD 0 default).rate) && decide ((s.levels.getD 0 default).lanes = s.k0) &&
      decide (t ≤ 2 ^ (s.levels.getD 0 default).cols) && decide (s.k0 + (s.levels.getD 0 default).cols + 7 = m)
  | .error _ => true

set_option maxRecDepth 100000 in
theorem round0Facts_all : ∀ (w : Bool) (m : ℕ), 25 ≤ m → m ≤ 35 → round0Facts w m = true := by
  intro w m h1 h2
  cases w <;> interval_cases m <;> decide +kernel

theorem schedOf_level₀ (s : _root_.Flock.Schedule) (h0 : 0 < s.levels.size) :
    (schedOf s).level₀ = levelOf (s.levels.getD 0 default) := by
  rcases s with ⟨m, n, k0, ⟨l⟩, ext⟩
  cases l with
  | nil => simp at h0
  | cons a l => rfl

theorem zkLevel0 {wideLeaf : Bool} {m : ℕ} {s : _root_.Flock.Schedule} {t : ℕ}
    (h : _root_.Flock.Zk.schedule wideLeaf m = .ok (s, t)) :
    (s.levels.getD 0 default).d = (s.levels.getD 0 default).cols + (s.levels.getD 0 default).rate ∧
      1 ≤ (s.levels.getD 0 default).rate ∧ (s.levels.getD 0 default).lanes = s.k0 ∧
      t ≤ 2 ^ (s.levels.getD 0 default).cols ∧ (schedOf s).level₀ = levelOf (s.levels.getD 0 default) ∧
      s.k0 + (s.levels.getD 0 default).cols + 7 = m := by
  have hf := round0Facts_all wideLeaf m (FlockSoundness.Discharge.ZkHidden.schedule_range h).1
    (FlockSoundness.Discharge.ZkHidden.schedule_range h).2
  unfold round0Facts at hf
  rw [h] at hf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hf
  obtain ⟨⟨⟨⟨⟨h0, hd⟩, hr⟩, hl⟩, ht⟩, hp⟩ := hf
  exact ⟨hd, hr, hl, ht, schedOf_level₀ s h0, hp⟩

open FlockSoundness.Discharge

variable {wideLeaf : Bool}

/-- Level 0 of table `j`'s `--zk` schedule. -/
noncomputable def lv0 (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Level := (zkTab x j).sch.levels.getD 0 default

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

theorem col0At_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest)
    (w : Model.Witness (tabC P x j).S.m) (b : ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (p : Fin ((tabC P x j).z.nS + (tabC P x j).z.nU)) :
    ((tabC P x j).T c).col0At w b p = Composed.colAt ((tabC P x j).T c) (lv0 x j).cols w b p.val := by
  obtain ⟨ht, hL, hc⟩ := tabSizes x j
  rw [← hc]
  exact Composed.coinedOf_col0At P (zkTab x j).table (stAt x j) c ht hL w b p

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

theorem schedsPZk_live {x : Msg} (hx : Live x) :
    ∃ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs ∧
      ∀ j : Fin (jOf x), schs[j.val]! = ((zkTab x j).sch, (zkTab x j).tPad) := by
  let g : _root_.Flock.Setup → _root_.Flock.Schedule × ℕ := fun st =>
    match _root_.Flock.Zk.schedule st.wideLeaf st.m with
    | .ok v => v
    | .error _ => default
  have hsch : ∀ i (hi : i < (stsOf x).size),
      _root_.Flock.Zk.schedule (stsOf x)[i].wideLeaf (stsOf x)[i].m = Except.ok (g (_root_.Flock.Setup.ofCircuit (stsOf x)[i])) := by
    intro i hi
    obtain ⟨y⟩ := hx.2 i hi
    show _ = Except.ok (match _root_.Flock.Zk.schedule (stsOf x)[i].wideLeaf (stsOf x)[i].m with
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
    show (match _root_.Flock.Zk.schedule (stAt x j).wideLeaf (stAt x j).m with
      | .ok v => v
      | .error _ => default) = _
    rw [(zkTab x j).hz]

open Classical in
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
