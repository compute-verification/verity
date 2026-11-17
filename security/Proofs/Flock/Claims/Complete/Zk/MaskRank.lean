module

public import Proofs.Flock.Level3.ArithFacts
public import Proofs.Flock.Level3.GF256
public import Proofs.Flock.Soundness.Discharge.Composed.Base
public import Proofs.Flock.Claims.Deploy

/-! The mask slot's rank, at which the `--zk` prover refuses rank-deficient coins (`maskRankOk`), and the rank bound's
statement (`RankBound`). -/

@[expose] public section

namespace FlockComplete.MaskRank

section Ring

variable {A : Type*} [CommRing A]

def eqProd {D : ℕ} (c : Fin D → A) (x : Fin D → Bool) : A := ∏ t, if x t then c t else 1 + c t

def span {D : ℕ} (c : Fin D → A) : AddSubgroup A := AddSubgroup.closure (Set.range (eqProd c))

theorem eqProd_cons {D : ℕ} (c₀ : A) (c : Fin D → A) (b : Bool) (x : Fin D → Bool) :
    eqProd (Fin.cons c₀ c : Fin (D + 1) → A) (Fin.cons b x) = (if b then c₀ else 1 + c₀) * eqProd c x := by
  simp only [eqProd, Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ]

theorem range_eqProd_cons {D : ℕ} (c₀ : A) (c : Fin D → A) :
    Set.range (eqProd (Fin.cons c₀ c : Fin (D + 1) → A)) =
      (fun v => (1 + c₀) * v) '' Set.range (eqProd c) ∪ (fun v => c₀ * v) '' Set.range (eqProd c) := by
  ext y
  constructor
  · rintro ⟨x, rfl⟩
    rw [← Fin.cons_self_tail x, eqProd_cons]
    cases x 0
    · exact Or.inl ⟨_, ⟨Fin.tail x, rfl⟩, by simp⟩
    · exact Or.inr ⟨_, ⟨Fin.tail x, rfl⟩, by simp⟩
  · rintro (⟨v, ⟨x, rfl⟩, rfl⟩ | ⟨v, ⟨x, rfl⟩, rfl⟩)
    · exact ⟨Fin.cons false x, by rw [eqProd_cons]; simp⟩
    · exact ⟨Fin.cons true x, by rw [eqProd_cons]; simp⟩

theorem span_cons (h2 : ∀ a : A, a + a = 0) {D : ℕ} (c₀ : A) (c : Fin D → A) :
    span (Fin.cons c₀ c : Fin (D + 1) → A) = span c ⊔ (span c).map (AddMonoidHom.mulLeft c₀) := by
  apply le_antisymm
  · rw [span, AddSubgroup.closure_le, range_eqProd_cons]
    rintro y (⟨v, hv, rfl⟩ | ⟨v, hv, rfl⟩)
    · show (1 + c₀) * v ∈ _
      rw [add_mul, one_mul]
      exact AddSubgroup.add_mem _ (AddSubgroup.mem_sup_left (AddSubgroup.subset_closure hv))
        (AddSubgroup.mem_sup_right ⟨v, AddSubgroup.subset_closure hv, rfl⟩)
    · exact AddSubgroup.mem_sup_right ⟨v, AddSubgroup.subset_closure hv, rfl⟩
  · refine sup_le ?_ ?_
    · rw [span, AddSubgroup.closure_le]
      intro v hv
      have e : v = (1 + c₀) * v + c₀ * v := by rw [add_mul, one_mul, add_assoc, h2, add_zero]
      rw [SetLike.mem_coe, e]
      refine AddSubgroup.add_mem _ (AddSubgroup.subset_closure ?_) (AddSubgroup.subset_closure ?_)
      · rw [range_eqProd_cons]; exact Or.inl ⟨v, hv, rfl⟩
      · rw [range_eqProd_cons]; exact Or.inr ⟨v, hv, rfl⟩
    · rw [span, AddMonoidHom.map_closure, AddSubgroup.closure_le]
      rintro _ ⟨v, hv, rfl⟩
      apply AddSubgroup.subset_closure
      rw [range_eqProd_cons]
      exact Or.inr ⟨v, hv, rfl⟩

theorem span_map {B : Type*} [CommRing B] (π : A →+* B) {D : ℕ} (c : Fin D → A) :
    (span c).map π.toAddMonoidHom = span (fun t => π (c t)) := by
  rw [span, AddMonoidHom.map_closure, span, ← Set.range_comp]
  congr 2
  funext x
  simp [eqProd, map_prod, apply_ite]

theorem span_top_of_map {B : Type*} [CommRing B] (π : A →+* B) (hπ : Function.Surjective π) {D : ℕ}
    (c : Fin D → A) (h : span c = ⊤) : span (fun t => π (c t)) = ⊤ := by
  rw [← span_map, h]
  exact AddSubgroup.map_top_of_surjective _ hπ

end Ring

open FlockLevel3

def RankFull {k D : ℕ} (p : Fin k → Fin D → GF128) : Prop := span (fun t i => p i t) = ⊤

theorem add_self_pi {k : ℕ} (a : Fin k → GF128) : a + a = 0 := funext fun i => CharTwo.add_self_eq_zero (a i)

/-- **Full rank at `k` points gives it at any sub-family**: the projection to the sub-family is a surjective ring hom.
So the prover's cumulative refusal (rep 0's `ab`, `c`; then all four points at rep 1) is the one event "the last
check's points are rank-deficient". -/
theorem rankFull_sub {j k D : ℕ} (e : Fin j ↪ Fin k) (p : Fin k → Fin D → GF128) (h : RankFull p) :
    RankFull (fun i => p (e i)) := by
  let π : (Fin k → GF128) →+* (Fin j → GF128) := RingHom.pi fun i => Pi.evalRingHom _ (e i)
  have hπ : Function.Surjective π := fun b =>
    ⟨Function.extend e b 0, funext fun i' => by
      simp only [π, RingHom.pi_apply, Pi.evalRingHom_apply]
      exact e.injective.extend_apply b 0 i'⟩
  have key : (fun t => π (fun i => p i t)) = fun t i => p (e i) t := by
    funext t i
    simp only [π, RingHom.pi_apply, Pi.evalRingHom_apply]
  unfold RankFull
  rw [← key]
  exact span_top_of_map π hπ _ h

/-- The prover's check evaluates `eq` over the whole word index, so the slot's fixed in-block bits scale point `i`'s
weights by `γ i`; the refusal's other cause is a coin that zeroes one of those factors. -/
theorem rankFull_scale_iff {k D : ℕ} (γ : Fin k → GF128) (hγ : ∀ i, γ i ≠ 0) (p : Fin k → Fin D → GF128) :
    AddSubgroup.closure (Set.range fun x => γ * eqProd (fun t i => p i t) x) = ⊤ ↔ RankFull p := by
  have hl : γ⁻¹ * γ = 1 := funext fun i => inv_mul_cancel₀ (hγ i)
  have hr : γ * γ⁻¹ = 1 := funext fun i => mul_inv_cancel₀ (hγ i)
  have hinj : Function.Injective (AddMonoidHom.mulLeft γ) := fun x y hxy => by
    have := congrArg (γ⁻¹ * ·) hxy
    simpa only [AddMonoidHom.coe_mulLeft, ← mul_assoc, hl, one_mul] using this
  have hsur : Function.Surjective (AddMonoidHom.mulLeft γ) := fun y =>
    ⟨γ⁻¹ * y, by rw [AddMonoidHom.coe_mulLeft, ← mul_assoc, hr, one_mul]⟩
  have hmap : AddSubgroup.closure (Set.range fun x => γ * eqProd (fun t i => p i t) x) =
      (span fun t i => p i t).map (AddMonoidHom.mulLeft γ) := by
    rw [span, AddMonoidHom.map_closure, ← Set.range_comp]
    rfl
  have htop : (⊤ : AddSubgroup (Fin k → GF128)).map (AddMonoidHom.mulLeft γ) = ⊤ :=
    AddSubgroup.map_top_of_surjective _ hsur
  unfold RankFull
  rw [hmap]
  exact ⟨fun h => AddSubgroup.map_injective hinj (h.trans htop.symm), fun h => h ▸ htop⟩

open Classical in
def RankBound (k D : ℕ) (fix : Fin k → Fin D → Option GF128) (κ : ℕ) : Prop :=
  (Finset.univ.filter fun r : Fin k → Fin D → GF128 => ¬ RankFull fun i t => (fix i t).getD (r i t)).card * 2 ^ κ ≤
    Fintype.card (Fin k → Fin D → GF128)

/-- The deployed table's four points, in order rep 0's `ab`, `c`, rep 1's `ab`, `c`, at `m = 25`, `mask_log = 14`
(`N = 1024` words, `D = 10`): a `c` claim's first six word coordinates are the pinned `pinned[1..7)`, every other
coordinate a coin of its own. -/
def deployedFix (i : Fin 4) (t : Fin 10) : Option GF128 :=
  if i.val % 2 = 1 ∧ t.val < 6 then some (Flock.pinned[t.val + 1]!) else none

/-- At most `2^-122 = 64·2^-128` of the coins refuse: the part of a table's `4·m·2^-128` that the ten coordinates the
bridge chooses take. -/
def DeployedRankBound : Prop := RankBound 4 10 deployedFix 122

open Flock (F128)
open FlockSoundness.Discharge.Composed (eqWord maskWordsOf)

/-- The sum of the values `g j` at the selected `j`, in `F128`. -/
def sumSel {n : ℕ} (f : Fin n → Bool) (g : Fin n → F128) : F128 :=
  (List.finRange n).foldl (fun acc j => if f j then acc + g j else acc) 0

/-- `F128`'s values, finitely many (`GF128`'s). -/
abbrev f128Fintype : Fintype F128 := instFintypeGF128

section
attribute [local instance] f128Fintype

/-- **The prover's rank check** (`zk_veil::mask_rank_ok`): every vector of `F128^k` is the sum of some of the weight
vectors `(eq(p_1, x), …, eq(p_k, x))` over the slot's words `x`, i.e. they have GF(2)-rank `128·k`. Decided by
enumeration, which nothing evaluates. -/
def maskRankOk (pts : Array (List F128)) (words : List ℕ) : Bool :=
  decide (∀ a : Fin pts.size → F128, ∃ f : Fin words.length → Bool,
    ∀ i, sumSel f (fun j => eqWord pts[i] words[j]) = a i)

def weights (pts : Array (List F128)) (x : ℕ) : Fin pts.size → GF128 := fun i => GF128.of (eqWord pts[i] x)

theorem of_foldl_sel {n : ℕ} (f : Fin n → Bool) (g : Fin n → F128) :
    ∀ (l : List (Fin n)) (acc : F128), GF128.of (l.foldl (fun acc j => if f j then acc + g j else acc) acc) =
      GF128.of acc + (l.map fun j => if f j then GF128.of (g j) else 0).sum
  | [], acc => by simp
  | j :: l, acc => by
    rw [List.foldl_cons, of_foldl_sel f g l, List.map_cons, List.sum_cons]
    split
    · rw [← GF128.add_eq, add_assoc]
    · rw [zero_add]

theorem of_sumSel {n : ℕ} (f : Fin n → Bool) (g : Fin n → F128) :
    GF128.of (sumSel f g) = ∑ j, if f j then GF128.of (g j) else 0 := by
  rw [sumSel, of_foldl_sel, ← GF128.zero_eq, zero_add, Fin.sum_univ_def]

theorem maskRankOk_iff (pts : Array (List F128)) (words : List ℕ) :
    maskRankOk pts words = true ↔
      AddSubgroup.closure (Set.range fun j : Fin words.length => weights pts words[j]) = ⊤ := by
  let w : Fin words.length → Fin pts.size → GF128 := fun j => weights pts words[j]
  have hsel : ∀ (f : Fin words.length → Bool) (i : Fin pts.size),
      GF128.of (sumSel f fun j => eqWord pts[i] words[j]) = (∑ j, if f j then w j else 0) i := by
    intro f i
    rw [of_sumSel, Finset.sum_apply]
    refine Finset.sum_congr rfl fun j _ => ?_
    split <;> rfl
  unfold maskRankOk
  rw [decide_eq_true_iff]
  constructor
  · intro h
    rw [eq_top_iff]
    intro a _
    obtain ⟨f, hf⟩ := h a
    have ha : a = ∑ j, if f j then w j else 0 := funext fun i => (hsel f i).symm.trans (hf i) |>.symm
    rw [ha]
    refine AddSubgroup.sum_mem _ fun j _ => ?_
    split
    · exact AddSubgroup.subset_closure ⟨j, rfl⟩
    · exact zero_mem _
  · intro h a
    have key : ∀ b ∈ AddSubgroup.closure (Set.range w), ∃ f : Fin words.length → Bool,
        b = ∑ j, if f j then w j else 0 := by
      intro b hb
      induction hb using AddSubgroup.closure_induction with
      | mem x hx =>
        obtain ⟨j₀, rfl⟩ := hx
        refine ⟨fun j => decide (j = j₀), ?_⟩
        simp
      | zero => exact ⟨fun _ => false, by simp⟩
      | add x y _ _ hx hy =>
        obtain ⟨f, rfl⟩ := hx
        obtain ⟨g, rfl⟩ := hy
        refine ⟨fun j => xor (f j) (g j), ?_⟩
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun j _ => ?_
        by_cases hf : f j <;> by_cases hg : g j <;> simp [hf, hg, add_self_pi]
      | neg x _ hx =>
        obtain ⟨f, hf⟩ := hx
        exact ⟨f, by rw [neg_eq_of_add_eq_zero_right (add_self_pi x), hf]⟩
    obtain ⟨f, hf⟩ := key (fun i => GF128.of (a i)) (h ▸ AddSubgroup.mem_top _)
    exact ⟨f, fun i => (hsel f i).trans (congrFun hf i).symm⟩

end

theorem maskRankOk_append_left {A B : Array (List F128)} {words : List ℕ} (h : maskRankOk (A ++ B) words = true) :
    maskRankOk A words = true := by
  rw [maskRankOk_iff] at h ⊢
  let e : Fin A.size → Fin (A ++ B).size := fun i => ⟨i.1, by rw [Array.size_append]; omega⟩
  let π : (Fin (A ++ B).size → GF128) →+* (Fin A.size → GF128) := RingHom.pi fun i => Pi.evalRingHom _ (e i)
  have hπ : Function.Surjective π := fun b =>
    ⟨fun i => if hi : i.1 < A.size then b ⟨i.1, hi⟩ else 0, funext fun i => by
      simp only [π, e, RingHom.pi_apply, Pi.evalRingHom_apply, i.2, dite_true]⟩
  have hw : ∀ x, π (weights (A ++ B) x) = weights A x := by
    intro x
    funext i
    simp only [π, e, RingHom.pi_apply, Pi.evalRingHom_apply, weights, Fin.getElem_fin]
    rw [Array.getElem_append_left]
  rw [eq_top_iff, ← AddSubgroup.map_top_of_surjective π.toAddMonoidHom hπ, ← h, AddMonoidHom.map_closure,
    ← Set.range_comp]
  refine AddSubgroup.closure_mono ?_
  rintro _ ⟨k, rfl⟩
  exact ⟨k, (hw _).symm⟩

/-- **A rep's two claim points** (`claim_word_points`) as word coordinates (`point_parts`: each claim's point but its
first coordinate), at its coins, `coin j e` the element `e` of the rep's round `j` in `Zk.replay`'s numbering
(`Lincheck.coinAt`): the `ab` claim at `t_{k_log − 7}, …, t_0` (the lincheck's round coins, round `5 + (m − 6) + j`)
then the zerocheck's `ρ_j` for `k_log − 6 ≤ j < m − 6` (round `3 + j`), and the `c` claim at `pinned ++ r_outer`
(round `1`'s `m − 13` coins), as `abcClaims` has them. -/
def claimPoints (m kLog : ℕ) (coin : ℕ → ℕ → F128) : Array (List F128) :=
  let rho := (Array.range (m - 6)).map fun j => coin (3 + j) 0
  let ts := (Array.range (kLog - 6)).map fun j => coin (5 + (m - 6) + j) 0
  #[(ts.reverse ++ rho.extract (kLog - 6) (m - 6)).toList.drop 1,
    (Flock.pinned ++ (Array.range (m - 13)).map (coin 1)).toList.drop 1]

/-- A rep's two claim points read off its rounds `R`, the rep's first at `0`. -/
def roundPoints (m kLog : ℕ) (R : Array Flock.Round) : Array (List F128) :=
  claimPoints m kLog fun j e => (R[j]!.coins).getD e 0

theorem claimPoints_size (m kLog : ℕ) (coin : ℕ → ℕ → F128) : (claimPoints m kLog coin).size = 2 := rfl

/-- Each table's mask slot's words, the session's rank checks read. -/
def masksOf (sts : Array Flock.Stmt) : Array (List ℕ) := sts.map maskWordsOf

theorem masksOf_getElem! {sts : Array Flock.Stmt} {i : ℕ} (hi : i < sts.size) :
    (masksOf sts)[i]! = maskWordsOf sts[i]! := by
  rw [getElem!_pos (masksOf sts) i (by simpa [masksOf] using hi), getElem!_pos sts i hi]
  simp [masksOf]

open FlockDeploy (Coin setupOf Ins)

def claimsAt (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n : ℕ) : Array (List F128) :=
  claimPoints st.m st.c.kLog fun j e => (Claims.coinSeq r (n + j)).at e

def MaskRankAt (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n₀ n₁ : ℕ) : Prop :=
  maskRankOk (claimsAt st r n₀ ++ claimsAt st r n₁) (maskWordsOf st) = true

def LayoutOk (st : Flock.Stmt) : Prop :=
  Flock.Zk.shapeOk st = .ok () ∧ st.m = st.c.kLog + st.nbl ∧
    ∀ r ∈ st.c.ranges.toList, r.kind = Flock.Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog

def RankBoundAt (st : Flock.Stmt) : Prop :=
  ∀ k n₀ n₁ : ℕ, n₀ + (st.m + st.c.kLog) ≤ n₁ →
    Claims.Pr (fun r : Fin k → Coin => n₁ + (st.m + st.c.kLog) ≤ k ∧ ¬ MaskRankAt st r n₀ n₁) ≤ 4 * st.m / 2 ^ 128

/-- `RankBoundAt` at every table of the layout the setup admits; it holds (`maskRankBound`). -/
def MaskRankBound : Prop := ∀ st : Flock.Stmt, LayoutOk st → RankBoundAt st

theorem maskWordsOf_length (st : Flock.Stmt) (r : Flock.Range)
    (hr : st.c.ranges.find? (·.kind == Flock.Kind.mask) = some r) :
    (maskWordsOf st).length = 2 ^ st.nbl * 2 ^ (r.slotLog - 7) := by
  unfold maskWordsOf
  rw [← Array.find?_toList] at hr
  rw [hr]
  simp [List.length_flatMap]

theorem maskWords_ge {st : Flock.Stmt} (h : Flock.Zk.shapeOk st = .ok ()) :
    ∃ D, 10 ≤ D ∧ (maskWordsOf st).length = 2 ^ D := by
  unfold Flock.Zk.shapeOk at h
  obtain ⟨_, -, h⟩ := FlockSoundness.except_bind_ok h
  cases hr : st.c.ranges.find? (·.kind == Flock.Kind.mask) with
  | none => simp [hr] at h
  | some r =>
    simp only [hr] at h
    refine ⟨st.nbl + (r.slotLog - 7), ?_, by rw [maskWordsOf_length st r hr, pow_add]⟩
    by_contra hlt
    have hle : 2 ^ (st.nbl + (r.slotLog - 7)) ≤ 2 ^ 9 := Nat.pow_le_pow_right (by norm_num) (by omega)
    rw [pow_add] at hle
    split at h
    · cases h
    · split at h
      · cases h
      · rename_i hge
        simp only [Flock.Zk.RANK_POINTS, Flock.Zk.RANK_SLACK] at hge
        omega

end FlockComplete.MaskRank
