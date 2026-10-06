import Proofs.Flock.Soundness.ZK.Table
import Mathlib.Tactic.FieldSimp

/-!
# Zero knowledge of one masked table: Lemmas A and B, and `table_shvzk`

For the model of `Table` (the reference prover `Table.view` and `S_shvzk` as `Table.sim`), at a fixed non-degenerate coin
vector:

* **The identity (★)** (§4.3), `star`: rep `r`'s two extra-lane entries at an opened position are determined by the
  witness lanes' entries there, `y₁'^r` and `ȳ^r`, because `W^r` is invertible.
* **Lemma B's rows** (§4.3), `card_realCore` and `card_simCore`: the prover's view is a public function (`psi`) of four
  components, each of `card_seq_masked`'s form: anything computed from earlier blocks plus a surjective image of a fresh
  block. Row (i) is the masked `ŝ` (the mask slot, onto by the rank check); rows (ii) and (iii) the witness lanes' entries
  at the opened positions with `(y₁'^r, ȳ^r)` (the paddings, onto by T4, and the extra lanes, onto because `W^r` is);
  rows (vi) and (vii) the masked values with `ρ_in`, `σ_in` (the pads); rows (viii)–(x) `C_h`, `Y₂`, `Y₁` (`pad_h` onto by
  T4, `pad_μ` and `μ` scaled by `β ≠ 0`), with the salts and ideal-leaf seeds. Rows (iv), (v), (xi) and (xii) are
  `psi`'s: the extra-lane entries by (★), levels ≥ 1 from `y₁'^r` and fresh salts, `τ` and `C_μ` from the replayed
  constraint, the leaves.
* **`table_shvzk`** (Lemma B in world `W₁`): the real view and `S_shvzk(x, e)` have the same distribution.
* **Lemma A at a non-degenerate coin vector** (§4.2), `table_prefinal_translate` and `table_prefinal_indep`: in world
  `W₀`, the pre-final views on two witnesses carrying the public region data agree along a translation of the blocks `u`,
  `R`, `h`, `μ` that reads only earlier blocks, so by `card_fiber_eq_of_triShift` they have the same distribution.

Everything here is about one table, at coins where the prover refuses nothing. The paper's lemmas are per session of `J`
tables: the product over tables (no claim or message reads two tables' witnesses or masks) is `Session`'s; Lemma A's
cases where the coins make the prover refuse (a failing rank check, dependent `ρ^r`, `β = 0`), with the schedule, are
`Refusal`'s; and Lemma C, which T5 builds from them, is `AdaptivePrefinal`'s.
-/

namespace FlockSoundness.ZK

open Finset

-- The randomness and the view's components are products of up to fifteen factors, and their `Fintype`, `DecidableEq`
-- and group instances exceed instance search's default size.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

/-! ## The identity (★) -/

theorem Wr_add (r : Fin 2) (a b a' b' : F) : T.Wr r (a + a') (b + b') = T.Wr r a b + T.Wr r a' b' := by
  simp only [Wr, map_add]
  ring

theorem Winv_Wr {r : Fin 2} (hW : Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (a b : F) :
    T.Winv r (T.Wr r a b) = (a, b) :=
  Function.leftInverse_invFun hW.1 (a, b)

theorem Wr_Winv {r : Fin 2} (hW : Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (x : K) :
    T.Wr r (T.Winv r x).1 (T.Winv r x).2 = x :=
  Function.rightInverse_invFun hW.2 x

/-- **The identity (★)** (§4.3): at every opened position, the folded witness lanes, rep `r`'s extra lanes weighted by
`ρ^r` and the padding of `ȳ^r` give the codeword of `y₁'^r`. Solved for the extra lanes: `W^r` of them is `xTarget`. -/
theorem star (w : W) (u : T.Mask) (μ0 : Fin T.nL → Fin T.tpad → F) (R : T.XLanes) (μR : T.XPad) (q : Fin T.nS)
    (r : Fin 2) :
    T.Wr r (T.xrows R μR q r 0) (T.xrows R μR q r 1) = T.xTarget (T.rows w u μ0) (T.y1' w u R) (T.ybar μ0 μR) q r := by
  have hA : ∑ c, T.y1 w u r c * T.embed (T.E q c)
      = ∑ l, T.rbar r l * T.embed (∑ c, T.lanes w u l c * T.E q c) := by
    simp only [y1, Finset.sum_mul, map_sum, map_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun l _ => Finset.sum_congr rfl fun c _ => by ring
  have hB : ∑ j, (∑ l, T.rbar r l * T.embed (μ0 l j)) * T.embed (T.Gp q j)
      = ∑ l, T.rbar r l * T.embed (∑ j, μ0 l j * T.Gp q j) := by
    simp only [Finset.sum_mul, map_sum, map_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun l _ => Finset.sum_congr rfl fun c _ => by ring
  simp only [xTarget, y1', ybar, rows, add_mul, Finset.sum_add_distrib, map_add, mul_add, hA, hB]
  have e : ∀ a b x y : K, a + x + (b + y) - (a + b) = x + y := fun a b x y => by ring
  rw [e]
  simp only [Wr, xrows, map_add, map_sum, map_mul, add_mul, Finset.sum_add_distrib, mul_add, Finset.mul_sum,
    mul_assoc]
  ring

/-! ## The view as a function of four masked components -/

/-- The four components the view is a function of (Lemma B's sampling order): the masked `ŝ`; the witness lanes' entries
at the opened positions with `(y₁'^r, ȳ^r)`; each rep's masked values with `(ρ_in, σ_in)`; each rep's `C_h`, `Y₂`, `Y₁`,
with every salt and ideal-leaf seed. -/
abbrev Core := (Fin T.nC → Fin 128 → F)
  × ((Fin T.nS → Fin T.nL → F) × (Fin 2 → Fin T.L → K) × (Fin 2 → Fin T.tpad → K))
  × (Fin 2 → (Fin T.nE → F) × F × F)
  × ((Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds)

/-- **The view from its components**, by the public computations of Lemma B's determined rows: the extra-lane entries by
(★) (row (iv)), levels ≥ 1 from `y₁'^r` and their salts (v), `τ` and `C_μ` from the replayed constraint (xi), and every
leaf (xii). -/
noncomputable def psi (c : T.Core) : T.View :=
  let sHat := c.1
  let rows := c.2.1.1
  let y1 := c.2.1.2.1
  let yb := c.2.1.2.2
  let masked : Fin 2 → Fin T.nE → F := fun r => (c.2.2.1 r).1
  let inner : Fin 2 → F × F := fun r => (c.2.2.1 r).2
  let ch := c.2.2.2.1
  let Y2 := c.2.2.2.2.1
  let Y1 := c.2.2.2.2.2.1
  let s := c.2.2.2.2.2.2
  let xrows : Fin T.nS → Fin 2 → Fin 2 → F := fun q r =>
    ![(T.Winv r (T.xTarget rows y1 yb q r)).1, (T.Winv r (T.xTarget rows y1 yb q r)).2]
  let lig := fun r => T.lig r (y1 r) (s.2.2.2.2.2.2.2.2 r)
  let pre := (sHat, masked, y1, yb, inner, lig)
  let tau := fun r => (∑ i, T.lam r pre i * Y1 r i - T.theta r pre) / T.β r
  let cm := fun r k => (∑ i, T.Ch r k i * Y1 r i + ∑ j, T.Cp r k j * Y2 r j - ch r k) / T.β r
  { sHat, sReg := T.pubReg, masked, y1, ybar := yb, lig, inner, Y1, Y2
    tau, ytau := s.2.2.2.2.2.2.1, tauCom := fun r => T.leafT (tau r) (s.2.2.2.2.2.2.1 r)
    rows, xrows, salts0 := s.1
    leaves0 := (fun q => T.leaf0 (rows q, xrows q) (s.1 q), fun q => T.ideal (s.2.2.1 q))
    ch, cm, saltsP := s.2.2.2.1
    leavesP := fun r => (fun k => T.leafP (ch r k, cm r k) (s.2.2.2.1 r k), fun k => T.ideal (s.2.2.2.2.2.1 r k)) }

/-! ## The masks -/

/-- Rows (ii) and (iii)'s mask: the paddings at the opened positions, `W^r` of the extra lanes, and `ȳ^r`. -/
def mask1 : ((Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad) →+
    ((Fin T.nS → Fin T.nL → F) × (Fin 2 → Fin T.L → K) × (Fin 2 → Fin T.tpad → K)) where
  toFun b := (fun q l => ∑ j, b.1 l j * T.Gp q j, fun r c => T.Wr r (b.2.1 r 0 c) (b.2.1 r 1 c),
    fun r j => ∑ l, T.rbar r l * T.embed (b.1 l j) + T.Wr r (b.2.2 r 0 j) (b.2.2 r 1 j))
  map_zero' := by
    simp only [Prod.fst_zero, Prod.snd_zero, Pi.zero_apply, zero_mul, Finset.sum_const_zero, map_zero, mul_zero,
      Wr, add_zero]
    rfl
  map_add' b b' := by
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · funext q l
      simp only [Prod.fst_add, Pi.add_apply, add_mul, Finset.sum_add_distrib]
    · funext r c
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Wr_add]
    · funext r j
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Wr_add, map_add, mul_add, Finset.sum_add_distrib]
      ring

/-- Rows (vi) and (vii)'s mask: each exposed pad, and `ρ_in`, `σ_in` through `h_tx`, `h_ty`. -/
def mask2 : (Fin 2 → (Fin T.nE → F) × F × F) →+ (Fin 2 → (Fin T.nE → F) × F × F) where
  toFun h r := ((h r).1, T.ζ r * (h r).1 T.ia + (h r).2.1, (h r).1 T.ib + (h r).2.2)
  map_zero' := by
    funext r
    simp
  map_add' h h' := by
    funext r
    simp only [Pi.add_apply, Prod.fst_add, Prod.snd_add, Prod.mk_add_mk]
    refine Prod.ext rfl (Prod.ext ?_ ?_) <;> simp only [] <;> ring

theorem mask1_apply (b : (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad) :
    T.mask1 b = (fun q l => ∑ j, b.1 l j * T.Gp q j, fun r c => T.Wr r (b.2.1 r 0 c) (b.2.1 r 1 c),
      fun r j => ∑ l, T.rbar r l * T.embed (b.1 l j) + T.Wr r (b.2.2 r 0 j) (b.2.2 r 1 j)) := rfl

theorem mask2_apply (h : Fin 2 → (Fin T.nE → F) × F × F) :
    T.mask2 h = fun r => ((h r).1, T.ζ r * (h r).1 T.ia + (h r).2.1, (h r).1 T.ib + (h r).2.2) := rfl

variable [CharP F 2] in
theorem maskHom_surjective (hrank : T.Rank) : Function.Surjective T.maskHom := by
  intro s
  choose c hc using fun v : Fin 128 => hrank fun k => s k v
  refine ⟨fun x v => c v x, funext fun k => funext fun v => ?_⟩
  exact congrFun (hc v) k

theorem mask1_surjective (hpad : T.PadOnto) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) :
    Function.Surjective T.mask1 := by
  rintro ⟨a, b, c⟩
  choose μ0 hμ0 using fun l => hpad fun q => a q l
  let R : T.XLanes := fun r i c' => ![(T.Winv r (b r c')).1, (T.Winv r (b r c')).2] i
  let μR : T.XPad := fun r i j =>
    ![(T.Winv r (c r j - ∑ l, T.rbar r l * T.embed (μ0 l j))).1,
      (T.Winv r (c r j - ∑ l, T.rbar r l * T.embed (μ0 l j))).2] i
  refine ⟨(μ0, R, μR), Prod.ext ?_ (Prod.ext ?_ ?_)⟩
  · funext q l
    exact congrFun (hμ0 l) q
  · funext r c'
    exact T.Wr_Winv (hW r) _
  · funext r j
    show ∑ l, T.rbar r l * T.embed (μ0 l j) + T.Wr r (μR r 0 j) (μR r 1 j) = c r j
    simp only [μR, Matrix.cons_val_zero, Matrix.cons_val_one, T.Wr_Winv (hW r)]
    ring

theorem mask2_surjective : Function.Surjective T.mask2 := by
  intro m
  refine ⟨fun r => ((m r).1, (m r).2.1 - T.ζ r * (m r).1 T.ia, (m r).2.2 - (m r).1 T.ib), funext fun r => ?_⟩
  show ((m r).1, T.ζ r * (m r).1 T.ia + ((m r).2.1 - T.ζ r * (m r).1 T.ia),
    (m r).1 T.ib + ((m r).2.2 - (m r).1 T.ib)) = m r
  refine Prod.ext rfl (Prod.ext ?_ ?_) <;> simp only [] <;> ring

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

/-- Rows (viii)–(x)'s mask, with the salts: `C_h` through `pad_h`, `Y₂` through `pad_μ`, `Y₁` through `μ`. -/
def mask3 : ((Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds) →+
    ((Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds) where
  toFun b := (fun r k => ∑ j, T.Cp r k j * b.1 r j, fun r j => b.1 r j + T.β r * b.2.1 r j,
    fun r i => T.β r * b.2.2.1 r i, b.2.2.2)
  map_zero' := by
    simp only [Prod.fst_zero, Prod.snd_zero, Pi.zero_apply, mul_zero, Finset.sum_const_zero, add_zero]
    rfl
  map_add' b b' := by
    refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl))
    · funext r k
      simp only [Prod.fst_add, Pi.add_apply, mul_add, Finset.sum_add_distrib]
    · funext r j
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply]
      ring
    · funext r i
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply]
      ring

theorem mask3_apply (b : (Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds) :
    T.mask3 b = (fun r k => ∑ j, T.Cp r k j * b.1 r j, fun r j => b.1 r j + T.β r * b.2.1 r j,
      fun r i => T.β r * b.2.2.1 r i, b.2.2.2) := rfl

theorem mask3_surjective (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0) : Function.Surjective T.mask3 := by
  rintro ⟨a, b, y, s⟩
  choose p hp using fun r => hpads r (a r)
  refine ⟨(p, fun r j => (b r j - p r j) / T.β r, fun r i => y r i / T.β r, s), Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl))⟩
  · funext r k
    exact congrFun (hp r) k
  · funext r j
    show p r j + T.β r * ((b r j - p r j) / T.β r) = b r j
    field_simp [hβ r]
    ring
  · funext r i
    show T.β r * (y r i / T.β r) = y r i
    field_simp [hβ r]

/-! ## Lemma B: the real view and `S_shvzk` -/

variable [CharP F 2]

/-- **The prover's view in Lemma B's sampling order**: each component is a function of the earlier blocks plus a mask of
its own block. -/
noncomputable def realCore (w : W) (ω : T.Rand) : T.Core :=
  (T.nu w + T.maskHom ω.1,
    (fun q l => ∑ c, T.lanes w ω.1 l c * T.E q c, fun r => T.y1 w ω.1 r, 0) + T.mask1 ω.2.1,
    (fun r => (T.vals r w ω.1 ω.2.1.2.1, 0, 0)) + T.mask2 ω.2.2.1,
    (fun r k => ∑ i, T.Ch r k i * T.hFull (ω.2.2.1 r) i, 0, fun r => T.hFull (ω.2.2.1 r), 0) + T.mask3 ω.2.2.2)

/-- `S_shvzk`'s last block: its fresh `C_h`, `Y₂`, `Y₁`, and the dummy run's salts and seeds. -/
def simMask3 : (((Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds)
      × (Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F)) →+
    ((Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds) where
  toFun b := (b.2.1, b.2.2.1, b.2.2.2, b.1.2.2.2)
  map_zero' := rfl
  map_add' _ _ := rfl

omit [CharP F 2] in
theorem simMask3_apply (b : ((Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds)
      × (Fin 2 → Fin T.nQ → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F)) :
    T.simMask3 b = (b.2.1, b.2.2.1, b.2.2.2, b.1.2.2.2) := rfl

/-- **`S_shvzk`'s output in the same order**: fresh values where the prover's are masked, the dummy run's where they are
kept. -/
noncomputable def simCore (w0 : W) (σ : T.SimRand) : T.Core :=
  (0 + AddMonoidHom.snd _ _ σ.1,
    (fun q l => ∑ c, T.lanes w0 σ.1.1 l c * T.E q c, fun r => T.y1 w0 σ.1.1 r, 0) + T.mask1 σ.2.1,
    0 + AddMonoidHom.snd _ _ σ.2.2.1,
    0 + T.simMask3 σ.2.2.2)

/-- **The real view is `psi` of its components**: the determined rows are what the prover computes. The extra-lane
entries by (★), `τ` by `⟨λ, h⟩ = θ`, `C_μ` by the linearity of the pads code; the region `ŝ` by H_reg. -/
theorem psi_realCore (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0)
    (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (ω : T.Rand) :
    T.psi (T.realCore w ω) = T.view w ω := by
  obtain ⟨u, ⟨μ0, R, μR⟩, h, ⟨padh, padμ, μ, s⟩⟩ := ω
  set P := (T.view w (u, (μ0, R, μR), h, padh, padμ, μ, s)).pre with hP
  have htau : ∀ r, (∑ i, T.lam r P i * (T.hFull (h r) i + T.β r * μ r i) - T.theta r P) / T.β r
      = ∑ i, T.lam r P i * μ r i := by
    intro r
    have hI : ∑ i, T.lam r P i * T.hFull (h r) i = T.theta r P := hinner (u, (μ0, R, μR), h, padh, padμ, μ, s) r
    have e : ∑ i, T.lam r P i * (T.hFull (h r) i + T.β r * μ r i)
        = T.theta r P + T.β r * ∑ i, T.lam r P i * μ r i := by
      rw [← hI, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [e, add_sub_cancel_left, mul_div_cancel_left₀ _ (hβ r)]
  have hcm : ∀ r k, (∑ i, T.Ch r k i * (T.hFull (h r) i + T.β r * μ r i)
      + ∑ j, T.Cp r k j * (padh r j + T.β r * padμ r j)
      - (∑ i, T.Ch r k i * T.hFull (h r) i + ∑ j, T.Cp r k j * padh r j)) / T.β r
      = ∑ i, T.Ch r k i * μ r i + ∑ j, T.Cp r k j * padμ r j := by
    intro r k
    have e1 : ∑ i, T.Ch r k i * (T.hFull (h r) i + T.β r * μ r i)
        = ∑ i, T.Ch r k i * T.hFull (h r) i + T.β r * ∑ i, T.Ch r k i * μ r i := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    have e2 : ∑ j, T.Cp r k j * (padh r j + T.β r * padμ r j)
        = ∑ j, T.Cp r k j * padh r j + T.β r * ∑ j, T.Cp r k j * padμ r j := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [e1, e2, div_eq_iff (hβ r)]
    ring
  have hx : ∀ q r, T.Winv r (T.xTarget (T.rows w u μ0) (T.y1' w u R) (T.ybar μ0 μR) q r)
      = (T.xrows R μR q r 0, T.xrows R μR q r 1) := fun q r => by
    rw [← T.star w u μ0 R μR q r, T.Winv_Wr (hW r)]
  have hxr : (fun q r => ![(T.Winv r (T.xTarget (T.rows w u μ0) (T.y1' w u R) (T.ybar μ0 μR) q r)).1,
      (T.Winv r (T.xTarget (T.rows w u μ0) (T.y1' w u R) (T.ybar μ0 μR) q r)).2]) = T.xrows R μR := by
    funext q r i
    rw [hx q r]
    fin_cases i <;> rfl
  apply View.ext
  all_goals simp only [psi, view, realCore, mask1_apply, mask2_apply, mask3_apply, Pi.add_apply, zero_add,
    Prod.mk_add_mk]
  all_goals (try rfl)
  · exact hreg.symm
  · funext r
    exact htau r
  · funext r
    exact congrArg (fun t => T.leafT t (s.2.2.2.2.2.2.1 r)) (htau r)
  · exact hxr
  · refine Prod.ext (funext fun q => ?_) rfl
    exact congrArg (fun x => T.leaf0 (T.rows w u μ0 q, x) (s.1 q)) (congrFun hxr q)
  · funext r k
    exact hcm r k
  · funext r
    refine Prod.ext (funext fun k => ?_) rfl
    exact congrArg (fun c => T.leafP (_, c) (s.2.2.2.1 r k)) (hcm r k)

/-- **`S_shvzk`'s output is `psi` of its components**: the kept extra-lane entries are the dummy run's, which (★) gives
from its other entries; `τ` and `C_μ` are computed as `psi` computes them. -/
theorem psi_simCore (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (w0 : W) (σ : T.SimRand) :
    T.psi (T.simCore w0 σ) = T.sim w0 σ := by
  obtain ⟨⟨u, s0⟩, ⟨μ0, R, μR⟩, ⟨h, fr⟩, ⟨⟨padh, padμ, μ, s⟩, ch, Y2, Y1⟩⟩ := σ
  have hx : ∀ q r, T.Winv r (T.xTarget (T.rows w0 u μ0) (T.y1' w0 u R) (T.ybar μ0 μR) q r)
      = (T.xrows R μR q r 0, T.xrows R μR q r 1) := fun q r => by
    rw [← T.star w0 u μ0 R μR q r, T.Winv_Wr (hW r)]
  have hxr : (fun q r => ![(T.Winv r (T.xTarget (T.rows w0 u μ0) (T.y1' w0 u R) (T.ybar μ0 μR) q r)).1,
      (T.Winv r (T.xTarget (T.rows w0 u μ0) (T.y1' w0 u R) (T.ybar μ0 μR) q r)).2]) = T.xrows R μR := by
    funext q r i
    rw [hx q r]
    fin_cases i <;> rfl
  apply View.ext
  all_goals simp only [psi, sim, view, simCore, dummyOf, mask1_apply, simMask3_apply, Pi.add_apply, zero_add,
    Prod.mk_add_mk]
  all_goals (try rfl)
  · exact hxr
  · refine Prod.ext (funext fun q => ?_) rfl
    exact congrArg (fun x => T.leaf0 (T.rows w0 u μ0 q, x) (s.1 q)) (congrFun hxr q)

/-! ## Lemma A: the pre-final view doesn't depend on the witness -/

/-- **Lemma A's translation at a non-degenerate coin vector** (§4.2), for one table: for witnesses `w`, `w'` that carry
the public region data, a translation of the prover's randomness maps `w`'s pre-final view to `w'`'s. The coins pass
every check the prover makes (`Rank`, `W^r` bijective, `β ≠ 0`); the cases where they don't are
`table_prefinal_translate_cut`'s, which uses the same amounts on what is sent before the prover refuses. Block by
block, each amount reads only earlier blocks:

1. the mask slot, `u' := u + Δ` with `Φ(Δ) = ν(w) − ν(w')`, which the rank check makes solvable (step 2);
2. the extra lanes, `R' := R + (W^r)⁻¹(y₁^r(w, u) − y₁^r(w', u'))`, so `y₁'^r` agrees (step 3); the paddings are unchanged
   (step 1);
3. the pads, `h'_i := h_i + v_i(w, u, R) − v_i(w', u', R')`, and `h'_tx`, `h'_ty` so that `ρ_in`, `σ_in` agree (step 4);
4. the inner mask, `μ' := μ + (h − h')/β` over the full pad vectors, so `Y₁` agrees (step 5); `pad_h`, `pad_μ` and every
   salt and seed are unchanged (step 1). -/
theorem table_prefinal_translate (hrank : T.Rank)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0)
    (w w' : W) (hreg : T.sReg w = T.pubReg) (hreg' : T.sReg w' = T.pubReg) :
    ∃ (d₀ : T.Mask) (d₁ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad)
      (d₂ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → Fin 2 → (Fin T.nE → F) × F × F)
      (d₃ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → (Fin 2 → (Fin T.nE → F) × F × F) →
        (Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds),
      ∀ ω, T.preView w ω = T.preView w' (ω.1 + d₀, ω.2.1 + d₁ ω.1, ω.2.2.1 + d₂ ω.1 ω.2.1,
        ω.2.2.2 + d₃ ω.1 ω.2.1 ω.2.2.1) := by
  obtain ⟨Δ, hΔ⟩ := T.maskHom_surjective hrank (T.nu w - T.nu w')
  let dR : T.Mask → T.XLanes := fun u r i c =>
    ![(T.Winv r (T.y1 w u r c - T.y1 w' (u + Δ) r c)).1, (T.Winv r (T.y1 w u r c - T.y1 w' (u + Δ) r c)).2] i
  let d₁ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad := fun u => (0, dR u, 0)
  let dv : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → Fin 2 → Fin T.nE → F := fun u b r =>
    T.vals r w u b.2.1 - T.vals r w' (u + Δ) (b.2.1 + dR u)
  let d₂ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → Fin 2 → (Fin T.nE → F) × F × F :=
    fun u b r => (dv u b r, -(T.ζ r * dv u b r T.ia), -(dv u b r T.ib))
  let d₃ : T.Mask → (Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad → (Fin 2 → (Fin T.nE → F) × F × F) →
      (Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) × T.Seeds :=
    fun u b h => (0, 0, fun r i => (T.hFull (h r) i - T.hFull ((h + d₂ u b) r) i) / T.β r, 0)
  refine ⟨Δ, d₁, d₂, d₃, fun ω => ?_⟩
  obtain ⟨u, ⟨μ0, R, μR⟩, h, ⟨padh, padμ, μ, s⟩⟩ := ω
  have hy : ∀ r c, T.y1' w u R r c = T.y1' w' (u + Δ) (R + dR u) r c := by
    intro r c
    have e : T.Wr r (dR u r 0 c) (dR u r 1 c) = T.y1 w u r c - T.y1 w' (u + Δ) r c := T.Wr_Winv (hW r) _
    simp only [y1', Pi.add_apply, Wr_add, e]
    ring
  have hy' : T.y1' w u R = T.y1' w' (u + Δ) (R + dR u) := funext fun r => funext fun c => hy r c
  apply PreView.ext
  all_goals simp only [preView, view, d₁, d₂, d₃, dv, Prod.mk_add_mk, Pi.add_apply, add_zero]
  case sHat =>
    rw [map_add, hΔ]
    abel
  case sReg => rw [hreg, hreg']
  case masked =>
    funext r i
    simp only [Prod.fst_add, Pi.add_apply, Pi.sub_apply]
    ring
  case y1 => exact hy'
  case lig => rw [hy']
  case inner =>
    funext r
    simp only [innerMsg, Prod.fst_add, Prod.snd_add, Pi.add_apply, Pi.sub_apply]
    refine Prod.ext ?_ ?_ <;> simp only [] <;> ring
  case Y1 =>
    funext r i
    field_simp [hβ r]
    ring

/-! ## Counting -/

variable [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
  [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]

instance instDecidableEqSeeds : DecidableEq T.Seeds := inferInstance
instance instFintypeSeeds : Fintype T.Seeds := inferInstance
instance instDecidableEqCore : DecidableEq T.Core := inferInstance
instance instFintypeCore : Fintype T.Core := inferInstance
instance instDecidableEqRand : DecidableEq T.Rand := inferInstance
instance instFintypeRand : Fintype T.Rand := inferInstance
instance instDecidableEqSimRand : DecidableEq T.SimRand := inferInstance
instance instFintypeSimRand : Fintype T.SimRand := inferInstance

omit [Fintype K] in
/-- **Lemma B's rows (i)–(iii) and (vi)–(x)**: the prover's view takes every value of its components equally often. -/
theorem card_realCore (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w : W) (v : T.Core) :
    #{ω | T.realCore w ω = v} = #{κ : T.Rand | T.maskHom κ.1 = 0 ∧ T.mask1 κ.2.1 = 0 ∧ T.mask2 κ.2.2.1 = 0 ∧
      T.mask3 κ.2.2.2 = 0} :=
  card_seq_masked4 T.maskHom T.mask1 T.mask2 T.mask3 (T.maskHom_surjective hrank) (T.mask1_surjective hpad hW)
    T.mask2_surjective (T.mask3_surjective hpads hβ) (T.nu w)
    (fun u => (fun q l => ∑ c, T.lanes w u l c * T.E q c, fun r => T.y1 w u r, 0))
    (fun u b => fun r => (T.vals r w u b.2.1, 0, 0))
    (fun _ _ h => (fun r k => ∑ i, T.Ch r k i * T.hFull (h r) i, 0, fun r => T.hFull (h r), 0)) v

omit [CharP F 2] [Fintype K] in
/-- **`S_shvzk` in the same form**: its output takes every value of the components equally often. -/
theorem card_simCore (hpad : T.PadOnto) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2)
    (w0 : W) (v : T.Core) :
    #{σ | T.simCore w0 σ = v} = #{κ : T.SimRand | AddMonoidHom.snd _ _ κ.1 = 0 ∧ T.mask1 κ.2.1 = 0 ∧
      AddMonoidHom.snd _ _ κ.2.2.1 = 0 ∧ T.simMask3 κ.2.2.2 = 0} :=
  card_seq_masked4 (AddMonoidHom.snd T.Mask (Fin T.nC → Fin 128 → F)) T.mask1
    (AddMonoidHom.snd (Fin 2 → (Fin T.nE → F) × F × F) (Fin 2 → (Fin T.nE → F) × F × F)) T.simMask3
    (fun b => ⟨(0, b), rfl⟩) (T.mask1_surjective hpad hW) (fun b => ⟨(0, b), rfl⟩)
    (fun b => ⟨((0, 0, 0, b.2.2.2), b.1, b.2.1, b.2.2.1), rfl⟩) 0
    (fun u : T.Mask × (Fin T.nC → Fin 128 → F) =>
      (fun q l => ∑ c, T.lanes w0 u.1 l c * T.E q c, fun r => T.y1 w0 u.1 r, 0)) (fun _ _ => 0)
    (fun _ _ _ => 0) v

/-- **Lemma B in world `W₁`** (`docs/zk-proof-public.md` §4.3; `table_shvzk`), for one table: at a fixed non-degenerate
coin vector, for a witness `w` whose region claims send the public values (H_reg) and whose pads satisfy the inner
proof's constraint (`InnerHolds`), the reference prover's view (`view`: the pre-final messages and the final message,
hm96 leaves opened or ideal) and `S_shvzk(x, e)` (`sim`, on any dummy witness `w₀`) have the same distribution.

Non-degenerate means the prover refuses nothing: the rank check passes (`Rank`), each `ρ^r` is independent over `F`
(`W^r` bijective), and `β^r ≠ 0`. `PadOnto` and `PadsOnto` are T4 for the two codes at their opened positions. -/
theorem table_shvzk (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (w0 : W) :
    SameDist (T.view w) (T.sim w0) :=
  ((SameDist.of_const (T.card_realCore hrank hpad hW hpads hβ w) (T.card_simCore hpad hW w0)).comp T.psi).of_eq
    (T.psi_realCore hW hβ w hreg hinner) (T.psi_simCore hW w0)

omit [Fintype K] [DecidableEq K] in
/-- **Lemma A at a non-degenerate coin vector** (§4.2, `card_fiber_eq_of_triShift` applied to its translation), for one
table: in world `W₀`, the pre-final views on any two witnesses carrying the public region data take every value equally
often. Lemma A needs no other property of the witnesses, so it holds for the dummy one. The coins where the prover refuses
are `table_prefinal_indep_cut`'s. -/
theorem table_prefinal_indep (hrank : T.Rank)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0)
    (w w' : W) (hreg : T.sReg w = T.pubReg) (hreg' : T.sReg w' = T.pubReg) :
    SameDist (T.preView w) (T.preView w') := by
  classical
  obtain ⟨d₀, d₁, d₂, d₃, h⟩ := T.table_prefinal_translate hrank hW hβ w w' hreg hreg'
  exact SameDist.of_card_fiber_eq (card_fiber_eq_of_triShift4 _ _ d₀ d₁ d₂ d₃ h)

end Table

end FlockSoundness.ZK
