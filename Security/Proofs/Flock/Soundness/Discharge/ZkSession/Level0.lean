import Proofs.Flock.Soundness.Discharge.ZkSession.Compose
import Proofs.Flock.Soundness.SoundnessPad

/-!
# C-Flock ZK's masked Ligerito level 0 (live/PROTOCOL.md §1, §4, §7)

Under `--zk` the prover sends level 0's out-of-domain value, its lane rounds' messages and the rep's extra-lane sums
`e^r` masked: `v' = v + h` with `h` a pad (§2: one `F` pad for the OOD value, four per lane message
`u₀.c0, u₀.c1, u₂.c0, u₂.c1`, two per `e_i`). The verifier can no longer compute the running claim, so it carries it as a
`K`-valued affine form in the pads (`LinK`, §4's `Lin2` read as one `K` value), receives the restart `T'` from the prover
(after `ρ`, before `ybar`, §1), and continues from `T'`. What it could not check becomes the lane-phase row
`claim_k + Σ_i ρ_i e_i + T'` (§4; written `claim_k + Σ_i ρ_i e_i − T'`, the same row in characteristic 2), two `F` rows by
its coordinates `K = F ⊕ F·u` (`limbRows`), which the inner proof enforces.

`ligeritoPadZK` is `Model.ligeritoPad` with those changes and nothing else; levels 1+ (`ligTail`) are verbatim. The
masked rep (`Algebraic.lean`) runs it after the masked algebraic phase.

**The coupling** (VEIL Lemma 4.3, "`v' − h` is a prover strategy for the unmasked protocol"): for every pad vector `h`,
masked level 0 ends with a true verdict and its row vanishing at `h` with probability at most the clear level 0's
acceptance probability (`ligeritoPadZK_couple`). Unmasking each message at `h` turns a masked strategy into a clear one;
the claim forms evaluate at `h` to the clear claims (`fold_couple`), and when the row vanishes `T'` is the clear
restart, after which the two games are the same game.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Field K]

/-! ## Simulation steps -/

/-- A shared prefix: continuations compared pointwise. -/
theorem value_bind_mono {α β γ : Type} (g : Game α) {f : α → Game β} {f' : α → Game γ} {p : β → Prop}
    {p' : γ → Prop} (h : ∀ a, value p (f a) ≤ value p' (f' a)) : value p (g.bind f) ≤ value p' (g.bind f') := by
  induction g with
  | ret a => exact h a
  | send M k ih => simp only [bind_send, value_send]; exact iSup_mono ih
  | coin C k ih =>
      simp only [bind_coin, value_coin]
      exact Finset.sum_le_sum fun c _ => by gcongr; exact ih c

/-- A message relabelled: each message of the first game is answered by some message of the second. -/
theorem value_recv_le {α β M M' : Type} {f : M → Game α} {f' : M' → Game β} {p : α → Prop} {p' : β → Prop}
    (h : ∀ m, ∃ m', value p (f m) ≤ value p' (f' m')) :
    value p ((recv M).bind f) ≤ value p' ((recv M').bind f') := by
  simp only [recv, bind_send, bind_ret, value_send]
  refine iSup_le fun m => ?_
  obtain ⟨m', hm⟩ := h m
  exact hm.trans (le_iSup (fun m' => value p' (f' m')) m')

/-! ## Forms over `K` -/

/-- A `K`-valued affine form in the pads `h : Fin np → F`: `Σ_i embed(h_i)·coef_i + const`. -/
abbrev LinK (K : Type) (np : ℕ) := (Fin np → K) × K

/-- The form's value at the pads `h`. -/
def LinK.eval (A : Arith F K) {np : ℕ} (h : Fin np → F) : LinK K np →ₗ[K] K where
  toFun ℓ := ∑ i, A.embed (h i) * ℓ.1 i + ℓ.2
  map_add' a b := by
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, mul_add, Finset.sum_add_distrib]
    ring
  map_smul' c a := by
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply, mul_add,
      Finset.mul_sum]
    congr 1
    exact Finset.sum_congr rfl fun i _ => by ring

/-- A constant form. -/
def cK {np : ℕ} (x : K) : LinK K np := (0, x)

/-- Pad `i` of an `F` value. -/
def padF {np : ℕ} (i : Fin np) : LinK K np := (Pi.single i 1, 0)

/-- The pad of a `K` value with coordinate pads `i₀, i₁`: `embed(h_{i₀}) + embed(h_{i₁})·u`. -/
def padK (A : Arith F K) {np : ℕ} (i₀ i₁ : Fin np) : LinK K np := (Pi.single i₀ 1 + Pi.single i₁ A.u, 0)

section Eval

variable (A : Arith F K) {np : ℕ} (h : Fin np → F)

theorem sum_mul_single (a : Fin np → K) (i : Fin np) (c : K) :
    ∑ j, a j * (Pi.single i c : Fin np → K) j = a i * c := by
  rw [Finset.sum_eq_single i (fun j _ hj => by rw [Pi.single_eq_of_ne hj, mul_zero])
    (fun hi => absurd (Finset.mem_univ i) hi), Pi.single_eq_same]

@[simp] theorem eval_cK (x : K) : LinK.eval A h (cK x) = x := by
  simp [LinK.eval, cK]

@[simp] theorem eval_padF (i : Fin np) : LinK.eval A h (padF i) = A.embed (h i) := by
  simp only [LinK.eval, padF, LinearMap.coe_mk, AddHom.coe_mk, add_zero]
  rw [sum_mul_single, mul_one]

@[simp] theorem eval_padK (i₀ i₁ : Fin np) : LinK.eval A h (padK A i₀ i₁) = A.ofLimbs (h i₀) (h i₁) := by
  simp only [LinK.eval, padK, LinearMap.coe_mk, AddHom.coe_mk, add_zero, Pi.add_apply, mul_add,
    Finset.sum_add_distrib]
  rw [sum_mul_single, sum_mul_single, mul_one, Arith.ofLimbs]

end Eval

/-! ## The lane-phase rows over `F` -/

/-- A `K` element's coordinates `(c0, c1)` with `x = c0 + c1·u` (§2.4; `K = F ⊕ F·u`). -/
noncomputable def coords (A : Arith F K) (x : K) : F × F :=
  Function.invFun (fun w : F × F => A.ofLimbs w.1 w.2) x

/-- The two `F` rows of a `K` row, coordinate by coordinate (§4's `ligerito lane phase (c0, c1)`). -/
noncomputable def limbRows (A : Arith F K) {np : ℕ} (row : LinK K np) : Fin 2 → Lin F np :=
  ![⟨fun i => (coords A (row.1 i)).1, (coords A row.2).1⟩, ⟨fun i => (coords A (row.1 i)).2, (coords A row.2).2⟩]

variable [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]

omit [DecidableEq F] [DecidableEq K] in
theorem ofLimbs_coords (A : Arith F K) (hA : A.Correct) (x : K) :
    A.ofLimbs (coords A x).1 (coords A x).2 = x :=
  Function.invFun_eq (f := fun w : F × F => A.ofLimbs w.1 w.2) (hA.ofLimbs_bijective.2 x)

/-- **Both coordinate rows vanish only if the `K` row does.** -/
theorem eval_eq_zero_of_limbRows (A : Arith F K) (hA : A.Correct) {np : ℕ} (row : LinK K np) (h : Fin np → F)
    (hr : ∀ k, (limbRows A row k).eval h = 0) : LinK.eval A h row = 0 := by
  have h0 := hr 0
  have h1 := hr 1
  simp only [limbRows, Lin.eval, Matrix.cons_val_zero, Matrix.cons_val_one] at h0 h1
  have key : LinK.eval A h row =
      A.ofLimbs (∑ i, (coords A (row.1 i)).1 * h i + (coords A row.2).1)
        (∑ i, (coords A (row.1 i)).2 * h i + (coords A row.2).2) := by
    rw [← ofLimbs_add, ← ofLimbs_sum, ofLimbs_coords A hA]
    simp only [LinK.eval, LinearMap.coe_mk, AddHom.coe_mk]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← ofLimbs_mul_embed, ofLimbs_coords A hA, mul_comm]
  rw [key, h0, h1, ofLimbs_zero]

/-! ## The masked lane rounds -/

/-- Where a rep's level-0 pads sit (§2): the OOD value's, lane round `j`'s message coordinates
(`u₀.c0, u₀.c1, u₂.c0, u₂.c1`), and the extra lanes' sums `e_i` (coordinates `c0, c1`). Soundness holds for any
placement; hiding is what needs them distinct. -/
structure Pads0 (np e : ℕ) where
  ood : Fin np
  msg : ℕ → Fin 4 → Fin np
  ev : Fin e → Fin 2 → Fin np

/-- A round polynomial whose coefficients are forms. -/
abbrev QuadL (K : Type) (np : ℕ) := LinK K np × LinK K np × LinK K np

/-- `from(msg, s)` (§13.2) on forms. -/
def fromMsgL {np : ℕ} (m : LinK K np × LinK K np) (s : LinK K np) : QuadL K np := (m.1, s + m.2, m.2)

/-- `q(x) = q₀ + q₁·x + q₂·x²` on forms. -/
def QuadL.eval {np : ℕ} (q : QuadL K np) (x : K) : LinK K np := q.1 + x • q.2.1 + x ^ 2 • q.2.2

/-- The coefficientwise value of a form polynomial at the pads. -/
def evalQ (A : Arith F K) {np : ℕ} (h : Fin np → F) (q : QuadL K np) : Quad K :=
  (LinK.eval A h q.1, LinK.eval A h q.2.1, LinK.eval A h q.2.2)

/-- Lane round `j`'s message unmasked, as forms: `(u₀' + pad, u₂' + pad)`. -/
def unmask (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (j : ℕ) (m : K × K) : LinK K np × LinK K np :=
  (cK m.1 + padK A (P.msg j 0) (P.msg j 1), cK m.2 + padK A (P.msg j 2) (P.msg j 3))

/-- Lane round `j`'s message unmasked at the pads `h`. -/
def unmaskAt (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (h : Fin np → F) (j : ℕ) (m : K × K) : K × K :=
  (m.1 + A.ofLimbs (h (P.msg j 0)) (h (P.msg j 1)), m.2 + A.ofLimbs (h (P.msg j 2)) (h (P.msg j 3)))

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem eval_quadL (A : Arith F K) {np : ℕ} (h : Fin np → F) (q : QuadL K np) (x : K) :
    LinK.eval A h (q.eval x) = (evalQ A h q).eval x := by
  simp only [QuadL.eval, map_add, map_smul, smul_eq_mul, evalQ, Quad.eval]
  ring

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem evalQ_fromMsgL (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (h : Fin np → F) (j : ℕ) (m : K × K)
    (s : LinK K np) : evalQ A h (fromMsgL (unmask A P j m) s) = fromMsg (unmaskAt A P h j m) (LinK.eval A h s) := by
  simp [evalQ, fromMsgL, unmask, unmaskAt, fromMsg]

/-- **The masked lane rounds** (`foldRoundsNL` with masked messages): the claim is a form; round `j`'s message is
unmasked by its pads. -/
def foldRoundsZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) :
    ℕ → ℕ → LinK K np → QuadL K np → Game (List K × LinK K np)
  | 0, _, s, _ => pure ([], s)
  | 1, _, _, q => do
      let r ← drawK A
      pure ([r], q.eval r)
  | k + 2, j, _, q => do
      let r ← drawK A
      let s' := q.eval r
      let msg ← recv (K × K)
      let (rs, s'') ← foldRoundsZK A P (k + 1) (j + 1) s' (fromMsgL (unmask A P (j + 1) msg) s')
      pure (r :: rs, s'')

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- **The lane rounds couple**: when the claim and the round polynomial evaluate at `h` to the clear ones, every
masked continuation is bounded by the clear continuation at the claim's value. -/
theorem fold_couple (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (h : Fin np → F) {α β : Type} (p : α → Prop)
    (p' : β → Prop) : ∀ (k j : ℕ) (cl : LinK K np) (qL : QuadL K np) (s : K) (q : Quad K)
    (f : List K × LinK K np → Game α) (f' : List K × K → Game β),
    LinK.eval A h cl = s → evalQ A h qL = q →
    (∀ rs cl', value p (f (rs, cl')) ≤ value p' (f' (rs, LinK.eval A h cl'))) →
    value p ((foldRoundsZK A P k j cl qL).bind f) ≤ value p' ((foldRoundsNL A k s q).bind f')
  | 0, _, cl, _, s, _, f, f', hs, _, hf => by
      simp only [foldRoundsZK, foldRoundsNL, pure_eq, bind_ret]
      rw [← hs]
      exact hf _ _
  | 1, _, _, qL, _, q, f, f', _, hq, hf => by
      simp only [foldRoundsZK, foldRoundsNL, bind_eq, pure_eq, bind_assoc, bind_ret]
      refine value_bind_mono _ fun r => ?_
      rw [← hq, ← eval_quadL]
      exact hf _ _
  | k + 2, j, _, qL, _, q, f, f', _, hq, hf => by
      simp only [foldRoundsZK, foldRoundsNL, bind_eq, pure_eq, bind_assoc, bind_ret]
      refine value_bind_mono _ fun r => ?_
      refine value_recv_le fun msg => ⟨unmaskAt A P h (j + 1) msg, ?_⟩
      refine fold_couple A P h p p' (k + 1) (j + 1) _ _ _ _ _ _ (by rw [eval_quadL, hq]) ?_ ?_
      · rw [evalQ_fromMsgL, eval_quadL, hq]
      · intro rs cl'
        exact hf _ _

/-! ## Level 0 -/

/-- Levels 1+ of `Model.ligeritoPad`, from the restarted claim `s` on: the code-switch message, level 1's commitment,
its OOD samples, level 0's padded queries, and the recursive rounds. -/
def ligTail (A : Arith F K) (sch : Schedule) (t : ℕ) (c₀ : F) (e r : ℕ) (b : ℕ → F) (C₀ : Oracle F)
    (lv₀ : Level) (rest : List Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K) (ρ : Fin e → K)
    (ybar : Fin t → K) (s : K) : Game Bool := do
  let n := sch.m - 7
  let msg' ← recv (K × K)
  let q := fromMsg msg' s
  match rest with
  | [] => pure false
  | lv₁ :: _ => do
      let C₁ ← recv (Oracle F)
      let dim := n - sch.k0 + 1
      let (s, q, oods) ← twoOod A dim 0 s q
      let st : LigState F K := ⟨s, q, dim, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, lv₀⟩
      let st ← queryAndBatchV A lv₀
        (padColumn A lv₀.logLen lv₀.logCols t c₀ initial ρ (extraLane sch.k0 e r) ybar C₀) 0 st
      recursiveRounds A n b initial sch.ext rest { st with prev := C₁, prevLevel := lv₁ }

/-- **C-Flock ZK's masked level 0** for rep `r` (§1, §7): `Model.ligeritoPad` with the OOD value, the lane messages and
`e^r` masked by the pads `P`, the claim carried as a form, and the restart `T'` received after `ρ` and before `ybar`.
Returns the lane-phase row `claim_k + Σ_i ρ_i·e_i − T'` and the verdict of levels 1+, run from `T'`. -/
def ligeritoPadZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (sch : Schedule) (t : ℕ) (c₀ : F) (r : ℕ) (T : F)
    (b : ℕ → F) (C₀ : Oracle F) : Game (LinK K np × Bool) := do
  let n := sch.m - 7
  match sch.levels with
  | [] => pure (0, false)
  | lv₀ :: rest => do
      let z₀ ← draw (Fin n → F)
      let y₀ ← recv F
      let β₀ ← draw F
      let s : LinK K np := cK (A.embed T) + A.embed β₀ • (cK (A.embed y₀) + padF P.ood)
      let msg ← recv (K × K)
      let (initial, s) ← foldRoundsZK A P sch.k0 0 s (fromMsgL (unmask A P 0 msg) s)
      let ev ← recv (Fin e → K)
      let ρ ← drawKs A e
      let T' ← recv K
      let ybar ← recv (Fin t → K)
      let row := s + ∑ i, ρ i • (cK (ev i) + padK A (P.ev i 0) (P.ev i 1)) - cK T'
      let acc ← ligTail A sch t c₀ e r b C₀ lv₀ rest z₀ β₀ initial ρ ybar T'
      pure (row, acc)

omit [DecidableEq F] [Fintype K] in
/-- **Level 0 couples**: at every pad vector `h`, the masked level 0 ends with a true verdict and a lane-phase row
vanishing at `h` with probability at most the clear level 0's acceptance probability. -/
theorem ligeritoPadZK_couple (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (sch : Schedule) (t : ℕ) (c₀ : F)
    (r : ℕ) (T : F) (b : ℕ → F) (C₀ : Oracle F) (h : Fin np → F) :
    value (fun o => o.2 = true ∧ LinK.eval A h o.1 = 0) (ligeritoPadZK A P sch t c₀ r T b C₀) ≤
      value (· = true) (ligeritoPad A sch t c₀ e r T b C₀) := by
  obtain ⟨m, k0, levels, ext⟩ := sch
  cases levels with
  | nil => simp [ligeritoPadZK]
  | cons lv₀ rest =>
    simp only [ligeritoPadZK, ligeritoPad, bind_eq, pure_eq]
    refine value_bind_mono _ fun z₀ => ?_
    refine value_recv_le fun y₀ => ⟨y₀ + h P.ood, ?_⟩
    refine value_bind_mono _ fun β₀ => ?_
    refine value_recv_le fun msg => ⟨unmaskAt A P h 0 msg, ?_⟩
    have hs : LinK.eval A h (cK (A.embed T) + A.embed β₀ • (cK (A.embed y₀) + padF P.ood)) =
        A.embed T + A.embed β₀ * A.embed (y₀ + h P.ood) := by
      simp [map_add, map_smul, smul_eq_mul, mul_add]
    refine fold_couple A P h _ _ k0 0 _ _ _ _ _ _ hs (by rw [evalQ_fromMsgL, hs]) ?_
    intro initial cl
    simp only
    refine value_recv_le fun ev =>
      ⟨fun i => ev i + A.ofLimbs (h (P.ev i 0)) (h (P.ev i 1)), ?_⟩
    refine value_bind_mono _ fun ρ => ?_
    simp only [recv, bind_send, bind_ret, value_send]
    refine iSup_le fun T' => iSup_mono fun ybar => ?_
    rw [value_bind_ret]
    by_cases hrow : LinK.eval A h (cl + ∑ i, ρ i • (cK (ev i) + padK A (P.ev i 0) (P.ev i 1)) - cK T') = 0
    · have hT : T' = LinK.eval A h cl + ∑ i, ρ i * (ev i + A.ofLimbs (h (P.ev i 0)) (h (P.ev i 1))) := by
        simp only [map_sub, map_add, map_sum, map_smul, smul_eq_mul, eval_cK, eval_padK] at hrow
        exact (sub_eq_zero.1 hrow).symm
      rw [value_congr (q := (· = true)) (fun acc => ⟨fun h' => h'.1, fun h' => ⟨h', hrow⟩⟩), hT]
      rfl
    · rw [value_congr (q := fun _ => False) (fun acc => ⟨fun h' => hrow h'.2, False.elim⟩), value_false]
      exact zero_le

end FlockSoundness.Discharge.ZkSession
