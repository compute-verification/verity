import Flock.Zk
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Zerocheck

/-!
# The masked zerocheck's forms evaluate to the clear replay

`Zk.replay` replays the zerocheck on the masked proof `zc'` (each sent value its clear value plus its pad, `Masks`) and
builds affine forms `Lin` in the pads. This file restates its zerocheck terms verbatim (`r1c`, `r1ab`, `computedC`, `fcL`,
`running`, `prodL`, at `s = lay.len`) and evaluates them at the pads `h` (`evalL l h = l.c + ⟨l.a, h⟩`):

* `running_eval`: the running form after `j` residual rounds evaluates to the clear replay's `claim`;
* `c_eval_eval`: `computedC + fc` (the constraint `zerocheck c_eval`) evaluates to `Pc(z) + final_c`;
* `final_eval`: at `h_ab = h_fa·h_fb`, `running + prod` (the constraint `zerocheck final a·b`) evaluates to
  `claim … n + final_a·final_b`.

At any coins, for any clear proof. With `Zerocheck.c_eval` and `Zerocheck.final_ab`, both constraints evaluate to 0 at the
honest prover's pads (`c_eval_zero`, `final_zero`). `evalL_combo` evaluates `verifyInner`'s batched form.
-/

namespace FlockSoundness.Discharge.Composed.ZerocheckMasked

open Flock Flock.Zk FlockLevel3
open FlockSoundness.Discharge.Composed.Zerocheck

local notation "𝔽" => FlockLevel3.GF128

/-! ## The replay's zerocheck terms, at `s = lay.len` -/

/-- The value of the affine form `l` at the pads `h`: `c + Σ a_i h_i`. -/
def evalL (l : Lin) (h : Array F128) : F128 := l.c + dot l.a h

def r1c (lay : PadLayout) (zc : ZerocheckProof) : Array Lin :=
  (Array.range 64).map fun i => Lin.masked zc.round1c[i]! (lay.r1c i) lay.len

def r1ab (lay : PadLayout) (zc : ZerocheckProof) : Array Lin :=
  (Array.range 64).map fun i => Lin.masked zc.round1ab[i]! (lay.r1ab i) lay.len

def computedC (lay : PadLayout) (zc : ZerocheckProof) (z : F128) : Lin :=
  Lin.combo (lagrange nodesL den6Inv z) (r1c lay zc) lay.len

def fcL (lay : PadLayout) (zc : ZerocheckProof) : Lin := Lin.masked zc.finalC lay.fc lay.len

/-- The running form before the residual rounds. -/
def running0 (lay : PadLayout) (zc : ZerocheckProof) (z : F128) : Lin :=
  (Lin.combo (combWeights z) ((Array.range 64).map fun i => (r1ab lay zc)[i]!.add (r1c lay zc)[i]!) lay.len).add
    (computedC lay zc z)

/-- Residual round `j`'s update of the running form, at `r_eq = r[j]` and the round's coin `rho[j]`. -/
def roundL (lay : PadLayout) (zc : ZerocheckProof) (r rho : Array F128) (j : ℕ) (running : Lin) : Lin :=
  let (m1, minf) := zc.rounds[j]!
  let req := r[j]!
  let g1 := Lin.masked m1 (lay.ml j 0) lay.len
  let ginf := Lin.masked minf (lay.ml j 1) lay.len
  let g0 := (running.add (g1.scale req)).scale (1 + req).inv
  let rr := rho[j]!
  ((g0.scale (1 + rr)).add (g1.scale rr)).add (ginf.scale (rr * (1 + rr)))

/-- The running form after `j` residual rounds. -/
def running (lay : PadLayout) (zc : ZerocheckProof) (r rho : Array F128) (z : F128) : ℕ → Lin
  | 0 => running0 lay zc z
  | j + 1 => roundL lay zc r rho j (running lay zc r rho z j)

/-- `a·b = a'b' + a'·h_b + b'·h_a + h_ab` on the masked `a' = final_a`, `b' = final_b`. -/
def prodL (lay : PadLayout) (zc : ZerocheckProof) : Lin :=
  ⟨zc.finalA * zc.finalB,
    (((Array.replicate lay.len 0).set! lay.fb zc.finalA).set! lay.fa zc.finalB).set! lay.hab 1⟩

/-- `zc'` is the clear `zc` masked by the pads `h` at `lay`'s indices, on its first `n` residual rounds. -/
structure Masks (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ) : Prop where
  round1ab : ∀ i < 64, zc'.round1ab[i]! = zc.round1ab[i]! + h[lay.r1ab i]!
  round1c : ∀ i < 64, zc'.round1c[i]! = zc.round1c[i]! + h[lay.r1c i]!
  rounds : ∀ j < n, zc'.rounds[j]! = (zc.rounds[j]!.1 + h[lay.ml j 0]!, zc.rounds[j]!.2 + h[lay.ml j 1]!)
  finalA : zc'.finalA = zc.finalA + h[lay.fa]!
  finalB : zc'.finalB = zc.finalB + h[lay.fb]!
  finalC : zc'.finalC = zc.finalC + h[lay.fc]!

/-! ## Arrays and the field -/

theorem add_add_self (v x : F128) : v + x + x = v := by
  apply eq_of_of
  rw [of_add, of_add, add_assoc, CharTwo.add_self_eq_zero, add_zero]

theorem add_self' (v : F128) : v + v = 0 := by
  apply eq_of_of
  rw [of_add, CharTwo.add_self_eq_zero, GF128.zero_eq]

theorem of_dot_range (a b : Array F128) (n : ℕ) (ha : a.size = n) :
    GF128.of (dot a b) = ∑ i ∈ Finset.range n, GF128.of a[i]! * GF128.of b[i]! := by
  rw [of_dot a b n ha, Fin.sum_univ_eq_sum_range (fun i => GF128.of a[i]! * GF128.of b[i]!) n]

theorem dot_congr (a b b' : Array F128) (n : ℕ) (ha : a.size = n) (hb : ∀ i < n, b[i]! = b'[i]!) :
    dot a b = dot a b' := by
  apply eq_of_of
  rw [of_dot_range a b n ha, of_dot_range a b' n ha]
  exact Finset.sum_congr rfl fun i hi => by rw [hb i (Finset.mem_range.1 hi)]

theorem get_set (a : Array F128) (i k : ℕ) (v : F128) (hk : k < a.size) :
    (a.set! i v)[k]! = if k = i then v else a[k]! := by
  rw [getElem!_pos _ k (by simpa using hk), getElem!_pos a k hk]
  simp only [Array.set!]
  rw [Array.getElem_setIfInBounds hk]
  by_cases h : k = i
  · simp [h]
  · simp [h, Ne.symm h]

theorem get_replicate (s k : ℕ) (hk : k < s) : (Array.replicate s (0 : F128))[k]! = 0 := by
  rw [getElem!_pos _ k (by simpa using hk)]
  simp

theorem get_map' {α : Type} [Inhabited α] (f : α → F128) (xs : Array α) (i : ℕ) (hi : i < xs.size) :
    (xs.map f)[i]! = f xs[i]! := by
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, ← getElem!_pos xs i hi]

theorem get_map_lin (f : ℕ → Lin) {n i : ℕ} (hi : i < n) : ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using hi)]
  simp

theorem mem_map_range {f : ℕ → Lin} {n : ℕ} {l : Lin} (hl : l ∈ (Array.range n).map f) : ∃ i < n, f i = l := by
  simpa using hl

/-! ## The forms' values -/

theorem of_evalL_range (l : Lin) (h : Array F128) (s : ℕ) (hl : l.a.size = s) :
    GF128.of (evalL l h) = GF128.of l.c + ∑ k ∈ Finset.range s, GF128.of l.a[k]! * GF128.of h[k]! := by
  rw [evalL, of_add, of_dot_range _ _ s hl]

@[simp] theorem size_masked (v : F128) (i s : ℕ) : (Lin.masked v i s).a.size = s := by simp [Lin.masked]

@[simp] theorem size_add (x y : Lin) : (x.add y).a.size = min x.a.size y.a.size := by simp [Lin.add]

@[simp] theorem size_scale (x : Lin) (k : F128) : (x.scale k).a.size = x.a.size := by simp [Lin.scale]

theorem evalL_masked (v : F128) (i s : ℕ) (hi : i < s) (h : Array F128) :
    evalL (Lin.masked v i s) h = v + h[i]! := by
  apply eq_of_of
  rw [of_evalL_range _ h s (size_masked v i s), of_add]
  congr 1
  rw [Finset.sum_eq_single i]
  · simp only [Lin.masked]
    rw [get_set _ _ _ _ (by simpa using hi), ite_eq_left rfl, of_one', one_mul]
  · intro k hk hki
    have hk' := Finset.mem_range.1 hk
    simp only [Lin.masked]
    rw [get_set _ _ _ _ (by simpa using hk'), ite_eq_right hki, get_replicate _ _ hk', ← GF128.zero_eq, zero_mul]
  · intro hi'
    exact absurd (Finset.mem_range.2 hi) hi'

theorem evalL_add (x y : Lin) (h : Array F128) (hxy : x.a.size = y.a.size) :
    evalL (x.add y) h = evalL x h + evalL y h := by
  apply eq_of_of
  have hs : (x.add y).a.size = x.a.size := by simp [hxy]
  rw [of_evalL_range _ h _ hs, of_add, of_evalL_range x h _ rfl, of_evalL_range y h _ hxy.symm]
  have hent : ∀ k ∈ Finset.range x.a.size, GF128.of (x.add y).a[k]! * GF128.of h[k]! =
      GF128.of x.a[k]! * GF128.of h[k]! + GF128.of y.a[k]! * GF128.of h[k]! := fun k hk => by
    have hk' := Finset.mem_range.1 hk
    simp only [Lin.add]
    rw [← hxy, min_self, getElem!_pos _ k (by simpa using hk'), Array.getElem_map, Array.getElem_range, of_add,
      add_mul]
  rw [Finset.sum_congr rfl hent, Finset.sum_add_distrib]
  show GF128.of (x.c + y.c) + _ = _
  rw [of_add]
  ring

theorem evalL_scale (x : Lin) (k : F128) (h : Array F128) : evalL (x.scale k) h = evalL x h * k := by
  apply eq_of_of
  rw [of_evalL_range _ h x.a.size (size_scale x k), of_mul, of_evalL_range x h _ rfl]
  have hent : ∀ i ∈ Finset.range x.a.size, GF128.of (x.scale k).a[i]! * GF128.of h[i]! =
      GF128.of x.a[i]! * GF128.of h[i]! * GF128.of k := fun i hi => by
    have hi' := Finset.mem_range.1 hi
    simp only [Lin.scale]
    rw [getElem!_pos _ i (by simpa using hi'), Array.getElem_map, ← getElem!_pos x.a i hi', of_mul]
    ring
  rw [Finset.sum_congr rfl hent, ← sum_mul']
  show GF128.of (x.c * k) + _ = _
  rw [of_mul, add_mul]

/-- `Lin.combo`'s inner update `a_k ← a_k + w·l_k`. -/
def innerStep (w : F128) (la b : Array F128) (k : ℕ) : Array F128 := b.set! k (b[k]! + w * la[k]!)

/-- `Lin.combo`'s outer update by `w_i·l_i`. -/
def outerStep (ws : Array F128) (ls : Array Lin) (s : ℕ) (st : F128 × Array F128) (i : ℕ) : F128 × Array F128 :=
  (st.1 + ws[i]! * ls[i]!.c, (List.range (min s ls[i]!.a.size)).foldl (innerStep ws[i]! ls[i]!.a) st.2)

theorem combo_eq (ws : Array F128) (ls : Array Lin) (s : ℕ) :
    Lin.combo ws ls s =
      ⟨((List.range (min ws.size ls.size)).foldl (outerStep ws ls s) (0, Array.replicate s 0)).1,
        ((List.range (min ws.size ls.size)).foldl (outerStep ws ls s) (0, Array.replicate s 0)).2⟩ := by
  unfold Lin.combo outerStep innerStep
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind]
  simp [Std.Legacy.Range.size, List.range_eq_range']
  rfl

theorem size_inner (w : F128) (la : Array F128) :
    ∀ (L : List ℕ) (b : Array F128), (L.foldl (innerStep w la) b).size = b.size
  | [], _ => rfl
  | k :: L, b => by
      rw [List.foldl_cons, size_inner w la L]
      simp [innerStep]

theorem size_outer (ws : Array F128) (ls : Array Lin) (s : ℕ) :
    ∀ (L : List ℕ) (st : F128 × Array F128), st.2.size = s → (L.foldl (outerStep ws ls s) st).2.size = s
  | [], _, h => h
  | i :: L, st, h => by
      rw [List.foldl_cons]
      exact size_outer ws ls s L _ (by simp only [outerStep]; rw [size_inner]; exact h)

@[simp] theorem size_combo (ws : Array F128) (ls : Array Lin) (s : ℕ) : (Lin.combo ws ls s).a.size = s := by
  rw [combo_eq]
  exact size_outer ws ls s _ _ (by simp)

theorem inner_fold (w : F128) (la a : Array F128) (s : ℕ) (ha : a.size = s) :
    ∀ n ≤ s, ∀ k < s, ((List.range n).foldl (innerStep w la) a)[k]! = if k < n then a[k]! + w * la[k]! else a[k]!
  | 0, _ => fun k _ => by simp
  | n + 1, hn => fun k hk => by
      have ih := inner_fold w la a s ha n (by omega)
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      simp only [innerStep]
      rw [get_set _ _ _ _ (by rw [size_inner, ha]; exact hk), ih k hk, ih n (by omega), ite_eq_right (Nat.lt_irrefl n)]
      by_cases hkn : k = n
      · rw [ite_eq_left hkn, hkn, ite_eq_left (Nat.lt_succ_self n)]
      · rw [ite_eq_right hkn]
        by_cases hk' : k < n
        · rw [ite_eq_left hk', ite_eq_left (by omega)]
        · rw [ite_eq_right hk', ite_eq_right (by omega)]

theorem outer_fold (ws : Array F128) (ls : Array Lin) (s : ℕ) (h : Array F128) :
    ∀ n, (∀ i < n, (ls[i]!).a.size = s) →
      GF128.of ((List.range n).foldl (outerStep ws ls s) (0, Array.replicate s 0)).1 +
        ∑ k ∈ Finset.range s,
          GF128.of ((List.range n).foldl (outerStep ws ls s) (0, Array.replicate s 0)).2[k]! * GF128.of h[k]! =
      ∑ i ∈ Finset.range n, GF128.of ws[i]! * GF128.of (evalL ls[i]! h)
  | 0, _ => by
      simp only [List.range_zero, List.foldl_nil, Finset.range_zero, Finset.sum_empty]
      rw [← GF128.zero_eq, zero_add]
      refine Finset.sum_eq_zero fun k hk => ?_
      rw [get_replicate _ _ (Finset.mem_range.1 hk), ← GF128.zero_eq, zero_mul]
  | n + 1, hl => by
      have ih := outer_fold ws ls s h n fun i hi => hl i (by omega)
      have hn : (ls[n]!).a.size = s := hl n (by omega)
      have h1 := size_outer ws ls s (List.range n) (0, Array.replicate s 0) (by simp)
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      generalize hst : (List.range n).foldl (outerStep ws ls s) (0, Array.replicate s 0) = st at ih h1
      simp only [outerStep, hn, min_self]
      have hent : ∀ k ∈ Finset.range s,
          GF128.of ((List.range s).foldl (innerStep ws[n]! ls[n]!.a) st.2)[k]! * GF128.of h[k]! =
            GF128.of st.2[k]! * GF128.of h[k]! + GF128.of ws[n]! * (GF128.of ls[n]!.a[k]! * GF128.of h[k]!) :=
        fun k hk => by
          have hk' := Finset.mem_range.1 hk
          rw [inner_fold _ _ _ s h1 s le_rfl k hk', ite_eq_left hk', of_add, of_mul]
          ring
      rw [Finset.sum_congr rfl hent, Finset.sum_add_distrib, ← mul_sum', Finset.sum_range_succ, ← ih,
        of_evalL_range _ h s hn, of_add, of_mul]
      ring

/-- **`Lin.combo` evaluates to the weighted sum of its forms' values**, all on `s` pads. -/
theorem evalL_combo (ws : Array F128) (ls : Array Lin) (s : ℕ) (h : Array F128)
    (hl : ∀ l ∈ ls, l.a.size = s) :
    evalL (Lin.combo ws ls s) h = dot ws (ls.map fun l => evalL l h) := by
  have hl' : ∀ i < min ws.size ls.size, (ls[i]!).a.size = s := fun i hi => by
    rw [getElem!_pos ls i (by omega)]
    exact hl _ (Array.getElem_mem _)
  apply eq_of_of
  rw [of_evalL_range _ h s (size_combo ws ls s), combo_eq]
  dsimp only
  rw [outer_fold ws ls s h _ hl', dot_eq_foldl, of_foldl_dot, ← GF128.zero_eq, zero_add, Array.size_map]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [get_map' _ ls i (by have := Finset.mem_range.1 hi; omega)]

/-! ## The layout's indices -/

theorem fa_lt_fb (lay : PadLayout) : lay.fa < lay.fb := by simp only [PadLayout.fb]; omega

theorem fb_lt_hab (lay : PadLayout) : lay.fb < lay.hab := by
  simp only [PadLayout.fb, PadLayout.hab, PadLayout.zp, PadLayout.lc, PadLayout.fc]; omega

theorem fc_lt_len (lay : PadLayout) : lay.fc < lay.len := by
  simp only [PadLayout.fc, PadLayout.len, PadLayout.tw, PadLayout.hab, PadLayout.zp, PadLayout.lc]; omega

theorem hab_lt_len (lay : PadLayout) : lay.hab < lay.len := by
  simp only [PadLayout.len, PadLayout.tw]; omega

theorem fa_lt_len (lay : PadLayout) : lay.fa < lay.len := by
  have := fa_lt_fb lay; have := fb_lt_hab lay; have := hab_lt_len lay; omega

theorem fb_lt_len (lay : PadLayout) : lay.fb < lay.len := by
  have := fb_lt_hab lay; have := hab_lt_len lay; omega

theorem r1_lt_len (lay : PadLayout) (i : ℕ) (hi : i < 128) : i < lay.len := by
  have := fa_lt_len lay
  simp only [PadLayout.fa] at this
  omega

theorem ml_lt_len (lay : PadLayout) (j t : ℕ) (hj : j < lay.nMlv) (ht : t < 2) : lay.ml j t < lay.len := by
  have := fa_lt_len lay
  simp only [PadLayout.fa] at this
  simp only [PadLayout.ml]
  omega

/-! ## The replay's terms' values -/

theorem size_r1c (lay : PadLayout) (zc : ZerocheckProof) (i : ℕ) (hi : i < 64) : (r1c lay zc)[i]!.a.size = lay.len := by
  simp only [r1c]; rw [get_map_lin _ hi]; simp

theorem size_r1ab (lay : PadLayout) (zc : ZerocheckProof) (i : ℕ) (hi : i < 64) :
    (r1ab lay zc)[i]!.a.size = lay.len := by
  simp only [r1ab]; rw [get_map_lin _ hi]; simp

theorem evalL_r1c (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ) (hm : Masks lay h zc zc' n)
    (i : ℕ) (hi : i < 64) : evalL (r1c lay zc')[i]! h = zc.round1c[i]! := by
  simp only [r1c]
  rw [get_map_lin _ hi, evalL_masked _ _ _ (r1_lt_len lay _ (by simp only [PadLayout.r1c]; omega)), hm.round1c i hi,
    add_add_self]

theorem evalL_r1ab (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ) (hm : Masks lay h zc zc' n)
    (i : ℕ) (hi : i < 64) : evalL (r1ab lay zc')[i]! h = zc.round1ab[i]! := by
  simp only [r1ab]
  rw [get_map_lin _ hi, evalL_masked _ _ _ (r1_lt_len lay _ (by simp only [PadLayout.r1ab]; omega)),
    hm.round1ab i hi, add_add_self]

theorem evalL_computedC (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ)
    (hm : Masks lay h zc zc' n) (z : F128) :
    evalL (computedC lay zc' z) h = dot (lagrange nodesL den6Inv z) zc.round1c := by
  rw [computedC, evalL_combo _ _ lay.len h fun l hl => ?_]
  · refine dot_congr _ _ _ 64 (by rw [lagrange_size, nodesL_size]) fun i hi => ?_
    rw [get_map' _ _ i (by simp [r1c]; exact hi), evalL_r1c lay h zc zc' n hm i hi]
  · obtain ⟨i, hi, rfl⟩ := mem_map_range hl
    simp

theorem size_computedC (lay : PadLayout) (zc : ZerocheckProof) (z : F128) :
    (computedC lay zc z).a.size = lay.len := size_combo _ _ _

theorem size_running0 (lay : PadLayout) (zc : ZerocheckProof) (z : F128) : (running0 lay zc z).a.size = lay.len := by
  simp [running0, size_computedC]

theorem size_running (lay : PadLayout) (zc : ZerocheckProof) (r rho : Array F128) (z : F128) :
    ∀ j, (running lay zc r rho z j).a.size = lay.len
  | 0 => size_running0 lay zc z
  | j + 1 => by
      have ih := size_running lay zc r rho z j
      simp only [running, roundL]
      rcases zc.rounds[j]! with ⟨m1, minf⟩
      simp [ih]

theorem evalL_prodL (lay : PadLayout) (zc : ZerocheckProof) (h : Array F128) :
    evalL (prodL lay zc) h = zc.finalA * zc.finalB + zc.finalA * h[lay.fb]! + zc.finalB * h[lay.fa]! + h[lay.hab]! := by
  have i1 := fa_lt_fb lay
  have i2 := fb_lt_hab lay
  have i3 := hab_lt_len lay
  apply eq_of_of
  rw [of_evalL_range _ h lay.len (by simp [prodL])]
  have hent : ∀ k ∈ Finset.range lay.len, GF128.of (prodL lay zc).a[k]! * GF128.of h[k]! =
      ((if lay.fb = k then GF128.of zc.finalA * GF128.of h[k]! else 0) +
        (if lay.fa = k then GF128.of zc.finalB * GF128.of h[k]! else 0)) +
        (if lay.hab = k then GF128.of h[k]! else 0) := fun k hk => by
    have hk' := Finset.mem_range.1 hk
    simp only [prodL]
    rw [get_set _ _ _ _ (by simp; omega), get_set _ _ _ _ (by simp; omega), get_set _ _ _ _ (by simp; omega),
      get_replicate _ _ hk']
    have n1 : lay.fb ≠ lay.hab := by omega
    have n2 : lay.fa ≠ lay.hab := by omega
    have n3 : lay.fa ≠ lay.fb := by omega
    by_cases a1 : k = lay.hab
    · rw [a1]
      simp [n1, n2, of_one']
    · by_cases a2 : k = lay.fa
      · rw [a2]
        simp [n2, Ne.symm n2, Ne.symm n3]
      · by_cases a3 : k = lay.fb
        · rw [a3]
          simp [n1, n3, Ne.symm n1, Ne.symm n3]
        · simp [a1, a2, a3, Ne.symm a1, Ne.symm a2, Ne.symm a3, ← GF128.zero_eq]
  rw [Finset.sum_congr rfl hent, Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.sum_ite_eq,
    Finset.sum_ite_eq, ite_eq_left (Finset.mem_range.2 (by omega)), ite_eq_left (Finset.mem_range.2 (by omega)),
    ite_eq_left (Finset.mem_range.2 i3)]
  show GF128.of (zc.finalA * zc.finalB) + _ = _
  simp only [of_add, of_mul]
  ring

/-! ## The three facts -/

/-- **The running form evaluates to the clear claim**, after `j ≤ n ≤ nMlv` residual rounds. -/
theorem running_eval (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ)
    (hn : n ≤ lay.nMlv) (hm : Masks lay h zc zc' n) (r rho : Array F128) (z : F128) (j : ℕ) (hj : j ≤ n) :
    evalL (running lay zc' r rho z j) h = claim zc r rho z j := by
  induction j with
  | zero =>
      rw [running, running0, claim, claim0, evalL_add _ _ h (by rw [size_combo, size_computedC]),
        evalL_computedC lay h zc zc' n hm z, evalL_combo _ _ lay.len h fun l hl => ?_]
      · congr 1
        refine dot_congr _ _ _ 64 (combWeights_size z) fun i hi => ?_
        rw [get_map' _ _ i (by simpa using hi), get_map_lin _ hi, get_map_range _ hi,
          evalL_add _ _ h (by rw [size_r1ab lay zc' i hi, size_r1c lay zc' i hi]), evalL_r1ab lay h zc zc' n hm i hi,
          evalL_r1c lay h zc zc' n hm i hi]
      · obtain ⟨i, hi, rfl⟩ := mem_map_range hl
        rw [size_add, size_r1ab lay zc' i hi, size_r1c lay zc' i hi, min_self]
  | succ j ih =>
      have hjn : j < n := by omega
      have h0 := ml_lt_len lay j 0 (by omega) (by omega)
      have h1 := ml_lt_len lay j 1 (by omega) (by omega)
      have hsz := size_running lay zc' r rho z j
      rw [running, claim, roundL, hm.rounds j hjn]
      dsimp only
      rw [evalL_add _ _ h (by simp [hsz]), evalL_add _ _ h (by simp [hsz])]
      simp only [evalL_scale]
      rw [evalL_add _ _ h (by simp [hsz])]
      simp only [evalL_scale]
      rw [ih (by omega), evalL_masked _ _ _ h0, evalL_masked _ _ _ h1, add_add_self, add_add_self]
      rfl

/-- **`zerocheck c_eval` evaluates to its two clear sides**: `Pc(z) + final_c`. -/
theorem c_eval_eval (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ)
    (hm : Masks lay h zc zc' n) (z : F128) :
    evalL ((computedC lay zc' z).add (fcL lay zc')) h = dot (lagrange nodesL den6Inv z) zc.round1c + zc.finalC := by
  rw [evalL_add _ _ h (by simp [size_computedC, fcL]), evalL_computedC lay h zc zc' n hm z, fcL,
    evalL_masked _ _ _ (fc_lt_len lay), hm.finalC, add_add_self]

/-- **`zerocheck final a·b` evaluates to its two clear sides** at `h_ab = h_fa·h_fb`: `claim … n + final_a·final_b`. -/
theorem final_eval (lay : PadLayout) (h : Array F128) (zc zc' : ZerocheckProof) (n : ℕ)
    (hn : n ≤ lay.nMlv) (hm : Masks lay h zc zc' n) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!)
    (r rho : Array F128) (z : F128) :
    evalL ((running lay zc' r rho z n).add (prodL lay zc')) h = claim zc r rho z n + zc.finalA * zc.finalB := by
  rw [evalL_add _ _ h (by rw [size_running]; simp [prodL]), running_eval lay h zc zc' n hn hm r rho z n le_rfl,
    evalL_prodL, hm.finalA, hm.finalB, hab]
  apply eq_of_of
  simp only [of_add, of_mul]
  linear_combination (GF128.of zc.finalA * GF128.of h[lay.fb]! + GF128.of zc.finalB * GF128.of h[lay.fa]! +
    2 * GF128.of h[lay.fa]! * GF128.of h[lay.fb]!) * two_eq_zero'

/-! ## At the honest prover -/

/-- `zerocheck c_eval` vanishes at the honest prover's pads, at every coin. -/
theorem c_eval_zero (lay : PadLayout) (h : Array F128) (n : ℕ) (a b c : ℕ → 𝔽)
    (r rho : Array F128) (z : F128) (zc' : ZerocheckProof) (hm : Masks lay h (honest n a b c r rho z) zc' n) :
    evalL ((computedC lay zc' z).add (fcL lay zc')) h = 0 := by
  rw [c_eval_eval lay h _ zc' n hm z, Zerocheck.c_eval, add_self']

/-- `zerocheck final a·b` vanishes at the honest prover's pads with `h_ab = h_fa·h_fb`, for valid tables, at every coin with
`r_k ≠ 1` for `k < n`. -/
theorem final_zero (lay : PadLayout) (h : Array F128) (n : ℕ) (hn : n ≤ lay.nMlv)
    (a b c : ℕ → 𝔽) (hvalid : ∀ i < 2 ^ (n + 6), a i * b i = c i) (r rho : Array F128) (hr : ∀ k < n, r[k]! ≠ 1)
    (z : F128) (zc' : ZerocheckProof) (hm : Masks lay h (honest n a b c r rho z) zc' n)
    (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) :
    evalL ((running lay zc' r rho z n).add (prodL lay zc')) h = 0 := by
  rw [final_eval lay h _ zc' n hn hm hab r rho z, final_ab n a b c r rho z hvalid hr, add_self']

/-- A form's value as a sum over its `s` pads in the field (`InnerHolds`'s shape). -/
theorem of_evalL (l : Lin) (h : Array F128) (s : ℕ) (hl : l.a.size = s) :
    GF128.of (evalL l h) = GF128.of l.c + ∑ k : Fin s, GF128.of l.a[k.val]! * GF128.of h[k.val]! := by
  rw [evalL, of_add, of_dot _ _ s hl]

end FlockSoundness.Discharge.Composed.ZerocheckMasked
