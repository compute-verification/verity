import Proofs.Flock.Claims.Complete.Replay.LigeritoSum

/-!
# The honest Ligerito's claim is the verifier's, step by step

The honest prover keeps a K-table `t` and a weight `w` (`LigState.tab`, `LigState.w`); the verifier keeps a claim `s`
and a round polynomial `q`. `Sync s q t w n` says the two agree: `s = Σ t·w` over the table's `2^(n+1)` entries, and
`q` evaluates everywhere to the round polynomial of `(t, w)` in their lowest variable. Every step of the verifier's
model keeps it, on the prover's matching step:

* **a challenge** (`Sync.chal`): `q(r)` is the folded tables' sum (`quad_roundPoly`, from the message `roundPoly`), and
  the next message is the folded tables' round polynomial;
* **a challenge that splits** (`Sync.chalSplit`): splitting the folded table into F lanes and lifting the weight by `u`
  keeps the sum (`sum_split`);
* **a claim** (`Sync.add`): an out-of-domain value or a consistency sum `val = Σ t·v`, scaled by its coin `b`, adds
  `b·v` to the weight and `b·(v`'s round polynomial`)` to `q` (`Sync.ood`: the out-of-domain value is `mleF`);
* **the end** (`lhsE_honest`): the final check's sum over the split table `yr = splitK f` is the claim, when the
  residual weights are the prover's weight plus the last consistency claim's.

The array operations read as the model's on their entries (`gv`): `foldK` is `foldLo` (`gv_foldK`), `liftU` multiplies
the odd entries by `u` (`gv_liftU`), `splitK` is the limbs (`splitK_get`), `addScaled` is entrywise (`gv_addScaled`).
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Lg

/-- A K-table's entries as model field elements. -/
noncomputable def gv (a : Array F256) (y : ℕ) : 𝕂 := gOf a[y]!

/-! ## The array operations on entries -/

theorem foldK_size (r : F256) (a : Array F256) : (foldK r a).size = a.size / 2 := by simp [foldK]

theorem gv_foldK (r : F256) (a : Array F256) {y : ℕ} (hy : y < a.size / 2) :
    gv (foldK r a) y = foldLo (gOf r) (gv a) y := by
  unfold gv foldK foldLo
  rw [getElem!_range_map _ hy, gOf_add, gOf_mul, gOf_add]

/-- Pushing two images of each entry in turn: the accumulator's entries, then `g0 v, g1 v` for each `v`. -/
theorem pairs_spec {α β : Type} [Inhabited α] [Inhabited β] (g0 g1 : α → β) :
    ∀ (l : List α) (acc : Array β),
      (l.foldl (fun acc v => (acc.push (g0 v)).push (g1 v)) acc).size = acc.size + 2 * l.length ∧
      (∀ i < acc.size, (l.foldl (fun acc v => (acc.push (g0 v)).push (g1 v)) acc)[i]! = acc[i]!) ∧
      ∀ x < l.length, (l.foldl (fun acc v => (acc.push (g0 v)).push (g1 v)) acc)[acc.size + 2 * x]! = g0 l[x]! ∧
        (l.foldl (fun acc v => (acc.push (g0 v)).push (g1 v)) acc)[acc.size + 2 * x + 1]! = g1 l[x]!
  | [], acc => ⟨by simp, fun _ _ => rfl, fun x hx => absurd hx (by simp)⟩
  | v :: l, acc => by
      obtain ⟨h1, h2, h3⟩ := pairs_spec g0 g1 l ((acc.push (g0 v)).push (g1 v))
      simp only [List.foldl_cons, Array.size_push, List.length_cons] at h1 h2 h3 ⊢
      refine ⟨by rw [h1]; ring, fun i hi => ?_, fun x hx => ?_⟩
      · rw [h2 i (by omega), getElem!_pos _ i (by simp; omega), getElem!_pos acc i hi]
        simp [Array.getElem_push, hi, show i < acc.size + 1 by omega]
      · rcases x with _ | x
        · simp only [Nat.mul_zero, Nat.add_zero]
          rw [h2 _ (by omega), h2 _ (by omega), getElem!_pos _ _ (by simp), getElem!_pos _ _ (by simp)]
          simp [Array.getElem_push]
        · have := h3 x (by omega)
          rw [show acc.size + 1 + 1 + 2 * x = acc.size + 2 * (x + 1) by ring] at this
          simpa using this

theorem liftU_spec (w : Array F256) :
    (liftU w).size = 2 * w.size ∧ ∀ x < w.size, (liftU w)[2 * x]! = w[x]! ∧ (liftU w)[2 * x + 1]! = F256.u * w[x]! := by
  obtain ⟨h1, -, h3⟩ := pairs_spec id (F256.u * ·) w.toList #[]
  unfold liftU
  rw [← Array.foldl_toList]
  refine ⟨by simpa using h1, fun x hx => ?_⟩
  have := h3 x (by simpa using hx)
  simp only [Array.size_empty, Nat.zero_add, id] at this
  rw [this.1, this.2, getElem!_pos w.toList x (by simpa using hx), getElem!_pos w x hx, Array.getElem_toList]
  exact ⟨rfl, rfl⟩

theorem splitK_spec (f : Array F256) :
    (splitK f).size = 2 * f.size ∧ ∀ x < f.size, (splitK f)[2 * x]! = f[x]!.c0 ∧ (splitK f)[2 * x + 1]! = f[x]!.c1 := by
  obtain ⟨h1, -, h3⟩ := pairs_spec F256.c0 F256.c1 f.toList #[]
  unfold splitK
  rw [← Array.foldl_toList]
  refine ⟨by simpa using h1, fun x hx => ?_⟩
  have := h3 x (by simpa using hx)
  simp only [Array.size_empty, Nat.zero_add] at this
  rw [this.1, this.2, getElem!_pos f.toList x (by simpa using hx), getElem!_pos f x hx, Array.getElem_toList]
  exact ⟨rfl, rfl⟩

theorem gv_liftU (w : Array F256) {y : ℕ} (hy : y < 2 * w.size) :
    gv (liftU w) y = (if y.testBit 0 then FlockLevel3.u else 1) * gv w (y / 2) := by
  obtain ⟨-, h⟩ := liftU_spec w
  rcases Nat.even_or_odd' y with ⟨x, rfl | rfl⟩
  · have hb : (2 * x).testBit 0 = false := by simp [Nat.testBit_zero]
    rw [gv, (h x (by omega)).1, hb, show 2 * x / 2 = x by omega]
    simp [gv]
  · have hb : (2 * x + 1).testBit 0 = true := by simp [Nat.testBit_zero]
    rw [gv, (h x (by omega)).2, hb, show (2 * x + 1) / 2 = x by omega, gOf_mul, gOf_u]
    simp [gv]

theorem embedF_size (g : Array F128) : (embedF g).size = g.size := by simp [embedF]

theorem gv_embedF (g : Array F128) {y : ℕ} (hy : y < g.size) : gv (embedF g) y = FlockLevel3.embed (fOf g[y]!) := by
  rw [gv, embedF, getElem!_pos _ y (by simpa using hy), Array.getElem_map, ← getElem!_pos g y hy, gOf_ofF]

theorem addScaled_size (w v : Array F256) (b : F128) : (addScaled w v b).size = w.size := by simp [addScaled]

theorem gv_addScaled (w v : Array F256) (b : F128) {y : ℕ} (hy : y < w.size) :
    gv (addScaled w v b) y = gv w y + gv v y * FlockLevel3.embed (fOf b) := by
  rw [gv, addScaled, getElem!_range_map _ hy, gOf_add, gOf_smul]
  rfl

/-! ## The round polynomial -/

theorem gOf_roundPoly (t w : Array F256) :
    gOf (roundPoly t w).1 = ∑ k ∈ Finset.range (t.size / 2), gv t (2 * k) * gv w (2 * k) ∧
      gOf (roundPoly t w).2 = ∑ k ∈ Finset.range (t.size / 2),
        (gv t (2 * k) + gv t (2 * k + 1)) * (gv w (2 * k) + gv w (2 * k + 1)) := by
  unfold roundPoly
  refine ⟨?_, ?_⟩
  · rw [gOf_foldl_add, gOf_zero, zero_add, FlockLevel3.list_sum_range]
    exact Finset.sum_congr rfl fun k _ => by rw [gOf_mul]; rfl
  · rw [gOf_foldl_add, gOf_zero, zero_add, FlockLevel3.list_sum_range]
    exact Finset.sum_congr rfl fun k _ => by rw [gOf_mul, gOf_add, gOf_add]; rfl

theorem quad_sum {K : Type} [CommRing K] (h2 : (2 : K) = 0) (N : ℕ) (t0 t1 w0 w1 : ℕ → K) (x : K) :
    (∑ k ∈ Finset.range N, t0 k * w0 k) + ((∑ k ∈ Finset.range N, (t0 k * w0 k + t1 k * w1 k)) +
      ∑ k ∈ Finset.range N, (t0 k + t1 k) * (w0 k + w1 k)) * x +
      (∑ k ∈ Finset.range N, (t0 k + t1 k) * (w0 k + w1 k)) * x ^ 2 =
    ∑ k ∈ Finset.range N, (t0 k + x * (t1 k + t0 k)) * (w0 k + x * (w1 k + w0 k)) := by
  rw [← Finset.sum_add_distrib, Finset.sum_mul, Finset.sum_mul, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  linear_combination (t1 k * w1 k * x) * h2

/-- **The round lemma**: the message `roundPoly t w` with the claim `s = Σ t·w` gives the round polynomial whose value
at every `x` is the sum of the tables folded at `x`. -/
theorem quad_roundPoly (t w : Array F256) (s : F256) (n : ℕ) (ht : t.size = 2 ^ (n + 1))
    (hs : gOf s = ∑ x ∈ Finset.range (2 ^ (n + 1)), gv t x * gv w x) (x : 𝕂) :
    (quadOf (Poly.ofMsg (roundPoly t w) s)).eval x =
      ∑ y ∈ Finset.range (2 ^ n), foldLo x (gv t) y * foldLo x (gv w) y := by
  obtain ⟨h1, h2⟩ := gOf_roundPoly t w
  have hh : t.size / 2 = 2 ^ n := by rw [ht, pow_succ]; omega
  rw [hh] at h1 h2
  rw [sum_range_pairs] at hs
  rw [quadOf_ofMsg]
  simp only [Model.fromMsg, Model.Quad.eval, foldLo]
  rw [h1, h2, hs]
  exact quad_sum CharTwo.two_eq_zero (2 ^ n) (fun k => gv t (2 * k)) (fun k => gv t (2 * k + 1))
    (fun k => gv w (2 * k)) (fun k => gv w (2 * k + 1)) x

/-! ## Splitting -/

theorem sum_pair {K : Type} [CommSemiring K] (N : ℕ) (a b c : ℕ → K) (u : K) :
    ∑ x ∈ Finset.range N, (a x * c x + b x * (u * c x)) = ∑ x ∈ Finset.range N, (a x + b x * u) * c x :=
  Finset.sum_congr rfl fun _ _ => by ring

/-- **A split keeps the sum**: the folded table split into F lanes against the weight lifted by `u`. -/
theorem sum_split (f w : Array F256) (n : ℕ) (hf : f.size = 2 ^ n) (hw : w.size = 2 ^ n) :
    ∑ y ∈ Finset.range (2 ^ (n + 1)), gv (embedF (splitK f)) y * gv (liftU w) y =
      ∑ x ∈ Finset.range (2 ^ n), gv f x * gv w x := by
  obtain ⟨hs, hg⟩ := splitK_spec f
  obtain ⟨-, hl⟩ := liftU_spec w
  rw [sum_range_pairs]
  have e : ∀ x ∈ Finset.range (2 ^ n), gv (embedF (splitK f)) (2 * x) * gv (liftU w) (2 * x) +
      gv (embedF (splitK f)) (2 * x + 1) * gv (liftU w) (2 * x + 1) =
      FlockLevel3.embed (fOf f[x]!.c0) * gv w x + FlockLevel3.embed (fOf f[x]!.c1) * (FlockLevel3.u * gv w x) := by
    intro x hx
    have hx' : x < 2 ^ n := Finset.mem_range.1 hx
    rw [gv_embedF _ (by omega), gv_embedF _ (by omega), (hg x (by omega)).1, (hg x (by omega)).2]
    simp only [gv]
    rw [(hl x (by omega)).1, (hl x (by omega)).2, gOf_mul, gOf_u]
  rw [Finset.sum_congr rfl e, sum_pair]
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [gv]
  rw [gOf_limbs f[x]!]

/-! ## The invariant -/

/-- **The verifier's claim is the honest prover's**: the table and the weight have `2^(n+1)` entries, the claim is
`Σ t·w`, and the round polynomial is `(t, w)`'s in their lowest variable. -/
structure Sync (s : F256) (q : Poly) (t w : Array F256) (n : ℕ) : Prop where
  ts : t.size = 2 ^ (n + 1)
  ws : w.size = 2 ^ (n + 1)
  sum : gOf s = ∑ x ∈ Finset.range (2 ^ (n + 1)), gv t x * gv w x
  quad : ∀ x, (quadOf q).eval x = ∑ y ∈ Finset.range (2 ^ n), foldLo x (gv t) y * foldLo x (gv w) y

/-- A message: the claim `s = Σ t·w` and the round polynomial of `roundPoly t w`. -/
theorem Sync.ofMsg {s : F256} {t w : Array F256} {n : ℕ} (ht : t.size = 2 ^ (n + 1)) (hw : w.size = 2 ^ (n + 1))
    (hs : gOf s = ∑ x ∈ Finset.range (2 ^ (n + 1)), gv t x * gv w x) :
    Sync s (Poly.ofMsg (roundPoly t w) s) t w n :=
  ⟨ht, hw, hs, quad_roundPoly t w s n ht hs⟩

/-- At a challenge `r` the round polynomial's value is the folded tables' sum. -/
theorem Sync.eval {s : F256} {q : Poly} {t w : Array F256} {n : ℕ} (h : Sync s q t w n) (r : F256) :
    gOf (q.eval r) = ∑ y ∈ Finset.range (2 ^ n), gv (foldK r t) y * gv (foldK r w) y := by
  rw [gOf_eval, h.quad]
  refine Finset.sum_congr rfl fun y hy => ?_
  have hy' : y < 2 ^ n := Finset.mem_range.1 hy
  rw [gv_foldK _ _ (by rw [h.ts, pow_succ]; omega), gv_foldK _ _ (by rw [h.ws, pow_succ]; omega)]

/-- **A challenge**: the claim `q(r)` and the folded tables' message. -/
theorem Sync.chal {s : F256} {q : Poly} {t w : Array F256} {n : ℕ} (h : Sync s q t w (n + 1)) (r : F256) :
    Sync (q.eval r) (Poly.ofMsg (roundPoly (foldK r t) (foldK r w)) (q.eval r)) (foldK r t) (foldK r w) n :=
  Sync.ofMsg (by rw [foldK_size, h.ts, pow_succ]; omega) (by rw [foldK_size, h.ws, pow_succ]; omega) (h.eval r)

/-- **A challenge that splits**: the claim `q(r)` and the message of the split table against the lifted weight. -/
theorem Sync.chalSplit {s : F256} {q : Poly} {t w : Array F256} {n : ℕ} (h : Sync s q t w n) (r : F256) :
    Sync (q.eval r) (Poly.ofMsg (roundPoly (embedF (splitK (foldK r t))) (liftU (foldK r w))) (q.eval r))
      (embedF (splitK (foldK r t))) (liftU (foldK r w)) n := by
  have hf : (foldK r t).size = 2 ^ n := by rw [foldK_size, h.ts, pow_succ]; omega
  have hw : (foldK r w).size = 2 ^ n := by rw [foldK_size, h.ws, pow_succ]; omega
  refine Sync.ofMsg (by rw [embedF_size, (splitK_spec _).1, hf, pow_succ]; ring)
    (by rw [(liftU_spec _).1, hw, pow_succ]; ring) ?_
  rw [sum_split _ _ n hf hw, h.eval r]

theorem sum_addScaled {K : Type} [CommSemiring K] (N : ℕ) (a c d : ℕ → K) (e : K) :
    ∑ x ∈ Finset.range N, a x * c x + (∑ x ∈ Finset.range N, a x * d x) * e =
      ∑ x ∈ Finset.range N, a x * (c x + d x * e) := by
  rw [Finset.sum_mul, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun _ _ => by ring

theorem quad_addScaled_eval (q p : Model.Quad 𝕂) (β x : 𝕂) :
    (q.addScaled p β).eval x = q.eval x + β * p.eval x := by
  simp only [Model.Quad.addScaled, Model.Quad.eval]
  ring

theorem foldLo_addScaled (x : 𝕂) (w v : Array F256) (b : F128) {y : ℕ} (hy : 2 * y + 1 < w.size) :
    foldLo x (gv (addScaled w v b)) y = foldLo x (gv w) y + foldLo x (gv v) y * FlockLevel3.embed (fOf b) := by
  simp only [foldLo]
  rw [gv_addScaled _ _ _ (by omega), gv_addScaled _ _ _ hy]
  ring

/-- **A claim** `val = Σ t·v` with coin `b`: the claim gains `b·val`, the weight `b·v`, the round polynomial `b·(v`'s
round polynomial`)`. -/
theorem Sync.add {s : F256} {q : Poly} {t w : Array F256} {n : ℕ} (h : Sync s q t w n) (v : Array F256)
    (val : F256) (hval : gOf val = ∑ x ∈ Finset.range (2 ^ (n + 1)), gv t x * gv v x)
    (b : F128) :
    Sync (s + val.smul b) (q.addScaled (Poly.ofMsg (roundPoly t v) val) b) t (addScaled w v b) n := by
  refine ⟨h.ts, by rw [addScaled_size, h.ws], ?_, fun x => ?_⟩
  · rw [gOf_add, gOf_smul, h.sum, hval, sum_addScaled]
    refine Finset.sum_congr rfl fun y hy => ?_
    rw [gv_addScaled _ _ _ (by rw [h.ws]; exact Finset.mem_range.1 hy)]
  · rw [quadOf_addScaled, quad_addScaled_eval, h.quad, quad_roundPoly t v val n h.ts hval, mul_comm,
      sum_addScaled]
    refine Finset.sum_congr rfl fun y hy => ?_
    have hy' : y < 2 ^ n := Finset.mem_range.1 hy
    rw [foldLo_addScaled _ _ _ _ (by rw [h.ws, pow_succ]; omega)]

/-- **An out-of-domain value**: `mleF g z` is the sum of the F-table `g` against `eq(z, ·)`. -/
theorem gOf_mleF (g z : Array F128) (n : ℕ) (hg : g.size = 2 ^ (n + 1)) (hz : z.size = n + 1) :
    gOf (F256.ofF (mleF g z)) =
      ∑ x ∈ Finset.range (2 ^ (n + 1)), gv (embedF g) x * gv (embedF (eqTable z)) x := by
  have he : (eqTable z).size = 2 ^ (n + 1) := by rw [FlockLevel3.eqTable_size, hz]
  rw [gOf_ofF, mleF]
  show FlockLevel3.embed (FlockLevel3.GF128.of _) = _
  rw [Zc.of_sumF, map_sum, hg]
  refine Finset.sum_congr rfl fun x hx => ?_
  have hx' : x < 2 ^ (n + 1) := Finset.mem_range.1 hx
  rw [gv_embedF _ (by omega), gv_embedF _ (by omega)]
  show FlockLevel3.embed (fOf _) = _
  rw [fOf_mul, map_mul]

/-- **An out-of-domain intro** on the F-table `g` the prover holds, at the point `z`, with coin `b`. -/
theorem Sync.ood {s : F256} {q : Poly} {g : Array F128} {w : Array F256} {n : ℕ} (h : Sync s q (embedF g) w n)
    (z : Array F128) (hz : z.size = n + 1) (b : F128) :
    Sync (s + (F256.ofF (mleF g z)).smul b)
      (q.addScaled (Poly.ofMsg (roundPoly (embedF g) (embedF (eqTable z))) (F256.ofF (mleF g z))) b) (embedF g)
      (addScaled w (embedF (eqTable z)) b) n := by
  have hg : g.size = 2 ^ (n + 1) := by rw [← embedF_size, h.ts]
  exact h.add _ _ (gOf_mleF g z n hg hz) b

/-! ## The end -/

/-- **The final identity, from the prover's side**: with the claim `s = Σ f·w` on the last folded table `f`, the last
consistency claim `val = Σ f·v` with coin `b`, and residual weights `w + b·v`, the final check's sum over `splitK f` is
`s + b·val`. -/
theorem lhsE_honest (f w v res : Array F256) (ext : ℕ) (s val : F256) (b : F128) (hf : f.size = 2 ^ ext)
    (hs : gOf s = ∑ x ∈ Finset.range (2 ^ ext), gv f x * gv w x)
    (hval : gOf val = ∑ x ∈ Finset.range (2 ^ ext), gv f x * gv v x)
    (hres : ∀ y < 2 ^ ext, gv res y = gv w y + gv v y * FlockLevel3.embed (fOf b)) :
    lhsE (splitK f) ext res = s + val.smul b := by
  obtain ⟨-, hg⟩ := splitK_spec f
  refine RecAlgebra.gOf_inj ?_
  rw [gOf_lhsE, gOf_add, gOf_smul, hs, hval, sum_addScaled]
  refine Finset.sum_congr rfl fun y hy => ?_
  have hy' : y < 2 ^ ext := Finset.mem_range.1 hy
  rw [(hg y (by omega)).1, (hg y (by omega)).2, show gOf res[y]! = gv res y from rfl, hres y hy']
  simp only [gv]
  rw [gOf_limbs f[y]!]
  exact mul_comm _ _

end Lg

end FlockComplete.Replay
