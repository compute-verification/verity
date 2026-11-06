import Proofs.Flock.VBridge.Karatsuba
import Proofs.Flock.Soundness.Discharge.Hm.Word

/-!
# V*'s residuals: sums of GF(2^128) products, summed leaf by leaf

`residualForms S W V` is `verity_flock.rec_residuals.residual_forms`, the gadget that `InnerRepCheck`'s unit
(`rec_residuals.unit_circuit`) and `RecOpen`'s running sums (`rec_open._Forms.residuals`) lay out, as the verifier's
builder (`Flock.HmNets`) states it, operation for operation. `S` is `GfResiduals`' S: each residual is a sum of added
terms and of products `ops[a]·ops[b]`, each operand a sum of terms, a term an element of the port `w` or `v` or a
transposed block of `w`. The products are not multiplied one by one: for each of Karatsuba's 3^7 = 2,187 leaves, the
leaf's AND of every product is XORed into one sum, each sum is committed, and the sums are recombined and reduced once.

`sound_residualForms`: whenever the rows hold and the port forms carry the bits of `w` and `v` (element `i` at bit
`128 i`), residual `r`'s 128 output forms carry the bits of its value in `Flock.F128`, its added terms' sum plus
`Σ_p ops[a_p]·ops[b_p]` (`resVal`, `gf2k.residuals`). The step beyond `sound_mul128` is `polyOf_recomb`: Karatsuba's
recombination of the leaf sums is the sum of the products, `Σ_p K(a_p, b_p) = recombine(Σ_p leaves_p)`; the reduction
is linear (`polyOf_reduceG`).

The Python cited is `backends/flock/python/verity_flock/rec_residuals.py` (blob `6984e566`) and
`catalog/verity_catalog/definitions/boolean/gf2k.py` (blob `52383dd2`); `C.AND` is `forms.Circuit.AND` without CSE
(`Flock.HmNets.and`) and `commit` is
`sha512.commit` (`Flock.HmNets.commitW`).
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock (F128)
open FlockSoundness.Discharge.Hm Polynomial
open FlockLevel3 (P b2 b2_true b2_false b2_xor b2_and w128 w128_lt coeff_natPoly natPoly_ext two_eq_zero_P)

/-! ## `GfResiduals`' S -/

/-- A term of `GfResiduals`' S (`gf2k`'s module doc): `["w", i]` or `["v", i]`, element `i` of a port, or `["wt", i, u]`,
the element whose bit `j` is bit `u` of `w[i + j]`. -/
inductive Term where
  | w (i : ℕ)
  | v (i : ℕ)
  | wt (i u : ℕ)

/-- A residual of S, `{"mul": [[a, b], ...], "add": [term, ...]}`: the sum of `ops[a]·ops[b]` over its products and of
its added terms. -/
structure Res where
  mul : List (ℕ × ℕ)
  add : List Term

/-- `GfResiduals`' S: the operands (each a list of terms, their sum) and the residuals. Its `nw` and `nv` are the ports'
lengths, which `residual_forms` does not read. -/
structure GfS where
  ops : List (List Term)
  res : List Res

/-! ## The builder -/

/-- `residual_forms.term` (`rec_residuals.py:298-304`): a term's 128 forms. `["w", i]` is `W[128 i : 128 i + 128]`,
`["v", i]` the same in `V`, and `["wt", i, u]`'s form `j` is `W[128 (i + j) + u]`. Past a port's end a form is zero, where
Python's slice would come up short and its index would raise. -/
def termForms (W V : List Lin) : Term → List Lin
  | .w i => (List.range 128).map fun j => W.getD (128 * i + j) Lin.zero
  | .v i => (List.range 128).map fun j => V.getD (128 * i + j) Lin.zero
  | .wt i u => (List.range 128).map fun j => W.getD (128 * (i + j) + u) Lin.zero

/-- Two lists of forms XORed position by position, as Python's `zip` pairs them (to the shorter's length). -/
def xorL (a b : List Lin) : List Lin := List.zipWith (· ^^^ ·) a b

/-- `residual_forms.xor` (`rec_residuals.py:306-310`): from 128 zero forms, each list XORed in. -/
def xorAll (xs : List (List Lin)) : List Lin := xs.foldl xorL (List.replicate 128 Lin.zero)

/-- `residual_forms.op` (`rec_residuals.py:312-317`): operand `k`, the XOR of its terms' forms. Python caches it; it adds
no row. Past the end of `ops` it has no terms, where Python's index would raise. -/
def opForms (S : GfS) (W V : List Lin) (k : ℕ) : List Lin := xorAll ((S.ops.getD k []).map (termForms W V))

/-- `_leaves` (`rec_residuals.py:266-269`): the operands of `gf2k.karatsuba`'s ANDs (`gf2k.py:159-167`) on `a` and `b` of
`2^n` forms each, in its order: the low halves' leaves, the high halves', then the halves' sums'. -/
def leaves : ℕ → List Lin → List Lin → List (Lin × Lin)
  | 0, a, b => [(a.headD Lin.zero, b.headD Lin.zero)]
  | n + 1, a, b =>
    leaves n (a.take (2 ^ n)) (b.take (2 ^ n)) ++ leaves n (a.drop (2 ^ n)) (b.drop (2 ^ n)) ++
      leaves n (addL (a.take (2 ^ n)) (a.drop (2 ^ n))) (addL (b.take (2 ^ n)) (b.drop (2 ^ n)))

/-- `gf2k.karatsuba`'s XORs (`gf2k.py:168-173`, `kara`'s) over its ANDs' results `l` (`3^n` of them, in `leaves`' order):
the product's `2·2^n − 1` coefficients. `_recombine` (`rec_residuals.py:256-263`) runs them and `gf2k.reduce_forms` on
the leaves as variables, so its `rec[k]` is the support of `reduceG (recomb 7 l)`'s form `k` in `l`. -/
def recomb : ℕ → List Lin → List Lin
  | 0, l => [l.headD Lin.zero]
  | n + 1, l =>
    let p0 := recomb n (l.take (3 ^ n))
    let p2 := recomb n ((l.drop (3 ^ n)).take (3 ^ n))
    let p1 := recomb n (l.drop (2 * 3 ^ n))
    addL (addL p0 (shiftL (2 ^ n) (addL (addL p1 p0) p2))) (shiftL (2 ^ n * 2) p2)

/-- The leaf sums (`rec_residuals.py:324-329`): for each product `(A, B)` in order and each of its leaves in order, one
`C.AND` of the leaf's operands, its form XORed into that leaf's sum. -/
def leafSums : List (List Lin × List Lin) → List Lin → M (List Lin)
  | [], sums => pure sums
  | p :: ps, sums => do
    let zs ← (leaves 7 p.1 p.2).mapM fun xy => Flock.HmNets.and xy.1 xy.2
    leafSums ps (xorL sums zs)

/-- One residual (`rec_residuals.py:322-337`): its added terms' XOR (`acc`); if it has products, their leaf sums from
`LEAVES = 3^7` zero forms, committed (`sha512.commit`), recombined and reduced (`rec[k]`), XORed into `acc`'s 128 forms. -/
def residual (S : GfS) (W V : List Lin) (r : Res) : M (List Lin) := do
  let acc := xorAll (r.add.map (termForms W V))
  if r.mul.isEmpty then pure acc
  else do
    let sums ← leafSums (r.mul.map fun ab => (opForms S W V ab.1, opForms S W V ab.2)) (List.replicate (3 ^ 7) Lin.zero)
    let leaf ← commitW sums
    pure (xorL acc (reduceG (recomb 7 leaf)))

/-- The residuals in order, each one's forms after the one before's (`bits += acc`, `rec_residuals.py:320-338`). -/
def residualsGo (S : GfS) (W V : List Lin) : List Res → M (List Lin)
  | [] => pure []
  | r :: rs => do
    let acc ← residual S W V r
    let bits ← residualsGo S W V rs
    pure (acc ++ bits)

/-- **`residual_forms`** (`rec_residuals.py:294-338`): `GfResiduals{S}`'s residuals over forms, 128 each, in order, from
`w`'s and `v`'s element bits `W` and `V` (element `i` is bits `128 i ..`), each residual's products summed leaf by leaf. -/
def residualForms (S : GfS) (W V : List Lin) : M (List Lin) := residualsGo S W V S.res

/-! ## What it computes -/

/-- Bit `q` of a port's elements laid end to end: bit `q % 128` of element `q / 128`. -/
def wbit (w : ℕ → F128) (q : ℕ) : Bool := bitOf (w (q / 128)) (q % 128)

/-- The element whose bit `j` is `f j` for `j < 128`: the sum of the monomials `x^j` (`Flock.F128.monomial`) at those `j`. -/
def ofBits (f : ℕ → Bool) : F128 :=
  ((List.range 128).map fun j => if f j then F128.monomial j else 0).foldl (· + ·) 0

/-- `gf2k._term_value` (`gf2k.py:236-244`) on the ports' elements. `["wt", i, u]`'s bit `j` is the circuit's,
`W[128 (i + j) + u]`, which for `u < 128` is `gf2k`'s bit `u` of `w[i + j]`. -/
def termVal (w v : ℕ → F128) : Term → F128
  | .w i => w i
  | .v i => v i
  | .wt i u => ofBits fun j => wbit w (128 * (i + j) + u)

/-- `gf2k.residuals`' `op` (`gf2k.py:253-259`): operand `k`'s value, its terms' sum. -/
def opVal (S : GfS) (val : Term → F128) (k : ℕ) : F128 := ((S.ops.getD k []).map val).foldl (· + ·) 0

/-- `gf2k.residuals` (`gf2k.py:262-268`): a residual's value, its added terms' sum, then each product `ops[a]·ops[b]` added,
in `Flock.F128`. -/
def resVal (S : GfS) (val : Term → F128) (r : Res) : F128 :=
  (r.mul.map fun ab => opVal S val ab.1 * opVal S val ab.2).foldl (· + ·) ((r.add.map val).foldl (· + ·) 0)

/-! ## Bits and polynomials -/

theorem coeff_poly (x : F128) (k : ℕ) : (FlockLevel3.F128.poly x).coeff k = b2 (bitOf x k) := by
  rw [FlockLevel3.F128.poly, coeff_natPoly, bitOf]

theorem bitOf_ge (x : F128) {k : ℕ} (hk : 128 ≤ k) : bitOf x k = false :=
  Nat.testBit_lt_two_pow (lt_of_lt_of_le (w128_lt _ _) (Nat.pow_le_pow_right (by norm_num) hk))

/-- Forms carry an element's bits exactly when their polynomial is the element's. -/
theorem polyOf_eq_poly_iff (z : Var → Bool) (f : List Lin) (x : F128) :
    polyOf z f = FlockLevel3.F128.poly x ↔ ∀ k, ev z (f.getD k Lin.zero) = bitOf x k := by
  constructor
  · intro h k
    have := congrArg (fun q => q.coeff k) h
    simp only [coeff_polyOf, coeff_poly] at this
    exact b2_inj this
  · intro h
    ext k
    rw [coeff_polyOf, coeff_poly, h k]

theorem polyOf_map_range (z : Var → Bool) (g : ℕ → Lin) (x : F128) (h : ∀ j < 128, ev z (g j) = bitOf x j) :
    polyOf z ((List.range 128).map g) = FlockLevel3.F128.poly x := by
  refine (polyOf_eq_poly_iff z _ x).2 fun k => ?_
  by_cases hk : k < 128
  · rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getElem_map, List.getElem_range, h k hk]
  · rw [List.getD_eq_default _ _ (by simp; omega), ev_zero, bitOf_ge x (by omega)]

theorem poly_monomial {j : ℕ} (hj : j < 128) : FlockLevel3.F128.poly (F128.monomial j) = X ^ j := by
  have hw : w128 (F128.monomial j).lo (F128.monomial j).hi = 2 ^ j := by
    unfold F128.monomial
    have one_lt : ∀ k < 64, (1 : UInt64).toNat < 2 ^ (64 - k) := fun k hk => by
      rw [UInt64.toNat_one]
      exact Nat.one_lt_two_pow (by omega)
    split_ifs with h
    · simp only [w128, UInt64.toNat_zero, mul_zero, zero_add]
      rw [FlockLevel3.shl_lit 1 _ j (FlockLevel3.toNat_toUInt64 j h) h (one_lt j h), UInt64.toNat_one, Nat.one_shiftLeft]
    · simp only [w128, UInt64.toNat_zero, add_zero]
      rw [FlockLevel3.shl_lit 1 _ (j - 64) (FlockLevel3.toNat_toUInt64 _ (by omega)) (by omega) (one_lt _ (by omega)),
        UInt64.toNat_one, Nat.one_shiftLeft, ← pow_add]
      congr 1
      omega
  rw [FlockLevel3.F128.poly, hw]
  refine natPoly_ext fun k => ?_
  rw [coeff_X_pow, Nat.testBit_two_pow]
  by_cases hk : k = j
  · subst hk; simp
  · simp [hk, Ne.symm hk]

theorem poly_foldl_add (l : List F128) (a : F128) :
    FlockLevel3.F128.poly (l.foldl (· + ·) a) = FlockLevel3.F128.poly a + (l.map FlockLevel3.F128.poly).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih =>
      simp only [List.foldl_cons, ih, FlockLevel3.F128.poly_add, List.map_cons, List.sum_cons]
      ring

theorem coeff_sum_range (g : ℕ → Bool) : ∀ (m k : ℕ),
    (((List.range m).map fun j => if g j then (X : P) ^ j else 0).sum).coeff k = if k < m then b2 (g k) else 0
  | 0, k => by simp
  | m + 1, k => by
      rw [List.range_succ, List.map_append, List.sum_append, coeff_add, coeff_sum_range g m k]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
      by_cases hk : k < m
      · have hne : k ≠ m := by omega
        cases g m <;> simp [hk, coeff_X_pow, hne, show k < m + 1 by omega]
      · by_cases he : k = m
        · subst he
          cases g k <;> simp
        · have : ¬ k < m + 1 := by omega
          cases g m <;> simp [hk, this, coeff_X_pow, he]

theorem bitOf_ofBits (f : ℕ → Bool) {j : ℕ} (hj : j < 128) : bitOf (ofBits f) j = f j := by
  have h : FlockLevel3.F128.poly (ofBits f) = ((List.range 128).map fun j => if f j then (X : P) ^ j else 0).sum := by
    rw [ofBits, poly_foldl_add, FlockLevel3.F128.poly_zero, zero_add, List.map_map]
    congr 1
    refine List.map_congr_left fun i hi => ?_
    simp only [Function.comp]
    split_ifs
    · exact poly_monomial (List.mem_range.1 hi)
    · exact FlockLevel3.F128.poly_zero
  have := congrArg (fun q => q.coeff j) h
  simp only [coeff_poly, coeff_sum_range, hj, ↓reduceIte] at this
  exact b2_inj this

theorem sum_modByMonic (l : List P) : l.sum %ₘ FlockLevel3.G = (l.map (· %ₘ FlockLevel3.G)).sum := by
  induction l with
  | nil => simp
  | cons x l ih => rw [List.sum_cons, add_modByMonic, ih, List.map_cons, List.sum_cons]

/-! ## Terms and operands -/

theorem bit_eq_getD (z : Var → Bool) (W : List Lin) (q : ℕ) : bit z W q = ev z (W.getD q Lin.zero) := by
  simp [bit, List.getD_eq_getElem?_getD]

theorem wbit_elem (w : ℕ → F128) (i : ℕ) {j : ℕ} (hj : j < 128) : wbit w (128 * i + j) = bitOf (w i) j := by
  rw [wbit, show (128 * i + j) / 128 = i by omega, show (128 * i + j) % 128 = j by omega]

theorem length_termForms (W V : List Lin) (t : Term) : (termForms W V t).length = 128 := by
  cases t <;> simp [termForms]

theorem polyOf_termForms (z : Var → Bool) {W V : List Lin} {w v : ℕ → F128} (hW : ∀ q, bit z W q = wbit w q)
    (hV : ∀ q, bit z V q = wbit v q) (t : Term) :
    polyOf z (termForms W V t) = FlockLevel3.F128.poly (termVal w v t) := by
  cases t with
  | w i =>
      simp only [termForms, termVal]
      exact polyOf_map_range z _ _ fun j hj => by rw [← bit_eq_getD, hW, wbit_elem w i hj]
  | v i =>
      simp only [termForms, termVal]
      exact polyOf_map_range z _ _ fun j hj => by rw [← bit_eq_getD, hV, wbit_elem v i hj]
  | wt i u =>
      simp only [termForms, termVal]
      exact polyOf_map_range z _ _ fun j hj => by rw [← bit_eq_getD, hW, bitOf_ofBits _ hj]

theorem xorL_eq_addL : ∀ (a b : List Lin), a.length = b.length → xorL a b = addL a b
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | f :: fs, g :: gs, h => by
      have := xorL_eq_addL fs gs (by simpa using h)
      simp only [xorL] at this ⊢
      rw [List.zipWith_cons_cons, addL, this]

theorem polyOf_replicate_zero (z : Var → Bool) (n : ℕ) : polyOf z (List.replicate n Lin.zero) = 0 := by
  have := polyOf_shiftL z n []
  simp only [shiftL, List.append_nil, polyOf, mul_zero] at this
  exact this

theorem xorAll_go (z : Var → Bool) : ∀ (xs : List (List Lin)) (acc : List Lin), acc.length = 128 →
    (∀ x ∈ xs, x.length = 128) →
    (xs.foldl xorL acc).length = 128 ∧ polyOf z (xs.foldl xorL acc) = polyOf z acc + (xs.map (polyOf z)).sum
  | [], acc, h, _ => by simp [h]
  | x :: xs, acc, hacc, hxs => by
      have hx := hxs x List.mem_cons_self
      have hl : (xorL acc x).length = 128 := by simp [xorL, hacc, hx]
      obtain ⟨h1, h2⟩ := xorAll_go z xs _ hl fun y hy => hxs y (List.mem_cons_of_mem _ hy)
      refine ⟨h1, ?_⟩
      rw [List.foldl_cons, h2, xorL_eq_addL _ _ (by rw [hacc, hx]), polyOf_addL, List.map_cons, List.sum_cons, add_assoc]

theorem length_xorAll (xs : List (List Lin)) (hxs : ∀ x ∈ xs, x.length = 128) : (xorAll xs).length = 128 :=
  (xorAll_go (fun _ => false) xs _ (List.length_replicate ..) hxs).1

theorem polyOf_xorAll (z : Var → Bool) (xs : List (List Lin)) (hxs : ∀ x ∈ xs, x.length = 128) :
    polyOf z (xorAll xs) = (xs.map (polyOf z)).sum := by
  rw [xorAll, (xorAll_go z xs _ (List.length_replicate ..) hxs).2, polyOf_replicate_zero, zero_add]

theorem length_xorAll_terms (W V : List Lin) (ts : List Term) : (xorAll (ts.map (termForms W V))).length = 128 :=
  length_xorAll _ (by simp [length_termForms])

theorem polyOf_xorAll_terms (z : Var → Bool) {W V : List Lin} {w v : ℕ → F128} (hW : ∀ q, bit z W q = wbit w q)
    (hV : ∀ q, bit z V q = wbit v q) (ts : List Term) :
    polyOf z (xorAll (ts.map (termForms W V))) = FlockLevel3.F128.poly ((ts.map (termVal w v)).foldl (· + ·) 0) := by
  rw [polyOf_xorAll z _ (by simp [length_termForms]), poly_foldl_add, FlockLevel3.F128.poly_zero, zero_add,
    List.map_map, List.map_map]
  exact congrArg List.sum (List.map_congr_left fun t _ => polyOf_termForms z hW hV t)

theorem length_opForms (S : GfS) (W V : List Lin) (k : ℕ) : (opForms S W V k).length = 128 :=
  length_xorAll_terms W V _

theorem polyOf_opForms (z : Var → Bool) (S : GfS) {W V : List Lin} {w v : ℕ → F128} (hW : ∀ q, bit z W q = wbit w q)
    (hV : ∀ q, bit z V q = wbit v q) (k : ℕ) :
    polyOf z (opForms S W V k) = FlockLevel3.F128.poly (opVal S (termVal w v) k) :=
  polyOf_xorAll_terms z hW hV _

/-! ## Karatsuba is linear in its leaves -/

/-- A leaf's AND under `z`, in GF(2). -/
def andv (z : Var → Bool) (xy : Lin × Lin) : ZMod 2 := b2 (ev z xy.1) * b2 (ev z xy.2)

theorem length_leaves : ∀ (n : ℕ) (a b : List Lin), (leaves n a b).length = 3 ^ n
  | 0, _, _ => rfl
  | n + 1, a, b => by
      simp only [leaves, List.length_append, length_leaves n]
      ring

theorem length_recomb : ∀ (n : ℕ) (l : List Lin), (recomb n l).length = 2 * 2 ^ n - 1
  | 0, _ => rfl
  | n + 1, l => by
      simp only [recomb, length_addL, shiftL, List.length_append, List.length_replicate, length_recomb n]
      have := Nat.two_pow_pos n
      rw [pow_succ]
      omega

theorem headD_eq_getD (l : List Lin) : l.headD Lin.zero = l.getD 0 Lin.zero := by
  cases l <;> rfl

theorem getD_take_of_lt {l : List Lin} {m i : ℕ} (h : i < m) : (l.take m).getD i Lin.zero = l.getD i Lin.zero := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_take, h, ↓reduceIte]

theorem getD_drop (l : List Lin) (m i : ℕ) : (l.drop m).getD i Lin.zero = l.getD (m + i) Lin.zero := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_drop]

theorem getD_append_left {l l' : List (Lin × Lin)} {i : ℕ} (d : Lin × Lin) (h : i < l.length) :
    (l ++ l').getD i d = l.getD i d := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_append_left h]

theorem getD_append_right {l l' : List (Lin × Lin)} (d : Lin × Lin) (i : ℕ) :
    (l ++ l').getD (l.length + i) d = l'.getD i d := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]

/-- The three halves Karatsuba splits a product into, the sums of products of each, and how they recombine. -/
theorem sum_split {α : Type} (ps : List α) (F L M H : α → P) (c : P)
    (key : ∀ p ∈ ps, F p = L p + c * (M p + L p + H p) + c ^ 2 * H p) :
    (ps.map F).sum = (ps.map L).sum + c * ((ps.map M).sum + (ps.map L).sum + (ps.map H).sum) +
      c ^ 2 * (ps.map H).sum := by
  induction ps with
  | nil => simp
  | cons p ps ih =>
      simp only [List.map_cons, List.sum_cons]
      rw [key p List.mem_cons_self, ih fun q hq => key q (List.mem_cons_of_mem _ hq)]
      ring

/-- **Karatsuba's recombination is linear in its leaves.** If each leaf form of `l` is, under `z`, the sum over the
products `ps` of that leaf's AND, its recombination is the sum of the products' polynomials. -/
theorem polyOf_recomb (z : Var → Bool) : ∀ (n : ℕ) (ps : List (List Lin × List Lin)) (l : List Lin),
    (∀ p ∈ ps, p.1.length ≤ 2 ^ n ∧ p.2.length ≤ 2 ^ n) →
    (∀ i < 3 ^ n, b2 (ev z (l.getD i Lin.zero)) =
      (ps.map fun p => andv z ((leaves n p.1 p.2).getD i (Lin.zero, Lin.zero))).sum) →
    polyOf z (recomb n l) = (ps.map fun p => polyOf z p.1 * polyOf z p.2).sum
  | 0, ps, l, hps, hl => by
      simp only [recomb, polyOf, mul_zero, add_zero, headD_eq_getD]
      rw [hl 0 (by norm_num), map_list_sum, List.map_map]
      refine congrArg List.sum (List.map_congr_left fun p hp => ?_)
      rw [polyOf_le_one z p.1 (by simpa using (hps p hp).1), polyOf_le_one z p.2 (by simpa using (hps p hp).2)]
      simp [leaves, andv]
  | n + 1, ps, l, hps, hl => by
      have hlt : ∀ f : List Lin, f.length ≤ 2 ^ (n + 1) → (f.take (2 ^ n)).length ≤ 2 ^ n ∧
          (f.drop (2 ^ n)).length ≤ 2 ^ n ∧ (addL (f.take (2 ^ n)) (f.drop (2 ^ n))).length ≤ 2 ^ n := by
        intro f hf
        simp only [List.length_take, List.length_drop, length_addL]
        rw [pow_succ] at hf
        omega
      have h3 : 3 ^ (n + 1) = 3 ^ n + 3 ^ n + 3 ^ n := by ring
      have hlen : ∀ a b : List Lin, (leaves n a b).length = 3 ^ n := length_leaves n
      have e0 := polyOf_recomb z n (ps.map fun p => (p.1.take (2 ^ n), p.2.take (2 ^ n))) (l.take (3 ^ n))
        (by
          simp only [List.mem_map]
          rintro _ ⟨p, hp, rfl⟩
          exact ⟨(hlt _ (hps p hp).1).1, (hlt _ (hps p hp).2).1⟩)
        (fun i hi => by
          rw [getD_take_of_lt hi, hl i (by omega), List.map_map]
          refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
          simp only [Function.comp, leaves, List.append_assoc]
          rw [getD_append_left _ (by rw [hlen]; exact hi)])
      have e2 := polyOf_recomb z n (ps.map fun p => (p.1.drop (2 ^ n), p.2.drop (2 ^ n))) ((l.drop (3 ^ n)).take (3 ^ n))
        (by
          simp only [List.mem_map]
          rintro _ ⟨p, hp, rfl⟩
          exact ⟨(hlt _ (hps p hp).1).2.1, (hlt _ (hps p hp).2).2.1⟩)
        (fun i hi => by
          rw [getD_take_of_lt hi, getD_drop, hl _ (by omega), List.map_map]
          refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
          simp only [Function.comp, leaves, List.append_assoc]
          rw [← hlen (p.1.take (2 ^ n)) (p.2.take (2 ^ n)), getD_append_right,
            getD_append_left _ (by rw [hlen]; exact hi)])
      have e1 := polyOf_recomb z n
        (ps.map fun p => (addL (p.1.take (2 ^ n)) (p.1.drop (2 ^ n)), addL (p.2.take (2 ^ n)) (p.2.drop (2 ^ n))))
        (l.drop (2 * 3 ^ n))
        (by
          simp only [List.mem_map]
          rintro _ ⟨p, hp, rfl⟩
          exact ⟨(hlt _ (hps p hp).1).2.2, (hlt _ (hps p hp).2).2.2⟩)
        (fun i hi => by
          rw [getD_drop, hl _ (by omega), List.map_map]
          refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
          simp only [Function.comp, leaves]
          rw [show 2 * 3 ^ n + i = (leaves n (p.1.take (2 ^ n)) (p.2.take (2 ^ n)) ++
              leaves n (p.1.drop (2 ^ n)) (p.2.drop (2 ^ n))).length + i by
                rw [List.length_append, hlen, hlen]; ring, getD_append_right])
      have key : ∀ p ∈ ps, polyOf z p.1 * polyOf z p.2 =
          polyOf z (p.1.take (2 ^ n)) * polyOf z (p.2.take (2 ^ n)) +
            X ^ 2 ^ n * (polyOf z (addL (p.1.take (2 ^ n)) (p.1.drop (2 ^ n))) *
                polyOf z (addL (p.2.take (2 ^ n)) (p.2.drop (2 ^ n))) +
              polyOf z (p.1.take (2 ^ n)) * polyOf z (p.2.take (2 ^ n)) +
              polyOf z (p.1.drop (2 ^ n)) * polyOf z (p.2.drop (2 ^ n))) +
            (X ^ 2 ^ n) ^ 2 * (polyOf z (p.1.drop (2 ^ n)) * polyOf z (p.2.drop (2 ^ n))) := by
        intro p _
        rw [polyOf_take_drop z (2 ^ n) p.1, polyOf_take_drop z (2 ^ n) p.2]
        simp only [polyOf_addL]
        linear_combination (-(X ^ 2 ^ n * (polyOf z (p.1.take (2 ^ n)) * polyOf z (p.2.take (2 ^ n)) +
          polyOf z (p.1.drop (2 ^ n)) * polyOf z (p.2.drop (2 ^ n))))) * two_eq_zero_P
      simp only [recomb, polyOf_addL, polyOf_shiftL]
      rw [e0, e1, e2, sum_split ps _ _ _ _ _ key]
      simp only [List.map_map, Function.comp_def]
      ring

/-! ## The gadget's rows -/

theorem sound_leafSums : ∀ (prods : List (List Lin × List Lin)) (sums : List Lin),
    Sound (leafSums prods sums) fun z out => sums.length = 3 ^ 7 → out.length = 3 ^ 7 ∧
      ∀ i < 3 ^ 7, b2 (ev z (out.getD i Lin.zero)) = b2 (ev z (sums.getD i Lin.zero)) +
        (prods.map fun p => andv z ((leaves 7 p.1 p.2).getD i (Lin.zero, Lin.zero))).sum
  | [], sums => Sound.pure' fun z hs => ⟨hs, fun i _ => by simp⟩
  | p :: ps, sums => by
      simp only [leafSums]
      refine Sound.bind' (sound_mapM _ (fun z xy => ev z xy.1 && ev z xy.2) (fun xy => sound_and xy.1 xy.2) _)
        fun zs => Sound.mono (sound_leafSums ps (xorL sums zs)) fun z out ih hzs hs => ?_
      obtain ⟨hzl, hzb⟩ := hzs
      rw [length_leaves] at hzl
      have hxl : (xorL sums zs).length = 3 ^ 7 := by simp [xorL, hs, hzl]
      obtain ⟨hol, hob⟩ := ih hxl
      refine ⟨hol, fun i hi => ?_⟩
      rw [hob i hi, xorL_eq_addL _ _ (by rw [hs, hzl]), ← coeff_polyOf, polyOf_addL, coeff_add, coeff_polyOf,
        coeff_polyOf, List.map_cons, List.sum_cons, ← bit_eq_getD z zs i, hzb i (by rw [length_leaves]; exact hi),
        List.getD_eq_getElem (leaves 7 p.1 p.2) (Lin.zero, Lin.zero) (by rw [length_leaves]; exact hi), andv, b2_and]
      ring

theorem length_reduceG_255 (f : List Lin) (h : f.length = 255) : (reduceG f).length = 128 := by
  simp only [reduceG, length_foldG, h]
  norm_num

theorem sound_residual (S : GfS) (W V : List Lin) (r : Res) :
    Sound (residual S W V r) fun z out => out.length = 128 ∧ ∀ w v : ℕ → F128, (∀ q, bit z W q = wbit w q) →
      (∀ q, bit z V q = wbit v q) → polyOf z out = FlockLevel3.F128.poly (resVal S (termVal w v) r) := by
  unfold residual
  refine Sound.ite' (fun he => Sound.pure' fun z => ⟨length_xorAll_terms W V _, fun w v hW hV => ?_⟩) fun hne => ?_
  · rw [polyOf_xorAll_terms z hW hV, resVal, List.isEmpty_iff.1 he, List.map_nil, List.foldl_nil]
  · simp only [commitW]
    refine Sound.bind' (sound_leafSums _ _) fun sums => Sound.bind' (sound_mapM _ (fun z f => ev z f) sound_commit sums)
      fun leaf => Sound.pure' fun z hleaf hsums => ?_
    obtain ⟨hsl, hsb⟩ := hsums (List.length_replicate ..)
    obtain ⟨hll, hlb⟩ := hleaf
    have hrl : (recomb 7 leaf).length = 255 := by rw [length_recomb]; norm_num
    refine ⟨?_, fun w v hW hV => ?_⟩
    · rw [xorL_eq_addL _ _ (by rw [length_xorAll_terms, length_reduceG_255 _ hrl]), length_addL,
        length_xorAll_terms, length_reduceG_255 _ hrl, max_self]
    have hrec := polyOf_recomb z 7 (r.mul.map fun ab => (opForms S W V ab.1, opForms S W V ab.2)) leaf
      (by
        simp only [List.mem_map]
        rintro _ ⟨ab, _, rfl⟩
        exact ⟨by rw [length_opForms]; norm_num, by rw [length_opForms]; norm_num⟩)
      (fun i hi => by
        rw [← bit_eq_getD, hlb i (by rw [hsl]; exact hi), ← List.getD_eq_getElem _ Lin.zero, hsb i hi,
          List.getD_replicate _ hi, ev_zero, b2_false, zero_add])
    rw [xorL_eq_addL _ _ (by rw [length_xorAll_terms, length_reduceG_255 _ hrl]), polyOf_addL,
      polyOf_reduceG z _ hrl.le, hrec, polyOf_xorAll_terms z hW hV, resVal, poly_foldl_add (r.mul.map _),
      sum_modByMonic, List.map_map, List.map_map, List.map_map]
    refine congrArg (FlockLevel3.F128.poly _ + ·) (congrArg List.sum (List.map_congr_left fun ab _ => ?_))
    simp only [Function.comp_def, FlockLevel3.F128.poly_mul, polyOf_opForms z S hW hV]

theorem sound_residualsGo (S : GfS) (W V : List Lin) : ∀ rs : List Res,
    Sound (residualsGo S W V rs) fun z out => out.length = 128 * rs.length ∧
      ∀ w v : ℕ → F128, (∀ q, bit z W q = wbit w q) → (∀ q, bit z V q = wbit v q) →
        ∀ r (hr : r < rs.length), ∀ k < 128, bit z out (128 * r + k) = bitOf (resVal S (termVal w v) rs[r]) k
  | [] => Sound.pure' fun z => ⟨rfl, fun w v _ _ r hr => absurd hr (by simp)⟩
  | r0 :: rs => by
      simp only [residualsGo]
      refine Sound.bind' (sound_residual S W V r0) fun acc => Sound.bind' (sound_residualsGo S W V rs) fun bits =>
        Sound.pure' fun z hb ha => ?_
      obtain ⟨hal, hav⟩ := ha
      obtain ⟨hbl, hbv⟩ := hb
      refine ⟨by simp [hal, hbl]; ring, fun w v hW hV r hr k hk => ?_⟩
      cases r with
      | zero =>
          rw [bit_eq_getD, Nat.mul_zero, Nat.zero_add, List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega),
            ← List.getD_eq_getElem?_getD, (polyOf_eq_poly_iff z acc _).1 (hav w v hW hV) k]
          rfl
      | succ r =>
          rw [show 128 * (r + 1) + k = acc.length + (128 * r + k) by rw [hal]; ring, bit, List.getElem?_append_right
            (Nat.le_add_right _ _), Nat.add_sub_cancel_left, ← bit, hbv w v hW hV r (by simpa using hr) k hk]
          rfl

/-- **V*'s residual gadget computes the residuals.** Whenever every row holds and the forms `W` and `V` carry the bits of
the elements `w` and `v` (element `i` at bit `128 i`), `residualForms S W V` has 128 forms a residual, and residual `r`'s
carry the bits of its value in `Flock.F128`: its added terms' sum plus `Σ_p ops[a_p]·ops[b_p]` (`resVal`,
`gf2k.residuals`). -/
theorem sound_residualForms (S : GfS) (W V : List Lin) :
    Sound (residualForms S W V) fun z out => out.length = 128 * S.res.length ∧
      ∀ w v : ℕ → F128, (∀ q, bit z W q = wbit w q) → (∀ q, bit z V q = wbit v q) →
        ∀ r (hr : r < S.res.length), ∀ k < 128, bit z out (128 * r + k) = bitOf (resVal S (termVal w v) S.res[r]) k :=
  sound_residualsGo S W V S.res

end FlockVBridge
