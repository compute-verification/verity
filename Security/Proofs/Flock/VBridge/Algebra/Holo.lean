import Proofs.Flock.VBridge.Algebra.RunSkel

/-!
# L4′: the lincheck with hidden units, at the residuals

A holo shape (`rec_algebra.Shape.holo`) absorbs the hidden part's fold `h` in a round of its own after `z_partial`
(`H_ROUND`), and its comb operands are `comb[l] + h[l]` (`("combh", l)`). Its `GfResiduals_v1{S}` is the plain one's
with `h` inserted into w at `p`, the elements before `H_ROUND` (`holoS`): w's later slots move up 64 and comb slot `l`'s
operand gains `w[p + l]`. `residuals_of_holo`: its residuals at w with `h` inserted (`insH`) are the plain ones with
comb `comb + h` (`addH`). `lincheck_of_residuals_holo`: so when they are zero the lincheck's rounds end at
`dot (comb + h) z_partial`, which over F is `holoLincheck`'s check `Σ_s (folded[s] + h[s])·z_partial[s] = claim`
(`fOf_dot_holo`). It starts from the plain S (`hpl`): at a holo shape `structureOf` is already the holo S, and
`lincheck_of_residuals_hw` reads its residuals at w itself, ending the rounds at `dot (comb + h) z_partial` for the `h`
w holds (`hOfW`).
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-! ## The holo structure -/

/-- w's slot `i` once `h` is inserted at `p`. -/
def shiftW (p i : Nat) : Nat := if i < p then i else i + 64

/-- A term of the holo S: w's slots moved past `h`, and comb slot `c0 + l` with `w[p + l]` added. -/
def holoTerm (p c0 : Nat) : Term → Array Term
  | .w i => #[.w (shiftW p i)]
  | .wt i u => #[.wt (shiftW p i) u]
  | .v j => if c0 ≤ j ∧ j < c0 + 64 then #[.v j, .w (p + (j - c0))] else #[.v j]

/-- **The holo S** (`rec_algebra._build` on a holo shape): the plain S with `h`'s 64 elements inserted into w at `p`,
the comb's slots from `c0`. -/
def holoS (p c0 : Nat) (S : GfS) : GfS :=
  ⟨S.nw + 64, S.nv, S.ops.map (·.flatMap (holoTerm p c0)),
    S.res.map fun r => ⟨r.mul, r.add.flatMap (holoTerm p c0)⟩⟩

/-- w with `h` inserted at `p`. -/
def insH (p : Nat) (w h : Array F128) : Array F128 :=
  Array.ofFn (n := w.size + 64) fun i =>
    if i.val < p then w[i.val]! else if i.val < p + 64 then h[i.val - p]! else w[i.val - 64]!

/-- `comb + h`. -/
def addH (comb h : Array F128) : Array F128 := Array.ofFn (n := 64) fun i => comb[i.val]! + h[i.val]!

/-- A ring switch's 128 slots lie on one side of `p`. -/
def WtClear (p : Nat) : Term → Prop
  | .wt i _ => p ≤ i ∨ i + 128 ≤ p
  | _ => True

/-- Every ring switch of S lies on one side of `p` (a holo shape's read the claims' `s`, which follow `h`). -/
def HoloOk (p : Nat) (S : GfS) : Prop :=
  (∀ op ∈ S.ops, ∀ t ∈ op, WtClear p t) ∧ ∀ r ∈ S.res, ∀ t ∈ r.add, WtClear p t

/-! ## Reads -/

theorem insH_get {p : Nat} {w : Array F128} (h : Array F128) (hp : p ≤ w.size) (i : Nat) :
    (insH p w h)[i]! = if i < p then w[i]! else if i < p + 64 then h[i - p]! else w[i - 64]! := by
  by_cases hi : i < w.size + 64
  · rw [getElem!_pos (insH p w h) i (by simpa [insH] using hi)]
    simp only [insH, Array.getElem_ofFn]
  · rw [getElem!_neg (insH p w h) i (by simpa [insH] using hi), ite_eq_right (by omega), ite_eq_right (by omega),
      getElem!_neg w (i - 64) (by omega)]

theorem insH_shift {p : Nat} {w : Array F128} (h : Array F128) (hp : p ≤ w.size) (i : Nat) :
    (insH p w h)[shiftW p i]! = w[i]! := by
  unfold shiftW
  by_cases hi : i < p
  · rw [ite_eq_left hi, insH_get h hp, ite_eq_left hi]
  · rw [ite_eq_right hi, insH_get h hp, ite_eq_right (by omega), ite_eq_right (by omega), Nat.add_sub_cancel]

theorem insH_comb {p : Nat} {w : Array F128} (h : Array F128) (hp : p ≤ w.size) {l : Nat} (hl : l < 64) :
    (insH p w h)[p + l]! = h[l]! := by
  rw [insH_get h hp, ite_eq_right (by omega), ite_eq_left (by omega), Nat.add_sub_cancel_left]

theorem insH_wt {p i : Nat} {w : Array F128} (h : Array F128) (hp : p ≤ w.size) (hi : p ≤ i ∨ i + 128 ≤ p)
    {v : Nat} (hv : v < 128) : (insH p w h)[shiftW p i + v]! = w[i + v]! := by
  unfold shiftW
  rcases hi with hi | hi
  · rw [ite_eq_right (by omega), insH_get h hp, ite_eq_right (by omega), ite_eq_right (by omega),
      show i + 64 + v - 64 = i + v by omega]
  · rw [ite_eq_left (by omega), insH_get h hp, ite_eq_left (by omega)]

theorem addH_size (comb h : Array F128) : (addH comb h).size = 64 := by
  simp [addH]

theorem addH_get (comb h : Array F128) {l : Nat} (hl : l < 64) : (addH comb h)[l]! = comb[l]! + h[l]! := by
  rw [getElem!_pos (addH comb h) l (by simpa [addH] using hl)]
  simp only [addH, Array.getElem_ofFn]

theorem app_get (V a : Array F128) (j : Nat) : (V ++ a)[j]! = if j < V.size then V[j]! else a[j - V.size]! := by
  by_cases hj : j < V.size
  · rw [ite_eq_left hj, getElem!_pos (V ++ a) j (by rw [Array.size_append]; omega), getElem!_pos V j hj,
      Array.getElem_append_left hj]
  · rw [ite_eq_right hj]
    by_cases ha : j - V.size < a.size
    · rw [getElem!_pos (V ++ a) j (by rw [Array.size_append]; omega), getElem!_pos a (j - V.size) ha,
        Array.getElem_append_right (by omega)]
    · rw [getElem!_neg (V ++ a) j (by rw [Array.size_append]; omega), getElem!_neg a (j - V.size) ha]

/-! ## Values -/

theorem afoldl_congr {α β : Type} (f g : β → α → β) :
    ∀ (l : List α), (∀ a ∈ l, ∀ b, f b a = g b a) → ∀ b, l.foldl f b = l.foldl g b
  | [], _, _ => rfl
  | a :: l, h, b => by
    rw [List.foldl_cons, List.foldl_cons, h a (by simp) b]
    exact afoldl_congr f g l (fun x hx => h x (by simp [hx])) _

theorem afoldlA_congr {α β : Type} (f g : β → α → β) (xs : Array α) (h : ∀ a ∈ xs, ∀ b, f b a = g b a) (b : β) :
    xs.foldl f b = xs.foldl g b := by
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  exact afoldl_congr f g xs.toList (fun a ha => h a (by simpa using ha)) b

theorem termValue_wt (w w' V V' : Array F128) (i i' u : Nat) (h : ∀ v < 128, w[i + v]! = w'[i' + v]!) :
    termValue w V (.wt i u) = termValue w' V' (.wt i' u) := by
  simp only [termValue]
  exact afoldlA_congr _ _ _ (fun v hv b => by simp only [h v (Array.mem_range.mp hv)]) 0

theorem foldl_single {β : Type} (f : β → Term → β) (t : Term) (b : β) : (#[t] : Array Term).foldl f b = f b t := by
  simp only [List.foldl_toArray', List.foldl_cons, List.foldl_nil]

variable {p : Nat} (V comb h w : Array F128)

theorem termValue_holo (hc : comb.size = 64) (hp : p ≤ w.size) (t : Term) (ht : WtClear p t) (x : F128) :
    (holoTerm p V.size t).foldl (fun x t => x + termValue (insH p w h) (V ++ comb) t) x =
      x + termValue w (V ++ addH comb h) t := by
  cases t with
  | w i => rw [holoTerm, foldl_single, termValue, termValue, insH_shift h hp]
  | wt i u =>
    rw [holoTerm, foldl_single, termValue_wt (insH p w h) w (V ++ comb) (V ++ addH comb h) (shiftW p i) i u
      (fun v hv => insH_wt h hp ht hv)]
  | v j =>
    simp only [holoTerm]
    split_ifs with hj
    · simp only [List.foldl_toArray', List.foldl_cons, List.foldl_nil, termValue]
      rw [app_get, app_get, ite_eq_right (by omega), ite_eq_right (by omega), addH_get comb h (by omega),
        insH_comb h hp (show j - V.size < 64 by omega)]
      apply fOf_injective
      simp only [fOf_add]
      ring
    · rw [foldl_single, termValue, termValue, app_get, app_get]
      by_cases hv : j < V.size
      · rw [ite_eq_left hv, ite_eq_left hv]
      · rw [ite_eq_right hv, ite_eq_right hv, getElem!_neg comb _ (by omega),
          getElem!_neg (addH comb h) _ (by rw [addH_size]; omega)]

variable {V comb h w}

theorem holoS_op {c0 : Nat} {S : GfS} {k : Nat} (hk : k < S.ops.size) :
    (holoS p c0 S).ops[k]! = (S.ops[k]'hk).flatMap (holoTerm p c0) := by
  rw [getElem!_pos (holoS p c0 S).ops k (by simpa [holoS] using hk)]
  exact Array.getElem_map _ _

theorem opValue_holo {S : GfS} (hc : comb.size = 64) (hp : p ≤ w.size) (hok : HoloOk p S) (k : Nat) :
    opValue (holoS p V.size S) (insH p w h) (V ++ comb) k = opValue S w (V ++ addH comb h) k := by
  unfold opValue
  by_cases hk : k < S.ops.size
  · rw [holoS_op hk, getElem!_pos S.ops k hk, Array.foldl_flatMap]
    exact afoldlA_congr _ _ _
      (fun t ht b => termValue_holo V comb h w hc hp t (hok.1 _ (Array.getElem_mem hk) t ht) b) 0
  · rw [getElem!_neg (holoS p V.size S).ops k (by simpa [holoS] using hk), getElem!_neg S.ops k hk]
    rfl

theorem resValue_holo {S : GfS} (hc : comb.size = 64) (hp : p ≤ w.size) (hok : HoloOk p S) (r : Res)
    (hr : ∀ t ∈ r.add, WtClear p t) :
    resValue (holoS p V.size S) (insH p w h) (V ++ comb) ⟨r.mul, r.add.flatMap (holoTerm p V.size)⟩ =
      resValue S w (V ++ addH comb h) r := by
  unfold resValue
  simp only [opValue_holo hc hp hok]
  rw [Array.foldl_flatMap, afoldlA_congr _ _ r.add
    (fun t ht b => termValue_holo V comb h w hc hp t (hr t ht) b) 0]

/-- **The holo residuals are the plain ones at `comb + h`.** -/
theorem residuals_of_holo {S : GfS} {rs : Array F128} (hc : comb.size = 64) (hp : p ≤ w.size) (hok : HoloOk p S)
    (hres : residuals (holoS p V.size S) (insH p w h) (V ++ comb) = .ok rs) :
    residuals S w (V ++ addH comb h) = .ok rs := by
  unfold residuals at hres ⊢
  have hs : (insH p w h).size = w.size + 64 := by simp [insH]
  have hv : (V ++ addH comb h).size = (V ++ comb).size := by simp [addH_size, hc]
  split at hres
  · cases hres
  · rename_i hn
    simp only [holoS, hs] at hn
    rw [ite_eq_right (by rw [hv]; omega)]
    cases hres
    simp only [holoS, Array.map_map]
    exact congrArg Except.ok (Array.map_congr_left fun r hr => (resValue_holo hc hp hok r (hok.2 r hr)).symm)

/-! ## The lincheck -/

/-- `verifierRows` places the comb last, and nothing else it returns reads it. -/
theorem verifierRows_comb {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    ∃ V : Array F128, v = V ++ comb ∧ ∀ comb', verifierRows sh rounds extras comb' = .ok (V ++ comb', info, kappa) := by
  obtain ⟨⟨S, hS⟩, ⟨lins, info'⟩, hl⟩ := verifierRows_ok_iff.mp ⟨_, hrows⟩
  obtain ⟨b, hb, -, hb₁⟩ := verifierRows_ok (comb := comb) hS hl
  rw [hrows] at hb₁
  cases hb₁
  refine ⟨b.vVals, rfl, fun comb' => ?_⟩
  obtain ⟨b', hb', -, hb₂⟩ := verifierRows_ok (comb := comb') hS hl
  rw [hb] at hb'
  cases hb'
  exact hb₂

/-- **L4′ at a rep.** When the holo S's residuals (`holoS`, the comb's slots from `v.size − 64`) at w with `h` inserted
at `p` are all zero, on the v `verifierRows` gives at the rep's coins, a proof carrying the lincheck messages ends the
verifier's lincheck rounds at `dot (comb + h) z_partial`: `holoLincheck`'s check (`fOf_dot_holo`). -/
theorem lincheck_of_residuals_holo {sh : Shape} {w comb h : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128} (hpl : sh.holo = false)
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64) (hp : p ≤ w.size) (hok : HoloOk p S)
    (hacc : ∀ r < 2, valOf sh w (addH comb h) cons (.acc1 r) + valOf sh w (addH comb h) cons (.acc0 r) =
      kapSum kappa r cons)
    (hres : residuals (holoS p (v.size - 64) S) (insH p w h) v = .ok rs) (hzero : ∀ x ∈ rs, x = 0)
    {pf : LincheckProof} (hpf : LcProofOf (decoded sh w (addH comb h) cons) pf)
    {C : Nat → Array F128} (hC : ∀ j < sh.kLog - 6, (C j).getD 0 0 = info.lc.fold.ts[j]!) :
    lcFold pf C (sh.kLog - 6) (info.lc.fold.alpha * (decoded sh w (addH comb h) cons).fa +
        (decoded sh w (addH comb h) cons).fb + info.lc.fold.beta) =
      (dot (addH comb h) pf.zPartial, info.lc.fold.ts) := by
  obtain ⟨V, rfl, hV⟩ := verifierRows_comb hrows
  rw [show (V ++ comb).size - 64 = V.size by rw [Array.size_append, hc, Nat.add_sub_cancel]] at hres
  exact lincheck_of_residuals hpl (hV (addH comb h)) hS hw (addH_size comb h) hacc
    (residuals_of_holo hc hp hok hres) hzero hpf hC

/-- w's `h` as the comb's elements read it (`hOf`): zero off a holo shape. -/
def hOfW (sh : Shape) (w : Array F128) : Array F128 := Array.ofFn (n := 64) fun l => hOf sh w l.val

theorem hOfW_get (sh : Shape) (w : Array F128) {l : Nat} (hl : l < 64) : (hOfW sh w)[l]! = hOf sh w l := by
  rw [getElem!_pos (hOfW sh w) l (by simpa [hOfW] using hl)]
  simp only [hOfW, Array.getElem_ofFn]

/-- The comb w holds is the comb plus `hOfW` (`decoded_comb_get`). -/
theorem decoded_comb_eq (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) :
    (decoded sh w comb cons).comb = addH comb (hOfW sh w) := by
  apply Array.ext
  · rw [decoded_comb_size, addH_size]
  · intro i _ h2
    have hi : i < 64 := by rwa [addH_size] at h2
    have e1 := getElem!_pos (decoded sh w comb cons).comb i (by rw [decoded_comb_size]; exact hi)
    have e2 := getElem!_pos (addH comb (hOfW sh w)) i (by rw [addH_size]; exact hi)
    rw [decoded_comb_get sh w comb cons hi] at e1
    rw [addH_get comb _ hi, hOfW_get sh w hi] at e2
    exact e1.symm.trans e2

/-- **L4′ at a holo shape.** `lincheck_of_residuals_dec` with the comb w holds written out: when the residuals of
`structureOf sh`, whose comb operands at a holo `sh` read w's `h`, are all zero, a proof carrying the lincheck messages
ends the verifier's lincheck rounds at `dot (comb + h) z_partial` for w's `h` (`hOfW`): `holoLincheck`'s check
(`fOf_dot_holo`). -/
theorem lincheck_of_residuals_hw {sh : Shape} {w comb : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons)
    (hres : residuals S w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0) {pf : LincheckProof}
    (hpf : LcProofOf (decoded sh w comb cons) pf)
    {C : Nat → Array F128} (hC : ∀ j < sh.kLog - 6, (C j).getD 0 0 = info.lc.fold.ts[j]!) :
    lcFold pf C (sh.kLog - 6)
        (info.lc.fold.alpha * (decoded sh w comb cons).fa + (decoded sh w comb cons).fb + info.lc.fold.beta) =
      (dot (addH comb (hOfW sh w)) pf.zPartial, info.lc.fold.ts) := by
  have h := lincheck_of_residuals_dec hrows hS hw hc hacc hres hzero hpf hC
  rwa [decoded_comb_eq sh w comb cons] at h

/-- **`holoLincheck`'s check over F**: `dot (comb + h) z_partial` is `Σ_s (comb[s] + h[s])·z_partial[s]`. -/
theorem fOf_dot_holo (comb h zp : Array F128) :
    fOf (dot (addH comb h) zp) = ∑ s : Fin 64, (fOf comb[s.val]! + fOf h[s.val]!) * fOf zp[s.val]! := by
  rw [fOf_dot _ _ 64 (addH_size comb h)]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [addH_get comb h s.isLt, fOf_add]

end Flock.RecAlgebra
