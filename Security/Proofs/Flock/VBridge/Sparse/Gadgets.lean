import Proofs.Flock.VBridge.Verifier
import Flock.RecSparse

/-!
# The sparse-phase unit's gadgets

The gadgets `Flock.RecSparse.steps` builds past `GfResiduals`' (circuits' #1751), each on forms that carry elements of
`Flock.F128` (`Holds`: 128 forms, form `k` carrying bit `k`), and each against a mirror of it on the elements, so that
what the gadget computes is stated apart from the algebra:

* `mulF` is the product (`sound_mulF`): both operands committed, Karatsuba's ANDs (`leaves 7`), recombined and reduced.
* `pairsMul` and `tree` are Φ's product tree in its pairing (`pairsVal`, `treeVal`).
* `demuxLevel` splits each slot's vectors by one bit of the slot's type, `demux` level by level (`demuxVal`).
* `RecOpen.xorL` and `xorAllL` are sums (`holds_xorL`, `sumVal`).
-/

namespace FlockVBridge.Sparse

open Flock.HmNets hiding and
open Flock (F128)
open FlockSoundness.Discharge.Hm Polynomial
open FlockLevel3 (P b2 b2_false b2_and)

/-- Forms `f` carry `x`: 128 forms, form `k` carrying bit `k` of `x`. -/
def Holds (z : Var → Bool) (f : List Lin) (x : F128) : Prop := f.length = 128 ∧ polyOf z f = FlockLevel3.F128.poly x

theorem f128_add_self (x : F128) : x + x = 0 :=
  FlockLevel3.F128.poly_injective (by
    rw [FlockLevel3.F128.poly_add, FlockLevel3.F128.poly_zero]
    linear_combination FlockLevel3.F128.poly x * FlockLevel3.two_eq_zero_P)

theorem f128_add_zero (x : F128) : x + 0 = x :=
  FlockLevel3.F128.poly_injective (by rw [FlockLevel3.F128.poly_add, FlockLevel3.F128.poly_zero, add_zero])

theorem polyOf_congr {z : Var → Bool} {a b : List Lin} (hl : a.length = b.length)
    (h : ∀ i (hi : i < b.length), bit z a i = ev z b[i]) : polyOf z a = polyOf z b := by
  ext i
  rw [coeff_polyOf, coeff_polyOf]
  by_cases hi : i < b.length
  · rw [← bit_eq_getD, h i hi, List.getD_eq_getElem b Lin.zero hi]
  · rw [List.getD_eq_default _ _ (by omega), List.getD_eq_default _ _ (by omega)]

theorem holds_xorL {z : Var → Bool} {a b : List Lin} {x y : F128} (ha : Holds z a x) (hb : Holds z b y) :
    Holds z (Flock.RecOpen.xorL a b) (x + y) := by
  rw [xorL_eq, xorL_eq_addL a b (ha.1.trans hb.1.symm)]
  refine ⟨by rw [length_addL, ha.1, hb.1, max_self], ?_⟩
  rw [polyOf_addL, ha.2, hb.2, FlockLevel3.F128.poly_add]

/-! ## The product -/

/-- **`mulF` is the product**: whenever its rows hold, it gives 128 forms carrying `x * y` of operands carrying `x` and
`y`. -/
theorem sound_mulF (a b : List Lin) :
    Sound (Flock.RecSparse.mulF a b) fun z c => ∀ x y, Holds z a x → Holds z b y → Holds z c (x * y) := by
  simp only [Flock.RecSparse.mulF, Flock.HmNets.commitW, leaves_eq, recomb_eq, reduceG_eq]
  refine Sound.bind' (sound_mapM _ (fun z f => ev z f) sound_commit a) fun a' => ?_
  refine Sound.bind' (sound_mapM _ (fun z f => ev z f) sound_commit b) fun b' => ?_
  refine Sound.bind' (sound_mapM _ (fun z xy => ev z xy.1 && ev z xy.2) (fun xy => sound_and xy.1 xy.2) _)
    fun zs => Sound.pure' fun z hzs hb' ha' x y hx hy => ?_
  obtain ⟨hal, hab⟩ := ha'
  obtain ⟨hbl, hbb⟩ := hb'
  obtain ⟨hzl, hzb⟩ := hzs
  have hrl : (recomb 7 zs).length = 255 := by rw [length_recomb]; norm_num
  have hrec := polyOf_recomb z 7 [(a', b')] zs
    (by
      simp only [List.mem_singleton]
      rintro _ rfl
      exact ⟨by simp [hal, hx.1], by simp [hbl, hy.1]⟩)
    (fun i hi => by
      have hi' : i < (leaves 7 a' b').length := by rw [length_leaves]; exact hi
      rw [← bit_eq_getD, hzb i hi']
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
      rw [List.getD_eq_getElem _ _ hi', andv, b2_and])
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero] at hrec
  refine ⟨length_reduceG_255 _ hrl, ?_⟩
  rw [polyOf_reduceG z _ hrl.le, hrec, polyOf_congr hal hab, polyOf_congr hbl hbb, hx.2, hy.2,
    FlockLevel3.F128.poly_mul]

/-! ## Lists of gadgets -/

/-- A gadget on each element of a list, each result related to the image of its element's value. -/
theorem sound_mapM_rel {β γ ε η : Type} (f : β → M γ) (R : (Var → Bool) → β → ε → Prop) (g : (Var → Bool) → ε → η)
    (Q : (Var → Bool) → γ → η → Prop) (hf : ∀ x, Sound (f x) fun z r => ∀ e, R z x e → Q z r (g z e)) :
    ∀ l : List β, Sound (l.mapM f) fun z rs => ∀ es, List.Forall₂ (R z) l es → List.Forall₂ (Q z) rs (es.map (g z))
  | [] => Sound.pure' fun z es hes => by cases hes; exact .nil
  | x :: l => by
      rw [List.mapM_cons]
      refine Sound.bind' (hf x) fun r => Sound.bind' (sound_mapM_rel f R g Q hf l) fun rs =>
        Sound.pure' fun z hrs hr es hes => ?_
      cases hes with
      | cons he hes' => exact .cons (hr _ he) (hrs _ hes')

/-- The same over a list's elements with their indices from `k`. -/
theorem sound_mapM_zipIdx {β γ ε η : Type} (f : β × ℕ → M γ) (R : (Var → Bool) → β → ε → Prop)
    (g : (Var → Bool) → ε → ℕ → η) (Q : (Var → Bool) → γ → η → Prop)
    (hf : ∀ x s, Sound (f (x, s)) fun z r => ∀ e, R z x e → Q z r (g z e s)) :
    ∀ (l : List β) (k : ℕ), Sound ((l.zipIdx k).mapM f) fun z rs => ∀ es, List.Forall₂ (R z) l es →
      List.Forall₂ (Q z) rs ((es.zipIdx k).map fun p => g z p.1 p.2)
  | [], k => Sound.pure' fun z es hes => by cases hes; exact .nil
  | x :: l, k => by
      rw [List.zipIdx_cons, List.mapM_cons]
      refine Sound.bind' (hf x k) fun r => Sound.bind' (sound_mapM_zipIdx f R g Q hf l (k + 1)) fun rs =>
        Sound.pure' fun z hrs hr es hes => ?_
      cases hes with
      | cons he hes' =>
        rw [List.zipIdx_cons, List.map_cons]
        exact .cons (hr _ he) (hrs _ hes')

theorem forall₂_flatten {α β : Type} {R : α → β → Prop} :
    ∀ {l : List (List α)} {l' : List (List β)}, List.Forall₂ (List.Forall₂ R) l l' →
      List.Forall₂ R l.flatten l'.flatten
  | _, _, .nil => .nil
  | _, _, .cons h hs => by
      rw [List.flatten_cons, List.flatten_cons]
      exact List.rel_append h (forall₂_flatten hs)

/-! ## Φ's product tree -/

/-- `pairsMul`'s values: consecutive pairs multiplied in order, an odd last factor carried. -/
def pairsVal : List F128 → List F128
  | [] => []
  | [x] => [x]
  | x :: y :: rest => x * y :: pairsVal rest

/-- `tree`'s value: the product tree on the factors' values, level by level, `n` levels at most. -/
def treeVal : ℕ → List F128 → F128
  | 0, xs => xs.headD 0
  | _ + 1, [] => 0
  | _ + 1, [x] => x
  | n + 1, x :: y :: rest => treeVal n (pairsVal (x :: y :: rest))

theorem sound_pairsMul : ∀ fac : List (List Lin), Sound (Flock.RecSparse.pairsMul fac) fun z out =>
    ∀ xs, List.Forall₂ (Holds z) fac xs → List.Forall₂ (Holds z) out (pairsVal xs)
  | [] => Sound.pure' fun z xs hxs => by cases hxs; exact .nil
  | [f] => Sound.pure' fun z xs hxs => by
      cases hxs with
      | cons hx hr => cases hr; exact .cons hx .nil
  | a :: b :: rest => by
      simp only [Flock.RecSparse.pairsMul]
      refine Sound.bind' (sound_mulF a b) fun p => Sound.bind' (sound_pairsMul rest) fun ps =>
        Sound.pure' fun z hps hp xs hxs => ?_
      cases hxs with
      | cons hx hxs' =>
        cases hxs' with
        | cons hy hrest => exact .cons (hp _ _ hx hy) (hps _ hrest)

/-- **`tree` is the product tree**: whenever its rows hold, on factors carrying `xs` it gives forms carrying
`treeVal n xs`. -/
theorem sound_tree : ∀ (n : ℕ) (fac : List (List Lin)), Sound (Flock.RecSparse.tree n fac) fun z out =>
    ∀ xs, fac ≠ [] → List.Forall₂ (Holds z) fac xs → Holds z out (treeVal n xs)
  | 0, fac => Sound.pure' fun z xs hne hxs => by
      cases hxs with
      | nil => exact absurd rfl hne
      | cons hx _ => exact hx
  | n + 1, [] => Sound.pure' fun z xs hne _ => absurd rfl hne
  | n + 1, [f] => Sound.pure' fun z xs _ hxs => by
      cases hxs with
      | cons hx hr => cases hr; exact hx
  | n + 1, a :: b :: rest => by
      simp only [Flock.RecSparse.tree]
      refine Sound.bind' (sound_pairsMul _) fun ps => Sound.mono (sound_tree n ps) fun z out ih hps xs _ hxs => ?_
      cases hxs with
      | cons hx hxs' =>
        cases hxs' with
        | cons hy hrest =>
          have hp := hps _ (.cons hx (.cons hy hrest))
          refine ih _ ?_ hp
          rintro rfl
          cases hp

/-! ## The demultiplexer -/

/-- `demuxLevel`'s split of one slot's vectors at bit value `b`: each `x` becomes `x` where `b` is 0, then `x` where it
is 1. -/
def splitVal (b : Bool) (xs : List F128) : List F128 := xs.flatMap fun x => [if b then 0 else x, if b then x else 0]

/-- `demuxLevel`'s values: slot `s`'s vectors split at bit `j` of its type (`tb s j`). -/
def demuxLevelVal (tb : ℕ → ℕ → Bool) (j : ℕ) (Xs : List (List F128)) : List (List F128) :=
  (Xs.zipIdx 0).map fun p => splitVal (tb p.2 j) p.1

/-- `demux`'s values: the levels in order. -/
def demuxVal (tb : ℕ → ℕ → Bool) : List ℕ → List (List F128) → List (List F128)
  | [], Xs => Xs
  | j :: js, Xs => demuxVal tb js (demuxLevelVal tb j Xs)

theorem sound_split (t : Lin) (x : List Lin) :
    Sound (do
        let hi ← x.mapM fun b => Flock.HmNets.and t b
        pure [Flock.RecOpen.xorL x hi, hi]) fun z r => ∀ X, Holds z x X →
      List.Forall₂ (Holds z) r [if ev z t then 0 else X, if ev z t then X else 0] := by
  refine Sound.bind' (sound_mapM _ (fun z b => ev z t && ev z b) (fun b => sound_and t b) x) fun hi =>
    Sound.pure' fun z hhi X hx => ?_
  obtain ⟨hl, hb⟩ := hhi
  have hh : Holds z hi (if ev z t then X else 0) := by
    refine ⟨hl.trans hx.1, ?_⟩
    cases ht : ev z t
    · simp only [Bool.false_eq_true, ↓reduceIte]
      rw [FlockLevel3.F128.poly_zero, ← polyOf_replicate_zero z 128]
      exact polyOf_congr (by rw [hl, hx.1, List.length_replicate]) fun i hi' => by
        rw [hb i (by rw [List.length_replicate, ← hx.1] at hi'; exact hi'), List.getElem_replicate, ht, ev_zero,
          Bool.false_and]
    · simp only [↓reduceIte]
      rw [← hx.2]
      exact polyOf_congr hl fun i hi' => by
        rw [hb i hi']
        simp [ht]
  have hlo := holds_xorL hx hh
  refine .cons ?_ (.cons hh .nil)
  cases ht : ev z t
  · simp only [ht, Bool.false_eq_true, ↓reduceIte, f128_add_zero] at hlo ⊢
    exact hlo
  · simp only [ht, ↓reduceIte, f128_add_self] at hlo ⊢
    exact hlo

theorem sound_demuxLevel (tau : Array Lin) (j : ℕ) (lists : List (List (List Lin))) :
    Sound (Flock.RecSparse.demuxLevel tau j lists) fun z out => ∀ Xs,
      List.Forall₂ (List.Forall₂ (Holds z)) lists Xs →
      List.Forall₂ (List.Forall₂ (Holds z)) out (demuxLevelVal (fun s j => ev z (tau.getD (16 * s + j) Lin.zero)) j Xs) := by
  unfold Flock.RecSparse.demuxLevel demuxLevelVal
  refine sound_mapM_zipIdx _ (fun z xs Xs => List.Forall₂ (Holds z) xs Xs)
    (fun z Xs s => splitVal (ev z (tau.getD (16 * s + j) Lin.zero)) Xs) (fun z r e => List.Forall₂ (Holds z) r e)
    (fun xs s => ?_) lists 0
  dsimp only
  refine Sound.bind' (sound_mapM_rel _ (fun z x X => Holds z x X)
    (fun z X => [if ev z (tau.getD (16 * s + j) Lin.zero) then 0 else X,
      if ev z (tau.getD (16 * s + j) Lin.zero) then X else 0])
    (fun z r e => List.Forall₂ (Holds z) r e) (fun x => sound_split _ x) xs) fun out =>
    Sound.pure' fun z hout Xs hXs => forall₂_flatten (hout Xs hXs)

/-- **`demux` selects by the slots' types**: whenever its rows hold, on slots of vectors carrying `Xs` it gives the
vectors `demuxVal` splits them into at the types' bits `js`, each bit read from `tau`'s forms. -/
theorem sound_demux (tau : Array Lin) : ∀ (js : List ℕ) (lists : List (List (List Lin))),
    Sound (Flock.RecSparse.demux tau js lists) fun z out => ∀ Xs,
      List.Forall₂ (List.Forall₂ (Holds z)) lists Xs →
      List.Forall₂ (List.Forall₂ (Holds z)) out (demuxVal (fun s j => ev z (tau.getD (16 * s + j) Lin.zero)) js Xs)
  | [], lists => by
      simp only [Flock.RecSparse.demux]
      exact Sound.pure' fun z Xs h => h
  | j :: js, lists => by
      simp only [Flock.RecSparse.demux]
      exact Sound.bind' (sound_demuxLevel tau j lists) fun l =>
        Sound.mono (sound_demux tau js l) fun z out ih hl Xs h => ih _ (hl Xs h)

/-! ## Sums -/

/-- `xorAllL`'s value: the elements' sum, none for no element. -/
def sumVal : List F128 → F128
  | [] => 0
  | x :: xs => xs.foldl (· + ·) x

theorem holds_foldl_xorL {z : Var → Bool} : ∀ {fs : List (List Lin)} {xs : List F128}, List.Forall₂ (Holds z) fs xs →
    ∀ {a : List Lin} {x : F128}, Holds z a x → Holds z (fs.foldl Flock.RecOpen.xorL a) (xs.foldl (· + ·) x)
  | _, _, .nil, _, _, h => h
  | _, _, .cons hf hfs, _, _, h => holds_foldl_xorL hfs (holds_xorL h hf)

theorem holds_xorAllL {z : Var → Bool} {fs : List (List Lin)} {xs : List F128} (h : List.Forall₂ (Holds z) fs xs)
    (hne : fs ≠ []) : Holds z (Flock.RecSparse.xorAllL fs) (sumVal xs) := by
  cases h with
  | nil => exact absurd rfl hne
  | cons hf hfs => exact holds_foldl_xorL hfs hf

end FlockVBridge.Sparse
