import Proofs.Flock.Soundness.TmplSplitOk

/-!
# Every walk means its class's

- `walkCall_exp`, `walkCall_sum`: a call's step (`walkCall`), from its callee's table (`walkChild_ok`), adds the callee's
  copies' `refVal`s (`callVal`), for an explicit and a summed class.
- `walk_ok`: by induction on the depth, every walk's table means its class's (`WalkOk`).
-/

open Flock Flock.Tmpl FlockLevel3 GF128 Std

namespace FlockSoundness.TmplFold

/-- Call `i` of Definition `d`'s copy at `(A, B)`, its inputs at `ip`: its callee's copies' `refVal`s at depth `fuel`. -/
noncomputable def callVal (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s fuel d A B : ℕ) (ip : List ℕ) (i : ℕ) :
    GF128 :=
  ∑ j ∈ Finset.range ((t.defs.getD d default).calls.getD i default).count,
    refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_
      (A + B * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs))
      (B * ((lays.getD d default).places.getD i default).ms)
      (((t.defs.getD d default).calls.getD i default).bound (readAt t lays pt d A B ip) j)

theorem refVal_succ (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s fuel d A B : ℕ) (ip : List ℕ) :
    refVal t lays pt s (fuel + 1) d A B ip =
      ((t.defs.getD d default).rows.toList.map fun r =>
        rowVal pt s (pt.e (A + B * Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)) (readAt t lays pt d A B ip) r).sum +
      ∑ i ∈ Finset.range (t.defs.getD d default).calls.size, callVal t lays pt s fuel d A B ip i := by
  simp only [refVal, callVal]

/-! ## Folds in `StateT`, and small facts -/

theorem foldlM_sumF {ι α β σ : Type} (S : ι → β → GF128) (Q : β → σ → Prop) (g : ι → α → GF128) :
    ∀ (l : List α) {f : β → α → StateT σ V β} {b b' : β} {st st' : σ},
      (l.foldlM f b).run st = .ok (b', st') → Q b st →
      (∀ a ∈ l, ∀ b st b1 st1, (f b a).run st = .ok (b1, st1) → Q b st → Q b1 st1 ∧ ∀ x, S x b1 = S x b + g x a) →
      Q b' st' ∧ ∀ x, S x b' = S x b + (l.map (g x)).sum
  | [], f, b, b', st, st', h, hq, _ => by
    rw [List.foldlM_nil, StateT.run_pure] at h
    cases h
    exact ⟨hq, fun x => by rw [List.map_nil, List.sum_nil, add_zero]⟩
  | a :: l, f, b, b', st, st', h, hq, hf => by
    rw [List.foldlM_cons] at h
    obtain ⟨b1, st1, h1, h⟩ := sv_bind_ok h
    obtain ⟨hq1, hs1⟩ := hf a (List.mem_cons_self ..) b st b1 st1 h1 hq
    obtain ⟨hq2, hs2⟩ := foldlM_sumF S Q g l h hq1 (fun a ha => hf a (List.mem_cons_of_mem _ ha))
    exact ⟨hq2, fun x => by rw [hs2, hs1, List.map_cons, List.sum_cons, add_assoc]⟩

theorem sv_throw_bind {α β : Type} (e : String) (f : α → SV β) (st : SplitSt) :
    ((throw e : SV α) >>= f).run st = .error e := rfl

theorem getD_toArray (l : List ℕ) (n : ℕ) : l.toArray.getD n 0 = l.getD n 0 := by
  first
  | (simp [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]; done)
  | (simp [Array.getD, List.getD]; done)

theorem eLow_eq (pt : FoldPt) (β a : ℕ) : pt.lw.getD (a % 64) 0 * Tmpl.eqAt pt.rho 0 (β - 6) (a / 64) = eLow pt β a :=
  rfl

theorem uOk_empty (ins nk : ℕ) : UOk ins nk {} := fun x hx => by
  first
  | simp at hx
  | exact absurd hx Std.HashMap.not_mem_emptyc

theorem low_lt {β : ℕ} {as : Array ℕ} (has : ∀ k < as.size, as.getD k 0 < 2 ^ β) (k : ℕ) : as.getD k 0 < 2 ^ β := by
  by_cases hk : k < as.size
  · exact has k hk
  · first
    | (simp [Array.getD, hk]; done)
    | (unfold Array.getD; rw [dif_neg hk]; exact Nat.two_pow_pos β)

theorem tabExp_trans_U {pt : FoldPt} {s L H : ℕ} {iv : ℕ → ℕ → GF128} (iv' : ℕ → ℕ → GF128) {T : Tab} {R : GF128}
    (h : ∀ x, x ∈ T.U → iv x.1 x.2.1 = iv' x.1 x.2.1) (h' : tabExp pt s L H iv' T = R) : tabExp pt s L H iv T = R :=
  (tabExp_congr_U pt s L H iv iv' T h).trans h'

theorem tabSum_trans_U {pt : FoldPt} {L H : ℕ} {gl gc : GF128} {iv : ℕ → GF128} (iv' : ℕ → GF128) {T : Tab}
    {e R : GF128} (h : ∀ x, x ∈ T.U → iv x.1 = iv' x.1) (h' : e * tabSum pt L H gl gc iv' T = R) :
    e * tabSum pt L H gl gc iv T = R := by
  rw [tabSum_congr_U pt L H gl gc iv iv' T h, h']

theorem hsum_congr {κ : Type} [BEq κ] [LawfulBEq κ] [Hashable κ] [LawfulHashable κ] (m : HashMap κ F128)
    (f f' : κ → GF128) (h : ∀ x, x ∈ m → f x = f' x) : hsum m f = hsum m f' := by
  unfold hsum
  refine congrArg List.sum (List.map_congr_left fun e he => ?_)
  show of e.2 * f e.1 = of e.2 * f' e.1
  rw [h e.1 (mem_of_toList he)]

theorem flatMap_sum (c : ℕ) (f : ℕ → ℕ → ℕ) (g : ℕ → GF128) : ∀ n,
    (((List.range n).flatMap fun k => (List.range c).map fun j => f k j).map g).sum =
      ∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, g (f k j)
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, List.map_append, List.sum_append, flatMap_sum c f g n,
      Finset.sum_range_succ, ← list_range_sum (fun j => g (f n j))]
    simp [Function.comp_def]

/-- A run of lanes's position: lane `j` of copy `a` at chunk origin `H` is the caller's position of the copy's `j`-th
child. -/
theorem lane_pos {a β j m ms H al off nb cs : ℕ} (hpow : 2 ^ m = ms) (hdiv : (off + nb) % ms = 0) (hm : m ≤ al)
    (hcs : cs = 1) :
    a + 2 ^ β * j + 2 ^ (β + m) * (H * 2 ^ (al - m) + (off + nb) / ms) =
      a + 2 ^ β * (H * 2 ^ al + off) + 2 ^ β * (nb + j * cs) := by
  have h2 : 2 ^ al = ms * 2 ^ (al - m) := by rw [← hpow, ← Nat.pow_add, Nat.add_sub_cancel' hm]
  calc a + 2 ^ β * j + 2 ^ (β + m) * (H * 2 ^ (al - m) + (off + nb) / ms)
      = a + 2 ^ β * j + 2 ^ β * (H * 2 ^ al) + 2 ^ β * (ms * ((off + nb) / ms)) := by
        rw [Nat.pow_add, hpow, h2]; ring
    _ = a + 2 ^ β * j + 2 ^ β * (H * 2 ^ al) + 2 ^ β * (off + nb) := by
        rw [Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hdiv)]
    _ = a + 2 ^ β * (H * 2 ^ al + off) + 2 ^ β * (nb + j * cs) := by subst hcs; ring

theorem slot_glue {R : Type} [CommRing R] (n c : ℕ) (e g U : ℕ → ℕ → R) (S C q : R) :
    (∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, e k j * g k j) * S +
        (∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, e k j) * q * C +
      ∑ j ∈ Finset.range c, ∑ k ∈ Finset.range n, e k j * U k j =
    ∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, e k j * (g k j * S + q * C + U k j) := by
  have hc : ∑ j ∈ Finset.range c, ∑ k ∈ Finset.range n, e k j * U k j =
      ∑ k ∈ Finset.range n, ∑ j ∈ Finset.range c, e k j * U k j := Finset.sum_comm
  rw [hc]
  simp only [Finset.sum_mul, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun j _ => by ring

theorem lane_glue2 {R : Type} [CommRing R] (eL X gl gc Sc Cc : R) (E Rr U : ℕ → R) (n : ℕ) :
    eL * (X + (gl * (∑ j ∈ Finset.range n, E j * Rr j) * Sc + gc * (∑ j ∈ Finset.range n, E j) * Cc +
      ∑ j ∈ Finset.range n, E j * U j)) =
    eL * X + ∑ j ∈ Finset.range n, eL * E j * (gl * Rr j * Sc + gc * Cc + U j) := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [Finset.sum_range_succ]
    linear_combination ih

/-! ## A bound input, read -/

theorem iv_exp (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {d s β al off H : ℕ}
    {as : Array ℕ} (h6 : 6 ≤ β + al) (hL : β + al - 6 ≤ pt.r.size) (ip : ℕ → List ℕ) (c : Call) {k j t' : ℕ}
    (hk : as.getD k 0 < 2 ^ β) (ht : t' < (c.bound id j).length) :
    refX (t.defs.getD d default) pt s (β + al - 6) H (2 ^ al) (2 ^ β) as (eqHalf pt.r 0 (β + al - 6))
        (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) k
        ((c.bound id j).toArray.getD t' 0) =
      Gs pt s ((c.bound (readAt t lays pt d (as.getD k 0 + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) (ip k)) j).getD t'
        pt.qc) := by
  rw [refX_eq t lays pt hR h6 hL hk, getD_toArray (c.bound id j), getD_bound c _ j t' pt.qc ht]

theorem iv_sum (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {d s β al off H a : ℕ}
    (h6 : 6 ≤ β) (hL : β + al - 6 ≤ pt.r.size) (ha : a < 2 ^ β) (ip : List ℕ) (c : Call) {j t' : ℕ}
    (ht : t' < (c.bound id j).length) :
    refS (t.defs.getD d default) pt (β + al - 6) H (2 ^ al) (eqHalf pt.r (β - 6) al)
        (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) (gLow pt s β a) (Gs pt s pt.qc)
        (fun t' => Gs pt s (ip.getD t' pt.qc)) ((c.bound id j).toArray.getD t' 0) =
      Gs pt s ((c.bound (readAt t lays pt d (a + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) ip) j).getD t' pt.qc) := by
  rw [refS_eq t lays pt hR h6 hL ha, getD_toArray (c.bound id j), getD_bound c _ j t' pt.qc ht]

/-- `child_expW`, its open inputs read through `Z`. -/
theorem child_expW' (D : Def) (pt : FoldPt) {s : ℕ} (hs : s < 64) (L H Lc Hc A B : ℕ) (as : Array ℕ) (gL : Half)
    (pos : ℕ → ℕ) (iv : ℕ → ℕ → GF128) (c : Call) (o ch : Tab) (count nk n : ℕ) (rm cm : ℕ → ℕ × F128)
    (W : Array F128) (EL : F128) (E : ℕ → ℕ → F128) (hk : ReKey pt L H Lc Hc rm cm) (Z : ℕ → ℕ → ℕ → GF128)
    (hZ : ∀ k j x, x ∈ ch.U → refX D pt s L H A B as gL pos iv k ((c.bound id j).toArray.getD x.1 0) = Z k j x.1) :
    tabExp pt s L H iv (Tab.binds (Tab.bind (Tab.ref D pt false A B as gL pos) c) (Tab.slots pt W EL n o ch rm cm)
        count ch rm fun _ j => (List.range nk).map fun k => (k, E k j)) = tabExp pt s L H iv o +
      (of (W.getD s 0) * hsum ch.S (fun e => hiE pt Lc Hc e.1 * hiG pt Lc Hc e.2) +
        of EL * Gs pt s pt.qc * hsum ch.C (fun c => hiE pt Lc Hc c) +
        ∑ j ∈ Finset.range count, ∑ k ∈ Finset.range nk,
          of (E k j) * hsum ch.U (fun e => hiE pt Lc Hc e.2.2 * Z k j e.1)) := by
  rw [child_expW D pt hs L H Lc Hc A B as gL pos iv c o ch count nk n rm cm W EL E hk]
  refine congrArg (HAdd.hAdd _) (congrArg (HAdd.hAdd _) (Finset.sum_congr rfl fun j _ =>
    Finset.sum_congr rfl fun k _ => ?_))
  rw [hsum_congr ch.U
    (fun e => hiE pt Lc Hc e.2.2 * refX D pt s L H A B as gL pos iv k ((c.bound id j).toArray.getD e.1 0))
    (fun e => hiE pt Lc Hc e.2.2 * Z k j e.1) fun x hx => by rw [hZ k j x hx]]

/-- `child_sumL`, its open inputs read through `Z`. -/
theorem child_sumL' (D : Def) (pt : FoldPt) (L H Lc Hc A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (gl gc : GF128) (iv : ℕ → GF128) (c : Call) (o ch : Tab) (count : ℕ) (rm cm : ℕ → ℕ × F128) (ls lc : F128)
    (E : ℕ → F128) (hk : ReKey pt L H Lc Hc rm cm) (Z : ℕ → ℕ → GF128)
    (hZ : ∀ j x, x ∈ ch.U → refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD x.1 0) = Z j x.1) :
    tabSum pt L H gl gc iv (Tab.binds (Tab.bind (Tab.ref D pt true A B as gL pos) c) (o.addChild ch rm cm ls lc)
        count ch rm fun _ j => [(0, E j)]) = tabSum pt L H gl gc iv o +
      (gl * of ls * hsum ch.S (fun e => hiE pt Lc Hc e.1 * hiG pt Lc Hc e.2) +
        gc * of lc * hsum ch.C (fun c => hiE pt Lc Hc c) +
        ∑ j ∈ Finset.range count, of (E j) * hsum ch.U (fun e => hiE pt Lc Hc e.2.2 * Z j e.1)) := by
  rw [child_sumL D pt L H Lc Hc A B as gL pos gl gc iv c o ch count rm cm ls lc E hk]
  refine congrArg (HAdd.hAdd _) (congrArg (HAdd.hAdd _) (Finset.sum_congr rfl fun j _ => ?_))
  rw [hsum_congr ch.U
    (fun e => hiE pt Lc Hc e.2.2 * refS D pt L H A gL pos gl gc iv ((c.bound id j).toArray.getD e.1 0))
    (fun e => hiE pt Lc Hc e.2.2 * Z j e.1) fun x hx => by rw [hZ j x hx]]

/-! ## A call's step -/

/-- **An explicit class's call**: it adds its callee's copies' `refVal`s, copy by copy. -/
theorem walkCall_exp {t : Tmpl} {lays : Array Lay} {multi : Array Bool} {pt : FoldPt} {s : ℕ} (hs : s < 64)
    (hT : pt.rhoT = Tmpl.halves pt.rho) (hR : pt.rT = Tmpl.halves pt.r) (hlay : LayOk t lays)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    {F : ℕ} (hw : WalkOk t lays multi pt s F) {d β off al i : ℕ} {as : Array ℕ} {o o' : Tab} {st st' : SplitSt}
    (h : (walkCall t lays pt (walkChild lays multi pt (walk t lays multi pt F)) d β as off al
      (eqHalf pt.r 0 (β + al - 6)) o i).run st = .ok (o', st'))
    (hst : StOk t lays pt s st) (hU : UOk (t.defs.getD d default).ins as.size o) (hd : d < F + 1)
    (hi : i < (t.defs.getD d default).calls.size) (h6 : ¬ 6 ≤ β) (hpre : Pre pt β al as) :
    StOk t lays pt s st' ∧ UOk (t.defs.getD d default).ins as.size o' ∧ ∀ (H : ℕ) (ip : ℕ → List ℕ),
      tabExp pt s (β + al - 6) H (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) o' =
        tabExp pt s (β + al - 6) H (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) o +
          ∑ k ∈ Finset.range as.size,
            callVal t lays pt s F d (as.getD k 0 + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) (ip k) i := by
  obtain ⟨h6a, hL1, hL2, has⟩ := hpre
  have he : ((t.defs.getD d default).calls.getD i default).def_ < F := by have := hcall d i hi; omega
  have hbd : ∀ j, (((t.defs.getD d default).calls.getD i default).bound id j).length =
      (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).ins := fun j => hlay.bound d i j hi
  have hlan := hlay.lanes d i hi
  have hnk : nkOf β as = as.size := if_neg h6
  unfold walkCall at h
  simp only [h6, decide_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte] at h
  unfold callVal
  generalize (t.defs.getD d default).calls.getD i default = c at h he hbd hlan ⊢
  generalize (lays.getD d default).places.getD i default = pl at h hlan ⊢
  split at h
  · -- a copy placed directly
    rename_i hm
    have hm1 : pl.ms = 1 := by simpa using hm
    have key := foldlM_sumF
      (fun (x : ℕ × (ℕ → List ℕ)) (o : Tab) =>
        tabExp pt s (β + al - 6) x.1 (fun t' k => Gs pt s ((x.2 k).getD t' pt.qc)) o)
      (fun (o : Tab) (st : SplitSt) => StOk t lays pt s st ∧ UOk (t.defs.getD d default).ins as.size o)
      (fun x j => ∑ k ∈ Finset.range as.size,
        refVal t lays pt s F c.def_ (as.getD k 0 + 2 ^ β * (x.1 * 2 ^ al + off) + 2 ^ β * (pl.nb + j * pl.cs))
          (2 ^ β * pl.ms) (c.bound (readAt t lays pt d (as.getD k 0 + 2 ^ β * (x.1 * 2 ^ al + off)) (2 ^ β) (x.2 k)) j))
      _ h ⟨hst, hU⟩ ?_
    · obtain ⟨hq, hv⟩ := key
      refine ⟨hq.1, hq.2, fun H ip => ?_⟩
      have hv' := hv (H, ip)
      dsimp only at hv'
      rw [hv', list_range_sum]
      exact congrArg (HAdd.hAdd _) Finset.sum_comm
    intro j _ b st1 b1 st2 hr hq
    obtain ⟨⟨ch, rm, cm⟩, st3, hc, hr⟩ := sv_bind_ok hr
    try dsimp only at hr
    obtain ⟨rfl, rfl⟩ := sv_pure_ok hr
    obtain ⟨hst3, al'', off'', hsp, hH⟩ := walkChild_ok hT hR hcall hw hc hq.1 he ⟨h6a, hL1, hL2, has⟩
    have hchU := hsp.1
    rw [hnk] at hchU
    refine ⟨⟨hst3, bind_U _ _ _ _ _ _ _ _ _ _ _ _ _ _ (fun e he kf hkf => ?_) hq.2⟩, fun x => ?_⟩
    · rw [List.mem_singleton] at hkf
      rw [hkf]
      exact (hchU e.1 (mem_of_toList he)).2
    obtain ⟨Hc, hHc, hr1, hr2⟩ := hH x.1
    rw [child_exp1 _ pt s _ _ _ _ _ _ as _ _ _ _ b ch j rm cm ⟨hr1, hr2⟩]
    refine congrArg (HAdd.hAdd _) (tabExp_trans_U
      (fun t' k => Gs pt s ((c.bound (readAt t lays pt d (as.getD k 0 + 2 ^ β * (x.1 * 2 ^ al + off)) (2 ^ β)
        (x.2 k)) j).getD t' pt.qc)) (fun y hy =>
        iv_exp t lays pt hR h6a hL2 x.2 c (low_lt (has h6) _) (by rw [hbd j]; exact (hchU y hy).1))
      ((hsp.2.1 h6 Hc _).trans (Finset.sum_congr rfl fun k _ => ?_)))
    rw [hHc, hm1, Nat.mul_one]
    congr 1
    ring
  · rename_i hm
    have hm' : pl.ms ≠ 1 := by simpa using hm
    obtain ⟨hcs, hpow, hcnt⟩ := hlan hm'
    have hc' : c.count ≤ 2 ^ pl.ms.log2 := hcnt.trans_eq hpow.symm
    have hB : 2 ^ β * pl.ms = 2 ^ (β + pl.ms.log2) := by rw [Nat.pow_add, hpow]
    split at h
    · first
      | (rw [sv_throw_bind] at h; cases h)
      | (rw [sv_throw] at h; cases h)
    rename_i hal
    have hal1 : (off + pl.nb) % pl.ms = 0 := by
      by_contra hc
      exact hal (by simp [hc])
    have hal2 : pl.ms.log2 ≤ al := by
      by_contra hc
      exact hal (by simp [Nat.lt_of_not_le hc])
    split at h
    · -- a run of lanes below an explicit class
      rename_i hlt
      obtain ⟨⟨ch, rm, cm⟩, st3, hc, hr⟩ := sv_bind_ok h
      try dsimp only at hr
      obtain ⟨rfl, rfl⟩ := sv_pure_ok hr
      have hpc : Pre pt (β + pl.ms.log2) (al - pl.ms.log2)
          ((Array.range (as.size * c.count)).map fun i => as.getD (i / c.count) 0 + 2 ^ β * (i % c.count)) :=
        ⟨by omega, by omega, by omega, fun _ k hk => by
          rw [Array.size_map, Array.size_range] at hk
          have hc0 : 0 < c.count := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hk; omega
          rw [getD_lanes _ _ _ _ hk]
          exact lane_lt (has h6 _ (Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hk)))
            (Nat.lt_of_lt_of_le (Nat.mod_lt _ hc0) hc')⟩
      obtain ⟨hst3, al'', off'', hsp, hH⟩ := walkChild_ok hT hR hcall hw hc hst he hpc
      have hchU := hsp.1
      simp only [nkOf, show ¬ 6 ≤ β + pl.ms.log2 by omega, ↓reduceIte, Array.size_map, Array.size_range] at hchU
      refine ⟨hst3, binds_U _ _ _ _ _ _ _ _ _ _ _ _ _ _ (fun j e he kf hkf => ?_) hU, fun H ip => ?_⟩
      · split_ifs at hkf
        · rw [List.mem_singleton] at hkf
          rw [hkf]
          exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact (hchU e.1 (mem_of_toList he)).2)
        · simp at hkf
      obtain ⟨Hc, hHc, hr1, hr2⟩ := hH H
      rw [show β + pl.ms.log2 + (al - pl.ms.log2) - 6 = β + al - 6 by omega] at hr1 hr2
      rw [child_expL _ pt s _ _ _ _ _ _ as _ _ _ _ o ch c.count as.size rm cm ⟨hr1, hr2⟩ fun y hy => (hchU y hy).2]
      refine congrArg (HAdd.hAdd _) (tabExp_trans_U
        (fun t' kc => Gs pt s ((c.bound (readAt t lays pt d (as.getD (kc / c.count) 0 + 2 ^ β * (H * 2 ^ al + off))
          (2 ^ β) (ip (kc / c.count))) (kc % c.count)).getD t' pt.qc)) (fun y hy =>
          iv_exp t lays pt hR h6a hL2 ip c (low_lt (has h6) _) (by rw [hbd]; exact (hchU y hy).1))
        ((hsp.2.1 (by omega) Hc _).trans ?_))
      rw [hHc, Array.size_map, Array.size_range, sum_lanes _ c.count as.size]
      refine Finset.sum_congr rfl fun k hk => Finset.sum_congr rfl fun j hj => ?_
      have hk := Finset.mem_range.mp hk
      have hj := Finset.mem_range.mp hj
      have hc0 : 0 < c.count := by omega
      have hlt' : k * c.count + j < as.size * c.count :=
        Nat.lt_of_lt_of_le (Nat.add_lt_add_left hj _) (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hk)
      rw [getD_lanes _ _ _ _ hlt', divmod_div hc0 hj, divmod_mod hj, lane_pos hpow hal1 hal2 hcs, hB]
    · -- a run of lanes from an explicit class into a summed one
      rename_i hlt
      simp only [eLow_eq] at h
      obtain ⟨⟨ch, rm, cm⟩, st3, hc, hr⟩ := sv_bind_ok h
      try dsimp only at hr
      obtain ⟨rfl, rfl⟩ := sv_pure_ok hr
      have h6' : 6 ≤ β + pl.ms.log2 := by omega
      have hpc : Pre pt (β + pl.ms.log2) (al - pl.ms.log2) #[] :=
        ⟨by omega, by omega, by omega, fun h' => absurd h6' h'⟩
      obtain ⟨hst3, al'', off'', hsp, hH⟩ := walkChild_ok hT hR hcall hw hc hst he hpc
      have hchU := hsp.1
      refine ⟨hst3, binds_U _ _ _ _ _ _ _ _ _ _ _ _ _ _ (fun j e he kf hkf => ?_) hU, fun H ip => ?_⟩
      · obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hkf
        exact List.mem_range.1 hk
      obtain ⟨Hc, hHc, hr1, hr2⟩ := hH H
      rw [show β + pl.ms.log2 + (al - pl.ms.log2) - 6 = β + al - 6 by omega] at hr1 hr2
      rw [child_expW' _ pt hs _ _ _ _ _ _ as _ _ _ _ o ch c.count as.size _ rm cm _ _
        (fun k j => eLow pt (β + pl.ms.log2) (as.getD k 0 + 2 ^ β * j)) ⟨hr1, hr2⟩
        (fun k j t' => Gs pt s ((c.bound (readAt t lays pt d (as.getD k 0 + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β)
          (ip k)) j).getD t' pt.qc)) ?_]
      · rw [slotW_get pt hs, slotEL, of_zero', zero_add,
          flatMap_sum c.count (fun k j => as.getD k 0 + 2 ^ β * j) _ as.size,
          flatMap_sum c.count (fun k j => as.getD k 0 + 2 ^ β * j) _ as.size, slot_glue]
        refine congrArg (HAdd.hAdd _) (Finset.sum_congr rfl fun k hk => Finset.sum_congr rfl fun j hj => ?_)
        have hk := Finset.mem_range.mp hk
        have hjm : j < 2 ^ pl.ms.log2 := Nat.lt_of_lt_of_le (Finset.mem_range.mp hj) hc'
        have ha := lane_lt (has h6 k hk) hjm
        rw [← lane_pos hpow hal1 hal2 hcs, ← hHc, hB, ← hsp.2.2 h6' Hc _ ha]
        unfold tabSum
        beta_reduce
        ring
      · intro k j x hx
        exact iv_exp t lays pt hR h6a hL2 ip c (low_lt (has h6) k) (by rw [hbd]; exact (hchU x hx).1)

/-- **A summed class's call**: it adds its callee's copies' `refVal`s, each times its copy's `E(a)`. -/
theorem walkCall_sum {t : Tmpl} {lays : Array Lay} {multi : Array Bool} {pt : FoldPt} {s : ℕ}
    (hT : pt.rhoT = Tmpl.halves pt.rho) (hR : pt.rT = Tmpl.halves pt.r) (hlay : LayOk t lays)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    {F : ℕ} (hw : WalkOk t lays multi pt s F) {d β off al i : ℕ} {as : Array ℕ} {o o' : Tab} {st st' : SplitSt}
    (h : (walkCall t lays pt (walkChild lays multi pt (walk t lays multi pt F)) d β as off al
      (eqHalf pt.r (β - 6) al) o i).run st = .ok (o', st'))
    (hst : StOk t lays pt s st) (hU : UOk (t.defs.getD d default).ins 1 o) (hd : d < F + 1)
    (hi : i < (t.defs.getD d default).calls.size) (h6 : 6 ≤ β) (hpre : Pre pt β al as) :
    StOk t lays pt s st' ∧ UOk (t.defs.getD d default).ins 1 o' ∧ ∀ (H a : ℕ), a < 2 ^ β → ∀ ip : List ℕ,
      of (eLow pt β a) * tabSum pt (β + al - 6) H (gLow pt s β a) (Gs pt s pt.qc)
          (fun t' => Gs pt s (ip.getD t' pt.qc)) o' =
        of (eLow pt β a) * tabSum pt (β + al - 6) H (gLow pt s β a) (Gs pt s pt.qc)
          (fun t' => Gs pt s (ip.getD t' pt.qc)) o +
        callVal t lays pt s F d (a + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) ip i := by
  obtain ⟨h6a, hL1, hL2, has⟩ := hpre
  have he : ((t.defs.getD d default).calls.getD i default).def_ < F := by have := hcall d i hi; omega
  have hbd : ∀ j, (((t.defs.getD d default).calls.getD i default).bound id j).length =
      (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).ins := fun j => hlay.bound d i j hi
  have hlan := hlay.lanes d i hi
  have hnk : nkOf β as = 1 := if_pos h6
  unfold walkCall at h
  simp only [h6, decide_true, Bool.true_and, ↓reduceIte] at h
  unfold callVal
  generalize (t.defs.getD d default).calls.getD i default = c at h he hbd hlan ⊢
  generalize (lays.getD d default).places.getD i default = pl at h hlan ⊢
  split at h
  · -- one copy placed directly
    rename_i hm
    have hm1 : pl.ms = 1 := by simpa using hm
    split at h
    · first
      | (rw [sv_throw_bind] at h; cases h)
      | (rw [sv_throw] at h; cases h)
    rename_i hc1
    have hc1' : c.count = 1 := by simpa using hc1
    simp only [hc1', List.range_one, List.foldlM_cons, List.foldlM_nil, bind_pure] at h
    obtain ⟨⟨ch, rm, cm⟩, st3, hc, hr⟩ := sv_bind_ok h
    try dsimp only at hr
    obtain ⟨rfl, rfl⟩ := sv_pure_ok hr
    obtain ⟨hst3, al'', off'', hsp, hH⟩ := walkChild_ok hT hR hcall hw hc hst he ⟨h6a, hL1, hL2, has⟩
    have hchU := hsp.1
    rw [hnk] at hchU
    refine ⟨hst3, bind_U _ _ _ _ _ _ _ _ _ _ _ _ _ _ (fun e he kf hkf => ?_) hU, fun H a ha ip => ?_⟩
    · rw [List.mem_singleton] at hkf
      rw [hkf]
      exact (hchU e.1 (mem_of_toList he)).2
    obtain ⟨Hc, hHc, hr1, hr2⟩ := hH H
    rw [child_sum1 _ pt _ _ _ _ _ _ as _ _ _ _ _ c o ch 0 rm cm ⟨hr1, hr2⟩, mul_add, hc1', Finset.sum_range_one]
    refine congrArg (HAdd.hAdd _) (tabSum_trans_U
      (fun t' => Gs pt s ((c.bound (readAt t lays pt d (a + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) ip) 0).getD t'
        pt.qc)) (fun y hy => iv_sum t lays pt hR h6 hL2 ha ip c (by rw [hbd]; exact (hchU y hy).1)) ?_)
    rw [hsp.2.2 h6 Hc a ha, hHc, hm1, Nat.mul_one]
    congr 1
    ring
  · rename_i hm
    have hm' : pl.ms ≠ 1 := by simpa using hm
    obtain ⟨hcs, hpow, hcnt⟩ := hlan hm'
    have hc' : c.count ≤ 2 ^ pl.ms.log2 := hcnt.trans_eq hpow.symm
    have hB : 2 ^ β * pl.ms = 2 ^ (β + pl.ms.log2) := by rw [Nat.pow_add, hpow]
    split at h
    · first
      | (rw [sv_throw_bind] at h; cases h)
      | (rw [sv_throw] at h; cases h)
    rename_i hal
    have hal1 : (off + pl.nb) % pl.ms = 0 := by
      by_contra hc
      exact hal (by simp [hc])
    have hal2 : pl.ms.log2 ≤ al := by
      by_contra hc
      exact hal (by simp [Nat.lt_of_not_le hc])
    obtain ⟨⟨ch, rm, cm⟩, st3, hc, hr⟩ := sv_bind_ok h
    try dsimp only at hr
    obtain ⟨rfl, rfl⟩ := sv_pure_ok hr
    have h6' : 6 ≤ β + pl.ms.log2 := by omega
    have hpc : Pre pt (β + pl.ms.log2) (al - pl.ms.log2) #[] :=
      ⟨by omega, by omega, by omega, fun h' => absurd h6' h'⟩
    obtain ⟨hst3, al'', off'', hsp, hH⟩ := walkChild_ok hT hR hcall hw hc hst he hpc
    have hchU := hsp.1
    refine ⟨hst3, binds_U _ _ _ _ _ _ _ _ _ _ _ _ _ _ (fun j e he kf hkf => ?_) hU, fun H a ha ip => ?_⟩
    · rw [List.mem_singleton] at hkf
      rw [hkf]
      exact Nat.one_pos
    obtain ⟨Hc, hHc, hr1, hr2⟩ := hH H
    rw [show β + pl.ms.log2 + (al - pl.ms.log2) - 6 = β + al - 6 by omega] at hr1 hr2
    have hLR : of (Tmpl.laneSum (fun b bit => if bit then pt.rho.getD (β - 6 + b) 0 * pt.r.getD (β - 6 + b) 0
        else (pt.rho.getD (β - 6 + b) 0 + 1) * (pt.r.getD (β - 6 + b) 0 + 1)) pl.ms.log2 c.count) =
        ∑ j ∈ Finset.range c.count, of (Tmpl.eqAt pt.rho (β - 6) pl.ms.log2 j) * of (Tmpl.eqAt pt.r (β - 6) pl.ms.log2 j) := by
      rw [laneSum_eq _ _ _ hc']
      exact Finset.sum_congr rfl fun j hj => prod_lane2 pt.rho pt.r (β - 6) _ j (Nat.lt_of_lt_of_le (Finset.mem_range.mp hj) hc')
    have hL : of (Tmpl.laneSum (fun b bit => if bit then pt.rho.getD (β - 6 + b) 0 else pt.rho.getD (β - 6 + b) 0 + 1)
        pl.ms.log2 c.count) = ∑ j ∈ Finset.range c.count, of (Tmpl.eqAt pt.rho (β - 6) pl.ms.log2 j) := by
      rw [laneSum_eq _ _ _ hc']
      exact Finset.sum_congr rfl fun j hj => prod_lane1 pt.rho (β - 6) _ j (Nat.lt_of_lt_of_le (Finset.mem_range.mp hj) hc')
    rw [child_sumL' _ pt _ _ _ _ _ _ as _ _ _ _ _ c o ch c.count rm cm _ _ _ ⟨hr1, hr2⟩
      (fun j t' => Gs pt s ((c.bound (readAt t lays pt d (a + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) ip) j).getD t'
        pt.qc)) fun j x hx => iv_sum t lays pt hR h6 hL2 ha ip c (by rw [hbd]; exact (hchU x hx).1),
      hLR, hL, lane_glue2]
    refine congrArg (HAdd.hAdd _) (Finset.sum_congr rfl fun j hj => ?_)
    have hjm : j < 2 ^ pl.ms.log2 := Nat.lt_of_lt_of_le (Finset.mem_range.mp hj) hc'
    rw [← lane_pos hpow hal1 hal2 hcs, ← hHc, hB, ← hsp.2.2 h6' Hc _ (lane_lt ha hjm), eLow_lane pt h6 ha hjm,
      gLow_lane pt h6 ha hjm]
    unfold tabSum
    beta_reduce
    ring

/-! ## The walk -/

/-- **A walk's step**: every class's walk at depth `F + 1` means its class's, from every walk's at `F`. -/
theorem walk_step {t : Tmpl} {lays : Array Lay} {multi : Array Bool} {pt : FoldPt} {s : ℕ} (hs : s < 64)
    (hT : pt.rhoT = Tmpl.halves pt.rho) (hR : pt.rT = Tmpl.halves pt.r) (hlay : LayOk t lays)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    {F : ℕ} (ih : WalkOk t lays multi pt s F) : WalkOk t lays multi pt s (F + 1) := by
  intro d β as off al st T st' h hst hd hpre
  obtain ⟨h6a, hL1, hL2, has⟩ := hpre
  rw [walk] at h
  simp only [show (!decide (6 ≤ β) && decide (β + al < 6)) = false by simp; omega, Bool.false_eq_true,
    ↓reduceIte] at h
  obtain ⟨eL, st1, hE, h⟩ := sv_bind_ok h
  obtain ⟨gL, st2, hG, h⟩ := sv_bind_ok h
  obtain ⟨rfl, hH1, ht1⟩ := half_ok hE hst.1
  obtain ⟨rfl, hH2, ht2⟩ := half_ok hG hH1
  have hst2 : StOk t lays pt s st2 := ⟨hH2, by rw [ht2, ht1]; exact hst.2⟩
  by_cases h6 : 6 ≤ β
  · simp only [h6, decide_true, ↓reduceIte, Bool.false_eq_true, eq_self_iff_true, if_true] at h
    obtain ⟨hq, hv⟩ := foldlM_sumF
      (fun (x : {x : ℕ × ℕ × List ℕ // x.2.1 < 2 ^ β}) (o : Tab) => of (eLow pt β x.1.2.1) *
        tabSum pt (β + al - 6) x.1.1 (gLow pt s β x.1.2.1) (Gs pt s pt.qc) (fun t' => Gs pt s (x.1.2.2.getD t' pt.qc)) o)
      (fun (o : Tab) (st : SplitSt) => StOk t lays pt s st ∧ UOk (t.defs.getD d default).ins 1 o)
      (fun x i => callVal t lays pt s F d (x.1.2.1 + 2 ^ β * (x.1.1 * 2 ^ al + off)) (2 ^ β) x.1.2.2 i)
      _ h ⟨hst2, rows_U _ _ _ _ _ _ _ _ _ _ _ _ _ _ (uOk_empty _ _)⟩ fun i hi b st b1 st1 hr hq => by
        obtain ⟨q1, q2, q3⟩ := walkCall_sum hT hR hlay hcall ih hr hq.1 hq.2 hd (List.mem_range.1 hi) h6
          ⟨h6a, hL1, hL2, has⟩
        exact ⟨⟨q1, q2⟩, fun x => q3 x.1.1 x.1.2.1 x.2 x.1.2.2⟩
    refine ⟨hq.1, by rw [show nkOf β as = 1 from if_pos h6]; exact hq.2, fun h6' => absurd h6 h6',
      fun _ H a ha ip => ?_⟩
    have hv' := hv ⟨(H, a, ip), ha⟩
    dsimp only at hv'
    rw [hv', list_range_sum, refVal_succ]
    refine congrArg₂ (· + ·) ?_ rfl
    rw [rows_val (fun o => of (eLow pt β a) * tabSum pt (β + al - 6) H (gLow pt s β a) (Gs pt s pt.qc)
        (fun t' => Gs pt s (ip.getD t' pt.qc)) o) (fun o n => rfl)
      (Tab.ref (t.defs.getD d default) pt true (2 ^ al) (2 ^ β) as (eqHalf pt.r (β - 6) al)
        (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y))
      (fun _ cx v => of (eLow pt β a) * (hiE pt (β + al - 6) H cx *
        refS (t.defs.getD d default) pt (β + al - 6) H (2 ^ al) (eqHalf pt.r (β - 6) al)
          (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) (gLow pt s β a) (Gs pt s pt.qc)
          (fun t' => Gs pt s (ip.getD t' pt.qc)) v))
      (fun o k cx w v => by rw [ref_tabSum]; ring), tabSum_empty, mul_zero, zero_add]
    refine congrArg List.sum (List.map_congr_left fun r _ => ?_)
    rw [Finset.sum_range_one]
    unfold rowVal grpVal
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    have hP := rowLow_e_sum (as := as) (k := 0) pt hT h6 hL1 ha H (off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)
    rw [show a + 2 ^ β * (H * 2 ^ al + off) + 2 ^ β * Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1 =
      a + 2 ^ β * (H * 2 ^ al + (off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)) by ring]
    refine congrArg₂ (· + ·) (congrArg List.sum (List.map_congr_left fun v _ => ?_))
      (congrArg List.sum (List.map_congr_left fun v _ => ?_))
    · rw [refS_eq t lays pt hR h6 hL2 ha]
      simp only [of_mul]
      rw [← hP]
      ring
    · rw [refS_eq t lays pt hR h6 hL2 ha]
      simp only [of_mul, ← one_eq]
      rw [← hP]
      ring
  · simp only [h6, decide_false, ↓reduceIte, Bool.false_eq_true, eq_self_iff_true, if_true] at h
    obtain ⟨hq, hv⟩ := foldlM_sumF
      (fun (x : ℕ × (ℕ → List ℕ)) (o : Tab) =>
        tabExp pt s (β + al - 6) x.1 (fun t' k => Gs pt s ((x.2 k).getD t' pt.qc)) o)
      (fun (o : Tab) (st : SplitSt) => StOk t lays pt s st ∧ UOk (t.defs.getD d default).ins as.size o)
      (fun x i => ∑ k ∈ Finset.range as.size,
        callVal t lays pt s F d (as.getD k 0 + 2 ^ β * (x.1 * 2 ^ al + off)) (2 ^ β) (x.2 k) i)
      _ h ⟨hst2, rows_U _ _ _ _ _ _ _ _ _ _ _ _ _ _ (uOk_empty _ _)⟩ fun i hi b st b1 st1 hr hq => by
        obtain ⟨q1, q2, q3⟩ := walkCall_exp hs hT hR hlay hcall ih hr hq.1 hq.2 hd (List.mem_range.1 hi) h6
          ⟨h6a, hL1, hL2, has⟩
        exact ⟨⟨q1, q2⟩, fun x => q3 x.1 x.2⟩
    refine ⟨hq.1, by rw [show nkOf β as = as.size from if_neg h6]; exact hq.2, fun _ H ip => ?_,
      fun h6' => absurd h6' h6⟩
    have hv' := hv (H, ip)
    dsimp only at hv'
    rw [hv', list_range_sum, Finset.sum_comm]
    simp only [refVal_succ]
    rw [Finset.sum_add_distrib]
    refine congrArg₂ (· + ·) ?_ rfl
    rw [rows_val (tabExp pt s (β + al - 6) H (fun t' k => Gs pt s ((ip k).getD t' pt.qc))) (fun o n => rfl)
      (Tab.ref (t.defs.getD d default) pt false (2 ^ al) (2 ^ β) as (eqHalf pt.r 0 (β + al - 6))
        (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y))
      (fun k cx v => hiE pt (β + al - 6) H cx *
        refX (t.defs.getD d default) pt s (β + al - 6) H (2 ^ al) (2 ^ β) as (eqHalf pt.r 0 (β + al - 6))
          (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) k v)
      (fun o k cx w v => by rw [ref_tabExp, mul_assoc]), tabExp_empty, zero_add, list_sum_comm]
    refine Finset.sum_congr rfl fun k hk => congrArg List.sum (List.map_congr_left fun r _ => ?_)
    have hk' := has h6 k (Finset.mem_range.mp hk)
    unfold rowVal grpVal
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    have hP := rowLow_e (k := k) pt hT h6a hL1 hk' H (off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)
    rw [show as.getD k 0 + 2 ^ β * (H * 2 ^ al + off) + 2 ^ β * Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1 =
      as.getD k 0 + 2 ^ β * (H * 2 ^ al + (off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)) by ring]
    refine congrArg₂ (· + ·) (congrArg List.sum (List.map_congr_left fun v _ => ?_))
      (congrArg List.sum (List.map_congr_left fun v _ => ?_))
    · rw [refX_eq t lays pt hR h6a hL2 hk']
      simp only [of_mul]
      rw [← hP]
      ring
    · rw [refX_eq t lays pt hR h6a hL2 hk']
      simp only [of_mul, ← one_eq]
      rw [← hP]
      ring

/-- **Every walk means its class's**, at every depth. -/
theorem walk_ok {t : Tmpl} {lays : Array Lay} {multi : Array Bool} {pt : FoldPt} {s : ℕ} (hs : s < 64)
    (hT : pt.rhoT = Tmpl.halves pt.rho) (hR : pt.rT = Tmpl.halves pt.r) (hlay : LayOk t lays)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d) :
    ∀ F, WalkOk t lays multi pt s F
  | 0 => fun _ _ _ _ _ _ _ _ _ _ hd _ => absurd hd (Nat.not_lt_zero _)
  | F + 1 => walk_step hs hT hR hlay hcall (walk_ok hs hT hR hlay hcall F)

end FlockSoundness.TmplFold
