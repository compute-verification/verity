import Proofs.Flock.VBridge.Algebra.Build
import Proofs.Flock.VBridge.Algebra.ToVStar

/-!
# The S `structureOf` builds has its `wt` bits below 128

A `wt` term is `atomTerm`'s for a ring switch's transpose `rsT k u` (RecAlgebra.lean:677–679), and `symbolic`'s
transposes have `u < 128` (RecAlgebra.lean:627). `run` makes no element: renaming the elements is a `Hom`, so `run`'s
residuals at `symbolic`, renamed by `bump` (which fixes exactly the elements with `u < 128`), are `run`'s at the renamed
`symbolic`, which is `symbolic` itself. Their elements are fixed by `bump` (`Lin.Good`), and `build` makes its
operands from them and its added terms of the running sums and v (`structureOf_bitsOk`).
-/

namespace Flock.RecAlgebra

open Flock

/-! ## Renaming the elements -/

/-- An element `run` can read: a transpose's bit is below 128. -/
def Key.Good : Key → Prop
  | .rsT _ u => u < 128
  | _ => True

/-- Moves the transposes' bits from 128 up by one, fixing exactly the good elements. -/
def bump : Key → Key
  | .rsT k u => if u < 128 then .rsT k u else .rsT k (u + 1)
  | k => k

def unbump : Key → Key
  | .rsT k u => if u < 128 then .rsT k u else .rsT k (u - 1)
  | k => k

theorem unbump_bump (k : Key) : unbump (bump k) = k := by
  cases k
  case rsT k u =>
    by_cases hu : u < 128
    · simp [bump, unbump, hu]
    · simp [bump, unbump, hu, show ¬u + 1 < 128 by omega]
  all_goals rfl

theorem bump_inj : Function.Injective bump := Function.LeftInverse.injective unbump_bump

theorem good_of_bump {k : Key} (h : bump k = k) : k.Good := by
  cases k
  case rsT k u =>
    show u < 128
    by_cases hu : u < 128
    · exact hu
    · simp only [bump, hu, ↓reduceIte, Key.rsT.injEq, true_and] at h
      omega
  all_goals trivial

/-- A dict's keys renamed. -/
def mapK {κ κ' : Type} (G : κ → κ') (l : Terms κ) : Terms κ' := l.map fun kx => (G kx.1, kx.2)

theorem mapK_addAt {κ κ' : Type} [DecidableEq κ] [DecidableEq κ'] {G : κ → κ'} (hG : Function.Injective G)
    (k : κ) (v : F128) : ∀ l : Terms κ, mapK G (addAt l k v) = addAt (mapK G l) (G k) v
  | [] => rfl
  | (k', x) :: r => by
    show mapK G (addAt ((k', x) :: r) k v) = addAt ((G k', x) :: mapK G r) (G k) v
    by_cases h : k' = k
    · subst h
      simp [addAt, mapK]
    · have e1 : addAt ((k', x) :: r) k v = (k', x) :: addAt r k v := by simp only [addAt, h, ↓reduceIte]
      have e2 : addAt ((G k', x) :: mapK G r) (G k) v = (G k', x) :: addAt (mapK G r) (G k) v := by
        simp only [addAt, hG.ne h, ↓reduceIte]
      rw [e1, e2]
      show (G k', x) :: mapK G (addAt r k v) = _
      rw [mapK_addAt hG k v r]

theorem mapK_mergeT {κ κ' : Type} [DecidableEq κ] [DecidableEq κ'] {G : κ → κ'} (hG : Function.Injective G) :
    ∀ b a : Terms κ, mapK G (mergeT a b) = mergeT (mapK G a) (mapK G b)
  | [], _ => rfl
  | (k, x) :: b, a => by
    show mapK G (mergeT (addAt a k x) b) = mergeT (addAt (mapK G a) (G k) x) (mapK G b)
    rw [mapK_mergeT hG b (addAt a k x), mapK_addAt hG]

theorem pair_inj {g : Key → Key} (hg : Function.Injective g) :
    Function.Injective fun ab : Key × Key => (g ab.1, g ab.2) := by
  intro ⟨a, b⟩ ⟨c, d⟩ h
  simp only [Prod.mk.injEq] at h ⊢
  exact ⟨hg h.1, hg h.2⟩

/-- A symbolic value with its elements renamed. -/
def Lin.ren (g : Key → Key) (f : Lin) : Lin := ⟨mapK g f.t, f.c, mapK (fun ab => (g ab.1, g ab.2)) f.q⟩

theorem ren_var (g : Key → Key) (k : Key) : Lin.ren g (Lin.var k) = Lin.var (g k) := rfl

/-- An injective renaming commutes with `run`'s operations. -/
theorem ren_hom {g : Key → Key} (hg : Function.Injective g) : Hom (Lin.ren g) where
  zero := rfl
  add := fun a b => by
    show Lin.ren g (Lin.xor a b) = Lin.xor (Lin.ren g a) (Lin.ren g b)
    simp only [Lin.ren, Lin.xor, mapK_mergeT hg, mapK_mergeT (pair_inj hg)]
  sm := fun c a => by
    show Lin.ren g (Lin.scale c a) = Lin.scale c (Lin.ren g a)
    simp only [Lin.ren, Lin.scale, mapK, List.map_map, Function.comp_def]
  mv := fun a b z h => by
    change Lin.mv a b = .ok z at h
    change Lin.mv (Lin.ren g a) (Lin.ren g b) = .ok (Lin.ren g z)
    unfold Lin.mv at h
    split at h
    · split at h
      · rename_i hxy
        obtain ⟨rfl, rfl⟩ := hxy
        cases h
        simp [Lin.ren, mapK, Lin.mv]
      · cases h
    · cases h
  cst := fun _ => rfl

/-! ## `run` at `symbolic` makes no element -/

theorem lmap_fixed {α : Type} {F : α → α} : ∀ {l : List α}, (∀ x ∈ l, F x = x) → l.map F = l
  | [], _ => rfl
  | a :: l, h => by
    rw [List.map_cons, h a (by simp), lmap_fixed fun x hx => h x (by simp [hx])]

theorem amap_fixed {α : Type} {F : α → α} {xs : Array α} (h : ∀ x ∈ xs, F x = x) : xs.map F = xs := by
  apply Array.toList_inj.mp
  rw [Array.toList_map]
  exact lmap_fixed fun x hx => h x (by simpa using hx)

theorem fixed_of_map {α : Type} {F : α → α} : ∀ {l : List α}, l.map F = l → ∀ x ∈ l, F x = x
  | [], _, _, hx => by cases hx
  | a :: l, h, x, hx => by
    simp only [List.map_cons, List.cons.injEq] at h
    rcases List.mem_cons.mp hx with rfl | hx
    · exact h.1
    · exact fixed_of_map h.2 x hx

theorem ren_vars {n : Nat} {K : Nat → Key} (hK : ∀ i < n, bump (K i) = K i) :
    ((Array.range n).map (Lin.var ∘ K)).map (Lin.ren bump) = (Array.range n).map (Lin.var ∘ K) := by
  apply amap_fixed
  intro x hx
  obtain ⟨i, hi, rfl⟩ := Array.mem_map.mp hx
  rw [Array.mem_range] at hi
  show Lin.ren bump (Lin.var (K i)) = Lin.var (K i)
  rw [ren_var, hK i hi]

/-- `symbolic`'s elements are good: `bump` fixes them (RecAlgebra.lean:615–634). -/
theorem symbolic_ren (sh : Shape) : (symbolic sh).map (Lin.ren bump) = symbolic sh := by
  simp only [symbolic, Messages.map, Messages.mk.injEq]
  and_intros
  all_goals first
    | exact ren_vars fun _ _ => rfl
    | (funext i ea; exact ren_vars fun _ _ => rfl)
    | (apply amap_fixed
       intro x hx
       obtain ⟨k, -, rfl⟩ := Array.mem_map.mp hx
       first
         | exact ren_vars fun _ _ => rfl
         | (refine ren_vars fun u hu => ?_; simp [bump, hu])
         | rfl)
    | rfl

theorem residualLins_ren {sh : Shape} {coins : CoinSource} {extras : Array Claim} {lins : Array Lin} {info : Info Lin}
    (h : residualLins sh coins extras = .ok (lins, info)) : lins.map (Lin.ren bump) = lins := by
  cases hr : run sh (symbolic sh) extras 0 coins with
  | error e =>
    have hl : residualLins sh coins extras = .error e := by
      show Prod.fst <$> run sh (symbolic sh) extras 0 coins = _
      rw [hr]; rfl
    rw [hl] at h; cases h
  | ok q =>
    obtain ⟨⟨lins', info'⟩, k⟩ := q
    have hl : residualLins sh coins extras = .ok (lins', info') := by
      show Prod.fst <$> run sh (symbolic sh) extras 0 coins = _
      rw [hr]; rfl
    rw [hl] at h
    cases h
    have hs := run_sim (ren_hom bump_inj) sh (symbolic sh) extras 0 coins _ _ hr
    rw [symbolic_ren, hr] at hs
    simp only [Except.ok.injEq, Prod.mk.injEq] at hs
    exact hs.1.1.symm

/-- A symbolic value whose elements are good. -/
def Lin.Good (f : Lin) : Prop := (∀ kx ∈ f.t, kx.1.Good) ∧ ∀ abx ∈ f.q, abx.1.1.Good ∧ abx.1.2.Good

theorem good_of_ren {f : Lin} (h : Lin.ren bump f = f) : f.Good := by
  have ht : mapK bump f.t = f.t := congrArg Lin.t h
  have hq : mapK (fun ab : Key × Key => (bump ab.1, bump ab.2)) f.q = f.q := congrArg Lin.q h
  refine ⟨fun kx hk => good_of_bump ?_, fun abx hab => ⟨good_of_bump ?_, good_of_bump ?_⟩⟩
  · exact congrArg Prod.fst (fixed_of_map ht kx hk)
  · exact congrArg (fun p => p.1.1) (fixed_of_map hq abx hab)
  · exact congrArg (fun p => p.1.2) (fixed_of_map hq abx hab)

/-! ## `build` keeps the bits -/

/-- Every operand's `wt` bits are below 128. -/
def OpsBits (ops : Array (Array Term)) : Prop := ∀ o ∈ ops, ∀ t ∈ o, t.BitOk

theorem OpsBits.push {ops : Array (Array Term)} {op : Array Term} (h : OpsBits ops) (hop : ∀ t ∈ op, t.BitOk) :
    OpsBits (ops.push op) := by
  intro o ho t ht
  rcases Array.mem_push.mp ho with ho | rfl
  · exact h o ho t ht
  · exact hop t ht

theorem single_bitOk {t : Term} (h : t.BitOk) : ∀ x ∈ (#[t] : Array Term), x.BitOk := by
  intro x hx
  simp at hx
  subst hx
  exact h

section Build

variable {sh : Shape} {comb0 : Nat}

theorem atom_bitOk {k : Key} {t : Term} (ha : atomTerm sh comb0 k = .ok t) (hk : k.Good) : t.BitOk := by
  cases k <;> simp only [atomTerm] at ha <;> (try split at ha) <;> (try cases ha) <;> exact hk

theorem atomOp_bitOk {k : Key} {op : Array Term} (ha : atomOp sh comb0 k = .ok op) (hk : k.Good) :
    ∀ x ∈ op, x.BitOk := by
  obtain ⟨t, ht, rfl | ⟨-, i, rfl⟩⟩ := atomOp_cases ha
  · exact single_bitOk (atom_bitOk ht hk)
  · intro x hx
    simp at hx
    rcases hx with rfl | rfl
    · exact atom_bitOk ht hk
    · show True
      trivial

theorem varOp_bits {k : Key} {st st' : BSt} {o : Nat} (hr : varOp sh comb0 k st = .ok (o, st'))
    (hi : OpsBits st.ops) (hk : k.Good) : OpsBits st'.ops := by
  unfold varOp at hr
  obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
  cases hg
  cases hl : st.opOf.lookup k with
  | some i =>
    simp only [hl] at hr
    cases hr
    exact hi
  | none =>
    simp only [hl] at hr
    obtain ⟨t', st2, ht, hr⟩ := bm_bind hr
    obtain ⟨ha, h2⟩ := liftB_ok ht
    subst st2
    obtain ⟨_, st3, hs, hr⟩ := bm_bind hr
    cases hs
    cases hr
    exact hi.push (atomOp_bitOk ha hk)

theorem coefOps_bits (ri : Nat) : ∀ (l : List (Key × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    coefOps sh comb0 ri l st = .ok (ps, st') → OpsBits st.ops → (∀ kx ∈ l, kx.1.Good) → OpsBits st'.ops
  | [], st, st', ps, hr, hi, _ => by
    cases hr
    exact hi
  | (k, x) :: rest, st, st', ps, hr, hi, hl => by
    obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    obtain ⟨o, st3, hv, hr⟩ := bm_bind hr
    obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
    cases hr
    have hA : OpsBits (st.ops.push #[Term.v st.vNames.size]) := hi.push (single_bitOk trivial)
    exact coefOps_bits ri rest st3 _ ps' hc (varOp_bits hv hA (hl (k, x) (by simp)))
      fun kx hkx => hl kx (by simp [hkx])

theorem prodOps_bits (ri : Nat) : ∀ (l : List ((Key × Key) × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    prodOps sh comb0 ri l st = .ok (ps, st') → OpsBits st.ops → (∀ abx ∈ l, abx.1.1.Good ∧ abx.1.2.Good) →
      OpsBits st'.ops
  | [], st, st', ps, hr, hi, _ => by
    cases hr
    exact hi
  | ((a, b), x) :: rest, st, st', ps, hr, hi, hl => by
    simp only [prodOps] at hr
    split at hr
    · obtain ⟨_, _, h1, _⟩ := bm_bind hr
      cases h1
    · obtain ⟨oa, st2, ha, hr⟩ := bm_bind hr
      obtain ⟨ob, st3, hb, hr⟩ := bm_bind hr
      obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
      cases hr
      have hab := hl ((a, b), x) (by simp)
      exact prodOps_bits ri rest st3 _ ps' hc (varOp_bits hb (varOp_bits ha hi hab.1) hab.2)
        fun abx h => hl abx (by simp [h])

theorem accTerms_bits {fin ri : Nat} {l : List (Key × F128)} {st st' : BSt} {acc : Array Term}
    (hr : accTerms sh fin ri l st = .ok (acc, st')) : st'.ops = st.ops ∧ ∀ t ∈ acc, t.BitOk := by
  refine ⟨(accTerms_spec (sh := sh) #[] #[] (fun _ _ => 0) hr).1, ?_⟩
  unfold accTerms at hr
  split at hr
  · cases hr
    intro t ht
    simp at ht
  · split at hr
    · cases hr
    · obtain ⟨_, st1, _, hr⟩ := bm_bind hr
      cases h1 : wi sh (.acc1 (ri - fin)) with
      | none => simp only [h1] at hr; cases hr
      | some i1 =>
        cases h0 : wi sh (.acc0 (ri - fin)) with
        | none => simp only [h1, h0] at hr; cases hr
        | some i0 =>
          simp only [h1, h0] at hr
          cases hr
          intro t ht
          simp at ht
          rcases ht with rfl | rfl <;> trivial

theorem constTerm_bits {ri : Nat} {c : Option F128} {st st' : BSt} {cp : Array Term}
    (hr : constTerm ri c st = .ok (cp, st')) : st'.ops = st.ops ∧ ∀ t ∈ cp, t.BitOk := by
  cases c with
  | none =>
    cases hr
    exact ⟨rfl, fun t ht => by simp at ht⟩
  | some c =>
    unfold constTerm at hr
    obtain ⟨_, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    cases hr
    exact ⟨rfl, single_bitOk trivial⟩

theorem buildRes_bits {fin ri : Nat} {f : Lin} {st st' : BSt} {res : Res}
    (hr : buildRes sh fin comb0 ri f st = .ok (res, st')) (hi : OpsBits st.ops) (hf : f.Good) :
    OpsBits st'.ops ∧ ∀ t ∈ res.add, t.BitOk := by
  unfold buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, abits⟩ := accTerms_bits h1
  obtain ⟨l1, st1', hs1, hr⟩ := bm_bind hr
  obtain ⟨hl1, h⟩ := liftB_ok hs1
  subst st1'
  obtain ⟨cm, st2, h2, hr⟩ := bm_bind hr
  obtain ⟨l2, st2', hs2, hr⟩ := bm_bind hr
  obtain ⟨hl2, h⟩ := liftB_ok hs2
  subst st2'
  obtain ⟨pm, st3, h3, hr⟩ := bm_bind hr
  obtain ⟨cp, st4, h4, hr⟩ := bm_bind hr
  cases hr
  obtain ⟨c4, cbits⟩ := constTerm_bits h4
  have hi1 : OpsBits st1.ops := by rw [a1]; exact hi
  have hg1 : ∀ kx ∈ l1, kx.1.Good := fun kx hk => hf.1 kx (List.mem_of_mem_filter ((sortBy_perm hl1).subset hk))
  have hg2 : ∀ abx ∈ l2, abx.1.1.Good ∧ abx.1.2.Good := fun abx hk => hf.2 abx ((sortBy_perm hl2).subset hk)
  have hi3 := prodOps_bits ri l2 st2 st3 pm h3 (coefOps_bits ri l1 st1 st2 cm h2 hi1 hg1) hg2
  refine ⟨by rw [c4]; exact hi3, fun t ht => ?_⟩
  change t ∈ acc ++ cp at ht
  rcases Array.mem_append.mp ht with ht | ht
  · exact abits t ht
  · exact cbits t ht

theorem buildAll_bits {fin : Nat} : ∀ (fs : List Lin) (ri : Nat) (st st' : BSt) (rs : Array Res),
    buildAll sh fin comb0 fs ri st = .ok (rs, st') → OpsBits st.ops → (∀ f ∈ fs, f.Good) →
      OpsBits st'.ops ∧ ∀ r ∈ rs, ∀ t ∈ r.add, t.BitOk
  | [], ri, st, st', rs, hr, hi, _ => by
    cases hr
    exact ⟨hi, fun r hr => by simp at hr⟩
  | f :: rest, ri, st, st', rs, hr, hi, hf => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    obtain ⟨h1i, h1a⟩ := buildRes_bits h1 hi (hf f (by simp))
    obtain ⟨h2i, h2a⟩ := buildAll_bits rest (ri + 1) st1 _ rs' h2 h1i fun g hg => hf g (by simp [hg])
    refine ⟨h2i, fun r' hr' => ?_⟩
    rcases Array.mem_append.mp hr' with hr' | hr'
    · simp at hr'
      subst hr'
      exact h1a
    · exact h2a r' hr'

end Build

theorem build_bitsOk {sh : Shape} {lins : Array Lin} {b : Built} (hb : build sh lins = .ok b)
    (hl : ∀ f ∈ lins.toList, f.Good) : b.S.BitsOk := by
  unfold build at hb
  generalize hK : sh.sched.levels.map (fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) = K
    at hb
  generalize hc0 : (lins.toList.map slotsOf).sum = comb0 at hb
  simp only at hb
  cases hB : (buildAll sh (2 + sh.claims) comb0 lins.toList 0).run { kappa := K } with
  | error e => rw [hB] at hb; cases hb
  | ok p =>
    obtain ⟨res, st⟩ := p
    rw [hB] at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · cases hb
    · cases hb
      have hB' : buildAll sh (2 + sh.claims) comb0 lins.toList 0 { kappa := K } = .ok (res, st) := hB
      exact buildAll_bits lins.toList 0 _ _ _ hB' (fun _ h => by simp at h) hl

/-- The S the verifier's `GfResiduals` reads (`structureOf`, RecAlgebra.lean:818–820) has every `wt` bit below 128,
so `residuals_toV` reads it as V*'s. -/
theorem structureOf_bitsOk {sh : Shape} {S : GfS} (h : structureOf sh = .ok S) : S.BitsOk := by
  unfold structureOf at h
  cases hl : residualLins sh CoinSource.zeros (extrasLike sh) with
  | error e => rw [hl] at h; cases h
  | ok p =>
    obtain ⟨lins, info⟩ := p
    rw [hl] at h
    simp only [bind, Except.bind] at h
    cases hb : build sh lins with
    | error e => rw [hb] at h; cases h
    | ok b =>
      rw [hb] at h
      (try simp only at h)
      cases h
      have hfix : lins.toList.map (Lin.ren bump) = lins.toList := by
        rw [← Array.toList_map, residualLins_ren hl]
      exact build_bitsOk hb fun f hf => good_of_ren (fixed_of_map hfix f hf)

end Flock.RecAlgebra
