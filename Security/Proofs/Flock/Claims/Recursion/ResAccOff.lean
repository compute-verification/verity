import Proofs.Flock.VBridge.Algebra.BitsOk
import Proofs.Flock.VBridge.Algebra.RingSwitch
import Proofs.Flock.VBridge.PartS
import Proofs.Flock.VBridge.Sparse.Gadgets
import Proofs.Flock.Claims.Recursion.VerdictRowEDeployed
import Proofs.Flock.Claims.Recursion.VerdictPieces

/-!
# The residuals don't see a shift of the running sums

At a plain shape, the verifier's algebra (`RecAlgebra.structureOf`) reads w's running sums (`acc0`, `acc1`, past the
rounds' `A` elements) only as `acc1 c + acc0 c` in a residual's added terms, and its operands read only the rounds'
elements. So a part's residuals (`RecResiduals.partS`) vanish at elements that a shift `δ c` of both of rep's running sums
`c` moves, wherever they vanish at the unshifted ones: in characteristic 2 the shift cancels.

* `structureOf_accLow`: `structureOf`'s S at a plain shape. `run` makes no element, so the symbolic residuals hold no
  running sum and only the claims' transposes (`residualLins_bumpA`, the renaming argument of `structureOf_bitsOk`);
  `build` makes its operands from those (`atomTerm`: w's elements below `A`, `wi_low`; a transpose's 128 elements,
  `wi_rs`) and adds a residual's running sums as `w[A + 2 + c]`, `w[A + c]` (`wi_acc1`, `wi_acc0`).
* `offOf A wI δ e`: the shift at the part's element `e`: `δ ((j - A) % 2)` where the part's w index `wI` places `e` at
  w's element `j ≥ A`, else 0.
* `resZero_off`: a part's residuals vanish at `E` wherever they vanish at elements equal to `E` plus `offOf` on the
  part's w.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-! ## `run` makes no running sum and no transpose of a claim the shape lacks -/

/-- An element the symbolic messages hold: no running sum, and a transpose only of one of the `n` claims. -/
def Key.Sym (n : Nat) : Key → Prop
  | .acc0 _ => False
  | .acc1 _ => False
  | .rsT k _ => k < n
  | _ => True

/-- Moves the running sums and the transposes of claims past `n`, fixing exactly the elements `Key.Sym n` admits. -/
def bumpA (n : Nat) : Key → Key
  | .acc0 r => .acc0 (r + 1)
  | .acc1 r => .acc1 (r + 1)
  | .rsT k u => if k < n then .rsT k u else .rsT (k + 1) u
  | k => k

def unbumpA (n : Nat) : Key → Key
  | .acc0 r => .acc0 (r - 1)
  | .acc1 r => .acc1 (r - 1)
  | .rsT k u => if k < n then .rsT k u else .rsT (k - 1) u
  | k => k

theorem unbumpA_bumpA (n : Nat) (k : Key) : unbumpA n (bumpA n k) = k := by
  cases k
  case acc0 r => show Key.acc0 (r + 1 - 1) = _; rw [Nat.add_sub_cancel]
  case acc1 r => show Key.acc1 (r + 1 - 1) = _; rw [Nat.add_sub_cancel]
  case rsT k u =>
    by_cases hk : k < n
    · simp [bumpA, unbumpA, hk]
    · simp [bumpA, unbumpA, hk, show ¬k + 1 < n by omega]
  all_goals rfl

theorem bumpA_inj (n : Nat) : Function.Injective (bumpA n) := Function.LeftInverse.injective (unbumpA_bumpA n)

theorem sym_of_bumpA {n : Nat} {k : Key} (h : bumpA n k = k) : k.Sym n := by
  cases k
  case acc0 r => simp [bumpA] at h
  case acc1 r => simp [bumpA] at h
  case rsT k u =>
    show k < n
    by_cases hk : k < n
    · exact hk
    · simp [bumpA, hk] at h
  all_goals trivial

theorem ren_varsG {g : Key → Key} {n : Nat} {K : Nat → Key} (hK : ∀ i < n, g (K i) = K i) :
    ((Array.range n).map (Lin.var ∘ K)).map (Lin.ren g) = (Array.range n).map (Lin.var ∘ K) := by
  apply amap_fixed
  intro x hx
  obtain ⟨i, hi, rfl⟩ := Array.mem_map.mp hx
  rw [Array.mem_range] at hi
  show Lin.ren g (Lin.var (K i)) = Lin.var (K i)
  rw [ren_var, hK i hi]

/-- `symbolic`'s elements are fixed by `bumpA` at its claims (RecAlgebra.lean:621–641). -/
theorem symbolic_bumpA (sh : Shape) : (symbolic sh).map (Lin.ren (bumpA sh.claims)) = symbolic sh := by
  simp only [symbolic, Messages.map, Messages.mk.injEq]
  and_intros
  all_goals first
    | exact ren_varsG fun _ _ => rfl
    | (funext i ea; exact ren_varsG fun _ _ => rfl)
    | (apply amap_fixed
       intro x hx
       obtain ⟨k, hk, rfl⟩ := Array.mem_map.mp hx
       first
         | exact ren_varsG fun _ _ => rfl
         | (refine ren_varsG fun u hu => ?_; simp [bumpA, Array.mem_range.mp hk])
         | rfl)
    | rfl

theorem residualLins_bumpA {sh : Shape} {coins : Coins} {extras : Array Claim} {lins : Array Lin}
    {info : Info Lin} (h : residualLins sh coins extras = .ok (lins, info)) :
    lins.map (Lin.ren (bumpA sh.claims)) = lins := by
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
    have hs := run_sim (ren_hom (bumpA_inj sh.claims)) sh (symbolic sh) extras 0 coins _ _ hr
    rw [symbolic_bumpA, hr] at hs
    simp only [Except.ok.injEq, Prod.mk.injEq] at hs
    exact hs.1.1.symm

/-- A symbolic value of elements the symbolic messages hold. -/
def Lin.Sym (n : Nat) (f : Lin) : Prop := (∀ kx ∈ f.t, kx.1.Sym n) ∧ ∀ abx ∈ f.q, abx.1.1.Sym n ∧ abx.1.2.Sym n

theorem sym_of_ren {n : Nat} {f : Lin} (h : Lin.ren (bumpA n) f = f) : f.Sym n := by
  have ht : mapK (bumpA n) f.t = f.t := congrArg Lin.t h
  have hq : mapK (fun ab : Key × Key => (bumpA n ab.1, bumpA n ab.2)) f.q = f.q := congrArg Lin.q h
  refine ⟨fun kx hk => sym_of_bumpA ?_, fun abx hab => ⟨sym_of_bumpA ?_, sym_of_bumpA ?_⟩⟩
  · exact congrArg Prod.fst (fixed_of_map ht kx hk)
  · exact congrArg (fun p => p.1.1) (fixed_of_map hq abx hab)
  · exact congrArg (fun p => p.1.2) (fixed_of_map hq abx hab)

/-! ## Where `build`'s terms read w -/

/-- The number of the rounds' elements: w's running sums start here. -/
abbrev nR (sh : Shape) : Nat := (rounds sh).flatten.size

/-- A term reading only the rounds' elements of w (below `A`), or v. -/
def Term.Low (A : Nat) : Term → Prop
  | .w i => i < A
  | .v _ => True
  | .wt i _ => i + 128 ≤ A

/-- A residual's added terms: a rep's running sums after and before its openings, `w[A + 2 + c] + w[A + c]`, or none,
then terms reading below `A`. -/
def AddLow (A : Nat) (add : Array Term) : Prop :=
  ∃ acc cp : Array Term, add = acc ++ cp ∧ (acc = #[] ∨ ∃ c < 2, acc = #[.w (A + (2 + c)), .w (A + c)]) ∧
    ∀ t ∈ cp, t.Low A

/-- **S reads the running sums only as their pairs**: every operand reads below `A`, every residual's added terms are
`AddLow`. -/
def GfS.AccLow (A : Nat) (S : GfS) : Prop := (∀ o ∈ S.ops, ∀ t ∈ o, t.Low A) ∧ ∀ r ∈ S.res, AddLow A r.add

theorem getElem?_of_idxOf?_key {xs : Array Key} {a : Key} {i : Nat} (h : xs.idxOf? a = some i) : xs[i]? = some a := by
  rcases xs with ⟨l⟩
  rw [List.idxOf?_toArray, List.idxOf?_eq_some_iff] at h
  obtain ⟨hi, he, -⟩ := h
  simp only [List.getElem?_toArray]
  exact List.getElem?_eq_some_iff.2 ⟨hi, he⟩

section Build

variable {sh : Shape} {comb0 : Nat}

theorem wi_low {n : Nat} {k : Key} {i : Nat} (h : wi sh k = some i) (hk : k.Sym n) : i < nR sh := by
  have he := getElem?_of_idxOf?_key h
  unfold wNames at he
  by_contra hlt
  unfold nR at hlt
  rw [Array.getElem?_append_right (by omega)] at he
  obtain ⟨p, hp⟩ := Array.getElem?_eq_some_iff.1 he
  have hm : k ∈ accNames := hp ▸ Array.getElem_mem p
  simp [accNames] at hm
  rcases hm with rfl | rfl | rfl | rfl <;> exact hk

theorem wi_rs0_low {k : Nat} (hk : k < sh.claims) {i : Nat} (h : wi sh (.rs k 0) = some i) : i + 128 ≤ nR sh := by
  rw [wi_rs sh hk (by norm_num)] at h
  have hi := (Option.some.inj h).symm
  subst hi
  unfold nR
  rw [rounds_split, Array.flatten_append, Array.flatten_append]
  have hB : (((Array.range sh.claims).map fun k => (Array.range 128).map (Key.rs k)).flatten) =
      blocks Key.rs sh.claims := rfl
  rw [hB]
  simp only [Array.size_append, (blocks_spec Key.rs sh.claims).1]
  omega

theorem atom_low {k : Key} {t : Term} (ha : atomTerm sh comb0 k = .ok t) (hk : k.Sym sh.claims) :
    t.Low (nR sh) := by
  cases k <;> simp only [atomTerm] at ha <;> (try split at ha) <;> (try cases ha)
  all_goals first
    | trivial
    | exact wi_low ‹_› hk
    | exact wi_rs0_low (k := _) hk ‹_›

theorem atomOp_low (hh : sh.holo = false) {k : Key} {op : Array Term} (ha : atomOp sh comb0 k = .ok op)
    (hk : k.Sym sh.claims) : ∀ t ∈ op, t.Low (nR sh) := by
  obtain ⟨t, ht, rfl⟩ := atomOp_plain hh ha
  intro x hx
  simp at hx
  subst hx
  exact atom_low ht hk

/-- Every operand reads below `A`. -/
def OpsLow (A : Nat) (ops : Array (Array Term)) : Prop := ∀ o ∈ ops, ∀ t ∈ o, t.Low A

theorem OpsLow.push {A : Nat} {ops : Array (Array Term)} {op : Array Term} (h : OpsLow A ops)
    (hop : ∀ t ∈ op, t.Low A) : OpsLow A (ops.push op) := by
  intro o ho t ht
  rcases Array.mem_push.mp ho with ho | rfl
  · exact h o ho t ht
  · exact hop t ht

theorem single_v_low (A n : Nat) : ∀ x ∈ (#[Term.v n] : Array Term), x.Low A := by
  intro x hx
  simp at hx
  subst hx
  trivial

theorem varOp_low (hh : sh.holo = false) {k : Key} {st st' : BSt} {o : Nat}
    (hr : varOp sh comb0 k st = .ok (o, st')) (hi : OpsLow (nR sh) st.ops) (hk : k.Sym sh.claims) :
    OpsLow (nR sh) st'.ops := by
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
    exact hi.push (atomOp_low hh ha hk)

theorem coefOps_low (hh : sh.holo = false) (ri : Nat) : ∀ (l : List (Key × F128)) (st st' : BSt)
    (ps : Array (Nat × Nat)), coefOps sh comb0 ri l st = .ok (ps, st') → OpsLow (nR sh) st.ops →
      (∀ kx ∈ l, kx.1.Sym sh.claims) → OpsLow (nR sh) st'.ops
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
    have hA : OpsLow (nR sh) (st.ops.push #[Term.v st.vNames.size]) := hi.push (single_v_low _ _)
    exact coefOps_low hh ri rest st3 _ ps' hc (varOp_low hh hv hA (hl (k, x) (by simp)))
      fun kx hkx => hl kx (by simp [hkx])

theorem prodOps_low (hh : sh.holo = false) (ri : Nat) : ∀ (l : List ((Key × Key) × F128)) (st st' : BSt)
    (ps : Array (Nat × Nat)), prodOps sh comb0 ri l st = .ok (ps, st') → OpsLow (nR sh) st.ops →
      (∀ abx ∈ l, abx.1.1.Sym sh.claims ∧ abx.1.2.Sym sh.claims) → OpsLow (nR sh) st'.ops
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
      exact prodOps_low hh ri rest st3 _ ps' hc (varOp_low hh hb (varOp_low hh ha hi hab.1) hab.2)
        fun abx h => hl abx (by simp [h])

theorem accTerms_low (hh : sh.holo = false) {fin ri : Nat} {l : List (Key × F128)} {st st' : BSt}
    {acc : Array Term} (hr : accTerms sh fin ri l st = .ok (acc, st')) :
    st'.ops = st.ops ∧ (acc = #[] ∨ ∃ c < 2, acc = #[.w (nR sh + (2 + c)), .w (nR sh + c)]) := by
  refine ⟨(accTerms_spec (sh := sh) #[] #[] (fun _ _ => 0) hr).1, ?_⟩
  unfold accTerms at hr
  split at hr
  · cases hr
    exact .inl rfl
  · split at hr
    · cases hr
    · rename_i hne
      obtain ⟨_, st1, _, hr⟩ := bm_bind hr
      have hc : ri - fin < 2 := by omega
      simp only [RecDeploy.Rows.wi_acc1 hh hc, RecDeploy.Rows.wi_acc0 hh hc] at hr
      cases hr
      exact .inr ⟨ri - fin, hc, rfl⟩

theorem constTerm_low {ri : Nat} {c : Option F128} {st st' : BSt} {cp : Array Term}
    (hr : constTerm ri c st = .ok (cp, st')) : st'.ops = st.ops ∧ ∀ t ∈ cp, t.Low (nR sh) := by
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
    exact ⟨rfl, single_v_low _ _⟩

theorem buildRes_low (hh : sh.holo = false) {fin ri : Nat} {f : Lin} {st st' : BSt} {res : Res}
    (hr : buildRes sh fin comb0 ri f st = .ok (res, st')) (hi : OpsLow (nR sh) st.ops) (hf : f.Sym sh.claims) :
    OpsLow (nR sh) st'.ops ∧ AddLow (nR sh) res.add := by
  unfold buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, hacc⟩ := accTerms_low hh h1
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
  obtain ⟨c4, cl⟩ := constTerm_low (sh := sh) h4
  have hi1 : OpsLow (nR sh) st1.ops := by rw [a1]; exact hi
  have hg1 : ∀ kx ∈ l1, kx.1.Sym sh.claims :=
    fun kx hk => hf.1 kx (List.mem_of_mem_filter ((sortBy_perm hl1).subset hk))
  have hg2 : ∀ abx ∈ l2, abx.1.1.Sym sh.claims ∧ abx.1.2.Sym sh.claims :=
    fun abx hk => hf.2 abx ((sortBy_perm hl2).subset hk)
  have hi3 := prodOps_low hh ri l2 st2 st3 pm h3 (coefOps_low hh ri l1 st1 st2 cm h2 hi1 hg1) hg2
  exact ⟨by rw [c4]; exact hi3, acc, cp, rfl, hacc, cl⟩

theorem buildAll_low (hh : sh.holo = false) {fin : Nat} : ∀ (fs : List Lin) (ri : Nat) (st st' : BSt)
    (rs : Array Res), buildAll sh fin comb0 fs ri st = .ok (rs, st') → OpsLow (nR sh) st.ops →
      (∀ f ∈ fs, f.Sym sh.claims) → OpsLow (nR sh) st'.ops ∧ ∀ r ∈ rs, AddLow (nR sh) r.add
  | [], ri, st, st', rs, hr, hi, _ => by
    cases hr
    exact ⟨hi, fun r hr => by simp at hr⟩
  | f :: rest, ri, st, st', rs, hr, hi, hf => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    obtain ⟨h1i, h1a⟩ := buildRes_low hh h1 hi (hf f (by simp))
    obtain ⟨h2i, h2a⟩ := buildAll_low hh rest (ri + 1) st1 _ rs' h2 h1i fun g hg => hf g (by simp [hg])
    refine ⟨h2i, fun r' hr' => ?_⟩
    rcases Array.mem_append.mp hr' with hr' | hr'
    · simp at hr'
      subst hr'
      exact h1a
    · exact h2a r' hr'

end Build

theorem build_accLow {sh : Shape} (hh : sh.holo = false) {lins : Array Lin} {b : Built} (hb : build sh lins = .ok b)
    (hl : ∀ f ∈ lins.toList, f.Sym sh.claims) : b.S.AccLow (nR sh) := by
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
      exact buildAll_low hh lins.toList 0 _ _ _ hB' (fun _ h => by simp at h) hl

/-- **The verifier's S reads the running sums only as their pairs** (`structureOf`, RecAlgebra.lean:835–837), at a
plain shape. -/
theorem structureOf_accLow {sh : Shape} (hh : sh.holo = false) {S : GfS} (h : structureOf sh = .ok S) :
    S.AccLow (nR sh) := by
  unfold structureOf at h
  cases hl : residualLins sh Coins.zeros (extrasLike sh) with
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
      have hfix : lins.toList.map (Lin.ren (bumpA sh.claims)) = lins.toList := by
        rw [← Array.toList_map, residualLins_bumpA hl]
      exact build_accLow hh hb fun f hf => sym_of_ren (fixed_of_map hfix f hf)

end Flock.RecAlgebra

namespace FlockVBridge.PartS

open Flock
open Flock.RecResiduals (positions pos partS starts)
open FlockSoundness.Refine (fOf fOf_add fOf_zero fOf_injective)

/-- **The running sums' shift at the part's element `e`**: `δ` of the rep's running sum `(j - A) % 2` where the part's
w index `wI` places `e` at w's element `j ≥ A`, else 0. -/
def offOf (A : ℕ) (wI : Array ℕ) (δ : ℕ → F128) (e : ℕ) : F128 :=
  match wI[e]? with
  | some j => if A ≤ j then δ ((j - A) % 2) else 0
  | none => 0

theorem offOf_low {A : ℕ} {wI : Array ℕ} {δ : ℕ → F128} {e j : ℕ} (he : wI[e]? = some j) (hj : j < A) :
    offOf A wI δ e = 0 := by
  simp [offOf, he, Nat.not_le.2 hj]

theorem offOf_acc {A : ℕ} {wI : Array ℕ} {δ : ℕ → F128} {e c : ℕ} (he : wI[e]? = some (A + c)) :
    offOf A wI δ e = δ (c % 2) := by
  simp [offOf, he]

theorem f128_add_zero' (x : F128) : x + 0 = x := by
  (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem ofBits_congr {f g : ℕ → Bool} (h : ∀ j < 128, f j = g j) : ofBits f = ofBits g := by
  unfold ofBits
  congr 1
  exact List.map_congr_left fun j hj => by rw [h j (List.mem_range.1 hj)]

section Shift

variable {S : RecAlgebra.GfS} {wI V : Array ℕ} {A : ℕ} {δ : ℕ → F128} {E E' vv : ℕ → F128}

/-- **A looked-up term reading below `A` reads `E'` as `E`**, where `E'` is `E` shifted by `offOf` on the part's w. -/
theorem termE_low (hE : ∀ e < wI.size, E' e = E e + offOf A wI δ e) {t : RecAlgebra.Term} {b : RecOpen.Term}
    (ht : t.Low A) (h : termE S wI V t = .ok b) : termVal E' vv (termOf b) = termVal E vv (termOf b) := by
  cases t with
  | w i =>
    simp only [termE] at h
    split at h
    · rename_i i' hp
      cases h
      obtain ⟨hlt, hx⟩ := pos_some hp
      show E' i' = E i'
      rw [hE i' hlt, offOf_low (by rw [Array.getElem?_eq_getElem hlt, hx]) ht, f128_add_zero']
    · cases h
  | v j =>
    simp only [termE] at h
    split at h
    · cases h
      rfl
    · cases h
  | wt i u =>
    simp only [termE] at h
    split at h
    · rename_i i' hp
      split at h
      · rename_i hc
        cases h
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range, beq_iff_eq] at hc
        obtain ⟨hu, hb⟩ := hc
        show ofBits (fun j => wbit E' (128 * (i' + j) + u)) = ofBits (fun j => wbit E (128 * (i' + j) + u))
        refine ofBits_congr fun j hj => ?_
        have hq : (128 * (i' + j) + u) / 128 = i' + j := by omega
        unfold wbit
        rw [hq]
        have hlt : i' + j < wI.size := (Array.getElem?_eq_some_iff.1 (hb j hj)).1
        have hlow : i + j < A := by
          have : i + 128 ≤ A := ht
          omega
        rw [hE _ hlt, offOf_low (hb j hj) hlow, f128_add_zero']
      · cases h
    · cases h

theorem termsE_low (hE : ∀ e < wI.size, E' e = E e + offOf A wI δ e) :
    ∀ {ts : List RecAlgebra.Term} {bs : List RecOpen.Term}, List.Forall₂ (fun t b => termE S wI V t = .ok b) ts bs →
      (∀ t ∈ ts, t.Low A) → (bs.map (termVal E' vv ∘ termOf)) = bs.map (termVal E vv ∘ termOf)
  | _, _, .nil, _ => rfl
  | _, _, .cons h t, hl => by
    simp only [List.map_cons, Function.comp_apply, termE_low hE (hl _ (by simp)) h,
      termsE_low hE t fun x hx => hl x (by simp [hx])]

theorem termE_acc {i : ℕ} {b : RecOpen.Term} (h : termE S wI V (.w i) = .ok b) :
    ∃ i', i' < wI.size ∧ wI[i']? = some i ∧ b = .w i' := by
  simp only [termE] at h
  split at h
  · rename_i i' hp
    cases h
    obtain ⟨hlt, hx⟩ := pos_some hp
    exact ⟨i', hlt, by rw [Array.getElem?_eq_getElem hlt, hx], rfl⟩
  · cases h

theorem foldl_add_shift (g : RecOpen.Term → F128) (l : List RecOpen.Term) (c : F128) :
    (l.map g).foldl (· + ·) c = c + (l.map g).foldl (· + ·) 0 := by
  have := RecAlgebra.foldl_shift g l c
  simpa only [List.foldl_map] using this

theorem foldl_cons2 (g : RecOpen.Term → F128) (a b : RecOpen.Term) (l : List RecOpen.Term) :
    ((a :: b :: l).map g).foldl (· + ·) 0 = (g a + g b) + (l.map g).foldl (· + ·) 0 := by
  simp only [List.map_cons, List.foldl_cons]
  rw [foldl_add_shift g l]
  congr 1
  (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

/-- **An added part's sum doesn't see the shift**: the terms of `AddLow` looked up, the running sums' pair
`w[A + 2 + c] + w[A + c]` gaining `δ c` twice. -/
theorem addSum_off (hE : ∀ e < wI.size, E' e = E e + offOf A wI δ e) {add : Array RecAlgebra.Term}
    (ha : RecAlgebra.AddLow A add) {bs : List RecOpen.Term}
    (h : add.toList.mapM (termE S wI V) = .ok bs) :
    ((bs.map termOf).map (termVal E' vv)).foldl (· + ·) 0 = ((bs.map termOf).map (termVal E vv)).foldl (· + ·) 0 := by
  simp only [List.map_map]
  obtain ⟨acc, cp, rfl, hacc, hcp⟩ := ha
  have h2 := mapM_forall₂ h
  rw [Array.toList_append] at h2
  rcases hacc with rfl | ⟨c, hc, rfl⟩
  · change List.Forall₂ _ cp.toList bs at h2
    rw [termsE_low hE h2 fun t ht => hcp t (Array.mem_toList_iff.1 ht)]
  · change List.Forall₂ _ (RecAlgebra.Term.w (A + (2 + c)) :: RecAlgebra.Term.w (A + c) :: cp.toList) bs at h2
    rcases h2 with _ | ⟨h1, h2⟩
    rcases h2 with _ | ⟨h0, hrest⟩
    obtain ⟨i1, hl1, he1, rfl⟩ := termE_acc h1
    obtain ⟨i0, hl0, he0, rfl⟩ := termE_acc h0
    have hr := termsE_low (vv := vv) hE hrest fun t ht => hcp t (Array.mem_toList_iff.1 ht)
    rw [foldl_cons2, foldl_cons2, hr]
    congr 1
    show E' i1 + E' i0 = E i1 + E i0
    rw [hE i1 hl1, hE i0 hl0, offOf_acc (c := 2 + c) he1, offOf_acc he0,
      show (2 + c) % 2 = c % 2 by omega]
    have hd := Sparse.f128_add_self (δ (c % 2))
    have hd' : fOf (δ (c % 2)) + fOf (δ (c % 2)) = 0 := by rw [← fOf_add, hd, fOf_zero]
    apply fOf_injective
    simp only [fOf_add]
    linear_combination hd'

end Shift

/-- **A part's residuals vanish where they vanish at the shifted elements**: at the verifier's part `S'` of an S that
reads the running sums only as their pairs (`AccLow`), elements `E'` equal to `E` plus `offOf` on the part's w. -/
theorem resZero_off {S : RecAlgebra.GfS} {sizes R P V : Array ℕ} {S' : RecOpen.GfS}
    (h : partS S sizes R P V = .ok S') {A : ℕ} (hS : S.AccLow A) {δ : ℕ → F128} {E E' vv : ℕ → F128}
    (hE : ∀ e < (wIdx sizes P).size, E' e = E e + offOf A (wIdx sizes P) δ e)
    (hz : RecDeploy.ResZero S' E' vv) : RecDeploy.ResZero S' E vv := by
  unfold partS at h
  obtain ⟨used, -, h⟩ := bind_ok h
  obtain ⟨ops, hops, h⟩ := bind_ok h
  obtain ⟨rs, hrs, h⟩ := bind_ok h
  cases h
  have hops' : used.toList.mapM (opE S (wIdx sizes P) V) = .ok ops := hops
  have hrs' : R.toList.mapM (resE S (wIdx sizes P) V used) = .ok rs := hrs
  have ho := mapM_forall₂ hops'
  have hr := mapM_forall₂ hrs'
  -- every operand reads `E'` as `E`
  have hop : ∀ o ∈ ops, (o.map termOf).map (termVal E' vv) = (o.map termOf).map (termVal E vv) := by
    intro o hoM
    obtain ⟨k, hk⟩ := exists_of_forall₂ ho.flip o hoM
    change opE S (wIdx sizes P) V k = .ok o at hk
    unfold opE at hk
    split at hk
    · rename_i hlt
      simp only [List.map_map]
      exact termsE_low (A := A) (δ := δ) hE (mapM_forall₂ hk)
        fun t ht => hS.1 _ (Array.getElem_mem hlt) t (Array.mem_toList_iff.1 ht)
    · cases hk
  have hopv : ∀ k, opVal (gfsOf ⟨ops, rs⟩) (termVal E' vv) k = opVal (gfsOf ⟨ops, rs⟩) (termVal E vv) k := by
    intro k
    unfold opVal gfsOf
    simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
    cases hk : ops[k]? with
    | none => rfl
    | some o =>
      simp only [Option.map_some, Option.getD_some]
      obtain ⟨hlt, he⟩ := List.getElem?_eq_some_iff.1 hk
      rw [hop o (he ▸ List.getElem_mem hlt)]
  have key : ∀ r' ∈ rs, resVal (gfsOf ⟨ops, rs⟩) (termVal E' vv) (resOf r') =
      resVal (gfsOf ⟨ops, rs⟩) (termVal E vv) (resOf r') := by
    intro r' hr'
    obtain ⟨r, hrb⟩ := exists_of_forall₂ hr.flip r' hr'
    change resE S (wIdx sizes P) V used r = .ok r' at hrb
    unfold resE at hrb
    split at hrb
    · rename_i rr hrr
      obtain ⟨hlt, hrr⟩ := Array.getElem?_eq_some_iff.1 hrr
      obtain ⟨mul, -, hrb⟩ := bind_ok hrb
      obtain ⟨add, hadd, hrb⟩ := bind_ok hrb
      cases hrb
      have hA := addSum_off (vv := vv) hE (hS.2 _ (hrr ▸ Array.getElem_mem hlt)) hadd
      simp only [resVal, resOf, hopv]
      rw [hA]
    · cases hrb
  intro q hq
  exact (key _ (List.getElem_mem hq)).symm.trans (hz q hq)

end FlockVBridge.PartS
