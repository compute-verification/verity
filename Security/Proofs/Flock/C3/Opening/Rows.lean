import Proofs.Flock.C3.Opening.Read
import Proofs.Flock.VBridge.Algebra.Sizes

/-!
# n4's rows: the `rec-opening` unit's residuals are `RecOpening.run`'s

`residuals_eq_runO`, the opening-only analogue of `RecAlgebra.residuals_eq_run`: on the S and the v `RecCoins.opening`
builds at R's coins (`RecOpening.buildAll` over the residuals `RecOpening.run` returns on the symbolic messages and the
two claims, `kappa` from zero), `GfResiduals`' residuals at a w are `RecOpening.run` over F at those coins, on the
messages w holds (`decodedO`) and on the claims at their points with w's `claim` elements for values (`claimO`), when
w's running sums differ by the consistency sums weighted by `kappa`. `openingRead_of_rows`: so `OpeningRead` holds at
every accepted outcome where those residuals are all zero, the claims are the run's registered claims and the tape
begins with the opening's steps on w's ring switches.

* `buildAll_specO`, `build_evalO`: `RecAlgebra.buildAll_spec` and `build_eval` at the opening's w (no comb; `valO`),
  where residual `i < claims` adds claim `i`'s value from w (`claimAdd`).
* `run_simO`: `RecOpening.run` commutes with a `Hom`, as `RecAlgebra.run_sim`.
* `run_shift`: over F, raising each claim's value raises its ring switch's residual by as much; the rest is unchanged.
* `run_of_inline`: `RecCoins.opening`'s run, which also returns Ligerito's points, is `RecOpening.run`.
-/

namespace FlockC3.Opening

section Bridge

open Flock Flock.RecAlgebra FlockSoundness.Refine

/-! ## Values at the opening's w -/

/-- A message element's value at the opening's w (`RecAlgebra.valOf` with no comb): its element of w, its ring
switch's transpose of w, or the level's consistency sum. -/
def valO (sh : Shape) (w : Array F128) (cons : Nat → Nat → F128) : Key → F128
  | .rsT k u => match RecOpening.wi sh (.rs k 0) with
    | some i => termValue w #[] (.wt i u)
    | none => 0
  | .comb _ => 0
  | .cons level l => cons level l
  | k => match RecOpening.wi sh k with
    | some i => w[i]!
    | none => 0

/-- Residual `i`'s claim value as `RecOpening.buildRes` adds it: w's `claim i` for a claim's ring switch, else zero. -/
def claimAdd (sh : Shape) (w : Array F128) (i : Nat) : F128 :=
  if i < sh.claims then w[RecOpening.claimW sh i]! else 0

/-- Residual `fi.2`'s value as `RecOpening.buildRes` builds it from `fi.1` (`RecAlgebra.resForm`), its claim value
added. -/
def resFormO (sh : Shape) (w : Array F128) (val : Key → F128) (fi : Lin × Nat) : F128 :=
  resForm val sh.claims fi + claimAdd sh w fi.2

/-- `kappa` before the passes (`RecOpening.build`, `RecCoins.opening`). -/
abbrev kappa0 (sh : Shape) : Array (Array (Array F128)) :=
  sh.sched.levels.map fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))

/-- The op table's operands for message elements at the opening's w: each is the element's term alone. -/
def OpsOkO (sh : Shape) (st : BSt) : Prop :=
  st.vNames.size = st.vVals.size ∧
    ∀ k i, st.opOf.lookup k = some i → ∃ t, RecOpening.atomTerm sh k = .ok t ∧ st.ops[i]? = some #[t]

/-! ## What `RecOpening.buildAll`'s passes compute -/

variable {sh : Shape}

theorem OpsOkO.push {st : BSt} (hi : OpsOkO sh st) (n : VName) (x : F128) (op : Array Term) :
    OpsOkO sh { st with vNames := st.vNames.push n, vVals := st.vVals.push x, ops := st.ops.push op } := by
  refine ⟨by simp [hi.1], fun k i hk => ?_⟩
  obtain ⟨t, ht, ho⟩ := hi.2 k i hk
  exact ⟨t, ht, (Ext.push st n x op).ops? ho⟩

theorem OpsOkO.of_eq {st st' : BSt} (hi : OpsOkO sh st) (h1 : st'.ops = st.ops) (h2 : st'.vVals = st.vVals)
    (h3 : st'.vNames = st.vNames) (h4 : st'.opOf = st.opOf) : OpsOkO sh st' := by
  refine ⟨by rw [h2, h3]; exact hi.1, fun k i hk => ?_⟩
  rw [h4] at hk
  rw [h1]
  exact hi.2 k i hk

theorem varOp_specO {k : Key} {st st' : BSt} {o : Nat} (hi : OpsOkO sh st)
    (hr : RecOpening.varOp sh k st = .ok (o, st')) :
    OpsOkO sh st' ∧ Ext st st' ∧ st'.vNames = st.vNames ∧ st'.vVals = st.vVals ∧ st'.kappa = st.kappa ∧
      ∃ t, RecOpening.atomTerm sh k = .ok t ∧ st'.ops[o]? = some #[t] := by
  unfold RecOpening.varOp at hr
  obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
  cases hg
  cases hl : st.opOf.lookup k with
  | some i =>
    simp only [hl] at hr
    cases hr
    exact ⟨hi, Ext.refl _, rfl, rfl, rfl, hi.2 k _ hl⟩
  | none =>
    simp only [hl] at hr
    obtain ⟨t', st2, ht, hr⟩ := bm_bind hr
    obtain ⟨ha, h2⟩ := liftB_ok ht
    subst st2
    obtain ⟨_, st3, hs, hr⟩ := bm_bind hr
    cases hs
    cases hr
    refine ⟨⟨hi.1, fun k' i' hk' => ?_⟩, ⟨⟨#[#[t']], Array.push_eq_append⟩, ⟨#[], by simp⟩⟩, rfl, rfl,
      rfl, t', ha, by simp⟩
    simp only [List.lookup_append] at hk'
    cases hk'' : st.opOf.lookup k' with
    | some j =>
      rw [hk''] at hk'
      cases hk'
      obtain ⟨t2, h2, h3⟩ := hi.2 k' _ hk''
      exact ⟨t2, h2, (Ext.push st (.const 0) 0 #[t']).ops? h3⟩
    | none =>
      rw [hk''] at hk'
      simp only [List.lookup, Option.none_or] at hk'
      split at hk'
      · rename_i hkk
        cases hk'
        have : k' = k := by simpa using hkk
        subst this
        exact ⟨t', ha, by simp⟩
      · cases hk'

variable (w : Array F128) (cons : Nat → Nat → F128)

theorem atomO_val {k : Key} {t : Term} (V : Array F128) (ha : RecOpening.atomTerm sh k = .ok t) :
    termValue w V t = valO sh w cons k := by
  cases k
  all_goals (simp only [RecOpening.atomTerm, valO] at ha ⊢; (try split at ha) <;> (try cases ha) <;>
    (try simp_all [termValue]))

theorem opValue_atomO {S : GfS} {o : Nat} {k : Key} {t : Term} {V : Array F128} (h : S.ops[o]? = some #[t])
    (ha : RecOpening.atomTerm sh k = .ok t) : opValue S w V o = valO sh w cons k := by
  rw [opValue_single w V h]
  exact atomO_val w cons V ha

theorem coefOps_specO (ri : Nat) : ∀ (l : List (Key × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    OpsOkO sh st → RecOpening.coefOps sh ri l st = .ok (ps, st') →
    OpsOkO sh st' ∧ Ext st st' ∧ st'.kappa = st.kappa ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V → mulSum w S V ps = tEval (valO sh w cons) l)
  | [], st, st', ps, hi, hr => by
    cases hr
    exact ⟨hi, Ext.refl _, rfl, fun _ _ _ => rfl⟩
  | (k, x) :: rest, st, st', ps, hi, hr => by
    obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    obtain ⟨o, st3, hv, hr⟩ := bm_bind hr
    obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
    cases hr
    have hA := hi.push (.coef ri k) x #[.v st.vNames.size]
    obtain ⟨h3i, h3e, -, -, h3k, t, hop, ho⟩ := varOp_specO hA hv
    obtain ⟨h4i, h4e, h4k, hval⟩ := coefOps_specO ri rest st3 st' ps' h3i hc
    refine ⟨h4i, (Ext.push st _ _ _).trans (h3e.trans h4e), by rw [h4k, h3k], fun S V hSV => ?_⟩
    rw [mulSum_cons, hval S V hSV, tEval]
    have hslot : S.ops[st.ops.size]? = some #[Term.v st.vNames.size] := (h3e.trans (h4e.trans hSV)).ops? (by simp)
    have hx : V[st.vNames.size]? = some x := (h3e.trans (h4e.trans hSV)).vVals? (by simp [hi.1])
    rw [opValue_single w _ hslot, opValue_atomO w cons ((h4e.trans hSV).ops? ho) hop]
    simp only [termValue, getElem!_def, hx]

theorem prodOps_specO (ri : Nat) : ∀ (l : List ((Key × Key) × F128)) (st st' : BSt) (ps : Array (Nat × Nat)),
    OpsOkO sh st → RecOpening.prodOps sh ri l st = .ok (ps, st') →
    OpsOkO sh st' ∧ Ext st st' ∧ st'.kappa = st.kappa ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V →
        mulSum w S V ps = tEval (fun ab => valO sh w cons ab.1 * valO sh w cons ab.2) l)
  | [], st, st', ps, hi, hr => by
    cases hr
    exact ⟨hi, Ext.refl _, rfl, fun _ _ _ => rfl⟩
  | ((a, b), x) :: rest, st, st', ps, hi, hr => by
    simp only [RecOpening.prodOps] at hr
    split at hr
    · obtain ⟨_, _, h1, _⟩ := bm_bind hr
      cases h1
    · rename_i hx1
      obtain ⟨oa, st2, ha, hr⟩ := bm_bind hr
      obtain ⟨ob, st3, hb, hr⟩ := bm_bind hr
      obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
      cases hr
      obtain ⟨h2i, h2e, -, -, h2k, ta, hopa, hoa⟩ := varOp_specO hi ha
      obtain ⟨h3i, h3e, -, -, h3k, tb, hopb, hob⟩ := varOp_specO h2i hb
      obtain ⟨h4i, h4e, h4k, hval⟩ := prodOps_specO ri rest st3 st' ps' h3i hc
      refine ⟨h4i, h2e.trans (h3e.trans h4e), by rw [h4k, h3k, h2k], fun S V hSV => ?_⟩
      have hx : x = 1 := by simpa using hx1
      rw [mulSum_cons, hval S V hSV, tEval, hx,
        opValue_atomO w cons ((h3e.trans (h4e.trans hSV)).ops? hoa) hopa,
        opValue_atomO w cons ((h4e.trans hSV).ops? hob) hopb]
      (apply fOf_injective; simp only [fOf_add, fOf_mul, fOf_one]; ring)

theorem consOf_valO (f : Lin) :
    tEval (consVal cons) (RecAlgebra.consOf f) = tEval (valO sh w cons) (RecAlgebra.consOf f) := by
  apply tEval_congr
  intro kx hk
  simp only [RecAlgebra.consOf, List.mem_filter] at hk
  obtain ⟨k, x⟩ := kx
  cases k <;> simp_all [Key.isCons, consVal, valO]

theorem accTerms_specO {ri : Nat} {l : List (Key × F128)} {st st' : BSt} {acc : Array Term}
    (hr : RecOpening.accTerms sh ri l st = .ok (acc, st')) :
    st'.ops = st.ops ∧ st'.vVals = st.vVals ∧ st'.vNames = st.vNames ∧ st'.opOf = st.opOf ∧
      (∀ r', kapSum st'.kappa r' cons =
        kapSum st.kappa r' cons + (if ri = sh.claims + r' then tEval (consVal cons) l else 0)) ∧
      (l ≠ [] → ri = sh.claims ∨ ri = sh.claims + 1) ∧
      ∀ v : Array F128, acc.foldl (fun x t => x + termValue w v t) 0 = (if l = [] then 0 else
        valO sh w cons (.acc1 (ri - sh.claims)) + valO sh w cons (.acc0 (ri - sh.claims))) := by
  simp only [RecOpening.accTerms] at hr
  split at hr
  · rename_i he
    cases hr
    have : l = [] := List.isEmpty_iff.mp he
    subst this
    refine ⟨rfl, rfl, rfl, rfl, fun r' => ?_, fun h => absurd rfl h, fun v => rfl⟩
    split <;> simp only [tEval, f128_add_zero]
  · rename_i he
    have hne : l ≠ [] := fun h => by simp [h] at he
    split at hr
    · cases hr
    · rename_i hri
      have hfin : ri = sh.claims ∨ ri = sh.claims + 1 := by omega
      obtain ⟨_, st1, hk, hr⟩ := bm_bind hr
      obtain ⟨e1, e2, e3, e4, hks⟩ := kapAdd_spec cons (ri - sh.claims) l st st1 hk
      cases h1 : RecOpening.wi sh (.acc1 (ri - sh.claims)) with
      | none => simp only [h1] at hr; cases hr
      | some i1 =>
        cases h0 : RecOpening.wi sh (.acc0 (ri - sh.claims)) with
        | none => simp only [h1, h0] at hr; cases hr
        | some i0 =>
          simp only [h1, h0] at hr
          cases hr
          refine ⟨e1, e2, e3, e4, fun r' => ?_, fun _ => hfin, fun v => ?_⟩
          · rw [hks]
            split_ifs <;> first | rfl | (exfalso; omega)
          · simp only [hne, ite_false, List.foldl_toArray', List.foldl_cons, List.foldl_nil, termValue, valO, h1,
              h0, f128_zero_add]

theorem constTerm_specO {ri : Nat} {c : Option F128} {st st' : BSt} {cp : Array Term} (hi : OpsOkO sh st)
    (hr : constTerm ri c st = .ok (cp, st')) :
    OpsOkO sh st' ∧ Ext st st' ∧ st'.kappa = st.kappa ∧
      ∀ (F : Array (Array Term)) (V : Array F128), ExtTo st' F V →
        cp.foldl (fun x t => x + termValue w V t) 0 = c.getD 0 := by
  cases c with
  | none =>
    cases hr
    exact ⟨hi, Ext.refl _, rfl, fun _ _ _ => rfl⟩
  | some c =>
    unfold constTerm at hr
    obtain ⟨_, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    cases hr
    refine ⟨⟨by simp [hi.1], hi.2⟩, ⟨⟨#[], by simp⟩, ⟨#[c], Array.push_eq_append⟩⟩, rfl, fun F V hFV => ?_⟩
    have hx : V[st.vNames.size]? = some c := hFV.vVals? (by simp [hi.1])
    simp only [List.foldl_toArray', List.foldl_cons, List.foldl_nil, termValue, getElem!_def, hx,
      Option.getD_some, f128_zero_add]

theorem buildRes_specO {ri : Nat} {f : Lin} {st st' : BSt} {res : Res} (hi : OpsOkO sh st)
    (hr : RecOpening.buildRes sh ri f st = .ok (res, st')) :
    OpsOkO sh st' ∧ Ext st st' ∧
      (∀ r', kapSum st'.kappa r' cons =
        kapSum st.kappa r' cons + (if ri = sh.claims + r' then tEval (valO sh w cons) (RecAlgebra.consOf f) else 0)) ∧
      (RecAlgebra.consOf f ≠ [] → ri = sh.claims ∨ ri = sh.claims + 1) ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V →
        resValue S w V res = resFormO sh w (valO sh w cons) (f, ri)) := by
  unfold RecOpening.buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, a2, a3, a4, aks, afin, aval⟩ := accTerms_specO w cons h1
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
  have hi1 : OpsOkO sh st1 := hi.of_eq a1 a2 a3 a4
  obtain ⟨h2i, h2e, h2k, h2val⟩ := coefOps_specO w cons ri l1 st1 st2 cm hi1 h2
  obtain ⟨h3i, h3e, h3k, h3val⟩ := prodOps_specO w cons ri l2 st2 st3 pm h2i h3
  obtain ⟨h4i, h4e, h4k, h4val⟩ := constTerm_specO w h3i h4
  have hcl : ∀ V : Array F128, (if ri < sh.claims then #[Term.w (RecOpening.claimW sh ri)] else #[]).foldl
      (fun x t => x + termValue w V t) 0 = claimAdd sh w ri := by
    intro V
    unfold claimAdd
    by_cases hc : ri < sh.claims
    · rw [ite_eq_left hc, ite_eq_left hc]; exact afoldl_single _ _
    · rw [ite_eq_right hc, ite_eq_right hc]; rfl
  refine ⟨h4i, (Ext.of_eq a1 a2).trans (h2e.trans (h3e.trans h4e)), fun r' => ?_, afin, fun S V hSV => ?_⟩
  · have hc := consOf_valO (sh := sh) w cons f
    simp only [RecAlgebra.consOf] at hc ⊢
    rw [h4k, h3k, h2k, aks, hc]
  · rw [resValue_eq, afoldl_append, afoldl_append, mulSum_append, aval, hcl, h4val S.ops V hSV,
      h2val S V (h3e.trans (h4e.trans hSV)), h3val S V (h4e.trans hSV), tEval_perm _ (sortBy_perm hl1),
      tEval_perm _ (sortBy_perm hl2)]
    simp only [resFormO, resForm, noCons, RecAlgebra.consOf]
    by_cases hc : List.filter (fun x => x.1.isCons) f.t = [] <;> simp only [hc, ite_true, ite_false]
    all_goals (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)

theorem buildAll_specO : ∀ (fs : List Lin) (ri : Nat) (st st' : BSt) (rs : Array Res),
    OpsOkO sh st → RecOpening.buildAll sh fs ri st = .ok (rs, st') →
    OpsOkO sh st' ∧ Ext st st' ∧
      (∀ r', kapSum st'.kappa r' cons = kapSum st.kappa r' cons +
        (if ri ≤ sh.claims + r' then tEval (valO sh w cons) (RecAlgebra.consOf (fs.getD (sh.claims + r' - ri) {}))
          else 0)) ∧
      (∀ fi ∈ fs.zipIdx ri, RecAlgebra.consOf fi.1 ≠ [] → fi.2 = sh.claims ∨ fi.2 = sh.claims + 1) ∧
      (∀ (S : GfS) (V : Array F128), ExtTo st' S.ops V →
        rs.toList.map (resValue S w V) = (fs.zipIdx ri).map (resFormO sh w (valO sh w cons)))
  | [], ri, st, st', rs, hi, hr => by
    cases hr
    refine ⟨hi, Ext.refl _, fun r' => ?_, by simp, fun _ _ _ => rfl⟩
    have : tEval (valO sh w cons) (RecAlgebra.consOf (([] : List Lin).getD (sh.claims + r' - ri) {})) = 0 := rfl
    split <;> simp only [this, f128_add_zero]
  | f :: rest, ri, st, st', rs, hi, hr => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    obtain ⟨h1i, h1e, h1k, h1fin, h1val⟩ := buildRes_specO w cons hi h1
    obtain ⟨h2i, h2e, h2k, h2fin, h2val⟩ := buildAll_specO rest (ri + 1) st1 st' rs' h1i h2
    refine ⟨h2i, h1e.trans h2e, fun r' => ?_, fun fi hfi => ?_, fun S V hSV => ?_⟩
    · rw [h2k, h1k]
      rcases Nat.lt_trichotomy ri (sh.claims + r') with hlt | heq | hgt
      · rw [show sh.claims + r' - ri = (sh.claims + r' - (ri + 1)) + 1 by omega, List.getD_cons_succ]
        simp only [show ¬ri = sh.claims + r' by omega, show ri + 1 ≤ sh.claims + r' by omega,
          show ri ≤ sh.claims + r' by omega, ite_true, ite_false]
        (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
      · simp only [heq, show ¬sh.claims + r' + 1 ≤ sh.claims + r' by omega, Nat.le_refl, Nat.sub_self,
          List.getD_cons_zero, ite_true, ite_false]
        (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
      · simp only [show ¬ri = sh.claims + r' by omega, show ¬ri + 1 ≤ sh.claims + r' by omega,
          show ¬ri ≤ sh.claims + r' by omega, ite_false]
        (apply fOf_injective; simp only [fOf_add, fOf_zero]; ring)
    · rw [List.zipIdx_cons, List.mem_cons] at hfi
      rcases hfi with rfl | hfi
      · exact h1fin
      · exact h2fin fi hfi
    · rw [List.zipIdx_cons, List.map_cons, Array.toList_append, List.map_append, h2val S V hSV]
      show [resValue S w V r] ++ _ = _
      rw [h1val S V (h2e.trans hSV)]
      rfl

/-- **The opening's built residuals are its symbolic residuals evaluated**, at the message elements w holds, each
consistency sum read through the running sums (`hacc`), and residual `i < claims` raised by claim `i`'s value in w. -/
theorem build_evalO {lins : Array Lin} {res : Array Res} {st : BSt}
    (hb : (RecOpening.buildAll sh lins.toList 0).run { kappa := kappa0 sh } = .ok (res, st))
    (hw : w.size = RecOpening.nw sh)
    (hacc : ∀ r < 2, valO sh w cons (.acc1 r) + valO sh w cons (.acc0 r) = kapSum st.kappa r cons) :
    ∃ rs, residuals ⟨RecOpening.nw sh, st.vNames.size, st.ops, res⟩ w st.vVals = .ok rs ∧
      rs.toList = (lins.toList.zipIdx 0).map fun fi => fi.1.eval (valO sh w cons) + claimAdd sh w fi.2 := by
  have hB : RecOpening.buildAll sh lins.toList 0 { kappa := kappa0 sh } = .ok (res, st) := hb
  have hinit : OpsOkO sh { kappa := kappa0 sh } := ⟨rfl, fun _ _ hk => by cases hk⟩
  obtain ⟨hst, -, hk, hfin, hval⟩ := buildAll_specO w cons lins.toList 0 _ _ _ hinit hB
  have hres : residuals ⟨RecOpening.nw sh, st.vNames.size, st.ops, res⟩ w st.vVals =
      .ok (res.map (resValue ⟨RecOpening.nw sh, st.vNames.size, st.ops, res⟩ w st.vVals)) := by
    unfold residuals
    exact ite_eq_right (by simp [hw, hst.1])
  refine ⟨_, hres, ?_⟩
  rw [Array.toList_map, hval _ st.vVals ⟨⟨#[], by simp⟩, ⟨#[], by simp⟩⟩]
  apply List.map_congr_left
  intro fi hfi
  obtain ⟨f, i⟩ := fi
  obtain ⟨-, hi, hf⟩ := List.mem_zipIdx hfi
  simp only [resFormO, resForm]
  rw [lin_eval_split]
  by_cases hcf : RecAlgebra.consOf f = []
  · simp only [hcf, ite_true, tEval, f128_add_zero]
  · simp only [hcf, ite_false]
    have hr := hfin (f, i) hfi hcf
    simp only at hr
    rw [hacc _ (by omega), hk, show kapSum (kappa0 sh) (i - sh.claims) cons = 0 from kapSum_init cons _ _]
    simp only [Nat.zero_le, ite_true]
    rw [show sh.claims + (i - sh.claims) - 0 = i by omega, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem (by simpa using hi), Option.getD_some, f128_zero_add]
    simp only [Nat.sub_zero] at hf
    rw [← hf]

/-! ## `RecOpening.run` -/

/-- `RecOpening.run` commutes with a `Hom` (`RecAlgebra.run_sim`). -/
theorem run_simO {α β : Type} [Val α] [Val β] {h : α → β} (hh : Hom h) (sh : Shape) (M : Messages α)
    (claims : Array (AClaim α)) :
    Sim (·.map h) (RecOpening.run sh M claims) (RecOpening.run sh (M.map h) (claims.map (AClaim.map h))) := by
  unfold RecOpening.run
  refine Sim.bind (ringSwitch_sim hh M claims) fun ⟨rsr, rs⟩ => ?_
  have ht : (M.map h).tAtom.getD (rs.map h).T = h (M.tAtom.getD rs.T) := by
    simp only [Messages.map, RsPts.map]
    cases M.tAtom <;> rfl
  simp only [ht]
  refine Sim.bind (ligerito_sim hh _ M _ _) fun ⟨lgr, lg⟩ => Sim.pure' ?_
  simp [RsPts.map, hh.add]

/-- `RecCoins.opening`'s run, which also returns Ligerito's points, is `RecOpening.run`. -/
theorem run_of_inline {α : Type} [Val α] (sh : Shape) (M : Messages α) (claims : Array (AClaim α)) :
    Sim Prod.fst (do
      let (rsr, rs) ← RecAlgebra.ringSwitch M claims
      let tIn := M.tAtom.getD rs.T
      let (lgr, lg) ← RecAlgebra.ligerito sh.sched M tIn rs.sws
      return (rsr ++ lgr ++ #[Val.add tIn rs.T], lg) : CM (Array α × LgPts)) (RecOpening.run sh M claims) := by
  unfold RecOpening.run
  exact Sim.bind_same fun ⟨rsr, rs⟩ => Sim.bind_same fun ⟨lgr, lg⟩ => Sim.pure' rfl

/-- The claims from index `k` on, each value raised by `g` at its index. -/
def shiftC (g : Nat → F128) (k : Nat) (cls : List (AClaim F128)) : List (AClaim F128) :=
  (cls.zipIdx k).map fun ci => ⟨ci.1.value + g ci.2, ci.1.skip, ci.1.x⟩

/-- The elements from index `k` on, each raised by `g` at its index. -/
def shiftA (g : Nat → F128) (k : Nat) (xs : Array F128) : Array F128 :=
  ⟨(xs.toList.zipIdx k).map fun xi => xi.1 + g xi.2⟩

theorem addv_shift (a b g : F128) : (Val.add a b : F128) + g = Val.add a (b + g) := by
  show a + b + g = a + (b + g)
  (apply fOf_injective; simp only [fOf_add]; ring)

theorem map_zipIdx_ge (g : Nat → F128) (l : List F128) (n : Nat) (hg : ∀ i, n ≤ i → g i = 0) :
    (l.zipIdx n).map (fun xi => xi.1 + g xi.2) = l := by
  conv => rhs; rw [← List.zipIdx_map_fst n l]
  apply List.map_congr_left
  intro xi hxi
  obtain ⟨x, i⟩ := xi
  show x + g i = x
  rw [hg i (List.mem_zipIdx hxi).1, f128_add_zero]

theorem shiftA_append (g : Nat → F128) {xs ys : Array F128} (hg : ∀ i, xs.size ≤ i → g i = 0) :
    shiftA g 0 (xs ++ ys) = shiftA g 0 xs ++ ys := by
  apply Array.toList_inj.mp
  simp only [shiftA, Array.toList_append, List.zipIdx_append, List.map_append]
  congr 1
  exact map_zipIdx_ge g _ _ fun i hi => hg i (by simpa using hi)

theorem zipIdx_map_add {γ : Type} (f : γ → F128) (g : Nat → F128) : ∀ (l : List γ) (n : Nat),
    ((l.map f).zipIdx n).map (fun xi => xi.1 + g xi.2) = (l.zipIdx n).map (fun fi => f fi.1 + g fi.2)
  | [], _ => rfl
  | b :: l, n => by simp only [List.map_cons, List.zipIdx_cons, zipIdx_map_add f g l (n + 1)]

theorem rsClaim_shift (M : Messages F128) (kc : Nat) (cl : AClaim F128) (g : F128) :
    Sim (fun p => (p.1 + g, p.2)) (rsClaim M kc cl) (rsClaim M kc ⟨cl.value + g, cl.skip, cl.x⟩) := by
  unfold rsClaim
  exact Sim.bind_same fun _ => Sim.pure' (Prod.ext (addv_shift _ _ _) rfl)

theorem rsClaims_shift (M : Messages F128) (g : Nat → F128) : ∀ (cls : List (AClaim F128)) (k : Nat),
    Sim (fun p => (shiftA g k p.1, p.2)) (rsClaims M cls k) (rsClaims M (shiftC g k cls) k)
  | [], k => by
    simp only [shiftC, List.zipIdx_nil, List.map_nil, rsClaims]
    exact Sim.pure' rfl
  | cl :: rest, k => by
    simp only [shiftC, List.zipIdx_cons, List.map_cons]
    refine Sim.bind (rsClaim_shift M k cl (g k)) fun ⟨o, b, r, z⟩ => ?_
    refine Sim.bind (rsClaims_shift M g rest (k + 1)) fun ⟨outs, betas, pts⟩ => Sim.pure' (Prod.ext ?_ rfl)
    apply Array.toList_inj.mp
    simp [shiftA, List.zipIdx_cons]

theorem ringSwitch_shift (M : Messages F128) (g : Nat → F128) (claims : Array (AClaim F128)) :
    Sim (fun p => (shiftA g 0 p.1, p.2)) (RecAlgebra.ringSwitch M claims)
      (RecAlgebra.ringSwitch M (shiftC g 0 claims.toList).toArray) := by
  unfold RecAlgebra.ringSwitch
  have hs : (shiftC g 0 claims.toList).toArray.size = claims.size := by simp [shiftC]
  rw [hs]
  exact Sim.bind (rsClaims_shift M g claims.toList 0) fun ⟨outs, betas, pts⟩ =>
    Sim.bind_same fun _ => Sim.pure' rfl

/-- **Raising the claims' values raises their ring switches' residuals** by as much, over F: the rest of
`RecOpening.run` does not read the claims' values. -/
theorem run_shift {sh : Shape} {M : Messages F128} {claims : Array (AClaim F128)} {g : Nat → F128}
    (hn : claims.size ≠ 0) (hg : ∀ i, claims.size ≤ i → g i = 0) {k k' : Nat} {src : RecAlgebra.CoinSource} {rs : Array F128}
    (h : RecOpening.run sh M claims k src = .ok (rs, k')) :
    RecOpening.run sh M (shiftC g 0 claims.toList).toArray k src = .ok (shiftA g 0 rs, k') := by
  unfold RecOpening.run at h ⊢
  obtain ⟨⟨rsr, rp⟩, k₁, h1, h⟩ := cm_bind h
  obtain ⟨-, hsz, -⟩ := ringSwitch_spec hn h1
  rw [cm_bind_of (ringSwitch_shift M g claims k src _ _ h1)]
  obtain ⟨⟨lgr, lg⟩, k₂, h2, h⟩ := cm_bind h
  cases cm_pure h
  rw [Array.append_assoc, shiftA_append g (fun i hi => hg i (by omega)), ← Array.append_assoc]
  exact (cm_bind_of h2).trans rfl

/-! ## The bridge -/

/-- The messages w holds (`RecOpening`'s elements at `valO`): its elements and the levels' consistency sums `cons`. -/
def decodedO (sh : Shape) (w : Array F128) (cons : Nat → Nat → F128) : Messages F128 :=
  (symbolic sh).map (Lin.eval (valO sh w cons))

/-- Claim `i` as `RecOpening.run` reads it at w: the symbolic claim's point, w's `claim i` its value. -/
def claimO (sh : Shape) (w : Array F128) (c : Fin 2 → AClaim Lin) (i : Fin 2) : AClaim F128 :=
  ⟨w[RecOpening.claimW sh i]!, (c i).skip, (c i).x⟩

theorem claimO_eq {c : Fin 2 → AClaim Lin} (hc : ∀ i, (c i).value = Val.zero) (hcl : sh.claims = 2) (i : Fin 2) :
    (⟨(AClaim.map (Lin.eval (valO sh w cons)) (c i)).value + claimAdd sh w i, (c i).skip, (c i).x⟩ : AClaim F128) =
      claimO sh w c i := by
  have hz : Lin.eval (valO sh w cons) (c i).value = 0 := by rw [hc i]; exact Lin.eval_zero _
  show (⟨Lin.eval (valO sh w cons) (c i).value + claimAdd sh w i, (c i).skip, (c i).x⟩ : AClaim F128) =
    ⟨w[RecOpening.claimW sh i]!, (c i).skip, (c i).x⟩
  rw [hz, f128_zero_add, claimAdd, ite_eq_left (by have := i.isLt; omega)]

/-- **`GfResiduals` on the opening's S and v computes `RecOpening.run`** (`RecAlgebra.residuals_eq_run` at the
`rec-opening` shape, two claims): at the S and the v the passes build over the residuals `RecOpening.run` returns on
the symbolic messages and two claims of value zero at some coins, the residuals at a w are `RecOpening.run` over F at
those coins, on the messages w holds and on the claims with w's `claim` elements for values, when w's running sums
differ by the consistency sums weighted by the passes' `kappa` (`hacc`). -/
theorem residuals_eq_runO {c : Fin 2 → AClaim Lin} {src : RecAlgebra.CoinSource} {lins : Array Lin} {k : Nat}
    {res : Array Res} {st : BSt} (hcl : sh.claims = 2) (hc : ∀ i, (c i).value = Val.zero)
    (hrun : RecOpening.run sh (symbolic sh) #[c 0, c 1] 0 src = .ok (lins, k))
    (hb : (RecOpening.buildAll sh lins.toList 0).run { kappa := kappa0 sh } = .ok (res, st))
    (hw : w.size = RecOpening.nw sh)
    (hacc : ∀ r < 2, valO sh w cons (.acc1 r) + valO sh w cons (.acc0 r) = kapSum st.kappa r cons) :
    ∃ rs, RecOpening.run sh (decodedO sh w cons) #[claimO sh w c 0, claimO sh w c 1] 0 src = .ok (rs, k) ∧
      residuals ⟨RecOpening.nw sh, st.vNames.size, st.ops, res⟩ w st.vVals = .ok rs := by
  have h1 := run_simO (Lin.eval_hom (valO sh w cons)) sh (symbolic sh) #[c 0, c 1] 0 src _ _ hrun
  simp only [List.map_toArray, List.map_cons, List.map_nil] at h1
  have h2 := run_shift (g := claimAdd sh w) (by simp)
    (fun i hi => ite_eq_right (by simp at hi; omega)) h1
  have e : (shiftC (claimAdd sh w) 0 (#[AClaim.map (Lin.eval (valO sh w cons)) (c 0),
      AClaim.map (Lin.eval (valO sh w cons)) (c 1)]).toList).toArray = #[claimO sh w c 0, claimO sh w c 1] := by
    rw [← claimO_eq w cons hc hcl 0, ← claimO_eq w cons hc hcl 1]
    rfl
  obtain ⟨rs, hres, hl⟩ := build_evalO w cons hb hw hacc
  have hrs : shiftA (claimAdd sh w) 0 (lins.map (Lin.eval (valO sh w cons))) = rs := by
    apply Array.toList_inj.mp
    rw [hl]
    simp only [shiftA, Array.toList_map]
    exact zipIdx_map_add _ _ _ _
  refine ⟨rs, ?_, hres⟩
  rw [← e, ← hrs]
  exact h2

end Bridge

/-! ## `OpeningRead` from the rows -/

section Read

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

theorem shape_claims {m c : ℕ} {sh : Flock.Shape} (h : Flock.RecOpening.shape? m c = some sh) : sh.claims = c := by
  unfold Flock.RecOpening.shape? at h
  cases hf : Flock.fast100 m with
  | none => simp [hf] at h
  | some s =>
    by_cases hc : c < 1
    · simp [hf, hc] at h
    · simp [hf, hc] at h
      rw [← h]

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n4's read fact from the rows) `OpeningRead`**, at every accepted outcome where the `rec-opening` unit's
residuals are all zero on the S and v `RecCoins.opening` builds at R's coins (two claims of value zero, the passes
from `kappa` zero) and on a w whose running sums differ by the consistency sums weighted by `kappa`, the claims at
w's `claim` values are the run's two registered claims, and the tape past both sparse phases begins with the
opening's steps on the ring switches w holds. -/
theorem openingRead_of_rows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) {dg : ByteArray → ByteArray}
    {N k Rw : ℕ}
    (hrows : F.AtAccepted dg N k Rw fun R ω t Lb =>
      let u := F.run R ω t Lb
      ∃ (sh : Flock.Shape) (w : Array F128) (cons : ℕ → ℕ → F128) (c : Fin 2 → RecAlgebra.AClaim RecAlgebra.Lin)
        (src : RecAlgebra.CoinSource) (lins : Array RecAlgebra.Lin) (k₁ : ℕ) (res : Array RecAlgebra.Res)
        (bs : RecAlgebra.BSt) (rs : Array F128) (ρ : Fin 2 → Fin 7 → GF128) (γ : Fin 2 → GF128),
        RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh ∧ (∀ i, (c i).value = RecAlgebra.Val.zero) ∧
        RecOpening.run sh (RecAlgebra.symbolic sh) #[c 0, c 1] 0 src = .ok (lins, k₁) ∧
        (RecOpening.buildAll sh lins.toList 0).run { kappa := kappa0 sh } = .ok (res, bs) ∧
        w.size = RecOpening.nw sh ∧
        (∀ r < 2, valO sh w cons (.acc1 r) + valO sh w cons (.acc0 r) = RecAlgebra.kapSum bs.kappa r cons) ∧
        RecAlgebra.residuals ⟨RecOpening.nw sh, bs.vNames.size, bs.ops, res⟩ w bs.vVals = .ok rs ∧
        (∀ x ∈ rs, x = 0) ∧
        (∀ i, FlockSoundness.Refine.ClaimRel (holoAt st hp).cls.shape.mReg (claimO sh w c i).toClaim
          (u.claim i).toClaim) ∧
        openSteps (fun i => FlockSoundness.Refine.vecOf 128 (decodedO sh w cons).rs[i.val]!) ρ γ <+:
          afterSparse u (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd)))) :
    OpeningRead F dg N k Rw := by
  intro R ω t R₂ τ o hacc
  obtain ⟨sh, w, cons, c, src, lins, k₁, res, bs, rs, ρ, γ, hsh, hc, hrun, hb, hw, hk, hres, hz, hrel, hpre⟩ :=
    hrows R ω t R₂ τ o hacc
  obtain ⟨rs', hop, hres'⟩ := residuals_eq_runO w cons (shape_claims hsh) hc hrun hb hw hk
  have e : rs' = rs := Except.ok.inj (hres'.symm.trans hres)
  rw [e] at hop
  exact ⟨sh, decodedO sh w cons, claimO sh w c, src, 0, k₁, rs, ρ, γ, hsh, hop, hz, hrel, hpre⟩

end Read

end FlockC3.Opening
