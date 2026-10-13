import Proofs.Flock.VBridge.Renumber
import Proofs.Flock.VBridge.Verifier
import Flock.RecResiduals

/-!
# The verifier's part of S keeps S's residuals

`RecResiduals.partS` builds a part's S from the whole S's terms, every lookup checked (`pos`). When it succeeds, the
part's residual `k` is the whole S's residual `R[k]` (`res_partS`), at part ports that read the whole ports where the
part's index says: its `w` element `k` is the whole `w`'s element `wIdx[k]`, its `v` element `k` is the whole `v`'s `V[k]`.

The maps `renS` renumbers by are the verifier's own lookups (`fw`, `fv`, `fo`), and `PartAgrees` follows from their
checks alone: the grouping that chose `R`, `P` and `V` is not read.
-/

namespace FlockVBridge.PartS

open Flock FlockVBridge.OfVStar
open Flock.RecResiduals (positions pos partS starts)

/-! ## The verifier's S read as V*'s -/

def vTerm : RecAlgebra.Term → VStar.Term
  | .w i => .w i
  | .v j => .v j
  | .wt i u => .wt i u

def vRes (r : RecAlgebra.Res) : VStar.Res := ⟨r.mul.toList, r.add.toList.map vTerm⟩

/-- The verifier's whole S (`RecAlgebra.structureOf`'s) read as V*'s. -/
def vS (S : RecAlgebra.GfS) : VStar.GfS :=
  ⟨S.nw, S.nv, S.ops.toList.map (·.toList.map vTerm), S.res.toList.map vRes⟩

theorem vS_ops_getD (S : RecAlgebra.GfS) {k : ℕ} (hk : k < S.ops.size) :
    (vS S).ops.getD k [] = S.ops[k].toList.map vTerm := by
  simp [vS, List.getD_eq_getElem?_getD, hk]

theorem vS_res_getD (S : RecAlgebra.GfS) {r : ℕ} (hr : r < S.res.size) :
    (vS S).res.getD r default = vRes S.res[r] := by
  simp [vS, List.getD_eq_getElem?_getD, hr]

/-! ## `partS`' pieces -/

/-- The part's `w` index: its row ports' elements of the whole `w`, in order. -/
def wIdx (sizes P : Array Nat) : Array Nat :=
  P.flatMap fun p => (Array.range sizes[p]!).map ((starts sizes)[p]! + ·)

/-- `partS`' lookup of one term into the part's ports. -/
def termE (S : RecAlgebra.GfS) (wI V : Array Nat) (t : RecAlgebra.Term) : Except String RecOpen.Term :=
  match t with
  | .w i => match pos wI (positions S.nw wI) i with
    | some i' => pure (.w i')
    | none => throw s!"[\"w\", {i}]: an element of a row port the part does not read"
  | .v j => match pos V (positions S.nv V) j with
    | some j' => pure (.v j')
    | none => throw s!"[\"v\", {j}]: a v element the part does not read"
  | .wt i u => match pos wI (positions S.nw wI) i with
    | some i' =>
      if u < 128 && (List.range 128).all fun j => wI[i' + j]? == some (i + j) then pure (.wt i' u)
      else throw s!"[\"wt\", {i}, {u}]: a transposed block over row ports the part does not read in a run"
    | none => throw s!"[\"wt\", {i}, {u}]: an element of a row port the part does not read"

/-- `partS`' operand `k`: S's operand, its terms looked up. -/
def opE (S : RecAlgebra.GfS) (wI V : Array Nat) (k : Nat) : Except String (List RecOpen.Term) :=
  if h : k < S.ops.size then S.ops[k].toList.mapM (termE S wI V) else throw s!"product operand {k} of {S.ops.size} operands"

/-- `partS`' residual `r`: S's residual, its operands and terms looked up. -/
def resE (S : RecAlgebra.GfS) (wI V used : Array Nat) (r : Nat) : Except String RecOpen.Res := do
  let some rr := S.res[r]? | throw s!"residual {r} of {S.res.size}"
  let mul ← rr.mul.toList.mapM fun (a, b) =>
    match pos used (positions S.ops.size used) a, pos used (positions S.ops.size used) b with
    | some a', some b' => pure (a', b')
    | _, _ => throw s!"product [{a}, {b}]: an operand the part does not hold"
  pure ({ mul, add := ← rr.add.toList.mapM (termE S wI V) } : RecOpen.Res)

/-- The maps the lookups define. -/
def fw (S : RecAlgebra.GfS) (wI : Array Nat) (i : ℕ) : ℕ := (pos wI (positions S.nw wI) i).getD 0
def fv (S : RecAlgebra.GfS) (V : Array Nat) (j : ℕ) : ℕ := (pos V (positions S.nv V) j).getD 0
def fo (S : RecAlgebra.GfS) (used : Array Nat) (k : ℕ) : ℕ := (pos used (positions S.ops.size used) k).getD 0

/-! ## Lemmas about `Except` -/

theorem bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem mapM_forall₂ {α β : Type} {f : α → Except String β} :
    ∀ {xs : List α} {ys : List β}, xs.mapM f = .ok ys → List.Forall₂ (fun a b => f a = .ok b) xs ys
  | [], ys, h => by
    rw [List.mapM_nil] at h
    cases h
    exact .nil
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := bind_ok h
    obtain ⟨bs, hbs, h⟩ := bind_ok h
    cases h
    exact .cons hb (mapM_forall₂ hbs)

theorem map_of_forall₂ {α β γ : Type} {P : α → β → Prop} {f : α → γ} {g : β → γ} (hfg : ∀ a b, P a b → g b = f a) :
    ∀ {as : List α} {bs : List β}, List.Forall₂ P as bs → bs.map g = as.map f
  | _, _, .nil => rfl
  | _, _, .cons h t => by simp only [List.map_cons, hfg _ _ h, map_of_forall₂ hfg t]

theorem exists_of_forall₂ {α β : Type} {P : α → β → Prop} :
    ∀ {as : List α} {bs : List β}, List.Forall₂ P as bs → ∀ a ∈ as, ∃ b, P a b
  | _, _, .nil, _, ha => by cases ha
  | _, _, .cons h t, a, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨_, h⟩
    · exact exists_of_forall₂ t a ha

theorem length_of_forall₂ {α β : Type} {P : α → β → Prop} :
    ∀ {as : List α} {bs : List β}, List.Forall₂ P as bs → bs.length = as.length
  | _, _, .nil => rfl
  | _, _, .cons _ t => by simp only [List.length_cons, length_of_forall₂ t]

theorem getElem_of_forall₂ {α β : Type} {P : α → β → Prop} :
    ∀ {as : List α} {bs : List β} (h : List.Forall₂ P as bs) (k : ℕ) (hk : k < bs.length),
      P (as[k]'(length_of_forall₂ h ▸ hk)) bs[k]
  | _, _, .nil, _, hk => by simp at hk
  | _, _, .cons h t, 0, _ => h
  | _, _, .cons _ t, k + 1, hk => getElem_of_forall₂ t k (by simpa using hk)

/-! ## The checked lookups -/

theorem pos_some {xs : Array Nat} {ps : Array (Option Nat)} {k p : Nat} (h : pos xs ps k = some p) :
    ∃ hp : p < xs.size, xs[p] = k := by
  unfold pos at h
  split at h
  · split at h
    · cases h
      rename_i hx
      exact Array.getElem?_eq_some_iff.1 hx
    · cases h
  · cases h

section ports

variable {w v w' v' : ℕ → F128} {S : RecAlgebra.GfS} {wI V : Array Nat}

/-- **A looked-up term reads what S's term read**, at part ports `w'`, `v'` that hold the whole ports' elements at `wI`
and `V`. -/
theorem termE_ok (hw : ∀ k (hk : k < wI.size), w' k = w wI[k]) (hv : ∀ k (hk : k < V.size), v' k = v V[k])
    {t : RecAlgebra.Term} {b : RecOpen.Term} (h : termE S wI V t = .ok b) :
    termOf b = term (renTerm (fw S wI) (fv S V) (vTerm t)) ∧ TermAgrees w v w' v' (fw S wI) (fv S V) (vTerm t) := by
  cases t with
  | w i =>
    simp only [termE] at h
    split at h
    · rename_i i' hp
      cases h
      obtain ⟨hlt, hx⟩ := pos_some hp
      have hf : fw S wI i = i' := by simp [fw, hp]
      refine ⟨by simp [termOf, term, renTerm, vTerm, hf], ?_⟩
      show w' (fw S wI i) = w i
      rw [hf, hw i' hlt, hx]
    · cases h
  | v j =>
    simp only [termE] at h
    split at h
    · rename_i j' hp
      cases h
      obtain ⟨hlt, hx⟩ := pos_some hp
      have hf : fv S V j = j' := by simp [fv, hp]
      refine ⟨by simp [termOf, term, renTerm, vTerm, hf], ?_⟩
      show v' (fv S V j) = v j
      rw [hf, hv j' hlt, hx]
    · cases h
  | wt i u =>
    simp only [termE] at h
    split at h
    · rename_i i' hp
      have hf : fw S wI i = i' := by simp [fw, hp]
      split at h
      · rename_i hc
        cases h
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range, beq_iff_eq] at hc
        obtain ⟨hu, hb⟩ := hc
        refine ⟨by simp [termOf, term, renTerm, vTerm, hf], hu, fun j hj => ?_⟩
        rw [hf]
        obtain ⟨hlt, hx⟩ := Array.getElem?_eq_some_iff.1 (hb j hj)
        rw [hw _ hlt, hx]
      · cases h
    · cases h

theorem termsE_ok (hw : ∀ k (hk : k < wI.size), w' k = w wI[k]) (hv : ∀ k (hk : k < V.size), v' k = v V[k])
    {ts : List RecAlgebra.Term} {bs : List RecOpen.Term} (h : ts.mapM (termE S wI V) = .ok bs) :
    bs.map termOf = ts.map (fun t => term (renTerm (fw S wI) (fv S V) (vTerm t))) ∧
      ∀ t ∈ ts, TermAgrees w v w' v' (fw S wI) (fv S V) (vTerm t) := by
  have h2 := mapM_forall₂ h
  refine ⟨map_of_forall₂ (g := termOf) (f := fun t => term (renTerm (fw S wI) (fv S V) (vTerm t)))
    (fun _ _ hb => (termE_ok hw hv hb).1) h2, fun t ht => ?_⟩
  obtain ⟨b, hb⟩ := exists_of_forall₂ h2 t ht
  exact (termE_ok hw hv hb).2

/-- What a looked-up residual says: `r` is one of S's, it is S's renumbered by the lookups, every operand of its products
is held where `fo` says, and each added term reads what S's read. -/
theorem resE_ok (hw : ∀ k (hk : k < wI.size), w' k = w wI[k]) (hv : ∀ k (hk : k < V.size), v' k = v V[k])
    {used : Array Nat} {r : ℕ} {b : RecOpen.Res} (h : resE S wI V used r = .ok b) :
    ∃ hr : r < S.res.size,
      resOf b = res (renRes (fw S wI) (fv S V) (fo S used) (vRes S.res[r])) ∧
      (∀ ab ∈ S.res[r].mul.toList, ∀ k ∈ [ab.1, ab.2], ∃ p, pos used (positions S.ops.size used) k = some p) ∧
      ∀ t ∈ S.res[r].add.toList, TermAgrees w v w' v' (fw S wI) (fv S V) (vTerm t) := by
  unfold resE at h
  split at h
  · rename_i rr hrr
    obtain ⟨hr, hrr⟩ := Array.getElem?_eq_some_iff.1 hrr
    subst hrr
    obtain ⟨mul, hmul, h⟩ := bind_ok h
    obtain ⟨add, hadd, h⟩ := bind_ok h
    cases h
    have hm := mapM_forall₂ hmul
    have hpair : ∀ (ab : ℕ × ℕ) (m : ℕ × ℕ),
        (match ab with
          | (a, b) => match pos used (positions S.ops.size used) a, pos used (positions S.ops.size used) b with
            | some a', some b' => (pure (a', b') : Except String (ℕ × ℕ))
            | _, _ => throw s!"product [{a}, {b}]: an operand the part does not hold") = .ok m →
        pos used (positions S.ops.size used) ab.1 = some m.1 ∧ pos used (positions S.ops.size used) ab.2 = some m.2 := by
      intro ab m hab
      obtain ⟨a, b⟩ := ab
      simp only at hab
      split at hab
      · rename_i a' b' ha hb
        cases hab
        exact ⟨ha, hb⟩
      · cases hab
    obtain ⟨hadd1, hadd2⟩ := termsE_ok hw hv hadd
    refine ⟨hr, ?_, fun ab hab k hk => ?_, hadd2⟩
    · simp only [resOf, res, renRes, vRes, List.map_map]
      congr 1
      · rw [← List.map_id mul]
        refine map_of_forall₂ (g := id) (f := fun ab => (fo S used ab.1, fo S used ab.2)) (fun ab m hab => ?_) hm
        obtain ⟨h1, h2⟩ := hpair ab m hab
        simp [fo, h1, h2]
    · obtain ⟨m, hab'⟩ := exists_of_forall₂ hm ab hab
      obtain ⟨h1, h2⟩ := hpair ab m hab'
      rcases List.mem_pair.1 hk with rfl | rfl
      · exact ⟨_, h1⟩
      · exact ⟨_, h2⟩
  · cases h

end ports

/-! ## The part -/

/-- **The verifier's part keeps S's residuals.** When `partS` builds a part `S'` from S's residuals `R` and the ports
`P` and `V`, `S'` has a residual for each of `R`, and at part ports `w'`, `v'` holding the whole ports' elements where
the part's index says (`wIdx`, `V`), the part's residual `k` has S's residual `R[k]`'s value at `w`, `v`. -/
theorem res_partS {S : RecAlgebra.GfS} {sizes R P V : Array Nat} {S' : RecOpen.GfS}
    (h : partS S sizes R P V = .ok S') {w v w' v' : ℕ → F128}
    (hw : ∀ k (hk : k < (wIdx sizes P).size), w' k = w (wIdx sizes P)[k])
    (hv : ∀ k (hk : k < V.size), v' k = v V[k]) :
    ∃ hn : S'.res.length = R.size, ∀ k (hk : k < S'.res.length),
      resVal (gfsOf S') (termVal w' v') (resOf S'.res[k]) =
        resVal (gfS (vS S)) (termVal w v) (res ((vS S).res.getD (R[k]'(hn ▸ hk)) default)) := by
  unfold partS at h
  obtain ⟨used, -, h⟩ := bind_ok h
  obtain ⟨ops, hops, h⟩ := bind_ok h
  obtain ⟨rs, hrs, h⟩ := bind_ok h
  cases h
  have hops' : used.toList.mapM (opE S (wIdx sizes P) V) = .ok ops := hops
  have hrs' : R.toList.mapM (resE S (wIdx sizes P) V used) = .ok rs := hrs
  have ho := mapM_forall₂ hops'
  have hr := mapM_forall₂ hrs'
  have hn : rs.length = R.size := by simpa using length_of_forall₂ hr
  -- every operand `partS` holds is one of S's, its terms looked up
  have hop : ∀ k ∈ used.toList, ∃ hk : k < S.ops.size, ∃ o,
      S.ops[k].toList.mapM (termE S (wIdx sizes P) V) = .ok o := by
    intro k hk
    obtain ⟨o, hko⟩ := exists_of_forall₂ ho k hk
    unfold opE at hko
    split at hko
    · exact ⟨‹_›, o, hko⟩
    · cases hko
  have hpart : PartAgrees (vS S) R.toList used.toList (fw S (wIdx sizes P)) (fv S V) (fo S used) w v w' v' := by
    refine ⟨fun r hrm ab hab k hk => ?_, fun r hrm t ht => ?_, fun r hrm ab hab k hk t ht => ?_⟩
    all_goals
      obtain ⟨b, hb⟩ := exists_of_forall₂ hr r hrm
      obtain ⟨hlt, -, hops2, hadd2⟩ := resE_ok hw hv hb
      rw [vS_res_getD S hlt] at *
    · obtain ⟨p, hp⟩ := hops2 ab hab k hk
      obtain ⟨hpl, hpx⟩ := pos_some hp
      have hf : fo S used k = p := by simp [fo, hp]
      exact ⟨by simpa [hf] using hpl, by simp [hf, hpx]⟩
    · simp only [vRes, List.mem_map] at ht
      obtain ⟨t0, ht0, rfl⟩ := ht
      exact hadd2 t0 ht0
    · obtain ⟨p, hp⟩ := hops2 ab hab k hk
      obtain ⟨hpl, hpx⟩ := pos_some hp
      have hku : k ∈ used.toList := by
        rw [Array.mem_toList_iff]
        exact hpx ▸ Array.getElem_mem hpl
      obtain ⟨hk', o, hko⟩ := hop k hku
      rw [vS_ops_getD S hk', List.mem_map] at ht
      obtain ⟨t0, ht0, rfl⟩ := ht
      exact (termsE_ok hw hv hko).2 t0 ht0
  have hS : gfsOf ⟨ops, rs⟩ =
      gfS (renS (vS S) S.nw S.nv R.toList used.toList (fw S (wIdx sizes P)) (fv S V) (fo S used)) := by
    simp only [gfsOf, gfS, renS, List.map_map]
    congr 1
    · refine map_of_forall₂ (fun k o hko => ?_) ho
      unfold opE at hko
      split at hko
      · rename_i hk
        rw [(termsE_ok hw hv hko).1, Function.comp_apply, vS_ops_getD S hk, List.map_map, List.map_map]
        rfl
      · cases hko
    · refine map_of_forall₂ (fun r b hb => ?_) hr
      obtain ⟨hlt, hres, -, -⟩ := resE_ok hw hv hb
      rw [hres, Function.comp_apply, vS_res_getD S hlt]
  refine ⟨hn, fun k hk => ?_⟩
  have hkR : k < R.toList.length := by simpa [hn] using hk
  obtain ⟨hlt, hres, -, -⟩ := resE_ok hw hv (getElem_of_forall₂ hr k hk)
  simp only [Array.getElem_toList] at hlt hres
  rw [hS, hres, vS_res_getD S hlt]
  have hmem : R[k] ∈ R.toList := Array.mem_toList_iff.2 (Array.getElem_mem _)
  rw [← vS_res_getD S hlt]
  exact resVal_renS hpart hmem

end FlockVBridge.PartS
