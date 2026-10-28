import Proofs.Flock.Claims.Sampling.Select.Misses

/-!
# The design the draw runs is well formed (`Properties.SampledProofs.DesignWellFormed`)

* `parseLaw_k`: a law `parseLaw` writes names a `k` of at least one, where it names one.
* `derive_bounds`, `deriveWork_wf`, `workTable_floor`: the expansions' refusals: `1 ≤ k ≤ N`, a floor of at least one
  per stratum, and some work.
* `strataOf_pos`, `partition_covered`: every stratum of an evaluated query has a unit: `TemplateQuery.evaluate` refuses
  a template with no instance of one, and `callRanges` skips a Call with none.
* `designWellFormed`.
-/

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json)
open Claims.Sampling

theorem nat_toJson (m : ℕ) : nat? (Lean.toJson m) = some m := by
  simp [nat?, int?, Lean.toJson, Lean.JsonNumber.fromNat]

theorem getD_k (v : Json) (m : ℕ) : getD (Json.mkObj [("law", v), ("k", Lean.toJson m)]) "k" = Lean.toJson m := rfl

theorem getD_p (v : Json) : getD (Json.mkObj [("law", "bernoulli"), ("p", v)]) "k" = Json.null := rfl

/-- **A law `parseLaw` writes names `k ≥ 1`**, where it names a `k`: `subset:K`, `stratified:K` and `work:K` at
`K ≥ 1`, and a Bernoulli law none. -/
theorem parseLaw_k (t : String) (named : Json) (h : Select.parseLaw t = .ok named) (k : ℕ)
    (hk : nat? (getD named "k") = some k) : 1 ≤ k := by
  unfold Select.parseLaw at h
  repeat' split at h
  all_goals first
    | exact absurd h (throw_ne _ _)
    | exact absurd h (throw_bind_ne _ _ _)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       first
         | (rw [getD_p] at hk; simp [nat?, int?] at hk)
         | (rw [getD_k, nat_toJson, Option.some.injEq] at hk; subst hk; simp_all))

/-- **`Flock.Draw.derive`'s bounds**: `1 ≤ k ≤ N`. -/
theorem derive_bounds (k : ℕ) (S : Array (String × Array (ℕ × ℕ))) (law' : Flock.Draw.Law)
    (h : Flock.Draw.derive k S = .ok law') : 1 ≤ k ∧ k ≤ strataSize S := by
  unfold Flock.Draw.derive at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  · rename_i hc
    simp only [size_rs, size_strata, Bool.or_eq_true, decide_eq_true_eq, not_or, not_lt] at hc
    omega

/-- **`Flock.Draw.deriveWork`'s law, with its refusals**: each stratum its ranges and a floor from the table, the
population doing some work. -/
theorem deriveWork_wf (k : ℕ) (S : Array (String × Array (ℕ × ℕ))) (tab : Array (String × ℕ × ℕ))
    (law' : Flock.Draw.Law) (h : Flock.Draw.deriveWork k S tab = .ok law') :
    ∃ ws : Array Flock.Draw.WorkStratum,
      List.Forall₂ (fun p (w : Flock.Draw.WorkStratum) => w.units = p.2 ∧ ∃ e ∈ tab, w.floor = e.2.2) S.toList
        ws.toList ∧ Flock.Draw.totalWork ws ≠ 0 ∧
      law' = .work k (ws.map fun s => { s with k := Flock.Draw.workK k s.work s.size (Flock.Draw.totalWork ws) s.floor }) := by
  unfold Flock.Draw.deriveWork at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  · obtain ⟨ws, hws, h⟩ := bind_eq_ok.1 h
    refine ⟨ws, (mapM_ok _ _ _ hws).imp fun {p w} hp => ?_, ?_⟩
    · obtain ⟨t, rs⟩ := p
      split at hp
      · rename_i e he
        obtain rfl := (Except.ok.inj hp).symm
        exact ⟨rfl, _, Array.mem_of_find?_eq_some he, rfl⟩
      · exact absurd hp (throw_ne _ _)
    · split at h
      · exact absurd h (throw_bind_ne _ _ _)
      · rename_i hc
        exact ⟨by simpa using hc, (Except.ok.inj h).symm⟩

theorem forall₂_mem_right {α β : Type} {R : α → β → Prop} :
    ∀ {xs : List α} {ys : List β}, List.Forall₂ R xs ys → ∀ y ∈ ys, ∃ x ∈ xs, R x y
  | _, _, .nil, _, hy => absurd hy (by simp)
  | x :: _, _ :: _, .cons hxy hs, y, hy => by
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨x, List.mem_cons_self, hxy⟩
    · obtain ⟨x', hx', h'⟩ := forall₂_mem_right hs y hy
      exact ⟨x', List.mem_cons_of_mem _ hx', h'⟩

/-- **A work table's floors are at least one**. -/
theorem workTable_floor (j : Json) (tab : Array (String × ℕ × ℕ)) (h : Flock.Draw.workTableOfJson j = .ok tab) :
    ∀ e ∈ tab, 1 ≤ e.2.2 := by
  unfold Flock.Draw.workTableOfJson at h
  split at h
  · intro e he
    obtain ⟨⟨t, v⟩, -, hx⟩ := forall₂_mem_right (mapM_ok _ _ _ h) e (Array.mem_toList_iff.2 he)
    dsimp only at hx
    repeat' split at hx
    all_goals first
      | exact absurd hx (throw_ne _ _)
      | exact absurd hx (throw_bind_ne _ _ _)
      | (simp only [pure, Except.pure, Except.ok.injEq] at hx; subst hx; dsimp only; omega)
  · exact absurd h (throw_ne _ _)

/-! ## Every stratum has a unit -/

/-- Each template ranges name has a range of at least one unit. -/
def Covered (L : List (String × ℕ × ℕ)) : Prop := ∀ x ∈ L, ∃ y ∈ L, y.1 = x.1 ∧ 0 < y.2.2

/-- `strataOf`'s invariant over the ranges `seen` so far: one stratum per template, each a template seen and each
template seen a stratum's; each range `lo ≤ hi`; and a stratum whose template a range of some units has has a non-empty
range. -/
def PosInv (out : Array (String × Array (ℕ × ℕ))) (seen : List (String × ℕ × ℕ)) : Prop :=
  (out.toList.map (·.1)).Nodup ∧ (∀ p ∈ out.toList, ∀ r ∈ p.2.toList, r.1 ≤ r.2) ∧
    (∀ p ∈ out.toList, ∃ x ∈ seen, x.1 = p.1) ∧ (∀ x ∈ seen, ∃ p ∈ out.toList, p.1 = x.1) ∧
    ∀ p ∈ out.toList, ∀ x ∈ seen, x.1 = p.1 → 0 < x.2.2 → ∃ r ∈ p.2.toList, r.1 < r.2

theorem strataStep_pos (out : Array (String × Array (ℕ × ℕ))) (seen : List (String × ℕ × ℕ)) (t : String)
    (lo n : ℕ) (h : PosInv out seen) : PosInv (strataStep out (t, lo, n)) (seen ++ [(t, lo, n)]) := by
  obtain ⟨hnd, hle, hseen, hcov, hpos⟩ := h
  unfold strataStep
  dsimp only
  split
  · rename_i hnone
    have hnot : ∀ p ∈ out.toList, p.1 ≠ t := by
      intro p hp hpt
      have := Array.findIdx?_eq_none_iff.1 hnone p (Array.mem_toList_iff.1 hp)
      simp [hpt] at this
    refine ⟨?_, fun p hp r hr => ?_, fun p hp => ?_, fun x hx => ?_, fun p hp x hx hxp hn => ?_⟩
    · rw [Array.toList_push, List.map_append, List.nodup_append]
      refine ⟨hnd, List.nodup_singleton _, ?_⟩
      intro a ha b hb
      simp only [List.map_cons, List.map_nil, List.mem_singleton] at hb
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 ha
      rw [hb]
      exact hnot p hp
    · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
      rcases hp with hp | rfl
      · exact hle p hp r hr
      · simp only [List.mem_singleton] at hr
        subst hr
        dsimp only
        omega
    · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
      rcases hp with hp | rfl
      · obtain ⟨x, hx, hxp⟩ := hseen p hp
        exact ⟨x, List.mem_append_left _ hx, hxp⟩
      · exact ⟨(t, lo, n), by simp, rfl⟩
    · rw [List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · obtain ⟨p, hp, hpx⟩ := hcov x hx
        exact ⟨p, by simp [hp], hpx⟩
      · exact ⟨(t, #[(lo, lo + n)]), by simp, rfl⟩
    · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
      rw [List.mem_append, List.mem_singleton] at hx
      rcases hp with hp | rfl
      · rcases hx with hx | rfl
        · exact hpos p hp x hx hxp hn
        · exact absurd hxp.symm (hnot p hp)
      · rcases hx with hx | rfl
        · obtain ⟨q, hq, hqx⟩ := hcov x hx
          exact absurd (hqx.trans hxp) (hnot q hq)
        · exact ⟨(lo, lo + n), by simp, by dsimp only at hn ⊢; omega⟩
  · rename_i i hi
    have hlt : i < out.size := (Array.findIdx?_eq_some_iff_getElem.1 hi).1
    have hti : out[i].1 = t := by
      obtain ⟨_, h2, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hi
      simpa using h2
    set rs := out[i]!.2 with hrs
    have hrs' : rs = out[i].2 := by rw [hrs, getElem!_pos out i hlt]
    have hmi : out[i] ∈ out.toList := Array.getElem_mem_toList hlt
    have hold : ∀ r ∈ rs.toList, r.1 ≤ r.2 := fun r hr' => hle _ hmi r (by rw [← hrs']; exact hr')
    set rs' := (match rs.back? with
        | some (a', b) => if (b == lo) = true then rs.pop.push (a', lo + n) else rs.push (lo, lo + n)
        | none => rs.push (lo, lo + n)) with hrs''
    have hnew : (∀ r ∈ rs'.toList, r.1 ≤ r.2) ∧ (0 < n → ∃ r ∈ rs'.toList, r.1 < r.2) ∧
        ((∃ r ∈ rs.toList, r.1 < r.2) → ∃ r ∈ rs'.toList, r.1 < r.2) := by
      have hpush : (∀ r ∈ (rs.push (lo, lo + n)).toList, r.1 ≤ r.2) ∧
          (0 < n → ∃ r ∈ (rs.push (lo, lo + n)).toList, r.1 < r.2) ∧
          ((∃ r ∈ rs.toList, r.1 < r.2) → ∃ r ∈ (rs.push (lo, lo + n)).toList, r.1 < r.2) := by
        refine ⟨fun r hr' => ?_, fun hn => ⟨(lo, lo + n), by simp, by dsimp only; omega⟩,
          fun ⟨r, hr', hlt'⟩ => ⟨r, by simp [hr'], hlt'⟩⟩
        simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr'
        rcases hr' with hr' | rfl
        · exact hold r hr'
        · dsimp only; omega
      rw [hrs'']
      split
      · rename_i a' b hb
        have hback : rs.toList = rs.pop.toList ++ [(a', b)] := by
          rw [Array.back?_eq_some_iff] at hb
          obtain ⟨ys, hys⟩ := hb
          rw [hys]; simp
        have hab : a' ≤ b := hold (a', b) (by rw [hback]; simp)
        split
        · rename_i hba
          have hba : b = lo := by simpa using hba
          subst hba
          refine ⟨fun r hr' => ?_, fun hn => ⟨(a', b + n), by simp, by dsimp only; omega⟩,
            fun ⟨r, hr', hlt'⟩ => ?_⟩
          · simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hr'
            rcases hr' with hr' | rfl
            · exact hold r (by rw [hback]; exact List.mem_append_left _ hr')
            · dsimp only; omega
          · rw [hback, List.mem_append, List.mem_singleton] at hr'
            rcases hr' with hr' | rfl
            · exact ⟨r, by rw [Array.toList_push]; exact List.mem_append_left _ hr', hlt'⟩
            · exact ⟨(a', b + n), by simp, by dsimp only at hlt' ⊢; omega⟩
        · exact hpush
      · exact hpush
    have hset : (out.set! i (t, rs')).toList = out.toList.set i (t, rs') := by
      rw [Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds]
    have hlt' : i < out.toList.length := by rw [Array.length_toList]; exact hlt
    have hmap : (out.toList.set i (t, rs')).map (·.1) = out.toList.map (·.1) := by
      rw [List.map_set]
      refine List.ext_getElem (by simp) fun j h1 h2 => ?_
      rw [List.getElem_set]
      split
      · rename_i hij
        subst hij
        simp [hti]
      · rfl
    have hmem_new : (t, rs') ∈ out.toList.set i (t, rs') := List.mem_set hlt' _
    have huniq : ∀ p ∈ out.toList.set i (t, rs'), p.1 = t → p = (t, rs') := fun p hp hpt =>
      List.inj_on_of_nodup_map (hmap ▸ hnd) hp hmem_new hpt
    have hold_mem : ∀ p ∈ out.toList.set i (t, rs'), p.1 ≠ t → p ∈ out.toList := fun p hp hpt =>
      (List.mem_or_eq_of_mem_set hp).resolve_right (by rintro rfl; exact hpt rfl)
    have hkeep : ∀ p ∈ out.toList, p.1 ≠ t → p ∈ out.toList.set i (t, rs') := by
      intro p hp hpt
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hp
      have hji : i ≠ j := by
        rintro rfl
        exact hpt (by rw [Array.getElem_toList]; exact hti)
      rw [← List.getElem_set_ne hji (a := (t, rs')) (by simpa using hj)]
      exact List.getElem_mem _
    show PosInv (out.set! i (t, rs')) (seen ++ [(t, lo, n)])
    unfold PosInv
    rw [hset]
    refine ⟨hmap ▸ hnd, fun p hp r hr => ?_, fun p hp => ?_, fun x hx => ?_, fun p hp x hx hxp hn => ?_⟩
    · by_cases hpt : p.1 = t
      · obtain rfl := huniq p hp hpt
        exact hnew.1 r hr
      · exact hle p (hold_mem p hp hpt) r hr
    · by_cases hpt : p.1 = t
      · exact ⟨(t, lo, n), by simp, hpt.symm⟩
      · obtain ⟨x, hx, hxp⟩ := hseen p (hold_mem p hp hpt)
        exact ⟨x, List.mem_append_left _ hx, hxp⟩
    · rw [List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · obtain ⟨p, hp, hpx⟩ := hcov x hx
        by_cases hpt : p.1 = t
        · exact ⟨(t, rs'), hmem_new, hpt ▸ hpx⟩
        · exact ⟨p, hkeep p hp hpt, hpx⟩
      · exact ⟨(t, rs'), hmem_new, rfl⟩
    · rw [List.mem_append, List.mem_singleton] at hx
      by_cases hpt : p.1 = t
      · obtain rfl := huniq p hp hpt
        rcases hx with hx | rfl
        · exact hnew.2.2 (hrs' ▸ hpos _ hmi x hx (hxp.trans hti.symm) hn)
        · exact hnew.2.1 hn
      · have hp' := hold_mem p hp hpt
        rcases hx with hx | rfl
        · exact hpos p hp' x hx hxp hn
        · exact absurd hxp.symm hpt

/-- **Every stratum `strataOf` makes has a unit**, over ranges each of whose templates has a range of some units. -/
theorem strataOf_pos (L : Array (String × ℕ × ℕ)) (hc : Covered L.toList) :
    ∀ p ∈ (Flock.Draw.strataOf L).toList, 1 ≤ rsSize p.2 := by
  rw [strataOf_eq]
  have key : ∀ (l : List (String × ℕ × ℕ)) (out : Array (String × Array (ℕ × ℕ))) (seen : List (String × ℕ × ℕ)),
      PosInv out seen → PosInv (l.foldl strataStep out) (seen ++ l) := by
    intro l
    induction l with
    | nil => intro out seen h; simpa using h
    | cons x xs ih =>
      intro out seen h
      obtain ⟨t, lo, n⟩ := x
      rw [List.foldl_cons, show seen ++ (t, lo, n) :: xs = (seen ++ [(t, lo, n)]) ++ xs by simp]
      exact ih _ _ (strataStep_pos out seen t lo n h)
  obtain ⟨-, -, hseen, -, hpos⟩ := key L.toList #[] [] ⟨by simp, by simp, by simp, by simp, by simp⟩
  intro p hp
  rw [List.nil_append] at hseen hpos
  obtain ⟨x, hx, hxp⟩ := hseen p hp
  obtain ⟨y, hy, hyx, hy0⟩ := hc x hx
  obtain ⟨r, hr, hlt⟩ := hpos p hp y hy (hyx.trans hxp) hy0
  have : r.1 ∈ expand p.2.toList := List.mem_flatMap.2 ⟨r, hr, List.mem_range'.2 ⟨0, by omega, by simp⟩⟩
  exact List.length_pos_of_mem this

/-- **A template query's ranges cover their templates**: each node's template is one the query names, and the query
refuses a template with no instance of a unit. -/
theorem evaluate_covered (d : Flock.TemplateQuery.Desc) (ts : Array String) (rs : Array Flock.TemplateQuery.TRange)
    (h : Flock.TemplateQuery.evaluate d ts = .ok rs) :
    ∀ r ∈ rs.toList, ∃ r' ∈ rs.toList, r'.template = r.template ∧ 0 < r'.count := by
  unfold Flock.TemplateQuery.evaluate at h
  split at h
  · rename_i root _
    dsimp only at h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
    cases hl : forIn (m := Except String) (List.range' ([:root.nodes.size] : Std.Legacy.Range).start
        ([:root.nodes.size] : Std.Legacy.Range).size ([:root.nodes.size] : Std.Legacy.Range).step)
        ((#[] : Array Flock.TemplateQuery.TRange), 0) _ with
    | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨ranges, at_⟩ := y
      rw [hl, bind_ok] at h
      dsimp only at h
      have hinv : ∀ r ∈ ranges.toList, ts.contains r.template = true :=
        forIn_inv (fun (b : Array Flock.TemplateQuery.TRange × ℕ) => ∀ r ∈ b.1.toList, ts.contains r.template = true) _
          (fun k b hb s hs => ?_) _ ((#[] : Array Flock.TemplateQuery.TRange), 0) (by simp) _ hl
      · split at h
        · exact absurd h (throw_ne _ _)
        · rename_i hc
          obtain rfl := (Except.ok.inj h).symm
          intro r hr
          have hmiss : r.template ∉ ts.filter fun t => !rs.any fun r => r.template == t && r.count > 0 := by
            intro hm
            simp only [Bool.not_eq_true', Bool.not_eq_false] at hc
            rw [Array.isEmpty_iff] at hc
            rw [hc] at hm
            simp at hm
          rw [Array.mem_filter] at hmiss
          have hany : (rs.any fun r' => r'.template == r.template && r'.count > 0) = true := by
            by_contra hne
            exact hmiss ⟨Array.contains_iff_mem.1 (hinv r hr), by simpa using hne⟩
          obtain ⟨r', hr', hc'⟩ := Array.any_eq_true'.1 hany
          simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc'
          exact ⟨r', Array.mem_toList_iff.2 hr', hc'⟩
      · obtain ⟨ranges₀, at₀⟩ := b
        dsimp only at hs hb
        split at hs
        · obtain rfl := Except.ok.inj hs; exact hb
        · split at hs
          · exact absurd hs (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
          · rename_i hc
            obtain rfl := Except.ok.inj hs
            simp only [ForInStep.value, Array.toList_push, List.mem_append, List.mem_singleton]
            rintro r (hr | rfl)
            · exact hb r hr
            · simp only [Bool.or_eq_true, Bool.not_eq_true', not_or, Bool.not_eq_false] at hc
              exact hc.2
  · exact absurd h (throw_ne _ _)

theorem callStep_pos : ∀ (l : List (ℕ × String × ℕ)) (acc : Array Flock.TemplateQuery.TRange × ℕ),
    (∀ r ∈ acc.1.toList, 0 < r.count) → ∀ r ∈ (l.foldl callStep acc).1.toList, 0 < r.count
  | [], _, h => h
  | x :: l, acc, h => by
    rw [List.foldl_cons]
    refine callStep_pos l _ ?_
    unfold callStep
    split
    · exact h
    · rename_i hx
      simp only [Array.toList_push, List.mem_append, List.mem_singleton]
      rintro r (hr | rfl)
      · exact h r hr
      · simp only [beq_iff_eq] at hx
        exact Nat.pos_of_ne_zero hx

/-- **The query's ranges cover their templates**: a template query's (`evaluate_covered`), and a Call query's, whose
ranges each have a unit. -/
theorem partition_covered (d : Program) (query : Lean.Json) (rs : Array Flock.TemplateQuery.TRange)
    (h : Partition.evaluate d query = .ok rs) :
    Covered (rs.map fun r => (r.template, r.first, r.count)).toList := by
  suffices hc : ∀ r ∈ rs.toList, ∃ r' ∈ rs.toList, r'.template = r.template ∧ 0 < r'.count by
    intro x hx
    rw [Array.toList_map, List.mem_map] at hx
    obtain ⟨r, hr, rfl⟩ := hx
    obtain ⟨r', hr', ht, hn⟩ := hc r hr
    exact ⟨(r'.template, r'.first, r'.count), by rw [Array.toList_map]; exact List.mem_map_of_mem hr', ht, hn⟩
  unfold Partition.evaluate at h
  dsimp only at h
  split at h
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
  · split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · split at h
      · rename_i ts _
        cases hts : ts with
        | error e => rw [hts] at h; exact absurd h (by rw [bind_error]; simp)
        | ok ts' =>
          rw [hts, bind_ok] at h
          obtain ⟨rs', hrs', h⟩ := bind_eq_ok.1 h
          obtain ⟨_, -, h⟩ := bind_eq_ok.1 h
          obtain rfl := (Except.ok.inj h).symm
          exact evaluate_covered d.desc ts' _ hrs'
      · obtain ⟨cs, _, h⟩ := bind_eq_ok.1 h
        obtain rfl := (Except.ok.inj h).symm
        have heq : Partition.callRanges cs = (cs.toList.foldl callStep (#[], 0)).1 := by
          unfold Partition.callRanges; rw [Array.foldl_toList]; rfl
        intro r hr
        rw [heq] at hr ⊢
        exact ⟨r, hr, rfl, callStep_pos _ _ (by simp) r hr⟩

/-! ## The design -/

/-- **The population's first stage is well formed over its strata**: the law the draw samples, over strata of sizes
`ns` summing to `n`, each of at least one unit. -/
theorem first_wf (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) (n : ℕ) (named : Json)
    (derived : Option Flock.Draw.Law) (parsed : Flock.Draw.Law) (S : Array (String × Array (ℕ × ℕ)))
    (hS : Partition.strata d query = .ok S) (hne : rule ≠ .all) (hnamed : rule.lawOver n = .ok named)
    (hd : Draw.deriveLaw d query named table = .ok derived) (hp : Draw.flockLaw (lawObj named derived) n = .ok parsed)
    (hsum : strataSize S = n) (hpos : ∀ m ∈ S.toList.map (fun p => rsSize p.2), 1 ≤ m) :
    (oneStageOf (derived.getD parsed)).WellFormed (S.toList.map fun p => rsSize p.2) := by
  have hsum' : (S.toList.map fun p => rsSize p.2).sum = n := hsum
  rcases deriveLaw_cases d query named table derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S', k, law', hS', rfl, hlaw⟩
  · have hnamed' : ∃ t, Select.parseLaw t = .ok named := by
      cases rule with
      | all => exact absurd rfl hne
      | law t => exact ⟨t, hnamed⟩
    obtain ⟨t, ht⟩ := hnamed'
    rcases flockLaw_cases _ _ _ hp with ⟨k, rfl, hk, hnat⟩ | ⟨a, b, rfl, hb, hab⟩ | hk | hk
    · exact ⟨parseLaw_k t named ht k hnat, by rw [hsum']; exact hk⟩
    · exact ⟨hb, hab⟩
    · exact absurd hk hk1
    · exact absurd hk hk2
  · rw [hS] at hS'
    obtain rfl := Except.ok.inj hS'
    rcases hlaw with hder | ⟨t, tab, ht, hwork⟩
    · obtain ⟨hk1, hkn⟩ := derive_bounds k _ law' hder
      obtain rfl := derive_ok k _ law' hder
      exact ⟨hk1, by rw [hsum']; rw [hsum] at hkn; exact hkn, hpos⟩
    · obtain ⟨ws, hF, htot, rfl⟩ := deriveWork_wf k _ tab law' hwork
      have hunits := units_of_forall₂ _ _ (hF.imp fun _ _ h => h.1)
      have hsz : S.toList.map (fun p => rsSize p.2) = ws.toList.map (·.size) := by
        have := congrArg (List.map rsSize) hunits
        simp only [List.map_map, Function.comp_def] at this
        rw [← this]
        simp [stratum_size]
      have hlen : ws.toList.length = S.toList.length := hF.length_eq.symm
      simp only [Option.getD_some, oneStageOf, OneStage.WellFormed, Array.toList_map, List.map_map,
        Function.comp_def, List.length_map]
      refine ⟨by rw [hlen], by rw [hlen], fun x hx => ?_, ?_, hpos⟩
      · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hx
        obtain ⟨p, -, -, e, he, hfe⟩ := forall₂_mem_right hF w hw
        rw [hfe]
        exact workTable_floor _ tab ht e he
      · rw [hsz, List.zip_map', List.map_map]
        rw [totalWork_eq] at htot
        exact Nat.pos_of_ne_zero (by simpa [Function.comp_def] using htot)

/-- **The design the draw runs is well formed on the partition it runs over**
(`Properties.SampledProofs.DesignWellFormed`). -/
theorem designWellFormed : DesignWellFormed := by
  intro rule d query table l D hl hD
  obtain ⟨rs, he, pop, rfl, hlays⟩ := layout_ok d query l hl
  set S := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) with hS
  have hperm : (popUnits S).Perm (List.range' 0 (Flock.TemplateQuery.population rs)) :=
    strataOf_perm _ _ (partition_evaluate_tiles d query rs he)
  obtain ⟨n, named, derived, hn, hnamed, hd, hDc⟩ := slotDesign_ok rule d query table D hD
  have hnU : n = (popUnits S).length := by
    rw [units_eq d query rs he] at hn
    rw [hperm.length_eq, List.length_range']
    exact (Except.ok.inj hn).symm
  have hns : pop.toList.map (·.size) = S.toList.map fun p => rsSize p.2 := (laysOut_units _ _ hlays).2
  have hsum : strataSize S = n := by rw [hnU, length_popUnits]; rfl
  have hpos : ∀ m ∈ S.toList.map (fun p => rsSize p.2), 1 ≤ m := by
    intro m hm
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hm
    exact strataOf_pos _ (partition_covered d query rs he) p hp
  suffices h : ∀ first : OneStage, first.WellFormed (S.toList.map fun p => rsSize p.2) →
      (⟨[first, .all]⟩ : Design).WellFormed (slotPartition { classes := #[pop, inputClass d] }) by
    rcases hDc with ⟨-, rfl⟩ | ⟨hne, parsed, hp, rfl⟩
    · exact h .all trivial
    · exact h _ (first_wf rule d query table n named derived parsed S (strata_eq d query rs he) hne hnamed hd hp
        hsum hpos)
  intro first hf
  refine ⟨rfl, fun p hp => ?_⟩
  simp only [slotPartition, List.map_cons, List.map_nil, List.zip_cons_cons, List.zip_nil_left, List.mem_cons,
    List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · simpa [strataSizes, Function.comp_def, hns] using hf
  · trivial

end Properties.SampledProofs.SecurityProofs
