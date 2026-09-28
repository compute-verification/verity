import Proofs.Flock.Soundness.Discharge.Hidden.Flat
import Proofs.Flock.Soundness.Discharge.Layout.OutShape

set_option Elab.async false

/-!
# Where the circuit computes each output word

`HmOut.delta` copies an output row's message bits from `HmOut.outCol c g o`, the first column of output word `o` at
template instance `g`. In an untyped circuit whose unit returns the outputs, that is the unit copy that returns `o`, at
the column of its returned-output port: `outCol` reads which copy and which port from `ret`, the last write of each
output word over the copies' leaf maps. That write is the only one, because `HmRow.check` throws on an output word some
copy returns twice:

* `check_returned`: `HmRow.check`'s `seen` loop, so the copies' leaf maps, in order, are distinct (`returned`);
* `ret_leaf`: `ret` at copy `u'`'s output `j` is `(u', j)`;
* `outCol_leaf`: `HmOut.outCol` there is the unit slot `g · upv + u'`, column `16 j` of the returned-output group;
* `outPortOf_spec`: output word `o` is word `o - outAt jo` of output port `jo`;
* `checkRows_words`: each output row's words hold its port's (`HmOut.checkRows`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 FlockSoundness.Discharge.Layout

/-! ## Loop invariants -/

/-- A loop in `Except` that finished keeps an invariant of the elements it walked. -/
theorem forIn_list_inv {α σ ε : Type} (I : List α → σ → Prop) {f : α → σ → Except ε (ForInStep σ)}
    (hf : ∀ pre a s t, I pre s → f a s = .ok t → ∃ s', t = .yield s' ∧ I (pre ++ [a]) s') :
    ∀ (l pre : List α) (s v : σ), I pre s → forIn l s f = .ok v → I (pre ++ l) v
  | [], pre, s, v, hs, h => by
    rw [List.forIn_nil] at h
    cases h
    simpa using hs
  | a :: l, pre, s, v, hs, h => by
    rw [List.forIn_cons] at h
    obtain ⟨t, ht, h⟩ := except_bind_ok h
    obtain ⟨s', rfl, hs'⟩ := hf pre a s t hs ht
    simpa using forIn_list_inv I hf l (pre ++ [a]) s' v hs' h

theorem forIn_range_inv {σ ε : Type} (I : List ℕ → σ → Prop) {f : ℕ → σ → Except ε (ForInStep σ)} {n : ℕ}
    (hf : ∀ pre a s t, I pre s → f a s = .ok t → ∃ s', t = .yield s' ∧ I (pre ++ [a]) s') {s v : σ}
    (hs : I [] s) (h : forIn [:n] s f = .ok v) : I (List.range n) v := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have := forIn_list_inv I hf _ [] s v hs h
  simp only [Std.Legacy.Range.size, List.nil_append, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at this
  rwa [List.range_eq_range']

/-! ## `HmRow.check`'s `seen` loop -/

/-- The output words copies `us` return, in order. -/
def returned (c : Circuit) (us : List ℕ) : List ℕ := us.flatMap fun u => (c.leavesOut.getD u #[]).toList

theorem bool_get (s : Array Bool) (o : ℕ) : s[o]! = (s[o]?).getD false := by
  rw [Hidden.getElem!_eq_getD', Array.getD_eq_getD_getElem?]
  rfl

/-- What `seen` holds after the output words `L`: exactly them, each once. -/
def SeenInv (c : Circuit) (L : List ℕ) (s : Array Bool) : Prop :=
  s.size = c.outWords ∧ (∀ o, s[o]! = true → o ∈ L) ∧ L.Nodup ∧ ∀ o ∈ L, s[o]! = true

theorem seenInv_step {c : Circuit} {L : List ℕ} {s : Array Bool} (hs : SeenInv c L s) {o : ℕ}
    (hc : ¬ (decide (o ≥ c.outWords) || s[o]!) = true) : SeenInv c (L ++ [o]) (s.set! o true) := by
  obtain ⟨hsz, hin, hnd, hset⟩ := hs
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_le, Bool.not_eq_true] at hc
  obtain ⟨hlt, hf⟩ := hc
  have hos : o < s.size := by omega
  have hnot : o ∉ L := fun hm => by rw [hset o hm] at hf; cases hf
  refine ⟨by rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds, hsz], fun o' ho' => ?_,
    List.nodup_append.2 ⟨hnd, List.nodup_singleton o, fun a ha b hb => ?_⟩, fun o' ho' => ?_⟩
  · rw [bool_get, Array.set!_eq_setIfInBounds] at ho'
    by_cases he : o = o'
    · subst he; exact List.mem_append_right _ (List.mem_singleton_self _)
    · rw [Array.getElem?_setIfInBounds_ne he, ← bool_get] at ho'
      exact List.mem_append_left _ (hin o' ho')
  · rw [List.mem_singleton] at hb
    subst hb
    exact fun he => hnot (he ▸ ha)
  · rw [bool_get, Array.set!_eq_setIfInBounds]
    by_cases he : o = o'
    · subst he; rw [Array.getElem?_setIfInBounds_self_of_lt hos]; rfl
    · rw [Array.getElem?_setIfInBounds_ne he, ← bool_get]
      rcases List.mem_append.1 ho' with hm | hm
      · exact hset o' hm
      · exact absurd (List.mem_singleton.1 hm).symm he

/-- **`HmRow.check`'s `seen` loop**: in an untyped circuit whose unit returns the outputs, the copies' leaf maps, in
order, return each output word at most once. -/
theorem check_returned {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet = c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) : (returned c (List.range c.upv)).Nodup := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := except_bind_ok h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨-, h⟩ := ite_throw_cond h
  replace h := ite_join h fun ha => (ite_throw_cond ha).2
  obtain ⟨u1, -, h⟩ := except_bind_ok h
  obtain ⟨u2, -, h⟩ := except_bind_ok h
  obtain ⟨wired, -, h⟩ := except_bind_ok h
  obtain ⟨wired2, -, h⟩ := except_bind_ok h
  obtain ⟨u3, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_cond h
  obtain ⟨seen, hseen, -⟩ := except_bind_ok h
  refine (forIn_range_inv (fun pre s => SeenInv c (returned c pre) s) (fun pre a s t hs hst => ?_) ?_ hseen).2.2.1
  · obtain ⟨-, hst⟩ := ite_throw_cond hst
    obtain ⟨s', hin, hst⟩ := except_bind_ok hst
    cases hst
    refine ⟨_, rfl, ?_⟩
    have e0 : (default : Array ℕ) = #[] := rfl
    rw [Hidden.getElem!_eq_getD', e0, ← Array.forIn_toList] at hin
    have := forIn_list_inv (fun pre' s => SeenInv c (returned c pre ++ pre') s) (fun pre' o s t hs hst => by
      obtain ⟨hc, hst⟩ := ite_throw_cond hst
      cases hst
      refine ⟨_, rfl, ?_⟩
      rw [← List.append_assoc]
      exact seenInv_step hs hc) _ [] s s' (by simpa using hs) hin
    simpa [returned, List.flatMap_append] using this
  · refine ⟨Array.size_replicate, fun o h1 => ?_, List.nodup_nil, fun o h => by cases h⟩
    rw [bool_get, Array.getElem?_replicate, ho, bne_self_eq_false] at h1
    split at h1 <;> cases h1

/-! ## `ret` -/

/-- `HmOut.outCol`'s step of copy `u`'s leaf map: output word `o` is returned by `u`'s output `q`. -/
def retStep (u : ℕ) : Array (ℕ × ℕ) × ℕ → ℕ → Array (ℕ × ℕ) × ℕ := fun (r, q) o => (r.set! o (u, q), q + 1)

theorem retStep_size (u : ℕ) : ∀ (L : List ℕ) (r : Array (ℕ × ℕ)) (q : ℕ), (L.foldl (retStep u) (r, q)).1.size = r.size
  | [], _, _ => rfl
  | o :: L, r, q => by
    rw [List.foldl_cons]
    exact (retStep_size u L _ _).trans (by simp)

theorem retStep_not_mem (u : ℕ) {x : ℕ} : ∀ (L : List ℕ) (r : Array (ℕ × ℕ)) (q : ℕ), x ∉ L →
    (L.foldl (retStep u) (r, q)).1[x]? = r[x]?
  | [], _, _, _ => rfl
  | o :: L, r, q, hx => by
    rw [List.foldl_cons]
    rw [List.mem_cons, not_or] at hx
    exact (retStep_not_mem u L _ _ hx.2).trans (by simp [Array.getElem?_setIfInBounds_ne (Ne.symm hx.1)])

theorem retStep_mem (u : ℕ) : ∀ (L : List ℕ) (r : Array (ℕ × ℕ)) (q : ℕ), L.Nodup → ∀ i (hi : i < L.length),
    L[i] < r.size → (L.foldl (retStep u) (r, q)).1[L[i]]? = some (u, q + i)
  | [], _, _, _, _, hi, _ => absurd hi (by simp)
  | o :: L, r, q, hnd, i, hi, hlt => by
    rw [List.foldl_cons]
    rw [List.nodup_cons] at hnd
    rcases i with _ | i
    · simp only [List.getElem_cons_zero] at hlt ⊢
      rw [retStep_not_mem u L _ _ hnd.1]
      simp [retStep, hlt]
    · simp only [List.getElem_cons_succ] at hlt ⊢
      have hi' : i < L.length := by simpa using hi
      show (L.foldl (retStep u) (r.set! o (u, q), q + 1)).1[L[i]'hi']? = some (u, q + (i + 1))
      rw [retStep_mem u L _ _ hnd.2 i hi' (by simpa using hlt)]
      simp only [Option.some.injEq, Prod.mk.injEq, true_and]
      omega

/-- `HmOut.outCol`'s `ret`: per output word, the copy that returns it and the copy's output, its last write over the
copies `us`' leaf maps. -/
def retOf (c : Circuit) (us : List ℕ) (r : Array (ℕ × ℕ)) : Array (ℕ × ℕ) :=
  us.foldl (fun r u => ((c.leavesOut.getD u #[]).foldl (retStep u) (r, 0)).1) r

theorem retOf_size (c : Circuit) : ∀ (us : List ℕ) (r : Array (ℕ × ℕ)), (retOf c us r).size = r.size
  | [], _ => rfl
  | u :: us, r => by
    unfold retOf
    rw [List.foldl_cons]
    exact (retOf_size c us _).trans (by rw [← Array.foldl_toList, retStep_size])

theorem retOf_not_mem (c : Circuit) {x : ℕ} : ∀ (us : List ℕ) (r : Array (ℕ × ℕ)), x ∉ returned c us →
    (retOf c us r)[x]? = r[x]?
  | [], _, _ => rfl
  | u :: us, r, hx => by
    unfold retOf
    rw [List.foldl_cons]
    simp only [returned, List.flatMap_cons, List.mem_append, not_or] at hx
    exact (retOf_not_mem c us _ hx.2).trans (by rw [← Array.foldl_toList, retStep_not_mem u _ _ _ hx.1])

theorem retOf_mem (c : Circuit) : ∀ (us : List ℕ) (r : Array (ℕ × ℕ)), (returned c us).Nodup → ∀ u' ∈ us,
    ∀ j (hj : j < (c.leavesOut.getD u' #[]).size), (c.leavesOut.getD u' #[])[j] < r.size →
      (retOf c us r)[(c.leavesOut.getD u' #[])[j]]? = some (u', j)
  | [], _, _, _, hu, _, _, _ => absurd hu (by simp)
  | u :: us, r, hnd, u', hu, j, hj, hlt => by
    have e : retOf c (u :: us) r = retOf c us ((c.leavesOut.getD u #[]).foldl (retStep u) (r, 0)).1 := rfl
    rw [e]
    simp only [returned, List.flatMap_cons] at hnd
    obtain ⟨hnd1, hnd2, hdis⟩ := List.nodup_append.1 hnd
    by_cases he : u' = u
    · subst he
      have hm : (c.leavesOut.getD u' #[])[j] ∈ (c.leavesOut.getD u' #[]).toList := Array.getElem_mem_toList hj
      rw [retOf_not_mem c us _ fun hx => hdis _ hm _ hx rfl, ← Array.foldl_toList]
      have := retStep_mem u' (c.leavesOut.getD u' #[]).toList r 0 hnd1 j (by simpa using hj) (by simpa using hlt)
      simpa using this
    · exact retOf_mem c us _ hnd2 u' ((List.mem_cons.1 hu).resolve_left he) j hj
        (by rw [← Array.foldl_toList, retStep_size]; exact hlt)

/-- **`ret` at a copy's returned output**: copy `u'`'s output `j` is returned by `u'`'s output `j`, when the copies'
leaf maps return each output word at most once (`check_returned`), each below `outWords`. -/
theorem ret_leaf {c : Circuit} (hnd : (returned c (List.range c.upv)).Nodup)
    (hlt : ∀ u < c.upv, ∀ o ∈ (c.leavesOut.getD u #[]).toList, o < c.outWords) {u' j : ℕ} (hu' : u' < c.upv)
    (hj : j < (c.leavesOut.getD u' #[]).size) :
    (retOf c (List.range c.upv) (Array.replicate c.outWords (0, 0))).getD (c.leavesOut.getD u' #[])[j] (0, 0) =
      (u', j) := by
  rw [Array.getD_eq_getD_getElem?, retOf_mem c _ _ hnd u' (List.mem_range.2 hu') j hj
    (by rw [Array.size_replicate]; exact hlt u' hu' _ (Array.getElem_mem_toList hj))]
  rfl

/-! ## `HmOut.outCol` at a returned output -/

theorem portCols_16 : ∀ (L : List ℕ) (a : ℕ), (∀ w ∈ L, w = 16) →
    ((forIn L (#[], a) (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList) =
      (List.range L.length).map fun j => (a + 16 * j, 16)
  | [], _, _ => rfl
  | w :: L, a, h => by
    rw [List.forIn_cons]
    show ((forIn L (#[].push (a, w), a + w) _ : Id _).1.toList) = _
    rw [inner_append L (#[].push (a, w)), portCols_16 L (a + w) fun w hw => h w (List.mem_cons_of_mem _ hw),
      h w List.mem_cons_self, List.length_cons, List.range_succ_eq_map]
    simp only [List.map_cons, List.map_map]
    simp only [Array.toList_push, List.nil_append, Nat.mul_zero, Nat.add_zero, List.cons_append,
      List.cons.injEq, true_and]
    refine List.map_congr_left fun j _ => ?_
    simp only [Function.comp, Prod.mk.injEq, and_true, Nat.succ_eq_add_one]
    omega

/-- A group of 16-bit ports: port `q`'s first column is `col · WORD + 16 q`. -/
theorem portCols_16_get {g : Group} (h16 : ∀ b ∈ g.bits.toList, b = 16) {q : ℕ} (hq : q < g.bits.size) :
    ((portCols #[g]).getD q default).1 = g.col * WORD + 16 * q := by
  have e := (Aliased.portCols_one g).trans (portCols_16 g.bits.toList _ h16)
  have hq' : q < (portCols #[g]).toList.length := by rw [e]; simpa using hq
  rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, List.getElem?_eq_getElem hq', Option.getD_some]
  simp [e]

/-- `HmOut.outCol` in an untyped circuit whose unit returns the outputs: the copy `ret` names, at the column of its
returned-output port. -/
theorem outCol_eq {c : Circuit} (ht : c.typed = none) (ho : c.outNet = c.unitNet) (g o : ℕ) :
    HmOut.outCol c g o =
      c.slotCol c.unitNet g ((retOf c (List.range c.upv) (Array.replicate c.outWords (0, 0))).getD o (0, 0)).1 +
        ((portCols #[c.unit.outGroups.getD 0 default]).getD
          ((retOf c (List.range c.upv) (Array.replicate c.outWords (0, 0))).getD o (0, 0)).2 default).1 := by
  unfold HmOut.outCol
  simp only [ht, ho, Option.isSome_none, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  rfl

/-- **`HmOut.outCol` at copy `u'`'s output `j`**: in an untyped circuit whose unit returns the outputs, its copies'
leaf maps returning each output word at most once and below `outWords`, its returned-output group's ports 16-bit and
one per output of each copy, and `upv` slots of the unit per template instance, it is the unit slot `g · upv + u'` at
column `16 j` of the returned-output group. -/
theorem outCol_leaf {c : Circuit} (ht : c.typed = none) (ho : c.outNet = c.unitNet)
    (hnd : (returned c (List.range c.upv)).Nodup)
    (hlt : ∀ u < c.upv, ∀ o ∈ (c.leavesOut.getD u #[]).toList, o < c.outWords)
    (h16 : ∀ b ∈ (c.unit.outGroups.getD 0 default).bits.toList, b = 16)
    (hsz : ∀ u < c.upv, (c.leavesOut.getD u #[]).size = (c.unit.outGroups.getD 0 default).bits.size)
    (hpv : c.perVu.getD c.unitNet 0 = c.upv) (g : ℕ) {u' j : ℕ} (hu' : u' < c.upv)
    (hj : j < (c.leavesOut.getD u' #[]).size) :
    HmOut.outCol c g (c.leavesOut.getD u' #[])[j] =
      c.unitRange.slot (g * c.upv + u') + ((c.unit.outGroups.getD 0 default).col * WORD + 16 * j) := by
  rw [outCol_eq ht ho, ret_leaf hnd hlt hu' hj, portCols_16_get h16 (by rw [← hsz u' hu']; exact hj)]
  simp only [Circuit.slotCol, Circuit.unitRange, hpv]

/-! ## The output port of an output word -/

/-- The output port output word `o` is in: the last whose first word is at most `o`. -/
def outPortOf (c : Circuit) (o : ℕ) : ℕ := Nat.findGreatest (fun j => Hidden.outAt c j ≤ o) c.outPorts.size

theorem foldl_snd_sum : ∀ (l : List (String × ℕ)) (a : ℕ), l.foldl (· + ·.2) a = a + (l.map (·.2)).sum
  | [], a => by simp
  | p :: l, a => by
    rw [List.foldl_cons, foldl_snd_sum l, List.map_cons, List.sum_cons]
    ring

theorem outAt_size (c : Circuit) : Hidden.outAt c c.outPorts.size = c.outWords := by
  unfold Hidden.outAt Circuit.outWords
  rw [← Array.foldl_toList, foldl_snd_sum, Nat.zero_add, List.take_of_length_le (by simp)]

/-- **Output word `o` is word `o - outAt jo` of output port `jo = outPortOf c o`**, for `o < outWords`. -/
theorem outPortOf_spec (c : Circuit) {o : ℕ} (ho : o < c.outWords) :
    outPortOf c o < c.outPorts.size ∧ Hidden.outAt c (outPortOf c o) ≤ o ∧
      o - Hidden.outAt c (outPortOf c o) < (c.outPorts.getD (outPortOf c o) default).2 := by
  have hP : Hidden.outAt c (outPortOf c o) ≤ o :=
    Nat.findGreatest_spec (P := fun j => Hidden.outAt c j ≤ o) (Nat.zero_le _) (by simp [Hidden.outAt])
  have hle : outPortOf c o ≤ c.outPorts.size := Nat.findGreatest_le _
  have hlt : outPortOf c o < c.outPorts.size := by
    refine lt_of_le_of_ne hle fun he => ?_
    rw [he, outAt_size] at hP
    omega
  refine ⟨hlt, hP, ?_⟩
  have hng := Nat.findGreatest_is_greatest (P := fun j => Hidden.outAt c j ≤ o) (k := outPortOf c o + 1)
    (Nat.lt_succ_self _) hlt
  simp only [not_le] at hng
  rw [Hidden.outAt_succ c hlt, Hidden.getElem!_eq_getD'] at hng
  omega

/-! ## An output row's words -/

/-- **Each output row's words hold its port's** (`HmOut.checkRows`: a padded lone port's are the port's, zero-padded to
whole SHA-512 blocks; a bit row of `16 · words` bits holds exactly its port's). -/
theorem checkRows_words {c : Circuit} (h : HmOut.checkRows c = .ok ())
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (hb : ∀ p ∈ c.ports.toList, p.bits = 16 * p.words) {j : ℕ} (hj : j < c.outPorts.size) :
    (c.outPorts.getD j default).2 ≤ (c.ports.getD (HmOut.nIn c + j) default).words := by
  have hj' : j < c.outRowPorts.size := by rw [hr, Array.size_replicate]; exact hj
  obtain ⟨hlt, hrow⟩ := Hidden.checkRows_out h hj'
  have hmem : c.ports.getD (HmOut.nIn c + j) default ∈ c.ports.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    exact Array.getElem_mem_toList hlt
  rw [Hidden.outGroup_single hr hj] at hrow
  dsimp only at hrow
  have := hb _ hmem
  rcases hrow with ⟨-, hw, -⟩ | ⟨-, hb'⟩
  · rw [hw]; omega
  · omega

end FlockSoundness.Discharge.HiddenExec
