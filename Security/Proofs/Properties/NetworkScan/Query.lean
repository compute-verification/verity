import Properties.NetworkScan.Flattens
import Proofs.Flock.Claims.Sampling.Select.Layout
import Mathlib.Data.List.Nodup

/-!
# The template query's ranges, and where an instance sits among the audit's members

`Flock.TemplateQuery.evaluate` lays one range a root node that doesn't call an input, in node order, each with its
node's template and units (a batch's members, else one), the ranges end to end from `0`
(`Properties.SampledProofs.SecurityProofs.evaluate_tiles`). So unit `w` is located at node `k`'s member `t`
(`Flock.TemplateQuery.locate`) exactly when `w` is `t` past the first unit of `k`'s range, `t` below its count. Over
members holding each of the population's units once (`MembersOf`'s first half), the claims position of template `i`'s
instance `t` (`instPos`) is the one its unit's member gives it, from that member's `base`.

* `evaluate_inv`, `evaluate_range`: an accepted query's ranges, one a root node that doesn't call an input, each with
  its node's template and count, and where `locate` places a unit.
* `locate_iff`, `locate_any_iff`: what locating a unit at a range's node and member says of the unit.
* `unit_exists`, `unit_unique`: a unit of the population is one member's unit, once.
* `instPos_eq`, `instPos_at`: an instance's claims position, at the member and unit holding it.
-/

namespace Properties.NetworkScan.Query

open Flock.TemplateQuery (TRange)
open SampledProofs.Bits (Members base)
open NetworkCheck.Period (templateNodes)
open Properties.SampledProofs.SecurityProofs (Tiles evaluate_tiles bind_ok bind_error)

/-! ## The query's ranges -/

/-- A `forIn` over `[s, s + n)` in `Except` keeps an invariant of the index that each step keeps, yielding. -/
theorem forIn_range'_inv {β ε : Type} (I : ℕ → β → Prop) (f : ℕ → β → Except ε (ForInStep β))
    (hf : ∀ k b, I k b → ∀ st, f k b = .ok st → ∃ c, st = .yield c ∧ I (k + 1) c) :
    ∀ (n s : ℕ) (b b' : β), I s b → forIn (List.range' s n) b f = .ok b' → I (s + n) b'
  | 0, s, b, b', hb, h => by
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain rfl := Except.ok.inj h
    simpa using hb
  | n + 1, s, b, b', hb, h => by
    rw [List.range'_succ, List.forIn_cons] at h
    cases hs : f s b with
    | error e => rw [hs] at h; exact absurd h (by rw [bind_error]; simp)
    | ok st =>
      obtain ⟨c, rfl, hc⟩ := hf s b hb _ hs
      rw [hs, bind_ok] at h
      have := forIn_range'_inv I f hf n (s + 1) c b' hc h
      rwa [show s + 1 + n = s + (n + 1) by omega] at this

/-- The template root node `k` calls, as the query reads it. -/
def tidAt (d : Flock.TemplateQuery.Desc) (root : Flock.TemplateQuery.DDef) (k : ℕ) : String :=
  d.callees.getD (root.nodes[k]!).fn ""

/-- Root node `k`'s units, as the query counts them: a batch's members, else one. -/
def countAt (root : Flock.TemplateQuery.DDef) (k : ℕ) : ℕ :=
  if (root.nodes[k]!).form == "batch" then (root.nodes[k]!).n else 1

/-- The query before root node `k`: its ranges' nodes ascend, each range is a node below `k` with that node's template
and count, and each node below `k` that doesn't call an input has a range. -/
def Inv (d : Flock.TemplateQuery.Desc) (root : Flock.TemplateQuery.DDef) (k : ℕ) (rs : Array TRange) : Prop :=
  (rs.toList.map (·.node)).Pairwise (· < ·) ∧
    (∀ r ∈ rs.toList, r.node < k ∧ r.template = tidAt d root r.node ∧ r.count = countAt root r.node) ∧
    ∀ k' < k, Flock.TemplateQuery.isInput (tidAt d root k') = false → ∃ r ∈ rs.toList, r.node = k'

theorem inv_skip {d : Flock.TemplateQuery.Desc} {root : Flock.TemplateQuery.DDef} {k : ℕ} {rs : Array TRange}
    (h : Inv d root k rs) (hin : Flock.TemplateQuery.isInput (tidAt d root k) = true) : Inv d root (k + 1) rs := by
  obtain ⟨hP, hR, hC⟩ := h
  refine ⟨hP, fun r hr => ?_, fun k' hk' hk'in => ?_⟩
  · obtain ⟨h1, h2, h3⟩ := hR r hr
    exact ⟨by omega, h2, h3⟩
  · by_cases hlt : k' < k
    · exact hC k' hlt hk'in
    · obtain rfl : k' = k := by omega
      simp [hin] at hk'in

theorem inv_push {d : Flock.TemplateQuery.Desc} {root : Flock.TemplateQuery.DDef} {k : ℕ} {rs : Array TRange}
    (h : Inv d root k rs) (first : ℕ) :
    Inv d root (k + 1) (rs.push ⟨k, tidAt d root k, first, countAt root k⟩) := by
  obtain ⟨hP, hR, hC⟩ := h
  refine ⟨?_, fun r hr => ?_, fun k' hk' hk'in => ?_⟩
  · rw [Array.toList_push, List.map_append, List.map_cons, List.map_nil, List.pairwise_append]
    refine ⟨hP, List.pairwise_singleton _ _, fun a ha b hb => ?_⟩
    obtain ⟨r, hr, rfl⟩ := List.mem_map.1 ha
    rw [List.mem_singleton] at hb
    subst hb
    exact (hR r hr).1
  · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hr
    rcases hr with hr | rfl
    · obtain ⟨h1, h2, h3⟩ := hR r hr
      exact ⟨by omega, h2, h3⟩
    · exact ⟨Nat.lt_succ_self k, rfl, rfl⟩
  · rw [Array.toList_push]
    by_cases hlt : k' < k
    · obtain ⟨r, hr, hrk⟩ := hC k' hlt hk'in
      exact ⟨r, List.mem_append_left _ hr, hrk⟩
    · obtain rfl : k' = k := by omega
      exact ⟨_, List.mem_append_right _ (List.mem_singleton_self _), rfl⟩

/-- **An accepted query's ranges**: the root's, their nodes ascending, each a root node with its template and count,
one each root node that doesn't call an input. -/
theorem evaluate_inv {d : Flock.TemplateQuery.Desc} {tids : Array String} {rs : Array TRange}
    (h : Flock.TemplateQuery.evaluate d tids = .ok rs) :
    ∃ root, d.def? d.root = some root ∧ Inv d root root.nodes.size rs := by
  unfold Flock.TemplateQuery.evaluate at h
  split at h
  · rename_i root hroot
    dsimp only at h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
    have hsize : ([:root.nodes.size] : Std.Legacy.Range).size = root.nodes.size := by
      show (root.nodes.size - 0 + 1 - 1) / 1 = _
      omega
    rw [hsize, show ([:root.nodes.size] : Std.Legacy.Range).start = 0 from rfl,
      show ([:root.nodes.size] : Std.Legacy.Range).step = 1 from rfl] at h
    cases hl : forIn (m := Except String) (List.range' 0 root.nodes.size 1) ((#[] : Array TRange), 0) _ with
    | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨ranges, at_⟩ := y
      rw [hl, bind_ok] at h
      dsimp only at h
      have hinv : Inv d root (0 + root.nodes.size) ranges :=
        forIn_range'_inv (fun k (b : Array TRange × ℕ) => Inv d root k b.1) _ (fun k b hb st hs => ?_)
          root.nodes.size 0 ((#[] : Array TRange), 0) (ranges, at_) (by simp [Inv]) hl
      · have hrs : rs = ranges := by
          split at h
          · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
          · exact (Except.ok.inj h).symm
        subst hrs
        rw [Nat.zero_add] at hinv
        exact ⟨root, hroot, hinv⟩
      · obtain ⟨ranges₀, at₀⟩ := b
        dsimp only at hs hb
        split at hs
        · rename_i hin
          obtain rfl := Except.ok.inj hs
          exact ⟨_, rfl, inv_skip hb hin⟩
        · split at hs
          · exact absurd hs (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
          · obtain rfl := Except.ok.inj hs
            exact ⟨_, rfl, inv_push hb at₀⟩
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])

theorem tiles_le : ∀ (l : List (String × ℕ × ℕ)) (a b : ℕ), Tiles l a b → a ≤ b
  | [], _, _, h => Nat.le_of_eq h
  | (_, _, n) :: l, a, _, ⟨_, h2⟩ => Nat.le_trans (Nat.le_add_right a n) (tiles_le l _ _ h2)

/-- Each of ranges laid end to end from `a` to `b` lies between them. -/
theorem tiles_mem : ∀ (l : List TRange) (a b : ℕ), Tiles (l.map fun r => (r.template, r.first, r.count)) a b →
    ∀ r ∈ l, a ≤ r.first ∧ r.first + r.count ≤ b
  | [], _, _, _, _, hr => by simp at hr
  | x :: l, a, b, h, r, hr => by
    simp only [List.map_cons, Tiles] at h
    obtain ⟨h1, h2⟩ := h
    have hle := tiles_le _ _ _ h2
    rcases List.mem_cons.1 hr with rfl | hr
    · omega
    · have := tiles_mem l _ _ h2 r hr
      omega

/-- Ranges laid end to end share no unit. -/
theorem tiles_unique : ∀ (l : List TRange) (a b : ℕ), Tiles (l.map fun r => (r.template, r.first, r.count)) a b →
    ∀ x ∈ l, ∀ y ∈ l, ∀ v, x.first ≤ v → v < x.first + x.count → y.first ≤ v → v < y.first + y.count → x = y
  | [], _, _, _, _, hx, _, _, _, _, _, _, _ => by simp at hx
  | z :: l, a, b, h, x, hx, y, hy, v, hx1, hx2, hy1, hy2 => by
    simp only [List.map_cons, Tiles] at h
    obtain ⟨h1, h2⟩ := h
    have hl := tiles_mem l _ _ h2
    rcases List.mem_cons.1 hx with rfl | hx <;> rcases List.mem_cons.1 hy with rfl | hy
    · rfl
    · exfalso; have := hl y hy; omega
    · exfalso; have := hl x hx; omega
    · exact tiles_unique l _ _ h2 x hx y hy v hx1 hx2 hy1 hy2

theorem locate_eq {rs : Array TRange} {r : TRange} {v : ℕ} (hr : r ∈ rs) (h1 : r.first ≤ v)
    (h2 : v < r.first + r.count) (hu : ∀ x ∈ rs, x.first ≤ v → v < x.first + x.count → x = r) :
    Flock.TemplateQuery.locate rs v = some (r.node, v - r.first, r.template) := by
  have hf : rs.find? (fun x => x.first ≤ v && v < x.first + x.count) = some r := by
    cases hf : rs.find? (fun x => x.first ≤ v && v < x.first + x.count) with
    | none =>
      have := Array.find?_eq_none.1 hf r hr
      simp [h1, h2] at this
    | some r' =>
      have hp := Array.find?_some hf
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
      rw [hu r' (Array.mem_of_find?_eq_some hf) hp.1 hp.2]
  unfold Flock.TemplateQuery.locate
  rw [hf, Option.map_some]

/-- **Where `locate` places a unit**: over ranges laid end to end with distinct nodes, unit `v` is at range `r`'s node,
member `t` and template `s` exactly when `v` is `t` past `r`'s first unit, `t` below its count, and `s` its template. -/
theorem locate_iff {rs : Array TRange} {a b : ℕ}
    (hT : Tiles (rs.toList.map fun r => (r.template, r.first, r.count)) a b)
    (hN : (rs.toList.map (·.node)).Nodup) {r : TRange} (hr : r ∈ rs) (v t : ℕ) (s : String) :
    Flock.TemplateQuery.locate rs v = some (r.node, t, s) ↔ v = r.first + t ∧ t < r.count ∧ s = r.template := by
  constructor
  · intro h
    unfold Flock.TemplateQuery.locate at h
    obtain ⟨r', hf, he⟩ := Option.map_eq_some_iff.1 h
    have hp := Array.find?_some hf
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
    obtain ⟨hp1, hp2⟩ := hp
    simp only [Prod.mk.injEq] at he
    obtain ⟨hn, ht, hs⟩ := he
    obtain rfl : r' = r := List.inj_on_of_nodup_map hN (Array.mem_def.1 (Array.mem_of_find?_eq_some hf))
      (Array.mem_def.1 hr) hn
    exact ⟨by omega, by omega, hs.symm⟩
  · rintro ⟨rfl, ht, rfl⟩
    rw [locate_eq (v := r.first + t) hr (Nat.le_add_right _ _) (by omega)
      (fun x hx h1 h2 => tiles_unique _ _ _ hT x (Array.mem_def.1 hx) r (Array.mem_def.1 hr) _ h1 h2
        (Nat.le_add_right _ _) (by omega)),
      Nat.add_sub_cancel_left]

/-- **Locating an instance**: unit `w` is located at range `r`'s node and member `t`, a member it has, exactly when
`w` is `t` past `r`'s first unit. -/
theorem locate_any_iff {rs : Array TRange} {a b : ℕ}
    (hT : Tiles (rs.toList.map fun r => (r.template, r.first, r.count)) a b)
    (hN : (rs.toList.map (·.node)).Nodup) {r : TRange} (hr : r ∈ rs) {t : ℕ} (ht : t < r.count) (w : ℕ) :
    ((Flock.TemplateQuery.locate rs w).any fun l => l.1 == r.node && l.2.1 == t) = true ↔ w = r.first + t := by
  rw [Option.any_eq_true]
  constructor
  · rintro ⟨⟨k, t', s⟩, hl, hp⟩
    simp only [Bool.and_eq_true, beq_iff_eq] at hp
    obtain ⟨rfl, rfl⟩ := hp
    exact ((locate_iff hT hN hr w _ _).1 hl).1
  · rintro rfl
    exact ⟨(r.node, t, r.template), (locate_iff hT hN hr _ t _).2 ⟨rfl, ht, rfl⟩, by simp⟩

/-- **A node's range**: when the query accepts, a root node `k` that doesn't call an input has a range, with `k`'s
template and count, inside the population, and unit `w` is located at `k`'s member `t`, one it has, exactly when `w`
is `t` past the range's first unit. -/
theorem evaluate_range {d : Flock.TemplateQuery.Desc} {tids : Array String} {rs : Array TRange}
    {root : Flock.TemplateQuery.DDef} (h : Flock.TemplateQuery.evaluate d tids = .ok rs)
    (hroot : d.def? d.root = some root) {k : ℕ} (hk : k < root.nodes.size)
    (hin : Flock.TemplateQuery.isInput (tidAt d root k) = false) :
    ∃ r ∈ rs, r.node = k ∧ r.template = tidAt d root k ∧ r.count = countAt root k ∧
      r.first + r.count ≤ Flock.TemplateQuery.population rs ∧
      ∀ w t, t < r.count →
        (((Flock.TemplateQuery.locate rs w).any fun l => l.1 == k && l.2.1 == t) = true ↔ w = r.first + t) := by
  obtain ⟨root', hroot', hP, hR, hC⟩ := evaluate_inv h
  rw [hroot] at hroot'
  obtain rfl := Option.some.inj hroot'
  obtain ⟨r, hr, rfl⟩ := hC k hk hin
  obtain ⟨-, htid, hcount⟩ := hR r hr
  have hT := evaluate_tiles d tids rs h
  rw [Array.toList_map] at hT
  have hN : (rs.toList.map (·.node)).Nodup := List.Pairwise.imp (fun h => Nat.ne_of_lt h) hP
  exact ⟨r, Array.mem_def.2 hr, rfl, htid, hcount, (tiles_mem _ _ _ hT r hr).2,
    fun w t ht => locate_any_iff hT hN (Array.mem_def.2 hr) ht w⟩

/-! ## The members' units -/

theorem list_getD_of_lt {α : Type} {l : List α} {i : ℕ} (d : α) (h : i < l.length) : l.getD i d = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]

theorem array_getD_of_lt {α : Type} {xs : Array α} {i : ℕ} (d : α) (h : i < xs.size) : xs.getD i d = xs[i] := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h, Option.getD_some]

theorem findSome?_eq_of_unique {α β : Type} {f : α → Option β} {a : α} :
    ∀ (l : List α), a ∈ l → (∀ x ∈ l, x ≠ a → f x = none) → l.findSome? f = f a
  | [], ha, _ => absurd ha (by simp)
  | x :: l, ha, hf => by
    rw [List.findSome?_cons]
    by_cases hx : x = a
    · rw [← hx]
      cases h : f x with
      | none =>
        exact List.findSome?_eq_none_iff.2 fun y hy => by
          by_cases hyx : y = x
          · rw [hyx]; exact h
          · exact hf y (List.mem_cons_of_mem _ hy) fun hya => hyx (hya.trans hx.symm)
      | some b => rfl
    · rw [hf x List.mem_cons_self hx]
      exact findSome?_eq_of_unique l ((List.mem_cons.1 ha).resolve_left (Ne.symm hx))
        fun y hy => hf y (List.mem_cons_of_mem _ hy)

theorem find?_range_eq {p : ℕ → Bool} {n u : ℕ} (hu : u < n) (hp : ∀ u' < n, p u' = true ↔ u' = u) :
    (List.range n).find? p = some u := by
  rw [List.find?_range_eq_some]
  refine ⟨(hp u hu).2 rfl, List.mem_range.2 hu, fun j hj => ?_⟩
  cases h : p j
  · rfl
  · exact absurd ((hp j (by omega)).1 h) (by omega)

/-- The first `m` whose `find?` finds something, and what it finds, when exactly one `(m, u)` satisfies `p`. -/
theorem findSome_bind_eq {β : Type} {n : ℕ} {sz : ℕ → ℕ} {p : ℕ → ℕ → Bool} {g : ℕ → ℕ → Option β}
    {m u : ℕ} (hm : m < n) (hu : u < sz m) (hp : ∀ m' < n, ∀ u' < sz m', p m' u' = true ↔ m' = m ∧ u' = u) :
    (List.range n).findSome? (fun m' => ((List.range (sz m')).find? (p m')).bind (g m')) = g m u := by
  have hfm : (List.range (sz m)).find? (p m) = some u :=
    find?_range_eq hu fun u' hu' => (hp m hm u' hu').trans ⟨fun h => h.2, fun h => ⟨rfl, h⟩⟩
  have hnone : ∀ m' ∈ List.range n, m' ≠ m → ((List.range (sz m')).find? (p m')).bind (g m') = none := by
    intro m' hm' hne
    have : (List.range (sz m')).find? (p m') = none := by
      rw [List.find?_range_eq_none]
      intro u' hu'
      cases h : p m' u'
      · rfl
      · exact absurd ((hp m' (List.mem_range.1 hm') u' hu').1 h).1 hne
    rw [this, Option.bind_none]
  rw [findSome?_eq_of_unique _ (List.mem_range.2 hm) hnone, hfm, Option.bind_some]

/-- A unit the members hold is unit `u` of some member `m`. -/
theorem unit_exists {ms : Members} {v : ℕ} (h : v ∈ ms.flatMap (·.2.toList)) :
    ∃ m < ms.length, ∃ u < (ms.getD m default).2.size, (ms.getD m default).2.getD u 0 = v := by
  obtain ⟨ci, hci, hv⟩ := List.mem_flatMap.1 h
  obtain ⟨m, hm, rfl⟩ := List.mem_iff_getElem.1 hci
  obtain ⟨u, hu, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_def.2 hv)
  refine ⟨m, hm, u, ?_, ?_⟩
  · rw [list_getD_of_lt _ hm]; exact hu
  · rw [list_getD_of_lt _ hm, array_getD_of_lt _ hu]

/-- Members holding no unit twice hold each at one member and unit. -/
theorem unit_unique {ms : Members} (hnd : (ms.flatMap (·.2.toList)).Nodup) {m m' u u' : ℕ}
    (hm : m < ms.length) (hm' : m' < ms.length) (hu : u < (ms.getD m default).2.size)
    (hu' : u' < (ms.getD m' default).2.size)
    (he : (ms.getD m default).2.getD u 0 = (ms.getD m' default).2.getD u' 0) : m = m' ∧ u = u' := by
  rw [list_getD_of_lt _ hm] at hu he
  rw [list_getD_of_lt _ hm'] at hu' he
  rw [array_getD_of_lt _ hu, array_getD_of_lt _ hu'] at he
  obtain ⟨hnd1, hnd2⟩ := List.nodup_flatMap.1 hnd
  have hdisj : ∀ (i j x : ℕ) (hi : i < ms.length) (hj : j < ms.length), i < j →
      x ∈ ms[i].2.toList → x ∈ ms[j].2.toList → False :=
    fun _ _ _ hi hj hij hx hy => List.pairwise_iff_getElem.1 hnd2 _ _ hi hj hij hx hy
  rcases Nat.lt_trichotomy m m' with h | rfl | h
  · exact (hdisj m m' _ hm hm' h (Array.getElem_mem_toList hu)
      (by rw [he]; exact Array.getElem_mem_toList hu')).elim
  · exact ⟨rfl, (List.Nodup.getElem_inj_iff (hnd1 _ (List.getElem_mem hm)) (hi := hu) (hj := hu')).1 he⟩
  · exact (hdisj m' m _ hm' hm h (Array.getElem_mem_toList hu')
      (by rw [← he]; exact Array.getElem_mem_toList hu)).elim

/-! ## An instance's claims position -/

/-- **An instance's claims position, at the member holding it**: over members holding each of the population's units
once, when the units located at template `i`'s node and member `t` are exactly `v`, the claims position of template
`i`'s instance `t` is the one `pos` gives it at the member `m` and unit `u` holding `v`, from `base ms m`. -/
theorem instPos_eq {P : Flock.Program.Prog} {rs : Array TRange} {ms : Members} {i t v : ℕ}
    {pos : Flock.Circuit → ℕ → ℕ → Option ℕ}
    (hperm : (ms.flatMap (·.2.toList)).Perm (List.range (Flock.TemplateQuery.population rs)))
    (hv : v < Flock.TemplateQuery.population rs)
    (hq : ∀ w, ((Flock.TemplateQuery.locate rs w).any fun l => l.1 == (templateNodes P).getD i 0 && l.2.1 == t) =
      true ↔ w = v) :
    ∃ m < ms.length, ∃ u < (ms.getD m default).2.size, (ms.getD m default).2.getD u 0 = v ∧
      instPos P rs ms i t pos =
        ((pos (ms.getD m default).1 (ms.getD m default).2.size u).map (base ms m + ·)).getD
          (jointOf ms).gates.length := by
  have hnd : (ms.flatMap (·.2.toList)).Nodup := hperm.nodup_iff.2 List.nodup_range
  obtain ⟨m, hm, u, hu, huv⟩ := unit_exists (hperm.mem_iff.2 (List.mem_range.2 hv))
  refine ⟨m, hm, u, hu, huv, ?_⟩
  have h := findSome_bind_eq (n := ms.length) (sz := fun m => (ms.getD m default).2.size)
    (p := fun m u => (Flock.TemplateQuery.locate rs ((ms.getD m default).2.getD u 0)).any
      fun l => l.1 == (templateNodes P).getD i 0 && l.2.1 == t)
    (g := fun m u => (pos (ms.getD m default).1 (ms.getD m default).2.size u).map (base ms m + ·)) hm hu
    fun m' hm' u' hu' => (hq ((ms.getD m' default).2.getD u' 0)).trans
      ⟨fun h' => unit_unique hnd hm' hm hu' hu (h'.trans huv.symm), fun h' => by rw [h'.1, h'.2]; exact huv⟩
  exact congrArg (fun o => o.getD (jointOf ms).gates.length) h

/-- **Template `i`'s instance `t`'s claims position**: when the query accepts with the ranges `rs`, template `i`'s node
`k` is a root node that doesn't call an input and `t` is below its count, the members holding each of the population's
units once hold the unit `t` past the first of `k`'s range at some member `m` and unit `u`, which give the instance's
claims position, from `base ms m`. -/
theorem instPos_at {P : Flock.Program.Prog} {d : Flock.TemplateQuery.Desc} {tids : Array String}
    {rs : Array TRange} {ms : Members} {root : Flock.TemplateQuery.DDef} {i t : ℕ}
    {pos : Flock.Circuit → ℕ → ℕ → Option ℕ} (h : Flock.TemplateQuery.evaluate d tids = .ok rs)
    (hroot : d.def? d.root = some root) (hk : (templateNodes P).getD i 0 < root.nodes.size)
    (hin : Flock.TemplateQuery.isInput (tidAt d root ((templateNodes P).getD i 0)) = false)
    (ht : t < countAt root ((templateNodes P).getD i 0))
    (hperm : (ms.flatMap (·.2.toList)).Perm (List.range (Flock.TemplateQuery.population rs))) :
    ∃ r ∈ rs, r.node = (templateNodes P).getD i 0 ∧ r.template = tidAt d root ((templateNodes P).getD i 0) ∧
      ∃ m < ms.length, ∃ u < (ms.getD m default).2.size, (ms.getD m default).2.getD u 0 = r.first + t ∧
        instPos P rs ms i t pos =
          ((pos (ms.getD m default).1 (ms.getD m default).2.size u).map (base ms m + ·)).getD
            (jointOf ms).gates.length := by
  obtain ⟨r, hr, hnode, htid, hcount, hpop, hloc⟩ := evaluate_range h hroot hk hin
  rw [← hcount] at ht
  obtain ⟨m, hm, u, hu, huv, hpos⟩ :=
    instPos_eq (pos := pos) (v := r.first + t) hperm (by omega) fun w => hloc w t ht
  exact ⟨r, hr, hnode, htid, m, hm, u, hu, huv, hpos⟩

end Properties.NetworkScan.Query
