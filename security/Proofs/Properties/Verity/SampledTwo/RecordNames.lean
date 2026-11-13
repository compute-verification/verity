module

public import Proofs.Flock.SpotFlock.HeldOf
public import Proofs.Flock.SpotFlock.Jeq
public import Proofs.Flock.SpotFlock.RunsAs.Outcome
public import Proofs.Flock.Soundness.ExecDraw
public import Proofs.Properties.Verity.SampledTwo.JpWalk

/-!
# A registration check with no code names its registration's names

`Registration.check rec v` collects R1–R8's failure codes for the record `rec` at the verifier `v`. Where it collects
none and `v` lists each port once, `rec` names `v`'s names (`recordOk_of_check`): its roots' ports are `v`'s ports and
its registered values `v`'s registered values, each once. R3-ports holds the roots' distinct ports to `v`'s ports by
count and membership, and R7-values the registered values to the verifier's own entries the same way; a record with no
registered value has no `registered` key, since its keys are exactly R1's `KEYS`.

The two checks with no code are walked down the one path that adds none (`JpWalk`'s `jp_walk`): `check`'s
(`checkFacts_of_check`) and R7's (`checkRegistered_pass`).
-/

@[expose] public section

namespace SpotFlock.RecordNames

open Lean (Json)
open Spot.Registration
open SpotFlock.HeldOf (namesOf)

/-! ## Counting names -/

theorem perm_of_nodup_subset {α : Type} [DecidableEq α] : ∀ {S D : List α}, S.Nodup → (∀ x ∈ S, x ∈ D) →
    D.length ≤ S.length → S.Perm D
  | [], D, _, _, hl => by
    cases D with
    | nil => exact List.Perm.refl _
    | cons _ _ => simp at hl
  | a :: S, D, hs, hsub, hl => by
    have ha : a ∈ D := hsub a (by simp)
    have hn := List.nodup_cons.1 hs
    have ih : S.Perm (D.erase a) := perm_of_nodup_subset hn.2
      (fun x hx => (List.mem_erase_of_ne (fun e => hn.1 (by subst e; exact hx))).2 (hsub x (by simp [hx])))
      (by rw [List.length_erase_of_mem ha]; simp at hl; omega)
    exact (List.Perm.cons a ih).trans (List.perm_cons_erase ha).symm

theorem length_eraseDups_le {α : Type} [BEq α] :
    ∀ (n : Nat) (l : List α), l.length ≤ n → l.eraseDups.length ≤ l.length
  | _, [], _ => by simp
  | 0, _ :: _, h => by simp at h
  | n + 1, a :: as, h => by
    rw [List.eraseDups_cons]
    have := length_eraseDups_le n (as.filter fun b => !b == a)
      (Nat.le_trans (List.length_filter_le _ _) (by simp at h; omega))
    have := List.length_filter_le (fun b => !b == a) as
    simp; omega

theorem eraseDups_eq_of_length {α : Type} [BEq α] [LawfulBEq α] :
    ∀ (n : Nat) (l : List α), l.length ≤ n → l.eraseDups.length = l.length → l.eraseDups = l ∧ l.Nodup
  | _, [], _, _ => by simp
  | 0, _ :: _, h, _ => by simp at h
  | n + 1, a :: as, h, he => by
    rw [List.eraseDups_cons] at he ⊢
    have h1 := length_eraseDups_le as.length (as.filter fun b => !b == a) (List.length_filter_le _ _)
    have h2 := List.length_filter_le (fun b => !b == a) as
    simp only [List.length_cons, Nat.add_right_cancel_iff] at he
    have hf : (as.filter fun b => !b == a).length = as.length := by omega
    have hfa : as.filter (fun b => !b == a) = as := List.filter_eq_self.2 (List.length_filter_eq_length_iff.1 hf)
    rw [hfa] at he ⊢
    obtain ⟨ih1, ih2⟩ := eraseDups_eq_of_length n as (by simp at h; omega) he
    refine ⟨by rw [ih1], List.nodup_cons.2 ⟨fun ha => ?_, ih2⟩⟩
    have := List.length_filter_eq_length_iff.1 hf a ha
    simp at this

theorem foldl_pushes {α β : Type} (p : Array β → α → Bool) (q : α → β) : ∀ (l : List α) (acc : Array β),
    (l.foldl (fun acc r => if p acc r then acc else acc.push (q r)) acc).size ≤ acc.size + l.length ∧
    ((l.foldl (fun acc r => if p acc r then acc else acc.push (q r)) acc).size = acc.size + l.length →
      (l.foldl (fun acc r => if p acc r then acc else acc.push (q r)) acc).toList = acc.toList ++ l.map q)
  | [], acc => by simp
  | r :: l, acc => by
    simp only [List.foldl_cons, List.length_cons, List.map_cons]
    by_cases hp : p acc r = true
    · simp only [hp, ↓reduceIte]
      have := (foldl_pushes p q l acc).1
      exact ⟨by omega, fun h => by omega⟩
    · simp only [hp, Bool.false_eq_true, ↓reduceIte]
      have ih := foldl_pushes p q l (acc.push (q r))
      simp only [Array.size_push, Array.toList_push] at ih
      refine ⟨by omega, fun h => ?_⟩
      rw [ih.2 (by omega)]
      simp

theorem foldl_pushes_array {α β : Type} (p : Array β → α → Bool) (q : α → β) (xs : Array α)
    (h : (xs.foldl (fun acc r => if p acc r then acc else acc.push (q r)) #[]).size = xs.size) :
    (xs.foldl (fun acc r => if p acc r then acc else acc.push (q r)) #[]).toList = xs.toList.map q := by
  rw [← Array.foldl_toList] at h ⊢
  have := (foldl_pushes p q xs.toList #[]).2 (by simpa using h)
  simpa using this

theorem mapM_some {α β : Type} (f : α → Option β) (g : α → β) :
    ∀ (l : List α), (∀ x ∈ l, f x = some (g x)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | x :: l, h => by
    rw [List.mapM_cons, h x (by simp), mapM_some f g l (fun y hy => h y (by simp [hy]))]
    rfl

theorem mapM_ok_list {α β γ ε : Type} (f : α → Except ε β) (g : β → γ) (k : α → γ)
    (hf : ∀ x y, f x = .ok y → g y = k x) : ∀ (xs : List α) (ys : List β), xs.mapM f = .ok ys → ys.map g = xs.map k
  | [], ys, h => by simp [List.mapM_nil, pure, Except.pure] at h; subst h; rfl
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | error e => rw [hx] at h; cases h
    | ok y =>
      rw [hx] at h
      cases hxs : xs.mapM f with
      | error e => simp [hxs, bind, Except.bind] at h
      | ok ys' =>
        simp only [hxs, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        simp [hf x y hx, mapM_ok_list f g k hf xs ys' hxs]

theorem mapM_ok_array {α β γ ε : Type} (f : α → Except ε β) (g : β → γ) (k : α → γ)
    (hf : ∀ x y, f x = .ok y → g y = k x) (xs : Array α) (ys : Array β) (h : xs.mapM f = .ok ys) :
    ys.map g = xs.map k := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => simp [hl, Functor.map, Except.map] at h
  | ok l =>
    simp only [hl, Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    apply Array.ext'
    simp [mapM_ok_list f g k hf _ _ hl]

/-! ## A key an object's keys don't name -/

open Std.DTreeMap.Internal in
theorem mem_toListModel_of_get? {δ : Type} : ∀ (t : Impl String fun _ => δ) (k : String) (v : δ),
    Impl.Const.get? t k = some v → k ∈ t.toListModel.map (·.1)
  | .leaf, k, v, h => by simp [Impl.Const.get?] at h
  | .inner _ k' v' l r, k, v, h => by
    have hl := mem_toListModel_of_get? l k v
    have hr := mem_toListModel_of_get? r k v
    simp only [Impl.toListModel_inner, List.map_append, List.map_cons, List.mem_append, List.mem_cons]
    unfold Impl.Const.get? at h
    split at h
    · exact Or.inl (hl h)
    · exact Or.inr (Or.inr (hr h))
    · rename_i he
      exact Or.inr (Or.inl (Std.LawfulEqOrd.eq_of_compare he))

theorem toList_foldl_push {δ : Type} : ∀ (l : List ((_ : String) × δ)) (acc : Array (String × δ)),
    (l.foldl (fun acc p => acc.push (p.1, p.2)) acc).toList = acc.toList ++ l.map (fun p => (p.1, p.2))
  | [], acc => by simp
  | p :: l, acc => by simp only [List.foldl_cons, toList_foldl_push l]; simp

open Std.DTreeMap.Internal in
theorem getElem?_none_of_not_key (m : Std.TreeMap.Raw String Json) (k : String)
    (h : k ∉ m.toArray.map (·.1)) : m[k]? = none := by
  cases hk : m[k]? with
  | none => rfl
  | some v =>
    exfalso
    apply h
    have h1 := mem_toListModel_of_get? m.inner.inner k v hk
    rw [Array.mem_def, Array.toList_map]
    simp only [Std.TreeMap.Raw.toArray, Std.DTreeMap.Raw.Const.toArray, Impl.Const.toArray, Impl.foldl_eq_foldl,
      toList_foldl_push]
    simpa [List.map_map] using h1

/-- **A record whose keys are R1's `KEYS` has no `registered` entry.** -/
theorem registered_null {rec : Json} (hk : Spot.keys? rec = some KEYS) : Spot.getD rec "registered" = .null := by
  cases rec with
  | obj m =>
    simp only [Spot.keys?, Option.some.injEq] at hk
    rw [SpotFlock.RunsAs.getD_obj, getElem?_none_of_not_key m "registered" (fun hm => ?_)]
    · rfl
    · have h2 := (FlockSoundness.Audit.DrawExec.mem_qsort _ (· < ·) _).2 hm
      rw [show (m.toArray.map Prod.fst).qsort (· < ·) = KEYS from hk] at h2
      unfold KEYS Spot.sortStrings at h2
      rw [FlockSoundness.Audit.DrawExec.mem_qsort] at h2
      simp at h2
  | _ => simp [Spot.keys?] at hk

/-! ## The two names -/

/-- A JSON string's text; none for anything else. -/
def strOf : Json → Option String
  | .str s => some s
  | _ => none

/-- **R3-ports names the ports**: a record whose roots' distinct ports are as many as its roots and as `ports`, each
port among them, names `ports` each once at its roots, where `ports` lists each once. -/
theorem ports_names {rec : Json} {rs : Array Json} {ports : Array (String × Json)}
    (hn : (ports.map (·.1)).toList.Nodup) (hrs : Spot.getD rec "roots" = .arr rs)
    (h1 : (portIndex rs).count = rs.size) (h2 : (portIndex rs).count = ports.size)
    (h3 : ∀ p ∈ ports, (portIndex rs).has p.1 = true) :
    namesEq (namesAt rec "roots" "port") (ports.map (·.1)).toList = true := by
  have h3' : ∀ p ∈ ports, (rs.foldl (fun (acc : Array Json) r =>
      if acc.any (Spot.jeq · (Spot.getD r "port")) then acc else acc.push (Spot.getD r "port")) #[]).any
        (Spot.jeq · (Json.str p.1)) = true := fun p hp => by rw [← portIndex_has]; exact h3 p hp
  rw [portIndex_count] at h1 h2
  have hD := foldl_pushes_array (fun (acc : Array Json) r => acc.any (Spot.jeq · (Spot.getD r "port")))
    (Spot.getD · "port") rs h1
  generalize hDdef : rs.foldl (fun (acc : Array Json) r =>
    if acc.any (Spot.jeq · (Spot.getD r "port")) then acc else acc.push (Spot.getD r "port")) #[] = D at hD h2 h3'
  generalize hP : (ports.map (·.1)).toList = P at hn ⊢
  have hmem : ∀ x ∈ P, Json.str x ∈ D.toList := by
    intro x hx
    rw [← hP, Array.toList_map, List.mem_map] at hx
    obtain ⟨p, hp, rfl⟩ := hx
    obtain ⟨i, hi, hj⟩ := Array.any_eq_true.1 (h3' p (Array.mem_def.2 hp))
    have he := str_of_jeq (jeq_symm hj)
    rw [← Array.mem_def, ← he]
    exact Array.getElem_mem hi
  have hS : (P.map Json.str).Nodup := hn.map (fun a b h => Json.str.inj h)
  have hlen : D.toList.length ≤ (P.map Json.str).length := by
    rw [← hP]
    simp [h2]
  have hperm : (P.map Json.str).Perm D.toList := by
    classical
    exact perm_of_nodup_subset hS (fun x hx => by
      obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
      exact hmem y hy) hlen
  have hstr : ∀ e ∈ rs.toList, ∃ s ∈ P, Spot.getD e "port" = .str s := by
    intro e he
    have : Spot.getD e "port" ∈ D.toList := by rw [hD]; exact List.mem_map_of_mem he
    obtain ⟨s, hs, he'⟩ := List.mem_map.1 (hperm.mem_iff.2 this)
    exact ⟨s, hs, he'.symm⟩
  let g : Json → String := fun e => (strOf (Spot.getD e "port")).getD ""
  have hmap : namesAt rec "roots" "port" = some (rs.toList.map g) := by
    unfold namesAt
    rw [hrs]
    exact mapM_some _ g rs.toList (fun e he => by
      obtain ⟨s, -, hs⟩ := hstr e he
      simp only [g, hs, strOf]
      rfl)
  have hN : (rs.toList.map g).Perm P := by
    have h4 : rs.toList.map g = D.toList.map (fun j => (strOf j).getD "") := by
      rw [hD, List.map_map]
      rfl
    rw [h4]
    have := hperm.symm.map (fun j => (strOf j).getD "")
    simpa [List.map_map, Function.comp_def, strOf] using this
  rw [hmap]
  simp only [namesEq, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.contains_iff_mem]
  exact ⟨⟨hN.nodup_iff.2 hn, fun x hx => hN.mem_iff.1 hx⟩, fun x hx => hN.mem_iff.2 hx⟩

/-- **R7-values names the registered values**: a record whose `registered` entries' values are strings, distinct, as
many as `R` and each in `R`, names `R` each once there. -/
theorem values_names {rec : Json} {g : Array Json} {R : List String}
    (hg : Spot.getD rec "registered" = .arr g)
    (h1 : ((g.map (Spot.getD · "value")).filterMap strOf).size = g.size)
    (h2 : ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups.length = g.size)
    (h3 : R.length ≤ g.size)
    (h4 : ∀ x ∈ ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups, x ∈ R) :
    namesEq (namesAt rec "registered" "value") R = true := by
  generalize hN : ((g.map (Spot.getD · "value")).filterMap strOf).toList = N at h2 h4
  have hNl : N.length = g.size := by rw [← hN, Array.length_toList]; exact h1
  obtain ⟨hE, hNd⟩ := eraseDups_eq_of_length N.length N (Nat.le_refl _) (by rw [h2, hNl])
  have hN' : N = (g.toList.map (Spot.getD · "value")).filterMap strOf := by
    rw [← hN, Array.toList_filterMap, Array.toList_map]
  have hsome : ∀ e ∈ g.toList, (strOf (Spot.getD e "value")).isSome := by
    have : ((g.toList.map (Spot.getD · "value")).filterMap strOf).length =
        (g.toList.map (Spot.getD · "value")).length := by
      rw [← hN', hNl, List.length_map, Array.length_toList]
    intro e he
    exact List.filterMap_length_eq_length.1 this _ (List.mem_map_of_mem he)
  have hlist : ∀ (l : List Json), (∀ e ∈ l, (strOf (Spot.getD e "value")).isSome) →
      l.mapM (fun e => match Spot.getD e "value" with | .str s => some s | _ => none) =
        some ((l.map (Spot.getD · "value")).filterMap strOf) := by
    intro l
    induction l with
    | nil => intro _; rfl
    | cons e l ih =>
      intro h
      obtain ⟨s, hs⟩ := Option.isSome_iff_exists.1 (h e (by simp))
      have he : Spot.getD e "value" = .str s := by
        revert hs
        unfold strOf
        split <;> intro hs <;> simp_all
      rw [List.mapM_cons, ih (fun e' he' => h e' (by simp [he']))]
      simp [he, strOf]
  have hmap : namesAt rec "registered" "value" = some N := by
    unfold namesAt
    rw [hg, hN']
    exact hlist _ hsome
  rw [hmap]
  have hperm : N.Perm R := perm_of_nodup_subset hNd (fun x hx => h4 x (by rw [hE]; exact hx)) (by omega)
  simp only [namesEq, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.contains_iff_mem]
  exact ⟨⟨hNd, fun x hx => hperm.mem_iff.1 hx⟩, fun x hx => hperm.mem_iff.2 hx⟩

/-! ## The walks -/

/-- **R7-values passed**: the record's `registered` is a list whose values are strings, distinct, as many as `mine`'s
and each one of `mine`'s values. -/
def ValuesHold (g : Array Json) (mine : Array Spot.Registered.Entry) : Prop :=
  ((g.map (Spot.getD · "value")).filterMap strOf).size = g.size ∧
  ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups.length = g.size ∧
  ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups.length = mine.size ∧
  ∀ x ∈ ((g.map (Spot.getD · "value")).filterMap strOf).toList.eraseDups, ∃ e ∈ mine, e.value = x

/-- R7-values' test as `checkRegistered` writes it, at any reading `f` of a string that is `strOf`. -/
theorem valuesHold_of {g : Array Json} {mine : Array Spot.Registered.Entry} {f : Json → Option String}
    (hf : ∀ v, f v = strOf v)
    (h : ¬(((g.map (Spot.getD · "value")).filterMap f).size != (g.map (Spot.getD · "value")).size ||
          ((g.map (Spot.getD · "value")).filterMap f).toList.eraseDups.length != (g.map (Spot.getD · "value")).size ||
          ((g.map (Spot.getD · "value")).filterMap f).toList.eraseDups.length != mine.size ||
          !((g.map (Spot.getD · "value")).filterMap f).toList.eraseDups.all fun v => mine.any fun x => x.value == v) =
        true) : ValuesHold g mine := by
  obtain rfl : f = strOf := funext hf
  unfold ValuesHold
  generalize ((g.map (Spot.getD · "value")).filterMap strOf) = N at h ⊢
  rw [Array.size_map] at h
  simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not, Bool.not_eq_true', Bool.not_eq_false] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  refine ⟨h1, h2, h3, fun x hx => ?_⟩
  obtain ⟨i, hi, hj⟩ := Array.any_eq_true.1 (List.all_eq_true.1 h4 x hx)
  exact ⟨mine[i], Array.getElem_mem hi, eq_of_beq hj⟩

/-- **`checkRegistered` with no code passed R7-values**: the record's `registered` is a list `ValuesHold` holds to the
entries. The walk stops at the loops, which it does not read. -/
theorem checkRegistered_pass {c : Array String} {got : Json} {mine : Array Spot.Registered.Entry}
    {held : Array (String × String)} {supplied : Array (String × Spot.Supplied.Rows)}
    {rootOf : String → Spot.Supplied.Rows → Spot.Registered.Entry → Flock.V String}
    (h' : (checkRegistered c got (.ok mine) held supplied rootOf).isEmpty = true) :
    ∃ g, got = .arr g ∧ ValuesHold g mine := by
  unfold checkRegistered at h'
  cases got with
  | arr g =>
    refine ⟨g, rfl, ?_⟩
    dsimp (config := { zeta := false }) only at h'
    lift_lets at h'
    extract_lets at h'
    name_lets
    jp_mono []
    clear_value jp0
    jp_walk h' []
    refine valuesHold_of (f := _) (fun v => ?_) hc1
    cases v <;> rfl
  | _ => simp [Id.run, pure, isEmpty_addCode] at h'

/-- **`checkOutputs` with no code passed R7-values** at the entries `applyOutputs` leaves, whose values are the
verifier's own entries'. -/
theorem checkOutputs_pass {c : Array String} {got : Json} {mine : Flock.V (Array Spot.Registered.Entry)}
    {held : Array (String × String)} {supplied : Array (String × Spot.Supplied.Rows)}
    {rootOf : String → Spot.Supplied.Rows → Spot.Registered.Entry → Flock.V String}
    {outputs : Array (String × String × Earlier)} {log : Array Json}
    (h : (checkOutputs c got mine held supplied rootOf outputs log).isEmpty = true) :
    ∃ m g m', mine = .ok m ∧ got = .arr g ∧ m'.map (·.value) = m.map (·.value) ∧ ValuesHold g m' := by
  unfold checkOutputs at h
  cases mine with
  | error e => simp [isEmpty_addCode] at h
  | ok m =>
    have hv := applyOutputs_values c m held outputs log
    dsimp only at h
    generalize applyOutputs c m held outputs log = s at h hv
    obtain ⟨c', m', held'⟩ := s
    obtain ⟨g, hg, hvh⟩ := checkRegistered_pass (c := c') (mine := m') (held := held') h
    exact ⟨m, g, m', rfl, hg, hv, hvh⟩

/-- **What a check with no code says of the names**: the record's keys are R1's, its roots a list R3-ports holds to the
ports, and with registered values, its `registered` a list R7-values holds to the verifier's own entries. -/
def CheckFacts (rec : Json) (v : Verifier) : Prop :=
  Spot.keys? rec = some (if !v.registered.isEmpty then KEYS_V2 else KEYS) ∧
  (∃ rs, Spot.getD rec "roots" = .arr rs ∧ (portIndex rs).count = rs.size ∧ (portIndex rs).count = v.ports.size ∧
    ∀ p ∈ v.ports, (portIndex rs).has p.1 = true) ∧
  (v.registered.isEmpty = false → ∃ mine g mine', ownEntries v = .ok mine ∧ Spot.getD rec "registered" = .arr g ∧
    mine'.map (·.value) = mine.map (·.value) ∧ ValuesHold g mine')

set_option maxHeartbeats 4000000 in
theorem checkFacts_of_check {rec : Json} {v : Verifier} (h : check rec v = #[]) : CheckFacts rec v := by
  have h' : (check rec v).isEmpty = true := by rw [h]; rfl
  unfold check at h'
  lift_lets at h'
  extract_lets at h'
  name_lets
  cases hv2 : v.registered.isEmpty
  all_goals
    jp_mono [poolMarks, r8Codes, isEmpty_checkOutputs, hv2, l_v2]
    jp_walk h' [poolMarks, r8Codes, isEmpty_checkOutputs, hv2, l_v2]
  -- the branch of `if v2` the registered values rule out
  all_goals
    first
    | (exfalso; exact ‹¬(!false) = true› rfl)
    | (exfalso; exact absurd ‹(!true) = true› (by decide))
    | skip
  -- `hc0` is the keys' test, `hc9` R3-roots' and `hc10` R3-ports'
  all_goals
    have hk : Spot.keys? rec = some (if !v.registered.isEmpty then KEYS_V2 else KEYS) := by
      rw [hv2]
      simpa using hc0
    have hr : ∃ rs, Spot.getD rec "roots" = .arr rs ∧ l_roots = rs := by
      cases hr : Spot.getD rec "roots" with
      | arr rs => exact ⟨rs, rfl, by simp only [l_roots, l_get, hr]⟩
      | _ => exact absurd (by simp [l_roots, l_get, hr]) hc9
    obtain ⟨rs, hrs, hl⟩ := hr
    simp only [l_ix, hl, Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not, Bool.not_eq_true',
      Bool.not_eq_false] at hc10
    obtain ⟨⟨hp1, hp2⟩, hp3⟩ := hc10
    refine ⟨hk, ⟨rs, hrs, hp1, hp2, fun p hp => ?_⟩, fun hne => ?_⟩
    · exact Array.all_eq_true_iff_forall_mem.1 hp3 p hp
    · first
        | (rw [hv2] at hne; cases hne; done)
        | exact checkOutputs_pass h'

/-! ## The record names its registration's names -/

theorem entries_values {d : Flock.TemplateQuery.Desc} {ps : Array Spot.Registered.Param} {values : Array (String × Nat)}
    {schemas : Array (String × String)} {window : Option String} {fixed : Array String}
    {es : Array Spot.Registered.Entry} (h : Spot.Registered.entries d ps values schemas window fixed = .ok es) :
    es.map (·.value) = values.map (·.1) := by
  unfold Spot.Registered.entries at h
  refine mapM_ok_array _ (·.value) (·.1) (fun x y hx => ?_) _ _ h
  obtain ⟨v, w⟩ := x
  simp only [bind, Except.bind, pure, Except.pure] at hx
  repeat'
    first
    | split at hx
    | focus (cases hx; done)
    | (cases hx; rfl)

theorem recordOk_of_facts {rec : Json} {v : Verifier} (hf : CheckFacts rec v) (hn : (v.ports.map (·.1)).toList.Nodup) :
    recordOk rec (namesOf v) = true := by
  obtain ⟨hk, ⟨rs, hrs, h1, h2, h3⟩, h7⟩ := hf
  unfold recordOk
  rw [Bool.and_eq_true]
  refine ⟨ports_names hn hrs h1 h2 h3, ?_⟩
  show namesEq (namesAt rec "registered" "value") (v.registered.map (·.1)).toList = true
  cases hv : v.registered.isEmpty
  · obtain ⟨mine, g, mine', hmine, hg, hm', hv1, hv2, hv3, hv4⟩ := h7 hv
    have hvals : mine'.map (·.value) = v.registered.map (·.1) := by rw [hm', entries_values hmine]
    have hR : (v.registered.map (·.1)).toList.length = mine'.size := by
      have := congrArg Array.size hvals
      simpa using this.symm
    refine values_names hg hv1 hv2 (by omega) (fun x hx => ?_)
    obtain ⟨e, he, rfl⟩ := hv4 x hx
    have : e.value ∈ mine'.map (·.value) := Array.mem_map_of_mem he
    rw [hvals] at this
    exact Array.mem_def.1 this
  · simp only [hv, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hk
    have : v.registered = #[] := Array.isEmpty_iff.1 hv
    unfold namesAt
    rw [registered_null hk, this]
    simp [namesEq]

/-- **A registration check with no code names the registration's names**, where the verifier lists each port once. -/
theorem recordOk_of_check {rec : Json} {v : Verifier} (h : check rec v = #[])
    (hn : (v.ports.map (·.1)).toList.Nodup) : recordOk rec (namesOf v) = true :=
  recordOk_of_facts (checkFacts_of_check h) hn

end SpotFlock.RecordNames
