module

public import Proofs.Properties.Nci.ClosureDigest
public import Proofs.Properties.Challenger.CanonLine
public import Proofs.Flock.Claims.Complete.Json.Parse
public import Proofs.Flock.Bytes
public import Std.Data.TreeMap.Raw.Lemmas
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# One closure digest, one canonical entry for each Definition it reaches

Two descriptors that give `key` one closure digest hash, by `closureDigest_spec`, two canonical objects with one
SHA-512. Either those two byte strings differ, or the objects print alike and so map the same ids to the same digests
(`CanonLine.canon_inj`): `key` reaches the same Definitions in both, each with one `definitionDigest`. Each such digest
is the SHA-512 of a canonical entry's text, so again either the two texts differ, or the two canonical entries are
equivalent. A difference anywhere is `Collides` (`entries_agree`).

`canon_inj` reads sorted values; a parsed descriptor is well formed (`FlockComplete.JsonRT.parse_wf`), and
`canonicalEntry` keeps it so (`wf_canonicalEntry`): it inserts into the entry's trees, reads the interning tables, and
writes each flat reference sequence (`flatParts`) of parts the entry holds. `canonicalEntry_spec` says what it builds,
node by node (`Steps`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Claims.Json (Wf)
open NciCheck.DefinitionDigest (isComposite flatParts written canonicalEntry definitionDigest closureDigest)

/-- `canonicalEntry`'s field write: an object's field set, anything else as it is. -/
def setJ (j : Json) (k : String) (v : Json) : Json := match j with
  | .obj kvs => .obj (kvs.insert k v)
  | j => j

/-- `canonicalEntry`'s table read: entry `i` of `xs`, or `null`. -/
def atJ (xs : Array Json) (i : Json) : Json := match i.getNat? with
  | .ok i => xs.getD i .null
  | .error _ => .null

/-- An argument as the canonical entry writes it. -/
def canonArg (types : Array Json) (x : Json × Array Json) : Json :=
  setJ (setJ x.1 "type" (atJ types (x.1.getObjValD "type"))) "refs" (written x.2)

/-- A node as the canonical entry writes it, from its arguments and their flat sequences. -/
def canonNode (callees types : Array Json) (n : Json) (args : Array Json) (row : Array (Array Json)) : Json :=
  setJ (setJ (setJ n "fn" (atJ callees (n.getObjValD "fn"))) "out" (atJ types (n.getObjValD "out"))) "args"
    (.arr ((args.zip row).map (canonArg types)))

/-- `canonicalEntry`'s loop over the nodes `l`, from `rows` and `out` to `rows'` and `out'`. -/
inductive Steps (callees types : Array Json) :
    Array (Array (Array Json)) → Array Json → List Json → Array (Array (Array Json)) → Array Json → Prop
  | nil (rows out) : Steps callees types rows out [] rows out
  | cons {rows out n l rows' out'} (args : Array Json) (row : Array (Array Json)) :
      n.getObjValD "args" = .arr args →
      args.mapM (fun a => flatParts rows Flock.Steps.FUEL (a.getObjValD "refs")) = .ok row →
      Steps callees types (rows.push row) (out.push (canonNode callees types n args row)) l rows' out' →
      Steps callees types rows out (n :: l) rows' out'

/-- **What `canonicalEntry` builds**: the entry itself, or a composite's with its nodes and return rewritten. -/
theorem canonicalEntry_spec {callees types : Array Json} {e ce : Json} (h : canonicalEntry callees types e = .ok ce) :
    ce = e ∨ ∃ nodes rows out ret, isComposite e = true ∧
      (e.getObjValD "body").getObjValD "nodes" = .arr nodes ∧ Steps callees types #[] #[] nodes.toList rows out ∧
      flatParts rows Flock.Steps.FUEL ((e.getObjValD "body").getObjValD "ret") = .ok ret ∧
      ce = setJ e "body" (setJ (setJ (e.getObjValD "body") "nodes" (.arr out)) "ret" (written ret)) := by
  unfold canonicalEntry at h
  simp only [bind, Except.bind] at h
  split at h
  · split at h
    · rename_i nodes hn
      split at h
      · cases h
      rename_i r hr
      split at h
      · cases h
      rename_i ret hret
      simp only [pure, Except.pure, Except.ok.injEq] at h
      refine .inr ⟨nodes, r.1, r.2, ret, by assumption, hn, ?_, hret, h.symm⟩
      rw [← Array.forIn_toList] at hr
      clear h hret hn
      generalize (#[] : Array (Array (Array Json))) = rows0 at hr
      generalize (#[] : Array Json) = out0 at hr
      generalize nodes.toList = l at hr ⊢
      induction l generalizing rows0 out0 with
      | nil =>
        simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at hr
        subst hr
        exact .nil _ _
      | cons n l ih =>
        rw [List.forIn_cons] at hr
        simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        rename_i s hs
        split at hs
        · rename_i args ha
          split at hs
          · cases hs
          rename_i row hrow
          simp only [pure, Except.pure, Except.ok.injEq] at hs
          subst hs
          exact .cons args row ha hrow (ih _ _ hr)
        · rename_i x hx
          simp only [throw, throwThe, MonadExceptOf.throw] at hs
          cases hs
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      exact .inl h.symm
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    exact .inl h.symm

theorem wf_getObjValD {j : Json} (hj : Wf j) (k : String) : Wf (j.getObjValD k) := by
  cases hj with
  | obj hm h =>
    unfold Json.getObjValD Json.getObjVal?
    dsimp only
    split
    · rename_i v hv; exact h k v hv
    · exact .null
  | _ => exact .null

theorem wf_setJ {j v : Json} (hj : Wf j) (hv : Wf v) (k : String) : Wf (setJ j k v) := by
  cases hj with
  | obj hm h =>
    refine .obj hm.insert fun k' v' h' => ?_
    rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm] at h'
    split at h'
    · cases h'; exact hv
    · exact h k' v' h'
  | null => exact .null
  | bool b => exact .bool b
  | num n => exact .num n
  | str s => exact .str s
  | arr h => exact .arr h

theorem wf_atJ {xs : Array Json} (hx : ∀ x ∈ xs, Wf x) (i : Json) : Wf (atJ xs i) := by
  unfold atJ
  split
  · rename_i n _
    rw [Array.getD_eq_getD_getElem?]
    cases h : xs[n]? with
    | none => exact .null
    | some x => exact hx x (Array.mem_of_getElem? h)
  · exact .null

theorem mem_foldl_append {α : Type} : ∀ (l : List (Array α)) (acc : Array α) (p : α),
    p ∈ l.foldl (· ++ ·) acc → p ∈ acc ∨ ∃ y ∈ l, p ∈ y
  | [], _, _, h => .inl h
  | y :: l, acc, p, h => by
    rcases mem_foldl_append l (acc ++ y) p h with h | ⟨z, hz, hp⟩
    · rcases Array.mem_append.1 h with h | h
      · exact .inl h
      · exact .inr ⟨y, List.mem_cons_self, h⟩
    · exact .inr ⟨z, List.mem_cons_of_mem _ hz, hp⟩

theorem list_mapM_mem {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → ∀ y ∈ r, ∃ x ∈ l, f x = .ok y
  | [], r, h, y, hy => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; cases hy
  | x :: l, r, h, y, hy => by
    rw [List.mapM_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    rename_i a ha
    split at h
    · cases h
    rename_i r' hr'
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨x, List.mem_cons_self, ha⟩
    · obtain ⟨z, hz, e⟩ := list_mapM_mem f l r' hr' y hy
      exact ⟨z, List.mem_cons_of_mem _ hz, e⟩

theorem array_mapM_mem {α β ε : Type} (f : α → Except ε β) (xs : Array α) (r : Array β)
    (h : xs.mapM f = .ok r) : ∀ y ∈ r, ∃ x ∈ xs, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [Functor.map, Except.map] at h
  split at h
  · cases h
  rename_i l hl
  injection h with h
  subst h
  intro y hy
  obtain ⟨x, hx, e⟩ := list_mapM_mem f xs.toList l hl y (by simpa using hy)
  exact ⟨x, Array.mem_toList_iff.1 hx, e⟩

theorem wf_mkObj1 {k : String} {v : Json} (hv : Wf v) : Wf (Json.mkObj [(k, v)]) := by
  refine .obj Std.TreeMap.Raw.WF.ofList fun k' v' h => ?_
  rw [Std.TreeMap.Raw.get?_eq_getElem?] at h
  by_cases hk : compare k k' = .eq
  · rw [Std.TreeMap.Raw.getElem?_ofList_of_mem hk (by simp) List.mem_cons_self] at h
    cases h; exact hv
  · rw [Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false] at h
    · cases h
    · simp only [List.map_cons, List.map_nil, List.contains_cons, List.contains_nil, Bool.or_false]
      rw [beq_eq_false_iff_ne]
      intro e; subst e; exact hk Std.ReflCmp.compare_self

theorem wf_written {ps : Array Json} (hp : ∀ p ∈ ps, Wf p) : Wf (written ps) := by
  rcases ps with ⟨l⟩
  rcases l with _ | ⟨p, _ | ⟨q, l⟩⟩
  · exact wf_mkObj1 (.arr hp)
  · exact hp p (by simp)
  · exact wf_mkObj1 (.arr hp)

/-- Every part `flatParts` returns is well formed, when the rows' parts and the sequence are. -/
theorem wf_flatParts {rows : Array (Array (Array Json))} (hr : ∀ r ∈ rows, ∀ a ∈ r, ∀ p ∈ a, Wf p) :
    ∀ (fuel : Nat) (j : Json) (ps : Array Json), Wf j → flatParts rows fuel j = .ok ps → ∀ p ∈ ps, Wf p := by
  intro fuel
  induction fuel with
  | zero => intro j ps _ h; cases h
  | succ n ih =>
    intro j ps hj h
    delta flatParts at h
    dsimp only at h
    generalize hb : (Nat.rec ⟨flatParts._f rows Nat.zero PUnit.unit, PUnit.unit⟩
      (fun n n_ih => ⟨flatParts._f rows n.succ n_ih, n_ih⟩) n :
        Nat.below (motive := fun _ => Json → Flock.V (Array Json)) (n + 1)) = b at h
    have hb1 : ∀ x ps, Wf x → b.1 x = .ok ps → ∀ p ∈ ps, Wf p := by
      subst hb
      intro x ps hx h'
      cases n with
      | zero => exact ih x ps hx h'
      | succ m =>
        delta flatParts at ih
        exact ih x ps hx h'
    clear hb ih
    delta flatParts._f at h
    dsimp only at h
    have concat : (match j.getObjVal? "concat" with
        | Except.ok (Json.arr ps) => do
          let __do_lift ← Array.mapM b.1 ps
          pure (Array.foldl (fun x1 x2 => x1 ++ x2) #[] __do_lift)
        | _ => pure #[j] : Flock.V (Array Json)) = .ok ps → ∀ p ∈ ps, Wf p := by
      intro h
      split at h
      · rename_i cs hc
        simp only [bind, Except.bind] at h
        split at h
        · cases h
        rename_i r hr
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        have hcw : ∀ c ∈ cs, Wf c := by
          have := wf_getObjValD hj "concat"
          unfold Json.getObjValD at this
          rw [hc] at this
          cases this with
          | arr h => exact h
        intro p hp
        rw [← Array.foldl_toList] at hp
        rcases mem_foldl_append r.toList #[] p hp with hp | ⟨y, hy, hp⟩
        · simp at hp
        · obtain ⟨x, hx, e⟩ := array_mapM_mem _ cs r hr y (Array.mem_toList_iff.1 hy)
          exact hb1 x y (hcw x hx) e p hp
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        intro p hp
        rw [Array.mem_singleton] at hp
        subst hp
        exact hj
    split at h
    · rename_i a ha
      split at h
      · rename_i ps' hps
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
        all_goals try simp at hps
        rcases l with _ | ⟨k, _ | ⟨x, _ | ⟨y, l⟩⟩⟩
        all_goals conv at hps => lhs; whnf
        all_goals try (simp at hps; done)
        split at hps
        · rename_i k' x' _ _
          intro p hp
          cases hrow : rows[k']? with
          | none => rw [hrow] at hps; cases hps
          | some r =>
            rw [hrow] at hps
            simp only [Option.bind_some] at hps
            exact hr r (Array.mem_of_getElem? hrow) ps' (Array.mem_of_getElem? hps) p hp
        · cases hps
      · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    · exact concat h

/-- The rows' parts, each well formed. -/
def RowsWf (rows : Array (Array (Array Json))) : Prop := ∀ r ∈ rows, ∀ a ∈ r, ∀ p ∈ a, Wf p

theorem wf_canonNode {callees types : Array Json} (hc : ∀ x ∈ callees, Wf x) (ht : ∀ x ∈ types, Wf x) {n : Json}
    (hn : Wf n) {args : Array Json} (ha : ∀ a ∈ args, Wf a) {row : Array (Array Json)} (hrow : ∀ a ∈ row, ∀ p ∈ a, Wf p) :
    Wf (canonNode callees types n args row) := by
  refine wf_setJ (wf_setJ (wf_setJ hn (wf_atJ hc _) _) (wf_atJ ht _) _) (.arr fun x hx => ?_) _
  obtain ⟨y, hy, rfl⟩ := Array.mem_map.1 hx
  have hy' := Array.mem_toList_iff.2 hy
  rw [Array.toList_zip] at hy'
  have h1 := List.of_mem_zip hy'
  exact wf_setJ (wf_setJ (ha _ (Array.mem_toList_iff.1 h1.1)) (wf_atJ ht _) _)
    (wf_written (hrow _ (Array.mem_toList_iff.1 h1.2))) _

theorem wf_steps {callees types : Array Json} (hc : ∀ x ∈ callees, Wf x) (ht : ∀ x ∈ types, Wf x)
    {rows out l rows' out'} (h : Steps callees types rows out l rows' out') :
    RowsWf rows → (∀ o ∈ out, Wf o) → (∀ n ∈ l, Wf n) → RowsWf rows' ∧ ∀ o ∈ out', Wf o := by
  induction h with
  | nil => exact fun h1 h2 _ => ⟨h1, h2⟩
  | @cons rows out n l rows' out' args row hargs hrow _ ih =>
    intro h1 h2 h3
    have hn : Wf n := h3 n List.mem_cons_self
    have ha : ∀ a ∈ args, Wf a := by
      have := wf_getObjValD hn "args"
      rw [hargs] at this
      cases this with
      | arr h => exact h
    have hr : ∀ a ∈ row, ∀ p ∈ a, Wf p := by
      intro a hm p hp
      obtain ⟨x, hx, e⟩ := array_mapM_mem _ args row hrow a hm
      exact wf_flatParts h1 _ _ _ (wf_getObjValD (ha x hx) _) e p hp
    refine ih (fun r hm => ?_) (fun o hm => ?_) (fun m hm => h3 m (List.mem_cons_of_mem _ hm))
    · rcases Array.mem_push.1 hm with hm | rfl
      · exact h1 r hm
      · exact hr
    · rcases Array.mem_push.1 hm with hm | rfl
      · exact h2 o hm
      · exact wf_canonNode hc ht hn ha hr

/-- **A canonical entry is well formed** when the entry and the interning tables are. -/
theorem wf_canonicalEntry {callees types : Array Json} (hc : ∀ x ∈ callees, Wf x) (ht : ∀ x ∈ types, Wf x)
    {e ce : Json} (he : Wf e) (h : canonicalEntry callees types e = .ok ce) : Wf ce := by
  rcases canonicalEntry_spec h with rfl | ⟨nodes, rows, out, ret, -, hn, hs, hret, rfl⟩
  · exact he
  have hb := wf_getObjValD he "body"
  have hnw : ∀ n ∈ nodes, Wf n := by
    have := wf_getObjValD hb "nodes"
    rw [hn] at this
    cases this with
    | arr h => exact h
  obtain ⟨hr, ho⟩ := wf_steps hc ht hs (fun _ h => absurd h (by simp)) (fun _ h => absurd h (by simp))
    (fun n h => hnw n (Array.mem_toList_iff.1 h))
  exact wf_setJ he (wf_setJ (wf_setJ hb (.arr ho) _)
    (wf_written (wf_flatParts hr _ _ _ (wf_getObjValD hb _) hret)) _) _


/-- **A well-formed value is sorted.** -/
theorem sorted_of_wf {j : Json} (hj : Wf j) : Spot.sortedKeys j = true := by
  induction hj with
  | null => simp [Spot.sortedKeys]
  | bool => simp [Spot.sortedKeys]
  | num => simp [Spot.sortedKeys]
  | str => simp [Spot.sortedKeys]
  | arr _ ih => exact Properties.Challenger.CanonRead.sortedKeys_arr.2 ih
  | @obj m hm _ ih =>
    refine Properties.Challenger.CanonRead.sortedKeys_obj.2 ⟨?_, fun p hp => ?_⟩
    · rw [Std.TreeMap.Raw.toList_toArray, Properties.Challenger.CanonRead.increasing_iff, List.pairwise_map]
      exact (Std.TreeMap.Raw.ordered_keys_toList hm).imp fun hlt => compareOfLessAndEq_eq_lt.1 hlt
    · have hp' : p ∈ m.toList := by
        rw [← Std.TreeMap.Raw.toList_toArray]
        exact Array.mem_toList_iff.2 hp
      refine ih p.1 p.2 ?_
      rw [Std.TreeMap.Raw.get?_eq_getElem?]
      exact (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp'

theorem wf_tab {j : Json} (hj : Wf j) (k : String) : ∀ x ∈ tab j k, Wf x := by
  have := wf_getObjValD hj k
  unfold tab
  split
  · rename_i xs hx
    rw [hx] at this
    cases this with
    | arr h => exact h
  · intro x hx; simp at hx

theorem wf_entry {j : Json} (hj : Wf j) (k : String) : Wf (entry j k) :=
  wf_getObjValD (wf_getObjValD hj _) _

theorem definitionDigest_ok {callees types : Array Json} {e : Json} {d : String}
    (h : definitionDigest callees types e = .ok d) :
    ∃ ce s, canonicalEntry callees types e = .ok ce ∧ Flock.canon ce = .ok s ∧
      d = Flock.hexOf (Flock.Sha512.hash s.toUTF8) := by
  unfold definitionDigest at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i ce hce
  split at h
  · cases h
  rename_i s hs
  simp only [pure, Except.pure, Except.ok.injEq] at h
  exact ⟨ce, s, hce, hs, h.symm⟩

theorem distinct_of_nodup {l : List (String × Json)} (h : (l.map Prod.fst).Nodup) :
    l.Pairwise fun a b => ¬ compare a.1 b.1 = .eq := by
  rw [List.Nodup, List.pairwise_map] at h
  exact h.imp fun hne he => hne (Std.LawfulEqCmp.eq_of_compare he)

theorem get?_ofList_mem {l : List (String × Json)} (hd : l.Pairwise fun a b => ¬ compare a.1 b.1 = .eq)
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Std.TreeMap.Raw.ofList l compare).get? k = some v := by
  rw [Std.TreeMap.Raw.get?_eq_getElem?]
  exact Std.TreeMap.Raw.getElem?_ofList_of_mem Std.ReflCmp.compare_self hd hm

theorem mem_of_get?_ofList {l : List (String × Json)} (hd : l.Pairwise fun a b => ¬ compare a.1 b.1 = .eq)
    {k : String} {v : Json} (hv : (Std.TreeMap.Raw.ofList l compare).get? k = some v) : (k, v) ∈ l := by
  cases hc : (l.map Prod.fst).contains k with
  | false =>
    rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hc] at hv
    cases hv
  | true =>
    obtain ⟨p, hp, hpk⟩ := List.mem_map.1 (List.contains_iff_mem.1 hc)
    have e := get?_ofList_mem hd (k := k) (v := p.2) (by rw [← hpk]; exact hp)
    rw [e] at hv
    cases hv
    rw [← hpk]
    exact hp

/-- The object `closureDigest` hashes: well formed, as its values are strings. -/
theorem wf_seen {seen : List (String × Json)} (hd : seen.Pairwise fun a b => ¬ compare a.1 b.1 = .eq)
    (hs : ∀ p ∈ seen, ∃ d : String, p.2 = .str d) : Wf (Json.mkObj seen) := by
  refine .obj Std.TreeMap.Raw.WF.ofList fun k v hv => ?_
  obtain ⟨d, hd'⟩ := hs _ (mem_of_get?_ofList hd hv)
  rw [show v = Json.str d from hd']
  exact .str d

theorem getD_seen {seen : List (String × Json)} (hd : seen.Pairwise fun a b => ¬ compare a.1 b.1 = .eq)
    {k : String} {v : Json} (hm : (k, v) ∈ seen) : Spot.getD (Json.mkObj seen) k = v := by
  rw [Json.mkObj, Properties.Challenger.CanonRead.getD_obj, get?_ofList_mem hd hm]
  rfl

theorem mem_of_getD_seen {seen : List (String × Json)} (hd : seen.Pairwise fun a b => ¬ compare a.1 b.1 = .eq)
    {k d : String} (h : Spot.getD (Json.mkObj seen) k = .str d) : (k, .str d) ∈ seen := by
  rw [Json.mkObj, Properties.Challenger.CanonRead.getD_obj] at h
  cases hv : (Std.TreeMap.Raw.ofList seen compare).get? k with
  | none => rw [hv] at h; cases h
  | some v =>
    rw [hv] at h
    cases h
    exact mem_of_get?_ofList hd hv

/-- **One closure digest, one canonical entry for each Definition reached**: two well-formed descriptors that give
`key` one closure digest reach the same Definitions from it, and each has, in the two, equivalent canonical entries,
or two of the byte strings the digests hashed differ and have one SHA-512. -/
theorem entries_agree {j j' : Json} {key h : String} (hj : Wf j) (hj' : Wf j')
    (H : closureDigest j key = .ok h) (H' : closureDigest j' key = .ok h) :
    Collides j j' key ∨ ∀ k, Reaches j key k → Reaches j' key k ∧
      ∃ ce ce', canonicalEntry (tab j "callees") (tab j "types") (entry j k) = .ok ce ∧
        canonicalEntry (tab j' "callees") (tab j' "types") (entry j' k) = .ok ce' ∧ FlockDeploy.JsonEquiv ce ce' := by
  by_cases hC : Collides j j' key
  · exact .inl hC
  refine .inr fun k hk => ?_
  obtain ⟨seen, s, hR, hD, hN, hs, he⟩ := closureDigest_spec H
  obtain ⟨seen', s', hR', hD', hN', hs', he'⟩ := closureDigest_spec H'
  have hsh := Properties.NetworkRecords.Bytes.hexOf_inj (he.symm.trans he')
  have hss : s.toUTF8 = s'.toUTF8 := by
    refine Classical.byContradiction fun hne => hC ⟨_, _, .inr ⟨seen.toList, s, hR, hD, hs, rfl⟩,
      .inr ⟨seen'.toList, s', hR', hD', hs', rfl⟩, hne, hsh⟩
  obtain rfl : s = s' := String.toByteArray_inj.mp (by simpa only [String.toUTF8_eq_toByteArray] using hss)
  have hd := distinct_of_nodup hN
  have hd' := distinct_of_nodup hN'
  have strs : ∀ (seen : Array (String × Json)) (j : Json), (∀ p ∈ seen.toList, ∃ d,
      definitionDigest (tab j "callees") (tab j "types") (entry j p.1) = .ok d ∧ p.2 = .str d) →
      ∀ p ∈ seen.toList, ∃ d : String, p.2 = .str d := fun _ _ hD p hp => by
    obtain ⟨d, -, e⟩ := hD p hp
    exact ⟨d, e⟩
  have heq := Properties.Challenger.CanonLine.canon_inj (sorted_of_wf (wf_seen hd (strs _ _ hD)))
    (sorted_of_wf (wf_seen hd' (strs _ _ hD'))) hs hs'
  obtain ⟨p, hp, hpk⟩ := (hR k).1 hk
  obtain ⟨d, hdig, hpd⟩ := hD p hp
  rw [hpk] at hdig
  have hm : (k, Json.str d) ∈ seen.toList := by
    have : p = (k, .str d) := Prod.ext hpk hpd
    exact this ▸ hp
  have g := Properties.Challenger.CanonRead.getD_equiv (sorted_of_wf (wf_seen hd (strs _ _ hD)))
    (sorted_of_wf (wf_seen hd' (strs _ _ hD'))) heq k
  rw [getD_seen hd hm] at g
  have hm' := mem_of_getD_seen hd' (Properties.Challenger.CanonRead.equiv_str g.symm)
  have hk' : Reaches j' key k := (hR' k).2 ⟨_, hm', rfl⟩
  obtain ⟨d', hdig', hd'e⟩ := hD' _ hm'
  obtain rfl : d = d' := by injection hd'e
  obtain ⟨ce, t, hce, ht, hdt⟩ := definitionDigest_ok hdig
  obtain ⟨ce', t', hce', ht', hdt'⟩ := definitionDigest_ok hdig'
  have hsh2 := Properties.NetworkRecords.Bytes.hexOf_inj (hdt.symm.trans hdt')
  have htt : t.toUTF8 = t'.toUTF8 := Classical.byContradiction fun hne =>
    hC ⟨_, _, .inl ⟨k, ce, t, hk, hce, ht, rfl⟩, .inl ⟨k, ce', t', hk', hce', ht', rfl⟩, hne, hsh2⟩
  obtain rfl : t = t' := String.toByteArray_inj.mp (by simpa only [String.toUTF8_eq_toByteArray] using htt)
  exact ⟨hk', ce, ce', hce, hce', Properties.Challenger.CanonLine.canon_inj
    (sorted_of_wf (wf_canonicalEntry (wf_tab hj _) (wf_tab hj _) (wf_entry hj k) hce))
    (sorted_of_wf (wf_canonicalEntry (wf_tab hj' _) (wf_tab hj' _) (wf_entry hj' k) hce')) ht ht'⟩

end NciCheck.Properties.Record.SecurityProofs
