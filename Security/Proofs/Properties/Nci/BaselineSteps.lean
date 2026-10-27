import Properties.Nci.Baseline

/-!
# θ's provenance check, step by step

What each step of `NciCheck.Baseline.evaluate` establishes when it returns: the trusted input's reading (`trustedOf_ok`:
the log in R6's chain, each record's digest, the run's record and the parameters), the provenance's (`provenanceOf_ok`),
one θ entry's checks (`entryStep_ok`), and the loop over θ (`provenance_ok`). `Proofs.Properties.Nci.Baseline` joins them
with the policy check's θ fold into `BaselineProvenance`.

The check compares fields with strings through `strOf?`, never with `Json`'s `==`, which is `partial`: what it compares
is then what these lemmas read. This module imports no Mathlib.
-/

namespace NciCheck.Properties.SecurityProofs.BaselineSteps

open Lean (Json)
open Flock.Program (Ty)
open NciCheck.Baseline

/-- `mapM` in `Except`, entry by entry. -/
theorem list_mapM_ok_get {α β ε : Type} [Inhabited α] [Inhabited β] (f : α → Except ε β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys → ys.length = l.length ∧ ∀ n < l.length, f l[n]! = .ok ys[n]!
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; simp
  | x :: l, ys, h => by
    rw [List.mapM_cons] at h
    rcases hx : f x with e | y <;> simp only [hx, bind, Except.bind] at h
    · cases h
    rcases hl : l.mapM f with e | cs <;> simp only [hl, pure, Except.pure] at h
    · cases h
    simp only [Except.ok.injEq] at h
    subst h
    obtain ⟨hlen, hget⟩ := list_mapM_ok_get f l cs hl
    refine ⟨by simp [hlen], fun n hn => ?_⟩
    cases n with
    | zero => simpa using hx
    | succ n =>
      have := hget n (by simp at hn; omega)
      simpa using this

theorem array_mapM_ok_get {α β ε : Type} [Inhabited α] [Inhabited β] (f : α → Except ε β) (xs : Array α)
    (ys : Array β) (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ n < xs.size, f xs[n]! = .ok ys[n]! := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hl : xs.toList.mapM f with e | l <;> simp only [hl, Functor.map, Except.map] at h
  · cases h
  simp only [Except.ok.injEq] at h
  subst h
  obtain ⟨hlen, hget⟩ := list_mapM_ok_get f _ l hl
  refine ⟨by simp [hlen], fun n hn => ?_⟩
  have := hget n (by simpa using hn)
  have e1 : xs.toList[n]! = xs[n]! := by
    rw [getElem!_pos xs.toList n (by simpa using hn), getElem!_pos xs n hn]; simp
  have e2 : l[n]! = l.toArray[n]! := by
    rw [getElem!_pos l n (by rw [hlen]; simpa using hn), getElem!_pos l.toArray n (by simp [hlen]; simpa using hn)]
    simp
  rw [← e1, ← e2]; exact this

theorem strOf?_eq_some (j : Json) (s : String) : strOf? j = some s ↔ j = .str s := by
  cases j <;> simp [strOf?]

theorem prevIs_none (prev : Json) (h : prevIs prev none = true) : prev = .null := by
  cases prev <;> simp_all [prevIs]

theorem prevIs_some (prev : Json) (d : String) (h : prevIs prev (some d) = true) : strOf? prev = some d := by
  simpa [prevIs] using h

/-- `forIn` over a list in `Except`, with an invariant on the prefix it has run through. -/
theorem forIn_list_prefix {α β ε : Type} (Q : List α → β → Prop) (f : α → β → Except ε (ForInStep β)) :
    ∀ (l pre : List α) (b r : β),
      (∀ pre' x b s, Q pre' b → f x b = .ok s → ∃ b', s = .yield b' ∧ Q (pre' ++ [x]) b') →
      Q pre b → forIn l b f = .ok r → Q (pre ++ l) r
  | [], pre, b, r, _, hb, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    simpa [← h] using hb
  | x :: l, pre, b, r, hf, hb, h => by
    rw [List.forIn_cons] at h
    rcases hx : f x b with e | s
    · simp [hx, bind, Except.bind] at h
    · obtain ⟨b', rfl, hb'⟩ := hf pre x b _ hb hx
      simp only [hx, bind, Except.bind] at h
      have := forIn_list_prefix Q f l (pre ++ [x]) b' r hf hb' h
      simpa using this

/-- The digests the loop keeps, for the records it has read. -/
def LoopInv (pre : List Json) (ds : Array String) : Prop :=
  ds.size = pre.length ∧ (∀ n < pre.length, digest pre[n]! = .ok ds[n]!) ∧
    (∀ n < pre.length, (n = 0 → getD pre[n]! "previous" = .null) ∧
      ∀ m, n = m + 1 → strAt pre[n]! "previous" = some ds[m]!)

theorem list_getElem!_append_lt {α : Type} [Inhabited α] (l : List α) (x : α) (n : Nat) (h : n < l.length) :
    (l ++ [x])[n]! = l[n]! := by
  rw [getElem!_pos (l ++ [x]) n (by simp; omega), getElem!_pos l n h, List.getElem_append_left h]

theorem list_getElem!_append_eq {α : Type} [Inhabited α] (l : List α) (x : α) : (l ++ [x])[l.length]! = x := by
  rw [getElem!_pos (l ++ [x]) l.length (by simp)]; simp

theorem array_getElem!_push_lt {α : Type} [Inhabited α] (b : Array α) (x : α) (n : Nat) (h : n < b.size) :
    (b.push x)[n]! = b[n]! := by
  rw [getElem!_pos (b.push x) n (by simp; omega), getElem!_pos b n h, Array.getElem_push_lt h]

theorem array_getElem!_push_eq {α : Type} [Inhabited α] (b : Array α) (x : α) : (b.push x)[b.size]! = x := by
  rw [getElem!_pos (b.push x) b.size (by simp)]; simp

theorem toList_getElem! {α : Type} [Inhabited α] (xs : Array α) (n : Nat) : xs.toList[n]! = xs[n]! := by
  by_cases h : n < xs.size
  · rw [getElem!_pos xs.toList n (by simpa using h), getElem!_pos xs n h]; simp
  · rw [getElem!_neg xs.toList n (by simpa using h), getElem!_neg xs n h]

theorem loopInv_step (pre : List Json) (x : Json) (b : Array String) (d : String) (hQ : LoopInv pre b)
    (hp : prevIs (getD x "previous") b.back? = true) (hd : digest x = .ok d) : LoopInv (pre ++ [x]) (b.push d) := by
  obtain ⟨hs, hdg, hpv⟩ := hQ
  refine ⟨by simp [hs], fun n hn => ?_, fun n hn => ?_⟩
  · simp only [List.length_append, List.length_singleton] at hn
    rcases Nat.lt_succ_iff_lt_or_eq.1 hn with hn | rfl
    · rw [list_getElem!_append_lt _ _ _ hn, array_getElem!_push_lt _ _ _ (by omega)]; exact hdg n hn
    · rw [list_getElem!_append_eq, ← hs, array_getElem!_push_eq]; exact hd
  · simp only [List.length_append, List.length_singleton] at hn
    rcases Nat.lt_succ_iff_lt_or_eq.1 hn with hn | rfl
    · rw [list_getElem!_append_lt _ _ _ hn]
      refine ⟨(hpv n hn).1, fun m hm => ?_⟩
      rw [array_getElem!_push_lt _ _ _ (by omega)]; exact (hpv n hn).2 m hm
    · rw [list_getElem!_append_eq]
      refine ⟨fun h0 => ?_, fun m hm => ?_⟩
      · have : b.back? = none := by
          rw [Array.back?_eq_getElem?]; simp [hs, h0]
        rw [this] at hp; exact prevIs_none _ hp
      · have hb : b.back? = some b[m]! := by
          rw [Array.back?_eq_getElem?, getElem!_pos b m (by omega)]; simp [hs, hm]
        rw [array_getElem!_push_lt _ _ _ (by omega)]
        rw [hb] at hp; exact prevIs_some _ _ hp

theorem paramOf_ok (p : Json) (q : String × Ty) (h : paramOf p = .ok q) :
    ∃ b, p = .arr b ∧ b.size = 2 ∧ b[0]! = .str q.1 ∧ Ty.ofJson b[1]! = .ok q.2 := by
  unfold paramOf at h
  split at h
  · rename_i b
    simp only [bind, Except.bind] at h
    split at h
    · simp [throw, throwThe, MonadExceptOf.throw] at h
    · rename_i hsz
      split at h
      · rename_i k hk
        split at h
        · rename_i ty hty
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          exact ⟨b, rfl, by simpa using hsz, hk, hty⟩
        · cases h
      · cases h
  · cases h

theorem trustedOf_ok (tr : Json) (T : Trusted) (h : trustedOf tr = .ok T) :
    getD tr "log" = .arr T.log ∧ Chained T.log ∧ T.digests.size = T.log.size ∧
      (∀ n < T.log.size, digest T.log[n]! = .ok T.digests[n]!) ∧
      (∃ r, strAt tr "run" = some r ∧ T.digests.findIdx? (· == r) = some T.run) ∧ ParamsOf tr T.params := by
  unfold trustedOf at h
  by_cases hk : (keys? tr != some TRUSTED_KEYS) = true
  · simp only [hk, ite_true, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h; cases h
  simp only [hk, Bool.false_eq_true, ite_false] at h
  split at h
  · rename_i log hlog
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    rename_i ds hds
    rw [← Array.forIn_toList] at hds
    have inv := forIn_list_prefix LoopInv _ log.toList [] #[] ds ?_ ?_ hds
    · simp only [List.nil_append] at inv
      obtain ⟨hs, hdg, hpv⟩ := inv
      simp only [Array.length_toList] at hs hdg hpv
      split at h
      · rename_i run hrun
        split at h
        · rename_i r hr
          split at h
          · rename_i ps hps
            split at h
            · cases h
            rename_i params hparams
            split at h
            · simp [throw, throwThe, MonadExceptOf.throw] at h
            · simp only [pure, Except.pure, Except.ok.injEq] at h
              subst h
              dsimp only
              refine ⟨hlog, fun n hn => ?_, hs, fun n hn => ?_, ⟨run, by simp [strAt, hrun, strOf?], hr⟩, ?_⟩
              · refine ⟨⟨_, by rw [← toList_getElem!]; exact hdg n hn⟩, fun h0 => ?_, fun m d hm hd => ?_⟩
                · rw [← toList_getElem!]; exact (hpv n hn).1 h0
                · have := hdg m (by omega)
                  rw [toList_getElem!, hd] at this
                  cases this
                  rw [← toList_getElem!]; exact (hpv n hn).2 m hm
              · rw [← toList_getElem!]; exact hdg n hn
              · obtain ⟨hsz, hget⟩ := array_mapM_ok_get paramOf ps params hparams
                exact ⟨ps, hps, hsz.symm, fun n hn => paramOf_ok _ _ (hget n hn)⟩
          · cases h
        · cases h
      · cases h
    · intro pre x b s hQ hf
      have hobj : ∃ kvs, x = Json.obj kvs := by
        cases x <;> first | exact ⟨_, rfl⟩ | simp [throw, throwThe, MonadExceptOf.throw] at hf
      obtain ⟨kvs, rfl⟩ := hobj
      simp only [Bool.not_true, Bool.false_eq_true, ite_false] at hf
      split at hf
      · simp [throw, throwThe, MonadExceptOf.throw] at hf
      · rename_i hp
        split at hf
        · rename_i d hd
          simp only [pure, Except.pure, Except.ok.injEq] at hf
          subst hf
          exact ⟨_, rfl, loopInv_step pre _ b d hQ (by simpa using hp) hd⟩
        · simp [throw, throwThe, MonadExceptOf.throw] at hf
    · simp [LoopInv]
  · cases h


theorem provOf_ok (p : Json) (q : Prov) (h : provOf p = some q) :
    strAt p "kind" = some "registered" ∧ strAt p "registration" = some q.registration ∧
      strAt p "value" = some q.value ∧ strAt p "root" = some q.root := by
  unfold provOf at h
  rcases hk : strAt p "kind" with _ | kind <;> simp only [hk, Option.bind_none, Option.bind_some, bind] at h
  · cases h
  split at h
  · rcases hg : (guard ((keys? p == some REGISTERED_KEYS) = true) : Option Unit) with _ | ⟨⟩ <;>
      simp only [hg, Option.bind_none, Option.bind_some] at h
    · cases h
    rcases h1 : strAt p "registration" with _ | a <;> simp only [h1, Option.bind_none, Option.bind_some] at h
    · cases h
    rcases h2 : strAt p "value" with _ | b <;> simp only [h2, Option.bind_none, Option.bind_some] at h
    · cases h
    rcases h3 : strAt p "root" with _ | c <;> simp only [h3, Option.bind_none, Option.bind_some, pure, Option.some.injEq] at h
    · cases h
    subst h
    exact ⟨rfl, rfl, rfl, rfl⟩
  · cases h

theorem provenanceOf_ok (j : Json) (n : Nat) (ps : Array Prov) (h : provenanceOf j n = .ok ps) :
    ∃ prov, getD j "provenance" = .arr prov ∧ prov.size = n ∧ ps.size = n ∧
      ∀ t < n, strAt prov[t]! "kind" = some "registered" ∧ strAt prov[t]! "registration" = some ps[t]!.registration ∧
        strAt prov[t]! "value" = some ps[t]!.value ∧ strAt prov[t]! "root" = some ps[t]!.root := by
  unfold provenanceOf at h
  split at h
  · rename_i prov hprov
    split at h
    · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    · rename_i hsz
      have hsz : prov.size = n := by simpa using hsz
      obtain ⟨hs, hget⟩ := array_mapM_ok_get _ prov ps h
      refine ⟨prov, hprov, hsz, by omega, fun t ht => ?_⟩
      have := hget t (by omega)
      split at this
      · rename_i q hq
        simp only [pure, Except.pure, Except.ok.injEq] at this
        subst this
        exact provOf_ok _ _ hq
      · cases this
  · cases h

theorem entryStep_ok (T : Trusted) (program : String) (own : Array (Option Nat)) (counts : Array Nat) (k width : Nat)
    (p : Prov) (h : entryStep T program own counts k width p = .ok ()) :
    T.digests.findIdx? (· == p.registration) = some 0 ∧
      (∃ e, lastWith T.log[0]! "registered" "value" p.value = some e ∧ strAt e "root" = some p.root) ∧
      strAt T.log[T.run]! "program" = some program ∧
      ∃ i, own[k]? = some (some i) ∧
        ∃ e', lastWith T.log[T.run]! "registered" "value" T.params[i]!.1 = some e' ∧ strAt e' "root" = some p.root := by
  unfold entryStep at h
  rcases hd : T.digests.findIdx? (· == p.registration) with _ | d <;> simp only [hd] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  by_cases hd0 : (d != 0) = true
  · simp [hd0, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hd0, Bool.false_eq_true, ite_false] at h
  have hd0 : d = 0 := by simpa using hd0
  subst hd0
  rcases he : lastWith T.log[0]! "registered" "value" p.value with _ | e <;> simp only [he] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  by_cases hr : (strAt e "root" != some p.root) = true
  · simp [hr, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hr, Bool.false_eq_true, ite_false] at h
  by_cases hprog : (strAt T.log[T.run]! "program" != some program) = true
  · simp [hprog, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hprog, Bool.false_eq_true, ite_false] at h
  rcases hi : own[k]? with _ | _ | i <;> simp only [hi] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  rcases he' : lastWith T.log[T.run]! "registered" "value" T.params[i]!.1 with _ | e' <;> simp only [he'] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  by_cases hr' : (strAt e' "root" != some p.root) = true
  · simp [hr', throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  exact ⟨by first | exact hd | rfl, ⟨e, by first | exact he | rfl, by simpa using hr⟩, by simpa using hprog, i,
    by first | exact hi | rfl, e', by first | exact he' | rfl, by simpa using hr'⟩

theorem forIn_each {α β ε : Type} (f : α → β → Except ε (ForInStep β))
    (hy : ∀ x b s, f x b = .ok s → ∃ b', s = .yield b') :
    ∀ (l : List α) (b r : β), forIn l b f = .ok r → ∀ x ∈ l, ∃ b' s, f x b' = .ok s
  | [], _, _, _, x, hx => by simp at hx
  | y :: l, b, r, h, x, hx => by
    rw [List.forIn_cons] at h
    rcases hfy : f y b with e | s
    · simp [hfy, bind, Except.bind] at h
    obtain ⟨b', rfl⟩ := hy y b _ hfy
    simp only [hfy, bind, Except.bind] at h
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨b, _, hfy⟩
    · exact forIn_each f hy l b' r h x hx

theorem provenance_ok (j tr : Json) (h : Baseline.provenance j tr (fun _ k => k) = .ok ()) :
    ∃ i P ps T, parse j = .ok i ∧ Flock.Program.decode i.program.toUTF8 = .ok P ∧ provenanceOf j i.theta.size = .ok ps ∧
      trustedOf tr = .ok T ∧ ∀ t < i.theta.size, ∃ n counts width,
        entryStep T (Flock.hexOf P.sha512) ((ownersOf P T.params).getD (Array.replicate n none)) counts
          ((natOf? i.theta[t]!).getD 0) width ps[t]! = .ok () := by
  unfold Baseline.provenance at h
  rcases h1 : parse j with e | i <;> simp only [h1, bind, Except.bind] at h
  · cases h
  rcases h2 : Flock.Program.decode i.program.toUTF8 with e | P <;> simp only [h2] at h
  · cases h
  simp only [pure, Except.pure] at h
  rcases h3 : provenanceOf j i.theta.size with e | ps <;> simp only [h3] at h
  · cases h
  rcases h4 : trustedOf tr with e | T <;> simp only [h4] at h
  · cases h
  refine ⟨i, P, ps, T, by first | exact h1 | rfl, by first | exact h2 | rfl, by first | exact h3 | rfl,
    by first | exact h4 | rfl, ?_⟩
  simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  split at h
  · cases h
  rename_i r hr
  intro t ht
  have := forIn_each _ ?_ _ _ _ hr t (by simp; omega)
  · obtain ⟨⟨⟩, s, hs⟩ := this
    split at hs
    · cases hs
    · rename_i v hv; exact ⟨_, _, _, hv⟩
  · intro x b s hs
    split at hs
    · cases hs
    · simp only [Except.ok.injEq] at hs; exact ⟨_, hs.symm⟩

theorem evaluate_ok (j tr out : Json) (h : Baseline.evaluate j tr = .ok out) :
    Desc.evaluate j = .ok out ∧ Baseline.provenance j tr (fun _ k => k) = .ok () := by
  unfold Baseline.evaluate at h
  rcases h1 : Desc.evaluate j with e | o <;> simp only [h1, bind, Except.bind] at h
  · cases h
  rcases h2 : Baseline.provenance j tr (fun _ k => k) with e | ⟨⟩ <;> simp only [h2, pure, Except.pure] at h
  · cases h
  cases h
  exact ⟨rfl, rfl⟩

theorem findIdx_eq (xs : Array String) (r : String) (i : Nat) (h : xs.findIdx? (· == r) = some i) :
    i < xs.size ∧ xs[i]! = r := by
  obtain ⟨hi, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 h
  refine ⟨hi, ?_⟩
  rw [getElem!_pos xs i hi]
  simpa using hp

/-- The owners `provenance` reads: `ownersOf`'s, or no owner for any node where the parameters don't lower. -/
theorem owners_some (o : Option (Array (Option Nat))) (n k p : Nat)
    (h : (o.getD (Array.replicate n none))[k]? = some (some p)) : ∃ own, o = some own ∧ own[k]? = some (some p) := by
  cases o with
  | none =>
    simp only [Option.getD_none, Array.getElem?_replicate] at h
    split at h <;> simp at h
  | some own => exact ⟨own, rfl, by simpa using h⟩

/-- `BaselineProvenance`'s conclusion, given that every θ entry names a node (which the policy check's θ fold gives,
`Proofs.Properties.Nci.Baseline`). -/
theorem provenanceHeld_of (j tr out : Json) (h : Baseline.evaluate j tr = .ok out)
    (hθ : ∀ i, parse j = .ok i → ∀ t < i.theta.size, ∃ k, natOf? i.theta[t]! = some k) : ProvenanceHeld j tr := by
  obtain ⟨-, hprov⟩ := evaluate_ok j tr out h
  obtain ⟨i, P, ps, T, h1, h2, h3, h4, hsteps⟩ := provenance_ok j tr hprov
  obtain ⟨hlog, hchain, hsz, hdg, ⟨r, hrun, hfind⟩, hparams⟩ := trustedOf_ok tr T h4
  obtain ⟨prov, hprovj, hpsz, -, hfields⟩ := provenanceOf_ok j _ ps h3
  obtain ⟨hrlt, hr⟩ := findIdx_eq _ _ _ hfind
  refine ⟨i, P, T.log, T.log[T.run]!, T.params, h1, h2,
    ⟨hlog, hchain, ⟨T.run, by omega, rfl, r, by rw [← hr]; exact hdg _ (by omega), hrun⟩, hparams⟩, fun t ht => ?_⟩
  obtain ⟨n, counts, width, hstep⟩ := hsteps t ht
  obtain ⟨h0, ⟨e, he, her⟩, hprog, p, hp, e', he', her'⟩ := entryStep_ok _ _ _ _ _ _ _ hstep
  obtain ⟨k, hk⟩ := hθ i h1 t ht
  rw [hk, Option.getD_some] at hp
  obtain ⟨own, hown, hp⟩ := owners_some _ _ _ _ hp
  obtain ⟨h0lt, h0r⟩ := findIdx_eq _ _ _ h0
  obtain ⟨fk, freg, fval, froot⟩ := hfields t ht
  exact ⟨prov, ps[t]!.registration, ps[t]!.value, ps[t]!.root, hprovj, hpsz, fk, freg, fval, froot, by omega,
    by rw [← h0r]; exact hdg 0 (by omega), ⟨e, he, her⟩, k, p, own, hk, hown, hp, hprog, e', he', her'⟩

end NciCheck.Properties.SecurityProofs.BaselineSteps
