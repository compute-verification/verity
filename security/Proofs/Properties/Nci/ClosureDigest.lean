module

public import Properties.Nci.Record
public import Proofs.Properties.Nci.DefinitionDigest

/-!
# What a closure digest hashes

`closureDigest j key` walks the Definitions `key` reaches in the descriptor `j`, from a list of names still to visit
(`todo`) and the ids already digested (`seen`). Its loop keeps `Inv`: every id in `seen` or `todo` is reached, each id in
`seen` is there once with its `definitionDigest`, `key` is in one of the two, and every callee of a composite in `seen` is
in one of the two (`inv_skip`, `inv_step`). When the loop ends with nothing left to visit, `seen` holds exactly the
Definitions `key` reaches (`closureDigest_spec`), and the digest is the SHA-512 of `seen`'s canonical object, the second
of the byte strings `Hashed` names.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open NciCheck.DefinitionDigest (isComposite definitionDigest closureDigest)
open NciCheck.DefinitionDigest.SecurityProofs (isComposite_iff)

/-- A loop over a list keeps an invariant that each step keeps. -/
theorem forIn_inv {α β ε : Type} (l : List α) (f : α → β → Except ε (ForInStep β)) (I : β → Prop)
    (hf : ∀ a ∈ l, ∀ b r, I b → f a b = .ok r → I r.value) : ∀ b r, I b → forIn l b f = .ok r → I r := by
  induction l with
  | nil => intro b r hb h; simp only [List.forIn_nil, pure, Except.pure] at h; cases h; exact hb
  | cons a l ih =>
    intro b r hb h
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    rename_i s hs
    have hs' := hf a List.mem_cons_self b s hb hs
    cases s with
    | done b' => simp only [pure, Except.pure] at h; cases h; exact hs'
    | yield b' => exact ih (fun a ha => hf a (List.mem_cons_of_mem _ ha)) b' r hs' h

/-- The callee a node names, as `closureDigest` and `Reaches` read it. -/
def calleeOf (j n : Json) : Option Json :=
  (n.getObjValD "fn").getNat?.toOption.bind fun i => (tab j "callees")[i]?

/-- The closure loop's invariant over `seen` and `todo`. -/
structure Inv (j : Json) (key : String) (seen : Array (String × Json)) (todo : List String) : Prop where
  seen_reach : ∀ p ∈ seen.toList, Reaches j key p.1
  seen_digest : ∀ p ∈ seen.toList, ∃ d, definitionDigest (tab j "callees") (tab j "types") (entry j p.1) = .ok d ∧
    p.2 = .str d
  todo_reach : ∀ k ∈ todo, Reaches j key k
  closed : ∀ p ∈ seen.toList, isComposite (entry j p.1) = true → ∀ n ∈ nodesOf (entry j p.1),
    ∃ c, calleeOf j n = some (.str c) ∧ ((∃ q ∈ seen.toList, q.1 = c) ∨ c ∈ todo)
  key_in : (∃ q ∈ seen.toList, q.1 = key) ∨ key ∈ todo
  nodup : (seen.toList.map Prod.fst).Nodup

/-- The inner loop: from `t`, each node's callee pushed on the front. -/
theorem inner_spec (j : Json) (f : Json → List String → Except String (ForInStep (List String)))
    (hf : ∀ n t r, f n t = .ok r → ∃ c, calleeOf j n = some (.str c) ∧ r = .yield (c :: t)) :
    ∀ (l : List Json) (t r : List String), forIn l t f = .ok r →
      (∀ x ∈ t, x ∈ r) ∧ (∀ c ∈ r, c ∈ t ∨ ∃ n ∈ l, calleeOf j n = some (.str c)) ∧
        ∀ n ∈ l, ∃ c, calleeOf j n = some (.str c) ∧ c ∈ r := by
  intro l
  induction l with
  | nil =>
    intro t r h
    simp only [List.forIn_nil, pure, Except.pure] at h
    cases h
    exact ⟨fun _ hx => hx, fun c hc => .inl hc, fun _ hn => absurd hn List.not_mem_nil⟩
  | cons a l ih =>
    intro t r h
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    rename_i s hs
    obtain ⟨c, hc, rfl⟩ := hf a t s hs
    obtain ⟨h1, h2, h3⟩ := ih (c :: t) r h
    refine ⟨fun x hx => h1 x (List.mem_cons_of_mem _ hx), fun c' hc' => ?_, fun n hn => ?_⟩
    · rcases h2 c' hc' with h | ⟨n, hn, hn'⟩
      · rcases List.mem_cons.1 h with rfl | h
        · exact .inr ⟨a, List.mem_cons_self, hc⟩
        · exact .inl h
      · exact .inr ⟨n, List.mem_cons_of_mem _ hn, hn'⟩
    · rcases List.mem_cons.1 hn with rfl | hn
      · exact ⟨c, hc, h1 c List.mem_cons_self⟩
      · exact h3 n hn

/-- A name already in `seen` is dropped from `todo`. -/
theorem inv_skip {j : Json} {key k : String} {seen : Array (String × Json)} {rest : List String}
    (hI : Inv j key seen (k :: rest)) (hk : ∃ q ∈ seen.toList, q.1 = k) : Inv j key seen rest := by
  have mv : ∀ c, ((∃ q ∈ seen.toList, q.1 = c) ∨ c ∈ k :: rest) → (∃ q ∈ seen.toList, q.1 = c) ∨ c ∈ rest := by
    intro c h
    rcases h with h | h
    · exact .inl h
    · rcases List.mem_cons.1 h with rfl | h
      · exact .inl hk
      · exact .inr h
  exact ⟨hI.seen_reach, hI.seen_digest, fun x hx => hI.todo_reach x (List.mem_cons_of_mem _ hx),
    fun p hp hc n hn => by
      obtain ⟨c, hc1, hc2⟩ := hI.closed p hp hc n hn
      exact ⟨c, hc1, mv c hc2⟩,
    mv key hI.key_in, hI.nodup⟩

theorem reach_callee {j : Json} {key k c : String} {n : Json} (hk : Reaches j key k)
    (hc : isComposite (entry j k) = true) (hn : n ∈ nodesOf (entry j k)) (h : calleeOf j n = some (.str c)) :
    Reaches j key c :=
  .callee k c n hk ((isComposite_iff _).1 hc) hn h

/-- A new name is digested into `seen`, and a composite's callees join `todo`. -/
theorem inv_step {j : Json} {key k v : String} {seen : Array (String × Json)} {rest todo' : List String}
    (hI : Inv j key seen (k :: rest)) (hany : ¬ ∃ q ∈ seen.toList, q.1 = k)
    (hd : definitionDigest (tab j "callees") (tab j "types") (entry j k) = .ok v)
    (h1 : ∀ x ∈ rest, x ∈ todo')
    (h2 : ∀ c ∈ todo', c ∈ rest ∨
      (isComposite (entry j k) = true ∧ ∃ n ∈ nodesOf (entry j k), calleeOf j n = some (.str c)))
    (h3 : isComposite (entry j k) = true → ∀ n ∈ nodesOf (entry j k), ∃ c, calleeOf j n = some (.str c) ∧ c ∈ todo') :
    Inv j key (seen.push (k, .str v)) todo' := by
  have hk : Reaches j key k := hI.todo_reach k List.mem_cons_self
  have mem : ∀ q, q ∈ (seen.push (k, Json.str v)).toList ↔ q ∈ seen.toList ∨ q = (k, Json.str v) := by
    intro q; simp [Array.toList_push]
  have mv : ∀ c, ((∃ q ∈ seen.toList, q.1 = c) ∨ c ∈ k :: rest) →
      (∃ q ∈ (seen.push (k, Json.str v)).toList, q.1 = c) ∨ c ∈ todo' := by
    intro c h
    rcases h with ⟨q, hq, rfl⟩ | h
    · exact .inl ⟨q, (mem q).2 (.inl hq), rfl⟩
    · rcases List.mem_cons.1 h with rfl | h
      · exact .inl ⟨(c, Json.str v), (mem _).2 (.inr rfl), rfl⟩
      · exact .inr (h1 c h)
  refine ⟨fun p hp => ?_, fun p hp => ?_, fun c hc => ?_, fun p hp hc n hn => ?_, mv key hI.key_in, ?_⟩
  · rcases (mem p).1 hp with hp | rfl
    · exact hI.seen_reach p hp
    · exact hk
  · rcases (mem p).1 hp with hp | rfl
    · exact hI.seen_digest p hp
    · exact ⟨v, hd, rfl⟩
  · rcases h2 c hc with h | ⟨hcomp, n, hn, hn'⟩
    · exact hI.todo_reach c (List.mem_cons_of_mem _ h)
    · exact reach_callee hk hcomp hn hn'
  · rcases (mem p).1 hp with hp | rfl
    · obtain ⟨c, hc1, hc2⟩ := hI.closed p hp hc n hn
      exact ⟨c, hc1, mv c hc2⟩
    · obtain ⟨c, hc1, hc2⟩ := h3 hc n hn
      exact ⟨c, hc1, .inr hc2⟩
  · rw [Array.toList_push, List.map_append, List.nodup_append]
    refine ⟨hI.nodup, List.nodup_cons.2 ⟨List.not_mem_nil, List.nodup_nil⟩, ?_⟩
    intro a ha b hb
    simp only [List.map_cons, List.map_nil, List.mem_singleton] at hb
    subst hb
    intro e; subst e
    obtain ⟨q, hq, hq'⟩ := List.mem_map.1 ha
    exact hany ⟨q, hq, hq'⟩

/-- The loop ends with `Inv` and nothing left to visit, and the digest hashes `seen`'s canonical object. -/
theorem closureDigest_loop {j : Json} {key h : String} (H : closureDigest j key = .ok h) :
    ∃ (seen : Array (String × Json)), Inv j key seen [] ∧ ∃ s, Flock.canon (Json.mkObj seen.toList) = .ok s ∧
      h = Flock.hexOf (Flock.Sha512.hash s.toUTF8) := by
  unfold closureDigest at H
  simp only [bind, Except.bind] at H
  split at H
  · cases H
  rename_i r hr
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
  have hI : Inv j key r.1 r.2 :=
    forIn_inv _ _ (fun (b : Array (String × Json) × List String) => Inv j key b.1 b.2) ?hstep (#[], [key]) r ?hinit hr
  case hinit =>
    exact ⟨fun _ h => absurd h List.not_mem_nil, fun _ h => absurd h List.not_mem_nil,
      fun k hk => by rw [List.mem_singleton.1 hk]; exact .self, fun _ h => absurd h List.not_mem_nil,
      .inr List.mem_cons_self, List.nodup_nil⟩
  case hstep =>
    intro _ _ b r hb hf
    clear H hr
    obtain ⟨seen, todo⟩ := b
    dsimp only at hb hf
    have any_iff : ∀ k, (seen.any fun x => x.fst == k) = true ↔ ∃ q ∈ seen.toList, q.1 = k := by
      intro k; rw [← Array.any_toList, List.any_eq_true]; simp only [beq_iff_eq]
    rcases todo with _ | ⟨k, rest⟩
    · dsimp only at hf; simp only [pure, Except.pure] at hf; cases hf; exact hb
    · dsimp only at hf
      split at hf
      · rename_i hany; simp only [pure, Except.pure] at hf; cases hf; exact inv_skip hb ((any_iff k).1 hany)
      · rename_i hany
        have hany' : ¬ ∃ q ∈ seen.toList, q.1 = k := fun h => hany ((any_iff k).2 h)
        split at hf
        · split at hf
          · cases hf
          · rename_i heq; nomatch heq
        split at hf
        · cases hf
        rename_i v hv
        have hv' : definitionDigest (tab j "callees") (tab j "types") (entry j k) = .ok v := hv
        split at hf
        · rename_i hc
          split at hf
          · cases hf
          rename_i todo' hloop
          simp only [pure, Except.pure] at hf
          cases hf
          rw [← Array.forIn_toList] at hloop
          obtain ⟨i1, i2, i3⟩ := inner_spec j _ (fun n t r h => by
            split at h
            · rename_i c hc
              simp only [pure, Except.pure, Except.ok.injEq] at h
              exact ⟨c, hc, h.symm⟩
            · split at h
              · cases h
              · rename_i heq; nomatch heq) _ rest todo' hloop
          refine inv_step hb hany' hv' i1 (fun c hc' => ?_) (fun _ n hn => i3 n (Array.mem_def.1 hn))
          rcases i2 c hc' with h | ⟨n, hn, hn'⟩
          · exact .inl h
          · exact .inr ⟨hc, n, Array.mem_def.2 hn, hn'⟩
        · rename_i hc
          simp only [pure, Except.pure] at hf
          cases hf
          exact inv_step hb hany' hv' (fun x hx => hx) (fun c hc => .inl hc) (fun h => absurd h hc)
  rcases r with ⟨seen, todo⟩
  dsimp only at hI H
  split at H
  · rename_i he
    obtain rfl : todo = [] := List.isEmpty_iff.1 he
    split at H
    · cases H
    rename_i s hs
    simp only [pure, Except.pure, Except.ok.injEq] at H
    exact ⟨seen, hI, s, hs, H.symm⟩
  · split at H
    · cases H
    · rename_i heq; nomatch heq

/-- **What the closure digest hashes**: the Definitions `key` reaches, each once, with its digest. -/
theorem closureDigest_spec {j : Json} {key h : String} (H : closureDigest j key = .ok h) :
    ∃ (seen : Array (String × Json)) (s : String),
      (∀ k, Reaches j key k ↔ ∃ p ∈ seen.toList, p.1 = k) ∧
      (∀ p ∈ seen.toList, ∃ d, definitionDigest (tab j "callees") (tab j "types") (entry j p.1) = .ok d ∧
        p.2 = .str d) ∧
      (seen.toList.map Prod.fst).Nodup ∧
      Flock.canon (Json.mkObj seen.toList) = .ok s ∧ h = Flock.hexOf (Flock.Sha512.hash s.toUTF8) := by
  obtain ⟨seen, hI, s, hs, he⟩ := closureDigest_loop H
  refine ⟨seen, s, fun k => ⟨fun hk => ?_, fun ⟨p, hp, e⟩ => e ▸ hI.seen_reach p hp⟩, hI.seen_digest, hI.nodup, hs,
    he⟩
  induction hk with
  | self => exact hI.key_in.resolve_right List.not_mem_nil
  | callee k c n _ hform hn hcall ih =>
    obtain ⟨p, hp, rfl⟩ := ih
    obtain ⟨c', hc', hin⟩ := hI.closed p hp ((isComposite_iff _).2 hform) n hn
    have : c' = c := by
      have e : calleeOf j n = some (.str c) := hcall
      rw [hc'] at e
      injection e with e
      injection e
    subst this
    exact hin.resolve_right List.not_mem_nil

end NciCheck.Properties.Record.SecurityProofs
