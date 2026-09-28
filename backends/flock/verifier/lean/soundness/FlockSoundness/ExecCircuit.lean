import FlockSoundness.ExecParse

/-!
# What `HmRow.parse` accepted

`parse_facts`: a circuit file `HmRow.parse` accepted has only text nets (`Net.parse`) and lookup slots (`Lookup.build`),
ports of whole SHA-512 blocks, the unit, `sha512x3` and `hm96` nets and every range's net among its nets, and passed
`HmRow.check`. The walk lifts each level's join points into local definitions (`extract_lets`), so a branch point is one
goal: the branches meet at the join point.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Loops and maps in `Except` -/

/-- A `while` loop in `Except` keeps an invariant when each step keeps it and every continuing step lowers a measure. -/
theorem loop_inv {β : Type} (P : β → Prop) (μ : β → ℕ) {f : Unit → β → Except String (ForInStep β)}
    (hf : ∀ b, P b → ∀ s, f () b = .ok s →
      (∃ b', s = .done b' ∧ P b') ∨ ∃ b', s = .yield b' ∧ P b' ∧ μ b' < μ b) :
    ∀ n b v, μ b ≤ n → P b → forIn Lean.Loop.mk b f = .ok v → P v := by
  intro n
  induction n with
  | zero =>
    intro b v hμ hb h
    change Lean.Loop.forIn Lean.Loop.mk b f = .ok v at h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    rcases hf b hb s hs with ⟨b', rfl, hb'⟩ | ⟨b', rfl, -, hlt⟩
    · cases h; exact hb'
    · omega
  | succ n ih =>
    intro b v hμ hb h
    change Lean.Loop.forIn Lean.Loop.mk b f = .ok v at h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    rcases hf b hb s hs with ⟨b', rfl, hb'⟩ | ⟨b', rfl, hb', hlt⟩
    · cases h; exact hb'
    · exact ih b' v (by omega) hb' h

/-- A `for` loop in `Except` keeps an invariant its body keeps. -/
theorem forIn_except_inv {α β : Type} (P : β → Prop) {f : α → β → Except String (ForInStep β)} :
    ∀ (l : List α), (∀ a ∈ l, ∀ b, P b → ∀ s, f a b = .ok s → ∃ b', (s = .yield b' ∨ s = .done b') ∧ P b') →
      ∀ b v, P b → forIn l b f = .ok v → P v
  | [], _, b, v, hb, h => by rw [List.forIn_nil] at h; cases h; exact hb
  | a :: l, hf, b, v, hb, h => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    obtain ⟨b', hs', hb'⟩ := hf a List.mem_cons_self b hb s hs
    rcases hs' with rfl | rfl
    · exact forIn_except_inv P l (fun a ha => hf a (List.mem_cons_of_mem _ ha)) b' v hb' h
    · cases h; exact hb'

theorem list_mapM_ok {α β : Type} {f : α → Except String β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys → ∀ y ∈ ys, ∃ x ∈ l, f x = .ok y
  | [], ys, h => by
    rw [List.mapM_nil] at h
    cases h
    simp
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := except_bind_ok h
    obtain ⟨bs, hbs, h⟩ := except_bind_ok h
    cases h
    intro y hy
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨a, List.mem_cons_self, hb⟩
    · obtain ⟨x, hx, hfx⟩ := list_mapM_ok hbs y hy
      exact ⟨x, List.mem_cons_of_mem _ hx, hfx⟩

theorem array_mapM_ok {α β : Type} {f : α → Except String β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ∀ y ∈ ys.toList, ∃ x ∈ xs.toList, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hm : xs.toList.mapM f with e | zs
  · rw [hm] at h; cases h
  · rw [hm] at h
    cases h
    simpa using list_mapM_ok hm

theorem findIdx?_lt {α : Type} {xs : Array α} {p : α → Bool} {i : ℕ} (h : xs.findIdx? p = some i) : i < xs.size :=
  (Array.findIdx?_eq_some_iff_findIdx_eq.1 h).1

/-- Lift the top-level `have`s of `h` (the `do` block's join points and values) into local definitions. -/
syntax "lets" (ppSpace colGt ident)+ : tactic
macro_rules
  | `(tactic| lets $a) =>
    `(tactic| (try beta_reduce at $(Lean.mkIdent `h):ident); extract_lets (config := { underBinder := false, onlyGivenNames := true }) $a:ident at $(Lean.mkIdent `h):ident)
  | `(tactic| lets $a $b) =>
    `(tactic| (try beta_reduce at $(Lean.mkIdent `h):ident); extract_lets (config := { underBinder := false, onlyGivenNames := true }) $a:ident $b:ident at $(Lean.mkIdent `h):ident)
  | `(tactic| lets $a $b $c) =>
    `(tactic| (try beta_reduce at $(Lean.mkIdent `h):ident); extract_lets (config := { underBinder := false, onlyGivenNames := true }) $a:ident $b:ident $c:ident at $(Lean.mkIdent `h):ident)
  | `(tactic| lets $a $b $c $d) =>
    `(tactic| (try beta_reduce at $(Lean.mkIdent `h):ident); extract_lets (config := { underBinder := false, onlyGivenNames := true }) $a:ident $b:ident $c:ident $d:ident at $(Lean.mkIdent `h):ident)
  | `(tactic| lets $a $b $c $d $e) =>
    `(tactic| (try beta_reduce at $(Lean.mkIdent `h):ident); extract_lets (config := { underBinder := false, onlyGivenNames := true }) $a:ident $b:ident $c:ident $d:ident $e:ident at $(Lean.mkIdent `h):ident)
  | `(tactic| lets $a $b $c $d $e $f) =>
    `(tactic| (try beta_reduce at $(Lean.mkIdent `h):ident); extract_lets (config := { underBinder := false, onlyGivenNames := true }) $a:ident $b:ident $c:ident $d:ident $e:ident $f:ident at $(Lean.mkIdent `h):ident)

/-- Peel one bind off `h : (x >>= f) = .ok v`: `x`'s value and `h : f x = .ok v`. -/
syntax "bindok" (ppSpace colGt rcasesPat) (ppSpace colGt rcasesPat) : tactic
macro_rules
  | `(tactic| bindok $x $hx) =>
    `(tactic| (have hh := except_bind_ok $(Lean.mkIdent `h); clear $(Lean.mkIdent `h); obtain ⟨$x, $hx, $(Lean.mkIdent `h):ident⟩ := hh))

/-- Unfold the join point `j` in `h`, keeping its body's own join points. -/
macro "unjp " j:ident : tactic =>
  `(tactic| simp (config := { zeta := false }) only [$j:ident] at $(Lean.mkIdent `h):ident)

/-! ## What `HmRow.parse` accepted -/

/-- **What `HmRow.parse` accepted.** -/
structure ParseFacts (tags : Tags) (c : Circuit) : Prop where
  nets : ∀ x ∈ c.nets.toList,
    (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2
  ports : ∀ p ∈ c.ports.toList, 0 < p.chunks
  unitNet : c.unitNet < c.nets.size
  shaNet : c.shaNet < c.nets.size
  hmNet : c.hmNet < c.nets.size
  kinds : ∀ r ∈ c.ranges.toList, ∀ i, r.kind = .net i → i < c.nets.size
  check : ∃ ls, HmRow.check c tags ls = .ok ()

set_option maxHeartbeats 4000000 in
theorem parse_facts {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) : ParseFacts tags c := by
  unfold HmRow.parse at h
  lets sha jp0
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  bindok text hte
  cases hte
  unjp jp0
  lets lines0 lines head jp1
  replace h := ite_throw_ok h
  unjp jp1
  lets name metaLine jp2
  replace h := ite_throw_ok h
  unjp jp2
  lets jp3
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  bindok mj hte
  cases hte
  unjp jp3
  bindok f1 hf1
  bindok f2 hf2
  lets jp4
  replace h := ite_throw_ok h
  unjp jp4
  lets jp5 jp6
  obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at h
    · replace h := ite_throw_ok h
      unjp jp6
      lets part rule jp7
      replace h := ite_throw_ok h
      unjp jp7
      obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
    · obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
  unjp jp5
  lets nets0 at0
  bindok s1 hwhile
  lets nets1
  bindok u1 -
  lets jpL
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  unjp jpL
  bindok s2 hlook
  lets nets2
  bindok _ -
  bindok _ -
  lets jpS
  replace h := ite_throw_ok h
  unjp jpS
  bindok _ -
  bindok ports -
  bindok _ hpcheck
  bindok listed -
  lets jpR
  replace h := ite_throw_ok h
  unjp jpR
  bindok _ -
  bindok _ -
  lets jpK
  replace h := ite_throw_ok h
  unjp jpK
  bindok g -
  bindok upv -
  split at h
  swap
  · cases h
  rename_i unitNet hunit
  lets netOf
  bindok _ -
  bindok ranges hranges
  bindok leavesIn -
  bindok _ -
  bindok leavesOut -
  bindok dummy -
  bindok _ -
  bindok dummyDigests -
  bindok _ -
  bindok dummyOutputs -
  bindok _ -
  bindok outPorts -
  lets perVu0 jpP
  obtain ⟨pv, h⟩ : ∃ pv, jpP () pv = .ok c := by
    split at h
    · obtain ⟨p', -, h⟩ := except_bind_ok h
      exact ⟨_, h⟩
    · exact ⟨_, h⟩
  unjp jpP
  bindok shaNet hsha
  bindok hmNet hhm
  bindok outNet -
  lets jpO
  obtain ⟨onp, h⟩ : ∃ x, jpO x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  unjp jpO
  lets jpC
  obtain ⟨lc, h⟩ : ∃ x, jpC x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  unjp jpC
  lets jpW
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
  unjp jpW
  lets derived rowNet jpD
  replace h := ite_throw_ok h
  unjp jpD
  bindok ls -
  bindok leafId -
  lets jpF
  replace h := ite_throw_ok h
  unjp jpF
  bindok kLog -
  lets c0
  bindok w hcheck
  have hc : c0 = c := Except.ok.inj h
  subst hc
  have hnetOf : ∀ nm i, netOf nm = .ok i → i < nets2.size := by
    intro nm i hi
    simp only [netOf] at hi
    split at hi
    · rename_i i' hi'; cases hi; exact findIdx?_lt hi'
    · cases hi
  -- the text nets (the `while` loop over the `NET` blocks)
  have htext : ∀ x ∈ nets1.toList, ∃ lines, Net.parse lines = .ok x.2 := by
    refine loop_inv (fun b : Array (String × Net) × ℕ => ∀ x ∈ b.1.toList, ∃ lines, Net.parse lines = .ok x.2)
      (fun b => lines.size - b.2) ?_ _ (nets0, at0) s1 le_rfl (by simp [nets0]) hwhile
    intro b hb s h
    lets nets at_
    split at h
    · rename_i hlt
      lets hh jp
      replace h := ite_throw_ok h
      unjp jp
      split at h
      · rename_i n hn
        lets jp'
        obtain ⟨hnot, h⟩ := ite_throw_ok' h
        unjp jp'
        lets jp''
        split at h
        · rename_i v hv
          bindok net hnet
          cases hnet
          unjp jp''
          lets nets' at'
          have hs := Except.ok.inj h
          subst hs
          refine Or.inr ⟨_, rfl, ?_, ?_⟩
          · intro x hx
            simp only [nets', Array.toList_push, List.mem_append, List.mem_singleton] at hx
            rcases hx with hx | rfl
            · exact hb x hx
            · exact ⟨_, hv⟩
          · simp only [at', at_]; omega
        · bindok net hnet
          cases hnet
      · bindok w hw
        cases hw
    · have hs := Except.ok.inj h
      subst hs
      exact Or.inl ⟨_, rfl, hb⟩
  -- the lookup slots
  have hnets : ∀ x ∈ nets2.toList,
      (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2 := by
    rw [← Array.forIn_toList] at hlook
    refine forIn_except_inv (fun nets : Array (String × Net) => ∀ x ∈ nets.toList,
      (∃ lines, Net.parse lines = .ok x.2) ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2)
      _ ?_ nets1 s2 (fun x hx => Or.inl (htext x hx)) hlook
    intro a _ b hb s h
    obtain ⟨lname, spec⟩ := a
    lets nets
    split at h
    · lets size
      bindok _ -
      bindok _ -
      bindok _ -
      lets jpA
      replace h := ite_throw_ok h
      unjp jpA
      bindok n -
      bindok lo -
      lets jpB
      replace h := ite_throw_ok h
      unjp jpB
      split at h
      · lets jpC
        replace h := ite_throw_ok h
        unjp jpC
        bindok net hbuild
        lets nets'
        have hs := Except.ok.inj h
        subst hs
        refine ⟨_, Or.inl rfl, ?_⟩
        intro x hx
        simp only [nets', Array.toList_push, List.mem_append, List.mem_singleton] at hx
        rcases hx with hx | rfl
        · exact hb x hx
        · exact Or.inr ⟨_, _, _, _, _, hbuild⟩
      · bindok w hw
        cases hw
    · bindok w hw
      cases hw
  -- the ports
  have hports : ∀ p ∈ ports.toList, 0 < p.chunks := by
    intro p hp
    have := forIn_array_yield_ok (fun a s hs => by
      split at hs
      · obtain ⟨_, hx, -⟩ := except_bind_ok hs; cases hx
      · cases hs; rfl) hpcheck p hp
    split at this
    · obtain ⟨_, hx, -⟩ := except_bind_ok this; cases hx
    · rename_i hbad
      simp only [Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, not_or, not_not] at hbad
      omega
  -- the ranges' nets
  have hkinds : ∀ r ∈ ranges.toList, ∀ i, r.kind = .net i → i < nets2.size := by
    intro r hr i hi
    obtain ⟨x, -, h⟩ := array_mapM_ok hranges r hr
    bindok _ -
    lets jpK
    split at h
    · bindok kind hk
      cases hk
      unjp jpK
      lets packed
      bindok _ -
      bindok _ -
      bindok _ -
      have hs := Except.ok.inj h
      subst hs
      cases hi
    · bindok nm -
      bindok i' hi'
      bindok kind hk
      cases hk
      unjp jpK
      lets packed
      bindok _ -
      bindok _ -
      bindok _ -
      have hs := Except.ok.inj h
      subst hs
      cases hi
      exact hnetOf _ _ hi'
    · bindok kind hk
      cases hk
  exact { nets := hnets, ports := hports, unitNet := findIdx?_lt hunit, shaNet := hnetOf _ _ hsha,
          hmNet := hnetOf _ _ hhm, kinds := hkinds, check := ⟨ls, by cases w; exact hcheck⟩ }

end FlockSoundness
