import Flock.With.SimSetup

/-!
# What the setup's parse, load and draw return, in any hash

`Stmt.setupHiddenWith`'s hash calls depend on the parsed circuit's ports, the loaded public file's shared-row counts
and the drawn population (`SetupCount.setupHidden_cost`). Each is a function of the files alone, whatever the hash:

* `hmOutParse_ports`: an untyped circuit's ports are `HmRow.rowPorts` of its META line (`metaOf`).
* `loadPublic_counts`: a public file's shared-row counts are `HmRow.sharedRows` of its header (`headerOf`) at the
  circuit's ports (`sharedRows_ports`).
* `drawn_facts`: a draw keeps the counts, and its population is its JSON's (`Draw.UnitDraw.ofJson`).

So the values at any hash are the ones at SHA-512 (`parse_ports_eq`, `load_counts_eq`, `drawn_pop`), where the
verifier's own checks bound them.
-/

namespace RecDeploy.SetupValues

open Flock
open Lean (Json)

theorem except_bind_ok {α β ε : Type} {x : Except ε α} {f : α → Except ε β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem ite_throw_ok {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at h; cases h
  · rwa [ite_eq_right_of_eq_false _ _ (eq_false hc)] at h

theorem ite_throw_ok' {α : Type} {c : Prop} [Decidable c] {e : String} {x : Except String α} {v : α}
    (h : (if c then (throw e : Except String α) else x) = .ok v) : ¬c ∧ x = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at h; cases h
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hc)] at h; exact ⟨hc, h⟩

theorem ite_throw_okP {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : ¬ c ∧ g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · exact ⟨hc, by rwa [ite_eq_right hc] at h⟩

set_option maxHeartbeats 4000000 in
theorem parseRest_ports (tags : Tags) (tables : List (String × ByteArray)) (pre : Array (String × Net)) (sha : ByteArray)
    (lines : Array String) (name : String) (mj : Json) (cs : ByteArray) (c : Circuit)
    (h : HmRow.parseRest tags tables none pre sha lines name mj cs = .ok c) : HmRow.rowPorts mj tags.typed = .ok c.ports := by
  unfold HmRow.parseRest at h
  simp (config := { zeta := false }) only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Option.mapM_none] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s1, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h
  simp (config := { zeta := false }) only [jpT] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> first | (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩) | exact ⟨_, h⟩ | cases h
  simp (config := { zeta := false }) only [jpL] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s2, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets2 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ports, hrp, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpK] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨g, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨upv, -, h⟩ := hh)
  split at h
  rotate_left
  · cases h
  rename_i unitNet hunit
  ((try dsimp (config := { zeta := false }) only at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) netOf at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨ranges, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesIn, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesOut, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummy, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyDigests, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyOutputs, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outPorts, -, h⟩ := hh)
  ((try dsimp (config := { zeta := false }) only at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpR at h)
  obtain ⟨outRowPorts, h⟩ : ∃ r, jpR r = .ok c := by
    split at h
    · exact ⟨_, h⟩
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨r, -, h⟩ := except_bind_ok h
      exact ⟨r, (ite_throw_okP h).2⟩
  simp (config := { zeta := false }) only [jpR] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) perVu0 jpP at h
  obtain ⟨pv, h⟩ : ∃ pv, jpP () pv = .ok c := by
    split at h
    · obtain ⟨p', -, h⟩ := except_bind_ok h
      exact ⟨_, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jpP] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨shaNet, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨hmNet, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outNet, -, h⟩ := hh)
  ((try dsimp (config := { zeta := false }) only at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpO at h)
  obtain ⟨onp, h⟩ : ∃ x, jpO x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpO] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨lc, h⟩ : ∃ x, jpC x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpC] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpW at h
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
  simp (config := { zeta := false }) only [jpW] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) derived wiresT rowNet at h
  obtain ⟨-, h⟩ := ite_throw_okP h
  (have hh := except_bind_ok h; clear h; obtain ⟨ls, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leafId, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpF at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpF] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨kLog, -, h⟩ := hh)
  simp (config := { zeta := false }) only [pure_bind] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) c0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hc : c0 = c := Except.ok.inj h
  subst hc
  exact hrp

/-- The META object a circuit file's second line states, read from its bytes alone. -/
def metaOf (bytes : ByteArray) : Option Json :=
  match String.fromUTF8? bytes with
  | none => none
  | some text =>
    let lines0 := (text.splitOn "\n").toArray
    let lines := if lines0.back? == some "" then lines0.pop else lines0
    let metaLine := (lines.getD 1 "").trimAsciiEnd.toString
    (Json.parse (metaLine.drop 5).toString).toOption

set_option maxHeartbeats 4000000 in
theorem parseWith_ports (h : ByteArray → ByteArray) (tags : Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) (c : Circuit)
    (hp : HmRow.parseWith (M := V) (Hashes.ofSha512 h) tags bytes tables = .ok c) :
    ∃ mj, metaOf bytes = some mj ∧ HmRow.rowPorts mj tags.typed = .ok c.ports := by
  unfold HmRow.parseWith at hp
  simp (config := { zeta := false }) only [Hashes.ofSha512, pure_bind, monadLift_self] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at hp
  split at hp
  rotate_left
  · obtain ⟨_, hte, -⟩ := except_bind_ok hp; cases hte
  rename_i text hutf
  simp (config := { zeta := false }) only [jp0] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp1] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp2] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at hp
  split at hp
  rotate_left
  · obtain ⟨_, hte, -⟩ := except_bind_ok hp; cases hte
  rename_i mj hmj
  refine ⟨mj, ?_, ?_⟩
  · simp only [metaLine, lines, lines0] at hmj
    unfold metaOf
    rw [hutf]
    dsimp only
    rw [hmj]
    rfl
  simp (config := { zeta := false }) only [jp3] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨f1, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨f2, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp4] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at hp
  obtain ⟨classSha, hp⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at hp
    · replace hp := ite_throw_ok hp
      simp (config := { zeta := false }) only [jp6] at hp
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at hp
      replace hp := ite_throw_ok hp
      simp (config := { zeta := false }) only [jp7] at hp
      obtain ⟨cs, -, hp⟩ := except_bind_ok hp
      exact ⟨cs, hp⟩
    · exact ⟨_, hp⟩
  simp (config := { zeta := false }) only [jp5] at hp
  exact parseRest_ports _ _ _ _ _ _ _ _ _ hp

theorem hmOutParse_ports (h : ByteArray → ByteArray) (tags : Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) (c : Circuit) (ht : tags.typed = false)
    (hp : HmOut.parseWith (M := V) (Hashes.ofSha512 h) tags bytes tables = .ok c) :
    ∃ mj, metaOf bytes = some mj ∧ HmRow.rowPorts mj tags.typed = .ok c.ports := by
  unfold HmOut.parseWith at hp
  simp (config := { zeta := false }) only [ht, Bool.not_false, ↓reduceIte] at hp
  obtain ⟨c0, h0, hp⟩ := except_bind_ok hp
  obtain ⟨_, -, hp⟩ := except_bind_ok hp
  cases hp
  exact parseWith_ports h tags bytes tables c h0

theorem hmInLoad_row (h : ByteArray → ByteArray) (tags : Tags) (c : Circuit) (pi : HmIn.Ports) (f : ByteArray)
    (pop : Public) (hp : HmIn.loadPublicWith (M := V) (Hashes.ofSha512 h) tags c pi f = .ok pop) :
    HmRow.loadPublicWith (M := V) (Hashes.ofSha512 h) tags
      { c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] } f = .ok pop := by
  unfold HmIn.loadPublicWith at hp
  obtain ⟨p0, h0, hp⟩ := except_bind_ok hp
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hp
  repeat' split at hp
  all_goals first | (cases hp; done) | (cases hp; exact h0)

/-- The public file's header, read from its bytes alone. -/
def headerOf (f : ByteArray) : Option Json :=
  match findByte f 10 with
  | none => none
  | some nl => match String.fromUTF8? (f.extract 0 nl) with
    | none => none
    | some text => (Json.parse text).toOption.map Flock.norm

theorem loadPublic_counts (h : ByteArray → ByteArray) (tags : Tags) (c : Circuit) (f : ByteArray) (pop : Public)
    (hp : HmRow.loadPublicWith (M := V) (Hashes.ofSha512 h) tags c f = .ok pop) :
    ∃ hd, headerOf f = some hd ∧ HmRow.sharedRows tags c hd = .ok (pop.tables.map (·.counts)) := by
  unfold HmRow.loadPublicWith at hp
  simp (config := { zeta := false }) only [Hashes.ofSha512, monadLift_self] at hp
  cases hnl : findByte f 10 with
  | none => simp (config := { zeta := false }) only [hnl] at hp; cases hp
  | some nl =>
  simp (config := { zeta := false }) only [hnl] at hp
  cases htx : String.fromUTF8? (f.extract 0 nl) with
  | none =>
    simp (config := { zeta := false }) only [htx] at hp
    obtain ⟨_, hte, -⟩ := except_bind_ok hp; cases hte
  | some text =>
  simp (config := { zeta := false }) only [htx, pure_bind] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at hp
  simp (config := { zeta := false }) only [jpA] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at hp
  cases hj : Json.parse text with
  | error e =>
    simp (config := { zeta := false }) only [hj] at hp
    obtain ⟨_, hte, -⟩ := except_bind_ok hp; cases hte
  | ok hd0 =>
  simp (config := { zeta := false }) only [hj, pure_bind, jpB] at hp
  refine ⟨Flock.norm hd0, by simp only [headerOf, hnl, htx, hj, Except.toOption, Option.map_some], ?_⟩
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) pctx at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp1] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp2] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨n, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp3] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) np nOut at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨counts, hcounts, hp⟩ := hh)
  rw [hcounts]
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nBcs refsLen body jp4 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp4] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) bcs o outputs jpT at hp
  obtain ⟨tables, htab, hp⟩ : ∃ tables, tables.map (·.counts) = counts ∧ jpT tables = .ok pop := by
    cases counts with
    | none => exact ⟨none, rfl, by simpa only [pure_bind] using hp⟩
    | some cs =>
      simp (config := { zeta := false }) only at hp
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) refs jpS at hp
      replace hp := ite_throw_ok hp
      simp (config := { zeta := false }) only [jpS, pure_bind] at hp
      exact ⟨_, rfl, hp⟩
  simp (config := { zeta := false }) only [jpT] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨indices, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 at hp
  obtain ⟨u5, hp⟩ : ∃ u, jp5 u = .ok pop := by
    split at hp
    · (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
      (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
      exact ⟨(), ite_throw_ok hp⟩
    · exact ⟨(), hp⟩
  simp (config := { zeta := false }) only [jp5] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp6 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp6] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp7 at hp
  replace hp := ite_throw_ok hp
  simp (config := { zeta := false }) only [jp7] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  cases hp
  exact congrArg Except.ok htab.symm

theorem drawn_facts (h : ByteArray → ByteArray) (c : Circuit) (pop pub : Public) (d : Json)
    (hp : HmRow.drawnWith (M := V) (Hashes.ofSha512 h) c pop d = .ok pub) :
    ∃ D, Draw.UnitDraw.ofJson d = .ok D ∧ D.population = pop.n ∧
      (pub.population.getD (pub.n, pub.digests, pub.outputs)).1 = pop.n ∧
      pub.tables.map (·.counts) = pop.tables.map (·.counts) := by
  unfold HmRow.drawnWith at hp
  simp (config := { zeta := false }) only [Hashes.ofSha512, monadLift_self] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨D, hD, hp⟩ := hh)
  refine ⟨D, hD, ?_⟩
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) proved jpA at hp
  obtain ⟨hb, hp⟩ := ite_throw_okP hp
  have hpop : D.population = pop.n := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hb
    exact hb.1
  refine ⟨hpop, ?_⟩
  simp (config := { zeta := false }) only [jpA] at hp
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) np nOut pctx at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) indices header pubs jpB at hp
  obtain ⟨p1, hp⟩ : ∃ p, jpB () p = .ok pub := by
    split at hp
    · obtain ⟨p, -, hp⟩ := except_bind_ok hp
      exact ⟨_, hp⟩
    · exact ⟨_, hp⟩
  simp (config := { zeta := false }) only [jpB] at hp
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  (have hh := except_bind_ok hp; clear hp; obtain ⟨_, -, hp⟩ := hh)
  cases hp
  refine ⟨rfl, ?_⟩
  cases pop.tables <;> rfl


/-- `HmRow.sharedRows` reads the circuit only through its ports. -/
theorem sharedRows_ports {tags : Tags} {c c' : Circuit} (hd : Json) (h : c.ports = c'.ports) :
    HmRow.sharedRows tags c hd = HmRow.sharedRows tags c' hd := by
  unfold HmRow.sharedRows
  rw [h]

/-- **An untyped circuit's ports are the same at any two hashes.** -/
theorem parse_ports_eq (h h' : ByteArray → ByteArray) {tags : Tags} {cf : ByteArray}
    {tables tables' : List (String × ByteArray)} {c c' : Circuit} (ht : tags.typed = false)
    (hp : HmOut.parseWith (M := V) (Hashes.ofSha512 h) tags cf tables = .ok c)
    (hp' : HmOut.parseWith (M := V) (Hashes.ofSha512 h') tags cf tables' = .ok c') : c.ports = c'.ports := by
  obtain ⟨mj, hm, hr⟩ := hmOutParse_ports h tags cf tables c ht hp
  obtain ⟨mj', hm', hr'⟩ := hmOutParse_ports h' tags cf tables' c' ht hp'
  rw [hm] at hm'
  cases hm'
  rw [hr] at hr'
  exact Except.ok.inj hr'

/-- **A public file's shared-row counts are the same at any two hashes**, at circuits with the same ports. -/
theorem load_counts_eq (h h' : ByteArray → ByteArray) {tags : Tags} {c c' : Circuit} {pi : HmIn.Ports}
    {f : ByteArray} {pop pop' : Public}
    (hp : HmIn.loadPublicWith (M := V) (Hashes.ofSha512 h) tags c pi f = .ok pop)
    (hp' : HmRow.loadPublicWith (M := V) (Hashes.ofSha512 h') tags c' f = .ok pop') (hc : c.ports = c'.ports) :
    pop.tables.map (·.counts) = pop'.tables.map (·.counts) := by
  obtain ⟨hd, e, s⟩ := loadPublic_counts h tags _ f pop (hmInLoad_row h tags c pi f pop hp)
  obtain ⟨hd', e', s'⟩ := loadPublic_counts h' tags c' f pop' hp'
  rw [e] at e'
  cases e'
  have hs := sharedRows_ports (tags := tags) hd
    (show ({ c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] } : Circuit).ports
      = c'.ports from hc)
  rw [hs, s'] at s
  exact (Except.ok.inj s).symm

/-- Equal shared-row counts keep a bound on them. -/
theorem counts_le {pop pop' : Public} {B : Nat} (he : pop.tables.map (·.counts) = pop'.tables.map (·.counts))
    (hb : ∀ t, pop'.tables = some t → ∀ r ∈ t.counts, r ≤ B) : ∀ t, pop.tables = some t → ∀ r ∈ t.counts, r ≤ B := by
  intro t ht r hr
  cases ht' : pop'.tables with
  | none => rw [ht, ht'] at he; cases he
  | some t' =>
    rw [ht, ht'] at he
    have he' : t.counts = t'.counts := Option.some.inj he
    exact hb t' ht' r (he' ▸ hr)

/-- **A draw's population is its JSON's**, and it keeps the shared-row counts, at any hash. -/
theorem drawn_pop (h : ByteArray → ByteArray) {c : Circuit} {pi : HmIn.Ports} {pop pub : Public} {d : Json}
    (hp : HmIn.drawnWith (M := V) (Hashes.ofSha512 h) c pi pop d = .ok pub) :
    ∃ D, Draw.UnitDraw.ofJson d = .ok D ∧ (pub.population.getD (pub.n, pub.digests, pub.outputs)).1 = D.population ∧
      pub.tables.map (·.counts) = pop.tables.map (·.counts) := by
  obtain ⟨D, hD, hDp, hn, ht⟩ := drawn_facts h _ pop pub d hp
  exact ⟨D, hD, hn.trans hDp.symm, ht⟩

end RecDeploy.SetupValues
