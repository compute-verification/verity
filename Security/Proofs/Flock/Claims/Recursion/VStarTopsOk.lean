import Proofs.Flock.Claims.Recursion.VStarDense
import Proofs.Flock.Claims.Recursion.VStarCapL
import Proofs.Flock.Claims.Recursion.SlotsCalls
import Proofs.Flock.Claims.Recursion.PartsVStar

/-!
# Every level's tops are read and every cap accepts, at the honest run (`hT_honest`, `hC_honest`)

`hok_shaped` takes, besides `IrShaped` and the algebra port names, that the stager reads each level's tops (`topsAt`)
and that each cap's `capIn` accepts. At the fitted records (`InnerRec.fit`) both hold wherever the history holds both
reps' stream rounds and the two reps' tops differ at each later level (the tape exception, `hTops`).

* **The caps** (`capLevel_of_partAt`): V\*'s cap statements are RecCap at level 0, the list's last (`capAt`), and the
  later levels' caps, the `g`-th `.cap (g + 1) (2n)` with `n` the schedule's `g`-th top (`levelCap_at`,
  `topsListed_at`); none of V\*'s levels, algebra parts or candidate 3's statements is a cap.
* **Level 0** (`capIn_zero_honest`): `portRow_cap0` at the last part.
* **Level `g + 1`** (`capIn_later_honest`): the schedule's `g`-th top is round `k`'s `e`-th (`topSizes_split`), and
  `portRow_capl` at stream round `k`.
* **The levels** (`recOpenLevel_of_partAt`): V\*'s level `l` is below the schedule's level count, one more than its
  later caps, so level `l`'s tops are its cap's, and `capIn` reads them (`topsAt_of_capIn`).

`hok_honest` is `hok_shaped` with both discharged.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-! ## The schedule's tops -/

theorem flatMap_index_split {α β : Type} (f : α → List β) :
    ∀ (L : List α) (g : ℕ), g < (L.flatMap f).length →
      ∃ k, ∃ hk : k < L.length, ∃ e < (f L[k]).length, g = ((L.take k).flatMap f).length + e
  | [], g, h => by simp at h
  | a :: L, g, h => by
    by_cases hg : g < (f a).length
    · exact ⟨0, by simp, g, by simpa using hg, by simp⟩
    · rw [List.flatMap_cons, List.length_append] at h
      obtain ⟨k, hk, e, he, hge⟩ := flatMap_index_split f L (g - (f a).length) (by omega)
      refine ⟨k + 1, by simpa using hk, e, by simpa using he, ?_⟩
      rw [List.take_succ_cons, List.flatMap_cons, List.length_append]
      omega

/-- **The schedule's `g`-th top is a round's**: round `k`'s `e`-th, `g = topsBefore rs k + e`. -/
theorem topSizes_split (rs : Array RecVStar.SchedRound) {g : ℕ} (hg : g < (RecVStar.topSizes rs).length) :
    ∃ k e, k < rs.size ∧ e < ((rs.getD k default).2.1).size ∧ g = topsBefore rs k + e := by
  obtain ⟨k, hk, e, he, rfl⟩ :=
    flatMap_index_split (fun w : RecVStar.SchedRound => w.2.1.toList) rs.toList g hg
  have hk' : k < rs.size := by simpa using hk
  refine ⟨k, e, hk', ?_, rfl⟩
  simpa [Array.getD, hk'] using he

theorem arrGetD_of_lt (a : Array ℕ) {e : ℕ} (h : e < a.size) : a.getD e 0 = a[e] := by
  simp [Array.getD, h]

theorem levelCaps_size {sh : _root_.VStar.Algebra.Shape} {K : Array _root_.VStar.Stmt} {s : _root_.Flock.Schedule}
    (h : _root_.VStar.levelCapStatements sh = .ok K) (hs : sh.sched = .ok s) : K.size = s.levels.size - 1 := by
  unfold _root_.VStar.levelCapStatements at h
  rw [hs] at h
  cases h
  simp

/-! ## V\*'s caps and levels -/

/-- **The `g`-th later cap's part** has a circuit that V\* parsed as `.cap (g + 1) (2n)`. -/
theorem capPart_parsed {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {g n : ℕ} (hn : (RecVStar.topSizes rs)[g]? = some n) :
    ∃ c, r.w.parts[capsFrom r rs + g]?.bind partCircuit = some c ∧
      RecVStar.checkParsed (.cap (g + 1) (2 * n)) c = .ok () := by
  have hst := capStmt_at hr hrs hn
  obtain ⟨vs, hvs, -, hall⟩ := readX_partVStar hr
  rw [hvs] at hst
  simp only [Except.toOption, Option.bind_some] at hst
  obtain ⟨hsz, he⟩ := Array.getElem?_eq_some_iff.1 hst
  obtain ⟨-, p, c, -, hp, hc, -, hcp, -⟩ := hall _ hsz
  rw [he] at hcp
  exact ⟨c, by rw [hp, Option.bind_some, hc], hcp⟩

/-- **A cap of V\*'s list** is RecCap at level 0, `capAt`'s, or the `g`-th later cap, `.cap (g + 1) (2n)` with `n`
the schedule's `g`-th top. -/
theorem capLevel_of_partAt {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {s l N : ℕ} {c : Circuit} (hp : partAt x s = some (.cap l N, c)) :
    (l = 0 ∧ capAt r = some (nS x - 1, N)) ∨
      ∃ g m, (RecVStar.topSizes rs)[g]? = some m ∧ l = g + 1 ∧ N = 2 * m := by
  obtain ⟨hst, -⟩ := partAt_some hr hp
  obtain ⟨vs, hvs, hlen⟩ := readX_statements hr
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  rw [hvs] at hst
  simp only [Except.toOption, Option.bind_some] at hst
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, hA, hC, K, n0, hK, hT, rfl⟩ := FlockC3.Reps.list_ok hvs
  obtain ⟨s', _, ps, hsch, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  have hm := Array.mem_of_getElem? hst
  rcases Array.mem_append.1 hm with hm | hm
  · exfalso
    rcases Array.mem_append.1 hm with hm | hm
    · rcases Array.mem_append.1 hm with hm | hm
      · obtain ⟨a, -, he⟩ := List.mem_map.1 (List.mem_toArray.1 hm)
        cases he
      · obtain ⟨a, -, he⟩ := List.mem_map.1 (List.mem_toArray.1 hm)
        cases he
    · rcases FlockC3.Reps.candidate3_kinds hC _ hm with ⟨_, _, he⟩ | ⟨_, _, _, _, he⟩ |
        ⟨_, _, _, _, _, _, he⟩ <;> cases he
  · rcases Array.mem_append.1 hm with hm | hm
    · right
      obtain ⟨g, hg, he⟩ := Array.mem_iff_getElem.1 hm
      have hts := topsListed_size hT hrs
      have hgt : g < (RecVStar.topSizes rs).length := by omega
      have hi := topsListed_at hT hrs (List.getElem?_eq_getElem hgt) hg
      obtain ⟨N', hN'⟩ := levelCap_at hK hg
      rw [hN'] at he hi
      simp only [_root_.VStar.Stmt.cap.injEq] at he
      simp only [_root_.VStar.Stmt.instances] at hi
      exact ⟨g, _, List.getElem?_eq_getElem hgt, he.1.symm, by rw [← he.2]; exact hi⟩
    · left
      simp at hm
      obtain ⟨rfl, rfl⟩ := hm
      refine ⟨rfl, ?_⟩
      rw [hn, hlen]
      simp [capAt, hvs, Except.toOption]

/-- **A level of V\*'s list** is level 0 or one past one of the schedule's tops: V\*'s levels number one more than its
later caps (`levelCapStatements`), one a top (`checkTopsListed`). -/
theorem recOpenLevel_of_partAt {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {s l L H n : ℕ} {c : Circuit}
    (hp : partAt x s = some (.recOpen l L H n, c)) :
    l = 0 ∨ ∃ g m, (RecVStar.topSizes rs)[g]? = some m ∧ l = g + 1 := by
  obtain ⟨hst, -⟩ := partAt_some hr hp
  obtain ⟨vs, hvs, -⟩ := readX_statements hr
  rw [hvs] at hst
  simp only [Except.toOption, Option.bind_some] at hst
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, hA, hC, K, n0, hK, hT, rfl⟩ := FlockC3.Reps.list_ok hvs
  obtain ⟨s', _, ps, hsch, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  have hm := Array.mem_of_getElem? hst
  have hlt : l < s'.levels.size := by
    rcases Array.mem_append.1 hm with hm | hm
    · rcases Array.mem_append.1 hm with hm | hm
      · rcases Array.mem_append.1 hm with hm | hm
        · obtain ⟨a, ha, he⟩ := List.mem_map.1 (List.mem_toArray.1 hm)
          simp only [_root_.VStar.Stmt.recOpen.injEq] at he
          rw [← he.1]
          exact List.mem_range.1 ha
        · obtain ⟨a, -, he⟩ := List.mem_map.1 (List.mem_toArray.1 hm)
          cases he
      · rcases FlockC3.Reps.candidate3_kinds hC _ hm with ⟨_, _, he⟩ | ⟨_, _, _, _, he⟩ |
          ⟨_, _, _, _, _, _, he⟩ <;> cases he
    · rcases Array.mem_append.1 hm with hm | hm
      · obtain ⟨g, hg, he⟩ := Array.mem_iff_getElem.1 hm
        obtain ⟨N', hN'⟩ := levelCap_at hK hg
        rw [hN'] at he
        cases he
      · simp at hm
  have hsz := levelCaps_size hK hsch
  have hts := topsListed_size hT hrs
  rcases Nat.eq_zero_or_pos l with h0 | h0
  · exact .inl h0
  · obtain ⟨g, rfl⟩ : ∃ g, l = g + 1 := ⟨l - 1, by omega⟩
    have hg : g < (RecVStar.topSizes rs).length := by omega
    exact .inr ⟨g, _, List.getElem?_eq_getElem hg, rfl⟩

/-! ## The stager's caps at the honest source -/

theorem srcOf_read {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)}
    {src : _root_.VStar.In.Src} (hs : srcOf I x w ρ hist = some src) :
    ∃ r rs, readX x = some r ∧ repSched r.shape = some rs := by
  cases hr : readX x with
  | none => simp [srcOf, hr] at hs
  | some r =>
    cases hrs : repSched r.shape with
    | none => simp [srcOf, hr, hrs] at hs
    | some rs => exact ⟨r, rs, by first | rfl | exact hr, by first | rfl | exact hrs⟩

/-- **A cap's `capIn` reads its level's tops.** -/
theorem topsAt_of_capIn {src : _root_.VStar.In.Src} {l N : ℕ} {pin : _root_.VStar.In.PartIn}
    (h : _root_.VStar.In.capIn src l N = .ok pin) : ∃ T, _root_.VStar.In.topsAt src l = .ok T := by
  cases ht : _root_.VStar.In.topsAt src l with
  | ok T => exact ⟨T, rfl⟩
  | error e =>
    unfold _root_.VStar.In.capIn at h
    simp [ht, bind, Except.bind] at h

/-- **RecCap at level 0 accepts** at the fitted records (`portRow_cap0` at the last part). -/
theorem capIn_zero_honest {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {src : _root_.VStar.In.Src} (hs : srcOf (I.fit x) x w ρ hist = some src)
    {N : ℕ} (hcap : capAt r = some (nS x - 1, N)) : ∃ pin, _root_.VStar.In.capIn src 0 N = .ok pin := by
  have hst := RecDeploy.OpenSlots.capAt_stmt hr hcap
  obtain ⟨vs, hvs, -, hall⟩ := readX_partVStar hr
  rw [hvs] at hst
  simp only [Except.toOption, Option.bind_some] at hst
  obtain ⟨hsz, he⟩ := Array.getElem?_eq_some_iff.1 hst
  obtain ⟨-, p, c, -, hp, hc, -, hcp, -⟩ := hall _ hsz
  rw [he] at hcp
  obtain ⟨src', pin, hs', hpin, -⟩ := portRow_cap0 (I := I) (w := w) (ρ := ρ) (hist := hist) hr hrs hcap
    (by rw [hp, Option.bind_some, hc]) hcp
  rw [hs] at hs'
  cases hs'
  exact ⟨pin, hpin⟩

/-- **The `g`-th later cap accepts** at the fitted records, where the history holds both reps' stream rounds and
the reps' tops at level `g + 1` differ (`portRow_capl` at the round committing the `g`-th top). -/
theorem capIn_later_honest {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) (hH : 2 * rs.size ≤ hist.length) {src : _root_.VStar.In.Src}
    (hs : srcOf (I.fit x) x w ρ hist = some src) {g n : ℕ} (hn : (RecVStar.topSizes rs)[g]? = some n)
    (hd : ∀ a b, (src.tops[0]!)[g + 1]? = some a → (src.tops[1]!)[g + 1]? = some b →
      _root_.VStar.In.sameTop a b = false) :
    ∃ pin, _root_.VStar.In.capIn src (g + 1) (2 * n) = .ok pin := by
  obtain ⟨hg, -⟩ := List.getElem?_eq_some_iff.1 hn
  obtain ⟨k, e, hk, he, rfl⟩ := topSizes_split rs hg
  rw [topSizes_at hk he] at hn
  cases hn
  obtain ⟨c, hc, hcp⟩ := capPart_parsed hr hrs (topSizes_at hk he)
  have hkR : k % rs.size = k := Nat.mod_eq_of_lt hk
  have hsrc := srcOf_eq hr hrs hs
  subst hsrc
  obtain ⟨src', pin, hs', hpin, -⟩ := portRow_capl (I := I) (x := x) (w := w) (ρ := ρ) (hist := hist) (j := k)
    (e := e) (N := 2 * (rs.getD k default).2.1[e]) (c := c) hr hrs (by omega) hH (by rw [hkR]; exact he)
    (by rw [hkR]; simpa only [Nat.add_assoc] using hc) (by rw [hkR]; exact hcp) (by rw [hkR, arrGetD_of_lt _ he]) (by rw [hkR]; exact hd)
  rw [hs] at hs'
  cases hs'
  rw [hkR] at hpin
  exact ⟨pin, hpin⟩

/-! ## `hok_shaped`'s `hT` and `hC` -/

/-- **Each cap's `capIn` accepts** at the honest run (`hok_shaped`'s `hC`): where the history holds both reps' stream
rounds (`hH`) and the two reps' tops differ at each later level (`hTops`, the tape exception: it fails only where two
192-byte salts collide, `sameTop_false_of_salt`). -/
theorem hC_honest {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape} {hist : List (Msg × Msg)}
    {s : ℕ} (hH : ∀ r rs, readX x = some r → repSched r.shape = some rs → 2 * rs.size ≤ hist.length)
    (hTops : ∀ src, srcOf (Ir.fit x) x w ρ hist = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) :
    ∀ l N c src, partAt x s = some (.cap l N, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.capIn src l N = .ok pin := by
  intro l N c src hp hs
  obtain ⟨r, rs, hr, hrs⟩ := srcOf_read hs
  rcases capLevel_of_partAt hr hrs hp with ⟨rfl, hcap⟩ | ⟨g, m, hm, rfl, rfl⟩
  · exact capIn_zero_honest hr hrs hs hcap
  · exact capIn_later_honest hr hrs (hH r rs hr hrs) hs hm (hTops src hs (g + 1) (Nat.succ_pos g))

/-- **Each level's tops are read** at the honest run (`hok_shaped`'s `hT`): level `l`'s are its cap's, which
accepts (`hC_honest`'s premises). -/
theorem hT_honest {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape} {hist : List (Msg × Msg)}
    {s : ℕ} (hH : ∀ r rs, readX x = some r → repSched r.shape = some rs → 2 * rs.size ≤ hist.length)
    (hTops : ∀ src, srcOf (Ir.fit x) x w ρ hist = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) :
    ∀ l L H n c src, partAt x s = some (.recOpen l L H n, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ T, _root_.VStar.In.topsAt src l = .ok T := by
  intro l L H n c src hp hs
  obtain ⟨r, rs, hr, hrs⟩ := srcOf_read hs
  rcases recOpenLevel_of_partAt hr hrs hp with rfl | ⟨g, m, hm, rfl⟩
  · obtain ⟨N, hcap, -⟩ := capAt_readX hr
    obtain ⟨pin, h⟩ := capIn_zero_honest hr hrs hs hcap
    exact topsAt_of_capIn h
  · obtain ⟨pin, h⟩ :=
      capIn_later_honest hr hrs (hH r rs hr hrs) hs hm (hTops src hs (g + 1) (Nat.succ_pos g))
    exact topsAt_of_capIn h

/-- **The stager accepts every part of a dense statement at the honest run**: `hok_shaped` with its `hT` and `hC`
discharged (`hT_honest`, `hC_honest`). -/
theorem hok_honest {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape} {hist : List (Msg × Msg)}
    {st : Claims.Recursion.CompiledStatement} {s : ℕ} (hx : parse x = some st) (hf : Claims.Recursion.Form.Dense st.form)
    (hI : IrShaped Ir x w ρ hist)
    (hN : ∀ (j : ℕ) (nm : String) (p : _root_.VStar.Part) c, partAt x s = some (.inner j nm p, c) → ∀ xr rs,
      readX x = some xr → repSched xr.shape = some rs → ∀ pn ∈ p.S.ports.map Prod.fst,
        pn = "acc0" ∨ pn = "acc1" ∨ ∃ i, RecVStar.mIndex pn = some i ∧ i < (RecVStar.commitRounds rs).length)
    (hH : ∀ r rs, readX x = some r → repSched r.shape = some rs → 2 * rs.size ≤ hist.length)
    (hTops : ∀ src, srcOf (Ir.fit x) x w ρ hist = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) :
    ∀ sv c src, partAt x s = some (sv, c) → srcOf (Ir.fit x) x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin :=
  hok_shaped hx hf hI (hT_honest hH hTops) hN (hC_honest hH hTops)

end RecDeploy.VStarBuild
