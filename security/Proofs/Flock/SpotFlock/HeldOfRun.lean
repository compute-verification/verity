module

public import Proofs.Flock.SpotFlock.HeldOfCheck
public import Proofs.Flock.SpotFlock.ScopeDesc
public import Proofs.Flock.SpotFlock.Placement
public import Proofs.Flock.SpotFlock.RunsAs.PopulationOf

/-!
# What the run's population check gives the held copies

`descOk` (`ScopeDesc`) decides the two-stage scope at the members from the descriptor; two of its checks,
`filesOk` and `fillOk`, take a step per population slot and unit. At the copies `heldOf` builds, neither is a new check:
the run refuses a boundary registration whose population is not its `n` units in order (`Slots.popOk`,
`population-order`), and given that, both hold by construction, since `heldOf` gives template `t` the units
`[0, n·n_t)` and request `ρ`'s block of it the slots `filesBlockOf … ρ t`.

* `pop_facts`: under `popOk`, the layout's blocks are its population, units `0` to `n - 1` in slot order, then input
  slots, which name no unit (`layout_shape`).
* `unit_of_popOk`, `block_of_popOk`, `requests_of_popOk`: a slot naming a unit names its own index, below `n`; every
  index below `n` is such a slot; and `n` slots name a unit.
* `filesOk_held₀`, `fillOk_held₀`: `filesOk` and `fillOk` at the copies, from `popOk` and, for `filesOk`, the cells
  naming one count per template.
* `built_of_templates`: every member's unit net is built where every template's reference statement reads as a
  member's (`templateOk`, through `templateOk_facts`), so `two` at the copies (`two_held₀`) takes no `Built`.
* `twoHolds_of_pop`: a held copy `heldOf` builds gives RUNCOINS's `TwoHolds` where the run's population check passes.
* `twoHolds_of`, `twoHolds_of_ok`, `twoHolds_of_files`: the same at an accepted two-stage run, whose population check
  passed (`RunsAs.population_of_accepted`).
* `xS₀_of_files`: a held copy built from the form's members and blobs has template `t`'s reference statement
  `refRawOf`'s bytes.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs SpotFlock.RunsAs.Round2
open Lean (Json)

/-- **A layout's classes**: its population, then a class of input slots, which name no unit. -/
theorem layout_shape {p : Spot.Program} {query : Json} {l : Spot.Slots.SlotOrder}
    (h : Spot.Slots.layout p query = .ok l) :
    ∃ pop ins, l.classes = #[pop, ins] ∧ ∀ s ∈ ins, ∀ b ∈ s, b.unit = none := by
  unfold Spot.Slots.layout at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h
       refine ⟨_, _, rfl, ?_⟩
       simp)

theorem pop_facts {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.SlotOrder}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) :
    ∃ ins : Array (Array Spot.Slots.ProofBlock), (∀ s ∈ ins, ∀ b ∈ s, b.unit = none) ∧
      l.blocks.toList = Spot.Slots.popOf l ++ (ins.toList.map Array.toList).flatten ∧
      (Spot.Slots.popOf l).map (·.unit) = (List.range n).map some := by
  obtain ⟨pop, ins, hc, hins⟩ := layout_shape hl
  refine ⟨ins, hins, ?_, ?_⟩
  · unfold Spot.Slots.SlotOrder.blocks Spot.Slots.popOf
    rw [hc]
    simp
  · unfold Spot.Slots.popOk at hp
    rw [hl] at hp
    exact of_decide_eq_true hp

theorem length_popOf {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.SlotOrder}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) :
    (Spot.Slots.popOf l).length = n := by
  obtain ⟨-, -, -, hpop⟩ := pop_facts hl hp
  simpa using congrArg List.length hpop

/-- **A slot naming a unit names its own index**, below `n`, under `popOk`. -/
theorem unit_of_popOk {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.SlotOrder}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true)
    {ρ : ℕ} {b : Spot.Slots.ProofBlock} (hb : l.blocks[ρ]? = some b) {v : ℕ} (hv : b.unit = some v) :
    v = ρ ∧ ρ < n := by
  obtain ⟨ins, hins, hbl, hpop⟩ := pop_facts hl hp
  have hlen := length_popOf hl hp
  rw [← Array.getElem?_toList, hbl] at hb
  rcases Nat.lt_or_ge ρ (Spot.Slots.popOf l).length with hρ | hρ
  · rw [List.getElem?_append_left hρ] at hb
    have h1 : ((Spot.Slots.popOf l).map (·.unit))[ρ]? = some (some v) := by
      rw [List.getElem?_map, hb]
      simp [hv]
    rw [hpop, List.getElem?_map, List.getElem?_range (by omega)] at h1
    simp only [Option.map_some, Option.some.injEq] at h1
    exact ⟨h1.symm, by omega⟩
  · rw [List.getElem?_append_right hρ] at hb
    have hmem := List.mem_of_getElem? hb
    simp only [List.mem_flatten, List.mem_map] at hmem
    obtain ⟨_, ⟨s, hs, rfl⟩, hbs⟩ := hmem
    have hn := hins s (by simpa using hs) b (by simpa using hbs)
    rw [hn] at hv
    cases hv

/-- **Every index below `n` is a slot naming itself**, under `popOk`. -/
theorem block_of_popOk {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.SlotOrder}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) {ρ : ℕ} (hρ : ρ < n) :
    ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ := by
  obtain ⟨ins, -, hbl, hpop⟩ := pop_facts hl hp
  have hρ' : ρ < (Spot.Slots.popOf l).length := by rw [length_popOf hl hp]; exact hρ
  refine ⟨(Spot.Slots.popOf l)[ρ], ?_, ?_⟩
  · rw [← Array.getElem?_toList, hbl, List.getElem?_append_left hρ', List.getElem?_eq_getElem hρ']
  · have h1 := congrArg (·[ρ]?) hpop
    simp only [List.getElem?_map, List.getElem?_eq_getElem hρ', List.getElem?_range hρ, Option.map_some,
      Option.some.injEq] at h1
    exact h1

/-- **`n` slots name a unit**, under `popOk`: `heldOf`'s request count is the population. -/
theorem requests_of_popOk {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.SlotOrder}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) : requests l = n := by
  obtain ⟨ins, hins, hbl, hpop⟩ := pop_facts hl hp
  have h1 : (Spot.Slots.popOf l).filter (fun b => b.unit.isSome) = Spot.Slots.popOf l := by
    refine List.filter_eq_self.2 fun b hb => ?_
    have hm : b.unit ∈ (List.range n).map some := hpop ▸ List.mem_map_of_mem hb
    obtain ⟨k, -, hk⟩ := List.mem_map.1 hm
    simp [← hk]
  have h2 : ((ins.toList.map Array.toList).flatten).filter (fun b => b.unit.isSome) = [] := by
    refine List.filter_eq_nil_iff.2 fun b hb => ?_
    simp only [List.mem_flatten, List.mem_map] at hb
    obtain ⟨_, ⟨s, hs, rfl⟩, hbs⟩ := hb
    simp [hins s (by simpa using hs) b (by simpa using hbs)]
  unfold requests
  rw [← Array.length_toList, Array.toList_filter, hbl, List.filter_append, h1, h2, List.append_nil,
    length_popOf hl hp]

theorem getElem?_ms_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg}
    {ns ks : Array ℕ} {r : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true} {t : ℕ} (ht : t < it.circuits.size) :
    ((held₀ i it u first ns ks r hz).ms[t]?).map (·.2.size) = some (r * ns.getD t 0) := by
  simp [Held.ms, held₀, unitsOf, ht, List.getD_eq_getElem?_getD]

/-- **The layout fits the members' files** (`filesOk`) at the copies `heldOf` builds, from the population check and
the cells naming one count per template. -/
theorem filesOk_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg} {ns ks : Array ℕ}
    {l : Spot.Slots.SlotOrder} {n : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true}
    (hl : Spot.Slots.layout first.program first.query = .ok l)
    (hp : Spot.Slots.popOk first.program first.query n = true) (hT : ns.size = it.circuits.size) :
    filesOk (held₀ i it u first ns ks (requests l) hz).reg (held₀ i it u first ns ks (requests l) hz).ms l
      (held₀ i it u first ns ks (requests l) hz).k = true := by
  have hr := requests_of_popOk hl hp
  have hlen := ms_length_held₀ i it u first ns ks (requests l) hz
  unfold filesOk
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range]
  refine ⟨⟨?_, fun t _ => ?_⟩, fun ρ _ => ?_⟩
  · rw [hlen]
    exact hT
  · show ks.toList.getD t 0 = ks.getD t 0
    by_cases h : t < ks.size <;> simp [List.getD_eq_getElem?_getD, Array.getD, h]
  · rcases hb : l.blocks[ρ]? with _ | b
    · rfl
    · dsimp only
      rcases hv : b.unit with _ | v
      · rfl
      · dsimp only
        obtain ⟨rfl, hρ⟩ := unit_of_popOk hl hp hb hv
        simp only [beq_self_eq_true, Bool.true_and, List.all_eq_true, List.mem_range, Bool.and_eq_true,
          decide_eq_true_eq, beq_iff_eq]
        intro t ht
        rw [hlen] at ht
        refine ⟨?_, fun _ _ => rfl⟩
        rw [getElem?_ms_held₀ ht, Option.getD_some, hr]
        exact Nat.mul_le_mul_right _ hρ

/-- **The layout fills the members' files** (`fillOk`) at the copies `heldOf` builds, from the population check. -/
theorem fillOk_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg} {ns ks : Array ℕ}
    {l : Spot.Slots.SlotOrder} {n : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true}
    (hl : Spot.Slots.layout first.program first.query = .ok l)
    (hp : Spot.Slots.popOk first.program first.query n = true) :
    fillOk (held₀ i it u first ns ks (requests l) hz).ms ns l = true := by
  have hr := requests_of_popOk hl hp
  unfold fillOk
  simp only [List.all_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq]
  intro t ht x hx
  rw [ms_length_held₀] at ht
  rw [getElem?_ms_held₀ ht, Option.getD_some, hr] at hx
  have hn : 0 < ns.getD t 0 := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hx; omega
  obtain ⟨b, hb, hu⟩ := block_of_popOk hl hp (Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hx))
  refine ⟨hn, ?_⟩
  rw [hb]
  simp [hu]

/-! ## `descOk` at the copies -/

theorem reg_withFrame (h : Held) (c₀ : Code) : (withFrame h c₀).reg = h.reg := by
  unfold withFrame
  dsimp only

theorem k_withFrame (h : Held) (c₀ : Code) : (withFrame h c₀).k = h.k := by
  unfold withFrame
  dsimp only

theorem xS₀_withFrame (h : Held) (c₀ : Code) : (withFrame h c₀).xS₀ = h.xS₀ := by
  unfold withFrame
  dsimp only

theorem descOk_withFrame (h : Held) (c₀ : Code) : descOk (withFrame h c₀) = descOk h := by
  simp only [descOk, descAt, replayDesign, ms_withFrame, reg_withFrame, k_withFrame]

/-- **The registration's descriptor is in the two-stage scope** (`descOk`) at the copies `heldOf` builds, from the
run's population check and what is left per template: each member fits (`memberOk`), the first stage's design is well
formed on the layout's strata (`designOk`), and one count of at least `1` and one cell per template. -/
theorem descOk_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg} {ns ks : Array ℕ}
    {l : Spot.Slots.SlotOrder} {d : Design} {n : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true}
    (hl : Spot.Slots.layout first.program first.query = .ok l)
    (hd : slotDesign first.rule first.program first.query first.table = .ok d)
    (hp : Spot.Slots.popOk first.program first.query n = true) (hT : ns.size = it.circuits.size)
    (hm : (held₀ i it u first ns ks (requests l) hz).ms.all memberOk = true) (hdes : designOk d l = true)
    (hk : ks.size = it.circuits.size) (hk1 : ∀ x ∈ ks.toList, 1 ≤ x) :
    descOk (held₀ i it u first ns ks (requests l) hz) = true := by
  have hlen := ms_length_held₀ i it u first ns ks (requests l) hz
  have e1 : (Spot.Slots.layout (held₀ i it u first ns ks (requests l) hz).reg.first.program
      (held₀ i it u first ns ks (requests l) hz).reg.first.query).toOption = some l := by
    show (Spot.Slots.layout first.program first.query).toOption = some l
    rw [hl]
    rfl
  have e2 : replayDesign (held₀ i it u first ns ks (requests l) hz) = some d := by
    show (slotDesign first.rule first.program first.query first.table).toOption = some d
    rw [hd]
    rfl
  unfold descOk
  rw [e1, e2]
  dsimp only
  unfold descAt
  simp only [Bool.and_eq_true, beq_iff_eq]
  refine ⟨⟨⟨⟨⟨hm, filesOk_held₀ hl hp hT⟩, fillOk_held₀ hl hp⟩, hdes⟩, ?_⟩, ?_⟩
  · show ks.toList.length = _
    rw [hlen, Array.length_toList, hk]
  · show ks.toList.all (1 ≤ ·) = true
    exact List.all_eq_true.2 fun x hx => decide_eq_true (hk1 x hx)

/-- **Every member's unit net is built** where every template's reference statement reads as a member's: its circuit
file's descriptor parses at untyped tags (`templateOk_facts`). -/
theorem built_of_templates (h : Held) (ht : ∀ t < h.ms.length, templateOk h t = true) :
    ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit := by
  intro ci hci
  obtain ⟨t, htl, rfl⟩ := List.getElem_of_mem hci
  exact (templateOk_facts h (ht t htl) htl).2.1

/-- **The design evaluates at the copies `heldOf` builds** (`two`), from `descOk_held₀`'s premises and every template's
reference statement reading as a member's (`templateOk`). -/
theorem two_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg} {ns ks : Array ℕ}
    {l : Spot.Slots.SlotOrder} {d : Design} {n : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true} (s : AuditStatement)
    (hl : Spot.Slots.layout first.program first.query = .ok l)
    (hd : slotDesign first.rule first.program first.query first.table = .ok d)
    (hp : Spot.Slots.popOk first.program first.query n = true) (hT : ns.size = it.circuits.size)
    (hm : (held₀ i it u first ns ks (requests l) hz).ms.all memberOk = true) (hdes : designOk d l = true)
    (hk : ks.size = it.circuits.size) (hk1 : ∀ x ∈ ks.toList, 1 ≤ x)
    (ht : ∀ t < (held₀ i it u first ns ks (requests l) hz).ms.length,
      templateOk (held₀ i it u first ns ks (requests l) hz) t = true)
    (hsc : membersScoped (held₀ i it u first ns ks (requests l) hz) = true)
    (hw : wiringOk (held₀ i it u first ns ks (requests l) hz) = true) :
    (two (withFrame (held₀ i it u first ns ks (requests l) hz) u.c₀) s).isSome := by
  rw [two_withFrame]
  exact two_of_descOk _ s (built_of_templates _ ht) (descOk_held₀ hl hd hp hT hm hdes hk hk1) ht hsc hw

/-- **The boundary registration `heldOf` hands the first stage reads the params' program and query.** -/
theorem firstOf_program {i : Spot.Run.Input} {first : Reg} (hf : firstOf i = some first) :
    first.program = i.params.verifier.program ∧ first.query = i.params.verifier.query := by
  unfold firstOf at hf
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hf
  obtain ⟨n, hn, rule, hrule, law, hlaw, workTable, hwt, resolved, hres, rfl⟩ := hf
  exact ⟨rfl, rfl⟩

/-- **The design evaluates** (RUNCOINS's `TwoHolds`) at a held copy `heldOf` builds, where the run's population check
passes (`Slots.popOk` at the params' program and query, which `Run.run` refuses without: `population-order`): `heldOf`
decides the descriptor's checks (`descCore`), and the population gives the layout's fit to the members' files. -/
theorem twoHolds_of_pop {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held} {n : ℕ}
    (hh : heldOf i it u = some h)
    (hp : Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true) : TwoHolds h := by
  obtain ⟨first, ns, ks, l, hz, hf, -, hl, rfl, hd⟩ := heldOf_spec hh
  obtain ⟨hpr, hq⟩ := firstOf_program hf
  rw [← hpr, ← hq] at hp
  unfold decided at hd
  simp only [Bool.and_eq_true] at hd
  have hcore := hd.2
  rw [descCore_withFrame] at hcore
  have hlen := ms_length_held₀ i it u first ns ks (requests l) hz
  have e1 : (Spot.Slots.layout (held₀ i it u first ns ks (requests l) hz).reg.first.program
      (held₀ i it u first ns ks (requests l) hz).reg.first.query).toOption = some l := by
    show (Spot.Slots.layout first.program first.query).toOption = some l
    rw [hl]
    rfl
  unfold descCore at hcore
  rw [e1] at hcore
  rcases hdz : replayDesign (held₀ i it u first ns ks (requests l) hz) with _ | d
  · rw [hdz] at hcore
    simp at hcore
  rw [hdz] at hcore
  dsimp only at hcore
  simp only [Bool.and_eq_true, beq_iff_eq] at hcore
  obtain ⟨⟨⟨⟨⟨⟨⟨hm, hdes⟩, hk⟩, hk1⟩, hT⟩, ht⟩, hsc⟩, hw⟩ := hcore
  have hd' : (slotDesign first.rule first.program first.query first.table).toOption = some d := hdz
  have hk' : ks.size = it.circuits.size := by
    rw [← hlen, ← Array.length_toList]
    exact hk
  have hT' : ns.size = it.circuits.size := by
    rw [← hlen]
    exact hT
  intro s
  exact two_held₀ s hl (ok_of_toOption hd') hp hT' hm hdes hk'
    (fun x hx => of_decide_eq_true (List.all_eq_true.1 hk1 x hx))
    (fun t htl => List.all_eq_true.1 ht t (List.mem_range.2 htl)) hsc hw

/-- **The design evaluates** at a held copy `heldOf` builds for an accepted two-stage run (`AcceptedTwo`), whose
population check passed (`population_of_accepted`). -/
theorem twoHolds_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held} {j : Json}
    (hh : heldOf i it u = some h) (ha : AcceptedTwo i it j) : TwoHolds h := by
  obtain ⟨_, -, hp⟩ := population_of_accepted ha
  exact twoHolds_of_pop hh hp

theorem twoHolds_of_ok {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (u : Unread)
    (hk : heldOk i it u = true) (ha : AcceptedTwo i it j) :
    ∃ h, heldOf i it u = some h ∧ TwoHolds h := by
  obtain ⟨h, hh⟩ := heldOf_of_ok u hk
  exact ⟨h, hh, twoHolds_of hh ha⟩

/-- **The form's check of the held copies gives `TwoHolds`** at an accepted two-stage run. -/
theorem twoHolds_of_files {p : VStar.Recursive.SampledTwo.Params} {bs : Core.Blobs} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} {j : Json} (u₀ : Unread) (hk : heldOkDesc p bs i it u₀ = true)
    (ha : AcceptedTwo i it j) :
    ∃ u h, unreadOf p bs u₀.desc u₀.query u₀.c₀ u₀.more = some u ∧ heldOf i it u = some h ∧ TwoHolds h := by
  unfold heldOkDesc at hk
  cases hx : refStmtsOf p bs with
  | none => rw [hx] at hk; simp at hk
  | some xs =>
    rw [hx] at hk
    obtain ⟨h, hh, ht⟩ := twoHolds_of_ok { u₀ with xs } hk ha
    exact ⟨_, h, by simp [unreadOf, hx], hh, ht⟩

/-- **A held copy's reference statements are the form's** (`refRawOf`'s bytes, template by template) where `heldOf`
builds it from what the form's members and blobs give (`unreadOf`). -/
theorem xS₀_of_files {p : VStar.Recursive.SampledTwo.Params} {bs : Core.Blobs} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} {desc : Msg} {query : Query} {c₀ : Code} {more : ℕ} {u : Unread} {h : Held} {t : ℕ}
    (hu : unreadOf p bs desc query c₀ more = some u) (hh : heldOf i it u = some h) (ht : t < p.circuits.size) :
    ∃ r, refRawOf p bs t = some r ∧ h.xS₀ t = r.encode := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  rw [xS₀_withFrame]
  unfold unreadOf at hu
  cases hx : refStmtsOf p bs with
  | none => rw [hx] at hu; cases hu
  | some xs =>
    rw [hx] at hu
    obtain rfl : (⟨desc, query, xs, c₀, more⟩ : Unread) = u := Option.some.inj hu
    obtain ⟨r, hr, hxt⟩ := refStmtsOf_getElem? hx ht
    refine ⟨r, hr, ?_⟩
    show refStmt xs t = r.encode
    simp [refStmt, List.getD_eq_getElem?_getD, hxt]

end SpotFlock.HeldOf
