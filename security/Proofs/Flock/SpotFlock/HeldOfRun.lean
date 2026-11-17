module

public import Proofs.Flock.SpotFlock.HeldOfCheck
public import Proofs.Flock.SpotFlock.ScopeDesc
public import Proofs.Flock.SpotFlock.Placement
public import Proofs.Flock.SpotFlock.RunsAs.PopulationOf

/-! The run's population check gives `filesOk` and `fillOk` at the copies `heldOf` builds by construction, so the design
evaluates there with no step per population slot (`twoHolds_of_pop`). -/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs SpotFlock.RunsAs.Round2
open Lean (Json)

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

theorem built_of_templates (h : Held) (ht : ∀ t < h.ms.length, templateOk h t = true) :
    ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit := by
  intro ci hci
  obtain ⟨t, htl, rfl⟩ := List.getElem_of_mem hci
  exact (templateOk_facts h (ht t htl) htl).2.1

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

theorem firstOf_program {i : Spot.Run.Input} {first : Reg} (hf : firstOf i = some first) :
    first.program = i.params.verifier.program ∧ first.query = i.params.verifier.query := by
  unfold firstOf at hf
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hf
  obtain ⟨n, hn, rule, hrule, samplingDesign, hlaw, workTable, hwt, resolved, hres, rfl⟩ := hf
  exact ⟨rfl, rfl⟩

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

theorem twoHolds_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held} {j : Json}
    (hh : heldOf i it u = some h) (ha : AcceptedTwo i it j) : TwoHolds h := by
  obtain ⟨_, -, hp⟩ := population_of_accepted ha
  exact twoHolds_of_pop hh hp

theorem twoHolds_of_ok {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (u : Unread)
    (hk : heldOk i it u = true) (ha : AcceptedTwo i it j) :
    ∃ h, heldOf i it u = some h ∧ TwoHolds h := by
  obtain ⟨h, hh⟩ := heldOf_of_ok u hk
  exact ⟨h, hh, twoHolds_of hh ha⟩

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
