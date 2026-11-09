module

public import Proofs.Flock.SpotFlock.DrawnAt
public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.SpotFlock.HeaderRoots
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.SpotFlock.Depths
import Proofs.Flock.Soundness.Refine.Walk
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The unit each instance of a session is

A registered read's table rows are opened at the population's unit labels (`Registered.unitIndices`). The public file
the verifier draws from the population (`HmRow.drawn`) labels its `i`-th instance with the population's label of the
`i`-th proved unit, and a population whose header round 2's `populationOk` accepts labels its `u`-th unit with slot
`lo + u` of the part.
-/

@[expose] public section

namespace SpotFlock.ReadsAt

open Lean (Json)

/-- `unitsEntry` lists its indices. -/
theorem unitsEntry_indices (c : _root_.Flock.Circuit) (indices : Array ℕ) :
    (_root_.Flock.HmRow.unitsEntry c indices).getObjVal? "indices" = .ok (Spot.natsJson indices) := by
  unfold _root_.Flock.HmRow.unitsEntry
  split
  · exact FlockComplete.JsonRT.getObjVal?_mkObj (by simp) (by simp [Spot.natsJson])
  · exact FlockComplete.JsonRT.getObjVal?_mkObj (by simp) (by simp [Spot.natsJson])

/-- `unitIndices` reads a units entry's indices back. -/
theorem unitIndices_of_get {pub : _root_.Flock.Public} {u : Json} {indices : Array ℕ}
    (hu : pub.header.getObjVal? "units" = .ok u) (hi : u.getObjVal? "indices" = .ok (Spot.natsJson indices)) :
    _root_.Flock.Registered.unitIndices pub = .ok indices := by
  unfold _root_.Flock.Registered.unitIndices _root_.Flock.J.arr _root_.Flock.J.get _root_.Flock.J.asArr
  simp only [hu, hi, bind, Except.bind, Spot.natsJson, Json.getArr?, pure, Except.pure]
  rw [Array.mapM_eq_mapM_toList, Array.toList_map, DrawnAt.mapM_asNat_toJson]
  rfl

/-- **The drawn public file labels its `i`-th instance with the population's label of the `i`-th proved unit.** -/
theorem unitIndices_drawn {c : _root_.Flock.Circuit} {pop pub : _root_.Flock.Public} {d : Json}
    {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pop.header = .obj m)
    (h : _root_.Flock.HmRow.drawn c pop d = .ok pub) :
    ∃ D popIdx, _root_.Flock.Draw.UnitDraw.ofJson d = .ok D ∧ _root_.Flock.Registered.unitIndices pop = .ok popIdx ∧
      _root_.Flock.Registered.unitIndices pub = .ok (D.proved.map fun u => popIdx.getD u 0) := by
  unfold _root_.Flock.HmRow.drawn at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨_, _, ‹_›, ?_, unitIndices_of_get ?_ (unitsEntry_indices _ _)⟩
    · simp only [_root_.Flock.Registered.unitIndices, bind, Except.bind,
        ‹_root_.Flock.J.get pop.header "units" "public file" = .ok _›,
        ‹_root_.Flock.J.arr _ "indices" "public file" = .ok _›]
      exact ‹Array.mapM _ _ = .ok _›
    · dsimp only
      rw [hp]
      simp only [Json.setObjVal!, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?,
        Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert]
      simp

private theorem eq_of_nat? {y : Json} {b : ℕ} (h : Spot.nat? y = some b) : y = Lean.toJson b := by
  cases y with
  | num n =>
    obtain ⟨m, e⟩ := n
    cases e with
    | succ e => simp [Spot.nat?, Spot.int?] at h
    | zero =>
      cases m with
      | ofNat k =>
        have hk : k = b := by simpa [Spot.nat?, Spot.int?] using h
        subst hk
        rfl
      | negSucc k =>
        have hn : ¬ (0 : Int) ≤ Int.negSucc k := Int.not_le.2 (Int.negSucc_lt_zero k)
        simp [Spot.nat?, Spot.int?, hn] at h
  | _ => simp [Spot.nat?, Spot.int?] at h

/-- The only value equivalent to a list of numbers is that list. -/
theorem eq_of_equiv_nats {us : Array ℕ} {w : Json} (he : FlockDeploy.JsonEquiv (Spot.natsJson us) w) :
    w = Spot.natsJson us := by
  cases w with
  | arr a =>
    obtain ⟨hsz, hel⟩ := (FlockDeploy.jsonEquiv_arr_iff (us.map Lean.toJson) a).1 he
    unfold Spot.natsJson
    congr 1
    refine Array.ext hsz.symm fun i h₁ h₂ => ?_
    have := hel i h₂ h₁
    simp only [Array.getElem_map] at this ⊢
    exact Depths.num_of_jsonEquiv this
  | _ => simp [Spot.natsJson, FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at he

/-- A field `Spot.getD` reads as anything but `null` is present. -/
theorem getObjVal_of_getD {j : Json} {k : String} (h : Spot.getD j k ≠ .null) :
    j.getObjVal? k = .ok (Spot.getD j k) := by
  unfold Spot.getD Spot.field? at *
  cases hj : j.getObjVal? k with
  | error e => simp [hj, Except.toOption] at h
  | ok v => simp [hj, Except.toOption]

/-- **A population whose header round 2's `populationOk` accepts labels its `u`-th unit with slot `lo + u` of the
part.** -/
theorem unitIndices_pop {j : Json} (hw : Claims.Json.Wf j) {pop : _root_.Flock.Public}
    (hp : pop.header = _root_.Flock.norm j) {q : ℕ} {p : Spot.SecondStage.Part}
    (hok : Spot.SecondStage.populationOk q p j = true) :
    _root_.Flock.Registered.unitIndices pop = .ok ((Array.range (q * p.n)).map (p.lo + ·)) := by
  unfold Spot.SecondStage.populationOk at hok
  split at hok
  · rename_i is _ hi _
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hok
    obtain ⟨-, hl⟩ := hok
    have hlen := congrArg List.length hl
    simp only [List.length_map, List.length_range, Array.length_toList] at hlen
    have hisJ : Json.arr is = Spot.natsJson ((Array.range (q * p.n)).map (p.lo + ·)) := by
      unfold Spot.natsJson
      congr 1
      refine Array.ext (by simp [hlen]) fun i h₁ _ => ?_
      have hx := List.getElem_of_eq hl (by simpa using h₁)
      simp only [List.getElem_map, List.getElem_range, Array.getElem_toList] at hx
      simp only [Array.getElem_map, Array.getElem_range]
      exact eq_of_nat? hx
    have hun : Spot.getD j "units" ≠ .null := fun h0 => by
      rw [h0] at hi
      simp [Spot.getD, Spot.field?, Json.getObjVal?, Except.toOption] at hi
    have hu := getObjVal_of_getD hun
    have hiu : (Spot.getD j "units").getObjVal? "indices" = .ok (.arr is) := by
      rw [← hi]
      exact getObjVal_of_getD (by rw [hi]; simp)
    obtain ⟨u', hu', -, hwu⟩ := HeaderRoots.getObjVal_equiv hw hw (FlockDeploy.JsonEquiv.refl j) hu
    rw [hu] at hu'
    cases hu'
    obtain ⟨w, hwn, hew, hww⟩ :=
      HeaderRoots.getObjVal_equiv hw (FlockDeploy.wf_norm j) (FlockDeploy.jsonEquiv_norm j hw).symm hu
    obtain ⟨w', hwi, hew', -⟩ := HeaderRoots.getObjVal_equiv hwu hww hew hiu
    rw [hisJ] at hew'
    rw [eq_of_equiv_nats hew'] at hwi
    exact unitIndices_of_get (by rw [hp]; exact hwn) hwi
  · cases hok

/-! ## A registered read's positions, as round 2 reads them -/

/-- A key the file doesn't hold reads as `null`. -/
theorem getD_of_not_has {e : Json} {k : String} (h : ¬ _root_.Flock.J.has e k = true) : Spot.getD e k = .null := by
  unfold _root_.Flock.J.has at h
  unfold Spot.getD Spot.field?
  cases hg : e.getObjVal? k with
  | ok v => rw [hg] at h; exact absurd rfl h
  | error s => rfl

theorem nat?_num (n : ℕ) : Spot.nat? (.num ⟨Int.ofNat n, 0⟩) = some n := by
  simp [Spot.nat?, Spot.int?]

/-- **What a `ports` file says of a port's reads** at its entry `e`: its first slot, `reads_from`'s, or `0` where the
entry names none, and each position it reads at its place in the entry's `reads`. -/
abbrev ReadFields (e : Json) (reg : _root_.Flock.Registered.Port) : Prop :=
  ((Spot.getD e "reads_from" = .null ∧ reg.readsFrom = 0) ∨
      Spot.getD e "reads_from" = .num ⟨Int.ofNat reg.readsFrom, 0⟩) ∧
    ∃ rs, Spot.getD e "reads" = .arr rs ∧
      ∀ k pos, reg.readsRel[k]? = some (some pos) → rs[k]? = some (.num ⟨Int.ofNat pos, 0⟩)

/-- **A `ports` file's ports read as its entries say** (`ReadFields`), each at an entry under its name. -/
theorem portsOf_reads {j : Json} {own : Array _root_.Flock.Registered.Port}
    (h : _root_.Flock.Registered.portsOf j = .ok own) :
    ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
      ∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ ReadFields e reg := by
  unfold _root_.Flock.Registered.portsOf at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    refine ⟨_, ‹_›, fun reg hreg => ?_⟩
    obtain ⟨⟨nm, e⟩, hkv, h⟩ := (mapM_ok_mem h).1 reg hreg
    dsimp only at h
    simp only [FlockSoundness.Refine.throw_bind'] at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      refine ⟨e, hkv, ?_, ?_⟩
      · have hx := ‹(if _root_.Flock.J.has e _root_.Flock.Registered.READS_FROM_KEY = true then _ else _) = .ok _›
        by_cases hc : _root_.Flock.J.has e _root_.Flock.Registered.READS_FROM_KEY = true
        · rw [if_pos hc] at hx
          exact .inr (HeaderRoots.getD_of_getObjVal (FlockComplete.JsonRT.nat_ok hx))
        · rw [if_neg hc] at hx
          cases hx
          exact .inl ⟨getD_of_not_has hc, rfl⟩
      · obtain ⟨hsz, hel⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem
          ‹Array.mapM _ _ = (Except.ok _ : _root_.Flock.V (Array (Option ℕ)))›
        refine ⟨_, HeaderRoots.getD_of_getObjVal (DrawnAt.getObjVal_of_arr ‹_root_.Flock.J.arr e "reads" _ = .ok _›),
          fun k pos hk => ?_⟩
        obtain ⟨hk1, hk2⟩ := Array.getElem?_eq_some_iff.1 hk
        have hk0 : k < _ := hsz ▸ hk1
        have hfk := hel k hk0 hk1
        rw [hk2] at hfk
        rw [Array.getElem?_eq_getElem hk0]
        dsimp only at hfk
        split at hfk
        · cases hfk
        · simp only [Functor.map, Except.map] at hfk
          split at hfk
          · cases hfk
          · rename_i n hn
            cases hfk
            rw [FlockComplete.JsonRT.asNat_ok hn]

/-- **Round 2's rows at a read**: where `rowsOk` accepts a registered read's entry, the position it reads at slot `j`
is the first row of the unit `unit j` names, `leaves / units` rows a unit, and the slot is the part's. -/
theorem pos_of_rowsOk {q : ℕ} {p : Spot.SecondStage.Part} {e : Json} {units : ℕ} {unit : ℕ → Option ℕ}
    (hrows : Spot.SecondStage.rowsOk q p e units unit = true) {reg : _root_.Flock.Registered.Port}
    (hf : ReadFields e reg) (hl : Spot.getD e "leaves" = .num ⟨Int.ofNat reg.leaves, 0⟩)
    {j pos : ℕ} (hj : reg.reads[j]? = some (some pos)) :
    0 < units ∧ ∃ b, unit j = some b ∧ p.holds q j = true ∧ pos = b * (reg.leaves / units) := by
  rw [← _root_.Flock.Registered.Port.readAt_eq, _root_.Flock.Registered.Port.readAt] at hj
  split at hj
  · cases hj
  rename_i hlt
  obtain ⟨hfirst, rs, hrs, hrd⟩ := hf
  have hk := hrd _ _ hj
  have hsz : j - reg.readsFrom < rs.size := (Array.getElem?_eq_some_iff.1 hk).1
  have hget : rs.getD (j - reg.readsFrom) .null = .num ⟨Int.ofNat pos, 0⟩ := by
    rw [Array.getD_eq_getD_getElem?, hk, Option.getD_some]
  have hfj : reg.readsFrom + (j - reg.readsFrom) = j := by omega
  unfold Spot.SecondStage.rowsOk at hrows
  rcases hfirst with ⟨h0, h1⟩ | h0
  · simp only [h0, hl, hrs, nat?_num, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true,
      List.mem_range] at hrows
    obtain ⟨⟨hu, -⟩, hall⟩ := hrows
    have hx := hall _ hsz
    rw [hget] at hx
    have hfj0 : 0 + (j - reg.readsFrom) = j := by omega
    simp only [nat?_num, hfj0] at hx
    refine ⟨hu, ?_⟩
    cases hb : unit j with
    | none => simp [hb] at hx
    | some b =>
      simp only [hb, Bool.and_eq_true, beq_iff_eq] at hx
      exact ⟨b, rfl, hx⟩
  · simp only [h0, hl, hrs, nat?_num, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true,
      List.mem_range] at hrows
    obtain ⟨⟨hu, -⟩, hall⟩ := hrows
    have hx := hall _ hsz
    rw [hget] at hx
    simp only [nat?_num, hfj] at hx
    refine ⟨hu, ?_⟩
    cases hb : unit j with
    | none => simp [hb] at hx
    | some b =>
      simp only [hb, Bool.and_eq_true, beq_iff_eq] at hx
      exact ⟨b, rfl, hx⟩

end SpotFlock.ReadsAt
