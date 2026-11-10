module

public import Proofs.Flock.SpotFlock.DrawnAt
public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.SpotFlock.HeaderRoots
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.Complete.Json.RoundTrip
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.SpotFlock.Depths
import Proofs.Flock.SpotFlock.Opens
import Proofs.Flock.SpotFlock.Reads
import Proofs.Flock.SpotFlock.RowsOpen
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Layout.QSort
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk
meta import Proofs.Flock.Soundness.Refine.Walk
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The unit each instance of a session is

A registered read's table rows are opened at the population's unit labels (`Registered.unitIndices`). The public file
the verifier draws from the population (`HmRow.drawn`) labels its `i`-th instance with the population's label of the
`i`-th proved unit, and a population whose header round 2's `populationOk` accepts labels its `u`-th unit with slot
`lo + u` of the part (`CheckedRun.labelAt`). A registered read's entry in the session statement states its leaves,
first slot and positions (`ReadFields`). Round 2 holds each position it reads to its unit's slot: at the template's own
port, to the first row of the slot's unit (`rowsOk`, `pos_of_rowsOk`); at a port a link or a wire names, to the row the
design's rows give the slot's unit (`placedOk`, `pos_of_placedOk`). `arm_of_readOk` says which of the two an accepted
read's arm takes.

The unit's instance is the first slot of its session's sample whose proof block is the unit (`CheckedRun.blockAt_rank`):
slot `i·n_t + p` of its template's part, with the unit's replay block the sample's `i`-th and the unit that block's
`p`-th proof block of the template (`CheckedRun.slot_cell`). So a slot of stride `n_t` at position `p` places the
instance under its replay block's roots, which is the arm fact `leafAt_reg` takes where the read is the template's own
port (`CheckedRun.harm_own`, `CheckedRun.leafAt_own`). A wired read's slot is the row its unit reads of its writer's
rows in the cell, of stride the rows' `per` at that row (`CheckedRun.harm_wire`), and a linked read's is that row of
the replay block's rows in the boundary port (`CheckedRun.harm_link`).
-/

namespace SpotFlock.ReadsAt

open Lean Meta in
/-- A step of `ret_walk_inlining` on the goal `Ret m Q`: where `m` binds a join point whose parameter is named one of
`names`, `m` with the join point inlined; none on any other goal. -/
meta def inlineStep (names : Array Name) (g : MVarId) : MetaM (Option (List MVarId)) := g.withContext do
  let ty ← instantiateMVars (← g.getType)
  unless ty.isAppOfArity ``FlockSoundness.Refine.Ret 3 do return none
  let .letE _ _ v b _ := (ty.getArg! 1).headBeta.consumeMData | return none
  let .lam x _ _ _ := v.consumeMData | return none
  unless names.contains x.eraseMacroScopes do return none
  let g' ← mkFreshExprSyntheticOpaqueMVar
    (mkApp3 (mkConst ``FlockSoundness.Refine.Ret) (ty.getArg! 0) (b.instantiate1 v) (ty.getArg! 2))
  g.assign g'
  return some [g'.mvarId!]

open Lean Elab Tactic Meta in
/-- `ret_walk_inlining [x, …] [c, …]` is `ret_walk [c, …]` but inlines each join point whose parameter is named one of
the `x`s. `ret_walk` walks a join point's body once, at a value it knows nothing of; inlined, the body is walked at each
jump, at the value the jump passes and with the equation the walk kept for it. -/
elab "ret_walk_inlining" "[" xs:ident,* "]" "[" cs:ident,* "]" : tactic => do
  let names := xs.getElems.map (·.getId)
  let mut keep := #[]
  for c in cs.getElems do keep := keep.push (← realizeGlobalConstNoOverloadWithInfo c)
  let mut todo := (← getGoals)
  let mut left := #[]
  while true do
    match todo with
    | [] => break
    | g :: rest =>
      match ← inlineStep names g with
      | some new => todo := new ++ rest
      | none =>
        match ← FlockSoundness.Refine.retStep keep g with
        | some new => todo := new ++ rest
        | none => left := left.push g; todo := rest
  setGoals left.toList

end SpotFlock.ReadsAt

@[expose] public section

namespace SpotFlock.ReadsAt

open Claims Claims.Sampling SamplingDeploy
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

/-- `unitIndices` as `drawn` reads it, one bind at a time. -/
theorem unitIndices_walk {pub : _root_.Flock.Public} {u : Json} {a : Array Json} {popIdx : Array ℕ}
    (h1 : _root_.Flock.J.get pub.header "units" "public file" = .ok u)
    (h2 : _root_.Flock.J.arr u "indices" "public file" = .ok a)
    (h3 : a.mapM (_root_.Flock.J.asNat · "a unit index") = .ok popIdx) :
    _root_.Flock.Registered.unitIndices pub = .ok popIdx := by
  simp only [_root_.Flock.Registered.unitIndices, h1, h2, h3, bind, Except.bind]

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
    exact ⟨_, _, ‹_›, unitIndices_walk ‹_› ‹_› ‹_›, unitIndices_of_get (by
      dsimp only
      rw [hp]
      simp only [Json.setObjVal!, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?,
        Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert]
      simp [pure, Except.pure]) (unitsEntry_indices c _)⟩

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
      rw [h0, show Spot.getD Json.null "indices" = Json.null from rfl] at hi
      cases hi
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

/-- **What a `ports` file says of a port's reads** at its entry `e`: its leaf count, its first slot, `reads_from`'s,
or `0` where the entry names none, and each position it reads at its place in the entry's `reads`. -/
abbrev ReadFields (e : Json) (reg : _root_.Flock.Registered.Port) : Prop :=
  Spot.getD e "leaves" = .num ⟨Int.ofNat reg.leaves, 0⟩ ∧
  ((Spot.getD e "reads_from" = .null ∧ reg.readsFrom = 0) ∨
      Spot.getD e "reads_from" = .num ⟨Int.ofNat reg.readsFrom, 0⟩) ∧
    ∃ rs : Array Json, Spot.getD e "reads" = .arr rs ∧
      ∀ (k pos : ℕ), reg.readsRel[k]? = some (some pos) → rs[k]? = some (.num ⟨Int.ofNat pos, 0⟩)

/-- A port's reads as a `ports` file's entry gives them: each position read at its place in the entry's `reads`. -/
theorem reads_of {e : Json} {ctx : String} {rs : Array Json} {reads : Array (Option ℕ)}
    {f : Json → _root_.Flock.V (Option ℕ)} (ha : _root_.Flock.J.arr e "reads" ctx = .ok rs)
    (hm : rs.mapM f = .ok reads) (hf : ∀ r n, f r = .ok (some n) → r = .num ⟨Int.ofNat n, 0⟩) :
    ∃ rs : Array Json, Spot.getD e "reads" = .arr rs ∧
      ∀ (k pos : ℕ), reads[k]? = some (some pos) → rs[k]? = some (.num ⟨Int.ofNat pos, 0⟩) := by
  obtain ⟨hsz, hel⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hm
  refine ⟨rs, HeaderRoots.getD_of_getObjVal (DrawnAt.getObjVal_of_arr ha), fun k pos hk => ?_⟩
  obtain ⟨hk1, hk2⟩ := Array.getElem?_eq_some_iff.1 hk
  have hk0 : k < rs.size := hsz ▸ hk1
  have hfk := hel k hk0 hk1
  rw [hk2] at hfk
  rw [Array.getElem?_eq_getElem hk0, hf _ _ hfk]

/-- A number a `ports` file's entry states under key `k`, as it reads it. -/
theorem getD_nat {e : Json} {k ctx : String} {n : ℕ} (h : _root_.Flock.J.nat e k ctx = .ok n) :
    Spot.getD e k = .num ⟨Int.ofNat n, 0⟩ :=
  HeaderRoots.getD_of_getObjVal (FlockComplete.JsonRT.nat_ok h)

/-- **A `ports` file's ports read as its entries say** (`ReadFields`), each at an entry under its name. -/
theorem portsOf_reads {j : Json} {own : Array _root_.Flock.Registered.Port}
    (h : _root_.Flock.Registered.portsOf j = .ok own) :
    ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
      ∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ ReadFields e reg := by
  revert own
  show FlockSoundness.Refine.Ret (_root_.Flock.Registered.portsOf j) fun own =>
    ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧ ∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ ReadFields e reg
  rw [_root_.Flock.Registered.portsOf.eq_def]
  ret_walk [Array.mapM]
  all_goals
    obtain ⟨hall, -⟩ := mapM_ret (by assumption)
        (fun (kv : String × Json) (reg : _root_.Flock.Registered.Port) => reg.name = kv.1 ∧ ReadFields kv.2 reg)
        fun kv => by
      obtain ⟨nm, e⟩ := kv
      dsimp (config := { zeta := false }) only
      by_cases hc : _root_.Flock.J.has e _root_.Flock.Registered.READS_FROM_KEY = true
      · simp (config := { zeta := false }) only [ite_eq_left hc]
        ret_walk_inlining [readsFrom]
          [_root_.Flock.J.str, _root_.Flock.J.nat, _root_.Flock.J.arr, _root_.Flock.unhex, Array.mapM]
        all_goals exact ⟨rfl, getD_nat ‹_root_.Flock.J.nat e "leaves" _ = .ok _›,
          .inr (getD_nat ‹_root_.Flock.J.nat e _root_.Flock.Registered.READS_FROM_KEY _ = .ok _›),
          reads_of ‹_root_.Flock.J.arr e "reads" _ = .ok _› ‹_› fun r n hv => by
            try dsimp only at hv
            split at hv
            · cases hv
            · exact FlockComplete.JsonRT.asNat_ok (map_some_ok hv)⟩
      · simp (config := { zeta := false }) only [ite_eq_right hc, pure_bind]
        ret_walk_inlining [readsFrom]
          [_root_.Flock.J.str, _root_.Flock.J.nat, _root_.Flock.J.arr, _root_.Flock.unhex, Array.mapM]
        all_goals exact ⟨rfl, getD_nat ‹_root_.Flock.J.nat e "leaves" _ = .ok _›, .inl ⟨getD_of_not_has hc, rfl⟩,
          reads_of ‹_root_.Flock.J.arr e "reads" _ = .ok _› ‹_› fun r n hv => by
            try dsimp only at hv
            split at hv
            · cases hv
            · exact FlockComplete.JsonRT.asNat_ok (map_some_ok hv)⟩
    refine ⟨_, by assumption, fun reg hreg => ?_⟩
    obtain ⟨⟨nm, e⟩, hkv, hn, hP⟩ := hall reg hreg
    obtain rfl : reg.name = nm := hn
    exact ⟨e, hkv, hP⟩

/-- **Round 2's rows at a read**: where `rowsOk` accepts a registered read's entry, the position it reads at slot `j`
is the first row of the unit `unit j` names, `leaves / units` rows a unit, and the slot is the part's. -/
theorem pos_of_rowsOk {q : ℕ} {p : Spot.SecondStage.Part} {e : Json} {units : ℕ} {unit : ℕ → Option ℕ}
    (hrows : Spot.SecondStage.rowsOk q p e units unit = true) {reg : _root_.Flock.Registered.Port}
    (hf : ReadFields e reg) {j pos : ℕ} (hj : reg.reads[j]? = some (some pos)) :
    0 < units ∧ ∃ b, unit j = some b ∧ p.holds q j = true ∧ pos = b * (reg.leaves / units) := by
  rw [← _root_.Flock.Registered.Port.readAt_eq, _root_.Flock.Registered.Port.readAt] at hj
  split at hj
  · cases hj
  rename_i hlt
  obtain ⟨hl, hfirst, rs, hrs, hrd⟩ := hf
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

/-- **Round 2's rows at a placed read**: where `placedOk` accepts a registered read's entry, the entry states the
tree's leaves, the position it reads at slot `j` is the row `row j` names, and the slot is the part's. -/
theorem pos_of_placedOk {q : ℕ} {p : Spot.SecondStage.Part} {e : Json} {leaves : ℕ} {row : ℕ → Option ℕ}
    (hpl : Spot.SecondStage.placedOk q p e leaves row = true) {reg : _root_.Flock.Registered.Port}
    (hf : ReadFields e reg) {j pos : ℕ} (hj : reg.reads[j]? = some (some pos)) :
    reg.leaves = leaves ∧ ∃ b, row j = some b ∧ p.holds q j = true ∧ pos = b := by
  rw [← _root_.Flock.Registered.Port.readAt_eq, _root_.Flock.Registered.Port.readAt] at hj
  split at hj
  · cases hj
  rename_i hlt
  obtain ⟨hl, hfirst, rs, hrs, hrd⟩ := hf
  have hk := hrd _ _ hj
  have hsz : j - reg.readsFrom < rs.size := (Array.getElem?_eq_some_iff.1 hk).1
  have hget : rs.getD (j - reg.readsFrom) .null = .num ⟨Int.ofNat pos, 0⟩ := by
    rw [Array.getD_eq_getD_getElem?, hk, Option.getD_some]
  have hfj : reg.readsFrom + (j - reg.readsFrom) = j := by omega
  unfold Spot.SecondStage.placedOk at hpl
  rcases hfirst with ⟨h0, h1⟩ | h0
  · simp only [h0, hl, hrs, nat?_num, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hpl
    obtain ⟨hle, hall⟩ := hpl
    have hx := hall _ hsz
    rw [hget] at hx
    have hfj0 : 0 + (j - reg.readsFrom) = j := by omega
    simp only [nat?_num, hfj0] at hx
    refine ⟨hle, ?_⟩
    cases hb : row j with
    | none => simp [hb] at hx
    | some b =>
      simp only [hb, Bool.and_eq_true, beq_iff_eq] at hx
      exact ⟨b, rfl, hx⟩
  · simp only [h0, hl, hrs, nat?_num, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hpl
    obtain ⟨hle, hall⟩ := hpl
    have hx := hall _ hsz
    rw [hget] at hx
    simp only [nat?_num, hfj] at hx
    refine ⟨hle, ?_⟩
    cases hb : row j with
    | none => simp [hb] at hx
    | some b =>
      simp only [hb, Bool.and_eq_true, beq_iff_eq] at hx
      exact ⟨b, rfl, hx⟩

/-- **A wired slot's row**: the row `wiredRow` gives slot `u` of a part is its unit's row `x` (`rowAt`) of its cell's
`per` rows. -/
theorem wiredRow_some {p : Spot.SecondStage.Part} {ro : Spot.SecondStage.Rows} {u b : ℕ}
    (hb : Spot.SecondStage.wiredRow p ro u = some b) :
    ∃ x, Spot.SecondStage.rowAt ro.runs (p.unit u) = some x ∧ b = p.span u * ro.per + x := by
  unfold Spot.SecondStage.wiredRow at hb
  obtain ⟨x, hx, hb⟩ := Option.bind_eq_some_iff.1 hb
  have hb' : (if x < ro.per then some (p.span u * ro.per + x) else none) = some b := hb
  by_cases hlt : x < ro.per
  · rw [ite_eq_left hlt] at hb'
    exact ⟨x, hx, (Option.some.inj hb').symm⟩
  · rw [ite_eq_right hlt] at hb'
    cases hb'

/-- **A linked slot's row**: the row `linkedRow` gives slot `u` of a part is its unit's row `x` (`rowAt`) of the `per`
rows of its cell's replay block `a`, the sample's entry at the slot's cell. -/
theorem linkedRow_some {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {ro : Spot.SecondStage.Rows}
    {u b : ℕ} (hb : Spot.SecondStage.linkedRow own p ro u = some b) :
    ∃ a x, own.sample[p.span u]? = some a ∧ Spot.SecondStage.rowAt ro.runs (p.unit u) = some x ∧
      b = a * ro.per + x := by
  unfold Spot.SecondStage.linkedRow at hb
  obtain ⟨a, ha, hb⟩ := Option.bind_eq_some_iff.1 hb
  obtain ⟨x, hx, hb⟩ := Option.bind_eq_some_iff.1 hb
  have hb' : (if x < ro.per then some (a * ro.per + x) else none) = some b := hb
  by_cases hlt : x < ro.per
  · rw [ite_eq_left hlt] at hb'
    exact ⟨a, x, ha, hx, (Option.some.inj hb').symm⟩
  · rw [ite_eq_right hlt] at hb'
    cases hb'

/-- **The tree round 2 holds a registered read's rows to**, by the arm the entry's value takes, as its units and the
unit each slot reads:
* a link's: the boundary port's rows, `per` a replay block of the population, each slot reading its unit's row of its
  cell's replay block (`linkedRow`);
* a wire's: its writer's output port's rows, `per` a cell, each slot reading its unit's writer unit's row in its cell
  (`wiredRow`);
* the template's own interior port: the part's slots, each reading its own first row.

A link's and a wire's tree counts rows, one a unit (`placedOk`); the own port's counts slots (`rowsOk`). None for a
registered value, which round 2 holds to its registration alone. -/
def armOf (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) (port : String) (e : Json) :
    Option (ℕ × (ℕ → Option ℕ)) :=
  match Spot.getD e "value", Spot.SecondStage.linkOf own p.template port,
      Spot.SecondStage.wireOf own p.template port with
  | .str _, some _, _ =>
    (Spot.nat? (Spot.getD own.boundary "population")).bind fun m =>
      (Spot.SecondStage.rowsOf own p.template port).map fun r => (m * r.per, Spot.SecondStage.linkedRow own p r)
  | .str _, none, some _ =>
    (Spot.SecondStage.rowsOf own p.template port).map fun r =>
      (own.sample.size * r.per, Spot.SecondStage.wiredRow p r)
  | .str v, none, none =>
    ((Spot.Record.byPort own.interior).find? (·.1 == v)).map fun _ => (own.sample.size * p.n, fun u => some (u - p.lo))
  | _, _, _ => none

/-- **Round 2 holds a registered read's rows to its arm's tree** (`armOf`): where `readOk` accepts an entry whose arm
is the template's own port, no link or wire names the port and `rowsOk` holds of the arm's slots; where its arm is a
link's or a wire's, `placedOk` holds of the arm's rows. -/
theorem arm_of_readOk {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} (h : Spot.SecondStage.readOk own named p port e = true) {units : ℕ} {unit : ℕ → Option ℕ}
    (ha : armOf own p port e = some (units, unit)) :
    (Spot.SecondStage.linkOf own p.template port = none ∧ Spot.SecondStage.wireOf own p.template port = none ∧
        Spot.SecondStage.rowsOk own.sample.size p e units unit = true) ∨
      (((Spot.SecondStage.linkOf own p.template port).isSome = true ∨
          (Spot.SecondStage.wireOf own p.template port).isSome = true) ∧
        Spot.SecondStage.placedOk own.sample.size p e units unit = true) := by
  unfold Spot.SecondStage.readOk at h
  unfold armOf at ha
  revert h ha
  generalize Spot.getD e "value" = val
  generalize Spot.SecondStage.linkOf own p.template port = lk
  generalize Spot.SecondStage.wireOf own p.template port = wr
  intro h ha
  rcases val with _ | _ | _ | v | _ | _ <;> rcases lk with _ | l <;> rcases wr with _ | w <;>
    simp only [Bool.and_eq_true, reduceCtorEq] at h ha
  · obtain ⟨⟨k, x⟩, hkx, hval⟩ := Option.map_eq_some_iff.1 ha
    simp only [Prod.mk.injEq] at hval
    obtain ⟨rfl, rfl⟩ := hval
    simp only [hkx, Bool.and_eq_true] at h
    exact .inl ⟨rfl, rfl, h.2⟩
  · obtain ⟨ro, hro, hval⟩ := Option.map_eq_some_iff.1 ha
    simp only [Prod.mk.injEq] at hval
    obtain ⟨rfl, rfl⟩ := hval
    obtain ⟨-, h⟩ := h
    split at h
    · have hro' := ‹Spot.SecondStage.rowsOf own p.template port = some _›
      rw [hro, Option.some.injEq] at hro'
      subst hro'
      simp only [Bool.and_eq_true] at h
      exact .inr ⟨.inr rfl, h.2⟩
    · cases h
  all_goals
    obtain ⟨m, hm, ha⟩ := Option.bind_eq_some_iff.1 ha
    obtain ⟨ro, hro, hval⟩ := Option.map_eq_some_iff.1 ha
    simp only [Prod.mk.injEq] at hval
    obtain ⟨rfl, rfl⟩ := hval
    obtain ⟨-, h⟩ := h
    split at h
    · have hm' := ‹Spot.nat? (Spot.getD own.boundary "population") = some _›
      rw [hm, Option.some.injEq] at hm'
      subst hm'
      have hro' := ‹Spot.SecondStage.rowsOf own p.template port = some _›
      rw [hro, Option.some.injEq] at hro'
      subst hro'
      simp only [Bool.and_eq_true] at h
      exact .inr ⟨.inl rfl, h.2⟩
    · cases h

/-- **Round 2 holds the template's own port's read to its slots' first rows**: where `readOk` accepts an entry at a
port no link or wire names, `rowsOk` holds of its arm's unit count and slots (`arm_of_readOk`). -/
theorem rowsOk_of_readOk {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} (h : Spot.SecondStage.readOk own named p port e = true)
    (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = none) {units : ℕ} {unit : ℕ → Option ℕ}
    (ha : armOf own p port e = some (units, unit)) :
    Spot.SecondStage.rowsOk own.sample.size p e units unit = true := by
  rcases arm_of_readOk h ha with ⟨-, -, hr⟩ | ⟨hlw, -⟩
  · exact hr
  · rw [hl, hw] at hlw
    rcases hlw with h' | h' <;> simp at h'

/-- **Round 2 holds a linked or wired read to its units' rows**: where `readOk` accepts an entry at a port a link or a
wire names, `placedOk` holds of its arm's rows and the row each slot reads (`arm_of_readOk`). -/
theorem placedOk_of_readOk {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} (h : Spot.SecondStage.readOk own named p port e = true)
    (hlw : (Spot.SecondStage.linkOf own p.template port).isSome = true ∨
      (Spot.SecondStage.wireOf own p.template port).isSome = true) {units : ℕ} {unit : ℕ → Option ℕ}
    (ha : armOf own p port e = some (units, unit)) :
    Spot.SecondStage.placedOk own.sample.size p e units unit = true := by
  rcases arm_of_readOk h ha with ⟨hl, hw, -⟩ | ⟨-, hp⟩
  · rw [hl, hw] at hlw
    rcases hlw with h' | h' <;> simp at h'
  · exact hp

/-- **`armOf` at the template's own port**: an entry whose value names a port of the interior record that no link or
wire of the part's template names reads that port's rows, the part's `q·n` units, each slot its own row. -/
theorem armOf_own {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {port : String} {e : Json} {v : String}
    (hv : Spot.getD e "value" = .str v) (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = none)
    (hf : ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome) :
    armOf own p port e = some (own.sample.size * p.n, fun u => some (u - p.lo)) := by
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hf
  unfold armOf
  rw [hv, hl, hw]
  simp [hx]

/-- **`armOf` at a wired port**: an entry whose value names a port that no link of the part's template names, but a
wire does, reads its writer's output port's rows, `per` a cell as the port's rows give it, each slot its unit's writer
unit's row in its cell. -/
theorem armOf_wire {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {port : String} {e : Json} {v : String}
    {w : Spot.SecondStage.Wire} {ro : Spot.SecondStage.Rows} (hv : Spot.getD e "value" = .str v)
    (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = some w)
    (hro : Spot.SecondStage.rowsOf own p.template port = some ro) :
    armOf own p port e = some (own.sample.size * ro.per, Spot.SecondStage.wiredRow p ro) := by
  unfold armOf
  rw [hv, hl, hw]
  simp [hro]

/-- **`armOf` at a linked port**: an entry whose value names a port that a link of the part's template names reads the
boundary port's rows, `per` a replay block of its population of `M` as the port's rows give it, each slot its unit's
row of its cell's replay block. -/
theorem armOf_link {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {port : String} {e : Json} {v : String}
    {l : Spot.SecondStage.Link} {M : ℕ} {ro : Spot.SecondStage.Rows} (hv : Spot.getD e "value" = .str v)
    (hl : Spot.SecondStage.linkOf own p.template port = some l)
    (hM : Spot.nat? (Spot.getD own.boundary "population") = some M)
    (hro : Spot.SecondStage.rowsOf own p.template port = some ro) :
    armOf own p port e = some (M * ro.per, Spot.SecondStage.linkedRow own p ro) := by
  unfold armOf
  rw [hv, hl]
  simp [hM, hro]

/-- **A slot at a registered read's unit opens the read's position**: where `rowsOk` accepts the read's entry, a slot
not sized, counting the arm's units under the roots and placing its unit at the arm's unit for slot `j`, opens at an
entry of the read's leaf count the position the read reads at `j` (`pos_of_rowsOk`). -/
theorem leafAt_of_rowsOk {q : ℕ} {p : Spot.SecondStage.Part} {e : Json} {units : ℕ} {unit : ℕ → Option ℕ}
    (hrows : Spot.SecondStage.rowsOk q p e units unit = true) {reg : _root_.Flock.Registered.Port}
    (hf : ReadFields e reg) {j pos : ℕ} (hj : reg.reads[j]? = some (some pos)) {roots : List Msg} {s : Slot}
    (hs : s.sized = false) (hu : unitsAt roots s = units)
    (hb : ∀ b, unit j = some b → s.pos + placeOf roots * s.stride = b) {en : Entry} (hl : en.leaves = reg.leaves) :
    leafAt roots s en = pos := by
  obtain ⟨hpos, b, hb', -, rfl⟩ := pos_of_rowsOk hrows hf hj
  rw [leafAt, hu, ite_eq_right (Nat.pos_iff_ne_zero.1 hpos), leavesAt, ite_eq_right (by rw [hs]; decide), hl,
    hb b hb']

/-- **A slot at a placed read's row opens the read's position**: where `placedOk` accepts the read's entry, a slot not
sized, counting the arm's rows under the roots and placing itself at the arm's row for slot `j`, opens at an entry of
the read's leaf count the position the read reads at `j` (`pos_of_placedOk`): one leaf a row. A slot of a tree of no
rows opens its own position, so it takes no stride there (`hz`). -/
theorem leafAt_of_placedOk {q : ℕ} {p : Spot.SecondStage.Part} {e : Json} {leaves : ℕ} {row : ℕ → Option ℕ}
    (hpl : Spot.SecondStage.placedOk q p e leaves row = true) {reg : _root_.Flock.Registered.Port}
    (hf : ReadFields e reg) {j pos : ℕ} (hj : reg.reads[j]? = some (some pos)) {roots : List Msg} {s : Slot}
    (hs : s.sized = false) (hu : unitsAt roots s = leaves) (hz : leaves = 0 → s.stride = 0)
    (hb : ∀ b, row j = some b → s.pos + placeOf roots * s.stride = b) {en : Entry} (hl : en.leaves = reg.leaves) :
    leafAt roots s en = pos := by
  obtain ⟨hle, b, hb', -, hpb⟩ := pos_of_placedOk hpl hf hj
  have hP := hb b hb'
  rw [hpb, leafAt, hu]
  by_cases h0 : leaves = 0
  · rw [ite_eq_left h0]
    rw [hz h0, Nat.mul_zero, Nat.add_zero] at hP
    exact hP
  · rw [ite_eq_right h0, leavesAt, ite_eq_right (by rw [hs]; decide), hl, hle, hP,
      Nat.div_self (Nat.pos_of_ne_zero h0), Nat.mul_one]

/-- **A drawn session's unit labels**: where a public file whose header round 2's `populationOk` accepts for part `p`
of a sample of `q` blocks loads `pop`, and the draw `d` gives `pub`, `pub` labels its instances with the slots of
`p` its draw proves. -/
theorem unitIndices_session {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {f : ByteArray}
    {pop pub : _root_.Flock.Public} {hdr d : Json} (hpop : _root_.Flock.HmOut.loadPublic tags c f = .ok pop)
    (hH : headerOf f = some hdr) (hd : _root_.Flock.HmOut.drawn c pop d = .ok pub) {q : ℕ}
    {p : Spot.SecondStage.Part} (hok : Spot.SecondStage.populationOk q p hdr = true) :
    ∃ D, _root_.Flock.Draw.UnitDraw.ofJson d = .ok D ∧ _root_.Flock.Registered.unitIndices pub =
      .ok (D.proved.map fun v => ((Array.range (q * p.n)).map (p.lo + ·)).getD v 0) := by
  obtain ⟨h₀, hh₀, hhdr, -, m, hm⟩ :=
    HeaderRoots.loadPublic_raw (FlockSoundness.Discharge.Hidden.loadPublic_nil hpop)
  rw [hH, Option.some.injEq] at hh₀
  subst hh₀
  have hmw : m.WF := by
    have hw := FlockDeploy.wf_norm hdr
    rw [← hhdr, hm] at hw
    cases hw with
    | obj hmw _ => exact hmw
  obtain ⟨D, popIdx, hD, hpi, hpub⟩ := unitIndices_drawn hmw hm hd
  rw [unitIndices_pop (HeaderRoots.wf_of_headerOf hH) hhdr hok] at hpi
  cases hpi
  exact ⟨D, hD, hpub⟩

/-- **A population round 2's `populationOk` accepts has its part's instances**: the instance count Flock reads off
the normalized header is `q·n`. -/
theorem n_pop {j : Json} (hw : Claims.Json.Wf j) {k : ℕ}
    (hn : _root_.Flock.J.nat (_root_.Flock.norm j) "instances" "public file" = .ok k) {q : ℕ}
    {p : Spot.SecondStage.Part} (hok : Spot.SecondStage.populationOk q p j = true) : k = q * p.n := by
  unfold Spot.SecondStage.populationOk at hok
  split at hok
  · rename_i _ m _ hm
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hok
    obtain ⟨⟨hmq, -⟩, -⟩ := hok
    obtain ⟨w, hwj, hew, -⟩ := HeaderRoots.getObjVal_equiv (FlockDeploy.wf_norm j) hw
      (FlockDeploy.jsonEquiv_norm j hw) (FlockComplete.JsonRT.nat_ok hn)
    rw [Depths.num_of_jsonEquiv hew] at hwj
    rw [HeaderRoots.getD_of_getObjVal hwj, nat?_num, Option.some.injEq] at hm
    rw [hm]
    exact hmq
  · cases hok

/-- **At a port a link names, round 2 reads the linked boundary port**, under the statement's roots, whatever the
value. -/
theorem readsInterior_linked {own : Spot.SecondStage.Own} {t port v : String} {l : Spot.SecondStage.Link}
    (hl : Spot.SecondStage.linkOf own t port = some l) :
    readsInterior own t port v = false ∧ readKey own t port v = .port l.value := by
  simp [readsInterior, readKey, hl]

/-- **At a port no link names, a value the interior record has a port for is read off that port**, under the replay
block's roots. -/
theorem readsInterior_held {own : Spot.SecondStage.Own} {t port v : String}
    (hl : Spot.SecondStage.linkOf own t port = none)
    (hv : ((Spot.Record.byPort own.interior).find? (·.1 == v)).isSome = true) :
    readsInterior own t port v = true ∧ readKey own t port v = .port v := by
  simp [readsInterior, readKey, hl, hv]

/-- **A wired read round 2 accepts is read off the interior record**, at its writer's output port
(`SecondStage.readOk_wired`), under the replay block's roots. -/
theorem readsInterior_wired {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port v : String}
    {e : Json} {w : Spot.SecondStage.Wire} (hl : Spot.SecondStage.linkOf own p.template port = none)
    (hw : Spot.SecondStage.wireOf own p.template port = some w)
    (hr : Spot.SecondStage.readOk own named p port e = true) (hv : Spot.getD e "value" = .str v) :
    readsInterior own p.template port v = true ∧ readKey own p.template port v = .port v := by
  obtain ⟨x, -, -, he, hm, -⟩ := Spot.SecondStage.readOk_wired hl hw hr
  rw [hv] at he
  injection he with he
  subst he
  exact readsInterior_held hl (Array.find?_isSome.2 ⟨_, hm, by simp⟩)

variable (h : Held)

/-- **A checked unit's registered read as round 2 reads it**: in an accepted run, each registered read of a checked
unit's session has an entry `e` of the statement's registered reads under its name, which round 2's `readOk` accepts
for the unit's template's part and which states the read's leaves, first slot and positions (`ReadFields`). -/
theorem regEntry {x m₁ m₂ : Msg} {s} {c₀ c₁ : FlockDeploy.Coin} {u : ℕ} {ins : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁))
    (hset : FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts))
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : ins.registered = some own) (hreg : reg ∈ own) :
    ∃ st rec draw e, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      Spot.SecondStage.readOk (ownOf h st rec draw c₀) (Spot.SecondStage.byTemplate rec)
        (partAt h (ownOf h st rec draw c₀).sample.size (templateOf h.ms u)) reg.name e = true ∧
      ReadFields e reg := by
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins (ins_of_setupOf hw hset) hown
  obtain ⟨-, -, hSr, -⟩ := stmtJsonOf_fields h hS
  have hSj : S.registered = jR := by
    have h4 := hSr bR hbR
    rw [hjR] at h4
    exact (Option.some.inj h4).symm
  obtain ⟨kvs, hports, hrds⟩ := portsOf_reads (portsOf_of_ofJson hoj)
  obtain ⟨e, hm, hrf⟩ := hrds reg hreg
  obtain ⟨-, -, -, -, -, -, hreads, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hents : Spot.entries (Spot.getD S.registered "ports") = kvs.toArray := by
    simp only [hSj, getD_of_ok hports, Spot.entries]
  rw [hents] at hreads
  exact ⟨st, rec, draw, e, hst, hrec, hdraw, Array.all_eq_true_iff_forall_mem.1 hreads _ hm, hrf⟩

/-- **A statement's roots are its record's**: the statement `parse` gives decodes, at the decoded record's roots. -/
theorem parse_roots {x : Msg} {s : AuditStatement} (hs : parse h x = some s) :
    ∃ st, decode x = some st ∧ s.roots = st.roots := by
  unfold parse at hs
  cases hd : decode x with
  | none => simp [hd] at hs
  | some st => rw [hd] at hs; cases hs; exact ⟨st, rfl, rfl⟩

/-- **The entry round 2 reads a registered read off**, in a run whose statement, round-1 record and second stage's
draw are `st`, `rec` and `draw`: `readKey` at that run's records. -/
theorem readKeyAt_of {x m₁ : Msg} {c₀ c₁ : FlockDeploy.Coin} {st : Roots} {rec draw : Json}
    (hx : decode x = some st) (hrec : jsonMsg m₁ = some rec) (hdraw : secondDraw h c₀ c₁ = some draw) (t : ℕ)
    (port v : String) :
    readKeyAt h x m₁ c₀ c₁ t port v = readKey (ownOf h st rec draw c₀) (h.names.getD t "") port v := by
  simp only [readKeyAt, hx, hrec, hdraw]

end SpotFlock.ReadsAt

namespace SpotFlock.CheckedRun

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- The part of the second stage's sample that a checked unit's template's slots are in. -/
abbrev drawnPart (r : CheckedRun) : Spot.SecondStage.Part :=
  partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)

/-- **A checked unit's one-table session labels its instances with their slots**: in an accepted run, the session's
sample (`drawnAt`'s) proves units of its population, which round 2 holds to the `q·n` slots of the unit's template's
part (`drawnPart`, `ReadsAt.n_pop`), and the session's public file labels the instance the sample's `i`-th proved
unit is with that unit's slot, counted from the part's first (`ReadsAt.unitIndices_session`). -/
theorem labelAt {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) :
    ∃ D : _root_.Flock.Draw.UnitDraw, drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList ∧
      r.sts[0]!.pub.n = D.proved.size ∧
      (∀ v ∈ D.proved, v < (h.reg.sample (src r.c₀)).size * (r.drawnPart h).n) ∧
      _root_.Flock.Registered.unitIndices r.sts[0]!.pub = .ok (D.proved.map fun v =>
        ((Array.range ((h.reg.sample (src r.c₀)).size * (r.drawnPart h).n)).map ((r.drawnPart h).lo + ·)).getD v
          0) := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  have hsc := sessionScoped h r.x r.m₁ (unitStmt h r.m₂ r.u) r.c₀ r.c₁ _ hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨-, -, -, hpopOk, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hwi := ins_of_setupOf hw hset
  obtain ⟨-, hSH, -, -⟩ := stmtJsonOf_fields h hS
  have hH : headerOf r.ins.read.I.publicFile = some S.header := by
    rw [(ins_reads_public hwi).2]
    exact hSH
  obtain ⟨b, hb, hjo⟩ := stmtJsonOf_draw h hS
  obtain ⟨draw', hdm, hdr⟩ := FlockDeploy.Raw.ins_draw hwi
  rw [hb, Option.mapM_some] at hdm
  obtain ⟨dr, hdr', rfl⟩ := Depths.someMap_ok hdm
  rw [hjo] at hdr'
  cases hdr'
  have hrd : r.ins.read.draw = some (_root_.Flock.norm S.draw) := hdr
  obtain ⟨hcheck, hh⟩ := hsc r.ins spec r.sts hset
  obtain ⟨hRu, -⟩ := (_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hcheck
  have hpi : r.ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  obtain ⟨hins, hsetup⟩ := FlockDeploy.setupOf_some hset
  have hset' := FlockDeploy.Ins.setup_read hsetup
  unfold _root_.Flock.Readable.Read.setup at hset'
  rw [hpi] at hset'
  obtain ⟨hpos, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset'
  obtain ⟨d₀, hd, hdraw₀⟩ := hall 0 hpos
  rw [hdraw₀ hc1] at hd
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd
  rw [_root_.Flock.Readable.tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  obtain ⟨pop, hpop, -, hsome⟩ := (FlockSoundness.Discharge.Hidden.setupHidden_spec' hd).pub
  obtain ⟨D, hD, hpopn, hn, -, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec (hsome _ hrd)
  obtain ⟨D', hD', hidx⟩ := ReadsAt.unitIndices_session hpop hH (hsome _ hrd) hpopOk
  rw [hD] at hD'
  cases hD'
  obtain ⟨j, hj, hnj, -⟩ := Depths.loadPublic_counts (FlockSoundness.Discharge.Hidden.loadPublic_nil hpop)
  rw [hH, Option.some.injEq] at hj
  subst hj
  have hq := ReadsAt.n_pop (HeaderRoots.wf_of_headerOf hH) hnj hpopOk
  have hread : FlockDeploy.readOf true (unitStmt h r.m₂ r.u) = some r.ins.read := by
    simp [FlockDeploy.readOf, hins]
  refine ⟨D, ?_, ?_, fun v hv => ?_, ?_⟩
  · show drawnOf ((FlockDeploy.readOf true (unitStmt h r.m₂ r.u)).bind (·.draw)) = _
    rw [hread]
    simp only [Option.bind_some, hrd, drawnOf, hD]
  · rw [getElem!_pos r.sts 0 hpos]
    exact hn
  · have hlt := FlockSoundness.Discharge.Layout.proved_lt_population hD v hv
    rw [hpopn, hq] at hlt
    exact hlt
  · rw [getElem!_pos r.sts 0 hpos]
    exact hidx

/-- **A registered read's position at the unit's instance is the one it reads at the unit's slot**: in an accepted
run whose checked unit's session has one table and counts the unit's instance (`Counted`), a registered read `reg` of
the session, at its port `p` of the circuit, reads at the slot of the unit's template's part that the instance is
(`labelAt`) the position the session header gives the instance's row of `p`'s table (`opens_of_setup`). -/
theorem readPosAt {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hcnt : r.Counted h)
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : r.ins.registered = some own) (hreg : reg ∈ own) {p : ℕ}
    (hp : (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some p)
    {t : _root_.Flock.Tables} {positions : Array ℕ} {paths : Array ByteArray} (ht : r.sts[0]!.pub.tables = some t)
    (hent : _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths)) :
    ∃ D : _root_.Flock.Draw.UnitDraw, drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList ∧
      ∃ hn : r.n h < D.proved.size, D.proved[r.n h] < (h.reg.sample (src r.c₀)).size * (r.drawnPart h).n ∧
        reg.reads[(r.drawnPart h).lo + D.proved[r.n h]]? =
          some (some (positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size p (r.n h)) 0)) := by
  obtain ⟨D, hdr, hn, hlt, hidx⟩ := labelAt h hr hc1
  have hmem : r.sts[0]! ∈ r.sts := by
    rw [getElem!_pos r.sts 0 (sts_pos hr)]
    exact Array.getElem_mem _
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, hset⟩ := hr
  obtain ⟨tb, idx, htb, hidx', hall⟩ := opens_of_setup hset hown _ hmem
  obtain ⟨p', positions', paths', hp', hent', hinst⟩ := hall reg hreg
  rw [ht, Option.some.injEq] at htb
  subst htb
  rw [hidx, Except.ok.injEq] at hidx'
  subst hidx'
  rw [hp, Option.some.injEq] at hp'
  subst hp'
  rw [hent, Except.ok.injEq, Prod.mk.injEq] at hent'
  obtain ⟨rfl, rfl⟩ := hent'
  have hnD : r.n h < D.proved.size := by
    rw [← hn]
    exact hcnt
  obtain ⟨-, hrd, -⟩ := hinst (r.n h) hcnt
  have hv := hlt _ (Array.getElem_mem hnD)
  refine ⟨D, hdr, hnD, hv, ?_⟩
  rw [← hrd]
  congr 1
  simp [hnD, hv]

/-- **A registered read's slot opens the session header's position for the instance's row**: in an accepted run whose
checked unit's session has one table and counts the unit's instance, a registered read `reg` at the circuit's port
`p`, and a slot not sized that, under the roots, at each entry round 2 accepts for the read (`readOk`), counts the
units of the tree round 2 holds the read to (`armOf`), with no stride where the tree has none, and places its unit at
that tree's unit for the unit's slot (`harm`), the slot's leaf at an entry of `reg`'s leaf count is the position the
header gives the instance's row of `p`'s table (`readPosAt`; `ReadsAt.leafAt_of_rowsOk` at the template's own port,
`ReadsAt.leafAt_of_placedOk` at a link's or a wire's). -/
theorem leafAt_reg {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hcnt : r.Counted h)
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : r.ins.registered = some own) (hreg : reg ∈ own) {p : ℕ}
    (hp : (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some p)
    {roots : List Msg} {s : Slot} (hs : s.sized = false)
    (harm : ∀ st rec draw e (D : _root_.Flock.Draw.UnitDraw) (hn : r.n h < D.proved.size), decode r.x = some st →
      jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg →
      ∃ units unit, ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e = some (units, unit) ∧
        unitsAt roots s = units ∧ (units = 0 → s.stride = 0) ∧
        ∀ b, unit ((r.drawnPart h).lo + D.proved[r.n h]) = some b → s.pos + placeOf roots * s.stride = b)
    {t : _root_.Flock.Tables} {positions : Array ℕ} {paths : Array ByteArray} (ht : r.sts[0]!.pub.tables = some t)
    (hent : _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths)) {en : Entry}
    (hl : en.leaves = reg.leaves) :
    leafAt roots s en = positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size p (r.n h)) 0 := by
  obtain ⟨D, hdr, hn, -, hj⟩ := readPosAt h hr hc1 hcnt hown hreg hp ht hent
  obtain ⟨_, _, _, hx, _, _, _, hok, hu, _, _, _, _, hset⟩ := hr
  obtain ⟨st, rec, draw, e, hst, hrec, hdraw, hrd, hf⟩ := ReadsAt.regEntry h hx hok hu hset hown hreg
  obtain ⟨units, unit, ha, hun, hz, hb⟩ := harm st rec draw e D hn hst hrec hdraw hdr hrd hf
  rcases ReadsAt.arm_of_readOk hrd ha with ⟨-, -, hrows⟩ | ⟨-, hpl⟩
  · exact ReadsAt.leafAt_of_rowsOk hrows hf hj hs hun hb hl
  · exact ReadsAt.leafAt_of_placedOk hpl hf hj hs hun hz hb hl

/-- **The checked unit is the proof block its instance's slot names**: the unit's instance in its session (`rankAt`)
is the first slot of the session's sample whose proof block (`RegTwo.blockAt`, over the replay sample round 1's record
lists) is the unit, so where the session counts it, that slot of the unit's template's part is the unit. -/
theorem blockAt_rank {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) {D : _root_.Flock.Draw.UnitDraw}
    (hdr : drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList) (hn : r.n h < D.proved.size) :
    h.reg.blockAt (h.reg.sample (src r.c₀)) ((r.drawnPart h).lo + D.proved[r.n h]) = r.u := by
  obtain ⟨_, rec, _, _, hrec, hbl, _⟩ := drawnAt h hr hc1
  have key : ∀ k (hk : k < D.proved.toList.length), k = D.proved.toList.findIdx
      (fun p => h.reg.blockAt (h.reg.sample (src r.c₀)) ((r.drawnPart h).lo + p) == r.u) →
      h.reg.blockAt (h.reg.sample (src r.c₀)) ((r.drawnPart h).lo + D.proved.toList[k]) = r.u := by
    rintro k hk rfl
    exact beq_iff_eq.1 (List.findIdx_getElem (w := hk))
  have hfi : r.n h = D.proved.toList.findIdx
      (fun p => h.reg.blockAt (h.reg.sample (src r.c₀)) ((r.drawnPart h).lo + p) == r.u) := by
    show (drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw))).findIdx (fun p =>
        h.reg.blockAt ((((jsonMsg r.m₁).map blocksOf).getD []).toArray)
          (Properties.Spot.cellOff ((((jsonMsg r.m₁).map blocksOf).getD []).toArray).size h.reg.ns
            (templateOf h.ms r.u) + p) == r.u) = _
    rw [hdr, hrec, Option.map_some, Option.getD_some, hbl, Array.toArray_toList]
    rfl
  have hk := key (r.n h) (by rw [Array.length_toList]; exact hn) hfi
  rwa [Array.getElem_toList] at hk

/-- **The checked unit's slot is its position in its replay block's cell**: where the first stage samples unit slots
(`FirstSlots`), the slot `D.proved[n] = i·n_t + p` of the unit's template's part that its counted instance is names
the sample's `i`-th replay block, which is the unit's (`r.ρ`), and the unit is that block's `p`-th proof block of its
template, on whose `n_t` positions the registration is one to one (`regMatchesTwo`). -/
theorem slot_cell {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hS : FirstSlots h)
    {D : _root_.Flock.Draw.UnitDraw}
    (hdr : drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList) (hn : r.n h < D.proved.size)
    (hlt : D.proved[r.n h] < (h.reg.sample (src r.c₀)).size * (r.drawnPart h).n) :
    ∃ i p, i < (h.reg.sample (src r.c₀)).size ∧ p < (r.drawnPart h).n ∧ D.proved[r.n h] = i * (r.drawnPart h).n + p ∧
      (h.reg.sample (src r.c₀)).getD i 0 = r.ρ ∧ h.reg.blockOf r.ρ (templateOf h.ms r.u) p = r.u ∧
      Set.InjOn (h.reg.blockOf r.ρ (templateOf h.ms r.u)) (Set.Iio (r.drawnPart h).n) := by
  have hb := blockAt_rank h hr hc1 hdr hn
  obtain ⟨Pt, d, _, hx, hP, h2, hin, _, _, _, _, hρ, _, _⟩ := hr
  have hpos : 0 < (r.drawnPart h).n := by
    rcases Nat.eq_zero_or_pos (r.drawnPart h).n with h0 | h0
    · rw [h0, Nat.mul_zero] at hlt
      exact absurd hlt (Nat.not_lt_zero _)
    · exact h0
  have ht : templateOf h.ms r.u < h.reg.ns.size := by
    by_contra hge
    have h0 : (r.drawnPart h).n = 0 := by
      show h.reg.ns.getD (templateOf h.ms r.u) 0 = 0
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none]
    omega
  obtain ⟨i, p, hi, hp, hdec⟩ : ∃ i p, i < (h.reg.sample (src r.c₀)).size ∧ p < (r.drawnPart h).n ∧
      D.proved[r.n h] = i * (r.drawnPart h).n + p := by
    refine ⟨D.proved[r.n h] / (r.drawnPart h).n, D.proved[r.n h] % (r.drawnPart h).n,
      (Nat.div_lt_iff_lt_mul hpos).2 hlt, Nat.mod_lt _ hpos, ?_⟩
    have hdm := Nat.div_add_mod (D.proved[r.n h]) (r.drawnPart h).n
    rw [Nat.mul_comm] at hdm
    exact hdm.symm
  have hu : r.u = h.reg.blockOf ((h.reg.sample (src r.c₀)).getD i 0) (templateOf h.ms r.u) p := by
    rw [hdec, ← Nat.add_assoc] at hb
    exact hb.symm.trans (h.reg.blockAt_cell (h.reg.sample (src r.c₀)) ht hi hp)
  have hreg : reg h r.x = some h.reg := by
    unfold reg
    unfold parse at hx
    cases hdx : decode r.x with
    | none => simp [hdx] at hx
    | some st => rfl
  obtain ⟨g, l, hg, hl, _, _, hT, _, hcl⟩ := regMatchesTwo h r.x r.s Pt d hx hP h2 hin
  obtain rfl : g = h.reg := Option.some.inj (hg.symm.trans hreg)
  obtain ⟨b, hbl, hbu⟩ := hS l hl (src r.c₀) i hi
  rcases hcl _ b hbl with ⟨_, hcell⟩ | ⟨hnone, _⟩
  · obtain ⟨hinj, hceq⟩ := hcell (templateOf h.ms r.u) (hT ▸ ht)
    have hmem : r.u ∈ cell Pt d.part ((h.reg.sample (src r.c₀)).getD i 0) (templateOf h.ms r.u) := by
      rw [hceq]
      exact Finset.mem_image.2 ⟨p, Finset.mem_range.2 hp, hu.symm⟩
    simp only [cell, Finset.mem_filter] at hmem
    have hρi : (h.reg.sample (src r.c₀)).getD i 0 = r.ρ := Option.some.inj (hmem.2.1.symm.trans hρ)
    refine ⟨i, p, hi, hp, hdec, hρi, ?_, ?_⟩
    · rw [← hρi]
      exact hu.symm
    · rw [← hρi]
      exact hinj
  · simp [hbu] at hnone

/-- **The unit's replay block's roots place it at its cell**: where the sample's `i`-th replay block is the unit's,
`r.ρ`, the replay block's roots (`irootsOf`) state place `i` among the sample's `q` replay blocks, since round 1's
record lists the sample (`drawnAt`), each block once (`selectSecond_sample`). -/
theorem placeOf_cell {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) {i : ℕ}
    (hi : i < (h.reg.sample (src r.c₀)).size) (hρi : (h.reg.sample (src r.c₀)).getD i 0 = r.ρ) :
    placeOf (irootsOf r.m₁ r.ρ) = i ∧ qOf (irootsOf r.m₁ r.ρ) = (h.reg.sample (src r.c₀)).size := by
  obtain ⟨draw, rec, _, hdraw, hrec, hbl, _⟩ := drawnAt h hr hc1
  obtain ⟨_, hsel⟩ := secondDraw_ok h hdraw
  have hnd := (selectSecond_sample hsel).2
  have hi' : i < (h.reg.sample (src r.c₀)).toList.length := by
    rw [Array.length_toList]
    exact hi
  have hρ' : r.ρ = (h.reg.sample (src r.c₀)).toList[i] := by
    rw [← hρi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some, Array.getElem_toList]
  refine ⟨?_, ?_⟩
  · rw [placeOf_irootsOf hrec, hbl, hρ', List.Nodup.idxOf_getElem hnd i hi']
  · rw [qOf_irootsOf hrec, hbl, Array.length_toList]

/-- Slot `i·n + k` of a part, `k < n`, is in its cell `i` (`Part.span`). -/
theorem span_cell {p : Spot.SecondStage.Part} {i k : ℕ} (hk : k < p.n) : p.span (p.lo + (i * p.n + k)) = i := by
  rw [Spot.SecondStage.Part.span, Nat.add_sub_cancel_left, Nat.add_comm, Nat.add_mul_div_right _ _ (by omega),
    Nat.div_eq_of_lt hk, Nat.zero_add]

/-- Slot `i·n + k` of a part, `k < n`, is its cell's unit `k` (`Part.unit`). -/
theorem unit_cell {p : Spot.SecondStage.Part} {i k : ℕ} (hk : k < p.n) : p.unit (p.lo + (i * p.n + k)) = k := by
  rw [Spot.SecondStage.Part.unit, Nat.add_sub_cancel_left, Nat.add_comm, Nat.add_mul_mod_self_right,
    Nat.mod_eq_of_lt hk]

/-- **The own arm's slot**: a slot with no units of its own, stride `n_t` and position `p` where the unit is its replay
block's `p`-th proof block of its template counts, under that replay block's roots (`irootsOf`), the `q·n_t` units of
the template's part, and places the unit's counted instance's slot (`slot_cell`, `placeOf_cell`). -/
theorem ownSlot {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hS : FirstSlots h)
    {D : _root_.Flock.Draw.UnitDraw}
    (hdr : drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList) (hn : r.n h < D.proved.size)
    (hlt : D.proved[r.n h] < (h.reg.sample (src r.c₀)).size * (r.drawnPart h).n) {s : Slot} (hs0 : s.units = 0)
    (hst : s.stride = (r.drawnPart h).n) (hpos : s.pos < (r.drawnPart h).n)
    (hu : h.reg.blockOf r.ρ (templateOf h.ms r.u) s.pos = r.u) :
    unitsAt (irootsOf r.m₁ r.ρ) s = (h.reg.sample (src r.c₀)).size * (r.drawnPart h).n ∧
      s.pos + placeOf (irootsOf r.m₁ r.ρ) * s.stride = D.proved[r.n h] := by
  obtain ⟨i, p, hi, hp, hdec, hρi, hup, hinj⟩ := slot_cell h hr hc1 hS hdr hn hlt
  have hps : p = s.pos := hinj (Set.mem_Iio.2 hp) (Set.mem_Iio.2 hpos) (hup.trans hu.symm)
  obtain ⟨hpl, hq⟩ := placeOf_cell h hr hc1 hi hρi
  refine ⟨?_, ?_⟩
  · rw [unitsAt, hq, hs0, hst, Nat.zero_add]
  · rw [hpl, hst, hdec, ← hps, Nat.add_comm]

/-- **The wire arm's slot**: a slot with no units of its own and stride `n_w` counts, under the unit's replay block's
roots, the `q·n_w` rows of the writer's output port, `n_w` a cell, and places at the unit's counted instance's slot
the row at its own position in the same cell (`span_cell`). -/
theorem wireSlot {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hS : FirstSlots h)
    {D : _root_.Flock.Draw.UnitDraw}
    (hdr : drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList) (hn : r.n h < D.proved.size)
    (hlt : D.proved[r.n h] < (h.reg.sample (src r.c₀)).size * (r.drawnPart h).n) {s : Slot} {nw : ℕ}
    (hs0 : s.units = 0) (hst : s.stride = nw) :
    unitsAt (irootsOf r.m₁ r.ρ) s = (h.reg.sample (src r.c₀)).size * nw ∧
      s.pos + placeOf (irootsOf r.m₁ r.ρ) * s.stride =
        (r.drawnPart h).span ((r.drawnPart h).lo + D.proved[r.n h]) * nw + s.pos := by
  obtain ⟨i, p, hi, hp, hdec, hρi, _, _⟩ := slot_cell h hr hc1 hS hdr hn hlt
  obtain ⟨hpl, hq⟩ := placeOf_cell h hr hc1 hi hρi
  refine ⟨?_, ?_⟩
  · rw [unitsAt, hq, hs0, hst, Nat.zero_add]
  · rw [hpl, hst, hdec, span_cell hp]
    exact Nat.add_comm _ _

/-- **`leafAt_reg`'s arm fact at the template's own port**: where each entry round 2 accepts for the read takes the own
arm in the accepted run (`ReadsAt.armOf_own`), a slot with no units of its own, stride `n_t` and position the unit's in
its replay block's cell has it under that replay block's roots (`ownSlot`, at the slot `labelAt` bounds). -/
theorem harm_own {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hS : FirstSlots h)
    {reg : _root_.Flock.Registered.Port} {s : Slot} (hs0 : s.units = 0) (hst : s.stride = (r.drawnPart h).n)
    (hpos : s.pos < (r.drawnPart h).n) (hu : h.reg.blockOf r.ρ (templateOf h.ms r.u) s.pos = r.u)
    (harm : ∀ st rec draw e, decode r.x = some st → jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg → ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e =
        some ((h.reg.sample (src r.c₀)).size * (r.drawnPart h).n, fun u => some (u - (r.drawnPart h).lo))) :
    ∀ st rec draw e (D : _root_.Flock.Draw.UnitDraw) (hn : r.n h < D.proved.size), decode r.x = some st →
      jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg →
      ∃ units unit, ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e = some (units, unit) ∧
        unitsAt (irootsOf r.m₁ r.ρ) s = units ∧ (units = 0 → s.stride = 0) ∧
        ∀ b, unit ((r.drawnPart h).lo + D.proved[r.n h]) = some b →
          s.pos + placeOf (irootsOf r.m₁ r.ρ) * s.stride = b := by
  intro st rec draw e D hn hx hrec hdraw hdr hok hf
  obtain ⟨D', hdr', _, hlt', _⟩ := labelAt h hr hc1
  have hDD : D'.proved = D.proved := Array.toList_inj.1 (hdr'.symm.trans hdr)
  rw [hDD] at hlt'
  have hlt := hlt' _ (Array.getElem_mem hn)
  obtain ⟨hU, hP⟩ := ownSlot h hr hc1 hS hdr hn hlt hs0 hst hpos hu
  refine ⟨_, _, harm st rec draw e hx hrec hdraw hok hf, hU, fun h0 => ?_, fun b hb => ?_⟩
  · rw [h0] at hlt
    exact absurd hlt (Nat.not_lt_zero _)
  · rw [hP]
    simp only [Nat.add_sub_cancel_left, Option.some.injEq] at hb
    exact hb

/-- **`leafAt_reg`'s arm fact at a wired port**: where each entry round 2 accepts for the read takes the wire arm in the
accepted run, over the port's rows `ro` (`ReadsAt.armOf_wire`), a slot with no units of its own, stride the rows' `per`
and position the row `ro` gives the unit's position `k` in its replay block's cell has it under the unit's replay
block's roots (`wireSlot`, `unit_cell`): round 2 holds a wired read to its unit's writer unit's row in its cell. -/
theorem harm_wire {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hS : FirstSlots h)
    {reg : _root_.Flock.Registered.Port} {s : Slot} {ro : Spot.SecondStage.Rows} {k : ℕ} (hs0 : s.units = 0)
    (hst : s.stride = ro.per) (hk : k < (r.drawnPart h).n)
    (hu : h.reg.blockOf r.ρ (templateOf h.ms r.u) k = r.u)
    (hrow : Spot.SecondStage.rowAt ro.runs k = some s.pos)
    (harm : ∀ st rec draw e, decode r.x = some st → jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg → ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e =
        some ((h.reg.sample (src r.c₀)).size * ro.per, Spot.SecondStage.wiredRow (r.drawnPart h) ro)) :
    ∀ st rec draw e (D : _root_.Flock.Draw.UnitDraw) (hn : r.n h < D.proved.size), decode r.x = some st →
      jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg →
      ∃ units unit, ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e = some (units, unit) ∧
        unitsAt (irootsOf r.m₁ r.ρ) s = units ∧ (units = 0 → s.stride = 0) ∧
        ∀ b, unit ((r.drawnPart h).lo + D.proved[r.n h]) = some b →
          s.pos + placeOf (irootsOf r.m₁ r.ρ) * s.stride = b := by
  intro st rec draw e D hn hx hrec hdraw hdr hok hf
  obtain ⟨D', hdr', _, hlt', _⟩ := labelAt h hr hc1
  have hDD : D'.proved = D.proved := Array.toList_inj.1 (hdr'.symm.trans hdr)
  rw [hDD] at hlt'
  have hlt := hlt' _ (Array.getElem_mem hn)
  obtain ⟨i, p, _, hp, hdec, _, hup, hinj⟩ := slot_cell h hr hc1 hS hdr hn hlt
  have hpk : p = k := hinj (Set.mem_Iio.2 hp) (Set.mem_Iio.2 hk) (hup.trans hu.symm)
  obtain ⟨hU, hP⟩ := wireSlot h hr hc1 hS hdr hn hlt hs0 hst
  refine ⟨_, _, harm st rec draw e hx hrec hdraw hok hf, hU, fun h0 => ?_, fun b hb => ?_⟩
  · rw [hst]
    rcases Nat.mul_eq_zero.1 h0 with hq | hper
    · rw [hq, Nat.zero_mul] at hlt
      exact absurd hlt (Nat.not_lt_zero _)
    · exact hper
  · obtain ⟨y, hy, rfl⟩ := ReadsAt.wiredRow_some hb
    rw [hdec, unit_cell hp, hpk, hrow, Option.some.injEq] at hy
    rw [hP, hy]

/-- **`leafAt_reg`'s arm fact at a linked port**: where each entry round 2 accepts for the read takes the link arm in
the accepted run, over a boundary population of `M` and the port's rows `ro` (`ReadsAt.armOf_link`), a slot of the
boundary port's `M·per` rows, stride `0` and position the row `ro` gives the unit's position `k` in its replay block's
cell, of the unit's replay block's rows, has it under any roots: round 2 holds a linked read to its unit's row of its
cell's replay block, and a slot of the sample's cell `i` reads the sample's `i`-th entry (`span_cell`, `unit_cell`,
`slot_cell`). -/
theorem harm_link {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hS : FirstSlots h)
    {reg : _root_.Flock.Registered.Port} {roots : List Msg} {s : Slot} {M : ℕ} {ro : Spot.SecondStage.Rows}
    {k x : ℕ} (hsu : s.units = M * ro.per) (hst : s.stride = 0) (hk : k < (r.drawnPart h).n)
    (hu : h.reg.blockOf r.ρ (templateOf h.ms r.u) k = r.u) (hrow : Spot.SecondStage.rowAt ro.runs k = some x)
    (hpos : s.pos = r.ρ * ro.per + x)
    (harm : ∀ st rec draw e, decode r.x = some st → jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg → ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e =
        some (M * ro.per, Spot.SecondStage.linkedRow (ownOf h st rec draw r.c₀) (r.drawnPart h) ro)) :
    ∀ st rec draw e (D : _root_.Flock.Draw.UnitDraw) (hn : r.n h < D.proved.size), decode r.x = some st →
      jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg →
      ∃ units unit, ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e = some (units, unit) ∧
        unitsAt roots s = units ∧ (units = 0 → s.stride = 0) ∧
        ∀ b, unit ((r.drawnPart h).lo + D.proved[r.n h]) = some b → s.pos + placeOf roots * s.stride = b := by
  intro st rec draw e D hn hx hrec hdraw hdr hok hf
  obtain ⟨D', hdr', _, hlt', _⟩ := labelAt h hr hc1
  have hDD : D'.proved = D.proved := Array.toList_inj.1 (hdr'.symm.trans hdr)
  rw [hDD] at hlt'
  obtain ⟨i, p, hi, hp, hdec, hρi, hup, hinj⟩ := slot_cell h hr hc1 hS hdr hn (hlt' _ (Array.getElem_mem hn))
  have hpk : p = k := hinj (Set.mem_Iio.2 hp) (Set.mem_Iio.2 hk) (hup.trans hu.symm)
  have hρ : (h.reg.sample (src r.c₀))[i]? = some r.ρ := by
    rw [← hρi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
  refine ⟨_, _, harm st rec draw e hx hrec hdraw hok hf, by rw [unitsAt, hsu, hst, Nat.mul_zero, Nat.add_zero],
    fun _ => hst, fun b hb => ?_⟩
  obtain ⟨a, y, ha, hy, rfl⟩ := ReadsAt.linkedRow_some hb
  rw [hdec, span_cell hp] at ha
  rw [hdec, unit_cell hp, hpk, hrow, Option.some.injEq] at hy
  have ha' : (h.reg.sample (src r.c₀))[i]? = some a := ha
  rw [hρ, Option.some.injEq] at ha'
  rw [hpos, hst, Nat.mul_zero, Nat.add_zero, ha', hy]

/-- **A registered read at the template's own port opens the header's position** for the unit's instance, at a slot
with no units of its own, stride `n_t` and position the unit's in its replay block's cell, under that replay block's
roots, where each entry round 2 accepts for the read takes the own arm (`leafAt_reg` at `harm_own`). -/
theorem leafAt_own {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (hcnt : r.Counted h)
    (hS : FirstSlots h) {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : r.ins.registered = some own) (hreg : reg ∈ own) {p : ℕ}
    (hp : (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) = some p)
    {s : Slot} (hs : s.sized = false) (hs0 : s.units = 0) (hst : s.stride = (r.drawnPart h).n)
    (hpos : s.pos < (r.drawnPart h).n) (hu : h.reg.blockOf r.ρ (templateOf h.ms r.u) s.pos = r.u)
    (harm : ∀ st rec draw e, decode r.x = some st → jsonMsg r.m₁ = some rec → secondDraw h r.c₀ r.c₁ = some draw →
      Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e =
        true → ReadsAt.ReadFields e reg → ReadsAt.armOf (ownOf h st rec draw r.c₀) (r.drawnPart h) reg.name e =
        some ((h.reg.sample (src r.c₀)).size * (r.drawnPart h).n, fun u => some (u - (r.drawnPart h).lo)))
    {t : _root_.Flock.Tables} {positions : Array ℕ} {paths : Array ByteArray} (ht : r.sts[0]!.pub.tables = some t)
    (hent : _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths)) {en : Entry}
    (hl : en.leaves = reg.leaves) :
    leafAt (irootsOf r.m₁ r.ρ) s en =
      positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size p (r.n h)) 0 :=
  leafAt_reg h hr hc1 hcnt hown hreg hp hs (harm_own h hr hc1 hS hs0 hst hpos hu harm) ht hent hl

/-- **A checked unit's gate is opened under its replay block's interior roots**, as the round-1 message states them,
where the gate is one of that block's gates and none of its boundary, and under the statement's otherwise
(`Reveal.rootsAt_eq`). -/
theorem roots_eq {r : CheckedRun} (hr : r.Ok h) (g : ℕ) :
    r.roots h g = if g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false then irootsOf r.m₁ r.ρ else r.s.roots := by
  obtain ⟨Pt, d, spec, -, hP, h2, -, -, -, -, -, hρ, hrb, -⟩ := hr
  exact Reveal.rootsAt_eq (partition h) (two h) (iroots []) hP h2 hρ hrb

/-- **A registered read's roots are the gate's** where round 2 reads it off the interior record exactly when the gate
is one of the unit's replay block's and none of its boundary: `ReadsRegistered`'s roots, given which arm the read
takes. -/
theorem roots_readRoots {r : CheckedRun} (hr : r.Ok h) {st : Roots} {rec draw : Lean.Json}
    (hx : decode r.x = some st) (hrec : jsonMsg r.m₁ = some rec) (hdraw : secondDraw h r.c₀ r.c₁ = some draw)
    {g t : ℕ} {port v : String}
    (harm : (g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false) ↔
      readsInterior (ownOf h st rec draw r.c₀) (h.names.getD t "") port v = true) :
    r.roots h g = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ t port v := by
  obtain ⟨-, -, -, hs, -⟩ := id hr
  obtain ⟨st', hx', hroots⟩ := ReadsAt.parse_roots h hs
  rw [hx] at hx'
  cases hx'
  rw [roots_eq h hr g]
  simp only [readRoots, hx, hrec, hdraw]
  by_cases hg : g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false
  · rw [ite_eq_left hg, ite_eq_left (harm.1 hg)]
  · rw [ite_eq_right hg, ite_eq_right (fun h' => hg (harm.2 h')), hroots]

end SpotFlock.CheckedRun
