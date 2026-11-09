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
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Layout.QSort
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.Soundness.Refine.Setup
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
`lo + u` of the part (`CheckedRun.labelAt`). A registered read's entry in the session statement states its leaves,
first slot and positions (`ReadFields`), and round 2's `rowsOk` holds each position it reads to its unit's slot
(`pos_of_rowsOk`).
-/

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

/-- A port's first slot as a `ports` file's entry gives it: `reads_from`'s, or `0` where the entry names none. -/
theorem readsFrom_of {e : Json} {ctx : String} {rf : ℕ}
    (h : (if _root_.Flock.J.has e _root_.Flock.Registered.READS_FROM_KEY = true then
      _root_.Flock.J.nat e _root_.Flock.Registered.READS_FROM_KEY ctx else pure 0 : _root_.Flock.V ℕ) = .ok rf) :
    (Spot.getD e "reads_from" = .null ∧ rf = 0) ∨ Spot.getD e "reads_from" = .num ⟨Int.ofNat rf, 0⟩ := by
  by_cases hc : _root_.Flock.J.has e _root_.Flock.Registered.READS_FROM_KEY = true
  · rw [if_pos hc] at h
    exact .inr (HeaderRoots.getD_of_getObjVal (FlockComplete.JsonRT.nat_ok h))
  · rw [if_neg hc] at h
    cases h
    exact .inl ⟨getD_of_not_has hc, rfl⟩

/-- A port's reads as a `ports` file's entry gives them: each position read at its place in the entry's `reads`. -/
theorem reads_of {e : Json} {ctx : String} {rs : Array Json} {reads : Array (Option ℕ)}
    {f : Json → _root_.Flock.V (Option ℕ)} (ha : _root_.Flock.J.arr e "reads" ctx = .ok rs)
    (hm : rs.mapM f = .ok reads)
    (hf : ∀ r v, f r = .ok v → v = none ∨ ∃ n, v = some n ∧ r = .num ⟨Int.ofNat n, 0⟩) :
    ∃ rs : Array Json, Spot.getD e "reads" = .arr rs ∧
      ∀ (k pos : ℕ), reads[k]? = some (some pos) → rs[k]? = some (.num ⟨Int.ofNat pos, 0⟩) := by
  obtain ⟨hsz, hel⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hm
  refine ⟨rs, HeaderRoots.getD_of_getObjVal (DrawnAt.getObjVal_of_arr ha), fun k pos hk => ?_⟩
  obtain ⟨hk1, hk2⟩ := Array.getElem?_eq_some_iff.1 hk
  have hk0 : k < rs.size := hsz ▸ hk1
  have hfk := hel k hk0 hk1
  rw [hk2] at hfk
  rw [Array.getElem?_eq_getElem hk0]
  rcases hf _ _ hfk with h0 | ⟨n, hn, hr⟩
  · cases h0
  · cases hn
    rw [hr]

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
      ret_walk [_root_.Flock.J.str, _root_.Flock.J.nat, _root_.Flock.J.arr, _root_.Flock.unhex, ite, Array.mapM]
      all_goals exact ⟨rfl,
        HeaderRoots.getD_of_getObjVal (FlockComplete.JsonRT.nat_ok ‹_root_.Flock.J.nat e "leaves" _ = .ok _›),
        readsFrom_of ‹_›, reads_of ‹_root_.Flock.J.arr e "reads" _ = .ok _› ‹_› fun r v hv => by
          try dsimp only at hv
          split at hv
          · cases hv
            exact .inl rfl
          · simp only [Functor.map, Except.map] at hv
            split at hv
            · cases hv
            · rename_i n hn
              cases hv
              exact .inr ⟨n, rfl, FlockComplete.JsonRT.asNat_ok hn⟩⟩
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

/-- **The tree round 2 holds a registered read's rows to** (`readOk`'s `rowsOk`), by the arm the entry's value takes:
a link's (the boundary's population, each slot reading the replay sample's block at its span), a wire's (its writer's
part's rows, each slot reading its writer's block's first), or the template's own interior port (each slot reading its
own row); none for a registered value, which round 2 holds to its registration alone. -/
def armOf (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) (port : String) (e : Json) :
    Option (ℕ × (ℕ → Option ℕ)) :=
  match Spot.getD e "value", Spot.SecondStage.linkOf own p.template port,
      Spot.SecondStage.wireOf own p.template port with
  | .str _, some _, _ =>
    (Spot.nat? (Spot.getD own.boundary "population")).map fun m => (m, fun u => own.sample[p.span u]?)
  | .str _, none, some w =>
    (Spot.SecondStage.writerOf own w).map fun pw => (own.sample.size * pw.n, fun u => some (p.span u * pw.n))
  | .str v, none, none =>
    ((Spot.Record.byPort own.interior).find? (·.1 == v)).map fun _ => (own.sample.size * p.n, fun u => some (u - p.lo))
  | _, _, _ => none

/-- **Round 2 holds a registered read's rows to its arm's tree** (`armOf`): where `readOk` accepts an entry whose arm
is a link, a wire or the template's own port, `rowsOk` holds of the arm's unit count and slots. -/
theorem rowsOk_of_readOk {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} (h : Spot.SecondStage.readOk own named p port e = true) {units : ℕ} {unit : ℕ → Option ℕ}
    (ha : armOf own p port e = some (units, unit)) :
    Spot.SecondStage.rowsOk own.sample.size p e units unit = true := by
  unfold Spot.SecondStage.readOk at h
  unfold armOf at ha
  cases hv : Spot.getD e "value" <;> simp only [hv] at h ha
  rename_i v
  cases hl : Spot.SecondStage.linkOf own p.template port <;>
    cases hw : Spot.SecondStage.wireOf own p.template port <;> simp only [hl, hw, Bool.and_eq_true] at h ha
  · cases hx : (Spot.Record.byPort own.interior).find? (·.1 == v) with
    | none => simp [hx] at ha
    | some kx =>
      obtain ⟨k, x⟩ := kx
      simp only [hx, Option.map_some, Option.some.injEq, Prod.mk.injEq] at ha
      obtain ⟨rfl, rfl⟩ := ha
      simp only [hx, Bool.and_eq_true] at h
      exact h.2
  · cases hpw : Spot.SecondStage.writerOf own _ with
    | none => simp [hpw] at ha
    | some pw =>
      simp only [hpw, Option.map_some, Option.some.injEq, Prod.mk.injEq] at ha
      obtain ⟨rfl, rfl⟩ := ha
      obtain ⟨-, h⟩ := h
      split at h
      · rename_i heq
        simp only [hpw, Option.some.injEq] at heq
        subst heq
        simp only [Bool.and_eq_true] at h
        exact h.2
      · cases h
  all_goals
    cases hm : Spot.nat? (Spot.getD own.boundary "population") with
    | none => simp [hm] at ha
    | some m =>
      simp only [hm, Option.map_some, Option.some.injEq, Prod.mk.injEq] at ha
      obtain ⟨rfl, rfl⟩ := ha
      obtain ⟨-, h⟩ := h
      split at h
      · rename_i heq
        simp only [hm, Option.some.injEq] at heq
        subst heq
        simp only [Bool.and_eq_true] at h
        exact h.2
      · cases h

/-- **A drawn session's unit labels**: where a public file whose header round 2's `populationOk` accepts for part `p`
of a sample of `q` blocks loads `pop`, and the draw `d` gives `pub`, `pub` labels its instances with the slots of
`p` its draw proves. -/
theorem unitIndices_session {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {f : ByteArray}
    {pop pub : _root_.Flock.Public} {hdr d : Json} (hpop : _root_.Flock.HmOut.loadPublic tags c f = .ok pop)
    (hH : headerOf f = some hdr) (hd : _root_.Flock.HmOut.drawn c pop d = .ok pub) {q : ℕ}
    {p : Spot.SecondStage.Part} (hok : Spot.SecondStage.populationOk q p hdr = true) :
    ∃ D, _root_.Flock.Draw.UnitDraw.ofJson d = .ok D ∧ _root_.Flock.Registered.unitIndices pub =
      .ok (D.proved.map fun v => ((Array.range (q * p.n)).map (p.lo + ·)).getD v 0) := by
  obtain ⟨h₀, hh₀, hhdr, -, m, hm⟩ := HeaderRoots.loadPublic_raw hpop
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
  obtain ⟨j, hj, hnj, -⟩ := Depths.loadPublic_counts hpop
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

end SpotFlock.CheckedRun
