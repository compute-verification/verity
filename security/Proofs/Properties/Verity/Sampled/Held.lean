module

public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Claims.Bridge.Custody
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
public import Proofs.Flock.Soundness.Discharge.Layout.Drawn
public import Proofs.Flock.Soundness.Discharge.Hidden.Spec
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Bridge.V5.Accepted
public import Proofs.Flock.Claims.Complete.Json.Parse
public import Proofs.Properties.Spot
public import Proofs.Core.QSort
public import Std.Data.String.ToInt
public import Mathlib.Data.List.Find
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import Proofs.Flock.Soundness.Discharge.Hidden.Public

/-!
# What v2 holds of the record's sample

The record's sample (`recordDraw_firstStop`, `MovesOf`'s clause (i)) reads v2's log at its coins (`DrawReadAll`) and
v2's hold on the record's `unit_draw` (`DrawHeld`). This module gives what v2's code gives of them.

* **v2's log at its coins** (`drawReadAll_coinsRead`): `coinsRead` is `logRead` (`coinsRead_eq_logRead`), whose read
  shows the sample wherever v2 accepts (`drawReadAll_logRead`).
* **v2 accepts only a subset sample** (`recordDraw_subset`): where v2's case accepts, the record v2 reads holds a sample,
  since `verify --zk` refuses a session without one (`verdictOf_drawn`); its sampling design is one a soundness claim samples by at
  `--zk` (`ProvedScope.drawLaw`, `lawWhy_zk`), the subset or the stratified sampling design; and the session v2 sets up has no
  `--stratified` input, so `Stmt.setupTables` refuses a stratified sample (`setupTables_stratified`). So the record's sample
  is the `.subset` print of the member's share of the window's sample under the verifier's law, which `DrawHeld` reads at
  the units v2 drew; the bound is that sampling design's, from v1's window check (`Spot.OneStage.checkAt`), not the
  print's.
* **A subset sample that decodes is its own print** (`ofJson_subset_eq`): a value `norm` fixes that `UnitDraw.ofJson`
  decodes to a subset sample is an object of exactly `law`, `population`, `k` and `units` (`obj_eq_mkObj4`), so it is the
  sample's print at no closure, its units ascending below its population. The record's sample is rebuilt
  (`Claims.Bridge.Custody.recordDraw_norm`), so it is its decoded sample's subset print (`recordDrawn_of_subset`).
* **The record sample's population** (`setupTables_population`, `population_held`): the session v2 sets up at the record's
  sample loads the public file to a population of the sample's population (`HmRow.drawn`), and the load reads the header's
  `instances` (`Bridge.loadPublic_core`, `header_of_headerOf`), which v2 held to the record's population
  (`publicHeld_population`).
* **The record sample's units** (`provedUnits_eq`, `checkAt_proved`): where v2's run accepts, the session passed the
  population check, so the public file's unit indices are `0, …, N − 1`
  (`Properties.Spot.FilePart.check_indices`), and proved the units the record's sample names (`step_proved`),
  which v2 holds to the units v1 proved (`provedHeld_ok`); v1's window check proves the units it drew, a sample v2's
  evidence gives without a closure (`evidenceOf_closure`).
* **`DrawHeld`** (`drawHeld_of`, `drawHeld`): wherever the subset sample the record's decodes to is over v2's `n` units at
  the units v2 drew, `DrawHeld` holds; and it is, for every composite.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## v2's log at its coins -/

/-- **v2's log at its coins shows the sample** wherever v2 accepts: `coinsRead` is `logRead`. -/
theorem drawReadAll_coinsRead (k : Verity.Registration.Composite) : DrawReadAll k coinsRead :=
  coinsRead_eq_logRead ▸ drawReadAll_logRead k

/-! ## v2 accepts only a subset sample -/

/-- **`verify --zk` refuses a session without a sample**: where a session's verdict passes, its record's sample is some. -/
theorem verdictOf_drawn {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit}
    {coinKeys : Array String} {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) : draw.isSome = true := by
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨_, _, _⟩, -, h⟩ := ebind_ok h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, hd, -⟩ := ebind_ok h
  cases draw with
  | none => simp [Flock.ProvedScope.drawLaw] at hd
  | some _ => rfl

/-- **One session's `step` from the empty run, where every session accepted, reads a record that holds a sample.** -/
theorem step_drawn {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.Spot.OneStageZk.RunFiles} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) :
    ∃ recB record, bs.get s.record s!"session {s.name}'s record" = .ok recB ∧
      String.fromUTF8? recB = some record ∧ (Flock.Witness.recordDraw record).isSome = true := by
  unfold _root_.Spot.OneStageZk.step at h
  obtain ⟨recB, hb, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · rename_i record hrec
    obtain ⟨_, -, h⟩ := ebind_ok h
    exact ⟨recB, record, hb, hrec, verdictOf_drawn (push_ok h hall)⟩
  · cases h

/-- **Where v2's check accepts, the session's record holds a sample.** -/
theorem checkBlobsAt_drawn {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ record, sessionRecord e bs = some record ∧ (Flock.Witness.recordDraw record).isSome = true := by
  obtain ⟨r, _, hsz, -, -, hrun, hall⟩ := checkBlobsAt_run h
  have hs := eq_single hsz
  rw [hs] at hrun
  obtain ⟨_, _, _, _, -, -, -, -, _, _, _, _, hstep, -⟩ := run_verified hrun
  obtain ⟨recB, record, hb, hrec, hd⟩ := step_drawn hstep hall
  exact ⟨record, sessionRecord_eq hs hb hrec, hd⟩

/-- **`setupAt` at a sample holds the sample's law** (`ProvedScope.drawLaw` at `--zk`). -/
theorem setupAt_drawLaw {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {dj : Json}
    {x : Flock.SessionSpec × Array Flock.Setup × Json × Json}
    (h : _root_.Spot.OneStageZk.setupAt W rd (some dj) = .ok x) :
    Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags W) false (some dj) = .ok () := by
  unfold _root_.Spot.OneStageZk.setupAt at h
  obtain ⟨⟨_, _⟩, -, h⟩ := ebind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, hu, -⟩ := ebind_ok h
  simpa only [Option.isSome_some, ↓reduceIte] using hu

/-- **A sample `drawLaw` passes decodes**, to a sample whose sampling design `lawWhy` doesn't refuse. -/
theorem drawLaw_ok {zk : Bool} {tags : Flock.Tags} {pt : Bool} {dj : Json}
    (h : Flock.ProvedScope.drawLaw zk tags pt (some dj) = .ok ()) :
    ∃ D, Flock.Draw.UnitDraw.ofJson dj = .ok D ∧ Flock.ProvedScope.lawWhy zk tags pt D = none := by
  simp only [Flock.ProvedScope.drawLaw] at h
  cases hD : Flock.Draw.UnitDraw.ofJson dj with
  | error e => simp [hD] at h
  | ok D =>
    simp only [hD] at h
    cases hw : Flock.ProvedScope.lawWhy zk tags pt D with
    | some why => simp [hw] at h
    | none => exact ⟨D, rfl, hw⟩

/-- **The laws a soundness claim samples by at `--zk`**, at public tables: the subset and the stratified sampling design. -/
theorem lawWhy_zk {tags : Flock.Tags} {D : Flock.Draw.UnitDraw}
    (h : Flock.ProvedScope.lawWhy true tags false D = none) :
    (∃ kd, D.law = .subset kd) ∨ ∃ K St, D.law = .stratified K St := by
  unfold Flock.ProvedScope.lawWhy at h
  cases hl : D.law with
  | subset kd => exact .inl ⟨kd, rfl⟩
  | stratified K St => exact .inr ⟨K, St, rfl⟩
  | bernoulli a b => simp [hl] at h
  | work K ws => simp [hl] at h

/-- **A session without `--stratified` refuses a stratified sample**: `Stmt.setupTables` holds a stratified sample's law to
the verifier's own `K`, and refuses it without one. -/
theorem setupTables_stratified {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {dj : Json} {partition : Option Json} {program : Option ByteArray}
    {r : Flock.SessionSpec × Array Flock.Stmt} {D : Flock.Draw.UnitDraw} {K : ℕ} {St : Array Flock.Draw.Stratum}
    (h : Flock.Stmt.setupTables tags cf pf coins count tables (some dj) partition program = .ok r)
    (hD : Flock.Draw.UnitDraw.ofJson dj = .ok D) (hl : D.law = .stratified K St) : False := by
  unfold Flock.Stmt.setupTables at h
  simp only [hD, hl, Except.map] at h
  cases partition <;> cases program <;> simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- **v2 accepts only a subset sample**: where composite `k` accepted what the prover sent and v2's case in it accepts,
the record v2 reads holds a sample (`unit_draw`), which decodes to a sample by the subset sampling design. -/
theorem recordDraw_subset {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : _root_.Properties.Challenger.Draws}
    (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) {record : String}
    (hr : sentRecord s = some record) :
    ∃ dj D kd, Flock.Witness.recordDraw record = some dj ∧ Flock.Draw.UnitDraw.ofJson dj = .ok D ∧
      D.law = .subset kd := by
  obtain ⟨record', _, _, _, _, -, hr', -, hv⟩ := sent_verified hS hcase hacc
  obtain ⟨-, -, -, -, -, _, _, hck⟩ := hacc
  obtain ⟨record'', hrec'', hdrawn⟩ := checkBlobsAt_drawn hck
  obtain rfl := Option.some.inj (hr'.symm.trans hr)
  obtain ⟨_, _, _, _, _, -, -, -, -, -, hrec, hst, -, -⟩ := hv
  obtain rfl := Option.some.inj (hrec''.symm.trans hrec)
  obtain ⟨dj, hdj⟩ := Option.isSome_iff_exists.1 hdrawn
  rw [hdj] at hst
  obtain ⟨D, hD, hw⟩ := drawLaw_ok (setupAt_drawLaw hst)
  obtain ⟨_, hss, -⟩ := setupAt_ok hst
  obtain ⟨hset, -⟩ := sessionAt_ok hss
  rcases lawWhy_zk hw with ⟨kd, hk⟩ | ⟨K, St, hk⟩
  · exact ⟨dj, D, kd, hdj, hD, hk⟩
  · exact (setupTables_stratified hset hD hk).elim

/-! ## A subset sample that decodes is its own print -/

section Print

open FlockComplete.JsonRT (getObj_ok str_ok nat_ok arr_ok asNat_ok key_mem mem_of_getObjVal getObjVal_of_get)

/-- **The values `mapM` reads as natural numbers are their prints.** -/
theorem mapM_asNat_eq {a : Array Json} {c : String} {us : Array ℕ} (h : a.mapM (Flock.J.asNat · c) = .ok us) :
    a = us.map fun x => Lean.toJson x := by
  obtain ⟨hs, hb⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
  refine Array.ext (by rw [Array.size_map, hs]) fun i h₁ h₂ => ?_
  rw [Array.getElem_map]
  exact asNat_ok (hb i h₁ (by rw [hs]; exact h₁))

/-- **An object of four keys is their object**: a well-formed object `norm` fixes, of four keys, `law`, `population`,
`k` and `units` among them, is `Json.mkObj` of those keys at its values, wherever `norm` fixes that too. -/
theorem obj_eq_mkObj4 {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hn : Flock.norm (.obj m) = .obj m)
    (hs : m.toArray.size = 4) {a b c d : Json} (ha : m.get? "law" = some a) (hb : m.get? "population" = some b)
    (hc : m.get? "k" = some c) (hd : m.get? "units" = some d)
    (hN : Flock.norm (Json.mkObj [("law", a), ("population", b), ("k", c), ("units", d)]) =
      Json.mkObj [("law", a), ("population", b), ("k", c), ("units", d)]) :
    Json.obj m = Json.mkObj [("law", a), ("population", b), ("k", c), ("units", d)] := by
  have hD : [("law", a), ("population", b), ("k", c), ("units", d)].Pairwise
      (fun p q : String × Json => ¬ compare p.1 q.1 = .eq) := by simp
  have hO : ∀ (key : String) (v : Json), (key, v) ∈ [("law", a), ("population", b), ("k", c), ("units", d)] →
      (Std.TreeMap.Raw.ofList [("law", a), ("population", b), ("k", c), ("units", d)] compare).get? key = some v :=
    fun key v hkv => by
      rw [Std.TreeMap.Raw.get?_eq_getElem?]
      exact Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hD hkv
  have hmem : ∀ key ∈ ["law", "population", "k", "units"], key ∈ m := by
    simp only [List.mem_cons, List.mem_nil_iff, or_false]
    rintro key (rfl | rfl | rfl | rfl)
    · exact (mem_of_getObjVal hm (getObjVal_of_get ha)).1
    · exact (mem_of_getObjVal hm (getObjVal_of_get hb)).1
    · exact (mem_of_getObjVal hm (getObjVal_of_get hc)).1
    · exact (mem_of_getObjVal hm (getObjVal_of_get hd)).1
  have hget : ∀ key, m.get? key =
      (Std.TreeMap.Raw.ofList [("law", a), ("population", b), ("k", c), ("units", d)] compare).get? key := by
    intro key
    by_cases hk : key ∈ ["law", "population", "k", "units"]
    · simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk
      rcases hk with rfl | rfl | rfl | rfl
      · rw [ha, hO "law" a (by simp)]
      · rw [hb, hO "population" b (by simp)]
      · rw [hc, hO "k" c (by simp)]
      · rw [hd, hO "units" d (by simp)]
    · have h1 : key ∉ m := fun h => hk (key_mem hm (by decide) hmem
        (by rw [← Std.TreeMap.Raw.size_toArray hm]; exact hs) h)
      have h2 : ([("law", a), ("population", b), ("k", c), ("units", d)].map Prod.fst).contains key = false := by
        rw [Bool.eq_false_iff]
        intro h
        apply hk
        simpa using h
      rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.get?_eq_getElem?,
        Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false h2]
      cases e : m[key]? with
      | none => rfl
      | some v => exact absurd ((Std.TreeMap.Raw.mem_iff_isSome_getElem? hm).2 (by rw [e]; rfl)) h1
  refine FlockDeploy.eq_of_jsonEquiv_norm hn hN ?_
  show FlockDeploy.JsonEquiv (.obj m) (.obj (Std.TreeMap.Raw.ofList _ compare))
  refine (FlockDeploy.jsonEquiv_obj_iff
    (n := Std.TreeMap.Raw.ofList [("law", a), ("population", b), ("k", c), ("units", d)] compare) hm
    Std.TreeMap.Raw.WF.ofList).2 ⟨fun key => by rw [hget], fun key x y hx hy => ?_⟩
  rw [hget, hy] at hx
  cases hx
  exact FlockDeploy.JsonEquiv.refl _

/-- **A subset sample's object is its print**: an object as `obj_eq_mkObj4` reads it, at the sampling design `subset`, the population
`pop`, `k` the units' count and the units `us`, is the subset print of `us` over `pop` units at no closure. -/
theorem obj_eq_subsetPrint {m : Std.TreeMap.Raw String Json compare} (hm : m.WF)
    (hn : Flock.norm (.obj m) = .obj m) (hs : m.toArray.size = 4) {s : String} {pop k : ℕ} {us : Array ℕ}
    (hlaw : m.get? "law" = some (.str s)) (hls : s = "subset")
    (hpop : m.get? "population" = some (.num ⟨Int.ofNat pop, 0⟩)) (hk : m.get? "k" = some (.num ⟨Int.ofNat k, 0⟩))
    (hks : k = us.size) (hus : m.get? "units" = some (.arr (us.map fun x => Lean.toJson x))) :
    Json.obj m = Flock.Draw.UnitDraw.toJson ⟨.subset us.size, pop, us, #[]⟩ := by
  subst hls hks
  exact obj_eq_mkObj4 hm hn hs hlaw hpop hk hus (norm_toJson_subset us.size pop us #[])

theorem eq_true_of_not_not {b : Bool} (h : ¬ (!b) = true) : b = true := by
  cases b <;> simp_all

set_option maxHeartbeats 4000000 in
/-- **A subset sample that decodes is its own print**: a value `norm` fixes that decodes (`UnitDraw.ofJson`) to a sample
`D` by the subset sampling design has `D`'s units ascending below its population, and is `D`'s print at no closure. -/
theorem ofJson_subset_eq {j : Json} (hn : Flock.norm j = j) {D : Flock.Draw.UnitDraw}
    (h : Flock.Draw.UnitDraw.ofJson j = .ok D) {k : ℕ} (hl : D.law = .subset k) :
    unitsAscending D.population D.units = true ∧
      j = Flock.Draw.UnitDraw.toJson ⟨.subset D.units.size, D.population, D.units, #[]⟩ := by
  have hw : Claims.Json.Wf j := hn ▸ FlockDeploy.wf_norm j
  obtain rfl : D.units.size = k := (FlockSoundness.Discharge.Composed.ofJson_shape h).2.2 k hl
  unfold Flock.Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals
    have hp := ‹(pure _ : Flock.V Flock.Draw.Law) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq] at hp
  all_goals
    try simp only [reduceCtorEq] at hl
  all_goals
    have hj := getObj_ok ‹j.getObj? = _›
    subst hj
    cases hw with
    | obj hm hwv =>
    have hc := ‹¬ (_ != 4) = true›
    simp only [bne_iff_ne, ne_eq, not_not] at hc
    have hu := mapM_asNat_eq ‹Array.mapM (Flock.J.asNat · "U2: a drawn unit") _ = Except.ok _›
    try simp only [Flock.Draw.Law.subset.injEq] at hp
    try simp only [Flock.Draw.Law.subset.injEq] at hl
    subst_vars
    exact ⟨eq_true_of_not_not ‹¬ (!(List.range _).all _) = true›,
      obj_eq_subsetPrint hm hn (by simpa using hc) (mem_of_getObjVal hm (str_ok ‹Flock.J.str _ "law" _ = _›)).2 rfl
        (mem_of_getObjVal hm (nat_ok ‹Flock.J.nat _ "population" _ = _›)).2
        (mem_of_getObjVal hm (nat_ok ‹Flock.J.nat _ "k" _ = _›)).2 rfl
        (mem_of_getObjVal hm (arr_ok ‹Flock.J.arr _ "units" _ = _›)).2⟩

/-- **A subset record sample is its own print**: where the record's sample (`recordDraw`) decodes to a sample `D` by the
subset sampling design, the record's sample is `D`'s subset print (`recordDrawn`). -/
theorem recordDrawn_of_subset {record : String} {dj : Json} {D : Flock.Draw.UnitDraw} {kd : ℕ}
    (hr : Flock.Witness.recordDraw record = some dj) (hD : Flock.Draw.UnitDraw.ofJson dj = .ok D)
    (hl : D.law = .subset kd) : recordDrawn (.subset D.units.size) D.population D.units record := by
  obtain ⟨ha, he⟩ :=
    ofJson_subset_eq (Claims.Bridge.Custody.recordDraw_norm ((exec_recordDraw record).trans hr)) hD hl
  exact ⟨ha, by rw [hr, ← he]⟩

end Print

/-! ## The record sample's population -/

theorem forIn_first {β : Type} (p : ℕ → Bool) (f : ℕ → β) : ∀ l : List ℕ,
    (forIn (m := Id) l ((none : Option β), ()) fun i _ =>
      if p i = true then pure (ForInStep.done (some (f i), ())) else pure (ForInStep.yield (none, ()))) =
      ((l.find? p).map f, ())
  | [] => rfl
  | a :: l => by
    rw [List.forIn_cons]
    by_cases ha : p a = true
    · simp only [ha, ↓reduceIte, List.find?_cons, Option.map_some]
      rfl
    · simp only [ha, Bool.false_eq_true, ↓reduceIte, List.find?_cons]
      exact forIn_first p f l

/-- **The header line ends at the first newline**, as `Flock.findByte` finds it. -/
theorem headerLine_of_findByte {b : ByteArray} {nl : ℕ} (h : Flock.findByte b 10 = some nl) :
    _root_.Spot.FilePart.headerLine b = b.extract 0 nl := by
  unfold Flock.findByte at h
  unfold _root_.Spot.FilePart.headerLine
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range'] at h ⊢
  rw [forIn_first] at h
  rw [forIn_first]
  have hc : (List.range' 0 (Std.Legacy.Range.size [:b.size])).find? (fun i => b[i]! == 10) =
      (List.range' 0 (Std.Legacy.Range.size [:b.size])).find? (fun i => b.get! i == 10) := by
    refine List.find?_congr fun i hi => ?_
    have hi' : i < b.size := by simpa [List.mem_range', Std.Legacy.Range.size] using hi
    rw [getElem!_pos b i hi']
    simp only [ByteArray.get!]
    rw [getElem!_pos b.data i hi']
    rfl
  rw [hc]
  generalize List.find? (fun i => b.get! i == 10) (List.range' 0 (Std.Legacy.Range.size [:b.size])) = o at h ⊢
  cases o with
  | none => cases h
  | some j =>
    cases h
    rfl

/-- **The public file's header as the population's load reads it is v2's** (`FilePart.header`). -/
theorem header_of_headerOf {f : ByteArray} {hd : Json} (h : RecDeploy.SetupValues.headerOf f = some hd) :
    _root_.Spot.FilePart.header f = .ok hd := by
  unfold RecDeploy.SetupValues.headerOf at h
  cases hnl : Flock.findByte f 10 with
  | none => simp only [hnl] at h; cases h
  | some nl =>
    simp only [hnl] at h
    unfold _root_.Spot.FilePart.header
    rw [headerLine_of_findByte hnl]
    cases ht : String.fromUTF8? (f.extract 0 nl) with
    | none => simp only [ht] at h; cases h
    | some text =>
      simp only [ht] at h
      cases hj : Flock.JsonParse.parse text with
      | error e => simp only [hj, Except.toOption, Option.map_none] at h; cases h
      | ok j =>
        simp only [hj, Except.toOption, Option.map_some, Option.some.injEq] at h
        subst h
        simp only [hj, Except.mapError]
        rfl

theorem fail_codes (c : _root_.Spot.Check) (code : String) (w : Json) :
    (c.fail code w).codes.isEmpty = false := by
  unfold _root_.Spot.Check.fail
  split
  · rename_i hc
    rw [Bool.eq_false_iff]
    intro he
    rw [Array.isEmpty_iff] at he
    rw [he] at hc
    simp at hc
  · simp

/-- **`matches_public` holds the instances to the record's population.** -/
theorem matchesPublic_population {rec s r d pop : Json}
    (h : (_root_.Spot.Record.matchesPublic rec s r d pop).codes.isEmpty = true) :
    _root_.Spot.jeq pop (_root_.Spot.getD rec "population") = true := by
  unfold _root_.Spot.Record.matchesPublic at h
  simp only [Id.run] at h
  cases hp : _root_.Spot.jeq pop (_root_.Spot.getD rec "population")
  · simp only [hp, Bool.not_false, ↓reduceIte] at h
    repeat' split at h
    all_goals
      simp only [pure, fail_codes] at h
      cases h
  · rfl

/-- **v2 holds the public file's instances to the record's population** (`publicHeld`). -/
theorem publicHeld_population {rec : Json} {pubB : ByteArray} {hd : Json}
    (h : _root_.Spot.OneStageZk.publicHeld rec pubB = .ok ())
    (hh : _root_.Spot.FilePart.header pubB = .ok hd) :
    _root_.Spot.jeq (_root_.Spot.getD hd "instances")
      (_root_.Spot.getD rec "population") = true := by
  unfold _root_.Spot.OneStageZk.publicHeld at h
  obtain ⟨_, he, h⟩ := ebind_ok h
  rw [hh] at he
  cases he
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    exact matchesPublic_population (by simpa using hc)

/-- **A natural number `jeq` holds to a value is the value's natural number.** -/
theorem nat_of_jeq {m n : ℕ} {j : Json}
    (hj : _root_.Spot.jeq (.num ⟨Int.ofNat m, 0⟩) j = true) (hn : _root_.Spot.nat? j = some n) :
    m = n := by
  cases j with
  | num x =>
    obtain ⟨mx, ex⟩ := x
    unfold _root_.Spot.nat? _root_.Spot.int? at hn
    by_cases he : ex = 0
    · subst he
      simp only [beq_self_eq_true, ↓reduceIte] at hn
      split at hn
      · rename_i hle
        simp only [Option.some.injEq] at hn
        subst hn
        obtain ⟨k, rfl⟩ := Int.eq_ofNat_of_zero_le hle
        simp only [_root_.Spot.jeq, Flock.canon, beq_self_eq_true, ↓reduceIte, pure, Except.pure,
          beq_iff_eq] at hj
        rw [Int.toNat_natCast]
        exact Int.ofNat.inj (Int.repr_injective hj)
      · cases hn
    · simp [he] at hn
  | _ => simp [_root_.Spot.nat?, _root_.Spot.int?] at hn

/-- **v2's session at a sample loads the population the sample is over**: at one table, `Stmt.setupTables` at a sample `dj`
that decodes to `D` loads the public file to a population of `D.population` instances (`HmRow.drawn`), on the
statement's circuit (`Stmt.setupH`) or, its outputs hidden, on the circuit without output ports
(`Stmt.setupHidden`). -/
theorem setupTables_population {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {dj : Json} {partition : Option Json} {program : Option ByteArray}
    {r : Flock.SessionSpec × Array Flock.Stmt} {D : Flock.Draw.UnitDraw}
    (h : Flock.Stmt.setupTables tags cf pf coins 1 tables (some dj) partition program = .ok r)
    (hD : Flock.Draw.UnitDraw.ofJson dj = .ok D) :
    ∃ c pop, Flock.HmRow.loadPublic tags c pf = .ok pop ∧ D.population = pop.n := by
  obtain ⟨spec, sts⟩ := r
  cases hh : tags.hiddenOutputs with
  | false =>
    rcases FlockSoundness.Discharge.Exec.setupTables_ok hh h with ⟨-, st, hst, -, -⟩ | ⟨h1, -⟩
    · obtain ⟨pop, hl, hdr⟩ := FlockSoundness.Discharge.Layout.setupH_drawn hst
      obtain ⟨D', hD', hp, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdr
      rw [hD] at hD'
      cases hD'
      exact ⟨_, pop, hl, hp⟩
    · exact absurd h1 (Nat.lt_irrefl 1)
  | true =>
    rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH hh h with ⟨-, st, hst, -, -⟩ | ⟨h1, -⟩
    · obtain ⟨pop, hl, -, hdr⟩ := (FlockSoundness.Discharge.Hidden.setupHidden_spec' hst).pub
      obtain ⟨D', hD', hp, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec (hdr dj rfl)
      rw [hD] at hD'
      cases hD'
      exact ⟨_, pop, FlockSoundness.Discharge.Hidden.loadPublic_nil hl, hp⟩
    · exact absurd h1 (Nat.lt_irrefl 1)

/-- **A population v2 loads from the public file it held to the record is the record's**: the load reads the header's
`instances` (`Bridge.loadPublic_core`), which `publicHeld` holds to the record's population. -/
theorem population_held {rec : Json} {pubB : ByteArray} {tags : Flock.Tags} {c : Flock.Circuit}
    {pop : Flock.Public} {n : ℕ} (hph : _root_.Spot.OneStageZk.publicHeld rec pubB = .ok ())
    (hl : Flock.HmRow.loadPublic tags c pubB = .ok pop)
    (hn : _root_.Spot.nat? (_root_.Spot.getD rec "population") = some n) : pop.n = n := by
  rw [← Flock.With.loadPublicWith_std] at hl
  obtain ⟨nl, hd, counts, -, hhd, hi, -⟩ := Flock.Bridge.loadPublic_core _ _ _ _ _ hl
  have hj := publicHeld_population hph (header_of_headerOf hhd)
  have hg : _root_.Spot.getD hd "instances" = .num ⟨Int.ofNat pop.n, 0⟩ := by
    simp only [_root_.Spot.getD, _root_.Spot.field?, FlockComplete.JsonRT.nat_ok hi]
    rfl
  rw [hg] at hj
  exact nat_of_jeq hj hn

/-! ## The record sample's units -/

theorem verdictOf_population {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit}
    {coinKeys : Array String} {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) : population = .ok () := by
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨_, _, _⟩, -, h⟩ := ebind_ok h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  exact h

theorem push_proved {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads}
    {programSha : Option String} {want : Json} {r r' : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {st : _root_.Spot.OneStageZk.SetUp}
    {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.push I rd programSha want r s record st coinKey (.ok ()) = .ok r') :
    ∃ units, _root_.Spot.OneStageZk.provedUnits I.publicFile record = .ok units ∧
      r'.proved = r.proved ++ units := by
  unfold _root_.Spot.OneStageZk.push at h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨units, hu, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨units, hu, rfl⟩

/-- **One session's `step` from the empty run, where every session accepted, passed the population check and proved
the public file's units its record samples** (`provedUnits`). -/
theorem step_proved {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.Spot.OneStageZk.RunFiles} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) :
    ∃ recB record, bs.get s.record s!"session {s.name}'s record" = .ok recB ∧
      String.fromUTF8? recB = some record ∧ population = .ok () ∧
      _root_.Spot.OneStageZk.provedUnits I.publicFile record = .ok r.proved := by
  unfold _root_.Spot.OneStageZk.step at h
  obtain ⟨recB, hb, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · rename_i record hrec
    obtain ⟨_, -, h⟩ := ebind_ok h
    have hv := push_ok h hall
    rw [hv] at h
    obtain ⟨units, hu, hp⟩ := push_proved h
    refine ⟨recB, record, hb, hrec, verdictOf_population hv, ?_⟩
    rw [hp, Array.empty_append]
    exact hu
  · cases h

theorem provedHeld_ok {proved want : Array ℕ}
    (h : _root_.Spot.OneStageZk.provedHeld proved want = .ok ()) : proved.qsort (· < ·) = want := by
  unfold _root_.Spot.OneStageZk.provedHeld at h
  split at h
  · cases h
  · rename_i hne
    simpa using hne

theorem oneStage_closure {b : ByteArray} {e : _root_.Spot.OneStage.Evidence}
    (h : _root_.Spot.OneStage.evidenceOf b = .ok e) :
    _root_.Spot.field? e.draw "closure" = none := by
  unfold _root_.Spot.OneStage.evidenceOf at h
  obtain ⟨e', -, h⟩ := ebind_ok h
  split at h
  · cases h
  · rename_i hc
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simpa using hc

theorem zk_closure {b : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    (h : _root_.Spot.OneStageZk.evidenceOf b = .ok e) :
    _root_.Spot.field? e.base.draw "closure" = none := by
  unfold _root_.Spot.OneStageZk.evidenceOf at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact oneStage_closure (by assumption))

/-- **v2's evidence names no closure**: `OneStage.evidenceOf` refuses a sample with one. -/
theorem evidenceOf_closure {b : ByteArray} {ev : _root_.Spot.OneStageZk.Evidence}
    (h : VStar.Recursive.Sampled.evidenceOf b = .ok (.inl ev)) :
    _root_.Spot.field? ev.base.draw "closure" = none := by
  unfold VStar.Recursive.Sampled.evidenceOf at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact zk_closure (by assumption))

/-- **The units v2 proves are the record sample's**: where the public file's unit indices are `0, …, N − 1`, the units
`provedUnits` reads at the record's `unit_draw` are its units, sorted. -/
theorem provedUnits_eq {pubB : ByteArray} {record : String} {hd j : Json} {N : ℕ} {us units : Array ℕ}
    (hh : _root_.Spot.FilePart.header pubB = .ok hd)
    (hi : _root_.Spot.FilePart.indicesOf hd = .ok (Array.range N))
    (hj : Flock.JsonParse.parse record = .ok j)
    (hu : _root_.Spot.getD (_root_.Spot.getD j "unit_draw") "units" =
      .arr (us.map fun x => Lean.toJson x))
    (h : _root_.Spot.OneStageZk.provedUnits pubB record = .ok units) : units = us.qsort (· < ·) := by
  unfold _root_.Spot.OneStageZk.provedUnits at h
  obtain ⟨_, e1, h⟩ := ebind_ok h
  rw [hh] at e1
  cases e1
  obtain ⟨_, e2, h⟩ := ebind_ok h
  rw [hi] at e2
  cases e2
  obtain ⟨_, e3, h⟩ := ebind_ok h
  rw [hj] at e3
  cases e3
  simp only [hu] at h
  obtain ⟨vs, hm, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨hs, hb⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hm
  have hvs : vs = us := by
    refine Array.ext (by simpa using hs) fun i h₁ h₂ => ?_
    have := hb i (by simpa using h₂) h₁
    simp only [Array.getElem_map] at this
    have hn : _root_.Spot.nat? (Lean.toJson us[i]) = some us[i] := rfl
    simp only [hn, bind, Option.bind] at this
    cases hr : (Array.range N)[us[i]]? with
    | none => rw [hr] at this; cases this
    | some u =>
      rw [hr] at this
      simp only [pure, Except.pure, Except.ok.injEq] at this
      rw [Array.getElem?_range] at hr
      split at hr
      · cases hr; exact this.symm
      · cases hr
  rw [hvs]

theorem eq_of_jsonEquiv_nat {x : Json} {u : ℕ} (h : FlockDeploy.JsonEquiv (Lean.toJson u) x) :
    x = Lean.toJson u := by
  cases x with
  | num n =>
    have : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .num n := h
    cases this
    rfl
  | null => cases (h : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .null)
  | bool b => cases (h : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .bool b)
  | str s => cases (h : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .str s)
  | arr a =>
    obtain ⟨a⟩ := a
    cases (h : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .arr _)
  | obj m =>
    rw [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm_obj] at h
    cases (h : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .obj _)

theorem jsonForm_arr' (a : Array Json) : ∃ f, FlockDeploy.jsonForm (.arr a) = .arr f := by
  obtain ⟨a⟩ := a
  exact ⟨_, rfl⟩

/-- **A value whose `norm` is a subset sample's print holds the sample's units**, as numbers, under `units`. -/
theorem units_of_norm {d : Json} (hw : Claims.Json.Wf d) {pop : ℕ} {us : Array ℕ}
    (hn : Flock.norm d = Flock.Draw.UnitDraw.toJson ⟨.subset us.size, pop, us, #[]⟩) :
    _root_.Spot.getD d "units" = .arr (us.map fun x => Lean.toJson x) := by
  have he := FlockDeploy.jsonEquiv_norm d hw
  rw [hn] at he
  have hL : Flock.Draw.UnitDraw.toJson ⟨.subset us.size, pop, us, #[]⟩ =
      .obj (Std.TreeMap.Raw.ofList [("law", Json.str "subset"), ("population", Lean.toJson pop),
        ("k", Lean.toJson us.size), ("units", Json.arr (us.map fun x => Lean.toJson x))] compare) := rfl
  rw [hL] at he
  cases hw with
  | obj hm hv =>
    rename_i m
    obtain ⟨hk, hvals⟩ := (FlockDeploy.jsonEquiv_obj_iff Std.TreeMap.Raw.WF.ofList hm).1 he
    have hD : [("law", Json.str "subset"), ("population", Lean.toJson pop), ("k", Lean.toJson us.size),
        ("units", Json.arr (us.map fun x => Lean.toJson x))].Pairwise
        (fun p q : String × Json => ¬ compare p.1 q.1 = .eq) := by simp
    have hU : (Std.TreeMap.Raw.ofList [("law", Json.str "subset"), ("population", Lean.toJson pop),
        ("k", Lean.toJson us.size), ("units", Json.arr (us.map fun x => Lean.toJson x))] compare).get? "units" =
        some (.arr (us.map fun x => Lean.toJson x)) := by
      rw [Std.TreeMap.Raw.get?_eq_getElem?]
      exact Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hD (by simp)
    have hs : (m.get? "units").isSome = true := by rw [← hk, hU]; rfl
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hs
    have hE := hvals "units" _ b hU hb
    have hg : _root_.Spot.getD (.obj m) "units" = b := by
      simp only [_root_.Spot.getD, _root_.Spot.field?, Json.getObjVal?, hb]
      rfl
    rw [hg]
    obtain ⟨f, hf⟩ := jsonForm_arr' (us.map fun x => Lean.toJson x)
    cases b with
    | arr bs =>
      obtain ⟨hsz, he'⟩ := (FlockDeploy.jsonEquiv_arr_iff _ _).1 hE
      congr 1
      refine Array.ext (by simpa using hsz.symm) fun i h₁ h₂ => ?_
      have := he' i h₂ h₁
      rw [Array.getElem_map] at this ⊢
      exact eq_of_jsonEquiv_nat this
    | obj m' =>
      rw [FlockDeploy.JsonEquiv, hf, FlockDeploy.jsonForm_obj] at hE
      cases hE
    | _ => rw [FlockDeploy.JsonEquiv, hf] at hE; cases hE
  | _ =>
    obtain ⟨f, hf⟩ : ∃ f, FlockDeploy.jsonForm (.obj (Std.TreeMap.Raw.ofList [("law", Json.str "subset"),
        ("population", Lean.toJson pop), ("k", Lean.toJson us.size),
        ("units", Json.arr (us.map fun x => Lean.toJson x))] compare)) = .obj f := ⟨_, FlockDeploy.jsonForm_obj _⟩
    rw [FlockDeploy.JsonEquiv, hf] at he
    cases he

/-- **The record's sample is its `unit_draw` rebuilt**: where the record holds a sample, the record parses, and the sample is
`norm` of its `unit_draw`, a well-formed value. -/
theorem recordDraw_raw {record : String} {dj : Json} (h : Flock.Witness.recordDraw record = some dj) :
    ∃ j d, Flock.JsonParse.parse record = .ok j ∧ _root_.Spot.getD j "unit_draw" = d ∧
      Claims.Json.Wf d ∧ dj = Flock.norm d := by
  unfold Flock.Witness.recordDraw at h
  split at h
  · rename_i j hj
    have hw := FlockComplete.JsonRT.parse_wf _ _ hj
    split at h
    · cases h
    · rename_i d _ hd
      refine ⟨j, d, hj, ?_, ?_, (Option.some.inj h).symm⟩
      · simp only [_root_.Spot.getD, _root_.Spot.field?, hd]
        rfl
      · cases hw with
        | obj hm hv => exact hv _ _ (FlockComplete.JsonRT.mem_of_getObjVal hm hd).2
        | _ => cases hd
    · cases h
  · cases h

theorem qsort_pairwise {xs : Array ℕ} (h : xs.toList.Pairwise (· < ·)) : xs.qsort (· < ·) = xs := by
  have hl : ∀ i j (hi : i < xs.size) (hj : j < xs.size), i < j → xs[i] < xs[j] := fun i j hi hj hij => by
    have := List.pairwise_iff_getElem.1 h i j (by simpa using hi) (by simpa using hj) hij
    simpa using this
  refine Verity.QSort.qsort_sorted _ _ fun i hi j hj => ?_
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · simp only [hl i j hi hj hij, hij, decide_true]
  · simp only [Nat.lt_irrefl, decide_false]
  · have := hl j i hj hi hij
    simp only [show ¬ xs[i] < xs[j] by omega, show ¬ i < j by omega, decide_false]

theorem ascending_lt {xs : Array Int} (h : _root_.Properties.Spot.Ascending xs) :
    ∀ i j (hi : i < xs.size) (hj : j < xs.size), i < j → xs[i] < xs[j] := by
  intro i j hi hj hij
  induction j with
  | zero => omega
  | succ j ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 hij with hlt | rfl
    · exact Int.lt_trans (ih (by omega) hlt) (h j hj)
    · exact h i hj

theorem qsort_ascending {xs : Array Int} (h : _root_.Properties.Spot.Ascending xs) :
    xs.qsort (· < ·) = xs := by
  refine Verity.QSort.qsort_sorted _ _ fun i hi j hj => ?_
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · simp only [ascending_lt h i j hi hj hij, hij, decide_true]
  · simp only [Int.lt_irrefl, Nat.lt_irrefl, decide_false]
  · have := ascending_lt h j i hj hi hij
    simp only [show ¬ xs[i] < xs[j] from Int.not_lt.2 (Int.le_of_lt this), show ¬ i < j by omega, decide_false]

theorem list_filterMap_int : ∀ {us : List Json} {is : List Int},
    us.mapM _root_.Spot.int? = some is → us.filterMap _root_.Spot.int? = is
  | [], is, h => by
    cases is with
    | nil => rfl
    | cons _ _ => simp at h
  | u :: us, is, h => by
    rw [List.mapM_cons] at h
    cases hu : _root_.Spot.int? u with
    | none => simp [hu] at h
    | some i =>
      cases hr : us.mapM _root_.Spot.int? with
      | none => simp [hu, hr] at h
      | some is' =>
        simp only [hu, hr, Option.bind_eq_bind, Option.bind_some, Option.pure_def, Option.some.injEq] at h
        subst h
        rw [List.filterMap_cons, hu, list_filterMap_int hr]

theorem filterMap_int {us : Array Json} {is : Array Int} (h : us.mapM _root_.Spot.int? = some is) :
    us.filterMap _root_.Spot.int? = is := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : us.toList.mapM _root_.Spot.int? with
  | none => simp [hl] at h
  | some l =>
    simp only [hl, Option.map_eq_map, Option.map_some, Option.some.injEq] at h
    apply Array.toList_inj.1
    rw [Array.toList_filterMap, list_filterMap_int hl, ← h]

theorem list_filterMap_nat : ∀ {us : List Json} {ps : List ℕ},
    us.mapM _root_.Spot.int? = some (ps.map fun n : ℕ => (n : Int)) →
      us.filterMap _root_.Spot.nat? = ps
  | [], ps, h => by
    cases ps with
    | nil => rfl
    | cons _ _ => simp at h
  | u :: us, ps, h => by
    rw [List.mapM_cons] at h
    cases hu : _root_.Spot.int? u with
    | none => simp [hu] at h
    | some i =>
      cases hr : us.mapM _root_.Spot.int? with
      | none => simp [hu, hr] at h
      | some is =>
        simp only [hu, hr, Option.bind_eq_bind, Option.bind_some, Option.pure_def] at h
        cases ps with
        | nil => cases h
        | cons p ps =>
          simp only [List.map_cons, Option.some.injEq, List.cons.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          have hn : _root_.Spot.nat? u = some p := by
            simp only [_root_.Spot.nat?, hu]
            simp
          rw [List.filterMap_cons, hn, list_filterMap_nat hr]

theorem filterMap_nat {us : Array Json} {ps : Array ℕ}
    (h : us.mapM _root_.Spot.int? = some (ps.map fun n : ℕ => (n : Int))) :
    us.filterMap _root_.Spot.nat? = ps := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : us.toList.mapM _root_.Spot.int? with
  | none => simp [hl] at h
  | some l =>
    simp only [hl, Option.map_eq_map, Option.map_some, Option.some.injEq] at h
    have hl' : us.toList.mapM _root_.Spot.int? = some (ps.toList.map fun n : ℕ => (n : Int)) := by
      rw [hl, ← Array.toList_map, ← h]
    have := list_filterMap_nat hl'
    apply Array.toList_inj.1
    rw [Array.toList_filterMap, this]

/-- **v1's window check at the coins `ξ` proves the units it drew**: where it accepts a sample without a closure, the
evidence's proved units are the sample's (`UnitsProved`, `Drawn`). -/
theorem checkAt_proved {p : _root_.Spot.OneStage.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStage.Evidence} {pub : _root_.Spot.OneStage.Public}
    {prof : _root_.Core.Integrity} (h : _root_.Spot.OneStage.checkAt p c (some ξ) e = .accept pub prof)
    (hc : _root_.Spot.field? e.draw "closure" = none) : drawnUnits e.draw = e.provedUnits := by
  unfold _root_.Spot.OneStage.checkAt at h
  cases hw : _root_.Spot.OneStage.coinsWhy p c e ξ with
  | some why =>
    have hb : (some ξ).bind (_root_.Spot.OneStage.coinsWhy p c e) = some why := hw
    rw [hb] at h; simp at h
  | none =>
    have hb : (some ξ).bind (_root_.Spot.OneStage.coinsWhy p c e) = none := hw
    rw [hb] at h
    have hck : _root_.Spot.OneStage.check p c e = .accept pub prof := h
    obtain ⟨_, _, _, -, hd, hu, -⟩ := _root_.Properties.Spot.acceptMeans _ _ _ _ _ hck
    dsimp only [_root_.Properties.Spot.Drawn] at hd
    obtain ⟨-, -, -, -, -, us, is, hus, hm, hasc, -⟩ := hd
    have hcl : _root_.Spot.getD e.draw "closure" = .null := by
      simp only [_root_.Spot.getD, hc, Option.getD_none]
    have hpr : _root_.Spot.Draw.proved e.draw = is := by
      unfold _root_.Spot.Draw.proved
      dsimp only
      rw [hus, hcl]
      dsimp only
      rw [Array.append_empty, filterMap_int hm]
      exact qsort_ascending hasc
    have hp := hu.2.1
    rw [hpr] at hp
    rw [← hp] at hm
    unfold drawnUnits
    rw [hus]
    exact filterMap_nat hm

/-- **`DrawHeld` from the record sample's units**: where, wherever composite `k` accepted what the prover sent and v2
drew `units` over `n` units there, the subset sample the record's decodes to is over `n` units at `units`, v2 holds the
record's sample to its own. v2 accepts only a subset record sample (`recordDraw_subset`), which is its own print
(`recordDrawn_of_subset`). -/
theorem drawHeld_of {k : Verity.Registration.Composite}
    (h : ∀ s dr L n ξ units record dj D kd, Verity.SameAccepts k s → Drew k s dr L n ξ units →
      sentRecord s = some record → Flock.Witness.recordDraw record = some dj →
      Flock.Draw.UnitDraw.ofJson dj = .ok D → D.law = .subset kd → D.population = n ∧ D.units = units) :
    DrawHeld k := by
  intro s dr L n ξ units record hS hd hr
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, -⟩ := id hd
  obtain ⟨dj, D, kd, hdj, hD, hl⟩ := recordDraw_subset hS hcase hacc hr
  obtain ⟨rfl, rfl⟩ := h s dr L n ξ units record dj D kd hS hd hr hdj hD hl
  exact recordDrawn_of_subset hdj hD hl

/-- **v2 holds the record's sample to its own**, for every composite `k`: the subset sample the record's decodes to is over
the record's population, since the session loads the public file to the sample's population (`setupTables_population`),
which v2 held to the record's (`population_held`); and it is at the units v2 drew, since the session proved the public
file's units the record samples (`provedUnits_eq`, at the indices the population check pins,
`Properties.Spot.FilePart.check_indices`), v2 held them to the units v1 proved (`provedHeld_ok`), and v1's
window check proves the units it drew (`checkAt_proved`). -/
theorem drawHeld (k : Verity.Registration.Composite) : DrawHeld k := by
  refine drawHeld_of fun s dr L n ξ units record dj D kd hS hd hr hdj hD hl => ?_
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, ⟨_, _, _, -, hn, -⟩, hu, -⟩ := hd
  obtain ⟨record', _, _, _, _, -, hr', -, hv⟩ := sent_verified hS hcase hacc
  obtain rfl := Option.some.inj (hr'.symm.trans hr)
  obtain ⟨_, _, _, pubB, _, -, -, -, hpub, -, hrec, hst, -, -⟩ := hv
  obtain ⟨-, -, hev, -, -, _, _, hck⟩ := hacc
  obtain ⟨r, pubB', hsz, hpub', hph, hrun, hall⟩ := checkBlobsAt_run hck
  obtain rfl := Except.ok.inj (hpub'.symm.trans hpub)
  rw [hdj] at hst
  obtain ⟨_, hss, -⟩ := setupAt_ok hst
  obtain ⟨hset, -⟩ := sessionAt_ok hss
  obtain ⟨_, pop, hlp, hpop⟩ := setupTables_population hset hD
  refine ⟨hpop.trans (population_held hph hlp hn), ?_⟩
  have hs1 := eq_single hsz
  rw [hs1] at hrun
  obtain ⟨_, _, pub1, _, -, -, hpub1, -, _, _, _, _, hstep, hw⟩ := run_verified hrun
  obtain rfl := Except.ok.inj (hpub1.symm.trans hpub')
  obtain ⟨recB, record'', hb, hrec'', hpc, hpu⟩ := step_proved hstep hall
  obtain rfl := Option.some.inj ((sessionRecord_eq hs1 hb hrec'').symm.trans hrec)
  obtain ⟨_, _, _, hd, -, -, -, hhd, hidx⟩ := _root_.Properties.Spot.FilePart.check_indices hpc
  obtain ⟨j, d, hj, hjd, hwd, hdn⟩ := recordDraw_raw hdj
  obtain ⟨hasc, hdj'⟩ :=
    ofJson_subset_eq (Claims.Bridge.Custody.recordDraw_norm ((exec_recordDraw _).trans hdj)) hD hl
  have hud : _root_.Spot.getD (_root_.Spot.getD j "unit_draw") "units" =
      .arr (D.units.map fun x => Lean.toJson x) := by
    rw [hjd]
    exact units_of_norm hwd (hdn.symm.trans hdj')
  have hpr := provedUnits_eq hhd hidx hj hud hpu
  have hq := provedHeld_ok (hw _ rfl hall)
  have hsorted := qsort_pairwise (unitsAscending_spec hasc).2
  rw [hpr, hsorted, hsorted] at hq
  obtain ⟨_, _, _, hat⟩ := checkBlobsAt_checkAt hck
  have hv1 := checkAt_proved hat (evidenceOf_closure hev)
  exact hq.trans (hv1.symm.trans hu.symm)

end Verity.Sampled
