module

public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Claims.Bridge.Custody
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# What v2 holds of the record's draw

The record's draw (`recordDraw_firstStop`, `MovesOf`'s clause (i)) reads v2's log at its coins (`DrawReadAll`) and
v2's hold on the record's `unit_draw` (`DrawHeld`). This module gives what v2's code gives of them.

* **v2's log at its coins** (`drawReadAll_coinsRead`): `coinsRead` is `logRead` (`coinsRead_eq_logRead`), whose read
  shows the draw wherever v2 accepts (`drawReadAll_logRead`).
* **v2 accepts only a subset draw** (`recordDraw_subset`): where v2's case accepts, the record v2 reads holds a draw,
  since `verify --zk` refuses a session without one (`verdictOf_drawn`); its law is one a soundness claim draws by at
  `--zk` (`ProvedScope.drawLaw`, `lawWhy_zk`), the subset or the stratified law; and the session v2 sets up has no
  `--stratified` input, so `Stmt.setupTables` refuses a stratified draw (`setupTables_stratified`). So the record's draw
  is the `.subset` print of the member's share of the window's draw under the verifier's law, which `DrawHeld` reads at
  the units v2 drew; the bound is that law's, from v1's window check (`SampledProofs.OneStage.checkAt`), not the
  print's.
* **A subset draw that decodes is its own print** (`ofJson_subset_eq`): a value `norm` fixes that `UnitDraw.ofJson`
  decodes to a subset draw is an object of exactly `law`, `population`, `k` and `units` (`obj_eq_mkObj4`), so it is the
  draw's print at no closure, its units ascending below its population. The record's draw is rebuilt
  (`Claims.Bridge.Custody.recordDraw_norm`), so it is its decoded draw's subset print (`recordDrawn_of_subset`).
* **`DrawHeld` from the record draw's units** (`drawHeld_of`): wherever the subset draw the record's decodes to is over
  v2's `n` units at the units v2 drew, `DrawHeld` holds.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## v2's log at its coins -/

/-- **v2's log at its coins shows the draw** wherever v2 accepts: `coinsRead` is `logRead`. -/
theorem drawReadAll_coinsRead (k : Verity.Registration.Composite) : DrawReadAll k coinsRead :=
  coinsRead_eq_logRead ▸ drawReadAll_logRead k

/-! ## v2 accepts only a subset draw -/

/-- **`verify --zk` refuses a session without a draw**: where a session's verdict passes, its record's draw is some. -/
theorem verdictOf_drawn {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit}
    {coinKeys : Array String} {record : String} {draw : Option Json} {st : _root_.SampledProofs.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.SampledProofs.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) : draw.isSome = true := by
  unfold _root_.SampledProofs.OneStageZk.verdictOf at h
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

/-- **One session's `step` from the empty run, where every session accepted, reads a record that holds a draw.** -/
theorem step_drawn {I : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.SampledProofs.OneStageZk.RunFiles} {r : _root_.SampledProofs.OneStageZk.Run}
    (h : _root_.SampledProofs.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) :
    ∃ recB record, bs.get s.record s!"session {s.name}'s record" = .ok recB ∧
      String.fromUTF8? recB = some record ∧ (Flock.Witness.recordDraw record).isSome = true := by
  unfold _root_.SampledProofs.OneStageZk.step at h
  obtain ⟨recB, hb, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · rename_i record hrec
    exact ⟨recB, record, hb, hrec, verdictOf_drawn (push_ok h hall)⟩
  · cases h

/-- **Where v2's check accepts, the session's record holds a draw.** -/
theorem checkBlobsAt_drawn {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {ξ : ByteArray} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.SampledProofs.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.SampledProofs.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ record, sessionRecord e bs = some record ∧ (Flock.Witness.recordDraw record).isSome = true := by
  obtain ⟨r, hsz, hrun, hall⟩ := checkBlobsAt_run h
  have hs := eq_single hsz
  rw [hs] at hrun
  obtain ⟨_, _, _, _, -, -, -, -, _, _, _, _, _, hstep⟩ := run_verified hrun
  obtain ⟨recB, record, hb, hrec, hd⟩ := step_drawn hstep hall
  exact ⟨record, sessionRecord_eq hs hb hrec, hd⟩

/-- **`setupAt` at a draw holds the draw's law** (`ProvedScope.drawLaw` at `--zk`). -/
theorem setupAt_drawLaw {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads} {dj : Json}
    {x : Flock.SessionSpec × Array Flock.Setup × Json × Json}
    (h : _root_.SampledProofs.OneStageZk.setupAt W rd (some dj) = .ok x) :
    Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags W) false (some dj) = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.setupAt at h
  obtain ⟨⟨_, _⟩, -, h⟩ := ebind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, hu, -⟩ := ebind_ok h
  simpa only [Option.isSome_some, ↓reduceIte] using hu

/-- **A draw `drawLaw` passes decodes**, to a draw whose law `lawWhy` doesn't refuse. -/
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

/-- **The laws a soundness claim draws by at `--zk`**, at public tables: the subset and the stratified law. -/
theorem lawWhy_zk {tags : Flock.Tags} {D : Flock.Draw.UnitDraw}
    (h : Flock.ProvedScope.lawWhy true tags false D = none) :
    (∃ kd, D.law = .subset kd) ∨ ∃ K St, D.law = .stratified K St := by
  unfold Flock.ProvedScope.lawWhy at h
  cases hl : D.law with
  | subset kd => exact .inl ⟨kd, rfl⟩
  | stratified K St => exact .inr ⟨K, St, rfl⟩
  | bernoulli a b => simp [hl] at h
  | work K ws => simp [hl] at h

/-- **A session without `--stratified` refuses a stratified draw**: `Stmt.setupTables` holds a stratified draw's law to
the verifier's own `K`, and refuses it without one. -/
theorem setupTables_stratified {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {dj : Json} {partition : Option Json} {program : Option ByteArray}
    {r : Flock.SessionSpec × Array Flock.Stmt} {D : Flock.Draw.UnitDraw} {K : ℕ} {St : Array Flock.Draw.Stratum}
    (h : Flock.Stmt.setupTables tags cf pf coins count tables (some dj) partition program = .ok r)
    (hD : Flock.Draw.UnitDraw.ofJson dj = .ok D) (hl : D.law = .stratified K St) : False := by
  unfold Flock.Stmt.setupTables at h
  simp only [hD, hl, Except.map] at h
  cases partition <;> cases program <;> simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- **v2 accepts only a subset draw**: where composite `k` accepted what the prover sent and v2's case in it accepts,
the record v2 reads holds a draw (`unit_draw`), which decodes to a draw by the subset law. -/
theorem recordDraw_subset {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray} {dr : _root_.Properties.Challenger.Draws}
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

/-! ## A subset draw that decodes is its own print -/

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

/-- **A subset draw's object is its print**: an object as `obj_eq_mkObj4` reads it, at the law `subset`, the population
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
/-- **A subset draw that decodes is its own print**: a value `norm` fixes that decodes (`UnitDraw.ofJson`) to a draw
`D` by the subset law has `D`'s units ascending below its population, and is `D`'s print at no closure. -/
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

/-- **A subset record draw is its own print**: where the record's draw (`recordDraw`) decodes to a draw `D` by the
subset law, the record's draw is `D`'s subset print (`recordDrawn`). -/
theorem recordDrawn_of_subset {record : String} {dj : Json} {D : Flock.Draw.UnitDraw} {kd : ℕ}
    (hr : Flock.Witness.recordDraw record = some dj) (hD : Flock.Draw.UnitDraw.ofJson dj = .ok D)
    (hl : D.law = .subset kd) : recordDrawn (.subset D.units.size) D.population D.units record := by
  obtain ⟨ha, he⟩ :=
    ofJson_subset_eq (Claims.Bridge.Custody.recordDraw_norm ((exec_recordDraw record).trans hr)) hD hl
  exact ⟨ha, by rw [hr, ← he]⟩

end Print

/-- **`DrawHeld` from the record draw's units**: where, wherever composite `k` accepted what the prover sent and v2
drew `units` over `n` units there, the subset draw the record's decodes to is over `n` units at `units`, v2 holds the
record's draw to its own. v2 accepts only a subset record draw (`recordDraw_subset`), which is its own print
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

end Verity.Sampled
