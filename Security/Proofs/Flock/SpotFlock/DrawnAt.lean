module

public import Proofs.Flock.SpotFlock.Depths
public import Proofs.Flock.SpotFlock.SlotsAt
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Claims.Complete.Json.RoundTrip
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk
import Std.Data.TreeMap.Raw.Lemmas
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# A checked unit's session draws it

`CheckedRun.counted_of`: at an accepted run's checked unit that isn't always checked, whose session has one table, the
unit's instance (`rankAt`) is below the table's count (`Counted`), where the drawn slot that names the unit is in its
template's part of the second stage's population.

The unit is checked because the second stage drew a slot whose proof block it is (`RegTwo.checked`). Round 2 holds the
session's sample to the verifier's own draw within the template's part (`second-sample`, `SecondStage.ownDraw`) by
canonical text (`jeq`), and two printable values with one canonical text are equivalent, the parser reading `canon`'s
text back to each (`jsonEquiv_of_jeq`). The session sets up with its statement's sample, normalized
(`Raw.ins_draw`), so its sample's units are the drawn slots in the part, counted from its first (`units_of_equiv`).
Its first table's instances are the units the sample proves (`Layout.drawn_spec`), and the unit's instance is the first
of them whose proof block is the unit (`CheckedRun.drawnAt`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace DrawnAt

/-- A printable value is well formed. -/
theorem wf_of_good {j : Json} (h : Claims.Json.Good j) : Claims.Json.Wf j := by
  induction h with
  | null => exact .null
  | bool b => exact .bool b
  | num m => exact .num _
  | str s => exact .str s
  | arr _ ih => exact .arr ih
  | obj hm _ ih => exact .obj hm ih

/-- **A well-formed value `canon` prints is printable**: its numbers are integers, the only ones `canon` prints. -/
theorem good_of_canon : ∀ j : Json, Claims.Json.Wf j → ∀ s, _root_.Flock.canon j = .ok s → Claims.Json.Good j := by
  refine FlockDeploy.json_ind ?_ ?_ ?_ ?_ ?_ ?_
  · intro _ _ _; exact .null
  · intro b _ _ _; exact .bool b
  · intro n _ s h
    rw [_root_.Flock.canon] at h
    split at h
    · rename_i he
      obtain ⟨m, e⟩ := n
      have he' : e = 0 := by simpa using he
      subst he'
      exact .num m
    · cases h
  · intro t _ _ _; exact .str t
  · intro a ih hw s h
    cases hw with
    | arr hw =>
      rw [_root_.Flock.canon] at h
      cases hp : a.attach.mapM (fun x => _root_.Flock.canon x.1) with
      | error e => rw [hp] at h; cases h
      | ok parts =>
        refine .arr fun x hx => ?_
        obtain ⟨y, -, hy⟩ := (mapM_ok_mem hp).2 ⟨x, hx⟩ (Array.mem_attach a ⟨x, hx⟩)
        exact ih x hx (hw x hx) y hy
  · intro m ih hw s h
    cases hw with
    | obj hm hw =>
      rw [_root_.Flock.canon] at h
      cases hp : m.toArray.attach.mapM
          (fun p => do pure (p.1.1, ← _root_.Flock.canon p.1.2) : _ → _root_.Flock.V (String × String)) with
      | error e => rw [hp] at h; cases h
      | ok vals =>
        refine .obj hm fun k v hkv => ?_
        have hmem : (k, v) ∈ m.toList := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).2 hkv
        have hmemA : (k, v) ∈ m.toArray := by
          rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray]; exact hmem
        obtain ⟨y, -, hy⟩ := (mapM_ok_mem hp).2 ⟨(k, v), hmemA⟩ (Array.mem_attach _ _)
        cases hc : _root_.Flock.canon v with
        | error e => simp [hc, bind, Except.bind] at hy
        | ok sv => exact ih (k, v) hmem (hw k v hkv) sv hc

/-- **Two printable values with one canonical text are equivalent**: the parser reads `canon`'s text back to each
(`JsonRT.parse_canon`). -/
theorem jsonEquiv_of_jeq {a b : Json} (ha : Claims.Json.Good a) (hb : Claims.Json.Good b)
    (h : Spot.jeq a b = true) : FlockDeploy.JsonEquiv a b := by
  unfold Spot.jeq at h
  cases hca : _root_.Flock.canon a with
  | error e => simp [hca] at h
  | ok s =>
    cases hcb : _root_.Flock.canon b with
    | error e => simp [hca, hcb] at h
    | ok t =>
      simp only [hca, hcb, beq_iff_eq] at h
      subst h
      obtain ⟨a', hpa, hsa⟩ := FlockComplete.JsonRT.parse_canon a s ha hca
      obtain ⟨b', hpb, hsb⟩ := FlockComplete.JsonRT.parse_canon b s hb hcb
      rw [hpa] at hpb
      cases hpb
      exact hsa.jsonEquiv.trans hsb.jsonEquiv.symm

/-- A value `jeq` compares has a canonical text. -/
theorem canon_of_jeq {a b : Json} (h : Spot.jeq a b = true) : ∃ s, _root_.Flock.canon a = .ok s := by
  unfold Spot.jeq at h
  cases hca : _root_.Flock.canon a with
  | error e => simp [hca] at h
  | ok s => exact ⟨s, rfl⟩

/-- **The verifier's own draw is printable.** -/
theorem good_ownDraw (q : ℕ) (p : Spot.SecondStage.Part) (drawn : Array ℕ) :
    Claims.Json.Good (Spot.SecondStage.ownDraw q p drawn) := by
  simp only [Spot.SecondStage.ownDraw]
  refine FlockComplete.JsonRT.good_mkObj (by simp) fun e he => ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at he
  rcases he with rfl | rfl | rfl | rfl
  · exact .str _
  · exact FlockComplete.JsonRT.good_toJson _
  · exact FlockComplete.JsonRT.good_toJson _
  · refine FlockComplete.JsonRT.good_arr_of fun x hx => ?_
    obtain ⟨y, -, rfl⟩ := Array.mem_map.1 hx
    exact FlockComplete.JsonRT.good_toJson y

/-- **The verifier's own draw's units**, at its key. -/
theorem ownDraw_units (q : ℕ) (p : Spot.SecondStage.Part) (drawn : Array ℕ) :
    (Spot.SecondStage.ownDraw q p drawn).getObjVal? "units" =
      .ok (Spot.natsJson ((drawn.filter (p.holds q)).map (· - p.lo))) := by
  simp only [Spot.SecondStage.ownDraw]
  exact FlockComplete.JsonRT.getObjVal?_mkObj (by simp) (by simp)

-- `walk_step` down the unfolded do-block of `ofJson`, the sample's decoder, a goal per branch
set_option maxHeartbeats 2000000 in
/-- **A sample's units are its `units` field's**, as natural numbers. -/
theorem ofJson_units {d : Json} {D : _root_.Flock.Draw.UnitDraw} (h : _root_.Flock.Draw.UnitDraw.ofJson d = .ok D) :
    ∃ a, _root_.Flock.J.arr d "units" "unit_draw" = .ok a ∧
      a.mapM (_root_.Flock.J.asNat · "U2: a drawn unit") = .ok D.units := by
  unfold _root_.Flock.Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, ‹_›, ‹_›⟩

/-- `Json.getArr?` reads only an array. -/
theorem arr_of_getArr {v : Json} {a : Array Json} (h : v.getArr? = .ok a) : v = .arr a := by
  cases v <;> first | (cases h; rfl) | cases h

/-- `J.arr` reads an array at its key. -/
theorem getObjVal_of_arr {d : Json} {k ctx : String} {a : Array Json} (h : _root_.Flock.J.arr d k ctx = .ok a) :
    d.getObjVal? k = .ok (.arr a) := by
  unfold _root_.Flock.J.arr at h
  obtain ⟨v, hv, ha⟩ := FlockSoundness.Refine.ebind_ok h
  have hg : d.getObjVal? k = .ok v := by
    unfold _root_.Flock.J.get at hv
    split at hv
    · cases hv; assumption
    · cases hv
  have hva : v.getArr? = .ok a := by
    unfold _root_.Flock.J.asArr at ha
    split at ha
    · cases ha; assumption
    · cases ha
  rw [hg, arr_of_getArr hva]

/-- `J.asNat` reads back the numbers `toJson` writes. -/
theorem mapM_asNat_toJson (ctx : String) :
    ∀ l : List ℕ, (l.map Lean.toJson).mapM (_root_.Flock.J.asNat · ctx) = .ok l
  | [] => rfl
  | m :: l => by
    rw [List.map_cons, List.mapM_cons, mapM_asNat_toJson ctx l]
    rfl

/-- **An array equivalent to `natsJson us` reads as `us`.** -/
theorem nats_of_equiv {a : Array Json} {us : Array ℕ} (he : FlockDeploy.JsonEquiv (.arr a) (Spot.natsJson us))
    {ctx : String} {ns : Array ℕ} (hm : a.mapM (_root_.Flock.J.asNat · ctx) = .ok ns) : ns = us := by
  obtain ⟨hsz, hel⟩ := (FlockDeploy.jsonEquiv_arr_iff a (us.map Lean.toJson)).1 he
  have hau : a = us.map Lean.toJson := by
    refine Array.ext hsz fun i h₁ h₂ => ?_
    have := hel i h₁ h₂
    simp only [Array.getElem_map] at this ⊢
    exact Depths.num_of_jsonEquiv this.symm
  rw [hau, Array.mapM_eq_mapM_toList, Array.toList_map, mapM_asNat_toJson] at hm
  simp only [Functor.map, Except.map, Except.ok.injEq] at hm
  subst hm
  rfl

/-- **A sample equivalent to the verifier's own draw has its units**: the drawn slots in the part, counted from its
first. -/
theorem units_of_equiv {d : Json} (hw : Claims.Json.Wf d) {q : ℕ} {p : Spot.SecondStage.Part} {drawn : Array ℕ}
    (he : FlockDeploy.JsonEquiv d (Spot.SecondStage.ownDraw q p drawn)) {D : _root_.Flock.Draw.UnitDraw}
    (hD : _root_.Flock.Draw.UnitDraw.ofJson d = .ok D) :
    D.units = (drawn.filter (p.holds q)).map (· - p.lo) := by
  obtain ⟨a, ha, hm⟩ := ofJson_units hD
  obtain ⟨w, hw', hvw, -⟩ :=
    HeaderRoots.getObjVal_equiv hw (wf_of_good (good_ownDraw q p drawn)) he (getObjVal_of_arr ha)
  rw [ownDraw_units] at hw'
  cases hw'
  exact nats_of_equiv hvw hm

/-- `Stmt.ofJson?` reads its sample at `draw`. -/
theorem ofJson?_draw {j : Json} {S : Spot.SecondStage.Stmt} (hS : Spot.SecondStage.Stmt.ofJson? j = some S) :
    Spot.getD j "draw" = S.draw := by
  unfold Spot.SecondStage.Stmt.ofJson? at hS
  split at hS
  · cases hS
  · split at hS <;> cases hS <;> assumption

/-- A value the deployed statement parses is well formed. -/
theorem wf_of_jsonOf {what : String} {b : ByteArray} {j : Json} (h : FlockDeploy.jsonOf what b = .ok j) :
    Claims.Json.Wf j := by
  unfold FlockDeploy.jsonOf at h
  split at h
  · cases h
  · rename_i s _
    cases hp : _root_.Flock.JsonParse.parse s with
    | error e => rw [hp] at h; cases h
    | ok j' =>
      rw [hp] at h
      cases h
      exact FlockComplete.JsonRT.parse_wf s _ hp

end DrawnAt

variable (h : Held)

/-- **Round 2's statement reads the session statement's sample**: the JSON its `draw` bytes parse to. -/
theorem stmtJsonOf_draw {t : ℕ} {w : FlockDeploy.Raw} {S : Spot.SecondStage.Stmt} (hS : stmtJsonOf h t w = some S) :
    ∃ b, w.draw = some b ∧ FlockDeploy.jsonOf "draw" b = .ok S.draw := by
  unfold stmtJsonOf at hS
  obtain ⟨hdr, -, hS⟩ := Option.bind_eq_some_iff.1 hS
  obtain ⟨b, hb, hS⟩ := Option.bind_eq_some_iff.1 hS
  obtain ⟨dr, hdr', hS⟩ := Option.bind_eq_some_iff.1 hS
  have key : ∀ r, Spot.SecondStage.Stmt.ofJson? (Json.mkObj [("draw", dr), ("header", hdr), ("registered", r),
      ("template", Json.str (h.names.getD t ""))]) = some S → dr = S.draw := by
    intro r hr
    have h1 := DrawnAt.ofJson?_draw hr
    have hd : FlockComplete.JsonRT.Distinct
        [("draw", dr), ("header", hdr), ("registered", r), ("template", Json.str (h.names.getD t ""))] := by simp
    rwa [getD_mkObj hd (k := "draw") (v := dr) (by simp)] at h1
  have hdS : dr = S.draw := by
    dsimp only at hS
    split at hS
    · obtain ⟨r, -, hS⟩ := Option.bind_eq_some_iff.1 hS
      exact key r hS
    · exact key .null hS
  refine ⟨b, hb, ?_⟩
  cases hj : FlockDeploy.jsonOf "draw" b with
  | error e => rw [hj] at hdr'; cases hdr'
  | ok j => rw [hj] at hdr'; cases hdr'; rw [hdS]

/-- **Round 1's record lists the replay sample** where template `t`'s session statement reads the run's. -/
theorem readsOk_order {t : ℕ} {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} (hr : readsOk h t x m₁ xS c₀ c₁ = true)
    {st : Roots} {rec draw : Json} (hst : decode x = some st) (hrec : jsonMsg m₁ = some rec)
    (hdraw : secondDraw h c₀ c₁ = some draw) :
    Spot.SecondStage.interiorOrderOk (ownOf h st rec draw c₀) = true := by
  unfold readsOk at hr
  rw [hst, hrec, hdraw] at hr
  cases hw : FlockDeploy.decode xS with
  | none => rw [hw] at hr; cases hr
  | some w =>
    rw [hw] at hr
    simp only [Bool.and_eq_true] at hr
    exact hr.1.1

namespace CheckedRun

/-- **A checked unit's one-table session draws its template's part of the second stage's sample**: in an accepted
run, the second stage drew `draw` (`secondDraw`) and round 1's record lists the replay sample (`blocksOf`); the units
the session's sample proves (`drawnOf`) are its first table's instances (`Layout.drawn_spec`); and its sample's units
are the drawn slots in the unit's template's part, counted from its first (`units_of_equiv`). -/
theorem drawnAt {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) :
    ∃ draw rec (D : _root_.Flock.Draw.UnitDraw), secondDraw h r.c₀ r.c₁ = some draw ∧ jsonMsg r.m₁ = some rec ∧
      blocksOf rec = (h.reg.sample (src r.c₀)).toList ∧
      drawnOf ((FlockDeploy.readOf true (r.xS h)).bind (·.draw)) = D.proved.toList ∧
      r.sts[0]!.pub.n = D.proved.size ∧
      D.units = (((Spot.Draw.proved draw).map Int.toNat).filter
          ((partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).holds (h.reg.sample (src r.c₀)).size)).map
        (· - (partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).lo) := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  have hsc := sessionScoped h r.x r.m₁ (unitStmt h r.m₂ r.u) r.c₀ r.c₁ _ hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  have hio := readsOk_order h hat.2 hst hrec hdraw
  obtain ⟨-, -, -, -, -, hjeq, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hwi := ins_of_setupOf hw hset
  obtain ⟨b, hb, hjo⟩ := stmtJsonOf_draw h hS
  obtain ⟨draw', hdm, hdr⟩ := FlockDeploy.Raw.ins_draw hwi
  rw [hb, Option.mapM_some] at hdm
  obtain ⟨dr, hdr', rfl⟩ := Depths.someMap_ok hdm
  rw [hjo] at hdr'
  cases hdr'
  have hrd : r.ins.read.draw = some (_root_.Flock.norm S.draw) := hdr
  have hwfS := DrawnAt.wf_of_jsonOf hjo
  obtain ⟨s, hcs⟩ := DrawnAt.canon_of_jeq hjeq
  have he := (FlockDeploy.jsonEquiv_norm S.draw hwfS).trans
    (DrawnAt.jsonEquiv_of_jeq (DrawnAt.good_of_canon S.draw hwfS s hcs) (DrawnAt.good_ownDraw _ _ _) hjeq)
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
  obtain ⟨pop, -, -, hsome⟩ := (FlockSoundness.Discharge.Hidden.setupHidden_spec' hd).pub
  obtain ⟨D, hD, -, hn, -, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec (hsome _ hrd)
  have hread : FlockDeploy.readOf true (unitStmt h r.m₂ r.u) = some r.ins.read := by
    simp [FlockDeploy.readOf, hins]
  refine ⟨draw, rec, D, hdraw, hrec, blocksOf_of_interiorOrderOk hio, ?_, ?_,
    DrawnAt.units_of_equiv (FlockDeploy.wf_norm S.draw) he hD⟩
  · show drawnOf ((FlockDeploy.readOf true (unitStmt h r.m₂ r.u)).bind (·.draw)) = _
    rw [hread]
    simp only [Option.bind_some, hrd, drawnOf, hD]
  · rw [getElem!_pos r.sts 0 hpos]
    exact hn

/-- **`Counted` where the drawn slot naming the unit is in its template's part**: at an accepted run's checked unit
that isn't always checked, whose session has one table, the second stage drew a slot whose proof block is the unit
(`RegTwo.checked`); where that slot is in the unit's template's part, the session's sample proves it
(`drawnAt`), and the unit's instance, the first proved unit whose proof block is the unit (`rankAt`), is below the
table's count. -/
theorem counted_of {r : CheckedRun} (hr : r.Ok h) (hc1 : r.ins.read.I.count ≤ 1) (ha : h.reg.always r.u = false)
    (hfit : ∀ draw, secondDraw h r.c₀ r.c₁ = some draw → ∀ j ∈ Spot.Draw.proved draw,
      h.reg.blockAt (h.reg.sample (src r.c₀)) j.toNat = r.u →
      (partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).holds (h.reg.sample (src r.c₀)).size
        j.toNat = true) :
    r.Counted h := by
  obtain ⟨draw, rec, D, hdraw, hrec, hbl, hdr, hn, hun⟩ := drawnAt h hr hc1
  obtain ⟨-, -, -, -, -, -, -, -, hu, -⟩ := hr
  have hsel : ∃ rd, Spot.Select.selectSecond (h.reg.sample (src r.c₀)) h.reg.ns h.reg.ks (src r.c₁) =
      .ok (draw, rd) := by
    unfold secondDraw at hdraw
    split at hdraw
    · rename_i draw' rd heq
      cases hdraw
      exact ⟨rd, heq⟩
    · cases hdraw
  obtain ⟨rd, hsel⟩ := hsel
  have hu' : r.u ∈ (List.range (unitCount h r.x)).filter fun v => h.reg.always v ||
      (Spot.Draw.proved draw).any fun j => h.reg.blockAt (h.reg.sample (src r.c₀)) j.toNat == v := by
    have := hu
    unfold RegTwo.checked at this
    rw [hsel] at this
    exact this
  obtain ⟨-, hany⟩ := List.mem_filter.1 hu'
  rw [ha, Bool.false_or] at hany
  obtain ⟨j, hj, hju⟩ := Array.any_eq_true'.1 hany
  have hju' : h.reg.blockAt (h.reg.sample (src r.c₀)) j.toNat = r.u := beq_iff_eq.1 hju
  have hP := hfit draw hdraw j hj hju'
  have hlo : (partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).lo ≤ j.toNat := by
    simp only [Spot.SecondStage.Part.holds, Bool.and_eq_true, decide_eq_true_eq] at hP
    exact hP.1
  have hmemU : j.toNat - (partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).lo ∈ D.units := by
    rw [hun]
    exact Array.mem_map.2 ⟨j.toNat, Array.mem_filter.2 ⟨Array.mem_map.2 ⟨j, hj, rfl⟩, hP⟩, rfl⟩
  have hmemP : j.toNat - (partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).lo ∈ D.proved := by
    unfold _root_.Flock.Draw.UnitDraw.proved
    split
    · exact hmemU
    · exact (FlockSoundness.Audit.DrawExec.mem_qsort _ _ _).2 (Array.mem_append_left _ hmemU)
  show rankAt h [] r.m₁ (r.xS h) r.u < r.sts[0]!.pub.n
  unfold rankAt
  simp only [hrec, Option.map_some, Option.getD_some, hbl, Array.toArray_toList, hdr, hn]
  rw [← Array.length_toList]
  refine List.findIdx_lt_length_of_exists ⟨_, Array.mem_toList_iff.2 hmemP, ?_⟩
  show (h.reg.blockAt (h.reg.sample (src r.c₀))
    ((partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).lo +
      (j.toNat - (partAt h (h.reg.sample (src r.c₀)).size (templateOf h.ms r.u)).lo)) == r.u) = true
  rw [Nat.add_sub_of_le hlo, hju', beq_self_eq_true]

end CheckedRun

end SpotFlock
