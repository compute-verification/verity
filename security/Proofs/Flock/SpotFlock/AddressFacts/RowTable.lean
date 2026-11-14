module

public import Proofs.Flock.SpotFlock.AddressFacts.RowOf
import Proofs.Flock.Claims.CanonEq
import Proofs.Flock.Claims.Complete.Json.Parse
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Claims.JsonEq
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Layout.QSort
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk
import Proofs.Flock.SpotFlock.AddressFacts.RowDraw
import Proofs.Flock.SpotFlock.Depths
import Proofs.Flock.SpotFlock.EntriesAt
import Proofs.Flock.SpotFlock.HeaderRoots
import Proofs.Flock.SpotFlock.Opens
import Proofs.Flock.SpotFlock.ScopeTable
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The labels of a checked inner block's session

`AddressRow.Sits h` (`RowOf`) asks, at an accepted run, that the session's instances be labelled by their slots
(`LabelledAt` at `drawSlot`). This file proves that part, `CheckedRun.labelled_of_ok`, from what the second stage
checks of the session's statement and what Flock's setup computes of its public file:

* the session is one table, at a table count of at most one (`CheckedRun.one_table`): the statement check holds a
  checked unit's statement to its template's scope, where the design evaluates only at a one-table reference statement
  (`Verity.SampledTwo.oneTable_of_sessionOk`);
* the session's one statement is set up at the statement's sample (`setupTables_setupOne` at one table) and its public
  file is the loaded population's sampled sample (`setupHidden_spec'`), whose header's `units.indices` are the
  population's indices at the sampled proof blocks (`drawn_units`, `unitIndices_of_units`);
* the second stage's population check holds the raw header's indices to the part's slots and its instance count to
  the part's (`populationOk`), and Flock loads the header normalized, an equivalent value (`slots_of_population`);
* the second stage's sample check holds the statement's sample to its own subset sample, which the normalized sample Flock
  reads parses to (`proved_of_wf`), so each sampled proof block is below the part's count.

`drawn_of_ok` gives the sampled proof blocks as the second sample's units in the part, which the rest of `Sits` reads.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

/-- **A statement's JSON keeps its sample**. -/
theorem ofJson?_draw {j : Json} {S : Spot.SecondStage.Stmt} (hS : Spot.SecondStage.Stmt.ofJson? j = some S) :
    Spot.getD j "draw" = S.draw := by
  unfold Spot.SecondStage.Stmt.ofJson? at hS
  split at hS
  · cases hS
  · split at hS <;> cases hS <;> assumption

/-- **The second stage's statement of a session holds the session's sample**, parsed. -/
theorem stmtJsonOf_draw {h : Held} {t : ℕ} {w : FlockDeploy.Raw} {S : Spot.SecondStage.Stmt}
    (hS : stmtJsonOf h t w = some S) : ∃ b, w.draw = some b ∧ FlockDeploy.jsonOf "draw" b = .ok S.draw := by
  unfold stmtJsonOf at hS
  obtain ⟨hdr, -, hS⟩ := Option.bind_eq_some_iff.1 hS
  obtain ⟨b, hb, hS⟩ := Option.bind_eq_some_iff.1 hS
  obtain ⟨dr, hdr', hS⟩ := Option.bind_eq_some_iff.1 hS
  have key : ∀ r, Spot.SecondStage.Stmt.ofJson? (Json.mkObj [("draw", dr), ("header", hdr), ("registered", r),
      ("template", Json.str (h.names.getD t ""))]) = some S → dr = S.draw := by
    intro r hr
    have hd : FlockComplete.JsonRT.Distinct
        [("draw", dr), ("header", hdr), ("registered", r), ("template", Json.str (h.names.getD t ""))] := by simp
    have h1 := ofJson?_draw hr
    rwa [getD_mkObj hd (k := "draw") (v := dr) (by simp)] at h1
  dsimp only at hS
  have hdS : dr = S.draw := by
    split at hS
    · obtain ⟨r, -, hS⟩ := Option.bind_eq_some_iff.1 hS
      exact key r hS
    · exact key .null hS
  refine ⟨b, hb, ?_⟩
  cases hj : FlockDeploy.jsonOf "draw" b with
  | error e => simp [hj, Except.toOption] at hdr'
  | ok v =>
    rw [hj] at hdr'
    simp only [Except.toOption, Option.some.injEq] at hdr'
    rw [hdr', hdS]

/-- **A JSON file a statement names is well formed**: the parser builds well-formed trees (`JsonRT.parse_wf`). -/
theorem wf_of_jsonOf {what : String} {b : ByteArray} {j : Json} (h : FlockDeploy.jsonOf what b = .ok j) :
    Claims.Json.Wf j := by
  unfold FlockDeploy.jsonOf at h
  split at h
  · cases h
  · cases hp : _root_.Flock.JsonParse.parse _ with
    | error e => rw [hp] at h; cases h
    | ok v =>
      rw [hp] at h
      simp only [Except.mapError, Except.ok.injEq] at h
      subst h
      exact FlockComplete.JsonRT.parse_wf _ _ hp

/-- **A header's `units` entry gives its indices back** (`Registered.unitIndices`). -/
theorem unitIndices_of_units {pub : _root_.Flock.Public} {c : _root_.Flock.Circuit} {xs : Array ℕ}
    (h : pub.header.getObjVal? "units" = .ok (_root_.Flock.HmRow.unitsEntry c xs)) :
    _root_.Flock.Registered.unitIndices pub = .ok xs := by
  have hi : (_root_.Flock.HmRow.unitsEntry c xs).getObjVal? "indices" =
      .ok (.arr (xs.map fun x => Lean.toJson x)) := by
    unfold _root_.Flock.HmRow.unitsEntry
    dsimp only
    split <;> exact FlockComplete.JsonRT.getObjVal?_mkObj (by simp) (by simp)
  have hg : _root_.Flock.J.get pub.header "units" "public file" = .ok (_root_.Flock.HmRow.unitsEntry c xs) := by
    simp [_root_.Flock.J.get, h]
  have ha : _root_.Flock.J.arr (_root_.Flock.HmRow.unitsEntry c xs) "indices" "public file" =
      .ok (xs.map fun x => Lean.toJson x) := by
    have hgi : _root_.Flock.J.get (_root_.Flock.HmRow.unitsEntry c xs) "indices" "public file" =
        .ok (.arr (xs.map fun x => Lean.toJson x)) := by
      simp [_root_.Flock.J.get, hi]
    unfold _root_.Flock.J.arr
    rw [hgi]
    rfl
  unfold _root_.Flock.Registered.unitIndices
  simp only [hg, ha, bind, Except.bind]
  exact FlockSoundness.Discharge.Composed.mapM_asNat_toJson _ xs

-- `walk_step` down `HmRow.drawn`'s unfolded do-block, a goal per branch
set_option maxHeartbeats 2000000 in
/-- **A sampled sample's `units` are the population's indices at the sampled proof blocks**: `HmRow.drawn` reads the loaded
header's `units.indices` and writes, at the sample's proved units, their indices as the new header's `units`. -/
theorem drawn_units {c : _root_.Flock.Circuit} {pub pub' : _root_.Flock.Public} {d : Json}
    {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pub.header = .obj m)
    (h : _root_.Flock.HmRow.drawn c pub d = .ok pub') :
    ∃ D u a popIdx, _root_.Flock.Draw.UnitDraw.ofJson d = .ok D ∧
      _root_.Flock.J.get pub.header "units" "public file" = .ok u ∧
      _root_.Flock.J.arr u "indices" "public file" = .ok a ∧
      a.mapM (_root_.Flock.J.asNat · "a unit index") = .ok popIdx ∧ pub'.n = D.proved.size ∧
      pub'.header.getObjVal? "units" = .ok (_root_.Flock.HmRow.unitsEntry c (D.proved.map fun x => popIdx.getD x 0)) := by
  unfold _root_.Flock.HmRow.drawn at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨_, _, _, _, ‹_›, ‹_›, ‹_›, ‹_›, rfl, ?_⟩
    dsimp only
    rw [hp]
    simp only [Json.setObjVal!, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?,
      Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert, Std.ReflCmp.compare_self, ↓reduceIte]
    rfl

/-- **A loaded population the second stage's population check passes is the part's slots**: its instance count is
the part's, and its header's `units.indices` the part's slots in order. Flock loads the header normalized
(`loadPublic_raw`), equivalent to the raw header the check reads (`jsonEquiv_norm`). -/
theorem slots_of_population {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {f : ByteArray}
    {pop : _root_.Flock.Public} (hl : _root_.Flock.HmRow.loadPublic tags c f = .ok pop) {hd : Json}
    (hH : headerOf f = some hd) {q : ℕ} {p : Spot.SecondStage.Part}
    (hpo : Spot.SecondStage.populationOk q p hd = true) {u : Json} {a : Array Json} {popIdx : Array ℕ}
    (hu : _root_.Flock.J.get pop.header "units" "public file" = .ok u)
    (ha : _root_.Flock.J.arr u "indices" "public file" = .ok a)
    (hidx : a.mapM (_root_.Flock.J.asNat · "a unit index") = .ok popIdx) :
    pop.n = q * p.n ∧ ∀ i < q * p.n, popIdx.getD i 0 = p.lo + i := by
  obtain ⟨j, hj, hn, -⟩ := Depths.loadPublic_counts hl
  obtain ⟨j', hj', hph, -, -⟩ := HeaderRoots.loadPublic_raw hl
  have e : j = hd := Option.some.inj (hj.symm.trans hH)
  have e' : j' = hd := Option.some.inj (hj'.symm.trans hH)
  rw [e] at hn
  rw [e'] at hph
  have hW := HeaderRoots.wf_of_headerOf hH
  have hWn := FlockDeploy.wf_norm hd
  have hE := FlockDeploy.jsonEquiv_norm hd hW
  rw [hph] at hu
  cases hgi : Spot.getD (Spot.getD hd "units") "indices" <;>
    cases hgm : Spot.nat? (Spot.getD hd "instances") <;>
    simp only [Spot.SecondStage.populationOk, hgi, hgm, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      reduceCtorEq, Bool.false_eq_true] at hpo
  rename_i is m
  obtain ⟨⟨hm, -⟩, hlist⟩ := hpo
  obtain ⟨wi, hwi, hei, -⟩ := HeaderRoots.getObjVal_equiv hWn hW hE (FlockComplete.JsonRT.nat_ok hn)
  rw [HeaderRoots.getD_of_getObjVal hwi, FlockDeploy.jsonEquiv_num hei, nat_num] at hgm
  obtain rfl := Option.some.inj hgm
  have hu' := FlockComplete.JsonRT.get_ok hu
  obtain ⟨w₁, hw₁, he₁, hW₁⟩ := HeaderRoots.getObjVal_equiv hWn hW hE hu'
  have hWu := HeaderRoots.wf_of_getObjVal hWn hu'
  obtain ⟨w₂, hw₂, he₂, -⟩ := HeaderRoots.getObjVal_equiv hWu hW₁ he₁ (FlockComplete.JsonRT.arr_ok ha)
  rw [HeaderRoots.getD_of_getObjVal hw₁, HeaderRoots.getD_of_getObjVal hw₂] at hgi
  subst hgi
  obtain ⟨hsz, hel⟩ := (FlockDeploy.jsonEquiv_arr_iff a is).1 he₂
  obtain ⟨hsz', hall⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hidx
  have hlen : is.size = q * p.n := by simpa using congrArg List.length hlist
  refine ⟨hm, fun i hi => ?_⟩
  have hia : i < a.size := by omega
  have hip : i < popIdx.size := by omega
  have hii : i < is.size := by omega
  have h2 := hel i hia hii
  rw [FlockComplete.JsonRT.asNat_ok (hall i hia hip)] at h2
  have h4 : Spot.nat? is[i] = some (p.lo + i) := by
    have := congrArg (fun l => l[i]?) hlist
    simpa [hii, hi] using this
  rw [FlockDeploy.jsonEquiv_num h2, nat_num] at h4
  simp [hip, Option.some.inj h4]

end AddressRow

namespace CheckedRun

open AddressRow

variable {h : Held} {r : CheckedRun h.stage}

/-- **A checked inner block's session is one table, at a table count of at most one**: the statement check holds the
unit's statement to its template's scope (`stmtOkAt_unit`), the checked unit's template is one of the members'
(`templateOf_lt_of_checked`), and a statement in a template's scope sets up one table where the design evaluates
(`Verity.SampledTwo.oneTable_of_sessionOk`). -/
theorem one_table (hr : r.Ok h) : r.sts.size = 1 ∧ r.ins.read.I.count ≤ 1 := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  exact Verity.SampledTwo.oneTable_of_sessionOk h2 (templateOf_lt_of_checked h hx hu) hat.1.2 hset

/-- **The session's sampled proof blocks**: the session's statement is one table (`one_table`), set up at its sample, which parses
to the second sample's units in the part counted from its first (`proved_of_wf` at the second stage's sample check), and
the session's public file labels each sampled proof block by its slot in the part (`slots_of_population`). -/
theorem drawn_of_ok (hr : r.Ok h) :
    ∃ dj D draw, r.ins.read.draw = some dj ∧ _root_.Flock.Draw.UnitDraw.ofJson dj = .ok D ∧
      secondDraw h r.c₀ r.c₁ = some draw ∧ r.sts[0]!.pub.n = D.proved.size ∧
      D.proved = ((((Spot.Draw.proved draw).map Int.toNat).filter
        ((partOf h r).holds (sampleOf h r).size)).map (· - (partOf h r).lo)) ∧
      ∀ idx, _root_.Flock.Registered.unitIndices r.sts[0]!.pub = .ok idx → ∀ n < D.proved.size,
        idx.getD n 0 = (partOf h r).lo + D.proved.getD n 0 := by
  have hpos := sts_pos hr
  have hc1 : r.ins.read.I.count ≤ 1 := (one_table hr).2
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hsc := sessionScoped h r.x r.m₁ (unitStmt h r.m₂ r.u) r.c₀ r.c₁ _ (stmtOkAt_unit h hx hok hu)
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, -, -, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨-, -, -, hpo, -, hsamp, -, -, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hwins := ins_of_setupOf hw hset
  obtain ⟨-, hH, -, -⟩ := stmtJsonOf_fields h hS
  have hpf := (ins_reads_public hwins).2
  obtain ⟨dr, hdrm, hdr⟩ := FlockDeploy.Raw.ins_draw hwins
  obtain ⟨b, hb', hjb⟩ := stmtJsonOf_draw hS
  rw [hb', Option.mapM_some, hjb] at hdrm
  obtain ⟨_, he, rfl⟩ := Depths.someMap_ok hdrm
  cases he
  have hdj : r.ins.read.draw = some (_root_.Flock.norm S.draw) := by rw [hdr]; rfl
  obtain ⟨hc', hh⟩ := hsc r.ins spec r.sts hset
  obtain ⟨hRu, -⟩ := (_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc'
  have hpi : r.ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  obtain ⟨-, hsetup⟩ := FlockDeploy.setupOf_some hset
  have hset' := FlockDeploy.Ins.setup_read hsetup
  unfold _root_.Flock.Readable.Read.setup at hset'
  rw [hpi] at hset'
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset'
  obtain ⟨d0, hd, hdraw0⟩ := hall 0 hpos
  rw [hdraw0 hc1] at hd
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd
  rw [_root_.Flock.Readable.tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  have hspec := FlockSoundness.Discharge.Hidden.setupHidden_spec' hd
  obtain ⟨pop, hpopT, -, hsome⟩ := hspec.pub
  have hpop := (_root_.Flock.Readable.loadPublicH_tagsAt _ _ _ _ _).symm.trans hpopT
  have hl := FlockSoundness.Discharge.Hidden.loadPublic_nil hpop
  have hdrawn : _root_.Flock.HmRow.drawn { r.sts[0].c with outPorts := #[] } pop (_root_.Flock.norm S.draw) =
      .ok r.sts[0].pub := hsome _ hdj
  obtain ⟨j0, -, hph, -, m, hm⟩ := HeaderRoots.loadPublic_raw hl
  have hmw : m.WF := by
    have hWp := FlockDeploy.wf_norm j0
    rw [← hph, hm] at hWp
    cases hWp with
    | obj hmw _ => exact hmw
  obtain ⟨D, u, a, popIdx, hD, hu', ha, hidxs, hn', hunits⟩ := drawn_units hmw hm hdrawn
  have hH' : headerOf r.ins.read.I.publicFile = some S.header := by rw [hpf]; exact hH
  obtain ⟨hpn, hslot⟩ := slots_of_population (q := (sampleOf h r).size) (p := partOf h r) hl hH' hpo hu' ha hidxs
  obtain ⟨D', hD', hpopD, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdrawn
  rw [hD] at hD'
  cases hD'
  have hdl := FlockComplete.JsonRT.Ins.setup_drawLaw hsetup (by rw [hdj]; rfl)
  rw [hdj] at hdl
  simp only [_root_.Flock.ProvedScope.drawLaw, hD] at hdl
  have hlw : _root_.Flock.ProvedScope.lawWhy r.ins.read.zk r.ins.read.tags false D = none := by
    cases e : _root_.Flock.ProvedScope.lawWhy r.ins.read.zk r.ins.read.tags false D with
    | none => rfl
    | some _ => rw [e] at hdl; cases hdl
  have hWd : Claims.Json.Wf S.draw := wf_of_jsonOf hjb
  have hj : Spot.jeq (_root_.Flock.norm S.draw) (subsetJson ((sampleOf h r).size * (partOf h r).n)
      ((((Spot.Draw.proved draw).map Int.toNat).filter ((partOf h r).holds (sampleOf h r).size)).map
        (· - (partOf h r).lo))) = true := by
    unfold Spot.jeq
    rw [FlockDeploy.canon_jsonEquiv (FlockDeploy.jsonEquiv_norm _ hWd)]
    exact hsamp
  have hprov := proved_of_wf (FlockDeploy.wf_norm _) hD (FlockComplete.JsonRT.law_of_lawWhy hlw) hj
  have h0 : r.sts[0]! = r.sts[0] := getElem!_pos r.sts 0 hpos
  refine ⟨_, D, draw, hdj, hD, hdraw, by rw [h0, hn'], hprov, fun idx hidx n hn => ?_⟩
  rw [h0, unitIndices_of_units hunits] at hidx
  cases hidx
  have hnD : n < D.proved.size := hn
  have hlt : D.proved[n] < (sampleOf h r).size * (partOf h r).n := by
    rw [← hpn, ← hpopD]
    exact FlockSoundness.Discharge.Layout.proved_lt_population hD _ (Array.getElem_mem hnD)
  have e1 : (D.proved.map fun x => popIdx.getD x 0).getD n 0 = popIdx.getD D.proved[n] 0 := by simp [hnD]
  have e2 : D.proved.getD n 0 = D.proved[n] := by simp [hnD]
  rw [e1, e2, hslot _ hlt]

/-- **A checked inner block's session labels its instances by their slots** (`drawn_of_ok`). -/
theorem labelled_of_ok (hr : r.Ok h) : LabelledAt (drawSlot (partOf h r) r.ins.read.draw) r.sts := by
  obtain ⟨dj, D, -, hdj, hD, -, hn, -, hidx⟩ := drawn_of_ok hr
  intro idx hi n hlt
  rw [hn] at hlt
  rw [hidx idx hi n hlt]
  simp only [drawSlot, hdj, unitOf, hD]

end CheckedRun

end SpotFlock
