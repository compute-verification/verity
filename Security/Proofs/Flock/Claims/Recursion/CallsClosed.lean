import Proofs.Flock.Claims.Recursion.CallsR
import Proofs.Flock.Claims.Recursion.VerdictFramedJson
import Proofs.Flock.Claims.Recursion.NextRRows
import Proofs.Flock.Claims.NumbersDeploy
import Proofs.Flock.Claims.Recursion.InnerSetupCount
import Proofs.Flock.Claims.Recursion.AcceptedScope

/-!
# `c_V = 2^24` at a plain session in scope

At a decision `actR` makes, the record the verdict decodes is the reader's (`recJ`) through the substitutions, whose
streams are the reps' (`jAt_recJ_streams`): `2·n` of them at `n ≤ 64` tables (`setup_tables`), each of at most its
rep's rounds, `2200` at a session in scope (`repLens_le_in`); and `readPs` reads `2·n` proofs. The verdict runs at the
session the setup returns against `H`, which has at most `64` tables at any hash too (`setupWith_tables`). So a
decision calls SHA-512 at most `2^24` times on every run (`decisionAtS_calls_in`). With the inner setup's
`c_B = SetupCount.innerB ≤ 2^43` at any hash (`innerSetupCalls_le_acc`), its inputs (`innerSetupIns_le_acc`) and the
scope (`plainIn_of_run_acc`), all read from an accepted run, `λ₂ ≥ 100` holds at KS's deployed record with no count
premise (`lamLin2_R_ge_in`).
-/

namespace FlockDeploy

open Flock
open Lean (Json)
open FlockComplete.JsonRT (streamObj recordObj)

section Walk

open FlockSoundness.Refine FlockSoundness.Discharge.Exec

set_option maxHeartbeats 4000000 in
/-- **`setupTables_le` at any hash**: at most `Tags.MAX_TABLES` tables, and the spec names one per statement. -/
theorem setupTablesWith_le {h₀ : ByteArray → ByteArray} {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {publicInputs : Option ByteArray} {spec : SessionSpec}
    {sts : Array Stmt}
    (h : Stmt.setupTablesWith (Flock.Hashes.ofSha512 h₀ : Flock.Hashes Flock.V) tags cf pf coins count tables draw partition
      program work stratified closure publicInputs = .ok (spec, sts)) :
    sts.size ≤ Tags.MAX_TABLES ∧ spec.tables.size = sts.size := by
  unfold Stmt.setupTablesWith at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals
    first
    | (simp [Stmt.spec, Tags.MAX_TABLES]; done)
    | (obtain ⟨hsz, -⟩ := range_mapM_ok (by assumption)
       refine ⟨by omega, ?_⟩
       simp [Tags.tableNames, show ¬ count ≤ 1 by omega, hsz])

end Walk

/-- **A setup at any hash has at most 64 tables, and its spec names one per statement.** -/
theorem setupWith_tables {h₀ : ByteArray → ByteArray} {r : Readable.Read} {spec : SessionSpec} {sts : Array Stmt}
    (hr : setupWith h₀ r = .ok (spec, sts)) : sts.size ≤ 64 ∧ spec.tables.size = sts.size := by
  unfold setupWith setupM at hr
  have h := setupTablesWith_le hr
  have hmax : Tags.MAX_TABLES = 64 := rfl
  exact ⟨by omega, h.2⟩

theorem cutLens_getD_length_le {α : Type} : ∀ (ns : List ℕ) (l : List α) (i : ℕ),
    ((cutLens ns l).getD i []).length ≤ ns.getD i 0
  | [], _, _ => by simp [cutLens]
  | n :: ns, l, 0 => by simp only [cutLens, List.getD_cons_zero, List.length_take]; exact Nat.min_le_left _ _
  | n :: ns, l, i + 1 => by
    simp only [cutLens, List.getD_cons_succ]
    exact cutLens_getD_length_le ns (l.drop n) i

theorem getD_le_of_forall {l : List ℕ} {B : ℕ} (h : ∀ y ∈ l, y ≤ B) (i : ℕ) : l.getD i 0 ≤ B := by
  rw [List.getD_eq_getElem?_getD]
  cases hx : l[i]? with
  | none => simp
  | some a => exact h a (List.mem_of_getElem? hx)

private theorem forall₂_mem_r {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ b ∈ l₂, ∃ a ∈ l₁, R a b
  | _, _, .nil => fun b hb => by simp at hb
  | _, _, .cons hab h => fun b hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨_, by simp, hab⟩
    · obtain ⟨a, ha, hr⟩ := forall₂_mem_r h b hb
      exact ⟨a, by simp [ha], hr⟩

/-- **A reader's stream has at most its rep's rounds.** -/
theorem recStreams_size_le {spec : SessionSpec} {sts : Array Stmt} {ms : List Claims.Msg} {cs : List Coin} {R : ℕ}
    (hR : ∀ y ∈ repLens spec sts, y ≤ R) : ∀ a ∈ recStreams spec sts ms cs, a.size ≤ R := by
  intro a ha
  unfold recStreams at ha
  rw [List.mem_toArray] at ha
  obtain ⟨k, -, rfl⟩ := List.mem_map.1 ha
  rw [List.size_toArray, List.length_mapIdx]
  exact (cutLens_getD_length_le _ _ k).trans (getD_le_of_forall hR k)

/-- **A rep has at most `2200` rounds at a plain session in scope.** -/
theorem repLens_le_in {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (hin : PlainIn x s spec sts) :
    ∀ y ∈ repLens spec sts, y ≤ 2200 := by
  intro y hy
  unfold repLens at hy
  obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hy
  have hk' : k / 2 < sts.size := by have := List.mem_range.1 hk; omega
  rw [getElem!_pos sts (k / 2) hk']
  obtain ⟨-, -, hr⟩ := hin.tab _ (Array.getElem_mem_toList hk')
  have hr' : sts[k / 2].c.kLog ≤ 27 ∧ sts[k / 2].regions.size ≤ 1026 ∧ spec.mPts ≤ 64 := hr
  have hp : spec.points * sts[k / 2].regions.size ≤ 2 * 1026 := Nat.mul_le_mul hin.points hr'.2.1
  exact RecDeploy.repSchedAt_length_le spec _ hr'.1 (by omega)

/-- **The substitutions keep a record's streams readable.** -/
theorem arr_streams_jAt_ok {H : Claims.Msg → Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {recs₀ : Array Json} (hf : Record.arr f.j "streams" = .ok recs₀) :
    ∃ recs, Record.arr (jAt H spec sts c₁ mids f) "streams" = .ok recs := by
  unfold jAt coinSubJ fillCoins
  simp only [mapRounds, setPoints]
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link"), arr_eraseCoins, shaSubJ,
    arr_mapKey_ne _ _ (proofKey_ne_streams spec).symm, hf]
  exact ⟨_, rfl⟩

/-- **The decided record's streams at the reader's record**: at most one a rep stream, each of at most `R` rounds where
every rep has at most `R`. -/
theorem jAt_recJ_streams {H : Claims.Msg → Claims.Msg} {spec spec' : SessionSpec} {sts sts' : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {ms : List Claims.Msg} {cs : List Coin}
    {ps : Array Flock.Proof} {mids : List (Claims.Msg × Coin)} {x : Claims.Msg} {R : ℕ}
    (hR : ∀ y ∈ repLens spec sts, y ≤ R) :
    ∃ sj, Record.arr (jAt H spec' sts' c₁ mids ⟨x, recJ spec sts link c₁ ms cs ps, ps⟩) "streams" = .ok sj ∧
      sj.size ≤ spec.repStreams.size ∧ ∀ s ∈ sj, ∀ rj, Record.arr s "rounds" = .ok rj → rj.size ≤ R := by
  have h0 : Record.arr (⟨x, recJ spec sts link c₁ ms cs ps, ps⟩ : FinalP).j "streams" =
      .ok ((spec.repStreams.zip (recStreams spec sts ms cs)).map (streamObj spec _ link.1.toArray)) :=
    arr_recordObj_streams _ _ _ _ _ _
  obtain ⟨sj, hsj⟩ := arr_streams_jAt_ok (H := H) (spec := spec') (sts := sts') (c₁ := c₁) (mids := mids) h0
  obtain ⟨recs₀, h1, h2⟩ := arr_streams_jAt hsj
  have e := Except.ok.inj (h1.symm.trans h0)
  subst e
  refine ⟨sj, hsj, ?_, fun s hs rj hrj => ?_⟩
  · have hl := h2.length_eq
    simp only [Array.length_toList, Array.size_map, Array.size_zip] at hl
    omega
  · obtain ⟨t, ht, hk⟩ := forall₂_mem_r h2 s (Array.mem_toList_iff.2 hs)
    rw [Array.mem_toList_iff, Array.mem_map] at ht
    obtain ⟨q, hq, rfl⟩ := ht
    obtain ⟨rs', h3, h4⟩ := hk.rounds _ (arr_streamObj_rounds spec _ _ q)
    rw [hrj] at h3
    cases h3
    have hl := h4.length_eq
    simp only [Array.length_toList, Array.size_map, Array.size_range] at hl
    have := recStreams_size_le hR _ (Array.of_mem_zip hq).2
    omega

end FlockDeploy

namespace FlockDeploy.KS

open Flock

/-- **A decision's SHA-512 calls at a plain session in scope, on every run: at most `2^24`**, at the session the setup
returns against `H`: at most `128` files, `128` streams of at most `2200` rounds, and `64` tables of two reps. -/
theorem decisionAtS_calls_in {rd : Claims.Msg → Option FinalP} (H : Claims.Msg → Claims.Msg) {x : Claims.Msg}
    {s₀ : Ins} {spec₀ : SessionSpec} {sts₀ : Array Stmt} (hin : PlainIn x s₀ spec₀ sts₀)
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec spec' : SessionSpec} {sts sts' : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hw : ((setupM Claims.Hashes.oc s.read).run.eval H).1 = .ok (spec', sts')) :
    ((decisionAtS s spec' sts' x link c₁ mids f).run.eval H).2.length ≤ 2 ^ 24 := by
  obtain ⟨h64', htab'⟩ := setupWith_tables ((setupM_eval H s.read).1.symm.trans hw)
  obtain ⟨v, c₁', r₀, rss, fl, link', pf, hv, -, -, -, -, hread, he⟩ := decR_some (actR_decide ha)
  obtain ⟨hs, -, -, hlink, hc₁, -, hf⟩ := DecR.mk.inj he
  have hv' : v = (s₀, spec₀, sts₀) := Option.some.inj (hv.symm.trans hin.setup)
  subst hv'
  dsimp only at hs hread hf
  rw [hs, hlink, hc₁, hf]
  obtain ⟨sj, hsj, hsjs, hR⟩ := jAt_recJ_streams (H := H) (spec := spec₀) (sts := sts₀) (spec' := spec')
    (sts' := sts') (link := link') (c₁ := c₁') (ms := pf.2) (cs := cs.tail.dropLast) (ps := pf.1) (mids := mids)
    (x := x) (repLens_le_in hin)
  refine (decisionAtS_calls_le H s₀ spec' sts' x link' c₁' mids _ sj 2200 hsj hR).trans ?_
  have hps := (readPs_fits hread).1
  obtain ⟨h64, htab⟩ := setup_tables hin.setup
  have hrs := repStreams_size spec₀
  have hrs' := repStreams_size spec'
  have htb : (tabsOf sts').size = sts'.size := by simp [tabsOf]
  have e16 : (2 : ℕ) ^ 16 = 65536 := rfl
  have e24 : (2 : ℕ) ^ 24 = 16777216 := rfl
  try dsimp only
  try simp only [e16, e24]
  omega

end FlockDeploy.KS

namespace RecDeploy

open Flock
open Claims (Msg Out Prover run coinSeq)
open FlockDeploy (Coin Ins)

/-- **`λ₂ ≥ 100` at KS's deployed record, at an accepted statement**: the inner verifier's calls are at most
`2^28 + (c_B + c_V)` on every run, with `c_B = SetupCount.innerB` the inner setup's at any hash and `c_V = 2^24` a
decision's, and route 2C's `λ₂` at that count is at least `100`. Every premise is read from the accepted run `hacc`. -/
theorem lamLin2_R_ge_in (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (rd : Msg → Option FlockDeploy.FinalP)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (ps : List (Msg × Out)) :
    (∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤
        2 ^ 28 + (SetupCount.innerB + 2 ^ 24)) ∧
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record
        (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x (2 ^ 28 + (SetupCount.innerB + 2 ^ 24)) (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L
        (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).vstar x ps)).length ≤
        2 ^ 55 := by
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_run_acc hacc
  have hB : ∀ v, FlockDeploy.setupOf false (xIOf x) = some v →
      FlockDeploy.setupCallsOf (Claims.hOf H) v.1.read ≤ SetupCount.innerB := fun v hv => by
    obtain ⟨s', spec', sts', hset, hle⟩ := innerSetupCalls_le_acc hacc
    rw [hset] at hv
    cases hv
    exact hle _
  have hc : 2 ^ 28 + (SetupCount.innerB + 2 ^ 24) ≤ 2 ^ 55 :=
    le_trans (Nat.add_le_add_left (Nat.add_le_add_right SetupCount.innerB_le (2 ^ 24)) (2 ^ 28)) (by norm_num)
  exact lamLin2_R_ge_acc' rel b L rd colsE wE join rowIn final w decI H hacc (innerSetupIns_le_acc hacc) ps
    SetupCount.innerB (2 ^ 24) hB
    (fun _ _ _ _ _ _ _ _ _ _ _ ha hw => FlockDeploy.KS.decisionAtS_calls_in H hin ha hw) hc

end RecDeploy
