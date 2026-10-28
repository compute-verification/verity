import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.KS.PlainDrawJRun

/-!
# Two accepted last steps of the deployed reader: what they share, and the verdict at a parsed value

The columns and calls halves of `RecDeploy.RLastBinds` compare two decisions of `nextR rd` on the same rounds and
coins that differ only in the last message. Here what both halves read:

* `nextR_shared`: the two decisions share the setup, the link, the first coin and the round messages. `decR` reads
  the last message only for what no frame reads (`readRep_last`, `readPs_last`), so the round messages agree.
* `decideR_runs`, `verdictR_within`: an accepted decision runs the verdict at `jAt sha512` and the read proofs' files
  within its own log, so the verdict accepts with SHA-512 answered only within that log (`decisionE_runs`,
  `verdictE_within` at `nextR`). `checksAtJ_leafL`: the checks pin every table's Merkle scheme (`checks_leaf`).
* The plain verdict at a parsed value, at any `keep` (`verifyJWithMK`): its tables and reps
  (`verifyJWithMK_tables`, `verifyJWithMK_rep`, `verifyJWithMK_proofs_size`), its round counts
  (`rep_plainJK`, `streams_ns_plainJK`, `rounds_ns_plainJK`, `streams_sizes_plainJK`), and two records under one
  custody (`live_keys_eqJK`, `sess_rounds_eqJK`, `streams_coins_eqJK`): the text forms' proofs less the parse.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine FlockSoundness.Discharge.Exec
open Lean (Json)

/-! ## What two accepted last steps share -/

/-- **The round messages `readPs` reads do not depend on the last message.** -/
theorem readPs_last {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl fl' : FinalP} {pf pf' : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) (h' : readPs spec sts c₁ r₀ rss cs fl' = some pf') :
    pf'.2 = pf.2 := by
  unfold readPs at h h'
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · by_cases hc' : fl'.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at h
      rw [ite_eq_left_of_eq_true _ _ (eq_true hc')] at h'
      obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨rs', hrs', rfl⟩ := Option.map_eq_some_iff.1 h'
      obtain ⟨hlen, hget⟩ := mapM_some_get hrs
      obtain ⟨hlen', hget'⟩ := mapM_some_get hrs'
      show (rs'.map Prod.snd).flatten = (rs.map Prod.snd).flatten
      have hl₁ : rs'.length = rs.length := hlen'.trans hlen.symm
      congr 1
      refine List.ext_getElem (by simp [hl₁]) fun i h₁ h₂ => ?_
      rw [List.length_map] at h₁ h₂
      simp only [List.getElem_map]
      have hi : i < (List.range (2 * sts.size)).length := hlen ▸ h₂
      have hr := hget i hi h₂
      have hr' := hget' i hi h₁
      beta_reduce at hr hr'
      rw [show rs[i] = (rs[i].1, rs[i].2) from rfl] at hr
      rw [show rs'[i] = (rs'[i].1, rs'[i].2) from rfl] at hr'
      exact (readRep_last hr hr').2
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hc')] at h'
      cases h'
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hc)] at h
    cases h

/-- **Two decisions of `nextR` on the same rounds and coins, differing only in the last message, share the setup, the
link, the first coin and the round messages.** -/
theorem nextR_shared {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {l l' : Claims.Msg}
    {cs : List Coin} {s s' : Ins} {spec spec' : SessionSpec} {sts sts' : Array Stmt}
    {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin} {mids mids' : List (Claims.Msg × Coin)}
    {f f' : FinalP} (hms : ms ≠ [])
    (ha : actR rd x (ms ++ [l]) cs = .decide s spec sts link c₁ mids f)
    (ha' : actR rd x (ms ++ [l']) cs = .decide s' spec' sts' link' c₁' mids' f') :
    s' = s ∧ spec' = spec ∧ sts' = sts ∧ link' = link ∧ c₁' = c₁ ∧ mids' = mids := by
  obtain ⟨v, c₀, r₀, rss, fl, lk, pf, hv, hc, hr, hrs, hlk, hpf, hd⟩ := decR_some (actR_decide ha)
  obtain ⟨v', c₀', r₀', rss', fl', lk', pf', hv', hc', hr', hrs', hlk', hpf', hd'⟩ := decR_some (actR_decide ha')
  rw [hv] at hv'
  cases hv'
  rw [hc] at hc'
  cases hc'
  have hh : (ms ++ [l']).headD [] = (ms ++ [l]).headD [] := by
    cases ms with
    | nil => exact absurd rfl hms
    | cons b t => rfl
  have ht : (ms ++ [l']).tail.dropLast = (ms ++ [l]).tail.dropLast := by
    cases ms with
    | nil => exact absurd rfl hms
    | cons b t => simp
  rw [hh, hr] at hr'
  cases hr'
  rw [ht, hrs] at hrs'
  cases hrs'
  rw [hlk] at hlk'
  cases hlk'
  have hp := readPs_last hpf hpf'
  simp only [DecR.mk.injEq] at hd hd'
  obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := hd
  obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := hd'
  exact ⟨rfl, rfl, rfl, rfl, rfl, by rw [hp]⟩

/-- **A last step of `nextR` that decides follows at least one earlier message**: with a single message the statement
sets up no session (`V false`'s round count is `1` only there). -/
theorem ms_ne_nil_of_actR {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {l : Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP} (hk : ms.length + 1 = (FlockDeploy.V false).k x)
    (ha : actR rd x (ms ++ [l]) cs = .decide s spec sts link c₁ mids f) : ms ≠ [] := by
  intro hms
  subst hms
  have h := actR_setupOf ha
  simp only [List.length_nil, Nat.zero_add] at hk
  simp only [FlockDeploy.V, h] at hk
  omega

/-! ## An accepted decision, within its own log -/

/-- **An accepted decision of `nextR` runs the verdict within its own log**, at SHA-512: the verdict at `jAt sha512`
and the read proofs' files (`decisionE_runs` at `nextR`). -/
theorem decideR_runs {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) :
    Runs sha512 (verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (jAt sha512 spec sts c₁ mids f)
      (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)) () (((nextR rd x as cs).eval sha512).2.map Prod.fst) := by
  have hsu := (Claims.Bridge.PlainDraw.decideR_sha512 ha hacc).2.2.1
  have hrun : Runs sha512 (decisionSF s x link c₁ mids f) ()
      (((nextR rd x as cs).eval sha512).2.map Prod.fst) := by
    have ha' : ((outOf (decisionSF s x link c₁ mids f)).eval sha512).1 = .acc := by
      rw [← nextR_decide ha]
      exact hacc
    have hl : ((nextR rd x as cs).eval sha512).2 = ((decisionSF s x link c₁ mids f).run.eval sha512).2 := by
      rw [nextR_decide ha, outOf_log]
    rw [hl]
    exact Runs.self sha512 (outOf_acc sha512 _ ha')
  unfold decisionSF at hrun
  obtain ⟨⟨spec', sts'⟩, h0, h⟩ := Runs.bind hrun
  have e : setupWith (Claims.hOf sha512) s.read = .ok (spec', sts') := by
    rw [← (setupM_eval sha512 s.read).1]
    exact h0.1
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hsu] at e
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj e)
  unfold decisionAtS at h
  obtain ⟨ds, h1, h2⟩ := Runs.bind h
  have hds : ds = (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).map
      (FlockComplete.dgOf spec (Claims.hOf sha512)) := by
    have e₁ : ((((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).mapM
        (spec.dgWith Claims.Hashes.oc)).run).eval sha512).1 =
        .ok ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).map (FlockComplete.dgOf spec (Claims.hOf sha512))) :=
      evOk_mapM_array (evOk_dg sha512 spec) _
    exact Except.ok.inj (h1.1.symm.trans e₁)
  subst hds
  unfold decisionAtP decisionAtJ at h2
  obtain ⟨⟨⟩, h3, -⟩ := Runs.bind h2
  exact h3

/-- **The verdict at a parsed value within `L` is the verdict with SHA-512 answered only within `L`.** -/
theorem verdictR_within {spec : SessionSpec} {tabs : Array Setup} {j : Json} {proofs : Array ByteArray}
    {L : List Claims.Msg} (h : Runs sha512 (verifyJWithM Claims.Hashes.oc spec tabs j proofs) () L) :
    verifyJWithMK id (hashesIn L) spec tabs j proofs = .ok () :=
  (With.verifyJWithM_sim (within L) (within_hashes L) spec tabs j proofs) () h

/-- **`checksAtJ` pins the tables' Merkle scheme** (`checks_leaf` at a parsed value): every table the statement sets up
commits under `hm96Sha512`, or, for a statement read without `--zk`, under plain SHA-512 leaves. -/
theorem checksAtJ_leafL {s : Ins} {spec : SessionSpec} {x : Claims.Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {v : Json} {sts : Array Stmt}
    (hc : checksAtJ s spec x link c₁ mids f v = .ok ()) (hs : s.setup = .ok (spec, sts)) :
    ∀ (j : ℕ) (hj : j < sts.size), sts[j].merkle = MerkleScheme.hm96Sha512 ∨
      (s.read.zk = false ∧ sts[j].merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") := by
  have hck' : ProvedScope.check s.read.tags s.read.zk s.read.I.circuitFile s.read.I.publicFile s.read.I.tables
      s.read.I.count s.read.publicInputs.isSome s.read.registered = .ok () :=
    Claims.Bridge.Custody.checksAtJ_scope hc
  obtain ⟨⟨-, -, -, hl, -, hpi⟩, -⟩ := (Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck'
  have hpi' : s.read.publicInputs = none := by
    cases hp : s.read.publicInputs with
    | none => rfl
    | some _ => rw [hp] at hpi; cases hpi
  have hset := Ins.setup_read hs
  unfold Readable.Read.setup at hset
  rw [hpi'] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  intro j hj
  obtain ⟨d, hd, -⟩ := hall j hj
  rcases hl with hl | ⟨hz, hh, hl⟩
  · exact Or.inl (FlockSoundness.Discharge.HiddenExec.setupOne_tagsAt_hm96 hl hd)
  · rcases FlockSoundness.Discharge.HiddenExec.setupOne_tagsAt_leaf (Or.inr ⟨hh, hl⟩) hd with ⟨-, hm⟩ | ⟨-, hm⟩
    · exact Or.inl hm
    · exact Or.inr ⟨hz, hm⟩

/-! ## The plain verdict at a parsed value: tables, reps and round counts -/

/-- `verifyWithMK_tables` at a parsed value. -/
theorem verifyJWithMK_tables (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {j : Json} {proofs : Array ByteArray} (h : verifyJWithMK keep hs spec tabs j proofs = .ok ()) :
    ∃ (sess : RunRecord) (schs : Array Schedule), Record.decodeJWithK keep hs spec j = .ok sess ∧
      tabs.size = spec.tables.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs' : i < schs.size) (c₀ c₁ : ByteArray),
        fast100 tabs[i].m = some schs[i] ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c₀ ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c₁ := by
  obtain ⟨sess, schs, hd, hsz, -, hss, hall⟩ := verifyJWithMK_tables_cap keep hs h
  refine ⟨sess, schs, hd, hsz, hss, fun i hi => ?_⟩
  obtain ⟨hi', c, hf, h₀, h₁⟩ := hall i hi
  exact ⟨hi', c, c, hf, h₀, h₁⟩

theorem forall₂_swap {α β : Type} {R : α → β → Prop} {S : β → α → Prop} (hRS : ∀ a b, R a b → S b a) :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → List.Forall₂ S l' l
  | _, _, .nil => .nil
  | _, _, .cons h hl => .cons (hRS _ _ h) (forall₂_swap hRS hl)

/-- `liveOf_decodeWithK` at a parsed value. -/
theorem liveOfJ_decodeJWithK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {lv : Live} {sess : RunRecord} (hl : FlockDeploy.liveOfJ spec j = .ok lv)
    (hd : Record.decodeJWithK keep hs spec j = .ok sess) :
    lv.points = sess.points ∧ sess.points.size = spec.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ (k : ℕ) (hk : k < spec.repStreams.size) (hk' : k < lv.streams.size), ∃ stream : StreamRec,
        sess.streams.find? (fun x => x.name == spec.repStreams[k].2.2) = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList := by
  obtain ⟨-, hpts, hsz, hstr⟩ := decodeJWithK_live keep hs hd hl
  refine ⟨hpts.symm, (decodeJWithK_points keep hs hd).1, hsz, fun k hk hk' => ?_⟩
  obtain ⟨y, hy, hfa⟩ := hstr k hk
  rw [getElem!_pos lv.streams k hk'] at hfa
  exact ⟨y, hy, forall₂_swap (fun _ _ h => ⟨h.1.symm, h.2.1.symm⟩) hfa⟩

/-- `rep_plainK` at a parsed value. -/
theorem rep_plainJK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {j : Json} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv) :
    lv.streams.size = 2 * sts.size ∧ ∀ (k : ℕ) (hk : k < lv.streams.size),
      ∃ (sess : RunRecord) (sch : Schedule) (stream : StreamRec) (c : ByteArray),
        sess.points = lv.points ∧ sess.points.size = spec.points ∧ fast100 sts[k / 2]!.m = some sch ∧
        verifyRepWith hs spec (Setup.ofCircuit sts[k / 2]!) sch sess (k / 2) (k % 2) proofs[k]! = .ok c ∧
        sess.streams.find? (fun x => x.name == s!"{spec.tables[k / 2]!}/rep{k % 2}") = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList := by
  obtain ⟨sess, schs, hdec, hsz, -, hall⟩ := verifyJWithMK_tables keep hs hv
  obtain ⟨hpts, hpn, hlsz, hstr⟩ := liveOfJ_decodeJWithK keep hs hl hdec
  have hn : sts.size = spec.tables.size := by simpa [tabsOf] using hsz
  have hk₂ : ∀ k, k < lv.streams.size → k < spec.repStreams.size := fun k hk => hlsz ▸ hk
  rw [repStreams_size, ← hn] at hlsz
  refine ⟨hlsz, fun k hk => ?_⟩
  obtain ⟨stream, hfind, hfa⟩ := hstr k (hk₂ k hk) hk
  have hi : k / 2 < (tabsOf sts).size := by simp only [tabsOf, Array.size_map]; omega
  have hi' : k / 2 < sts.size := by omega
  obtain ⟨_, c₀, c₁, hf, h₀, h₁⟩ := hall (k / 2) hi
  have htab : (tabsOf sts)[k / 2] = Setup.ofCircuit sts[k / 2]! := by
    rw [getElem!_pos sts (k / 2) hi']; exact Array.getElem_map ..
  rw [htab] at h₀ h₁ hf
  simp only [repStreams_getElem spec (hk₂ k hk)] at hfind
  refine ⟨sess, schs[k / 2], stream, ?_⟩
  rcases Nat.mod_two_eq_zero_or_one k with h | h
  · rw [show 2 * (k / 2) = k by omega] at h₀
    rw [h] at hfind ⊢
    exact ⟨c₀, hpts.symm, hpn, hf, h₀, hfind, hfa⟩
  · rw [show 2 * (k / 2) + 1 = k by omega] at h₁
    rw [h] at hfind ⊢
    exact ⟨c₁, hpts.symm, hpn, hf, h₁, hfind, hfa⟩

/-- `streams_ns_plainK` at a parsed value. -/
theorem streams_ns_plainJK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {j : Json} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].toList.map Round.n = plainRepNs spec sts[k / 2]! := by
  obtain ⟨hsz, hall⟩ := rep_plainJK keep hs hv hl
  refine ⟨hsz, fun k hk => ?_⟩
  obtain ⟨sess, sch, stream, c, -, hpn, hf, hrep, hfind, hfa⟩ := hall k hk
  obtain ⟨stream', extras, hfind', hex, hns⟩ := verifyRepWith_readsN hs rfl hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := extraClaims_size hex
  have hsch : Claims.Flock.plainSched sts[k / 2]!.m = sch := by
    simp only [Claims.Flock.plainSched, hf, Option.getD_some]
  rw [map_n_of_forall₂ hfa, hns, hx, hpn, plainRepNs, hsch, Nat.mul_comm]
  rfl

/-- `rounds_ns_plainK` at a parsed value. -/
theorem rounds_ns_plainJK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {j : Json} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv) :
    (lv.streams.toList.flatMap (·.toList)).map Round.n = plainNs spec sts := by
  obtain ⟨hsz, hall⟩ := streams_ns_plainJK keep hs hv hl
  have e : lv.streams.toList = (List.range lv.streams.size).map fun k => lv.streams[k]! := by
    refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
    simp only [Array.length_toList] at h₁
    simp [getElem!_pos lv.streams i h₁]
  rw [List.map_flatMap, e, List.flatMap_map, plainNs, ← hsz, List.flatMap_def, List.flatMap_def]
  refine congrArg List.flatten (List.map_congr_left fun k hk => ?_)
  have hk' : k < lv.streams.size := List.mem_range.1 hk
  rw [getElem!_pos lv.streams k hk']
  exact hall k hk'

/-- `streams_sizes_plainK` at a parsed value. -/
theorem streams_sizes_plainJK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {j : Json} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv) :
    lv.streams.toList.map (fun s => s.size) =
      (List.range (2 * sts.size)).map fun k => (plainRepNs spec sts[k / 2]!).length := by
  obtain ⟨hsz, hall⟩ := streams_ns_plainJK keep hs hv hl
  refine List.ext_getElem (by simp [hsz]) fun i h₁ h₂ => ?_
  have hi : i < lv.streams.size := by simpa using h₁
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  rw [← hall i hi]
  simp only [List.length_map, Array.length_toList]

/-! ## Two values under one custody -/

/-- `live_keys_eqK` at parsed values. -/
theorem live_keys_eqJK (keep keep' : Bool → Bool) (hs hs' : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {j j' : Json} {proofs proofs' : Array ByteArray} {lv lv' : Live}
    {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin} {mids : List (Claims.Msg × Coin)}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv)
    (hv' : verifyJWithMK keep' hs' spec (tabsOf sts) j' proofs' = .ok ())
    (hl' : FlockDeploy.liveOfJ spec j' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ()) :
    lv.streams.toList.map (fun s => s.toList.map roundKey) =
      lv'.streams.toList.map (fun s => s.toList.map roundKey) := by
  refine streams_eq_of_flat roundKey
    ((streams_sizes_plainJK keep hs hv hl).trans (streams_sizes_plainJK keep' hs' hv' hl').symm) ?_
  rw [keys_of_forall₂ (Claims.Bridge.Custody.custody_ok hc).2.2.2,
    keys_of_forall₂ (Claims.Bridge.Custody.custody_ok hc').2.2.2, rounds_ns_plainJK keep hs hv hl,
    rounds_ns_plainJK keep' hs' hv' hl']

/-- `sess_rounds_eqK` at parsed values: two values the plain verdict accepts at any `keep`s, under custody of the
same link and round messages, decode to sessions with the same rounds on every rep stream as replay reads them, all
retained. -/
theorem sess_rounds_eqJK (keep keep' : Bool → Bool) (hs hs' : Hashes Flock.V) {spec : SessionSpec}
    {sts : Array Stmt} {j j' : Json} {proofs proofs' : Array ByteArray} {lv lv' : Live}
    {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin} {mids : List (Claims.Msg × Coin)}
    {sess sess' : RunRecord}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv)
    (hv' : verifyJWithMK keep' hs' spec (tabsOf sts) j' proofs' = .ok ())
    (hl' : FlockDeploy.liveOfJ spec j' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ())
    (hd : Record.decodeJWithK keep hs spec j = .ok sess)
    (hd' : Record.decodeJWithK keep' hs' spec j' = .ok sess') {q : ℕ} (hq : q < spec.repStreams.size) :
    ∃ y y' : StreamRec, sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y ∧
      sess'.streams.find? (·.name == spec.repStreams[q].2.2) = some y' ∧
      y.rounds.map Round.erase = y'.rounds.map Round.erase ∧ AllRetained y.rounds := by
  obtain ⟨-, -, -, hst⟩ := decodeJWithK_live keep hs hd hl
  obtain ⟨-, -, -, hst'⟩ := decodeJWithK_live keep' hs' hd' hl'
  obtain ⟨y, hy, hfa⟩ := hst q hq
  obtain ⟨y', hy', hfa'⟩ := hst' q hq
  have hk := keys_at (live_keys_eqJK keep keep' hs hs' hv hl hv' hl' hc hc') q
  have hr := live_retained hc q
  have e : y.rounds.toList.map Round.erase = y'.rounds.toList.map Round.erase := rounds_erase_of_keys hfa hfa' hk hr
  refine ⟨y, y', hy, hy', Array.ext' (by simpa only [Array.toList_map] using e), fun j hj => ?_⟩
  rw [getElem!_pos y.rounds j hj]
  exact retained_ofK hfa hr _ (Array.mem_toList_iff.2 (Array.getElem_mem hj))

/-- `streams_coins_eqK` at parsed values. -/
theorem streams_coins_eqJK (keep keep' : Bool → Bool) (hs hs' : Hashes Flock.V) {spec : SessionSpec}
    {sts : Array Stmt} {j j' : Json} {proofs proofs' : Array ByteArray} {lv lv' : Live}
    {link link' : List ByteArray × List ByteArray} {c₁ c₁' : Coin} {mids : List (Claims.Msg × Coin)}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : FlockDeploy.liveOfJ spec j = .ok lv)
    (hv' : verifyJWithMK keep' hs' spec (tabsOf sts) j' proofs' = .ok ())
    (hl' : FlockDeploy.liveOfJ spec j' = .ok lv')
    (hc : custody spec lv link c₁ mids = .ok ()) (hc' : custody spec lv' link' c₁' mids = .ok ()) :
    lv.streams.toList.map (fun s => s.toList.map Round.coins) =
      lv'.streams.toList.map (fun s => s.toList.map Round.coins) := by
  refine coins_streams_eq
    ((streams_sizes_plainJK keep hs hv hl).trans (streams_sizes_plainJK keep' hs' hv' hl').symm) ?_
  rw [coins_of_forall₂ (Claims.Bridge.Custody.custody_ok hc).2.2.2 (fun _ _ h => h.2),
    coins_of_forall₂ (Claims.Bridge.Custody.custody_ok hc').2.2.2 (fun _ _ h => h.2),
    rounds_ns_plainJK keep hs hv hl, rounds_ns_plainJK keep' hs' hv' hl']

/-! ## The verdict's reps at a parsed value -/

/-- `verifyWithMK_proofs_size` at a parsed value. -/
theorem verifyJWithMK_proofs_size (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec}
    {tabs : Array Setup} {j : Json} {proofs : Array ByteArray}
    (h : verifyJWithMK keep hs spec tabs j proofs = .ok ()) : proofs.size = spec.repStreams.size := by
  unfold verifyJWithMK at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rename_i hne
  simpa [bne_iff_ne] using hne

/-- `verifyWithMK_rep` at a parsed value: table `k / 2`'s rep `k % 2` passes on proof file `k`, on the session the
decoder reads. -/
theorem verifyJWithMK_rep (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {j : Json} {proofs : Array ByteArray} (h : verifyJWithMK keep hs spec tabs j proofs = .ok ()) :
    ∃ (sess : RunRecord) (schs : Array Schedule), Record.decodeJWithK keep hs spec j = .ok sess ∧
      tabs.size = spec.tables.size ∧
      ∀ (k : ℕ) (hk : k / 2 < tabs.size), fast100 tabs[k / 2].m = some schs[k / 2]! ∧
        ∃ c, verifyRepWith hs spec tabs[k / 2] schs[k / 2]! sess (k / 2) (k % 2) proofs[k]! = .ok c := by
  obtain ⟨sess, schs, hdec, hsz, hss, hall⟩ := verifyJWithMK_tables keep hs h
  refine ⟨sess, schs, hdec, hsz, fun k hk => ?_⟩
  obtain ⟨hk', c₀, c₁, hf, h₀, h₁⟩ := hall (k / 2) hk
  rw [getElem!_pos schs (k / 2) hk']
  refine ⟨hf, ?_⟩
  rcases Nat.mod_two_eq_zero_or_one k with e | e
  · have e' : proofs[k]! = proofs[2 * (k / 2)]! := by rw [show 2 * (k / 2) = k by omega]
    rw [e, e']
    exact ⟨c₀, h₀⟩
  · have e' : proofs[k]! = proofs[2 * (k / 2) + 1]! := by rw [show 2 * (k / 2) + 1 = k by omega]
    rw [e, e']
    exact ⟨c₁, h₁⟩

end FlockDeploy.KS
