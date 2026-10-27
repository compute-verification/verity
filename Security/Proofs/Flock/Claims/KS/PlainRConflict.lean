import Proofs.Flock.Claims.KS.PlainR
import Proofs.Flock.Claims.Recursion.ConflictTiedE

/-!
# Conflicts tied to collisions at the deployed reader `VR rd`

`ConflictTied` at the plain record at `VR rd` (`plainR_conflictTied`), at SHA-512, by `plainE_conflictTied`'s proof
at `nextR`'s decision, which runs the verdict at the record's parsed value (`verifyJWithM`) on the read proofs' files:

* `VR rd` answers each round with its coin's bytes (`answersCoin_VR`), so an accepted run's answers show its coins
  but for the last (`run_outCoin_VR`), and the reader on the run's view is `colsR` at its messages and coins
  (`colsRV_run`).
* An accepted run decides within its own verifier log (`accR_decide`): the verdict at the substituted value accepts
  with SHA-512 answered only within that log (`decisionSF_runs`, `verdictJ_within`). Its query round places the
  reader's columns among level 0's Merkle openings (`plainQueryRound_verifyJWithMK`), the decoded session's `root_B`
  is `liveOfJ`'s (`decodeJWithK_live`), and custody ties it to the link `linkR` reads off the run's first message, so
  each opening climbs to that link's `root_B` (`plainR_opening`).
* Two accepted runs at one tape read one first message, so one link: different columns at one position collide in
  their joined logs (`plainR_conflictTied`), which discharges `plainR_KSFlockLin`'s first premise
  (`plainR_KSFlockLin_of_table`).
-/

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)
open FlockSoundness.Refine (ebind_ok)
open FlockSoundness.Discharge.Exec (verifyJWithMK_tables_cap)
open Claims.Bridge.Custody (checksAtJ_ok checksAtJ_scope)
open Claims.Bridge.PlainDraw (decideR_sha512)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The answers of a run of `VR rd` -/

/-- A coin `nextR` answers is the plain verifier's. -/
theorem act_of_actR_coin {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {c : Coin} (h : actR rd x as cs = .coin c) : act false x as cs = .coin c := by
  unfold actR at h
  split at h
  · rename_i c' hc'
    cases h
    exact hc'
  · split at h <;> cases h

/-- **`VR rd` answers each round with its coin's bytes** (`answersCoin_VK`'s proof, at `nextR rd`). -/
theorem answersCoin_VR (rd : Claims.Msg → Option FinalP) (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) :
    Claims.Bridge.AnswersCoin H (VR rd) x Coin.bytes := by
  intro as cs c m h
  change ((nextR rd x as (cs ++ [c])).eval H).1 = .msg m at h
  cases ha : actR rd x as (cs ++ [c]) with
  | rej => revert h; simp [nextR, ha, Claims.OC.eval]
  | coin c' =>
    rw [nextR_coin rd H x as (cs ++ [c]) c' (act_of_actR_coin ha)] at h
    have hc : c = c' := by simpa using Claims.Bridge.act_coin (act_of_actR_coin ha)
    rw [← Claims.Out.msg.inj h, hc]
  | decide s spec sts link c₁ mids f =>
    rw [nextR_decide ha] at h
    exact absurd h (Claims.Bridge.outOf_ne_msg _ _ _)

/-- **An accepted run's answers show its coins but the last**, wherever the verifier answers each round with its
coin's bytes (`stepWith_dep`'s induction): the last answer is the decision, which shows `default`. -/
theorem stepWith_outCoin {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (V' : Claims.Verifier Coin)
    (x : Claims.Msg) (ω : Ω) (c : ℕ → Coin) (hV : Claims.Bridge.AnswersCoin H V' x Coin.bytes) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      t.msgs.map Prod.fst = as → (t.msgs.map Prod.snd).map RecDeploy.outCoin = (List.range as.length).map c →
      (Claims.stepWith H P x ω (fun as => V'.next x as ((List.range as.length).map c)) n hist as t).out = .acc →
      ((Claims.stepWith H P x ω (fun as => V'.next x as ((List.range as.length).map c)) n hist as t).msgs.map
          Prod.snd).map RecDeploy.outCoin =
        ((List.range (Claims.stepWith H P x ω (fun as => V'.next x as ((List.range as.length).map c)) n hist as
          t).msgs.length).map c).dropLast ++ [default] := by
  intro n
  induction n with
  | zero =>
    intro hist as t _ _ h
    simp [Claims.stepWith] at h
  | succ n ih =>
    intro hist as t hm hc
    have hl' : (as ++ [((P.next x hist ω).eval H).1]).length = as.length + 1 := by
      rw [List.length_append, List.length_singleton]
    have hlt : t.msgs.length = as.length := by rw [← List.length_map (f := Prod.fst), hm]
    simp only [Claims.stepWith]
    cases hv : ((V'.next x (as ++ [((P.next x hist ω).eval H).1])
        ((List.range (as ++ [((P.next x hist ω).eval H).1]).length).map c)).eval H).1 with
    | msg m =>
      intro h
      have hm' : m = (c as.length).bytes := by
        rw [hl', List.range_succ, List.map_append] at hv
        exact hV _ _ _ _ hv
      refine ih _ _ _ ?_ ?_ h
      · rw [List.map_append, hm]
        rfl
      · rw [List.map_append, List.map_append, hc, hl', List.range_succ, List.map_append, hm']
        simp [RecDeploy.outCoin, RecDeploy.coinOf_bytes]
    | acc =>
      intro _
      dsimp only
      rw [List.map_append, List.map_append, hc, List.length_append, List.length_singleton, hlt, List.range_succ,
        List.map_append]
      simp only [List.map_cons, List.map_nil, List.dropLast_concat]
      all_goals rfl
    | rej =>
      intro h
      simp at h

/-- **An accepted run of `VR rd` shows its coins but the last** (`stepWith_outCoin`, `answersCoin_VR`). -/
theorem run_outCoin_VR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (r : Fin ((VR rd).k x) → Coin) (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) :
    ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.snd).map RecDeploy.outCoin =
      ((List.range (Claims.run sha512 P (VR rd) x ω r).msgs.length).map (Claims.coinSeq r)).dropLast ++ [default] :=
  stepWith_outCoin sha512 P (VR rd) x ω (Claims.coinSeq r) (answersCoin_VR rd sha512 x) _ [] [] _ rfl rfl h

/-- **The reader on an accepted run's view is `colsR` at its messages and coins** (`colsRV_eq`): the run's answers
show its coins but the last (`run_outCoin_VR`), and it decides at its last round (`accLast_VR`). -/
theorem colsRV_run {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) :
    colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen =
      colsR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) := by
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  refine colsRV_eq rfl ?_ ?_ hacc
  · rw [List.length_map]
    exact run_outCoin_VR rd P x ω r h
  · have hk := accLast_VR rd x _ _ (by simp) hacc
    rw [List.length_map] at hk
    simp only [List.length_map, List.length_range]
    exact hk

/-! ## An accepted run of `VR rd`, within its own log -/

/-- **An accepted run of `VR rd` decides, within its own verifier log** (`accK_decide`'s proof, at `nextR rd`). -/
theorem accR_decide {rd : Claims.Msg → Option FinalP} {H : Claims.Msg → Claims.Msg} {Ω : Type}
    {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((VR rd).k x) → Coin}
    (h : (Claims.run H P (VR rd) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      actR rd x ((Claims.run H P (VR rd) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run H P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      Runs H (decisionSF s x link c₁ mids f) () ((Claims.run H P (VR rd) x ω r).vLog.map Prod.fst) := by
  have ha := FlockDeploy.KS.run_acc_next H P (VR rd) x ω r h
  have hl := run_acc_log H P (VR rd) x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha hl
  replace ha : ((nextR rd x ((Claims.run H P (VR rd) x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  replace hl : ∀ p ∈ ((nextR rd x ((Claims.run H P (VR rd) x ω r).msgs.map Prod.fst) cs).eval H).2,
      p ∈ (Claims.run H P (VR rd) x ω r).vLog := hl
  cases hact : actR rd x ((Claims.run H P (VR rd) x ω r).msgs.map Prod.fst) cs with
  | rej => revert ha; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert ha; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_, fun p hp => List.mem_map_of_mem (hl p ?_)⟩
    · rw [nextR_decide hact] at ha
      exact outOf_acc H _ ha
    · rw [nextR_decide hact, outOf_log]
      exact hp

/-- **A `nextR` decision within `L`**, at SHA-512 on a statement that sets up at `(spec, sts)`: the verdict at the
substituted value `jAt sha512` and the read proofs' files, within `L` (`decisionK_runs`'s proof, through the files'
digests, which the oracle answers as `dgOf` at SHA-512). -/
theorem decisionSF_runs {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP}
    {L : List Claims.Msg} (hs : s.setup = .ok (spec, sts)) (h : Runs sha512 (decisionSF s x link c₁ mids f) () L) :
    Runs sha512 (verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (jAt sha512 spec sts c₁ mids f)
      (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)) () L := by
  unfold decisionSF at h
  obtain ⟨⟨spec', sts'⟩, h0, h⟩ := Runs.bind h
  have e : setupWith (Claims.hOf sha512) s.read = .ok (spec', sts') := by
    rw [← (setupM_eval sha512 s.read).1]
    exact h0.1
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at e
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj e)
  unfold decisionAtS at h
  obtain ⟨ds, hds, h⟩ := Runs.bind h
  have hd : ds = (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).map
      (FlockComplete.dgOf spec (Claims.hOf sha512)) :=
    Except.ok.inj (hds.1.symm.trans (evOk_mapM_array (evOk_dg sha512 spec) _))
  subst hd
  unfold decisionAtP decisionAtJ at h
  obtain ⟨⟨⟩, h1, -⟩ := Runs.bind h
  exact h1

/-- **The verdict at a parsed value within `L` is that verdict with SHA-512 answered only within `L`.** -/
theorem verdictJ_within {spec : SessionSpec} {tabs : Array Setup} {j : Json} {proofs : Array ByteArray}
    {L : List Claims.Msg} (h : Runs sha512 (verifyJWithM Claims.Hashes.oc spec tabs j proofs) () L) :
    verifyJWithMK id (hashesIn L) spec tabs j proofs = .ok () :=
  (With.verifyJWithM_sim (within L) (within_hashes L) spec tabs j proofs) () h

/-- **`checksAtJ` pins the tables' Merkle scheme** (`checks_leaf` at a parsed value): every table the statement sets
up commits under `hm96Sha512`, or, plain, under unsalted SHA-512 leaves. -/
theorem checksAtJ_leaf {s : Ins} {spec : SessionSpec} {x : Claims.Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {v : Json} {sts : Array Flock.Stmt}
    (hc : checksAtJ s spec x link c₁ mids f v = .ok ()) (hs : s.setup = .ok (spec, sts)) :
    ∀ (j : ℕ) (hj : j < sts.size), sts[j].merkle = MerkleScheme.hm96Sha512 ∨
      (s.read.zk = false ∧ sts[j].merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") := by
  have hck' : ProvedScope.check s.read.tags s.read.zk s.read.I.circuitFile s.read.I.publicFile s.read.I.tables
      s.read.I.count s.read.publicInputs.isSome s.read.registered = .ok () := checksAtJ_scope hc
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

/-! ## The query round, at the parsed value -/

/-- **Level 0's query round in a session `verifyJWithMK keep` accepts**, under any oracle and whatever `keep`
(`plainQueryRound_verifyWithMK` at the record's parsed value). -/
theorem plainQueryRound_verifyJWithMK (keep : Bool → Bool) (hs : Hashes Flock.V) (spec : SessionSpec)
    (tabs : Array Setup) (j : Json) (proofs : Array ByteArray)
    (hf : ∀ st ∈ tabs, (∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog) ∧
      st.fold.kLog = st.kLog)
    (h : verifyJWithMK keep hs spec tabs j proofs = .ok ()) :
    ∃ sess : Session, Record.decodeJWithK keep hs spec j = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ sch : Flock.Schedule, fast100 tabs[i].m = some sch ∧ ∀ r < 2,
        ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧
          decodeProof (FlockSoundness.Refine.shapeOf tabs[i] sch extras.size) proofs[2 * i + r]! = .ok p ∧
          Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
          Claims.Flock.plainRepColsOf (Claims.Flock.plainProofShape tabs[i].m tabs[i].kLog (2 + extras.size)
              tabs[i].merkle.digestLen
              (if tabs[i].merkle.saltBytes = 0 && !tabs[i].openedSalts then none else some tabs[i].merkle.saltBytes))
              stream.rounds proofs[2 * i + r]! =
            Claims.Flock.zkRepCols (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins
              p.ligerito.initialProof.rows ∧
          merkleCheckWith (tabs[i].merkle.opsWith hs) tabs[i].merkle p.cap (sch.levels.getD 0 default).cap
            (sch.levels.getD 0 default).d
            (positions (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins)
            p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
            (2 ^ (sch.levels.getD 0 default).lanes) = .ok () ∧
          p.cap.foldl (· ++ ·) ByteArray.empty = sess.rootB[i]! := by
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verifyJWithMK_tables_cap keep hs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨hi', c, hsch, h0, h1⟩ := hall i hi
  obtain ⟨hfi, hkl⟩ := hf tabs[i] (Array.getElem_mem hi)
  refine ⟨schs[i]'hi', hsch, fun r hr2 => ?_⟩
  have hr : verifyRepWith hs spec tabs[i] (schs[i]'hi') sess i r proofs[2 * i + r]! = .ok c := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · simpa using h0
    · exact h1
  exact plainRepColsOf_verifyRepWith hs spec tabs[i] (schs[i]'hi') sess i r _ c hsch hfi hkl hr

/-! ## An accepted run's openings, and `ConflictTied` -/

/-- **What the reader takes of a `nextR` decision's table 0** (`runOf_eqJ` at `runOfR`): read off the decision's
setup, its live session at `jAt sha512` and the read proofs' files. -/
theorem runOfR_eq {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {lv : Live} {rr : Claims.Flock.ZkRun}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hlv : liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv) (hr : runOfR rd x as cs 0 = some rr) :
    ∃ st extras, sts[0]? = some st ∧ (Setup.ofCircuit st).extra lv.points = .ok extras ∧
      rr = { m := (Setup.ofCircuit st).m, kLog := (Setup.ofCircuit st).kLog, claims := 2 + extras.size,
             hashLen := (Setup.ofCircuit st).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit st),
             reps := [(lv.streams.getD (2 * 0) #[],
                       (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * 0) ByteArray.empty),
                      (lv.streams.getD (2 * 0 + 1) #[],
                       (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * 0 + 1) ByteArray.empty)] } := by
  unfold runOfR at hr
  rw [ha] at hr
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hr
  obtain ⟨st, hst, lv', hlv', extras, hext, rfl⟩ := hr
  rw [hlv] at hlv'
  obtain rfl : lv = lv' := Option.some.inj hlv'
  refine ⟨st, extras, hst, ?_, rfl⟩
  cases he : (Setup.ofCircuit st).extra lv.points with
  | error e => rw [he] at hext; cases hext
  | ok v => rw [he] at hext; cases hext; rfl

/-- **An accepted run of `VR rd` opens a column only as a Merkle opening** of table 0's level 0 in its own log
(`plainK_opening_raw`'s proof, at `nextR`'s verdict at the parsed value): a column `u` the run's reader opens at `q`
is an opened row whose opening under table 0's scheme against `root_B[0]` of the link `linkR` reads off the tape's
first message passed, every SHA-512 input of it in the run's verifier log. -/
theorem plainR_opening {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (r₀ : List ByteArray) (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      (rowsOf ((P.next x [] ω).eval sha512).1).map (List.map nodeOf) = some r₀ ∧ linkR sts r₀ = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P (VR rd) x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  rw [colsRV_run h] at hu
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accR_decide h
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  obtain ⟨-, hc, hs, -, -⟩ := decideR_sha512 hact hacc
  have hso := actR_setupOf hact
  have hv := verdictJ_within (decisionSF_runs hs hrun)
  obtain ⟨⟨s', spec', sts'⟩, c₁', r₀, rss, fl, link', pf, -, -, h6, -, h8, -, hd⟩ := decR_some (actR_decide hact)
  simp only [DecR.mk.injEq] at hd
  obtain ⟨rfl, rfl, rfl, rfl, -⟩ := hd
  obtain ⟨rest, hhead⟩ := run_head sha512 P (VR rd) x ω r h
  rw [hhead] at h6
  generalize hL : (Claims.run sha512 P (VR rd) x ω r).vLog = L at hv ⊢
  obtain ⟨-, lv, hlv, hcu⟩ := checksAtJ_ok hc
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  unfold colsR at hu
  split at hu
  swap
  · simp [Claims.Flock.lookCol] at hu
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOfR_eq hact hlv hrr
  have hmem := lookCol_mem hu
  obtain ⟨ρ, hρ, hp⟩ := List.mem_flatMap.1 (mem_of_distinctBelow hmem)
  obtain ⟨k, hk, rfl⟩ : ∃ k < 2, ρ = (lv.streams.getD (2 * 0 + k) #[],
      (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * 0 + k) ByteArray.empty) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hρ
    rcases hρ with rfl | rfl
    · exact ⟨0, by omega, rfl⟩
    · exact ⟨1, by omega, rfl⟩
  have hsts0 : 0 < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hst0 : sts[0] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, hdec, hsz, hreps⟩ := plainQueryRound_verifyJWithMK id (hashesIn (L.map Prod.fst)) spec
    (tabsOf sts) _ _ (tabsOf_fold sts) hv
  obtain ⟨sch, hsch, hrep⟩ := hreps 0 h0
  obtain ⟨stream, extras', p, hstream, hext', hp', hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [ht0] at hsch hext' hp' hlt hcols hmc
  obtain ⟨hrB, hpts, hlsz, hstreams⟩ := decodeJWithK_live id (hashesIn (L.map Prod.fst)) hdec hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hcolsEq : Claims.Flock.plainRepColsOf
      (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
      (lv.streams.getD (2 * 0 + k) #[])
      ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * 0 + k) ByteArray.empty) =
      Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcols]
    have e1 : lv.streams.getD (2 * 0 + k) #[] = lv.streams[2 * 0 + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    have e2 : (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * 0 + k) ByteArray.empty =
        (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine plainRepColsOf_congr _ _ fun i => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  have hp2 : (q, u) ∈ Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcolsEq]
    exact hp
  obtain ⟨j, hj, -, hjq, hju⟩ := mem_zkRepCols hp2
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checksAtJ_leaf hc hs 0 hsts0 with e | ⟨-, e⟩
    · rw [hst0] at e
      exact Or.inl e
    · rw [hst0] at e
      exact Or.inr e
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := FlockSoundness.Refine.merkleCheck_cap hpure
  refine ⟨s, spec, sts, st, sch, r₀, link, p.ligerito.initialProof.rows[j]!, hso, hst, hsch, h6, h8,
    ⟨hms, p.cap, p.ligerito.initialProof.salts.getD j ByteArray.empty,
      sibsOf p.ligerito.initialProof.siblings ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) j,
      hcs, fun b hb => (hcw b hb).trans hdl, ?_, ?_,
      fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure j b hb).trans hdl, ?_⟩, hju⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hjq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure j hj
  · rw [← hjq]
    exact hlog j hj

/-- **`ConflictTied` at the plain record at `VR rd`**: two accepted runs of the deployed reader at one tape that open
different columns at one position collide in their joined verifier logs, at SHA-512. Both read the link off the
tape's one first message (`plainR_opening`), so both openings climb to one `root_B`. -/
theorem plainR_conflictTied (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) :
    Claims.Flock.ConflictTied sha512 (deployPlainR rel rd) (wordLen false) := by
  intro x Ω P ω r r' q u v hacc hacc' _ hu hv huv
  obtain ⟨s, spec, sts, st, sch, r₀, link, row, hs, hst, hsch, hr₀, hl, ho, hu'⟩ :=
    plainR_opening (P := P) (r := r) hacc hu
  obtain ⟨s', spec', sts', st', sch', r₀', link', row', hs', hst', hsch', hr₀', hl', ho', hv'⟩ :=
    plainR_opening (P := P) (r := r') hacc' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hr₀] at hr₀'
  cases hr₀'
  rw [hl] at hl'
  cases hl'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

/-- **`KSFlockLin` at the plain record at `VR rd`, from the table bound alone**: `plainR_KSFlockLin` with its
conflicts tied to collisions (`plainR_conflictTied`). -/
theorem plainR_KSFlockLin_of_table (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (htab : Claims.Recursion.TableBoundAbortLin sha512 (deployPlainR rel rd) (wordLen false) (wide false) CpDep) :
    Claims.Recursion.KSFlockLin sha512 (deployPlainTrimR rel rd) CpDep :=
  plainR_KSFlockLin rel rd (plainR_conflictTied rel rd) htab

end FlockDeploy.KS
