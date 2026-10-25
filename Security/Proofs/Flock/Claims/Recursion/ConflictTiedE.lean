import Proofs.Flock.Claims.Recursion.FlockKSDeployE
import Proofs.Flock.Claims.KS.ConflictTied

/-!
# Conflicts tied to collisions at the element-level inner verifier (`FlockDeploy.VE`)

`ConflictTied` at the plain record with `VE` as its verifier (`plainE_conflictTied`), at SHA-512, by
`plain_conflictTied`'s proof at the element-level verdict. An accepted run of `VE` decides within its own verifier log
(`accE_decide`), so the element-level verdict accepts with SHA-512 answered only within that log (`accE_within`). Its
query round places the reader's columns among level 0's Merkle openings, as the shipped verdict's does
(`plainQueryRound_verifyWithMK`): the element-level verdict drops only digests of encodings, and keeps every rep's
checks and every hash call. The decoded session's `root_B` and rep streams are `liveOf`'s (`decodeWithK_rootB_live`,
`liveOf_decodeWithK`), whatever `keep`, so each opening climbs to the link's `root_B` (`plainE_opening`), and two
accepted runs at one tape that open different columns at one position collide in their joined logs.

That discharges (P1)'s first premise at `VE`: (P1) at the bound record at `VE`'s steps
(`RecDeploy.plainKSAbort_bdE`) rests on the table bound counted per round at `deployPlainE rel` alone, proved at
`deployPlain rel` (`plain_tableBoundθOnAccR`'s) and not yet at `VE`.
-/

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The element-level decoder's `root_B` -/

/-- **What the verifier's decoder reads of each table's `root_B`**, whatever the hashes and `keep` (S5's parse). -/
theorem decodeWithK_rootB (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : Session} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∃ j l, Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧
      List.Forall₂ (Claims.Bridge.Custody.RootField l) spec.tables.toList sess.rootB.toList := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, by assumption, by assumption,
      List.Forall₂.imp (fun t rb h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))⟩
    beta_reduce at h
    obtain ⟨v, h1, h⟩ := ebind_ok h
    obtain ⟨s, h2, h⟩ := ebind_ok h
    obtain ⟨rb', h3, h⟩ := ebind_ok h
    repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨v, s, h1, h2, h3⟩

/-- **The element-level decoder's `root_B` is `liveOf`'s**, whatever the hashes and `keep`. -/
theorem decodeWithK_rootB_live (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : Session} {lv : Live} (hd : Record.decodeWithK keep hs spec text = .ok sess)
    (hl : liveOf spec text = .ok lv) : sess.rootB = lv.rootB := by
  obtain ⟨j, l, hp, hl1, hroot⟩ := decodeWithK_rootB keep hs hd
  obtain ⟨j', l', recs', hp', hl1', -, hroot', -, -⟩ := Claims.Bridge.Custody.liveOf_ok hl
  rw [hp] at hp'
  cases hp'
  rw [hl1] at hl1'
  cases hl1'
  exact Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := Claims.Bridge.Custody.RootField l)
    (fun _ _ _ => Claims.Bridge.Custody.rootField_unique) hroot hroot')

/-! ## The query round, at the element-level verdict -/

/-- **Level 0's query round in a session `verifyWithMK keep` accepts**, under any oracle and whatever `keep`:
`plainQueryRound_verifyWithM`'s conclusion, with the session `Record.decodeWithK keep` decodes. -/
theorem plainQueryRound_verifyWithMK (keep : Bool → Bool) (hs : Hashes Flock.V) (spec : SessionSpec)
    (tabs : Array Setup) (record : String) (proofs : Array ByteArray)
    (hf : ∀ st ∈ tabs, (∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog) ∧
      st.fold.kLog = st.kLog)
    (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) :
    ∃ sess : Session, Record.decodeWithK keep hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
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
  obtain ⟨sess, schs, hsess, hsz, -, hall⟩ := verifyWithMK_tables keep hs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨hi', c₀, c₁, hsch, h0, h1⟩ := hall i hi
  obtain ⟨hfi, hkl⟩ := hf tabs[i] (Array.getElem_mem hi)
  refine ⟨schs[i]'hi', hsch, fun r hr2 => ?_⟩
  have hr : ∃ c, verifyRepWith hs spec tabs[i] (schs[i]'hi') sess i r proofs[2 * i + r]! = .ok c := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · exact ⟨c₀, by simpa using h0⟩
    · exact ⟨c₁, h1⟩
  obtain ⟨c, hc⟩ := hr
  exact plainRepColsOf_verifyRepWith hs spec tabs[i] (schs[i]'hi') sess i r _ c hsch hfi hkl hc

/-! ## An accepted run of `VE`, within its own log -/

/-- **An accepted run of `VE` decides, within its own verifier log** (`acc_decide`'s proof, at `nextE`). -/
theorem accE_decide {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} (h : (Claims.run H P VE x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act false x ((Claims.run H P VE x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run H P VE x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      Runs H (decisionE s x link c₁ mids f) () ((Claims.run H P VE x ω r).vLog.map Prod.fst) := by
  have ha := FlockDeploy.KS.run_acc_next H P VE x ω r h
  have hl := run_acc_log H P VE x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha hl
  replace ha : ((nextE x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  replace hl : ∀ p ∈ ((nextE x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs).eval H).2,
      p ∈ (Claims.run H P VE x ω r).vLog := hl
  cases hact : act false x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs with
  | rej => rw [nextE_rej H hact] at ha; cases ha
  | coin c => rw [nextE_coin H hact] at ha; cases ha
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_, fun p hp => List.mem_map_of_mem (hl p ?_)⟩
    · simp only [nextE, hact] at ha
      exact outOf_acc H _ ha
    · simp only [nextE, hact, outOf_log]
      exact hp

/-- **An element-level decision within `L`**, at SHA-512 on a statement that sets up at `(spec, sts)`: the
element-level verdict within `L`, and the checks (`decision_runs`'s proof). -/
theorem decisionE_runs {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {L : List Claims.Msg} (hs : s.setup = .ok (spec, sts)) (h : Runs sha512 (decisionE s x link c₁ mids f) () L) :
    Runs sha512 (verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs) () L ∧
      checks false s spec x link c₁ mids f = .ok () := by
  unfold decisionE at h
  obtain ⟨⟨spec', sts'⟩, h0, h⟩ := Runs.bind h
  have e : setupWith (Claims.hOf sha512) s.read = .ok (spec', sts') := by
    rw [← (setupM_eval sha512 s.read).1]
    exact h0.1
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at e
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj e)
  unfold decisionAtE at h
  obtain ⟨⟨⟩, h1, h2⟩ := Runs.bind h
  exact ⟨h1, Runs.lift h2⟩

/-- **The element-level verdict within `L` is the element-level verdict with SHA-512 answered only within `L`.** -/
theorem verdictE_within {spec : SessionSpec} {tabs : Array Setup} {record : String} {proofs : Array ByteArray}
    {L : List Claims.Msg} (h : Runs sha512 (verifyWithME Claims.Hashes.oc spec tabs record proofs) () L) :
    verifyWithME (hashesIn L) spec tabs record proofs = .ok () :=
  (With.verifyWithME_sim (within L) (within_hashes L) spec tabs record proofs) () h

/-- **An accepted run of `VE` at `sha512`** (`acc_within`'s proof): the statement sets up, the run decides on its last
message, read as a final message on the statement, and on the link of the prover's first message; the custody checks
pass; and the element-level verdict accepts with SHA-512 answered only within the run's verifier log. -/
theorem accE_within {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin (VE.k x) → Coin}
    (h : (Claims.run sha512 P VE x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P VE x ω r).seen = some f ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧ checks false s spec x link c₁ mids f = .ok () ∧
      verifyWithME (hashesIn ((Claims.run sha512 P VE x ω r).vLog.map Prod.fst)) spec (tabsOf sts) f.record
        f.proofs = .ok () := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accE_decide h
  obtain ⟨hv, hc⟩ := decisionE_runs (setupOf_some (act_decide hact)).2 hrun
  obtain ⟨rest, hhead⟩ := run_head sha512 P VE x ω r h
  obtain ⟨rest', has, hlink⟩ := act_decide_link hact
  rw [hhead] at hlink
  refine ⟨s, spec, sts, link, c₁, mids, f, act_decide hact, ?_, hlink, hc, verdictE_within hv⟩
  have hl := act_decide_final hact
  rw [List.getLast?_map, Option.bind_map] at hl
  exact hl

/-! ## An accepted run's openings, and `ConflictTied` -/

/-- **An accepted run of `VE` opens a column only as a Merkle opening** of table 0's level 0 in its own log
(`plain_opening_raw`'s proof, at the element-level verdict): a column `u` a rep opens at `q` is an opened row whose
opening under table 0's scheme against `root_B[0]` of the tape's first message's link passed, every SHA-512 input of it
in the run's verifier log. -/
theorem plainE_opening_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P VE x ω r).out = .acc)
    (hu : (q, u) ∈ plainRawCols (Claims.run sha512 P VE x ω r).seen) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P VE x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := accE_within h
  generalize hL : (Claims.run sha512 P VE x ω r).vLog = L at hv ⊢
  obtain ⟨hx, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  unfold plainRawCols at hu
  split at hu
  swap
  · simp at hu
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eq hf hx hs hlv hrr
  obtain ⟨ρ, hρ, hp⟩ := List.mem_flatMap.1 hu
  obtain ⟨k, hk, rfl⟩ : ∃ k < 2, ρ = (lv.streams.getD k #[], f.proofs.getD k ByteArray.empty) := by
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
  replace hv : verifyWithMK (fun _ => true) (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs =
      .ok () := hv
  obtain ⟨sess, hdec, hsz, hreps⟩ := plainQueryRound_verifyWithMK (fun _ => true) (hashesIn (L.map Prod.fst)) spec
    (tabsOf sts) f.record f.proofs (tabsOf_fold sts) hv
  obtain ⟨sch, hsch, hrep⟩ := hreps 0 h0
  obtain ⟨stream, extras', p, hstream, hext', hp', hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [ht0] at hsch hext' hp' hlt hcols hmc
  have hrB := decodeWithK_rootB_live (fun _ => true) (hashesIn (L.map Prod.fst)) hdec hlv
  obtain ⟨hpts, -, hlsz, hstreams⟩ := liveOf_decodeWithK (fun _ => true) (hashesIn (L.map Prod.fst)) hlv hdec
  rw [← hpts, hext] at hext'
  cases hext'
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  have hk2 : 2 * 0 + k < lv.streams.size := by rw [hlsz]; exact hk'
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk' hk2
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2) hfa
  have hcolsEq : Claims.Flock.plainRepColsOf
      (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
      (lv.streams.getD k #[]) (f.proofs.getD k ByteArray.empty) =
      Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcols]
    have e1 : lv.streams.getD k #[] = lv.streams[2 * 0 + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]; simp
    have e2 : f.proofs.getD k ByteArray.empty = f.proofs[2 * 0 + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]; simp
    rw [e1, e2]
    refine plainRepColsOf_congr _ _ fun j => ?_
    rw [getElem!_pos lv.streams (2 * 0 + k) hk2, ← Array.getElem?_toList, ← Array.getElem?_toList]
    exact hcoins j
  have hp2 : (q, u) ∈ Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcolsEq]
    exact hp
  obtain ⟨j, hj, -, hjq, hju⟩ := mem_zkRepCols hp2
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checks_leaf hc (setupOf_some hs).2 0 hsts0 with h | ⟨-, h⟩
    · rw [hst0] at h
      exact Or.inl h
    · rw [hst0] at h
      exact Or.inr h
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := FlockSoundness.Refine.merkleCheck_cap hpure
  refine ⟨s, spec, sts, st, sch, link, p.ligerito.initialProof.rows[j]!, hs, hst, hsch, hlink,
    ⟨hms, p.cap, p.ligerito.initialProof.salts.getD j ByteArray.empty,
      sibsOf p.ligerito.initialProof.siblings ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) j,
      hcs, fun b hb => (hcw b hb).trans hdl, ?_, ?_,
      fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure j b hb).trans hdl, ?_⟩, hju⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hjq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure j hj
  · rw [← hjq]
    exact hlog j hj

/-- **An accepted run of `VE` opens a column only as a Merkle opening** of table 0's level 0 in its own log: a column
`u` the run's reader opens at `q` is one a rep opens (`plainE_opening_raw`). -/
theorem plainE_opening {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin (VE.k x) → Coin}
    {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P VE x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (cols false (Claims.run sha512 P VE x ω r).seen 0) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P VE x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of :=
  plainE_opening_raw h (mem_plainRawCols (lookCol_mem hu))

/-- **`ConflictTied` at the plain record at `VE`**: two accepted runs of the element-level verifier at one tape that
open different columns at one position collide in their joined verifier logs, at SHA-512. -/
theorem plainE_conflictTied (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ConflictTied sha512 (deployPlainE rel) (wordLen false) := by
  intro x Ω P ω r r' q u v hacc hacc' _ hu hv huv
  obtain ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, ho, hu'⟩ :=
    plainE_opening (P := P) (r := r) hacc hu
  obtain ⟨s', spec', sts', st', sch', link', row', hs', hst', hsch', hlink', ho', hv'⟩ :=
    plainE_opening (P := P) (r := r') hacc' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

/-- **(P1) at `VE`, from the table bound alone**: `plainE_KSAbortR` with its conflicts tied to collisions
(`plainE_conflictTied`). -/
theorem plainE_KSAbort_of_table (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAccR sha512 (deployPlainE rel) (wordLen false) (wide false))
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK ((deployPlainE rel).mapCols trimCol) decode) :
    (Claims.Recursion.Inner.ofFlock ((deployPlainE rel).mapCols trimCol) decode).KSAbort sha512 :=
  plainE_KSAbortR rel (plainE_conflictTied rel) htab decode hdec

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **(P1) at the bound record at `VE`'s steps** (option (b)), from the table bound counted per round at the plain
record at `VE` (`FlockDeploy.KS.plainE_KSAbort_of_table`), whatever the bound record's other choices. -/
theorem plainKSAbort_bdE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAccR FlockDeploy.sha512 (FlockDeploy.deployPlainE rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (join : List Msg → Msg) (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :
    (bdDeployed rel FlockDeploy.VE.next (FlockDeploy.deployPlainTrim rel).cols
      (FlockDeploy.deployPlainTrim rel).weight join rowIn final w).PlainKSAbort FlockDeploy.sha512 :=
  fun decI hdec => FlockDeploy.KS.plainE_KSAbort_of_table rel htab decI hdec

end RecDeploy
