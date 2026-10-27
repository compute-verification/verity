import Proofs.Flock.Claims.KS.PlainQueried
import Proofs.Flock.Claims.KS.PlainForkClash
import Proofs.Flock.Claims.Recursion.CapLinkRd
import Proofs.Flock.Claims.Recursion.AbortFork
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The deployed plain record's caps, and what its runs read at them

`KSFlockLin` charges the recursive caps' openings to `FC` at a `Caps` record: each position's round and which coins
query it. Here is that record at the deployed plain verifier, and the read sets `Pr_rdOff_le_capHit` takes there.

* `plain_liveRow`: at an accepted run, a later-level opening that a decoded live output records is one of the run's
  own level columns (`plain_liveOpening`'s proof, stopped at the columns).
* `CpDep`: one position per rep `r < 2`, level `ℓ + 1` below table 0's `fast100` levels and position of that level;
  its round is the plain fork round `forkRound`, and its coins query it where the level's query coin
  (`plainQRound`, `plainQN`) draws it. A statement without a plain table-0 schedule has none.
* `rdAt`: what an aborting run reads at a position, the rows its level columns hold there where it accepts, at any
  lane count.
* `Pr_rdAt_le_capHit`: `Pr_rdOff_le_capHit` at `CpDep` and `rdAt`. A run reads something exactly where it opens the
  position (`plain_levelCols_queried`), and two runs sharing the coins before the round reading two rows there tie a
  collision (`plain_levelTie`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **An accepted run's recorded later openings are its own level columns**: at the concrete simulator's coins where
`P`'s run accepts and `verify` accepts what `wrPD` writes for the live outcome, in an output whose rep `k` records the
deployed rep's later openings (`LaterEntriesAt`), each level-`ℓ ≥ 1` opening `(q, v)` is one of the run's level-`ℓ`
columns of rep `k`: `plain_liveOpening`'s proof up to `rawLevelCols_plain`. -/
theorem plain_liveRow {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) {u : plainTape x}
    {r' : Fin (plainN x + 2) → LCoin 65536}
    (hacc : (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).out = .acc)
    (hA : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN
      (plainDg spec) (plainW spec (plain_mPts hs hin hset).1) sha512 (plainSim P x spec st) x (ω, u) r')))
    {sch : Accounting.Schedule} {out : Model.OutCL 𝔽 Dig (List UInt8)} {k : ℕ} (hk : k < 2)
    (hL : LaterEntriesAt sch ins.read.I d (wrPD ins.read.I d (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN
      (plainDg spec) (plainW spec (plain_mPts hs hin hset).1) sha512 (plainSim P x spec st) x (ω, u) r')) out k) :
    ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensL out k ℓ (lvOf sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
      (q, v) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen k ℓ := by
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', -, s₀, p, hs₀, -, -, hent⟩ := hL
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := acc_decide hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRun P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hA hdec hent
  have hacc' := run_acc_next sha512 P (V false) x ω _ hacc
  replace hacc' : ((next false x ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc'
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hsu₀⟩ := setupOf_some (act_decide hact)
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtD_wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hA
  obtain ⟨-, hrs⟩ := hall spec #[st'] sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨k, hk⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decision_runs hsu₀ hrun
  have hv' : verifyWithM (hashesIn ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdict_within hvr
  have hcols := rawLevelCols_plain (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1 hk
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ y : StreamRec,
      y.rounds.toList = ((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec) →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro y hy j
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  rw [hpr'] at hent
  intro ℓ h1 q v hqv
  rw [hent ℓ h1, plainLevelColsOf_congr _ _ _ (hco y₀ hr₀), ← getD_empty, ← hcols ℓ] at hqv
  exact hqv

/-! ## The deployed caps -/

/-- **The plain table-0 setup of `x`**: its session spec, table 0's statement and that statement's `fast100`
schedule, where they exist. -/
def depSetup (x : Claims.Msg) : Option (SessionSpec × Stmt × Flock.Schedule) :=
  (setupOf false x).bind fun t => (t.2.2[0]?).bind fun st => (fast100 st.m).map fun sch => (t.2.1, st, sch)

theorem depSetup_eq {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {sch : Flock.Schedule} (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st)
    (hsch : fast100 st.m = some sch) : depSetup x = some (spec, st, sch) := by
  simp [depSetup, hs, hst, hsch]

theorem depSetup_some {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} {sch : Flock.Schedule}
    (hd : depSetup x = some (spec, st, sch)) :
    ∃ s sts, setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch := by
  unfold depSetup at hd
  rcases hs : setupOf false x with _ | ⟨s, spec', sts⟩
  · simp [hs] at hd
  rcases hst : sts[0]? with _ | st'
  · simp [hs, hst] at hd
  rcases hsch : fast100 st'.m with _ | sch'
  · simp [hs, hst, hsch] at hd
  simp only [hs, hst, hsch, Option.bind_some, Option.map_some, Option.some.injEq, Prod.mk.injEq] at hd
  obtain ⟨rfl, rfl, rfl⟩ := hd
  exact ⟨s, sts, rfl, hst, hsch⟩

/-- The cap slots of `x`: one per level after level 0 but the last, whose cap is the next level's root. -/
def depL (x : Claims.Msg) : ℕ :=
  match depSetup x with
  | some (_, _, sch) => sch.levels.size - 1
  | none => 0

/-- The positions of level `ℓ` of `x`'s schedule are `2 ^ depLogLen x ℓ`. -/
def depLogLen (x : Claims.Msg) (ℓ : ℕ) : ℕ :=
  match depSetup x with
  | some (_, _, sch) => (lvOf (schedOf sch) ℓ).logLen
  | none => 0

/-- **A cap position of `x`**: a rep `r < 2`, a slot `ℓ` (level `ℓ + 1`) and a position of that level. -/
abbrev DepIdx (x : Claims.Msg) : Type := Σ (_ : Fin 2) (ℓ : Fin (depL x)), Fin (2 ^ depLogLen x (ℓ.val + 1))

/-- A position's round: the plain fork round of its rep and slot. -/
def depRound (x : Claims.Msg) (j : DepIdx x) : ℕ :=
  match depSetup x with
  | some (spec, st, sch) => forkRound spec st sch.k0 j.1.val j.2.1.val
  | none => 0

/-- Whether coins `cs` query a position: its level's query coin draws it. -/
def depOpens (x : Claims.Msg) (j : DepIdx x) (cs : ℕ → Coin) : Bool :=
  match depSetup x with
  | some (spec, st, sch) =>
    decide (j.2.2.val ∈ positions (sch.levels.getD (j.2.1.val + 1) default)
      ((cs (plainQRound spec st sch j.1.val (j.2.1.val + 1))).vals (plainQN spec st sch (j.2.1.val + 1))))
  | none => false

/-- **The deployed plain record's caps**: `DepIdx x`'s positions, numbered. -/
noncomputable def CpDep : Claims.Recursion.Caps Coin where
  width x := Fintype.card (DepIdx x)
  round x i := depRound x ((Fintype.equivFin (DepIdx x)).symm i)
  opens x i cs := depOpens x ((Fintype.equivFin (DepIdx x)).symm i) cs

/-! ## What an aborting run reads at a position -/

open Classical in
/-- **What an aborting run reads at rep `r`'s level `ℓ + 1` position `q`**, as rows of `lanes` lanes: where it
accepts, the rows its level columns hold at `q`; nothing elsewhere. -/
noncomputable def rdAt {Ω A : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (keep : Ω → (Fin ((V false).k x) → Coin) → A → Bool) (r ℓ q lanes : ℕ) (ω : Ω)
    (d : (Fin ((V false).k x) → Coin) × A) : Finset (Fin (2 ^ lanes) → 𝔽) :=
  if (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).out = .acc then
    (((plainRawLevelCols (Claims.Recursion.abortRun sha512 P (V false) x keep ω d).seen r (ℓ + 1)).filter
      fun e => e.1 = q).map fun e => rowFin lanes e.2).toFinset
  else ∅

open Classical in
/-- **What an aborting run reads at rep `r`'s level `ℓ + 1` position `q`**, as rows of `lanes` lanes: where it
accepts, the rows its level columns hold at `q`; nothing elsewhere. -/
noncomputable def rdAtK (keepV : Bool → Bool) {Ω A : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (keep : Ω → (Fin ((V false).k x) → Coin) → A → Bool) (r ℓ q lanes : ℕ) (ω : Ω)
    (d : (Fin ((V false).k x) → Coin) × A) : Finset (Fin (2 ^ lanes) → 𝔽) :=
  if (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).out = .acc then
    (((plainRawLevelCols (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).seen r (ℓ + 1)).filter
      fun e => e.1 = q).map fun e => rowFin lanes e.2).toFinset
  else ∅

theorem mem_rdAtK (keepV : Bool → Bool) {Ω A : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {keep : Ω → (Fin ((V false).k x) → Coin) → A → Bool} {r ℓ q lanes : ℕ} {ω : Ω}
    {d : (Fin ((V false).k x) → Coin) × A} {u : Fin (2 ^ lanes) → 𝔽} :
    u ∈ rdAtK keepV P x keep r ℓ q lanes ω d ↔ keep ω d.1 d.2 = true ∧
      (Claims.run sha512 P (VK keepV) x ω d.1).out = .acc ∧
      ∃ a, (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (VK keepV) x ω d.1).seen r (ℓ + 1) ∧
        rowFin lanes a = u := by
  unfold rdAtK
  by_cases hacc : (Claims.Recursion.abortRun sha512 P (VK keepV) x keep ω d).out = .acc
  · obtain ⟨hk, hr⟩ := (Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := VK keepV) (x := x)
      (keep := keep) (ω := ω) (d := d)).1 hacc
    rw [if_pos hacc, Claims.Recursion.abortRun_of_keep (H := sha512) (P := P) (V := VK keepV) (x := x)
      (keep := keep) (ω := ω) (t := d) hk]
    simp only [List.mem_toFinset, List.mem_map, List.mem_filter, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨q', a⟩, ⟨hm, rfl⟩, rfl⟩
      exact ⟨hk, hr, a, hm, rfl⟩
    · rintro ⟨-, -, a, hm, rfl⟩
      exact ⟨(q, a), ⟨hm, rfl⟩, rfl⟩
  · rw [if_neg hacc]
    simp only [Finset.notMem_empty, false_iff, not_and]
    intro hk hr
    exact absurd ((Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := VK keepV) (x := x)
      (keep := keep) (ω := ω) (d := d)).2 ⟨hk, hr⟩) hacc

/-- `mem_rdAtK` at the deployed plain verifier (`keep := id`). -/
theorem mem_rdAt {Ω A : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {keep : Ω → (Fin ((V false).k x) → Coin) → A → Bool} {r ℓ q lanes : ℕ} {ω : Ω}
    {d : (Fin ((V false).k x) → Coin) × A} {u : Fin (2 ^ lanes) → 𝔽} :
    u ∈ rdAt P x keep r ℓ q lanes ω d ↔ keep ω d.1 d.2 = true ∧
      (Claims.run sha512 P (V false) x ω d.1).out = .acc ∧
      ∃ a, (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω d.1).seen r (ℓ + 1) ∧
        rowFin lanes a = u :=
  mem_rdAtK id (Ω := Ω) (A := A) (P := P) (x := x) (keep := keep) (r := r) (ℓ := ℓ) (q := q) (lanes := lanes) (ω := ω) (d := d) (u := u)

/-- **`CpDep`'s position `i` at a statement with a plain table-0 schedule**: its round and query coins are its
decoded rep, slot and position's. -/
theorem CpDep_at {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {sch : Flock.Schedule} (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st)
    (hsch : fast100 st.m = some sch) (i : Fin (CpDep.width x)) :
    CpDep.round x i = forkRound spec st sch.k0 ((Fintype.equivFin (DepIdx x)).symm i).1.val
        ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ∧
      ∀ cs, CpDep.opens x i cs = decide (((Fintype.equivFin (DepIdx x)).symm i).2.2.val ∈
        positions (sch.levels.getD (((Fintype.equivFin (DepIdx x)).symm i).2.1.val + 1) default)
          ((cs (plainQRound spec st sch ((Fintype.equivFin (DepIdx x)).symm i).1.val
            (((Fintype.equivFin (DepIdx x)).symm i).2.1.val + 1))).vals
            (plainQN spec st sch (((Fintype.equivFin (DepIdx x)).symm i).2.1.val + 1)))) := by
  have hd := depSetup_eq hs hst hsch
  refine ⟨?_, fun cs => ?_⟩
  · show depRound x _ = _
    unfold depRound
    rw [hd]
  · show depOpens x _ cs = _
    unfold depOpens
    rw [hd]

/-- The slots of a statement with a plain table-0 schedule are its levels after level 0 but the last. -/
theorem depL_eq {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {sch : Flock.Schedule} (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st)
    (hsch : fast100 st.m = some sch) : depL x = sch.levels.size - 1 := by
  unfold depL
  rw [depSetup_eq hs hst hsch]

/-- **A cap position's read sets at the deployed plain verifier, linearly**: `Pr_rdOff_le_capHit` at `CpDep`'s
position `i` and `rdAt` at its rep, slot and position. Where a run reads something it accepts and its coins draw the
position (`plain_levelCols_queried`), and two runs sharing the coins before the plain fork round reading two rows
there tie a collision (`plain_levelTie`). -/
theorem Pr_rdAt_le_capHitK (keepV : Bool → Bool) {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (P : Claims.Prover Ω)
    (x : Claims.Msg) (keep : Ω → (Fin ((V false).k x) → Coin) → A → Bool) (M : ℕ) (i : Fin (CpDep.width x))
    (lanes : ℕ) (ω : Ω) :
    (Claims.Pr fun d => (rdAtK keepV P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val
        ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω d).Nonempty ∧
      rdAtK keepV P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val ((Fintype.equivFin (DepIdx x)).symm i).2.1.val
          ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω d ≠
        {Claims.Recursion.capPlurRd (VK keepV) x (CpDep.round x i)
          (rdAtK keepV P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val ((Fintype.equivFin (DepIdx x)).symm i).2.1.val
            ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω) d}) ≤
      (Claims.Pr fun s : ((Fin ((V false).k x) → Coin) × A) × (Fin M → (Fin ((V false).k x) → Coin) × A) =>
        ∃ u v, (Claims.Recursion.FC sha512 (VK keepV) CpDep P x keep M i.pos).out (i, ω, s.1, s.2) = some (u, v) ∧
          Claims.Collision sha512 u v) + 1 / ((M : ℝ) + 1) := by
  set j := (Fintype.equivFin (DepIdx x)).symm i with hj
  have hW : 0 < depL x := by
    have := j.2.1.isLt
    omega
  obtain ⟨⟨spec, st, sch⟩, hd⟩ : ∃ t, depSetup x = some t := by
    unfold depL at hW
    rcases hd : depSetup x with _ | t
    · rw [hd] at hW
      exact absurd hW (Nat.lt_irrefl 0)
    · exact ⟨t, rfl⟩
  obtain ⟨s, sts, hs, hst, hsch⟩ := depSetup_some hd
  obtain ⟨hround, hopens⟩ := CpDep_at hs hst hsch i
  have hℓ : j.2.1.val + 1 < sch.levels.size := by
    have := j.2.1.isLt
    have := depL_eq hs hst hsch
    omega
  refine Claims.Recursion.Pr_rdOff_le_capHit sha512 (VK keepV) CpDep P x keep M i ω _ (fun d => ?_)
    (fun d d' u u' hc hu hu' hne => ?_)
  · constructor
    · rintro ⟨u, hu⟩
      obtain ⟨hk, hr, a, ha, -⟩ := (mem_rdAtK keepV).1 hu
      refine ⟨(Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := VK keepV) (x := x) (keep := keep)
        (ω := ω) (d := d)).2 ⟨hk, hr⟩, ?_⟩
      rw [hopens, decide_eq_true_eq]
      exact (plain_levelCols_queriedK keepV hr hs hst hsch j.1.isLt (by omega) hℓ _).1 ⟨a, ha⟩
    · rintro ⟨hacc, hq⟩
      obtain ⟨hk, hr⟩ := (Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := VK keepV) (x := x)
        (keep := keep) (ω := ω) (d := d)).1 hacc
      rw [hopens, decide_eq_true_eq] at hq
      obtain ⟨a, ha⟩ := (plain_levelCols_queriedK keepV hr hs hst hsch j.1.isLt (by omega) hℓ _).2 hq
      exact ⟨_, (mem_rdAtK keepV).2 ⟨hk, hr, a, ha, rfl⟩⟩
  · obtain ⟨hk, hr, a, ha, rfl⟩ := (mem_rdAtK keepV).1 hu
    obtain ⟨hk', hr', a', ha', rfl⟩ := (mem_rdAtK keepV).1 hu'
    rw [Claims.Recursion.abortRun_of_keep (H := sha512) (P := P) (V := VK keepV) (x := x) (keep := keep) (ω := ω)
        (t := d) hk,
      Claims.Recursion.abortRun_of_keep (H := sha512) (P := P) (V := VK keepV) (x := x) (keep := keep) (ω := ω)
        (t := d') hk']
    rw [hround] at hc
    exact plain_levelTieK keepV hr hr' hs hst hsch j.1.isLt hℓ hc ha ha' fun he => hne (by rw [he])

/-- `Pr_rdAt_le_capHitK` at the deployed plain verifier (`keep := id`). -/
theorem Pr_rdAt_le_capHit {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (P : Claims.Prover Ω)
    (x : Claims.Msg) (keep : Ω → (Fin ((V false).k x) → Coin) → A → Bool) (M : ℕ) (i : Fin (CpDep.width x))
    (lanes : ℕ) (ω : Ω) :
    (Claims.Pr fun d => (rdAt P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val
        ((Fintype.equivFin (DepIdx x)).symm i).2.1.val ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω d).Nonempty ∧
      rdAt P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val ((Fintype.equivFin (DepIdx x)).symm i).2.1.val
          ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω d ≠
        {Claims.Recursion.capPlurRd (V false) x (CpDep.round x i)
          (rdAt P x keep ((Fintype.equivFin (DepIdx x)).symm i).1.val ((Fintype.equivFin (DepIdx x)).symm i).2.1.val
            ((Fintype.equivFin (DepIdx x)).symm i).2.2.val lanes ω) d}) ≤
      (Claims.Pr fun s : ((Fin ((V false).k x) → Coin) × A) × (Fin M → (Fin ((V false).k x) → Coin) × A) =>
        ∃ u v, (Claims.Recursion.FC sha512 (V false) CpDep P x keep M i.pos).out (i, ω, s.1, s.2) = some (u, v) ∧
          Claims.Collision sha512 u v) + 1 / ((M : ℝ) + 1) :=
  Pr_rdAt_le_capHitK id (Ω := Ω) (A := A) (P := P) (x := x) (keep := keep) (M := M) (i := i) (lanes := lanes) (ω := ω)

end FlockDeploy.KS
