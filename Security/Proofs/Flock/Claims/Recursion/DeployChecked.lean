import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Complete.Walk
import Proofs.Flock.Claims.Complete.Json.UnitDraw
import Proofs.Flock.Claims.Recursion.Assumptions.PlainLeaves

/-!
# The checked reader: `Decomp` under Flock's admission of plain leaves

`readX` requires Flock's own setup and statement check on the inner statement and its draw (`innerOk`), besides the
inner setup at the recursion's inner tags (`RecVStar.innerOf`). `innerOf` reads plain leaves off the circuit file, and
`ProvedScope.check` passes them at hidden outputs without `--zk` (`inner_check_plain`), under Flock's
soundness headline at the statement's leaf. What the recursion needs of an inner statement is one Prop, stated in
`Assumptions/PlainLeaves.lean`; this file proves what follows from it.

* **`PlainLeavesAdmitted rel`**: some inner statement passes every check the reader makes of it (it decodes, its
  partition reads, `RecVStar.innerOf` sets it up and `innerOk` passes, so it carries a draw) and is small enough that
  Flock's completeness error is below 1 (`FlockComplete.δc_sha512_lt_one`), is in Flock's scope, where Flock's soundness
  headline holds (`Claims.Flock.Scope`), and is true (`Claims.Flock.R`): facts about one concrete inner statement,
  which a recorded run measured.
* **`Decomp` at the checked reader** (`decomp_checked_iff`): at every statement the reader parses, the inner statement
  passes Flock's statement check and carries a draw (`parse_innerOk`), so the cut `decomp_deployed_iff` asks of the
  parts is the inner run's conditions alone (`InnerRun`, `RunCut`: the link and last messages, the verdict, the record's
  statement and draw, the custody, the record's canonical text), every other condition of `FlockDeploy.checks` being the reader's (`checks_iff`).
  The inner statement's draw (`InnerDrawn`) holds at the checked reader (`innerDrawn_checked`): the
  recursion's inner session carries the verifier's full stratified draw, `readX` refuses an inner statement without a
  draw (`innerOk`), and the compiled verifier refuses, last in each V* session's verdict, a session over an inner
  session whose record holds none (`RecSetup.innerDrawn`, `fullDrawO`).
* **The inner acceptance is satisfiable under the Prop** (`admitted_acc`): Flock's completeness at a true in-scope
  statement whose error is below 1 gives an accepted run of its prover (`exists_of_pr_pos`), whose own messages replayed
  are accepted (`run_replay_out`), so `InnerAccSha` holds at some coins and messages (`innerAccSha_of_complete`). At
  `Readable.relation` that completeness is `FlockComplete.Replay.complete_plain` (`admitted_acc_plain`). The reader's
  draw tie at one table (`drawTiedB`) is no part of the Prop: in scope, the setup writes the statement's draw into the
  header the session's draw is read from (`drawTied_of_scope`).
* **The inner readiness the deployed stop reads** (`RecDeploy.stopsRec`): at a statement in scope, `FlockComplete.ready`
  at SHA-512 is three facts of the inner session, its size, its regions and its honest record
  (`FlockComplete.RecordFitsAt`), since the scope check already sets it up with a draw (`ready_inScope_iff`); at the
  inner statement the Prop names, it holds (`ready_of_admitted`).
-/

namespace RecDeploy

open Claims (Msg Out Prover Trace run replay stepWith)
open FlockDeploy (Coin)
open Lean (Json)

/-! ## Replaying a prover's own messages -/

/-- **Replaying a prover's own messages gives its decision**, from one history and traces of one length: at each round
the replayed message is the prover's, so the verifier answers alike. -/
theorem stepWith_replay_out {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vn : List Msg → Claims.OC Out) (ms : List Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Trace), hist.length = t.msgs.length →
      t'.msgs.length = t.msgs.length →
      (∀ j, t.msgs.length ≤ j → j < (stepWith H P x ω vn n hist as t).msgs.length →
        ms.getD j [] = ((stepWith H P x ω vn n hist as t).msgs.map Prod.fst).getD j []) →
      (stepWith H (replay ms) x () vn n hist as t').out = (stepWith H P x ω vn n hist as t).out
  | 0, _, _, _, _, _, _, _ => rfl
  | n + 1, hist, as, t, t', hl, hl', h => by
    have hlt := Claims.Recursion.stepWith_length_gt H P x ω vn (Nat.succ_pos n) hist as t
    have h2 : ((stepWith H P x ω vn (n + 1) hist as t).msgs.map Prod.fst)[t.msgs.length]? =
        some ((P.next x hist ω).eval H).1 := by
      have := congrArg (·[t.msgs.length]?) (Claims.Recursion.stepWith_succ_fst_take H P x ω vn n hist as t)
      simpa [List.getElem?_take] using this
    have hrep : ms.getD hist.length [] = ((P.next x hist ω).eval H).1 := by
      rw [hl, h _ le_rfl hlt, List.getD_eq_getElem?_getD, h2]
      rfl
    have e1 : ((replay ms).next x hist ()).eval H = (((P.next x hist ω).eval H).1, []) := by
      simp only [replay, Claims.OC.eval, hrep]
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · have hv' : ((vn (as ++ [(((replay ms).next x hist ()).eval H).1])).eval H).1 = .msg m := by
        rw [e1]
        exact hv
      rw [Claims.Recursion.stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      rw [Claims.Recursion.stepWith_msg H (replay ms) x () vn n hist as t' hv']
      simp only [e1]
      exact stepWith_replay_out H P x ω vn ms n _ _ _ _ (by simp [hl]) (by simp [hl'])
        fun j hj1 hj2 => h j (by simp only [List.length_append, List.length_singleton] at hj1; omega) hj2
    · simp only [stepWith, e1, hv]
    · simp only [stepWith, e1, hv]

/-- **Replaying a run's own messages gives its decision**, for every prover. -/
theorem run_replay_out {C : Type} [Inhabited C] {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (V : Claims.Verifier C)
    (x : Msg) (ω : Ω) (r : Fin (V.k x) → C) :
    (run H (replay ((run H P V x ω r).msgs.map Prod.fst)) V x () r).out = (run H P V x ω r).out :=
  stepWith_replay_out H P x ω _ _ (V.k x) [] [] _ _ rfl rfl fun _ _ _ => rfl

/-- An event of positive probability happens at some coins. -/
theorem exists_of_pr_pos {α : Type} [Fintype α] {p : α → Prop} (h : 0 < Claims.Pr p) : ∃ a, p a := by
  classical
  by_contra hn
  unfold Claims.Pr at h
  rw [Finset.filter_false_of_mem fun a _ ha => hn ⟨a, ha⟩] at h
  simp at h

/-- **Flock's completeness gives an accepted inner replay**: at a true statement in Flock's scope whose completeness
error is below 1, some coins accept the prover's run (`exists_of_pr_pos`), and replaying its messages at those coins is
accepted (`run_replay_out`), so `InnerAccSha` holds there (`innerAcc_of_run`, `innerAcc_sha512_iff`). -/
theorem innerAccSha_of_complete {Ω : Type} [Nonempty Ω] {rel : Flock.Readable.Parsed → Flock.Readable.Relation}
    {P : Claims.Flock.Wit → Prover Ω} {δc : Msg → ℝ}
    (hC : Claims.Complete FlockDeploy.sha512 (FlockDeploy.V false) P
      (Claims.Flock.R (FlockDeploy.deployPlain rel).parse) (Claims.Flock.Scope (FlockDeploy.deployPlain rel)) δc)
    (h64 : Claims.Out64 FlockDeploy.sha512) {x : Msg}
    {w : Claims.Flock.Wit} (hS : Claims.Flock.Scope (FlockDeploy.deployPlain rel) x)
    (hR : Claims.Flock.R (FlockDeploy.deployPlain rel).parse x w) (hδ : δc x < 1) :
    ∃ (c : ℕ → Coin) (ms : List Msg), InnerAccSha x c ms := by
  obtain ⟨ω⟩ := ‹Nonempty Ω›
  obtain ⟨r, hr⟩ : ∃ r : Fin ((FlockDeploy.V false).k x) → Coin,
      (run FlockDeploy.sha512 (P w) (FlockDeploy.V false) x ω r).out = .acc :=
    exists_of_pr_pos (lt_of_lt_of_le (by linarith) (hC h64 x w hS hR ω))
  refine ⟨Claims.coinSeq r, (run FlockDeploy.sha512 (P w) (FlockDeploy.V false) x ω r).msgs.map Prod.fst, ?_⟩
  rw [← innerAcc_sha512_iff]
  apply innerAcc_of_run
  have hr' : (fun i : Fin ((FlockDeploy.V false).k x) => Claims.coinSeq r i.val) = r :=
    funext fun i => by
      unfold Claims.coinSeq
      rw [dite_eq_left i.isLt]
  rw [hr', run_replay_out]
  exact hr

/-! ## What the checked reader reads of the inner statement -/

theorem innerOk_iff {inner : Msg} : innerOk inner = .ok () ↔
    ∃ s spec sts, FlockDeploy.setupOf false inner = some (s, spec, sts) ∧ s.read.check = .ok () ∧
      s.read.draw.isSome = true ∧ drawTiedB s spec sts = true := by
  unfold innerOk
  split
  · next s spec sts hs =>
    constructor
    · intro h
      obtain ⟨u, hu, h⟩ := FlockSoundness.Refine.ebind_ok h
      cases u
      refine ⟨s, spec, sts, hs, hu, ?_, ?_⟩
      · cases hd : s.read.draw with
        | none =>
          rw [hd] at h
          cases h
        | some _ => rfl
      · cases ht : drawTiedB s spec sts with
        | false =>
          cases hd : s.read.draw with
          | none =>
            rw [hd] at h
            cases h
          | some _ =>
            rw [hd, ht] at h
            cases h
        | true => rfl
    · rintro ⟨s', spec', sts', hs', hc, hd, ht⟩
      rw [hs] at hs'
      cases hs'
      show (s.read.check >>= fun _ => _) = _
      rw [hc]
      cases hdr : s.read.draw with
      | none =>
        rw [hdr] at hd
        cases hd
      | some _ =>
        rw [ht]
        rfl
  · next hs =>
    refine ⟨fun h => (by cases h), fun ⟨s, spec, sts, h, _⟩ => ?_⟩
    rw [hs] at h
    cases h

/-- **At every statement the reader parses, Flock's plain verifier sets the inner statement up, its statement check
passes and it carries a draw** (`readX_inner`'s `innerOk`). -/
theorem parse_innerOk {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) :
    ∃ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) ∧ s.read.check = .ok () ∧
      s.read.draw.isSome = true := by
  obtain ⟨r, hr, hx⟩ := readX_of_parse hst
  rw [hx]
  obtain ⟨s, spec, sts, h1, h2, h3, -⟩ := innerOk_iff.1 (readX_inner hr).2.2
  exact ⟨s, spec, sts, h1, h2, h3⟩

/-- **At every statement the reader parses, at one table the session's draw is the statement's** (`innerOk`'s
`drawTiedB`): the inner setup reads the session's `unitDraw` from the public file's header, and the reader refuses a
header whose draw is not the record's. -/
theorem parse_drawTiedOne {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) :
    ∀ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) → s.read.check = .ok () →
      sts.size = 1 → FlockDeploy.drawEq (drawOfU spec.unitDraw) s.read.draw = true := by
  intro s spec sts hs _ h1
  obtain ⟨r, hr, hx⟩ := readX_of_parse hst
  rw [hx] at hs
  obtain ⟨s', spec', sts', hs', -, -, ht⟩ := innerOk_iff.1 (readX_inner hr).2.2
  rw [hs] at hs'
  cases hs'
  exact drawTiedB_iff.1 ht h1

/-! ## Flock's admission of plain leaves (`Assumptions/PlainLeaves.lean`) -/

/-- **In Flock's scope, at one table, the session's draw is the statement's** (`innerOk`'s `drawTiedB`): the setup writes
the statement's draw into the public file's header, whose draw the session's `unitDraw` is
(`FlockComplete.JsonRT.unitDraw_one`). -/
theorem drawTied_of_scope {rel : Flock.Readable.Parsed → Flock.Readable.Relation} {inner : Msg} {s : FlockDeploy.Ins}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (hS : Claims.Flock.Scope (FlockDeploy.deployPlain rel) inner)
    (hs : FlockDeploy.setupOf false inner = some (s, spec, sts)) (hd : s.read.draw.isSome = true) :
    sts.size = 1 → FlockDeploy.drawEq (drawOfU spec.unitDraw) s.read.draw = true := by
  intro h1
  obtain ⟨d, hU, hdd, hn⟩ := FlockComplete.JsonRT.unitDraw_one hS hs h1 hd
  have hnn := (FlockDeploy.insOf_draw_norm (FlockDeploy.setupOf_some hs).1 hdd).1
  have hu : drawOfU (some d) = some d := by
    cases d with
    | null => exact absurd rfl hn
    | _ => exact congrArg some hnn
  rw [hU, hdd, hu]
  exact FlockDeploy.drawEq_refl _

/-- **Under the Prop, the inner verifier's acceptance is satisfiable at an inner statement the checked reader
admits**, wherever Flock's honest prover is complete (`innerAccSha_of_complete`, at `δc < 1` from the draw and the
sizes, `FlockComplete.δc_sha512_lt_one`); the reader's draw tie at one table is the setup's (`drawTied_of_scope`). -/
theorem admitted_acc {rel : Flock.Readable.Parsed → Flock.Readable.Relation} (hA : PlainLeavesAdmitted rel)
    (hC : Claims.Flock.Complete FlockDeploy.sha512 (FlockDeploy.deployPlain rel))
    (h64 : Claims.Out64 FlockDeploy.sha512) :
    ∃ (inner : Msg) (ri : FlockDeploy.Raw) (partition : Option Json) (dims : Flock.RecVStar.Inner),
      FlockDeploy.decode inner = some ri ∧ (ri.partition.mapM (FlockDeploy.jsonOf "partition")).toOption = some partition ∧
        Flock.RecVStar.innerOf ri.circuit ri.publicFile partition ri.program = .ok dims ∧ innerOk inner = .ok () ∧
        ∃ (c : ℕ → Coin) (ms : List Msg), InnerAccSha inner c ms := by
  obtain ⟨inner, ri, partition, dims, w, s, spec, sts, hri, hpa, hdims, hs, hck, hd, hl, hε, hreg, hrec, hS, hR⟩ :=
    hA
  exact ⟨inner, ri, partition, dims, hri, hpa, hdims,
    innerOk_iff.2 ⟨s, spec, sts, hs, hck, hd, drawTiedB_iff.2 (drawTied_of_scope hS hs hd)⟩,
    innerAccSha_of_complete hC h64 hS hR
      (FlockComplete.δc_sha512_lt_one hs hd (hl.trans (by norm_num)) hreg hrec hε)⟩

/-- **`admitted_acc` at `Readable.relation`**, where Flock's completeness is `FlockComplete.Replay.complete_plain`, under its
hypotheses. -/
theorem admitted_acc_plain (hA : PlainLeavesAdmitted Flock.Readable.relation)
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon) (h64 : Claims.Out64 FlockDeploy.sha512)
    (hfin : FlockComplete.Replay.LgFinal FlockDeploy.sha512 Flock.Readable.relation) :
    ∃ (inner : Msg) (ri : FlockDeploy.Raw) (partition : Option Json) (dims : Flock.RecVStar.Inner),
      FlockDeploy.decode inner = some ri ∧ (ri.partition.mapM (FlockDeploy.jsonOf "partition")).toOption = some partition ∧
        Flock.RecVStar.innerOf ri.circuit ri.publicFile partition ri.program = .ok dims ∧ innerOk inner = .ok () ∧
        ∃ (c : ℕ → Coin) (ms : List Msg), InnerAccSha inner c ms :=
  admitted_acc hA (FlockComplete.Replay.complete_plain FlockDeploy.sha512 S h64 hfin) h64

/-! ## The inner readiness the deployed stop reads -/

/-- **A statement in scope parses.** -/
theorem parse_of_scopeCheck {ivScope : Msg → Bool} {x : Msg} (hs : scopeCheck ivScope x = true) :
    ∃ st, parse x = some st := by
  unfold scopeCheck Flock.RecScope.scopeCheckOf at hs
  split at hs
  · next r hr => exact ⟨Flock.RecScope.toStmt r, by simp [parse, Flock.RecScope.parseOf, hr]⟩
  · cases hs

/-- **In scope, the inner readiness is three facts of the inner session**: the scope check already has Flock's plain
verifier set the inner statement up with a draw (`parse_innerOk`), so `FlockComplete.ready` at SHA-512, which the
deployed stop reads (`stopsRec`), asks only that the inner statement be under `2^64` bytes, that its tables' regions
fit, and that the honest record fit (`FlockComplete.RecordFitsAt`). -/
theorem ready_inScope_iff {ivScope : Msg → Bool} {x : Msg} (hs : scopeCheck ivScope x = true) :
    ∃ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) ∧ s.read.draw.isSome = true ∧
      (FlockComplete.ready FlockDeploy.sha512 (xIOf x) = true ↔
        (xIOf x).length < 2 ^ 64 ∧ (∀ st ∈ sts, FlockComplete.RegionsFit st) ∧
          FlockComplete.RecordFitsAt FlockDeploy.sha512 (xIOf x) spec sts) := by
  obtain ⟨st, hst⟩ := parse_of_scopeCheck hs
  obtain ⟨s, spec, sts, hset, -, hd⟩ := parse_innerOk hst
  refine ⟨s, spec, sts, hset, hd, ?_⟩
  rw [FlockComplete.ready_sha512, hset]
  simp only [hd, Bool.true_and, Bool.and_eq_true, decide_eq_true_eq, and_assoc]

/-- **At the inner statement Flock admits, the inner readiness holds**: under `PlainLeavesAdmitted`, some inner
statement in Flock's scope and true is one `FlockComplete.ready` at SHA-512 takes. -/
theorem ready_of_admitted {rel : Flock.Readable.Parsed → Flock.Readable.Relation} (hA : PlainLeavesAdmitted rel) :
    ∃ inner : Msg, FlockComplete.ready FlockDeploy.sha512 inner = true ∧
      Claims.Flock.Scope (FlockDeploy.deployPlain rel) inner ∧
        ∃ w, Claims.Flock.R (FlockDeploy.deployPlain rel).parse inner w := by
  obtain ⟨inner, -, -, -, w, s, spec, sts, -, -, -, hs, -, hd, hl, -, hreg, hrec, hS, hR⟩ := hA
  refine ⟨inner, ?_, hS, w, hR⟩
  rw [FlockComplete.ready_sha512, hs]
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨⟨hd, hl.trans (by norm_num)⟩, hreg⟩, hrec⟩

/-! ## `Decomp` at the checked reader -/

/-- **The checks after the verdict that read the run**: the last message's statement is `x`, the record's draw is the
statement's, the live session's custody, and the record's canonical text (S0) (`FlockDeploy.checks` without the scope
check and the draw law). -/
def checksRun (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (x : Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Msg × Coin)) (f : FlockDeploy.Final) : Flock.V Unit := do
  Flock.Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Flock.Record.check (FlockDeploy.drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw)
    "live: the record's draw is not the statement's"
  FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids
  Flock.RecordText.check f.record

/-- **The checks after the verdict are the run's, the statement check and a draw.** -/
theorem checks_iff {zk : Bool} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {x : Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.Final} :
    FlockDeploy.checks zk s spec x link c₁ mids f = .ok () ↔
      checksRun s spec x link c₁ mids f = .ok () ∧ s.read.check = .ok () ∧ s.read.draw.isSome = true := by
  constructor
  · intro h
    unfold FlockDeploy.checks at h
    obtain ⟨u1, h1, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨u2, h2, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨lv, hlv, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨u4, h4, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨u5, h5, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨u6, h6, h7⟩ := FlockSoundness.Refine.ebind_ok h
    cases u1; cases u2; cases u4; cases u5; cases u6
    refine ⟨?_, h5, ?_⟩
    · unfold checksRun
      rw [h1]
      show (Flock.Record.check _ _ >>= _) = _
      rw [h2]
      show (FlockDeploy.liveOf spec f.record >>= _) = _
      rw [hlv]
      show (FlockDeploy.custody spec lv link c₁ mids >>= _) = _
      rw [h4]
      exact h7
    · cases hd : s.read.draw with
      | none => simp [hd, Flock.ProvedScope.drawLaw] at h6
      | some _ => rfl
  · rintro ⟨hr, hc, hd⟩
    unfold checksRun at hr
    obtain ⟨u1, h1, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨u2, h2, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨lv, hlv, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨u4, h4, h7⟩ := FlockSoundness.Refine.ebind_ok hr
    cases u1; cases u2; cases u4
    unfold FlockDeploy.checks
    rw [h1]
    show (Flock.Record.check _ _ >>= _) = _
    rw [h2]
    show (FlockDeploy.liveOf spec f.record >>= _) = _
    rw [hlv]
    show (FlockDeploy.custody spec lv link c₁ mids >>= _) = _
    rw [h4]
    show (s.read.check >>= _) = _
    rw [hc]
    show ((if s.read.draw.isNone = true then Flock.ProvedScope.drawLaw zk s.read.tags false none else pure ()) >>= _) = _
    cases hdr : s.read.draw with
    | none =>
      rw [hdr] at hd
      cases hd
    | some _ => exact h7

/-- **The inner verifier's run conditions**: `InnerAccSha` with the run's checks (`checksRun`) in place of
`FlockDeploy.checks`, so without the statement check and the draw law. -/
def InnerRun (x : Msg) (c : ℕ → Coin) (ms : List Msg) : Prop :=
  ∃ (N : ℕ) (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt)
    (link : List ByteArray × List ByteArray) (F : FlockDeploy.Final),
    FlockDeploy.setupOf false x = some (s, spec, sts) ∧ N + 1 ≤ (FlockDeploy.V false).k x ∧
      (∀ j < N, ∃ t, ms.getD j [] = 0 :: t) ∧ FlockDeploy.linkOf (ms.getD 0 []) = some link ∧
      FlockDeploy.finalOf (ms.getD N []) = some F ∧
      FlockDeploy.verdictWith false Flock.Sha512.hash spec (FlockDeploy.tabsOf sts) F.record F.proofs = .ok () ∧
      checksRun s spec x link (c 0)
        (FlockComplete.midsOf ((List.range (N + 1)).map fun j => ms.getD j []) ((List.range (N + 1)).map c))
        F = .ok ()

/-- **Where Flock's statement check passes, the inner acceptance is the run's conditions and a draw.** -/
theorem innerAccSha_iff_run {x : Msg} {s₀ : FlockDeploy.Ins} {spec₀ : Flock.SessionSpec} {sts₀ : Array Flock.Stmt}
    (hs₀ : FlockDeploy.setupOf false x = some (s₀, spec₀, sts₀)) (hc₀ : s₀.read.check = .ok ()) (c : ℕ → Coin)
    (ms : List Msg) : InnerAccSha x c ms ↔ InnerRun x c ms ∧ s₀.read.draw.isSome = true := by
  constructor
  · rintro ⟨N, s, spec, sts, link, F, hs, hk, htag, hlink, hfin, hv, hc⟩
    obtain ⟨hr, -, hd⟩ := checks_iff.1 hc
    rw [hs₀] at hs
    cases hs
    exact ⟨⟨N, s₀, spec₀, sts₀, link, F, hs₀, hk, htag, hlink, hfin, hv, hr⟩, hd⟩
  · rintro ⟨⟨N, s, spec, sts, link, F, hs, hk, htag, hlink, hfin, hv, hr⟩, hd⟩
    rw [hs₀] at hs
    cases hs
    exact ⟨N, s₀, spec₀, sts₀, link, F, hs₀, hk, htag, hlink, hfin, hv, checks_iff.2 ⟨hr, hc₀, hd⟩⟩

/-- **The parts' cut of the inner run** (a code fact about the compiled statement at the shape): wherever every part's
inner check holds at a statement the reader parses, of the shape, the inner verifier's run conditions hold. -/
def RunCut (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ s < nS x, chk s x c ms) → InnerRun (xIOf x) c ms

/-- **The inner statement is drawn wherever the parts' inner checks all hold**: at a statement the reader parses, of
the shape, where some coins and inner messages pass every part's inner check, Flock's plain verifier reads a draw off
the inner statement. It holds at the checked reader (`innerDrawn_checked`). -/
def InnerDrawn (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ s < nS x, chk s x c ms) →
      ∃ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) ∧ s.read.draw.isSome = true

/-- **The inner statement is drawn at the checked reader**: `readX` refuses an inner statement without a draw
(`innerOk`, `parse_innerOk`), whatever the parts' checks. -/
theorem innerDrawn_checked (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) :
    InnerDrawn shape chk := fun _ _ hst _ _ _ _ => by
  obtain ⟨s, spec, sts, hs, -, hd⟩ := parse_innerOk hst
  exact ⟨s, spec, sts, hs, hd⟩

/-- **`Decomp` at the deployed recursion is the run's cut**, at every shape: the reader passes the inner statement's
statement check and its draw at every statement it parses (`parse_innerOk`, `innerDrawn_checked`), so of
`decomp_deployed_iff`'s cut there remain the run's conditions (`innerAccSha_iff_run`). -/
theorem decomp_checked_iff (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (ivScope : Msg → Bool) (b : Brings)
    (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) :
    Claims.Recursion.VStar.Decomp FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI).V shape chk ↔
      RunCut shape chk := by
  rw [decomp_deployed_iff]
  constructor
  · intro h x st hst hf c ms hc
    obtain ⟨s, spec, sts, hs, hck, -⟩ := parse_innerOk hst
    exact ((innerAccSha_iff_run hs hck c ms).1 (h x st hst hf c ms hc)).1
  · intro hr x st hst hf c ms hc
    obtain ⟨s, spec, sts, hs, hck, hd⟩ := parse_innerOk hst
    exact (innerAccSha_iff_run hs hck c ms).2 ⟨hr x st hst hf c ms hc, hd⟩

end RecDeploy
