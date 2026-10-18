import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Complete.Walk

/-!
# The checked reader: `Decomp` under Flock's admission of plain leaves

`readX` requires Flock's own setup and statement check on the inner statement (`innerOk`), besides the inner setup at
the recursion's inner tags (`RecVStar.innerOf`). Today the two never both pass: `innerOf` reads plain leaves off the
circuit file and `ProvedScope.check` passes only `hm96-sha512/v1` leaves (`check_refused_of_decode`), so the checked
reader parses nothing (`parse_none_today`). Flock's plain-leaves change (`cursor/flock-plain-leaves-sound-c7a9`) extends
its soundness headline to plain-leaves statements and has its verifier admit them outside `--zk`. This file states what
the recursion needs of that change as one Prop and proves what follows from it.

* **`PlainLeavesAdmitted rel`**: some inner statement passes every check the reader makes of it (it decodes, its
  partition reads, `RecVStar.innerOf` sets it up and `innerOk` passes), is in Flock's scope, where Flock's soundness
  headline holds (`Claims.Flock.Scope`), is true (`Claims.Flock.R`), and has Flock's completeness error below 1
  (`FlockComplete.δc`: a drawn session). Its first four conditions are those `check_refused_of_decode` refutes together
  today (`plainLeavesAdmitted_false`).
* **`Decomp` at the checked reader** (`decomp_checked_iff`): at every statement the reader parses, the inner statement
  passes Flock's statement check (`parse_innerOk`), so the cut `decomp_deployed_iff` asks of the parts is the inner
  run's conditions (`InnerRun`, `RunCut`: the link and last messages, the verdict, the record's statement and draw, the
  custody) and the inner statement's draw (`InnerDrawn`), the one condition of `FlockDeploy.checks` the reader leaves
  out (`checks_iff`). The recursion's inner session holds no draw (`rec_live.py` records `unit_draw: None`), so a reader
  that required it would refuse every honest statement.
* **The inner acceptance is satisfiable under the Prop** (`admitted_acc`): Flock's completeness at a true in-scope
  statement whose error is below 1 gives an accepted run of its prover (`exists_of_pr_pos`), whose own messages replayed
  are accepted (`run_replay_out`), so `InnerAccSha` holds at some coins and messages (`innerAccSha_of_complete`). At
  `Readable.relation` that completeness is `FlockComplete.Replay.complete_plain` (`admitted_acc_plain`).
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
    (hC : Claims.Flock.Complete FlockDeploy.sha512 (FlockDeploy.deployPlain rel) P δc) {x : Msg}
    {w : Claims.Flock.Wit} (hS : Claims.Flock.Scope (FlockDeploy.deployPlain rel) x)
    (hR : Claims.Flock.R (FlockDeploy.deployPlain rel).parse x w) (hδ : δc x < 1) :
    ∃ (c : ℕ → Coin) (ms : List Msg), InnerAccSha x c ms := by
  obtain ⟨ω⟩ := ‹Nonempty Ω›
  obtain ⟨r, hr⟩ : ∃ r : Fin ((FlockDeploy.V false).k x) → Coin,
      (run FlockDeploy.sha512 (P w) (FlockDeploy.V false) x ω r).out = .acc :=
    exists_of_pr_pos (lt_of_lt_of_le (by linarith) (hC x w hS hR ω))
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
    ∃ s spec sts, FlockDeploy.setupOf false inner = some (s, spec, sts) ∧ s.read.check = .ok () := by
  unfold innerOk
  split
  · next s spec sts hs =>
    exact ⟨fun h => ⟨s, spec, sts, hs, h⟩, fun ⟨s', spec', sts', hs', h⟩ => by
      rw [hs] at hs'
      cases hs'
      exact h⟩
  · next hs =>
    refine ⟨fun h => (by cases h), fun ⟨s, spec, sts, h, _⟩ => ?_⟩
    rw [hs] at h
    cases h

/-- **At every statement the reader parses, Flock's plain verifier sets the inner statement up and its statement check
passes** (`readX_inner`'s `innerOk`). -/
theorem parse_innerOk {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) :
    ∃ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) ∧ s.read.check = .ok () := by
  obtain ⟨r, hr, hx⟩ := readX_of_parse hst
  rw [hx]
  exact innerOk_iff.1 (readX_inner hr).2.2

/-- **Today the checked reader parses nothing**: it requires Flock's statement check on the inner statement
(`parse_innerOk`), which refuses the plain leaves the inner setup reads (`inner_check_refused`). -/
theorem parse_none_today (x : Msg) : parse x = none := by
  cases hp : parse x with
  | none => rfl
  | some st =>
    obtain ⟨s, spec, sts, hs, hc⟩ := parse_innerOk hp
    exact absurd hc (inner_check_refused hp (FlockDeploy.setupOf_some hs).1)

/-! ## Flock's admission of plain leaves -/

/-- **Flock admits a plain-leaves statement of the inner shape** (the contribution of
`cursor/flock-plain-leaves-sound-c7a9`): some inner statement passes every check the checked reader makes of it (it
decodes, its partition reads, the inner setup at the recursion's inner tags accepts it, `RecVStar.innerOf`, and Flock's
plain verifier sets it up and its statement check passes, `innerOk`), is in Flock's scope, where its soundness headline
holds (`Claims.Flock.Scope`), is true (`Claims.Flock.R`), and has Flock's completeness error below 1 there
(`FlockComplete.δc`, which asks for a drawn session). -/
def PlainLeavesAdmitted (rel : Flock.Readable.Parsed → Flock.Readable.Relation) : Prop :=
  ∃ (inner : Msg) (ri : FlockDeploy.Raw) (partition : Option Json) (dims : Flock.RecVStar.Inner)
    (w : Claims.Flock.Wit),
    FlockDeploy.decode inner = some ri ∧ (ri.partition.mapM (FlockDeploy.jsonOf "partition")).toOption = some partition ∧
      Flock.RecVStar.innerOf ri.circuit ri.publicFile partition ri.program = .ok dims ∧ innerOk inner = .ok () ∧
      Claims.Flock.Scope (FlockDeploy.deployPlain rel) inner ∧
      Claims.Flock.R (FlockDeploy.deployPlain rel).parse inner w ∧ FlockComplete.δc FlockDeploy.sha512 inner < 1

/-- **Today Flock admits no plain-leaves statement of the inner shape**: its statement check refuses every circuit file
the inner setup accepts (`check_refused_of_decode`). -/
theorem plainLeavesAdmitted_false (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    ¬ PlainLeavesAdmitted rel := by
  rintro ⟨inner, ri, partition, dims, w, hri, -, hdims, hok, -⟩
  obtain ⟨s, spec, sts, hs, hc⟩ := innerOk_iff.1 hok
  exact check_refused_of_decode hri hdims (FlockDeploy.setupOf_some hs).1 hc

/-- **Under the Prop, the inner verifier's acceptance is satisfiable at an inner statement the checked reader
admits**, wherever Flock's honest prover is complete (`innerAccSha_of_complete`). -/
theorem admitted_acc {rel : Flock.Readable.Parsed → Flock.Readable.Relation} (hA : PlainLeavesAdmitted rel)
    (hC : Claims.Flock.Complete FlockDeploy.sha512 (FlockDeploy.deployPlain rel) FlockComplete.honestP
      (FlockComplete.δc FlockDeploy.sha512)) :
    ∃ (inner : Msg) (ri : FlockDeploy.Raw) (partition : Option Json) (dims : Flock.RecVStar.Inner),
      FlockDeploy.decode inner = some ri ∧ (ri.partition.mapM (FlockDeploy.jsonOf "partition")).toOption = some partition ∧
        Flock.RecVStar.innerOf ri.circuit ri.publicFile partition ri.program = .ok dims ∧ innerOk inner = .ok () ∧
        ∃ (c : ℕ → Coin) (ms : List Msg), InnerAccSha inner c ms := by
  obtain ⟨inner, ri, partition, dims, w, hri, hpa, hdims, hok, hS, hR, hδ⟩ := hA
  exact ⟨inner, ri, partition, dims, hri, hpa, hdims, hok, innerAccSha_of_complete hC hS hR hδ⟩

/-- **`admitted_acc` at `Readable.relation`**, where Flock's completeness is `FlockComplete.Replay.complete_plain`, under its
hypotheses. -/
theorem admitted_acc_plain (hA : PlainLeavesAdmitted Flock.Readable.relation) (S : FlockComplete.JsonRT.JsonSpec)
    (h64 : FlockComplete.JsonRT.Out64 FlockDeploy.sha512) (hS : FlockComplete.JsonRT.StmtOk FlockDeploy.sha512 Flock.Readable.relation)
    (hPR : FlockComplete.JsonRT.ProverRuns FlockDeploy.sha512 Flock.Readable.relation)
    (hR : FlockComplete.JsonRT.RepShaped FlockDeploy.sha512 Flock.Readable.relation)
    (hF : FlockComplete.JsonRT.RecordFits FlockDeploy.sha512 Flock.Readable.relation)
    (hfin : FlockComplete.Replay.LgFinal FlockDeploy.sha512 Flock.Readable.relation) :
    ∃ (inner : Msg) (ri : FlockDeploy.Raw) (partition : Option Json) (dims : Flock.RecVStar.Inner),
      FlockDeploy.decode inner = some ri ∧ (ri.partition.mapM (FlockDeploy.jsonOf "partition")).toOption = some partition ∧
        Flock.RecVStar.innerOf ri.circuit ri.publicFile partition ri.program = .ok dims ∧ innerOk inner = .ok () ∧
        ∃ (c : ℕ → Coin) (ms : List Msg), InnerAccSha inner c ms :=
  admitted_acc hA (FlockComplete.Replay.complete_plain FlockDeploy.sha512 S h64 hS hPR hR hF hfin)

/-! ## `Decomp` at the checked reader -/

/-- **The checks after the verdict that read the run**: the last message's statement is `x`, the record's draw is the
statement's, and the live session's custody (`FlockDeploy.checks` before the scope check and the draw law). -/
def checksRun (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (x : Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Msg × Coin)) (f : FlockDeploy.Final) : Flock.V Unit := do
  Flock.Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Flock.Record.check (FlockDeploy.drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw)
    "live: the record's draw is not the statement's"
  FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids

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
    cases u1; cases u2; cases u4; cases u5
    refine ⟨?_, h5, ?_⟩
    · unfold checksRun
      rw [h1]
      show (Flock.Record.check _ _ >>= _) = _
      rw [h2]
      show (FlockDeploy.liveOf spec f.record >>= _) = _
      rw [hlv]
      exact h4
    · cases hd : s.read.draw with
      | none => simp [hd, Flock.ProvedScope.drawLaw] at h
      | some _ => rfl
  · rintro ⟨hr, hc, hd⟩
    unfold checksRun at hr
    obtain ⟨u1, h1, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨u2, h2, hr⟩ := FlockSoundness.Refine.ebind_ok hr
    obtain ⟨lv, hlv, h4⟩ := FlockSoundness.Refine.ebind_ok hr
    cases u1; cases u2
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
    show (if s.read.draw.isNone = true then Flock.ProvedScope.drawLaw zk none else pure ()) = _
    cases hdr : s.read.draw with
    | none =>
      rw [hdr] at hd
      cases hd
    | some _ => rfl

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

/-- **The inner statement is drawn wherever the parts' inner checks all hold** (a fact about the run): at a statement
the reader parses, of the shape, where some coins and inner messages pass every part's inner check, Flock's plain
verifier reads a draw off the inner statement. The reader does not require it: the recursion's inner session holds
none. -/
def InnerDrawn (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ s < nS x, chk s x c ms) →
      ∃ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) ∧ s.read.draw.isSome = true

/-- **`Decomp` at the deployed recursion is the run's cut and the draw**, at every shape: the reader passes the inner
statement's statement check at every statement it parses (`parse_innerOk`), so of `decomp_deployed_iff`'s cut there
remain the run's conditions (`innerAccSha_iff_run`) and the inner statement's draw. -/
theorem decomp_checked_iff (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (ivScope : Msg → Bool) (b : Brings)
    (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) :
    Claims.Recursion.VStar.Decomp FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI).V shape chk ↔
      RunCut shape chk ∧ InnerDrawn shape chk := by
  rw [decomp_deployed_iff]
  constructor
  · intro h
    refine ⟨fun x st hst hf c ms hc => ?_, fun x st hst hf c ms hc => ?_⟩
    · obtain ⟨s, spec, sts, hs, hck⟩ := parse_innerOk hst
      exact ((innerAccSha_iff_run hs hck c ms).1 (h x st hst hf c ms hc)).1
    · obtain ⟨s, spec, sts, hs, hck⟩ := parse_innerOk hst
      exact ⟨s, spec, sts, hs, ((innerAccSha_iff_run hs hck c ms).1 (h x st hst hf c ms hc)).2⟩
  · rintro ⟨hr, hd⟩ x st hst hf c ms hc
    obtain ⟨s, spec, sts, hs, hck⟩ := parse_innerOk hst
    obtain ⟨s', spec', sts', hs', hd'⟩ := hd x st hst hf c ms hc
    rw [hs] at hs'
    cases hs'
    exact (innerAccSha_iff_run hs hck c ms).2 ⟨hr x st hst hf c ms hc, hd'⟩

end RecDeploy
