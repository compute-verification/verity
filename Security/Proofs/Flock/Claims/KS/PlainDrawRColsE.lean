import Proofs.Flock.Claims.KS.PlainDrawRCols
import Proofs.Flock.Claims.Recursion.LastBindsDeployed
import Proofs.Flock.Claims.Recursion.RecRecord

/-!
# The deployed reader's columns as the bound record reads them

`RecDeploy.bdDeployed` takes the element-level columns `colsE` as a parameter, read off a view (`Claims.Seen`): the
messages and answers, with neither the statement nor the last coin. `FlockDeploy.colsRV rd` is `colsR`'s reader on a
view. The statement is the last message's (`rd`), the coins are read back from the answers (`RecDeploy.outCoin`,
`default` at the decision), and the run is `runOfR`'s at what `decR` decides on there (`runOfDecR`). `decR` reads the
coins only at `cs.head?` and `cs.tail.dropLast`, which the answers keep, and the statement only through its setup,
which an accepting decision's checks fix to the last message's (`checksAtJ_ok`). So:

* `colsRV_eq`: on a view whose messages are `ms` and whose answers read back to `cs` but for its last coin, where
  `nextR rd` accepts `ms` at `cs` after `V false`'s rounds, `colsRV rd` is `colsR rd x ms cs`.
* `bdR`: `bdDeployed` at `nextE := nextR rd` and `colsE := colsRV rd`.
* `deployCols_runR`: on a run `IVStar` of `bdR` accepts at SHA-512, at any prover, `nextR rd` accepts the opened
  messages at the run's coins, and the deployed columns of the run (`Bound.deploy`'s, of the run's view opened) are
  `colsR` there (`Bound.accFacts`).
* `plainVerdictAtJK_of_runIVR`: there `nextR` decides, and `plainVerdictAtJK id` holds at its live outcome wherever
  the deployed columns of the run look up no column off `C` at table 0 (`plainVerdictAtJK_of_nextR_cols`).
-/

namespace FlockDeploy

open Flock
open RecDeploy (outCoin)

/-- **`runOfR`'s run at what `decR` decides on**, table `j`'s. -/
noncomputable def runOfDecR (d : DecR) (j : ℕ) : Option Claims.Flock.ZkRun := do
  let st ← d.sts[j]?
  let tab := Setup.ofCircuit st
  let lv ← (liveOfJ d.spec (jAt sha512 d.spec d.sts d.c₁ d.mids d.f)).toOption
  let extras ← (tab.extra lv.points).toOption
  pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
         saltLen := saltLenOf false tab,
         reps := [(lv.streams.getD (2 * j) #[],
                   (filesAt (tabsOf d.sts) (pointsOf d.spec d.c₁) d.f.ps).getD (2 * j) ByteArray.empty),
                  (lv.streams.getD (2 * j + 1) #[],
                   (filesAt (tabsOf d.sts) (pointsOf d.spec d.c₁) d.f.ps).getD (2 * j + 1) ByteArray.empty)] }

/-- A run's opened level-0 columns, none without a run (`colsR`'s). -/
noncomputable def colsOf : Option Claims.Flock.ZkRun → List (ℕ × List FlockLevel3.GF128)
  | some r => Claims.Flock.plainRunCols r
  | none => []

/-- **The column reader at `nextR rd`, on a view**: at the statement of the view's last message (`rd`), its messages
and the coins its answers show (`outCoin`), the opened level-0 columns of table `j` of the run `runOfR` reads at what
`decR` decides on. -/
noncomputable def colsRV (rd : Claims.Msg → Option FinalP) (v : Claims.Seen) (j : ℕ) :
    List (ℕ × List FlockLevel3.GF128) :=
  match ((v.msgs.map Prod.fst).getLast?).bind rd with
  | some fl =>
    match decR rd fl.stmt (v.msgs.map Prod.fst) ((v.msgs.map Prod.snd).map outCoin) with
    | some d => colsOf (runOfDecR d j)
    | none => []
  | none => []

theorem colsR_colsOf (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (j : ℕ) : colsR rd x as cs j = colsOf (runOfR rd x as cs j) := rfl

theorem runOfR_of_actR {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (j : ℕ) : runOfR rd x as cs j = runOfDecR ⟨s, spec, sts, link, c₁, mids, f⟩ j := by
  unfold runOfR
  rw [h]
  rfl

/-- A decision's statement is its last message's (`rd`). -/
theorem decR_rd {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {d : DecR} (h : decR rd x as cs = some d) : ∃ fl, (as.getLast?).bind rd = some fl ∧ d.f.stmt = fl.stmt := by
  simp only [decR, Option.bind_eq_bind, Option.pure_def] at h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨a, ha, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨fl, hfl, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨hfl, hst⟩ := Option.filter_eq_some_iff.1 hfl
  refine ⟨fl, by rw [ha]; exact hfl, ?_⟩
  obtain rfl := Option.some.inj h
  exact (beq_iff_eq.1 hst).symm

/-- `decR` reads the coins only at the first and between the first and the last. -/
theorem decR_congr_coins (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg)
    {cs cs' : List Coin} (h1 : cs.head? = cs'.head?) (h2 : cs.tail.dropLast = cs'.tail.dropLast) :
    decR rd x as cs = decR rd x as cs' := by
  unfold decR
  rw [h1, h2]

/-- **`decR` doesn't read the last coin**, after two rounds. -/
theorem decR_dropLast (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) {cs : List Coin}
    (h : 2 ≤ cs.length) (c : Coin) : decR rd x as (cs.dropLast ++ [c]) = decR rd x as cs := by
  rcases cs with _ | ⟨a, _ | ⟨b, t⟩⟩
  · simp at h
  · simp at h
  · exact decR_congr_coins rd x as (by simp) (by simp)

end FlockDeploy

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody Claims.Bridge.PlainDraw
open RecDeploy (outCoin)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The reader on a view is `colsR` at the view's messages and the run's coins**: where `nextR rd` accepts the
messages `ms` at the coins `cs` after `V false`'s rounds, at SHA-512, a view whose messages are `ms` and whose answers
show `cs` but for the last coin has `colsR rd x ms cs` for its `colsRV rd`. The decision's checks fix its statement to
`x` (`checksAtJ_ok`), and `decR` doesn't read the last coin (`decR_dropLast`). -/
theorem colsRV_eq {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
    {v : Claims.Seen} (hm : v.msgs.map Prod.fst = ms)
    (hcs : (v.msgs.map Prod.snd).map outCoin = cs.dropLast ++ [default]) (hk : cs.length = (FlockDeploy.V false).k x)
    (hacc : ((nextR rd x ms cs).eval sha512).1 = .acc) : colsRV rd v = colsR rd x ms cs := by
  cases hact : actR rd x ms cs with
  | rej => revert hacc; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert hacc; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    have hk' : (FlockDeploy.V false).k x = 2 + FlockDeploy.rounds false spec sts := by
      simp only [FlockDeploy.V, actR_setupOf hact]
    have h2 : 2 ≤ cs.length := by omega
    obtain ⟨fl, hfl, hfs⟩ := decR_rd (actR_decide hact)
    have hx : fl.stmt = x := hfs.symm.trans (checksAtJ_ok (decideR_sha512 hact hacc).2.1).1
    funext j
    unfold colsRV
    rw [colsR_colsOf, runOfR_of_actR hact, hm, hcs, hfl]
    simp only [hx, decR_dropLast rd x ms h2, actR_decide hact]

/-- **The deployment at the deployed reader**: `bdDeployed` with the element-level steps `nextR rd` and the columns
`colsRV rd`. -/
noncomputable abbrev bdR (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (wE : Claims.Msg → ℝ) (join : List Claims.Msg → Claims.Msg)
    (rowIn : Claims.Msg → Claims.Msg) (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) ×
      Claims.Msg)) (w : Claims.Msg → ℝ) : Claims.Recursion.Bound Coin (List 𝔽) Claims.Flock.PlainTrace :=
  RecDeploy.bdDeployed rel (nextR rd) (colsRV rd) wE join rowIn final w

/-- **An accepted run of the deployment reads `colsR` at its opened messages and coins**: on a run `IVStar` of `bdR`
accepts at SHA-512, at any prover, `final` reads the run's last message as `(os, last)`, `nextR rd` accepts the opened
messages at the run's coins, and the deployed columns of the run, `colsRV rd` at its view opened, are `colsR rd` there
(`Bound.accFacts`, `colsRV_eq`). -/
theorem deployCols_runR (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (wE : Claims.Msg → ℝ) (join : List Claims.Msg → Claims.Msg)
    (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (r : Fin ((bdR rel rd wE join rowIn final w).IVStar.k x) → Coin)
    (hacc : (Claims.run sha512 P (bdR rel rd wE join rowIn final w).IVStar x ω r).out = .acc) :
    ∃ os last, (bdR rel rd wE join rowIn final w).final
        (((Claims.run sha512 P (bdR rel rd wE join rowIn final w).IVStar x ω r).msgs.map Prod.fst).getD
          ((FlockDeploy.V false).k x - 1) []) = some (os, last) ∧
      ((nextR rd x ((bdR rel rd wE join rowIn final w).opened os last)
        ((List.range ((FlockDeploy.V false).k x)).map (Claims.coinSeq r))).eval sha512).1 = .acc ∧
      (bdR rel rd wE join rowIn final w).deploy.cols
          (Claims.run sha512 P (bdR rel rd wE join rowIn final w).IVStar x ω r).seen =
        colsR rd x ((bdR rel rd wE join rowIn final w).opened os last)
          ((List.range ((FlockDeploy.V false).k x)).map (Claims.coinSeq r)) := by
  have hDk : (bdR rel rd wE join rowIn final w).DE.V.k x = (FlockDeploy.V false).k x := rfl
  obtain ⟨hk0, os, last, hf, hlen, -, hstep, hview⟩ :=
    (bdR rel rd wE join rowIn final w).accFacts sha512 x P ω r hacc
  rw [hDk] at hk0 hf hlen hstep hview
  have hol : ((bdR rel rd wE join rowIn final w).opened os last).length = (FlockDeploy.V false).k x := by
    simp only [Claims.Recursion.Bound.opened, List.length_append, List.length_map, List.length_singleton]
    exact hlen
  have hms : (List.range ((FlockDeploy.V false).k x)).map
      (fun i => ((bdR rel rd wE join rowIn final w).opened os last).getD i []) =
      (bdR rel rd wE join rowIn final w).opened os last := by
    rw [← Claims.Recursion.take_eq_range_getD _ hol.ge, List.take_of_length_le hol.le]
  have hans : (bdR rel rd wE join rowIn final w).ansAt x r ((FlockDeploy.V false).k x - 1) = .acc := by
    unfold Claims.Recursion.Bound.ansAt
    rw [hDk, ite_eq_right (show ¬ ((FlockDeploy.V false).k x - 1 + 1 < (FlockDeploy.V false).k x) by omega)]
  have hacc' := (hstep ((FlockDeploy.V false).k x - 1) (by omega)).1
  rw [Nat.sub_add_cancel (by omega : 1 ≤ (FlockDeploy.V false).k x), hms, hans] at hacc'
  refine ⟨os, last, hf, hacc', ?_⟩
  show colsRV rd ((bdR rel rd wE join rowIn final w).openView _) = _
  rw [hview]
  refine colsRV_eq ?_ ?_ (by simp) hacc'
  · simp only [List.map_map]
    exact hms
  · have hacc0 : outCoin .acc = default := rfl
    obtain ⟨n, hn⟩ : ∃ n, (FlockDeploy.V false).k x = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hk0).symm⟩
    have hr : List.range ((FlockDeploy.V false).k x) = List.range n ++ [n] := by rw [hn, List.range_succ]
    rw [hr]
    simp only [List.map_append, List.map_map, List.map_singleton, List.dropLast_concat]
    congr 1
    · refine List.map_congr_left fun i hi => ?_
      rw [List.mem_range] at hi
      have hc : outCoin (.msg (Claims.coinSeq r i).bytes) = Claims.coinSeq r i := RecDeploy.coinOf_bytes _
      simp only [Function.comp_apply, Claims.Recursion.Bound.ansAt, hDk,
        show i + 1 < (FlockDeploy.V false).k x by omega, ↓reduceIte, RecDeploy.bdDeployed_ans, hc]
    · simp only [Claims.Recursion.Bound.ansAt, hDk, show ¬ (n + 1 < (FlockDeploy.V false).k x) by omega, ↓reduceIte,
        hacc0]

/-- **An accepted run of the deployment gives the verdict at the draw, at its deployed columns**: on a run `IVStar` of
`bdR` accepts at SHA-512, at any prover, `nextR rd` decides on the opened messages at the run's coins, and wherever the
decided statement carries the draw `d` and the deployed columns of the run look up no column off `C` at table 0,
`plainVerdictAtJK id` holds at the decision's live outcome (`deployCols_runR`, `plainVerdictAtJK_of_nextR_cols`). -/
theorem plainVerdictAtJK_of_runIVR (mPts : ℕ) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FinalP) (wE : Claims.Msg → ℝ) (join : List Claims.Msg → Claims.Msg)
    (rowIn : Claims.Msg → Claims.Msg)
    (final : Claims.Msg → Option (List (List (Claims.Msg × RecDeploy.Salt)) × Claims.Msg)) (w : Claims.Msg → ℝ)
    {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (r : Fin ((bdR rel rd wE join rowIn final w).IVStar.k x) → Coin)
    (hacc : (Claims.run sha512 P (bdR rel rd wE join rowIn final w).IVStar x ω r).out = .acc) :
    ∃ os last s spec sts link c₁ mids f,
      actR rd x ((bdR rel rd wE join rowIn final w).opened os last)
          ((List.range ((FlockDeploy.V false).k x)).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ (d : Json), s.read.draw = some d → ∀ {N : ℕ} (C : Fin N → List 𝔽),
        (¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol ((bdR rel rd wE join rowIn final w).deploy.cols
          (Claims.run sha512 P (bdR rel rd wE join rowIn final w).IVStar x ω r).seen 0) p.val = some u ∧ u ≠ C p) →
        plainVerdictAtJK id s.read.I d C link.1.toArray
          (viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
            (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)) := by
  obtain ⟨os, last, -, ha, hcols⟩ := deployCols_runR rel rd wE join rowIn final w P x ω r hacc
  rw [hcols]
  generalize (List.range ((FlockDeploy.V false).k x)).map (Claims.coinSeq r) = cs at ha ⊢
  cases hact : actR rd x ((bdR rel rd wE join rowIn final w).opened os last) cs with
  | rej => revert ha; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert ha; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    exact ⟨os, last, s, spec, sts, link, c₁, mids, f, hact, fun d hd _ C hno =>
      plainVerdictAtJK_of_nextR_cols mPts C hact ha hd hno⟩

end FlockDeploy.KS
