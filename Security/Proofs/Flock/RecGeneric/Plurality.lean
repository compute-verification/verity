module

public import Proofs.Flock.RecGeneric.Comp
public import Proofs.Flock.RecGeneric.ServerRun
public import Proofs.Flock.Claims.Recursion.BoundMerged
public import Proofs.Flock.Claims.Sampling.Cost

/-!
# The plurality prover on the composition

`E*` (`Comp.E`) is the served `E^rep` of `Claims.servedERep` at the decoding `Comp.dec` (`Comp.E_eval`): its outer
stage is a rerun program whose inner view is `Comp.replayOC` through `ofOC`, and `replayOC` returns the replayed run's
view (`Comp.replayOC_eval`).

The inner stage's failure, where every outer extraction succeeds (`Claims.PrBadGood`), is bounded through one prover
against the inner protocol's own verifier:

* `P_O` (`Comp.innerPO`), at the inner statement, sends at inner round `j`, on the inner coins so far, the message that
  the inner views of `E_O`'s draws with those coins first open most often there (`Comp.opAtO`), over the draws where
  `E_O` gives witnesses in `R_O` (`Comp.GivesRO`). Its run at a draw is kept where `E_O` gives witnesses in `R_O`, the
  inner view accepts, and its openings agree with `P_O` at every inner round (`Comp.keepDO`); there it is the inner
  view opened (`Comp.runO`).
* Every outer extraction succeeding, `E_O` gives witnesses outside `R_O` at no rerun (`Comp.OutRO`, by `Replays`), so
  the served runs are those aborted there (`Comp.servedG`), whose openings `E^rep` at the inner protocol sees as `P_O`'s
  aborting runs off the draws that open otherwise (`Comp.BadO`, `Comp.servedO_seen`).
* The inner stage fails at most as `E^rep` at the inner protocol against `P_O`'s aborting runs, plus `N_I` times one
  draw's chance to be bad, plus a judged draw whose inner view rejects (`Comp.PrBadGood_le`).

Port of `Claims.Recursion.Bound.PrBadGood_le_merged` to any compiled verifier, outer extraction and server.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- The first `m` of the coins `r`, `default` past them. -/
def firstCoins {Coin : Type} [Inhabited Coin] {n : ℕ} (m : ℕ) (r : Fin n → Coin) : Fin m → Coin :=
  fun i => coinSeq r i.val

theorem coinSeq_firstCoins {Coin : Type} [Inhabited Coin] {n m : ℕ} (r : Fin n → Coin) {i : ℕ} (hi : i < m) :
    coinSeq (firstCoins m r) i = coinSeq r i := by
  simp only [coinSeq, hi, ↓reduceDIte, firstCoins]

theorem firstCoins_extend {Coin : Type} [Inhabited Coin] {m n : ℕ} (hn : m ≤ n) (ρ : Fin m → Coin)
    (r : Fin n → Coin) : firstCoins m (extend ρ r) = ρ := by
  funext i
  have hi : i.val < n := lt_of_lt_of_le i.isLt hn
  simp only [firstCoins, coinSeq, hi, ↓reduceDIte, extend, fork, i.isLt, ↓reduceIte, Fin.eta]

/-- A view opened where it accepts, as a trace, and aborted elsewhere: what `E^rep` at the inner protocol sees of
it. -/
def Server.openT (S : Server) (v : Seen) : Trace :=
  if v.out = .acc then ⟨(S.openView v).msgs, [], [], .acc⟩ else abortTrace

/-- **A view and its opening accept alike**, and an accepted view's opening is the view opened. -/
theorem Server.openT_rel (S : Server) (v : Seen) :
    (v.out = .acc ↔ (S.openT v).seen.out = .acc) ∧ (v.out = .acc → S.openView v = (S.openT v).seen) := by
  by_cases h : v.out = .acc
  · have e : S.openT v = ⟨(S.openView v).msgs, [], [], .acc⟩ := ite_eq_left h
    rw [e]
    exact ⟨⟨fun _ => rfl, fun _ => h⟩, fun _ => S.openView_eq_of_acc h⟩
  · have e : S.openT v = abortTrace := ite_eq_right h
    rw [e]
    exact ⟨⟨fun h' => absurd h' h, fun h' => by simp [abortTrace, Trace.seen] at h'⟩, fun h' => absurd h' h⟩

namespace Comp

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-! ## `E*` as the served `E^rep` -/

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`replayOC` returns the replayed run's view**: from `n` rounds left, the history `hist` and the trace `t`, with as
many prover messages sent as the history holds. -/
theorem stepWith_replayOC (H : Msg → Msg) (IV : Verifier Coin) (xI : Msg) (as : List Msg) (cs : ℕ → Coin) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (sent : List Msg) (t : Trace), hist.length = sent.length →
      ((replayOC IV xI as cs n sent t.msgs).eval H).1 =
        (stepWith H (replay as) xI () (fun as' => IV.next xI as' ((List.range as'.length).map cs)) n hist sent
          t).seen
  | 0, _, _, _, _ => rfl
  | n + 1, hist, sent, t, hl => by
    have hp : ((replay as).next xI hist ()).eval H = (as.getD sent.length [], []) := by
      show (OC.ret (as.getD hist.length [])).eval H = _
      rw [hl]
      rfl
    simp only [replayOC, stepWith, hp, OC.eval_bind, List.length_append, List.length_singleton]
    rcases hv : ((IV.next xI (sent ++ [as.getD sent.length []])
        ((List.range (sent.length + 1)).map cs)).eval H).1 with m | _ | _
    · refine Eq.trans ?_ (stepWith_replayOC H IV xI as cs n _ _ _ (by simp [hl]))
      rfl
    · rfl
    · rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem replayOC_eval (H : Msg → Msg) (IV : Verifier Coin) (xI : Msg) (as : List Msg)
    (ρ : Fin (IV.k xI) → Coin) :
    ((replayOC IV xI as (coinSeq ρ) (IV.k xI) [] []).eval H).1 = (run H (replay as) IV xI () ρ).seen :=
  stepWith_replayOC H IV xI as (coinSeq ρ) (IV.k xI) [] [] ⟨[], [], [], .rej⟩ rfl

/-- **The inner view's rerun program returns the inner view**, against any runs. -/
theorem decProg_eval (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) (x : Msg) (r : Fin (K.V.k x) → Coin)
    (o : Option K.WO) (run₂ : (Fin (K.V.k x) → Coin) → Trace) :
    ((K.decProg IV x r o).eval H run₂).1 = K.dec IV H x r o := by
  cases o with
  | none => rfl
  | some w =>
    simp only [Comp.decProg, Comp.dec]
    rw [Claims.Sampling.eval_ofOC]
    exact replayOC_eval H IV (K.xI x) (K.msgs x w) _

theorem stage_eval (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) (x : Msg)
    (run₂ : (Fin (K.V.k x) → Coin) → Trace) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) :
    ((K.stage IV x d).eval H run₂).1 = ((Claims.outerStage K.EO (K.dec IV H) x d).eval H run₂).1 := by
  rw [Comp.stage, (RP.eval_bind H run₂ _ _).1, decProg_eval, outerStage_eval]

/-- **`E*` is the served `E^rep` at the decoding `dec`**: one result at every run. -/
theorem E_eval (K : Comp Coin) (I : Proto Coin W) (S : Server) (H : Msg → Msg) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (r₀ : Fin (K.V.k x) → Coin) (c : (K.E I S).C x) :
    ((K.E I S).eval H P x ω r₀ c).1 =
      ((servedERep K.V K.EO (K.dec (S.IVStar I.V I.ans) H) (fun x => I.waits (K.xI x)) (fun x => I.k (K.xI x))
        (fun x => I.cap (K.xI x)) (fun v => I.opens (S.openView v)) (fun x => I.assemble (K.xI x))
        (fun x => I.check (K.xI x))).eval H P x ω r₀ c).1 := by
  have hs : servedRun H (K.stage (S.IVStar I.V I.ans) x) (run H P K.V x ω) =
      servedRun H (Claims.outerStage K.EO (K.dec (S.IVStar I.V I.ans) H) x) (run H P K.V x ω) := by
    funext d
    unfold servedRun
    rw [stage_eval]
  show ((serve (K.stage (S.IVStar I.V I.ans) x) (erep (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x))
      (fun v => I.opens (S.openView v)) (I.assemble (K.xI x)) (I.check (K.xI x)) (r₀, c.1) c.2)).eval H
        (run H P K.V x ω)).1 =
    (((erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) (fun v => I.opens (S.openView v))
      (I.assemble (K.xI x)) (I.check (K.xI x)) (r₀, c.1) c.2).serve
        (Claims.outerStage K.EO (K.dec (S.IVStar I.V I.ans) H) x)).eval H (run H P K.V x ω)).1
  rw [serve_eq, erep_eq, (RP.serve_eval H _ _ _).1, (RP.serve_eval H _ _ _).1, hs]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- The relation, inherited, at an extractor's output. -/
theorem R_ext (K : Comp Coin) (I : Proto Coin W) (x : Msg) (o : Option W) :
    (K.R I).ext x o ↔ K.parses x ∧ I.R.ext (K.xI x) o := by
  cases o with
  | none => simp [Rel.ext]
  | some w => exact Iff.rfl

/-! ## The plurality prover -/

section Plurality

variable (K : Comp Coin) (I : Proto Coin W) (S : Server) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg)

/-- `E_O` gives witnesses in `R_O` at the draw. -/
def GivesRO (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Prop :=
  ∃ w, K.wits H P x ω d = some w ∧ K.RO x w

/-- `E_O` gives witnesses outside `R_O` at the draw: where every outer extraction succeeds, no rerun shows it. -/
def OutRO (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Prop :=
  ∃ w, K.wits H P x ω d = some w ∧ ¬ K.RO x w

/-- The message the inner view at a draw opens at inner round `j`. -/
noncomputable def oMsg (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) (j : ℕ) : Msg :=
  ((S.openView (K.dec (S.IVStar I.V I.ans) H x d.1 (K.wits H P x ω d))).msgs.map Prod.fst).getD j []

/-- **`P_O`'s message at inner round `j`**, on the inner coins `l` so far: the message that the inner views of `E_O`'s
draws with those coins first open most often there, over the draws where `E_O` gives witnesses in `R_O`. -/
noncomputable def opAtO (ω : Ω) (j : ℕ) (l : List Coin) : Msg :=
  plur (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x => K.GivesRO H P x ω (fork j (preCoins l) u.1, u.2))
    (fun u => K.oMsg I S H P x ω (fork j (preCoins l) u.1, u.2) j)

/-- **`P_O`, the inner prover against the inner verifier**: it reads the inner coins so far off the inner verifier's
answers (`decC`) and sends `opAtO` there. -/
noncomputable def innerPO (decC : Msg → Coin) : Prover Ω :=
  ⟨fun _ hist ω => .ret (K.opAtO I S H P x ω hist.length (hist.map fun h => decC h.2))⟩

/-- The draw's openings agree with `P_O` at every inner round. -/
def AgreesO (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Prop :=
  ∀ j < I.V.k (K.xI x), K.oMsg I S H P x ω d j = K.opAtO I S H P x ω j ((List.range j).map (coinSeq d.1))

/-- **A draw `P_O`'s runs don't show**: `E_O` gives witnesses in `R_O` there, and its openings disagree with `P_O`. -/
def BadO (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Prop :=
  K.GivesRO H P x ω d ∧ ¬ K.AgreesO I S H P x ω d

open Classical in
/-- `P_O`'s run at a draw is kept where `E_O` gives witnesses in `R_O`, the inner view accepts, and its openings agree
with `P_O`. -/
noncomputable def keepDO (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Bool :=
  decide (K.GivesRO H P x ω d ∧ (K.dec (S.IVStar I.V I.ans) H x d.1 (K.wits H P x ω d)).out = .acc ∧
    K.AgreesO I S H P x ω d)

/-- **`P_O`'s run at a draw, as it aborts**: at the draw's inner coins, unless `keepDO` refuses it. -/
noncomputable def innerFO (decC : Msg → Coin) (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Trace :=
  if K.keepDO I S H P x ω d then
    run H (K.innerPO I S H P x decC) I.V (K.xI x) ω (firstCoins (I.V.k (K.xI x)) d.1)
  else abortTrace

/-- **`P_O`'s aborts**: a run at the inner coins `ρ` and the abort coins `a` is kept as `keepDO` keeps the draw they
extend to. -/
noncomputable def keepAO (ω : Ω) (ρ : Fin (I.V.k (K.xI x)) → Coin) (a : (Fin (K.V.k x) → Coin) × K.EO.C x) :
    Bool :=
  K.keepDO I S H P x ω (extendDraw (ρ, a))

open Classical in
/-- The served run at a draw, aborted where `E_O` gives witnesses outside `R_O`. -/
noncomputable def servedG (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Trace :=
  if K.OutRO H P x ω d then abortTrace
  else servedRun H (Claims.outerStage K.EO (K.dec (S.IVStar I.V I.ans) H) x) (run H P K.V x ω) d

/-- `servedG` opened (`Server.openT`): what `E^rep` at the inner protocol sees where `E*`'s inner stage sees the served
run. -/
noncomputable def servedO (ω : Ω) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : Trace :=
  S.openT (K.servedG I S H P x ω d).seen

/-- **An accepted run whose inner view rejects**, at the tape, the compiled verifier's coins and `E_O`'s: `E_O` gives
witnesses in `R_O` there, and the inner view read off them is not accepted. -/
def InnerRejectsO (t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x) : Prop :=
  ∃ w, K.Gives H P x t.1 t.2.1 t.2.2 w ∧ K.RO x w ∧ (K.view (S.IVStar I.V I.ans) H x t.2.1 w).out ≠ .acc

variable {K I S H P x}

omit [Fintype Coin] [Nonempty Coin] in
/-- At a draw where `E_O` gives witnesses whose inner view opens to the messages `ms`, the draw's openings are `ms`'. -/
theorem oMsg_eq {ω : Ω} {d : (Fin (K.V.k x) → Coin) × K.EO.C x} {w : K.WO} (hw : K.wits H P x ω d = some w)
    {ms : List (Msg × Out)} (hov : S.openView (K.view (S.IVStar I.V I.ans) H x d.1 w) = ⟨ms, .acc⟩) (j : ℕ) :
    K.oMsg I S H P x ω d j = (ms.map Prod.fst).getD j [] := by
  unfold Comp.oMsg
  rw [hw]
  show ((S.openView (K.view (S.IVStar I.V I.ans) H x d.1 w)).msgs.map Prod.fst).getD j [] = _
  rw [hov]

omit [Nonempty Coin] in
theorem keepDO_true {ω : Ω} {d : (Fin (K.V.k x) → Coin) × K.EO.C x} {w : K.WO} (hw : K.wits H P x ω d = some w)
    (hRO : K.RO x w) (hacc : (K.view (S.IVStar I.V I.ans) H x d.1 w).out = .acc) (ha : K.AgreesO I S H P x ω d) :
    K.keepDO I S H P x ω d = true := by
  simp only [Comp.keepDO, decide_eq_true_eq]
  refine ⟨⟨w, hw, hRO⟩, ?_, ha⟩
  rw [hw]
  exact hacc

omit [Nonempty Coin] in
/-- **On a kept draw, `P_O`'s run is the inner view opened**: it sends at each inner round what the inner view opens
there, which the inner verifier answers as `Open(V, S)` does (the round's coin's bytes, which `decC` reads back), so it
accepts, sees the inner view opened, and makes only calls of the inner view's run. -/
theorem runO (decC : Msg → Coin) (hdec : ∀ c, decC (I.ans c) = c) (ω : Ω)
    (d : (Fin (K.V.k x) → Coin) × K.EO.C x) {w : K.WO} (hw : K.wits H P x ω d = some w)
    (hacc : (K.view (S.IVStar I.V I.ans) H x d.1 w).out = .acc) (ha : K.AgreesO I S H P x ω d) :
    (run H (K.innerPO I S H P x decC) I.V (K.xI x) ω (firstCoins (I.V.k (K.xI x)) d.1)).seen =
        S.openView (K.view (S.IVStar I.V I.ans) H x d.1 w) ∧
      (run H (K.innerPO I S H P x decC) I.V (K.xI x) ω (firstCoins (I.V.k (K.xI x)) d.1)).out = .acc ∧
      ∀ e ∈ (run H (K.innerPO I S H P x decC) I.V (K.xI x) ω (firstCoins (I.V.k (K.xI x)) d.1)).vLog,
        e ∈ K.innerLog (S.IVStar I.V I.ans) H x d.1 w := by
  obtain ⟨hk, os, last, -, -, -, hag, hov⟩ := S.accFacts I.V I.ans H (K.xI x) (replay (K.msgs x w)) ()
    (firstCoins (I.V.k (K.xI x)) d.1) hacc
  rw [S.coinSeq_iv I.V I.ans (K.xI x) (firstCoins (I.V.k (K.xI x)) d.1)] at hag
  have hov' : S.openView (K.view (S.IVStar I.V I.ans) H x d.1 w) =
      ⟨(List.range (I.V.k (K.xI x))).map fun j => ((S.opened os last).getD j [],
        Server.ansAt I.V I.ans (K.xI x) (firstCoins (I.V.k (K.xI x)) d.1) j), .acc⟩ := hov
  have hfO : ∀ j < I.V.k (K.xI x), K.oMsg I S H P x ω d j = (S.opened os last).getD j [] := fun j hj => by
    rw [oMsg_eq hw hov', List.map_map, getD_range_map _ _ hj]
    rfl
  have hc : ∀ n ≤ I.V.k (K.xI x), (List.range n).map (coinSeq (firstCoins (I.V.k (K.xI x)) d.1)) =
      (List.range n).map (coinSeq d.1) := fun n hn =>
    List.map_congr_left fun i hi => by
      rw [List.mem_range] at hi
      exact coinSeq_firstCoins d.1 (show i < I.V.k (K.xI x) by omega)
  have hcongr : run H (K.innerPO I S H P x decC) I.V (K.xI x) ω (firstCoins (I.V.k (K.xI x)) d.1) =
      run H (replay (S.opened os last)) I.V (K.xI x) () (firstCoins (I.V.k (K.xI x)) d.1) := by
    unfold run
    refine stepWith_congr H _ _ (K.xI x) ω () _ (fun M hist as => hist.length + M = I.V.k (K.xI x) ∧
      as = (List.range hist.length).map (fun i => (S.opened os last).getD i []) ∧
      hist.map (fun h => decC h.2) = (List.range hist.length).map
        (coinSeq (firstCoins (I.V.k (K.xI x)) d.1))) ?_ ?_ _ _ _ _ ⟨by simp, by simp, by simp⟩
    · intro M hist as hG
      have hj : hist.length < I.V.k (K.xI x) := by omega
      show (OC.ret (K.opAtO I S H P x ω hist.length (hist.map fun h => decC h.2))).eval H =
        (OC.ret ((S.opened os last).getD hist.length [])).eval H
      rw [hG.2.2, hc hist.length hj.le, ← ha _ hj, hfO _ hj]
    · intro M hist as m hG hv
      have hj : hist.length < I.V.k (K.xI x) := by omega
      have hpa : ((K.innerPO I S H P x decC).next (K.xI x) hist ω).eval H =
          ((S.opened os last).getD hist.length [], []) := by
        have e : K.opAtO I S H P x ω hist.length (hist.map fun h => decC h.2) =
            (S.opened os last).getD hist.length [] := by
          rw [hG.2.2, hc hist.length hj.le, ← ha _ hj, hfO _ hj]
        show (OC.ret (K.opAtO I S H P x ω hist.length (hist.map fun h => decC h.2))).eval H = _
        simp only [e, OC.eval]
      have has : as ++ [(S.opened os last).getD hist.length []] =
          (List.range (hist.length + 1)).map fun i => (S.opened os last).getD i [] := by
        rw [hG.2.1, List.range_succ, List.map_append, List.map_singleton]
      rw [hpa] at hv ⊢
      dsimp only at hv ⊢
      simp only [has, List.length_map, List.length_range] at hv
      rw [(hag hist.length hj).1] at hv
      unfold Server.ansAt at hv
      split_ifs at hv with hL
      injection hv with hm
      subst hm
      refine ⟨by simp only [List.length_append, List.length_singleton]; omega, by rw [has]; simp, ?_⟩
      simp only [List.map_append, hG.2.2, List.length_append, List.length_singleton, List.range_succ,
        List.map_singleton, hdec]
      rfl
  obtain ⟨h1, h2, h3⟩ := run_replay_known H I.V (K.xI x) (S.opened os last) (firstCoins (I.V.k (K.xI x)) d.1)
    (Server.ansAt I.V I.ans (K.xI x) (firstCoins (I.V.k (K.xI x)) d.1)) hk
    (fun j hj => (hag j hj).1) (fun j hj => ⟨_, by unfold Server.ansAt; exact ite_eq_left hj⟩)
    (by simp only [Server.ansAt, show ¬ (I.V.k (K.xI x) - 1 + 1 < I.V.k (K.xI x)) by omega, ↓reduceIte])
  rw [hcongr, hov']
  refine ⟨?_, h2, fun e he => ?_⟩
  · simp only [Trace.seen, h1, h2]
  · obtain ⟨j, hj, hej⟩ := h3 e he
    exact (hag j hj).2 e hej

/-- **Off the bad draws, `P_O`'s aborting run shows what the gated served run opened shows**: a draw where `E_O` gives
up, gives witnesses outside `R_O`, or gives witnesses whose inner view rejects aborts at both, and a kept draw is the
inner view opened (`runO`). -/
theorem servedO_seen (decC : Msg → Coin) (hdec : ∀ c, decC (I.ans c) = c) (ω : Ω)
    (d : (Fin (K.V.k x) → Coin) × K.EO.C x) (hb : ¬ K.BadO I S H P x ω d) :
    (K.servedO I S H P x ω d).seen = (K.innerFO I S H P x decC ω d).seen := by
  unfold Comp.servedO Comp.servedG
  by_cases hout : K.OutRO H P x ω d
  · have hk : K.keepDO I S H P x ω d = false := by
      obtain ⟨w, hw, hn⟩ := hout
      simp only [Comp.keepDO, decide_eq_false_iff_not]
      rintro ⟨⟨w', hw', hR⟩, -⟩
      rw [hw] at hw'
      obtain rfl : w = w' := Option.some_inj.1 hw'
      exact hn hR
    have e1 : S.openT abortTrace.seen = abortTrace := ite_eq_right (by simp [abortTrace, Trace.seen])
    have e2 : K.innerFO I S H P x decC ω d = abortTrace := by
      unfold Comp.innerFO
      rw [hk]
      rfl
    rw [ite_eq_left hout, e1, e2]
  rw [ite_eq_right hout, servedRun_seen, outerStage_eval]
  change (S.openT (K.dec (S.IVStar I.V I.ans) H x d.1 (K.wits H P x ω d))).seen = _
  rcases ho : K.wits H P x ω d with _ | w
  · have hk : K.keepDO I S H P x ω d = false := by
      simp only [Comp.keepDO, decide_eq_false_iff_not]
      rintro ⟨⟨w', hw', -⟩, -⟩
      rw [ho] at hw'
      cases hw'
    have e1 : S.openT (K.dec (S.IVStar I.V I.ans) H x d.1 none) = abortTrace := ite_eq_right (by simp [Comp.dec])
    have e2 : K.innerFO I S H P x decC ω d = abortTrace := by
      unfold Comp.innerFO
      rw [hk]
      rfl
    rw [e1, e2]
  · have hRO : K.RO x w := by
      by_contra hn
      exact hout ⟨w, ho, hn⟩
    by_cases hacc : (K.view (S.IVStar I.V I.ans) H x d.1 w).out = .acc
    · have ha : K.AgreesO I S H P x ω d := by
        by_contra hn
        exact hb ⟨⟨w, ho, hRO⟩, hn⟩
      have hk := keepDO_true ho hRO hacc ha
      have e : K.innerFO I S H P x decC ω d =
          run H (K.innerPO I S H P x decC) I.V (K.xI x) ω (firstCoins (I.V.k (K.xI x)) d.1) := by
        unfold Comp.innerFO
        rw [ite_eq_left hk]
      rw [e, (runO decC hdec ω d ho hacc ha).1]
      exact ((S.openT_rel _).2 hacc).symm
    · have hk : K.keepDO I S H P x ω d = false := by
        simp only [Comp.keepDO, decide_eq_false_iff_not]
        rintro ⟨-, hv, -⟩
        rw [ho] at hv
        exact hacc hv
      have e1 : S.openT (K.dec (S.IVStar I.V I.ans) H x d.1 (some w)) = abortTrace := ite_eq_right hacc
      have e2 : K.innerFO I S H P x decC ω d = abortTrace := by
        unfold Comp.innerFO
        rw [hk]
        rfl
      rw [e1, e2]

omit [Nonempty Coin] in
theorem abortRun_eqO (decC : Msg → Coin) (hn : I.V.k (K.xI x) ≤ K.V.k x) (ω : Ω)
    (t : (Fin (I.V.k (K.xI x)) → Coin) × (Fin (K.V.k x) → Coin) × K.EO.C x) :
    abortRun H (K.innerPO I S H P x decC) I.V (K.xI x) (K.keepAO I S H P x) ω t =
      K.innerFO I S H P x decC ω (extendDraw t) := by
  have hc : firstCoins (I.V.k (K.xI x)) (extendDraw t).1 = t.1 := firstCoins_extend hn t.1 t.2.1
  unfold abortRun Comp.keepAO
  show (if K.keepDO I S H P x ω (extendDraw t) = true then
      run H (K.innerPO I S H P x decC) I.V (K.xI x) ω t.1 else abortTrace) =
    K.innerFO I S H P x decC ω (extendDraw t)
  unfold Comp.innerFO
  rw [hc]

/-- **The inner stage fails where `E^rep` at the inner protocol fails against `P_O`'s aborting runs, at a bad rerun, or
where the judged draw's inner view rejects**: where every outer extraction succeeds, `E_O` gives witnesses outside
`R_O` at no rerun (`Replays`), so `E*`'s inner stage sees the gated served runs (`servedG`), and at the inner protocol's
openings what `E^rep` there sees of them opened (`erep_rel₂`), which off the bad draws are `P_O`'s aborting runs
(`servedO_seen`); the judged draw is `P_O`'s accepted run where its inner view accepts, and its statement parses
(`Refuses`). A bad rerun costs `N_I` times one draw's (`Pr_exists_rerun_draws`), and `E^rep` at `P_O`'s aborting runs
over their own coins is `Inner.PrBadAbort` at the inner protocol (`unifMap_extendDraw`, `InnerRounds`). -/
theorem PrBadGood_le (decC : Msg → Coin) (hdec : ∀ c, decC (I.ans c) = c) (hRp : K.Replays H x)
    (hRef : K.Refuses H x) (hR : K.InnerRounds I.V x) [Fintype Ω] :
    PrBadGood H K.V (K.R I) K.RO K.EO (K.dec (S.IVStar I.V I.ans) H) (fun x => I.waits (K.xI x))
        (fun x => I.k (K.xI x)) (fun x => I.cap (K.xI x)) (fun v => I.opens (S.openView v))
        (fun x => I.assemble (K.xI x)) (fun x => I.check (K.xI x)) x P ≤
      I.toInner.PrBadAbort H (K.xI x) (K.innerPO I S H P x decC) (K.keepAO I S H P x) +
        K.NI I x * Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2) +
        Pr (K.InnerRejectsO I S H P x) := by
  have hn : I.V.k (K.xI x) ≤ K.V.k x := hR
  let Yb : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x ×
      (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x) → Prop :=
    fun s => (K.innerFO I S H P x decC s.1 (s.2.1, s.2.2.1)).out = .acc ∧
      ¬ I.R.ext (K.xI x) ((erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) I.opens
        (I.assemble (K.xI x)) (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2).eval H
          (K.innerFO I S H P x decC s.1)).1
  let Zb : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x ×
      (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x) → Prop :=
    fun s => ∃ d ∈ (erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) I.opens (I.assemble (K.xI x))
        (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H (K.servedO I S H P x s.1), K.BadO I S H P x s.1 d
  let Jb : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x ×
      (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x) → Prop :=
    fun s => K.InnerRejectsO I S H P x (s.1, s.2.1, s.2.2.1)
  have hYZ : PrBadGood H K.V (K.R I) K.RO K.EO (K.dec (S.IVStar I.V I.ans) H) (fun x => I.waits (K.xI x))
      (fun x => I.k (K.xI x)) (fun x => I.cap (K.xI x)) (fun v => I.opens (S.openView v))
      (fun x => I.assemble (K.xI x)) (fun x => I.check (K.xI x)) x P ≤ Pr Yb + Pr Zb + Pr Jb := by
    refine (Pr_mono fun s hs => ?_).trans ((Pr_or_le (fun s => Yb s ∨ Zb s) Jb).trans
      (add_le_add (Pr_or_le Yb Zb) le_rfl))
    by_cases hz : Zb s
    · exact Or.inl (Or.inr hz)
    by_cases hj : Jb s
    · exact Or.inr hj
    left; left
    have hnb : ∀ d ∈ (erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) I.opens (I.assemble (K.xI x))
        (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H (K.servedO I S H P x s.1),
        ¬ K.BadO I S H P x s.1 d :=
      fun d hd hb => hz ⟨d, hd, hb⟩
    have hG : ∀ d ∈ (erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x))
        (fun v => I.opens (S.openView v)) (I.assemble (K.xI x)) (I.check (K.xI x)) (s.2.1, s.2.2.1)
          s.2.2.2).rerunsAt H
          (servedRun H (Claims.outerStage K.EO (K.dec (S.IVStar I.V I.ans) H) x) (run H P K.V x s.1)),
        ¬ K.OutRO H P x s.1 d := by
      rintro d hd ⟨w, hw, hnR⟩
      have hw' : (K.EO.eval H P x s.1 d.1 d.2).1 = some w := hw
      exact hs.2.1 d hd ⟨hRp Ω P s.1 d.1 d.2 w hw', fun h => hnR (by rw [hw'] at h; exact h)⟩
    have heq0 := RP.eval_congr H
      (servedRun H (Claims.outerStage K.EO (K.dec (S.IVStar I.V I.ans) H) x) (run H P K.V x s.1))
      (K.servedG I S H P x s.1) (erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x))
        (fun v => I.opens (S.openView v)) (I.assemble (K.xI x)) (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2)
      fun d hd => by
        unfold Comp.servedG
        rw [ite_eq_right (hG d hd)]
    have heq1 : ((erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) (fun v => I.opens (S.openView v))
        (I.assemble (K.xI x)) (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2).eval H (K.servedG I S H P x s.1)).1 =
        ((erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) I.opens (I.assemble (K.xI x))
          (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2).eval H (K.servedO I S H P x s.1)).1 :=
      erep_rel₂ H _ (K.servedO I S H P x s.1) _ _ _ (fun v => I.opens (S.openView v)) I.opens _ _ _ _ fun d =>
        ⟨(S.openT_rel _).1, fun h => congrArg I.opens ((S.openT_rel _).2 h)⟩
    have heq2 := RP.eval_congr H (K.servedO I S H P x s.1) (K.innerFO I S H P x decC s.1)
      (erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) I.opens (I.assemble (K.xI x))
        (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2)
      fun d hd => servedO_seen decC hdec s.1 d (hnb d hd)
    have hd₀ : (s.2.1, s.2.2.1) ∈ (erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x))
        (fun v => I.opens (S.openView v)) (I.assemble (K.xI x)) (I.check (K.xI x)) (s.2.1, s.2.2.1)
          s.2.2.2).rerunsAt H
          (servedRun H (Claims.outerStage K.EO (K.dec (S.IVStar I.V I.ans) H) x) (run H P K.V x s.1)) :=
      List.mem_cons_self
    have hd₀' : (s.2.1, s.2.2.1) ∈ (erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) I.opens
        (I.assemble (K.xI x)) (I.check (K.xI x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
          (K.servedO I S H P x s.1) := List.mem_cons_self
    have hgood := hs.2.1 _ hd₀
    obtain ⟨w, hw, hRO⟩ : ∃ w, K.wits H P x s.1 (s.2.1, s.2.2.1) = some w ∧ K.RO x w := by
      rcases ho : K.wits H P x s.1 (s.2.1, s.2.2.1) with _ | w
      · refine absurd ⟨hs.1, fun h => ?_⟩ hgood
        rw [show (K.EO.eval H P x s.1 s.2.1 s.2.2.1).1 = none from ho] at h
        exact h
      · refine ⟨w, rfl, ?_⟩
        by_contra hnR
        refine hgood ⟨hs.1, fun h => hnR ?_⟩
        rw [show (K.EO.eval H P x s.1 s.2.1 s.2.2.1).1 = some w from ho] at h
        exact h
    have ha : K.AgreesO I S H P x s.1 (s.2.1, s.2.2.1) := by
      by_contra hna
      exact hnb _ hd₀' ⟨⟨w, hw, hRO⟩, hna⟩
    have hacc : (K.view (S.IVStar I.V I.ans) H x s.2.1 w).out = .acc := by
      by_contra hne
      exact hj ⟨w, ⟨hs.1, hw⟩, hRO, hne⟩
    have hk := keepDO_true hw hRO hacc ha
    refine ⟨?_, fun hI => hs.2.2 ?_⟩
    · unfold Comp.innerFO
      rw [ite_eq_left hk]
      exact (runO decC hdec s.1 (s.2.1, s.2.2.1) hw hacc ha).2.1
    · show (K.R I).ext x (((erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x))
          (fun v => I.opens (S.openView v)) (I.assemble (K.xI x)) (I.check (K.xI x)) (s.2.1, s.2.2.1)
            s.2.2.2).serve (Claims.outerStage K.EO (K.dec (S.IVStar I.V I.ans) H) x)).eval H
              (run H P K.V x s.1)).1
      rw [(RP.serve_eval H _ _ _).1, heq0, heq1, heq2]
      exact (R_ext K I x _).2 ⟨hRef Ω P s.1 s.2.1 hs.1, hI⟩
  have hY : Pr Yb = I.toInner.PrBadAbort H (K.xI x) (K.innerPO I S H P x decC) (K.keepAO I S H P x) := by
    have hΦ : UnifMap (fun s : Ω × ((Fin (I.V.k (K.xI x)) → Coin) × (Fin (K.V.k x) → Coin) × K.EO.C x) ×
        (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) →
          (Fin (I.V.k (K.xI x)) → Coin) × (Fin (K.V.k x) → Coin) × K.EO.C x) =>
        (s.1, (extendDraw s.2.1).1, (extendDraw s.2.1).2, fun i j => extendDraw (s.2.2 i j))) := by
      let e : Ω × ((Fin (K.V.k x) → Coin) × K.EO.C x) ×
          (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x) ≃
          Ω × (Fin (K.V.k x) → Coin) × K.EO.C x ×
          (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x) :=
        { toFun := fun s => (s.1, s.2.1.1, s.2.1.2, s.2.2)
          invFun := fun s => (s.1, (s.2.1, s.2.2.1), s.2.2.2)
          left_inv := fun _ => rfl
          right_inv := fun _ => rfl }
      exact (UnifMap.self.prod ((unifMap_extendDraw hn).prod
        (UnifMap.pi (UnifMap.pi (unifMap_extendDraw hn) _) _))).comp (UnifMap.equiv e)
    rw [← hΦ.Pr Yb]
    refine Pr_congr fun s => ?_
    show (K.innerFO I S H P x decC s.1 (extendDraw s.2.1)).out = .acc ∧
      ¬ I.R.ext (K.xI x) ((erepProg (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) I.opens
        (I.assemble (K.xI x)) (I.check (K.xI x)) (extendDraw s.2.1) (fun i j => extendDraw (s.2.2 i j))).eval H
          (K.innerFO I S H P x decC s.1)).1 ↔ _
    have hf : (fun d => K.innerFO I S H P x decC s.1 (extendDraw d)) =
        abortRun H (K.innerPO I S H P x decC) I.V (K.xI x) (K.keepAO I S H P x) s.1 :=
      funext fun d => (abortRun_eqO decC hn s.1 d).symm
    rw [erepProg_eval_comp H (K.innerFO I S H P x decC s.1) extendDraw, hf, ← abortRun_eqO decC hn]
    rfl
  have hZ : Pr Zb ≤ K.NI I x *
      Pr (fun t : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x => K.BadO I S H P x t.1 t.2) := by
    have hN : (K.NI I x : ℝ) = 1 + (I.waits (K.xI x) : ℝ) * (I.k (K.xI x) : ℝ) := by
      simp only [Comp.NI]; push_cast; ring
    rw [hN]
    let e : Ω × (Fin (K.V.k x) → Coin) × K.EO.C x ×
        (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x) ≃
        Ω × ((Fin (K.V.k x) → Coin) × K.EO.C x) ×
        (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x) :=
      { toFun := fun s => (s.1, (s.2.1, s.2.2.1), s.2.2.2)
        invFun := fun t => (t.1, t.2.1.1, t.2.1.2, t.2.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    have hpe := Pr_exists_rerun_draws H (K.servedO I S H P x) (I.waits (K.xI x)) (I.k (K.xI x))
      (I.cap (K.xI x)) I.opens (I.assemble (K.xI x)) (I.check (K.xI x)) (K.BadO I S H P x)
    rw [← Pr_equiv e] at hpe
    exact hpe
  have hJ : Pr Jb = Pr (K.InnerRejectsO I S H P x) :=
    (unifMap_drop (Ω := Ω) (R := Fin (K.V.k x) → Coin) (C := K.EO.C x)).Pr (K.InnerRejectsO I S H P x)
  linarith

end Plurality

end Comp

end Claims.RecGeneric
