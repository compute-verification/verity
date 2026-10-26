import Proofs.Flock.Claims.Recursion.BoundBridge
import Proofs.Flock.Claims.Recursion.BoundForks

/-!
# The recursion's inner stage from (P1) at `DE`, through one plurality (route 2B)

`KS_of_flockKS_binds` reads `Inner.KSAbort` at the bound record, which the bridge (`Bound.bridge_of_dec`) gives from
(P1) at `DE` at the cost of `WeightCovers` (P3). That cost is a second plurality: `E*`'s inner prover `P_I` against
`IVStar` sends the plurality of the inner views' messages, and the bridge's prover against `DE` the plurality of what
`P_I`'s runs open. This file takes one plurality, straight from `E_O`'s draws to `DE`:

* `P_O` (`Bound.innerPO`), against `DE` at the inner statement, sends at inner round `j`, on the inner coins so far, the
  element-level message that the inner views of `E_O`'s draws with those coins first open most often there
  (`Bound.opAtO`, over the draws where `E_O` gives witnesses). Its run at a draw is kept where `E_O` gives witnesses,
  the inner view accepts, and its openings agree with `P_O` at every inner round (`Bound.keepDO`); there it is the
  inner view opened (`Bound.runO`), so `E^rep` at `DE` against `P_O`'s aborting runs sees what `E*`'s inner stage sees
  of the served runs opened (`Bound.servedO_seen`, `erep_rel₂`), off the draws that open otherwise (`Bound.BadO`).
* The inner stage fails at most as `E^rep` at `DE` against `P_O`'s aborting runs, plus `N_I` times one draw's chance to
  be bad, plus a rejecting judged draw (`Bound.PrBadGood_le_merged`); `B` at `P_O`'s aborting runs finds only what
  `B_I` finds (`Bound.PrBAbort_le_merged`).
* Two draws forked at inner round `j`, with witnesses and different openings at `j`, are a collision `Be`'s fork finds
  (`Bound.fork_collO`): a rejecting view by `InnerAcceptsColl`, inner views that differ at message `j` by `ForksBind`,
  equal commit strings opened two ways by `Binds`, and at the last round `final` reads one message. So the bad draws
  cost `N_I·√(k_I·k(x)·Pr[Be])` with the rejecting ones (`Bound.bad_rej_le_merged`), `InnerKS`'s own term.

`Bound.innerKS_of_plainKSAbort_binds` is `InnerKS` at the bound record from `Inner.KSAbort` at `DE` with no bridge and
no `WeightCovers`: `γ` at the bound record need only be at least `DE`'s (`Bd.DE.weight ≤ Bd.weight`).
`Bound.KS_of_plainKSAbort_binds` is the recursion's KS from it.
-/

namespace Claims.Recursion

open Claims

section OpenT

variable {Coin Col T : Type}

/-- A view opened where it accepts, as a trace, and aborted elsewhere: what `E^rep` at `DE` sees of it. -/
def Bound.openT (Bd : Bound Coin Col T) (v : Seen) : Trace :=
  if v.out = .acc then ⟨(Bd.openView v).msgs, [], [], .acc⟩ else abortTrace

/-- **A view and its opening accept alike**, and an accepted view's opening is the view opened. -/
theorem Bound.openT_rel (Bd : Bound Coin Col T) (v : Seen) :
    (v.out = .acc ↔ (Bd.openT v).seen.out = .acc) ∧ (v.out = .acc → Bd.openView v = (Bd.openT v).seen) := by
  by_cases h : v.out = .acc
  · have e : Bd.openT v = ⟨(Bd.openView v).msgs, [], [], .acc⟩ := ite_eq_left h
    rw [e]
    exact ⟨⟨fun _ => rfl, fun _ => h⟩, fun _ => Bd.openView_eq_of_acc h⟩
  · have e : Bd.openT v = abortTrace := ite_eq_right h
    rw [e]
    exact ⟨⟨fun h' => absurd h' h, fun h' => by simp [abortTrace, Trace.seen] at h'⟩, fun h' => absurd h' h⟩

end OpenT

section Merged

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (H : Msg → Msg) (D : Deploy Coin)
  (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} (P : Prover Ω) (x : Msg)

/-- The element-level message the inner view at a draw opens at round `j`. -/
noncomputable def Bound.oMsg (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (j : ℕ) : Msg :=
  ((Bd.openView (dec H D x d.1 (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d))).msgs.map Prod.fst).getD j []

/-- **`P_O`'s message at inner round `j`**, on the inner coins `l` so far: the element-level message that the inner
views of `E_O`'s draws with those coins first open most often there, over the draws where `E_O` gives witnesses. -/
noncomputable def Bound.opAtO (ω : Ω) (j : ℕ) (l : List Coin) : Msg :=
  plur (fun u : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
      (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω (fork j (preCoins l) u.1, u.2)).isSome = true)
    (fun u => Bd.oMsg decI H D DO decO P x ω (fork j (preCoins l) u.1, u.2) j)

/-- **`P_O`, the inner prover against `DE`**: it reads the inner coins so far off `DE`'s answers (`decC`) and sends
`opAtO` there. -/
noncomputable def Bound.innerPO (decC : Msg → Msg → Coin) : Prover Ω :=
  ⟨fun _ hist ω => .ret (Bd.opAtO decI H D DO decO P x ω hist.length (hist.map fun h => decC (xI D x) h.2))⟩

/-- The draw's openings agree with `P_O` at every inner round. -/
def Bound.AgreesO (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) : Prop :=
  ∀ j < Bd.DE.V.k (xI D x),
    Bd.oMsg decI H D DO decO P x ω d j = Bd.opAtO decI H D DO decO P x ω j ((List.range j).map (coinSeq d.1))

/-- **A draw `P_O`'s runs don't show**: `E_O` gives witnesses there, and its openings disagree with `P_O`. -/
def Bound.BadO (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) : Prop :=
  (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true ∧ ¬ Bd.AgreesO decI H D DO decO P x ω d

open Classical in
/-- `P_O`'s run at a draw is kept where `E_O` gives witnesses, the inner view accepts, and its openings agree with
`P_O`. -/
noncomputable def Bound.keepDO (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) : Bool :=
  decide ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true ∧
    (dec H D x d.1 (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d)).out = .acc ∧
    Bd.AgreesO decI H D DO decO P x ω d)

/-- **`P_O`'s run at a draw, as it aborts**: at the draw's inner coins, unless `keepDO` refuses it. -/
noncomputable def Bound.innerFO (decC : Msg → Msg → Coin) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) : Trace :=
  if Bd.keepDO decI H D DO decO P x ω d then
    run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) ω
      (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)
  else abortTrace

/-- **`P_O`'s aborts**: a run at the inner coins `ρ` and the abort coins `a` is kept as `keepDO` keeps the draw they
extend to. -/
noncomputable def Bound.keepAO (ω : Ω) (ρ : Fin (Bd.DE.V.k (xI D x)) → Coin)
    (a : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) : Bool :=
  Bd.keepDO decI H D DO decO P x ω (extendDraw (ρ, a))

/-- The served run at a draw, opened (`openT`): what `E^rep` at `DE` sees where `E*`'s inner stage sees the served
run. -/
noncomputable def Bound.servedO (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) : Trace :=
  Bd.openT (servedRun H (Claims.outerStage (EO D (Inner.ofFlock Bd.deploy decI) DO decO) (dec H D) x)
    (run H P D.V x ω) d).seen

variable {Bd decI H D DO decO P x}

omit [DecidableEq Col] in
/-- At a draw where `E_O` gives witnesses whose inner view is `IVStar`'s run and accepts, the draw's openings are the
messages `final` opens. -/
theorem Bound.oMsg_eq {ω : Ω} {d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x}
    {ws : List Flock.Wit} (hws : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d = some ws)
    {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last : Msg}
    (hov : Bd.openView (D.innerView H x d.1 ws) =
      ⟨(List.range (Bd.DE.V.k (xI D x))).map fun j =>
        ((Bd.opened os last).getD j [], Bd.ansAt (xI D x) (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1) j), .acc⟩)
    {j : ℕ} (hj : j < Bd.DE.V.k (xI D x)) :
    Bd.oMsg decI H D DO decO P x ω d j = (Bd.opened os last).getD j [] := by
  unfold Bound.oMsg
  rw [hws]
  show ((Bd.openView (D.innerView H x d.1 ws)).msgs.map Prod.fst).getD j [] = _
  rw [hov, List.map_map, getD_range_map _ _ hj]
  rfl

omit [DecidableEq Col] in
theorem Bound.keepDO_true {ω : Ω}
    {d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x} {ws : List Flock.Wit}
    (hws : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d = some ws)
    (hacc : (D.innerView H x d.1 ws).out = .acc) (ha : Bd.AgreesO decI H D DO decO P x ω d) :
    Bd.keepDO decI H D DO decO P x ω d = true := by
  simp only [Bound.keepDO, decide_eq_true_eq]
  refine ⟨by rw [hws]; rfl, ?_, ha⟩
  rw [hws]
  exact hacc

/-- **On a kept draw, `P_O`'s run is the inner view opened**: it sends at each inner round what the inner view opens
there, which `DE` answers as `IVStar` does (the round's coin's bytes, which `decC` reads back), so it accepts, sees the
inner view opened, and makes only calls of the inner view's `IVStar` run. -/
theorem Bound.runO (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) {ws : List Flock.Wit}
    (hws : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d = some ws)
    (hacc : (D.innerView H x d.1 ws).out = .acc) (ha : Bd.AgreesO decI H D DO decO P x ω d) :
    (run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) ω
        (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)).seen = Bd.openView (D.innerView H x d.1 ws) ∧
      (run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) ω
        (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)).out = .acc ∧
      ∀ e ∈ (run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) ω
        (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)).vLog,
        e ∈ (innerRun H D (Inner.ofFlock Bd.deploy decI) x d.1 ws).vLog := by
  have hv := hV decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws)
  have haccI : (run H (replay ((D.innerView H x d.1 ws).msgs.map Prod.fst)) Bd.IVStar (xI D x) ()
      (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)).out = .acc := by
    have h := hacc
    rw [hv] at h
    exact h
  obtain ⟨hk, os, last, -, -, -, hag, hov⟩ := Bd.accFacts H (xI D x)
    (replay ((D.innerView H x d.1 ws).msgs.map Prod.fst)) () (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1) haccI
  rw [Bd.coinSeq_iv (xI D x) (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)] at hag
  have hov' := (congrArg Bd.openView hv).trans hov
  have hfO : ∀ j < Bd.DE.V.k (xI D x), Bd.oMsg decI H D DO decO P x ω d j = (Bd.opened os last).getD j [] :=
    fun j hj => Bound.oMsg_eq hws hov' hj
  have hc : ∀ n ≤ Bd.DE.V.k (xI D x),
      (List.range n).map (@coinSeq Coin _ (Bd.DE.V.k (xI D x)) (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)) =
        (List.range n).map (coinSeq d.1) := fun n hn =>
    List.map_congr_left fun i hi => by
      rw [List.mem_range] at hi
      exact coinSeq_innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1 (show i < Bd.DE.V.k (xI D x) by omega)
  have hcongr : run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) ω
      (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1) =
      run H (replay (Bd.opened os last)) Bd.DE.V (xI D x) () (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1) := by
    unfold run
    refine stepWith_congr H _ _ (xI D x) ω () _ (fun M hist as => hist.length + M = Bd.DE.V.k (xI D x) ∧
      as = (List.range hist.length).map (fun i => (Bd.opened os last).getD i []) ∧
      hist.map (fun h => decC (xI D x) h.2) = (List.range hist.length).map
        (@coinSeq Coin _ (Bd.DE.V.k (xI D x)) (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1))) ?_ ?_ _ _ _ _
      ⟨by simp, by simp, by simp⟩
    · intro M hist as hG
      have hj : hist.length < Bd.DE.V.k (xI D x) := by omega
      show (OC.ret (Bd.opAtO decI H D DO decO P x ω hist.length (hist.map fun h => decC (xI D x) h.2))).eval H =
        (OC.ret ((Bd.opened os last).getD hist.length [])).eval H
      rw [hG.2.2, hc hist.length hj.le, ← ha _ hj, hfO _ hj]
    · intro M hist as m hG hv
      have hj : hist.length < Bd.DE.V.k (xI D x) := by omega
      have hpa : ((Bd.innerPO decI H D DO decO P x decC).next (xI D x) hist ω).eval H =
          ((Bd.opened os last).getD hist.length [], []) := by
        have e : Bd.opAtO decI H D DO decO P x ω hist.length (hist.map fun h => decC (xI D x) h.2) =
            (Bd.opened os last).getD hist.length [] := by
          rw [hG.2.2, hc hist.length hj.le, ← ha _ hj, hfO _ hj]
        show (OC.ret (Bd.opAtO decI H D DO decO P x ω hist.length (hist.map fun h => decC (xI D x) h.2))).eval H = _
        simp only [e, OC.eval]
      have has : as ++ [(Bd.opened os last).getD hist.length []] =
          (List.range (hist.length + 1)).map fun i => (Bd.opened os last).getD i [] := by
        rw [hG.2.1, List.range_succ, List.map_append, List.map_singleton]
      rw [hpa] at hv ⊢
      dsimp only at hv ⊢
      simp only [has, List.length_map, List.length_range] at hv
      rw [(hag hist.length hj).1] at hv
      unfold Bound.ansAt at hv
      split_ifs at hv with hL
      injection hv with hm
      subst hm
      refine ⟨by simp only [List.length_append, List.length_singleton]; omega, by rw [has]; simp, ?_⟩
      simp only [List.map_append, hG.2.2, List.length_append, List.length_singleton, List.range_succ,
        List.map_singleton, hdec]
      rfl
  obtain ⟨h1, h2, h3⟩ := run_replay_known H Bd.DE.V (xI D x) (Bd.opened os last)
    (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)
    (Bd.ansAt (xI D x) (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1)) hk
    (fun j hj => (hag j hj).1) (fun j hj => ⟨_, by unfold Bound.ansAt; exact ite_eq_left hj⟩)
    (by simp only [Bound.ansAt, show ¬ (Bd.DE.V.k (xI D x) - 1 + 1 < Bd.DE.V.k (xI D x)) by omega, ↓reduceIte])
  rw [hcongr, hov']
  refine ⟨?_, h2, fun e he => ?_⟩
  · simp only [Trace.seen, h1, h2]
  · obtain ⟨j, hj, hej⟩ := h3 e he
    exact (hag j hj).2 e hej

/-- **Off the bad draws, `P_O`'s aborting run shows what the served run opened shows**: a draw where `E_O` gives up or
whose inner view rejects aborts at both, and a kept draw is the inner view opened (`runO`). -/
theorem Bound.servedO_seen (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (hb : ¬ Bd.BadO decI H D DO decO P x ω d) :
    (Bd.servedO decI H D DO decO P x ω d).seen = (Bd.innerFO decI H D DO decO P x decC ω d).seen := by
  unfold Bound.servedO
  rw [servedRun_seen, outerStage_eval]
  change (Bd.openT (dec H D x d.1 (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d))).seen = _
  rcases ho : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d with _ | ws
  · have hk : Bd.keepDO decI H D DO decO P x ω d = false := by
      unfold Bound.keepDO
      simp [ho]
    have e1 : Bd.openT (dec H D x d.1 none) = abortTrace := ite_eq_right (by simp [dec])
    have e2 : Bd.innerFO decI H D DO decO P x decC ω d = abortTrace := by
      unfold Bound.innerFO
      rw [hk]
      rfl
    rw [e1, e2]
  · by_cases hacc : (D.innerView H x d.1 ws).out = .acc
    · have ha : Bd.AgreesO decI H D DO decO P x ω d := by
        by_contra hn
        exact hb ⟨by rw [ho]; rfl, hn⟩
      have hk := Bound.keepDO_true ho hacc ha
      have e : Bd.innerFO decI H D DO decO P x decC ω d = run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V
          (xI D x) ω (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1) := by
        unfold Bound.innerFO
        rw [ite_eq_left hk]
      rw [e, (Bound.runO decC hdec hV ω d ho hacc ha).1]
      exact ((Bd.openT_rel _).2 hacc).symm
    · have hk : Bd.keepDO decI H D DO decO P x ω d = false := by
        unfold Bound.keepDO
        simp [ho, dec, hacc]
      have e1 : Bd.openT (dec H D x d.1 (some ws)) = abortTrace := ite_eq_right hacc
      have e2 : Bd.innerFO decI H D DO decO P x decC ω d = abortTrace := by
        unfold Bound.innerFO
        rw [hk]
        rfl
      rw [e1, e2]

omit [DecidableEq Col] in
theorem Bound.abortRun_eqO (decC : Msg → Msg → Coin) (hn : Bd.DE.V.k (xI D x) ≤ D.V.k x) (ω : Ω)
    (t : (Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    abortRun H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) (Bd.keepAO decI H D DO decO P x) ω t =
      Bd.innerFO decI H D DO decO P x decC ω (extendDraw t) := by
  have hc : innerCoins D (Inner.ofFlock Bd.deploy decI) x (extendDraw t).1 = t.1 :=
    innerCoins_extend D (Inner.ofFlock Bd.deploy decI) x hn t.1 t.2.1
  unfold abortRun Bound.keepAO
  show (if Bd.keepDO decI H D DO decO P x ω (extendDraw t) = true then
      run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) ω t.1 else abortTrace) =
    Bd.innerFO decI H D DO decO P x decC ω (extendDraw t)
  unfold Bound.innerFO
  rw [hc]

/-- **The inner stage fails where `E^rep` at `DE` fails against `P_O`'s aborting runs, at a bad rerun, or where the
judged draw's inner view rejects**: `E*`'s inner `E^rep` at the bound record's openings sees what `E^rep` at `DE`'s sees
of the served runs opened (`erep_rel₂`), which off the bad draws are `P_O`'s aborting runs (`servedO_seen`); the
judged draw is `P_O`'s accepted run where its inner view accepts. A bad rerun costs `N_I` times one draw's
(`Pr_exists_rerun_draws`), and `E^rep` at `P_O`'s aborting runs over their own coins is `Inner.PrBadAbort` at `DE`'s
(`unifMap_extendDraw`). -/
theorem Bound.PrBadGood_le_merged (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI))
    [Fintype Ω] :
    PrBadGood H D.V (innerR D (Inner.ofFlock Bd.deploy decI)) outR (EO D (Inner.ofFlock Bd.deploy decI) DO decO)
        (dec H D) (fun x => (Inner.ofFlock Bd.deploy decI).waits (xI D x))
        (fun x => (Inner.ofFlock Bd.deploy decI).k (xI D x)) (fun x => (Inner.ofFlock Bd.deploy decI).cap (xI D x))
        (Inner.ofFlock Bd.deploy decI).opens (fun x => (Inner.ofFlock Bd.deploy decI).assemble (xI D x))
        (fun x => (Inner.ofFlock Bd.deploy decI).check (xI D x)) x P ≤
      (Inner.ofFlock Bd.DE decI).PrBadAbort H (xI D x) (Bd.innerPO decI H D DO decO P x decC)
          (Bd.keepAO decI H D DO decO P x) +
        NI D (Inner.ofFlock Bd.deploy decI) x *
          Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
            Bd.BadO decI H D DO decO P x t.1 t.2) +
        Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) := by
  have hn : Bd.DE.V.k (xI D x) ≤ D.V.k x := hR x
  let Yb : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
      (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
        (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) → Prop :=
    fun s => (Bd.innerFO decI H D DO decO P x decC s.1 (s.2.1, s.2.2.1)).out = .acc ∧
      ¬ (Inner.ofFlock Bd.DE decI).R.ext (xI D x)
        ((erepProg ((Inner.ofFlock Bd.DE decI).waits (xI D x)) ((Inner.ofFlock Bd.DE decI).k (xI D x))
          ((Inner.ofFlock Bd.DE decI).cap (xI D x)) (Inner.ofFlock Bd.DE decI).opens
          ((Inner.ofFlock Bd.DE decI).assemble (xI D x)) ((Inner.ofFlock Bd.DE decI).check (xI D x))
          (s.2.1, s.2.2.1) s.2.2.2).eval H (Bd.innerFO decI H D DO decO P x decC s.1)).1
  let Zb : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
      (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
        (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) → Prop :=
    fun s => ∃ d ∈ (erepProg ((Inner.ofFlock Bd.DE decI).waits (xI D x))
        ((Inner.ofFlock Bd.DE decI).k (xI D x)) ((Inner.ofFlock Bd.DE decI).cap (xI D x))
        (Inner.ofFlock Bd.DE decI).opens ((Inner.ofFlock Bd.DE decI).assemble (xI D x))
        ((Inner.ofFlock Bd.DE decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
          (Bd.servedO decI H D DO decO P x s.1), Bd.BadO decI H D DO decO P x s.1 d
  let Jb : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
      (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
        (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) → Prop :=
    fun s => InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x (s.1, s.2.1, s.2.2.1)
  have hYZ : PrBadGood H D.V (innerR D (Inner.ofFlock Bd.deploy decI)) outR
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO) (dec H D) (fun x => (Inner.ofFlock Bd.deploy decI).waits (xI D x))
      (fun x => (Inner.ofFlock Bd.deploy decI).k (xI D x)) (fun x => (Inner.ofFlock Bd.deploy decI).cap (xI D x))
      (Inner.ofFlock Bd.deploy decI).opens (fun x => (Inner.ofFlock Bd.deploy decI).assemble (xI D x))
      (fun x => (Inner.ofFlock Bd.deploy decI).check (xI D x)) x P ≤ Pr Yb + Pr Zb + Pr Jb := by
    refine (Pr_mono fun s hs => ?_).trans ((Pr_or_le (fun s => Yb s ∨ Zb s) Jb).trans
      (add_le_add (Pr_or_le Yb Zb) le_rfl))
    by_cases hz : Zb s
    · exact Or.inl (Or.inr hz)
    by_cases hj : Jb s
    · exact Or.inr hj
    left; left
    have hnb : ∀ d ∈ (erepProg ((Inner.ofFlock Bd.DE decI).waits (xI D x))
        ((Inner.ofFlock Bd.DE decI).k (xI D x)) ((Inner.ofFlock Bd.DE decI).cap (xI D x))
        (Inner.ofFlock Bd.DE decI).opens ((Inner.ofFlock Bd.DE decI).assemble (xI D x))
        ((Inner.ofFlock Bd.DE decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
          (Bd.servedO decI H D DO decO P x s.1), ¬ Bd.BadO decI H D DO decO P x s.1 d :=
      fun d hd hb => hz ⟨d, hd, hb⟩
    have heq1 : ((erepProg ((Inner.ofFlock Bd.deploy decI).waits (xI D x))
        ((Inner.ofFlock Bd.deploy decI).k (xI D x)) ((Inner.ofFlock Bd.deploy decI).cap (xI D x))
        (Inner.ofFlock Bd.deploy decI).opens ((Inner.ofFlock Bd.deploy decI).assemble (xI D x))
        ((Inner.ofFlock Bd.deploy decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).eval H
          (servedRun H (Claims.outerStage (EO D (Inner.ofFlock Bd.deploy decI) DO decO) (dec H D) x)
            (run H P D.V x s.1))).1 =
        ((erepProg ((Inner.ofFlock Bd.DE decI).waits (xI D x))
          ((Inner.ofFlock Bd.DE decI).k (xI D x)) ((Inner.ofFlock Bd.DE decI).cap (xI D x))
          (Inner.ofFlock Bd.DE decI).opens ((Inner.ofFlock Bd.DE decI).assemble (xI D x))
          ((Inner.ofFlock Bd.DE decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).eval H
            (Bd.servedO decI H D DO decO P x s.1)).1 :=
      erep_rel₂ H _ (Bd.servedO decI H D DO decO P x s.1) _ _ _ (Inner.ofFlock Bd.deploy decI).opens
        (Inner.ofFlock Bd.DE decI).opens _ _ _ _ fun d =>
          ⟨(Bd.openT_rel _).1, fun h => congrArg (Inner.ofFlock Bd.DE decI).opens ((Bd.openT_rel _).2 h)⟩
    have heq2 := RP.eval_congr H (Bd.servedO decI H D DO decO P x s.1) (Bd.innerFO decI H D DO decO P x decC s.1)
      (erepProg ((Inner.ofFlock Bd.DE decI).waits (xI D x))
        ((Inner.ofFlock Bd.DE decI).k (xI D x)) ((Inner.ofFlock Bd.DE decI).cap (xI D x))
        (Inner.ofFlock Bd.DE decI).opens ((Inner.ofFlock Bd.DE decI).assemble (xI D x))
        ((Inner.ofFlock Bd.DE decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2)
      fun d hd => Bound.servedO_seen decC hdec hV s.1 d (hnb d hd)
    have hd₀ : (s.2.1, s.2.2.1) ∈ (erepProg ((Inner.ofFlock Bd.deploy decI).waits (xI D x))
        ((Inner.ofFlock Bd.deploy decI).k (xI D x)) ((Inner.ofFlock Bd.deploy decI).cap (xI D x))
        (Inner.ofFlock Bd.deploy decI).opens ((Inner.ofFlock Bd.deploy decI).assemble (xI D x))
        ((Inner.ofFlock Bd.deploy decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
          (servedRun H (Claims.outerStage (EO D (Inner.ofFlock Bd.deploy decI) DO decO) (dec H D) x)
            (run H P D.V x s.1)) := List.mem_cons_self
    have hd₀' : (s.2.1, s.2.2.1) ∈ (erepProg ((Inner.ofFlock Bd.DE decI).waits (xI D x))
        ((Inner.ofFlock Bd.DE decI).k (xI D x)) ((Inner.ofFlock Bd.DE decI).cap (xI D x))
        (Inner.ofFlock Bd.DE decI).opens ((Inner.ofFlock Bd.DE decI).assemble (xI D x))
        ((Inner.ofFlock Bd.DE decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
          (Bd.servedO decI H D DO decO P x s.1) := List.mem_cons_self
    have hgood := hs.2.1 _ hd₀
    obtain ⟨ws, hws⟩ : ∃ ws, eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1 (s.2.1, s.2.2.1) = some ws := by
      rcases ho : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1 (s.2.1, s.2.2.1) with _ | ws
      · refine absurd ⟨hs.1, fun h => ?_⟩ hgood
        rw [show ((EO D (Inner.ofFlock Bd.deploy decI) DO decO).eval H P x s.1 s.2.1 s.2.2.1).1 = none from ho] at h
        exact h
      · exact ⟨ws, rfl⟩
    have ha : Bd.AgreesO decI H D DO decO P x s.1 (s.2.1, s.2.2.1) := by
      by_contra hn
      exact hnb _ hd₀' ⟨by rw [hws]; rfl, hn⟩
    have hacc : (D.innerView H x s.2.1 ws).out = .acc := by
      by_contra hne
      exact hj ⟨ws, eoGives_of_some hws, hne⟩
    have hk := Bound.keepDO_true hws hacc ha
    refine ⟨?_, fun hI => hs.2.2 ?_⟩
    · unfold Bound.innerFO
      rw [ite_eq_left hk]
      exact (Bound.runO decC hdec hV s.1 (s.2.1, s.2.2.1) hws hacc ha).2.1
    · show (innerR D (Inner.ofFlock Bd.deploy decI)).ext x (((erepProg ((Inner.ofFlock Bd.deploy decI).waits (xI D x))
        ((Inner.ofFlock Bd.deploy decI).k (xI D x)) ((Inner.ofFlock Bd.deploy decI).cap (xI D x))
        (Inner.ofFlock Bd.deploy decI).opens ((Inner.ofFlock Bd.deploy decI).assemble (xI D x))
        ((Inner.ofFlock Bd.deploy decI).check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).serve
          (Claims.outerStage (EO D (Inner.ofFlock Bd.deploy decI) DO decO) (dec H D) x)).eval H
            (run H P D.V x s.1)).1
      rw [(RP.serve_eval H _ _ _).1, heq1, heq2]
      exact (innerR_ext D (Inner.ofFlock Bd.deploy decI) x _).2 hI
  have hY : Pr Yb = (Inner.ofFlock Bd.DE decI).PrBadAbort H (xI D x) (Bd.innerPO decI H D DO decO P x decC)
      (Bd.keepAO decI H D DO decO P x) := by
    have hΦ : UnifMap (fun s : Ω × ((Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
        (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
        (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
          (Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
            (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
        (s.1, (extendDraw s.2.1).1, (extendDraw s.2.1).2, fun i j => extendDraw (s.2.2 i j))) := by
      let e : Ω × ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
          (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
            (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ≃
          Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
          (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
            (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) :=
        { toFun := fun s => (s.1, s.2.1.1, s.2.1.2, s.2.2)
          invFun := fun s => (s.1, (s.2.1, s.2.2.1), s.2.2.2)
          left_inv := fun _ => rfl
          right_inv := fun _ => rfl }
      exact (UnifMap.self.prod ((unifMap_extendDraw hn).prod
        (UnifMap.pi (UnifMap.pi (unifMap_extendDraw hn) _) _))).comp (UnifMap.equiv e)
    rw [← hΦ.Pr Yb]
    refine Pr_congr fun s => ?_
    show (Bd.innerFO decI H D DO decO P x decC s.1 (extendDraw s.2.1)).out = .acc ∧
      ¬ (Inner.ofFlock Bd.DE decI).R.ext (xI D x)
        ((erepProg ((Inner.ofFlock Bd.DE decI).waits (xI D x)) ((Inner.ofFlock Bd.DE decI).k (xI D x))
          ((Inner.ofFlock Bd.DE decI).cap (xI D x)) (Inner.ofFlock Bd.DE decI).opens
          ((Inner.ofFlock Bd.DE decI).assemble (xI D x)) ((Inner.ofFlock Bd.DE decI).check (xI D x))
          (extendDraw s.2.1) (fun i j => extendDraw (s.2.2 i j))).eval H
            (Bd.innerFO decI H D DO decO P x decC s.1)).1 ↔ _
    have hf : (fun d => Bd.innerFO decI H D DO decO P x decC s.1 (extendDraw d)) =
        abortRun H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) (Bd.keepAO decI H D DO decO P x) s.1 :=
      funext fun d => (Bound.abortRun_eqO decC hn s.1 d).symm
    rw [erepProg_eval_comp H (Bd.innerFO decI H D DO decO P x decC s.1) extendDraw, hf,
      ← Bound.abortRun_eqO decC hn]
    rfl
  have hZ : Pr Zb ≤ NI D (Inner.ofFlock Bd.deploy decI) x *
      Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) := by
    have hN : (NI D (Inner.ofFlock Bd.deploy decI) x : ℝ) = 1 + ((Inner.ofFlock Bd.deploy decI).waits (xI D x) : ℝ) *
        ((Inner.ofFlock Bd.deploy decI).k (xI D x) : ℝ) := by
      simp only [NI]; push_cast; ring
    rw [hN]
    let e : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
        (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
          (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ≃
        Ω × ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
        (Fin ((Inner.ofFlock Bd.DE decI).waits (xI D x)) → Fin ((Inner.ofFlock Bd.DE decI).cap (xI D x)) →
          (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) :=
      { toFun := fun s => (s.1, (s.2.1, s.2.2.1), s.2.2.2)
        invFun := fun t => (t.1, t.2.1.1, t.2.1.2, t.2.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    have hpe := Pr_exists_rerun_draws H (Bd.servedO decI H D DO decO P x)
      ((Inner.ofFlock Bd.DE decI).waits (xI D x)) ((Inner.ofFlock Bd.DE decI).k (xI D x))
      ((Inner.ofFlock Bd.DE decI).cap (xI D x)) (Inner.ofFlock Bd.DE decI).opens
      ((Inner.ofFlock Bd.DE decI).assemble (xI D x)) ((Inner.ofFlock Bd.DE decI).check (xI D x))
      (Bd.BadO decI H D DO decO P x)
    rw [← Pr_equiv e] at hpe
    exact hpe
  have hJ : Pr Jb = Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) :=
    (unifMap_drop (Ω := Ω) (R := Fin (D.V.k x) → Coin)
      (C := (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)).Pr
      (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x)
  linarith

/-- `P_O`'s aborting run records only calls among those `cl` searches at its draw, or those collide
(`CallsLogCollBy`). -/
theorem Bound.innerFO_vLog_subBy (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    {cl : (x : Msg) → (Fin (D.V.k x) → Coin) → List Flock.Wit → List (Msg × Msg)}
    (hL : CallsLogCollBy H D (Inner.ofFlock Bd.deploy decI) DO cl)
    (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    (∀ e ∈ (Bd.innerFO decI H D DO decO P x decC ω d).vLog,
      e ∈ cl x d.1 ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).getD [])) ∨
      ∃ u v, findColl (cl x d.1 ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).getD [])) =
        some (u, v) ∧ Collision H u v := by
  by_cases hk : Bd.keepDO decI H D DO decO P x ω d = true
  · have hk' : (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true ∧
        (dec H D x d.1 (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d)).out = .acc ∧
        Bd.AgreesO decI H D DO decO P x ω d := by
      simpa only [Bound.keepDO, decide_eq_true_eq] using hk
    obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hk'.1
    have hacc : (D.innerView H x d.1 ws).out = .acc := by
      have h := hk'.2.1
      rw [hws] at h
      exact h
    have hrun := Bound.runO decC hdec hV ω d hws hacc hk'.2.2
    have hF : Bd.innerFO decI H D DO decO P x decC ω d = run H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V
        (xI D x) ω (innerCoins D (Inner.ofFlock Bd.deploy decI) x d.1) := by
      unfold Bound.innerFO
      rw [ite_eq_left hk]
    rw [hF, hws]
    rcases hL decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws) with h | h
    · exact Or.inl fun e he => h e (hrun.2.2 e he)
    · exact Or.inr h
  · exact Or.inl fun e he => by simp [Bound.innerFO, hk, abortTrace] at he

/-- `P_O`'s aborting run records only calls that `E_O`'s witnesses there record, or those witnesses' calls collide
(`CallsLogColl`). -/
theorem Bound.innerFO_vLog_sub (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    (∀ e ∈ (Bd.innerFO decI H D DO decO P x decC ω d).vLog,
      e ∈ D.calls x ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).getD [])) ∨
      ∃ u v, findColl (D.calls x ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).getD [])) =
        some (u, v) ∧ Collision H u v :=
  Bound.innerFO_vLog_subBy (cl := fun x _ ws => D.calls x ws) decC hdec hV hL ω d

/-- **`B` at `P_O`'s aborting runs finds only collisions among the calls `cl` searches at `B_I`'s fork**: each kept
run's calls are among those at the draw it extends to, or those collide (`CallsLogCollBy`), and an aborted run makes
none; `B`'s fork of the inner coins extends to `B_I`'s fork of the compiled verifier's (`extend_forkCoins`,
`unifMap_bAbort`). -/
theorem Bound.PrBAbort_le_by (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    {cl : (x : Msg) → (Fin (D.V.k x) → Coin) → List Flock.Wit → List (Msg × Msg)}
    (hL : CallsLogCollBy H D (Inner.ofFlock Bd.deploy decI) DO cl)
    (ht : ∀ (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x),
      ∀ e ∈ cl x d.1 ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).getD []), H e.1 = e.2)
    (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI)) [Fintype Ω] [Nonempty Ω] (hk : 0 < Bd.DE.V.k (xI D x)) :
    PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) ≤
      Pr (fun s : Fin (Bd.DE.V.k (xI D x)) × Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) ×
          (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        ∃ u v, findColl (cl x s.2.2.1 ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.2.1
            (s.2.2.1, s.2.2.2.2.1)).getD []) ++
          cl x (fork s.1 s.2.2.1 s.2.2.2.1) ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.2.1
            (fork s.1 s.2.2.1 s.2.2.2.1, s.2.2.2.2.2)).getD [])) = some (u, v) ∧ Collision H u v) := by
  have hn : Bd.DE.V.k (xI D x) ≤ D.V.k x := hR x
  have hcoll : ∀ (j : Fin (Bd.DE.V.k (xI D x))) (ω : Ω) (ρ ρ' : Fin (Bd.DE.V.k (xI D x)) → Coin)
      (a a' : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x),
      (findColl ((abortRun H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x)
          (Bd.keepAO decI H D DO decO P x) ω (ρ, a)).vLog ++
        (abortRun H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x) (Bd.keepAO decI H D DO decO P x) ω
          (forkCoins j ρ ρ', a')).vLog)).isSome →
      ∃ u v, findColl (cl x (extend ρ a.1) ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω
          (extend ρ a.1, a.2)).getD []) ++
        cl x (fork j.val (extend ρ a.1) (extend ρ' a'.1)) ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω
          (fork j.val (extend ρ a.1) (extend ρ' a'.1), a'.2)).getD [])) = some (u, v) ∧ Collision H u v := by
    intro j ω ρ ρ' a a' h
    rw [Bound.abortRun_eqO decC hn, Bound.abortRun_eqO decC hn] at h
    rw [extend_forkCoins]
    have htrue : ∀ p ∈ cl x (extend ρ a.1) ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω
          (extend ρ a.1, a.2)).getD []) ++
        cl x (extend (forkCoins j ρ ρ') a'.1) ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω
          (extend (forkCoins j ρ ρ') a'.1, a'.2)).getD []), H p.1 = p.2 :=
      fun p hp => (List.mem_append.1 hp).elim (ht ω (extend ρ a.1, a.2) p)
        (ht ω (extend (forkCoins j ρ ρ') a'.1, a'.2) p)
    have hsome : (findColl (cl x (extend ρ a.1) ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω
          (extend ρ a.1, a.2)).getD []) ++
        cl x (extend (forkCoins j ρ ρ') a'.1) ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω
          (extend (forkCoins j ρ ρ') a'.1, a'.2)).getD []))).isSome := by
      rcases Bound.innerFO_vLog_subBy decC hdec hV hL ω (extendDraw (ρ, a)) with h1 | ⟨u, v, huv, -⟩
      · rcases Bound.innerFO_vLog_subBy decC hdec hV hL ω (extendDraw (forkCoins j ρ ρ', a')) with
          h2 | ⟨u, v, huv, -⟩
        · exact findColl_isSome_of_sub H htrue (fun p hp => (List.mem_append.1 hp).elim
            (fun hp => List.mem_append_left _ (h1 p hp)) fun hp => List.mem_append_right _ (h2 p hp)) h
        · exact findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_right _ hp)
            (Option.isSome_iff_exists.2 ⟨_, huv⟩)
      · exact findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_left _ hp)
          (Option.isSome_iff_exists.2 ⟨_, huv⟩)
    obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hsome
    exact ⟨u, v, huv, findColl_spec H _ htrue u v huv⟩
  unfold PrBAbort
  simp only [hk, ↓reduceDIte]
  unfold Finder.Pr
  refine (Pr_mono fun s hs => ?_).trans_eq
    ((unifMap_bAbort (Ω := Ω) (C := (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) hn).Pr _)
  obtain ⟨u, v, huv, -⟩ := hs
  exact hcoll s.1.1 s.1.2.1 s.1.2.2.1 s.1.2.2.2 s.2.1 s.2.2 (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩)

/-- **`B` at `P_O`'s aborting runs finds only collisions `B_I` finds**: `PrBAbort_le_by` at the recorded calls
(`CallsLogColl`), true where `E_O` gives witnesses and where it gives up (`calls_true`). -/
theorem Bound.PrBAbort_le_merged (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hN0 : CallsNil H D)
    (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI)) [Fintype Ω] [Nonempty Ω] :
    PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) ≤
      PrBI H D (Inner.ofFlock Bd.deploy decI) DO decO P x := by
  unfold PrBI
  by_cases hk : 0 < Bd.DE.V.k (xI D x)
  · have hk' : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI D x) := hk
    simp only [hk', ↓reduceDIte]
    exact Bound.PrBAbort_le_by (cl := fun x _ ws => D.calls x ws) decC hdec hV hL
      (fun ω d => calls_true hT hN0 ω d) hR hk
  · have hk' : ¬ 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI D x) := hk
    unfold PrBAbort
    simp only [hk, hk', ↓reduceDIte, le_refl]

/-- **The binding core of `fork_collO`, over the calls `cl` searches**: `fork_collO_core` with the runs' calls among
`cl`'s at each draw, or those colliding (`CallsLogCollBy`). -/
theorem Bound.fork_collO_coreBy {Q : Prop} (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    {cl : (x : Msg) → (Fin (D.V.k x) → Coin) → List Flock.Wit → List (Msg × Msg)}
    (hT : ∀ (ω : Ω) r c ws, EOGives H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c ws →
      ∀ e ∈ cl x r ws, H e.1 = e.2)
    (hL : CallsLogCollBy H D (Inner.ofFlock Bd.deploy decI) DO cl)
    (j : ℕ) (hj : j < Bd.DE.V.k (xI D x))
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) {ws ws' : List Flock.Wit}
    (hws : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (t.2.1, t.2.2.2.1) = some ws)
    (hws' : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) = some ws')
    (hacc₁ : (D.innerView H x t.2.1 ws).out = .acc) (hacc₂ : (D.innerView H x (fork j t.2.1 t.2.2.1) ws').out = .acc)
    (hraw : ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD j [] =
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD j [])
    (hne : Bd.oMsg decI H D DO decO P x t.1 (t.2.1, t.2.2.2.1) j ≠
      Bd.oMsg decI H D DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) j)
    (hlift : ∀ l : List (Msg × Msg), (∀ p ∈ l, p ∈ cl x t.2.1 ws ++ cl x (fork j t.2.1 t.2.2.1) ws') →
      (findColl l).isSome → Q) : Q := by
  have hE₁ : EOGives H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 t.2.1 t.2.2.2.1 ws := eoGives_of_some hws
  have hE₂ : EOGives H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' :=
    eoGives_of_some hws'
  have htrue : ∀ p ∈ cl x t.2.1 ws ++ cl x (fork j t.2.1 t.2.2.1) ws', H p.1 = p.2 := fun p hp =>
    (List.mem_append.1 hp).elim (hT t.1 _ _ ws hE₁ p) (hT t.1 _ _ ws' hE₂ p)
  have hleft : (∃ u v, findColl (cl x t.2.1 ws) = some (u, v) ∧ Collision H u v) → Q := fun ⟨_, _, huv, _⟩ =>
    hlift _ (fun p hp => List.mem_append_left _ hp) (Option.isSome_iff_exists.2 ⟨_, huv⟩)
  have hright : (∃ u v, findColl (cl x (fork j t.2.1 t.2.2.1) ws') = some (u, v) ∧ Collision H u v) → Q :=
    fun ⟨_, _, huv, _⟩ =>
      hlift _ (fun p hp => List.mem_append_right _ hp) (Option.isSome_iff_exists.2 ⟨_, huv⟩)
  have hv₁ := hV decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hE₁
  have hv₂ := hV decO x Ω P t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hE₂
  have haccI₁ : (run H (replay ((D.innerView H x t.2.1 ws).msgs.map Prod.fst)) Bd.IVStar (xI D x) ()
      (innerCoins D (Inner.ofFlock Bd.deploy decI) x t.2.1)).out = .acc := by
    have h := hacc₁
    rw [hv₁] at h
    exact h
  have haccI₂ : (run H (replay ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst)) Bd.IVStar (xI D x)
      () (innerCoins D (Inner.ofFlock Bd.deploy decI) x (fork j t.2.1 t.2.2.1))).out = .acc := by
    have h := hacc₂
    rw [hv₂] at h
    exact h
  obtain ⟨-, os, last, hf, hlen, hrounds, -, hov⟩ := Bd.accFacts H (xI D x)
    (replay ((D.innerView H x t.2.1 ws).msgs.map Prod.fst)) () (innerCoins D (Inner.ofFlock Bd.deploy decI) x t.2.1)
    haccI₁
  obtain ⟨-, os', last', hf', hlen', hrounds', -, hov'⟩ := Bd.accFacts H (xI D x)
    (replay ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst)) ()
    (innerCoins D (Inner.ofFlock Bd.deploy decI) x (fork j t.2.1 t.2.2.1)) haccI₂
  have hfO₁ : Bd.oMsg decI H D DO decO P x t.1 (t.2.1, t.2.2.2.1) j = (Bd.opened os last).getD j [] :=
    Bound.oMsg_eq hws ((congrArg Bd.openView hv₁).trans hov) hj
  have hfO₂ : Bd.oMsg decI H D DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) j =
      (Bd.opened os' last').getD j [] :=
    Bound.oMsg_eq hws' ((congrArg Bd.openView hv₂).trans hov') hj
  have hm₁ : (D.innerView H x t.2.1 ws).msgs.map Prod.fst =
      (run H (replay ((D.innerView H x t.2.1 ws).msgs.map Prod.fst)) Bd.IVStar (xI D x) ()
        (innerCoins D (Inner.ofFlock Bd.deploy decI) x t.2.1)).msgs.map Prod.fst :=
    congrArg (fun v : Seen => v.msgs.map Prod.fst) hv₁
  have hm₂ : (D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst =
      (run H (replay ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst)) Bd.IVStar (xI D x) ()
        (innerCoins D (Inner.ofFlock Bd.deploy decI) x (fork j t.2.1 t.2.2.1))).msgs.map Prod.fst :=
    congrArg (fun v : Seen => v.msgs.map Prod.fst) hv₂
  have hms : ((run H (replay ((D.innerView H x t.2.1 ws).msgs.map Prod.fst)) Bd.IVStar (xI D x) ()
        (innerCoins D (Inner.ofFlock Bd.deploy decI) x t.2.1)).msgs.map Prod.fst).getD j [] =
      ((run H (replay ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst)) Bd.IVStar (xI D x) ()
        (innerCoins D (Inner.ofFlock Bd.deploy decI) x (fork j t.2.1 t.2.2.1))).msgs.map Prod.fst).getD j [] :=
    (congrArg (fun l : List Msg => l.getD j []) hm₁).symm.trans
      (hraw.trans (congrArg (fun l : List Msg => l.getD j []) hm₂))
  by_cases hj1 : j + 1 < Bd.DE.V.k (xI D x)
  · have hjo : j < os.length := by omega
    have hjo' : j < os'.length := by omega
    obtain ⟨hn, hop, hlog⟩ := hrounds j hjo
    obtain ⟨hn', hop', hlog'⟩ := hrounds' j hjo'
    have hne' : (Bd.opened os last).getD j [] ≠ (Bd.opened os' last').getD j [] :=
      fun e => hne (hfO₁.trans (e.trans hfO₂.symm))
    rw [Bd.opened_getD_lt hjo, Bd.opened_getD_lt hjo'] at hne'
    have hLo : os.getD j [] ≠ os'.getD j [] := fun e => hne' (by rw [e])
    obtain ⟨i, hi, hi', hdi⟩ : ∃ i, ∃ (hi : i < (os.getD j []).length) (hi' : i < (os'.getD j []).length),
        (os.getD j [])[i] ≠ (os'.getD j [])[i] := by
      by_contra hno
      push Not at hno
      exact hLo (List.ext_getElem (hn.trans hn'.symm) hno)
    rw [hms] at hop hlog
    obtain ⟨h1, hl1⟩ := (Bd.opensRound_true H hop).2 i hi
    obtain ⟨h2, hl2⟩ := (Bd.opensRound_true H hop').2 i hi'
    obtain ⟨u, hu, v, hvm, hc⟩ := Bd.opens_coll H hB h1 h2 fun e => hdi e
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hu
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hvm
    have hp₁ : p ∈ (innerRun H D (Inner.ofFlock Bd.deploy decI) x t.2.1 ws).vLog := hlog p (hl1 p hp)
    have hq₂ : q ∈ (innerRun H D (Inner.ofFlock Bd.deploy decI) x (fork j t.2.1 t.2.2.1) ws').vLog :=
      hlog' q (hl2 q hq)
    rcases hL decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hE₁ with hs₁ | hc₁
    swap
    · exact hleft hc₁
    rcases hL decO x Ω P t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hE₂ with hs₂ | hc₂
    swap
    · exact hright hc₂
    have hpm : p ∈ cl x t.2.1 ws ++ cl x (fork j t.2.1 t.2.2.1) ws' := List.mem_append_left _ (hs₁ p hp₁)
    have hqm : q ∈ cl x t.2.1 ws ++ cl x (fork j t.2.1 t.2.2.1) ws' := List.mem_append_right _ (hs₂ q hq₂)
    obtain ⟨u', v', huv', -⟩ := findColl_of_coll H htrue (List.mem_map.2 ⟨p, hpm, rfl⟩)
      (List.mem_map.2 ⟨q, hqm, rfl⟩) hc
    exact hlift _ (fun p hp => hp) (Option.isSome_iff_exists.2 ⟨_, huv'⟩)
  · exfalso
    have hj' : Bd.DE.V.k (xI D x) - 1 = j := by omega
    rw [hj'] at hf hf'
    rw [hms, hf'] at hf
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf)
    exact hne (hfO₁.trans hfO₂.symm)

/-- **The binding core of `fork_collO`**: two draws forked at inner round `j`, with witnesses `ws`, `ws'` whose inner
views accept and agree at message `j` but open it differently, show a collision among the calls their witnesses record,
which any `Q` that such a collision gives (`hlift`) inherits. The views are accepted `IVStar` runs on one commit string:
before the last round its rows differ at some commit string, opened two ways (`opens_coll`, `Binds`), among the runs'
calls, which the witnesses record (`CallsLogColl`); at the last round `final` reads one message, so they open alike. -/
theorem Bound.fork_collO_core {Q : Prop} (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (j : ℕ) (hj : j < Bd.DE.V.k (xI D x))
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) {ws ws' : List Flock.Wit}
    (hws : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (t.2.1, t.2.2.2.1) = some ws)
    (hws' : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) = some ws')
    (hacc₁ : (D.innerView H x t.2.1 ws).out = .acc) (hacc₂ : (D.innerView H x (fork j t.2.1 t.2.2.1) ws').out = .acc)
    (hraw : ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD j [] =
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD j [])
    (hne : Bd.oMsg decI H D DO decO P x t.1 (t.2.1, t.2.2.2.1) j ≠
      Bd.oMsg decI H D DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) j)
    (hlift : ∀ l : List (Msg × Msg), (∀ p ∈ l, p ∈ D.calls x ws ++ D.calls x ws') → (findColl l).isSome → Q) : Q :=
  Bound.fork_collO_coreBy (cl := fun x _ ws => D.calls x ws) hB hV (fun ω r c ws hg => hT decO x Ω P ω r c ws hg) hL
    j hj t hws hws' hacc₁ hacc₂ hraw hne hlift

/-- **Two draws forked at inner round `j`, with witnesses, whose openings differ at `j`, are a collision `Be`'s fork
finds.** A rejecting inner view is one among its witnesses' calls (`InnerAcceptsColl`). Inner views that differ at
message `j` are `ForksBind`'s. Inner views that agree at message `j` are `fork_collO_core`'s. -/
theorem Bound.fork_collO (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBindAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    (j : ℕ) (hj : j < Bd.DE.V.k (xI D x))
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (h : (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (t.2.1, t.2.2.2.1)).isSome = true ∧
      (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2)).isSome = true ∧
      Bd.oMsg decI H D DO decO P x t.1 (t.2.1, t.2.2.2.1) j ≠
        Bd.oMsg decI H D DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) j) :
    forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j t := by
  obtain ⟨hg₁, hg₂, hne⟩ := h
  obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hg₁
  obtain ⟨ws', hws'⟩ := Option.isSome_iff_exists.1 hg₂
  have hE₁ : EOGives H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 t.2.1 t.2.2.2.1 ws := eoGives_of_some hws
  have hE₂ : EOGives H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' :=
    eoGives_of_some hws'
  have htrue : ∀ p ∈ D.calls x ws ++ D.calls x ws', H p.1 = p.2 := fun p hp =>
    (List.mem_append.1 hp).elim (hT decO x Ω P t.1 _ _ ws hE₁ p) (hT decO x Ω P t.1 _ _ ws' hE₂ p)
  have hlift : ∀ l : List (Msg × Msg), (∀ p ∈ l, p ∈ D.calls x ws ++ D.calls x ws') → (findColl l).isSome →
      forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j t := by
    intro l hsub hs
    have hl₁ : ((EO D (Inner.ofFlock Bd.deploy decI) DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD [] = ws := by
      rw [hE₁.2]; rfl
    have hl₂ : ((EO D (Inner.ofFlock Bd.deploy decI) DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1)
        t.2.2.2.2).1.getD [] = ws' := by
      rw [hE₂.2]; rfl
    unfold forkColl
    rw [hl₁, hl₂]
    obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome_of_sub H htrue hsub hs)
    exact ⟨u, v, huv, findColl_spec H _ htrue u v huv⟩
  rcases hA decO Ω P t.1 t.2.1 t.2.2.2.1 ws hE₁ with hacc₁ | ⟨_, _, huv, _⟩
  swap
  · exact hlift _ (fun p hp => List.mem_append_left _ hp) (Option.isSome_iff_exists.2 ⟨_, huv⟩)
  rcases hA decO Ω P t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hE₂ with hacc₂ | ⟨_, _, huv, _⟩
  swap
  · exact hlift _ (fun p hp => List.mem_append_right _ hp) (Option.isSome_iff_exists.2 ⟨_, huv⟩)
  by_cases hraw : ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD j [] =
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD j []
  swap
  · exact hFk decO Ω P j t ⟨ws, ws', hE₁, hE₂, hraw⟩
  exact Bound.fork_collO_core hB hV hT hL j hj t hws hws' hacc₁ hacc₂ hraw hne hlift

/-- **A draw is bad at most as `√(k_I·Σ_{j<k_I} Pr[forkColl j])`**: it misses `P_O` at some inner round `j`, at most as
`√(Pr[two draws forked at j open round j differently])` (`Pr_plur_fork`), a collision `Be`'s fork at `j` finds
(`fork_collO`); the rounds' roots sum under one (Cauchy–Schwarz). -/
theorem Bound.Pr_badO_le_fork (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBind H D (Inner.ofFlock Bd.deploy decI) DO) (hA : InnerAcceptsColl H D (Inner.ofFlock Bd.deploy decI) DO)
    [Fintype Ω] [Nonempty Ω] :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) ≤
      Real.sqrt ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
        ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
          Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j)) := by
  have h1 : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) ≤
      ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
        Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 t.2).isSome = true ∧
            Bd.oMsg decI H D DO decO P x t.1 t.2 j ≠
              Bd.opAtO decI H D DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) := by
    refine (Pr_mono fun t ht => ?_).trans (Pr_exists_le _)
    obtain ⟨hs, hna⟩ := ht
    simp only [Bound.AgreesO, not_forall] at hna
    obtain ⟨j, hj, hne⟩ := hna
    exact ⟨⟨j, hj⟩, hs, hne⟩
  have h2 : ∀ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
      Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 t.2).isSome = true ∧
          Bd.oMsg decI H D DO decO P x t.1 t.2 j ≠
            Bd.opAtO decI H D DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) ≤
      Real.sqrt (Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j)) := fun j => by
    have hp := Pr_plur_fork j (fun ω d => (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true)
      (fun ω d => Bd.oMsg decI H D DO decO P x ω d j)
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans (Real.sqrt_le_sqrt (Pr_mono fun t ht => ?_)))
    · simp only [Bound.opAtO, fork_preCoins]
    · exact Bound.fork_collO hB hV hT hL (hFk.toAt x) (hA.toAt x) j j.isLt t ht
  have hcs := Real.sum_sqrt_mul_sqrt_le (Finset.univ : Finset (Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x))))
    (f := fun _ : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) => (1 : ℝ))
    (g := fun j => Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j))
    (fun _ => zero_le_one) (fun j => Pr_nonneg _)
  simp only [Real.sqrt_one, one_mul, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at hcs
  calc Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2)
      ≤ ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
          Real.sqrt (Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j)) :=
        h1.trans (Finset.sum_le_sum fun j _ => h2 j)
    _ ≤ Real.sqrt ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) *
          Real.sqrt (∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
            Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j)) := hcs
    _ = Real.sqrt ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
            Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j)) :=
        (Real.sqrt_mul (Nat.cast_nonneg _) _).symm

/-- **Bad draws and a rejecting judged draw together cost `InnerKS`'s bad-rerun term**: `bad_rej_le_binds`'s argument at
`P_O`'s bad draws (`Pr_badO_le_fork`). -/
theorem Bound.bad_rej_le_merged (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBind H D (Inner.ofFlock Bd.deploy decI) DO) (hA : InnerAcceptsColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI)) (hN0 : CallsNil H D)
    (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI)) [Fintype Ω] [Nonempty Ω] :
    NI D (Inner.ofFlock Bd.deploy decI) x *
        Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          Bd.BadO decI H D DO decO P x t.1 t.2) +
        Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) ≤
      NI D (Inner.ofFlock Bd.deploy decI) x * Real.sqrt ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) * D.V.k x *
        PrBe H D (Inner.ofFlock Bd.deploy decI) DO decO P x) := by
  have hn0 : (0 : ℝ) ≤ NI D (Inner.ofFlock Bd.deploy decI) x := Nat.cast_nonneg _
  by_cases hE : Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) = 0
  · rw [hE, add_zero]
    refine mul_le_mul_of_nonneg_left ?_ hn0
    rw [mul_assoc]
    exact (Bound.Pr_badO_le_fork hB hV hT hL hFk hA).trans (Real.sqrt_le_sqrt
      (mul_le_mul_of_nonneg_left (sum_forkColl_inner_le hR decO x P) (Nat.cast_nonneg _)))
  obtain ⟨t, ws, hg, -⟩ : ∃ t, InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x t := by
    by_contra hn
    refine hE ((Pr_congr fun t => ⟨fun h => hn ⟨t, h⟩, False.elim⟩).trans Pr_false)
  obtain ⟨hkI, hk3⟩ := hM x Ω P t.1 t.2.1 hg.1
  let S : ℕ → ℝ := fun j => Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j)
  have hk : 0 < D.V.k x := by omega
  have hES : ∀ j, Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) ≤ S j := fun j =>
    Pr_innerRejects_le_fork hA hT hN0 decO x P j
  have htot : (D.V.k x : ℝ) * PrBe H D (Inner.ofFlock Bd.deploy decI) DO decO P x =
      ∑ j ∈ Finset.range (D.V.k x), S j := by
    rw [PrBe_eq_avg H D (Inner.ofFlock Bd.deploy decI) DO decO P x hk, mul_div_cancel₀ _ (by exact_mod_cast hk.ne'),
      Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j))]
  have hsplit : ∑ j ∈ Finset.range (D.V.k x), S j =
      ∑ j ∈ Finset.range ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), S j +
        ∑ j ∈ Finset.Ico ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) (D.V.k x), S j :=
    (Finset.sum_range_add_sum_Ico S (hR x)).symm
  have hB3 : 3 * Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) ≤
      ∑ j ∈ Finset.Ico ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) (D.V.k x), S j := by
    have hcard : (3 : ℝ) ≤ (Finset.Ico ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) (D.V.k x)).card := by
      rw [Nat.card_Ico]
      exact_mod_cast (by omega : 3 ≤ D.V.k x - (Inner.ofFlock Bd.deploy decI).V.k (xI D x))
    calc 3 * Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x)
        ≤ (Finset.Ico ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) (D.V.k x)).card *
            Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) :=
          mul_le_mul_of_nonneg_right hcard (Pr_nonneg _)
      _ = ∑ _j ∈ Finset.Ico ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) (D.V.k x),
            Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ Finset.Ico ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) (D.V.k x), S j :=
          Finset.sum_le_sum fun j _ => hES j
  have hAa : ∑ j ∈ Finset.range ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), S j ≤
      (Inner.ofFlock Bd.deploy decI).V.k (xI D x) := by
    calc ∑ j ∈ Finset.range ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), S j
        ≤ ∑ _j ∈ Finset.range ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), (1 : ℝ) :=
          Finset.sum_le_sum fun j _ => Pr_le_one _
      _ = (Inner.ofFlock Bd.deploy decI).V.k (xI D x) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
  have hA0 : 0 ≤ ∑ j ∈ Finset.range ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), S j :=
    Finset.sum_nonneg fun j _ => Pr_nonneg _
  have hbad := Bound.Pr_badO_le_fork (decO := decO) (P := P) (x := x) hB hV hT hL hFk hA
  rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D (Inner.ofFlock Bd.deploy decI) DO decO P x j))] at hbad
  have hn1 : (1 : ℝ) ≤ NI D (Inner.ofFlock Bd.deploy decI) x := by
    have : 1 ≤ NI D (Inner.ofFlock Bd.deploy decI) x := by simp only [NI]; omega
    exact_mod_cast this
  have hm := sqrt_margin hn1 (by exact_mod_cast hkI : (1 : ℝ) ≤ (Inner.ofFlock Bd.deploy decI).V.k (xI D x)) hA0 hAa
    (Pr_nonneg _) (Pr_le_one _) hB3
  calc (NI D (Inner.ofFlock Bd.deploy decI) x : ℝ) *
        Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          Bd.BadO decI H D DO decO P x t.1 t.2) +
        Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x)
      ≤ NI D (Inner.ofFlock Bd.deploy decI) x * Real.sqrt ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
            ∑ j ∈ Finset.range ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), S j) +
          Pr (InnerRejects H D (Inner.ofFlock Bd.deploy decI) DO decO P x) :=
        add_le_add (mul_le_mul_of_nonneg_left hbad hn0) le_rfl
    _ ≤ NI D (Inner.ofFlock Bd.deploy decI) x * Real.sqrt ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          (∑ j ∈ Finset.range ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), S j +
            ∑ j ∈ Finset.Ico ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) (D.V.k x), S j)) := hm
    _ = NI D (Inner.ofFlock Bd.deploy decI) x * Real.sqrt ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) * D.V.k x *
          PrBe H D (Inner.ofFlock Bd.deploy decI) DO decO P x) := by
        rw [mul_assoc, htot, hsplit]

/-- **`InnerKS` at the bound record from (P1) at `DE`, with no bridge**: the inner stage fails at most as `E^rep` at
`DE` against `P_O`'s aborting runs, which `Inner.KSAbort` at `DE` bounds by `κ + √(γ_DE·Pr[B at P_O's aborting runs])`,
with `Pr[B at P_O's aborting runs] ≤ Pr[B_I]` (`PrBAbort_le_merged`), plus `P_O`'s bad and rejecting draws, at most
`N_I·√(k_I·k(x)·Pr[Be])` (`bad_rej_le_merged`). The bound record's `κ` is `DE`'s, and its `γ` is at least `DE`'s
where its weight is (`hw`): no weight pays a bridge. -/
theorem Bound.innerKS_of_plainKSAbort_binds [DecidableEq Col'] [Zero Col'] (decC : Msg → Msg → Coin)
    (hKS : (Inner.ofFlock Bd.DE decI).KSAbort H) (hw : ∀ x, Bd.DE.weight x ≤ Bd.weight x)
    (hdec : ∀ x c, decC x (Bd.ans c) = c) (hB : Bd.Binds) (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI))
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO)
    (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO) (hFk : ForksBind H D (Inner.ofFlock Bd.deploy decI) DO)
    (hA : InnerAcceptsColl H D (Inner.ofFlock Bd.deploy decI) DO) (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI))
    (hN0 : CallsNil H D) : InnerKS H D (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO _ x Ω _ _ P
  have h1 := Bound.PrBadGood_le_merged (decO := decO) (P := P) (x := x) decC hdec hV hR
  have h2 : (Inner.ofFlock Bd.DE decI).PrBadAbort H (xI D x) (Bd.innerPO decI H D DO decO P x decC)
      (Bd.keepAO decI H D DO decO P x) ≤
      (Inner.ofFlock Bd.deploy decI).κ (xI D x) + Real.sqrt ((Inner.ofFlock Bd.DE decI).γ (xI D x) *
        PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)) :=
    hKS (xI D x) Ω (Bd.innerPO decI H D DO decO P x decC)
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (Bd.keepAO decI H D DO decO P x)
  have h3 := Bound.PrBAbort_le_merged (decO := decO) (P := P) (x := x) decC hdec hV hL hT hN0 hR
  have h4 := Bound.bad_rej_le_merged (decO := decO) (P := P) (x := x) hB hV hT hL hFk hA hR hN0 hM
  have hBA0 : 0 ≤ PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
      (Bd.keepAO decI H D DO decO P x) := PrBAbort_nonneg H _ _ _ _
  have hk0 : (0 : ℝ) ≤ Bd.DE.V.k (xI D x) := Nat.cast_nonneg _
  have hs : Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) ≤
      Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.weight (xI D x)) :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hw _) (by positivity))
  have h0 : 0 ≤ Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x) := by
    positivity
  have hs' : Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x) ≤
      Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.weight (xI D x)) + Bd.DE.V.k (xI D x) := by linarith
  have hγ : (Inner.ofFlock Bd.DE decI).γ (xI D x) ≤ (Inner.ofFlock Bd.deploy decI).γ (xI D x) := by
    show 16 * (Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x)) ^ 2 ≤
      16 * (Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.weight (xI D x)) + Bd.DE.V.k (xI D x)) ^ 2
    nlinarith [mul_le_mul hs' hs' h0 (h0.trans hs')]
  have hγ0 : 0 ≤ (Inner.ofFlock Bd.DE decI).γ (xI D x) := by
    show 0 ≤ 16 * (Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x)) ^ 2
    positivity
  have h5 : Real.sqrt ((Inner.ofFlock Bd.DE decI).γ (xI D x) *
        PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)) ≤
      Real.sqrt ((Inner.ofFlock Bd.deploy decI).γ (xI D x) * PrBI H D (Inner.ofFlock Bd.deploy decI) DO decO P x) :=
    Real.sqrt_le_sqrt (mul_le_mul hγ h3 hBA0 (hγ0.trans hγ))
  linarith

/-- **The recursion's KS at the bound record from Flock's KS at the outer deployment and (P1) at `DE`**, with no
bridge and no `WeightCovers`: `KS_of_flockKS_binds` with `innerKS_of_plainKSAbort_binds` for its inner stage. -/
theorem Bound.KS_of_plainKSAbort_binds [DecidableEq Col'] [Zero Col'] (decC : Msg → Msg → Coin)
    (hF : Flock.KS H DO) (hKS : (Inner.ofFlock Bd.DE decI).KSAbort H) (hw : ∀ x, Bd.DE.weight x ≤ Bd.weight x)
    (hdec : ∀ x c, decC x (Bd.ans c) = c) (hB : Bd.Binds) (hRef : Refuses H D)
    (hRounds : InnerRounds D (Inner.ofFlock Bd.deploy decI)) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : SessionRounds D (Inner.ofFlock Bd.deploy decI) DO) (hS : SessionAccepts D)
    (hRs : SessionRuns H D (Inner.ofFlock Bd.deploy decI) DO) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBind H D (Inner.ofFlock Bd.deploy decI) DO) (hA : InnerAcceptsColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI)) (hN0 : CallsNil H D) :
    KSA H D (Inner.ofFlock Bd.deploy decI) DO :=
  KS_of H D (Inner.ofFlock Bd.deploy decI) DO hRef hRounds (outerKS_of_flockKS hF hN hε hRd hS hRs)
    (Bound.innerKS_of_plainKSAbort_binds decC hKS hw hdec hB hRounds hV hT hL hFk hA hM hN0)

end Merged

end Claims.Recursion
