module

public import Proofs.Flock.RecGeneric.AtFlockReruns
public import Proofs.Flock.Claims.Recursion.HresAsm

/-!
# L1 at Flock: the compiler's contract at the deployment's dense statements

At today's compiled record (`Comp.ofRec D I.V I DO decO`) each of `Comp.Contract`'s fields at a statement `x` is a fact
`Claims.Recursion` states about the record `D.record I.V`:

* `replays` is `EO_some_acc` and `ownTrue` is `setupCalls_true`; `refuses`, `rounds`, `margin` and `callsTrue` are
  `Claims.Recursion.Refuses`, `InnerRounds`, `RoundsMargin` and `CallsTrue` read at `x`;
* `callsLog` is `CallsLogCollAt`, the inner view's run being the record's (`innerRun_ofRec`, from `RecAt.runs`);
* `accepts` is `InnerAcceptsCollAtV`, the compiled verifier's own evaluations being the setup calls at its run's staged
  statements;
* `forks`, over the bound record (`compAt`), is `Bound.LastBindsEC` at the last inner round and void before it
  (`forks_compAt`): two accepted views at a fork at round `j` agree before the last round, where the readers' messages
  are a choice of the run's, which the fork keeps. The inner extractor reads a view as `DE`'s columns of the view
  opened, so two readings that differ, or two call sets that differ, are `LastBindsEC`'s premise as it stands.

At the deployment (`RecDeploy.recBoundAt`, `RecDeploy.bdDeployed`) those facts are the ones `ksLin2_route_stagedVS`
derives (`RecDeploy.contract_route_stagedVS`), from the suppliers `RecursiveKS`'s proof reads
(`RecDeploy.contract_deployed`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

theorem coll_lift {H : Msg → Msg} {L l : List (Msg × Msg)} (hL : ∀ p ∈ L, H p.1 = p.2) (hsub : ∀ p ∈ l, p ∈ L)
    (h : ∃ u v, findColl l = some (u, v) ∧ Collision H u v) : ∃ u v, findColl L = some (u, v) ∧ Collision H u v := by
  obtain ⟨u, v, huv, -⟩ := h
  exact VStar.coll_of_sub hL hsub (Option.isSome_iff_exists.2 ⟨_, huv⟩)

theorem coll_of_mem {H : Msg → Msg} {L l : List (Msg × Msg)} (hL : ∀ p ∈ L, H p.1 = p.2) (hsub : ∀ p ∈ l, p ∈ L)
    {q q' : Msg} (hq : q ∈ l.map Prod.fst) (hq' : q' ∈ l.map Prod.fst) (hc : Collision H q q') :
    ∃ u v, findColl L = some (u, v) ∧ Collision H u v := by
  have hm : ∀ y ∈ l.map Prod.fst, y ∈ L.map Prod.fst := fun y hy => by
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hy
    exact List.mem_map.2 ⟨e, hsub e he, rfl⟩
  exact findColl_of_coll H hL (hm q hq) (hm q' hq') hc

/-! ## The contract's fields at today's compiled record -/

section Bridge

variable {Coin W Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col']
  (D : RecAt Coin) (I : Claims.Recursion.Inner Coin W) (DO : Claims.Flock.Deploy Coin Col' T')
  (decO : List (ℕ × Col') → List T') {H : Msg → Msg} {x : Msg}

theorem replays_ofRec (IV : Verifier Coin) : (Comp.ofRec D IV I DO decO).Replays H x :=
  fun _ P ω r c _ h => EO_some_acc H (D.record IV) I DO decO P x ω r c h

theorem refuses_ofRec (IV : Verifier Coin) (h : Claims.Recursion.Refuses H (D.record IV)) :
    (Comp.ofRec D IV I DO decO).Refuses H x :=
  fun Ω P ω r hr => h x Ω P ω r hr

theorem rounds_ofRec (h : Claims.Recursion.InnerRounds (D.record I.V) I) :
    (Comp.ofRec D I.V I DO decO).InnerRounds I.V x :=
  h x

theorem margin_ofRec (h : Claims.Recursion.RoundsMargin H (D.record I.V) I) :
    (Comp.ofRec D I.V I DO decO).RoundsMargin H I.V x :=
  fun Ω P ω r hr => h x Ω P ω r hr

theorem callsTrue_ofRec (IV : Verifier Coin) (h : Claims.Recursion.CallsTrue H (D.record IV) I DO) :
    (Comp.ofRec D IV I DO decO).CallsTrue H x :=
  fun Ω P ω r c w hg => callsAt_true_of (h decO x Ω P ω r c w hg)

theorem ownTrue_ofRec (IV : Verifier Coin) : (Comp.ofRec D IV I DO decO).OwnTrue H x :=
  fun _ _ _ _ => setupCalls_true H (D.record IV) _

/-- **The inner view's run is the record's** (`RecAt.runs`): the inner verifier's run of the inner messages the
witnesses give. -/
theorem innerRun_ofRec (hxI : D.xI x = xI (D.record I.V) x) (r : Fin ((D.record I.V).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    innerRun H (D.record I.V) I x r ws =
      run H (replay (D.msgs x ws)) I.V (xI (D.record I.V) x) () fun i => coinSeq r i.val := by
  unfold innerRun
  rw [D.runs I.V H x r ws, hxI]
  exact RecDeploy.run_replay_seen H I.V _ _ _

theorem callsLog_ofRec (hxI : D.xI x = xI (D.record I.V) x) (h : CallsLogCollAt H (D.record I.V) I DO) :
    (Comp.ofRec D I.V I DO decO).CallsLogColl H I.V x := by
  intro Ω P ω r c w hg _
  rcases h decO x Ω P ω r c w hg with h | h
  · refine Or.inl fun e he => h e ?_
    rw [innerRun_ofRec D I hxI r w]
    exact he
  · exact Or.inr h

theorem accepts_ofRec (hxI : D.xI x = xI (D.record I.V) x)
    (hT : Claims.Recursion.CallsTrue H (D.record I.V) I DO) (hA : InnerAcceptsCollAtV H (D.record I.V) I DO x) :
    (Comp.ofRec D I.V I DO decO).InnerAcceptsColl H I.V x := by
  intro Ω P ω r c w hg _
  rcases hA decO Ω P ω r c w hg with h | h
  · exact Or.inl ((congrArg (·.out) (dec_ofRec D I.V I DO decO hxI r (some w))).trans h)
  · exact Or.inr (coll_lift
      (l := (D.record I.V).calls x w ++ setupCalls H (D.record I.V) (runVstar H (D.record I.V) I P x ω r))
      (fun e he => (List.mem_append.1 he).elim (callsAt_true_of (hT decO x Ω P ω r c w hg) e)
        (setupCalls_true H _ _ e))
      (fun _ hp => mem_callsAt_append hp) h)

end Bridge

/-! ## The forks bind over the bound record -/

section Forks

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col'] (D : RecAt Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
  (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {H : Msg → Msg} {x : Msg}

/-- **Two accepted reruns forked at an inner round bind**, at today's compiled record over the bound record, where the
inner extractor reads a view as `DE`'s columns of the view opened: before the last inner round the two views' messages
agree (the readers' messages are `pk` of the run's, `hP`, which the fork keeps), and at the last, two readings or call
sets that differ are `LastBindsEC`'s premise, both views accepting (`hA`), so `IVStar`'s calls collide, and they are
among those the witnesses record (`hL`). -/
theorem forks_compAt (hxI : D.xI x = xI (D.record Bd.IVStar) x) (pk : ℕ → Msg → Msg) (hEC : Bd.LastBindsEC H)
    (hT : Claims.Recursion.CallsTrue H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : CallsLogCollAt H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO)
    (hA : InnerAcceptsCollAtV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO x)
    (hP : ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin ((D.record Bd.IVStar).V.k x) → Coin)
      (c : (EO (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x) (ws : List Claims.Flock.Wit),
      EOGives H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c ws →
      (∀ j, j + 1 < Bd.DE.V.k (xI (D.record Bd.IVStar) x) → (D.msgs x ws).getD j [] =
          pk j (((run H P (D.record Bd.IVStar).V x ω r).msgs.map Prod.fst).getD j [])) ∨
        ∃ u v, findColl ((D.record Bd.IVStar).calls x ws) = some (u, v) ∧ Collision H u v) :
    (compAt D Bd decI DO decO).ForksBind H Bd.IVStar (fun v => [Bd.DE.cols (Bd.openView v)]) x := by
  intro Ω P j ω r r' c c' w w' hj hg hg' _ _ hne hcond
  have htrue : ∀ p ∈ (D.record Bd.IVStar).callsAt H x r w ++
      setupCalls H (D.record Bd.IVStar) (runVstar H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) P x ω r) ++
      ((D.record Bd.IVStar).callsAt H x (fork j r r') w' ++
        setupCalls H (D.record Bd.IVStar)
          (runVstar H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) P x ω (fork j r r'))), H p.1 = p.2 := by
    intro p hp
    rcases List.mem_append.1 hp with hp | hp <;> rcases List.mem_append.1 hp with hp | hp
    · exact callsAt_true_of (hT decO x Ω P ω r c w hg) p hp
    · exact setupCalls_true H _ _ p hp
    · exact callsAt_true_of (hT decO x Ω P ω (fork j r r') c' w' hg') p hp
    · exact setupCalls_true H _ _ p hp
  have hk : j < Bd.DE.V.k (xI (D.record Bd.IVStar) x) := hj
  have hv : ∀ (r₀ : Fin ((D.record Bd.IVStar).V.k x) → Coin) (w₀ : List Claims.Flock.Wit),
      (((compAt D Bd decI DO decO).view Bd.IVStar H x r₀ w₀).msgs.map Prod.fst).getD j [] =
        (D.msgs x w₀).getD j [] := fun r₀ w₀ =>
    (Bd.run_ivstar_fst H (xI (D.record Bd.IVStar) x) (D.msgs x w₀) (fun i => coinSeq r₀ i.val) j).trans (if_pos hk)
  have hne' : (D.msgs x w).getD j [] ≠ (D.msgs x w').getD j [] := fun h =>
    hne ((hv r w).trans (h.trans (hv _ w').symm))
  rcases hP Ω P ω r c w hg with h₁ | h₁
  swap
  · exact coll_lift htrue (fun p hp => List.mem_append_left _ (List.mem_append_left _ (mem_callsAt hp))) h₁
  rcases hP Ω P ω (fork j r r') c' w' hg' with h₂ | h₂
  swap
  · exact coll_lift htrue (fun p hp => List.mem_append_right _ (List.mem_append_left _ (mem_callsAt hp))) h₂
  have hag : ∀ i, i + 1 < Bd.DE.V.k (xI (D.record Bd.IVStar) x) → i ≤ j →
      (D.msgs x w).getD i [] = (D.msgs x w').getD i [] := fun i hi hij =>
    (h₁ i hi).trans ((congrArg (pk i) (run_fst_getD H P _ x ω i fun i' hi' =>
      fork_agree j r r' i' (by omega))).trans (h₂ i hi).symm)
  by_cases hj1 : j + 1 < Bd.DE.V.k (xI (D.record Bd.IVStar) x)
  · exact absurd (hag j hj1 le_rfl) hne'
  have hk0 : 0 < Bd.DE.V.k (xI (D.record Bd.IVStar) x) := by omega
  have hdec : ∀ (r₀ : Fin ((D.record Bd.IVStar).V.k x) → Coin) (w₀ : List Claims.Flock.Wit),
      (run H (replay (D.msgs x w₀)) Bd.IVStar (xI (D.record Bd.IVStar) x) () fun i => coinSeq r₀ i.val).out =
          (Bd.decAt H (xI (D.record Bd.IVStar) x) (D.msgs x w₀) fun i => coinSeq r₀ i.val).1 ∧
        (run H (replay (D.msgs x w₀)) Bd.IVStar (xI (D.record Bd.IVStar) x) () fun i => coinSeq r₀ i.val).vLog =
          (Bd.decAt H (xI (D.record Bd.IVStar) x) (D.msgs x w₀) fun i => coinSeq r₀ i.val).2 := fun r₀ w₀ => by
    rw [Bd.run_ivstar H _ _ _ hk0]
    exact ⟨rfl, rfl⟩
  have hcols : ∀ (r₀ : Fin ((D.record Bd.IVStar).V.k x) → Coin) (w₀ : List Claims.Flock.Wit),
      Bd.DE.cols (Bd.openView ((compAt D Bd decI DO decO).view Bd.IVStar H x r₀ w₀)) =
        Bd.DE.cols (Bd.openView (Bd.viewAt H (xI (D.record Bd.IVStar) x)
          ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map fun i => (D.msgs x w₀).getD i [])
          ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map
            (coinSeq fun i : Fin (Bd.IVStar.k (xI (D.record Bd.IVStar) x)) => coinSeq r₀ i.val)))) :=
    fun r₀ w₀ => congrArg (fun v => Bd.DE.cols (Bd.openView v))
      (Bd.seen_viewAt H (xI (D.record Bd.IVStar) x) (D.msgs x w₀) (fun i => coinSeq r₀ i.val) hk0)
  have hdiff : Bd.DE.cols (Bd.openView (Bd.viewAt H (xI (D.record Bd.IVStar) x)
        ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map fun i => (D.msgs x w).getD i [])
        ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map
          (coinSeq fun i : Fin (Bd.IVStar.k (xI (D.record Bd.IVStar) x)) => coinSeq r i.val)))) ≠
      Bd.DE.cols (Bd.openView (Bd.viewAt H (xI (D.record Bd.IVStar) x)
        ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map fun i => (D.msgs x w').getD i [])
        ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map
          (coinSeq fun i : Fin (Bd.IVStar.k (xI (D.record Bd.IVStar) x)) => coinSeq (fork j r r') i.val)))) ∨
      (Bd.decAt H (xI (D.record Bd.IVStar) x) (D.msgs x w) fun i => coinSeq r i.val).2.toFinset ≠
        (Bd.decAt H (xI (D.record Bd.IVStar) x) (D.msgs x w') fun i => coinSeq (fork j r r') i.val).2.toFinset := by
    rcases hcond with hc | hc | hc
    · exact absurd hc hj1
    · exact Or.inl fun h => hc (congrArg (fun l => [l]) ((hcols r w).trans (h.trans (hcols _ w').symm)))
    · exact Or.inr fun h => hc ((congrArg List.toFinset (hdec r w).2).trans
        (h.trans (congrArg List.toFinset (hdec _ w').2).symm))
  have hview : ∀ (r₀ : Fin ((D.record Bd.IVStar).V.k x) → Coin) (w₀ : List Claims.Flock.Wit),
      ((D.record Bd.IVStar).innerView H x r₀ w₀).out =
        (run H (replay (D.msgs x w₀)) Bd.IVStar (xI (D.record Bd.IVStar) x) () fun i => coinSeq r₀ i.val).out :=
    fun r₀ w₀ => congrArg (·.out) (dec_ofRec D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO hxI r₀ (some w₀)).symm
  have hacc : ∀ (r₀ : Fin ((D.record Bd.IVStar).V.k x) → Coin) (w₀ : List Claims.Flock.Wit),
      ((D.record Bd.IVStar).innerView H x r₀ w₀).out = .acc →
      (Bd.decAt H (xI (D.record Bd.IVStar) x) (D.msgs x w₀) fun i => coinSeq r₀ i.val).1 = .acc :=
    fun r₀ w₀ h => (hdec r₀ w₀).1.symm.trans ((hview r₀ w₀).symm.trans h)
  have hlog : ∀ (r₀ : Fin ((D.record Bd.IVStar).V.k x) → Coin) (w₀ : List Claims.Flock.Wit),
      (∀ e ∈ (innerRun H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x r₀ w₀).vLog,
        e ∈ (D.record Bd.IVStar).callsAt H x r₀ w₀) →
      ∀ e ∈ (Bd.decAt H (xI (D.record Bd.IVStar) x) (D.msgs x w₀) fun i => coinSeq r₀ i.val).2,
        e ∈ (D.record Bd.IVStar).callsAt H x r₀ w₀ := by
    intro r₀ w₀ h e he
    have hr : (innerRun H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x r₀ w₀).vLog =
        (Bd.decAt H (xI (D.record Bd.IVStar) x) (D.msgs x w₀) fun i => coinSeq r₀ i.val).2 :=
      (congrArg Trace.vLog (innerRun_ofRec D (Inner.ofFlock Bd.deploy decI) hxI r₀ w₀)).trans (hdec r₀ w₀).2
    exact h e (by rw [hr]; exact he)
  rcases hA decO Ω P ω r c w hg with ha₁ | ha₁
  swap
  · exact coll_lift htrue (fun p hp => (List.mem_append.1 hp).elim
      (fun h => List.mem_append_left _ (List.mem_append_left _ (mem_callsAt h)))
      (fun h => List.mem_append_left _ (List.mem_append_right _ h))) ha₁
  rcases hA decO Ω P ω (fork j r r') c' w' hg' with ha₂ | ha₂
  swap
  · exact coll_lift htrue (fun p hp => (List.mem_append.1 hp).elim
      (fun h => List.mem_append_right _ (List.mem_append_left _ (mem_callsAt h)))
      (fun h => List.mem_append_right _ (List.mem_append_right _ h))) ha₂
  rcases hL decO x Ω P ω r c w hg with hl₁ | hl₁
  swap
  · exact coll_lift htrue (fun p hp => List.mem_append_left _ (List.mem_append_left _ hp)) hl₁
  rcases hL decO x Ω P ω (fork j r r') c' w' hg' with hl₂ | hl₂
  swap
  · exact coll_lift htrue (fun p hp => List.mem_append_right _ (List.mem_append_left _ hp)) hl₂
  have hcs : ∀ i, i < j → coinSeq (fun i : Fin (Bd.IVStar.k (xI (D.record Bd.IVStar) x)) => coinSeq r i.val) i =
      coinSeq (fun i : Fin (Bd.IVStar.k (xI (D.record Bd.IVStar) x)) => coinSeq (fork j r r') i.val) i :=
    fun i hi => coinSeq_congr fun _ => coinSeq_congr fun h => fork_agree j r r' ⟨i, h⟩ hi
  obtain ⟨q, hq, q', hq', hc⟩ := hEC (xI (D.record Bd.IVStar) x)
    ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map fun i => (D.msgs x w).getD i [])
    ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map fun i => (D.msgs x w').getD i [])
    ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map
      (coinSeq fun i : Fin (Bd.IVStar.k (xI (D.record Bd.IVStar) x)) => coinSeq r i.val))
    ((List.range (Bd.DE.V.k (xI (D.record Bd.IVStar) x))).map
      (coinSeq fun i : Fin (Bd.IVStar.k (xI (D.record Bd.IVStar) x)) => coinSeq (fork j r r') i.val))
    (by simp) (by simp) (by simp) (by simp)
    (by
      rw [range_map_dropLast, range_map_dropLast]
      exact List.map_congr_left fun i hi => by
        rw [List.mem_range] at hi
        exact hag i (by omega) (by omega))
    (by
      rw [range_map_dropLast, range_map_dropLast]
      exact List.map_congr_left fun i hi => hcs i (by rw [List.mem_range] at hi; omega))
    hdiff (hacc r w ha₁) (hacc _ w' ha₂)
  exact coll_of_mem htrue (fun e he => (List.mem_append.1 he).elim
    (fun h => List.mem_append_left _ (List.mem_append_left _ (hlog r w hl₁ e h)))
    (fun h => List.mem_append_right _ (List.mem_append_left _ (hlog _ w' hl₂ e h)))) hq hq' hc

/-- **The contract at today's compiled record over the bound record**, at the statement `x`, from the record's facts
there: `Claims.Recursion`'s refusal, rounds, margin, true calls, searched inner calls and V\*'s inner check at `x`, the
readers' messages before the last round a choice of the run's, and the last message's binding (`LastBindsEC`). -/
theorem contract_compAt (hxI : D.xI x = xI (D.record Bd.IVStar) x) (pk : ℕ → Msg → Msg) (hEC : Bd.LastBindsEC H)
    (hRef : Claims.Recursion.Refuses H (D.record Bd.IVStar))
    (hR : Claims.Recursion.InnerRounds (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
    (hM : Claims.Recursion.RoundsMargin H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
    (hT : Claims.Recursion.CallsTrue H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : CallsLogCollAt H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO)
    (hA : InnerAcceptsCollAtV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO x)
    (hP : ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin ((D.record Bd.IVStar).V.k x) → Coin)
      (c : (EO (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x) (ws : List Claims.Flock.Wit),
      EOGives H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c ws →
      (∀ j, j + 1 < Bd.DE.V.k (xI (D.record Bd.IVStar) x) → (D.msgs x ws).getD j [] =
          pk j (((run H P (D.record Bd.IVStar).V x ω r).msgs.map Prod.fst).getD j [])) ∨
        ∃ u v, findColl ((D.record Bd.IVStar).calls x ws) = some (u, v) ∧ Collision H u v) :
    (compAt D Bd decI DO decO).Contract H Bd.IVStar (fun v => [Bd.DE.cols (Bd.openView v)]) x where
  replays := replays_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO Bd.IVStar
  refuses := refuses_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO Bd.IVStar hRef
  rounds := rounds_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO hR
  margin := margin_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO hM
  callsTrue := callsTrue_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO Bd.IVStar hT
  ownTrue := ownTrue_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO Bd.IVStar
  callsLog := callsLog_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO hxI hL
  forks _ := forks_compAt D Bd decI DO decO hxI pk hEC hT hL hA hP
  accepts _ := accepts_ofRec D (Inner.ofFlock Bd.deploy decI) DO decO hxI hT hA

/-- **The contract at the server's `Open(DE, hm96)` and its reading of opened views**, which `RecoversLinkOn` reads,
is the contract at the bound record's `IVStar` and opened views (`IVStarOfBound`, `OpenViewOfBound`): the element-level
Flock's opening of a view is `DE`'s columns of it. -/
theorem contract_ofBound (Cp : Caps Coin) (hIV : IVStarOfBound Bd) (hOV : OpenViewOfBound Bd)
    (h : (compAt D Bd decI DO decO).Contract H Bd.IVStar (fun v => [Bd.DE.cols (Bd.openView v)]) x) :
    (compAt D Bd decI DO decO).Contract H ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)
      (fun v => (Proto.ofFlock Bd decI Cp).opens ((Server.ofBound Bd).openView v)) x := by
  rw [hIV, hOV]
  exact h

end Forks

end Claims.RecGeneric

/-! ## At the deployment -/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin FinalP)
open Claims.Recursion (Bound Inner)

/-- **The contract at the deployed records' dense statements**, from the premises of `ksLin2_route_stagedVS` that its
V\* facts read: the calls of no witnesses are true (`callsNil`), and at every inner and outer decoder the compiled
record's refusal (`refuses`), rounds (`innerRounds`, `roundsMargin_deployRec`), true calls (`callsTrue`), searched inner
calls (`callsLogCollAt_deployRec`), V\*'s inner check at a dense statement (`innerAcceptsCollAtV_of_dense` at
`vstarAcceptsV_boundVS`), the readers' messages before the last round (`readers_run`) and `LastBindsEC`. -/
theorem contract_route_stagedVS (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole)
    (ok : Msg → List Msg → List Claims.Flock.Wit → Prop) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      Bd.LastBindsEC FlockDeploy.sha512 →
      (∀ shape ∈ [Claims.Recursion.Form.Dense],
        ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.StagedB FlockDeploy.sha512 D
            (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          (∀ decI, OkRuns FlockDeploy.sha512 D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) shape ok) ∧
          Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
            (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt) ok) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar { b with rd := L.readers } (deployZkRec relO) →
      ∀ x, DenseAt x → (∀ e ∈ D.calls x [], FlockDeploy.sha512 e.1 = e.2) ∧
        ∀ decI decO, (Claims.RecGeneric.compAt (recBoundAt rel b L) Bd decI (deployZkRec relO) decO).Contract
          FlockDeploy.sha512 Bd.IVStar (fun v => [Bd.DE.cols (Bd.openView v)]) x := by
  intro nextE colsE wE join rowIn final w Bd D hEC hshapes hT x hx
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  let bL : Brings := { b with rd := L.readers }
  refine ⟨callsNil Bd.IVStar ivScope bL FlockDeploy.sha512 x, fun decI decO => ?_⟩
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  obtain ⟨shows, cmt, hHe, hSt, hOp, hOk, hDc⟩ := hshapes _ (List.mem_singleton_self _)
  have hacc : Claims.Recursion.VStarAcceptsV FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO)
      Claims.Recursion.Form.Dense :=
    vstarAcceptsV_boundVS FlockDeploy.sha512 I ivScope b L (deployZkRec relO) _ shows cmt ok hHW (hHe decI)
      (hSt decI) hOp hDc (hOk decI) hRows
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hA := Claims.Recursion.innerAcceptsCollAtV_of_dense (refuses I.V ivScope bL FlockDeploy.sha512) hTc hacc hx
  exact Claims.RecGeneric.contract_compAt (recBoundAt rel b L) Bd decI (deployZkRec relO) decO rfl (L.pick x) hEC
    (refuses I.V ivScope bL FlockDeploy.sha512) (innerRounds I ivScope bL)
    (roundsMargin_deployRec FlockDeploy.sha512 I ivScope bL fun x => V_k_pos_flock false x) hTc
    (callsLogCollAt_deployRec FlockDeploy.sha512 I ivScope bL (deployZkRec relO)) hA
    (fun Ω P ω r c ws hg => by
      obtain ⟨st, hst, -⟩ := refuses I.V ivScope bL FlockDeploy.sha512 x Ω P ω r hg.1
      exact readers_run FlockDeploy.sha512 I ivScope b L (deployZkRec relO) _ shows cmt hHW (hHe decI) hOp hst
        (hx st hst) (hSt decI x st hst (hx st hst) Ω P ω r hg.1) (Claims.Recursion.eoGives_outerHolds hg)
        (hTc decO x Ω P ω r c ws hg))

section LayR

open Flock

/-- **The contract at `showsDec`** (`contract_route_stagedVS` as `ksLin2_stepsC_atVD` reaches `ksLin2_route_stagedVS`):
`StagedB` from `stagedB_deployRec_dec`, the openings at `showsPart`, `LastBindsEC` from `StepsBaseC`. -/
theorem contract_stepsC_atVD (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ReadersTrue FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (deployZkRec _root_.Flock.Readable.relation) →
    (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L
      Claims.Recursion.Form.Dense) →
    (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
      Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s) →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (ResLower.showsDec recRead bR) (cmtTr recRead))
        fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) →
    ∀ x, DenseAt x → (∀ e ∈ D.calls x [], FlockDeploy.sha512 e.1 = e.2) ∧
      ∀ decI decO, (Claims.RecGeneric.compAt (recBoundAt _root_.Flock.Readable.relation bR L) Bd decI
        (deployZkRec _root_.Flock.Readable.relation) decO).Contract FlockDeploy.sha512 Bd.IVStar
        (fun v => [Bd.DE.cols (Bd.openView v)]) x := by
  intro bR Bd D hRT hHe hPO hD
  obtain ⟨-, hEC, -⟩ := hB
  have hrb : ReadsBackB FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (fun _ => True) recRead :=
    readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl
  refine contract_route_stagedVS _root_.Flock.Readable.relation _root_.Flock.Readable.relation bR L hRows hHW
    (fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) nextE colsE wE join rowInDeployed final w hEC ?_ hRT
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  exact ⟨ResLower.showsDec recRead bR, cmtTr recRead, hHe,
    fun decI => stagedB_deployRec_dec FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } recRead L _
      (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    fun s => OpenLower.partOpensB_shows (fun h => h.1) (hPO s),
    fun decI => okRuns_deployRec FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } _, hD _ hsh⟩

/-- **`contract_stepsC_atVD` at `layR`**, as `ksLin2_stepsC_layRVD`. -/
theorem contract_stepsC_layRVD (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (ResLower.showsDec recRead bR) (cmtTr recRead))
        fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) →
    ∀ x, DenseAt x → (∀ e ∈ D.calls x [], FlockDeploy.sha512 e.1 = e.2) ∧
      ∀ decI decO, (Claims.RecGeneric.compAt (recBoundAt _root_.Flock.Readable.relation bR L) Bd decI
        (deployZkRec _root_.Flock.Readable.relation) decO).Contract FlockDeploy.sha512 Bd.IVStar
        (fun v => [Bd.DE.cols (Bd.openView v)]) x :=
  fun hD => contract_stepsC_atVD nextE colsE b aux nOf hd recRest (DeployedR.layR elem) (readsRows_layR elem)
    (heldWhole_layR elem) wE join final w hB (readersTrue_layR elem _ _ _ hmSlotsCommit_of_full hM)
    (heldB_dense_layR elem nextE colsE _ wE join final w) (partOpensB_dense_layR elem _ _) hD

/-- **`contract_stepsC_layRVD` at the deployed reader's steps** (`stepsBase_R`), as `ksLin2_deployedR_layRVD`. -/
theorem contract_deployedR_layRVD (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (ResLower.showsDec recRead bR) (cmtTr recRead))
        fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) →
    ∀ x, DenseAt x → (∀ e ∈ D.calls x [], FlockDeploy.sha512 e.1 = e.2) ∧
      ∀ decI decO, (Claims.RecGeneric.compAt (recBoundAt _root_.Flock.Readable.relation bR L) Bd decI
        (deployZkRec _root_.Flock.Readable.relation) decO).Contract FlockDeploy.sha512 Bd.IVStar
        (fun v => [Bd.DE.cols (Bd.openView v)]) x :=
  contract_stepsC_layRVD _ _ b aux nOf hd recRest elem wE join final w (stepsBase_R rd wE join final w hIn) hM

end LayR

section Deployed

open RecDeploy.DecompBR RecDeploy.RestLow

/-- **The contract at the deployment's dense statements**, at every inner and outer decoder: `contract_deployedR_layRVD`
at the deployed reader `DeployedR.rdR`, its last message's reading `DeployedR.finalR`, the elements `OpenRows.elemR`,
the deployed weights and brings, with `DecompBVS` from chain B's leaves as
`Flock.SecurityProofs.recursiveKSAt_of_leaves` discharges them, at the residual leaf `ResLower.hres_deployed` (any
key), the setup's listing `resSetup_deployedB` and the last message's reading `RestRead.finalReads`,
`RestElem.readsAt_elemR_closed`. It takes no hypothesis. -/
theorem contract_deployed (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) :
    let bR : Brings := { deployedB b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR OpenRows.elemR
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER FlockDeploy.joinR
      rowInDeployed DeployedR.finalR DeployedR.wR
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ∀ x, DenseAt x → (∀ e ∈ D.calls x [], FlockDeploy.sha512 e.1 = e.2) ∧
      ∀ decI decO, (Claims.RecGeneric.compAt (recBoundAt _root_.Flock.Readable.relation bR L) Bd decI
        (deployZkRec _root_.Flock.Readable.relation) decO).Contract FlockDeploy.sha512 Bd.IVStar
        (fun v => [Bd.DE.cols (Bd.openView v)]) x := by
  have hF := RestRead.finalReads OpenRows.elemR
  have hRd := RestElem.readsAt_elemR_closed
  have hres := ResLower.hres_deployed DeployedR.wER DeployedR.wR b aux nOf hd recRest key
  have hset := resSetup_deployedB b
  have hO : OpenLower.CopyOpensR FlockDeploy.sha512
      (recL DeployedR.rdR (deployedB b) aux nOf hd recRest DeployedR.wER DeployedR.finalR DeployedR.wR
        (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB (deployedB b) aux nOf hd recRest) (cmtTr recRead)
      (DeployedR.layR OpenRows.elemR) (bdR DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR)
      (OpenLower.sideAtR DeployedR.rdR) (Rows.rowDPieces key) :=
    _root_.Flock.SecurityProofs.copyOpensR_bdDeployed (OpenRows.copyOpensR_layR FlockDeploy.sha512
      (recL DeployedR.rdR (deployedB b) aux nOf hd recRest DeployedR.wER DeployedR.finalR DeployedR.wR
        (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB (deployedB b) aux nOf hd recRest) (cmtTr recRead) (Rows.rowDPieces key)
      (fun _ => rfl) (fun _ h => h) (fun _ => rfl) (OpenLower.piecesInRangeR_rowDPieces key))
  have hI : ∀ s x c (st0 : Claims.Recursion.CompiledStatement), RecDeploy.parse x = some st0 →
      Claims.Recursion.Form.Dense st0.form → OpenDirs.InnerMR recRead (bB (deployedB b) aux nOf hd recRest) s x c :=
    fun s x c _ hst hdn xO hsh i hins f unit inn hf hload => by
      obtain ⟨v, hv, hm, -⟩ := OpenLower.staged_innerAt (b := bB (deployedB b) aux nOf hd recRest)
        (Bk := (FlockDeploy.V false).k) (s := s) (c := c) hst hdn le_rfl xO hsh i hins f unit inn hf hload
      exact ⟨v, hv, hm⟩
  have hpos : ∀ s x c (st0 : Claims.Recursion.CompiledStatement), RecDeploy.parse x = some st0 →
      Claims.Recursion.Form.Dense st0.form →
      OpenDirs.DirsCoinR recRead (bB (deployedB b) aux nOf hd recRest) (Rows.rowDPieces key) Rows.posAt
        (bdR DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR).DE.V.k s x c :=
    fun s x c _ hst hdn => OpenLower.posTieR_dense (b := bB (deployedB b) aux nOf hd recRest)
      (Bk := (bdR DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR).DE.V.k) (s := s) (c := c) hst hdn
      (OpenLower.piecesInRangeR_rowDPieces key) le_rfl
  have hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512
      (recL DeployedR.rdR (deployedB b) aux nOf hd recRest DeployedR.wER DeployedR.finalR DeployedR.wR
        (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB (deployedB b) aux nOf hd recRest) (cmtTr recRead)
      (DeployedR.layR OpenRows.elemR) (bdR DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR)
      (Rows.rowDPieces key) Rows.posAt :=
    OpenDirs.copyDirsTieR_layR (Rows.rowDPieces key) Rows.posAt
      (OpenDirs.dirsSideR_of (Rows.rowDPieces key) Rows.posAt (fun _ => rfl) (fun _ h => h)
        (OpenLower.piecesInRangeR_rowDPieces key) hset hI hpos)
  have hOut : CapLower.OutBindRP FlockDeploy.sha512
      (recL DeployedR.rdR (deployedB b) aux nOf hd recRest DeployedR.wER DeployedR.finalR DeployedR.wR
        (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB (deployedB b) aux nOf hd recRest) (cmtTr recRead)
      (DeployedR.layR OpenRows.elemR) (bdR DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR)
      (Rows.rowDPieces key) Rows.posAt :=
    CapLower.outBindRP_deployed (b := bB (deployedB b) aux nOf hd recRest) (elem := OpenRows.elemR)
      (IV := (bdR DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR).IVStar)
      (shape := Claims.Recursion.Form.Dense) (Bd := bdR DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR)
      (Rows.rowDPieces key) (fun _ => rfl) (fun _ h => h) (OpenLower.piecesInRangeR_rowDPieces key) (fun _ => le_rfl)
  refine contract_deployedR_layRVD DeployedR.rdR (deployedB b) aux nOf hd recRest OpenRows.elemR DeployedR.wER
    FlockDeploy.joinR DeployedR.finalR DeployedR.wR (fun x _ => FlockDeploy.KS.plainRdInR DeployedR.rdR x)
    (MoreSlots.moreSlotsTrue_deployed _) fun shape hs => ?_
  obtain rfl := List.mem_singleton.1 hs
  exact recKSDecompDV_of_leavesJD DeployedR.rdR (deployedB b) aux nOf hd recRest OpenRows.elemR DeployedR.wER
    DeployedR.finalR DeployedR.wR resPD key (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead ownRD
    (targetRDep_readers DeployedR.rdR (resPiecesAt_D DeployedR.rdR key _ _))
    (Top.rowEJoint_holds DeployedR.rdR key _ _ _ _ _ _)
    (basesHold_spec DeployedR.rdR) hF hRd (saltRowsJ_layR DeployedR.rdR DeployedR.wER DeployedR.wR OpenRows.elemR)
    (piecesOwned_D key)
    (resLowJPD_of_resLoweredPD DeployedR.rdR (deployedB b) aux nOf hd recRest DeployedR.wER DeployedR.finalR
      DeployedR.wR OpenRows.elemR _ _ _ _ hres)
    hO hDirs hOut (CapLower.sideCapR_leaf DeployedR.wER DeployedR.wR (deployedB b) aux nOf hd recRest key hRd)

end Deployed

end RecDeploy

namespace Claims.RecGeneric

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **The contract at the deployment's dense statements, as `RecoversLinkOn` reads it**: at the deployed bound record
and every compiled record `Flock.Guarantees.RecursiveKSEvery` reads, at every statement `RecDeploy.DenseAt` admits, the
calls of no witnesses are true, and at every inner and outer decoder the composite meets the compiler's contract at
the server's `Open(DE, hm96)` and the element-level Flock's openings (`RecDeploy.contract_deployed`,
`contract_ofBound`). It takes F1's `IVStarOfBound` and `OpenViewOfBound` at the deployed bound. -/
theorem contractDenseOn
    (hIV : IVStarOfBound (RecDeploy.bdDeployed _root_.Flock.Readable.relation
      (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR))
    (hOV : OpenViewOfBound (RecDeploy.bdDeployed _root_.Flock.Readable.relation
      (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR))
    (b : RecDeploy.Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    let D := RecDeploy.recBoundAt _root_.Flock.Readable.relation
      { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)
    let Bd := RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR
    ∀ x, RecDeploy.DenseAt x → (∀ e ∈ (D.record Bd.IVStar).calls x [], FlockDeploy.sha512 e.1 = e.2) ∧
      ∀ decI decO, (compAt D Bd decI (RecDeploy.deployZkRec _root_.Flock.Readable.relation) decO).Contract
        FlockDeploy.sha512 ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)
        (fun v => (Proto.ofFlock Bd decI FlockDeploy.KS.CpDep).opens ((Server.ofBound Bd).openView v)) x := by
  intro D Bd x hx
  have h := RecDeploy.contract_deployed b aux nOf hd recRest (fun _ _ _ _ => (0, 0, 0, 0)) x hx
  exact ⟨h.1, fun decI decO =>
    contract_ofBound D Bd decI (RecDeploy.deployZkRec _root_.Flock.Readable.relation) decO FlockDeploy.KS.CpDep
      hIV hOV (h.2 decI decO)⟩

/-- **F7 at the deployment** (`ContractDense`): `contractDenseOn`, whose contract holds at every decoder, so at those
meeting `decOK`. It takes F1's `IVStarOfBound` and `OpenViewOfBound` at the deployed bound. -/
theorem contractDense
    (hIV : IVStarOfBound (RecDeploy.bdDeployed _root_.Flock.Readable.relation
      (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR))
    (hOV : OpenViewOfBound (RecDeploy.bdDeployed _root_.Flock.Readable.relation
      (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR)) :
    ContractDense := fun b aux nOf hd recRest x hx =>
  ⟨(contractDenseOn hIV hOV b aux nOf hd recRest x hx).1,
    fun decI _ decO _ => (contractDenseOn hIV hOV b aux nOf hd recRest x hx).2 decI decO⟩

end Claims.RecGeneric
