module

public import Proofs.Flock.RecGeneric.Plurality
public import Proofs.Flock.Claims.Recursion.KSLin2

/-!
# `P_O`'s aborting runs, as the composite's reruns

The inner protocol's terms bound `P_O`'s aborting runs in the inner protocol's own model of runs (`runModel` at
`Comp.innerPO` and `Comp.keepAO`); the composite's bound reads the same finders on its reruns (`Comp.innerModel`). A run
of the first is a rerun of the second through `extendDraw`, and the two models agree on it but where the rerun is bad:

* a run's coins are its rerun's inner coins (`Comp.coins_extend`), and a fork at an inner round is its rerun's fork
  there (`Comp.named_extend`, `extend_fork_min`);
* `P_O`'s run accepts exactly where it is kept (`Comp.innerFO_acc_iff`), and it is kept exactly where its rerun's inner
  view accepts, but where the rerun accepts at a sample `P_O`'s runs don't show (`Comp.BadO`) or at one where `E_O` gives
  witnesses outside `R_O` (`Comp.OutRO`) (`Comp.accAt_iff`);
* a run's calls are among its rerun's searched evaluations (`Comp.rerunCalls`), or those collide (`CallsLogColl`), so
  a collision among runs' calls is one among their reruns' (`Comp.findColl_cover`).

A rerun where `E_O` gives witnesses outside `R_O` is `E_O`'s failure on an accepted run (`Comp.PrOut_le`).

Port of `Claims.Recursion.Bound.accAt_iff`, `Bound.innerFO_acc_iff`, `Bound.keepDO_iff_accAtI`,
`Bound.innerFO_vLog_subBy` and `Bound.findColl_cover` to any compiled verifier, outer extraction and server.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- An inner run's coins forked at `j` onto another's, extended, are the extended samples forked at `j` capped at the
inner rounds. -/
theorem extendDraw_fork {Coin C : Type} [Inhabited Coin] {n N : ℕ} (j : ℕ)
    (a b : (Fin n → Coin) × (Fin N → Coin) × C) :
    extendDraw ((fork j a.1 b.1, b.2) : (Fin n → Coin) × (Fin N → Coin) × C) =
      (fork (min j n) (extendDraw a).1 (extendDraw b).1, (extendDraw b).2) :=
  Prod.ext (extend_fork_min j a.1 b.1 a.2.1 b.2.1) rfl

namespace Comp

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {K : Comp Coin} {I : Proto Coin W}
  {S : Server} {H : Msg → Msg} {Ω : Type} {P : Prover Ω} {x : Msg}

/-! ## Acceptance -/

omit [Nonempty Coin] in
/-- **`P_O`'s run at a sample accepts exactly where it is kept.** -/
theorem innerFO_acc_iff (hCF : I.LastCoinFree H) (decC : Msg → Coin) (hdec : ∀ c, decC (I.ans c) = c) (ω : Ω)
    (d : (Fin (K.V.k x) → Coin) × K.EO.C x) :
    (K.innerFO I S H P x decC ω d).out = .acc ↔ K.keepDO I S H P x ω d = true := by
  constructor
  · intro h
    by_contra hk
    have e : K.innerFO I S H P x decC ω d = abortTranscript := by
      unfold Comp.innerFO
      rw [ite_eq_right hk]
    rw [e] at h
    simp [abortTranscript] at h
  · intro hk
    obtain ⟨⟨w, hw, hRO⟩, hacc, ha⟩ : K.GivesRO H P x ω d ∧
        (K.dec (S.IVStar I.V I.ans) H x d.1 (K.wits H P x ω d)).out = .acc ∧ K.AgreesO I S H P x ω d := by
      simpa only [Comp.keepDO, decide_eq_true_eq] using hk
    have hacc' : (K.view (S.IVStar I.V I.ans) H x d.1 w).out = .acc := by
      rw [hw] at hacc
      exact hacc
    unfold Comp.innerFO
    rw [ite_eq_left hk]
    exact (runO hCF decC hdec ω d hw hRO hacc' ha).2.1

omit [Nonempty Coin] in
/-- A kept sample is not bad. -/
theorem not_badO_of_keepDO {ω : Ω} {d : (Fin (K.V.k x) → Coin) × K.EO.C x}
    (h : K.keepDO I S H P x ω d = true) : ¬ K.BadO I S H P x ω d := fun hb => by
  simp only [Comp.keepDO, decide_eq_true_eq] at h
  exact hb.2 h.2.2

omit [Nonempty Coin] in
/-- **Off the bad samples and those where `E_O` gives witnesses outside `R_O`, a sample is kept exactly where its inner
view accepts.** -/
theorem keepDO_iff_acc {ω : Ω} {d : (Fin (K.V.k x) → Coin) × K.EO.C x} (hb : ¬ K.BadO I S H P x ω d)
    (ho : ¬ K.OutRO H P x ω d) :
    K.keepDO I S H P x ω d = true ↔ (K.innerModel (S.IVStar I.V I.ans) H P x).acc ω d := by
  simp only [Comp.keepDO, decide_eq_true_eq]
  change _ ↔ (K.dec (S.IVStar I.V I.ans) H x d.1 (K.wits H P x ω d)).out = .acc
  constructor
  · exact fun h => h.2.1
  · intro h
    obtain ⟨w, hw⟩ : ∃ w, K.wits H P x ω d = some w := by
      rcases hw : K.wits H P x ω d with _ | w
      · rw [hw] at h
        simp [Comp.dec] at h
      · exact ⟨w, rfl⟩
    have hg : K.GivesRO H P x ω d := ⟨w, hw, Classical.byContradiction fun hn => ho ⟨w, hw, hn⟩⟩
    exact ⟨hg, h, Classical.byContradiction fun hn => hb ⟨hg, hn⟩⟩

omit [Nonempty Coin] in
/-- **`P_O`'s aborting run accepts exactly where its rerun's inner view does**, but where the rerun accepts at a bad
sample or at one where `E_O` gives witnesses outside `R_O`: a kept run's rerun accepts, and off those samples an accepting
rerun is kept. -/
theorem accAt_iff (hCF : I.LastCoinFree H) (decC : Msg → Coin) (hdec : ∀ c, decC (I.ans c) = c)
    (hn : I.V.k (K.xI x) ≤ K.V.k x) (ω : Ω)
    (t : (Fin (I.V.k (K.xI x)) → Coin) × (Fin (K.V.k x) → Coin) × K.EO.C x)
    (h : ¬ ((K.BadO I S H P x ω (extendDraw t) ∨ K.OutRO H P x ω (extendDraw t)) ∧
      (K.innerModel (S.IVStar I.V I.ans) H P x).acc ω (extendDraw t))) :
    (runModel H I.V (K.innerPO I S H P x decC) (K.xI x) (K.keepAO I S H P x)).acc ω t ↔
      (K.innerModel (S.IVStar I.V I.ans) H P x).acc ω (extendDraw t) := by
  show (abortRun H (K.innerPO I S H P x decC) I.V (K.xI x) (K.keepAO I S H P x) ω t).out = .acc ↔ _
  rw [abortRun_eqO decC hn, innerFO_acc_iff hCF decC hdec]
  refine ⟨fun hk => ?_, fun ha => (keepDO_iff_acc (fun hb => h ⟨Or.inl hb, ha⟩) (fun ho => h ⟨Or.inr ho, ha⟩)).2 ha⟩
  simp only [Comp.keepDO, decide_eq_true_eq] at hk
  exact hk.2.1

/-! ## Coins and forks -/

omit [Nonempty Coin] in
/-- **A run's coins are its rerun's inner coins.** -/
theorem coins_extend (decC : Msg → Coin) (hn : I.V.k (K.xI x) ≤ K.V.k x)
    (t : (Fin (I.V.k (K.xI x)) → Coin) × (Fin (K.V.k x) → Coin) × K.EO.C x) :
    (K.innerModel (S.IVStar I.V I.ans) H P x).coins (extendDraw t) =
      (runModel H I.V (K.innerPO I S H P x decC) (K.xI x) (K.keepAO I S H P x)).coins t := by
  show coinSeq (firstCoins (I.V.k (K.xI x)) (extend t.1 t.2.1)) = coinSeq t.1
  rw [firstCoins_extend hn]

omit [Nonempty Coin] in
/-- **A named run's rerun is the rerun named alike**: forks at an inner round commute with extending runs to
reruns. -/
theorem named_extend (decC : Msg → Coin) (hn : I.V.k (K.xI x) ≤ K.V.k x)
    (us : ℕ → (Fin (I.V.k (K.xI x)) → Coin) × (Fin (K.V.k x) → Coin) × K.EO.C x) :
    ∀ e : RunExp,
      extendDraw ((runModel H I.V (K.innerPO I S H P x decC) (K.xI x) (K.keepAO I S H P x)).named us e) =
        (K.innerModel (S.IVStar I.V I.ans) H P x).named (fun i => extendDraw (us i)) e
  | .fresh _ => rfl
  | .fork j u v => by
    simp only [RunModel.named]
    rw [← named_extend decC hn us u, ← named_extend decC hn us v]
    exact extendDraw_fork j _ _

/-! ## Calls -/

omit [Fintype Coin] [Nonempty Coin] in
/-- `E_O`'s witnesses on a run it replays record true evaluations (`CallsTrue`), so a rerun's searched evaluations are
true. -/
theorem rerunCalls_true (hRp : K.Replays H x) (hT : K.CallsTrue H x) (ω : Ω)
    (d : (Fin (K.V.k x) → Coin) × K.EO.C x) : ∀ e ∈ K.rerunCalls H P x ω d, H e.1 = e.2 := by
  intro e he
  unfold Comp.rerunCalls at he
  rcases hw : K.wits H P x ω d with _ | w
  · rw [hw] at he
    simp at he
  · rw [hw] at he
    have hw' : (K.EO.eval H P x ω d.1 d.2).1 = some w := hw
    exact hT Ω P ω d.1 d.2 w ⟨hRp Ω P ω d.1 d.2 w hw', hw'⟩ e he

omit [Nonempty Coin] in
/-- **`P_O`'s aborting run records only calls its rerun's searched evaluations hold, or those collide**: a kept run's
calls are its inner view's (`runO`), which are among `E_O`'s witnesses' there, in `R_O`, or collide there
(`CallsLogColl`); an aborted run records none. -/
theorem innerFO_vLog_sub (hCF : I.LastCoinFree H) (decC : Msg → Coin) (hdec : ∀ c, decC (I.ans c) = c)
    (hRp : K.Replays H x) (hL : K.CallsLogColl H (S.IVStar I.V I.ans) x) (ω : Ω)
    (d : (Fin (K.V.k x) → Coin) × K.EO.C x) :
    (∀ e ∈ (K.innerFO I S H P x decC ω d).vLog, e ∈ K.rerunCalls H P x ω d) ∨
      ∃ u v, findColl (K.rerunCalls H P x ω d) = some (u, v) ∧ Collision H u v := by
  by_cases hk : K.keepDO I S H P x ω d = true
  · obtain ⟨⟨w, hw, hRO⟩, hacc, ha⟩ : K.GivesRO H P x ω d ∧
        (K.dec (S.IVStar I.V I.ans) H x d.1 (K.wits H P x ω d)).out = .acc ∧ K.AgreesO I S H P x ω d := by
      simpa only [Comp.keepDO, decide_eq_true_eq] using hk
    have hacc' : (K.view (S.IVStar I.V I.ans) H x d.1 w).out = .acc := by
      rw [hw] at hacc
      exact hacc
    have hrun := runO hCF decC hdec ω d hw hRO hacc' ha
    have hF : K.innerFO I S H P x decC ω d =
        run H (K.innerPO I S H P x decC) I.V (K.xI x) ω (firstCoins (I.V.k (K.xI x)) d.1) := by
      unfold Comp.innerFO
      rw [ite_eq_left hk]
    have hrc : K.rerunCalls H P x ω d = K.calls H x d.1 w := by
      unfold Comp.rerunCalls
      rw [hw]
      rfl
    have hw' : (K.EO.eval H P x ω d.1 d.2).1 = some w := hw
    rw [hF, hrc]
    rcases hL Ω P ω d.1 d.2 w ⟨hRp Ω P ω d.1 d.2 w hw', hw'⟩ hRO with h | h
    · exact Or.inl fun e he => h e (hrun.2.2 e he)
    · exact Or.inr h
  · refine Or.inl fun e he => ?_
    have e0 : K.innerFO I S H P x decC ω d = abortTranscript := by
      unfold Comp.innerFO
      rw [ite_eq_right hk]
    rw [e0] at he
    simp [abortTranscript] at he

omit [Nonempty Coin] in
/-- **A collision among `P_O`'s aborting runs' calls is one among their reruns' searched evaluations**, where every
run's call is among some rerun's searched evaluations that the true list `L'` holds: each run's calls are among its
rerun's, or those collide (`innerFO_vLog_sub`). -/
theorem findColl_cover (hCF : I.LastCoinFree H) (decC : Msg → Coin) (hdec : ∀ c, decC (I.ans c) = c)
    (hRp : K.Replays H x) (hL : K.CallsLogColl H (S.IVStar I.V I.ans) x) (ω : Ω) {L L' : List (Msg × Msg)}
    (ht : ∀ p ∈ L', H p.1 = p.2)
    (hc : ∀ e ∈ L, ∃ d, e ∈ (K.innerFO I S H P x decC ω d).vLog ∧ ∀ p ∈ K.rerunCalls H P x ω d, p ∈ L')
    (h : (findColl L).isSome) : ∃ u v, findColl L' = some (u, v) ∧ Collision H u v := by
  have hs : (findColl L').isSome := by
    by_contra hn
    refine hn (findColl_isSome_of_sub H ht (fun e he => ?_) h)
    obtain ⟨d, hed, hsub⟩ := hc e he
    rcases innerFO_vLog_sub hCF decC hdec hRp hL ω d with h1 | ⟨u, v, huv, -⟩
    · exact hsub e (h1 e hed)
    · exact absurd (findColl_isSome_of_sub H ht hsub (Option.isSome_iff_exists.2 ⟨_, huv⟩)) hn
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hs
  exact ⟨u, v, huv, findColl_spec H _ ht u v huv⟩

/-! ## Witnesses outside `R_O` -/

omit [Nonempty Coin] in
/-- **`E_O` gives witnesses outside `R_O` at most as often as it fails**, where it gives witnesses only on runs it
replays (`Replays`). -/
theorem PrOut_le_PrBad (hRp : K.Replays H x) [Fintype Ω] (P : Prover Ω) :
    K.PrOut H P x ≤ PrBad H K.V K.RO K.EO x P :=
  Pr_mono fun t ⟨w, hw, hn⟩ => by
    have hw' : (K.EO.eval H P x t.1 t.2.1 t.2.2).1 = some w := hw
    refine ⟨hRp Ω P t.1 t.2.1 t.2.2 w hw', fun h => hn ?_⟩
    rw [hw'] at h
    exact h

/-- **`Pr_out ≤ κ_O + √(γ_O·Pr[B])`**, where `E_O` replays the judged run and is knowledge sound at `x`. -/
theorem PrOut_le (hRp : K.Replays H x) (hO : K.OuterKSAt H x) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    K.PrOut H P x ≤ K.κO x + Real.sqrt (K.γO x * PrB H K.V P x) :=
  (PrOut_le_PrBad hRp P).trans (hO Ω P)

omit [Nonempty Coin] in
/-- **`Pr_out = 0` where `R_O` holds at every witness**, as at Flock's (`Comp.ofRec`). -/
theorem PrOut_eq_zero (h : ∀ w, K.RO x w) [Fintype Ω] (P : Prover Ω) : K.PrOut H P x = 0 :=
  le_antisymm ((Pr_mono fun _ ⟨w, _, hn⟩ => hn (h w)).trans (le_of_eq Pr_false)) (Pr_nonneg _)

end Comp

end Claims.RecGeneric
