import Definitions.Pouw.Guarantees.DitherInstance
import Proofs.Pouw.RelativeSecurity
import Proofs.Pouw.Dither
import Proofs.Core.Accel.Model

/-!
# The headline on the cards from the headline on the model

* `workSoundAtX_of_workSound`: `WorkSound` gives `WorkSoundAtX c pb hr x` on every card table at every open parameter:
  an instance of its quantifiers at the counted and timed verdict, which accepts no more than the service's verdict,
  with the exchange bound from the certifier's counts and `pub_mm`, `pub_sha` the card's public bests at `μ`. The
  sampling blocks' commit phases and work fit the period from the clocks: block `b`'s commit phase ends by its operand
  root's arrival, before its coins at `t b` (`CommitAfterWork`, `Timed`), and its work ends by its root's arrival, at
  most `Δ` after them (`RootAfterCoin`), so the two run in a window from `t b − wall(commit b)` to `t b + wall(work b)`.
  G2 bounds their use by `ρ_r` times that window, and block `b + 1`'s commit phase starts after block `b`'s work
  (`CommitAfterWork`), so the windows are disjoint inside `[0, T]` (`sum_windows_le_T`). A selected answer starts after
  its block's dither coins, inside the period, and arrives within `τ_ans` of its check coin, before the period's end
  (`AnswerAfterWork`, `Timed`), so it runs within the period (`AnswersFit`). No count of sampling blocks or of selected
  ones is read.
* `workSoundAt_of_workSound`, `workSoundOnCard_of_workSound`: at every `x` meeting (P1) and (P2), and at the `x`
  `CardWithinAccel` gives; `WorkSoundH200`, `WorkSoundSm120`, `WorkSoundOnCardH200` and `WorkSoundOnCardSm120` from
  `Proofs.Pouw.Dither.WorkSound`. None reads a premise beyond `WorkSound`'s and the statement's.
* `h200_semantics`, `sm120_semantics`: the exact `s8` semantics (`Verity.Accel.s8Open`) meets (P1) and (P2) on each
  card, so `WorkSoundAt`'s hypotheses on `x` are not empty.
* `h200_onCard`, `sm120_onCard`: at that semantics, on any period's devices, `cardAccel` (programs as runs within the
  devices' rates, no oracle calls, its call step at a public compression) is the devices under the identity match
  (`OnCard`), meets (G1), (G2) and (G3), and `CallsAtPub`, so the hypotheses on `G` are not empty either. `cardAccel` is
  a witness, not the canonical card accelerator, whose programs are exactly the card's: its programs are any runs within
  the rates, so `SAKMatmul` is false at it.
-/

namespace Pouw.SecurityProofs.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Guarantees.DitherInstance

open scoped NNRat

/-- **Windows in sequence telescope**: when each window ends no later than the next one starts, the windows' lengths
add up to at most the span from the first start to the last end. -/
theorem sum_windows_le : ∀ (n : ℕ) (a e : Fin (n + 1) → ℝ),
    (∀ b b' : Fin (n + 1), (b : ℕ) + 1 = b' → e b ≤ a b') → ∑ b, (e b - a b) ≤ e (Fin.last n) - a 0
  | 0, a, e, _ => by simp
  | n + 1, a, e, h => by
    rw [Fin.sum_univ_castSucc]
    have ih := sum_windows_le n (fun b => a b.castSucc) (fun b => e b.castSucc) fun b b' hb =>
      h _ _ (by simpa using hb)
    have hl := h (Fin.last n).castSucc (Fin.last (n + 1)) (by simp)
    simp only [Fin.castSucc_zero] at ih
    linarith

/-- **Windows in sequence inside `[0, T]` fit in `T`**: sampling block `b`'s window runs from `s b` to `e b ≤ T`, the
first starting at `s 0 ≥ 0`, and the next block's window starts only after it ends, so the windows' lengths add up
to at most `T`, at any number of blocks. -/
theorem sum_windows_le_T {n : ℕ} (s e : Fin n → ℝ) (T : ℝ) (hT : 0 ≤ T) (h0 : ∀ b : Fin n, (b : ℕ) = 0 → 0 ≤ s b)
    (he : ∀ b, e b ≤ T) (hs : ∀ b b' : Fin n, (b : ℕ) + 1 = b' → e b ≤ s b') : ∑ b, (e b - s b) ≤ T := by
  cases n with
  | zero => simpa using hT
  | succ n =>
    have h1 := sum_windows_le n s e hs
    have h2 := h0 0 rfl
    have h3 := he (Fin.last n)
    linarith

/-- `WorkSound` gives the headline on every card table at every open parameter. -/
theorem workSoundAtX_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) (c : Verity.Accel.Card)
    (pb : PubBest c) (hr : HostRow) (x : Verity.Accel.Open) : WorkSoundAtX c pb hr x := by
  intro D P L G e _ _ _ nsb Reg Obs verdict dev Pd μ Wmm Wsha Wans macs comps ηw δ nmax Wmac Ein Eout κS Δ Q
    κdmm κdsha Badv ξ κx κsha cin cout opArrive t arrive sel τans tc ansAt hdual hinst hH hη hδ hδ1 hn hWmac hEi hEo
    hT0 hξ hκd hcap g1 g2 g3 p1 p2 sak sha out svc sees commit root ans ext
  have hinst' : ∀ b, 0 < Wmm b ∧ 0 < Wsha b ∧ 0 ≤ Wans b ∧ pubCost e μ pb.mm (macs b) ≤ Wmm b ∧
      pubCost e μ pb.sha (comps b) ≤ Wsha b := fun b =>
    ⟨(hinst b).1, (hinst b).2.1, (hinst b).2.2.1, (hinst b).2.2.2.1, (hinst b).2.2.2.2.1⟩
  have svc' : ProofSystemSound G P nsb Reg
      (Timed (Counted verdict cin cout Ein Eout) Pd Δ opArrive t arrive sel τans tc ansAt) dev κS := fun A =>
    le_trans (prob_mono fun ox hox => ⟨hox.1.1.1, hox.2⟩) (svc A)
  have imp : ImportWithinCertifier G P nsb
      (Timed (Counted verdict cin cout Ein Eout) Pd Δ opArrive t arrive sel τans tc ansAt) dev (Ein + Eout) :=
    fun A o x hv => (sees A o x).trans (add_le_add hv.1.2.1 hv.1.2.2)
  -- A selected answer runs within the period: it starts after its block's work and arrives before the end.
  have hans : AnswersFit G P nsb (Timed (Counted verdict cin cout Ein Eout) Pd Δ opArrive t arrive sel τans tc ansAt)
      dev sel Pd := by
    intro A o x hv b hb r
    have h1 := (g2.2 (A.answer b) (A.answerIn o x b)).2 r
    have hwa : G.wall (A.answer b) (A.answerIn o x b) ≤ Pd.T := by
      have h2 := ans A o x b hb
      have h3 := (hv.2.1 b).1
      have h4 := hv.2.2.2.2.2 b hb
      linarith [h4.2.1, h4.2.2]
    show G.use (A.answer b) (A.answerIn o x b) r ≤ G.rate r * Pd.T
    exact h1.2.trans (mul_le_mul_of_nonneg_left hwa (g2.1 r).le)
  refine ⟨h G P nsb Reg Obs _ dev Pd μ Wmm Wsha Wans (fun b => pubCost e μ pb.mm (macs b))
    (fun b => pubCost e μ pb.sha (comps b)) ηw δ (nmax * Wmac) (Ein + Eout) κS Q κdmm κdsha sel Badv ξ κx κsha
    hdual hinst' hH hη hδ hδ1 (mul_nonneg hn hWmac) (add_nonneg hEi hEo) hξ hκd hcap g1 g2 g3 p1 p2 sak sha out svc'
    imp hans ext, ?_, hans⟩
  -- The commit phases and work run one block at a time: block `b`'s from `t b − wall(commit b)` to
  -- `t b + wall(work b)`, inside `[0, T]`.
  intro A o x hv r
  have hrate : 0 ≤ G.rate r := (g2.1 r).le
  have hw : ∀ b, G.use (A.commit b) (A.commitIn o x b) r + G.use (A.work b) (A.workIn o x b) r ≤
      G.rate r * ((t (dev A o x) b + G.wall (A.work b) (A.workIn o x b)) -
        (t (dev A o x) b - G.wall (A.commit b) (A.commitIn o x b))) := fun b => by
    have h1 := ((g2.2 (A.commit b) (A.commitIn o x b)).2 r).2
    have h2 := ((g2.2 (A.work b) (A.workIn o x b)).2 r).2
    rw [show t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) -
        (t (dev A o x) b - G.wall (A.commit b) (A.commitIn o x b)) =
        G.wall (A.commit b) (A.commitIn o x b) + G.wall (A.work b) (A.workIn o x b) by ring, mul_add]
    exact add_le_add h1 h2
  show ∑ b, (G.use (A.commit b) (A.commitIn o x b) r + G.use (A.work b) (A.workIn o x b) r) ≤ G.rate r * Pd.T
  refine (Finset.sum_le_sum fun b _ => hw b).trans ?_
  rw [← Finset.mul_sum]
  refine mul_le_mul_of_nonneg_left (sum_windows_le_T
    (fun b => t (dev A o x) b - G.wall (A.commit b) (A.commitIn o x b))
    (fun b => t (dev A o x) b + G.wall (A.work b) (A.workIn o x b)) Pd.T hT0 (fun b hb => ?_) (fun b => ?_)
    fun b b' hb => ?_) hrate
  · have h1 := (commit A o x).1 b hb
    have h2 := hv.2.2.1 b
    linarith
  · have h1 := root A o x b
    have h2 := hv.2.1 b
    linarith [h2.2.1, h2.2.2]
  · have h1 := (commit A o x).2 b b' hb
    have h2 := hv.2.2.1 b'
    linarith


/-- `WorkSound` gives the headline on every card table at every semantics meeting (P1) and (P2). -/
theorem workSoundAt_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) (c : Verity.Accel.Card)
    (pb : PubBest c) (hr : HostRow) : WorkSoundAt c pb hr :=
  fun x _ _ => workSoundAtX_of_workSound h c pb hr x

/-- `WorkSound` gives the headline on the card itself: `CardWithinAccel` gives the semantics, with the card's runs as
its model's there, and the headline holds at them. -/
theorem workSoundOnCard_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) (c : Verity.Accel.Card)
    (pb : PubBest c) (hr : HostRow) : WorkSoundOnCard c pb hr := by
  intro can hcw
  obtain ⟨x, hf, hp, hrun⟩ := hcw
  exact ⟨x, hf, hp, hrun, workSoundAtX_of_workSound h c pb hr x⟩

theorem workSoundH200_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) : WorkSoundH200 :=
  workSoundAt_of_workSound h _ _ _

theorem workSoundSm120_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) : WorkSoundSm120 :=
  workSoundAt_of_workSound h _ _ _

/-- **The headline on the H200 at θ**, from `WorkSound`'s proof. -/
theorem WorkSoundH200 : Pouw.Guarantees.DitherInstance.WorkSoundH200 :=
  workSoundH200_of_workSound Pouw.SecurityProofs.Dither.WorkSound

/-- **The headline on the RTX PRO 6000 at θ**, from `WorkSound`'s proof. -/
theorem WorkSoundSm120 : Pouw.Guarantees.DitherInstance.WorkSoundSm120 :=
  workSoundSm120_of_workSound Pouw.SecurityProofs.Dither.WorkSound

/-- **The headline on the H200 itself**, from `WorkSound`'s proof, under `card-within-accel(H200)`. -/
theorem WorkSoundOnCardH200 : Pouw.Guarantees.DitherInstance.WorkSoundOnCardH200 :=
  workSoundOnCard_of_workSound Pouw.SecurityProofs.Dither.WorkSound _ _ _

/-- **The headline on the RTX PRO 6000 itself**, from `WorkSound`'s proof, under
`card-within-accel(RTX PRO 6000)`. -/
theorem WorkSoundOnCardSm120 : Pouw.Guarantees.DitherInstance.WorkSoundOnCardSm120 :=
  workSoundOnCard_of_workSound Pouw.SecurityProofs.Dither.WorkSound _ _ _

/-- The H200's (P1) and (P2) are met, by the exact `s8` semantics. -/
theorem h200_semantics :
    ∃ x, Verity.Accel.H200.card.Exact x ∧ Verity.Assumptions.AccelProductForm x :=
  ⟨Verity.Accel.s8Open, Verity.Accel.s8Open_exact _, Verity.Accel.s8Open_productForm⟩

/-- The RTX PRO 6000's (P1) and (P2) are met, by the exact `s8` semantics. -/
theorem sm120_semantics :
    ∃ x, Verity.Accel.Sm120.card.Exact x ∧ Verity.Assumptions.AccelProductForm x :=
  ⟨Verity.Accel.s8Open, Verity.Accel.s8Open_exact _, Verity.Accel.s8Open_productForm⟩

section Witness

/-- Programs as runs with a use and a wall time, each resource used at most at its rate over the run's wall time;
composing runs adds their uses and wall times. -/
noncomputable def boundedMc {R : Type} (rate : R → ℝ) : Machine where
  Alg I O := {a : (I → O) × (I → R → ℝ) × (I → ℝ) // ∀ i r, 0 ≤ a.2.1 i r ∧ a.2.1 i r ≤ rate r * a.2.2 i}
  eval f := f.1.1
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := ⟨(fun i => g.1.1 (f.1.1 i), fun i r => f.1.2.1 i r + g.1.2.1 (f.1.1 i) r,
      fun i => f.1.2.2 i + g.1.2.2 (f.1.1 i)), fun i r =>
    ⟨add_nonneg (f.2 i r).1 (g.2 _ r).1, by rw [mul_add]; exact add_le_add (f.2 i r).2 (g.2 _ r).2⟩⟩
  pair f g := ⟨(fun i => (f.1.1 i, g.1.1 i), fun i r => f.1.2.1 i r + g.1.2.1 i r, fun i => f.1.2.2 i + g.1.2.2 i),
    fun i r => ⟨add_nonneg (f.2 i r).1 (g.2 i r).1, by rw [mul_add]; exact add_le_add (f.2 i r).2 (g.2 i r).2⟩⟩
  fst := ⟨(fun p => p.1, fun _ _ => 0, fun _ => 0), fun _ _ => ⟨le_rfl, by rw [mul_zero]⟩⟩
  snd := ⟨(fun p => p.2, fun _ _ => 0, fun _ => 0), fun _ _ => ⟨le_rfl, by rw [mul_zero]⟩⟩
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

/-- The period's devices' rates: the pooled cards' a second, and the other devices'. -/
noncomputable def devRate {c : Verity.Accel.Card} (D : Devices c) : Verity.Accel.Res ⊕ D.oth.Res → ℝ
  | .inl r => ((poolCard c D.n D.n_pos).ρ r : ℝ)
  | .inr h => D.oth.rate h

/-- An accelerator on the period's devices at `x`: `boundedMc` at the devices' rates, the pooled cards' memory and the
other devices', nothing exchanged, no oracle calls, a call step at the card units `u`, and the machine's (P1) and
(P2) at `x`. -/
noncomputable def cardAccel (c : Verity.Accel.Card) (x : Verity.Accel.Open) (D : Devices c) (u : ResUnits) :
    Accel where
  Mc := boundedMc (devRate D)
  Res := Verity.Accel.Res ⊕ D.oth.Res
  rate := devRate D
  mem := ((D.machine x).bits : ℝ) + D.oth.mem
  use f i r := f.1.2.1 i r
  held _ _ := 0
  wall f i := f.1.2.2 i
  ext _ _ := 0
  calls _ _ := 0
  callUse := Sum.elim (fun r => (u r : ℝ)) fun _ => 0
  P1 := c.Exact x
  P2 := Verity.Assumptions.AccelProductForm x

open Classical in
/-- **`OnCard` is met, with G1, G2 and G3**: at positive card rates, positive other rates and their memory,
`cardAccel` is the period's devices at `x` under the identity match, meeting (G1), (G2) and (G3). -/
theorem cardAccel_onCard (c : Verity.Accel.Card) (x : Verity.Accel.Open) (D : Devices c) (u : ResUnits)
    (P : RoundProto) (L : RoundPlan (D.machine x) P) (hρ : ∀ r, 0 < c.ρ r)
    (hoth : ∀ h, 0 < D.oth.rate h) (hmem : 0 ≤ D.oth.mem) :
    OnCard (cardAccel c x D u) c x D P L (Equiv.refl _) ∧ (cardAccel c x D u).G1 ∧ (cardAccel c x D u).G2 ∧
      (cardAccel c x D u).G3 := by
  have hrate : ∀ r, 0 < devRate D r := by
    rintro (r | h)
    · exact NNRat.cast_pos.mpr (mul_pos (Nat.cast_pos.mpr D.n_pos) (hρ r))
    · exact hoth h
  refine ⟨⟨fun _ => rfl, fun _ => rfl, rfl, Iff.rfl, Iff.rfl, ?_⟩, ?_, ⟨hrate, fun f i => ⟨le_rfl, f.2 i⟩⟩,
    fun f i => ⟨le_rfl, fun r => ?_⟩⟩
  · intro M₀ Prg sched T hfits
    let m := D.machine x
    let wl : m.Mem × P.Draw → ℝ := fun p => (T p.2 : ℝ)
    let uw : m.Mem × P.Draw → Verity.Accel.Res ⊕ D.oth.Res → ℝ := fun p q =>
      match q with
      | .inl r => if p.1 = M₀ then (cardUnits m (cardRun m Prg (L.putDraw p.2 p.1) sched).2 r : ℝ) else 0
      | .inr _ => 0
    have hu : ∀ p q, 0 ≤ uw p q ∧ uw p q ≤ devRate D q * wl p := by
      have hwl : ∀ p, 0 ≤ wl p := fun p => NNRat.cast_nonneg _
      rintro ⟨M, d⟩ (r | h)
      · by_cases hM : M = M₀
        · subst hM
          have hle : cardUnits m (cardRun m Prg (L.putDraw d M) sched).2 r ≤ (poolCard c D.n D.n_pos).ρ r * T d :=
            Verity.Accel.Machine.units_le_of_fits Prg _ sched (T d) (hfits d) r
          simp only [uw, ite_true]
          refine ⟨NNRat.cast_nonneg _, ?_⟩
          show _ ≤ ((poolCard c D.n D.n_pos).ρ r : ℝ) * (T d : ℝ)
          exact_mod_cast hle
        · simp only [uw, hM, ite_false]
          exact ⟨le_rfl, mul_nonneg (hrate _).le (hwl _)⟩
      · exact ⟨le_rfl, mul_nonneg (hrate _).le (hwl _)⟩
    refine ⟨⟨(fun p => L.getTr (cardRun m Prg (L.putDraw p.2 p.1) sched).1.mem, uw, wl), hu⟩, fun d =>
      ⟨rfl, fun r => ?_, fun _ => rfl, rfl, rfl, rfl⟩⟩
    show uw (M₀, d) (.inl r) = _
    simp only [uw, ite_true]
    rfl
  · intro I O f i
    show (0 : ℝ) + 0 ≤ _
    rw [add_zero]
    exact add_nonneg (Nat.cast_nonneg _) hmem
  · show (0 : ℝ) * _ ≤ f.1.2.1 i r
    rw [zero_mul]
    exact (f.2 i r).1

theorem h200_rates_pos : ∀ r, 0 < Verity.Accel.H200.card.ρ r := by
  intro r; cases r <;> norm_num [Verity.Accel.H200.card, Verity.Accel.H200.clock]

theorem sm120_rates_pos : ∀ r, 0 < Verity.Accel.Sm120.card.ρ r := by
  intro r; cases r <;> norm_num [Verity.Accel.Sm120.card, Verity.Accel.Sm120.clock]

/-- **An oracle call at a public compression**: with its call step at the units of one of `pb`'s SHA-512
schedules, `cardAccel` meets `CallsAtPub` under the identity match. -/
theorem cardAccel_callsAtPub {c : Verity.Accel.Card} (x : Verity.Accel.Open) (D : Devices c) (pb : PubBest c)
    {u : ResUnits} (hu : u ∈ pb.sha) : CallsAtPub (cardAccel c x D u) (Equiv.refl _) pb :=
  ⟨u, hu, fun _ => rfl, fun _ => rfl⟩

/-- **`WorkSoundH200`'s hypotheses on the card are met**: at the exact `s8` semantics (P1, P2), on any
period's devices with positive other rates, some accelerator is the devices under some match, meets (G1), (G2) and
(G3), and has its oracle calls at a public compression. -/
theorem h200_onCard (D : Devices Verity.Accel.H200.card) (P : RoundProto)
    (L : RoundPlan (D.machine Verity.Accel.s8Open) P) (hoth : ∀ h, 0 < D.oth.rate h) (hmem : 0 ≤ D.oth.mem) :
    ∃ (G : Accel) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res),
      OnCard G Verity.Accel.H200.card Verity.Accel.s8Open D P L e ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧
        CallsAtPub G e pubBestH200 := by
  obtain ⟨h1, h2, h3, h4⟩ :=
    cardAccel_onCard _ _ D (shaUnits 3400 152 (3608 / 32)) P L h200_rates_pos hoth hmem
  exact ⟨_, _, h1, h2, h3, h4, cardAccel_callsAtPub Verity.Accel.s8Open D pubBestH200 (List.Mem.head _)⟩

/-- **`WorkSoundSm120`'s hypotheses on the card are met**, as on the H200. -/
theorem sm120_onCard (D : Devices Verity.Accel.Sm120.card) (P : RoundProto)
    (L : RoundPlan (D.machine Verity.Accel.s8Open) P) (hoth : ∀ h, 0 < D.oth.rate h) (hmem : 0 ≤ D.oth.mem) :
    ∃ (G : Accel) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res),
      OnCard G Verity.Accel.Sm120.card Verity.Accel.s8Open D P L e ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧
        CallsAtPub G e pubBestSm120 := by
  obtain ⟨h1, h2, h3, h4⟩ :=
    cardAccel_onCard _ _ D (shaUnits 3415 0 (3415 / 32)) P L sm120_rates_pos hoth hmem
  exact ⟨_, _, h1, h2, h3, h4, cardAccel_callsAtPub Verity.Accel.s8Open D pubBestSm120 (List.Mem.head _)⟩

/-- **The hosts are counted at devices whose other devices are the row's hosts**: with `h200_onCard` and
`sm120_onCard` at such devices, `OnCard` and `HostsCounted` hold together. -/
theorem hostsCounted_others {c : Verity.Accel.Card} (h : HostRow) (n : ℕ) (hn : 0 < n) (mem : ℝ) :
    HostsCounted (⟨n, hn, h.others n mem⟩ : Devices c) h :=
  ⟨id, Function.injective_id, le_rfl, le_rfl⟩

theorem hostH200_pos : 0 < hostH200.amx ∧ 0 < hostH200.sha := by
  constructor <;> norm_num [hostH200]

theorem hostSm120_pos : 0 < hostSm120.amx ∧ 0 < hostSm120.sha := by
  constructor <;> norm_num [hostSm120]

end Witness

end Pouw.SecurityProofs.DitherInstance
