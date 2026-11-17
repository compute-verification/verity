module

public import Proofs.Pouw.DitherInstance
public import Proofs.Pouw.PoolWitness

/-! Honest and nontrivial witnesses of `WorkSoundAtTables`, at `unitCard`, a table that models no hardware, of
which they read only that its rates are positive. -/

@[expose] public section

namespace Pouw.SecurityProofs.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Guarantees.DitherInstance Pouw.Witness

open scoped NNRat

section HalfPd

noncomputable def halfPd (G : Accel) (μ : G.Res → ℝ) : Period G where
  T := 1
  J := Unit
  val _ := ∑ r, μ r * G.rate r
  needs _ r := G.rate r
  hon r := G.rate r / 2
  rv := ∑ r, μ r * (G.rate r / 2)

theorem halfPd_dualOpt (G : Accel) (hρ : ∀ r, 0 < G.rate r) (μ : G.Res → ℝ) (hμ : ∀ r, 0 ≤ μ r) :
    (halfPd G μ).DualOpt μ 1 := by
  refine ⟨hμ, fun _ r => (hρ r).le, fun _ => le_rfl, fun _ => 1 / 2, ⟨fun _ => by norm_num, fun r => ?_⟩, ?_⟩
  · show 1 * (G.rate r / 2) + ∑ _j : Unit, (1 / 2 : ℝ) * G.rate r ≤ G.rate r * 1
    simp only [Fintype.sum_unique]
    linarith
  · show ∑ _j : Unit, (∑ r, μ r * G.rate r) * (1 / 2 : ℝ) = ∑ r, μ r * (G.rate r * 1 - 1 * (G.rate r / 2))
    simp only [Fintype.sum_unique]
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun r _ => by ring

theorem halfPd_idle_fits (G : Accel) (μ : G.Res → ℝ) : (halfPd G μ).Fits (fun _ => 0) fun _ => 1 := by
  refine ⟨fun _ => zero_le_one, fun r => ?_⟩
  show (0 : ℝ) + ∑ _j : Unit, (1 : ℝ) * G.rate r ≤ G.rate r * 1
  simp only [Fintype.sum_unique]
  linarith

theorem halfPd_not_freed (G : Accel) (r₀ : G.Res) (hρ : ∀ r, 0 < G.rate r) :
    ¬ (halfPd G fun _ => 1).FreedWithin 1 (fun _ => 1) (1 / 2)
      (∑ r, (fun _ => (1 : ℝ)) r * (1 * (halfPd G fun _ => 1).hon r)) := by
  change ¬ ∃ (W' : ℝ) (z' : Unit → ℝ), 0 ≤ W' ∧
    ((∀ j, 0 ≤ z' j) ∧ ∀ r, W' * (G.rate r / 2) + ∑ j, z' j * G.rate r ≤ G.rate r * 1) ∧
    1 * (∑ r, (1 : ℝ) * (G.rate r / 2)) + ∑ _j : Unit, (∑ r, (1 : ℝ) * G.rate r) * 1 ≤
      W' * (∑ r, (1 : ℝ) * (G.rate r / 2)) + ∑ j, (∑ r, (1 : ℝ) * G.rate r) * z' j +
        1 / 2 * ∑ r, (1 : ℝ) * (1 * (G.rate r / 2))
  rintro ⟨W', z', -, ⟨-, hfit⟩, hval⟩
  have hV : 0 < ∑ r, G.rate r := Finset.sum_pos (fun r _ => hρ r) ⟨r₀, Finset.mem_univ _⟩
  have hfr := hfit r₀
  simp only [Fintype.sum_unique] at hfr hval
  have hz : W' / 2 + z' default ≤ 1 := by
    have h := hρ r₀
    by_contra hc
    have hc' := not_le.mp hc
    nlinarith
  have e1 : ∑ r, (1 : ℝ) * (G.rate r / 2) = (∑ r, G.rate r) / 2 := by
    rw [Finset.sum_div]; exact Finset.sum_congr rfl fun r _ => by ring
  have e2 : ∑ r, (1 : ℝ) * G.rate r = ∑ r, G.rate r := Finset.sum_congr rfl fun r _ => by ring
  have e3 : ∑ r, (1 : ℝ) * (1 * (G.rate r / 2)) = (∑ r, G.rate r) / 2 := by
    rw [Finset.sum_div]; exact Finset.sum_congr rfl fun r _ => by ring
  rw [e1, e2, e3] at hval
  nlinarith

end HalfPd

section Instance

noncomputable abbrev devs (c : Verity.Accel.Card) (hr : HostRow) : Devices c := ⟨1, Nat.one_pos, hr.others 1 0⟩

noncomputable abbrev accelAt (c : Verity.Accel.Card) (x : Verity.Accel.Open) (hr : HostRow) (u : ResUnits) : Accel :=
  cardAccel c x (devs c hr) u

noncomputable def idLayout (c : Verity.Accel.Card) (x : Verity.Accel.Open) (hr : HostRow) :
    RoundPlan ((devs c hr).machine x) twRound :=
  ⟨fun _ M => M, fun _ => fun _ _ _ => 0⟩

theorem devs_oth_pos (c : Verity.Accel.Card) (hr : HostRow) (hh : 0 < hr.amx ∧ 0 < hr.sha) :
    ∀ h, 0 < (devs c hr).oth.rate h := by
  intro h
  cases h
  · show (0 : ℝ) < ((1 : ℕ) : ℝ) * hr.amx
    rw [Nat.cast_one, one_mul]
    exact hh.1
  · show (0 : ℝ) < ((1 : ℕ) : ℝ) * hr.sha
    rw [Nat.cast_one, one_mul]
    exact hh.2

theorem devs_mem (c : Verity.Accel.Card) (hr : HostRow) : 0 ≤ (devs c hr).oth.mem := le_refl (0 : ℝ)

theorem honest_at (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow) (hh : 0 < hr.amx ∧ 0 < hr.sha)
    (pb : PubBest c) (u : ResUnits) (hu : u ∈ pb.sha) :
    c.Exact Verity.Accel.s8Open ∧ Verity.Assumptions.AccelProductForm Verity.Accel.s8Open ∧
      OnCard (accelAt c Verity.Accel.s8Open hr u) c Verity.Accel.s8Open (devs c hr) twRound
        (idLayout c Verity.Accel.s8Open hr) (Equiv.refl _) ∧
      HostsCounted (devs c hr) hr ∧
      CallsAtPub (D := devs c hr) (accelAt c Verity.Accel.s8Open hr u) (Equiv.refl _) pb ∧
      (accelAt c Verity.Accel.s8Open hr u).G1 ∧ (accelAt c Verity.Accel.s8Open hr u).G2 ∧
      (accelAt c Verity.Accel.s8Open hr u).G3 ∧
      (halfPd (accelAt c Verity.Accel.s8Open hr u) fun _ => 1).DualOpt (fun _ => 1) 1 := by
  obtain ⟨h1, h2, h3, h4⟩ := cardAccel_onCard c Verity.Accel.s8Open (devs c hr) u twRound
    (idLayout c Verity.Accel.s8Open hr) hρ (devs_oth_pos c hr hh) (devs_mem c hr)
  exact ⟨Verity.Accel.s8Open_exact c, Verity.Accel.s8Open_productForm, h1, hostsCounted_others hr 1 Nat.one_pos 0,
    cardAccel_callsAtPub Verity.Accel.s8Open (devs c hr) pb hu, h2, h3, h4,
    halfPd_dualOpt _ h3.1 _ fun _ => zero_le_one⟩

noncomputable def idleAlg {R : Type} (rate : R → ℝ) {I O : Type} (v : O) : (boundedMc rate).Alg I O :=
  ⟨(fun _ => v, fun _ _ => 0, fun _ => 0), fun _ _ => ⟨le_rfl, by rw [mul_zero]⟩⟩

noncomputable def idleAt (c : Verity.Accel.Card) (x : Verity.Accel.Open) (hr : HostRow) (u : ResUnits) :
    Prover (accelAt c x hr u) twRound 2 where
  St := Unit
  start := ()
  commit _ := idleAlg (devRate (devs c hr)) (((), twInst) : Unit × twRound.Inst)
  work _ := idleAlg (devRate (devs c hr)) (((), fun _ _ _ => 0) : Unit × twRound.Root)
  absorb _ _ _ := ()
  answer _ := idleAlg (devRate (devs c hr)) ((fun _ => (0, ())) : twRound.Ans)

theorem idleAt_timedUse (c : Verity.Accel.Card) (x : Verity.Accel.Open) (hr : HostRow) (u : ResUnits)
    (o : twRound.Orc) (cs : Fin 2 → twRound.Draw × twRound.Chk) (r : (accelAt c x hr u).Res) :
    (idleAt c x hr u).timedUse o cs r = 0 := by
  show ∑ _b : Fin 2, ((0 : ℝ) + 0) = 0
  simp

theorem idleAt_fits (c : Verity.Accel.Card) (x : Verity.Accel.Open) (hr : HostRow) (u : ResUnits)
    (μ : (accelAt c x hr u).Res → ℝ) (o : twRound.Orc) (cs : Fin 2 → twRound.Draw × twRound.Chk) :
    (halfPd (accelAt c x hr u) μ).Fits ((idleAt c x hr u).timedUse o cs) fun _ => 1 := by
  have e : (idleAt c x hr u).timedUse o cs = fun _ => 0 := funext (idleAt_timedUse c x hr u o cs)
  rw [e]
  exact halfPd_idle_fits _ _

theorem freedSound_fails_at (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow)
    (hh : 0 < hr.amx ∧ 0 < hr.sha) (u : ResUnits) (x : Verity.Accel.Open) :
    ¬ FreedSound (accelAt c x hr u) twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ())
      (halfPd (accelAt c x hr u) fun _ => 1) 1 (fun _ => 1 / 2)
      (∑ r, (fun _ => (1 : ℝ)) r * (1 * (halfPd (accelAt c x hr u) fun _ => 1).hon r)) (fun _ => 1 / 2) := by
  intro h
  have hrate : ∀ r, 0 < (accelAt c x hr u).rate r :=
    (cardAccel_onCard c x (devs c hr) u twRound (idLayout c x hr) hρ (devs_oth_pos c hr hh)
      (devs_mem c hr)).2.2.1.1
  have : Nonempty (twRound.Orc × (Fin 2 → twRound.Draw × twRound.Chk)) :=
    (inferInstance : Nonempty (Unit × (Fin 2 → TDraw × Fin 2)))
  have h1 := h (idleAt c x hr u)
  rw [Pouw.SecurityProofs.DitherWitness.prob_of_forall] at h1
  · norm_num at h1
  intro ox
  exact ⟨trivial, fun _ => 1, idleAt_fits c x hr u _ ox.1 ox.2,
    halfPd_not_freed (accelAt c x hr u) (Sum.inl Verity.Accel.Res.tensor : Verity.Accel.Res ⊕ (devs c hr).oth.Res)
      hrate⟩

theorem nontrivial_at (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow) (hh : 0 < hr.amx ∧ 0 < hr.sha)
    (u : ResUnits) (x : Verity.Accel.Open) :
    ¬ WorkBound (accelAt c x hr u) twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ())
      (halfPd (accelAt c x hr u) fun _ => 1) (fun _ => 1 / 2)
      (∑ r, (fun _ => (1 : ℝ)) r * (1 * (halfPd (accelAt c x hr u) fun _ => 1).hon r)) (fun _ => 1 / 2)
      (fun _ => True) :=
  fun h => freedSound_fails_at c hρ hr hh u x h.1

/-- A table that models no hardware; the witnesses read only that its rates are positive. -/
def unitCard : Verity.Accel.Card where
  size _ := 1
  size_pos _ := Nat.one_pos
  ρ _ := 1
  weight _ := 1
  exact := 0
  text := 0

theorem unitCard_rates_pos : ∀ r, 0 < unitCard.ρ r := fun _ => zero_lt_one

def unitSha : ResUnits := shaUnits 1 0 0

def unitMenu : PubBest unitCard := ⟨[], [unitSha]⟩

noncomputable def unitHost : HostRow := ⟨1, 1⟩

theorem unitHost_pos : 0 < unitHost.amx ∧ 0 < unitHost.sha := ⟨zero_lt_one, zero_lt_one⟩

end Instance

section Witnesses

theorem honest_exists (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow) (hh : 0 < hr.amx ∧ 0 < hr.sha)
    (pb : PubBest c) (u : ResUnits) (hu : u ∈ pb.sha) :
    ∃ (x : Verity.Accel.Open) (D : Devices c) (G : Accel) (L : RoundPlan (D.machine x) twRound)
      (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G) (μ : G.Res → ℝ),
      c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ HostsCounted D hr ∧
        CallsAtPub G e pb ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1 :=
  ⟨Verity.Accel.s8Open, devs c hr, accelAt c Verity.Accel.s8Open hr u, idLayout c Verity.Accel.s8Open hr,
    Equiv.refl _, halfPd (accelAt c Verity.Accel.s8Open hr u) fun _ => 1, fun _ => 1, honest_at c hρ hr hh pb u hu⟩

/-- The premises are an `∃` over their proof, so that past its `∃`s the statement is `¬ WorkBound …`. -/
theorem nontrivial_exists (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow)
    (hh : 0 < hr.amx ∧ 0 < hr.sha) (pb : PubBest c) (u : ResUnits) (hu : u ∈ pb.sha) :
    ∃ (x : Verity.Accel.Open) (D : Devices c) (G : Accel) (L : RoundPlan (D.machine x) twRound)
      (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G) (μ : G.Res → ℝ)
      (_ : c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ HostsCounted D hr ∧
        CallsAtPub G e pb ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1),
      ¬ WorkBound G twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ()) Pd (fun _ => 1 / 2)
        (∑ r, μ r * (1 * Pd.hon r)) (fun _ => 1 / 2) (fun _ => True) :=
  ⟨Verity.Accel.s8Open, devs c hr, accelAt c Verity.Accel.s8Open hr u, idLayout c Verity.Accel.s8Open hr,
    Equiv.refl _, halfPd (accelAt c Verity.Accel.s8Open hr u) fun _ => 1, fun _ => 1, honest_at c hρ hr hh pb u hu,
    nontrivial_at c hρ hr hh u Verity.Accel.s8Open⟩

/-- **`WorkSoundAtTables`'s honest witness**: its premises on the table and the instance hold together somewhere: at
some card table `c`, menu `pb`, host row `hr`, semantics `x` exact on the table and of the product form, devices `D`,
accelerator `G` that is those devices at `x` under a match `e` and a layout `L` of the no-fold round `twRound`
(`OnCard`) with the row's hosts counted (`HostsCounted`) and its oracle calls at a public compression (`CallsAtPub`),
meeting (G1), (G2) and (G3), and period with a dual price at λ = 1 (`DualOpt`).

The statement names no table. **The proof's instance is `unitCard`'s**, a table that models no hardware, with
`unitMenu` and `unitHost`, and at them `honest_exists`'s instance with the call step at `unitSha`. It shows no
honest prover passing, none of the named assumptions (`SAKMatmul` is false at `cardAccel`, whose programs are any runs
within the rates) and none of the statement's hypotheses on the work split and the call bound. -/
theorem WorkSoundAtTables_honest :
    ∃ (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) (x : Verity.Accel.Open) (D : Devices c) (G : Accel)
      (L : RoundPlan (D.machine x) twRound) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G)
      (μ : G.Res → ℝ),
      c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ HostsCounted D hr ∧
        CallsAtPub G e pb ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1 :=
  ⟨unitCard, unitMenu, unitHost,
    honest_exists unitCard unitCard_rates_pos unitHost unitHost_pos unitMenu unitSha (List.Mem.head _)⟩

/-- **`WorkSoundAtTables`'s nontrivial witness**: at some accelerator `G`, period `Pd` and prices `μ`, the statement's
conclusion (`WorkBound`, at `W_reg = μ · hon`) fails for the no-fold round `twRound` with two sampling
blocks under the verdict that accepts everything (`fun _ _ _ _ => True`), at `ε = κ = 1/2`: a verdict that checks
nothing bounds no prover's freed compute.

The statement names no table. **The proof's instance is `unitCard`'s**, the one `WorkSoundAtTables_honest`'s proof
supplies: `cardAccel` on one card of `unitCard` with `unitHost`'s hosts at `Verity.Accel.s8Open` (`accelAt`), the
period `halfPd` and the prices 1; the prover the bound fails at is the idle one (`idleAt`). -/
theorem WorkSoundAtTables_nontrivial :
    ∃ (G : Accel) (Pd : Period G) (μ : G.Res → ℝ),
      ¬ WorkBound G twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ()) Pd (fun _ => 1 / 2)
        (∑ r, μ r * (1 * Pd.hon r)) (fun _ => 1 / 2) (fun _ => True) :=
  ⟨accelAt unitCard Verity.Accel.s8Open unitHost unitSha,
    halfPd (accelAt unitCard Verity.Accel.s8Open unitHost unitSha) fun _ => 1, fun _ => 1,
    nontrivial_at unitCard unitCard_rates_pos unitHost unitHost_pos _ _⟩

end Witnesses

end Pouw.SecurityProofs.DitherInstance
