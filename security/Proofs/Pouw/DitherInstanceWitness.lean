module

public import Proofs.Pouw.DitherInstance
public import Proofs.Pouw.PoolWitness

/-!
# The headline on the cards says something

The honest and nontrivial witnesses of `WorkSoundAtTables`, `WorkSoundH200` and `WorkSoundSm120`. No statement here
reads the proofs tree: each binds by `∃` every part of its instance that is built here (the semantics, the devices,
the accelerator, the layout, the resource match, the period and the prices), and its proof supplies them.

* `WorkSoundAtTables_honest`, `WorkSoundAtTables_nontrivial`: the statement over every table. They name no table: the
  table, the menu and the host row are `∃`s too, and the proofs supply the H200's.
* `WorkSoundH200_honest`, `WorkSoundH200_nontrivial`: the statement at the H200. They name its table
  `Verity.Accel.H200.card`, its menu `pubBestH200` and its host row `hostH200`, which the definitions tree holds.
* `WorkSoundSm120_honest`, `WorkSoundSm120_nontrivial`: the same at the RTX PRO 6000's `Verity.Accel.Sm120.card`,
  `pubBestSm120` and `hostSm120`.

An honest witness concludes, at one instance, the premises its statement takes on the semantics, the devices, the
accelerator and the period. A nontrivial witness states that at such an instance the statement's conclusion
(`WorkBound`) fails under the verdict that accepts everything.

**The instance every proof here supplies**, at the table `c`, the menu `pb` and the host row `hr`: the exact `s8`
semantics `Verity.Accel.s8Open`; one card with its row's hosts and no other memory (`devs`); the accelerator
`cardAccel` on those devices with its call step at the first SHA-512 schedule of the menu (`accelAt`); the layout that
moves nothing (`idLayout`); the identity match; the one-second period `halfPd` at the prices 1. `honest_at` and
`nontrivial_at` hold at any table whose rates are positive, so a table's numbers enter through `h200_rates_pos`,
`sm120_rates_pos`, `hostH200_pos` and `hostSm120_pos` alone.
-/

@[expose] public section

namespace Pouw.SecurityProofs.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Guarantees.DitherInstance Pouw.Witness

open scoped NNRat

section HalfPd

/-- One second of the devices `G`: the one other job runs every resource at its rate for a second and is worth its use
at `μ`, and the credited work uses half of every resource. -/
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

/-- With no credited use, the other job at one unit fits. -/
theorem halfPd_idle_fits (G : Accel) (μ : G.Res → ℝ) : (halfPd G μ).Fits (fun _ => 0) fun _ => 1 := by
  refine ⟨fun _ => zero_le_one, fun r => ?_⟩
  show (0 : ℝ) + ∑ _j : Unit, (1 : ℝ) * G.rate r ≤ G.rate r * 1
  simp only [Fintype.sum_unique]
  linarith

/-- At the prices 1, the other job at one unit beside one credit is worth more than any honest strategy plus half the
credited cost. -/
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

/-- One card of the table `c` with its row's hosts and no other memory. -/
noncomputable abbrev devs (c : Verity.Accel.Card) (hr : HostRow) : Devices c := ⟨1, Nat.one_pos, hr.others 1 0⟩

/-- `cardAccel` on those devices at the semantics `x`, its call step at the units `u`. -/
noncomputable abbrev accelAt (c : Verity.Accel.Card) (x : Verity.Accel.Open) (hr : HostRow) (u : ResUnits) : Accel :=
  cardAccel c x (devs c hr) u

/-- The layout that moves nothing. -/
noncomputable def idLayout (c : Verity.Accel.Card) (x : Verity.Accel.Open) (hr : HostRow) :
    Layout ((devs c hr).machine x) twRound :=
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

/-- **The honest instance, at any table**: with every rate of the table positive, both host rates positive and `u` one
of the menu's SHA-512 schedules, the premises of `WorkSoundAt c pb hr` on the semantics, the devices, the accelerator
and the period hold together at one card, its row's hosts, the exact `s8` semantics, `cardAccel` and `halfPd` at the
prices 1. -/
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

/-- A program of `boundedMc` that returns `v` at no use and no wall time. -/
noncomputable def idleAlg {R : Type} (rate : R → ℝ) {I O : Type} (v : O) : (boundedMc rate).Alg I O :=
  ⟨(fun _ => v, fun _ _ => 0, fun _ => 0), fun _ _ => ⟨le_rfl, by rw [mul_zero]⟩⟩

/-- **The idle prover on the card**: every sampling block commits the zero instance and the zero root and answers
zero, at no use. -/
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

/-- **The bound on freed compute fails, at any table and any semantics**: with every rate of the table and both host
rates positive, `FreedSound` fails on the card under the verdict that accepts everything, at `ε = κ = 1/2`: the idle
prover is accepted on every oracle and coins and leaves the whole period to other work. -/
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

/-- **The nontrivial instance, at any table and any semantics**: `WorkSoundAtX`'s conclusion (`WorkBound`) fails there
under the verdict that accepts everything, since its first part does (`freedSound_fails_at`). -/
theorem nontrivial_at (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow) (hh : 0 < hr.amx ∧ 0 < hr.sha)
    (u : ResUnits) (x : Verity.Accel.Open) :
    ¬ WorkBound (accelAt c x hr u) twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ())
      (halfPd (accelAt c x hr u) fun _ => 1) (fun _ => 1 / 2)
      (∑ r, (fun _ => (1 : ℝ)) r * (1 * (halfPd (accelAt c x hr u) fun _ => 1).hon r)) (fun _ => 1 / 2)
      (fun _ => True) :=
  fun h => freedSound_fails_at c hρ hr hh u x h.1

end Instance

section Witnesses

/-- **The honest statement at the table `c`**, with the rest of the instance bound by `∃`: at some semantics `x`,
devices `D`, accelerator `G`, layout `L` of the no-fold round `twRound`, match `e`, period `Pd` and prices `μ`, the
premises of `WorkSoundAt c pb hr` on them hold together. The proof's instance is `honest_at`'s. -/
theorem honest_exists (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow) (hh : 0 < hr.amx ∧ 0 < hr.sha)
    (pb : PubBest c) (u : ResUnits) (hu : u ∈ pb.sha) :
    ∃ (x : Verity.Accel.Open) (D : Devices c) (G : Accel) (L : Layout (D.machine x) twRound)
      (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G) (μ : G.Res → ℝ),
      c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ HostsCounted D hr ∧
        CallsAtPub G e pb ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1 :=
  ⟨Verity.Accel.s8Open, devs c hr, accelAt c Verity.Accel.s8Open hr u, idLayout c Verity.Accel.s8Open hr,
    Equiv.refl _, halfPd (accelAt c Verity.Accel.s8Open hr u) fun _ => 1, fun _ => 1, honest_at c hρ hr hh pb u hu⟩

/-- **The nontrivial statement at the table `c`**, with the rest of the instance bound by `∃`: at some instance
where the premises `honest_exists` concludes hold, `WorkBound` fails under the verdict that accepts everything. The
premises are an `∃` over their proof, so that past its `∃`s the statement is `¬ WorkBound …`. -/
theorem nontrivial_exists (c : Verity.Accel.Card) (hρ : ∀ r, 0 < c.ρ r) (hr : HostRow)
    (hh : 0 < hr.amx ∧ 0 < hr.sha) (pb : PubBest c) (u : ResUnits) (hu : u ∈ pb.sha) :
    ∃ (x : Verity.Accel.Open) (D : Devices c) (G : Accel) (L : Layout (D.machine x) twRound)
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

The statement names no table. **The proof's instance is the H200's**: `Verity.Accel.H200.card`, `pubBestH200`,
`hostH200`, and at them `honest_exists`'s instance with the call step at `pubShaH200`'s first schedule. It shows no
honest prover passing, none of the named assumptions (`SAKMatmul` is false at `cardAccel`, whose programs are any runs
within the rates) and none of the statement's hypotheses on the work split and the call bound. -/
theorem WorkSoundAtTables_honest :
    ∃ (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) (x : Verity.Accel.Open) (D : Devices c) (G : Accel)
      (L : Layout (D.machine x) twRound) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G)
      (μ : G.Res → ℝ),
      c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ HostsCounted D hr ∧
        CallsAtPub G e pb ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1 :=
  ⟨Verity.Accel.H200.card, pubBestH200, hostH200,
    honest_exists Verity.Accel.H200.card h200_rates_pos hostH200 hostH200_pos pubBestH200
      (shaUnits 3400 152 (3608 / 32)) (List.Mem.head _)⟩

/-- **`WorkSoundAtTables`'s nontrivial witness**: at some accelerator `G`, period `Pd` and prices `μ`, the statement's
conclusion (`WorkBound`, at `W_reg = μ · hon`) fails for the no-fold round `twRound` with two sampling
blocks under the verdict that accepts everything (`fun _ _ _ _ => True`), at `ε = κ = 1/2`: a verdict that checks
nothing bounds no prover's freed compute.

The statement names no table. **The proof's instance is the H200's**, the one `WorkSoundAtTables_honest`'s proof
supplies: `cardAccel` on one card of `Verity.Accel.H200.card` with `hostH200`'s hosts at `Verity.Accel.s8Open`
(`accelAt`), the period `halfPd` and the prices 1; the prover the bound fails at is the idle one (`idleAt`). -/
theorem WorkSoundAtTables_nontrivial :
    ∃ (G : Accel) (Pd : Period G) (μ : G.Res → ℝ),
      ¬ WorkBound G twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ()) Pd (fun _ => 1 / 2)
        (∑ r, μ r * (1 * Pd.hon r)) (fun _ => 1 / 2) (fun _ => True) :=
  ⟨accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32)),
    halfPd (accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32))) fun _ => 1,
    fun _ => 1,
    nontrivial_at Verity.Accel.H200.card h200_rates_pos hostH200 hostH200_pos _ _⟩

/-- **`WorkSoundH200`'s honest witness**: on the H200's table `Verity.Accel.H200.card`, with its menu `pubBestH200`
and its host row `hostH200`, the statement's premises on the semantics, the devices, the accelerator and the period
hold together at some instance. The statement names the table, the menu and the row, which the definitions tree
holds, and binds the rest by `∃`; the proof supplies `honest_exists`'s instance with the call step at `pubShaH200`'s
first schedule. It shows no honest prover passing and none of the named assumptions. -/
theorem WorkSoundH200_honest :
    ∃ (x : Verity.Accel.Open) (D : Devices Verity.Accel.H200.card) (G : Accel) (L : Layout (D.machine x) twRound)
      (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G) (μ : G.Res → ℝ),
      Verity.Accel.H200.card.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧
        OnCard G Verity.Accel.H200.card x D twRound L e ∧ HostsCounted D hostH200 ∧ CallsAtPub G e pubBestH200 ∧
        G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1 :=
  honest_exists Verity.Accel.H200.card h200_rates_pos hostH200 hostH200_pos pubBestH200
    (shaUnits 3400 152 (3608 / 32)) (List.Mem.head _)

/-- **`WorkSoundH200`'s nontrivial witness**: on the H200's table, at some instance where the premises
`WorkSoundH200_honest` concludes hold, the statement's conclusion (`WorkBound`) fails for the no-fold round with two
sampling blocks under the verdict that accepts everything, at `ε = κ = 1/2`. The statement names the table, the menu
and the row and binds the rest by `∃`; the proof's instance is `WorkSoundH200_honest`'s, and the prover the bound
fails at is the idle one (`idleAt`). -/
theorem WorkSoundH200_nontrivial :
    ∃ (x : Verity.Accel.Open) (D : Devices Verity.Accel.H200.card) (G : Accel) (L : Layout (D.machine x) twRound)
      (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G) (μ : G.Res → ℝ)
      (_ : Verity.Accel.H200.card.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧
        OnCard G Verity.Accel.H200.card x D twRound L e ∧ HostsCounted D hostH200 ∧ CallsAtPub G e pubBestH200 ∧
        G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1),
      ¬ WorkBound G twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ()) Pd (fun _ => 1 / 2)
        (∑ r, μ r * (1 * Pd.hon r)) (fun _ => 1 / 2) (fun _ => True) :=
  nontrivial_exists Verity.Accel.H200.card h200_rates_pos hostH200 hostH200_pos pubBestH200
    (shaUnits 3400 152 (3608 / 32)) (List.Mem.head _)

/-- **`WorkSoundSm120`'s honest witness**: on the RTX PRO 6000's table `Verity.Accel.Sm120.card`, with its menu
`pubBestSm120` and its host row `hostSm120`, the statement's premises on the semantics, the devices, the accelerator
and the period hold together at some instance. The statement names the table, the menu and the row, which the
definitions tree holds, and binds the rest by `∃`; the proof supplies `honest_exists`'s instance with the call step
at `pubShaSm120`'s first schedule. It shows no honest prover passing and none of the named assumptions. -/
theorem WorkSoundSm120_honest :
    ∃ (x : Verity.Accel.Open) (D : Devices Verity.Accel.Sm120.card) (G : Accel) (L : Layout (D.machine x) twRound)
      (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G) (μ : G.Res → ℝ),
      Verity.Accel.Sm120.card.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧
        OnCard G Verity.Accel.Sm120.card x D twRound L e ∧ HostsCounted D hostSm120 ∧ CallsAtPub G e pubBestSm120 ∧
        G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1 :=
  honest_exists Verity.Accel.Sm120.card sm120_rates_pos hostSm120 hostSm120_pos pubBestSm120
    (shaUnits 3415 0 (3415 / 32)) (List.Mem.head _)

/-- **`WorkSoundSm120`'s nontrivial witness**: on the RTX PRO 6000's table, at some instance where the premises
`WorkSoundSm120_honest` concludes hold, the statement's conclusion (`WorkBound`) fails for the no-fold round with two
sampling blocks under the verdict that accepts everything, at `ε = κ = 1/2`. The statement names the table, the menu
and the row and binds the rest by `∃`; the proof's instance is `WorkSoundSm120_honest`'s, and the prover the bound
fails at is the idle one (`idleAt`). -/
theorem WorkSoundSm120_nontrivial :
    ∃ (x : Verity.Accel.Open) (D : Devices Verity.Accel.Sm120.card) (G : Accel) (L : Layout (D.machine x) twRound)
      (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G) (μ : G.Res → ℝ)
      (_ : Verity.Accel.Sm120.card.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧
        OnCard G Verity.Accel.Sm120.card x D twRound L e ∧ HostsCounted D hostSm120 ∧ CallsAtPub G e pubBestSm120 ∧
        G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1),
      ¬ WorkBound G twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ()) Pd (fun _ => 1 / 2)
        (∑ r, μ r * (1 * Pd.hon r)) (fun _ => 1 / 2) (fun _ => True) :=
  nontrivial_exists Verity.Accel.Sm120.card sm120_rates_pos hostSm120 hostSm120_pos pubBestSm120
    (shaUnits 3415 0 (3415 / 32)) (List.Mem.head _)

end Witnesses

end Pouw.SecurityProofs.DitherInstance
