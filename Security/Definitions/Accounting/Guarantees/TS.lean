module

public import Definitions.Accounting.Assumptions.TS

/-!
# The tensor–scalar model's guarantee statements (trusted)

EPS1's statements over `Accounting.TS`,
each `def G : Prop`, proved (or a `sorry` stub) as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS`:

* `TimeBound`: Lemma 1, `T(P) ≥ B(P)/f_max`.
* `DIdeal`, `DIdealFloor`: Theorem D-ideal, (2.7) and (2.7♭), in 𝒫_full from (I1)–(I8).
* `CompilerC`: Lemma C, the compiler, in 𝒫_samp: the same bound with `ε_sample` added.
* `TensorSlotCount`, `TensorSlotTight`: T1, the tensor slot count, and its tightness criterion.
* `OutputCount`: Lemma 2's count, owned committed positions at most `Σ_k ω_k n_k`.
* `FanInCount`, `SeesOfInjective`, `ThinFoldInjective`: the fan-in count, `|ι| ≤ |O| + Σ_i (sources − 1)` for a
  straight-line program whose outputs see every input; a receipt injective in each word sees it; the thin fold's is.
* `FoldOptOfFanIn`, `FillOptOfFanIn`: (I6) from `FanIn` at the fan-in menu's price.

Probability statements put acceptance in the event: the probability that the window is accepted and the bound fails is
at most `δ`.
-/

@[expose] public section

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **Lemma 1 (time).** On the stated device, every run's time is at least its bound over `f_max`. -/
def TimeBound : Prop :=
  ∀ {K R : Type} [Fintype K] [Fintype R] [Nonempty R] (G : Machine K R) (P : Run K),
    FacetCeilings G P → G.bound P.n / G.fmax ≤ P.time

/-- **Theorem D-ideal (2.7).** In 𝒫_full (every committed value and every C1′ operand proved each round, one σ a
round), under (I1) `FacetCeilings`, (I2)/(I4) `WordsWritten`, (I3) `ScalarRoute`, (I5) `SideUnits` and `NoSum`, (I6)
`FoldOpt` and `FillOpt`, (I7) `LuckAt p` and (I8) `ProofsSound`, for every certificate `π` and every cheater, outside an
event of probability at most `δ`, every accepted run satisfies
`T(M)/T(H) ≥ (1 − ε_skip − ε_mand − ε_machine − ε_jl)(π)·(1 − ε_sched)(1 − ε_clock)`, with
`δ′ = δ − S(δ_bind + n_π·δ_ZK)` in `ε_jl` and `ε_adapt = 0`. -/
def DIdeal : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (H : Honest K) (Ω : Type) [Fintype Ω] (S : ℕ) (P : Play Ω S K T L U)
    (lam : R → ℝ) (p δ δbind δzk : ℝ) (nπ : ℕ),
    IsCert lam → Z.Wf H.prog → 0 < H.CH → 0 < H.fH → 0 < S → 0 < p → p < 1 →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → LuckAt p P.game → ProofsSound P Z.Nw δbind δzk nπ →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / H.time <
      (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
        Z.epsJl H.prog lam p (δ - S * (δbind + nπ * δzk)) S) * (1 - Z.epsSched H) * (1 - Z.epsClock H)) ≤ δ

/-- **Theorem D-ideal (2.7♭), under the floor credit.** As `DIdeal`, against the credit `T_cred = B_♭/f_max` of any
labelled program ♭: outside an event of probability at most `δ`, every accepted run satisfies
`T(M)/T_cred ≥ 1 − ε^♭_skip − ε^♭_cert − ε^♭_route − ε^♭_mand − ε^♭_jl` (each term at ♭, over `B_♭`). -/
def DIdealFloor : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (Fl : Labelled K) (Ω : Type) [Fintype Ω] (S : ℕ) (P : Play Ω S K T L U)
    (lam : R → ℝ) (p δ δbind δzk : ℝ) (nπ : ℕ),
    IsCert lam → Z.Wf Fl → 0 < S → 0 < p → p < 1 →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → LuckAt p P.game → ProofsSound P Z.Nw δbind δzk nπ →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        Z.epsJl Fl lam p (δ - S * (δbind + nπ * δzk)) S) ≤ δ

/-- **Lemma C (the compiler).** In 𝒫_samp(u, s, Q), whose ideal game's share is a round's deficient, wrong and
adaptive work share and whose check is the sampled proof blocks' proofs, under (I1)–(I6) as in `DIdeal`, (C1)–(C3) with (I8)
as `ProofsSound`, and any `Λ > 0` bounding the round's saving per unit of detection (`RoundGame.BudgetAt Λ`, Λ's
definition): for every certificate `π`, outside an event of probability at most `δ`, every accepted run satisfies
(2.7) with `ε_sample = (Λ − 1/ln(1/p))·(ln(1/δ′)/S)·κ(π)` subtracted in the first factor. This is Lemma C(b), at a
given `Λ`. Lemma C(a), `Λ ≤ max(Λ_luck, Fan_Q/s)`, is not stated: it needs (C4), that every unit holding a C1′
derivation a steer passes through (residual adds included) has `Q`-weight at least its consumers' work share over a
stated Fan cap, or a closure that checks `h = h_prev + y` (REFEREE S13). Without it, a direct write to the residual
steers every consumer from a unit of near-zero `Q`-mass, and `BudgetAt` holds only at a `Λ` of about the steered
share over that mass: this bound stays true, and is vacuous there. -/
def CompilerC : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (H : Honest K) (Ω : Type) [Fintype Ω] (S : ℕ) (P : Play Ω S K T L U)
    (lam : R → ℝ) (Λ p δ δbind δzk : ℝ) (nπ : ℕ),
    IsCert lam → Z.Wf H.prog → 0 < H.CH → 0 < H.fH → 0 < S → 0 < p → p < 1 → 0 < Λ →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → RoundGame.BudgetAt Λ P.game → ProofsSound P Z.Nw δbind δzk nπ →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / H.time <
      (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
        Z.epsJl H.prog lam p (δ - S * (δbind + nπ * δzk)) S -
        Z.epsSample H.prog lam Λ p (δ - S * (δbind + nπ * δzk)) S) * (1 - Z.epsSched H) * (1 - Z.epsClock H)) ≤ δ

/-- **T1 (tensor slot counting).** A program that commits `N` fresh words, each read whole by the fold from its own
slot write of a tensor instruction (freshness makes the map injective), holds the tensor pipe at least `N/ρ_T`
cycles: `Σ_i c(i) n_i ≥ N/ρ_T`. Injectivity is the premise (one word a slot): design A's packed `T_1` words, two a
s32 slot (REFEREE S14), are outside it. -/
def TensorSlotCount : Prop :=
  ∀ (ι : Type) [Fintype ι] [Nonempty ι] (w c n : ι → ℕ) (N : ℕ) (slot : Fin N → SlotWrite ι w n),
    (∀ i, 0 < c i) → Function.Injective slot → (N : ℝ) / rhoT w c ≤ ∑ i, (c i : ℝ) * n i

/-- **Lemma 2 (output count).** In a straight-line program whose class-`k` instances write at most `ω_k` words each,
the owned committed positions (one per head write their values trace to) are at most `Σ_k ω_k n_k`. -/
def OutputCount : Prop :=
  ∀ {K : Type} [Fintype K] [DecidableEq K] (ω : K → ℕ) (G : ℕ) (cls : Fin G → K) (Nc : ℕ)
    (head : Fin Nc → HeadWrite ω G cls),
    ((Finset.univ.image head).card : ℝ) ≤ ∑ k, (ω k : ℝ) * classCount cls k

/-- **The fan-in count.** A straight-line program whose outputs `O` see every one of its fresh inputs has merged them
into at most `|O|` components, a node merging its sources less one: `|ι| ≤ |O| + Σ_i (|src i| − 1)`. -/
def FanInCount : Prop :=
  ∀ {ι W : Type} [Fintype ι] [Nontrivial W] [Inhabited W] (D : Dataflow ι W) (O : Finset (ι ⊕ Fin D.n)),
    D.Sees O → Fintype.card ι ≤ O.card + ∑ i, D.merges i

/-- **FOLD-OPT from fan-in.** If the run's fold is a straight-line program over the `N_w` committed words whose
receipt `O` sees every word, issued as instructions `ins` of classes `cls` that `FanIn` bounds at `m ≥ 0`, labelled to
the fold (at most `A.nfold k` instructions of class `k`), with `|O| ≤ μ_f·N_w`, and the fold menu holds every class's
fan-in route (a word at `a_{r,k}/m_k` on every facet, for every `k` with `m_k > 0`), then `FoldOpt`. -/
def FoldOptOfFanIn : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty F] [Nonempty X] {ι W J : Type} [Fintype ι] [Nontrivial W] [Inhabited W] [Fintype J] [DecidableEq J]
    [DecidableEq K] (Z : Model K R T L U F X) (P : Labelled K) (A : Anatomy K T L U) (D : Dataflow ι W)
    (O : Finset (ι ⊕ Fin D.n)) (ins : Fin D.n → J) (cls : J → K) (m : K → ℝ),
    Z.Wf P → D.Sees O → FanIn D ins cls m → (∀ k, 0 ≤ m k) →
    (∀ k, ((Finset.univ.filter fun j => cls j = k).card : ℝ) ≤ A.nfold k) →
    (Fintype.card ι : ℝ) = Z.Nw → (O.card : ℝ) ≤ Z.muF * Z.Nw →
    (∀ k, 0 < m k → ∃ φ, ∀ r, Z.fold φ r ≤ Z.G.a r k / m k) →
    FoldOpt Z A

/-- **FILL-OPT from fan-in.** As `FoldOptOfFanIn` for the filler over the `N_x = |ι| > 0` fresh inputs its receipt
sees, with `|O| ≤ μ_x·N_x` and the fill menu's routes at `N_x·a_{r,k}/m_k`. A filler over public inputs (its start
and the coins) sees none of the committed words, so this prices it at most its own inputs' fan-in. -/
def FillOptOfFanIn : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty F] [Nonempty X] {ι W J : Type} [Fintype ι] [Nonempty ι] [Nontrivial W] [Inhabited W] [Fintype J]
    [DecidableEq J] [DecidableEq K] (Z : Model K R T L U F X) (P : Labelled K) (A : Anatomy K T L U)
    (D : Dataflow ι W) (O : Finset (ι ⊕ Fin D.n)) (ins : Fin D.n → J) (cls : J → K) (m : K → ℝ),
    Z.Wf P → D.Sees O → FanIn D ins cls m → (∀ k, 0 ≤ m k) →
    (∀ k, ((Finset.univ.filter fun j => cls j = k).card : ℝ) ≤ A.nfill k) →
    (O.card : ℝ) ≤ Z.muX * Fintype.card ι →
    (∀ k, 0 < m k → ∃ ψ, ∀ r, Z.fill ψ r ≤ Fintype.card ι * Z.G.a r k / m k) →
    FillOpt Z A

end Accounting.Guarantees.TS
