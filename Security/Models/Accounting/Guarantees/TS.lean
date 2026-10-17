import Models.Accounting.Assumptions.TS

/-!
# The tensor–scalar model's guarantee statements (trusted)

EPS1's statements over `Accounting.TS` (note:20261017T0324Z-draft-eps1-spec, EPS1-TENSOR
note:20261017T0321Z-draft-eps1-tensor), each `def G : Prop`, proved (or a `sorry` stub) as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS`:

* `TimeBound`: Lemma 1, `T(P) ≥ B(P)/f_max`.
* `DecompositionD`: Theorem D(a), the master inequality from (Π), (RD) and (Lk).
* `DIdeal`, `DIdealFloor`: Theorem D-ideal, (2.7) and (2.7♭), in 𝒫_full from (I1)–(I8).
* `CompilerC`: Lemma C, the compiler, in 𝒫_samp: the same bound with `ε_sample` added.
* `FloorCreditD3`: Corollary D3, the credit at the floor, with every term `≥ 0` and `T(H) = γ_♭·T_cred`.
* `BindingFacetD1`, `VectorThresholdD1`: Corollary D1, one binding facet, and the vector route's threshold
  `q ≥ 32(α + β·φ_del)/(β·ω)`.
* `RoutesOffD2`: Corollary D2, routes off the binding facet.
* `TensorSlotCount`, `TensorSlotTight`: T1, the tensor slot count, and its tightness criterion.
* `OutputCount`: Lemma 2's count, owned committed positions at most `Σ_k ω_k n_k`.

Probability statements put acceptance in the event: the probability that the window is accepted and the bound fails is
at most `δ`.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **Lemma 1 (time).** On the stated device, every run's time is at least its bound over `f_max`. -/
def TimeBound : Prop :=
  ∀ {K R : Type} [Fintype K] [Fintype R] [Nonempty R] (G : Machine K R) (P : Run K),
    FacetCeilings G P → G.bound P.n / G.fmax ≤ P.time

/-- **Theorem D(a) (the master inequality).** Given (Π) on every run, (RD) on every accepted run and (Lk) at
`(d̄, ā, δ)`, for every certificate `π`, outside an event of probability at most `δ`, every accepted run satisfies
`T(M)/T(H) ≥ (1 − ε_skip − ε_mand − ε_machine − ε_luck − ε_adapt)(π)·(1 − ε_sched)(1 − ε_clock)`. -/
def DecompositionD : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (H : Honest K) (Ω : Type) [Fintype Ω] (accepts : Ω → Prop) (M : Ω → Run K)
    (fixed : Ω → ℝ) (lam : R → ℝ) (dbar abar δ : ℝ),
    IsCert lam → Z.Wf H.prog → 0 < H.CH → 0 < H.fH →
    (∀ ω, FacetCeilings Z.G (M ω)) →
    (∀ ω, accepts ω → Z.RouteDominated (M ω).n (fixed ω)) →
    prob (fun ω => accepts ω ∧ fixed ω < (1 - dbar - abar) * Z.Nw) ≤ δ →
    prob (fun ω => accepts ω ∧ (M ω).time / H.time <
      (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam - Z.epsLuck H.prog lam dbar -
        Z.epsAdapt H.prog lam abar) * (1 - Z.epsSched H) * (1 - Z.epsClock H)) ≤ δ

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
adaptive work share and whose check is the drawn units' proofs, under (I1)–(I6) as in `DIdeal`, (C1)–(C3) with (I8)
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

/-- **Corollary D3 (credit at the floor).** Given (Π), (RD) and (Lk) as in `DecompositionD`, for every labelled
program ♭ and certificate `π`: outside an event of probability at most `δ`, every accepted run satisfies
`T(M)/T_cred ≥ 1 − ε^♭_skip − ε^♭_cert − ε^♭_route − ε^♭_mand − ε^♭_luck − ε^♭_adapt`; when ♭ is on the known menu
(and its skippable work, `d̄` and `ā` are `≥ 0`) every term is `≥ 0`; and `T(H) = γ_♭·T_cred`. -/
def FloorCreditD3 : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (H : Honest K) (Fl : Labelled K) (Ω : Type) [Fintype Ω] (accepts : Ω → Prop)
    (M : Ω → Run K) (fixed : Ω → ℝ) (lam : R → ℝ) (dbar abar δ : ℝ),
    IsCert lam → Z.Wf Fl → 0 < H.CH → 0 < H.fH →
    (∀ ω, FacetCeilings Z.G (M ω)) →
    (∀ ω, accepts ω → Z.RouteDominated (M ω).n (fixed ω)) →
    prob (fun ω => accepts ω ∧ fixed ω < (1 - dbar - abar) * Z.Nw) ≤ δ →
    prob (fun ω => accepts ω ∧ (M ω).time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam - Z.epsLuck Fl lam dbar -
        Z.epsAdapt Fl lam abar) ≤ δ ∧
    (Z.FloorKnown Fl → (∀ k, 0 ≤ Fl.ns k) → 0 ≤ dbar → 0 ≤ abar →
      0 ≤ Z.epsSkip Fl lam ∧ 0 ≤ Z.epsCert Fl lam ∧ 0 ≤ Z.epsRoute Fl lam ∧ 0 ≤ Z.epsMand Fl lam ∧
        0 ≤ Z.epsLuck Fl lam dbar ∧ 0 ≤ Z.epsAdapt Fl lam abar) ∧
    H.time = Z.gammaFloor H Fl * Z.credit Fl

/-- **Corollary D1 (one binding facet).** If honest binds `r*` and every menu costs at least honest's on `r*`
(`w*(a_{r*}) = w_H`, `φ* = φ_H`, `χ* = χ_H`), then `ε_machine(a_{r*}) = 0`. -/
def BindingFacetD1 : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [DecidableEq R] (Z : Model K R T L U F X) (H : Honest K) (r : R),
    Z.G.load H.prog.n r = Z.bnd H.prog → 0 < Z.bnd H.prog →
    Z.wStar (onFacet r) = Z.G.price (onFacet r) H.prog.nw / Z.Nw →
    Z.phiStar (onFacet r) = Z.G.price (onFacet r) H.prog.nf / Z.Nw →
    Z.chiStar (onFacet r) = Z.G.price (onFacet r) H.prog.nx →
    Z.epsMachine H.prog (onFacet r) = 0

/-- **Corollary D1 (the vector route's threshold).** On a facet whose MMA coefficient is `α = a_{r,κ}` and whose
cheapest vector coefficient is `β > 0`, the vector route `S_V(q)` costs at least `T_κ` with operand delivery `φ_del`
iff `q ≥ 32(α + β·φ_del)/(β·ω_κ)` (2.029 on JI3e_0, 2.022 on the refit; 11.29 on JI3e_1, 22.45 on JI3e_2). -/
def VectorThresholdD1 : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    (Z : Model K R T L U F X) (t : T) (phiDel : ℝ) (r : R), 0 < Z.vecMin r → 0 < Z.omega t →
    (Z.tensorDemandDel t phiDel r ≤ Z.demand .vector r ↔
      32 * (Z.G.a r (Z.cls t) + Z.vecMin r * phiDel) / (Z.vecMin r * Z.omega t) ≤ Z.q)

/-- **Corollary D2 (routes off `r*`).** If at `a_{r*}` the run's load is at least `B₀ − Σ_u N_u s_u` for the words
`N_u ≥ 0` it puts on routes `u`, each saving `s_u ≥ 0` a word, and a facet `r_u` caps route `u` at `κ_u ≥ 0` words a
cycle (`N_u ≤ κ_u · a_{r_u}·n`), then `B(M) ≥ B₀/(1 + Σ_u κ_u s_u)`. -/
def RoutesOffD2 : Prop :=
  ∀ {K R ι : Type} [Fintype K] [Fintype R] [Fintype ι] (G : Machine K R) (n : K → ℝ) (rs : R) (cap : ι → R)
    (N s κ : ι → ℝ) (B₀ : ℝ),
    (∀ u, 0 ≤ N u) → (∀ u, 0 ≤ s u) → (∀ u, 0 ≤ κ u) →
    B₀ - ∑ u, N u * s u ≤ G.load n rs → (∀ u, N u ≤ κ u * G.load n (cap u)) →
    B₀ / (1 + ∑ u, κ u * s u) ≤ G.bound n

/-- **T1 (tensor slot counting).** A program that commits `N` fresh words, each read whole by the fold from its own
slot write of a tensor instruction (freshness makes the map injective), holds the tensor pipe at least `N/ρ_T`
cycles: `Σ_i c(i) n_i ≥ N/ρ_T`. -/
def TensorSlotCount : Prop :=
  ∀ (ι : Type) [Fintype ι] [Nonempty ι] (w c n : ι → ℕ) (N : ℕ) (slot : Fin N → SlotWrite ι w n),
    (∀ i, 0 < c i) → Function.Injective slot → (N : ℝ) / rhoT w c ≤ ∑ i, (c i : ℝ) * n i

/-- **T1's tightness.** Honest writes `w_H` slots in `c_H` cycles (`ρ_H = w_H/c_H ≤ ρ_T`) and commits one word in
every `L/d_H` of them (`d_H ≤ L`), so `ε_tensor ≤ 1 − (ρ_H/ρ_T)(d_H/L)`; it is `0` iff `L = d_H` and `ρ_H = ρ_T`. -/
def TensorSlotTight : Prop :=
  ∀ ρH ρT dH L : ℝ, 0 < ρH → ρH ≤ ρT → 0 < dH → dH ≤ L →
    (1 - ρH / ρT * (dH / L) = 0 ↔ L = dH ∧ ρH = ρT)

/-- **Lemma 2 (output count).** In a straight-line program whose class-`k` instances write at most `ω_k` words each,
the owned committed positions (one per head write their values trace to) are at most `Σ_k ω_k n_k`. -/
def OutputCount : Prop :=
  ∀ {K : Type} [Fintype K] [DecidableEq K] (ω : K → ℕ) (G : ℕ) (cls : Fin G → K) (Nc : ℕ)
    (head : Fin Nc → HeadWrite ω G cls),
    ((Finset.univ.image head).card : ℝ) ≤ ∑ k, (ω k : ℝ) * classCount cls k

end Accounting.Guarantees.TS
