import Definitions.Accounting.Assumptions.TS

/-!
# The tensor–scalar model's guarantee statements (trusted)

EPS1's statements over `Accounting.TS`,
each `def G : Prop`, proved (or a `sorry` stub) as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS`:

* `TimeBound`: Lemma 1, `T(P) ≥ B(P)/f_max`.
* `DecompositionD`: Theorem D(a), the master inequality from (Π), (RD) and (Lk).
* `DIdeal`, `DIdealFloor`: Theorem D-ideal, (2.7) and (2.7♭), in 𝒫_full from (I1)–(I8).
* `CompilerC`: Lemma C, the compiler, in 𝒫_samp: the same bound with `ε_sample` added.
* `ProofsSoundOfPinned`: (I8) `ProofsSound` from binding at `δ_bind` of the leaves each round commits before its σ,
  proof soundness at `δ_ZK` and the Definitions' pinning, each relation on its path's coins, and E1(c) (`RightFixed`);
  `BindingOfFork`: binding's failure squared is at most its fork form's, two runs off one commitment opening it
  differently, so `δ_bind = √δ_fork`.
* `DrawnRightOff`: a sampled window's drawn units are right at their committed values outside `S·(δ_bind + δ_ks)`, from
  binding at every position's cut, joint proof soundness and pinning; `ProofsSoundOfDrawn`: (I8) at `n_π = 1`,
  `δ_ZK = δ_ks`, from it, the share covering every unit that isn't good (a unit whose late cone the run computed, drawn
  at each unit's coin), and E1(c) per unit with its fold clause (`UnitFixed`); `DrawnBindingOfFork`: a round's binding
  failure squared is at most `|L|²` times the fork form at its positions' cuts, `L` the positions its units read, so
  `δ_bind = |L|·√δ_fork`.
* `FloorCreditD3`: Corollary D3, the credit at the floor, with every term `≥ 0` and `T(H) = γ_♭·T_cred`.
* `BindingFacetD1`, `VectorThresholdD1`: Corollary D1, one binding facet, and the vector route's threshold
  `q ≥ 32(α + β·φ_del)/(β·ω)`.
* `RoutesOffD2`: Corollary D2, routes off the binding facet.
* `TensorSlotCount`, `TensorSlotTight`: T1, the tensor slot count, and its tightness criterion.
* `OutputCount`: Lemma 2's count, owned committed positions at most `Σ_k ω_k n_k`.
* `FanInCount`, `SeesOfInjective`, `ThinFoldInjective`: the fan-in count, `|ι| ≤ |O| + Σ_i (sources − 1)` for a
  straight-line program whose outputs see every input; a receipt injective in each word sees it; the thin fold's is.
* `FoldOptOfFanIn`, `FillOptOfFanIn`: (I6) from `FanIn` at the fan-in menu's price.

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
(and its skippable work, `d̄` and `ā` are `≥ 0`) every term is `≥ 0`; and `T(H) = γ_♭·T_cred`. It holds at every ♭,
`π` and fold menu, so the fold's instructions a word `f` are no hypothesis. Its numbers (`ε^♭ ≤ 0.3%`) need
`f ≥ f* = 2.53` (REFEREE X1; `R_eff ≥ 1.69` for XOR-in): below it JI3e_1 binds honest, where `S_V` at `q = 8` (eight
IDP.4A a word) is cheaper than `T_κ`, and the terms reach 0.83–3.0% on 8B's linears at `R = 1`. -/
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
cycles: `Σ_i c(i) n_i ≥ N/ρ_T`. Injectivity is the premise (one word a slot): design A's packed `T_1` words, two a
s32 slot (REFEREE S14), are outside it. -/
def TensorSlotCount : Prop :=
  ∀ (ι : Type) [Fintype ι] [Nonempty ι] (w c n : ι → ℕ) (N : ℕ) (slot : Fin N → SlotWrite ι w n),
    (∀ i, 0 < c i) → Function.Injective slot → (N : ℝ) / rhoT w c ≤ ∑ i, (c i : ℝ) * n i

/-- **T1's tightness.** Honest writes `w_H` slots in `c_H` cycles (`ρ_H = w_H/c_H ≤ ρ_T`) and commits one word in
every `L/d_H` of them (`d_H ≤ L`), so `ε_tensor ≤ 1 − (ρ_H/ρ_T)(d_H/L)`; it is `0` iff `L = d_H` and `ρ_H = ρ_T`. -/
def TensorSlotTight : Prop :=
  ∀ ρH ρT dH L : ℝ, 0 < ρH → ρH ≤ ρT → 0 < dH → dH ≤ L →
    (1 - ρH / ρT * (dH / L) = 0 ↔ L = dH ∧ ρH = ρT)

/-- **(I8) from its pieces.** In 𝒫_full over a window's proofs (`ProofLayer`, `n_π` a round, each round's leaves those
it commits before its σ, its relations reading the path's coins and the words keyed by them) whose play is theirs
(`Of`: the window is accepted only if every proof is, and the ideal game's check is the proofs' relations at the
committed values on the path): if at every round binding fails with probability at most `δ_bind` (`BindingAt`; `Moved`:
an accepted round's proof opens its leaves to values other than those committed before σ_w), every proof is unsound
with probability at most `δ_ZK` (`SoundAt`), the relations pin every committed value on every path (`Pins`, a property
of the Definitions the proofs pass), and E1(c) holds with the equal shares (`RightFixed`), then (I8) `ProofsSound`
holds at `S·(δ_bind + n_π·δ_ZK)`. -/
def ProofsSoundOfPinned : Prop :=
  ∀ {K T L U V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] (Ω : Type) [Fintype Ω] (S nπ : ℕ)
    (P : Play Ω S K T L U) (pf : ProofLayer Ω S nπ V) (Nw δbind δzk : ℝ),
    pf.Of P → pf.BindingAt δbind → pf.SoundAt δzk → pf.Pins → RightFixed P pf Nw →
    ProofsSound P Nw δbind δzk nπ

/-- **Binding from the fork form.** For a window's proofs and a round `w`, the probability that binding fails at `w`
(`Moved`, against the values committed before σ_w), squared, is at most the probability that two runs off one
commitment, the second continuing from round `w` on fresh coins, are both accepted and open round `w`'s leaves
differently (`Forked`). So a bound `δ_fork` on the fork form, which `cr/sha-512` gives at a finder that runs the window
twice from round `w`, is binding at `δ_bind = √δ_fork`. -/
def BindingOfFork : Prop :=
  ∀ {V : Type} (Ω : Type) [Fintype Ω] (S nπ : ℕ) (pf : ProofLayer Ω S nπ V) (w : Fin S),
    prob (fun x => pf.Moved x w) ^ 2 ≤ prob (fun p : Paths Ω S × Paths Ω S => pf.Forked w p.1 p.2)

/-- **A sampled window's drawn units are right, outside binding and soundness failures.** Over `S` rounds whose coins
are `m` coins before the draws and the live draw coin with the proofs' coins, the draw naming `s` units a round, each
proved alone with its own noise's generation, its slots' values committed per position (`DrawnLayer`): if in every round
binding fails with probability at most `δ_bind` (`BindingAt`; `MovedIn`: an accepted proof of a drawn unit opens a slot
to a value other than its position's committed value, bound before the coin that follows its commitment, so a value
opened one way for the unit that wrote it and another way for a unit that reads it is a binding failure), the proofs
are sound jointly over each round's draws at `δ_ks` (`JointSound`) and each unit's relation pins its rightness
(`Pins`), then outside an event of probability at most `S·(δ_bind + δ_ks)` every drawn unit whose proof is accepted is
right at its committed values. It reads no play: what it gives is the drawn units' rightness, which the ideal game's
check reads (`ProofsSoundOfDrawn`) and a composition's folded credit takes as its premise. -/
def DrawnRightOff : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : DrawnLayer C B S m s J Q Pos V) (δbind δks : ℝ),
    pf.BindingAt δbind → pf.JointSound δks → pf.Pins →
    ∃ E : Paths (Paths C m × B) S → Prop, prob E ≤ S * (δbind + δks) ∧
      ∀ x, ¬ E x → ∀ w i, pf.accepted x w i → pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i))

/-- **(I8) for a sampled window, from its drawn units' proofs.** Over `DrawnRightOff`'s window whose positions' writers
read them (`WriterReads`, checked for each layer), with a play that is its drawn units' proofs' (`Of`: accepted only if
every drawn unit's proof is, and accepted by the ideal game when every drawn unit is right at its committed values),
unit weights `wt` summing to the window's words `S·N_w`, the run's modes `mode` and folded words `folded`, a share
that counts every unit that isn't good (`Covers`: good is the run having computed the unit's late cone; a property of
the share's definition, proved for each game; a share charging each unit the run didn't pay for its late cone's weight
`coneWt` has it) and E1(c) per unit with its fold clause (`UnitFixed`): if binding, joint soundness and pinning hold as
in `DrawnRightOff`, then (I8) `ProofsSound` holds at `n_π = 1`, `δ_ZK = δ_ks`, `S·(δ_bind + δ_ks)`. Its event is
`DrawnRightOff`'s, so an undrawn unit's values enter only through the ideal game and the share. -/
def ProofsSoundOfDrawn : Prop :=
  ∀ {K T L U J Q Pos V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype J] [Nonempty V]
    (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ) (P : Play (Paths C m × B) S K T L U)
    (pf : DrawnLayer C B S m s J Q Pos V) (mode : Paths (Paths C m × B) S → Fin S × J → Mode)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) (Nw δbind δks : ℝ),
    0 < S → ∑ w, ∑ j, wt w j = S * Nw → pf.WriterReads → pf.Of P → pf.BindingAt δbind →
    pf.JointSound δks → pf.Pins → pf.Covers P mode wt Nw → UnitFixed P pf mode wt folded →
      ProofsSound P Nw δbind δks 1

/-- **Binding from the fork form, at every position's cut.** For a sampled window's proofs, a round `w` and a set `L` of
positions holding every slot of round `w`'s units, the probability that binding fails in round `w` (`MovedIn`),
squared, is at most `|L|²` times the probability that, for some position of `L`, two runs off one commitment, the second
continuing from that position's cut on fresh coins, open it differently in accepted proofs of drawn units (`Forked`).
Each opening is an accepted proof's, and a proof exists only for a drawn unit (proof `i` of round `w` is drawn unit
`drawnAt x w i`'s), so accepted implies drawn by construction. A bound `δ_fork` on the fork form is binding at
`δ_bind = |L|·√δ_fork`; `δ_fork` itself is two steps a finder that runs the window twice from the cut takes: extracting
both proofs' openings with C-Flock's knowledge extractor, at `ε_ks` a proof, and then two openings of one committed
frame-v3-sha512 root at one position to different hm96 rows, a SHA-512 collision along their paths (`cr/sha-512`). -/
def DrawnBindingOfFork : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : DrawnLayer C B S m s J Q Pos V) (w : Fin S) (L : Finset Pos), (∀ j k, pf.pos w j k ∈ L) →
    prob (fun x => pf.MovedIn x w) ^ 2 ≤
      (L.card : ℝ) ^ 2 *
        prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2)

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

/-- **A receipt injective in each word sees it.** If every run's outputs `O` are `f` of its inputs, and `f` changes
whenever exactly one input does, the outputs see every input. -/
def SeesOfInjective : Prop :=
  ∀ {ι W : Type} (D : Dataflow ι W) (O : Finset (ι ⊕ Fin D.n)) (f : (ι → W) → O → W),
    (∀ x v, D.Runs x v → ∀ o : O, v o = f x o) →
    (∀ j x x', (∀ j', j' ≠ j → x j' = x' j') → x j ≠ x' j → f x ≠ f x') → D.Sees O

/-- **The thin fold's state is injective in every word of its run**, the state and the other words fixed (each step
is a bijection of the state and of each word it absorbs), and in the starting state. -/
def ThinFoldInjective : Prop :=
  ∀ (xor : Bool) (s : BitVec 32) (pre post : List (BitVec 32 × BitVec 32)) (w₁ w₁' w₂ w₂' : BitVec 32),
    (w₁ ≠ w₁' → thinFold xor s (pre ++ (w₁, w₂) :: post) ≠ thinFold xor s (pre ++ (w₁', w₂) :: post)) ∧
    (w₂ ≠ w₂' → thinFold xor s (pre ++ (w₁, w₂) :: post) ≠ thinFold xor s (pre ++ (w₁, w₂') :: post)) ∧
    ∀ s', s ≠ s' → thinFold xor s (pre ++ post) ≠ thinFold xor s' (pre ++ post)

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
