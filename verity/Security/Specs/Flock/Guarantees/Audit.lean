import Definitions.Flock

/-!
# The audit's guarantees (trusted)

The guarantees of `Proofs.Flock.Soundness.Audit` that a document outside the package cites: the draw laws' escapes
(`Law`), the sizing of record, the work audit's floors, the per-call split at the sizing of record, and exfiltration with
the location term. Each keeps the text and the docstring of the theorem it was in its source module, with its section's
variables bound in the order the theorem took them; its proof is `Flock.SecurityProofs.Audit.G`, that theorem under its
old name.
-/

namespace Flock.Guarantees.Audit

open FlockSoundness FlockSoundness.Audit FlockSoundness.Game Finset
open scoped ENNReal

/-- **The closure draw's escape** is the base law's escape of `B` with the units it makes unsound. -/
def ClosureEscape : Prop :=
  ∀ {n : ℕ} (L : Law n) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)),
    (L.closure cl).escape B = L.escape (B ∪ Law.unsoundTiles cl B)

/-- **A wrong unit's whole harm is in the unsound work**, whether or not a draw reaches it: it is wrong itself, and every
unit whose closure holds it is unsound. -/
def HarmLeUnsoundWork : Prop :=
  ∀ {n m : ℕ} (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) {B : Finset (Fin n)} {u : Fin n},
    u ∈ B → Law.harm σ w cl u ≤ Law.unsoundWork σ w cl B

/-- **X-SPC-106: the per-call split covers the window's work**, when each call's work is in one stratum. -/
def CoversWindow : Prop :=
  ∀ {n m C : ℕ} (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ),
    0 < Law.totalWork σ w → (∀ s, 0 < w s → Law.callWork σ call w (call s) = w s * (Law.stratum σ s).card) →
    Law.Covers σ (Law.windowK σ call w f K) w K

/-- **The sizing of record** (Daniel, Sep 29: `δ = 2^-40`, `ε = 0.1%`): `(1 − 1/1000)^27713 ≤ 2^-40`. -/
def RecordSizing : Prop := ((999 : ℝ≥0∞) / 1000) ^ 27713 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹

/-- One draw fewer does not meet it under the work bound: `(1 − 1/1000)^27712 > 2^-40`. -/
def RecordSizingOneFewer : Prop := ¬ ((999 : ℝ≥0∞) / 1000) ^ 27712 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹

/-- **The stratified count curve.** A count vector `a` that some `τ ∈ (0, 1]` separates, every factor it takes at least
`τ` and every factor it leaves at most `τ`, attains the stratified law's `miss` at `∑ a`: no set of at least `∑ a` wrong
units escapes more often than the sets with counts `a`, which escape with `∏ s C(n_s − a_s, k_s)/C(n_s, k_s)`. A greedy
fill, taking the largest factor left one unit at a time, has such a `τ`. -/
def StratifiedMissEqGreedy : Prop :=
  ∀ {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card) (a : Fin m → ℕ),
    (∀ s, a s ≤ (Law.stratum σ s).card) → ∀ {τ : ℝ≥0∞}, 0 < τ → τ ≤ 1 →
    (∀ s, 0 < a s → τ ≤ Law.stratumRatio σ k s (a s - 1)) →
    (∀ s, a s < (Law.stratum σ s).card → Law.stratumRatio σ k s (a s) ≤ τ) →
    (Law.stratified σ k hk).miss (∑ s, a s) = ∏ s, Law.stratumEscape σ k s (a s)

/-- **The work bound.** Under the work law, a set of units doing `W_B` of the population's work `W` escapes with
probability at most `((W − W_B)/W)^K`. So wrong units doing a fraction `ε` of the work escape with at most `(1 − ε)^K`. -/
def WorkEscapeLe : Prop :=
  ∀ {n m : ℕ} (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ), (∀ s, 1 ≤ f s) →
    0 < Law.totalWork σ w → ∀ B : Finset (Fin n),
    (Law.work σ w f K).escape B ≤
      (((Law.totalWork σ w - Law.workOf σ w B : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K

/-- **The integrity floor, audited** (oracle layer): a wholly wrong non-empty stratum, one that does no PoUW work
included, is caught: the audit accepts while all of it is wrong with probability at most `ε_ks + δ_link`. -/
def WorkWholeStratum : Prop :=
  ∀ {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ} {Reg : Type}
    {session : Finset (Fin n) → Reg → Game Bool} (A : Analysis (Law.work σ w f K) Reg session) {εks δlink : ℝ≥0∞},
    A.KnowledgeSound εks → A.LinkSound δlink → ∀ (s : Fin m), 1 ≤ f s → (Law.stratum σ s).Nonempty →
    ∀ σ' : Strategy (audit (Law.work σ w f K) Reg session),
    prob (fun o => o.1 = true ∧ Law.stratum σ s ⊆ A.wrong (A.committedOf σ'))
        (audit (Law.work σ w f K) Reg session) σ' ≤ εks + δlink

/-- **The per-stratum floor, audited** (oracle layer): the audit accepts while at least `b` units of a non-empty stratum
`s` are wrong with probability at most `((n_s − b)/n_s)^(f_s) + ε_ks + δ_link`. -/
def WorkFloor : Prop :=
  ∀ {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ} {Reg : Type}
    {session : Finset (Fin n) → Reg → Game Bool} (A : Analysis (Law.work σ w f K) Reg session) {εks δlink : ℝ≥0∞},
    A.KnowledgeSound εks → A.LinkSound δlink → ∀ (s : Fin m), (Law.stratum σ s).Nonempty → ∀ (b : ℕ)
    (σ' : Strategy (audit (Law.work σ w f K) Reg session)),
    prob (fun o => o.1 = true ∧ b ≤ ((A.wrong (A.committedOf σ')).filter (σ · = s)).card)
        (audit (Law.work σ w f K) Reg session) σ' ≤
      ((((Law.stratum σ s).card - b : ℕ) : ℝ≥0∞) / ((Law.stratum σ s).card : ℝ≥0∞)) ^ (f s) + εks + δlink

/-- **The per-call split with y's floors at the sizing of record** (#364; X-SPC-105 and X-SPC-106): every call drawn by
its own work law at its share of `K = 27,713`, each stratum's floor covering `v` at `K_y = 27,713`, with closures. -/
def WindowSplitOfRecord : Prop :=
  ∀ {n m : ℕ} {σ : Fin n → Fin m} {cl : Fin n → Finset (Fin n)} {Reg : Type}
    {session : Finset (Fin n) → Reg → Game Bool} {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ}
    (A : Analysis ((Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).closure cl)
      Reg session) {εks δlink : ℝ≥0∞}, A.KnowledgeSound εks → A.LinkSound δlink →
    0 < Law.totalWork σ w →
    (∀ s, 0 < w s → Law.callWork σ call w (call s) = w s * (Law.stratum σ s).card) →
    0 < Law.totalWork σ v → (∀ s, 27713 * v s * (Law.stratum σ s).card ≤ f s * Law.totalWork σ v) →
    ∀ σ' : Strategy (audit ((Law.stratified σ (Law.windowK σ call w f 27713)
      (Law.windowK_le σ call w f 27713)).closure cl) Reg session),
    prob (fun o => o.1 = true ∧ (Law.totalWork σ w ≤ 1000 * Law.unsoundWork σ w cl (A.wrong (A.committedOf σ')) ∨
        Law.totalWork σ v ≤ 1000 * Law.unsoundWork σ v cl (A.wrong (A.committedOf σ'))))
        (audit ((Law.stratified σ (Law.windowK σ call w f 27713)
          (Law.windowK_le σ call w f 27713)).closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + δlink

/-- **The location term**: when `miss (K + 1) < δ`, the sets escaping with probability at least `δ` hold at most `K`
units, so there are at most `∑_{j ≤ K} C(n, j)` of them. -/
def CardAdmissibleLe : Prop :=
  ∀ {n : ℕ} {L : Law n} {δ : ℝ≥0∞} {K : ℕ}, L.miss (K + 1) < δ →
    (univ.filter fun B : Finset (Fin n) => δ ≤ L.escape B).card ≤ ∑ j ∈ range (K + 1), n.choose j

/-- **Exfiltration, with the location term**: a prover that picks its strategy `σ m` from a uniform message `m`, and a
receiver `g` that sees only what is delivered on `D`, succeed together with probability at most
`2^H · ∑_{j ≤ k} C(n, j) / |M| + δ + ε_ks + δ_link + δ_in`. Here `H` is a harm bound at `δ` of each unit's width `hb` on
a separator (the value bits) and `miss (k + 1) < δ` (which units are wrong). With `|M| = 2^ℓ`, the first term is
`2^(H + log₂ ∑_{j ≤ k} C(n, j) − ℓ)`. -/
def Exfiltration : Prop :=
  ∀ {C : Circuit} {n : ℕ} (P : Partition C n) {L : Law n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}
    {X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)}
    {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
    {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞} {D : Finset (Fin C.N)} {M : Type} [Fintype M] [Nonempty M]
    [DecidableEq M],
    (P.analysis X ext).KnowledgeSound εks → (P.analysis X ext).LinkSound δlink →
    Partition.AnchorsSound X a δin → D ⊆ P.committed → ∀ (hb : Fin n → ℕ) (K : Finset (Fin n) → Finset (Fin C.N)),
    (∀ B, K B ⊆ P.committed) → (∀ B, P.Separates (K B) B D) → (∀ B, (K B).card ≤ ∑ u ∈ B, hb u) →
    ∀ {δ : ℝ≥0∞} {H k : ℕ}, IsHarmBound L (fun u => (hb u : ℝ)) δ H → L.miss (k + 1) < δ →
    ∀ (σ : M → Strategy (audit L Reg session)) (g : (D → Bool) → M),
    avg (fun m => prob (fun o => o.1 = true ∧ g (Partition.out D (X (reg (σ m)) (cont (σ m)))) = m)
        (audit L Reg session) (σ m)) ≤
      ((2 ^ H * ∑ j ∈ range (k + 1), n.choose j : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
        (δ + εks + δlink + δin)

end Flock.Guarantees.Audit
