import Definitions.Accounting.TS

/-!
# The tensor–scalar model's named assumptions (trusted)

The premises of EPS1-SPEC's Theorem D-ideal ((I1)–(I8), §2.7) and of the compiler lemma (Lemma C), each a `Prop` over
`Accounting.TS`'s vocabulary, taken as a hypothesis, never an `axiom`. Each docstring opens with its claim id
(`verity.core.claims`). An assumption over an `Anatomy` is claimed at §0.4's and §1.3's labelling of the cheater's
instructions (heads, ownership, copies); an assumption over a run is claimed for the stated device.

Which form each takes:

* (I1) `FacetCeilings`: a device-model assumption, scoped by device (sm_120 RTX PRO 6000, rows VF3p-16 on VF3o-16),
  checked by measurement (Daniel, 8 Oct): a corpus program that runs faster than a facet falsifies it.
* (I2) and the uniform part of (I4): `WordsWritten`, the ISA's `ω` (P12). Lemma 2 (`Accounting.Guarantees.TS.OutputCount`)
  is its counting; the ISA fact that an instruction writes at most `ω` words is the assumption.
* (I3) `ScalarRoute`: SCALAR-ROUTE at its floors `(q, q_T, b_T)`. QSTAR's joint statement, which gave them from
  class P and a per-instruction cap `κ` (`q = 2.03`, `q_T = 1.22`, `b_T` from `κ`), rests on Theorem C, which is
  false as stated (FLOORS r3). The proved count is FLOORS' `Pouw.Guarantees.Eps1.ClassPSlots` (#2436): a class-P
  program whose writers each create at most `w` incidences costs at least `1 + (32 − w)/(65·d)` issue slots a word,
  `d` words a slot (`1.185` at `w = 8`, `d = 2`). No cap on `w` is proved, class P is a modeling restriction, and a
  table route lies outside `ClassPSlots`, so the floors stay an assumption; they are the model's parameters, so every
  statement here holds at every floor. ISA-CAP (`κ = 32/7`) is false (REFEREE Q4: an IDP.4A writes 5 incidences into
  one word, a signed IMAD.WIDE 8) and is no premise.
* (I5) `SideUnits` (measured per-word vectors, P11) and `NoSum` (NOSUM, P13), a measured statement: no unit outside
  the menu heads a fresh word.
* (I6) `FoldOpt` and `FillOpt` in the price form D-ideal's proof reads. FOLD-OPT's per-word kernel (the duo step
  `u = y ⊕ d ⊕ k`, `y' = u·a + z`, `z' = rotl(u, r) + y'` needs three instructions over the modeled ISA,
  `benchmarks/pouw/theory/fold_opt.py`, #2240) is a mechanical fact; the multi-round residual, that no program folds
  below the menu's price across rounds, is what stays assumed. `FanIn`, the ISA's operand counts, gives both at the
  fan-in menu's price (`Accounting.Guarantees.TS.FoldOptOfFanIn`, `FillOptOfFanIn`): for a fold or filler presented as
  a straight-line dataflow over the words it absorbs, priced at least the menu's routes of `a_k/m_k` a word.
* (I7) `LuckAt`: (L) at `p` with freshness (F1)–(F2), one bet a round. Split at the receipt
  (`Accounting.Guarantees.TS.LuckOfReceipt`), it is two luck premises:
  * the operand luck, which `OperandHMu` gives: (H-μ) at the deficient words (decision 110), through Lemma B at
    fibre 1;
  * `ReceiptLuckAt`, the receipt luck, which is HMU's B′ case.
  `OperandLuckCube` proves the operand luck at ¾ for degree-2 operands on cube bits. Its hypothesis restricts the
  cheater to affine-operand runs (RT-HEADLINE), so it is a proved route, not a premise.
* (I8) `ProofsSound`: deterministic computation, the circuit and inputs committed before σ, Theorem S's order, binding
  `δ_bind` and proof soundness `δ_ZK`, with Lemma E1(c) (every adaptive word lies below a committed value wrong for its
  own derivation). E1(c) is proved on paper; it sits inside this premise until the adaptive model is in Lean.
  `Accounting.Guarantees.TS.ProofsSoundOfPinned` derives it from a window's proofs (`Accounting.TS.ProofLayer`):
  binding at `δ_bind` (at `√δ_fork` from its fork form, `Accounting.Guarantees.TS.BindingOfFork`), proof soundness at
  `δ_ZK`, the Definitions' pinning, and `RightFixed`. For a sampled window, `Accounting.Guarantees.TS.ProofsSoundOfDrawn`
  derives it at `n_π = 1`, `δ_ZK = δ_ks`, from the drawn units' proofs (`Accounting.TS.DrawnLayer`): binding of every
  committed position before the coin that follows its commitment (at `|L|·√δ_fork`,
  `Accounting.Guarantees.TS.DrawnBindingOfFork`), the proofs' soundness jointly over a round's draws at `δ_ks`, each
  unit's pinning, the share covering every unit that isn't good (`DrawnLayer.Covers`, proved for each game), and
  `UnitFixed`. Its crypto half, the drawn units right, is `Accounting.Guarantees.TS.DrawnRightOff`.
* `RightFixed`: E1(c) with the rounds' equal shares, the part of (I8) that is the TS model's accounting rather than the
  proof service's.
* `UnitFixed`: E1(c) per unit, which doesn't read the draw: the good units' words are fixed, a unit being good when
  the run computed it and the units that write it values after its coin (not skipped and guessed), and the words of
  every unit not skipped whole are folded (its fold clause). A sampled window's proofs vouch for the drawn units only,
  so `RightFixed`, which asks for every committed value right, says nothing there; and a unit's check can pass by luck,
  so goodness reads the run's modes, not rightness.
* `LinSteer`: SAMPLING's LIN-STEER, for the compiler's closure.

The joint-luck premise JL is not used: D-ideal keeps the proved `ε_jl`.

`LinSteer` is false (`Accounting.Guarantees.TS.LinSteerSegments`: a segmented consumer satisfies it only when its
noise-active segments carry equal counts), and no guarantee rests on it; `Accounting.Guarantees.TS.SegmentSteer` is the
bound that holds.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Pouw.RelativeSecurity

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- `accounting-ts-facet-ceilings` (measured, device-scoped). **The measured facets are ceilings** for the run `P` on
the stated device (sm_120 RTX PRO 6000; Π_G = VF3p-16 on VF3o-16, decision 73): the run's counts and cycles are
`≥ 0` and its clock is in `(0, f_max]`, every SM-clocked facet's load is at most the run's cycles (S_r), and every
fixed-clock facet's load, priced at `f_max`, is at most `f_max` times the run's time. Rows take the best configuration
measured plus a 2% margin; a joint row is a facet on the mixes measured, and beyond them it is a red-team
obligation. -/
def FacetCeilings (G : Machine K R) (P : Run K) : Prop :=
  (∀ k, 0 ≤ P.n k) ∧ 0 ≤ P.cycles ∧ 0 < P.clock ∧ P.clock ≤ G.fmax ∧
    ∀ r, (G.smClocked r = true → G.load P.n r ≤ P.cycles) ∧
      (G.smClocked r = false → G.load P.n r ≤ G.fmax * P.time)

/-- `accounting-ts-words-written` (ISA, P12). **An instruction writes at most `ω` words**: the owned right fixed words
headed by writer kind `t` are at most `ω_t` times the instructions of `t`'s class the labelling gives them. An MMA
kind's `ω_t` counts one committed word a slot: design A's packed `T_1` words, two a s32 slot through int8 B entries
`s_j + 64·s_j′` (REFEREE S14), fit no route at the stated `ω` and floors (as writer words they exceed `ω`, as vector
words their ½–1 op of extraction is below `q`) and need their own (a side unit) or 2-bit entries. -/
def WordsWritten (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ t, A.words (.writer t) ≤ Z.omega t * A.nword (.writer t) (Z.cls t)

/-- `accounting-ts-scalar-route` (assumed; QSTAR's joint statement, from class P and a per-instruction cap `κ`).
**SCALAR-ROUTE**, in §0.6's units (lane-instructions a word, net of the mandatory fold and of operand delivery): the
gather-free vector-headed words cost at least `q` lane-instructions each (2.03, for `κ < 27.2` a slot), and the table
words served from level `l` at least `q_T` lane-instructions (1.22, for `κ ≤ 128`) and `b_T` bytes read from `l` each
(`32(28 − 1.03κ)/(c_L − κ)`, 3.0 at `κ = 8`). -/
def ScalarRoute (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  Z.q * A.words .vector ≤ 32 * ∑ k ∈ Z.vec, A.nword .vector k ∧
    ∀ l, Z.qT * A.words (.table l) ≤ 32 * ∑ k ∈ Z.vec, A.nword (.table l) k ∧
      Z.bT * A.words (.table l) ≤ A.nword (.table l) (Z.bytes l)

/-- `accounting-ts-side-units` (measured, P11). **A side unit's words cost its per-word vector**: the instructions the
labelling gives side unit `u`'s words are at least its per-word vector times its words, class by class. -/
def SideUnits (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ u k, A.words (.side u) * Z.side u k ≤ A.nword (.side u) k

/-- `accounting-ts-nosum` (measured, P13). **NOSUM**: no unit outside the menu heads a fresh committed word, so every
owned right fixed word has a head of the menu. -/
def NoSum (A : Anatomy K T L U) : Prop := A.fixed ≤ ∑ h, A.words h

/-- `accounting-ts-fold-opt` (assumed; its per-word kernel a mechanical fact). **FOLD-OPT_μ**: at every certificate,
the run's fold is priced at least `(1 − μ_f)·N_w·φ*(π)`. -/
def FoldOpt (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * Z.Nw * Z.phiStar lam ≤ Z.G.price lam A.nfold

/-- `accounting-ts-fill-opt` (assumed). **FILL-OPT_μ**: at every certificate, the run's filler is priced at least
`(1 − μ_x)·χ*(π)`. -/
def FillOpt (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muX) * Z.chiStar lam ≤ Z.G.price lam A.nfill

/-- `accounting-ts-fan-in` (ISA; register operands read from SASS on sm_80, sm_89, sm_90, sm_100 and sm_120). **An
instruction merges at most `m` components**: the nodes a straight-line dataflow issues as instruction `j` (its lanes,
or the one node of a warp-wide instruction) merge at most `m_{cls j}` components into one, `Σ (sources − 1)`. A lane
of LOP3, IADD3, IMAD, IDP.4A or FFMA reads three 32-bit registers (2 a lane, 64 a warp instruction); counted by
dependence its predicate and 64-bit operands count too (IADD3.X: three registers and two carry-ins, 4 a lane;
IMAD.WIDE: 3); a warp-wide instruction counts its whole operand set (SHFL 63; IMMA.16832: ten registers a thread, 319).
Memory is a node each access reads: a load merges its address and that node (1 a lane), a store or atomic its address,
its value registers and that node (STS.32 and ATOMS 2 a lane; STS.128 and STSM.x4 5 a lane, 160 a warp instruction),
so the classes that merge most per instruction are the wide stores. -/
def FanIn {ι W J : Type} [DecidableEq J] (D : Dataflow ι W) (ins : Fin D.n → J) (cls : J → K) (m : K → ℝ) :
    Prop :=
  ∀ j, (D.issued ins j : ℝ) ≤ m (cls j)

/-- `accounting-ts-luck` ([P+R] per noise; ¾ in Lean for affine operands; 0.854 in Lean for the pfill filler's cells
(`Accounting.Guarantees.FillerLuck.FillerLuckAt`), 0 for a filler's pinned receipts (`PinnedFillLuckAt`)). **(L) at
`p`, with freshness (F1)–(F2)**: the ideal game is one bet a round, a deficient round passing with probability at most
`p` given the past. -/
def LuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p : ℝ) (g : RoundGame Ω S) : Prop := RoundGame.LuckAt p g

/-- `accounting-ts-operand-h-mu` ((H-μ), decision 110, the claim Compose's `HMu` states per template; the headlines'
first luck premise). **A cheater's deficient words are unpredictable from its view**: in the ideal game split at the
receipt, at every round, whatever the past, the values of the round's deficient words have point mass at most `μ`
given everything the cheater's skip and its values for them are functions of (`ReceiptGame.OperandPointMass`, HMU's
`SkipRound.PointMass` at word level). The headlines take it at `¾`. Lemma B at fibre 1 (`Accounting.Guarantees.HMu.LemmaB`) turns it into the operand luck at
`μ`. Where it holds:
* the coins give it for a fresh-bit skipped value (`Accounting.Guarantees.HMu.CoinPointMass`: `½` at one bit);
* for degree-2 operands on cube bits `OperandLuckCube` gives the operand luck at `¾` directly, on affine-operand runs
  only;
* where the cheater computes a word that shares the skipped value's coins, it is computational: the cheater hasn't
  computed the product (`Models.Accounting.Guarantees.HMu`'s module doc).
A cheater whose view pins the skipped value, one that reads a coin and skips exactly when it is right, falsifies it.
The game names no view, so the premise is that some view that decides the share has the point mass. It implies the
operand luck at `μ` on each positive level of the share, and that implies it on levels of at least `1/μ` outcomes,
with the share itself as the view. -/
def OperandHMu {Ω : Type} [Fintype Ω] {S : ℕ} (μ : ℝ) (G : ReceiptGame Ω S) : Prop := G.OperandPointMass μ

/-- `accounting-ts-receipt-luck` (HMU's B′ case; the headlines' second luck premise, beside `OperandHMu`; on card
8b00536a). **The receipt luck at `pr`**: in the ideal game split at the receipt, at every round, whatever the past, a
deficient round whose deficient words are not all right keeps the honest committed value with probability at most
`pr` (`ReceiptGame.ReceiptLuck`). The headlines take it at `½`: LEDGER §1.3's case (b), two or more wrong places a set
(note:20261017T1723Z-draft-ledger), for DUO-CP-W and FX1-CP-W at `KS = 4`. Every real skip is in this case at
`KS = 4`, because one skipped MMA puts a wrong word into each of 4 chains of a set (PROJ r4,
note:20261018T0515Z-draft-proj-r4).

(H-μ) doesn't imply it. `Accounting.Guarantees.HMu.ReceiptLuckOfCoins` needs a fibre bound at the start state the
cheater knows, and for two or more places a set that bound fails (PROJ r5, note:20261018T1026Z-draft-proj-r5): two
words of a chain at the served keys, three or more at every key. Case (a), one deficient word a unit, gives 0
(`Accounting.Guarantees.TS.ReceiptLuckCaseA`), but it restricts the cheater, so it is no premise.

The evidence (PROJ r5): no sure-win cancellation at the served keys when the wrong words sit in a set's last two
places (`Pouw.SecurityProofs.Receipt.DuoPairXorNoKernel` and `Fx1PairXorNoKernel`), and a sweep at places 1–2 with no
counterexample. A per-state fibre bound for two or more places would retire it. -/
def ReceiptLuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (pr : ℝ) (G : ReceiptGame Ω S) : Prop := G.ReceiptLuck pr

/-- `accounting-ts-proofs-sound` (binding H-μ⁺, proof soundness, the protocol's order, Lemma E1(c)). **The proofs are
sound up to `S·(δ_bind + n_π·δ_ZK)`**: outside an event of that probability, an accepted window is accepted by the
ideal game, and its owned right fixed words per unit are at least `(1 − Σ_w d_w / S)·N_w` (no committed value is wrong
for its derivation, so every word that isn't fixed is in its round's share; the rounds carry equal shares of the
window's words). -/
def ProofsSound {Ω : Type} [Fintype Ω] {S : ℕ} (P : Play Ω S K T L U) (Nw δbind δzk : ℝ) (nπ : ℕ) : Prop :=
  ∃ E : Paths Ω S → Prop, prob E ≤ S * (δbind + nπ * δzk) ∧
    ∀ x, P.accepts x → ¬ E x → P.game.accepts x ∧ (1 - P.game.deficiency x / S) * Nw ≤ (P.run x).fixed

/-- `accounting-ts-lin-steer` (SAMPLING; false, `Accounting.Guarantees.TS.LinSteerSegments` and `LinSteerFalse`).
**LIN-STEER**: zeroing a fraction `φ = |Z|/|act|` of the row's noise-active coordinates saves at most `φ` of the consumer's work. -/
def LinSteer {m : ℕ} (c : Consumer m) : Prop :=
  ∀ Z ⊆ c.act, c.cost ∅ - c.cost Z ≤ (Z.card : ℝ) / c.act.card * c.cost ∅

/-- `accounting-ts-right-fixed` (Lemma E1(c), proved on paper; the rounds' equal shares). **With every committed value
right, what isn't fixed is in the shares**: on every accepted path whose committed values (each round's, read before its
σ, `ProofLayer.committed`) are right for their derivations, the run's owned right fixed words per unit are at least
`(1 − Σ_w d_w / S)·N_w`. An adaptive word lies below a committed value wrong for its own derivation (E1(c)), so with none
wrong every word that isn't fixed is in its round's share, and the rounds carry equal shares of the window's words.
`ProofsSound` held this inside it; it stays an assumption until the adaptive model is in Lean. -/
def RightFixed {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (P : Play Ω S K T L U) (pf : ProofLayer Ω S nπ V)
    (Nw : ℝ) : Prop :=
  ∀ x, P.accepts x → pf.right (pf.committed x) → (1 - P.game.deficiency x / S) * Nw ≤ (P.run x).fixed

open Classical in
/-- `accounting-ts-unit-fixed` (Lemma E1(c) per unit with its fold clause, proved on paper; the rounds' equal shares).
**The good units' words are fixed, and a unit not skipped whole has its words folded**: for a layer whose positions'
writers read them (`DrawnLayer.WriterReads`) and whose cuts are no later than the protocol's commitments, under the
run's modes `mode` (a unit is paid only if the run computed it, from the operands it committed, under its noise), on
every accepted path:
* the run's owned right fixed words per unit are at least the words `wt w j` of the good units over `S`. Unit `j` of
  round `w` is good (`DrawnLayer.Good`, decision 119) when the run computed it and every unit that writes it a value
  after the coin its noise reads (`DrawnLayer.Late`, transitively; under decision 107, the K-chunks before it in its
  running sum), none of them skipped and guessed. Values committed before a unit's coin are fixed when its noise is
  drawn, and a late value from a computed unit is the honest function of values fixed before it, so a good unit's
  words are its own and none is adaptive (E1(c): an adaptive word lies below a committed value wrong for its own
  derivation). A skipped unit whose check passes by luck (`PouwWarpCheck_v2` compares its receipt's checkpoints, not
  its words) isn't credited: its words are guesses, and the luck term pays for it;
* the fold clause: the run's folded words `folded x` per unit are at least the words of the units not skipped whole
  over `S`: a paid unit folds its words, a folded unit its committed ones, and only a whole unit skips its fold.

It reads neither the draw nor the proofs: the modes are the run's, and the proofs only vouch for the drawn units
(`Accounting.Guarantees.TS.DrawnRightOff`). With `Σ wt = S·N_w` and the share covering every unit that isn't good
(`DrawnLayer.Covers`), it gives `(1 − Σ_w d_w / S)·N_w` fixed words. It stays an assumption until the play's fixed and
folded words are defined from the committed values (the adaptive model). -/
def UnitFixed {C B : Type} {S m s : ℕ} {J Q Pos V : Type} [Fintype J] (P : Play (Paths C m × B) S K T L U)
    (pf : DrawnLayer C B S m s J Q Pos V) (mode : Paths (Paths C m × B) S → Fin S × J → Mode)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) : Prop :=
  pf.WriterReads → ∀ x, P.accepts x →
    (∑ w, ∑ j, if pf.Good mode x w j then wt w j else 0) / S ≤ (P.run x).fixed ∧
      (∑ w, ∑ j, if mode x (w, j) = .whole then 0 else wt w j) / S ≤ folded x

end Accounting.Assumptions.TS
