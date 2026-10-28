import Definitions.Accounting.TS

/-!
# The tensor–scalar model's named assumptions (trusted)

The premises of EPS1-SPEC's luck and proof-soundness steps ((I7) and (I8), §2.7), each a `Prop` over
`Accounting.TS`'s vocabulary, taken as a hypothesis, never an `axiom`. Each docstring opens with its claim id
(`verity.core.claims`). An assumption over a play is claimed at §0.4's and §1.3's labelling of the cheater's
instructions (heads, ownership, copies).

Which form each takes:

* `FanIn`: the sources the ISA lets an instruction read (every form, guarded), measured. No guarantee takes it as a
  premise: it gives the `f` that `Accounting.Guarantees.TS.FoldFanIn` reads.
* (I7) `LuckAt`: (L) at `p` with freshness (F1)–(F2), one bet a round. Split at the receipt
  (`Accounting.Guarantees.TS.LuckOfReceipt`), it is two luck premises:
  * the operand luck, which `OperandHMu` gives: (H-μ) at the deficient words, through Lemma B at fibre 1;
  * `ReceiptLuckAt`, the receipt luck, which is HMU's B′ case.
  `WholeLuckAt` is B′ at a unit whose fold the run skips, the whole-unit luck at `p₀` (`½`, B′'s value).
  `OperandLuckCube` proves the operand luck at ¾ for degree-2 operands on cube bits. Its hypothesis restricts the
  cheater to affine-operand runs, so it is a proved route, not a premise.
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
* `BindingForkAt`, `ProofSoundAt`, `ProofsPin`: `ProofsSoundOfPinned`'s proof-service pieces as named premises, binding
  in its fork form, so the headlines' binding is `√δ_fork` and no advantage bound.
* `UnitFixed`: E1(c) per unit, which doesn't read the draw: the good units' words are fixed, a unit being good when
  the run computed it and the units that write it values after its coin (not skipped and guessed), and the words of
  every unit not skipped whole are folded (its fold clause). A sampled window's proofs vouch for the drawn units only,
  so `RightFixed`, which asks for every committed value right, says nothing there; and a unit's check can pass by luck,
  so goodness reads the run's modes, not rightness.
* `LinSteer`: SAMPLING's LIN-STEER, for the compiler's closure.

`LinSteer` is false (`Accounting.Guarantees.TS.LinSteerSegments`: a segmented consumer satisfies it only when its
noise-active segments carry equal counts), and no guarantee rests on it; `Accounting.Guarantees.TS.SegmentSteer` is the
bound that holds.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Pouw.RelativeSecurity

variable {K T L U : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U]

/-- `accounting-ts-fan-in` (ISA; every form nvdisasm decodes on sm_80, sm_89, sm_90, sm_100 and sm_120, counted by
dependence). **An instruction merges at most `m` components**: the nodes a straight-line dataflow issues as instruction
`j` (its lanes, or the one node of a warp-wide or uniform instruction) merge at most `m_{cls j}` components into one,
`Σ (sources − 1)`. A node's sources are every value it depends on: each register or uniform-register word it reads (a
pair is two), each predicate it reads (an operand, a carry-in, its guard), a constant bank (`c[..][..]`, and
`cx[UR][..]`'s uniform register besides), the memory node of each access, and, since any instruction can carry a guard
and keeps what it writes when the guard is false, every word and predicate it writes (its destination, carry-outs and
predicate destinations). `cls` is the census's class: an MMA form's family and shape; a texture, surface, TMA, bulk or
tensor-memory access's unit and width; else the opcode's (`loads.class_of`), a memory access the census doesn't name in
its space's class, and every other form in NOROW. Each class at its widest form, a lane, on sm_100 and sm_120: ALU 8
(IADD3.X: three registers, two carry-ins, the guard, the kept destination and two kept carry-outs), IMAD 6, IMAD.WIDE 8,
IADD.64 8, LEA 6, FFMA 4 (sm_100's FFMA2 8), HFMA2, IDP.4A and F2FS 4, VIMNMX 6 (sm_120's IMNMX.U64 10: three pairs,
two predicate operands, two kept predicate destinations and the guard), SEL 4 (sm_120's SEL.64 7), XU 4 (BAR.RED.POPC,
the barrier a node), DFMA and DADD 8, DMUL 6, a constant load 5 (LDC.64 from `cx[UR][R]`), NOROW 9 (P2R: two
registers, P0–P6 and the kept destination). sm_80, sm_89 and sm_90 read a `cx[UR][imm]` operand where the others read a
register, one source more: ALU 9, IMAD 7, IMAD.WIDE 9, LEA 7, FFMA, HFMA2, IDP.4A, F2FS, SEL and VIMNMX 5 (sm_90's
VIMNMX.U32 7), NOROW 10 (P2R). Memory, a lane: a shared load or store 7 (LDS.128, STS.128), a global load 15
(LDG.E.ENL2.256 with its descriptor, address pair, predicate operand and predicate destination), a global or generic
store 13 (STG.E.ENL2.256), a shared atomic 15 (ATOMS.CAS.128), a global or generic atomic 18 (ATOMG.E.CAS.128), a
reduction 9 (REDG.E.ADD.F32x4), LDGSTS 9, LDSM and STSM 7, a texture or surface access 20. A warp-wide instruction
counts its lanes together: SHFL 191; UNI's R2UR 96 and VOTEU 65; an MMA form its A, B and C words, the kept
D, a guard a lane and its UP operand, with a block-scaled form's two scale registers a lane and uniform register and a
sparse form's metadata register a lane (IMMA.16832 480, QMMA.SF.16832 545); sm_100's tensor-memory load or store its
registers on 32 lanes, which it keeps under one uniform guard, with the guard, address and node (LDTM.32x32b.x128
4098). A uniform one counts once: UNI64's UIADD3.64.X 12, a TMA or bulk copy 15. -/
def FanIn {ι W J : Type} [DecidableEq J] (D : Dataflow ι W) (ins : Fin D.n → J) (cls : J → K) (m : K → ℝ) :
    Prop :=
  ∀ j, (D.issued ins j : ℝ) ≤ m (cls j)

/-- `accounting-ts-luck` ([P+R] per noise; ¾ in Lean for affine operands; 0.854 in Lean for the pfill filler's cells
(`Accounting.Guarantees.FillerLuck.FillerLuckAt`), 0 for a filler's pinned receipts (`PinnedFillLuckAt`)). **(L) at
`p`, with freshness (F1)–(F2)**: the ideal game is one bet a round, a deficient round passing with probability at most
`p` given the past. -/
def LuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p : ℝ) (g : RoundGame Ω S) : Prop := RoundGame.LuckAt p g

/-- `accounting-ts-operand-h-mu` ((H-μ), the claim Compose's `HMu` states per template; the headlines' first luck
premise). **A cheater's deficient words are unpredictable from its view**: in the ideal game split at the receipt, at
every round, whatever the past, the values of the round's deficient words have point mass at most `μ` given everything
the cheater's skip and its values for them are functions of (`ReceiptGame.OperandPointMass`, HMU's `SkipRound.PointMass`
at word level). The headlines take it at `¾`. Lemma B at fibre 1 (`Accounting.Guarantees.HMu.LemmaB`) turns it into the
operand luck at `μ`. Where it holds:
* the coins give it for a fresh-bit skipped value (`Accounting.Guarantees.HMu.CoinPointMass`: `½` at one bit);
* for degree-2 operands on cube bits `OperandLuckCube` gives the operand luck at `¾` directly, on affine-operand runs
  only;
* where the cheater computes a word that shares the skipped value's coins, it is computational: the cheater hasn't
  computed the product (`Definitions.Accounting.Guarantees.HMu`'s module doc).
A cheater whose view pins the skipped value, one that reads a coin and skips exactly when it is right, falsifies it.
The game names no view, so the premise is that some view that decides the share has the point mass.
Implied by (H-μ) at the cheater's real view, and weaker: the view and the skipped value are existential.
It implies the operand luck at `μ` on each positive level of the share, and that implies it on levels of at least
`1/μ` outcomes, with the share itself as the view. -/
def OperandHMu {Ω : Type} [Fintype Ω] {S : ℕ} (μ : ℝ) (G : ReceiptGame Ω S) : Prop := G.OperandPointMass μ

/-- `accounting-ts-receipt-luck` (HMU's B′ case; the headlines' second luck premise, beside `OperandHMu`;
named and conjectured). **The receipt luck at `pr`**: in the ideal game split at the receipt, at every round, whatever the past, a
deficient round whose deficient words are not all right keeps the honest committed value with probability at most
`pr` (`ReceiptGame.ReceiptLuck`). The headlines take it at `½`: LEDGER §1.3's case (b), two or more wrong places a set,
for DUO-CP-W and FX1-CP-W at `KS = 4`. Every real skip is in this case at
`KS = 4`, because one skipped MMA puts a wrong word into each of 4 chains of a set.

(H-μ) doesn't imply it. `Accounting.Guarantees.HMu.ReceiptLuckOfCoins` needs a fibre bound at the start state the
cheater knows, and for two or more places a set that bound fails: two
words of a chain at the served keys, three or more at every key. Case (a), one deficient word a unit, gives 0
(`Accounting.Guarantees.TS.ReceiptLuckCaseA`), but it restricts the cheater, so it is no premise.

The evidence (PROJ r5): no sure-win cancellation at the served keys when the wrong words sit in a set's last two
places (`Pouw.SecurityProofs.Receipt.DuoPairXorNoKernel` and `Fx1PairXorNoKernel`), and a sweep at places 1–2 with no
counterexample. A per-state fibre bound for two or more places would retire it. -/
def ReceiptLuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (pr : ℝ) (G : ReceiptGame Ω S) : Prop := G.ReceiptLuck pr

/-- `accounting-ts-receipt-luck` (HMU's B′ at a unit whose fold the run skips; at `p₀ = ½`, B′'s value, for live coins
only). **The whole-unit luck at `p₀`**: in the ideal game split at the receipt, every round's unfolded share (the share
of its words whose fold the run skips, fixed before the round's coins) lies in `[0, 1]`, and at every round, whatever
the past, a round with a positive unfolded share keeps the honest committed value with probability at most `p₀`
(`ReceiptGame.WholeLuck`). A unit whose fold is skipped commits a value for its receipt that no word it folded
determines, so its check passes only on a guess. The headlines take it at `½`. The whole-unit guess bound `2^-64` for a
64-bit fold state was proposed and isn't granted until an attack on its value fails, and none has been made. It fails if
the cheater's view pins the fold state without the fold: a state that a computed word, or a coin read before the skip,
determines. -/
def WholeLuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p0 : ℝ) (G : ReceiptGame Ω S) (Uf : Unfolded Ω S) : Prop :=
  G.WholeLuck p0 Uf

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
σ, `ProofLayer.committed`) are right for their derivations on that path's coins, the run's owned right fixed words per
unit are at least `(1 − Σ_w d_w / S)·N_w`. An adaptive word lies below a committed value wrong for its own derivation
(E1(c)), so with none wrong every word that isn't fixed is in its round's share, and the rounds carry equal shares of the
window's words.
`ProofsSound` held this inside it; it stays an assumption until the adaptive model is in Lean. -/
def RightFixed {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (P : Play Ω S K T L U) (pf : ProofLayer Ω S nπ V)
    (Nw : ℝ) : Prop :=
  ∀ x, P.accepts x → pf.right x (pf.committed x) → (1 - P.game.deficiency x / S) * Nw ≤ (P.run x).fixed

/-- `accounting-ts-proofs-sound` (binding in its fork form, as ruled; `cr/sha-512` bounds it at a finder that
runs the window twice from round `w`). **Binding's fork form at `δ_fork`**: at every round `w`, on a run and on its
continuation from round `w` on fresh coins, which share round `w`'s commitment, every proof of round `w` is accepted
and some proof opens round `w`'s leaves differently, with probability at most `δ_fork` over the pair
(`ProofLayer.Forked`; the rest of the window need not be accepted). Different openings of hm96 rows of
one frame-v3-sha512 tree are two inputs hashed to one SHA-512 digest along their paths. `BindingOfFork` gives binding
at `√δ_fork` from it. -/
def BindingForkAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) (δfork : ℝ) : Prop :=
  ∀ w, prob (fun p : Paths Ω S × Paths Ω S => pf.Forked w p.1 p.2) ≤ δfork

/-- `accounting-ts-proofs-sound` (proof soundness; C-Flock's `Flock.Guarantees.EndToEndHiddenDrawn` for a session).
**Proof soundness at `δ_ZK`**: each proof of the window is accepted with an opening that fails its relation on the path
(the round's coin and the words keyed by it that its proof gives there) with probability at most `δ_ZK` over the
window's coins (`ProofLayer.SoundAt`). C-Flock's bound is `t·2^-λ` for a prover making `t` hash calls, so it gives
`δ_ZK` only for a cheater whose hash calls over the window are capped. Its "wrong" is against the committed transcript
fixed at registration, and its statement leaves out the rows link: that the committed inputs and the program's outputs
on them are the rows the public file commits "is not stated here" (`Proofs/Flock/EndToEndHidden/Statement.lean`).
`ProofLayer.rel` reads exactly that link (the Definition with the leaves' rows among its outputs), so C-Flock's bound
doesn't discharge this premise until it states the link. -/
def ProofSoundAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) (δzk : ℝ) : Prop :=
  pf.SoundAt δzk

/-- `accounting-ts-proofs-pin` (the Definitions' pinning; conjectured, and no Lean checks it of the served relations).
**The relations pin every committed value** (`ProofLayer.Pins`): values that satisfy every proof's relation on a path
are right for their derivations on that path's coins. It is a property of the Definitions the window's proofs pass,
which no Lean here models. As `ProofLayer`'s module doc says, `PouwSegCheck_v2` lacks it: it ties only the drawn
segment's prefix-word set to the one before, so its window's other sets are free witness. `PouwSegCheck_v3`,
`PouwCallCheck_v1`, `PouwWarpCheck_v1` and `_v2` recompute every word they absorb, and nothing states that the
served relations are those. -/
def ProofsPin {Ω : Type} {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) : Prop := pf.Pins

open Classical in
/-- `accounting-ts-unit-fixed` (Lemma E1(c) per unit with its fold clause, proved on paper; the rounds' equal shares).
**The good units' words are fixed, and a unit not skipped whole has its words folded**: for a layer whose positions'
writers read them (`DrawnLayer.WriterReads`) and whose cuts are no later than the protocol's commitments, under the
run's modes `mode` (a unit is paid only if the run computed it, from the operands it committed, under its noise), on
every accepted path:
* the run's owned right fixed words per unit are at least the words `wt w j` of the good units over `S`. Unit `j` of
  round `w` is good (`DrawnLayer.Good`) when the run computed it and every unit that writes it a value after the coin
  its noise reads (`DrawnLayer.Late`, transitively; for a K-chunk, the K-chunks before it in its running sum), none of
  them skipped and guessed. Values committed before a unit's coin are fixed when its noise is drawn, and a late value
  from a computed unit is the honest function of values fixed before it, so a good unit's words are its own and none is
  adaptive (E1(c): an adaptive word lies below a committed value wrong for its own derivation). A skipped unit whose
  check passes by luck (`PouwWarpCheck_v2` compares its receipt's checkpoints, not its words) isn't credited: its words
  are guesses, and the luck term pays for it;
* the fold clause: the run's folded words `folded x` per unit are at least the words of the units not skipped whole
  over `S`: a paid unit folds its words, a folded unit its committed ones, and only a whole unit skips its fold.

It reads neither the draw nor the proofs: the modes are the run's, and the proofs only vouch for the drawn units
(`Accounting.Guarantees.TS.DrawnRightOff`). Under live coins a unit's coin is the live coin its noise reads; under
seeded (Fiat–Shamir) coins it is read per query, on the circuit's cut layer: the layer's coins are the seed answers the
prover queried, and a unit's coin the query its noise's seed answers (it may stay on that layer because it reads no
committed value). With `Σ wt = S·N_w` and the share covering every unit that isn't good (`DrawnLayer.Covers`), it gives
`(1 − Σ_w d_w / S)·N_w` fixed words. It stays an assumption until the play's fixed and folded words are defined from the
committed values (the adaptive model). -/
def UnitFixed {C B : Type} {S m s : ℕ} {J Q Pos V : Type} [Fintype J] (P : Play (Paths C m × B) S K T L U)
    (pf : DrawnLayer C B S m s J Q Pos V) (mode : Paths (Paths C m × B) S → Fin S × J → Mode)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) : Prop :=
  pf.WriterReads → ∀ x, P.accepts x →
    (∑ w, ∑ j, if pf.Good mode x w j then wt w j else 0) / S ≤ (P.run x).fixed ∧
      (∑ w, ∑ j, if mode x (w, j) = .whole then 0 else wt w j) / S ≤ folded x

end Accounting.Assumptions.TS
