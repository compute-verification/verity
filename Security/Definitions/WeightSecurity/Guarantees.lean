module

public import Definitions.Assumptions.WeightSecurity
public import Definitions.WeightSecurity.Assumptions
public import Definitions.WeightSecurity.Faulted
public import Definitions.WeightSecurity.Toy
public import Definitions.WeightSecurity.ToyFaulted
public import Definitions.WeightSecurity.ToyServed

/-!
# Model weight security

`Theft`: if honest use of the served function doesn't hand out the weights (`HonestUseHard`, at `p` for `q` queries),
and what leaves the datacenter goes through a covert channel of `|S|` strings except with probability `ε`
(`LeaksThrough`), then the outsider steals the weights with probability at most `|S| · p + ε`. With `|S| = 2^𝒞`:
`Pr[steal] ≤ 2^𝒞 · p_hon(q) + ε`, so the covert capacity `𝒞` must approach `log₂ (1/p_hon(q))` before theft is likely.
The datacenter knows the weights and may do anything; the outsider is adaptive, and the covert string may arrive in
pieces between its queries and steer them.

The proof is the leakage lemma: for each covert string `s`, the outsider with `s` written in (`Outsider.withString`)
is an honest-use party that sees, when `s` is the datacenter's, exactly what the outsider sees, so the outsider's
success is at most the sum over `s` of theirs.

`TheftHonest` and `TheftNontrivial` are its witnesses, at the toy instance (`Definitions.WeightSecurity.Toy`), where a
lemma, `TheftTight` (`Security/Proofs/WeightSecurity/Lemmas.lean`), shows the factor `|S|` is attained.

`TheftServed` is `Theft` at an audited deployment (`Definitions.WeightSecurity.Served`). The channel is the served one:
advice, timing buckets, the overwrites the audit lets through and the halt. The leakage term is the probability `δ` that
the audit leaves `M` incorrect proof blocks undetected, plus the probability that a release isn't local, which the
network certifier and the dataflow discharge. Its witnesses
`TheftServedHonest` and `TheftServedNontrivial` are at the toy deployment (`Definitions.WeightSecurity.ToyServed`).

`TheftFaulted` is the same deployment under SPOT's fault model (`Definitions.WeightSecurity.Faulted`): a fault is an
input to the faulted circuit, not an overwrite of a release, so SPOT's part of the channel is the fault patterns, of
sizes summing to at most `M`, and the reach of a fault drops out. What faults at the patterns no cut covers do to the
weights' exposure is the named assumption `FaultedUseHard` (λ), on a typical set of weights; a pattern an honest cut
`κ` cuts (`Definitions.WeightSecurity.Cut`) costs the cut's hint in the channel instead, with no assumption.
`TheftServed` stays as the bound that takes no claim about the model. Its witnesses `TheftFaultedHonest` and
`TheftFaultedNontrivial` are at the toy deployment with faults (`Definitions.WeightSecurity.ToyFaulted`), at the cut
that cuts nothing.

`TheftServedReq` is `TheftServed` per charge unit, in bits: the released outputs grouped into units (a request, or a
request with the requests that read its state), each changed unit charged its whole output and its name, and the leakage
term the probability that an incorrect proof block changes more than `F` units, in its window or later ones, before the
first rejection (`Served.Isolated`). `TheftServedMinEntropy` is `TheftServedReq` against every party, where honest use
leaves the weights `K` bits of min-entropy: `2^(B - K)`. `TheftServedHonest` is the first's honest witness,
`TheftServedMinEntropyHonest` the second's, and each has a nontrivial witness at the leaking toy deployment.
-/

@[expose] public section

namespace WeightSecurity.Guarantees

open WeightSecurity WeightSecurity.Assumptions

/-- **Model weight security.** The weights are `gen ω` on uniform coins `ω`, and every other coin is one uniform `r`.
For every datacenter `real` and every outsider `O` making `q` queries against it: if every party of the class `A`
steals the weights from `q` honest queries with probability at most `p` (`HonestUseHard`), the outsider's view goes
through the channel `ch` except with probability `ε` (`LeaksThrough`), and `A` holds the outsider with each covert
string written in (`Simulable`), then `O` outputs a stolen model with probability at most `|ch.S| · p + ε`. -/
def Theft : Prop :=
  ∀ {Ω R Θ X Y M Θ' : Type} [Fintype Ω] [Fintype R] (C : Θ → X → Y) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop)
    (q : ℕ) (A : Set (Outsider R X Y Θ')) (p : ℝ) (real : Θ → R → List (X × M) → X → M) (O : Outsider R X M Θ')
    (ch : Channel R X Y M) (ε : ℝ),
    HonestUseHard C gen Stolen q A p → LeaksThrough C gen real O q ch ε → Simulable A O ch →
      Claims.Pr (fun z : Ω × R => Stolen (gen z.1) (O.model z.2 (real (gen z.1) z.2) q)) ≤
        Fintype.card ch.S * p + ε

/-- **`Theft`'s premises hold together at the honest toy datacenter**, not degenerately: one query of the toy's
served function steals the weights with probability at most `1/2` for every party, and the user's view of the honest
datacenter goes through the plain channel (one covert string) with `ε = 0`, which every class holds with that string
written in. So `Theft` bounds the theft at `1 · 1/2 + 0 = 1/2`, below `1`. -/
def TheftHonest : Prop :=
  HonestUseHard Toy.C Toy.gen Toy.Stolen 1 Toy.anyone 2⁻¹ ∧
    LeaksThrough Toy.C Toy.gen Toy.honestDC Toy.user 1 Toy.plain 0 ∧ Simulable Toy.anyone Toy.user Toy.plain

/-- **`Theft`'s bound fails once `LeaksThrough` does.** At the cheating toy datacenter, which appends the second
weight bit to each answer, the thief steals the weights on every coin. Through the silent channel (one covert string)
`HonestUseHard` at `1/2` and `Simulable` hold, `LeaksThrough` at `ε = 0` fails, and so does the bound `1 · 1/2 + 0`. -/
def TheftNontrivial : Prop :=
  HonestUseHard Toy.C Toy.gen Toy.Stolen 1 Toy.anyone 2⁻¹ ∧
    Simulable Toy.anyone Toy.thief Toy.silent ∧
    ¬ LeaksThrough Toy.C Toy.gen Toy.cheatDC Toy.thief 1 Toy.silent 0 ∧
    ¬ Claims.Pr (fun z : (Bool × Bool) × Unit =>
        Toy.Stolen (Toy.gen z.1) (Toy.thief.model z.2 (Toy.cheatDC (Toy.gen z.1) z.2) 1)) ≤
      Fintype.card Toy.silent.S * 2⁻¹ + 0

open scoped ENNReal in
/-- **Model weight security at an audited deployment that serves through the network certifier.** The weights are
`gen ω` on uniform coins `ω`, and every other coin is one uniform `r` or the audit's own. The deployment serves the
outsider `Ou`'s `T` queries one per window, and `run θ r` is its run at the weights `θ`, a game of the audit's coins
(`Served`): each window's release, then its verdict. The deployment halts `L` windows after the first rejection. Suppose:

* every party of the class `A` steals the weights from `T` honest queries with probability at most `p`
  (`HonestUseHard`);
* `A` holds `Ou` with each covert string of the served channel written in (`Simulable`): each window's advice and
  timing bucket, at most `E · M + (L + 1) · |O|` overwritten values and the halt;
* the audit leaves `M` incorrect proof blocks undetected with probability at most `δ` on every run
  (`Served.UndetectedLe`).

Then `Ou` outputs a stolen model with probability at most
$$|S| \cdot p + \delta + \Pr[\neg \mathrm{Local}],$$
`S` the served channel's covert strings. `Served.Local` says each window's release changes at most `E` slots per
incorrect proof block. The layers below the composition give it: the network certifier, that what leaves is the
trace's outputs at its bucket, and the dataflow, that the trace is honest outside the incorrect proof blocks' egress
cones. Neither layer reads the other. -/
def TheftServed : Prop :=
  ∀ {Ω R Θ Q Adv O V Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv] [Fintype O]
    [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick]
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (A : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L E M : ℕ) (δ : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served O V Tick Adv))),
    HonestUseHard C gen Stolen T A p →
    Simulable A Ou (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)) →
    Served.UndetectedLe M δ run →
      Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (run (gen z.1) z.2)) ≤
        ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S * p) +
          δ + Verity.Game.avg (fun z : Ω × R =>
            Verity.Game.value (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l) (run (gen z.1) z.2))

/-- **`TheftServed`'s premises hold together at the honest toy deployment**, with no halt lag, no slot changed per
incorrect proof block and the budget `1`. Its served function answers `0`, so two honest queries steal the weights with
probability at most `1/256` for every party. Every class holds the user with each covert string written in. The honest
run's audit leaves no incorrect proof block undetected. -/
def TheftServedHonest : Prop :=
  HonestUseHard ToyServed.C ToyServed.gen ToyServed.Stolen 2 ToyServed.anyone (1 / 256) ∧
    Simulable ToyServed.anyone ToyServed.user
      (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2 (0 * 1 + (0 + 1) * Fintype.card Unit)) ∧
    Served.UndetectedLe 1 0 ToyServed.honestRun

open scoped ENNReal in
/-- **`TheftServed`'s bound fails at the verifier that accepts everything.** The leaking toy deployment writes the
weights into its two releases, one incorrect proof block per window, and its audit accepts on an empty sample. The thief
then steals the weights on every coin, with every release local and the overwrite budget one value. The bound at
`p = 1/256` and `δ = 0` is below `1`, so it fails. -/
def TheftServedNontrivial : Prop :=
  ¬ Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
        (fun l => ToyServed.Stolen (ToyServed.gen z.1) (ToyServed.thief.out z.2 (Served.view ToyServed.thief 0 z.2 l 2)))
        (ToyServed.leakRun (ToyServed.gen z.1) z.2)) ≤
      ENNReal.ofReal (Fintype.card (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2
          (1 * 0 + (0 + 1) * Fintype.card Unit)).S * (1 / 256)) + 0 +
        Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
          (fun l => ¬ Served.Local ToyServed.C 1 ToyServed.thief 0 (ToyServed.gen z.1) z.2 l)
          (ToyServed.leakRun (ToyServed.gen z.1) z.2))

open scoped ENNReal in
/-- **Model weight security at an audited deployment, faults as inputs.** The deployment serves `T` windows as in
`TheftServed`, and its releases are faulted runs: each accepted window that releases released the faulted served
function `Cf`'s answer at a fault pattern of size at most its incorrect proof blocks (`Served.Faithful`). If honest use
steals a weight of the typical set `Typ` with probability at most `p` (`HonestUseHard`), faults at the patterns no cut
covers hand out such a weight at most `N^(λ - 1)` times faster than honest use (`FaultedUseHard` at the parties whose
patterns are `κ.Interior`, `N = patternCount size T M`), the class `Af` holds the outsider with each fault channel
string written in (`Simulable`), and the audit leaves `M` incorrect proof blocks undetected with probability at most
`δ` (`Served.UndetectedLe`), then the outsider steals the weights with probability at most
`Pr[gen ω ∉ Typ] + |S| · N^(λ - 1) · p + δ + Pr[a release isn't faithful]`. `|S|` is the fault channel's covert
strings: each window's interior pattern or, where the cut `κ` cuts the window's pattern, `κ`'s hint, times the served
channel's at the overwrite budget `(L + 1) · |O|`. So interior faults cost `N^λ = 2^(λ · C_arb)` and a cut fault its
hint, where `TheftServed` charges every value their reach `E` can overwrite. At `Cut.none` every pattern is interior.
`φ₀` is the empty pattern (`NoFault`). -/
def TheftFaulted : Prop :=
  ∀ {Ω R Θ Q Adv O V Tick F Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv] [Fintype O]
    [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick] [Fintype F]
    (C : Θ → Q × Adv → O → V) (Cf : Θ → (Q × Adv) × F → O → V) (size : F → ℕ) (φ₀ : F)
    (κ : Cut (Q × Adv) (O → V)) (gen : Ω → Θ) (Typ : Set Θ) (Stolen : Θ → Θ' → Prop)
    (A : Set (Outsider R (Q × Adv) (O → V) Θ')) (Af : Set (Outsider R ((Q × Adv) × F) (O → V) Θ')) (p lam : ℝ)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L M : ℕ) (δ : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served O V Tick Adv))),
    NoFault size φ₀ →
    HonestUseHard C gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T A p →
    FaultedUseHard C Cf size gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T M A
      (Af ∩ PatternsIn (κ.Interior Cf size φ₀) T) lam →
    Simulable Af (Ou.withFaults φ₀)
      (faultChannel R Q Adv O V Tick F κ Cf size φ₀ T M ((L + 1) * Fintype.card O)) →
    Served.UndetectedLe M δ run →
      Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (run (gen z.1) z.2)) ≤
        ENNReal.ofReal (Claims.Pr fun z : Ω × R => gen z.1 ∉ Typ) +
          ENNReal.ofReal
            (Fintype.card (faultChannel R Q Adv O V Tick F κ Cf size φ₀ T M ((L + 1) * Fintype.card O)).S *
              ((patternCount size T M : ℝ) ^ (lam - 1) * p)) +
          δ + Verity.Game.avg (fun z : Ω × R =>
            Verity.Game.value (fun l => ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l) (run (gen z.1) z.2))

/-- **`TheftFaulted`'s premises hold together at the honest toy deployment**, with every weight typical, no halt lag,
the budget `1` and `λ = 1`. Neither the served function nor the faulted one reads the weights, so every party steals
them with probability at most `1/256`, faults or not. Every class holds the user with each covert string written in,
and the honest run's audit leaves no incorrect proof block undetected. -/
def TheftFaultedHonest : Prop :=
  NoFault ToyServed.size none ∧
    HonestUseHard ToyServed.C ToyServed.gen (fun θ θ' => θ ∈ Set.univ ∧ ToyServed.Stolen θ θ') 2 ToyServed.anyone
      (1 / 256) ∧
    FaultedUseHard ToyServed.C ToyServed.Cf ToyServed.size ToyServed.gen
      (fun θ θ' => θ ∈ Set.univ ∧ ToyServed.Stolen θ θ') 2 1 ToyServed.anyone
      (ToyServed.anyoneF ∩ PatternsIn ((Cut.none _ _).Interior ToyServed.Cf ToyServed.size none) 2) 1 ∧
    Simulable ToyServed.anyoneF (ToyServed.user.withFaults (none : ToyServed.Fault))
      (faultChannel Unit Unit Unit Unit (Fin 16) Unit ToyServed.Fault (Cut.none _ _) ToyServed.Cf ToyServed.size none
        2 1 ((0 + 1) * Fintype.card Unit)) ∧
    Served.UndetectedLe 1 0 ToyServed.honestRun

open scoped ENNReal in
/-- **`TheftFaulted`'s bound fails at the verifier that accepts everything.** The leaking toy deployment's window `i` is
the faulted run with the weights' `i`-th value as its one fault, and its audit accepts on an empty sample. The thief then
steals the weights on every coin, with every weight typical and every release faithful. At the budget `0`, `p = 1/256`,
`λ = 1` and `δ = 0` the bound is below `1`, so it fails. -/
def TheftFaultedNontrivial : Prop :=
  ¬ Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
        (fun l => ToyServed.Stolen (ToyServed.gen z.1) (ToyServed.thief.out z.2 (Served.view ToyServed.thief 0 z.2 l 2)))
        (ToyServed.leakRun (ToyServed.gen z.1) z.2)) ≤
      ENNReal.ofReal (Claims.Pr fun z : ToyServed.W × Unit => ToyServed.gen z.1 ∉ (Set.univ : Set ToyServed.W)) +
        ENNReal.ofReal (Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit ToyServed.Fault (Cut.none _ _)
            ToyServed.Cf ToyServed.size none 2 0 ((0 + 1) * Fintype.card Unit)).S *
          ((patternCount ToyServed.size 2 0 : ℝ) ^ ((1 : ℝ) - 1) * (1 / 256))) + 0 +
        Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
          (fun l => ¬ Served.Faithful ToyServed.Cf ToyServed.size ToyServed.thief 0 (ToyServed.gen z.1) z.2 l)
          (ToyServed.leakRun (ToyServed.gen z.1) z.2))

open scoped ENNReal in
/-- **Model weight security per charge unit, in bits.** `TheftServed` with each window's released outputs grouped into
charge units `U` and a unit's whole output a value of `W`: a request, or a request with every request that reads state it
wrote (a conversation, a prefix tree), whichever an incorrect proof block's changes stay within. Under `TheftServed`'s
premises (`HonestUseHard`, `Simulable` and `Served.UndetectedLe`), `Ou` outputs a stolen model with probability at most
$$2^B \cdot p + \delta + \Pr[\neg \mathrm{Isolated}],$$
`B` the covert capacity in bits (`servedBits`),
$$B = T\,(\log_2|\mathrm{Adv}| + \log_2|\mathrm{Tick}|) + K\,(\log_2|W| + \log_2(T\,|U| + 1)) + \log_2(T + 1),
\qquad K = F M + (L + 1)\,|U|:$$
each window's advice and timing bucket, each changed unit's whole output with its name among the run's `T · |U|` units,
and the halt. `Served.Isolated C F` says each incorrect proof block changes at most `F` units, in its own window or
later ones, through the windows accepted before the first rejection, so `F M` counts the units the audit's undetected
blocks change there and `(L + 1) |U|` the windows released from the first rejection to the halt. `Served.Local` at
`E = F` gives `Served.Isolated`. -/
def TheftServedReq : Prop :=
  ∀ {Ω R Θ Q Adv U W Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv] [Fintype U]
    [DecidableEq U] [Fintype W] [Nonempty W] [Fintype Tick] [Inhabited Tick]
    (C : Θ → Q × Adv → U → W) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (A : Set (Outsider R (Q × Adv) (U → W) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((U → W) × Tick)) Θ') (T L F M : ℕ) (δ : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served U W Tick Adv))),
    HonestUseHard C gen Stolen T A p →
    Simulable A Ou (servedChannel R Q Adv U W Tick T (F * M + (L + 1) * Fintype.card U)) →
    Served.UndetectedLe M δ run →
      Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (run (gen z.1) z.2)) ≤
        ENNReal.ofReal ((2 : ℝ) ^ servedBits Adv U W Tick T (F * M + (L + 1) * Fintype.card U) * p) + δ +
          Verity.Game.avg (fun z : Ω × R =>
            Verity.Game.value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l) (run (gen z.1) z.2))

open scoped ENNReal in
/-- **`TheftServedReq`'s bound fails at the verifier that accepts everything.** At the leaking toy deployment every
release is isolated at one unit (its one slot) per incorrect proof block, and the thief steals the weights on every coin.
The covert capacity at one changed unit is `log₂ 144` bits (`16` values, `3` names, `3` halts), so the bound at
`p = 1/256` and `δ = 0` is `144/256`, below `1`. -/
def TheftServedReqNontrivial : Prop :=
  ¬ Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
        (fun l => ToyServed.Stolen (ToyServed.gen z.1) (ToyServed.thief.out z.2 (Served.view ToyServed.thief 0 z.2 l 2)))
        (ToyServed.leakRun (ToyServed.gen z.1) z.2)) ≤
      ENNReal.ofReal ((2 : ℝ) ^ servedBits Unit Unit (Fin 16) Unit 2 (1 * 0 + (0 + 1) * Fintype.card Unit) * (1 / 256)) +
        0 + Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
          (fun l => ¬ Served.Isolated ToyServed.C 1 ToyServed.thief 0 (ToyServed.gen z.1) z.2 l)
          (ToyServed.leakRun (ToyServed.gen z.1) z.2))

open scoped ENNReal in
/-- **Model weight security per charge unit, from the weights' min-entropy.** `TheftServedReq` against every party.
If honest use leaves the weights `K` bits of min-entropy, so that every party, whatever its time, steals them from `T`
honest queries with probability at most `2^(-K)` (`HonestUseHard` at `A = Set.univ`), then every class holds the
outsider with each covert string written in, and `Ou` outputs a stolen model with probability at most
$$2^{B - K} + \delta + \Pr[\neg \mathrm{Isolated}],$$
`B` as in `TheftServedReq`. The guarantee says something while `B < K`. -/
def TheftServedMinEntropy : Prop :=
  ∀ {Ω R Θ Q Adv U W Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv] [Fintype U]
    [DecidableEq U] [Fintype W] [Nonempty W] [Fintype Tick] [Inhabited Tick]
    (C : Θ → Q × Adv → U → W) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (K : ℝ)
    (Ou : Outsider R (Q × Adv) (Option ((U → W) × Tick)) Θ') (T L F M : ℕ) (δ : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served U W Tick Adv))),
    HonestUseHard C gen Stolen T (Set.univ : Set (Outsider R (Q × Adv) (U → W) Θ')) ((2 : ℝ) ^ (-K)) →
    Served.UndetectedLe M δ run →
      Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (run (gen z.1) z.2)) ≤
        ENNReal.ofReal ((2 : ℝ) ^ (servedBits Adv U W Tick T (F * M + (L + 1) * Fintype.card U) - K)) + δ +
          Verity.Game.avg (fun z : Ω × R =>
            Verity.Game.value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l) (run (gen z.1) z.2))

/-- **`TheftServedMinEntropy`'s premises hold together at the honest toy deployment**: honest use leaves the toy's
weights `8` bits of min-entropy against every party, and the honest run's audit leaves no incorrect proof block
undetected. -/
def TheftServedMinEntropyHonest : Prop :=
  HonestUseHard ToyServed.C ToyServed.gen ToyServed.Stolen 2
      (Set.univ : Set (Outsider Unit (Unit × Unit) (Unit → Fin 16) ToyServed.W)) ((2 : ℝ) ^ (-(8 : ℝ))) ∧
    Served.UndetectedLe 1 0 ToyServed.honestRun

open scoped ENNReal in
/-- **`TheftServedMinEntropy`'s bound fails at the verifier that accepts everything.** At the leaking toy deployment,
whose weights have `8` bits of min-entropy, every release is isolated and the thief steals the weights on every coin,
while the bound at `δ = 0` is `2^(log₂ 144 - 8) = 144/256`, below `1`. -/
def TheftServedMinEntropyNontrivial : Prop :=
  ¬ Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
        (fun l => ToyServed.Stolen (ToyServed.gen z.1) (ToyServed.thief.out z.2 (Served.view ToyServed.thief 0 z.2 l 2)))
        (ToyServed.leakRun (ToyServed.gen z.1) z.2)) ≤
      ENNReal.ofReal ((2 : ℝ) ^ (servedBits Unit Unit (Fin 16) Unit 2 (1 * 0 + (0 + 1) * Fintype.card Unit) - 8)) +
        0 + Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
          (fun l => ¬ Served.Isolated ToyServed.C 1 ToyServed.thief 0 (ToyServed.gen z.1) z.2 l)
          (ToyServed.leakRun (ToyServed.gen z.1) z.2))

end WeightSecurity.Guarantees
