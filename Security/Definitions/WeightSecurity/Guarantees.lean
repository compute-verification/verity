module

public import Definitions.WeightSecurity.Assumptions
public import Definitions.WeightSecurity.Faulted
public import Definitions.WeightSecurity.Toy
public import Definitions.WeightSecurity.ToyFaulted
public import Definitions.WeightSecurity.ToyServed
public import Definitions.WeightSecurity.ToyTied

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

`TheftHonest` and `TheftNontrivial` are its witnesses, and `TheftTight` shows the factor `|S|` is attained, all at the
toy instance (`Definitions.WeightSecurity.Toy`).

`TheftServed` is `Theft` at an audited deployment (`Definitions.WeightSecurity.Served`). The channel is the served one:
advice, timing buckets, the overwrites the audit lets through and the halt. The leakage term is the probability `δ` that
the audit leaves `M` incorrect proof blocks undetected, plus the probability that a release isn't local, which the
network certifier and the dataflow discharge. Its witnesses
`TheftServedHonest` and `TheftServedNontrivial` are at the toy deployment (`Definitions.WeightSecurity.ToyServed`).

`TheftFaulted` is the same deployment under SPOT's fault model (`Definitions.WeightSecurity.Faulted`): a fault is an
input to the faulted circuit, not an overwrite of a release, so SPOT's part of the channel is the fault patterns, of
sizes summing to at most `M`, and the reach of a fault drops out. What faults do to the weights' exposure is the model
claim `FaultedUseHard` (λ), on a typical set of weights. `TheftServed` stays as the bound that takes no claim about
the model. Its witnesses `TheftFaultedHonest` and `TheftFaultedNontrivial` are at the toy deployment with faults
(`Definitions.WeightSecurity.ToyFaulted`).

`TheftTied` is `TheftFaulted` at a served circuit that is a checked population, its run recording each window's
extracted trace (`Definitions.WeightSecurity.Tied`). The probability that a release isn't faithful gives way to the
failure probabilities of the two premises that make it faithful: the releases are their traces' outputs (`EgressTied`)
and the traces hold the windows' inputs (`AdviceCovered`). Its witnesses `TheftTiedHonest` and `TheftTiedNontrivial`
are at the toy deployment as a population (`Definitions.WeightSecurity.ToyTied`).
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

/-- **`Theft`'s factor `|S|` is attained.** The cheating toy datacenter's view goes through the one-bit channel (two
covert strings) with `ε = 0`, and the thief steals the weights with probability `1`, the bound `2 · 1/2 + 0`: each
covert bit can double the theft's probability, so it is only unlikely while the covert capacity stays below
`log₂ (1/p)`. -/
def TheftTight : Prop :=
  LeaksThrough Toy.C Toy.gen Toy.cheatDC Toy.thief 1 Toy.oneBit 0 ∧
    Claims.Pr (fun z : (Bool × Bool) × Unit =>
        Toy.Stolen (Toy.gen z.1) (Toy.thief.model z.2 (Toy.cheatDC (Toy.gen z.1) z.2) 1)) =
      Fintype.card Toy.oneBit.S * 2⁻¹ + 0

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
steals a weight of the typical set `Typ` with probability at most `p` (`HonestUseHard`), faults hand out such a weight
at most `N^(λ - 1)` times faster than honest use (`FaultedUseHard`, `N = patternCount size T M`), the class `Af` holds
the outsider with each fault channel string written in (`Simulable`), and the audit leaves `M` incorrect proof blocks
undetected with probability at most `δ` (`Served.UndetectedLe`), then the outsider steals the weights with probability
at most `Pr[gen ω ∉ Typ] + |S| · N^(λ - 1) · p + δ + Pr[a release isn't faithful]`. `|S|` is the fault channel's
covert strings, `N` times the served channel's at the overwrite budget `(L + 1) · |O|`, so the faults cost
`N^λ = 2^(λ · C_arb)`, where `TheftServed` charges every value their reach `E` can overwrite. `φ₀` is the empty
pattern (`NoFault`). -/
def TheftFaulted : Prop :=
  ∀ {Ω R Θ Q Adv O V Tick F Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv] [Fintype O]
    [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick] [Fintype F]
    (C : Θ → Q × Adv → O → V) (Cf : Θ → (Q × Adv) × F → O → V) (size : F → ℕ) (φ₀ : F) (gen : Ω → Θ)
    (Typ : Set Θ) (Stolen : Θ → Θ' → Prop) (A : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (Af : Set (Outsider R ((Q × Adv) × F) (O → V) Θ')) (p lam : ℝ)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L M : ℕ) (δ : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served O V Tick Adv))),
    NoFault size φ₀ →
    HonestUseHard C gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T A p →
    FaultedUseHard C Cf size gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T M A Af lam →
    Simulable Af (Ou.withFaults φ₀) (faultChannel R Q Adv O V Tick F size T M ((L + 1) * Fintype.card O)) →
    Served.UndetectedLe M δ run →
      Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (run (gen z.1) z.2)) ≤
        ENNReal.ofReal (Claims.Pr fun z : Ω × R => gen z.1 ∉ Typ) +
          ENNReal.ofReal (Fintype.card (faultChannel R Q Adv O V Tick F size T M ((L + 1) * Fintype.card O)).S *
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
      (fun θ θ' => θ ∈ Set.univ ∧ ToyServed.Stolen θ θ') 2 1 ToyServed.anyone ToyServed.anyoneF 1 ∧
    Simulable ToyServed.anyoneF (ToyServed.user.withFaults (none : ToyServed.Fault))
      (faultChannel Unit Unit Unit Unit (Fin 16) Unit ToyServed.Fault ToyServed.size 2 1 ((0 + 1) * Fintype.card Unit)) ∧
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
        ENNReal.ofReal (Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit ToyServed.Fault ToyServed.size 2 0
            ((0 + 1) * Fintype.card Unit)).S *
          ((patternCount ToyServed.size 2 0 : ℝ) ^ ((1 : ℝ) - 1) * (1 / 256))) + 0 +
        Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
          (fun l => ¬ Served.Faithful ToyServed.Cf ToyServed.size ToyServed.thief 0 (ToyServed.gen z.1) z.2 l)
          (ToyServed.leakRun (ToyServed.gen z.1) z.2))

open scoped ENNReal in
/-- **Model weight security under SPOT's fault model, the releases tied to their traces.** `TheftFaulted` where the
served circuit is a checked population `P` (`Population.Dataflow.Checked`) with unit functions `f`, egress slots `eg`,
and inputs `inp θ x` at the weights `θ` and a query served with an advice `x`. The faulted function is the population's
(`faultedServed`), a fault pattern's size is how many units it names, `fun _ => none` is the empty pattern, and the run
records each window's extracted trace (`Served.Traced`). Where `TheftFaulted`'s bound adds the probability that a
release isn't faithful, this one adds the failure probabilities of the two premises that make every release faithful:
`ηe` that a release isn't its trace's outputs (`EgressTied`), and `ηa` that a trace holds inputs other than the
weights', the query's and the advice's (`AdviceCovered`). The audit's `δ` is at the run counted at its traces
(`Served.Traced.countRun`), whose incorrect proof blocks are each trace's wrong units. -/
def TheftTied : Prop :=
  ∀ {Ω R Θ Q Adv O V Tick Θ' U S : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv] [Fintype O]
    [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick] [Fintype U] [DecidableEq U] [DecidableEq S]
    (P : Population.Dataflow U S) (f : P.Fns V) (eg : O → S) (inp : Θ → Q × Adv → S → V)
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Typ : Set Θ) (Stolen : Θ → Θ' → Prop)
    (A : Set (Outsider R (Q × Adv) (O → V) Θ')) (Af : Set (Outsider R ((Q × Adv) × P.Faults V) (O → V) Θ'))
    (p lam : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L M : ℕ) (δ ηe ηa : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served.Traced O V Tick Adv S))),
    P.Checked →
    HonestUseHard C gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T A p →
    FaultedUseHard C (faultedServed P f eg inp) P.faultCount gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T M A Af lam →
    Simulable Af (Ou.withFaults (fun _ => none : P.Faults V))
      (faultChannel R Q Adv O V Tick (P.Faults V) P.faultCount T M ((L + 1) * Fintype.card O)) →
    Served.UndetectedLe M δ (Served.Traced.countRun P f run) →
    EgressTied eg L ηe run →
    AdviceCovered P inp Ou L ηa run →
      Verity.Game.avg (fun z : Ω × R => Verity.Game.value (fun l => Stolen (gen z.1)
          (Ou.out z.2 (Served.view Ou L z.2 (l.map Served.Traced.toServed) T))) (run (gen z.1) z.2)) ≤
        ENNReal.ofReal (Claims.Pr fun z : Ω × R => gen z.1 ∉ Typ) +
          ENNReal.ofReal (Fintype.card (faultChannel R Q Adv O V Tick (P.Faults V) P.faultCount T M
              ((L + 1) * Fintype.card O)).S * ((patternCount (P.faultCount (V := V)) T M : ℝ) ^ (lam - 1) * p)) +
          δ + ηe + ηa

/-- **`TheftTied`'s premises hold together at the honest toy deployment as a population**, with every weight typical,
no halt lag, the budget `1` and `λ = 1`. The population's one unit reads nothing, so it is checked, and its faulted
function reads no weight, so faults hand out the weights no faster than honest use. Every class holds the user with
each covert string written in. The honest run's traces are the honest assignment, so its audit leaves no incorrect
proof block undetected, its releases are their traces' outputs, and there is no input slot to hold. -/
def TheftTiedHonest : Prop :=
  ToyServed.pop.Checked ∧
    HonestUseHard ToyServed.C ToyServed.gen (fun θ θ' => θ ∈ Set.univ ∧ ToyServed.Stolen θ θ') 2 ToyServed.anyone
      (1 / 256) ∧
    FaultedUseHard ToyServed.C (faultedServed ToyServed.pop ToyServed.fns ToyServed.eg ToyServed.inp)
      ToyServed.pop.faultCount ToyServed.gen (fun θ θ' => θ ∈ Set.univ ∧ ToyServed.Stolen θ θ') 2 1
      ToyServed.anyone ToyServed.anyoneP 1 ∧
    Simulable ToyServed.anyoneP (ToyServed.user.withFaults (fun _ => none : ToyServed.pop.Faults (Fin 16)))
      (faultChannel Unit Unit Unit Unit (Fin 16) Unit (ToyServed.pop.Faults (Fin 16)) ToyServed.pop.faultCount 2 1
        ((0 + 1) * Fintype.card Unit)) ∧
    Served.UndetectedLe 1 0 (Served.Traced.countRun ToyServed.pop ToyServed.fns ToyServed.honestTraced) ∧
    EgressTied ToyServed.eg 0 0 ToyServed.honestTraced ∧
    AdviceCovered ToyServed.pop ToyServed.inp ToyServed.user 0 0 ToyServed.honestTraced

open scoped ENNReal in
/-- **`TheftTied`'s bound fails at the verifier that accepts everything.** The leaking toy deployment's window `i`
carries the weights' `i`-th value in the population's one slot and releases it, and its audit accepts on an empty
sample. The thief then steals the weights on every coin, with every weight typical, every release its trace's outputs
and no input slot to hold. At the budget `0`, `p = 1/256`, `λ = 1` and `δ = ηe = ηa = 0` the bound is below `1`, so it
fails. -/
def TheftTiedNontrivial : Prop :=
  ¬ Verity.Game.avg (fun z : ToyServed.W × Unit => Verity.Game.value
        (fun l => ToyServed.Stolen (ToyServed.gen z.1)
          (ToyServed.thief.out z.2 (Served.view ToyServed.thief 0 z.2 (l.map Served.Traced.toServed) 2)))
        (ToyServed.leakTraced (ToyServed.gen z.1) z.2)) ≤
      ENNReal.ofReal (Claims.Pr fun z : ToyServed.W × Unit => ToyServed.gen z.1 ∉ (Set.univ : Set ToyServed.W)) +
        ENNReal.ofReal (Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit (ToyServed.pop.Faults (Fin 16))
            ToyServed.pop.faultCount 2 0 ((0 + 1) * Fintype.card Unit)).S *
          ((patternCount (ToyServed.pop.faultCount (V := Fin 16)) 2 0 : ℝ) ^ ((1 : ℝ) - 1) * (1 / 256))) + 0 + 0 + 0

end WeightSecurity.Guarantees
