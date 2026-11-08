module

public import Definitions.WeightSecurity.Assumptions
public import Definitions.WeightSecurity.Toy
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

`TheftHonest` and `TheftNontrivial` are its witnesses, and `TheftTight` shows the factor `|S|` is attained, all at the
toy instance (`Definitions.WeightSecurity.Toy`).

`TheftServed` is `Theft` at an audited deployment (`Definitions.WeightSecurity.Served`). The channel is the served one:
advice, timing buckets, the overwrites the audit lets through and the halt. The leakage term is the audit's escape `δ`
plus the probability that a release isn't local, which the network certifier and the dataflow discharge. Its witnesses
`TheftServedHonest` and `TheftServedNontrivial` are at the toy deployment (`Definitions.WeightSecurity.ToyServed`).
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
* the audit escapes at the budget `M` with probability at most `δ` on every run (`Served.Escapes`).

Then `Ou` outputs a stolen model with probability at most
$$|S| \cdot p + \delta + \Pr[\neg \mathrm{Local}],$$
`S` the served channel's covert strings. `Served.Local` says each window's release changes at most `E` slots per
incorrect proof block. The layers below the composition give it: the network certifier, that what leaves is the
trace's outputs at its bucket, and the dataflow, that the trace is honest outside the incorrect proof blocks' egress
cones. Neither layer reads the other. -/
def TheftServed : Prop :=
  ∀ {Ω R Θ Q Adv O V Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv] [Fintype O]
    [DecidableEq O] [Fintype V] [DecidableEq V] [Fintype Tick] [Inhabited Tick]
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (A : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L E M : ℕ) (δ : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served O V Tick Adv))),
    HonestUseHard C gen Stolen T A p →
    Simulable A Ou (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)) →
    (∀ θ r, Verity.Game.value (Served.Escapes M) (run θ r) ≤ δ) →
      Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (run (gen z.1) z.2)) ≤
        ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S * p) +
          δ + Verity.Game.avg (fun z : Ω × R =>
            Verity.Game.value (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l) (run (gen z.1) z.2))

/-- **`TheftServed`'s premises hold together at the honest toy deployment**, with no halt lag, no slot changed per
incorrect proof block and the budget `1`. Its served function answers `0`, so two honest queries steal the weights with
probability at most `1/256` for every party. Every class holds the user with each covert string written in. The honest
run's audit never escapes at the budget `1`. -/
def TheftServedHonest : Prop :=
  HonestUseHard ToyServed.C ToyServed.gen ToyServed.Stolen 2 ToyServed.anyone (1 / 256) ∧
    Simulable ToyServed.anyone ToyServed.user
      (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2 (0 * 1 + (0 + 1) * Fintype.card Unit)) ∧
    ∀ θ r, Verity.Game.value (Served.Escapes 1) (ToyServed.honestRun θ r) ≤ 0

open scoped ENNReal in
/-- **`TheftServed`'s bound fails at the verifier that accepts everything.** The leaking toy deployment writes the
weights into its two releases, one incorrect proof block per window, and its audit accepts without a draw. The thief
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

end WeightSecurity.Guarantees
