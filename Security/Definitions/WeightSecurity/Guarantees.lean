module

public import Definitions.WeightSecurity.Assumptions
public import Definitions.WeightSecurity.Toy

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

end WeightSecurity.Guarantees
