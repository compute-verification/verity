import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Defs
import Proofs.Flock.Soundness.Game.Prob

/-!
# Protocol 2, the private circuit: the target statements

* `CircuitPrivate`: circuit privacy in the simulation form. Whatever the hidden circuit `C` of class `c` and its honest
  witness, the verifier's view (the program's registration and the session's proof files) and the simulator's, which
  reads the class and the owed outputs only, give every event probabilities within `ε + zkGap` both ways, where the
  registration hides up to `ε` and `zkGap` is protocol 1's `ZkWireFixed` bound at the class's sizes.
* `CircuitIndist`: the same as indistinguishability: two circuits of one class with the same owed outputs give the
  verifier views within `2·(ε + zkGap)`.
* `RanRegistered`: the session ran the registered program, under `cr/sha-512`. Accepting while protocol 1's bad event
  holds or the session's program is not the registered one `p` is at most protocol 1's bound for `U c` plus
  `q²/2^513`, when the program commitment binds under SHA-512.

`circuit_private_of_zk` and `circuit_indist_of_zk` take protocol 1's zero knowledge at the `--zk` wire
(`ZkView.ZkWireFixed`) as a hypothesis, so that they compose with the composed `--zk` headline; `circuit_private` and
`circuit_indist` discharge it with `zk_wire_fixed`.

These are the reduction layer: `Scheme`, `Commit`, `Q` and `B` are abstract, so citing them for the deployed protocol
needs each instantiated. `S.U` must be the universal unit's real `--zk` tables (protocol 1's composed headline), and
`S.reg` the real registration, with `Hides` discharged from `Hm96Hiding` or `hash-derived-key`. `com.Binding` must be
proved for the real program commitment, and `Q`, `B` must be protocol 1's headline at `U c`, with `miss(K)` taken at
the padded `N`. `cost` counts every SHA-512 evaluation of the finder, including checking the registrant's opening
`op₀`. The gap is exact in the hidden values' count: `card Hid = nU + 2·nPU` is `card_hid`, an equality.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView
open scoped ENNReal

/-- **Circuit privacy, the simulation form.** At a class `c` whose registration hides up to `ε` (`Hides`): for every
owed outputs `o`, hidden circuit `C` and honest witness `w` (a function of the registrant's salts), where the universal
tables at `C`'s registrations meet `ZkWireFixed`'s conditions (`ZkOk`) and `w` satisfies them (`Sat`), every dummy
program `p₀` and dummy witness `w₀`, every encoding `enc` of the session's views and every event `E` of the
registration's view and the encoding, the real view (`realPr` at `pad C c`) and the simulator's (`simPr` at `p₀`, `w₀`)
give `E` probabilities within `ε + zkGap (zs c)`, both ways. -/
def CircuitPrivate : Prop :=
  ∀ {Cls Circ Out : Type} {Prog Rr Rv : Cls → Type} [∀ c, Fintype (Rr c)] {K W Hi VL : Type} [Field K] [Fintype K]
    [DecidableEq K] [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] (S : Scheme Cls Circ Out Prog Rr Rv K W Hi VL)
    (c : Cls) (ε : ℝ≥0∞), Hides (S.reg c) ε →
    ∀ (o : Out) (C : Circ) (w : Rr c → Fin (S.J c) → W), (∀ r, ZkOk (S.U c o (S.reg c (S.pad C c) r))) →
      (∀ r, Sat (S.U c o (S.reg c (S.pad C c) r)) (w r)) →
    ∀ (p₀ : Prog c) (w₀ : Fin (S.J c) → W) {Obs : Type} (enc : ∀ v, (∀ j, (S.U c o v j).tab.View) → Obs)
      (E : Rv c → Obs → Prop),
      realPr S c o (S.pad C c) w enc E ≤ simPr S c o p₀ w₀ enc E + (ε + zkGap (S.zs c)) ∧
        simPr S c o p₀ w₀ enc E ≤ realPr S c o (S.pad C c) w enc E + (ε + zkGap (S.zs c))

/-- **Circuit privacy, the indistinguishability form.** At a class `c` as in `CircuitPrivate`, any two hidden circuits
`C`, `C'` with honest witnesses `w`, `w'` for the same owed outputs `o`, at registrations whose tables meet `ZkOk`, give
every event `E` of the verifier's view probabilities within `2·(ε + zkGap (zs c))`. -/
def CircuitIndist : Prop :=
  ∀ {Cls Circ Out : Type} {Prog Rr Rv : Cls → Type} [∀ c, Fintype (Rr c)] {K W Hi VL : Type} [Field K] [Fintype K]
    [DecidableEq K] [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] (S : Scheme Cls Circ Out Prog Rr Rv K W Hi VL)
    (c : Cls) (ε : ℝ≥0∞), Hides (S.reg c) ε →
    ∀ (o : Out) (C C' : Circ) (w w' : Rr c → Fin (S.J c) → W), (∀ r, ZkOk (S.U c o (S.reg c (S.pad C c) r))) →
      (∀ r, ZkOk (S.U c o (S.reg c (S.pad C' c) r))) → (∀ r, Sat (S.U c o (S.reg c (S.pad C c) r)) (w r)) →
      (∀ r, Sat (S.U c o (S.reg c (S.pad C' c) r)) (w' r)) →
    ∀ {Obs : Type} (enc : ∀ v, (∀ j, (S.U c o v j).tab.View) → Obs) (E : Rv c → Obs → Prop),
      realPr S c o (S.pad C c) w enc E ≤ realPr S c o (S.pad C' c) w' enc E + 2 * (ε + zkGap (S.zs c))

/-- **The session ran the registered program**, under `cr/sha-512`. The root `rt` commits to a registered program `p`
(for a private circuit, `pad C c`) through the registrant's opening `op₀`, and the commitment binds under SHA-512. In a
session (a game `G` against any prover `σ`) whose acceptance `acc` shows that the program of its committed values
(`prog b`) opens `rt` (through `op b`): if protocol 1's soundness of `U c` bounds accepting with its bad event `Q` (at
least `K` wrong units of the committed values, say) by `B`, then accepting with `Q`, or with a program other than `p`,
has probability at most `B + q²/2^513`, under `SHA512CRStrict` (`cr/sha-512`) for the finder that plays the session
once and outputs the extractor's pair, at most `q` SHA-512 evaluations on every outcome (checking `op₀` included). -/
def RanRegistered : Prop :=
  ∀ {Root P Op : Type} (com : Commit Root P Op), com.Binding →
    ∀ (rt : Root) (p : P) (op₀ : Op), com.Opens rt p op₀ →
    ∀ {β : Type} (G : Game β) (σ : Strategy G) (acc Q : β → Prop) (prog : β → P) (op : β → Op),
      (∀ b, acc b → com.Opens rt (prog b) (op b)) →
    ∀ (B : ℝ≥0∞), prob (fun b => acc b ∧ Q b) G σ ≤ B →
    ∀ (cost : β → ℝ) (q : ℝ), (∀ b, cost b ≤ q) →
      Assumptions.SHA512CRStrict Refine.H512 G σ (fun b => some (com.ext (op b) op₀)) cost q →
      prob (fun b => acc b ∧ (Q b ∨ prog b ≠ p)) G σ ≤ B + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ))

end FlockSoundness.Discharge.PrivateCircuit
