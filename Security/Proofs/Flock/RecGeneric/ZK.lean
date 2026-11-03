module

public import Proofs.Flock.Recursive.Statements

/-!
# The ZK server, as the recursion's zero knowledge reads it

Zero knowledge reads the server through what it releases, never through Flock's message formats:

* **hiding leaves** (`Leaves`): it commits to every inner message under a fresh salt, and the auditor sees only the
  commitment (`leaf`). `Leaves.Hides δ` is hiding against adaptive messages: message `i` may read the commitments before
  it, since each inner coin is live, sampled after the commitment before it, and a malicious auditor's coin is a function
  of its tape and what it has seen; swapping every commitment for an ideal one (`ideal`, reading no message) moves any
  event by at most `n·δ`;
* **the release rule** (`ZkServer`): after the `n_h` commitments (`n_h` is protocol structure, public) it releases the
  outer sessions' view only when the workers' values are a witness for the circuits the sessions prove (`ok`), and
  otherwise a refusal that reads only public data (`refuse`): nothing else of the inner messages, the salts or the
  developer's randomness leaves.

Zero knowledge is in leakage form (`LeakBelow`): the real view gives every event at most `2^l` times the simulator's
probability plus an error, `Pr_real[E] ≤ 2^l·(Pr_sim[E] + ε(w, V*))`, the error reading the witness and the verifier; at
`l = 0` it holds both ways, at the event's complement. It is the sampled audit's form (the ZK server's
`Pr_real[E] ≤ 2·(Pr_sim[E] + ε)`), so the two compose as one notion.

`GenericRecursiveZK`: for a server indexed by the developer's witness and the auditor (`Zs w V*`) and one simulator
fixed before both, which reads the auditor and public data only: where the outer sessions are zero knowledge at that
simulator with an `l`-bit leak on every witness the workers give (`ZkServer.OuterZK` at `ok`), the auditor's whole view
is `l`-below the simulator's on ideal leaves at `ε_O(w, V*) + n_h·δ + Pr[¬ok]`. `Pr[¬ok]` is what a misbehaving worker
costs: zero where the workers' values always satisfy the circuits (today's premise), and the probability they don't
otherwise. Without the refusal, the outer prover would run on a non-witness, where its zero knowledge says nothing.

At hm96 (`Leaves.hm96`, through any encoding of a commit string's two halves) `Hides` holds at the key's hiding gap
(`Hm96Hides`, from `ZK.ideal_leaves_adaptive`), and `Flock.Guarantees.RecursiveZK` is the generic statement at the
the ZK server's leaves with no leak and the workers always honest (`RecoversRecursiveZK`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Verity.Game FlockSoundness.Discharge
open scoped ENNReal

/-- **Hiding commitments as the auditor sees them**: salts (`Salt`) and ideal randomness (`Rand`), both finite and
nonempty, what the auditor sees of one commitment (`Shown`), the commitment to a message under a salt (`leaf`), and an
ideal commitment, which reads no message (`ideal`). -/
structure Leaves where
  Salt : Type
  [saltFin : Fintype Salt]
  [saltNe : Nonempty Salt]
  Rand : Type
  [randFin : Fintype Rand]
  [randNe : Nonempty Rand]
  Shown : Type
  leaf : ByteArray → Salt → Shown
  ideal : Rand → Shown

attribute [instance] Leaves.saltFin Leaves.saltNe Leaves.randFin Leaves.randNe

/-- **The commitments to adaptive messages**: commitment `i` is `leaf` of message `i`, which reads the commitments
before it, under the salt `y i`. -/
def Leaves.commits (Lv : Leaves) {n : ℕ} (msgs : (i : Fin n) → (Fin i → Lv.Shown) → ByteArray) (y : Fin n → Lv.Salt)
    (i : Fin n) : Lv.Shown :=
  Lv.leaf (msgs i fun j => Lv.commits msgs y (Fin.castLE i.isLt.le j)) (y i)
termination_by i.val
decreasing_by exact j.isLt

/-- **The leaves hide, against adaptive messages, within `δ` each**: in an experiment on a uniform `ω` and salts, whose
message `i` reads `ω` and the commitments before it, swapping every commitment for an ideal one on fresh uniform `Rand`s
raises the probability of any event by at most `n·δ`. -/
def Leaves.Hides (Lv : Leaves) (δ : ℝ≥0∞) : Prop :=
  ∀ (n : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (msgs : Ω → (i : Fin n) → (Fin i → Lv.Shown) → ByteArray)
    (G : Ω → (Fin n → Lv.Shown) → Prop),
    prCoin (fun p : Ω × (Fin n → Lv.Salt) => G p.1 (Lv.commits (msgs p.1) p.2)) ≤
      prCoin (fun p : Ω × (Fin n → Lv.Rand) => G p.1 fun i => Lv.ideal (p.2 i)) + n * δ

/-- **hm96 leaves through an encoding `H` of a commit string's two halves**: the commitment to `m` under the salt `y`
is `H(SHA-512(m) ⊕ M·y, c(y))`, the ideal one `H(U, c(y))` (`OuterZK.idealStrs`'s). At `H = commitBytes` they are the
commit strings the ZK server sends; at `H = ZkView.leafH`, the leaves `RecursiveZK` reads (`firewallLeaf`,
`idealLeaf`). -/
def Leaves.hm96 {D : Type} (H : ZkView.Bits × ByteArray → D) : Leaves where
  Salt := ZkView.Salt
  Rand := ZkView.Bits × ZkView.Salt
  Shown := D
  leaf m y := H (ZkView.bitsOf (Flock.Sha512.hash m) + ZkView.My y, ZkView.saltHash y)
  ideal u := H (u.1, ZkView.saltHash u.2)

/-- **Zero knowledge with an `l`-bit leak, at the error `ε`**: the real view `e`, on uniform coins, gives every event at
most `2^l` times its probability at the simulator's view `f`, on uniform coins, plus `ε`:
`Pr_real[E] ≤ 2^l·(Pr_sim[E] + ε)`, as a simulator that guesses `l` leaked bits would. At `l = 0` it is `ε`-closeness
both ways, at the event's complement. -/
def LeakBelow {A B V : Type} [Fintype A] [Fintype B] (e : A → V) (f : B → V) (l : ℕ) (ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, prCoin (fun a => E (e a)) ≤ 2 ^ l * (prCoin (fun b => E (f b)) + ε)

/-- **The ZK server's release, at the leaves `Lv` and `nh` commitments**: on the auditor's tape `Ωa` and the developer's
randomness `Ωp`, the `nh` inner messages its workers send (each reading the commitments before it); whether the workers'
values are a witness for the circuits the outer sessions prove, at the salts (`ok`); the outer sessions' view at the
commitments (`view`, on their own randomness `Ωo`), released only where `ok` holds; and the refusal released otherwise,
which reads only the auditor's tape and the commitments (`refuse`). `nh` is protocol structure, so a simulator may read
it. -/
structure ZkServer (Lv : Leaves) (nh : ℕ) (Ωa Ωp Ωo V : Type) where
  msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → Lv.Shown) → ByteArray
  ok : Ωa → Ωp → (Fin nh → Lv.Salt) → Prop
  view : Ωa → (Fin nh → Lv.Shown) → Ωp → (Fin nh → Lv.Salt) → Ωo → V
  refuse : Ωa → (Fin nh → Lv.Shown) → V

namespace ZkServer

variable {Lv : Leaves} {nh : ℕ} {Ωa Ωp Ωo V : Type}

/-- The commitments the server sends. -/
def commits (Zs : ZkServer Lv nh Ωa Ωp Ωo V) (a : Ωa) (p : Ωp) (y : Fin nh → Lv.Salt) : Fin nh → Lv.Shown :=
  Lv.commits (Zs.msgs a p) y

open Classical in
/-- **The auditor's whole view**: its tape, the commitments, and the outer view or the refusal. -/
noncomputable def real (Zs : ZkServer Lv nh Ωa Ωp Ωo V) (a : Ωa) (p : Ωp) (y : Fin nh → Lv.Salt) (o : Ωo) :
    Ωa × (Fin nh → Lv.Shown) × V :=
  (a, Zs.commits a p y, if Zs.ok a p y then Zs.view a (Zs.commits a p y) p y o else Zs.refuse a (Zs.commits a p y))

/-- **The outer sessions are zero knowledge with an `l`-bit leak on every witness the workers give**: where `ok` holds,
the outer view at the commitments is `l`-below the simulator `sim`'s, which reads the auditor's tape and the
commitments only, at `ε_O` (`LeakBelow`; at `l = 0`, `Flock.Guarantees.OuterViewZK`'s form, at `ok`). -/
def OuterZK [Fintype Ωo] {Ωs : Type} [Fintype Ωs] (Zs : ZkServer Lv nh Ωa Ωp Ωo V)
    (sim : Ωa → (Fin nh → Lv.Shown) → Ωs → V) (l : ℕ) (εo : ℝ≥0∞) : Prop :=
  ∀ a p y, Zs.ok a p y →
    LeakBelow (fun o : Ωo => Zs.view a (Zs.commits a p y) p y o) (fun s : Ωs => sim a (Zs.commits a p y) s) l εo

end ZkServer

/-- **Z2: hm96 leaves hide at the key's hiding bound**, through every encoding of the commit string (from
`ZK.ideal_leaves_adaptive`). -/
def Hm96Hides : Prop :=
  ∀ {D : Type} (H : ZkView.Bits × ByteArray → D) (δ : ℝ≥0∞), Claims.Hm96Hiding ZkView.My ZkView.saltHash δ →
    (Leaves.hm96 H).Hides δ

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims.RecGeneric Verity.Game
open scoped ENNReal

/-- **Z1: the recursion is zero knowledge, generic in the server's leaves and the outer sessions.** For every server
indexed by the developer's witness `w` and the auditor `V*` (`Zs w V*`, its `n_h` commitments public), every simulator
`sim`, fixed before both, which reads the auditor, its tape and the commitments only, and every leak `l` and outer error
`ε_O(w, V*)`: where the leaves hide within `δ` and the outer sessions are zero knowledge at `sim` with an `l`-bit leak
within `ε_O` on every witness the workers give, then at every witness and auditor the auditor's whole view
(`ZkServer.real`) is `l`-below the simulator that samples ideal commitments and runs `sim` on them (`LeakBelow`):
`Pr_real[E] ≤ 2^l·(Pr_sim[E] + ε_O(w, V*) + n_h·δ + Pr[¬ok])` at every event. -/
def GenericRecursiveZK : Prop :=
  ∀ (Lv : Leaves) (nh : ℕ) {Wit VS Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (Zs : Wit → VS → ZkServer Lv nh Ωa Ωp Ωo V)
    (sim : VS → Ωa → (Fin nh → Lv.Shown) → Ωs → V) (l : ℕ) (εo : Wit → VS → ℝ≥0∞) {δ : ℝ≥0∞},
    Lv.Hides δ → (∀ w v, (Zs w v).OuterZK (sim v) l (εo w v)) →
    ∀ w v, LeakBelow (fun u : ((Ωa × Ωp) × (Fin nh → Lv.Salt)) × Ωo => (Zs w v).real u.1.1.1 u.1.1.2 u.1.2 u.2)
      (fun u : (Ωa × (Fin nh → Lv.Rand)) × Ωs =>
        (u.1.1, (fun i => Lv.ideal (u.1.2 i)), sim v u.1.1 (fun i => Lv.ideal (u.1.2 i)) u.2)) l
      (εo w v + nh * δ + prCoin (fun u : (Ωa × Ωp) × (Fin nh → Lv.Salt) => ¬ (Zs w v).ok u.1.1 u.1.2 u.2))

/-- **Z3: today's statement is the generic one at the ZK server's leaves**: `GenericRecursiveZK` and `Hm96Hides` give
`RecursiveZK` (the leaves `ZkView.leafH`, messages that read no commitment, no leak (`l = 0`, so both ways), workers
always honest, so `Pr[¬ok] = 0`, and `δ` the pinned key's hiding gap, `CROnly.hm96Hiding_gap`). -/
def RecoversRecursiveZK : Prop := GenericRecursiveZK → Hm96Hides → RecursiveZK

end Flock.Guarantees
