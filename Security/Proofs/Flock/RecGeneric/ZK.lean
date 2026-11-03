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
* **the release rule** (`ZkServer`): after the commitments it releases the outer sessions' view only when the workers'
  values are a witness for the circuits the sessions prove (`ok`), and otherwise a refusal that reads only public data
  (`refuse`): nothing else of the inner messages, the salts or the developer's randomness leaves.

`GenericRecursiveZK`: where the outer sessions are zero knowledge on every witness the workers give (`OuterViewZK`'s
form at `ok`), the auditor's whole view is within `ε_O + n_h·δ + Pr[¬ok]` of the simulator's on ideal leaves, at every
event, both ways. `Pr[¬ok]` is what a misbehaving worker costs: zero where the workers' values always satisfy the
circuits (today's premise), and the probability they don't otherwise. Without the refusal, the outer prover would run
on a non-witness, where its zero knowledge says nothing.

At hm96 (`Leaves.hm96`, through any encoding of a commit string's two halves) `Hides` holds at the key's hiding gap
(`Hm96Hides`, from `ZK.ideal_leaves_adaptive`), and `Flock.Guarantees.RecursiveZK` is the generic statement at the
the ZK server's leaves with the workers always honest (`RecoversRecursiveZK`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Verity.Game FlockSoundness.Discharge
open scoped ENNReal

/-- **Hiding commitments as the auditor sees them**: salts `Y` and ideal randomness `U` (finite, nonempty), what the
auditor sees of one commitment (`L`), the commitment to a message under a salt (`leaf`), and an ideal commitment, which
reads no message (`ideal`). -/
structure Leaves where
  Y : Type
  [yFin : Fintype Y]
  [yNe : Nonempty Y]
  U : Type
  [uFin : Fintype U]
  [uNe : Nonempty U]
  L : Type
  leaf : ByteArray → Y → L
  ideal : U → L

attribute [instance] Leaves.yFin Leaves.yNe Leaves.uFin Leaves.uNe

/-- **The commitments to adaptive messages**: commitment `i` is `leaf` of message `i`, which reads the commitments
before it, under the salt `y i`. -/
def Leaves.commits (Lv : Leaves) {n : ℕ} (msgs : (i : Fin n) → (Fin i → Lv.L) → ByteArray) (y : Fin n → Lv.Y)
    (i : Fin n) : Lv.L :=
  Lv.leaf (msgs i fun j => Lv.commits msgs y (Fin.castLE i.isLt.le j)) (y i)
termination_by i.val
decreasing_by exact j.isLt

/-- **The leaves hide, against adaptive messages, within `δ` each**: in an experiment on a uniform `ω` and salts, whose
message `i` reads `ω` and the commitments before it, swapping every commitment for an ideal one on fresh uniform `U`s
raises the probability of any event by at most `n·δ`. -/
def Leaves.Hides (Lv : Leaves) (δ : ℝ≥0∞) : Prop :=
  ∀ (n : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (msgs : Ω → (i : Fin n) → (Fin i → Lv.L) → ByteArray)
    (G : Ω → (Fin n → Lv.L) → Prop),
    prCoin (fun p : Ω × (Fin n → Lv.Y) => G p.1 (Lv.commits (msgs p.1) p.2)) ≤
      prCoin (fun p : Ω × (Fin n → Lv.U) => G p.1 fun i => Lv.ideal (p.2 i)) + n * δ

/-- **hm96 leaves through an encoding `H` of a commit string's two halves**: the commitment to `m` under the salt `y`
is `H(SHA-512(m) ⊕ M·y, c(y))`, the ideal one `H(U, c(y))` (`OuterZK.idealStrs`'s). At `H = commitBytes` they are the
commit strings the ZK server sends; at `H = ZkView.leafH`, the leaves `RecursiveZK` reads (`firewallLeaf`,
`idealLeaf`). -/
def Leaves.hm96 {D : Type} (H : ZkView.Bits × ByteArray → D) : Leaves where
  Y := ZkView.Salt
  U := ZkView.Bits × ZkView.Salt
  L := D
  leaf m y := H (ZkView.bitsOf (Flock.Sha512.hash m) + ZkView.My y, ZkView.saltHash y)
  ideal u := H (u.1, ZkView.saltHash u.2)

/-- **The ZK server's release, at the leaves `Lv`**: on the auditor's tape `Ωa` and the developer's randomness `Ωp`,
the `nh` inner messages its workers send (each reading the commitments before it); whether the workers' values are a
witness for the circuits the outer sessions prove, at the salts (`ok`); the outer sessions' view at the commitments
(`view`, on their own randomness `Ωo`), released only where `ok` holds; and the refusal released otherwise, which reads
only the auditor's tape and the commitments (`refuse`). -/
structure ZkServer (Lv : Leaves) (Ωa Ωp Ωo V : Type) where
  nh : ℕ
  msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → Lv.L) → ByteArray
  ok : Ωa → Ωp → (Fin nh → Lv.Y) → Prop
  view : Ωa → (Fin nh → Lv.L) → Ωp → (Fin nh → Lv.Y) → Ωo → V
  refuse : Ωa → (Fin nh → Lv.L) → V

namespace ZkServer

variable {Lv : Leaves} {Ωa Ωp Ωo V : Type}

/-- The commitments the server sends. -/
def commits (Z : ZkServer Lv Ωa Ωp Ωo V) (a : Ωa) (p : Ωp) (y : Fin Z.nh → Lv.Y) : Fin Z.nh → Lv.L :=
  Lv.commits (Z.msgs a p) y

open Classical in
/-- **The auditor's whole view**: its tape, the commitments, and the outer view or the refusal. -/
noncomputable def real (Z : ZkServer Lv Ωa Ωp Ωo V) (a : Ωa) (p : Ωp) (y : Fin Z.nh → Lv.Y) (o : Ωo) :
    Ωa × (Fin Z.nh → Lv.L) × V :=
  (a, Z.commits a p y, if Z.ok a p y then Z.view a (Z.commits a p y) p y o else Z.refuse a (Z.commits a p y))

/-- **The outer sessions are zero knowledge on every witness the workers give**: where `ok` holds, the outer view at
the commitments and the simulator `sim`, which reads the auditor's tape and the commitments only, give every event
probabilities within `ε_O` (`Flock.Guarantees.OuterViewZK`'s form, at `ok`). -/
def OuterZK [Fintype Ωo] {Ωs : Type} [Fintype Ωs] (Z : ZkServer Lv Ωa Ωp Ωo V)
    (sim : Ωa → (Fin Z.nh → Lv.L) → Ωs → V) (εo : ℝ≥0∞) : Prop :=
  ∀ a p y (E : V → Prop), Z.ok a p y →
    prCoin (fun o : Ωo => E (Z.view a (Z.commits a p y) p y o)) ≤
      prCoin (fun s : Ωs => E (sim a (Z.commits a p y) s)) + εo

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

/-- **Z1: the recursion is zero knowledge, generic in the server's leaves and the outer sessions**: where the leaves
hide within `δ` and the outer sessions are zero knowledge within `ε_O` on every witness the workers give, the auditor's
whole view (`ZkServer.real`) is within `ε_O + n_h·δ + Pr[¬ok]` of the simulator that samples ideal commitments and runs
`sim` on them, at every event (so both ways, at its complement). The simulator reads the auditor's tape and public
data only. -/
def GenericRecursiveZK : Prop :=
  ∀ (Lv : Leaves) {Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo]
    [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (Z : ZkServer Lv Ωa Ωp Ωo V) (sim : Ωa → (Fin Z.nh → Lv.L) → Ωs → V)
    {εo δ : ℝ≥0∞}, Lv.Hides δ → Z.OuterZK sim εo →
    ∀ E : Ωa × (Fin Z.nh → Lv.L) × V → Prop,
      prCoin (fun w : ((Ωa × Ωp) × (Fin Z.nh → Lv.Y)) × Ωo => E (Z.real w.1.1.1 w.1.1.2 w.1.2 w.2)) ≤
        prCoin (fun w : (Ωa × (Fin Z.nh → Lv.U)) × Ωs =>
            E (w.1.1, (fun i => Lv.ideal (w.1.2 i)), sim w.1.1 (fun i => Lv.ideal (w.1.2 i)) w.2)) +
          εo + Z.nh * δ + prCoin (fun w : (Ωa × Ωp) × (Fin Z.nh → Lv.Y) => ¬ Z.ok w.1.1 w.1.2 w.2)

/-- **Z3: today's statement is the generic one at the ZK server's leaves**: `GenericRecursiveZK` and `Hm96Hides` give
`RecursiveZK` (the leaves `ZkView.leafH`, messages that read no commitment, workers always honest, so
`Pr[¬ok] = 0`, and `δ` the pinned key's hiding gap, `CROnly.hm96Hiding_gap`). -/
def RecoversRecursiveZK : Prop := GenericRecursiveZK → Hm96Hides → RecursiveZK

end Flock.Guarantees
