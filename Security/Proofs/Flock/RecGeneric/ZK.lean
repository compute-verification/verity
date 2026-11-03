module

public import Proofs.Flock.Recursive.Statements

/-!
# The ZK server, as the recursion's zero knowledge reads it

Zero knowledge reads the server through what it releases, never through Flock's message formats:

* **hiding leaves** (`Leaves`): it commits to every inner message under a fresh salt, and the auditor sees only the
  commitment (`leaf`). `Leaves.Hides δ` is hiding against adaptive messages: message `i` may read the commitments
  before it, since each inner coin is live, sampled after the commitment before it, and a malicious auditor's coin is a
  function of its tape and what it has seen; swapping every commitment for an ideal one (`ideal`, reading no message)
  moves any event by at most `n·δ`;
* **the release rule** (`ZkServer`): after the `n_h` commitments (`n_h` is protocol structure, public) it releases the
  outer sessions' view and their public outcome (the verdict and where they stopped) only when the workers' values are a
  witness for the circuits the sessions prove (`ok`), and otherwise a refusal, its own code's, that reads only the
  auditor's tape and the commitments: nothing else of the inner messages, the salts or the developer's randomness
  leaves.

Zero knowledge is per outcome: the release's public outcome `s` (the outer sessions' outcome, or the refusal) is given
to the simulator, fixed before the witness, and at each `s` and event `E`,
`Pr_real[outcome = s ∧ E] ≤ Pr_sim(s)[E] + ε_s(w, V*)`, the error reading the witness and the verifier. Summed over the
outcomes it is the leakage form, `Pr_real[E] ≤ |St|·Pr_sim[E] + Σ_s ε_s` with the simulator drawing `s` uniformly:
the outcome is the one leak, `log₂|St|` bits. `Flock.Guarantees.SessionLoopZK`'s `Pr_real[E] ≤ 2·Pr_sim[E] + ε` is
it at two outcomes, summed.

`GenericRecursiveZK`: for a refusal and one simulator fixed before the developer's witness and the auditor, and a server
indexed by both (`Zs w V*`): where the outer sessions are zero knowledge per outcome at that simulator on every witness
the workers give (`ZkServer.OuterZK` at `ok`), the auditor's whole view meets every event jointly with each outcome at
most as often as the simulator on ideal leaves given that outcome (`idealView`), plus `ε_O(w, V*, s) + n_h·δ`, and plus
`n_h·δ` alone at the refusal. A misbehaving worker costs only the refusal's outcome: the refused view reads no value.
Without the refusal, the outer prover would run on a non-witness, where its zero knowledge says nothing.

At hm96 (`Leaves.hm96`, through any encoding of a commit string's two halves) `Hides` holds at the key's hiding gap
(`Hm96Hides`, from `ZK.ideal_leaves_adaptive`), and `Flock.Guarantees.RecursiveZK` is the generic statement at the ZK
server's leaves with one outcome and the workers always honest (`RecoversRecursiveZK`).

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

/-- **The ZK server's release, at the leaves `Lv`, `nh` commitments and the outcomes `St`**: on the auditor's tape `Ωa`
and the developer's randomness `Ωp`, the `nh` inner messages its workers send (each reading the commitments before it);
whether the workers' values are a witness for the circuits the outer sessions prove, at the salts (`ok`); and, released
only where `ok` holds, the outer sessions' view at the commitments (`view`, on their own randomness `Ωo`) and their
public outcome (`out`: the verdict and where they stopped). Where `ok` fails the server releases its refusal (`real`'s
`refuse`). `nh` is protocol structure, so a simulator may read it. -/
structure ZkServer (Lv : Leaves) (nh : ℕ) (Ωa Ωp Ωo St V : Type) where
  msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → Lv.Shown) → ByteArray
  ok : Ωa → Ωp → (Fin nh → Lv.Salt) → Prop
  view : Ωa → (Fin nh → Lv.Shown) → Ωp → (Fin nh → Lv.Salt) → Ωo → V
  out : Ωa → (Fin nh → Lv.Shown) → Ωp → (Fin nh → Lv.Salt) → Ωo → St

namespace ZkServer

variable {Lv : Leaves} {nh : ℕ} {Ωa Ωp Ωo St V : Type}

/-- The commitments the server sends. -/
def commits (Zs : ZkServer Lv nh Ωa Ωp Ωo St V) (a : Ωa) (p : Ωp) (y : Fin nh → Lv.Salt) : Fin nh → Lv.Shown :=
  Lv.commits (Zs.msgs a p) y

open Classical in
/-- **The release's public outcome**: the outer sessions' (`some`) where `ok` holds, the refusal (`none`) otherwise. -/
noncomputable def outcome (Zs : ZkServer Lv nh Ωa Ωp Ωo St V) (a : Ωa) (p : Ωp) (y : Fin nh → Lv.Salt) (o : Ωo) :
    Option St :=
  if Zs.ok a p y then some (Zs.out a (Zs.commits a p y) p y o) else none

open Classical in
/-- **The auditor's whole view**, at the server's refusal `refuse`, which reads only the auditor's tape and the
commitments: its tape, the commitments, and the outer view or the refusal. -/
noncomputable def real (Zs : ZkServer Lv nh Ωa Ωp Ωo St V) (refuse : Ωa → (Fin nh → Lv.Shown) → V) (a : Ωa) (p : Ωp)
    (y : Fin nh → Lv.Salt) (o : Ωo) : Ωa × (Fin nh → Lv.Shown) × V :=
  (a, Zs.commits a p y, if Zs.ok a p y then Zs.view a (Zs.commits a p y) p y o else refuse a (Zs.commits a p y))

/-- **The outer sessions are zero knowledge per outcome on every witness the workers give**: where `ok` holds, at each
outcome `s` and event `E`, the outer view meets `E` with the outcome `s` at most as often as the simulator `sim` given
`s`, which reads the auditor's tape and the commitments only, plus `ε_O(s)`
(`Pr[out = s ∧ E(view)] ≤ Pr[E(sim(s))] + ε_O(s)`; at one outcome, `Flock.Guarantees.OuterViewZK`'s form, at `ok`). -/
def OuterZK [Fintype Ωo] {Ωs : Type} [Fintype Ωs] (Zs : ZkServer Lv nh Ωa Ωp Ωo St V)
    (sim : Ωa → (Fin nh → Lv.Shown) → St → Ωs → V) (εo : St → ℝ≥0∞) : Prop :=
  ∀ a p y, Zs.ok a p y → ∀ (s : St) (E : V → Prop),
    prCoin (fun o : Ωo => Zs.out a (Zs.commits a p y) p y o = s ∧ E (Zs.view a (Zs.commits a p y) p y o)) ≤
      prCoin (fun σ : Ωs => E (sim a (Zs.commits a p y) s σ)) + εo s

end ZkServer

/-- **The simulator's whole view at the outcome `s`**, on the commitments `c`: the auditor's tape, `c`, and `sim` on
them given the outer sessions' outcome, or the refusal at the refusal's. -/
def idealView {Lv : Leaves} {nh : ℕ} {Ωa Ωs St V : Type} (refuse : Ωa → (Fin nh → Lv.Shown) → V)
    (sim : Ωa → (Fin nh → Lv.Shown) → St → Ωs → V) (s : Option St) (a : Ωa) (c : Fin nh → Lv.Shown) (σ : Ωs) :
    Ωa × (Fin nh → Lv.Shown) × V :=
  (a, c, s.elim (refuse a c) fun t => sim a c t σ)

/-- **Z2: hm96 leaves hide at the key's hiding bound**, through every encoding of the commit string (from
`ZK.ideal_leaves_adaptive`). -/
def Hm96Hides : Prop :=
  ∀ {D : Type} (H : ZkView.Bits × ByteArray → D) (δ : ℝ≥0∞), Claims.Hm96Hiding ZkView.My ZkView.saltHash δ →
    (Leaves.hm96 H).Hides δ

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims.RecGeneric Verity.Game
open scoped ENNReal

/-- **Z1: the recursion is zero knowledge per outcome, generic in the server's leaves and the outer sessions.** For
every refusal `refuse` and simulator `sim`, fixed before the developer's witness `w`, each reading the auditor, its tape
and the commitments only, every server indexed by `w` and the auditor `V*` (`Zs w V*`, its `n_h` commitments public)
and every outer error `ε_O(w, V*, s)`: where the leaves hide within `δ` and the outer sessions are zero knowledge per
outcome at `sim` within `ε_O` on every witness the workers give, then at every witness, auditor, outcome `s` (the outer
sessions' `some t`, or the refusal's `none`) and event `E` on the auditor's whole view (`ZkServer.real`),
`Pr_real[outcome = s ∧ E] ≤ Pr[E(idealView(s))] + ε_O(w, V*, t) + n_h·δ` (no `ε_O` at the refusal), `idealView` on
uniform ideal commitments. -/
def GenericRecursiveZK : Prop :=
  ∀ (Lv : Leaves) (nh : ℕ) {Wit VS Ωa Ωp Ωo Ωs St V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (refuse : Ωa → (Fin nh → Lv.Shown) → V)
    (sim : VS → Ωa → (Fin nh → Lv.Shown) → St → Ωs → V) (Zs : Wit → VS → ZkServer Lv nh Ωa Ωp Ωo St V)
    (εo : Wit → VS → St → ℝ≥0∞) {δ : ℝ≥0∞},
    Lv.Hides δ → (∀ w v, (Zs w v).OuterZK (sim v) (εo w v)) →
    ∀ (w : Wit) (v : VS) (s : Option St) (E : Ωa × (Fin nh → Lv.Shown) × V → Prop),
      prCoin (fun u : ((Ωa × Ωp) × (Fin nh → Lv.Salt)) × Ωo =>
          (Zs w v).outcome u.1.1.1 u.1.1.2 u.1.2 u.2 = s ∧ E ((Zs w v).real refuse u.1.1.1 u.1.1.2 u.1.2 u.2)) ≤
        prCoin (fun u : (Ωa × (Fin nh → Lv.Rand)) × Ωs =>
          E (idealView refuse (sim v) s u.1.1 (fun i => Lv.ideal (u.1.2 i)) u.2)) + (s.elim 0 (εo w v) + nh * δ)

/-- **Z3: today's statement is the generic one at the ZK server's leaves**: `GenericRecursiveZK` and `Hm96Hides` give
`RecursiveZK` (the leaves `ZkView.leafH`, messages that read no commitment, one outcome, workers always honest, so the
refusal never shows, and `δ` the pinned key's hiding gap, `CROnly.hm96Hiding_gap`). -/
def RecoversRecursiveZK : Prop := GenericRecursiveZK → Hm96Hides → RecursiveZK

end Flock.Guarantees
