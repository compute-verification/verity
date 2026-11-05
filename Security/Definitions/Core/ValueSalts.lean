module

public import Definitions.Core.Game
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Prod
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# The value commitments' salts, the ZK server's: what a registration shows

The ZK server's value service holds a key `k` and derives every value commitment's salt from it and the commit session
`s` it draws (`Service.salts`). A session's view is the session, the public commitments of the stack's values under its
salts (`Service.out`) and their tags under `k` (`Service.tags`). A registration's record is the number `m` of sessions the
service's epoch opened, the registered session's view and a record tag under `k` over both (`Service.rtag`), and the
session loop admits a record whose record tag is the server's (`Service.admits`). The code is `Flock.Firewall.Salts`.

A serving stack commits its values in `m` sessions, sees each one's view and hands the loop a record (`Stack`): one of its
sessions' (`Choice`'s `inl`), which the service registers, or any record of its own making (`inr`). As a function of
everything it could read, the key among them: the key-separation premise (`Verity.Assumptions.ZkServerOwnUser`) is that
it is the same function under every key.

* `real`: the probability, over the key and the `m` sessions, that the loop admits the record the stack hands it and the
  record meets an event `A`;
* `ideal`: that of `A` on one record of a session whose salts, tags and record tag are uniform;
* `genuine`, `prfAdvantage`: that of `A` on one session's genuine record, and how much more often than on the ideal one,
  the gap that SHA-512 keyed by `k` as a PRF bounds;
* `forgeAdvantage`: how often a stack hands an admitted record that none of its sessions has.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

namespace ValueSalts

/-- The value service, as functions of its key `k : K`: a session's salts (`salts k s : Y`), the public commitments of
the values `v : V` under salts `y` (`out v y : O`), their tags (`tags k s o : T`), and a record tag over an epoch's
session count and the registered session's view (`rtag k m w : R`). -/
structure Service (K S Y V O T R : Type) where
  salts : K → S → Y
  out : V → Y → O
  tags : K → S → O → T
  rtag : K → ℕ → S × O × T → R

/-- A session's view: the session, its public commitments and their tags. -/
abbrev View (S O T : Type) : Type := S × O × T

/-- A registration's record: the number of sessions its epoch opened, the registered session's view and its record
tag. -/
abbrev RegistrationRecord (S O T R : Type) : Type := ℕ × View S O T × R

/-- What a stack hands the loop after `m` sessions: one of them, which the service registers, or a record of its own. -/
abbrev Choice (m : ℕ) (S O T R : Type) : Type := Fin m ⊕ RegistrationRecord S O T R

/-- A serving stack that commits its values in `m` sessions: what it hands the loop, as a function of the key, its values
and its sessions' views. -/
abbrev Stack (K V S O T R : Type) (m : ℕ) : Type := K → V → (Fin m → View S O T) → Choice m S O T R

namespace Service

variable {K S Y V O T R : Type} (sv : Service K S Y V O T R)

/-- Session `s`'s view of the values `v` under the key `k`. -/
def view (k : K) (v : V) (s : S) : View S O T :=
  (s, sv.out v (sv.salts k s), sv.tags k s (sv.out v (sv.salts k s)))

/-- The record the service answers when the stack registers session `s` of an epoch of `m` sessions. -/
def record (k : K) (m : ℕ) (v : V) (s : S) : RegistrationRecord S O T R :=
  (m, sv.view k v s, sv.rtag k m (sv.view k v s))

/-- The loop admits a record whose record tag is the server's for its session count and view. -/
def admits (k : K) (ρ : RegistrationRecord S O T R) : Prop :=
  ρ.2.2 = sv.rtag k ρ.1 ρ.2.1

/-- The record a stack's choice hands the loop, its sessions `s`. -/
def handed {m : ℕ} (k : K) (v : V) (s : Fin m → S) : Choice m S O T R → RegistrationRecord S O T R
  | .inl i => sv.record k m v (s i)
  | .inr ρ => ρ

/-- The views of the sessions `s`. -/
def views {m : ℕ} (k : K) (v : V) (s : Fin m → S) : Fin m → View S O T :=
  fun i => sv.view k v (s i)

/-- The probability, over a uniform key and `m` uniform sessions, that the loop admits the record the stack `st` hands it
and that record meets `A`. -/
noncomputable def real [Fintype K] [Fintype S] (m : ℕ) (v : V) (st : Stack K V S O T R m)
    (A : RegistrationRecord S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : K × (Fin m → S) =>
    sv.admits p.1 (sv.handed p.1 v p.2 (st p.1 v (sv.views p.1 v p.2))) ∧
      A (sv.handed p.1 v p.2 (st p.1 v (sv.views p.1 v p.2)))

/-- The probability that `A` holds of the record of one session of an epoch of `m`, its salts, tags and record tag
uniform. -/
noncomputable def ideal [Fintype S] [Fintype Y] [Fintype T] [Fintype R] (m : ℕ) (v : V)
    (A : RegistrationRecord S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : S × Y × T × R => A (m, (p.1, sv.out v p.2.1, p.2.2.1), p.2.2.2)

/-- The probability that `A` holds of the genuine record of one session of an epoch of `m`. -/
noncomputable def genuine [Fintype K] [Fintype S] (m : ℕ) (v : V) (A : RegistrationRecord S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : K × S => A (sv.record p.1 m v p.2)

/-- How much more often `A` holds of a genuine record than of an ideal one: the gap SHA-512 keyed by the server's key, as
a PRF, bounds. -/
noncomputable def prfAdvantage [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R] (m : ℕ) (v : V)
    (A : RegistrationRecord S O T R → Prop) : ℝ≥0∞ :=
  sv.genuine m v A - sv.ideal m v A

/-- The probability that `F`, a stack's choice from its sessions' views, hands a record the loop admits and that none of
its `m` sessions has: a forged record tag. -/
noncomputable def forgeAdvantage [Fintype K] [Fintype S] (m : ℕ) (v : V)
    (F : (Fin m → View S O T) → Choice m S O T R) : ℝ≥0∞ :=
  Game.prCoin fun p : K × (Fin m → S) =>
    ∃ ρ, F (sv.views p.1 v p.2) = .inr ρ ∧ sv.admits p.1 ρ ∧ ∀ i, ρ ≠ sv.record p.1 m v (p.2 i)

end Service

end ValueSalts

end Verity
