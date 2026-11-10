module

public import Definitions.Core.Game

/-!
# The value commitments' salts, the ZK server's: what a registration shows

The ZK server's value service holds a key `k` and derives every value commitment's salt from it and the commit session
`s` it samples (`Service.salts`). A session's view is the session, the public commitments of the stack's values under its
salts (`Service.out`) and their tags under `k` (`Service.tags`). A registration's record is its number `n` on the ZK
server's record, the number `m` of sessions the service's epoch opened, the registered session's view and a record tag
under `k` over all three (`Service.rtag`), and the session loop admits a record whose record tag is the server's
(`Service.admits`). The code is `Flock.Firewall.Salts`.

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

A stream of `r` registrations, registration `i` of an epoch of `m i` sessions (`StreamStack`): the loop admits each
registration once, oldest first, and passes over (`skip`) one no statement reads before a later one is read, each a
failure of the stream, which stops at its failure budget (`Core.Firewall.admitReads`). What the loop admits is each
registration's record, or none where it passed it over (`StreamRecord`); `streamReal`, `streamIdeal`, `streamGenuine`,
`streamPrfAdvantage` and `streamForgeAdvantage` are the five above over the stream, at each of the stack's choices
(`streamChoices`: a session of each registration's epoch, or none, at most `b` none).
-/

@[expose] public section

namespace Verity

open scoped ENNReal

namespace ValueSalts

/-- The value service, as functions of its key `k : K`: a session's salts (`salts k s : Y`), the public commitments of
the values `v : V` under salts `y` (`out v y : O`), their tags (`tags k s o : T`), and a record tag over a registration's
number, its epoch's session count and the registered session's view (`rtag k n m w : R`). -/
structure Service (K S Y V O T R : Type) where
  salts : K → S → Y
  out : V → Y → O
  tags : K → S → O → T
  rtag : K → ℕ → ℕ → S × O × T → R

/-- A session's view: the session, its public commitments and their tags. -/
abbrev View (S O T : Type) : Type := S × O × T

/-- A registration's record: its number on the ZK server's record, the number of sessions its epoch opened, the
registered session's view and its record tag. -/
abbrev RegistrationRecord (S O T R : Type) : Type := ℕ × ℕ × View S O T × R

/-- What a stack hands the loop after `m` sessions: one of them, which the service registers, or a record of its own. -/
abbrev Choice (m : ℕ) (S O T R : Type) : Type := Fin m ⊕ RegistrationRecord S O T R

/-- A serving stack that commits its values in `m` sessions: what it hands the loop, as a function of the key, its values
and its sessions' views. -/
abbrev Stack (K V S O T R : Type) (m : ℕ) : Type := K → V → (Fin m → View S O T) → Choice m S O T R

/-- What a stack hands the loop over a stream of `r` registrations, registration `i` of an epoch of `m i` sessions: at
each, `none` where the loop passes it over (a failure of the stream), else its choice there. -/
abbrev StreamChoice (r : ℕ) (m : Fin r → ℕ) (S O T R : Type) : Type := (i : Fin r) → Option (Choice (m i) S O T R)

/-- A serving stack over a stream of `r` registrations: what it hands the loop, as a function of the key, each
registration's values and every session's view. -/
abbrev StreamStack (K V S O T R : Type) (r : ℕ) (m : Fin r → ℕ) : Type :=
  K → (Fin r → V) → ((i : Fin r) → Fin (m i) → View S O T) → StreamChoice r m S O T R

/-- What the loop admits of a stream of `r` registrations: each one's record, or `none` where it passed it over. -/
abbrev StreamRecord (r : ℕ) (S O T R : Type) : Type := Fin r → Option (RegistrationRecord S O T R)

/-- The registrations a choice of sessions, one of each registration's epoch or none, passes over. -/
def passed {r : ℕ} {m : Fin r → ℕ} (d : (i : Fin r) → Option (Fin (m i))) : Finset (Fin r) :=
  Finset.univ.filter fun i => d i = none

/-- The stack's choices the loop can admit over a stream: a session of each registration's epoch, or none, with at most
`b` none. There are `∑_{P ⊆ Fin r, |P| ≤ b} ∏_{i ∉ P} m i` of them. -/
def streamChoices (r : ℕ) (m : Fin r → ℕ) (b : ℕ) : Finset ((i : Fin r) → Option (Fin (m i))) :=
  Finset.univ.filter fun d => (passed d).card ≤ b

namespace Service

variable {K S Y V O T R : Type} (sv : Service K S Y V O T R)

/-- Session `s`'s view of the values `v` under the key `k`. -/
def view (k : K) (v : V) (s : S) : View S O T :=
  (s, sv.out v (sv.salts k s), sv.tags k s (sv.out v (sv.salts k s)))

/-- The record the service answers when the stack registers session `s` of an epoch of `m` sessions as registration
`n`. -/
def record (k : K) (n m : ℕ) (v : V) (s : S) : RegistrationRecord S O T R :=
  (n, m, sv.view k v s, sv.rtag k n m (sv.view k v s))

/-- The loop admits a record whose record tag is the server's for its number, session count and view. -/
def admits (k : K) (ρ : RegistrationRecord S O T R) : Prop :=
  ρ.2.2.2 = sv.rtag k ρ.1 ρ.2.1 ρ.2.2.1

/-- The record a stack's choice hands the loop as registration `n`, its sessions `s`. -/
def handed {m : ℕ} (k : K) (n : ℕ) (v : V) (s : Fin m → S) : Choice m S O T R → RegistrationRecord S O T R
  | .inl i => sv.record k n m v (s i)
  | .inr ρ => ρ

/-- The views of the sessions `s`. -/
def views {m : ℕ} (k : K) (v : V) (s : Fin m → S) : Fin m → View S O T :=
  fun i => sv.view k v (s i)

/-- The probability, over a uniform key and `m` uniform sessions, that the loop admits the record the stack `st` hands it
as registration `n` and that record meets `A`. -/
noncomputable def real [Fintype K] [Fintype S] (n m : ℕ) (v : V) (st : Stack K V S O T R m)
    (A : RegistrationRecord S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : K × (Fin m → S) =>
    sv.admits p.1 (sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2))) ∧
      A (sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2)))

/-- The probability that `A` holds of the record of registration `n`, one session of an epoch of `m`, its salts, tags
and record tag uniform. -/
noncomputable def ideal [Fintype S] [Fintype Y] [Fintype T] [Fintype R] (n m : ℕ) (v : V)
    (A : RegistrationRecord S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : S × Y × T × R => A (n, m, (p.1, sv.out v p.2.1, p.2.2.1), p.2.2.2)

/-- The probability that `A` holds of the genuine record of registration `n`, one session of an epoch of `m`. -/
noncomputable def genuine [Fintype K] [Fintype S] (n m : ℕ) (v : V) (A : RegistrationRecord S O T R → Prop) :
    ℝ≥0∞ :=
  Game.prCoin fun p : K × S => A (sv.record p.1 n m v p.2)

/-- How much more often `A` holds of a genuine record than of an ideal one: the gap SHA-512 keyed by the server's key, as
a PRF, bounds. -/
noncomputable def prfAdvantage [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R] (n m : ℕ) (v : V)
    (A : RegistrationRecord S O T R → Prop) : ℝ≥0∞ :=
  sv.genuine n m v A - sv.ideal n m v A

/-- The probability that `F`, a stack's choice from its sessions' views, hands as registration `n` a record the loop
admits that none of its `m` sessions has: a forged record tag. -/
noncomputable def forgeAdvantage [Fintype K] [Fintype S] (n m : ℕ) (v : V)
    (F : (Fin m → View S O T) → Choice m S O T R) : ℝ≥0∞ :=
  Game.prCoin fun p : K × (Fin m → S) =>
    ∃ ρ, F (sv.views p.1 v p.2) = .inr ρ ∧ sv.admits p.1 ρ ∧ ∀ i, ρ ≠ sv.record p.1 n m v (p.2 i)

section Stream

variable {r : ℕ} {m : Fin r → ℕ}

/-- The views of a stream's sessions `s`, registration `i`'s of its values `v i`. -/
def streamViews (k : K) (v : Fin r → V) (s : (i : Fin r) → Fin (m i) → S) : (i : Fin r) → Fin (m i) → View S O T :=
  fun i j => sv.view k (v i) (s i j)

/-- What the loop holds of a stream at the stack's choice `c`: at registration `i`, the record `c` hands as registration
`i`, or none where `c` passes it over. -/
def streamHanded (k : K) (v : Fin r → V) (s : (i : Fin r) → Fin (m i) → S) (c : StreamChoice r m S O T R) :
    StreamRecord r S O T R :=
  fun i => (c i).map (sv.handed k i (v i) (s i))

/-- The loop admits a stream that passes over at most `b` registrations, each a failure of the stream, and whose every
other registration's record is the server's at that registration's number. -/
def streamAdmits (k : K) (b : ℕ) (ρs : StreamRecord r S O T R) : Prop :=
  (Finset.univ.filter fun i => ρs i = none).card ≤ b ∧ ∀ i ρ, ρs i = some ρ → ρ.1 = i ∧ sv.admits k ρ

/-- The probability, over a uniform key and every registration's uniform sessions, that the loop admits the stream the
stack `st` hands it, passing over at most `b`, and that stream meets `A`. -/
noncomputable def streamReal [Fintype K] [Fintype S] (b : ℕ) (v : Fin r → V) (st : StreamStack K V S O T R r m)
    (A : StreamRecord r S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : K × ((i : Fin r) → Fin (m i) → S) =>
    sv.streamAdmits p.1 b (sv.streamHanded p.1 v p.2 (st p.1 v (sv.streamViews p.1 v p.2))) ∧
      A (sv.streamHanded p.1 v p.2 (st p.1 v (sv.streamViews p.1 v p.2)))

/-- The probability that `A` holds of a stream that passes over the registrations `P` and whose every other registration
`i` is the record of one session of an epoch of `m i`, its salts, tags and record tag uniform, independent across the
registrations. -/
noncomputable def streamIdeal [Fintype S] [Fintype Y] [Fintype T] [Fintype R] (m : Fin r → ℕ) (v : Fin r → V)
    (P : Finset (Fin r)) (A : StreamRecord r S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : Fin r → S × Y × T × R =>
    A fun i => if i ∈ P then none
      else some ((i : ℕ), m i, ((p i).1, sv.out (v i) (p i).2.1, (p i).2.2.1), (p i).2.2.2)

/-- The probability that `A` holds of the genuine records of the sessions `d` chooses, none where it chooses none. -/
noncomputable def streamGenuine [Fintype K] [Fintype S] (v : Fin r → V) (d : (i : Fin r) → Option (Fin (m i)))
    (A : StreamRecord r S O T R → Prop) : ℝ≥0∞ :=
  Game.prCoin fun p : K × ((i : Fin r) → Fin (m i) → S) =>
    A fun i => (d i).map fun j => sv.record p.1 i (m i) (v i) (p.2 i j)

/-- How much more often `A` holds of the genuine records of the sessions `d` chooses than of ideal ones: the gap SHA-512
keyed by the server's key, as a PRF, bounds, at the fixed choice `d`. -/
noncomputable def streamPrfAdvantage [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R] (v : Fin r → V)
    (d : (i : Fin r) → Option (Fin (m i))) (A : StreamRecord r S O T R → Prop) : ℝ≥0∞ :=
  sv.streamGenuine v d A - sv.streamIdeal m v (passed d) A

/-- The probability that `F`, a stack's choice from a stream's views, hands at some registration `i` a record the loop
admits there that none of the sessions of `i`'s epoch has: a forged record tag. -/
noncomputable def streamForgeAdvantage [Fintype K] [Fintype S] (v : Fin r → V)
    (F : ((i : Fin r) → Fin (m i) → View S O T) → StreamChoice r m S O T R) : ℝ≥0∞ :=
  Game.prCoin fun p : K × ((i : Fin r) → Fin (m i) → S) =>
    ∃ i ρ, F (sv.streamViews p.1 v p.2) i = some (.inr ρ) ∧ ρ.1 = i ∧ sv.admits p.1 ρ ∧
      ∀ j, ρ ≠ sv.record p.1 i (m i) (v i) (p.2 i j)

end Stream

end Service

end ValueSalts

end Verity
