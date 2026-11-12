module

public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Algebra.BigOperators.Group.Multiset.Defs
public import Mathlib.Algebra.Field.Defs
public import Mathlib.Data.Fintype.Card
public import Mathlib.FieldTheory.Finite.GaloisField

/-!
# Offline memory checking

Offline memory checking (Blum, Evans, Gemmell, Kannan and Naor) checks a log of memory accesses without keeping the
memory. Each access reads an address, finding a value and the time it was written, and writes a value there at its own
time (`Access`); `ram` is the memory the log describes. The check (`Checked`) asks that times increase along the log, that
each access reads a time before its own, and one equality of multisets of records (address, value, time): the records the
reads consume with the final contents of the touched addresses, against the records the writes produce with their initial
contents at time `0`. What an accepted log reads is `Verity.Guarantees.MemoryChecked`.

The verifier tests that equality by a fingerprint over a finite field `F`, at coins it draws after both multisets are
committed. A record is packed into `m` chunks `c : Fin m → F`; its element at a coin `γ` is `∑ i, γ^i · c i` (`element`),
and a multiset of packed records has the fingerprint `∏ c, (α + element γ c)` at coins `α` and `γ` (`fingerprint`). The
same `element`, at a coin of its own, prints a key packed into `m` chunks as one field element. Packing is the caller's:
an injective one keeps different multisets of records different. The deployed field is GF(2^128), Mathlib's
`GaloisField 2 128`, finite by `gf128Fintype`.

What a collision of each costs is `Verity.Guarantees.FingerprintCollision` and `Verity.Guarantees.PrintCollision`.
-/

@[expose] public section

namespace Verity.MemCheck

/-- One access of a memory log: it reads the address `a`, finding the value `v` written there at time `t`, and writes the
value `w` there at its own time `now`. -/
structure Access (A V : Type) where
  /-- The address it accesses. -/
  a : A
  /-- The value it reads. -/
  v : V
  /-- The time the value it reads was written. -/
  t : ℕ
  /-- The value it writes. -/
  w : V
  /-- Its own time. -/
  now : ℕ

/-- The memory a log leaves, from the initial values `v0`: at each address its last write's value and time, else its
initial value at time `0`. -/
def ram {A V : Type} [DecidableEq A] (v0 : A → V) (log : List (Access A V)) : A → V × ℕ :=
  log.foldl (fun s x => Function.update s x.a (x.w, x.now)) fun a => (v0 a, 0)

/-- The records a log's accesses read: `(a, v, t)` each. -/
def reads {A V : Type} (log : List (Access A V)) : Multiset (A × V × ℕ) :=
  (log.map fun x => (x.a, x.v, x.t) : List (A × V × ℕ))

/-- The records a log's accesses write: `(a, w, now)` each. -/
def writes {A V : Type} (log : List (Access A V)) : Multiset (A × V × ℕ) :=
  (log.map fun x => (x.a, x.w, x.now) : List (A × V × ℕ))

/-- What offline memory checking accepts of a log, its set of touched addresses, the initial values `v0` and the final
contents `fin`: times increase along the log, each access reads a time before its own, and the records the reads consume
with the touched addresses' final contents are, as a multiset, the records the writes produce with their initial values at
time `0`. -/
def Checked {A V : Type} (v0 : A → V) (log : List (Access A V)) (touched : Finset A) (fin : A → V × ℕ) : Prop :=
  log.Pairwise (fun x y => x.now < y.now) ∧ (∀ x ∈ log, x.t < x.now) ∧
    reads log + touched.val.map (fun a => (a, fin a)) = writes log + touched.val.map fun a => (a, v0 a, 0)

/-- A log read its memory correctly: each access found the value and time `ram` holds at its address before it, each
touched address's final content is what `ram` holds there after the log, and every address the log accesses is touched. -/
def ReadsCorrect {A V : Type} [DecidableEq A] (v0 : A → V) (log : List (Access A V)) (touched : Finset A)
    (fin : A → V × ℕ) : Prop :=
  (∀ i (h : i < log.length), (log[i].v, log[i].t) = ram v0 (log.take i) log[i].a) ∧
    (∀ a ∈ touched, fin a = ram v0 log a) ∧ ∀ x ∈ log, x.a ∈ touched

/-- A record packed into `m` chunks `c`, read at the coin `γ`: `∑ i, γ^i · c i`. -/
def element {F : Type} [Field F] {m : ℕ} (γ : F) (c : Fin m → F) : F :=
  ∑ i : Fin m, γ ^ (i : ℕ) * c i

/-- The fingerprint of a multiset `A` of records, each packed into `m` chunks, at the coins `α` and `γ`:
`∏ c ∈ A, (α + element γ c)`. -/
def fingerprint {F : Type} [Field F] {m : ℕ} (α γ : F) (A : Multiset (Fin m → F)) : F :=
  (A.map fun c => α + element γ c).prod

/-- GF(2^128), the field of the deployed memory check's coins, as a finite type: Mathlib's Galois field of order
`2^128` is finite. -/
noncomputable instance gf128Fintype : Fintype (GaloisField 2 128) := Fintype.ofFinite _

end Verity.MemCheck
