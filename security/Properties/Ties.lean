module

public import Properties.Verity.Game
public import Definitions.Accounting.Registration.View
public import Flock.ProvedScope
public import NetworkCheck.Check

/-!
# What a protocol names of the registration

A protocol's answer or its parameters name things registration holds: a circuit, a link, a key, a registered value. Each such tie joins two protocols, registration and the one that names, so it is stated in the composition's
area, under a prefix of its own (`Properties.Ties`), the one module there that reads a protocol's code (Flock's verifier's
reading of a circuit file, the network certifier's parameters). Every tie has one shape
(`Registration.NamesRegistered`): what the answer names is `f x` for some `x` the registration holds, under a public map
`f`, and it is stated at the head the answer names (`Registration.AtHead`): `h` heads a prefix of the accepted log, and
what is registered is what that prefix registers. No entry retires what an earlier one registered, so what a prefix
registers every longer prefix registers too (`Proofs.Properties.Ties`). `f` is the identity, up to
projecting what the view holds to the field the answer names, for a key, a link, a memory leaf and a registered θ
entry. For the circuit, the parameters name the circuit file and the view holds the Program's digest, so the answer
side is read through the one public map from the first to the second (`Registration.boundProgram`, the circuit file's
META `program_sha512`), and `f` then projects a declaration to its program.

* `Registration.DeclaredIsParams`: the Program the parameters' circuit file binds is the program the prefix declares
  for the day.
* `Registration.CertifiedIsParams`: a network certifier check's links, each with its key, are registered boundary links
  with that key pinned for their certifier.
* `Registration.RootIsRegistered`: a θ entry an answer names, root, rows and domain, is the one θ registers for its
  value.

Each is a predicate on a view (`…In`), and the tie is it at a head of the log's chain. None is read by a theorem yet, and no code checks any of them: each docstring says which check would make it a lemma.
-/

@[expose] public section

namespace Verity

namespace Registration

/-- **What an answer names is something the registration holds**: `y`, what a protocol's answer or parameters name,
is `f x` for some `x` in `holds`, what the registration's view holds, under the public map `f`. -/
def NamesRegistered {X Y : Type} (holds : Set X) (f : X → Y) (y : Y) : Prop := ∃ x ∈ holds, f x = y

/-- **The Program a circuit file binds**: the META `program_sha512` (`Flock.ProvedScope.metaProgram`) of the circuit
file `cf` as the verifier reads it at the statement's tags and tables (`Flock.Desc.parse`, the reading
`Flock.Bridge.circuitOfFile` takes the circuit from). It is the SHA-512 of the Program descriptor's canonical JSON
without `annotations`, in lower-case hex: the digest Flock's verifier holds its own Program to
(`Flock.ProvedScope.programWhyP`) and the one a declaration's `program` is. `none` when the file doesn't read or its
META binds no Program. -/
def boundProgram (tags : Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)) : Option String :=
  match Flock.Desc.parse tags cf tables with
  | .ok D => Flock.ProvedScope.metaProgram D.unroll
  | .error _ => none

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The parameters' circuit is the one the view declares for day `j`**: the circuit file the parameters `θ` name
among the committed files `F` (`Vocab.circuitFile`, the file whose SHA-512 is `Vocab.programDigest θ`) binds a Program
(`boundProgram`, read at the statement's `tags` and `tables`), and that Program's digest is the program of the view's
declaration of day `j` (`NamesRegistered`, `f` = `Declared.program`).

The two digests are of different objects: `programDigest θ` is the circuit file's SHA-512, and a declaration's
`program` is the Program descriptor's. `boundProgram` is the public map from the first to the second. The log declares no
digest of the public file, so this ties only the circuit.

No theorem reads it, and no Lean check holds the circuit file the parameters name to the day's declaration: Flock's
verifier holds its own Program to the circuit file's META, and registration's view holds the declaration, but nothing
compares the two. It becomes a lemma when the composite's check refuses parameters whose circuit file binds a Program
other than the day's declared one, or when the parameters are derived from the log. -/
def DeclaredIn (V : Vocab d α) (tags : Flock.Tags) (tables : List (String × ByteArray)) (θ : ByteArray)
    (F : Array ByteArray) (j : ℕ) (v : View) : Prop :=
  ∃ p, boundProgram tags (V.circuitFile θ F) tables = some p ∧
    NamesRegistered {e | e ∈ v.declarations ∧ e.day = j} Declared.program p

/-- **The parameters' circuit is the one declared for day `j` at the head `h`**: `DeclaredIn` at the view of the prefix
of the log `h` heads (`AtHead`, under the digests `D` and the composite `c`). -/
def DeclaredIsParams (V : Vocab d α) (tags : Flock.Tags) (tables : List (String × ByteArray)) (θ : ByteArray)
    (F : Array ByteArray) (j : ℕ) (D : Digests) (c : Composite) (log : Registration) (h : String) : Prop :=
  AtHead D c log h (DeclaredIn V tags tables θ F j)

/-- **The links network certifiers certify, each with its certifier's key**: the id of a boundary link of the view's
Site (`Site.Boundary`) with a network certifier at an end, beside that certifier's registered key (`Site.Loc.key`: its
LMS public key, I ‖ T[1], in lower-case hex, which registration's `owner` check holds to the auditor's own). -/
def certifiedKeys (v : View) : Set (String × String) :=
  {lk | ∃ k ∈ v.site.links, v.site.Boundary k ∧
    ∃ i ∈ [k.ends.1, k.ends.2], ∃ x ∈ v.site.loc? i, x.kind = .certifier ∧ lk = (k.id, x.key)}

/-- **A network certifier check's links and key are registered in the view**: each link the check `p` reads, egress or
ingress, with its key `p.key` in lower-case hex (`Flock.hexOf`), is a boundary link of the view's Site with that key
pinned for its certifier (`certifiedKeys`; `NamesRegistered`, `f` the identity).

The check's parameters hold one key today. Once the certifier's parameters carry key entries (each a key, its session
and its start), this becomes per entry: each entry's key pinned for its link in the prefix that entry's head names.

No theorem reads it, and the certifier's check doesn't read the registration: its link names and key are its
parameters. The code reads both off the registered Site (`NetworkCheck.Registered.registered`), but nothing holds the
check's parameters to that reading. It becomes a lemma when the certifier's parameters are the ones
`NetworkCheck.Registered.registered` returns at an accepted log's view, with a property that this reading is
`certifiedKeys`. -/
def CertifiedIn (p : NetworkCheck.Check.Params) (v : View) : Prop :=
  ∀ l ∈ p.egress.toList ++ p.ingress.toList, NamesRegistered (certifiedKeys v) id (l, Flock.hexOf p.key)

/-- **A network certifier check's links and key are registered at the head `h`**: `CertifiedIn` at the view of the
prefix of the log `h` heads (`AtHead`). -/
def CertifiedIsParams (p : NetworkCheck.Check.Params) (D : Digests) (c : Composite) (log : Registration) (h : String) :
    Prop :=
  AtHead D c log h (CertifiedIn p)

/-- **A registered value an answer names is the view's θ's, whole**: the θ entry `t` an answer names (a value's name,
its root, its rows' shape and the domain its leaves are hashed under) is the entry the view's θ registers for that
value's name (`NamesRegistered`, `f` the identity). A tie on the root alone could not say that the leaves, words and
domain an answer reads are the registered ones.

No theorem reads it. It becomes a lemma when the check that reads the answer's θ entry (sampled proofs' registered
values, a protocol's parameters) reads it from θ in the view of an accepted log rather than from its own parameters. -/
def RootIn (t : Theta) (v : View) : Prop :=
  NamesRegistered {x | x ∈ v.theta ∧ x.value = t.value} id t

/-- **A registered value an answer names is θ's at the head `h`**: `RootIn` at the view of the prefix of the log `h`
heads (`AtHead`). -/
def RootIsRegistered (t : Theta) (D : Digests) (c : Composite) (log : Registration) (h : String) : Prop :=
  AtHead D c log h (RootIn t)

end Registration

end Verity
