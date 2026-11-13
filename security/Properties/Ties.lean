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
what is registered is what that prefix registers. A head names one prefix: two prefixes of accepted logs with one
head are one, or a search of their entries finds two whose digests collide (`Properties.Registration.HeadBinds`), and
a prefix's view is the same read under the names of any accepted log that holds it. No entry retires what an earlier
one registered, so what a prefix registers every longer prefix registers too (`Proofs.Properties.Ties`). `f` is the identity, up to
projecting what the view holds to the field the answer names, for a key, a link, a memory leaf and a registered θ
entry. For the circuit, the parameters name the circuit file and the view holds the Program's digest, so the answer
side is read through the one public map from the first to the second (`Registration.boundProgram`, the circuit file's
META `program_sha512`), and `f` then projects a declaration to its program.

* `Registration.DeclaredIsParams`: the Program the circuit file the verifier reads binds is the program the prefix
  declares for the day the window lies in.
* `Registration.CertifiedIsParams`: a network certifier check's links, each with its key, are exactly the registered
  boundary links with a certifier at an end, each with the key pinned for its certifier.
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

/-- **The parameters' circuit is the one the view declares for the window's day**: the days are the cut `cut` of the
monitoring period `m`'s span into `n` consecutive spans, as in `RunsDeclared`, and the window's span `w` lies within
day `j`'s; the circuit file the verifier reads at the parameters `θ` and the committed files `F` (`Vocab.read`, which
also gives the statement's tags and tables) binds a Program (`boundProgram`); and that Program's digest is the program
of the view's declaration of day `j` (`NamesRegistered`, `f` = `Declared.program`). Nothing it reads is free: the tags,
the tables and the circuit file are the verifier's reading, and the day is the one the window lies in.

The two digests are of different objects: `programDigest θ` is the circuit file's SHA-512, and a declaration's
`program` is the Program descriptor's. `boundProgram` is the public map from the first to the second. The log declares no
digest of the public file, so this ties only the circuit.

No theorem reads it, and no Lean check holds the circuit file the parameters name to the day's declaration: Flock's
verifier holds its own Program to the circuit file's META, and registration's view holds the declaration, but nothing
compares the two. It becomes a lemma when the composite's check refuses parameters whose circuit file binds a Program
other than the day's declared one, or when the parameters are derived from the log. -/
def DeclaredIn (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (m : MonitoringPeriod) {n : ℕ}
    (cut : m.span.Cut n) (w : TickSpan) (v : View) : Prop :=
  ∃ j : Fin n, w.Within (cut.part j) ∧
    ∃ p, boundProgram (V.read θ F).tags (V.read θ F).circuitFile (V.read θ F).tables = some p ∧
      NamesRegistered {e | e ∈ v.declarations ∧ e.day = j} Declared.program p

/-- **The parameters' circuit is the one declared for the window's day at the head `h`**: `DeclaredIn` at the view of
the prefix of the log `h` heads (`AtHead`, under the digests `D` and the composite `c`). -/
def DeclaredIsParams (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (m : MonitoringPeriod) {n : ℕ}
    (cut : m.span.Cut n) (w : TickSpan) (D : Digests) (c : Composite) (log : Registration) (h : String) : Prop :=
  AtHead D c log h (DeclaredIn V θ F m cut w)

/-- **The links network certifiers certify, each with its certifier's key**: the id of a boundary link of the view's
Site (`Site.Boundary`) with a network certifier at an end, beside that certifier's registered key (`Site.Loc.key`: its
LMS public key, I ‖ T[1], in lower-case hex, which registration's `owner` check holds to the auditor's own). -/
def certifiedKeys (v : View) : Set (String × String) :=
  {lk | ∃ k ∈ v.site.links, v.site.Boundary k ∧
    ∃ i ∈ [k.ends.1, k.ends.2], ∃ x ∈ v.site.loc? i, x.kind = .certifier ∧ lk = (k.id, x.key)}

/-- **A network certifier check reads exactly the view's certified links, with their keys**: the links the check `p`
reads, egress and ingress, each with its key `p.key` in lower-case hex (`Flock.hexOf`), are the view's boundary links
with a network certifier at an end, each with the key pinned for its certifier (`certifiedKeys`). It is an equality, so
a registered certified link the check doesn't read fails it as a link the check reads unregistered does, and a check
reading no link meets it only where no certified link is registered.

The check's parameters hold one key today, so it holds only where every registered certifier pins that key. Once the
certifier's parameters carry key entries (each a key, its session and its start), this becomes per entry: each entry's
key pinned for its link in the prefix that entry's head names.

No theorem reads it, and the certifier's check doesn't read the registration: its link names and key are its
parameters. The code reads both off the registered Site (`NetworkCheck.Registered.registered`), but nothing holds the
check's parameters to that reading. It becomes a lemma when the certifier's parameters are the ones
`NetworkCheck.Registered.registered` returns at an accepted log's view, with a property that this reading is
`certifiedKeys`. -/
def CertifiedIn (p : NetworkCheck.Check.Params) (v : View) : Prop :=
  certifiedKeys v = (fun l => (l, Flock.hexOf p.key)) '' {l | l ∈ p.egress.toList ++ p.ingress.toList}

/-- **A network certifier check reads exactly the certified links registered at the head `h`**: `CertifiedIn` at the
view of the prefix of the log `h` heads (`AtHead`). -/
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
