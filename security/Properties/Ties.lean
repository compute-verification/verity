module

public import Properties.Verity.Game
public import Definitions.Core.Registration
public import Flock.ProvedScope
public import NetworkCheck.Check

/-!
# What a protocol names of the registration

A protocol's answer or its parameters name things registration holds: a circuit, a link, a key, a registered value's
root. Each such tie joins two protocols, registration and the one that names, so it is stated here, in the
composition's area, and every one has one shape (`Registration.NamesRegistered`): what the answer names is `f x` for
some `x` the registration's view (`Registration.View`) holds, under a public map `f`. `f` is the identity, up to
projecting what the view holds to the field the answer names, for a key, a link, a memory leaf and a registered value's
root. For the circuit, the parameters name the circuit file and the view holds the Program's digest, so the answer
side is read through the one public map from the first to the second (`Registration.boundProgram`, the circuit file's
META `program_sha512`), and `f` then projects a declaration to its program.

* `Registration.DeclaredIsParams`: the Program the parameters' circuit file binds is the program the log declares for
  the day.
* `Registration.CertifiedIsParams`: a network certifier check's links, each with its key, are registered boundary links
  with that key pinned for their certifier.
* `Registration.RootIsRegistered`: a value's root an answer names is θ's registered root of that value.

None is read by a theorem yet, and no code checks any of them: each docstring says which check would make it a lemma.
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

/-- **The parameters' circuit is the one the log declares for day `j`**: the circuit file the parameters `θ` name
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
def DeclaredIsParams (V : Vocab d α) (tags : Flock.Tags) (tables : List (String × ByteArray)) (θ : ByteArray)
    (F : Array ByteArray) (v : View) (j : ℕ) : Prop :=
  ∃ p, boundProgram tags (V.circuitFile θ F) tables = some p ∧
    NamesRegistered {e | e ∈ v.declarations ∧ e.day = j} Declared.program p

/-- **The links network certifiers certify, each with its certifier's key**: the id of a boundary link of the view's
Site (`Site.Boundary`) with a network certifier at an end, beside that certifier's registered key (`Site.Loc.key`: its
LMS public key, I ‖ T[1], in lower-case hex, which registration's `owner` check holds to the auditor's own). -/
def certifiedKeys (v : View) : Set (String × String) :=
  {lk | ∃ k ∈ v.site.links, v.site.Boundary k ∧
    ∃ i ∈ [k.ends.1, k.ends.2], ∃ x ∈ v.site.loc? i, x.kind = .certifier ∧ lk = (k.id, x.key)}

/-- **A network certifier check's links and key are registered**: each link the check `p` reads, egress or ingress,
with its key `p.key` in lower-case hex (`Flock.hexOf`), is a boundary link of the view's Site with that key pinned for
its certifier (`certifiedKeys`; `NamesRegistered`, `f` the identity).

No theorem reads it, and the certifier's check doesn't read the registration: its link names and key are its
parameters. The code reads both off the registered Site (`NetworkCheck.Registered.registered`), but nothing holds the
check's parameters to that reading. It becomes a lemma when the certifier's parameters are the ones
`NetworkCheck.Registered.registered` returns at an accepted log's view, with a property that this reading is
`certifiedKeys`. -/
def CertifiedIsParams (p : NetworkCheck.Check.Params) (v : View) : Prop :=
  ∀ l ∈ p.egress.toList ++ p.ingress.toList, NamesRegistered (certifiedKeys v) id (l, Flock.hexOf p.key)

/-- **A registered value's root an answer names is θ's**: the root `root` an answer names for the value `value` is the
root θ registers for it (`NamesRegistered`, `f` = `Theta.root`).

No theorem reads it. It becomes a lemma when the check that reads the answer's root (sampled proofs' registered roots,
a protocol's parameters) reads it from θ in the view of an accepted log rather than from its own parameters. -/
def RootIsRegistered (value root : String) (v : View) : Prop :=
  NamesRegistered {t | t ∈ v.theta ∧ t.value = value} Theta.root root

end Registration

end Verity
