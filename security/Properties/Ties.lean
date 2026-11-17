module

public import Properties.Verity.Game
public import Definitions.Registration.View
public import Flock.ProvedScope
public import NetworkCheck.Check

/-! What a protocol names of the registration, each tie in the shape `Registration.NamesRegistered`: no theorem reads one,
and no code checks one. -/

@[expose] public section

namespace Verity

namespace Registration

def NamesRegistered {X Y : Type} (holds : Set X) (f : X → Y) (y : Y) : Prop := ∃ x ∈ holds, f x = y

/-- The SHA-512 of the Program descriptor's canonical JSON without `annotations`, in lower-case hex, as the circuit file's
META binds it; `none` when the file doesn't read or binds no Program. -/
def boundProgram (tags : Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)) : Option String :=
  match Flock.Desc.parse tags cf tables with
  | .ok D => Flock.ProvedScope.metaProgram D.unroll
  | .error _ => none

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- `programDigest θ` is the circuit file's SHA-512 and a declaration's `program` the Program descriptor's; `boundProgram`
maps one to the other. No theorem reads this, and no Lean check holds the parameters' circuit file to the day's
declaration. -/
def DeclaredIn (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (m : MonitoringPeriod) {n : ℕ}
    (cut : m.span.Cut n) (w : TickSpan) (v : View) : Prop :=
  ∃ j : Fin n, w.Within (cut.part j) ∧
    ∀ r ∈ (V.read θ F).members,
      ∃ p, boundProgram r.tags r.circuitFile r.tables = some p ∧
        NamesRegistered {e | e ∈ v.declarations ∧ e.day = j} Declared.program p

def DeclaredIsParams (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (m : MonitoringPeriod) {n : ℕ}
    (cut : m.span.Cut n) (w : TickSpan) (D : Digests) (c : Composite) (log : Registration) (h : String) : Prop :=
  AtHead D c log h (DeclaredIn V θ F m cut w)

/-- Each key is its certifier's LMS public key, `I ‖ T[1]`, in lower-case hex. -/
def certifiedKeys (v : View) : Set (String × String) :=
  {lk | ∃ k ∈ v.site.links, v.site.Boundary k ∧
    ∃ i ∈ [k.ends.1, k.ends.2], ∃ x ∈ v.site.loc? i, x.kind = .certifier ∧ lk = (k.id, x.key)}

/-- An equality, so a registered certified link the check doesn't read fails it; the check's parameters hold one key, so it
holds only where every registered certifier pins that key. No theorem reads it, and nothing holds the check's
parameters to the registered Site. -/
def CertifiedIn (p : NetworkCheck.Check.Params) (v : View) : Prop :=
  certifiedKeys v = (fun l => (l, Flock.hexOf p.key)) '' {l | l ∈ p.egress.toList ++ p.ingress.toList}

def CertifiedIsParams (p : NetworkCheck.Check.Params) (D : Digests) (c : Composite) (log : Registration) (h : String) :
    Prop :=
  AtHead D c log h (CertifiedIn p)

/-- A tie on the root alone could not say that the leaves, words and domain an answer reads are the registered ones. No
theorem reads it. -/
def RootIn (t : Theta) (v : View) : Prop :=
  NamesRegistered {x | x ∈ v.theta ∧ x.value = t.value} id t

def RootIsRegistered (t : Theta) (D : Digests) (c : Composite) (log : Registration) (h : String) : Prop :=
  AtHead D c log h (RootIn t)

def KeyIn (k : String) (v : View) : Prop :=
  NamesRegistered {x | x ∈ v.site.locs ∧ x.kind.auditor = true} Site.Loc.key k

/-- The period's account (`Core.Account`) looks an answer's key up at the period's last head, so a key registered later in
the period than the answer is still found there. No theorem reads this, and no code checks it. -/
def KeyIsRegistered (k : String) (D : Digests) (c : Composite) (log : Registration) (h : String) : Prop :=
  AtHead D c log h (KeyIn k)

end Registration

end Verity
