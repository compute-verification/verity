import NetworkCheck.Period
import Mathlib.Algebra.BigOperators.Fin

/-!
# The network period check's bound: what its `K` counts in each form (`NetworkCheck.Period`)

On acceptance, `NetworkCheck.Period.check` publishes `K`, the sum of a ceiling of `log2` of each factor of
`NetTiming.Guarantees.PeriodLeakAccepted`'s count at the number the check reads for it: `terms`'s in the clock form,
`rowsTerms`'s in the rows form. A sum of ceilings is at least the ceiling of the sum, so `K` is at least `⌈log2⌉` of
the count, not equal to it; `ClockKBoundsCount` and `RowsKBoundsCount` say the count at the check's numbers is at most
`2 ^ K`. `Proofs.Properties.NetworkPeriod` proves both.

The count is the statement's, factor by factor, over the network certifier's `m` connections and the period's `n`
windows: `#Ss · ((∏_c (cap_c + 1)) · ∏_i (2^#(free inputs) · #faultySets · 2^β · Dj^k))`. A reader of `K` (the
communication audit's bridge to `PeriodLeakAccepted`) instantiates it at the run's caps and the windows' facts, and
supplies each of the statement's quantities as at most the check's number:

- in the clock form the windows are the compute audits: `#Ss ≤ #Σ_sync ^ (link-windows)` (each link-window under one
  sync of Σ_sync), an audit's free inputs at most those its registered circuit leaves free under its registration
  (`Audit.free`), its faulty sets at most the sets of at most `b` of its `N` units, `β` at most `b · bits` with `bits`
  the most one of its Program's units commits (`faulty_bits`), and `Dj^k` at most `jitter ^ steps`;
- in the rows form the windows are the link-windows' scans: `#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes'
  lengths), a window's free inputs at most its scan's Program's and its tied audits' (`tiedFree`: the egress renders the
  outputs of the audits its scan's `prog` is tied to, so the bits the prover chose of their inputs are the window's),
  each window's faulty sets at most the sets of at most `b` of its `N` units, `β` at most `b · max(b_ing, b_eg)`
  (`faulty_bits`) and, when a tied audit is sampled in part with a bound of at least one incorrect proof block, every
  slot entry bit (`faulty_entries`, `E_w`), and `Dj^k` at most `jitter ^ steps`.

The rows form's terms charge a tied audit (one a scan's `prog` is an output of) its free inputs, and its incorrect
proof blocks only as the slot entries they can set, since the scan reads the audit only through its `prog` and its
state is the scan's own. The check accepts a tied audit at any sample; the consistency check of one is `RowsProgWired`
and `RowsStateTied`. `RowsProgWired` says it accepts no link-window whose scan's `prog` is not a compute audit's slot
entries, a lane's tree the entries port of its chunk of an entry template of that audit's own Program at the closure
digest its circuit record registers, a record its registration binds. It reads the entry templates from what the
integration registered, over any registration: no constant of the check names a model, a template or a binding.
`RowsStateTied` says it accepts no link-window whose scan's state is not the `SCAN_STATE` of an audit its `prog` is
wired to, so a state that skips or alters one of the scan's transitions is no Program's input. Each says its
conclusion (`RowsProgWiredAt`, `RowsStateTiedAt`) of every period `check` accepts, and so is, by definition, its
property of any period check at `check` (`RowsProgWiredOf`, `RowsStateTiedOf`), which the nontrivial lemmas state
false at `acceptAllRows`, a check that accepts every period. Each guarantee writes its acceptance of `check` out rather
than being its `…Of check`: the shape rule reads the acceptance off the statement's own binders.

The free inputs `K` charges are the check's own count, from what the integration registered, never a number the caller
supplies. `AuditFreeOfLayout` says an audit `auditOf` accepts counts its free inputs from the circuit record its window
registers (`inputsOf`, at the record's SHA-512, which `Audit.free` holds its registration to), or, without one, every
input bit of its Program; `FreeOfLayout` says each term the check publishes charges those counts (`Audit.free`) under
the audits' own registrations, and in the rows form the tied audits' under each link-window's (`tiedFree`).
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period

/-- **The clock form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, `n` compute audits `a`, their registrations `r` and their facts `w`, the jitter bound `jitter`,
`#Σ_sync` `syncs` and `windows` link-windows: `#Ss` at `syncs ^ windows` (`schedules`), `∏_c (cap_c + 1)` at
`∏_j (caps j + 1)` (`cells`), and audit `i`'s `2^#(free inputs)` at `2 ^ (a i).free (r i)` (its registered circuit's
count under its registration, `Audit.free`), `#faultySets` at `atMost N b`, `2^β` at `2 ^ (b · (a i).bits)` (each of
its at most `b` wrong units commits at most its Program's widest unit's bits) and `Dj^k` at `jitter ^ (a i).steps`,
their product is at most `2 ^ K`. -/
def ClockKBoundsCount : Prop :=
  ∀ (jitter syncs windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (a : Fin n → Audit) (r : Fin n → Lean.Json)
    (w : Fin n → Facts),
    syncs ^ windows * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (a i).free (r i) * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * (a i).bits) * jitter ^ (a i).steps)) ≤
      2 ^ (terms (Array.ofFn a) (Array.ofFn r) (Array.ofFn w) (Array.ofFn caps) jitter syncs windows).K

/-- **The rows form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, the facts `w` of `n` link-windows' scans, their tied audits' free inputs `tied` (`tiedAudits`) and the
slot entry bits their tied audits' wrong units can set `faulty` (`faultyEntries`), under one scan's terms `scan`, the
jitter bound `jitter` and the charged audits' facts `charged`: `#Ss` and `∏_c (cap_c + 1)` each at `∏_j (caps j + 1)`
(`schedules` and `cells`), and window `i`'s `2^#(free inputs)` at `2 ^ (scan.free + tied i)`, `#faultySets` at
`atMost N b`, `2^β` at `2 ^ (b · max(b_ing, b_eg))` times `2 ^ faulty i` and `Dj^k` at `jitter ^ scan.steps`, their
product is at most `2 ^ K`. -/
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied faulty : Fin n → ℕ) (charged : Array Facts),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (scan.free + tied i) * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps * 2 ^ faulty i)) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows
        charged).K

/-- What `RowsProgWired` says of a period its check accepted: in the rows form, each link-window's scan's registration
is wired to the audits' slot entries (`wired`). -/
def RowsProgWiredAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → wired p.audits s c.audits reg = true

/-- `RowsProgWired` of the period check `chk`: `RowsProgWiredAt` of every period `chk` accepts. -/
def RowsProgWiredOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsProgWiredAt p c

/-- **The rows form accepts no link-window whose scan's `prog` is not a compute audit's slot entries.** If the check
accepts a period in the rows form, each link-window's scan's registration has, for each lane `j`, its `scan/prog<j>`
entry at the root and domain of one audit's entries port of chunk `j`, `<descriptor id>/c<j>/o0` of `T / every` leaves,
of a template node of that audit's own Program that its circuit record registers as an entry template, at the closure
digest the record registers for it (so a template under that id with another body is refused), a record the audit's
registration binds before any coin (`Audit.boundIn`), with one member a slot of the window (`wired`), so chunk `j` is
lane `j`'s slots and each slot's entry, read bit included, is one its Program computed. The statement holds for any
registration: whatever entry templates, vocabulary size and stop ids the integration registers. -/
def RowsProgWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsProgWiredAt p c

/-- What `RowsStateTied` says of a period its check accepted: in the rows form, each link-window's scan's state
entries are at one root and domain, and every audit wired to it has its own there (`stateTied`). -/
def RowsStateTiedAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → stateTied p.audits s c.audits reg = true

/-- `RowsStateTied` of the period check `chk`: `RowsStateTiedAt` of every period `chk` accepts. -/
def RowsStateTiedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsStateTiedAt p c

/-- **The rows form accepts no link-window whose scan's state is not its wired audit's `SCAN_STATE`.** If the check
accepts a period in the rows form, each link-window's scan's registration has a state entry (`stateEntries`: a `roots`
port `scan/state` or a `registered` value `verity/certifier/scan-state/v1`), every one at one root and domain, and an
audit whose slot entries are the scan's `prog` (`wired`) has one, every one at that root and domain (`stateTied`). The
scan's units read their state at that root and register the state their transition computes there, so the state a
slot's Program reads is the scan's, and a `SCAN_STATE` that skips or alters a transition, at another root or domain, is
refused. -/
def RowsStateTied : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsStateTiedAt p c

/-- **A compute audit's free inputs are counted from its registered circuit** (`#(freeInputs \ reg)`). If `auditOf`
accepts the audit `a`, its `program` decodes to a Program `P` and either its free inputs are every input bit of `P`'s
root templates (`inputBits`) under any registration, or its window's circuit record `r` is the one whose SHA-512
(`circuitDigest`) its registration must bind (`Audit.circuit`, which `Audit.free` reads through `Audit.boundIn`), and its
inputs are those `inputsOf` counts from `r` under the verifier's pins (`pinsOf`): the input bits, those the record's wires
and ingress fill and those its registered reads read by parameter. No other field of `a` is read for them, so a `free`
count the caller supplies, right or wrong, changes no audit's inputs. -/
def AuditFreeOfLayout : Prop :=
  ∀ (a : Lean.Json) (ctx : String) (A : Audit), auditOf a ctx = .ok A →
    ∃ (pj : Lean.Json) (c : String) (P : Flock.Program.Prog),
      a.getObjVal? "program" = .ok pj ∧ Flock.canon pj = .ok c ∧ Flock.Program.decode c.toUTF8 = .ok P ∧
      ((∀ reg, A.free reg = inputBits (rootTemplates P)) ∨
        ∃ (r : Lean.Json) (pins : Array (String × String)), a.getObjVal? "circuit" = .ok r ∧
          circuitDigest r = .ok A.circuit ∧ pinsOf a ctx = .ok pins ∧ inputsOf r (rootTemplates P) pins = .ok A.inputs)

/-- What `FreeOfLayout` says of a period its check accepted, publishing `pub`: each term's free inputs are counted by
`Audit.free` under a registration of the commitment's, in the clock form one compute audit's under its own, and in the
rows form the scan's own and its tied audits' (`tiedFree`, each under its own) of one link-window's. -/
def FreeOfLayoutAt (p : Params) (c : Commitment) (pub : Public) : Prop :=
  ∀ t ∈ pub.units, match p.form with
    | .clock _ => ∃ a reg, (a, reg) ∈ p.audits.zip c.audits ∧ t.free = a.free reg
    | .rows _ _ scan => ∃ reg ∈ c.windows, t.free = scan.free + tiedFree p.audits c.audits reg

/-- **`K` charges the free inputs the registered circuits leave, under the registrations.** If the check accepts a
period, each term it publishes charges as free inputs `Audit.free`'s count (`AuditFreeOfLayout`) under a registration of
the commitment: a compute audit's under its own in the clock form, and in the rows form the scan's own and its tied
audits' (`tiedFree`) under one link-window's. -/
def FreeOfLayout : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → FreeOfLayoutAt p c pub

/-- A period check that accepts every period, publishing the rows form's terms of no link-window: what
`RowsProgWiredOf` and `RowsStateTiedOf` must fail at for `RowsProgWired` and `RowsStateTied` (each its `…Of` at
`check`) to say something. -/
def acceptAllRows (p : Params) (_ : Commitment) (_ : Evidence) : Core.Outcome Public Profile :=
  .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] p.cells p.jitter p.windows) { link := p.link, windows := p.windows }

end Properties.NetworkPeriod
