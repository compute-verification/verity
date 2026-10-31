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
  sync of Σ_sync), an audit's free inputs at most its layout's (`free`), its faulty sets at most the sets of at most `b`
  of its `N` units, `β` at most `b · bits` with `bits` the most one of its Program's units commits (`faulty_bits`), and
  `Dj^k` at most `jitter ^ steps`;
- in the rows form the windows are the link-windows' scans: `#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes'
  lengths), a window's free inputs at most its scan's Program's and its tied audits' (`free`: the egress renders the
  outputs of the audits its scan's `prog` is tied to, so the bits the prover chose of their inputs are the window's),
  each window's faulty sets at most the sets of at most `b` of its `N` units, `β` at most `b · max(b_ing, b_eg)`
  (`faulty_bits`), and `Dj^k` at most `jitter ^ steps`.

The rows form's terms charge a tied audit (one a scan's `prog` is an output of) its free inputs and no faulty term,
though a wrong unit's outputs would reach the egress; `RowsTiedDrawn` says the check accepts no tied audit with a unit
undrawn. `RowsProgWired` says it accepts no link-window whose scan's `prog` is not a compute audit's slot entries,
the entries port of a `SlotEntry_v1` template of that audit's own Program. `RowsStateTied` says it accepts no
link-window whose scan's state is not the `SCAN_STATE` of an audit its `prog` is wired to, so a state that skips or
alters one of the scan's transitions is no Program's input. Beside them, their nontrivial lemmas' objects:
`RowsTiedDrawnOf`, `RowsProgWiredOf` and `RowsStateTiedOf`, the properties of any period check, and `acceptAllRows`, a
check that accepts every period.
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period

/-- **The clock form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, `n` compute audits `a` and their facts `w`, the jitter bound `jitter`, `#Σ_sync` `syncs` and `windows`
link-windows: `#Ss` at `syncs ^ windows` (`schedules`), `∏_c (cap_c + 1)` at `∏_j (caps j + 1)` (`cells`), and audit
`i`'s `2^#(free inputs)` at `2 ^ (a i).free`, `#faultySets` at `atMost N b`, `2^β` at `2 ^ (b · (a i).bits)` (each of
its at most `b` wrong units commits at most its Program's widest unit's bits) and `Dj^k` at `jitter ^ (a i).steps`,
their product is at most `2 ^ K`. -/
def ClockKBoundsCount : Prop :=
  ∀ (jitter syncs windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (a : Fin n → Audit) (w : Fin n → Facts),
    syncs ^ windows * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (a i).free * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * (a i).bits) * jitter ^ (a i).steps)) ≤
      2 ^ (terms (Array.ofFn a) (Array.ofFn w) (Array.ofFn caps) jitter syncs windows).K

/-- **The rows form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, the facts `w` of `n` link-windows' scans and their tied audits' free inputs `tied` (`tiedAudits`), under
one scan's terms `scan` and the jitter bound `jitter`: `#Ss` and `∏_c (cap_c + 1)` each at `∏_j (caps j + 1)`
(`schedules` and `cells`), and window `i`'s `2^#(free inputs)` at `2 ^ (scan.free + tied i)`, `#faultySets` at
`atMost N b`, `2^β` at `2 ^ (b · max(b_ing, b_eg))` and `Dj^k` at `jitter ^ scan.steps`, their product is at most
`2 ^ K`. -/
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied : Fin n → ℕ),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (scan.free + tied i) * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps)) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i)) (Array.ofFn caps) jitter windows).K

/-- Audit `i`'s answer drew every unit: one-stage accepted its registration with a profile of one level, drawn
`subset:k` or stratified, whose `Σ k` is the level's size (`factsOf`'s `drawn` and `units`). -/
def DrawnWhole (i : ℕ) (reg : Registration) (x : Core.Answer) : Prop :=
  ∃ (prof : Core.Integrity) (pub : Lean.Json) (f : Facts),
    answered s!"audit {i}" reg x = .ok (prof, pub) ∧ factsOf s!"audit {i}" prof pub = .ok f ∧ f.drawn = f.units

/-- **The rows form accepts no tied audit with a unit undrawn.** If the check accepts a period in the rows form, every
audit some link-window's scan's `prog` is an output of (`tiedIn` over the commitment's registrations) drew every unit
(`DrawnWhole`): `rowsTerms` charges such an audit's free inputs to its link-window and no faulty term for it. -/
def RowsTiedDrawn : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
      ∀ (i : ℕ) (reg : Registration) (x : Core.Answer), c.audits[i]? = some reg → e.audits[i]? = some x →
        tiedIn c.audits c.windows i = true → DrawnWhole i reg x

/-- `RowsTiedDrawn` of any period check. -/
def RowsTiedDrawnOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
      ∀ (i : ℕ) (reg : Registration) (x : Core.Answer), c.audits[i]? = some reg → e.audits[i]? = some x →
        tiedIn c.audits c.windows i = true → DrawnWhole i reg x

/-- **The rows form accepts no link-window whose scan's `prog` is not a compute audit's slot entries.** If the check
accepts a period in the rows form, each link-window's scan's registration has its `scan/prog` entry at the root and
domain of an audit's entries port, `<descriptor id>/c0/o0` of a template node of that audit's own Program calling
`SlotEntry_v1`, a Definition the check pins by its closure digest (`slotEntryPins`, so a template under that name with
another body is refused), with one member a slot of the window (`wired`), so each slot's entry, read bit included, is
one its Program computed. -/
def RowsProgWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
      ∀ (i : ℕ) (reg : Registration), c.windows[i]? = some reg → wired p.audits s c.audits reg = true

/-- `RowsProgWired` of any period check. -/
def RowsProgWiredOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
      ∀ (i : ℕ) (reg : Registration), c.windows[i]? = some reg → wired p.audits s c.audits reg = true

/-- **The rows form accepts no link-window whose scan's state is not its wired audit's `SCAN_STATE`.** If the check
accepts a period in the rows form, each link-window's scan's registration has a state entry (`stateEntries`: a `roots`
port `scan/state` or a `registered` value `verity/certifier/scan-state/v1`), every one at one root and domain, and an
audit whose slot entries are the scan's `prog` (`wired`) has one, every one at that root and domain (`stateTied`). The
scan's units read their state at that root and register the state their transition computes there, so the state a
slot's Program reads is the scan's, and a `SCAN_STATE` that skips or alters a transition, at another root or domain, is
refused. -/
def RowsStateTied : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
      ∀ (i : ℕ) (reg : Registration), c.windows[i]? = some reg → stateTied p.audits s c.audits reg = true

/-- `RowsStateTied` of any period check. -/
def RowsStateTiedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
      ∀ (i : ℕ) (reg : Registration), c.windows[i]? = some reg → stateTied p.audits s c.audits reg = true

/-- A period check that accepts every period, publishing the rows form's terms of no link-window: what
`RowsTiedDrawnOf`, `RowsProgWiredOf` and `RowsStateTiedOf` must fail at for `RowsTiedDrawn`, `RowsProgWired` and
`RowsStateTied` to say something. -/
def acceptAllRows (p : Params) (_ : Commitment) (_ : Evidence) : Core.Outcome Public Profile :=
  .accept (rowsTerms #[] ⟨0, 0, 0, 0⟩ #[] p.cells p.jitter p.windows) { link := p.link, windows := p.windows }

end Properties.NetworkPeriod
