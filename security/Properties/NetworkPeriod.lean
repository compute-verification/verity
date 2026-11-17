module

public import NetworkVerifier.Period
public import Spot.OneStage
public import Lms.Sign
public import Mathlib.Algebra.BigOperators.Fin

/-!
# The network period check's bound: what its `K` counts (`NetworkCheck.Period`)

On acceptance, `NetworkCheck.Period.check` publishes `K`, the sum of a ceiling of `log2` of each factor of
`NetTiming.Guarantees.MonitoringPeriodLeakAccepted`'s count at the number the check reads for it, `rowsTerms`'s. A sum
of ceilings is at least the ceiling of the sum, so `K` is at least `⌈log2⌉` of the count, not equal to it;
`RowsKBoundsCount` says the count at the check's numbers is at most `2 ^ K`. `Proofs.Properties.NetworkPeriod` proves
it.

The count is the statement's, factor by factor, over the network attester's `m` connections and the period's `n`
windows: `#Ss · ((∏_c (cap_c + 1)) · ∏_i (2^#(free inputs) · #incorrectSets · 2^β · Dj^k))`. A reader of `K` (the
communication audit's bridge to `MonitoringPeriodLeakAccepted`) instantiates it at the run's caps and the windows' facts, and
supplies each of the statement's quantities as at most the check's number. The windows are the execution phases' scans:
`#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes' lengths), a window's free inputs at most its scan's Program's and its
tied audits' (`free`: the egress renders the outputs of the audits its scan's `prog` is tied to, so the bits the prover
chose of their inputs are the window's), each window's incorrect sets at most the sets of at most `b` of its `N` units,
`β` at most `b · max(b_ing, b_eg)` (`faulty_bits`) and, when a tied audit is sampled in part with a bound of at least
one incorrect proof block, every slot entry bit (`faulty_entries`, `E_w`), and `Dj^k` at most `jitter ^ steps`.

The rows form's terms charge a tied audit (one a scan's `prog` is an output of) its free inputs, and its incorrect
proof blocks only as the slot entries they can set, since the scan reads the audit only through its `prog` and its
state is the scan's own. The check accepts a tied audit at any sample; the consistency check of one is `RowsProgWired`,
`RowsStateTied` and `RowsTiedInputWired`, and what it publishes of one sampled in part is `RowsTiedCharged`.
`RowsProgWired` says it accepts
no execution phase whose scan's `prog` is not a compute audit's slot entries, a lane's tree the entries port of its chunk
of a `SlotEntry_v1` or `SelectEntry_v1` template of that audit's own Program. `RowsStateTied` says it accepts no
execution phase whose scan's state is not the `SCAN_STATE` of an audit its `prog` is wired to, so a state that skips or
alters one of the scan's transitions is no Program's input. `RowsTiedInputWired` says it accepts no execution phase
whose scan's answer names a tied Input other than the scan's lanes' `prog<j>`, which are wired as `RowsProgWired` says:
a tied Input is a run Input several proof blocks read, which one-stage's registration check lets through only named.
`RowsStateEmpty` says it accepts none whose scan's
registration does not open row 0 of that state to the empty state, the one row no unit of the scan computes.
`RowsCircuitHeld` says it accepts none whose tied audit's registration names a Program other than the class circuit,
the parameters' `circuit`. `RowsStartCharged` says the public it accepts a period with charges the period's first start,
where that is not the run's first, the scan's state's full width as a term of `K` of its own, once a period.
`RowsTiedCharged` says the public it accepts with supplies
the charge itself: `charged` is the facts of each tied audit sampled in part (`drawn < units`), each once, so each
one's δ is in `eps` (`Public.eps` sums the terms' facts' and `charged`'s, and nothing else), and where such an audit's
bound is at least one incorrect proof block, the term of every execution phase whose scan's `prog` is its output has
every slot entry bit (`faultyEntries` at `scan.entries`), the number `RowsKBoundsCount` takes as `faulty`. No tied audit
names more sampled than its level holds (`factsOf` refuses a profile that does), so one that is not charged names as
many: `drawn = units`. Each says its conclusion (`RowsProgWiredAt`, `RowsStateTiedAt`, `RowsTiedInputWiredAt`,
`RowsTiedChargedAt`) of every period `check` accepts, and so is, by definition, its property of any period check at
`check` (`RowsProgWiredOf`, `RowsStateTiedOf`, `RowsTiedInputWiredOf`, `RowsTiedChargedOf`), which the nontrivial
lemmas state false at
`acceptAllRows`, a check that accepts every period. Each guarantee writes its acceptance of `check` out rather than
being its `…Of check`: the shape rule reads the acceptance off the statement's own binders.
`RowsCapPublished` says an accepted period's public names its schedule's response cap, and `RowsCapPublishedOf` is
its property of any period check, false at `acceptAllRows` too; so of `RowsStartCharged` and `RowsStartChargedOf`.

The counts those terms read are the check's own, and so is the sample each is of. `RegisteredSampleHeld` says the
facts an accepted period's public holds of each compute audit, and of each scan, are of the sample that answer's
registration states, one its units hold (`SampleHeld`): no more sampled than the units, no stratum sampled above its
size, strata whose sizes sum to the units. `SampleBoundCharged` says their count of incorrect proof blocks is at least
that sample's bound at their δ (`SampleCharged`): one-stage's bound of a `subset:k`, and of a stratified sample the
sum of its strata's own, each stratum one `subset:k_s` of its own units at δ split equally over the strata. Both are
`AnswersAt` of every period `check` accepts, and `AnswersOf` is that of any period check, false at `acceptAllRows`
whatever it says of an answer.

`RowsOkHeld` says the rows form accepts no execution phase whose scan's registration names its proof blocks' ok bits
anywhere but at the constants of the schedule's `T` (`NetworkCheck.ScanOk`): the schedule has at least two ticks, each
member of the scan (`ScanOk.members`) has a `roots` entry at its constant (`OkEntry`: the port `<member>/o0`, the
member's proof blocks' count, and the frame domain and the root of that many ok rows under the public salts), and
every `roots` entry whose port ends `/o0` is some member's constant. So the root a member's `o0` is registered at is
one the check computed from `T`, never one the registration states. It is stated in the constants' own terms
(`ScanOk.domain`, `ScanOk.root`), not as `ScanOk.held` being true; `RowsOkHeldOf` is its property of any period
check, false at `acceptAllRows` too, and `pairParams 1` with `unnamedCommitment` is a period it is false of.
-/

@[expose] public section

namespace Properties.NetworkPeriod

open NetworkVerifier.Period

/-- **The rows form's `K` bounds `MonitoringPeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, the facts `w` of `n` execution phases' scans, their tied audits' free inputs `tied` (`tiedAudits`) and the
slot entry bits their tied audits' incorrect proof blocks can set `faulty` (`faultyEntries`), under one scan's terms `scan`, the
jitter bound `jitter`, the charged audits' facts `charged` and the period's charged start `start`: the start's free
inputs at `2 ^ start`, `#Ss` and `∏_c (cap_c + 1)` each at `∏_j (caps j + 1)` (`schedules` and `cells`), and window
`i`'s `2^#(free inputs)` at `2 ^ (scan.free + tied i)`, `#incorrectSets` at `atMost N b`, `2^β` at
`2 ^ (b · max(b_ing, b_eg))` times `2 ^ faulty i` and `Dj^k` at `jitter ^ scan.steps`, their product is at most
`2 ^ K`. -/
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied faulty : Fin n → ℕ) (charged : Array Facts) (start : ℕ),
    2 ^ start * ((∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (scan.free + tied i) *
        atMost (w i).units (w i).wrong * 2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps *
        2 ^ faulty i))) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows
        charged start).K

/-- What `RowsProgWired` says of a period its check accepted: in the rows form, each execution phase's scan's
registration is wired to the audits' slot entries (`wired`). -/
def RowsProgWiredAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → wired p.audits s c.audits reg = true

/-- `RowsProgWired` of the period check `chk`: `RowsProgWiredAt` of every period `chk` accepts. -/
def RowsProgWiredOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsProgWiredAt p c

/-- **The rows form accepts no execution phase whose scan's `prog` is not a compute audit's slot entries.** If the check
accepts a period in the rows form, each execution phase's scan's registration has, for each lane `j`, its `scan/prog<j>`
entry at the root and domain of one audit's entries port of chunk `j`, `<descriptor id>/c<j>/o0` of `T / every` leaves,
of a template node of that audit's own Program calling `SlotEntry_v1` or `SelectEntry_v1` (whose select and collector
end in `SlotEntry_v1`), a Definition the check pins by its closure digest (`entryPins`, so a template under either name
with another body is refused), with one member a slot of the window (`wired`), so chunk `j` is lane `j`'s slots and each
slot's entry, read bit included, is one its Program computed. -/
def RowsProgWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsProgWiredAt p c

/-- What `RowsStateTied` says of a period its check accepted: in the rows form, each execution phase's scan's state
entries are at one root and domain, its registration states its units' reads of that state as the chain
(`stateRead`), and an audit wired to it has its own there (`stateTied`). -/
def RowsStateTiedAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → stateTied p.audits s c.audits reg = true

/-- `RowsStateTied` of the period check `chk`: `RowsStateTiedAt` of every period `chk` accepts. -/
def RowsStateTiedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsStateTiedAt p c

/-- **The rows form accepts no execution phase whose scan's state is not its wired audit's `SCAN_STATE`.** If the check
accepts a period in the rows form, each execution phase's scan's registration has a state entry (`stateEntries`: a
`roots` port `scan/state` or a `registered` value `verity/certifier/scan-state/v1`), every one at one root and domain,
and an audit whose slot entries are the scan's `prog` (`wired`) has one, every one at that root and domain
(`stateTied`). The scan's units read their state at that root and register the state their transition computes there, as
the scan's registration states their reads (`stateRead`: unit `t` reads row `t` and registers row `t + 1`), so the state
a slot's Program reads is the scan's, and a `SCAN_STATE` that skips or alters a transition, at another root or domain,
is refused. -/
def RowsStateTied : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsStateTiedAt p c

/-- What `RowsTiedInputWired` says of a period its check accepted: in the rows form, each execution phase's scan's
answer names no tied Input but the scan's lanes' `prog<j>` (`tiedInputWired`), and its registration is wired to the
audits' slot entries (`wired`). -/
def RowsTiedInputWiredAt (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json) (x : ExecutionPhase), c.windows[i]? = some reg → e.windows[i]? = some x →
      tiedInputWired s x.answer = true ∧ wired p.audits s c.audits reg = true

/-- `RowsTiedInputWired` of the period check `chk`: `RowsTiedInputWiredAt` of every period `chk` accepts. -/
def RowsTiedInputWiredOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsTiedInputWiredAt p c e

/-- **The rows form accepts no execution phase whose scan's answer names a tied Input the check does not wire.**
One-stage's registration check refuses a run Input that several proof blocks read, unless its verifier names the Input
tied: a later check then ties its rows to the registrations that computed them, and its registration verdict names
each Input it let through (`rests_on_period_check`), a list the scan's answer carries in its public outputs. If the
check accepts a period in the rows form, every name each execution phase's scan's answer carries there is one of the
scan's lanes' `prog<j>` (`tiedInputWired`, `progParams`: in any order, a lane's more than once or not at all, and none
where the answer has no such list), and that execution phase's `scan/prog<j>` entries are wired, each lane's to a
compute audit's slot entries (`wired`, as `RowsProgWired` says): every tied Input an accepted scan's answer names is
one this check ties. It is a statement about the names the answer carries. That the answer carries every Input the
registration check let through, and that lane `j`'s name `prog<j>` is the Input the entry `scan/prog<j>` registers,
which is one-stage's registration check's, are not said here. -/
def RowsTiedInputWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsTiedInputWiredAt p c e

/-- What `RowsCircuitHeld` says of a period its check accepted: in the rows form, each execution phase's tied compute
audits are registered under the parameters' class circuit (`held`). -/
def RowsCircuitHeldAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → held p.circuit c.audits reg = true

/-- `RowsCircuitHeld` of the period check `chk`: `RowsCircuitHeldAt` of every period `chk` accepts. -/
def RowsCircuitHeldOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsCircuitHeldAt p c

/-- **The rows form accepts no execution phase whose tied compute audit's window circuit is not the class circuit.**
If the check accepts a period in the rows form, every audit an execution phase's scan's `prog` is an output of
(`tiedAudits`) has a registration whose `program` is the parameters' `circuit`, the SHA-512 of the Program of the one
window circuit the workload's class registers (`held`). A window circuit's own digest binds its window's run; its
Program is one for the period, so no execution phase's tied audit is registered under a Program the prover chose for
that phase. That its term is read off that Program's descriptor is the audit check's (`auditCheck`), not stated
here. -/
def RowsCircuitHeld : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsCircuitHeldAt p c

/-- What `RowsStateEmpty` says of a period its check accepted: where its first execution phase is the run's first
(`p.first = 0`), in the rows form, the scan's registration of each execution phase's transfer blocks registers its
start, the start unit's `sin`, as the empty state (`startEmpty`). -/
def RowsStateEmptyAt (p : Params) (c : Commitment) : Prop :=
  p.first = 0 → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → startEmpty reg = true

/-- `RowsStateEmpty` of the period check `chk`: `RowsStateEmptyAt` of every period `chk` accepts. -/
def RowsStateEmptyOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsStateEmptyAt p c

/-- **The rows form accepts no execution phase's transfer blocks of the run's first period whose scan's state's row 0 is
not the empty state.** If the check accepts a period in the rows form whose first execution phase is the run's first,
the scan's registration of each execution phase's transfer blocks has a start entry, the start unit's `sin`, and opens every one, at its root and domain, to a row of zero u16 words under the
salt it states (`startEmpty`). The start unit copies its `sin` to row 0 of the scan's state. With `RowsStateTied`,
whose `stateTied` holds the scan's units' reads of that state to the chain (`stateRead`: the start unit registers row
0, unit `t` reads row `t` and registers row `t + 1`), every row a slot's Program reads is the empty state or one the
scan's transition computed from the row before it: row 0, which the Programs anchored at bucket 0 read, is not the
prover's word. A period whose first execution phase is not the run's first starts from a state the check holds to
nothing, and is charged it instead (`RowsStartCharged`). -/
def RowsStateEmpty : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsStateEmptyAt p c

/-- Audit `i`'s answer gives the facts `f`: one-stage accepted its window record `reg` with a profile of one level,
sampled `subset:k` or stratified, the sample `reg` states, and `f` is what the check reads of that registration, that
profile and its public (`answered`, `factsOf`). -/
def AuditFacts (i : ℕ) (reg : Lean.Json) (x : Core.Answer) (f : Facts) : Prop :=
  ∃ (prof : Core.Integrity) (pub : Lean.Json),
    answered s!"audit {i}" reg x = .ok (prof, pub) ∧ factsOf s!"audit {i}" reg prof pub = .ok f

/-- The facts `f` are of the sample the registration `reg` states, a sample its units hold (`sampleOf`: its
`population` of `n` units under its sampling design, a `subset:k` or a stratified one, which the registration fixes
before any coin): `f.units` is `n`, `f.design` is its design, which a reading at another δ reads (`Facts.readAt`),
and the facts' count of sampled proof blocks is at most `n`. Of a `subset:k` sample that count is `k`. Of a
stratified sample of strata `ss` it is the strata's `k_s` summed, no stratum samples more than it holds, and the
strata's sizes sum to `n`. -/
def SampleHeld (reg : Lean.Json) (f : Facts) : Prop :=
  ∃ (n : ℕ) (d : Core.SamplingDesign), sampleOf reg = some (n, d) ∧ f.units = n ∧ f.design = d ∧ f.drawn ≤ n ∧
    ((∃ k : ℕ, d = .subset k ∧ f.drawn = k) ∨
      ∃ ss : Array Core.Stratum, d = .stratified ss ∧ f.drawn = ss.foldl (fun m st => m + st.k) 0 ∧
        (∀ st ∈ ss, st.k ≤ st.size) ∧ ss.foldl (fun m st => m + st.size) 0 = n)

/-- The facts `f` charge the sample the registration `reg` states (`sampleOf`): `f.wrong`, the count of incorrect
proof blocks the bound reads, is at least the sample's own bound at `f.delta`, computed with one-stage's
`Spot.Bound.boundOf` at a `subset` design and no other.

- A `subset:k` sample of `n` units: the count is at least `boundOf n (.subset k) δ`.
- A stratified sample of strata `ss`: the count is at least the sum over the strata of
  `boundOf n_s (.subset k_s) (δ / S)`, `S` the number of strata. Each stratum is one `subset:k_s` of its own `n_s`
  units, δ is split equally over the strata, and the strata, each one's units and its `k_s` are the registration's.
  The stratified rule is read nowhere as a sampling design of the whole population. -/
def SampleCharged (reg : Lean.Json) (f : Facts) : Prop :=
  ∃ (n : ℕ) (d : Core.SamplingDesign), sampleOf reg = some (n, d) ∧
    ((∃ k : ℕ, d = .subset k ∧ Spot.Bound.boundOf n (.subset k) f.delta ≤ f.wrong) ∨
      ∃ ss : Array Core.Stratum, d = .stratified ss ∧
        ss.foldl (fun b st =>
          b + Spot.Bound.boundOf st.size (.subset st.k) ⟨f.delta.num, f.delta.den * ss.size⟩) 0 ≤ f.wrong)

/-- `P reg f` of every answer of a period accepted with the public `pub`, `reg` the answer's registration and `f` the
facts the public holds of it: each compute audit `i`'s facts `pub.audits[i]`, and the facts of each scan's term
`pub.units[w]`. -/
def AnswersAt (P : Lean.Json → Facts → Prop) (p : Params) (c : Commitment) (pub : Public) : Prop :=
  (∀ (i : ℕ) (reg : Lean.Json), c.audits[i]? = some reg →
      ∃ f : Facts, pub.audits[i]? = some f ∧ P reg f) ∧
    ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
      ∀ (w : ℕ) (reg : Lean.Json), c.windows[w]? = some reg →
        ∃ t : AuditTerms, pub.units[w]? = some t ∧ P reg t.facts

/-- `AnswersAt P` of every period the period check `chk` accepts, at the public it accepts with: what
`RegisteredSampleHeld` (at `SampleHeld`) and `SampleBoundCharged` (at `SampleCharged`) are of `check`. -/
def AnswersOf (P : Lean.Json → Facts → Prop)
    (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → AnswersAt P p c pub

/-- **Every answer of an accepted period is of its registration's sample, one its units hold.** If the check accepts
a period with the public `pub`, then the facts `pub` holds of each compute audit, and of each scan, are of the
sample that answer's registration states (`SampleHeld`): its units and its count of sampled proof blocks are the
registered ones, the sample names no more sampled than its units, and of a stratified sample no
stratum samples more than it holds and the strata's sizes sum to the units.

The check refuses an answer whose profile names more sampled than its level holds, one with a stratum that samples
more than the stratum holds, and one whose strata's sizes do not sum to the level's; and it refuses a profile that
is another sample than the one its registration states, so the three hold of the registered sample, whose strata,
each one's size and its `k_s` are fixed before any coin. An answer the period check did not verify thus gives it no
count of sampled proof blocks past its units, and no stratum the bound would read as sampled whole where it is not.
That the strata are the level's units is the answer's own verifier's, which the period check does not read. -/
def RegisteredSampleHeld : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → AnswersAt SampleHeld p c pub

/-- **The count of incorrect proof blocks an accepted period charges is at least its registered sample's bound.** If
the check accepts a period with the public `pub`, then the count of incorrect proof blocks in the facts `pub` holds
of each compute audit, and of each scan, is at least the bound of the sample that answer's registration states, at
the facts' δ (`SampleCharged`). The check computes the bound itself and reads the larger of it and the `bound` the
answer states, so an answer that states a smaller `bound` than its sample gives lowers no term of `K`:
`RowsKBoundsCount` is read at these counts. The rows form's charge of a tied audit reads the same sample at the period's own δ, and `RowsTiedCharged` says the same of that reading.
The facts are of that sample by `RegisteredSampleHeld`.

The bound is a stratum at a time. A stratum of a registered sample is one uniform `subset:k_s` of its own `n_s`
units, and its incorrect proof blocks are bounded at δ split equally over the strata, so the strata's bounds hold
together except with probability δ; no step reads the strata as one sampling design of the level's units. That the
units a sample takes are the registered rule's under the coins is the answer's own verifier's, which the period check
does not read. -/
def SampleBoundCharged : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → AnswersAt SampleCharged p c pub

/-- Among the audits' facts `facts`, in their order and each once, those of each audit `tied` ties that is sampled in
part (`drawn < units`), each read at the period's own δ (`Facts.readAt` at `tiedDelta`). -/
def tiedInPart (tied : ℕ → Bool) (facts : Array Facts) : Array Facts :=
  (Array.range facts.size).filterMap fun i =>
    (facts[i]?.filter fun f => tied i && decide (f.drawn < f.units)).map (·.readAt tiedDelta)

/-- What `RowsTiedCharged` says of a period its check accepted with the public `pub`: in the rows form, `pub.charged`
is `tiedInPart` of the public's audits' facts under the commitment's ties (`tiedIn`), and each tied audit `i` has an
answer `x` whose facts `f` (`AuditFacts`) are the public's `audits[i]` and name no more sampled than its level holds
(`f.drawn ≤ f.units`); if it names as many (`f.drawn = f.units`) and its registration states a stratified sample
(`sampleOf`), every stratum of that sample is sampled whole; if it is sampled in part (`f.drawn < f.units`), its
reading at the period's own δ (`f.readAt tiedDelta`) is among `pub.charged`, its count is at least its registered
sample's bound at that δ (`SampleCharged`) and, when that count is at least one incorrect proof block, every
execution phase `w` whose scan's `prog` is its output (`tiedAudits`) has `faultyEntries = scan.entries` in the public's
term `pub.units[w]`. -/
def RowsTiedChargedAt (p : Params) (c : Commitment) (e : Evidence) (pub : Public) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    pub.charged = tiedInPart (tiedIn c.audits c.windows) pub.audits ∧
    ∀ (i : ℕ) (reg : Lean.Json), c.audits[i]? = some reg → tiedIn c.audits c.windows i = true →
      ∃ (x : Core.Answer) (f : Facts), e.audits[i]? = some x ∧ AuditFacts i reg x f ∧ pub.audits[i]? = some f ∧
        f.drawn ≤ f.units ∧
        (f.drawn = f.units → ∀ (n : ℕ) (ss : Array Core.Stratum), sampleOf reg = some (n, .stratified ss) →
          ∀ st ∈ ss, st.k = st.size) ∧
        (f.drawn < f.units → f.readAt tiedDelta ∈ pub.charged ∧ SampleCharged reg (f.readAt tiedDelta) ∧
          (1 ≤ (f.readAt tiedDelta).wrong →
            ∀ (w : ℕ) (wreg : Lean.Json), c.windows[w]? = some wreg →
              (tiedAudits c.audits wreg).contains i = true →
                ∃ t : AuditTerms, pub.units[w]? = some t ∧ t.faultyEntries = scan.entries))

/-- `RowsTiedCharged` of the period check `chk`: `RowsTiedChargedAt` of every period `chk` accepts, at the public it
accepts with. -/
def RowsTiedChargedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsTiedChargedAt p c e pub

/-- **The rows form charges every tied audit it accepts sampled in part.** If the check accepts a period in the rows
form with the public `pub`, then `pub.charged` is the facts of each audit that some execution phase's scan's `prog` is
an output of (`tiedIn`) and that is sampled in part (`drawn < units`), in the audits' order and each once, each read
at the period's own δ, `tiedDelta`, a constant of the check and so fixed before any coin (`tiedInPart` of
`pub.audits`). That reading is of the same sample, with no second one: its count is the larger of the count the facts
hold and the registered sample's bound at `tiedDelta`, so it is at least that bound (`SampleCharged`). `Public.eps`
sums the δ of the terms' facts and of `charged`, and of nothing else, so `tiedDelta` is in `eps` once for each such
audit. And for each such audit whose count at `tiedDelta` is at least one incorrect proof block, the public's term of
every execution phase whose scan's `prog` is its output has `faultyEntries` at the scan's every slot entry bit
(`Scan.entries`), which `K` sums. So the `faulty` `RowsKBoundsCount` takes and the δ of the readings it is read from
are the check's own, of every period it accepts.

Every tied audit of an accepted period names no more sampled than its level holds: the check reads both counts off the
answer's profile, the level's size and its `subset`'s `k` or its strata's `k` summed, and refuses a profile that names
more sampled than the level has (`factsOf`). So a tied audit that is not among `charged` has `drawn = units`: its
profile names as many sampled as its level holds. Of a `subset` sample that is the level sampled whole. Of a
stratified one it is each stratum sampled whole, because the check also refuses a profile with a stratum that samples
more than it holds, and one whose strata's sizes do not sum to the level's; the strata are the ones its registration
states (`sampleOf`), since the check refuses a profile that is another sample than the registered one. That the
strata are the level's units is the audit's own verifier's, which the period check does not read. -/
def RowsTiedCharged : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsTiedChargedAt p c e pub

/-- A period check that accepts every period, publishing the rows form's terms of no execution phase, no audit's facts
and no charge: what `RowsProgWiredOf`, `RowsStateTiedOf`, `RowsTiedInputWiredOf`, `RowsTiedChargedOf`,
`RowsCapPublishedOf`, `RowsStartChargedOf`, `RowsOkHeldOf` and `AnswersOf` must fail at for `RowsProgWired`,
`RowsStateTied`, `RowsTiedInputWired`, `RowsTiedCharged`, `RowsCapPublished`, `RowsStartCharged`, `RowsOkHeld`,
`RegisteredSampleHeld` and `SampleBoundCharged` (each its `…Of` at `check`) to say something. -/
def acceptAllRows (p : Params) (_ : Commitment) (_ : Evidence) : Core.Outcome Public Profile :=
  .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0, 0⟩ #[] p.cells p.jitter p.windows) { link := p.link, windows := p.windows }

/-- The public `pub` publishes the response cap of the period's schedule (`Form.maxTokensCap`): its
`max_tokens_cap`, none when its schedule has none. -/
def CapPublished (p : Params) (pub : Public) : Prop := pub.maxTokensCap = p.form.maxTokensCap

/-- **The period check publishes its schedule's response cap.** If the check accepts a period, its public's
`max_tokens_cap` is the cap of the schedule the parameters name (`CapPublished`), the schedule whose digest every
execution phase's registrations bind (`rowsCheck`): a reader of the public, the communication audit's charge, reads the
cap the network attester held each released response to (`NetTiming.Guarantees.ResponseCapped`), and none where the
schedule has none. -/
def RowsCapPublished : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → CapPublished p pub

/-- `RowsCapPublished` of any period check. -/
def RowsCapPublishedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → CapPublished p pub

/-- A rows-form schedule of one lane and one slot stride of two ticks (T = 2, `every` 2), with a response cap of 4. -/
def cappedSchedule : Schedule := ⟨0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, some 4, none, none, false⟩

/-- The rows-form period of no execution phase and no audit under `cappedSchedule`, at the deployed LMS parameters: the
period `RowsCapPublished`'s witnesses are stated at. -/
def cappedPeriod : Params :=
  ⟨#[], #[], 0, 1, "", Lms.deployed, #[], "", .rows cappedSchedule .empty ⟨0, 0, 0, 0, 0, 0⟩, 0, none⟩

/-- The public `pub` charges the period's first start (`Public.start`): none where it is the run's first
(`p.first = 0`), whose start `RowsStateEmpty` holds to the empty state, and otherwise the scan's state's full width,
the bits of the start unit's `sin` (`Scan.start`). -/
def StartCharged (p : Params) (pub : Public) : Prop :=
  pub.start = match p.form with
    | .rows _ _ scan => if p.first = 0 then 0 else scan.start

/-- **The period check charges a period that does not begin the run its first start, at the state's full width.** If
the check accepts a period, its public's `start` is the scan's state's full width where the period's first execution
phase is not the run's first, and 0 where it is (`StartCharged`); `K` adds it once, a term of its own (`Public.K`). That
start is the state the execution phase before the period ended in, which the check holds to nothing, so an
adversary-chosen start carries no more than the bits `K` counts for it (`RowsKBoundsCount`'s `start`). -/
def RowsStartCharged : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → StartCharged p pub

/-- `RowsStartCharged` of any period check. -/
def RowsStartChargedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → StartCharged p pub

/-- The rows-form period of no execution phase and no audit under `cappedSchedule` that begins at the run's execution
phase 1, under a scan whose state is 5 bits wide: the period `RowsStartCharged`'s witnesses are stated at. -/
def chargedPeriod : Params :=
  ⟨#[], #[], 1, 1, "", Lms.deployed, #[], "", .rows cappedSchedule .empty ⟨0, 0, 0, 0, 0, 5⟩, 0, none⟩

/-- The `roots` entry `e` is the ok bits' constant of the scan's member `m` (its name, template and proof blocks'
count, `ScanOk.members`): its port is `<name>/o0`, its leaves are the member's proof blocks' count, and its domain and
root are, in hex, the member's `o0` frame domain `dom` (`ScanOk.domain`, the template's and the count's) and the root
of that many ok rows under the public salts at `dom` (`ScanOk.root`). -/
def OkEntry (m : String × String × ℕ) (e : Lean.Json) : Prop :=
  ∃ dom : ByteArray, NetworkVerifier.ScanOk.domain m.2.1 m.2.2 = .ok dom ∧
    e.getObjValD "port" = .str (m.1 ++ ("/" ++ NetworkVerifier.ScanOk.PORT)) ∧
    (e.getObjValD "leaves").getNat? = .ok m.2.2 ∧
    e.getObjValD "domain" = .str (Flock.hexOf dom) ∧
    e.getObjValD "root" = .str (Flock.hexOf (NetworkVerifier.ScanOk.root dom m.2.2))

/-- What `RowsOkHeld` says of a period its check accepted: in the rows form under a schedule of `T` ticks, of each
execution phase's scan's registration `reg`: `T` is at least 2; each member of the scan at `T` (`ScanOk.members`) has
a `roots` entry (`ScanOk.rootsOf`) that is its constant (`OkEntry`); and every `roots` entry whose port ends `/o0`
(`ScanOk.okPort`) but the start unit's (`ScanOk.START`, `start/o0`) is the constant of a member of the scan at `T`.
The start unit's `o0` is the state it copies, a row of the scan's state, and no ok bit; no statement here reads its
entry. That it is the one port exempt leans on the check's guard: `ScanOk.held` refuses, through `ScanOk.entryHeld`,
every other added port ending `/o0` that is no member's constant, as `ScanOk.lean`'s last `#guard` holds (`other/o0`
refused, `START` held). -/
def RowsOkHeldAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg →
      2 ≤ s.window ∧
      (∀ m ∈ NetworkVerifier.ScanOk.members s.window, ∃ e ∈ NetworkVerifier.ScanOk.rootsOf reg, OkEntry m e) ∧
      ∀ e ∈ NetworkVerifier.ScanOk.rootsOf reg, ∀ q : String, e.getObjValD "port" = .str q →
        NetworkVerifier.ScanOk.okPort q = true → q ≠ NetworkVerifier.ScanOk.START →
          ∃ m ∈ NetworkVerifier.ScanOk.members s.window, OkEntry m e

/-- `RowsOkHeld` of the period check `chk`: `RowsOkHeldAt` of every period `chk` accepts. -/
def RowsOkHeldOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsOkHeldAt p c

/-- **The rows form accepts no execution phase whose scan's ok bits are registered anywhere but at the constants of
its schedule.** If the check accepts a period in the rows form under a schedule of `T` ticks, then `T ≥ 2` wherever it
has an execution phase, and each execution phase's scan's registration names, for each member of the scan at `T`
(`ScanOk.members`: `ingress-0`, buckets `0 … T − 2`, `last`, bucket `T − 1`, and `egress-0`, the `T` egress buckets),
the port `<member>/o0` at the member's proof blocks' count, at its `o0` frame domain and at the root of that many ok
rows under the public salts (`OkEntry`), and names no port ending `/o0` other than a member's constant and the start
unit's (`ScanOk.START`, `start/o0`). The domain and the root are functions of `T` alone, which the check computes
(`ScanOk.consts`), so a registration that names a member's `o0` at a root of the prover's choosing, at another count
or domain, or under a name neither a member nor the start unit has, is refused. A member's proof blocks' ok words are
rows of the tree its statement reads at `o0`; that the statement reads the tree at the root the registration names is
the registration's check's, which the period check does not read. The start unit's `o0` is the state it copies, a row
of the scan's state, and no ok bit, and its entry is at any root, leaves and domain here: `RowsStateTied` holds the
scan's reads of its state, unit 0's `o0` at leaf 0 (`stateRead`), and no statement here holds the entry `start/o0` to
that state's root. That `start/o0` is the one port exempt leans on the check's guard: `ScanOk.held` refuses, through
`ScanOk.entryHeld`, every other added port ending `/o0` that is no member's constant, as `ScanOk.lean`'s last `#guard`
holds (`other/o0` refused, `START` held). -/
def RowsOkHeld : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsOkHeldAt p c

/-- A rows-form schedule of one slot, one lane and two ticks (`T = 2`). -/
def pairSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, none, none, none, false⟩

/-- The rows-form period of `n` execution phases and no audit under `pairSchedule`, at the deployed LMS parameters:
the periods `RowsOkHeld`'s witnesses are stated at. -/
def pairParams (n : ℕ) : Params :=
  ⟨#[], #[], 0, 1, "", Lms.deployed, #[], "", .rows pairSchedule .empty ⟨0, 0, 0, 0, 0, 0⟩, n, none⟩

/-- The rows-form period of no execution phase and no audit under `pairSchedule` whose parameters name the class
circuit `c`, at the deployed LMS parameters: the period `RowsCircuitHeld`'s honest witness is stated at. -/
def heldMonitoringPeriod : Params :=
  ⟨#[], #[], 0, 1, "", Lms.deployed, #[], "", .rows pairSchedule .empty ⟨0, 0, 0, 0, 0, 0⟩, 0, some "c"⟩

/-- A commitment of one execution phase whose scan's registration names no root, so no member's ok bits. -/
def unnamedCommitment : Commitment := ⟨#[], #[Lean.Json.mkObj []], .empty⟩


/-! ## An accepted rows-form period with an execution phase: the steps, unproved

The rows form's statements (`RowsProgWired`, `RowsStateTied`, `RowsTiedCharged`, `RowsTiedInputWired`, `RowsOkHeld`)
speak of each execution phase, and every acceptance Lean shows of `check` has none (`pairParams 0` among them).
These state what a witness with one execution phase and one compute audit rests on. Each is unproved, and no
guarantee reads one. -/

/-- A certificate that the key `k` at the deployed LMS parameters signs passes `Certificate.signs` under `k`'s public
key, at any link, direction, window and digest whose certificate message forms: the step from Lean's signer
(`Lms.signAt`) to the check's verification of each certificate. -/
def SignsCertificateAt : Prop :=
  ∀ (I seed : ByteArray) (k : Lms.PrivateKey) (session : ByteArray) (link : String) (d : NetworkVerifier.Direction)
    (w : ℕ) (digest : ByteArray),
    Lms.key Lms.deployed I seed = .ok k →
    (∃ m, NetworkVerifier.certificateMessage k.I session link d w digest = .ok m) →
    ∃ x : NetworkVerifier.Check.Certificate, x.keyId = k.I ∧ x.link = link ∧ x.direction = d ∧ x.window = w ∧
      x.digest = digest ∧ x.signs Lms.deployed k.publicKey session = true

/-- A ratio in lowest terms with a positive denominator reads back from the text `Core.Ratio.toString` writes: the
step a profile's `delta` takes when the check reads an audit's answer. -/
def RatioReadsBack : Prop :=
  ∀ (r : Core.Ratio) (ctx : String), 0 < r.den → Nat.gcd r.num r.den = 1 →
    Core.Ratio.ofString r.toString ctx = .ok r

/-- The profile `i` reads back from the JSON `Core.Integrity.toJson` writes. Not every profile does (an unsorted
`prescribed`, a ratio not in lowest terms), so a witness states it at its own audits' profiles. -/
def ProfileReadsBackAt (i : Core.Integrity) : Prop := Core.Integrity.ofJson i.toJson = .ok i

/-- `check` accepts a rows-form period that has an execution phase and a compute audit, under parameters one of whose
entries' keys is the public key of a key at the deployed LMS parameters. -/
def RowsExecutionPhaseAccepted : Prop :=
  ∃ (I seed : ByteArray) (k : Lms.PrivateKey) (p : Params) (c : Commitment) (e : Evidence) (pub : Public)
    (prof : Profile) (s : Schedule) (sd : ByteArray) (scan : Scan),
    Lms.key Lms.deployed I seed = .ok k ∧ p.lms = Lms.deployed ∧ (∃ x ∈ p.entries, x.key = k.publicKey) ∧
      p.form = .rows s sd scan ∧ 0 < c.windows.size ∧ 0 < c.audits.size ∧ check p c e = .accept pub prof

/-- The one-stage answers the period check reads: each compute audit's, then each execution phase's. -/
def answersRead (e : Evidence) : Array Core.Answer := e.audits ++ e.windows.map (·.answer)

/-- What `AnswersVerified` says of a period its check accepted: each answer the check reads is the one one-stage's
compiled verifier gives (`Spot.OneStage.protocol.answer`) on some parameters, commitment and evidence. The answer's
digest is that commitment's, which the check holds to the registration (`answered`), and an accepting one-stage check
holds its parameters' Program to that registration's; the parameters' query, samplingDesign and δ are not yet the period's. -/
def AnswersVerifiedAt (e : Evidence) : Prop :=
  ∀ a ∈ answersRead e, ∃ params commitment evidence : ByteArray,
    Spot.OneStage.protocol.answer params commitment evidence = a

/-- `AnswersVerified` of the period check `chk`: `AnswersVerifiedAt` of every period `chk` accepts. -/
def AnswersVerifiedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → AnswersVerifiedAt e

/-- **The period check accepts an execution phase only where the compiled Lean verifier verified every answer it
reads**: each compute audit's and each execution phase's one-stage answer is `Spot.OneStage.protocol.answer`'s on some
inputs (`AnswersVerifiedAt`). Not yet proved: the check reads the answers as the evidence states them. -/
def AnswersVerified : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → AnswersVerifiedAt e

/-! ## A key's listed sessions: the list is complete, unproved

A key may be listed in several sessions, each entry held to its session's signed start (`startChecks`).
`RowsEntriesParsed` states what the check holds of the entries themselves, and `RowsSessionsComplete` what it holds of
their starts: the listed sessions under one key are consecutive among its sessions, and each starts on the key's life
where the list's execution phases put it, after every execution phase the list gives the one before it. They and their
witnesses are unproved, and no guarantee reads them. -/

/-- The listed entries' starts strictly increase, and two entries whose keys have one identifier have one key. A key is
`I ‖ T[1]`, and its first 16 bytes are its LMS identifier `I`, which `check` pairs a key's entries by (`startRefusal`,
and `startCheck`'s choice of the entries `heldTo` and `follows` read) while it verifies each signature under the whole
key. Two LMS public keys can share an identifier (`I` is the key generator's to choose), so this is not assumed: the
check refuses both of any parameters (`entriesOrderRefusal`, `entriesKeyRefusal`), and the decoder refuses both too
(`certifierEntriesOf`). -/
def EntriesParsed (p : Params) : Prop :=
  (∀ (i : ℕ) (a b : Entry), p.entries[i]? = some a → p.entries[i + 1]? = some b → a.start < b.start) ∧
  ∀ (i j : ℕ) (a b : Entry), p.entries[i]? = some a → p.entries[j]? = some b →
    a.key.extract 0 16 = b.key.extract 0 16 → a.key = b.key

/-- `RowsEntriesParsed` of the period check `chk`: every period `chk` accepts has `EntriesParsed` entries. -/
def EntriesParsedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → EntriesParsed p

/-- **The check accepts only entries in order whose identifiers each name one key.** If the check accepts a period, its
listed entries' starts strictly increase and two of them with one key identifier have one key (`EntriesParsed`),
whatever parameters it was given: `startChecks` refuses the rest first (`entriesOrderRefusal`, `entriesKeyRefusal`). -/
def RowsEntriesParsed : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → EntriesParsed p

/-- `RowsEntriesParsed`'s nontrivial case of order, unproved: `acceptAllRows` accepts, and `check` refuses, a period
whose entry listed `i + 1`-th does not start after the one listed `i`-th. -/
def RowsEntriesOrderNontrivial : Prop :=
  ∃ (p : Params) (c : Commitment) (e : Evidence) (i : ℕ) (a b : Entry),
    p.entries[i]? = some a ∧ p.entries[i + 1]? = some b ∧ b.start ≤ a.start ∧
      (∃ (pub : Public) (prof : Profile), acceptAllRows p c e = .accept pub prof) ∧
      ∀ (pub : Public) (prof : Profile), check p c e ≠ .accept pub prof

/-- `RowsEntriesParsed`'s nontrivial case of keys, unproved: `acceptAllRows` accepts, and `check` refuses, a period
that lists two different keys with one key identifier. -/
def RowsEntriesKeyNontrivial : Prop :=
  ∃ (p : Params) (c : Commitment) (e : Evidence) (i j : ℕ) (a b : Entry),
    p.entries[i]? = some a ∧ p.entries[j]? = some b ∧ a.key.extract 0 16 = b.key.extract 0 16 ∧ a.key ≠ b.key ∧
      (∃ (pub : Public) (prof : Profile), acceptAllRows p c e = .accept pub prof) ∧
      ∀ (pub : Public) (prof : Profile), check p c e ≠ .accept pub prof

/-- The listed entries `i < j` of `p` are consecutive under one key: both name one key, and no entry listed between them
does. In a period the check accepts (`RowsEntriesParsed`) this is `check`'s own pairing, by the keys' identifiers. -/
def ConsecutiveUnderKey (p : Params) (i j : ℕ) : Prop :=
  i < j ∧ ∃ a b : Entry, p.entries[i]? = some a ∧ p.entries[j]? = some b ∧ a.key = b.key ∧
    ∀ (k : ℕ) (x : Entry), i < k → k < j → p.entries[k]? = some x → x.key ≠ a.key

/-- What `RowsSessionsComplete` says of a period its check accepted: the evidence shows one signed start for each
listed entry; each start's certificate signs the start digest of its base, its ordinal and its schedule
(`startDigest`) and verifies under its entry's key in its entry's session; and of two entries `a` (listed `i`-th) and
`b` (`j`-th) consecutive under one key (`ConsecutiveUnderKey`), with starts `xa` and `xb`, `n` the entry listed
`i + 1`-th, of any key, and `L = window · bucket_ms` an execution phase's milliseconds:

* `b`'s ordinal is `a`'s plus one;
* `xb.baseMs + a.start · L = xa.baseMs + b.start · L`: `a` and `b` put the list's execution phase 0 at one millisecond
  of the key's life, `xa.baseMs − a.start · L = xb.baseMs − b.start · L` as integers (before the key's life began if
  negative), which `heldTo` holds by tying each of the key's entries to its first;
* `xa.baseMs + n.start · L ≤ xb.baseMs + a.start · L`: measured from that phase 0, `a`'s stretch, which runs from
  `a.start` up to `n.start` (to `b`'s own start when no other key's entry lies between), ends no later on the key's
  life than `b`'s session starts.

Both lines add `a.start · L` to each side, so neither subtracts. -/
def RowsSessionsCompleteAt (p : Params) (e : Evidence) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    e.starts.size = p.entries.size ∧
    (∀ (i : ℕ) (a : Entry) (x : Start), p.entries[i]? = some a → e.starts[i]? = some x →
      startDigest x.baseMs x.ordinal x.schedule = some x.certificate.digest ∧
        x.certificate.signs p.lms a.key a.session = true) ∧
    ∀ (i j : ℕ) (a b n : Entry) (xa xb : Start), ConsecutiveUnderKey p i j → p.entries[i]? = some a →
      p.entries[j]? = some b → p.entries[i + 1]? = some n → e.starts[i]? = some xa → e.starts[j]? = some xb →
      xb.ordinal = xa.ordinal + 1 ∧ xb.baseMs + a.start * s.windowMs = xa.baseMs + b.start * s.windowMs ∧
        xa.baseMs + n.start * s.windowMs ≤ xb.baseMs + a.start * s.windowMs

/-- `RowsSessionsComplete` of the period check `chk`: `RowsSessionsCompleteAt` of every period `chk` accepts. -/
def RowsSessionsCompleteOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsSessionsCompleteAt p e

/-- **A key's listed sessions are complete from its first listed entry on.** If the check accepts a period, whatever
parameters it was given: each listed entry's session's start is shown, its certificate signed over its base, its
ordinal and its schedule under the entry's key in the entry's session; the listed sessions under one key carry
consecutive ordinals in the entries' order; and each such session's signed base is where the list's execution phases
put it, after every execution phase of its predecessor's stretch (`RowsSessionsCompleteAt`). The network attester's
device state gives each session it starts the next ordinal and none twice, and moves the key's life on by the
session's whole span (`certificate.DeviceState`, on `trusted-device`), so no session of the key started between two such
entries' sessions.

The base condition is read off the certificates, through `heldTo` and the entries' order, which the check holds itself
(`RowsEntriesParsed`), and needs no device state. It is of the earlier entry's stretch, not its session's whole span:
the start signs the session's base, its ordinal and the schedule's parameters digest, and no certificate signs the
span's length. It says nothing of how long the earlier session ran on its stretch, or whether the period's execution
phases after its last certificate have any: those are the execution phases' own checks'.

The list is complete from its first entry on and no further: the check reads no log of an earlier period, so nothing
here holds a key's first listed entry's ordinal to the last one of the period before, or says anything of a session
of the key before its first listed entry or after its last. -/
def RowsSessionsComplete : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsSessionsCompleteAt p e

/-- `RowsSessionsComplete`'s honest witness, unproved: `check` accepts a period at the deployed LMS parameters that
lists one key in two sessions. A period is one execution phase (`startCarriedRefusal`), so its witness is a period
after the restart, in the second session's stretch: a period spanning a restart is refused, and this says nothing of
a restarted device's period that holds the restart. -/
def RowsSessionsCompleteHonest : Prop :=
  ∃ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile) (s : Schedule) (sd : ByteArray)
    (scan : Scan) (a b : Entry),
    p.lms = Lms.deployed ∧ p.form = .rows s sd scan ∧ p.entries[0]? = some a ∧ p.entries[1]? = some b ∧
      a.key = b.key ∧ a.session ≠ b.session ∧ check p c e = .accept pub prof

/-- `RowsSessionsComplete`'s nontrivial witness, unproved: `RowsSessionsCompleteOf` fails at `acceptAllRows`. -/
def RowsSessionsCompleteNontrivial : Prop := ¬ RowsSessionsCompleteOf acceptAllRows

/-- A window record that states a sample: `population` units under the sampling design `design`. -/
def sampleReg (population : ℕ) (design : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("population", Lean.toJson population), ("law", design)]

/-- The sampling design `subset:k`, as a registration states it. -/
def subsetDesign (k : ℕ) : Lean.Json := Lean.Json.mkObj [("law", "subset"), ("k", Lean.toJson k)]

/-- A stratified sampling design of two strata, as a registration states it: `a`, the units `[0, na)` with `ka` of
them sampled, and `b`, the next `nb` units with `kb` of them sampled. -/
def strataDesign (na ka nb kb : ℕ) : Lean.Json :=
  Lean.Json.mkObj [("law", "stratified"), ("strata", Lean.Json.arr #[
    Lean.Json.mkObj [("template", "a"), ("k", Lean.toJson ka),
      ("units", Lean.Json.arr #[Lean.Json.arr #[Lean.toJson (0 : ℕ), Lean.toJson na]])],
    Lean.Json.mkObj [("template", "b"), ("k", Lean.toJson kb),
      ("units", Lean.Json.arr #[Lean.Json.arr #[Lean.toJson na, Lean.toJson (na + nb)]])]])]

/-- A registration of 16 units in two strata of 8, 4 of each sampled: the sample `RegisteredSampleHeld`'s and
`SampleBoundCharged`'s honest lemmas are stated at. -/
def halvesReg : Lean.Json := sampleReg 16 (strataDesign 8 4 8 4)

/-- One-stage's profile of that sample at δ = 1/10: one level of 16 units, stratified as registered. -/
def halvesProfile : Core.Integrity :=
  ⟨#[], #[], false, #[], #[⟨"unit", none, 16, some (.stratified #[⟨"a", 8, 4⟩, ⟨"b", 8, 4⟩])⟩], ⟨1, 10⟩, #[]⟩

/-- An answer's public that states the bound `b`. -/
def statedBound (b : ℕ) : Lean.Json := Lean.Json.mkObj [("bound", Lean.toJson b)]

/-- A public that holds the facts `f` of one compute audit and no term: what a period check publishes that reads
its audit's facts as the answer states them. -/
def statedPublic (f : Facts) : Public := rowsTerms #[f] ⟨0, 0, 0, 0, 0, 0⟩ #[] #[] 0 0

end Properties.NetworkPeriod
