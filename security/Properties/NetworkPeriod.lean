module

public import NetworkCheck.Period
public import Spot.OneStage
public import Lms.Sign
public import Mathlib.Algebra.BigOperators.Fin

/-! The network period check's bound: `K` sums ceilings of `log2` of the count's factors, so it is at least `⌈log2⌉` of
the count and not equal to it, and `RowsKBoundsCount` says the count is at most `2 ^ K`. -/

@[expose] public section

namespace Properties.NetworkPeriod

open NetworkCheck.Period

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
  ∃ dom : ByteArray, NetworkCheck.ScanOk.domain m.2.1 m.2.2 = .ok dom ∧
    e.getObjValD "port" = .str (m.1 ++ ("/" ++ NetworkCheck.ScanOk.PORT)) ∧
    (e.getObjValD "leaves").getNat? = .ok m.2.2 ∧
    e.getObjValD "domain" = .str (Flock.hexOf dom) ∧
    e.getObjValD "root" = .str (Flock.hexOf (NetworkCheck.ScanOk.root dom m.2.2))

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
      (∀ m ∈ NetworkCheck.ScanOk.members s.window, ∃ e ∈ NetworkCheck.ScanOk.rootsOf reg, OkEntry m e) ∧
      ∀ e ∈ NetworkCheck.ScanOk.rootsOf reg, ∀ q : String, e.getObjValD "port" = .str q →
        NetworkCheck.ScanOk.okPort q = true → q ≠ NetworkCheck.ScanOk.START →
          ∃ m ∈ NetworkCheck.ScanOk.members s.window, OkEntry m e

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


/-! ## An accepted rows-form period with one execution phase: the steps a witness rests on, each unproved and read by no guarantee -/

def SignsCertificateAt : Prop :=
  ∀ (I seed : ByteArray) (k : Lms.PrivateKey) (session : ByteArray) (link : String) (d : NetworkCheck.Direction)
    (w : ℕ) (digest : ByteArray),
    Lms.key Lms.deployed I seed = .ok k →
    (∃ m, NetworkCheck.certificateMessage k.I session link d w digest = .ok m) →
    ∃ x : NetworkCheck.Check.Certificate, x.keyId = k.I ∧ x.link = link ∧ x.direction = d ∧ x.window = w ∧
      x.digest = digest ∧ x.signs Lms.deployed k.publicKey session = true

def RatioReadsBack : Prop :=
  ∀ (r : Core.Ratio) (ctx : String), 0 < r.den → Nat.gcd r.num r.den = 1 →
    Core.Ratio.ofString r.toString ctx = .ok r

/-- Not every profile reads back (an unsorted `prescribed`, a ratio not in lowest terms), so a witness states it at its own
audits' profiles. -/
def ProfileReadsBackAt (i : Core.Integrity) : Prop := Core.Integrity.ofJson i.toJson = .ok i

def RowsExecutionPhaseAccepted : Prop :=
  ∃ (I seed : ByteArray) (k : Lms.PrivateKey) (p : Params) (c : Commitment) (e : Evidence) (pub : Public)
    (prof : Profile) (s : Schedule) (sd : ByteArray) (scan : Scan),
    Lms.key Lms.deployed I seed = .ok k ∧ p.lms = Lms.deployed ∧ (∃ x ∈ p.entries, x.key = k.publicKey) ∧
      p.form = .rows s sd scan ∧ 0 < c.windows.size ∧ 0 < c.audits.size ∧ check p c e = .accept pub prof

def answersRead (e : Evidence) : Array Core.Answer := e.audits ++ e.windows.map (·.answer)

/-- The parameters' query, sampling design and δ are not held to the period's. -/
def AnswersVerifiedAt (e : Evidence) : Prop :=
  ∀ a ∈ answersRead e, ∃ params commitment evidence : ByteArray,
    Spot.OneStage.protocol.answer params commitment evidence = a

def AnswersVerifiedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → AnswersVerifiedAt e

/-- Unproved: the check reads the answers as the evidence states them. -/
def AnswersVerified : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → AnswersVerifiedAt e

/-! ## A key's listed sessions are complete: these statements and their witnesses are unproved, and no guarantee reads them -/

/-- Two LMS public keys can share an identifier `I`, the key generator's to choose, so this is not assumed: the check
refuses both (`entriesOrderRefusal`, `entriesKeyRefusal`), and so does the decoder (`certifierEntriesOf`). -/
def EntriesParsed (p : Params) : Prop :=
  (∀ (i : ℕ) (a b : Entry), p.entries[i]? = some a → p.entries[i + 1]? = some b → a.start < b.start) ∧
  ∀ (i j : ℕ) (a b : Entry), p.entries[i]? = some a → p.entries[j]? = some b →
    a.key.extract 0 16 = b.key.extract 0 16 → a.key = b.key

def EntriesParsedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → EntriesParsed p

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

def ConsecutiveUnderKey (p : Params) (i j : ℕ) : Prop :=
  i < j ∧ ∃ a b : Entry, p.entries[i]? = some a ∧ p.entries[j]? = some b ∧ a.key = b.key ∧
    ∀ (k : ℕ) (x : Entry), i < k → k < j → p.entries[k]? = some x → x.key ≠ a.key

/-- Both inequalities add `a.start · L` to each side, so neither subtracts. -/
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

def RowsSessionsCompleteOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsSessionsCompleteAt p e

/-- Rests on the network attester's device state (`certificate.DeviceState`, on `trusted-device`). The list is complete
from its first entry on and no further: the check reads no log of an earlier period, and nothing here says how long an
earlier session ran on its stretch. -/
def RowsSessionsComplete : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsSessionsCompleteAt p e

/-- `RowsSessionsComplete`'s honest witness, unproved: a period after a restart, in the second session's stretch; a period
spanning a restart is refused (`startCarriedRefusal`), and this says nothing of one. -/
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
