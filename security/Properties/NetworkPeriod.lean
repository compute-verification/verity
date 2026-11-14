module

public import NetworkCheck.Period
public import Mathlib.Algebra.BigOperators.Fin

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
  sync of Σ_sync), an audit's free inputs at most its layout's (`free`), its incorrect sets at most the sets of at most `b`
  of its `N` units, `β` at most `b · bits` with `bits` the most one of its Program's units commits (`faulty_bits`), and
  `Dj^k` at most `jitter ^ steps`;
- in the rows form the windows are the link-windows' scans: `#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes'
  lengths), a window's free inputs at most its scan's Program's and its tied audits' (`free`: the egress renders the
  outputs of the audits its scan's `prog` is tied to, so the bits the prover chose of their inputs are the window's),
  each window's incorrect sets at most the sets of at most `b` of its `N` units, `β` at most `b · max(b_ing, b_eg)`
  (`faulty_bits`) and, when a tied audit is sampled in part with a bound of at least one incorrect proof block, every
  slot entry bit (`faulty_entries`, `E_w`), and `Dj^k` at most `jitter ^ steps`.

The rows form's terms charge a tied audit (one a scan's `prog` is an output of) its free inputs, and its incorrect
proof blocks only as the slot entries they can set, since the scan reads the audit only through its `prog` and its
state is the scan's own. The check accepts a tied audit at any sample; the consistency check of one is `RowsProgWired`
and `RowsStateTied`, and what it publishes of one sampled in part is `RowsTiedCharged`. `RowsProgWired` says it accepts
no link-window whose scan's `prog` is not a compute audit's slot entries, a lane's tree the entries port of its chunk
of a `SlotEntry_v1` or `SelectEntry_v1` template of that audit's own Program. `RowsStateTied` says it accepts no
link-window whose scan's state is not the `SCAN_STATE` of an audit its `prog` is wired to, so a state that skips or
alters one of the scan's transitions is no Program's input. `RowsTiedCharged` says the public it accepts with supplies
the charge itself: `charged` is the facts of each tied audit sampled in part (`drawn < units`), each once, so each
one's δ is in `eps` (`Public.eps` sums the terms' facts' and `charged`'s, and nothing else), and where such an audit's
bound is at least one incorrect proof block, the term of every link-window whose scan's `prog` is its output has every
slot entry bit (`faultyEntries` at `scan.entries`), the number `RowsKBoundsCount` takes as `faulty`. No tied audit
names more sampled than its level holds (`factsOf` refuses a profile that does), so one that is not charged names as
many: `drawn = units`. Each says its conclusion (`RowsProgWiredAt`, `RowsStateTiedAt`,
`RowsTiedChargedAt`) of every period `check` accepts, and so is, by definition, its property of any period check at
`check` (`RowsProgWiredOf`, `RowsStateTiedOf`, `RowsTiedChargedOf`), which the nontrivial lemmas state false at
`acceptAllRows`, a check that accepts every period. Each guarantee writes its acceptance of `check` out rather than
being its `…Of check`: the shape rule reads the acceptance off the statement's own binders.
`RowsCapPublished` says an accepted period's public names its schedule's response cap, and `RowsCapPublishedOf` is
its property of any period check, false at `acceptAllRows` too.
-/

@[expose] public section

namespace Properties.NetworkPeriod

open NetworkCheck.Period

/-- **The clock form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, `n` compute audits `a` and their facts `w`, the jitter bound `jitter`, `#Σ_sync` `syncs` and `windows`
link-windows: `#Ss` at `syncs ^ windows` (`schedules`), `∏_c (cap_c + 1)` at `∏_j (caps j + 1)` (`cells`), and audit
`i`'s `2^#(free inputs)` at `2 ^ (a i).free`, `#faultySets` at `atMost N b`, `2^β` at `2 ^ (b · (a i).bits)` (each of
its at most `b` incorrect proof blocks commits at most its Program's widest unit's bits) and `Dj^k` at `jitter ^ (a i).steps`,
their product is at most `2 ^ K`. -/
def ClockKBoundsCount : Prop :=
  ∀ (jitter syncs windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (a : Fin n → Audit) (w : Fin n → Facts),
    syncs ^ windows * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (a i).free * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * (a i).bits) * jitter ^ (a i).steps)) ≤
      2 ^ (terms (Array.ofFn a) (Array.ofFn w) (Array.ofFn caps) jitter syncs windows).K

/-- **The rows form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, the facts `w` of `n` link-windows' scans, their tied audits' free inputs `tied` (`tiedAudits`) and the
slot entry bits their tied audits' incorrect proof blocks can set `faulty` (`faultyEntries`), under one scan's terms `scan`, the
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
of a template node of that audit's own Program calling `SlotEntry_v1` or `SelectEntry_v1` (whose select and collector
end in `SlotEntry_v1`), a Definition the check pins by its closure digest (`entryPins`, so a template under either name
with another body is refused), with one member a slot of the window (`wired`), so chunk `j` is lane `j`'s slots and each
slot's entry, read bit included, is one its Program computed. -/
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

/-- Audit `i`'s answer gives the facts `f`: one-stage accepted its window record `reg` with a profile of one level,
sampled `subset:k` or stratified, and `f` is what the check reads of that profile and its public (`answered`,
`factsOf`). -/
def AuditFacts (i : ℕ) (reg : Lean.Json) (x : Core.Answer) (f : Facts) : Prop :=
  ∃ (prof : Core.Integrity) (pub : Lean.Json),
    answered s!"audit {i}" reg x = .ok (prof, pub) ∧ factsOf s!"audit {i}" prof pub = .ok f

/-- Among the audits' facts `facts`, in their order and each once, those of each audit `tied` ties that is sampled in
part (`drawn < units`). -/
def tiedInPart (tied : ℕ → Bool) (facts : Array Facts) : Array Facts :=
  (Array.range facts.size).filterMap fun i => facts[i]?.filter fun f => tied i && decide (f.drawn < f.units)

/-- What `RowsTiedCharged` says of a period its check accepted with the public `pub`: in the rows form, `pub.charged`
is `tiedInPart` of the public's audits' facts under the commitment's ties (`tiedIn`), and each tied audit `i` has an
answer `x` whose facts `f` (`AuditFacts`) are the public's `audits[i]` and name no more sampled than its level holds
(`f.drawn ≤ f.units`); if it is sampled in part (`f.drawn < f.units`), `f` is among
`pub.charged` and, when its bound is at least one incorrect proof block (`1 ≤ f.wrong`), every link-window `w` whose
scan's `prog` is its output (`tiedAudits`) has `faultyEntries = scan.entries` in the public's term `pub.units[w]`. -/
def RowsTiedChargedAt (p : Params) (c : Commitment) (e : Evidence) (pub : Public) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    pub.charged = tiedInPart (tiedIn c.audits c.windows) pub.audits ∧
    ∀ (i : ℕ) (reg : Lean.Json), c.audits[i]? = some reg → tiedIn c.audits c.windows i = true →
      ∃ (x : Core.Answer) (f : Facts), e.audits[i]? = some x ∧ AuditFacts i reg x f ∧ pub.audits[i]? = some f ∧
        f.drawn ≤ f.units ∧
        (f.drawn < f.units → f ∈ pub.charged ∧ (1 ≤ f.wrong →
          ∀ (w : ℕ) (wreg : Lean.Json), c.windows[w]? = some wreg →
            (tiedAudits c.audits wreg).contains i = true →
              ∃ t : AuditTerms, pub.units[w]? = some t ∧ t.faultyEntries = scan.entries))

/-- `RowsTiedCharged` of the period check `chk`: `RowsTiedChargedAt` of every period `chk` accepts, at the public it
accepts with. -/
def RowsTiedChargedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsTiedChargedAt p c e pub

/-- **The rows form charges every tied audit it accepts sampled in part.** If the check accepts a period in the rows
form with the public `pub`, then `pub.charged` is the facts of each audit that some link-window's scan's `prog` is an
output of (`tiedIn`) and that is sampled in part (`drawn < units`), in the audits' order and each once (`tiedInPart` of
`pub.audits`). `Public.eps` sums the δ of the terms' facts and of `charged`, and of nothing else, so each such audit's
δ is in `eps`. And for each such audit whose bound is at least one incorrect proof block, the public's term of every
link-window whose scan's `prog` is its output has `faultyEntries` at the scan's every slot entry bit (`Scan.entries`),
which `K` sums. So the `faulty` `RowsKBoundsCount` takes and the δ of the audits it is read from are the check's own,
of every period it accepts.

Every tied audit of an accepted period names no more sampled than its level holds: the check reads both counts off the
answer's profile, the level's size and its `subset`'s `k` or its strata's `k` summed, and refuses a profile that names
more sampled than the level has (`factsOf`). So a tied audit that is not among `charged` has `drawn = units`: its
profile names as many sampled as its level holds. Of a `subset` sample that is the level sampled whole. Of a
stratified one it is each stratum sampled whole, because the check also refuses a profile with a stratum that samples
more than it holds, and one whose strata's sizes do not sum to the level's. That the strata are the level's units is
the audit's own verifier's, which the period check does not read. -/
def RowsTiedCharged : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsTiedChargedAt p c e pub

/-- A period check that accepts every period, publishing the rows form's terms of no link-window, no audit's facts and
no charge: what `RowsProgWiredOf`, `RowsStateTiedOf`, `RowsTiedChargedOf` and `RowsCapPublishedOf` must fail at for
`RowsProgWired`, `RowsStateTied`, `RowsTiedCharged` and `RowsCapPublished` (each its `…Of` at `check`) to say
something. -/
def acceptAllRows (p : Params) (_ : Commitment) (_ : Evidence) : Core.Outcome Public Profile :=
  .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] p.cells p.jitter p.windows) { link := p.link, windows := p.windows }

/-- The public `pub` publishes the response cap of the period's schedule (`Form.maxTokensCap`): the rows form's
`max_tokens_cap`, none when its schedule has none and in the clock form. -/
def CapPublished (p : Params) (pub : Public) : Prop := pub.maxTokensCap = p.form.maxTokensCap

/-- **The period check publishes its schedule's response cap.** If the check accepts a period, its public's
`max_tokens_cap` is the cap of the schedule the parameters name (`CapPublished`), the schedule whose digest every
link-window's registrations bind (`rowsCheck`): a reader of the public, the communication audit's charge, reads the cap
the network certifier held each released response to (`NetTiming.Guarantees.ResponseCapped`), and none where the
schedule has none. -/
def RowsCapPublished : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → CapPublished p pub

/-- `RowsCapPublished` of any period check. -/
def RowsCapPublishedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → CapPublished p pub

/-- A rows-form schedule of one slot, one lane and a window of one step, with a response cap of 4. -/
def cappedSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, some 4⟩

/-- The rows-form period of no link-window and no audit under `cappedSchedule`, at the deployed LMS parameters: the
period `RowsCapPublished`'s witnesses are stated at. -/
def cappedPeriod : Params :=
  ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows cappedSchedule .empty ⟨0, 0, 0, 0, 0⟩, 0⟩

end Properties.NetworkPeriod
