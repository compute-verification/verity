module

public import Spot.PeriodAudit

/-!
# SPOT's period: what its check's accept means (`Spot.Period.checkWith`)

The properties of the period's check (`Spot.Period`, `Spot.PeriodAudit`), stated against its functions as they run,
for every input of the types its decoders produce. They take no premise: each is about the ledger the check was
presented, and says nothing of whose it is.

**The monitoring period in the ledger.** The check reads the monitoring period as a run of the ledger's entries:
from the first entry stamped at or after `start` (the run's end, where none is) up to the first stamped at or after
`stop` (`Run`). The lane commits a root in the monitoring period when a v2 open of the lane in that run opens a
session on it (`Committed`); the session can be sampled when its open's layout holds the sample's family
(`Spot.Period.holds`).

* `LedgerMeans`: what the ledger's half returns (`Spot.Period.windowsOf`). The ledger verifies under the verifier's
  key and reaches the monitoring period's end. Every root the lane committed in the monitoring period is one of the
  returned windows, when its session can be sampled, and one of the other sessions returned, when it cannot; nothing
  else is returned; and no root and no session is returned twice. Each window's session has ended, in a closed
  period all of whose sessions are windows.
* `AcceptMeans`: what the check accepts (`Spot.Period.checkWith`, at any audit of one registration; the check itself
  is `checkBlobs`, at one-stage v2's). It publishes the lane, the monitoring period, the ledger's half's windows in
  order and its other sessions. Each window has exactly one audit (`AuditedAt`): its session is the one the ledger
  sealed, run from a commit the pins log records on main's first-parent line, with its coin commitment withheld;
  its log shows the sample's coins released after exactly one registration; and the audit accepted that
  registration at the verifier's parameters for the window, which name the period's challenger and the window's own
  root and hold the records of the registrations audited at the windows before it.
* `CommittedOnce`: the two together, as a reader of the published answer needs them. Under an accept, every root the
  lane committed in the monitoring period on a session that can be sampled is the root of exactly one published
  window, which has one audit entry; every other root the lane committed in it is published among the other
  sessions; and each published window is such a root. What the one audit is, is `AcceptMeans`'.
* `EveryRootAudited`: the answer in which every root was audited. Under an accept that publishes no other session,
  every root the lane committed in the monitoring period is the root of exactly one published window, whose one
  audit is the check's audit at that window (`AuditedAt`).
* `DecodedMeans`: what the decoder gives the check (`AuditParams.of`). The key the audits are handed reads as the
  key the ledger is checked under, and `audit` names none of the keys a window sets or leaves unset. `AcceptMeans`
  is stated at every parameters, the decoder's or not; a reader who holds an accept to one-stage v2's own guarantee
  takes these two from here.
* `AuditedSession`: the session audited at a window is the window's own: the ledger opens a root once, so the
  session `AuditedAt` reads on a window's root is the one the window names.

**What these do not say.**

* Whose ledger it is. That the ledger presented is the auditor's device's own and whole is
  `Properties.Spot.PeriodLedger`'s.
* That every root the lane committed was audited. An accept holds the audit of each root on a session reserved with
  the sample's family. A root on any other session is published among the other sessions, and nothing is held of
  it. The lane names the families when it reserves a session, so only an answer with no other session says that
  every root was audited (`EveryRootAudited`).
* An other session's fields. Only its root and its session are tied to the ledger; its `families`, `withheld`,
  `place` and `period` are published, and no statement holds them.
* Who the lane is. `Committed` reads the name the open's body carries (`laneOf`: its `for` when non-empty, else its
  `by`) and asks a v2 open. Who may reserve under a name is the device's gate's, and a v1 open in the run is in
  neither list.
* The order of the windows. A window's phase is its place among the windows; nothing says the windows are in the
  order of their opens.
* Time. The monitoring period is two places in the ledger, picked by the stamps the device wrote; an entry with no
  stamp is passed over, and no statement reads a clock.
* What one-stage v2 holds through `seed_salt`, `day` and `declarations`. The check sets none of them at a window,
  and one-stage v2 holds nothing through one that is absent: the lane's declared program and the window's seed salt
  are not held in a period's audit.
* A registration under another family. "Exactly one registration" is of the sample's family: a second registration
  after which the session released another family's coins is passed over.
* That some registration passes. The statements are of every audit `inner`; that one-stage v2 accepts a window is
  not among them.
* A window whose session sampled several registrations of the sample's family, and a registration whose public
  file's header names `shared_rows`: the check refuses both.
-/

@[expose] public section

namespace Properties.Spot.Period

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger _root_.Spot.Period

/-- The ledger reaches the monitoring period's end: one of its entries is stamped at or after `stop`. -/
def Reaches (p : Params) (ledger : Json) : Prop :=
  (reaches (entriesOf ledger) p.stop).isSome

/-- The monitoring period as a run of the ledger's entries, `start` included and `stop` not: `stop` is the number of
the first entry stamped at or after the period's `stop`, and `start` of the first stamped at or after its `start`,
which is `stop` where none is. -/
def Run (p : Params) (ledger : Json) (start stop : Nat) : Prop :=
  reaches (entriesOf ledger) p.stop = some stop ∧ start = (reaches (entriesOf ledger) p.start).getD stop

/-- The lane committed the root of the ledger's window `x` in the monitoring period: `x`'s open is a v2 entry of the
lane whose number is in the monitoring period's run. -/
def Committed (p : Params) (ledger : Json) (x : Win) : Prop :=
  x.opened.schema = LEDGER_SCHEMA ∧ laneOf x.opened.body = p.lane ∧
    ∃ start stop, Run p ledger start stop ∧ start ≤ x.opened.seq ∧ x.opened.seq < stop

/-- The session the ledger's window `x` opens can be sampled: its open's layout holds the sample's family. -/
def Sampled (p : Params) (x : Win) : Prop :=
  holds p.label (getD x.opened.body "families") = true

/-- **What the ledger's half returns.** For every ledger whose windows `windowsOf` returns under the verifier's
parameters: the ledger passes `checked` under the verifier's key and reaches the monitoring period's end; every root
the lane committed in the monitoring period is a returned window, when its session can be sampled, and a returned
other session, when it cannot; every returned window is such a root whose session has ended, in a closed period that
lists it and lists only windows; every returned other session is such a root that cannot be sampled; there is at
least one window; no root and no session is returned twice, among the windows and the others together; and a
window's phase is its place among the windows. -/
def LedgerMeans : Prop :=
  ∀ (p : Params) (ledger : Json) (ws : Windows), windowsOf p ledger = .ok ws →
    ∃ wins periods, checked ledger p.key = .ok (wins, periods) ∧ Reaches p ledger ∧
      (∀ x ∈ wins, Committed p ledger x →
        (Sampled p x → ∃ w ∈ ws.windows, w.root = x.root ∧ w.session = x.opened.session) ∧
        (¬ Sampled p x → ∃ s ∈ ws.others, s.root = x.root ∧ s.session = x.opened.session)) ∧
      (∀ w ∈ ws.windows, ∃ x ∈ wins, Committed p ledger x ∧ Sampled p x ∧ x.root = w.root ∧
        x.opened.session = w.session ∧ x.ended.isSome ∧
        ∃ P ∈ periods, P.name = w.period ∧ P.closed = true ∧ w.session ∈ P.windows ∧
          ∀ s ∈ P.windows, ∃ v ∈ ws.windows, v.session = s) ∧
      (∀ s ∈ ws.others, ∃ x ∈ wins, Committed p ledger x ∧ ¬ Sampled p x ∧ x.root = s.root ∧
        x.opened.session = s.session) ∧
      0 < ws.windows.size ∧
      (ws.windows.toList.map (·.root) ++ ws.others.toList.map (·.root)).Nodup ∧
      (ws.windows.toList.map (·.session) ++ ws.others.toList.map (·.session)).Nodup ∧
      ∀ i (h : i < ws.windows.size), ws.windows[i].phase = i

/-- The check's one audit at a window. Of the window `w`, its evidence `we`, the records `before` of the
registrations audited at the windows before it, and the audit `a` published for it with the record `record`: the
ledger's window on `w`'s root is read (`auditedIn`); its session ran from a commit the pins log records
(`pinsReason`) on main's first-parent line (`firstParentReason`); the evidence's session files are that session's,
under the verifier's key (`sessionNamed`); every open of the root withholds its coin commitment (`withheldOn`); the
session's log shows the sample's coins released after exactly one registration, `a`'s (`sampledOn`); the evidence
holds one case, of that registration, whose record is `record`; the parameters at the window, which the check
composes from the verifier's own, the window's root, the case's public file's digest and `before` (`paramsAt`),
are one-stage v2's; that public file's header names no `shared_rows`; the case's evidence, with the period's
ledger, pins log and session files for its coins (`evidenceAt`), is one-stage v2's; and the audit `inner` accepted
them, publishing what `a` holds. -/
def AuditedAt (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (r : Read)
    (w : Window) (we : WindowEvidence) (before : Array Json) (record : Json) (a : Audit) : Prop :=
  ∃ (au : Audited) (opened : Json) (lc : LiveCommitment) (roots : Roots) (k : Case) (cm : OneStage.Commitment)
      (q : OneStageZk.Params) (ev : OneStageZk.Evidence),
    auditedIn r.wins r.periods (Json.mkObj [("root", Json.str w.root)]) = .ok (au, opened) ∧
    pinsReason au.session opened e.pins = none ∧
    firstParentReason au.session au.pinned p.main = none ∧
    sessionNamed au p.ledger.key we.session = .ok (lc, roots) ∧
    withheldOn c.ledger w.root = true ∧
    sampledOn p.ledger.label w.root roots = #[a.registration] ∧
    we.audits = #[k] ∧
    (Flock.canon k.registration >>= fun s => OneStage.commitmentOf s.toUTF8) = .ok cm ∧
    Flock.hexOf cm.digest = a.registration ∧ record = cm.record ∧
    OneStageZk.paramsOfJson (paramsAt p w.root k.pub before) = .ok q ∧
    (∃ header, (bs.get k.pub "public file" >>= FilePart.header) = .ok header ∧
      (field? header "shared_rows").isSome = false) ∧
    (evidenceAt p c.ledger e.pins we.session k.evidence >>= Flock.canon >>=
      fun s => OneStageZk.evidenceOf s.toUTF8) = .ok ev ∧
    inner q cm ev bs = .accept a.published a.profile

/-- **What the period's check accepts.** For every audit of one registration `inner` (the check's own is one-stage
v2's, `checkBlobs`), parameters, commitment, evidence and blobs: if `checkWith` accepts, it publishes the lane and
the monitoring period it was asked; the windows it publishes are the ledger's half's, in order, with its other
sessions; the evidence holds one window's evidence for each; its profile is the audits' profiles in order; and each
window has exactly one audit, which is the check's audit at that window after the records of the windows before it
(`AuditedAt`). -/
def AcceptMeans : Prop :=
  ∀ (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public)
      (prof : Array Core.Integrity),
    checkWith inner p c e bs = .accept pub prof →
      pub.lane = p.ledger.lane ∧ pub.start = p.ledger.start ∧ pub.stop = p.ledger.stop ∧
      ∃ r ws, read p.ledger c.ledger = .ok r ∧ windowsIn p.ledger r = .ok ws ∧
        pub.windows.map (·.window) = ws.windows ∧ pub.others = ws.others ∧ e.windows.size = ws.windows.size ∧
        prof = pub.windows.flatMap (fun x => x.audits.map (·.profile)) ∧
        ∃ records : Array Json, records.size = pub.windows.size ∧
          ∀ i (h : i < pub.windows.size), ∃ we a record,
            e.windows[i]? = some we ∧ pub.windows[i].audits = #[a] ∧ records[i]? = some record ∧
            AuditedAt inner p c e bs r pub.windows[i].window we (records.extract 0 i) record a

/-- **Each root committed on a session with the sample's family is one published window, with one audit entry.** For
every audit of one registration `inner`, parameters, commitment, evidence and blobs: if `checkWith` accepts, the
ledger passes `checked` under the verifier's key and reaches the monitoring period's end; every root the lane
committed in the monitoring period on a session that can be sampled is the root of exactly one published window, of
that session, which has one audit entry; every other root the lane committed in it is one of the published other
sessions; and every published window is a root the lane committed in the monitoring period on a session that can be
sampled. It says nothing of the audit entry: what the one audit is, is `AcceptMeans`, and the two are joined in
`EveryRootAudited`. -/
def CommittedOnce : Prop :=
  ∀ (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public)
      (prof : Array Core.Integrity),
    checkWith inner p c e bs = .accept pub prof →
      ∃ wp : Array Win × Array MonitoringPeriod, checked c.ledger p.ledger.key = .ok wp ∧
        Reaches p.ledger c.ledger ∧
        (∀ x ∈ wp.1, Committed p.ledger c.ledger x → Sampled p.ledger x →
          ∃ i, ∃ h : i < pub.windows.size, pub.windows[i].window.root = x.root ∧
            pub.windows[i].window.session = x.opened.session ∧ (∃ a, pub.windows[i].audits = #[a]) ∧
            ∀ j (hj : j < pub.windows.size), pub.windows[j].window.root = x.root → j = i) ∧
        (∀ x ∈ wp.1, Committed p.ledger c.ledger x → ¬ Sampled p.ledger x →
          ∃ s ∈ pub.others, s.root = x.root ∧ s.session = x.opened.session) ∧
        ∀ w ∈ pub.windows, ∃ x ∈ wp.1, Committed p.ledger c.ledger x ∧ Sampled p.ledger x ∧
          x.root = w.window.root ∧ x.opened.session = w.window.session

/-- **With no other session published, every root the lane committed is audited.** For every audit of one
registration `inner`, parameters, commitment, evidence and blobs: if `checkWith` accepts and publishes no other
session, the ledger's half reads the ledger, which passes `checked` under the verifier's key and reaches the
monitoring period's end, and every root the lane committed in the monitoring period is the root of exactly one
published window, of that root's session, whose one audit is the check's audit at that window after the records of
the windows before it (`AuditedAt`). Where the answer publishes another session the statement is silent: the root
on such a session was committed and no audit of it is held. -/
def EveryRootAudited : Prop :=
  ∀ (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public)
      (prof : Array Core.Integrity),
    checkWith inner p c e bs = .accept pub prof → pub.others = #[] →
      ∃ r, read p.ledger c.ledger = .ok r ∧ checked c.ledger p.ledger.key = .ok (r.wins, r.periods) ∧
        Reaches p.ledger c.ledger ∧
        ∃ records : Array Json, records.size = pub.windows.size ∧
          ∀ x ∈ r.wins, Committed p.ledger c.ledger x →
            ∃ i, ∃ h : i < pub.windows.size, pub.windows[i].window.root = x.root ∧
              pub.windows[i].window.session = x.opened.session ∧
              (∀ j (hj : j < pub.windows.size), pub.windows[j].window.root = x.root → j = i) ∧
              ∃ we a record, e.windows[i]? = some we ∧ pub.windows[i].audits = #[a] ∧
                records[i]? = some record ∧
                AuditedAt inner p c e bs r pub.windows[i].window we (records.extract 0 i) record a

/-- **What the decoder gives the check.** For every parameters `AuditParams.of` decodes: the key's JSON that the
check hands one-stage v2 at each window (`keyJ`, `paramsAt`'s `challenger.key`) reads, as one-stage v2 reads it
(`ChallengerKey.ofJson`), as the key the ledger and each session's files are checked under (`ledger.key`); and
`audit` names none of the keys the check sets at a window (`challenger`, `window_root`) or leaves unset
(`seed_salt`, `day`, `declarations`), so at a window one-stage v2's parameters hold the first two as the check set
them and none of the last three. -/
def DecodedMeans : Prop :=
  ∀ (j : Json) (p : AuditParams), AuditParams.of j = .ok p →
    ChallengerKey.ofJson p.keyJ = .ok p.ledger.key ∧
      field? p.audit "challenger" = none ∧ field? p.audit "window_root" = none ∧
      field? p.audit "seed_salt" = none ∧ field? p.audit "day" = none ∧ field? p.audit "declarations" = none

/-- **The session audited at a window is the window's.** For every ledger the ledger's half reads and every window
it returns: the session `auditedIn` reads on the window's root, which is the session `AuditedAt` holds every clause
of an audit to, is the session the window names. `checked` refuses a second open of a root, so the ledger opens the
window's root on one session. -/
def AuditedSession : Prop :=
  ∀ (p : Params) (ledger : Json) (r : Read) (ws : Windows) (w : Window) (au : Audited) (opened : Json),
    read p ledger = .ok r → windowsIn p r = .ok ws → w ∈ ws.windows →
      auditedIn r.wins r.periods (Json.mkObj [("root", Json.str w.root)]) = .ok (au, opened) →
        au.session = w.session

end Properties.Spot.Period
