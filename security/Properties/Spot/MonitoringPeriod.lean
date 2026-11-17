module

public import Spot.MonitoringPeriodAudit

/-! The properties of SPOT's period check (`Spot.MonitoringPeriod.checkWith`), each about the ledger presented, whose
ledger it is aside, and taking no premise. -/

@[expose] public section

namespace Properties.Spot.MonitoringPeriod

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger _root_.Spot.MonitoringPeriod

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

/-- **What the ledger's half returns**: the ledger passes `checked` under the verifier's key and reaches the period's
end, and every root the lane committed in the period is returned once, as a window when its session can be
sampled and as an other session when not; nothing else is returned. -/
def LedgerMeans : Prop :=
  ∀ (p : Params) (ledger : Json) (ws : Runs), runsOf p ledger = .ok ws →
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

/-- The check's one audit at window `w`: its session ran from a pinned commit on main's first-parent line, withholds
its coin commitment and shows exactly one registration, `a`'s, and the audit `inner` accepted it at the
parameters the check composes for the window (`paramsAt`), publishing what `a` holds. -/
def AuditedAt (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (r : Read)
    (w : _root_.Spot.MonitoringPeriod.Run) (we : RunEvidence) (before : Array Json) (record : Json) (a : Audit) : Prop :=
  ∃ (au : Audited) (opened : Json) (lc : LiveCommitment) (roots : Roots) (k : Case) (cm : OneStage.Commitment)
      (q : OneStageZk.Params) (ev : OneStageZk.Evidence),
    auditedIn r.wins r.periods (Json.mkObj [("root", Json.str w.root)]) = .ok (au, opened) ∧
    pinsReason au.session opened e.pins = none ∧
    firstParentReason au.session au.pinned p.main = none ∧
    challengePhaseNamed au p.ledger.key we.session = .ok (lc, roots) ∧
    withheldOn c.ledger w.root = true ∧
    sampledOn w.root roots = #[a.registration] ∧
    we.audits = #[k] ∧
    (Flock.canon k.registration >>= fun s => OneStage.commitmentOf s.toUTF8) = .ok cm ∧
    Flock.hexOf cm.digest = a.registration ∧ record = cm.record ∧
    OneStageZk.paramsOfJson (paramsAt p w.root k.pub before) = .ok q ∧
    (∃ header, (bs.get k.pub "public file" >>= FilePart.header) = .ok header ∧
      (field? header "shared_rows").isSome = false) ∧
    (evidenceAt p c.ledger e.pins we.session k.evidence >>= Flock.canon >>=
      fun s => OneStageZk.evidenceOf s.toUTF8) = .ok ev ∧
    inner q cm ev bs = .accept a.published a.profile

/-- **What the period's check accepts**: for every audit `inner` of one registration, an accept publishes the lane,
the period, the registration asked, the ledger's half's windows in order and its other sessions as `registeredIn`
returns them, and each window has exactly one audit (`AuditedAt`). -/
def AcceptMeans : Prop :=
  ∀ (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public)
      (prof : Array Core.Integrity),
    checkWith inner p c e bs = .accept pub prof →
      pub.lane = p.ledger.lane ∧ pub.start = p.ledger.start ∧ pub.stop = p.ledger.stop ∧
      pub.registration = p.registration ∧
      ∃ r ws, read p.ledger c.ledger = .ok r ∧ runsIn p.ledger r = .ok ws ∧
        registeredIn p r ws = .ok pub.others ∧
        pub.windows.map (·.window) = ws.windows ∧ e.windows.size = ws.windows.size ∧
        prof = pub.windows.flatMap (fun x => x.audits.map (·.profile)) ∧
        ∃ records : Array Json, records.size = pub.windows.size ∧
          ∀ i (h : i < pub.windows.size), ∃ we a record,
            e.windows[i]? = some we ∧ pub.windows[i].audits = #[a] ∧ records[i]? = some record ∧
            AuditedAt inner p c e bs r pub.windows[i].window we (records.extract 0 i) record a

/-- **Every root committed under this registration is one published window, with one audit entry**, and every
published window is such a root on a session that can be sampled. What the one audit is, is `AcceptMeans`'. -/
def CommittedOnce : Prop :=
  ∀ (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public)
      (prof : Array Core.Integrity),
    checkWith inner p c e bs = .accept pub prof →
      ∃ r, read p.ledger c.ledger = .ok r ∧ checked c.ledger p.ledger.key = .ok (r.wins, r.periods) ∧
        Reaches p.ledger c.ledger ∧
        (∀ x ∈ r.wins, Committed p.ledger c.ledger x →
          (∃ d, registrationOf r x.opened.session = some d) ∧
          (registrationOf r x.opened.session = some p.registration →
            ∃ i, ∃ h : i < pub.windows.size, pub.windows[i].window.root = x.root ∧
              pub.windows[i].window.session = x.opened.session ∧ (∃ a, pub.windows[i].audits = #[a]) ∧
              ∀ j (hj : j < pub.windows.size), pub.windows[j].window.root = x.root → j = i)) ∧
        ∀ w ∈ pub.windows, ∃ x ∈ r.wins, Committed p.ledger c.ledger x ∧ Sampled p.ledger x ∧
          registrationOf r x.opened.session = some p.registration ∧
          x.root = w.window.root ∧ x.opened.session = w.window.session

/-- **The other sessions published are exactly the lane's roots under other registrations**, each once, by its
circuit and session. Nothing else of an other session is published, and nothing is held of the root. -/
def OthersElsewhere : Prop :=
  ∀ (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public)
      (prof : Array Core.Integrity),
    checkWith inner p c e bs = .accept pub prof →
      ∃ r, read p.ledger c.ledger = .ok r ∧
        (∀ x ∈ r.wins, Committed p.ledger c.ledger x →
          ∀ d, registrationOf r x.opened.session = some d → d ≠ p.registration →
            ∃ o ∈ pub.others, o.registration = d ∧ o.session = x.opened.session) ∧
        (∀ o ∈ pub.others, o.registration ≠ p.registration ∧
          ∃ x ∈ r.wins, Committed p.ledger c.ledger x ∧
            registrationOf r x.opened.session = some o.registration ∧ x.opened.session = o.session) ∧
        (pub.others.toList.map (·.session)).Nodup

/-- **What the decoder gives the check**: the key the check hands one-stage v2 at each window reads as the key the
ledger and each session's files are checked under, and `audit` names none of the keys the check sets or leaves
unset at a window. -/
def DecodedMeans : Prop :=
  ∀ (j : Json) (p : AuditParams), AuditParams.of j = .ok p →
    ChallengerKey.ofJson p.keyJ = .ok p.ledger.key ∧
      field? p.audit "challenger" = none ∧ field? p.audit "window_root" = none ∧
      field? p.audit "seed_salt" = none ∧ field? p.audit "day" = none ∧ field? p.audit "declarations" = none

/-- **The session audited at a window is the window's**: `checked` refuses a second open of a root, so the session
`auditedIn` reads on a window's root is the one the window names. -/
def AuditedSession : Prop :=
  ∀ (p : Params) (ledger : Json) (r : Read) (ws : Runs) (w : _root_.Spot.MonitoringPeriod.Run) (au : Audited) (opened : Json),
    read p ledger = .ok r → runsIn p r = .ok ws → w ∈ ws.windows →
      auditedIn r.wins r.periods (Json.mkObj [("root", Json.str w.root)]) = .ok (au, opened) →
        au.session = w.session

end Properties.Spot.MonitoringPeriod
