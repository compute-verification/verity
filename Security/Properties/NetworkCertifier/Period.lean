import NetworkCheck.Period

/-!
# The network certifier over a period: what its Lean code (`NetworkCheck.Period`) establishes of each link-window

The property of `NetworkCheck.Period.protocol`'s check (`verity/certifier/v0`, onsite's `.network` component,
`Verity.component .network`), stated against its functions as they run: if the check accepts, the commitment registers
and the evidence answers exactly the period's `m` link-windows, and every link-window `i` (the session's window
`w = first + i`) is accepted as the period's own per-window check, `windowCheck`, accepts it (`PeriodWindowAccepted`):

* proved: one-stage's answer for its registration's digest accepted, with the profile of one unit drawn `subset:1`
  (`PeriodWindowProved`), its registration naming the link's Program at `w` (`PeriodWindowRegistered`);
* both ends certified: two certificates, egress then ingress, each the key's for its end of the link at `w`, verifying
  under LMS in the session (`PeriodWindowCertified`);
* its clock synced: the key's egress clock certificate of the link at `w`, its digest neither a sync's unsalted
  `clock_digest` nor an earlier link-window's in the period (`PeriodWindowClocked`).

The per-window predicate is the period check's own (`windowCheck`), not N2's link-window verdict
(`NetworkCheck.LinkWindow.verdict`, `Properties.NetworkCertifier.LinkWindowAcceptMeans`): the period check never runs
that verdict; it reads each link-window's one-stage answer, whose verifier of record is N2's (`LinkWindows.verify`).
What the certifier signs about the clock is witness, so the clock is synced as the auditor's device certified it
(`trusted-nebius-host`), not by a clear-text sync in Σ_sync.

Comparisons the code makes with a `BEq` instance not known to be lawful (`Json`, `ByteArray`, the profile's levels)
are stated as the code's `==` returning `true`.
-/

open NetworkCheck.Period
open Lean (Json)

namespace Properties.NetworkCertifier

/-- Link-window `w`'s registration names the link's Program and the provenance `{"kind": "served", "window": w}`. -/
def PeriodWindowRegistered (p : Params) (w : Nat) (reg : Registration) : Prop :=
  (reg.record.getObjValD "program" == .str p.program) = true ∧
    canonText (reg.record.getObjValD "window") = canonText (served w)

/-- Both ends of link-window `w` certified: two certificates, egress then ingress, each the key's for its end of the
link at `w`, its LMS signature verifying over `certificate_message` in the session (`certifies`). -/
def PeriodWindowCertified (p : Params) (w : Nat) (x : Window) : Prop :=
  ∃ eg ig, x.certificates = #[eg, ig] ∧ certifies p .egress w eg = true ∧ certifies p .ingress w ig = true

/-- Link-window `w`'s clock synced as the certifier certified it: its clock certificate is the key's egress certificate
of the link at `w`, and the digest it signs is salted (no sync of Σ_sync's `clock_digest` itself) and fresh (no
earlier link-window's clock digest in the period, `earlier`). -/
def PeriodWindowClocked (p : Params) (earlier : Array ByteArray) (w : Nat) (x : Window) : Prop :=
  certifies p .egress w x.clock = true ∧ (clearClocks p.syncSet).contains x.clock.digest = false ∧
    earlier.contains x.clock.digest = false

/-- The link-window was proved: its answer is one-stage's for the registration's digest and accepted, with a public
output and a profile of one level of one unit drawn `subset:1`, that one unit drawn. -/
def PeriodWindowProved (reg : Registration) (x : Window) : Prop :=
  x.answer.protocol = oneStage ∧ (x.answer.digest == some reg.digest) = true ∧ x.answer.verdict = .accept ∧
    ∃ pub prof i, x.answer.published = some pub ∧ x.answer.profile = some prof ∧ Core.Integrity.ofJson prof = .ok i ∧
      (i.levels.map (fun l => (l.name, l.size, l.draw)) == #[(unitLevel, 1, some (Core.Draw.subset 1))]) = true ∧
      (i.drawn == #[(unitLevel, 1)]) = true

/-- Link-window `w` (registered as `reg`, answered by `x`, after the period's earlier clock digests `earlier`) is
accepted: registered as the link's Program at `w`, both ends certified, its clock synced, and proved. -/
def PeriodWindowAccepted (p : Params) (earlier : Array ByteArray) (w : Nat) (reg : Registration) (x : Window) : Prop :=
  PeriodWindowRegistered p w reg ∧ PeriodWindowCertified p w x ∧ PeriodWindowClocked p earlier w x ∧
    PeriodWindowProved reg x

/-- The commitment registers and the evidence answers exactly the period's `p.windows` link-windows, and every one,
the `i`-th being the session's window `p.first + i`, is accepted after the clock digests of the period's link-windows
before it. -/
def PeriodWindowsAccepted (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  c.windows.size = p.windows ∧ e.windows.size = p.windows ∧
    ∀ (i : Nat) (reg : Registration) (x : Window), c.windows[i]? = some reg → e.windows[i]? = some x →
      PeriodWindowAccepted p ((e.windows.extract 0 i).map (·.clock.digest)) (p.first + i) reg x

/-- **ONE-13: an accepted period accepted every link-window of it**: if the network certifier's period check accepts,
every one of the period's link-windows was proved, certified on both ends and its clock synced. -/
def PeriodAcceptMeans : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    protocol.check p c e = .accept pub prof → PeriodWindowsAccepted p c e

end Properties.NetworkCertifier
