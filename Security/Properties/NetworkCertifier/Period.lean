import NetworkCheck.Period
import Lms.Sign

/-!
# The network certifier over a period: what its Lean code (`NetworkCheck.Period`) establishes of each link-window

The property of `NetworkCheck.Period.protocol`'s check (`verity/certifier/v0`, onsite's `.network` component,
`Verity.component .network`), stated against its functions as they run: if the check accepts, the commitment registers
and the evidence answers exactly the period's `m` link-windows, and every link-window `i` (the session's window
`w = first + i`) is accepted as the period's own per-window check of its form accepts it.

In the clock form (`windowCheck`, `PeriodWindowAccepted`):

* proved: one-stage's answer for its registration's digest accepted, with the profile of one unit drawn `subset:1`
  (`PeriodWindowProved`), its registration naming the link's Program at `w` (`PeriodWindowRegistered`);
* both ends certified: two certificates, egress then ingress, each the key's for its end of the link at `w`, verifying
  under LMS in the session (`PeriodWindowCertified`);
* its clock synced: the key's egress clock certificate of the link at `w`, its digest neither a sync's unsalted
  `clock_digest` nor an earlier link-window's in the period (`PeriodWindowClocked`).

In the rows form (`rowsWindowCheck`, `PeriodRowsWindowAccepted`):

* its registration names the scan's Program (`PeriodWindowProgram`);
* its three records certified: the requests' (the link's ingress), the responses' (its egress) and the fails'
  (`<link>-fails`, egress) at `w`, each the key's and verifying in the session (`PeriodRowsCertified`);
* its rows registered: the ingress then the egress rows, each `warden_rows.check_registration`'s (`PeriodRowsChecked`:
  the link-window's value, link, end and window, the bound schedule's digest and rows, the link-window's domain, and the
  key's certificate on the head), both held by the scan's registration (`PeriodRowsRegistered`);
* proved: one-stage's answer for its registration's digest accepted, with one level of the scan's `2T` units drawn
  `subset:k` or stratified and a `bound` (`PeriodRowsProved`).

The per-window predicate is the period check's own, not N2's link-window verdict (`NetworkCheck.LinkWindow.verdict`,
`Properties.NetworkCertifier.LinkWindowAcceptMeans`): the period check never runs that verdict; it reads each
link-window's one-stage answer, whose verifier of record is N2's (`LinkWindows.verify`) or the scan's. What the
certifier signs about the clock is witness, so the clock is synced as the auditor's device certified it
(`trusted-host`), not by a clear-text sync in Σ_sync.

Comparisons the code makes with a `BEq` instance not known to be lawful (`Json`, `ByteArray`, `String`, the profile's
levels) are stated as the code's `==` returning `true`.

Beside the property, its two lemmas' objects: `PeriodAcceptMeansOf`, the property of any period check, and
`acceptAllPeriod`, a check that accepts every period; and an honest period of one clock-form link-window
(`honestParams`, `honestCommitment`, `honestEvidence`), its certificates signed with the certifier's key at the deployed
LMS parameters, with the two named assumptions about code the kernel can't evaluate that its acceptance takes
(`JsonStrBeq`, `HonestProfileReads`).
-/

open NetworkCheck.Period
open NetworkCheck.Check (Certificate)
open Lean (Json)

namespace Properties.NetworkCertifier

/-- Link-window `w`'s registration names the period's Program: N2's for the link in the clock form, the scan's in the
rows form. -/
def PeriodWindowProgram (p : Params) (reg : Registration) : Prop :=
  (reg.record.getObjValD "program" == .str p.program) = true

/-- Link-window `w`'s registration names the link's Program and the provenance `{"kind": "served", "window": w}`. -/
def PeriodWindowRegistered (p : Params) (w : Nat) (reg : Registration) : Prop :=
  PeriodWindowProgram p reg ∧ canonText (reg.record.getObjValD "window") = canonText (served w)

/-- Both ends of link-window `w` certified: two certificates, egress then ingress, each the key's for its end of the
link at `w`, its LMS signature verifying over `certificate_message` in the session (`certifies`). -/
def PeriodWindowCertified (p : Params) (w : Nat) (x : Window) : Prop :=
  ∃ eg ig, x.certificates = #[eg, ig] ∧ certifies p .egress w eg = true ∧ certifies p .ingress w ig = true

/-- Link-window `w`'s clock synced as the certifier certified it: it shows a clock certificate, the key's egress
certificate of the link at `w`, and the digest it signs is salted (no sync of Σ_sync `ss`'s `clock_digest` itself) and
fresh (no earlier link-window's clock digest in the period, `earlier`). -/
def PeriodWindowClocked (p : Params) (ss : NetworkCertifier.SyncSet) (earlier : Array ByteArray) (w : Nat)
    (x : Window) : Prop :=
  ∃ clock, x.shown = .clock clock ∧ certifies p .egress w clock = true ∧
    (clearClocks ss).contains clock.digest = false ∧ earlier.contains clock.digest = false

/-- The link-window was proved: its answer is one-stage's for the registration's digest and accepted, with a public
output and a profile of one level of one unit drawn `subset:1`, that one unit drawn. -/
def PeriodWindowProved (reg : Registration) (x : Window) : Prop :=
  x.answer.protocol = oneStage ∧ (x.answer.digest == some reg.digest) = true ∧ x.answer.verdict = .accept ∧
    ∃ pub prof i, x.answer.published = some pub ∧ x.answer.profile = some prof ∧ Core.Integrity.ofJson prof = .ok i ∧
      (i.levels.map (fun l => (l.name, l.size, l.draw)) ==
        #[(unitLevel, 1, some (Core.Draw.subset 1))]) = true ∧
      (i.drawn == #[(unitLevel, 1)]) = true

/-- Clock-form link-window `w` (registered as `reg`, answered by `x`, under Σ_sync `ss`, after the period's earlier
clock digests `earlier`) is accepted: registered as the link's Program at `w`, both ends certified, its clock synced,
and proved. -/
def PeriodWindowAccepted (p : Params) (ss : NetworkCertifier.SyncSet) (earlier : Array ByteArray) (w : Nat)
    (reg : Registration) (x : Window) : Prop :=
  PeriodWindowRegistered p w reg ∧ PeriodWindowCertified p w x ∧ PeriodWindowClocked p ss earlier w x ∧
    PeriodWindowProved reg x

/-- Rows-form link-window `w`'s three records certified: the requests' (the link's ingress), the responses' (its
egress) and the fails' (`<link>-fails`, egress), each the key's at `w` and verifying in the session (`certifiesOn`). -/
def PeriodRowsCertified (p : Params) (w : Nat) (x : Window) : Prop :=
  ∃ rq rp fl, x.certificates = #[rq, rp, fl] ∧ certifiesOn p p.link .ingress w rq = true ∧
    certifiesOn p p.link .egress w rp = true ∧ certifiesOn p (p.link ++ "-fails") .egress w fl = true

/-- **Link-window `w`'s rows in direction `d` are `warden_rows.check_registration`'s** under the schedule `s` whose
digest is `sd`: the link-window's value, link, end and window; the bound schedule's digest; the schedule's rows, words,
word bits, row schema and layout; the link-window's domain in the session; and a certificate under the deployment key
in the session that signs the rows' head. -/
def PeriodRowsChecked (p : Params) (s : Schedule) (sd : ByteArray) (w : Nat) (d : NetworkCheck.Direction) (r : Rows) :
    Prop :=
  (r.format == rowsFormat && r.value == s!"{p.link}/{d.value}/{w}" && r.link == p.link && r.direction == d.value &&
      r.window == w) = true ∧
    (r.parameters == Flock.hexOf sd) = true ∧
    (r.cellBytes == s.cellBytes && r.rows == s.window && r.words == s.allocation * s.cellBytes / 2 &&
      r.wordBits == outBits && r.rowSchema == rowSchema &&
      r.layout == Layout.mk s.window s.allocation s.slots (w * s.window)) = true ∧
    ∃ dom head, rowsDomain s sd p.session p.link d w = .ok dom ∧ (r.domain == Flock.hexOf dom) = true ∧
      headDigest p.link d w sd s r.wiring dom r.root = .ok head ∧ (r.digest == head) = true ∧
      certifiesOn p p.link d w
        { keyId := r.key, link := p.link, direction := d, window := w, digest := r.digest, signature := r.signature } =
        true

/-- Rows-form link-window `w`'s rows registered: it shows two registrations, the ingress rows then the egress rows,
each `PeriodRowsChecked`, and the scan's registration `reg` holds both (an entry of its `roots` or `registered` names
each one's root and domain). -/
def PeriodRowsRegistered (p : Params) (s : Schedule) (sd : ByteArray) (w : Nat) (reg : Registration) (x : Window) :
    Prop :=
  ∃ ri re, x.shown = .rows #[ri, re] ∧ PeriodRowsChecked p s sd w .ingress ri ∧
    PeriodRowsChecked p s sd w .egress re ∧ holds reg ri = true ∧ holds reg re = true

/-- The rows-form link-window was proved: its answer is one-stage's for the registration's digest and accepted, with a
public output holding a `bound` and a profile of one level drawn `subset:k` or stratified, of the scan's `2T` units
under the schedule `s` (the ingress and the egress rows, `T` each). -/
def PeriodRowsProved (s : Schedule) (reg : Registration) (x : Window) : Prop :=
  x.answer.protocol = oneStage ∧ (x.answer.digest == some reg.digest) = true ∧ x.answer.verdict = .accept ∧
    ∃ pub prof i l b, x.answer.published = some pub ∧ x.answer.profile = some prof ∧
      Core.Integrity.ofJson prof = .ok i ∧ i.levels = #[l] ∧ l.size = 2 * s.window ∧
      ((∃ k, l.draw = some (.subset k)) ∨ ∃ ss, l.draw = some (.stratified ss)) ∧
      Flock.J.nat pub "bound" "public" = .ok b

/-- Rows-form link-window `w` (registered as `reg`, answered by `x`, under the schedule `s` whose digest is `sd`) is
accepted: registered as the scan's Program, its three records certified, its rows registered, and proved. -/
def PeriodRowsWindowAccepted (p : Params) (s : Schedule) (sd : ByteArray) (w : Nat) (reg : Registration) (x : Window) :
    Prop :=
  PeriodWindowProgram p reg ∧ PeriodRowsCertified p w x ∧ PeriodRowsRegistered p s sd w reg x ∧
    PeriodRowsProved s reg x

/-- The commitment registers and the evidence answers exactly the period's `p.windows` link-windows, and every one,
the `i`-th being the session's window `p.first + i`, is accepted in the parameters' form: in the clock form after the
clock digests of the period's link-windows before it. -/
def PeriodWindowsAccepted (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  c.windows.size = p.windows ∧ e.windows.size = p.windows ∧
    ∀ (i : Nat) (reg : Registration) (x : Window), c.windows[i]? = some reg → e.windows[i]? = some x →
      match p.form with
      | .clock ss => PeriodWindowAccepted p ss ((e.windows.extract 0 i).filterMap (·.clockDigest?)) (p.first + i) reg x
      | .rows s sd _ => PeriodRowsWindowAccepted p s sd (p.first + i) reg x

/-- **An accepted period accepted every link-window of it**: if the network certifier's period check accepts,
every one of the period's link-windows passed its form's check: in the clock form proved, certified on both ends and
its clock synced; in the rows form proved, its three records certified and its rows registered. -/
def PeriodAcceptMeans : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    protocol.check p c e = .accept pub prof → PeriodWindowsAccepted p c e

/-! ## Another period check in the check's place -/

/-- `PeriodAcceptMeans` of a period check `check` in `protocol.check`'s place: whenever `check` accepts a period, every
one of its link-windows was accepted. At `protocol.check` it is `PeriodAcceptMeans` itself. -/
def PeriodAcceptMeansOf (check : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → PeriodWindowsAccepted p c e

/-- A period check that accepts every period, publishing the terms of a period with no audit: what
`PeriodAcceptMeansOf` must fail at for `PeriodAcceptMeans` to say something. -/
def acceptAllPeriod (p : Params) (_ : Commitment) (_ : Evidence) : Core.Outcome Public Profile :=
  .accept (terms #[] #[] p.cells p.jitter 0 p.windows) { link := p.link, windows := p.windows }

/-! ## An honest period: one clock-form link-window -/

/-- Σ_sync of one sync, δ = 0 and ρ = 1. -/
def honestSyncs : NetworkCertifier.SyncSet where
  deltas := [0]
  rhos := [1]
  deltas_nonempty := by decide
  rhos_nonempty := by decide
  rhos_pos := by decide
  deltas_nodup := by decide
  rhos_nodup := by decide

/-- The certifier key's 16-byte LMS identifier. -/
def honestI : ByteArray := ⟨Array.replicate 16 1⟩

/-- The certifier's key: `Lms.key` at the deployed parameters, which derives it from any 16-byte identifier. Written
without a `match`: the kernel reduces a `match` on `Lms.key` by building the key's 2^20-leaf tree, so a proof relating
the key to `Lms.key`'s result must unfold `honestKey` before anything else. -/
def honestKey : Lms.PrivateKey :=
  (Lms.key Lms.deployed honestI ⟨Array.replicate 64 2⟩).toOption.getD ⟨Lms.deployed, honestI, .empty, #[]⟩

/-- The certifier's 32-byte session. -/
def honestSession : ByteArray := ⟨Array.replicate 32 3⟩

/-- The certificate of the link's end `d` at window `w` on the 64-byte digest `x`, signed with the key's one-time key `q`
over its `certificate_message` (without a `match`, as `honestKey`). -/
def honestCert (d : NetworkCheck.Direction) (w q : Nat) (x : ByteArray) : Certificate :=
  { keyId := honestI, link := "link", direction := d, window := w, digest := x,
    signature := (((NetworkCheck.certificateMessage honestI honestSession "link" d w x).bind
      (Lms.signAt honestKey q)).toOption.map Lms.Signature.encode).getD .empty }

/-- The period's parameters: the link `link` under N2's Program `program`, one clock-form link-window from window 0, no
audit, and the certifier's key at the deployed LMS parameters in its session. -/
def honestParams : Params where
  audits := #[]
  cells := #[]
  first := 0
  jitter := 0
  link := "link"
  lms := Lms.deployed
  key := honestKey.publicKey
  session := honestSession
  program := "program"
  form := .clock honestSyncs
  windows := 1

/-- Link-window 0's registration: the link's Program served at window 0. -/
def honestRegistration : Registration :=
  ⟨Json.mkObj [("program", .str "program"), ("window", served 0)], ⟨Array.replicate 64 4⟩⟩

/-- One-stage's profile of the link-window's proof: one level of one unit drawn `subset:1`, that unit drawn. -/
def honestIntegrity : Core.Integrity :=
  { programs := #[], prescribed := #[], linked := true, covered := #[],
    levels := #[{ name := unitLevel, query := none, size := 1, draw := some (.subset 1) }], delta := ⟨0, 1⟩,
    drawn := #[(unitLevel, 1)] }

/-- One-stage's acceptance of the registration, with that profile. -/
def honestAnswer : Core.Answer :=
  { protocol := oneStage, verdict := .accept, reason := none, digest := some honestRegistration.digest,
    published := some (Json.mkObj []), profile := some honestIntegrity.toJson }

/-- Link-window 0 as the evidence answers it: one-stage's acceptance, the certificates of both ends of the link at window
0 (egress then ingress) and the clock's, each on its own one-time key. -/
def honestWindow : Window :=
  { answer := honestAnswer,
    certificates := #[honestCert .egress 0 0 ⟨Array.replicate 64 5⟩, honestCert .ingress 0 1 ⟨Array.replicate 64 6⟩],
    shown := .clock (honestCert .egress 0 2 ⟨Array.replicate 64 7⟩) }

def honestCommitment : Commitment := ⟨#[], #[honestRegistration], .empty⟩

def honestEvidence : Evidence := ⟨#[], #[honestWindow]⟩

/-- **Named assumption: `Json`'s `==` is `String`'s on two strings.** Lean core's `Json` `==` is a `partial def`
(`Json.beq'`), so no proof evaluates it; its case for two strings compares them with `String`'s `==`, as its source
states. The period check compares a registration's Program with it (`windowCheck`, `rowsWindowCheck`, `auditCheck`); a
check comparing the Program as a `String` needs no assumption. -/
def JsonStrBeq : Prop := ∀ s : String, (Json.str s == Json.str s) = true

/-- **Named assumption: the honest profile reads back.** `Core.Integrity.ofJson` reads `honestIntegrity`'s JSON back as
it: true by evaluation, but `Core.Ratio.ofString` reads δ with `String.splitOn`, `String.all` and `String.toNat!`, which
recurse well-founded over UTF-8 positions, so the kernel doesn't evaluate them. -/
def HonestProfileReads : Prop := Core.Integrity.ofJson honestIntegrity.toJson = .ok honestIntegrity

end Properties.NetworkCertifier
