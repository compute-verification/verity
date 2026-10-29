import NetworkCheck.Period
import Lms.Sign

/-!
# The network certifier over a period: what its Lean code (`NetworkCheck.Period`) establishes of each link-window

The property of `NetworkCheck.Period.check`, `NetworkCheck.Period.protocol`'s check on blobs (`verity/certifier/v0`,
onsite's `.network` component, `Verity.component .network`, a `Core.BlobProtocol`), stated against its functions as
they run on the caller's blobs: if
the check accepts, the commitment registers and the evidence answers exactly the period's `m` link-windows, and every
link-window `i` (the session's window `w = first + i`) is accepted as the period's own per-window check of its form
accepts it.

In the clock form (`windowCheck`, `PeriodWindowAccepted`):

* registered: its registration names the link's Program at `w` (`PeriodWindowRegistered`);
* both ends certified: two certificates, egress then ingress, each the key's for its end of the link at `w`, verifying
  under LMS in the session (`PeriodWindowCertified`);
* its clock synced: the key's egress clock certificate of the link at `w`, its digest neither a sync of the link's
  Σ_sync's unsalted `clock_digest` nor an earlier link-window's in the period (`PeriodWindowClocked`);
* proved: one-stage v2 accepted its proof on the blobs (`SampledProofs.OneStageZk.checkBlobs`), under the auditor's
  parameters with the link-window's own port and the period's earlier registrations in its log, with the profile of one
  unit drawn `subset:1` (`PeriodWindowV2`);
* opened: the proof's public row for the drawn unit opens under the delivered salt to the verifier's own, and its output
  row to the word 3 (`PeriodWindowOpened`).

In the rows form (`rowsWindowCheck`, `PeriodRowsWindowAccepted`):

* its registration names the scan's Program (`PeriodWindowProgram`);
* its three records certified: the requests' (the link's ingress), the responses' (its egress) and the fails'
  (`<link>-fails`, egress) at `w`, each the key's and verifying in the session (`PeriodRowsCertified`);
* its rows registered: the ingress then the egress rows, each `warden_rows.check_registration`'s (`PeriodRowsChecked`:
  the link-window's value, link, end and window, the bound schedule's digest and rows, the link-window's domain, and the
  key's certificate on the head), both held by the scan's registration (`PeriodRowsRegistered`);
* proved: the one-stage answer it shows is for its registration's digest and accepted, with one level of the scan's
  `2T` units drawn `subset:k` or stratified and a `bound` (`PeriodRowsProved`).

The per-window predicate is the period check's own, not N2's link-window verdict (`NetworkCheck.LinkWindow.verdict`,
`Properties.NetworkCertifier.LinkWindowAcceptMeans`): the period check never runs that verdict. In the clock form it
runs one-stage v2 on the link-window's proof itself and makes the verdict's opening beside it; in the rows form it reads
the scan's one-stage answer. What the certifier signs about the clock is witness, so the clock is synced as the
auditor's device certified it (`trusted-host`), not by a clear-text sync in Σ_sync.

Comparisons the code makes with a `BEq` instance not known to be lawful (`Json`, `ByteArray`, `String`, the profile's
levels) are stated as the code's `==` returning `true`.

Beside the property, its two lemmas' objects: `PeriodAcceptMeansOf`, the property of any period check, and
`acceptAllPeriod`, a check that accepts every period; and an honest period of one clock-form link-window on a clock link
`l` under N2's Program `prog` (`honestParams`, `honestCommitment`, `honestEvidence`), its certificates signed with the
certifier's key at the deployed LMS parameters, with the named assumption its acceptance takes: that some clock link,
Program, registration, certified digests and proof of that link-window pass v2 and open (`HonestWindowProved`), which
no kernel evaluation makes.
-/

open NetworkCheck.Period
open NetworkCheck.Check (Certificate)
open Lean (Json)

namespace Properties.NetworkCertifier

/-- Link-window `w`'s registration names the period's Program: N2's for the link in the clock form, the scan's in the
rows form. -/
def PeriodWindowProgram (p : Params) (reg : Json) : Prop :=
  (reg.getObjValD "program" == .str p.program) = true

/-- Link-window `w`'s registration names the link's Program and the provenance `{"kind": "served", "window": w}`. -/
def PeriodWindowRegistered (p : Params) (w : Nat) (reg : Json) : Prop :=
  PeriodWindowProgram p reg ∧ canonText (reg.getObjValD "window") = canonText (served w)

/-- Both ends of link-window `w` certified: two certificates, egress then ingress, each the key's for its end of the
link at `w`, its LMS signature verifying over `certificate_message` in the session (`certifies`). -/
def PeriodWindowCertified (p : Params) (w : Nat) (x : Window) : Prop :=
  ∃ eg ig, x.certificates = #[eg, ig] ∧ certifies p .egress w eg = true ∧ certifies p .ingress w ig = true

/-- Link-window `w`'s clock synced as the certifier certified it: it shows a clock certificate, the key's egress
certificate of the link at `w`, and the digest it signs is salted (no sync of the clock link `l`'s Σ_sync's
`clock_digest` itself) and fresh (no earlier link-window's clock digest in the period, `earlier`). -/
def PeriodWindowClocked (p : Params) (l : ClockLink) (earlier : Array ByteArray) (w : Nat) (x : Window) : Prop :=
  ∃ clock salts ev, x.shown = .clock clock salts ev ∧ certifies p .egress w clock = true ∧
    (clearClocks l.syncSet).contains clock.digest = false ∧ earlier.contains clock.digest = false

/-- **One-stage v2 accepted link-window `w`'s proof** on the blobs `bs`: under the auditor's parameters for the clock
link `l` with the link-window's own port, its served domain, and the period's earlier link-windows' registrations `log`
after the auditor's (`windowParams`), v2's check accepts the registration `reg` (under its window record digest) and
the evidence the link-window shows, with a profile of one level of one unit drawn `subset:1`, that one unit drawn. -/
def PeriodWindowV2 (p : Params) (l : ClockLink) (log : Array Json) (w : Nat) (reg : Json) (x : Window)
    (bs : Core.Blobs) : Prop :=
  ∃ clock salts ev, x.shown = .clock clock salts ev ∧
    ∃ dom ps d pub prof, NetworkCheck.LinkWindow.domainId (linkCase p l w .empty #[] #[] none salts) = .ok dom ∧
      windowParams l dom log = .ok ps ∧ Flock.WindowRecord.digest reg = .ok d ∧
      SampledProofs.OneStageZk.checkBlobs ps ⟨reg, d⟩ ev bs = .accept pub prof ∧
      (prof.levels.map (fun lv => (lv.name, lv.size, lv.draw)) ==
        #[(unitLevel, 1, some (Core.SamplingDesign.subset 1))]) = true ∧
      (prof.drawn == #[(unitLevel, 1)]) = true

/-- **Link-window `w`'s proof opens to the verifier's own public row** (`opening`, N2's verdict of record's opening):
the public file its v2 evidence names is a blob, its ``pub`` row commitment for the drawn unit opens under the
delivered salt to the verifier's own public row of the link-window (`NetworkCheck.LinkWindow.pubRow` of `linkCase`: both
certificates' signed digests, then each end's registered root, served domain, link, window, grid and profile row), and
its ``o0`` row commitment to the word 3, verdict and deadline bits 1 (`NetworkCheck.LinkWindow.opened`). -/
def PeriodWindowOpened (p : Params) (l : ClockLink) (w : Nat) (reg : Json) (x : Window) (bs : Core.Blobs) : Prop :=
  ∃ clock salts ev, x.shown = .clock clock salts ev ∧ ∃ file row,
    bs.get ev.pub "public file" = .ok file ∧
    NetworkCheck.LinkWindow.pubRow (linkCase p l w (rootOf reg) x.certificates ev.base.provedUnits (some file) salts) =
      .ok (some row) ∧
    NetworkCheck.LinkWindow.opened
        (linkCase p l w (rootOf reg) x.certificates ev.base.provedUnits (some file) salts) (some row) = some 3

/-- Clock-form link-window `w` of the clock link `l` (registered as `reg`, answered by `x`, after the period's earlier
clock digests `earlier` and registrations `log`, on the blobs `bs`) is accepted: registered as the link's Program at `w`,
both ends certified, its clock synced, its proof accepted by one-stage v2, and opened. -/
def PeriodWindowAccepted (p : Params) (l : ClockLink) (earlier : Array ByteArray) (log : Array Json) (w : Nat)
    (reg : Json) (x : Window) (bs : Core.Blobs) : Prop :=
  PeriodWindowRegistered p w reg ∧ PeriodWindowCertified p w x ∧ PeriodWindowClocked p l earlier w x ∧
    PeriodWindowV2 p l log w reg x bs ∧ PeriodWindowOpened p l w reg x bs

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
def PeriodRowsRegistered (p : Params) (s : Schedule) (sd : ByteArray) (w : Nat) (reg : Json) (x : Window) :
    Prop :=
  ∃ ri re a, x.shown = .rows #[ri, re] a ∧ PeriodRowsChecked p s sd w .ingress ri ∧
    PeriodRowsChecked p s sd w .egress re ∧ holds reg ri = true ∧ holds reg re = true

/-- The rows-form link-window was proved: the one-stage answer it shows is for the registration's window record digest
and accepted, with a public output holding a `bound` and a profile of one level drawn `subset:k` or stratified, of the
scan's `2T` units under the schedule `s` (the ingress and the egress rows, `T` each). -/
def PeriodRowsProved (s : Schedule) (reg : Json) (x : Window) : Prop :=
  ∃ rs a, x.shown = .rows rs a ∧ a.protocol = oneStage ∧
    (∃ d, Flock.WindowRecord.digest reg = .ok d ∧ (a.digest == some d) = true) ∧ a.verdict = .accept ∧
    ∃ pub prof i l b, a.published = some pub ∧ a.profile = some prof ∧
      Core.Integrity.ofJson prof = .ok i ∧ i.levels = #[l] ∧ l.size = 2 * s.window ∧
      ((∃ k, l.draw = some (.subset k)) ∨ ∃ ss, l.draw = some (.stratified ss)) ∧
      Flock.J.nat pub "bound" "public" = .ok b

/-- Rows-form link-window `w` (registered as `reg`, answered by `x`, under the schedule `s` whose digest is `sd`) is
accepted: registered as the scan's Program, its three records certified, its rows registered, and proved. -/
def PeriodRowsWindowAccepted (p : Params) (s : Schedule) (sd : ByteArray) (w : Nat) (reg : Json) (x : Window) :
    Prop :=
  PeriodWindowProgram p reg ∧ PeriodRowsCertified p w x ∧ PeriodRowsRegistered p s sd w reg x ∧
    PeriodRowsProved s reg x

/-- The commitment registers and the evidence answers exactly the period's `p.windows` link-windows, and every one,
the `i`-th being the session's window `p.first + i`, is accepted in the parameters' form on the blobs `bs`: in the clock
form after the clock digests and the registrations of the period's link-windows before it. -/
def PeriodWindowsAccepted (p : Params) (c : Commitment) (e : Evidence) (bs : Core.Blobs) : Prop :=
  c.windows.size = p.windows ∧ e.windows.size = p.windows ∧
    ∀ (i : Nat) (reg : Json) (x : Window), c.windows[i]? = some reg → e.windows[i]? = some x →
      match p.form with
      | .clock l =>
        PeriodWindowAccepted p l ((e.windows.extract 0 i).filterMap (·.clockDigest?)) (c.windows.extract 0 i)
          (p.first + i) reg x bs
      | .rows s sd _ => PeriodRowsWindowAccepted p s sd (p.first + i) reg x

/-- **An accepted period accepted every link-window of it**: if the network certifier's period check accepts on the
caller's blobs, every one of the period's link-windows passed its form's check: in the clock form certified on both ends,
its clock synced, its proof accepted by one-stage v2 on those blobs and opened; in the rows form proved, its three
records certified and its rows registered. -/
def PeriodAcceptMeans : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public) (prof : Profile),
    check p c e bs = .accept pub prof → PeriodWindowsAccepted p c e bs

/-! ## Another period check in the check's place -/

/-- `PeriodAcceptMeans` of a period check `check` in `NetworkCheck.Period.check`'s place: whenever `check` accepts a
period on some blobs, every one of its link-windows was accepted on them. At `NetworkCheck.Period.check` it is
`PeriodAcceptMeans` itself. -/
def PeriodAcceptMeansOf (check : Params → Commitment → Evidence → Core.Blobs → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (pub : Public) (prof : Profile),
    check p c e bs = .accept pub prof → PeriodWindowsAccepted p c e bs

/-- A period check that accepts every period, publishing the terms of a period with no audit: what
`PeriodAcceptMeansOf` must fail at for `PeriodAcceptMeans` to say something. -/
def acceptAllPeriod (p : Params) (_ : Commitment) (_ : Evidence) (_ : Core.Blobs) : Core.Outcome Public Profile :=
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

/-- The period's parameters on the clock link `l` under N2's registered Program `prog`: the link `link`, one clock-form
link-window from window 0, no audit, and the certifier's key at the deployed LMS parameters in its session. -/
def honestParams (l : ClockLink) (prog : String) : Params where
  audits := #[]
  cells := #[]
  first := 0
  jitter := 0
  link := "link"
  lms := Lms.deployed
  key := honestKey.publicKey
  session := honestSession
  program := prog
  form := .clock l
  windows := 1

/-- Link-window 0 as the evidence shows it: the certificates of both ends of the link at window 0 on the digests `de`
(egress) and `di` (ingress) and the clock's, each on its own one-time key, with the delivered salts `s` and the v2
evidence `ev` of its proof. -/
def honestWindow (de di : ByteArray) (s : Salts) (ev : SampledProofs.OneStageZk.Evidence) : Window :=
  { certificates := #[honestCert .egress 0 0 de, honestCert .ingress 0 1 di],
    shown := .clock (honestCert .egress 0 2 ⟨Array.replicate 64 7⟩) s ev }

def honestCommitment (reg : Json) : Commitment := ⟨#[], #[reg], .empty⟩

def honestEvidence (de di : ByteArray) (s : Salts) (ev : SampledProofs.OneStageZk.Evidence) : Evidence :=
  ⟨#[], #[honestWindow de di s ev]⟩

/-- **Named assumption: an honest link-window is proved.** For some clock link under Σ_sync `honestSyncs` (N2's
declared ends and the auditor's one-stage v2 parameters for it) and N2's registered Program `prog` for it, some
registration of `prog` at window 0, the 64-byte digests both ends' certificates sign, and the salts, v2 evidence and
blobs of a proof of link-window 0 under those certificates pass one-stage v2 and open to the verifier's own public row:
N2's completeness on that link-window, which needs a proof v2 accepts, and the kernel runs no prover. -/
def HonestWindowProved : Prop :=
  ∃ (l : ClockLink) (prog : String) (reg : Json) (de di : ByteArray) (s : Salts)
    (ev : SampledProofs.OneStageZk.Evidence) (bs : Core.Blobs),
    l.syncSet = honestSyncs ∧ de.size = 64 ∧ di.size = 64 ∧ PeriodWindowRegistered (honestParams l prog) 0 reg ∧
      PeriodWindowV2 (honestParams l prog) l #[] 0 reg (honestWindow de di s ev) bs ∧
      PeriodWindowOpened (honestParams l prog) l 0 reg (honestWindow de di s ev) bs

end Properties.NetworkCertifier
