import Proofs.Properties.NetworkCertifier.Outcomes
import Proofs.Properties.CertifierNetwork.Certify

/-!
# The completeness theorems' honest cases, instantiated

Each completeness theorem of `Outcomes` (`honestAccepted`, `accepts_of_linkWindowsAccepted`, `undecided_of_somePending`,
`answer_of_decodedAccepted`) has one instance here whose hypotheses hold: a unit of one egress link (`out`) and one
ingress link (`in`) over one window of a one-bucket grid, a toy LMS key (`N = 8`, `W = 8`, `H = 1`, two leaves), and
the two records an honest certifier issues for it, each under a 64-byte salt and signed at its own leaf
(`honestAccepted_instance`); the same unit with no records is undecided (`somePending_instance`).

Two hypotheses stay hypotheses, each of a function no proof can evaluate: the parameters digests (`ParamsDigests`),
since `parametersDigest` reads `Flock.canonUtf8`, well-founded recursion the kernel does not evaluate; and, for `answer`,
that the bytes decode to the unit, since the parameters' and evidence's decoders read canonical JSON through it too.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open _root_.CertifierDevice
open NetworkCertifier (ClockSync)

namespace Properties.NetworkCertifier.Instance

open Properties.CertifierDevice (ValidKey)
open Properties.CertifierDevice.Proofs.Lms (key_spec)
open Properties.CertifierDevice.Proofs.Issuer (verify_cert prepare_spec)
open Properties.CertifierNetwork.Proofs (prepare_of verify_facts parameters_size)

/-! ## A toy LMS key -/

def toyParams : Lms.Params := ⟨8, 8, 1⟩

theorem toyParams_ok : Lms.Params.make 8 8 1 true = .ok toyParams := rfl

def ident : ByteArray := ⟨Array.replicate 16 0⟩

def seed : ByteArray := ⟨Array.replicate 8 0⟩

theorem ident_size : ident.size = 16 := by decide +kernel

/-- `Lms.key` succeeds on every 16-byte identifier. -/
theorem key_ok (p : Lms.Params) {I : ByteArray} (hI : I.size = 16) (seed : ByteArray) :
    ∃ k, Lms.key p I seed = .ok k := by
  unfold Lms.key
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  simp only [hI, ne_eq, not_true_eq_false, ↓reduceIte, pure, Except.pure, bind, Except.bind]
  exact ⟨_, rfl⟩

/-- The key `Lms.key` derives from `toyParams`, `ident` and `seed`. -/
def toyKey : Lms.PrivateKey :=
  match Lms.key toyParams ident seed with
  | .ok k => k
  | .error _ => ⟨toyParams, ident, seed, #[]⟩

theorem toyKey_eq : Lms.key toyParams ident seed = .ok toyKey := by
  obtain ⟨k, hk⟩ := key_ok toyParams ident_size seed
  unfold toyKey
  rw [hk]

theorem toyKey_params : toyKey.params = toyParams := (key_spec toyKey_eq).1

theorem toyKey_valid : ValidKey toyKey :=
  ⟨8, 8, 1, true, ident, seed, by rw [toyKey_params]; exact toyParams_ok, by rw [toyKey_params]; exact toyKey_eq⟩

/-- A certifier session id of 32 zero bytes. -/
def session : ByteArray := ⟨Array.replicate 32 0⟩

theorem session_size : session.size = 32 := by decide +kernel

def issuer : Issuer := { key := toyKey, session }

theorem issuer_I : issuer.key.I.size = 16 := by
  show toyKey.I.size = 16
  rw [(key_spec toyKey_eq).2.1]
  exact ident_size

/-- A certifier salt of 64 zero bytes, the one the honest certifier draws for each record. -/
def salt : ByteArray := ⟨Array.replicate 64 0⟩

theorem salt_size : salt.size = saltBytes := by decide +kernel

/-! ## The unit -/

/-- One bucket of one slot of 9-byte frames. -/
def grid : Grid := ⟨1, 1, 9, 1, 1⟩

def syncSet : NetworkCertifier.SyncSet := ⟨[0], [1], by simp, by simp, by simp, by simp, by simp⟩

def sync : ClockSync := ⟨0, 1, by decide⟩

/-- One egress link and one ingress link, one window, the toy key, the session. -/
def params : NetworkCheck.Check.Params where
  egress := #["out"]
  ingress := #["in"]
  windows := 1
  grid := grid
  ingressGrid := grid
  syncSet := syncSet
  lms := toyKey.params
  key := toyKey.publicKey
  session := session

/-- The egress link opens no frames and declares `sync`. -/
def link : NetworkCheck.Check.Link := ⟨[], #[sync]⟩

def evidence : Evidence := ⟨#[link]⟩

theorem fits : EvidenceFits params evidence :=
  ⟨rfl, by simp [evidence, link, params], by simp [evidence, link]⟩

theorem declares : Declares (W := params.windows) link fun _ => sync := by
  intro ⟨v, hv⟩
  obtain rfl : v = 0 := by simpa [params] using hv
  rfl

theorem programGrid_eq : programGrid params (fun _ => sync) link.frames 0 = #[#[none]] := by decide +kernel

/-- An empty one-bucket record: its one slot is padding. -/
def record (l : String) : Record := ⟨l, 0, #[#[none]], .complete⟩

/-- What a record's head copies of its grid. -/
theorem recordHead_grid {l : String} {d : Direction} {w : Int} {gr : Array (Array (Option ByteArray))}
    {st : Option String} {g : Grid} {par : ByteArray} {dl : Bool} {h : RecordHead}
    (e : recordHead l d w gr st g par dl = .ok h) :
    h.buckets = (g.buckets : Int) ∧ h.allocation = (g.allocation : Int) ∧ h.frameBytes = (g.frameBytes : Int) ∧
      h.bucketMs = (g.bucketMs : Int) := by
  unfold recordHead at e
  cases hF : realFrames gr g with
  | error _ => rw [hF] at e; cases e
  | ok F =>
    cases hr : framesRoot F g with
    | error _ =>
      simp only [hF, hr, bind, Except.bind] at e
      cases e
    | ok rt =>
      simp only [hF, hr, bind, Except.bind, pure, Except.pure] at e
      cases Except.ok.inj e
      exact ⟨rfl, rfl, rfl, rfl⟩

/-- The honest certifier's record of link-window `(d, l, 0)` under parameters digest `pd` and `salt`, signed at leaf
`q`: its head, the head's digest and a signature that verifies under the key over the certificate message of that
digest under `salt`. -/
theorem signed (d : Direction) (l : String) {pd : ByteArray} (hpd : pd.size = 64) (hl : l.toUTF8.size < 2 ^ 16)
    {q : Nat} (hq : q < 2) :
    ∃ (h : RecordHead) (dg σ : ByteArray), recordHead l d 0 #[#[none]] none grid pd true = .ok h ∧
      h.digest = .ok dg ∧
      ∃ m, certificateMessage (params.key.extract 0 16) params.session l d 0 (certificateDigest salt dg) = .ok m ∧
        LmsVerifies params m σ := by
  obtain ⟨h, D, m, hh, hprep⟩ := prepare_of (i := issuer) (r := record l) (d := d) (g := grid) (pd := pd) issuer_I
    session_size salt_size rfl (by
      intro row hrow
      have : row = #[none] := by simpa [record] using hrow
      subst this
      exact ⟨#[], by decide, by simp, by decide +kernel⟩)
    hpd hl (by show (0 : Nat) < 2 ^ 64; decide) (by decide) (by decide) (by decide) (by decide) rfl
  obtain ⟨-, -, -, -, ⟨R, hR, hD⟩, -⟩ := prepare_spec hprep
  obtain ⟨sig, -, hv⟩ := verify_cert toyKey_valid hprep (q := q) (by show q < 2 ^ toyKey.params.H; rw [toyKey_params]; exact hq)
  obtain ⟨hkid, m', hm', s, hs, hv'⟩ := verify_facts hv
  refine ⟨h, R, sig.encode, hh, hR, m', ?_, s, hs, hv'⟩
  have hk : params.key.extract 0 16 = issuer.key.I := hkid.symm
  rw [hk, ← hD]
  exact hm'

/-- The commitment of the two records the honest certifier issues. -/
def honest (he hi : RecordHead) (dge dgi σe σi : ByteArray) : Commitment :=
  ⟨#[honestRecord params .egress "out" 0 he salt dge σe, honestRecord params .ingress "in" 0 hi salt dgi σi],
    ByteArray.empty⟩

theorem syncSet_eq : checkSyncSet syncSet = manifestSet syncSet := by
  simp [checkSyncSet, manifestSet]

/-! ## The instances -/

/-- **`honestAccepted`'s honest case holds**: for parameters digests `eg` and `ig`, the commitment of the two honest
records is accepted with the evidence. -/
theorem honestAccepted_instance {eg ig : ByteArray} (hpd : ParamsDigests params eg ig) :
    ∃ c, protocol.Accepts params c evidence := by
  have heg : eg.size = 64 := parameters_size (g := grid) (ss := some syncSet)
    (by show parametersDigest grid (some (checkSyncSet syncSet)) = .ok eg; rw [syncSet_eq]; exact hpd.1)
  have hig : ig.size = 64 := parameters_size (g := grid) (ss := none) hpd.2
  obtain ⟨he, dge, σe, hhe, hdge, me, hme, hσe⟩ := signed .egress "out" heg (by decide) (q := 0) (by decide)
  obtain ⟨hi, dgi, σi, hhi, hdgi, mi, hmi, hσi⟩ := signed .ingress "in" hig (by decide) (q := 1) (by decide)
  obtain ⟨hel, hed, hew, -⟩ := GridProofs.recordHead_head hhe
  obtain ⟨hil, hid, hiw, his, hidl, hip⟩ := GridProofs.recordHead_head hhi
  obtain ⟨hib, hia, hif, hibm⟩ := recordHead_grid hhi
  have hrece : recordsOf (honest he hi dge dgi σe σi) .egress "out" 0 =
      [honestRecord params .egress "out" 0 he salt dge σe] := by
    simp [recordsOf, honest, honestRecord, hel, hed, hew, hil, hid, hiw]
  have hreci : recordsOf (honest he hi dge dgi σe σi) .ingress "in" 0 =
      [honestRecord params .ingress "in" 0 hi salt dgi σi] := by
    simp [recordsOf, honest, honestRecord, hel, hed, hew, hil, hid, hiw]
  refine ⟨honest he hi dge dgi σe σi, honestAccepted params _ evidence eg ig fits hpd ?_ ?_ ?_⟩
  · intro l hl w hw
    have hl : l = link := by simpa [evidence] using hl
    have hw : w = 0 := by simpa [params] using hw
    subst hl hw
    exact ⟨sync, rfl, by simp [params, syncSet, sync], by simp [params, syncSet, sync], link, sync, rfl, rfl, rfl, rfl⟩
  · intro l lk hm w hw
    have hm : l = "out" ∧ lk = link := by simpa [params, evidence] using hm
    obtain ⟨rfl, rfl⟩ := hm
    have hw : w = 0 := by simpa [params] using hw
    subst hw
    refine ⟨fun _ => sync, declares, he, salt, dge, σe, ?_, hdge, hrece, me, hme, hσe⟩
    rw [programGrid_eq]
    exact hhe
  · intro l hl w hw
    have hl : l = "in" := by simpa [params] using hl
    have hw : w = 0 := by simpa [params] using hw
    subst hl hw
    exact ⟨hi, salt, dgi, σi, hib, hia, hif, hibm, hip, his, hidl, hdgi, hreci, mi, hmi, hσi⟩

/-- **`accepts_of_linkWindowsAccepted`'s hypothesis holds**: for parameters digests `eg` and `ig`, every link-window of
the honest commitment is accepted. -/
theorem linkWindowsAccepted_instance {eg ig : ByteArray} (hpd : ParamsDigests params eg ig) :
    ∃ c, LinkWindowsAccepted params c evidence :=
  let ⟨c, h⟩ := honestAccepted_instance hpd
  ⟨c, acceptsLinkWindows params c evidence h⟩

/-- **`undecided_of_somePending`'s hypothesis holds**: for parameters digests `eg` and `ig`, the unit with no records
has every link-window pending, so it is undecided. -/
theorem somePending_instance {eg ig : ByteArray} (hpd : ParamsDigests params eg ig) :
    SomePending params ⟨#[], ByteArray.empty⟩ evidence ∧
      ∃ why, protocol.check params ⟨#[], ByteArray.empty⟩ evidence = .undecided why := by
  have hnone : ∀ d l w, recordsOf ⟨#[], ByteArray.empty⟩ d l w = [] := fun _ _ _ => rfl
  have h : SomePending params ⟨#[], ByteArray.empty⟩ evidence :=
    ⟨fits, eg, ig, hpd, fun _ _ _ _ _ => Or.inr (Or.inl (hnone _ _ _)), fun _ _ _ _ => Or.inr (Or.inl (hnone _ _ _)),
      .ingress, "in", by simp [linkNames, params], 0, by simp [params], Or.inl (hnone _ _ _)⟩
  exact ⟨h, undecided_of_somePending _ _ _ h⟩

/-- **`answer_of_decodedAccepted`'s hypothesis holds** for bytes that decode to the honest unit: for parameters digests
`eg` and `ig`, there is a commitment such that any bytes decoding to the parameters, it and the evidence are accepted by
`answer`. -/
theorem decodedAccepted_instance {eg ig : ByteArray} (hpd : ParamsDigests params eg ig) :
    ∃ c, ∀ a b b' : ByteArray, (Core.canonicalJson a >>= protocol.params) = .ok params →
      protocol.commitment b = .ok c → protocol.evidence b' = .ok evidence →
        DecodedAccepted a b b' ∧ (protocol.answer a b b').verdict = .accept := by
  obtain ⟨c, h⟩ := honestAccepted_instance hpd
  refine ⟨c, fun a b b' ha hb hb' => ?_⟩
  have hd : DecodedAccepted a b b' := ⟨params, c, evidence, ha, hb, hb', h⟩
  exact ⟨hd, answer_of_decodedAccepted a b b' hd⟩

end Properties.NetworkCertifier.Instance
