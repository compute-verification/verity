import Proofs.Properties.CertifierNetwork.Accepted

/-!
# The completeness statements' honest cases, instantiated

Each completeness statement (`Proofs.Lms.LmsSigns`, `Proofs.Lms.LmsCorrect`, `Proofs.Issuer.IssuerSigns`,
`CertifierNetwork.DeviceOutputAccepted`) has one instance here whose hypotheses all hold: a toy LMS key (`N = 8`,
`W = 8`, `H = 1`, two leaves) from a 16-byte identifier and an 8-byte seed, a one-bucket record under it and a 64-byte
salt, and a one-window configuration with a declared sync whose run ticks each grid once and hands nothing, its every
salt that one. The run's hypotheses are `step` evaluated by the kernel (`decide +kernel`);
the key's are `Lms.key`'s acceptance of a 16-byte identifier (`key_ok`), not its tree, which the kernel need not compute.

`Digested` is the one hypothesis left as one: `parameters` reads `Flock.canon`, well-founded recursion the kernel does
not evaluate, so no proof evaluates it on a configuration (`DeviceOutputAccepted`'s module docstring).
-/

namespace Properties.CertifierNetwork.Instance

open _root_.CertifierDevice
open Properties.CertifierDevice (ValidKey Packed)
open Properties.CertifierDevice.Proofs.Lms (key_ok key_spec lmsSigns)

/-! ## A toy LMS key -/

def toyParams : Lms.Params := ⟨8, 8, 1⟩

theorem toyParams_ok : Lms.Params.make 8 8 1 true = .ok toyParams := rfl

def ident : ByteArray := ⟨Array.replicate 16 0⟩

def seed : ByteArray := ⟨Array.replicate 8 0⟩

theorem ident_size : ident.size = 16 := by decide +kernel

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

/-- **`LmsSigns`'s honest case holds**: toy parameters `Lms.Params.make` admits, the key `Lms.key` derives under them,
and leaf `0 < 2^1`. -/
theorem lmsSigns_instance : ∃ (N W H : Nat) (toy : Bool) (p : Lms.Params) (I s : ByteArray) (k : Lms.PrivateKey)
    (q : Nat), Lms.Params.make N W H toy = .ok p ∧ Lms.key p I s = .ok k ∧ q < 2 ^ H :=
  ⟨8, 8, 1, true, toyParams, ident, seed, toyKey, 0, toyParams_ok, toyKey_eq, by decide⟩

/-- **`LmsCorrect`'s honest case holds**: the toy key signs the empty message at leaf `0`. -/
theorem lmsCorrect_instance : ∃ (N W H : Nat) (toy : Bool) (p : Lms.Params) (I s : ByteArray) (k : Lms.PrivateKey)
    (q : Nat) (message : ByteArray) (sig : Lms.Signature), Lms.Params.make N W H toy = .ok p ∧ Lms.key p I s = .ok k ∧
    Lms.signAt k q message = .ok sig := by
  obtain ⟨sig, hs⟩ := lmsSigns 8 8 1 true toyParams ident seed toyKey 0 .empty toyParams_ok toyKey_eq (by decide)
  exact ⟨8, 8, 1, true, toyParams, ident, seed, toyKey, 0, .empty, sig, toyParams_ok, toyKey_eq, hs⟩

/-! ## A record the issuer certifies -/

def grid : Params := ⟨1, 1, 9, 1, 1⟩

/-- A certifier session id of 32 zero bytes. -/
def session : ByteArray := ⟨Array.replicate 32 0⟩

def issuer : Issuer := { key := toyKey, session }

def record : Record := ⟨"responses", 0, #[#[none]], .complete⟩

def digest : ByteArray := ⟨Array.replicate 64 0⟩

/-- A certifier salt of 64 zero bytes. -/
def salt : ByteArray := ⟨Array.replicate 64 0⟩

theorem salt_size : salt.size = NetworkCheck.saltBytes := by decide +kernel

/-- **`IssuerSigns`'s honest case holds**: the toy key is valid, leaf `0` is below `2^H`, and the one-bucket record has
a head under a 64-byte parameters digest and the 64-byte salt. -/
theorem issuerSigns_instance : ∃ prepared, ValidKey issuer.key ∧ 0 < 2 ^ issuer.key.params.H ∧
    issuer.prepare salt record .egress grid digest true = .ok prepared := by
  obtain ⟨head, dg, m, -, hprep⟩ := Properties.CertifierNetwork.Proofs.prepare_of (i := issuer) (r := record)
    (d := .egress) (g := grid) (pd := digest) (by show toyKey.I.size = 16; rw [(key_spec toyKey_eq).2.1]; exact ident_size)
    (by decide +kernel) salt_size (by decide +kernel) (by
      intro row hrow
      have : row = #[none] := by simpa [record] using hrow
      subst this
      exact ⟨#[], by decide, by simp, by decide +kernel⟩)
    (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) rfl
  exact ⟨(head, dg, m), toyKey_valid, Nat.two_pow_pos _, hprep⟩

/-! ## A run of the device -/

def syncSet : NetworkCertifier.SyncSet := ⟨[0], [1], by simp, by simp, by simp, by simp, by simp⟩

def sync : NetworkCertifier.ClockSync := ⟨0, 1, by decide⟩

/-- One window of one bucket on each grid, its sync declared. -/
def config : Config :=
  { egress := grid, ingress := grid, syncSet, syncs := #[sync], windows := 1, origin := 0, slack := 0 }

/-- Each grid ticks once, on time, the link's backlog empty. -/
def trace : List Event := [.tick .egress 0 0, .tick .ingress 0 0]

theorem ticking : Ticking config := by unfold Ticking; decide +kernel

theorem fieldsFit : FieldsFit config := by unfold FieldsFit; decide +kernel

theorem distinct : Distinct config := by unfold Distinct; decide +kernel

theorem closed : Closed config trace := by unfold Closed; decide +kernel

theorem noFault : NoFault config trace := by unfold NoFault; decide +kernel

theorem syncsDeclared : SyncsDeclared config trace := by unfold SyncsDeclared; decide +kernel

theorem syncsAdmitted : SyncsAdmitted config trace := by unfold SyncsAdmitted; decide +kernel

theorem sessionFits : SessionFits issuer := by unfold SessionFits; decide +kernel

/-- The device's coins give it `salt` for every record. -/
def salts : Nat → ByteArray := fun _ => salt

theorem saltsFit : SaltsFit salts := fun _ => salt_size

theorem leavesLeft : LeavesLeft issuer 0 config trace := by
  unfold LeavesLeft
  show 0 + (published config trace).length ≤ 2 ^ toyKey.params.H
  rw [toyKey_params]
  decide +kernel

/-- **`DeviceOutputAccepted`'s honest case holds** on this configuration, issuer and run, `Digested` aside. -/
theorem deviceOutputAccepted_hypotheses : Ticking config ∧ FieldsFit config ∧ Distinct config ∧
    ValidKey issuer.key ∧ SessionFits issuer ∧ SaltsFit salts ∧ Closed config trace ∧ NoFault config trace ∧
    SyncsDeclared config trace ∧ SyncsAdmitted config trace ∧ LeavesLeft issuer 0 config trace :=
  ⟨ticking, fieldsFit, distinct, toyKey_valid, sessionFits, saltsFit, closed, noFault, syncsDeclared, syncsAdmitted,
    leavesLeft⟩

/-- So, if its grids have a `parameters_digest`, the check accepts this run's output. -/
theorem deviceOutputAccepted_instance (hdg : Digested config) :
    ∃ i' crs, certifyAll config issuer 0 salts (published config trace) = .ok (i', crs) ∧
      ∀ bytes, NetworkCheck.Check.protocol.Accepts (unitParams config issuer.key issuer.session)
        (commitment crs bytes)
        (evidence config trace) :=
  Proofs.deviceOutputAccepted_proof config trace issuer 0 salts ticking fieldsFit distinct hdg toyKey_valid sessionFits
    saltsFit closed noFault syncsDeclared syncsAdmitted leavesLeft

end Properties.CertifierNetwork.Instance
