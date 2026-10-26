import NetworkCheck.LinkWindow

/-!
# N2's link-window verdict of record: what its Lean code (`NetworkCheck.LinkWindow`) does

The property of `NetworkCheck.LinkWindow.verdict`, the verdict of record on one link-window of N2 (`service.OF_RECORD`
in `verity/core/protocols/certifier/service.py`), stated against its functions as they run: for every case, if the
verdict accepts, the link-window's unit was proved, its rows opened to an output word of 3, both certificates are the
certifier's, its declared sync is one of Σ_sync, and its public outputs are verdict 1 and deadline 1.
`Proofs.Properties.NetworkCertifier.LinkWindow` proves it.

The predicates below say what the findings, the public file's row commitments, the certificates and the sync are, in the
verdict's own terms. The public row (`pubRow`), the commitments' parse (`commitments`), the LMS verifier and the
certificate's message are the code's own functions, read as they are.
-/

open NetworkCheck NetworkCheck.LinkWindow
open Lean (Json toJson)

namespace Properties.NetworkCertifier

/-- The verifier's own run of C-Flock's verifier proved the link's unit and the draw: every gate of the unit names a
pinned circuit digest, and it found one or more sessions, each accepted, naming the registered Program and proving
exactly the drawn units. -/
def WindowProved (c : Case) : Prop :=
  (∀ g ∈ c.gates, pinned g = true) ∧ ∃ fs, c.findings = some fs ∧ fs ≠ #[] ∧
    ∀ f ∈ fs, f.accepted = true ∧ f.program = some c.program ∧ f.units = c.units

/-- `bc` is the hm96-sha512 commitment, under its default key, to the `bits`-bit row `row` (its `sha512/row/v2` digest)
under the delivered salt, which is 192 bytes. -/
def CommitsRow (row : ByteArray) (bits : Nat) (salt : Option ByteArray) (bc : ByteArray) : Prop :=
  ∃ s, salt = some s ∧ s.size = LinkWindow.saltBytes ∧
    Flock.Hm96.Default512.commitString (rowDigest row bits) s = bc

/-- The delivered output word is 3 (verdict and deadline bits both 1), and the statement's public file names the drawn
unit once and holds its row commitments (``pub``, ``wit``, ``o0``): the ``pub`` commitment is of the verifier's own
public row of the link-window, and the ``o0`` commitment of the word. -/
def WindowOpened (c : Case) : Prop :=
  c.output = some 3 ∧ ∃ row file u bcPub bcWit bcOut, pubRow c = .ok (some row) ∧ c.statement = some file ∧
    c.units[0]? = some u ∧ commitments file u = some [bcPub, bcWit, bcOut] ∧
    CommitsRow row (8 * row.size) c.saltPub bcPub ∧ CommitsRow (le 2 3) outBits c.saltOut bcOut

/-- Certificate `x` is the certifier's for end `d` (its code: egress 0, ingress 1) of the link-window: it names the
link, the end and the window, carries the key's identifier, and its signature decodes and verifies under the key over
`certificateMessage` of its fields in the certifier session. -/
def WindowCertificate (c : Case) (d : Nat) (x : Cert) : Prop :=
  x.link = c.link ∧ x.direction.code = d ∧ x.window = c.window ∧ x.keyId = c.key.extract 0 16 ∧
    ∃ m sig, certificateMessage x.keyId c.session x.link x.direction x.window x.digest = .ok m ∧
      Lms.Signature.decode c.lms x.signature = .ok sig ∧ Lms.verify c.lms c.key m sig = .ok true

/-- There are two certificates, egress first, each the certifier's for its end of the link-window. -/
def WindowCertified (c : Case) : Prop :=
  c.certificates.size = 2 ∧ ∀ d < 2, ∃ x, c.certificates[d]? = some (some x) ∧ WindowCertificate c d x

/-- The window's declared sync is one of Σ_sync: its δ one of the deltas and its ρ equal to one of the rhos. -/
def WindowSynced (c : Case) : Prop :=
  ∃ s, c.syncs[c.window]? = some s ∧ s.delta ∈ c.deltas ∧ ∃ r ∈ c.rhos, s.num * r.2 = r.1 * s.den

/-- An accepted link-window's public outputs: `verdict` 1 and `deadline` 1, and on an advised link the advice it read. -/
def acceptedOutputs (c : Case) : Json :=
  Json.mkObj ([("verdict", toJson (1 : Nat)), ("deadline", toJson (1 : Nat))] ++
    match c.kind with
    | .advised _ advice => [("advice", match advice with | some a => .str a | none => .null)]
    | _ => [])

/-- The link-window is proved, opened, certified and synced. -/
def WindowAccepted (c : Case) : Prop :=
  WindowProved c ∧ WindowOpened c ∧ WindowCertified c ∧ WindowSynced c

/-- The verdict's public outputs are `acceptedOutputs`, and its digest binds them, the registration, the sessions and
the accept bit (`verdictDigest`). -/
def WindowVerdict (c : Case) : Prop :=
  ∃ v, verdict c = .ok v ∧ v.outputs = acceptedOutputs c ∧ verdictDigest c true v.outputs = .ok v.digest

/-- **An accepted link-window is proved, opened to the word 3, certified on both ends and synced within Σ_sync, and its
public outputs are verdict 1 and deadline 1.** -/
def LinkWindowAcceptMeans : Prop :=
  ∀ c : Case, (verdict c).map Verdict.accepted = .ok true → WindowAccepted c ∧ WindowVerdict c

end Properties.NetworkCertifier
