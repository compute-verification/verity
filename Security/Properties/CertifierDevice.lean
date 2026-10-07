import CertifierDevice.Device
import CertifierDevice.Issuer
import CertifierDevice.Difftest

/-!
# The certifier device's properties, stated against `step` over traces

`CertifierDevice.step` (`verity/experimental/core/devices/certifier/lean/CertifierDevice/Device.lean`, package
`certifier_device`) is `active.py`'s proxy on one boundary link as a pure state machine: one event of the world in, the
state it leaves and the proxy's actions out. The code has no fold of it over a trace, so this module defines one
(`outputs`, `after`, from `init c`); every statement is about it, for every configuration and every trace, whatever its
events' clock readings, order or bytes. `Issuer.certify` (`Issuer.lean`) signs a closed window's record. Each statement
is a named `Prop`, proved as the lower-case theorem of the same name in `Properties.Proofs.CertifierDevice`, and a
guarantee of this package (`lean-audit.json`). No statement takes an assumption about the world: the hypotheses each
takes are in its statement.

**(1) The link's schedule is content-independent.**

* `LinkSchedule`: the lengths of the link writes each event makes are `writeSchedule` of the trace's tick views
  (`tickView`: a tick's grid and, on egress, the link's backlog), nothing else of the trace: an egress tick that runs
  (one of the first `windows · T`) writes one bucket, `r · B` bytes, unless the backlog is over four buckets' bytes,
  when it writes nothing; every other event writes nothing.
* `ContentIndependent`: two traces with the same tick views in the same positions make link writes of the same lengths
  at every event, whatever their link and server bytes, connections and clock readings.

**(2) The records.**

* `Records`: there is a row `ρ k` for each egress bucket `k`, of `r` slots, frames of `B` bytes left-packed and then
  padding (`Packed`), and a status `σ w` for each window, such that egress bucket `k`'s tick writes exactly `ρ k`'s
  frames and its padding to the link (`wire`, under the backlog's exception) and, at the window's last bucket, publishes
  `⟨egressLink, w, ρ (w·T) … ρ (w·T + T − 1), σ w⟩` with the egress grid's parameters: every published egress record's
  grid is the `T` rows its buckets released, each the frames its bucket wrote. No other event writes the link or
  publishes anything but an ingress record.
* `CompleteRows`: a record whose status is complete, of a window with a declared sync, has every row equal to
  `NetworkCertifier.expectedRow` over the frames the shaper handed (their bits, `bitsOf`, as the difftest compares) under
  the syncs `step` declared, provided each frame was handed at the bucket the Program hands it
  (`NetworkCertifier.handingBuckets`). That hypothesis is honest timing, stated over the trace (`handed`): it fails
  where a frame is read after its logical hand bucket's tick, or ready after its logical hand time.

**(3) Fail-closed.** `FailClosed`: once the egress certifier holds a fault `r`, every later egress tick of that window
releases nothing (it writes `r · B` zero bytes), and the window's record has status `notEstablished r` and padding in
every row from the fault on. The certifier keeps the first fault of a window, so `r` is the first fault's reason.

**(4) The issuer.**

* `LmsCorrect`: for a key `Lms.key` derives under parameters `Lms.Params.make` admits, `Lms.signAt` signs every message
  at every leaf below `2^H`, `Lms.verify` accepts the signature under the key's public key, and the signature's encoding
  decodes back to it.
* `IssuerVerifies`: every certificate `Issuer.certify` returns verifies (`CertifiedRecord.verify`) under the key's
  public key; `IssuerSigns`: given a leaf below `2^H` and a record with a head, it returns one.
-/

namespace Properties.CertifierDevice

open _root_.CertifierDevice
open _root_.CertifierDevice.Difftest (bitsOf)
open NetworkCertifier (Reason Status LogicalFrame)

/-! ## A trace -/

/-- `step` folded over a trace from `st`: each event's actions, in order. -/
def outputs : State → List Event → List (List Action)
  | _, [] => []
  | st, ev :: tr => (step st ev).2 :: outputs (step st ev).1 tr

/-- The state `step` leaves after a trace from `st`. -/
def after (st : State) (tr : List Event) : State := tr.foldl (fun st ev => (step st ev).1) st

/-- What the link's schedule may read of an event: a tick's grid and, on egress, the link's backlog; not its clock
reading, and nothing of any other event. -/
def tickView : Event → Option (GridKind × Nat)
  | .tick .egress _ backlog => some (.egress, backlog)
  | .tick .ingress _ _ => some (.ingress, 0)
  | _ => none

/-- The egress grid's horizon, `windows · T`: how many egress ticks run. -/
def horizon (c : Config) : Nat := c.windows * c.egress.buckets

/-- How many egress buckets have run after the tick views `vs`: an egress tick runs while fewer than `horizon c` have. -/
def egressRun (c : Config) (vs : List (Option (GridKind × Nat))) : Nat :=
  vs.foldl (fun k v => if v.map Prod.fst = some .egress ∧ k < horizon c then k + 1 else k) 0

/-- One bucket's bytes on the link, `r · B`. -/
def bucketBytes (p : Params) : Nat := p.allocation * p.frameBytes

/-! ## (1) The link's schedule -/

/-- The lengths of the link writes an event of tick view `v` makes, `k` egress buckets having run before it: one
bucket's bytes at an egress tick that runs, unless the link's backlog is over four buckets' bytes; nothing otherwise. -/
def writesAt (c : Config) (k : Nat) : Option (GridKind × Nat) → List Nat
  | some (.egress, backlog) =>
    if k < horizon c ∧ backlog ≤ 4 * bucketBytes c.egress then [bucketBytes c.egress] else []
  | _ => []

/-- The link writes' lengths, event by event, of a trace with tick views `vs`. -/
def writeSchedule (c : Config) (vs : List (Option (GridKind × Nat))) : List (List Nat) :=
  vs.mapIdx fun n v => writesAt c (egressRun c (vs.take n)) v

/-- The lengths of an event's link writes, in order. -/
def linkWrites (acts : List Action) : List Nat :=
  acts.filterMap fun | .writeLink b => some b.size | _ => none

/-- **The schedule.** Every event's link writes have the lengths `writeSchedule` gives from the trace's tick views. -/
def LinkSchedule : Prop :=
  ∀ (c : Config) (tr : List Event), (outputs (init c) tr).map linkWrites = writeSchedule c (tr.map tickView)

/-- **Content independence.** Traces with the same tick views in the same positions write the link in the same lengths
at every event. -/
def ContentIndependent : Prop :=
  ∀ (c : Config) (tr tr' : List Event), tr.map tickView = tr'.map tickView →
    (outputs (init c) tr).map linkWrites = (outputs (init c) tr').map linkWrites

/-! ## (2) The records -/

/-- A row as the certifier releases it: `r` slots, `n ≤ r` frames of `B` bytes in queue order, then padding. -/
def Packed (p : Params) (row : Slots) : Prop :=
  ∃ fs : Array ByteArray, fs.size ≤ p.allocation ∧ (∀ f ∈ fs, f.size = p.frameBytes) ∧
    row = fs.map some ++ Array.replicate (p.allocation - fs.size) none

/-- What a bucket of row `row` puts on the link: its frames in slot order, then `r − n` frames of zeros. -/
def wire (p : Params) (row : Slots) : ByteArray :=
  (row.filterMap id).foldl (· ++ ·) ByteArray.empty ++
    ⟨Array.replicate ((p.allocation - rowCount row) * p.frameBytes) 0⟩

/-- Event `n` of `tr` is an egress tick that runs, egress bucket `k`, read with the link's backlog `backlog`. -/
def EgressAt (c : Config) (tr : List Event) (n k backlog : Nat) : Prop :=
  (∃ now, tr[n]? = some (.tick .egress now backlog)) ∧ k = egressRun c ((tr.take n).map tickView) ∧ k < horizon c

/-- What egress bucket `k` does when the buckets release rows `ρ` and windows close with statuses `σ`: it writes its
row (unless the backlog is over four buckets' bytes) and, at its window's last bucket, publishes the window's record. -/
def egressActions (c : Config) (ρ : Nat → Slots) (σ : Nat → Status) (k backlog : Nat) : List Action :=
  let T := c.egress.buckets
  (if backlog ≤ 4 * bucketBytes c.egress then [.writeLink (wire c.egress (ρ k))] else []) ++
    if (k + 1) % T = 0 then
      [.publish ⟨c.egressLink, k / T, Array.ofFn fun j : Fin T => ρ (k / T * T + j), σ (k / T)⟩ c.egress]
    else []

/-- **The records are the rows written.** Some packed row per egress bucket and status per window make every egress
tick's actions exactly `egressActions`; no other event writes the link or publishes but an ingress record. -/
def Records : Prop :=
  ∀ (c : Config) (tr : List Event), ∃ (ρ : Nat → Slots) (σ : Nat → Status),
    (∀ k, Packed c.egress (ρ k)) ∧
    (∀ n k backlog, EgressAt c tr n k backlog → (outputs (init c) tr)[n]? = some (egressActions c ρ σ k backlog)) ∧
    ∀ n acts, (outputs (init c) tr)[n]? = some acts → (¬ ∃ k backlog, EgressAt c tr n k backlog) →
      ∀ a ∈ acts, (∀ b, a ≠ .writeLink b) ∧ ∀ r p, a = .publish r p → r.link = c.ingressLink ∧ p = c.ingress

/-- The frames the shaper hands its certifier at event `ev` from `st`, in order: the canonical positions its handing
passes, as it held them. -/
def handedAt (st : State) (ev : Event) : List Frame :=
  (List.range' st.shaper.next ((step st ev).1.shaper.next - st.shaper.next)).filterMap fun k =>
    (st.shaper.pending.get? k).map (·.1)

/-- The frames the shaper hands over a trace from `st`, in order, each with the egress bucket that hands it. -/
def handed : State → List Event → List (Frame × Nat)
  | _, [] => []
  | st, ev :: tr => (handedAt st ev).map (·, st.shaper.t) ++ handed (step st ev).1 tr

/-- **A complete record is the Program's.** If each frame the shaper hands over the trace is handed at the bucket
`NetworkCertifier.handingBuckets` gives it, under the syncs `step` declared, then every published egress record whose
status is complete, of a window with a declared sync, has as row `j`, slot by slot, `NetworkCertifier.expectedRow` at
bucket `w·T + j` of the handed frames (their bits, as the difftest's honest check reads them). -/
def CompleteRows : Prop :=
  ∀ (c : Config) (tr : List Event),
    let hs := handed (init c) tr
    let syncs := (after (init c) tr).shaper.syncs
    let frames : List LogicalFrame := hs.map fun x => ⟨bitsOf x.1.bytes, x.1.anchor, x.1.index⟩
    let T := c.egress.buckets
    hs.map Prod.snd = NetworkCertifier.handingBuckets T syncs.size (fun w => syncs[w]) frames →
    ∀ n k backlog acts rec p, EgressAt c tr n k backlog → (outputs (init c) tr)[n]? = some acts →
      .publish rec p ∈ acts → rec.status = .complete → rec.window < syncs.size →
      ∀ j < T, ∀ (s : Nat) (hs : s < c.egress.allocation),
        NetworkCertifier.expectedRow c.egress.allocation T syncs.size (fun w => syncs[w]) frames (rec.window * T + j)
          ⟨s, hs⟩ = (rec.grid[j]!)[s]!.map bitsOf

/-! ## (3) Fail-closed -/

/-- **Fail-closed.** If after the first `n` events the egress certifier holds fault `r`, every later egress tick of the
same window writes only padding and, at the window's last bucket, publishes the record with status `notEstablished r`,
its rows from the fault's bucket on padding. -/
def FailClosed : Prop :=
  ∀ (c : Config) (tr : List Event) (n : Nat) (r : Reason),
    let st := after (init c) (tr.take n)
    let T := c.egress.buckets
    st.shaper.certifier.fault = some r →
    ∀ m k backlog, n ≤ m → EgressAt c tr m k backlog → k / T = st.shaper.t / T →
      ∃ grid : Array Slots, (∀ j, st.shaper.t % T ≤ j → j < grid.size →
          grid[j]! = Array.replicate c.egress.allocation none) ∧
        (outputs (init c) tr)[m]? = some
          ((if backlog ≤ 4 * bucketBytes c.egress then
              [.writeLink ⟨Array.replicate (bucketBytes c.egress) 0⟩] else []) ++
            if (k + 1) % T = 0 then [.publish ⟨c.egressLink, k / T, grid, .notEstablished r⟩ c.egress] else [])

/-! ## (4) The issuer -/

/-- **LMS signatures verify.** For a key `Lms.key` derives under parameters `Lms.Params.make` admits, every leaf below
`2^H` signs every message, `Lms.verify` accepts the signature under the key's public key, and its encoding decodes back
to it. -/
def LmsCorrect : Prop :=
  ∀ (N W H : Nat) (toy : Bool) (p : Lms.Params) (I seed : ByteArray) (k : Lms.PrivateKey) (q : Nat)
    (message : ByteArray),
    Lms.Params.make N W H toy = .ok p → Lms.key p I seed = .ok k → q < 2 ^ H →
    ∃ sig, Lms.signAt k q message = .ok sig ∧ Lms.verify p k.publicKey message sig = .ok true ∧
      Lms.Signature.decode p sig.encode = .ok sig

/-- An issuer's key: one `Lms.key` derives under parameters `Lms.Params.make` admits. -/
def ValidKey (k : Lms.PrivateKey) : Prop :=
  ∃ (N W H : Nat) (toy : Bool) (I seed : ByteArray), Lms.Params.make N W H toy = .ok k.params ∧
    Lms.key k.params I seed = .ok k

/-- **Every certificate the issuer returns verifies** under its key's public key. -/
def IssuerVerifies : Prop :=
  ∀ (i i' : Issuer) (leaf : Except String Nat) (record : Record) (direction : NetworkCheck.Direction)
    (params : Params) (parameters : ByteArray) (deadline : Bool) (cr : CertifiedRecord),
    ValidKey i.key → i.certify leaf record direction params parameters deadline = .ok (i', some cr) →
    cr.verify i.key.params i.key.publicKey = true

/-- **The issuer certifies at every leaf.** Given a leaf below `2^H` and a record with a head (`Issuer.prepare`
succeeds), `Issuer.certify` returns a certificate, which verifies. -/
def IssuerSigns : Prop :=
  ∀ (i : Issuer) (q : Nat) (record : Record) (direction : NetworkCheck.Direction) (params : Params)
    (parameters : ByteArray) (deadline : Bool) (prepared : NetworkCheck.RecordHead × ByteArray × ByteArray),
    ValidKey i.key → q < 2 ^ i.key.params.H →
    i.prepare record direction params parameters deadline = .ok prepared →
    ∃ cr, i.certify (.ok q) record direction params parameters deadline = .ok (i, some cr) ∧
      cr.verify i.key.params i.key.publicKey = true

end Properties.CertifierDevice
