import CertifierDevice.Device
import CertifierDevice.Issuer

/-!
# The certifier device's properties, stated against `step` over traces

`CertifierDevice.step` (`verity/core/devices/certifier/lean/CertifierDevice/Device.lean`, package
`certifier_device`) is `active.py`'s proxy on one boundary link as a pure state machine: one event of the world in, the
state it leaves and the proxy's actions out. The code has no fold of it over a trace, so this module defines one
(`outputs`, `after`, from `init c`); every statement is about it, for every configuration and every trace, whatever its
events' clock readings, order or bytes. `Issuer.certify` (`Issuer.lean`) signs a closed window's record. Each statement
is a named `Prop`, proved as the lower-case theorem of the same name in `Proofs.Properties.CertifierDevice`, and a
guarantee of this package (`lean-audit.json`). Each has one shape: for every input, if the code accepts it, the
conclusion holds. `step` always returns, so the trace statements take no hypothesis; the issuer's takes the acceptances
of the calls that made the key and of the one they are about. No statement takes an assumption about the world.

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
* `CompleteRows`: a record an egress tick publishes whose status is complete has every row equal to the FIFO's release
  at the allocation (`NetworkCertifier.fifoGrid`) of the frames the shaper handed through that tick, each enqueued at
  the bucket that handed it (`handed`) and leaving in the order handed (their bits, `bitsOf`, as the difftest compares)
  (`ProgramRows`). It reads no sync and no `NetworkCertifier.expectedRow`: the buckets are the ones the device handed
  the frames at. That each was handed at the bucket the Program gives it is a separate fact of a run whose windows are
  all complete (`Proofs.CertifierDevice.hands_honest`, and `Proofs.CertifierDevice.completeRows_honest` for the
  Program's rows).

**(3) Fail-closed.** `FailClosed`: once the egress certifier holds a fault `r`, every later egress tick of that window
releases nothing (it writes `r · B` zero bytes), and the window's record has status `notEstablished r` and padding in
every row from the fault on. The certifier keeps the first fault of a window, so `r` is the first fault's reason.

**(4) The issuer.** `IssuerVerifies`: for every key `Lms.key` derives under parameters `Lms.Params.make` admits,
every certificate `Issuer.certify` returns with it, in a certifier session and under any salt, verifies (`CertifiedRecord.verify`) under the
key's parameters and public key in that session. Its subject is such a key: the two calls' acceptances are its
hypotheses, beside `certify`'s.

That `signAt` and `certify` do return, at every leaf below `2^H` and for every record with a head, and that every
signature `signAt` returns verifies, are completeness, not guarantees: `Proofs.CertifierDevice.Lms.lmsSigns`,
`Proofs.CertifierDevice.Lms.lmsCorrect` and `Proofs.CertifierDevice.Issuer.issuerSigns`, each with a toy-LMS instance in
`Proofs.CertifierNetwork.Instance`.
-/

namespace Properties.CertifierDevice

open _root_.CertifierDevice
open NetworkCertifier (Reason Status)

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

/-- If `tr` and `tr'` have the same tick views in the same positions, they write the link in the same lengths at every
event. -/
def SameWrites (c : Config) (tr tr' : List Event) : Prop :=
  tr.map tickView = tr'.map tickView → (outputs (init c) tr).map linkWrites = (outputs (init c) tr').map linkWrites

/-- **Content independence**, for every pair of traces. -/
def ContentIndependent : Prop := ∀ (c : Config) (tr tr' : List Event), SameWrites c tr tr'

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

/-- **The records of `tr` are the rows written.** Some packed row per egress bucket and status per window make every
egress tick's actions exactly `egressActions`; no other event writes the link or publishes but an ingress record. -/
def RowsWritten (c : Config) (tr : List Event) : Prop :=
  ∃ (ρ : Nat → Slots) (σ : Nat → Status),
    (∀ k, Packed c.egress (ρ k)) ∧
    (∀ n k backlog, EgressAt c tr n k backlog → (outputs (init c) tr)[n]? = some (egressActions c ρ σ k backlog)) ∧
    ∀ n acts, (outputs (init c) tr)[n]? = some acts → (¬ ∃ k backlog, EgressAt c tr n k backlog) →
      ∀ a ∈ acts, (∀ b, a ≠ .writeLink b) ∧ ∀ r p, a = .publish r p → r.link = c.ingressLink ∧ p = c.ingress

/-- **The records are the rows written**, on every trace. -/
def Records : Prop := ∀ (c : Config) (tr : List Event), RowsWritten c tr

/-- The frames the shaper hands its certifier at event `ev` from `st`, in order: the canonical positions its handing
passes, as it held them. -/
def handedAt (st : State) (ev : Event) : List Frame :=
  (List.range' st.shaper.next ((step st ev).1.shaper.next - st.shaper.next)).filterMap fun k =>
    (st.shaper.pending.get? k).map (·.1)

/-- The frames the shaper hands over a trace from `st`, in order, each with the egress bucket that hands it. -/
def handed : State → List Event → List (Frame × Nat)
  | _, [] => []
  | st, ev :: tr => (handedAt st ev).map (·, st.shaper.t) ++ handed (step st ev).1 tr

/-- The bits of the `i`-th frame of `hs`, in the order handed, and none past the last. -/
def handedBits (hs : List (Frame × Nat)) (i : Nat) : List Bool := (hs.map fun x => bitsOf x.1.bytes).getD i []

/-- How many frames of `hs` bucket `u` handed. -/
def handedCount (hs : List (Frame × Nat)) (u : Nat) : Nat := (hs.map Prod.snd).count u

/-- **The complete records of `tr` are the FIFO's release of what was handed.** Every record an egress tick of `tr`
publishes, event `n`, whose status is complete, has as row `j`, slot by slot, the FIFO certifier's release at the
allocation (`NetworkCertifier.fifoGrid`) at bucket `w·T + j`, when the frames the shaper handed through event `n` are
enqueued each at the bucket that handed it and leave in the order handed (their bits, as the difftest compares them).

The count is of every frame handed, in a faulted window too, where the certifier enqueued nothing: a window that closes
faulted with frames still counted faults the next one, so a complete window starts from what the count holds. -/
def ProgramRows (c : Config) (tr : List Event) : Prop :=
  ∀ n k backlog acts rec p, EgressAt c tr n k backlog → (outputs (init c) tr)[n]? = some acts → .publish rec p ∈ acts →
    let hs := handed (init c) (tr.take (n + 1))
    let T := c.egress.buckets
    rec.status = .complete →
    ∀ j < T, ∀ (s : Nat) (hs' : s < c.egress.allocation),
      NetworkCertifier.fifoGrid (rec.window * T + j + 1) c.egress.allocation (handedBits hs) (handedCount hs)
        (Fin.last (rec.window * T + j)) ⟨s, hs'⟩ = (rec.grid[j]!)[s]!.map bitsOf

/-- **A complete record is the FIFO's release of what was handed**, on every trace. -/
def CompleteRows : Prop := ∀ (c : Config) (tr : List Event), ProgramRows c tr

/-! ## (3) Fail-closed -/

/-- **`tr` fails closed.** If after its first `n` events the egress certifier holds fault `r`, every later egress tick
of the same window writes only padding and, at the window's last bucket, publishes the record with status
`notEstablished r`, its rows from the fault's bucket on padding. -/
def ClosedAfterFault (c : Config) (tr : List Event) : Prop :=
  ∀ (n : Nat) (r : Reason),
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

/-- **Fail-closed**, on every trace. -/
def FailClosed : Prop := ∀ (c : Config) (tr : List Event), ClosedAfterFault c tr

/-! ## (4) The issuer -/

/-- **Every certificate the issuer returns verifies.** For a key `Lms.key` derives under parameters `Lms.Params.make`
admits, every certificate `Issuer.certify` returns with that key in certifier session `session`, under any salt,
verifies (`CertifiedRecord.verify`) under the parameters and the key's public key in that session. -/
def IssuerVerifies : Prop :=
  ∀ (N W H : Nat) (toy : Bool) (p : Lms.Params) (I seed : ByteArray) (k : Lms.PrivateKey) (session : ByteArray)
    (refused : Array (String × NetworkCheck.Direction × Nat)) (i' : Issuer) (salt : ByteArray) (leaf : Except String Nat)
    (record : Record) (direction : NetworkCheck.Direction) (params : Params) (parameters : ByteArray) (deadline : Bool)
    (cr : CertifiedRecord),
    Lms.Params.make N W H toy = .ok p → Lms.key p I seed = .ok k →
    ({ key := k, session, refused } : Issuer).certify salt leaf record direction params parameters deadline =
      .ok (i', some cr) →
    cr.verify p k.publicKey session = true

end Properties.CertifierDevice
