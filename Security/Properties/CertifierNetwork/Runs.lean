import Properties.CertifierNetwork
import Definitions.NetworkCertifier
import Mathlib.Data.Set.Card

/-!
# What an accepted run of the certifier device is

The network certificate check (`NetworkCheck.Check.protocol`) meets the unit's certifier device through what the device
returns: the records `CertifierDevice.step` publishes over the trace the world gives it, and the certificates the host's
assembly has its issuer sign for them. Each property here is stated over those outputs, for every input of the device and
every commitment and evidence the check accepts under the device's parameters (`unitParams`), given the one premise of
the network lane, that the device runs as built (`NetTiming.DeviceIntact`, the listed `trusted-nebius-host`). Everything
else the old abstract device was assumed to do (each head its link-window's, each egress head `record_head` of its
recorded grid) is now a fact about the device's code.

**The unit is one link pair**, the device's, as in `Properties.CertifierNetwork`.

Nothing is assumed of SHA-512 or LMS. Each claim about an accepted run holds, or the run's own data carries a break:
two distinct strings of `runStrings`, the strings the check and the device hash, with one SHA-512 digest
(`cr/sha-512`), or a certificate of the commitment whose signature verifies under the device's key over its message in
this run's certifier session (`certMessage x.issuer.session`) though the device never signed that message in this run
(`Signed`, `euf-cma/lms-sha512-n64`).

**Other runs of the key.** The session id is the challenger's draw, and the check's parameters name it, so the
forgery disjunct names a message of this run alone. Reading it as a forgery of the key also needs that no other run of
the key signed that message: (a) the ids of two sessions differ, by the challenger's draw; (b) the device runs once per
session id: it signs nothing when its signing state has already started its issuer's session (`DeviceInput.sessions`,
`deviceCode`); (c) no other run's message input has the SHA-512 digest of this run's. (c) is `cr/sha-512` over the
key's whole signing history, the hash-then-sign composition behind `euf-cma/lms-sha512-n64`, not a break tied to the
run: only a model of the key's runs would make it one. (b), and the one-time use of each leaf that the forgery reading
already rests on (`DeviceInput.next` is the host's), rest on one presumption: the device's signing state (next leaf,
used session ids) is not rolled back, the persistent-state part of `trusted-nebius-host`, whose part about one run is
`NetTiming.DeviceIntact`. No statement here takes it as a premise, since a statement of one run can't name other runs.

`Properties.Proofs.CertifierNetwork.Runs` proves `AcceptedRowsAreProgramRows`; `Proofs.NetworkCertifier.Accepted` (in
`security_proofs`, through the reference's `Proofs.NetworkCertifier.Code`) proves `AcceptedRunsAreProgramRuns` and the
count it gives, `EncardAcceptedObservationsLe`.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open _root_.CertifierDevice
open NetworkCertifier (ClockSync)
open Properties.CertifierNetwork (published unitParams directionOf gridOf syncSetOf reserve advance certified)

namespace Properties.NetworkCertifier

/-! ## The device -/

/-- What the certifier device is given in one run: its configuration, its issuer (its key and certifier session), the
signer's state (its next unused leaf and the session ids it has started) and the trace of events the world gives it. -/
structure DeviceInput where
  config : Config
  issuer : Issuer
  next : Nat
  sessions : List ByteArray
  trace : List Event

/-- What the device returns: the records it publishes and the certificates it signs. -/
structure DeviceOutput where
  records : List Record
  certificates : List CertifiedRecord

/-- The certificates the host's assembly (`Properties.CertifierNetwork.certifyAll`) has the issuer sign, in order: each
record certified at the reserved leaf under its link's direction, grid and parameters digest; where the assembly stops
on an error, the ones signed before it. -/
def issued (c : Config) : Issuer → Nat → List Record → List CertifiedRecord
  | _, _, [] => []
  | i, next, r :: rs =>
    match parameters (gridOf c r) (syncSetOf c r) >>=
        fun pd => i.certify (reserve i next) r (directionOf c r) (gridOf c r) pd with
    | .ok (i', cr?) => cr?.toList ++ issued c i' (advance i next) rs
    | .error _ => []

/-- **The device's code**: the records `step` publishes over the trace, and the certificates of them its issuer signs;
none where the signer's state has started the issuer's session already (it is in `x.sessions`), since the state refuses
to start it again (`Core.World.start`). -/
def deviceCode (x : DeviceInput) : DeviceOutput :=
  ⟨published x.config x.trace, if x.sessions.any (fun s => decide (s = x.issuer.session)) then []
    else issued x.config x.issuer x.next (published x.config x.trace)⟩

/-- The check's parameters of the device's unit: its configuration's links, windows, grids and Σ_sync, its key and its
issuer's certifier session. -/
def deviceParams (x : DeviceInput) : NetworkCheck.Check.Params := unitParams x.config x.issuer.key x.issuer.session

/-- The egress grid's parameters digest under Σ_sync, the one the check recomputes heads under (empty where it has
none; an accepted unit has one). -/
def egressDigest (p : NetworkCheck.Check.Params) : ByteArray :=
  (parametersDigest p.grid (some (manifestSet p.syncSet))).toOption.getD ByteArray.empty

/-! ## The breaks -/

/-- What the device hashes: each certificate's head and its message's input in certifier session `session`, and the
frame tree of each record it published, under its grid. -/
def deviceStrings (c : Config) (session : ByteArray) (out : DeviceOutput) : List ByteArray :=
  out.certificates.flatMap (fun cr => headStrings cr.head ++
    (messageInput cr.certificate.keyId session cr.certificate.link cr.certificate.direction cr.certificate.window
      cr.certificate.digest).toOption.toList) ++
  out.records.flatMap fun r => treeStrings (gridOf c r) r.grid

/-- The strings the run hashes: the check's (`checkStrings`) and the device's (`deviceStrings`). -/
def runStrings (x : DeviceInput) (out : DeviceOutput) (c : Commitment) (e : Evidence) : List ByteArray :=
  checkStrings (deviceParams x) c e (egressDigest (deviceParams x)) ++ deviceStrings x.config x.issuer.session out

/-- The messages the device signs in certifier session `session`: each of its certificates' own message in it. -/
def Signed (session : ByteArray) (out : DeviceOutput) (m : ByteArray) : Prop :=
  ∃ cr ∈ out.certificates, certMessage session (certified cr) = some m

/-! ## (5a) An accepted run's records -/

/-- The device's records of the `W` windows are the Program's: on each of its two links it published a record of every
window `w < W`; every record it published on them for a window `w < W` is complete; and every one on its egress link is
the Program's grid of its window (`programGrid`) of the frames `e` opens on that link under the syncs `e` declares. -/
def ProgramRecords (x : DeviceInput) (out : DeviceOutput) (e : Evidence) : Prop :=
  (∀ l ∈ [x.config.egressLink, x.config.ingressLink], ∀ w < x.config.windows,
    ∃ r ∈ out.records, r.link = l ∧ r.window = w) ∧
  (∀ r ∈ out.records, r.link ∈ [x.config.egressLink, x.config.ingressLink] → r.window < x.config.windows →
    r.status = .complete) ∧
  ∀ link, e.links[0]? = some link → ∀ syncs : Fin x.config.windows → ClockSync, Declares link syncs →
    ∀ r ∈ out.records, r.link = x.config.egressLink → r.window < x.config.windows →
      r.grid = programGrid (deviceParams x) syncs link.frames r.window

/-- **An accepted run's records are the Program's (5a).** For every input of the device and every commitment and
evidence the check accepts under the device's parameters, if the device runs as built, its records of the `W` windows
are the Program's (`ProgramRecords`); or two distinct strings the run hashes have one SHA-512 digest, or a certificate of
the commitment verifies under the device's key over its message in this run's session, which the device never signed
in this run (the module docstring reads it across the key's runs). -/
def AcceptedRowsAreProgramRows : Prop :=
  ∀ (device : DeviceInput → DeviceOutput) (x : DeviceInput) (c : Commitment) (e : Evidence),
    protocol.Accepts (deviceParams x) c e → NetTiming.DeviceIntact device deviceCode →
    ProgramRecords x (device x) e ∨
    (∃ y ∈ runStrings x (device x) c e, ∃ z ∈ runStrings x (device x) c e, y ≠ z ∧
      Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
    ∃ r ∈ c.records.toList, ∃ m, certMessage x.issuer.session r = some m ∧
      LmsVerifies (deviceParams x) m r.certificate.signature ∧ ¬ Signed x.issuer.session (device x) m

/-! ## (5b) An accepted run is a run of the certifier program -/

/-- Σ_sync as the reference's sync set: its lists as `Finset`s. -/
def refSyncSet (ss : NetworkCertifier.SyncSet) : NetTiming.SyncSet where
  deltas := ss.deltas.toFinset
  rhos := ss.rhos.toFinset
  deltas_nonempty := List.toFinset_nonempty_iff _ |>.mpr ss.deltas_nonempty
  rhos_nonempty := List.toFinset_nonempty_iff _ |>.mpr ss.rhos_nonempty
  rhos_pos ρ h := ss.rhos_pos ρ (List.mem_toFinset.mp h)

/-- `o` is read off the device's records: at each bucket `t`, the egress row is the device's (row `t % T` of its record
of window `t / T` on its egress link, each slot's bytes as bits), and boundary link `κ`'s status is complete exactly
where the device's record of `κ`'s link and window `t / T` is. -/
def Observes (cfg : Config) (out : DeviceOutput) {K : Nat} (linkOf : Fin K → String)
    (o : List (NetTiming.Obs 1 K cfg.egress.allocation)) : Prop :=
  (∀ t (ht : t < o.length), ∀ r ∈ out.records, r.link = cfg.egressLink → r.window = t / cfg.egress.buckets →
    o[t].rows = fun _ s => ((r.grid[t % cfg.egress.buckets]?.bind fun row => row[s.1]?).join).map bitsOfBytes) ∧
  ∀ t (ht : t < o.length) κ, ∀ r ∈ out.records, r.link = linkOf κ → r.window = t / cfg.egress.buckets →
    (o[t].status κ = .complete ↔ r.status = .complete)

/-- The frames `frames` opens from the operator's ingress up to each bucket `t < W·T` and the content advice `a` are
the ones `e` opens on the device's egress link: the same payloads, anchors and indices. -/
def OpensFrames (cfg : Config) (e : Evidence) {K : Nat} {In Ac : Type}
    (frames : List In → Ac → Fin 1 → List NetTiming.LogicalFrame) (a : Ac)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In)
    (o : List (NetTiming.Obs 1 K cfg.egress.allocation)) : Prop :=
  ∀ t < cfg.windows * cfg.egress.buckets, ∀ link, e.links[0]? = some link →
    (frames (NetTiming.ingressUpTo op o t) a 0).map (fun f => (f.payload, f.anchor, f.index)) =
      link.frames.map (fun f => (f.payload, f.anchor, f.index))

/-- Every observation sequence read off the run is the reference's: for boundary links `linkOf`, each one of the
device's two, and a sequence `o` of the `W·T` buckets that `Observes` the device's records and whose opened frames
(`frames` of the operator's ingress and content advice `a`) are the ones `e` opens, the reference `Accepted` holds of
`o` for the constant-rate Program with advice `(a, σ)`, `σ` the syncs `e` declares (so Theorems 1(b) and 3,
`EncardAccSeqsLeConstantRate` and the rest, count `o`). -/
def RunIsProgramRun (x : DeviceInput) (out : DeviceOutput) (e : Evidence) : Prop :=
  ∀ {K : Nat} {In Ac : Type} (frames : List In → Ac → Fin 1 → List NetTiming.LogicalFrame) (a : Ac)
    (op : Nat → List (NetTiming.Obs 1 K x.config.egress.allocation) → In)
    (o : List (NetTiming.Obs 1 K x.config.egress.allocation)) (linkOf : Fin K → String),
    (∀ κ, linkOf κ ∈ [x.config.egressLink, x.config.ingressLink]) →
    o.length = x.config.windows * x.config.egress.buckets → Observes x.config out linkOf o →
    OpensFrames x.config e frames a op o →
    ∃ σ : Fin x.config.windows → (refSyncSet x.config.syncSet).Elem,
      (∀ (w : Fin x.config.windows) link, e.links[0]? = some link →
        ∃ s, link.syncs[w.1]? = some s ∧ (σ w).1.1 = s.delta ∧ (σ w).2.1 = s.rho) ∧
      NetTiming.Accepted (NetTiming.constantRate (S := x.config.egress.allocation) x.config.egress.buckets
        x.config.windows (refSyncSet x.config.syncSet) frames) op (x.config.windows * x.config.egress.buckets) o (a, σ)

/-- **An accepted run is a run of the certifier program under the declared advice (5b).** For every input of the device
and every commitment and evidence the check accepts under the device's parameters, if the device runs as built, every
observation sequence read off its run is accepted by the reference's constant-rate Program (`RunIsProgramRun`); or, as
in `AcceptedRowsAreProgramRows`, the run's strings hold a SHA-512 collision or its certificates a message of this run's
session the device never signed in it. -/
def AcceptedRunsAreProgramRuns : Prop :=
  ∀ (device : DeviceInput → DeviceOutput) (x : DeviceInput) (c : Commitment) (e : Evidence),
    protocol.Accepts (deviceParams x) c e → NetTiming.DeviceIntact device deviceCode →
    RunIsProgramRun x (device x) e ∨
    (∃ y ∈ runStrings x (device x) c e, ∃ z ∈ runStrings x (device x) c e, y ≠ z ∧
      Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
    ∃ r ∈ c.records.toList, ∃ m, certMessage x.issuer.session r = some m ∧
      LmsVerifies (deviceParams x) m r.certificate.signature ∧ ¬ Signed x.issuer.session (device x) m

/-- `o` is the observation sequence of an accepted run of `device` on input `x` with configuration `cfg` and key `key`
(in the session of `x`'s issuer), commitment `c`, evidence `e` and content advice `a`: the check accepts the unit, the
boundary links `linkOf` are the device's, `o` `Observes` the device's records, and its opened frames are the ones `e`
opens. -/
def AcceptedRun (cfg : Config) (key : Lms.PrivateKey) (device : DeviceInput → DeviceOutput) {K : Nat} {In Ac : Type}
    (frames : List In → Ac → Fin 1 → List NetTiming.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String)
    (o : List (NetTiming.Obs 1 K cfg.egress.allocation)) (x : DeviceInput) (c : Commitment) (e : Evidence) (a : Ac) :
    Prop :=
  x.config = cfg ∧ x.issuer.key = key ∧ protocol.Accepts (unitParams cfg key x.issuer.session) c e ∧
    (∀ κ, linkOf κ ∈ [cfg.egressLink, cfg.ingressLink]) ∧ Observes cfg (device x) linkOf o ∧
    OpensFrames cfg e frames a op o

/-- The observation sequences of accepted runs, of the `W·T` buckets with jitter classes below `D`. -/
def AcceptedObservations (cfg : Config) (key : Lms.PrivateKey) (device : DeviceInput → DeviceOutput) {K : Nat}
    {In Ac : Type} (frames : List In → Ac → Fin 1 → List NetTiming.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String) (D : Nat) :
    Set (List (NetTiming.Obs 1 K cfg.egress.allocation)) :=
  {o | o.length = cfg.windows * cfg.egress.buckets ∧ (∀ y ∈ o, y.jitter < D) ∧
    ∃ (x : DeviceInput) (c : Commitment) (e : Evidence) (a : Ac), AcceptedRun cfg key device frames op linkOf o x c e a}

/-- The accepted observation sequences number at most `#Ac · |Σ_sync|^W · D^(W·T)`, or some accepted run carries a
break: two distinct strings it hashes with one SHA-512 digest, or a certificate that verifies under the device's key
over its message in the run's session, which the device never signed in that run. -/
def ObservationsBounded (cfg : Config) (key : Lms.PrivateKey) (device : DeviceInput → DeviceOutput) {K : Nat}
    {In Ac : Type} [Fintype Ac] (frames : List In → Ac → Fin 1 → List NetTiming.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String) (D : Nat) : Prop :=
  (AcceptedObservations cfg key device frames op linkOf D).encard ≤
    ((Fintype.card Ac * (refSyncSet cfg.syncSet).size ^ cfg.windows * D ^ (cfg.windows * cfg.egress.buckets) : ℕ) :
      ℕ∞) ∨
  ∃ o x c e a, AcceptedRun cfg key device frames op linkOf o x c e a ∧
    ((∃ y ∈ runStrings x (device x) c e, ∃ z ∈ runStrings x (device x) c e, y ≠ z ∧
        Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
      ∃ r ∈ c.records.toList, ∃ m, certMessage x.issuer.session r = some m ∧
        LmsVerifies (deviceParams x) m r.certificate.signature ∧ ¬ Signed x.issuer.session (device x) m)

/-- **What accepted runs can carry (Theorem 3 for the check).** For a device that runs as built, of every configuration
and key, the observation sequences of its accepted runs number at most `#Ac · |Σ_sync|^W · D^(W·T)`
(`NetTiming.Guarantees.EncardAccSeqsLeConstantRate` with `n = W·T`), or some accepted run carries a break
(`ObservationsBounded`). -/
def EncardAcceptedObservationsLe : Prop :=
  ∀ (device : DeviceInput → DeviceOutput) (cfg : Config) (key : Lms.PrivateKey) {K : Nat} {In Ac : Type} [Fintype Ac]
    (frames : List In → Ac → Fin 1 → List NetTiming.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String) (D : Nat),
    NetTiming.DeviceIntact device deviceCode → ObservationsBounded cfg key device frames op linkOf D

end Properties.NetworkCertifier
