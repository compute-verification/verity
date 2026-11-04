import Properties.CertifierNetwork
import Definitions.NetworkCertifier
import Mathlib.Data.Set.Card

/-!
# What an accepted run of the certifier device is

The network certificate check (`NetworkCheck.Check.protocol`) meets the unit's certifier device through what the device
returns: the records `CertifierDevice.step` publishes over the trace the world gives it, and the certificates the host's
assembly has its issuer sign for them. Each property here is stated over those outputs, for every input of the device and
every commitment and evidence the check accepts under the device's parameters (`unitParams`), given the one premise of
the network lane, that the device runs as built (`NetTiming.DeviceIntact`, the listed `trusted-host`). Everything
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
already rests on (`DeviceInput.next` is the host's), rest on `trusted-host`, where the state the device keeps on
its host is its next LMS leaf and the session ids it has started. That claim's part about one run is
`NetTiming.DeviceIntact`; no statement here takes the rest as a premise, since a statement of one run can't name other
runs.

`Proofs.Properties.CertifierNetwork.Runs` proves `AcceptedRowsAreProgramRows`; `Proofs.NetworkCertifier.Accepted` (in
`security_proofs`, through the reference's `Proofs.NetworkCertifier.Code`) proves `AcceptedRunsAreProgramRuns` and the
count it gives, `EncardAcceptedObservationsLe`; `Proofs.NetworkCertifier.Causal` proves `ExpectedGridCausal` and the
count over the first `m` windows, `EncardProgramRunsLeWindows`.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open _root_.CertifierDevice
open NetworkCertifier (ClockSync)
open Properties.CertifierNetwork (published unitParams directionOf gridOf syncSetOf reserve advance certified)

namespace Properties.NetworkCertifier

/-! ## The device -/

/-- What the certifier device is given in one run: its configuration, its issuer (its key and certifier session), the
signer's state (its next unused leaf and the session ids it has started), its own coins (`salts n`, the salt it draws
for the `n`-th record it certifies) and the trace of events the world gives it. -/
structure DeviceInput where
  config : Config
  issuer : Issuer
  next : Nat
  sessions : List ByteArray
  salts : Nat → ByteArray
  trace : List Event

/-- What the device returns: the records it publishes and the certificates it signs. -/
structure DeviceOutput where
  records : List BlockGrid
  certificates : List CertifiedRecord

/-- The certificates the host's assembly (`Properties.CertifierNetwork.certifyAll`) has the issuer sign, in order: each
record certified under the next salt of the device's coins at the reserved leaf, under its link's direction, grid and
parameters digest; where the assembly stops on an error, the ones signed before it. -/
def issued (c : Config) : Issuer → Nat → (Nat → ByteArray) → List BlockGrid → List CertifiedRecord
  | _, _, _, [] => []
  | i, next, salts, r :: rs =>
    match parameters (gridOf c r) (syncSetOf c r) >>=
        fun pd => i.certify (salts 0) (reserve i next) r (directionOf c r) (gridOf c r) pd with
    | .ok (i', cr?) => cr?.toList ++ issued c i' (advance i next) (fun n => salts (n + 1)) rs
    | .error _ => []

/-- **The device's code**: the records `step` publishes over the trace, and the certificates of them its issuer signs;
none where the signer's state has started the issuer's session already (it is in `x.sessions`), since the state refuses
to start it again (`Core.World.start`). -/
def deviceCode (x : DeviceInput) : DeviceOutput :=
  ⟨published x.config x.trace, if x.sessions.any (fun s => decide (s = x.issuer.session)) then []
    else issued x.config x.issuer x.next x.salts (published x.config x.trace)⟩

/-- The check's parameters of the device's unit: its configuration's links, windows, grids and Σ_sync, its key and its
issuer's certifier session. -/
def deviceParams (x : DeviceInput) : NetworkCheck.Check.Params := unitParams x.config x.issuer.key x.issuer.session

/-- The egress grid's parameters digest under Σ_sync, the one the check recomputes heads under (empty where it has
none; an accepted unit has one). -/
def egressDigest (p : NetworkCheck.Check.Params) : ByteArray :=
  (parametersDigest p.grid (some (manifestSet p.syncSet))).toOption.getD ByteArray.empty

/-! ## The breaks -/

/-- What the device hashes: each certificate's head, the head's digest under the certificate's salt and its message's
input in certifier session `session`, and the frame tree of each record it published, under its grid. -/
def deviceStrings (c : Config) (session : ByteArray) (out : DeviceOutput) : List ByteArray :=
  out.certificates.flatMap (fun cr => headStrings cr.head ++ saltedStrings cr.salt cr.head ++
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

/-- A value of Σ_sync: a δ of its list and a ρ of its list. -/
abbrev SyncValue (ss : NetworkCertifier.SyncSet) : Type := {d // d ∈ ss.deltas.toFinset} × {ρ // ρ ∈ ss.rhos.toFinset}

/-- The clock sync a value of Σ_sync names. -/
def SyncValue.sync {ss : NetworkCertifier.SyncSet} (v : SyncValue ss) : ClockSync :=
  ⟨v.1.1, v.2.1, ss.rhos_pos _ (List.mem_toFinset.mp v.2.2)⟩

/-- The reference's frames of the unit's one execution block, its `W` windows, under the advice `(a, σ)`: the frames
`frames` opens from the ingress and the content advice `a`, in order, each sent at the program's hand tick
(`NetworkCertifier.handingBuckets`) under the syncs `σ` names. -/
def programFrames (cfg : Config) {In Ac : Type} (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame) :
    List In → Fin 1 → Ac × (Fin cfg.windows → SyncValue cfg.syncSet) → Fin 1 → List NetTiming.LogicalFrame :=
  fun ins _ v ℓ =>
    ((frames ins v.1 ℓ).zip (NetworkCertifier.handingBuckets cfg.egress.buckets cfg.windows (fun w => (v.2 w).sync)
      (frames ins v.1 ℓ))).map fun p => ⟨p.1.payload, p.2⟩

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
the ones `e` opens on the device's egress link. -/
def OpensFrames (cfg : Config) (e : Evidence) {K : Nat} {In Ac : Type}
    (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame) (a : Ac)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In)
    (o : List (NetTiming.Obs 1 K cfg.egress.allocation)) : Prop :=
  ∀ t < cfg.windows * cfg.egress.buckets, ∀ link, e.links[0]? = some link →
    frames (NetTiming.ingressUpTo op o t) a 0 = link.frames

/-- Every observation sequence read off the run is the reference's: for boundary links `linkOf`, each one of the
device's two, and a sequence `o` of the `W·T` buckets that `Observes` the device's records and whose opened frames
(`frames` of the operator's ingress and content advice `a`) are the ones `e` opens, the reference `Accepted` holds of
`o` for the constant-rate Program of one execution block of the `W` windows, with advice `(a, σ)`, `σ` the syncs `e`
declares, and the frames sent at the program's hand ticks under them (`programFrames`; so Theorems 1(b) and 3 count
`o`). -/
def RunIsProgramRun (x : DeviceInput) (out : DeviceOutput) (e : Evidence) : Prop :=
  ∀ {K : Nat} {In Ac : Type} (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame) (a : Ac)
    (op : Nat → List (NetTiming.Obs 1 K x.config.egress.allocation) → In)
    (o : List (NetTiming.Obs 1 K x.config.egress.allocation)) (linkOf : Fin K → String),
    (∀ κ, linkOf κ ∈ [x.config.egressLink, x.config.ingressLink]) →
    o.length = x.config.windows * x.config.egress.buckets → Observes x.config out linkOf o →
    OpensFrames x.config e frames a op o →
    ∃ σ : Fin x.config.windows → SyncValue x.config.syncSet,
      (∀ (w : Fin x.config.windows) link, e.links[0]? = some link →
        ∃ s, link.syncs[w.1]? = some s ∧ (σ w).1.1 = s.delta ∧ (σ w).2.1 = s.rho) ∧
      NetTiming.Accepted (NetTiming.constantRate (S := x.config.egress.allocation) x.config.egress.buckets
        x.config.windows 1 (programFrames x.config frames)) op (x.config.windows * x.config.egress.buckets) o
        (fun _ => (a, σ))

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
    (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String)
    (o : List (NetTiming.Obs 1 K cfg.egress.allocation)) (x : DeviceInput) (c : Commitment) (e : Evidence) (a : Ac) :
    Prop :=
  x.config = cfg ∧ x.issuer.key = key ∧ protocol.Accepts (unitParams cfg key x.issuer.session) c e ∧
    (∀ κ, linkOf κ ∈ [cfg.egressLink, cfg.ingressLink]) ∧ Observes cfg (device x) linkOf o ∧
    OpensFrames cfg e frames a op o

/-- The observation sequences of accepted runs, of the `W·T` buckets with jitter classes below `D`. -/
def AcceptedObservations (cfg : Config) (key : Lms.PrivateKey) (device : DeviceInput → DeviceOutput) {K : Nat}
    {In Ac : Type} (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String) (D : Nat) :
    Set (List (NetTiming.Obs 1 K cfg.egress.allocation)) :=
  {o | o.length = cfg.windows * cfg.egress.buckets ∧ (∀ y ∈ o, y.jitter < D) ∧
    ∃ (x : DeviceInput) (c : Commitment) (e : Evidence) (a : Ac), AcceptedRun cfg key device frames op linkOf o x c e a}

/-- The accepted observation sequences number at most `#Ac · |Σ_sync|^W · D^(W·T)`, `|Σ_sync|` the count of its δs
times the count of its ρs, or some accepted run carries a break: two distinct strings it hashes with one SHA-512
digest, or a certificate that verifies under the device's key over its message in the run's session, which the device
never signed in that run (the module docstring reads it across the key's runs). -/
def ObservationsBounded (cfg : Config) (key : Lms.PrivateKey) (device : DeviceInput → DeviceOutput) {K : Nat}
    {In Ac : Type} [Fintype Ac] (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String) (D : Nat) : Prop :=
  (AcceptedObservations cfg key device frames op linkOf D).encard ≤
    ((Fintype.card Ac * (cfg.syncSet.deltas.length * cfg.syncSet.rhos.length) ^ cfg.windows *
      D ^ (cfg.windows * cfg.egress.buckets) : ℕ) : ℕ∞) ∨
  ∃ o x c e a, AcceptedRun cfg key device frames op linkOf o x c e a ∧
    ((∃ y ∈ runStrings x (device x) c e, ∃ z ∈ runStrings x (device x) c e, y ≠ z ∧
        Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
      ∃ r ∈ c.records.toList, ∃ m, certMessage x.issuer.session r = some m ∧
        LmsVerifies (deviceParams x) m r.certificate.signature ∧ ¬ Signed x.issuer.session (device x) m)

/-- **What accepted runs can carry (Theorem 3 for the check).** For a device that runs as built, of every configuration
and key, the observation sequences of its accepted runs number at most `#Ac · |Σ_sync|^W · D^(W·T)`
(Theorem 3's `2^A` cap, `NetTiming.Guarantees.EncardAccSeqsLeAdvice`, at the constant-rate Program of one execution
block, whose advice is `Ac × (Fin W → Σ_sync)`, with `n = W·T`), or some accepted run carries a break
(`ObservationsBounded`). -/
def EncardAcceptedObservationsLe : Prop :=
  ∀ (device : DeviceInput → DeviceOutput) (cfg : Config) (key : Lms.PrivateKey) {K : Nat} {In Ac : Type} [Fintype Ac]
    (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (linkOf : Fin K → String) (D : Nat),
    NetTiming.DeviceIntact device deviceCode → ObservationsBounded cfg key device frames op linkOf D

/-! ## (5c) What a window's sync reaches -/

/-- **A later window's sync never changes an earlier window's grid.** The grid the check expects of an egress link in
window `w` (`expectedGrid`, whose `record_head` digest `egressVerdict` compares) reads only the syncs the link declares
for windows `0 … w`: two links with the same frames whose syncs agree there (`early`, window `w` among them) get the
same grid in window `w`, whatever each declares after it (`later`, `later'`: any clock-sync values, in Σ_sync or not).
A frame is eligible no earlier than its anchor, its hand tick (`NetworkCertifier.handingBuckets`) is never before the
one before it, and the queue carries forward, so a window's sync moves only hand ticks at or after the window's start. -/
def ExpectedGridCausal : Prop :=
  ∀ (g : NetworkCheck.Check.Grid) (W : Nat) (frames : List NetworkCertifier.LogicalFrame)
    (early later later' : Array ClockSync) (w : Fin early.size),
    expectedGrid g W ⟨frames, early ++ later⟩ w = expectedGrid g W ⟨frames, early ++ later'⟩ w

/-- **The program's runs over its first `m` windows (Theorem 3 for the program, window by window).** The reference's
constant-rate Program of one execution block of the `W` windows, each frame sent at the program's hand tick under the
syncs its advice names (`programFrames`, the Program `RunIsProgramRun` reads an accepted run by), accepts at most
`#Ac · |Σ_sync|^min(W, m) · D^(m·T)` sequences of the first `m` windows' `m·T` buckets with jitter classes below `D`,
`|Σ_sync|` the count of its δs times the count of its ρs: a window's sync changes no bucket before the window
(`ExpectedGridCausal`), so the content advice and the first `m` windows' syncs fix those buckets' rows. The syncs are
`log2 |Σ_sync|` bits per window opened, 7 at 16 × 8, counted from the run's first window. A window on its own, given
the run before it, may carry more: the queue and the hand ticks carry across windows. -/
def EncardProgramRunsLeWindows : Prop :=
  ∀ (cfg : Config) {K : Nat} {In Ac : Type} [Fintype Ac]
    (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K cfg.egress.allocation) → In) (m D : Nat),
    (NetTiming.AccSeqs (NetTiming.constantRate (S := cfg.egress.allocation) cfg.egress.buckets cfg.windows 1
      (programFrames cfg frames)) op (m * cfg.egress.buckets) D).encard ≤
      ((Fintype.card Ac * (cfg.syncSet.deltas.length * cfg.syncSet.rhos.length) ^ min cfg.windows m *
        D ^ (m * cfg.egress.buckets) : ℕ) : ℕ∞)

end Properties.NetworkCertifier

/-! ## A one-window instance

The configuration the completeness statements' honest cases run (`Proofs.Properties.CertifierNetwork.Instance`), and
the honest network certifier's record that `EncardProgramRunsLeWindows`'s honest lemma counts at it. -/

namespace Properties.CertifierNetwork.Instance

def grid : _root_.CertifierDevice.Params := ⟨1, 1, 9, 1, 1⟩

def syncSet : _root_.NetworkCertifier.SyncSet := ⟨[0], [1], by simp, by simp, by decide, by simp, by simp⟩

def sync : _root_.NetworkCertifier.ClockSync := ⟨0, 1, by decide⟩

/-- One window of one bucket on each grid, its sync declared. -/
def config : _root_.CertifierDevice.Config :=
  { egress := grid, ingress := grid, syncSet, syncs := #[sync], windows := 1, origin := 0, slack := 0 }

/-- One frame of content `[true]`, anchored at bucket 0. -/
def oneFrame : List Unit → Unit → Fin 1 → List _root_.NetworkCertifier.LogicalFrame := fun _ _ _ => [⟨[true], 0, 0⟩]

/-- The advice of `config`'s one window: the content `()` and Σ_sync's one value, δ = 0 and ρ = 1. -/
def oneAdvice : Fin 1 → Unit × (Fin config.windows → Properties.NetworkCertifier.SyncValue config.syncSet) :=
  fun _ => ((), fun _ => (⟨0, List.mem_toFinset.mpr (List.mem_singleton_self 0)⟩,
    ⟨1, List.mem_toFinset.mpr (List.mem_singleton_self 1)⟩))

/-- The honest network certifier's record of `config`'s one bucket: the Program's row of `oneFrame` under `oneAdvice`,
jitter class 0, its boundary link complete. -/
def honestObs : List (NetTiming.Obs 1 1 config.egress.allocation) :=
  [⟨(NetTiming.constantRate (S := config.egress.allocation) config.egress.buckets
      config.windows 1 (Properties.NetworkCertifier.programFrames config oneFrame)).rows 0 [()] oneAdvice, 0,
    fun _ => .complete⟩]

end Properties.CertifierNetwork.Instance
