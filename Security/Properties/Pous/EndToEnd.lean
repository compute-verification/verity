import PousCheck.Run
import Properties.Pous
import Properties.Pous.TimedAudit
import Properties.Pous.MemoryChallenger
import Properties.Identity
import Definitions.Pous.Assumptions.P2
import Definitions.Pous.P2.Cost

/-!
# PoUS end to end: what an accepted run certifies, and P2's audit in the ring model

Two kinds of statement at P2's deployed point (`P2.Params.deployed`: `p = 2^16448 − 21065`, `B = 2^19` blocks of 2,056
bytes, `k = 105` rounds, `Δ + RTT ≤ 854,000 ns`), which nothing joins inside Lean.

* `RunCheckAccepts` and `RunCertifies`, about the deployed code, deterministic and assumption free. The composed check
  of a run (`PousCheck.Run.check`, `pous-run`, Python `run.check`) reads the verifier's list of memory-challenger keys,
  the setup's public outputs, the memory challenger's table of `C*`'s row digests, its signed session and the id the
  auditor designated for the session; when it accepts, the session audits that setup at the deployed point
  (`AuditsSetup`: every block, `k = 105`, within `Δ + RTT`, blocks of 2,056 bytes, bound to the published root), runs
  under the designated id and is signed under a key of the verifier's list, and for every `C*` of the point's shape
  (`Shaped`) whose leaves have the published root, the memory
  challenger holding that key ran a session with that commitment in which every one of the 105 answers it received was
  the challenged block of that `C*`, within `Δ + RTT` on its own clock; or the run's own data holds a SHA-512
  collision (`RunBreak`, `TableBreak`) or an LMS forgery (`RunBreak`). `RunCertifies` adds the
  setup check's acceptance of `C*` (`Accepted`): `C*` decodes to the committed `W` and setup published its root, so
  every `C*` setup accepts is one `RunCheckAccepts` speaks of. Both are proved: `RunCertifies` from `RunCheckAccepts`
  and the setup check's `AcceptsDecodes` and `PublishesRoot` (`Properties.Pous`).
* `EndToEndRing`, `EndToEndRingDeployed`, `DrawExact`, `HonestPassesRing` and `HonestPassesAccepts`, **in the ring
  model, with `ω` uniform**: P2's square–mask–square over `Z/p` with a uniformly random odd labeling and uniformly random
  per-block keys (`p2ring`), against a server that is a `Server S D Q L` of this module, with the memory challenger's
  coins drawn as the code draws them (`CoinTree.make`, `FlockSoundness.Randomness.uniformOn` at `DRAW_BYTES`) from
  uniform bytes.

The step from the ring game to the deployed system is four claims, each its own id in `verity.core.claims`, cited
separately so that each can be rejected alone:

1. `pous-ring-device` (the model): the deployed server is a `Server`: a ring-model server whose state is bounded between
   setup, the session and its rounds, with one session per audit (the memory challenger's `SessionLog`, of which its
   host never restores an earlier copy: `trusted-host`) and the coins drawn online, after its state is fixed. It
   names no number.
2. `pous-ring-speed` (the numbers, by measurement): no allowed hardware finishes a `(D + 1)`-th sequential
   multiplication round, `Q + 1` key queries or `L + 1` ring operations within `Δ + RTT`, at the corollary's
   `D = 16,446`, `Q = 2^20` and `L = 2^64`; `EndToEndRingDeployed`'s docstring derives `D` and its margin.
3. `pous-ring-instantiation`: P2 with the identity labeling (`φ = id`), SHA-512-CTR keys and its published Merkle root of
   `C*` is no easier than `p2ring` with uniform `ω`. Every model in which P2 is secure hides structure the deployed code
   computes with, so "the real P2 resists every Lean function" is false (a server storing nothing recomputes `C*`
   locally, which a Lean function does for free): this step is a claim, never a theorem.
4. `prf/sha-512`: the indices the memory challenger derives from its secret source (`Randomness.derive`, then the
   counter-mode SHA-512 stream that `Randomness.uniform` reads) are as good as uniform: each stream is replaced here by
   `DRAW_BYTES` uniform bytes, and `uniformOn` reads them exactly as the code does.

The ring game takes one named assumption, `P2RingAuditSecure`, the conjecture that P2 is secure in the ring model. It
takes none on the coin tree's hiding: the memory challenger releases the tree's root and key only after the session's
last answer (`memory_challenger.MemoryChallenger.commitment`), so the server sees each index as its round opens and
nothing else of the coins. The challenger is held to its coins by being an auditor-trusted device that runs its stated
code (`pous-ring-device`), not by an early commitment.

Two gaps stood between these statements and the deployed verifier:

* **G-1**: the check's half is fixed in code; one session per id is the memory challenger's `SessionLog` under its host
  presumption, which `trusted-host` covers once it is widened to the challenger's host. The check's half: a session's
  `binding` is required and is the setup's root (`Run.check` refuses any other), so a session names the setup it audits,
  and its id is the one the
  auditor designated for the audit (`Run.check` refuses any other); `RunCheckAccepts` states it. The memory challenger
  runs at most one session under an id (`memory_challenger.SessionLog`, a file of its own, recorded before the session's
  first challenge), so an audit has at most one session the check accepts; that refusal is the device's code and log,
  not modeled here;
* **G-2**, fixed: the composed setup-to-challenger check is `PousCheck.Run.check`, and `RunCheckAccepts` is what its
  acceptance gives.
-/

open MemoryChallenger MemoryChallenger.Properties
open PousCheck PousCheck.Properties

namespace Pous.EndToEnd

open scoped ENNReal

/-! ## The deployed point -/

/-- P2's registered parameters (`p2-p16448-c21065`, `p2-2054B`, 8 segments of `2^16` blocks). -/
def q : P2.Params := P2.Params.deployed

/-- The memory challenger's limit at the operating point: `Δ + RTT` with the RTT allowance at its cap, in ns. -/
def LIMIT_NS : Nat := DELTA_NS + RTT_CAP_NS

/-! ## A run's blocks and what a run hashes -/

/-- `C*`'s blocks in block order: block `i` is block `i mod 2^16` of segment `⌊i / 2^16⌋` when every segment has
`2^16` blocks, which `Decodes` requires. -/
def flat (C : Array (Array ByteArray)) : Array ByteArray := C.flatten

/-- What `blockDigest` hashes for a block: its `sha512/row/v2` prefix, then the block. -/
def blockInput (block : ByteArray) : ByteArray := Flock.HmRow.rowPrefixV2 (8 * block.size) ++ block

/-- What `Commit.commit` hashes for a payload `W` (`Properties.Identity.TaggedIsFlock` reads it as one SHA-512). -/
def commitInput (W : ByteArray) : Option ByteArray :=
  (do
    let m ← Commit.manifest q
    Properties.Identity.taggedInput Commit.COMMIT_TAG (Flock.be64 m.size.toUInt64 ++ m ++ W)).toOption

/-- The answers the device records in a session: the observations of the right size, up to the session's `k` (once it
has ended complete, `Recorded.round` skips the rest). -/
def answers (r : Run) : List Observed :=
  (r.observed.filter fun o => o.block.size == r.blockBytes).take r.coins.k

/-- Every one of the session's `k` recorded answers is the challenged block of `C*` and arrived within the session's
limit on the device's clock. -/
def AnsweredOnTime (r : Run) (C : Array (Array ByteArray)) : Prop :=
  (answers r).length = r.coins.k ∧
    ∀ j (hj : j < (answers r).length),
      (flat C)[r.coins.indices[j]!]? = some (answers r)[j].block ∧
        (answers r)[j].sendNs ≤ (answers r)[j].receiptNs ∧
        (answers r)[j].receiptNs ≤ (answers r)[j].sendNs + r.limitNs

/-- The strings the run hashes, beyond `runStrings`: each answer the device received under the key, and each block of
`C*`, as `blockDigest` hashes them. -/
def blockStrings (pk : ByteArray) (dev : Device) (C : Array (Array ByteArray)) : List ByteArray :=
  (dev.sessions pk).flatMap (fun r => r.observed.map fun o => blockInput o.block) ++ (flat C).toList.map blockInput

/-- The run's own data holds a break: two distinct strings it hashes with one SHA-512 digest (`cr/sha-512`), or a
signature verifying under a trusted key over a session message the device never signed (`euf-cma/lms-sha512-n64`,
under the one-role reading of `MemoryChallenger.Properties`). Every witness is bounded by the run. -/
def RunBreak (trusted : List ByteArray) (s : SignedSession) (dev : Device) (C : Array (Array ByteArray)) : Prop :=
  (let xs := runStrings s dev ++ blockStrings s.publicKey dev C
   ∃ x ∈ xs, ∃ y ∈ xs, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) ∨
    ∃ p m, SCHEMES.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧ sessionMessage s = some m ∧
      LmsVerifies p s.publicKey m s.signature ∧ ¬ Signed dev s.publicKey m

/-- `C*` decodes to a payload other than `W` whose commitment is `W`'s: two distinct strings `Commit.commit` hashes,
both of the run, with one SHA-512 digest (`cr/sha-512`). A server that commits to a payload other than the one it was
given is caught here, not by the audit: `RunCertifies` concludes `DecodesToCommitment`, and a commitment the decoded
`W` does not match is this break. -/
def CommitBreak (W : ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) : Prop :=
  ∃ W', Decodes q c.salt C W' ∧ W' ≠ W ∧ ∃ x y, commitInput W = some x ∧ commitInput W' = some y ∧
    Flock.Sha512.hash x = Flock.Sha512.hash y

/-! ## The composed check -/

/-- The session audits the setup `pub` at the deployed point: the setup published `2^19` blocks under a 24-byte salt,
and the session audits every one of them in `k = 105` rounds, within `Δ + RTT`, with answers of the point's 2,056
bytes, bound to the published root. -/
def AuditsSetup (pub : SetupProtocol.Public) (sc : SessionCommitment) : Prop :=
  pub.blocks = q.blocks ∧ pub.salt.size = Run.SALT ∧ sc.blocks = q.blocks ∧ sc.k = K ∧ sc.limitNs ≤ LIMIT_NS ∧
    sc.blockBytes = q.blockBytes ∧ sc.binding = pub.root

/-- `C*` has the deployed point's shape: 8 segments of `2^16` blocks of 2,056 bytes, as `Decodes` requires. -/
def Shaped (C : Array (Array ByteArray)) : Prop :=
  C.size = q.segments ∧ ∀ s (hs : s < C.size), C[s].size = q.blocksPerSegment ∧
    ∀ v (hv : v < C[s].size), C[s][v].size = q.blockBytes

/-- What a leaf hashes for block `v` of segment `s` with row digest `R` (`Run.rowLeaf`, `Merkle.packageLeaf`):
`0x00 ‖ salt ‖ u64le(s) ‖ u16le(v) ‖ R`. -/
def leafInput (salt : ByteArray) (s v : Nat) (row : ByteArray) : ByteArray :=
  ByteArray.mk #[0] ++ salt ++ Merkle.le s 8 ++ Merkle.le v 2 ++ row

/-- What the run hashes for its two trees: over the table's rows (the composed check's `Merkle.root`) and over `C*`'s
blocks (setup's), each inner node and each of the `2^19` leaves, block `i` being block `i mod 2^16` of segment
`⌊i / 2^16⌋`. -/
def tableStrings (pub : SetupProtocol.Public) (digests : ByteArray) (C : Array (Array ByteArray)) : List ByteArray :=
  let T := Run.rowLeaves q pub.salt digests
  let L := leaves pub.salt C
  nodeInputs T 0 T.size ++ nodeInputs L 0 L.size ++
    (List.range q.blocks).flatMap fun i =>
      [leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Run.row digests i),
        leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Merkle.rowDigest ((flat C)[i]!))]

/-- The table and `C*` hold a break: two distinct strings their trees hash with one SHA-512 digest (`cr/sha-512`). -/
def TableBreak (pub : SetupProtocol.Public) (digests : ByteArray) (C : Array (Array ByteArray)) : Prop :=
  let xs := tableStrings pub digests C
  ∃ x ∈ xs, ∃ y ∈ xs, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

/-- **An accepted run is the device's honest record of the setup's blocks, or carries a break.** Deterministic and
assumption free: if the composed check (`Run.check`) accepts, against the verifier's list `trusted` of memory-challenger
keys, the setup's public outputs `pub`, the memory challenger's table `digests` and its session `r` under the designated
id `sid`, then the session audits `pub` at the deployed point (`AuditsSetup`), runs under `sid` and is signed under a
key in `trusted`; and for every device and every `C*` of the point's shape whose leaves have the published root, that
key's holder ran a session with the session's commitment in which every recorded answer is the challenged block of
`C*`, on time on its own clock, or the run's own data holds a SHA-512 collision or an LMS forgery under a key in
`trusted`. `trusted` is the verifier's list of memory-challenger keys and never comes from the run's record: whoever
writes the record could otherwise list its own key, and its own signature would then count as a forgery. The times are
the device's own readings: what the server reports about its latency enters nowhere. The table is the memory
challenger's own: whichever table it held, an accepted one has the published root, so it is `C*`'s or there is a
collision. -/
def RunCheckAccepts : Prop :=
  ∀ (trusted : List ByteArray) (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray),
    Run.check trusted pub digests r sid = .ok () →
    ∀ (dev : Device) (C : Array (Array ByteArray)),
      AuditsSetup pub r.commitment ∧ r.commitment.session = sid ∧ r.signed.publicKey ∈ trusted ∧
        (Shaped C → Merkle.root (leaves pub.salt C) = some pub.root →
          (∃ run ∈ dev.sessions r.signed.publicKey, run.commitment = r.commitment ∧ AnsweredOnTime run C) ∨
            RunBreak trusted r.signed dev C ∨ TableBreak pub digests C)

/-- **What a run must pass**, the verifier's whole decision: setup accepts `C*` against the server's commitment and
salt and publishes `pub`, and the composed check accepts, against the verifier's list `trusted` of memory-challenger
keys (never the run's record), `pub`, the memory challenger's table and its session under the id `sid` the auditor
designated. -/
def Accepted (trusted : List ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray) : Prop :=
  SetupProtocol.protocol.check q c C = .accept pub () ∧ Run.check trusted pub digests r sid = .ok ()

/-- **An accepted run certifies `C*`.** For every verifier's list `trusted` of memory-challenger keys (never the run's
record) and every device, if setup and the composed check accept a run (`Accepted`), then `C*` decodes to a payload
with the server's commitment, setup published its root (`Published`: `Merkle.root (leaves c.salt C) = some pub.root`,
so the session's `binding` is the root of the `C*` setup checked), the session audits that setup at the deployed point
under the designated id and is signed under a key in `trusted`, and that key's holder ran a session with the accepted
commitment in which every recorded answer is the challenged block of `C*`, on time on its own clock; or the run's own
data holds a SHA-512 collision or an LMS forgery under a key in `trusted`. A dishonest commitment (`C*` decoding to a
`W'` other than the `W` the server was handed) is `CommitBreak`'s. It follows from `RunCheckAccepts` and the setup
check's `AcceptsDecodes` and `PublishesRoot`, all proved. -/
def RunCertifies : Prop :=
  ∀ (trusted : List ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray) (dev : Device),
    Accepted trusted c C pub digests r sid →
      DecodesToCommitment q c C ∧ Published q c C pub ∧ AuditsSetup pub r.commitment ∧ r.commitment.session = sid ∧
        r.signed.publicKey ∈ trusted ∧
        ((∃ run ∈ dev.sessions r.signed.publicKey, run.commitment = r.commitment ∧ AnsweredOnTime run C) ∨
          RunBreak trusted r.signed dev C ∨ TableBreak pub digests C)

/-! ## The ring game -/

/-- P2 in the ring model at the deployed point: square–mask–square over `Z/p` with a random odd labeling, `2^19`
blocks of 16,448 bits, each carrying 16,432 payload bits (2,054 bytes); `P2RingAuditSecure`'s scheme. -/
noncomputable abbrev sch : Scheme (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)) :=
  p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432

/-- The state bound `⌊ρ·|C|⌋` in bits, `|C| = 2^19 · 16,448` bits (`ρ = 18/19`). -/
noncomputable def S : ℕ := sch.stateBound Params.ρ

/-- The full state: all of `C`, `2^19 · 16,448` bits, more than `S`. -/
def FULL : ℕ := 2 ^ 19 * 16448

/-- `n` bytes, each uniform when drawn. -/
abbrev Bytes (n : ℕ) : Type := Fin n → Fin 256

/-- The bytes, as the code holds them. -/
def bytesOf {n : ℕ} (x : Bytes n) : ByteArray := ⟨Array.ofFn fun i => (x i).val.toUInt8⟩

/-- The memory challenger's fresh coins for one session, from the OS (`memory_challenger.CoinTree.draw`'s key and
salts): its hm96 key, 256 bytes, and one 192-byte salt per round. -/
abbrev HmCoins : Type := Bytes 256 × (Fin 105 → Bytes 192)

/-- The fresh hm96 key: 256 bytes, bit 2047 cleared (`HM96.check_key`). -/
def coinKey (k : Bytes 256) : ByteArray :=
  let b := bytesOf k
  b.set! 255 (b.get! 255 &&& 0x7f)

/-- The streams the indices are read from, one per round: `DRAW_BYTES` uniform bytes each, standing for the counter-mode
SHA-512 stream `Randomness.uniform` reads (`prf/sha-512`). -/
abbrev Streams : Type := Fin 105 → Bytes DRAW_BYTES

/-- What the memory challenger releases about the indices `I` of session `session` once the session is signed, after
its last answer: the coin tree's root and its hm96 key (`CoinTree.make`), or nothing when the code refuses to build the
tree. -/
def coinView (session : ByteArray) (c : HmCoins) (I : Fin 105 → Fin (2 ^ 19)) : Option (ByteArray × ByteArray) :=
  (CoinTree.make session (2 ^ 19) (Array.ofFn fun j => (I j).val) (coinKey c.1)
    (Array.ofFn fun j => bytesOf (c.2 j))).toOption.map fun t => (t.root, t.key)

set_option exponentiation.threshold 20000 in
/-- **A server in the ring model** holding at most `S` bits between setup and the session and between rounds, each
answer within `(D, Q, L)` (`RingBounded`: at most `D` rounds with a multiplication, `Q` key queries and `L` ring
operations, for every `ω`):

* `pre W pp C ω` is all that survives setup, `S` bits, a function of `W`, `C` and the whole primitive (perfect
  isolation); it is fixed before the memory challenger's coins are drawn (online coins);
* `step W pp j σ i` answers round `j`'s index `i` from the state `σ` and what is public (`W`, `pp`), returning the
  block and the next `S`-bit state.

Each index is revealed as the previous answer returns, as the memory challenger reveals it, and nothing else of the
coins reaches the server before its last answer: the coin tree's root and key are released after the session is signed.
The real memory challenger also reveals the next index after a late answer, but a late answer ends the session `late`
and fails it, so what a server learns from indices seen after one only helps a server the game counts as failed. -/
structure Server (S D Q L : ℕ) : Type where
  pre : Preprocessor sch S
  step : Bits sch.n → Bits sch.t → ℕ → Bits S → Fin sch.B →
    OracleProg (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)) (Bits sch.ℓ × Bits S)
  bounded : ∀ W pp j σ i, (step W pp j σ i).RingBounded D Q L

/-- **The server passes the session**: `ω`, the coins and the streams `x`; the code's sampler reads index `I j` from
stream `j` at `2^19` blocks, the code builds the coin tree over those indices (`coinView`, released after the session),
and the server returns block `I j` of `C` in every one of the 105 rounds, sequentially.

On the device's side, `RunCheckAccepts` is what an accepted composed check gives: the device's session at the
deployed point, each of its recorded answers the challenged block of `C*`, on time. That a deployed server answering
such a session is a `Server` passing this event is `pous-ring-device`'s claim, not a theorem here. -/
def Passes {S D Q L : ℕ} (session : ByteArray) (W : Bits sch.n) (A : Server S D Q L)
    (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) : Prop :=
  ∃ (I : Fin 105 → Fin (2 ^ 19)) (v : ByteArray × ByteArray),
    (∀ j, FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf (x.2.2 j)) = some (I j).val) ∧
      coinView session x.2.1 I = some v ∧
      seqAnswers ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1)
          (A.step W (sch.enc W x.1 ()).1) 0 (List.ofFn I)
          (A.pre W (sch.enc W x.1 ()).1 (sch.enc W x.1 ()).2 x.1) =
        List.ofFn fun j => (sch.enc W x.1 ()).2 (I j)

/-- **The draw's rejection sampling never rejects at `2^19` blocks (`ε_rs = 0`)**: `uniformOn (2^19)` on `DRAW_BYTES`
uniform bytes returns an index, and it is exactly uniform. Its word is `width (2^19) = 11` bytes, and `2^19` divides
`2^88`, so no word is rejected. -/
def DrawExact : Prop :=
  ∀ P : Fin (2 ^ 19) → Prop,
    pr (fun s : Bytes DRAW_BYTES =>
        ∃ i : Fin (2 ^ 19), FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf s) = some i.val ∧ P i) = pr P

/-- **P2's audit in the ring model, with `ω` uniform, at the deployed point.** For every `(D, Q, L)` at which P2 is
secure in the ring model (`P2RingAuditSecure`): for every session id, every payload `W` and every `S`-bit,
`(D, Q, L)`-bounded server (`Server`), the server passes the session with probability at most `1% + 2^-128` over the
primitive `ω`, the memory challenger's coins and its uniform streams. The budget is `2^-129` (the ideal term),
`2^-130` (the instantiation's term in `P2RingAuditSecure`) and `ε_rs = 0` (`DrawExact`), which sum to less than
`2^-128`. No hiding term: the coin tree's root and key reach no server before its last answer.

It speaks of the deployed system only through the four claims of this module's header, each cited on its own:
`pous-ring-device` (the deployed server is a `Server`: state bounded between rounds, one session, online coins),
`pous-ring-speed` (the hardware's `D`, `Q` and `L` within `Δ + RTT`, by measurement), `pous-ring-instantiation` (P2 with
`φ = id`, SHA-512-CTR keys and its public root is no easier than `p2ring` with uniform `ω`) and `prf/sha-512` (the
derived streams are as good as uniform bytes). G-1: the check's half is fixed in code; one session per id is the memory
challenger's `SessionLog` under its host presumption, which `trusted-host` covers once it is widened to the challenger's
host. G-2 (the composed check) is fixed in code. What the deployed
verifier's acceptance gives is `RunCheckAccepts` (this module's header). -/
def EndToEndRing : Prop :=
  ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (session : ByteArray) (W : Bits sch.n) (A : Server S D Q L),
      pr (Passes session W A) ≤ ENNReal.ofReal Params.δ + Params.εMax

/-- **`EndToEndRing` at one root's depth, in the ring model, with `ω` uniform: `D = 16,446`, `Q = 2^20`, `L = 2^64`.**
The margin first. `P2RingAuditSecure` is claimed at `D = 16,446`, about half of `RingTight`'s falsification at
`D = 32,894` (`PousP2Ring.tight_p16448`, a responder that recomputes the block's two roots, if `p` is prime), and
`pous-ring-speed` keeps the timing's 2× margin: 1.995× at the measured 8-core speedup, 1.64× and 1.25× under the timing
red team's sync model.

The count. In the ring model a root of `y` is `y^((p+1)/4)` by `rootProg`: `n = 16,446` rounds over the exponent's bits
(`(p+1)/4 < 2^16446`, `PousP2Ring.p16448_root_bits`) and one more that squares the result to fix its sign, so
`n + 1 = 16,447` multiplication rounds (`PousP2Ring.rootProg_cost`); recomputing a block takes two sequential roots,
`2 · 16,447 = 32,894` (`recBlock_cost`). `D = 16,446` is the largest `D` at which that root program cannot finish one
root; at `16,447` a responder completes a root. The two roots are sequential (the second's input is `σ_K` of the
first), so even a root in fewer rounds leaves the attack about twice as far as `D`.

The timing. `pous-ring-speed` at this `D` says no allowed hardware finishes 16,447 sequential multiplication rounds
within `Δ + RTT ≤ 854 µs`. `ROOT_CALL_NS` (PROTOCOL.md, P2's SeqRoot) is half the fastest modelled root, the timing red
team's 361.55 ns per 16,448-bit squaring on one Zen 5 core cut by the measured 8-core cooperation speedup of 3.490×:
about 103.6 ns per round, so 16,447 rounds take at least 1,704 µs, 1.995 × 854 µs. The
PROTOCOL's 2.0001× counts the root's 20 extra multiplications, which a round of the ring model performs beside its
squaring. Under the red team's sync model (4.24× and 5.55× at 8 cores) the margin is 1.64× and 1.25×.

`L` is bounded because `P2RingAuditSecure` is not claimed for every `L` at `k = 105`: the truncation attack
(`PousP2Ring.trunc_seq_accepted`, which refutes `k ≤ 88` from `L = 2^30`) passes `k = 105` once it drops about 167
bits of every block, which takes about `2^168` ring operations per answer; `2^64` is far below that. -/
def EndToEndRingDeployed : Prop :=
  Pous.P2.Assumptions.P2RingAuditSecure 16446 (2 ^ 20) (2 ^ 64) →
    ∀ (session : ByteArray) (W : Bits sch.n) (A : Server S 16446 (2 ^ 20) (2 ^ 64)),
      pr (Passes session W A) ≤ ENNReal.ofReal Params.δ + Params.εMax

/-- **Non-vacuity in the ring model, with `ω` uniform: the honest full-state server passes P2's audit.** At the full
state (`FULL = 2^19 · 16,448` bits, more than `S`), a responder that keeps all of `C` and answers each index from it
with no query at all (`RingBounded 0 0 0`) is accepted by `seqAcceptsOn` with probability 1, for every `W`. -/
def HonestPassesRing : Prop :=
  S < FULL ∧ ∃ (A₁ : Preprocessor sch FULL) (A₂ : SeqResponderOn sch FULL 0 0 0),
    ∀ W, pr (seqAcceptsOn sch 105 W A₁ A₂) = 1

/-- **Non-vacuity of `EndToEndRing`'s event, in the ring model, with `ω` uniform**: the honest full-state server, a
`Server FULL 0 0 0`, passes the session with probability 1 for every 32-byte session id and `W`: the code always
builds the coin tree, and the draw never fails (`ε_rs = 0`, `DrawExact`). -/
def HonestPassesAccepts : Prop :=
  ∃ A : Server FULL 0 0 0, ∀ session : ByteArray, session.size = 32 → ∀ W : Bits sch.n,
    pr (Passes session W A) = 1

end Pous.EndToEnd
