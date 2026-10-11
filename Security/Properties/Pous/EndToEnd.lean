import Properties.Pous
import Properties.Pous.MemoryChallenger
import Properties.Identity
import Models.Pous.Guarantees.P2
import Models.Pous.Assumptions.P2
import Definitions.Pous.P2.Cost

/-!
# PoUS end to end: what an accepted run certifies about the server's state (DRAFT, statements only)

**Draft for statement review, not a root.** Nothing here is proved and nothing is built by `lake build`: this module is
in no `roots` list, so `check`'s audit reports it as an orphan until it lands. Landing it needs a root
(`lakefile.toml`, `lean-audit.json`), `Models.Pous` and `Properties.Identity` in the `Properties.Pous` layer's imports,
a `shape.frozen` entry (`unstated: …`) for each statement until its proof lands, and a home for the four new
assumptions (see "Assumptions").

Two statements at P2's deployed point (`P2.Params.deployed`: `p = 2^16448 − 21065`, `B = 2^19` blocks of 2,056 bytes,
`k = 105` rounds, `Δ + RTT ≤ 854,000 ns`), over the Lean verdicts that judge a run: the setup check
(`SetupProtocol.protocol.check`) and the memory challenger's verdict (`MemoryChallenger.verdict`), joined by the
glue `Accepted` names.

* `Pous.Guarantees.RunCertifies`, deterministic, no assumption: an accepted run's `C*` decodes to the committed `W`, its
  root is published, and the memory challenger holding the trusted key ran a session in which every one of the 105
  answers it received was the challenged block of that `C*`, within `Δ + RTT` on its own clock; or the run's own data
  holds a SHA-512 collision or an LMS forgery.
* `Pous.Guarantees.EndToEnd`, probabilistic: for every `W` and every admissible server holding at most `⌊ρ·|C|⌋` bits
  (`ρ = 18/19`) between setup and audit, the run is accepted with no break with probability at most `1% + 2^-128`,
  over the setup salt and the memory challenger's coins.

The probabilistic statement crosses a model boundary that Lean cannot remove: every model in which P2 is secure hides
structure the deployed code computes with (the labels are `φ = id`, the keys SHA-512-CTR), so "the real P2 resists every
Lean function" is false (a server storing nothing recomputes `C*` locally, which a Lean function does for free). The
statement is therefore relative to a class `Adm` of servers (those realisable on the allowed hardware, answering within
the measured latency, under perfect isolation), and the two device assumptions say what that class can do. `Adm` is
a parameter: at `Adm = ⊥` both hold and `EndToEnd` says nothing, so its content is the claim that the deployment's
servers form a class the device assumptions hold for.

The game builds in three facts of the deployment rather than assuming them: the memory challenger runs the Lean code
(`trusted-nebius-host`), it runs one session under its key per audit (no verifier check enforces this yet), and the salt
and its coins are uniform (`uniform/io-getrandombytes`). `PousCheck.TimedAudit.verdict` does not enter: its challenge
source and elapsed times are the server's to supply.
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

/-- The ring-model scheme at the deployed point: P2's square–mask–square over `Z/p` with a random odd labeling,
`2^19` blocks of 16,448 bits, each carrying 16,432 payload bits (2,054 bytes). -/
noncomputable abbrev sch : Scheme (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)) :=
  p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432

/-- The state bound `⌊ρ·|C|⌋` in bits, `|C| = 2^19 · 16,448` bits, the same for the code's `C*` and the model's. -/
noncomputable def S : ℕ := sch.stateBound Params.ρ

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
both of the run, with one SHA-512 digest (`cr/sha-512`). -/
def CommitBreak (W : ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) : Prop :=
  ∃ W', Decodes q c.salt C W' ∧ W' ≠ W ∧ ∃ x y, commitInput W = some x ∧ commitInput W' = some y ∧
    Flock.Sha512.hash x = Flock.Sha512.hash y

/-! ## The composed acceptance -/

/-- The expected digests the session is judged against: `blockDigest` of block `i` of `C*`. -/
def expectedOf (C : Array (Array ByteArray)) (i : Nat) : Option ByteArray := ((flat C)[i]?).map blockDigest

/-- **What a run must pass**, the verifier's whole decision: setup accepts `C*` against the server's commitment and
salt and publishes `pub`; the memory challenger's verdict, under the deployed scheme (`SCHEMES`: LMS at `Lms.deployed`)
and the trusted keys, accepts the signed session against the expected digests of `C*`; and the session audits this
setup at the deployed point: every block of it, `k = 105` rounds, within `Δ + RTT`, blocks of the setup's size, bound to
the published root.

The last five conjuncts are checks no Lean function makes yet: they are the spec of the composed check that has to
exist (a `pous-run` check reading both records) before `RunCertifies` can be a property of one function. -/
def Accepted (trusted : List ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : SetupProtocol.Public) (s : SignedSession) (sc : SessionCommitment) (openings : Array CoinOpening) : Prop :=
  SetupProtocol.protocol.check q c C = .accept pub () ∧
    (verdict SCHEMES trusted (expectedOf C) s sc openings).accepted = true ∧
    sc.blocks = q.blocks ∧ sc.k = K ∧ sc.limitNs ≤ LIMIT_NS ∧ sc.blockBytes = q.blockBytes ∧ sc.binding = pub.root

/-! ## The game -/

/-- `n` bytes, each uniform when drawn. -/
abbrev Bytes (n : ℕ) : Type := Fin n → Fin 256

def bytesOf {n : ℕ} (x : Bytes n) : ByteArray := ⟨Array.ofFn fun i => (x i).val.toUInt8⟩

/-- The setup salt the verifier draws after the server commits to `W`: 24 bytes (`Keys.SALT_BYTES`). -/
abbrev Salt : Type := Bytes 24

/-- The memory challenger's coins for one session: its secret source, its fresh hm96 key and one 192-byte salt per
round (`memory_challenger.CoinTree.draw`, from the OS). -/
abbrev DevCoins : Type := Bytes 32 × Bytes 256 × (Fin K → Bytes 192)

/-- The fresh hm96 key: 256 bytes, bit 2047 cleared (`HM96.check_key`). -/
def coinKey (k : Bytes 256) : ByteArray :=
  let b := bytesOf k
  b.set! 255 (b.get! 255 &&& 0x7f)

/-- The memory challenger's draw: `CoinTree.draw` of its source over the setup's blocks, `k = 105` coins. -/
def drawTree (session : ByteArray) (d : DevCoins) : Except String CoinTree :=
  CoinTree.draw (bytesOf d.1) session q.blocks K (coinKey d.2.1) (Array.ofFn fun j => bytesOf (d.2.2 j))

/-- What the memory challenger publishes before the session (`Run.commitment` of its run). -/
def commitmentOf (tree : CoinTree) (binding : ByteArray) : SessionCommitment :=
  ⟨tree.session, tree.root, tree.key, tree.blocks, tree.k, LIMIT_NS, q.blockBytes, binding⟩

/-- A server holding `S` bits between setup and audit, deterministic (a randomized one is an average of these):

* `encode W salt` is the `C*` it hands setup, any `C*` (setup judges it);
* `pre W salt C` is all that survives setup, `S` bits (perfect isolation);
* `step W salt sc j σ i` answers round `j`'s index `i` from the state `σ` and what is public (`W`, the salt, the session
  commitment), returning the answer, the next state, and the latency the memory challenger's clock measures between
  revealing `i` and receiving the answer.

`latency` is the server's to report here: what ties it to the computation is `Adm`, the device model. -/
structure Prover (S : ℕ) : Type where
  encode : ByteArray → ByteArray → Array (Array ByteArray)
  pre : ByteArray → ByteArray → Array (Array ByteArray) → Bits S
  step : ByteArray → ByteArray → SessionCommitment → ℕ → Bits S → ℕ → ByteArray × Bits S × ℕ

/-- The memory challenger's observations of rounds `j, j+1, …` over the indices `is`, from clock reading `t` and state
`σ`: each index is revealed as the previous answer arrives. -/
def observe {S : ℕ} (step : ℕ → Bits S → ℕ → ByteArray × Bits S × ℕ) : ℕ → ℕ → List ℕ → Bits S → List Observed
  | _, _, [], _ => []
  | j, t, i :: is, σ =>
    let out := step j σ i
    ⟨t, t + out.2.2, out.1⟩ :: observe step (j + 1) (t + out.2.2) is out.2.1

/-- The session the memory challenger runs against `τ` (the code's own `Recorded` fold reads it). -/
def run {S : ℕ} (τ : Prover S) (W salt : ByteArray) (tree : CoinTree) (binding : ByteArray) : Run :=
  ⟨tree, LIMIT_NS, q.blockBytes, binding,
    observe (τ.step W salt (commitmentOf tree binding)) 0 0 tree.indices.toList (τ.pre W salt (τ.encode W salt))⟩

/-- The deployed memory challenger: one session under its key `pk` in this audit, none under any other. -/
def device {S : ℕ} (τ : Prover S) (W salt : ByteArray) (tree : CoinTree) (binding pk : ByteArray) : Device :=
  ⟨fun k => if k == pk then [run τ W salt tree binding] else []⟩

/-- **The event**: setup and the memory challenger's verdict accept a record of the audit (any signed session, coin
openings and published root), and the run's data holds no break. -/
def Event {S : ℕ} (τ : Prover S) (W session pk : ByteArray) (x : Salt × DevCoins) : Prop :=
  let salt := bytesOf x.1
  let C := τ.encode W salt
  ∃ cw tree pub s sc openings,
    Commit.commit q W = .ok cw ∧ drawTree session x.2 = .ok tree ∧
      Accepted [pk] ⟨cw, salt⟩ C pub s sc openings ∧
      ¬ (RunBreak [pk] s (device τ W salt tree pub.root pk) C ∨ CommitBreak W ⟨cw, salt⟩ C)

/-! ## The two games the assumptions compare -/

/-- `τ` passes on the indices `is`: its `C*` decodes to `W` under the salt, and each of the `k` answers is the
challenged block of `C*`, within `Δ + RTT`. -/
def Pass {S : ℕ} (τ : Prover S) (W salt : ByteArray) (sc : SessionCommitment) (is : List ℕ) : Prop :=
  let C := τ.encode W salt
  let obs := observe (τ.step W salt sc) 0 0 is (τ.pre W salt C)
  Decodes q salt C W ∧ obs.length = K ∧
    ∀ j (hj : j < obs.length), (flat C)[is[j]!]? = some obs[j].block ∧ obs[j].receiptNs ≤ obs[j].sendNs + LIMIT_NS

/-- The root the session is bound to: `C*`'s published root (empty when `C*` has no block). -/
def rootOf (salt : ByteArray) (C : Array (Array ByteArray)) : ByteArray :=
  (Merkle.root (leaves salt C)).getD .empty

/-- The real game: the indices are the memory challenger's own, derived from its source, and the server sees their
commitment. -/
noncomputable def realPr {S : ℕ} (τ : Prover S) (W session : ByteArray) : ℝ≥0∞ :=
  pr fun x : Salt × DevCoins =>
    let salt := bytesOf x.1
    ∃ tree, drawTree session x.2 = .ok tree ∧
      Pass τ W salt (commitmentOf tree (rootOf salt (τ.encode W salt))) tree.indices.toList

/-- The ideal game: the server sees the same commitment, but the indices are uniform over the `2^19` blocks and
independent of it. -/
noncomputable def unifPr {S : ℕ} (τ : Prover S) (W session : ByteArray) : ℝ≥0∞ :=
  pr fun x : Salt × DevCoins × (Fin K → Fin (2 ^ 19)) =>
    let salt := bytesOf x.1
    ∃ tree, drawTree session x.2.1 = .ok tree ∧
      Pass τ W salt (commitmentOf tree (rootOf salt (τ.encode W salt))) (List.ofFn fun j => (x.2.2 j).val)

end Pous.EndToEnd

/-! ## Assumptions

Two new named assumptions, each a `Prop` taken as a hypothesis, beside `P2RingAuditSecure` (in
`Models.Pous.Assumptions.P2`). `DeviceIsRingResponder` and `CoinsHide` read the code's decoder and draw, so they need an
assumptions module on the `security_properties` side, which has none yet. -/

namespace Pous.P2.Assumptions

open Pous.EndToEnd
open scoped ENNReal

/-- **NEW, device model (`generic-ring`, `ring-mul-floor`, `keccak-rate-ceiling`, `ring-op-ceiling`,
`perfect-isolation`, `random-oracle` for the SHA-512-CTR key expander).** Every admissible server's chance in the
ideal-index game is at most that of some `S`-bit, `(D, Q, L)`-bounded responder against P2 in the ring model: answering
within `Δ + RTT` on the allowed hardware buys at most `D` sequential multiplications, `Q` key queries and `L` ring
operations, and computing on the integers' representation (`φ = id`) and on SHA-512-CTR keys gains nothing over the
random labeling and random keys. The deadline-to-`(D, Q, L)` conversion is this assumption's, from the certificate's
numbers: Lean never sees a nanosecond. -/
def DeviceIsRingResponder (Adm : Prover S → Prop) (D Q L : ℕ) : Prop :=
  ∀ τ, Adm τ → ∀ (W session : ByteArray), W.size = q.wBytes →
    ∃ (W' : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponderOn sch S D Q L),
      unifPr τ W session ≤ pr (seqAcceptsOn sch K W' A₁ A₂)

/-- **NEW, the challenge draw (`prf/sha-512` for `Randomness.derive` keyed by the source, `statistical-hiding` and
`hash-derived-key` for the hm96 coin commitments).** For every admissible server and every session id, deriving the
indices from the memory challenger's secret source rather than drawing them uniformly, independent of the commitment the
server sees, raises its chance of passing by at most `2^-131`. Relative to `Adm`: an unbounded server could search a
short source. -/
def CoinsHide (Adm : Prover S → Prop) : Prop :=
  ∀ τ, Adm τ → ∀ (W session : ByteArray), realPr τ W session ≤ unifPr τ W session + (2⁻¹ : ℝ≥0∞) ^ 131

end Pous.P2.Assumptions

/-! ## The statements -/

namespace Pous.Guarantees

open Pous.EndToEnd Pous.P2.Assumptions
open scoped ENNReal

/-- **An accepted run is the device's honest record of `C*`'s blocks, or carries a break.** Deterministic and assumption
free: for every device, if setup and the memory challenger's verdict accept a run (`Accepted`), then `C*` decodes to a
payload with the server's commitment, setup published its root, and the trusted key's holder ran a session with the
accepted commitment in which every recorded answer is the challenged block of `C*`, on time on its own clock; or the
run's own data holds a SHA-512 collision or an LMS forgery. -/
def RunCertifies : Prop :=
  ∀ (trusted : List ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public)
    (s : SignedSession) (sc : SessionCommitment) (openings : Array CoinOpening) (dev : Device),
    Accepted trusted c C pub s sc openings →
      DecodesToCommitment q c C ∧ Published q c C pub ∧
        ((∃ r ∈ dev.sessions s.publicKey, r.commitment = sc ∧ AnsweredOnTime r C) ∨ RunBreak trusted s dev C)

/-- **PoUS end to end at P2's deployed point.** For every class `Adm` of servers the device assumptions hold for, every
`(D, Q, L)` at which P2 is secure in the ring model, every session id and memory challenger key: for every payload `W`
the server commits to and every admissible server holding at most `⌊ρ·|C|⌋` bits between setup and audit, fixed before
the salt and the coins are drawn, the run is accepted with no break with probability at most `1% + 2^-128`, over the
setup salt and the memory challenger's coins. -/
def EndToEnd : Prop :=
  ∀ (Adm : Prover S → Prop) (D Q L : ℕ) (session pk : ByteArray),
    DeviceIsRingResponder Adm D Q L → CoinsHide Adm → P2RingAuditSecure D Q L →
    ∀ (W : ByteArray), W.size = q.wBytes → ∀ τ : Prover S, Adm τ →
      pr (Event τ W session pk) ≤ ENNReal.ofReal Params.δ + Params.εMax

end Pous.Guarantees
