import Properties.Pous.EndToEnd

/-!
# PoUS end to end at any point: the composed check and the ring game

`Pous.EndToEnd` states the composed check's property (`RunCheckAccepts`, `RunCertifies`) at `Run.DEPLOYED` and the ring
game (`Server`, `Passes`, `coinView`, `DrawExact`, `EndToEndRing`) at `2^19` blocks, because it fixes the deployed point
and P2's scheme there. Nothing else in them reads the point: the code's check (`Run.checkAt SCHEMES pt`), its sampler
(`FlockSoundness.Randomness.uniformOn N`), its coin tree (`CoinTree.make session N`) and the ring model take the point
or `N` as an argument. This module states both at every point and block count, for stores such as Demo B's,
`1038·2^16 ≈ 2^26.02` blocks.

* `AuditsSetupAt`, `ShapedAt`, `tableStringsAt`, `TableBreakAt`, `AcceptedAt`: their namesakes at a point `pt` (or its
  parameters); `RunCheckAcceptsAt` and `RunCertifiesAt`: `RunCheckAccepts` and `RunCertifies` at every point.

* `schAt N`, `SAt N`, `coinViewAt N`, `commitmentOfAt N`, `ServerAt N`, `PassesAt N`: `sch`, `S`, `coinView`,
  `commitmentOf`, `Server` and `Passes` with `N` in place of `2^19`; at `N = 2^19` each is its deployed-point namesake.
* `DrawAtMost`: the code's sampler favors no index at any `N`. At `2^19` it never rejects (`DrawExact`); at an `N` that
  is not a power of two it reads `width N` bytes per word and rejects a word with probability below `N / 2^(8·width N)`
  (`2^-70` at Demo B's `N`), and a rejection only lowers the chance of each index, so each still has at most `1/N`.
* `EndToEndRingAt`: at every `N`, every bound `c` on P2's sequential audit in the ring model at `N` (`seqAcceptsOn`)
  and every hiding bound `ε` on the coin commitments at `N` (`CoinCommitHides (coinViewAt N) ε`) give
  `pr (PassesAt N …) ≤ c + ε`. The sampler's term is `0` at every `N`, by `DrawAtMost`.
* `EndToEndRingOfRoundSecure`: from `P2RingRoundSecure` (`2^19 ≤ N ≤ 2^25`), the deployed budget `1% + 2^-128` at every
  `N` in that range, as `EndToEndRing` gives it at `2^19`.
* `EndToEndRingOfRoundSecureWide`, `EndToEndRingDemo`: the same from `P2RingRoundSecureWide` (`2^19 ≤ N ≤ 2^27`), and
  its instance at Demo B's `N = 1038·2^16`, with the state bound `⌊ρ·|C|⌋ = 1,060,008,348,186` bits.

The claims that carry the game to the deployed system are `Pous.EndToEnd`'s four (`pous-ring-device`,
`pous-ring-speed`, `pous-ring-instantiation`, `prf/sha-512`), read at `N`; none of them names a block count.
-/

open MemoryChallenger MemoryChallenger.Properties
open PousCheck PousCheck.Properties

namespace Pous.EndToEnd

open scoped ENNReal

/-! ### The composed check at any point -/

/-- The session audits the setup `pub` at the point `pt`: the setup published `pt`'s blocks under a 24-byte salt, and
the session audits every one of them in `pt.k` rounds, within `pt.limitNs`, with answers of `pt`'s block size, bound
to the published root; `AuditsSetup` at `Run.DEPLOYED`. -/
def AuditsSetupAt (pt : Run.Point) (pub : SetupProtocol.Public) (sc : SessionCommitment) : Prop :=
  pub.blocks = pt.params.blocks ∧ pub.salt.size = Run.SALT ∧ sc.blocks = pt.params.blocks ∧ sc.k = pt.k ∧
    sc.limitNs ≤ pt.limitNs ∧ sc.blockBytes = pt.params.blockBytes ∧ sc.binding = pub.root

/-- `C*` has the shape of the parameters `q`: `q.segments` segments of `q.blocksPerSegment` blocks of `q.blockBytes`
bytes, as `Decodes` requires; `Shaped` at `q = P2.Params.deployed`. -/
def ShapedAt (q : P2.Params) (C : Array (Array ByteArray)) : Prop :=
  C.size = q.segments ∧ ∀ s (hs : s < C.size), C[s].size = q.blocksPerSegment ∧
    ∀ v (hv : v < C[s].size), C[s][v].size = q.blockBytes

/-- What the run hashes for its two trees at the parameters `q`: each inner node and each leaf of the table's tree and
of `C*`'s, block `i` being block `i mod q.blocksPerSegment` of segment `⌊i / q.blocksPerSegment⌋`; `tableStrings` at
`q = P2.Params.deployed`. -/
def tableStringsAt (q : P2.Params) (pub : SetupProtocol.Public) (digests : ByteArray) (C : Array (Array ByteArray)) :
    List ByteArray :=
  let T := Run.rowLeaves q pub.salt digests
  let L := leaves pub.salt C
  nodeInputs T 0 T.size ++ nodeInputs L 0 L.size ++
    (List.range q.blocks).flatMap fun i =>
      [leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Run.row digests i),
        leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Merkle.rowDigest ((flat C)[i]!))]

/-- The table and `C*` hold a break at the parameters `q`: two distinct strings their trees hash with one SHA-512
digest (`cr/sha-512`); `TableBreak` at `q = P2.Params.deployed`. -/
def TableBreakAt (q : P2.Params) (pub : SetupProtocol.Public) (digests : ByteArray) (C : Array (Array ByteArray)) :
    Prop :=
  let xs := tableStringsAt q pub digests C
  ∃ x ∈ xs, ∃ y ∈ xs, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

/-- **`RunCheckAccepts` at every point.** Deterministic and assumption free: if the composed check at the point `pt`
(`Run.checkAt SCHEMES pt`, which `Run.check` runs at `Run.DEPLOYED`) accepts against the verifier's list `trusted`, the
setup's public outputs `pub`, the memory challenger's table `digests` and its session `r` under the id `sid`, then the
session audits `pub` at `pt`, runs under `sid` and is signed under a key in `trusted`; and for every device and every
`C*` of `pt`'s shape whose leaves have the published root, that key's holder ran a session with the session's
commitment in which every recorded answer is the challenged block of `C*`, on time on its own clock, or the run's own
data holds a SHA-512 collision or an LMS forgery under a key in `trusted`. -/
def RunCheckAcceptsAt : Prop :=
  ∀ (pt : Run.Point) (trusted : List ByteArray) (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session)
    (sid : ByteArray),
    Run.checkAt SCHEMES pt trusted pub digests r sid = .ok () →
    ∀ (dev : Challengers) (C : Array (Array ByteArray)),
      AuditsSetupAt pt pub r.commitment ∧ r.commitment.session = sid ∧ r.signed.publicKey ∈ trusted ∧
        (ShapedAt pt.params C → Merkle.root (leaves pub.salt C) = some pub.root →
          (∃ run ∈ dev.sessions r.signed.publicKey, run.commitment = r.commitment ∧ AnsweredOnTime run C) ∨
            RunBreak trusted r.signed dev C ∨ TableBreakAt pt.params pub digests C)

/-- **What a run must pass at the point `pt`**: setup accepts `C*` at `pt`'s parameters against the server's commitment
and salt and publishes `pub`, and the composed check at `pt` accepts; `Accepted` at `Run.DEPLOYED`. -/
def AcceptedAt (pt : Run.Point) (trusted : List ByteArray) (c : SetupProtocol.Commitment)
    (C : Array (Array ByteArray)) (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session)
    (sid : ByteArray) : Prop :=
  SetupProtocol.protocol.check pt.params c C = .accept pub () ∧ Run.checkAt SCHEMES pt trusted pub digests r sid = .ok ()

/-- **An accepted run certifies `C*` at every point**: `RunCertifies` with `pt` in place of `Run.DEPLOYED`. It follows
from `RunCheckAcceptsAt` and the setup check's `AcceptsDecodes` and `PublishesRoot`, which quantify over every parameter
set, all proved. -/
def RunCertifiesAt : Prop :=
  ∀ (pt : Run.Point) (trusted : List ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray) (dev : Challengers),
    AcceptedAt pt trusted c C pub digests r sid →
      DecodesToCommitment pt.params c C ∧ Published pt.params c C pub ∧ AuditsSetupAt pt pub r.commitment ∧
        r.commitment.session = sid ∧ r.signed.publicKey ∈ trusted ∧
        ((∃ run ∈ dev.sessions r.signed.publicKey, run.commitment = r.commitment ∧ AnsweredOnTime run C) ∨
          RunBreak trusted r.signed dev C ∨ TableBreakAt pt.params pub digests C)

/-! ### The ring game at any block count -/

/-- P2 in the ring model at `N` blocks of 16,448 bits, each carrying 16,432 payload bits; `sch` at `N = 2^19`. -/
noncomputable abbrev schAt (N : ℕ) [NeZero N] : Scheme (ringRO N 16448 (2 ^ 16448 - 21065)) :=
  p2ring N 16448 (2 ^ 16448 - 21065) 16432

/-- The state bound `⌊ρ·|C|⌋` in bits at `N` blocks, `|C| = N · 16,448` bits (`ρ = 18/19`); `S` at `N = 2^19`. -/
noncomputable abbrev SAt (N : ℕ) [NeZero N] : ℕ := (schAt N).stateBound Params.ρ

/-- What the memory challenger publishes about the indices `I` at `N` blocks before session `session`: the coin tree's
root and its hm96 key (`CoinTree.make session N`), or nothing when the code refuses to build the tree; `coinView` at
`N = 2^19`. -/
def coinViewAt (N : ℕ) (session : ByteArray) (c : HmCoins) (I : Fin 105 → Fin N) : Option (ByteArray × ByteArray) :=
  (CoinTree.make session N (Array.ofFn fun j => (I j).val) (coinKey c.1)
    (Array.ofFn fun j => bytesOf (c.2 j))).toOption.map fun t => (t.root, t.key)

/-- The session commitment the server sees at `N` blocks: the session id, the coin tree's root and key `v`, `N`
blocks, and the deployed point's rounds, limit and block size, bound to `binding`; `commitmentOf` at `N = 2^19`. -/
def commitmentOfAt (N : ℕ) (session binding : ByteArray) (v : ByteArray × ByteArray) : SessionCommitment :=
  ⟨session, v.1, v.2, N, K, LIMIT_NS, q.blockBytes, binding⟩

/-- **A server in the ring model for a scheme `sch`**: `Server`'s fields over any scheme in the ring model. `pre` is all
that survives setup, `S` bits, fixed before the memory challenger's coins are drawn; `step sc W pp j σ i` answers round
`j`'s index `i` from the state `σ`, what is public and the one session commitment `sc`, within `(D, Q, L)`. It is
stated over the scheme, not at `N`, because Lean's structure command unfolds `N * 16432` with `N` free. -/
structure RingServer {B w p : ℕ} [NeZero p] (sch : Scheme (ringRO B w p)) (S D Q L : ℕ) : Type where
  pre : Preprocessor sch S
  step : SessionCommitment → Bits sch.n → Bits sch.t → ℕ → Bits S → Fin sch.B →
    Prog (ringRO B w p) (Bits sch.ℓ × Bits S)
  bounded : ∀ sc W pp j σ i, (step sc W pp j σ i).RingBounded D Q L

set_option exponentiation.threshold 20000 in
/-- **A server in the ring model at `N` blocks**: `Server` with `schAt N` in place of `sch`. -/
abbrev ServerAt (N : ℕ) [NeZero N] (S D Q L : ℕ) : Type := RingServer (schAt N) S D Q L

/-- **The server passes the session at `N` blocks**: `Passes` with `N` in place of `2^19`. The code's sampler reads
index `I j` from stream `j` at `N` blocks (`uniformOn N`), the code builds the coin tree over those indices
(`coinViewAt N`), and the server, shown the session commitment, returns block `I j` of `C` in every one of the 105
rounds, sequentially. A stream from which the sampler draws no index fails the session. -/
def PassesAt (N : ℕ) [NeZero N] {S D Q L : ℕ} (session binding : ByteArray) (W : Bits (schAt N).n)
    (A : ServerAt N S D Q L) (x : (ringRO N 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) : Prop :=
  ∃ (I : Fin 105 → Fin N) (v : ByteArray × ByteArray),
    (∀ j, FlockSoundness.Randomness.uniformOn N (bytesOf (x.2.2 j)) = some (I j).val) ∧
      coinViewAt N session x.2.1 I = some v ∧
      seqAnswers ((ringRO N 16448 (2 ^ 16448 - 21065)).answer x.1)
          (A.step (commitmentOfAt N session binding v) W ((schAt N).enc W x.1 ()).1) 0 (List.ofFn I)
          (A.pre W ((schAt N).enc W x.1 ()).1 ((schAt N).enc W x.1 ()).2 x.1) =
        List.ofFn fun j => ((schAt N).enc W x.1 ()).2 (I j)

/-- **The draw favors no index at any block count (`ε_rs = 0` in the upper bound)**: for every positive block count
`N + 1` and every event `P` on indices, `uniformOn (N + 1)` on `DRAW_BYTES` uniform bytes returns an index in `P` with
probability at most `P`'s under the uniform index. At `2^19` it is equality (`DrawExact`); elsewhere a rejected word
only takes probability away from every index. -/
def DrawAtMost : Prop :=
  ∀ (N : ℕ) (P : Fin (N + 1) → Prop),
    pr (fun s : Bytes DRAW_BYTES =>
        ∃ i : Fin (N + 1), FlockSoundness.Randomness.uniformOn (N + 1) (bytesOf s) = some i.val ∧ P i) ≤ pr P

/-- **P2's audit in the ring model at any block count, with `ω` uniform.** For every block count `N`, every state
size `S` and `(D, Q, L)`, every bound `c` on P2's sequential audit at `N` (every `S`-bit preprocessing and
`(D, Q, L)`-bounded responder passes `seqAcceptsOn (schAt N) 105` with probability at most `c`) and every `ε` within
which the coin commitments at `N` hide the indices (`CoinCommitHides (coinViewAt N) ε`): for every session id and
binding, every payload `W` and every `S`-bit, `(D, Q, L)`-bounded server at `N` (`ServerAt`), the server passes the
session with probability at most `c + ε` over the primitive `ω`, the memory challenger's coins and its uniform
streams. The sampler adds nothing (`DrawAtMost`). It speaks of the deployed system through `Pous.EndToEnd`'s four
claims, read at `N`. -/
def EndToEndRingAt : Prop :=
  ∀ (N : ℕ) [NeZero N] (S D Q L : ℕ) (c ε : ℝ≥0∞),
    Pous.P2.Assumptions.CoinCommitHides (coinViewAt N) ε →
    (∀ (W : Bits (schAt N).n) (A₁ : Preprocessor (schAt N) S) (A₂ : SeqResponderOn (schAt N) S D Q L),
        pr (seqAcceptsOn (schAt N) 105 W A₁ A₂) ≤ c) →
      ∀ (session binding : ByteArray) (W : Bits (schAt N).n) (A : ServerAt N S D Q L),
        pr (PassesAt N session binding W A) ≤ c + ε

/-- **`EndToEndRing`'s budget at every block count `P2RingRoundSecure` covers.** If `P2RingRoundSecure p ε₀ D Q L` and
`(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130` (which `p = 0.043`, `ε₀ = 2^-137` meet,
`PousP2Rounds.floor_043_eps`), then at every `N` from `2^19` to `2^25` at which the coin commitments hide the indices up
to `2^-131`, every `⌊ρ·|C|⌋`-bit, `(D, Q, L)`-bounded server passes the session with probability at most `1% + 2^-128`
(`δ + ε_max`). At `N = 2^19` this is `EndToEndRing`'s bound, from the per-round conjecture rather than the whole-audit
one. -/
def EndToEndRingOfRoundSecure : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecure p ε₀ D Q L →
      ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 25 →
        Pous.P2.Assumptions.CoinCommitHides (coinViewAt N) ((2⁻¹ : ℝ≥0∞) ^ 131) →
        ∀ (session binding : ByteArray) (W : Bits (schAt N).n) (A : ServerAt N (SAt N) D Q L),
          pr (PassesAt N session binding W A) ≤ ENNReal.ofReal Params.δ + Params.εMax

/-- **`EndToEndRing`'s budget to `2^27` blocks, from the widened per-round floor.** `EndToEndRingOfRoundSecure` with
`P2RingRoundSecureWide` in place of `P2RingRoundSecure`: at every `N` from `2^19` to `2^27` at which the coin
commitments hide the indices up to `2^-131`, every `⌊ρ·|C|⌋`-bit, `(D, Q, L)`-bounded server passes the session with
probability at most `1% + 2^-128`. Rests on the open conjecture `P2RingRoundSecureWide`, unchecked in `N` above
`2^25`. -/
def EndToEndRingOfRoundSecureWide : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
      ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 27 →
        Pous.P2.Assumptions.CoinCommitHides (coinViewAt N) ((2⁻¹ : ℝ≥0∞) ^ 131) →
        ∀ (session binding : ByteArray) (W : Bits (schAt N).n) (A : ServerAt N (SAt N) D Q L),
          pr (PassesAt N session binding W A) ≤ ENNReal.ofReal Params.δ + Params.εMax

/-- **The demo's ring game**: `EndToEndRingOfRoundSecureWide` at Demo B's store, `N = 1038·2^16` blocks of 16,448
bits (139.7 GB of payload), whose state bound `⌊ρ·|C|⌋` is 1,060,008,348,186 bits. If the coin commitments at that
`N` hide the indices up to `2^-131`, every server holding at most that many bits between rounds, within `(D, Q, L)`,
passes the session with probability at most `1% + 2^-128`. Rests on `P2RingRoundSecureWide`, unchecked in `N` above
`2^25`. -/
def EndToEndRingDemo : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
      ∀ (N : ℕ) [NeZero N], N = 1038 * 2 ^ 16 →
        Pous.P2.Assumptions.CoinCommitHides (coinViewAt N) ((2⁻¹ : ℝ≥0∞) ^ 131) →
        ∀ (session binding : ByteArray) (W : Bits (schAt N).n) (A : ServerAt N 1060008348186 D Q L),
          pr (PassesAt N session binding W A) ≤ ENNReal.ofReal Params.δ + Params.εMax

end Pous.EndToEnd
