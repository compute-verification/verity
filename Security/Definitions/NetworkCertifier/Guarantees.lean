import Definitions.NetworkCertifier
import Definitions.NetworkCertifier.Assumptions
import Mathlib.Data.Sym.Card
import Mathlib.Data.Set.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The network certifier's guarantees

Each guarantee is a `def G : Prop`, stated over `Protocol`'s definitions with the assumptions it rests on as hypotheses
(`Assumptions`). Its proof is the theorem `NetTiming.SecurityProofs.G : NetTiming.Guarantees.G`, and a guarantee without
one is open: nothing may cite it until it is proved. The statements that are only steps of these proofs, Theorem 1(b)'s
counting bound among them, are in `Proofs.NetworkCertifier.Lemmas`, under the same names.

`docs/network-transparency/timing-channel.md`: Proposition 2 (§2.3), Theorem 1(b) (§2.2), Theorem 3 (§2.4), and the
ingress bound (§2.5, "Ingress content"; §3.2).
-/

namespace NetTiming.Guarantees

open Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-! ## Proposition 2: timing is occupancy, its counts, and tightness -/

/-- Free slots: the sender addresses slots, and there are `2^(T·S)` occupancies. -/
def CardFree : Prop :=
  ∀ (T S : ℕ), Fintype.card (Fin T → Fin S → Bool) = 2 ^ (T * S)

/-- Left-packed: `(S + 1)^T` occupancies. -/
def CardLeftPacked : Prop :=
  ∀ (T S : ℕ), Fintype.card {P : Fin T → Fin S → Bool // LeftPacked P} = (S + 1) ^ T

/-- **Fixed M, all released:** at most `binom(T + M − 1, M)` count vectors, which is the doc's `binom(M+T−1, T−1)`
for `T ≥ 1`. -/
def CardCountsEqLe : Prop :=
  ∀ (T S M : ℕ),
    (univ.filter fun n : Fin T → Fin (S + 1) => ∑ t, (n t : ℕ) = M).card ≤ (T + M - 1).choose M

/-- **Fixed M, at the certifier.** When the sender enqueues at most `M` frames over the window, the FIFO certifier can release
at most `binom(T + M, M)` distinct grids: the doc's `binom(M+T, T)`. -/
def EncardFifoGridsLeOfAvailable : Prop :=
  ∀ {F : Type} (T S M : ℕ) (f : ℕ → F),
    {g | ∃ enq : ℕ → ℕ, ∑ t ∈ Finset.range T, enq t ≤ M ∧ g = fifoGrid T S f enq}.encard ≤
      ((T + M).choose M : ℕ)

/-- **Tightness.** Over all enqueue schedules, with the frames leaving in queue order `f`, the FIFO certifier can release
exactly `(S + 1)^T` distinct grids. So `T·log2(S+1)` bits are attained with zero error. -/
def EncardFifoGrids : Prop :=
  ∀ {F : Type} (T S : ℕ) (f : ℕ → F),
    (Set.range fun enq : ℕ → ℕ => fifoGrid T S f enq).encard = ((S + 1) ^ T : ℕ)

/-! ## Theorem 3: the advice caps -/

/-- **Theorem 3, the fixed advice budget.** This is the `2^A` cap with the advice type `Fin E → Fin k → Bool`: one
string of exactly `k` bits for each of `E` execution blocks, padded to that length whatever is used. The accepted
observation sequences of length `n` with jitter below `D` number at most `2^(E·k) · D^n`, that is `k` bits per
execution block plus jitter. Every advice value has the same length, so the charge is `k` whatever an honest session
used, and nothing used is counted: no cap per `T` buckets, no Kraft inequality and no `BitsCount`. -/
def EncardAccSeqsLeAdviceBits : Prop :=
  ∀ {L K S : ℕ} {In : Type} (E k : ℕ) (P : Program L S In (Fin E → Fin k → Bool))
    (op : ℕ → List (Obs L K S) → In) (n D : ℕ),
    (AccSeqs P op n D).encard ≤ ((2 ^ (E * k) * D ^ n : ℕ) : ℕ∞)

/-! ## Theorem 3 for the reference schedule: the constant-rate Program

`constantRate T g E frames` (`Definitions.NetworkCertifier.Schedule`) is the Program of one link's monitoring period:
`E` execution blocks, each of `g·T` buckets, each block's frames and send ticks `frames ins e v ℓ` read from the
ingress and that block's advice `v`, a string of `k = |a|` bits. None of these guarantees has a dataflow
hypothesis. The counts are over the monitoring period's first `n` steps, within its `E` blocks (`hn : n ≤ E·(g·T)`):
the audit gives verdicts for those blocks only, so every step counted is a step it checks. -/

/-- **Theorem 3 for the reference schedule.** The constant-rate Program's accepted observation sequences of length `n`
with jitter below `D` number at most `2^(E·k) · D^n`: `|a|` bits per execution block plus jitter. This holds for every
frame function and operator. -/
def EncardAccSeqsLeConstantRate : Prop :=
  ∀ {L K S : ℕ} {In : Type} (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D : ℕ) (hn : n ≤ E * (g * T)),
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤ ((2 ^ (E * k) * D ^ n : ℕ) : ℕ∞)

/-- **A step's rows read only its execution block's advice.** Two advice values that agree on the block holding step
`t`, `t / (g·T)`, give the same rows at `t`: the block's frames read only its own advice, and the FIFO starts empty at
the block's first bucket. The advice is any value, in a registered set or not, so advice declared for a later block
never changes an earlier block's verdict. -/
def ExpectedRowCausal : Prop :=
  ∀ {L S : ℕ} {In Adv : Type} (T g E : ℕ) (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (t : ℕ)
    (ins : List In) (a a' : Fin E → Adv) (ha : ∀ e : Fin E, (e : ℕ) = t / (g * T) → a e = a' e),
    (constantRate (S := S) T g E frames).rows t ins a = (constantRate (S := S) T g E frames).rows t ins a'

/-- **`|a|` bits per execution block opened.** If the monitoring period's first `n` steps lie within its first `m`
execution blocks (`n ≤ m·(g·T)`), the constant-rate Program's accepted sequences number at most
`2^(min(E, m)·k) · D^n`: only the advice of blocks those steps reach counts, however many blocks the advice covers. -/
def EncardAccSeqsLeConstantRateWindows : Prop :=
  ∀ {L K S : ℕ} {In : Type} (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hn : n ≤ E * (g * T)) (hm : n ≤ m * (g * T)),
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤ ((2 ^ (min E m * k) * D ^ n : ℕ) : ℕ∞)

/-- **The composed statement for the reference schedule (W2 and J).** A decoder reads the wire of the first `n` steps,
within the first `m` execution blocks, and recovers every secret in `Zs`, all of whose runs the constant-rate Program
accepts up to `n`. Then `Zs` has at most `2^(min(E, m)·k) · D^n` elements: a covert sender inside X moves at most
`m·|a| + n·log2 D` bits through egress timing and content advice together, and under J only the advice's. -/
def EncardDecodableLeConstantRate : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire : Type}
    (I : Interaction L K S Z In Act Wire) (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (n D m : ℕ) (hn : n ≤ E * (g * T))
    (hm : n ≤ m * (g * T))
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T g E frames) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ ((2 ^ (min E m * k) * D ^ n : ℕ) : ℕ∞)

/-- **`|a|` bits per execution block, from any block's start.** Fix what a run showed through its first `e₀` execution
blocks, `p`, and take the `m` blocks after them, within the `E` blocks (`hn` is
`EncardAccSeqsLeConstantRateWindows`'s `n ≤ E·(g·T)` for the monitoring period's first `n = (e₀ + m)·(g·T)` steps). The
constant-rate Program's accepted sequences of that length with jitter below `D` that continue `p` number at most
`(2^k · D^(g·T))^m`: `|a| + g·T·log2 D` bits per execution block, `|a|` under J (`D = 1`), whatever came before. At
`e₀ = 0` they are the monitoring period's first `m` blocks. -/
def EncardAccSeqsLeConstantRatePerWindow : Prop :=
  ∀ {L K S : ℕ} {In : Type} (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (e₀ m D : ℕ) (hn : (e₀ + m) * (g * T) ≤ E * (g * T)) (p : List (Obs L K S)),
    {o | o ∈ AccSeqs (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) D ∧
        o.take (e₀ * (g * T)) = p}.encard ≤ (((2 ^ k * D ^ (g * T)) ^ m : ℕ) : ℕ∞)

/-- **`|a|` bits per execution block through the wire, from any block's start (W2 and J).** Take secrets whose runs
show the same observations `p` through the first `e₀` execution blocks, each accepted through the `m` blocks after
them. A decoder of the wire through those blocks recovers at most `(2^k · D^(g·T))^m` of them: whatever the first `e₀`
blocks showed, a covert sender inside X moves at most `m·(|a| + g·T·log2 D)` more bits through egress timing and
content advice over the next `m` blocks. At `e₀ = 0` they are the monitoring period's first `m` blocks. -/
def EncardDecodableLeConstantRatePerWindow : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire : Type}
    (I : Interaction L K S Z In Act Wire) (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (e₀ m D : ℕ)
    (hn : (e₀ + m) * (g * T) ≤ E * (g * T))
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z) (p : List (Obs L K S)) (hpre : ∀ z ∈ Zs, I.obsRun z (e₀ * (g * T)) = p)
    (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T g E frames) I.op ((e₀ + m) * (g * T))
      (I.obsRun z ((e₀ + m) * (g * T))))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z ((e₀ + m) * (g * T))) = z),
    Zs.encard ≤ (((2 ^ k * D ^ (g * T)) ^ m : ℕ) : ℕ∞)

/-! ## The ingress bound: what the outside can deliver into X through timing -/

/-- **The ingress timing bound, from the records.** As `EncardIngressDecodableLe` below, but the decoder inside X reads
the records of the first `n` steps, everything X's devices (`op`) read, rather than the wire. This needs no W2, and the
wire version follows from it. -/
def EncardIngressObsDecodableLe : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ m ∈ Ms, dec (I.obsRun m n) = m),
    Ms.encard ≤ (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **The ingress timing bound.** Read the interaction inward (`NetTiming.Protocol.Ingress`): the secret `m` is the
outside's message, `snd m` chooses when ingress frames arrive and reacts to every record (and so to X's egress), `op`
is X's devices, the record's `rows` are what the `L` ingress links release into X, and `wire` is everything that enters
X. Take messages `Ms` whose runs keep every status complete up to `n`, and whose frames on each link `ℓ`, in the order
released, are a prefix of the prescribed content `c ℓ` (the Program's input, charged as reads). If a decoder inside X
reads the wire and recovers every message in `Ms`, then under W2, W4 with allocation `r ℓ`, and J, there are at most
`(∏_ℓ (r ℓ + 1))^n · D^n` of them: `n · Σ_ℓ log2(r ℓ + 1)` bits of timing, which is `T·log2(r_in + 1)` per ingress
link-window, and `n·log2 D` of jitter, none under J's `D = 1`. -/
def EncardIngressDecodableLe : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List Wire → Msg) (hdec : ∀ m ∈ Ms, dec (I.wireRun m n) = m),
    Ms.encard ≤ (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **Content plus timing, from the records.** As `EncardIngressObsDecodableLe`, but the content is no longer
prescribed: each message's frames on every link `ℓ` are a prefix of `c ℓ` for some content assignment `c` in `Cs`.
Then there are at most `#Cs · (∏_ℓ (r ℓ + 1))^n · D^n` messages: `log2 #Cs` bits of content plus the timing bound. -/
def EncardIngressObsDecodableLeContents : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ m ∈ Ms, dec (I.obsRun m n) = m),
    Ms.encard ≤ Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **Content plus timing.** As `EncardIngressDecodableLe` (the decoder reads the wire, under W2), but each message's
frames follow some content assignment in `Cs`: at most `#Cs · (∏_ℓ (r ℓ + 1))^n · D^n` messages. -/
def EncardIngressDecodableLeContents : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List Wire → Msg) (hdec : ∀ m ∈ Ms, dec (I.wireRun m n) = m),
    Ms.encard ≤ Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **The K charge's unit: `T·log2(r ℓ + 1)` bits per ingress link-window.** `EncardIngressObsDecodableLe` over the
monitoring period's first `m` whole windows of `T` buckets, its first `n = m·T` steps: for one content assignment `c`,
at most `(∏_ℓ (r ℓ + 1)^T · D^T)^m` messages. Each link-window contributes the factor `(r ℓ + 1)^T`, its
`T·log2(r ℓ + 1)` bits, and each window `D^T` of jitter, 1 under J. -/
def EncardIngressObsDecodableLePerWindow : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (T m D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg),
    Ms.encard ≤ ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞)

/-- **Content plus timing, per link-window.** `EncardIngressObsDecodableLeContents` over the first `m` whole windows of
`T` buckets: at most `#Cs · (∏_ℓ (r ℓ + 1)^T · D^T)^m` messages, `log2 #Cs` bits of content plus
`T·Σ_ℓ log2(r ℓ + 1)` bits of timing and `T·log2 D` of jitter per window. -/
def EncardIngressObsDecodableLeContentsPerWindow : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (T m D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg),
    Ms.encard ≤ Cs.encard * ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞)

/-!
## The schedule's declared bound and the coins

This section fixes the two terms of the charge that the capacity bounds leave open: what the advice's content part
counts and the coins (D7). Nothing in it is implemented yet. The numbers are at the llama8b-node1 calibration
(`calibrations/llama8b-node1.json`): T = 600 buckets of 100 ms, B = 4,096, one egress link at r = 1,686, one ingress
link (`requests`) at r_in = 34 and τ_in = 100 ms, and J (D = 1).

### What the advice counts

`constantRate T g E frames` takes one advice value per execution block, and `frames ins e v ℓ` gives every egress frame
of block `e` (its payload, send tick and canonical position) from the ingress `ins` and the block's advice `v`, a string
of exactly `|a|` bits. Acceptance is existential: a run is accepted if some advice value explains it (`AcceptedUpTo`).
The proof (`encard_accSeqs_le_advice`) maps each accepted sequence to an advice value that accepts it, plus its jitter
code. So each execution block counts `2^|a|`, whatever value was chosen and whenever: B's codec decides how the bits are
spent, on send ticks, on content decisions or on anything else, and the charge is `|a|` either way. The number means
something only if `|a|` and B are registered before the session, independently of the sender's secret, and the audit
refuses advice of any other length.

For a served Program the content part of the advice is the schedule: the per-step batch composition from which the
circuit is built. The schedule reaches the egress frames through their canonical order, which is step-major, through
how many tokens each event holds, and through any batch-variant numerics. Today the schedule is whatever the workload
manifest declares: each request's `arrive_step` and the engine's scheduler settings. A monotone map from buckets to
steps that is free at up to s_max steps per bucket is worth `T·log2(s_max + 1)` bits per `T` buckets, which `|a|`
would have to hold: 2,400 bits at s_max = 15. The declaration below makes it a few bits.

### The declaration

- **A deterministic scheduler.** The pinned engine environment includes its scheduler. Given a capacity profile p and
  each request's admission step, the scheduler fixes the per-step schedule: `s = Sched_E(p, arrivals)`. Bit-exact
  replay already relies on this, and `verity_vllm` compares each request's realized arrival step with the declared one.
- **Admission on a step clock.** The profile sets κ, a whole number of engine steps per bucket. Request q is admitted
  at step `κ · anchor(q)`. Its anchor is the bucket in which the ingress certifier released its last frame, read from
  the ingress certificate. Requests released in one bucket are admitted in the ingress release order. The engine never
  starts step κ·b before the releases of bucket b are in. When it runs slower, it lags, and a lag past the send ticks
  the advice gives is an honest miss. Given p, the schedule is then a function of the ingress certificates, and p is
  the only free choice. `verity_vllm.engine.arrivals` already admits each request right before engine step
  `arrive_step`, on the step counter and never on wall time. The declaration sets `arrive_step = κ · anchor`. The
  module's rule that an idle engine admits the next arrival at once is itself a function of the schedule.
- **A registered list of profiles.** `P_cap = (p_0, …, p_{N−1})`. Each profile holds κ and every scheduler setting the
  schedule reads: the batch caps (sequences and tokens per step), the chunked-prefill threshold, the KV block count,
  which fixes preemption, and the output lag of asynchronous scheduling. The list is a public parameter of the audit,
  registered with `|a|`, (T, r, B, τ_b, Q), r_in and τ_in before the audit's first execution phase and before any
  coins are flipped. Like `|a|`, it is the deployment's structure.
  - If the developer wants the profiles hidden, the list is a frame-v3-sha512 root over hm96-sha512 rows, one per
    profile, with N public. A sender inside X could then pick the list itself, so a hidden list also costs its own size
    once per registration: at most N·b_p bits for b_p-bit rows. A public list costs nothing.
- **The profile in use.** Its index i < N goes under the schedule root, with the per-step schedule rows. That root is
  a frame-v3-sha512 tree of hm96-sha512 rows (circuits' X3), hidden, and registered before any coin that depends on it.
  The registration statement proves that the rows are `Sched_E(p_i, arrivals)` for the arrivals the ingress
  certificates give, and that i < N. The count holds whenever i was chosen, since acceptance takes any value of the
  advice. The rule that the root precedes the coins is there for the sampled proofs, so that the circuit can't be shaped
  around the sample.

**The count.** The profile's index is `log2 N` of each execution block's `|a|` bits, read by B's codec beside the send
ticks. `SecurityProofs.EncardAccSeqsLeConstantRate` counts it with the rest of the advice, so no new guarantee is
needed.

**At the recorded parameters.** The recordings use one scheduler configuration. κ needs a grid that covers the honest
step rate, and κ has none. Taking 8 values of κ gives N = 8, 3 bits of each execution block's `|a|`. A codec that spends
7 bits per `T` buckets on send ticks, as an offset and a rate from 16 × 8 values, then needs `|a| = 7g + 3`: 17 bits
at g = 2, which is 12,240 bits a day of 720 execution blocks.

### The coins (D7)

The challenger's coins reach the developer's site on a certified link, so they are ingress. Their timing is charged on
the ingress grid, and their content is exempt.

- **Timing.** The coin link is one more ingress link, at τ_in and an allocation r_coin. It adds a factor
  `(r_coin + 1)^T` per window to the ingress count, which is `T·log2(r_coin + 1)` bits per link-window, the K charge's
  unit (`SecurityProofs.EncardIngressObsDecodableLePerWindow`). r_coin isn't calibrated. At r_coin = 1, one 4 KiB frame
  per bucket, it is 600 bits per link-window (10 bit/s): 864,000 bits a day, 0.0095 of K = 9.12e7. That is on top of
  the `requests` link's 3,077.6 bits per link-window (51.3 bit/s, 0.0486 of K a day), for 0.0581 of K a day in all.
- **Content: 0 bits.** `SecurityProofs.EncardIngressObsDecodableLeContentsPerWindow` charges `log2 #Cs` for the
  content, where `Cs` is one set of content assignments covering every message counted. Conditioned on the
  challenger's coins, which is how the model treats a randomized outside party, the coin link's content is one fixed
  stream `c_coin`: the challenger's coin tree, read in order. So `Cs = Cs_requests × {c_coin}`, and the coins add
  `log2 1 = 0` to the content. Charged as reads, they would cost up to 8·B·r_coin/τ_in = 327,680 bit/s.
- **Why the exemption is sound.**
  - The coins are the challenger's. No outside party can choose their content: every accepted frame on the coin link
    opens against the challenger's committed coin root. This is a declared item of the consistency relation between
    the circuit and the network certificates, and any other frame on the link is unexplained, so the window is
    rejected.
  - They are fixed before the developer sees them. The coin tree is committed before the audit, so no coin depends on
    anything X sends, and `c_coin` is the same for every message the bound counts.
  - Only the count is free. The frames on the coin link are a prefix of `c_coin`, in order, since a retry never
    redraws. How far the prefix has reached at each bucket is timing, and the grid charges it. Which coins the firewall
    asks for, and when, is egress, which the egress grid charges.
- **What it doesn't cover.** The exemption is for coin openings only. Anything else the challenger sends on the link
  must be fixed by the public parameters, or it is charged as content.
-/

end NetTiming.Guarantees
