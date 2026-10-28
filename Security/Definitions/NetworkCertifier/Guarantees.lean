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

/-- **Theorem 3, the clock-sync form.** This is the `2^A` cap with the advice type `Ac × (Fin W → Sync)`: one
clock-sync value per window from `Sync`, for example an offset δ and a rate ρ, 16 × 8 values. Over `W` windows the
accepted sequences number at most `#Ac · #Sync^W · D^n`, that is `A_c + W·log2 #Sync` bits plus jitter.

Instantiated for constant-rate release, this bounds the timing channel when the Program's only timing advice is the
clock sync. The shipped schedule is such a Program by its definition (`constantRate`), and
`EncardAccSeqsLeConstantRate` states this bound for it. A sync alphabet that varies by window is covered by
`encard_accSeqs_le_advice` directly. -/
def EncardAccSeqsLeSync : Prop :=
  ∀ {L K S : ℕ} {In Ac Sync : Type} [Fintype Ac] [Fintype Sync] (W : ℕ)
    (P : Program L S In (Ac × (Fin W → Sync))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ),
    (AccSeqs P op n D).encard ≤ ((Fintype.card Ac * Fintype.card Sync ^ W * D ^ n : ℕ) : ℕ∞)

/-- **A fixed clock.** With the clock-sync value fixed (`Sync = Unit`), the accepted sequences number at most
`#Ac · D^n`. Instantiated for constant-rate release under a fixed logical schedule, timing carries nothing beyond the
jitter, and nothing at all under J (`D = 1`): the zero-capacity theorem, with content advice. -/
def EncardAccSeqsLeFixedClock : Prop :=
  ∀ {L K S : ℕ} {In Ac : Type} [Fintype Ac] (W : ℕ)
    (P : Program L S In (Ac × (Fin W → Unit))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ),
    (AccSeqs P op n D).encard ≤ ((Fintype.card Ac * D ^ n : ℕ) : ℕ∞)

/-! ## Theorem 3 for the shipped schedule: the constant-rate Program

`constantRate T W ss frames` (`NetTiming.Protocol.Schedule`) is the Program of `schedule.py`: windows of `T` buckets,
`W` of them with advice, one clock-sync value of `ss` per window, and the Program's frames, anchors and logical indices
`frames ins c ℓ` read from the ingress and the content advice `c`. None of these guarantees has a dataflow hypothesis.
The four counts take a horizon within the windows with advice (`hn : n ≤ W * T`): the audit (`Audit.check_link`) gives
verdicts for those `W` windows only, so every step counted is a step it checks. -/

/-- **Theorem 3's clock-sync form, for the shipped schedule.** The constant-rate Program's accepted observation
sequences of length `n` with jitter below `D` number at most `#Ac · #Σ_sync^W · D^n`: `A_c + W·log2 #Σ_sync` bits plus
jitter, which is 7 bits per window at 16 × 8 sync values. This holds for every frame function, sync set and operator. -/
def EncardAccSeqsLeConstantRate : Prop :=
  ∀ {L K S : ℕ} {In Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D : ℕ)
    (hn : n ≤ W * T),
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((Fintype.card Ac * ss.size ^ W * D ^ n : ℕ) : ℕ∞)

/-- **A fixed clock, for the shipped schedule.** With one clock-sync value (`#Σ_sync = 1`), the constant-rate
Program's accepted sequences number at most `#Ac · D^n`: timing carries nothing beyond the jitter, and under J
(`D = 1`) nothing at all, at most `#Ac` sequences for the content advice. -/
def EncardAccSeqsLeConstantRateFixedClock : Prop :=
  ∀ {L K S : ℕ} {In Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet) (hfix : ss.size = 1)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D : ℕ)
    (hn : n ≤ W * T),
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤ ((Fintype.card Ac * D ^ n : ℕ) : ℕ∞)

/-- **A window's rows read only its own and earlier windows' syncs** (`expected_grids`).
Two lists of declared syncs that agree on every window up to step `t`'s, `t / T`, give the same row at `t`: a frame is
eligible no earlier than its anchor, hand times only rise in canonical order, and the queue carries forward. The syncs
are any clock-sync values, in the set or not, so an out-of-set sync never changes an earlier window's verdict. -/
def ExpectedRowCausal : Prop :=
  ∀ (S T W : ℕ) (syncs syncs' : Fin W → ClockSync) (frames : List LogicalFrame) (t : ℕ)
    (hw : ∀ w : Fin W, (w : ℕ) ≤ t / T → syncs w = syncs' w),
    expectedRow S T W syncs frames t = expectedRow S T W syncs' frames t

/-- **Seven bits per window opened.** If the horizon `n` lies within the first `m` windows (`n ≤ m·T`), the
constant-rate Program's accepted sequences number at most `#Ac · #Σ_sync^min(W, m) · D^n`: only the syncs of windows
the horizon reaches count, however many windows the advice covers. -/
def EncardAccSeqsLeConstantRateWindows : Prop :=
  ∀ {L K S : ℕ} {In Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D m : ℕ)
    (hn : n ≤ W * T) (hm : n ≤ m * T),
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((Fintype.card Ac * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **The composed statement for the shipped schedule (W2 and J).** A decoder reads the wire of the first `n` steps,
within the first `m` windows, and recovers every secret in `Zs`, all of whose runs the constant-rate Program accepts up
to `n`. Then `Zs` has at most `#Ac · #Σ_sync^min(W, m) · D^n` elements: a covert sender inside X moves at most
`A_c + m·log2 #Σ_sync + n·log2 D` bits through egress timing, and with a fixed clock under J only the content advice. -/
def EncardDecodableLeConstantRate : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Ac : Type} [Fintype Ac]
    (I : Interaction L K S Z In Act Wire) (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T W ss frames) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ ((Fintype.card Ac * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **Seven bits per window, over the run's first `m` windows.** Take the run's first `m` whole windows of `T` buckets
within the `W` windows with advice (`hn` is `EncardAccSeqsLeConstantRateWindows`'s `n ≤ W·T` at the horizon `n = m·T`).
The constant-rate Program's accepted sequences of length `m·T` with jitter below `D` number at most
`#Ac · (#Σ_sync · D^T)^m`: the content advice's `A_c` once, then `log2 #Σ_sync + T·log2 D` bits per window, which is 7
bits at 16 × 8 sync values under J (`D = 1`). The factor is the bound's, not a count per window: content advice and an
earlier window's sync can first show in a later window, so one window may continue a given prefix in more than
`#Σ_sync · D^T` ways. So the bound is cumulative from the run's first window: a temporal unit that starts later is not
bounded by its own windows. -/
def EncardAccSeqsLeConstantRatePerWindow : Prop :=
  ∀ {L K S : ℕ} {In Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (m D : ℕ)
    (hn : m * T ≤ W * T),
    (AccSeqs (constantRate (S := S) T W ss frames) op (m * T) D).encard ≤
      ((Fintype.card Ac * (ss.size * D ^ T) ^ m : ℕ) : ℕ∞)

/-- **Seven bits per window, through the wire (W2 and J).** `EncardDecodableLeConstantRate` at the horizon of `m` whole
windows, `n = m·T`: a decoder of the wire recovers at most `#Ac · (#Σ_sync · D^T)^m` secrets, so over the run's first
`m` windows a covert sender inside X moves at most `A_c + m·(log2 #Σ_sync + T·log2 D)` bits through egress timing. -/
def EncardDecodableLeConstantRatePerWindow : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Ac : Type} [Fintype Ac]
    (I : Interaction L K S Z In Act Wire) (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (m D : ℕ) (hn : m * T ≤ W * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T W ss frames) I.op (m * T) (I.obsRun z (m * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (m * T)) = z),
    Zs.encard ≤ ((Fintype.card Ac * (ss.size * D ^ T) ^ m : ℕ) : ℕ∞)

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
first `m` whole windows of `T` buckets, the horizon `n = m·T`: for one content assignment `c`, at most
`(∏_ℓ (r ℓ + 1)^T · D^T)^m` messages. Each link-window contributes the factor `(r ℓ + 1)^T`, its `T·log2(r ℓ + 1)` bits,
and each window `D^T` of jitter, 1 under J. -/
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

This section fixes the two terms of the charge that the capacity bounds leave open: the content advice `A_c` and the
coins (D7). Nothing in it is implemented yet. The numbers are at the llama8b-node1 calibration
(`calibrations/llama8b-node1.json`): T = 600 buckets of 100 ms, B = 4,096, one egress link at r = 1,686, one ingress
link (`requests`) at r_in = 34 and τ_in = 100 ms, Σ_sync of 16 × 8 values, and J (D = 1).

### What `A_c` counts

`constantRate T W ss frames` takes the advice `Ac × (Fin W → Σ_sync)`, and `frames ins c ℓ` gives every egress frame
(its payload, anchor, logical index and canonical position) from the ingress `ins` and the content advice `c`.
Acceptance is existential: a run is accepted if some advice value explains it (`AcceptedUpTo`). The proof
(`encard_accSeqs_le_advice`) maps each accepted sequence to an advice value that accepts it, plus its jitter code. So
`A_c = log2 #Ac` counts every content-advice value the audit would admit as an explanation, whenever and however it was
chosen. It is charged once per run of W windows, not per window. The number means something only if the set `Ac` and
the function `frames` are fixed before the run, independently of the sender's secret, and the audit refuses advice
outside `Ac`, as it refuses a sync outside Σ_sync.

For a served Program the content advice is the schedule: the per-step batch composition from which the circuit is
built. The anchors and logical indices are read from the ingress and the session. The schedule reaches the egress
frames through their canonical order, which is step-major, through how many tokens each event holds, and through any
batch-variant numerics. Today the schedule is whatever the workload manifest declares: each request's `arrive_step`
and the engine's scheduler settings. So `#Ac` has no bound, and the charge has no number. A monotone map from buckets
to steps that is free at up to s_max steps per bucket is worth `T·log2(s_max + 1)` bits per window. At s_max = 15
that is 2,400 bits per window, 343 times the sync's 7.

### The declaration

- **A deterministic scheduler.** The pinned engine environment includes its scheduler. Given a capacity profile p and
  each request's admission step, the scheduler fixes the per-step schedule: `s = Sched_E(p, arrivals)`. Bit-exact
  replay already relies on this, and `verity_vllm` compares each request's realized arrival step with the declared one.
- **Admission on a step clock.** The profile sets κ, a whole number of engine steps per bucket. Request q is admitted
  at step `κ · anchor(q)`. Its anchor is the bucket in which the ingress certifier released its last frame, read from
  the ingress certificate. Requests released in one bucket are admitted in the ingress release order. The engine never
  starts step κ·b before the releases of bucket b are in. When it runs slower, it lags, and a lag past the declared sync
  is an honest miss, as now. Given p, the schedule is then a function of the ingress certificates, and p is the only
  free choice. `verity_vllm.engine.arrivals` already admits each request right before engine step `arrive_step`, on
  the step counter and never on wall time. The declaration sets `arrive_step = κ · anchor`. The module's rule that an
  idle engine admits the next arrival at once is itself a function of the schedule.
- **A registered list of profiles.** `P_cap = (p_0, …, p_{N−1})`. Each profile holds κ and every scheduler setting the
  schedule reads: the batch caps (sequences and tokens per step), the chunked-prefill threshold, the KV block count,
  which fixes preemption, and the output lag of asynchronous scheduling. The list is a public parameter of the audit,
  registered with Σ_sync, (T, r, B, τ_b, Q), r_in and τ_in before the audit's first window and before any coin. Like
  Σ_sync, it is the deployment's structure.
  - If the developer wants the profiles hidden, the list is a frame-v3-sha512 root over hm96-sha512 rows, one per
    profile, with N public. A sender inside X could then pick the list itself, so a hidden list also costs its own size
    once per registration: at most N·b_p bits for b_p-bit rows. A public list costs nothing.
- **The profile in use.** Its index i < N goes under the schedule root, with the per-step schedule rows. That root is
  a frame-v3-sha512 tree of hm96-sha512 rows (circuits' X3), hidden, and registered before any coin that depends on it.
  The registration statement proves that the rows are `Sched_E(p_i, arrivals)` for the arrivals the ingress
  certificates give, and that i < N. The count holds whenever i was chosen, since acceptance takes any value in the set.
  The rule that the root precedes the coins is there for the sampled proofs, so that the circuit can't be shaped around
  the draw.

**The count.** With one profile per audit, `#Ac = N` and `A_c = log2 N`, once. Switching profiles at window
boundaries, declared at each window's first bucket with the sync, makes the advice one (profile, sync) pair per window:
`log2(N · #Σ_sync)` bits per window. `SecurityProofs.EncardAccSeqsLeConstantRate` with `Ac = Fin N`, or with
`Fin W → Fin N` for switching, states both, so no new guarantee is needed.

**At the recorded parameters.** The recordings use one scheduler configuration. κ needs a grid that covers the honest
step rate, calibrated as ρ's 8 values are, and κ has none. Taking 8 values of κ gives N = 8:

| Charge | Per window | Per audit of W = 1,440 windows (a day) |
|---|---|---|
| Egress, one profile per audit | 7 bits, plus 3 bits once | 3 + 1,440 · 7 = 10,083 bits |
| Egress, a profile per window | 7 + 3 = 10 bits | 14,400 bits |
| Egress with one clock, `A_c` left out | 7 bits | 10,080 bits (`egress_bits_per_day_one_clock`) |

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
