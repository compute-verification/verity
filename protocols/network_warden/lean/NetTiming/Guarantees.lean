import NetTiming.Protocol
import NetTiming.Assumptions
import Mathlib.Data.Sym.Card
import Mathlib.Data.Set.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The network warden's guarantees

Each guarantee is a `def G : Prop`, stated over `Protocol`'s definitions with the assumptions it rests on as hypotheses
(`Assumptions`). Its proof is the theorem `NetTiming.SecurityProofs.G : NetTiming.Guarantees.G`, and a guarantee without
one is open: nothing may cite it until it is proved.

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

/-- **Timing is occupancy, (i).** For a fixed frame sequence, the grid determines the counts. -/
def FillInjective : Prop :=
  ∀ {T S : ℕ} {F : Type} (f : ℕ → F), Function.Injective (fill (T := T) (S := S) f)

/-- **Timing is occupancy, (ii).** A grid whose occupancy is left-packed is the fill of some frame sequence at some
counts. -/
def ExistsFillOfLeftPacked : Prop :=
  ∀ {T S : ℕ} {F : Type} [Inhabited F] (g : Grid T S F) (h : LeftPacked (occ g)),
    ∃ (f : ℕ → F) (n : Fin T → Fin (S + 1)), g = fill f n

/-- **Fixed M, all released:** at most `binom(T + M − 1, M)` count vectors, which is the doc's `binom(M+T−1, T−1)`
for `T ≥ 1`. -/
def CardCountsEqLe : Prop :=
  ∀ (T S M : ℕ),
    (univ.filter fun n : Fin T → Fin (S + 1) => ∑ t, (n t : ℕ) = M).card ≤ (T + M - 1).choose M

/-- **At most M released:** at most `binom(T + M, M)`, which is the doc's `binom(M+T, T)`. -/
def CardCountsLeLe : Prop :=
  ∀ (T S M : ℕ),
    (univ.filter fun n : Fin T → Fin (S + 1) => ∑ t, (n t : ℕ) ≤ M).card ≤ (T + M).choose M

/-- **Tightness, the schedule.** If the sender enqueues `n t ≤ S` frames in bucket `t`, the queue stays empty and the
FIFO warden releases exactly `n t`. Over `T` buckets this uses `∑ n t ≤ T·S` frames. -/
def ReleasedSelf : Prop :=
  ∀ (S : ℕ) (n : ℕ → Fin (S + 1)), released S (fun t => (n t : ℕ)) = n

/-- **The warden never releases more than was enqueued.** -/
def SumReleasedLe : Prop :=
  ∀ (T S : ℕ) (enq : ℕ → ℕ),
    ∑ t ∈ Finset.range T, (released S enq t : ℕ) ≤ ∑ t ∈ Finset.range T, enq t

/-- **Fixed M, at the warden.** When the sender enqueues at most `M` frames over the window, the FIFO warden can release
at most `binom(T + M, M)` distinct grids: the doc's `binom(M+T, T)`. -/
def EncardFifoGridsLeOfAvailable : Prop :=
  ∀ {F : Type} (T S M : ℕ) (f : ℕ → F),
    {g | ∃ enq : ℕ → ℕ, ∑ t ∈ Finset.range T, enq t ≤ M ∧ g = fifoGrid T S f enq}.encard ≤
      ((T + M).choose M : ℕ)

/-- **Tightness.** Over all enqueue schedules, with the frames leaving in queue order `f`, the FIFO warden can release
exactly `(S + 1)^T` distinct grids. So `T·log2(S+1)` bits are attained with zero error. -/
def EncardFifoGrids : Prop :=
  ∀ {F : Type} (T S : ℕ) (f : ℕ → F),
    (Set.range fun enq : ℕ → ℕ => fifoGrid T S f enq).encard = ((S + 1) ^ T : ℕ)

/-! ## Theorem 1(b): the counting bound on accepted runs -/

/-- **The tree lemma.** Take a set of length-`n` sequences in which, after any prefix of length `t`, the next element
takes at most `N t` values. It has at most `∏_{t<n} N t` elements. -/
def EncardLeProdOfBranching : Prop :=
  ∀ {α : Type} (n : ℕ) (Sq : Set (List α)) (N : ℕ → ℕ)
    (hlen : ∀ o ∈ Sq, o.length = n)
    (hN : ∀ t < n, ∀ h : List α, {x | ∃ o ∈ Sq, o.take (t + 1) = h ++ [x]}.encard ≤ N t),
    Sq.encard ≤ ((∏ t ∈ Finset.range n, N t : ℕ) : ℕ∞)

/-- **Theorem 1(b), accepted runs.** Take the observation sequences of the runs accepted up to the horizon `n`,
whatever the sender's strategy (over all secrets) and however the operator reacts. Suppose that after any prefix of
length `t`, the runs accepted up to `t + 1` continue with at most `N t` distinct observations; this is the doc's
`N_acc`, defined by prefix acceptance. Then there are at most `∏_{t<n} N t` such sequences. -/
def EncardAcceptedRunsLe : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Adv : Type}
    (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (n : ℕ) (N : ℕ → ℕ)
    (hN : ∀ t < n, ∀ h : List (Obs L K S),
      {x | ∃ z, AcceptedUpTo P I.op (t + 1) (I.obsRun z (t + 1)) ∧ I.obsRun z (t + 1) = h ++ [x]}.encard ≤ N t),
    {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard ≤ ((∏ t ∈ Finset.range n, N t : ℕ) : ℕ∞)

/-- **Zero error, from the observations.** A decoder reads the observations of the first `n` steps (the published
records, jitter and statuses) and recovers every secret in `Zs`, all of whose runs are accepted up to `n`. Then `Zs`
has at most as many elements as there are accepted observation sequences. This needs no W2, and the wire version below
follows from it. -/
def EncardObsDecodableLe : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Adv : Type}
    (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (n : ℕ)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List (Obs L K S) → Z) (hdec : ∀ z ∈ Zs, dec (I.obsRun z n) = z),
    Zs.encard ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard

/-- **Zero error, through the wire (uses W2).** A decoder reads the wire of the first `n` steps and recovers every
secret in `Zs`, all of whose runs are accepted up to `n`. Then `Zs` has at most as many elements as there are accepted
observation sequences. -/
def EncardDecodableLe : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Adv : Type}
    (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (n : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard

/-- **W6 on accepted runs.** Under fail-closed wardening, a run accepted up to `n` takes no fault action at any step
`i` whose window closes before `n`. (`hwin`: every step lies in its own window, which ends no earlier.) -/
def NoFaultOfAccepted : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Adv : Type}
    (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (winEnd : ℕ → ℕ)
    (hwin : ∀ i, i ≤ winEnd i)
    (hW6 : FailClosed I.warden winEnd) (z : Z) (n : ℕ) (hacc : AcceptedUpTo P I.op n (I.obsRun z n))
    (i : ℕ) (hi : i < (I.actRun z n).length) (hend : winEnd i < n),
    ¬ I.warden.fault (I.actRun z n)[i]

/-! ## Theorem 3: the advice caps -/

/-- **W4 at the warden: timing is occupancy.** Under packing, every recorded egress row is the fill of some frame
sequence at its count, written as a one-bucket grid. -/
def RecordedRowEqFill : Prop :=
  ∀ {L K S : ℕ} {In Act Wire : Type} (w : Warden L K S In Act Wire) (hW4 : Packed w)
    (k : ℕ) (xs : List In) (as : List Act) (ℓ : Fin L),
    ∃ (f : ℕ → List Bool) (c : Fin 1 → Fin (S + 1)), (fun _ : Fin 1 => (w.obs k xs as).rows ℓ) = fill f c

/-- **Theorem 3, the `2^A` cap.** With `Fintype.card Adv = 2^A`, the accepted observation sequences of length `n` with
jitter below `D` number at most `2^A · D^n`. -/
def EncardAccSeqsLeAdvice : Prop :=
  ∀ {L K S : ℕ} {In Adv : Type} [Fintype Adv] (P : Program L S In Adv)
    (op : ℕ → List (Obs L K S) → In) (n D : ℕ),
    (AccSeqs P op n D).encard ≤ ((Fintype.card Adv * D ^ n : ℕ) : ℕ∞)

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
jitter, and nothing at all under J (`D = 1`): the Notion zero-capacity theorem, with content advice. -/
def EncardAccSeqsLeFixedClock : Prop :=
  ∀ {L K S : ℕ} {In Ac : Type} [Fintype Ac] (W : ℕ)
    (P : Program L S In (Ac × (Fin W → Unit))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ),
    (AccSeqs P op n D).encard ≤ ((Fintype.card Ac * D ^ n : ℕ) : ℕ∞)

/-- **Theorem 3, the block cap, under the two dataflow conditions.** The advice is content advice `Ac` and timing
advice `At β` per block `β`. The cells `C` carry the Program's occupancy predictions `occ c : U c`, at step `step c`,
in block `blk c`. Then the accepted sequences number at most
`#Ac · D^n · ∏_β min(#At β, ∏_{c ∈ β} #U c)`. -/
def EncardAccSeqsLeBlocks : Prop :=
  ∀ {L K S : ℕ} {In Ac B C : Type} {At : B → Type} {U : C → Type}
    [Fintype Ac] [Fintype B] [DecidableEq B] [∀ β, Fintype (At β)] [Fintype C] [DecidableEq C] [∀ c, Fintype (U c)]
    (P : Program L S In (Ac × ((β : B) → At β))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ)
    (step : C → ℕ) (blk : C → B) (occ : (c : C) → List In → Ac × ((β : B) → At β) → U c)
    (h1 : FramesReadNoTimingAdvice P step occ) (h2 : OccupancyReadsOwnBlock blk occ),
    (AccSeqs P op n D).encard ≤
      ((Fintype.card Ac * D ^ n *
        ∏ β, min (Fintype.card (At β)) (∏ c ∈ univ.filter (fun c => blk c = β), Fintype.card (U c)) : ℕ) : ℕ∞)

/-- **The left-packed instance.** Cells are the (link, step) pairs of the horizon, and the Program's occupancy at a
cell is the count of its row, a value in `Fin (S + 1)`. Then the block cap reads
`#Ac · D^n · ∏_β min(#At β, (S + 1)^|β|)`. -/
def EncardAccSeqsLeBlocksCounts : Prop :=
  ∀ {L K S : ℕ} {In Ac B : Type} {At : B → Type}
    [Fintype Ac] [Fintype B] [DecidableEq B] [∀ β, Fintype (At β)]
    (P : Program L S In (Ac × ((β : B) → At β))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ)
    (blk : Fin L × Fin n → B)
    (h1 : FramesReadNoTimingAdvice P (fun c : Fin L × Fin n => (c.2 : ℕ))
      (fun c ins a => rowCount (P.rows c.2 ins a c.1)))
    (h2 : OccupancyReadsOwnBlock blk (fun c : Fin L × Fin n => fun ins a => rowCount (P.rows c.2 ins a c.1))),
    (AccSeqs P op n D).encard ≤
      ((Fintype.card Ac * D ^ n *
        ∏ β, min (Fintype.card (At β)) ((S + 1) ^ (univ.filter (fun c => blk c = β)).card) : ℕ) : ℕ∞)

/-- **The composed statement (W2, J and both conditions).** A decoder reads the wire and recovers every secret in `Zs`,
all of whose runs are accepted up to `n`. Then `Zs` is no larger than the block cap. -/
def EncardDecodableLeBlocks : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Ac B C : Type} {At : B → Type} {U : C → Type}
    [Fintype Ac] [Fintype B] [DecidableEq B] [∀ β, Fintype (At β)] [Fintype C] [DecidableEq C] [∀ c, Fintype (U c)]
    (I : Interaction L K S Z In Act Wire) (P : Program L S In (Ac × ((β : B) → At β))) (n D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hJ : JitterBounded I.warden D)
    (step : C → ℕ) (blk : C → B) (occ : (c : C) → List In → Ac × ((β : B) → At β) → U c)
    (h1 : FramesReadNoTimingAdvice P step occ) (h2 : OccupancyReadsOwnBlock blk occ)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤
      ((Fintype.card Ac * D ^ n *
        ∏ β, min (Fintype.card (At β)) (∏ c ∈ univ.filter (fun c => blk c = β), Fintype.card (U c)) : ℕ) : ℕ∞)

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

/-- **A window's rows read only its own and earlier windows' syncs** (PROTOCOL.md, "The audit"; `expected_grids`).
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
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hJ : JitterBounded I.warden D)
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
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hJ : JitterBounded I.warden D)
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
    (hW4 : Packed I.warden) (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
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
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hW4 : Packed I.warden)
    (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
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
    (hW4 : Packed I.warden) (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
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
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hW4 : Packed I.warden)
    (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
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
    (hW4 : Packed I.warden) (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
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
    (hW4 : Packed I.warden) (hr : Allocated I.warden r) (hJ : JitterBounded I.warden D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg),
    Ms.encard ≤ Cs.encard * ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞)

end NetTiming.Guarantees
