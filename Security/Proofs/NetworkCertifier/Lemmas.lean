import Definitions.NetworkCertifier
import Definitions.NetworkCertifier.Assumptions
import Definitions.NetworkCertifier.Guarantees.AdviceV1
import Proofs.NetworkCertifier.AdviceV1Counted
import Mathlib.Data.Sym.Card
import Mathlib.Data.Set.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The network certifier's lemmas

The statements that are steps of the guarantees' proofs (`Definitions.NetworkCertifier.Guarantees`) rather than bounds or rules
a document cites: the occupancy facts, Theorem 1(b)'s counting bound, the advice cap the clock-sync and block forms
specialise, and the composed block statement; and the retired guarantees, proved and cited by nothing. Each is a
`def G : Prop` under the guarantees' namespace, proved by `NetTiming.SecurityProofs.G`, so they are audited with them
and pinned by no record.
-/

namespace NetTiming.Guarantees

open Finset

-- A lemma keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **Timing is occupancy, (i).** For a fixed frame sequence, the grid determines the counts. -/
def FillInjective : Prop :=
  ∀ {T S : ℕ} {F : Type} (f : ℕ → F), Function.Injective (fill (T := T) (S := S) f)

/-- **Timing is occupancy, (ii).** A grid whose occupancy is left-packed is the fill of some frame sequence at some
counts. -/
def ExistsFillOfLeftPacked : Prop :=
  ∀ {T S : ℕ} {F : Type} [Inhabited F] (g : Grid T S F) (h : LeftPacked (occ g)),
    ∃ (f : ℕ → F) (n : Fin T → Fin (S + 1)), g = fill f n

/-- **At most M released:** at most `binom(T + M, M)`, which is the doc's `binom(M+T, T)`. -/
def CardCountsLeLe : Prop :=
  ∀ (T S M : ℕ),
    (univ.filter fun n : Fin T → Fin (S + 1) => ∑ t, (n t : ℕ) ≤ M).card ≤ (T + M).choose M

/-- **Tightness, the schedule.** If the sender enqueues `n t ≤ S` frames in bucket `t`, the queue stays empty and the
FIFO certifier releases exactly `n t`. Over `T` buckets this uses `∑ n t ≤ T·S` frames. -/
def ReleasedSelf : Prop :=
  ∀ (S : ℕ) (n : ℕ → Fin (S + 1)), released S (fun t => (n t : ℕ)) = n

/-- **The certifier never releases more than was enqueued.** -/
def SumReleasedLe : Prop :=
  ∀ (T S : ℕ) (enq : ℕ → ℕ),
    ∑ t ∈ Finset.range T, (released S enq t : ℕ) ≤ ∑ t ∈ Finset.range T, enq t

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
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard

/-- **W6 on accepted runs.** Under fail-closed certification, a run accepted up to `n` takes no fault action at any step
`i` whose window closes before `n`. (`hwin`: every step lies in its own window, which ends no earlier.) -/
def NoFaultOfAccepted : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Adv : Type}
    (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (winEnd : ℕ → ℕ)
    (hwin : ∀ i, i ≤ winEnd i)
    (hW6 : FailClosed I.certifier winEnd) (z : Z) (n : ℕ) (hacc : AcceptedUpTo P I.op n (I.obsRun z n))
    (i : ℕ) (hi : i < (I.actRun z n).length) (hend : winEnd i < n),
    ¬ I.certifier.fault (I.actRun z n)[i]

/-- **W4 at the certifier: timing is occupancy.** Under packing, every recorded egress row is the fill of some frame
sequence at its count, written as a one-bucket grid. -/
def RecordedRowEqFill : Prop :=
  ∀ {L K S : ℕ} {In Act Wire : Type} (w : Certifier L K S In Act Wire) (hW4 : Packed w)
    (k : ℕ) (xs : List In) (as : List Act) (ℓ : Fin L),
    ∃ (f : ℕ → List Bool) (c : Fin 1 → Fin (S + 1)), (fun _ : Fin 1 => (w.obs k xs as).rows ℓ) = fill f c

/-- **Theorem 3, the `2^A` cap.** With `Fintype.card Adv = 2^A`, the accepted observation sequences of length `n` with
jitter below `D` number at most `2^A · D^n`. -/
def EncardAccSeqsLeAdvice : Prop :=
  ∀ {L K S : ℕ} {In Adv : Type} [Fintype Adv] (P : Program L S In Adv)
    (op : ℕ → List (Obs L K S) → In) (n D : ℕ),
    (AccSeqs P op n D).encard ≤ ((Fintype.card Adv * D ^ n : ℕ) : ℕ∞)

/-- **The composed statement (W2, J and both conditions).** A decoder reads the wire and recovers every secret in `Zs`,
all of whose runs are accepted up to `n`. Then `Zs` is no larger than the block cap. -/
def EncardDecodableLeBlocks : Prop :=
  ∀ {L K S : ℕ} {Z In Act Wire Ac B C : Type} {At : B → Type} {U : C → Type}
    [Fintype Ac] [Fintype B] [DecidableEq B] [∀ β, Fintype (At β)] [Fintype C] [DecidableEq C] [∀ c, Fintype (U c)]
    (I : Interaction L K S Z In Act Wire) (P : Program L S In (Ac × ((β : B) → At β))) (n D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (step : C → ℕ) (blk : C → B) (occ : (c : C) → List In → Ac × ((β : B) → At β) → U c)
    (h1 : FramesReadNoTimingAdvice P step occ) (h2 : OccupancyReadsOwnBlock blk occ)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤
      ((Fintype.card Ac * D ^ n *
        ∏ β, min (Fintype.card (At β)) (∏ c ∈ univ.filter (fun c => blk c = β), Fintype.card (U c)) : ℕ) : ℕ∞)

/-! ## Retired: advice v1 with a separate count code

The charge with counts under a separate count code (`Proofs.NetworkCertifier.AdviceV1Counted`) is not how advice v1
sends counts, and the whole-window charge (`EncardChargedLeWhole`, `ChargeCount_v2`) bounds the counts coded inside
the coder, so these four left the guarantees. -/

section Counted

open AdviceV1

/-- **Advice with counts: the charge counts (Kraft).** Under any registered book `bk` and any count code `cc` (lengths
meeting Kraft's inequality, as every prefix-free code's do), with windows of `N` slots, at most `2^c` advice values of
`W` windows, each window's decision stream and counts, are charged `c` bits or fewer in all, with no window refused. A
window's charge is its decisions' charge (`AdviceV1.window`) plus `cc`'s length for its counts (`windowCounted`), so
the windows may send nonzero counts. The code is the same in every window and carries nothing across windows. That
the circuit computes the decisions' charge is the hypothesis left, as for `EncardChargedLe`, and the
count code is a parameter, a separate code (not advice v1's coding of counts, which is inside the coder). -/
def EncardChargedLeCounted : Prop :=
  ∀ {C : Type} (bk : Book) (N W c : ℕ) (cc : CountCode C),
    {a : Fin W → List Bool × C | ∃ ls, chargesCounted bk N cc a = some ls ∧ ls.sum ≤ c}.encard ≤
      ((2 ^ c : ℕ) : ℕ∞)

/-- **The leak bound with the charge with counts.** `EncardAccSeqsWithinLeConstantRate` with the advice string each
window's decision stream and counts (`StreamsCounted`) and its charge `bitsCounted` under any count code `cc`, so with
no assumption on the charge: the constant-rate Program's observation sequences of length `n` (within the `W` windows
with advice, and the first `m` windows), with jitter below `D`, accepted with advice charged at most `b w` in every
window `w`, number at most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n`. The windows may send nonzero counts, and the
scope is `EncardChargedLeCounted`'s. -/
def EncardAccSeqsWithinLeConstantRateV1Counted : Prop :=
  ∀ {L K S W : ℕ} {In C : Type} (bk : Book) (N : ℕ) (cc : CountCode C) (b : Fin W → ℕ) (T : ℕ) (ss : SyncSet)
    (frames : List In → StreamsCounted bk N cc W → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T),
    (AccSeqsWithin (AdviceV1.bitsCounted bk N cc) b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **The leak bound through the wire, with the charge with counts (W2 and J).** `EncardDecodableWithinLeCompiled`
with each window's decision stream and counts (`StreamsCounted`) under the registered book `bk`, any count code `cc`
and windows of `N` slots, so with no assumption on the charge: a decoder of the wire's first `n` steps recovers at
most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n` secrets whose runs the compiled Program accepts with advice charged at
most `b w` in every window `w`. The windows may send nonzero counts, and the scope is `EncardChargedLeCounted`'s. -/
def EncardDecodableWithinLeCompiledV1Counted : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Ω C : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : Book) (N : ℕ) (cc : CountCode C)
    (b : Fin W → ℕ) (compilerOf : Z → Compiler L In (StreamsCounted bk N cc W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin (AdviceV1.bitsCounted bk N cc) b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n
        (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **The leak bound per window, with the charge with counts (W2 and J).** `EncardDecodableWithinLeCompiledPerWindow`
with each window's decision stream and counts and the charge under any count code `cc`: over all `W` windows, a
decoder of the wire recovers at most `∏_w (2^(b w) · #Σ_sync · D^T)` secrets. The windows may send nonzero counts, and
the scope is `EncardChargedLeCounted`'s. -/
def EncardDecodableWithinLeCompiledPerWindowV1Counted : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Ω C : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : Book) (N : ℕ) (cc : CountCode C)
    (b : Fin W → ℕ) (compilerOf : Z → Compiler L In (StreamsCounted bk N cc W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin (AdviceV1.bitsCounted bk N cc) b
      (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T) (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z),
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞)

end Counted

end NetTiming.Guarantees
