import Definitions.NetworkCertifier
import Definitions.NetworkCertifier.Assumptions
import Mathlib.Data.Sym.Card
import Mathlib.Data.Set.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The network certifier's lemmas

The statements that are steps of the guarantees' proofs (`Definitions.NetworkCertifier.Guarantees`) rather than bounds or rules
a document cites: the occupancy facts, Theorem 1(b)'s counting bound and the advice cap the clock-sync form
specialises. Each is a `def G : Prop` under the guarantees' namespace, proved by `NetTiming.SecurityProofs.G`, so they
are audited with them and pinned by no record.
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

end NetTiming.Guarantees
