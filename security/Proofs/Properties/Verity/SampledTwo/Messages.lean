module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Flock.Claims.Recursion.DeploySessions
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Recursion.Inner
public import Proofs.Properties.Verity.Sampled.Messages

/-!
# The two-stage leg's messages: the frame's prover, and the messages fixed by their digests

The projection (`Proofs.Properties.Verity.SampledTwo.Projection`) reads a play of the period's game as a run of the
two-stage verifier `A.L.V` at a prover the strategy is read as (`Frame.P`). This module builds that prover and states
the two pieces about it, `CallsLe` and `MessagesFixed`.

* **The prover** (`proverOf`, `modePick`, `prover`): a prover that reads its messages off its tape makes no SHA-512 call
  (`run_pLog`). The frame's prover sends, at each round, the most likely message (`modeOn`) the play reads there over
  the run's coins its history shows (`Shows`), among those that open the digest the play records for the round
  (`opening`). A stateless prover learns the run's coins only from the verifier's answers: an answer that is a coin's
  bytes shows the coin (`shownCoin`); the statements' round and a session's deciding round show none.
* **The calls** (`VerifierCallsLe`, `callsLe_of`): the prover's calls are none, so `CallsLe` is the verifier's calls
  within the strategy's, `V.cost`. That is a fact about the instance's declared cost, not about the prover: at a
  vocabulary whose `cost` is `0` it fails as soon as the verifier hashes once. The recursive leg takes the same fact as
  a field of its session (`SessionIn.calls_le`).
* **The rounds the run reaches** (`MessagesReached`, `RunsThrough`, `messagesFixed_of_reached`): `MessagesFixed` asks
  the messages to agree at every round below `k`, but a run that refuses early has no message past its last round, so
  it also asks that the run does not stop early, which is the accept piece's fact. `MessagesReached` asks agreement at
  the rounds the run reaches; with `RunsThrough` it gives `MessagesFixed`.
* **The first difference** (`runHist`, `MissAt`, `reachedOfRounds`): on a play where the messages differ at a reached
  round, they differ first at some round `i`, where the run's history is the play's (`playHist_eq`).
* **Forking** (`miss_le_sqrt`): on a finite space split into classes, with a uniform resample within a class, a member
  whose message is not its class's most likely one is at most as likely as the square root of a member and its resample
  having two different messages.
* **The miss at one round** (`MsgOpens`, `DigestBefore`, `RunShown`, `merge`, `msgFinder`, `missBound`): where the
  message the check reads opens the digest the play records (`MsgOpens`, the form's parse's), that digest is fixed by
  the coins the history shows (`DigestBefore`, the coin map's and the cut's), and the verifier's answers show the
  run's coins and depend on them only through what they show (`RunShown`, a fact about `SpotFlock.V`), the round's
  miss is at most the square root of a two-continuation finder's success. The classes are the frame's run's histories
  before round `i`; the finder flips the coins again the history does not show (`merge`) and outputs the two messages the
  two plays read when they differ, a SHA-512 collision since both open one digest. The finder's cost is declared, as
  in the one-stage leg's.
* **The messages' bound** (`MessagesBound`, `messagesBound`): `MessagesFixed` at the sum over the rounds of those
  square roots.

`Proofs.Properties.Verity.SampledTwo.MessagesShown`, which proves `RunShown`, and
`Proofs.Properties.Verity.SampledTwo.SessionMsgs` import this module.
-/

@[expose] public section

namespace Verity.SampledTwo.Messages

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The prover -/

/-- **A prover that reads its messages off its tape**: at the history `hist` and the tape `ω`, the message
`pick ω hist`, with no SHA-512 call. -/
def proverOf {Ω : Type} (pick : Ω → List (Msg × Msg) → Msg) : Claims.Prover Ω :=
  ⟨fun _ hist ω => .ret (pick ω hist)⟩

theorem stepWith_pLog (H : Msg → Msg) {Ω : Type} (pick : Ω → List (Msg × Msg) → Msg) (x : Msg) (ω : Ω)
    (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      (stepWith H (proverOf pick) x ω vn n hist as t).pLog = t.pLog
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
    rcases hvb : (vn (as ++ [(((proverOf pick).next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg m =>
      rw [RecDeploy.stepWith_succ_msg H _ x ω vn n hist as t hvb, stepWith_pLog]
      simp [proverOf, OC.eval]
    | acc =>
      rw [RecDeploy.stepWith_succ_acc H _ x ω vn n hist as t hvb]
      simp [proverOf, OC.eval]
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H _ x ω vn n hist as t hvb]
      simp [proverOf, OC.eval]

/-- **A prover that reads its messages off its tape makes no SHA-512 call** in any run. -/
theorem run_pLog (H : Msg → Msg) {Ω Coin : Type} [Inhabited Coin] (pick : Ω → List (Msg × Msg) → Msg)
    (Vr : Verifier Coin) (x : Msg) (ω : Ω) (r : Fin (Vr.k x) → Coin) :
    (Claims.run H (proverOf pick) Vr x ω r).pLog = [] :=
  stepWith_pLog H pick x ω _ _ [] [] ⟨[], [], [], .rej⟩

/-- **The coin an answer shows**: the coin whose bytes the verifier's answer is, if any. -/
noncomputable def shownCoin (m : Msg) : Option FlockDeploy.Coin := by
  classical
  exact if h : ∃ c : FlockDeploy.Coin, c.bytes = m then some h.choose else none

/-- **The coin the history `hist` shows at round `j`**, if its answer there shows one. -/
noncomputable def shownAt (hist : List (Msg × Msg)) (j : ℕ) : Option FlockDeploy.Coin :=
  (hist[j]?).bind fun e => shownCoin e.2

/-- **The run's coins agree with every coin the history `hist` shows**: at each round before the history's end whose
answer shows a coin, the run's coin is that coin. -/
def Shows {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (r : Coins A x) : Prop :=
  ∀ (j : Fin (A.L.V.k x)) c, shownAt hist j.val = some c → r j = c

/-- **The coins the history `hist` shows, `y` at every other round.** -/
noncomputable def merge {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (y : Coins A x) : Coins A x :=
  fun j => (shownAt hist j.val).getD (y j)

/-- **The completion of a history**: the coins it shows, `default` at every other round. -/
noncomputable def complete {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) : Coins A x :=
  merge hist fun _ => default

/-- **The message read at round `i` that opens the history's digest**: the message the reading `Rd` reads at round `i`
of the play `out ω r`, where its SHA-512 is the digest `Dg` reads for round `i` in the play at the history's
completion. -/
noncomputable def opening {A : Live₂} {x : Msg} {Ω : Type} (out : Ω → Coins A x → Verity.Exchange) (Rd : Reading A)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ) (ω : Ω) (hist : List (Msg × Msg)) (r : Coins A x) :
    Option Msg := by
  classical
  exact (Rd.msg (out ω r) i).filter fun m => decide (Dg (out ω (complete hist)) i = some (Verity.sha512Msg m))

open Classical in
/-- **The most likely value of `f` on `S`**: a message `f` takes on at least as many members of `S` as any other;
`[]` where it takes none. -/
noncomputable def modeOn {β : Type} [Fintype β] (S : β → Prop) (f : β → Option Msg) : Msg :=
  if ∃ m b, S b ∧ f b = some m then
    Classical.epsilon fun m => (∃ b, S b ∧ f b = some m) ∧
      ∀ m', (Finset.univ.filter fun b => S b ∧ f b = some m').card ≤
        (Finset.univ.filter fun b => S b ∧ f b = some m).card
  else []

/-- **The frame's message at a history**: the most likely message the play reads at the history's round, over the run's
coins the history shows, among those that open the digest the play at the history's completion records for it. -/
noncomputable def modePick {A : Live₂} {x : Msg} {Ω : Type} (out : Ω → Coins A x → Verity.Exchange) (Rd : Reading A)
    (Dg : Verity.Exchange → ℕ → Option Msg) (ω : Ω) (hist : List (Msg × Msg)) : Msg :=
  modeOn (Shows (A := A) (x := x) hist) (opening out Rd Dg hist.length ω hist)

/-- **The play at the tape `ω` with the run's coins `r` written in by the coin map `place`.** -/
noncomputable def outOf (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (ω : TapeOf V θ F c σ) (r : Coins A x) : Verity.Exchange :=
  Verity.Game.Tape.out (V.game θ F c) σ (place ω r)

/-- **The frame's prover** at the coin map `place`, the reading `Rd` and the digests `Dg`: at each history, the most
likely message (`modePick`) on the plays the coin map makes. -/
noncomputable def prover (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) : Claims.Prover (TapeOf V θ F c σ) :=
  proverOf (modePick (outOf V θ F c σ place) Rd Dg)

/-- **The frame** at the coin map `place`, with the frame's prover. -/
noncomputable def frame (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) : Frame V A x θ F c σ :=
  ⟨place, prover V θ F c σ place Rd Dg⟩

theorem frame_out (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) (ω : TapeOf V θ F c σ) (r : Coins A x) :
    (frame V A x θ F c σ place Rd Dg).out ω r = outOf V θ F c σ place ω r := rfl

/-! ## The calls -/

/-- **The verifier's calls in each run of the prover `P` are within the strategy's** (the instance's, as the recursive
leg's `SessionIn.calls_le`): `V.cost` is declared, so nothing else ties it to the verifier's SHA-512 calls. -/
def VerifierCallsLe (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (P : Claims.Prover (TapeOf V θ F c σ)) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x),
    (Claims.run FlockDeploy.sha512 P A.L.V x ω r).vLog.length ≤ V.cost θ F c σ

/-- **The prover's calls are the strategy's**, at a frame whose prover reads its messages off its tape, where the
verifier's calls are within the strategy's. -/
theorem callsLe_of (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (pick : TapeOf V θ F c σ → List (Msg × Msg) → Msg) (hP : Fr.P = proverOf pick)
    (h : VerifierCallsLe V A x θ F c σ Fr.P) : CallsLe V A x θ F c σ Fr := by
  intro ω r
  have h0 := run_pLog FlockDeploy.sha512 pick A.L.V x ω r
  rw [← hP] at h0
  rw [h0, List.length_nil, Nat.zero_add]
  exact h ω r

/-- **`CallsLe` at the frame**, where the verifier's calls are within the strategy's. -/
theorem callsLe (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg)
    (h : VerifierCallsLe V A x θ F c σ (prover V θ F c σ place Rd Dg)) :
    CallsLe V A x θ F c σ (frame V A x θ F c σ place Rd Dg) :=
  callsLe_of V A x θ F c σ _ _ rfl h

/-! ## The rounds the run reaches -/

/-- **The messages fixed at the rounds the run reaches**: `MessagesFixed` with the rounds below the run's length in
place of those below `k`. -/
def MessagesReached (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (miss : ℝ≥0∞) : Prop :=
  ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
      Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
        ∃ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).msgs.length, Rd.msg (Fr.out s.1 s.2) i ≠
          msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i) ≤ miss

/-- **The run reaches every round** on a play the check accepts whose log is the device's, where its messages are
the check's at every round it reaches (the accept piece's: a run that accepts reaches its last round). -/
def RunsThrough {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
    (∀ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length,
      Rd.msg (Fr.out ω r) i = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r) i) →
    A.L.V.k x ≤ (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length

/-- **The messages fixed at every round**, from the messages fixed at the rounds the run reaches and the run reaching
every round. -/
theorem messagesFixed_of_reached (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (miss : ℝ≥0∞) (hR : RunsThrough Fr Rd Acc Good)
    (h : MessagesReached V A x θ F c σ Fr Rd Acc Good miss) : MessagesFixed V A x θ F c σ Fr Rd Acc Good miss := by
  refine le_trans (ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun s hs => ?_)) h
  obtain ⟨hA, hG, i, hi, hne⟩ := hs
  refine ⟨hA, hG, ?_⟩
  by_contra hall
  push Not at hall
  exact hne (hall i (lt_of_lt_of_le hi (hR s.1 s.2 hA hG hall)))

/-! ## The first difference -/

/-- **The run's history before round `i`**, as its prover sees it. -/
def runHist (t : Claims.Transcript) (i : ℕ) : List (Msg × Msg) := (t.msgs.take i).map RecDeploy.histOf

/-- **The history before round `i` of a run whose messages are `ms`**, against `A.L.V` at the statement `x` and the
coins `r`: each round's message, with the verifier's answer to the messages up to it. -/
noncomputable def playHist (A : Live₂) (x : Msg) (r : Coins A x) (ms : ℕ → Msg) (i : ℕ) : List (Msg × Msg) :=
  (List.range i).map fun j => (ms j, msgOf ((A.L.V.next x ((List.range (j + 1)).map ms)
    ((List.range (j + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1)

/-- **The play's messages**: each round's message the reading reads of the play, `[]` where it reads none. -/
def playMsgs {A : Live₂} (Rd : Reading A) (o : Verity.Exchange) (j : ℕ) : Msg := (Rd.msg o j).getD []

/-- **The messages first differ at round `i`**, on a play the check accepts whose log is the device's: the run reaches
round `i`, its message there is not the one the check reads, and at every round before it it is. -/
def MissAt {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (i : ℕ) (s : TapeOf V θ F c σ × Coins A x) : Prop :=
  Acc (Fr.out s.1 s.2) ∧ Good (Fr.out s.1 s.2) ∧
    i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).msgs.length ∧
    Rd.msg (Fr.out s.1 s.2) i ≠ msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) i ∧
    ∀ j < i, Rd.msg (Fr.out s.1 s.2) j = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) j

/-- **The messages fixed at the rounds the run reaches, round by round**: wherever each round's first difference is at
most `miss i` likely, the messages fixed at the rounds the run reaches at their sum. -/
def ReachedOfRounds (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (Fr : Frame V A x θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (miss : ℕ → ℝ≥0∞),
    (∀ i < A.L.V.k x, ENNReal.ofReal (Claims.Pr (MissAt Fr Rd Acc Good i)) ≤ miss i) →
      MessagesReached V A x θ F c σ Fr Rd Acc Good (∑ i ∈ Finset.range (A.L.V.k x), miss i)

/-- **The messages fixed at the rounds the run reaches, round by round**: a run whose messages differ from the check's
at a round it reaches differs first at one of them, below `k`. -/
theorem reachedOfRounds (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : ReachedOfRounds V A x θ F c σ := by
  classical
  intro Fr Rd Acc Good miss h
  unfold MessagesReached
  refine le_trans (ENNReal.ofReal_le_ofReal ((Claims.Pr_mono fun s hs => ?_).trans
    (Verity.Game.pr_exists_le (fun i => MissAt Fr Rd Acc Good i) (Finset.range (A.L.V.k x))))) ?_
  · obtain ⟨hA, hG, hex⟩ := hs
    have hlen := Claims.Recursion.run_length_le FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2
    refine ⟨Nat.find hex, Finset.mem_range.2 (lt_of_lt_of_le (Nat.find_spec hex).1 hlen), hA, hG,
      (Nat.find_spec hex).1, (Nat.find_spec hex).2, fun j hj => ?_⟩
    by_contra hne
    exact Nat.find_min hex hj ⟨lt_trans hj (Nat.find_spec hex).1, hne⟩
  · rw [ENNReal.ofReal_sum_of_nonneg fun i _ => Claims.Pr_nonneg _]
    exact Finset.sum_le_sum fun i hi => h i (Finset.mem_range.1 hi)

/-! ## Forking at one round

On a finite space split into classes, where each member's message is a function of its class (`F`) and a sampler
resamples a member of the same class uniformly (`res`), a member whose message is not its class's most likely one is at
most as likely as the square root of two members of one class having two different messages. -/

section Fork

open Classical

variable {β γ κ : Type} [Fintype β] [Fintype γ]

/-- **The members of `x`'s class** under `cls`. -/
noncomputable def classOf (cls : β → κ) (x : β) : Finset β := Finset.univ.filter fun x' => cls x' = cls x

theorem mem_classOf (cls : β → κ) {x x' : β} : x' ∈ classOf cls x ↔ cls x' = cls x := by
  simp [classOf]

theorem classOf_card_pos (cls : β → κ) (x : β) : (0 : ℝ) < (classOf cls x).card :=
  Nat.cast_pos.2 (Finset.card_pos.2 ⟨x, (mem_classOf cls).2 rfl⟩)

theorem classOf_eq (cls : β → κ) {x x' : β} (h : cls x' = cls x) : classOf cls x' = classOf cls x := by
  unfold classOf
  rw [h]

/-- **A sum is the sum of each member's class average.** -/
theorem sum_classAvg (cls : β → κ) (g : β → ℝ) :
    ∑ x, (∑ x' ∈ classOf cls x, g x') / (classOf cls x).card = ∑ x, g x := by
  have h : ∀ x, (∑ x' ∈ classOf cls x, g x') / ((classOf cls x).card : ℝ) =
      ∑ x', if cls x' = cls x then g x' / ((classOf cls x').card : ℝ) else 0 := by
    intro x
    rw [Finset.sum_div]
    unfold classOf
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun x' _ => ?_
    by_cases hx : cls x' = cls x
    · rw [if_pos hx, if_pos hx, hx]
    · rw [if_neg hx, if_neg hx]
  rw [Finset.sum_congr rfl fun x _ => h x, Finset.sum_comm]
  refine Finset.sum_congr rfl fun x' _ => ?_
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
  have hfe : (Finset.univ.filter fun x => cls x' = cls x) = classOf cls x' := by
    unfold classOf
    exact Finset.filter_congr fun _ _ => eq_comm
  rw [hfe, mul_div_left_comm, div_self (classOf_card_pos cls x').ne', mul_one]

/-- `x'` has a message, at `x`'s class, other than the class's most likely one. -/
def bad (cls : β → κ) (F : κ → β → Option Msg) (v : κ → Msg) (x x' : β) : Prop :=
  ∃ m, F (cls x) x' = some m ∧ m ≠ v (cls x)

/-- `x` and `x'` have two different messages at `x`'s class. -/
def fork (cls : β → κ) (F : κ → β → Option Msg) (x x' : β) : Prop :=
  ∃ u w, F (cls x) x = some u ∧ F (cls x) x' = some w ∧ u ≠ w

/-- The members of `x`'s class with a message other than its most likely one. -/
noncomputable def badCount (cls : β → κ) (F : κ → β → Option Msg) (v : κ → Msg) (x : β) : ℝ :=
  (((classOf cls x).filter (bad cls F v x)).card : ℝ)

/-- The members of `x`'s class with a message other than `x`'s. -/
noncomputable def forkCount (cls : β → κ) (F : κ → β → Option Msg) (x : β) : ℝ :=
  (((classOf cls x).filter (fork cls F x)).card : ℝ)

/-- **Fewer members differ from the most likely message than from any other.** -/
theorem card_ne_le (s : Finset β) (f : β → Option Msg) (u v : Msg)
    (h : (s.filter fun x => f x = some u).card ≤ (s.filter fun x => f x = some v).card) :
    (s.filter fun x => ∃ m, f x = some m ∧ m ≠ v).card ≤ (s.filter fun x => ∃ m, f x = some m ∧ m ≠ u).card := by
  have key : ∀ a, (s.filter fun x => f x = some a).card + (s.filter fun x => ∃ m, f x = some m ∧ m ≠ a).card =
      (s.filter fun x => (f x).isSome).card := by
    intro a
    rw [← Finset.card_filter_add_card_filter_not (s := s.filter fun x => (f x).isSome)
      (fun x => f x = some a), Finset.filter_filter, Finset.filter_filter]
    congr 2
    · exact Finset.filter_congr fun x _ => by cases f x <;> simp
    · exact Finset.filter_congr fun x _ => by cases f x <;> simp
  have hu := key u
  have hv := key v
  omega

/-- **The forking bound**: where each class's sampler is uniform on it (`hunif`), `v` is each class's most likely
message (`hmode`) and a miss has a message other than its class's most likely one (`hmiss`), a miss is at most as
likely as the square root of a member and a redrawn member of its class having two different messages. -/
theorem miss_le_sqrt [Nonempty β] [Nonempty γ] (cls : β → κ) (res : β → γ → β) (F : κ → β → Option Msg)
    (v : κ → Msg) (Miss : β → Prop)
    (hunif : ∀ x (φ : β → ℝ), (∑ y, φ (res x y)) * (classOf cls x).card =
      Fintype.card γ * ∑ x' ∈ classOf cls x, φ x')
    (hmode : ∀ x m, ((classOf cls x).filter fun x' => F (cls x) x' = some m).card ≤
      ((classOf cls x).filter fun x' => F (cls x) x' = some (v (cls x))).card)
    (hmiss : ∀ x, Miss x → bad cls F v x x) :
    Claims.Pr Miss ≤ Real.sqrt (Claims.Pr fun s : β × γ => fork cls F s.1 (res s.1 s.2)) := by
  have hn : (0 : ℝ) < Fintype.card β := Nat.cast_pos.2 Fintype.card_pos
  have hm : (0 : ℝ) < Fintype.card γ := Nat.cast_pos.2 Fintype.card_pos
  have hN : ∀ x, (0 : ℝ) < (classOf cls x).card := classOf_card_pos cls
  have hA : Claims.Pr Miss ≤ (∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β := by
    rw [Claims.Pr_eq_sum]
    refine div_le_div_of_nonneg_right ?_ hn.le
    calc (∑ x, if Miss x then (1 : ℝ) else 0) ≤ ∑ x, if bad cls F v x x then (1 : ℝ) else 0 :=
          Finset.sum_le_sum fun x _ => by
            by_cases hx : Miss x
            · rw [if_pos hx, if_pos (hmiss x hx)]
            · rw [if_neg hx]
              split_ifs <;> norm_num
      _ = ∑ x, badCount cls F v x / (classOf cls x).card := by
          rw [← sum_classAvg cls (fun x => if bad cls F v x x then (1 : ℝ) else 0)]
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [Finset.sum_boole]
          unfold badCount
          congr 3
          exact Finset.filter_congr fun x' hx' => by
            rw [mem_classOf] at hx'
            unfold bad
            rw [hx']
  have hrow : ∀ x, (∑ y, if fork cls F x (res x y) then (1 : ℝ) else 0) =
      Fintype.card γ * (forkCount cls F x / (classOf cls x).card) := by
    intro x
    have h1 := hunif x fun x' => if fork cls F x x' then 1 else 0
    have h2 : (∑ x' ∈ classOf cls x, if fork cls F x x' then (1 : ℝ) else 0) = forkCount cls F x := by
      rw [Finset.sum_boole]
      rfl
    rw [h2] at h1
    rw [mul_div_assoc', eq_div_iff (hN x).ne']
    exact h1
  have hge : ∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2 ≤
      ∑ x, forkCount cls F x / (classOf cls x).card := by
    rw [← sum_classAvg cls (fun x => forkCount cls F x / (classOf cls x).card)]
    refine Finset.sum_le_sum fun x _ => ?_
    have hstep : ∀ x'' ∈ (classOf cls x).filter (bad cls F v x), badCount cls F v x ≤ forkCount cls F x'' := by
      intro x'' hx''
      obtain ⟨hcx, u, hu, -⟩ := Finset.mem_filter.1 hx''
      have hc : cls x'' = cls x := (mem_classOf cls).1 hcx
      unfold badCount forkCount
      rw [classOf_eq cls hc]
      refine Nat.cast_le.2 (le_trans (le_of_eq ?_)
        (le_of_le_of_eq (card_ne_le (classOf cls x) (F (cls x)) u (v (cls x)) (hmode x u)) ?_))
      · exact congrArg Finset.card (Finset.filter_congr fun x' _ => Iff.rfl)
      congr 1
      refine Finset.filter_congr fun x' _ => ?_
      unfold fork
      constructor
      · rintro ⟨w, hw, hne⟩
        exact ⟨u, w, by rw [hc, hu], by rw [hc]; exact hw, Ne.symm hne⟩
      · rintro ⟨u', w, hu', hw, hne⟩
        rw [hc, hu] at hu'
        rw [hc] at hw
        cases hu'
        exact ⟨w, hw, Ne.symm hne⟩
    have hsq : badCount cls F v x ^ 2 ≤ ∑ x'' ∈ classOf cls x, forkCount cls F x'' := by
      calc badCount cls F v x ^ 2
            = ∑ _x'' ∈ (classOf cls x).filter (bad cls F v x), badCount cls F v x := by
              rw [Finset.sum_const, nsmul_eq_mul, sq]
              rfl
        _ ≤ ∑ x'' ∈ (classOf cls x).filter (bad cls F v x), forkCount cls F x'' := Finset.sum_le_sum hstep
        _ ≤ ∑ x'' ∈ classOf cls x, forkCount cls F x'' :=
            Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => by
              unfold forkCount
              positivity
    have hcl : ∀ x'' ∈ classOf cls x, forkCount cls F x'' / ((classOf cls x'').card : ℝ) =
        forkCount cls F x'' / ((classOf cls x).card : ℝ) := fun x'' hx'' => by
      rw [classOf_eq cls ((mem_classOf cls).1 hx'')]
    rw [Finset.sum_congr rfl hcl, ← Finset.sum_div, div_div, div_pow, ← sq]
    exact div_le_div_of_nonneg_right hsq (by positivity)
  have hB : (∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2) / Fintype.card β ≤
      Claims.Pr fun s : β × γ => fork cls F s.1 (res s.1 s.2) := by
    rw [Claims.Pr_eq_sum, Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul]
    dsimp only
    rw [Finset.sum_congr rfl fun x _ => hrow x, ← Finset.mul_sum, mul_comm (Fintype.card β : ℝ), ← div_div,
      mul_div_cancel_left₀ _ hm.ne']
    exact div_le_div_of_nonneg_right hge hn.le
  have hCS : (∑ x, badCount cls F v x / (classOf cls x).card) ^ 2 ≤
      (Fintype.card β : ℝ) * ∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2 := by
    have := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ => (1 : ℝ))
      (fun x => badCount cls F v x / (classOf cls x).card)
    simpa [Finset.card_univ] using this
  have hA0 : 0 ≤ (∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β :=
    div_nonneg (Finset.sum_nonneg fun x _ => div_nonneg (by unfold badCount; positivity) (hN x).le) hn.le
  calc Claims.Pr Miss ≤ (∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β := hA
    _ = Real.sqrt (((∑ x, badCount cls F v x / (classOf cls x).card) / Fintype.card β) ^ 2) :=
        (Real.sqrt_sq hA0).symm
    _ ≤ Real.sqrt (Claims.Pr fun s : β × γ => fork cls F s.1 (res s.1 s.2)) := by
        refine Real.sqrt_le_sqrt ?_
        rw [div_pow]
        calc (∑ x, badCount cls F v x / (classOf cls x).card) ^ 2 / (Fintype.card β : ℝ) ^ 2
            ≤ ((Fintype.card β : ℝ) * ∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2) /
                (Fintype.card β : ℝ) ^ 2 := div_le_div_of_nonneg_right hCS (by positivity)
          _ = (∑ x, (badCount cls F v x / (classOf cls x).card) ^ 2) / Fintype.card β := by
              rw [sq (Fintype.card β : ℝ), mul_div_mul_left _ _ hn.ne']
          _ ≤ _ := hB

end Fork

/-! ## The frame's prover at one round -/

section Mode

open Classical

/-- **The most likely value is at least as frequent as any other.** -/
theorem modeOn_spec {β : Type} [Fintype β] (S : β → Prop) (f : β → Option Msg) (m : Msg) :
    (Finset.univ.filter fun b => S b ∧ f b = some m).card ≤
      (Finset.univ.filter fun b => S b ∧ f b = some (modeOn S f)).card := by
  by_cases hex : ∃ m b, S b ∧ f b = some m
  · have hP : ∃ m₀, (∃ b, S b ∧ f b = some m₀) ∧ ∀ m', (Finset.univ.filter fun b => S b ∧ f b = some m').card ≤
        (Finset.univ.filter fun b => S b ∧ f b = some m₀).card := by
      obtain ⟨m₁, b₁, hb₁⟩ := hex
      have hne : ((Finset.univ.filter S).biUnion fun b => (f b).toFinset).Nonempty :=
        ⟨m₁, Finset.mem_biUnion.2 ⟨b₁, by simp [hb₁.1], by simp [hb₁.2]⟩⟩
      obtain ⟨m₀, hm₀, hmax⟩ := Finset.exists_max_image _
        (fun m => (Finset.univ.filter fun b => S b ∧ f b = some m).card) hne
      refine ⟨m₀, ?_, fun m' => ?_⟩
      · obtain ⟨b, hb, hfb⟩ := Finset.mem_biUnion.1 hm₀
        exact ⟨b, (Finset.mem_filter.1 hb).2, Option.mem_toFinset.1 hfb⟩
      · by_cases hm' : m' ∈ (Finset.univ.filter S).biUnion fun b => (f b).toFinset
        · exact hmax m' hm'
        · have h0 : (Finset.univ.filter fun b => S b ∧ f b = some m') = ∅ := by
            refine Finset.filter_eq_empty_iff.2 fun b _ hb => hm' ?_
            exact Finset.mem_biUnion.2 ⟨b, by simp [hb.1], by simp [hb.2]⟩
          rw [h0, Finset.card_empty]
          exact Nat.zero_le _
    unfold modeOn
    rw [if_pos hex]
    exact (Classical.epsilon_spec hP).2 m
  · have h0 : (Finset.univ.filter fun b => S b ∧ f b = some m) = ∅ :=
      Finset.filter_eq_empty_iff.2 fun b _ hb => hex ⟨m, b, hb⟩
    rw [h0, Finset.card_empty]
    exact Nat.zero_le _

theorem shows_merge {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (y : Coins A x) : Shows hist (merge hist y) := by
  intro j c h
  simp only [merge, h, Option.getD_some]

theorem merge_of_shows {A : Live₂} {x : Msg} {hist : List (Msg × Msg)} {r : Coins A x} (h : Shows hist r) :
    merge hist r = r := by
  funext j
  unfold merge
  rcases hs : shownAt hist j.val with _ | c
  · rfl
  · exact (h j c hs).symm

/-- **Every coin the history shows fixes as many resamples**: the resamples that `merge` takes to two coins the history
shows are as many. -/
theorem fiber_card {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) {r r' : Coins A x} (hr : Shows hist r)
    (hr' : Shows hist r') :
    (Finset.univ.filter fun y => merge hist y = r).card = (Finset.univ.filter fun y => merge hist y = r').card := by
  refine Finset.card_nbij' (fun y j => if (shownAt hist j.val).isSome then y j else r' j)
    (fun y j => if (shownAt hist j.val).isSome then y j else r j) ?_ ?_ ?_ ?_
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy ⊢
    funext j
    unfold merge
    rcases hs : shownAt hist j.val with _ | c
    · simp [hs]
    · exact (hr' j c hs).symm
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy ⊢
    funext j
    unfold merge
    rcases hs : shownAt hist j.val with _ | c
    · simp [hs]
    · exact (hr j c hs).symm
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy
    funext j
    rcases hs : shownAt hist j.val with _ | c
    · have := congrFun hy j
      simp only [merge, hs, Option.getD_none] at this
      simp [hs, this]
    · simp [hs]
  · intro y hy
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq, Finset.mem_filter] at hy
    funext j
    rcases hs : shownAt hist j.val with _ | c
    · have := congrFun hy j
      simp only [merge, hs, Option.getD_none] at this
      simp [hs, this]
    · simp [hs]

/-- **`merge` resamples uniformly among the coins the history shows.** -/
theorem sum_merge {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (φ : Coins A x → ℝ) :
    (∑ y, φ (merge hist y)) * ((Finset.univ.filter fun r : Coins A x => Shows hist r).card : ℝ) =
      (Fintype.card (Coins A x) : ℝ) * ∑ r ∈ Finset.univ.filter (fun r => Shows hist r), φ r := by
  have hS : ∀ y ∈ (Finset.univ : Finset (Coins A x)),
      merge hist y ∈ Finset.univ.filter fun r : Coins A x => Shows hist r :=
    fun y _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, shows_merge hist y⟩
  have hfib : ∀ ψ : Coins A x → ℝ, ∑ y, ψ (merge hist y) =
      ((Finset.univ.filter fun y : Coins A x => merge hist y = complete hist).card : ℝ) *
        ∑ r ∈ Finset.univ.filter (fun r : Coins A x => Shows hist r), ψ r := by
    intro ψ
    rw [← Finset.sum_fiberwise_of_maps_to hS (fun y => ψ (merge hist y)), Finset.mul_sum]
    refine Finset.sum_congr rfl fun r hr => ?_
    rw [Finset.sum_congr rfl (fun y hy => by rw [(Finset.mem_filter.1 hy).2]), Finset.sum_const, nsmul_eq_mul,
      fiber_card hist (Finset.mem_filter.1 hr).2 (shows_merge hist _)]
    rfl
  have h1 := hfib fun _ => 1
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one] at h1
  rw [hfib φ, h1]
  ring

end Mode

/-! ## The miss at one round -/

/-- **The message the check reads opens the digest the play records** (the form's parse's, and the sessions' reading
for the sessions' rounds): on a play the check accepts whose log is the device's, the check reads a message at each
round below `k`, and `Dg` reads its SHA-512 for that round. -/
def MsgOpens (A : Live₂) (x : Msg) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) : Prop :=
  ∀ o, Acc o → Good o → ∀ i < A.L.V.k x, ∃ m, Rd.msg o i = some m ∧ Dg o i = some (Verity.sha512Msg m)

/-- **The digest for round `i` is fixed by the coins the history before it shows** (the coin map's and the cut's: the
device records the digest before it releases round `i`'s coin, and the play before it reads no coin the history
does not show): on a play the check accepts whose log is the device's, the play at any run coins that agree with the
coins the play's history before round `i` shows records the same digest for round `i`. -/
def DigestBefore {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy (V.game θ F c)} (Fr : Frame V A x θ F c σ) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ) : Prop :=
  ∀ (ω : TapeOf V θ F c σ) (r r' : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
    Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r' → Dg (Fr.out ω r') i = Dg (Fr.out ω r) i

/-- **The verifier's answers show the run's coins, and depend on them only through what they show** (a fact about
`SpotFlock.V`: an answer is its round's coin's bytes, or a function of the messages and the coins earlier answers
showed): at any prover, the run's coins agree with every coin its history before round `i` shows, and any coins that
do give the same rounds before `i`. -/
def RunShown (A : Live₂) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x) (i : ℕ),
    Shows (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) r ∧
      ∀ r', Shows (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) r' →
        (Claims.run FlockDeploy.sha512 P A.L.V x ω r').msgs.take i =
          (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take i

/-- **The frame's run's history before round `i`**, at the tape `ω` and the run's coins `r`. -/
noncomputable def histAt (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
    (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ) (ω : TapeOf V θ F c σ) (r : Coins A x) :
    List (Msg × Msg) :=
  runHist (Claims.run FlockDeploy.sha512 (prover V θ F c σ place Rd Dg) A.L.V x ω r) i

/-- **Round `i`'s two-continuation finder**, at `q` SHA-512 calls on every coin: on a tape `ω` and run coins `r` and
`y`, the frame's run's history before round `i` at `r`, and the coins it shows with `y` at every other round
(`merge`); its output the two messages the plays at the two coins read at round `i` that open the history's digest,
where they differ. -/
noncomputable def msgFinder (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i q : ℕ) : Claims.Finder where
  S := (TapeOf V θ F c σ × Coins A x) × Coins A x
  out s :=
    match opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.1.2,
      opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2)
        (merge (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.2) with
    | some m, some m' => if m = m' then none else some (m, m')
    | _, _ => none
  cost _ := q

/-- **Round `i`'s miss is at most the square root of its finder's success**: at the frame's prover, where the message
the check reads opens the digest the play records, the digest for round `i` is fixed by the coins the history before
it shows, and the verifier's answers show the run's coins and depend on them only through what they show. -/
def MissBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i q : ℕ),
    MsgOpens A x Rd Acc Good Dg → DigestBefore (frame V A x θ F c σ place Rd Dg) Rd Acc Good Dg i → RunShown A x →
      ENNReal.ofReal (Claims.Pr (MissAt (frame V A x θ F c σ place Rd Dg) Rd Acc Good i)) ≤
        ENNReal.ofReal (Real.sqrt ((msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg))

/-! ## The miss at one round, proved -/

theorem opening_some {A : Live₂} {x : Msg} {Ω : Type} {out : Ω → Coins A x → Verity.Exchange} {Rd : Reading A}
    {Dg : Verity.Exchange → ℕ → Option Msg} {i : ℕ} {ω : Ω} {hist : List (Msg × Msg)} {r : Coins A x} {u : Msg}
    (h : opening out Rd Dg i ω hist r = some u) :
    Rd.msg (out ω r) i = some u ∧ Dg (out ω (complete hist)) i = some (Verity.sha512Msg u) := by
  unfold opening at h
  rw [Option.filter_eq_some_iff] at h
  exact ⟨h.1, of_decide_eq_true h.2⟩

theorem opening_of {A : Live₂} {x : Msg} {Ω : Type} {out : Ω → Coins A x → Verity.Exchange} {Rd : Reading A}
    {Dg : Verity.Exchange → ℕ → Option Msg} {i : ℕ} {ω : Ω} {hist : List (Msg × Msg)} {r : Coins A x} {u : Msg}
    (h1 : Rd.msg (out ω r) i = some u) (h2 : Dg (out ω (complete hist)) i = some (Verity.sha512Msg u)) :
    opening out Rd Dg i ω hist r = some u := by
  unfold opening
  rw [h1, Option.filter_some_eq_some]
  exact decide_eq_true h2

theorem runHist_length (t : Claims.Transcript) {i : ℕ} (h : i ≤ t.msgs.length) : (runHist t i).length = i := by
  simp [runHist, h]

theorem outMsg_eq (o : Out) : Claims.Recursion.outMsg o = msgOf o := by
  cases o <;> rfl

/-- **A run's message at a round it reaches is the prover's at the run's history before it.** -/
theorem run_msgAt {Ω : Type} (pick : Ω → List (Msg × Msg) → Msg) {Coin : Type} [Inhabited Coin]
    (Vr : Verifier Coin) (x : Msg) (ω : Ω) (r : Fin (Vr.k x) → Coin) (i : ℕ)
    (h : i < (Claims.run FlockDeploy.sha512 (proverOf pick) Vr x ω r).msgs.length) :
    msgAt (Claims.run FlockDeploy.sha512 (proverOf pick) Vr x ω r) i =
      some (pick ω (runHist (Claims.run FlockDeploy.sha512 (proverOf pick) Vr x ω r) i)) := by
  obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 (proverOf pick) Vr x ω r i h
  unfold msgAt runHist
  rw [hget, ← hnew]
  rfl

theorem map_fst_take_eq {L : List (Msg × Out)} {ms : ℕ → Msg} {n : ℕ} (hn : n ≤ L.length)
    (h : ∀ l < n, (L[l]?).map Prod.fst = some (ms l)) : (L.take n).map Prod.fst = (List.range n).map ms := by
  apply List.ext_getElem?
  intro l
  rw [List.getElem?_map, List.getElem?_take, List.getElem?_map]
  by_cases hl : l < n
  · rw [if_pos hl, h l hl, List.getElem?_range hl]
    rfl
  · rw [if_neg hl, List.getElem?_eq_none (l := List.range n) (by simpa using hl)]
    rfl

/-- **The play's history is the run's** where the run's messages before round `i` are the play's. -/
theorem playHist_eq {Ω : Type} (A : Live₂) (x : Msg) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A x) (ms : ℕ → Msg)
    (i : ℕ) (hi : i ≤ (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length)
    (h : ∀ l < i, msgAt (Claims.run FlockDeploy.sha512 P A.L.V x ω r) l = some (ms l)) :
    playHist A x r ms i = runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i := by
  apply List.ext_getElem?
  intro n
  unfold playHist runHist
  rw [List.getElem?_map, List.getElem?_map, List.getElem?_take]
  by_cases hn : n < i
  · rw [List.getElem?_range hn, if_pos hn]
    obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 P A.L.V x ω r n (by omega)
    rw [hget]
    have hp : ((P.next x (new.map RecDeploy.histOf) ω).eval FlockDeploy.sha512).1 = ms n := by
      have := h n hn
      unfold msgAt at this
      rw [hget] at this
      exact Option.some.inj this
    have hfst : new.map Prod.fst = (List.range n).map ms := by
      rw [hnew]
      exact map_fst_take_eq (by omega) fun l hl => h l (by omega)
    simp only [Option.map_some]
    rw [hp, hfst, List.range_succ, List.map_append, List.map_singleton]
    simp only [RecDeploy.histOf, outMsg_eq]
  · rw [if_neg hn, List.getElem?_eq_none (l := List.range i) (by simpa using hn)]
    rfl

/-- **The runs at coins with one history before round `i` are the coins it shows.** -/
theorem hist_eq_iff {A : Live₂} {x : Msg} (hS : RunShown A x) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r r' : Coins A x) (i : ℕ) :
    runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r') i = runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i ↔
      Shows (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) r' := by
  constructor
  · intro h
    rw [← h]
    exact (hS _ P ω r' i).1
  · intro h
    unfold runHist
    rw [(hS _ P ω r i).2 r' h]

section Round

variable (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
  (σ : Verity.Game.Strategy (V.game θ F c)) (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ)
  (Rd : Reading A) (Dg : Verity.Exchange → ℕ → Option Msg) (i : ℕ)

/-- A tape and run coins' class at round `i`: the tape, and the frame's run's history before round `i`. -/
noncomputable def clsOf (s : TapeOf V θ F c σ × Coins A x) : TapeOf V θ F c σ × List (Msg × Msg) :=
  (s.1, histAt V θ F c σ place Rd Dg i s.1 s.2)

/-- The resample of a tape and run coins within their class. -/
noncomputable def resOf (s : TapeOf V θ F c σ × Coins A x) (y : Coins A x) : TapeOf V θ F c σ × Coins A x :=
  (s.1, merge (histAt V θ F c σ place Rd Dg i s.1 s.2) y)

/-- The message read at round `i` that opens the class's digest. -/
noncomputable def openAt (K : TapeOf V θ F c σ × List (Msg × Msg)) (s : TapeOf V θ F c σ × Coins A x) :
    Option Msg :=
  opening (outOf V θ F c σ place) Rd Dg i K.1 K.2 s.2

/-- The class's most likely message at round `i`. -/
noncomputable def modeAt (K : TapeOf V θ F c σ × List (Msg × Msg)) : Msg :=
  modeOn (Shows (A := A) (x := x) K.2) (opening (outOf V θ F c σ place) Rd Dg i K.1 K.2)

open Classical in
theorem classOf_clsOf (hS : RunShown A x) (s : TapeOf V θ F c σ × Coins A x) :
    classOf (clsOf V θ F c σ place Rd Dg i) s =
      (Finset.univ.filter fun r' => Shows (histAt V θ F c σ place Rd Dg i s.1 s.2) r').map
        ⟨fun r' => (s.1, r'), fun a b h => (Prod.mk.inj h).2⟩ := by
  ext ⟨ω', r'⟩
  rw [Finset.mem_map]
  unfold classOf
  simp only [Finset.mem_filter]
  constructor
  · rintro ⟨-, h⟩
    have h1 : ω' = s.1 := congrArg Prod.fst h
    have h2 : histAt V θ F c σ place Rd Dg i ω' r' = histAt V θ F c σ place Rd Dg i s.1 s.2 := congrArg Prod.snd h
    subst h1
    exact ⟨r', ⟨Finset.mem_univ _, (hist_eq_iff hS (prover V θ F c σ place Rd Dg) s.1 s.2 r' i).1 h2⟩, rfl⟩
  · rintro ⟨a, ha, he⟩
    have he' : (s.1, a) = (ω', r') := he
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj he'
    exact ⟨Finset.mem_univ _, Prod.ext rfl
      ((hist_eq_iff hS (prover V θ F c σ place Rd Dg) s.1 s.2 _ i).2 ha.2)⟩

end Round

/-- **Round `i`'s miss is at most the square root of its finder's success.** -/
theorem missBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : MissBound V A x θ F c σ := by
  classical
  intro place Rd Acc Good Dg i q hO hD hS
  have hunif : ∀ (s : TapeOf V θ F c σ × Coins A x) (φ : TapeOf V θ F c σ × Coins A x → ℝ),
      (∑ y, φ (resOf V θ F c σ place Rd Dg i s y)) * (classOf (clsOf V θ F c σ place Rd Dg i) s).card =
        Fintype.card (Coins A x) * ∑ x' ∈ classOf (clsOf V θ F c σ place Rd Dg i) s, φ x' := by
    intro s φ
    rw [classOf_clsOf V θ F c σ place Rd Dg i hS, Finset.sum_map, Finset.card_map]
    exact sum_merge (histAt V θ F c σ place Rd Dg i s.1 s.2) fun r' => φ (s.1, r')
  have hmode : ∀ (s : TapeOf V θ F c σ × Coins A x) m,
      ((classOf (clsOf V θ F c σ place Rd Dg i) s).filter fun x' =>
          openAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i s) x' = some m).card ≤
        ((classOf (clsOf V θ F c σ place Rd Dg i) s).filter fun x' =>
          openAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i s) x' =
            some (modeAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i s))).card := by
    intro s m
    rw [classOf_clsOf V θ F c σ place Rd Dg i hS, Finset.filter_map, Finset.filter_map, Finset.card_map,
      Finset.card_map, Finset.filter_filter, Finset.filter_filter]
    exact modeOn_spec _ _ m
  have hmiss : ∀ s, MissAt (frame V A x θ F c σ place Rd Dg) Rd Acc Good i s →
      bad (clsOf V θ F c σ place Rd Dg i) (openAt V θ F c σ place Rd Dg i) (modeAt V θ F c σ place Rd Dg i) s s := by
    rintro ⟨ω, r⟩ hs
    dsimp only [MissAt] at hs
    obtain ⟨hA, hG, hlen, hne, hagree⟩ := hs
    have hk : i < A.L.V.k x := lt_of_lt_of_le hlen (Claims.Recursion.run_length_le _ _ _ _ _ _)
    obtain ⟨m, hm, hdg⟩ := hO _ hA hG i hk
    have hKlen : (histAt V θ F c σ place Rd Dg i ω r).length = i := runHist_length _ hlen.le
    have hmsg : msgAt (Claims.run FlockDeploy.sha512 (frame V A x θ F c σ place Rd Dg).P A.L.V x ω r) i =
        some (modePick (outOf V θ F c σ place) Rd Dg ω (histAt V θ F c σ place Rd Dg i ω r)) :=
      run_msgAt (modePick (outOf V θ F c σ place) Rd Dg) A.L.V x ω r i hlen
    have hplay : playHist A x r (playMsgs Rd ((frame V A x θ F c σ place Rd Dg).out ω r)) i =
        histAt V θ F c σ place Rd Dg i ω r := by
      refine playHist_eq A x (frame V A x θ F c σ place Rd Dg).P ω r _ i hlen.le fun l hl => ?_
      obtain ⟨ml, hml, -⟩ := hO _ hA hG l (lt_trans hl hk)
      rw [← hagree l hl, hml]
      simp [playMsgs, hml]
    have hdig : Dg ((frame V A x θ F c σ place Rd Dg).out ω (complete (histAt V θ F c σ place Rd Dg i ω r))) i =
        some (Verity.sha512Msg m) := by
      rw [← hdg]
      refine hD ω r _ hA hG ?_
      rw [hplay]
      exact shows_merge _ _
    have hv : modePick (outOf V θ F c σ place) Rd Dg ω (histAt V θ F c σ place Rd Dg i ω r) =
        modeAt V θ F c σ place Rd Dg i (clsOf V θ F c σ place Rd Dg i (ω, r)) := by
      unfold modePick
      rw [hKlen]
      rfl
    unfold bad
    refine Exists.intro m (And.intro (opening_of hm hdig) fun hmv => hne ?_)
    rw [hm, hmsg, hmv, hv]
  have key := miss_le_sqrt (clsOf V θ F c σ place Rd Dg i) (resOf V θ F c σ place Rd Dg i)
    (openAt V θ F c σ place Rd Dg i) (modeAt V θ F c σ place Rd Dg i)
    (MissAt (frame V A x θ F c σ place Rd Dg) Rd Acc Good i) hunif hmode hmiss
  have hfin : (Claims.Pr fun s : (TapeOf V θ F c σ × Coins A x) × Coins A x =>
      fork (clsOf V θ F c σ place Rd Dg i) (openAt V θ F c σ place Rd Dg i) s.1
        (resOf V θ F c σ place Rd Dg i s.1 s.2)) ≤
      (msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg := by
    unfold Claims.Finder.Pr
    refine Claims.Pr_mono fun s hs => ?_
    obtain ⟨u, w, hu, hw, huw⟩ := hs
    have hu' : opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.1.2 =
        some u := hu
    have hw' : opening (outOf V θ F c σ place) Rd Dg i s.1.1 (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2)
        (merge (histAt V θ F c σ place Rd Dg i s.1.1 s.1.2) s.2) = some w := hw
    refine Exists.intro u (Exists.intro w (And.intro ?_ (And.intro huw ?_)))
    · show (msgFinder V θ F c σ place Rd Dg i q).out s = some (u, w)
      simp [msgFinder, hu', hw', huw]
    · have h1 := (opening_some hu').2
      have h2 := (opening_some hw').2
      rw [h1] at h2
      exact Option.some.inj h2
  exact le_trans (ENNReal.ofReal_le_ofReal key) (ENNReal.ofReal_le_ofReal (Real.sqrt_le_sqrt hfin))

/-! ## The messages' bound -/

/-- **The messages' bound at the frame's prover**: where the message the check reads opens the digest the play records,
each round's digest is fixed by the coins the history before it shows, the verifier's answers show the run's coins and
depend on them only through what they show, and the run reaches every round, `MessagesFixed` holds at the sum over
the rounds of the square roots of their finders' successes. -/
def MessagesBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ),
    MsgOpens A x Rd Acc Good Dg → (∀ i < A.L.V.k x, DigestBefore (frame V A x θ F c σ place Rd Dg) Rd Acc Good Dg i) →
      RunShown A x → RunsThrough (frame V A x θ F c σ place Rd Dg) Rd Acc Good →
      MessagesFixed V A x θ F c σ (frame V A x θ F c σ place Rd Dg) Rd Acc Good
        (∑ i ∈ Finset.range (A.L.V.k x),
          ENNReal.ofReal (Real.sqrt ((msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg)))

/-- **The messages' bound from each round's miss.** -/
theorem messagesBound_of (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (hM : MissBound V A x θ F c σ) :
    MessagesBound V A x θ F c σ := by
  intro place Rd Acc Good Dg q hO hB hS hT
  exact messagesFixed_of_reached V A x θ F c σ _ Rd Acc Good _ hT
    (reachedOfRounds V A x θ F c σ _ Rd Acc Good _ fun i hi => hM place Rd Acc Good Dg i q hO (hB i hi) hS)

/-- **The messages' bound at the frame's prover.** -/
theorem messagesBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : MessagesBound V A x θ F c σ :=
  messagesBound_of V A x θ F c σ (missBound V A x θ F c σ)

end Verity.SampledTwo.Messages
