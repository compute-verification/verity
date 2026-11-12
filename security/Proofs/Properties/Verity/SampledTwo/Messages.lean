module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Flock.Claims.Recursion.DeploySessions
public import Proofs.Flock.Claims.Prob

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
* **The first difference** (`runHist`, `MissAt`): on a play where the messages differ at a reached round, they differ
  first at some round `i`, where the run's history is the play's (`playHist`).
* **The miss at one round** (`MsgOpens`, `DigestBefore`, `RunShown`, `merge`, `msgFinder`, `MissBound`): where the
  message the check reads opens the digest the play records (`MsgOpens`, the form's parse's), that digest is fixed by
  the coins the history shows (`DigestBefore`, the coin map's and the cut's), and the verifier's answers depend on the
  coins only through the coins they show (`RunShown`, a fact about `SpotFlock.V`), the round's miss is at most the
  square root of a two-continuation finder's success: it resamples the coins the history does not show (`merge`) and
  outputs the two messages the two plays read when they differ, a SHA-512 collision since both open one digest.
* **The messages' bound** (`MessagesBound`): `MessagesReached` at the sum over the rounds of those square roots.

Statements and their bookkeeping; `MissBound` and `MessagesBound` are to be proved. Nothing imports this module.
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

/-- **The run's coins agree with every coin the history `hist` shows**: at each round before the history's end whose
answer shows a coin, the run's coin is that coin. -/
def Shows {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (r : Coins A x) : Prop :=
  ∀ (j : Fin (A.L.V.k x)) (e : Msg × Msg), hist[j.val]? = some e → ∀ c, shownCoin e.2 = some c → r j = c

/-- **The coins the history `hist` shows, `y` at every other round.** -/
noncomputable def merge {A : Live₂} {x : Msg} (hist : List (Msg × Msg)) (y : Coins A x) : Coins A x :=
  fun j => ((hist[j.val]?).bind fun e => shownCoin e.2).getD (y j)

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
  push_neg at hall
  exact hne (hall i (lt_of_lt_of_le hi (hR s.1 s.2 hA hG hall)))

/-! ## The first difference -/

/-- **The run's history before round `i`**, as its prover sees it. -/
def runHist (t : Claims.Transcript) (i : ℕ) : List (Msg × Msg) := Claims.histOf (t.msgs.take i)

/-- **The history before round `i` of a run whose messages are `ms`**, against `A.L.V` at the statement `x` and the
coins `r`: each round's message, with the verifier's answer to the messages up to it. -/
def playHist (A : Live₂) (x : Msg) (r : Coins A x) (ms : ℕ → Msg) (i : ℕ) : List (Msg × Msg) :=
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

/-- **The verifier's answers depend on the run's coins only through the coins they show** (a fact about `SpotFlock.V`:
an answer is a coin's bytes, or a function of the messages and the coins earlier answers showed): at any prover, two
run coins that agree with every coin the history before round `i` shows give the same rounds before `i`. -/
def RunShown (A : Live₂) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r r' : Coins A x) (i : ℕ),
    Shows (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) r' →
      (Claims.run FlockDeploy.sha512 P A.L.V x ω r').msgs.take i =
        (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take i

/-- **Round `i`'s two-continuation finder**, at `q` SHA-512 calls on every coin: on a tape `ω` and run coins `r` and
`y`, the play's history before round `i` at `r`, and the coins it shows with `y` at every other round (`merge`); its
output the two messages the plays at the two coins read at round `i` that open the history's digest, where they
differ. -/
noncomputable def msgFinder (V : Vocab d α) {A : Live₂} {x : Msg} (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i q : ℕ) : Claims.Finder where
  S := TapeOf V θ F c σ × Coins A x × Coins A x
  out s :=
    let hist := playHist A x s.2.1 (playMsgs Rd (outOf V θ F c σ place s.1 s.2.1)) i
    match opening (outOf V θ F c σ place) Rd Dg i s.1 hist s.2.1,
      opening (outOf V θ F c σ place) Rd Dg i s.1 hist (merge hist s.2.2) with
    | some m, some m' => if m = m' then none else some (m, m')
    | _, _ => none
  cost _ := q

/-- **Round `i`'s miss is at most the square root of its finder's success**: at the frame's prover, where the message
the check reads opens the digest the play records, the digest for round `i` is fixed by the coins the history before
it shows, and the verifier's answers depend on the coins only through those they show. -/
def MissBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) (i q : ℕ),
    MsgOpens A x Rd Acc Good Dg → DigestBefore (frame V A x θ F c σ place Rd Dg) Rd Acc Good Dg i → RunShown A x →
      ENNReal.ofReal (Claims.Pr (MissAt (frame V A x θ F c σ place Rd Dg) Rd Acc Good i)) ≤
        ENNReal.ofReal (Real.sqrt ((msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg))

/-! ## The messages' bound -/

/-- **The messages' bound at the frame's prover**: where the message the check reads opens the digest the play records,
each round's digest is fixed by the coins the history before it shows, the verifier's answers depend on the coins only
through those they show, and the run reaches every round, `MessagesFixed` holds at the sum over the rounds of the
square roots of their finders' successes. -/
def MessagesBound (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) : Prop :=
  ∀ (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ),
    MsgOpens A x Rd Acc Good Dg → (∀ i < A.L.V.k x, DigestBefore (frame V A x θ F c σ place Rd Dg) Rd Acc Good Dg i) →
      RunShown A x → RunsThrough (frame V A x θ F c σ place Rd Dg) Rd Acc Good →
      MessagesFixed V A x θ F c σ (frame V A x θ F c σ place Rd Dg) Rd Acc Good
        (∑ i ∈ Finset.range (A.L.V.k x),
          ENNReal.ofReal (Real.sqrt ((msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg)))

/-- **The messages' bound from its two steps**: the messages fixed at the rounds the run reaches, round by round, and
each round's miss. -/
theorem messagesBound_of (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (hR : ReachedOfRounds V A x θ F c σ) (hM : MissBound V A x θ F c σ) :
    MessagesBound V A x θ F c σ := by
  intro place Rd Acc Good Dg q hO hB hS hT
  exact messagesFixed_of_reached V A x θ F c σ _ Rd Acc Good _ hT
    (hR _ Rd Acc Good _ fun i hi => hM place Rd Acc Good Dg i q hO (hB i hi) hS)

end Verity.SampledTwo.Messages
