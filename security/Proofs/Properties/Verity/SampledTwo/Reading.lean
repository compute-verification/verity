module

public import Proofs.Properties.Verity.SampledTwo.Projection
public import VStarRecursive.SampledTwo

/-!
# What the two-stage form's check reads of a play

The projection's pieces (`Projection.lean`) read a play through a `Reading` and the form's acceptance `Acc`. This module
gives both at the two-stage form (`VStar.Recursive.SampledTwo`), on the sampled component's case of a play, which the
instance's composite decodes (`CaseOf`):

* **The case** (`Case`, `Decoded`): the parameters', commitment's and evidence's bytes and the blobs, decoded by the
  form's own parsers.
* **The acceptance** (`AcceptsOn`, `FormAccepts`): the form's check accepts the case with its switch on
  (`checkBlobsAtWith true`), so that nothing a piece proves of an accepted play rests on the switch being off.
* **The run of record's input** (`inputOfCase`): the `Spot.Run.Input` the check hands `Spot.Run.run` (`inputAt`), whose
  acceptance RUNCOINS's `RunAcceptsPlay` reads (`AcceptedTwo`).
* **The reading** (`formReading`, at the coin map's reading of the device's releases, `dev`, a parameter), round by
  round of `SpotFlock.V h`, as RUNCOINS's play reads the run
  (`proverSeq`, `seqCoins`, `PlayCoins`):
  * the statement: the boundary registration's record, its canonical JSON's bytes (`recordMsg`, which
    `SpotFlock.jsonMsg` reads back);
  * round 0: no message, and the coin whose bytes are the first stage's bytes and then zeros (`coinOf`);
  * rounds 1 to `stage`: round 1's message the interior registration's record and the later rounds' none; their
    coins the second stage's bytes and then zeros, coin by coin (`stageCoinsOf`);
  * round `stage + 1`: the session statements' message, and no coin read;
  * a session's round: the session's message and coin there; and the verdict's round no message;
  * the sessions' messages and coins and the statements' message are the sessions' pieces' to define
    (`SessionReading`), on the files the check verified each session on;
  * a coin read agrees with a run's coin (`agree`) on the bytes the run's bytes carry of it (`prefixAgree`): at round
    0 the first stage's bytes, at round `j + 1` the second stage's bytes from `j · L` on, at most `L`; at a session's
    round as the sessions' pieces say.

Definitions only. Nothing imports this module.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)

/-- **A sampled component's case**: its parameters', commitment's and evidence's bytes, and the blobs. -/
structure Case where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray
  blobs : Core.Blobs

/-- **The case decodes** to the form's parameters `p`, the boundary registration `c` and the form's evidence `e`. -/
def Decoded (k : Case) (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (e : VStar.Recursive.SampledTwo.Evidence) : Prop :=
  (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) = .ok p ∧
    Spot.OneStage.commitmentOf k.commitment = .ok c ∧ VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e

/-- **The form's check accepts the case, with its switch on.** -/
def AcceptsOn (k : Case) : Prop :=
  ∃ p c e pub prof, Decoded k p c e ∧
    VStar.Recursive.SampledTwo.checkBlobsAtWith true p c e.source e k.blobs = .accept pub prof

/-- **The run of record's input on the case**: the one the check hands `Spot.Run.run`; none where the case doesn't
decode or the check refuses before the run. -/
def inputOfCase (k : Case) : Option Spot.Run.Input :=
  match (do
      let p ← Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson
      let c ← Spot.OneStage.commitmentOf k.commitment
      let e ← VStar.Recursive.SampledTwo.evidenceOf k.evidence
      let (_, _, _, i) ← VStar.Recursive.SampledTwo.inputAt p c e.source e k.blobs
      pure i : Flock.V Spot.Run.Input) with
  | .ok i => some i
  | .error _ => none

/-- **The sampled component's case of a play**, as the instance's composite decodes it. -/
abbrev CaseOf : Type := Verity.Exchange → Option Case

/-- **The form accepts the play**: the play's case is accepted by the form's check with its switch on. -/
def FormAccepts (C : CaseOf) (o : Verity.Exchange) : Prop :=
  ∃ k, C o = some k ∧ AcceptsOn k

/-- **A record's message**: its canonical JSON's bytes. -/
def recordMsg (j : Json) : Option Msg :=
  (Flock.canon j).toOption.map fun s => s.toUTF8.data.toList

/-- **The coin the run's bytes `b` are**: the coin whose bytes are `b` and then zeros, as `RunsAs.PlayCoins` reads the
first stage's; none past one coin's bytes. -/
noncomputable def coinOf (b : ByteArray) : Option FlockDeploy.Coin :=
  if b.size ≤ SpotFlock.L then
    some (Classical.epsilon fun c => SpotFlock.src c = b ++ ⟨Array.replicate (SpotFlock.L - b.size) 0⟩)
  else none

/-- **The second stage's coins the run's bytes `b` are**: coin by coin, those whose bytes are `b` and then zeros, as
`RunsAs.PlayCoins` reads them; none past `stage` coins' bytes. -/
noncomputable def stageCoinsOf (h : SpotFlock.Held) (b : ByteArray) : Option (SpotFlock.Stage h) :=
  if b.size ≤ h.stage * SpotFlock.L then
    some (Classical.epsilon fun cs : SpotFlock.Stage h =>
      SpotFlock.stageSrc cs = b ++ ⟨Array.replicate (h.stage * SpotFlock.L - b.size) 0⟩)
  else none

/-- **What the sessions' pieces define of a play**: the session statements' message (round `stage + 1`'s), and
session `t`'s message and coin at its round `i`, each as the check read it from the files it verified the session on.
Workers E (the coins) and F (the messages) fix these fields. -/
structure SessionReading where
  stmts : Verity.Exchange → Option Msg
  msg : Verity.Exchange → ℕ → ℕ → Option Msg
  coin : Verity.Exchange → ℕ → ℕ → Option FlockDeploy.Coin
  agree : Verity.Exchange → ℕ → ℕ → FlockDeploy.Coin → FlockDeploy.Coin → Prop

/-- **Two coins agree on their first `w` bytes.** -/
def prefixAgree (w : ℕ) (c c' : FlockDeploy.Coin) : Prop :=
  (SpotFlock.src c).extract 0 w = (SpotFlock.src c').extract 0 w

/-- How many bytes of the second stage's coin `j` the run's bytes `b` carry. -/
def stageWidth (b : ByteArray) (j : ℕ) : ℕ := min SpotFlock.L (b.size - j * SpotFlock.L)

/-- The run of record's input on the play `o`'s case. -/
def inputOn (C : CaseOf) (o : Verity.Exchange) : Option Spot.Run.Input :=
  (C o).bind inputOfCase

/-- **The rounds whose coin the run's verifier reads**: every round before the verdict's but the statements'. -/
def readsAt (h : SpotFlock.Held) (n : ℕ) : Prop :=
  n ≠ h.stage + 1 ∧ n < SpotFlock.off h h.ms.length

/-- **The two-stage form's reading** at the session of record `A`, the instance's case of a play `C`, the sessions'
reading `S` and the coin map's reading of the device's releases in a log `dev`, which the coin map owes (module
docstring, **the reading**). -/
noncomputable def formReading (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) : Reading A where
  coin o n :=
    if n = 0 then (inputOn C o).bind fun i => coinOf i.source
    else if hn : n - 1 < A.h.stage then
      (inputOn C o).bind fun i => i.interior.bind fun it =>
        (stageCoinsOf A.h it.source).map fun cs => cs ⟨n - 1, hn⟩
    else if n = A.h.stage + 1 then none
    else match SpotFlock.slotOf A.h (n + 1) with
      | some t => S.coin o t (n - SpotFlock.off A.h t)
      | none => none
  msg o n :=
    if n = 0 then some []
    else if n = 1 then (inputOn C o).bind fun i => i.interior.bind fun it => recordMsg it.record
    else if n ≤ A.h.stage then some []
    else if n = A.h.stage + 1 then S.stmts o
    else match SpotFlock.slotOf A.h (n + 1) with
      | some t => S.msg o t (n - SpotFlock.off A.h t)
      | none => some []
  dev := dev
  reads n := readsAt A.h n
  stmt o := (inputOn C o).bind fun i => recordMsg i.record
  agree o n c c' :=
    if n = 0 then ∃ i, inputOn C o = some i ∧ prefixAgree i.source.size c c'
    else if n ≤ A.h.stage then
      ∃ i it, inputOn C o = some i ∧ i.interior = some it ∧ prefixAgree (stageWidth it.source (n - 1)) c c'
    else match SpotFlock.slotOf A.h (n + 1) with
      | some t => S.agree o t (n - SpotFlock.off A.h t) c c'
      | none => False

end Verity.SampledTwo
