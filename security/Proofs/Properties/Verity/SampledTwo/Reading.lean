module

public import Proofs.Properties.Verity.SampledTwo.Projection
public import VStarRecursive.SampledTwo

/-! What the two-stage form's check reads of a play (`formReading`), and its acceptance (`FormAccepts`). -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)

structure Case where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray
  blobs : Core.Blobs

def Decoded (k : Case) (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (e : VStar.Recursive.SampledTwo.Evidence) : Prop :=
  (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) = .ok p ∧
    Spot.OneStage.commitmentOf k.commitment = .ok c ∧ VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e

/-- The form's check accepts the case with its switch on, so nothing a piece proves of an accepted play
rests on the switch being off. -/
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

abbrev CaseOf : Type := Verity.Exchange → Option Case

def FormAccepts (C : CaseOf) (o : Verity.Exchange) : Prop :=
  ∃ k, C o = some k ∧ AcceptsOn k

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

structure SessionReading where
  stmts : Verity.Exchange → Option Msg
  msg : Verity.Exchange → ℕ → ℕ → Option Msg
  coin : Verity.Exchange → ℕ → ℕ → Option FlockDeploy.Coin
  agree : Verity.Exchange → ℕ → ℕ → FlockDeploy.Coin → FlockDeploy.Coin → Prop

def prefixAgree (w : ℕ) (c c' : FlockDeploy.Coin) : Prop :=
  (SpotFlock.src c).extract 0 w = (SpotFlock.src c').extract 0 w

def stageWidth (b : ByteArray) (j : ℕ) : ℕ := min SpotFlock.L (b.size - j * SpotFlock.L)

def inputOn (C : CaseOf) (o : Verity.Exchange) : Option Spot.Run.Input :=
  (C o).bind inputOfCase

def readsAt (h : SpotFlock.Held) (n : ℕ) : Prop :=
  n ≠ h.stage + 1 ∧ n < SpotFlock.off h h.ms.length

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
