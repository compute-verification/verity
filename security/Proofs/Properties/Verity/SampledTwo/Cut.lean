module

public import Proofs.Properties.Verity.SampledTwo.Projection
public import Proofs.Properties.Verity.Sampled.Binding
public import VStarRecursive.SampledTwo

/-! The cut of the two-stage subprotocol: the log the form's check read is the device's, or a break; at a
one-play history it says nothing of a strategy that replays an earlier genuine signature, which shows as the
forger's term. -/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Binding (Read collisionOf forgeryOf)
open scoped ENNReal

local instance lawfulBEqByteArrayCut : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

def paramsAt (θ : ByteArray) : Flock.V VStar.Recursive.SampledTwo.Params := do
  let ps ← _root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams
  let some x := ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the parameters name no sampled proofs"
  _root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.SampledTwo.paramsOfJson

def caseIn (s : Verity.Sent) : Flock.V (Spot.OneStage.Commitment × VStar.Recursive.SampledTwo.Evidence) := do
  let ev ← _root_.Core.Composition.decodeEvidence s.evidence
  let some y := ev.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the evidence holds no sampled proofs case"
  let c ← Spot.OneStage.commitmentOf y.commitment
  let e ← VStar.Recursive.SampledTwo.evidenceOf y.evidence
  pure (c, e)

def caseOf (s : Verity.Sent) : Flock.V (VStar.Recursive.SampledTwo.Params × Spot.OneStage.Commitment ×
    VStar.Recursive.SampledTwo.Evidence) := do
  let p ← paramsAt s.params
  let ce ← caseIn s
  pure (p, ce.1, ce.2)

/-- The form's check with its switch on accepts the case; the check refuses while `FORM` is off, so a cut at
the check itself would hold because it accepts nothing. -/
def Accepts (s : Verity.Sent) : Prop :=
  ∃ p c e pub prof, caseOf s = .ok (p, c, e) ∧
    VStar.Recursive.SampledTwo.checkBlobsAtWith true p c e.source e (_root_.Core.Blobs.ofArray s.blobs) =
      .accept pub prof

def atDeployed (key : ChallengerKey) : Bool :=
  key.params.N == Lms.deployed.N && key.params.W == Lms.deployed.W && key.params.H == Lms.deployed.H

theorem atDeployed_eq {key : ChallengerKey} (h : atDeployed key = true) : key.params = Lms.deployed := by
  obtain ⟨⟨N, W, H⟩, pub⟩ := key
  simp only [atDeployed, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  subst h1 h2 h3
  rfl

def keyOf (θ : ByteArray) : Option ChallengerKey :=
  ((paramsAt θ).toOption.bind fun p => p.audit.base.challenger.map (·.1)).filter atDeployed

theorem keyOf_deployed (θ : ByteArray) : ∀ key ∈ keyOf θ, key.params = Lms.deployed := by
  intro key hk
  exact atDeployed_eq (Option.filter_eq_some_iff.1 (Option.mem_def.1 hk)).2

def firstCoins (e : VStar.Recursive.SampledTwo.Evidence) : Json :=
  e.coins

def firstLabel (p : VStar.Recursive.SampledTwo.Params) : Option String :=
  p.boundary.challenger.map (·.2)

def boundaryOf (p : VStar.Recursive.SampledTwo.Params) : Flock.V Spot.OneStage.Params :=
  pure p.boundary

def firstSamples (key : ChallengerKey) (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (e : VStar.Recursive.SampledTwo.Evidence) : Option Draws := do
  let bp ← (boundaryOf p).toOption
  let w ← bp.verifier.windowRoot
  let label ← firstLabel p
  pure ⟨key, w, label, c.digest⟩

def interiorOf (e : VStar.Recursive.SampledTwo.Evidence) : Flock.V Spot.OneStage.Commitment := do
  Spot.OneStage.commitmentOf (← Flock.canon e.record).toUTF8

def secondSamples (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence) : Option Draws := do
  oneStageZkSource p.audit (← (interiorOf e).toOption)

def records (s : Verity.Sent) : List (Draws × Json) :=
  match caseOf s, keyOf s.params with
  | .ok (p, c, e), some key =>
    ((firstSamples key p c e).map fun d => (d, firstCoins e)).toList ++
      ((secondSamples p e).map fun d => (d, e.interior.base.coins)).toList
  | _, _ => []

def firstLines (e : VStar.Recursive.SampledTwo.Evidence) : Flock.V (List String) := do
  let .arr ls := Spot.getD (Spot.getD (firstCoins e) "session") "log" | throw "the session's log is not a list of lines"
  ls.toList.mapM fun | .str l => pure l | _ => throw "a line of the session's log is not a string"

def secondRecords (bs : _root_.Core.Blobs) (e : VStar.Recursive.SampledTwo.Evidence) : Option (List Json) :=
  (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn bs)

def firstRead (key : ChallengerKey) (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (e : VStar.Recursive.SampledTwo.Evidence) (ξ : ByteArray) : Option (List Release) := do
  let d ← firstSamples key p c e
  Verity.Sampled.readLog d.label d.after [] ξ (← (firstLines e).toOption)

def secondRead (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence)
    (bs : _root_.Core.Blobs) (ξ : ByteArray) : Option (List Release) := do
  let d ← secondSamples p e
  Verity.Sampled.readLog d.label d.after (← secondRecords bs e) ξ (← (Verity.Sampled.linesOf e.interior).toOption)

/-- **The logs the session reads make of the play's log**: where both sessions are on one window, they are one session
(a ledger opens a window's root once), so each read is the whole log; on two windows, the first session's releases
then the interior's are. -/
def logsOf (w₁ w₂ : String) (r₁ r₂ : List Release) : List (List Release) :=
  if w₁ = w₂ then [r₁, r₂] else [r₁ ++ r₂]

def readsLogs (s : Verity.Sent) : List (List Release) :=
  (do
    let (p, c, e) ← (caseOf s).toOption
    let key ← keyOf s.params
    let d₁ ← firstSamples key p c e
    let ξ ← (Spot.Challenger.sampleCoins d₁.key d₁.window d₁.label d₁.after (firstCoins e)).toOption
    let c₂ ← (interiorOf e).toOption
    let d₂ ← secondSamples p e
    let ξ₂ ← (Spot.OneStageZk.coinsOf p.audit c₂ e.interior).toOption
    let bs := _root_.Core.Blobs.ofArray s.blobs
    pure (logsOf d₁.window d₂.window (← firstRead key p c e ξ) (← secondRead p e bs ξ₂)) :
    Option (List (List Release))).getD []

def opensReads (s : Verity.Sent) : List (List Release) :=
  (do
    let (p, c, e) ← (caseOf s).toOption
    let key ← keyOf s.params
    let d₁ ← firstSamples key p c e
    let d₂ ← secondSamples p e
    let bs := _root_.Core.Blobs.ofArray s.blobs
    pure (logsOf d₁.window d₂.window (← firstRead key p c e .empty) (← secondRead p e bs .empty)) :
    Option (List (List Release))).getD []

def LogIsDevice (o : Exchange) : Prop :=
  ∀ l ∈ readsLogs o.sent, List.Forall₂ Read.Of (l.map Verity.Sampled.Binding.Read.ofV2) o.log

def openBreak (log : List Release) (l : List Release) : Option (ByteArray × ByteArray) :=
  (Verity.Sampled.Lines.openCollision l log).filter fun uv =>
    !(uv.1 == uv.2) && Flock.Sha512.hash uv.1 == Flock.Sha512.hash uv.2

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

def breakCalls (dev : History) (s : Verity.Sent) : ℕ :=
  ((records s).map fun x => (signatures x.1.key x.2).length * Verity.Sampled.Binding.lmsCalls x.1.key.params +
    (readHashed x.1.key x.1.label x.2).length + (deviceHashed dev x.1.key.pub).length).sum

noncomputable def readFinder (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Claims.Finder where
  S := Verity.Game.Tape (V.game θ F c) σ
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    (((records o.sent).filterMap fun x => findBreak (dev o) x.1.key x.1.label x.2).findSome? collisionOf).or
      (((opensReads o.sent).findSome? (openBreak o.log)).map fun uv => (uv.1.data.toList, uv.2.data.toList))
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent + o.sent.blobs.size

theorem readFinder_out (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ) :
    (readFinder V dev θ F c σ).out t =
      ((((records (Verity.Game.Tape.out _ σ t).sent).filterMap fun x =>
          findBreak (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2).findSome? collisionOf).or
        (((opensReads (Verity.Game.Tape.out _ σ t).sent).findSome? (openBreak (Verity.Game.Tape.out _ σ t).log)).map
          fun uv => (uv.1.data.toList, uv.2.data.toList))) :=
  rfl

open Classical in
noncomputable def forgeryFinds (dev : History) (θ : ByteArray) (s : Verity.Sent) : List Finding :=
  (records s).filterMap fun x => if keyOf θ = some x.1.key then findBreak dev x.1.key x.1.label x.2 else none

theorem mem_forgeryFinds {dev : History} {θ : ByteArray} {s : Verity.Sent} {f : Finding} :
    f ∈ forgeryFinds dev θ s ↔
      ∃ x ∈ records s, keyOf θ = some x.1.key ∧ findBreak dev x.1.key x.1.label x.2 = some f := by
  unfold forgeryFinds
  rw [List.mem_filterMap]
  constructor
  · rintro ⟨x, hx, h⟩
    by_cases hk : keyOf θ = some x.1.key
    · rw [if_pos hk] at h
      exact ⟨x, hx, hk, h⟩
    · rw [if_neg hk] at h
      cases h
  · rintro ⟨x, hx, hk, h⟩
    refine ⟨x, hx, ?_⟩
    rw [if_pos hk]
    exact h

noncomputable def forger (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Hash.Assumptions.LmsForger where
  S := Verity.Game.Tape (V.game θ F c) σ
  key := ((keyOf θ).map fun k => k.pub.data.toList).getD []
  signed t := (Verity.Game.Tape.out _ σ t).signed.map Prod.fst
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    (forgeryFinds (dev o) θ o.sent).findSome? forgeryOf
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent

noncomputable def terms (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  ENNReal.ofReal ((readFinder V dev θ F c σ).Pr Verity.sha512Msg) +
    ENNReal.ofReal ((forger V dev θ F c σ).Pr Verity.lmsVerify)

def Wired (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) : Prop :=
  ∀ o : Exchange, o.params = θ → V.Verdict k .sampled o → Accepts o.sent

/-- **The device's history at each play signed what the play's signatures are on**, under the parameters' key
(`Binding.SignsPlay` at `keyOf`). At a history holding one play's signatures, an earlier genuine signature the evidence
carries is none the history signed, so it is the forger's (module docstring). -/
def SignsPlay (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    Prop :=
  ∀ o, Verity.Game.Reaches (V.game θ F c) o → ∀ key, keyOf θ = some key →
    ∀ m ∈ o.signed, Signed (dev o) key.pub ⟨m.1.toArray⟩

def Binds (dev : Exchange → History) (θ : ByteArray) : Prop :=
  ∀ o : Exchange, o.params = θ → Accepts o.sent → ¬ LogIsDevice o →
    (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
      ∃ l ∈ opensReads o.sent, ∃ u v, Verity.Sampled.Lines.openCollision l o.log = some (u, v) ∧ u ≠ v ∧
        Flock.Sha512.hash u = Flock.Sha512.hash v

def PrintsOn (dev : History) (w : String) (log : List Release) : Prop :=
  ∀ pk r, r ∈ dev.runs pk → OpensOn dev pk w r.id →
    Verity.Sampled.Lines.Prints [] r.log.tail log ∧ (∃ ts, coinTreesOf r.log = .ok ts) ∧
      ((∃ c roots, r.commitment = some c ∧ releasedRootsOf c r.log = .ok roots) →
        ∀ label after, Verity.Sampled.Lines.Holds (r.released label after)
          (Verity.Sampled.Lines.freshOf label after log))

def PrintsTwo (dev : History) (w₁ w₂ : String) (log : List Release) : Prop :=
  if w₁ = w₂ then PrintsOn dev w₁ log
  else ∃ log₁ log₂, log = log₁ ++ log₂ ∧ PrintsOn dev w₁ log₁ ∧ PrintsOn dev w₂ log₂

def executionPhasesOf (θ : ByteArray) : Option (String × String) := do
  let p ← (paramsAt θ).toOption
  let bp ← (boundaryOf p).toOption
  pure (← bp.verifier.windowRoot, ← p.audit.base.verifier.windowRoot)

def BindsOfPrints (dev : Exchange → History) (θ : ByteArray) : Prop :=
  (∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → PrintsTwo (dev o) w₁ w₂ o.log) →
    Binds dev θ

theorem finders_of_break (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ) (hs : SignsPlay V dev θ F c) {x : Draws × Json}
    (hx : x ∈ records (Verity.Game.Tape.out _ σ t).sent) (hk : keyOf θ = some x.1.key)
    (hb : Break (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2) :
    (∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
      (forger V dev θ F c σ).Forges Verity.lmsVerify t := by
  classical
  obtain ⟨w, hw, -⟩ := hb
  cases w with
  | collision a b =>
    left
    refine Verity.Sampled.Binding.findSome_collisionOf_or (fun a' b' h => ?_) (List.mem_filterMap.2 ⟨x, hx, hw⟩)
    obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 h
    exact findBreak_valid hx'
  | forgery m sig =>
    right
    have hd : x.1.key.params = Lms.deployed := keyOf_deployed θ _ hk
    let o := Verity.Game.Tape.out _ σ t
    have hmem : Finding.forgery m sig ∈ forgeryFinds (dev o) θ o.sent := mem_forgeryFinds.2 ⟨x, hx, hk, hw⟩
    have hl : ∀ m' sig', Finding.forgery m' sig' ∈ forgeryFinds (dev o) θ o.sent →
        LmsVerifies x.1.key m' sig' ∧ ¬ Signed (dev o) x.1.key.pub m' := by
      intro m' sig' h
      obtain ⟨x', -, hk', hx'⟩ := mem_forgeryFinds.1 h
      have e : x'.1.key = x.1.key := Option.some.inj (hk'.symm.trans hk)
      have hv : LmsVerifies x'.1.key m' sig' ∧ ¬ Signed (dev o) x'.1.key.pub m' := findBreak_valid hx'
      exact e ▸ hv
    obtain ⟨_, v, hf, hv, m', sig', hm', rfl⟩ :=
      Verity.Sampled.Binding.findSome_forgeryOf hd (fun m' sig' h => (hl m' sig' h).1) hmem
    have hout : (forger V dev θ F c σ).out t = (forgeryFinds (dev o) θ o.sent).findSome? forgeryOf := rfl
    refine ⟨_, v, hout.trans hf, fun hu => ?_, ?_⟩
    · have hu' : m'.data.toList ∈ o.signed.map Prod.fst := hu
      obtain ⟨p, hp, hpu⟩ := List.mem_map.1 hu'
      have hsg := hs o (Verity.Game.Tape.reaches_out _ σ t) x.1.key hk p hp
      rw [hpu, Verity.Sampled.Binding.bytes_ofList] at hsg
      exact (hl m' sig' hm').2 hsg
    · show Verity.lmsVerify (((keyOf θ).map fun k => k.pub.data.toList).getD []) m'.data.toList v = true
      rw [hk]
      exact hv

theorem finder_of_open (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ)
    {l : List Release} (hl : l ∈ opensReads (Verity.Game.Tape.out _ σ t).sent) {u v : ByteArray}
    (h : Verity.Sampled.Lines.openCollision l (Verity.Game.Tape.out _ σ t).log = some (u, v))
    (huv : u ≠ v ∧ Flock.Sha512.hash u = Flock.Sha512.hash v) :
    ∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  rw [readFinder_out]
  cases hf : (((records (Verity.Game.Tape.out _ σ t).sent).filterMap fun x =>
      findBreak (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2).findSome? collisionOf) with
  | some uv =>
    obtain ⟨u', v'⟩ := uv
    refine ⟨u', v', rfl, Verity.Sampled.Binding.findSome_collisionOf_valid (fun a b hm => ?_) hf⟩
    obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 hm
    exact findBreak_valid hx'
  | none =>
    have hb : openBreak (Verity.Game.Tape.out _ σ t).log l = some (u, v) := by
      unfold openBreak
      rw [h]
      simp [huv.1, huv.2]
    obtain ⟨⟨u', v'⟩, hf', hv'⟩ : ∃ uv, (opensReads (Verity.Game.Tape.out _ σ t).sent).findSome?
        (openBreak (Verity.Game.Tape.out _ σ t).log) = some uv ∧
          uv.1 ≠ uv.2 ∧ Flock.Sha512.hash uv.1 = Flock.Sha512.hash uv.2 := by
      cases hg : (opensReads (Verity.Game.Tape.out _ σ t).sent).findSome? (openBreak (Verity.Game.Tape.out _ σ t).log)
        with
      | none => exact absurd (List.findSome?_eq_none_iff.1 hg l hl) (by rw [hb]; simp)
      | some uv =>
        refine ⟨uv, rfl, ?_⟩
        obtain ⟨l', -, hl'⟩ := List.exists_of_findSome?_eq_some hg
        unfold openBreak at hl'
        obtain ⟨-, hp⟩ := Option.filter_eq_some_iff.1 hl'
        simpa using hp
    refine ⟨u'.data.toList, v'.data.toList, ?_, Verity.Sampled.Binding.collision_of_valid hv'⟩
    simp only [Option.none_or, hf', Option.map_some]

theorem cutBound (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ)
    (hb : Binds dev θ) (hs : SignsPlay V dev θ F c) :
    CutBound V k θ F c σ LogIsDevice (terms V dev θ F c σ) := by
  unfold CutBound terms
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) ∧
          ¬ LogIsDevice (Verity.Game.Tape.out _ σ t))
      ≤ Verity.Game.prCoin (fun t =>
          (∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
            (forger V dev θ F c σ).Forges Verity.lmsVerify t) := by
        refine Verity.Game.prCoin_mono fun t ⟨hv, hl⟩ => ?_
        have hθ := (Verity.Sampled.Binding.play_reaches (Verity.Game.Tape.reaches_out _ σ t)).1
        rcases hb _ hθ (hw _ hθ hv) hl with ⟨x, hx, hk, hbr⟩ | ⟨l, hl', u, v, huv, hne, hh⟩
        · exact finders_of_break V dev θ F c σ t hs hx hk hbr
        · exact .inl (finder_of_open V dev θ F c σ t hl' huv ⟨hne, hh⟩)
    _ ≤ _ := Verity.Game.prCoin_or_le _ _
    _ = _ := congrArg₂ (· + ·) (Verity.Game.prCoin_eq_ofReal_pr _) (Verity.Game.prCoin_eq_ofReal_pr _)

end Verity.SampledTwo
