module

public import Proofs.Properties.Verity.SampledTwo.Projection
public import Proofs.Properties.Verity.Sampled.Binding
public import VStarRecursive.SampledTwo

/-!
# The cut of the two-stage leg: the log the form's check read is the device's, or a break

`ProjectionOf` reads, beside the verdict, a `Good` on each play: the log the form's check read is the device's. This
module defines it for the two-stage form (`LogIsDevice`) and bounds the accepted plays where it fails (`CutBound`) by
the read finder's SHA-512 collision plus the forger's forgery of the challenger's signature.

* **The case** (`caseOf`): the sampled component's parameters, registration and evidence in what the prover sent, each
  read by the two-stage form's decoders (`VStar.Recursive.SampledTwo.paramsOfJson`, `evidenceOf`).
* **What the form accepts, the switch aside** (`Accepts`): the form's check's steps 1 to 5 (`prepare`) at the first
  stage's bytes the evidence carries, the interior audit's verdict accepting and the run of record's symbol `accept`.
  The form's last step refuses while `FORM` is off (`checkBlobsAt_refuses`), so a cut at the check itself would hold
  because the check accepts nothing; this is the check with the switch on.
* **The two sessions** (`records`): the play has two challenger sessions, the first stage's (the boundary
  registration's coins, `Evidence.coins`) and the interior's (the second stage's coins and the coin trees its sessions
  open, the interior audit's `coins`). Each is read under the parameters' challenger key (`keyOf`, the interior audit's,
  at the deployed LMS parameters), its window, its family and what its coins follow.
* **The log the check read** (`reads`, `LogIsDevice`): the first session's releases, then the interior's, each read
  as v2 reads a session's log (`Verity.Sampled.readLog`), the first with the coins its draw read
  (`Spot.Challenger.drawCoins`) and the interior with the second stage's (`OneStageZk.coinsOf`) and its sessions'
  records. `LogIsDevice o` is `Binding.Logged reads o`: the read is the play's log `o.log`, release by release, in
  order and no other.
* **The finders** (`readFinder`, `forger`): on a uniform tape of the play, the first collision `findBreak` finds in
  either session's record, else the first round opened whose coins the read without the samples' coins shows other
  than the play released (`Lines.openCollision`); and the first forgery `findBreak` finds in either session's record
  under the parameters' key. `terms` is the sum of their probabilities; both sessions count in each.
* **The premises** (`Binds`, `SignsPlay`): pointwise, an accepted play whose log the check read isn't the device's
  carries a break in one of the two records, or an opening collision (`Binds`); and the device's history at each play
  signed, under the parameters' key, what the play's signatures are on (`SignsPlay`).
* **The cut** (`cutBound`): the sampled verdict only where the form accepts (the wiring, at the parameters `θ`),
  `Binds` and `SignsPlay` give `CutBound` at `LogIsDevice` and `terms`.

**The earlier signature.** `findBreak` counts as a forgery every signature that verifies under the key over a message
the history `dev o` didn't sign (`Properties.Challenger.Signed`). A history that holds one play's signatures makes an
earlier period's genuine signature in the evidence a forgery here: the forger's term then reports it, `SignsPlay` and
`Binds` hold of such a history, and the forger's probability is what the deployed bound reads against
`Hash.Assumptions.LmsUnforgeable`, which a genuine signature the forger replays does not meet. So at a history of one
play the cut says nothing of a strategy that replays an earlier genuine signature. That case shows as the forger's
term, never as `LogIsDevice`.
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Binding (Read Logged collisionOf forgeryOf)
open scoped ENNReal

/-! ## The case -/

/-- The sampled component's parameters in the composition's parameters `θ`, read as the two-stage form's. -/
def paramsAt (θ : ByteArray) : Flock.V VStar.Recursive.SampledTwo.Params := do
  let ps ← _root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams
  let some x := ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the parameters name no sampled proofs"
  _root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.SampledTwo.paramsOfJson

/-- **The two-stage case in what the prover sent**: the sampled component's parameters (`paramsAt`), and the
registration and the evidence of the case the composition's evidence names under its id, each read by the two-stage
form's decoder. -/
def caseOf (s : Verity.Sent) : Flock.V (VStar.Recursive.SampledTwo.Params × Spot.OneStage.Commitment ×
    VStar.Recursive.SampledTwo.Evidence) := do
  let p ← paramsAt s.params
  let ev ← _root_.Core.Composition.decodeEvidence s.evidence
  let some y := ev.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) | throw "the evidence holds no sampled proofs case"
  let c ← Spot.OneStage.commitmentOf y.commitment
  let e ← VStar.Recursive.SampledTwo.evidenceOf y.evidence
  pure (p, c, e)

/-- **The form accepts, the switch aside**: the form's check's steps 1 to 5 (`prepare`) pass at the first stage's bytes
the evidence carries, the interior audit's verdict accepts, and the run of record's symbol is `accept`; what
`checkBlobs` accepts once `FORM` is on. -/
def Accepts (s : Verity.Sent) : Prop :=
  ∃ p c e two run pub prof, caseOf s = .ok (p, c, e) ∧
    VStar.Recursive.SampledTwo.prepare p c e.source e (_root_.Core.Blobs.ofArray s.blobs) = .ok (two, run) ∧
    two = .accept pub prof ∧ Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true

/-! ## The two sessions -/

/-- Whether the key is at the deployed LMS parameters (`Lms.deployed`). -/
def atDeployed (key : ChallengerKey) : Bool :=
  key.params.N == Lms.deployed.N && key.params.W == Lms.deployed.W && key.params.H == Lms.deployed.H

theorem atDeployed_eq {key : ChallengerKey} (h : atDeployed key = true) : key.params = Lms.deployed := by
  obtain ⟨⟨N, W, H⟩, pub⟩ := key
  simp only [atDeployed, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  simp only at h1 h2 h3
  subst h1 h2 h3
  rfl

/-- **The parameters' challenger key**: the interior audit's challenger's, where the parameters read as the two-stage
form's and it is at the deployed LMS parameters. -/
def keyOf (θ : ByteArray) : Option ChallengerKey :=
  ((paramsAt θ).toOption.bind fun p => p.audit.base.challenger.map (·.1)).filter atDeployed

theorem keyOf_deployed (θ : ByteArray) : ∀ key ∈ keyOf θ, key.params = Lms.deployed := by
  intro key hk
  exact atDeployed_eq (Option.filter_eq_some_iff.1 (Option.mem_def.1 hk)).2

/-- The first stage's coin evidence as `drawCoins` reads it: the form's `coins` without the key it carries. -/
def firstCoins (e : VStar.Recursive.SampledTwo.Evidence) : Json :=
  Json.mkObj ((Spot.entries e.coins).filter (·.1 != "key")).toList

/-- The family the first stage's coins are of, as its segment names it. -/
def firstLabel (e : VStar.Recursive.SampledTwo.Evidence) : Option String :=
  match Spot.getD (Spot.getD e.coins "segment") "label" with
  | .str l => some l
  | _ => none

/-- The boundary registration's parameters, as the form and the run of record read them. -/
def boundaryOf (p : VStar.Recursive.SampledTwo.Params) : Flock.V Spot.OneStage.Params := do
  Spot.OneStage.paramsParsed (← _root_.Core.canonicalJson (← Flock.canon p.boundary).toUTF8)

/-- **What the first session's coins are read under**: the parameters' key (the interior audit's challenger's), the
boundary registration's window, the family its segment names, after the boundary registration's digest. -/
def firstDraws (key : ChallengerKey) (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (e : VStar.Recursive.SampledTwo.Evidence) : Option Draws := do
  let bp ← (boundaryOf p).toOption
  let w ← bp.verifier.windowRoot
  let label ← firstLabel e
  pure ⟨key, w, label, c.digest⟩

/-- The interior registration's commitment, as the form's check reads it from the evidence's record. -/
def interiorOf (e : VStar.Recursive.SampledTwo.Evidence) : Flock.V Spot.OneStage.Commitment := do
  Spot.OneStage.commitmentOf (← Flock.canon e.record).toUTF8

/-- **What the interior session's coins are read under**: the interior audit's challenger's key and family, its
window, after the interior registration's digest (`oneStageZkSource`). -/
def secondDraws (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence) : Option Draws := do
  oneStageZkSource p.audit (← (interiorOf e).toOption)

/-- **The two sessions' records in what the prover sent**: the first stage's draw with its coin evidence, then the
interior's with the interior audit's, each where the case reads as the two-stage form's and the parameters name a key
at the deployed parameters; at most two. -/
def records (s : Verity.Sent) : List (Draws × Json) :=
  match caseOf s, keyOf s.params with
  | .ok (p, c, e), some key =>
    ((firstDraws key p c e).map fun d => (d, firstCoins e)).toList ++
      ((secondDraws p e).map fun d => (d, e.interior.base.coins)).toList
  | _, _ => []

/-! ## The log the check read -/

/-- The first session's release log, a line each, as v2's `linesOf` reads a session's. -/
def firstLines (e : VStar.Recursive.SampledTwo.Evidence) : Flock.V (List String) := do
  let .arr ls := Spot.getD (Spot.getD e.coins "session") "log" | throw "the session's log is not a list of lines"
  ls.toList.mapM fun | .str l => pure l | _ => throw "a line of the session's log is not a string"

/-- The records of the interior audit's sessions on the blobs `bs`, every statement's in order, as v2's per-member
form holds them to the coin trees (`Verity.Sampled.recordOn`). -/
def secondRecords (bs : _root_.Core.Blobs) (e : VStar.Recursive.SampledTwo.Evidence) : Option (List Json) :=
  (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn bs)

/-- **The first session's releases as the check read them**, with the coins `ξ` placed (`.empty` for none). -/
def firstRead (key : ChallengerKey) (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (e : VStar.Recursive.SampledTwo.Evidence) (ξ : ByteArray) : Option (List Release) := do
  let d ← firstDraws key p c e
  Verity.Sampled.readLog d.label d.after [] ξ (← (firstLines e).toOption)

/-- **The interior session's releases as the check read them**, with the coins `ξ` placed (`.empty` for none). -/
def secondRead (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence)
    (bs : _root_.Core.Blobs) (ξ : ByteArray) : Option (List Release) := do
  let d ← secondDraws p e
  Verity.Sampled.readLog d.label d.after (← secondRecords bs e) ξ (← (Verity.Sampled.linesOf e.interior).toOption)

/-- **The device's log as the two-stage form's check read it**, from what the prover sent: the first session's
releases with the coins its draw read under the parameters' key (`drawCoins`), then the interior's with the second
stage's coins (`OneStageZk.coinsOf`); none where either doesn't read. -/
def readsLog (s : Verity.Sent) : Option (List Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let key ← keyOf s.params
  let d ← firstDraws key p c e
  let ξ ← (Spot.Challenger.drawCoins d.key d.window d.label d.after (firstCoins e)).toOption
  let c₂ ← (interiorOf e).toOption
  let ξ₂ ← (Spot.OneStageZk.coinsOf p.audit c₂ e.interior).toOption
  let bs := _root_.Core.Blobs.ofArray s.blobs
  pure ((← firstRead key p c e ξ) ++ (← secondRead p e bs ξ₂))

/-- The same read without the samples' coins, where each opened round's coins show as the records give them. -/
def opensRead (s : Verity.Sent) : Option (List Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let key ← keyOf s.params
  let bs := _root_.Core.Blobs.ofArray s.blobs
  pure ((← firstRead key p c e .empty) ++ (← secondRead p e bs .empty))

/-- **The log read as `Binding.Read`s**: each release with its coins where the read shows any (`Read.ofV2`). -/
def reads (s : Verity.Sent) : Option (List Read) := (readsLog s).map (·.map Verity.Sampled.Binding.Read.ofV2)

/-- **Every line of the challenger's log the evidence carries is one the device released, in order**: for both
stages' sessions, the releases the form's check read (`reads`) are the play's log, each of the device's releases in
order and no other (`Binding.Logged`). -/
def LogIsDevice (o : Exchange) : Prop := Logged reads o

/-! ## The read finder and the forger -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **`findBreak`'s SHA-512 calls on the two sessions' records** at the device's history `dev`, as
`Binding.breakCalls` counts them on one. -/
def breakCalls (dev : History) (s : Verity.Sent) : ℕ :=
  ((records s).map fun x => (signatures x.1.key x.2).length * Verity.Sampled.Binding.lmsCalls x.1.key.params +
    (readHashed x.1.key x.1.label x.2).length + (deviceHashed dev x.1.key.pub).length).sum

/-- **The read finder at a strategy**, at the device's history at each play `dev`: on a uniform tape of the play, the
first collision `findBreak` finds in one of the two sessions' records, else the first round opened whose coins the read
without the samples' coins shows other than the play released (`Lines.openCollision`). -/
noncomputable def readFinder (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Claims.Finder where
  S := Verity.Game.Tape (V.game θ F c) σ
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    (((records o.sent).filterMap fun x => findBreak (dev o) x.1.key x.1.label x.2).findSome? collisionOf).or
      (((opensRead o.sent).bind fun l => Verity.Sampled.Lines.openCollision l o.log).map fun uv =>
        (uv.1.data.toList, uv.2.data.toList))
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent + o.sent.blobs.size

theorem readFinder_out (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ) :
    (readFinder V dev θ F c σ).out t =
      ((((records (Verity.Game.Tape.out _ σ t).sent).filterMap fun x =>
          findBreak (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2).findSome? collisionOf).or
        (((opensRead (Verity.Game.Tape.out _ σ t).sent).bind fun l =>
          Verity.Sampled.Lines.openCollision l (Verity.Game.Tape.out _ σ t).log).map fun uv =>
            (uv.1.data.toList, uv.2.data.toList))) :=
  rfl

/-- **The forger at a strategy**, against the parameters' key (`keyOf`), at the device's history at each play `dev`:
on a uniform tape of the play, the first forgery `findBreak` finds in one of the two sessions' records under that key.
-/
noncomputable def forger (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Hash.Assumptions.LmsForger where
  S := Verity.Game.Tape (V.game θ F c) σ
  key := ((keyOf θ).map fun k => k.pub.data.toList).getD []
  signed t := (Verity.Game.Tape.out _ σ t).signed.map Prod.fst
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    ((records o.sent).filterMap fun x =>
      if keyOf θ = some x.1.key then findBreak (dev o) x.1.key x.1.label x.2 else none).findSome? forgeryOf
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent

/-- **The cut's terms**: the read finder's collision plus the forger's forgery, each over both sessions. -/
noncomputable def terms (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  ENNReal.ofReal ((readFinder V dev θ F c σ).Pr Verity.sha512Msg) +
    ENNReal.ofReal ((forger V dev θ F c σ).Pr Verity.lmsVerify)

/-! ## The premises -/

/-- **The sampled verdict accepts only where the form does**, at the parameters `θ` (the composite's wiring). -/
def Wired (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) : Prop :=
  ∀ o : Exchange, o.params = θ → V.Verdict k .sampled o → Accepts o.sent

/-- **The device's history at each play signed what the play's signatures are on**, under the parameters' key
(`Binding.SignsPlay` at `keyOf`). At a history holding one play's signatures, an earlier genuine signature the evidence
carries is none the history signed, so it is the forger's (module docstring). -/
def SignsPlay (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    Prop :=
  ∀ o, Verity.Game.Reaches (V.game θ F c) o → ∀ key, keyOf θ = some key →
    ∀ m ∈ o.signed, Signed (dev o) key.pub ⟨m.1.toArray⟩

/-- **The binding at the parameters `θ`, pointwise**: on every play after `θ` that the form accepts and whose log, as
its check read it, isn't the device's, one of the two sessions' records under the parameters' key carries a break at
the device's history at the play, or the read without the samples' coins shows a round opened's coins other than the
device's, of one SHA-512 with them. -/
def Binds (dev : Exchange → History) (θ : ByteArray) : Prop :=
  ∀ o : Exchange, o.params = θ → Accepts o.sent → ¬ LogIsDevice o →
    (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
      ∃ u v, ((opensRead o.sent).bind fun l => Verity.Sampled.Lines.openCollision l o.log) = some (u, v) ∧ u ≠ v ∧
        Flock.Sha512.hash u = Flock.Sha512.hash v

/-! ## The cut -/

/-- **A play with a break in one of the two records under the parameters' key is one where the read finder or the
forger succeeds**, where the device's history at each play signed what the play's signatures are on. -/
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
    let l := (records o.sent).filterMap fun x' =>
      if keyOf θ = some x'.1.key then findBreak (dev o) x'.1.key x'.1.label x'.2 else none
    have hfx : (if keyOf θ = some x.1.key then findBreak (dev o) x.1.key x.1.label x.2 else none) =
        some (Finding.forgery m sig) := by
      rw [if_pos hk]
      exact hw
    have hmem : Finding.forgery m sig ∈ l := List.mem_filterMap.2 ⟨x, hx, hfx⟩
    have hl : ∀ m' sig', Finding.forgery m' sig' ∈ l →
        LmsVerifies x.1.key m' sig' ∧ ¬ Signed (dev o) x.1.key.pub m' := by
      intro m' sig' h
      obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 h
      have hx'' : (if keyOf θ = some x'.1.key then findBreak (dev o) x'.1.key x'.1.label x'.2 else none) =
          some (Finding.forgery m' sig') := hx'
      by_cases hk' : keyOf θ = some x'.1.key
      · rw [if_pos hk'] at hx''
        have e : x'.1.key = x.1.key := Option.some.inj (hk'.symm.trans hk)
        have hv : LmsVerifies x'.1.key m' sig' ∧ ¬ Signed (dev o) x'.1.key.pub m' := findBreak_valid hx''
        exact e ▸ hv
      · rw [if_neg hk'] at hx''
        cases hx''
    obtain ⟨_, v, hf, hv, m', sig', hm', rfl⟩ :=
      Verity.Sampled.Binding.findSome_forgeryOf hd (fun m' sig' h => (hl m' sig' h).1) hmem
    refine ⟨_, v, hf, fun hu => ?_, ?_⟩
    · have hu' : m'.data.toList ∈ o.signed.map Prod.fst := hu
      obtain ⟨p, hp, hpu⟩ := List.mem_map.1 hu'
      have hsg := hs o (Verity.Game.Tape.reaches_out _ σ t) x.1.key hk p hp
      rw [hpu, Verity.Sampled.Binding.bytes_ofList] at hsg
      exact (hl m' sig' hm').2 hsg
    · show Verity.lmsVerify (((keyOf θ).map fun k => k.pub.data.toList).getD []) m'.data.toList v = true
      rw [hk]
      exact hv

/-- **A play whose read without the samples' coins shows a round opened's coins other than the device's, of one
SHA-512 with them, is one where the read finder succeeds.** -/
theorem finder_of_open (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ)
    {u v : ByteArray} (h : ((opensRead (Verity.Game.Tape.out _ σ t).sent).bind fun l =>
      Verity.Sampled.Lines.openCollision l (Verity.Game.Tape.out _ σ t).log) = some (u, v))
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
    refine ⟨u.data.toList, v.data.toList, ?_, Verity.Sampled.Binding.collision_of_valid huv⟩
    simp only [Option.none_or, h, Option.map_some]

/-- **The cut**: where the sampled verdict accepts only where the form does (`Wired`), every accepted play the form
read a log of that isn't the device's carries a break or an opening collision (`Binds`), and the device's history at
each play signed what the play's signatures are on (`SignsPlay`), an accepted play whose log the check read isn't the
device's is at most as likely as the read finder's collision plus the forger's forgery (`terms`), over both sessions.
-/
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
        rcases hb _ hθ (hw _ hθ hv) hl with ⟨x, hx, hk, hbr⟩ | ⟨u, v, huv, hne, hh⟩
        · exact finders_of_break V dev θ F c σ t hs hx hk hbr
        · exact .inl (finder_of_open V dev θ F c σ t huv ⟨hne, hh⟩)
    _ ≤ _ := Verity.Game.prCoin_or_le _ _
    _ = _ := congrArg₂ (· + ·) (Verity.Game.prCoin_eq_ofReal_pr _) (Verity.Game.prCoin_eq_ofReal_pr _)

end Verity.SampledTwo
