import Proofs.Properties.Verity.SampledRecs
import Proofs.Properties.Verity.SampledCanonDraw

/-!
# The reduction at a live session, from what the play's releases read back to

`Canon.drive` builds `Drive` at a live session `A` of `SampledDeploy`: its transcript of the play is session 0's
(`Canon.proj`), its last message the roots that record decodes to and its proof files (`Canon.pfOf`), and it reads the
device's releases with `Readers`. `Drive.canon` then follows from what the releases read back to:

* `RunsAt`: off the break, where sampled proofs' verdict accepts, the play's tagged releases are the draw's block, the
  link points', then one block per round, each one release per chunk of its coin (`Blocks`);
* `DrawsAt`: at those blocks, the record's draw is the unit draw the draw's block reads back to (`Canon.DrawAt`);
* `LinksAt`: at those blocks, the record's link points and every rep stream's rounds are what their blocks read back to
  (`Canon.PointsAt`, `Canon.RoundsAt`), and the link points are released after the hash the points' block carries
  (`Canon.PointsAfter`).

`Canon.recordOfZKJ_proj` and `Canon.pointsHash_pfOf` turn these into `Drive.canon`'s record and hash. `covers` needs
one more fact, `InputsAt`: where v2 accepts off the break, the session's inputs are `A`'s (`acceptsZK_proj`).
`accept_le_of_canon` is `SampledDeploy.accept_le` at `τOf := (drive …).τ` and `col := colAt recsAt pairs`: the
pieces' breaks, on records of an accepting case's uses, are its collisions or the forger's success (`brk_of_break`).
-/

open Lean (Json)

namespace Verity.Canon

open Accounting Core
open FlockSoundness.Discharge
open Properties.SampledProofs.OneStageZk (SessionInputs)

local notation "𝔽" => FlockLevel3.GF128

variable {ι α : Type} [Fintype α] {d : Device ι}

/-! ## The session's inputs -/

/-- **The play's session is at the inputs `I`**: off the break, wherever sampled proofs' verdict accepts, the inputs
of v2's sessions are `I` (`SessionInputs`). -/
def InputsAt (V : Vocab d α) (Brk : Outcome → Prop) (I : Exec.Inputs) : Prop :=
  ∀ o, V.Verdict .sampled o → ¬ Brk o → ∀ p c' e pub prof, SampledCase o.sent p c' e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c' e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
    SessionInputs p e o.sent.blobs I

/-- **`InputsAt` from the statement's inputs**: inputs that are the statement's wherever v2 accepts off the break
(`StatementInputs`, which reads the registration's public file and no evidence) are the session's, since v2 refuses an
evidence's public file the registration doesn't name (`sessionInputs_of_accept`). -/
theorem inputsAt_of_statement {V : Vocab d α} {Brk : Outcome → Prop} {I : Exec.Inputs}
    (h : ∀ o, V.Verdict .sampled o → ¬ Brk o → ∀ p c' e pub prof, SampledCase o.sent p c' e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c' e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
      Properties.SampledProofs.OneStageZk.StatementInputs p c' o.sent.blobs I) :
    InputsAt V Brk I := fun o hv hb p c' e pub prof hc hacc =>
  Properties.SampledProofs.OneStageZk.sessionInputs_of_accept hacc (h o hv hb p c' e pub prof hc hacc)

/-- **Where sampled proofs' verdict accepts, session 0 has a transcript.** -/
theorem sess0_of_verdict {V : Vocab d α} {o : Outcome} (hv : V.Verdict .sampled o) : ∃ t, Sess0 o.sent t := by
  obtain ⟨p, c', e, pub, prof, hcase, hacc⟩ := componentAccepts_sampled hv.2.2
  obtain ⟨s0, hs0⟩ := Properties.SampledProofs.OneStageZk.session_of_accept hacc
  rcases hl : e.sessions.toList with _ | ⟨s, rest⟩
  · have h := Array.mem_toList_iff.2 hs0
    rw [hl] at h
    cases h
  · exact sess0_of_accept hcase hacc hl

/-- **`Drive.covers` at `proj`**: off the break, where sampled proofs' verdict accepts, flock-verify `--zk` accepts
session 0's transcript at the session's inputs `I`. -/
theorem acceptsZK_proj {V : Vocab d α} {Brk : Outcome → Prop} {I : Exec.Inputs} (hIn : InputsAt V Brk I)
    {o : Outcome} (hv : V.Verdict .sampled o) (hb : ¬ Brk o) : ZkExec.AcceptsZK I (proj o) :=
  acceptsZK_of_sess0 (transcriptOf_spec (sess0_of_verdict hv)) (hIn o hv hb)

/-! ## How the reduction reads the releases -/

section

variable {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}

/-- **How the reduction reads the play's releases** at the live session `A` (`Drive`'s readers): the registration it
sends the live audit, a release's bits, their tags, the chunks of the draw, the link points and a round's coins, the
hash the link points' request carries, and the bytes a round's request carries. -/
structure Readers (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) where
  R₀ : A.Reg
  m : ℕ
  tagOf : Game.LiveDrive.Req ByteArray → Option Game.LiveDrive.Tag
  spΩ : Game.Split ((i : Fin 1) → Fin (A.Lb i) → Fin 256) m
  spP : Game.Split (Fin 2 → Fin (Exec.mPtsOf D.unroll.g (kd / Composed.nTab A.I)) → 𝔽) m
  spR : (k : ℕ) → Game.Split (Fin k → 𝔽) m
  hOf : Game.LiveDrive.Req ByteArray → ByteArray
  bytesOf : Game.LiveDrive.Req ByteArray → ByteArray

variable {A : Flock.Guarantees.LiveAt n D hU kd K₀ desc}

/-- The draw the draw's block reads back to. -/
noncomputable def Readers.drawOf (Rd : Readers A) (kd' : Game.LiveDrive.Block (Game.LiveDrive.Req ByteArray) Rd.m) :
    Finset (Fin n) :=
  Game.LiveDrive.drawOS (.subset kd) n (Rd.spΩ.e (Game.LiveDrive.decChunks Rd.m _ kd'.bits))

/-- **The play's tagged releases split into blocks** (`Drive.canon`'s first five conjuncts): the draw's block, then the
link points', then one block per round, each one release per chunk of its coin, at most `A.N` rounds, each of a table
of the draw. -/
def Blocks (Rd : Readers A) (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool)))
    (kd' kp : Game.LiveDrive.Block (Game.LiveDrive.Req ByteArray) Rd.m)
    (rds : List (Game.LiveDrive.RoundBlock (Game.LiveDrive.Req ByteArray) Rd.m)) : Prop :=
  (Game.LiveDrive.annot [] hs).filter (Game.LiveDrive.Tagged Rd.m Rd.tagOf) =
      (kd'.rels ++ kp.rels ++ rds.flatMap fun r => r.k.rels).map (Game.LiveDrive.rel Rd.m) ∧
    kd'.Fits Rd.m Rd.tagOf .draw Rd.spΩ ∧ kp.Fits Rd.m Rd.tagOf .points Rd.spP ∧ rds.length ≤ A.N ∧
    (∀ r ∈ rds, r.Fits Game.LiveDrive.nOfJ Game.LiveDrive.mkJ
      (fun x => (ZkHidden.planZHJ (Reg := A.Reg) A.dj A.y₀ (Game.LiveDrive.drawOS (.subset kd) n x) Rd.R₀).length)
      Rd.m Rd.tagOf Rd.spR Rd.bytesOf (Rd.spΩ.e (Game.LiveDrive.decChunks Rd.m _ kd'.bits)))

/-- The rounds the round blocks read back to, at the draw the draw's block reads back to. -/
noncomputable def Readers.roundsAt (Rd : Readers A) (kd' : Game.LiveDrive.Block (Game.LiveDrive.Req ByteArray) Rd.m)
    (rds : List (Game.LiveDrive.RoundBlock (Game.LiveDrive.Req ByteArray) Rd.m)) :
    Fin (ZkHidden.planZHJ (Reg := A.Reg) A.dj A.y₀ (Rd.drawOf kd') Rd.R₀).length → Fin 2 → List Flock.Round :=
  Game.LiveDrive.roundsOf Game.LiveDrive.nOfJ Game.LiveDrive.mkJ (Game.LiveDrive.appJ A.dg)
    (fun x => (ZkHidden.planZHJ (Reg := A.Reg) A.dj A.y₀ (Game.LiveDrive.drawOS (.subset kd) n x) Rd.R₀).length)
    Rd.m Rd.spR Rd.bytesOf (Rd.spΩ.e (Game.LiveDrive.decChunks Rd.m _ kd'.bits)) rds fun _ _ => []

/-- The play's outcome at the releases `hs` and the last message `e`. -/
def outAt (θ F c : ByteArray) (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))) (e : ByteArray × Array ByteArray) :
    Outcome :=
  ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩

/-- **The releases split into blocks** off the break, wherever sampled proofs' verdict accepts, the releases of the
play's bit counts (`Vocab.rounds`). -/
def RunsAt (V : Vocab d α) (θ F c : ByteArray) (Brk : Outcome → Prop) (Rd : Readers A) : Prop :=
  ∀ hs e, hs.map Sigma.fst = V.rounds θ → V.Verdict .sampled (outAt θ F c hs e) → ¬ Brk (outAt θ F c hs e) → ∃ kd' kp rds, Blocks Rd hs kd' kp rds

/-- **The record's draw is the one the draw's block reads back to**, at any split of the releases into blocks, off the
break, wherever sampled proofs' verdict accepts. -/
def DrawsAt (V : Vocab d α) (θ F c : ByteArray) (Brk : Outcome → Prop) (Rd : Readers A) : Prop :=
  ∀ hs e, V.Verdict .sampled (outAt θ F c hs e) → ¬ Brk (outAt θ F c hs e) → ∀ kd' kp rds, Blocks Rd hs kd' kp rds →
    DrawAt A.dj (Rd.drawOf kd') (proj (outAt θ F c hs e))

/-- **The record's link points and rounds are the ones their blocks read back to**, and its link points are released
after the hash the points' block carries, at any split of the releases into blocks, off the break, wherever sampled
proofs' verdict accepts. -/
def LinksAt (V : Vocab d α) (θ F c : ByteArray) (Brk : Outcome → Prop) (Rd : Readers A) : Prop :=
  ∀ hs e, V.Verdict .sampled (outAt θ F c hs e) → ¬ Brk (outAt θ F c hs e) → ∀ kd' kp rds, Blocks Rd hs kd' kp rds →
    PointsAt A.I (proj (outAt θ F c hs e)) (Rd.spP.e (Game.LiveDrive.decChunks Rd.m _ kp.bits)) ∧
    RoundsAt A.I (proj (outAt θ F c hs e)) (Rd.roundsAt kd' rds) ∧
    PointsAfter (proj (outAt θ F c hs e)) (Rd.hOf kp.b)

/-- **The reduction at the live session `A`**, from what the releases read back to: `Drive` at `proj`, `pfOf` at `A`'s
inputs, and the readers `Rd`. -/
noncomputable def drive (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (Brk : Outcome → Prop)
    (Rd : Readers A) (hIn : InputsAt V Brk A.I) (hRuns : RunsAt V θ F c Brk Rd) (hDraw : DrawsAt V θ F c Brk Rd)
    (hLink : LinksAt V θ F c Brk Rd) : Drive V θ F c σ Brk A where
  proj := proj
  covers _ hv hb := acceptsZK_proj hIn hv hb
  R₀ := Rd.R₀
  m := Rd.m
  tagOf := Rd.tagOf
  spΩ := Rd.spΩ
  spP := Rd.spP
  spR := Rd.spR
  hOf := Rd.hOf
  bytesOf := Rd.bytesOf
  pfOf := pfOf A.I θ F c
  canon hs e hsz hv hb := by
    obtain ⟨kd', kp, rds, hbl⟩ := hRuns hs e hsz hv hb
    obtain ⟨hPts, hRds, hAft⟩ := hLink hs e hv hb kd' kp rds hbl
    have hD := hDraw hs e hv hb kd' kp rds hbl
    obtain ⟨hc, hd, hp, hN, hr⟩ := hbl
    exact ⟨kd', kp, rds, hc, hd, hp, hN, hr, recordOfZKJ_proj hD hPts hRds,
      pointsHash_pfOf (sess0_of_verdict hv) (hIn _ hv hb) hAft⟩

end

/-- **`SampledDeploy.accept_le` from what the releases read back to**: at each session `A θ F c σ r hr`, its readers
`Rd`, and the play's session at `A`'s inputs (`InputsAt`), its releases split into blocks (`RunsAt`), the record's draw
(`DrawsAt`) and its link points and rounds (`LinksAt`) those blocks' reads, off the collision `colAt recsAt pairs` reads
(a break on an accepting case's uses, `brk_of_break`, or a collision among `pairs`) and the forger's success, sampled
proofs' verdict accepts at most as often as the live audit does at the reduction's strategy. -/
theorem accept_le_of_canon {V : Vocab d α} (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (pairs : ByteArray → Outcome → List (ByteArray × ByteArray))
    (Rd : ∀ θ F c σ r hr, Readers (A θ F c σ r hr))
    (hIn : ∀ θ F c σ r hr,
      InputsAt V (fun o => ColEvent (V.colAt recsAt pairs θ) o ∨ V.Forges .sampled θ o) (A θ F c σ r hr).I)
    (hRuns : ∀ θ F c σ r hr,
      RunsAt V θ F c (fun o => ColEvent (V.colAt recsAt pairs θ) o ∨ V.Forges .sampled θ o) (Rd θ F c σ r hr))
    (hDraw : ∀ θ F c σ r hr,
      DrawsAt V θ F c (fun o => ColEvent (V.colAt recsAt pairs θ) o ∨ V.Forges .sampled θ o) (Rd θ F c σ r hr))
    (hLink : ∀ θ F c σ r hr,
      LinksAt V θ F c (fun o => ColEvent (V.colAt recsAt pairs θ) o ∨ V.Forges .sampled θ o) (Rd θ F c σ r hr)) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), V.Registered θ F →
      Game.prob (fun o => V.Verdict .sampled o ∧ ¬ ColEvent (V.colAt recsAt pairs θ) o ∧ ¬ V.Forges .sampled θ o)
          (V.game θ F c) σ ≤
        FlockSoundness.Game.prob (fun o => ZkExec.LiveAcceptsZK (A θ F c σ r hr).I o) (A θ F c σ r hr).GH
          (drive V θ F c σ _ (Rd θ F c σ r hr) (hIn θ F c σ r hr) (hRuns θ F c σ r hr) (hDraw θ F c σ r hr)
            (hLink θ F c σ r hr)).τ :=
  accept_le_of_drives kd desc A (fun θ _ _ => V.colAt recsAt pairs θ) fun θ F c σ r hr =>
    drive V θ F c σ _ (Rd θ F c σ r hr) (hIn θ F c σ r hr) (hRuns θ F c σ r hr) (hDraw θ F c σ r hr)
      (hLink θ F c σ r hr)

end Verity.Canon
