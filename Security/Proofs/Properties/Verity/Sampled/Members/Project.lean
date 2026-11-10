module

public import Proofs.Properties.Verity.Sampled.CoinMap
public import Proofs.Properties.Verity.Sampled.Members.Session
public import Proofs.Properties.Verity.Sampled.Members.Widen

/-!
# A strategy of the period's game as a strategy of the window's audit

The one-session projection (`projectOf`, in `Proofs.Properties.Verity.Sampled.Project`) reads a strategy `σ` of the
period's game as a strategy of one session's live audit. Here it is read as a strategy of the window's audit
(`Flock.Guarantees.JointAuditV5`), whose members' sessions are played one after another on the one sample:

* `MemberDecoders`, `CoinMaps`: what the projection reads, member by member. The registration is the window's joint
  one; each member, by its place in the window, has its own decoders of `Commit`'s roots, the rounds' submissions and
  the proof files. The coin map has one widening and one start, the period's tape on the window's sample, and writes
  each member's live link points and round coins in at that member's stops (`openMaps`, into the stops' openings).
* `memberL`, `membersL`: a member's session's projection from a position of `σ`'s play, leaving the position its session
  ends at (`sessionL`), and the members' one after another, each from the position the one before leaves.
* `projectMembers`: the strategy of the window's audit widened by the coin map's coins (`JointAuditV5.extend`): the
  registration decoded from the record's most likely message; on the widened sample's coins, the members' projections
  from the play's start on the coin map's tape there. Each member's session is played at its part of the window's sample
  at the sample's own coins (`projectMembers_part`).
* `membersG`, `prob_projectMembers`, `membersG_reaches`: the joint law the projection simulates, the projected play's
  outcome beside the period's tape `σ` is played on, over the window's audit's coins; the projected play's events have
  their probabilities under it, and its support is where the members' projections end.
* `lands_memberL`, `lands_membersL_inv`: where a member's projection ends is where its session's does, relabelled by the
  member's writer, and the position the members leave is a later one that keeps the invariant the start does.
* `sampled_projectMembers`: `JointSampledV5`'s bound at the projection, in the registered audit's terms.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game Flock.Guarantees
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Discharge.Exec (mPtsOf Transcript)
open FlockSoundness.Discharge.Composed (nTab)
open FlockSoundness.Discharge.ZkHidden (planZHJP)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The parameters -/

/-- **The decoders a window's projection reads**: the joint registration, from the record's most likely message; and
for the member at each place of the window, `Commit`'s roots and a round's submission at `J` tables, from the most
likely message of each round whose cut is fixed at a stop, and the proof files, from the play's output. -/
structure MemberDecoders (R : ℕ) (Reg : Type) where
  reg : Option Claims.Msg → Reg
  root : ℕ → (Fin R → Option Claims.Msg) → Array ByteArray
  sub : ℕ → (J : ℕ) → (Fin R → Option Claims.Msg) → Option (SubmitJ J)
  proofs : ℕ → Exchange → Array ByteArray

/-- The decoders the `k`-th member's session's projection reads. -/
def MemberDecoders.member {R : ℕ} {Reg : Type} (D : MemberDecoders R Reg) (k : ℕ) : Decoders R Reg :=
  ⟨D.reg, D.root k, D.sub k, D.proofs k⟩

/-- **The coin map a window's projection reads**, at a strategy `σ` of the period's game `g`: the widening of the
window's sample (`T`); the period's tape on the window's sample and the widening (`start`); and at a stop of the
`k`-th member's session, the period's tape from there with the member's live link points (`link`), of any count, or a
round's live coins (`round`) written in. -/
structure CoinMaps (g : Game Exchange) (σ : Strategy g) (Ω : Type) (R : ℕ) where
  T : Type
  [fT : Fintype T]
  [nT : Nonempty T]
  start : Ω × T → CoinTape g σ
  link : ℕ → (p : Pos g σ Claims.Msg (Fin R)) → (mPts : ℕ) → (Fin 2 → Fin mPts → 𝔽) → CoinTape p.g' p.s' →
    CoinTape p.g' p.s'
  round : ℕ → (p : Pos g σ Claims.Msg (Fin R)) → (J : ℕ) → (s : SubmitJ J) → (Fin s.n → 𝔽) → CoinTape p.g' p.s' →
    CoinTape p.g' p.s'

attribute [instance] CoinMaps.fT CoinMaps.nT

/-- The coin map the `k`-th member's session's projection reads, at its link points' count. -/
def CoinMaps.member {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ} (M : CoinMaps g σ Ω R) (k mPts : ℕ) :
    CoinMap g σ Ω R mPts :=
  ⟨M.T, M.start, fun p => M.link k p mPts, M.round k⟩

/-- The widening a window's coin map reads. -/
def CoinMaps.widening {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ} (M : CoinMaps g σ Ω R) : Widening :=
  ⟨M.T⟩

/-- **The window's coin map that writes the live coins into the openings**, at the widening `T` and the start `start`:
at each stop of any member's session, the link points' bits or a round's coins' bits under `e`, written into the stop's
opening in a play that ends in `E` (`openMap`). -/
noncomputable def openMaps (e : 𝔽 ≃ (Fin 128 → Bool)) {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ) :
    CoinMaps g σ Ω R where
  T := T
  start := start
  link _ p _ pts := writeOpening E p.s' (linkBits e pts)
  round _ p _ _ c := writeOpening E p.s' (roundBits e c)

/-- Each member's coin map of the window's that writes into the openings is the session's (`openMap`). -/
theorem openMaps_member (e : 𝔽 ≃ (Fin 128 → Bool)) {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ)
    (k mPts : ℕ) : (openMaps e (R := R) E T start).member k mPts = openMap e E T start :=
  rfl

/-! ## The members' projections -/

section Members

variable {g : Game Exchange} {σ : Strategy g} {R : ℕ} (w : JointAuditV5) (D : MemberDecoders R w.Reg)
  (M : CoinMaps g σ w.Ω R)

/-- **The `k`-th member's session's projection** on the sample `S` at the registration `Rg`, from the position `p`: the
session's projection at the member's decoders and coin map, relabelled by the member's writer, leaving the position its
session ends at. -/
noncomputable def memberL (S : Finset (Fin w.nJ)) (Rg : w.Reg) (k : ℕ) (m : MemberV5 w.Reg w.nJ)
    (p : Pos g σ Claims.Msg (Fin R)) : Landing (Pos g σ Claims.Msg (Fin R)) (m.live (m.part S) Rg) :=
  Landing.map (m.wr (m.part S) Rg) _
    (sessionL (D.member k) (M.member k (mPtsOf m.D.unroll.g (m.kd / nTab m.I))) m.dg
      (planZHJP (Reg := w.Reg) m.dj m.y₀ (m.part S) Rg).length m.N p)

/-- **The members' sessions' projections** on the sample `S` at the registration `Rg`, from the position `p`: one after
another in the window's order, each from the position the one before leaves. -/
noncomputable def membersL (S : Finset (Fin w.nJ)) (Rg : w.Reg) (p : Pos g σ Claims.Msg (Fin R)) :
    Landing (Pos g σ Claims.Msg (Fin R)) (w.sessions S Rg) :=
  Landing.seq _ (memberL w D M S Rg) w.ms 0 p

/-- **Where a member's projection ends**: where its session's does, on the session's outcome as the member's writer
relabels it. -/
theorem lands_memberL (S : Finset (Fin w.nJ)) (Rg : w.Reg) (k : ℕ) (m : MemberV5 w.Reg w.nJ)
    (p : Pos g σ Claims.Msg (Fin R)) (t : Transcript) (q : Pos g σ Claims.Msg (Fin R)) :
    Landing.Lands _ (memberL w D M S Rg k m p) t q ↔
      ∃ o, Landing.Lands _ (sessionL (D.member k) (M.member k (mPtsOf m.D.unroll.g (m.kd / nTab m.I))) m.dg
          (planZHJP (Reg := w.Reg) m.dj m.y₀ (m.part S) Rg).length m.N p) o q ∧
        t = m.wr (m.part S) Rg o :=
  Landing.lands_map _ _ _ _ _

/-- **The position the members' projections leave is a later one than the start, and keeps the invariant the start
does.** -/
theorem lands_membersL_inv (S : Finset (Fin w.nJ)) (Rg : w.Reg) (p : Pos g σ Claims.Msg (Fin R))
    (ts : List Transcript) (q : Pos g σ Claims.Msg (Fin R)) (h : Landing.Lands _ (membersL w D M S Rg p) ts q)
    (Ks₀ : Fin R → DigestCut Claims.Msg g) (hp : p.Inv Ks₀) : p.Le q ∧ q.Inv Ks₀ :=
  Landing.lands_seq_rel _ (memberL w D M S Rg) (fun p q => p.Inv Ks₀ → p.Le q ∧ q.Inv Ks₀)
    (fun p hp => ⟨Pos.Le.refl p, hp⟩)
    (fun _ _ _ h₁ h₂ hp => ⟨(h₁ hp).1.trans (h₂ (h₁ hp).2).1, (h₂ (h₁ hp).2).2⟩)
    (fun k m p t q h hp =>
      let ⟨_, ho, _⟩ := (lands_memberL w D M S Rg k m p t q).1 h
      let ⟨_, _, hi⟩ := lands_sessionL _ _ _ _ _ _ _ _ ho
      hi Ks₀ hp)
    w.ms 0 p ts q h hp

end Members

/-! ## The projection on the window's audit -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The joint registration the projection sends**: the decoding of the record's most likely message. -/
noncomputable def membersReg {R : ℕ} {Reg : Type} (rd : Reading R) (σ : Strategy (V.game θ F c))
    (D : MemberDecoders R Reg) : Reg :=
  D.reg ((recordCut V θ F c rd).cut.modeHere σ)

/-- **The projection** of `σ` at the window's audit `w`, the reading `rd`, the decoders `D` and the coin map `M`: the
joint registration decoded from the record's most likely message; on the window's widened sample's coins `ω`, the
members' sessions' projections one after another from the play's start on `M`'s tape at `ω`, at the rounds' cuts. -/
noncomputable def projectMembers (w : JointAuditV5) {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c))
    (D : MemberDecoders R w.Reg) (M : CoinMaps (V.game θ F c) σ w.Ω R) :
    FlockSoundness.Game.Strategy (w.extend M.widening).game :=
  ⟨membersReg V θ F c rd σ D, fun ω => FlockSoundness.Game.Strategy.map _ _ (Landing.strat _
    (membersL w D M ((w.extend M.widening).L.draw ω) (membersReg V θ F c rd σ D)
      (Pos.root (roundCuts V θ F c rd) (M.start ω))))⟩

/-- **The projection plays each member's session at its part of the window's sample at the sample's own coins**: the
widening's coins move no member's sample. -/
theorem projectMembers_part (w : JointAuditV5) {g : Game Exchange} {σ : Strategy g} {R : ℕ}
    (M : CoinMaps g σ w.Ω R) (k : Fin w.ms.length) (ω : (w.extend M.widening).L.Ω) :
    (w.ms.get k).part ((w.extend M.widening).L.draw ω) = (w.design k).draw ω.1 :=
  w.extend_design_draw M.widening k ω

/-- **The joint law the projection simulates**: over the window's audit's coins, the projected play's outcome beside
the period's tape `σ` is played on. -/
noncomputable def membersG (w : JointAuditV5) {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c))
    (D : MemberDecoders R w.Reg) (M : CoinMaps (V.game θ F c) σ w.Ω R) :
    Game ((w.Reg × Finset (Fin w.nJ) × List Transcript) × CoinTape (V.game θ F c) σ) :=
  .coin (w.extend M.widening).L.Ω fun ω =>
    Landing.law (fun ts q => ((membersReg V θ F c rd σ D, (w.extend M.widening).L.draw ω, ts), Pos.tape q)) _
      (membersL w D M ((w.extend M.widening).L.draw ω) (membersReg V θ F c rd σ D)
        (Pos.root (roundCuts V θ F c rd) (M.start ω)))

/-- The joint law's strategy: it has no message to send. -/
noncomputable def membersS (w : JointAuditV5) {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c))
    (D : MemberDecoders R w.Reg) (M : CoinMaps (V.game θ F c) σ w.Ω R) : Strategy (membersG V θ F c w rd σ D M) :=
  fun _ => Landing.lawS _ _ _

/-- **The projected play's events have their probabilities under the joint law.** -/
theorem prob_projectMembers (w : JointAuditV5) {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c))
    (D : MemberDecoders R w.Reg) (M : CoinMaps (V.game θ F c) σ w.Ω R)
    (E : w.Reg × Finset (Fin w.nJ) × List Transcript → Prop) (ws : Strategy (membersG V θ F c w rd σ D M)) :
    FlockSoundness.Game.prob E (w.extend M.widening).game (projectMembers V θ F c w rd σ D M) =
      prob (fun o => E o.1) (membersG V θ F c w rd σ D M) ws := by
  refine Finset.sum_congr rfl fun ω _ => congrArg (HMul.hMul _) ?_
  exact (FlockSoundness.Discharge.Exec.prob_strategy_map _ _ _ _).trans
    (Landing.prob_strat
      (fun ts q => ((membersReg V θ F c rd σ D, (w.extend M.widening).L.draw ω, ts), Pos.tape q))
      (fun ts => E (membersReg V θ F c rd σ D, (w.extend M.widening).L.draw ω, ts)) (fun o => E o.1)
      (fun _ _ => Iff.rfl) _ _ (ws ω))

/-- **The joint law's support**: the joint registration, the sample of a widened sample's coins, and the members'
transcripts and the tape of the position their projections end on, from the play's start on the coin map's tape
there. -/
theorem membersG_reaches (w : JointAuditV5) {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c))
    (D : MemberDecoders R w.Reg) (M : CoinMaps (V.game θ F c) σ w.Ω R) (ws : Strategy (membersG V θ F c w rd σ D M))
    (o : (w.Reg × Finset (Fin w.nJ) × List Transcript) × CoinTape (V.game θ F c) σ)
    (h : PlayReaches (membersG V θ F c w rd σ D M) ws o) :
    ∃ (ω : (w.extend M.widening).L.Ω) (ts : List Transcript) (q : Pos (V.game θ F c) σ Claims.Msg (Fin R)),
      Landing.Lands _ (membersL w D M ((w.extend M.widening).L.draw ω) (membersReg V θ F c rd σ D)
        (Pos.root (roundCuts V θ F c rd) (M.start ω))) ts q ∧
      o = ((membersReg V θ F c rd σ D, (w.extend M.widening).L.draw ω, ts), q.tape) := by
  obtain ⟨ω, h⟩ := h
  obtain ⟨ts, q, hl, ho⟩ := Landing.lands_of_law _ _ _ _ _ h
  exact ⟨ω, ts, q, hl, ho⟩

end Verity.Sampled

namespace Flock.Guarantees.JointAuditV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **`JointSampledV5`'s bound at the projection, in the registered audit's terms**: for a strategy `σ` of the period's
game, read as the strategy `τ` of the window's audit widened by the coin map's coins (`Verity.Sampled.projectMembers`),
the probability that every member's session with a proof block in the sample is accepted and every proof block of `W` is
incorrect at `τ` is at most the registered sampling design's own miss of `W` plus the registered members' `ε_ks` and
`ε_bind`, under A3 at the window's bytes and `cr/sha-512` and `ecr/sha-512` at each member's session played alone at
`τ`. -/
theorem sampled_projectMembers (V : Verity.Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (w : JointAuditV5) {R : ℕ} (rd : Verity.Sampled.Reading R) (σ : Strategy (V.game θ F c))
    (D : Verity.Sampled.MemberDecoders R w.Reg) (M : Verity.Sampled.CoinMaps (V.game θ F c) σ w.Ω R)
    (hA3 : Assumptions.UniformRandomBytes w.Lb w.src)
    (hCR : ∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP ((w.extend M.widening).ms.get k).dj ((w.extend M.widening).ms.get k).y₀
        (((w.extend M.widening).memberLaw (Verity.Sampled.projectMembers V θ F c w rd σ D M) k).draw ω)
        (reg ((w.extend M.widening).memberSim (Verity.Sampled.projectMembers V θ F c w rd σ D M) k))) j
      (zcStrat execArith hs512 (planZHJP ((w.extend M.widening).ms.get k).dj ((w.extend M.widening).ms.get k).y₀)
        (cont ((w.extend M.widening).memberSim (Verity.Sampled.projectMembers V θ F c w rd σ D M) k) ω))
      ((w.extend M.widening).ms.get k).qF ((w.extend M.widening).ms.get k).qS)
    (hA2 : ∀ k, Assumptions.LinkCRZCAt execArith hs512 ((w.extend M.widening).ms.get k).P
      (planZHJP ((w.extend M.widening).ms.get k).dj ((w.extend M.widening).ms.get k).y₀)
      (tabZHJP ((w.extend M.widening).ms.get k).dj ((w.extend M.widening).ms.get k).y₀)
      ((w.extend M.widening).ms.get k).vb
      (reg ((w.extend M.widening).memberSim (Verity.Sampled.projectMembers V θ F c w rd σ D M) k))
      (cont ((w.extend M.widening).memberSim (Verity.Sampled.projectMembers V θ F c w rd σ D M) k))
      ((w.extend M.widening).ms.get k).k ((w.extend M.widening).ms.get k).Rw ((w.extend M.widening).ms.get k).M
      ((w.extend M.widening).ms.get k).t')
    (W : Finset (Fin w.nJ)) :
    prob (fun o => (w.extend M.widening).Accepts o ∧
        W ⊆ (w.extend M.widening).wrongSet (Verity.Sampled.projectMembers V θ F c w rd σ D M))
      (w.extend M.widening).game (Verity.Sampled.projectMembers V θ F c w rd σ D M) ≤
      ENNReal.ofReal (w.pa W) + w.eps :=
  w.sampled_extend hA3 M.widening (Verity.Sampled.projectMembers V θ F c w rd σ D M) hCR hA2 W

end Flock.Guarantees.JointAuditV5
