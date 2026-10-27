import Proofs.Properties.Verity.SampledAccept

/-!
# `PlayCoupling.prob_le` by driving the play through the live audit (ONE-8's `accept_le`, PRF-205)

The reduction runs a strategy of the play against the live audit (`Exec.auditLive` over `Composed.liveOfZJ`), one
release of the challenger device at a time (`Game.dom_reach`). It reads each release's bytes for what the release is to
the live audit (`LiveDrive.Tag`): a chunk of the window's draw, of session 0's link points, or of one of its rounds.

* The draw's chunks give the live draw through A3 (`dom_coin_right_map` at `uniform_of_A3`), the points' chunks the link
  points, after the roots the release answers (`rootOf`), and a round's chunks its coins, after the round the release
  answers (`bytesOf`). The device answers on the bytes' SHA-512 digest, so the reduction forwards the bytes it holds,
  and a record whose bytes differ under the same digest is a collision. A coin longer than a release is drawn chunk by
  chunk (`Game.Split`), each chunk the release's bits through a map keeping the uniform law and reading back to the
  chunk, its unread bits fixed at their best value.
* Every other release is left to the play, its bits fixed at their best value (`Move.skip`).

Each chunk reads back from its release's bits, so the live state the reduction reaches is a function of the play's
releases (`LiveDrive.reached`). `LiveDrive.dom_live` is the reduction, generic in the live session's rounds
(`LiveDrive.rounds`), and `toVerity_auditLive` is the live audit in that form. `dominated_of_drive` is the domination
`PlayCoupling.ofDominated` takes, under one hypothesis on the play's outcome: where it is accepted off the break, its
releases bring the reduction into the session's rounds, and the live audit's transcript at the proof files is
accepted. ONE-4's session check (`RoundsByDevice`: each round's coins the device's release after its digest, one
release per (session, round), in the live order) is what discharges it.
-/

namespace Verity.Game.LiveDrive

open Verity.Game

/-- What a release of the play is, to the reduction: a chunk of the draw, of session 0's link points, or of one of its
rounds (table, rep, coin count). -/
inductive Tag
  | draw
  | points
  | round (tab : ℕ) (rep : Fin 2) (n : ℕ)
  deriving DecidableEq

section

variable {F Rd β : Type} [Fintype F] [Nonempty F] {Sub : ℕ → Type} (nOf : {J : ℕ} → Sub J → ℕ)
  (mkS : {J : ℕ} → Fin J → Fin 2 → ByteArray → ℕ → Sub J)
  (app : {J : ℕ} → (s : Sub J) → (Fin (nOf s) → F) → (Fin J → Fin 2 → List Rd) → Fin J → Fin 2 → List Rd)

/-- The live rounds: the prover submits a round `s` on any stream or stops, at most `N` times; the round's `nOf s`
coins are drawn once it is fixed, and `app` records it. -/
def rounds (J : ℕ) : ℕ → (Fin J → Fin 2 → List Rd) → Game (Fin J → Fin 2 → List Rd)
  | 0, rs => .ret rs
  | N + 1, rs => .send (Option (Sub J)) fun
    | none => .ret rs
    | some s => .coin (Fin (nOf s) → F) fun c => rounds J N (app s c rs)

variable {X : Type} [Fintype X] [Nonempty X] (J : X → ℕ) (mPts N : ℕ)
  (out : (x : X) → Array ByteArray → (Fin 2 → Fin mPts → F) → (Fin (J x) → Fin 2 → List Rd) → Array ByteArray → β)

/-- The live session's rounds and proof files, at the draw `x`, the roots and the points. -/
def tail (x : X) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) (N' : ℕ) (rs : Fin (J x) → Fin 2 → List Rd) :
    Game β :=
  (rounds nOf app (J x) N' rs).bind fun rs' => .send (Array ByteArray) fun pf => .ret (out x rB pts rs' pf)

/-- The live session at the draw `x`: the roots, the link points, the rounds and the proof files. -/
def session (x : X) : Game β :=
  .send (Array ByteArray) fun rB => .coin (Fin 2 → Fin mPts → F) fun pts =>
    tail nOf app J mPts out x rB pts N fun _ _ => []

/-- Where the live audit is, between the play's releases. -/
inductive Phase
  | pre
  | sess (x : X)
  | rnds (x : X) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) (N' : ℕ) (rs : Fin (J x) → Fin 2 → List Rd)

variable {Ω : Type} [Fintype Ω] [Nonempty Ω] (t : Ω → X)

def phaseG : Phase J mPts (F := F) (Rd := Rd) → Game β
  | .pre => .coin Ω fun ω => session nOf app J mPts N out (t ω)
  | .sess x => session nOf app J mPts N out x
  | .rnds x rB pts N' rs => tail nOf app J mPts out x rB pts N' rs

variable (m : ℕ)

/-- The reduction's state: at a phase, or inside a coin drawn chunk by chunk (its tag, the chunks still to draw, and
the phase their values lead to). -/
inductive St
  | at (ph : Phase J mPts (F := F) (Rd := Rd))
  | mid (tg : Tag) (c : Chunk m) (cs : List (Chunk m)) (k : Chunks (c :: cs) → Phase J mPts (F := F) (Rd := Rd))

def G : St J mPts (F := F) (Rd := Rd) m → Game β
  | .at ph => phaseG nOf app J mPts N out t ph
  | .mid _ c cs k => nested (c :: cs) fun y => phaseG nOf app J mPts N out t (k y)

/-- The state once the chunks `cs` are left to draw toward `k`. -/
def St.after (tg : Tag) : (cs : List (Chunk m)) → (Chunks cs → Phase J mPts (F := F) (Rd := Rd)) →
    St J mPts (F := F) (Rd := Rd) m
  | [], k => .at (k ())
  | c :: cs, k => .mid tg c cs k

theorem G_after (tg : Tag) : ∀ (cs : List (Chunk m)) (k : Chunks cs → Phase J mPts (F := F) (Rd := Rd)),
    G nOf app J mPts N out t m (St.after J mPts m tg cs k) = nested cs fun y => phaseG nOf app J mPts N out t (k y)
  | [], _ => rfl
  | _ :: _, _ => rfl

/-! ### The plan -/

variable {α : Type} {p : α → Prop} {q : β → Prop} (ht : Uniform t) (tagOf : ByteArray → Option Tag)
  (spΩ : Split X m) (spP : Split (Fin 2 → Fin mPts → F) m) (spR : (n : ℕ) → Split (Fin n → F) m)
  (rootOf : ByteArray → Array ByteArray) (bytesOf : ByteArray → ByteArray)

/-- The draw's first chunk: the live draw through A3 (`t`), drawn chunk by chunk. -/
def stepDraw : Step q (G nOf app J mPts N out t m) (.at .pre) m where
  C := spΩ.first.P
  R := spΩ.first.R
  f := spΩ.first.f
  hf := spΩ.first.hf
  dec := spΩ.first.dec
  dec_f := spΩ.first.dec_f
  next a := St.after J mPts m .draw spΩ.rest fun r => .sess (spΩ.e (a, r))
  reach := fun _ _ _ h => by
    simp only [G_after] at h
    exact dom_coin_right_map (K := fun x => session nOf app J mPts N out x) t ht (dom_split spΩ h)

/-- The link points' first chunk, after the roots the reduction sends. -/
def stepPts (x : X) (rB : Array ByteArray) : Step q (G nOf app J mPts N out t m) (.at (.sess x)) m where
  C := spP.first.P
  R := spP.first.R
  f := spP.first.f
  hf := spP.first.hf
  dec := spP.first.dec
  dec_f := spP.first.dec_f
  next a := St.after J mPts m .points spP.rest fun r => .rnds x rB (spP.e (a, r)) N fun _ _ => []
  reach := fun _ _ _ h => by
    simp only [G_after] at h
    exact dom_send_right rB (dom_split spP h)

/-- A round's first chunk, after the round the reduction submits. -/
def stepRound (x : X) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) (N' : ℕ)
    (rs : Fin (J x) → Fin 2 → List Rd) (tab : Fin (J x)) (rep : Fin 2) (n : ℕ) (bytes : ByteArray) :
    Step q (G nOf app J mPts N out t m) (.at (.rnds x rB pts (N' + 1) rs)) m where
  C := (spR (nOf (mkS tab rep bytes n))).first.P
  R := (spR (nOf (mkS tab rep bytes n))).first.R
  f := (spR (nOf (mkS tab rep bytes n))).first.f
  hf := (spR (nOf (mkS tab rep bytes n))).first.hf
  dec := (spR (nOf (mkS tab rep bytes n))).first.dec
  dec_f := (spR (nOf (mkS tab rep bytes n))).first.dec_f
  next a := St.after J mPts m (.round tab rep n) (spR (nOf (mkS tab rep bytes n))).rest fun r =>
    .rnds x rB pts N' (app (mkS tab rep bytes n) ((spR (nOf (mkS tab rep bytes n))).e (a, r)) rs)
  reach := fun _ _ _ h => by
    simp only [G_after] at h
    exact dom_send_right (some (mkS tab rep bytes n)) (dom_split (spR _) h)

/-- A later chunk of the coin being drawn. -/
def stepMid (tg : Tag) (c : Chunk m) (cs : List (Chunk m)) (k : Chunks (c :: cs) → Phase J mPts (F := F) (Rd := Rd)) :
    Step q (G nOf app J mPts N out t m) (.mid tg c cs k) m where
  C := c.P
  R := c.R
  f := c.f
  hf := c.hf
  dec := c.dec
  dec_f := c.dec_f
  next a := St.after J mPts m tg cs fun r => k (a, r)
  reach := fun _ _ _ h => by
    simp only [G_after] at h
    exact h

/-- **The reduction's plan at a release of `m` bits** tagged `tagOf b`: the draw's chunks before the session, session
0's points' chunks after the draw, a round's chunks while rounds remain, and each coin's later chunks while it is being
drawn; every other release is left to the play, its bits at their best value. -/
def move (st : St J mPts (F := F) (Rd := Rd) m) (b : ByteArray) :
    Move q (G nOf app J mPts N out t m) st m :=
  match st with
  | .at .pre => if tagOf b = some .draw then .step (stepDraw nOf app J mPts N out t m ht spΩ) else .skip _ rfl
  | .at (.sess x) =>
    if tagOf b = some .points then .step (stepPts nOf app J mPts N out t m spP x (rootOf b)) else .skip _ rfl
  | .at (.rnds x rB pts N' rs) =>
    match N', tagOf b with
    | N'' + 1, some (.round tab rep n) =>
      if h : tab < J x then .step (stepRound nOf mkS app J mPts N out t m spR x rB pts N'' rs ⟨tab, h⟩ rep n (bytesOf b))
      else .skip _ rfl
    | _, _ => .skip _ rfl
  | .mid tg c cs k => if tagOf b = some tg then .step (stepMid nOf app J mPts N out t m tg c cs k) else .skip _ rfl

def plan (st : St J mPts (F := F) (Rd := Rd) m) (m' : ℕ) (b : ByteArray) :
    Move q (G nOf app J mPts N out t m) st m' :=
  if hm : m' = m then by subst hm; exact move nOf mkS app J mPts N out t m' ht tagOf spΩ spP spR rootOf bytesOf st b
  else .skip _ rfl

/-! ### The end of the play -/

/-- The prover stops: no more rounds, no proof files. -/
def stratTail (x : X) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) :
    (N' : ℕ) → (rs : Fin (J x) → Fin 2 → List Rd) → Strategy (tail nOf app J mPts out x rB pts N' rs)
  | 0, _ => ⟨#[], PUnit.unit⟩
  | _ + 1, _ => ⟨none, ⟨#[], PUnit.unit⟩⟩

def stratPhase : (ph : Phase J mPts (F := F) (Rd := Rd)) → Strategy (phaseG nOf app J mPts N out t ph)
  | .pre => fun _ => ⟨#[], fun _ => stratTail nOf app J mPts out _ _ _ N _⟩
  | .sess _ => ⟨#[], fun _ => stratTail nOf app J mPts out _ _ _ N _⟩
  | .rnds _ _ _ N' rs => stratTail nOf app J mPts out _ _ _ N' rs

def stratNested {γ : Type} : (cs : List (Chunk m)) → (K : Chunks cs → Game γ) → (∀ y, Strategy (K y)) →
    Strategy (nested cs K)
  | [], _, hK => hK ()
  | _ :: cs, _, hK => fun a => stratNested cs _ fun r => hK (a, r)

def stratG : (st : St J mPts (F := F) (Rd := Rd) m) → Strategy (G nOf app J mPts N out t m st)
  | .at ph => stratPhase nOf app J mPts N out t ph
  | .mid _ c cs k => stratNested m (c :: cs) _ fun y => stratPhase nOf app J mPts N out t (k y)

/-- The session's tail ends at the proof files the reduction sends. -/
theorem dom_tail {a : α} {s : Strategy (Game.ret a)} (x : X) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F)
    (pf : Array ByteArray) :
    ∀ (N' : ℕ) (rs : Fin (J x) → Fin 2 → List Rd), (p a → q (out x rB pts rs pf)) →
      Dominated p (.ret a) s q (tail nOf app J mPts out x rB pts N' rs)
  | 0, _, h => dom_send_right pf (dom_ret h s)
  | _ + 1, _, h => dom_send_right none (dom_send_right pf (dom_ret h s))

/-- **The live state the reduction reaches** after the play's releases `hs` (each its bit count, the bytes it answers
and its bits): the plan run over them from before the draw. -/
def reached (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))) : St J mPts (F := F) (Rd := Rd) m :=
  run (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (.at .pre) hs

/-- **The play's rounds driven through the live audit**: a strategy of the play (its rounds `ns` of released bits,
then a message `e` and its outcome `outc log e`) is dominated by the live audit after the draw, wherever every outcome
in `p` has its releases bring the reduction into the session's rounds (`reached`), with an outcome in `q` at the proof
files `pfOf e`. -/
theorem dom_live {ρ E : Type} (mk : ByteArray → List Bool → ρ) (outc : List ρ → E → α) (pfOf : E → Array ByteArray)
    (hok : ∀ hs e, p (outc (hs.map (recOf mk)) e) → ∃ x rB pts N' rs,
      reached (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf hs = .at (.rnds x rB pts N' rs) ∧
        q (out x rB pts rs (pfOf e)))
    (ns : List ℕ) (s : Strategy (roundsThen mk (fun log => .send E fun e => .ret (outc log e)) ns)) :
    Dominated p (roundsThen mk (fun log => .send E fun e => .ret (outc log e)) ns) s q
      (.coin Ω fun ω => session nOf app J mPts N out (t ω)) := by
  refine dom_reach mk (G nOf app J mPts N out t m) (plan nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf)
    (.at .pre) _ (fun hs K s' hK => ?_) ns s
  subst hK
  refine dom_send_left ?_
  by_cases hp : p (outc (hs.map (recOf mk) ++ []) s'.1)
  · rw [List.append_nil] at hp
    obtain ⟨x, rB, pts, N', rs, hst, hq⟩ := hok hs s'.1 hp
    rw [show run _ _ hs = _ from hst]
    exact dom_tail nOf app J mPts out x rB pts (pfOf s'.1) N' rs fun _ => hq
  · exact dom_ret_not hp _ (stratG nOf app J mPts N out t m _)

end

end Verity.Game.LiveDrive

/-! ## The live audit in the driver's form -/

namespace Verity.Game.LiveDrive

open FlockSoundness.Discharge.Tables (SubmitJ addRoundJ liveRoundsJ LiveOutJ KLJ liveSession)
open Pouw.PearlC.FlockTile (toVerity toVerity_bind toVerity_map)

local notation "𝔽" => FlockLevel3.GF128

/-- A submitted round's coin count. -/
abbrev nOfJ {J : ℕ} (s : SubmitJ J) : ℕ := s.n

/-- The round the reduction submits. -/
abbrev mkJ {J : ℕ} (tab : Fin J) (rep : Fin 2) (b : ByteArray) (n : ℕ) : SubmitJ J := ⟨tab, rep, b, n⟩

/-- The challenger appends a round at its coins. -/
abbrev appJ (dg : ByteArray → ByteArray) {J : ℕ} (s : SubmitJ J) (c : Fin (nOfJ s) → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) : Fin J → Fin 2 → List Flock.Round :=
  addRoundJ rs s.tab s.rep (FlockSoundness.Refine.roundOf dg s.bytes s.n c)

theorem toVerity_liveRoundsJ (dg : ByteArray → ByteArray) (J : ℕ) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round),
      toVerity (liveRoundsJ dg J N rs) = rounds nOfJ (appJ dg) J N rs
  | 0, _ => rfl
  | N + 1, rs => congrArg (Verity.Game.send _) (funext fun
    | none => rfl
    | some s => congrArg (Verity.Game.coin _) (funext fun c => toVerity_liveRoundsJ dg J N _))

/-- The draw `Audit.Law.execOS` makes from the bytes `x`. -/
noncomputable def drawOS {n mL : ℕ} {Lb : Fin mL → ℕ} (law : Flock.Draw.Law) (pop : ℕ)
    (x : (i : Fin mL) → Fin (Lb i) → Fin 256) : Finset (Fin n) :=
  (FlockSoundness.Audit.Law.execOS (n := n) (fun _ : Unit => x) law pop).draw ()

/-- **The live audit, in the driver's form**: the registration, the draw's bytes, then the session at the draw. -/
theorem toVerity_auditLive {n mL mPts : ℕ} {Lb : Fin mL → ℕ} {Reg Ω : Type} [Fintype Ω] [Nonempty Ω]
    (src : Ω → (i : Fin mL) → Fin (Lb i) → Fin 256) (law : Flock.Draw.Law) (pop : ℕ) (dg : ByteArray → ByteArray)
    (planZ : Finset (Fin n) → Reg → List (FlockSoundness.Discharge.ZkSession.TabZK 𝔽 mPts)) (N : ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts →
      FlockSoundness.Discharge.Exec.Transcript) :
    toVerity (FlockSoundness.Discharge.Exec.auditLive (FlockSoundness.Audit.Law.execOS (n := n) src law pop) Reg
      (FlockSoundness.Discharge.Composed.liveOfZJ dg planZ N wr)) =
      .send Reg fun R => .coin Ω fun ω =>
        session nOfJ (appJ dg) (fun x => (planZ (drawOS law pop x) R).length) mPts N
          (fun x rB pts rs pf => (R, drawOS law pop x, wr (drawOS law pop x) R ⟨rB, pts, rs, pf⟩)) (src ω) := by
  refine congrArg (Verity.Game.send Reg) (funext fun R => ?_)
  show Verity.Game.coin Ω (fun ω : Ω => toVerity
    ((FlockSoundness.Discharge.Composed.liveOfZJ dg planZ N wr (drawOS law pop (src ω)) R).map
      fun t' => (R, drawOS law pop (src ω), t'))) = _
  refine congrArg (Verity.Game.coin Ω) (funext fun ω => ?_)
  simp only [FlockSoundness.Discharge.Composed.liveOfZJ, toVerity_map, liveSession, toVerity, toVerity_bind,
    toVerity_liveRoundsJ, KLJ, session, tail, Verity.Game.map, Verity.Game.bind, Verity.Game.bind_assoc]

end Verity.Game.LiveDrive

/-! ## The play as rounds of releases, and the coupling -/

namespace Verity

open Accounting Core
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Pouw.PearlC.FlockTile (toVerity)

local notation "𝔽" => FlockLevel3.GF128

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **The device's releases, then `k`, are rounds of released bits** (`Game.roundsThen`), each recorded as its
release. -/
theorem releases_bind {β : Type} :
    ∀ (ns : List ℕ) (k : List Release → Game β),
      (releases ns).bind k = Game.roundsThen (fun b x => (⟨b, x⟩ : Release)) k ns
  | [], _ => rfl
  | n :: ns, k => by
    show Game.send _ (fun b => Game.coin _ fun x => ((releases ns).map _).bind k) = _
    unfold Game.roundsThen
    congr 1
    funext b
    congr 1
    funext x
    rw [Game.map, Game.bind_assoc]
    simp only [Game.bind_ret]
    exact releases_bind ns _

/-- **`prob_le`'s domination from the reduction**: the play at `σ` is dominated by the live audit, its draw from the
operating system's bytes `src` (A3), wherever every outcome the reduction reaches that is accepted off the break has its
live session in its rounds, at a live transcript flock-verify `--zk` accepts (`hok`). -/
theorem dominated_of_drive {V : Vocab d α} {θ F c : ByteArray} (σ : Game.Strategy (V.game θ F c))
    {Brk : Outcome → Prop} {n mL mPts : ℕ} {Lb : Fin mL → ℕ} {Reg Ω : Type} [Fintype Ω] [Nonempty Ω] {I : Exec.Inputs}
    (proj : Outcome → Exec.Transcript) (src : Ω → (i : Fin mL) → Fin (Lb i) → Fin 256)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src) (law : Flock.Draw.Law) (pop : ℕ)
    (dg : ByteArray → ByteArray) (planZ : Finset (Fin n) → Reg → List (ZkSession.TabZK 𝔽 mPts)) (N : ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript) (R₀ : Reg) (m : ℕ)
    (tagOf : ByteArray → Option Game.LiveDrive.Tag) (spΩ : Game.Split ((i : Fin mL) → Fin (Lb i) → Fin 256) m)
    (spP : Game.Split (Fin 2 → Fin mPts → 𝔽) m) (spR : (k : ℕ) → Game.Split (Fin k → 𝔽) m)
    (rootOf : ByteArray → Array ByteArray) (bytesOf : ByteArray → ByteArray) (pfOf : ByteArray → Array ByteArray)
    (hok : ∀ hs e, ZkExec.AcceptsZK I (proj ⟨θ, F, c, hs.map (Game.recOf fun b x => (⟨b, x⟩ : Release)), e⟩) →
      ¬ Brk ⟨θ, F, c, hs.map (Game.recOf fun b x => (⟨b, x⟩ : Release)), e⟩ →
      ∃ x rB pts N' rs, Game.LiveDrive.reached (q := fun o => ZkExec.LiveAcceptsZK I o) Game.LiveDrive.nOfJ
          Game.LiveDrive.mkJ (Game.LiveDrive.appJ dg) (fun x => (planZ (Game.LiveDrive.drawOS law pop x) R₀).length)
          mPts N (fun x rB pts rs pf => (R₀, Game.LiveDrive.drawOS law pop x,
            wr (Game.LiveDrive.drawOS law pop x) R₀ ⟨rB, pts, rs, pf⟩)) src m (uniform_of_A3 hA3) tagOf spΩ spP spR
          rootOf bytesOf hs = .at (.rnds x rB pts N' rs) ∧
        ZkExec.AcceptsZK I (wr (Game.LiveDrive.drawOS law pop x) R₀ ⟨rB, pts, rs, pfOf e⟩)) :
    Game.Dominated (fun o => ZkExec.AcceptsZK I (proj o) ∧ ¬ Brk o) (V.game θ F c) σ
      (fun o => ZkExec.LiveAcceptsZK I o)
      (toVerity (Exec.auditLive (FlockSoundness.Audit.Law.execOS (n := n) src law pop) Reg
        (Composed.liveOfZJ dg planZ N wr))) := by
  rw [Game.LiveDrive.toVerity_auditLive]
  refine Game.dom_send_right R₀ (Game.Dominated.of_eq
    (releases_bind (V.rounds θ) fun log => Game.send ByteArray fun e => Game.ret (⟨θ, F, c, log, e⟩ : Outcome)) ?_)
  exact Game.LiveDrive.dom_live Game.LiveDrive.nOfJ Game.LiveDrive.mkJ (Game.LiveDrive.appJ dg) _ mPts N _ src m
    (uniform_of_A3 hA3) tagOf spΩ spP spR rootOf bytesOf _ _ pfOf (fun hs e h => hok hs e h.1 h.2) (V.rounds θ) _

end Verity
