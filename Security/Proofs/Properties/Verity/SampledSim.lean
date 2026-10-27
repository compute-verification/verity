import Proofs.Properties.Verity.SampledAccept
import Proofs.Properties.Verity.PlayDevice
import Proofs.Properties.Verity.PlayDeviceLog
import Proofs.Properties.Verity.AcceptRounds

/-!
# `PlayCoupling.prob_le` by driving the play through the live audit (`SampledDeploy.accept_le`'s reduction)

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
accepted.

`dominated_of_canon` states that hypothesis as the device's refusals read in the play (`RoundsByDevice` with v2's
`coinsMismatch`: each round's coins the device's release after its digest, one release per (session, round), in the
live order). An untagged release never moves the reduction (`reached_filter`); where the tagged releases are the draw's
block, the link points', then one block per round, each one release per chunk, the reduction is in the rounds at what
they read back to (`reached_canon`); and the record is the live audit's transcript there.

`Drive` bundles the reduction at one live session of `SampledDeploy.A` (the coupling's `proj` and `covers`, how it
reads the releases, and those refusals), and gives the strategy `Drive.τ` and the coupling `Drive.coupling`.
`accept_le_of_drives` is `SampledDeploy.accept_le` at `τOf := (Dr …).τ`.
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

variable {Xd : Type} [Fintype Xd] [Nonempty Xd] (J : Xd → ℕ) (mPts N : ℕ)
  (out : (x : Xd) → Array ByteArray → (Fin 2 → Fin mPts → F) → (Fin (J x) → Fin 2 → List Rd) → Array ByteArray → β)

/-- The live session's rounds and proof files, at the draw `x`, the roots and the points. -/
def tail (x : Xd) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) (N' : ℕ) (rs : Fin (J x) → Fin 2 → List Rd) :
    Game β :=
  (rounds nOf app (J x) N' rs).bind fun rs' => .send (Array ByteArray) fun pf => .ret (out x rB pts rs' pf)

/-- The live session at the draw `x`: the roots, the link points, the rounds and the proof files. -/
def session (x : Xd) : Game β :=
  .send (Array ByteArray) fun rB => .coin (Fin 2 → Fin mPts → F) fun pts =>
    tail nOf app J mPts out x rB pts N fun _ _ => []

/-- Where the live audit is, between the play's releases. -/
inductive Phase
  | pre
  | sess (x : Xd)
  | rnds (x : Xd) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) (N' : ℕ) (rs : Fin (J x) → Fin 2 → List Rd)

variable {Ω : Type} [Fintype Ω] [Nonempty Ω] (t : Ω → Xd)

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
  (spΩ : Split Xd m) (spP : Split (Fin 2 → Fin mPts → F) m) (spR : (n : ℕ) → Split (Fin n → F) m)
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
def stepPts (x : Xd) (rB : Array ByteArray) : Step q (G nOf app J mPts N out t m) (.at (.sess x)) m where
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
def stepRound (x : Xd) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) (N' : ℕ)
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
def stratTail (x : Xd) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) :
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
theorem dom_tail {a : α} {s : Strategy (Game.ret a)} (x : Xd) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F)
    (pf : Array ByteArray) :
    ∀ (N' : ℕ) (rs : Fin (J x) → Fin 2 → List Rd), (p a → q (out x rB pts rs pf)) →
      Dominated p (.ret a) s q (tail nOf app J mPts out x rB pts N' rs)
  | 0, _, h => dom_send_right pf (dom_ret h s)
  | _ + 1, _, h => dom_send_right none (dom_send_right pf (dom_ret h s))

/-- **The live state the reduction reaches** after the play's releases `hs` (each its bit count, the bytes it answers
and its bits): the plan run over them from before the draw. -/
def reached (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))) : St J mPts (F := F) (Rd := Rd) m :=
  runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (.at .pre) hs

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
    rw [show runPlan _ _ hs = _ from hst]
    exact dom_tail nOf app J mPts out x rB pts (pfOf s'.1) N' rs fun _ => hq
  · exact dom_ret_not hp _ (stratG nOf app J mPts N out t m _)

/-! ### The releases the session's record reads -/

/-- A release of `m` bits, as the play records it. -/
def rel (z : ByteArray × (Fin m → Bool)) : Σ m' : ℕ, ByteArray × (Fin m' → Bool) := ⟨m, z⟩

/-- A release the plan may step: `m` bits, answering bytes with a tag. -/
def Tagged (z : Σ m' : ℕ, ByteArray × (Fin m' → Bool)) : Bool := decide (z.1 = m) && (tagOf z.2.1).isSome

/-- The chunks a list of releases' bits read back to (a missing release's at no bits set). -/
def decChunks : (cs : List (Chunk m)) → List (Fin m → Bool) → Chunks cs
  | [], _ => ()
  | c :: cs, [] => (c.dec fun _ => false, decChunks cs [])
  | c :: cs, x :: xs => (c.dec x, decChunks cs xs)

theorem plan_self (st : St J mPts (F := F) (Rd := Rd) m) (b : ByteArray) :
    plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf st m b = move nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf st b := by
  simp only [plan, dite_true]

/-- **An untagged release never moves the reduction**: the live state the play's releases reach is the one their
tagged releases reach. -/
theorem run_filter : ∀ (st : St J mPts (F := F) (Rd := Rd) m) (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))),
    runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) st hs = runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) st (hs.filter (Tagged m tagOf))
  | _, [] => rfl
  | st, ⟨m', b, x⟩ :: hs => by
    by_cases hz : Tagged m tagOf ⟨m', b, x⟩ = true
    · rw [List.filter_cons_of_pos hz]
      exact run_filter _ hs
    · rw [List.filter_cons_of_neg hz]
      have hst : (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf st m' b).next x = st := by
        simp only [Tagged, Bool.and_eq_true, decide_eq_true_eq, not_and, Option.isSome_iff_ne_none, not_not] at hz
        by_cases hm : m' = m
        · subst hm
          rw [plan_self]
          have hb := hz rfl
          rcases st with ((_ | _ | ⟨_, _, _, _ | _, _⟩) | ⟨_, _, _, _⟩) <;> simp [move, hb, Move.next]
        · simp [plan, hm, Move.next]
      show runPlan _ ((plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf st m' b).next x) hs = _
      rw [hst]
      exact run_filter st hs

theorem reached_filter (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))) :
    reached (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf hs =
      reached (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf (hs.filter (Tagged m tagOf)) :=
  run_filter nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf _ hs

theorem run_append' (st : St J mPts (F := F) (Rd := Rd) m) :
    ∀ hs hs' : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool)), runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) st (hs ++ hs') = runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) st hs) hs'
  | [], _ => rfl
  | ⟨_, _, _⟩ :: hs, hs' => run_append' _ hs hs'

/-- A coin's later chunks: releases tagged `tg`, one per chunk, bring the reduction to the phase at the chunks they read
back to. -/
theorem run_mid (tg : Tag) : ∀ (cs : List (Chunk m)) (k : Chunks cs → Phase J mPts (F := F) (Rd := Rd))
    (rels : List (ByteArray × (Fin m → Bool))), rels.length = cs.length → (∀ z ∈ rels, tagOf z.1 = some tg) →
    runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (St.after J mPts m tg cs k) (rels.map (rel m)) = .at (k (decChunks m cs (rels.map Prod.snd)))
  | [], _, [], _, _ => rfl
  | c :: cs, k, (b, x) :: rels, hl, htg => by
    show runPlan _ ((plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf (.mid tg c cs k) m b).next x) (rels.map (rel m)) = _
    rw [plan_self]
    simp only [move, htg (b, x) List.mem_cons_self, ↓reduceIte]
    exact run_mid tg cs (fun r => k (c.dec x, r)) rels (by simpa using hl)
      (fun z hz => htg z (List.mem_cons_of_mem _ hz))

/-- **A block of releases**: the first answers the bytes the coin is drawn after, and the coin is drawn from its bits
and the rest's. -/
structure Block (m : ℕ) where
  b : ByteArray
  x : Fin m → Bool
  rest : List (ByteArray × (Fin m → Bool))

def Block.rels (k : Block m) : List (ByteArray × (Fin m → Bool)) := (k.b, k.x) :: k.rest

def Block.bits (k : Block m) : List (Fin m → Bool) := k.x :: k.rest.map Prod.snd

/-- Every release of the block answers bytes tagged `tg`, and it has one release per chunk of `sp`. -/
def Block.Fits {Yd : Type} (k : Block m) (tg : Tag) (sp : Split Yd m) : Prop :=
  (∀ z ∈ k.rels, tagOf z.1 = some tg) ∧ k.rest.length = sp.rest.length

/-- The draw's block, before the session: the reduction reaches the session at the draw it reads back to. -/
theorem run_draw (k : Block m) (hk : k.Fits m tagOf .draw spΩ) :
    runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (.at .pre) (k.rels.map (rel m)) = .at (.sess (spΩ.e (decChunks m _ k.bits))) := by
  show runPlan _ ((plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf (.at .pre) m k.b).next k.x) (k.rest.map (rel m)) = _
  rw [plan_self]
  simp only [move, hk.1 (k.b, k.x) List.mem_cons_self, ↓reduceIte]
  exact run_mid nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf .draw spΩ.rest _ k.rest hk.2
    (fun z hz => hk.1 z (List.mem_cons_of_mem _ hz))

/-- The link points' block, at the session: the reduction reaches the rounds after the roots its first release answers,
at the points it reads back to. -/
theorem run_points (x : Xd) (k : Block m) (hk : k.Fits m tagOf .points spP) :
    runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (.at (.sess x)) (k.rels.map (rel m)) =
      .at (.rnds x (rootOf k.b) (spP.e (decChunks m _ k.bits)) N fun _ _ => []) := by
  show runPlan _ ((plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf (.at (.sess x)) m k.b).next k.x) (k.rest.map (rel m)) = _
  rw [plan_self]
  simp only [move, hk.1 (k.b, k.x) List.mem_cons_self, ↓reduceIte]
  exact run_mid nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf .points spP.rest _ k.rest hk.2
    (fun z hz => hk.1 z (List.mem_cons_of_mem _ hz))

/-- A round's releases: its table, rep and coin count, and its block, whose first release answers the round's bytes. -/
structure RoundBlock (m : ℕ) where
  tab : ℕ
  rep : Fin 2
  n : ℕ
  k : Block m

/-- The round a block submits at the draw `x`, where its table is one of the draw's. -/
def RoundBlock.sub (x : Xd) (r : RoundBlock m) (h : r.tab < J x) : Sub (J x) :=
  mkS ⟨r.tab, h⟩ r.rep (bytesOf r.k.b) r.n

/-- The block fits its round: a table of the draw's, its releases tagged with the round, one per chunk. -/
def RoundBlock.Fits (x : Xd) (r : RoundBlock m) : Prop :=
  ∃ h : r.tab < J x, r.k.Fits m tagOf (.round r.tab r.rep r.n) (spR (nOf (r.sub mkS J m bytesOf x h)))

/-- The rounds the blocks submit, at the coins they read back to. -/
def roundsOf (x : Xd) : List (RoundBlock m) → (Fin (J x) → Fin 2 → List Rd) → (Fin (J x) → Fin 2 → List Rd)
  | [], rs => rs
  | r :: rds, rs => roundsOf x rds
    (if h : r.tab < J x then
      app (r.sub mkS J m bytesOf x h) ((spR (nOf (r.sub mkS J m bytesOf x h))).e (decChunks m _ r.k.bits)) rs
    else rs)

/-- The rounds' blocks, in the rounds: the reduction submits each round and appends it at the coin it reads back to. -/
theorem run_rounds (x : Xd) (rB : Array ByteArray) (pts : Fin 2 → Fin mPts → F) :
    ∀ (rds : List (RoundBlock m)) (N' : ℕ) (rs : Fin (J x) → Fin 2 → List Rd), rds.length ≤ N' →
      (∀ r ∈ rds, r.Fits nOf mkS J m tagOf spR bytesOf x) →
      runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (.at (.rnds x rB pts N' rs)) ((rds.flatMap fun r => r.k.rels).map (rel m)) =
        .at (.rnds x rB pts (N' - rds.length) (roundsOf nOf mkS app J m spR bytesOf x rds rs))
  | [], N', rs, _, _ => by simp [roundsOf, runPlan]
  | r :: rds, N', rs, hl, hf => by
    obtain ⟨N'', rfl⟩ : ∃ N'', N' = N'' + 1 := ⟨N' - 1, by simp at hl; omega⟩
    obtain ⟨h, hk⟩ := hf r List.mem_cons_self
    have h1 : runPlan (plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf) (.at (.rnds x rB pts (N'' + 1) rs)) (r.k.rels.map (rel m)) =
        .at (.rnds x rB pts N'' (app (r.sub mkS J m bytesOf x h)
          ((spR (nOf (r.sub mkS J m bytesOf x h))).e (decChunks m _ r.k.bits)) rs)) := by
      show runPlan _ ((plan (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf (.at (.rnds x rB pts (N'' + 1) rs)) m r.k.b).next r.k.x) (r.k.rest.map (rel m)) = _
      rw [plan_self]
      simp only [move, hk.1 (r.k.b, r.k.x) List.mem_cons_self, h, ↓reduceDIte]
      exact run_mid nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf (.round r.tab r.rep r.n) _ _ r.k.rest hk.2
        (fun z hz => hk.1 z (List.mem_cons_of_mem _ hz))
    rw [List.flatMap_cons, List.map_append, run_append', h1,
      run_rounds x rB pts rds N'' _ (by simp at hl; omega) (fun r' hr' => hf r' (List.mem_cons_of_mem _ hr'))]
    simp only [roundsOf, h, ↓reduceDIte, List.length_cons]
    congr 2
    omega

/-- **The record's releases, in the live order, reach the rounds** (the device's refusals, read in the play): where the
play's tagged releases are the draw's block, then the link points', then one block per round, each block one release
per chunk of its coin (one release per (session, round), the draw before the points before the rounds), the reduction
is in the session's rounds at the draw, roots, points and rounds those releases read back to. -/
theorem reached_canon (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))) (kd kp : Block m)
    (rds : List (RoundBlock m))
    (hcanon : hs.filter (Tagged m tagOf) = (kd.rels ++ kp.rels ++ rds.flatMap fun r => r.k.rels).map (rel m))
    (hd : kd.Fits m tagOf .draw spΩ) (hp : kp.Fits m tagOf .points spP) (hN : rds.length ≤ N)
    (hr : ∀ r ∈ rds, r.Fits nOf mkS J m tagOf spR bytesOf (spΩ.e (decChunks m _ kd.bits))) :
    reached (q := q) nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf hs =
      .at (.rnds (spΩ.e (decChunks m _ kd.bits)) (rootOf kp.b) (spP.e (decChunks m _ kp.bits)) (N - rds.length)
        (roundsOf nOf mkS app J m spR bytesOf (spΩ.e (decChunks m _ kd.bits)) rds fun _ _ => [])) := by
  rw [reached_filter, hcanon, reached, List.map_append, List.map_append, run_append', run_append',
    run_draw nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf kd hd, run_points nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf _ kp hp, run_rounds nOf mkS app J mPts N out t m ht tagOf spΩ spP spR rootOf bytesOf _ _ _ rds N _ hN hr]

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
open Properties.SampledProofs.OneStageZk (SessionInputs SessionTranscript)

local notation "𝔽" => FlockLevel3.GF128

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- The release the device records for the bytes `b` and the coins `l`. -/
def releaseOf (b : ByteArray) (l : List Bool) : Release := ⟨b, l⟩

/-- **The device's releases, then `k`, are rounds of released bits** (`Game.roundsThen`), each recorded as its
release. -/
theorem releases_bind {β : Type} :
    ∀ (ns : List ℕ) (k : List Release → Game β),
      (releases ns).bind k = Game.roundsThen releaseOf k ns
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

/-- **The play dominated by the live audit**: the play at `σ`, in the event `P`, is dominated by the live audit, its
draw from the operating system's bytes `src` (A3), wherever every outcome in `P` has its releases bring the reduction
into the session's rounds, at a live transcript flock-verify `--zk` accepts (`hok`). -/
theorem dominated_of_drive {V : Vocab d α} {θ F c : ByteArray} (σ : Game.Strategy (V.game θ F c))
    (P : Outcome → Prop) {n mL mPts : ℕ} {Lb : Fin mL → ℕ} {Reg Ω : Type} [Fintype Ω] [Nonempty Ω] {I : Exec.Inputs}
    (src : Ω → (i : Fin mL) → Fin (Lb i) → Fin 256)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src) (law : Flock.Draw.Law) (pop : ℕ)
    (dg : ByteArray → ByteArray) (planZ : Finset (Fin n) → Reg → List (ZkSession.TabZK 𝔽 mPts)) (N : ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript) (R₀ : Reg) (m : ℕ)
    (tagOf : ByteArray → Option Game.LiveDrive.Tag) (spΩ : Game.Split ((i : Fin mL) → Fin (Lb i) → Fin 256) m)
    (spP : Game.Split (Fin 2 → Fin mPts → 𝔽) m) (spR : (k : ℕ) → Game.Split (Fin k → 𝔽) m)
    (rootOf : ByteArray → Array ByteArray) (bytesOf : ByteArray → ByteArray) (pfOf : ByteArray × Array ByteArray → Array ByteArray)
    (hok : ∀ hs e, P ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩ →
      ∃ x rB pts N' rs, Game.LiveDrive.reached
          (q := fun o : Reg × Finset (Fin n) × Exec.Transcript => ZkExec.LiveAcceptsZK I o) Game.LiveDrive.nOfJ
          Game.LiveDrive.mkJ (Game.LiveDrive.appJ dg) (fun x => (planZ (Game.LiveDrive.drawOS law pop x) R₀).length)
          mPts N (fun x rB pts rs pf => (R₀, Game.LiveDrive.drawOS (n := n) law pop x,
            wr (Game.LiveDrive.drawOS law pop x) R₀ ⟨rB, pts, rs, pf⟩)) src m (uniform_of_A3 hA3) tagOf spΩ spP spR
          rootOf bytesOf hs = .at (.rnds x rB pts N' rs) ∧
        ZkExec.AcceptsZK I (wr (Game.LiveDrive.drawOS law pop x) R₀ ⟨rB, pts, rs, pfOf e⟩)) :
    Game.Dominated P (V.game θ F c) σ (fun o => ZkExec.LiveAcceptsZK I o)
      (toVerity (Exec.auditLive (FlockSoundness.Audit.Law.execOS (n := n) src law pop) Reg
        (Composed.liveOfZJ dg planZ N wr))) := by
  rw [Game.LiveDrive.toVerity_auditLive]
  refine Game.dom_send_right R₀ (Game.Dominated.of_eq
    (releases_bind (V.rounds θ) fun log => Game.send (ByteArray × Array ByteArray) fun eb =>
      Game.ret (⟨θ, F, c, log, eb.1, eb.2⟩ : Outcome)) ?_)
  exact Game.LiveDrive.dom_live Game.LiveDrive.nOfJ Game.LiveDrive.mkJ (Game.LiveDrive.appJ dg) _ mPts N _ src m
    (uniform_of_A3 hA3) tagOf spΩ spP spR rootOf bytesOf _ _ pfOf hok (V.rounds θ) _

/-- **`prob_le`'s domination from the device's refusals, read in the play**: where sampled proofs' verdict accepts the
play's outcome off the break, its tagged releases are the draw's block, then the link points', then one block per round, each one release per
chunk of its coin (one release per (session, round), in the live order: `hcanon`'s first five conjuncts), and the
transcript `proj` reads is the live audit's at the draw, roots, points and rounds those releases read back to, and the
proof files (the record's coins the device's releases: its last). -/
theorem dominated_of_canon {V : Vocab d α} {θ F c : ByteArray} (σ : Game.Strategy (V.game θ F c))
    {Brk : Outcome → Prop} {n mL mPts : ℕ} {Lb : Fin mL → ℕ} {Reg Ω : Type} [Fintype Ω] [Nonempty Ω] {I : Exec.Inputs}
    (proj : Outcome → Exec.Transcript) (src : Ω → (i : Fin mL) → Fin (Lb i) → Fin 256)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src) (law : Flock.Draw.Law) (pop : ℕ)
    (dg : ByteArray → ByteArray) (planZ : Finset (Fin n) → Reg → List (ZkSession.TabZK 𝔽 mPts)) (N : ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript) (R₀ : Reg) (m : ℕ)
    (tagOf : ByteArray → Option Game.LiveDrive.Tag) (spΩ : Game.Split ((i : Fin mL) → Fin (Lb i) → Fin 256) m)
    (spP : Game.Split (Fin 2 → Fin mPts → 𝔽) m) (spR : (k : ℕ) → Game.Split (Fin k → 𝔽) m)
    (rootOf : ByteArray → Array ByteArray) (bytesOf : ByteArray → ByteArray) (pfOf : ByteArray × Array ByteArray → Array ByteArray)
    (hcanon : ∀ hs e, V.Verdict .sampled ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩ →
      ¬ Brk ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩ →
      ∃ (kd kp : Game.LiveDrive.Block m) (rds : List (Game.LiveDrive.RoundBlock m)),
        hs.filter (Game.LiveDrive.Tagged m tagOf) =
            (kd.rels ++ kp.rels ++ rds.flatMap fun r => r.k.rels).map (Game.LiveDrive.rel m) ∧
          kd.Fits m tagOf .draw spΩ ∧ kp.Fits m tagOf .points spP ∧ rds.length ≤ N ∧
          (∀ r ∈ rds, r.Fits Game.LiveDrive.nOfJ Game.LiveDrive.mkJ
            (fun x => (planZ (Game.LiveDrive.drawOS law pop x) R₀).length) m tagOf spR bytesOf
            (spΩ.e (Game.LiveDrive.decChunks m _ kd.bits))) ∧
          proj ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩ =
            wr (Game.LiveDrive.drawOS law pop (spΩ.e (Game.LiveDrive.decChunks m _ kd.bits))) R₀
              ⟨rootOf kp.b, spP.e (Game.LiveDrive.decChunks m _ kp.bits),
                Game.LiveDrive.roundsOf Game.LiveDrive.nOfJ Game.LiveDrive.mkJ (Game.LiveDrive.appJ dg)
                  (fun x => (planZ (Game.LiveDrive.drawOS law pop x) R₀).length) m spR bytesOf
                  (spΩ.e (Game.LiveDrive.decChunks m _ kd.bits)) rds fun _ _ => [], pfOf e⟩) :
    Game.Dominated (fun o => V.Verdict .sampled o ∧ ZkExec.AcceptsZK I (proj o) ∧ ¬ Brk o) (V.game θ F c) σ
      (fun o => ZkExec.LiveAcceptsZK I o)
      (toVerity (Exec.auditLive (FlockSoundness.Audit.Law.execOS (n := n) src law pop) Reg
        (Composed.liveOfZJ dg planZ N wr))) := by
  refine dominated_of_drive σ _ src hA3 law pop dg planZ N wr R₀ m tagOf spΩ spP spR rootOf bytesOf pfOf
    fun hs e ⟨hv, hacc, hb⟩ => ?_
  obtain ⟨kd, kp, rds, hc, hd, hp, hN, hr, hpr⟩ := hcanon hs e hv hb
  refine ⟨_, _, _, _, _, Game.LiveDrive.reached_canon (q := fun o => ZkExec.LiveAcceptsZK I o) Game.LiveDrive.nOfJ
    Game.LiveDrive.mkJ (Game.LiveDrive.appJ dg) _ mPts N _ src m (uniform_of_A3 hA3) tagOf spΩ spP spR rootOf bytesOf
    hs kd kp rds hc hd hp hN hr, ?_⟩
  rw [← hpr]
  exact hacc

/-- **The reduction at a live session `A`**: the coupling's transcript of the play (`proj`: off the break, where
sampled proofs' verdict accepts, a session's transcript of v2's evidence at `A`'s inputs, `covers`), the registration
it sends the live audit (`R₀`), and how it reads the device's releases: their bits (`m`), their tags, the chunks of the
draw, the link points and the rounds, the roots and the rounds' bytes the releases answer, and the proof files the
evidence holds. `canon` is the device's refusals and v2's coins, read in the play (`dominated_of_canon`). -/
structure Drive (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (Brk : Outcome → Prop)
    {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) where
  proj : Outcome → Exec.Transcript
  covers : ∀ o, V.Verdict .sampled o → ¬ Brk o → ∀ p c' e pub prof, SampledCase o.sent p c' e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c' e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
    SessionInputs p e o.sent.blobs A.I ∧ ∃ s ∈ e.sessions, SessionTranscript p e o.sent.blobs s (proj o)
  R₀ : A.Reg
  m : ℕ
  tagOf : ByteArray → Option Game.LiveDrive.Tag
  spΩ : Game.Split ((i : Fin 1) → Fin (A.Lb i) → Fin 256) m
  spP : Game.Split (Fin 2 → Fin (Exec.mPtsOf D.unroll.g (kd / Composed.nTab A.I)) → 𝔽) m
  spR : (k : ℕ) → Game.Split (Fin k → 𝔽) m
  rootOf : ByteArray → Array ByteArray
  bytesOf : ByteArray → ByteArray
  pfOf : ByteArray × Array ByteArray → Array ByteArray
  canon : ∀ hs e, V.Verdict .sampled ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩ →
    ¬ Brk ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩ →
    ∃ (kd' kp : Game.LiveDrive.Block m) (rds : List (Game.LiveDrive.RoundBlock m)),
      hs.filter (Game.LiveDrive.Tagged m tagOf) =
          (kd'.rels ++ kp.rels ++ rds.flatMap fun r => r.k.rels).map (Game.LiveDrive.rel m) ∧
        kd'.Fits m tagOf .draw spΩ ∧ kp.Fits m tagOf .points spP ∧ rds.length ≤ A.N ∧
        (∀ r ∈ rds, r.Fits Game.LiveDrive.nOfJ Game.LiveDrive.mkJ
          (fun x => (ZkHidden.planZHJ (Reg := A.Reg) A.dj A.y₀ (Game.LiveDrive.drawOS (.subset kd) n x) R₀).length)
          m tagOf spR bytesOf (spΩ.e (Game.LiveDrive.decChunks m _ kd'.bits))) ∧
        proj ⟨θ, F, c, hs.map (Game.recOf releaseOf), e.1, e.2⟩ =
          A.wr (Game.LiveDrive.drawOS (.subset kd) n (spΩ.e (Game.LiveDrive.decChunks m _ kd'.bits))) R₀
            ⟨rootOf kp.b, spP.e (Game.LiveDrive.decChunks m _ kp.bits),
              Game.LiveDrive.roundsOf Game.LiveDrive.nOfJ Game.LiveDrive.mkJ (Game.LiveDrive.appJ A.dg)
                (fun x => (ZkHidden.planZHJ (Reg := A.Reg) A.dj A.y₀ (Game.LiveDrive.drawOS (.subset kd) n x) R₀).length)
                m spR bytesOf (spΩ.e (Game.LiveDrive.decChunks m _ kd'.bits)) rds fun _ _ => [], pfOf e⟩

namespace Drive

variable {V : Vocab d α} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {Brk : Outcome → Prop} {n : ℕ}
  {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg} {A : Flock.Guarantees.LiveAt n D hU kd K₀ desc}
  (Dr : Drive V θ F c σ Brk A)

/-- The play at `σ` is dominated by `A`'s live audit (`dominated_of_canon`). -/
theorem dominated : Game.Dominated (fun o => V.Verdict .sampled o ∧ ZkExec.AcceptsZK A.I (Dr.proj o) ∧ ¬ Brk o)
    (V.game θ F c) σ (fun o => ZkExec.LiveAcceptsZK A.I o) (toVerity A.G) :=
  dominated_of_canon σ Dr.proj A.src A.hA3 (.subset kd) n A.dg (ZkHidden.planZHJ (Reg := A.Reg) A.dj A.y₀) A.N A.wr
    Dr.R₀ Dr.m Dr.tagOf Dr.spΩ Dr.spP Dr.spR Dr.rootOf Dr.bytesOf Dr.pfOf Dr.canon

/-- **The live audit's strategy the reduction builds.** -/
noncomputable def τ : FlockSoundness.Game.Strategy A.G :=
  Pouw.PearlC.FlockTile.stratOfVerity A.G (Classical.choose Dr.dominated)

/-- **The play coupled to `A`'s live audit** at the reduction's strategy. -/
noncomputable def coupling : PlayCoupling V θ F c σ Brk A.I A.G Dr.τ :=
  PlayCoupling.ofDominated Dr.proj Dr.covers Dr.dominated

end Drive

/-- **`SampledDeploy.accept_le` from the reductions**: at each session `A θ F c σ r hr` a reduction (`Drive`, its break
the collision `col` reads off the outcome or the forger's success), and the session's strategy the reduction builds
(`SampledDeploy.τOf := (Dr …).τ`): off the break, sampled proofs' verdict accepts at most as often as the live audit
does. -/
theorem accept_le_of_drives {V : Vocab d α} (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (col : ByteArray → ByteArray → ByteArray → Outcome → Option (Claims.Msg × Claims.Msg))
    (Dr : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r),
      Drive V θ F c σ (fun o => ColEvent (col θ F c) o ∨ V.Forges .sampled θ o) (A θ F c σ r hr)) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), V.Registered θ F →
      Game.prob (fun o => V.Verdict .sampled o ∧ ¬ ColEvent (col θ F c) o ∧ ¬ V.Forges .sampled θ o) (V.game θ F c) σ ≤
        FlockSoundness.Game.prob (fun o => ZkExec.LiveAcceptsZK (A θ F c σ r hr).I o)
          (A θ F c σ r hr).G (Dr θ F c σ r hr).τ :=
  accept_le_of_couplings kd desc A (fun θ F c σ r hr => (Dr θ F c σ r hr).τ) col
    fun θ F c σ r hr _ => (Dr θ F c σ r hr).coupling

end Verity
