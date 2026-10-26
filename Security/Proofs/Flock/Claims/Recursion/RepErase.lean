import Flock.With.SimVerify
import Flock.With.Log

/-!
# A rep reads its rounds as replay does

Replay reads a round through its coin count, its coins and its retained bytes; it reads the round's digest only where
the server kept no bytes (`Round.answersWith`). The element-level decoder (`Record.decodeWithK` at `keep := fun _ =>
true`) drops S12's check that a retained round's bytes are its recorded digest's, so two records the element-level
verdict accepts after the same round messages can differ in their rounds' digests. `Round.erase` drops a retained
round's digest, and two streams whose rounds agree after it are read alike by a rep:
`Flock.With.verifyRepWith_erase`, under any oracle and in any lawful verdict monad that lifts `V` through `<$>`
(`Sim.LiftMap`: `V` itself and the logging monads `StV σ`).

The proof is a simulation (`Flock.With.Sim`) between two runs of one transcript action from transcripts whose rounds
agree after erasure (`Sim.er`). The squeeze reads its round through what erasure keeps (`er_squeezeWith`), every other
transcript action only frames bytes, and `sim_walk` carries the relation through each of the rep's `…With` functions.
-/

namespace Flock

/-- A round as replay reads it: a round that retained its bytes with its digest erased. -/
def Round.erase (r : Round) : Round :=
  match r.bytes with
  | some _ => { r with sha := ByteArray.empty }
  | none => r

theorem Round.erase_erase (r : Round) : r.erase.erase = r.erase := by
  rcases r with ⟨n, sha, coins, _ | b⟩ <;> rfl

theorem Round.erase_n (r : Round) : r.erase.n = r.n := by rcases r with ⟨_, _, _, _ | _⟩ <;> rfl
theorem Round.erase_coins (r : Round) : r.erase.coins = r.coins := by rcases r with ⟨_, _, _, _ | _⟩ <;> rfl
theorem Round.erase_bytes (r : Round) : r.erase.bytes = r.bytes := by rcases r with ⟨_, _, _, _ | _⟩ <;> rfl

/-- A transcript as replay reads it: its rounds erased (`Round.erase`). -/
def Transcript.erase (t : Transcript) : Transcript := { t with rounds := t.rounds.map Round.erase }

theorem Transcript.erase_erase (t : Transcript) : t.erase.erase = t.erase := by
  simp only [Transcript.erase, Array.map_map, Function.comp_def, Round.erase_erase]

theorem Transcript.erase_with {t t' : Transcript} (h : t.erase = t'.erase) (p : ByteArray) (k : Nat) :
    ({ t with pending := p, k := k } : Transcript).erase = ({ t' with pending := p, k := k } : Transcript).erase := by
  have hr : t.rounds.map Round.erase = t'.rounds.map Round.erase := by
    have e := congrArg Transcript.rounds h; exact e
  simp only [Transcript.erase, hr]

theorem Transcript.erase_pending {t t' : Transcript} (h : t.erase = t'.erase) (f : ByteArray → ByteArray) :
    ({ t with pending := f t.pending } : Transcript).erase =
      ({ t' with pending := f t'.pending } : Transcript).erase := by
  have hp : t.pending = t'.pending := by have e := congrArg Transcript.pending h; exact e
  have hk : t.k = t'.k := by have e := congrArg Transcript.k h; exact e
  have e := Transcript.erase_with h (f t.pending) t.k
  rw [hp, hk] at e ⊢
  exact e

namespace With

variable {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- A run's result with its transcript erased. -/
def erP {α : Type} (p : α × Transcript) : α × Transcript := (p.1, p.2.erase)

namespace Sim

/-- **Two runs of one transcript action from transcripts replay reads alike end alike**, up to erasure. -/
def er (M : Type → Type) [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M] :
    Sim (TMW M) (TMW M) where
  R x y := ∀ t t', t.erase = t'.erase → erP <$> x.run t = erP <$> y.run t'
  pure a t t' h := by
    simp only [StateT.run_pure, map_pure, erP, h]
  bind {α β x y f g} hxy hfg t t' h := by
    simp only [StateT.run_bind, map_bind]
    have hF : ∀ p : α × Transcript, erP <$> (f p.1).run p.2 = erP <$> (g (erP p).1).run (erP p).2 :=
      fun p => hfg p.1 p.2 _ (Transcript.erase_erase p.2).symm
    have hG : ∀ p : α × Transcript, erP <$> (g p.1).run p.2 = erP <$> (g (erP p).1).run (erP p).2 := fun p =>
      (hfg p.1 p.2 p.2 rfl).symm.trans (hfg p.1 p.2 _ (Transcript.erase_erase p.2).symm)
    calc x.run t >>= (fun p => erP <$> (f p.1).run p.2)
        = x.run t >>= (fun p => erP <$> (g (erP p).1).run (erP p).2) := by
          congr 1; funext p; exact hF p
      _ = (erP <$> x.run t) >>= fun p => erP <$> (g p.1).run p.2 := by rw [bind_map_left]
      _ = (erP <$> y.run t') >>= fun p => erP <$> (g p.1).run p.2 := by rw [hxy t t' h]
      _ = y.run t' >>= fun p => erP <$> (g p.1).run p.2 := by
          rw [bind_map_left]; congr 1; funext p; exact (hG p).symm
  throw e t t' h := by
    show erP <$> ((MonadExcept.throw e : M _) >>= fun a => Pure.pure (a, t)) =
      erP <$> ((MonadExcept.throw e : M _) >>= fun a => Pure.pure (a, t'))
    simp only [map_bind, map_pure, erP, h]
  lift v t t' h := by
    show erP <$> ((monadLift v : M _) >>= fun a => Pure.pure (a, t)) =
      erP <$> ((monadLift v : M _) >>= fun a => Pure.pure (a, t'))
    simp only [map_bind, map_pure, erP, h]

theorem er_monadLift {α : Type} (m : M α) : (er M).R (monadLift m : TMW M α) (monadLift m) := by
  intro t t' h
  show erP <$> (m >>= fun a => Pure.pure (a, t)) = erP <$> (m >>= fun a => Pure.pure (a, t'))
  simp only [map_bind, map_pure, erP, h]

theorem er_monadLift2 {σ α : Type} (m : M α) :
    ((er M).st σ).R (monadLift m : StateT σ (TMW M) α) (monadLift m) :=
  (er M).lift_st (er_monadLift m)

theorem er_modifyPending (f : Transcript → ByteArray → ByteArray) (hf : ∀ t t', t.erase = t'.erase → f t = f t') :
    (er M).R (_root_.modify fun t => { t with pending := f t t.pending } : TMW M PUnit)
      (_root_.modify fun t => { t with pending := f t t.pending }) := by
  intro t t' h
  show erP <$> Pure.pure (PUnit.unit, _) = erP <$> Pure.pure (PUnit.unit, _)
  simp only [map_pure, erP, hf t t' h]
  rw [Transcript.erase_pending h]

/-- How `M` lifts a `V` result: through `<$>`. -/
def LiftMap (M : Type → Type) [Monad M] [MonadLiftT V M] : Prop :=
  ∀ {α β : Type} (f : α → β) (v : V α), f <$> (monadLift v : M α) = monadLift (f <$> v)

theorem er_liftTM (hl : LiftMap M) {α : Type} {x : Transcript.TM α} (h : (er V).R x x) :
    (er M).R (Flock.liftTM x : TMW M α) (Flock.liftTM x) := by
  intro t t' ht
  show erP <$> (monadLift (x t) : M _) = erP <$> (monadLift (x t') : M _)
  rw [hl, hl]
  exact congrArg _ (h t t' ht)

theorem er_liftLM (hl : LiftMap M) {α : Type} {x : LM α} (h : ((er V).st LState).R x x) :
    ((er M).st LState).R (liftLM x : LMW M α) (liftLM x) := by
  intro s t t' ht
  show erP <$> (monadLift (x s t) : M _) = erP <$> (monadLift (x s t') : M _)
  rw [hl, hl]
  exact congrArg _ (h s t t' ht)

theorem er_liftW {α : Type} {x y : TMW M α} (h : (er M).R x y) : ((er M).st LState).R (liftW x) (liftW y) :=
  (er M).lift_st h

theorem er_ligLift {α : Type} {x : Transcript.TM α} (h : (er V).R x x) :
    ((er V).st LState).R (Flock.lift x) (Flock.lift x) :=
  (er V).lift_st h

theorem er_label (l : String) : (er V).R (Transcript.label l) (Transcript.label l) :=
  er_modifyPending (fun _ p => p ++ Transcript.header 2 0 l.toUTF8.size ++ Transcript.padded l.toUTF8)
    (fun _ _ _ => rfl)

theorem er_absorbF (v : F128) : (er V).R (Transcript.absorbF v) (Transcript.absorbF v) :=
  er_modifyPending (fun _ p => F128.pushBytes (p ++ Transcript.header 3 1 1) v) (fun _ _ _ => rfl)

theorem er_absorbSlice (vs : Array F128) : (er V).R (Transcript.absorbSlice vs) (Transcript.absorbSlice vs) :=
  er_modifyPending (fun _ p => vs.foldl F128.pushBytes (p ++ Transcript.header 3 2 vs.size)) (fun _ _ _ => rfl)

theorem er_absorbBytes (b : ByteArray) : (er V).R (Transcript.absorbBytes b) (Transcript.absorbBytes b) :=
  er_modifyPending (fun _ p => p ++ Transcript.header 5 0 b.size ++ Transcript.padded b) (fun _ _ _ => rfl)

theorem er_absorbK (v : F256) : (er V).R (Transcript.absorbK v) (Transcript.absorbK v) := er_absorbSlice _

theorem er_check (b : Bool) (why : String) : (er V).R (check b why) (check b why) := by
  unfold check
  split
  · exact (er V).pure _
  · exact (er V).throw _

theorem er_pow (n : UInt64) : (er V).R (Transcript.pow n) (Transcript.pow n) := er_absorbBytes _

omit [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M] in
theorem erase_answersWith (hs : Flock.Hashes M) {r r' : Round} (h : r.erase = r'.erase) (c : ByteArray) :
    r.answersWith hs c = r'.answersWith hs c := by
  rcases r with ⟨n, sha, coins, _ | b⟩ <;> rcases r' with ⟨n', sha', coins', _ | b'⟩ <;>
    simp only [Round.erase, Round.mk.injEq, reduceCtorEq, and_false] at h
  · rw [h.1, h.2.1, h.2.2.1]
  · obtain ⟨-, -, -, hb⟩ := h
    simp only [Round.answersWith, Option.some.injEq] at hb ⊢
    rw [hb]

theorem er_squeezeWith (hs : Flock.Hashes M) (kind : UInt8) (n : Nat) :
    (er M).R (Transcript.squeezeWith hs kind n) (Transcript.squeezeWith hs kind n) := by
  intro t t' h
  have hp : t.pending = t'.pending := by have e := congrArg Transcript.pending h; exact e
  have hk : t.k = t'.k := by have e := congrArg Transcript.k h; exact e
  have hr : t.rounds.map Round.erase = t'.rounds.map Round.erase := by
    have e := congrArg Transcript.rounds h; exact e
  have hq : (t.rounds[t.k]?).map Round.erase = (t'.rounds[t'.k]?).map Round.erase := by
    rw [← Array.getElem?_map, ← Array.getElem?_map, hr, hk]
  show erP <$> Transcript.squeezeWith hs kind n t = erP <$> Transcript.squeezeWith hs kind n t'
  unfold Transcript.squeezeWith Transcript.framed
  rw [hp]
  split
  · simp only [map_pure, erP, hk, Transcript.erase_with h]
  · rw [hk] at hq ⊢
    dsimp only
    rcases h₁ : t.rounds[t'.k]? with _ | r <;> rcases h₂ : t'.rounds[t'.k]? with _ | r' <;> rw [h₁, h₂] at hq <;>
      simp only [Option.map_none, Option.map_some, reduceCtorEq, Option.some.injEq] at hq
    all_goals try rfl
    · dsimp only
      have hn : r.n = r'.n := (Round.erase_n r).symm.trans ((congrArg Round.n hq).trans (Round.erase_n r'))
      have hc : r.coins = r'.coins :=
        (Round.erase_coins r).symm.trans ((congrArg Round.coins hq).trans (Round.erase_coins r'))
      have hb : r.bytes = r'.bytes :=
        (Round.erase_bytes r).symm.trans ((congrArg Round.bytes hq).trans (Round.erase_bytes r'))
      rw [hn, hb, hc, erase_answersWith hs hq]
      split
      · rfl
      · simp only [map_bind]
        congr 1
        funext ok
        split
        · rfl
        · simp only [map_pure, erP, Transcript.erase_with h]

theorem liftMap_V : LiftMap V := fun _ _ => rfl

theorem liftMap_StV (σ : Type) : LiftMap (StV σ) := by
  intro α β f v
  cases v <;> rfl

end Sim

open Transcript Sim

variable (hl : LiftMap M) (hs : Flock.Hashes M)
include hl

omit hl in
theorem er_sampleWith : (er M).R (sampleWith hs) (sampleWith hs) := by
  unfold sampleWith
  sim_walk [er_squeezeWith hs]

omit hl in
theorem er_sampleVecWith (n : Nat) : (er M).R (sampleVecWith hs n) (sampleVecWith hs n) :=
  er_squeezeWith hs 2 n

omit hl in
theorem er_sampleKWith : (er M).R (sampleKWith hs) (sampleKWith hs) := by
  unfold sampleKWith
  sim_walk [er_squeezeWith hs]

theorem er_bindAndZerocheckWith (m : Nat) (digest cap : ByteArray) (p : ZerocheckProof) :
    (er M).R (bindAndZerocheckWith hs m digest cap p) (bindAndZerocheckWith hs m digest cap p) := by
  unfold bindAndZerocheckWith
  sim_walk [er_sampleWith hs, er_sampleVecWith hs, er_liftTM hl, er_label, er_absorbF, er_absorbSlice,
    er_absorbBytes, er_check, er_monadLift]

theorem er_lincheckWith (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) :
    (er M).R (lincheckWith hs c zc p) (lincheckWith hs c zc p) := by
  unfold lincheckWith
  sim_walk [er_sampleWith hs, er_liftTM hl, er_label, er_absorbF, er_absorbSlice, er_absorbBytes, er_check,
    er_monadLift]

theorem er_ringSwitchWith (m : Nat) (claims : Array Claim) (rs : Array (Array F128)) :
    (er M).R (ringSwitchWith hs m claims rs) (ringSwitchWith hs m claims rs) := by
  unfold ringSwitchWith
  sim_walk [er_sampleVecWith hs, er_liftTM hl, er_label, er_absorbF, er_absorbSlice, er_absorbBytes, er_check,
    er_monadLift]

omit hl in
theorem er_nextMsg (p : LigeritoProof) : ((er V).st LState).R (nextMsg p) (nextMsg p) := by
  unfold nextMsg
  sim_walk [er_ligLift, er_absorbK]

omit hl in
theorem er_nextOod (p : LigeritoProof) : ((er V).st LState).R (nextOod p) (nextOod p) := by
  unfold nextOod
  sim_walk

omit hl in
theorem er_nonceFrom (xs : Array UInt64) (i : Nat) (what : String) :
    ((er V).st LState).R (nonceFrom xs i what) (nonceFrom xs i what) := by
  unfold nonceFrom
  sim_walk

omit hl in
theorem er_claimPow (p : LigeritoProof) : ((er V).st LState).R (claimPow p) (claimPow p) := by
  unfold claimPow
  sim_walk [er_nonceFrom, er_ligLift, er_pow]

theorem er_challengeWith (p : LigeritoProof) : ((er M).st LState).R (challengeWith hs p) (challengeWith hs p) := by
  unfold challengeWith
  sim_walk [er_liftW, er_liftLM hl, er_sampleKWith hs, er_nextMsg]

theorem er_oodIntroWith (p : LigeritoProof) (dim split : Nat) :
    ((er M).st LState).R (oodIntroWith hs p dim split) (oodIntroWith hs p dim split) := by
  unfold oodIntroWith
  sim_walk [er_liftW, er_liftLM hl, er_sampleWith hs, er_sampleVecWith hs, er_nextMsg, er_nextOod,
    er_claimPow, er_ligLift, er_absorbF]

theorem er_queryLevelWith (ms : MerkleScheme) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray)
    (rows : Array (Array F128)) (sibs salts : Array ByteArray) (fold : Array F256) :
    ((er M).st LState).R (queryLevelWith hs ms p lv cap rows sibs salts fold)
      (queryLevelWith hs ms p lv cap rows sibs salts fold) := by
  unfold queryLevelWith
  sim_walk [er_liftW, er_liftLM hl, er_sampleVecWith hs, er_nonceFrom, er_ligLift, er_pow, er_monadLift2]

theorem er_ligeritoWith (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → Nat → Array F256) :
    (er M).R (ligeritoWith hs ms sch T cap p basisAt) (ligeritoWith hs ms sch T cap p basisAt) := by
  unfold ligeritoWith
  sim_walk [er_liftW, er_liftLM hl, er_sampleWith hs, er_sampleVecWith hs, er_challengeWith hl hs,
    er_oodIntroWith hl hs, er_queryLevelWith hl hs, er_nextMsg, er_nextOod, er_claimPow, er_ligLift, er_pow,
    er_liftTM hl, er_label, er_absorbF, er_absorbSlice, er_absorbBytes, er_check, er_monadLift, er_monadLift2]

omit hl in
/-- Two runs of an action from transcripts replay reads alike, bound to a continuation that reads its transcript as
replay does, are one computation. -/
theorem er_bind {α β : Type} {x : TMW M α} (hx : (er M).R x x) {t t' : Transcript} (ht : t.erase = t'.erase)
    (f : α × Transcript → M β) (hf : ∀ p, f p = f (erP p)) : x.run t >>= f = x.run t' >>= f := by
  have e : ∀ s, x.run s >>= f = (erP <$> x.run s) >>= f := fun s => by
    rw [bind_map_left]; exact congrArg _ (funext hf)
  rw [e t, e t', hx t t' ht]

/-- **A rep reads its session through its stream's rounds as replay reads them, the points and its table's
`root_B`** (`verifyRep_congr`'s, with the rounds compared after `Round.erase`). -/
theorem verifyRepWith_erase (spec : SessionSpec) (st : Setup) (sch : Schedule) {sess sess' : Session} {i r : Nat}
    (bytes : ByteArray) {y y' : StreamRec}
    (hy : sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some y)
    (hy' : sess'.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some y')
    (hr : y.rounds.map Round.erase = y'.rounds.map Round.erase) (hP : sess.points = sess'.points)
    (hB : sess.rootB[i]! = sess'.rootB[i]!) :
    verifyRepWith hs spec st sch sess i r bytes = verifyRepWith hs spec st sch sess' i r bytes := by
  have hsz : y.rounds.size = y'.rounds.size := by simpa using congrArg Array.size hr
  unfold verifyRepWith
  dsimp only
  rw [hy, hy']
  dsimp only
  rw [hP, hB, hsz]
  have ht : (init y.rounds (spec.domain spec.tables[i]! r).toUTF8).erase =
      (init y'.rounds (spec.domain spec.tables[i]! r).toUTF8).erase := by
    simp only [Transcript.init, Transcript.erase, hr]
  refine congrArg (_ >>= ·) (funext fun extras => ?_)
  refine congrArg (_ >>= ·) (funext fun p => ?_)
  split
  · refine congrArg (_ >>= ·) (funext fun _ => ?_)
    refine er_bind ?_ ht _ fun _ => rfl
    sim_walk [er_bindAndZerocheckWith hl hs, er_lincheckWith hl hs, er_ringSwitchWith hl hs, er_ligeritoWith hl hs]
  · refine er_bind ?_ ht _ fun _ => rfl
    sim_walk [er_bindAndZerocheckWith hl hs, er_lincheckWith hl hs, er_ringSwitchWith hl hs, er_ligeritoWith hl hs]

end With

end Flock
