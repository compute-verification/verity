import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.HoloValues.Shape
import Proofs.Flock.C3.HoloValues.ZcLc
import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.C3.Compose.Plays
import Proofs.Flock.Soundness.Refine.Paths
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Refine.Table

/-!
# n2's read fact: each rep's accepting run from the tape

`RepsRead`: at every accepted outcome, the inner transcript's steps past the cap's and the points' begin with rep 0's
steps and then rep 1's, in the reps' interleaving (`firstSteps`), and each rep's steps are an accepting run of the
executable's holo rep (`RepUnit`): values V*'s sessions hold that meet E's preconditions (`AlgebraHolds`, `AccHolds`,
the proof's zerocheck and lincheck messages, the circuit fold at `verifierRows`' points), on a record from round 0
whose rounds' coins are the values' and which answers the rep's frames, through ring switching and Ligerito on a
`fast100` schedule that is the model's; the steps take the kinds of `holoRepC`'s shape (`holoRepShape`), their
messages are the proof's (`holoRepMsgs`) and their coin steps the rounds the program consumes but `z_partial`'s; the
proof's level-0 cap is the run's, and the run's openings at the rep are the proof's rows (`openRow`). It reads the
tape and the run, not the decoder of V*'s messages.

`hReps_of_read`: `HReps` from `RepsRead`, at any family whose run decodes the reps from the tape as `decodeRun` does
(`HRepStages`, which the honest family has: `hRepStages_honest`). Each unit's program runs (`holoProg_of_zclc`), it
refines to `holoRepC` (`holoRep_refines`), the run is a play on the unit's steps (`playsT_of_spine`) and its openings
pass the plain leaf's path checks (`opensOK_of`); two games play on at most one pair of prefixes of a tape in the
reps' interleaving (`Read.repsOf_of_playsT`), so the reps the run decodes are those plays.
-/

namespace FlockC3.HoloValues

noncomputable section

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Read

open FlockSoundness FlockSoundness.Game
open FlockVBridge.InnerAccepts (firstSteps)
open FlockC3.Compose (repsOf playsT_prefix)

theorem firstSteps_prefix : ∀ {p p' : List Step}, firstSteps p <+: firstSteps p' → p <+: p'
  | [], _, _ => List.nil_prefix
  | s :: p, [], h => by
      simp only [firstSteps, List.flatMap_cons, List.flatMap_nil, List.cons_append, List.nil_append] at h
      exact absurd (List.prefix_nil.mp h) (by simp)
  | s :: p, s' :: p', h => by
      simp only [firstSteps, List.flatMap_cons, List.cons_append, List.nil_append, List.cons_prefix_cons] at h
      obtain ⟨-, rfl, h⟩ := h
      exact List.cons_prefix_cons.mpr ⟨rfl, firstSteps_prefix h⟩

theorem prefix_drop {a b l : List Step} (h : a ++ b <+: l) : b <+: l.drop a.length := by
  obtain ⟨t, rfl⟩ := h
  rw [List.append_assoc, List.drop_left]
  exact List.prefix_append _ _

/-- **Two games play on at most one pair of prefixes of a tape** in the reps' interleaving, to one pair of outputs. -/
theorem reps_unique {α β : Type} {g₀ : Game α} {g₁ : Game β} {l p₀ p₁ q₀ q₁ : List Step} {a a' : α} {b b' : β}
    (h₀ : PlaysT g₀ p₀ a) (h₁ : PlaysT g₁ p₁ b) (hp : firstSteps p₀ ++ p₁ <+: l) (hq : firstSteps q₀ ++ q₁ <+: l)
    (hq₀ : PlaysT g₀ q₀ a') (hq₁ : PlaysT g₁ q₁ b') : q₀ = p₀ ∧ a' = a ∧ q₁ = p₁ ∧ b' = b := by
  have hf : firstSteps q₀ <+: l := (List.prefix_append _ _).trans hq
  have hf' : firstSteps p₀ <+: l := (List.prefix_append _ _).trans hp
  obtain ⟨rfl, rfl⟩ : q₀ = p₀ ∧ a' = a := by
    rcases List.prefix_or_prefix_of_prefix hf hf' with h | h
    · exact playsT_prefix h₀ hq₀ (List.prefix_refl p₀) (firstSteps_prefix h)
    · exact playsT_prefix h₀ hq₀ (firstSteps_prefix h) (List.prefix_refl q₀)
  obtain ⟨rfl, rfl⟩ := playsT_prefix h₁ hq₁ (prefix_drop hp) (prefix_drop hq)
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- **The reps' decode is their plays** on any prefix of the tape they play on in the reps' interleaving. -/
theorem repsOf_of_playsT {α β : Type} {g₀ : Game α} {g₁ : Game β} {l p₀ p₁ : List Step} {a : α} {b : β}
    (d : List Step × α × List Step × β) (h₀ : PlaysT g₀ p₀ a) (h₁ : PlaysT g₁ p₁ b)
    (hp : firstSteps p₀ ++ p₁ <+: l) : repsOf g₀ g₁ l d = (p₀, a, p₁, b) := by
  have hex : ∃ q : List Step × α × List Step × β,
      firstSteps q.1 ++ q.2.2.1 <+: l ∧ PlaysT g₀ q.1 q.2.1 ∧ PlaysT g₁ q.2.2.1 q.2.2.2 := ⟨(p₀, a, p₁, b), hp, h₀, h₁⟩
  unfold repsOf
  rw [dite_eq_left hex]
  obtain ⟨e₁, e₂, e₃, e₄⟩ := reps_unique h₀ h₁ hp hex.choose_spec.1 hex.choose_spec.2.1 hex.choose_spec.2.2
  exact Prod.ext e₁ (Prod.ext e₂ (Prod.ext e₃ e₄))

end Read

/-! ## One rep's unit -/

section Unit

open Flock FlockSoundness FlockSoundness.Refine
open Flock.RecAlgebra (AnswersL ZcProofOf LcProofOf decoded valOf verifierRows hOfW)
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)
open FlockVBridge.Runs (rhoOf tsOf)
open FlockSoundness.Discharge.Exec (kindOf)
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Model (holoRepC OpensOKL)

/-- **One rep's accepting run of the executable** at rep `r`, on the tape steps `ts`, for the model's statement `S`,
schedule `sch₀` and extra claims `extra`, the level-0 cap `cap₀` and the openings `os`: values `d` at a shape `sh`
meeting E's preconditions, the proof's zerocheck and lincheck messages and the circuit fold at `verifierRows`' points
(`holoProg_of_zclc`'s), on a record `t` from round 0 whose rounds' coins are `d`'s and which answers the rep's frames
with hidden units (`HoloLcAnswered`), then ring switching from past `r'`'s round and Ligerito to `t₄`, at a setup
whose `m`, `k_log` and circuit fold are `S`'s, on a `fast100` schedule `sch` whose model schedule is `sch₀`, with the
extra claims `extras` the model's claim by claim. The steps take the kinds of `holoRepC`'s shape, their messages are
the proof's and `h` (`holoRepMsgs`), their coin steps the rounds the program consumes but `z_partial`'s; the proof's
level-0 cap is `cap₀`, and the openings at rep `r` are the proof's rows and siblings (`openRow`). -/
structure RepUnit (S : Model.Statement) (sch₀ : Accounting.Schedule) (extra : List (Model.Claim 𝔽 S.m))
    (cap₀ : Model.Cap Dig) (os : Model.Opens 𝔽 Dig) (r : ℕ) (ts : List Game.Step) : Type where
  sh : Flock.Shape
  d : RepData
  st : Flock.Setup
  msId : String
  sch : Flock.Schedule
  p : Flock.Proof
  extras : Array Flock.Claim
  t : Flock.Transcript
  claims : Array Flock.Claim
  rpp : ℕ → Array Flock.F128
  gamma : Array Flock.F128
  t₄ : Flock.Transcript
  alg : AlgebraHolds sh d
  acc : AccHolds sh d
  coins : d.rounds = t.rounds.map (·.coins)
  zp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) p.zerocheck
  lp : LcProofOf (decoded sh d.w d.comb d.cons) p.lincheck
  foldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
      .ok d.comb
  m : st.m = sh.m
  kL : st.fold.kLog = sh.kLog
  mS : st.m = S.m
  kS : st.kLog = S.kLog
  fold : FoldRealizes st.fold S
  fast : Flock.fast100 st.m = some sch
  sched : schedOf sch = sch₀
  merkle : st.merkle = Flock.MerkleScheme.sha512 msId
  extrasLen : extras.size = extra.length
  extrasRel : ∀ (j : ℕ) (hj : j < extra.length), ClaimRel S.m extras[j]! extra[j]
  k0 : t.k = 0
  wf : RoundsWF t.rounds
  n₀ : (t.rounds[t.k]!).n = 6
  n₁ : (t.rounds[t.k + 1]!).n = st.m - 13
  n₂ : (t.rounds[t.k + 2]!).n = 1
  nr : ∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1
  a₀ : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
    .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6])
  a₁ : AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)])
  a₂ : AnswersL (t.rounds[t.k + 2]!)
    (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1])
  ar : ∀ i < st.m - 6, AnswersL (t.rounds[t.k + 3 + i]!)
    (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1])
  lc : HoloLcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
    ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck (hOfW sh d.w)
  cl : claims = Flock.abcClaims st.m st.kLog
    (zcOf p.zerocheck (pinned ++ (t.rounds[t.k + 1]!).coins) ((t.rounds[t.k + 2]!).coins.getD 0 0) (rhoOf t st.m))
    (lcOf p.lincheck ((t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1]!).coins.getD 0 0)
      (tsOf t (t.k + 3 + (st.m - 6)) st.fold.kLog)) ++ extras
  rs : p.ringSwitches.size = claims.size
  rsShape : ∀ k < claims.size,
    p.ringSwitches[k]!.size = 128 ∧ claims[k]!.skip.size = 64 ∧ claims[k]!.x.size = st.m - 6
  rsVal : ∀ k < claims.size, dot (wOf claims[k]!) p.ringSwitches[k]! = claims[k]!.value
  rsRounds : RsRounds ⟨t.rounds, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2, ByteArray.empty⟩
    p.ringSwitches claims.size rpp gamma
  lg : Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
    (Flock.basis (rsSws claims rpp gamma))
    ⟨t.rounds, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 2 + claims.size + 1, ByteArray.empty⟩ = .ok ((), t₄)
  kinds : ts.map kindOf <+: holoRepShape 𝔽 𝕂 Dig S.m S.kLog extra.length sch₀
  msgs : ts.filterMap Sum.getLeft? = holoRepMsgs st sch (2 + extra.length) p (hOfW sh d.w) capDig
  coinSteps : ts.filterMap Sum.getRight? =
    (consumed t.rounds t.k (t.k + 3 + (st.m - 6) + 2 + (S.kLog - 6)) ++
      consumed t.rounds (t.k + 3 + (st.m - 6) + 2 + (S.kLog - 6) + 1) t₄.k).map (coinArr fOf)
  cap : capDig p.cap = cap₀
  opens : ∀ ℓ j, os r ℓ j = openRow p.ligerito sch ℓ j

/-- **A rep's unit plays `holoRepC` to an output** on its steps, whose result accepts the openings and whose level-0
paths verify against `cap₀` at any salts. -/
theorem rep_of_unit {S : Model.Statement} {sch₀ : Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
    {cap₀ : Model.Cap Dig} {os : Model.Opens 𝔽 Dig} {r : ℕ} {ts : List Game.Step} (hm : 13 ≤ S.m)
    (U : RepUnit S sch₀ extra cap₀ os r ts) :
    ∃ zc lc pend, Game.PlaysT (holoRepC (D := Dig) execArith r S sch₀ hm extra) ts (some (zc, lc), pend) ∧
      pend.verdict os = true ∧ ∀ sl, OpensOKL H512 enc512 leafPlain cap₀ r pend os sl := by
  have h22 := exec_fast100_ge U.fast
  obtain ⟨hprog, zc₁, t₁, h₁, hk₁⟩ := holoProg_of_zclc U.sh U.d U.alg U.acc U.st U.sch U.p U.extras U.t U.k0 U.coins
    U.wf U.m (by omega) U.kL U.zp U.lp U.foldPts U.n₀ U.n₁ U.n₂ U.nr U.a₀ U.a₁ U.a₂ U.ar U.lc U.cl U.rs U.rsShape
    U.rsVal U.rsRounds U.lg
  have hdims := fast100_dims U.st.m U.sch U.fast
  have hlv : ∀ i < U.sch.levels.size, LevelSeam (U.sch.levels.getD i default) := fun i hi =>
    fast100_level U.st.m U.sch U.fast _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨-, kz, -, -, -, hkz, zc, lc, pend, hpl, -, hcaps, hreqs, hver⟩ := holoRep_refines U.st S hm
    (by have := U.mS; omega) U.sch U.mS U.kS U.fold extra U.extras U.extrasLen U.extrasRel hdims.1 hlv
    (fun i hi => fast100_alpha U.st.m U.sch U.fast _
      (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi))
    (hdims.2.trans U.mS) U.p (hOfW U.sh U.d.w) r U.t U.t₄ U.wf hprog capDig os
    (fun ℓ _ j _ => show (os r ℓ j).1 = (openRow U.p.ligerito U.sch ℓ j).1 from congrArg Prod.fst (U.opens ℓ j))
  have hkz' := hkz zc₁ t₁ h₁
  rw [hk₁] at hkz'
  subst hkz'
  rw [U.sched] at hpl
  refine ⟨zc, lc, pend, playsT_of_spine (holoRepC_spine (D := Dig) execArith r S sch₀ hm extra) hpl U.kinds U.msgs
    U.coinSteps, hver, fun sl => ?_⟩
  have hreqs' : ∀ req ∈ pend.reqs, ReqOK (Flock.MerkleScheme.sha512 U.msId) U.sch U.p.cap U.p.ligerito req := by
    rw [← U.merkle]; exact hreqs
  rw [← U.cap]
  exact opensOK_of U.msId U.sch hlv U.p.cap U.p.ligerito r pend os hcaps hreqs' U.opens

end Unit

/-! ## The read fact -/

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockVBridge.InnerAccepts (firstSteps)

/-- Rep `r`'s game at the run `u`, as `HReps` and `decodeRun` take it. -/
abbrev repAt (e : ExecHolo) (u : C3Run e) (r : ℕ) :=
  holoRepC (D := Dig) execArith r (e.cls.pubStatement e.regs) e.sch e.cls.hm
    (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts)

/-- The two reps' decode at the run `u` from the tape `l`, past the cap's and the points' steps, as `decodeRun` takes
it. -/
def repsAt (e : ExecHolo) (l : List Step) (u : C3Run e) :=
  repsOf (repAt e u 0) (repAt e u 1) (l.drop 2) ([], (none, PendL.reject), [], (none, PendL.reject))

/-- **The read fact at a tape** `l` and a run `u`: past the cap's and the points' steps, `l` begins with rep 0's steps
and then rep 1's in the reps' interleaving, each an accepting run of the executable's holo rep (`RepUnit`) at the
run's statement, schedule, extra claims at the run's points, level-0 cap and openings. -/
def RepsReadAt (e : ExecHolo) (l : List Step) (u : C3Run e) : Prop :=
  ∃ ts₀ ts₁ : List Step, firstSteps ts₀ ++ ts₁ <+: l.drop 2 ∧
    Nonempty (RepUnit (e.cls.pubStatement e.regs) e.sch
      (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts) u.cap₀ u.os.1 0 ts₀) ∧
    Nonempty (RepUnit (e.cls.pubStatement e.regs) e.sch
      (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts) u.cap₀ u.os.1 1 ts₁)

/-- **The run's reps are the tape's**, as `decodeRun` decodes them: each rep's steps and pending result are the
reps' decode's (`repsAt`), and each rep's output is the decode's where it is one. -/
def RepStages (e : ExecHolo) (l : List Step) (u : C3Run e) : Prop :=
  u.ts 0 = (repsAt e l u).1 ∧ u.ts 1 = (repsAt e l u).2.2.1 ∧ u.pend 0 = (repsAt e l u).2.1.2 ∧
    u.pend 1 = (repsAt e l u).2.2.2.2 ∧ (∀ x, (repsAt e l u).2.1.1 = some x → u.rep 0 = x) ∧
    ∀ x, (repsAt e l u).2.2.2.1 = some x → u.rep 1 = x

theorem repStages_decodeRun (e : ExecHolo) (l : List Step) : RepStages e l (decodeRun e l) := by
  refine ⟨rfl, rfl, rfl, rfl, fun x hx => ?_, fun x hx => ?_⟩
  · change (repsAt e l (decodeRun e l)).2.1.1.getD ((c3Default e).rep 0) = x
    rw [hx, Option.getD_some]
  · change (repsAt e l (decodeRun e l)).2.2.2.1.getD ((c3Default e).rep 1) = x
    rw [hx, Option.getD_some]

theorem rep_at {e : ExecHolo} {u : C3Run e} {i : Fin 2} {ts : List Step} {zl} {pend : PendL GF128 Dig}
    (hts : u.ts i = ts) (hp : u.pend i = pend) (hr : u.rep i = zl) (hT : PlaysT (repAt e u i) ts (some zl, pend))
    (hv : pend.verdict u.os.1 = true) (ho : OpensOKL H512 enc512 leafPlain u.cap₀ i pend u.os.1 u.os.2) :
    PlaysT (repAt e u i) (u.ts i) (some (u.rep i), u.pend i) ∧ (u.pend i).verdict u.os.1 = true ∧
      OpensOKL H512 enc512 leafPlain u.cap₀ i (u.pend i) u.os.1 u.os.2 := by
  subst hts hp hr
  exact ⟨hT, hv, ho⟩

/-- **Each rep plays `holoRepC` to its decoded output**, whose result accepts the openings and whose level-0 paths
verify against the run's cap, at a tape whose reps' units the read fact gives. -/
theorem reps_of_readAt {e : ExecHolo} {l : List Step} {u : C3Run e} (hs : RepStages e l u)
    (hR : RepsReadAt e l u) :
    ∀ i : Fin 2, PlaysT (repAt e u i) (u.ts i) (some (u.rep i), u.pend i) ∧ (u.pend i).verdict u.os.1 = true ∧
      OpensOKL H512 enc512 leafPlain u.cap₀ i (u.pend i) u.os.1 u.os.2 := by
  obtain ⟨hts₀, hts₁, hp₀, hp₁, hr₀, hr₁⟩ := hs
  obtain ⟨ts₀, ts₁, hpre, ⟨U₀⟩, ⟨U₁⟩⟩ := hR
  obtain ⟨zc₀, lc₀, pend₀, hT₀, hv₀, ho₀⟩ := rep_of_unit e.cls.hm U₀
  obtain ⟨zc₁, lc₁, pend₁, hT₁, hv₁, ho₁⟩ := rep_of_unit e.cls.hm U₁
  have hrs : repsAt e l u = (ts₀, (some (zc₀, lc₀), pend₀), ts₁, (some (zc₁, lc₁), pend₁)) :=
    Read.repsOf_of_playsT _ hT₀ hT₁ hpre
  rw [hrs] at hts₀ hts₁ hp₀ hp₁ hr₀ hr₁
  exact Fin.forall_fin_two.mpr ⟨rep_at hts₀ hp₀ (hr₀ _ rfl) hT₀ hv₀ (ho₀ _), rep_at hts₁ hp₁ (hr₁ _ rfl) hT₁ hv₁ (ho₁ _)⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n2's read fact) Each rep's accepting run is on the inner transcript**: at every accepted outcome, the inner
transcript's steps (V*'s layers' messages zipped with the rounds' coins) hold, past the cap's and the points' steps,
rep 0's and then rep 1's accepting runs of the executable's holo rep in the reps' interleaving (`RepsReadAt`). -/
def RepsRead (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    RepsReadAt (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

/-- **The family's run decodes the reps from the inner transcript's steps** as `decodeRun` does, at every accepted
outcome. -/
def HRepStages (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    RepStages (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

/-- **(n2) Each rep plays `holoRepC` to its decoded output, from the read fact**, at any family whose run decodes the
reps as `decodeRun` does. -/
theorem hReps_of_read {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hst : HRepStages F dg N k Rw) (hread : RepsRead F dg N k Rw) : HReps F dg N k Rw := by
  intro R ω t R₂ τ o hacc
  exact reps_of_readAt (hst R ω t R₂ τ o hacc) (hread R ω t R₂ τ o hacc)

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {tr : Lean.Json} {record : String} {reg : Lean.Json}

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zs own sel bc dec tr record reg

/-- The honest family's run is `decodeRun` of the inner transcript's steps, at any decoder `dec`. -/
theorem hRepStages_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} : HRepStages 𝓗 dg N k Rw := by
  intro _ _ _ _ _ _ _
  exact repStages_decodeRun (holoAt st hp) _

/-- **(n2) at the honest family, from the read fact.** -/
theorem hReps_honest_of_read {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hread : RepsRead 𝓗 dg N k Rw) :
    HReps 𝓗 dg N k Rw :=
  hReps_of_read hRepStages_honest hread

end

end FlockC3.HoloValues
