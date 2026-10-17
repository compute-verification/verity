import Proofs.Flock.Claims.KS.PlainEvent
import Proofs.Flock.Claims.KS.LiveMix

/-!
# The first bound of the plain mixture, from a simulation

`PlainLiveWrPAt`'s first clause asks that `V false`'s accepting runs with no column off `C` be no likelier than the
average over a tape of the live session's `plainVerdict`. `plain_first_of_sim` gives that bound from two things about
the deployed runs, each deterministic:

* a simulation (`PlainSim`): a prover `P'` of the live session at the tape `(ω, u)` and the live coins `r'`, and coins
  `Φ (u, r')` of `V false` with the law of uniform coins, such that wherever `V false` accepts at `Φ (u, r')`, the live
  outcome of `P'` is the run's (`viewOf`);
* the record's draw (`RecordDrawVerbatim`): on `V false`'s accepting runs the record carries the statement's draw
  verbatim, which `plainVerdict_of_run` reads.

Then `Pr_le_liveMix` moves the bound to the mixture of `P'`'s strategies, at the root of the prover's first message.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Custody Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **`P`'s accepting runs carry the statement's draw verbatim**: on every accepting run of `V false` at `x` and `ω`,
the last message's record has the draw of the statement it decides, as the same term. `checks` compares the two only
up to `JsonEquiv`, and one draw parsed with its keys in another order is another term, so this is a premise on `P`, not
a consequence of acceptance. -/
def RecordDrawVerbatim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : Prop :=
  ∀ r : Fin ((V false).k x) → Coin, (Claims.run sha512 P (V false) x ω r).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst).length).map
            (Claims.coinSeq r)) = .decide s spec sts link c₁ mids f →
        Exec.recordDraw f.record = s.read.draw

/-- **`drawEq` passes on different terms**: two objects with the same fields in trees of different shapes are
`JsonEquiv`, so the checks passing does not give `RecordDrawVerbatim`. -/
theorem drawEq_ne : ∃ a b : Json, drawEq (some a) (some b) = true ∧ a ≠ b := by
  refine ⟨.obj ⟨⟨.inner 2 "a" .null .leaf (.inner 1 "b" .null .leaf .leaf)⟩⟩,
    .obj ⟨⟨.inner 2 "b" .null (.inner 1 "a" .null .leaf .leaf) .leaf⟩⟩, (drawEq_iff _ _).2 (.some rfl), fun h => ?_⟩
  have e := congrArg (fun j : Json => match j with | .obj ⟨⟨.inner _ k _ _ _⟩⟩ => k | _ => "") h
  exact absurd e (by decide)

/-- **A prover of the live session at one table simulates `P`**: `Φ (u, r')`, the coins of `V false` at the tape `u`
and the live coins `r'`, have the law of uniform coins, and wherever `V false` accepts at them, the live outcome of
`P'` at the tape `(ω, u)` is the run's (`viewOf`). -/
def PlainSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ} (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  (∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E) ∧
  ∀ u r', (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (V false) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V false) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r' = viewOf 1 mPts spec link c₁ f

open Classical in
/-- **The plain mixture's first bound, from a simulation**: at a plain statement `x` that sets up at `s₀` with inputs
`I` and draw `d`, if `P`'s accepting runs carry the statement's draw verbatim
(`RecordDrawVerbatim`) and a prover `P'` of the live session at one table simulates `P` (`PlainSim`), then the
accepting runs with no column off `C` are no likelier than the average over `u` of `plainVerdict` against `P'`'s
strategies, at the root of `P`'s first message. -/
theorem plain_first_of_sim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec} {sts₀ : Array Stmt} {d : Json}
    (hset : setupOf false x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d) (hdraw : RecordDrawVerbatim P x ω)
    {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts)
    {τ : Type} [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin)
    (hsim : PlainSim P x ω mPts N dg w P' Φ) :
    Claims.Pr (fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (plainVerdict s₀.read.I d C ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray)
        (liveSession dg 1 mPts (N x)) (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card τ := by
  set E : (Fin ((V false).k x) → Coin) → Prop := fun r => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
    ¬ ∃ (p : Fin M) (v : List 𝔽),
      Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some v ∧ v ≠ C p
    with hEdef
  obtain ⟨hΦ, hsim⟩ := hsim
  refine Pr_le_liveMix (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x ω
    (plainVerdict s₀.read.I d C ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray) E Φ (hΦ E)
    fun u r' hE => ?_
  obtain ⟨hacc, hno⟩ := hE
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hv⟩ := plainVerdict_of_run mPts C hacc
  obtain ⟨rfl, -, -⟩ : s = s₀ ∧ spec = spec₀ ∧ sts = sts₀ := by
    have e := (act_decide hact).symm.trans hset
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    exact e
  have hroot : ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray = link.1.toArray := by
    obtain ⟨rest, hhead⟩ := run_head sha512 P (V false) x ω (Φ (u, r')) hacc
    obtain ⟨rest', -, hlink⟩ := act_decide_link hact
    rw [hhead] at hlink
    simp only [List.headD_cons] at hlink
    rw [hlink]
    rfl
  rw [hroot, hsim u r' hacc s spec sts link c₁ mids f hact]
  exact hv d (hdraw _ hacc s spec sts link c₁ mids f hact) hd hno

end FlockDeploy.KS
