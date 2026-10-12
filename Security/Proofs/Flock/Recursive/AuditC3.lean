import Proofs.Flock.Recursive.PublicAudit
import Proofs.Flock.Recursive.HoloExec

/-!
# The recursive audit at candidate 3 (trusted)

`RecursiveAuditC3` is `PublicAudit` at candidate 3's inner protocol, the design that runs: the inner session's
circuits have hidden unit circuits, whose templates are the registered table R's, opened by the sparse phase and R's
registered opening (`Model.holoTableCL`, `Recursion.holoInnerAt`). It is a conjunction:

* (i) soundness in fork form, at an inner statement `st` that Lean's verifier sets up from its own partition object and
  program (`AcceptsPublic`), the sparse phase's parameters `hp` (the verifier's own) and R's registration: the event is
  empty, or the sessions type and there is a registrant fork, a round fork, a session fork or an inner fork of the
  prover's own runs whose decoded transcripts play candidate 3's table to outcomes that clash (`clashPickHL`), or
  Lean's verifier accepts `st`, builds V*'s statement list and accepts every session as V*'s statement of its index while
  no table near the registered word has a satisfiable hidden statement, with probability at most `RecursiveSound`'s
  bound with candidate 3's error (`holoError`) as its inner term;
* (ii) zero knowledge: `RecursiveFirewallPrivate`, as in `PublicAudit`.

What it takes, against `PublicAudit`:

* the inner protocol `holoInnerAt T CT st hp` in place of `innerAt T CT st`: `holoTableCL` at `holoAt st hp`, the class,
  regions and schedule of `execClassAt st` with the sparse parameters `hp`, R's registered schedule `fast100`'s at
  `mReg` and R opened at `STACKED_CLAIMS = 2` claims, the two reps' sparse claims (a constant of the model, which
  `regPhase` opens, and of the error, `regOpenError`; no parameter the prover picks);
* R's registration, which says which table the audit is about, as `RecursiveAudit`'s `p`, `path₀` and `_hreg` say which
  private circuit: the record `reg`, the verifier of record's `--rec-inner-registration`, which `st`'s META binds by
  its digest (`metaRegistration st`, as the verifier reads it off `--rec-inner-circuit`, the file `st` is set up from),
  the registered word and cap `xR`, the registrant's openings of every level-0 position (`_hreg`, `HoloReg.Opens`) and
  the cap the record's `stacked` entry (`_hstk`, `StackedTop`), so `st` fixes the cap up to a collision. The word's closeness to a codeword is no hypothesis: the statement holds only when some
  table within the unique-decoding radius of the word has a satisfiable hidden statement, so a far word is a false
  statement that the bound covers;
* V*'s statements indexed by the verifier's own list (`vstarCount ish`, `RecVStar.statements`; gap 4(c)), in place of
  the formula `outerStatementsAt st`; the event adds the verifier's own check that its list builds;
* R's `m`: Lean's verifier never compares META's `stacked.m` with the bucket's `table_m`; it holds R's stream and each
  `rec-opening` unit to META's shape, and the unit names come from the outer program it compiles (the list). The model
  takes R's `m` from the class (`mReg`), and VBridge (`_hbr`, V*'s circuits compute this inner verifier at the list's
  statements) carries the transitive link, which piece (h) derives from the list (`stackedShape_of_list`): derived, not
  a hypothesis of this statement;
* the inner fork's pick `clashPickHL` and the error `holoError (holoAt st hp)` in place of `clashPickCL` and
  `tableError`.

Its proof is `Flock.SecurityProofs.RecursiveAuditC3` (`Proofs.Flock.Recursive.AuditC3Proof`), from piece (a)
(`FlockC3.Fork.holoForkSound_holoAt`). Its non-vacuity needs VBridge at candidate 3 (piece (b), `FlockC3.Compose`) and
V*'s statements over the verifier's list (piece (h)), which at candidate 3's inner shape is circuits' gap 2's eleven:
`rec-open`, `rec-residuals`, `rec-sparse`, R's levels (`rec-open-stacked`) and `rec-opening` at META's `stacked`.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The recursive audit at candidate 3.**

(i) For every partition object `obj` and program `prog` of the verifier's own, every inner statement `st` and sparse
parameters `hp`, R's registration (the verifier of record's record `reg`, which `st`'s META binds by its digest, the
registered word and cap `xR` with the registrant's openings `_hreg`, its cap the record's `stacked` entry `_hstk`), the
verifier of record's inner
shape `ish`, and every prover `σ` of the recursive audit with candidate 3's compiled table as its inner protocol
(`holoInnerAt`), at the law of the registration's draws (`drawLaw`): V*'s statements `Vs`, one for each of the
verifier's list (`vstarCount ish`), V*'s sessions `zr R ω t R₂ j` as the prover supplies them (`ZkOuterR`), their
records' custody `_hCust`, the verifier of record's own inputs (V*'s bindings `bind` and the shared rows `rows`; session
`j` runs as V*'s statement `j`), at which V*'s statements' files parse to `Vs` (`VStarParses`), every session's inputs
carrying the tags their name names (`TagsFound`), the messages V*'s layers hold and the rounds' commitments (`wmsg`,
`Sk`, `cmt`, `msg`). Taken: A3 (`_hA3`).
Then for every count curve `(k, R_w, M, ρ)` that bounds every session's rate at each setup of its inputs, given that
V*'s circuits compute candidate 3's inner verifier at the typed sessions and R's registration (VBridge): the event below
is empty; or the sessions type and there is a registrant fork, a round fork, a session fork, or an inner fork of the
transported prover's own runs whose decoded transcripts play candidate 3's table to outcomes that clash; or
$$\Pr[\text{Lean's verifier builds } st \wedge \text{V*'s list builds} \wedge \text{every session verifies as V*'s
  statement of its index} \wedge \neg\,\exists R \text{ near } w,\ R\text{'s hidden statement holds}]$$
is at most V*'s sessions' expected statistical errors under the full draw's model, `r` times their expected stage
slack, and candidate 3's error.

(ii) `RecursiveFirewallPrivate`, as in `PublicAudit`. -/
def RecursiveAuditC3 : Prop :=
  (∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Inhabited (CT k)]
    -- the registration's draws, a finite space with an element, and the units each draws
    {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] {draw : Ω → Finset (Fin n)} {Reg Cm : Type} {Reg' Reg₂ : Type}
    -- the verifier's own partition object and program, the inner statement and the sparse phase's parameters
    (obj : Lean.Json) (prog : ByteArray) (st : Flock.Stmt) (hp : HoloParams)
    -- R's registration: the verifier of record's own input, the record `--rec-inner-registration` that `st`'s META
    -- binds; the registered word and cap, the registrant's openings and the record's `stacked` entry
    (reg : Lean.Json) (xR : HoloReg FlockLevel3.GF128 Refine.Dig)
    (_hreg : xR.Opens Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp).rsch.sch.level₀)
    (_hstk : StackedTop (holoAt st hp).rsch.sch.level₀ st reg xR.cap)
    -- the verifier of record's inner shape (`--rec-inner`), and V*'s statements over its list, as the prover supplies
    -- their sessions
    (ish : RecVStar.Inner) {Vs : Fin (vstarCount ish) → VStmt}
    (zr : Reg → (drawLaw draw).Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ →
      (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg')
    -- the records' custody
    (_hCust : Flock.Premises.RecordCustody zr)
    -- the verifier of record's own inputs: V*'s bindings for each statement and the shared rows
    (bind : Fin (vstarCount ish) → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    -- obligation (gap 4(c)): V*'s statement `j`'s file under `bind j` parses to `Vs j`'s circuit
    (_hob : VStarParses Flock.Tags.circuit ish bind Vs)
    -- obligation (our code): every session's inputs carry the tags their name names
    (_hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover
    (σ : Strategy (recGame (drawLaw draw) Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerVR (zr R ω t) dg N))
    -- the messages V*'s layers hold, the rounds' commit strings and commitments
    (wmsg : ((j : Fin (vstarCount ish)) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (Sk : ℕ → (j : Fin (vstarCount ish)) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv →
      (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (vstarCount ish)) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (holoInnerAt T CT st hp).Msg)
    -- A3
    (_hA3 : Flock.Premises.UniformRandomBytes Vs)
    -- the count curve, bounding every session's rate
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S₀
      (d₀ : Lean.Json) (y₀ : ZkHidden.DrawSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    -- VBridge, a premise: V*'s circuits compute candidate 3's inner verifier, at the typed sessions and R's
    -- registration
    (_hbr : ∀ h : Typed zr,
      Flock.Premises.VBridge (custodied zr _hCust h) dg N k Rw (fun _ _ => Subtype.mk xR _hreg) wmsg Sk cmt msg),
    prob (fun o => (AcceptsPublic obj prog st ∧ (RecVStar.statements ish).isOk = true) ∧
          VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ¬ (holoInnerAt T CT st hp).holds (Subtype.mk xR _hreg))
        (recGame (drawLaw draw) Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerVR (zr R ω t) dg N) σ = 0 ∨
    ∃ h : Typed zr,
      (regFinder (custodied zr _hCust h) dg N (transported zr _hCust h σ)).Finds Refine.H512 ∨
      RoundFork (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) cmt ∨
      SessionFork (custodied zr _hCust h) dg N k Rw M (transported zr _hCust h σ) ∨
      InnerForkRec (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) (fun _ _ => Subtype.mk xR _hreg)
        wmsg Refine.H512 (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)) ∨
      prob (fun o => (AcceptsPublic obj prog st ∧ (RecVStar.statements ish).isOk = true) ∧
            VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (holoInnerAt T CT st hp).holds (Subtype.mk xR _hreg))
          (recGame (drawLaw draw) Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerVR (zr R ω t) dg N) σ ≤
        expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
            (fun R₂ j τ =>
              (custodied zr _hCust h σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((holoInnerAt T CT st hp).r : ℝ≥0∞) *
            expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
              (fun R₂ j τ => (custodied zr _hCust h σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp))) ∧
  RecursiveFirewallPrivate

end Flock.Guarantees
