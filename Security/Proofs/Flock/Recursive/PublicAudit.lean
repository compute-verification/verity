import Proofs.Flock.Recursive.Property
import Proofs.Flock.Recursive.Firewall

/-!
# The recursive audit at public units (trusted)

`PublicAudit` is `RecursiveAudit` where Lean's verifier builds the inner statement itself, from its own partition object
and program: the inner unit is public, so there is no registration, no private circuit's description and no hidden
statement read from registered rows. It is a conjunction:

* (i) soundness in fork form, at an inner statement `st` that Lean's verifier sets up from its own partition object
  `obj` and program `prog` (`AcceptsPublic`), its class, regions and schedule `execClassAt st` as in `RecursiveAudit`:
  the event is empty, or the sessions type and there is a registrant fork (`regFinder`), a round fork, a session fork
  or an inner fork of the prover's own runs, or Lean's verifier accepts `st` and the verifier of record every session
  while `st` fails with probability at most `RecursiveSound`'s bound with the compiled table's error as its inner term;
* (ii) zero knowledge: `RecursiveFirewallPrivate`, `RecursiveAudit`'s (ii), named; at a public unit it is read at no
  registration rows, the simulator reading the public circuit and the owed outputs.

What it takes, against `RecursiveAudit`:

* gone: the registration's root `rt`, the private circuit's statement `stmt`, the registered commit strings `p` and
  the developer's paths `path₀` with their openings (`_hreg`), the description's acceptance (`AcceptsDescription`) in
  the event, and the registered-collision disjunct; in (ii), read at `nr = 0`, the registration's term is zero;
* the inner statement's hidden part is `publicHidden st`, the class's one statement (`execClassAt`'s class has no
  hidden entries, `execClassAt_nnz`), in place of `hiddenAt` and `stmtAt`;
* `AcceptsPublic obj prog st` in place of `AcceptsStmt st`: the partition object and program are the verifier's own,
  not some the prover picks (`AcceptsPublic.acceptsStmt`: it implies `AcceptsStmt`, so `ExecClass`'s lemmas apply);
* premises: record custody (`Flock.Premises.RecordCustody`), A3 (`Flock.Premises.UniformRandomBytes`) and VBridge at
  the committed rounds (`Flock.Premises.VBridgeC`, which `VBridge` gives: V*'s circuits compute C-Flock's inner
  verifier at the rounds before the last, here at the public statement), as in
  `RecursiveAudit`. The coin server's randomness is the game's own coin draw (`uniform/flock-coin-server`, no `Prop`),
  and SHA-512 enters only as the fork disjuncts;
* obligations, facts about our own code: in (i), `VStarParses` (V*'s files parse to `Vs`, until gap 4) and `TagsFound`
  (each session's inputs carry the tags their name names, `_hF`), as in `RecursiveAudit`; in (ii), whatever
  `RecursiveFirewallPrivate` takes.

Its proof is `Flock.SecurityProofs.PublicAudit` (`Proofs.Flock.Recursive.PublicAuditProof`). Its non-vacuity is to be
shown at one recorded recursive run with a public inner unit (the F1 gate).
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Lean's verifier builds the inner statement from its own partition object and program**: `setupTables` at the inner
tags, the coin server's coins and one table, given the circuit and public files and the verifier's own `obj` and `prog`,
returns `st`, and `Flock.Zk.schedule` accepts its `m`. -/
def AcceptsPublic (obj : Lean.Json) (prog : ByteArray) (st : Flock.Stmt) : Prop :=
  (∃ (cf pf : ByteArray) (spec : Flock.SessionSpec),
    Flock.Stmt.setupTables Flock.RecVStar.innerTags cf pf Flock.Coins.os 1 (partition := some obj)
      (program := some prog) = .ok (spec, #[st])) ∧
  ∃ p, Flock.Zk.schedule st.m = .ok p

theorem AcceptsPublic.acceptsStmt {obj : Lean.Json} {prog : ByteArray} {st : Flock.Stmt}
    (h : AcceptsPublic obj prog st) : AcceptsStmt st := by
  obtain ⟨⟨cf, pf, spec, hs⟩, hp⟩ := h
  exact ⟨⟨cf, pf, some obj, some prog, spec, hs⟩, hp⟩

/-- `execClassAt`'s class has no hidden entries. -/
theorem execClassAt_nnz (st : Flock.Stmt) : (execClassAt st).cls.nnz = 0 := by
  unfold execClassAt
  split <;> rfl

/-- **The public inner statement's hidden part**: the one statement of a class with no hidden entries. -/
def publicHidden (st : Flock.Stmt) : Hidden (execClassAt st).cls :=
  fun i => (Nat.not_lt_zero _ (lt_of_lt_of_eq i.isLt (execClassAt_nnz st))).elim

/-- **The recursive audit at public units.**

(i) For every partition object `obj` and program `prog` of the verifier's own, every inner statement `st`, with its
class, regions and schedule `execClassAt st`, and every prover `σ` of the recursive audit with C-Flock's compiled table
there as its inner protocol (`innerAt`), at the law of the registration's draws (`drawLaw`: a finite space with an
element, so the law's `nonempty` is no hypothesis): V*'s statements `Vs` (`outerStatementsAt st` of them, each drawing
all its units), V*'s sessions `zr R ω t R₂ j` as the prover supplies them (`ZkOuterR`), their records' custody
`_hCust`, the verifier of record's own inputs to them (the inner shape `ish`, V*'s bindings `bind` and the shared rows
`rows`; session `j` runs as V*'s statement `j`), at which V*'s statements' files parse to `Vs` (`VStarParses`, an
obligation until gap 4), every session's inputs carrying the tags their name names (`TagsFound`, `_hF`, an obligation
on our code), the messages V*'s layers hold and the rounds' commitments (`wmsg`, `Sk`, `cmt`, `msg`), as in
`RecursiveAudit`. Taken: A3 (`_hA3`).
Then for every count curve `(k, R_w, M, ρ)` that bounds every session's rate at each setup of its inputs, given that
V*'s circuits compute the inner verifier at the typed sessions and the public statement, at the rounds before the last
(VBridge at the committed rounds, `Flock.Premises.VBridgeC`): the event below is empty; or the sessions type and there
is a registrant fork (one run of the transported prover reaching two sessions whose registrant records clash,
`regFinder`), a round fork, a session fork, or an inner fork of the transported prover's own runs whose decoded
transcripts play the compiled table to outcomes that clash; or
$$\Pr[\text{Lean's verifier builds } st \text{ from } obj, prog \wedge \text{every session verifies as V*'s statement
  of its index} \wedge \neg\,st \text{ holds}]$$
is at most V*'s sessions' expected statistical errors under the full draw's model, `r` times their expected stage
slack, and the compiled table's error.

(ii) `RecursiveFirewallPrivate`, `RecursiveAudit`'s (ii), named rather than restated. At a public unit it is read at
no registration rows (`nr = 0`) and the public circuit `P` as the simulator's `p₀`: the registration's term
`2 n_r·δ_{K*}` is zero, the bound is `ε_V(δ_{K*}) + n_h·δ_{K*}`, and the simulator reads `P`, the schedule's transcript
and the owed outputs. -/
def PublicAudit : Prop :=
  (∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Inhabited (CT k)]
    -- the registration's draws, a finite space with an element, and the units each draws
    {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] {draw : Ω → Finset (Fin n)} {Reg Cm : Type} {Reg' Reg₂ : Type}
    -- the verifier's own partition object and program, and the inner statement
    (obj : Lean.Json) (prog : ByteArray) (st : Flock.Stmt)
    -- V*'s statements and sessions, as the prover supplies them
    {Vs : Fin (outerStatementsAt st) → VStmt}
    (zr : Reg → (drawLaw draw).Ω → List (Cm × (innerAt T CT st).Coin) → Reg₂ →
      (j : Fin (outerStatementsAt st)) → (Vs j).OuterR Reg')
    -- the records' custody
    (_hCust : Flock.Premises.RecordCustody zr)
    -- the verifier of record's own inputs to V*'s sessions: the inner shape (`--rec-inner`), V*'s bindings for each
    -- statement (`--rec-bindings`) and the shared rows (`--rec-rows`); session `j` runs with `--rec-part j`
    (ish : RecVStar.Inner) (bind : Fin (outerStatementsAt st) → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port))
    -- obligation (gap 4): V*'s statement `j`'s file under `bind j` parses to `Vs j`'s circuit
    (_hob : VStarParses Flock.Tags.circuit ish bind Vs)
    -- obligation (our code): every session's inputs carry the tags their name names, as `verifyCmd` forms them
    (_hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover
    (σ : Strategy (recGame (drawLaw draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N))
    -- the messages V*'s layers hold, the rounds' commit strings and commitments
    (wmsg : ((j : Fin (outerStatementsAt st)) → (Vs j).Bits) → List (innerAt T CT st).Msg)
    (Sk : ℕ → (j : Fin (outerStatementsAt st)) →
      Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin (outerStatementsAt st)) → Cm →
      PosK (Vs j).c (Vs j).hU (Vs j).nv →
      (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (outerStatementsAt st)) →
      PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (innerAt T CT st).Msg)
    -- A3
    (_hA3 : Flock.Premises.UniformRandomBytes Vs)
    -- the count curve, bounding every session's rate
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S₀
      (d₀ : Lean.Json) (y₀ : ZkHidden.DrawSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    -- VBridge at the committed rounds, a premise: V*'s circuits compute the inner verifier at the rounds before the
    -- last, at the typed sessions and the public statement
    (_hbr : ∀ h : Typed zr,
      Flock.Premises.VBridgeC (custodied zr _hCust h) dg N k Rw (fun _ _ => publicHidden st) wmsg Sk cmt msg),
    prob (fun o => AcceptsPublic obj prog st ∧ VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ¬ (innerAt T CT st).holds (publicHidden st))
        (recGame (drawLaw draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N) σ = 0 ∨
    ∃ h : Typed zr,
      (regFinder (custodied zr _hCust h) dg N (transported zr _hCust h σ)).Finds Refine.H512 ∨
      RoundFork (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) cmt ∨
      SessionFork (custodied zr _hCust h) dg N k Rw M (transported zr _hCust h σ) ∨
      InnerForkRec (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) (fun _ _ => publicHidden st) wmsg
        Refine.H512 (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (execClassAt st).cls
          (execClassAt st).regs (execClassAt st).sch) ∨
      prob (fun o => AcceptsPublic obj prog st ∧ VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (innerAt T CT st).holds (publicHidden st))
          (recGame (drawLaw draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N) σ ≤
        expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
            (fun R₂ j τ =>
              (custodied zr _hCust h σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          ((innerAt T CT st).r : ℝ≥0∞) *
            expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
              (fun R₂ j τ => (custodied zr _hCust h σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError (execClassAt st).sch
            ⟨(execClassAt st).cls.kLog, (execClassAt st).regs.length, (execClassAt st).cls.mPts⟩)) ∧
  RecursiveFirewallPrivate

end Flock.Guarantees
