import Proofs.Flock.Claims.KS.ZkAcceptsAt

/-!
# An accepted run of the deployed `--zk` verifier is flock-verify `--zk`'s acceptance, at a normal draw

`acceptsAt_of_run` gives `verify --zk`'s acceptance at the statement's draw; `AcceptsZK` sets the session up at the
record's draw, and the checks pin that draw to the statement's only up to key order (`Custody.checks_draw`). Where the
verifier reads both draws in a normal form, two equivalent draws are one:

* `DrawNorm`: a map `norm` sending equivalent values to one value; its fixed points are the normal draws, and two
  equivalent normal draws are equal (`DrawNorm.eq_of_equiv`, `DrawNorm.eq_of_rel`).
* `recordDraw_eq_of_checks`: where `Deploy.V`'s checks pass and both the record's draw and the statement's are normal,
  the record's draw is the statement's as a value (the `hdraw` of `Custody.accepts_of_decide`).
* `liveAcceptsZK_of_run_norm`: an accepted run of `V true` against SHA-512 is `AcceptsZK`, and so `LiveAcceptsZK` at
  every outcome carrying the last message's transcript, at every normal form the verifier reads both draws in.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge
open Lean (Json)

/-- **A normal form of draws**: a map sending equivalent values (`JsonEquiv`) to one value. -/
structure DrawNorm where
  norm : Json → Json
  congr : ∀ {a b : Json}, JsonEquiv a b → norm a = norm b

/-- **Two equivalent normal draws are equal.** -/
theorem DrawNorm.eq_of_equiv (N : DrawNorm) {a b : Json} (ha : N.norm a = a) (hb : N.norm b = b)
    (h : JsonEquiv a b) : a = b := by
  rw [← ha, ← hb]
  exact N.congr h

/-- **Two optional draws, equivalent and each normal where present, are equal.** -/
theorem DrawNorm.eq_of_rel (N : DrawNorm) {a b : Option Json} (ha : ∀ d, a = some d → N.norm d = d)
    (hb : ∀ d, b = some d → N.norm d = d) (h : Option.Rel JsonEquiv a b) : a = b := by
  cases h with
  | none => rfl
  | some h => exact congrArg some (N.eq_of_equiv (ha _ rfl) (hb _ rfl) h)

/-- **The record's draw is the statement's**, where `Deploy.V`'s checks pass and the verifier reads both in normal form:
the checks give them equivalent (`Custody.checks_draw`), and equivalent normal draws are equal. -/
theorem recordDraw_eq_of_checks (N : DrawNorm) {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hrec : ∀ d, Exec.recordDraw f.record = some d → N.norm d = d)
    (hst : ∀ d, s.read.draw = some d → N.norm d = d)
    (hc : checks zk s spec x link c₁ mids f = .ok ()) : Exec.recordDraw f.record = s.read.draw :=
  N.eq_of_rel hrec hst (Claims.Bridge.Custody.checks_draw hc)

/-- **An accepted run of `V true` is flock-verify `--zk`'s acceptance**, at a normal form the verifier reads every
record's draw (`hrec`) and the statement's draw (`hst`) in: against SHA-512, the run decided at the statement's inputs
`s`, and `verify --zk` accepts the last message's record and proof files `f` at `s.read.I` (`AcceptsZK`), so
`LiveAcceptsZK` holds at every outcome carrying that transcript. -/
theorem liveAcceptsZK_of_run_norm (N : DrawNorm) {n : ℕ} {Reg : Type} {Ω : Type} {P : Claims.Prover Ω}
    {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    (hrec : ∀ (record : String) (d : Json), Exec.recordDraw record = some d → N.norm d = d)
    (hst : ∀ (s : Ins) (d : Json), insOf true x = some s → s.read.draw = some d → N.norm d = d)
    (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ (s : Ins) (f : Final), insOf true x = some s ∧
      ZkExec.AcceptsZK s.read.I (⟨f.record, f.proofs⟩ : Exec.Transcript) ∧
      ∀ (R : Reg) (S : Finset (Fin n)),
        ZkExec.LiveAcceptsZK s.read.I (R, S, (⟨f.record, f.proofs⟩ : Exec.Transcript)) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, -⟩ := acc_decide h
  have hacc : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := run_acc_next sha512 P (V true) x ω r h
  obtain ⟨-, hc⟩ := (next_decide_sha512 hact).1 hacc
  have hins := (setupOf_some (act_decide hact)).1
  have hdraw := recordDraw_eq_of_checks N (hrec f.record) (fun d hd => hst s d hins hd) hc
  obtain ⟨d, hd⟩ := checks_draw hc
  have hA := (Claims.Bridge.Custody.acceptsAt_of_decide (J := 0) (mPts := 0) hact hacc hd (Nat.zero_le _)).1
  have hZ := (acceptsAt_iff_acceptsZK (hdraw.trans hd)).1 hA
  exact ⟨s, f, hins, hZ, fun _ _ => hZ⟩

end FlockDeploy.KS
