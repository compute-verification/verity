import Proofs.Flock.Claims.KS.ZkAcceptsAt

/-!
# An accepted run of the deployed `--zk` verifier is flock-verify `--zk`'s acceptance, where it reads draws normed

`acceptsAt_of_run` gives `verify --zk`'s acceptance at the statement's draw; `AcceptsZK` sets the session up at the
record's draw, and the checks pin that draw to the statement's only up to key order (`Custody.checks_draw`). Where the
verifier reads both draws through one normal form, two equivalent draws are one:

* `DrawNorm`: a map `norm` sending equivalent values to one value (`congr`) and fixing its own values (`idem`); what it
  returns is normal (`DrawNorm.normal`), and two equivalent normal draws are equal (`DrawNorm.eq_of_equiv`,
  `DrawNorm.eq_of_rel`).
* `recordDraw_eq_of_checks`: where `Deploy.V`'s checks pass and both the record's draw and the statement's are normal,
  the record's draw is the statement's as a value (`hdraw`).
* `liveAcceptsZK_of_run_norm`: an accepted run of `V true` against SHA-512 is `AcceptsZK`, and so `LiveAcceptsZK` at
  every outcome carrying the last message's transcript, wherever the verifier reads every draw as `norm` of a value.
* `flockNorm`, `liveAcceptsZK_of_run_rebuilt`: the verifier's rebuild `Flock.norm` is such a normal form, and it reads
  both draws through it, so the last holds with no premise on the draw.

`idem` is a field, not a consequence of `congr`: a rebuild that inserts an object's fields in its tree's order into an
empty map can change the shape of a tree whose order is not the keys' order, and rebuilding the result (now in key order)
can change it again.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge
open Lean (Json)

/-- **A normal form of draws**: a map sending equivalent values (`JsonEquiv`) to one value, and fixing its values. -/
structure DrawNorm where
  norm : Json → Json
  congr : ∀ {a b : Json}, JsonEquiv a b → norm a = norm b
  idem : ∀ a, norm (norm a) = norm a

/-- **What the normal form returns is normal.** -/
theorem DrawNorm.normal (N : DrawNorm) {d d₀ : Json} (h : d = N.norm d₀) : N.norm d = d := by
  subst h
  exact N.idem d₀

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
    (hrec : ∀ d, Exec.recordDraw f.record = some d → ∃ d₀, d = N.norm d₀)
    (hst : ∀ d, s.read.draw = some d → ∃ d₀, d = N.norm d₀)
    (hc : checks zk s spec x link c₁ mids f = .ok ()) : Exec.recordDraw f.record = s.read.draw :=
  N.eq_of_rel (fun d hd => (hrec d hd).elim fun _ h => N.normal h)
    (fun d hd => (hst d hd).elim fun _ h => N.normal h) (Claims.Bridge.Custody.checks_draw hc)

/-- **An accepted run of `V true` is flock-verify `--zk`'s acceptance**, wherever the verifier reads every record's draw
(`hrec`) and the statement's draw (`hst`) as `norm` of a value: against SHA-512, the run decided at the statement's
inputs `s`, and `verify --zk` accepts the last message's record and proof files `f` at `s.read.I` (`AcceptsZK`), so
`LiveAcceptsZK` holds at every outcome carrying that transcript. -/
theorem liveAcceptsZK_of_run_norm (N : DrawNorm) {n : ℕ} {Reg : Type} {Ω : Type} {P : Claims.Prover Ω}
    {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    (hrec : ∀ (record : String) (d : Json), Exec.recordDraw record = some d → ∃ d₀, d = N.norm d₀)
    (hst : ∀ (s : Ins) (d : Json), insOf true x = some s → s.read.draw = some d → ∃ d₀, d = N.norm d₀)
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

/-- **The verifier's rebuild is a normal form of draws** (`Flock.norm`: `JsonNorm.norm_congr`, `JsonNorm.norm_norm`). -/
def flockNorm : DrawNorm where
  norm := Flock.norm
  congr h := FlockDeploy.norm_congr _ _ h
  idem := FlockDeploy.norm_norm

/-- **An accepted run of `V true` is flock-verify `--zk`'s acceptance, with no premise on the draw**: against SHA-512,
the run decided at the statement's inputs `s`, and `verify --zk` accepts the last message's record and proof files `f`
at `s.read.I` (`AcceptsZK`), so `LiveAcceptsZK` holds at every outcome carrying that transcript. The verifier rebuilds
every draw it reads (`flockNorm`; the record's, `Custody.recordDraw_norm`, and the statement's,
`DeployInputs.insOf_draw_norm`), so the record's draw is the statement's as a value. -/
theorem liveAcceptsZK_of_run_rebuilt {n : ℕ} {Reg : Type} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ (s : Ins) (f : Final), insOf true x = some s ∧
      ZkExec.AcceptsZK s.read.I (⟨f.record, f.proofs⟩ : Exec.Transcript) ∧
      ∀ (R : Reg) (S : Finset (Fin n)),
        ZkExec.LiveAcceptsZK s.read.I (R, S, (⟨f.record, f.proofs⟩ : Exec.Transcript)) :=
  liveAcceptsZK_of_run_norm flockNorm
    (fun _ d hd => ⟨d, (Claims.Bridge.Custody.recordDraw_norm hd).symm⟩)
    (fun _ d hs hd => ⟨d, (FlockDeploy.insOf_draw_norm hs hd).1.symm⟩) h

end FlockDeploy.KS
