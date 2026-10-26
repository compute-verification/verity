import Proofs.Flock.Claims.KS.ZkDrawJ
import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.KS.PlainWrP

/-!
# An accepted run of the deployed `--zk` verifier, at the statement's draw

The claims form's Flock verifier is `V true` (`deployZk`'s and `deployZkTrim`'s `V`). Where its run accepts against
SHA-512 (`acceptsAt_of_run`), it decided at the statement's inputs `s` (`insOf true x = some s`), the statement carries a
draw `d` (the checks refuse a session without one, `checks_draw`), and `verify --zk` accepts the last message's record and
proof files at the session it sets up at `d` (`Composed.AcceptsAt`). Nothing is assumed of the record's own draw.

flock-verify `--zk` itself sets the session up at the record's draw (`AcceptsZK`); the checks pin that draw to the
statement's only up to key order (`Custody.checks_draw`). Where the two are equal as values, `AcceptsAt` at `d` is
`AcceptsZK` (`acceptsAt_iff_acceptsZK`), so `liveAcceptsZK_of_runAt` gives `LiveAcceptsZK` at every outcome carrying the
last message's transcript.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge
open Lean (Json)

/-- **`AcceptsAt` at the record's own draw is `AcceptsZK`**: `verify --zk` sets the session up at the record's draw, so
where that draw is `d`, accepting at the setup at `d` is accepting. -/
theorem acceptsAt_iff_acceptsZK {I : Exec.Inputs} {d : Json} {t : Exec.Transcript}
    (hd : Exec.recordDraw t.record = some d) : Composed.AcceptsAt I d t ↔ ZkExec.AcceptsZK I t := by
  have hs : Exec.setupOf (ZkExec.zkInputs I) t.record = setupAt (ZkExec.zkInputs I) (some d) := by
    rw [setupOf_eq_setupAt, hd]
  constructor
  · rintro ⟨spec, sts, h, hsh, hv⟩
    exact ⟨spec, sts, hs.trans h, hsh, hv⟩
  · rintro ⟨spec, sts, h, hsh, hv⟩
    exact ⟨spec, sts, hs.symm.trans h, hsh, hv⟩

/-- **An accepted run of `V true` is accepted by `verify --zk` at the statement's draw**: against SHA-512, the run
decided at the statement's inputs `s`, whose draw is some `d`, and `verify --zk` accepts the last message's record and
proof files `f` at the session it sets up at `d`, with no premise on the record's own draw. -/
theorem acceptsAt_of_run {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ (s : Ins) (d : Json) (f : Final), insOf true x = some s ∧ s.read.draw = some d ∧
      Composed.AcceptsAt s.read.I d (⟨f.record, f.proofs⟩ : Exec.Transcript) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, -⟩ := acc_decide h
  have hacc : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := run_acc_next sha512 P (V true) x ω r h
  obtain ⟨-, hc⟩ := (next_decide_sha512 hact).1 hacc
  obtain ⟨d, hd⟩ := checks_draw hc
  exact ⟨s, d, f, (setupOf_some (act_decide hact)).1, hd,
    (Claims.Bridge.Custody.acceptsAt_of_decide (J := 0) (mPts := 0) hact hacc hd (Nat.zero_le _)).1⟩

/-- **`acceptsAt_of_run` as the live event**: where the last message's record carries the statement's draw as a value,
`LiveAcceptsZK` holds at the statement's inputs at every outcome whose transcript is that record and its proof files. -/
theorem liveAcceptsZK_of_runAt {n : ℕ} {Reg : Type} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ (s : Ins) (d : Json) (f : Final), insOf true x = some s ∧ s.read.draw = some d ∧
      Composed.AcceptsAt s.read.I d (⟨f.record, f.proofs⟩ : Exec.Transcript) ∧
      (Exec.recordDraw f.record = some d → ∀ (R : Reg) (S : Finset (Fin n)),
        ZkExec.LiveAcceptsZK s.read.I (R, S, (⟨f.record, f.proofs⟩ : Exec.Transcript))) := by
  obtain ⟨s, d, f, hins, hd, hA⟩ := acceptsAt_of_run h
  exact ⟨s, d, f, hins, hd, hA, fun hr _ _ => (acceptsAt_iff_acceptsZK hr).1 hA⟩

end FlockDeploy.KS
