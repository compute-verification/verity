import FlockSoundness.Discharge.Layout.Scope
import FlockSoundness.Discharge.ZkExec.Event
import FlockSoundness.Discharge.Hidden.Facts
import FlockSoundness.ExecSetup
import Flock.ProvedScope
import FlockSoundness.Protocol.Discharge.FailClosed.Scope
import FlockSoundness.Discharge.Layout.KeyProg

/-!
# What the verifier accepts: `Accepts` and its statement check

`flock-verify verify` refuses every session of a statement whose form the soundness proof does not cover:
`Main.verifyCmd` runs `Flock.ProvedScope.check` on the statement and turns every session's verdict into the check's
refusal when it fails. So what the verifier accepts is `Verifies` (under `--zk`, `VerifiesZK`): the setup and verdict
`Accepts` models, and the check. Every condition on the statement's form that the headlines took as a hypothesis follows
from the check: an untyped statement (`Checks.typed`), retained rounds and `hm96-sha512/v1` leaves, at most one table
without `--zk`, the circuit and public file as the headlines read them (`circ`, `pubOf`), no shared-row public file
(`Checks.tables`), and a circuit `Layout.scopeOk` accepts or, at a `--zk` statement with hidden outputs, one in the
canonical format (`Checks.covered`: the executable's `ProvedScope.scopeOk` is `Layout.scopeOk`, its port bits a mask,
`scopeOk_eq`; `ProvedScope.canonicalOk` is `Canonical.formatOk` and the unit returning the outputs,
`Canonical.parse_formatOk`). So outside `--zk` statements with hidden outputs the circuit is in `Layout.scopeOk`
(`Checks.scope`), and outside `Layout.scopeOk` it is canonical (`Checks.canonical`). The unconditional headlines
(`FailClosed.Composed`, `FailClosed.ZkHidden`, `FailClosed.Exec`, `Canonical.Headline`) are over `Verifies`, with no
hypothesis on the statement's form.

`Checks` is `verifyCmd`'s call `ProvedScope.check tags zk circuitB pubB tables count publicInputs.isSome` argument for
argument. `tags` is `Inputs.tags` without `--zk` and `zkTags` under it (`--statement`'s, after `withCoins`, then `Zk.tags`
at `maskLogOf circuitB` and `count`, on the coin tree of `tableNames count`). The circuit, public file and tables are the
bytes `verifyCmd` sets the session up from, and `count` is `--session-tables`. The flag is `false`: with
`--public-inputs` the check refuses every session, and without it `buildSession` passes `setupTables` no public input
words, the setup `Accepts` models. A session's verdict is `verifyCmd`'s `setupOf` at the record's `recordDraw` (the check
requires `hm96Rows`): `setupTables`, under `--zk` `Zk.shapeOk` on every table, then `ProvedScope.drawLaw` on that draw,
the check, and then `verify` or `Zk.verify`. So a session `verifyCmd` accepts satisfies `Verifies` (under `--zk`,
`VerifiesZK`). `verifyCmd` refuses more, which only shrinks the event: a draw by a law no soundness claim draws by
(`ProvedScope.drawLaw`: a bernoulli or work draw, or a subset draw without `--zk`), registered reads (`checkRegistered`:
a header naming registered values without `--registered`; with it, `Registered.check` needs a shared-row public file,
which the check refuses), a coin-tree key an earlier session of the run used, and a session file it cannot read. A later session of a run whose draw has an earlier
one's canonical form is verified against that earlier setup; `canon` takes integers only, so the two draws hold the same
content.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec

/-! ## The executable's `scopeOk` is `Layout.scopeOk` -/

/-! ## The check, and what it gives -/

namespace Checks

variable {t : Tags} {zk : Bool} {I : Inputs} (h : Checks t zk I)
include h

theorem hm96Rows : t.hm96Rows = true := (check_ok h).1.1

end Checks

/-! ## What the verifier accepts -/

variable {n : ℕ} {Reg : Type}

/-- `Verifies` at the live session's transcript. -/
def LiveVerifies (I : Inputs) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop := Verifies I o.2.2

theorem ScopeZ.outNet {I : Inputs} (h : ScopeZ I) : (circZ I).outNet = (circZ I).unitNet := Checks.outNet h

/-! ## A bound under the statement check -/

/-- **A bound at every statement the check passes, lifted to the event that implies the check.** When the event `V`
implies `A` and the statement fact `Q`, and `A ∧ E q` has probability at most `B q` for every proof `q` of `Q`, then
`V ∧ ∀ q, E q` has probability at most `⨆ q, B q`: at a statement without `Q`, `V` is empty and the bound is `0`. -/
theorem prob_scope {α : Type} {V A : α → Prop} {Q : Prop} (hA : ∀ o, V o → A o) (hQ : ∀ o, V o → Q)
    (E : Q → α → Prop) (B : Q → ENNReal) (g : FlockSoundness.Game α) (s : Game.Strategy g)
    (h : ∀ q, Game.prob (fun o => A o ∧ E q o) g s ≤ B q) :
    Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ ⨆ q, B q := by
  by_cases hq : Q
  · calc Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ Game.prob (fun o => A o ∧ E hq o) g s :=
          Game.prob_mono (fun o ho => ⟨hA o ho.1, ho.2 hq⟩) g s
      _ ≤ B hq := h hq
      _ ≤ ⨆ q, B q := le_iSup B hq
  · calc Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ Game.prob (fun _ => False) g s :=
          Game.prob_mono (fun o ho => hq (hQ o ho.1)) g s
      _ = 0 := Game.prob_false g s
      _ ≤ ⨆ q, B q := zero_le

end FlockSoundness.Discharge.FailClosed
