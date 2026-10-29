import Proofs.Flock.Claims.Recursion.VerdictAssembly
import Proofs.Flock.Claims.Recursion.NextShaDec
import Proofs.Flock.Claims.Recursion.NextRRows

/-!
# The decision's facts `runs_assembled` takes

`DecAt.runs_assembled` takes, beside the rows, facts of the decision itself. Every decision makes three of them hold:

* **The session's points** are the reader's (`DecAt.points`): the record the verdict decodes substitutes them
  (`decodeJWith_sub`).
* **Each table has its two proofs** (`DecAt.ps_size`, `DecAt.rep_lt`): `decR` reads two per statement (`decR_fits`).
* **Each statement has its table** (`DecAt.tables_size`): the setup returns one statement per table (`setup_tables`).

At an inner statement read at the dense form, the setup returns one statement (`DecAt.sts_dense`), so table `0` is
the one table `runs_assembled` is at (`DecAt.zero_lt`, `DecAt.zero_lt_tables`).
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP pointsOf)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs)

/-- **The session's points are the reader's**, at every decision: the record the verdict decodes substitutes the
points the reader draws at the first coin. -/
theorem points : D.sess.points = pointsOf D.spec D.d.c₁ :=
  (FlockDeploy.decodeJWith_sub _ D.decode).1

/-- **The decision reads two proofs per statement.** -/
theorem ps_size : D.d.f.ps.size = 2 * D.sts.size := by
  obtain ⟨-, -, -, -, h, -⟩ := FlockDeploy.decR_fits D.dec
  rw [D.setup_eq.2]
  exact h

/-- **Each rep of a table the setup returns has its proof.** -/
theorem rep_lt {i : ℕ} (hi : i < D.sts.size) (r : Fin 2) : 2 * i + r < D.d.f.ps.size := by
  rw [D.ps_size]
  omega

/-- **The setup returns one statement per table.** -/
theorem tables_size : D.spec.tables.size = D.sts.size := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  have h := (FlockDeploy.setup_tables hv).2
  obtain ⟨hs, ht⟩ := D.setup_eq
  rw [hs, ht, hd]
  exact h

/-- **A table the setup returns is a table of the spec.** -/
theorem lt_tables {i : ℕ} (hi : i < D.sts.size) : i < D.spec.tables.size := by
  rw [D.tables_size]
  exact hi

/-- **At an inner statement read at the dense form the setup returns one statement**, the inner statement's. -/
theorem sts_dense {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) : D.sts.size = 1 := by
  obtain ⟨s, spec, st, -, hset, -, -, -⟩ := dense_inner hst hf
  obtain ⟨⟨s', spec', sts'⟩, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hx, hset] at hv
  cases hv
  rw [D.setup_eq.2, hd]
  rfl

/-- **Table `0` is a table of the setup** at the dense form. -/
theorem zero_lt {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) : 0 < D.sts.size := by
  rw [D.sts_dense hst hf hx]
  exact Nat.one_pos

/-- **Table `0` is a table of the spec** at the dense form. -/
theorem zero_lt_tables {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) : 0 < D.spec.tables.size :=
  D.lt_tables (D.zero_lt hst hf hx)

end DecAt

end RecDeploy.Rows
