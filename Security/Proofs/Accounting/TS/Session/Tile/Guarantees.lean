import Proofs.Accounting.TS.Session.Tile.Assumptions
import Proofs.Accounting.TS.Session.Checked.Guarantees

/-!
# A tile unit's proof layer with its Definition's checks is sound through its C-Flock sessions (the statement)

`Accounting.Guarantees.TS.TileSessionSound`: `Accounting.Guarantees.TS.CheckedSessionSound`'s window, with the layer
`Accounting.TS.Tile.tileLayer` in place of `checkedLayer`. Its relation is the unit's Definition with its check words
fixed to `1` (`PouwH1TileZy_v1` with `ok = 1`), what `Accounting.Served`'s docstring asks of a realizing layer's `pf.rel`;
`CheckedSessionSound` proves the checked layer only, whose `CheckedRel` says nothing of the product or `dq`. This
statement names what the Definition reading takes:

* (a) the `ok` rows are the verifier's, proved from flock-verify's known-content refusal (`Registered.checkKnownAll`)
  as in `Accounting.Guarantees.TS.HeldSessionSound`, not a premise;
* (b) every drawn unit's decoded view meets the Definition's checks (`TileOk`), under the named premise
  `Accounting.Assumptions.TS.TileChecks`, at sessions whose every circuit port is a port of `own`, what flock-verify's
  complete check establishes (`Flock.Registered.checkComplete_ok`).

The bound is `CheckedSessionSound`'s `δ`: an accepted proof whose rows meet `CheckedRel` meets `TileRel` on every path,
so the tile layer's unsound event is inside the checked layer's. It sits in the proofs package because the sessions do
(`reads_exempt`).
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Checked Accounting.TS.Held Accounting.TS.Tile Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A tile unit's proof layer, with its Definition's checks, is sound at `δ` through its C-Flock sessions, fully
checked.** `CheckedSessionSound`'s window and premises, at the tile layer `tileLayer sess out ok known dq dec right`
(accepted where flock-verify `--zk --registered` accepts, at the registered file's known content `known x w i`;
opening the committed rows at the registered reads; its relation `TileRel`), and in addition at every session: every
port of its circuit is a port of `own` (what `Flock.Registered.checkComplete` establishes, `checkComplete_ok`), the
known content names the port `ok` with rows each the word `1` (`okRow`), and its circuit computes `PouwH1TileZy_v1`'s
Definition at the decode `dec x w i` (`Accounting.Assumptions.TS.TileChecks`). Then the layer accepts only where
flock-verify accepts (`AcceptedLive`), and each proof is accepted with committed rows that fail `TileRel` on the path
with probability at most `δ`: outside `δ` the rows are one assignment the circuit's evaluation agrees with on every
drawn unit, at which every drawn unit's decoded cells have `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_b]`, or the
registrant's `ok` openings exhibit an explicit SHA-512 collision (`OkColl`). -/
def TileSessionSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ R C K : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (known : Paths Ω S → Fin S → Fin nπ → Array (String × Array (Array ℕ × ByteArray))) (dq : List ℤ → ℤ)
    (dec : Paths Ω S → Fin S → Fin nπ → (ℕ → ℕ → Option Bool) → ℕ → TileView R C K)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) (δ : ℝ),
    (∀ x w i, (sess x w i).CR) →
    Accounting.Assumptions.TS.WindowLaw sess out →
    (∀ x w i, (sess x w i).kd = (sess x w i).n) →
    (∀ x w i (q q' : Fin (sess x w i).c.ports.size),
      (sess x w i).c.ports[q].name = (sess x w i).c.ports[q'].name → q = q') →
    (∀ x w i u, (sess x w i).ServedRegistered u) →
    (∀ x w i, (∃ q : Fin (sess x w i).c.ports.size, (sess x w i).c.ports[q].name = ok) ∧
      ∃ pt ∈ (sess x w i).own, pt.name = ok) →
    (∀ x w i, ∀ p ∈ (sess x w i).c.ports, ∃ r ∈ (sess x w i).own, r.name = p.name) →
    (∀ x w i, ∃ ks, (ok, ks) ∈ known x w i ∧ ∀ r ∈ ks, r.1 = okRow) →
    (∀ x w i, Accounting.Assumptions.TS.TileChecks dq (sess x w i) ok (dec x w i)) →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    Accounting.Assumptions.TS.AcceptedLive (tileLayer sess out ok known dq dec right) sess out ∧
      (tileLayer sess out ok known dq dec right).SoundAt δ

end Accounting.Guarantees.TS
