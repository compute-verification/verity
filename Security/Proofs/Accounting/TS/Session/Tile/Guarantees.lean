import Proofs.Accounting.TS.Session.Tile.Computes
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
* (b) every drawn unit's view of the words its registered reads carry (`Accounting.TS.Tile.tileView` of `rd`) meets
  the Definition's checks (`TileOk`) at the Definition's own dequant (`Accounting.TS.Lanes.dqOf dqW`), under the named
  premise `Accounting.TS.Tile.TileComputes` (the circuit computes `PouwH1TileZy_v1{TILE}` there, TILE `PouwH1Tile_v1`
  or `PouwH1Tile_v3`). That every circuit port is a port of `own` is not a hypothesis: `tileLayer`'s acceptance
  includes flock-verify's complete check (`Flock.Registered.checkComplete`), from which `checkComplete_ok` reads it.
  That the Definition's `ok = 1` means `TileOk` is proved
  (`Accounting.TS.Tile.tileChecks_of_computes`, which gives `Accounting.Assumptions.TS.TileChecks`), at a `K` the
  tile's EXACT binds at (`32 ∣ K`, `K < 2^17`), which the statement takes of its `K`: the tile verifier pins each
  template's circuit by digest (`PouwCheck.tiles`) and reads no `K` it could refuse, and the served tiles' `K`s are
  in range.

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
(accepted where flock-verify `--zk --registered` accepts, at the registered file's known content `known x w i` and
under its complete check, `Flock.Registered.checkComplete`: every port of the circuit is a port of `own`; opening the
committed rows at the registered reads; its relation `TileRel`), and in addition at every session: the known content
names the port `ok` with rows each the word `1` (`okRow`), and, at a `K` the tile's EXACT binds at
(`32 ∣ K`, `K < 2^17`), its circuit computes `PouwH1TileZy_v1{TILE}`'s Definition (`Pouw.H1Zy.tileOk` at TILE's form
`form` and the dequant's word view `dqW`) on the words `rd x w i` reads from the registered reads
(`Accounting.TS.Tile.TileComputes`). Then the layer, at the Definition's dequant `dqOf dqW` and the view of those words
(`tileView`), accepts only where flock-verify accepts (`AcceptedLive`), and each proof is accepted with committed rows
that fail `TileRel` on the path with probability at most `δ`: outside `δ` the rows are one assignment the circuit's
evaluation agrees with on every drawn unit, at which every drawn unit's cells have `z = Σ_k a_k b_k` and
`y = dq [z, s_a, s_b]`, or the registrant's `ok` openings exhibit an explicit SHA-512 collision (`OkColl`). -/
def TileSessionSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ R C K : ℕ} {P : Type} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (known : Paths Ω S → Fin S → Fin nπ → Array (String × Array (Array ℕ × ByteArray)))
    (form : Pouw.H1Zy.TileForm R C K P) (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16)
    (rd : Paths Ω S → Fin S → Fin nπ → (ℕ → ℕ → Option Bool) → ℕ → Pouw.H1Zy.TileIn R C K P)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) (δ : ℝ),
    (∀ x w i, (sess x w i).CR) →
    Accounting.Assumptions.TS.WindowLaw sess out →
    (∀ x w i, (sess x w i).kd = (sess x w i).n) →
    (∀ x w i u, (sess x w i).ServedRegistered u) →
    (∀ x w i, (∃ q : Fin (sess x w i).c.ports.size, (sess x w i).c.ports[q].name = ok) ∧
      ∃ pt ∈ (sess x w i).own, pt.name = ok) →
    (∀ x w i, ∃ ks, (ok, ks) ∈ known x w i ∧ ∀ r ∈ ks, r.1 = okRow) →
    32 ∣ K → K < 2 ^ 17 →
    (∀ x w i, Accounting.TS.Tile.TileComputes form dqW (sess x w i) ok (rd x w i)) →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    Accounting.Assumptions.TS.AcceptedLive
      (tileLayer sess out ok known (Accounting.TS.Lanes.dqOf dqW)
        (fun x w i v u => Accounting.TS.Tile.tileView (rd x w i v u)) right) sess out ∧
      (tileLayer sess out ok known (Accounting.TS.Lanes.dqOf dqW)
        (fun x w i v u => Accounting.TS.Tile.tileView (rd x w i v u)) right).SoundAt δ

end Accounting.Guarantees.TS
