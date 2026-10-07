import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup

/-!
# The challenger's draw at every draw of the live law, at hidden-output `--zk`

`zk_session_soundH` takes `DrawOkZKH I dj mPts` at each of `Law.execOS`'s draws: the draw file `dj S` proves `S`'s units,
and setupHidden, under the `--zk` tags, sets up at `mPts` link points. As at public outputs (`ZkLink/DrawOk.lean`), the
first half is custody (`DrawFileZK`) and the second is flock-verify's own arithmetic.

* `setupHidden_mPts_drawn`, `setupHidden_mPts_of_proved`: setupHidden's `m_pts` at a draw is `mPtsOf` of the circuit's
  `g` and the units the draw proves, as setupH's (`Exec.setupH_mPts_drawn`, `Exec.setupH_mPts_of_proved`), from
  `Hidden.setupHidden_spec`.
* `drawOkZKH_of_exec`: from `DrawFileZK` at the live law's own stratified law, `DrawOkZKH` at every draw of
  `Law.execOS`, at `mPtsOf c.g (∑ s, ks s)`: `ZkLink.drawOkZK_of_exec`'s proof at setupHidden.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Exec ZkExec ZkLink HiddenExec Finset FlockSoundness.Refine
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt}

/-- **`m_pts` at a drawn hidden-output session** is `mPtsOf` of the circuit's `g` and the number of units the draw
proves. -/
theorem setupHidden_mPts_drawn {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ ud, Draw.UnitDraw.ofJson d = .ok ud ∧ st.mPts = mPtsOf st.c.g ud.proved.size := by
  have s := Hidden.setupHidden_spec h
  obtain ⟨pop, -, hd⟩ := setupHidden_drawn h
  obtain ⟨ud, hud, hn⟩ := drawn_n (c := { st.c with outPorts := #[] }) hd
  refine ⟨ud, hud, ?_⟩
  rw [s.mPts, s.nbl, s.blocksReal, s.n, hn]
  rfl

/-- **`m_pts` from the circuit and the draw's proved count**: at a draw `d` whose decoding proves `N₀` units, every
statement `setupHidden` accepts has `m_pts = mPtsOf c.g N₀`, `c` the circuit file's `HmOut.parse`. -/
theorem setupHidden_mPts_of_proved {c : Circuit} {d : Json} {N₀ : ℕ} (hc : HmOut.parse tags cf tables = .ok c)
    (hD : ∀ D, Draw.UnitDraw.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf c.g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupHidden_mPts_drawn h
  have e := setupHidden_parseC h
  rw [hc] at e
  rw [hm, ← Except.ok.inj e, hD ud hud]

variable {n : ℕ}

/-- **The challenger's draw is `DrawOkZKH` at every draw of the live law.** On tags whose circuit file `HmOut.parse`
accepts under the `--zk` tags as `c`, with the executable strata `St` describing `σs`'s, each drawing `ks s` of its
units: when the record's draw file is the verifier's draw under the live law's own stratified law (`DrawFileZK`),
`DrawOkZKH` holds at `mPtsOf c.g (∑ s, ks s)` link points at every draw of `Law.execOS`, the run that reads past the
source's bytes (every unit) included. -/
theorem drawOkZKH_of_exec (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Audit.Law.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Audit.Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    DrawOkZKH I dj (mPtsOf c.g (∑ s, ks s))
      ((Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) := by
  refine ⟨fun D hD u hu => ?_, fun st hst => ?_⟩
  · rw [hdj _ D hD, drawOf_proved, drawOf_mem]
    exact ⟨u, hu, rfl⟩
  · rw [setupHidden_mPts_of_proved hc (fun D hD => by rw [hdj _ D hD, drawOf_proved, drawOf_size]) hst]
    rcases execOS_draw_card_or σs ks St hS Kd pop src ω with hcard | huniv
    · rw [hcard]
    · obtain ⟨ud, hud, -⟩ := setupHidden_mPts_drawn hst
      rw [hdj _ ud hud, huniv] at hud
      rw [huniv, card_univ, Fintype.card_fin, sum_ks_of_univ σs ks hks St hS Kd pop hud]

end FlockSoundness.Discharge.ZkHidden
