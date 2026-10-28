import Proofs.Flock.Claims.Recursion.SideCapUp
import Proofs.Flock.Claims.Recursion.DecompBDeployed
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.RowDLevels
import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.Claims.Complete.Replay.LigeritoMerkle
import Proofs.Flock.Recursive.HoloRepCoins

/-!
# The opened positions' range (`PosCap0`, `PosCapUp`) at row D's pieces and `Rows.posAt`

`sideCapR_deployed` gives `SideCapR` at the deployed layout from `ReadsAt` and two ranges: a level-`ℓ`
piece's position, below its `HH` siblings, is one of its cap's nodes. Both hold at every decision, from its own checks:

* `posAt` is the position the verifier rows draw (`verifierRows`) at the statement's own shape, its table's `fast100`
  schedule. Those rows run at no empty round (`canonRounds_noEmpty`: each round reads its schedule's coin count, and
  `repRounds` counts none empty), at `m ≥ 22` and a schedule of the seams, so level `ℓ`'s positions are Flock's
  `positions` at that level (`FlockVBridge.Family.rows_pos`), each below `2^d` (`Lg.positions_lt_d`). Where the rows
  don't run, or the query is past them, `posAt` is 0 (`posAt_lt`).
* at a piece `ownQR` gives the level part `.recOpen ℓ LANES HH n` of a dense statement, `HH = d - cap` at table 0's
  schedule (`levelMatch`), so the position shifted by `HH` is below `2^cap` (`shift_lt`; `pieceCtx`).
* level 0's cap holds `level0Nodes = 2^cap₀` nodes (`capAt_level0`), and the schedule's `(ℓ - 1)`-th top, level `ℓ`'s,
  `2^cap_ℓ` (`topSize_cap`); it exists, since `repRounds` commits one top a level past level 0 (`topSizes_length`).

`sideCapR_rowD` is `SideCapR` at row D's pieces and `Rows.posAt`, at the deployed layout and bound record, from
`ReadsAt` alone, and `sideCapR_leaf` is `recursiveKSAt_of_leaves`' `hCap` at its records.
-/

namespace RecDeploy.CapLower

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open RecDeploy.OpenLower (AtPieces ownQR ownQR_some schAt levelMatch inner_m recOpen_dense)

/-! ## Arithmetic -/

/-- A position below `2^d`, shifted past `d - c` siblings, is below `2^c`. -/
theorem shift_lt {p d c : ℕ} (h : p < 2 ^ d) : p >>> (d - c) < 2 ^ c := by
  rw [Nat.shiftRight_eq_div_pow]
  by_cases hc : c ≤ d
  · apply Nat.div_lt_of_lt_mul
    rw [← Nat.pow_add, Nat.sub_add_cancel hc]
    exact h
  · have e : d - c = 0 := by omega
    rw [e, Nat.pow_zero, Nat.div_one]
    exact Nat.lt_of_lt_of_le h (Nat.pow_le_pow_right (by decide) (by omega))

/-! ## The rows' position -/

theorem repLens_getD_le (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) (k : ℕ) :
    (FlockDeploy.repLens spec sts).getD k 0 ≤ (RecDeploy.repSchedAt spec sts[k / 2]!).length := by
  unfold FlockDeploy.repLens
  rw [List.getD_eq_getElem?_getD]
  by_cases hk : k < 2 * sts.size
  · rw [List.getElem?_map, List.getElem?_range hk]
    simp
  · rw [List.getElem?_eq_none (by simpa using hk)]
    simp

/-- Every round of a table's rep schedule reads a coin: `repRounds` at the session's `2 + points · regions > 0`
claims (`repRounds_coins`). -/
theorem repSchedAt_pos (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) :
    ∀ x ∈ RecDeploy.repSchedAt spec st, 0 < x.1 := by
  intro x hx
  unfold RecDeploy.repSchedAt at hx
  cases h : RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size) with
  | none => rw [h] at hx; simp at hx
  | some out =>
    rw [h] at hx
    simp only [Option.map_some, Option.getD_some] at hx
    exact Flock.Guarantees.HoloRepCoins.repRounds_coins (by omega) h x (Array.mem_toList_iff.1 hx)

/-- **No coin round of a decision is empty**: rep `k`'s round `j` reads its schedule's `n_j > 0` values. -/
theorem canonRounds_noEmpty (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) (cs : List Coin) (k : ℕ) :
    Flock.RecAlgebra.NoEmpty (Rows.canonRounds spec sts cs k) := by
  intro r hr
  unfold Rows.canonRounds at hr
  obtain ⟨j, hj, rfl⟩ := Array.mem_map.1 hr
  rw [Array.mem_range] at hj
  rw [FlockDeploy.Coin.vals, Array.size_map, Array.size_range]
  have hjl : j < (RecDeploy.repSchedAt spec sts[k / 2]!).length := Nat.lt_of_lt_of_le hj (repLens_getD_le spec sts k)
  have hmem : (RecDeploy.repSchedAt spec sts[k / 2]!).getD j default ∈ RecDeploy.repSchedAt spec sts[k / 2]! := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjl, Option.getD_some]
    exact List.getElem_mem hjl
  exact Nat.pos_iff_ne_zero.1 (repSchedAt_pos spec _ _ hmem)

theorem positions_getElem!_lt {lv : _root_.Flock.Level} (hl : FlockSoundness.Refine.LevelSeam lv) (coins : Array F128)
    (j : ℕ) :
    (_root_.Flock.positions lv coins)[j]! < 2 ^ lv.d := by
  by_cases hj : j < (_root_.Flock.positions lv coins).size
  · rw [getElem!_pos _ j hj]
    exact FlockComplete.Replay.Lg.positions_lt_d hl coins _ (Array.getElem_mem hj)
  · rw [getElem!_neg _ j hj]
    exact Nat.two_pow_pos _

/-- **The position a piece reads is below `2^d`** of its level at its table's schedule: where the statement sets up
and the table's `m` has a `fast100` schedule, the verifier rows draw level `ℓ`'s positions as Flock's `positions`
(`rows_pos`, at no empty round, `m ≥ 22` and the seams), each below `2^d`; elsewhere `posAt` is 0. -/
theorem posAt_lt {x : Msg} {cs : List Coin} {q : OpenPiece} {v}
    (hset : FlockDeploy.setupOf false x = some v) {sch : _root_.Flock.Schedule}
    (hf : fast100 (FlockDeploy.tabsOf v.2.2)[q.table]!.m = some sch) (hl : q.level < sch.levels.size) :
    Rows.posAt x cs q < 2 ^ (sch.levels[q.level]!).d := by
  have hlv := FlockSoundness.Refine.fast100_level _ _ hf
  have hseam : FlockSoundness.Refine.LevelSeam (sch.levels[q.level]!) := by
    rw [getElem!_pos sch.levels q.level hl]
    exact hlv _ (Array.getElem_mem hl)
  unfold Rows.posAt
  rw [hset]
  dsimp only
  rw [hf, Option.getD_some]
  split
  · next _ info _ h =>
    have hds := FlockSoundness.Refine.fast100_dims _ _ hf
    have hm := FlockSoundness.Refine.exec_fast100_ge hf
    have hlv' : ∀ i < sch.levels.size, FlockSoundness.Refine.LevelSeam (sch.levels.getD i default) := fun i hi => by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
      exact hlv _ (Array.getElem_mem hi)
    rw [FlockVBridge.Family.rows_pos (canonRounds_noEmpty _ _ _ _)
      (by show 14 ≤ (FlockDeploy.tabsOf v.2.2)[q.table]!.m; omega) hds.1 hlv' h q.level hl]
    exact positions_getElem!_lt hseam _ _
  · exact Nat.two_pow_pos _

/-! ## A piece's level -/

/-- **What a piece's level reads**, at a dense statement where `ownQR` gives row D's piece `q` the level part
`.recOpen lv LANES HH n`: the inner `m` has a `fast100` schedule, `q`'s level is one of its levels, `q` is one of
the table's two reps, `HH = d - cap` at that level (`levelMatch`), and the position `q` reads is below `2^d`
(`posAt_lt`, at table 0's setup, `inner_m`). -/
theorem pieceCtx {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) {x : Msg} {st : Claims.Recursion.Stmt}
    (hx : D.parse x = some st) (hf : shape st.form) {r : XR} {ss : Array VStar.Stmt} {s lv LANES HH n : ℕ}
    (hr : readX x = some r) (hss : RecVStar.statements r.shape = .ok ss)
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {q : OpenPiece}
    (hq : q ∈ Rows.rowDPieces key (Claims.Recursion.xI D x)) (ho : ownQR x q = some s) :
    ∃ sch : _root_.Flock.Schedule, fast100 r.shape.m = some sch ∧ r.shape.holo = false ∧ q.level < sch.levels.size ∧
      q.rep < 2 ∧ HH = (sch.levels[q.level]!).d - (sch.levels[q.level]!).cap ∧
      ∀ cs, Rows.posAt (Claims.Recursion.xI D x) cs q < 2 ^ (sch.levels[q.level]!).d := by
  have hx' : parse x = some st := (hP x).symm.trans hx
  have hd := hS _ hf
  have hxI : Claims.Recursion.xI D x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP x]
  obtain ⟨h0q, r2, ss2, hr2, hss2, L2, H2, n2, hsv2⟩ := ownQR_some ho
  rw [hr] at hr2
  cases hr2
  rw [hss] at hss2
  cases hss2
  rw [hsv] at hsv2
  simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv2
  obtain ⟨hlvq, -, -, -⟩ := hsv2
  obtain ⟨hls, -, hHH, -⟩ := levelMatch hx' hd hr hss hsv
  have hql : q.level = s := hlvq.symm.trans hls
  obtain ⟨hrep, hqry⟩ := OpenLower.piecesInRangeR_rowDPieces key _ q hq
  rw [hxI, h0q] at hqry
  obtain ⟨_, _, t, hset, htm⟩ := inner_m hx' hr
  obtain ⟨r', hr', -, rfl⟩ := parse_read hx'
  rw [hr] at hr'
  cases hr'
  have hh : r.shape.holo = false := hd.1
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hd.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  obtain ⟨sch, hsch, -, -, -, -⟩ := recOpen_dense hss hsp hh hsv
  have h0 : (FlockDeploy.tabsOf #[t])[0]!.m = t.m := by
    have htabs : FlockDeploy.tabsOf #[t] = #[_root_.Flock.Setup.ofCircuit t] := by simp [FlockDeploy.tabsOf]
    rw [htabs]
    rfl
  have e : schAt (xIOf x) 0 = sch := by
    unfold schAt
    rw [hset]
    dsimp only
    rw [h0, htm, hsch]
    rfl
  rw [e] at hqry hHH
  have hlev : q.level < sch.levels.size := by
    by_contra hge
    rw [getElem!_neg _ q.level hge] at hqry
    exact Nat.not_lt_zero _ hqry
  have hf' : fast100 (FlockDeploy.tabsOf #[t])[q.table]!.m = some sch := by rw [h0q, h0, htm, hsch]
  refine ⟨sch, hsch, hh, hlev, hrep, by rw [hHH, hql], fun cs => ?_⟩
  rw [hxI]
  exact posAt_lt hset hf' hlev

/-! ## The levels' tops -/

/-- **`repRounds` commits one top a level past level 0**: `topSizes` is `schedPat`'s tops (`repRounds_flagTop`),
level 1's then each later level's (`schedPat_tops`). -/
theorem topSizes_length {m kLog claims k : ℕ} {sch : _root_.Flock.Schedule} {rs : Array RecVStar.SchedRound}
    (hs : fast100 m = some sch) (h : RecVStar.repRounds m kLog claims = some rs) (hk : sch.levels.size - 1 = k + 1) :
    (RecVStar.topSizes rs).length = k + 1 := by
  have e : RecVStar.topSizes rs = ((rs.toList.map FlockDeploy.flagTop).map Prod.snd).flatten := by
    unfold RecVStar.topSizes
    rw [List.flatMap_def, List.map_map]
    rfl
  rw [e, FlockDeploy.repRounds_flagTop hs h, FlockDeploy.schedPat_tops hk]
  simp

/-! ## The ranges -/

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (rd : Msg → Option FinalP)

/-- **`PosCap0` at row D's pieces and `Rows.posAt`**: a level-0 position is below `2^d₀`, so past its
`HH = d₀ - cap₀` siblings below `2^cap₀ = level0Nodes`, RecCap's node count (`capAt_level0`). -/
theorem posCap0_rowD (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) :
    PosCap0 H D shape untr b cmt L Bd (Rows.rowDPieces key) Rows.posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht hl0
  obtain ⟨sch, hsch, -, hlev, hrep, hHH, hpos⟩ := pieceCtx key hP hS hx hf hr hss hsv hq ho
  refine ⟨hrep, fun r' i N hr' hcap => ?_⟩
  rw [hr] at hr'
  cases hr'
  obtain ⟨_, _, t, -, htm⟩ := inner_m ((hP x).symm.trans hx) hr
  rw [RestRead.capAt_level0 hr htm] at hcap
  cases hcap
  have h0 : 0 < sch.levels.size := by rw [← hl0]; exact hlev
  have hg : sch.levels.getD 0 default = sch.levels[q.level]! := by
    rw [hl0, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h0, Option.getD_some,
      getElem!_pos sch.levels 0 h0]
  have hN : level0Nodes t = 2 ^ (sch.levels[q.level]!).cap := by
    unfold level0Nodes
    rw [htm, hsch]
    simp only [Option.map_some, Option.getD_some, hg]
  rw [hN, hHH]
  exact shift_lt (hpos _)

/-- **`PosCapUp` at row D's pieces and `Rows.posAt`**: a level-`ℓ ≥ 1` position is below `2^d_ℓ`, so past its
`HH = d_ℓ - cap_ℓ` siblings below `2^cap_ℓ`, the schedule's `(ℓ - 1)`-th top (`topSize_cap`), which exists
(`topSizes_length`). -/
theorem posCapUp_rowD (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (hP : ∀ x, D.parse x = parse x)
    (hS : ∀ f, shape f → Claims.Recursion.Form.Dense f) :
    PosCapUp D shape untr b cmt L Bd H (Rows.rowDPieces key) Rows.posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht hl0
  obtain ⟨sch, hsch, hh, hlev, hrep, hHH, hpos⟩ := pieceCtx key hP hS hx hf hr hss hsv hq ho
  refine ⟨hrep, fun r' rs hr' hrs => ?_⟩
  rw [hr] at hr'
  cases hr'
  have hrs' : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs := by
    simpa [repSched, hh] using hrs
  have hk : sch.levels.size - 1 = (sch.levels.size - 2) + 1 := by omega
  have hlen := topSizes_length hsch hrs' hk
  have hg : q.level - 1 < (RecVStar.topSizes rs).length := by omega
  obtain ⟨nn, hnn⟩ : ∃ nn, (RecVStar.topSizes rs)[q.level - 1]? = some nn := ⟨_, List.getElem?_eq_getElem hg⟩
  obtain ⟨s', hs', rfl⟩ := topSize_cap hr hrs hnn
  rw [hsch] at hs'
  cases hs'
  refine ⟨_, hnn, ?_⟩
  rw [Nat.sub_add_cancel (by omega : 1 ≤ q.level), hHH]
  exact shift_lt (hpos _)

/-- **`SideCapR` at row D's pieces and `Rows.posAt`**, at SHA-512, the deployed layout `layR elem` and the bound
record at `finalR`: `sideCapR_deployed` from `ReadsAt` and the two ranges, which every decision's own
checks give (`posCap0_rowD`, `posCapUp_rowD`). -/
theorem sideCapR_rowD (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out) (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Msg → ℝ) (rowIn : Msg → Msg) (w : Msg → ℝ) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hP : ∀ x, D.parse x = parse x) (hN : ∀ x, D.nS x = nS x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hRd : RestRead.ReadsAt rd DeployedR.finalR (DeployedR.restR elem)) :
    SideCapR FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) rd (Rows.rowDPieces key) Rows.posAt :=
  sideCapR_deployed D shape untr b cmt rd elem rel nextE colsE wE rowIn w (Rows.rowDPieces key) Rows.posAt hP hN hD
    hRd (posCap0_rowD FlockDeploy.sha512 D shape untr b cmt _ _ key hP hD)
    (posCapUp_rowD FlockDeploy.sha512 D shape untr b cmt _ _ key hP hD)

end Deployed

/-- **`recursiveKSAt_of_leaves`' `hCap`**, at its records: `sideCapR_rowD` at `elemR`, `rdR`, the compiled record
`recL` and the bound record `bdR`, from `ReadsAt` alone. -/
theorem sideCapR_leaf (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hRd : RestRead.ReadsAt DeployedR.rdR DeployedR.finalR (DeployedR.restR OpenRows.elemR)) :
    SideCapR FlockDeploy.sha512
      (DecompBR.recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (DecompBR.bB b aux nOf hd recRest) (cmtTr recRead)
      (DeployedR.layR OpenRows.elemR) (bdR DeployedR.rdR wE DeployedR.finalR w) DeployedR.rdR (Rows.rowDPieces key)
      Rows.posAt :=
  sideCapR_rowD _ _ _ _ _ DeployedR.rdR OpenRows.elemR _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols wE rowInDeployed w key
    (fun _ => rfl) (fun _ => rfl) (fun _ h => h) hRd

end RecDeploy.CapLower
