import Proofs.Flock.C3.Reps.TsRead
import Proofs.Flock.C3.Reps.SrcOpens
import Proofs.Flock.VBridge.Decode
import Proofs.Flock.VBridge.Verifier

/-!
# The openings' columns: in range, and read under a stage's mask

* `lt_nIn_of_params`: a unit the verifier accepted (`RecOpen.check_ok`) as `rec-open/l<LANES>-h<H>` has its first
  `128 · LANES + 512 · H` columns (the row and the `aux` blocks) among its inputs (`inWords_unit`, `unitIn_eq`), so
  `openCols_of` gives `OpenCols`' in-range half from the units' parameters (`OpenParams`, `hOpenParams_of`'s statement),
  leaving its `ReadCol` half.
* `copyVal_maskOf`: an input column a row reads (`IoColAt`) whose position is in stage `i`'s commit strings is the same
  in the masked layer: `colOf_mask`'s lemma at any `Sk`, the one both `IoColAt` and `RowColsRead` go through.
  `posK_of_key`: a column keyed by bit `k` of row port `q` is at `q`'s commit string.
* `SelPort`: the reads-file fact `skOf` membership is (`mem_skOf_inl`), at a port, a unit and a stage. `RowColsRead`
  follows (`rowColsRead_of`) from `OpenCols`, the row columns' keys (`RowKeys`, a circuit fact) and `SelPort` at the
  ports they key (`RowSel`); vacuously where no stage reads a consistency sum (`rowColsRead_of_noCons`).
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (maskOf skOf IoColAt ioColAt_of_readCol mem_skOf_inl)
open FlockVBridge.OfVStar (ReadCol)
open FlockVBridge.AccChain (levelLanes)

/-- Every circuit the verifier parsed passes `RecOpen.check` (`ParseFacts.recNet`). -/
theorem recCheck {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg') : Flock.RecOpen.check Q.c = .ok () :=
  (parse_facts zo.hc).recNet

/-- **An opening's row and `aux` columns are inputs**: the unit is `unitLaid LANES H`, whose input bits
(`unitIn`: `128 · LANES` row bits, then at least `512 · H` of `aux`) are below its inputs' end. -/
theorem lt_nIn_of_params {Q : VStmt} (hrec : Flock.RecOpen.check Q.c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params Q.c.unit.name = some (LANES, H)) {c : ℕ} (hc : c < 128 * LANES + 512 * H) :
    c < (Rows.ofNet Q.hU).nIn := by
  rw [Layout.ofNet_nIn_eq, FlockVBridge.inWords_unit (Flock.RecOpen.check_ok hrec hp).2]
  refine Nat.lt_of_lt_of_le ?_ (Flock.HmNets.le_groupWords _)
  rw [FlockVBridge.foldl_wordPorts, FlockVBridge.unitIn_eq]
  have := Nat.le_max_right 1 ((H + 1) / 2)
  omega

/-- **An input column a row reads, at a position of stage `i`'s commit strings, is the same in the masked layer.** -/
theorem copyVal_maskOf {S : ℕ} {Vs : Fin S → VStmt} {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) (j : Fin S) (u : Fin (Vs j).nv) (hj : 0 < (Vs j).c.upv) {c : ℕ}
    (hc : c < (Rows.ofNet (Vs j).hU).nIn) (hio : IoColAt (Vs j) u c)
    (hsk : ∀ hcw : c < (Rows.ofNet (Vs j).hU).w, posK (Vs j).c (Vs j).hU (Vs j).nv
      ((FlockVBridge.keyInst (Vs j).hU (Vs j).nv (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u).wire
        ((Rows.ofNet (Vs j).hU).stackCol (Vs j).c.upv hj ⟨c, hcw⟩)) ∈ Sk i j) :
    FlockVBridge.copyVal (FlockVBridge.keyInst (Vs j).hU (Vs j).nv (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u)
        (maskOf Vs Sk i Xs j) hj c =
      FlockVBridge.copyVal (FlockVBridge.keyInst (Vs j).hU (Vs j).nv (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u)
        (Xs j) hj c := by
  refine (FlockVBridge.copyVal_input _ _ hj hc).trans (Eq.trans ?_ (FlockVBridge.copyVal_input _ _ hj hc).symm)
  simp only [maskOf]
  exact ite_eq_left ⟨hsk _, u, hio hj _⟩

/-- **A column keyed by bit `k` of row port `q` is at `q`'s commit string** (`keyRowOf_keyInst`, `rowOfH_of_key`). -/
theorem posK_of_key {Q : VStmt} (u : Fin Q.nv) (hj : 0 < Q.c.upv) {c : ℕ} (hc : c < (Rows.ofNet Q.hU).nIn)
    {q : Fin Q.c.ports.size} {k : ℕ} (hk : k < Layout.portRowW Q.c q)
    (hkey : Aliased.srcKey Q.c (0 * (Rows.ofNet Q.hU).nIn + c) = .msg q.val (k / 8) (k % 8))
    (hcw : c < (Rows.ofNet Q.hU).w) :
    posK Q.c Q.hU Q.nv ((FlockVBridge.keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).wire
      ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨c, hcw⟩)) = .inl (u, q) := by
  have hx : ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨c, hcw⟩).val = 0 * (Rows.ofNet Q.hU).nIn + c := by
    rw [Rows.stackCol, dite_eq_left hc]
  have hr := HiddenExec.rowOfH_of_key (Layout.outsOf Q.c Q.hU Q.nv)
    (FlockVBridge.Decode.keyRowOf_keyInst (Layout.outsOf Q.c Q.hU Q.nv) u
      ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨c, hcw⟩) (by rw [hx]; exact Rows.mul_add_lt hj hc) hk
      (by rw [hx]; exact hkey))
  simp only [posK, hr]

/-- **Stage `s` reads port `q` of unit `u` of session `j`** (`skOf`'s condition, `mem_skOf_inl`): `j`'s reads file has
an entry named as the port whose read of the unit is at a position `x` of a value `sel s` selects. H's `RsRead` and C's
`WCombRead` ask the same of the reads file, at round `c₀ + i`'s `rec-opening-c` and round `j`'s `rec-c` ports. -/
def SelPort {S : ℕ} (Vs : Fin S → VStmt) (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (s : ℕ) (j : Fin S) (u : Fin (Vs j).nv) (q : Fin (Vs j).c.ports.size) : Prop :=
  ∃ p ∈ own j, p.name = (Vs j).c.ports[q].name ∧ ∃ x, p.reads[u.val]? = some (some x) ∧ sel s p.value x

variable {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} {sh : Flock.Shape}

namespace RepMaps

variable (M : RepMaps stmt Vs sh)

/-- **The openings' units' parameters** (`hOpenParams_of`'s statement at the maps, from `HOpenName` and the statements):
level `ℓ`'s unit is `rec-open/l<levelLanes>-h<d - cap>`. -/
def OpenParams : Prop :=
  ∀ ℓ < sh.sched.levels.size, Flock.RecOpen.params (Vs (M.sessO ℓ)).c.unit.name =
    some (levelLanes sh ℓ, (sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap)

/-- **`OpenCols` from the units' parameters and the `ReadCol` half.** -/
theorem openCols_of (hrec : ∀ j, Flock.RecOpen.check (Vs j).c = .ok ()) (hP : M.OpenParams)
    (hR : ∀ ℓ < sh.sched.levels.size, ∀ c < 128 * levelLanes sh ℓ +
      512 * ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap), ReadCol (Rows.ofNet (Vs (M.sessO ℓ)).hU) c) :
    M.OpenCols sh.sched := by
  intro ℓ hℓ c hc
  rw [FlockVBridge.Family.Conss.getD_default] at hc
  exact ⟨lt_nIn_of_params (hrec _) (hP ℓ hℓ) hc, hR ℓ hℓ c hc⟩

/-- Obligation: discharged by the openings' circuits (`RecOpen.check_ok`: the unit is `unitLaid`, whose first port is
`row`), as `PartRowCols` for the parts. **The openings' row columns are keyed by row bits**: at each level, the Δ key of
each of copy 0's first `128 · levelLanes` columns is message bit `k` of a port `q`, within its row. -/
def RowKeys : Prop :=
  ∀ ℓ < sh.sched.levels.size, ∀ t < 128 * levelLanes sh ℓ,
    ∃ (q : Fin (Vs (M.sessO ℓ)).c.ports.size) (k : ℕ) (_ : k < Layout.portRowW (Vs (M.sessO ℓ)).c q),
      Aliased.srcKey (Vs (M.sessO ℓ)).c (0 * (Rows.ofNet (Vs (M.sessO ℓ)).hU).nIn + t) = .msg q.val (k / 8) (k % 8)

/-- **Hypothesis on the reads file: where stage `s`'s keys read a sum of level `ℓ`, stage `s` reads the ports keying the
row columns of rep `ρ`'s level-`ℓ` openings** (`SelPort`). -/
def RowSel (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2) ℓ l, K s ρ (.cons ℓ l) → ∀ j < M.q ℓ, ∀ t < 128 * levelLanes sh ℓ,
    ∀ (q : Fin (Vs (M.sessO ℓ)).c.ports.size) (k : ℕ), k < Layout.portRowW (Vs (M.sessO ℓ)).c q →
      Aliased.srcKey (Vs (M.sessO ℓ)).c (0 * (Rows.ofNet (Vs (M.sessO ℓ)).hU).nIn + t) = .msg q.val (k / 8) (k % 8) →
        SelPort Vs own sel s (M.sessO ℓ) (M.unO ℓ ρ j) q

/-- **`RowColsRead` from the openings' columns, their keys and the reads file**: at each copy's row column, both layers
equal their masks there (`copyVal_maskOf`), and the masks are equal. -/
theorem rowColsRead_of {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop} (hK : ∀ s ρ ℓ l, K s ρ (.cons ℓ l) → ℓ < sh.sched.levels.size)
    (hk : M.OpenCols sh.sched) (hkey : M.RowKeys) (hsel : M.RowSel own sel K) : RowColsRead own sel M K := by
  intro s ρ ℓ l hc
  have hℓ := hK s ρ ℓ l hc
  refine ⟨hℓ, fun j hjq hj Lb Lb' he t ht => ?_⟩
  have hb : t < 128 * 2 ^ (sh.sched.levels.getD ℓ default).lanes +
      512 * ((sh.sched.levels.getD ℓ default).d - (sh.sched.levels.getD ℓ default).cap) := by
    rw [FlockVBridge.Family.Conss.getD_default]
    unfold levelLanes at ht
    omega
  obtain ⟨hn, hr⟩ := hk ℓ hℓ t hb
  obtain ⟨q, k, hkq, hkey'⟩ := hkey ℓ hℓ t ht
  have hm : ∀ X : (x : Fin S) → (Vs x).Bits,
      colsAt (Vs (M.sessO ℓ)) (maskOf Vs (skOf Vs own sel) s X (M.sessO ℓ)) (M.unO ℓ ρ j) hj t =
        colsAt (Vs (M.sessO ℓ)) (X (M.sessO ℓ)) (M.unO ℓ ρ j) hj t := fun X =>
    copyVal_maskOf s X (M.sessO ℓ) (M.unO ℓ ρ j) hj hn (ioColAt_of_readCol _ hn hr) fun hcw => by
      rw [posK_of_key (M.unO ℓ ρ j) hj hn hkq hkey' hcw]
      exact mem_skOf_inl.2 (hsel s ρ ℓ l hc j hjq t ht q k hkq hkey')
  rw [← hm Lb, ← hm Lb', he]

/-- **`RowColsRead` where no stage's keys read a consistency sum.** -/
theorem rowColsRead_of_noCons {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop} (hK : ∀ s ρ ℓ l, ¬ K s ρ (.cons ℓ l)) : RowColsRead own sel M K :=
  fun s ρ ℓ l hc => absurd hc (hK s ρ ℓ l)

end RepMaps

end

end FlockC3.Reps
