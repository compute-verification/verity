import Proofs.Flock.Claims.Recursion.VerdictRowECoin
import Proofs.Flock.Claims.Recursion.LayoutBAcc
import Proofs.Flock.Claims.Recursion.RowSlots
import Proofs.Flock.Claims.Recursion.OpenPieces
import Proofs.Flock.VBridge.AccUnits
import Proofs.Flock.Soundness.Discharge.Hm.Row

/-!
# Row E's running sums at a level copy, through the chain's commits

`Top.targetE_rowD` (`VerdictRowECoin`) leaves `hholds`: each `rec-open` piece's next chain row is its opening's
consistency terms of its row, the chain's row at its position and its `coef` (`accHolds`). A level's copy computes
those terms on its `acc_out` output (`LevelOuts`: `OpenOuts`' bytes from `outCol + 1024`), and V\* registers `acc_out`
at the chain's row `p + 1` (`rec_outer.stage`'s reads file: `acc_out` reads `rec-acc` at `p + 1`, as `acc` reads it at
`p`). `layoutBAcc` reads row `p` as its owner holds it (`chainAt_held`), and `acc_out` is no holding there, since
`rowHidden` reads input columns only. This file ties the copy's columns to the chain through the rows' commits, the
argument of `rowsAgree_of_commits`, with `acc_out` read at its recorded row call, not as a slot:

* `VStarB.commit_agree`: two recorded evaluations of a frame of `v` and of `v'` whose digests climb to one root, with
  every climb input among the evaluations and no collision among them, have `v = v'` (`rowsAgree_of_commits`' core:
  `VStar.chain_coll`, then the frame's injectivity).
* `chainAt_commit`: where bytes `b` are committed under row `p`'s registered root, as row `p`'s owner's holding is,
  `chainAt rest` reads `b`'s first two elements at row `p`.
* `elemsOf_accOut`, `openAcc_of_cols`: a level copy's `acc_out` columns carry its consistency terms, and its `acc`
  columns carry `openAcc`, as element reads of the bytes on those columns.
* `accHolds_of_copy`: a piece holds (`accHolds`) at a copy whose outputs are `LevelOuts`, which reads the piece's
  opened row (`OpenCopy`'s row tie) and `coef`, where the chain's rows at the piece's position and the next are the
  bytes on the copy's `acc` and `acc_out` columns.
* `accHolds_layoutBAcc`: the same at `chainAt rest` and the readers' messages, where those bytes are committed under
  the two rows' roots (`chainAt_commit` twice), with no collision among the evaluations `l`.

The commits' premises are `rowsAgree_of_commits`' at the chain's rows, plus the copy's own two: its `acc` and `acc_out`
rows' recorded evaluations, in `l`. `layoutBAcc` records the chain ports' hm96 evaluations (`accCalls`), so they come
off its calls:

* `accFrame`, `accFrame_inj`: a row's hm96 input, its `sha512/row/v1` prefix then the row, injective in the row.
* `accCommit_layoutBAcc`: every held slot's row evaluation is among its holder's calls, to the slot's digest
  (`accDig`), which is `rowsAgree_of_commits`' `hC`; `rowsAgree_layoutBAcc` is `rowsAgree_of_commits` at `layoutBAcc`.
* `accHolds_layoutBAcc_recorded`: `accHolds_layoutBAcc` with its four commit premises off the calls, where the copy's
  bytes are what its witness reads at its `acc` holding and its `acc_out` row.

What stays open is the climbs: each digest's climb to its row's registered root, its inputs among the evaluations, and
the climbs' steps apart (`VStar.StepsApart`).
-/

namespace Claims.Recursion.VStarB

open Claims (Msg)

/-- **Two commits to one root commit one value, or two evaluations collide**: where `l`'s calls are true evaluations
with no collision among them, `l` holds an evaluation of `frame v σ` to `d` and one of `frame v' σ'` to `d'`, the two
digests climb to one root along climbs whose steps keep values apart and whose inputs are among `l`'s, and `frame` is
injective in the value, `v = v'` (`rowsAgree_of_commits`' argument at one pair). -/
theorem commit_agree {H : Msg → Msg} {l : List (Msg × Msg)} (hT : ∀ e ∈ l, H e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {frame : Msg → Msg → Msg}
    (hinj : ∀ v σ v' σ', frame v σ = frame v' σ' → v = v') {v σ d v' σ' d' root : Msg}
    (h1 : (frame v σ, d) ∈ l) (h2 : (frame v' σ', d') ∈ l) {fs gs : List (Msg → Msg)}
    (hapart : VStar.StepsApart fs gs) (hc1 : VStar.chainH H fs d = root) (hc2 : VStar.chainH H gs d' = root)
    (hi1 : ∀ u ∈ VStar.chainIns H fs d, u ∈ l.map Prod.fst)
    (hi2 : ∀ u ∈ VStar.chainIns H gs d', u ∈ l.map Prod.fst) : v = v' := by
  have hcoll : ∀ u u', u ∈ l.map Prod.fst → u' ∈ l.map Prod.fst → u ≠ u' → H u = H u' → False :=
    fun u u' hu hu' hne hH => by
      rw [Claims.findColl_isSome_mem H l hT u u' hu hu' ⟨hne, hH⟩] at hnc
      exact absurd hnc (by decide)
  have hd : d = d' := by
    by_contra hne
    obtain ⟨u, hu, u', hu', hne', hH⟩ := VStar.chain_coll H hapart hne (hc1.trans hc2.symm)
    exact hcoll u u' (hi1 u hu) (hi2 u' hu') hne' hH
  refine hinj v σ v' σ' (Classical.byContradiction fun hne => ?_)
  exact hcoll _ _ (List.mem_map.2 ⟨_, h1, rfl⟩) (List.mem_map.2 ⟨_, h2, rfl⟩) hne
    ((hT _ h1).trans (hd.trans (hT _ h2).symm))

end Claims.Recursion.VStarB

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Hm (bitL)
open FlockSoundness.Refine (openAt)
open FlockVBridge (ColsHold)
open FlockVBridge.Family (outCol plain)
open RecDeploy.Rows (accHolds chainPos)

/-! ## A level copy's `acc` and `acc_out` columns as elements -/

theorem pair_elemsOf {bs : List UInt8} {k : ℕ} (hk : k < 2) :
    (#[elemsOf bs 2 0, elemsOf bs 2 1] : Array F128)[k]! = elemsOf bs 2 k := by
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl

/-- **Bytes on the `acc_out` half of a copy's outputs are its consistency terms**: where the columns `v` from `base`
hold 128 bytes `pre`, then the bytes of `#[c0, c1]` and `rest` (`OpenOuts`' order), and the bytes `b`, of at least two
elements, sit from `base + 1024`, `b`'s first two elements are `c0, c1` (`acc_of_cols`). -/
theorem elemsOf_accOut {v : ℕ → Bool} {base : ℕ} {pre rest : ByteArray} (hX : pre.size = 128) {c0 c1 : F128}
    (hcols : ColsHold v base (pre ++ (MerkleScheme.rowBytes #[c0, c1] ++ rest))) {b : ByteArray}
    (hb : ColsHold v (base + 1024) b) (hsz : 32 ≤ b.size) {k : ℕ} (hk : k < 2) :
    elemsOf b.data.toList 2 k = (#[c0, c1] : Array F128)[k]! := by
  have hout : ∀ t < 256, v (base + 1024 + t) =
      bitL (MerkleScheme.rowBytes #[elemsOf b.data.toList 2 0, elemsOf b.data.toList 2 1]).data.toList t := by
    intro t ht
    have hq : t / 128 < 2 := by omega
    rw [hb t (by omega)]
    simp only [FlockVBridge.bitL_rowBytes, FlockVBridge.wbit]
    have hget : [elemsOf b.data.toList 2 0, elemsOf b.data.toList 2 1].getD (t / 128) 0 =
        elemsOf b.data.toList 2 (t / 128) := by
      rcases (by omega : t / 128 = 0 ∨ t / 128 = 1) with h | h <;> rw [h] <;>
        generalize elemsOf b.data.toList 2 0 = e0 <;> generalize elemsOf b.data.toList 2 1 = e1 <;> rfl
    rw [hget]
    unfold elemsOf
    rw [if_pos hq, FlockVBridge.bitOf_ofBits _ (Nat.mod_lt _ (by omega)), Nat.div_add_mod]
  rw [← pair_elemsOf hk]
  exact FlockVBridge.AccChain.acc_of_cols hX hcols hout hk

/-- **A copy's `acc` columns read the bytes on them**: where the bytes `b`, of at least two elements, sit on a level
copy's `acc` columns, its `openAcc` is `b`'s first two elements. -/
theorem openAcc_of_cols {v : ℕ → Bool} {LANES HH : ℕ} {b : ByteArray}
    (hb : ColsHold v (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) b) (hsz : 32 ≤ b.size) {k : ℕ} (hk : k < 2) :
    (Flock.Readable.openAcc v LANES HH)[k]! = (#[elemsOf b.data.toList 2 0, elemsOf b.data.toList 2 1] : Array F128)[k]! := by
  rw [pair_elemsOf hk, Flock.Readable.openAcc, FlockVBridge.Family.elems_get _ _ hk]
  unfold elemsOf
  rw [if_pos hk]
  exact FlockVBridge.OfVStar.ofBits_congr fun t ht =>
    (congrArg v (Nat.add_assoc _ _ _)).trans (hb _ (by omega))

/-- **A level copy's `acc_out` bytes are its consistency terms** at what it reads (`LevelOuts`, at `plain`'s climb
size). -/
theorem accOut_of_levelOuts {v : ℕ → Bool} {LANES HH : ℕ} (hO : Flock.Readable.LevelOuts v LANES HH) {b : ByteArray}
    (hb : ColsHold v (outCol LANES HH + 1024) b) (hsz : 32 ≤ b.size) {k : ℕ} (hk : k < 2) :
    elemsOf b.data.toList 2 k = FlockVBridge.consTerm LANES (Flock.Readable.openRow v LANES)
      (Flock.Readable.openAcc v LANES HH) (Flock.Readable.openCoefs v LANES HH) k := by
  have hO' := openOuts_plain hO.1
  unfold Flock.Readable.OpenOuts at hO'
  have hX : (Flock.climb plain (Flock.Sha512.hash (MerkleScheme.rowBytes (Flock.Readable.openRow v LANES)))
      (Flock.Readable.openIdx v LANES HH) (Flock.Readable.openSibs v LANES HH) ++ ⟨Array.replicate 64 0⟩).size =
      128 := by
    rw [ByteArray.size_append,
      FlockVBridge.AccChain.climb_size_of plain (fun _ _ => rfl) _ _ _ (FlockSoundness.Refine.sha512_size _)]
    rfl
  rw [elemsOf_accOut hX hO' hb hsz hk]
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl

/-! ## A piece's running sum at its copy -/

/-- **A piece holds its running sum at its copy** (`accHolds`): where the copy's columns `v` are a level table's
outputs at what they read (`LevelOuts`), at the piece's level's lanes (`LANES = 2^lanes`), reading the piece's opened
row (`OpenCopy`'s row tie) and its `coef`, and the chain's rows at the piece's position and the next are the first two
elements of the bytes `bi` on the copy's `acc` columns and `bo` on its `acc_out` columns. -/
theorem accHolds_of_copy (sideAt : Msg → List Msg → OpenPiece → OpenSide) (coefAt : Msg → List Coin → OpenPiece → Array F128)
    (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (q : OpenPiece) (x : Msg) (ms : List Msg) (cs : List Coin)
    {v : ℕ → Bool} {LANES HH : ℕ} (hO : Flock.Readable.LevelOuts v LANES HH)
    (hLA : LANES = 2 ^ ((sideAt x ms q).sch.levels[q.level]!).lanes)
    (hrow : Flock.Readable.openRow v LANES =
      (openAt (sideAt x ms q).lig ((sideAt x ms q).sch.levels.size - 1) q.level).1[q.query]!)
    (hcf : Flock.Readable.openCoefs v LANES HH = coefAt x cs q)
    {bi bo : ByteArray} (hbi : ColsHold v (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) bi) (hszi : 32 ≤ bi.size)
    (hbo : ColsHold v (outCol LANES HH + 1024) bo) (hszo : 32 ≤ bo.size)
    (hin : chainAt x ms q.table (chainPos (sideAt x ms q).sch q.rep q.level q.query) =
      #[elemsOf bi.data.toList 2 0, elemsOf bi.data.toList 2 1])
    (hout : chainAt x ms q.table (chainPos (sideAt x ms q).sch q.rep q.level q.query + 1) =
      #[elemsOf bo.data.toList 2 0, elemsOf bo.data.toList 2 1]) :
    accHolds sideAt coefAt chainAt q x ms cs := by
  intro k hk
  rw [hout, pair_elemsOf hk, accOut_of_levelOuts hO hbo hszo hk, hin, ← hLA, ← hrow, ← hcf]
  exact FlockVBridge.AccChain.consTerm_acc _ _ _ _ _ (openAcc_of_cols hbi hszi hk)

/-! ## Through the chain's commits, at `layoutBAcc` -/

section Commits

variable (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

/-- **Bytes committed under row `p`'s root are the row `chainAt` reads**: where `l`'s calls are true evaluations with
no collision among them, slot `Nat.pair p 0` is placed and its owner `o` holds `v`, and `l` holds an evaluation of
`frame b σ` and one of the owner's `frame v σ'` whose digests climb to one root (`commit_agree`'s premises), `chainAt`
reads `b`'s first two elements at row `p` (`chainAt_held`). -/
theorem chainAt_commit {H : Msg → Msg} {l : List (Msg × Msg)} (hT : ∀ e ∈ l, H e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {frame : Msg → Msg → Msg}
    (hinj : ∀ v σ v' σ', frame v σ = frame v' σ' → v = v') {x : Msg} {ws : List Claims.Flock.Wit}
    {p i off o : ℕ} {w : Claims.Flock.Wit} {v : Msg} (hin : accIn rest x (Nat.pair p 0) = some (i, off))
    (ho : (layoutBAcc rest).rowOwner x (Nat.pair p 0) = some o) (hw : ws[o]? = some w)
    (hv : (layoutBAcc rest).rowHeld x o w (Nat.pair p 0) = some v) {b : Msg} (hb : 32 ≤ b.length)
    {σ σ' d d' root : Msg} (hcb : (frame b σ, d) ∈ l) (hcv : (frame v σ', d') ∈ l) {fs gs : List (Msg → Msg)}
    (hapart : Claims.Recursion.VStar.StepsApart fs gs) (hclb : Claims.Recursion.VStar.chainH H fs d = root)
    (hclv : Claims.Recursion.VStar.chainH H gs d' = root)
    (hinb : ∀ u ∈ Claims.Recursion.VStar.chainIns H fs d, u ∈ l.map Prod.fst)
    (hinv : ∀ u ∈ Claims.Recursion.VStar.chainIns H gs d', u ∈ l.map Prod.fst) (t : ℕ) :
    chainAt rest x ((layoutBAcc rest).msgs x ws) t p = #[elemsOf b 2 0, elemsOf b 2 1] := by
  have e := commit_agree hT hnc hinj hcb hcv hapart hclb hclv hinb hinv
  rw [chainAt_held rest hin ho hw hv (by rw [← e]; exact hb) t, e]

/-- **A piece holds its running sum at `layoutBAcc`'s chain**, through the commits: `accHolds_of_copy` at
`chainAt rest` and the readers' messages, where the bytes `bi` on the copy's `acc` columns are committed under the
root of the chain's row at the piece's position (`pin`) and the bytes `bo` on its `acc_out` columns under the next
row's (`pout`), each beside that row's owner's holding (`chainAt_commit`), with no collision among the evaluations
`l`. -/
theorem accHolds_layoutBAcc {H : Msg → Msg} {l : List (Msg × Msg)} (hT : ∀ e ∈ l, H e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {frame : Msg → Msg → Msg}
    (hinj : ∀ v σ v' σ', frame v σ = frame v' σ' → v = v') {x : Msg} {ws : List Claims.Flock.Wit}
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (coefAt : Msg → List Coin → OpenPiece → Array F128)
    (q : OpenPiece) (cs : List Coin) {v : ℕ → Bool} {LANES HH : ℕ} (hO : Flock.Readable.LevelOuts v LANES HH)
    (hLA : LANES = 2 ^ ((sideAt x ((layoutBAcc rest).msgs x ws) q).sch.levels[q.level]!).lanes)
    (hrow : Flock.Readable.openRow v LANES =
      (openAt (sideAt x ((layoutBAcc rest).msgs x ws) q).lig
        ((sideAt x ((layoutBAcc rest).msgs x ws) q).sch.levels.size - 1) q.level).1[q.query]!)
    (hcf : Flock.Readable.openCoefs v LANES HH = coefAt x cs q)
    {bi bo : ByteArray} (hbi : ColsHold v (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) bi) (hszi : 32 ≤ bi.size)
    (hbo : ColsHold v (outCol LANES HH + 1024) bo) (hszo : 32 ≤ bo.size)
    {pin : ℕ} (hpin : pin = chainPos (sideAt x ((layoutBAcc rest).msgs x ws) q).sch q.rep q.level q.query)
    {i₀ off₀ o₀ : ℕ} {w₀ : Claims.Flock.Wit} {v₀ : Msg}
    (hin₀ : accIn rest x (Nat.pair pin 0) = some (i₀, off₀))
    (ho₀ : (layoutBAcc rest).rowOwner x (Nat.pair pin 0) = some o₀) (hw₀ : ws[o₀]? = some w₀)
    (hv₀ : (layoutBAcc rest).rowHeld x o₀ w₀ (Nat.pair pin 0) = some v₀)
    {σi σ₀ di d₀ ri : Msg} (hci : (frame bi.data.toList σi, di) ∈ l) (hc₀ : (frame v₀ σ₀, d₀) ∈ l)
    {fsi gs₀ : List (Msg → Msg)} (hapi : Claims.Recursion.VStar.StepsApart fsi gs₀)
    (hcli : Claims.Recursion.VStar.chainH H fsi di = ri) (hcl₀ : Claims.Recursion.VStar.chainH H gs₀ d₀ = ri)
    (hini : ∀ u ∈ Claims.Recursion.VStar.chainIns H fsi di, u ∈ l.map Prod.fst)
    (hin0 : ∀ u ∈ Claims.Recursion.VStar.chainIns H gs₀ d₀, u ∈ l.map Prod.fst)
    {i₁ off₁ o₁ : ℕ} {w₁ : Claims.Flock.Wit} {v₁ : Msg}
    (hin₁ : accIn rest x (Nat.pair (pin + 1) 0) = some (i₁, off₁))
    (ho₁ : (layoutBAcc rest).rowOwner x (Nat.pair (pin + 1) 0) = some o₁) (hw₁ : ws[o₁]? = some w₁)
    (hv₁ : (layoutBAcc rest).rowHeld x o₁ w₁ (Nat.pair (pin + 1) 0) = some v₁)
    {σo σ₁ dout d₁ ro : Msg} (hco : (frame bo.data.toList σo, dout) ∈ l) (hc₁ : (frame v₁ σ₁, d₁) ∈ l)
    {fso gs₁ : List (Msg → Msg)} (hapo : Claims.Recursion.VStar.StepsApart fso gs₁)
    (hclo : Claims.Recursion.VStar.chainH H fso dout = ro) (hcl₁ : Claims.Recursion.VStar.chainH H gs₁ d₁ = ro)
    (hino : ∀ u ∈ Claims.Recursion.VStar.chainIns H fso dout, u ∈ l.map Prod.fst)
    (hin1 : ∀ u ∈ Claims.Recursion.VStar.chainIns H gs₁ d₁, u ∈ l.map Prod.fst) :
    accHolds sideAt coefAt (chainAt rest) q x ((layoutBAcc rest).msgs x ws) cs := by
  have hli : 32 ≤ bi.data.toList.length := by rw [Array.length_toList]; exact hszi
  have hlo : 32 ≤ bo.data.toList.length := by rw [Array.length_toList]; exact hszo
  subst hpin
  exact accHolds_of_copy sideAt coefAt (chainAt rest) q x _ cs hO hLA hrow hcf hbi hszi hbo hszo
    (chainAt_commit rest hT hnc hinj hin₀ ho₀ hw₀ hv₀ hli hci hc₀ hapi hcli hcl₀ hini hin0 _)
    (chainAt_commit rest hT hnc hinj hin₁ ho₁ hw₁ hv₁ hlo hco hc₁ hapo hclo hcl₁ hino hin1 _)

/-! ## The commits off `layoutBAcc`'s recorded calls -/

/-- **A row's hm96 input**: the `sha512/row/v1` prefix at `σ`'s length in words, then the row. -/
def accFrame (v σ : Msg) : Msg := (HmRow.rowPrefix σ.length).data.toList ++ v

/-- Every prefix is 128 bytes, so `accFrame` is injective in the row. -/
theorem accFrame_inj : ∀ v σ v' σ', accFrame v σ = accFrame v' σ' → v = v' := fun v σ v' σ' h => by
  have hl : ∀ k, (HmRow.rowPrefix k).data.toList.length = 128 := fun k => by
    rw [Array.length_toList]
    exact FlockSoundness.Discharge.Hm.size_rowPrefix k
  unfold accFrame at h
  exact (List.append_inj h (by rw [hl, hl])).2

/-- **Slot `σ`'s digest at part `s`**: its holding's port's hm96 digest (`hmDigest`), read off `s`'s witness; empty
where `s` holds no such slot or has no witness. -/
def accDig (x : Msg) (ws : List Claims.Flock.Wit) (σ s : ℕ) : Msg :=
  (ws[s]?.bind fun w => (readX x).bind fun r => (accHoldAt r s σ).bind fun e =>
    (e.1.ports.findIdx? (·.name == e.2.1)).map fun q => readForms w (hmDigest e.1 q e.2.2)).getD []

/-- **Every held slot's commit is among its holder's recorded calls** (`rowsAgree_of_commits`' `hC`): the row's
evaluation at its holding (`rowCall_held`), framed by `accFrame` at the port's words, to the slot's digest
(`accDig`), where each holding's port is the circuit's. -/
theorem accCommit_layoutBAcc {x : Msg} {ws : List Claims.Flock.Wit}
    (hq : ∀ (r : XR) (s : ℕ) (c : Circuit) (h : String × ℕ), readX x = some r →
      r.w.parts[s]?.bind partCircuit = some c → h ∈ accHeld r s → (c.ports.findIdx? (·.name == h.1)).isSome = true) :
    ∀ σ s w v, ws[s]? = some w → (layoutBAcc rest).rowHeld x s w σ = some v →
      ∃ σ', (accFrame v σ', accDig x ws σ s) ∈ (layoutBAcc rest).calls x s w := by
  intro σ s w v hw hv
  simp only [LayoutB.rowHeld, Option.map_eq_some_iff] at hv
  obtain ⟨segs, hseg, rfl⟩ := hv
  change ((readX x).bind fun r => accRowAt r s σ) = some segs at hseg
  obtain ⟨r, hr, hseg⟩ := Option.bind_eq_some_iff.1 hseg
  rw [accRowAt_eq] at hseg
  obtain ⟨⟨c, h⟩, he, rfl⟩ := Option.map_eq_some_iff.1 hseg
  obtain ⟨hc, hh⟩ := accHoldAt_spec he
  have hh' := mem_accHeld hh
  obtain ⟨qa, hqa⟩ := Option.isSome_iff_exists.1 (hq r s c h hr hc hh')
  obtain ⟨hlt, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqa
  refine ⟨List.replicate (c.ports[qa]'hlt).words 0, ?_⟩
  have hd : accDig x ws σ s = readForms w (hmDigest c qa h.2) := by
    simp [accDig, hw, hr, he, hqa]
  rw [hd]
  unfold accFrame
  rw [List.length_replicate]
  exact rowCall_held rest hr hc hh' hqa (Array.getElem?_eq_getElem hlt) w

/-- **The parts agree on the chain's rows, or two evaluations collide** (`rowsAgree_of_commits` at `layoutBAcc`): the
commits are the recorded calls (`accCommit_layoutBAcc`) and every held slot's owner holds it (`layoutBAcc_owner`). What
stays open is each holding's climb from its digest (`accDig`) to the row's registered root, with its inputs among the
verifier's evaluations and the climbs' steps apart. -/
theorem rowsAgree_layoutBAcc {H : Msg → Msg} {x : Msg} {ws : List Claims.Flock.Wit} {b : Brings} {xs : List Msg}
    (hT : ∀ e ∈ (layoutBAcc rest).allCalls x ws, H e.1 = e.2) (hws : ws.length = nS x)
    (hq : ∀ (r : XR) (s : ℕ) (c : Circuit) (h : String × ℕ), readX x = some r →
      r.w.parts[s]?.bind partCircuit = some c → h ∈ accHeld r s → (c.ports.findIdx? (·.name == h.1)).isSome = true)
    (path : ℕ → ℕ → List (Msg → Msg)) (root : ℕ → Msg)
    (hclimb : ∀ σ s w v, ws[s]? = some w → (layoutBAcc rest).rowHeld x s w σ = some v →
      Claims.Recursion.VStar.chainH H (path σ s) (accDig x ws σ s) = root σ)
    (hin : ∀ σ s w v, ws[s]? = some w → (layoutBAcc rest).rowHeld x s w σ = some v →
      ∀ u ∈ Claims.Recursion.VStar.chainIns H (path σ s) (accDig x ws σ s), ∃ xO ∈ xs, u ∈ b.setup xO)
    (hapart : ∀ σ s t, Claims.Recursion.VStar.StepsApart (path σ s) (path σ t)) :
    (layoutBAcc rest).RowsAgree x ws ∨ (Claims.findColl ((layoutBAcc rest).allCalls x ws ++ vcalls H b xs)).isSome :=
  LayoutB.rowsAgree_of_commits hT accFrame accFrame_inj (accDig x ws) (accCommit_layoutBAcc rest hq) path root
    hclimb hin hapart (layoutBAcc_owner rest hws)

/-- **A piece holds its running sum at `layoutBAcc`'s chain, off the recorded calls** (`accHolds_layoutBAcc`): the
copy's part `s` (circuit `c`, witness `w`) holds the chain at its unit `u`'s `acc`, port `qa`, and has the output port
`acc_out`, port `qo`, the bytes `bi` and `bo` are what its witness reads at those rows, and `l` holds `layoutBAcc`'s
calls. The copy's two row calls are recorded (`rowCall_held`, `outCall_held`), and so are the two rows' owners'
(`accCommit_layoutBAcc`), so the four commit premises go; the climbs from the four digests stay. -/
theorem accHolds_layoutBAcc_recorded {H : Msg → Msg} {l : List (Msg × Msg)} (hT : ∀ e ∈ l, H e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {x : Msg} {ws : List Claims.Flock.Wit}
    (hl : ∀ e ∈ (layoutBAcc rest).allCalls x ws, e ∈ l)
    (hq : ∀ (r : XR) (s : ℕ) (c : Circuit) (h : String × ℕ), readX x = some r →
      r.w.parts[s]?.bind partCircuit = some c → h ∈ accHeld r s → (c.ports.findIdx? (·.name == h.1)).isSome = true)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (coefAt : Msg → List Coin → OpenPiece → Array F128)
    (q : OpenPiece) (cs : List Coin) {v : ℕ → Bool} {LANES HH : ℕ} (hO : Flock.Readable.LevelOuts v LANES HH)
    (hLA : LANES = 2 ^ ((sideAt x ((layoutBAcc rest).msgs x ws) q).sch.levels[q.level]!).lanes)
    (hrow : Flock.Readable.openRow v LANES =
      (openAt (sideAt x ((layoutBAcc rest).msgs x ws) q).lig
        ((sideAt x ((layoutBAcc rest).msgs x ws) q).sch.levels.size - 1) q.level).1[q.query]!)
    (hcf : Flock.Readable.openCoefs v LANES HH = coefAt x cs q)
    {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {w : Claims.Flock.Wit} (hw : ws[s]? = some w) {u : ℕ} (hh : ("acc", u) ∈ accHeld r s)
    {qa qo : ℕ} (hqa : c.ports.findIdx? (·.name == "acc") = some qa)
    (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo)
    {bi bo : ByteArray} (hbi : ColsHold v (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) bi) (hszi : 32 ≤ bi.size)
    (hbo : ColsHold v (outCol LANES HH + 1024) bo) (hszo : 32 ≤ bo.size)
    (hbi' : bi.data.toList = readSegs w (accSegs c ("acc", u)))
    (hbo' : bo.data.toList = readForms w (accOutForms c qo u))
    {pin : ℕ} (hpin : pin = chainPos (sideAt x ((layoutBAcc rest).msgs x ws) q).sch q.rep q.level q.query)
    {i₀ off₀ o₀ : ℕ} {w₀ : Claims.Flock.Wit} {v₀ : Msg}
    (hin₀ : accIn rest x (Nat.pair pin 0) = some (i₀, off₀))
    (ho₀ : (layoutBAcc rest).rowOwner x (Nat.pair pin 0) = some o₀) (hw₀ : ws[o₀]? = some w₀)
    (hv₀ : (layoutBAcc rest).rowHeld x o₀ w₀ (Nat.pair pin 0) = some v₀) {ri : Msg}
    {fsi gs₀ : List (Msg → Msg)} (hapi : Claims.Recursion.VStar.StepsApart fsi gs₀)
    (hcli : Claims.Recursion.VStar.chainH H fsi (readForms w (hmDigest c qa u)) = ri)
    (hcl₀ : Claims.Recursion.VStar.chainH H gs₀ (accDig x ws (Nat.pair pin 0) o₀) = ri)
    (hini : ∀ u' ∈ Claims.Recursion.VStar.chainIns H fsi (readForms w (hmDigest c qa u)), u' ∈ l.map Prod.fst)
    (hin0 : ∀ u' ∈ Claims.Recursion.VStar.chainIns H gs₀ (accDig x ws (Nat.pair pin 0) o₀), u' ∈ l.map Prod.fst)
    {i₁ off₁ o₁ : ℕ} {w₁ : Claims.Flock.Wit} {v₁ : Msg}
    (hin₁ : accIn rest x (Nat.pair (pin + 1) 0) = some (i₁, off₁))
    (ho₁ : (layoutBAcc rest).rowOwner x (Nat.pair (pin + 1) 0) = some o₁) (hw₁ : ws[o₁]? = some w₁)
    (hv₁ : (layoutBAcc rest).rowHeld x o₁ w₁ (Nat.pair (pin + 1) 0) = some v₁) {ro : Msg}
    {fso gs₁ : List (Msg → Msg)} (hapo : Claims.Recursion.VStar.StepsApart fso gs₁)
    (hclo : Claims.Recursion.VStar.chainH H fso (readForms w (hmDigest c qo u)) = ro)
    (hcl₁ : Claims.Recursion.VStar.chainH H gs₁ (accDig x ws (Nat.pair (pin + 1) 0) o₁) = ro)
    (hino : ∀ u' ∈ Claims.Recursion.VStar.chainIns H fso (readForms w (hmDigest c qo u)), u' ∈ l.map Prod.fst)
    (hin1 : ∀ u' ∈ Claims.Recursion.VStar.chainIns H gs₁ (accDig x ws (Nat.pair (pin + 1) 0) o₁),
      u' ∈ l.map Prod.fst) :
    accHolds sideAt coefAt (chainAt rest) q x ((layoutBAcc rest).msgs x ws) cs := by
  obtain ⟨hla, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqa
  obtain ⟨hlo, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqo
  have hci : (accFrame bi.data.toList (List.replicate (c.ports[qa]'hla).words 0),
      readForms w (hmDigest c qa u)) ∈ l := by
    refine hl _ (LayoutB.mem_allCalls hw ?_)
    unfold accFrame
    rw [List.length_replicate, hbi']
    exact rowCall_held rest hr hc hh hqa (Array.getElem?_eq_getElem hla) w
  have hco : (accFrame bo.data.toList (List.replicate (c.ports[qo]'hlo).words 0),
      readForms w (hmDigest c qo u)) ∈ l := by
    refine hl _ (LayoutB.mem_allCalls hw ?_)
    unfold accFrame
    rw [List.length_replicate, hbo']
    exact outCall_held rest hr hc hh hqo (Array.getElem?_eq_getElem hlo) w
  obtain ⟨σ₀, hc₀⟩ := accCommit_layoutBAcc rest hq _ _ _ _ hw₀ hv₀
  obtain ⟨σ₁, hc₁⟩ := accCommit_layoutBAcc rest hq _ _ _ _ hw₁ hv₁
  exact accHolds_layoutBAcc rest hT hnc accFrame_inj sideAt coefAt q cs hO hLA hrow hcf hbi hszi hbo hszo hpin
    hin₀ ho₀ hw₀ hv₀ hci (hl _ (LayoutB.mem_allCalls hw₀ hc₀)) hapi hcli hcl₀ hini hin0
    hin₁ ho₁ hw₁ hv₁ hco (hl _ (LayoutB.mem_allCalls hw₁ hc₁)) hapo hclo hcl₁ hino hin1

end Commits

end RecDeploy
