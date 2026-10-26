import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.SlotsCommit

/-!
# The `rec-residuals` pieces' lowering at the deployed run (U12)

`resPieces_computes` gives a `rec-residuals` part's share of the verdict's pieces from two hypotheses: check 5's tie
(`ResTie`: the copy's `out` columns are zero and its `v` port reads the verifier's values) and the layout fact `hcopy`.
This file states both at the deployed parameters and proves the tie's core.

**Check 5's tie needs the part's row calls in the layout.** Check 5 (`RecCoins.checkResiduals`, at the statement's
recursive setup, `RecIns.setup`) holds each instance's public commit string of port `v` to `commitRow` of the verifier's
own row under `verifierSalt`, and of port `out` to the zero row's under `zeroSalt` (`RecCoins.checkRows`). The part's hm96
slot outputs the commit string of the row its columns hold (`SlotCommits`). Two commit strings agree only where the two
salt inputs and the two row inputs collide or are equal (`commit_tie`). The verifier's inputs are its own evaluations
(`Brings.setup`, `vcallsAt`); the part's must be among its recorded calls for `PartAccepts`' refusal
(`findColl (L.calls x s w ++ vcallsAt H D xO)`) to charge the collision. `layoutBTops` records the hm96 calls of the
message-round ports only (`heldRows`), not of `v` or `out`. `layoutBTopsV` adds them (`valCalls`): each instance's salt
and row calls at `v` (its input columns, `rowHidden`) and at `out` (its output columns, `outForms`).

* `commit_tie`: two commit strings at SHA-512 agree, and both salts' and both rows' inputs are among a list of true
  evaluations with no collision, so the rows are equal.
* `layoutBTopsV`, `layVR`: V\*'s bound layout with check 5's calls; `resSetup`: check 5's own evaluations at a V\*
  statement, which the deployed `Brings.setup` is to list (`hset`).
* `VVTie`: the coin tie, `vvAt` at the coins is check 5's row at the V\* statement (`staged_rep_drawn`).
* `ResLowered`: the share at the deployed layout, the statement `hlowP`'s `rec-residuals` part reads.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

noncomputable section

/-! ## Check 5's tie at one row -/

/-- **Two commit strings agree only at one row, up to a collision**: where `commitString A y = commitString B σ` (hm96
at the pinned key), and `A`, `B` and both salts' inputs `saltPrefix ++ y`, `saltPrefix ++ σ` are inputs of a list of
true SHA-512 evaluations with no collision, `A = B`: the second halves give the salts (a collision otherwise), and then
the first halves give the digests. -/
theorem commit_tie {l : List (Msg × Msg)} (hT : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {A B y σ : Msg}
    (hA : A ∈ l.map Prod.fst) (hB : B ∈ l.map Prod.fst)
    (hy : FlockSoundness.Refine.K512.saltPrefix ++ y ∈ l.map Prod.fst)
    (hσ : FlockSoundness.Refine.K512.saltPrefix ++ σ ∈ l.map Prod.fst)
    (heq : FlockSoundness.Binding.commitString A y = FlockSoundness.Binding.commitString B σ) : A = B := by
  have hno : ∀ u v, u ∈ l.map Prod.fst → v ∈ l.map Prod.fst → FlockDeploy.sha512 u = FlockDeploy.sha512 v → u = v :=
    fun u v hu hv hH => Classical.byContradiction fun hne => by
      rw [Claims.findColl_isSome_mem FlockDeploy.sha512 l hT u v hu hv ⟨hne, hH⟩] at hnc
      exact absurd hnc (by decide)
  have hlen : ∀ r z : Msg, (FlockSoundness.Merkle.xorBytes (FlockSoundness.Binding.digBytes
      (FlockSoundness.Refine.H512 r)) (FlockSoundness.Refine.K512.mask z)).length = 64 :=
    fun r z => length_xorMask r z
  unfold FlockSoundness.Binding.commitString at heq
  obtain ⟨h1, h2⟩ := List.append_inj heq ((hlen A y).trans (hlen B σ).symm)
  have hs : y = σ := by
    have := hno _ _ hy hσ (by rw [sha512_eq, sha512_eq]; exact h2)
    exact List.append_cancel_left this
  subst hs
  refine hno _ _ hA hB ?_
  rw [sha512_eq, sha512_eq]
  have hl : ∀ r : Msg, (FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 r)).length =
      (FlockSoundness.Refine.K512.mask y).length := fun r => by
    rw [FlockSoundness.Discharge.Hm.length_digBytes, FlockSoundness.Refine.K512.mask_len]; rfl
  exact FlockSoundness.Merkle.xorBytes_inj (hl A) (hl B) h1

/-- **The executable's commit string is the model's**: `Hm96.Default512.commitString` of a row's SHA-512 under a salt
is `Binding.commitString` of the row's bytes and the salt's. -/
theorem commitString_exec (r : Msg) (salt : ByteArray) :
    (Hm96.Default512.commitString (Sha512.hash ⟨r.toArray⟩) salt).data.toList =
      FlockSoundness.Binding.commitString r salt.data.toList := by
  have hsp : (⟨(FlockSoundness.Refine.K512.saltPrefix ++ salt.data.toList).toArray⟩ : ByteArray) =
      Hm96.Default512.saltPrefix ++ salt := by
    rw [show FlockSoundness.Refine.K512.saltPrefix = Hm96.Default512.saltPrefix.data.toList from rfl,
      ← Array.toList_append, ← ByteArray.data_append, FlockSoundness.Refine.bytes_of_list]
  rw [FlockSoundness.Binding.commitString, FlockSoundness.Binding.digBytes, FlockSoundness.Binding.digBytes,
    FlockSoundness.Refine.dig_eq, FlockSoundness.Refine.dig_eq, FlockSoundness.Refine.H512_val,
    FlockSoundness.Refine.H512_val, FlockSoundness.Refine.K512_mask, FlockSoundness.Refine.bytes_of_list, hsp,
    ← FlockSoundness.Refine.xor_range _ _ 64 (FlockSoundness.Refine.sha512_size _)
      (FlockSoundness.Refine.mask_size salt), ← Array.toList_append, ← ByteArray.data_append]
  rfl

/-! ## V\*'s bound layout with check 5's calls -/

/-- **Output port `q`'s row at instance `n`**, at forms: word `k` bit `t` (row bit `16 k + t`) the output bit
`HmOut.outCol` places it at in the instance's block and VU, where `k` is within the port's run of output words
(`HmOut.outGroup`), and zero past it (`HmOut.delta`'s forced zero). -/
def outForms (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range (16 * (c.ports.getD q default).words)).map fun i =>
    if i / 16 < (HmOut.outGroup c (q - HmOut.nIn c)).2.2.2 then
      HmNets.Lin.var (n / c.g * 2 ^ c.kLog +
        (HmOut.outCol c (n % c.g) ((HmOut.outGroup c (q - HmOut.nIn c)).2.2.1 + i / 16) + i % 16))
    else ⟨[], false⟩

/-- **Port `q`'s row at instance `n`**, at forms: an input row's unit input columns (`rowHidden`), an output row's
output bits (`outForms`). -/
def rowForms (c : Circuit) (nm : String) (q n : ℕ) : List HmNets.Lin :=
  if q < HmOut.nIn c then ((rowHidden c nm n).getD []).map HmNets.Lin.var else outForms c q n

/-- **Port `q`'s hm96 calls at instance `n`**, `hmCalls` at `rowForms`: the salt's `SHA-512(p_salt ‖ y) = c`, the row's
`SHA-512(rowIn row) = d`. -/
def portCalls (c : Circuit) (nm : String) (q n : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  [(Hm96.Default512.saltPrefix.data.toList, hmSalt c q n, (hmCommit c q n).drop 512),
    (rowPfx c q, rowForms c nm q n, hmDigest c q n)]

/-- The ports check 5 holds to the verifier's own rows at a `rec-residuals` part (`RecCoins.checkResiduals`): `v`, the
verifier's values, and `out`, its zero rows. -/
def resPorts : List String := ["v", "out"]

/-- **The rows check 5 holds at part `s`**: at a part whose V\* statement is an algebra part (`.inner`), ports `v` and
`out` (`resPorts`) at each of its two instances (rep `r` at instance `r`), with the port's index. -/
def valPorts (r : XR) (s : ℕ) : List (String × ℕ × ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (·[s]?), r.w.parts[s]?.bind partCircuit with
  | some (.inner _ _ _), some c => resPorts.flatMap fun nm => match c.ports.findIdx? (·.name == nm) with
    | some q => (List.range 2).map fun n => (nm, q, n)
    | none => []
  | _, _ => []

/-- **Check 5's calls at part `s`**: each of `valPorts`' hm96 calls (`portCalls`). -/
def valCalls (r : XR) (s : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  ((r.w.parts[s]?.bind partCircuit).map fun c => (valPorts r s).flatMap fun h => portCalls c h.1 h.2.1 h.2.2).getD []

/-- **V\*'s bound layout with check 5's calls**: `layoutBTops fin` with every part's `valCalls` recorded. -/
def layoutBTopsV (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) : LayoutB :=
  { layoutBTops fin with
    shaAt := fun x s => (layoutBTops fin).shaAt x s ++ ((readX x).map fun r => valCalls r s).getD [] }

/-- V\*'s deployed bound layout with check 5's calls, at `elem` (`DeployedR.layR` with `layoutBTopsV`). -/
abbrev layVR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : LayoutB :=
  layoutBTopsV (finOpen (DeployedR.restR elem))

theorem layVR_calls (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg) (s : ℕ)
    (w : Claims.Flock.Wit) : (layVR elem).calls x s w = (DeployedR.layR elem).calls x s w ++
      (((readX x).map fun r => valCalls r s).getD []).map fun p => (p.1 ++ readForms w p.2.1, readForms w p.2.2) := by
  simp only [LayoutB.calls, layoutBTopsV, List.map_append]

/-- **A part accepting at `layVR` accepts at `layR`**: the two layouts differ only in `layVR`'s extra calls, which the
compiled record reads only as its readers' calls, so a part's true and collision-free calls at `layVR` are so at `layR`.
The other pieces' statements at `layR` hold at `layVR`. -/
theorem partAccepts_layR {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {shape : Claims.Recursion.Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts FlockDeploy.sha512 (recBound _root_.Flock.Readable.relation b (layVR elem) IV)
      (deployZkRec _root_.Flock.Readable.relation) (layVR elem) shape shows cmt s x c ms) :
    PartAccepts FlockDeploy.sha512 (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV)
      (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR elem) shape shows cmt s x c ms := by
  obtain ⟨hst, xO, w, hck, hT, hc, hs, hh, hp, hm, hr⟩ := h
  rw [layVR_calls] at hT hc
  refine ⟨hst, xO, w, hck, fun e he => hT e (List.mem_append_left _ he), ?_, hs, hh, hp, hm, hr⟩
  have ht : ∀ e ∈ ((DeployedR.layR elem).calls x s w ++ (((readX x).map fun r => valCalls r s).getD []).map
      fun p => (p.1 ++ readForms w p.2.1, readForms w p.2.2)) ++
      vcallsAt FlockDeploy.sha512 (recBound _root_.Flock.Readable.relation b (layVR elem) IV) xO,
      FlockDeploy.sha512 e.1 = e.2 := fun e he => by
    rcases List.mem_append.1 he with he | he
    · exact hT e he
    · exact vcallsAt_true _ _ _ e he
  refine Bool.eq_false_iff.2 fun h' => ?_
  have hs' := Claims.Recursion.findColl_isSome_of_sub FlockDeploy.sha512 ht (fun e he => by
    rcases List.mem_append.1 he with he | he
    · exact List.mem_append_left _ (List.mem_append_left _ he)
    · exact List.mem_append_right _ he) h'
  rw [hs'] at hc
  exact absurd hc (by decide)

/-! ## Check 5's own evaluations -/

/-- The bytes `Firewall.rowDigest` hashes: the `sha512/row/v1` prefix at the row's word count, then its words, each
two bytes little-endian. -/
def rowInput (row : Array ℕ) : Msg :=
  (row.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8)
    (Hm96.pad ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16] ++ be32 row.size.toUInt32) 128)).data.toList

/-- **`checkRows`' evaluations at the rows `rows` under the salts `salt`**: each row's `commitRow` input (`rowInput` at
its words padded as `Registered.rowWords` pads them) and its salt's. -/
def commitIns (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) : List Msg :=
  (List.range rows.size).flatMap fun i =>
    [rowInput (RecCoins.padTo rows[i]! (Registered.rowWords rows[i]!.size 16)),
      (Hm96.Default512.saltPrefix ++ salt i).data.toList]

/-- **Check 5's `v` rows at a `rec-residuals` part `j`** (`RecCoins.checkResiduals`' `rows`): rep `r`'s, the verifier's
values at its coins of part `j`'s elements of `v`, the comb last, as words padded to the port's. -/
def resRows (inn : RecCoins.Inner) (j : ℕ) : Except String (Array (Array ℕ)) := do
  let (sh, extras) ← inn.algebra
  let gs ← RecResiduals.groups (← RecAlgebra.structureOf sh) (RecResiduals.portSizes sh)
  let some (_, _, vs) := gs[j]? | throw "no part"
  #[0, 1].mapM fun r => do
    let rep ← inn.rep sh extras r
    let base := rep.v.extract 0 (rep.v.size - 64)
    let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure RecCoins.ZERO_COMB
    let full := base ++ comb
    pure (RecCoins.padTo (RecCoins.words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size))

/-- **Check 5's evaluations at one statement `st`** of a `rec-residuals` unit, at its source `src`: `checkValue`'s at
`v` (the verifier's rows under `verifierSalt` of META's `program_sha512`) and `checkZero`'s at `out` (two zero rows of
`OUT_WORDS` under `zeroSalt`). None at any other unit. -/
def resSetupSt (src : Option RecCoins.Source) (st : Stmt) : List Msg :=
  match RecResiduals.params st.c.unit.name, src with
  | some (_, _, _, j), some f =>
    match (f.load st.c.unit.name).toOption, (st.c.metaJ.getObjValAs? String "program_sha512").toOption with
    | some inn, some prog => match (resRows inn j).toOption with
      | some rows => commitIns rows (RecCoins.verifierSalt prog RecCoins.V_VALUE) ++
          commitIns (RecCoins.zeroRows 2 RecCoins.OUT_WORDS) (RecCoins.zeroSalt RecCoins.ZERO_VALUE)
      | none => []
    | _, _ => []
  | _, _ => []

/-- **Check 5's evaluations at a V\* statement `xO`**: `resSetupSt` at each statement its recursive setup sets up, at
check 5's source (`RecCoins.Source.atC3`, as `RecIns.setup` runs `RecCoins.check`). The deployed `Brings.setup` lists
them (`hset : ∀ xO, resSetup xO ⊆ b.setup xO`). -/
def resSetup (xO : Msg) : List Msg :=
  match setupO xO with
  | some (i, _, sts) => sts.toList.flatMap (resSetupSt (RecCoins.Source.atC3 i.src i.inner i.commits))
  | none => []

theorem rowDigest_eq (row : Array ℕ) : Firewall.rowDigest row = Sha512.hash ⟨(rowInput row).toArray⟩ := by
  rw [rowInput, FlockSoundness.Refine.bytes_of_list]
  rfl

/-- A row's words as `Firewall.rowDigest` lays them after its prefix: each two bytes, little-endian. -/
def wordBytes (ws : Array ℕ) : Msg := ws.toList.flatMap fun w => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]

/-- The instance copy `(o, g, 0)` of a part's table is: block `o`'s VU `g`, its first unit (`rowHidden`'s index). Check 5
holds instances `0` and `1` (reps `0` and `1`), so only copies `(o, g, 0)` with `instOf c o g < 2` are tied. -/
def instOf (c : Circuit) (o g : ℕ) : ℕ := o * c.g + g

/-- The `nv` elements a row of words carries, 8 words an element (`RecCoins.words`), zero past `nv`. -/
def wordsElems (nv : ℕ) (ws : Array ℕ) (e : ℕ) : _root_.Flock.F128 :=
  if e < nv then FlockVBridge.ofBits fun t => (ws.getD (8 * e + t / 16) 0).testBit (t % 16) else 0

theorem foldl_words (l : List ℕ) (b : ByteArray) :
    (l.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8) b).data.toList =
      b.data.toList ++ l.flatMap fun w => [(w % 256).toUInt8, (w / 256 % 256).toUInt8] := by
  induction l generalizing b with
  | nil => simp
  | cons w l ih =>
    rw [List.foldl_cons, ih, List.flatMap_cons]
    simp [ByteArray.data_push, Array.toList_push]

/-- **`rowInput` is the row's `sha512/row/v1` prefix, then its words' bytes.** -/
theorem rowInput_eq (row : Array ℕ) :
    rowInput row = (HmRow.rowPrefix row.size).data.toList ++ wordBytes row := by
  rw [rowInput, ← Array.foldl_toList, foldl_words]
  rfl

theorem bitL_wordBytes : ∀ (ws : List ℕ) (i : ℕ), i < 16 * ws.length →
    FlockSoundness.Discharge.Hm.bitL (ws.flatMap fun w => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]) i =
      (ws.getD (i / 16) 0).testBit (i % 16)
  | [], _, h => by simp at h
  | w :: ws, i, h => by
    rw [List.flatMap_cons]
    by_cases hi : i < 16
    · rw [FlockSoundness.Discharge.Hm.bitL, List.getD_eq_getElem?_getD, Nat.div_eq_of_lt hi, Nat.mod_eq_of_lt hi]
      by_cases h8 : i < 8
      · rw [Nat.div_eq_of_lt h8]
        simp only [List.cons_append, List.getElem?_cons_zero, Option.getD_some, List.getD_cons_zero]
        rw [Nat.toUInt8, UInt8.toNat_ofNat', Nat.mod_mod_of_dvd _ (by norm_num), Nat.testBit_mod_two_pow,
          Nat.mod_eq_of_lt h8]
        simp [h8]
      · rw [show i / 8 = 1 by omega]
        simp only [List.cons_append, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some,
          List.getD_cons_zero]
        rw [Nat.toUInt8, UInt8.toNat_ofNat', Nat.mod_mod_of_dvd _ (by norm_num), Nat.testBit_mod_two_pow,
          show (256 : ℕ) = 2 ^ 8 from rfl, Nat.testBit_div_two_pow, show i % 8 + 8 = i by omega]
        simp [show i % 8 < 8 by omega]
    · have ih := bitL_wordBytes ws (i - 16) (by simp at h; omega)
      rw [show (i / 16) = (i - 16) / 16 + 1 by omega, List.getD_cons_succ, show i % 16 = (i - 16) % 16 by omega, ← ih]
      rw [FlockSoundness.Discharge.Hm.bitL, FlockSoundness.Discharge.Hm.bitL, List.cons_append, List.cons_append,
        List.nil_append, show i / 8 = (i - 16) / 8 + 2 by omega, show i % 8 = (i - 16) % 8 by omega]
      simp [List.getD_eq_getElem?_getD]

/-- **The row's bytes, past the prefix**: where port `q` has the row's word count, bytes that are the port's
`sha512/row/v1` prefix then `fs`'s bytes, and the row's `rowInput`, give `fs`'s bytes as the row's words'. -/
theorem readForms_of_rowInput {c : Circuit} {q : ℕ} {pt : Port} (hpt : c.ports[q]? = some pt) {row : Array ℕ}
    (hw : pt.words = row.size) {w : Claims.Flock.Wit} {fs : List HmNets.Lin}
    (h : rowPfx c q ++ readForms w fs = rowInput row) : readForms w fs = wordBytes row := by
  rw [rowInput_eq, rowPfx, hpt, Option.map_some, Option.getD_some, hw] at h
  exact List.append_cancel_left h

theorem getD_padTo (ws : Array ℕ) (n i : ℕ) : (RecCoins.padTo ws n).getD i 0 = ws.getD i 0 := by
  unfold RecCoins.padTo
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_append]
  split_ifs with h
  · rfl
  · rw [Array.getElem?_eq_none (Nat.le_of_not_lt h)]
    simp only [Array.getElem?_replicate, Option.getD_none]
    split_ifs <;> rfl

theorem wordsElems_padTo (nv : ℕ) (ws : Array ℕ) (n : ℕ) :
    wordsElems nv (RecCoins.padTo ws n) = wordsElems nv ws := by
  funext e
  unfold wordsElems
  split_ifs
  · exact FlockVBridge.OfVStar.ofBits_congr fun t _ => by rw [getD_padTo]
  · rfl

/-- **Check 5's tie at a copy, from its rows**: where the bytes the copy's `v` row forms `fv` read are the words of
`row` (the verifier's, `8 nv` words at least) and those its `out` row forms `fo` read are zero words, and the copy's
`v` columns and residual columns are those forms' bits, the copy's residual columns are zero and its `v` reads `row`'s
elements (`ResTie` at `wordsElems nv row`). -/
theorem resTie_of_rows {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {v : ℕ → Bool} {w : Claims.Flock.Wit}
    {fv fo : List HmNets.Lin} {row : Array ℕ} {nz : ℕ} (hsz : 8 * nv ≤ row.size)
    (hvb : readForms w fv = wordBytes row) (hob : readForms w fo = wordBytes (Array.replicate nz 0))
    (hVc : ∀ i (hi : i < 128 * nv), ∃ hl : i < fv.length,
      v (FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) + i) = formBit w fv[i])
    (hOc : ∀ q (hq : q < 128 * S.res.length), ∃ hl : q < fo.length,
      v (FlockVBridge.OfVStar.partBase S ns nv + q) = formBit w fo[q]) :
    ResTie S ns nv (wordsElems nv row) v := by
  refine ⟨fun q hq => ?_, funext fun e => ?_⟩
  · obtain ⟨hl, hq'⟩ := hOc q hq
    rw [hq', ← bitL_readForms hl, hob]
    by_cases hz : q < 16 * nz
    · rw [wordBytes, bitL_wordBytes _ _ (by simpa using hz)]
      simp
    · rw [FlockSoundness.Discharge.Hm.bitL, List.getD_eq_getElem?_getD, List.getElem?_eq_none]
      · simp
      · simp [wordBytes]; omega
  · unfold _root_.Flock.Readable.vRead _root_.Flock.Readable.readElems wordsElems
    split_ifs with he
    · refine FlockVBridge.OfVStar.ofBits_congr fun t ht => ?_
      obtain ⟨hl, ht'⟩ := hVc (128 * e + t) (by omega)
      rw [ht', ← bitL_readForms hl, hvb, wordBytes, bitL_wordBytes _ _ (by simp; omega),
        show (128 * e + t) / 16 = 8 * e + t / 16 by omega, show (128 * e + t) % 16 = t % 16 by omega]
      simp [List.getD_eq_getElem?_getD, Array.getD]
    · rfl

/-! ## Check 5's tie at one port and instance -/

/-- **Check 5's tie at one port's row** (`commit_tie` at `PartAccepts`' refusal): where part `s` records port `q`'s
calls at instance `n` (`portCalls`, true and collision-free with the verifier's evaluations), its slot outputs the commit
string of the row its forms read (`hslot`), check 5 holds that commit string to `commitRow row salt` (`hpub`), and the
verifier evaluates that row's and salt's inputs (`hv`, `hs`), the part's row bytes are the verifier's row's. -/
theorem port_tie {Coin' : Type} {H : Msg → Msg} (hH : H = FlockDeploy.sha512) {D : Claims.Recursion.Deploy Coin'}
    {L : LayoutB} {x xO : Msg} {s : ℕ} {w : Claims.Flock.Wit} {c : Circuit} {nm : String} {q n : ℕ}
    (hT : ∀ e ∈ L.calls x s w, H e.1 = e.2) (hnc : (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false)
    (hmem : ∀ e ∈ portCalls c nm q n, e ∈ L.shaAt x s)
    (hslot : readForms w (hmCommit c q n) = FlockSoundness.Binding.commitString
      (rowPfx c q ++ readForms w (rowForms c nm q n)) (readForms w (hmSalt c q n)))
    {row : Array ℕ} {salt : ByteArray}
    (hpub : (RecCoins.commitRow row salt).data.toList = readForms w (hmCommit c q n))
    (hv : rowInput (RecCoins.padTo row (Registered.rowWords row.size 16)) ∈ D.setup xO)
    (hs : (Hm96.Default512.saltPrefix ++ salt).data.toList ∈ D.setup xO) :
    rowPfx c q ++ readForms w (rowForms c nm q n) = rowInput (RecCoins.padTo row (Registered.rowWords row.size 16)) := by
  subst hH
  have hT' := vcallsAt_append_true hT (D := D) (xO := xO)
  have hcall : ∀ e ∈ portCalls c nm q n, e.1 ++ readForms w e.2.1 ∈
      (L.calls x s w ++ vcallsAt FlockDeploy.sha512 D xO).map Prod.fst := fun e he =>
    List.mem_map.2 ⟨_, List.mem_append_left _ (List.mem_map.2 ⟨e, hmem e he, rfl⟩), rfl⟩
  have hver : ∀ u ∈ D.setup xO, u ∈ (L.calls x s w ++ vcallsAt FlockDeploy.sha512 D xO).map Prod.fst := fun u hu =>
    List.mem_map.2 ⟨(u, _), List.mem_append_right _ (List.mem_map.2 ⟨u, hu, rfl⟩), rfl⟩
  refine commit_tie (y := readForms w (hmSalt c q n)) (σ := salt.data.toList) hT' hnc
    (hcall _ (List.mem_cons_of_mem _ List.mem_cons_self)) (hver _ hv) (hcall _ List.mem_cons_self) ?_ ?_
  · rw [show FlockSoundness.Refine.K512.saltPrefix ++ salt.data.toList =
      (Hm96.Default512.saltPrefix ++ salt).data.toList by
        rw [ByteArray.data_append, Array.toList_append]; rfl]
    exact hver _ hs
  · rw [← hslot, ← hpub, RecCoins.commitRow, rowDigest_eq, commitString_exec]

/-! ## The share at the pieces' copies -/

/-- **`resPieces_computes` with the tie at the pieces' copies only**, and the whole acceptance at hand: wherever part
`s` accepts at an algebra part, each piece `ownQ` gives it holds where, at every staged statement and witness that pass
`Flock.check`, agree with the messages and record true, collision-free calls, the piece sits at one of the statement's
copies, of its group, at which check 5's tie holds (`ResTie`) and whose rows are what `elemAt` reads. Check 5 holds only
two instances, so `resPieces_computes`' tie at every copy does not follow from it; this one asks it where the pieces
are. -/
theorem resPieces_computesAt {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm pt))
    (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (hpc : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w = true →
      showsPart untr b s x xO c → L.Agrees cmt x s w xO ms → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : q.key.1 < p.sts.size, q.key.2.1 < 2 ^ (p.sts[q.key.1].m - p.sts[q.key.1].c.kLog) ∧
          q.key.2.2.1 < p.sts[q.key.1].c.g ∧ q.key.2.2.2 < p.sts[q.key.1].c.upv ∧
          ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
            (nv : ℕ), _root_.Flock.RecResiduals.params p.sts[q.key.1].c.unit.name = some (m, k, cl, jj) →
            _root_.Flock.RecResiduals.shape? m k cl = some sh →
            _root_.Flock.RecResiduals.holoOf p.sts[q.key.1].c = .ok holo →
            _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
            S = q.S ∧
            ResTie S ns nv
              (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)
              (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2) ∧
            ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
              os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
              _root_.Flock.Readable.rowsRead ns nv
                  (_root_.Flock.Readable.copyOf p q.key.1 ht w q.key.2.1 q.key.2.2.1 q.key.2.2.2) =
                elemAt (Claims.Recursion.xI D x) (Bd.opened os last) q.key) :
    PiecesComputes Bd D pieces (resHolds elemAt vvAt) ownQ s x c ms := by
  obtain ⟨-, xO, w, hck, hT, hc, hs, hh, hp, hm, hrw⟩ := h
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, -, htab⟩ := stagedReads hs
  have hck' : Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w =
      true := hck
  have hcp := _root_.Flock.Readable.stmtCopies_of_allOuts htab
    (_root_.Flock.Readable.allOuts_of_check hck' hx ht hpi)
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  intro q hq ho os last hf hl
  obtain ⟨ht', ho', hg, hu, hch⟩ := hpc xO w p hx hck' hs ⟨hh, hp, hm, hrw⟩ hT hc q hq ho
  obtain ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, hro⟩ :=
    _root_.Flock.Readable.innerCopies ht' ho' hg hu hcp
  obtain ⟨rfl, htie, hE⟩ := hch m k cl jj sh holo S ns nv h1 h2 h3 h4
  have hz : ResZero q.S (_root_.Flock.Readable.rowsRead ns nv
      (_root_.Flock.Readable.copyOf p q.key.1 ht' w q.key.2.1 q.key.2.2.1 q.key.2.2.2))
      (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key) :=
    resVal_zero_of_tie hro htie
  rw [hE os last hf hl] at hz
  exact hz

/-! ## The statements -/

/-- **The coin tie**: at a V\* statement `xO` part `s` stages at `x` and shows the coins `c` at, wherever check 5 loads
a `rec-residuals` statement's inner session and computes its `v` rows (`resRows`), the verifier's values the coins give
at a copy (`vvAt` at the inner statement and the coins) are the elements of that copy's instance's row, at the
verifier's part `(S, ns, nv)`. `staged_rep_drawn` is its coin side. -/
def VVTie (untr : TrReader) (b : Brings) (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ t (ht : t < sts.size) (m k cl jj : ℕ),
      RecResiduals.params sts[t].c.unit.name = some (m, k, cl, jj) →
      ∀ (f : RecCoins.Source) (inn : RecCoins.Inner) (rows : Array (Array ℕ)),
        RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[t].c.unit.name = .ok inn →
        resRows inn jj = .ok rows →
        ∀ (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
          RecResiduals.shape? m k cl = some sh → RecResiduals.holoOf sts[t].c = .ok holo →
          RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
          ∀ o g, g < sts[t].c.g → instOf sts[t].c o g < 2 →
            vvAt (xIOf x) ((List.range (Bk (xIOf x))).map c) (t, o, g, 0) =
              wordsElems nv rows[instOf sts[t].c o g]!

/-- **The `rec-residuals` pieces' share of `hlowP` at the deployed layout** (U12's statement): at the compiled record
`recBound` over V\*'s bound layout with check 5's calls (`layVR elem`), the deployed outer record and the staging
`showsPart`, wherever part `s` accepts and its V\* statement at `x`'s inner shape is an algebra part (`.inner`), each
`rec-residuals` piece `ownQ` gives it holds (`resHolds elemAt vvAt`) at the messages the last message opens and the
coins (`PiecesComputes` at `DeployedR.bdR`). -/
def ResLowered (b : Brings) (IV : Claims.Verifier Coin)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128) : Prop :=
  ∀ s x c ms, PartAccepts FlockDeploy.sha512 (recBound _root_.Flock.Readable.relation b (layVR elem) IV)
      (deployZkRec _root_.Flock.Readable.relation) (layVR elem) shape (showsPart untr b) cmt s x c ms →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), RecVStar.statements r.shape = .ok ss →
      ∀ (j : ℕ) (nm : String) (pt : VStar.Part), ss[s]? = some (.inner j nm pt) →
        PiecesComputes DeployedR.bdR (recBound _root_.Flock.Readable.relation b (layVR elem) IV) pieces
          (resHolds elemAt vvAt) ownQ s x c ms

end

end RecDeploy.ResLower
