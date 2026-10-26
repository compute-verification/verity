import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.SlotsCommit
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Soundness.ExecRows
import Proofs.Flock.Recursive.ValueInner

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

/-- **Check 5's evaluations at one statement `st`** of a `rec-residuals` unit, at its source `src`: `checkValue`'s at
`v` (the verifier's rows under `verifierSalt` of META's `program_sha512`) and `checkZero`'s at `out` (two zero rows of
`OUT_WORDS` under `zeroSalt`). None at any other unit. -/
def resSetupSt (src : Option RecCoins.Source) (st : Stmt) : List Msg :=
  match RecResiduals.params st.c.unit.name, src with
  | some (_, _, _, j), some f =>
    match (f.load st.c.unit.name).toOption, (st.c.metaJ.getObjValAs? String "program_sha512").toOption with
    | some inn, some prog => match FlockSoundness.Discharge.Recursion.residualRows inn j with
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
`row` (the verifier's, `8 nv` words at least) and those its `out` row forms `fo` read are words all zero, and the copy's
`v` columns and residual columns are those forms' bits, the copy's residual columns are zero and its `v` reads `row`'s
elements (`ResTie` at `wordsElems nv row`). -/
theorem resTie_of_rows {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {v : ℕ → Bool} {w : Claims.Flock.Wit}
    {fv fo : List HmNets.Lin} {row zrow : Array ℕ} (hsz : 8 * nv ≤ row.size) (hz : ∀ i, zrow.getD i 0 = 0)
    (hvb : readForms w fv = wordBytes row) (hob : readForms w fo = wordBytes zrow)
    (hVc : ∀ i < 128 * nv, ∃ hl : i < fv.length,
      v (FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) + i) = formBit w fv[i])
    (hOc : ∀ q < 128 * S.res.length, ∃ hl : q < fo.length,
      v (FlockVBridge.OfVStar.partBase S ns nv + q) = formBit w fo[q]) :
    ResTie S ns nv (wordsElems nv row) v := by
  refine ⟨fun q hq => ?_, funext fun e => ?_⟩
  · obtain ⟨hl, hq'⟩ := hOc q hq
    rw [hq', ← bitL_readForms hl, hob]
    by_cases hzq : q < 16 * zrow.size
    · rw [wordBytes, bitL_wordBytes _ _ (by simpa using hzq), List.getD_eq_getElem?_getD, Array.getElem?_toList,
        ← Array.getD_eq_getD_getElem?, hz]
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
      rcases Nat.lt_or_ge (8 * e + t / 16) row.size with hk | hk
      · simp [List.getD_eq_getElem?_getD, Array.getD, hk]
      · simp [List.getD_eq_getElem?_getD, Array.getD, Nat.not_lt.2 hk]
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
  have hT' := vcallsAt_append_true hT (D := D) (xO := xO)
  have hcall : ∀ e ∈ portCalls c nm q n, e.1 ++ readForms w e.2.1 ∈
      (L.calls x s w ++ vcallsAt H D xO).map Prod.fst := fun e he =>
    List.mem_map.2 ⟨_, List.mem_append_left _ (List.mem_map.2 ⟨e, hmem e he, rfl⟩), rfl⟩
  have hver : ∀ u ∈ D.setup xO, u ∈ (L.calls x s w ++ vcallsAt H D xO).map Prod.fst := fun u hu =>
    List.mem_map.2 ⟨(u, H u), List.mem_append_right _ (List.mem_map.2 ⟨u, hu, rfl⟩), rfl⟩
  have hy : FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n) ∈
      (L.calls x s w ++ vcallsAt H D xO).map Prod.fst := by
    have h := hcall _ List.mem_cons_self
    dsimp only at h
    rw [show Hm96.Default512.saltPrefix.data.toList = FlockSoundness.Refine.K512.saltPrefix from rfl] at h
    exact h
  subst hH
  refine commit_tie (σ := salt.data.toList) hT' hnc
    (hcall _ (List.mem_cons_of_mem _ List.mem_cons_self)) (hver _ hv) hy ?_ ?_
  · rw [show FlockSoundness.Refine.K512.saltPrefix ++ salt.data.toList =
      (Hm96.Default512.saltPrefix ++ salt).data.toList by
        rw [ByteArray.data_append, Array.toList_append]; rfl]
    exact hver _ hs
  · rw [← hslot, ← hpub, RecCoins.commitRow, rowDigest_eq, commitString_exec]

theorem size_padTo_ge (ws : Array ℕ) (n : ℕ) : ws.size ≤ (RecCoins.padTo ws n).size := by
  simp [RecCoins.padTo]

/-- **Check 5's tie at one instance**: `port_tie` at the instance's `v` row (the verifier's `rowV`) and its `out` row
(a zero row `zr`), each port's word count the padded row's, and the copy's `v` and residual columns those rows' form
bits, give `ResTie` at `rowV`'s elements. -/
theorem resTie_copy {Coin' : Type} {H : Msg → Msg} (hH : H = FlockDeploy.sha512) {D : Claims.Recursion.Deploy Coin'}
    {L : LayoutB} {x xO : Msg} {s : ℕ} {w : Claims.Flock.Wit} {c : Circuit} {qv qo n : ℕ} {ptv pto : Port}
    (hT : ∀ e ∈ L.calls x s w, H e.1 = e.2) (hnc : (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false)
    (hmv : ∀ e ∈ portCalls c "v" qv n, e ∈ L.shaAt x s) (hmo : ∀ e ∈ portCalls c "out" qo n, e ∈ L.shaAt x s)
    (hslv : readForms w (hmCommit c qv n) = FlockSoundness.Binding.commitString
      (rowPfx c qv ++ readForms w (rowForms c "v" qv n)) (readForms w (hmSalt c qv n)))
    (hslo : readForms w (hmCommit c qo n) = FlockSoundness.Binding.commitString
      (rowPfx c qo ++ readForms w (rowForms c "out" qo n)) (readForms w (hmSalt c qo n)))
    (hptv : c.ports[qv]? = some ptv) (hpto : c.ports[qo]? = some pto)
    {rowV zr : Array ℕ} {sv so : ByteArray}
    (hwv : ptv.words = (RecCoins.padTo rowV (Registered.rowWords rowV.size 16)).size)
    (hwo : pto.words = (RecCoins.padTo zr (Registered.rowWords zr.size 16)).size) (hzr : ∀ i, zr.getD i 0 = 0)
    (hpv : (RecCoins.commitRow rowV sv).data.toList = readForms w (hmCommit c qv n))
    (hpo : (RecCoins.commitRow zr so).data.toList = readForms w (hmCommit c qo n))
    (hvv : rowInput (RecCoins.padTo rowV (Registered.rowWords rowV.size 16)) ∈ D.setup xO)
    (hsv : (Hm96.Default512.saltPrefix ++ sv).data.toList ∈ D.setup xO)
    (hvo : rowInput (RecCoins.padTo zr (Registered.rowWords zr.size 16)) ∈ D.setup xO)
    (hso : (Hm96.Default512.saltPrefix ++ so).data.toList ∈ D.setup xO)
    {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {v : ℕ → Bool} (hsz : 8 * nv ≤ rowV.size)
    (hVc : ∀ i < 128 * nv, ∃ hl : i < (rowForms c "v" qv n).length,
      v (FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) + i) = formBit w (rowForms c "v" qv n)[i])
    (hOc : ∀ q < 128 * S.res.length, ∃ hl : q < (rowForms c "out" qo n).length,
      v (FlockVBridge.OfVStar.partBase S ns nv + q) = formBit w (rowForms c "out" qo n)[q]) :
    ResTie S ns nv (wordsElems nv rowV) v := by
  have hv := readForms_of_rowInput hptv hwv (port_tie hH hT hnc hmv hslv hpv hvv hsv)
  have ho := readForms_of_rowInput hpto hwo (port_tie hH hT hnc hmo hslo hpo hvo hso)
  rw [← wordsElems_padTo nv rowV (Registered.rowWords rowV.size 16)]
  exact resTie_of_rows (le_trans hsz (size_padTo_ge _ _)) (fun i => by rw [getD_padTo, hzr]) hv ho hVc hOc

theorem mem_valPorts {r : XR} {s : ℕ} {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {nm : String} (hnm : nm ∈ resPorts) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) {n : ℕ} (hn : n < 2) : (nm, q, n) ∈ valPorts r s := by
  have hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner j nm' pt) := by
    rw [hss]; exact hsv
  unfold valPorts
  rw [hst, hc]
  dsimp only
  exact List.mem_flatMap.2 ⟨nm, hnm, by rw [hq]; exact List.mem_map.2 ⟨n, List.mem_range.2 hn, rfl⟩⟩

/-- **`layVR` records check 5's calls**: at an algebra part, each instance `n < 2`'s hm96 calls at ports `v` and `out`
are among part `s`'s recorded calls. -/
theorem portCalls_sub_layVR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {j : ℕ} {nm' : String} {pt : VStar.Part}
    (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {nm : String} (hnm : nm ∈ resPorts) {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) {n : ℕ}
    (hn : n < 2) : ∀ e ∈ portCalls c nm q n, e ∈ (layVR elem).shaAt x s := fun e he => by
  show e ∈ (layoutBTops _).shaAt x s ++ ((readX x).map fun r => valCalls r s).getD []
  rw [hr]
  refine List.mem_append_right _ ?_
  simp only [Option.map_some, Option.getD_some, valCalls, hc]
  exact List.mem_flatMap.2 ⟨_, mem_valPorts hss hsv hc hnm hq hn, he⟩

/-- **Check 5's tie at one instance of a part, at `layVR`**: `resTie_copy` at the compiled record over `layVR`, whose
calls hold check 5's (`portCalls_sub_layVR`), with `v`'s slot at its unit input columns (`SlotCommits`) and `out`'s
at its output bits (`outForms`). What is left are facts of the code: the slots' commit strings, check 5's public rows
and the setup's evaluations of them (`hset` gives these at `resSetup`), and where the copy's columns sit. -/
theorem resTie_layVR {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit}
    (hT : ∀ e ∈ (layVR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((layVR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (layVR elem) IV) xO)).isSome = false)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {qv qo n : ℕ} (hn : n < 2)
    (hqv : c.ports.findIdx? (·.name == "v") = some qv) (hqo : c.ports.findIdx? (·.name == "out") = some qo)
    (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    (hslv : SlotCommits c "v" qv n w)
    (hslo : readForms w (hmCommit c qo n) = FlockSoundness.Binding.commitString
      (rowPfx c qo ++ readForms w (outForms c qo n)) (readForms w (hmSalt c qo n)))
    {ptv pto : Port} (hptv : c.ports[qv]? = some ptv) (hpto : c.ports[qo]? = some pto)
    {rowV zr : Array ℕ} {sv so : ByteArray}
    (hwv : ptv.words = (RecCoins.padTo rowV (Registered.rowWords rowV.size 16)).size)
    (hwo : pto.words = (RecCoins.padTo zr (Registered.rowWords zr.size 16)).size) (hzr : ∀ i, zr.getD i 0 = 0)
    (hpv : (RecCoins.commitRow rowV sv).data.toList = readForms w (hmCommit c qv n))
    (hpo : (RecCoins.commitRow zr so).data.toList = readForms w (hmCommit c qo n))
    (hvv : rowInput (RecCoins.padTo rowV (Registered.rowWords rowV.size 16)) ∈ b.setup xO)
    (hsv' : (Hm96.Default512.saltPrefix ++ sv).data.toList ∈ b.setup xO)
    (hvo : rowInput (RecCoins.padTo zr (Registered.rowWords zr.size 16)) ∈ b.setup xO)
    (hso : (Hm96.Default512.saltPrefix ++ so).data.toList ∈ b.setup xO)
    {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {v : ℕ → Bool} (hsz : 8 * nv ≤ rowV.size)
    (hVc : ∀ i < 128 * nv, ∃ hl : i < (rowForms c "v" qv n).length,
      v (FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) + i) = formBit w (rowForms c "v" qv n)[i])
    (hOc : ∀ q < 128 * S.res.length, ∃ hl : q < (rowForms c "out" qo n).length,
      v (FlockVBridge.OfVStar.partBase S ns nv + q) = formBit w (rowForms c "out" qo n)[q]) :
    ResTie S ns nv (wordsElems nv rowV) v := by
  have hfv : rowForms c "v" qv n = ((rowHidden c "v" n).getD []).map HmNets.Lin.var := by
    rw [rowForms, if_pos hvin]
  have hfo : rowForms c "out" qo n = outForms c qo n := by rw [rowForms, if_neg hoout]
  refine resTie_copy rfl hT hnc
    (portCalls_sub_layVR elem hr hss hsv hc (by simp [resPorts]) hqv hn)
    (portCalls_sub_layVR elem hr hss hsv hc (by simp [resPorts]) hqo hn)
    (by rw [hfv]; exact hslv) (by rw [hfo]; exact hslo) hptv hpto hwv hwo hzr hpv hpo hvv hsv' hvo hso hsz hVc hOc

/-! ## Check 5's public rows -/

theorem bytes_eq_of_not_bne {a b : ByteArray} (h : ¬ (a != b) = true) : a = b := by
  have hb : (a == b) = true := by simpa [bne] using h
  exact ByteArray.ext (beq_iff_eq.1 (hb : (a.data == b.data) = true))

/-- **What `checkRows` checked of the statement's rows**: the reads file's entry `reg` of the port, unit `i` reading
row `i`, and each of the port's table rows `r` (at the entry's positions) the commit string of the verifier's row at
its position, under `salt` of META's `program_sha512`. -/
theorem checkRows_tableRows {salt : String → ℕ → ByteArray} {c : Circuit} {pub : Public}
    {own : Array Registered.Port} {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (h : RecCoins.checkRows salt c pub own port value rows what = .ok ()) :
    ∃ (reg : Registered.Port) (prog : String) (t : Tables) (p : ℕ) (positions : Array ℕ) (paths : Array ByteArray),
      own.find? (·.name == port) = some reg ∧ reg.reads = (Array.range rows.size).map some ∧
      c.metaJ.getObjValAs? String "program_sha512" = .ok prog ∧ pub.tables = some t ∧
      (List.range c.ports.size).find? (fun p => c.ports[p]!.name == port) = some p ∧
      Registered.entry pub reg = .ok (positions, paths) ∧
      ∀ r < positions.size, Registered.tableRow pub t p r =
        ((Array.range rows.size).map fun i => RecCoins.commitRow rows[i]! (salt prog i)).getD positions[r]!
          ByteArray.empty := by
  unfold RecCoins.checkRows at h
  dsimp only at h
  split at h
  · rename_i reg hf
    split at h
    · cases h
    split at h
    · cases h
    split at h
    · cases h
    rename_i hr
    split at h
    · rename_i prog hprog
      split at h
      · rename_i t ht
        split at h
        · rename_i p hp
          obtain ⟨⟨positions, paths⟩, hent, h⟩ := FlockSoundness.except_bind_ok h
          obtain ⟨u, hloop, -⟩ := FlockSoundness.except_bind_ok h
          cases u
          simp only [bne_iff_ne, ne_eq, not_not] at hr
          have hout := FlockSoundness.forIn_range_yield_ok (fun a s hs => by
            split at hs
            · cases hs
            · cases hs; rfl) hloop
          refine ⟨reg, prog, t, p, positions, paths, hf, hr, hprog, ht, hp, hent, fun r hr' => ?_⟩
          have hb := hout r (Nat.zero_le _) hr'
          split at hb
          · cases hb
          · rename_i hne
            exact bytes_eq_of_not_bne hne
        · cases h
      · cases h
    · cases h
  · cases h

/-- **Check 5's row at a slot**: at a staged part the witness checks against, where the setup's table passed
`checkRows` at port `port` (the reads file's `reg`, of the verifier's `rows` under `salt`) and labels its instances in
order (`hidx`), port `q` (the circuit's `port`)'s hm96 slot at instance `n < 2` reads as the commit string of the
verifier's row `n`. The slot is its instance's public row (`slotPub_table`), at the ref `checkRegistered`'s
`checkPort` reads row `n` at (`reg.reads` is `checkRows`' identity), which `checkRows` held to `commitRow`. -/
theorem check5_bytes {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own)
    {salt : String → ℕ → ByteArray} {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (hrows : RecCoins.checkRows salt sts[0].c sts[0].pub own port value rows what = .ok ())
    {q : ℕ} (hq : c.ports.findIdx? (·.name == port) = some q) {n : ℕ} (hn2 : n < 2) (hn : n < rows.size)
    (hnp : n < sts[0].pub.n) :
    ∃ prog, sts[0].c.metaJ.getObjValAs? String "program_sha512" = .ok prog ∧
      readForms w (hmCommit c q n) = (RecCoins.commitRow rows[n]! (salt prog n)).data.toList := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', -, h0p, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  obtain ⟨-, hset⟩ := setupO_some hsu
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  obtain ⟨hsz1, -⟩ := oneFull_staged hS hsu
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hsz : p.sts.size = 1 := by rw [hp]; exact hsz1
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0p
  obtain ⟨L, hz⟩ := layout_of_check_regs hck hx ht hpi h0p
  have hz1 : (L.model (FlockSoundness.Refine.regionsOf p.sts[0])).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨_, hset0⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset0
  rw [hpi] at hset0
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset0
  obtain ⟨d, hd, -⟩ := hall 0 h0p
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0p
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0p).2.2
  obtain ⟨-, -, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L _ hz1
  subst hc0
  have hreg := RecIns.setup_registered hset sts[0] (Array.getElem_mem h0)
  rw [← e0] at hrows hidx hnp hreg ⊢
  rw [hown] at hreg
  obtain ⟨reg, prog, t, pp, positions, paths, hf, hreads, hprog, htt, hpp, hent, hrow⟩ := checkRows_tableRows hrows
  have hmem : reg ∈ own := Array.mem_of_find?_eq_some hf
  have hname : reg.name = port := by simpa using Array.find?_some hf
  obtain ⟨t', idx, ht2, hidx2, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hreg)
  rw [htt] at ht2
  cases ht2
  rw [hidx] at hidx2
  cases hidx2
  obtain ⟨p', positions', paths', hp', hent', hall'⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  obtain ⟨p'', positions'', paths'', hp'', hent'', hsize⟩ := FlockC3.PubRows.checkPort_size (hports _ hmem)
  rw [hname, hpp] at hp' hp''
  cases hp'
  cases hp''
  rw [hent] at hent' hent''
  cases hent'
  cases hent''
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hqq : pp = q := find_range_findIdx (f := fun (x : Port) => x.name == port) hpp hq
  rw [hqq] at hrow hall' hsize
  obtain ⟨hlt, hrd, -⟩ := hall' n hnp
  rw [hreads, range_getD hnp] at hrd
  have hpos : positions.getD (Registered.ref t p.sts[0].c.ports.size q n) 0 = n := by
    rw [Array.getD_eq_getD_getElem?]
    simp [hn] at hrd
    omega
  have hlt' : Registered.ref t p.sts[0].c.ports.size q n < positions.size := by rw [hsize]; exact hlt
  have hsl := slotPub_table ht' hd (by omega) hv1 L hz1 hq' hnp (div_lt_blocks_two hn2 hnbl)
  rw [digest_tables htt, hrow _ hlt', getElem!_pos positions _ hlt'] at hsl
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt', Option.getD_some] at hpos
  rw [hpos] at hsl
  refine ⟨prog, hprog, ?_⟩
  rw [hsl]
  simp [Array.getD_eq_getD_getElem?, hn]

/-- **An algebra part's `v` slot commits**: `v` is an input of the part's unit (`innerSub`'s outputs are `out` alone), so
its row lies in the table's leaves (`in_block`), and `slotCommits_table` gives its commit string, as
`hmSlotsCommit_of_full` does for the held rows. -/
theorem slotV_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss) {j : ℕ} {nm' : String} {pt : VStar.Part}
    (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    {qv n : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv) (hn2 : n < 2) :
    SlotCommits c "v" qv n w := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨-, i, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨L, regs, hz⟩ := Flock.Readable.layout_of_check hck hx ht hpi h0
  have hz1 : (L.model regs).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨u, -, hupv, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨-, -, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L regs hz1
  subst hc0
  have hout : "v" ∉ (VStar.Stmt.inner j nm' pt).sub.outputs := by show "v" ∉ ["out"]; decide
  exact slotCommits_table ht' hd (by omega) hv1 hl L regs hz1 hqv (in_block hrows hout hqv)
    (div_lt_blocks_two hn2 hnbl)

/-- **Where a copy's `v` columns sit**: at a staged algebra part, instance `o·g_c + g`'s `v` row (`rowHidden_eq`) is
copy `(0, o, g, 0)`'s columns from the port's first word past the first input group's column, which lie below the unit's
constant (`in_block`, `leaves_le_constPos`). So `resTie_copyOf`'s `hVc` holds wherever VBridge's `vBase` is that column
(`hvb`). -/
theorem vCols_of_base {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {qv : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv) (hvin : qv < HmOut.nIn c)
    {ptv : Port} (hptv : c.ports[qv]? = some ptv) {ns : List ℕ} {nv : ℕ} (hwb : 8 * nv ≤ ptv.words)
    {o g n : ℕ} (hg : g < c.g) (hn : o * c.g + g = n)
    (hvb : FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) =
      (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (c.ports.extract 0 qv).foldl (· + ·.words) 0) :
    ∀ i < 128 * nv, ∃ hl : i < (rowForms c "v" qv n).length,
      Flock.Readable.copyOf p 0 ht w o g 0 (FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) + i) =
        formBit w (rowForms c "v" qv n)[i] := by
  obtain ⟨p', r', sv, hx', ht', hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hx] at hx'
  cases hx'
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hS'
  have hp : p = { i0.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have hsz : p.sts.size = 1 := by rw [hp]; exact (oneFull_staged hS hsu).1
  have hc0 := partCircuit_staged hS hx ht' hpi hr hpc h0
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht' hpi h0
  obtain ⟨u, -, hupv, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  subst hc0
  have hout : "v" ∉ (VStar.Stmt.inner j nm' pt).sub.outputs := by show "v" ∉ ["out"]; decide
  have hN := in_block hrows hout hqv
  obtain ⟨hqs, -, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqv
  have hpq : p.sts[0].c.ports[qv] = ptv := Option.some.inj ((Array.getElem?_eq_getElem hqs).symm.trans hptv)
  rw [getElem!_pos p.sts[0].c.ports qv hqs] at hN
  have hrh := rowHidden_eq n hl hqv hqs hN
  have hbits : ptv.bits = 16 * ptv.words := hpq ▸ (hv1 _ (Array.getElem_mem_toList hqs)).2.1
  rw [hpq] at hN hrh
  have hdiv : n / p.sts[0].c.g = o := by
    subst hn
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hg, Nat.zero_add]
  have hmod : n % p.sts[0].c.g = g := by
    subst hn
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hg]
  intro i hi
  have hcp := leaves_le_constPos P hl (by omega)
  have hlen : (rowForms p.sts[0].c "v" qv n).length = ptv.bits := by simp [rowForms, hvin, hrh]
  refine ⟨by omega, ?_⟩
  have hel : (rowForms p.sts[0].c "v" qv n)[i]'(by omega) = HmNets.Lin.var (n / p.sts[0].c.g * 2 ^ p.sts[0].c.kLog +
      (p.sts[0].c.unitRange.slot (n % p.sts[0].c.g * p.sts[0].c.upv) +
        ((p.sts[0].c.unit.inGroups.getD 0 default).col * Flock.WORD +
          16 * (p.sts[0].c.ports.extract 0 qv).foldl (· + ·.words) 0 + i))) := by
    simp [rowForms, hvin, hrh]
  rw [hel, formBit_var, hvb, Claims.Recursion.VStar.copyOf_eq (by omega), hdiv, hmod]
  simp [Claims.Recursion.VStar.copyIdx, hsz, hupv]

/-! ## Where `v`'s port starts -/

/-- **The part check's counts**: at an algebra part's statement, V\*'s row ports' elements and `v`'s are those of the
verifier's part at the unit's name, shape and META. -/
theorem checkPart_ns {j : ℕ} {nm : String} {pt : VStar.Part} {c : Circuit}
    (h : RecVStar.checkPart (.inner j nm pt) c = .ok ()) {m k cl jj : ℕ} {sh : Flock.Shape} {holo : Bool}
    {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h1 : RecResiduals.params c.unit.name = some (m, k, cl, jj)) (h2 : RecResiduals.shape? m k cl = some sh)
    (h3 : RecResiduals.holoOf c = .ok holo) (h4 : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv)) :
    pt.S.ports.map (·.2) = ns ∧ pt.S.gf.nv = nv := by
  unfold RecVStar.checkPart at h
  dsimp only at h
  rw [h1] at h
  dsimp only at h
  rw [h2] at h
  dsimp only at h
  rw [h3] at h
  dsimp only at h
  rw [h4] at h
  dsimp only at h
  by_cases hc : (pt.S.ports.map (·.2) != ns || pt.S.gf.nv != nv) = true
  · rw [if_pos hc] at h
    cases h
  · simpa only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Classical.not_not] using hc

/-- V\*'s `InnerRepCheck` unit has no row port named `v` (`checkRepS`). -/
theorem noV_of_innerUnit {S : VStar.RepS} {nm : String} {u : VStar.UnitC} (h : VStar.innerUnit S nm = .ok u) :
    "v" ∉ S.ports.map (·.1) := by
  unfold VStar.innerUnit at h
  cases hc : VStar.checkRepS S with
  | error e => simp [hc, bind, Except.bind] at h
  | ok _ =>
    intro hv
    unfold VStar.checkRepS at hc
    have h1 : ((S.ports.map (·.1)).eraseDups.length != (S.ports.map (·.1)).length ||
        (S.ports.map (·.1)).contains "v" || S.ports.any (·.2 < 1)) = true := by
      simp [hv]
    simp only [h1, ↓reduceIte] at hc
    contradiction

/-- The words of the rows before a run `A` of a circuit's row ports. -/
theorem take_words {c : Circuit} {A B : List (String × ℕ × ℕ)}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = A ++ B) :
    (c.ports.extract 0 A.length).foldl (· + ·.words) 0 = (A.map (·.2.1)).sum := by
  have he : (c.ports.extract 0 A.length).foldl (· + ·.words) 0 =
      ((c.ports.toList.take A.length).map (·.words)).sum := extract_words (·.words) c.ports A.length
  rw [he]
  have h := congrArg (fun l => (l.take A.length).map (fun r : String × ℕ × ℕ => r.2.1)) hr
  simp only [List.take_left] at h
  rw [← h, ← List.map_take, List.map_map]
  rfl

/-- **`v`'s port starts past the row ports' words**: at an algebra part's statement, `checkParsed`'s row ports are
V\*'s (`vstarRows`): the row ports, `v`, then `out`. `checkRepS` names no row port `v`, so `v` is the port after them. -/
theorem v_words {j : ℕ} {nm : String} {pt : VStar.Part} {c : Circuit} {u : VStar.UnitC}
    (hu : (VStar.Stmt.inner j nm pt).unit = .ok u)
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      RecVStar.vstarRows (VStar.Stmt.inner j nm pt).sub u)
    {qv : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv) :
    (c.ports.extract 0 qv).foldl (· + ·.words) 0 = (pt.S.ports.map fun p => VStar.portWordsOf p.2).sum := by
  have hu' : VStar.innerUnit pt.S nm = .ok u := hu
  have hnv := noV_of_innerUnit hu'
  obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.innerUnit_ok hu'
  have hlen : (pt.S.ports.map (·.1)).length = (pt.S.ports.map fun p => VStar.portWordsOf p.2).length := by simp
  have hr' : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      ((pt.S.ports.map (·.1)).zip (pt.S.ports.map fun p => VStar.portWordsOf p.2)).map
        (fun p => (p.1, p.2, 16 * p.2)) ++
      (("v", VStar.portWordsOf pt.S.gf.nv, 16 * VStar.portWordsOf pt.S.gf.nv) ::
        (["out"].zip u.outPorts).map fun p => (p.1, (p.2 + 63) / 64 * 64, 16 * ((p.2 + 63) / 64 * 64))) := by
    rw [hr]
    have e1 : (VStar.Stmt.inner j nm pt).sub.inputs = pt.S.ports.map (·.1) ++ ["v"] := rfl
    have e2 : (VStar.Stmt.inner j nm pt).sub.outputs = ["out"] := rfl
    unfold RecVStar.vstarRows
    rw [e1, e2, hin, FlockVBridge.OfVStar.repPorts, List.zip_append hlen, List.map_append, List.append_assoc]
    rfl
  have hAl : (((pt.S.ports.map (·.1)).zip (pt.S.ports.map fun p => VStar.portWordsOf p.2)).map
      (fun p => (p.1, p.2, 16 * p.2))).length = pt.S.ports.length := by simp
  have hfind : (c.ports.toList.map (fun p => (p.name, p.words, p.bits))).findIdx? (fun r => r.1 == "v") =
      some pt.S.ports.length := by
    rw [hr', List.findIdx?_append, List.findIdx?_cons, hAl]
    have hnone : (((pt.S.ports.map (·.1)).zip (pt.S.ports.map fun p => VStar.portWordsOf p.2)).map
        (fun p => (p.1, p.2, 16 * p.2))).findIdx? (fun r => r.1 == "v") = none := by
      refine List.findIdx?_eq_none_iff.2 fun x hx => ?_
      obtain ⟨⟨a, b⟩, hy, rfl⟩ := List.mem_map.1 hx
      have hne : a ≠ "v" := fun he => hnv (he ▸ (List.of_mem_zip hy).1)
      simpa using hne
    rw [hnone]
    simp
  rw [List.findIdx?_map] at hfind
  have hq : c.ports.findIdx? (·.name == "v") = some pt.S.ports.length := by
    rw [← List.findIdx?_toArray, Array.toArray_toList]
    exact hfind
  rw [hqv] at hq
  cases hq
  have hw := take_words hr'
  rw [hAl] at hw
  rw [hw, List.map_map]
  congr 1
  exact List.map_snd_zip (le_of_eq hlen.symm)

/-- The verifier's `rec-residuals` unit's first input group starts at column 0 (`layout`'s). -/
theorem inCol_unitLaid {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String} {net : Flock.Net}
    (hn : HmNets.netOf name (RecResiduals.unitLaid S ns nv) = .ok net) : (net.inGroups.getD 0 default).col = 0 := by
  obtain ⟨-, -, -, -, -, hi, -⟩ := Flock.Net.ofRows_ok hn
  rw [hi]
  simp [RecResiduals.unitLaid, FlockSoundness.Discharge.Hm.layout_inGroups]

/-- **VBridge's `vBase` is the `v` port's column** (`resTie_copyOf`'s `hvb`, from the verifier's checks): at a staged
algebra part whose unit the verifier accepted as part `jj` of its shape, the unit is the verifier's `unitLaid`
(`check_part_at`), whose first input group starts at column 0 (`inCol_unitLaid`), and `v`'s port starts past the row
ports' words (`v_words`), which the part check holds to the verifier's part (`checkPart_ns`). -/
theorem vBase_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {qv : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv)
    {m k cl jj : ℕ} {sh : Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hres : RecResiduals.params c.unit.name = some (m, k, cl, jj)) (hshp : RecResiduals.shape? m k cl = some sh)
    (hholo : RecResiduals.holoOf c = .ok holo) (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv)) :
    FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) =
      (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (c.ports.extract 0 qv).foldl (· + ·.words) 0 := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  obtain ⟨-, hpk, hpa⟩ := htab 0 h0
  subst hc0
  obtain ⟨S', ns', nv', hpart', hn⟩ := FlockVBridge.OfVStar.check_part_at P.recResiduals hres hshp hholo
  rw [hpart] at hpart'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hpart'
  obtain ⟨rfl, rfl, rfl⟩ := hpart'
  obtain ⟨hns, -⟩ := checkPart_ns hpk hres hshp hholo hpart
  obtain ⟨u, hu, -, -, -, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hpa
  rw [inCol_unitLaid hn, Nat.zero_mul, Nat.zero_add, v_words hu hrows hqv, FlockVBridge.OfVStar.vBase_partRep, ← hns,
    List.map_map]
  rfl

/-- **Check 5's tie at one instance of a staged `rec-residuals` part**: `resTie_layVR` with check 5's public rows from
the statement's setup. The setup ran `RecCoins.check` at the part's table, so `checkResiduals` at its inner session
(`residualsCheck_of_check`, `residualsBoth_of_check`), whose `v` and `out` rows the slots read (`check5_bytes`), and
the deployed `Brings.setup` lists their evaluations (`hset`). What is left are facts of the code: the table's labels
(`hidx`), the unit's name being no other V\* unit's, the `out` slot (`v`'s is `slotV_staged`), the ports' word counts
and where the copy's columns sit. -/
theorem resTie_staged {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit}
    (hT : ∀ e ∈ (layVR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((layVR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (layVR elem) IV) xO)).isSome = false)
    (hS : StagedPart b.st s x xO) (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own)
    {m k cl jj : ℕ} (hres : RecResiduals.params sts[0].c.unit.name = some (m, k, cl, jj))
    (hso : RecCoins.stackedParams sts[0].c.unit.name = none) (hop : RecOpening.params sts[0].c.unit.name = none)
    (hsp : RecSparse.params sts[0].c.unit.name = none) (ho : RecOpen.params sts[0].c.unit.name = none)
    {f : RecCoins.Source} (hf : RecCoins.Source.atC3 i.src i.inner i.commits = some f)
    (hset : ∀ u ∈ resSetup xO, u ∈ b.setup xO)
    {qv qo n : ℕ} (hn2 : n < 2) (hnp : n < sts[0].pub.n)
    (hqv : c.ports.findIdx? (·.name == "v") = some qv) (hqo : c.ports.findIdx? (·.name == "out") = some qo)
    (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    (hslo : readForms w (hmCommit c qo n) = FlockSoundness.Binding.commitString
      (rowPfx c qo ++ readForms w (outForms c qo n)) (readForms w (hmSalt c qo n))) :
    ∃ (inn : RecCoins.Inner) (rows : Array (Array ℕ)), f.load sts[0].c.unit.name = .ok inn ∧
      FlockSoundness.Discharge.Recursion.residualRows inn jj = some rows ∧
      ∀ {ptv pto : Port} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {v : ℕ → Bool},
        c.ports[qv]? = some ptv → c.ports[qo]? = some pto →
        ptv.words = (RecCoins.padTo rows[n]! (Registered.rowWords rows[n]!.size 16)).size →
        pto.words = (RecCoins.padTo (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!
          (Registered.rowWords (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!.size 16)).size →
        n < rows.size → 8 * nv ≤ rows[n]!.size →
        (∀ i < 128 * nv, ∃ hl : i < (rowForms c "v" qv n).length,
          v (FlockVBridge.OfVStar.vBase (FlockVBridge.OfVStar.partRep ns nv) + i) =
            formBit w (rowForms c "v" qv n)[i]) →
        (∀ q < 128 * S.res.length, ∃ hl : q < (rowForms c "out" qo n).length,
          v (FlockVBridge.OfVStar.partBase S ns nv + q) = formBit w (rowForms c "out" qo n)[q]) →
        ResTie S ns nv (wordsElems nv rows[n]!) v := by
  obtain ⟨-, hset0⟩ := setupO_some hsu
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset0) 0 h0
  rw [hown, hf] at hcoin
  obtain ⟨inn, hinn, hcr⟩ := FlockSoundness.Discharge.Recursion.residualsCheck_of_check hcoin hso hop hsp ho hres
  obtain ⟨rows, what, hrows, hcv, hcz⟩ := FlockSoundness.Discharge.Recursion.residualsBoth_of_check hcr
  refine ⟨inn, rows, hinn, hrows, fun hptv hpto hwv hwo hnr hsz hVc hOc => ?_⟩
  have hzn : n < (RecCoins.zeroRows 2 RecCoins.OUT_WORDS).size := by simpa [RecCoins.zeroRows] using hn2
  obtain ⟨prog, hprog, hpv⟩ := check5_bytes hS hck hr hpc hsu h0 hidx hown hcv hqv hn2 hnr hnp
  obtain ⟨-, -, hpo⟩ := check5_bytes hS hck hr hpc hsu h0 hidx hown hcz hqo hn2 hzn hnp
  have hst : resSetupSt (RecCoins.Source.atC3 i.src i.inner i.commits) sts[0] =
      commitIns rows (RecCoins.verifierSalt prog RecCoins.V_VALUE) ++
        commitIns (RecCoins.zeroRows 2 RecCoins.OUT_WORDS) (RecCoins.zeroSalt RecCoins.ZERO_VALUE) := by
    unfold resSetupSt
    simp only [hres, hf, hinn, hprog, Except.toOption, hrows]
  have hmemS : ∀ u ∈ resSetupSt (RecCoins.Source.atC3 i.src i.inner i.commits) sts[0], u ∈ b.setup xO :=
    fun u hu => hset u (by unfold resSetup; rw [hsu]; exact List.mem_flatMap.2 ⟨sts[0], by simp, hu⟩)
  rw [hst] at hmemS
  have hci : ∀ {rs : Array (Array ℕ)} {sl : ℕ → ByteArray}, n < rs.size →
      rowInput (RecCoins.padTo rs[n]! (Registered.rowWords rs[n]!.size 16)) ∈ commitIns rs sl ∧
        (Hm96.Default512.saltPrefix ++ sl n).data.toList ∈ commitIns rs sl := fun hn' => by
    unfold commitIns
    exact ⟨List.mem_flatMap.2 ⟨n, List.mem_range.2 hn', List.mem_cons_self⟩,
      List.mem_flatMap.2 ⟨n, List.mem_range.2 hn', List.mem_cons_of_mem _ List.mem_cons_self⟩⟩
  have hzr : ∀ i, ((RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!).getD i 0 = 0 := fun i => by
    rw [Array.getD_eq_getD_getElem?]
    by_cases hi : i < ((RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!).size
    · rw [Array.getElem?_eq_getElem hi, Option.getD_some]
      have hz0 := FlockSoundness.Discharge.Recursion.zeroRows_get 2 RecCoins.OUT_WORDS n i
      rwa [getElem!_pos ((RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!) i hi] at hz0
    · rw [Array.getElem?_eq_none (Nat.le_of_not_lt hi)]
      rfl
  exact resTie_layVR hT hnc hr hss hsv hpc hn2 hqv hqo hvin hoout
    (slotV_staged hS hck hr hss hsv hpc hqv hn2) hslo hptv hpto hwv hwo hzr hpv.symm hpo.symm
    (hmemS _ (List.mem_append_left _ (hci hnr).1)) (hmemS _ (List.mem_append_left _ (hci hnr).2))
    (hmemS _ (List.mem_append_right _ (hci hzn).1)) (hmemS _ (List.mem_append_right _ (hci hzn).2))
    hsz hVc hOc

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
a `rec-residuals` statement's inner session and computes its `v` rows (`residualRows`), the verifier's values the coins give
at a copy (`vvAt` at the inner statement and the coins) are the elements of that copy's instance's row, at the
verifier's part `(S, ns, nv)`. `staged_rep_drawn` is its coin side. -/
def VVTie (untr : TrReader) (b : Brings) (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ t (ht : t < sts.size) (m k cl jj : ℕ),
      RecResiduals.params sts[t].c.unit.name = some (m, k, cl, jj) →
      ∀ (f : RecCoins.Source) (inn : RecCoins.Inner) (rows : Array (Array ℕ)),
        RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[t].c.unit.name = .ok inn →
        FlockSoundness.Discharge.Recursion.residualRows inn jj = some rows →
        ∀ (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
          RecResiduals.shape? m k cl = some sh → RecResiduals.holoOf sts[t].c = .ok holo →
          RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
          ∀ o g, g < sts[t].c.g → instOf sts[t].c o g < 2 →
            vvAt (xIOf x) ((List.range (Bk (xIOf x))).map c) (t, o, g, 0) =
              wordsElems nv rows[instOf sts[t].c o g]!

/-- **Check 5's tie at a piece's copy** (the `ResTie` conjunct `resPieces_computesAt` asks of each piece): at copy
`(0, o, g, 0)` of an instance `n` check 5 holds (`n < 2`), between the verifier's values the coins give there and the
copy's columns. `resTie_staged` at `n`, with the copy's `v` columns where its slot reads (`vCols_of_base`, VBridge's
`vBase` being the port's column by `vBase_staged`), its residual columns where the `out` slot reads (`hOc`), and the
coins' values check 5's row (`VVTie`). -/
theorem resTie_copyOf {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} {cs : ℕ → Coin} {untr : TrReader}
    {vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128} {Bk : Msg → ℕ}
    (hT : ∀ e ∈ (layVR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((layVR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (layVR elem) IV) xO)).isSome = false)
    (hsh : showsPart untr b s x xO cs)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {own : Array Registered.Port} (hown : i.base.registered = some own)
    {m k cl jj : ℕ} (hres : RecResiduals.params sts[0].c.unit.name = some (m, k, cl, jj))
    (hso : RecCoins.stackedParams sts[0].c.unit.name = none) (hop : RecOpening.params sts[0].c.unit.name = none)
    (hsp : RecSparse.params sts[0].c.unit.name = none) (ho : RecOpen.params sts[0].c.unit.name = none)
    {f : RecCoins.Source} (hf : RecCoins.Source.atC3 i.src i.inner i.commits = some f)
    (hset : ∀ u ∈ resSetup xO, u ∈ b.setup xO) (hvv : VVTie untr b vvAt Bk s x cs)
    {sh : _root_.Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hshp : RecResiduals.shape? m k cl = some sh) (hholo : RecResiduals.holoOf sts[0].c = .ok holo)
    (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv))
    {o g n : ℕ} (hg : g < sts[0].c.g) (hn : instOf sts[0].c o g = n) (hn2 : n < 2) (hnp : n < sts[0].pub.n)
    {qv qo : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv)
    (hqo : c.ports.findIdx? (·.name == "out") = some qo) (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    {ptv pto : Port} (hptv : c.ports[qv]? = some ptv) (hpto : c.ports[qo]? = some pto)
    (hslo : readForms w (hmCommit c qo n) = FlockSoundness.Binding.commitString
      (rowPfx c qo ++ readForms w (outForms c qo n)) (readForms w (hmSalt c qo n)))
    (hrows : ∀ (inn : RecCoins.Inner) (rows : Array (Array ℕ)), f.load sts[0].c.unit.name = .ok inn →
      FlockSoundness.Discharge.Recursion.residualRows inn jj = some rows →
      ptv.words = (RecCoins.padTo rows[n]! (Registered.rowWords rows[n]!.size 16)).size ∧ n < rows.size ∧
        8 * nv ≤ rows[n]!.size)
    (hwo : pto.words = (RecCoins.padTo (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!
      (Registered.rowWords (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!.size 16)).size)
    (hOc : ∀ q < 128 * S.res.length, ∃ hl : q < (rowForms c "out" qo n).length,
      Flock.Readable.copyOf p 0 ht w o g 0 (FlockVBridge.OfVStar.partBase S ns nv + q) =
        formBit w (rowForms c "out" qo n)[q]) :
    ResTie S ns nv (vvAt (xIOf x) ((List.range (Bk (xIOf x))).map cs) (0, o, g, 0))
      (Flock.Readable.copyOf p 0 ht w o g 0) := by
  obtain ⟨inn, rows, hinn, hrw, hall⟩ := resTie_staged hT hnc hsh.2 hck hr hss hsv hpc hsu h0 hidx hown hres hso hop
    hsp ho hf hset hn2 hnp hqv hqo hvin hoout hslo
  obtain ⟨hwv, hnr, hsz⟩ := hrows inn rows hinn hrw
  obtain ⟨p', -, -, hx', ht', hpi, -, -, -, -⟩ := stagedTables hsh.2
  rw [hx] at hx'
  cases hx'
  have hc0 : sts[0].c = c := by
    have e := partCircuit_staged hsh.2 hx ht' hpi hr hpc ht
    have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
    subst hp
    exact e
  have hvb := vBase_staged hsh.2 hr hss hsv hpc hqv (by rw [← hc0]; exact hres) hshp (by rw [← hc0]; exact hholo)
    hpart
  have hVc := vCols_of_base (w := w) hsh.2 hr hss hsv hpc hx ht hqv hvin hptv
    (by rw [hwv]; exact le_trans hsz (size_padTo_ge _ _)) (hc0 ▸ hg)
    (show o * c.g + g = n by rw [← hc0]; exact hn) hvb
  rw [hvv xO hsh i spec sts hsu 0 h0 m k cl jj hres f inn rows hf hinn hrw sh holo S ns nv hshp hholo hpart o g hg
    (by rw [hn]; exact hn2), hn]
  exact hall hptv hpto hwv hwo hnr hsz hVc hOc

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
