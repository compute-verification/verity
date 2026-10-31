import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.SlotsCommit
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Soundness.ExecRows
import Proofs.Flock.Recursive.ValueInner
import Proofs.Flock.VBridge.Family.OutCols
import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.Recursion.CopySiteDeployed

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
message-round ports only (`heldRows`), not of `v` or `out`. The layout of record, `DeployedR.layR`, adds them
(`DeployedR.valCalls`): each instance's salt and row calls at `v` (its input columns, `rowHidden`) and at `out` (its
output columns, `outRowForms`), and at a `rec-open` part's `coef` and `dirs` (`openPorts`, whose check 5 evaluations
are `openSetupSt`).

* `commit_tie`: two commit strings at SHA-512 agree, and both salts' and both rows' inputs are among a list of true
  evaluations with no collision, so the rows are equal.
* `valCalls_shaAt_layR`: `DeployedR.layR` records check 5's calls; `resSetup`: check 5's own evaluations at a V\*
  statement, which the deployed `Brings.setup` is to list (`hset`).
* `vCols_of_base`, `oCols_staged`: where a copy's `v` and `out` columns sit, from the verifier's checks (`checkParsed`
  holds the unit's input and output leaves to V\*'s, its words in order).
* `resTie_copyOf`, `resTie_site`: check 5's tie at a piece's copy; `resTie_site` takes the setup's site facts from
  `CopySite.copySite_staged`.
* `VVTie`: the coin tie, `vvAt` at the coins is check 5's row at the V\* statement (`staged_rep_drawn`).
* `ResLowered`: the share at the layout of record and `bdR rd wE final w`, per algebra part; `resLowered_parts` gives
  it at every part where the pieces are owned by algebra parts only, which is `DecompBR.ResLowJ`'s conclusion at each
  part of a `Joint` acceptance.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR (outRowForms rowForms portCalls resPorts openPorts valPorts valCalls)

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

/-- **Check 5's evaluations at one statement `st`** of a `rec-open` unit, at its source `src`: `checkOpen`'s at `coef`
and at `dirs` (the verifier's `coef` and `dirs` rows, `openRows`, under `verifierSalt` of META's `program_sha512`), in
`checkOpen`'s order. None at any other unit. -/
def openSetupSt (src : Option RecCoins.Source) (st : Stmt) : List Msg :=
  match RecOpen.params st.c.unit.name, src with
  | some (L, H), some f =>
    match (f.load st.c.unit.name).toOption, (st.c.metaJ.getObjValAs? String "program_sha512").toOption with
    | some inn, some prog => match FlockSoundness.Discharge.Recursion.openRows inn L H with
      | some (cr, dr) => commitIns cr (RecCoins.verifierSalt prog RecCoins.COEF_VALUE) ++
          commitIns dr (RecCoins.verifierSalt prog RecCoins.DIRS_VALUE)
      | none => []
    | _, _ => []
  | _, _ => []

/-- **Check 5's evaluations at a V\* statement `xO`**: `resSetupSt` and `openSetupSt` at each statement its recursive
setup sets up, at check 5's source (`RecCoins.Source.atC3`, as `RecIns.setup` runs `RecCoins.check`). The deployed
`Brings.setup` lists them (`hset : ∀ xO, resSetup xO ⊆ b.setup xO`). -/
def resSetup (xO : Msg) : List Msg :=
  match setupO xO with
  | some (i, _, sts) => sts.toList.flatMap fun st =>
      resSetupSt (RecCoins.Source.atC3 i.src i.inner i.commits) st ++
        openSetupSt (RecCoins.Source.atC3 i.src i.inner i.commits) st
  | none => []

theorem rowDigest_eq (row : Array ℕ) : Firewall.rowDigest row = Sha512.hash ⟨(rowInput row).toArray⟩ := by
  rw [rowInput, FlockSoundness.Refine.bytes_of_list]
  rfl

/-- A row's words as `Firewall.rowDigest` lays them after its prefix: each two bytes, little-endian. -/
def wordBytes (ws : Array ℕ) : Msg := ws.toList.flatMap fun w => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]

/-- **The copy a piece keyed `(jj, o, g, u)` reads**: instance `n = o + g` of the outer part's one table, at block
`n / c.g`'s VU `n % c.g` of that part's own circuit `c` (`Circuit.instOf`), unit `u`. The key does not know `c.g`, which
the prover chooses and check 5 does not pin, so a piece at `(jj, r, 0, 0)` is instance `r` at every `c.g ≥ 1`. Check 5
holds instances `0` and `1` (reps `0` and `1`), so only keys with `o + g < 2` are tied. -/
abbrev keyCopy (p : Flock.Readable.Parsed) (ht : 0 < p.sts.size) (w : Claims.Flock.Wit) (key : ℕ × ℕ × ℕ × ℕ) :
    ℕ → Bool :=
  Flock.Readable.copyOf p 0 ht w ((key.2.1 + key.2.2.1) / p.sts[0].c.g) ((key.2.1 + key.2.2.1) % p.sts[0].c.g)
    key.2.2.2

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

/-- **`DeployedR.layR` records check 5's calls**: each of `valCalls`' at part `s` is among part `s`'s recorded calls
(`moreCalls`). -/
theorem valCalls_shaAt_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {x : Msg} {r : XR} (hr : readX x = some r) (s : ℕ) :
    ∀ e ∈ valCalls r s, e ∈ (DeployedR.layR elem).shaAt x s := fun e he => by
  show e ∈ (layoutBTops _).shaAt x s ++ ((readX x).map fun r => DeployedR.moreCalls r s).getD []
  rw [hr, Option.map_some, Option.getD_some]
  refine List.mem_append_right _ ?_
  unfold DeployedR.moreCalls
  exact List.mem_append_left _ (List.mem_append_right _ he)

/-- **`DeployedR.layR` records check 5's calls at an algebra part**: each instance `n < 2`'s hm96 calls at ports `v`
and `out` are among part `s`'s recorded calls. -/
theorem portCalls_sub_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {j : ℕ} {nm' : String} {pt : VStar.Part}
    (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {nm : String} (hnm : nm ∈ resPorts) {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q) {n : ℕ}
    (hn : n < 2) : ∀ e ∈ portCalls c nm q n, e ∈ (DeployedR.layR elem).shaAt x s := fun e he => by
  refine valCalls_shaAt_layR elem hr s e ?_
  simp only [Option.map_some, Option.getD_some, valCalls, hc]
  exact List.mem_flatMap.2 ⟨_, mem_valPorts hss hsv hc hnm hq hn, he⟩

theorem mem_valPorts_open {r : XR} {s : ℕ} {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {nm : String} (hnm : nm ∈ openPorts) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) {i : ℕ} (hi : i < n) : (nm, q, i) ∈ valPorts r s := by
  have hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen lv LANES HH n) := by
    rw [hss]; exact hsv
  unfold valPorts
  rw [hst, hc]
  dsimp only
  exact List.mem_flatMap.2 ⟨nm, hnm, by rw [hq]; exact List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩⟩

/-- **`DeployedR.layR` records check 5's calls at a `rec-open` part**: each instance `i < n`'s hm96 calls at ports `coef`
and `dirs` are among part `s`'s recorded calls. -/
theorem portCalls_sub_layR_open (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n))
    {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c) {nm : String} (hnm : nm ∈ openPorts) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) {i : ℕ} (hi : i < n) :
    ∀ e ∈ portCalls c nm q i, e ∈ (DeployedR.layR elem).shaAt x s := fun e he => by
  refine valCalls_shaAt_layR elem hr s e ?_
  simp only [Option.map_some, Option.getD_some, valCalls, hc]
  exact List.mem_flatMap.2 ⟨_, mem_valPorts_open hss hsv hc hnm hq hi, he⟩

/-- **Check 5's tie at one instance of a part, at `DeployedR.layR`**: `resTie_copy` at the compiled record over it, whose
calls hold check 5's (`portCalls_sub_layR`), with `v`'s slot at its unit input columns (`SlotCommits`) and `out`'s
at its output bits (`outRowForms`). What is left are facts of the code: the slots' commit strings, check 5's public rows
and the setup's evaluations of them (`hset` gives these at `resSetup`), and where the copy's columns sit. -/
theorem resTie_layR {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV) xO)).isSome = false)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {qv qo n : ℕ} (hn : n < 2)
    (hqv : c.ports.findIdx? (·.name == "v") = some qv) (hqo : c.ports.findIdx? (·.name == "out") = some qo)
    (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    (hslv : SlotCommits c "v" qv n w)
    (hslo : readForms w (hmCommit c qo n) = FlockSoundness.Binding.commitString
      (rowPfx c qo ++ readForms w (outRowForms c qo n)) (readForms w (hmSalt c qo n)))
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
  have hfo : rowForms c "out" qo n = outRowForms c qo n := by rw [rowForms, if_neg hoout]
  refine resTie_copy rfl hT hnc
    (portCalls_sub_layR elem hr hss hsv hc (by simp [resPorts]) hqv hn)
    (portCalls_sub_layR elem hr hss hsv hc (by simp [resPorts]) hqo hn)
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
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2)
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
  have hq : c.ports.findIdx? (·.name == "v") = some pt.S.ports.length :=
    (List.findIdx?_toArray _ c.ports.toList).trans hfind
  rw [hqv] at hq
  cases hq
  have hw := take_words hr'
  rw [hAl] at hw
  rw [hw, List.map_map]
  congr 1
  exact List.map_snd_zip (le_of_eq hlen.symm)

/-- **`v` is the port after the row ports, and at most `out` follows it**: at an algebra part's statement, `checkParsed`'s
row ports are V\*'s (`vstarRows`): the row ports, `v`, then V\*'s one output row. -/
theorem v_idx {j : ℕ} {nm : String} {pt : VStar.Part} {c : Circuit} {u : VStar.UnitC}
    (hu : (VStar.Stmt.inner j nm pt).unit = .ok u)
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      RecVStar.vstarRows (VStar.Stmt.inner j nm pt).sub u)
    {qv : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv) :
    qv = pt.S.ports.length ∧ c.ports.size ≤ pt.S.ports.length + 2 := by
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
  have hq : c.ports.findIdx? (·.name == "v") = some pt.S.ports.length :=
    (List.findIdx?_toArray _ c.ports.toList).trans hfind
  rw [hqv] at hq
  cases hq
  refine ⟨rfl, ?_⟩
  have hl := congrArg List.length hr'
  simp only [List.length_append, List.length_cons, List.length_map, List.length_zip, Array.length_toList, min_self,
    List.length_singleton, List.length_nil] at hl
  omega

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
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2)
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

/-! ## The `out` slot -/

/-- **Every output word is returned**: in an untyped circuit whose unit returns the outputs, `HmRow.check`'s `seen`
loop marks each output word its copies' leaf maps return (`check_returned`'s invariant), and the check refuses a word
left unmarked ("an output leaf no unit returns"). -/
theorem check_complete {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet = c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) {o : ℕ} (hoW : o < c.outWords) :
    o ∈ FlockSoundness.Discharge.HiddenExec.returned c (List.range c.upv) := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨u, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  replace h := FlockSoundness.ite_join h fun ha => (FlockSoundness.Discharge.Layout.ite_throw_cond ha).2
  obtain ⟨u1, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨u2, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨wired, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨wired2, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨u3, -, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨-, h⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  obtain ⟨seen, hseen, h⟩ := FlockSoundness.except_bind_ok h
  obtain ⟨hany, -⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond h
  have hI := FlockSoundness.Discharge.HiddenExec.forIn_range_inv
    (fun pre s => FlockSoundness.Discharge.HiddenExec.SeenInv c
      (FlockSoundness.Discharge.HiddenExec.returned c pre) s)
    (fun pre a s t hs hst => by
      obtain ⟨-, hst⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond hst
      obtain ⟨s', hin, hst⟩ := FlockSoundness.except_bind_ok hst
      cases hst
      refine ⟨_, rfl, ?_⟩
      have e0 : (default : Array ℕ) = #[] := rfl
      rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD', e0, ← Array.forIn_toList] at hin
      have := FlockSoundness.Discharge.HiddenExec.forIn_list_inv
        (fun pre' s => FlockSoundness.Discharge.HiddenExec.SeenInv c
          (FlockSoundness.Discharge.HiddenExec.returned c pre ++ pre') s) (fun pre' o s t hs hst => by
          obtain ⟨hc, hst⟩ := FlockSoundness.Discharge.Layout.ite_throw_cond hst
          cases hst
          refine ⟨_, rfl, ?_⟩
          rw [← List.append_assoc]
          exact FlockSoundness.Discharge.HiddenExec.seenInv_step hs hc) _ [] s s' (by simpa using hs) hin
      simpa [FlockSoundness.Discharge.HiddenExec.returned, List.flatMap_append] using this)
    (by
      refine ⟨Array.size_replicate, fun o h1 => ?_, List.nodup_nil, fun o h => by cases h⟩
      rw [FlockSoundness.Discharge.HiddenExec.bool_get, Array.getElem?_replicate, ho, bne_self_eq_false] at h1
      split at h1 <;> cases h1) hseen
  have hlt : o < seen.size := by rw [hI.1]; exact hoW
  refine hI.2.1 o ?_
  by_contra hf
  rw [getElem!_pos seen o hlt] at hf
  exact hany (Array.any_eq_true.2 ⟨o, hlt, by simpa using hf⟩)

/-- **An output word's column is in the block**: output word `o` is returned by some copy `u'`'s output `j`
(`check_complete`), at the unit slot `g · upv + u'`, column `16 j` of the returned-output group (`outCol_leaf`), which
lies in the slot wherever the group's bits do (`hin`). -/
theorem outCol_lt {st : Flock.Stmt} (L : FlockSoundness.Layout st) {tags : Flock.Tags} {ls : Lean.Json}
    (hty : st.c.typed = none) (ho : st.c.outNet = st.c.unitNet) (hls : HmRow.check st.c tags ls = .ok ())
    (hin : ∀ j < (st.c.unit.outGroups.getD 0 default).bits.size, ∀ t < 16,
      (st.c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * j + t < 2 ^ st.c.unit.unitLog)
    {g o t : ℕ} (hg : g < st.c.g) (hoW : o < st.c.outWords) (ht : t < 16) :
    HmOut.outCol st.c g o + t < 2 ^ st.c.kLog := by
  have C := FlockSoundness.check_facts hty hls
  obtain ⟨h16, -, hlo⟩ := FlockSoundness.Discharge.Layout.check_out hty ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD']; rfl
  have hupv : 0 < st.c.upv := FlockSoundness.Discharge.Hm.upv_pos C ho
  have hpv := FlockSoundness.perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hr := check_complete hty ho hls hoW
  unfold FlockSoundness.Discharge.HiddenExec.returned at hr
  obtain ⟨u', hu', hm⟩ := List.mem_flatMap.1 hr
  rw [List.mem_range] at hu'
  obtain ⟨j, hj, hjo⟩ := List.getElem_of_mem hm
  have hj' : j < (st.c.leavesOut.getD u' #[]).size := by simpa using hj
  have e : (st.c.leavesOut.getD u' #[])[j] = o := by simpa using hjo
  have hleaf := FlockSoundness.Discharge.HiddenExec.outCol_leaf hty ho
    (FlockSoundness.Discharge.HiddenExec.check_returned hty ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu' hj'
  rw [← e, hleaf]
  have hjs : j < (st.c.unit.outGroups.getD 0 default).bits.size := by
    rw [← (hlo u' hu').1, ← hget u' hu']; exact hj'
  have := L.unit_slot_lt (L.slot_lt hg hu') (hin j hjs t ht)
  omega

/-- **Output word `k`'s column, where the one copy returns the output words in order**: copy `0`'s output `k`
(`outCol_leaf`), at the unit slot `g · upv`, column `16 k` of the returned-output group. -/
theorem outCol_range {st : Flock.Stmt} (L : FlockSoundness.Layout st) {tags : Flock.Tags} {ls : Lean.Json}
    (hty : st.c.typed = none) (ho : st.c.outNet = st.c.unitNet) (hls : HmRow.check st.c tags ls = .ok ())
    (hupv : st.c.upv = 1) {N : ℕ} (hl : st.c.leavesOut.getD 0 #[] = Array.range N) {g : ℕ} (hg : g < st.c.g) {k : ℕ}
    (hk : k < N) :
    HmOut.outCol st.c g k =
      st.c.unitRange.slot (g * st.c.upv) + ((st.c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * k) := by
  have C := FlockSoundness.check_facts hty hls
  obtain ⟨h16, -, hlo⟩ := FlockSoundness.Discharge.Layout.check_out hty ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD']; rfl
  have hpv := FlockSoundness.perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hj : k < (st.c.leavesOut.getD 0 #[]).size := by rw [hl, Array.size_range]; exact hk
  have hleaf := FlockSoundness.Discharge.HiddenExec.outCol_leaf hty ho
    (FlockSoundness.Discharge.HiddenExec.check_returned hty ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g (by omega : 0 < st.c.upv) hj
  have e : (st.c.leavesOut.getD 0 #[])[k]'hj = k := by simp only [hl, Array.getElem_range]
  rw [e, Nat.add_zero] at hleaf
  exact hleaf

/-- **An output port's hm96 slot at instance `n` commits its output bits** (`outRowForms`), at a hidden-output table the
verifier sets up whose model the witness's one-table share satisfies: its out columns are the model's `b ‖ c` (`bc_bit`),
the commit string of the row's message and salt (`table_rows`); the message's bits are `HmOut.delta`'s copies of the
output bits (`Hidden.msg_copy`), at the columns `outRowForms` reads below the port's run of output words, which lie in the
block (`outCol_lt`), and the forced-zero row past them (`Hm.zero_bit`); the salt is at its salt columns (`salt_bit`). -/
theorem slotOut_table {tags : Flock.Tags} {cf pf : ByteArray} {coins : _} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt} (ht : tags.typed = false)
    (hh : tags.hiddenOutputs = true)
    (hd : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st)
    (hnt : st.c.tmpl = none) (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (ho : st.c.outNet = st.c.unitNet)
    (hin : ∀ j < (st.c.unit.outGroups.getD 0 default).bits.size, ∀ t < 16,
      (st.c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * j + t < 2 ^ st.c.unit.unitLog)
    (L : FlockSoundness.Layout st) (regs : List (FlockSoundness.Model.Region st.c.kLog (st.m - st.c.kLog)))
    {w : Claims.Flock.Wit} (hz : (L.model regs).Satisfies (Flock.Readable.zOf 1 w 0 st.m)) {q n : ℕ}
    (hqs : q < st.c.ports.size) (hq : ¬ q < HmOut.nIn st.c) (hb : n / st.c.g < 2 ^ st.nbl) :
    readForms w (hmCommit st.c q n) = FlockSoundness.Binding.commitString
      (rowPfx st.c q ++ readForms w (outRowForms st.c q n)) (readForms w (hmSalt st.c q n)) := by
  have hd' := hd
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd'
  rw [hh] at hd'
  simp only [↓reduceIte] at hd'
  have s := FlockSoundness.Discharge.Hidden.setupHidden_spec' hd'
  have hi : tags.isTmpl = false := by
    obtain ⟨pre, c₀, -, -, -, -, hfl | ⟨tu, -, -, htu⟩⟩ :=
      FlockSoundness.Discharge.Hidden.parseCommit_untyped ht s.parse
    · exact hfl.1
    · rw [hnt] at htu
      cases htu
  obtain ⟨hp, hcr⟩ := FlockSoundness.Discharge.Hidden.parse_untyped ht (s.parseF hi)
  have P := FlockSoundness.parse_facts hp
  have L0 : FlockSoundness.Layout (FlockSoundness.Discharge.Hidden.flatTwin st) :=
    FlockSoundness.layout_of (st := FlockSoundness.Discharge.Hidden.flatTwin st) P s.pin rfl rfl s.m s.kLog
  have F0 : FlockSoundness.Discharge.Aliased.UnitFacts tags (FlockSoundness.Discharge.Hidden.flatTwin st) :=
    FlockSoundness.Discharge.Hidden.unitFacts_of_parse (st := FlockSoundness.Discharge.Hidden.flatTwin st) hp rfl
  have hda : st.da = (Flock.HmRow.delta st.c st.pin).1 ++
      (FlockSoundness.Discharge.Hidden.outPairs st.c).toArray := by
    rw [s.da, FlockSoundness.Discharge.Hidden.hidden_delta_unpinned
      (FlockSoundness.Discharge.Hidden.unpinned_of_v1 hcr fun p hp => (hv1 p hp).1)]
  have hrows : FlockSoundness.Discharge.Hidden.OutRows st.c :=
    FlockSoundness.Discharge.Hidden.out_rows_v1 hp hcr fun p hp => (hv1 p hp).1
  have hz0 := FlockSoundness.Discharge.Hidden.sat_flat L L0 F0 hv1 hda hrows hz
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  have hupv : 0 < st.c.upv := FlockSoundness.Discharge.Hm.upv_pos C ho
  obtain ⟨R, hg0, -, hbc, -⟩ := table_rows ht hd hupv hv1 L regs hz
  have hgn : n % st.c.g < st.c.g := Nat.mod_lt _ hg0
  have hq! : st.c.ports[q]! = st.c.ports[q] := getElem!_pos st.c.ports q hqs
  replace hb : n / st.c.g < 2 ^ (L.model regs).nbl := by
    show n / st.c.g < 2 ^ (st.m - st.c.kLog)
    rw [L.m, Nat.add_sub_cancel_left]
    exact hb
  let o : Fin (2 ^ (L.model regs).nbl) := ⟨n / st.c.g, hb⟩
  have hzero : FlockSoundness.Discharge.Hm.blockBit (L.model regs) (Flock.Readable.zOf 1 w 0 st.m) o
      (FlockSoundness.Discharge.Hm.zeroCol st.c) = false :=
    FlockSoundness.Discharge.Hm.zero_bit L0 F0 hupv hg0 hz0 o
  have hnI : HmOut.nIn st.c = st.c.ports.size - st.c.outRowPorts.size := rfl
  have hrs := FlockSoundness.Discharge.Hidden.OutRows.size hrows
  have hj : q - HmOut.nIn st.c < st.c.outPorts.size := by omega
  have hqj : HmOut.nIn st.c + (q - HmOut.nIn st.c) = q := by omega
  have hG := FlockSoundness.Discharge.Hidden.outGroup_single hrows.2.1 hj
  have hwq : (st.c.ports.getD q default).words = st.c.ports[q].words := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hqs]; rfl
  have hoW : ∀ k < (st.c.outPorts.getD (q - HmOut.nIn st.c) default).2,
      FlockSoundness.Discharge.Hidden.outAt st.c (q - HmOut.nIn st.c) + k < st.c.outWords := by
    intro k hk
    have h1 := FlockSoundness.Discharge.Hidden.outAt_succ st.c hj
    have h2 : FlockSoundness.Discharge.Hidden.outAt st.c (q - HmOut.nIn st.c + 1) ≤
        FlockSoundness.Discharge.Hidden.outAt st.c st.c.outPorts.size := by
      unfold FlockSoundness.Discharge.Hidden.outAt
      exact words_take_mono (·.2) st.c.outPorts.toList (a := q - HmOut.nIn st.c + 1) (b := st.c.outPorts.size)
        (by omega)
    rw [FlockSoundness.Discharge.HiddenExec.outAt_size] at h2
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD'] at h1
    omega
  have hrow : readForms w (outRowForms st.c q n) =
      FlockSoundness.Discharge.Hm.bytesOf (FlockSoundness.Discharge.Hm.blockBit (L.model regs)
        (Flock.Readable.zOf 1 w 0 st.m) o) (2 * st.c.ports[q]!.words)
        (FlockSoundness.Typed.msgCol st.c (n % st.c.g) q) := by
    refine readForms_eq_of_bits (by
      simp only [outRowForms, List.length_map, List.length_range, FlockSoundness.Discharge.Hm.length_bytesOf, hwq, hq!]
      omega) fun i hi => ?_
    have hi' : i < 16 * st.c.ports[q].words := by
      simpa [outRowForms, Array.getElem?_eq_getElem hqs] using hi
    have hk : i / 16 < (st.c.ports.getD (HmOut.nIn st.c + (q - HmOut.nIn st.c)) default).words := by
      rw [hqj, hwq]; omega
    have hm := FlockSoundness.Discharge.Hidden.msg_copy L L0 F0 hv1 hda hrows hz o hgn hj hk
      (Nat.mod_lt i (by omega : 16 > 0))
    rw [hqj, show 2 * (i / 16) + i % 16 / 8 = i / 8 by omega, show i % 16 % 8 = i % 8 by omega] at hm
    rw [FlockSoundness.Discharge.Hm.bitL_bytesOf _ _ (by rw [hq!]; omega),
      FlockSoundness.Discharge.Hidden.msgCol_delta, hm]
    simp only [outRowForms, List.getElem_map, List.getElem_range]
    unfold FlockSoundness.Discharge.Hidden.outSrc
    by_cases hk2 : i / 16 < (HmOut.outGroup st.c (q - HmOut.nIn st.c)).2.2.2
    · rw [if_pos hk2, if_pos hk2, formBit_var]
      have hy : HmOut.outCol st.c (n % st.c.g) ((HmOut.outGroup st.c (q - HmOut.nIn st.c)).2.2.1 + i / 16) +
          i % 16 < 2 ^ st.c.kLog := by
        rw [hG] at hk2 ⊢
        exact outCol_lt L P.typed ho hls hin hgn (hoW _ hk2) (Nat.mod_lt i (by omega))
      exact (blockBit_zOf_one (L.model regs) w o hy).symm
    · rw [if_neg hk2, if_neg hk2, hzero]
      rfl
  have hpfx : rowPfx st.c q = (Flock.HmRow.rowPrefix st.c.ports[q]!.words).data.toList := by
    unfold rowPfx
    rw [Array.getElem?_eq_getElem hqs, hq!]
    rfl
  have hsalt : readForms w (hmSalt st.c q n) =
      FlockSoundness.Discharge.Hm.bytesOf (FlockSoundness.Discharge.Hm.blockBit (L.model regs)
        (Flock.Readable.zOf 1 w 0 st.m) o) 192 (FlockSoundness.Discharge.Hm.saltCol st.c (n % st.c.g) q) := by
    refine readForms_eq_of_bits (by simp [hmSalt, FlockSoundness.Discharge.Hm.length_bytesOf]) fun j hj => ?_
    have hj' : j < 1536 := by simpa [hmSalt] using hj
    have hy : hmBase st.c q n + (hmIn st.c 2 + j) < 2 ^ st.c.kLog := by
      have := FlockSoundness.Discharge.Hm.saltCol_lt R hgn hqs (j := j / 8) (t := j % 8) (by omega) (by omega)
      unfold FlockSoundness.Discharge.Hm.saltCol at this
      rw [hmIn_eq] at this
      unfold hmBase
      omega
    simp only [hmSalt, List.getElem_map, List.getElem_range, formBit_var]
    exact (salt_bit (L.model regs) w st.c rfl hb hy hj').symm
  refine readForms_eq_of_bits (by simp [hmCommit, FlockSoundness.Discharge.Hm.length_commitString]) fun t ht => ?_
  have ht' : t < 1024 := by simpa [hmCommit] using ht
  simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var]
  have hy : hmBase st.c q n + (hmOutCol st.c + t) < 2 ^ st.c.kLog := by
    have := FlockSoundness.Discharge.Hm.bcCol_lt R hgn hqs (j := t / 8) (t := t % 8) (by omega) (by omega)
    unfold FlockSoundness.Discharge.Hm.bcCol at this
    rw [hmOut_eq] at this
    unfold hmBase
    omega
  rw [hpfx, hrow, hsalt, ← hbc o hgn hqs]
  exact (bc_bit (L.model regs) w st.c rfl hb hy ht').symm

/-- A hidden-output table's output rows are one per output port (`out_rows_v1`), from the checks its setup ran. -/
theorem outRows_setupOne {tags : Flock.Tags} {cf pf : ByteArray} {coins : _} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt} (ht : tags.typed = false)
    (hh : tags.hiddenOutputs = true)
    (hd : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) :
    FlockSoundness.Discharge.Hidden.OutRows st.c := by
  have hd' := hd
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd'
  rw [hh] at hd'
  simp only [↓reduceIte] at hd'
  have s := FlockSoundness.Discharge.Hidden.setupHidden_spec' hd'
  exact (FlockSoundness.Discharge.Hidden.committed_of_commit s.parse).outRows ht fun p hp => (hv1 p hp).1

/-- `FlockDeploy.Raw.ins` reads its tags from `--statement`'s name, and C's under the session's coins. -/
theorem raw_ins_tags {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    Tags.find? w.statement = some s.stmtTags ∧ s.read.I.tags = s.stmtTags.withCoins s.read.I.coins := by
  unfold FlockDeploy.Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨by assumption, rfl⟩)

/-- A recursive statement's tags are a known statement's, and C's are them under the session's coins. -/
theorem insO_tags {x : Msg} {i : RecIns} (h : insO x = some i) :
    (∃ nm, Tags.find? nm = some i.base.stmtTags) ∧ ∃ coins, i.base.read.I.tags = i.base.stmtTags.withCoins coins := by
  cases hd : decodeRec x with
  | none => unfold insO at h; rw [hd] at h; simp at h
  | some w =>
    unfold insO at h
    rw [hd] at h
    simp only [Option.bind_eq_bind, Option.bind_some] at h
    split_ifs at h
    · simp at h
    have h' : w.ins.toOption = some i := by simpa using h
    obtain ⟨hf, hc⟩ := raw_ins_tags (RecRaw.ins_base (toOption_eq_some h'))
    exact ⟨⟨_, hf⟩, _, hc⟩

/-- With an inner shape, `RecVStar.checkStmt` refuses a statement other than `verity/flock-circuit`. -/
theorem checkStmt_name {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {part : Option ℕ} {bindings : Option VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program registered part bindings rows =
      .ok ()) :
    tags.name = Flock.Tags.circuit.name := by
  unfold Flock.RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · rename_i hn
    simpa using hn

/-- **A staged part's tables have hidden outputs**: its setup's `checkStmt` holds it to `verity/flock-circuit`
(`checkStmt_name`), the one statement of that name (`Tags.find?`), whose outputs are hidden, under any coins and in
`verify --zk`'s tags (`zkTags_hiddenOutputs`). -/
theorem staged_hidden {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) {t : ℕ} (ht : t < p.sts.size) :
    p.tags.hiddenOutputs = true := by
  obtain ⟨ms, i, spec, sts, rfl, hsu, -⟩ := hS
  have e : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  subst e
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r, -, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hn := checkStmt_name (checks_stmt hC t ht)
  obtain ⟨⟨nm, hf⟩, coins, hI⟩ := insO_tags hins
  have hf' := hf
  unfold Flock.Tags.find? at hf'
  have hnm : nm = Flock.Tags.circuit.name := by
    have h2 : i.base.stmtTags.name = nm := by simpa using List.find?_some hf'
    rw [← h2, hn]
  subst hnm
  rw [FlockSoundness.Discharge.Recursion.find?_circuit] at hf
  have hT : i.base.stmtTags = Flock.Tags.circuit := (Option.some.inj hf).symm
  have hI' : i.base.read.I.tags.hiddenOutputs = true := by
    rw [hI, hT]
    unfold Flock.Tags.withCoins
    split <;> rfl
  show i.base.read.tags.hiddenOutputs = true
  unfold Flock.Readable.Read.tags
  split
  · rw [FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs]
    exact hI'
  · exact hI'

/-- **A staged `rec-residuals` part's `out` slot commits its output bits** (`resTie_staged`'s `hslo`): the part's table
has hidden outputs (`staged_hidden`) and returns them from its unit (`scopeOk`), the verifier's `unitLaid`
(`check_part_at`), whose output bits lie in its computed columns (`outCols_of_unitLaid`), so `slotOut_table`. -/
theorem slotOut_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss) {j : ℕ} {nm' : String} {pt : VStar.Part}
    (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    {m k cl jj : ℕ} {sh : Flock.Shape} {holo : Bool}
    (hres : RecResiduals.params c.unit.name = some (m, k, cl, jj)) (hshp : RecResiduals.shape? m k cl = some sh)
    (hholo : RecResiduals.holoOf c = .ok holo)
    {qo n : ℕ} (hqo : c.ports.findIdx? (·.name == "out") = some qo) (hoout : ¬ qo < HmOut.nIn c) (hn2 : n < 2) :
    readForms w (hmCommit c qo n) = FlockSoundness.Binding.commitString
      (rowPfx c qo ++ readForms w (outRowForms c qo n)) (readForms w (hmSalt c qo n)) := by
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
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2)
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  have hh : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).hiddenOutputs = true := by
    unfold FlockSoundness.Discharge.Exec.tagsAt
    split <;> exact staged_hidden hS hx h0
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0
  have hs := (FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports hs
  have ho : p.sts[0].c.outNet = p.sts[0].c.unitNet := by
    unfold FlockSoundness.Discharge.Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨-, -, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L regs hz1
  have hb := div_lt_blocks_two hn2 hnbl
  subst hc0
  obtain ⟨S', ns', nv', -, hn⟩ := FlockVBridge.OfVStar.check_part_at P.recResiduals hres hshp hholo
  have hnt := FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2
  have hU := L.built_flat hnt
  have hin : ∀ j < (p.sts[0].c.unit.outGroups.getD 0 default).bits.size, ∀ t < 16,
      (p.sts[0].c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * j + t < 2 ^ p.sts[0].c.unit.unitLog := by
    obtain ⟨hcol, hsz', -, hlt⟩ := FlockVBridge.OfVStar.outCols_of_unitLaid hU hn
    have hcp : (FlockSoundness.Rows.ofNet hU).nIn + (FlockSoundness.Rows.ofNet hU).nComp =
        p.sts[0].c.unit.constPos := FlockSoundness.Rows.ofNet_one hU
    have hfit := (FlockSoundness.built_fit hU).1
    intro j hj t ht
    rw [hsz'] at hj
    have := hlt (16 * j + t) (by omega)
    omega
  exact slotOut_table ht' hh hd hnt hv1 ho hin L regs hz1 (Array.findIdx?_eq_some_iff_getElem.1 hqo).1 hoout hb

/-- **Where a copy's `out` columns sit** (`resTie_copyOf`'s `hOc`): at a staged algebra part, `out` is the unit's one
output row, right after `v` (`v_idx`, one output row per output port by `outRows_setupOne`), so its bit `q` is output
word `q / 16`'s bit `q % 16`. The unit returns its output words in order (`checkParsed_leavesOut`), so output word `k`
is copy `(0, o, g, 0)`'s column `16 k` past the output group (`outCol_range`), which starts at VBridge's `partBase`
(`outCols_of_unitLaid`). -/
theorem oCols_staged {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {qv qo : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv)
    (hqo : c.ports.findIdx? (·.name == "out") = some qo) (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    {pto : Port} (hpto : c.ports[qo]? = some pto) (hwo : 128 ≤ pto.words)
    {m k cl jj : ℕ} {sh : Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hres : RecResiduals.params c.unit.name = some (m, k, cl, jj)) (hshp : RecResiduals.shape? m k cl = some sh)
    (hholo : RecResiduals.holoOf c = .ok holo) (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv))
    {o g n : ℕ} (hg : g < c.g) (hn : o * c.g + g = n) :
    ∀ q < 128 * S.res.length, ∃ hl : q < (rowForms c "out" qo n).length,
      Flock.Readable.copyOf p 0 ht w o g 0 (FlockVBridge.OfVStar.partBase S ns nv + q) =
        formBit w (rowForms c "out" qo n)[q] := by
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
  obtain ⟨L, -, -⟩ := Flock.Readable.layout_of_check hck hx ht' hpi h0
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht' hpi h0
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2)
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨spec0, hset⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have htd : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht'
  have hh : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).hiddenOutputs = true := by
    unfold FlockSoundness.Discharge.Exec.tagsAt
    split <;> exact staged_hidden hS hx h0
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0
  have hs := (FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports hs
  have ho : p.sts[0].c.outNet = p.sts[0].c.unitNet := by
    unfold FlockSoundness.Discharge.Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  have hrw := outRows_setupOne htd hh hd hv1
  obtain ⟨u, hu, hupv, -, -, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨u', hu', hlo⟩ := FlockSoundness.Discharge.Recursion.checkParsed_leavesOut (htab 0 h0).2.2
  rw [hu] at hu'
  cases hu'
  subst hc0
  obtain ⟨S', ns', nv', hpart', hnet⟩ := FlockVBridge.OfVStar.check_part_at P.recResiduals hres hshp hholo
  rw [hpart] at hpart'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hpart'
  obtain ⟨rfl, rfl, rfl⟩ := hpart'
  obtain ⟨hcol, hbits, -, hlt⟩ := FlockVBridge.OfVStar.outCols_of_unitLaid (L.built_flat P.tmpl) hnet
  have hcp : (FlockSoundness.Rows.ofNet (L.built_flat P.tmpl)).nIn + (FlockSoundness.Rows.ofNet (L.built_flat P.tmpl)).nComp =
      p.sts[0].c.unit.constPos := FlockSoundness.Rows.ofNet_one (L.built_flat P.tmpl)
  have hsm := FlockVBridge.OfVStar.part_small hpart
  have hR : RecResiduals.RES_OUT_WORDS = 128 := rfl
  obtain ⟨hqv', hsize⟩ := v_idx hu hrows hqv
  have hqs : qo < p.sts[0].c.ports.size := (Array.findIdx?_eq_some_iff_getElem.1 hqo).1
  have hqn : qo = HmOut.nIn p.sts[0].c := by omega
  have hnI : HmOut.nIn p.sts[0].c = p.sts[0].c.ports.size - p.sts[0].c.outRowPorts.size := rfl
  have hrs := FlockSoundness.Discharge.Hidden.OutRows.size hrw
  have hop : p.sts[0].c.outPorts.size = 1 := by omega
  have hG := FlockSoundness.Discharge.Hidden.outGroup_single hrw.2.1 (j := 0) (by omega)
  have h0at : FlockSoundness.Discharge.Hidden.outAt p.sts[0].c 0 = 0 := by
    simp [FlockSoundness.Discharge.Hidden.outAt]
  have hW : (p.sts[0].c.outPorts.getD 0 default).2 = p.sts[0].c.outWords := by
    have h1 := FlockSoundness.Discharge.Hidden.outAt_succ p.sts[0].c (j := 0) (by omega)
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD', h0at] at h1
    simp only [Nat.zero_add] at h1
    have h2 := FlockSoundness.Discharge.HiddenExec.outAt_size p.sts[0].c
    rw [hop] at h2
    exact h1.symm.trans h2
  have hg0 : HmOut.outGroup p.sts[0].c (qo - HmOut.nIn p.sts[0].c) = (0, 1, 0, p.sts[0].c.outWords) := by
    rw [← hqn, Nat.sub_self, hG, h0at, hW]
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlo'⟩ := FlockSoundness.Discharge.Layout.check_out P.typed ho hls
  have hget0 : p.sts[0].c.leavesOut.getD 0 #[] = p.sts[0].c.leavesOut[0]! := by
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD']; rfl
  have hN : u.outWords = RecResiduals.RES_OUT_WORDS := by
    have e := (hlo' 0 (by omega)).1
    rw [← hget0, hlo, Array.size_range, hbits] at e
    exact e
  have hdiv : n / p.sts[0].c.g = o := by
    subst hn
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hg, Nat.zero_add]
  have hmod : n % p.sts[0].c.g = g := by
    subst hn
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hg]
  intro q hq
  have hq16 : q / 16 < u.outWords := by omega
  have hkW : q / 16 < p.sts[0].c.outWords := by
    have hmem : q / 16 ∈ (p.sts[0].c.leavesOut[0]!).toList := by
      rw [← hget0, hlo]
      simpa using hq16
    exact (hlo' 0 (by omega)).2 _ hmem
  have hlen : (rowForms p.sts[0].c "out" qo n).length = 16 * pto.words := by
    simp [rowForms, outRowForms, hoout, Array.getD_eq_getD_getElem?, hpto]
  have hql : q < (rowForms p.sts[0].c "out" qo n).length := by rw [hlen]; omega
  refine ⟨hql, ?_⟩
  have hel : (rowForms p.sts[0].c "out" qo n)[q] = HmNets.Lin.var (n / p.sts[0].c.g * 2 ^ p.sts[0].c.kLog +
      (HmOut.outCol p.sts[0].c (n % p.sts[0].c.g) (q / 16) + q % 16)) := by
    simp [rowForms, outRowForms, hoout, hg0, hkW]
  have hb := hlt q (by omega)
  have hq' : ∀ A, A + (FlockVBridge.OfVStar.partBase S ns nv + 16 * (q / 16)) + q % 16 =
      A + (FlockVBridge.OfVStar.partBase S ns nv + q) := fun A => by omega
  rw [hel, formBit_var, outCol_range L P.typed ho hls hupv hlo (Nat.mod_lt _ (by omega)) hq16, hcol, hq',
    Claims.Recursion.VStar.copyOf_eq (by omega), hdiv, hmod]
  simp [Claims.Recursion.VStar.copyIdx, hsz, hupv]

/-- **Check 5's tie at one instance of a staged `rec-residuals` part**: `resTie_layR` with check 5's public rows from
the statement's setup. The setup ran `RecCoins.check` at the part's table, so `checkResiduals` at its inner session
(`residualsCheck_of_check`, `residualsBoth_of_check`), whose `v` and `out` rows the slots read (`check5_bytes`), and
the deployed `Brings.setup` lists their evaluations (`hset`). What is left are facts of the code: the table's labels
(`hidx`), the unit's name being no other V\* unit's, the `out` slot (`slotOut_staged`; `v`'s is `slotV_staged`), the
ports' word counts and where the copy's columns sit. -/
theorem resTie_staged {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV) xO)).isSome = false)
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
      (rowPfx c qo ++ readForms w (outRowForms c qo n)) (readForms w (hmSalt c qo n))) :
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
    fun u hu => hset u (by
      unfold resSetup; rw [hsu]; exact List.mem_flatMap.2 ⟨sts[0], by simp, List.mem_append_left _ hu⟩)
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
  exact resTie_layR hT hnc hr hss hsv hpc hn2 hqv hqo hvin hoout
    (slotV_staged hS hck hr hss hsv hpc hqv hn2) hslo hptv hpto hwv hwo hzr hpv.symm hpo.symm
    (hmemS _ (List.mem_append_left _ (hci hnr).1)) (hmemS _ (List.mem_append_left _ (hci hnr).2))
    (hmemS _ (List.mem_append_right _ (hci hzn).1)) (hmemS _ (List.mem_append_right _ (hci hzn).2))
    hsz hVc hOc

/-! ## The share at the pieces' copies -/

/-- **`resPieces_computes` with the tie at the pieces' copies only**, and the whole acceptance at hand: wherever part
`s` accepts at an algebra part, each piece `ownQ` gives it holds where, at every staged statement and witness that pass
`Flock.check`, agree with the messages and record true, collision-free calls, the piece sits at one of the statement's
copies, of its group, at which check 5's tie holds (`ResTie`) and whose residuals vanish at what `elemAt` reads wherever
they vanish at the copy's rows (`ResZero`; the copy's rows being `elemAt`'s gives it, and so does their being
`elemAt`'s up to an offset the residuals cancel, `PartS.resZero_off`). The piece's key is
`(jj, o, g, u)`: the part's group `jj` (so pieces of different groups have different keys) and instance `o + g`'s unit
`u` of the part's one table (`keyCopy`: block `(o + g) / c.g`, VU `(o + g) % c.g` at the part's own `c.g`, which the
prover chooses), so the pieces needn't know `c.g`. Check 5 holds only
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
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (hpc : ∀ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p →
      Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w = true →
      showsPart untr b s x xO c → L.Agrees cmt x s w xO ms → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false →
      ∀ q ∈ pieces (Claims.Recursion.xI D x), ownQ x q = some s →
        ∃ ht : 0 < p.sts.size, 0 < p.sts[0].c.g ∧
          (q.key.2.1 + q.key.2.2.1) / p.sts[0].c.g < 2 ^ (p.sts[0].m - p.sts[0].c.kLog) ∧
          q.key.2.2.2 < p.sts[0].c.upv ∧
          ∀ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
            (nv : ℕ), _root_.Flock.RecResiduals.params p.sts[0].c.unit.name = some (m, k, cl, jj) →
            _root_.Flock.RecResiduals.shape? m k cl = some sh →
            _root_.Flock.RecResiduals.holoOf p.sts[0].c = .ok holo →
            _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) →
            S = q.S ∧ q.key.1 = jj ∧
            ResTie S ns nv
              (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)
              (keyCopy p ht w q.key) ∧
            ∀ os last, Bd.final (ms.getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
              os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
              ResZero S (_root_.Flock.Readable.rowsRead ns nv (keyCopy p ht w q.key))
                  (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key) →
                ResZero S (elemAt (Claims.Recursion.xI D x) (Bd.opened os last)
                    ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)
                  (vvAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q.key)) :
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
  obtain ⟨ht', hG, ho', hu, hch⟩ := hpc xO w p hx hck' hs ⟨hh, hp, hm, hrw⟩ hT hc q hq ho
  obtain ⟨m, k, cl, jj, sh, holo, S, ns, nv, h1, h2, h3, h4, hro⟩ :=
    _root_.Flock.Readable.innerCopies ht' ho' (Nat.mod_lt _ hG) hu hcp
  obtain ⟨rfl, -, htie, hE⟩ := hch m k cl jj sh holo S ns nv h1 h2 h3 h4
  exact hE os last hf hl (resVal_zero_of_tie hro htie)

/-! ## The statements -/

/-- **The coin tie**: at a V\* statement `xO` part `s` stages at `x` and shows the coins `c` at, wherever check 5 loads
a `rec-residuals` statement's inner session and computes its `v` rows (`residualRows`), the verifier's values the coins give
at a key (`vvAt` at the inner statement and the coins, keyed `(jj, o, g, 0)` by the part's group) are the elements
of instance `o + g`'s row, at the verifier's part `(S, ns, nv)`. `staged_rep_drawn` is its coin side. -/
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
          ∀ o g, o + g < 2 →
            vvAt (xIOf x) ((List.range (Bk (xIOf x))).map c) (jj, o, g, 0) = wordsElems nv rows[o + g]!

/-- **Check 5's tie at a piece's copy** (the `ResTie` conjunct `resPieces_computesAt` asks of each piece): at key
`(jj, o, g, 0)` of an instance `n = o + g` check 5 holds (`n < 2`), between the verifier's values the coins give at the
key and its copy's columns (`keyCopy`: block `n / c.g`, VU `n % c.g`, at any `c.g ≥ 1`). `resTie_staged` at `n`, with
the copy's `v` columns where its slot reads (`vCols_of_base`, VBridge's
`vBase` being the port's column by `vBase_staged`), the `out` slot committing its output bits (`slotOut_staged`), its
residual columns where the `out` slot reads (`oCols_staged`, from the output leaves `checkParsed` holds to V\*'s), and
the coins' values check 5's row (`VVTie`). -/
theorem resTie_copyOf {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} {cs : ℕ → Coin} {untr : TrReader}
    {vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128} {Bk : Msg → ℕ}
    (hT : ∀ e ∈ (DeployedR.layR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV) xO)).isSome = false)
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
    {o g n : ℕ} (hG : 0 < sts[0].c.g) (hn : o + g = n) (hn2 : n < 2) (hnp : n < sts[0].pub.n)
    {qv qo : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv)
    (hqo : c.ports.findIdx? (·.name == "out") = some qo) (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    {ptv pto : Port} (hptv : c.ports[qv]? = some ptv) (hpto : c.ports[qo]? = some pto)
    (hrows : ∀ (inn : RecCoins.Inner) (rows : Array (Array ℕ)), f.load sts[0].c.unit.name = .ok inn →
      FlockSoundness.Discharge.Recursion.residualRows inn jj = some rows →
      ptv.words = (RecCoins.padTo rows[n]! (Registered.rowWords rows[n]!.size 16)).size ∧ n < rows.size ∧
        8 * nv ≤ rows[n]!.size)
    (hwo : pto.words = (RecCoins.padTo (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!
      (Registered.rowWords (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!.size 16)).size) :
    ResTie S ns nv (vvAt (xIOf x) ((List.range (Bk (xIOf x))).map cs) (jj, o, g, 0)) (keyCopy p ht w (jj, o, g, 0)) := by
  obtain ⟨p', -, -, hx', ht', hpi, -, -, -, -⟩ := stagedTables hsh.2
  rw [hx] at hx'
  cases hx'
  have hpc0 : p.sts[0].c = c := partCircuit_staged hsh.2 hx ht' hpi hr hpc ht
  have hc0 : sts[0].c = c := by
    have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
    subst hp
    exact hpc0
  have hg : n % c.g < c.g := Nat.mod_lt _ (hc0 ▸ hG)
  have hk : keyCopy p ht w (jj, o, g, 0) = Flock.Readable.copyOf p 0 ht w (n / c.g) (n % c.g) 0 := by
    simp only [keyCopy]
    rw [hpc0, hn]
  rw [hk]
  have hslo := slotOut_staged (w := w) hsh.2 hck hr hss hsv hpc (by rw [← hc0]; exact hres) hshp
    (by rw [← hc0]; exact hholo) hqo hoout hn2
  obtain ⟨inn, rows, hinn, hrw, hall⟩ := resTie_staged hT hnc hsh.2 hck hr hss hsv hpc hsu h0 hidx hown hres hso hop
    hsp ho hf hset hn2 hnp hqv hqo hvin hoout hslo
  obtain ⟨hwv, hnr, hsz⟩ := hrows inn rows hinn hrw
  have hvb := vBase_staged hsh.2 hr hss hsv hpc hqv (by rw [← hc0]; exact hres) hshp (by rw [← hc0]; exact hholo)
    hpart
  have hVc := vCols_of_base (w := w) hsh.2 hr hss hsv hpc hx ht hqv hvin hptv
    (by rw [hwv]; exact le_trans hsz (size_padTo_ge _ _)) hg (Nat.div_add_mod' n c.g) hvb
  have hzs : 128 ≤ ((RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!).size := by
    rw [getElem!_pos (RecCoins.zeroRows 2 RecCoins.OUT_WORDS) n (by simpa [RecCoins.zeroRows] using hn2)]
    simp [RecCoins.zeroRows, RecCoins.OUT_WORDS]
  have hOc := oCols_staged (w := w) hsh.2 hck hr hss hsv hpc hx ht hqv hqo hvin hoout hpto
    (by rw [hwo]; exact le_trans hzs (size_padTo_ge _ _)) (by rw [← hc0]; exact hres) hshp
    (by rw [← hc0]; exact hholo) hpart hg (Nat.div_add_mod' n c.g)
  rw [hvv xO hsh i spec sts hsu 0 h0 m k cl jj hres f inn rows hf hinn hrw sh holo S ns nv hshp hholo hpart o g
    (hn ▸ hn2), hn]
  exact hall hptv hpto hwv hwo hnr hsz hVc hOc

/-- **`resTie_copyOf` at check 5's site** (`CopySite.copySite_staged`): the setup's registered ports, its source at C3
and the four "not another V\* unit" facts come from the staged setup's own checks at a `rec-residuals` table. -/
theorem resTie_site {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} {cs : ℕ → Coin} {untr : TrReader}
    {vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128} {Bk : Msg → ℕ}
    (hT : ∀ e ∈ (DeployedR.layR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV) xO)).isSome = false)
    (hsh : showsPart untr b s x xO cs)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {m k cl jj : ℕ} (hres : RecResiduals.params sts[0].c.unit.name = some (m, k, cl, jj))
    (hset : ∀ u ∈ resSetup xO, u ∈ b.setup xO) (hvv : VVTie untr b vvAt Bk s x cs)
    {sh : _root_.Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hshp : RecResiduals.shape? m k cl = some sh) (hholo : RecResiduals.holoOf sts[0].c = .ok holo)
    (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv))
    {o g n : ℕ} (hG : 0 < sts[0].c.g) (hn : o + g = n) (hn2 : n < 2) (hnp : n < sts[0].pub.n)
    {qv qo : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv)
    (hqo : c.ports.findIdx? (·.name == "out") = some qo) (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    {ptv pto : Port} (hptv : c.ports[qv]? = some ptv) (hpto : c.ports[qo]? = some pto)
    (hrows : ∀ (f : RecCoins.Source), RecCoins.Source.atC3 i.src i.inner i.commits = some f →
      ∀ (inn : RecCoins.Inner) (rows : Array (Array ℕ)), f.load sts[0].c.unit.name = .ok inn →
      FlockSoundness.Discharge.Recursion.residualRows inn jj = some rows →
      ptv.words = (RecCoins.padTo rows[n]! (Registered.rowWords rows[n]!.size 16)).size ∧ n < rows.size ∧
        8 * nv ≤ rows[n]!.size)
    (hwo : pto.words = (RecCoins.padTo (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!
      (Registered.rowWords (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!.size 16)).size) :
    ResTie S ns nv (vvAt (xIOf x) ((List.range (Bk (xIOf x))).map cs) (jj, o, g, 0)) (keyCopy p ht w (jj, o, g, 0)) := by
  have hrs : (RecResiduals.params sts[0].c.unit.name).isSome := by simp [hres]
  obtain ⟨own, f, hown, hf, hso, hop, hsp, hro⟩ := RecDeploy.CopySite.copySite_staged hsu h0 (.inr hrs)
  exact resTie_copyOf hT hnc hsh hck hr hss hsv hpc hx ht hsu h0 hidx hown hres hso hop hsp (hro hrs) hf hset hvv
    hshp hholo hpart hG hn hn2 hnp hqv hqo hvin hoout hptv hpto (hrows f hf) hwo

/-- **The `rec-residuals` pieces' share of `hlowP` at the layout of record** (U12's statement): at the compiled record
`recBound` over `DeployedR.layR elem` and `IVStar` of `bdR rd wE final w`, the staging `showsPart`, wherever part `s`
accepts and its V\* statement at `x`'s inner shape is an algebra part (`.inner`), each `rec-residuals` piece `ownQ`
gives it holds (`resHolds elemAt vvAt`) at the messages the last message opens and the coins (`PiecesComputes` at
`bdR rd wE final w`). The deployed constants are `rd wE final w := rdR wER finalR wR`. -/
def ResLowered (rd : Msg → Option FlockDeploy.FinalP) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128) : Prop :=
  ∀ s x c ms, PartAccepts FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) (_root_.RecDeploy.bdR rd wE final w).IVStar)
      (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR elem) shape (showsPart untr b) cmt s x c ms →
    ∀ (r : XR), readX x = some r → ∀ (ss : Array VStar.Stmt), RecVStar.statements r.shape = .ok ss →
      ∀ (j : ℕ) (nm : String) (pt : VStar.Part), ss[s]? = some (.inner j nm pt) →
        PiecesComputes (_root_.RecDeploy.bdR rd wE final w)
          (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) (_root_.RecDeploy.bdR rd wE final w).IVStar)
          pieces (resHolds elemAt vvAt) ownQ s x c ms

/-- **`ResLowered` at every part**: where `ownQ` gives a piece only to a part whose V\* statement is an algebra part
(`hin`), the share holds wherever any part accepts. At a `Joint` acceptance (every part's `PartAccepts` at the parts'
own messages) this is `DecompBR.ResLowJ`'s conclusion, part by part. -/
theorem resLowered_parts {rd : Msg → Option FlockDeploy.FinalP} {wE : Msg → ℝ}
    {final : Msg → Option (List (List (Msg × Salt)) × Msg)} {w : Msg → ℝ} {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg}
    {pieces : Msg → List ResPiece} {ownQ : Msg → ResPiece → Option ℕ}
    {elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128}
    {vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128}
    (h : ResLowered rd wE final w b elem shape untr cmt pieces ownQ elemAt vvAt)
    (hin : ∀ x q s, q ∈ pieces (Claims.Recursion.xI
        (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) (_root_.RecDeploy.bdR rd wE final w).IVStar)
        x) → ownQ x q = some s →
      ∃ (r : XR) (ss : Array VStar.Stmt) (j : ℕ) (nm : String) (pt : VStar.Part), readX x = some r ∧
        RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some (.inner j nm pt))
    {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (ha : PartAccepts FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) (_root_.RecDeploy.bdR rd wE final w).IVStar)
      (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR elem) shape (showsPart untr b) cmt s x c ms) :
    PiecesComputes (_root_.RecDeploy.bdR rd wE final w)
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) (_root_.RecDeploy.bdR rd wE final w).IVStar)
      pieces (resHolds elemAt vvAt) ownQ s x c ms := by
  intro q hq ho
  obtain ⟨r, ss, j, nm, pt, hr, hss, hsv⟩ := hin x q s hq ho
  exact h s x c ms ha r hr ss hss j nm pt hsv q hq ho

end

end RecDeploy.ResLower
