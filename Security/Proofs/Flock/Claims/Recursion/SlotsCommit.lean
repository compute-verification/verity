import Proofs.Flock.Claims.Recursion.RecOpen
import Proofs.Flock.Soundness.Discharge.Hm.Commit

/-!
# `ReadersTrue` at `layoutBTops`, from the hm96 slots' commit strings

`layoutBTops` records two SHA-512 calls at each hm96 slot it reads (`hmCalls`): at each held port (`heldRows`) and, at a
RecCap, at port `top` of each instance (`capsCalls`). Off a witness, at the slot's forms:

* the salt's, `SHA-512(salt_prefix ‖ y) = c`, with `y` the slot's salt inputs and `c` the last 64 bytes of its output
  `b ‖ c`;
* the row's, `SHA-512(row_prefix ‖ row) = d`, with the row at the port's unit input columns and `d` the XOR form
  `b ⊕ Φy` (`hmDigest`).

Both are true evaluations wherever the slot's output is the commit string of its row and salt (`SlotCommits`,
`Binding.commitString`, `b = SHA-512(row) ⊕ M·y`, `c = SHA-512(salt_prefix ‖ y)`): `hmCalls_true`, bit for bit, where
`d`'s mask `Φy` cancels `b`'s `M·y` (`maskV_filter`). So `ReadersTrue` at `layoutBTops` is `HmSlotsCommit`, every
slot's commit string at the outer relation (`readersTrue_layoutBTops`), and `ksLin2_deployed_slots` and
`bitsLin2_deployed_slots` take that in its place.

`HmSlotsCommit` is a fact about the slots' circuits at `Flock.check`, the form `Hm.row_bc_flat` and
`Hidden.port_commit_rowsP` prove at a satisfied table's columns, and is open here.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms formBit)
open FlockSoundness.Discharge.Hm (bitL)

/-! ## Bits off forms -/

theorem testBit_bitsNat : ∀ (bs : List Bool) (i : ℕ), (Claims.Recursion.VStar.bitsNat bs).testBit i = bs.getD i false
  | [], i => by simp [Claims.Recursion.VStar.bitsNat]
  | b :: bs, 0 => by
      show ((if b then 1 else 0) + 2 * Claims.Recursion.VStar.bitsNat bs).testBit 0 = b
      cases b <;> simp [Nat.testBit_zero]
  | b :: bs, i + 1 => by
      show ((if b then 1 else 0) + 2 * Claims.Recursion.VStar.bitsNat bs).testBit (i + 1) = bs.getD i false
      rw [Nat.testBit_succ, show ((if b then 1 else 0) + 2 * Claims.Recursion.VStar.bitsNat bs) / 2 =
        Claims.Recursion.VStar.bitsNat bs by cases b <;> simp <;> omega, testBit_bitsNat bs i]

/-- **A bit of bits read as bytes** is that bit. -/
theorem bitL_bytesOf {bs : List Bool} {q : ℕ} (hq : q < bs.length) :
    bitL (Claims.Recursion.VStar.bytesOf bs) q = bs[q] := by
  unfold bitL Claims.Recursion.VStar.bytesOf
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some,
    Option.getD_some, Claims.Recursion.VStar.toNat_byte, show (256 : ℕ) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow,
    testBit_bitsNat, List.getD_eq_getElem?_getD, List.getElem?_take, List.getElem?_drop,
    show 8 * (q / 8) + q % 8 = q by omega, List.getElem?_eq_getElem hq]
  simp [Nat.mod_lt _ (show 8 > 0 by omega)]

theorem bitL_readForms {w : Claims.Flock.Wit} {fs : List HmNets.Lin} {q : ℕ} (hq : q < fs.length) :
    bitL (readForms w fs) q = formBit w fs[q] := by
  unfold readForms
  rw [bitL_bytesOf (by simpa using hq), List.getElem_map]

/-- **Forms whose bits are a byte string's read as it.** -/
theorem readForms_eq_of_bits {w : Claims.Flock.Wit} {fs : List HmNets.Lin} {l : List UInt8}
    (hlen : fs.length = 8 * l.length) (h : ∀ q (hq : q < fs.length), formBit w fs[q] = bitL l q) :
    readForms w fs = l := by
  unfold readForms
  have e : fs.map (formBit w) = (List.range (8 * l.length)).map (bitL l) :=
    List.ext_getElem (by simp [hlen]) fun q h1 _ => by
      simp only [List.getElem_map, List.getElem_range]
      exact h q (by simpa using h1)
  rw [e, Claims.Recursion.VStar.bytesOf_bitL]

theorem formBit_var (w : Claims.Flock.Wit) (v : ℕ) : formBit w (HmNets.Lin.var v) = w.hid.getD v false := by
  simp [formBit, HmNets.Lin.var]

/-- **The mask's bit as an XOR over the key's set bits**: `maskV`'s sum of `key_{i+j} · y_j`, the XOR of `y_j` over
the `j` at which key bit `i + j` is set, as `hmDigest` lists them. -/
theorem foldl_xor_filter (p u : ℕ → Bool) : ∀ (l : List ℕ) (c : Bool),
    l.foldl (fun acc j => acc ^^ (p j && u j)) c = (c ^^ ((l.filter p).map u).foldr (fun b acc => b ^^ acc) false)
  | [], c => by simp
  | j :: l, c => by
      rw [List.foldl_cons, foldl_xor_filter p u l]
      cases hp : p j <;> simp [hp]

theorem maskV_filter (key : ByteArray) (y : ℕ → Bool) (i : ℕ) :
    FlockSoundness.Discharge.Hm.maskV key y i =
      (((List.range 1536).filter fun j => Hm96.bit key (i + j)).map y).foldr (fun b acc => b ^^ acc) false := by
  rw [FlockSoundness.Discharge.Hm.maskV, foldl_xor_filter, Bool.false_xor]

/-! ## A slot's calls -/

/-- The prefix port `q`'s row is hashed under, `sha512/row/v1`'s: none past the ports. -/
def rowPfx (c : Circuit) (q : ℕ) : Msg := (c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD []

/-- **Port `q`'s hm96 slot at instance `n` outputs its commit string**, off the witness `w`: its `b ‖ c` (`hmCommit`)
is the commit string of the row prefix and the row at port `nm`'s unit input columns (`rowHidden`), under the salt at
the slot's salt inputs (`hmSalt`). -/
def SlotCommits (c : Circuit) (nm : String) (q n : ℕ) (w : Claims.Flock.Wit) : Prop :=
  readForms w (hmCommit c q n) = FlockSoundness.Binding.commitString
    (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)) (readForms w (hmSalt c q n))

theorem length_xorMask (r y : List UInt8) :
    (Merkle.xorBytes (FlockSoundness.Binding.digBytes (Refine.H512 r)) (Refine.K512.mask y)).length = 64 := by
  rw [Merkle.xorBytes, List.length_zipWith, FlockSoundness.Discharge.Hm.length_digBytes, Refine.K512.mask_len]
  rfl

/-- **A slot's calls are true evaluations where it outputs its commit string**: the salt's is the commit string's
second half; the row's digest form `b ⊕ Φy` is `SHA-512(row) ⊕ M·y ⊕ Φy`, and `Φy` is `M·y` (`maskV_filter`). -/
theorem sha512_eq (m : Msg) : FlockDeploy.sha512 m = FlockSoundness.Binding.digBytes (Refine.H512 m) := rfl

theorem hmCalls_true {c : Circuit} {nm : String} {q n : ℕ} {w : Claims.Flock.Wit} (h : SlotCommits c nm q n w) :
    ∀ p ∈ hmCalls c nm q n, FlockDeploy.sha512 (p.1 ++ readForms w p.2.1) = readForms w p.2.2 := by
  have hb : ∀ t (ht : t < 1024), w.hid.getD (slotHid c n (hmBase c q n) (hmOutCol c + t)) false =
      bitL (FlockSoundness.Binding.commitString
        (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))
        (readForms w (hmSalt c q n))) t := fun t ht => by
    have e := bitL_readForms (w := w) (fs := hmCommit c q n) (q := t) (by simpa [hmCommit] using ht)
    rw [h] at e
    rw [e]
    simp [hmCommit, formBit_var]
  have hy : ∀ j (hj : j < 1536), w.hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false =
      bitL (readForms w (hmSalt c q n)) j := fun j hj => by
    rw [bitL_readForms (fs := hmSalt c q n) (by simpa [hmSalt] using hj)]
    simp [hmSalt, formBit_var]
  intro p hp
  simp only [hmCalls, List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases hp with rfl | rfl
  · rw [sha512_eq]
    refine (readForms_eq_of_bits (by simp [hmCommit, FlockSoundness.Discharge.Hm.length_digBytes])
      fun t ht => ?_).symm
    have ht' : t < 512 := by simpa [hmCommit] using ht
    rw [List.getElem_drop]
    simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var]
    rw [hb _ (by omega), FlockSoundness.Binding.commitString,
      FlockSoundness.Discharge.Hm.bitL_append_hi (length_xorMask _ _)]
    rfl
  · rw [sha512_eq]
    refine (readForms_eq_of_bits (by simp [hmDigest, FlockSoundness.Discharge.Hm.length_digBytes])
      fun i hi => ?_).symm
    have hi' : i < 512 := by simpa [hmDigest] using hi
    have hm := FlockSoundness.Discharge.Hm.bitL_mask (readForms w (hmSalt c q n)) hi'
    rw [← Refine.K512_mask] at hm
    have hx := FlockSoundness.Discharge.Hm.bitL_xorBytes (FlockSoundness.Discharge.Hm.length_digBytes
      (Refine.H512 (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))))
      (Refine.K512.mask_len (readForms w (hmSalt c q n))) hi'
    have hci := hb i (by omega)
    rw [FlockSoundness.Binding.commitString, FlockSoundness.Discharge.Hm.bitL_append_lo (length_xorMask _ _) hi', hx,
      hm, maskV_filter] at hci
    have hmap : ((((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
        fun j => slotHid c n (hmBase c q n) (hmIn c 2 + j)).map fun v => w.hid.getD v false) =
        ((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
          (bitL (readForms w (hmSalt c q n))) := by
      rw [List.map_map]
      exact List.map_congr_left fun j hj => hy j (List.mem_range.1 (List.mem_filter.1 hj).1)
    simp only [hmDigest, List.getElem_map, List.getElem_range, formBit, List.foldr_cons, List.foldr_map]
    simp only [← hmap, List.foldr_map] at hci
    rw [hci, Bool.xor_assoc, Bool.xor_self, Bool.xor_false]
    rfl

/-! ## The layout's slots -/

/-- **Every hm96 slot `layoutBTops` reads at part `s` outputs its commit string** off the part's witness `w`, part `s`
of `r` having the circuit `c`: at each held port (`heldRows`) and, where part `s`'s statement is a RecCap, at port `top`
of each of its instances. -/
def SlotsCommit (r : XR) (s : ℕ) (c : Circuit) (w : Claims.Flock.Wit) : Prop :=
  (∀ h ∈ heldRows r s, SlotCommits c h.2.1 h.2.2.1 h.2.2.2 w) ∧
    ∀ l N, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N) → ∀ q, capPort c = some q →
      ∀ n < N, SlotCommits c "top" q n w

/-- **V\*'s hm96 slots output their commit strings at the outer relation** (a code fact about V*'s circuits at
`Flock.check`): wherever part `s`'s V\* statement, staged at any rounds, is in the outer relation at `w`, every hm96
slot `layoutBTops` reads at part `s` outputs its commit string (`SlotsCommit`). -/
def HmSlotsCommit {Col T : Type} (DO : Claims.Flock.Deploy Coin Col T) (st : Stage) : Prop :=
  ∀ (x : Msg) (ms : List (Msg × Out)) (s : ℕ) (w : Claims.Flock.Wit) (r : XR) (c : Circuit),
    Claims.Flock.check DO.parse (staged st x ms s) w = true → readX x = some r →
      r.w.parts[s]?.bind partCircuit = some c → SlotsCommit r s c w

theorem mem_capsCalls {r : XR} {s : ℕ} {p : Msg × List HmNets.Lin × List HmNets.Lin} (hp : p ∈ capsCalls r s) :
    ∃ l N c q n, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N) ∧
      r.w.parts[s]?.bind partCircuit = some c ∧ capPort c = some q ∧ n < N ∧ p ∈ hmCalls c "top" q n := by
  unfold capsCalls at hp
  generalize hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = o at hp
  rcases o with _ | st
  · simp at hp
  · simp only [Option.bind_some] at hp
    split at hp
    · rename_i l N
      generalize hc : r.w.parts[s]?.bind partCircuit = oc at hp
      rcases oc with _ | c
      · simp at hp
      · generalize hq : capPort c = oq at hp
        rcases oq with _ | q
        · simp at hp
        · simp only [Option.bind_some, Option.map_some, Option.getD_some, List.mem_flatMap, List.mem_range] at hp
          obtain ⟨n, hn, hp⟩ := hp
          exact ⟨l, N, c, q, n, hst, hc, hq, hn, hp⟩
    · simp at hp

/-- **`ReadersTrue` at `layoutBTops` from the slots' commit strings**: every call the readers record is a part's, at one
of the slots `layoutBTops` reads (`layoutBDeployed`'s held ports, or a RecCap's port `top`), off a witness in the outer
relation at the part's staged statement; there the slot outputs its commit string, so the call is true
(`hmCalls_true`). -/
theorem readersTrue_layoutBTops {Col T : Type} (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin)))
    (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) (h : HmSlotsCommit DO b.st) :
    ReadersTrue FlockDeploy.sha512 IV { b with rd := (layoutBTops fin).readers } DO := by
  intro x as cs ws _ _ hO e he
  obtain ⟨s, hs, he⟩ := Claims.Recursion.VStar.part_of_mem_calls he
  have hsn : s < nS x := hO.1 ▸ hs
  have hck : Claims.Flock.check DO.parse (staged b.st x (roundsOf as cs) s) ws[s] = true := by
    have h2 := hO.2 s hs
    rwa [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hsn, Option.map_some,
      Option.getD_some] at h2
  change e ∈ (layoutBTops fin).calls x s ws[s] at he
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
  cases hr : readX x with
  | none => simp [layoutBTops, layoutBDeployed, hr] at hp
  | some r =>
    have hp' : p ∈ layoutBDeployed.shaAt x s ∨ p ∈ capsCalls r s := by
      simpa [layoutBTops, hr] using hp
    rcases hp' with hp | hp
    · cases hc : r.w.parts[s]?.bind partCircuit with
      | none => simp [layoutBDeployed, hr, hc] at hp
      | some c =>
        simp only [layoutBDeployed, hr, hc, Option.bind_some, Option.map_some, Option.getD_some,
          List.mem_flatMap] at hp
        obtain ⟨hh, hmem, hp⟩ := hp
        exact hmCalls_true ((h x _ s ws[s] r c hck hr hc).1 hh hmem) p hp
    · obtain ⟨l, N, c, q, n, hst, hc, hq, hn, hp⟩ := mem_capsCalls hp
      exact hmCalls_true ((h x _ s ws[s] r c hck hr hc).2 l N hst q hq n hn) p hp

/-! ## `KSLin2` and `BitsLin2` with the slots' commit strings -/

/-- **`KSLin2` at the coin server's record, with `ReadersTrue` as the slots' commit strings**: `ksLin2_deployed_rec`,
with `ReadersTrue` from `HmSlotsCommit` (`readersTrue_layoutBTops`). What stays open at the dense form is `PartChecksB`,
`PartOpensB` and `DecompB`, `HmSlotsCommit`, `ReadersLogColl` and `ReadersRunPlayed`. -/
theorem ksLin2_deployed_slots (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    HmSlotsCommit (deployZkRec _root_.Flock.Readable.relation) b.st →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hS hRL hRP
  exact ksLin2_deployed_rec b aux nOf rest fin wE join final w hP (readersTrue_layoutBTops _ _ bR _ hS) hRL hRP

/-- **`BitsLin2` at the coin server's record, with `ReadersTrue` as the slots' commit strings**:
`bitsLin2_deployed_rec`, with `ReadersTrue` from `HmSlotsCommit`. It reads `StrictBirthday` and `ExpectedCR`, the named
SHA-512 collision assumptions, and `ksLin2_deployed_slots`' open premises: a bound on a number under named premises, not
a proved security level. -/
theorem bitsLin2_deployed_slots (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    HmSlotsCommit (deployZkRec _root_.Flock.Readable.relation) b.st →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hS hRL hRP
  exact bitsLin2_deployed_rec b aux nOf rest fin wE join final w hP (readersTrue_layoutBTops _ _ bR _ hS) hRL hRP

end RecDeploy
