import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.Claims.Recursion.CapOpens
import Proofs.Flock.Claims.Recursion.SlotsCommit
import Proofs.Flock.Claims.Recursion.DeployedR
import Proofs.Flock.Soundness.Binding.Leaf

/-!
# The cap's lowering at the deployed layout (U14b)

A `rec-open` piece's registration tie (`OpenTie`) has two halves: the position bits the copy reads
(`CopyBitsR`), and its climb's top, the first 64 bytes of its `out` row, which must be the side's cap entry at the
opened position (`CopyTopR`). This file reduces the second half to two facts.

* **The binding** (`OutBindR`): a level part's `out` slot at an opening and a RecCap part's `top` slot at the matching
  instance commit to one registered row of `rec-top-L<l>` (`RecCoins.checkOpenTops`, `RecVStar.checkTopRows`;
  `capRow_bytes`). Both ports are 64 words, so their rows' hm96 inputs share a prefix, and the two slots' calls force
  the rows equal unless some two calls collide (`rows_eq_of_calls`; true calls are the slot's commit string,
  `commit_of_true`). The calls are two parts', so the collision escape this needs is over every part's calls
  (`LayoutB.allCalls`), which `Joint` excludes, and `layR` records both slots' (`out_top_rows`).
* **The reader** (`SideCapR`): the side's cap entry is the node RecCap's opened row carries. A RecCap's opened row is
  its whole hidden `top` row, 1024 bits (`rowHidden`, eight bits a byte), the node then 64 bytes of padding, while
  a climb's top and `FitsRep.sizes`' cap entries are 64 bytes (`DecodeZL.ProofSizes.capEach`, `rcapsEach`). So the
  reader takes each cap row's node (`FlockDeploy.nodeOf`), as `topGroups` and `capR` do.

`capTop_of` gives `CopyTopR` at `sideAtR rd` from the two, and `openTieR_deployed` its `OpenTieR`.
-/

namespace RecDeploy.CapLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP nodeOf)
open Claims.Recursion.VStarB
open FlockVBridge.Family (bytesAt outCol)
open RecDeploy.OpenLower (AtPieces AtOpened copyR schAt sideAtR sideLevel CopyBitsR CopyTopR OpenTieR openTieR_of)

/-! ## What a cap row carries -/

/-- RecCap's `top` row at instance `n`, off part witness `w'` in its circuit `c'` (`rowHidden`, eight bits a byte):
the node then 64 bytes of padding (`FlockDeploy.nodeRow`), whose node is its `nodeOf`. -/
def capRow (c' : Circuit) (w' : Claims.Flock.Wit) (n : ℕ) : ByteArray :=
  ⟨(readForms w' (((rowHidden c' "top" n).getD []).map HmNets.Lin.var)).toArray⟩

/-- **Where a piece's rep's nodes start among `rec-top-L<l>`'s rows** (`RecVStar.topRow`, `checkLevelTopReads`' `first`):
level 0's `root_B` is both reps' one top, and above it rep 1's rows follow rep 0's `2^cap` nodes. -/
def capFirst (xI : Msg) (q : OpenPiece) : ℕ :=
  if q.level = 0 then 0 else q.rep * 2 ^ ((schAt xI 0).levels[q.level]!).cap

/-- The `rec-top-L<l>` row a piece's opening at position `pos` climbs to, below `HH` siblings. -/
def capIdx (xI : Msg) (q : OpenPiece) (pos HH : ℕ) : ℕ := capFirst xI q + (pos >>> HH)

/-! ## A commit string fixes its row, up to a collision among recorded calls -/

/-- **Equal commit strings have equal rows**, up to a collision among true SHA-512 evaluations `l` that hold both
rows' and both salts' hashes: equal `c` halves give equal salts, so equal masks, so equal row digests
(`hm96Pair_spec`'s argument, at recorded calls). -/
theorem commit_bind {l : List (Msg × Msg)} (hT : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {r y r' y' : Msg}
    (hr : r ∈ l.map Prod.fst) (hr' : r' ∈ l.map Prod.fst)
    (hy : FlockSoundness.Refine.K512.saltPrefix ++ y ∈ l.map Prod.fst)
    (hy' : FlockSoundness.Refine.K512.saltPrefix ++ y' ∈ l.map Prod.fst)
    (h : FlockSoundness.Binding.commitString r y = FlockSoundness.Binding.commitString r' y') : r = r' := by
  have hcol : ∀ u v, u ∈ l.map Prod.fst → v ∈ l.map Prod.fst → u ≠ v →
      FlockDeploy.sha512 u ≠ FlockDeploy.sha512 v := fun u v hu hv hne he => by
    rw [Claims.findColl_isSome_mem FlockDeploy.sha512 l hT u v hu hv ⟨hne, he⟩] at hnc
    exact absurd hnc (by decide)
  obtain ⟨hb, hc⟩ := FlockSoundness.Binding.commitString_split h
  have hyy : y = y' := by
    by_contra hne
    exact hcol _ _ hy hy' (fun e => hne (List.append_cancel_left e)) (by rw [sha512_eq, sha512_eq]; exact hc)
  subst hyy
  by_contra hne
  refine hcol r r' hr hr' hne ?_
  rw [sha512_eq, sha512_eq]
  exact FlockSoundness.Merkle.xorBytes_inj
    (by rw [FlockSoundness.Refine.hL512, FlockSoundness.Refine.K512.mask_len])
    (by rw [FlockSoundness.Refine.hL512, FlockSoundness.Refine.K512.mask_len]) hb

/-- **A slot whose two recorded calls are true outputs its commit string** (`hmCalls_true`'s converse, at any row
forms): the salt's call makes `c` the salt's hash, and the row's makes `b ⊕ Φy` the row's, where `Φy` is the mask
`M·y` (`maskV_filter`), so `b = SHA-512(row) ⊕ M·y`. -/
theorem commit_of_true {c : Circuit} {q n : ℕ} {w : Claims.Flock.Wit} {R : Msg}
    (hs : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n)) =
      readForms w ((hmCommit c q n).drop 512))
    (hr : FlockDeploy.sha512 R = readForms w (hmDigest c q n)) :
    readForms w (hmCommit c q n) = FlockSoundness.Binding.commitString R (readForms w (hmSalt c q n)) := by
  have hsp : FlockSoundness.Refine.K512.saltPrefix = Hm96.Default512.saltPrefix.data.toList := by
    unfold FlockSoundness.Refine.K512; rfl
  have hy : ∀ j (hj : j < 1536), w.hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false =
      FlockSoundness.Discharge.Hm.bitL (readForms w (hmSalt c q n)) j := fun j hj => by
    rw [bitL_readForms (fs := hmSalt c q n) (by simpa [hmSalt] using hj)]
    simp [hmSalt, formBit_var]
  refine readForms_eq_of_bits (by simp [hmCommit, FlockSoundness.Discharge.Hm.length_commitString]) fun t ht => ?_
  have ht' : t < 1024 := by simpa [hmCommit] using ht
  simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var]
  by_cases hlo : t < 512
  · have hm := FlockSoundness.Discharge.Hm.bitL_mask (readForms w (hmSalt c q n)) hlo
    rw [← FlockSoundness.Refine.K512_mask] at hm
    have hx := FlockSoundness.Discharge.Hm.bitL_xorBytes (FlockSoundness.Discharge.Hm.length_digBytes
      (FlockSoundness.Refine.H512 R)) (FlockSoundness.Refine.K512.mask_len (readForms w (hmSalt c q n))) hlo
    rw [FlockSoundness.Binding.commitString, FlockSoundness.Discharge.Hm.bitL_append_lo (length_xorMask _ _) hlo, hx,
      hm, maskV_filter, ← sha512_eq, hr, bitL_readForms (by simpa [hmDigest] using hlo)]
    have hmap : ((((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (t + j)).map
        fun j => slotHid c n (hmBase c q n) (hmIn c 2 + j)).map fun v => w.hid.getD v false) =
        ((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (t + j)).map
          (FlockSoundness.Discharge.Hm.bitL (readForms w (hmSalt c q n))) := by
      rw [List.map_map]
      exact List.map_congr_left fun j hj => hy j (List.mem_range.1 (List.mem_filter.1 hj).1)
    simp only [← hmap]
    simp only [hmDigest, List.getElem_map, List.getElem_range, formBit, List.foldr_cons, List.foldr_map]
    rw [Bool.xor_assoc, Bool.xor_self, Bool.xor_false]
  · obtain ⟨u, rfl⟩ : ∃ u, t = 512 + u := ⟨t - 512, by omega⟩
    have e := bitL_readForms (w := w) (fs := (hmCommit c q n).drop 512) (q := u) (by simp [hmCommit]; omega)
    rw [← hs, sha512_eq, List.getElem_drop] at e
    simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var] at e
    rw [FlockSoundness.Binding.commitString, FlockSoundness.Discharge.Hm.bitL_append_hi (length_xorMask _ _)]
    exact e.symm

/-- **Two hm96 slots with one commit string hold one row**, wherever both slots' salt and row calls are among true
evaluations `l` with no collision and their rows are hashed under one prefix (`rowPfx`: ports of one word count).
The calls being true is the slot's commit string (`commit_of_true`), so no code fact about the slots' circuits is
read: the rows may be at any forms (an input row's `rowHidden`, an output row's `outRowForms`). -/
theorem rows_eq_of_calls {l : List (Msg × Msg)} (hT : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {c c' : Circuit} {q q' n n' : ℕ} {w w' : Claims.Flock.Wit}
    {row row' : List HmNets.Lin}
    (hs : (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n),
      readForms w ((hmCommit c q n).drop 512)) ∈ l)
    (hr : (rowPfx c q ++ readForms w row, readForms w (hmDigest c q n)) ∈ l)
    (hs' : (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n'),
      readForms w' ((hmCommit c' q' n').drop 512)) ∈ l)
    (hr' : (rowPfx c' q' ++ readForms w' row', readForms w' (hmDigest c' q' n')) ∈ l)
    (hpfx : rowPfx c q = rowPfx c' q')
    (hcm : readForms w (hmCommit c q n) = readForms w' (hmCommit c' q' n')) :
    readForms w row = readForms w' row' := by
  have hsp : FlockSoundness.Refine.K512.saltPrefix = Hm96.Default512.saltPrefix.data.toList := by
    unfold FlockSoundness.Refine.K512; rfl
  have hs1 : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n)) =
      readForms w ((hmCommit c q n).drop 512) := by
    have h := hT _ hs; exact h
  have hr1 : FlockDeploy.sha512 (rowPfx c q ++ readForms w row) = readForms w (hmDigest c q n) := by
    have h := hT _ hr; exact h
  have hs1' : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n')) =
      readForms w' ((hmCommit c' q' n').drop 512) := by
    have h := hT _ hs'; exact h
  have hr1' : FlockDeploy.sha512 (rowPfx c' q' ++ readForms w' row') = readForms w' (hmDigest c' q' n') := by
    have h := hT _ hr'; exact h
  have e := commit_of_true (c := c) (q := q) (n := n) (w := w) (R := rowPfx c q ++ readForms w row) hs1 hr1
  have e' := commit_of_true (c := c') (q := q') (n := n') (w := w') (R := rowPfx c' q' ++ readForms w' row') hs1' hr1'
  have hy0 : Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst :=
    List.mem_map.2 ⟨_, hs, rfl⟩
  have hy0' : Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst :=
    List.mem_map.2 ⟨_, hs', rfl⟩
  have hy : FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0
  have hy' : FlockSoundness.Refine.K512.saltPrefix ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0'
  have hr0 : rowPfx c q ++ readForms w row ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr, rfl⟩
  have hr0' : rowPfx c' q' ++ readForms w' row' ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr', rfl⟩
  have hb := commit_bind (r := rowPfx c q ++ readForms w row) (r' := rowPfx c' q' ++ readForms w' row')
    (y := readForms w (hmSalt c q n)) (y' := readForms w' (hmSalt c' q' n')) hT hnc hr0 hr0' hy hy'
    (e.symm.trans (hcm.trans e'))
  rw [hpfx] at hb
  exact List.append_cancel_left hb

/-! ## A level's `out` row is its cap's `top` row, at the joint witnesses -/

theorem mem_allCalls {L : LayoutB} {x : Msg} {ws : List Claims.Flock.Wit} {s : ℕ} {w : Claims.Flock.Wit}
    (hw : ws[s]? = some w) {e : Msg × Msg} (he : e ∈ L.calls x s w) : e ∈ L.allCalls x ws := by
  obtain ⟨hs, rfl⟩ := List.getElem?_eq_some_iff.1 hw
  unfold LayoutB.allCalls
  exact List.mem_flatten.2 ⟨L.calls x s ws[s],
    List.mem_iff_getElem.2 ⟨s, by rw [List.length_mapIdx]; exact hs, by rw [List.getElem_mapIdx]⟩, he⟩

theorem mem_capsCalls_of {r : XR} {s l N : ℕ} {c : Circuit} {q n : ℕ}
    (hsv : (_root_.Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N))
    (hc : r.w.parts[s]?.bind partCircuit = some c) (hq : capPort c = some q) (hn : n < N)
    {p : Msg × List HmNets.Lin × List HmNets.Lin} (hp : p ∈ hmCalls c "top" q n) : p ∈ capsCalls r s := by
  unfold capsCalls
  rw [hsv]
  simp only [Option.bind_some, hc, hq, Option.map_some, Option.getD_some, List.mem_flatMap, List.mem_range]
  exact ⟨n, hn, hp⟩

theorem mem_outsCalls_recOpen {r : XR} {s lv LANES HH n : ℕ} {c : Circuit} {q i : ℕ}
    (hsv : (_root_.Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen lv LANES HH n))
    (hc : r.w.parts[s]?.bind partCircuit = some c) (hq : c.ports.findIdx? (·.name == "out") = some q) (hi : i < n)
    {p : Msg × List HmNets.Lin × List HmNets.Lin} (hp : p ∈ DeployedR.outCalls c q i) :
    p ∈ DeployedR.outsCalls r s := by
  simp only [DeployedR.outsCalls, hsv, hc, DeployedR.outPortCalls, hq, List.mem_flatMap, List.mem_range]
  exact ⟨i, hi, hp⟩

/-- **A level's climbed top is RecCap's node row, at `layR` and the joint witnesses**: where every part's recorded
calls off the witnesses `ws` are true and free of collisions (`Joint`), a level part `s`'s `out` slot at an opening
`i` and a RecCap part `s'`'s `top` slot at a node `m` with one commit string, their ports of one word count, hold one
row: `layR` records both slots' calls (`outsCalls`, `capsCalls`), so `rows_eq_of_calls` applies. -/
theorem out_top_rows (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    {s lv LANES HH n : ℕ}
    (hsv : (_root_.Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen lv LANES HH n))
    {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == "out") = some q) {i : ℕ} (hi : i < n) {w : Claims.Flock.Wit}
    (hw : ws[s]? = some w) {s' l N : ℕ}
    (hsv' : (_root_.Flock.RecVStar.statements r.shape).toOption.bind (·[s']?) = some (.cap l N))
    {c' : Circuit} (hc' : r.w.parts[s']?.bind partCircuit = some c') {q' : ℕ} (hq' : capPort c' = some q') {m : ℕ}
    (hm : m < N) {w' : Claims.Flock.Wit} (hw' : ws[s']? = some w') (hpfx : rowPfx c q = rowPfx c' q')
    (hcm : readForms w (hmCommit c q i) = readForms w' (hmCommit c' q' m)) :
    readForms w (DeployedR.outRowForms c q i) =
      readForms w' (((rowHidden c' "top" m).getD []).map HmNets.Lin.var) := by
  have hL : ∀ p ∈ DeployedR.outCalls c q i,
      (p.1 ++ readForms w p.2.1, readForms w p.2.2) ∈ (DeployedR.layR elem).allCalls x ws := fun p hp =>
    mem_allCalls hw (DeployedR.outsCalls_sub_layR elem hr s w p (mem_outsCalls_recOpen hsv hc hq hi hp))
  have hC : ∀ p ∈ hmCalls c' "top" q' m,
      (p.1 ++ readForms w' p.2.1, readForms w' p.2.2) ∈ (DeployedR.layR elem).allCalls x ws := fun p hp =>
    mem_allCalls hw' (by
      rw [DeployedR.layR_calls]
      refine List.mem_append_left _ ?_
      simp only [LayoutB.calls, layoutBTops, hr, Option.map_some, Option.getD_some, List.map_append]
      exact List.mem_append_right _ (List.mem_map.2 ⟨p, mem_capsCalls_of hsv' hc' hq' hm hp, rfl⟩))
  have h1 : (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q i),
      readForms w ((hmCommit c q i).drop 512)) ∈ (DeployedR.layR elem).allCalls x ws := by
    have h := hL (Hm96.Default512.saltPrefix.data.toList, hmSalt c q i, (hmCommit c q i).drop 512)
      (by unfold DeployedR.outCalls; exact List.mem_cons_self)
    dsimp only at h ⊢
    exact h
  have h2 : (rowPfx c q ++ readForms w (DeployedR.outRowForms c q i), readForms w (hmDigest c q i)) ∈
      (DeployedR.layR elem).allCalls x ws := by
    have h := hL ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
      DeployedR.outRowForms c q i, hmDigest c q i)
      (by unfold DeployedR.outCalls; exact List.mem_cons_of_mem _ List.mem_cons_self)
    unfold rowPfx
    dsimp only at h ⊢
    exact h
  have h3 : (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' m),
      readForms w' ((hmCommit c' q' m).drop 512)) ∈ (DeployedR.layR elem).allCalls x ws := by
    have h := hC (Hm96.Default512.saltPrefix.data.toList, hmSalt c' q' m, (hmCommit c' q' m).drop 512)
      (by unfold hmCalls; exact List.mem_cons_self)
    dsimp only at h ⊢
    exact h
  have h4 : (rowPfx c' q' ++ readForms w' (((rowHidden c' "top" m).getD []).map HmNets.Lin.var),
      readForms w' (hmDigest c' q' m)) ∈ (DeployedR.layR elem).allCalls x ws := by
    have h := hC ((c'.ports[q']?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
      ((rowHidden c' "top" m).getD []).map HmNets.Lin.var, hmDigest c' q' m)
      (by unfold hmCalls; exact List.mem_cons_of_mem _ List.mem_cons_self)
    unfold rowPfx
    dsimp only at h ⊢
    exact h
  exact rows_eq_of_calls hT hnc h1 h2 h3 h4 hpfx hcm

/-! ## `OpenTie`'s two halves -/

/-- `OpenTie`'s first half: the copy's position bits are `pos`'s low `d − c`. -/
def BitsTie (sd : OpenSide) (q : OpenPiece) (LANES H pos : ℕ) (v : ℕ → Bool) : Prop :=
  ∀ h < (sd.sch.levels[q.level]!).d - (sd.sch.levels[q.level]!).cap,
    (_root_.Flock.Readable.openIdx v LANES H).testBit h = pos.testBit h

/-- `OpenTie`'s second half: the copy's climb's top is the cap's entry `pos >>> (d − c)`. -/
def TopTie (sd : OpenSide) (q : OpenPiece) (LANES H pos : ℕ) (v : ℕ → Bool) : Prop :=
  bytesAt v (outCol LANES H) 64 =
    (FlockSoundness.Refine.capAt sd.cap sd.lig q.level).getD
      (pos >>> ((sd.sch.levels[q.level]!).d - (sd.sch.levels[q.level]!).cap)) ByteArray.empty

theorem openTie_iff {sd : OpenSide} {q : OpenPiece} {LANES H pos : ℕ} {v : ℕ → Bool} :
    OpenTie sd q LANES H pos v ↔ BitsTie sd q LANES H pos v ∧ TopTie sd q LANES H pos v :=
  Iff.rfl

/-! ## The cap tie at the deployed layout -/

section Deployed

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (rd : Msg → Option FinalP)

/-- **The binding** (U14b's own fact): at a level part's piece, at every RecCap part of the piece's level holding the
row the opening climbs to (`capIdx`), the copy's climb's top is the node RecCap's `top` row carries there. -/
def OutBindR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    ∀ (r : XR) (ss : Array VStar.Stmt), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') →
      r.w.parts[s']?.bind partCircuit = some c' → ws[s']? = some w' →
      capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH < N' →
      bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
        nodeOf (capRow c' w' (capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH))

/-- **A piece's cap entry at the opened messages `ms'`**, at compiled statement `x`, inner statement `xI` and coins
`cs`: some RecCap part of the piece's level holds the row the opening climbs to (`capIdx`), and the side's cap entry
at the opening's node is the node that row carries. -/
abbrev CapEntryR (posAt : Msg → List Coin → OpenPiece → ℕ) (x xI : Msg) (cs : List Coin)
    (ws : List Claims.Flock.Wit) (HH : ℕ) (q : OpenPiece) (ms' : List Msg) : Prop :=
  ∃ (r : XR) (ss : Array VStar.Stmt), readX x = some r ∧ _root_.Flock.RecVStar.statements r.shape = .ok ss ∧
  ∃ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') ∧
    r.w.parts[s']?.bind partCircuit = some c' ∧ ws[s']? = some w' ∧ capIdx xI q (posAt xI cs q) HH < N' ∧
    (FlockSoundness.Refine.capAt (sideAtR rd xI ms' q).cap (sideAtR rd xI ms' q).lig q.level).getD
      (posAt xI cs q >>> HH) ByteArray.empty = nodeOf (capRow c' w' (capIdx xI q (posAt xI cs q) HH))

/-- **The reader** at `sideAtR rd`: at every reading of the last message, some RecCap part of the piece's level holds
the row the opening climbs to, and the side's cap entry at the opening's node is the node that row carries. -/
def SideCapR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws _ HH _ _ _ _ _ q =>
    AtOpened D Bd x (L.msgs x ws) (CapEntryR rd posAt x (Claims.Recursion.xI D x)
      ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) ws HH q)

/-- **The cap tie from the binding and the reader**, at the deployed reader's statements at the dense form: the level
part's siblings below the cap, `HH`, are the side's level's `d − c` (`sideLevel`). -/
theorem capTop_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hO : OutBindR H D shape untr b cmt L Bd pieces posAt) (hS : SideCapR H D shape untr b cmt L Bd rd pieces posAt) :
    CopyTopR H D shape untr b cmt L Bd (sideAtR rd) pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl
  obtain ⟨r', ss', hr', hss', s', N', c', w', hs', hc', hw', hlt, hcap⟩ :=
    hS x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl
  obtain ⟨-, hHH⟩ := sideLevel D shape (sideAtR rd) hP hD hx hf hr hss hsv ho (ms := Bd.opened os last) rfl
  rw [← hHH]
  exact (hO x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht r' ss' hr' hss' s' N' c' w'
    hs' hc' hw' hlt).trans hcap.symm

/-- **The registration ties at the deployed layout at `sideAtR rd`**: `OpenTieR`, its cap half
(`CopyTopR`) from the binding and the reader (`capTop_of`). -/
theorem openTieR_deployed (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hB : CopyBitsR H D shape untr b cmt L Bd (sideAtR rd) pieces posAt)
    (hO : OutBindR H D shape untr b cmt L Bd pieces posAt) (hS : SideCapR H D shape untr b cmt L Bd rd pieces posAt) :
    OpenTieR H D shape untr b cmt L Bd (sideAtR rd) pieces posAt :=
  openTieR_of H D shape untr b cmt L Bd (sideAtR rd) pieces posAt hB (capTop_of H D shape untr b cmt L Bd rd pieces
    posAt hP hD hO hS)

end Deployed

end RecDeploy.CapLower
