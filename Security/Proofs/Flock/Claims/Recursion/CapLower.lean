import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.Claims.Recursion.CapOpens
import Proofs.Flock.Claims.Recursion.SlotsCommit
import Proofs.Flock.Soundness.Binding.Leaf

/-!
# The cap's lowering at the deployed layout (U14b)

A `rec-open` piece's registration tie (`OpenTie`) has two halves: the position bits the copy reads (LOWOPEN's
`CopyBitsR`), and its climb's top, the first 64 bytes of its `out` row, which must be the side's cap entry at the
opened position (`CopyTopR`). This file reduces the second half to two facts.

* **The binding** (`OutBindR`): a level part's `out` slot at an opening and a RecCap part's `top` slot at the matching
  instance commit to one registered row of `rec-top-L<l>` (`RecCoins.checkOpenTops`, `RecVStar.checkTopRows`;
  `capRow_bytes`). Both ports are 64 words, so their rows' hm96 inputs share a prefix, and the two slots' calls force
  the rows equal unless some two calls collide (`slot_rows_eq`). The calls are two parts', so the collision escape
  this needs is `DecompB`'s, over every part's calls (`LayoutB.allCalls`), not a part's own (`PartAccepts`, `Joint`).
  `JointNC` is `Joint` with that escape excluded, and `decompB_of_jointNC` is `decompB_of_joint` at it: `DecompB`'s
  conclusion already has the escape as its right disjunct.
* **The reader** (`SideCapR`): the side's cap entry is the node RecCap's opened row carries. A RecCap's opened row is
  its whole hidden `top` row, 1024 bits (`rowHidden`, eight bits a byte), the node then 64 bytes of padding, while
  a climb's top and `FitsRep.sizes`' cap entries are 64 bytes (`DecodeZL.ProofSizes.capEach`, `rcapsEach`). So the
  reader takes each cap row's node (`FlockDeploy.nodeOf`): `topGroups` does, so `ligR`'s recursive caps do, but
  `capR` reads round 0's rows whole, at which `SideCapR` fails at level 0.

`capTop_of` gives the climb's top (`CapTopNC`, `CopyTopR` away from a joint collision) from the two, and
`openPieces_deployedNC` is LOWOPEN's `openPieces_deployed` at `JointNC`.
-/

namespace RecDeploy.CapLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP nodeOf)
open Claims.Recursion.VStarB
open FlockVBridge.Family (bytesAt outCol)
open RecDeploy.OpenLower (Joint StagedAt AtPieces AtOpened ownQR ownQR_some copyR copyAtR schAt sideAtR sideLevel
  OpenCopyR CopyBitsR CopyTopR openPieces_computesA)

/-! ## `DecompB` with the parts' joint collision escape -/

/-- **`Joint`, with no collision among every part's calls**: what `DecompB` leaves to prove once its escape
(`findColl (L.allCalls x ws)`) is excluded. -/
def JointNC (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (L : LayoutB)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) : Prop :=
  Joint H D L chk x c ws ∧ (Claims.findColl (L.allCalls x ws)).isSome = false

/-- **`DecompB` is antitone in the parts' checks at their own messages, away from a joint collision**: it holds at
`chk'` wherever it holds at `chk` and, with no two of every part's calls colliding, every part's `chk'` at the parts'
own messages gives every part's `chk` there. -/
theorem decompB_of_jointNC {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {IV : Claims.Verifier Coin}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop} {chk chk' : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), JointNC H D L chk' x c ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompB H D IV L shape chk' := by
  intro x st hx hf c ws hl hT hC
  by_cases hcol : (Claims.findColl (L.allCalls x ws)).isSome = true
  · exact Or.inr hcol
  · exact hD x st hx hf c ws hl hT (hle x st hx hf c ws ⟨⟨hl, hT, hC⟩, by simpa using hcol⟩)

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

set_option maxRecDepth 20000 in
/-- **Two hm96 slots with one commit string hold one row**, wherever both output their commit strings
(`SlotCommits`), their rows are hashed under one prefix (`rowPfx`: ports of one word count), and both slots' calls
(`hmCalls`) are among true evaluations `l` with no collision. -/
theorem slot_rows_eq {c c' : Circuit} {nm nm' : String} {q q' n n' : ℕ} {w w' : Claims.Flock.Wit}
    (h : SlotCommits c nm q n w) (h' : SlotCommits c' nm' q' n' w') (hpfx : rowPfx c q = rowPfx c' q')
    (hcm : readForms w (hmCommit c q n) = readForms w' (hmCommit c' q' n'))
    {l : List (Msg × Msg)} (hT : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2) (hnc : (Claims.findColl l).isSome = false)
    (hin : ∀ p ∈ hmCalls c nm q n, (p.1 ++ readForms w p.2.1, readForms w p.2.2) ∈ l)
    (hin' : ∀ p ∈ hmCalls c' nm' q' n', (p.1 ++ readForms w' p.2.1, readForms w' p.2.2) ∈ l) :
    readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var) =
      readForms w' (((rowHidden c' nm' n').getD []).map HmNets.Lin.var) := by
  have hr := List.mem_map_of_mem (f := Prod.fst) (hin ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
    ((rowHidden c nm n).getD []).map HmNets.Lin.var, hmDigest c q n) (by simp [hmCalls]))
  have hr' := List.mem_map_of_mem (f := Prod.fst) (hin' ((c'.ports[q']?.map fun p =>
    (HmRow.rowPrefix p.words).data.toList).getD [], ((rowHidden c' nm' n').getD []).map HmNets.Lin.var,
    hmDigest c' q' n') (by simp [hmCalls]))
  have hsp : FlockSoundness.Refine.K512.saltPrefix = Hm96.Default512.saltPrefix.data.toList := by
    unfold FlockSoundness.Refine.K512; rfl
  have hy0 := List.mem_map_of_mem (f := Prod.fst)
    (hin (Hm96.Default512.saltPrefix.data.toList, hmSalt c q n, (hmCommit c q n).drop 512) (by simp [hmCalls]))
  have hy0' := List.mem_map_of_mem (f := Prod.fst)
    (hin' (Hm96.Default512.saltPrefix.data.toList, hmSalt c' q' n', (hmCommit c' q' n').drop 512) (by simp [hmCalls]))
  have hy : FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0
  have hy' : FlockSoundness.Refine.K512.saltPrefix ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0'
  have e := commit_bind (r := rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))
    (r' := rowPfx c' q' ++ readForms w' (((rowHidden c' nm' n').getD []).map HmNets.Lin.var))
    (y := readForms w (hmSalt c q n)) (y' := readForms w' (hmSalt c' q' n')) hT hnc hr hr' hy hy'
    (h.symm.trans (hcm.trans h'))
  rw [hpfx] at e
  exact List.append_cancel_left e

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

/-- No two of every part's calls collide (`DecompB`'s escape excluded; `JointNC`'s second conjunct). -/
abbrev NoJointColl (x : Msg) (ws : List Claims.Flock.Wit) : Prop :=
  (Claims.findColl (L.allCalls x ws)).isSome = false

/-- **The binding** (U14b's own fact): at a level part's piece, away from a joint collision, at every RecCap part of
the piece's level holding the row the opening climbs to (`capIdx`), the copy's climb's top is the node RecCap's `top`
row carries there. -/
def OutBindR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    NoJointColl L x ws → ∀ (r : XR) (ss : Array VStar.Stmt), readX x = some r →
      _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∀ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') →
      r.w.parts[s']?.bind partCircuit = some c' → ws[s']? = some w' →
      capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH < N' →
      bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
        nodeOf (capRow c' w' (capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH))

/-- **The reader** at `sideAtR rd`: at every reading of the last message, some RecCap part of the piece's level holds
the row the opening climbs to, and the side's cap entry at the opening's node is the node that row carries. -/
def SideCapR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ _ _ _ q =>
    AtOpened D Bd x (L.msgs x ws) fun ms' =>
      ∃ (r : XR) (ss : Array VStar.Stmt), readX x = some r ∧ _root_.Flock.RecVStar.statements r.shape = .ok ss ∧
      ∃ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') ∧
        r.w.parts[s']?.bind partCircuit = some c' ∧ ws[s']? = some w' ∧
        capIdx (Claims.Recursion.xI D x) q
          (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH < N' ∧
        (FlockSoundness.Refine.capAt (sideAtR rd (Claims.Recursion.xI D x) ms' q).cap
            (sideAtR rd (Claims.Recursion.xI D x) ms' q).lig q.level).getD
          ((posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) >>> HH)
          ByteArray.empty =
          nodeOf (capRow c' w' (capIdx (Claims.Recursion.xI D x) q
            (posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) HH))

/-- **The cap tie away from a joint collision**: LOWOPEN's `CopyTopR`, given `NoJointColl`. -/
def CapTopNC (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c ws LANES HH _ _ w p ht q =>
    NoJointColl L x ws → AtOpened D Bd x (L.msgs x ws) fun ms' =>
      bytesAt (copyR (Claims.Recursion.xI D x) w p ht q) (outCol LANES HH) 64 =
        (FlockSoundness.Refine.capAt (sideAtR rd (Claims.Recursion.xI D x) ms' q).cap
            (sideAtR rd (Claims.Recursion.xI D x) ms' q).lig q.level).getD
          ((posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q) >>>
            (((sideAtR rd (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).d -
              ((sideAtR rd (Claims.Recursion.xI D x) ms' q).sch.levels[q.level]!).cap)) ByteArray.empty

/-- **The cap tie from the binding and the reader**, at the deployed reader's statements at the dense form: the level
part's siblings below the cap, `HH`, are the side's level's `d − c` (`sideLevel`). -/
theorem capTop_of (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hO : OutBindR H D shape untr b cmt L Bd pieces posAt) (hS : SideCapR H D shape untr b cmt L Bd rd pieces posAt) :
    CapTopNC H D shape untr b cmt L Bd rd pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht hnc os last hfin hl
  obtain ⟨r', ss', hr', hss', s', N', c', w', hs', hc', hw', hlt, hcap⟩ :=
    hS x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht os last hfin hl
  obtain ⟨-, hHH⟩ := sideLevel D shape rd hP hD hx hf hr hss hsv ho (Bd.opened os last)
  rw [← hHH]
  exact (hO x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho ht hnc r' ss' hr' hss' s' N' c' w'
    hs' hc' hw' hlt).trans hcap.symm

/-- **The `rec-open` pieces' share at the deployed layout, away from a joint collision**: LOWOPEN's
`openPieces_deployed` at `JointNC`, with `OpenTie` as its two halves, the cap tie (`CapTopNC`) among them. -/
theorem openPieces_deployedNC (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hC : OpenCopyR H D shape untr b cmt L Bd rd pieces) (hB : CopyBitsR H D shape untr b cmt L Bd rd pieces posAt)
    (hT : CapTopNC H D shape untr b cmt L Bd rd pieces posAt)
    {x : Msg} {st : Claims.Recursion.Stmt} (hx : D.parse x = some st) (hf : shape st.form) {c : ℕ → Coin}
    {ws : List Claims.Flock.Wit}
    (hJ : JointNC H D L (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt)
      x c ws) {s : ℕ} (hs : s < D.nS x) :
    PiecesComputes Bd D pieces (openHolds (sideAtR rd) posAt) ownQR s x c (L.msgs x ws) := by
  intro q hq ho
  obtain ⟨-, r, ss, hr, hss, LANES, HH, n, hsv⟩ := ownQR_some ho
  refine openPieces_computesA (hJ.1.2.2 s hs) hr hss (Or.inl hsv) Bd pieces ownQR (sideAtR rd) posAt
    (fun xO w p hxO hsz hcp hA hT' hc hsh q' hq' ho' => ?_) q hq ho
  have hst : StagedAt H D untr cmt L s x c (L.msgs x ws) (.recOpen q.level LANES HH n) xO w p :=
    ⟨hxO, hsz, hcp _ hsv, hA, hT', hc, hsh⟩
  obtain ⟨hob, hgb, hub, hcp'⟩ :=
    hC x st hx hf c ws hJ.1 r ss s q.level LANES HH n hr hss hsv xO w p hst q' hq' ho' hsz
  exact ⟨0, hsz, _, _, _, hob, hgb, hub, fun os last hfin hl =>
    ⟨hcp' os last hfin hl, And.intro
      (hB x st hx hf c ws hJ.1 r ss s q.level LANES HH n hr hss hsv xO w p hst q' hq' ho' hsz os last hfin hl)
      (hT x st hx hf c ws hJ.1 r ss s q.level LANES HH n hr hss hsv xO w p hst q' hq' ho' hsz hJ.2 os last hfin
        hl)⟩⟩

end Deployed

end RecDeploy.CapLower
