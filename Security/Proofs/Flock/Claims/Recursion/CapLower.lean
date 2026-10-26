import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.Claims.Recursion.CapOpens

/-!
# The cap's lowering at the deployed layout (U14b)

A `rec-open` piece's registration tie (`OpenTie`) has two halves: the position bits the copy reads, and its climb's top,
the first 64 bytes of its `out` row, which must be the side's cap entry at the opened position. This file states the
second half at the deployed layout `layR elem` and reduces it to two facts.

* **The binding** (`OutBindR`): a level part's `out` slot at an opening and a RecCap part's `top` slot at the matching
  instance commit to one registered row of `rec-top-L<l>` (`RecCoins.checkOpenTops`, `RecVStar.checkTopRows`;
  `capRow_bytes`). Both ports are 64 words, so their rows' hm96 inputs share a prefix, and the two slots' calls force
  the rows equal unless some two calls collide. The calls are two parts', so the collision escape this needs is
  `DecompB`'s, over every part's calls (`LayoutB.allCalls`), not a part's own (`PartAccepts`, `Joint`). `JointNC` is
  `Joint` with that escape excluded, and `decompB_of_jointNC` is `decompB_of_joint` at it: `DecompB`'s conclusion
  already has the escape as its right disjunct.
* **The reader** (`SideCapR`): the side's cap entry is the node RecCap's opened row carries. A RecCap's opened row is
  its whole hidden `top` row, 1024 bits (`rowHidden`, eight bits a byte), the node then 64 bytes of padding, while
  a climb's top and `FitsRep.sizes`' cap entries are 64 bytes (`DecodeZL.ProofSizes.capEach`, `rcapsEach`). So the
  reader has to take each cap row's first 64 bytes (`capNode`). `RecDeploy.Rows.decR` and `sideAtR` read the whole row
  today, at which `SideCapR` fails and no opened message fits.

`capTop_of` gives the climb's top from the two, and `openPieces_deployedNC` is LOWOPEN's `openPieces_deployed` at
`JointNC`.
-/

namespace RecDeploy.CapLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin setupOf tabsOf)
open Claims.Recursion.VStarB
open FlockVBridge.Family (bytesAt outCol)
open RecDeploy.DeployedR (rdR finalR bdR layR)
open RecDeploy.OpenLower (Joint StagedAt ownQR isLevel copyAtR sideAtR OpenCopyR openPieces_computesA)

/-! ## `DecompB` with the parts' joint collision escape -/

section JointNC

variable {Coin' : Type} [Fintype Coin'] [Nonempty Coin']

/-- **`Joint`, with no collision among every part's calls**: what `DecompB` leaves to prove once its escape
(`findColl (L.allCalls x ws)`) is excluded. -/
def JointNC (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin') (L : LayoutB)
    (chk : ℕ → Msg → (ℕ → Coin') → List Msg → Prop) (x : Msg) (c : ℕ → Coin') (ws : List Claims.Flock.Wit) : Prop :=
  Joint H D L chk x c ws ∧ (Claims.findColl (L.allCalls x ws)).isSome = false

/-- **`DecompB` is antitone in the parts' checks at their own messages, away from a joint collision**: it holds at
`chk'` wherever it holds at `chk` and, with no two of every part's calls colliding, every part's `chk'` at the parts'
own messages gives every part's `chk` there. -/
theorem decompB_of_jointNC {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin'} {IV : Claims.Verifier Coin'}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop} {chk chk' : ℕ → Msg → (ℕ → Coin') → List Msg → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin') (ws : List Claims.Flock.Wit), JointNC H D L chk' x c ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompB H D IV L shape chk' := by
  intro x st hx hf c ws hl hT hC
  cases hcol : (Claims.findColl (L.allCalls x ws)).isSome with
  | true => exact Or.inr hcol
  | false => exact hD x st hx hf c ws hl hT (hle x st hx hf c ws ⟨⟨hl, hT, hC⟩, hcol⟩)

end JointNC

/-! ## What a cap row carries -/

/-- **The node a cap row carries**: its first 64 bytes (CAPPIECES' `nodeOf`). RecCap's `top` row is the node then 64
bytes of padding (`VStar.capUnit`'s `out`, registered as zero rows), and a climb's top is the node alone. -/
def capNode (b : ByteArray) : ByteArray := ⟨(b.data.toList.take 64).toArray⟩

/-- RecCap's `top` row at instance `n`, off part witness `w'` in its circuit `c'` (`rowHidden`, eight bits a byte). -/
def capRow (c' : Circuit) (w' : Claims.Flock.Wit) (n : ℕ) : ByteArray :=
  ⟨(readForms w' (((rowHidden c' "top" n).getD []).map HmNets.Lin.var)).toArray⟩

/-- The inner statement `xI`'s table 0's `fast100` schedule. -/
def schAt (xI : Msg) : Flock.Schedule :=
  match setupOf false xI with
  | some v => (fast100 (tabsOf v.2.2)[0]!.m).getD default
  | none => default

/-- **Where a piece's rep's nodes start among `rec-top-L<l>`'s rows** (`RecVStar.topRow`, `checkLevelTopReads`' `first`):
level 0's `root_B` is both reps' one top, and above it rep 1's rows follow rep 0's `2^cap` nodes. -/
def capFirst (xI : Msg) (q : OpenPiece) : ℕ :=
  if q.level = 0 then 0 else q.rep * 2 ^ ((schAt xI).levels[q.level]!).cap

/-- The copy `copyAtR` names for piece `q` at a parsed level statement `p`. -/
def copyR (p : _root_.Flock.Readable.Parsed) (ht : 0 < p.sts.size) (w : Claims.Flock.Wit) (xI : Msg)
    (q : OpenPiece) : ℕ → Bool :=
  _root_.Flock.Readable.copyOf p 0 ht w (copyAtR p xI q).1 (copyAtR p xI q).2.1 (copyAtR p xI q).2.2

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
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg)
  (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

/-- **A `rec-open` piece of level part `s`, at the parts' own messages, every part accepting and no two calls
colliding**: `OpenCopyR`'s and `OpenTieR`'s context, at `JointNC`. -/
def PieceAt (pieces : Msg → List OpenPiece) (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) (r : XR)
    (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ) (xO : Msg) (w : Claims.Flock.Wit)
    (p : _root_.Flock.Readable.Parsed) (q : OpenPiece) : Prop :=
  JointNC H D (layR elem) (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) (layR elem) shape
      (showsPart untr b) cmt) x c ws ∧
    readX x = some r ∧ _root_.Flock.RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some (.recOpen lv LANES HH n) ∧
    StagedAt H D untr cmt elem s x c ((layR elem).msgs x ws) xO w p ∧
    q ∈ pieces (Claims.Recursion.xI D x) ∧ ownQR x q = some s

/-- The coins the verdict reads at statement `x`: its `k` rounds' (`List.range k`). -/
def coinsAt (x : Msg) (c : ℕ → Coin) : List Coin :=
  (List.range (bdR.DE.V.k (Claims.Recursion.xI D x))).map c

/-- **The binding** (U14b's own fact): at a level part's piece, at every RecCap part of the piece's level holding the
instance the opening's node is, the copy's climb's top is the node RecCap's `top` row carries there. -/
def OutBindR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  ∀ (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ)
    (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed) (q : OpenPiece),
    PieceAt H D shape untr b cmt elem pieces x c ws r ss s lv LANES HH n xO w p q → ∀ ht : 0 < p.sts.size,
    ∀ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') →
      r.w.parts[s']?.bind partCircuit = some c' → ws[s']? = some w' →
      capFirst (Claims.Recursion.xI D x) q + (posAt (Claims.Recursion.xI D x) (coinsAt D x c) q >>> HH) < N' →
      bytesAt (copyR p ht w (Claims.Recursion.xI D x) q) (outCol LANES HH) 64 =
        capNode (capRow c' w' (capFirst (Claims.Recursion.xI D x) q +
          (posAt (Claims.Recursion.xI D x) (coinsAt D x c) q >>> HH)))

/-- **The reader** at a side `sideAt`: at every reading of the last message, some RecCap part of the piece's level
holds the instance the opening's node is, and the side's cap entry there is the node its `top` row carries. -/
def SideCapR (sideAt : Msg → List Msg → OpenPiece → OpenSide) (pieces : Msg → List OpenPiece)
    (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  ∀ (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ)
    (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed) (q : OpenPiece),
    PieceAt H D shape untr b cmt elem pieces x c ws r ss s lv LANES HH n xO w p q →
    ∀ os last, bdR.final (((layR elem).msgs x ws).getD (bdR.DE.V.k (Claims.Recursion.xI D x) - 1) []) =
        some (os, last) → os.length + 1 = bdR.DE.V.k (Claims.Recursion.xI D x) →
      ∃ (s' N' : ℕ) (c' : Circuit) (w' : Claims.Flock.Wit), ss[s']? = some (.cap q.level N') ∧
        r.w.parts[s']?.bind partCircuit = some c' ∧ ws[s']? = some w' ∧
        capFirst (Claims.Recursion.xI D x) q + (posAt (Claims.Recursion.xI D x) (coinsAt D x c) q >>> HH) < N' ∧
        (FlockSoundness.Refine.capAt (sideAt (Claims.Recursion.xI D x) (bdR.opened os last) q).cap
            (sideAt (Claims.Recursion.xI D x) (bdR.opened os last) q).lig q.level).getD
          (posAt (Claims.Recursion.xI D x) (coinsAt D x c) q >>> HH) ByteArray.empty =
          capNode (capRow c' w' (capFirst (Claims.Recursion.xI D x) q +
            (posAt (Claims.Recursion.xI D x) (coinsAt D x c) q >>> HH)))

/-- **The cap tie** (`TopTie`) at a side `sideAt`, at every piece in `PieceAt`'s context. -/
def CapTopR (sideAt : Msg → List Msg → OpenPiece → OpenSide) (pieces : Msg → List OpenPiece)
    (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  ∀ (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ)
    (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed) (q : OpenPiece),
    PieceAt H D shape untr b cmt elem pieces x c ws r ss s lv LANES HH n xO w p q → ∀ ht : 0 < p.sts.size,
    ∀ os last, bdR.final (((layR elem).msgs x ws).getD (bdR.DE.V.k (Claims.Recursion.xI D x) - 1) []) =
        some (os, last) → os.length + 1 = bdR.DE.V.k (Claims.Recursion.xI D x) →
      TopTie (sideAt (Claims.Recursion.xI D x) (bdR.opened os last) q) q LANES HH
        (posAt (Claims.Recursion.xI D x) (coinsAt D x c) q) (copyR p ht w (Claims.Recursion.xI D x) q)

/-- **`OpenTie`'s first half at `JointNC`** (LOWOPEN's, the `dirs` tie). -/
def BitsTieR (sideAt : Msg → List Msg → OpenPiece → OpenSide) (pieces : Msg → List OpenPiece)
    (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  ∀ (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) (r : XR) (ss : Array VStar.Stmt) (s lv LANES HH n : ℕ)
    (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed) (q : OpenPiece),
    PieceAt H D shape untr b cmt elem pieces x c ws r ss s lv LANES HH n xO w p q → ∀ ht : 0 < p.sts.size,
    ∀ os last, bdR.final (((layR elem).msgs x ws).getD (bdR.DE.V.k (Claims.Recursion.xI D x) - 1) []) =
        some (os, last) → os.length + 1 = bdR.DE.V.k (Claims.Recursion.xI D x) →
      BitsTie (sideAt (Claims.Recursion.xI D x) (bdR.opened os last) q) q LANES HH
        (posAt (Claims.Recursion.xI D x) (coinsAt D x c) q) (copyR p ht w (Claims.Recursion.xI D x) q)

/-- **The cap tie from the binding and the reader**, wherever the level part's siblings below the cap, `HH`, are the
side's level's `d − c` (`OpenCopy`'s second conjunct). -/
theorem capTop_of (sideAt : Msg → List Msg → OpenPiece → OpenSide) (pieces : Msg → List OpenPiece)
    (posAt : Msg → List Coin → OpenPiece → ℕ) (hO : OutBindR H D shape untr b cmt elem pieces posAt)
    (hS : SideCapR H D shape untr b cmt elem sideAt pieces posAt)
    (hH : ∀ (x : Msg) (c : ℕ → Coin) (ws : List Claims.Flock.Wit) (r : XR) (ss : Array VStar.Stmt)
      (s lv LANES HH n : ℕ) (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed) (q : OpenPiece),
      PieceAt H D shape untr b cmt elem pieces x c ws r ss s lv LANES HH n xO w p q →
      ∀ os last, bdR.final (((layR elem).msgs x ws).getD (bdR.DE.V.k (Claims.Recursion.xI D x) - 1) []) =
          some (os, last) → os.length + 1 = bdR.DE.V.k (Claims.Recursion.xI D x) →
        HH = ((sideAt (Claims.Recursion.xI D x) (bdR.opened os last) q).sch.levels[q.level]!).d -
          ((sideAt (Claims.Recursion.xI D x) (bdR.opened os last) q).sch.levels[q.level]!).cap) :
    CapTopR H D shape untr b cmt elem sideAt pieces posAt := by
  intro x c ws r ss s lv LANES HH n xO w p q hP ht os last hf hl
  obtain ⟨s', N', c', w', hs', hc', hw', hlt, hcap⟩ := hS x c ws r ss s lv LANES HH n xO w p q hP os last hf hl
  unfold TopTie
  rw [← hH x c ws r ss s lv LANES HH n xO w p q hP os last hf hl, hcap]
  exact hO x c ws r ss s lv LANES HH n xO w p q hP ht s' N' c' w' hs' hc' hw' hlt

/-- **The cap tie at `sideAtR rdR`**, from the binding and the reader there, `HH` being the side's `d − c` by
`OpenCopyR`. -/
theorem capTop_sideAtR (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hC : OpenCopyR H D shape untr b cmt elem pieces) (hO : OutBindR H D shape untr b cmt elem pieces posAt)
    (hS : SideCapR H D shape untr b cmt elem (sideAtR rdR) pieces posAt) :
    CapTopR H D shape untr b cmt elem (sideAtR rdR) pieces posAt :=
  capTop_of H D shape untr b cmt elem (sideAtR rdR) pieces posAt hO hS
    fun x c ws r ss s lv LANES HH n xO w p q hP os last hf hl => by
      obtain ⟨hJ, hr, hss, hsv, hst, hq, ho⟩ := hP
      obtain ⟨_, _, _, _, hcp⟩ := hC x c ws hJ.1 r ss s lv LANES HH n hr hss hsv xO w p hst q hq ho
      exact (hcp os last hf hl).2.1

/-- **The `rec-open` pieces' share at the deployed layout, away from a joint collision**: LOWOPEN's
`openPieces_deployed` at `JointNC`, with `OpenTie` as its two halves, the cap tie (`CapTopR`) among them. -/
theorem openPieces_deployedNC (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hC : OpenCopyR H D shape untr b cmt elem pieces)
    (hB : BitsTieR H D shape untr b cmt elem (sideAtR rdR) pieces posAt)
    (hT : CapTopR H D shape untr b cmt elem (sideAtR rdR) pieces posAt)
    {x : Msg} {c : ℕ → Coin} {ws : List Claims.Flock.Wit}
    (hJ : JointNC H D (layR elem) (PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) (layR elem) shape
      (showsPart untr b) cmt) x c ws) {s : ℕ} (hs : s < D.nS x) :
    PiecesComputes bdR D pieces (openHolds (sideAtR rdR) posAt) ownQR s x c ((layR elem).msgs x ws) := by
  intro q hq ho
  have ho' := ho
  unfold ownQR at ho'
  split at ho'
  · obtain ⟨r, hr, ho'⟩ := Option.bind_eq_some_iff.1 ho'
    obtain ⟨ss, hss, hfi⟩ := Option.bind_eq_some_iff.1 ho'
    have hss' : _root_.Flock.RecVStar.statements r.shape = .ok ss := by
      cases h : _root_.Flock.RecVStar.statements r.shape with
      | error e => rw [h] at hss; cases hss
      | ok v => rw [h] at hss; cases hss; rfl
    obtain ⟨hsz, hlev, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hfi
    have hsv : ∃ lv LANES HH n, ss[s]? = some (.recOpen lv LANES HH n) := by
      rw [Array.getElem?_eq_getElem hsz]
      revert hlev
      cases ss[s] with
      | recOpen l L' H' n' => exact fun _ => ⟨l, L', H', n', rfl⟩
      | _ => intro h; simp [isLevel] at h
    obtain ⟨lv, LANES, HH, n, hsv⟩ := hsv
    refine openPieces_computesA (hJ.1.2.2 s hs) hr hss' (Or.inl hsv) bdR pieces ownQR (sideAtR rdR) posAt
      (fun xO w p hx hA hT' hc hsh q hq ho => ?_) q hq ho
    have hst : StagedAt H D untr cmt elem s x c ((layR elem).msgs x ws) xO w p := ⟨hx, hA, hT', hc, hsh⟩
    have hP : PieceAt H D shape untr b cmt elem pieces x c ws r ss s lv LANES HH n xO w p q :=
      ⟨hJ, hr, hss', hsv, hst, hq, ho⟩
    obtain ⟨ht, hob, hgb, hub, hcp⟩ := hC x c ws hJ.1 r ss s lv LANES HH n hr hss' hsv xO w p hst q hq ho
    exact ⟨0, ht, _, _, _, hob, hgb, hub, fun os last hf hl =>
      ⟨hcp os last hf hl, hB x c ws r ss s lv LANES HH n xO w p q hP ht os last hf hl,
        hT x c ws r ss s lv LANES HH n xO w p q hP ht os last hf hl⟩⟩
  · cases ho'

end Deployed

end RecDeploy.CapLower
