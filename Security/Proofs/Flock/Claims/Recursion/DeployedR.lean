import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.Claims.Recursion.BoundWeightDeploy
import Proofs.Flock.Claims.Recursion.BindsDeployed

/-!
# The deployed parameters of the bound record at `nextR` (U8)

The bound record at the recursive verifier, `bdDeployed rel (nextR rdR) cols wER joinR rowInDeployed finalR wR`, at
V\*'s bound layout `layoutBTops (finOpen (restR elem))`, fixes every choice the code side makes:

* **The weights** (ASK-72): `DE`'s is the deployed plain record's (`wER`), `γ`'s at `IVStar` its cover (`wR`,
  `wCover`). **The join** is `FlockDeploy.joinR`, the rows' digest input `rowInDeployed`.
* **The element-level last message** (`rdR`): tag `1`, the statement (length-prefixed), then each rep's proof's
  openings, the fields no frame reads (`readRep`'s `q`, all that `absorbedFrom` keeps of it): the initial level's,
  each recursive level's and the final level's opened rows, siblings and salts (`encOpens`), every list
  length-prefixed. `rdR` returns the proofs with those fields and every other field default; `readPs` takes every
  other field from the opened rounds. The record value is not in it: `decR` builds it (`recJ`).
* **V\*'s last message** (`finalR`): every earlier round's openings, row then salt, round by round in commit order;
  then the element-level last message; then a trailer the statement fixes (`trailer`): each opened row's byte length,
  round by round, then the trailer's own length. `finalR` reads the trailer from the end, then cuts the rest at it
  (`cutRounds`). Every byte `finalR` reads as a length is a constant of the statement, so it reads every witness.
* **The last round's pieces past the openings** (`restR elem`): the element-level last message's pieces (`elem`),
  then the trailer, constant bytes at the last part (`trailerPiece`). `elem` is the one choice still open: the
  pieces at the `rec-open` level parts' hidden rows and siblings. `ElemFramed elem` is the fact it is to meet: its
  pieces read as `encLast` of the statement and some openings at every witness.
-/

namespace RecDeploy.DeployedR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP Rd rdTake rdNat rdBytes rdList encNat encBytes encList)
open Claims.Recursion.VStarB (pieceLen readSegs)
open Lean (Json)

/-! ## The weights and the join -/

/-- **`DE`'s weight** (ASK-72): the deployed plain record's. -/
noncomputable def wER : Msg → ℝ := (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).weight

/-- **`γ`'s weight at `IVStar`** (ASK-72): `wCover` of the deployed plain record's. -/
noncomputable def wR : Msg → ℝ := wCover _root_.Flock.Readable.relation

/-! ## The element-level last message (`rdR`) -/

/-- An `F128`'s 16 bytes, `le64(lo) ‖ le64(hi)`. -/
def encF (a : F128) : Msg := a.toBytes.data.toList

/-- Little-endian bytes as a number. -/
def leNat (l : List UInt8) : ℕ := l.foldr (fun b acc => b.toNat + 256 * acc) 0

/-- An `F128` off 16 bytes: any 16 bytes read. -/
def rdF : Rd F128 := do
  let b ← rdTake 16
  pure ⟨(leNat (b.take 8)).toUInt64, (leNat (b.drop 8)).toUInt64⟩

/-- **A level's openings**: its opened rows, its siblings and its salts, each list length-prefixed. -/
def encLevel (rows : Array (Array F128)) (sibs salts : Array ByteArray) : Msg :=
  encList (encList encF) (rows.toList.map Array.toList) ++ encList encBytes sibs.toList ++
    encList encBytes salts.toList

def rdLevel : Rd (Array (Array F128) × Array ByteArray × Array ByteArray) := do
  let rows ← rdList (rdList rdF)
  let sibs ← rdList rdBytes
  let salts ← rdList rdBytes
  pure ((rows.map List.toArray).toArray, sibs.toArray, salts.toArray)

/-- **A proof's openings**: its initial level's, each recursive level's, then its final level's. -/
def encOpens (p : Flock.Proof) : Msg :=
  encLevel p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts ++
    encList (fun q : LevelProof => encLevel q.rows q.siblings q.salts) p.ligerito.recursiveProofs.toList ++
    encLevel p.ligerito.finalProof.rows p.ligerito.finalProof.siblings p.ligerito.finalProof.salts

/-- **A proof with only its openings**: every other field default. -/
def opensOnly (ip : LevelProof) (rps : Array LevelProof) (fr : Array (Array F128)) (fs ft : Array ByteArray) :
    Flock.Proof :=
  { (default : Flock.Proof) with
    ligerito := { (default : LigeritoProof) with
      initialProof := ip, recursiveProofs := rps, finalProof := { yr := #[], rows := fr, siblings := fs, salts := ft } } }

/-- A proof's openings, read: `opensOnly` at them. -/
def rdOpens : Rd Flock.Proof := do
  let i ← rdLevel
  let rps ← rdList (do let l ← rdLevel; pure ({ rows := l.1, siblings := l.2.1, salts := l.2.2 } : LevelProof))
  let f ← rdLevel
  pure (opensOnly { rows := i.1, siblings := i.2.1, salts := i.2.2 } rps.toArray f.1 f.2.1 f.2.2)

/-- What of a proof the element-level last message carries: `opensOnly` at its openings. -/
def opensOf (p : Flock.Proof) : Flock.Proof :=
  opensOnly p.ligerito.initialProof p.ligerito.recursiveProofs p.ligerito.finalProof.rows
    p.ligerito.finalProof.siblings p.ligerito.finalProof.salts

/-- **The element-level last message**: tag `1`, the statement, then each proof's openings. -/
def encLast (stmt : Msg) (ps : List Flock.Proof) : Msg :=
  1 :: (encNat stmt.length ++ stmt ++ encList encOpens ps)

def rdBody : Rd (Msg × List Flock.Proof) := do
  let n ← rdNat
  let stmt ← rdTake n
  let ps ← rdList rdOpens
  pure (stmt, ps)

/-- **The deployed reader of `nextR`'s last message**: `encLast`'s statement and openings, read whole; no record
value (`decR` builds it). -/
def rdR : Msg → Option FinalP
  | 1 :: body =>
    match rdBody.run body with
    | some ((stmt, ps), []) => some ⟨stmt, Json.null, ps.toArray⟩
    | _ => none
  | _ => none

/-! ## V\*'s last message (`finalR`) -/

/-- A salt off its bytes. -/
def saltOf (m : Msg) : Salt := fun i => m.getD i.val 0

/-- **One round's openings**, at its rows' byte lengths: each row, then its salt. -/
def cutRound : List ℕ → Msg → Option (List (Msg × Salt) × Msg)
  | [], m => some ([], m)
  | ℓ :: ls, m =>
    if ℓ + Hm96.sha512.saltBytes ≤ m.length then
      (cutRound ls ((m.drop ℓ).drop Hm96.sha512.saltBytes)).map fun p =>
        ((m.take ℓ, saltOf ((m.drop ℓ).take Hm96.sha512.saltBytes)) :: p.1, p.2)
    else none

/-- **Every round's openings**, round by round; what follows is the element-level last message. -/
def cutRounds : List (List ℕ) → Msg → Option (List (List (Msg × Salt)) × Msg)
  | [], m => some ([], m)
  | l :: ls, m => (cutRound l m).bind fun p => (cutRounds ls p.2).map fun q => (p.1 :: q.1, q.2)

/-- **The trailer**: each opened row's byte length, round by round, then the trailer's own length. -/
def trailer (lens : List (List ℕ)) : Msg :=
  encList (encList encNat) lens ++ encNat (encList (encList encNat) lens).length

/-- **The deployed reader of V\*'s last message**: the trailer off its end, then the openings at its lengths, then
the element-level last message. -/
def finalR (m : Msg) : Option (List (List (Msg × Salt)) × Msg) :=
  if m.length < 8 then none
  else
    match rdNat.run (m.drop (m.length - 8)) with
    | some (n, []) =>
      if n + 8 ≤ m.length then
        match (rdList (rdList rdNat)).run ((m.take (m.length - 8)).drop (m.length - 8 - n)) with
        | some (lens, []) => cutRounds lens (m.take (m.length - 8 - n))
        | _ => none
      else none
    | _ => none

/-! ## The last round's pieces past the openings (`restR`) -/

/-- The row lengths of a list of openings (`openSegs`' row, then salt), in bytes. -/
def rowLens : List (Msg × List HmNets.Lin) → List ℕ
  | row :: _ :: t => pieceLen [row] :: rowLens t
  | _ => []

/-- **Each earlier round's opened rows' byte lengths**, as `openPieces` lays them: round 0's at RecCap, then each
stream round's, its tops' then its row's. -/
def openLens (r : XR) : List (List ℕ) :=
  (capAt r).toList.map (fun p => rowLens (capOpen r p.1 (List.range p.2))) ++
    ((repSched r.shape).map fun rs =>
      (List.range (2 * rs.size)).map fun j => (roundOpen r rs j).flatMap fun p => rowLens p.2).getD []

/-- **The trailer's piece**: its constant bytes, at the last part. -/
def trailerPiece (r : XR) : List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)) :=
  match h : r.w.parts.length with
  | 0 => []
  | n + 1 => [(⟨n, by omega⟩, [(trailer (openLens r), [])])]

/-- **The last round's pieces past the openings**: the element-level last message's (`elem`), then the trailer. -/
def restR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (r : XR) :
    List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)) :=
  elem r ++ trailerPiece r

/-- Pieces read off the witnesses, as `LayoutB.assembled` reads them. -/
def readPieces (ps : List (ℕ × List (Msg × List HmNets.Lin))) (ws : List Claims.Flock.Wit) : Msg :=
  ps.flatMap fun p => (ws[p.1]?.map fun w => readSegs w p.2).getD (List.replicate (pieceLen p.2) 0)

/-- **What `elem` is to meet**: at every compiled statement and every witnesses, its pieces read as `encLast` of the
inner statement and some proofs, two a table where the inner statement sets up. -/
def ElemFramed (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x r, readX x = some r → ∀ ws : List Claims.Flock.Wit, ws.length = r.w.parts.length →
    ∃ ps : List Flock.Proof, (∀ v, FlockDeploy.setupOf false (xIOf x) = some v → ps.length = 2 * v.2.2.size) ∧
      readPieces ((elem r).map fun p => (p.1.val, p.2)) ws = encLast (xIOf x) ps

/-- The deployed bound record at `nextR rdR`. -/
noncomputable abbrev bdR : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rdR)
    (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wER FlockDeploy.joinR rowInDeployed finalR wR

/-- V\*'s deployed bound layout, at `elem`. -/
abbrev layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    Claims.Recursion.VStarB.LayoutB :=
  layoutBTops (finOpen (restR elem))

/-! ## The readers read what they're laid at -/

theorem leNat_append (a b : List UInt8) : leNat (a ++ b) = leNat a + 256 ^ a.length * leNat b := by
  induction a with
  | nil => simp [leNat]
  | cons x a ih =>
    simp only [leNat, List.cons_append, List.foldr_cons, List.length_cons] at ih ⊢
    rw [ih, pow_succ]
    ring

/-- **`finalR` reads its trailer**: at any body `B`, the openings at the trailer's lengths and what follows. -/
theorem finalR_trailer (B : Msg) (lens : List (List ℕ)) (hl : lens.length < 2 ^ 64)
    (hl' : ∀ l ∈ lens, l.length < 2 ^ 64 ∧ ∀ n ∈ l, n < 2 ^ 64)
    (hh : (encList (encList encNat) lens).length < 2 ^ 64) :
    finalR (B ++ trailer lens) = cutRounds lens B := by
  have hrd : (rdList (rdList rdNat)).run (encList (encList encNat) lens) = some (lens, []) := by
    simpa using FlockComplete.rdList_encList (r := rdList rdNat) (e := encList encNat) (l := lens)
      (fun l hm rest => FlockComplete.rdList_encList
        (fun n hn rest => FlockComplete.rdNat_encNat ((hl' l hm).2 n hn) rest) (hl' l hm).1 rest) hl []
  unfold trailer
  generalize encList (encList encNat) lens = h at hh hrd ⊢
  have hm : B ++ (h ++ encNat h.length) = (B ++ h) ++ encNat h.length := (List.append_assoc ..).symm
  have hlen : ((B ++ h) ++ encNat h.length).length = (B ++ h).length + 8 := by
    rw [List.length_append, FlockComplete.encNat_length]
  have hn : rdNat.run (encNat h.length) = some (h.length, []) := by
    simpa using FlockComplete.rdNat_encNat hh []
  have hBh : (B ++ h).length = B.length + h.length := List.length_append
  rw [hm]
  unfold finalR
  rw [hlen, if_neg (by omega), Nat.add_sub_cancel, List.drop_left, hn]
  simp only
  rw [if_pos (by omega), List.take_left, List.drop_left' (by omega : B.length = (B ++ h).length - h.length), hrd]
  simp only
  rw [List.append_assoc, List.take_left' (by omega : B.length = (B ++ h).length - h.length)]

end RecDeploy.DeployedR
