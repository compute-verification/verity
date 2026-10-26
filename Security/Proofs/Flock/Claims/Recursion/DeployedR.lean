import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.Claims.Recursion.BoundWeightDeploy
import Proofs.Flock.Claims.Recursion.BindsDeployed
import Proofs.Flock.Claims.KS.PlainR

/-!
# The deployed parameters of the bound record at `nextR` (U8)

The bound record at the recursive verifier, `bdDeployed rel (nextR rdR) cols wER joinR rowInDeployed finalR wR`, at
V\*'s bound layout `layoutBTops (finOpen (restR elem))`, fixes every choice the code side makes:

* **The columns** are the reader's, the trimmed plain record's at `rdR` (`deployPlainTrimR rel rdR`), as
  `RecursiveKSAt` reads them: `bdR` is `RecDeploy.bdR rdR wER finalR wR`.

* **The weights** (ASK-72): `DE`'s is the deployed plain record's (`wER`), `γ`'s at `IVStar` its cover (`wR`,
  `wCover`). **The join** is `FlockDeploy.joinR`, the rows' digest input `rowInDeployed`.
* **The element-level last message** (`rdR`): tag `1`, the statement (length-prefixed), then each rep's proof's
  openings, the fields no frame reads (`readRep`'s `q`, all that `absorbedFrom` keeps of it): the initial level's,
  each recursive level's and the final level's opened rows, siblings and salts (`encOpens`), every list
  length-prefixed. `rdR` returns the proofs with those fields and every other field default; `readPs` takes every
  other field from the opened rounds. The record value is not in it: `decR` builds it (`recJ`).
* **V\*'s last message** (`finalR`): every earlier round's openings, row then salt, round by round in commit order;
  then the element-level last message; then a trailer the statement fixes (`trailer`): each opened row's byte length,
  round by round, then the trailer's own length, every number in variable-length bytes (`FlockDeploy.encV`, so no
  length needs to fit a word). `finalR` reads the trailer from the end, then cuts the rest at it (`cutRounds`). Every
  byte `finalR` reads as a length is a constant of the statement, so it reads every witness.
* **The last round's pieces past the openings** (`restR elem`): the element-level last message's pieces (`elem`),
  then the trailer, constant bytes at the last part (`trailerPiece`). `elem` is the one choice still open: the
  pieces at the `rec-open` level parts' hidden rows and siblings. `ElemFramed elem` is the fact it is to meet: its
  pieces read as `encLast` of the statement and some openings at every witness.
-/

namespace RecDeploy.DeployedR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP Rd rdTake rdNat rdBytes rdList encNat encBytes encList encV rdVL rdV encVL rdVLs rdVL_encV
  rdVLs_encVL)
open Claims.Recursion.VStarB (pieceLen readSegs)
open Lean (Json)

/-! ## The weights and the join -/

/-- **`DE`'s weight** (ASK-72): the deployed plain record's. -/
noncomputable def wER : Msg → ℝ := (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).weight

/-- **`γ`'s weight at `IVStar`** (ASK-72): `wCover` of the deployed plain record's. -/
noncomputable def wR : Msg → ℝ := wCover _root_.Flock.Readable.relation

/-! ## The element-level last message (`rdR`) -/

/-- An `F128`'s 16 bytes, `le64(lo) ‖ le64(hi)` (`F128.toBytes`' order). -/
def encF (a : F128) : Msg := encNat a.lo.toNat ++ encNat a.hi.toNat

/-- An `F128` off 16 bytes: any 16 bytes read. -/
def rdF : Rd F128 := do
  let lo ← rdNat
  let hi ← rdNat
  pure ⟨lo.toUInt64, hi.toUInt64⟩

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

/-- **The trailer**: each opened row's byte length, round by round, then the trailer's own length, its bytes
reversed so it reads off the message's end. -/
def trailer (lens : List (List ℕ)) : Msg :=
  encVL (encVL encV) lens ++ (encV (encVL (encVL encV) lens).length).reverse

/-- **The deployed reader of V\*'s last message**: the trailer off its end, then the openings at its lengths, then
the element-level last message. -/
def finalR (m : Msg) : Option (List (List (Msg × Salt)) × Msg) :=
  match rdVL m.reverse with
  | some (n, rest) =>
    if n ≤ rest.length then
      match (rdVLs (rdVLs rdV)).run (rest.take n).reverse with
      | some (lens, []) => cutRounds lens (rest.drop n).reverse
      | _ => none
    else none
  | none => none

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
  if h : 0 < r.w.parts.length then [(⟨r.w.parts.length - 1, by omega⟩, [(trailer (openLens r), [])])] else []

/-- **The last round's pieces past the openings**: the element-level last message's (`elem`), then the trailer. -/
def restR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (r : XR) :
    List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)) :=
  elem r ++ trailerPiece r

/-- Pieces read off the witnesses, as `LayoutB.assembled` reads them. -/
def readPieces (ps : List (ℕ × List (Msg × List HmNets.Lin))) (ws : List Claims.Flock.Wit) : Msg :=
  ps.flatMap fun p => (ws[p.1]?.map fun w => readSegs w p.2).getD (List.replicate (pieceLen p.2) 0)

/-- **What `elem` is to meet**: at every compiled statement and every witnesses, `rdR` reads its pieces, at the inner
statement, two proofs a table where the inner statement sets up. -/
def ElemFramed (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x r, readX x = some r → ∀ ws : List Claims.Flock.Wit, ws.length = r.w.parts.length →
    ∃ ps : Array Flock.Proof, (∀ v, FlockDeploy.setupOf false (xIOf x) = some v → ps.size = 2 * v.2.2.size) ∧
      rdR (readPieces ((elem r).map fun p => (p.1.val, p.2)) ws) = some ⟨xIOf x, Json.null, ps⟩

/-- The deployed bound record at `nextR rdR`. -/
noncomputable abbrev bdR : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rdR).cols wER FlockDeploy.joinR rowInDeployed finalR
    wR

/-- V\*'s deployed bound layout, at `elem`. -/
abbrev layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    Claims.Recursion.VStarB.LayoutB :=
  layoutBTops (finOpen (restR elem))

/-! ## The readers read what they're laid at -/

theorem rdF_encF (a : F128) (rest : Msg) : rdF.run (encF a ++ rest) = some (a, rest) := by
  simp only [rdF, encF, List.append_assoc, StateT.run_bind, FlockComplete.rdNat_encNat a.lo.toNat_lt,
    FlockComplete.rdNat_encNat a.hi.toNat_lt, Option.bind_eq_bind, Option.bind_some]
  show some ((⟨a.lo.toNat.toUInt64, a.hi.toNat.toUInt64⟩ : F128), rest) = some (a, rest)
  simp

/-- A level's lists and byte strings fit their length words. -/
def LevelFits (rows : Array (Array F128)) (sibs salts : Array ByteArray) : Prop :=
  rows.size < 2 ^ 64 ∧ (∀ r ∈ rows.toList, r.size < 2 ^ 64) ∧ sibs.size < 2 ^ 64 ∧ (∀ b ∈ sibs.toList, b.size < 2 ^ 64) ∧
    salts.size < 2 ^ 64 ∧ ∀ b ∈ salts.toList, b.size < 2 ^ 64

theorem rdBytes_list {l : List ByteArray} (hl : l.length < 2 ^ 64) (hb : ∀ b ∈ l, b.size < 2 ^ 64) (rest : Msg) :
    (rdList rdBytes).run (encList encBytes l ++ rest) = some (l, rest) :=
  FlockComplete.rdList_encList (fun b hm rest => FlockComplete.rdBytes_encBytes (hb b hm) rest) hl rest

theorem rdLevel_encLevel {rows : Array (Array F128)} {sibs salts : Array ByteArray} (h : LevelFits rows sibs salts)
    (rest : Msg) : rdLevel.run (encLevel rows sibs salts ++ rest) = some ((rows, sibs, salts), rest) := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  have e1 : ∀ rest, (rdList (rdList rdF)).run (encList (encList encF) (rows.toList.map Array.toList) ++ rest) =
      some (rows.toList.map Array.toList, rest) := fun rest =>
    FlockComplete.rdList_encList (fun l hm rest => by
        obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hm
        exact FlockComplete.rdList_encList (fun b _ rest => rdF_encF b rest) (by simpa using h2 a ha) rest)
      (by simpa using h1) rest
  simp only [rdLevel, encLevel, List.append_assoc, StateT.run_bind, e1,
    rdBytes_list (by simpa using h3) h4, rdBytes_list (by simpa using h5) h6, Option.bind_eq_bind, Option.bind_some]
  simp [StateT.run_pure, Option.pure_def, Function.comp_def]

/-- A proof's openings fit their length words. -/
def OpensFit (p : Flock.Proof) : Prop :=
  LevelFits p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts ∧
    p.ligerito.recursiveProofs.size < 2 ^ 64 ∧ (∀ q ∈ p.ligerito.recursiveProofs.toList, LevelFits q.rows q.siblings q.salts) ∧
    LevelFits p.ligerito.finalProof.rows p.ligerito.finalProof.siblings p.ligerito.finalProof.salts

theorem rdOpens_encOpens {p : Flock.Proof} (h : OpensFit p) (rest : Msg) :
    rdOpens.run (encOpens p ++ rest) = some (opensOf p, rest) := by
  obtain ⟨hi, hn, hr, hf⟩ := h
  have e2 : ∀ rest, (rdList (do let l ← rdLevel; pure ({ rows := l.1, siblings := l.2.1, salts := l.2.2 } : LevelProof))).run
      (encList (fun q : LevelProof => encLevel q.rows q.siblings q.salts) p.ligerito.recursiveProofs.toList ++ rest) =
      some (p.ligerito.recursiveProofs.toList, rest) := fun rest =>
    FlockComplete.rdList_encList (fun q hq rest => by
        simp only [StateT.run_bind, rdLevel_encLevel (hr q hq), Option.bind_eq_bind, Option.bind_some]
        rfl)
      (by simpa using hn) rest
  simp only [rdOpens, encOpens, List.append_assoc, StateT.run_bind, rdLevel_encLevel hi, e2, rdLevel_encLevel hf,
    Option.bind_eq_bind, Option.bind_some]
  simp only [StateT.run_pure, Option.pure_def]
  rfl

/-- **What a reader takes of the last message's proof is its openings**: `absorbedFrom` keeps of `q` only the fields
`opensOf` keeps. -/
theorem absorbedFrom_opensOf (c : FlockDeploy.RepCtx) (a q : Flock.Proof) :
    FlockDeploy.absorbedFrom c a (opensOf q) = FlockDeploy.absorbedFrom c a q := rfl

/-- **A rep reads the same off a proof's openings as off the proof** (so `ReadsBack` at `fl.ps = ps` carries over
to `rdR`'s `ps.map opensOf`). -/
theorem readRep_opensOf (c : FlockDeploy.RepCtx) (rows : List (List ByteArray)) (q : Flock.Proof) :
    FlockDeploy.readRep c rows (opensOf q) = FlockDeploy.readRep c rows q := by
  unfold FlockDeploy.readRep
  simp only [absorbedFrom_opensOf]

/-- **`rdR` reads `encLast`**: the statement, and each proof's openings (`opensOf`), where every length fits its
word. -/
theorem rdR_encLast {stmt : Msg} {ps : List Flock.Proof} (hs : stmt.length < 2 ^ 64) (hl : ps.length < 2 ^ 64)
    (hp : ∀ p ∈ ps, OpensFit p) :
    rdR (encLast stmt ps) = some ⟨stmt, Json.null, (ps.map opensOf).toArray⟩ := by
  have e3 : (rdList rdOpens).run (encList encOpens ps ++ []) = some (ps.map opensOf, []) := by
    simp only [rdList, encList, List.append_assoc, StateT.run_bind, FlockComplete.rdNat_encNat hl,
      Option.bind_eq_bind, Option.bind_some]
    have : ∀ (l : List Flock.Proof), (∀ p ∈ l, OpensFit p) → ∀ rest,
        (FlockDeploy.rdListN rdOpens l.length).run (l.flatMap encOpens ++ rest) = some (l.map opensOf, rest) := by
      intro l
      induction l with
      | nil => intro _ rest; rfl
      | cons a l ih =>
        intro h rest
        simp only [List.length_cons, FlockDeploy.rdListN, List.flatMap_cons, List.append_assoc, StateT.run_bind,
          rdOpens_encOpens (h a List.mem_cons_self), Option.bind_eq_bind, Option.bind_some,
          ih (fun p hp => h p (List.mem_cons_of_mem a hp)) rest]
        rfl
    exact this ps hp []
  have hb : rdBody.run (encNat stmt.length ++ stmt ++ encList encOpens ps) = some ((stmt, ps.map opensOf), []) := by
    have ht := FlockComplete.rdTake_append stmt (encList encOpens ps ++ [])
    simp only [List.append_nil] at ht e3
    simp only [rdBody, List.append_assoc, StateT.run_bind, FlockComplete.rdNat_encNat hs, ht, e3,
      Option.bind_eq_bind, Option.bind_some]
    rfl
  unfold rdR encLast
  simp only [hb]

/-- **`finalR` reads its trailer**: at any body `B`, the openings at the trailer's lengths and what follows. -/
theorem finalR_trailer (B : Msg) (lens : List (List ℕ)) : finalR (B ++ trailer lens) = cutRounds lens B := by
  have hrd : (rdVLs (rdVLs rdV)).run (encVL (encVL encV) lens) = some (lens, []) := by
    simpa using rdVLs_encVL (r := rdVLs rdV) (e := encVL encV) (l := lens)
      (fun l _ rest => rdVLs_encVL (fun n _ rest => rdVL_encV n rest) rest) []
  unfold trailer
  generalize encVL (encVL encV) lens = T at hrd ⊢
  have hrev : (B ++ (T ++ (encV T.length).reverse)).reverse = encV T.length ++ (T.reverse ++ B.reverse) := by
    simp [List.reverse_append]
  unfold finalR
  rw [hrev, rdVL_encV]
  simp only
  rw [if_pos (by simp), List.take_left' (by simp), List.reverse_reverse, hrd]
  simp only
  rw [List.drop_left' (by simp), List.reverse_reverse]

end RecDeploy.DeployedR
