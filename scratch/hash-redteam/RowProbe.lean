import Definitions.Hash
import PouwCommit.Rows
import Lean.Data.Json

/-! hash-redteam scratch (never for a PR): #2474's `RowLeaf.accepts` / `Tree.accepts` at `H = G = SHA-512` (C-Flock's
`Flock.Sha512.hash`) against the served code, end to end.

* `cases.json` (`gen_cases.py`, the served Python's `audit.commit_rows`, `MerkleTree.open`, `verify_opening`): every rank of
  hm96-sha512 row trees of counts 0, 1, 2, 3, 5, 6, 7, 8, 9, rows of 0 … 3073 bytes. The model must accept every Python
  opening, its frame leaf must be the served Lean's (`RowsDomain.leaf`), and it must refuse the zero-padded row, a
  wrong rank, a short or long path, and (at rank 0 of each tree) a changed salt, a short salt and another key.
* `vectors_sha512.json`: `row_seg_leaves` (ℓ = 0, 1, 13, 8191, 8192, 8193, 24581, 65536 bits) and
  `bit_row_hiding_leaves` (hm96 `b ‖ c`, the hm96 leaf, `H(key)` and the frame leaf). -/

open Hash Lean

namespace RTProbe

def hexVal (c : Char) : Nat := if c.isDigit then c.toNat - 48 else c.toNat - 87

def unhexL : List Char → List UInt8
  | a :: b :: r => (16 * hexVal a + hexVal b).toUInt8 :: unhexL r
  | _ => []

def unhex (s : String) : Msg := unhexL s.toList

def ba (m : Msg) : ByteArray := ⟨m.toArray⟩

def toDigest (b : ByteArray) : Digest := fun i => b.get! i.val

def sha (m : Msg) : Digest := toDigest (Flock.Sha512.hash (ba m))

def bitsOf (bs : Msg) : Row := bs.flatMap fun b => (List.range 8).map fun s => b.toNat / 2 ^ s % 2 == 1

def str (j : Json) (k : String) : String := (j.getObjValAs? String k).toOption.getD ""
def nat (j : Json) (k : String) : Nat := (j.getObjValAs? Nat k).toOption.getD 0
def arr (j : Json) (k : String) : Array Json := ((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]
def hexes (j : Json) (k : String) : List Msg := (arr j k).toList.map fun x => unhex (x.getStr?.toOption.getD "")

def logT (s : String) : IO Unit := do
  let h ← IO.FS.Handle.mk "progress.txt" .append
  h.putStrLn s!"{← IO.monoMsNow} {s}"
  h.flush

#eval logT "elaborated helpers"

structure Tally where
  pass : Nat := 0
  fail : Nat := 0

def chk (r : IO.Ref Tally) (name : String) (ok : Bool) : IO Unit := do
  r.modify fun t => if ok then { t with pass := t.pass + 1 } else { t with fail := t.fail + 1 }
  unless ok do IO.println s!"FAIL {name}"

/-! `Hm96.commitString` reads `mask key y` once per byte, and the interpreter recomputes it each time (a dense salt's
mask is ~10⁸ steps). The `…With` forms take the mask as an argument; each is the model's definition with
`m := Hm96.mask key y` (the `…_eq` theorems), so evaluating them at that `m` evaluates the model. -/

def csWith (H : Msg → Digest) (x : Digest) (y m : Msg) : Msg :=
  (List.ofFn fun i : Fin 64 => x i ^^^ m.getD i 0) ++ (H (Hm96.saltPrefix ++ y)).bytes

theorem cs_eq (H : Msg → Digest) (key : Msg) (x : Digest) (y : Msg) :
    Hm96.commitString H key x y = csWith H x y (Hm96.mask key y) := by
  unfold Hm96.commitString csWith <;> rfl

#eval logT "elaborated cs_eq"

def hm96LeafWith (H : Msg → Digest) (key : Msg) (x : Digest) (y m : Msg) : Digest :=
  H (Hm96.leafPrefix H key ++ csWith H x y m)

theorem hm96Leaf_eq (H : Msg → Digest) (key : Msg) (x : Digest) (y : Msg) :
    Hm96.leaf H key x y = hm96LeafWith H key x y (Hm96.mask key y) := by
  unfold Hm96.leaf hm96LeafWith <;> rw [cs_eq]

def leafMsgWith (H G : Msg → Digest) (V : RowLeaf.Verifier) (rank : ℕ) (o : RowLeaf.Opening) (m : Msg) : Msg :=
  Tree.leafMsg V.domain rank RowLeaf.SCHEMA (hm96LeafWith H V.key (RowSeg.digest G o.row) o.salt m).bytes

theorem leafMsg_eq (H G : Msg → Digest) (V : RowLeaf.Verifier) (rank : ℕ) (o : RowLeaf.Opening) :
    RowLeaf.leafMsg H G V rank o = leafMsgWith H G V rank o (Hm96.mask V.key o.salt) := by
  unfold RowLeaf.leafMsg leafMsgWith <;> rw [hm96Leaf_eq]

def acceptsWith (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o : RowLeaf.Opening) (m : Msg) : Prop :=
  o.row.length < 2 ^ 64 ∧ o.salt.length = Hm96.SALT_BYTES ∧
    Tree.accepts sha V.domain root rank (sha (leafMsgWith sha sha V rank o m)) o.sibs

theorem accepts_iff (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o : RowLeaf.Opening) :
    RowLeaf.accepts sha sha V root rank o ↔ acceptsWith V root rank o (Hm96.mask V.key o.salt) := by
  unfold RowLeaf.accepts acceptsWith <;> rw [leafMsg_eq]

def outerHashedWith (H G : Msg → Digest) (V : RowLeaf.Verifier) (rank : ℕ) (o : RowLeaf.Opening) (m : Msg) : List Msg :=
  [Hm96.saltPrefix ++ o.salt, Hm96.leafPrefix H V.key ++ csWith H (RowSeg.digest G o.row) o.salt m] ++
    [leafMsgWith H G V rank o m] ++ Tree.hashed H V.domain rank (H (leafMsgWith H G V rank o m)) o.sibs

theorem outerHashed_eq (H G : Msg → Digest) (V : RowLeaf.Verifier) (rank : ℕ) (o : RowLeaf.Opening) :
    RowLeaf.outerHashed H G V rank o = outerHashedWith H G V rank o (Hm96.mask V.key o.salt) := by
  unfold RowLeaf.outerHashed outerHashedWith Hm96.hashed <;> rw [leafMsg_eq, cs_eq]

#eval logT "elaborated theorems"

instance (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o : RowLeaf.Opening) (m : Msg) :
    Decidable (acceptsWith V root rank o m) := by
  unfold acceptsWith Tree.accepts; infer_instance

/-- `decide (RowLeaf.accepts sha sha V root rank o)` when `m = Hm96.mask V.key o.salt` (`accepts_iff`). -/
def acc (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o : RowLeaf.Opening) (m : Msg) : Bool :=
  decide (acceptsWith V root rank o m)

/-- The same, computing the mask. -/
def acc' (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o : RowLeaf.Opening) : Bool :=
  acc V root rank o (Hm96.mask V.key o.salt)

def distinct (xs : List Msg) : Bool := xs.eraseDups.length == xs.length

def treeCase (r : IO.Ref Tally) (key : Msg) (t : Json) : IO Unit := do
  let count := nat t "count"
  let binding := ba (unhex (str t "binding"))
  let owner : Int := nat t "owner"
  let rows := hexes t "rows"
  let rootB := unhex (str t "root")
  let root := toDigest (ba rootB)
  let dom ← IO.ofExcept (PouwCommit.rowsDomain binding owner count "hm96-sha512" false)
  chk r s!"c={count} domain id: served Lean = Python" (dom.frame.id.data.toList == unhex (str t "domain_id"))
  let served := PouwCommit.commitRows binding owner (rows.map ba).toArray "hm96-sha512"
    (some (ba (unhex (str t "salt_key")))) (nat t "nonce") false
  chk r s!"c={count} root: served Lean = Python" (served.toOption.map (·.data.toList) == some rootB)
  let D : Tree.Domain := ⟨toDigest dom.frame.id, count, id⟩
  let V : RowLeaf.Verifier := ⟨D, key⟩
  if count == 0 then
    chk r "c=0 empty root = Python" (dom.frame.empty.data.toList == unhex (str t "empty_root") && rootB == unhex (str t "empty_root"))
    for rank in [0:3] do
      chk r s!"c=0 r={rank}: model refuses (empty tree)" (!acc' V root rank ⟨[], List.replicate 192 0, []⟩)
  for op in arr t "openings" do
    let rank := nat op "rank"
    let rowB := unhex (str op "row")
    let salt := unhex (str op "salt")
    let sibs := (hexes op "path").map fun p => toDigest (ba p)
    let o : RowLeaf.Opening := ⟨bitsOf rowB, salt, sibs⟩
    let tag := s!"c={count} r={rank} len={rowB.length}"
    logT s!"{tag} start"
    -- the mask depends on the key and salt only, so row, rank and path variants share it
    let m := Hm96.mask key salt
    chk r s!"{tag}: model accepts the Python opening" (acc V root rank o m)
    chk r s!"{tag}: served Lean verify accepts it"
      ((dom.verify (ba rootB) rank "verity/pouw/row/v1" (ba (rowB ++ salt)) ((hexes op "path").map ba)).toOption == some true)
    let servedLeaf := (dom.leaf rank "verity/pouw/row/v1" (ba (rowB ++ salt))).toOption.map (·.data.toList)
    chk r s!"{tag}: model frame leaf = served Lean leaf" (some (sha (leafMsgWith sha sha V rank o m)).bytes == servedLeaf)
    chk r s!"{tag}: model row-seg digest = served segDigest"
      ((RowSeg.digest sha o.row).bytes == (Flock.HmRow.segDigest (8 * rowB.length) (ba rowB)).data.toList)
    chk r s!"{tag}: refuses the row padded by one zero byte" (!acc V root rank { o with row := o.row ++ List.replicate 8 false } m)
    chk r s!"{tag}: refuses the row padded by one zero bit" (!acc V root rank { o with row := o.row ++ [false] } m)
    if rowB.length > 0 then
      chk r s!"{tag}: refuses the row less its last bit" (!acc V root rank { o with row := o.row.dropLast } m)
    if rank == 0 then
      chk r s!"{tag}: refuses a changed salt" (!acc' V root rank { o with salt := (salt.head! ^^^ 1) :: salt.tail })
      chk r s!"{tag}: refuses a 191-byte salt" (!acc' V root rank { o with salt := salt.dropLast })
      chk r s!"{tag}: refuses another key" (!acc' ⟨D, (key.head! ^^^ 1) :: key.tail⟩ root rank o)
    chk r s!"{tag}: refuses rank+1" (!acc V root (rank + 1) o m)
    if sibs.length > 0 then
      chk r s!"{tag}: refuses a short path" (!acc V root rank { o with sibs := sibs.dropLast } m)
    chk r s!"{tag}: refuses a long path" (!acc V root rank { o with sibs := sibs ++ [root] } m)
    let outer := outerHashedWith sha sha V rank o m
    let inner := RowLeaf.innerHashed sha o
    chk r s!"{tag}: |outerHashed| = 3 + depth" (outer.length == 3 + Tree.depth count)
    chk r s!"{tag}: |innerHashed| = s + 1" (inner.length == RowSeg.segCount (8 * rowB.length) + 1)
    chk r s!"{tag}: hashed messages distinct, digests distinct"
      (distinct outer && distinct inner && distinct (outer.map fun m => (sha m).bytes) &&
        distinct (inner.map fun m => (sha m).bytes))

def segCase (r : IO.Ref Tally) (e : Json) : IO Unit := do
  if (e.getObjVal? "row_bytes").toOption.isNone then return
  let n := nat e "n_bits"
  let bytes := unhex (str e "row_bytes")
  let row := (bitsOf bytes).take n
  let tag := s!"row-seg {str e "name"}"
  chk r s!"{tag}: packBits = vector bytes" (RowSeg.packBits row == bytes)
  chk r s!"{tag}: Q(ℓ) = vector prefix" (RowSeg.outerPrefix n == unhex (str e "prefix"))
  chk r s!"{tag}: digest = vector" ((RowSeg.digest sha row).bytes == unhex (str e "digest"))
  let segs := arr e "segments"
  chk r s!"{tag}: segment count" (segs.size == RowSeg.segCount n)
  for j in [0:segs.size] do
    let sj := segs[j]!
    chk r s!"{tag} seg {j}: m_j" ((RowSeg.segment row j).length == nat sj "n_bits")
    chk r s!"{tag} seg {j}: S(j, s, m) = vector prefix"
      (RowSeg.segmentPrefix j segs.size (RowSeg.segment row j).length == unhex (str sj "prefix"))
    chk r s!"{tag} seg {j}: d_j = vector" ((sha (RowSeg.segmentMsg row j)).bytes == unhex (str sj "digest"))

def hidingCase (r : IO.Ref Tally) (key : Msg) (e : Json) : IO Unit := do
  let tag := s!"hiding {str e "name"}"
  let x := toDigest (ba (unhex (str e "digest")))
  let y := unhex (str e "salt")
  let dj := (e.getObjVal? "domain").toOption.getD Json.null
  let rank := nat e "rank"
  let D : Tree.Domain := ⟨toDigest (ba (unhex (str dj "domain_id"))), rank + 1, id⟩
  chk r s!"{tag}: H(key) = vector" ((sha key).bytes == unhex (str e "key_sha512"))
  let m := Hm96.mask key y
  chk r s!"{tag}: b ‖ c = vector" (csWith sha x y m == unhex (str e "bc"))
  let hl := hm96LeafWith sha key x y m
  chk r s!"{tag}: hm96 leaf = vector" (hl.bytes == unhex (str e "hm96_leaf"))
  chk r s!"{tag}: frame leaf = vector" ((sha (Tree.leafMsg D rank (str e "schema") hl.bytes)).bytes == unhex (str e "leaf"))

def main (cases vectors : String) : IO Unit := do
  logT "main start"
  let r ← IO.mkRef ({} : Tally)
  let c ← IO.ofExcept (Json.parse (← IO.FS.readFile cases))
  let key := unhex (str c "key")
  chk r "model key bytes = served Lean Default512.key" (key == Flock.Hm96.Default512.key.data.toList)
  let v ← IO.ofExcept (Json.parse (← IO.FS.readFile vectors))
  for e in arr v "row_seg_leaves" do segCase r e
  logT s!"row-seg vectors done: {(← r.get).pass} passed, {(← r.get).fail} failed"
  for e in arr v "bit_row_hiding_leaves" do hidingCase r key e
  logT s!"hiding vectors done: {(← r.get).pass} passed, {(← r.get).fail} failed"
  for t in arr c "trees" do
    treeCase r key t
    logT s!"tree count={nat t "count"} done: {(← r.get).pass} passed, {(← r.get).fail} failed"
  let t ← r.get
  logT s!"RowProbe: {t.pass} checks passed, {t.fail} failed"
  IO.println s!"RowProbe: {t.pass} checks passed, {t.fail} failed"

end RTProbe

#eval RTProbe.main "cases.json" "../../verity/ml/commitments/frame_v3/vectors_sha512.json"
