import Definitions.Hash
import PouwCommit.Rows
import Lean.Data.Json

/-! hash-redteam scratch (never for a PR): #2474's `RowLeaf.accepts` / `Tree.accepts` at `H = G = SHA-512` (C-Flock's
`Flock.Sha512.hash`) against the served code, end to end.

* `cases.json` (`gen_cases.py`, the served Python's `audit.commit_rows`, `MerkleTree.open`, `verify_opening`): every rank of
  hm96-sha512 row trees of counts 0, 1, 2, 3, 5, 6, 7, 8, 9, rows of 0 … 3073 bytes. The model must accept every Python
  opening, its frame leaf must be the served Lean's (`RowsDomain.leaf`), and it must refuse the zero-padded row, a
  wrong rank, a short path, a changed salt and another key.
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

structure Tally where
  pass : Nat := 0
  fail : Nat := 0

def chk (r : IO.Ref Tally) (name : String) (ok : Bool) : IO Unit := do
  r.modify fun t => if ok then { t with pass := t.pass + 1 } else { t with fail := t.fail + 1 }
  unless ok do IO.println s!"FAIL {name}"

instance (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o : RowLeaf.Opening) :
    Decidable (RowLeaf.accepts sha sha V root rank o) := by
  unfold RowLeaf.accepts Tree.accepts; infer_instance

def acc (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o : RowLeaf.Opening) : Bool :=
  decide (RowLeaf.accepts sha sha V root rank o)

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
      chk r s!"c=0 r={rank}: model refuses (empty tree)" (!acc V root rank ⟨[], List.replicate 192 0, []⟩)
  for op in arr t "openings" do
    let rank := nat op "rank"
    let rowB := unhex (str op "row")
    let salt := unhex (str op "salt")
    let sibs := (hexes op "path").map fun p => toDigest (ba p)
    let o : RowLeaf.Opening := ⟨bitsOf rowB, salt, sibs⟩
    let tag := s!"c={count} r={rank} len={rowB.length}"
    chk r s!"{tag}: model accepts the Python opening" (acc V root rank o)
    chk r s!"{tag}: served Lean verify accepts it"
      ((dom.verify (ba rootB) rank "verity/pouw/row/v1" (ba (rowB ++ salt)) ((hexes op "path").map ba)).toOption == some true)
    let servedLeaf := (dom.leaf rank "verity/pouw/row/v1" (ba (rowB ++ salt))).toOption.map (·.data.toList)
    chk r s!"{tag}: model frame leaf = served Lean leaf" (some (sha (RowLeaf.leafMsg sha sha V rank o)).bytes == servedLeaf)
    chk r s!"{tag}: model row-seg digest = served segDigest"
      ((RowSeg.digest sha o.row).bytes == (Flock.HmRow.segDigest (8 * rowB.length) (ba rowB)).data.toList)
    chk r s!"{tag}: refuses the row padded by one zero byte" (!acc V root rank { o with row := o.row ++ List.replicate 8 false })
    chk r s!"{tag}: refuses the row padded by one zero bit" (!acc V root rank { o with row := o.row ++ [false] })
    if rowB.length > 0 then
      chk r s!"{tag}: refuses the row less its last bit" (!acc V root rank { o with row := o.row.dropLast })
    chk r s!"{tag}: refuses a changed salt" (!acc V root rank { o with salt := (salt.head! ^^^ 1) :: salt.tail })
    chk r s!"{tag}: refuses a 191-byte salt" (!acc V root rank { o with salt := salt.dropLast })
    chk r s!"{tag}: refuses another key" (!acc ⟨D, (key.head! ^^^ 1) :: key.tail⟩ root rank o)
    chk r s!"{tag}: refuses rank+1" (!acc V root (rank + 1) o)
    if sibs.length > 0 then
      chk r s!"{tag}: refuses a short path" (!acc V root rank { o with sibs := sibs.dropLast })
    chk r s!"{tag}: refuses a long path" (!acc V root rank { o with sibs := sibs ++ [root] })
    let outer := RowLeaf.outerHashed sha sha V rank o
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
  chk r s!"{tag}: b ‖ c = vector" (Hm96.commitString sha key x y == unhex (str e "bc"))
  chk r s!"{tag}: hm96 leaf = vector" ((Hm96.leaf sha key x y).bytes == unhex (str e "hm96_leaf"))
  chk r s!"{tag}: frame leaf = vector"
    ((sha (Tree.leafMsg D rank (str e "schema") (Hm96.leaf sha key x y).bytes)).bytes == unhex (str e "leaf"))

def main (cases vectors : String) : IO Unit := do
  let r ← IO.mkRef ({} : Tally)
  let c ← IO.ofExcept (Json.parse (← IO.FS.readFile cases))
  let key := unhex (str c "key")
  chk r "model key bytes = served Lean Default512.key" (key == Flock.Hm96.Default512.key.data.toList)
  for t in arr c "trees" do treeCase r key t
  let v ← IO.ofExcept (Json.parse (← IO.FS.readFile vectors))
  for e in arr v "row_seg_leaves" do segCase r e
  for e in arr v "bit_row_hiding_leaves" do hidingCase r key e
  let t ← r.get
  IO.println s!"RowProbe: {t.pass} checks passed, {t.fail} failed"

end RTProbe

#eval RTProbe.main "cases.json" "../../verity/ml/commitments/frame_v3/vectors_sha512.json"
