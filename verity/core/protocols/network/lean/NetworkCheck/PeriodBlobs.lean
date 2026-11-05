import NetworkCheck.Period
import NetworkCheck.CircuitFile
import CoreProtocol.Blobs

/-! # The period's listed circuit files, among the blobs or read by path

A composite checks the network certifier's period with the circuit files its parameters list, each audit's and then
the scan's (`NetworkCheck.Period.Listing.files`). Each file is read whole, as a session's verifier parses it
(`NetworkCheck.CircuitFile.whole`); an audit's is held trimmed (`NetworkCheck.Period.trim`) and the scan's whole
(`ofFile`, `ofFiles`). The files are then checked with the period (`NetworkCheck.Period.checkWith`), as
`network-period check` checks them, but a file that is a pinned lookup table is refused (`pinnedTableOf`), since the
composite's other subprotocols look those up among the blobs. A period that lists no file is checked as
`NetworkCheck.Period.protocol` checks it (`filesCheck`). The check has two forms:

* `blobProtocol` is the composites' verifier of record. Each listed file is the blob with its listed SHA-512
  (`Core.Blobs.get`, `filesOf`).
* `readProtocol fs` is what a composite's executable runs when a period's files are too large to hold as blobs. The
  files are read by path within a budget (`NetworkCheck.PeriodFiles.read`), each as `ofFile` reads its bytes: the last
  `NetworkCheck.Period.scanCount` of them, the scan's, whole, and the others trimmed.

Where the blobs hold the listed files' bytes, the two give one verdict (`blobProtocol_check`).
-/

namespace NetworkCheck.Period

open Flock (V)

/-- **A listed file's bytes `b` as the period holds them**: read whole (`NetworkCheck.CircuitFile.whole`), its SHA-512
and circuit, with the circuit of an audit's file trimmed (`trim`) and the scan's (`scan`) kept whole, as
`NetworkCheck.PeriodFiles.read` holds the file at a path. -/
def ofFile (scan : Bool) (b : ByteArray) : V (ByteArray × Flock.Circuit) :=
  (NetworkCheck.CircuitFile.whole b).map fun (sha, cf) => (sha, if scan then cf else trim cf)

/-- **The files `bs` of listing `l`**, in its order (`Listing.files`), each as `ofFile` holds it; or the first that
doesn't parse, named by its place. -/
def ofFiles (l : Listing) (bs : Array ByteArray) : V (Array (ByteArray × Flock.Circuit)) :=
  (l.files.zip bs).zipIdx.mapM fun (((_, scan, _, _), b), i) => (ofFile scan b).mapError (s!"circuit file {i}: " ++ ·)

/-- **The files listing `l` lists, among the blobs `bs`**, in its order: each the blob whose SHA-512 is its listing's
(`Core.Blobs.get`); or the first that no blob has, named by its place. -/
def blobsOf (l : Listing) (bs : Core.Blobs) : V (Array ByteArray) :=
  l.files.zipIdx.mapM fun ((_, _, x, k), i) => bs.get x.files[k]! s!"circuit file {i}"

/-- **The listed files among the blobs `bs`**, each as `ofFile` holds it (`blobsOf`, `ofFiles`). -/
def filesOf (l : Listing) (bs : Core.Blobs) : V (Array (ByteArray × Flock.Circuit)) :=
  blobsOf l bs >>= ofFiles l

/-- **The first of the files `fs` that is a pinned lookup table** (`Flock.MUFU_TABLES`) by its SHA-512, its place and
the table's name, or none. A composite's other subprotocols look the pinned tables up among the blobs
(`SampledProofs.OneStageZk.tablesOf`), so a listed file that is one would reach them where the files are among the
blobs, and not where they are read by path. -/
def pinnedTableOf (fs : Array (ByteArray × Flock.Circuit)) : Option (Nat × String) :=
  fs.zipIdx.findSome? fun ((sha, _), i) =>
    (Flock.MUFU_TABLES.find? (·.sha512 == Flock.hexOf sha)).map fun t => (i, t.name)

/-- **The period's check with its listed files `fs`**, or why they couldn't be read. A file that is a pinned lookup
table is refused (`pinnedTableOf`). A period that lists no file, given none, is checked as `protocol` checks it
(`check`). Otherwise it is checked with them (`checkWith`), which refuses files that are not the listed ones, their
count among what it checks. -/
def filesCheck (fs : V (Array (ByteArray × Flock.Circuit))) (p : Params × Listing) (c : Commitment) (e : Evidence) :
    Core.Outcome Public Profile :=
  match fs with
  | .error why => .reject s!"circuit files: {why}"
  | .ok fs =>
    match pinnedTableOf fs with
    | some (i, t) => .reject s!"circuit files: circuit file {i} is the pinned lookup table {t}, by its SHA-512"
    | none => if !p.2.listed && fs.isEmpty then check p.1 c e else checkWith fs p c e

/-- **The period protocol with its listed files `fs`, read by path** (`NetworkCheck.PeriodFiles.read`), or with why they
couldn't be read: `filesCheck fs`. A composite's executable runs it as the network certifier's subprotocol when the files
are given by path. It reads no blobs. -/
def readProtocol (fs : V (Array (ByteArray × Flock.Circuit))) : Core.Protocol := protocolBy (filesCheck fs)

/-- **The period protocol with its listed files among the caller's blobs**, the composites' verifier of record: each
listed file is the blob with its listed SHA-512 (`filesOf`), and the check is `filesCheck`'s. Without blobs, a period
that lists no file is checked as `protocol` checks it. -/
def blobProtocol : Core.BlobProtocol :=
  { protocolBy fun p c e => filesCheck (filesOf p.2 .empty) p c e with
    checkBlobs := fun (p : Params × Listing) c e bs => filesCheck (filesOf p.2 bs) p c e }

/-- **Where the blobs hold the listed files, the verifier of record's verdict is the executable's**: where the blobs `bs`
hold each file the period lists, as the bytes `b` (`blobsOf`), `blobProtocol` checks the period as `readProtocol`
checks it with those bytes read as `NetworkCheck.PeriodFiles.read` reads the files at their paths (`ofFiles`). -/
theorem blobProtocol_check (p : Params × Listing) (c : Commitment) (e : Evidence) (bs : Core.Blobs)
    (b : Array ByteArray) (h : blobsOf p.2 bs = .ok b) :
    blobProtocol.checkBlobs p c e bs = (readProtocol (ofFiles p.2 b)).check p c e := by
  show filesCheck (blobsOf p.2 bs >>= ofFiles p.2) p c e = filesCheck (ofFiles p.2 b) p c e
  rw [h]
  rfl

/-- Both forms share the period protocol's id and decoders: only the check differs. -/
theorem blobProtocol_id : blobProtocol.id = protocolId := rfl

theorem readProtocol_id (fs : V (Array (ByteArray × Flock.Circuit))) : (readProtocol fs).id = protocolId := rfl

/- A period that lists nothing needs no blob; a listed file is found by its SHA-512, and one that isn't, or doesn't
parse, is named by its place. -/
#guard match filesOf {} .empty with
  | .ok fs => fs.isEmpty
  | .error _ => false
#guard match blobsOf { audits := #[some toyListed] } .empty with
  | .error _ => true
  | .ok _ => false
#guard
  let b := "not a circuit file".toUTF8
  let d := Flock.hexOf (Flock.Sha512.hash b)
  let l : Listing := { audits := #[none, some { toyListed with files := #[d, d] }] }
  (match blobsOf l (.ofArray #[b]) with
    | .ok bs => bs.map (·.data) == #[b.data, b.data]
    | .error _ => false) &&
  (match blobsOf l .empty with
    | .error e => e == s!"circuit file 0: no blob has the SHA-512 {d}"
    | .ok _ => false) &&
  (match filesOf l (.ofArray #[b]) with
    | .error e => e.startsWith "circuit file 0: "
    | .ok _ => false)

/- A file whose SHA-512 is a pinned lookup table's is named by its place and the table; other files are not. -/
#guard
  let other := (Flock.Sha512.hash "not a table".toUTF8, (default : Flock.Circuit))
  match Flock.MUFU_TABLES[1]? >>= fun t => (Flock.unhex t.sha512).toOption with
  | some d => pinnedTableOf #[other, (d, default)] == some (1, "sqrt") && pinnedTableOf #[other] == none
  | none => false

end NetworkCheck.Period
