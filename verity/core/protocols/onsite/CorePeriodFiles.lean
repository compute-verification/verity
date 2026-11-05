/-
The command line of a composite with the network certifier's period (onsite, remote with the network certifier, with
PoUW or without), whose `check` also takes the circuit files the period lists by path:

    check PARAMS COMMITMENT EVIDENCE [BLOB ...] --circuits CIRCUIT ... [--read-gb N]

reads `CIRCUIT ...`, in the order of the period's listing (`network-period check`'s), with
`NetworkCheck.PeriodFiles.read` on the network certifier's own parameters, the `verity/certifier/v0` entry of the
composite's `{"components":[{"id":ID,"params":PARAMS},...]}` (`certifierParams`), at most `N` GB of the audits' files at
once (by default `NetworkCheck.PeriodFiles.Budget.period`'s 48 GB, 3 files at most either way). It then checks the
composite with `NetworkCheck.Period.readProtocol` on those files as the network certifier's period, on the blobs, which
it reads only after the circuit files. A file that can't be read rejects the network certifier's period, naming the
file by its place in the listing. Without `--circuits` the command is the verifier of record's (`BlobProtocol.main`),
with the period's listed files among the blobs (`NetworkCheck.Period.blobProtocol`); where the blobs hold those files,
the two give one verdict (`NetworkCheck.Period.blobProtocol_check`).
-/
import CoreComposition.Blobs
import NetworkCheck.PeriodBlobs
import NetworkCheck.PeriodFiles

namespace Core.PeriodFiles

open Lean (Json)

/-- The network certifier's parameters in the composite's parameters `j`, `null` where they name none. -/
def certifierParams (j : Json) : Json :=
  match j.getObjVal? "components" with
  | .ok (.arr cs) =>
    let named (c : Json) : Bool := match c.getObjValAs? String "id" with
      | .ok i => i == NetworkCheck.Period.protocolId
      | .error _ => false
    match cs.find? named with
    | some c => (c.getObjVal? "params").toOption.getD .null
    | none => .null
  | _ => .null

/-- `args` without `--read-gb N`, and `N`, or why not. -/
def readGb : List String → Except String (List String × Option Nat)
  | [] => .ok ([], none)
  | ["--read-gb"] => .error "--read-gb needs a number of GB"
  | "--read-gb" :: n :: rest => do
    let some g := n.toNat? | throw s!"--read-gb {n} is not a whole number of GB"
    let (r, none) ← readGb rest | throw "--read-gb is given twice"
    return (r, some g)
  | a :: rest => do
    let (r, g) ← readGb rest
    return (a :: r, g)

/-- The budget of `--read-gb`'s `gb`, `NetworkCheck.PeriodFiles.Budget.period` without it. -/
def budget : Option Nat → NetworkCheck.PeriodFiles.Budget
  | some gb => ⟨NetworkCheck.PeriodFiles.Budget.period.files, gb * 10 ^ 9⟩
  | none => .period

/-- The command line of the composite `protocolWith net`, `net` its network certifier's period: the verifier of record
`protocolWith NetworkCheck.Period.blobProtocol`'s, and `check ... --circuits CIRCUIT ...`'s with the period's files
read by path. -/
def main (protocolWith : BlobProtocol → BlobProtocol) (args : List String) : IO UInt32 := do
  let P := protocolWith NetworkCheck.Period.blobProtocol
  let usage : IO UInt32 := do
    IO.eprintln s!"usage: {P.id} id | check PARAMS COMMITMENT EVIDENCE [BLOB ...] [--circuits CIRCUIT ... [--read-gb N]] | serve"
    return 64
  match args with
  | "check" :: p :: c :: e :: rest =>
    match readGb rest with
    | .error why =>
      IO.eprintln s!"{P.id}: {why}"
      usage
    | .ok (rest, gb) =>
      let blobs := rest.takeWhile (· != "--circuits")
      match rest.dropWhile (· != "--circuits"), gb with
      | [], none => P.main ("check" :: p :: c :: e :: blobs)
      | [], some _ =>
        IO.eprintln s!"{P.id}: --read-gb is the budget of the files after --circuits"
        usage
      | _ :: files, _ =>
        if files.contains "--circuits" then
          IO.eprintln s!"{P.id}: --circuits is given twice"
          usage
        else
          let j ← NetworkCheck.Period.paramsAt p
          let fs ← NetworkCheck.PeriodFiles.read (budget gb) (certifierParams j) files
          (protocolWith (NetworkCheck.Period.readProtocol fs).withoutBlobs).main ("check" :: p :: c :: e :: blobs)
  | ["id"] | ["serve"] => P.main args
  | _ => usage

#guard (readGb ["b", "--circuits", "f", "--read-gb", "40"]).toOption == some (["b", "--circuits", "f"], some 40)
#guard (readGb ["--read-gb", "40", "b"]).toOption == some (["b"], some 40)
#guard (readGb ["b", "--circuits", "f"]).toOption == some (["b", "--circuits", "f"], none)
#guard (readGb ["--read-gb"]).toOption == none
#guard (readGb ["--read-gb", "x"]).toOption == none
#guard (readGb ["--read-gb", "1", "--read-gb", "2"]).toOption == none
#guard (budget (some 40)).bytes == 40 * 10 ^ 9 && (budget (some 40)).files == 3
#guard (budget none).bytes == 48 * 10 ^ 9

private def scans : Json := Json.mkObj [("scan_circuits", Json.arr #[Json.str "a", Json.str "b"])]

private def certifierEntry : Json :=
  Json.mkObj [("id", Json.str NetworkCheck.Period.protocolId), ("params", scans)]

#guard certifierParams (Json.mkObj [("components",
  Json.arr #[Json.mkObj [("id", Json.str "verity/inference-only/v0"), ("params", Json.mkObj [])], certifierEntry])]) ==
  scans
#guard NetworkCheck.Period.scanCount (certifierParams (Json.mkObj [("components", Json.arr #[certifierEntry])])) == 2
#guard certifierParams (Json.mkObj [("components", Json.arr #[])]) == .null
#guard certifierParams .null == .null

end Core.PeriodFiles
