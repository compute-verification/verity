import Proofs.Flock.Claims.Recursion.Deploy

/-!
# Every part holds the verifier's one record of V*'s shared rows (`rowsShared`)

The compiled statement carries `--rec-rows` per V* part (`PartX.rows`), and each part's session ties its rec-acc ports
only to its own rows' `acc0` (`RecVStar.checkAcc`). `readX` refuses a statement whose parts' rows differ
(`rowsShared`), so wherever it reads one, any two parts, and the recursive statements they are staged at (`partRaw`'s
`recRows`), hold the same rows (`readX_rows_eq`, `partRaw_recRows_eq`): two parts' sessions check their rec-acc ports
against one `acc0`. The honest stager writes the same rows file for every part, all V*'s shared rows with `acc0` among
them (`85-rec-reprice.sh`, `forge_after.py`), which `rowsShared` passes; rows that differ in a byte, or a part without
rows beside one with, are refused (the `#guard`s).
-/

namespace RecDeploy

/-- **Any two parts hold the same rows** where `rowsShared` passes. -/
theorem rows_eq_of_rowsShared {parts : List PartX} (h : rowsShared parts = true) {s s' : ℕ} {p p' : PartX}
    (hp : parts[s]? = some p) (hp' : parts[s']? = some p') : p.rows = p'.rows := by
  have hall := List.all_eq_true.1 h
  rw [of_decide_eq_true (hall p (List.mem_iff_getElem?.2 ⟨s, hp⟩)),
    of_decide_eq_true (hall p' (List.mem_iff_getElem?.2 ⟨s', hp'⟩))]

/-- What `readX` read of the parts' rows: they are shared (`rowsShared`). -/
theorem readX_rowsShared {x : Claims.Msg} {r : XR} (h : readX x = some r) : rowsShared r.w.parts = true := by
  unfold readX at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, -, ri, -, sh, -, pa, -, dims, -, vs, -, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  obtain ⟨-, h2⟩ := not_or.1 hl
  obtain ⟨-, h3⟩ := not_or.1 h2
  obtain ⟨-, h4⟩ := not_or.1 h3
  simpa using h4

/-- **On acceptance any two parts hold the same rows**: wherever `readX` reads the statement, as it does on every run
the compiled verifier accepts (`run_acc_readX`), parts `s` and `s'` have the same `--rec-rows`. -/
theorem readX_rows_eq {x : Claims.Msg} {r : XR} (h : readX x = some r) {s s' : ℕ} {p p' : PartX}
    (hp : r.w.parts[s]? = some p) (hp' : r.w.parts[s']? = some p') : p.rows = p'.rows :=
  rows_eq_of_rowsShared (readX_rowsShared h) hp hp'

/-- **The parts' recursive statements hold the same rows**: wherever `readX` reads the statement, part `s`'s and part
`s'`'s statements, at any staged inputs and rounds, carry the same `--rec-rows` (`partRaw`'s `recRows`). -/
theorem partRaw_recRows_eq {x : Claims.Msg} {r : XR} (h : readX x = some r) {s s' : ℕ} {p p' : PartX}
    (hp : r.w.parts[s]? = some p) (hp' : r.w.parts[s']? = some p') (st st' : Stage)
    (ms ms' : List (Claims.Msg × Claims.Out)) :
    (partRaw r st x ms s p).recRows = (partRaw r st' x ms' s' p').recRows :=
  readX_rows_eq h hp hp'

/-! ## The deployed parts pass, rows that differ are refused -/

/-- A part's Flock statement, its fields empty: `rowsShared` reads only the rows. -/
private def partAt (name : String) (rows : Option ByteArray) : PartX :=
  ⟨{ statement := name, circuit := .empty, publicFile := .empty, tables := [], zk := true, osCoins := false,
      count := 0, draw := none, partition := none, program := none, work := none, workTable := none,
      stratified := none, closure := none, registered := none, publicInputs := none }, rows⟩

/-- The shared rows file as the honest stager writes it for every part (`rows-<part>.json`). -/
private def sharedRows : ByteArray :=
  ("{\"format\": \"verity/registered-reads/v0\", \"ports\": {\"acc0\": {\"value\": \"rec-acc\", \"leaves\": 64, " ++
    "\"root\": \"01\"}, \"acc1\": {\"value\": \"rec-acc\", \"leaves\": 64, \"root\": \"01\"}, " ++
    "\"rec-top-L0\": {\"value\": \"rec-top-L0\", \"leaves\": 128, \"root\": \"02\"}}}").toUTF8

/-- The deployed shape's parts: the algebra, the stacked levels, the open levels and RecCap, each with the one file. -/
private def deployedParts : List PartX :=
  ["rec-residuals", "rec-open-stacked/l16-h4", "rec-open/l16-h7", "rec-open/l64-h8", "rec-cap/n128"].map
    (partAt · (some sharedRows))

/-- `sharedRows` with its `acc0` registration's root changed. -/
private def otherRows : ByteArray := sharedRows.set! 104 (Char.toNat '9').toUInt8

#guard otherRows != sharedRows
#guard rowsShared deployedParts
#guard rowsShared []
#guard !rowsShared (deployedParts.set 2 (partAt "rec-open/l16-h7" (some otherRows)))
#guard !rowsShared (deployedParts.set 0 (partAt "rec-residuals" (some otherRows)))
#guard !rowsShared (deployedParts.set 4 (partAt "rec-cap/n128" none))
#guard !rowsShared (deployedParts.set 0 (partAt "rec-residuals" none))

end RecDeploy
