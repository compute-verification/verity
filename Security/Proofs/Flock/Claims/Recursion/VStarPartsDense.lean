import Proofs.Flock.Claims.Recursion.VStarParts

/-!
# V*'s parts at the dense shape

The dense shape's V* parts are its rec-open parts (its first statements, one per Merkle level) and its algebra parts.
Each part's three facts (`VStar.lean`) follow from the shipped circuit's semantics (`VStarParts.lean`) and named links
that are not the circuit's:

* `PartChecks`: a rec-open part's from `OpenLinks` (`partChecks_of_open`), an algebra part's from `ResReads`
  (`partChecks_of_reads`).
* `PartLogs`: a rec-open part's from `OpenLogLinks` (`partLogs_of_open`), its inner check's calls the climb's SHA-512
  evaluations; an algebra part's with no calls (`partLogs_of_nil`).
* `PartOpens`, at every part (`partOpens_of_climbs`): the part's hm96 row opens each message it holds to the digest its
  public file names (`HeldOpens`), and the setup climbs that digest and the round's commit string to one registered
  root (`RegClimbs`); where the two differ, the climbs collide, a registered tie `Br` finds or a collision among the
  recorded calls.

`vstarParts_dense` puts them together: with each part a rec-open or an algebra part (`DensePart`), the staged V*
statements showing the run (`Staged`), the parts' checks and calls the inner verifier's (`Decomp`, `LogDecomp`) and
some part holding each inner round (`Held`), V*'s parts meet `VStarParts` at the dense shape, which
`codeFacts_of_parts` and `RecDeploy.ksA_deployed_parts` read.
-/

namespace Claims.Recursion.VStar

/-- **Part `s` at the dense shape is a rec-open or an algebra part**: a rec-open part's links for its check and its
calls (`OpenLinks`, `OpenLogLinks`), or an algebra part's links for its check (`ResReads`), whose check makes no
SHA-512 call. -/
def DensePart {Coin : Type} (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) (s : ℕ) : Prop :=
  (OpenLinks D decode L Form.Dense shows chk s ∧ OpenLogLinks D decode L Form.Dense shows logs s) ∨
    (ResReads D decode L Form.Dense shows chk s ∧ ∀ x c ms, logs s x c ms = [])

/-- **`PartChecks` at a dense part** (`partChecks_of_open`, `partChecks_of_reads`). -/
theorem partChecks_dense {Coin Col T : Type} {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : DensePart D decode L shows chk logs s) : PartChecks H D DO L Form.Dense shows chk s := by
  rcases h with ⟨hO, -⟩ | ⟨hR, -⟩
  · exact partChecks_of_open hDO hO
  · exact partChecks_of_reads hDO hR

/-- **`PartLogs` at a dense part** (`partLogs_of_open`, `partLogs_of_nil`). -/
theorem partLogs_dense {Coin Col T : Type} {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : DensePart D decode L shows chk logs s) : PartLogs H D DO L Form.Dense shows logs s := by
  rcases h with ⟨-, hL⟩ | ⟨-, h0⟩
  · exact partLogs_of_open hDO hL
  · exact partLogs_of_nil h0

/-- **`PartOpens` at a dense part, rec-open or algebra** (`partOpens_of_climbs`). -/
theorem partOpens_dense {Coin Col T : Type} {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg} {s : ℕ}
    (hO : HeldOpens D DO L Form.Dense opening dig s) (hC : RegClimbs H D DO L Form.Dense cmt dig s) :
    PartOpens H D DO L Form.Dense opening cmt s :=
  partOpens_of_climbs hO hC

/-- **V\*'s parts at the dense shape**: every part a rec-open or an algebra part (`DensePart`), opening what it holds
through registered climbs (`HeldOpens`, `RegClimbs`), with the staging, the inner verifier's checks and calls, and the
held rounds (`Staged`, `Decomp`, `LogDecomp`, `Held`). -/
theorem vstarParts_dense {Coin W Col T : Type} [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {decode : Msg → Option Flock.Readable.Read}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, DensePart D decode L shows chk logs s) (hO : ∀ s, HeldOpens D DO L Form.Dense opening dig s)
    (hC : ∀ s, RegClimbs H D DO L Form.Dense cmt dig s) (hS : Staged H D I L Form.Dense shows cmt)
    (hD : Decomp H D I.V Form.Dense chk) (hLD : LogDecomp H D I.V Form.Dense logs) (hH : Held D I L Form.Dense) :
    VStarParts H D I DO L opening Form.Dense :=
  ⟨shows, chk, cmt, logs, fun s => partChecks_dense hDO (hP s), fun s => partOpens_dense (hO s) (hC s),
    fun s => partLogs_dense hDO (hP s), hS, hD, hLD, hH⟩

end Claims.Recursion.VStar
