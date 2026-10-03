import FlockSoundness.Discharge.ZkPubIn.Composed
import FlockSoundness.Discharge.ZkPubIn.Tables
import FlockSoundness.Discharge.ZkHidden.SoundRuns
import FlockSoundness.Discharge.ZkHidden.SoundDrawOk

/-!
# `RunsZC` at the `--zk` tables with the verifier's public inputs (statements)

ZkHidden's `RecordsLiveZK`, `opnAtZH`, `TablesAtZKHLive`, `RunsZCLiveH` and `drawOkZKH_of_exec` with
`--public-inputs F` (`own` F's bytes): each reads flock-verify's acceptance (`AcceptsZKP I own`) and setupHidden's
statement at `some own` where the original reads `AcceptsZK I` and setupHidden's at `none`.

* `RecordsLiveZKP`: `RecordsLiveZK` at `AcceptsZKP I own` and `setupOfP`.
* `opnAtZHP`: `opnAtZH` at setupHidden's statement with `own`.
* `TablesAtZKHPLive`: `TablesAtZKHLive` at `AcceptsZKP I own`, `DrawOkZKHP`, `DrawSetupZKHP` and `tablesAtZKHP`.
* `RunsZCLiveHP`: `RunsZCLiveH` at those, concluding `RunsOnZCA` at `AcceptsZKP I own`.
* `drawOkZKHP_of_exec`: `drawOkZKH_of_exec` with `own`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec ZkLink ZkHidden HiddenDec

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`wr` is the coin server's record of the live `--zk --public-inputs F` session** (`RecordsLiveZK` at
`AcceptsZKP I own`): whenever flock-verify accepts what `wr` writes for the outcome `o` at draw `S`, the record carries
the draw `dj S`, the proof files are `o`'s, and the record decodes to a session whose table's `root_B`, link points and
rep streams' rounds are `o`'s. -/
def RecordsLiveZKP (I : Inputs) (own : ByteArray) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ S R o, AcceptsZKP I own (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOfP (zkInputs I) own (wr S R o).record = .ok (spec, sts) →
      Record.decode spec (wr S R o).record = .ok sess →
      sess.rootB[0]! = o.rootB ∧ ptsOf mPts sess.points = o.pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = pick (o.rep0, o.rep1) r

/-- **The `--zk` openings at draw `S` with the verifier's public inputs** (`opnAtZH` at setupHidden's statement with
`own`). -/
noncomputable def opnAtZHP (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) :
    ZkLink.OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
      I.program (some own) with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZ T S R) #[proofs[0]!, proofs[1]!]
    | .error _ => default
  | .error _ => default

/-- **Wherever flock-verify accepts a session with the verifier's public inputs, the `--zk` table is the
executable's** (`TablesAtZKHLive` at `AcceptsZKP I own`, `DrawOkZKHP`, `DrawSetupZKHP` and `tablesAtZKHP`). -/
def TablesAtZKHPLive : Prop :=
  ∀ (I : Inputs) (own : ByteArray) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZKP I own dj wr →
    ∀ S R o, DrawOkZKHP I own dj mPts S → AcceptsZKP I own (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program (some own) = .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZKHP I own mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKHP (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

/-- **`RunsZC` at the live `--zk` tables with the verifier's public inputs, at the law's draws** (`RunsZCLiveH` at
`AcceptsZKP I own`, `tablesAtZKHP`, `opnAtZHP`, `TablesAtZKHPLive`, `RecordsLiveZKP` and `DrawOkZKHP`). -/
def RunsZCLiveHP : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (own : ByteArray) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    TablesAtZKHPLive (n := n) (mPts := mPts) (Reg := Reg) → TagsOkH I → RecordsLiveZKP I own dj wr →
    (∀ ω, DrawOkZKHP I own dj mPts (L.draw ω)) →
    RunsOnZCA (K := 𝕂) execArith hs512 (tablesAtZKHP (Reg := Reg) dj y₀) (AcceptsZKP I own) wr
      (opnAtZHP (tablesAtZKHP dj y₀) I own dj) (decAtZ (tablesAtZKHP dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **Wherever flock-verify accepts with the verifier's public inputs, the `--zk` table is the executable's.** -/
theorem tablesAtZKHP_live : TablesAtZKHPLive (n := n) (mPts := mPts) (Reg := Reg) := by
  sorry

/-- **`RunsZC` at the live `--zk` tables with the verifier's public inputs** (`runsZC_liveH`'s proof). -/
theorem runsZC_liveHP : RunsZCLiveHP := by
  sorry

/-- **The coin server's draw is `DrawOkZKHP` at every draw of the live law** (`drawOkZKH_of_exec` with `own`). -/
theorem drawOkZKHP_of_exec (I : Inputs) (own : ByteArray) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Audit.Law.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Audit.Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    DrawOkZKHP I own dj (mPtsOf c.g (∑ s, ks s))
      ((Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) := by
  sorry

end FlockSoundness.Discharge.ZkPubIn
