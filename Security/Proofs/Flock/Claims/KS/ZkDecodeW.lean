import Proofs.Flock.Claims.KS.ZkDecJ
import Proofs.Flock.Claims.KS.ZkOffSC

/-!
# The deployed `--zk` decoders, with no witness-lane opening off the table

`DecodeSC` at `--zk`, at the setup at the draw, with the level-0 event on the witness lanes (`OffW₀Z`, which
`session_sound_of_table_accZ_w` reads). Where the deployed masked reader looks up no column of table `j` off `C`, every
accepted output table `j`'s reps play to on the decoded tapes opens no level-0 position off `C` on a witness lane unless
it self-clashes:

* `ZkTxNoOff`: wherever the setup at the draw has table `j` with `Zk.schedule`'s schedule and the record decodes to a
  session with table `j`'s two rep streams, the reader looks up no column off `C` in the reps' `zkRepColsOf` entries
  below level 0's `2^d`.
* `not_offW_of_entries`: an output whose level-0 openings are those entries at its own cap (`OpensEntries`) opens no
  position off `C` on a witness lane without a self-clash, in a transcript with `ZkTxNoOff`
  (`Claims.Flock.exists_ne_of_zk_entries`).
* `runsAtZCJ_live_offW`, `runsAtZCJH_live_offW`: `runsAtZCJ_live_entries` and `runsAtZCJH_live_entries` at that
  verdict at table `j₀`, on the outcomes `verify --zk` accepts at the setup at the draw whose transcript has
  `ZkTxNoOff` at table `j₀`, at public and hidden outputs.
* `decodes_of_runsOnZCJW`: a `RunsOnZCJW` gives the decoder table by table (`DecodesTable`) at any verdict its
  verdict implies; `DecodesTable.ok_of_plays`: such a decoder is right at every output its reps play to on the
  decoded tapes, one tape playing to one result (`playsT_det`).
* `decodesTableW_zkTabs`, `decodesTableHW_zkTabs`, `decodesTableW_deploy`: the deployed `opnD` and `decD` at
  `zkTabs` are right at `OkOffWJ` (accepted, and at table `j₀` at the cap of `rootB₀[j₀]` no `OffNoClashWZ` at `C`)
  on the outcomes of `AccTxJ` (`AcceptsAt`, `ZkTxNoOff` at table `j₀` and `root_B[j₀] = rootB₀[j₀]`), at public and
  hidden outputs and at a masked statement in scope, as `decodesTable_zkTabs`, `decodesTableH_zkTabs` and
  `decodesTable_deploy` are at `accepted`.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **No column of table `j` off `C` in a transcript**: wherever the setup at the draw `d` has a table `j`, whose
schedule is `Zk.schedule`'s, and the record decodes to a session with table `j`'s two rep streams, the deployed masked
reader looks up no column off `C` in the reps' `zkRepColsOf` entries at `2·|regions|` extra claims on the streams'
rounds and table `j`'s proof files, below level 0's `2^d`. -/
def ZkTxNoOff (I : Inputs) (d : Lean.Json) (j : ℕ) {N : ℕ} (C : Fin N → List 𝔽) (t : Exec.Transcript) : Prop :=
  ∀ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord) (sch : Flock.Schedule) (tPad : ℕ)
    (s₀ s₁ : StreamRec) (h : j < sts.size),
    FlockDeploy.KS.setupAt (zkInputs I) (some d) = .ok (spec, sts) → Record.decode spec t.record = .ok sess →
    Zk.schedule sts[j].m = .ok (sch, tPad) →
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ →
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ →
    ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.zkRepColsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₀.rounds
          t.proofs[2 * j]! ++
        Claims.Flock.zkRepColsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₁.rounds
          t.proofs[2 * j + 1]!)) p.val = some u ∧ u ≠ C p

/-- **No witness-lane opening off `C` without a self-clash where the reader looks up no column off `C`**: an output
whose level-0 openings are the reader's entries at table `j` at its own cap (`OpensEntries`), in a transcript with
`ZkTxNoOff` at table `j`, opens no level-0 position off the oracle of `C` on a witness lane unless it self-clashes. -/
theorem not_offW_of_entries {I : Inputs} {d : Lean.Json} {t : Exec.Transcript} {j : ℕ} {T : TabZK 𝔽 mPts}
    {cap : Cap Dig} {out : OutZC 𝔽 Dig (List UInt8)} {N : ℕ} (hN : N = 2 ^ (lvOf T.sch 0).logLen)
    (C : Fin N → List 𝔽) (hcap : out.cap₀ = cap) (he : OpensEntries I d t j T cap out) (hno : ZkTxNoOff I d j C t) :
    ¬ (OffW₀Z hs512 T cap (Claims.Flock.oracleOf C) out ∧ ¬ SelfClashZ hs512 T out) := by
  rintro ⟨hoff, hsc⟩
  obtain ⟨spec, sts, sess, sch, tPad, s₀, s₁, h, hs, hd, hz, hf₀, hf₁, hver, hent⟩ := he
  exact hno spec sts sess sch tPad s₀ s₁ h hs hd hz hf₀ hf₁
    (Claims.Flock.exists_ne_of_zk_entries hs512 T cap out hN C _ _ hcap hver hent hoff hsc)

/-- **`runsAtZCJ_live` with no witness-lane opening off `C` at table `j₀` unless it self-clashes**: on the outcomes
`verify --zk` accepts at the setup at the law's draw whose transcript has `ZkTxNoOff` at table `j₀`, each table's
output is accepted and, at table `j₀` at its own cap, opens no level-0 position off `C` on a witness lane without a
self-clash (`runsAtZCJ_live_entries`, `not_offW_of_entries`). -/
theorem runsAtZCJ_live_offW (I : Inputs) (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) (j₀ : ℕ) {N : ℕ} (C : Fin N → List 𝔽) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZJ dj y₀) (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o) ∧ ZkTxNoOff I (dj S) j₀ C (wr S R o))
      (fun S R o j out => out.accepted = true ∧ (j.val = j₀ → N = 2 ^ (lvOf ((planZJ dj y₀ S R).get j).sch 0).logLen →
        ¬ (OffW₀Z hs512 ((planZJ dj y₀ S R).get j) (capOfRoot o.rootB[j.val]!) (Claims.Flock.oracleOf C) out ∧
          ¬ SelfClashZ hs512 ((planZJ dj y₀ S R).get j) out))) := by
  intro ω R o ho j
  obtain ⟨hacc, hno⟩ := ho
  obtain ⟨pend, h₀, h₁, hok, he⟩ := runsAtZCJ_live_entries I ht dj y₀ wr L hT hRec hDraw ω R o hacc j
  refine ⟨pend, h₀, h₁, hok, fun hj hN => ?_⟩
  subst hj
  exact not_offW_of_entries hN C rfl he hno

section Runs

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
  {A : Arith 𝔽 K} {Hs : Hashes 𝔽 D Sl} {planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)}
  {opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl}
  {dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg}
  {fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg}
  {capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D} {L : Law n}
  {acc : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Prop}

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
/-- **The decoder at the law's draws, table by table, from `RunsOnZCJW`**: right at a verdict `ok'` on the outcomes of
`pL`, where each is in `acc` and its outputs of `ok` are accepted and of `ok'` (`decodes_of_runsOnZCJV`). -/
theorem decodes_of_runsOnZCJW {ok : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts →
      Fin (planZ S R).length → OutZC 𝔽 D Sl → Prop}
    (h : RunsOnZCJW A Hs planZ opn dec fin capOf L acc ok) (ω : L.Ω) (R : Reg)
    (ok' : Fin (planZ (L.draw ω) R).length → OutZC 𝔽 D Sl → Prop)
    (pL : LiveOutJ (planZ (L.draw ω) R).length mPts → Prop)
    (hok : ∀ o, pL o → acc (L.draw ω) R o ∧ ∀ j out, ok (L.draw ω) R o j out → out.accepted = true ∧ ok' j out)
    (j : Fin (planZ (L.draw ω) R).length) :
    DecodesTable (outJ Hs (planZ (L.draw ω) R)) ok' pL (opn (L.draw ω) R) (dec (L.draw ω) R) (fun _ _ => true)
      (fin (L.draw ω) R) (capOf (L.draw ω) R) (repJ A Hs (planZ (L.draw ω) R)) j := by
  intro o ho
  obtain ⟨hacc, hk⟩ := hok o ho
  obtain ⟨pend, h₀, h₁, hw⟩ := h ω R o hacc j
  obtain ⟨hacc', hok'⟩ := hk j _ hw
  obtain ⟨hl₀, hl₁⟩ := outZC_lig Hs hacc'
  exact ⟨pend, PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 0 hl₀) h₀,
    PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 1 hl₁) h₁, hok'⟩

end Runs

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`runsAtZCJH_live` with no witness-lane opening off `C` at table `j₀` unless it self-clashes**, under hidden
outputs: `runsAtZCJ_live_offW` from `runsAtZCJH_live_entries`. -/
theorem runsAtZCJH_live_offW (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) (j₀ : ℕ) {N : ℕ} (C : Fin N → List 𝔽) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZHJ dj y₀) (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o) ∧ ZkTxNoOff I (dj S) j₀ C (wr S R o))
      (fun S R o j out => out.accepted = true ∧ (j.val = j₀ → N = 2 ^ (lvOf ((planZHJ dj y₀ S R).get j).sch 0).logLen →
        ¬ (OffW₀Z hs512 ((planZHJ dj y₀ S R).get j) (capOfRoot o.rootB[j.val]!) (Claims.Flock.oracleOf C) out ∧
          ¬ SelfClashZ hs512 ((planZHJ dj y₀ S R).get j) out))) := by
  intro ω R o ho j
  obtain ⟨hacc, hno⟩ := ho
  obtain ⟨pend, h₀, h₁, hok, he⟩ := runsAtZCJH_live_entries I ht hc dj y₀ wr L hT hRec hDraw ω R o hacc j
  refine ⟨pend, h₀, h₁, hok, fun hj hN => ?_⟩
  subst hj
  exact not_offW_of_entries hN C rfl he hno

end FlockSoundness.Discharge.ZkHidden

namespace FlockSoundness.Discharge.Tables

open FlockSoundness.Game

/-- **A game plays a tape to one result.** -/
theorem playsT_det {α : Type} {g : Game α} {ts : List Step} {a b : α} (ha : PlaysT g ts a) (hb : PlaysT g ts b) :
    a = b := by
  induction ha generalizing b with
  | ret _ => exact hb.ret_inv.2.symm
  | send _ _ ih => exact ih hb.send_msg
  | coin _ _ ih => exact ih hb.coin_val

/-- **A decoder right at `ok` is right at every output its tapes play to**: a run on one tape ends at one outcome
(`playsT_det`), so on a live outcome of `pL` every pair of results table `j`'s reps play to on the decoded tapes gives
an output of `ok`. -/
theorem DecodesTable.ok_of_plays {P OutT Cp Op : Type} {J mPts : ℕ} {out : Fin J → Cp → P × P → Op → OutT}
    {ok : Fin J → OutT → Prop} {pL : LiveOutJ J mPts → Prop} {opn : Array ByteArray → Fin J → Op}
    {dec : Fin J → ℕ → ByteArray → List Msg} {one : Fin J → ℕ → Bool}
    {fin : Fin J → Fin 2 → Array ByteArray → List Msg} {capOf : Array ByteArray → Fin J → Cp}
    {G : Fin J → (Fin 2 → Fin mPts → FlockLevel3.GF128) → Fin 2 → Game P} {j : Fin J}
    (h : DecodesTable out ok pL opn dec one fin capOf G j) {o : LiveOutJ J mPts} (ho : pL o) (pend : P × P)
    (h₀ : PlaysT (G j o.pts 0) (repTapeJ dec one fin o j 0) pend.1)
    (h₁ : PlaysT (G j o.pts 1) (repTapeJ dec one fin o j 1) pend.2) :
    ok j (out j (capOf o.rootB j) pend (opn o.proofs j)) := by
  obtain ⟨pend', h₀', h₁', hok⟩ := h o ho
  obtain rfl : pend = pend' := Prod.ext (playsT_det h₀ h₀') (playsT_det h₁ h₁')
  exact hok

end FlockSoundness.Discharge.Tables

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (planZJ outJ repJ)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ planZHJ)
open FlockSoundness.Discharge.Tables (DecodesTable LiveOutJ repTapeJ)

local notation "𝔽" => FlockLevel3.GF128

/-- **A masked output opens a level-0 position off `C` on a witness lane without a self-clash**, at the cap `cap₀`. -/
def OffNoClashWZ {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D)
    (C : Model.Oracle 𝔽) (o : OutZC 𝔽 D Sl) : Prop :=
  OffW₀Z Hs T cap₀ C o ∧ ¬ SelfClashZ Hs T o

/-- **The verdict `accepted`, with no level-0 opening off `C` on a witness lane at table `j₀` unless it
self-clashes**, at the cap of `rootB₀[j₀]`, where `C` has level 0's `2^logLen` columns. -/
def OkOffWJ {mPts : ℕ} (ts : List (TabZK 𝔽 mPts)) (j₀ : ℕ) (rootB₀ : Array ByteArray) {N : ℕ} (C : Fin N → List 𝔽)
    (j : Fin ts.length) (out : OutZC 𝔽 Refine.Dig (List UInt8)) : Prop :=
  out.accepted = true ∧ (j.val = j₀ → N = 2 ^ (lvOf (ts.get j).sch 0).logLen →
    ¬ OffNoClashWZ ZkExec.hs512 (ts.get j) (Exec.capOfRoot rootB₀[j₀]!) (Claims.Flock.oracleOf C) out)

/-- **The deployed live verdict, with no column of table `j₀` off `C`**: `verify --zk` accepts at the setup at the
draw `d` at `wrS` (`AcceptsAt`), the deployed masked reader looks up no column of table `j₀` off `C` in what `wrS`
writes (`ZkTxNoOff`), and the outcome's `root_B[j₀]` is `rootB₀[j₀]`. -/
def AccTxJ (I : Exec.Inputs) (d : Lean.Json) (j₀ : ℕ) (rootB₀ : Array ByteArray) {N : ℕ} (C : Fin N → List 𝔽)
    {J mPts : ℕ} (o : LiveOutJ J mPts) : Prop :=
  Composed.AcceptsAt I d (Claims.Bridge.Custody.wrS I d o) ∧
    Composed.ZkTxNoOff I d j₀ C (Claims.Bridge.Custody.wrS I d o) ∧ o.rootB[j₀]! = rootB₀[j₀]!

/-- **`runsAtZCJ_live_offW`'s verdict is `OkOffWJ` on the outcomes of `AccTxJ`**, whose `root_B[j₀]` is
`rootB₀[j₀]`. -/
theorem okOffW_of_entries {mPts n : ℕ} {Reg : Type} {ts : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)}
    {S : Finset (Fin n)} {R : Reg} {I : Exec.Inputs} {d : Lean.Json} {j₀ : ℕ} {rootB₀ : Array ByteArray} {N : ℕ}
    {C : Fin N → List 𝔽} {o : LiveOutJ (ts S R).length mPts} (ho : AccTxJ I d j₀ rootB₀ C o) (j : Fin (ts S R).length)
    (out : OutZC 𝔽 Refine.Dig (List UInt8))
    (hw : out.accepted = true ∧ (j.val = j₀ → N = 2 ^ (lvOf ((ts S R).get j).sch 0).logLen →
      ¬ (OffW₀Z ZkExec.hs512 ((ts S R).get j) (Exec.capOfRoot o.rootB[j.val]!) (Claims.Flock.oracleOf C) out ∧
        ¬ SelfClashZ ZkExec.hs512 ((ts S R).get j) out))) :
    out.accepted = true ∧ OkOffWJ (ts S R) j₀ rootB₀ C j out := by
  refine ⟨hw.1, hw.1, fun hj hN => ?_⟩
  subst hj
  rw [← ho.2.2]
  exact hw.2 rfl hN

/-- **The deployed `--zk` decoders are right at `zkTabs` at the witness-lane verdict at table `j₀`**: where `verify
--zk` sets its session up at the empty draw `dj ∅`, on untyped inputs under `TagsOkZJ`, the openings `opnD` and
decoder `decD` at `tabsJ` of `zkTabs` are right at `OkOffWJ` on the outcomes of `AccTxJ` (`decodesTable_zkTabs` with
`runsAtZCJ_live_offW`). -/
theorem decodesTableW_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : Composed.TagsOkZJ I) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j₀ : ℕ) (rootB₀ : Array ByteArray) {N : ℕ} (C : Fin N → List 𝔽)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (OkOffWJ (tabsJ sts.size (zkTabs h0 L hz hk hlay)) j₀ rootB₀ C) (AccTxJ I (dj ∅) j₀ rootB₀ C)
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ := setupZJ_of_setupAt hT.1 ht dj hs hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJW
    (Composed.runsAtZCJ_live_offW (Reg := Unit) I ht dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok) j₀ C) () () (OkOffWJ (planZJ dj y₀ ∅ ()) j₀ rootB₀ C) (AccTxJ I (dj ∅) j₀ rootB₀ C)
    (fun o ho => ⟨⟨ho.1, ho.2.1⟩, fun j out hw => okOffW_of_entries ho j out hw⟩)
  have hp := planZJ_eq_tabsJ hT.1 ht dj y₀ hs hok h0 L hz hk hlay ()
  revert i
  rw [← hp, ← opnAtZJ_eq_opnD (planZJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  exact h

/-- **The deployed hidden-output `--zk` decoders are right at `zkTabs` at the witness-lane verdict at table `j₀`**
(`decodesTableH_zkTabs` with `runsAtZCJH_live_offW`). -/
theorem decodesTableHW_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts)) (hshape : sts.forM Flock.Zk.shapeOk = .ok ())
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j₀ : ℕ) (rootB₀ : Array ByteArray) {N : ℕ} (C : Fin N → List 𝔽)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (OkOffWJ (tabsJ sts.size (zkTabs h0 L hz hk hlay)) j₀ rootB₀ C) (AccTxJ I (dj ∅) j₀ rootB₀ C)
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKHJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ :=
    setupZHJ_of_setupAt hT.1 ht hc dj hs hshape hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJW
    (ZkHidden.runsAtZCJH_live_offW (Reg := Unit) I ht hc dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok) j₀ C) () () (OkOffWJ (planZHJ dj y₀ ∅ ()) j₀ rootB₀ C) (AccTxJ I (dj ∅) j₀ rootB₀ C)
    (fun o ho => ⟨⟨ho.1, ho.2.1⟩, fun j out hw => okOffW_of_entries ho j out hw⟩)
  have hp := planZHJ_eq_tabsJ hT.1 ht hc dj y₀ hs hshape hok h0 L hz hk hlay ()
  revert i
  rw [← hp, ← opnAtZHJ_eq_opnD (planZHJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  exact h

namespace Gen

variable [S : ZkSetup]

/-- **The deployed `--zk` decoders are right at `zkTabs` at the witness-lane verdict at a masked statement in
scope**: `decodesTable_deploy` at `OkOffWJ` on the outcomes of `AccTxJ` (`decodesTableW_zkTabs` at public outputs,
`decodesTableHW_zkTabs` at hidden outputs). -/
theorem decodesTableW_deploy {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) (j₀ : ℕ) (rootB₀ : Array ByteArray) {N : ℕ} (C : Fin N → List 𝔽)
    (i : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (OkOffWJ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)) j₀ rootB₀ C)
      (AccTxJ ins.read.I d j₀ rootB₀ C)
      (opnD sts (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (decD (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) i := by
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hset]
    simp
  revert i
  rw [hlen]
  obtain ⟨hzk, hsA⟩ := setupAt_of_setupOf hs hin hset
  rw [hd] at hsA
  obtain ⟨ht, hrr, hl⟩ := tags_of_inScope hs hin hset
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact decodesTableW_zkTabs ht ⟨hh, hrr, hl⟩ (fun _ : Finset (Fin 0) => d) hsA h0 L hz hk hlay j₀ rootB₀ C
  · have hshape := Claims.Bridge.Custody.setup_shapeOk (setupOf_some hset).2 hzk
    obtain ⟨-, hstJ⟩ := setupAt_tableJH hh hsA
    have hp := (Hidden.setupHidden_spec' (hstJ 0 h0)).parse
    have hc : HmOut.parse (ZkExec.zkTags ins.read.I) ins.read.I.circuitFile ins.read.I.tables = .ok sts[0].c :=
      (ZkHidden.parseH_tagsAt ((ZkExec.zkTags_typed _).trans ht) _ _ _ _).symm.trans hp
    exact decodesTableHW_zkTabs ht ⟨hh, hrr, hl⟩ hc (fun _ : Finset (Fin 0) => d) hsA hshape h0 L hz hk hlay j₀
      rootB₀ C

end Gen

/-- **The deployed `--zk` decoders are right at `zkTabs` at the witness-lane verdict at a masked statement in
scope**: `decodesTable_deploy` at `OkOffWJ` on the outcomes of `AccTxJ` (`decodesTableW_zkTabs` at public outputs,
`decodesTableHW_zkTabs` at hidden outputs). -/
theorem decodesTableW_deploy {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) (j₀ : ℕ) (rootB₀ : Array ByteArray) {N : ℕ} (C : Fin N → List 𝔽)
    (i : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (OkOffWJ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)) j₀ rootB₀ C)
      (AccTxJ ins.read.I d j₀ rootB₀ C)
      (opnD sts (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (decD (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) i :=
  Gen.decodesTableW_deploy (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (j₀ := j₀)
    (rootB₀ := rootB₀) (N := N) (C := C) (i := i)

end FlockDeploy.KS
