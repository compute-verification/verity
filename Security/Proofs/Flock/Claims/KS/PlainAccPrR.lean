import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.KS.PlainRdRunK
import Proofs.Flock.Claims.KS.PlainRConflict
import Proofs.Flock.Claims.KS.PlainKSLinR

/-!
# The acceptance bound at `VR rd` against the reader's read sets

`PlainRdRunK`'s `plain_accPr_le_rdAllK` over the deployed recursive verifier `VR rd`: at `nextR rd`'s columns
(`colsRV rd`) and its level columns (`plainRawLevelColsRV rd`, read off the run `runOfRV rd` that `decR` decides on).

* `runOfRV rd v j`: `runOfR`'s run, table `j`'s, on a view (`colsRV`'s reading), and `colsRV_runOfRV`: `colsRV` is its
  level-0 columns.
* `plainRawLevelColsRV rd v k ℓ`: `plainRawLevelCols` at that run, rep `k`'s level-`ℓ` columns of table 0.
* `PlainAccPrRdR rd x`: `plain_accPr_le_rdAllK`'s statement at `VR rd`, with `GoodTable` at `deployPlain` (it reads
  neither `V` nor `cols`, so it is the same `GoodTable` at any record built on `deployPlain` with `V := VR rd`).
* `plain_first_of_simDR`, `plain_le_sessionsRawLR`, `plainOffL_le_rdR`, `plain_accPr_le_rdLR`: the `K` chain at
  `VR rd`, at a simulator `P'` sending `plainRootR`'s roots (`PlainSimR`) with its clashes (`PlainClashRunR`).
* `plain_accPr_le_rdAllR_of rd hno hIn : PlainAccPrRdR rd x`, from what the simulation and the decoders still owe
  (`PlainRdInR rd x`), and `rdAllAt_R`: `PlainKSLinR`'s `RdAllAt` at `deployPlainR Readable.relation rd`.
-/

namespace FlockDeploy

open Flock
open RecDeploy (outCoin)

/-- **The run `colsRV rd` reads on a view**: at the statement of the view's last message (`rd`), its messages and the
coins its answers show (`outCoin`), `runOfR`'s run, table `j`'s, at what `decR` decides on. -/
noncomputable def runOfRV (rd : Claims.Msg → Option FinalP) (v : Claims.Seen) (j : ℕ) : Option Claims.Flock.ZkRun :=
  match ((v.msgs.map Prod.fst).getLast?).bind rd with
  | some fl =>
    match decR rd fl.stmt (v.msgs.map Prod.fst) ((v.msgs.map Prod.snd).map outCoin) with
    | some d => runOfDecR d j
    | none => none
  | none => none

theorem colsRV_runOfRV (rd : Claims.Msg → Option FinalP) (v : Claims.Seen) (j : ℕ) :
    colsRV rd v j = colsOf (runOfRV rd v j) := by
  unfold colsRV runOfRV
  cases ((v.msgs.map Prod.fst).getLast?).bind rd with
  | none => rfl
  | some fl =>
    dsimp only
    cases decR rd fl.stmt (v.msgs.map Prod.fst) ((v.msgs.map Prod.snd).map outCoin) <;> rfl

end FlockDeploy

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- **Rep `k`'s level-`ℓ` columns of table 0 at `VR rd`, on a view**: `plainRawLevelCols` at the run `runOfRV rd`
reads. -/
noncomputable def plainRawLevelColsRV (rd : Claims.Msg → Option FinalP) (v : Claims.Seen) (k ℓ : ℕ) :
    List (ℕ × List 𝔽) :=
  match runOfRV rd v 0 with
  | some r =>
    match r.reps[k]? with
    | some y => plainLevelColsOf r.plainShape y.1 y.2 ℓ
    | none => []
  | none => []

/-- **An accepting run at `VR rd` with a collision among the verifier's calls** (`plainEv₁K`'s). -/
def plainEv₁R (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (s : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (VR rd) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (VR rd) x ω s).vLog

open Classical in
/-- **`plain_accPr_le_rdAllK`'s statement at `VR rd`**: at the plain table's setup and schedule, at any prover, tape
and committed word `C` off the good tables, and any read sets `rdD` holding every level-`ℓ + 1` row an accepting run
opens at a slot (`plainRawLevelColsRV`), with `w` a guess fixed by the coins before the slot's fork, the acceptance
probability is at most `ε_table`, the collision run's, the read sets' spread and the off-`C` opening's at level 0
(`colsRV rd`). -/
def PlainAccPrRdR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) : Prop :=
  ∃ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (m k0 : ℕ)
      (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ),
    setupOf false x = some (ins, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
    schedOf sch = ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧ tables false x = [(spec, st)] ∧
    Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧
    ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
      (∀ p, wide false x (C p)) → ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ∀ (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
          Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
            Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
        (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
          Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽),
      (∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
        (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) →
          w r ℓ c = w r ℓ c') →
      (∀ r ℓ (c : Fin ((V false).k x) → Coin), (Claims.run sha512 P (VR rd) x ω c).out = .acc →
        ∀ q a, (q.val, a) ∈ plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω c).seen r.val (ℓ.val + 1) →
          rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes a ∈ rdD r ℓ c q) →
      Claims.accPr sha512 (VR rd) P x ω ≤ εtable false x +
        (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁R rd P x ω s) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
          Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
        Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
          ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
            Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧
              u ≠ C p

/-! ## The draw -/

/-- **`checksAtJ` refuses a session without a draw** (`drawLaw false tags none`). -/
theorem checksAtJ_drawSome {s : Ins} {spec : SessionSpec} {x : Claims.Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {v : Json}
    (h : checksAtJ s spec x link c₁ mids f v = .ok ()) : ∃ d, s.read.draw = some d := by
  unfold checksAtJ at h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases hd : s.read.draw with
  | none => simp [hd, ProvedScope.drawLaw] at h
  | some d => exact ⟨d, rfl⟩

/-- **An accepting step of `VR rd` at SHA-512 is at a statement that sets up at a draw** (`decideR_sha512`,
`checksAtJ_drawSome`). -/
theorem nextR_acc_draw {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((nextR rd x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts d, setupOf false x = some (ins, spec, sts) ∧ ins.read.draw = some d := by
  cases ha : actR rd x as cs with
  | rej => revert h; simp [nextR, ha, Claims.OC.eval]
  | coin c => revert h; simp [nextR, ha, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨-, hc, -, -, -⟩ := decideR_sha512 ha h
    obtain ⟨d, hd⟩ := checksAtJ_drawSome hc
    exact ⟨s, spec, sts, d, actR_setupOf ha, hd⟩

/-- **A statement some run of `VR rd` accepts at SHA-512 sets up at a draw** (`accK_draw` at `VR rd`). -/
theorem accR_draw {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 (VR rd) x) :
    ∃ ins spec sts d, setupOf false x = some (ins, spec, sts) ∧ ins.read.draw = some d := by
  unfold Claims.Flock.NoAcc at hacc
  push Not at hacc
  obtain ⟨Ω, P, ω, r, hr⟩ := hacc
  obtain ⟨as, has⟩ := FlockDeploy.KS.stepWith_acc _ _ _ _ _ _ _ _ _ hr
  exact nextR_acc_draw has

/-! ## The simulator -/

/-- Proof files as lists of byte values. -/
def filesNat (fs : Array ByteArray) : List (List ℕ) := fs.toList.map fun b => b.data.toList.map UInt8.toNat

theorem filesNat_inj : Function.Injective filesNat := by
  have hb : Function.Injective fun b : ByteArray => b.data.toList.map UInt8.toNat := by
    intro p q hpq
    rcases p with ⟨⟨p⟩⟩
    rcases q with ⟨⟨q⟩⟩
    have e : p = q := List.map_injective_iff.mpr (fun _ _ h => UInt8.toNat_inj.mp h) hpq
    rw [e]
  intro a b h
  rcases a with ⟨a⟩
  rcases b with ⟨b⟩
  have e : a = b := List.map_injective_iff.mpr hb h
  rw [e]

/-- **Proof files as a message**: tag `1`, then one zero per their code, so no message codes two and none is empty. -/
def filesCode (fs : Array ByteArray) : Claims.Msg := 1 :: List.replicate (Encodable.encode (filesNat fs)) 0

theorem filesCode_inj : Function.Injective filesCode := by
  intro a b h
  have e := congrArg List.length (List.cons.inj h).2
  rw [List.length_replicate, List.length_replicate] at e
  exact filesNat_inj (Encodable.encode_injective e)

theorem filesCode_ne_nil (fs : Array ByteArray) : filesCode fs ≠ [] := List.cons_ne_nil _ _

open Classical in
/-- The proof files a message codes, none where it codes none. -/
noncomputable def filesDecode (a : Claims.Msg) : Array ByteArray :=
  if h : ∃ fs, filesCode fs = a then Classical.choose h else #[]

theorem filesDecode_code (fs : Array ByteArray) : filesDecode (filesCode fs) = fs := by
  have h : ∃ fs', filesCode fs' = filesCode fs := ⟨fs, rfl⟩
  rw [filesDecode, dif_pos h]
  exact filesCode_inj (Classical.choose_spec h)

theorem filesDecode_nil : filesDecode [] = #[] := by
  rw [filesDecode, dif_neg]
  rintro ⟨fs, h⟩
  exact filesCode_ne_nil fs h

/-- **The simulator's wire at `VR rd`**: the plain wire, with the roots `linkR` reads from the first message's rows'
nodes at the statement's tables, and the proof files `filesCode` writes. -/
noncomputable def plainWR (spec : SessionSpec) (h : 2 * spec.mPts ≤ 65536) : Claims.Msg → Wire 65536 1 spec.mPts :=
  fun x => { plainWire spec.mPts h with
    root := fun a => (((setupOf false x).bind fun v =>
      ((rowsOf a).map (List.map nodeOf)).bind (linkR v.2.2)).getD default).1.toArray
    proofs := filesDecode }

/-- **The roots `VR rd` links at `P`'s first message**: `linkR` at the statement's tables and the first message's rows'
nodes (`plainRoot` at `decR`'s link). -/
noncomputable def plainRootR {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : Array ByteArray :=
  (((setupOf false x).bind fun v =>
    ((rowsOf ((P.next x [] ω).eval sha512).1).map (List.map nodeOf)).bind (linkR v.2.2)).getD default).1.toArray

/-- **A prover of the live session at one table simulates `P` at `VR rd`** (`PlainSimRunK` at `VR rd`): wherever
`VR rd` accepts at `Φ (u, r')`, the live outcome of `P'` at the tape `(ω, u)` and `r'` is the run's (`viewOfJ` at the
decided session's `jAt` and files). -/
def PlainSimRunR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (mPts : ℕ) {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts)
    {τ : Type} (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) :
    Prop :=
  ∀ u r', (Claims.run sha512 P (VR rd) x ω (Φ (u, r'))).out = .acc →
    ∀ s spec sts link c₁ mids f,
      actR rd x ((Claims.run sha512 P (VR rd) x ω (Φ (u, r'))).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
            (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f →
        liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r' =
          viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
            (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)

/-- **A live event at `P'`'s outcome gives an accepting run of `P` at `VR rd` with a collision** (`PlainClashRunK` at
`VR rd`). -/
def PlainClashRunR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (mPts : ℕ) (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin)
    (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → plainEv₁R rd P x ω (Φ (u, r'))

/-- **A simulator of `P` at the plain table, at `VR rd`**: on every tape `u` it sends `plainRootR`'s roots, submits
`repRounds` rounds on rep 0 and the rest on rep 1 at every link point (`SchedL`), its live outcome is `P`'s run at
`plainΦ`'s coins (`PlainSimRunR`), and where `VR rd` refuses it sends no proofs. -/
def PlainSimR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (spec : SessionSpec) (st : Stmt) (hw : 2 * spec.mPts ≤ 65536) (P' : Claims.Prover (Ω × plainTape x)) : Prop :=
  (∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512 P' x
      (ω, u)).1 = plainRootR P x ω) ∧
    (∀ u pts, SchedL (plainDg spec)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512 P' x (ω, u)).1 pts
      (plainN x) (fun _ _ => []) (repRounds false spec st)
      ((liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512 P' x
        (ω, u)).2 pts)) ∧
    PlainSimRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec hw) P' (plainΦ x spec st) ∧
    ∀ u r', (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, r'))).out ≠ .acc →
      (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512 P' x (ω, u) r').proofs =
        #[]

open Classical in
/-- **The plain mixture's first bound at the statement's draw, from a simulation, at `VR rd`**
(`plain_first_of_simDK`'s proof, at `plainVerdictAtJK id` by `plainVerdictAtJK_of_runR`, with `colsRV rd` the run's
columns by `PlainRConflict`'s `colsRV_run`). -/
theorem plain_first_of_simDR (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (ω : Ω) {M : ℕ} (C : Fin M → List 𝔽) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec} {sts₀ : Array Stmt} {d : Json} (hset : setupOf false x = some (s₀, spec₀, sts₀))
    (hd : s₀.read.draw = some d) {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type} [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin)
    (hΦ : ∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E)
    (hsim : PlainSimRunR rd P x ω mPts N dg w P' Φ) :
    Claims.Pr (fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (plainVerdictAtJK id s₀.read.I d C (plainRootR P x ω))
        (liveSession dg 1 mPts (N x)) (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card τ := by
  set E : (Fin ((V false).k x) → Coin) → Prop := fun r => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
    ¬ ∃ (p : Fin M) (v : List 𝔽),
      Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some v ∧ v ≠ C p
    with hEdef
  refine Pr_le_liveMix (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x ω
    (plainVerdictAtJK id s₀.read.I d C (plainRootR P x ω)) E Φ (hΦ E) fun u r' hE => ?_
  obtain ⟨hacc, hno⟩ := hE
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hv⟩ := plainVerdictAtJK_of_runR mPts rd hacc
  obtain ⟨rfl, -, -⟩ : s = s₀ ∧ spec = spec₀ ∧ sts = sts₀ := by
    have e := (actR_setupOf hact).symm.trans hset
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    exact e
  have hroot : plainRootR P x ω = link.1.toArray := by
    obtain ⟨rest, hhead⟩ := run_head sha512 P (VR rd) x ω (Φ (u, r')) hacc
    obtain ⟨v, c₁', r₀, rss, fl, link', pf, hv', -, hr₀, -, hl, -, hd'⟩ := decR_some (actR_decide hact)
    rw [hhead, List.headD_cons] at hr₀
    simp only [DecR.mk.injEq] at hd'
    obtain ⟨-, -, -, rfl, -⟩ := hd'
    simp only [plainRootR, hv', Option.bind_some, hr₀, hl, Option.getD_some]
  rw [hroot, hsim u r' hacc s spec sts link c₁ mids f hact]
  rw [colsRV_run hacc] at hno
  exact hv d hd C hno

open Classical in
/-- `plain_le_sessionsRawLK` at `VR rd`: the acceptance split at `VR rd`'s columns (`colsRV rd`). -/
theorem plain_le_sessionsRawLR (rd : Claims.Msg → Option FinalP) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Lf
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ) :
    Claims.accPr sha512 (VR rd) P x ω ≤ εtable false x +
      (∑ t, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
          SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        (∑ t, Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ)
          (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
              opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD 0 dfltOutL))
            (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
            (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ))
          (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
            PT_LOCAL spec.mPts [] [] (Claims.Flock.oracleOf C) cap₀) (σ t)) / Fintype.card τ +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hper : ∀ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t) ≤
      Accounting.tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ((L.model (Refine.regionsOf st)).shape spec.mPts) +
      expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
          SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t) +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ)
          (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
              opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD 0 dfltOutL))
            (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
            (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ))
          (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
            PT_LOCAL spec.mPts [] [] (Claims.Flock.oracleOf C) cap₀) (σ t) := by
    intro t
    have hs := session_sound_of_table_accL_nodeOff execArith execArith_correct Refine.H512 Refine.enc512 Lf
      (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hsch hm PT_LOCAL spec.mPts hlay [] [] cap₀
      (fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (Claims.Flock.oracleOf C)) = ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512
          Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL)) :=
      congrArg ind (funext fun _ => propext (and_iff_left hT))
    have e₀ : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (outs.getD 0 dfltOutL) ∧
        Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd ⟨h0.2.2, h0.2.1⟩ h0.1.2
      · rfl
    rw [← e]
    refine le_trans hs (le_of_eq ?_)
    simp only [List.length_nil]
    rw [e₀, expect_const, add_zero]
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hdiv := div_le_div_of_nonneg_right hsum (Nat.cast_nonneg (Fintype.card τ))
  rw [Claims.Flock.mean_add, Claims.Flock.mean_add, Claims.Flock.mean_const, mean_sum_sum,
    εtable_one_plain h hsch L] at hdiv
  have hsplit : Claims.accPr sha512 (VR rd) P x ω ≤
      (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
        Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  linarith [hsplit, hacc, hdiv]


open Classical in
/-- `plainOffL_le_rdK` at a simulator `P'` sending `plainRootR`'s roots (`hσ`), at `plainVerdictAtJK id`. -/
theorem plainOffL_le_rdR {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtJK id ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (P' : Claims.Prover (Ω × plainTape x))
    (hσ : ∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512 P' x
      (ω, u)).1 = plainRootR P x ω)
    (C : Fin (wordLen false x) → List 𝔽) (T : Model.Oracle 𝔽) (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ)
    (hL : plainN x + 2 = (V false).k x)
    (hslot : ∀ u, SlotRSL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
      hk0 hm PT_LOCAL spec.mPts (capOfRoot (plainRootR P x ω)[0]!) r.val ℓ.val j
      (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
        (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn (plainDg spec)
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (plainRootR P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
          P' x (ω, u)) (hσ u)))
    (rdD : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ c c' : Fin ((V false).k x) → Coin, (∀ i : Fin ((V false).k x), i.val < j → c i = c' i) → w c = w c')
    (hrel : ∀ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
          (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
            P' x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          v ∈ rdD (plainΦ x spec st (u, rr)) q) :
    (∑ u, Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0)
        (slotOf r.val (ℓ.val + 1))
        (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
            opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val (ℓ.val + 1) d
              (outs.getD 0 dfltOutL))
          (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen)
          (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r.val (ℓ.val + 1)))
        (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm PT_LOCAL spec.mPts [] [] T (capOfRoot (plainRootR P x ω)[0]!))
        (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
          (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn (plainDg spec)
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (plainRootR P x ω)
          (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
            P' x (ω, u)) (hσ u))) /
        Fintype.card (plainTape x) ≤
      ∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q} := by
  have hR : (∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) =
      Claims.Ex fun p : plainTape x × (Fin (plainN x + 2) → LCoin 65536) =>
        ∑ q, ind (fun c => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) (plainΦ x spec st p) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [← Pr_plainΦ (spec := spec) (st := st) hL, Claims.Bridge.Pr_eq_Ex]
    rfl
  rw [hR, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((slotOffRSL_le_offC execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁
    rest ext hk0 hm PT_LOCAL spec.mPts (capOfRoot (plainRootR P x ω)[0]!) T r ℓ.val j _ (hslot u)).trans
    ((offC_simAt_le execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (plainRootR P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        P' x (ω, u)) (hσ u) _ r.val ℓ.val j).trans
      (Claims.Bridge.Live.offC_liveStrat_le plainN (plainDg spec) (plainWR spec hw) sha512 P' x
        (ω, u) _ j (fun rr => w (plainΦ x spec st (u, rr)))
        (fun rr rr' h => hwj _ _ (plainΦ_prefix hL u h))))).trans
    (Claims.Ex_mono fun rr => Finset.sum_le_sum fun q _ => Claims.Bridge.ind_le_ind_imp fun hoff => ?_)
  obtain ⟨hs, hne⟩ := hoff
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  obtain ⟨a, ha, hav⟩ := Claims.Bridge.liveOp_some hv
  have hd := hrel u rr a ha q v hav
  refine ⟨⟨v, hd⟩, fun he => hne ?_⟩
  rw [he, Finset.mem_singleton] at hd
  rw [hv]
  exact congrArg some hd

open Classical in
/-- `plain_accPr_le_rdLK` at `VR rd`, at a simulator `P'` of `P` (`hσ`, `hrun`), its clashes (`hsc`) and slots
(`hslot`). -/
theorem plain_accPr_le_rdLR (rd : Claims.Msg → Option FinalP)
    {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (hset : setupOf false x = some (ins, spec, sts))
    (h : tables false x = [(spec, st)]) {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtJK id ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω)
    (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (P' : Claims.Prover (Ω × plainTape x))
    (hσ : ∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512 P' x
      (ω, u)).1 = plainRootR P x ω)
    (hrun : PlainSimRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec hw) P' (plainΦ x spec st))
    (hsc : PlainClashRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec hw) P'
      (plainΦ x spec st)
      (LiveSelfClash Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
          ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j)))
    (hslot : ∀ (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) u,
      SlotRSL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
        PT_LOCAL spec.mPts (capOfRoot (plainRootR P x ω)[0]!) r.val ℓ.val (forkRound spec st k0 r.val ℓ.val)
        (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
          (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn (plainDg spec)
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (plainRootR P x ω)
          (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
            P' x (ω, u)) (hσ u)))
    (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
      (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) → w r ℓ c = w r ℓ c')
    (hrel : ∀ r ℓ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
          (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
            P' x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          v ∈ rdD r ℓ (plainΦ x spec st (u, rr)) q) :
    Claims.accPr sha512 (VR rd) P x ω ≤ εtable false x +
      (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁R rd P x ω s) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
        Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hL := plainN_add hset
  have htr := plain_first_of_simDR rd P x ω C spec.mPts hset hd plainN (plainDg spec) (plainWR spec hw)
    P' (plainΦ x spec st) (Pr_plainΦ hL) hrun
  have hmain := plain_le_sessionsRawLR rd Lf h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay P ω C hT (plainTape x)
    (capOfRoot (plainRootR P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (plainRootR P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        P' x (ω, u)) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (plainRootR P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        P' x (ω, u)) (hσ)))
  refine hmain.trans (add_le_add (add_le_add (add_le_add le_rfl ?_) ?_) le_rfl)
  · exact le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (plainRootR P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        P' x (ω, u)) (hσ))
      (liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec hw) sha512
        P' x ω _ (plainEv₁R rd P x ω) (plainΦ x spec st) (Pr_plainΦ hL _) hsc)
  · rw [Finset.sum_range]
    refine Finset.sum_le_sum fun r _ => ?_
    rw [Claims.Flock.sum_Icc_one]
    exact Finset.sum_le_sum fun ℓ _ => plainOffL_le_rdR Lf L m k0 l₀ l₁ rest ext hsch hk0 hm hlay d opn hdec hw P ω P' hσ C
      (Claims.Flock.oracleOf C) r ℓ _ hL (hslot r ℓ) (rdD r ℓ) (w r ℓ) (hwj r ℓ) (hrel r ℓ)

/-! ## The bound at every accepted statement -/

open Classical in
/-- **What the bound at `VR rd` still reads off the simulation and the decoders**, at every plain statement in scope
that sets up at a draw, its one table's `fast100` schedule and layout: a leaf, openings and decoders of the table
(`hdec`, at `plainVerdictAtJK id`), and at each prover `P` and tape `ω` a simulator `P'` of `P` (`PlainSimR`) whose
self-clashes are `VR rd`'s collisions (`PlainClashRunR`) and whose later-level openings are rows an accepting run of
`VR rd` opens (`plainRawLevelColsRV`, `plainRel_ofK`'s conclusion). -/
def PlainRdInR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (d : Json) (_ : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (_ : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts),
    ∃ (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512)
      (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
      (_ : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
        DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
          (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
            ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
            ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
          (plainVerdictAtJK id ins.read.I d C rootB₀) opn
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          j),
      ∀ {Ω : Type} (P : Claims.Prover Ω) (ω : Ω), ∃ P' : Claims.Prover (Ω × plainTape x),
        PlainSimR rd P x ω spec st (plain_mPts hs hin hset).1 P' ∧
        ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
          PlainClashRunR rd P x ω spec.mPts plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1) P'
            (plainΦ x spec st)
            (LiveSelfClash Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
              (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
              (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
              (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
                ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j)) ∧
          ∀ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
            Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
                (plainVerdictAtJK id ins.read.I d C (plainRootR P x ω)) opn
                (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
                (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
                (fun j => repsOf (D := Refine.Dig) execArith
                  ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
                (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec)
                  (plainWR spec (plain_mPts hs hin hset).1) sha512 P' x (ω, u) rr) a →
              ∀ (r : Fin 2) (ℓ : ℕ) (q : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).logLen))
                (v : Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes) → 𝔽),
                opCapL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ a q = some v →
                (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, rr))).out = .acc ∧
                  ∃ e, (q.val, e) ∈ plainRawLevelColsRV rd
                      (Claims.run sha512 P (VR rd) x ω (plainΦ x spec st (u, rr))).seen r.val (ℓ + 1) ∧
                    rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes e = v

open Classical in
/-- **The bound at `VR rd` against the reader's read sets** (`plain_accPr_le_rdAllK`'s proof at `VR rd`): at a
statement some run of `VR rd` accepts, from the simulation and the decoders (`PlainRdInR`). -/
theorem plain_accPr_le_rdAllR_of (rd : Claims.Msg → Option FinalP) {x : Claims.Msg}
    (hno : ¬ Claims.Flock.NoAcc sha512 (VR rd) x) (hIn : PlainRdInR rd x) : PlainAccPrRdR rd x := by
  obtain ⟨s, hs, hin⟩ := accR_inScope Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := accR_draw hno
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  obtain ⟨sch, hF, hF'⟩ := exec_sched hsch
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs hin hset h0 L C hw hg
    rwa [e] at this
  refine ⟨ins, spec, sts, sts[0]'h0, sch, (sts[0]'h0).m, k0, l₀, l₁, rest, ext, hset,
    Array.getElem?_eq_getElem h0, hF, hF'.symm, ht, hsch, ?_⟩
  intro Ω _ _ P ω C hw hg rdD w hwj hrd
  obtain ⟨Lf, opn, hdec, hP⟩ := hIn s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay
  obtain ⟨P', ⟨hσ, hS, hrun, -⟩, hC⟩ := hP P ω
  obtain ⟨hsc, hrel⟩ := hC C hw
  exact plain_accPr_le_rdLR rd Lf hset ht hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay opn hdec
    (plain_mPts hs hin hset).1 P ω C (hnc C hw hg) P' hσ hrun hsc
    (fun r ℓ u => slotRSL_lead L (plain_mPts hs hin hset).2.2 execArith Refine.H512 Refine.enc512 Lf _ k0 l₀ l₁ rest
      ext hsch hk0 hm13 PT_LOCAL spec.mPts (capOfRoot (plainRootR P x ω)[0]!)
      (leadsAfter_simAt execArith Refine.H512 Refine.enc512 Lf
        (plainTab L _ k0 l₀ l₁ rest ext hsch hm13 PT_LOCAL hlay) _ _ opn (plainDg spec)
        (plainDec (plainTab L _ k0 l₀ l₁ rest ext hsch hm13 PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRootR P x ω)) (fun _ => rfl) (plainRootR P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainWR spec (plain_mPts hs hin hset).1)
          sha512 P' x (ω, u)) (hσ u) (hS u)) r ℓ.val)
    rdD w hwj fun r ℓ u rr a ha q v hv => by
      obtain ⟨hacc, e, he, rfl⟩ := hrel u rr a ha r ℓ.val q v hv
      exact hrd r ℓ _ hacc q e he

/-- **`RdAllAt` at the plain record at the deployed reader** (U2b's form), from `plain_accPr_le_rdAllR_of` at every
statement some run of `VR rd` accepts. -/
theorem rdAllAt_R (rd : Claims.Msg → Option FinalP)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc sha512 (VR rd) x → PlainRdInR rd x) :
    RdAllAt (deployPlainR Readable.relation rd) (fun _ => plainRawLevelColsRV rd) :=
  fun x hno => plain_accPr_le_rdAllR_of rd hno (hIn x hno)

end FlockDeploy.KS
