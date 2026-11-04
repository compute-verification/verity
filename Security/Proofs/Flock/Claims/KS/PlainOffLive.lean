import Proofs.Flock.Claims.KS.PlainLeafFork
import Proofs.Flock.Claims.KS.PlainOff
import Proofs.Flock.Claims.KS.OffSim

/-!
# The plain slot's off mass, on the deployed coins

`slotOffRSL_le_offC` (`PlainOff`) bounds the plain slot `(r, ℓ + 1)`'s raw term at a strategy that reaches the slot's
node after `j` coins by the session's plurality off mass after `j` coins at the recorded cap (`opCapL`). Route (c)
carries that mass to the deployed coins as one run's, never as a fork:

* `Claims.Bridge.liveOp_some`, `Claims.Bridge.retOK_det`: a related live opening is a related model outcome's, and a
  live outcome's decoded model outcome is one (`playsT_det`).
* `offC_afterAt`, `offC_simSessG`, `offC_simAt_le`: against the simulated strategy after `Commit`, the off mass after
  `j` coins at `opCapL` is at most the live session's at the decoded outcome's openings (`Claims.Bridge.Sim.offC_le`;
  the dead branches' empty openings open nothing, `opRL_nul`), as `forkC_simAt_le` is for the fork.
* `plainΦ_prefix`: the simulator's first `j` coins read only the first `j` live coins (`histKey`).
* `plainOffAtU`, `plainOffAtU_le_live`: the slot's term at the concrete simulator at tape `u`, at `leafPlain`, is at
  most the chance over the live coins of opening off a table that reads the first `j` of them
  (`Claims.Bridge.Live.offC_liveStrat_le`).
* `plainOffU_le`: averaged over the tape, at most the chance over the deployed coins of opening off a table that reads
  their first `j`, at any deployed openings `opD` that every decoded opening is (`Pr_plainΦ`).
-/

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables

open Classical in
/-- An opening of the related model outcome is one of a model outcome the live outcome is related to. -/
theorem liveOp_some {β α Pt V : Type} {R : β → α → Prop} {op : α → Pt → Option V} {b : β} {p : Pt} {v : V}
    (h : liveOp R op b p = some v) : ∃ a, R b a ∧ op a p = some v := by
  unfold liveOp at h
  split_ifs at h with he
  exact ⟨_, Classical.choose_spec he, h⟩

/-- **A live outcome decodes to one model outcome**: the reps' runs on the decoded tapes end at one result each
(`PlainDraw.playsT_det`). -/
theorem retOK_det {P OutT Cp Op : Type} {J mPts : ℕ} {out : Fin J → Cp → P × P → Op → OutT}
    {pL : LiveOutJ J mPts → Prop} {opn : Array ByteArray → Fin J → Op} {dec : Fin J → ℕ → ByteArray → List Game.Msg}
    {one : Fin J → ℕ → Bool} {fin : Fin J → Fin 2 → Array ByteArray → List Game.Msg}
    {capOf : Array ByteArray → Fin J → Cp} {G : Fin J → (Fin 2 → Fin mPts → FlockLevel3.GF128) → Fin 2 → Game P}
    (b : LiveOutJ J mPts) (a a' : List OutT) (h : RetOK out pL opn dec one fin capOf G b a)
    (h' : RetOK out pL opn dec one fin capOf G b a') : a = a' := by
  obtain ⟨-, pend, hp, rfl⟩ := h
  obtain ⟨-, pend', hp', rfl⟩ := h'
  have e : pend = pend' := funext fun j => by
    have h0 := PlainDraw.playsT_det (hp j 0) (hp' j 0)
    have h1 := PlainDraw.playsT_det (hp j 1) (hp' j 1)
    simp only [sel] at h0 h1
    exact Prod.ext (by simpa using h0) (by simpa using h1)
  rw [e]

end Claims.Bridge

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-! ## Through the simulation -/

section Sim

variable {Pt V : Type} [Fintype Pt] [Fintype V] [DecidableEq V] [Nonempty V]

theorem offC_afterAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t])) {cap₀ : Model.Cap D}
    (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) (op : List (Model.OutCL 𝔽 D Sl) → Pt → Option V) (j : ℕ) :
    offC op j (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] cap₀) (afterAt A Hc E Lf t τ h) =
      offC op j (sessionBL A Hc E Lf [t]) τ := by
  subst h
  show offC op j _ (afterOne A Hc E Lf t τ) = _
  unfold afterOne
  rw [offC_cast op (sessAfterBL_one A Hc E Lf t τ.1)]
  exact (offC_send op j τ).symm

theorem offC_simSessG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (ts : List (TabSpec mPts)) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop) (opn : Array ByteArray → Fin ts.length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg)
    (one : Fin ts.length → ℕ → Bool) (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin ts.length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) (op : List (Model.OutCL 𝔽 D Sl) → Pt → Option V) (j : ℕ) :
    offC op j (sessionBL A Hc E Lf ts) (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr) =
      offC op j (modelSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os)
          (fun j => repsOf (D := D) A (ts.get j)))
        ((Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)))
          (simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL opn dg
            dec one fin capOf N (fun j => repsOf (D := D) A (ts.get j))
            (fun j _ _ => Model.allInh_repC A _ (ts.get j).S (ts.get j).sch (ts.get j).hm _) hdec)).strategy Pr) := by
  unfold simSessG
  exact offC_eqRec _ _ _ _

theorem bind_none_fun {α β : Type} (o : Option α) : (o.bind fun _ => (none : Option β)) = none := by
  cases o <;> rfl

open Classical in
/-- **The off mass after `j` coins at the recorded cap, against the simulated strategy after `Commit`,** is at most
the live session's after `j` coins at the decoded outcome's openings (`Claims.Bridge.liveOp`): the decoded outcome is
one (`retOK_det`), and the dead branches open nothing (`opRL_nul`). -/
theorem offC_simAt_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (t : TabSpec mPts) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ [t].length mPts → Prop) (opn : Array ByteArray → Fin [t].length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg)
    (one : Fin [t].length → ℕ → Bool) (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin [t].length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀)
    (sch : FlockSoundness.Accounting.Schedule) (r ℓ j : ℕ) :
    offC (opCapL Hc E Lf sch r ℓ) j
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) ≤
      offC (Claims.Bridge.liveOp (Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Hc E Lf cap pend os) pL opn dec
          one fin capOf (fun j => repsOf (D := D) A ([t].get j))) (opCapL Hc E Lf sch r ℓ)) j
        (liveSession dg [t].length mPts N) Pr := by
  unfold simAt
  rw [offC_afterAt, offC_simSessG]
  have hnul : ∀ (cap : Model.Cap D) pend (outs : List (Model.OutCL 𝔽 D Sl)),
      outs.getD 0 dfltOutL = tableOutL Hc E Lf cap pend (nulOp (D := D) (Sl := Sl)) →
        ¬ ∃ p, (opCapL Hc E Lf sch r ℓ outs p).isSome := by
    rintro cap pend outs hp ⟨p, hs⟩
    simp only [opCapL, hp, opRL_nul, bind_none_fun, Option.isSome_none, Bool.false_eq_true] at hs
  have hall := Claims.Bridge.Sim.all_mapDead (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)))
    (D' := fun g σ => prob (fun outs => ∃ p, (opCapL Hc E Lf sch r ℓ outs p).isSome) g σ = 0)
    (fun _ σ ⟨caps, _, _, hg, _, e⟩ => by
      subst e
      exact Claims.Bridge.prob_posOf_answerWith _ _ caps hg dfltOutL
        (fun outs h => by
          obtain ⟨pend, hp⟩ := h ⟨0, Nat.one_pos⟩
          exact hnul _ pend outs hp) σ)
    _ (Claims.Bridge.all_simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o)
      dfltOutL pL opn dg dec one fin capOf N (fun j => repsOf (D := D) A ([t].get j))
      (fun j _ _ => Model.allInh_repC A _ ([t].get j).S ([t].get j).sch ([t].get j).hm _) hdec)
  exact Claims.Bridge.Sim.offC_le (fun a p h => ⟨p, h⟩)
    (fun b a hba p v hv => by rw [Claims.Bridge.liveOp_eq (Claims.Bridge.retOK_det) _ hba]; exact hv) _ hall j Pr

end Sim

/-! ## At the concrete simulator -/

section Plain

open Flock FlockSoundness.Discharge
open Lean (Json)

/-- **The simulator's first `j` coins read only the first `j` live coins**: coordinate `i` of `fnAt` reads the live
coins at rounds `i' ≤ i` only (`histKey`). -/
theorem plainΦ_prefix {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (hL : plainN x + 2 = (V false).k x)
    (u : plainTape x) {j : ℕ} {r r' : Fin (plainN x + 2) → LCoin 65536}
    (h : ∀ i : Fin (plainN x + 2), i.val < j → r i = r' i) :
    ∀ i : Fin ((V false).k x), i.val < j → plainΦ x spec st (u, r) i = plainΦ x spec st (u, r') i := by
  intro i hi
  rw [plainΦ_eq hL]
  have hk : plainKey x spec st i (fun i' => r (Fin.cast hL.symm i')) =
      plainKey x spec st i (fun i' => r' (Fin.cast hL.symm i')) :=
    (Claims.Flock.FnFork.prefixKey_histKey _).read i _ _ fun i' hi' m _ => by
      rw [h (Fin.cast hL.symm i') (by simp only [Fin.val_cast]; exact lt_of_le_of_lt hi' hi)]
  show fnAt hL (plainNOf x spec st) (plainKey x spec st) u r i = fnAt hL (plainNOf x spec st) (plainKey x spec st) u r' i
  unfold fnAt fnC mergeC
  funext m
  dsimp only
  rw [hk, h (Fin.cast hL.symm i) (by simp only [Fin.val_cast]; exact hi)]

/-- **The plain slot `(r, ℓ + 1)`'s raw term at the concrete simulator at tape `u`, at `leafPlain`**: the per-position
off mass at the slot's node at the plurality chooser, `session_sound_of_table_accL_nodeOff`'s slot term. -/
noncomputable def plainOffAtU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (T : Model.Oracle 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (u : plainTape x) : ℝ :=
  Lock.nodeE (plurSL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0)
    (slotOf r.val (ℓ.val + 1))
    (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
        opRL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val (ℓ.val + 1) d
          (outs.getD 0 dfltOutL))
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen)
      (plurSL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r.val (ℓ.val + 1)))
    (lockSessL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
      hk0 hm PT_LOCAL spec.mPts [] [] T (capOfRoot (plainRoot P x ω)[0]!))
    (plainSimAtU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- **The slot's term at tape `u` is at most the chance over the live coins of opening off a table that reads their
first `j`**, where the simulated strategy reaches the slot's node after `j` coins (`PlainSlotQU`): through the
recorded cap (`slotOffRSL_le_offC`), the simulation (`offC_simAt_le`) and the live coins (`offC_liveStrat_le`). -/
theorem plainOffAtU_le_live {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (T : Model.Oracle 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x)
    (hslot : PlainSlotQU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u)
    (W : (Fin (plainN x + 2) → LCoin 65536) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hW : ∀ rr rr' : Fin (plainN x + 2) → LCoin 65536, (∀ i : Fin (plainN x + 2), i.val < j → rr i = rr' i) →
      W rr = W rr') :
    plainOffAtU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C T r ℓ u ≤
      Claims.Ex fun rr : Fin (plainN x + 2) → LCoin 65536 => ∑ q, ind (fun b => Rewinding.Off
        (Claims.Bridge.liveOp (Claims.Bridge.RetOK
          (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
          (plainVerdictAt I d C (plainRoot P x ω)) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j)))
          (opCapL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val))
        (W rr) b q)
        (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
          (plainSim P x spec st) x (ω, u) rr) :=
  (slotOffRSL_le_offC execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀ l₁
    rest ext hk0 hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) T r ℓ.val j _ hslot).trans
    ((offC_simAt_le execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u) _ r.val ℓ.val j).trans
      (Claims.Bridge.Live.offC_liveStrat_le plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
        (ω, u) _ j W hW))

open Classical in
/-- **The slot's term averaged over the tape is at most the chance over the deployed coins of opening off a table
that reads their first `j`**, at deployed openings `opD` that every decoded live opening is: the competitor at the
live coins is the table at the simulator's coins, which reads only the first `j` live coins (`plainΦ_prefix`), and
the simulator's coins are uniform (`Pr_plainΦ`). -/
theorem plainOffU_le {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (T : Model.Oracle 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (hL : plainN x + 2 = (V false).k x)
    (hslot : ∀ u, PlainSlotQU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u)
    (opD : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Option (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ c c' : Fin ((V false).k x) → Coin, (∀ i : Fin ((V false).k x), i.val < j → c i = c' i) → w c = w c')
    (hrel : ∀ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
          (plainVerdictAt I d C (plainRoot P x ω)) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSim P x spec st) x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          opD (plainΦ x spec st (u, rr)) q = some v) :
    (∑ u, plainOffAtU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C T r ℓ u) /
        Fintype.card (plainTape x) ≤
      ∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => Rewinding.Off opD (w c) c q := by
  have hR : (∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => Rewinding.Off opD (w c) c q) =
      Claims.Ex fun p : plainTape x × (Fin (plainN x + 2) → LCoin 65536) =>
        ∑ q, ind (fun c => Rewinding.Off opD (w c) c q) (plainΦ x spec st p) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [← Pr_plainΦ (spec := spec) (st := st) hL, Claims.Bridge.Pr_eq_Ex]
    rfl
  rw [hR, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine (plainOffAtU_le_live L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C T r ℓ j u (hslot u)
    (fun rr => w (plainΦ x spec st (u, rr)))
    (fun rr rr' h => hwj _ _ (plainΦ_prefix hL u h))).trans
    (Claims.Ex_mono fun rr => Finset.sum_le_sum fun q _ => Claims.Bridge.ind_le_ind_imp fun hoff => ?_)
  obtain ⟨hs, hne⟩ := hoff
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  obtain ⟨a, ha, hav⟩ := Claims.Bridge.liveOp_some hv
  have hd := hrel u rr a ha q v hav
  refine ⟨by rw [hd]; rfl, ?_⟩
  rw [hd]
  rw [hv] at hne
  exact hne

end Plain

end FlockDeploy.KS
