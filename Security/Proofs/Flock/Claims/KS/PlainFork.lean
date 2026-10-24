import Proofs.Flock.Claims.KS.PlainClashLater
import Proofs.Flock.Claims.KS.PlainPartsK
import Proofs.Flock.Claims.KS.ForkSim

/-!
# The concrete simulator's fork advantage, from its alignment and its clash

`PlainSimForkLe` bounds the strategies' fork advantage at rep `r`'s level `ℓ + 1` (`plainAdv`, `Lock.forkE` at that
slot's table node) by the simulated fork at the plain fork round `plainJ r ℓ`. It splits in two:

* `PlainForkAlign`: against the simulated strategy after `Commit` (`plainSimAt`), the fork at the slot's node is at
  most the fork after `plainJ r ℓ` coins of the event `ForkEvL`: both runs record one cap `d` for rep `r`'s level
  `ℓ + 1` and open a position of it to different rows;
* `PlainForkClash`: two accepting runs of `P` at the concrete simulator's coins whose decoded live outcomes have
  `ForkEvL` collide among the verifier's calls in the two (`PlainForkQ`).

The rest is the simulation's: a fork after `j` coins against the simulated strategy is at most the live fork after `j`
coins of the decoded event (`forkC_simAt_le`, by `Claims.Bridge.Sim.forkC_le`, the dead branches' empty openings
opening nothing, `opRL_nul`), which is the deployed runs' fork at round `j` (`forkC_liveStrat`).

* `plainAdv_le_of`, `plainSimForkLe_of`: so `PlainSimForkLe` holds from `PlainForkAlign` and `PlainForkClash`.
* `PlainHDecK`, …, `plainSimForkLe_ofK`: the same at `VK keep` (`plainSimK keep`, `plainVerdictAtK keep`), so
  `PlainSimForkLeK keep` holds from `PlainForkAlignK keep` and `PlainForkClashK keep`.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-! ## Through the simulation -/

/-- **Two live outcomes decoding to a pair with `Ev`** under `R`. -/
def LiveEv {β α : Type} (R : β → α → Prop) (Ev : α → α → Prop) (b b' : β) : Prop :=
  ∃ a a', R b a ∧ R b' a' ∧ Ev a a'

theorem ind_le_ind {α β : Type} {p : α → Prop} {q : β → Prop} {a : α} {b : β} (h : p a → q b) :
    ind p a ≤ ind q b := by
  unfold ind
  split_ifs with h1 h2 <;> first | exact absurd (h h1) h2 | norm_num

theorem forkC_afterAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t])) {cap₀ : Model.Cap D}
    (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) (F : List (Model.OutCL 𝔽 D Sl) → List (Model.OutCL 𝔽 D Sl) → ℝ) (j : ℕ) :
    forkC F j (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] cap₀) (afterAt A Hc E Lf t τ h) =
      forkC F j (sessionBL A Hc E Lf [t]) τ := by
  subst h
  show forkC F j _ (afterOne A Hc E Lf t τ) = _
  unfold afterOne
  rw [forkC_cast (sessAfterBL_one A Hc E Lf t τ.1)]
  exact (forkC_send F j τ).symm

theorem forkC_simSessG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (ts : List (TabSpec mPts)) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop) (opn : Array ByteArray → Fin ts.length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N))
    (F : List (Model.OutCL 𝔽 D Sl) → List (Model.OutCL 𝔽 D Sl) → ℝ) (j : ℕ) :
    forkC F j (sessionBL A Hc E Lf ts) (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr) =
      forkC F j (modelSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os)
          (fun j => repsOf (D := D) A (ts.get j)))
        ((Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)))
          (simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL opn dg
            dec one fin capOf N (fun j => repsOf (D := D) A (ts.get j))
            (fun j _ _ => Model.allInh_repC A _ (ts.get j).S (ts.get j).sch (ts.get j).hm _) hdec)).strategy Pr) := by
  unfold simSessG
  exact forkC_eqRec _ _ _ _

/-- **The dead branches' empty openings open nothing.** -/
theorem opRL_nul {D Sl : Type} [Inhabited Sl] (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule) (r ℓ : ℕ) (d cap : Model.Cap D)
    (pend : Model.PendL 𝔽 D × Model.PendL 𝔽 D) (p : Fin (2 ^ (lvOf sch ℓ).logLen)) :
    opRL Hc E Lf sch r ℓ d (tableOutL Hc E Lf cap pend (nulOp (D := D) (Sl := Sl))) p = none := by
  unfold opRL opOfL
  rw [List.filter_eq_nil_iff.2 fun x hx => ?_]
  · rfl
  · simp only [decide_eq_true_eq, not_and]
    exact fun _ => not_verifiedL_nil Hc E Lf d _ _
      (slotOpensL_row_nil (tableOutL_nulOp_opens Hc E Lf cap pend) hx)

open Classical in
/-- **A fork after `j` coins against the simulated strategy after `Commit`** is at most the live fork after `j` coins
of the decoded event, for an event `Ev` of two model outcomes whose each run is in an event `Q` no output at the empty
openings has. -/
theorem forkC_simAt_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (t : TabSpec mPts) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ [t].length mPts → Prop) (opn : Array ByteArray → Fin [t].length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀)
    {Ev : List (Model.OutCL 𝔽 D Sl) → List (Model.OutCL 𝔽 D Sl) → Prop} {Q : List (Model.OutCL 𝔽 D Sl) → Prop}
    (hQ : ∀ a a', Ev a a' → Q a ∧ Q a')
    (hnul : ∀ (cap : Model.Cap D) pend (outs : List (Model.OutCL 𝔽 D Sl)),
      outs.getD 0 dfltOutL = tableOutL Hc E Lf cap pend (nulOp (D := D) (Sl := Sl)) → ¬ Q outs) (j : ℕ) :
    forkC (fun a a' => ind (fun p : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) => Ev p.1 p.2) (a, a')) j
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) ≤
      forkC (fun b b' => ind (fun p : LiveOutJ [t].length mPts × LiveOutJ [t].length mPts =>
          LiveEv (Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Hc E Lf cap pend os) pL opn dec one fin capOf
            (fun j => repsOf (D := D) A ([t].get j))) Ev p.1 p.2) (b, b')) j
        (liveSession dg [t].length mPts N) Pr := by
  unfold simAt
  rw [forkC_afterAt, forkC_simSessG]
  have hall := Claims.Bridge.Sim.all_mapDead (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)))
    (D' := fun g σ => prob Q g σ = 0)
    (fun _ σ ⟨caps, _, _, hg, _, e⟩ => by
      subst e
      exact Claims.Bridge.prob_posOf_answerWith _ _ caps hg dfltOutL
        (fun outs h => by
          obtain ⟨pend, hp⟩ := h ⟨0, Nat.one_pos⟩
          exact hnul _ pend outs hp) σ)
    _ (Claims.Bridge.all_simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o)
      dfltOutL pL opn dg dec one fin capOf N (fun j => repsOf (D := D) A ([t].get j))
      (fun j _ _ => Model.allInh_repC A _ ([t].get j).S ([t].get j).sch ([t].get j).hm _) hdec)
  have hFQ : ∀ a a', ind (fun p : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) => Ev p.1 p.2) (a, a') ≤
      ind Q a := fun a a' => ind_le_ind fun h1 => (hQ _ _ h1).1
  have hFQ' : ∀ a a', ind (fun p : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) => Ev p.1 p.2) (a, a') ≤
      ind Q a' := fun a a' => ind_le_ind fun h1 => (hQ _ _ h1).2
  exact Claims.Bridge.Sim.forkC_le hFQ hFQ' (fun b b' => ind_nonneg _ _)
    (fun b a b' a' hb hb' => ind_le_ind fun h1 => ⟨a, a', hb, hb', h1⟩) _ hall j Pr

/-- **The live fork after `j` coins is the deployed runs' fork at round `j`**: two runs at coins sharing their first
`j` (`Claims.forkCoins`). -/
theorem forkC_liveStrat {nMax J mPts : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : Claims.Msg → Wire nMax J mPts) (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) (F : LiveOutJ J mPts → LiveOutJ J mPts → ℝ) (j : Fin (N x + 2)) :
    forkC F j.val (liveSession dg J mPts (N x))
        (Claims.Bridge.Live.liveStrat (fun _ => J) (fun _ => mPts) N dg w H P x ω) =
      Claims.Ex fun rr : (Fin (N x + 2) → LCoin nMax) × (Fin (N x + 2) → LCoin nMax) =>
        F (Claims.Bridge.Live.liveOut (fun _ => J) (fun _ => mPts) N dg w H P x ω rr.1)
          (Claims.Bridge.Live.liveOut (fun _ => J) (fun _ => mPts) N dg w H P x ω (Claims.forkCoins j rr.1 rr.2)) := by
  rw [Claims.Bridge.Live.liveStrat, forkC_eqRec]
  exact ((Claims.Bridge.ex_fork_run H P (fun x => N x + 2) (fun x => sessionK dg J mPts (N x))
    (fun x => Claims.Bridge.Live.sessionIf (w x) dg (N x)) x ω
    (Claims.Bridge.Live.sessionLe (w x) dg (N x) _) j F).trans
    (Claims.Bridge.Live.forkX_sessionK (w x) dg (N x) F j.val _)).symm

/-! ## The fork event at a rep's later level -/

/-- Rep `r`'s recorded cap at index `ℓ`, its level-`ℓ + 1` cap, is `d`. -/
def CapAtL {D Sl : Type} (r ℓ : ℕ) (out : Model.OutCL 𝔽 D Sl) (d : Model.Cap D) : Prop :=
  (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ]? = some d

/-- **The fork event at rep `r`'s level `ℓ + 1`**: table 0's outputs in both runs record one cap `d` for it and open a
position of it to different rows. -/
def ForkEvL {D Sl : Type} [Inhabited D] [Inhabited Sl] (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule) (r ℓ : ℕ) (a a' : List (Model.OutCL 𝔽 D Sl)) :
    Prop :=
  ∃ d, CapAtL r ℓ (a.getD 0 dfltOutL) d ∧ CapAtL r ℓ (a'.getD 0 dfltOutL) d ∧
    ∃ p, Rewinding.ConflictAt (fun outs : List (Model.OutCL 𝔽 D Sl) =>
      opRL Hc E Lf sch r (ℓ + 1) d (outs.getD 0 dfltOutL)) (a, a') p

/-- Table 0's output opens some position at rep `r`'s level `ℓ + 1`. -/
def OpensAtL {D Sl : Type} [Inhabited D] [Inhabited Sl] (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule) (r ℓ : ℕ) (a : List (Model.OutCL 𝔽 D Sl)) :
    Prop :=
  ∃ d p, (opRL Hc E Lf sch r (ℓ + 1) d (a.getD 0 dfltOutL) p).isSome

theorem forkEvL_opens {D Sl : Type} [Inhabited D] [Inhabited Sl] (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule) (r ℓ : ℕ) (a a' : List (Model.OutCL 𝔽 D Sl))
    (h : ForkEvL Hc E Lf sch r ℓ a a') : OpensAtL Hc E Lf sch r ℓ a ∧ OpensAtL Hc E Lf sch r ℓ a' :=
  let ⟨d, _, _, p, h₁, h₂, _⟩ := h
  ⟨⟨d, p, h₁⟩, ⟨d, p, h₂⟩⟩

theorem not_opensAtL_nul {D Sl : Type} [Inhabited D] [Inhabited Sl] (Hc : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule) (r ℓ : ℕ)
    (cap : Model.Cap D) (pend : Model.PendL 𝔽 D × Model.PendL 𝔽 D) (outs : List (Model.OutCL 𝔽 D Sl))
    (h : outs.getD 0 dfltOutL = tableOutL Hc E Lf cap pend (nulOp (D := D) (Sl := Sl))) :
    ¬ OpensAtL Hc E Lf sch r ℓ outs := by
  rintro ⟨d, p, hs⟩
  rw [h, opRL_nul] at hs
  exact Bool.false_ne_true hs

/-! ## At the concrete simulator -/

open Flock FlockSoundness.Discharge
open Lean (Json)

/-- The decoders `plainAdv` reads: right, at the verdict at the draw, for accepting with no level-0 opening off `C`
unless it self-clashes. -/
abbrev PlainHDec {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
    DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAt I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j

/-- **The simulated strategy after `Commit` at the concrete simulator's tape `u`**: `simAt` at `plainSim P`'s live
strategy and its roots (`plainSim_root`), as `plainAdv` reads it. -/
noncomputable def plainSimAt {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDec L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) :=
  simAt execArith Refine.H512 Refine.enc512 Refine.leaf512 (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
    (plainVerdictAt I d C (plainRoot P x ω)) opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u)

/-- **The strategies' fork advantage at tape `u`**: `plainAdv`'s term, `Adv` at rep `r`'s level `ℓ + 1` against
`plainSimAt`. -/
noncomputable def plainAdvAt {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDec L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (u : plainTape x) : ℝ :=
  advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
    (plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- **The simulated fork's alignment at tape `u`**: against `plainSimAt`, the fork at rep `r`'s level-`ℓ + 1` table
node is at most the fork after `j` coins of `ForkEvL`. -/
def PlainAlignQ {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDec L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  plainAdvAt L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ u ≤
    forkC (fun a a' => ind (fun p : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) ×
        List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
      ForkEvL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val p.1 p.2) (a, a')) j
      _ (plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- **The live fork event at the concrete simulator**: the live outcomes at live coins `r₁` and `r₂` decode, at the
verdict at the draw, to outputs with `ForkEvL` at rep `r`'s level `ℓ + 1`. -/
def PlainLiveFork {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x)
    (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2) (ℓ : ℕ) : Prop :=
  LiveEv (Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (plainVerdictAt I d C (plainRoot P x ω)) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)))
    (ForkEvL Refine.H512 Refine.enc512 Refine.leaf512 sch r.val ℓ)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
      (ω, u) r₁)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
      (ω, u) r₂)

/-- **The live fork event gives `P`'s two accepting runs a collision**: wherever `PlainLiveFork` holds and `V false`
accepts `P` at `plainΦ x` at both live coins, the verifier's calls in the two runs collide. -/
def PlainForkQ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2)
    (ℓ : ℕ), PlainLiveFork P x ω st T hw I d opn sch C u r₁ r₂ r ℓ →
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₁))).out = .acc →
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₂))).out = .acc →
    Claims.TiedColl sha512 ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₁))).vLog ++
      (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₂))).vLog)

/-- **The simulated fork is aligned with the plain fork round**: at every plain statement in scope and its one table's
`fast100` schedule, at the plain decoders where the outputs are public (`plainOpn`, `plain_hdecAt`) and where they are
hidden under hm96 leaves (`plainOpnH`, `plain_hdecHAt`), at each tape the fork at rep `r`'s level-`ℓ + 1` table node
against the simulated strategy after `Commit` is at most the fork after `plainJ r ℓ` coins of `ForkEvL`
(`PlainAlignQ`). -/
def PlainForkAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false, ∀ C r ℓ u,
      PlainAlignQ L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r ℓ
        (plainJ (tables_head_one h) hsch r ℓ).val u) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"), ∀ C r ℓ u,
      PlainAlignQ L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecHAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r ℓ
        (plainJ (tables_head_one h) hsch r ℓ).val u)

/-- **The live fork event's collision**: at every plain statement in scope and its one table's `fast100` schedule, at
the plain decoders' openings where the outputs are public (`plainOpn`) and where they are hidden under hm96 leaves
(`plainOpnH`), the live fork event at the concrete simulator gives `P`'s two accepting runs a collision
(`PlainForkQ`). -/
def PlainForkClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (d : Json) (_ : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    (ins.read.I.tags.hiddenOutputs = false →
      PlainForkQ P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) ∧
    (ins.read.I.tags.hiddenOutputs = true → ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" →
      PlainForkQ P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)

open Classical in
/-- **The fork advantage from the alignment and the live fork's pair**: `plainAdv` is at most the simulated fork at the
plain fork round `plainJ r ℓ`, where at each tape the fork at the slot's node is at most the fork after `plainJ r ℓ`
coins of `ForkEvL` and the live fork event gives the deployed runs' pair event. -/
theorem plainAdv_le_of {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) (hL : plainN x + 2 = (V false).k x) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDec L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length)
    (halign : ∀ u, PlainAlignQ L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ
      (plainJ h0 hsch r ℓ).val u)
    (hpair : ∀ u r₁ r₂, PlainLiveFork P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) hw I d opn
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C u r₁ r₂ r ℓ.val →
      Claims.Flock.pairEv sha512 (V false) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂))) :
    plainAdv L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ ≤
      Claims.Flock.simForkPr sha512 (V false) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
        (plainJ h0 hsch r ℓ).val := by
  have hj : (plainJ h0 hsch r ℓ).val < plainN x + 2 := by have := (plainJ h0 hsch r ℓ).isLt; omega
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((halign u).trans (forkC_simAt_le execArith Refine.H512 Refine.enc512 Refine.leaf512
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt I d C (plainRoot P x ω)) opn
    (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
    (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u)
    (forkEvL_opens Refine.H512 Refine.enc512 Refine.leaf512 _ r.val ℓ.val)
    (not_opensAtL_nul Refine.H512 Refine.enc512 Refine.leaf512 _ r.val ℓ.val) _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStrat plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
    (ω, u) _ ⟨_, hj⟩)) ?_
  exact Claims.Ex_mono fun rr => ind_le_ind fun h => hpair u _ _ h

/-- **A live fork event's pair**: where `P'` aborts on rejected runs, the live fork event gives two accepting runs,
which collide by `PlainForkQ`. -/
theorem pairEv_of_forkQ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)} {sch : Accounting.Schedule}
    {C : Fin (wordLen false x) → List 𝔽}
    (habort : PlainSimAbort P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSim P x spec st)
      (plainΦ x spec st) (plainVerdictAt I d C (plainRoot P x ω)))
    (hq : PlainForkQ P x ω st T hw I d opn sch) {u : plainTape x} {r₁ r₂ : Fin (plainN x + 2) → LCoin 65536}
    {r : Fin 2} {ℓ : ℕ} (h : PlainLiveFork P x ω st T hw I d opn sch C u r₁ r₂ r ℓ) :
    Claims.Flock.pairEv sha512 (V false) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂)) := by
  obtain ⟨a, a', ha, ha', -⟩ := id h
  have h₁ := plain_acc_of_abort habort ha.1
  have h₂ := plain_acc_of_abort habort ha'.1
  exact ⟨h₁, h₂, hq C u r₁ r₂ r ℓ h h₁ h₂⟩

/-- **The fork bound from the alignment and the clash**: `PlainSimForkLe` from `PlainForkAlign` and `PlainForkClash`,
by `plainAdv_le_of` at the abort (`plainSimAbortAt`). -/
theorem plainSimForkLe_of {x : Claims.Msg} (ha : PlainForkAlign x) (hc : PlainForkClash x) : PlainSimForkLe x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω _ _ P ω
  have hL := plainN_add hset
  have hab := fun C : Fin (wordLen false x) → List 𝔽 =>
    plainSimAbortAt x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C (plainRoot P x ω)
  have ha' := ha s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω
  have hc' := hc s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω
  exact ⟨fun hh C _ r ℓ => plainAdv_le_of (tables_head_one h) hL L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _ _ _ _ P ω
      C r ℓ (ha'.1 hh C r ℓ) fun _ _ _ hq => pairEv_of_forkQ (hab C) (hc'.1 hh) hq,
    fun hh hl C _ r ℓ => plainAdv_le_of (tables_head_one h) hL L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _ _ _ _ P ω
      C r ℓ (ha'.2 hh hl C r ℓ) fun _ _ _ hq => pairEv_of_forkQ (hab C) (hc'.2 hh hl) hq⟩

/-! ## At `VK keep`

The concrete simulator's fork advantage at every `keep`: `plainSimK keep`, the runs of `VK keep` and the verdict at the
draw `plainVerdictAtK keep`. The originals read `plainVerdictAt`, which `plainVerdictAtK id` is not definitionally, so
they stay beside these. -/

/-- The decoders `plainAdvK keep` reads (`PlainHDec` at `plainVerdictAtK keep`). -/
abbrev PlainHDecK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
    DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAtK keep I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j

/-- **The simulated strategy after `Commit` at tape `u`, at `keep`** (`plainSimAt` at `plainSimK keep`). -/
noncomputable def plainSimAtK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) :=
  simAt execArith Refine.H512 Refine.enc512 Refine.leaf512 (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
    (plainVerdictAtK keep I d C (plainRoot P x ω)) opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u)

/-- **The strategies' fork advantage at tape `u`, at `keep`** (`plainAdvAt` at `plainSimAtK keep`). -/
noncomputable def plainAdvAtK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (u : plainTape x) : ℝ :=
  advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
    (plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- **The simulated fork's alignment at tape `u`, at `keep`** (`PlainAlignQ` at `plainSimAtK keep`). -/
def PlainAlignQK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  plainAdvAtK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ u ≤
    forkC (fun a a' => ind (fun p : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) ×
        List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
      ForkEvL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val p.1 p.2) (a, a')) j
      _ (plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- **The live fork event at the concrete simulator, at `keep`** (`PlainLiveFork` at `plainVerdictAtK keep` and
`plainSimK keep`). -/
def PlainLiveForkK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    {spec : SessionSpec} (st : Stmt) (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs)
    (d : Json) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x)
    (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2) (ℓ : ℕ) : Prop :=
  LiveEv (Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (plainVerdictAtK keep I d C (plainRoot P x ω)) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)))
    (ForkEvL Refine.H512 Refine.enc512 Refine.leaf512 sch r.val ℓ)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u) r₁)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u) r₂)

/-- **The live fork event gives `P`'s two accepting runs of `VK keep` a collision** (`PlainForkQ` at `keep`). -/
def PlainForkQK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec}
    (st : Stmt) (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2)
    (ℓ : ℕ), PlainLiveForkK keep P x ω st T hw I d opn sch C u r₁ r₂ r ℓ →
    (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₁))).out = .acc →
    (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₂))).out = .acc →
    Claims.TiedColl sha512 ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₁))).vLog ++
      (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₂))).vLog)

/-- **The simulated fork is aligned with the plain fork round, at `keep`** (`PlainForkAlign` at `PlainAlignQK keep`,
with the decoders `plain_hdecAtK keep` and `plain_hdecHAtK keep`). -/
def PlainForkAlignK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false, ∀ C r ℓ u,
      PlainAlignQK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C
        r ℓ (plainJ (x := x) (tables_head_one h) hsch r ℓ).val u) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"), ∀ C r ℓ u,
      PlainAlignQK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω
        C r ℓ (plainJ (x := x) (tables_head_one h) hsch r ℓ).val u)

/-- **The live fork event's collision, at `keep`** (`PlainForkClash` at `PlainForkQK keep`). -/
def PlainForkClashK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (d : Json) (_ : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    (ins.read.I.tags.hiddenOutputs = false →
      PlainForkQK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) ∧
    (ins.read.I.tags.hiddenOutputs = true → ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" →
      PlainForkQK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)

open Classical in
/-- **The fork advantage from the alignment and the live fork's pair, at `keep`** (`plainAdv_le_of`'s proof at
`plainSimK keep`). -/
theorem plainAdv_le_ofK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) (hL : plainN x + 2 = (V false).k x) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length)
    (halign : ∀ u, PlainAlignQK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ
      (plainJ (x := x) h0 hsch r ℓ).val u)
    (hpair : ∀ u r₁ r₂, PlainLiveForkK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) hw I d
      opn ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C u r₁ r₂ r ℓ.val →
      Claims.Flock.pairEv sha512 (VK keep) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂))) :
    plainAdvK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ ≤
      Claims.Flock.simForkPr sha512 (VK keep) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
        (plainJ (x := x) h0 hsch r ℓ).val := by
  have hj : (plainJ (x := x) h0 hsch r ℓ).val < plainN x + 2 := by have := (plainJ (x := x) h0 hsch r ℓ).isLt; omega
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((halign u).trans (forkC_simAt_le execArith Refine.H512 Refine.enc512 Refine.leaf512
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep I d C (plainRoot P x ω)) opn
    (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
    (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u)
    (forkEvL_opens Refine.H512 Refine.enc512 Refine.leaf512 _ r.val ℓ.val)
    (not_opensAtL_nul Refine.H512 Refine.enc512 Refine.leaf512 _ r.val ℓ.val) _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStrat plainN (plainDg spec) (plainW spec hw) sha512
    (plainSimK keep P x spec st) x (ω, u) _ ⟨_, hj⟩)) ?_
  exact Claims.Ex_mono fun rr => ind_le_ind fun h => hpair u _ _ h

/-- **A live fork event's pair, at `keep`** (`pairEv_of_forkQ` at `PlainSimAbortK keep`). -/
theorem pairEv_of_forkQK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {spec : SessionSpec} {st : Stmt} {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs}
    {d : Json} {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {sch : Accounting.Schedule} {C : Fin (wordLen false x) → List 𝔽}
    (habort : PlainSimAbortK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSimK keep P x spec st)
      (plainΦ x spec st) (plainVerdictAtK keep I d C (plainRoot P x ω)))
    (hq : PlainForkQK keep P x ω st T hw I d opn sch) {u : plainTape x} {r₁ r₂ : Fin (plainN x + 2) → LCoin 65536}
    {r : Fin 2} {ℓ : ℕ} (h : PlainLiveForkK keep P x ω st T hw I d opn sch C u r₁ r₂ r ℓ) :
    Claims.Flock.pairEv sha512 (VK keep) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂)) := by
  obtain ⟨a, a', ha, ha', -⟩ := id h
  have h₁ := plain_acc_of_abortK keep habort ha.1
  have h₂ := plain_acc_of_abortK keep habort ha'.1
  exact ⟨h₁, h₂, hq C u r₁ r₂ r ℓ h h₁ h₂⟩

/-- **The fork bound from the alignment and the clash, at `keep`**: `PlainSimForkLeK keep` from `PlainForkAlignK keep`
and `PlainForkClashK keep`, by `plainAdv_le_ofK` at the abort (`plainSimAbortAtK`). -/
theorem plainSimForkLe_ofK (keep : Bool → Bool) {x : Claims.Msg} (ha : PlainForkAlignK keep x)
    (hc : PlainForkClashK keep x) : PlainSimForkLeK keep x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω _ _ P ω
  have hL := plainN_add hset
  have hab := fun C : Fin (wordLen false x) → List 𝔽 =>
    plainSimAbortAtK keep x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C (plainRoot P x ω)
  have ha' := ha s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω
  have hc' := hc s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω
  exact ⟨fun hh C _ r ℓ => plainAdv_le_ofK keep (tables_head_one h) hL L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _ _
      _ _ P ω C r ℓ (ha'.1 hh C r ℓ) fun _ _ _ hq => pairEv_of_forkQK keep (hab C) (hc'.1 hh) hq,
    fun hh hl C _ r ℓ => plainAdv_le_ofK keep (tables_head_one h) hL L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _ _
      _ _ P ω C r ℓ (ha'.2 hh hl C r ℓ) fun _ _ _ hq => pairEv_of_forkQK keep (hab C) (hc'.2 hh hl) hq⟩

end FlockDeploy.KS
