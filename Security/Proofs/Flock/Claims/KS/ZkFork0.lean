import Proofs.Flock.Claims.KS.ZkClashL
import Proofs.Flock.Claims.KS.ForkSim

/-!
# The masked simulator's fork bound at level 0

`ZkSimForkLe` bounds the masked simulator's fork advantage at each event of table `j` (`zkAdvSim`) by the simulated
fork at the event's round. At level 0 (`.inr ()`, round `0`) it holds outright (`zkSimForkLe₀`):

* `adv₀SZ` is two independent runs after `Commit`, the fork after `0` coins (`forkC_zero`); against the simulated
  strategy after `Commit` at table `j` it is at most the live fork after `0` coins of the decoded event
  (`adv₀SZ_simAtJZ_le`, by `Claims.Bridge.Sim.forkC_le`, the dead branches' empty openings opening nothing at level 0,
  `op₀Z_nul`), which is the deployed runs at independent coins (`forkC_liveStratZ`);
* two live outcomes decoding to outputs that open a level-0 position of table `j` to different rows are two
  accepting runs of `P` (`zkSimAbortAt`) whose views open the runs' level-0 rows (`zkSimRunAt`, `zkViewRows₀`), so the
  rows differ at one position and collide in the two runs' logs (`zk_rowClash_tiedJ₂`; `zkPair₀`).

`ZkSimForkLeOn x S` is `ZkSimForkLe` at the events in `S`. `ZkSimForkLe` is its cap and `τ` events (`IsCapTauZ`), its
pads events (`IsPadsEvZ`) and level 0 (`zkSimForkLe_of_kinds`), so masked KS, the recursion's premise and its KS rest
on the cap/`τ` and pads fork bounds and the pads clash (`zk_KS_of_kinds`, `zk_tableBoundJOnAccR_of_kinds`,
`zk_KSFlock_of_kinds`).
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Sim

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable simSession)
open FlockSoundness.Discharge.Composed (outJ repJ sessionZC_eq_modelSession)
open Claims.Bridge.Live

/-- **Two live outcomes decoding to a pair with `Ev`** under `R`. -/
def LiveEvZ {β α : Type} (R : β → α → Prop) (Ev : α × α → Prop) (b b' : β) : Prop :=
  ∃ a a', R b a ∧ R b' a' ∧ Ev (a, a')

theorem ind_le_indZ {α β : Type} {p : α → Prop} {q : β → Prop} {a : α} {b : β} (h : p a → q b) :
    ind p a ≤ ind q b := by
  unfold ind
  split_ifs with h1 h2 <;> first | exact absurd (h h1) h2 | norm_num

/-- **The empty masked openings open nothing at level 0.** -/
theorem op₀Z_nul {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts : ℕ} (H : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : TabZK 𝔽 mPts) (cap : Model.Cap D)
    (out : OutZC 𝔽 D Sl) (h : out.opens = nulOpZ) : op₀Z H E Lf T cap out = fun _ => none := by
  funext p
  unfold op₀Z
  rw [List.filter_eq_nil_iff.2 fun x hx => ?_]
  · rfl
  · simp only [decide_eq_true_eq, not_and]
    refine fun _ => not_verifiedZ_nil H E Lf cap (width₀_pos T) _ ?_
    unfold opens₀Z at hx
    rcases List.mem_append.1 hx with hx | hx <;> exact slotOpensZ_row_nil h hx

/-- A level-0 opening is a row some level-0 opening of the output has at its position. -/
theorem op₀Z_some {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts : ℕ} {H : List UInt8 → D}
    {E : Merkle.Enc (List 𝔽) D} {Lf : Merkle.Leaf (List 𝔽) Sl D H} {T : TabZK 𝔽 mPts} {cap : Model.Cap D}
    {out : OutZC 𝔽 D Sl} {p v} (h : op₀Z H E Lf T cap out p = some v) :
    ∃ y ∈ opens₀Z T.sch out, y.1 = p.val ∧ v = fun lane => y.2.1.getD lane.val 0 := by
  classical
  unfold op₀Z at h
  obtain ⟨y, hy, rfl⟩ := Option.map_eq_some_iff.1 h
  have hm := List.mem_of_mem_head? hy
  rw [List.mem_filter, decide_eq_true_eq] at hm
  exact ⟨y, hm.1, hm.2.1, rfl⟩

/-- **The live fork after `j` coins is the deployed runs' fork at round `j`**: two runs at coins sharing their first
`j` (`Claims.forkCoins`). -/
theorem forkC_liveStratZ {nMax J mPts : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
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

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
theorem forkC_afterJZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (τ : Strategy (sessionZC A Hs ts)) {caps : Fin ts.length → Model.Cap D} (hc : τ.1 = caps) (j : Fin ts.length)
    {T : TabZK 𝔽 mPts} (hT : ts.get j = T) (F : List (OutZC 𝔽 D Sl) → List (OutZC 𝔽 D Sl) → ℝ) (t : ℕ) :
    forkC F t _ (afterJZ A Hs ts τ hc j hT) = forkC F t (sessionZC A Hs ts) τ := by
  subst hc hT
  exact (forkC_cast (sessAfterZC_split A Hs ts τ.1 j) F t τ.2).trans (forkC_send F t τ).symm

variable [Inhabited D] [Inhabited Sl]

theorem forkC_simSessZV (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) (F : List (OutZC 𝔽 D Sl) → List (OutZC 𝔽 D Sl) → ℝ)
    (t : ℕ) :
    forkC F t (sessionZC A Hs ts) (simSessZV A Hs ts ok pL opn dg dec one fin capOf N hdec Pr) =
      forkC F t _ ((Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOpZ (D := D) (Sl := Sl)))
        (simSession (outJ Hs ts) ok dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
          (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec)).strategy Pr) := by
  unfold simSessZV
  exact forkC_eqRec _ _ _ _

open Classical in
/-- **A fork after `t` coins against the simulated masked strategy after `Commit` at table `j`** is at most the live
fork after `t` coins of the decoded event, for an event `Ev` of two model outcomes whose each run is in an event `Q` no
outputs at the empty openings have, on reps' outcomes of their games at some link points. -/
theorem forkC_simAtJZ_le_out (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀)
    (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T)
    {Ev : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) → Prop} {Q : List (OutZC 𝔽 D Sl) → Prop}
    (hQ : ∀ y, Ev y → Q y.1 ∧ Q y.2)
    (hnul : ∀ (caps : Fin ts.length → Model.Cap D) (pts : Fin 2 → Fin mPts → 𝔽) outs, (∀ i : Fin ts.length,
      ∃ pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl, Out (repJ A Hs ts i pts 0) pend.1 ∧ Out (repJ A Hs ts i pts 1) pend.2 ∧
        outs.getD i dfltOutZC = outJ Hs ts i (caps i) pend nulOpZ) → ¬ Q outs) (t : ℕ) :
    forkC (fun a a' => ind Ev (a, a')) t _
        (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT) ≤
      forkC (fun b b' => ind (fun p : LiveOutJ ts.length mPts × LiveOutJ ts.length mPts =>
          LiveEvZ (Claims.Bridge.RetOK (outJ Hs ts) pL opn dec one fin capOf (repJ A Hs ts)) Ev p.1 p.2) (b, b')) t
        (liveSession dg ts.length mPts N) Pr := by
  unfold simAtJZ
  rw [forkC_afterJZ, forkC_simSessZV]
  have hall := Claims.Bridge.Sim.all_mapDead (Claims.Bridge.answerWith (nulOpZ (D := D) (Sl := Sl)))
    (D' := fun g σ => prob Q g σ = 0)
    (fun _ σ ⟨caps, pts, _, hg, hres, e⟩ => by
      subst e
      exact Claims.Bridge.prob_posOf_answerWith_out _ _ caps hg dfltOutZC (fun outs h => hnul caps pts outs fun i =>
        let ⟨pend, h0, h1, he⟩ := h i
        ⟨pend, (hres i 0).out h0, (hres i 1).out h1, he⟩) σ)
    _ (Claims.Bridge.all_simSession (outJ Hs ts) ok dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
      (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec)
  have hFQ : ∀ a a', ind Ev (a, a') ≤ ind Q a := fun a a' => ind_le_indZ fun h1 => (hQ _ h1).1
  have hFQ' : ∀ a a', ind Ev (a, a') ≤ ind Q a' := fun a a' => ind_le_indZ fun h1 => (hQ _ h1).2
  exact Claims.Bridge.Sim.forkC_le hFQ hFQ' (fun b b' => ind_nonneg _ _)
    (fun b a b' a' hb hb' => ind_le_indZ fun h1 => ⟨a, a', hb, hb', h1⟩) _ hall t Pr

open Classical in
/-- **A fork after `t` coins against the simulated masked strategy after `Commit` at table `j`** is at most the live
fork after `t` coins of the decoded event, for an event `Ev` of two model outcomes whose each run is in an event `Q` no
outputs at the empty openings have. -/
theorem forkC_simAtJZ_le (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀)
    (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T)
    {Ev : List (OutZC 𝔽 D Sl) × List (OutZC 𝔽 D Sl) → Prop} {Q : List (OutZC 𝔽 D Sl) → Prop}
    (hQ : ∀ y, Ev y → Q y.1 ∧ Q y.2)
    (hnul : ∀ (caps : Fin ts.length → Model.Cap D) outs, (∀ i : Fin ts.length, ∃ pend,
      outs.getD i dfltOutZC = outJ Hs ts i (caps i) pend nulOpZ) → ¬ Q outs) (t : ℕ) :
    forkC (fun a a' => ind Ev (a, a')) t _
        (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT) ≤
      forkC (fun b b' => ind (fun p : LiveOutJ ts.length mPts × LiveOutJ ts.length mPts =>
          LiveEvZ (Claims.Bridge.RetOK (outJ Hs ts) pL opn dec one fin capOf (repJ A Hs ts)) Ev p.1 p.2) (b, b')) t
        (liveSession dg ts.length mPts N) Pr :=
  forkC_simAtJZ_le_out A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT hQ
    (fun caps _ outs h => hnul caps outs fun i => let ⟨pend, _, _, he⟩ := h i; ⟨pend, he⟩) t

open Classical in
/-- **`Adv₀` against the simulated masked strategy** is at most the live fork after `0` coins, two independent live
runs, of two decoded outcomes opening a level-0 position of table `j` to different rows. -/
theorem adv₀SZ_simAtJZ_le (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : Fin ts.length → OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) ok pL opn dec one fin capOf (repJ A Hs ts) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg ts.length mPts N)) (hP : Pr.1 = rootB₀)
    (j : Fin ts.length) {T : TabZK 𝔽 mPts} (hT : ts.get j = T) :
    adv₀SZ A Hs T _ _ (capOf rootB₀ j) (simAtJZ A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT) ≤
      forkC (fun b b' => ind (fun p : LiveOutJ ts.length mPts × LiveOutJ ts.length mPts =>
          LiveEvZ (Claims.Bridge.RetOK (outJ Hs ts) pL opn dec one fin capOf (repJ A Hs ts))
            (fun y => ∃ q, Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
              op₀Z Hs.H Hs.E Hs.Lf T (capOf rootB₀ j) (outs.getD j dfltOutZC)) y q) p.1 p.2) (b, b')) 0
        (liveSession dg ts.length mPts N) Pr := by
  have e := take_gamesZC_length A Hs ts (capOf rootB₀) j
  unfold adv₀SZ
  rw [e, ← forkC_zero]
  exact forkC_simAtJZ_le A Hs ts ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP j hT
    (Q := fun a => ∃ q, (op₀Z Hs.H Hs.E Hs.Lf T (capOf rootB₀ j) (a.getD j dfltOutZC) q).isSome)
    (fun _ ⟨q, h₁, h₂, _⟩ => ⟨⟨q, h₁⟩, ⟨q, h₂⟩⟩)
    (fun caps outs h ⟨q, hq⟩ => by
      obtain ⟨pend, hp⟩ := h j
      rw [hp, op₀Z_nul Hs.H Hs.E Hs.Lf T _ (outJ Hs ts j (caps j) pend nulOpZ) rfl] at hq
      simp at hq) 0

end Sim

section Level0

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

namespace Gen

variable [S : ZkSetup]

/-- **A decoded live outcome of the masked simulator is an accepting run's view**: where the live verdict holds, `P`'s
run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of the record it decides (`zkSimRunAt`), and
the decoded output of table `j` opens the run's level-0 rows (`zkViewRows₀`). -/
theorem zkAccRows₀ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (ha : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r') a) :
    (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc ∧
      (opens₀Z (zkTabs h0 L hz hk hlay j).sch (a.getD j dfltOutZC)).map (fun y => (y.1, y.2.1)) =
        zkRawRowsJ (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).seen j := by
  have hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc := by
    by_contra hn
    exact zkSimAbortAt x s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r' hn ha.1
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, -⟩ := acc_decide hacc
  have hv := zkSimRunAt x s hs hin ins spec sts hset h0 L hz hk hlay Ω P ω u r' hacc s' spec' sts' link c₁ mids f hact
  rw [hv] at ha
  exact ⟨hacc, zkViewRows₀ hs hin hset h0 L hz hk hlay hd hj hacc hact ha⟩

end Gen

/-- **A decoded live outcome of the masked simulator is an accepting run's view**: where the live verdict holds, `P`'s
run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of the record it decides (`zkSimRunAt`), and
the decoded output of table `j` opens the run's level-0 rows (`zkViewRows₀`). -/
theorem zkAccRows₀ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (ha : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r') a) :
    (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc ∧
      (opens₀Z (zkTabs h0 L hz hk hlay j).sch (a.getD j dfltOutZC)).map (fun y => (y.1, y.2.1)) =
        zkRawRowsJ (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).seen j :=
  Gen.zkAccRows₀ (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (u := u) (r' := r') (a := a) (ha := ha)

namespace Gen

variable [S : ZkSetup]

/-- **The live level-0 fork event's pair**: two live outcomes of the masked simulator at one tape decoding to outputs
that open a level-0 position of table `j` to different rows are two accepting runs of `P` at `zkΦ`'s coins whose
level-0 rows differ at that position, so they collide in the two runs' logs (`zk_rowClash_tiedJ₂`). -/
theorem zkPair₀ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    {u : zkTape x} {r₁ r₂ : Fin (zkN x + 2) → LCoin 65536}
    (h : LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!)
        (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
      (fun y => ∃ q, Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 Refine.Dig (List UInt8)) =>
        op₀Z ZkExec.hs512.H ZkExec.hs512.E ZkExec.hs512.Lf (zkTabs h0 L hz hk hlay j)
          (Exec.capOfRoot (zkRoot P x ω)[j]!) (outs.getD j dfltOutZC)) y q)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₁)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₂)) :
    Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂)) := by
  obtain ⟨a, a', ha, ha', q, hs₁, hs₂, hne⟩ := h
  dsimp only at hs₁ hs₂ hne
  obtain ⟨hacc₁, hrows₁⟩ := zkAccRows₀ hs hin hset h0 L hz hk hlay hd hj hC u r₁ ha
  obtain ⟨hacc₂, hrows₂⟩ := zkAccRows₀ hs hin hset h0 L hz hk hlay hd hj hC u r₂ ha'
  obtain ⟨v₁, hv₁⟩ := Option.isSome_iff_exists.1 hs₁
  obtain ⟨v₂, hv₂⟩ := Option.isSome_iff_exists.1 hs₂
  obtain ⟨y₁, hy₁, hq₁, rfl⟩ := op₀Z_some hv₁
  obtain ⟨y₂, hy₂, hq₂, rfl⟩ := op₀Z_some hv₂
  have hm₁ : (y₁.1, y₁.2.1) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₁))).seen j := by
    rw [← hrows₁]
    exact List.mem_map_of_mem hy₁
  have hm₂ : (y₁.1, y₂.2.1) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r₂))).seen j := by
    rw [← hrows₂, hq₁, ← hq₂]
    exact List.mem_map_of_mem hy₂
  refine ⟨hacc₁, hacc₂, zk_rowClash_tiedJ₂ hacc₁ hacc₂ hm₁ hm₂ fun e => hne ?_⟩
  rw [hv₁, hv₂, e]

end Gen

/-- **The live level-0 fork event's pair**: two live outcomes of the masked simulator at one tape decoding to outputs
that open a level-0 position of table `j` to different rows are two accepting runs of `P` at `zkΦ`'s coins whose
level-0 rows differ at that position, so they collide in the two runs' logs (`zk_rowClash_tiedJ₂`). -/
theorem zkPair₀ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    {u : zkTape x} {r₁ r₂ : Fin (zkN x + 2) → LCoin 65536}
    (h : LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!)
        (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
      (fun y => ∃ q, Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 Refine.Dig (List UInt8)) =>
        op₀Z ZkExec.hs512.H ZkExec.hs512.E ZkExec.hs512.Lf (zkTabs h0 L hz hk hlay j)
          (Exec.capOfRoot (zkRoot P x ω)[j]!) (outs.getD j dfltOutZC)) y q)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₁)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₂)) :
    Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂)) :=
  Gen.zkPair₀ (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (u := u) (r₁ := r₁) (r₂ := r₂) (h := h)

/-- Level 0's event. -/
def IsLevel0Z {Lv : ℕ} : Claims.Flock.IZW Lv → Prop
  | .inr () => True
  | .inl _ => False

/-- A rep's pads event. -/
def IsPadsEvZ {Lv : ℕ} : Claims.Flock.IZW Lv → Prop
  | .inl (.inr (.inl _)) => True
  | _ => False

/-- A rep's cap event at a level past 0, or its `τ` event. -/
def IsCapTauZ {Lv : ℕ} : Claims.Flock.IZW Lv → Prop
  | .inl (.inl _) => True
  | .inl (.inr (.inr _)) => True
  | _ => False

open Classical in
/-- **`ZkSimForkLe` at the events in `S`.** -/
def ZkSimForkLeOn (x : Claims.Msg) (S : (Lv : ℕ) → Claims.Flock.IZW Lv → Prop) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length, S _ i →
      zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C i ≤
        Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω
          (zkRd hset h0 L hz hk hlay j i).val

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **`ZkSimForkLe` at the events in `S`.** -/
def ZkSimForkLeOn (x : Claims.Msg) (S : (Lv : ℕ) → Claims.Flock.IZW Lv → Prop) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length, S _ i →
      zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C i ≤
        Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω
          (zkRd hset h0 L hz hk hlay j i).val

/-- **The masked simulator's fork bound at level 0**: `Adv₀` after `Commit` against the simulated strategy is at most
two independent live runs decoding to a level-0 clash (`adv₀SZ_simAtJZ_le`, `forkC_liveStratZ`), which are two
accepting runs of `P` colliding in their logs (`zkPair₀`), the simulated fork at round `0`. -/
theorem zkSimForkLe₀ (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsLevel0Z i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with _ | ⟨⟩
  · exact False.elim hi
  rw [show (zkRd hset h0 L hz hk hlay j (.inr ())).val = 0 from rfl]
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine (adv₀SZ_simAtJZ_le execArith ZkExec.hs512 _
    (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C) (AccTxJ ins.read.I d j (zkRoot P x ω) C)
    (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
    (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω) _
    (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
    (tabsJ_get _ _ _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStratZ zkN spec.dg
    (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
    (zkSim P x spec sts) x (ω, u) _ ⟨0, by omega⟩)) ?_
  exact Claims.Ex_mono fun rr => ind_le_indZ fun h => zkPair₀ hs hin hset h0 L hz hk hlay hd hj hC h

end Gen

/-- **The masked simulator's fork bound at level 0**: `Adv₀` after `Commit` against the simulated strategy is at most
two independent live runs decoding to a level-0 clash (`adv₀SZ_simAtJZ_le`, `forkC_liveStratZ`), which are two
accepting runs of `P` colliding in their logs (`zkPair₀`), the simulated fork at round `0`. -/
theorem zkSimForkLe₀ (x : Claims.Msg) : ZkSimForkLeOn x fun _ i => IsLevel0Z i :=
  Gen.zkSimForkLe₀ (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **`ZkSimForkLe` from its cap and `τ` events and its pads events**, level 0's holding (`zkSimForkLe₀`). -/
theorem zkSimForkLe_of_kinds {x : Claims.Msg} (hct : ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ZkSimForkLeOn x fun _ i => IsPadsEvZ i) : ZkSimForkLe x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i
  rcases i with ((_ | (_ | _)) | ⟨⟩)
  · exact hct s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial
  · exact hp s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial
  · exact hct s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial
  · exact zkSimForkLe₀ x s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ trivial

end Gen

/-- **`ZkSimForkLe` from its cap and `τ` events and its pads events**, level 0's holding (`zkSimForkLe₀`). -/
theorem zkSimForkLe_of_kinds {x : Claims.Msg} (hct : ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ZkSimForkLeOn x fun _ i => IsPadsEvZ i) : ZkSimForkLe x :=
  Gen.zkSimForkLe_of_kinds (S := S₀) (x := x) (hct := hct) (hp := hp)

end Level0

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **The deployed masked verifier: KS, from the cap/`τ` and pads fork bounds and the pads clash**, at SHA-512 at the
trimmed record at the deployed relation: `zk_KS_of_pads` at `zkSimForkLe_of_kinds`. -/
theorem zk_KS_of_kinds (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_pads (fun x hno => zkSimForkLe_of_kinds (hct x hno) (hp x hno)) hpads

end Gen

/-- **The deployed masked verifier: KS, from the cap/`τ` and pads fork bounds and the pads clash**, at SHA-512 at the
trimmed record at the deployed relation: `zk_KS_of_pads` at `zkSimForkLe_of_kinds`. -/
theorem zk_KS_of_kinds (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_of_kinds (S := S₀) (hct := hct) (hp := hp) (hpads := hpads)

end FlockDeploy.KS

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **Every masked table's bound with escapes on accepted runs counted per round**, at full-width columns, from the
cap/`τ` and pads fork bounds and the view's pads clash at the statements some run accepts
(`zk_tableBoundJOnAccR_of_pads`): `zk_KSFlockJR`'s premise. -/
theorem zk_tableBoundJOnAccR_of_kinds
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJOnAccR_of_pads (fun x hno => zkSimForkLe_of_kinds (hct x hno) (hp x hno)) hpads

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, at every number of
tables, from the cap/`τ` and pads fork bounds and the view's pads clash at the statements some run accepts
(`zk_KSFlock_of_pads`). -/
theorem zk_KSFlock_of_kinds (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  zk_KSFlock_of_pads D DO opening hF (fun x hno => zkSimForkLe_of_kinds (hct x hno) (hp x hno)) hpads hRef hRounds
    hN hε hRd hS hR hV hT hL hO hAcc hN0

end FlockDeploy.KS
