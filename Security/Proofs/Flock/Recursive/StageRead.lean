import Proofs.Flock.Recursive.Stage

/-!
# Recursion: V*'s stages at the wires V* reads

`VFull` and `VDecodes` constrain every wire of the stages' commit strings. V*'s message rows have wires no unit reads
(`InnerRepCheck` pads a row port of `n` elements to `64 ⌈8 n / 64⌉` words and reads its first `128 n` bits), in no
unit's `io`, so `VFull` fails there. `VFullR` and `VDecodesR` are the two at the wires some unit reads (`ReadW`). The
transfer reads a commit string only at a wire where the layer disagrees with the openings' values, and a layer that
agrees with them on the read wires decodes alike (`VDecodesR`), so that wire is read, and the full draw draws its unit:

* `sessions_stage_le_read`, `outerV_stage_le_read`, `zk_sessions_recursive_read`, `zk_sessions_recursive_fork_read`:
  `sessions_stage_le`, `outerV_stage_le`, `zk_sessions_recursive_full` and `zk_sessions_recursive_fork` at `VFullR` and
  `VDecodesR`, with the same bounds;
* `vFullR_of_vFull`, `vDecodes_of_vDecodesR`: `VFull` gives `VFullR`, and `VDecodesR` gives `VDecodes`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- **A stage's transfer at V*'s sessions, at the read wires**: `sessions_stage_le` where every layer agreeing with the
values `wv` on the read wires of the commit strings `Sk j` satisfies `Q`, and each read wire of `Sk j` is a wire of
some unit. A layer failing `Q` disagrees with `wv` at a read wire, which every draw reads. -/
theorem sessions_stage_le_read
    (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ j S' R' i, rateZ (ZkHidden.planZHJ (Reg := Reg') (zs j).dj (zs j).y₀ S' R') i k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (qF qS : ℝ) (Sk : (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (hfull : ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j → ReadW Vs j g →
      ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u)
    (cs : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (hcs : ∀ j R' tr, (zs j).Writes R' tr → AcceptsZKR (zs j).I (zs j).own tr → ∀ pos ∈ Sk j,
      (zs j).vb.cs R' pos = cs j pos)
    (Q : ((j : Fin S) → (Vs j).Bits) → Prop)
    (wv : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c)
    (hQ : ∀ Xt : (j : Fin S) → (Vs j).Bits, (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j → ReadW Vs j g →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) → Q Xt)
    (s : Strategy (sessions zs dg N)) :
    prob (fun b => Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b)) (sessions zs dg N) s ≤
      sumS zs dg N (fun j τ => (zs j).slackIf dg N k Rw M ρ t' qF qS τ) s +
        ∑ j, ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
          exG (fun rs : VRec Vs Reg' dg N =>
            prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N (rs j).1) pos).g
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N (rs j).1) pos).s) (sessions zs dg N) s := by
  classical
  have hmono : ∀ b : VRec Vs Reg' dg N, Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b) →
      ∃ j, LiveAcceptsZKR (zs j).I (zs j).own (b j).2 ∧
        ∃ g, (posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j ∧ ReadW Vs j g) ∧
          (zs j).layer dg N k Rw ((zs j).ofBase dg N (b j).1) g ≠
            bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g := by
    rintro b ⟨hacc, hnq⟩
    by_contra hno
    refine hnq (hQ _ fun j g hg hgr => ?_)
    by_contra hne
    exact hno ⟨j, hacc j, g, ⟨hg, hgr⟩, hne⟩
  have hB : ∀ j τ₀, prob (fun a => LiveAcceptsZKR (zs j).I (zs j).own a ∧
        ∃ g, (posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j ∧ ReadW Vs j g) ∧
          (zs j).layer dg N k Rw ((zs j).ofBase dg N τ₀) g ≠
            bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g)
        ((zs j).game dg N) ((zs j).ofBase dg N τ₀) ≤
      (zs j).slackIf dg N k Rw M ρ t' qF qS ((zs j).ofBase dg N τ₀) +
        ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
          prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).g
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).s := by
    intro j τ₀
    by_cases hj : ∃ tr, (zs j).Writes ((zs j).ofBase dg N τ₀).1 tr ∧ AcceptsZKR (zs j).I (zs j).own tr
    swap
    · refine le_trans (le_of_eq ?_) (zero_le _)
      exact (zs j).prob_unwritten dg N (fun _ ha => ha.1) hj
    obtain ⟨tr₀, hw, hacc₀⟩ := hj
    unfold ZkOuter.slackIf
    split_ifs with hc
    · refine le_trans (prob_auditLive_mono _ _ ?_)
        (((zs j).candTransfer dg N hk hRw hM hρ (hr j) ht' _ hc (Sk j) (wv j)).trans
          (add_le_add le_rfl (Finset.sum_le_sum fun pos _ => ?_)))
      · rintro R' ω' tr ⟨hacc, g, ⟨hg, hgr⟩, hne⟩
        obtain ⟨u, hgu⟩ := hfull j g hg hgr
        refine ⟨hacc, u, execOS_subset_full_mem (Vs j).law.src ω' u, g, hgu, ?_⟩
        unfold CandBadZ
        rw [ZkOuter.vb_pos, ZkOuter.vb_bit]
        exact ⟨hg, hne⟩
      · by_cases hp : pos ∈ Sk j
        · rw [ZkOuter.reg_sim, hcs j _ tr₀ hw hacc₀ pos hp]
        · simp [candWeightK, hp]
    · exact (prob_le_one _ _ _).trans le_self_add
  refine ((prob_mono hmono _ _).trans (prob_seqC_exists_le S _ _
    (fun j r => LiveAcceptsZKR (zs j).I (zs j).own r.2 ∧
      ∃ g, (posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j ∧ ReadW Vs j g) ∧
        (zs j).layer dg N k Rw ((zs j).ofBase dg N r.1) g ≠
          bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g)
    (fun j τ₀ => (zs j).slackIf dg N k Rw M ρ t' qF qS ((zs j).ofBase dg N τ₀) +
      ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
        prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
          ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).g
          ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).s) hB s)).trans_eq ?_
  simp only [Finset.sum_add_distrib]
  rw [exG_add]
  refine congrArg₂ (· + ·) rfl ?_
  rw [exG_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [exG_sum]
  exact Finset.sum_congr rfl fun pos _ => exG_const_mul _ _ _ _

variable {Reg₂ : Type}

/-- **A stage's transfer at V*'s outer proof, at the read wires**: `sessions_stage_le_read` at the registration the
prover sends (`outerV_stage_le`'s proof). -/
theorem outerV_stage_le_read
    (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R₂ j S' R' i, rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R₂ j).dj (zs R₂ j).y₀ S' R') i k ≤ ρ) {t' : ℝ}
    (ht' : 0 ≤ t') (qF qS : ℝ) (Sk : (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (hfull : ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j → ReadW Vs j g →
      ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u)
    (cs : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (hcs : ∀ R₂ j R' tr, (zs R₂ j).Writes R' tr → AcceptsZKR (zs R₂ j).I (zs R₂ j).own tr → ∀ pos ∈ Sk j,
      (zs R₂ j).vb.cs R' pos = cs j pos)
    (Q : ((j : Fin S) → (Vs j).Bits) → Prop)
    (wv : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c)
    (hQ : ∀ Xt : (j : Fin S) → (Vs j).Bits, (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j → ReadW Vs j g →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) → Q Xt)
    (s : Strategy (outerV zs dg N)) :
    prob (fun o => AcceptsV zs o ∧ ¬ Q (layers (zs o.1) dg N k Rw o.2)) (outerV zs dg N) s ≤
      sumV zs dg N (fun R₂ j τ => (zs R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s +
        ∑ j, ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
          exG (fun rs : VRec Vs Reg' dg N =>
            prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
            ((zs s.1 j).trial dg N k Rw ((zs s.1 j).ofBase dg N (rs j).1) pos).g
            ((zs s.1 j).trial dg N k Rw ((zs s.1 j).ofBase dg N (rs j).1) pos).s) (sessions (zs s.1) dg N)
            (sessOf zs dg N s) := by
  rw [prob_outerV]
  exact sessions_stage_le_read (zs s.1) dg N hk hRw hM hρ (hr s.1) ht' qF qS Sk hfull cs (hcs s.1) Q wv hQ
    (sessOf zs dg N s)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

theorem vFullR_of_vFull {r : ℕ} {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    (h : VFull r Sk) : VFullR r Sk :=
  fun i hi j g hg _ => h i hi j g hg

theorem vDecodes_of_vDecodesR {wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg}
    {same : ℕ → List I.Coin → I.Msg → I.Msg → Prop} {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg}
    (h : VDecodesR wmsg same Sk msg) : VDecodes wmsg same Sk msg :=
  fun i hi ρs Xt v hX => h i hi ρs Xt v fun j g hg _ => hX j g hg

/-- **The recursive audit is sound at the read wires**: `zk_sessions_recursive_full` at `VDecodesR` and `VFullR`, its
transfer from `outerV_stage_le_read`, with the same bound. -/
theorem zk_sessions_recursive_read
    (hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : Recursion.VBridge zs dg N k Rw x wmsg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodesR wmsg same Sk msg) (hfull : VFullR I.r Sk)
    (q : ℕ → ℝ)
    (hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  refine zk_sessions_recursive_upto hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x wmsg hbr same hsame
    (obS Vs) (cmtS cmt) (msgS msg) (fun _ R ω t s => trialV (zs R ω t) dg N k Rw s)
    (fun _ R ω t s => sumV (zs R ω t) dg N (fun R₂ j τ => (zs R ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s)
    (weightS Vs Sk ρ) q (fun i hi ω => AllLeaves.of_forall (fun t s' v e he => ?_) _) hCR hin
  refine (outerV_stage_le_read (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' qF qS (Sk i) (hfull i hi)
    (fun j pos => cmt j e.1 pos) (fun R₂ j R' tr hw hj pos hp => hcmt i σ.1 ω t e he R₂ j R' tr hw hj pos hp)
    (fun Xs => Option.Rel (same i (t.map Prod.snd)) (wmsg Xs)[i]? (some (msgS msg i v)))
    (fun j p => projV j (v ⟨j, p⟩)) (fun Xt hXt => hdec i hi _ Xt (fun j p => projV j (v ⟨j, p⟩)) hXt) _).trans
    (add_le_add le_rfl ?_)
  rw [Fintype.sum_sigma]
  exact Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun p _ =>
    mul_le_mul' le_rfl (trialS_le (zs σ.1 ω t _) dg N k Rw _ j p _ _)

/-- **The recursive audit is sound at the read wires, in fork form**: `zk_sessions_recursive_fork` at `VDecodesR` and
`VFullR`, from `zk_sessions_recursive_read` at no finder budgets, with the same disjuncts and bound. -/
theorem zk_sessions_recursive_fork_read
    (hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : Recursion.VBridge zs dg N k Rw x wmsg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodesR wmsg same Sk msg) (hfull : VFullR I.r Sk)
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases h1 : RoundFork zs dg N k Rw σ cmt
  · exact Or.inl h1
  by_cases h2 : SessionFork zs dg N k Rw M σ
  · exact Or.inr (Or.inl h2)
  refine Or.inr (Or.inr ?_)
  have hno := allAt_of_not_exAt h2
  have hB : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ 0 0 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.boundIf_le _ _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  have hsl : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.slackIf_le _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  refine (zk_sessions_recursive_read hA3 zs dg N hk hRw hM hρ hr le_rfl 0 0 0 0 σ x wmsg hbr same hsame Sk cmt msg
    hcmt hdec hfull (fun _ => 0) (fun i hi ω pos => Finder.cr_of_not_finds (fun hf => h1 ⟨i, hi, ω, pos, hf⟩) 0)
    hin).trans (add_le_add (add_le_add hB ?_) le_rfl)
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, Real.sqrt_zero, ENNReal.ofReal_zero,
    mul_zero, add_zero]
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  exact mul_le_mul' le_rfl hsl

end FlockSoundness.Discharge.Recursion
