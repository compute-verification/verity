import Proofs.Flock.Recursive.StageRead
import Proofs.Flock.Recursive.Statements

/-!
# The stages' transfer at the wires V* reads

`Flock.Assumptions.VBridge` takes `VFull` and `VDecodes` at every wire of the stages' commit strings. V*'s message rows
have wires no unit reads (`InnerRepCheck` pads a row port of `n` elements to `64 ⌈8 n / 64⌉` words and reads its first
`128 n` bits, `InnerRepCheck.lean:24,39-44`), in no unit's `io`, so `VFull` fails there. `VFullR` and `VDecodesR` are
the two at the wires some unit reads (`ReadW`), and `VBridgeRead` is `VBridge` with them. The transfer reads a
commit string only at a wire where the layer disagrees with the openings' values, and a layer that agrees with them on
the read wires decodes alike (`VDecodesR`), so that wire is read, and `VFullR` draws it:

* `sessions_stage_le_read`, `outerV_stage_le_read`, `zk_sessions_recursive_read`: `sessions_stage_le`,
  `outerV_stage_le` and `zk_sessions_recursive_full` at `VFullR` and `VDecodesR`, with the same bounds;
* `RecursiveSoundRead`: `Flock.Guarantees.RecursiveSound` at `VBridgeRead`, proved by `recursiveSound_read`;
* `vFullR_of_vFull`, `vDecodes_of_vDecodesR`: `VFull` gives `VFullR`, and `VDecodesR` gives `VDecodes`.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

variable (Vs) in
/-- **Wire `g` of session `j` is read**: a committed wire of some unit. -/
def ReadW (j : Fin S) (g : Fin ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).circuit
    (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).N) : Prop :=
  ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u

/-- **A stage's transfer at V*'s sessions, at the read wires**: `sessions_stage_le` where every layer agreeing with the
values `wv` on the read wires of the commit strings `Sk j` satisfies `Q`, and each read wire of `Sk j` is a wire of a
unit drawn in full. A layer failing `Q` disagrees with `wv` at a read wire, which every draw reads. -/
theorem sessions_stage_le_read
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ j S' R' i, rateZ (planZAt (Reg := Reg') (zs j).dj (zs j).y₀ S' R') i k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (qF qS : ℝ) (Sk : (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (hfull : ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j → ReadW Vs j g →
      ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u ∧
        (Law.stratum (Vs j).law.σs ((Vs j).law.σs u)).card ≤ (Vs j).law.ks ((Vs j).law.σs u))
    (cs : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (hcs : ∀ j, ∀ pos ∈ Sk j, ∀ R' : Reg', (zs j).vb.cs R' pos = cs j pos)
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
    unfold ZkOuter.slackIf
    split_ifs with hc
    · refine le_trans (prob_auditLive_mono _ _ ?_)
        (((zs j).candTransfer (Vs j).law.St (Vs j).law.Kd (Vs j).law.pop (Vs j).law.Lb (Vs j).law.src dg N hk hRw
          hM hρ (hr j) ht' _ hc (Sk j) (wv j)).trans (add_le_add le_rfl (Finset.sum_le_sum fun pos _ => ?_)))
      · rintro R' ω' tr ⟨hacc, g, ⟨hg, hgr⟩, hne⟩
        obtain ⟨u, hgu, hu⟩ := hfull j g hg hgr
        refine ⟨hacc, u, execOS_full_mem (Vs j).law.σs (Vs j).law.ks (Vs j).law.St (hS j) (Vs j).law.Kd
          (Vs j).law.pop (Vs j).law.src ω' hu, g, hgu, ?_⟩
        unfold CandBadZ
        rw [ZkOuter.vb_pos, ZkOuter.vb_bit]
        exact ⟨hg, hne⟩
      · by_cases hp : pos ∈ Sk j
        · rw [hcs j pos hp]
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
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R₂ j).dj (zs R₂ j).y₀ S' R') i k ≤ ρ) {t' : ℝ}
    (ht' : 0 ≤ t') (qF qS : ℝ) (Sk : (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (hfull : ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j → ReadW Vs j g →
      ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u ∧
        (Law.stratum (Vs j).law.σs ((Vs j).law.σs u)).card ≤ (Vs j).law.ks ((Vs j).law.σs u))
    (cs : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (hcs : ∀ R₂ j, ∀ pos ∈ Sk j, ∀ R' : Reg', (zs R₂ j).vb.cs R' pos = cs j pos)
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
  exact sessions_stage_le_read hS (zs s.1) dg N hk hRw hM hρ (hr s.1) ht' qF qS Sk hfull cs (hcs s.1) Q wv hQ
    (sessOf zs dg N s)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **`VDecodes` at the read wires**: layers whose bits on the read wires of each `Sk i j` are those of the values `v j`
hold a round-`i` message the inner verifier reads as `msg i v`, at any coins. -/
def VDecodesR (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  ∀ i < I.r, ∀ (ρs : List I.Coin) (Xt : (j : Fin S) → (Vs j).Bits)
    (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c),
    (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j → ReadW Vs j g →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (v j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) →
    Option.Rel (same i ρs) (wmsg Xt)[i]? (some (msg i v))

/-- **`VFull` at the read wires**: each read wire of `Sk i j` (`i < r`) is a wire of a unit whose stratum session `j`'s
law draws in full. -/
def VFullR (r : ℕ) (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) : Prop :=
  ∀ i < r, ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j → ReadW Vs j g →
    ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u ∧
      (Law.stratum (Vs j).law.σs ((Vs j).law.σs u)).card ≤ (Vs j).law.ks ((Vs j).law.σs u)

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
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
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
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin :=
  FlockSoundness.Discharge.Recursion.zk_sessions_recursive_read hS hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x
    wmsg hbr same hsame Sk cmt msg hcmt hdec hfull q hCR hin

/-- **`v-bridge` at the read wires**: `Flock.Assumptions.VBridge` with `VDecodes` and `VFull` at the wires some unit
reads (`VDecodesR`, `VFullR`): V*'s circuits accept only an accepted inner transcript, their public files register the
firewall's commitments at the commit strings `Sk i j`, layers whose bits on the read wires of `Sk i j` are the values
`v j`'s hold `msg i v` as their round-`i` message, and each read wire of `Sk i j` is a wire of a unit whose stratum
session `j`'s law draws in full. -/
def VBridgeRead (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  Recursion.VBridge zs dg N k Rw x wmsg ∧ VCommits zs Sk cmt ∧ VDecodesR wmsg (fun _ _ a b => a = b) Sk msg ∧
    VFullR I.r Sk

/-- **The recursive audit is sound, at `VBridgeRead`**: `Flock.Guarantees.RecursiveSound` with `VBridgeRead` for
`Flock.Assumptions.VBridge`, and the same bound. -/
def RecursiveSoundRead : Prop :=
  ∀ {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (_hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (_hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (_ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (_hbr : VBridgeRead zs dg N k Rw x wmsg Sk cmt msg)
    (q : ℕ → ℝ)
    (_hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    {εin : ℝ≥0∞} (_hin : InnerSound I εin),
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin

/-- **The recursive audit is sound at `VBridgeRead`** (`Flock.SecurityProofs.RecursiveSound`'s proof at
`zk_sessions_recursive_read`). -/
theorem recursiveSound_read : RecursiveSoundRead := by
  intro n L Reg I Cm S Vs Reg' Reg₂ hS hA3 zs dg N k Rw M hk hRw hM ρ hρ hr t' ht' qF qS qW qR σ x wmsg Sk cmt msg hbr q
    hCR εin hin
  obtain ⟨hb, hcmt, hdec, hfull⟩ := hbr
  exact zk_sessions_recursive_read hS hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x wmsg hb (fun _ _ a b => a = b)
    I.accepts_of_rel_eq Sk cmt msg hcmt hdec hfull q hCR hin

end FlockVBridge.Family
