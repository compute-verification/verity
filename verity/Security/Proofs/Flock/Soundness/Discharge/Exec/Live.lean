import Proofs.Flock.Soundness.Refine.LiveCompiled
import Proofs.Flock.Soundness.Discharge.Exec.HeadlineL

/-!
# R11 for one-table sessions: the live audit simulated in the salted audit

`ExecRefines` (`Event.lean`) is R11 at one strategy pair. This file proves it, with `bad = ⊥`, for the live audit whose
session is one table's live session (`liveTable`, the challenger with retained round bytes, `Refine/Live.lean`) written
out as a session directory by `wr`, against the audit with the salted session `batchedSessionL` on one-table plans. What
it takes is `Decodes` (`Refine/LiveSim.lean`) at each draw and registration: on every live outcome flock-verify
accepts, each rep's game runs on the tape decoded from its stream and its proof file, and the salted table's verdict on
those runs holds.

* `sessionBL_one`: the salted session on one table is `modelTable`'s shape, with the reps of `tableAfterL` and the
  output `[tableOutL …]`.
* `live_le_sessionBL`: against any live prover of `liveTable`, the live verdict is no likelier than the salted
  session's acceptance against the simulated prover (`live_le` at `sessionBL_one`).
* `simAudit`, `execRefines_one`: the audit strategy built from a live one (the registration it sends, and at each draw
  the simulated session strategy), and R11 at it, `ExecRefines … (LiveAccepts I) ⊥ (audit … (batchedSessionL …)) …`.
* `Prog.flock_headline_live`: `Prog.flock_headline_execL` with `hRef` discharged by `execRefines_one`. For one-table
  plans and the live session `liveTable` written by `wr`, flock-verify accepts with at least `K` wrong units of the
  simulated prover's committed values with probability at most `miss(K) + ksAvgStrict + linkBoundE`, given `Decodes`.

What R11 still needs past this: `Decodes` itself, which is deterministic (R11b, the framing: the round bytes the coin
server retains decode to the messages the proof files replay, and `wr` is the server's record); sessions of more than
one table (the circuit prover's default is one, `--session-tables 1`); and that the challenger's session is `liveTable`,
whose coins are uniform and drawn after each round's bytes (A6).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Refine Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## Strategies through relabellings and casts -/

section Plumbing

variable {α β : Type}

theorem prob_strategy_map (p : β → Prop) (f : α → β) :
    ∀ (g : Game α) (s : Strategy g), prob p (g.map f) (Strategy.map f g s) = prob (fun a => p (f a)) g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_strategy_map p f (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      show avg (fun c => prob p ((k c).map f) (Strategy.map f (k c) (s c))) = avg fun c => prob _ (k c) (s c)
      exact congrArg avg (funext fun c => prob_strategy_map p f (k c) (s c))

theorem prob_cast {g g' : Game α} (h : g = g') (p : α → Prop) (s : Strategy g) : prob p g' (h ▸ s) = prob p g s := by
  subst h; rfl

theorem map_map' (f : α → β) {γ : Type} (f' : β → γ) (g : Game α) : (g.map f).map f' = g.map fun a => f' (f a) := by
  unfold Game.map
  rw [Game.bind_assoc]
  rfl

theorem bind_map {γ : Type} (g : Game α) (K : α → Game β) (f : β → γ) :
    (g.bind K).map f = g.bind fun a => (K a).map f := by
  unfold Game.map
  rw [Game.bind_assoc]

end Plumbing

/-! ## One table's salted session as `modelTable` -/

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

open Classical in
/-- The salted table's output, `tableAfterL`'s, as `modelTable`'s. -/
noncomputable def tableOutL (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (cap : Cap D) (pend : PendL 𝔽 D × PendL 𝔽 D) (os : Opens 𝔽 D × Salts Sl) : OutCL 𝔽 D Sl :=
  ⟨pend.1.verdict os.1 && pend.2.verdict os.1 &&
    decide (OpensOKL H E Lf cap 0 pend.1 os.1 os.2 ∧ OpensOKL H E Lf cap 1 pend.2 os.1 os.2), cap, pend, os.1, os.2⟩

/-- The reps of table `T`, at link points `pts`. -/
noncomputable abbrev repsOf (A : Arith 𝔽 K) (T : TabSpec mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) :
    Game (PendL 𝔽 D) :=
  repC A r.val T.S T.sch T.hm (extraClaims T.S T.ptLocal mPts pts)

omit [Fintype K] in
/-- **The salted session on one table is `modelTable`'s shape**: `Commit`'s caps, the link points, the reps
concurrently, the salted openings, and the one table's output. -/
theorem sessionBL_one (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : TabSpec mPts) :
    modelTable (repsOf (D := D) A T) (fun (caps : Fin 1 → Cap D) pend os => [tableOutL H E Lf (caps 0) pend os]) =
      sessionBL A H E Lf [T] := by
  show Game.send (Fin 1 → Cap D) (fun caps => Game.coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
      (interleave (repsOf (D := D) A T pts 0) (repsOf (D := D) A T pts 1)).bind fun pend =>
        Game.send (Opens 𝔽 D × Salts Sl) fun os => Game.ret [tableOutL H E Lf (caps 0) pend os]) =
    Game.send (Fin 1 → Cap D) fun caps => Game.coin (Fin 2 → Fin mPts → 𝔽) fun pts =>
      batch ((gamesBL A H E Lf [T] caps).map (· pts))
  congr 1
  funext caps
  congr 1
  funext pts
  simp only [gamesBL, List.ofFn_succ, List.ofFn_zero, List.map_cons, List.map_nil, batch, Plays.interleave_ret,
    map_map']
  rw [show (fun a : OutCL 𝔽 D Sl => (a, ([] : List (OutCL 𝔽 D Sl))).1 :: (a, ([] : List (OutCL 𝔽 D Sl))).2) =
    fun a => [a] from rfl]
  unfold tabAfterL tableAfterL
  rw [bind_map]
  rfl

/-! ## The transfer at one table -/

variable [Nonempty D] [Inhabited D] [Inhabited Sl]

omit [Fintype K] [Nonempty D] in
/-- `Decodes` for the one table's output, as `Decodes` for the session's list of one output. -/
theorem decodes_one {H : List UInt8 → D} {E : Merkle.Enc (List 𝔽) D} {Lf : Merkle.Leaf (List 𝔽) Sl D H}
    {T : TabSpec mPts} {A : Arith 𝔽 K} {pL : LiveOut mPts → Prop} {opn : Array ByteArray → Opens 𝔽 D × Salts Sl}
    {dec : ℕ → ByteArray → List Msg} {one : ℕ → Bool} {fin : Fin 2 → Array ByteArray → List Msg}
    {capOf : ByteArray → (Fin 1 → Cap D)}
    (hdec : Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os) pL (·.accepted = true)
      opn dec one fin capOf (repsOf (D := D) A T)) :
    Decodes (fun (caps : Fin 1 → Cap D) pend os => [tableOutL H E Lf (caps 0) pend os]) pL (acceptedBL [T]) opn dec
      one fin capOf (repsOf (D := D) A T) := fun o ho => by
  obtain ⟨pend, h0, h1, hacc⟩ := hdec o ho
  refine ⟨pend, h0, h1, fun i => ?_⟩
  obtain ⟨i, hi⟩ := i
  obtain rfl : i = 0 := by simp at hi; omega
  exact hacc

/-- The simulated session strategy against a live prover of `liveTable`. -/
noncomputable def simSessL (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : TabSpec mPts) (pL : LiveOut mPts → Prop)
    (opn : Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray) (dec : ℕ → ByteArray → List Msg)
    (one : ℕ → Bool) (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → (Fin 1 → Cap D)) (N : ℕ)
    (hdec : Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os) pL (·.accepted = true)
      opn dec one fin capOf (repsOf (D := D) A T))
    (Pr : Strategy (liveTable dg mPts N)) : Strategy (sessionBL A H E Lf [T]) :=
  sessionBL_one A H E Lf T ▸
    (simTable (fun (caps : Fin 1 → Cap D) pend os => [tableOutL H E Lf (caps 0) pend os]) pL (acceptedBL [T]) opn dg
      dec one fin capOf N (repsOf (D := D) A T) (fun _ _ => Model.allInh_repC A _ T.S T.sch T.hm _)
      (decodes_one hdec)).strategy Pr

/-- **The transfer at one table, for the salted session.** Against any live prover of `liveTable`, the live verdict
`pL` holds with probability at most the salted session's acceptance against the simulated prover, whenever the decoder
is right on accepted sessions (`Decodes`). -/
theorem live_le_sessionBL (A : Arith 𝔽 K) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : TabSpec mPts) (pL : LiveOut mPts → Prop)
    (opn : Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray) (dec : ℕ → ByteArray → List Msg)
    (one : ℕ → Bool) (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → (Fin 1 → Cap D)) (N : ℕ)
    (hdec : Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os) pL (·.accepted = true)
      opn dec one fin capOf (repsOf (D := D) A T))
    (Pr : Strategy (liveTable dg mPts N)) :
    prob pL (liveTable dg mPts N) Pr ≤
      prob (acceptedBL [T]) (sessionBL A H E Lf [T]) (simSessL A H E Lf T pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessL
  rw [prob_cast]
  exact live_le _ pL _ opn dg dec one fin capOf N _ _ (decodes_one hdec) Pr

/-! ## The audit -/

variable {n : ℕ} {Reg : Type}

/-- **The live audit's session at one table**: `liveTable`, its outcome written out as the session directory by `wr`
(the record the challenger writes, and the proof files). -/
def liveOf (dg : ByteArray → ByteArray) (mPts N : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (S : Finset (Fin n)) (R : Reg) : Game Transcript :=
  (liveTable dg mPts N).map (wr S R)

variable (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
  (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (N : ℕ)

/-- **The decoder is right on the sessions flock-verify accepts** (R11b, named): at every draw and registration, on
every live outcome whose session directory flock-verify accepts, each rep's game runs on the tape decoded from its
stream and its proof file, and the salted table's verdict on those runs, with the cap `root_B` decodes to and the
openings and salts the proof files decode to, holds. -/
def DecodesAll : Prop :=
  ∀ S R, Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os)
    (fun o => Accepts I (wr S R o)) (·.accepted = true) (opn S R) (dec S R) (one S R) (fin S R) (capOf S R)
    (repsOf (D := D) execArith (T S R))

variable {L : Law n}

/-- **The simulated audit strategy**: the live prover's registration, and at each draw the simulated session
strategy against its live session. -/
noncomputable def simAudit (hdec : DecodesAll H E Lf T I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessL execArith H E Lf (T (L.draw ω) τ.1) (fun o => Accepts I (wr (L.draw ω) τ.1 o)) (opn (L.draw ω) τ.1) dg
      (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec (L.draw ω) τ.1)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **R11 for one-table sessions.** Against any live prover `τ` of the live audit whose session is `liveTable` written
out by `wr`, flock-verify's acceptance is no likelier than the salted audit's acceptance against the simulated prover
`simAudit … τ`, with no live event added (`bad = ⊥`), whenever the decoder is right on accepted sessions
(`DecodesAll`). -/
theorem execRefines_one (hdec : DecodesAll H E Lf T I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAccepts I) (fun _ => False)
      (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) (simAudit H E Lf T I wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOf dg mPts N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAccepts I) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionL H E Lf (fun S R => [T S R]) (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simAudit H E Lf T I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAudit batchedSessionL
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  refine (live_le_sessionBL execArith H E Lf (T (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
    (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec (L.draw ω) τ.1) _).trans_eq ?_
  congr 1
  funext outs
  classical
  exact propext decide_eq_true_iff.symm

/-! ## The headline at the executable, for one-table sessions -/

variable {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))

/-- **The headline at the executable, for one-table sessions of the challenger.** Let the verifier draw the units
as in `Prog.flock_headline_L` (the live stratified draw under A3), let each draw's session be one table `T S R`, and let
the live session be `liveTable` (the challenger, its coins uniform per round and drawn after the round's retained
bytes), written out as a session directory by `wr`. Against any live prover `τ`, flock-verify accepts with at least `K`
wrong units of the committed values of the simulated prover (`simAudit … τ`) with probability at most
`miss(K) + ε_ks + δ_link`: `Prog.flock_headline_execL` with `hRef` discharged by `execRefines_one`, so `Pr[bad] = 0`.
Beyond `Prog.flock_headline_L`'s (`pp`, `hConst`, `hZero`, `lay`, `rs`, `hHm`, and the assumptions through `hA3`, `hKS`,
`hCR`), it takes `hdec`: the decoder is right on the sessions flock-verify accepts (R11b). -/
theorem _root_.FlockSoundness.Prog.flock_headline_live
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (T : Finset (Fin n) → Reg → TabSpec mPts)
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin [T S R].length)
    (pp : ProgPlaces p outs (fun S R => [T S R]) tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols (fun S R => [T S R]) tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols (fun S R => [T S R]) tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) (fun S R => [T S R]) tab
      ((p.partition outs).placeDecoder (fun S R => [T S R]) tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB [T S R] j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesAll H512 enc512 leaf512 T I wr opn dec one fin capOf)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
      (reg (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      [T ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))] j
      (batchStratL H512 enc512 leaf512 (fun S R => [T S R])
        (cont (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
              (reg (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (fun S R => [T S R]) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          (fun S R => [T S R]) tab (rs.binding hHm) k Rw M ρ t' := by
  have h := p.flock_headline_execL outs (fun S R => [T S R]) tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst
    zeros hZero lay rs hHm hk hRw hM hρ hr ht K I (liveOf dg mPts N wr) τ
    (simAudit H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_one H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ) hCR hKS
  rwa [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h

end FlockSoundness.Discharge.Exec
