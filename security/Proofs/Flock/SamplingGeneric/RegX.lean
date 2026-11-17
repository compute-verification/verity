module

public import Definitions.Claims.SamplingGenericX
public import Proofs.Flock.SamplingGeneric.Reg
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.SamplingGeneric.Assemble
import Proofs.Flock.SamplingGeneric.RegTrace

/-! The sampled compiler's committed form at the trace read at the statement, which no tape moves. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement InScope ownLog)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

def OffRegX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : Prop :=
  ∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ L.regValX H P x g ≠ some b

def ConflictX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))) : Prop :=
  ∃ g : Fin (L.nG x), (L.opensAt H P x qq.1.1 qq.1.2.1 qq.1.2.2 g).isSome ∧
    (L.opensAt H P x qq.2.1 qq.2.2.1 qq.2.2.2 g).isSome ∧
    L.opensAt H P x qq.1.1 qq.1.2.1 qq.1.2.2 g ≠ L.opensAt H P x qq.2.1 qq.2.2.1 qq.2.2.2 g

noncomputable def plurX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (g : ℕ) : Bool :=
  plurAt (fun (q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x q.1 q.2.1 q.2.2 g) g

theorem regValX_eq_plurX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    {ω : Ω} {r : Fin (L.V.k x) → Coin} {c : L.SlotEC x} {g : ℕ} {b : Bool}
    (h : L.opensAt H P x ω r c g = some b) : L.regValX H P x g = some (L.plurX H P x g) := by
  have hpos := Pr_pos_of (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
    L.opensAt H P x q.1 q.2.1 q.2.2 g = some b) (ω, (r, c)) h
  simp only [regValX, plurX, plurAt]
  split_ifs with h0
  · cases b
    · exact absurd h0.2 hpos.ne'
    · exact absurd h0.1 hpos.ne'
  · exact congrArg some (decide_eq_decide.2 Iff.rfl)

theorem pr_offRegX_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      L.nG x * Real.sqrt (Pr fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
        (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.ConflictX H P x qq) :=
  pr_off_le (L.nG x)
    (fun (q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x q.1 q.2.1 q.2.2 g)
    (L.regValX H P x) fun _ _ _ h => L.regValX_eq_plurX H P x h

theorem le_bconfX (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {Q : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) → Prop}
    (hQ : ∀ q, Q q → ∃ u v, findColl (ownLog H (run H P L.V x q.1.1)
        (L.runExt x s q.1.2.1 (run H P L.V x q.1.1 q.1.2.1).seen q.1.2.2) ++
      ownLog H (run H P L.V x q.2.1) (L.runExt x s q.2.2.1 (run H P L.V x q.2.1 q.2.2.1).seen q.2.2.2)) =
        some (u, v) ∧ Collision H u v) :
    Pr Q ≤ (BconfX H L Ω P x).Pr H := by
  simp only [BconfX, hs]
  exact Pr_mono hQ

theorem pr_inConfX_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) ≤
      (BconfX H L Ω P x).Pr H := by
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      = Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.InConf H P x s q.1 q.2.1 q.2.2) :=
        (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          L.InConf H P x s q.1 q.2.1 q.2.2)).symm
    _ = Pr (fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
          L.InConf H P x s qq.1.1 qq.1.2.1 qq.1.2.2) :=
        (Pr_fst (β := Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))
          (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.InConf H P x s q.1 q.2.1 q.2.2)).symm
    _ ≤ (BconfX H L Ω P x).Pr H :=
        le_bconfX H L hs P fun q hq => by
          obtain ⟨a, ha, b, hb, h1, h2, h3⟩ := hq
          exact bconfX_of_disagree hbind s P q.1.1 q.2.1 q.1.2 q.2.2 (Or.inl ha) (Or.inl hb) h1 h2 h3

theorem pr_conflictX_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
      L.ConflictX H P x qq) ≤ (BconfX H L Ω P x).Pr H := by
  refine le_bconfX H L hs P fun q hq => ?_
  obtain ⟨g, h1, h2, hne⟩ := hq
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  rw [L.opensAt_eq H P x q.1.1 hs] at hb1
  rw [L.opensAt_eq H P x q.2.1 hs] at hb2
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := val_some_mem hb1
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := val_some_mem hb2
  exact bconfX_of_disagree hbind s P q.1.1 q.2.1 q.1.2 q.2.2 (Or.inl ha) (Or.inr ha') (ha1.trans ha1'.symm)
    (ha2.trans ha2'.symm) (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))

theorem ex_offRegX_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2) ≤
      L.nG x * Real.sqrt ((BconfX H L Ω P x).Pr H) :=
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2)
      = Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) :=
        (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          L.OffRegX H P x q.1 q.2.1 q.2.2)).symm
    _ ≤ L.nG x * Real.sqrt (Pr fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
          (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.ConflictX H P x qq) := L.pr_offRegX_le H P x
    _ ≤ L.nG x * Real.sqrt ((BconfX H L Ω P x).Pr H) :=
        mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (pr_conflictX_le hbind hs P)) (Nat.cast_nonneg _)

set_option synthInstance.maxSize 1024 in
theorem ksRegX_of_sessionKS {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) : L.KSRegX H := by
  intro x hsc Ω _ _ P
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  have hFlt : ∀ v ∈ L.FregX H P x, v < L.nPU x := by
    intro v hv
    simp only [FregX, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1
  have hO : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffRegX H P x q.1 q.2.1 q.2.2) ≤ L.nG x * Real.sqrt ((BconfX H L Ω P x).Pr H) :=
    calc Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
          L.OffRegX H P x q.1 q.2.1 q.2.2)
        ≤ Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) :=
          Pr_mono fun _ h => h.2
      _ = Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2) :=
          Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            L.OffRegX H P x q.1 q.2.1 q.2.2)
      _ ≤ L.nG x * Real.sqrt ((BconfX H L Ω P x).Pr H) := ex_offRegX_le hbind hs P
  have h1 : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.FregX H P x) ≤ L.εregX H x P := by
    simp only [FregX_eq_FregAt]
    refine (event_le_of_off hS hSK hs hPt hin P (fun _ _ => L.regValX H P x) (pr_inConfX_le hbind hs P) hO).trans ?_
    unfold Compiled.εregX
    linarith
  refine ⟨h1, fun s' hs' δ hδ => ?_⟩
  rw [show s' = s from Option.some.inj (hs'.symm.trans hs)]
  exact miss_of_event hS hs hPt hin P (fun _ => L.FregX H P x) (fun _ => hFlt) h1 hδ

end Compiled

theorem compilerKSRegX_of (hsess : SessionKSOf) : CompilerKSRegX := by
  intro Coin _ _ _ A L H hA hB hS
  exact Compiled.ksRegX_of_sessionKS hB hS (hsess Coin A L H hA hS)

end Claims.SamplingGeneric
