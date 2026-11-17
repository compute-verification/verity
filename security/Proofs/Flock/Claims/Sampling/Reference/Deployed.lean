module

public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.Reference.Complete

/-! Sampling completeness at the deployed record, from `Reference.complete_of_sessions` with the record's facts and the rest as hypotheses. -/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling Claims.Sampling.Reference

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

def Framed (H : Msg → Msg) : Prop :=
  ∀ x {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) u, u ∈ c.checked x r →
    ∃ (xS : Msg) (emb : Fin ((FlockDeploy.V true).k xS) → Fin (c.V.k x)), Function.Injective emb ∧
      (∀ i, emb i ∈ c.rounds x u) ∧
      ∀ {ΩF : Type} (PF : Prover ΩF) (ωF : ΩF),
        (∀ hist hS, c.step x hist = .session xS hS →
          ((P.next x hist ω).eval H).1 = ((PF.next xS hS ωF).eval H).1) →
        ∀ r' : Fin (c.V.k x) → FlockDeploy.Coin, (∀ t ∉ c.rounds x u, r' t = r t) →
          c.session x u (run H P c.V x ω r').seen =
            some (xS, (run H PF (FlockDeploy.V true) xS ωF fun i => r' (emb i)).seen)

def AcceptsFramed (H : Msg → Msg) : Prop :=
  ∀ x {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin),
    (∃ s Pt, c.parse x = some s ∧ c.partition s = some Pt ∧ InScope Pt s) →
    (∀ hist m, c.step x hist = .say m → ((P.next x hist ω).eval H).1 = m) →
    (∀ u ∈ c.checked x r, ∀ xS vS, c.session x u (run H P c.V x ω r).seen = some (xS, vS) → vS.out = .acc) →
      (run H P c.V x ω r).out = .acc

def SessionsScope (H : Msg → Msg) : Prop :=
  ∀ x s Pt, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) u xS vS, u ∈ c.checked x r →
      c.session x u (run H P c.V x ω r).seen = some (xS, vS) → FlockDeploy.scopeCheck true xS = true

def SessionWit (H : Msg → Msg) : Prop :=
  ∀ x s Pt w, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    faulty H (commit c) s Pt w = ∅ →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) u xS vS, u ∈ c.checked x r →
      c.session x u (run H P c.V x ω r).seen = some (xS, vS) →
      ∀ os : List (Opening Bool),
        (∀ o ∈ os, (commit c).Opens H s.roots o = true ∧ w.val H (commit c) s o.pos = some o.val) →
        (∀ g ∈ c.reads s xS, ∀ b, w.val H (commit c) s g = some b → ∃ o ∈ os, o.pos = g) →
          Claims.Flock.R (FlockDeploy.deployZk rel).parse xS (c.wit s xS os)

/-- One session that proves every sampled proof block has it at `sessions = 1`, and one session per unit at the proof blocks' count (`sessionSlots_units`). -/
def SessionSlots (H : Msg → Msg) : Prop :=
  ∀ x, (∃ s Pt, c.parse x = some s ∧ c.partition s = some Pt ∧ InScope Pt s) →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) u, u ∈ c.checked x r →
      ∃ p < c.sessions x, ∃ u', (c.checked x r)[p]? = some u' ∧
        c.session x u (run H P c.V x ω r).seen = c.session x u' (run H P c.V x ω r).seen

theorem δcZkT_nonneg (H : Msg → Msg) (x : Msg) : 0 ≤ FlockComplete.ZkTables.δc H x := by
  unfold FlockComplete.ZkTables.δc FlockComplete.εZk
  split_ifs <;> positivity

theorem honestSessions_deploy (H : Msg → Msg) (hframe : Framed c H) (hscope : SessionsScope c H)
    (hwit : SessionWit rel c H) : HonestSessions H (deploy rel c) := by
  rintro x w ⟨s, Pt, hs, hP, hin⟩ ⟨s', Pt', hs', hP', hf⟩ ω r u hu
  change c.parse x = some s at hs
  change c.parse x = some s' at hs'
  rw [hs] at hs'
  cases hs'
  change c.partition s = some Pt at hP
  change c.partition s = some Pt' at hP'
  rw [hP] at hP'
  cases hP'
  obtain ⟨xS, emb, hinj, hemb, hrun⟩ :=
    hframe x (Reference.refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w) ω r u hu
  have hr := hrun (FlockComplete.ZkTables.honestPR (c.wit s xS (found H (commit c) s.roots w (c.reads s xS)))) (ω xS)
    fun hist hS h => refP_session H FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w hs h ω
  have h0 := hr r fun _ _ => rfl
  exact ⟨xS, c.wit s xS (found H (commit c) s.roots w (c.reads s xS)), ω xS, emb, hinj, hemb,
    (FlockDeploy.scopeMatches_zk rel xS).2 (hscope x s Pt hs hP hin _ ω r u xS _ hu h0),
    hwit x s Pt w hs hP hin hf _ ω r u xS _ hu h0 _ (fun o ho => ⟨found_opens ho, found_val ho⟩)
      (fun g hg b hb => found_covers hg hb), hr⟩

theorem acceptsSessions_deploy (H : Msg → Msg) (haccept : AcceptsFramed c H) : AcceptsSessions H (deploy rel c) := by
  rintro x w hx - ω r hsess
  exact haccept x (Reference.refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w) ω r hx
    (fun hist m h => refP_say H FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w h ω) hsess

theorem sessionSlots_units (H : Msg → Msg) (hslots : DrawSlots (deploy rel c)) (hn : ∀ x, units c x ≤ c.sessions x) :
    SessionSlots c H := by
  intro x hx Ω P ω r u hu
  obtain ⟨i, hi, hget⟩ := List.getElem_of_mem hu
  have hle : (c.checked x r).length ≤ units c x := checkedLe_of_slots _ hslots x hx r
  exact ⟨i, (hi.trans_le hle).trans_le (hn x), u, List.getElem?_eq_some_iff.2 ⟨hi, hget⟩, rfl⟩

theorem sessionsAmong_deploy (H : Msg → Msg) (h : SessionSlots c H) : SessionsAmong H (deploy rel c) c.sessions := by
  rintro x w hx - ω r u hu
  exact h x hx (Reference.refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w) ω r u hu

set_option synthInstance.maxSize 1024 in
theorem complete_deploy (H : Msg → Msg) (hF : Claims.Flock.Complete H (FlockDeploy.deployZk rel))
    (hfix : SampleFixed (deploy rel c)) (hslots : DrawSlots (deploy rel c)) (hframe : Framed c H)
    (haccept : AcceptsFramed c H) (hscope : SessionsScope c H) (hwit : SessionWit rel c H)
    (hsess : SessionSlots c H) : Complete H (deploy rel c) :=
  complete_of_sessions H (deploy rel c) hF hfix (checkedLe_of_slots _ hslots)
    (honestSessions_deploy rel c H hframe hscope hwit) (acceptsSessions_deploy rel c H haccept) c.sessions
    (sessionsAmong_deploy rel c H hsess) (δcZkT_nonneg H) fun _ => le_of_eq rfl

end SamplingDeploy
