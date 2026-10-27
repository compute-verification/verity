import Proofs.Flock.Claims.Sampling.Deployed
import Proofs.Flock.Claims.Sampling.Reference.Complete

/-!
# Sampling completeness at the deployed record

`complete_deploy`: **`Claims.Sampling.Complete H (deploy rel c)`**, of the deployed record's reference prover
(`Reference.refP` at the code's framing) at its error: `c.sessions x` times the largest `ZkTables.δc H`, capped at `1`,
over the statements of the reference prover's sessions at `x` (`Reference.sessErrAt`), from
`Reference.complete_of_sessions` with the record's facts. What remains, each a hypothesis:

* `hF : Claims.Flock.Complete H (deployZk rel)`: Flock's masked completeness, of `ZkTables.honestPR` at `ZkTables.δc H`.
* `DrawFixed` (`Sampling.Session`) and `DrawSlots` (`Sampling.Select.DrawLaw`), the sampled verifier's, as for KS: the
  checked units' sessions are in the second stage and the draw doesn't read them; the draw is `Slots.select`'s, so it
  checks at most the statement's proof units (`checkedLe_of_slots`).
* `Framed`: the reference prover's router (`step`) reads the sampled verifier's framing: at coins that check `u`, a
  prover whose messages, at the histories where the router puts the run in the session under `u`'s session's
  statement, are a Flock prover's at the session's history has `u`'s session be that Flock prover's run, on `u`'s
  rounds; the sampled verifier's and the code's.
* `AcceptsFramed`: on a statement in scope, a prover that says what the framing fixes outside the sessions is accepted
  when every checked unit's session accepts; the sampled verifier's.
* `SessionsScope`: on a statement in scope, at a run whose coins check `u`, `u`'s session's statement passes the
  deployed `--zk` scope check, so it is in Flock's masked scope (`FlockDeploy.scopeMatches_zk`); the code's.
* `SessionWit`: with no faulty unit in the trace, at a run whose coins check `u`, openings that pass the commitment's
  check, are the committed values and cover what `u`'s session reads under its statement make a Flock witness (`wit`)
  in that statement's relation; the code's. The session's statement may be several drawn units', as in the one-stage
  audit, whose one session proves every drawn unit.
* `SessionSlots`: on a statement in scope, every checked unit's session is the session of the unit checked at a slot
  below `c.sessions x`; the code's. One session that proves every drawn unit has it at `sessions = 1`
  (`Reference.DeployedCode`'s `sessionSlots_of_one`); one session per unit at the proof units' count, from `DrawSlots`
  (`sessionSlots_units`).

What is proved here: the reference prover's session messages are `ZkTables.honestPR`'s on what it reveals
(`refP_session`), it says the framing outside them (`refP_say`), what it reveals opens under the roots and is the
committed value (`found_opens`, `found_val`, `found_covers`), and Flock's masked error is nonnegative
(`δcZkT_nonneg`).
-/

namespace SamplingDeploy

open Claims Claims.Sampling Claims.Sampling.Reference

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

/-- **The reference prover's router reads the sampled verifier's framing**: for every prover `P` and coins `r` that
check `u`, a statement `xS` and an injective map `emb` of Flock's rounds into `u`'s such that, for a Flock prover `PF`
whose messages are `P`'s wherever the router puts the run in the session under `xS`, at coins agreeing with `r` off
`u`'s rounds, `u`'s session is `PF`'s run on `xS` at the coins `emb` reads. -/
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

/-- **The run accepts when its sessions do**: on a statement in scope, a prover that says what the framing fixes outside
the sessions is accepted when every checked unit's session that shows accepts. -/
def AcceptsFramed (H : Msg → Msg) : Prop :=
  ∀ x {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin),
    (∃ s Pt, c.parse x = some s ∧ c.partition s = some Pt ∧ InScope Pt s) →
    (∀ hist m, c.step x hist = .say m → ((P.next x hist ω).eval H).1 = m) →
    (∀ u ∈ c.checked x r, ∀ xS vS, c.session x u (run H P c.V x ω r).seen = some (xS, vS) → vS.out = .acc) →
      (run H P c.V x ω r).out = .acc

/-- **The sessions pass Flock's masked scope check**: on a statement in scope, at any run whose coins check `u`, `u`'s
session's statement passes the deployed `--zk` scope check (`FlockDeploy.scopeCheck true`), so it is in Flock's masked
scope (`FlockDeploy.scopeMatches_zk`). -/
def SessionsScope (H : Msg → Msg) : Prop :=
  ∀ x s Pt, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) u xS vS, u ∈ c.checked x r →
      c.session x u (run H P c.V x ω r).seen = some (xS, vS) → FlockDeploy.scopeCheck true xS = true

/-- **An honest unit's openings make a Flock witness in its session's relation**: on a statement in scope with no
faulty unit, at any run whose coins check `u` and whose `u` session has statement `xS`, openings that pass the
commitment's check under the roots, are the committed values `Wit.val` reads, and cover every position the session
reads under `xS` that has a committed value, encode (`wit`) a witness in `xS`'s relation. -/
def SessionWit (H : Msg → Msg) : Prop :=
  ∀ x s Pt w, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    faulty H (commit c) s Pt w = ∅ →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) u xS vS, u ∈ c.checked x r →
      c.session x u (run H P c.V x ω r).seen = some (xS, vS) →
      ∀ os : List (Opening Bool),
        (∀ o ∈ os, (commit c).Opens H s.roots o = true ∧ w.val H (commit c) s o.pos = some o.val) →
        (∀ g ∈ c.reads s xS, ∀ b, w.val H (commit c) s g = some b → ∃ o ∈ os, o.pos = g) →
          Claims.Flock.R (FlockDeploy.deployZk rel).parse xS (c.wit s xS os)

/-- **The run's sessions are its first `sessions x` slots'**: on a statement in scope, at any run, every checked unit's
session is the session of the unit checked at a slot below `c.sessions x`. One session that proves every drawn unit has
it at `sessions = 1`, and one session per unit at the proof units' count (`sessionSlots_units`). -/
def SessionSlots (H : Msg → Msg) : Prop :=
  ∀ x, (∃ s Pt, c.parse x = some s ∧ c.partition s = some Pt ∧ InScope Pt s) →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) u, u ∈ c.checked x r →
      ∃ p < c.sessions x, ∃ u', (c.checked x r)[p]? = some u' ∧
        c.session x u (run H P c.V x ω r).seen = c.session x u' (run H P c.V x ω r).seen

/-- **Flock's masked error is nonnegative**: `εZk`, a count over `2^128`, or `1`. -/
theorem δcZkT_nonneg (H : Msg → Msg) (x : Msg) : 0 ≤ FlockComplete.ZkTables.δc H x := by
  unfold FlockComplete.ZkTables.δc FlockComplete.εZk
  split_ifs <;> positivity

/-- **At the deployed record, a checked unit's session is `ZkTables.honestPR`'s run** on what the reference prover
reveals. -/
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

/-- **At the deployed record, the run accepts when its sessions do.** -/
theorem acceptsSessions_deploy (H : Msg → Msg) (haccept : AcceptsFramed c H) : AcceptsSessions H (deploy rel c) := by
  rintro x w hx - ω r hsess
  exact haccept x (Reference.refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w) ω r hx
    (fun hist m h => refP_say H FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w h ω) hsess

theorem nPU_deploy (x : Msg) : nPU (deploy rel c) x = units c x := rfl

/-- **At one session per unit, the proof units' count bounds the slots**: the draw checks at most them (`DrawSlots`). -/
theorem sessionSlots_units (H : Msg → Msg) (hslots : DrawSlots (deploy rel c)) (hn : ∀ x, units c x ≤ c.sessions x) :
    SessionSlots c H := by
  intro x hx Ω P ω r u hu
  obtain ⟨i, hi, hget⟩ := List.getElem_of_mem hu
  have hle : (c.checked x r).length ≤ units c x := checkedLe_of_slots _ hslots x hx r
  exact ⟨i, (hi.trans_le hle).trans_le (hn x), u, List.getElem?_eq_some_iff.2 ⟨hi, hget⟩, rfl⟩

/-- **At the deployed record, the checked units' sessions are the first `sessions x` slots'.** -/
theorem sessionsAmong_deploy (H : Msg → Msg) (h : SessionSlots c H) : SessionsAmong H (deploy rel c) c.sessions := by
  rintro x w hx - ω r u hu
  exact h x hx (Reference.refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit w) ω r u hu

set_option synthInstance.maxSize 1024 in
/-- **Sampling completeness at the deployed record**: the reference prover is accepted with probability at least
`1 − δc H x`, `c.sessions x` times the largest `ZkTables.δc H` over its sessions' statements, from Flock's masked
completeness and the facts the module docstring lists. -/
theorem complete_deploy (H : Msg → Msg) (hF : Claims.Flock.Complete H (FlockDeploy.deployZk rel))
    (hfix : DrawFixed (deploy rel c)) (hslots : DrawSlots (deploy rel c)) (hframe : Framed c H)
    (haccept : AcceptsFramed c H) (hscope : SessionsScope c H) (hwit : SessionWit rel c H)
    (hsess : SessionSlots c H) : Complete H (deploy rel c) :=
  complete_of_sessions H (deploy rel c) hF hfix (checkedLe_of_slots _ hslots)
    (honestSessions_deploy rel c H hframe hscope hwit) (acceptsSessions_deploy rel c H haccept) c.sessions
    (sessionsAmong_deploy rel c H hsess) (δcZkT_nonneg H) fun _ => le_of_eq rfl

end SamplingDeploy
