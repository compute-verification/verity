import Proofs.Flock.Soundness.Registered.ReadsX
import Proofs.Flock.Soundness.Discharge.Exec.Live

/-!
# Registered reads off the registrant's leaf, at the outcome's draw

`ReadsX.prob_off_le` bounds the probability that a registered read's session leaf is not the registrant's at one
registration and one set of reads. In a live session both are the outcome's: the registration is the one the prover sent,
and the verifier checks only the reads of the units it draws. So:

* `rowIn S`: a commit string that is a row of a unit of `S`; `ReadsX.restrict`: the reads at the positions `A` keeps;
* `ReadsX.prob_offOn_le`: `prob_off_le` with the reads, the registration and the paths read off the outcome;
* `prob_auditLive_draw`: an event of the live audit's draw alone has the law's probability (implications at every
  outcome of the session bound probabilities: `prob_auditLive_liveOfZJ_mono`, `Proofs.Flock.Recursive.Flock`);
* `reg_cast`: a strategy's registration survives a cast along an equation of sessions.
-/

namespace FlockSoundness.Discharge.Recursion

/-- **A commit string that is a row of a unit of `S`.** -/
def rowIn {n : ℕ} {Q G : Type} (S : Finset (Fin n)) : (Fin n × Q) ⊕ G → Prop
  | .inl (u, _) => u ∈ S
  | .inr _ => False

theorem rowIn_mono {n : ℕ} {Q G : Type} {S S' : Finset (Fin n)} (h : S ⊆ S') :
    ∀ p : (Fin n × Q) ⊕ G, rowIn S p → rowIn S' p
  | .inl (_, _), hp => h hp
  | .inr _, hp => hp.elim

end FlockSoundness.Discharge.Recursion

namespace FlockSoundness.Registered

open Game Refine Binding Audit
open scoped ENNReal

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace ReadsX

variable {lay : Layout P plan tab decode} {rs : HmRowsX lay}

/-- **The registered reads at the positions `A` keeps.** -/
def restrict (rd : ReadsX lay rs) (A : lay.Pos → Prop) : ReadsX lay rs where
  isRead p := rd.isRead p ∧ A p
  read_row p h := rd.read_row p h.1
  port := rd.port
  readAt := rd.readAt
  wv := rd.wv
  ws := rd.ws
  wpath := rd.wpath
  committed p h := rd.committed p h.1

/-- **`prob_off_le` at the outcome's reads and registration.** In a game `G` played by `σ` whose accepted outcomes `o`
(`accept`) the verifier's registered check holds on, for the reads `rdo o` at the registration `Rof o` through
`path o`: the probability of accepting while one of `rdo o`'s reads has a leaf under `Rof o` that is not the
registrant's is at most `qR²/2^513`, under `SHA512CRStrict` (`cr/sha-512`) for the finder that plays `G` once and
outputs the extractor's pair. -/
theorem prob_offOn_le {β : Type} (G : Game β) (σ : Strategy G) (rdo : β → ReadsX lay rs) (Rof : β → Reg)
    (accept : β → Prop) (path : β → lay.Pos → ByteArray) (hchk : ∀ o, accept o → (rdo o).Checked (Rof o) (path o))
    {qR : ℝ} (hR : (StrictCR.oneRun G σ fun o => (rdo o).pick (Rof o) (path o)).CR H512 qR) :
    prob (fun o => accept o ∧ (rdo o).Off (Rof o)) G σ ≤ ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  have hle : prob (fun o => accept o ∧ (rdo o).Off (Rof o)) G σ ≤
      prob (fun o => StrictCR.IsColl H512 ((rdo o).pick (Rof o) (path o))) G σ :=
    prob_mono (fun o ⟨ha, hoff⟩ => ((rdo o).openings (Rof o) (path o) ((rdo o).offAt (Rof o))).isColl
      ((rdo o).offAt_conflict (hchk o ha) hoff)) G σ
  have hev : (prob (fun o => StrictCR.IsColl H512 ((rdo o).pick (Rof o) (path o))) G σ).toReal ≤
      qR ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    have := StrictCR.Finder.CR.le hR
    rwa [StrictCR.oneRun_ev] at this
  have hfin : prob (fun o => StrictCR.IsColl H512 ((rdo o).pick (Rof o) (path o))) G σ ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  exact hle.trans ((ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 hev)

end ReadsX

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine
open scoped ENNReal

/-- **An event of the live audit's draw alone has the law's probability.** -/
theorem prob_auditLive_draw {n : ℕ} {L : Law n} {Reg : Type} (live : Finset (Fin n) → Reg → Game Exec.Transcript)
    (τ : Strategy (auditLive L Reg live)) (Q : Finset (Fin n) → Prop) :
    prob (fun o => Q o.2.1) (auditLive L Reg live) τ = prCoin fun ω => Q (L.draw ω) := by
  show avg (fun ω => prob (fun o => Q o.2.1) ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)) = _
  unfold prCoin
  refine congrArg avg (funext fun ω => ?_)
  rw [prob_map]
  by_cases h : Q (L.draw ω)
  · rw [if_pos h]
    exact le_antisymm (prob_le_one _ _ _) ((prob_true _ _).symm.le.trans (prob_mono (fun _ _ => h) _ _))
  · rw [if_neg h]
    exact le_antisymm ((prob_mono (fun _ hq => h hq) _ _).trans (prob_false _ _).le) zero_le

/-- **A strategy's registration survives a cast along an equation of sessions.** -/
theorem reg_cast {n : ℕ} {L : Law n} {Reg : Type} {g g' : Finset (Fin n) → Reg → Game Bool} (h : g = g')
    (x : Strategy (audit L Reg g)) : reg (cast (congrArg (fun g => Strategy (audit L Reg g)) h) x) = reg x := by
  subst h
  rfl

end FlockSoundness.Discharge.Recursion
