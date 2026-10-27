import Proofs.Accounting.JointBridge.Guarantees.JointHorizonBlocksDeployed
import Proofs.Accounting.JointBridge.IntegrityOfIntegrityBlock
import Proofs.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Proofs.Accounting.JointBridge.SpaceOfPeriodHeld
import Proofs.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod
import Proofs.Accounting.JointBridge.CommOfPeriodLeakAccepted
import Proofs.Accounting.JointBlocks

/-!
# Proof of the joint guarantee over a horizon of server-time blocks, every leg from its protocol's own guarantee

Each block's five per-period guarantees come from the five bridges at that block's realizations and premises:
`Bridge.IntegrityOfIntegrityBlock`, `Bridge.WorkOfH1CheckedCreditPeriod`, `Bridge.SpaceOfPeriodHeld`,
`Bridge.BandwidthOfAuditPerRunPeriod` and `Bridge.CommOfPeriodLeakAccepted`, each at its source's error. Then
`JointHorizonBlocks` (Demo A: `JointHorizonBlocksA`, the instance of `JointHorizonBlocksLegs`) bounds the joint event by
those errors summed over the blocks and the audits.
-/

namespace Accounting.SecurityProofs

/-- A sum over the five audits, written out. -/
theorem sum_univ_audit (g : Audit → ℝ) : ∑ a, g a = g .sampled + g .pouw + g .pous + g .poub + g .network := by
  rw [show (Finset.univ : Finset Audit) = {.sampled, .pouw, .pous, .poub, .network} from rfl,
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_singleton]
  ring

/-- **The joint guarantee over a horizon of server-time blocks, every leg from its protocol's own guarantee**
(`Accounting.Guarantees.JointHorizonBlocksDeployed`). -/
theorem JointHorizonBlocksDeployed : Accounting.Guarantees.JointHorizonBlocksDeployed := by
  intro ι α I _ _ _ _ Ω _ _ d C B s c p f εWipe δPous
    S q δI hq hδI hqt hA3 RI hT hL hcert hfin
    ι' K T L U Cn Ξ V _ _ _ _ _ _ _ Sc Cd n hn hN hr hM hK prm enc staged flat δs δbind δks ws opened salt Wp Pl folded
    macs u bUnit help RW hcost hhelp hScope hρ hw hluck hgame hdead hpre hbind hreg hdraw hsvc hsalt hpins hmu hrb hguess hfix
    hd hp Dr Qr Lr hP ΩW ΩQ run P round RS hdev hS
    N bits h sB G k lam hN0 RB hQ
    Coin Col Tn _ _ _ _ _ H D Ln Kₒ Sn lay kh δc decode regN Dj nc RN hks hbc hδc hdec εks εbind εtap herr hbindN htap
    hW hMem
  let δ : Fin B → Audit → ℝ := fun b a => match a with
    | .sampled => (ENNReal.ofReal δI + S.err b).toReal
    | .pouw => δs + n * (δbind + δks)
    | .pous => (Pous.Params.δ : ℝ) + Pous.Params.εMax.toReal
    | .poub => (((k - 1 : ℕ) : ℝ) / N) ^ sB + (sB : ℝ) / 2 ^ h +
        (((d.rate .oracle * (s.block b).win.len).choose k : ℕ) : ℝ) / 2 ^ lam
    | .network => nc * (δc + εks + εbind + εtap)
  have hδ : ∀ b a, (s.block b).Sound (c b) p a (δ b a) := by
    intro b a
    cases a
    · exact Bridge.IntegrityOfIntegrityBlock ι d α C I Ω (s.block b) S b q δI hq hδI (hqt b) (hA3 b) (RI b)
        (hT b) (hL b) (c b) p (hcert b) (hfin b)
    · exact Bridge.WorkOfH1CheckedCreditPeriod p macs u bUnit ι α I Ω d C (s.block b) (c b) Sc Cd hn hN hr hM hK prm enc
        staged flat δs δbind δks (ws b) (opened b) (salt b) (Wp b) (Pl b) (folded b) (help b) (RW b) (hcost b)
        (hhelp b) hScope hρ hw hluck (hgame b)
        (hdead b) (hpre b) (hbind b) (hreg b) (hdraw b) (hsvc b) (hsalt b) hpins (hmu b) (hrb b) (hguess b) (hfix b)
    · exact Bridge.SpaceOfPeriodHeld hd hp Dr Qr Lr hP ι α d C I Ω (s.block b) (ΩW b) (ΩQ b) (run b) (P b)
        (round b) (c b) p (RS b) (hdev b) (hS b)
    · exact Bridge.BandwidthOfAuditPerRunPeriod N bits h sB G k lam ι α d C I Ω (s.block b) (c b) p hN0 (RB b) (hQ b)
    · exact Bridge.CommOfPeriodLeakAccepted (c b) p (RN b) hks hbc hδc hdec εks εbind εtap (herr b) (hbindN b) (htap b)
  refine (JointHorizonBlocks ι α I Ω d C B s c p f δ εWipe δPous hδ hW hMem).trans_eq ?_
  simp only [sum_univ_audit]
  rfl

/-- **Demo A's composed theorem, every leg from its protocol's own guarantee**
(`Accounting.Guarantees.JointHorizonBlocksDeployedA`). -/
theorem JointHorizonBlocksDeployedA : Accounting.Guarantees.JointHorizonBlocksDeployedA := by
  intro ι α I _ _ _ Ω _ _ d C B r c p
    S q δI hq hδI hqt hA3 RI hT hL hcert hfin
    ι' K T L U Cn Ξ V _ _ _ _ _ _ _ Sc Cd n hn hN hr hM hK prm enc staged flat δs δbind δks ws opened salt Wp Pl folded
    macs u bUnit help RW hcost hhelp hScope hρ hw hluck hgame hdead hpre hbind hreg hdraw hsvc hsalt hpins hmu hrb hguess hfix
    Coin Col Tn _ _ _ _ _ H D Ln Kₒ Sn lay kh δc decode regN Dj nc RN hks hbc hδc hdec εks εbind εtap herr hbindN htap
  let δ : Fin B → Audit → ℝ := fun b a => match a with
    | .sampled => (ENNReal.ofReal δI + S.err b).toReal
    | .pouw => δs + n * (δbind + δks)
    | .network => nc * (δc + εks + εbind + εtap)
    | _ => 0
  have hδ : ∀ b, ∀ a ∈ ({.sampled, .pouw, .network} : Finset Audit), (r b).Sound (c b) p a (δ b a) := by
    intro b a ha
    simp only [Finset.mem_insert, Finset.mem_singleton] at ha
    rcases ha with rfl | rfl | rfl
    · exact Bridge.IntegrityOfIntegrityBlock ι d α C I Ω (r b) S b q δI hq hδI (hqt b) (hA3 b) (RI b)
        (hT b) (hL b) (c b) p (hcert b) (hfin b)
    · exact Bridge.WorkOfH1CheckedCreditPeriod p macs u bUnit ι α I Ω d C (r b) (c b) Sc Cd hn hN hr hM hK prm enc
        staged flat δs δbind δks (ws b) (opened b) (salt b) (Wp b) (Pl b) (folded b) (help b) (RW b) (hcost b)
        (hhelp b) hScope hρ hw hluck (hgame b)
        (hdead b) (hpre b) (hbind b) (hreg b) (hdraw b) (hsvc b) (hsalt b) hpins (hmu b) (hrb b) (hguess b) (hfix b)
    · exact Bridge.CommOfPeriodLeakAccepted (c b) p (RN b) hks hbc hδc hdec εks εbind εtap (herr b) (hbindN b) (htap b)
  refine (JointHorizonBlocksA ι α I Ω d C B r c p δ hδ).trans_eq (Finset.sum_congr rfl fun b _ => ?_)
  rw [Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton]
  exact (add_assoc _ _ _).symm

end Accounting.SecurityProofs
