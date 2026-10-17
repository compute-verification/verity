import Proofs.Flock.Claims.Recursion.DeployZK

/-!
# Probe: what the recursion's claims say at `deployRec` is what its parameters admit

`deployRec IV ivScope b` takes the inner verifier `IV`, its scope check `ivScope` and what the deployment brings `b` as
parameters. Its scope check reads `ivScope` (`RecScope.scopeCheckOf`): at an `ivScope` that refuses everything, the
record's scope check refuses every statement (`scopeCheck_deployRec_dead`), so a claim at `deployRec` whose antecedent
is a statement the scope check takes has nothing to speak of there.

The private variant is the other such case. The compiled verifier has a private-circuit form, the holo inner shape
(`Form.Holo`, the verifier's own `--hide`, `"private_circuit": true` in its verdict), but `parse` reads no circuit
commitment at any shape (`parse_circ`), so a holo statement parses into `R`'s domain and no statement is in `RPriv`
(`RPriv_deployRec_empty`). `ZKPriv`'s bound then holds at every prover, a prover that sends its witness in the clear
among them, at every simulator and zero error (`zkPriv_deployRec_any`), and `KSPriv`'s bad event is acceptance itself
(`prBad_RPriv_deployRec`), which an accepted honest run of the private form makes large.
-/

namespace RecDeploy

open FlockDeploy (Coin)

variable {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]

/-- **Probe: at an inner scope check that refuses everything, `deployRec`'s scope check refuses every statement.** -/
theorem scopeCheck_deployRec_dead (IV : Claims.Verifier FlockDeploy.Coin) (b : Brings) (x : Claims.Msg) :
    (deployRec IV (fun _ => false) b).scopeCheck x = false := by
  show Flock.RecScope.scopeCheckOf (fun _ => false) decodeR x = false
  unfold Flock.RecScope.scopeCheckOf
  split <;> simp

/-- **Probe: `RPriv` at `deployRec` is empty**, at every inner protocol. -/
theorem RPriv_deployRec_empty (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (x : Claims.Msg) (w : W) :
    ¬ Claims.Recursion.RPriv (deployRec IV ivScope b) I x w := by
  rintro ⟨s, hs, hc, -⟩
  rw [parse_circ (x := x) hs] at hc
  cases hc

/-- **Probe: `ZKPriv`'s bound at `deployRec` holds at every prover and every simulator, at zero error.** -/
theorem zkPriv_deployRec_any (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin W) (P : W → Claims.Prover (N × Ω)) (S : Claims.Simulator) :
    Claims.ZKmalR H (deployRec IV ivScope b).V.k P (Claims.Recursion.RPriv (deployRec IV ivScope b) I) S 0 0 :=
  fun x w hR => absurd hR (RPriv_deployRec_empty IV ivScope b I x w)

/-- **Probe: `KSPriv`'s bad event at `deployRec` is acceptance**: every extractor fails `RPriv`, so `PrBad` is the
probability that the run accepts. -/
theorem prBad_RPriv_deployRec (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin W) (E : Claims.Extractor Coin W (deployRec IV ivScope b).V)
    (x : Claims.Msg) {Θ : Type} [Fintype Θ] (P : Claims.Prover Θ) :
    Claims.PrBad H (deployRec IV ivScope b).V (Claims.Recursion.RPriv (deployRec IV ivScope b) I) E x P =
      Claims.Pr fun s : Θ × (Fin ((deployRec IV ivScope b).V.k x) → Coin) × E.C x =>
        (Claims.run H P (deployRec IV ivScope b).V x s.1 s.2.1).out = .acc := by
  have h : ∀ o, ¬ (Claims.Recursion.RPriv (deployRec IV ivScope b) I).ext x o := by
    rintro (_ | w) ho
    · exact ho
    · exact RPriv_deployRec_empty IV ivScope b I x w ho
  simp only [Claims.PrBad, h, not_false_eq_true, and_true]

end RecDeploy
