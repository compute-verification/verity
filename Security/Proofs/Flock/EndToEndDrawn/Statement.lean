import Proofs.Flock.Soundness.Discharge.Composed.DrawnJ

/-!
# C-Flock's one-stage audit, end to end, at the drawn units (trusted)

`EndToEndDrawn` is `EndToEnd`'s statement with its event at the drawn units: flock-verify accepts and some drawn unit is
wrong, in place of "at least `K₀` units are wrong". Its proof is
`Flock.SecurityProofs.EndToEndDrawn : Flock.Guarantees.EndToEndDrawn`.

C-Flock's definitions are still in its proofs package, so this module imports
`Proofs.Flock.Soundness.Discharge.Composed.DrawnJ`, and `Models.lean`, which the `security` package builds without
`security_proofs`, does not import it: `Proofs.Flock.EndToEndDrawn` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **C-Flock's one-stage audit, end to end, at the drawn units.** As in `EndToEnd`: the prover registers `R`; the
verifier draws a `kd`-subset of the `n` proof units from its operating system's bytes (`Law.execOS` on the byte source
`src`); one `--zk` session of `J = nTab I` tables proves the drawn units, `kd / J` in each; and
`flock-verify verify --zk --session-tables J` reads the session's record. Then for every live prover `τ`,
$$\Pr[\text{flock-verify accepts} \wedge \text{some drawn unit is wrong}] \le \varepsilon_{ks} + \delta_{link},$$
with "wrong" `EndToEnd`'s (`wrongRegZ`, against the public file, in the committed transcript fixed at registration), and
`ε_ks`, `δ_link` `EndToEnd`'s terms, averaged over the verifier's draw. The bound has no miss term, so it holds whatever
the verifier's bytes: it needs no A3.

Its named assumptions, each a hypothesis:
* `live-verifier` (`Assumptions.Zk.RecordCustodyZKJ`, with A6 `uniform/flock-coin-server` for its coins): the record
  `flock-verify` reads is the one the coin server wrote for the live session, and its draw file the one
  `flock-verify draw` printed for the verifier's own draw;
* `cr/sha-512` (`Assumptions.SHA512CRStrict`, through `TableCRZ`) for the knowledge finders of each table of the
  prover's session, at budgets `qF` (rewinding) and `qS` (self-clash);
* `ecr/sha-512` (A2, `Assumptions.SHA512CRExpected`, through `LinkCRZC`) for the prover's link finders.

The rest are the verify command's tags (`TagsOkZJ`), an untyped statement whose outputs are public, the bound's numeric
conditions, and `EndToEnd`'s reference setup `y₀`: table 0's, at the session's link points, at some draw `d₀`, which
every draw flock-verify accepts gives (`Flock.SecurityProofs.EndToEnd_refSetup`), at any number of tables. -/
def EndToEndDrawn : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (c : Flock.Circuit) (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOk c = true)
    -- the law: a `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the bound's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)))
    (qF qS : ℝ),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)
    let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZJ dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZJ dj y₀) (tabZJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => LiveAcceptsZK I o ∧ ∃ u ∈ o.2.1, u ∈ wrongRegZ execArith P (planZJ dj y₀) (tabZJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZJ dj y₀) (tabZJ dj y₀) vb (reg σ) (cont σ) k Rw))
        (auditLive L Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZJ dj y₀) (tabZJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

end Flock.Guarantees
