import Proofs.Flock.Soundness.Discharge.Composed.PublicJ

/-!
# C-Flock's one-stage audit, end to end (trusted)

`EndToEnd` is a `def G : Prop` in the form of `Pouw.Guarantees.EndToEnd`: stated over C-Flock's definitions, with the
named assumptions it rests on as hypotheses. Its proof is `Flock.SecurityProofs.EndToEnd : Flock.Guarantees.EndToEnd`.

C-Flock's definitions are still in its proofs package, so this module imports
`Proofs.Flock.Soundness.Discharge.Composed.PublicJ`, and `Models.lean`, which the `security` package builds without
`security_proofs`, does not import it: `Proofs.Flock.EndToEnd` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **C-Flock's one-stage audit, end to end.** The prover registers `R` (its roots); the verifier draws a uniform
`kd`-subset of the `n` proof units from its operating system's bytes (`Flock.Draw.drawOS`, `Law.execOS` on the byte
source `src`); one `--zk` session of `J = nTab I` tables proves the drawn units, `kd / J` in each; and the verifier of
record, `flock-verify verify --zk --session-tables J`, reads the session's record. Then for every live prover `τ`,
$$\Pr[\text{flock-verify accepts} \wedge \text{at least } K_0 \text{ units wrong}] \le
  \frac{\binom{n-K_0}{kd}}{\binom{n}{kd}} + \varepsilon_{ks} + \delta_{link},$$
* a unit wrong (`wrongRegZ`) unless it is right against the public file: in the committed transcript fixed at
  registration (`XplurZC`: the plurality of what the expected-time extractor recovers behind each commit string) its
  committed outputs are its gates applied to its committed inputs, and one opening per commit string of the public
  file's commitments carries, on each of its committed wires, the key program's evaluation on the committed inputs (on
  its outputs, the program's output on them). The commitments are the public file's (`rsZJ_wrongRegZ`): at a row its
  `b ‖ c`, an `hm96` commitment to the row's bytes under a 192-byte salt; elsewhere the SHA-512 leaf of the public
  output word's bits, or of the constant `1` or `0`. So at a unit that is not wrong, the program's output on the
  committed inputs is the public output and its constant columns hold their constants, each up to a SHA-512 collision
  on its leaf. The committed inputs are bound to nothing, not even to the public file's rows: `hm96` hides (under
  `hash-derived-key` an honest `b ‖ c` has an opening to any fixed row value except with probability `2^-192` over its
  salt), so at the input rows the opening conjunct adds essentially nothing to correctness in `XplurZC`. That the
  committed inputs are the data owner's bytes is not stated here. The units are those of the key program of the circuit
  the verifier parses (`keyProg c hU n`, partitioned at `Layout.outsOf`), each read at its `--zk` site in the table it
  is drawn into and bound to the rows it commits;
* `ε_ks = ksAvgStrictZ`: the worst table's knowledge error, `ksStrictZ / (1 − N₀/(ek))`, averaged over the draw;
* `δ_link = Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`, `Q_s` the expected number of commit strings the drawn
  units read.

Its named assumptions, each a hypothesis:
* A3 (`uniform/io-getrandombytes`, `Assumptions.UniformRandomBytes`): the verifier's bytes are uniform;
* `live-verifier` (`Assumptions.Zk.RecordCustodyZKJ`, with A6 `uniform/flock-coin-server` for its coins): the record
  `flock-verify` reads is the one the coin server wrote for the live session, and its draw file the one
  `flock-verify draw` printed for the verifier's own draw;
* `cr/sha-512` (`Assumptions.SHA512CRStrict`, through `TableCRZ`) for the knowledge finders of each table of the
  prover's session, at budgets `qF` (rewinding) and `qS` (self-clash);
* `ecr/sha-512` (A2, `Assumptions.SHA512CRExpected`, through `LinkCRZC`) for the prover's link finders.

The rest are the verify command's tags (`TagsOkZJ`), an untyped statement whose outputs are public, the count curve's
numeric conditions, and a reference setup `y₀`: table 0's, at the session's link points, at some draw `d₀`, which the
session's plan puts in place of a table that a draw doesn't set up. Every draw flock-verify accepts gives one
(`Flock.SecurityProofs.EndToEnd_refSetup`), at any number of tables. -/
def EndToEnd : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (c : Flock.Circuit) (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOk c = true)
    -- the law: a uniform `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
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
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZ execArith P (planZJ dj y₀) (tabZJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZJ dj y₀) (tabZJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgStrictZ (planZJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZJ dj y₀) (tabZJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

end Flock.Guarantees
