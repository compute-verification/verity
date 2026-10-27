import Flock.Desc
import Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJ
import Proofs.Flock.Soundness.Discharge.ZkLink.Rows

/-!
# C-Flock's one-stage audit, end to end, with the rows link (trusted)

`EndToEndRegisteredRows` is `EndToEndRegisteredDrawn`'s statement with the rows link added to its event: at the drawn
units, an accepted session's committed input rows are the rows the registered roots open to, and its committed output
rows are the program's outputs on them, which `EndToEndHiddenDrawn` and `EndToEndRegisteredDrawn` leave unstated. Its
proof is `Flock.SecurityProofs.EndToEndRegisteredRows`. It is stated over the circuit's descriptor `D` (`Flock.Desc`),
as `EndToEndRegisteredDrawn` is, and `EndToEndRegisteredRowsFlat` is the same statement at a flat circuit file,
`D = Desc.single c`.

It is the C-Flock side of PoUW's proof premise (`Accounting.TS.ProofLayer.SoundAt`), and does not discharge it: the
bridge is open. That needs a probability bridge from a window's round and proof `(w, i)` to a C-Flock session,
the link from `ProofLayer`'s opening to the extracted plurality `XplurZC`, and a union bound over the `n` units, since
`OwnReadsHJ` here is over all of a session's registered reads, not per unit.

C-Flock's definitions are still in its proofs package, so this module imports
`Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJ`, and `Models.lean`, which the `security` package builds without
`security_proofs`, does not import it: `Proofs.Flock.EndToEndRegisteredRows` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **C-Flock's one-stage audit, end to end, with the rows link, at the drawn units.** `EndToEndRegisteredDrawn`'s
setting, for `flock-verify verify --zk --registered F --session-tables J` over hidden outputs and any public file the
verifier loads (`pub`, whatever its header carries, a salt among it), in this order:
1. **the operands' roots are registered**: `own` holds F's registered roots, and `rd`
   (`Registered.ReadsX`) is a registrant's opening of them: at each registered read `p`, the row `rd.wv p` and salt
   `rd.ws p` whose `hm96` leaf opens the root `rd.port p` at the position `rd.readAt p` (`rd.committed`). Two openings
   of one root at one position give an explicit SHA-512 collision (`Registered.TwoOpenings.collides`), so the roots fix
   these rows;
2. **then the coin**: the statement `I` (its circuit and public file) is any value, so the bound holds for a live coin
   drawn after step 1 and carried by `I`, read by the program as a registered read of the root it is registered under
   or as a constant of the circuit;
3. **then the sinks**: the live prover `τ` registers the session's rows (`reg σ`), its output rows among them, after
   `I` is fixed, and the verifier draws the units it audits.

Then for every live prover `τ`,
$$\Pr[\text{flock-verify accepts} \wedge \text{some drawn unit is wrong or its rows are off}] \le
  \varepsilon_{ks} + \delta_{link} + \delta_{rows} + \delta_{reg},$$
a drawn unit `u` wrong as in `EndToEndRegisteredDrawn` (`wrongRegZ`), and its rows off unless, at the committed
transcript fixed at registration (`XplurZC`, the plurality of the rows extraction finds), on each of `u`'s committed
wires `g` (`RowsRightZC`):
* **input rows**: where `g`'s commit string is a registered read of `rd` that the verifier owns at the outcome's draw
  (`OwnReadsHJ`, as in `EndToEndRegistered`), `g`'s committed bit is the bit of the row the registered root opens to,
  `rd.wv`;
* **output rows**: `g`'s committed bit is the circuit's evaluation on the committed inputs (`Circuit.eval`), the key
  program's output at `g` on those rows and the coin.
So when `rd` reads every input row a drawn unit's outputs depend on, its output rows are the program's outputs on the
registered rows and the coin. `ε_ks`, `δ_link` are `EndToEnd`'s terms, averaged over the verifier's draw; `δ_rows` is
`δ_link`'s closed form, for the rows finders (`rowsOutZC`: one trial at a registered read, which outputs the binding's
extractor on a row it reads there other than `rd`'s, and `rd`'s opening); and `δ_reg = q_R²/2^513` is the registered
reads' term (`Registered.ReadsX.prob_off_le`): the probability that the session's leaf at an owned registered read is
not `rd`'s, at a budget `q_R` of SHA-512 evaluations of one play of the audit and the extractor's pair.

Its named assumptions, each a hypothesis, are `EndToEndRegisteredDrawn`'s and two more instances of SHA-512's:
`live-verifier` (`Assumptions.Zk.RecordCustodyZKJ`, with A6 `uniform/flock-coin-server` for its coins), `cr/sha-512`
(`Assumptions.SHA512CRStrict`, through `TableCRZ`) for each table's knowledge finders at budgets `qF` and `qS`,
`ecr/sha-512` (A2, `Assumptions.SHA512CRExpected`, through `LinkCRZC`) for the prover's link finders; `ecr/sha-512`
(through `RowsCRZC`) for the rows finders, and `cr/sha-512` (through `StrictCR.Finder.CR`) at budget `q_R` for the
registered reads' finder, which plays the audit once and outputs the session's opening at an off read and `rd`'s
(`Registered.ReadsX.pick`). The rest are `EndToEndRegisteredDrawn`'s: the verify command's tags under hidden outputs
(`TagsOkZHJ`), an untyped statement, the bound's numeric conditions, and `EndToEndHidden`'s reference setup `y₀`, which
every draw flock-verify accepts gives (`Flock.SecurityProofs.EndToEndHidden_refSetup`). -/
def EndToEndRegisteredRows : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- F's registered roots: registered before the statement and its coin
    (own : Array Flock.Registered.Port)
    -- the session's statement, its coin among it: its circuit's descriptor, which the verifier parses, and its
    -- public file
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the law: a `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    -- the registrant's opening of `own`'s roots at the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the bound's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the live prover: it registers the session's rows, the sinks among them, after the coin
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS qR : ℝ),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub).bindingZ
      (rsZHJS_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD)
          (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => Discharge.ZkReg.drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    -- ecr/sha-512 for the rows finders, at the registrant's openings
    RowsCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) rd.isRead wo k Rw t' →
    -- cr/sha-512 for the registered reads' finder: one play of the audit, and the extractor's pair at an off read
    (StrictCR.oneRun G τ fun o => rd.pick o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port)).CR H512 qR →
    prob (fun o => LiveAcceptsZKR I own o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw) ∨
        ¬ RowsRightZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
          (fun p => OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt ∧ rd.isRead p) wo k Rw u)
        G τ ≤
      ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
        ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ))

open Classical in
/-- **`EndToEndRegisteredRows` at a flat circuit file**: `EndToEndRegisteredRows` at the descriptor `Desc.single c`,
stated over the circuit `c` that `HmOut.parse` reads from the circuit file. -/
def EndToEndRegisteredRowsFlat : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- F's registered roots: registered before the statement and its coin
    (own : Array Flock.Registered.Port)
    -- the session's statement, its coin among it: its circuit, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (c : Flock.Circuit) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH c = true)
    -- the law: a `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    -- the registrant's opening of `own`'s roots at the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the bound's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the live prover: it registers the session's rows, the sinks among them, after the coin
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS qR : ℝ),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZHJS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => Discharge.ZkReg.drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    -- ecr/sha-512 for the rows finders, at the registrant's openings
    RowsCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) rd.isRead wo k Rw t' →
    -- cr/sha-512 for the registered reads' finder: one play of the audit, and the extractor's pair at an off read
    (StrictCR.oneRun G τ fun o => rd.pick o.1 (regPathsHJ I c pub o.2.2 rd.port)).CR H512 qR →
    prob (fun o => LiveAcceptsZKR I own o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw) ∨
        ¬ RowsRightZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
          (fun p => OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt ∧ rd.isRead p) wo k Rw u)
        G τ ≤
      ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
        ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ))

end Flock.Guarantees
