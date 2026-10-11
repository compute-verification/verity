import Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJ

/-!
# C-Flock's one-stage audit, end to end, with registered reads (trusted)

`EndToEndRegistered` is `EndToEndHidden`'s statement for `flock-verify verify --zk --registered F` sessions, over a
public file with or without shared-row tables, and with the registered check's consequence at every accepted outcome
as a second conjunct, and `EndToEndRegisteredDrawn` is the same statement with its first conjunct at the drawn units
(`EndToEndDrawn`'s event). Their proofs are
`Flock.SecurityProofs.EndToEndRegistered : Flock.Guarantees.EndToEndRegistered` and
`Flock.SecurityProofs.EndToEndRegisteredDrawn : Flock.Guarantees.EndToEndRegisteredDrawn`.

C-Flock's definitions are still in its proofs package, so this module imports
`Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJ`, and `Models.lean`, which the `security` package builds without
`security_proofs`, does not import it: `Proofs.Flock.EndToEndRegistered` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **C-Flock's one-stage audit, end to end, with registered reads.** As in `EndToEndHidden`, for
`flock-verify verify --zk --registered F --session-tables J` (`own` F's ports), over hidden outputs and a public file
with or without shared-row tables. Every row is registered under the public file's `b ‖ c` (`rsZHJS_wrongRegZ`). Then
for every live prover `τ`:
1. $$\Pr[\text{flock-verify accepts} \wedge \text{at least } K_0 \text{ units wrong}] \le
  \frac{\binom{n-K_0}{kd}}{\binom{n}{kd}} + \varepsilon_{ks} + \delta_{link},$$
   with "wrong" `EndToEndHidden`'s (`wrongRegZ`, against the public file, in the committed transcript fixed at
   registration) at the rows `rsZHJS`, and `ε_ks`, `δ_link` `EndToEnd`'s terms;
2. at every outcome flock-verify accepts, each registered read that is the verifier's own at the draw of the table its
   unit is drawn into (`OwnReadsHJ`) is checked at the outcome's registration (`Registered.Reads.Checked`): the SHA-512
   leaf of the public file's `b ‖ c` at the read row opens the root `own` holds, at the position `own` gives, through
   the path that table's drawn statement gives (`regPathsHJ`).
The first conjunct binds no row: `hm96` hides, so the opening half of `wrongRegZ` adds essentially nothing to
correctness in `XplurZC` (`EndToEndHidden`). The second says that at a read the verifier owns, the public file's
`b ‖ c` is the one registered under F's root. That the committed values are the rows that `b ‖ c` commits is not
stated here.

Its named assumptions, each a hypothesis, are `EndToEnd`'s, for the first conjunct; the second reads only the
executable's checks:
* A3 (`uniform/io-getrandombytes`, `Assumptions.UniformRandomBytes`): the verifier's bytes are uniform;
* `live-verifier` (`Assumptions.Zk.RecordCustodyZKJ`, with A6 `uniform/flock-coin-server` for its coins): the record
  `flock-verify` reads is the one the coin server wrote for the live session, and its draw file the one
  `flock-verify draw` printed for the verifier's own draw;
* `cr/sha-512` (`Assumptions.SHA512CRStrict`, through `TableCRZ`) for the knowledge finders of each table of the
  prover's session, at budgets `qF` (rewinding) and `qS` (self-clash);
* `ecr/sha-512` (A2, `Assumptions.SHA512CRExpected`, through `LinkCRZC`) for the prover's link finders.

The rest are the verify command's tags under hidden outputs (`TagsOkZHJ`), an untyped statement, the count curve's
numeric conditions, and `EndToEndHidden`'s reference setup `y₀`: table 0's, at the session's link points, at some draw
`d₀`, which every draw flock-verify accepts gives (`Flock.SecurityProofs.EndToEndHidden_refSetup`), at any number of
tables. -/
def EndToEndRegistered : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit, which the verifier parses, its public file, and F's ports
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (c : Flock.Circuit) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH c = true)
    -- the law: a uniform `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZHJS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)
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
    prob (fun o => LiveAcceptsZKR I own o ∧
        K₀ ≤ (wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) ∧
    ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
          (rsShHJS dj y₀ ht hc hscope hU hpub),
        OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port)

open Classical in
/-- **C-Flock's one-stage audit, end to end, with registered reads, at the drawn units.** `EndToEndRegistered`'s
setting, with its first conjunct at the drawn units: for every live prover `τ`,
1. $$\Pr[\text{flock-verify accepts} \wedge \text{some drawn unit is wrong}] \le \varepsilon_{ks} + \delta_{link},$$
   with "wrong" `EndToEndRegistered`'s (`wrongRegZ` at the rows `rsZHJS`), and `ε_ks`, `δ_link` `EndToEnd`'s terms,
   averaged over the verifier's draw;
2. `EndToEndRegistered`'s second conjunct: at every outcome flock-verify accepts, each registered read the verifier owns
   (`OwnReadsHJ`) is checked at the outcome's registration (`Registered.Reads.Checked`, through `regPathsHJ`).
The first conjunct's bound has no miss term, so it holds whatever the verifier's bytes: it needs no A3, and the draw need
not be uniform.

Its named assumptions, each a hypothesis, are `EndToEndDrawn`'s, for the first conjunct; the second reads only the
executable's checks: `live-verifier` (`Assumptions.Zk.RecordCustodyZKJ`, with A6 `uniform/flock-coin-server` for its
coins), `cr/sha-512` (`Assumptions.SHA512CRStrict`, through `TableCRZ`) for each table's knowledge finders at budgets
`qF` and `qS`, and `ecr/sha-512` (A2, `Assumptions.SHA512CRExpected`, through `LinkCRZC`) for the prover's link finders.
The rest are the verify command's tags under hidden outputs (`TagsOkZHJ`), an untyped statement, the bound's numeric
conditions, and `EndToEndHidden`'s reference setup `y₀`: table 0's, at the session's link points, at some draw `d₀`,
which every draw flock-verify accepts gives (`Flock.SecurityProofs.EndToEndHidden_refSetup`), at any number of
tables. -/
def EndToEndRegisteredDrawn : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit, which the verifier parses, its public file, and F's ports
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (c : Flock.Circuit) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH c = true)
    -- the law: a `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
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
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZHJS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)
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
    prob (fun o => LiveAcceptsZKR I own o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) ∧
    ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
          (rsShHJS dj y₀ ht hc hscope hU hpub),
        OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port)

end Flock.Guarantees
