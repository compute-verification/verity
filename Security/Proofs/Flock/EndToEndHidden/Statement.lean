import Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ
import Proofs.Flock.Soundness.Assumptions.Files
import Proofs.Flock.Bridge.RowsC
import Flock.Desc

/-!
# C-Flock's one-stage audit, end to end, under hidden outputs (trusted)

`EndToEndHidden` is `EndToEnd`'s statement under hidden outputs, the configuration production `--zk` runs (the default
`--statement`, whose tables `Stmt.setupHidden` sets up), and `EndToEndHiddenDrawn` is `EndToEndDrawn`'s, the same
statement at the drawn units. Both read the circuit through its descriptor (`Flock.Desc`): the circuit file is read as
a descriptor `D` and the units are `D.unroll`'s. `EndToEndHiddenFlat` and `EndToEndHiddenDrawnFlat` are the two at a
flat circuit file, `D = Desc.single c`, stated over the circuit `c` that `HmOut.parse` reads. Their proofs are
`Flock.SecurityProofs.EndToEndHidden : Flock.Guarantees.EndToEndHidden`,
`Flock.SecurityProofs.EndToEndHiddenDrawn : Flock.Guarantees.EndToEndHiddenDrawn` and the two `Flat` corollaries.

`EndToEndHiddenModel` is `EndToEndHidden`'s count curve at the model level: over the values the verifier parses (any
partition, value binding, decoder, law and prover of the audit game at the hidden-output tables), with no file and no
named assumption. `EndToEndHidden` is its corollary at the deployed files (`Assumptions.Flock.DeployedFilesHJ`) and the
live session.

C-Flock's definitions are still in its proofs package, so this module imports
`Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ`, and `Models.lean`, which the `security` package builds without
`security_proofs`, does not import it: `Proofs.Flock.EndToEndHidden` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **C-Flock's one-stage audit under hidden outputs, at the model level.** `EndToEndHidden`'s count curve over the
values the verifier parses rather than its files, at the registrant's rows: at the hidden-output tables of any statement
(`planZHJ dj y₀`, at a reference setup `y₀` of table 0), for any partition `P` of a circuit into `n` units whose units
read only their own gates and input gates, any decoder the tables lower soundly to (`LoweringSoundZK`), any value
binding `vb` of `P`'s committed wires whose openings in every satisfying witness are canonical at `canon`
(`OpeningsCanon`), any set `isRead` of registered reads with the registrant's opening `wo` at each, any law `L` over the
units and any prover `σ` of the audit game,
$$\Pr[\text{the session accepts} \wedge \text{at least } K_0 \text{ units wrong}] \le
  \mathrm{miss}_L(K_0) + \varepsilon_{ks} + (\delta_{link} + \delta_{rows}),$$
with a unit wrong, in the committed transcript fixed at registration (`XplurZC`), when it is wrong on the registration
with canonical openings (`wrongRegZc`) or carries, on a committed wire at a registered read, a bit other than the
registrant's row's (`rowsWrongZ`), and `ε_ks`, `δ_link`, `δ_rows` its additive terms (`ksAvgPrZ`, `linkPrZC`,
`rowsPrZC`), each a fixed finder's probability of outputting a SHA-512 collision.

It reads no file and takes no named assumption. `EndToEndHidden` is this statement with no registered read and every
opening canonical, where "wrong" contains `EndToEndHidden`'s (`wrongRegZ_subset_wrongRegZc`) and `δ_rows = 0`
(`rowsPrZC_noReads`), at the deployed files (`Assumptions.Flock.DeployedFilesHJ`), the key program's partition, the
binding of the public file's rows and the live session, which A3 and `live-verifier` tie to the audit game. -/
def EndToEndHiddenModel : Prop :=
  ∀ (n : ℕ) (Reg : Type) (I : Inputs) (mPts : ℕ)
    -- the hidden-output tables: the draw file at each draw, and table 0's reference setup
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    -- the units, their decoder and the value binding of their committed wires
    (C : FlockSoundness.Audit.Circuit) (P : FlockSoundness.Audit.Partition C n)
    (decode : Discharge.ZkSession.DecoderZK (C := C) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀))
    (Dh : Type) (Hc : List UInt8 → Dh) (vb : ValueBindingZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode Hc)
    (_hlow : Discharge.ZkSession.LoweringSoundZK execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode)
    (_hown : ∀ u, P.ReadsOwn u)
    -- the canonical openings, and the registered reads with the registrant's openings
    (canon : Fin C.N → vb.Opening → Prop) (_hcan : OpeningsCanon execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon)
    (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening)
    -- the law
    (L : Law n)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ)
    -- the prover
    (σ : Strategy (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)))),
    prob (fun o => o.1 = true ∧
        K₀ ≤ (wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon (reg σ)
            (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw) ∪
          rowsWrongZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb isRead wo
            (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀))) σ ≤
      L.miss K₀ + ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        (linkPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb k Rw M ρ (reg σ) (cont σ) +
          rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb isRead wo k ρ (reg σ) (cont σ))

open Classical in
/-- **C-Flock's one-stage audit, end to end, under hidden outputs.** As in `EndToEnd`, over a statement whose outputs
are hidden: the circuit is the one its descriptor unrolls to (`D.unroll`, for the `D` that `Desc.parse` reads from the
circuit file, so the circuit's pin is SHA-512 of `D`), the public file is `HmOut`'s, each table is
`Stmt.setupHidden`'s, and every row, input or output, is registered under the public file's `b ‖ c`, an `hm96`
commitment to the row's bytes under a 192-byte salt (`rsZHJ_wrongRegZ`). The prover registers `R`; the verifier draws
a uniform `kd`-subset of the `n` proof units from its operating system's bytes (`Law.execOS` on the byte source `src`); one `--zk` session of `J = nTab I` tables proves the
drawn units, `kd / J` in each; and `flock-verify verify --zk --session-tables J` reads the session's record. Then for
every live prover `τ`,
$$\Pr[\text{flock-verify accepts} \wedge \text{at least } K_0 \text{ units wrong}] \le
  \frac{\binom{n-K_0}{kd}}{\binom{n}{kd}} + \varepsilon_{ks} + \delta_{link},$$
* a unit wrong (`wrongRegZ`) unless it is right against the public file: in the committed transcript fixed at
  registration (`XplurZC`) its committed outputs are its gates applied to its committed inputs, and one opening per
  commit string of the public file's commitments carries, on each of its committed wires, the key program's evaluation
  on the committed inputs. That second half binds no row: `hm96` hides (under `hash-derived-key` an honest `b ‖ c` has
  an opening to any fixed row value except with probability `2^-192` over its salt), so at every row, input or hidden
  output, the opening conjunct adds essentially nothing to correctness in `XplurZC`. That the committed inputs and the
  program's outputs on them are the rows the public file commits is not stated here. The units are those of the key
  program of the circuit the verifier's descriptor unrolls to (`keyProg D.unroll hU n`, partitioned at
  `Layout.outsOf`), each read at its `--zk` site in the hidden-output table it is drawn into (`placeZHJ`);
* `ε_ks = ksAvgPrZ` and `δ_link = Σ_c π_c·(Pr[B_c outputs a collision] + 1/(eM) + k/(eR_w))/(1 − ρ)`, in the additive
  form: each term is a fixed finder's probability of outputting a SHA-512 collision, for finders built from `τ`. In
  `ε_ks` they are each table's knowledge finders after each draw (`advRSFinderZ`, `advPadsSFinderZ`, `advTauSFinderZ`,
  `adv₀SFinderZ`, `selfSFinderZ`); in `δ_link`, `B_c` is the link finder of commit string `c` (`linkFinderPrZC`), and
  `π_c` the probability that a draw reads `c`. Collision resistance enters only outside the theorem, on those finders:
  under `cr/sha-512` and `ecr/sha-512` the two terms are at most `ksAvgStrictZ` (`ksAvgPrZ_le_strict`) and
  `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))` (`linkPrZC_le_linkBoundZC`), `EndToEnd`'s closed forms.

Its named assumptions, each a hypothesis, are:
* A3 (`uniform/io-getrandombytes`, `Assumptions.UniformRandomBytes`): the verifier's bytes are uniform;
* `live-verifier` (`Assumptions.Zk.RecordCustodyZKJ`, with A6 `uniform/flock-coin-server` for its coins): the record
  `flock-verify` reads is the one the coin server wrote for the live session, and its draw file the one
  `flock-verify draw` printed for the verifier's own draw;
* `deployed-files` (`Assumptions.Flock.DeployedFilesHJ`): the circuit and public files, pinned by their SHA-512 digests
  `pinC` and `pinP`, parse to the descriptor `D` and the public file `pub`.

The rest are the verify command's tags under hidden outputs (`TagsOkZHJ`), an untyped statement with no shared-row
tables (`pub.tables = none`), the count curve's numeric conditions, and a reference setup `y₀`: table 0's, at the
session's link points, at some draw `d₀`, which the session's plan puts in place of a table that a draw doesn't set up.
Every draw flock-verify accepts gives one (`Flock.SecurityProofs.EndToEndHidden_refSetup`), at any number of tables.

It is `EndToEndHiddenModel` at these values (`Flock.SecurityProofs.EndToEndHidden_of_model`).

A flat circuit file is the descriptor `Desc.single c`, which unrolls to `c` (`Desc.parse_flat`, `Desc.parse_single`):
`EndToEndHiddenFlat` is this statement there, over `c`. -/
def EndToEndHidden : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit's descriptor, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hU : Built D.unroll.unit) (pub : Public)
    -- deployed-files: the pinned files parse to `D` and `pub`
    (pinC pinP : ByteArray) (hFiles : Assumptions.Flock.DeployedFilesHJ I D pub pinC pinP) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the law: a uniform `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hFiles.1) hscope hU hFiles.2.1 hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hFiles.1) hscope hU hFiles.2.1 hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hFiles.1) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _
          hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hFiles.1) kd n hdiv dj hCust.2 _)) τ
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

open Classical in
/-- **`EndToEndHidden` at A3 per test**: `EndToEndHidden` with the verifier's bytes passing the draw's escape test
(`Law.escapeTest`: the live draw runs and misses every unit of `B`) at every `B` up to `η`
(`Assumptions.UniformRandomBytesTest`), in place of every event exactly, and `η` added to the miss term. -/
def EndToEndHiddenPerTest : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit's descriptor, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the law: a uniform `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    {η : ENNReal} (_hA3 : ∀ B : Finset (Fin n),
      Assumptions.UniformRandomBytesTest Lb src (Law.escapeTest (.subset kd) n B) η)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD)
          (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) + η +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

open Classical in
/-- **C-Flock's one-stage audit, end to end, under hidden outputs, at the drawn units.** `EndToEndHidden`'s setting
with `EndToEndDrawn`'s event: for every live prover `τ`,
$$\Pr[\text{flock-verify accepts} \wedge \text{some drawn unit is wrong}] \le \varepsilon_{ks} + \delta_{link},$$
with "wrong" `EndToEndHidden`'s (`wrongRegZ`, against the public file's `b ‖ c` at every row, in the committed transcript
fixed at registration), and `ε_ks`, `δ_link` `EndToEndHidden`'s terms, averaged over the verifier's draw. The bound has
no miss term, so it holds whatever the verifier's bytes: it needs no A3, and the draw need not be uniform.

Its one named assumption, a hypothesis, is `live-verifier` (`Assumptions.Zk.RecordCustodyZKJ`, with A6
`uniform/flock-coin-server` for its coins); `ε_ks` and `δ_link` are `EndToEndHidden`'s additive terms, the finders'
collision probabilities. The rest are the verify command's tags under hidden outputs (`TagsOkZHJ`), an
untyped statement with no shared-row tables (`pub.tables = none`), the bound's numeric conditions, and
`EndToEndHidden`'s reference setup `y₀`: table 0's, at the session's link points, at some draw `d₀`, which every draw
flock-verify accepts gives (`Flock.SecurityProofs.EndToEndHidden_refSetup`), at any number of tables. As in
`EndToEndHidden`, the circuit is the one the verifier's descriptor unrolls to, and `EndToEndHiddenDrawnFlat` is this
statement at a flat circuit file. -/
def EndToEndHiddenDrawn : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit's descriptor, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the law: a `kd`-subset of the `n` units, split evenly over the session's tables
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    -- the record: the draw file at each draw, and the session at each draw and registration
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
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
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    prob (fun o => LiveAcceptsZK I o ∧ ∃ u ∈ o.2.1, u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

open Classical in
/-- **`EndToEndHidden` at a flat circuit file**: `EndToEndHidden` at the descriptor `Desc.single c`, stated over the
circuit `c` that `HmOut.parse` reads from the circuit file. -/
def EndToEndHiddenFlat : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (c : Flock.Circuit) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
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
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

open Classical in
/-- **`EndToEndHiddenDrawn` at a flat circuit file**: `EndToEndHiddenDrawn` at the descriptor `Desc.single c`, stated
over the circuit `c` that `HmOut.parse` reads from the circuit file. -/
def EndToEndHiddenDrawnFlat : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (c : Flock.Circuit) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
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
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    prob (fun o => LiveAcceptsZK I o ∧ ∃ u ∈ o.2.1, u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

open Classical in
/-- **`EndToEndHiddenPerTest` at a flat circuit file**: `EndToEndHiddenPerTest` at the descriptor `Desc.single c`,
stated over the circuit `c` that `HmOut.parse` reads from the circuit file. -/
def EndToEndHiddenPerTestFlat : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- the session's statement: its circuit, which the verifier parses, and its public file
    (I : Inputs) (ht : I.tags.typed = false)
    (c : Flock.Circuit) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
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
    {η : ENNReal} (_hA3 : ∀ B : Finset (Fin n),
      Assumptions.UniformRandomBytesTest Lb src (Law.escapeTest (.subset kd) n B) η)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) + η +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

end Flock.Guarantees
