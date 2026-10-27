import Proofs.Properties.Verity.SampledProofs
import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Bridge.Unopened
import Proofs.Properties.Verity.PlayFinder
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Soundness.Audit.Circuit
import Properties.Verity.Extract
import Proofs.Flock.Bridge.Lowers
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg

/-!
# Sampled proofs' target at the deployment, with no hypothesis on the strategy

`Vocab.SampledProofsSound` doesn't hold at every vocabulary: at `delta = 0`, `loss = 0`, `crBound = 1` and a forger
that never succeeds the error is `0`, and an extractor that returns a wrong transcript fails the claim at an honest play that passes. It holds at a
vocabulary whose sampled proofs' fields are the deployment's, C-Flock's live session at the committed files' parse
(`SampledDeploy`). `sampledProofsSound'` is the target there, at every `θ`, `F`, `c` and strategy, with no hypothesis
on the strategy.

`SampledDeploy` holds, for files that parse (`Vocab.parsed`), the session's setting at the parse (`A`, with its named
assumptions A3 and `RecordCustodyZKJ`), the session's strategy the play induces (`τOf`) and what the play's collision
is read off (`col`, `findBreak` at a `.collision` witness); the facts relating them that no lemma gives yet,
each its owner's:

* `accept_le`: outside the collision (`ColEvent col`) and the vocabulary's forgery (`Vocab.Forges .sampled`), the
  verdict at most as likely as the session's acceptance under `--registered` at the verifier's roots `own`
  (`acc_of_accepts` and a decode of `ComponentAccepts`; the coupling of the device's releases to the session's
  coins and `DrawnByDevice`);
* `finder_eq`: the vocabulary's sampled finder is C-Flock's finders at the session's strategy, uniformly mixed, with
  the replay finder of `col` (`Vocab.replayFinder`) in the spare slot (`LiveAt.crFinder`), so that one finder bounds
  both the play's collision and C-Flock's loss (`break_le_of`); the forgery is the vocabulary's own term
  (`Vocab.forges`), never reduced to a collision;
* `unbound_le`: at each outcome the session accepts, the unbound units at most the session's faulty ones there
  (`LiveAt.faultyE`, under the witness at the claims' values: the registrant's rows at the registered reads the verifier
  can hold to them, `LiveAt.vreg_of_read`; `unbound_le_of`, `place`'s openings bridge);

the bound `kMax θ` on `J²·(20 J + 2 |Pos| + 2)` at registered files (`kMax_ge`, `J` the session's tables, `Pos` its
committed positions; it bounds `J²·K`, `LiveAt.K_le`), the additive loss (`loss_ge`: `kMax θ · p` plus C-Flock's loss
`lossPubR` at `kMax θ · p`, stated here once until `Vocab` carries it), the deployment's constants (`k = kDep`,
`M = R_w = 2^192`, `ρ = 1/2`, at most `2^40` committed wires), `t'` the strategy's declared cost; and what the product
publishes: `ε ≥ 0` and `δ` at least the miss plus `2^-100` at costs up to `2^80`, `⊤` above. No collision-resistance
premise enters: `SampledKSLivePub` takes none. From these `SampledDeploy.sampledIn` builds the coupling `SampledIn` at
every strategy: `units_eq` is `unitsOfFile_eq`, `hε` is `LiveAt.sound_honest` at `K₀ = ⌊ε n⌋ + 1`, `break_le` is
`break_le_of` and `delta_le` is `deltaPubR_le_deployed`. Files that don't parse have no units, and files that fail the commit phase's check no verdict.

The honest side: `honestHolds_sampled_of_bits` is `honestHolds_sampled` at an extractor that reads wire bits off the
registrant's openings (`valsOfBits`), from the honest prover's bits being the key circuit's evaluation and the honest
commitment opening every gate that isn't a constant to them; the parse's constants are input gates
(`parsedConsts_isInput`) and their openings hold (`bits_of_consts`), and no draw enters.
`SampledDeploy.honestHolds_sampled_of_rows` is it at the registrant's rows: the bits of the registrant's openings `wo`
at each wire's committed position (`LiveAt.rowBits`), the reference `rowsWrongZ` reads at the registered reads
(`LiveAt.eq_rowBits_of_not_mem`), with no extracted transcript.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **The committed files' parse**, as `Vocab.read` reads them. -/
noncomputable def Vocab.parsed (V : Vocab d α) (F : ByteArray) : Option Flock.Bridge.Parsed :=
  Flock.Bridge.parsedOf (V.read F).tags (V.read F).circuitFile (V.read F).tables (V.read F).publicFile

/-- **`deltaPubR` at the deployment's constants** is the draw's miss plus at most `2^-100`: its rows term is at most
its link term, each at most `2^40·2^-150` (`Flock.Bridge.link_le`). -/
theorem deltaPubR_le_deployed (n kd K₀ W : ℕ) (hW : W ≤ 2 ^ 40) :
    Flock.Bridge.deltaPubR n kd K₀ Flock.Bridge.kDep (2 ^ 192) (2 ^ 192) W (1 / 2) ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(100 : ℝ)) := by
  unfold Flock.Bridge.deltaPubR Flock.Bridge.deltaPub
  have hl := Flock.Bridge.link_le Flock.Bridge.kDep_le (t' := 0) (by norm_num)
  rw [zero_mul, mul_zero, zero_div, zero_add] at hl
  have hW' : (W : ℝ) ≤ 2 ^ 40 := by exact_mod_cast hW
  have h1 : 0 ≤ 1 / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) := by positivity
  have hk0 : 0 ≤ (Flock.Bridge.kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) := by positivity
  have h150 : (2 : ℝ) ^ (-(150 : ℝ)) = 1 / 2 ^ 150 := by
    rw [Real.rpow_neg (by norm_num)]; norm_num
  rw [h150] at hl
  have hWl := mul_le_mul hW' hl (by positivity) (by positivity)
  have hWk := mul_le_mul hW' (show (Flock.Bridge.kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) ≤ 1 / 2 ^ 150 by
    linarith) hk0 (by positivity)
  have h203 : (2 : ℝ) ^ (-(203 : ℝ)) = 1 / 2 ^ 203 := by
    rw [Real.rpow_neg (by norm_num)]; norm_num
  have h100 : (2 : ℝ) ^ (-(100 : ℝ)) = 1 / 2 ^ 100 := by
    rw [Real.rpow_neg (by norm_num)]; norm_num
  have e2 : ∀ x : ℝ, x / (1 - 1 / 2) = 2 * x := fun x => by ring
  rw [h203, h100, e2, e2, e2]
  have h2 : (2 : ℝ) * (1 / 2 ^ 203) + 2 * (2 ^ 40 * (1 / 2 ^ 150)) + 2 * (2 ^ 40 * (1 / 2 ^ 150)) ≤ 1 / 2 ^ 100 := by
    norm_num
  linarith

/-- **`deltaPubR` at the session's deployed constants** is the draw's miss plus at most `2^-100`. -/
theorem deltaPubR_le_of {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ}
    {desc : Claims.Msg} (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc)
    (hk : A.k = Flock.Bridge.kDep) (hM : A.M = 2 ^ 192) (hRw : A.Rw = 2 ^ 192) (hρ : A.ρ = 1 / 2)
    (hW : Flock.Bridge.ioSum A.P ≤ 2 ^ 40) :
    Flock.Bridge.deltaPubR n kd K₀ A.k A.M A.Rw (Flock.Bridge.ioSum A.P) A.ρ ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(100 : ℝ)) := by
  rw [hk, hM, hRw, hρ]
  exact deltaPubR_le_deployed n kd K₀ _ hW

/-- Equal circuits, unit counts, partitions, openings and transcripts have as many unbound units. -/
theorem card_unbound_congr {C C' : Verity.Circuit} (hC : C = C') {n n' : ℕ} (hn : n = n')
    {P : Verity.Partition C n} {P' : Verity.Partition C' n'} (hP : HEq P P')
    {O : (g : Fin C.N) → BitVec (C.w g) → Prop} {O' : (g : Fin C'.N) → BitVec (C'.w g) → Prop} (hO : HEq O O')
    {T : C.Transcript} {T' : C'.Transcript} (hT : HEq T T') :
    (Accounting.unbound P O T).card = (Accounting.unbound P' O' T').card := by
  subst hC hn
  obtain rfl := eq_of_heq hP
  obtain rfl := eq_of_heq hO
  obtain rfl := eq_of_heq hT
  rfl

/-- **The replay finder**: on a uniform tape of the play, it plays `σ` and outputs what `col` reads off the outcome,
at the declared cost `k` (`Game.tapeFinder` at the play). -/
noncomputable def Vocab.replayFinder (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (col : Outcome → Option (Claims.Msg × Claims.Msg)) (k : ℕ) : Claims.Finder :=
  Game.tapeFinder (V.game θ F c) σ col fun _ => k

/-- **The collision `col` reads off the outcome.** -/
def ColEvent (col : Outcome → Option (Claims.Msg × Claims.Msg)) (o : Outcome) : Prop :=
  ∃ u v, col o = some (u, v) ∧ Claims.Collision sha512Msg u v

/-- **The play's collision event has the replay finder's probability** (`Game.prob_eq_ofReal_Pr`). -/
theorem prob_colEvent_eq (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (col : Outcome → Option (Claims.Msg × Claims.Msg)) (k : ℕ) :
    Game.prob (ColEvent col) (V.game θ F c) σ = ENNReal.ofReal ((V.replayFinder θ F c σ col k).Pr sha512Msg) :=
  Game.prob_eq_ofReal_Pr sha512Msg (V.game θ F c) σ col fun _ => k

/-- The sampling claims' SHA-512 is the vocabulary's. -/
theorem sampling_sha512Msg_eq : Claims.Sampling.sha512Msg = sha512Msg := rfl

/-- **`break_le` from the vocabulary's finder.** Where the vocabulary's sampled finder at `σ` is C-Flock's finders at
the session's strategy `τ` with the replay finder of `col` in the spare slot (`hfe`, `LiveAt.crFinder`), `kMax` is at
least `J²·(20 J + 2 |Pos| + 2)` (`hkm`, so at least `K` and `J²·K`, `LiveAt.K_le`), the session is at the deployment's
`k`, `ρ` and at most `2^40` committed wires, and the loss at `p` is at least `kMax·p` plus `lossPubR` at `kMax·p`
(`hloss`: the collision term is additive, never `loss = 0`), the collision event `col` reads off the play, with
C-Flock's loss at its finder `B^τ`, is at most the loss at the vocabulary's finder: `SampledIn.break_le` at
`Brk := ColEvent col`. The replay finder succeeds at most `K` times as often as the mixture (`LiveAt.extra_le`) and
`B^τ` at most as often (`LiveAt.pubB_le_crFinder`). -/
theorem break_le_of (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (col : Outcome → Option (Claims.Msg × Claims.Msg)) {n : ℕ} {D : Flock.Desc}
    {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) (τ : FlockSoundness.Game.Strategy A.G) (kMax : ℕ)
    (hfe : V.finder .sampled θ F c σ = A.crFinder τ (V.replayFinder θ F c σ col (V.cost θ F c σ)))
    (hkm : A.J ^ 2 * (20 * A.J + 2 * Fintype.card A.vb.Pos + 2) ≤ kMax)
    (hk : A.k = Flock.Bridge.kDep) (hρ : A.ρ = 1 / 2) (hW : Flock.Bridge.ioSum A.P ≤ 2 ^ 40)
    (hloss : ∀ p : ℝ, 0 ≤ p → ENNReal.ofReal (kMax * p) +
      ENNReal.ofReal (Flock.Bridge.lossPubR Flock.Bridge.kDep (2 ^ 40) (1 / 2) (kMax * p)) ≤
        V.loss .sampled θ (ENNReal.ofReal p)) :
    Game.prob (ColEvent col) (V.game θ F c) σ +
        ENNReal.ofReal (Flock.Bridge.lossPubR A.k (Flock.Bridge.ioSum A.P) A.ρ (A.lossArg τ)) ≤
      V.loss .sampled θ ((V.live θ F c).collides sha512Msg .sampled σ) := by
  have hx0 : 0 ≤ (V.finder .sampled θ F c σ).Pr sha512Msg := Claims.Pr_nonneg _
  have hJ : 1 ≤ A.J ^ 2 :=
    Nat.one_le_iff_ne_zero.mpr (pow_ne_zero 2 (FlockSoundness.Discharge.Composed.nTab_pos A.I).ne')
  have hKN : A.K ≤ kMax := A.K_le.trans ((le_mul_of_one_le_left (Nat.zero_le _) hJ).trans hkm)
  have hJKN : A.J ^ 2 * A.K ≤ kMax := (Nat.mul_le_mul le_rfl A.K_le).trans hkm
  have hK : (A.K : ℝ) ≤ kMax := by exact_mod_cast hKN
  have hJK : (A.J : ℝ) ^ 2 * A.K ≤ kMax := by exact_mod_cast hJKN
  have hcol : Game.prob (ColEvent col) (V.game θ F c) σ ≤
      ENNReal.ofReal (kMax * (V.finder .sampled θ F c σ).Pr sha512Msg) := by
    rw [prob_colEvent_eq V θ F c σ col (V.cost θ F c σ)]
    refine ENNReal.ofReal_le_ofReal ((A.extra_le sha512Msg τ _).trans ?_)
    rw [← hfe]
    exact mul_le_mul_of_nonneg_right hK hx0
  have hpub : (A.pubB τ).Pr Claims.Sampling.sha512Msg ≤ (V.finder .sampled θ F c σ).Pr sha512Msg := by
    rw [hfe, ← sampling_sha512Msg_eq]
    exact A.pubB_le_crFinder _ τ _
  have harg : A.lossArg τ ≤ kMax * (V.finder .sampled θ F c σ).Pr sha512Msg :=
    (mul_le_mul_of_nonneg_left hpub (by positivity)).trans (mul_le_mul_of_nonneg_right hJK hx0)
  have hl : Flock.Bridge.lossPubR A.k (Flock.Bridge.ioSum A.P) A.ρ (A.lossArg τ) ≤
      Flock.Bridge.lossPubR Flock.Bridge.kDep (2 ^ 40) (1 / 2) (kMax * (V.finder .sampled θ F c σ).Pr sha512Msg) := by
    rw [hk, hρ]
    exact Flock.Bridge.lossPubR_mono (by norm_num) hW (A.lossArg_nonneg τ) harg
  exact (add_le_add hcol (ENNReal.ofReal_le_ofReal hl)).trans (hloss _ hx0)

/-- The values the session's witness at the outcome `o` opens: its openings in the sampling statement, at the claims'
values `vreg τ o`. -/
noncomputable abbrev _root_.Flock.Guarantees.LiveAt.valE {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit}
    {kd K₀ : ℕ} {desc : Claims.Msg} (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) (q : Claims.Sampling.Query)
    (τ : FlockSoundness.Game.Strategy A.G) (o : A.Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) :
    ℕ → Option Bool :=
  (A.wit τ o).val Claims.Sampling.sha512Msg A.Cd (Flock.Bridge.stmtOf A.P A.κ kd desc q A.roots)

/-- **`unbound_le` from the extractor and the openings, at an outcome.** Where the vocabulary's extractor is the
transcript of the values the session's witness opens at `o` (`hX`, the vocabulary's `extract`), and the vocabulary's
openings are openings `O` of the parse's circuit (`hO`) that hold wherever that witness opens a value other than a
constant (`hopen`) and at the constants (`hconst`) (`place`'s openings bridge), the unbound units are at most the
session's faulty ones at `o` (`Flock.Bridge.card_unbound_le_faulty`). -/
theorem unbound_le_of {V : Vocab d α} (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    {r : Flock.Bridge.Parsed} (hr : V.parsed F = some r) {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt r.n r.D r.hU kd K₀ desc) (q : Claims.Sampling.Query)
    (τ : FlockSoundness.Game.Strategy A.G) (o : A.Reg × Finset (Fin r.n) × FlockSoundness.Discharge.Exec.Transcript)
    (hX : HEq (V.extract θ F c σ) (Flock.Bridge.transcriptOf r.D r.hU r.n A.κ (A.valE q τ o)))
    (O : (g : Fin (Flock.Bridge.circuitOf r.D r.hU r.n).N) → BitVec ((Flock.Bridge.circuitOf r.D r.hU r.n).w g) → Prop)
    (hO : HEq (V.Opens θ F c) O)
    (hopen : ∀ (g : Fin A.Cc.N) (b : Bool),
      (∀ b', (Flock.Bridge.progOf A.Cc A.κ).gates[Flock.Bridge.posOf A.Cc g]? ≠ some (.const b')) →
      A.valE q τ o (Flock.Bridge.posOf A.Cc g) = some b → O g (BitVec.ofBool b))
    (hconst : ∀ (g : Fin A.Cc.N) (b : Bool),
      (Flock.Bridge.progOf A.Cc A.κ).gates[Flock.Bridge.posOf A.Cc g]? = some (.const b) →
      O g (Flock.Bridge.transcriptOf r.D r.hU r.n A.κ (A.valE q τ o) g)) :
    (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card ≤ (A.faultyE q τ o).card := by
  have hC : V.circuitOf F = Flock.Bridge.circuitOf r.D r.hU r.n := by
    show Flock.Bridge.Parsed.circuit (V.parsed F) = _
    rw [hr]
    rfl
  have hn : V.units F = r.n := by
    show Flock.Bridge.Parsed.units (V.parsed F) = _
    rw [hr]
    rfl
  have hP : HEq (V.partsOf F) (Flock.Bridge.partsOf r.D r.hU r.n) := by
    show HEq (Flock.Bridge.Parsed.parts (V.parsed F)) _
    rw [hr]
    rfl
  refine (card_unbound_congr hC hn hP hO hX).le.trans ?_
  exact Flock.Bridge.card_unbound_le_faulty A.P A.κ kd desc q A.roots Claims.Sampling.sha512Msg A.Cd
    (A.wit τ o) O hopen hconst

theorem transcriptAt_heq {o : Option Flock.Bridge.Parsed}
    (T : ∀ r, o = some r → (Flock.Bridge.circuitOf r.D r.hU r.n).Transcript) {r : Flock.Bridge.Parsed}
    (h : o = some r) : HEq (transcriptAt o T) (T r h) := by
  subst h
  rfl

/-- The parse's constants (`parsedConsts`) are the session's (`Flock.Bridge.constsOf`). -/
theorem constsOf_eq_parsedConsts (r : Flock.Bridge.Parsed)
    (g : Fin (FlockSoundness.Discharge.Layout.keyN r.D.unroll r.hU r.n)) :
    Flock.Bridge.constsOf r.D.unroll r.hU r.n g = parsedConsts (some r) g := by
  simp only [Flock.Bridge.constsOf, parsedConsts]
  split_ifs <;> simp_all [Fin.ext_iff, FlockSoundness.Discharge.Layout.oneG, FlockSoundness.Discharge.Layout.zeroG]

theorem transcriptOf_parsedConsts (r : Flock.Bridge.Parsed) (val : ℕ → Option Bool) :
    Flock.Bridge.transcriptOf r.D r.hU r.n (fun g => parsedConsts (some r) g) val =
      Flock.Bridge.transcriptOf r.D r.hU r.n (fun g => Flock.Bridge.constsOf r.D.unroll r.hU r.n g) val := by
  congr 1
  funext g
  exact (constsOf_eq_parsedConsts r g).symm

/-! ## The extractor's completeness -/

/-- Each unit's local evaluation of an evaluation is that evaluation. -/
theorem localEval_evalAudit {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (a : Fin C.N → Bool) (u : Fin n) (g : Fin C.N) : P.localEval (C.eval a) u g = C.eval a g := by
  suffices h : ∀ m (g : Fin C.N), g.val = m → P.localEval (C.eval a) u g = C.eval a g from h _ g rfl
  intro m
  refine Nat.strong_induction_on m ?_
  intro m ih g hg
  by_cases hu : P.unit g = some u
  · have hin : (C.op g).isInput = false := by
      cases h : (C.op g).isInput
      · rfl
      · have h0 := (P.input_iff g).2 h
        rw [hu] at h0
        exact absurd h0 (Option.some_ne_none u)
    rw [P.localEval_of_mem _ hu, C.eval_op a hin]
    exact FlockSoundness.Audit.Op.apply_congr _ fun x hx =>
      ih x.val (by have hx' := C.topo g x hx; rw [Fin.lt_def] at hx'; omega) x rfl
  · exact P.localEval_of_not_mem _ hu

/-- An evaluation has no wrong unit. -/
theorem wrong_evalAudit {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (a : Fin C.N → Bool) : P.wrong (C.eval a) = ∅ := by
  ext u
  simp only [FlockSoundness.Audit.Partition.mem_wrong, Finset.notMem_empty, iff_false, not_not]
  exact fun g _ _ => (localEval_evalAudit P a u g).symm

/-- Equal circuits, unit counts, partitions and transcripts have as many wrong units. -/
theorem card_wrong_congr {C C' : Verity.Circuit} (hC : C = C') {n n' : ℕ} (hn : n = n')
    {P : Verity.Partition C n} {P' : Verity.Partition C' n'} (hP : HEq P P') {T : C.Transcript} {T' : C'.Transcript}
    (hT : HEq T T') : (P.wrong T).card = (P'.wrong T').card := by
  subst hC hn
  obtain rfl := eq_of_heq hP
  obtain rfl := eq_of_heq hT
  rfl

/-- **The deployment's extractor is complete where the opened values are an evaluation.** At files that parse to `r`,
where the wire values `vals` opens at `σ`, read at the parse's constants (`Flock.Bridge.openedOf`), are the key
circuit's evaluation on some inputs `xv` (`hval`), every unit of `extractOf`'s transcript is correct. -/
theorem correct_extractOf (read : ByteArray → FileRead) (rounds : ByteArray → List ℕ)
    (vals : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (play (rounds θ) θ F c)) (r : Flock.Bridge.Parsed),
      Flock.Bridge.parsedOf (read F).tags (read F).circuitFile (read F).tables (read F).publicFile = some r →
      ℕ → Option Bool)
    (θ F c : ByteArray) (σ : Game.Strategy (play (rounds θ) θ F c)) {r : Flock.Bridge.Parsed}
    (hr : Flock.Bridge.parsedOf (read F).tags (read F).circuitFile (read F).tables (read F).publicFile = some r)
    (xv : Fin (Flock.Bridge.keyCircuit r.D r.hU r.n).N → Bool)
    (hval : ∀ g, Flock.Bridge.openedOf (Flock.Bridge.keyCircuit r.D r.hU r.n) (fun g => parsedConsts (some r) g)
      (vals θ F c σ r hr) g = (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g) :
    ∀ u, (read F).parts.Correct (extractOf read rounds vals θ F c σ) u := by
  have hC : (read F).circuit = Flock.Bridge.circuitOf r.D r.hU r.n := by
    show Flock.Bridge.Parsed.circuit
      (Flock.Bridge.parsedOf (read F).tags (read F).circuitFile (read F).tables (read F).publicFile) = _
    rw [hr]
    rfl
  have hn : (read F).units = r.n := by
    show Flock.Bridge.Parsed.units
      (Flock.Bridge.parsedOf (read F).tags (read F).circuitFile (read F).tables (read F).publicFile) = _
    rw [hr]
    rfl
  have hP : HEq (read F).parts (Flock.Bridge.partsOf r.D r.hU r.n) := by
    show HEq (Flock.Bridge.Parsed.parts
      (Flock.Bridge.parsedOf (read F).tags (read F).circuitFile (read F).tables (read F).publicFile)) _
    rw [hr]
    rfl
  have hT : HEq (extractOf read rounds vals θ F c σ)
      (Flock.Bridge.transcriptOf r.D r.hU r.n (fun g => parsedConsts (some r) g) (vals θ F c σ r hr)) :=
    transcriptAt_heq _ hr
  have hw : ((read F).parts.wrong (extractOf read rounds vals θ F c σ)).card = 0 := by
    rw [card_wrong_congr hC hn hP hT, Flock.Bridge.transcriptOf, Flock.Bridge.partsOf_wrong, funext hval,
      wrong_evalAudit] <;> exact Finset.card_empty
  intro u
  by_contra hu
  have hmem : u ∈ (read F).parts.wrong (extractOf read rounds vals θ F c σ) := (read F).parts.mem_wrong.2 hu
  rw [Finset.card_eq_zero.1 hw] at hmem
  exact Finset.notMem_empty u hmem

/-- **Opened values that are an evaluation read as it.** Where the constants `κ` fix only input gates (`hκin`) and
agree with the circuit's evaluation on `xv` (`hκ`), and `val` opens that evaluation at every other wire's position
(`hw`), the wire values opened at `val`, read at `κ`, are that evaluation. -/
theorem openedOf_eq_eval (C : FlockSoundness.Audit.Circuit) (κ : Fin C.N → Option Bool) (val : ℕ → Option Bool)
    (xv : Fin C.N → Bool) (hκin : ∀ g b, κ g = some b → (C.op g).isInput = true)
    (hκ : ∀ g b, κ g = some b → C.eval xv g = b)
    (hw : ∀ g, κ g = none → val (Flock.Bridge.posOf C g) = some (C.eval xv g)) (g : Fin C.N) :
    Flock.Bridge.openedOf C κ val g = C.eval xv g := by
  unfold Flock.Bridge.openedOf
  cases hk : κ g with
  | some b => simp [hκin g b hk, hκ g b hk]
  | none => simp [hw g hk]

/-! ## The extractor at the registrant's openings -/

/-- **The parse's constants are inputs**: its constant's and zero's gates are input gates of the key circuit. -/
theorem parsedConsts_isInput (r : Flock.Bridge.Parsed) (g : Fin (Flock.Bridge.keyCircuit r.D r.hU r.n).N) (b : Bool)
    (h : parsedConsts (some r) g = some b) : ((Flock.Bridge.keyCircuit r.D r.hU r.n).op g).isInput = true := by
  have h01 : g.val = 0 ∨ g.val = 1 := by
    simp only [parsedConsts] at h
    split_ifs at h with h1 h2
    · exact Or.inl (by simpa [FlockSoundness.Discharge.Placed.oneGate] using h1)
    · exact Or.inr (by simpa [FlockSoundness.Discharge.Placed.zeroGate] using h2)
    all_goals simp at h
  have h2 : 2 ≤ FlockSoundness.Discharge.Placed.inGates r.n
      (FlockSoundness.Discharge.Placed.circuitRows r.D.unroll r.hU).nIn := Nat.le_add_right 2 _
  have hg : g.val < FlockSoundness.Discharge.Placed.inGates r.n
      (FlockSoundness.Discharge.Placed.circuitRows r.D.unroll r.hU).nIn := by
    omega
  have hu : (FlockSoundness.Discharge.Placed.keyProg r.D.unroll r.hU r.n).unit g = none :=
    FlockSoundness.Discharge.Layout.unit_uniform_input r.n g hg
  exact (FlockSoundness.Prog.input_iff _ g).1 hu

open Classical in
/-- **Wire bits as opened values**: `w g` at wire `g`'s position (`posOf`), nothing at any other position. -/
noncomputable def valsOfBits (C : FlockSoundness.Audit.Circuit) (w : Fin C.N → Bool) (p : ℕ) : Option Bool :=
  if h : ∃ g, Flock.Bridge.posOf C g = p then some (w h.choose) else none

theorem valsOfBits_posOf (C : FlockSoundness.Audit.Circuit) (w : Fin C.N → Bool) (g : Fin C.N) :
    valsOfBits C w (Flock.Bridge.posOf C g) = some (w g) := by
  have h : ∃ g', Flock.Bridge.posOf C g' = Flock.Bridge.posOf C g := ⟨g, rfl⟩
  rw [valsOfBits, dif_pos h, Flock.Bridge.posOf_injective h.choose_spec]

/-- **At a constant, an evaluation's transcript holds the constant**: where the opened values `val` read as the key
circuit's evaluation on `xv` (`hev`), whose constants are the parse's (`hκ`), a transcript equal to theirs (`hT`) holds
at each of the parse's constants its constant's bit. -/
theorem bits_of_consts {o : Option Flock.Bridge.Parsed} {r : Flock.Bridge.Parsed} (ho : o = some r)
    (val : ℕ → Option Bool) (xv : Fin (Flock.Bridge.keyCircuit r.D r.hU r.n).N → Bool)
    (hev : ∀ g, Flock.Bridge.openedOf (Flock.Bridge.keyCircuit r.D r.hU r.n) (fun g => parsedConsts (some r) g) val g =
      (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g)
    (hκ : ∀ g b, parsedConsts (some r) g = some b → (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g = b)
    (T : (Flock.Bridge.Parsed.circuit o).Transcript)
    (hT : HEq T (Flock.Bridge.transcriptOf r.D r.hU r.n (fun g => parsedConsts (some r) g) val))
    (g : Fin (Flock.Bridge.Parsed.circuit o).N) (b : Bool) (hb : parsedConsts o g = some b)
    (i : Fin ((Flock.Bridge.Parsed.circuit o).w g)) : (T g).getLsbD i.val = b := by
  subst ho
  obtain rfl := eq_of_heq hT
  have hi : i.val = 0 := by
    have : i.val < 1 := i.isLt
    omega
  show (BitVec.ofBool (Flock.Bridge.openedOf _ _ val g)).getLsbD i.val = b
  rw [hev g, hκ g b hb, hi]
  simp

/-- **Sampled proofs' honest lemma at the registrant's openings** (`Vocab.HonestHolds commit .sampled`). Where the
vocabulary's extractor is `extractOf` at wire bits `w` (`hext`: at files that parse to `r`, the commitment `c` and the
strategy `σ`, the bits of the registrant's openings, `valsOfBits`), and at registered files that parse, the honest
prover's bits are the key circuit's evaluation on inputs `xv` that agree with the parse's constants (`h`), and the
honest commitment opens every gate that isn't a constant to the extractor's bit there (`hopen`, `Vocab.Opens`), the
honest prover meets sampled proofs' claim on every play (`honestHolds_sampled`); at the constants `Vocab.Opens` holds
by `h`. The bits are defined at every wire, so no draw enters: nothing asks that a draw read each committed value. -/
theorem honestHolds_sampled_of_bits (V : Vocab d α) (commit : ByteArray → ByteArray → ByteArray)
    (hε : ∀ θ, 0 ≤ (V.certOf θ).ε)
    (w : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Fin (Flock.Bridge.keyCircuit r.D r.hU r.n).N → Bool)
    (hext : V.extract = extractOf V.read V.rounds
      fun θ F c σ r hr => valsOfBits (Flock.Bridge.keyCircuit r.D r.hU r.n) (w θ F c σ r hr))
    (h : ∀ θ F, V.Registered θ F → ∀ (r : Flock.Bridge.Parsed) (hr : V.parsed F = some r),
      ∃ xv : Fin (Flock.Bridge.keyCircuit r.D r.hU r.n).N → Bool,
        (∀ g b, parsedConsts (some r) g = some b → (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g = b) ∧
        ∀ g, parsedConsts (some r) g = none →
          w θ F (commit θ F) (V.honest θ F (commit θ F)) r hr g = (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g)
    (hopen : ∀ θ F, V.Registered θ F → ∀ g, (V.read F).consts g = none →
      V.Opens θ F (commit θ F) g (V.extract θ F (commit θ F) (V.honest θ F (commit θ F)) g)) :
    V.HonestHolds commit .sampled := by
  refine honestHolds_sampled V commit hε fun θ F hreg => ?_
  cases hr : V.parsed F with
  | none =>
    have hn : V.units F = 0 := by
      show Flock.Bridge.Parsed.units (V.parsed F) = 0
      rw [hr]
      rfl
    have hN : (V.circuitOf F).N = 0 := by
      show (Flock.Bridge.Parsed.circuit (V.parsed F)).N = 0
      rw [hr]
      rfl
    exact ⟨fun u => absurd u.isLt (by omega), fun g => absurd g.isLt (by omega)⟩
  | some r =>
    obtain ⟨xv, hκ, hw⟩ := h θ F hreg r hr
    let vals : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed),
        V.parsed F = some r → ℕ → Option Bool :=
      fun θ F c σ r hr => valsOfBits (Flock.Bridge.keyCircuit r.D r.hU r.n) (w θ F c σ r hr)
    have hev : ∀ g, Flock.Bridge.openedOf (Flock.Bridge.keyCircuit r.D r.hU r.n) (fun g => parsedConsts (some r) g)
        (vals θ F (commit θ F) (V.honest θ F (commit θ F)) r hr) g = (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g :=
      openedOf_eq_eval _ _ _ xv (parsedConsts_isInput r) hκ fun g hk => by
        show valsOfBits _ _ _ = _
        rw [valsOfBits_posOf, hw g hk]
    have hT : HEq (V.extract θ F (commit θ F) (V.honest θ F (commit θ F)))
        (Flock.Bridge.transcriptOf r.D r.hU r.n (fun g => parsedConsts (some r) g)
          (vals θ F (commit θ F) (V.honest θ F (commit θ F)) r hr)) := by
      rw [hext]
      exact transcriptAt_heq _ hr
    refine ⟨?_, fun g => ?_⟩
    · rw [hext]
      exact correct_extractOf V.read V.rounds vals θ F (commit θ F) (V.honest θ F (commit θ F)) hr xv hev
    · cases hc : (V.read F).consts g with
      | none => exact hopen θ F hreg g hc
      | some b =>
        unfold Vocab.Opens
        rw [hc]
        exact bits_of_consts (o := V.parsed F) hr _ xv hev hκ _ hT g b hc

/-- **The registrant's rows as wire bits**: at each wire, the bit of the registrant's opening `wo` at the wire's
committed position, the reference `rowsWrongZ` reads at a registered read. -/
noncomputable def _root_.Flock.Guarantees.LiveAt.rowBits {n : ℕ} {D : Flock.Desc}
    {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) (wo : A.vb.Pos → A.vb.Opening) (g : Fin A.Cc.N) : Bool :=
  A.vb.bit (A.vb.val (wo (A.vb.pos g))) g

open FlockSoundness FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkHidden in
/-- **Off `rowsWrongZ`, a unit's wires carry the registrant's rows** at every registered read. -/
theorem _root_.Flock.Guarantees.LiveAt.eq_rowBits_of_not_mem {n : ℕ} {D : Flock.Desc}
    {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
    (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) (isRead : A.vb.Pos → Prop) (wo : A.vb.Pos → A.vb.Opening)
    (Xv : Fin A.Cc.N → Bool) {u : Fin n}
    (hu : u ∉ rowsWrongZ execArith A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb isRead wo Xv)
    {g : Fin A.Cc.N} (hg : g ∈ A.P.io u) (hrd : isRead (A.vb.pos g)) : Xv g = A.rowBits wo g := by
  by_contra hne
  exact hu ((mem_rowsWrongZ execArith A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb isRead wo).2 ⟨g, hg, hrd, hne⟩)

/-- **Sampled proofs at the deployment.** At files that parse to `r` (`Vocab.parsed`), every `θ`, `c` and strategy
`σ`: C-Flock's live session at the parse (`A`, `K₀ = ⌊ε n⌋ + 1`, its cost `t'` the strategy's), the session's
strategy the play induces (`τOf`), what the play's collision is read off (`col`), and the facts relating them
(`accept_le`, `finder_eq`, `unbound_le`), each its owner's to prove; the bound on C-Flock's finders (`kMax_ge`); the
additive loss (`loss_ge`); the deployment's constants; and what the product publishes about sampled proofs
(`ε_nonneg`, `delta_ge`, `delta_top`). -/
structure SampledDeploy (V : Vocab d α) where
  kd : ByteArray → ℕ
  desc : ByteArray → Claims.Msg
  q : ByteArray → Claims.Sampling.Query
  A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
    Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ)
  τOf : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), FlockSoundness.Game.Strategy (A θ F c σ r hr).G
  col : ByteArray → ByteArray → ByteArray → Outcome → Option (Claims.Msg × Claims.Msg)
  accept_le : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), V.Registered θ F →
    Game.prob (fun o => V.Verdict .sampled o ∧ ¬ ColEvent (col θ F c) o ∧ ¬ V.Forges .sampled θ o) (V.game θ F c) σ ≤
      FlockSoundness.Game.prob
        (fun o => FlockSoundness.Discharge.ZkReg.LiveAcceptsZKR (A θ F c σ r hr).I (A θ F c σ r hr).own o)
        (A θ F c σ r hr).G (τOf θ F c σ r hr)
  finder_eq : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), V.Registered θ F →
    V.finder .sampled θ F c σ = (A θ F c σ r hr).crFinder (τOf θ F c σ r hr)
      (V.replayFinder θ F c σ (col θ F c) (V.cost θ F c σ))
  kMax : ByteArray → ℕ
  kMax_ge : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), V.Registered θ F →
    (A θ F c σ r hr).J ^ 2 * (20 * (A θ F c σ r hr).J + 2 * Fintype.card (A θ F c σ r hr).vb.Pos + 2) ≤ kMax θ
  loss_ge : ∀ θ (p : ℝ), 0 ≤ p → ENNReal.ofReal (kMax θ * p) +
    ENNReal.ofReal (Flock.Bridge.lossPubR Flock.Bridge.kDep (2 ^ 40) (1 / 2) (kMax θ * p)) ≤
      V.loss .sampled θ (ENNReal.ofReal p)
  unbound_le : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
    (hr : V.parsed F = some r), V.Registered θ F → ∀ o,
    FlockSoundness.Discharge.ZkReg.LiveAcceptsZKR (A θ F c σ r hr).I (A θ F c σ r hr).own o →
    (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card ≤
      ((A θ F c σ r hr).faultyE (q θ) (τOf θ F c σ r hr) o).card
  k_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).k = Flock.Bridge.kDep
  M_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).M = 2 ^ 192
  Rw_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).Rw = 2 ^ 192
  ρ_eq : ∀ θ F c σ r hr, (A θ F c σ r hr).ρ = 1 / 2
  ioSum_le : ∀ θ F c σ r hr, Flock.Bridge.ioSum (A θ F c σ r hr).P ≤ 2 ^ 40
  t'_eq : ∀ θ F c σ r hr τ, (A θ F c σ r hr).t' τ = V.cost θ F c σ
  ε_nonneg : ∀ θ, 0 ≤ (V.certOf θ).ε
  delta_ge : ∀ (θ F : ByteArray) (r : Flock.Bridge.Parsed), V.parsed F = some r → V.Registered θ F →
    ∀ t : ℕ, t ≤ 2 ^ 80 →
      ENNReal.ofReal (((r.n - (⌊(V.certOf θ).ε * r.n⌋₊ + 1)).choose (kd θ) : ℝ) / (r.n.choose (kd θ) : ℝ) +
        2 ^ (-(100 : ℝ))) ≤ V.delta θ .sampled t
  delta_top : ∀ (θ : ByteArray) (t : ℕ), 2 ^ 80 < t → V.delta θ .sampled t = ⊤

namespace SampledDeploy

variable {V : Vocab d α}

/-- **The coupling at every strategy**, at registered files that parse. -/
noncomputable def sampledIn (Dp : SampledDeploy V) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (r : Flock.Bridge.Parsed) (hr : V.parsed F = some r) (hreg : V.Registered θ F) : SampledIn V θ F c σ where
  n := r.n
  D := r.D
  hU := r.hU
  kd := Dp.kd θ
  K₀ := ⌊(V.certOf θ).ε * r.n⌋₊ + 1
  desc := Dp.desc θ
  A := Dp.A θ F c σ r hr
  τ := Dp.τOf θ F c σ r hr
  q := Dp.q θ
  units_eq := by
    show Flock.Bridge.unitsOfFile _ _ _ _ = r.n
    exact Flock.Bridge.unitsOfFile_eq hr
  hε := Flock.Guarantees.LiveAt.sound_honest (n := r.n) _ (mul_nonneg (Dp.ε_nonneg θ) (Nat.cast_nonneg _))
  Brk := ColEvent (Dp.col θ F c)
  accept_le := Dp.accept_le θ F c σ r hr hreg
  break_le := break_le_of V θ F c σ (Dp.col θ F c) (Dp.A θ F c σ r hr) (Dp.τOf θ F c σ r hr) (Dp.kMax θ)
    (Dp.finder_eq θ F c σ r hr hreg) (Dp.kMax_ge θ F c σ r hr hreg) (Dp.k_eq θ F c σ r hr) (Dp.ρ_eq θ F c σ r hr)
    (Dp.ioSum_le θ F c σ r hr) (Dp.loss_ge θ)
  unbound_le := Dp.unbound_le θ F c σ r hr hreg
  delta_le := by
    by_cases hc : V.cost θ F c σ ≤ 2 ^ 80
    · exact (ENNReal.ofReal_le_ofReal (deltaPubR_le_of _ (Dp.k_eq θ F c σ r hr) (Dp.M_eq θ F c σ r hr)
        (Dp.Rw_eq θ F c σ r hr) (Dp.ρ_eq θ F c σ r hr) (Dp.ioSum_le θ F c σ r hr))).trans
        (Dp.delta_ge θ F r hr hreg _ hc)
    · rw [Dp.delta_top θ _ (not_le.1 hc)]
      exact le_top

/-- **Sampled proofs' honest lemma at the registrant's rows** (`Vocab.HonestHolds commit .sampled`):
`honestHolds_sampled_of_bits` at the bits of the registrant's openings `wo` (`LiveAt.rowBits`, the reference
`rowsWrongZ` reads at the registered reads). Where the vocabulary's extractor reads those bits (`hext`), and at
registered files that parse, the registrant's rows at the honest commitment are the key circuit's evaluation on inputs
`xv` that agree with the parse's constants (`h`), and the honest commitment opens every gate that isn't a constant to
the extractor's bit there (`hopen`), the honest prover meets sampled proofs' claim on every play. No extracted
transcript and no draw enters. -/
theorem honestHolds_sampled_of_rows (Dp : SampledDeploy V) (commit : ByteArray → ByteArray → ByteArray)
    (wo : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), (Dp.A θ F c σ r hr).vb.Pos → (Dp.A θ F c σ r hr).vb.Opening)
    (hext : V.extract = extractOf V.read V.rounds fun θ F c σ r hr =>
      valsOfBits (Flock.Bridge.keyCircuit r.D r.hU r.n) ((Dp.A θ F c σ r hr).rowBits (wo θ F c σ r hr)))
    (h : ∀ θ F, V.Registered θ F → ∀ (r : Flock.Bridge.Parsed) (hr : V.parsed F = some r),
      ∃ xv : Fin (Flock.Bridge.keyCircuit r.D r.hU r.n).N → Bool,
        (∀ g b, parsedConsts (some r) g = some b → (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g = b) ∧
        ∀ g, parsedConsts (some r) g = none →
          (Dp.A θ F (commit θ F) (V.honest θ F (commit θ F)) r hr).rowBits
            (wo θ F (commit θ F) (V.honest θ F (commit θ F)) r hr) g = (Flock.Bridge.keyCircuit r.D r.hU r.n).eval xv g)
    (hopen : ∀ θ F, V.Registered θ F → ∀ g, (V.read F).consts g = none →
      V.Opens θ F (commit θ F) g (V.extract θ F (commit θ F) (V.honest θ F (commit θ F)) g)) :
    V.HonestHolds commit .sampled :=
  honestHolds_sampled_of_bits V commit Dp.ε_nonneg
    (fun θ F c σ r hr => (Dp.A θ F c σ r hr).rowBits (wo θ F c σ r hr)) hext h hopen

end SampledDeploy

/-- **Sampled proofs' target at the deployment** (`Vocab.SampledProofsSound`), with no hypothesis on the
strategy: at every vocabulary whose sampled proofs' fields are the deployment's (`SampledDeploy`), at every `θ`, `F`,
`c` and strategy, sampled proofs' verdict accepts while more than an `ε` share of the units are unbound in the
extractor's transcript with probability at most `Vocab.error .sampled`: `δ`, the collision finder's loss, and the LMS
forger's success probability (`euf-cma/lms-sha512-n64`). -/
theorem sampledProofsSound' (V : Vocab d α) (Dp : SampledDeploy V) : V.SampledProofsSound := fun θ F c σ => by
  show Game.prob (fun o => V.Verdict .sampled o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
    V.error .sampled θ F c σ
  by_cases hreg : V.Registered θ F
  · cases hr : V.parsed F with
    | some r => exact (Dp.sampledIn θ F c σ r hr hreg).le_error
    | none =>
      have h0 : V.units F = 0 := by
        show Flock.Bridge.Parsed.units (V.parsed F) = 0
        rw [hr]
        rfl
      have hH : ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
          (V.certOf θ).ε * V.units F := by
        have hc : (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card ≤ V.units F :=
          (Finset.card_le_univ _).trans_eq (Fintype.card_fin _)
        have hc' : ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
            ((V.units F : ℕ) : ℝ) := by exact_mod_cast hc
        have h0' : ((V.units F : ℕ) : ℝ) = 0 := by rw [h0, Nat.cast_zero]
        rw [h0', mul_zero]
        exact hc'.trans h0'.le
      refine (Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hH) _ σ).trans ?_
      rw [Game.prob_false]
      exact zero_le
  · exact (prob_play_eq_zero (V.rounds θ) θ F c _ (fun _ _ _ h => hreg h.1.1) σ).trans_le zero_le

end Verity
