import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.CircuitOf

/-!
# Sampled proofs' soundness at C-Flock's live audit, prover by prover

Sampled proofs' claim reads the values `SampledKSLivePub` states its event at (`LiveAt.vreg`): at each registered read
the registrant's row, where the registrant's leaves are the registered ones or the verifier owns the reads at the
outcome's draw, and elsewhere the circuit's evaluation on the extracted plurality's inputs; at most an `ε` share of the
circuit's units are wrong at them. Its bound holds at every prover `τ` with no hypothesis on `τ`, in
`SampledKSLivePub`'s additive form: a public error plus a loss in the probability that one finder built from `τ` and
the registrant's openings outputs a SHA-512 collision. No collision-resistance premise enters.

* `Flock.Guarantees.LiveAt n D hU kd K₀ desc`: `SampledKSLivePub`'s setting at one session (the verifier's registered
  roots `own`, the inputs flock-verify reads, the registration and the registrant's opening `rd` of its registered
  reads, the draw's setup, the verifier's bytes and A3, the analysis' `k`, `R_w`, `M`, `ρ`, and the SHA-512 evaluations
  of a run of the strategy `t'`, of a verification `v` and of a play of the audit `qR`), at the parsed circuit file `D`
  with `n` units, the draw size `kd`, `K₀` and the descriptor `desc`; `LiveAt.pubB τ`, `SampledKSLivePub`'s finder
  `B^τ` at `τ` (C-Flock's `K = LiveAt.K` finders, `Bridge.pubFinderR`); `LiveAt.ks`, its conclusion at `τ`.
* `LiveAt.wrongE q τ o`: the units wrong at the claims' values at the outcome `o`; `LiveAt.Holds q ε τ o`, the claim;
  `LiveAt.bound τ`, `deltaPubR + lossPubR(J²·K·Pr[B^τ])` (`LiveAt.err`, its loss's argument `LiveAt.lossArg`), `J` the
  session's tables (`LiveAt.J`); `lossPub` and `lossPubR` are monotone (`Bridge.lossPub_mono`, `Bridge.lossPubR_mono`).
  `LiveAt.vreg_of_read`: at a registered read the verifier can hold to the registrant, the claims read its row.
* `LiveAt.sound`: at every `τ`, `Pr[flock-verify --registered accepts ∧ ¬ Holds] ≤ bound τ`, where `K₀ − 1 ≤ ε n`.
* `LiveAt.faultyE q τ o`, `LiveAt.HoldsF`, `LiveAt.soundF`: the same on the units faulty in the sampling statement
  (wrong, or with a committed value unopened), a superset of `wrongE` (`wrongE_card_le`), from
  `Bridge.card_faulty_lt_of_relS`.
* `LiveAt.sound_honest`: `K₀ = ⌊ε n⌋ + 1` meets that hypothesis. `LiveAt.sound_nontrivial`: an accept-everything
  verifier breaks the bound at a prover at whose every outcome the claim fails, where `err τ < 1`.
* `LiveAt.crFinder τ extra`: C-Flock's finders with `extra` in `B^τ`'s spare slot, so that one finder bounds both
  `B^τ` (`pubB_le_crFinder`) and `extra` (`extra_le`, at most `K` times as often); a deployment puts the play's replay
  finder there.
-/

namespace Flock.Bridge

open Claims.Sampling

/-- **The sampling relation bounds the faulty units**: a witness in `RelS` at `δ = C(n − K₀, kd)/C(n, kd)` leaves
fewer than `K₀` of the circuit's units faulty (`card_wrong_lt_of_relS`, before its restriction to the wrong ones). -/
theorem card_faulty_lt_of_relS {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (κ : Fin C.N → Option Bool) (kd K₀ : ℕ) (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (w : Wit)
    (h : RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (stmtOf P κ kd desc q roots) (partOf P) w) :
    ((faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w).filter (· < n)).card < K₀ := by
  have hpa : ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) <
      (designOf kd).Pa (partOf P) (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w) := h
  rw [pa_partOf_window, window_zero_filter] at hpa
  generalize (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w).filter (· < n) = S at hpa ⊢
  by_contra hK
  have hc : (n - S.card).choose kd ≤ (n - K₀).choose kd := Nat.choose_le_choose _ (by omega)
  have hdiv : ((n - S.card).choose kd : ℝ) / (n.choose kd : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by
    gcongr
  linarith

/-- **`lossPub` is monotone** in the committed wires `W` and in the bound `p ≥ 0`, at a rate `ρ < 1`. -/
theorem lossPub_mono {k W W' : ℕ} {ρ p p' : ℝ} (hρ : ρ < 1) (hW : W ≤ W') (hp0 : 0 ≤ p) (hp : p ≤ p') :
    lossPub k W ρ p ≤ lossPub k W' ρ p' := by
  unfold lossPub
  have h1 : 0 < 1 - ρ := by linarith
  have hs : Real.sqrt p ≤ Real.sqrt p' := Real.sqrt_le_sqrt hp
  have hks : Real.sqrt (k * p) ≤ Real.sqrt (k * p') :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hp (Nat.cast_nonneg k))
  have hW' : (W : ℝ) ≤ W' := by exact_mod_cast hW
  have hWp : (W : ℝ) * p ≤ W' * p' := mul_le_mul hW' hp hp0 (Nat.cast_nonneg _)
  have hc : (0 : ℝ) ≤ 2 * 25638 + 2 ^ 14 + 2 := by norm_num
  have hcs := mul_le_mul_of_nonneg_left hs hc
  refine add_le_add (div_le_div_of_nonneg_right ?_ h1.le) (div_le_div_of_nonneg_right hWp h1.le)
  linarith

/-- **`lossPubR` is monotone** in the committed wires `W` and in the bound `p ≥ 0`, at a rate `ρ < 1`. -/
theorem lossPubR_mono {k W W' : ℕ} {ρ p p' : ℝ} (hρ : ρ < 1) (hW : W ≤ W') (hp0 : 0 ≤ p) (hp : p ≤ p') :
    lossPubR k W ρ p ≤ lossPubR k W' ρ p' := by
  unfold lossPubR
  have h1 : 0 < 1 - ρ := by linarith
  have hW' : (W : ℝ) ≤ W' := by exact_mod_cast hW
  have hWp : (W : ℝ) * p ≤ W' * p' := mul_le_mul hW' hp hp0 (Nat.cast_nonneg _)
  exact add_le_add (add_le_add (lossPub_mono hρ hW hp0 hp) (div_le_div_of_nonneg_right hWp h1.le)) hp

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal
open Classical

/-- **`SampledKSLivePub`'s setting at one session**: at the parsed circuit file `D` (its unit built, `hU`) with `n`
units, the draw size `kd`, `K₀` and the descriptor `desc`, everything else `SampledKSLivePub` quantifies before the
strategy, named assumption A3 included. `t' τ` bounds the SHA-512 evaluations of one run of the strategy `τ`, `v`
those of one verification, public, and `qR τ` those of one play of the audit; every finder's budget is a function of
them (`qF`, `qS`), so the bound at `τ` is at `τ`'s own cost. -/
structure LiveAt (n : ℕ) (D : Flock.Desc) (hU : Built D.unroll.unit) (kd K₀ : ℕ) (desc : Claims.Msg) where
  Reg : Type
  own : Array Flock.Registered.Port
  I : Inputs
  ht : I.tags.typed = false
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub
  hpt : pub.tables = none
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  hrows : Bridge.RowsOk D.unroll
  hdup : Flock.Registered.dupName D.unroll = none
  hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none
  qr : Lean.Json
  hpop : Bridge.PopMatches desc qr n
  hkd : kd ≤ n
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  rd : FlockSoundness.Registered.ReadsX
    (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
    (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub)
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hA3 : Assumptions.UniformRandomBytes Lb src
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hρk : rateM1 k ≤ ρ
  t' : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) → ℝ
  ht' : ∀ τ, 0 ≤ t' τ
  hK₀ : K₀ ≤ n
  v : ℝ
  hv : 0 ≤ v
  qR : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) → ℝ

attribute [instance] LiveAt.fΩ LiveAt.nΩ

namespace LiveAt

variable {n : ℕ} {D : Flock.Desc} {hU : Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
  (A : LiveAt n D hU kd K₀ desc)

/-- The draw's law: `kd` of the `n` units, from the verifier's bytes. -/
noncomputable abbrev L : Law n := Law.execOS (n := n) A.src (.subset kd) n

/-- The live audit. -/
noncomputable abbrev G := auditLive A.L A.Reg (liveOfZJ A.dg (planZHJ A.dj A.y₀) A.N A.wr)

/-- The key program's partition into the `n` units. -/
noncomputable abbrev P (_A : LiveAt n D hU kd K₀ desc) := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)

/-- The registration's binding. -/
noncomputable abbrev vb := (rsZHJ (Reg := A.Reg) A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope hU A.hpub A.hpt).bindingZ
  (rsZHJ_computes (Reg := A.Reg) A.dj A.y₀ A.ht (Flock.Desc.parse_flat A.hD) A.hscope hU A.hpub A.hpt)

/-- The simulated prover of a strategy of the live audit. -/
noncomputable abbrev sim (τ : Strategy A.G) :=
  simZJ execArith hs512 (planZHJ A.dj A.y₀) A.I A.wr (opnAtZHJ (planZHJ A.dj A.y₀) A.I A.dj) A.dg
    (decAtZJ (planZHJ A.dj A.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) A.N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ A.I A.wr _ _ _ _ _
      (runsZCJH_live A.I A.ht (Flock.Desc.parse_flat A.hD) (Discharge.Layout.scopeOkH_spec hU A.hscope).2.1 A.dj A.y₀
        A.wr _ A.hTags A.hCust.1
        fun _ => drawOkZKHJ_of_file A.I A.ht (Flock.Desc.parse_flat A.hD) kd n A.hdiv A.dj A.hCust.2 _)) τ

/-- The extracted values at a strategy. -/
noncomputable abbrev Xr (τ : Strategy A.G) :=
  XplurZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) A.k A.Rw

/-- The key program's circuit. -/
noncomputable abbrev Cc (_A : LiveAt n D hU kd K₀ desc) := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)

/-- The game's rows, read off the circuit file. -/
noncomputable abbrev rows (_A : LiveAt n D hU kd K₀ desc) : Fin _A.Cc.N → Option Bridge.RowAt := fun g =>
  (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
    ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩

/-- The registration's frame-v3 roots. -/
abbrev roots : List Claims.Msg := Bridge.rootsOf D.unroll A.pub n

/-- The deployed commitment. -/
noncomputable abbrev Cd := Bridge.commitDep A.Cc A.rows D.unroll n A.roots

/-- The circuit's constants. -/
noncomputable abbrev κ (_A : LiveAt n D hU kd K₀ desc) : Fin _A.Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g

/-- **The rewinding finders' budget at `τ`**: two runs of `τ` and the two conflicting openings verified. -/
noncomputable def qF (τ : Strategy A.G) : ℝ := 2 * (A.t' τ + A.v)

/-- **The self-clash finder's budget at `τ`**: one run of `τ` and one further verification. -/
noncomputable def qS (τ : Strategy A.G) : ℝ := A.t' τ + A.v

/-- **The session's tables**, `J`. -/
abbrev J : ℕ := nTab A.I

/-- **The registrant's openings** of the committed positions: its row `rd.wv` and data `rd.ws` there. -/
abbrev wo : A.vb.Pos → A.vb.Opening := fun p => ((p, A.rd.wv p, A.rd.ws p) : A.vb.Opening)

/-- **The claims' values at an outcome `o`**: at a registered read, the bit of the registrant's row there where the
registrant's leaves are the registered ones or the verifier owns the reads at `o`'s draw; at every other wire, its value
in the circuit's evaluation on the extracted plurality's inputs. -/
noncomputable abbrev vreg (τ : Strategy A.G) (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript)
    (g : Fin A.Cc.N) : Bool :=
  if (¬ A.rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ A.I A.own D.unroll A.pub o.2.2 A.rd.isRead A.rd.port A.rd.readAt) ∧
      A.rd.isRead (A.vb.pos g) then
    A.vb.bit (A.vb.val (A.wo (A.vb.pos g))) g
  else A.Cc.eval (A.Xr τ) g

/-- **At a registered read the verifier can hold to the registrant, the claims read the registrant's row.** -/
theorem vreg_of_read (τ : Strategy A.G) (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript) (g : Fin A.Cc.N)
    (ho : ¬ A.rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ A.I A.own D.unroll A.pub o.2.2 A.rd.isRead A.rd.port A.rd.readAt)
    (hg : A.rd.isRead (A.vb.pos g)) : A.vreg τ o g = A.vb.bit (A.vb.val (A.wo (A.vb.pos g))) g := by
  simp only [vreg, if_pos (And.intro ho hg)]

/-- **C-Flock's finders' index** at the session (`Bridge.PubIxR`): each table's knowledge finder slots, each committed
position's link and rows finders, the registered reads' finder, and one spare. -/
abbrev Ix := Bridge.PubIxR A.dj A.y₀ A.P A.vb A.L

/-- **The number of C-Flock's finders**, `K`. -/
noncomputable abbrev K : ℕ := Fintype.card A.Ix

/-- **`K ≤ 20 J + 2 |Pos| + 2`** (`Bridge.card_PubIxR_le`), `J` the session's tables and `Pos` the committed positions:
no draw, registration or strategy is in the bound. -/
theorem K_le : A.K ≤ 20 * A.J + 2 * Fintype.card A.vb.Pos + 2 :=
  Bridge.card_PubIxR_le A.dj A.y₀ A.P A.vb A.L

/-- **The registered reads' finder at `τ`**: one play of the audit at `τ`, at budget `qR τ`, outputting the registered
check's pair at an off read (`Registered.ReadsX.pick`). -/
noncomputable abbrev Breg (τ : Strategy A.G) : Claims.Finder :=
  Bridge.strictComp (StrictCR.oneRun A.G τ fun o =>
    A.rd.pick o.1 (Discharge.ZkReg.regPathsHJ A.I D.unroll A.pub o.2.2 A.rd.port)) (A.qR τ)

/-- **`SampledKSLivePub`'s finder `B^τ` at `τ`** (`Bridge.pubFinderR`): one of C-Flock's finders at the prover's run
and the registrant's openings, picked uniformly, at the budgets `qF τ`, `qS τ`, `t' τ` and `qR τ`. -/
noncomputable abbrev pubB (τ : Strategy A.G) : Claims.Finder :=
  Bridge.pubFinderR A.dj A.y₀ A.P A.vb A.wo A.k A.Rw A.M (A.qF τ) (A.qS τ) (A.t' τ) (reg (A.sim τ)) (cont (A.sim τ))
    (A.Breg τ)

/-- **The loss's argument at `τ`**: `J²·K·Pr[B^τ]`. -/
noncomputable abbrev lossArg (τ : Strategy A.G) : ℝ :=
  (A.J : ℝ) ^ 2 * A.K * (A.pubB τ).Pr Claims.Sampling.sha512Msg

theorem lossArg_nonneg (τ : Strategy A.G) : 0 ≤ A.lossArg τ :=
  mul_nonneg (by positivity) (Claims.Pr_nonneg _)

/-- **The session's error at `τ`**: `SampledKSLivePub`'s `δ + f(J²·K·Pr[B^τ])`. -/
noncomputable def err (τ : Strategy A.G) : ℝ :=
  Bridge.deltaPubR n kd K₀ A.k A.M A.Rw (Bridge.ioSum A.P) A.ρ +
    Bridge.lossPubR A.k (Bridge.ioSum A.P) A.ρ (A.lossArg τ)

set_option maxHeartbeats 1000000 in
/-- **`SampledKSLivePub` at the session**: at every strategy, flock-verify `--registered` accepts with the witness at
the claims' values `vreg` outside the sampling relation with probability at most `err τ`. -/
theorem ks (q : Claims.Sampling.Query) (τ : Strategy A.G) :
    prob (fun o => Discharge.ZkReg.LiveAcceptsZKR A.I A.own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg A.Cd
          (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (Bridge.stmtOf A.P A.κ kd desc q A.roots) (Bridge.partOf A.P)
          (Bridge.witRows A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.vreg τ o)))
      A.G τ ≤ ENNReal.ofReal (A.err τ) :=
  Flock.SecurityProofs.SampledKSLivePub n A.Reg A.own A.I A.ht D A.hD hU A.pub A.hpub A.hpt A.hscope A.hrows A.hdup
    A.hroots desc A.qr A.hpop kd A.hkd A.hdiv A.dj A.S₀ A.d₀ A.y₀ A.rd A.wr A.dg A.N A.Ω A.Lb A.src A.hA3 A.hTags A.hCust
    A.k A.Rw A.M A.hk A.hRw A.hM A.ρ A.hρ A.hρk (A.t' τ) K₀ A.hK₀ τ A.v (A.qR τ) q

/-- **C-Flock's finders at `τ`, with `extra` in the spare slot** (`Bridge.pubFamR` elsewhere). -/
noncomputable def crFam (τ : Strategy A.G) (extra : Claims.Finder) : A.Ix → Claims.Finder
  | none => extra
  | some i => Bridge.pubFamR A.dj A.y₀ A.P A.vb A.wo A.k A.Rw A.M (A.qF τ) (A.qS τ) (A.t' τ) (reg (A.sim τ))
      (cont (A.sim τ)) (A.Breg τ) (some i)

/-- **`B^τ` with `extra` in its spare slot**: one of `crFam τ extra`, picked uniformly with its own coins. -/
noncomputable def crFinder (τ : Strategy A.G) (extra : Claims.Finder) : Claims.Finder :=
  Claims.Finder.mix (A.crFam τ extra)

/-- **`extra` succeeds at most `K` times as often as `crFinder τ extra`** (`Claims.Finder.le_mix`). -/
theorem extra_le (H : Claims.Msg → Claims.Msg) (τ : Strategy A.G) (extra : Claims.Finder) :
    extra.Pr H ≤ A.K * (A.crFinder τ extra).Pr H :=
  Claims.Finder.le_mix H (A.crFam τ extra) none

/-- **`B^τ` succeeds at most as often as `crFinder τ extra`**: they differ only in the spare slot, where `B^τ` outputs
nothing (`Claims.Finder.mix_Pr`). -/
theorem pubB_le_crFinder (H : Claims.Msg → Claims.Msg) (τ : Strategy A.G) (extra : Claims.Finder) :
    (A.pubB τ).Pr H ≤ (A.crFinder τ extra).Pr H := by
  unfold pubB Bridge.pubFinderR crFinder
  rw [Claims.Finder.mix_Pr, Claims.Finder.mix_Pr]
  refine Claims.Ex_mono fun i => ?_
  rcases i with _ | i
  · have h0 : Claims.Finder.null.Pr H = 0 := by
      unfold Claims.Finder.Pr
      exact (Claims.Pr_congr (q := fun _ => False) fun _ => by simp [Claims.Finder.null]).trans Claims.Pr_false
    exact h0.le.trans (Claims.Pr_nonneg _)
  · exact le_rfl

/-- **The session's witness at an outcome**: `witRows` at the claims' values `vreg τ o`. -/
noncomputable abbrev wit (τ : Strategy A.G) (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript) :
    Claims.Sampling.Wit :=
  Bridge.witRows A.Cc A.Cd Claims.Sampling.sha512Msg A.roots (A.vreg τ o)

open Classical in
/-- **The units wrong at an outcome `o`** at `τ`: the wire values the witness at the claims' values `vreg τ o` opens
(`openedOf`), read against the key program's partition. -/
noncomputable def wrongE (q : Claims.Sampling.Query) (τ : Strategy A.G)
    (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript) : Finset (Fin n) :=
  A.P.wrong (Bridge.openedOf A.Cc A.κ
    ((A.wit τ o).val Claims.Sampling.sha512Msg A.Cd (Bridge.stmtOf A.P A.κ kd desc q A.roots)))

/-- **Sampled proofs' claim at `τ` and the outcome `o`**: at most an `ε` share of the `n` units are wrong there. -/
def Holds (q : Claims.Sampling.Query) (ε : ℝ) (τ : Strategy A.G)
    (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript) : Prop :=
  ((A.wrongE q τ o).card : ℝ) ≤ ε * n

/-- **The bound at `τ`**: `err τ`, `δ + f(J²·K·Pr[B^τ])`. -/
noncomputable def bound (τ : Strategy A.G) : ℝ≥0∞ := ENNReal.ofReal (A.err τ)

/-- **Sampled proofs' soundness at the live audit, prover by prover**: at every prover `τ`, flock-verify `--registered`
accepts while more than an `ε` share of the units are wrong at the claims' values with probability at most `bound τ`,
where `K₀ − 1 ≤ ε n`. -/
theorem sound (q : Claims.Sampling.Query) (ε : ℝ) (hε : (K₀ : ℝ) - 1 ≤ ε * n) (τ : Strategy A.G) :
    prob (fun o => Discharge.ZkReg.LiveAcceptsZKR A.I A.own o ∧ ¬ A.Holds q ε τ o) A.G τ ≤ A.bound τ := by
  refine (prob_mono (fun o ho => ⟨ho.1, fun hR => ho.2 ?_⟩) _ _).trans (A.ks q τ)
  have hlt := Bridge.card_wrong_lt_of_relS A.P A.κ kd K₀ desc q A.roots
    Claims.Sampling.sha512Msg A.Cd _ hR
  have hlt' : (A.wrongE q τ o).card < K₀ := hlt
  have h1 : ((A.wrongE q τ o).card : ℝ) + 1 ≤ K₀ := by exact_mod_cast hlt'
  show ((A.wrongE q τ o).card : ℝ) ≤ ε * n
  linarith

/-- **The units faulty in the session's sampling statement** at `τ` and the outcome `o`, under the witness at the
claims' values (`witRows` of `vreg τ o`): wrong in the values it opens, or with a committed value it reads or writes
unopened. They include the wrong ones (`wrongE_card_le`). -/
noncomputable def faultyE (q : Claims.Sampling.Query) (τ : Strategy A.G)
    (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript) : Finset ℕ :=
  (Claims.Sampling.faulty Claims.Sampling.sha512Msg A.Cd (Bridge.stmtOf A.P A.κ kd desc q A.roots) (Bridge.partOf A.P)
    (A.wit τ o)).filter (· < n)

theorem wrongE_card_le (q : Claims.Sampling.Query) (τ : Strategy A.G)
    (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript) : (A.wrongE q τ o).card ≤ (A.faultyE q τ o).card :=
  Bridge.card_wrong_le_faulty A.P A.κ kd desc q A.roots Claims.Sampling.sha512Msg A.Cd _

/-- **Sampled proofs' claim at `τ` and `o` on the faulty units**: at most an `ε` share of the `n` units faulty under
the witness at the claims' values. It implies `Holds` (`wrongE_card_le`). -/
def HoldsF (q : Claims.Sampling.Query) (ε : ℝ) (τ : Strategy A.G)
    (o : A.Reg × Finset (Fin n) × Discharge.Exec.Transcript) : Prop :=
  ((A.faultyE q τ o).card : ℝ) ≤ ε * n

/-- **`sound` on the faulty units**: at every prover `τ`, flock-verify `--registered` accepts while more than an `ε`
share of the units are faulty under the witness at the claims' values with probability at most `bound τ`, where
`K₀ − 1 ≤ ε n`. -/
theorem soundF (q : Claims.Sampling.Query) (ε : ℝ) (hε : (K₀ : ℝ) - 1 ≤ ε * n) (τ : Strategy A.G) :
    prob (fun o => Discharge.ZkReg.LiveAcceptsZKR A.I A.own o ∧ ¬ A.HoldsF q ε τ o) A.G τ ≤ A.bound τ := by
  refine (prob_mono (fun o ho => ⟨ho.1, fun hR => ho.2 ?_⟩) _ _).trans (A.ks q τ)
  have hlt := Bridge.card_faulty_lt_of_relS A.P A.κ kd K₀ desc q A.roots
    Claims.Sampling.sha512Msg A.Cd _ hR
  have hlt' : (A.faultyE q τ o).card < K₀ := hlt
  have h1 : ((A.faultyE q τ o).card : ℝ) + 1 ≤ K₀ := by exact_mod_cast hlt'
  show ((A.faultyE q τ o).card : ℝ) ≤ ε * n
  linarith

/-- **`sound`'s hypothesis holds at the deployed choice** `K₀ = ⌊ε n⌋ + 1`. -/
theorem sound_honest (ε : ℝ) (hεn : 0 ≤ ε * n) : (((⌊ε * n⌋₊ + 1 : ℕ) : ℝ)) - 1 ≤ ε * n := by
  push_cast
  linarith [Nat.floor_le hεn]

/-- **`sound` is not trivial**: for a verifier that accepts everything, its bound fails at a prover at whose every
outcome the claim fails, where `err τ < 1`. -/
theorem sound_nontrivial (q : Claims.Sampling.Query) (ε : ℝ) (τ : Strategy A.G) (hbad : ∀ o, ¬ A.Holds q ε τ o)
    (h1 : A.err τ < 1) :
    ¬ prob (fun o => True ∧ ¬ A.Holds q ε τ o) A.G τ ≤ A.bound τ := by
  intro h
  have h2 := (prob_true A.G τ).symm.le.trans
    ((prob_mono (p := fun _ => True) (q := fun o => True ∧ ¬ A.Holds q ε τ o) (fun o _ => ⟨trivial, hbad o⟩) A.G
      τ).trans h)
  exact not_le.2 (ENNReal.ofReal_lt_one.2 h1) h2

end LiveAt

end Flock.Guarantees
