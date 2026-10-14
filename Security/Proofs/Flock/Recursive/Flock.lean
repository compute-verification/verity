import Proofs.Flock.Recursive.Sound
import Proofs.Flock.Recursive.Multi
import Proofs.Flock.Soundness.Discharge.ZkReg.Sound
import Proofs.Flock.Recursive.ZkReg.Vals
import Proofs.Flock.Recursive.ZkReg.Off
import Proofs.Flock.Recursive.Finds
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
import Proofs.Flock.VBridge.Keyed

/-!
# Recursion: the outer proof, `--zk --registered` sessions of fixed public verifiers V*

The outer proof at a history `(R, ω, t)` (the registration, the draw of the hidden circuit's units, the rounds'
commitments and coins) is `S` sessions in a row, session `j` one `flock-verify verify --zk --registered F` session of
the public circuit `(Vs j).c` under its law `(Vs j).law` (`VStmt`), with hidden outputs and `(Vs j).J` tables, and with
inputs `zs R ω t j` (`ZkOuter`: public file, registered reads, the coin server's record). The audit accepts when every
session accepts (`Accepts`), and the decoded witness is each session's value layer (`XplurZC`, `ZkOuter.layer`).

* `ZkOuter.Good`: a layer with no wrong unit whose every unit reads, at each of its registered reads, the registrant's
  value (`RegAgree`), and whose constant gate is 1. `rd` records those values, the hidden circuit's rows as registered
  and the values the verifier registers itself (its coins, the comb `coef`).
* `ZkOuter.accepts_le`: an accepted session whose layer is not `Good` is at most `bound` likely: the draw misses every
  bad unit (`miss 1`); or a drawn unit computes wrongly or reads a registered value other than the registrant's at a
  leaf that is the registrant's (`zk_session_regDrawnR`: `ksAvgStrictZ + linkBoundZC + regTermZC`); or a drawn row's
  registered leaf is not the registrant's while the verifier's registered check passes (`ReadsX.prob_offOn_le`:
  `qR²/2^513`). The units are bad at the strategy alone, so one sampling step covers all three.
* `sessions`: the `S` sessions in a row, each played by a strategy the prover names at its start (`seqC`, `ofBase`);
  every strategy of a session is named (`ZkOuter.ofBase_surj`), so no prover is lost.
* `outerV`: the outer proof at a history: the prover registers `R₂`, the values V*'s sessions read that the history
  does not fix (V*'s running sums `rec-acc`, which every level's session and the algebra read, and with `link_rows`
  each level's opened rows `rec-rows-L<level>`), then V*'s sessions at `R₂` (`zs R ω t R₂ j`) in a row. `R₂` is one
  registration for all of them, sent before any session's draw, so each session's `own`, `rd` and `hown` may read its
  roots and values, and `Good` ties every session to the same values. Its binding is in each session's `bound`:
  `regTermZC` at the registrant's leaf and `qR²/2^513` off it.
* `VBridge`: what V*'s circuits compute: when every session accepts and every layer is `Good`, the messages `wmsg` reads
  from the layers hold an inner transcript the inner verifier accepts on the hidden circuit's statement `x R ω` at the
  rounds' coins.
* `zk_sessions_recursive_upto`: `recursive_sound_upto` at V*'s outer proof, its outer term the expected sum over the
  sessions of each one's `boundIf` (`bound` where its finders satisfy `cr/sha-512`, `ZkOuter.CR`, else `1`) at the
  registration the prover sends and the strategy it names there (`sumV`).
* `ZkOuter.Fork`: a run of one of `CR`'s finders or of the link finder, from the prover's own strategy, outputs a
  SHA-512 collision; with none, `CR` holds at budget `0` (`ZkOuter.cr_of_not_fork`) and `boundIf` there is at most
  `ZkOuter.boundCR`, the miss and the drawn units' statistical errors (`ZkOuter.boundIf_le`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open ZkHidden (planZHJ tabZHJ planZHJ_length DrawSetupZHJ TagsOkZHJ DrawOkZKHJ layShHJ)
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession)
open scoped ENNReal

set_option genSizeOfSpec false in
set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history, but for its record's custody**: what its soundness takes besides
the circuit `c` (fixed: V* is public), its number of tables `J`, the law `Lv` of V*'s units, the record's custody
(`ZkOuter.hRec`), the count curve and the prover's strategy. V*'s sessions hide their outputs: the circuit file parses
under `HmOut` (`hcH`), the tags are `TagsOkZHJ`, each table is set up by setupHidden (`y₀`, `hDraw`). `rd` is the
registrant's record of the registered reads: the hidden circuit's rows from its registration, the values of the outer
proof's registration `R₂` (`outerV`), and the values the verifier registers itself (the draw, the coins, the comb
`coef`). `hown`: at an accepted outcome, the reads at the rows of the drawn units are the verifier's own (`OwnReadsHJ`:
its ports, at the positions its program reads, in the table each unit is drawn into). -/
structure ZkOuterD (mPts nv J : ℕ) (Reg' : Type) (c : Flock.Circuit) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (Lv : Law nv) where
  I : Inputs
  ht : I.tags.typed = false
  own : Array Flock.Registered.Port
  dj : Finset (Fin nv) → Lean.Json
  S₀ : Finset (Fin nv)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀
  wr : (S : Finset (Fin nv)) → (R : Reg') → LiveOutJ (planZHJ (Reg := Reg') dj y₀ S R).length mPts → Exec.Transcript
  hcH : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub
  hDraw : ∀ ω, DrawOkZKHJ I dj mPts (Lv.draw ω)
  hTags : TagsOkZHJ I
  hJ : nTab I = J
  rd : FlockSoundness.Registered.ReadsX
    (layShHJ (Reg := Reg') dj y₀ ht hcH (Layout.scopeOkH_of_scopeOk hscope) hU (pub := pub))
    (rsShHJS dj y₀ ht hcH (Layout.scopeOkH_of_scopeOk hscope) hU hpub)
  hown : ∀ ω R' o, LiveAcceptsZKR I own (R', Lv.draw ω, wr (Lv.draw ω) R' o) →
    OwnReadsHJ I own c pub (wr (Lv.draw ω) R' o) (rd.restrict (rowIn (Lv.draw ω))).isRead rd.port rd.readAt

set_option genSizeOfSpec false in
set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history**: `ZkOuterD`'s, with the record's custody `hRec` (the record
`verify --zk` reads is the challenger's of the live session of `J` tables, `RecordsLiveZKJ`). -/
structure ZkOuter (mPts nv J : ℕ) (Reg' : Type) (c : Flock.Circuit) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (Lv : Law nv) extends ZkOuterD mPts nv J Reg' c hU hscope Lv where
  hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg') dj y₀ S R).length) wr

/-- **The names of a session's strategies**: a registration and, at each of the law's coins, a strategy of the live
session of `J` tables; they read the same at every history. -/
abbrev baseGame {nv : ℕ} (Lv : Law nv) (Reg' : Type) (dg : ByteArray → ByteArray) (J mPts N : ℕ) :
    Game (LiveOutJ J mPts) :=
  .send Reg' fun _ => .coin Lv.Ω fun _ => liveSession dg J mPts N

/-- A strategy of the live session of `J` tables, as one of `J'` tables when `J' = J`. -/
def castLive {J J' : ℕ} (h : J' = J) (dg : ByteArray → ByteArray) (mPts N : ℕ)
    (s : Strategy (liveSession dg J mPts N)) : Strategy (liveSession dg J' mPts N) :=
  cast (congrArg (fun J => Strategy (liveSession dg J mPts N)) h.symm) s

theorem castLive_castLive {J J' : ℕ} (h : J' = J) (dg : ByteArray → ByteArray) (mPts N : ℕ)
    (s : Strategy (liveSession dg J' mPts N)) : castLive h dg mPts N (castLive h.symm dg mPts N s) = s := by
  subst h
  rfl

/-- **Implications at every outcome of `liveOfZJ`'s live audit bound probabilities**: at the registration the prover
sends, each of the law's draws and each output of the live session of `J` tables written by `wr`. -/
theorem prob_auditLive_liveOfZJ_mono {n : ℕ} {L : Law n} {Reg : Type} {mPts : ℕ} (dg : ByteArray → ByteArray) (N : ℕ)
    {planZ : Finset (Fin n) → Reg → List (TabZK FlockLevel3.GF128 mPts)}
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr)))
    {p q : Reg × Finset (Fin n) × Exec.Transcript → Prop}
    (h : ∀ ω o', p (τ.1, L.draw ω, wr (L.draw ω) τ.1 o') → q (τ.1, L.draw ω, wr (L.draw ω) τ.1 o')) :
    prob p (auditLive L Reg (liveOfZJ dg planZ N wr)) τ ≤ prob q (auditLive L Reg (liveOfZJ dg planZ N wr)) τ := by
  show avg (fun ω => prob p ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)) ≤
    avg fun ω => prob q ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)
  refine avg_mono fun ω => ?_
  rw [prob_map, prob_map]
  unfold liveOfZJ
  rw [prob_map, prob_map]
  exact prob_mono (fun o' => h ω o') _ _

namespace ZkOuterD

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuterD mPts nv J Reg' c hU hscope Lv)

/-- The circuit file parses under `HmRow` too (`Hidden.parse_untyped`). -/
theorem hc : HmRow.parse (zkTags zo.I) zo.I.circuitFile zo.I.tables = .ok c :=
  (Hidden.parse_untyped (zkTyped zo.ht) zo.hcH).1

/-- Every table count of the plan is `J`. -/
theorem hlen (S : Finset (Fin nv)) (R : Reg') : (planZHJ (Reg := Reg') zo.dj zo.y₀ S R).length = J :=
  (planZHJ_length zo.dj zo.y₀ S R).trans zo.hJ

end ZkOuterD

namespace ZkOuter

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

/-- V*'s live session at the history. -/
noncomputable abbrev game : Game (Reg' × Finset (Fin nv) × Exec.Transcript) :=
  auditLive Lv Reg' (liveOfZJ dg (planZHJ zo.dj zo.y₀) N zo.wr)

/-- The registered-reads layout of V*'s session (`layShHJ`). -/
noncomputable abbrev lay :=
  layShHJ (Reg := Reg') zo.dj zo.y₀ zo.ht zo.hcH (Layout.scopeOkH_of_scopeOk hscope) hU (pub := zo.pub)

/-- The value binding of V*'s session (`vbHJ`). -/
noncomputable abbrev vb := vbHJ (Reg := Reg') zo.dj zo.y₀ zo.ht zo.hcH (Layout.scopeOkH_of_scopeOk hscope) hU zo.hpub

/-- The strategy a name stands for in V*'s session. -/
noncomputable def ofBase (τ₀ : Strategy (baseGame Lv Reg' dg J mPts N)) : Strategy (zo.game dg N) :=
  ⟨τ₀.1, fun ω => Strategy.map (fun t => (τ₀.1, Lv.draw ω, t)) _ (Strategy.map (zo.wr (Lv.draw ω) τ₀.1) _
    (castLive (zo.hlen (Lv.draw ω) τ₀.1) dg mPts N (τ₀.2 ω)))⟩

/-- **Every strategy of V*'s session has a name.** -/
theorem ofBase_surj (τ : Strategy (zo.game dg N)) : ∃ τ₀, zo.ofBase dg N τ₀ = τ := by
  obtain ⟨R, s⟩ := τ
  refine ⟨⟨R, fun ω => castLive (zo.hlen (Lv.draw ω) R).symm dg mPts N
    (Strategy.ofMap (zo.wr (Lv.draw ω) R) _
      (Strategy.ofMap (fun t => (R, Lv.draw ω, t)) (liveOfZJ dg (planZHJ zo.dj zo.y₀) N zo.wr (Lv.draw ω) R) (s ω)))⟩,
    congrArg (Sigma.mk R) (funext fun ω => ?_)⟩
  have h1 := map_ofMap (zo.wr (Lv.draw ω) R) (liveSession dg (planZHJ (Reg := Reg') zo.dj zo.y₀ (Lv.draw ω) R).length mPts N)
    (Strategy.ofMap (fun t => (R, Lv.draw ω, t)) (liveOfZJ dg (planZHJ zo.dj zo.y₀) N zo.wr (Lv.draw ω) R) (s ω))
  have h2 := map_ofMap (fun t => (R, Lv.draw ω, t)) (liveOfZJ dg (planZHJ zo.dj zo.y₀) N zo.wr (Lv.draw ω) R) (s ω)
  show Strategy.map (fun t => (R, Lv.draw ω, t)) (liveOfZJ dg (planZHJ zo.dj zo.y₀) N zo.wr (Lv.draw ω) R)
    (Strategy.map (zo.wr (Lv.draw ω) R) (liveSession dg (planZHJ (Reg := Reg') zo.dj zo.y₀ (Lv.draw ω) R).length mPts N)
      (castLive (zo.hlen (Lv.draw ω) R) dg mPts N (castLive (zo.hlen (Lv.draw ω) R).symm dg mPts N
        (Strategy.ofMap (zo.wr (Lv.draw ω) R) _ (Strategy.ofMap (fun t => (R, Lv.draw ω, t))
          (liveOfZJ dg (planZHJ zo.dj zo.y₀) N zo.wr (Lv.draw ω) R) (s ω)))))) = s ω
  rw [castLive_castLive, h1, h2]

/-- The prover's strategy in V*'s session, simulated in the compiled batched session (`simZHJ`). -/
noncomputable def sim (τ : Strategy (zo.game dg N)) :=
  simZHJ zo.dj zo.y₀ zo.ht zo.hcH (Layout.scopeOkH_of_scopeOk hscope) hU zo.wr dg N zo.hTags zo.hRec zo.hDraw τ

/-- The simulation registers what the prover sends. -/
theorem reg_sim (τ : Strategy (zo.game dg N)) : reg (zo.sim dg N τ) = τ.1 :=
  rfl

/-- **V*'s value layer at the strategy `τ`**: the decoded witness (`XplurZC`), each wire's bit in its commit string's
plurality value. -/
noncomputable def layer (k Rw : ℕ) (τ : Strategy (zo.game dg N)) : Fin (Layout.keyN c hU nv) → Bool :=
  XplurZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀) (tabZHJ zo.dj zo.y₀)
    zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw

/-- **`cr/sha-512` for the knowledge finders** of the simulation at the strategy `τ` (`TableCRZ`), at budgets
`qF, qS`. -/
def TabCR (qF qS : ℝ) (τ : Strategy (zo.game dg N)) : Prop :=
  ∀ ω j, TableCRZ execArith hs512 (planZHJ zo.dj zo.y₀ (Lv.draw ω) (reg (zo.sim dg N τ))) j
    (zcStrat execArith hs512 (planZHJ zo.dj zo.y₀) (cont (zo.sim dg N τ) ω)) qF qS

/-- **`cr/sha-512` at the strategy `τ`**: for the knowledge finders of its simulation (`TabCR`) at budgets `qF, qS`,
for its registered-value finders at `rd` (`RegCRZC`) at budget `qW`, and for the finder that plays the session once and
outputs the extractor's pair of an off-leaf read at a drawn row (`ReadsX.pick`) at budget `qR`. -/
def CR (k Rw : ℕ) (qF qS qW qR : ℝ) (τ : Strategy (zo.game dg N)) : Prop :=
  zo.TabCR dg N qF qS τ ∧
    RegCRZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀) (tabZHJ zo.dj zo.y₀)
      zo.vb (regOpenZ zo.ht zo.dj zo.y₀ zo.hcH hU zo.hpub (Layout.scopeOkH_of_scopeOk hscope) zo.rd
        (reg (zo.sim dg N τ))) (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw
      qW ∧
    (StrictCR.oneRun (zo.game dg N) τ fun o =>
      (zo.rd.restrict (rowIn o.2.1)).pick o.1 (regPathsHJ zo.I c zo.pub o.2.2 zo.rd.port)).CR H512 qR

/-- **The outer bound at the strategy `τ`**: the model's miss at one bad unit (`Lm` the model's law), the drawn units'
`ksAvgStrictZ + linkBoundZC + regTermZC`, and the registered paths' `qR²/2^513`. -/
noncomputable def bound (Lm : Law nv) (k Rw M : ℕ) (ρ t' qF qS qW qR : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  Lm.miss 1 +
    (ksAvgStrictZ (planZHJ zo.dj zo.y₀) Lv k qF qS (reg (zo.sim dg N τ)) +
      linkBoundZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
        (tabZHJ zo.dj zo.y₀) zo.vb k Rw M ρ t' (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) +
      regTermZC execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ (Reg := Reg') zo.dj zo.y₀)
        (tabZHJ zo.dj zo.y₀) zo.vb Lv k Rw ρ qW) +
    ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ))

open Classical in
/-- **The outer bound where the strategy's finders satisfy `cr/sha-512`**, else `1`. -/
noncomputable def boundIf (Lm : Law nv) (k Rw M : ℕ) (ρ t' qF qS qW qR : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  if zo.CR dg N k Rw qF qS qW qR τ then zo.bound dg N Lm k Rw M ρ t' qF qS qW qR τ else 1

/-- **A fork of the session at the strategy `τ`**: a run of one of `CR`'s finders, or of the link finder, from the
prover's own continuations in its simulation, outputs a SHA-512 collision: the knowledge finders of a drawn table
(`TableForkZ`), the registered-value finders (`RegForkZC`), the run that opens a drawn row's registered leaf off the
registrant's (`ReadsX.pick`), or the link finder (`LinkForkZC`). -/
def Fork (k Rw M : ℕ) (τ : Strategy (zo.game dg N)) : Prop :=
  (∃ ω j, TableForkZ execArith hs512 (planZHJ zo.dj zo.y₀ (Lv.draw ω) (reg (zo.sim dg N τ))) j
      (zcStrat execArith hs512 (planZHJ zo.dj zo.y₀) (cont (zo.sim dg N τ) ω))) ∨
    RegForkZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb (regOpenZ zo.ht zo.dj zo.y₀ zo.hcH hU zo.hpub
        (Layout.scopeOkH_of_scopeOk hscope) zo.rd (reg (zo.sim dg N τ))) (reg (zo.sim dg N τ))
      (cont (zo.sim dg N τ)) k Rw ∨
    (StrictCR.oneRun (zo.game dg N) τ fun o =>
      (zo.rd.restrict (rowIn o.2.1)).pick o.1 (regPathsHJ zo.I c zo.pub o.2.2 zo.rd.port)).Finds H512 ∨
    LinkForkZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw M

/-- **With no fork, the session's finders satisfy `cr/sha-512` at budget `0`** and its link finder `ecr/sha-512` at
cost `0`. -/
theorem cr_of_not_fork {k Rw M : ℕ} {τ : Strategy (zo.game dg N)} (h : ¬ zo.Fork dg N k Rw M τ) :
    zo.CR dg N k Rw 0 0 0 0 τ ∧
      LinkCRZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
        (tabZHJ zo.dj zo.y₀) zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw M 0 :=
  ⟨⟨fun ω j => tableCRZ_of_not_fork _ _ fun hf => h (Or.inl ⟨ω, j, hf⟩),
      regCRZC_of_not_fork _ _ _ _ _ _ _ _ _ fun hf => h (Or.inr (Or.inl hf)),
      Finder.cr_of_not_finds (fun hf => h (Or.inr (Or.inr (Or.inl hf)))) 0⟩,
    linkCRZC_of_not_fork _ _ _ _ _ _ _ _ fun hf => h (Or.inr (Or.inr (Or.inr hf)))⟩

/-- **The outer bound in fork form**: the model's miss at one bad unit and the drawn units' statistical errors, the
tables' (`tableErrZ`), the link's (`linkErrZC`) and the waits' (`waitErrZC`); `bound` with no finder budgets and no
SHA-512 evaluations to count. -/
noncomputable def boundCR (Lm : Law nv) (k Rw M : ℕ) (ρ : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  Lm.miss 1 +
    (tableErrZ (planZHJ zo.dj zo.y₀) Lv k (reg (zo.sim dg N τ)) +
      linkErrZC execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ (Reg := Reg') zo.dj zo.y₀)
        (tabZHJ zo.dj zo.y₀) zo.vb Lv k Rw M ρ +
      waitErrZC execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ (Reg := Reg') zo.dj zo.y₀)
        (tabZHJ zo.dj zo.y₀) zo.vb Lv k Rw ρ)

theorem bound_zero (Lm : Law nv) {k Rw M : ℕ} (ρ : ℝ) {τ : Strategy (zo.game dg N)}
    (h : LinkCRZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw M 0) :
    zo.bound dg N Lm k Rw M ρ 0 0 0 0 0 τ = zo.boundCR dg N Lm k Rw M ρ τ := by
  unfold bound boundCR
  rw [ksAvgStrictZ_zero, linkBoundZC_zero _ _ _ _ _ _ _ _ h, regTermZC_zero,
    show (0 : ℝ) ^ 2 / 2 ^ (513 : ℝ) = 0 by simp, ENNReal.ofReal_zero, add_zero]

/-- **With no fork, `boundIf` at no budgets is at most the fork-form bound.** -/
theorem boundIf_le (Lm : Law nv) {k Rw M : ℕ} (ρ : ℝ) {τ : Strategy (zo.game dg N)} (h : ¬ zo.Fork dg N k Rw M τ) :
    zo.boundIf dg N Lm k Rw M ρ 0 0 0 0 0 τ ≤ zo.boundCR dg N Lm k Rw M ρ τ := by
  unfold boundIf
  split_ifs with hc
  · exact (zo.bound_zero dg N Lm ρ (zo.cr_of_not_fork dg N h).2).le
  · exact absurd (zo.cr_of_not_fork dg N h).1 hc

/-- **A layer agrees with the registrant**: every wire a unit reads at a registered read carries the registrant's bit. -/
def RegAgree (bits : Fin (Layout.keyN c hU nv) → Bool) : Prop :=
  ∀ u, ∀ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u, zo.rd.isRead (zo.lay.pos g) →
    bits g = zo.lay.bit (zo.rd.wv (zo.lay.pos g)) g

/-- **Every unit reads the constant gate.** -/
theorem oneG_mem_io (z : ZkOuter mPts nv J Reg' c hU hscope Lv) (u : Fin nv) :
    Layout.oneG (c := c) (hU := hU) (n := nv) ∈
      ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u := by
  have hc : HmRow.parse (zkTags z.I) z.I.circuitFile z.I.tables = .ok c :=
    (Hidden.parse_untyped (zkTyped z.ht) z.hcH).1
  have P := parse_facts hc
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have ho : c.outNet = c.unitNet := by
    unfold Layout.scopeOk at hscope
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hscope
    exact hscope.1.1.1.2
  have hp := C.unit_pow2 ho
  rw [C.unit_range.2] at hp
  unfold Flock.isPow2 at hp
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
  have hupv : 0 < c.upv := Nat.pos_of_ne_zero fun h0 => by simp [h0] at hp
  have hone : Const.Prog.oneAt (keyProg c hU nv) u = oneGK c hU nv :=
    Fin.ext (Layout.oneAt_uniform_val (k := Placed.inGates nv (Layout.keyB c hU))
      (R := Placed.circuitRows c hU)
      (ins := Placed.insOf nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
      (one := Placed.oneGate nv (Layout.keyB c hU))
      (hone := Placed.insOf_ne_one nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) nv u)
  have hnone : ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).unit (oneGK c hU nv) = none := by
    show (keyProg c hU nv).unit (oneGK c hU nv) = none
    exact Layout.unit_uniform_input (k := Placed.inGates nv (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
      (one := Placed.oneGate nv (Layout.keyB c hU))
      (hone := Placed.insOf_ne_one nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) nv _
      (by simp [Layout.oneG, Placed.oneGate])
  have hm := (FlockVBridge.keyInst hU nv (Layout.outsOf c hU nv) u).mem_io_of_read
    (i := ⟨0 * ((Rows.ofNet hU).nComp + 1) + 0, Rows.mul_add_lt hupv (by omega)⟩)
    (c := ((Rows.ofNet hU).stack c.upv).one)
    (by rw [FlockVBridge.stack_a_one _ _ hupv]; simp)
    (by rw [FlockVBridge.keyInst_one, hone, hnone]; simp)
  rwa [FlockVBridge.keyInst_one, hone] at hm

/-- **A good layer**: no wrong unit, agreement with the registrant, and 1 at every unit's constant gate. -/
def Good (bits : Fin (Layout.keyN c hU nv) → Bool) : Prop :=
  ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong bits = ∅ ∧ zo.RegAgree bits ∧
    ∀ _u : Fin nv, bits (Layout.oneG (c := c) (hU := hU) (n := nv)) = true

open Classical in
/-- The units with a row at a registered read whose leaf under `R` is not the registrant's. -/
noncomputable def offUnits (R : Reg') : Finset (Fin nv) :=
  Finset.univ.filter fun u => ∃ p, zo.rd.isRead p ∧ rowIn {u} p ∧ zo.lay.regLeaf R p ≠ zo.rd.leaf p

/-- **The bad units at the strategy `τ`**: wrong in the layer, reading a registered-value disagreement at the
registrant's leaf, or holding an off-leaf registered read. -/
noncomputable def badUnits (k Rw : ℕ) (τ : Strategy (zo.game dg N)) : Finset (Fin nv) :=
  ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong (zo.layer dg N k Rw τ) ∪
    regWrongZ execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀) (tabZHJ zo.dj zo.y₀)
      zo.vb (regOpenZ zo.ht zo.dj zo.y₀ zo.hcH hU zo.hpub (Layout.scopeOkH_of_scopeOk hscope) zo.rd
        (reg (zo.sim dg N τ))) (reg (zo.sim dg N τ)) (zo.layer dg N k Rw τ) ∪
    zo.offUnits τ.1

/-- **A layer that is not good has a bad unit.** -/
theorem badUnits_nonempty (k Rw : ℕ) (τ : Strategy (zo.game dg N)) (hG : ¬ zo.Good (zo.layer dg N k Rw τ)) :
    (zo.badUnits dg N k Rw τ).Nonempty := by
  classical
  by_cases hw : ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong (zo.layer dg N k Rw τ) = ∅
  · by_cases hA : zo.RegAgree (zo.layer dg N k Rw τ)
    · obtain ⟨u, hone⟩ : ∃ u : Fin nv,
          zo.layer dg N k Rw τ (Layout.oneG (c := c) (hU := hU) (n := nv)) ≠ true := by
        by_contra h
        refine hG ⟨hw, hA, fun u => ?_⟩
        by_contra hne
        exact h ⟨u, hne⟩
      refine ⟨u, Finset.mem_union_left _ (Finset.mem_union_right _
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, Layout.oneG (c := c) (hU := hU) (n := nv),
          zo.oneG_mem_io u, ?_⟩))⟩
      exact regBadZ_oneG zo.ht zo.dj zo.y₀ zo.hcH hU zo.hpub (Layout.scopeOkH_of_scopeOk hscope) zo.rd
        (reg (zo.sim dg N τ)) _ hone
    · obtain ⟨u, g, hg, hread, hne⟩ : ∃ u, ∃ g ∈
          ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u,
          zo.rd.isRead (zo.lay.pos g) ∧ zo.layer dg N k Rw τ g ≠ zo.lay.bit (zo.rd.wv (zo.lay.pos g)) g := by
        by_contra h
        exact hA fun u g hg hread => by
          by_contra hne
          exact h ⟨u, g, hg, hread, hne⟩
      by_cases hleaf : zo.lay.regLeaf τ.1 (zo.lay.pos g) = zo.rd.leaf (zo.lay.pos g)
      · refine ⟨u, Finset.mem_union_left _ (Finset.mem_union_right _
          (Finset.mem_filter.2 ⟨Finset.mem_univ _, g, hg, ?_⟩))⟩
        exact regBadZ_of_leaf zo.ht zo.dj zo.y₀ zo.hcH hU zo.hpub (Layout.scopeOkH_of_scopeOk hscope) zo.rd _ _ g hread
          (by rw [zo.reg_sim dg N τ]; exact hleaf) hne
      · obtain ⟨⟨u', q⟩, hp⟩ := Sum.isLeft_iff.1 (zo.rd.read_row _ hread)
        refine ⟨u', Finset.mem_union_right _
          (Finset.mem_filter.2 ⟨Finset.mem_univ _, zo.lay.pos g, hread, ?_, hleaf⟩)⟩
        rw [hp]
        exact Finset.mem_singleton_self u'
  · exact (Finset.nonempty_iff_ne_empty.2 hw).mono (Finset.subset_union_left.trans Finset.subset_union_left)

/-- **An off-leaf registered read at a drawn row**, at an accepted outcome, is at most `qR²/2^513` likely, under
`cr/sha-512` for the finder that plays V*'s session once (`ReadsX.prob_offOn_le`): the verifier's registered check
holds at the drawn rows (`checked_of_acceptsHJ`, on `hown`). -/
theorem off_le {qR : ℝ} (τ : Strategy (zo.game dg N))
    (hR : (StrictCR.oneRun (zo.game dg N) τ fun o =>
      (zo.rd.restrict (rowIn o.2.1)).pick o.1 (regPathsHJ zo.I c zo.pub o.2.2 zo.rd.port)).CR H512 qR) :
    prob (fun o => LiveAcceptsZKR zo.I zo.own o ∧ ∃ u ∈ o.2.1, u ∈ zo.offUnits τ.1) (zo.game dg N) τ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  classical
  have hmono : ∀ ω o', (LiveAcceptsZKR zo.I zo.own (τ.1, Lv.draw ω, zo.wr (Lv.draw ω) τ.1 o') ∧
      ∃ u ∈ Lv.draw ω, u ∈ zo.offUnits τ.1) →
      (LiveAcceptsZKR zo.I zo.own (τ.1, Lv.draw ω, zo.wr (Lv.draw ω) τ.1 o') ∧
        ∃ ω' o'', Lv.draw ω = Lv.draw ω' ∧ zo.wr (Lv.draw ω) τ.1 o' = zo.wr (Lv.draw ω') τ.1 o'') ∧
      (zo.rd.restrict (rowIn (Lv.draw ω))).Off τ.1 := by
    rintro ω o' ⟨hacc, u, hu, hoff⟩
    obtain ⟨p, hp, hrow, hne⟩ := (Finset.mem_filter.1 hoff).2
    exact ⟨⟨hacc, ω, o', rfl, rfl⟩, p, ⟨hp, rowIn_mono (Finset.singleton_subset_iff.2 hu) p hrow⟩, hne⟩
  refine (prob_auditLive_liveOfZJ_mono dg N zo.wr τ
    (q := fun o => (LiveAcceptsZKR zo.I zo.own o ∧ ∃ ω o', o.2.1 = Lv.draw ω ∧ o.2.2 = zo.wr (Lv.draw ω) o.1 o') ∧
      (zo.rd.restrict (rowIn o.2.1)).Off o.1) hmono).trans
    (FlockSoundness.Registered.ReadsX.prob_offOn_le (zo.game dg N) τ (fun o => zo.rd.restrict (rowIn o.2.1))
      (fun o => o.1) (fun o => LiveAcceptsZKR zo.I zo.own o ∧ ∃ ω o', o.2.1 = Lv.draw ω ∧ o.2.2 = zo.wr (Lv.draw ω) o.1 o')
      (fun o => regPathsHJ zo.I c zo.pub o.2.2 zo.rd.port) ?_ hR)
  rintro ⟨R', S', tr⟩ ⟨hacc, ω, o', hS', htr⟩
  obtain rfl : S' = Lv.draw ω := hS'
  have htr' : tr = zo.wr (Lv.draw ω) R' o' := htr
  rw [htr'] at hacc ⊢
  exact checked_of_acceptsHJ zo.I zo.ht zo.own zo.dj zo.y₀ zo.hcH hU zo.hpub (Layout.scopeOkH_of_scopeOk hscope) zo.hTags
    _ hacc _ (zo.hown ω R' o' hacc)

/-- **V*'s session at one history, at one strategy**: an accepted outcome at which V*'s value layer is not `Good` is at
most `bound` likely. The bad units are fixed by the strategy: the draw misses them all (`Law.execOS_subset_escape_le`
under A3, then `miss 1`), or meets a wrong unit or a registered-value disagreement at the registrant's leaf
(`zk_session_regDrawnR`), or an off-leaf registered read (`off_le`). -/
theorem accepts_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (zo : ZkOuter mPts nv J Reg' c hU hscope (Law.execOS (n := nv) src (.subset nv) nv))
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg') zo.dj zo.y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    {qF qS qW qR : ℝ} (τ : Strategy (zo.game dg N)) (hCR : zo.CR dg N k Rw qF qS qW qR τ) :
    prob (fun o => LiveAcceptsZKR zo.I zo.own o ∧ ¬ zo.Good (zo.layer dg N k Rw τ)) (zo.game dg N) τ ≤
      zo.bound dg N (Law.subset nv nv le_rfl) k Rw M ρ t' qF qS qW qR τ := by
  classical
  by_cases hG : zo.Good (zo.layer dg N k Rw τ)
  · exact (prob_mono (fun o h => h.2 hG) _ _).trans ((prob_false _ _).le.trans zero_le)
  have h1 : prob (fun o => Disjoint o.2.1 (zo.badUnits dg N k Rw τ)) (zo.game dg N) τ ≤
      (Law.subset nv nv le_rfl).miss 1 :=
    (prob_auditLive_draw _ τ fun S => Disjoint S (zo.badUnits dg N k Rw τ)).trans_le
      ((Law.execOS_subset_escape_le nv nv le_rfl Lb src hA3 _).trans
        (Law.escape_le_miss _ (Finset.card_pos.2 (zo.badUnits_nonempty dg N k Rw τ hG))))
  have h2 := zk_session_regDrawnR zo.I zo.ht zo.own zo.dj zo.y₀ zo.wr dg N zo.hcH hU zo.hpub
    (Layout.scopeOkH_of_scopeOk hscope) _ zo.hDraw zo.hTags zo.hRec hk hRw hM hρ hr ht' zo.rd τ hCR.1 hCR.2.1
  have h3 := zo.off_le dg N τ hCR.2.2
  refine (prob_mono (fun o h => ?_) _ _).trans
    ((prob_or_le _ _ _ _).trans (add_le_add ((prob_or_le _ _ _ _).trans (add_le_add h1 h2)) h3))
  obtain ⟨hacc, -⟩ := h
  by_cases hd : Disjoint o.2.1 (zo.badUnits dg N k Rw τ)
  · exact Or.inl (Or.inl hd)
  · obtain ⟨u, hu, hub⟩ := Finset.not_disjoint_iff.1 hd
    rcases Finset.mem_union.1 hub with hub | hub
    · exact Or.inl (Or.inr ⟨hacc, Finset.not_disjoint_iff.2 ⟨u, hub, hu⟩⟩)
    · exact Or.inr ⟨hacc, u, hu, hub⟩

end ZkOuter

/-- **The law of a public verifier's units**: the full draw, the live subset law of all `nv` units over `[0, nv)` on a
byte source (`live`, `--draw subset:<units>`, as V* is staged), and the model's uniform `nv`-subset it escapes no more
often than (`model`, which misses no unit). -/
structure VLaw (nv : ℕ) : Type 1 where
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256

attribute [instance] VLaw.fintypeΩ VLaw.nonemptyΩ

namespace VLaw

variable {nv : ℕ} (law : VLaw nv)

/-- The live full draw on the source's bytes. -/
abbrev live : Law nv := Law.execOS (n := nv) law.src (.subset nv) nv

/-- The model's full draw. It reads nothing of the law but its unit count; the argument is there so that
`(Vs j).law.model` names it. -/
abbrev model (_law : VLaw nv) : Law nv := Law.subset nv nv le_rfl

end VLaw

/-- **A public verifier**: its circuit `c` over `nv` units, proved in sessions of `J` tables, and its law. -/
structure VStmt : Type 1 where
  nv : ℕ
  mPts : ℕ
  J : ℕ
  c : Flock.Circuit
  hU : Built c.unit
  hscope : Layout.scopeOk c = true
  law : VLaw nv

namespace VStmt

/-- A session of the verifier's circuit at one history. -/
abbrev Outer (Q : VStmt) (Reg' : Type) := ZkOuter Q.mPts Q.nv Q.J Reg' Q.c Q.hU Q.hscope Q.law.live

/-- A session of the verifier's circuit at one history, but for its record's custody. -/
abbrev OuterD (Q : VStmt) (Reg' : Type) := ZkOuterD Q.mPts Q.nv Q.J Reg' Q.c Q.hU Q.hscope Q.law.live

/-- A value layer of the verifier's circuit. -/
abbrev Bits (Q : VStmt) : Type := Fin (Layout.keyN Q.c Q.hU Q.nv) → Bool

/-- A name for a strategy of a session of the verifier's circuit. -/
abbrev Name (Q : VStmt) (Reg' : Type) (dg : ByteArray → ByteArray) (N : ℕ) : Type :=
  Strategy (baseGame Q.law.live Reg' dg Q.J Q.mPts N)

end VStmt

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- The record of V*'s sessions at one history: each session's name and outcome. -/
abbrev VRec (Vs : Fin S → VStmt) (Reg' : Type) (dg : ByteArray → ByteArray) (N : ℕ) : Type :=
  (j : Fin S) → (Vs j).Name Reg' dg N × (Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript)

/-- **V*'s sessions at one history, in a row**: each played by the strategy the prover names at its start (`seqC`). -/
noncomputable abbrev sessions (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ) :
    Game (VRec Vs Reg' dg N) :=
  seqC S (T := fun j => (Vs j).Name Reg' dg N) (α := fun j => Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript)
    (fun j => (zs j).game dg N) fun j => (zs j).ofBase dg N

/-- **Every session accepts.** -/
def Accepts (zs : (j : Fin S) → (Vs j).Outer Reg') {dg : ByteArray → ByteArray} {N : ℕ} (b : VRec Vs Reg' dg N) :
    Prop :=
  ∀ j, LiveAcceptsZKR (zs j).I (zs j).own (b j).2

/-- **V*'s layers** at the strategies the record names. -/
noncomputable def layers (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (b : VRec Vs Reg' dg N) : (j : Fin S) → (Vs j).Bits :=
  fun j => (zs j).layer dg N k Rw ((zs j).ofBase dg N (b j).1)

/-- **The expected sum over V*'s sessions** of `F j` at the strategy the prover names for session `j`. -/
noncomputable def sumS (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    (F : (j : Fin S) → Strategy ((zs j).game dg N) → ℝ≥0∞) (s : Strategy (sessions zs dg N)) : ℝ≥0∞ :=
  exG (fun rs => ∑ j, F j ((zs j).ofBase dg N (rs j).1)) (sessions zs dg N) s

/-- **V*'s sessions at one history**: when every outcome at which every session accepts and every layer is `Good`
satisfies `Q` of the layers, the sessions accept with `Q` failing at most the expected sum of the sessions' `boundIf`
(`accepts_le`, `prob_seqC_exists_le`). -/
theorem sessions_le
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ j S' R' i, rateZ (planZHJ (Reg := Reg') (zs j).dj (zs j).y₀ S' R') i k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (qF qS qW qR : ℝ) (Q : ((j : Fin S) → (Vs j).Bits) → Prop)
    (hQ : ∀ (τ : (j : Fin S) → Strategy ((zs j).game dg N))
      (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
      (∀ j, LiveAcceptsZKR (zs j).I (zs j).own (o j) ∧ (zs j).Good ((zs j).layer dg N k Rw (τ j))) →
      Q fun j => (zs j).layer dg N k Rw (τ j))
    (s : Strategy (sessions zs dg N)) :
    prob (fun b => Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b)) (sessions zs dg N) s ≤
      sumS zs dg N (fun j τ => (zs j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s := by
  classical
  have hmono : ∀ b : VRec Vs Reg' dg N, Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b) →
      ∃ j, LiveAcceptsZKR (zs j).I (zs j).own (b j).2 ∧
        ¬ (zs j).Good ((zs j).layer dg N k Rw ((zs j).ofBase dg N (b j).1)) := by
    rintro b ⟨hacc, hnq⟩
    by_contra hno
    refine hnq (hQ (fun j => (zs j).ofBase dg N (b j).1) (fun j => (b j).2) fun j => ⟨hacc j, ?_⟩)
    by_contra hg
    exact hno ⟨j, hacc j, hg⟩
  have hB : ∀ j τ₀, prob (fun a => LiveAcceptsZKR (zs j).I (zs j).own a ∧
        ¬ (zs j).Good ((zs j).layer dg N k Rw ((zs j).ofBase dg N τ₀))) ((zs j).game dg N) ((zs j).ofBase dg N τ₀) ≤
      (zs j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR ((zs j).ofBase dg N τ₀) := by
    intro j τ₀
    unfold ZkOuter.boundIf
    split_ifs with hc
    · exact ZkOuter.accepts_le (Vs j).law.Lb (Vs j).law.src (hA3 j) (zs j) dg N hk hRw hM hρ (hr j) ht' _ hc
    · exact prob_le_one _ _ _
  exact (prob_mono hmono _ _).trans
    (prob_seqC_exists_le S _ _ (fun j r => LiveAcceptsZKR (zs j).I (zs j).own r.2 ∧
        ¬ (zs j).Good ((zs j).layer dg N k Rw ((zs j).ofBase dg N r.1)))
      (fun j τ₀ => (zs j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR ((zs j).ofBase dg N τ₀)) hB s)

variable {Reg₂ : Type}

/-- **V*'s outer proof at one history**: the prover registers `R₂`, the values V*'s sessions read that the history does
not fix (V*'s running sums `rec-acc`, which every level's session and the algebra read, and with `link_rows` each
level's opened rows `rec-rows-L<level>`), and then V*'s sessions at `R₂` run in a row. The output records `R₂`. -/
noncomputable abbrev outerV (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ) :
    Game (Reg₂ × VRec Vs Reg' dg N) :=
  .send Reg₂ fun R₂ => (sessions (zs R₂) dg N).map fun b => (R₂, b)

/-- The prover's strategy for V*'s sessions after its registration `s.1`. -/
noncomputable def sessOf (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    (s : Strategy (outerV zs dg N)) : Strategy (sessions (zs s.1) dg N) :=
  Strategy.ofMap (fun b => (s.1, b)) (sessions (zs s.1) dg N) s.2

/-- **Every session accepts**, at the registration the outcome records. -/
def AcceptsV (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') {dg : ByteArray → ByteArray} {N : ℕ}
    (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  Accepts (zs o.1) o.2

/-- **The expected sum over V*'s sessions** of `F R₂ j`, at the registration `R₂` the prover sends and the strategy it
names for session `j`. -/
noncomputable def sumV (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    (F : (R₂ : Reg₂) → (j : Fin S) → Strategy ((zs R₂ j).game dg N) → ℝ≥0∞) (s : Strategy (outerV zs dg N)) :
    ℝ≥0∞ :=
  sumS (zs s.1) dg N (F s.1) (sessOf zs dg N s)

/-- An event of V*'s outer proof is an event of its sessions at the registration the prover sends. -/
theorem prob_outerV (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    (p : Reg₂ × VRec Vs Reg' dg N → Prop) (s : Strategy (outerV zs dg N)) :
    prob p (outerV zs dg N) s = prob (fun b => p (s.1, b)) (sessions (zs s.1) dg N) (sessOf zs dg N s) :=
  prob_map p (fun b => (s.1, b)) (sessions (zs s.1) dg N) s.2

/-- **V*'s outer proof at one history**: `sessions_le` at the registration the prover sends. -/
theorem outerV_le
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R₂ j S' R' i, rateZ (planZHJ (Reg := Reg') (zs R₂ j).dj (zs R₂ j).y₀ S' R') i k ≤ ρ) {t' : ℝ}
    (ht' : 0 ≤ t') (qF qS qW qR : ℝ) (Q : ((j : Fin S) → (Vs j).Bits) → Prop)
    (hQ : ∀ R₂ (τ : (j : Fin S) → Strategy ((zs R₂ j).game dg N))
      (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
      (∀ j, LiveAcceptsZKR (zs R₂ j).I (zs R₂ j).own (o j) ∧ (zs R₂ j).Good ((zs R₂ j).layer dg N k Rw (τ j))) →
      Q fun j => (zs R₂ j).layer dg N k Rw (τ j))
    (s : Strategy (outerV zs dg N)) :
    prob (fun o => AcceptsV zs o ∧ ¬ Q (layers (zs o.1) dg N k Rw o.2)) (outerV zs dg N) s ≤
      sumV zs dg N (fun R₂ j τ => (zs R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s := by
  rw [prob_outerV]
  exact sessions_le hA3 (zs s.1) dg N hk hRw hM hρ (hr s.1) ht' qF qS qW qR Q (hQ s.1) (sessOf zs dg N s)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **What V*'s circuits compute** (`k, Rw` the count curve's): at every history `(R, ω, t)`, registration `R₂` of the
outer proof and strategies `τ` of V*'s sessions there, when every session's outcome flock-verify `--zk --registered`
accepts and every session's value layer is `Good`, the messages `wmsg` reads from the layers hold an inner transcript
the inner verifier accepts on the hidden circuit's statement `x R ω` at the rounds' coins. -/
def VBridge (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) : Prop :=
  ∀ R ω t R₂ (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N))
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
    (∀ j, LiveAcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own (o j) ∧
      (zs R ω t R₂ j).Good ((zs R ω t R₂ j).layer dg N k Rw (τ j))) →
    I.accepts (x R ω) ((wmsg fun j => (zs R ω t R₂ j).layer dg N k Rw (τ j)).zip (t.map Prod.snd))

/-- **The recursive audit with V*'s `--zk --registered` sessions as the outer proof is sound, up to what the inner
verifier reads**: `recursive_sound_upto` at V*'s outer proof (`outerV_le` at every leaf, on `VBridge`). Stage `i` opens
to `msg i v`, and its transfer bounds the messages V*'s layers hold for round `i` not being `same i` as that. -/
theorem zk_sessions_recursive_upto
    -- V*'s laws: A3
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i, rateZ (planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    -- the stages: commitments' openings, transfer, and cr/sha-512 for the fork finders
    {V Pos : Type} [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] (ob : Openings H512 V)
    (cmt : Cm → Pos → ob.Dg) (msg : ℕ → (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (outerV (zs R ω t) dg N) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (outerV (zs R ω t) dg N) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    (htr : ∀ i < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[i]? = some e →
      prob (fun o => AcceptsV (zs σ.1 ω t) o ∧ ¬ Option.Rel (same i (t.map Prod.snd))
          (wmsg (layers (zs σ.1 ω t o.1) dg N k Rw o.2))[i]? (some (msg i v))) (outerV (zs σ.1 ω t) dg N) s ≤
        εtr i σ.1 ω t s +
          ∑ pos, κ i pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T i σ.1 ω t s pos).g (T i σ.1 ω t s pos).s)
    (hCR : ∀ i < I.r, ∀ ω pos, (forkAt ob σ cmt (T i) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (εtr i σ.1) + (∑ pos, κ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  have h := recursive_sound_upto ob σ (fun R ω t o => AcceptsV (zs R ω t) o) x
    (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt) (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω))
    Prod.snd (fun w => wmsg w.1) same hsame cmt msg T εtr κ q
    (εo := expAt σ fun ω t s => sumV (zs σ.1 ω t) dg N
      (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) (εP := 0)
    ?_ ?_ htr hCR hin
  · rwa [add_zero] at h
  · exact avg_mono fun ω => leafAvg_mono (fun t s => outerV_le hA3 (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht'
      qF qS qW qR (fun Xs => I.accepts (x σ.1 ω) ((wmsg Xs).zip (t.map Prod.snd))) (hbr σ.1 ω t) _) _
  · refine le_of_eq ((congrArg avg (funext fun ω => ?_)).trans (avg_const 0))
    refine (leafAvg_congr_len (G := fun _ _ => 0) (fun t s _ => ?_) _).trans (leafAvg_const 0 _)
    exact le_antisymm ((prob_mono (fun b hb => hb.2 rfl) _ _).trans (prob_false _ _).le) zero_le

end FlockSoundness.Discharge.Recursion
