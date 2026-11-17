module

public import Proofs.Flock.Soundness.Assumptions.CustodyZk
public import Proofs.Flock.Soundness.Discharge.Composed.Sound
import Proofs.Flock.Soundness.Discharge.Exec.SamplingDesignSample
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import Proofs.Flock.Soundness.Discharge.Recursion.Defs
public import Proofs.Flock.Soundness.Discharge.Recursion.Off
public import Proofs.Flock.Claims.Steps.VStar
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
public import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
import Proofs.Flock.Soundness.Game
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! # What V*'s bridge obligation reads of `Recursive.Flock`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg

open ZkHidden (planZHJ tabZHJ planZHJ_length SampleSetupZHJ TagsOkZHJ SampleOkZKHJ layShHJ)

open FlockSoundness.Discharge.Tables (LiveOutJ liveSession)

open scoped ENNReal

set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history, but for its record's custody**: what its soundness takes besides
the circuit `c` (fixed: V* is public), its number of tables `J`, the sampling design `Lv` of V*'s units, the record's custody
(`ZkOuter.hRec`), the count curve and the prover's strategy. V*'s sessions hide their outputs: the circuit file parses
under `HmOut` (`hcH`), the tags are `TagsOkZHJ`, each table is set up by setupHidden (`y₀`, `hDraw`). `rd` is the
registrant's record of the registered reads: the hidden circuit's rows from its registration, the values of the outer
proof's registration `R₂` (`outerV`), and the values the verifier registers itself (the sample, the coins, the comb
`coef`). `hown`: at an accepted outcome, the reads at the rows of the sampled proof blocks are the verifier's own (`OwnReadsHJ`:
its ports, at the positions its program reads, in the table each unit is sampled into). -/
structure ZkOuterD (mPts nv J : ℕ) (Reg' : Type) (c : Flock.Circuit) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (Lv : SamplingDesign nv) where
  I : Inputs
  ht : I.tags.typed = false
  own : Array Flock.Registered.Port
  dj : Finset (Fin nv) → Lean.Json
  S₀ : Finset (Fin nv)
  d₀ : Lean.Json
  y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀
  wr : (S : Finset (Fin nv)) → (R : Reg') → LiveOutJ (planZHJ (Reg := Reg') dj y₀ S R).length mPts → Exec.Transcript
  hcH : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub
  hDraw : FlockSoundness.Assumptions.Zk.DrawCustodyZKHJ I dj mPts Lv
  hTags : TagsOkZHJ I
  hJ : nTab I = J
  rd : FlockSoundness.Registered.ReadsX
    (layShHJ (Reg := Reg') dj y₀ ht hcH (Layout.scopeOkH_of_scopeOk hscope) hU (pub := pub))
    (rsShHJS dj y₀ ht hcH (Layout.scopeOkH_of_scopeOk hscope) hU hpub)
  hown : ∀ ω R' o, LiveAcceptsZKR I own (R', Lv.draw ω, wr (Lv.draw ω) R' o) →
    OwnReadsHJ I own c pub (wr (Lv.draw ω) R' o) (rd.restrict (rowIn (Lv.draw ω))).isRead rd.port rd.readAt

set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history**: `ZkOuterD`'s, with the record's custody `hRec` (the record
`verify --zk` reads is the challenger's of the live session of `J` tables, `RecordsLiveZKJ`). -/
structure ZkOuter (mPts nv J : ℕ) (Reg' : Type) (c : Flock.Circuit) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (Lv : SamplingDesign nv) extends ZkOuterD mPts nv J Reg' c hU hscope Lv where
  hRec : FlockSoundness.Assumptions.Zk.RecordsCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg') dj y₀ S R).length)
    wr

/-- **The names of a session's strategies**: a registration and, at each of the sampling design's coins, a strategy of the live
session of `J` tables; they read the same at every history. -/
abbrev baseGame {nv : ℕ} (Lv : SamplingDesign nv) (Reg' : Type) (dg : ByteArray → ByteArray) (J mPts N : ℕ) :
    Game (LiveOutJ J mPts) :=
  .send Reg' fun _ => .coin Lv.Ω fun _ => liveSession dg J mPts N

/-- A strategy of the live session of `J` tables, as one of `J'` tables when `J' = J`. -/
def castLive {J J' : ℕ} (h : J' = J) (dg : ByteArray → ByteArray) (mPts N : ℕ)
    (s : Strategy (liveSession dg J mPts N)) : Strategy (liveSession dg J' mPts N) :=
  cast (congrArg (fun J => Strategy (liveSession dg J mPts N)) h.symm) s

namespace ZkOuterD

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nv} (zo : ZkOuterD mPts nv J Reg' c hU hscope Lv)

/-- Every table count of the plan is `J`. -/
theorem hlen (S : Finset (Fin nv)) (R : Reg') : (planZHJ (Reg := Reg') zo.dj zo.y₀ S R).length = J :=
  (planZHJ_length zo.dj zo.y₀ S R).trans zo.hJ

end ZkOuterD

namespace ZkOuter

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

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

/-- The prover's strategy in V*'s session, simulated in the compiled batched session (`simZHJ`). -/
noncomputable def sim (τ : Strategy (zo.game dg N)) :=
  simZHJ zo.dj zo.y₀ zo.ht zo.hcH (Layout.scopeOkH_of_scopeOk hscope) hU zo.wr dg N zo.hTags zo.hRec zo.hDraw τ

/-- **V*'s value layer at the strategy `τ`**: the decoded witness (`XplurZC`), each wire's bit in its commit string's
plurality value. -/
noncomputable def layer (k Rw : ℕ) (τ : Strategy (zo.game dg N)) : Fin (Layout.keyN c hU nv) → Bool :=
  XplurZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀) (tabZHJ zo.dj zo.y₀)
    zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw

/-- **A layer agrees with the registrant**: every wire a unit reads at a registered read carries the registrant's bit. -/
def RegAgree (bits : Fin (Layout.keyN c hU nv) → Bool) : Prop :=
  ∀ u, ∀ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u, zo.rd.isRead (zo.lay.pos g) →
    bits g = zo.lay.bit (zo.rd.wv (zo.lay.pos g)) g

/-- **A good layer**: no incorrect proof block, and agreement with the registrant. -/
def Good (bits : Fin (Layout.keyN c hU nv) → Bool) : Prop :=
  ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).incorrect bits = ∅ ∧ zo.RegAgree bits

end ZkOuter

/-- **The sampling design of a public verifier's units**: the full sample, the live subset sampling design of all `nv` units over `[0, nv)` on a
byte source (`live`, `--draw subset:<units>`, as V* is staged), and the model's uniform `nv`-subset it escapes no more
often than (`model`, which misses no unit). -/
structure VSamplingDesign (nv : ℕ) : Type 1 where
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256

attribute [instance] VSamplingDesign.fintypeΩ VSamplingDesign.nonemptyΩ

/-- **A public verifier**: its circuit `c` over `nv` units, proved in sessions of `J` tables, and its sampling design. -/
structure VStmt : Type 1 where
  nv : ℕ
  mPts : ℕ
  J : ℕ
  c : Flock.Circuit
  hU : Built c.unit
  hscope : Layout.scopeOk c = true
  samplingDesign : VSamplingDesign nv

namespace VStmt

/-- A session of the verifier's circuit at one history. -/
abbrev Outer (Q : VStmt) (Reg' : Type) := ZkOuter Q.mPts Q.nv Q.J Reg' Q.c Q.hU Q.hscope Q.samplingDesign.live

/-- A value layer of the verifier's circuit. -/
abbrev Bits (Q : VStmt) : Type := Fin (Layout.keyN Q.c Q.hU Q.nv) → Bool

/-- A name for a strategy of a session of the verifier's circuit. -/
abbrev Name (Q : VStmt) (Reg' : Type) (dg : ByteArray → ByteArray) (N : ℕ) : Type :=
  Strategy (baseGame Q.samplingDesign.live Reg' dg Q.J Q.mPts N)

end VStmt

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- The record of V*'s sessions at one history: each session's name and outcome. -/
abbrev VRec (Vs : Fin S → VStmt) (Reg' : Type) (dg : ByteArray → ByteArray) (N : ℕ) : Type :=
  (j : Fin S) → (Vs j).Name Reg' dg N × (Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript)

/-- **Every session accepts.** -/
def Accepts (zs : (j : Fin S) → (Vs j).Outer Reg') {dg : ByteArray → ByteArray} {N : ℕ} (b : VRec Vs Reg' dg N) :
    Prop :=
  ∀ j, LiveAcceptsZKR (zs j).I (zs j).own (b j).2

/-- **V*'s layers** at the strategies the record names. -/
noncomputable def layers (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (b : VRec Vs Reg' dg N) : (j : Fin S) → (Vs j).Bits :=
  fun j => (zs j).layer dg N k Rw ((zs j).ofBase dg N (b j).1)

variable {Reg₂ : Type}

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **What V*'s circuits compute** (`k, Rw` the count curve's): at every history `(R, ω, t)`, registration `R₂` of the
outer proof and strategies `τ` of V*'s sessions there, when every session's outcome verity-verify-proof `--zk --registered`
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

end FlockSoundness.Discharge.Recursion
