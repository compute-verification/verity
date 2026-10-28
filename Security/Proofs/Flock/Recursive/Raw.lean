import Proofs.Flock.Recursive.Flock
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Mathlib.Data.Fintype.EquivFin

/-!
# Recursion: V*'s sessions as the prover supplies them (gap 8)

A session of V*'s statement is the verifier's own inputs, record writer and registered reads (`ZkOuterR`), at the live
session of its statement's `J` tables. That its inputs parse to V*'s circuit, load V*'s population and set up at some
draw is not taken: the verifier of record's checks establish it at a session that verifies
(`Proofs.Flock.Recursive.Loads`). The proof's sessions (`ZkOuter`) carry that typing in their fields, so this file
builds them from the raw ones and carries the prover's strategy across:

* `ZkOuterR.Typable`: the inputs pass the statement check with the verifier's registered reads (`ScopeZR`), their
  circuit is V*'s, their registered population has V*'s units, and table 0 sets up at some draw. `ZkOuterR.build` is
  the typed session then, with the same game up to the casts along its table count (`build_sigma`).
* `ZkOuter.refuse`: the session with one own port, named outside the circuit (`freshOwn`). It accepts no transcript
  (`not_acceptsZKR_fresh`): the setup's first table has the circuit (`setup_zero_c`), `Registered.check` runs
  `checkPort` on each own port, and `checkPort` throws on a name that no port of the setup's circuit has.
* `Typed zr`: every index has a session that types, at some history and registration. `zsOf` is then the typed
  sessions: the build where a session types, and where it doesn't, a session of the same index that types, refusing
  everything.
* `recT`: the prover's strategy against the raw sessions as one against `zsOf`'s, with the same registrations and
  rounds. Where every session at a registration types, the sessions are the same game (`sessions_zsOf`) and the
  strategy is cast (`sessT`); elsewhere it is any strategy, and the raw event is empty there (`Loads`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed
open ZkHidden (planZHJ planZHJ_length DrawSetupZHJ TagsOkZHJ DrawOkZKHJ layShHJ)
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession liveRoundsJ)
open scoped ENNReal

set_option genSizeOfSpec false in
set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history, as the prover supplies it**: the verifier's inputs `I`, its own
ports `own`, the draw file `dj` and the record writer `wr` of the live session of `J` tables, with `ZkOuterD`'s
`hDraw`, `hTags` and `hJ` (the verifier's own table count is `J`). The registrant's record of the registered reads
`rd`, and `hown`, are `ZkOuterD`'s at each typing of the session (`ht`, a setup `y₀` at table 0, the parse `hc` and the
load `hpub`), which they read. -/
structure ZkOuterR (mPts nv J : ℕ) (Reg' : Type) (c : Flock.Circuit) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (Lv : Law nv) where
  I : Inputs
  own : Array Flock.Registered.Port
  dj : Finset (Fin nv) → Lean.Json
  wr : Finset (Fin nv) → Reg' → LiveOutJ J mPts → Exec.Transcript
  hDraw : FlockSoundness.Assumptions.Zk.DrawCustodyZKHJ I dj mPts Lv
  hTags : TagsOkZHJ I
  hJ : nTab I = J
  rd : ∀ (ht : I.tags.typed = false) (S₀ : Finset (Fin nv)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok (pubZ I)),
    FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg') dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU (pub := pubZ I))
      (rsShHJS dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU hpub)
  hown : ∀ ht S₀ d₀ y₀ hc hpub ω R' o, LiveAcceptsZKR I own (R', Lv.draw ω, wr (Lv.draw ω) R' o) →
    OwnReadsHJ I own c (pubZ I) (wr (Lv.draw ω) R' o) ((rd ht S₀ d₀ y₀ hc hpub).restrict (rowIn (Lv.draw ω))).isRead
      (rd ht S₀ d₀ y₀ hc hpub).port (rd ht S₀ d₀ y₀ hc hpub).readAt

/-- V*'s live session of `J` tables at a draw and registration, its outcome written by `wr`. -/
def liveJ {nv : ℕ} {Reg' : Type} (dg : ByteArray → ByteArray) (J mPts N : ℕ)
    (wr : Finset (Fin nv) → Reg' → LiveOutJ J mPts → Exec.Transcript) (S : Finset (Fin nv)) (R : Reg') :
    Game Exec.Transcript :=
  (liveSession dg J mPts N).map (wr S R)

/-- A live session's outcome of `J'` tables, as one of `J` tables when `J' = J`. -/
def castOut {J J' : ℕ} (h : J' = J) (mPts : ℕ) (o : LiveOutJ J' mPts) : LiveOutJ J mPts :=
  cast (congrArg (fun J => LiveOutJ J mPts) h) o

/-- **Implications at every outcome of `liveJ`'s live audit bound probabilities**: at the registration the prover sends,
each of the law's draws and each output of the live session written by `wr`. -/
theorem prob_auditLive_liveJ_mono {nv : ℕ} {Lv : Law nv} {Reg' : Type} {J mPts : ℕ} (dg : ByteArray → ByteArray)
    (N : ℕ) (wr : Finset (Fin nv) → Reg' → LiveOutJ J mPts → Exec.Transcript)
    (τ : Strategy (auditLive Lv Reg' (liveJ dg J mPts N wr))) {p q : Reg' × Finset (Fin nv) × Exec.Transcript → Prop}
    (h : ∀ ω o', p (τ.1, Lv.draw ω, wr (Lv.draw ω) τ.1 o') → q (τ.1, Lv.draw ω, wr (Lv.draw ω) τ.1 o')) :
    prob p (auditLive Lv Reg' (liveJ dg J mPts N wr)) τ ≤ prob q (auditLive Lv Reg' (liveJ dg J mPts N wr)) τ := by
  show avg (fun ω => prob p ((liveJ dg J mPts N wr (Lv.draw ω) τ.1).map fun t => (τ.1, Lv.draw ω, t)) (τ.2 ω)) ≤
    avg fun ω => prob q ((liveJ dg J mPts N wr (Lv.draw ω) τ.1).map fun t => (τ.1, Lv.draw ω, t)) (τ.2 ω)
  refine avg_mono fun ω => ?_
  rw [prob_map, prob_map]
  unfold liveJ
  rw [prob_map, prob_map]
  exact prob_mono (fun o' => h ω o') _ _

/-- **A live audit whose table counts are all `J` is the live audit of `J` tables**, its names carried across by the
casts along the counts. -/
theorem liveJ_sigma {nv : ℕ} (Lv : Law nv) {Reg' : Type} (dg : ByteArray → ByteArray) (J mPts N : ℕ)
    (wr : Finset (Fin nv) → Reg' → LiveOutJ J mPts → Exec.Transcript) (L : Finset (Fin nv) → Reg' → ℕ)
    (hL : ∀ S R, L S R = J) :
    (⟨auditLive Lv Reg' fun S R => (liveSession dg (L S R) mPts N).map fun o => wr S R (castOut (hL S R) mPts o),
      fun τ₀ => ⟨τ₀.1, fun ω => Strategy.map (fun t => (τ₀.1, Lv.draw ω, t)) _
        (Strategy.map (fun o => wr (Lv.draw ω) τ₀.1 (castOut (hL (Lv.draw ω) τ₀.1) mPts o)) _
          (castLive (hL (Lv.draw ω) τ₀.1) dg mPts N (τ₀.2 ω)))⟩⟩ :
      Σ g : Game (Reg' × Finset (Fin nv) × Exec.Transcript), Strategy (baseGame Lv Reg' dg J mPts N) → Strategy g) =
    ⟨auditLive Lv Reg' (liveJ dg J mPts N wr), fun τ₀ => ⟨τ₀.1, fun ω => Strategy.map (fun t => (τ₀.1, Lv.draw ω, t)) _
      (Strategy.map (wr (Lv.draw ω) τ₀.1) _ (τ₀.2 ω))⟩⟩ := by
  obtain rfl : L = fun _ _ => J := funext fun S => funext fun R => hL S R
  rfl

/-- **The record's custody at `J` tables, at table counts that are all `J`.** -/
theorem recordsLiveZKJ_cast {n mPts J : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json}
    {wr : Finset (Fin n) → Reg → LiveOutJ J mPts → Exec.Transcript} {L : Finset (Fin n) → Reg → ℕ}
    (hL : ∀ S R, L S R = J) (h : RecordsLiveZKJ I dj (fun _ _ => J) wr) :
    RecordsLiveZKJ I dj L fun S R o => wr S R (castOut (hL S R) mPts o) := by
  obtain rfl : L = fun _ _ => J := funext fun S => funext fun R => hL S R
  exact h

/-- **`seqC` of games and names that agree pairwise**: equal as games, their names along. -/
theorem seqC_congr' {S : ℕ} {T α : Fin S → Type} {gs gs' : (j : Fin S) → Game (α j)}
    {of : (j : Fin S) → T j → Strategy (gs j)} {of' : (j : Fin S) → T j → Strategy (gs' j)}
    (h : ∀ j, (⟨gs j, of j⟩ : Σ g : Game (α j), T j → Strategy g) = ⟨gs' j, of' j⟩) :
    seqC S gs of = seqC S gs' of' := by
  have hg : gs = gs' := funext fun j => congrArg Sigma.fst (h j)
  subst hg
  have ho : of = of' := funext fun j => eq_of_heq (Sigma.mk.inj (h j)).2
  subst ho
  rfl

namespace ZkOuterR

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

/-- V*'s live session at the history. -/
abbrev game : Game (Reg' × Finset (Fin nv) × Exec.Transcript) := auditLive Lv Reg' (liveJ dg J mPts N x.wr)

/-- The strategy a name stands for in V*'s session. -/
def ofBase (τ₀ : Strategy (baseGame Lv Reg' dg J mPts N)) : Strategy (x.game dg N) :=
  ⟨τ₀.1, fun ω => Strategy.map (fun t => (τ₀.1, Lv.draw ω, t)) _ (Strategy.map (x.wr (Lv.draw ω) τ₀.1) _ (τ₀.2 ω))⟩

/-- **The session types**: its inputs pass the statement check with the verifier's registered reads (`ScopeZR`), their
circuit is `c`, their registered population has `nv` units, and table 0 sets up at some draw. -/
def Typable : Prop :=
  ScopeZR x.I ∧ circZ x.I = c ∧ (pubZ x.I).n = nv ∧
    ∃ (S₀ : Finset (Fin nv)) (d₀ : Lean.Json), Nonempty (DrawSetupZHJ (inputsJ x.I 0) mPts S₀ d₀)

variable {x}

theorem Typable.ht (hx : x.Typable) : x.I.tags.typed = false :=
  hx.1.typed

theorem Typable.hc (hx : x.Typable) : HmOut.parse (zkTags x.I) x.I.circuitFile x.I.tables = .ok c := by
  rw [hx.1.parseH x.hTags.1, hx.2.1]

theorem Typable.hpub (hx : x.Typable) : HmOut.loadPublic (zkTags x.I) c x.I.publicFile = .ok (pubZ x.I) := by
  have h := hx.1.loadH x.hTags.1
  rwa [hx.2.1] at h

/-- A draw table 0 of the typed session sets up at. -/
noncomputable def Typable.S₀ (hx : x.Typable) : Finset (Fin nv) := hx.2.2.2.choose

/-- Table 0's draw file at `Typable.S₀`. -/
noncomputable def Typable.d₀ (hx : x.Typable) : Lean.Json := hx.2.2.2.choose_spec.choose

/-- The typed session's setup of table 0 at `Typable.S₀`. -/
noncomputable def Typable.y₀ (hx : x.Typable) : DrawSetupZHJ (inputsJ x.I 0) mPts hx.S₀ hx.d₀ :=
  Classical.choice hx.2.2.2.choose_spec.choose_spec

/-- Every table count of the typed session's plan is `J`. -/
theorem Typable.hlen (hx : x.Typable) (S : Finset (Fin nv)) (R : Reg') :
    (planZHJ (Reg := Reg') x.dj hx.y₀ S R).length = J :=
  (planZHJ_length x.dj hx.y₀ S R).trans x.hJ

variable (x)

/-- **The typed session**, at a typing of the raw one and its record's custody: the same inputs, own ports and game, the
outcomes cast along the table count (`Typable.hlen`). -/
noncomputable def build (hx : x.Typable) (hRec : RecordsLiveZKJ x.I x.dj (fun _ _ => J) x.wr) :
    ZkOuter mPts nv J Reg' c hU hscope Lv where
  I := x.I
  ht := hx.ht
  own := x.own
  dj := x.dj
  S₀ := hx.S₀
  d₀ := hx.d₀
  y₀ := hx.y₀
  wr S R o := x.wr S R (castOut (hx.hlen S R) mPts o)
  hcH := hx.hc
  pub := pubZ x.I
  hpub := hx.hpub
  hDraw := x.hDraw
  hTags := x.hTags
  hJ := x.hJ
  rd := x.rd hx.ht hx.S₀ hx.d₀ hx.y₀ hx.hc hx.hpub
  hown ω R' o := x.hown hx.ht hx.S₀ hx.d₀ hx.y₀ hx.hc hx.hpub ω R' (castOut (hx.hlen (Lv.draw ω) R') mPts o)
  hRec := recordsLiveZKJ_cast hx.hlen hRec

/-- **The typed session is the raw one**: the same game and the same strategy at each name. -/
theorem build_sigma (hx : x.Typable) (hRec : RecordsLiveZKJ x.I x.dj (fun _ _ => J) x.wr) :
    (⟨(x.build hx hRec).game dg N, (x.build hx hRec).ofBase dg N⟩ :
      Σ g : Game (Reg' × Finset (Fin nv) × Exec.Transcript), Strategy (baseGame Lv Reg' dg J mPts N) → Strategy g) =
    ⟨x.game dg N, x.ofBase dg N⟩ :=
  liveJ_sigma Lv dg J mPts N x.wr _ hx.hlen

end ZkOuterR

/-- A port name that no port of `c` has. -/
noncomputable def freshName (c : Flock.Circuit) : String :=
  (Infinite.exists_notMem_finset (c.ports.toList.map (·.name)).toFinset).choose

theorem freshName_ne (c : Flock.Circuit) {p : ℕ} (hp : p < c.ports.size) : c.ports[p]!.name ≠ freshName c := by
  intro h
  refine (Infinite.exists_notMem_finset (c.ports.toList.map (·.name)).toFinset).choose_spec ?_
  rw [List.mem_toFinset, List.mem_map]
  exact ⟨c.ports[p], Array.getElem_mem_toList hp, by rw [← getElem!_pos c.ports p hp]; exact h⟩

/-- **An own list no session of `c` accepts**: one port, named outside `c`. -/
noncomputable def freshOwn (c : Flock.Circuit) : Array Flock.Registered.Port :=
  #[{ (default : Flock.Registered.Port) with name := freshName c }]

/-- **A hidden-output setup's first table has the session's circuit**: `setupTables` sets up at least one table, and
its statement's circuit is the one the circuit file parses to (`Spec.parse`; at several tables, table 0's tags differ
from the session's in the identity alone, which an untyped `HmOut.parse` does not read: `parseH_identity`). -/
theorem setup_zero_c {I : Inputs} (ht : I.tags.typed = false) (hh : I.tags.hiddenOutputs = true) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) {r : String} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf (zkInputs I) r = .ok (spec, sts)) : ∃ h0 : 0 < sts.size, sts[0].c = c := by
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  rcases HiddenExec.setupTables_okH hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hlt, hsz, hj, -, -⟩
  · refine ⟨by simp, ?_⟩
    have h1 : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok st.c := (Hidden.setupHidden_spec' hst).parse
    rw [hc] at h1
    exact (Except.ok.inj h1).symm
  · have h0 : 0 < sts.size := by omega
    obtain ⟨d, hd⟩ := hj 0 h0
    refine ⟨h0, ?_⟩
    have h1 : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok sts[0].c :=
      (ZkHidden.parseH_identity ht' _ _ _).symm.trans (Hidden.setupHidden_spec' hd).parse
    rw [hc] at h1
    exact (Except.ok.inj h1).symm

/-- **A session whose own list is `freshOwn c` accepts no transcript**: where the setup succeeds, its first table's
circuit is `c` (`setup_zero_c`), and the registered check's `checkPort` finds no port of `c` named as the own port
(`checkPort_at`). -/
theorem not_acceptsZKR_fresh (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (t : Exec.Transcript) :
    ¬ AcceptsZKR I (freshOwn c) t := by
  rintro ⟨spec, sts, hs, hreg, -, -⟩
  obtain ⟨h0, hc0⟩ := setup_zero_c ht hTags.1 hc hs
  have hck := checkRegistered_ok (array_forM_ok hreg 0 h0)
  rw [hc0] at hck
  obtain ⟨tb, idx, -, -, hports⟩ := check_tables hck
  obtain ⟨p, -, -, hp, -⟩ :=
    checkPort_at (hports { (default : Flock.Registered.Port) with name := freshName c } (by simp [freshOwn]))
  have hlt : p < c.ports.size := List.mem_range.mp (List.mem_of_find?_eq_some hp)
  have hname := List.find?_some hp
  simp only [beq_iff_eq] at hname
  exact freshName_ne c hlt hname

/-- **The session refusing everything**: `z` with the own list `freshOwn` (`not_acceptsZKR_fresh`). -/
noncomputable def ZkOuter.refuse {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (z : ZkOuter mPts nv J Reg' c hU hscope Lv) :
    ZkOuter mPts nv J Reg' c hU hscope Lv :=
  { z with
    own := freshOwn c
    hown := fun _ _ _ h => (not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH _ h).elim }

/-- A session of the verifier's circuit at one history, as the prover supplies it. -/
abbrev VStmt.OuterR (Q : VStmt) (Reg' : Type) := ZkOuterR Q.mPts Q.nv Q.J Reg' Q.c Q.hU Q.hscope Q.law.live

section Sessions

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {dg : ByteArray → ByteArray} {N : ℕ}

/-- **V*'s sessions at one history, in a row**, as the prover supplies them (`sessions`'). -/
abbrev sessionsR (zr : (j : Fin S) → (Vs j).OuterR Reg') (dg : ByteArray → ByteArray) (N : ℕ) :
    Game (VRec Vs Reg' dg N) :=
  seqC S (T := fun j => (Vs j).Name Reg' dg N) (α := fun j => Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript)
    (fun j => (zr j).game dg N) fun j => (zr j).ofBase dg N

/-- **Every session accepts**, as the prover supplies them. -/
def AcceptsR (zr : (j : Fin S) → (Vs j).OuterR Reg') (b : VRec Vs Reg' dg N) : Prop :=
  ∀ j, LiveAcceptsZKR (zr j).I (zr j).own (b j).2

variable {Reg₂ : Type}

/-- **V*'s outer proof at one history**, its sessions as the prover supplies them (`outerV`'s). -/
abbrev outerVR (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg') (dg : ByteArray → ByteArray) (N : ℕ) :
    Game (Reg₂ × VRec Vs Reg' dg N) :=
  .send Reg₂ fun R₂ => (sessionsR (zr R₂) dg N).map fun b => (R₂, b)

/-- **Every session accepts**, at the registration the outcome records, as the prover supplies them. -/
def AcceptsVR (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg') (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsR (zr o.1) o.2

theorem prob_outerVR (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg') (p : Reg₂ × VRec Vs Reg' dg N → Prop)
    (s : Strategy (outerVR zr dg N)) :
    prob p (outerVR zr dg N) s =
      prob (fun b => p (s.1, b)) (sessionsR (zr s.1) dg N) (Strategy.ofMap (fun b => (s.1, b)) _ s.2) :=
  prob_map p (fun b => (s.1, b)) (sessionsR (zr s.1) dg N) s.2

variable {A B C D : Type}

/-- **Every index has a session that types**, at some history and registration. -/
def Typed (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') : Prop :=
  ∀ j, ∃ q : A × B × C × D, (zr q.1 q.2.1 q.2.2.1 q.2.2.2 j).Typable

open Classical in
/-- **The typed sessions**: at a session that types, its build (`ZkOuterR.build`, the same game); at one that doesn't, a
session of the same index that types (`Typed`), refusing everything (`ZkOuter.refuse`). -/
noncomputable def zsOf (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr) :
    A → B → C → D → (j : Fin S) → (Vs j).Outer Reg' := fun R ω t R₂ j =>
  if hx : (zr R ω t R₂ j).Typable then (zr R ω t R₂ j).build hx (hRec R ω t R₂ j)
  else ((zr (h j).choose.1 (h j).choose.2.1 (h j).choose.2.2.1 (h j).choose.2.2.2 j).build (h j).choose_spec
    (hRec _ _ _ _ j)).refuse

open Classical in
theorem zsOf_pos (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr)
    (R : A) (ω : B) (t : C) (R₂ : D) (j : Fin S) (hx : (zr R ω t R₂ j).Typable) :
    zsOf zr hRec h R ω t R₂ j = (zr R ω t R₂ j).build hx (hRec R ω t R₂ j) :=
  dif_pos hx

open Classical in
theorem zsOf_neg (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr)
    (R : A) (ω : B) (t : C) (R₂ : D) (j : Fin S) (hx : ¬ (zr R ω t R₂ j).Typable) :
    zsOf zr hRec h R ω t R₂ j =
      ((zr (h j).choose.1 (h j).choose.2.1 (h j).choose.2.2.1 (h j).choose.2.2.2 j).build (h j).choose_spec
        (hRec _ _ _ _ j)).refuse :=
  dif_neg hx

/-- **Where every session types, the typed sessions are the raw ones**: each build has the raw session's game and
names. -/
theorem sessions_zsOf (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr)
    (R : A) (ω : B) (t : C) (R₂ : D) (hall : ∀ j, (zr R ω t R₂ j).Typable) :
    sessions (zsOf zr hRec h R ω t R₂) dg N = sessionsR (zr R ω t R₂) dg N := by
  have hz : zsOf zr hRec h R ω t R₂ = fun j => (zr R ω t R₂ j).build (hall j) (hRec R ω t R₂ j) :=
    funext fun j => zsOf_pos zr hRec h R ω t R₂ j (hall j)
  rw [hz]
  exact seqC_congr' fun j => (zr R ω t R₂ j).build_sigma dg N (hall j) (hRec R ω t R₂ j)

end Sessions

section Strategies

variable {α : Type}

/-- The prover's strategy in `g` played by `τ`: only coins remain. -/
def playStrat : (g : Game α) → (τ : Strategy g) → Strategy (play g τ)
  | .ret _, _ => PUnit.unit
  | .send _ k, τ => playStrat (k τ.1) τ.2
  | @Game.coin _ _ _ _ k, τ => fun c => playStrat (k c) (τ c)

/-- **A strategy of `seqC` naming `ns j` at each game.** -/
def seqCAt : (S : ℕ) → {T α : Fin S → Type} → (gs : (j : Fin S) → Game (α j)) →
    (of : (j : Fin S) → T j → Strategy (gs j)) → ((j : Fin S) → T j) → Strategy (seqC S gs of)
  | 0, _, _, _, _, _ => PUnit.unit
  | S + 1, T, α, gs, of, ns =>
      Strategy.bind (committed (gs 0) (of 0))
        (fun r => (seqC S (T := fun j => T j.succ) (fun j => gs j.succ) fun j => of j.succ).map fun rs =>
          (Fin.cons r rs : (j : Fin (S + 1)) → T j × α j))
        ⟨ns 0, Strategy.map (fun a => (ns 0, a)) _ (playStrat (gs 0) (of 0 (ns 0)))⟩ fun r =>
        Strategy.map (fun rs => (Fin.cons r rs : (j : Fin (S + 1)) → T j × α j)) _
          (seqCAt S (T := fun j => T j.succ) (α := fun j => α j.succ) (fun j => gs j.succ) (fun j => of j.succ)
            fun j => ns j.succ)

/-- The prover stopping the live rounds of `J` tables at once. -/
def liveRoundsJStop (dg : ByteArray → ByteArray) (J N : ℕ) (rs) : Strategy (liveRoundsJ dg J N rs) :=
  match N with
  | 0 => PUnit.unit
  | _ + 1 => ⟨none, PUnit.unit⟩

/-- A strategy of the live session of `J` tables: no roots, no rounds, no proof files. -/
def liveSessionStop (dg : ByteArray → ByteArray) (J mPts N : ℕ) : Strategy (liveSession dg J mPts N) :=
  ⟨#[], fun _ => Strategy.bind _ _ (liveRoundsJStop dg J N _) fun _ => ⟨#[], PUnit.unit⟩⟩

/-- A name for a strategy of a session of the verifier's circuit, registering `R`. -/
def nameAt (Q : VStmt) (Reg' : Type) (dg : ByteArray → ByteArray) (N : ℕ) (R : Reg') : Q.Name Reg' dg N :=
  ⟨R, fun _ => liveSessionStop dg Q.J Q.mPts N⟩

/-- `Strategy.ofMap` undoes `Strategy.map`. -/
theorem ofMap_map {β : Type} (f : α → β) : ∀ (g : Game α) (s : Strategy g), Strategy.ofMap f g (Strategy.map f g s) = s
  | .ret _, _ => rfl
  | .send _ k, s => by
      obtain ⟨m, s'⟩ := s
      exact congrArg (Sigma.mk m) (ofMap_map f (k m) s')
  | @Game.coin _ _ _ _ k, s => funext fun c => ofMap_map f (k c) (s c)

theorem prob_cast {G G' : Game α} (e : G = G') (q : α → Prop) (s : Strategy G') :
    prob q G (cast (congrArg Strategy e.symm) s) = prob q G' s := by
  subst e
  rfl

end Strategies

section Rounds

variable {M C : Type} [Fintype C] [Nonempty C]

/-- **A strategy of the rounds with each leaf's strategy carried by `φ`**: the same messages. -/
def mapRounds {γ γ' : Type} : (r : ℕ) → (f : List (M × C) → Game γ) → (f' : List (M × C) → Game γ') →
    ((t : List (M × C)) → Strategy (f t) → Strategy (f' t)) → Strategy (rounds M C r f) → Strategy (rounds M C r f')
  | 0, _, _, φ, s => φ [] s
  | r + 1, f, f', φ, s => ⟨s.1, fun ρ =>
      mapRounds r (fun t => f ((s.1, ρ) :: t)) (fun t => f' ((s.1, ρ) :: t)) (fun t => φ ((s.1, ρ) :: t)) (s.2 ρ)⟩

theorem leafAvg_mapRounds {γ γ' : Type} : ∀ (r : ℕ) (f : List (M × C) → Game γ) (f' : List (M × C) → Game γ')
    (φ : (t : List (M × C)) → Strategy (f t) → Strategy (f' t)) (F : (t : List (M × C)) → Strategy (f' t) → ℝ≥0∞)
    (s : Strategy (rounds M C r f)),
    leafAvg r f' F (mapRounds r f f' φ s) = leafAvg r f (fun t s' => F t (φ t s')) s
  | 0, _, _, _, _, _ => rfl
  | r + 1, f, f', φ, F, s => congrArg avg (funext fun ρ =>
      leafAvg_mapRounds r (fun t => f ((s.1, ρ) :: t)) (fun t => f' ((s.1, ρ) :: t)) (fun t => φ ((s.1, ρ) :: t))
        (fun t => F ((s.1, ρ) :: t)) (s.2 ρ))

end Rounds

section Transport

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}

/-- **The recursive prover's strategy with each outer session's strategy carried by `φ`**: the same registration,
messages and coins. -/
def recMap {outer outer' : Reg → L.Ω → List (Cm × I.Coin) → Game β}
    (φ : ∀ R ω t, Strategy (outer R ω t) → Strategy (outer' R ω t)) (σ : Strategy (recGame L Reg I Cm outer)) :
    Strategy (recGame L Reg I Cm outer') :=
  ⟨σ.1, fun ω => mapRounds I.r _ _ (fun t s => Strategy.map (fun b => (σ.1, ω, t, b)) (outer' σ.1 ω t)
    (φ σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s))) (σ.2 ω)⟩

/-- **A bound at every outer session bounds the recursive audit**: when each outer session's event at a whole history
(one entry per inner round, the only ones `rounds` reaches) is no likelier than `p'`'s at the strategy `φ` carries it
to, so is the recursive audit's. -/
theorem prob_recMap_le {outer outer' : Reg → L.Ω → List (Cm × I.Coin) → Game β}
    (φ : ∀ R ω t, Strategy (outer R ω t) → Strategy (outer' R ω t)) (p p' : Reg × L.Ω × List (Cm × I.Coin) × β → Prop)
    (σ : Strategy (recGame L Reg I Cm outer))
    (h : ∀ ω t s, t.length = I.r → prob (fun b => p (σ.1, ω, t, b)) (outer σ.1 ω t) s ≤
      prob (fun b => p' (σ.1, ω, t, b)) (outer' σ.1 ω t) (φ σ.1 ω t s)) :
    prob p (recGame L Reg I Cm outer) σ ≤ prob p' (recGame L Reg I Cm outer') (recMap φ σ) := by
  rw [prob_recGame, prob_recGame]
  show avg (fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
      (fun t s => prob (fun b => p (σ.1, ω, t, b)) (outer σ.1 ω t)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)) ≤
    avg (fun ω => leafAvg I.r (fun t => (outer' σ.1 ω t).map fun b => (σ.1, ω, t, b))
      (fun t s => prob (fun b => p' (σ.1, ω, t, b)) (outer' σ.1 ω t)
        (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer' σ.1 ω t) s))
      (mapRounds I.r _ _ (fun t s => Strategy.map (fun b => (σ.1, ω, t, b)) (outer' σ.1 ω t)
        (φ σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s))) (σ.2 ω)))
  refine avg_mono fun ω => ?_
  rw [leafAvg_mapRounds]
  refine leafAvg_mono_len (fun t s ht => ?_) _
  rw [ofMap_map]
  exact h ω t _ ht

/-- **An event empty at every outer session is empty.** -/
theorem prob_recGame_zero {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}
    (p : Reg × L.Ω × List (Cm × I.Coin) × β → Prop) (σ : Strategy (recGame L Reg I Cm outer))
    (h : ∀ ω t s, prob (fun b => p (σ.1, ω, t, b)) (outer σ.1 ω t) s = 0) : prob p (recGame L Reg I Cm outer) σ = 0 :=
  le_antisymm ((prob_recMap_le (fun _ _ _ s => s) p (fun _ => False) σ fun ω t s _ =>
    (h ω t s).le.trans (zero_le ..)).trans (prob_false _ _).le) (zero_le ..)

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}

/-- The registration the prover's first name in `s` sends (none when there are no sessions). -/
def regOf : {S : ℕ} → {Vs : Fin S → VStmt} → (zr : (j : Fin S) → (Vs j).OuterR Reg') →
    Strategy (sessionsR zr dg N) → Fin S → Reg'
  | 0, _, _, _, j => j.elim0
  | _ + 1, _, _, s, _ => s.1.1

open Classical in
/-- **The prover's strategy against the raw sessions, as one against `zs`**: cast where they are one game, and
otherwise naming at each session a strategy that stops at once, registering what the prover's first name does. -/
noncomputable def sessT (zs : (j : Fin S) → (Vs j).Outer Reg') (zr : (j : Fin S) → (Vs j).OuterR Reg')
    (s : Strategy (sessionsR zr dg N)) : Strategy (sessions zs dg N) :=
  if e : sessions zs dg N = sessionsR zr dg N then cast (congrArg Strategy e.symm) s
  else seqCAt S (T := fun j => (Vs j).Name Reg' dg N) (α := fun j => Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript)
    (fun j => (zs j).game dg N) (fun j => (zs j).ofBase dg N) fun j => nameAt (Vs j) Reg' dg N (regOf zr s j)

open Classical in
theorem prob_sessT (zs : (j : Fin S) → (Vs j).Outer Reg') (zr : (j : Fin S) → (Vs j).OuterR Reg')
    (e : sessions zs dg N = sessionsR zr dg N) (q : VRec Vs Reg' dg N → Prop) (s : Strategy (sessionsR zr dg N)) :
    prob q (sessions zs dg N) (sessT zs zr s) = prob q (sessionsR zr dg N) s := by
  unfold sessT
  rw [dif_pos e, prob_cast e]

/-- **The prover's strategy in the raw outer proof, as one in `zs`'s**: the same registration `R₂`, and `sessT` after
it. -/
noncomputable def outerT (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (s : Strategy (outerVR zr dg N)) : Strategy (outerV zs dg N) :=
  ⟨s.1, Strategy.map (fun b => (s.1, b)) _ (sessT (zs s.1) (zr s.1) (Strategy.ofMap (fun b => (s.1, b)) _ s.2))⟩

theorem prob_outerT (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (q : Reg₂ × VRec Vs Reg' dg N → Prop) (s : Strategy (outerVR zr dg N)) :
    prob q (outerV zs dg N) (outerT zs zr s) =
      prob (fun b => q (s.1, b)) (sessions (zs s.1) dg N)
        (sessT (zs s.1) (zr s.1) (Strategy.ofMap (fun b => (s.1, b)) _ s.2)) := by
  show prob q ((sessions (zs s.1) dg N).map fun b => (s.1, b)) (Strategy.map _ _ _) = _
  rw [prob_map, ofMap_map]

/-- **The recursive prover's strategy against the raw sessions, as one against the typed sessions `zsOf`**
(`outerT` at each outer session). -/
noncomputable def recT (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N)) :
    Strategy (recGame L Reg I Cm fun R ω t => outerV (zsOf zr hRec h R ω t) dg N) :=
  recMap (fun R ω t => outerT (zsOf zr hRec h R ω t) (zr R ω t)) σ

end Transport

end FlockSoundness.Discharge.Recursion
