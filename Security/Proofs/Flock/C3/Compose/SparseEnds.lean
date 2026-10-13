import Proofs.Flock.C3.Compose.PartRead
import Proofs.Flock.C3.Reps.Tie
import Proofs.Flock.C3.Sparse.SrcTie

/-!
# n10: the sparse phases' part ends in a coin, at a count the class fixes; R's ring switches are read at their rounds

`partEnds_sparse`: `PartEnds sparsePart` at `sparseCoins`, from the reps' and the sparse phases' ties. The part is the
cap's and the points' steps, then the reps' and the sparse phases'. A sparse phase takes its exact shape, which ends in
a coin (`sparseKinds_ends`). A rep's spine is only a prefix, but every early return of `holoRepC` is `PendL.reject`,
whose verdict is false (`holoRepC_on`), so an accepting rep takes its whole shape (`SpineOn.kinds`), which reads only
the statement, the extra claims' number (two per region) and the reps' schedule `e.sch`.

`rsRead_of_layer`: `RsRead` where the accepted runs' copies of the `rec-opening` unit are one (`OneCopy`) and the
copy's ring-switch wires are read wires of their rounds' commit strings (`RsSk`, `RsIo`). The layers are good at an
accepted outer proof, so each bit of a ring switch's element is the layer's at one wire in both runs.
-/

namespace FlockC3.Compose

noncomputable section

section shape

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting FlockSoundness.Discharge.Exec
open FlockC3.HoloValues (holoRepShape holoLincheck_exact)

/-- **A spine that stops early only where `P` fails**: every run of `g` takes steps of the kinds of `σ`, in order, and
returns before the end of `σ` only at a value outside `P`. -/
inductive SpineOn {α : Type} (P : α → Prop) : Game α → List Kind → Prop
  | ret (a : α) (σ : List Kind) : σ = [] ∨ ¬ P a → SpineOn P (.ret a) σ
  | send {M : Type} {k : M → Game α} {σ : List Kind} : (∀ m, SpineOn P (k m) σ) → SpineOn P (.send M k) (.msg M :: σ)
  | coin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} {σ : List Kind} :
      (∀ x, SpineOn P (k x) σ) → SpineOn P (@Game.coin α C iF iN k) (.coin C :: σ)

/-- **A run to a value in `P` takes the whole shape.** -/
theorem SpineOn.kinds {α : Type} {g : Game α} {ts : List Step} {a : α} (hp : PlaysT g ts a) :
    ∀ {P : α → Prop} {σ : List Kind}, SpineOn P g σ → P a → ts.map kindOf = σ := by
  induction hp with
  | ret a => intro P σ hs ha; cases hs with | ret _ _ h => exact h.elim Eq.symm fun h => absurd ha h
  | send m _ ih => intro P σ hs ha; cases hs with | send hk => exact congrArg (_ :: ·) (ih (hk m) ha)
  | coin x _ ih => intro P σ hs ha; cases hs with | coin hk => exact congrArg (_ :: ·) (ih (hk x) ha)

theorem SpineOn.of_exact {α : Type} {P : α → Prop} {g : Game α} {σ : List Kind} (h : Exact g σ) : SpineOn P g σ := by
  induction h with
  | ret a => exact .ret a [] (.inl rfl)
  | send _ ih => exact .send ih
  | coin _ ih => exact .coin ih

theorem SpineOn.exact_bind {α β : Type} {P : β → Prop} {g : Game α} {f : α → Game β} {σ₁ σ₂ : List Kind}
    (h₁ : Exact g σ₁) (h₂ : ∀ a, SpineOn P (f a) σ₂) : SpineOn P (g >>= f) (σ₁ ++ σ₂) := by
  induction h₁ with
  | ret a => exact h₂ a
  | send _ ih => exact .send ih
  | coin _ ih => exact .coin ih

theorem SpineOn.bind_pure {α β : Type} {Q : α → Prop} {P : β → Prop} {g : Game α} {σ : List Kind}
    (hs : SpineOn Q g σ) (f : α → β) (hf : ∀ a, P (f a) → Q a) : SpineOn P (g >>= fun a => pure (f a)) σ := by
  induction hs with
  | ret a σ h => exact .ret _ σ (h.imp_right fun hq hp => hq (hf a hp))
  | send _ ih => exact .send ih
  | coin _ ih => exact .coin ih

theorem SpineOn.recv_bind {β M : Type} {P : β → Prop} {f : M → Game β} {σ : List Kind}
    (h : ∀ m, SpineOn P (f m) σ) : SpineOn P (recv M >>= f) (.msg M :: σ) := .send h

theorem SpineOn.draw_bind {β C : Type} [Fintype C] [Nonempty C] {P : β → Prop} {f : C → Game β} {σ : List Kind}
    (h : ∀ x, SpineOn P (f x) σ) : SpineOn P (draw C >>= f) (.coin C :: σ) := .coin h

/-- A list of kinds' coins. -/
def kindCoins : List Kind → ℕ
  | [] => 0
  | .msg _ :: σ => kindCoins σ
  | .coin _ :: σ => kindCoins σ + 1

theorem coins_eq_kindCoins : ∀ l : List Step, (l.filterMap Sum.getRight?).length = kindCoins (l.map kindOf)
  | [] => rfl
  | .inl _ :: l => coins_eq_kindCoins l
  | .inr _ :: l => congrArg (· + 1) (coins_eq_kindCoins l)

theorem ends_of_kinds {l : List Step} {σ : List Kind} {C : Type} (h : l.map kindOf = σ ++ [.coin C]) :
    ∃ (l' : List Step) (c : Game.Msg), l = l' ++ [.inr c] := by
  obtain ⟨l', s, rfl⟩ : ∃ l' s, l = l' ++ [s] := by
    rcases List.eq_nil_or_concat l with h0 | ⟨l', s, h0⟩
    · subst h0; cases σ <;> simp at h
    · exact ⟨l', s, h0.trans (List.concat_eq_append ..)⟩
  rcases s with m | c
  · rw [List.map_append] at h
    have := List.append_inj_right' h rfl
    simp [kindOf] at this
  · exact ⟨l', c, rfl⟩

theorem ends_append {x : Kind} (a : List Kind) {b : List Kind} (h : ∃ σ, b = σ ++ [x]) :
    ∃ σ, a ++ b = σ ++ [x] := by
  obtain ⟨σ, rfl⟩ := h
  exact ⟨a ++ σ, (List.append_assoc _ _ _).symm⟩

theorem replicate_ends (x : Kind) (n : ℕ) : ∃ σ, List.replicate (n + 1) x = σ ++ [x] :=
  ⟨_, List.replicate_succ' ..⟩

/-- **The sparse phase ends in a coin**: `ρ_t`'s last where `T > 1`, `ζ_p`'s last (`pLog ≥ 1`) otherwise. -/
theorem sparseKinds_ends (F : Type) (sh : SparseShape) : ∃ σ, FlockC3.Sparse.sparseKinds F sh = σ ++ [.coin F] := by
  unfold FlockC3.Sparse.sparseKinds
  refine ends_append _ (ends_append _ (ends_append _ (ends_append _ (ends_append _ (ends_append _ ?_)))))
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, ht, List.replicate_zero, List.nil_append, List.append_nil]
    exact replicate_ends _ _
  · obtain ⟨n, hn⟩ : ∃ n, sh.tLog = n + 1 := ⟨sh.tLog - 1, by omega⟩
    exact ends_append _ (ends_append _ (hn ▸ replicate_ends _ n))

theorem extraClaims_length (S : Statement) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    (extraClaims S ptLocal mPts pts).length = 2 * S.regions.length := by
  unfold extraClaims
  induction S.regions with
  | nil => rfl
  | cons R rs ih => simp only [List.flatMap_cons, List.length_append, List.length_cons, List.length_nil, ih]; omega

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

omit [DecidableEq F] [Fintype K] in
/-- **Compiled Ligerito's recursive levels take their exact shape** at a level or more. -/
theorem recursiveRoundsC_exact (A : Arith F K) (r n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    ∀ (lv : Level) (rest : List Level) (st : LigStateC F K D),
      Exact (recursiveRoundsC A r n b initial ext (lv :: rest) st) (recShape F K D ext (lv :: rest))
  | lv, [], st => by
      unfold recursiveRoundsC recShape
      refine Exact.bind (foldRoundsC_exact A _ _ _) fun x => ?_
      exact Exact.recv_bind fun yr => by
        rw [← List.append_nil (qbShape F K lv false)]
        exact Exact.bind (queryAndBatchC_exact A r lv _ _ _ false) fun st' => .ret _
  | lv, lv' :: rest, st => by
      unfold recursiveRoundsC
      rw [recShape]
      refine Exact.bind (foldRoundsC_exact A _ _ _) fun x => ?_
      refine Exact.recv_bind fun C' => ?_
      refine Exact.bind (twoOodC_exact A _ _ _ _) fun y => ?_
      exact Exact.bind (queryAndBatchC_exact A r lv _ _ _ true) fun st' =>
        recursiveRoundsC_exact A r n b initial ext lv' rest _

omit [DecidableEq F] [Fintype K] in
/-- **Compiled Ligerito stops early only on a reject.** -/
theorem ligeritoC_on (A : Arith F K) (r : ℕ) (sch : Schedule) (T : F) (b : ℕ → F) (os : Opens F D) :
    SpineOn (fun p : PendL F D => p.verdict os = true) (ligeritoC (D := D) A r sch T b) (ligShape F K D sch) := by
  unfold ligeritoC ligShape
  rcases hl : sch.levels with _ | ⟨lv₀, rest⟩
  · exact .ret _ _ (.inl rfl)
  · refine SpineOn.draw_bind fun z₀ => SpineOn.recv_bind fun y₀ => SpineOn.draw_bind fun β₀ =>
      SpineOn.recv_bind fun msg => ?_
    refine SpineOn.exact_bind (foldRoundsC_exact A _ _ _) fun x => ?_
    rcases rest with _ | ⟨lv₁, rest'⟩
    · exact .ret _ _ (.inl rfl)
    · refine SpineOn.recv_bind fun C₁ => ?_
      refine SpineOn.exact_bind (twoOodC_exact A _ _ _ _) fun y => ?_
      exact SpineOn.exact_bind (queryAndBatchC_exact A r lv₀ _ _ _ true) fun st =>
        SpineOn.of_exact (recursiveRoundsC_exact A r _ b _ sch.ext lv₁ rest' _)

omit [Fintype K] in
/-- **A holo rep stops early only on a reject**: each early return is `PendL.reject`, whose verdict is false. -/
theorem holoRepC_on (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) (os : Opens F D) :
    SpineOn (fun x : Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) × PendL F D => x.2.verdict os = true)
      (holoRepC (D := D) A r S sch hm extra) (holoRepShape F K D S.m S.kLog extra.length sch) := by
  unfold holoRepC holoRepShape
  refine SpineOn.exact_bind (zerocheck_exact A S.m) fun z => ?_
  rcases z with _ | zc
  · exact .ret _ _ (.inr Bool.false_ne_true)
  · refine SpineOn.exact_bind (holoLincheck_exact A S zc) fun l => ?_
    rcases l with _ | lc
    · exact .ret _ _ (.inr Bool.false_ne_true)
    · refine SpineOn.exact_bind (opening_exact A _ (lc.ab :: cClaim A zc :: extra)) fun o => ?_
      rcases o with _ | op
      · exact .ret _ _ (.inr Bool.false_ne_true)
      · exact SpineOn.bind_pure (ligeritoC_on A r sch op.T op.b os) _ fun _ h => h

end shape

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (repPart RepTie SrcRepTie)
open FlockC3.Sparse (SparseTie SrcSparseTie sparseKinds)
open FlockVBridge.InnerAccepts (firstSteps)
open FlockVBridge (keyInst)
open FlockVBridge.Family (skOf maskOf)
open FlockVBridge.OfVStar (OpeningZeroAt rowCol partRep fill_rowCol part_eqOpening wIdx_size)
open FlockVBridge.PartS (wIdx)

/-- **The sparse phases' part's coins**: the points', two reps' at the class's statement, extra claims (two per
region) and schedule, and two sparse phases' at its shape. -/
def sparseCoins (e : ExecHolo) : ℕ :=
  1 + 2 * kindCoins (FlockC3.HoloValues.holoRepShape GF128 GF256 Dig (e.cls.pubStatement e.regs).m
    (e.cls.pubStatement e.regs).kLog (2 * (e.cls.pubStatement e.regs).regions.length) e.sch) +
    2 * kindCoins (sparseKinds GF128 e.cls.shape)

/-- **The sparse phases' part has `sparseCoins` coins and ends in one**, where both reps and both sparse phases are
tied. -/
theorem sparsePart_ends {e : ExecHolo} {u : C3Run e} (hr : ∀ i, RepTie e u i) (hs : ∀ i, SparseTie e u i) :
    ((sparsePart u).filterMap Sum.getRight?).length = sparseCoins e ∧
      ∃ (l : List Step) (c : Game.Msg), sparsePart u = l ++ [.inr c] := by
  have hkr : ∀ i, (u.ts i).map Exec.kindOf = FlockC3.HoloValues.holoRepShape GF128 GF256 Dig
      (e.cls.pubStatement e.regs).m (e.cls.pubStatement e.regs).kLog
      (2 * (e.cls.pubStatement e.regs).regions.length) e.sch := fun i => by
    obtain ⟨U⟩ := hr i
    obtain ⟨_, _, _, hT, hv, -⟩ := FlockC3.HoloValues.rep_of_unit e.cls.hm U
    rw [SpineOn.kinds hT (holoRepC_on _ _ _ _ _ _ u.os.1) hv, extraClaims_length]
  have hks : ∀ i, (u.sp i).map Exec.kindOf = sparseKinds GF128 e.cls.shape := fun i => by
    obtain ⟨_, hx⟩ := FlockC3.Sparse.plays_of_tie (hs i)
    exact Exec.Exact.kinds hx (FlockC3.Sparse.sparsePhase_exact _ _ _ _ _)
  have h2 : (((runSteps u).take 2).filterMap Sum.getRight?).length = 1 := by
    rw [runSteps, FlockC3.PubRows.holoSteps_eq]; rfl
  obtain ⟨σ, hσ⟩ := sparseKinds_ends GF128 e.cls.shape
  obtain ⟨l, c, hl⟩ := ends_of_kinds ((hks 1).trans hσ)
  refine ⟨?_, (runSteps u).take 2 ++ (firstSteps (u.ts 0) ++ u.ts 1 ++ u.sp 0) ++ l, c, ?_⟩
  · rw [sparsePart, coins_append, h2]
    simp only [coins_append, FlockVBridge.Family.firstSteps_rights, coins_eq_kindCoins, hkr, hks, sparseCoins]
    omega
  · rw [sparsePart, hl]
    simp only [List.append_assoc]

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`PartEnds sparsePart` at `sparseCoins`**, at a source tied to V*'s rep and RecSparse units. -/
theorem partEnds_sparse
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hr : SrcRepTie zs dg N k Rw src) (hs : SrcSparseTie zs dg N k Rw src) :
    PartEnds sparsePart zs dg N k Rw src (sparseCoins (holoAt st hp)) :=
  fun R ω t R₂ τ o hA => sparsePart_ends (hr R ω t R₂ τ o hA) (hs R ω t R₂ τ o hA)

/-! ## R's ring switches read at their rounds -/

/-- **A copy of the `rec-opening` unit registering w at an accepted outer proof**: copy `j` of unit `un` in session `s`
is part `p` of the groups `gs` of the opening at `sh`, its row ports w's (`OpeningZeroAt`, `RsAt`'s copy). -/
def OpenCopy (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (s : Fin S) (un : Fin (Vs s).nv) (j p : ℕ) : Prop :=
  ∃ R ω t R₂ τ o v, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o ∧
    OpeningZeroAt (zs R ω t R₂ s) (layers zs dg N k Rw R ω t R₂ τ s) un j sh gs p
      (RecAlgebra.el (w R ω t (layers zs dg N k Rw R ω t R₂ τ))) v

/-- **One copy of the `rec-opening` unit at `sh`** (named): the copies registering w at accepted outer proofs are one
copy of one unit in one session. -/
def OneCopy (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) : Prop :=
  ∀ gs (s s' : Fin S) (un : Fin (Vs s).nv) (un' : Fin (Vs s').nv) (j j' p p' : ℕ),
    OpenCopy zs dg N k Rw sh w gs s un j p → OpenCopy zs dg N k Rw sh w gs s' un' j' p' →
      s = s' ∧ un.val = un'.val ∧ j = j'

/-- **A wire of ring switch `i`'s columns**: at a copy of the `rec-opening` unit registering w, bit `b` of the part's
row-port element `q`, which is ring switch `i`'s slot `t` of w (`RecOpening.wi`). -/
def RsWire (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) (i : ℕ) (s : Fin S)
    (un : Fin (Vs s).nv) (g : Fin (Layout.keyN (Vs s).c (Vs s).hU (Vs s).nv)) : Prop :=
  ∃ (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (j p : ℕ) (hj : j < (Vs s).c.upv) (hpg : p < gs.size)
    (S₁ : RecOpen.GfS) (ns : List ℕ) (nv q : ℕ) (t : Fin 128) (e b : ℕ)
    (hc : rowCol (partRep ns nv).ports 0 (128 * q + b) < (Rows.ofNet (Vs s).hU).w),
    OpenCopy zs dg N k Rw sh w gs s un j p ∧ RecOpening.part sh p = .ok (S₁, ns, nv) ∧
      (wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]? = some e ∧ RecOpening.wi sh (.rs i t.val) = some e ∧ b < 128 ∧
      g = (keyInst (Vs s).hU (Vs s).nv (Layout.outsOf (Vs s).c (Vs s).hU (Vs s).nv) un).wire
        ((Rows.ofNet (Vs s).hU).stackCol (Vs s).c.upv hj ⟨_, hc⟩)

/-- **R's ring switches are registered under their rounds' values** (named, `skOf`'s form): a wire of ring switch
`i < 2`'s columns is on a commit string of stage `c₀ + i`. -/
def RsSk (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (c₀ : ℕ) : Prop :=
  ∀ i < 2, ∀ s un g, RsWire zs dg N k Rw sh w i s un g →
    posK (Vs s).c (Vs s).hU (Vs s).nv g ∈ skOf Vs own sel (c₀ + i) s

/-- **R's ring switches are read by their unit** (named, `IoColAt`'s form at copy `j`): a wire of ring switch `i < 2`'s
columns is one of the unit's. -/
def RsIo (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) : Prop :=
  ∀ i < 2, ∀ s un g, RsWire zs dg N k Rw sh w i s un g →
    g ∈ ((Placed.keyProg (Vs s).c (Vs s).hU (Vs s).nv).partition (Layout.outsOf (Vs s).c (Vs s).hU (Vs s).nv)).io un

/-- **R's ring switches are read at their rounds** (`RsRead`): the two runs' copies are one (`OneCopy`), so a bit of a
ring switch's element is the filled layer's at one wire in both (`fill_rowCol`, `copyVal_input`), the layer's there,
the layers being good (`fill_io`), and a read wire of stage `c₀ + i`'s commit strings (`RsSk`, `RsIo`). -/
theorem rsRead_of_layer
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {sh : Flock.Shape} {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (h1 : OneCopy zs dg N k Rw sh w) (hsk : RsSk zs dg N k Rw sh w own sel c₀) (hio : RsIo zs dg N k Rw sh w) :
    RsRead zs dg N k Rw sh w own sel c₀ := by
  intro i hi R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX S0 gs x x' hS hg hx hx'
  obtain ⟨s, un, j, p, k₀, v, hj, hpg, S₁, ns, nv, hz, hpart, hxb⟩ := hx
  obtain ⟨s', un', j', p', k₀', v', hj', hpg', S₁', ns', nv', hz', hpart', hxb'⟩ := hx'
  have hc1 : OpenCopy zs dg N k Rw sh w gs s un j p := ⟨R, ω, t, R₂, τ, o, v, hA, hz⟩
  obtain ⟨rfl, hu, rfl⟩ := h1 gs s s' un un' j j' p p' hc1 ⟨R', ω', t', R₂', τ', o', v', hA', hz'⟩
  obtain rfl : un = un' := Fin.ext hu
  obtain ⟨_, _, _, _, _, m, cl, hpm, _, S₂, ns₂, nv₂, hpart₂, hw, _, _⟩ := id hz
  obtain ⟨_, _, _, _, _, m', cl', hpm', -⟩ := id hz'
  obtain ⟨-, -, rfl⟩ : m = m' ∧ cl = cl' ∧ p = p' := by
    simpa only [Option.some.injEq, Prod.mk.injEq] using hpm.symm.trans hpm'
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₁' ∧ ns = ns' ∧ nv = nv' := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart')
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₂ ∧ ns = ns₂ ∧ nv = nv₂ := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart₂)
  have hAg : (zs R ω t R₂ s).RegAgree (layers zs dg N k Rw R ω t R₂ τ s) := (hA.1 s).2.2
  have hAg' : (zs R' ω' t' R₂' s).RegAgree (layers zs dg N k Rw R' ω' t' R₂' τ' s) := (hA'.1 s).2.2
  funext tt
  obtain ⟨e, he, hwi, hb⟩ := hxb tt
  obtain ⟨e', he', hwi', hb'⟩ := hxb' tt
  obtain rfl : e = e' := Option.some.inj (hwi.symm.trans hwi')
  obtain ⟨hk, hke⟩ := Array.getElem?_eq_some_iff.mp he
  obtain ⟨hk', hke'⟩ := Array.getElem?_eq_some_iff.mp he'
  refine FlockVBridge.AccChain.eq_of_bitOf fun b hb₀ => ?_
  obtain ⟨-, hns, -⟩ := part_eqOpening hS hg hpg hpart
  have hq : k₀' + tt.val < ns.sum := by rw [hns, ← wIdx_size]; exact hk'
  obtain ⟨hcn, -⟩ := hw (128 * (k₀' + tt.val) + b) (by omega)
  have hcw : rowCol (partRep ns nv).ports 0 (128 * (k₀' + tt.val) + b) < (Rows.ofNet (Vs s).hU).w := by
    simp only [Rows.w]; omega
  obtain ⟨g, hgd⟩ : ∃ g : Fin (Layout.keyN (Vs s).c (Vs s).hU (Vs s).nv),
      g = (keyInst (Vs s).hU (Vs s).nv (Layout.outsOf (Vs s).c (Vs s).hU (Vs s).nv) un).wire
        ((Rows.ofNet (Vs s).hU).stackCol (Vs s).c.upv hj ⟨_, hcw⟩) := ⟨_, rfl⟩
  have hW : RsWire zs dg N k Rw sh w i s un g :=
    ⟨gs, j, p, hj, hpg, S₁, ns, nv, k₀' + tt.val, tt, e, b, hcw, hc1, hpart, he', hwi, hb₀, hgd⟩
  have hm : layers zs dg N k Rw R ω t R₂ τ s g = layers zs dg N k Rw R' ω' t' R₂' τ' s g := by
    have h := congrFun (congrFun hX s) g
    simp only [maskOf] at h
    split_ifs at h with hC
    · exact h
    · exact (hC ⟨hsk i hi s un g hW, un, hio i hi s un g hW⟩).elim
  rw [← hb b hb₀, fill_rowCol _ hz hj hS hg hpg hpart hk hb₀, hke, ← hke',
    ← fill_rowCol _ hz hj hS hg hpg hpart hk' hb₀, ← hb' b hb₀, FlockVBridge.copyVal_input _ _ hj hcn,
    FlockVBridge.copyVal_input _ _ hj hcn, ← hgd, FlockVBridge.Decode.fill_io _ hAg (hio i hi s un g hW),
    FlockVBridge.Decode.fill_io _ hAg' (hio i hi s un g hW)]
  exact hm

end

end FlockC3.Compose
