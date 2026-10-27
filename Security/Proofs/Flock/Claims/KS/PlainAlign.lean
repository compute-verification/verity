import Proofs.Flock.Claims.KS.ForkAlign
import Proofs.Flock.Claims.KS.PlainForkClash

/-!
# The simulated fork's alignment, from the slot's coin count

`PlainAlignQ` compares the fork at rep `r`'s level-`ℓ + 1` table node against the simulated strategy after `Commit`
(`plainAdvAt`, `Lock.forkE`) with the fork after `plainJ r ℓ` coins of `ForkEvL`. By `Lock.forkE_le_forkC` the
comparison is the alignment alone, since the session's leaves record the cap of the entry they hold:

* `capAtL_of_sessRelL`: an outcome the session's leaf relation relates to an index holding rep `r`'s level-`ℓ + 1`
  entry with cap `d`, and which opens a position at `d` there, records `d` as that rep's level-`ℓ + 1` cap
  (`RecStruct`);
* `sessFork_le`: so at that entry the session's fork event at the slot (`advRSL`'s) is `ForkEvL`'s;
* `SlotRSL`, `advRSL_le_forkC`: where the strategy reaches the slot's node after `j` coins (`Lock.SlotAt` at
  `advRSL`'s lockstep), `advRSL` is at most the fork after `j` coins of `ForkEvL`;
* `opRL_of_nul`, `not_conflictAt_nul`, `opens_of_outN_bind`, `opens_of_outN_sessAfterL`, `slotAt_answerWith_nul`,
  `slotRSL_answerWith_nul`: a dead branch, which answers the openings with `nulOp`, reaches only empty rows, so two of
  its runs never conflict and it meets `Lock.SlotAt` by `nul` at any number of coins (`Lock.forkE_answerWith`);
* `PlainSlotAlign`, `plainForkAlign_of_slot`: so `PlainForkAlign` holds where the simulated strategy reaches each
  plain fork slot after `plainJ r ℓ` coins.
* `PlainSlotQK`, `plainAlignQ_of_slotK`, `PlainSlotAlignK`, `plainForkAlign_of_slotK`: the same at `VK keep`
  (`plainSimAtK keep`, `plain_hdecAtK keep`, `plain_hdecHAtK keep`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A leaf records the cap of the entry it holds** where it opens a position at it: an outcome the session's leaf
relation relates to an index holding rep `r`'s level-`ℓ + 1` entry `(t, d)`, opening a position at `d` at that level,
has `d` as rep `r`'s recorded level-`ℓ + 1` cap, since the opening is a request at level `ℓ + 1`, which `RecStruct`
pairs with the entry's cap. -/
theorem capAtL_of_sessRelL {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {A : Model.Arith 𝔽 K} {Hc : List UInt8 → D} {E : Merkle.Enc (List 𝔽) D} {Lf : Merkle.Leaf (List 𝔽) Sl D Hc}
    {S : Model.Statement} {sch : Accounting.Schedule} {C₀ : Model.Oracle 𝔽} {cap₀ : Model.Cap D}
    {i : IRep 𝔽 D × IRep 𝔽 D} {a b : List (Model.OutCL 𝔽 D Sl)} {r : Fin 2} {ℓ : ℕ} {t : Model.Oracle 𝔽}
    {d : Model.Cap D} (hR : SessRelL A Hc E Lf S sch C₀ cap₀ 0 i a b)
    (hg : Lock.getPair (getRep 0) (getRep 1) i (slotOf r.val (ℓ + 1)) = some (t, d))
    {p : Fin (2 ^ (lvOf sch (ℓ + 1)).logLen)}
    (hp : (opRL Hc E Lf sch r.val (ℓ + 1) d (b.getD 0 dfltOutL) p).isSome) :
    CapAtL r.val ℓ (b.getD 0 dfltOutL) d := by
  obtain ⟨_, _, htop⟩ := hR
  obtain ⟨-, -, -, -, hs₀, hs₁⟩ := htop
  obtain ⟨y, hy, -, -⟩ := opRL_eq_some hp
  have hreq : ∃ req, ((if r.val = 0 then (b.getD 0 dfltOutL).pend.1 else (b.getD 0 dfltOutL).pend.2).reqs.find?
      fun req => req.level == ℓ + 1) = some req := by
    unfold slotOpensL at hy
    split at hy
    · simp at hy
    · exact ⟨_, ‹_›⟩
  obtain ⟨req, hfind⟩ := hreq
  have hmem := List.mem_of_find?_eq_some hfind
  have hlev : req.level = ℓ + 1 := by simpa using List.find?_some hfind
  have key : ∀ (ir : IRep 𝔽 D) (pr : Model.PendL 𝔽 D), RecStruct sch ir pr → req ∈ pr.reqs →
      ir (ℓ + 1) = some (t, d) → pr.caps[ℓ]? = some d := by
    intro ir pr hst hm hi
    have hlt : ℓ + 1 < pr.reqs.length := by
      have h1 : req.level ∈ pr.reqs.map Model.Req.level := List.mem_map_of_mem hm
      rw [hst.1, List.mem_range, hlev] at h1
      exact h1
    obtain ⟨t', d', hi', hc⟩ := hst.2.2.1 (ℓ + 1) (by omega) hlt
    rw [hi] at hi'
    cases hi'
    simpa using hc
  obtain ⟨r, hr⟩ := r
  interval_cases r
  · have hi : i.1 (ℓ + 1) = some (t, d) := by
      have e1 : 2 * (ℓ + 1) % 2 = 0 := by omega
      have e2 : 2 * (ℓ + 1) / 2 = ℓ + 1 := by omega
      simpa [Lock.getPair, getRep, slotOf, e1, e2] using hg
    simp only [CapAtL, ↓reduceIte]
    simp only [↓reduceIte] at hmem
    exact key _ _ hs₀ hmem hi
  · obtain ⟨-, hi⟩ : 1 ≤ 2 * (ℓ + 1) ∧ i.2 (ℓ + 1) = some (t, d) := by
      have e1 : (2 * (ℓ + 1) + 1) % 2 = 1 := by omega
      have e2 : (2 * (ℓ + 1) + 1) / 2 = ℓ + 1 := by omega
      simpa [Lock.getPair, getRep, slotOf, e1, e2] using hg
    simp only [CapAtL, one_ne_zero, ↓reduceIte]
    simp only [one_ne_zero, ↓reduceIte] at hmem
    exact key _ _ hs₁ hmem hi

/-- **At the slot's entry the session's fork event is `ForkEvL`**: two outcomes the session relates to indices holding
rep `r`'s level-`ℓ + 1` entry `(t, d)` and opening a position at `d` there to different rows record `d` as that cap in
both. -/
theorem sessFork_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (S : Model.Statement) (sch : Accounting.Schedule) (C₀ : Model.Oracle 𝔽) (cap₀ : Model.Cap D) (r : Fin 2)
    (ℓ : ℕ) (t : Model.Oracle 𝔽) (d : Model.Cap D) (i₁ : IRep 𝔽 D × IRep 𝔽 D) (a₁ b₁ : List (Model.OutCL 𝔽 D Sl))
    (i₂ : IRep 𝔽 D × IRep 𝔽 D) (a₂ b₂ : List (Model.OutCL 𝔽 D Sl))
    (h₁ : SessRelL A Hc E Lf S sch C₀ cap₀ 0 i₁ a₁ b₁) (h₂ : SessRelL A Hc E Lf S sch C₀ cap₀ 0 i₂ a₂ b₂)
    (hg₁ : Lock.getPair (getRep 0) (getRep 1) i₁ (slotOf r.val (ℓ + 1)) = some (t, d))
    (hg₂ : Lock.getPair (getRep 0) (getRep 1) i₂ (slotOf r.val (ℓ + 1)) = some (t, d)) :
    ind (fun x : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
        (fun outs : List (Model.OutCL 𝔽 D Sl) => opRL Hc E Lf sch r.val (ℓ + 1) d (outs.getD 0 dfltOutL)) x p)
        (b₁, b₂) ≤
      ind (fun x : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) =>
        ForkEvL Hc E Lf sch r.val ℓ x.1 x.2) (b₁, b₂) :=
  ind_le_ind fun ⟨p, hs₁, hs₂, hne⟩ =>
    ⟨d, capAtL_of_sessRelL h₁ hg₁ hs₁, capAtL_of_sessRelL h₂ hg₂ hs₂, p, hs₁, hs₂, hne⟩

/-- **The strategy reaches rep `r`'s level-`ℓ + 1` table node after `j` coins** in the session after `Commit`:
`Lock.SlotAt` at `advRSL`'s chooser, slot, fork event and lockstep. -/
def SlotRSL {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (S : Model.Statement) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D) (r ℓ j : ℕ)
    (σ : Strategy (sessAfterL A Hc E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts [] [] cap₀)) : Prop :=
  Lock.SlotAt (plurSL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r (ℓ + 1))
    (fun d b b' => ind (fun x : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (Model.OutCL 𝔽 D Sl) =>
        opRL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r (ℓ + 1) d (outs.getD 0 dfltOutL)) x p) (b, b'))
    j (lockSessL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts [] [] default cap₀) σ

/-- **`Adv_{r,ℓ+1}` in the session is at most the fork after `j` coins of `ForkEvL`** where the strategy reaches rep
`r`'s level-`ℓ + 1` table node after `j` coins. -/
theorem advRSL_le_forkC {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (S : Model.Statement) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D) (r : Fin 2) (ℓ j : ℕ)
    (σ : Strategy (sessAfterL A Hc E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts [] [] cap₀))
    (h : SlotRSL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts cap₀ r.val ℓ j σ) :
    advRSL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts [] [] cap₀ r.val (ℓ + 1) σ ≤
      forkC (fun a a' => ind (fun p : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) =>
        ForkEvL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ p.1 p.2) (a, a')) j _ σ :=
  Lock.forkE_le_forkC _ _ (fun _ _ => ind_nonneg _ _)
    (fun t d i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hg₁ hg₂ =>
      sessFork_le A Hc E Lf S _ default cap₀ r ℓ t d i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hg₁ hg₂) h

/-- **Empty rows open only the empty row**: where every opening of an output has an empty row, every position its
rep-`r` level-`ℓ` openings open at a cap opens to the empty row. -/
theorem opRL_of_nul {D Sl : Type} (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule) (r ℓ : ℕ) (cap : Model.Cap D)
    {out : Model.OutCL 𝔽 D Sl} (ho : ∀ r ℓ i, (out.opens r ℓ i).1 = []) (p : Fin (2 ^ (lvOf sch ℓ).logLen))
    {v : Fin (2 ^ (lvOf sch ℓ).lanes) → 𝔽} (hv : opRL Hc E Lf sch r ℓ cap out p = some v) :
    v = rowFin (lvOf sch ℓ).lanes [] := by
  unfold opRL opOfL at hv
  obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.1 hv
  obtain ⟨hmem, -⟩ := List.mem_filter.1 (List.mem_of_mem_head? hx)
  unfold slotOpensL at hmem
  split at hmem
  · simp at hmem
  · obtain ⟨pj, -, rfl⟩ := List.mem_map.1 hmem
    simp only [ho]

/-- **Two runs with empty rows never conflict** at a position of a cap. -/
theorem not_conflictAt_nul {D Sl : Type} [Inhabited D] [Inhabited Sl] (Hc : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule) (r ℓ : ℕ)
    (cap : Model.Cap D) {b b' : List (Model.OutCL 𝔽 D Sl)}
    (hb : ∀ r ℓ i, ((b.getD 0 dfltOutL).opens r ℓ i).1 = [])
    (hb' : ∀ r ℓ i, ((b'.getD 0 dfltOutL).opens r ℓ i).1 = []) (p : Fin (2 ^ (lvOf sch ℓ).logLen)) :
    ¬ Rewinding.ConflictAt (fun outs : List (Model.OutCL 𝔽 D Sl) =>
      opRL Hc E Lf sch r ℓ cap (outs.getD 0 dfltOutL)) (b, b') p := by
  rintro ⟨h₁, h₂, hne⟩
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h₁
  obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.1 h₂
  exact hne (by
    rw [hv, hv', opRL_of_nul Hc E Lf sch r ℓ cap hb p hv, opRL_of_nul Hc E Lf sch r ℓ cap hb' p hv'])

/-- **Openings answered with `nulOp` are empty**: an outcome of a game ending in an openings message, reached with
that message `nulOp`, has empty rows. -/
theorem opens_of_outN_bind {D Sl P : Type} [Inhabited Sl]
    (mk : P → Model.Opens 𝔽 D × Model.Salts Sl → Model.OutCL 𝔽 D Sl) (hmk : ∀ p os, (mk p os).opens = os.1)
    {h : Game P} {o : Model.OutCL 𝔽 D Sl}
    (ho : Claims.Bridge.OutN (nulOp (D := D) (Sl := Sl))
      (h.bind fun p => Game.send (Model.Opens 𝔽 D × Model.Salts Sl) fun os => Game.ret (mk p os)) o)
    (r ℓ i : ℕ) : (o.opens r ℓ i).1 = [] := by
  obtain ⟨p, -, hk⟩ := Claims.Bridge.OutN.of_bind ho
  cases hk with
  | send m hm hr =>
    cases hr
    rw [hmk, hm rfl]
    rfl

theorem outN_batch_one {Op A : Type} {nul : Op} {g : Game A} {outs : List A} (d : A)
    (h : Claims.Bridge.OutN nul (batch [g]) outs) : Claims.Bridge.OutN nul g (outs.getD 0 d) := by
  cases Claims.Bridge.OutN.of_batch h with
  | cons hg hr => cases hr; exact hg

/-- **The session after `Commit`, answered with `nulOp`, reaches only empty rows.** -/
theorem opens_of_outN_sessAfterL {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
    [Inhabited Sl] (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (S : Model.Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (cap₀ : Model.Cap D) {outs : List (Model.OutCL 𝔽 D Sl)}
    (h : Claims.Bridge.OutN (nulOp (D := D) (Sl := Sl)) (sessAfterL A Hc E Lf S sch hm ptLocal mPts [] [] cap₀) outs)
    (r ℓ i : ℕ) : ((outs.getD 0 dfltOutL).opens r ℓ i).1 = [] := by
  unfold sessAfterL at h
  cases h with
  | coin pts h =>
    exact opens_of_outN_bind (P := Model.PendL 𝔽 D × Model.PendL 𝔽 D) (fun pend os => ⟨_, cap₀, pend, os.1, os.2⟩)
      (fun _ _ => rfl) (outN_batch_one dfltOutL h) r ℓ i

theorem allInh_sessAfterL {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (S : Model.Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D) :
    AllInh (sessAfterL A Hc E Lf S sch hm ptLocal mPts [] [] cap₀) := by
  unfold sessAfterL
  exact .coin fun _ => AllInh.map _ (AllInh.interleave
    (AllInh.bind (fun _ => .send ⟨nulOp (D := D) (Sl := Sl)⟩ fun _ => .ret _)
      (AllInh.interleave (Model.allInh_repC A 0 S sch hm _) (Model.allInh_repC A 1 S sch hm _))) (.ret []))

/-- **A dead branch meets `SlotAt` by `nul`**: below any lockstep whose compiled game reaches, with the openings
`nulOp`, only outcomes with empty rows, `answerWith nulOp` forks to `0` at the conflict event of any cap's openings,
whatever the chooser and the number of coins. -/
theorem slotAt_answerWith_nul {D Sl T I α : Type} [Inhabited D] [Inhabited Sl] (Hc : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : Accounting.Schedule)
    (plur : Lock.Chooser T (Model.Cap D) (List (Model.OutCL 𝔽 D Sl))) (slot r ℓ j : ℕ)
    {get : I → ℕ → Option (T × Model.Cap D)} {R : I → α → List (Model.OutCL 𝔽 D Sl) → Prop} {i : I} {g : Game α}
    {g' : Game (List (Model.OutCL 𝔽 D Sl))} (L : Lock T (Model.Cap D) get R i g g') (hg : AllInh g')
    (hN : ∀ b, Claims.Bridge.OutN (nulOp (D := D) (Sl := Sl)) g' b →
      ∀ r ℓ i, ((b.getD 0 dfltOutL).opens r ℓ i).1 = []) (σ : Strategy g') :
    Lock.SlotAt plur slot (fun d b b' => ind (fun x : List (Model.OutCL 𝔽 D Sl) × List (Model.OutCL 𝔽 D Sl) =>
      ∃ p, Rewinding.ConflictAt (fun outs : List (Model.OutCL 𝔽 D Sl) =>
        opRL Hc E Lf sch r ℓ d (outs.getD 0 dfltOutL)) x p) (b, b'))
      j L (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)) g' σ) :=
  .nul (Lock.forkE_answerWith plur slot _ L hg (fun d b b' hb hb' =>
    ite_eq_right fun ⟨p, hp⟩ => not_conflictAt_nul Hc E Lf sch r ℓ d (hN b hb) (hN b' hb') p hp) σ)

/-- **A strategy after `Commit` answering the openings with `nulOp` meets `SlotRSL`** at every rep, level and number
of coins. -/
theorem slotRSL_answerWith_nul {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (S : Model.Statement) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D) (r ℓ j : ℕ)
    (σ : Strategy (sessAfterL A Hc E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts [] [] cap₀)) :
    SlotRSL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts cap₀ r ℓ j
      (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)) _ σ) :=
  slotAt_answerWith_nul Hc E Lf _ _ _ r (ℓ + 1) j _ (allInh_sessAfterL A Hc E Lf S _ hm ptLocal mPts cap₀)
    (fun _ hb => opens_of_outN_sessAfterL A Hc E Lf S _ hm ptLocal mPts cap₀ hb) σ

/-- **The simulated strategy reaches the plain fork slot after `plainJ r ℓ` coins**, at tape `u`: `SlotRSL` at rep
`r`'s level-`ℓ + 1` table node against `plainSimAt`. -/
def PlainSlotQ {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDec L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  SlotRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val j
    (plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

theorem plainAlignQ_of_slot {spec : SessionSpec} {st : Stmt} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {I : Exec.Inputs} {d : Json}
    {x : Claims.Msg} {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : PlainHDec L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn} {hw : 2 * spec.mPts ≤ 65536} {Ω : Type}
    {P : Claims.Prover Ω} {ω : Ω} {C : Fin (wordLen false x) → List 𝔽} {r : Fin 2}
    {ℓ : Fin (l₀ :: l₁ :: rest).length} {j : ℕ} {u : plainTape x}
    (h : PlainSlotQ L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u) :
    PlainAlignQ L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u :=
  advRSL_le_forkC _ _ _ _ _ m k0 l₀ l₁ rest ext hk0 hm _ _ _ r ℓ.val j _ h

/-- **The simulated strategy reaches each plain fork slot after `plainJ r ℓ` coins**: `PlainForkAlign` with
`PlainSlotQ` for `PlainAlignQ`, at every plain statement in scope, its one table's `fast100` schedule and the plain
decoders where the outputs are public and where they are hidden under hm96 leaves. -/
def PlainSlotAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false, ∀ C r ℓ u,
      PlainSlotQ L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r ℓ
        (plainJ (tables_head_one h) hsch r ℓ).val u) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"), ∀ C r ℓ u,
      PlainSlotQ L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecHAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r ℓ
        (plainJ (tables_head_one h) hsch r ℓ).val u)

theorem plainForkAlign_of_slot {x : Claims.Msg} (h : PlainSlotAlign x) : PlainForkAlign x :=
  fun s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω =>
    ⟨fun hh C r ℓ u => plainAlignQ_of_slot
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).1 hh C r ℓ u),
      fun hh hl C r ℓ u => plainAlignQ_of_slot
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).2 hh hl C r ℓ u)⟩

/-! ## At `VK keep` -/

/-- `PlainSlotQ` at `VK keep`. -/
def PlainSlotQK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  SlotRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val j
    (plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- `plainAlignQ_of_slot` at `VK keep`. -/
theorem plainAlignQ_of_slotK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {I : Exec.Inputs} {d : Json}
    {x : Claims.Msg} {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : PlainHDecK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn} {hw : 2 * spec.mPts ≤ 65536} {Ω : Type}
    {P : Claims.Prover Ω} {ω : Ω} {C : Fin (wordLen false x) → List 𝔽} {r : Fin 2}
    {ℓ : Fin (l₀ :: l₁ :: rest).length} {j : ℕ} {u : plainTape x}
    (h : PlainSlotQK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u) :
    PlainAlignQK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u :=
  advRSL_le_forkC _ _ _ _ _ m k0 l₀ l₁ rest ext hk0 hm _ _ _ r ℓ.val j _ h

/-- `PlainSlotAlign` at `VK keep`. -/
def PlainSlotAlignK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false, ∀ C r ℓ u,
      PlainSlotQK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r ℓ
        (plainJ (x := x) (tables_head_one h) hsch r ℓ).val u) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"), ∀ C r ℓ u,
      PlainSlotQK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r ℓ
        (plainJ (x := x) (tables_head_one h) hsch r ℓ).val u)

/-- `plainForkAlign_of_slot` at `VK keep`. -/
theorem plainForkAlign_of_slotK (keep : Bool → Bool) {x : Claims.Msg} (h : PlainSlotAlignK keep x) : PlainForkAlignK keep x :=
  fun s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω =>
    ⟨fun hh C r ℓ u => plainAlignQ_of_slotK keep
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).1 hh C r ℓ u),
      fun hh hl C r ℓ u => plainAlignQ_of_slotK keep
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).2 hh hl C r ℓ u)⟩

end FlockDeploy.KS
