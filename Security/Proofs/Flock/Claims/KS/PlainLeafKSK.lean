import Proofs.Flock.Claims.KS.PlainLeafKS

/-!
# The plain-leaf simulator and the slack's fit at `VK keep`

`PlainLeafKS`'s run-level half over the verifier `VK keep`, and the facts the table bound at `deployPlainK keep` reads
beside it:

* `goodTable_withV`, `deployPlainK_k`, `deployPlainK_weight`, `goodTable_plainK`, `VK_k_pos`: `deployPlainK keep rel`
  is `deployPlain rel` off its verifier. They are stated over `{ D with V := W }` at a free `D` and `W`, so no
  elaboration compares `VK keep` with `V false`, which unfolds both verifiers' steps.
* `slackEsc_fitsFork_plainK`: `slackEsc_fitsFork_plain` at `VK keep`.
* `PlainSimWrPAtDUK`, `plainSimWrPAtD_of_partsUK`, `plainSimWrPAtDU_atK`, `plainSimWD_allK`: the concrete simulator at
  `plainVerdictAtK keep`, at plain SHA-512 leaves and at every statement.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Record

open Flock

/-- A record's table check doesn't read its verifier. -/
theorem goodTable_withV {Col T : Type} [DecidableEq Col] [Zero Col] (D : Claims.Flock.Deploy Coin Col T)
    (W : Claims.Verifier Coin) (N : Claims.Msg → ℕ) (x : Claims.Msg) (C : Fin (N x) → Col) :
    Claims.Flock.GoodTable ({ D with V := W } : Claims.Flock.Deploy Coin Col T) N x C ↔
      Claims.Flock.GoodTable D N x C :=
  Iff.rfl

theorem deploy_withV_k {Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (W : Claims.Verifier Coin) :
    ({ D with V := W } : Claims.Flock.Deploy Coin Col T).k = D.k :=
  rfl

theorem deploy_withV_weight {Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (W : Claims.Verifier Coin) :
    ({ D with V := W } : Claims.Flock.Deploy Coin Col T).weight = D.weight :=
  rfl

theorem deployPlainK_k (keep : Bool → Bool) (rel : Readable.Parsed → Readable.Relation) :
    (deployPlainK keep rel).k = (deployPlain rel).k :=
  deploy_withV_k (deployPlain rel) (VK keep)

theorem deployPlainK_weight (keep : Bool → Bool) (rel : Readable.Parsed → Readable.Relation) :
    (deployPlainK keep rel).weight = (deployPlain rel).weight :=
  deploy_withV_weight (deployPlain rel) (VK keep)

theorem goodTable_plainK (keep : Bool → Bool) (rel : Readable.Parsed → Readable.Relation) (N : Claims.Msg → ℕ)
    (x : Claims.Msg) (C : Fin (N x) → List 𝔽) :
    Claims.Flock.GoodTable (deployPlainK keep rel) N x C ↔ Claims.Flock.GoodTable (deployPlain rel) N x C :=
  goodTable_withV (deployPlain rel) (VK keep) N x C

theorem VK_k_pos (keep : Bool → Bool) (x : Claims.Msg) : 0 < (VK keep).k x := by
  rw [VK_k]
  exact V_k_pos false x

/-- `plain_k_le` at `deployPlainK keep`. -/
theorem plain_k_leK (keep : Bool → Bool) (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    ((deployPlainK keep rel).k x : ℝ) ≤ 2 * (deployPlainK keep rel).weight x := by
  rw [deployPlainK_k, deployPlainK_weight]
  exact plain_k_le rel x

/-- `slackEsc_fitsFork_plain` at `VK keep`. -/
theorem slackEsc_fitsFork_plainK (keep : Bool → Bool) (rel : Readable.Parsed → Readable.Relation) {Ω : Type}
    (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables false x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin ((VK keep).k x) → Coin) →
      (Fin ((VK keep).k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((VK keep).k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (VK keep) x ω s).vLog ++
        (Claims.run sha512 P (VK keep) x ω (Claims.forkCoins (plainJ (x := x) h0 hsch r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (VK keep) x ω s).vLog)
    (w : Fin 2 → Fin sch.levels.length → Fin ((VK keep).k x) → ℝ)
    (G : Fin ((VK keep).k x) → Ω → (Fin ((VK keep).k x) → Coin) →
      (Fin ((VK keep).k x) → Coin) → Prop)
    (tiedG : ∀ i ω s s', G i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (VK keep) x ω s).vLog ++
        (Claims.run sha512 P (VK keep) x ω (Claims.forkCoins i s s')).vLog))
    (hfit : ∑ i, Claims.Flock.slackC sch w i ≤
      2 * ((2 : ℝ) ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries)) :
    (Claims.Flock.slackEsc sha512 (VK keep) P x sch (plainJ (x := x) h0 hsch) ev ev₁ tied tied₁ w G tiedG).FitsFork
      ((deployPlainK keep rel).weight x) ((deployPlainK keep rel).k x) := by
  rw [deployPlainK_weight, deployPlainK_k]
  refine ⟨2, fun r => ?_, ?_, ?_⟩
  · show (Finset.univ.filter fun i : (Fin 2 × Fin sch.levels.length) ⊕ Fin ((VK keep).k x) =>
        (Sum.elim (fun a : Fin 2 × Fin sch.levels.length => plainJ (x := x) h0 hsch a.1 a.2) id i :
          Fin ((VK keep).k x)) = r).card + (if r.val = 0 then 1 else 0) ≤ 2
    rw [Finset.card_filter, Fintype.sum_sum_type]
    have h1 : (∑ a : Fin 2 × Fin sch.levels.length,
        if id (α := Fin ((VK keep).k x)) (plainJ (x := x) h0 hsch a.1 a.2) = r then 1 else 0) ≤
          if r.val = 0 then 0 else 1 := by
      rw [← Finset.card_filter]
      split_ifs with hr
      · rw [Nat.le_zero, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
        intro a _ he
        exact plainJ_ne_zero h0 hsch a.1 a.2 ((congrArg Fin.val he).trans hr)
      · exact Finset.card_le_one.2 fun a ha b hb => by
          simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
          obtain ⟨e1, e2⟩ := plainJ_inj h0 hsch _ _ _ _ (ha.trans hb.symm)
          exact Prod.ext e1 e2
    have h2 : (∑ i : Fin ((VK keep).k x), if i = r then 1 else 0) = 1 := by
      rw [Finset.sum_ite_eq']
      simp
    show (∑ a : Fin 2 × Fin sch.levels.length,
        if id (α := Fin ((VK keep).k x)) (plainJ (x := x) h0 hsch a.1 a.2) = r then 1 else 0) +
      (∑ i : Fin ((VK keep).k x), if i = r then 1 else 0) + (if r.val = 0 then 1 else 0) ≤ 2
    rw [h2]
    split_ifs at h1 ⊢ <;> omega
  · have hroom := plain_fork_room rel x h0 hsch
    have hs : ∑ i : (Fin 2 × Fin sch.levels.length) ⊕ Fin ((VK keep).k x),
        Sum.elim (fun a => Claims.Flock.cOf sch (a.2.val + 1)) (Claims.Flock.slackC sch w) i =
        ∑ a : Fin 2 × Fin sch.levels.length, Claims.Flock.cOf sch (a.2.val + 1) +
          ∑ i, Claims.Flock.slackC sch w i := by
      rw [Fintype.sum_sum_type]
      rfl
    show ((2 : ℕ) : ℝ) * ((deployPlain rel).k x + ∑ i : (Fin 2 × Fin sch.levels.length) ⊕
      Fin ((VK keep).k x), Sum.elim (fun a => Claims.Flock.cOf sch (a.2.val + 1))
        (Claims.Flock.slackC sch w) i) ≤ 2 * (deployPlain rel).weight x
    rw [hs]
    push_cast at hroom ⊢
    linarith
  · show ∑ _l : Unit, (1 : ℝ) ≤ (VK keep).k x
    rw [Fintype.sum_unique]
    exact_mod_cast VK_k_pos keep x

end Record

section Chain

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

open Classical in
/-- `PlainSimWrPAtDU` at `VK keep`. -/
def PlainSimWrPAtDUK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (δ : ℕ → ℝ) (_ : ∀ t, 0 ≤ δ t)
      (_ : ∑ i, Claims.Flock.slackC ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (fun r ℓ (i : Fin ((V false).k x)) =>
            Claims.Flock.slackW ((V false).k x) δ (plainJ (x := x) (tables_head_one h) hsch r ℓ).val i.val) i ≤
        2 * ((2 : ℝ) ^ l₀.logLen * l₀.queries)),
    ∀ ω : Ω, ∃ (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 spec.mPts) (τ : Type)
      (_ : Fintype τ) (_ : Nonempty τ) (P' : Claims.Prover (Ω × τ))
      (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin)
      (_ : ∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E)
      (hσ : ∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω),
      PlainSimRunK keep P x ω spec.mPts N dg w P' Φ ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
        PlainClashRunK keep P x ω spec.mPts N dg w P' Φ
          (LiveSelfClash Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (plainVerdictAtK keep I d C (plainRoot P x ω)) opn
            (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
            (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
              ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))) ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) → ∀ r ℓ,
        (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m
            k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
            (simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
              (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep I d C (plainRoot P x ω)) opn
              dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
              (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
              (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))) /
            Fintype.card τ ≤
          (Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) =>
            plainForkEvK keep h hsch P r ℓ ω s.1 s.2) +
            Claims.Flock.forkSlack sha512 (VK keep) P x δ (plainJ (x := x) (tables_head_one h) hsch r ℓ).val ω)

open Classical in
/-- `plainSimWrPAtD_of_partsU` at `VK keep`. -/
theorem plainSimWrPAtD_of_partsUK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt}
    (h : tables false x = [(spec, st)]) {d : Json} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep ins.read.I d C rootB₀) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hc : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω),
      PlainClashQUK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d opn (SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
    (hf : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
      (∀ p, wide false x (C p)) → ∀ r ℓ,
        plainAdvUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x opn hdec (plain_mPts hs hin hset).1 P ω C r
            ℓ ≤
          Claims.Flock.simForkPr sha512 (VK keep) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (x := x) (tables_head_one h) hsch r ℓ).val) :
    PlainSimWrPAtDUK keep h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec := by
  intro Ω _ _ P
  obtain ⟨hw, hm1, hp⟩ := plain_mPts hs hin hset
  have hL := plainN_add hset
  refine ⟨Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 (plainNOf x spec st),
    fun t => Claims.Flock.FnFork.δN_nonneg _ t,
    plain_slack_fits (tables_head_one h) hsch _ fun t _ => plainNOf_pos x spec st t, fun ω => ?_⟩
  refine ⟨plainN, plainDg spec, plainW spec hw, plainTape x, inferInstance, inferInstance, plainSimK keep P x spec st,
    plainΦ x spec st, Pr_plainΦ hL, plainSim_rootK keep P x spec st ω hw, plainSimRunK keep P ω hset h hw hm1 hp,
    fun C _ => ?_, fun C hC r ℓ => ?_⟩
  · refine plainClashRun_of_rawLK keep (plainRawClashRunL_of_abortK keep (fun u r' hna hq => ?_) (hc Ω P ω C))
    obtain ⟨a, ha, -⟩ := hq
    exact plainSimAbortAtK keep x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1
  · exact plain_fork_le_fnAtK keep h hsch P ω hL (plainNOf x spec st) (Claims.Flock.FnFork.prefixKey_histKey _) r ℓ
      ((hf Ω P ω C hC r ℓ).trans_eq (by rw [plainΦ_eq hL]; rfl))

/-- `plainSimWrPAtDU_at` at `VK keep`. -/
theorem plainSimWrPAtDU_atK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) :
    PlainSimWrPAtDUK keep h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (plain_hdecHUAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) :=
  plainSimWrPAtD_of_partsUK keep hs hin hset h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _
    (fun _ _ _ P ω => plainSimClashHUK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay P ω)
    (fun _ _ _ P ω C _ r ℓ => plainAdv_leUK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay P ω C r ℓ)

/-- `plainSimWD_all` at `VK keep`. -/
theorem plainSimWD_allK (keep : Bool → Bool) (x : Claims.Msg) : PlainSimWDK keep x :=
  plainSimWD_of_partsK keep (plainSimClashAt_of_laterK keep (plainSimLaterClashAtK keep x))
    (plainSimForkLe_ofK keep (plainForkAlign_of_slotK keep (plainSlotAlign_of_schedK keep (plainSchedAlignK keep x))) (plainForkClashK keep x))

end Chain

end FlockDeploy.KS
