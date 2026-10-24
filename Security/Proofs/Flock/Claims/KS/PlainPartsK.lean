import Proofs.Flock.Claims.KS.PlainParts

/-!
# The plain chain from the concrete simulator's parts, at `VK keep`

`PlainStep`'s abort and `PlainParts`' chain at every `keep`: the simulator `plainSimK keep P`, the runs of `VK keep` and
the verdict at the draw `plainVerdictAtK keep`. At `keep := id` the statements read `VK id`, which is `V false`
(`VK_id`), but `plainVerdictAtK id` decodes through `decodeWithK id` and compares rounds up to their digests, so the
originals stay beside these rather than becoming their wrappers.

* `not_plainVerdictAtK_nil`, `PlainSimAbortAtK`, `plainSimAbortAtK`: the abort at `plainVerdictAtK keep`.
* `plainDecQK`, `PlainClashQK`, `PlainSimClashAtK`, `plainAdvK`, `PlainSimForkLeK`: the two parts at `keep`.
* `plainSimWrPAtD_of_partsK`, `plainSimWD_of_partsK`: so `PlainSimWDK keep` holds from `PlainSimClashAtK keep` and
  `PlainSimForkLeK keep` alone.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

/-! ## The abort -/

/-- **The verdict at the draw refuses an outcome without proof files, at every `keep`** (`not_plainVerdictAt_nil`'s
proof at `wrPDK`). -/
theorem not_plainVerdictAtK_nil (keep : Bool → Bool) (I : Discharge.Exec.Inputs) (d : Lean.Json) {N mPts : ℕ}
    (C : Fin N → List 𝔽) (rootB₀ : Array ByteArray) {o : LiveOutJ 1 mPts} (h : o.proofs = #[]) :
    ¬ plainVerdictAtK keep I d C rootB₀ o := by
  rintro ⟨hA, -, -⟩
  have hp := (Claims.Bridge.PlainDraw.recordsLiveAtDK_wrPDK keep I d o hA).1
  rw [h] at hp
  generalize Claims.Bridge.PlainDraw.wrPDK keep I d o = t at hA hp
  rcases t with ⟨record, proofs⟩
  subst hp
  exact Claims.Bridge.PlainDraw.not_acceptsAtDK_nil keep I (some d) record hA

/-- **The abort at `VK keep`** (`PlainSimAbortAt` at `plainSimK keep` and `plainVerdictAtK keep`). -/
def PlainSimAbortAtK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (I : Discharge.Exec.Inputs)
    (d : Lean.Json) {N : ℕ} (C : Fin N → List 𝔽) (rootB₀ : Array ByteArray),
    PlainSimAbortK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
      (plainSimK keep P x spec st) (plainΦ x spec st) (plainVerdictAtK keep I d C rootB₀)

/-- **The abort holds at every statement, at every `keep`** (`plainSim_proofs_nilK`, `not_plainVerdictAtK_nil`). -/
theorem plainSimAbortAtK (keep : Bool → Bool) (x : Claims.Msg) : PlainSimAbortAtK keep x :=
  fun _ hs hin _ _ _ hset st _ _ P ω I d _ C rootB₀ u r' hna =>
    not_plainVerdictAtK_nil keep I d C rootB₀
      (plainSim_proofs_nilK keep P ω st hset _ (plain_mPts hs hin hset).2.1 u r' hna)

/-! ## The two parts -/

/-- **A live outcome decoded at the verdict at the draw, with a property `Q` of table 0's output, at `keep`**
(`plainDecQ` at `plainVerdictAtK keep`). -/
def plainDecQK (keep : Bool → Bool) {mPts : ℕ} (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) (T : TabSpec mPts)
    (I : Exec.Inputs) (d : Json) {M : ℕ} (C : Fin M → List 𝔽) (rootB₀ : Array ByteArray)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) (o : LiveOutJ 1 mPts) :
    Prop :=
  ∃ a, Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
    (plainVerdictAtK keep I d C rootB₀) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)) o a ∧
    Q (a.getD 0 dfltOutL)

/-- **`Q` at the decoded live outcome gives `P`'s accepting runs of `VK keep` a raw clash** (`PlainClashQ` at
`plainSimK keep`). -/
def PlainClashQK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec}
    (st : Stmt) (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536),
    plainDecQK keep Q T I d C (plainRoot P x ω) opn (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec)
      (plainW spec hw) sha512 (plainSimK keep P x spec st) x (ω, u) r') →
    (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).out = .acc →
    PlainRawClashLK keep P x ω (plainΦ x spec st (u, r'))

/-- **The live self-clash's clash, at `keep`** (`PlainSimClashAt` at `PlainClashQK keep`). -/
def PlainSimClashAtK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (d : Json) (_ : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Claims.Prover Ω) (ω : Ω),
    (ins.read.I.tags.hiddenOutputs = false →
      PlainClashQK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)) ∧
    (ins.read.I.tags.hiddenOutputs = true → ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" →
      PlainClashQK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))

open Classical in
/-- **The strategies' fork advantage at the concrete simulator, at `keep`** (`plainAdv` at `plainSimK keep`,
`plainSim_rootK keep` and `plainVerdictAtK keep`). -/
noncomputable def plainAdvK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) : ℝ :=
  (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
      k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
      (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep I d C (plainRoot P x ω)) opn
        (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
          (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u))) /
    Fintype.card (plainTape x)

/-- **The simulated fork bounds the strategies' fork advantage, at `keep`** (`PlainSimForkLe` at `plainAdvK keep`,
the decoders `plain_hdecAtK keep` and `plain_hdecHAtK keep`, and `VK keep`'s simulated fork). -/
def PlainSimForkLeK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Claims.Prover Ω) (ω : Ω),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false, ∀ C : Fin (wordLen false x) → List 𝔽,
      (∀ p, wide false x (C p)) → ∀ r ℓ,
        plainAdvK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
            (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
            (plain_hdecAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1
            P ω C r ℓ ≤
          Claims.Flock.simForkPr sha512 (VK keep) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (x := x) (tables_head_one h) hsch r ℓ).val) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"),
      ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) → ∀ r ℓ,
        plainAdvK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
            (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
            (plain_hdecHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1
            P ω C r ℓ ≤
          Claims.Flock.simForkPr sha512 (VK keep) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (x := x) (tables_head_one h) hsch r ℓ).val)

/-! ## The chain -/

open Classical in
/-- **The concrete simulator at the verdict at the draw, from its clash and its fork bound, at `keep`**
(`plainSimWrPAtD_of_parts`'s proof at `plainSimK keep`). -/
theorem plainSimWrPAtD_of_partsK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt}
    (h : tables false x = [(spec, st)]) {d : Json} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep ins.read.I d C rootB₀) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hc : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω),
      PlainClashQK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d opn (SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
    (hf : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
      (∀ p, wide false x (C p)) → ∀ r ℓ,
        plainAdvK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x opn hdec (plain_mPts hs hin hset).1 P ω
            C r ℓ ≤
          Claims.Flock.simForkPr sha512 (VK keep) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (x := x) (tables_head_one h) hsch r ℓ).val) :
    PlainSimWrPAtDK keep h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec := by
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

/-- **The concrete simulator at a plain statement, from its clash and its fork bound, at `keep`**: `PlainSimWDK keep`
from `PlainSimClashAtK keep` and `PlainSimForkLeK keep`. -/
theorem plainSimWD_of_partsK (keep : Bool → Bool) {x : Claims.Msg} (hc : PlainSimClashAtK keep x)
    (hf : PlainSimForkLeK keep x) : PlainSimWDK keep x :=
  fun s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay =>
    ⟨fun hh => plainSimWrPAtD_of_partsK keep hs hin hset h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _
        (fun Ω _ _ P ω => (hc s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).1 hh)
        (fun Ω _ _ P ω =>
          (hf s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).1 hh),
      fun hh hl => plainSimWrPAtD_of_partsK keep hs hin hset h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _
        (fun Ω _ _ P ω => (hc s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).2 hh hl)
        (fun Ω _ _ P ω =>
          (hf s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).2 hh hl)⟩

end FlockDeploy.KS
