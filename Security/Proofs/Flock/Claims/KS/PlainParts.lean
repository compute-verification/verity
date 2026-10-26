import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.KS.PlainDrawSim
import Proofs.Flock.Claims.KS.PlainSlackFit
import Proofs.Flock.Claims.KS.PlainClashL
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The plain chain from the concrete simulator's parts

The plain table bound asks, at each plain statement some run accepts, for a simulator of each prover at the verdict
at the draw (`PlainSimWD`). The concrete simulator `plainSim P` at the coins `plainΦ x` has its uniform law
(`Pr_plainΦ`), its roots (`plainSim_root`), the simulation (`plainSimRun`), the abort (`plainSimAbortAt`) and, at the
coupling's chances, the slack's fit (`plain_slack_fits`) and the fork's slack (`plain_fork_le_fnAt`). What it does not
have yet is two parts, each a named Prop here:

* `PlainSimClashAt`: a live outcome decoded at the verdict at the draw with a self-clash, where `P`'s run accepts,
  gives that run a raw clash at some level or a collision (`PlainRawClashL`);
* `PlainSimForkLe`: the strategies' fork advantage at each plain fork round is at most the simulated fork there.

* `plainDecQ`, `PlainClashQ`: the live outcome decoded at the verdict at the draw with a property `Q` of the table's
  output, and `Q` giving `P`'s accepting runs a raw clash; `LiveSelfClash` is `plainDecQ` at `SelfClashL`.
* `plainAdv`: the strategies' fork advantage `PlainSimWrPAtD` bounds, at the concrete simulator.
* `plainRawClashRunL_of_abort`, `plainSimWrPAtD_of_parts`, `plainSimWD_of_parts`: so `PlainSimWD` holds from
  `PlainSimClashAt` and `PlainSimForkLe` alone.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

/-- **A raw clash in `P`'s run at coins `c`**: two different columns its reps open at one level-0 position, two
openings of one position to different rows against one root at a later level, or a collision among the verifier's
calls (`PlainRawClashRunL`'s outcome). -/
def PlainRawClashL {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (c : Fin ((V false).k x) → Coin) :
    Prop :=
  (∃ (q : ℕ) (a b : List 𝔽), (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω c).seen ∧
      (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω c).seen ∧ a ≠ b) ∨
    (∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
      PlainLevelClash ms lv lanes (Claims.run sha512 P (V false) x ω c).vLog) ∨
    Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω c).vLog

/-- **A raw clash in `P`'s run at coins `c`**: two different columns its reps open at one level-0 position, two
openings of one position to different rows against one root at a later level, or a collision among the verifier's
calls (`PlainRawClashRunL`'s outcome). -/
def PlainRawClashLK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (c : Fin ((V false).k x) → Coin) :
    Prop :=
  (∃ (q : ℕ) (a b : List 𝔽), (q, a) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω c).seen ∧
      (q, b) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω c).seen ∧ a ≠ b) ∨
    (∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
      PlainLevelClash ms lv lanes (Claims.run sha512 P (VK keep) x ω c).vLog) ∨
    Claims.TiedColl sha512 (Claims.run sha512 P (VK keep) x ω c).vLog

/-- **The raw clash at any level from the abort and the accepting runs' clash**: `PlainRawClashRunL` where `P'`
aborts on rejected runs and, at the accepting ones, the live event gives a raw clash. -/
theorem plainRawClashRunL_of_abortK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (habort : PlainSimAbortK keep P x ω mPts N dg w P' Φ pL)
    (hclash : ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
      (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).out = .acc → PlainRawClashLK keep P x ω (Φ (u, r'))) :
    PlainRawClashRunLK keep P x ω mPts N dg w P' Φ pL := fun u r' hp =>
  ⟨plain_acc_of_abortK keep habort hp, hclash u r' hp (plain_acc_of_abortK keep habort hp)⟩

/-- `plainRawClashRunL_of_abortK` at the deployed plain verifier (`keep := id`). -/
theorem plainRawClashRunL_of_abort {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (habort : PlainSimAbort P x ω mPts N dg w P' Φ pL)
    (hclash : ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
      (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc → PlainRawClashL P x ω (Φ (u, r'))) :
    PlainRawClashRunL P x ω mPts N dg w P' Φ pL :=
  plainRawClashRunL_of_abortK id (Ω := Ω) (P := P) (x := x) (ω := ω) (mPts := mPts) (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (pL := pL) (habort := habort) (hclash := hclash)

/-- **A live outcome decoded at the verdict at the draw, with a property `Q` of table 0's output**: the verdict
`plainVerdictAt I d C rootB₀` holds, the reps' decoded tapes play to pendings, and the output at them and the
openings `opn` has `Q`. At `Q := SelfClashL` it is `LiveSelfClash` at the plain decoders. -/
def plainDecQ {mPts : ℕ} (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) (T : TabSpec mPts) (I : Exec.Inputs)
    (d : Json) {M : ℕ} (C : Fin M → List 𝔽) (rootB₀ : Array ByteArray)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) (o : LiveOutJ 1 mPts) :
    Prop :=
  ∃ a, Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
    (plainVerdictAt I d C rootB₀) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)) o a ∧
    Q (a.getD 0 dfltOutL)

/-- **`Q` at the decoded live outcome gives `P`'s accepting runs a raw clash**: at the concrete simulator, wherever
`plainDecQ Q` holds at `plainSim P`'s live outcome and `V false` accepts `P` at `plainΦ x`, that run has a raw clash
(`PlainRawClashL`). -/
def PlainClashQ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536),
    plainDecQ Q T I d C (plainRoot P x ω) opn (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec)
      (plainW spec hw) sha512 (plainSim P x spec st) x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).out = .acc →
    PlainRawClashL P x ω (plainΦ x spec st (u, r'))

/-- **The live self-clash's clash**: at every plain statement in scope and its one table's `fast100` schedule, at the
plain decoders' openings where the outputs are public (`plainOpn`) and where they are hidden under hm96 leaves
(`plainOpnH`), a live self-clash at the concrete simulator's outcome where `P`'s run accepts gives that run a raw
clash (`PlainClashQ` at `SelfClashL`). -/
def PlainSimClashAt (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (d : Json) (_ : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Claims.Prover Ω) (ω : Ω),
    (ins.read.I.tags.hiddenOutputs = false →
      PlainClashQ P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)) ∧
    (ins.read.I.tags.hiddenOutputs = true → ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" →
      PlainClashQ P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))

open Classical in
/-- **The strategies' fork advantage at the concrete simulator**: `PlainSimWrPAtD`'s fork clause's left side at
`plainSim P`, its tape `plainTape x`, round digest `plainDg`, wire `plainW` and roots `plainSim_root`, at rep `r`'s
level `ℓ + 1`. -/
noncomputable def plainAdv {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) : ℝ :=
  (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
      k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
      (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt I d C (plainRoot P x ω)) opn
        (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
          (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u))) /
    Fintype.card (plainTape x)

/-- **The simulated fork bounds the strategies' fork advantage**: at every plain statement in scope and its one
table's `fast100` schedule, at the plain decoders where the outputs are public (`plainOpn`, `plain_hdecAt`) and where
they are hidden under hm96 leaves (`plainOpnH`, `plain_hdecHAt`), the concrete simulator's fork advantage at each
plain fork round
(`plainAdv`) is at most its simulated fork there, at the live fork `forkN` (`Flock.simForkPr` at `plainΦ x`). -/
def PlainSimForkLe (x : Claims.Msg) : Prop :=
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
        plainAdv L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
            (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
            (plain_hdecAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r
            ℓ ≤
          Claims.Flock.simForkPr sha512 (V false) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (tables_head_one h) hsch r ℓ).val) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"),
      ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) → ∀ r ℓ,
        plainAdv L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
            (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
            (plain_hdecHAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r
            ℓ ≤
          Claims.Flock.simForkPr sha512 (V false) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (tables_head_one h) hsch r ℓ).val)

open Classical in
/-- **The concrete simulator at the verdict at the draw, from its clash and its fork bound**: `PlainSimWrPAtD` at any
openings `opn` and decoders `hdec`, with `plainSim P`, `plainΦ x` and the coupling's chances
`FnFork.δN GF128 65536 (plainNOf x spec st)`, where its live self-clashes give `P`'s accepting runs a raw clash and its
fork advantage is at most its simulated fork. -/
theorem plainSimWrPAtD_of_parts {x : Claims.Msg} {s : Claims.Flock.Stmt}
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
        (plainVerdictAt ins.read.I d C rootB₀) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hc : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω),
      PlainClashQ P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d opn (SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
    (hf : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
      (∀ p, wide false x (C p)) → ∀ r ℓ,
        plainAdv L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x opn hdec (plain_mPts hs hin hset).1 P ω C r
            ℓ ≤
          Claims.Flock.simForkPr sha512 (V false) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (tables_head_one h) hsch r ℓ).val) :
    PlainSimWrPAtD h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec := by
  intro Ω _ _ P
  obtain ⟨hw, hm1, hp⟩ := plain_mPts hs hin hset
  have hL := plainN_add hset
  refine ⟨Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 (plainNOf x spec st),
    fun t => Claims.Flock.FnFork.δN_nonneg _ t,
    plain_slack_fits (tables_head_one h) hsch _ fun t _ => plainNOf_pos x spec st t, fun ω => ?_⟩
  refine ⟨plainN, plainDg spec, plainW spec hw, plainTape x, inferInstance, inferInstance, plainSim P x spec st,
    plainΦ x spec st, Pr_plainΦ hL, plainSim_root P x spec st ω hw, plainSimRun P ω hset h hw hm1 hp,
    fun C _ => ?_, fun C hC r ℓ => ?_⟩
  · refine plainClashRun_of_rawL (plainRawClashRunL_of_abort (fun u r' hna hq => ?_) (hc Ω P ω C))
    obtain ⟨a, ha, -⟩ := hq
    exact plainSimAbortAt x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1
  · exact plain_fork_le_fnAt h hsch P ω hL (plainNOf x spec st) (Claims.Flock.FnFork.prefixKey_histKey _) r ℓ
      ((hf Ω P ω C hC r ℓ).trans_eq (by rw [plainΦ_eq hL]; rfl))

/-- **The concrete simulator at a plain statement, from its clash and its fork bound**: `PlainSimWD` from
`PlainSimClashAt` and `PlainSimForkLe`, by `plainSimWrPAtD_of_parts` at the public and the hidden decoders. -/
theorem plainSimWD_of_parts {x : Claims.Msg} (hc : PlainSimClashAt x) (hf : PlainSimForkLe x) : PlainSimWD x :=
  fun s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay =>
    ⟨fun hh => plainSimWrPAtD_of_parts hs hin hset h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _
        (fun Ω _ _ P ω => (hc s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).1 hh)
        (fun Ω _ _ P ω => (hf s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).1 hh),
      fun hh hl => plainSimWrPAtD_of_parts hs hin hset h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _
        (fun Ω _ _ P ω => (hc s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).2 hh hl)
        (fun Ω _ _ P ω =>
          (hf s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω).2 hh hl)⟩

end FlockDeploy.KS
