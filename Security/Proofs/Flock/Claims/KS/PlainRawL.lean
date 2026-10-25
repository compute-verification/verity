import Proofs.Flock.Claims.KS.PlainOffLive
import Proofs.Flock.Claims.KS.PlainLeafKS
import Proofs.Flock.Claims.KS.SessionRaw
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The plain table bound at one tape, each slot at read sets, at any leaf

The simulator's chain (`plain_le_sessionsE`, `plain_le_sessionsEU`, `plain_tableBoundθAt_of_simD(U)`) bounds each
plain fork slot by `Lock.leafE_off_le`'s root. Before that step each slot's term is its per-position off mass at its
node (`session_sound_of_table_accL_nodeOff`), and averaged over the simulator's tape that is one run's: the chance
over the deployed coins of reading anything but a table that reads their first `forkRound` coins
(`plainOffU_le_rd`'s chain). Here that chain is written once, at any model leaf `Lf` and decoders `hdec` at it, so
that it serves the public and hm96 cases at `leaf512` and the plain-leaf case at `leafPlain`:

* `plainOffL_le_rd`: one slot's term, averaged over the tape, at read sets `rdD` holding every decoded live opening
  (`plainOffAtU_le_live` then `plainOffU_le_rd`, written at `Lf`: `slotOffRSL_le_offC`, `offC_simAt_le`,
  `offC_liveStrat_le`).
* `plain_le_sessionsRawL`: the table bound at one tape from salted sessions after `Commit`, each slot's term raw
  (`plain_le_sessionsEU`'s proof at `session_sound_of_table_accL_nodeOff`).
* `plain_accPr_le_rdL`: at the concrete simulator, a run accepts with probability at most `ε_table`, the one-run
  event's probability, each slot's read-set term and the chance of opening off the table.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- Averaging a double sum of terms over `τ`: the double sum of the averages. -/
theorem mean_sum_sum {τ ι κ : Type} [Fintype τ] (s : Finset ι) (s' : Finset κ) (f : ι → κ → τ → ℝ) :
    (∑ t, ∑ i ∈ s, ∑ k ∈ s', f i k t) / Fintype.card τ = ∑ i ∈ s, ∑ k ∈ s', (∑ t, f i k t) / Fintype.card τ := by
  rw [Finset.sum_comm, Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_comm, Finset.sum_div]

open Classical in
/-- **A plain slot's term averaged over the tape, at any leaf, is at most the chance over the deployed coins of
reading anything but a table that reads their first `j`**, at read sets `rdD` that hold every decoded live opening,
where the simulated strategy reaches the slot's node after `j` coins: `plainOffAtU_le_live` then `plainOffU_le_rd`,
at `Lf`. -/
theorem plainOffL_le_rd {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (T : Model.Oracle 𝔽) (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ)
    (hL : plainN x + 2 = (V false).k x)
    (hslot : ∀ u, SlotRSL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
      hk0 hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val j
      (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
        (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn (plainDg spec)
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
          (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u)))
    (rdD : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ c c' : Fin ((V false).k x) → Coin, (∀ i : Fin ((V false).k x), i.val < j → c i = c' i) → w c = w c')
    (hrel : ∀ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
          (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSim P x spec st) x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          v ∈ rdD (plainΦ x spec st (u, rr)) q) :
    (∑ u, Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0)
        (slotOf r.val (ℓ.val + 1))
        (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
            opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val (ℓ.val + 1) d
              (outs.getD 0 dfltOutL))
          (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen)
          (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r.val (ℓ.val + 1)))
        (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm PT_LOCAL spec.mPts [] [] T (capOfRoot (plainRoot P x ω)[0]!))
        (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
          (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn (plainDg spec)
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
          (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u))) /
        Fintype.card (plainTape x) ≤
      ∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q} := by
  have hR : (∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) =
      Claims.Ex fun p : plainTape x × (Fin (plainN x + 2) → LCoin 65536) =>
        ∑ q, ind (fun c => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) (plainΦ x spec st p) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [← Pr_plainΦ (spec := spec) (st := st) hL, Claims.Bridge.Pr_eq_Ex]
    rfl
  rw [hR, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((slotOffRSL_le_offC execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁
    rest ext hk0 hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) T r ℓ.val j _ (hslot u)).trans
    ((offC_simAt_le execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u) _ r.val ℓ.val j).trans
      (Claims.Bridge.Live.offC_liveStrat_le plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
        (ω, u) _ j (fun rr => w (plainΦ x spec st (u, rr)))
        (fun rr rr' h => hwj _ _ (plainΦ_prefix hL u h))))).trans
    (Claims.Ex_mono fun rr => Finset.sum_le_sum fun q _ => Claims.Bridge.ind_le_ind_imp fun hoff => ?_)
  obtain ⟨hs, hne⟩ := hoff
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  obtain ⟨a, ha, hav⟩ := Claims.Bridge.liveOp_some hv
  have hd := hrel u rr a ha q v hav
  refine ⟨⟨v, hd⟩, fun he => hne ?_⟩
  rw [he, Finset.mem_singleton] at hd
  rw [hv]
  exact congrArg some hd

open Classical in
/-- **The plain bound at one tape and table, from salted sessions after `Commit`, each slot's term raw, at any
leaf**: `plain_le_sessionsEU`'s proof at `session_sound_of_table_accL_nodeOff`, so each plain fork slot keeps its
per-position off mass at its node, averaged over the mixture. -/
theorem plain_le_sessionsRawLK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Lf
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ) :
    Claims.accPr sha512 (VK keep) P x ω ≤ εtable false x +
      (∑ t, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
          SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        (∑ t, Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ)
          (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
              opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD 0 dfltOutL))
            (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
            (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ))
          (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
            PT_LOCAL spec.mPts [] [] (Claims.Flock.oracleOf C) cap₀) (σ t)) / Fintype.card τ +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hper : ∀ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t) ≤
      Accounting.tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ((L.model (Refine.regionsOf st)).shape spec.mPts) +
      expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
          SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t) +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ)
          (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
              opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD 0 dfltOutL))
            (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
            (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ))
          (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
            PT_LOCAL spec.mPts [] [] (Claims.Flock.oracleOf C) cap₀) (σ t) := by
    intro t
    have hs := session_sound_of_table_accL_nodeOff execArith execArith_correct Refine.H512 Refine.enc512 Lf
      (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hsch hm PT_LOCAL spec.mPts hlay [] [] cap₀
      (fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (Claims.Flock.oracleOf C)) = ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512
          Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL)) :=
      congrArg ind (funext fun _ => propext (and_iff_left hT))
    have e₀ : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (outs.getD 0 dfltOutL) ∧
        Off₀L Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd ⟨h0.2.2, h0.2.1⟩ h0.1.2
      · rfl
    rw [← e]
    refine le_trans hs (le_of_eq ?_)
    simp only [List.length_nil]
    rw [e₀, expect_const, add_zero]
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hdiv := div_le_div_of_nonneg_right hsum (Nat.cast_nonneg (Fintype.card τ))
  rw [Claims.Flock.mean_add, Claims.Flock.mean_add, Claims.Flock.mean_const, mean_sum_sum,
    εtable_one_plain h hsch L] at hdiv
  have hsplit : Claims.accPr sha512 (VK keep) P x ω ≤
      (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
        Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  linarith [hsplit, hacc, hdiv]

/-- `plain_le_sessionsRawLK` at the deployed plain verifier (`keep := id`). -/
theorem plain_le_sessionsRawL {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Lf
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      (∑ t, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
          SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL))
        (sessAfterL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        (∑ t, Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ)
          (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
              opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD 0 dfltOutL))
            (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
            (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r ℓ))
          (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
            PT_LOCAL spec.mPts [] [] (Claims.Flock.oracleOf C) cap₀) (σ t)) / Fintype.card τ +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_le_sessionsRawLK id (x := x) (spec := spec) (st := st) (Lf := Lf) (h := h) (L := L) (m := m) (k0 := k0) (l₀ := l₀) (l₁ := l₁) (rest := rest) (ext := ext) (hsch := hsch) (hk0 := hk0) (hm := hm) (hlay := hlay) (Ω := Ω) (P := P) (ω := ω) (C := C) (hT := hT) (τ := τ) (cap₀ := cap₀) (σ := σ) (hacc := hacc)

open Classical in
/-- **A plain run accepts with probability at most `ε_table`, its one-run event's, each slot's read-set term and
that of opening off the table**, at the concrete simulator at any leaf `Lf`, its decoders `hdec` and openings `opn`,
at a table off which the model statement isn't committed: `plain_le_sessionsRawL` at the simulated strategies
(`plain_first_of_simD`, `mean_live_le_simAt`), the self-clash through the live one to the one-run event
(`mean_selfClash_le_live`, `liveMix_le_Pr`), and each slot's term by `plainOffL_le_rd` at the slot's fork round,
where the simulated strategy meets the slot (`hslot`), at read sets holding every decoded opening (`hrel`) and
tables that read only the coins before that round (`hwj`). -/
theorem plain_accPr_le_rdL {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (hset : setupOf false x = some (ins, spec, sts))
    (h : tables false x = [(spec, st)]) {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (hp : spec.points = 2) {Ω : Type} (P : Claims.Prover Ω)
    (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (hsc : PlainClashRun P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSim P x spec st)
      (plainΦ x spec st)
      (LiveSelfClash Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
          ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j)))
    (hslot : ∀ (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) u,
      SlotRSL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
        PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val (forkRound spec st k0 r.val ℓ.val)
        (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
          (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn (plainDg spec)
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
          (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u)))
    (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
      (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) → w r ℓ c = w r ℓ c')
    (hrel : ∀ r ℓ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
          (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSim P x spec st) x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          v ∈ rdD r ℓ (plainΦ x spec st (u, rr)) q) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁ P x ω s) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
        Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hL := plainN_add hset
  have htr := plain_first_of_simD P x ω C spec.mPts hset hd plainN (plainDg spec) (plainW spec hw)
    (plainSim P x spec st) (plainΦ x spec st) ⟨Pr_plainΦ hL, plainSimRun P ω hset h hw hm1 hp⟩
  have hmain := plain_le_sessionsRawL Lf h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay P ω C hT (plainTape x)
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw)))
  refine hmain.trans (add_le_add (add_le_add (add_le_add le_rfl ?_) ?_) le_rfl)
  · exact le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw))
      (liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSim P x spec st) x ω _ (plainEv₁ P x ω) (plainΦ x spec st) (Pr_plainΦ hL _) hsc)
  · rw [Finset.sum_range]
    refine Finset.sum_le_sum fun r _ => ?_
    rw [Claims.Flock.sum_Icc_one]
    exact Finset.sum_le_sum fun ℓ _ => plainOffL_le_rd Lf L m k0 l₀ l₁ rest ext hsch hk0 hm hlay d opn hdec hw P ω C
      (Claims.Flock.oracleOf C) r ℓ _ hL (hslot r ℓ) (rdD r ℓ) (w r ℓ) (hwj r ℓ) (hrel r ℓ)

end FlockDeploy.KS
