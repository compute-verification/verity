import Proofs.Flock.Soundness.Discharge.ZkPubIn.Custody
import Proofs.Flock.Soundness.Discharge.Canonical.Verifier

set_option Elab.async false

/-!
# The canonical headline at the verifier's public inputs, over what flock-verify accepts

`flock-verify verify --zk --public-inputs F` (`own` F's bytes) sets every table up at `some own` (`AcceptsZKP`) and ends
each session's verdict with the statement check at `publicInputs` true (`ScopeZP`), which passes `--public-inputs` only at
a circuit `ProvedScope.canonicalOk` passes: under `--zk`, hidden outputs, the unit returning the outputs, in the
canonical format. So what it accepts is `VerifiesZKP`, and every fact `zk_session_soundCP` takes about the statement comes
from the check: an untyped statement, its tags (`ScopeZP.tagsOkZHJ`), the circuit (`circZ I`, `ScopeZP.parseH`), its unit
built, `formatOk` (`ScopeZP.formatOk`), and its public input ports and public file as `Stmt.setupHidden` reads them
(`ScopeZP.parsePi`, `ScopeZP.loadP`, at `piZP I` and `pubZP I`).

* `zk_session_soundC_verifiesP`: `zk_session_soundCP` over `VerifiesZKP` (`LiveVerifiesZKP`), at the draw and record
  facts (`hDraw`, `hRec`);
* `zk_session_sound_canonicalP`: `zk_session_soundCP_custody_json` over it, at the record's custody and the printed draw
  file.

At a statement the check refuses the event is empty (`prob_scope`). Without `--zk` the check refuses every statement at
`--public-inputs` (`check_publicInputs_noZk`). `FailClosed.flock_verify_sound` states both.

**Proof work.** None: every declaration here is proved, and so is every one it uses.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec

variable {n : ℕ} {Reg : Type}

/-- **`flock-verify verify --zk --public-inputs F`'s statement check**, at the session's tags (`zkTags`). -/
def ScopeZP (I : Inputs) : Prop :=
  ProvedScope.check (zkTags I) true I.circuitFile I.publicFile I.tables I.count true = .ok ()

/-- **What `flock-verify verify --zk --public-inputs F` accepts**, `own` F's bytes: the setup and verdict `AcceptsZKP`
models, and the statement check. -/
def VerifiesZKP (I : Inputs) (own : ByteArray) (t : Exec.Transcript) : Prop := ZkPubIn.AcceptsZKP I own t ∧ ScopeZP I

/-- `VerifiesZKP` at the live session's transcript. -/
def LiveVerifiesZKP (I : Inputs) (own : ByteArray) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  VerifiesZKP I own o.2.2

/-- The circuit's public input ports as the check reads them (`default` when they do not parse, and then the check
fails). -/
def piZP (I : Inputs) : HmIn.Ports :=
  match HmIn.parse (circZ I) with
  | .ok p => p
  | .error _ => default

/-- The public file with every instance's public words, as the check loads it at `--public-inputs` (`default` when it
does not load, and then the check fails). -/
def pubZP (I : Inputs) : Public :=
  match HmIn.loadPublic (zkTags I) (circZ I) (piZP I) I.publicFile with
  | .ok p => p
  | .error _ => default

theorem circZ_eq {I : Inputs} {c : Circuit}
    (hc : ProvedScope.circuitOf (zkTags I) I.circuitFile I.tables = .ok c) : circZ I = c := by
  unfold circZ circ; rw [hc]

/-- **Without `--zk` the check refuses every statement at `--public-inputs`**: it passes them only in the canonical
format, which is a `--zk` form. -/
theorem check_publicInputs_noZk {t : Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ} :
    ProvedScope.check t false cf pf tables count true ≠ .ok () := by
  intro h
  obtain ⟨-, c, -, -, -, -, -, hp⟩ := check_okP h
  have := hp rfl
  simp [ProvedScope.canonicalOk] at this

namespace ScopeZP

variable {I : Inputs} (h : ScopeZP I)
include h

theorem canonical : ProvedScope.canonicalOk (zkTags I) true (circZ I) = true := by
  obtain ⟨-, c, -, hc, -, -, -, hp⟩ := check_okP h
  rw [circZ_eq hc]; exact hp rfl

theorem typed : I.tags.typed = false := by rw [← zkTags_typed]; exact (check_okP h).1.2.1

theorem retainRounds : I.tags.retainRounds = true := by rw [← zkTags_retainRounds]; exact (check_okP h).1.2.2.1

theorem merkleLeaf : I.tags.merkleLeaf = "hm96-sha512/v1" := by rw [← zkTags_merkleLeaf]; exact (check_okP h).1.2.2.2.1

omit h in
private theorem canonicalOk_spec {t : Tags} {zk : Bool} {c : Circuit} (hs : ProvedScope.canonicalOk t zk c = true) :
    zk = true ∧ t.hiddenOutputs = true ∧ c.outNet = c.unitNet ∧ ProvedScope.formatOk c = true := by
  simp only [ProvedScope.canonicalOk, Bool.and_eq_true, beq_iff_eq] at hs
  exact ⟨hs.1.1.1, hs.1.1.2, hs.1.2, hs.2⟩

theorem hidden : I.tags.hiddenOutputs = true := by
  rw [← zkTags_hiddenOutputs]; exact (canonicalOk_spec h.canonical).2.1

theorem tagsOkZHJ : TagsOkZHJ I := ⟨h.hidden, h.retainRounds, h.merkleLeaf⟩

theorem parseH : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok (circZ I) := by
  obtain ⟨-, c, -, hc, -⟩ := check_okP h
  have hh : (zkTags I).hiddenOutputs = true := by rw [zkTags_hiddenOutputs]; exact h.hidden
  rw [circZ_eq hc]
  unfold ProvedScope.circuitOf at hc
  rwa [if_pos hh] at hc

theorem built : Built (circZ I).unit := by
  have P := parse_facts (Hidden.parse_untyped (by rw [zkTags_typed]; exact h.typed) h.parseH).1
  obtain ⟨ls, hls⟩ := P.check
  exact unit_built P (check_facts P.typed hls)

theorem formatOk : Canonical.formatOk (circZ I) = true :=
  Canonical.formatOk_of_provedScope h.built (canonicalOk_spec h.canonical).2.2.2

private theorem ports : ∃ pi pub, HmIn.parse (circZ I) = .ok pi ∧
    HmIn.loadPublic (zkTags I) (circZ I) pi I.publicFile = .ok pub := by
  obtain ⟨-, c, pub, hc, hp, -⟩ := check_okP h
  rw [circZ_eq hc]
  have hp' : (HmIn.parse c >>= fun pi => HmIn.loadPublic (zkTags I) c pi I.publicFile) = .ok pub := hp
  obtain ⟨pi, hpi, hpub⟩ := Refine.ebind_ok hp'
  exact ⟨pi, pub, hpi, hpub⟩

theorem parsePi : HmIn.parse (circZ I) = .ok (piZP I) := by
  obtain ⟨pi, -, hpi, -⟩ := ports h
  unfold piZP; rw [hpi]

theorem loadP : HmIn.loadPublic (zkTags I) (circZ I) (piZP I) I.publicFile = .ok (pubZP I) := by
  obtain ⟨pi, pub, hpi, hpub⟩ := ports h
  have : piZP I = pi := by unfold piZP; rw [hpi]
  unfold pubZP; rw [this, hpub]

end ScopeZP

end FlockSoundness.Discharge.FailClosed

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden StrictCR
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.FailClosed (ScopeZP LiveVerifiesZKP circZ piZP prob_scope)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **One canonical hidden-output `--zk --public-inputs F` sampled session of `J` tables that flock-verify accepts is
sound.** `zk_session_soundCP` over `VerifiesZKP`, at the draw and record facts. -/
theorem zk_session_soundC_verifiesP
    (I : Inputs) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJP I own dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    {qF qS : ℝ}
    -- cr/sha-512 for the knowledge finders of its simulation
    (hCR : ∀ hs : ScopeZP I,
        let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
          (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
            (runsZCJH_liveCP I own hs.typed hs.parseH hs.formatOk dj y₀ wr _ hs.tagsOkZHJ hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZKP I own o ∧ ∀ hs : ScopeZP I,
        let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
          (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
            (runsZCJH_liveCP I own hs.typed hs.parseH hs.formatOk dj y₀ wr _ hs.tagsOkZHJ hRec hDraw)) τ
        K₀ ≤ (((keyProgP (circZ I) hs.built n (piZP I)).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (Xpub (pubOnes (circZ I) hs.built n (piZP I) own) (pubZeros (circZ I) hs.built n (piZP I) own)
            (XplurZC execArith hs512 ((keyProgP (circZ I) hs.built n (piZP I)).partition (Layout.outsOf (circZ I) hs.built n))
              (planZHJP dj y₀) (tabZHJP dj y₀)
              ((rsZCP dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built).bindingZ
                (rsZCP_computes dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZP I,
        (let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
          (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
            (runsZCJH_liveCP I own hs.typed hs.parseH hs.formatOk dj y₀ wr _ hs.tagsOkZHJ hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJP dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProgP (circZ I) hs.built n (piZP I)).partition (Layout.outsOf (circZ I) hs.built n))
          (planZHJP dj y₀) (tabZHJP dj y₀)
          ((rsZCP dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built).bindingZ
            (rsZCP_computes dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZKP I own) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact zk_session_soundCP I own hs.typed dj y₀ wr dg N hs.parseH hs.built hs.formatOk hs.parsePi hs.loadP kd hkd Lb
    src hA3 hDraw hs.tagsOkZHJ hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs)

omit mPts in
/-- **One hidden-output `--zk --public-inputs F` sampled session of `J` tables that flock-verify accepts is sound.** For
any live prover `τ` of a `--zk --public-inputs F` session of `J ≥ 1` tables (`own` F's bytes), at the record's custody and
the printed draw file, `Pr[flock-verify --zk --public-inputs F accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks +
δ_link` (`zk_session_soundCP_custody_json` over `VerifiesZKP`), the units those of the key program with a gate per
public leaf bit (`keyProgP`) and the extraction the verifier's own bits on the public gates (`Xpub`). The check passes
`--public-inputs` only in the canonical format, at hidden outputs; no other hypothesis on the circuit or the statement's
form. -/
theorem zk_session_sound_canonicalP
    (I : Inputs) (own : ByteArray)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf (circZ I).g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf (circZ I).g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's custody of the record, and its draw file as `flock-verify draw` printed the verifier's draw
    (hRec : RecordsLiveZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr)
    (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZP I,
        let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
          (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
            (runsZCJH_liveCP I own hs.typed hs.parseH hs.formatOk dj y₀ wr _ hs.tagsOkZHJ hRec
              fun _ => drawOkZKHJP_of_file I own hs.typed hs.parseH kd n hdiv dj
                (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZKP I own o ∧ ∀ hs : ScopeZP I,
        let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
          (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
            (runsZCJH_liveCP I own hs.typed hs.parseH hs.formatOk dj y₀ wr _ hs.tagsOkZHJ hRec
              fun _ => drawOkZKHJP_of_file I own hs.typed hs.parseH kd n hdiv dj
                (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        K₀ ≤ (((keyProgP (circZ I) hs.built n (piZP I)).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (Xpub (pubOnes (circZ I) hs.built n (piZP I) own) (pubZeros (circZ I) hs.built n (piZP I) own)
            (XplurZC execArith hs512 ((keyProgP (circZ I) hs.built n (piZP I)).partition (Layout.outsOf (circZ I) hs.built n))
              (planZHJP dj y₀) (tabZHJP dj y₀)
              ((rsZCP dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built).bindingZ
                (rsZCP_computes dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZP I,
        (let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
          (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
            (runsZCJH_liveCP I own hs.typed hs.parseH hs.formatOk dj y₀ wr _ hs.tagsOkZHJ hRec
              fun _ => drawOkZKHJP_of_file I own hs.typed hs.parseH kd n hdiv dj
                (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJP dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProgP (circZ I) hs.built n (piZP I)).partition (Layout.outsOf (circZ I) hs.built n))
          (planZHJP dj y₀) (tabZHJP dj y₀)
          ((rsZCP dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built).bindingZ
            (rsZCP_computes dj y₀ hs.typed hs.parseH hs.formatOk hs.parsePi hs.loadP hs.built))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZKP I own) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact zk_session_soundCP_custody_json I own hs.typed hs.parseH hs.built hs.formatOk hs.parsePi hs.loadP kd hkd hdiv
    dj y₀ wr dg N Lb src hA3 hs.tagsOkZHJ hRec hdj hk hRw hM hρ hr ht' K₀ τ (hCR hs)

end FlockSoundness.Discharge.ZkPubIn
