import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.RightRegZc
import Proofs.Flock.Soundness.Discharge.Layout.RowEnc

/-!
# The bridge's targets with canonical openings (targets, not proved)

A registered opening counts only when it is canonical at its wire (`Bridge.canonOpen`): the wire has a row
(`rows g = some r`), the opening's value is a row (`RowShaped`) of the row's width, and its salt is the `saltLen` bytes
the deployed leaf takes. The game side counts the units wrong with canonical openings (`wrongRegZc`), and the bridge's
targets read them:

* `Bridge.canonRow rows g v`: `g` has a row `r`, `v` is `RowShaped` and `v`'s width is `r.width`;
* `Bridge.canonOpen rows g v y`: `canonRow rows g v`, and the salt `y` has `saltLen` bytes;
* `Bridge.OpeningsCanonTarget`: at `EndToEndHidden`'s session, a drawn unit's openings are canonical (`OpeningsCanon`
  at `canonOpen`), the conjunct `zk_flock_countC_regC` takes;
* `Bridge.EndToEndHiddenC`: `EndToEndHidden` with "wrong" `wrongRegZc` at `canonOpen`. It implies `EndToEndHidden`
  (`wrongRegZ_subset_wrongRegZc`), and follows from `EndToEndHidden`'s argument at `zk_flock_countC_regC` and
  `OpeningsCanonTarget`;
* `Bridge.TransportsLiveC`: `TransportsLive` at canonical openings only;
* `Bridge.FaultyRightTargetC`: `FaultyRightTarget` over `RightRegZc`'s second conjunct and canonical transport. Its
  `canon` reads a value, so at `vb` it is taken with `Val := vb.Opening`, `val := id` and `bit := vb.bit ∘ vb.val`;
* `SampledKSLiveV2`: `SampledKSLive` at circuit files whose ports are all unsegmented row/v2 rows (`Bridge.RowsV2`),
  which `TransportsLiveC` also takes;
* `Bridge.AssemblyTargetC`: those, `PartRusTarget`, `CountTarget` and `EndToEndHiddenC` give `SampledKSLiveV2`.
-/

namespace Flock.Bridge

/-- **A canonical value at wire `g`**: `g` has a row `r`, and the value is a row (`RowShaped`) of `r`'s width. -/
def canonRow {N W : ℕ} (rows : Fin N → Option RowAt) (g : Fin N) (v : FlockSoundness.Discharge.Layout.BitVal W) :
    Prop :=
  ∃ r, rows g = some r ∧ FlockSoundness.Discharge.Layout.RowShaped v ∧ v.1.val = r.width

/-- **A canonical opening at wire `g`**: its value `v` is canonical at `g` (`canonRow`), and its salt `y` has the
`saltLen` bytes the deployed leaf takes (`saltOf`). -/
def canonOpen {N W : ℕ} (rows : Fin N → Option RowAt) (g : Fin N) (v : FlockSoundness.Discharge.Layout.BitVal W)
    (y : List UInt8) : Prop :=
  canonRow rows g v ∧ y.length = saltLen

/-- **The row a layout builds of `16 k` bits is canonical**: at width `K = 16 k ≤ W`, `(widthOf W K, padBits b)` is
`RowShaped` of width `K`. -/
theorem rowShaped_rowOf {W K k : ℕ} (hK : K = 16 * k) (hKW : K ≤ W) (b : Fin K → Bool) :
    FlockSoundness.Discharge.Layout.RowShaped
        ((FlockSoundness.Discharge.Layout.widthOf W K, (FlockSoundness.Discharge.Layout.padBits b : Fin W → Bool)) :
          FlockSoundness.Discharge.Layout.BitVal W) ∧
      (FlockSoundness.Discharge.Layout.widthOf W K).val = K := by
  have hw : (FlockSoundness.Discharge.Layout.widthOf W K).val = K := by
    simp only [FlockSoundness.Discharge.Layout.widthOf]; omega
  refine ⟨⟨by rw [hw, hK]; omega, fun i hi => ?_⟩, hw⟩
  change (FlockSoundness.Discharge.Layout.widthOf W K).val ≤ i.val at hi
  rw [hw] at hi
  simp only [FlockSoundness.Discharge.Layout.padBits, show ¬ i.val < K by omega, ↓reduceDIte]

/-- **M1's core, with canonical openings**: a unit whose committed wires each carry, in some canonical opening of their
registered commit string, their value in `C`'s evaluation on `Xr`'s inputs, and each of whose canonical registered
openings opens its wire's position under `Cm` and `roots`, is not faulty in `witDep C Cm H roots Xr`. -/
def FaultyRightTargetC : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) {Pos Val Opn Com : Type}
    (pos : Fin C.N → Pos) (bit : Val → Fin C.N → Bool) (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com)
    (canon : Fin C.N → Val → Prop) (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.Commitment Bool)
    (roots : List Claims.Msg) (Xr : Fin C.N → Bool) (u : Fin n),
    (∃ op : Pos → Opn, ∀ g ∈ P.io u, com (op (pos g)) = cs (pos g) ∧ bit (val (op (pos g))) g = C.eval Xr g ∧
      canon g (val (op (pos g)))) →
    (∀ g ∈ P.io u, ∀ o : Opn, com o = cs (pos g) → canon g (val o) →
      ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C g, bit (val o) g, d⟩ = true) →
    ∀ (kd : ℕ) (desc : Claims.Msg) (q : Claims.Sampling.Query),
      u.val ∉ Claims.Sampling.faulty H Cm (stmtOf P kd desc q roots) (partOf P) (witDep C Cm H roots Xr)

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **A drawn unit's openings are canonical**, at `EndToEndHidden`'s session: in every satisfying witness of a drawn
unit, the opening of each of its committed wires' commit strings (`lay.row`'s value) is a row of its wire's row's width,
and its salt (`lay.salt`) has `saltLen` bytes (`OpeningsCanon` at `Bridge.canonOpen`). The game's rows are `(widthOf W (rowBits q), padBits b)`, canonical at
`rowBits q = 16·words ≤ W` (`Bridge.rowShaped_rowOf`). -/
def _root_.Flock.Bridge.OpeningsCanonTarget : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀),
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    OpeningsCanon execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (fun g o => Bridge.canonOpen rows g (vb.val o) o.2.2)

open Classical in
/-- **`EndToEndHidden` with canonical openings**: its setting and bound, with "wrong" the units wrong on the
registration with canonical openings (`wrongRegZc` at `Bridge.canonOpen`). -/
def _root_.Flock.Bridge.EndToEndHiddenC : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZc execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb
          (fun g o => Bridge.canonOpen rows g (vb.val o) o.2.2) (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

open Classical in
/-- **The canonical registered openings open the deployed commitment** (`TransportsLive` at canonical openings): in
`EndToEndHidden`'s setting, at every unit `u`, every committed wire `g` of `u` and every opening `o` of `g`'s registered
commit string at the prover's registration that is canonical at `g` (`Bridge.canonOpen`: `g` has a row, the value is a
row of its width, and the salt has `saltLen` bytes), some data opens `g`'s claims position to `o`'s bit there, under the registration's roots
at SHA-512, at a circuit file whose ports are all unsegmented row/v2 rows (`Bridge.RowsV2`). -/
def _root_.Flock.Bridge.TransportsLiveC : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hv2 : Bridge.RowsV2 D.unroll)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    ∀ u, ∀ g ∈ P.io u, ∀ o : vb.Opening, vb.com o = vb.cs (reg σ) (vb.pos g) →
      Bridge.canonOpen rows g (vb.val o) o.2.2 →
      ∃ d : Claims.Msg, Cd.Opens Claims.Sampling.sha512Msg roots ⟨Bridge.posOf Cc g, vb.bit (vb.val o) g, d⟩ = true

open Classical in
/-- **`SampledKSLive` at row/v2 circuit files**: `SampledKSLive`'s statement at a circuit file whose ports are all
unsegmented row/v2 rows (`Bridge.RowsV2`), the rows the deployed sampling commitment's leaves hash
(`SamplingDeploy.row`). `SampledKSLive` implies it (`sampledKSLiveV2_of`). -/
def SampledKSLiveV2 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hv2 : Bridge.RowsV2 D.unroll)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (desc : Claims.Msg) (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    -- the deployed commitment's layout: the game's rows, read off the circuit file
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- the registration's frame-v3 roots, over the public file's rows
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P kd desc q roots) (Bridge.partOf P)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

/-- `SampledKSLiveV2` is `SampledKSLive` with one more hypothesis. -/
theorem sampledKSLiveV2_of (h : SampledKSLive) : SampledKSLiveV2 :=
  fun n Reg I ht D hD hU pub hpub hpt hscope _ => h n Reg I ht D hD hU pub hpub hpt hscope

/-- **The assembly, with canonical openings**: canonical transport, M1's core over `RightRegZc`, the slot count and the
count lemma turn `EndToEndHiddenC` into `SampledKSLive` at row/v2 circuit files (`SampledKSLiveV2`). -/
def _root_.Flock.Bridge.AssemblyTargetC : Prop :=
  Bridge.TransportsLiveC → Bridge.FaultyRightTargetC → Bridge.PartRusTarget → Bridge.CountTarget →
    Bridge.EndToEndHiddenC → SampledKSLiveV2

end Flock.Guarantees
