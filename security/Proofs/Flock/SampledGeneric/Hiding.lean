module

public import Proofs.Flock.Masked
public import Proofs.Flock.SampledGeneric.HidingStatements

/-!
# The sampled audit's zero knowledge from a hiding trace commitment (proved)

* `commitSplit`: the executable's commit string of an inner digest `x` under a salt `y` is `(x ⊕ M·y) ‖ c(y)`, the
  pair `Claims.Hm96Hiding` is about (`leafSplit` before the tree leaf's hash).
* `Hider.exp_andThen_pr`, `Hider.Hiding.map`, `Hider.Hiding.comap`, `Hider.Hiding.mono`: a commitment's
  post-processed probability as an average over the prover's coins; hiding survives reading the commitment through a
  function and committing to a function of the trace, and grows with its error.
* `hm96Hider_hiding`: `hm96_commit_two_traces` at the test "the post-processing's event": hm96's trace commitment is a
  `Hider` that hides at `2·N·δ₁`. `hm96Exe_hiding`: so do the executable's commit strings, at the pinned key's gap
  (`CROnly.hm96Hiding_gap`, `commitSplit`).
* `Close.coinAvg`: closeness under a uniform coin flipped first, at each coin's error averaged.
* `zkOfMal_zkAt`: `ZKmalR` is `Zk.ZKAt` at `malErr`. `flockSession_zkAt`: `Flock.SecurityProofs.MaskedZK` in that form.
* `Flock.SecurityProofs.Hm96TraceHiding`, `Hm96ExeHiding`, `GenericSampledZKAt`, `SpotFlockZK`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkView

open Flock Hm

/-- **The executable's commit string splits**: for an inner digest `x` and a salt `y`,
`Default512.commitString x y = (x ⊕ M·y) ‖ SHA-512(sp ‖ y)`, its first 64 bytes the bits of `x` plus the salt's mask
under the pinned key. -/
theorem commitSplit (x : ByteArray) (y : Salt) :
    Hm96.Default512.commitString x (saltBytes y) = bytesOf (k := 64) (bitsOf x + My y) ++ saltHash y := by
  rw [Hm96.Default512.commitString, saltHash]
  refine congrArg (fun b => b ++ Sha512.hash (Hm96.Default512.saltPrefix ++ saltBytes y)) ?_
  have hA : ∀ g : ℕ → UInt8, ((Array.range 64).map g).toList.length = 64 := fun g => by
    rw [Array.toList_map, List.length_map, Array.toList_range, List.length_range]
  have hB : ∀ v : Fin (8 * 64) → ZMod 2, (bytesOf (k := 64) v).data.toList.length = 64 := fun v => by
    rw [bytesOf, Array.toList_ofFn, List.length_ofFn]
  refine ByteArray.ext (Array.toList_inj.mp (eq_of_bitL (by rw [hA, hB]) fun t ht => ?_))
  rw [hA] at ht
  have ht' : t < 512 := ht
  rw [bitL_bytesOf (k := 64) _ ht', bitL]
  rw [Array.toList_map, Array.toList_range, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (show t / 8 < 64 by omega), Option.map_some, Option.getD_some]
  rw [UInt8.toNat_xor, Nat.testBit_xor, get!_eq_getD, get!_eq_getD]
  have hm : ∀ s : ByteArray,
      bitL (Hm96.Default512.mask s).data.toList t = maskV Hm96.Default512.key (bitL s.data.toList) t := by
    intro s
    have h := bitL_mask s.data.toList ht'
    rwa [Array.toArray_toList] at h
  have hm' := hm (saltBytes y)
  rw [bitL] at hm'
  rw [hm']
  simp only [Pi.add_apply, bitsOf, My_apply y ⟨t, ht'⟩, bitL]
  exact xor_ind _ _

end FlockSoundness.Discharge.ZkView

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

/-! ## Closeness under a coin, at the coins' errors averaged -/

/-- **Closeness under a uniform coin, at each coin's error**: if each coin's two experiments are `ε r`-close, the
experiments that flip the coin first and relabel each view by the coin are close at `ε` averaged (`Close.coin`, at an
error that reads the coin). -/
theorem Close.coinAvg {Rnd V W : Type} [Fintype Rnd] [Nonempty Rnd] (a b : Rnd → Exp V) (f : Rnd → V → W)
    (ε : Rnd → ℝ≥0∞) (h : ∀ r, Close (a r) (b r) (ε r)) :
    Close ⟨.coin Rnd fun r => (a r).g.map (f r), fun r => Strategy.map _ _ (a r).s⟩
      ⟨.coin Rnd fun r => (b r).g.map (f r), fun r => Strategy.map _ _ (b r).s⟩ (avg ε) := by
  have key : ∀ c : Rnd → Exp V, ∀ E : W → Prop,
      Exp.pr ⟨.coin Rnd fun r => (c r).g.map (f r), fun r => Strategy.map _ _ (c r).s⟩ E =
        avg fun r => (c r).pr fun v => E (f r v) := fun c E =>
    congrArg avg (funext fun r => prob_map_strat E (f r) (c r).g (c r).s)
  intro E
  rw [key a, key b]
  constructor
  · rw [← avg_add]
    exact avg_mono fun r => (h r fun v => E (f r v)).1
  · rw [← avg_add]
    exact avg_mono fun r => (h r fun v => E (f r v)).2

/-! ## Zero knowledge at an error that reads the witness and the verifier -/

/-- `Zk.ZK` at `ε_zk` is `Zk.ZKAt` at the error `ε_zk(x)`. -/
theorem Zk.ZKAt.of_ZK {Stmt Wit VS View : Type} {Zs : Zk Stmt Wit VS View} {Rel : Stmt → Wit → Prop}
    {εzk : Stmt → ℝ≥0∞} (h : Zs.ZK Rel εzk) : Zs.ZKAt Rel fun x _ _ => εzk x :=
  h

/-- `Zk.ZKAt` at the statement `g x`, against a verifier that read `x` (`Zk.after`). -/
theorem Zk.ZKAt.after {Stmt Stmt' Wit VS View : Type} {Zs : Zk Stmt Wit VS View} {Rel : Stmt → Wit → Prop}
    {ε : Stmt → Wit → VS → ℝ≥0∞} (h : Zs.ZKAt Rel ε) (g : Stmt' → Stmt) :
    (Zs.after g).ZKAt (fun x w => Rel (g x) w) fun x w v => ε (g x) w (v x) :=
  fun x w hR v => h (g x) w hR (v x)

/-! ## Commitments -/

namespace Hider

variable {Tr Tr' Rnd Com Com' : Type}

/-- Two commitments with the same commitment to every trace under every coin are one. -/
theorem ext_commit {H H' : Hider Tr Rnd Com} (h : ∀ Xt r, H.commit Xt r = H'.commit Xt r) : H = H' := by
  cases H
  cases H'
  exact congrArg Hider.mk (funext fun Xt => funext fun r => h Xt r)

variable [Fintype Rnd] [Nonempty Rnd]

/-- **A post-processed commitment's probability**: the average over the prover's coins of the post-processing's. -/
theorem exp_andThen_pr {W : Type} (H : Hider Tr Rnd Com) (Xt : Tr) (f : Com → Exp W) (E : W → Prop) :
    ((H.exp Xt).andThen f).pr E = avg fun r => (f (H.commit Xt r)).pr E :=
  rfl

/-- **Hiding survives reading the commitment through a function**: what is computed from a hiding commitment hides. -/
theorem Hiding.map {H : Hider Tr Rnd Com} {εh : ℝ≥0∞} (h : H.Hiding εh) (g : Com → Com') : (H.map g).Hiding εh :=
  fun W Xt Xt' f => h W Xt Xt' fun c => f (g c)

/-- **Hiding survives committing to a function of the trace.** -/
theorem Hiding.comap {H : Hider Tr Rnd Com} {εh : ℝ≥0∞} (h : H.Hiding εh) (t : Tr' → Tr) : (H.comap t).Hiding εh :=
  fun W Xt Xt' f => h W (t Xt) (t Xt') f

/-- Hiding grows with its error. -/
theorem Hiding.mono {H : Hider Tr Rnd Com} {εh εh' : ℝ≥0∞} (h : H.Hiding εh) (hε : εh ≤ εh') : H.Hiding εh' :=
  fun W Xt Xt' f => (h W Xt Xt' f).mono hε

end Hider

/-! ## hm96's trace commitment hides -/

section Hm96

variable {ι Sa Bs C D : Type} [Fintype ι] [DecidableEq ι] [Fintype Sa] [Nonempty Sa] [AddCommGroup Bs] [Fintype Bs]

/-- **hm96's trace commitment is a `Hider` that hides at `2·N·δ₁`**: `hm96_commit_two_traces`, at the test that runs
the post-processing on the leaves and reads its event. -/
theorem hm96Hider_hiding (My : Sa → Bs) (c : Sa → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (Hf : Bs × C → D) :
    (hm96Hider ι My c Hf).Hiding (2 * Fintype.card ι * δ₁) := by
  intro W Xt Xt' f E
  have key : ∀ X₁ X₂ : ι → Bs, (((hm96Hider ι My c Hf).exp X₁).andThen f).pr E ≤
      (((hm96Hider ι My c Hf).exp X₂).andThen f).pr E + 2 * Fintype.card ι * δ₁ := by
    intro X₁ X₂
    rw [Hider.exp_andThen_pr, Hider.exp_andThen_pr]
    exact hm96_commit_two_traces My c hT1 Hf X₁ X₂ (fun d => (f d).pr E) fun d => prob_le_one E (f d).g (f d).s
  exact ⟨key Xt Xt', key Xt' Xt⟩

end Hm96

/-- **The executable's commit strings are hm96's trace commitment**: at the pinned key's mask and the salt's hash,
publishing each leaf's pair as its bytes `b ‖ c`, to the trace of the inner digests' bits (`commitSplit`). -/
theorem hm96Exe_eq (ι : Type) :
    hm96Exe ι = (hm96Hider ι Discharge.ZkView.My Discharge.ZkView.saltHash
      fun p : Discharge.ZkView.Bits × ByteArray => Discharge.ZkView.bytesOf (k := 64) p.1 ++ p.2).comap
        fun (Xt : ι → ByteArray) i => Discharge.ZkView.bitsOf (Xt i) :=
  Hider.ext_commit fun Xt y => funext fun i => Discharge.ZkView.commitSplit (Xt i) (y i)

open Classical in
/-- **The executable's commit strings hide at `2·N·δ₁`**, with `δ₁` the hiding gap at the pinned key, where
`Claims.Hm96Hiding` holds with no hypothesis (`CROnly.hm96Hiding_gap`). -/
theorem hm96Exe_hiding (ι : Type) [Fintype ι] [DecidableEq ι] :
    (hm96Exe ι).Hiding (2 * Fintype.card ι * hm96Gap) := by
  rw [hm96Exe_eq]
  exact (hm96Hider_hiding Discharge.ZkView.My Discharge.ZkView.saltHash
    (CROnly.hm96Hiding_gap Discharge.ZkView.My Discharge.ZkView.saltHash) _).comap _

/-! ## A claims-core record is zero knowledge at `ZKmalR`'s error -/

section Mal

variable {N Ω W : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]

theorem zkOfMal_real_pr (H : Claims.Msg → Claims.Msg) (k : Claims.Msg → ℕ) (P : W → Claims.Prover (N × Ω))
    (S : Claims.Simulator) (Θ : Type) (x : Claims.Msg) (w : W) (v : MalVerifier k Θ)
    (E : List (Claims.Msg × Claims.Out) → Prop) :
    ((zkOfMal H k P S Θ).real x w v).pr E =
      ENNReal.ofReal (Claims.Pr fun ω : N × Ω => E (Claims.runStar H (P w) v.Vs x ω v.θ).msgs) :=
  pr_uniform _ E

theorem zkOfMal_sim_pr (H : Claims.Msg → Claims.Msg) (k : Claims.Msg → ℕ) (P : W → Claims.Prover (N × Ω))
    (S : Claims.Simulator) (Θ : Type) (x : Claims.Msg) (v : MalVerifier k Θ)
    (E : List (Claims.Msg × Claims.Out) → Prop) :
    ((zkOfMal H k P S Θ).sim x v).pr E = ENNReal.ofReal (Claims.Pr fun c : S.C x => E (S.out H v.Vs v.θ x c)) :=
  pr_uniform _ E

/-- **`ZKmalR` is `Zk.ZKAt`**: a record that is zero knowledge against malicious verifiers of its rounds, at
`ε_zk(x) + γ(x)·Pr[Fork]` on every event, is a `Zk` (`zkOfMal`) whose honest view is close to its simulator's at that
error (`malErr`), both ways. -/
theorem zkOfMal_zkAt {H : Claims.Msg → Claims.Msg} {k : Claims.Msg → ℕ} {P : W → Claims.Prover (N × Ω)}
    {R : Claims.Rel W} {S : Claims.Simulator} {εzk γ : Claims.Msg → ℝ} (h : Claims.ZKmalR H k P R S εzk γ)
    (Θ : Type) : (zkOfMal H k P S Θ).ZKAt R fun x w v => malErr H P εzk γ x w v := by
  intro x w hR v E
  have hb := abs_le.1 (h x w hR Θ v.Vs v.θ (v.hk x) E)
  rw [zkOfMal_real_pr, zkOfMal_sim_pr]
  exact ⟨(ENNReal.ofReal_le_ofReal (by linarith [hb.2])).trans ENNReal.ofReal_add_le,
    (ENNReal.ofReal_le_ofReal (by linarith [hb.1])).trans ENNReal.ofReal_add_le⟩

end Mal

/-- **Flock's masked session is zero knowledge at `flockErr`**: `Flock.SecurityProofs.MaskedZK`, as a `Zk`. -/
theorem flockSession_zkAt (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) (ht : 0 < t) (Θ : Type) :
    (flockSession nr t Θ).ZKAt (Claims.Flock.R (flockZk nr t).parse) fun x w v => flockErr nr t x w v :=
  zkOfMal_zkAt (_root_.Flock.SecurityProofs.MaskedZK nr t ht).2 Θ

end FlockSoundness.SampledGeneric

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **hm96's trace commitment hides** (`Flock.Guarantees.Hm96TraceHiding`): `hm96Hider_hiding`. -/
theorem Flock.SecurityProofs.Hm96TraceHiding : Flock.Guarantees.Hm96TraceHiding := by
  intro ι Sa Bs C D _ _ _ _ _ _ My c δ₁ Hf hT1
  exact hm96Hider_hiding My c hT1 Hf

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The executable's hm96 commit strings hide** (`Flock.Guarantees.Hm96ExeHiding`): `hm96Exe_hiding`. -/
theorem Flock.SecurityProofs.Hm96ExeHiding : Flock.Guarantees.Hm96ExeHiding := by
  intro ι _ _
  exact hm96Exe_hiding ι

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit's zero knowledge, at a session's error that reads the witness and the verifier**
(`Flock.Guarantees.GenericSampledZKAt`): `GenericSampledZK`'s two hybrids. The real view is close to `auditSim` at the
real trace, which simulates each session on the real commitment, at the session's error averaged over the commitment
coins (`Close.coinAvg`); that is `ε_hide`-close to `auditSim` at `X₀` (`auditSim_eq_andThen`, `Hider.Hiding`). -/
theorem Flock.SecurityProofs.GenericSampledZKAt : Flock.Guarantees.GenericSampledZKAt := by
  intro n Tr Rnd Com Wit VS View _ _ H Zs Rel ε εh samp wit hZ hH Xt hX X₀ v
  have h1 : Close (auditView H Zs samp wit Xt v) (auditSim H Zs samp Xt v)
      (avg fun r => ε (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v) :=
    Close.coinAvg (fun r => Zs.real (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v)
      (fun r => Zs.sim (H.commit Xt r, samp v (H.commit Xt r)) v)
      (fun r w => (H.commit Xt r, samp v (H.commit Xt r), w)) _ fun r => hZ _ _ (hX r _) v
  have h2 : Close (auditSim H Zs samp Xt v) (auditSim H Zs samp X₀ v) εh := by
    rw [auditSim_eq_andThen, auditSim_eq_andThen]
    exact hH _ Xt X₀ (auditSimAfter Zs samp v)
  exact (h1.trans h2).mono (le_of_eq (add_comm _ _))

open FlockSoundness FlockSoundness.Game FlockSoundness.SampledGeneric in
/-- **The sampled audit over Flock's masked session is zero knowledge** (`Flock.Guarantees.SpotFlockZK`):
`GenericSampledZKAt` at the masked session (`flockSession_zkAt`, from `MaskedZK`) and the executable's commit strings
(`hm96Exe_hiding`). -/
theorem Flock.SecurityProofs.SpotFlockZK : Flock.Guarantees.SpotFlockZK := by
  intro nr t ht ι _ _ n Θ stmt samp wit Xt hX X₀ v
  exact Flock.SecurityProofs.GenericSampledZKAt n _ _ _ _ _ _ (hm96Exe ι) ((flockSession nr t Θ).after stmt)
    (fun s w => Claims.Flock.R (flockZk nr t).parse (stmt s) w) (fun s w v => flockErr nr t (stmt s) w (v s)) _ samp wit
    ((flockSession_zkAt nr t ht Θ).after stmt) (hm96Exe_hiding ι) Xt hX X₀ v
