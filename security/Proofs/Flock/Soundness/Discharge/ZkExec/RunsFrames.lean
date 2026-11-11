module

public import Proofs.Flock.Soundness.Discharge.ZkExec.RunsReplay
public import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams
public import Proofs.Flock.Soundness.Discharge.ZkExec.RunsStatements
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLig

/-!
# `repProg` emits a `--zk` rep's frames

`zk_rep_frames`: a run of `repProg` on retained rounds that consumes every round consumed one round per chunk of
`repFramesZ`, each round retaining its chunk's bytes. The domain, then `replay_emitted`, the non-ZK session's
`ringSwitch_emitted` on the two masked claims and the extras, `zkLigerito_frames` and `verifyInner_emitted` on the ten
constraints: the replay's three, the ring switch's two, the lane phase's two (`ligeritoConstraints_size`) and the
triple's three.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Transcript

theorem size_two_append {α : Type} (a b : α) (xs : Array α) : (#[a, b] ++ xs).size = 2 + xs.size := by
  simp [Nat.add_comm]

theorem size_inner_cons {α : Type} (xs : Array α) (a b : α) (ys : Array α) (hx : xs.size = 3)
    (hy : ys.size = 2) :
    (xs ++ #[a, b] ++ ys).size + 3 = 10 := by
  simp [hx, hy]

/-- **`repProg` emits a `--zk` rep's frames, as a run**: at the `--zk` schedule, a run of rep `r`'s transcript program
from the domain `dom` on rounds that all retained their bytes emitted `repFramesZ` (at the ring switch's batched value
`T`). -/
theorem zk_rep_emitted (st : Setup) (sch : Flock.Schedule) (tPad r : ℕ) (extras : Array Flock.Claim)
    (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (hsch : Zk.schedule st.m = .ok (sch, tPad))
    (hrun : (repProg st sch tPad r extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hret : AllRetained rounds) :
    ∃ T : F128, Emitted ⟨rounds, 0, ByteArray.empty⟩ tr (repFramesZ st sch dom (2 + extras.size) zp T) := by
  obtain ⟨h25, -, -, hk0, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨T, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | ok v =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨lg, t₄, h₄, hrun'⟩ := bind_ok hrun
    clear hrun
    obtain ⟨rfl, rfl⟩ := pure_ok h₄
    refine ⟨T, ?_⟩
    have e₀ : Emitted ⟨rounds, 0, ByteArray.empty⟩ (Flock.Transcript.init rounds dom) [.dom dom] :=
      emitted_absorb rfl rfl rfl (by rw [init_enc]; rfl)
    have r₀ : AllRetained (Flock.Transcript.init rounds dom).rounds := hret
    obtain ⟨hcs, e₁⟩ := replay_emitted st _ zp.proof zp.inner.root (by omega) r₀ h₁
    have e₂ := ringSwitch_emitted _ _ _ (by rw [size_two_append]; omega) _ _ _ (e₁.retained r₀) h₂
    rw [size_two_append] at e₂
    have e₃ := zkLigerito_frames _ sch _ _ _ _ _ hdim hlv hk0 (e₂.retained (e₁.retained r₀)) h₃
    have e₄ := verifyInner_emitted _ _ _ _ (e₃.retained (e₂.retained (e₁.retained r₀))) hrun'
    rw [size_inner_cons _ _ _ _ hcs (ligeritoConstraints_size hv)] at e₄
    have e := e₀.trans (e₁.trans (e₂.trans (e₃.trans e₄)))
    simpa only [repFramesZ, restFramesZ, List.append_assoc, List.cons_append, List.nil_append,
      List.singleton_append] using e
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄

theorem zk_rep_frames : ZkRepFrames := by
  intro st sch tPad r extras zp rounds dom tr hsch hrun hk hret
  obtain ⟨T, e⟩ := zk_rep_emitted st sch tPad r extras zp rounds dom tr hsch hrun hret
  obtain ⟨-, hk', hR, -, -⟩ := e
  refine ⟨T, by rw [← hk, hk']; simp, fun k hk2 => ?_⟩
  have hk3 : k < (chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.length := by
    rw [← hk, hk'] at hk2; simpa using hk2
  have := hR k hk3
  simpa using this

/-- **Each round of a `--zk` rep closes on a squeeze of its own coin count**: on a run of rep `r`'s transcript program
on rounds that all retained their bytes, consuming every round, round `k` retained some bytes and then the header of a
squeeze of `rounds[k].n` coins. -/
theorem zk_rep_counts (st : Setup) (sch : Flock.Schedule) (tPad r : ℕ) (extras : Array Flock.Claim)
    (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (hsch : Zk.schedule st.m = .ok (sch, tPad))
    (hrun : (repProg st sch tPad r extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) (hret : AllRetained rounds) :
    ∀ k < rounds.size, ∃ (pre : List UInt8) (kind : UInt8),
      Retains rounds[k]! (pre ++ (Frame.sq kind (rounds[k]!).n).enc) := by
  obtain ⟨T, e⟩ := zk_rep_emitted st sch tPad r extras zp rounds dom tr hsch hrun hret
  obtain ⟨-, hk', hR, -, hC⟩ := e
  intro k hk2
  have hk3 : k < (chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.length := by
    rw [← hk, hk'] at hk2; simpa using hk2
  obtain ⟨c, kind, hc⟩ := hC k hk3
  have hr := hR k hk3
  rw [hc] at hr
  refine ⟨encs c, kind, ?_⟩
  simpa [encs_append, encs] using hr

end FlockSoundness.Discharge.ZkExec
