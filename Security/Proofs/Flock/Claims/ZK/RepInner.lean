import Proofs.Flock.Claims.ZK.RepFree
import Proofs.Flock.Claims.ZK.SimInner

/-!
# The cut rep's inner proof returns on its coins alone

Answered by `asksCut`, the inner proof sends four bodies (`ε`, the batching coins, `β`, the query coins), and its only
refusal past them reads the coin `β`. The firewall's `innerP` also stops on a batched constraint that fails and commits
to `τ`, neither of which sends; the cut's `innerCut` does neither. So where the firewall's inner proof returns, the
cut's returns from a log of the same size, with a log of the same size, whatever either reads.

* **Steps** (`fw_coined`, `fw_sampleVecP`, `fw_throw_bind`, `fw_lift_pure_bind`): a step whose return reads only the
  log's size returns the same on both sides; two batching squeezes of nonzero sizes both send one body; a throw on the
  left returns nothing; a pure step on the left sends nothing.
* **`innerP_cut_fw`**: the firewall's inner proof and the cut's, from logs of one size.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.CutFree

open Flock Transcript Replay

variable {α β γ δ : Type}

theorem fw_coined {m : TMW PS α} (hm : Coined m) {t t' : Transcript} {l l' : Array Claims.Msg}
    (hl : l'.size = l.size) : Fw (· = ·) ((m.run t).go l) ((m.run t').go l') := by
  intro a t₁ L h
  obtain ⟨t₁', L', e, hL⟩ := hm t t' l l' a t₁ L h hl
  exact ⟨a, t₁', L', e, hL, rfl⟩

/-- **Two squeezes of nonzero sizes** each send one body, refused at the same log sizes. -/
theorem fw_sampleVecP (coin : ℕ → ℕ → F128) (n : ℕ) {k k' : ℕ} (hk : k ≠ 0) (hk' : k' ≠ 0) {t t' : Transcript}
    {l l' : Array Claims.Msg} (hl : l'.size = l.size) :
    Fw (fun _ _ => True) (((sampleVecP (asksCut coin n) k).run t).go l)
      (((sampleVecP (asksCut coin n) k').run t').go l') := by
  intro a t₁ L h
  rw [SimInner.sampleVecCut_go coin n hk] at h
  obtain ⟨hn, -, -, rfl⟩ := SimInner.ite_ok h
  rw [SimInner.sampleVecCut_go coin n hk', hl, if_neg hn]
  exact ⟨_, _, _, rfl, by simp only [Array.size_push, hl], trivial⟩

theorem fw_throw_bind {S : γ → δ → Prop} {e : String} {f : α → TMW PS γ}
    {y : Except String (δ × Transcript) × Array Claims.Msg} {t : Transcript} {l : Array Claims.Msg} :
    Fw S ((((throw e : TMW PS α) >>= f).run t).go l) y := by
  intro c t₂ L₂ h
  obtain ⟨_, _, _, h', _⟩ := run_ok h
  rw [_root_.Flock.Firewall.Finish.throw_run_go] at h'
  cases h'

theorem fw_throw {S : γ → δ → Prop} {e : String} {y : Except String (δ × Transcript) × Array Claims.Msg}
    {t : Transcript} {l : Array Claims.Msg} : Fw S (((throw e : TMW PS γ).run t).go l) y := by
  intro c t₂ L₂ h
  rw [_root_.Flock.Firewall.Finish.throw_run_go] at h
  cases h

theorem fw_lift_pure_bind {S : γ → δ → Prop} {a : α} {f : α → TMW PS γ}
    {y : Except String (δ × Transcript) × Array Claims.Msg} {t : Transcript} {l : Array Claims.Msg}
    (h : Fw S (((f a).run t).go l) y) : Fw S ((((StateT.lift (pure a) : TMW PS α) >>= f).run t).go l) y := by
  rw [bind_go (x := StateT.lift (pure a)) rfl]
  exact h

/-- **The cut's inner proof returns where the firewall's does**: answered by `asksCut`, from logs of one size, where
the firewall's `innerP` at SHA-512 returns, `innerCut` from any source, commitment, `final_c` and constraints returns,
with a log of the same size. -/
theorem innerP_cut_fw (coin : ℕ → ℕ → F128) (n : ℕ) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic ic' : InnerCommit) (finalC finalC' : F128) (src : RepSrc) (cons cons' : Array (String × Zk.Lin))
    {t t' : Transcript} {l l' : Array Claims.Msg} (hl : l'.size = l.size) :
    Fw (fun _ _ => True) (((innerP (asksCut coin n) (Hashes.ofSha512 h) lay p ic finalC cons).run t).go l)
      (((innerCut (asksCut coin n) lay ic' finalC' src cons').run t').go l') := by
  unfold innerP innerCut
  try dsimp only
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  try dsimp only
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  try dsimp only
  refine fw_bind (fw_coined (coined_sampleP coin n) hl) fun eps eps' _ _ L L' hL he => ?_
  subst he
  try dsimp only
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  try dsimp only
  refine fw_bind (fw_sampleVecP coin n (by simp [Zk.triple]) (by simp [Zk.triple]) hL)
    fun _ _ _ _ L₂ L₂' hL₂ _ => ?_
  try dsimp only
  split
  · first
    | exact fw_throw_bind
    | exact fw_throw
  try simp only [pure_bind]
  rw [tauCommitmentWith_ofSha512]
  refine fw_lift_pure_bind ?_
  try dsimp only
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  try dsimp only
  refine fw_bind (fw_coined (coined_sampleP coin n) hL₂) fun beta beta' _ _ L₃ L₃' hL₃ hb => ?_
  subst hb
  try dsimp only
  by_cases hz : beta.isZero = true
  · simp only [hz, ↓reduceIte]
    first
    | exact fw_throw_bind
    | exact fw_throw
  simp only [hz, Bool.false_eq_true, ↓reduceIte, pure_bind]
  refine fw_liftTM_bind ⟨_, rfl⟩ fun _ _ => ?_
  try dsimp only
  exact fw_bind (fw_coined (coined_sampleVecP coin n Zk.Q) hL₃) fun _ _ _ _ _ _ hL₄ _ => fw_pure hL₄ trivial

end FlockComplete.CutFree
