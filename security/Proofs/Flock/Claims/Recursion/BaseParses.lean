module

public import Proofs.Flock.Claims.Recursion.HonestHalfFixture
import all Init.Control.State
import Proofs.Flock.Claims.Recursion.NumsFixedR
import Proofs.Flock.Claims.Recursion.VPiecesDeployed

/-! A recursive statement's base parses at the masked deployment's parser, so the claims' relation at the recursive
parser gives it at the masked one (`R_base`). -/

@[expose] public section

namespace RecDeploy.BaseParses

open RecDeploy.HonestHalf

def Post {α : Type} (m : FlockDeploy.Rd α) (P : α → Prop) : Prop :=
  ∀ s a s', m.run s = some (a, s') → P a

theorem post_bind {α β : Type} {m : FlockDeploy.Rd α} {f : α → FlockDeploy.Rd β} {P : β → Prop}
    (h : ∀ a, Post (f a) P) : Post (m >>= f) P := by
  intro s b s' hb
  rw [StateT.run_bind] at hb
  rcases hm : m.run s with _ | ⟨a, s₁⟩
  · rw [hm] at hb
    cases hb
  · rw [hm] at hb
    exact h a s₁ b s' hb

theorem post_failure {α : Type} {P : α → Prop} : Post (failure : FlockDeploy.Rd α) P := by
  intro s a s' h
  cases h

theorem post_failure_bind {α β : Type} {f : α → FlockDeploy.Rd β} {P : β → Prop} :
    Post ((failure : FlockDeploy.Rd α) >>= f) P := by
  intro s b s' h
  cases h

theorem post_pure {α : Type} {a : α} {P : α → Prop} (h : P a) : Post (pure a : FlockDeploy.Rd α) P := by
  intro s b s' hb
  have hb' : some (a, s) = some (b, s') := hb
  cases hb'
  exact h

theorem read_post : Post RecRaw.read (fun w => ∃ y, FlockDeploy.decode y = some w.base) := by
  unfold RecRaw.read
  refine post_bind fun _ => ?_
  dsimp only
  split
  · exact post_failure_bind
  · refine post_bind fun _ => ?_
    split
    · rename_i hb
      repeat refine post_bind fun _ => ?_
      exact post_pure ⟨_, hb⟩
    · exact post_failure

theorem decodeRec_base {x : Claims.Msg} {w : RecRaw} (h : decodeRec x = some w) :
    ∃ y, FlockDeploy.decode y = some w.base := by
  unfold decodeRec at h
  split at h
  · split_ifs at h with he
    cases h
    exact read_post x _ [] (by assumption)
  · cases h

theorem readOf_base {x : Claims.Msg} {r : _root_.Flock.Readable.Read} (h : readOfO x = some r) :
    ∃ y, FlockDeploy.readOf true y = some r := by
  unfold readOfO insO at h
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  obtain ⟨w, hw0, hw⟩ := Option.bind_eq_some_iff.1 hs
  split_ifs at hw with hz
  · cases hw
  change w.ins.toOption = some s at hw
  cases hi : w.ins with
  | error e => rw [hi] at hw; cases hw
  | ok s' =>
    rw [hi] at hw
    cases hw
    obtain ⟨y, hy⟩ := decodeRec_base hw0
    refine ⟨y, ?_⟩
    have hz' : w.base.zk = true := by simpa using hz
    unfold FlockDeploy.readOf FlockDeploy.insOf
    rw [hy]
    simp [RecRaw.ins_base hi, hz', Except.toOption]

theorem R_base {rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation} {x : Claims.Msg}
    {w : Claims.Flock.Wit} (h : Claims.Flock.R (deployZkRec rel).parse x w) :
    ∃ y, Claims.Flock.R (FlockDeploy.deployZk rel).parse y w := by
  obtain ⟨st, hp, hs⟩ := h
  change ((readOfO x).bind _root_.Flock.Readable.Read.parse).map (_root_.Flock.Readable.toReadable rel) = some st at hp
  cases hr : readOfO x with
  | none => rw [hr] at hp; cases hp
  | some r =>
    obtain ⟨y, hy⟩ := readOf_base hr
    refine ⟨y, st, ?_, hs⟩
    change ((FlockDeploy.readOf true y).bind _root_.Flock.Readable.Read.parse).map
      (_root_.Flock.Readable.toReadable rel) = some st
    rw [hy]
    rw [hr] at hp
    exact hp

theorem fixture_stmt (h : HonestFixture) :
    ∃ (y : Claims.Msg) (w : Claims.Flock.Wit),
      Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse y w := by
  obtain ⟨mask, b0, x, aux, nOf, hd, recRest, decI, w, st, ω, r, hRun, hW, -, -⟩ := h
  obtain ⟨hR, hx, hf, hacc⟩ := hRun
  obtain ⟨xr, rs, hxr, -, -⟩ := schedMatches_dense x st hx hf
  obtain ⟨-, -, -, -, hrel, -⟩ := vstar_complete_run_setup_at FlockDeploy.sha512 (IAt decI)
    (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck
    (bLAt mask (VStarBuild.IrDep x) b0 x aux nOf hd recRest)
    (numsFixed _) hR ω r hacc hW (VPieces.nS_pos hxr)
  obtain ⟨y, hy⟩ := R_base hrel
  exact ⟨y, _, hy⟩

end RecDeploy.BaseParses
