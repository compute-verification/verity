import Proofs.Flock.Claims.KS.ConflictMerkle

/-!
# A run within a log, as a verdict

SHA-512 answered only on the messages of `L` (`hashesIn L`, a `Hashes V` that refuses every other input) turns a run
within `L` into a verdict and back: a program over `OC` that runs against `sha512` within `L` is accepted by its copy
with `hashesIn L` (`within`), and a copy that `hashesIn L` accepts runs within `L` (`back`); `hashesIn L` accepts no
more than the deployed hashes do (`sub`). So a lemma proved of the verifier in `V` under every oracle holds of a run
within its log, and the Merkle check it reaches runs within that log too.
-/

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)

/- An equation between two hashes' outputs is decided without evaluating them. -/
attribute [local irreducible] Sha512.hash Sha256.hash Blake3.hash Blake3.keyedHash

/-- **SHA-512 within `L`**: the deployed SHA-512 on inputs whose message is in `L`, a refusal on every other; the other
hashes the deployed ones. -/
def hashesIn (L : List Claims.Msg) : Hashes Flock.V :=
  { Hashes.ofSha512 Sha512.hash with
    sha512 := fun b => if Claims.toMsg b ∈ L then pure (Sha512.hash b) else throw "SHA-512: an input outside the log" }

theorem hashesIn_sha512 {L : List Claims.Msg} {b y : ByteArray} (h : (hashesIn L).sha512 b = .ok y) :
    y = Sha512.hash b ∧ Claims.toMsg b ∈ L := by
  by_cases hb : Claims.toMsg b ∈ L
  · have e : (hashesIn L).sha512 b = pure (Sha512.hash b) := by simp only [hashesIn, hb, ↓reduceIte]
    rw [e] at h
    exact ⟨(Except.ok.inj h).symm, hb⟩
  · have e : (hashesIn L).sha512 b = throw "SHA-512: an input outside the log" := by
      simp only [hashesIn, hb, ↓reduceIte]
    rw [e] at h
    cases h

theorem Runs.ret {H : Claims.Msg → Claims.Msg} {α : Type} (a : α) (L : List Claims.Msg) :
    Runs H (Pure.pure a : OCV α) a L :=
  ⟨rfl, fun _ hp => by simp [Claims.OC.eval] at hp⟩

/-- **Runs of the two parts are a run of the bind.** -/
theorem Runs.bind_intro {H : Claims.Msg → Claims.Msg} {α β : Type} {x : OCV α} {f : α → OCV β} {a : α} {b : β}
    {L : List Claims.Msg} (hx : Runs H x a L) (hf : Runs H (f a) b L) : Runs H (x >>= f) b L := by
  obtain ⟨hx1, hx2⟩ := hx
  obtain ⟨hf1, hf2⟩ := hf
  simp only [Runs, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rw [hx1]
  refine ⟨hf1, fun p hp => ?_⟩
  rcases List.mem_append.1 hp with hp | hp
  · exact hx2 p hp
  · exact hf2 p hp

/-- **A run within `L` is accepted with `hashesIn L`.** -/
def within (L : List Claims.Msg) : With.Sim OCV Flock.V where
  R x y := ∀ a, Runs sha512 x a L → y = .ok a
  pure a a' h := by rw [Runs.pure h]; rfl
  bind {α β x y f g} hxy hfg b h := by
    obtain ⟨a, h1, h2⟩ := Runs.bind h
    rw [hxy a h1]
    exact hfg a b h2
  throw _ _ h := (Runs.throw h).elim
  lift _ _ h := Runs.lift h

theorem within_hashes (L : List Claims.Msg) : (within L).Hashes Claims.Hashes.oc (hashesIn L) where
  sha512 b a h := by
    obtain ⟨rfl, hm⟩ := Runs.hash h
    simp only [hashesIn, hm, ↓reduceIte]
    rfl
  sha256 b a h := by rw [Runs.pure h]; rfl
  blake3 b a h := by rw [Runs.pure h]; rfl
  blake3Keyed k b a h := by rw [Runs.pure h]; rfl

/-- **What `hashesIn L` accepts runs within `L`.** -/
def back (L : List Claims.Msg) : With.Sim OCV Flock.V where
  R x y := ∀ a, y = .ok a → Runs sha512 x a L
  pure a a' h := by cases h; exact Runs.ret a L
  bind {α β x y f g} hxy hfg b h := by
    obtain ⟨a, h1, h2⟩ := FlockSoundness.Refine.ebind_ok h
    exact Runs.bind_intro (hxy a h1) (hfg a b h2)
  throw _ _ h := by cases h
  lift v a h := ⟨h, fun p hp => by
    change p ∈ (Claims.OC.eval sha512 (Claims.OC.ret v)).2 at hp
    simp [Claims.OC.eval] at hp⟩

theorem back_hashes (L : List Claims.Msg) : (back L).Hashes Claims.Hashes.oc (hashesIn L) where
  sha512 b a h := by
    obtain ⟨rfl, hm⟩ := hashesIn_sha512 h
    refine ⟨?_, fun p hp => ?_⟩
    · show Except.ok (Claims.hOf sha512 b) = _
      rw [hOf_sha512]
    · change p ∈ (Claims.OC.eval sha512 (Claims.OC.query (Claims.toMsg b)
        fun a => Claims.OC.ret (Except.ok (Claims.ofMsg a)))).2 at hp
      simp only [Claims.OC.eval, List.mem_cons, List.not_mem_nil, or_false] at hp
      rw [hp]
      exact hm
  sha256 b a h := by rw [← Except.ok.inj h]; exact Runs.ret _ L
  blake3 b a h := by rw [← Except.ok.inj h]; exact Runs.ret _ L
  blake3Keyed k b a h := by rw [← Except.ok.inj h]; exact Runs.ret _ L

/-- **What `hashesIn L` accepts the deployed hashes accept.** -/
def sub : With.Sim Flock.V Flock.V where
  R x y := ∀ a, x = .ok a → y = .ok a
  pure _ _ h := h
  bind {α β x y f g} hxy hfg b h := by
    obtain ⟨a, h1, h2⟩ := FlockSoundness.Refine.ebind_ok h
    rw [hxy a h1]
    exact hfg a b h2
  throw _ _ h := h
  lift _ _ h := h

theorem sub_hashes (L : List Claims.Msg) : sub.Hashes (hashesIn L) With.std where
  sha512 b a h := by
    obtain ⟨rfl, -⟩ := hashesIn_sha512 h
    rfl
  sha256 _ _ h := h
  blake3 _ _ h := h
  blake3Keyed _ _ _ h := h

end FlockDeploy.KS
