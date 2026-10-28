import Proofs.Properties.NetworkRecords.Accept

/-!
# An instance: a unit whose records are the Program's, which the check accepts

One egress link `out` with no frames and one ingress link `in`, one window of one bucket of one slot, Σ_sync
`{(0, 1)}`, and a complete raw record of each sampling block, `out`'s holding the Program's grid, one empty row
(`program₀`). Its records are the Program's (`programRecords₀`) and the check accepts it (`accepted₀`), under the
parameters' digests: `parametersDigest` reads `Flock.canon`, well-founded recursion the kernel does not evaluate, so
they are taken as given.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check (Params Link Evidence)
open NetworkCertifier (ClockSync)
open Properties.NetworkCertifier (ParamsDigests Declares programGrid)
open Properties.NetworkCertifier.CheckProofs (declares_unique declares_declared)

namespace Properties.NetworkRecords.Instance

def sync₀ : ClockSync := ⟨0, 1, by decide⟩

def p₀ : Params where
  egress := #["out"]
  ingress := #["in"]
  windows := 1
  grid := ⟨1, 1, 1, 1, 1⟩
  ingressGrid := ⟨1, 1, 1, 1, 1⟩
  syncSet := ⟨[0], [1], by decide, by decide, fun ρ h => by simp only [List.mem_singleton] at h; rw [h]; decide,
    by simp, by simp⟩
  lms := ⟨8, 1, 1⟩
  key := .empty
  session := ⟨Array.replicate 32 0⟩

def link₀ : Link := ⟨[], #[sync₀]⟩

def ev₀ : Evidence := ⟨#[link₀]⟩

/-- The evidence as the check reads it, its one link's payloads (none) kept as bytes. -/
def e₀ : Records.ByteEvidence := ⟨#[⟨[], #[sync₀]⟩]⟩

theorem toEvidence₀ : e₀.toEvidence = ev₀ := by
  simp [Records.ByteEvidence.toEvidence, Records.ByteLink.toLink, e₀, ev₀, link₀]

def grid₀ : Records.RawGrid := programGrid p₀ (fun _ => sync₀) [] 0

def c₀ : Records.Commitment := ⟨#[.raw "out" 0 none grid₀, .raw "in" 0 none #[]], .empty⟩

/-- The Program's grid of `out` in window 0: one row, its one slot padding. -/
theorem program₀ : grid₀ = #[#[none]] := by decide +kernel

theorem programRecords₀ {eg ig : ByteArray} (hpd : ParamsDigests p₀ eg ig) : ProgramRecords p₀ c₀ e₀.toEvidence := by
  rw [toEvidence₀]
  refine ⟨⟨rfl, ?_, ?_⟩, eg, ig, hpd, ?_, ?_⟩
  · intro link hl
    simp only [ev₀, List.mem_singleton] at hl
    subst hl
    rfl
  · intro link hl f hf
    simp only [ev₀, List.mem_singleton] at hl
    subst hl
    cases hf
  · intro l link hm w hw
    have hm' : (l, link) = ("out", link₀) := by simpa [p₀, ev₀] using hm
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hm'
    obtain rfl : w = 0 := by simp only [p₀] at hw; omega
    refine ⟨.raw "out" 0 none grid₀, rfl, rfl, ⟨sync₀, rfl, by simp [p₀, sync₀], by simp [p₀, sync₀], link₀, sync₀,
      rfl, rfl, rfl, rfl⟩, fun syncs hs => ?_⟩
    have hd : Declares link₀ (fun _ : Fin p₀.windows => sync₀) := fun v => by
      have hv : v.1 = 0 := by have := v.2; simp only [p₀] at this; omega
      simp only [link₀, hv]
      rfl
    obtain rfl := declares_unique hd hs
    rfl
  · intro l hl w hw
    have hl' : l = "in" := by simpa [p₀] using hl
    subst hl'
    obtain rfl : w = 0 := by simp only [p₀] at hw; omega
    exact ⟨.raw "in" 0 none #[], rfl, rfl, trivial⟩

theorem accepted₀ {eg ig : ByteArray} (hpd : ParamsDigests p₀ eg ig) :
    ∃ pub prof, Records.protocol.check p₀ c₀ e₀ = .accept pub prof :=
  programRecords_accepted (programRecords₀ hpd)

end Properties.NetworkRecords.Instance
