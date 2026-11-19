module

public import Properties.Flock.Circuit.Chain
public import Proofs.Core.Circuit

/-!
# The chain's statements, proved

* `portsBind`: `PortsBind`. A port is among the block's committed inputs and outputs, which `Rblk` opens
  (`into_union_exits_eq_io`) at the position `PortsOpenAt` opens it, so the two values agree unless the openings'
  messages collide (`Binds`); `enc` is injective, and C's port bits read only the port gates (`portBits_congr`).
-/

@[expose] public section

namespace Flock.Chain

open Flock.Program Flock.OfProgram

/-- C's bits at `ports` read only the ports' gates. -/
theorem portBits_congr {P : Prog} {X Y : (ofProgram P).Trace} {ports : Array Spot.BlockFront.Port}
    (h : ∀ q ∈ ports.toList, ∀ hq : q.gate < (ofProgram P).N, X ⟨q.gate, hq⟩ = Y ⟨q.gate, hq⟩) :
    portBits X ports = portBits Y ports := by
  unfold portBits
  induction ports.toList with
  | nil => rfl
  | cons q l ih =>
    simp only [List.flatMap_cons]
    rw [ih fun q' hq' => h q' (List.mem_cons_of_mem _ hq')]
    congr 1
    split
    · rename_i hq
      rw [h q List.mem_cons_self hq]
    · rfl

/-- **`PortsBind`**: at one commitment, a block's openings and the statement's own at the ports give `BlockCorrect`'s
port hypotheses, or two messages the openings hashed collide. -/
theorem portsBind : PortsBind := by
  classical
  intro P units fr t α Sc H enc blocks u L c X π π' z _ _ henc hio hbind hR hO
  obtain ⟨Y, hYi, hYo, hYop⟩ := hO
  by_cases hall : ∀ q ∈ fr.ins ++ fr.outs, ∀ hq : q.gate < (ofProgram P).N, X ⟨q.gate, hq⟩ = Y ⟨q.gate, hq⟩
  · left
    have hi : portBits X fr.ins = portBits Y fr.ins :=
      portBits_congr fun q hq => hall q (Array.mem_append.2 (Or.inl (Array.mem_toList_iff.1 hq)))
    have ho : portBits X fr.outs = portBits Y fr.outs :=
      portBits_congr fun q hq => hall q (Array.mem_append.2 (Or.inr (Array.mem_toList_iff.1 hq)))
    exact ⟨fun i hi' => by rw [hi]; exact hYi i hi', fun k hk => by rw [ho]; exact hYo k hk⟩
  · right
    push_neg at hall
    obtain ⟨q, hq, hlt, hne⟩ := hall
    obtain ⟨-, hmem⟩ := hio q hq
    have hg : (⟨q.gate, hlt⟩ : Fin (ofProgram P).N) ∈
        (ofProgram P).into ((partOf P blocks).gates u.castSucc) ∪
          (ofProgram P).exits ((partOf P blocks).gates u.castSucc) := by
      rw [Verity.Partition.into_union_exits_eq_io]
      exact hmem
    obtain ⟨p, hp, hop⟩ := hR.1 _ hg
    obtain ⟨p', hp', hop'⟩ := hYop q hq hlt
    rw [hp] at hp'
    cases hp'
    rcases hbind _ _ _ _ _ _ hop hop' with heq | hcol
    · exact absurd (henc _ heq) hne
    · exact ⟨_, hcol⟩

end Flock.Chain
