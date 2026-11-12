module

public import Proofs.Properties.Verity.Sampled.Window.UnboundLaid

/-!
# The claim at a window: what its members' files owe

The claim at a window (`Verity.Sampled.Window.unboundWrong`) reads each member at files that pass the verifier's checks
(`Flock.Guarantees.MemberV5.Checked`). A window's member (`Member`) fixes its circuit file's parse, the circuit's empty
public input ports and the public file's load over them; this file names what `Checked` asks beyond those
(`Member.FilesChecked`) and proves `Checked` from it at every cost (`Member.checked`, `Window.checked`):

* the public file loads as the committed files' circuit reads it, to the member's population;
* no two of the circuit's ports have one published domain;
* the member's roots at its public file's leaf map fix their domains.

Each is a fact about the member's own files, whichever members a sample holds.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden

namespace Member

variable {nJ : ℕ} (m : Member nJ)

/-- **The member's files pass the checks the verifier makes of a statement's files**, beyond what the member fixes:
its public file loads as the committed files' circuit reads it, to the member's population; its ports' published
domains are pairwise distinct; and its roots at the public file's leaf map fix their domains. -/
structure FilesChecked : Prop where
  loads : ∃ pubO : Flock.Public,
    Flock.HmOut.loadPublic (zkTags m.I) m.D.unroll m.I.publicFile = .ok pubO ∧ pubO.n = m.n
  distinct : Flock.PortRoots.dupId m.D.unroll m.pub = false
  fixes : Flock.Bridge.RootFixesDomainL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n)

/-- **A member whose files pass the checks is `Checked` as Flock's member**, at every cost. -/
theorem checked (h : m.FilesChecked) (t : ℕ) : (m.toV5 t).Checked := by
  obtain ⟨pubO, hp, hn⟩ := h.loads
  have hparsed := Flock.Bridge.parsedOf_eq m.hD hp m.hU
  rw [hn] at hparsed
  exact ⟨rfl, hparsed, h.distinct, h.fixes⟩

end Member

/-- **Every member of the window has files that pass the checks** (`Member.FilesChecked`), drawn or not. -/
def Window.FilesChecked (W : Window) : Prop := ∀ m ∈ W.ms, m.FilesChecked

/-- A member of Flock's window at `t` is a member of the window, as Flock's, so `Checked` where the window's members'
files pass the checks. -/
theorem Window.checked_of_mem (W : Window) (h : W.FilesChecked) (t : ℕ)
    {M : Flock.Guarantees.MemberV5 ByteArray W.nJ} (hM : M ∈ W.ms.map fun m => m.toV5 t) : M.Checked := by
  obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hM
  exact m.checked (h m hm) t

/-- **At a window whose members' files pass the checks, every member of Flock's window is `Checked`**, at every
cost. -/
theorem Window.checked (W : Window) (h : W.FilesChecked) (t : ℕ) (k : Fin (W.joint t).ms.length) :
    ((W.joint t).ms.get k).Checked :=
  W.checked_of_mem h t (List.get_mem (W.joint t).ms k)

end Verity.Sampled
