module

public import Proofs.Properties.Verity.Sampled.MemberForm.Sample
public import Proofs.Properties.Verity.Sampled.MemberForm.Apart
public import Proofs.Properties.Verity.Sampled.Window.Instance

/-!
# A window whose members' files pass is laid out in order

`Window.Laid` asks that the members' proof-block counts sum to the window's population and that each member's proof
blocks sit after the earlier members'. This file proves it of the window the parameters' members are on
(`WindowOn`), where every member's files pass (`FilesPass`), each member is one range of units and no unit is in two
members' ranges (`Apart`).

* **The counts sum to the population** (`WindowOn.total`): the committed files' units are the window's population
  (`WindowOn.units`) and are the members' reads' units summed (`unitsWith_sum`), and a member whose files pass has as
  many units in its read as proof blocks (`Flock.Guarantees.MemberV5.fileRead_units`).
* **Each member sits after the earlier ones** (`WindowOn.laid`): the members are in the order of their first units,
  each at the range from its first unit, the ranges share no unit and lie inside the population, and their counts sum
  to it, so each starts where the ones before it end (`emb_of_tiles`).
* **From the counts and the places to `Window.Laid`** (`Window.laid_of`).
-/

@[expose] public section

namespace Verity.Sampled

/-- **A window whose members' counts sum to its population and whose members sit at their bases is laid out in
order**: `Window.Laid` reads the members' counts and places alone. -/
theorem Window.laid_of {W : Window} (ht : (W.ms.map fun m => m.n).sum = W.nJ)
    (he : ∀ (j : Fin W.ms.length) (u : Fin (W.ms.get j).n),
      ((W.ms.get j).emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j.1 + u.val) : W.Laid := by
  show (W.joint 0).Laid
  refine ⟨?_, fun k v => ?_⟩
  · rw [W.counts_joint 0]
    exact ht
  · have hb : (W.joint 0).base k = MemberLayout.base (W.ms.map fun m => m.n) k.1 :=
      congrArg (fun cnt => MemberLayout.base cnt k.1) (W.counts_joint 0)
    rw [hb]
    exact get_map_of (fun m : Member W.nJ => m.toV5 0)
      (fun j m' => ∀ u : Fin m'.n, (m'.emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j + u.val)
      W.ms he k v

namespace MemberForm

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- The units of a member's files and each later member's, summed, are the sum of each one's. -/
theorem unitsWith_sum : ∀ (r : Verity.FileRead) (rs : List Verity.FileRead),
    Verity.FileRead.unitsWith r rs = ((r :: rs).map Verity.FileRead.units).sum
  | r, [] => by
    rw [Verity.FileRead.unitsWith, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  | r, s :: rs => by
    rw [Verity.FileRead.unitsWith, unitsWith_sum s rs, List.map_cons, List.sum_cons]

/-- **The members' counts sum to the window's population**, where every member's files pass: the committed files'
units are the population and the members' reads' units summed, and a member whose files pass has as many units in
its read as proof blocks. -/
theorem WindowOn.total {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : WindowOn V θ F W) (hp : ∀ m ∈ W.ms, FilesPass m) : (W.ms.map fun m => m.n).sum = W.nJ := by
  obtain ⟨m₀, rest, hW, hread⟩ := h.read
  have hu : V.units θ F = ((W.ms.map fileRead).map Verity.FileRead.units).sum := by
    rw [hW, List.map_cons]
    show (V.read θ F).units = _
    rw [hread]
    exact unitsWith_sum (fileRead m₀) (rest.map fileRead)
  refine Eq.trans ?_ h.units
  rw [hu, List.map_map]
  refine congrArg List.sum (List.map_congr_left fun m hm => ?_)
  exact ((m.toV5 0).fileRead_units ((hp m hm).checked_all 0)).symm

/-- **A window whose members' files pass is laid out in order**, where each of the parameters' members is one range
of units and no unit is in two members' ranges: the members are in the order of their first units, each at the range
from its first unit, inside the population, and their counts sum to it (`WindowOn.total`), so each member's proof
blocks sit after the earlier members' (`emb_of_tiles`). -/
theorem WindowOn.laid {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : WindowOn V θ F W) (hp : ∀ m ∈ W.ms, FilesPass m) {p : _root_.Spot.OneStageZk.Params}
    {ms : Array _root_.Spot.OneStageZk.Member} (hpa : ParamsAt θ p) (hms : p.members = some ms)
    (hone : ∀ vm ∈ ms, ∃ lo hi, vm.units = #[(lo, hi)]) (hapart : Apart ms.toList) : W.Laid := by
  obtain ⟨p', ms', vms, hp', hms', hperm, hsorted, hall, -⟩ := h.members
  have hpp : p = p' := hpa.unique hp'
  subst hpp
  have hmm : ms = ms' := Option.some.inj (hms.symm.trans hms')
  subst hmm
  have htotal := h.total hp
  have h1 : vms.Pairwise fun a b => ∀ x ∈ a.units.toList, ∀ y ∈ b.units.toList,
      ∀ u, x.1 ≤ u → u < x.2 → y.1 ≤ u → u < y.2 → False :=
    (List.Perm.pairwise_iff (fun {a b} h x hx y hy u a1 a2 a3 a4 => h y hy x hx u a3 a4 a1 a2) hperm).2 hapart
  have hap : vms.Pairwise fun a b =>
      ∀ x, a.first ≤ x → x < a.first + a.count → b.first ≤ x → x < b.first + b.count → False := by
    refine List.Pairwise.imp_of_mem (fun {a b} ha hb hab x c1 c2 c3 c4 => ?_) h1
    obtain ⟨lo, hi, hu⟩ := hone a (Array.mem_toList_iff.1 (hperm.mem_iff.1 ha))
    obtain ⟨lo', hi', hu'⟩ := hone b (Array.mem_toList_iff.1 (hperm.mem_iff.1 hb))
    rw [first_one hu] at c1 c2
    rw [count_one hu] at c2
    rw [first_one hu'] at c3 c4
    rw [count_one hu'] at c4
    exact hab (lo, hi) (by rw [hu]; exact List.mem_singleton.2 rfl) (lo', hi')
      (by rw [hu']; exact List.mem_singleton.2 rfl) x c1 (by omega) c3 (by omega)
  refine Window.laid_of htotal fun j u => ?_
  exact emb_of_tiles (fun vm : _root_.Spot.OneStageZk.Member => vm.first) (fun vm => vm.count)
    (fun m : Member W.nJ => m.n) (fun m => m.emb) (hall.imp fun _ _ hon => ⟨hon.units, hon.place⟩) hsorted hap
    htotal j u

end MemberForm

end Verity.Sampled
