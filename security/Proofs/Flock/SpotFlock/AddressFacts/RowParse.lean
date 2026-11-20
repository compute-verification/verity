module

public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.SpotFlock.AddressFacts.Row

/-!
# Flock's parse of a registered-reads file keeps each port's reads

`Flock.Registered.portsOf` reads a `verity/registered-reads/v2` file, whose ports name no positions: C and the
partition give them, so each port reads no label (`readsRel` empty, `readsFrom` 0). `portsOf_reads` is
`portsOf_entries` with one clause more, that at each label a port reads a position the port's entry reads the same
position (`readsAt`), which holds because a port of such a file reads none.
-/

@[expose] public section

namespace SpotFlock

open Lean (Json)
open FlockSoundness.Refine

namespace AddressRow

theorem nat_num (n : ℕ) : Spot.nat? (Json.num ⟨Int.ofNat n, 0⟩) = some n := by
  simp [Spot.nat?, Spot.int?]

/-- **A `ports` file's ports keep their entries' reads**: each port is an entry of the file's `ports` object under
the port's name, with the port's fields (`PortFields`), and at each label the port reads a position, the entry reads
the same position (`readsAt`). -/
theorem portsOf_reads {j : Json} {own : Array _root_.Flock.Registered.Port}
    (h : _root_.Flock.Registered.portsOf j = .ok own) :
    ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
      ∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ PortFields reg.name e reg ∧
        ∀ u pos, reg.readAt u = some (some pos) → readsAt e u = some pos := by
  revert own
  show Ret (_root_.Flock.Registered.portsOf j) fun own => ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
    ∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ PortFields reg.name e reg ∧
      ∀ u pos, reg.readAt u = some (some pos) → readsAt e u = some pos
  rw [_root_.Flock.Registered.portsOf.eq_def]
  ret_walk [Array.mapM]
  all_goals
    obtain ⟨hall, -⟩ := mapM_ret (by assumption)
        (fun (kv : String × Json) (reg : _root_.Flock.Registered.Port) => PortFields kv.1 kv.2 reg ∧
          ∀ u pos, reg.readAt u = some (some pos) → readsAt kv.2 u = some pos) fun kv => by
      obtain ⟨nm, e⟩ := kv
      dsimp (config := { zeta := false }) only
      ret_walk [_root_.Flock.J.str, _root_.Flock.J.nat, _root_.Flock.unhex]
      all_goals
        refine ⟨?_, fun u pos hu => ?_⟩
        · exact ⟨rfl, ⟨_, FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "root" _ = .ok _›, ‹_›⟩,
            ⟨_, FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "domain" _ = .ok _›, ‹_›⟩,
            FlockComplete.JsonRT.nat_ok ‹_root_.Flock.J.nat e "leaves" _ = .ok _›,
            FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "value" _ = .ok _›⟩
        · simp [_root_.Flock.Registered.Port.readAt] at hu
    refine ⟨_, by assumption, fun reg hreg => ?_⟩
    obtain ⟨⟨nm, e⟩, hkv, hP, hrd⟩ := hall reg hreg
    obtain rfl : reg.name = nm := hP.1
    exact ⟨e, hkv, hP, hrd⟩

end AddressRow

end SpotFlock
