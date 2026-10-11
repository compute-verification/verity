import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Soundness.Discharge.ZkView.OsSalts

/-!
# The recursive audit's zero knowledge at the operating system's salts

The firewall reads each commitment's salt from the operating system, 192 bytes (A3, `uniform/io-getrandombytes`), where
`Flock.Guarantees.RecursiveZK` draws it uniformly from `ZkView.Salt`. This module takes the guarantee to those salts:

* `firewallLeaf_os`: the executable's leaf of `SHA-512(msg)` at the OS's bytes `b` is `firewallLeaf msg (saltOfBytes b)`.
* `recursiveZK_os_le`, `recursiveZK_os_gap`: `RecursiveZK` with the salts decoded from `nh` streams of a byte source that
  meets A3, at any hm96 hiding bound `δ`, and at the pinned key's hiding gap `hidingGap(My, saltHash)`
  (`CROnly.hm96Hiding_gap`) with no premise but A3.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.ZK
open Game
open scoped ENNReal

section ZK

/-- **The firewall's commitment at the OS's bytes**: the executable's leaf of `SHA-512(msg)` at the 192 bytes `b` the
operating system gives is `firewallLeaf msg (saltOfBytes b)`. -/
theorem firewallLeaf_os (msg : ByteArray) (b : Fin 192 → Fin 256) :
    Hm96.Default512.leaf (Sha512.hash msg) (FlockSoundness.Audit.Law.streamOf b) =
      Flock.Guarantees.firewallLeaf msg (ZkView.saltOfBytes b) :=
  ZkView.leaf_os _ b (Sha512.hash_size _)

/-- **`RecursiveZK` at the OS's salts** (A3), at any hm96 hiding bound `δ`: the firewall's salt `i` is
`saltOfBytes (t o i)`, stream `i` of a byte source `t` on `Ωos` that meets A3 (`uniform/io-getrandombytes`), and the
outer session's witness holds those salts. The auditor's whole view is within `ε_o + n_h·δ` of the same simulator as
`RecursiveZK`'s. -/
theorem recursiveZK_os_le {Ωa Ωp Ωos Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    [Fintype Ωos] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (nh : ℕ)
    (msgs : Ωa → Ωp → Fin nh → ByteArray) {t : Ωos → Fin nh → Fin 192 → Fin 256}
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes (fun _ : Fin nh => 192) t)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo δ : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun r : Ωo => E (view a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) p y r))
        ≤ prCoin (fun s : Ωs => E (sim a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) s)) + εo)
    (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (E : Ωa × (Fin nh → ByteArray) × V → Prop) :
    prCoin (fun w : ((Ωa × Ωp) × Ωos) × Ωo =>
        E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i))),
          view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i)))
            w.1.1.2 (fun i => ZkView.saltOfBytes (t w.1.2 i)) w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
            sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2))
        + εo + nh * δ := by
  refine le_of_eq_of_le ?_ (recursiveZK_le nh msgs view sim hOuter hT1 E)
  have h := ZkView.prCoin_os_prod hA3 (ZkView.saltsEquiv nh)
    fun (w : (Ωa × Ωp) × Ωo) (y : Fin nh → ZkView.Salt) =>
      E (w.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (y i)),
        view w.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (y i)) w.1.2 y w.2)
  refine ((prCoin_congr_equiv (swapLast (Ωa × Ωp) Ωos Ωo)
    (fun w : ((Ωa × Ωp) × Ωos) × Ωo =>
      E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i))),
        view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i)))
          w.1.1.2 (fun i => ZkView.saltOfBytes (t w.1.2 i)) w.2))).symm.trans h).trans ?_
  exact prCoin_congr_equiv (swapLast (Ωa × Ωp) (Fin nh → ZkView.Salt) Ωo)
    (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
      E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)),
        view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)) w.1.1.2 w.1.2 w.2))

/-- **`RecursiveZK` at the OS's salts with no premise but A3**: within `ε_o + n_h·hidingGap(My, saltHash)`, the
guarantee's bound. -/
theorem recursiveZK_os_gap {Ωa Ωp Ωos Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    [Fintype Ωos] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (nh : ℕ)
    (msgs : Ωa → Ωp → Fin nh → ByteArray) {t : Ωos → Fin nh → Fin 192 → Fin 256}
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes (fun _ : Fin nh => 192) t)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun r : Ωo => E (view a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) p y r))
        ≤ prCoin (fun s : Ωs => E (sim a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) s)) + εo)
    (E : Ωa × (Fin nh → ByteArray) × V → Prop) :
    prCoin (fun w : ((Ωa × Ωp) × Ωos) × Ωo =>
        E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i))),
          view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i)))
            w.1.1.2 (fun i => ZkView.saltOfBytes (t w.1.2 i)) w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
            sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2))
        + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) :=
  recursiveZK_os_le nh msgs hA3 view sim hOuter (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) E

end ZK

end Flock.SecurityProofs
