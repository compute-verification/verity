import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Soundness.Discharge.ZkView.OsSalts

/-!
# The recursive audit's zero knowledge at the operating system's salts

The firewall reads each commitment's salt from the operating system, 192 bytes (A3, `uniform/io-getrandombytes`), where
`Flock.Guarantees.RecursiveZK` draws it uniformly from `ZkView.Salt`. This module takes the guarantee to those salts:

* `firewallLeaf_os`: the executable's leaf of `SHA-512(msg)` at the OS's bytes `b` is `firewallLeaf msg (saltOfBytes b)`.
* `recursiveZK_os_le`, `recursiveZK_os_gap`: `RecursiveZK` with the salts decoded from `nh` streams of a byte source that
  passes the event's test up to `η` (A3 per test, `Assumptions.UniformRandomBytesTest`), at any hm96 hiding bound `δ`,
  and at the pinned key's hiding gap `hidingGap(My, saltHash)` (`CROnly.hm96Hiding_gap`) with no premise but A3 per
  test. The bound gains `η`.
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

/-- **`RecursiveZK` at the OS's salts** (A3 per test), at any hm96 hiding bound `δ`: the firewall's salt `i` is
`saltOfBytes (t o i)`, stream `i` of a byte source `t` on `Ωos` (`uniform/io-getrandombytes`), and the outer session's
witness holds those salts. Where the source passes the event's test up to `η` at every auditor's, prover's and outer
session's outcome, the auditor's whole view is within `ε_o + n_h·δ + η` of the same simulator as `RecursiveZK`'s. -/
theorem recursiveZK_os_le {Ωa Ωp Ωos Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    [Fintype Ωos] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (nh : ℕ)
    (msgs : Ωa → Ωp → Fin nh → ByteArray) {t : Ωos → Fin nh → Fin 192 → Fin 256}
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo δ : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun r : Ωo => E (view a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) p y r))
        ≤ prCoin (fun s : Ωs => E (sim a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) s)) + εo)
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (E : Ωa × (Fin nh → ByteArray) × V → Prop) {η : ℝ≥0∞}
    (hA3 : ∀ w : (Ωa × Ωp) × Ωo, FlockSoundness.Assumptions.UniformRandomBytesTest (fun _ : Fin nh => 192) t
      (fun b => E (w.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (ZkView.saltOfBytes (b i))),
        view w.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (ZkView.saltOfBytes (b i))) w.1.2
          (fun i => ZkView.saltOfBytes (b i)) w.2)) η) :
    prCoin (fun w : ((Ωa × Ωp) × Ωos) × Ωo =>
        E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i))),
          view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i)))
            w.1.1.2 (fun i => ZkView.saltOfBytes (t w.1.2 i)) w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
            sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2))
        + εo + nh * δ + η := by
  refine le_trans ?_ (add_le_add (recursiveZK_le nh msgs view sim hOuter hT1 E) (le_refl η))
  have h := (ZkView.prCoin_os_prod (ZkView.saltsEquiv nh)
    (fun (w : (Ωa × Ωp) × Ωo) (y : Fin nh → ZkView.Salt) =>
      E (w.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (y i)),
        view w.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (y i)) w.1.2 y w.2)) hA3).1
  refine ((prCoin_congr_equiv (swapLast (Ωa × Ωp) Ωos Ωo)
    (fun w : ((Ωa × Ωp) × Ωos) × Ωo =>
      E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i))),
        view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i)))
          w.1.1.2 (fun i => ZkView.saltOfBytes (t w.1.2 i)) w.2))).symm.trans_le h).trans_eq ?_
  exact congrArg (· + η) (prCoin_congr_equiv (swapLast (Ωa × Ωp) (Fin nh → ZkView.Salt) Ωo)
    (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
      E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)),
        view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)) w.1.1.2 w.1.2 w.2)))

/-- **`RecursiveZK` at the OS's salts with no premise but A3 per test**: within `ε_o + n_h·hidingGap(My, saltHash) + η`,
the guarantee's bound plus the source's slack on the event's test. -/
theorem recursiveZK_os_gap {Ωa Ωp Ωos Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    [Fintype Ωos] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (nh : ℕ)
    (msgs : Ωa → Ωp → Fin nh → ByteArray) {t : Ωos → Fin nh → Fin 192 → Fin 256}
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun r : Ωo => E (view a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) p y r))
        ≤ prCoin (fun s : Ωs => E (sim a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) s)) + εo)
    (E : Ωa × (Fin nh → ByteArray) × V → Prop) {η : ℝ≥0∞}
    (hA3 : ∀ w : (Ωa × Ωp) × Ωo, FlockSoundness.Assumptions.UniformRandomBytesTest (fun _ : Fin nh => 192) t
      (fun b => E (w.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (ZkView.saltOfBytes (b i))),
        view w.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1 w.1.2 i) (ZkView.saltOfBytes (b i))) w.1.2
          (fun i => ZkView.saltOfBytes (b i)) w.2)) η) :
    prCoin (fun w : ((Ωa × Ωp) × Ωos) × Ωo =>
        E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i))),
          view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (ZkView.saltOfBytes (t w.1.2 i)))
            w.1.1.2 (fun i => ZkView.saltOfBytes (t w.1.2 i)) w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
            sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2))
        + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) + η :=
  recursiveZK_os_le nh msgs view sim hOuter (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) E hA3

end ZK

end Flock.SecurityProofs
