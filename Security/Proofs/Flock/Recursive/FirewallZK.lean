import Proofs.Flock.Recursive.Firewall
import Proofs.Flock.Recursive.OuterZK
import Proofs.Flock.Soundness.Discharge.ZkView.Leaf
import Proofs.Flock.Soundness.ZK.HidingAdaptive

/-!
# Proof of (ii) at the firewall and V*'s code

`Flock.SecurityProofs.RecursiveFirewallPrivate : Flock.Guarantees.RecursiveFirewallPrivate` is
`RecursiveCircuitPrivateLeaves`' argument at the commit strings the firewall sends, where that one is at their leaves,
with the inner coins live: message `i` reads the commit strings before it, so T1 is taken over adaptive leaves.

* `ZkView.commitString_split`: the executable's commit string of `x` at a salt's bytes is `commitBytes (x ⊕ M·y, c(y))`
  (`leafSplit` before its leaf hash). `FirewallZK.firewallStrings_os`: so the strings the firewall sends at the
  operating system's bytes `b` are `commitBytes` of `firewallPairs` at the salts `saltOfBytes b`, one message at a time.
* `FirewallZK.firewallPairs_adaptive`: the firewall's commit strings are T1's adaptive leaves (`ZK.adaptiveLeaves`).
* `FirewallZK.recursiveZK_enc`: `recursiveZK_le` with the auditor reading the commitments through any encoding `H` of
  their two halves (`encAt`, `encIdeal`), at any hm96 hiding bound `δ`, T1 over the adaptive leaves
  (`ZK.ideal_leaves_adaptive`); `FirewallZK.private_enc`: `RecursiveCircuitPrivate`'s two ways there, with the
  registration's hiding.
* `FirewallZK.fwRealPr_eq`: the firewall's real view is that at `H = commitBytes`, by `FirewallSends` at the OS's
  bytes, taken to salts (`ZkView.saltsEquiv`).
* `RecursiveFirewallPrivate`: `private_enc` at `commitBytes`, V*'s sessions' zero knowledge
  (`OuterZK.vstar_outer_zk` at `proverExe`, its tables passing `CoinedOK` by `SessionsCoined`) and the
  registration's hiding (`regLeaves_hides`), each at the pinned key's hiding gap (`CROnly.hm96Hiding_gap`). The
  simulated view is `private_enc`'s by definition.
-/

namespace FlockSoundness.Discharge.ZkView

open Flock Hm

/-- **The executable's commit string splits** (`leafSplit` before its leaf hash): `Default512.commitString x` at a
salt's bytes is `commitBytes (x ⊕ M·y, c(y))`. -/
theorem commitString_split (x : ByteArray) (y : Salt) :
    Hm96.Default512.commitString x (saltBytes y) = Flock.Guarantees.commitBytes (bitsOf x + My y, saltHash y) := by
  rw [Hm96.Default512.commitString, Flock.Guarantees.commitBytes, saltHash]
  refine congrArg (· ++ Sha512.hash (Hm96.Default512.saltPrefix ++ saltBytes y)) ?_
  show ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ (Hm96.Default512.mask (saltBytes y)).get! i) =
    bytesOf (k := 64) (bitsOf x + My y)
  have hA : ∀ g : ℕ → UInt8, ((Array.range 64).map g).toList.length = 64 := fun g => by
    rw [Array.toList_map, List.length_map, Array.toList_range, List.length_range]
  have hB : ∀ v : Fin (8 * 64) → ZMod 2, (bytesOf (k := 64) v).data.toList.length = 64 := fun v => by
    rw [bytesOf, Array.toList_ofFn, List.length_ofFn]
  refine ByteArray.ext (Array.toList_inj.mp (eq_of_bitL (by rw [hA, hB]) fun t ht => ?_))
  rw [hA] at ht
  have ht' : t < 512 := ht
  rw [bitL_bytesOf (k := 64) _ ht', bitL]
  rw [Array.toList_map, Array.toList_range, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (show t / 8 < 64 by omega), Option.map_some, Option.getD_some]
  rw [UInt8.toNat_xor, Nat.testBit_xor, get!_eq_getD, get!_eq_getD]
  have hm : ∀ s : ByteArray,
      bitL (Hm96.Default512.mask s).data.toList t = maskV Hm96.Default512.key (bitL s.data.toList) t := by
    intro s
    have h := bitL_mask s.data.toList ht'
    rwa [Array.toArray_toList] at h
  have hm' := hm (saltBytes y)
  rw [bitL] at hm'
  rw [hm']
  simp only [Pi.add_apply, bitsOf, My_apply y ⟨t, ht'⟩, bitL]
  exact xor_ind _ _

end FlockSoundness.Discharge.ZkView

namespace FlockSoundness.Discharge.Recursion.FirewallZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

theorem firewallPairs_eq {nh : ℕ} (m : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (y : Fin nh → ZkView.Salt)
    (i : Fin nh) :
    Flock.Guarantees.firewallPairs m y i =
      (ZkView.bitsOf (Flock.Sha512.hash (m i fun j =>
          Flock.Guarantees.commitBytes (Flock.Guarantees.firewallPairs m y (Fin.castLE i.isLt.le j)))) +
        ZkView.My (y i), ZkView.saltHash (y i)) := by
  -- `dsimp` first: unifying `(π i).1` with `ZkView.My (y i)` unfolds `My`'s 1536-term sum past `maxRecDepth`
  have h := ZK.adaptiveLeaves_eq (id : ZkView.Bits × ByteArray → ZkView.Bits × ByteArray)
    (fun i l => ZkView.bitsOf (Flock.Sha512.hash (m i fun j => Flock.Guarantees.commitBytes (l j))))
    (fun i => (ZkView.My (y i), ZkView.saltHash (y i))) i
  dsimp only [id_eq] at h
  exact h

theorem firewallStrings_eq {nh : ℕ} (m : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (ys : Fin nh → ByteArray)
    (i : Fin nh) :
    Flock.Guarantees.firewallStrings m ys i = Flock.Hm96.Default512.commitString
      (Flock.Sha512.hash (m i fun j => Flock.Guarantees.firewallStrings m ys (Fin.castLE i.isLt.le j))) (ys i) := by
  rw [Flock.Guarantees.firewallStrings]

/-- **The firewall's commit strings are T1's adaptive leaves** (`ZK.adaptiveLeaves` at `H = id`): string `i` is
`x_i(the strings before it) + M·y_i` beside `c(y_i)`, with `x_i` the SHA-512 of message `i` at those strings. -/
theorem firewallPairs_adaptive {nh : ℕ} (m : (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (y : Fin nh → ZkView.Salt) :
    Flock.Guarantees.firewallPairs m y = ZK.adaptiveLeaves (id : ZkView.Bits × ByteArray → ZkView.Bits × ByteArray)
      (fun i l => ZkView.bitsOf (Flock.Sha512.hash (m i fun j => Flock.Guarantees.commitBytes (l j))))
      (fun i => (ZkView.My (y i), ZkView.saltHash (y i))) :=
  rfl

/-- **The firewall's commit strings at salts' bytes split**, one message at a time (`ZkView.commitString_split`, each
message reading the strings before it): at the bytes of the salts `y` they are `commitBytes` of `firewallPairs`. -/
theorem firewallStrings_salts {nh : ℕ} (m : (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (y : Fin nh → ZkView.Salt) :
    Flock.Guarantees.firewallStrings m (fun i => ZkView.saltBytes (y i)) =
      fun i => Flock.Guarantees.commitBytes (Flock.Guarantees.firewallPairs m y i) := by
  suffices h : ∀ k (i : Fin nh), i.val = k →
      Flock.Guarantees.firewallStrings m (fun i => ZkView.saltBytes (y i)) i =
        Flock.Guarantees.commitBytes (Flock.Guarantees.firewallPairs m y i) from funext fun i => h _ i rfl
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    intro i hi
    have e : (fun j : Fin i =>
          Flock.Guarantees.firewallStrings m (fun i => ZkView.saltBytes (y i)) (Fin.castLE i.isLt.le j)) =
        fun j => Flock.Guarantees.commitBytes (Flock.Guarantees.firewallPairs m y (Fin.castLE i.isLt.le j)) :=
      funext fun j => ih j.val (lt_of_lt_of_eq j.isLt hi) (Fin.castLE i.isLt.le j) rfl
    refine (firewallStrings_eq m (fun i => ZkView.saltBytes (y i)) i).trans
      ((congrArg (fun F => Flock.Hm96.Default512.commitString (Flock.Sha512.hash (m i F)) (ZkView.saltBytes (y i)))
        e).trans ?_)
    exact (ZkView.commitString_split _ _).trans (congrArg Flock.Guarantees.commitBytes (firewallPairs_eq m y i).symm)

/-- The firewall's commitments to `m` under the salts `y`, the inner coins live (`firewallPairs`), through an encoding
`H` of their two halves. -/
def encAt {D : Type} (H : ZkView.Bits × ByteArray → D) {n : ℕ} (m : (i : Fin n) → (Fin i → ByteArray) → ByteArray)
    (y : Fin n → ZkView.Salt) : Fin n → D :=
  fun i => H (Flock.Guarantees.firewallPairs m y i)

/-- **The firewall's commit strings at the OS's salt bytes**: at the 192 bytes `b i` the operating system gives each
salt, they are `commitBytes` of the commitments' halves at the salts `saltOfBytes (b i)`. -/
theorem firewallStrings_os {nh : ℕ} (m : (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (b : Fin nh → Fin 192 → Fin 256) :
    Flock.Guarantees.firewallStrings m (fun i => FlockSoundness.Audit.Law.streamOf (b i)) =
      encAt Flock.Guarantees.commitBytes m fun i => ZkView.saltOfBytes (b i) :=
  (congrArg (Flock.Guarantees.firewallStrings m) (funext fun i => (ZkView.saltBytes_saltOfBytes (b i)).symm)).trans
    (firewallStrings_salts m fun i => ZkView.saltOfBytes (b i))

/-- Ideal commitments through `H`. -/
def encIdeal {D : Type} (H : ZkView.Bits × ByteArray → D) {n : ℕ} (u : Fin n → ZkView.Bits × ZkView.Salt) :
    Fin n → D :=
  fun i => H (OuterZK.idealStrs u i)

/-- **`recursiveZK_le` through any encoding of the commitments**, at any hm96 hiding bound `δ`: the outer session's
simulator at the commitments (`hOuter`), then T1 once per commitment over the adaptive leaves
(`ZK.ideal_leaves_adaptive`: message `i` reads the commit strings before it). -/
theorem recursiveZK_enc {D Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo]
    [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (H : ZkView.Bits × ByteArray → D) (nh : ℕ)
    (msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (view : Ωa → (Fin nh → D) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → D) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun q : Ωo => E (view a (encAt H (msgs a p) y) p y q))
        ≤ prCoin (fun s : Ωs => E (sim a (encAt H (msgs a p) y) s)) + εo)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (E : Ωa × (Fin nh → D) × V → Prop) :
    prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2,
          view w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.1.1.2 w.1.2 w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2))
        + εo + nh * δ := by
  -- the outer session, simulated at the commitments
  have h1 : prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2,
          view w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.1.1.2 w.1.2 w.2)) ≤
      prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2, sim w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.2))
        + εo :=
    calc _ = avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => prCoin fun q : Ωo =>
            E (ωy.1.1, encAt H (msgs ωy.1.1 ωy.1.2) ωy.2,
              view ωy.1.1 (encAt H (msgs ωy.1.1 ωy.1.2) ωy.2) ωy.1.2 ωy.2 q)) :=
          ZK.prCoin_prod_split fun (ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt)) (q : Ωo) =>
            E (ωy.1.1, encAt H (msgs ωy.1.1 ωy.1.2) ωy.2,
              view ωy.1.1 (encAt H (msgs ωy.1.1 ωy.1.2) ωy.2) ωy.1.2 ωy.2 q)
      _ ≤ avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => (prCoin fun s : Ωs =>
            E (ωy.1.1, encAt H (msgs ωy.1.1 ωy.1.2) ωy.2, sim ωy.1.1 (encAt H (msgs ωy.1.1 ωy.1.2) ωy.2) s)) + εo) :=
          avg_mono fun ωy => hOuter ωy.1.1 ωy.1.2 ωy.2 fun v => E (ωy.1.1, encAt H (msgs ωy.1.1 ωy.1.2) ωy.2, v)
      _ = avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => prCoin fun s : Ωs =>
            E (ωy.1.1, encAt H (msgs ωy.1.1 ωy.1.2) ωy.2, sim ωy.1.1 (encAt H (msgs ωy.1.1 ωy.1.2) ωy.2) s))
            + εo := by
          rw [avg_add, avg_const]
      _ = _ :=
          congrArg (· + εo) (ZK.prCoin_prod_split fun (ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt)) (s : Ωs) =>
            E (ωy.1.1, encAt H (msgs ωy.1.1 ωy.1.2) ωy.2,
              sim ωy.1.1 (encAt H (msgs ωy.1.1 ωy.1.2) ωy.2) s)).symm
  -- T1 once per commitment: the simulated session reads the salts only through the commitments, each message the
  -- commit strings before it
  have h2 : prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2, sim w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.2)) ≤
      prCoin (fun w : ((Ωa × Ωp) × Ωs) × (Fin nh → ZkView.Bits × ZkView.Salt) =>
        E (w.1.1.1, encIdeal H w.2, sim w.1.1.1 (encIdeal H w.2) w.1.2)) + nh * δ := by
    refine (ZK.prCoin_congr_equiv (Flock.SecurityProofs.swapLast (Ωa × Ωp) (Fin nh → ZkView.Salt) Ωs)
      (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2,
          sim w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.2))).symm.trans_le
      (le_of_eq_of_le ?_ (ZK.ideal_leaves_adaptive ZkView.My ZkView.saltHash hT1
        (id : ZkView.Bits × ByteArray → ZkView.Bits × ByteArray) nh
        (fun (ω : (Ωa × Ωp) × Ωs) i l =>
          ZkView.bitsOf (Flock.Sha512.hash (msgs ω.1.1 ω.1.2 i fun j => Flock.Guarantees.commitBytes (l j))))
        (fun ω l => E (ω.1.1, (fun i => H (l i)), sim ω.1.1 (fun i => H (l i)) ω.2))))
    refine congrArg prCoin (funext fun x => ?_)
    exact congrArg (fun l => E (x.1.1.1, (fun i => H (l i)), sim x.1.1.1 (fun i => H (l i)) x.1.2))
      (firewallPairs_adaptive (msgs x.1.1.1 x.1.1.2) x.2)
  -- the developer's randomness is no longer read
  have h3 : prCoin (fun w : ((Ωa × Ωp) × Ωs) × (Fin nh → ZkView.Bits × ZkView.Salt) =>
        E (w.1.1.1, encIdeal H w.2, sim w.1.1.1 (encIdeal H w.2) w.1.2)) =
      prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
        E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2)) :=
    calc _ = prCoin (fun z : Ωp × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
          E (z.2.1.1, encIdeal H z.2.1.2, sim z.2.1.1 (encIdeal H z.2.1.2) z.2.2)) :=
          ZK.prCoin_congr_equiv (Flock.SecurityProofs.pullMid Ωa Ωp Ωs (Fin nh → ZkView.Bits × ZkView.Salt))
            (fun z : Ωp × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
              E (z.2.1.1, encIdeal H z.2.1.2, sim z.2.1.1 (encIdeal H z.2.1.2) z.2.2))
      _ = avg (fun _ : Ωp => prCoin fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
            E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2)) :=
          ZK.prCoin_prod_split fun (_ : Ωp) (w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
            E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2)
      _ = _ := avg_const _
  refine h1.trans ((add_le_add h2 (le_refl εo)).trans (le_of_eq ?_))
  rw [h3, add_right_comm]

/-- The real view's probability of `E` at the registration's salts `r`, the commitments through `H`. -/
noncomputable def realAt {D Prog Wit Rr Rv Ωa Ωp Ωo V : Type} [Fintype Ωa] [Fintype Ωp] [Fintype Ωo]
    (H : ZkView.Bits × ByteArray → D) (reg : Prog → Rr → Rv) (nh : ℕ)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → D) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (P : Prog) (w : Wit) (E : Rv × Ωa × (Fin nh → D) × V → Prop) (r : Rr) : ℝ≥0∞ :=
  prCoin fun z : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
    E (reg P r, z.1.1.1, encAt H (msgs P w r z.1.1.1 z.1.1.2) z.1.2,
      view P w r z.1.1.1 (encAt H (msgs P w r z.1.1.1 z.1.1.2) z.1.2) z.1.1.2 z.1.2 z.2)

/-- The simulated view's probability of `E` at the registration's view `v`: ideal commitments through `H`, and the
outer session's simulator. -/
noncomputable def idealAt {D Out Rv Ωa Ωs V : Type} [Fintype Ωa] [Fintype Ωs] (H : ZkView.Bits × ByteArray → D)
    (nh : ℕ) (sim : Out → Rv → Ωa → (Fin nh → D) → Ωs → V) (o : Out) (E : Rv × Ωa × (Fin nh → D) × V → Prop)
    (v : Rv) : ℝ≥0∞ :=
  prCoin fun z : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
    E (v, z.1.1, encIdeal H z.1.2, sim o v z.1.1 (encIdeal H z.1.2) z.2)

/-- **`RecursiveCircuitPrivate` through any encoding of the commitments**, at any hm96 hiding bound `δ`: at each
registration salt, `recursiveZK_enc` (and on the event's complement, the other way), then the registration's hiding as
a test with values in `[0, 1]` (`Hides.avg_le`). -/
theorem private_enc {D Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Rr] [Nonempty Rr] [Fintype Ωa]
    [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    (H : ZkView.Bits × ByteArray → D) (reg : Prog → Rr → Rv) {ε : ℝ≥0∞} (hreg : PrivateCircuit.Hides reg ε)
    (honest : Out → Prog → Wit → Prop) (nh : ℕ)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → D) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → D) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (encAt H (msgs P w r a p) y) p y q))
        ≤ prCoin (fun s : Ωs => E (sim o (reg P r) a (encAt H (msgs P w r a p) y) s)) + εo)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit) (hw : honest o P w) (E : Rv × Ωa × (Fin nh → D) × V → Prop) :
    avg (realAt H reg nh msgs view P w E) ≤
        avg (fun r => idealAt H nh sim o E (reg p₀ r)) + (ε + εo + nh * δ) ∧
      avg (fun r => idealAt H nh sim o E (reg p₀ r)) ≤
        avg (realAt H reg nh msgs view P w E) + (ε + εo + nh * δ) := by
  have hstep : ∀ r, realAt H reg nh msgs view P w E r ≤
        idealAt H nh sim o E (reg P r) + (εo + nh * δ) ∧
      idealAt H nh sim o E (reg P r) ≤ realAt H reg nh msgs view P w E r + (εo + nh * δ) :=
    fun r => by
      have zk := fun E' => recursiveZK_enc H nh (msgs P w r) (view P w r) (sim o (reg P r)) (hOuter o P w hw r) hT1 E'
      refine ⟨(zk fun t => E (reg P r, t.1, t.2.1, t.2.2)).trans (le_of_eq (add_assoc _ _ _)), ?_⟩
      exact Flock.SecurityProofs.prCoin_flip _ _
        ((zk fun t => ¬ E (reg P r, t.1, t.2.1, t.2.2)).trans (le_of_eq (add_assoc _ _ _)))
  have hf1 : ∀ v, idealAt H nh sim o E v ≤ 1 := fun v => PrivateCircuit.prCoin_le_one' _
  constructor
  · calc avg (realAt H reg nh msgs view P w E)
        ≤ avg (fun r => idealAt H nh sim o E (reg P r) + (εo + nh * δ)) :=
          avg_mono fun r => (hstep r).1
      _ = avg (fun r => idealAt H nh sim o E (reg P r)) + (εo + nh * δ) := by
          rw [avg_add, avg_const]
      _ ≤ (avg (fun r => idealAt H nh sim o E (reg p₀ r)) + ε) + (εo + nh * δ) :=
          add_le_add (hreg.avg_le P p₀ (idealAt H nh sim o E) hf1) le_rfl
      _ = avg (fun r => idealAt H nh sim o E (reg p₀ r)) + (ε + εo + nh * δ) := by ring
  · calc avg (fun r => idealAt H nh sim o E (reg p₀ r))
        ≤ avg (fun r => idealAt H nh sim o E (reg P r)) + ε := hreg.avg_le p₀ P (idealAt H nh sim o E) hf1
      _ ≤ avg (fun r => realAt H reg nh msgs view P w E r + (εo + nh * δ)) + ε :=
          add_le_add (avg_mono fun r => (hstep r).2) le_rfl
      _ = avg (realAt H reg nh msgs view P w E) + (ε + εo + nh * δ) := by
          rw [avg_add, avg_const]; ring

/-- **The firewall's real view is the commit strings' view**: under `FirewallSends`, what the firewall sends at the
OS's bytes `b` is the schedule's transcript filled with `commitBytes` of the commitments at the salts `saltOfBytes b`
(`firewallStrings_os`), and the bytes are uniform exactly when those salts are (`ZkView.saltsEquiv`). V*'s
sessions read the commitments' leaves, `Default512.treeLeaf` of their commit strings. -/
theorem fwRealPr_eq {Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Fintype Ωp] {nr nh : ℕ}
    (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent)
    (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub) (prover : OuterZK.TableProver)
    (hF : Flock.Guarantees.FirewallSends s reqs msgs T) (P : Prog) (w : Wit)
    (E : Rv × Ωa × (Array Flock.Firewall.Sent × Bool) × V.View → Prop) :
    Flock.Guarantees.fwRealPr rows pub s reqs msgs V prover P w E =
      avg (realAt Flock.Guarantees.commitBytes (Flock.Guarantees.regLeaves rows pub) nh msgs
        (fun P w r a l p y q => V.view prover (P, w, r, a, (fun i => Flock.Hm96.Default512.treeLeaf (l i)), p, y) q)
        P w (fun t => E (t.1, t.2.1, (T t.2.2.1, true), t.2.2.2))) := by
  unfold Flock.Guarantees.fwRealPr
  refine congrArg avg (funext fun r => ?_)
  unfold realAt
  rw [← ZK.prCoin_congr_equiv
    (((Equiv.refl (Ωa × Ωp)).prodCongr (ZkView.saltsEquiv nh)).prodCongr (Equiv.refl V.Ωo))]
  refine congrArg prCoin (funext fun z => ?_)
  have hrun : Flock.Firewall.run s (reqs P w r z.1.1.1 z.1.1.2
        (Flock.Guarantees.firewallStrings (msgs P w r z.1.1.1 z.1.1.2)
          fun i => FlockSoundness.Audit.Law.streamOf (z.1.2 i)))
      (List.ofFn fun i => FlockSoundness.Audit.Law.streamOf (z.1.2 i)) =
        (T (encAt Flock.Guarantees.commitBytes (msgs P w r z.1.1.1 z.1.1.2) fun i => ZkView.saltOfBytes (z.1.2 i)),
          true) :=
    (hF P w r z.1.1.1 z.1.1.2 fun i => FlockSoundness.Audit.Law.streamOf (z.1.2 i)).trans
      (congrArg (fun f => (T f, true)) (firewallStrings_os (msgs P w r z.1.1.1 z.1.1.2) z.1.2))
  exact congrArg (fun u => E (Flock.Guarantees.regLeaves rows pub P r, z.1.1.1, u,
    V.view prover (OuterZK.hctxAt msgs P w r z.1.1.1 z.1.1.2 fun i => ZkView.saltOfBytes (z.1.2 i)) z.2)) hrun

end FlockSoundness.Discharge.Recursion.FirewallZK

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **(ii) at the firewall and V*'s code.** -/
theorem RecursiveFirewallPrivate : Flock.Guarantees.RecursiveFirewallPrivate := by
  intro Out Prog Wit Rv Ωa Ωp Pub _ _ _ _ s nr rows pub honest nh reqs msgs T V pubS stopped p₀ o P w hF hpub hC
    hH hw E
  have hT1 := CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash
  have : Nonempty Ωa := inferInstance
  have : Nonempty Ωp := inferInstance
  let x₀ : OuterZK.HCtx Prog Wit Ωa Ωp nr nh :=
    (P, w, fun _ => 0, default, fun _ => ByteArray.empty, default, fun _ => 0)
  have : ∀ j, Nonempty (Fin (V.NR j)) := fun j =>
    ⟨⟨0, V.nr_pos j (V.pubOf x₀) (OuterZK.commitStrs (V.m₂ x₀) fun _ => 0)⟩⟩
  have : ∀ j, Nonempty (Fin (V.NS j)) := fun j =>
    ⟨⟨0, V.ns_pos j (V.pubOf x₀) (OuterZK.commitStrs (V.m₂ x₀) fun _ => 0)⟩⟩
  obtain ⟨LP, Vs, Reg', D, draw, tab, y, hCeq⟩ := hC
  have hOuter := OuterZK.vstar_outer_zk rows pub honest msgs V pubS hpub
    (OuterZK.tablesOK_of_sessions V LP Vs D draw tab y hCeq) (OuterZK.selfCheckSilent_exe stopped) hH hT1
  have key := FirewallZK.private_enc Flock.Guarantees.commitBytes (Flock.Guarantees.regLeaves rows pub)
    (regLeaves_hides rows pub hT1) honest nh msgs
    (fun P w r a l p y q =>
      V.view (OuterZK.proverExe stopped) (P, w, r, a, (fun i => Flock.Hm96.Default512.treeLeaf (l i)), p, y) q)
    (fun o v a l s => V.sim (pubS o v a fun i => Flock.Hm96.Default512.treeLeaf (l i)) s)
    hOuter hT1 p₀ o P w hw (fun t => E (t.1, t.2.1, (T t.2.2.1, true), t.2.2.2))
  rw [FirewallZK.fwRealPr_eq rows pub s reqs msgs T V (OuterZK.proverExe stopped) hF P w E]
  exact key

end Flock.SecurityProofs
