import Proofs.Flock.Recursive.Firewall
import Proofs.Flock.Recursive.OuterZK
import Proofs.Flock.Soundness.Discharge.ZkView.Leaf

/-!
# Proof of (ii) at the firewall and V*'s code

`Flock.SecurityProofs.RecursiveFirewallPrivate : Flock.Guarantees.RecursiveFirewallPrivate` is
`RecursiveCircuitPrivateLeaves`' argument at the commit strings the firewall sends, where that one is at their leaves:

* `ZkView.commitString_split`, `ZkView.commitString_os`: the executable's commit string of `x` at a salt's bytes, or
  at the operating system's bytes `b`, is `commitBytes (x ⊕ M·y, c(y))` (`leafSplit` before its leaf hash).
* `FirewallZK.recursiveZK_enc`: `recursiveZK_le` with the auditor reading the commitments through any encoding `H` of
  their two halves (`encAt`, `encIdeal`; `ZK.ideal_leaves_swap` takes any), under `hash-derived-key`;
  `FirewallZK.private_enc`: `RecursiveCircuitPrivate`'s two ways there, with the registration's hiding.
* `FirewallZK.fwRealPr_eq`: the firewall's real view is that at `H = commitBytes`, by `FirewallSends` at the OS's
  bytes, taken to salts (`ZkView.saltsEquiv`).
* `RecursiveFirewallPrivate`: `private_enc` at `commitBytes`, V*'s sessions' zero knowledge (`OuterZK.vstar_outer_zk`
  at `proverExe`, its tables passing `CoinedOK` by `SessionsCoined`) and the registration's hiding
  (`regLeaves_hides`). The simulated view is `private_enc`'s by definition.
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
  dsimp only
  simp only [Pi.add_apply, bitsOf, My_apply y ⟨t, ht'⟩, bitL]
  exact xor_ind _ _

/-- **The executable's commit string at the OS's salt bytes**: at the 192 bytes `b` the operating system gives, the
commit string of `x` is `commitBytes (x ⊕ M·y, c(y))` with `y = saltOfBytes b`. -/
theorem commitString_os (x : ByteArray) (b : Fin 192 → Fin 256) :
    Hm96.Default512.commitString x (Audit.Law.streamOf b) =
      Flock.Guarantees.commitBytes (bitsOf x + My (saltOfBytes b), saltHash (saltOfBytes b)) := by
  rw [← saltBytes_saltOfBytes]
  exact commitString_split x _

end FlockSoundness.Discharge.ZkView

namespace FlockSoundness.Discharge.Recursion.FirewallZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- The commitments to `m` under the salts `y`, through an encoding `H` of their two halves. -/
def encAt {D : Type} (H : ZkView.Bits × ByteArray → D) {n : ℕ} (m : Fin n → ByteArray) (y : Fin n → ZkView.Salt) :
    Fin n → D :=
  fun i => H (OuterZK.commitStrs m y i)

/-- Ideal commitments through `H`. -/
def encIdeal {D : Type} (H : ZkView.Bits × ByteArray → D) {n : ℕ} (u : Fin n → ZkView.Bits × ZkView.Salt) :
    Fin n → D :=
  fun i => H (OuterZK.idealStrs u i)

/-- **`recursiveZK_le` through any encoding of the commitments**, under `hash-derived-key`: the outer session's
simulator at the commitments (`hOuter`), then T1 once per commitment (`ZK.ideal_leaves_swap` at `H`). -/
theorem recursiveZK_enc {D Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo]
    [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (H : ZkView.Bits × ByteArray → D) (nh : ℕ)
    (msgs : Ωa → Ωp → Fin nh → ByteArray) (view : Ωa → (Fin nh → D) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → D) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun q : Ωo => E (view a (encAt H (msgs a p) y) p y q))
        ≤ prCoin (fun s : Ωs => E (sim a (encAt H (msgs a p) y) s)) + εo)
    (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96) (E : Ωa × (Fin nh → D) × V → Prop) :
    prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2,
          view w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.1.1.2 w.1.2 w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, encIdeal H w.1.2, sim w.1.1 (encIdeal H w.1.2) w.2))
        + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
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
  -- T1 once per commitment: the simulated session reads the salts only through the commitments
  have h2 : prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2, sim w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.2)) ≤
      prCoin (fun w : ((Ωa × Ωp) × Ωs) × (Fin nh → ZkView.Bits × ZkView.Salt) =>
        E (w.1.1.1, encIdeal H w.2, sim w.1.1.1 (encIdeal H w.2) w.1.2)) + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
    refine (ZK.prCoin_congr_equiv (Flock.SecurityProofs.swapLast (Ωa × Ωp) (Fin nh → ZkView.Salt) Ωs)
      (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, encAt H (msgs w.1.1.1 w.1.1.2) w.1.2,
          sim w.1.1.1 (encAt H (msgs w.1.1.1 w.1.1.2) w.1.2) w.2))).symm.trans_le ?_
    exact ZK.ideal_leaves_swap ZkView.My ZkView.saltHash hKey H nh
      (fun (ω : (Ωa × Ωp) × Ωs) i => ZkView.bitsOf (Sha512.hash (msgs ω.1.1 ω.1.2 i)))
      (fun ω l => E (ω.1.1, l, sim ω.1.1 l ω.2))
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
    (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
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

/-- **`RecursiveCircuitPrivate` through any encoding of the commitments**: at each registration salt,
`recursiveZK_enc` (and on the event's complement, the other way), then the registration's hiding as a test with values
in `[0, 1]` (`Hides.avg_le`). -/
theorem private_enc {D Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Rr] [Nonempty Rr] [Fintype Ωa]
    [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]
    (H : ZkView.Bits × ByteArray → D) (reg : Prog → Rr → Rv) {ε : ℝ≥0∞} (hreg : PrivateCircuit.Hides reg ε)
    (honest : Out → Prog → Wit → Prop) (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → D) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → D) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (encAt H (msgs P w r a p) y) p y q))
        ≤ prCoin (fun s : Ωs => E (sim o (reg P r) a (encAt H (msgs P w r a p) y) s)) + εo)
    (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96) (p₀ : Prog) (o : Out) (P : Prog) (w : Wit)
    (hw : honest o P w) (E : Rv × Ωa × (Fin nh → D) × V → Prop) :
    avg (realAt H reg nh msgs view P w E) ≤
        avg (fun r => idealAt H nh sim o E (reg p₀ r)) + (ε + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
      avg (fun r => idealAt H nh sim o E (reg p₀ r)) ≤
        avg (realAt H reg nh msgs view P w E) + (ε + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  have hstep : ∀ r, realAt H reg nh msgs view P w E r ≤
        idealAt H nh sim o E (reg P r) + (εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
      idealAt H nh sim o E (reg P r) ≤ realAt H reg nh msgs view P w E r + (εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
    fun r => by
      have zk := fun E' => recursiveZK_enc H nh (msgs P w r) (view P w r) (sim o (reg P r)) (hOuter o P w hw r) hKey E'
      refine ⟨(zk fun t => E (reg P r, t.1, t.2.1, t.2.2)).trans (le_of_eq (add_assoc _ _ _)), ?_⟩
      exact Flock.SecurityProofs.prCoin_flip _ _
        ((zk fun t => ¬ E (reg P r, t.1, t.2.1, t.2.2)).trans (le_of_eq (add_assoc _ _ _)))
  have hf1 : ∀ v, idealAt H nh sim o E v ≤ 1 := fun v => PrivateCircuit.prCoin_le_one' _
  constructor
  · calc avg (realAt H reg nh msgs view P w E)
        ≤ avg (fun r => idealAt H nh sim o E (reg P r) + (εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ))) :=
          avg_mono fun r => (hstep r).1
      _ = avg (fun r => idealAt H nh sim o E (reg P r)) + (εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
          rw [avg_add, avg_const]
      _ ≤ (avg (fun r => idealAt H nh sim o E (reg p₀ r)) + ε) + (εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
          add_le_add (hreg.avg_le P p₀ (idealAt H nh sim o E) hf1) le_rfl
      _ = avg (fun r => idealAt H nh sim o E (reg p₀ r)) + (ε + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) := by ring
  · calc avg (fun r => idealAt H nh sim o E (reg p₀ r))
        ≤ avg (fun r => idealAt H nh sim o E (reg P r)) + ε := hreg.avg_le p₀ P (idealAt H nh sim o E) hf1
      _ ≤ avg (fun r => realAt H reg nh msgs view P w E r + (εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ))) + ε :=
          add_le_add (avg_mono fun r => (hstep r).2) le_rfl
      _ = avg (realAt H reg nh msgs view P w E) + (ε + εo + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
          rw [avg_add, avg_const]; ring

/-- **The firewall's real view is the commit strings' view**: under `FirewallSends`, what the firewall sends at the
OS's bytes `b` is the schedule's transcript filled with `commitBytes` of the commitments at the salts `saltOfBytes b`
(`ZkView.commitString_os`), and the bytes are uniform exactly when those salts are (`ZkView.saltsEquiv`). V*'s
sessions read the commitments' leaves, `Default512.treeLeaf` of their commit strings. -/
theorem fwRealPr_eq {Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Fintype Ωp] {nr nh : ℕ}
    (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → List ByteArray)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent)
    (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub) (prover : OuterZK.TableProver)
    (hF : Flock.Guarantees.FirewallSends s reqs msgs T) (P : Prog) (w : Wit)
    (E : Rv × Ωa × (Array Flock.Firewall.Sent × Bool) × V.View → Prop) :
    Flock.Guarantees.fwRealPr rows pub s reqs msgs V prover P w E =
      avg (realAt Flock.Guarantees.commitBytes (Flock.Guarantees.regLeaves rows pub) nh msgs
        (fun P w r a l p y q => V.view prover (P, w, r, a, (fun i => Flock.Hm96.Default512.treeLeaf (l i)), p, y) q)
        P w (fun t => E (t.1, t.2.1, (T t.2.2.1, true), t.2.2.2))) := by
  simp only [Flock.Guarantees.fwRealPr, realAt]
  refine congrArg avg (funext fun r => ?_)
  rw [← ZK.prCoin_congr_equiv
    (((Equiv.refl (Ωa × Ωp)).prodCongr (ZkView.saltsEquiv nh)).prodCongr (Equiv.refl V.Ωo))]
  refine congrArg prCoin (funext fun z => ?_)
  have hrun : Flock.Firewall.run s (reqs P w r z.1.1.1 z.1.1.2)
      (List.ofFn fun i => FlockSoundness.Audit.Law.streamOf (z.1.2 i)) =
        (T (encAt Flock.Guarantees.commitBytes (msgs P w r z.1.1.1 z.1.1.2) fun i => ZkView.saltOfBytes (z.1.2 i)),
          true) := by
    rw [hF]
    congr 2
    funext i
    exact ZkView.commitString_os _ _
  rw [hrun]
  rfl

end FlockSoundness.Discharge.Recursion.FirewallZK

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **(ii) at the firewall and V*'s code.** -/
theorem RecursiveFirewallPrivate : Flock.Guarantees.RecursiveFirewallPrivate := by
  intro Out Prog Wit Rv Ωa Ωp Pub _ _ _ _ hKey s nr rows pub honest nh reqs msgs T V pubS stopped p₀ o P w hF hpub hC
    hH hw E
  have : Nonempty Ωa := inferInstance
  have : Nonempty Ωp := inferInstance
  let x₀ : OuterZK.HCtx Prog Wit Ωa Ωp nr nh :=
    (P, w, fun _ => 0, default, fun _ => ByteArray.empty, default, fun _ => 0)
  have : ∀ j, Nonempty (Fin (V.NR j)) := fun j =>
    ⟨⟨0, V.nr_pos j (V.pubOf x₀) (OuterZK.commitStrs (V.m₂ x₀) fun _ => 0)⟩⟩
  have : ∀ j, Nonempty (Fin (V.NS j)) := fun j =>
    ⟨⟨0, V.ns_pos j (V.pubOf x₀) (OuterZK.commitStrs (V.m₂ x₀) fun _ => 0)⟩⟩
  obtain ⟨LP, Vs, Reg', D, draw, y, hCeq⟩ := hC
  have hOuter := OuterZK.vstar_outer_zk rows pub honest msgs V pubS hpub
    (OuterZK.tablesOK_of_sessions V LP Vs D draw y hCeq) (OuterZK.selfCheckSilent_exe stopped) hH hKey
  have key := FirewallZK.private_enc Flock.Guarantees.commitBytes (Flock.Guarantees.regLeaves rows pub)
    (regLeaves_hides rows pub hKey) honest nh msgs
    (fun P w r a l p y q =>
      V.view (OuterZK.proverExe stopped) (P, w, r, a, (fun i => Flock.Hm96.Default512.treeLeaf (l i)), p, y) q)
    (fun o v a l s => V.sim (pubS o v a fun i => Flock.Hm96.Default512.treeLeaf (l i)) s)
    hOuter hKey p₀ o P w hw (fun t => E (t.1, t.2.1, (T t.2.2.1, true), t.2.2.2))
  rw [FirewallZK.fwRealPr_eq rows pub s reqs msgs T V (OuterZK.proverExe stopped) hF P w E]
  exact key

end Flock.SecurityProofs
