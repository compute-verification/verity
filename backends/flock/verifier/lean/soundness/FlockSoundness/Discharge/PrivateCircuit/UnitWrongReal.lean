import FlockSoundness.Discharge.PrivateCircuit.UnitDefs
import FlockSoundness.Discharge.PrivateCircuit.UnitRows
import FlockSoundness.Discharge.PrivateCircuit.InstanceReal

/-!
# `UnitWrong` at the universal unit's session

* `unitWrong_trivial`: `UnitWrong P progOf xOf yOf` holds of every partition for some `yOf`, so as a hypothesis with
  free reads it carries nothing about the circuit.
* `unitWrong_universal`: at the session's partition of a parsed circuit of one copy per VU (`c.upv = 1`) whose net
  computes `UniversalUnit_v1` (`UniversalComputes`), `UnitWrong` holds at the unit's own reads (`uProg`, `uX`, `uY`) at
  every assignment that holds 1 at the constant's gate (`UnitWrongOn`).
* `private_circuit_sound_universal`: `private_circuit_sound_real` at those reads, with `UniversalComputes` and the
  constant's gate holding 1 in the committed values of accepted outcomes (`hone`) for `UnitWrong`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg Binding
open scoped ENNReal

/-- **`UnitWrong` with free reads holds of every partition**: given the program and data reads, the claimed outputs that
miss the run exactly at the partition's wrong units make it true. -/
theorem unitWrong_trivial {C : Audit.Circuit} {c : Universal.Cls} (P : Audit.Partition C (Universal.units c))
    (progOf : (Fin C.N → Bool) → c.Prog) (xOf : (Fin C.N → Bool) → ℕ → ℕ → Bool) :
    ∃ yOf, UnitWrong P progOf xOf yOf := by
  classical
  refine ⟨fun v i => if h : i < Universal.units c then
    (if (⟨i, h⟩ : Fin (Universal.units c)) ∈ P.wrong v then true :: Universal.run c (progOf v) i (xOf v i)
      else Universal.run c (progOf v) i (xOf v i)) else [], fun v i => ?_⟩
  simp only [i.isLt, ↓reduceDIte, Fin.eta]
  split_ifs with hw
  · simp only [hw, ne_eq, true_iff]
    exact fun h => List.cons_ne_self _ _ h.symm
  · simp [hw]

/-- **The session's partition of the universal unit is the interpreter's, at its own reads.** A circuit `c` of one copy
per VU (`hupv`) whose net computes `UniversalUnit_v1{G, nIn, nOut}` at the arguments' columns `L` (`hcomp`): at every
assignment `v` of the session's program's gates with `v oneG = true`, unit `i` of `units cls` is wrong exactly when
the universal unit running the program `v` holds at the units' input gates (`uProg L v`), on unit `i`'s data
(`uX L v i`), misses its claimed outputs (`uY cls v i`). -/
theorem unitWrong_universal {c : Flock.Circuit} (hU : Built c.unit) (cls : Universal.Cls) (hupv : c.upv = 1)
    (L : UIns cls (FlockSoundness.Rows.ofNet hU).nIn) (hcomp : UniversalComputes c hU cls L) :
    UnitWrongOn ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      (fun v => v Layout.oneG = true) (uProg L) (uX L) (uY cls) := by
  intro v hv i
  have eR : (keyProg c hU (Universal.units cls)).rows i = (FlockSoundness.Rows.ofNet hU).stack 1 := by
    rw [show (1 : ℕ) = c.upv from hupv.symm]
    exact rows_circuitProg c hU _ (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) i
  have eN : ((keyProg c hU (Universal.units cls)).rows i).nIn = (FlockSoundness.Rows.ofNet hU).nIn := by
    rw [congrArg FlockSoundness.Rows.nIn eR]; show 1 * _ = _; omega
  have eC : ((keyProg c hU (Universal.units cls)).rows i).nComp = (FlockSoundness.Rows.ofNet hU).nComp + 1 := by
    rw [congrArg FlockSoundness.Rows.nComp eR]; show 1 * _ = _; omega
  have hsize : outBits c = 16 * (c.leavesOut.getD 0 #[]).size := rfl
  have key := mem_wrong_iff_stack1
    (Const.Prog.instC (keyProg c hU (Universal.units cls)) (Layout.outsOf c hU (Universal.units cls)) i) eR v
    (uIn v i)
    (fun j x hx => by
      rw [Const.Prog.instC_wire_input _ _ _ x (by rw [eN, hx]; exact j.isLt)]
      simp only [uIn, hx])
    (fun x hx => by
      have e : x = ((keyProg c hU (Universal.units cls)).rows i).one := Fin.ext (by simp only [FlockSoundness.Rows.one]; exact hx)
      rw [e, Const.Prog.instC_wire_one]
      have h0 : (Const.Prog.oneAt (keyProg c hU (Universal.units cls)) i).val =
          (Placed.oneGate (Universal.units cls) (Layout.keyB c hU)).val :=
        Layout.oneAt_uniform_val (k := Placed.inGates (Universal.units cls) (Layout.keyB c hU))
          (R := Placed.circuitRows c hU)
          (ins := Placed.insOf (Universal.units cls) (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
          (one := Placed.oneGate (Universal.units cls) (Layout.keyB c hU))
          (hone := Placed.insOf_ne_one (Universal.units cls) (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
          (Universal.units cls) i
      have e1 : Const.Prog.oneAt (keyProg c hU (Universal.units cls)) i = Layout.oneG := Fin.ext h0
      rw [e1]
      exact hv)
    (colGate c hU (Universal.units cls) i) (fun x h => by simp only [colGate, h, ↓reduceDIte])
    (outBits c) (fun k => Layout.outCol c (k / 16) (k % 16)) (outStack c hU) hcomp.out_comp
    (fun k _ => by simp only [outStack, hupv])
    (fun i' => by
      constructor
      · intro hmem
        have hio : (Const.Prog.instC (keyProg c hU (Universal.units cls)) (Layout.outsOf c hU (Universal.units cls))
            i).wire (((keyProg c hU (Universal.units cls)).rows i).comp i') ∈
            ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls))).io i :=
          Finset.mem_filter.2 ⟨hmem, Or.inl ((Const.Prog.instC _ _ i).mem i')⟩
        obtain ⟨u', j, t, hu', hj, ht, h1, h2, hxv⟩ :=
          Layout.outsOut_outsOf c hU (Universal.units cls) i _ hio ((Const.Prog.instC _ _ i).mem i') _ rfl
        have hu0 : u' = 0 := by omega
        subst hu0
        refine ⟨16 * j + t, by rw [hsize]; omega, ?_⟩
        have hd : (16 * j + t) / 16 = j := by omega
        have hm : (16 * j + t) % 16 = t := by omega
        simp only [outStack, hd, hm]
        exact hxv
      · rintro ⟨k, hk, hik⟩
        obtain ⟨h1, h2⟩ := hcomp.out_comp k hk
        have hmem : (Const.Prog.instC (keyProg c hU (Universal.units cls)) (Layout.outsOf c hU (Universal.units cls))
            i).wire (((keyProg c hU (Universal.units cls)).rows i).comp i') ∈ Layout.outsOf c hU (Universal.units cls) :=
          (@Finset.mem_filter _ _ (Classical.decPred _) _ _).2 ⟨Finset.mem_univ _, i,
            ((keyProg c hU (Universal.units cls)).rows i).comp i', rfl, 0, k / 16, k % 16,
            by rw [hupv]; omega, by rw [hsize] at hk; omega, Nat.mod_lt _ (by omega), h1, h2, hik.trans rfl⟩
        exact Audit.Partition.mem_committed_of_output _ hmem)
  rw [key]
  have hi : i.val < cls.N := i.isLt
  have hdata : (fun j : Fin cls.nIn => uX L v i j.val) = L.data (uIn v i) :=
    funext fun j => by simp only [uX, i.isLt, j.isLt, ↓reduceDIte]
  simp only [Universal.run, uY, hi, i.isLt, ↓reduceDIte, Fin.eta, hdata]
  change _ ↔ List.ofFn (Universal.interp (L.prog (uIn v i)) (L.data (uIn v i))) ≠ _
  rw [ne_claim_iff _ _ cls.nOut (by simp) (by simp only [uRet, List.length_ofFn]; exact hcomp.out_le)]
  have hrhs : ∀ k (hk : k < outBits c), rowsEvalN (FlockSoundness.Rows.ofNet hU) (uIn v i) true
      (Layout.outCol c (k / 16) (k % 16)) =
      if h : k < cls.nOut then Universal.interp (L.prog (uIn v i)) (L.data (uIn v i)) ⟨k, h⟩ else false := by
    intro k hk
    have hw : Layout.outCol c (k / 16) (k % 16) < (FlockSoundness.Rows.ofNet hU).w := by
      have := (hcomp.out_comp k hk).2; simp only [FlockSoundness.Rows.w]; omega
    unfold rowsEvalN
    simp only [hw, ↓reduceDIte]
    exact hcomp.computes _ k hk
  simp only [uRet, List.length_ofFn, List.getElem_ofFn]
  constructor
  · rintro ⟨k, hk, hne⟩
    refine ⟨k, hk, ?_⟩
    rw [hrhs k hk] at hne
    convert hne using 1
  · rintro ⟨k, hk, hne⟩
    refine ⟨k, hk, ?_⟩
    rw [hrhs k hk]
    convert hne using 1

variable {mPts : ℕ} {Reg : Type}

/-- **Protocol 2 is sound at the universal unit's `--zk --registered` session, at the unit's own reads.**
`private_circuit_sound_real` with the program, data and claimed outputs the committed values hold at each unit's own
gates (`uProg`, `uX`, `uY`), for a circuit of one copy per VU (`hupv`) whose net computes `UniversalUnit_v1` at the
arguments' columns `L` (`hcomp`), and whose committed values at accepted outcomes hold 1 at the constant's gate
(`hone`), in place of `UnitWrong`. -/
theorem private_circuit_sound_universal {cls : Universal.Cls}
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin (Universal.units cls)) → Lean.Json) {S₀ : Finset (Fin (Universal.units cls))}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin (Universal.units cls)) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
    {m : ℕ} (σs : Fin (Universal.units cls) → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src) (hA6 : A6Z I dj wr St Kd pop Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') {qF qS : ℝ}
    (C : Universal.Circ) (hC : Universal.fits C cls)
    (hupv : c.upv = 1) (L : UIns cls (FlockSoundness.Rows.ofNet hU).nIn) (hcomp : UniversalComputes c hU cls L)
    (rd : Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsSh ht dj y₀ hc hU hpub hscope))
    {rt : Flock.Registered.Port} {L' : ℕ} (ps : ProgSites rd rt L')
    (row : cls.Prog → Fin L' → (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
    (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C cls) i)
    (τ : Strategy (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (hCR : (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).CR τ)
    (hone : ∀ o, LiveAcceptsZKR I own o →
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).vals τ o
        Layout.oneG = true)
    (valPair : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ (uProg L)
      (Universal.pad C cls) (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => o.2.1) valPair)
    (K₀ : ℕ) (cost : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      (fun o => some (progPairOn ps
        (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
        (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => regPaths c pub o.2.2 rd.port) valPair o)) cost q) :
    let s := realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS
    prob (fun o => LiveAcceptsZKR I own o ∧
        (K₀ ≤ wrongAgree C cls (uProg L (s.vals τ o)) (uX L (s.vals τ o)) (uY cls (s.vals τ o)) ∨
          (∃ u ∈ o.2.1, uProg L (s.vals τ o) u ≠ Universal.pad C cls u) ∨
          ∃ i, ownAt own c pub o.2.2 rd.port rd.readAt (ps.site i) ∧
            (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).regLeaf o.1 (ps.site i) ≠
              (ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf
                (row (Universal.pad C cls) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    ∀ K₁, ForgedMiss s τ (fun o => o.2.1) (fun o => forged C cls (uProg L (s.vals τ o))) K₁ →
      prob (fun o => LiveAcceptsZKR I own o ∧
          K₀ + K₁ ≤ wrong (Universal.eval C) (Universal.size C) (uX L (s.vals τ o)) (uY cls (s.vals τ o))) s.G τ
        ≤ s.Lm.miss K₀ + s.Lm.miss K₁ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  intro s
  classical
  obtain ⟨yT, hyT⟩ := unitWrong_trivial
    ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls))) (uProg L) (uX L)
  have hunit : UnitWrong ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      (uProg L) (uX L) (fun w => if w Layout.oneG = true then uY cls w else yT w) := by
    intro w i
    by_cases hw : w Layout.oneG = true
    · simp only [hw, ↓reduceIte]
      exact unitWrong_universal hU cls hupv L hcomp w hw i
    · simp only [hw]
      exact hyT w i
  have h := private_circuit_sound_real I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St hS Kd pop Lb src hA3 hA6
    hk hRw hM hρ hr ht' C hC (uProg L) (uX L) _ hunit rd ps row hreg τ hCR valPair hstep K₀ cost q hcost hBind
  have hy : ∀ o, LiveAcceptsZKR I own o →
      (if s.vals τ o Layout.oneG = true then uY cls (s.vals τ o) else yT (s.vals τ o)) = uY cls (s.vals τ o) :=
    fun o ho => by
      split_ifs with h
      · rfl
      · exact absurd (hone o ho) h
  refine ⟨le_of_eq_of_le (congrArg (fun E => prob E s.G τ)
    (funext fun o => propext (and_congr_right fun ho => by rw [hy o ho]))) h.1, fun K₁ hF => ?_⟩
  exact le_of_eq_of_le (congrArg (fun E => prob E s.G τ)
    (funext fun o => propext (and_congr_right fun ho => by rw [hy o ho]))) (h.2 K₁ hF)

end FlockSoundness.Discharge.PrivateCircuit
