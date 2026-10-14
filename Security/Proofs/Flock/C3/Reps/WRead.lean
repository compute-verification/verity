import Proofs.Flock.C3.Reps.SentKeys

/-!
# `rep₀`'s `w` read off the layers: `WDecodes`, and each part's `hres` at it

`wRead` reads element `i` of rep `ρ`'s `w` where a part `p` has it at position `a` of its row ports, each of its 128
bits at a column a computed row of the part's session reads and a row-port bit keys (`WLoc`, one chosen by `wLoc`):
bit `b` is copy 0's at column `128 a + b` of the part's unit for `ρ` (`bitAt`), and 0 where no part has it.

* `wDecodes_of_read`: `WDecodes` at `wLoc`'s session, unit and columns, for maps whose `rep₀` reads `w` so, given each
  session's `upv > 0` (`upv_of_outer`, from any outer proof).
* `resZeroAtLH_wRead`: each part's `ResZeroAtLH` at one `w` holds at `wRead`'s, where an element a part reads a bit of
  is read whole at some part (`ResReadCol`): at a registered bit a good layer holds the registrant's value
  (`bitAt_of_regBit`), which is `w`'s in every part.

Not every column of a part's row ports is read: a part's row ports hold the whole of each port of `w` its residuals
touch, and first-fit packing can leave elements of a touched port that none of the part's residuals reads (at V*'s
shape `m = 25, k = 22, c = 8`, part 1 reads no bit of `m76`'s first four elements), so `ResReadCol` asks only of the
bits some part reads.
-/

namespace FlockC3.Reps

noncomputable section

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (partsOf nwOf wIOf nsOf nvOf rowColOf)
open FlockVBridge.OfVStar (ReadCol ResZeroAtLH rowCol partRep)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.PartS (wIdx)

/-- Column `c` of `Q`'s unit is an input column a computed row reads, keyed at copy 0 by bit `kq` of a row port `q`. -/
def KeyedCol (Q : VStmt) (c : ℕ) : Prop :=
  c < (Rows.ofNet Q.hU).nIn ∧ ReadCol (Rows.ofNet Q.hU) c ∧
    ∃ (q : Fin Q.c.ports.size) (kq : ℕ), kq < Layout.portRowW Q.c q ∧
      Aliased.srcKey Q.c (0 * (Rows.ofNet Q.hU).nIn + c) = .msg q.val (kq / 8) (kq % 8)

/-- Position `a` of part `p`'s row ports (`pa = (p, a)`) is element `i` of `w`, its bits at keyed columns. -/
def WLoc {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (i : ℕ) (pa : ℕ × ℕ) : Prop :=
  pa.1 < (partsOf sh).size ∧ (wIOf sh pa.1)[pa.2]? = some i ∧
    ∀ b < 128, KeyedCol (Vs (sess pa.1)) (rowColOf sh pa.1 (128 * pa.2 + b))

open Classical in
/-- Where `w`'s element `i` is read: a part and position holding it (`WLoc`). -/
def wLoc {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (i : ℕ) : ℕ × ℕ :=
  if h : ∃ pa, WLoc Vs sh sess i pa then Classical.choose h else (0, 0)

/-- The session element `i` of `w` is read in. -/
def wSes {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (i : ℕ) : Fin S :=
  sess (wLoc Vs sh sess i).1

/-- Rep `ρ`'s unit element `i` of `w` is read in. -/
def wUn {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S)
    (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) (ρ : Fin 2) (i : ℕ) : Fin (Vs (wSes Vs sh sess i)).nv :=
  un (wLoc Vs sh sess i).1 ρ

/-- The columns element `i` of `w` is read at, none where no part holds it. -/
def wCols {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (i c : ℕ) : Prop :=
  (∃ pa, WLoc Vs sh sess i pa) ∧ ∃ b < 128, c = rowColOf sh (wLoc Vs sh sess i).1 (128 * (wLoc Vs sh sess i).2 + b)

open Classical in
/-- **Rep `ρ`'s `w` read off the layers `Lb`**: element `i`'s bit `b` is the layer's at `wLoc`'s column. -/
def wRead {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S)
    (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) (Lb : (j : Fin S) → (Vs j).Bits) (ρ : Fin 2) : Array F128 :=
  Array.ofFn (n := nwOf sh) fun i => if ∃ pa, WLoc Vs sh sess i.val pa then
    FlockVBridge.ofBits fun b => FlockVBridge.Family.bitAt (Vs (wSes Vs sh sess i.val)) (Lb (wSes Vs sh sess i.val))
      (wUn Vs sh sess un ρ i.val) (rowColOf sh (wLoc Vs sh sess i.val).1 (128 * (wLoc Vs sh sess i.val).2 + b))
  else 0

/-- **Hypothesis on V*'s circuits: an element of `w` a part reads is read whole at some part**: where a computed row of
part `p`'s session reads bit `b` of position `a` of its row ports (`ReadCol`), some part `p₀` holds the same element at
a position `a₀` all 128 of whose columns a computed row of `p₀`'s session reads. -/
def ResReadCol {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) : Prop :=
  ∀ p a b, p < (partsOf sh).size → (_ : a < (wIOf sh p).size) → b < 128 →
    ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOf sh p (128 * a + b)) →
    ∃ p₀ a₀, p₀ < (partsOf sh).size ∧ ∃ (_ : a₀ < (wIOf sh p₀).size), (wIOf sh p₀)[a₀] = (wIOf sh p)[a] ∧
      ∀ b₀ < 128, ReadCol (Rows.ofNet (Vs (sess p₀)).hU) (rowColOf sh p₀ (128 * a₀ + b₀))

section read

variable {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S}
  {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv}

theorem wLoc_spec {i : ℕ} (h : ∃ pa, WLoc Vs sh sess i pa) : WLoc Vs sh sess i (wLoc Vs sh sess i) := by
  unfold wLoc
  rw [dite_eq_left h]
  exact Classical.choose_spec h

theorem wRead_size (Lb : (j : Fin S) → (Vs j).Bits) (ρ : Fin 2) : (wRead Vs sh sess un Lb ρ).size = nwOf sh := by
  rw [wRead, Array.size_ofFn]

open Classical in
theorem getElem!_wRead (Lb : (j : Fin S) → (Vs j).Bits) (ρ : Fin 2) {i : ℕ} (hi : i < nwOf sh) :
    (wRead Vs sh sess un Lb ρ)[i]! = if ∃ pa, WLoc Vs sh sess i pa then
      FlockVBridge.ofBits fun b => FlockVBridge.Family.bitAt (Vs (wSes Vs sh sess i)) (Lb (wSes Vs sh sess i))
        (wUn Vs sh sess un ρ i) (rowColOf sh (wLoc Vs sh sess i).1 (128 * (wLoc Vs sh sess i).2 + b))
    else 0 := by
  rw [getElem!_pos (wRead Vs sh sess un Lb ρ) i (by rw [wRead_size]; exact hi)]
  simp only [wRead, Array.getElem_ofFn]

theorem el_wRead (Lb : (j : Fin S) → (Vs j).Bits) (ρ : Fin 2) {i : ℕ} (hi : i < nwOf sh)
    (hex : ∃ pa, WLoc Vs sh sess i pa) :
    Flock.RecAlgebra.el (wRead Vs sh sess un Lb ρ) i =
      FlockVBridge.ofBits fun b => FlockVBridge.Family.bitAt (Vs (wSes Vs sh sess i)) (Lb (wSes Vs sh sess i))
        (wUn Vs sh sess un ρ i) (rowColOf sh (wLoc Vs sh sess i).1 (128 * (wLoc Vs sh sess i).2 + b)) := by
  have hs : i < (wRead Vs sh sess un Lb ρ).size := by rw [wRead_size]; exact hi
  unfold Flock.RecAlgebra.el
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs, Option.getD_some,
    ← getElem!_pos (wRead Vs sh sess un Lb ρ) i hs, getElem!_wRead Lb ρ hi, ite_eq_left hex]

/-- **A statement V* is drawn over has `upv > 0`**, given any outer proof of it (`upv_pos`). -/
theorem upv_of_outer {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg') : 0 < Q.c.upv :=
  FlockVBridge.Family.upv_pos zo.hc Q.hscope

theorem bitAt_eq {Q : VStmt} {bits : Q.Bits} {u : Fin Q.nv} {col : ℕ} (hj : 0 < Q.c.upv) :
    FlockVBridge.Family.bitAt Q bits u col =
      FlockVBridge.copyVal (FlockVBridge.keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj col := by
  unfold FlockVBridge.Family.bitAt
  rw [dite_eq_left hj]

/-- **Each part's `hres` at `wRead`**: where every part `p` of rep `ρ`'s unit holds part `p` at one `w` in a good layer
(`ResZeroAtLH`), each at `w`'s registered bits, it does at `wRead`'s: element `(wIOf sh p)[a]`'s bit `b` is the
layer's at `wLoc`'s part `p₀` (`el_wRead`), which holds `w`'s bit there (`bitAt_of_regBit`), a part reading the element
whole where `p` reads the bit (`ResReadCol`), each of its columns keyed (`RegBit`). -/
theorem resZeroAtLH_wRead {Reg' : Type} (hrc : ResReadCol Vs sh sess) {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} (hA : ∀ s, (zs s).RegAgree (Lb s)) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : Flock.RecAlgebra.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecResiduals.portSizes sh) = .ok gs) (ρ : Fin 2) {w : ℕ → F128}
    {rounds : Array (Array F128)} {extras : Array Flock.Claim} {comb : Array F128}
    (h : ∀ p < gs.size, ResZeroAtLH (zs (sess p)) (Lb (sess p)) (un p ρ) 0 sh gs p w rounds extras comb) {p : ℕ}
    (hp : p < gs.size) :
    ResZeroAtLH (zs (sess p)) (Lb (sess p)) (un p ρ) 0 sh gs p (Flock.RecAlgebra.el (wRead Vs sh sess un Lb ρ))
      rounds extras comb := by
  have hP : partsOf sh = gs := FlockVBridge.Family.partsOf_eq hS hg
  have hof : ∀ p₁ (_ : p₁ < gs.size) P ns nv, Flock.RecResiduals.part sh p₁ = .ok (P, ns, nv) →
      nsOf sh p₁ = ns ∧ nvOf sh p₁ = nv ∧ wIOf sh p₁ = wIdx (Flock.RecResiduals.portSizes sh) gs[p₁].2.1 ∧
        (wIOf sh p₁).size = ns.sum := by
    intro p₁ hp₁ P ns nv hpart
    obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hS hg hp₁ hpart
    have hgp : (partsOf sh)[p₁]! = gs[p₁] := by rw [hP, getElem!_pos gs p₁ hp₁]
    have hwI : wIOf sh p₁ = wIdx (Flock.RecResiduals.portSizes sh) gs[p₁].2.1 := by rw [wIOf, hgp]
    refine ⟨by rw [nsOf, hgp, hns], by rw [nvOf, hgp, hnv], hwI, ?_⟩
    rw [hwI, FlockVBridge.OfVStar.wIdx_size, hns]
  have hloc : ∀ p₁ (_ : p₁ < gs.size) a (_ : a < (wIOf sh p₁).size) b, b < 128 →
      ReadCol (Rows.ofNet (Vs (sess p₁)).hU) (rowColOf sh p₁ (128 * a + b)) →
      0 < (Vs (sess p₁)).c.upv ∧ (wIOf sh p₁)[a] < nwOf sh ∧
      KeyedCol (Vs (sess p₁)) (rowColOf sh p₁ (128 * a + b)) ∧
        RegBit (zs (sess p₁)) (un p₁ ρ) 0 (rowColOf sh p₁ (128 * a + b))
          (FlockVBridge.bitOf (w (wIOf sh p₁)[a]) b) := by
    intro p₁ hp₁ a ha b hb hrd
    obtain ⟨_, hj, _, _, _, _, _, _, _, _, _, _, _, P, ns, nv, hpart, hw, _, _⟩ := h p₁ hp₁
    obtain ⟨hns, hnv, hwI, hsz⟩ := hof p₁ hp₁ P ns nv hpart
    have hq : 128 * a + b < 128 * ns.sum := by rw [← hsz]; omega
    rw [rowColOf, hns, hnv] at hrd ⊢
    have hreg := hw _ hq hrd
    have hval : FlockVBridge.wbit (FlockVBridge.Decode.sub (wIdx (Flock.RecResiduals.portSizes sh) gs[p₁].2.1) w)
        (128 * a + b) = FlockVBridge.bitOf (w (wIOf sh p₁)[a]) b := by
      unfold FlockVBridge.wbit
      rw [← hwI, show (128 * a + b) / 128 = a by omega, show (128 * a + b) % 128 = b by omega,
        FlockVBridge.Decode.sub_lt _ _ _ ha]
    rw [hval] at hreg
    have hi : (wIOf sh p₁)[a] < nwOf sh := by
      rw [FlockVBridge.Family.nwOf_eq hS, ← FlockVBridge.Family.part_nw hS hpart]
      exact FlockVBridge.Family.wIdx_lt (Flock.RecResiduals.portSizes sh) ((partsOf sh)[p₁]!.2.1) ha
    refine ⟨hj, hi, ⟨?_, hrd, ?_⟩, hreg⟩
    · obtain ⟨hlt, -⟩ := hreg
      exact hlt
    · obtain ⟨-, q, kq, hkq, hkey, -⟩ := hreg
      exact ⟨q, kq, hkq, hkey⟩
  have hread : ∀ p₁ (_ : p₁ < gs.size) a (_ : a < (wIOf sh p₁).size) b, b < 128 →
      ReadCol (Rows.ofNet (Vs (sess p₁)).hU) (rowColOf sh p₁ (128 * a + b)) →
      FlockVBridge.bitOf (Flock.RecAlgebra.el (wRead Vs sh sess un Lb ρ) (wIOf sh p₁)[a]) b =
        FlockVBridge.bitOf (w (wIOf sh p₁)[a]) b := by
    intro p₁ hp₁ a ha b hb hrd₁
    obtain ⟨p₂, a₂, hp₂, ha₂, he₂, hall⟩ := hrc p₁ a b (by rw [hP]; exact hp₁) ha hb hrd₁
    rw [hP] at hp₂
    have hex : ∃ pa, WLoc Vs sh sess (wIOf sh p₁)[a] pa :=
      ⟨(p₂, a₂), by rw [hP]; exact hp₂, (Array.getElem?_eq_getElem ha₂).trans (congrArg some he₂),
        fun b₁ hb₁ => (hloc p₂ hp₂ a₂ ha₂ b₁ hb₁ (hall b₁ hb₁)).2.2.1⟩
    obtain ⟨hp₀, ha₀, hkc⟩ := wLoc_spec hex
    rw [hP] at hp₀
    obtain ⟨ha₀, he₀⟩ := Array.getElem?_eq_some_iff.mp ha₀
    obtain ⟨hlt, hrd₀, -⟩ := hkc b hb
    obtain ⟨hj₀, -, -, hreg⟩ := hloc _ hp₀ _ ha₀ b hb hrd₀
    rw [el_wRead Lb ρ (hloc p₁ hp₁ a ha b hb hrd₁).2.1 hex, FlockVBridge.bitOf_ofBits _ hb]
    have hdec := FlockVBridge.Family.bitAt_of_regBit (zs _) (hA _) hreg
      (FlockVBridge.Family.ioColAt_of_readCol _ hlt hrd₀) hj₀
    rw [he₀] at hdec
    exact hdec
  obtain ⟨hpg, hj, hone, hchk, hholo, hlog, m, k, cl, hpar, sh0, hsh0, hsheq, P, ns, nv, hpart, hw, hv, hz⟩ := h p hp
  obtain ⟨hns, hnv, hwI, hsz⟩ := hof p hp P ns nv hpart
  refine ⟨hpg, hj, hone, hchk, hholo, hlog, m, k, cl, hpar, sh0, hsh0, hsheq, P, ns, nv, hpart, fun q hq hrd => ?_,
    hv, hz⟩
  have ha : q / 128 < (wIOf sh p).size := by rw [hsz]; omega
  have hrd' : ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOf sh p (128 * (q / 128) + q % 128)) := by
    rw [Nat.div_add_mod, rowColOf, hns, hnv]
    exact hrd
  have hval : FlockVBridge.wbit (FlockVBridge.Decode.sub (wIdx (Flock.RecResiduals.portSizes sh) gs[p].2.1)
      (Flock.RecAlgebra.el (wRead Vs sh sess un Lb ρ))) q =
      FlockVBridge.wbit (FlockVBridge.Decode.sub (wIdx (Flock.RecResiduals.portSizes sh) gs[p].2.1) w) q := by
    unfold FlockVBridge.wbit
    rw [← hwI, FlockVBridge.Decode.sub_lt _ _ _ ha, FlockVBridge.Decode.sub_lt _ _ _ ha]
    exact hread p hp _ ha _ (Nat.mod_lt _ (by norm_num)) hrd'
  rw [hval]
  exact hw q hq hrd

end read

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {stmt : Fin S → VStar.Stmt} {sh : Flock.Shape}

/-- **`WDecodes` where `rep₀` reads `w` off the layers** (`wRead`): element `i` at `wLoc`'s session, unit and columns,
each keyed and read (`KeyedCol`, `ioColAt_of_readCol`), and the same at layers agreeing there (`ofBits_congr`). -/
theorem wDecodes_of_read (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (holoInnerAt T CT st hp).Coin))
    (hupv : ∀ s, 0 < (Vs s).c.upv) (hM : ∀ Lb ρ, (M.rep₀ Lb ρ).w = wRead Vs sh M.sess M.un Lb ρ) :
    WDecodes M (fun _ => wSes Vs sh M.sess) (wUn Vs sh M.sess M.un) (fun _ => wCols Vs sh M.sess) := by
  intro ρ _ i _
  refine ⟨hupv _, fun c hc => ?_, fun Lb Lb' he => ?_⟩
  · obtain ⟨hex, b, hb, rfl⟩ := hc
    obtain ⟨hlt, hrd, hkey⟩ := (wLoc_spec hex).2.2 b hb
    exact ⟨hlt, FlockVBridge.Family.ioColAt_of_readCol _ hlt hrd, hkey⟩
  rw [hM, hM]
  by_cases hi : i < nwOf sh
  · rw [getElem!_wRead Lb ρ hi, getElem!_wRead Lb' ρ hi]
    by_cases hex : ∃ pa, WLoc Vs sh M.sess i pa
    · rw [ite_eq_left hex, ite_eq_left hex]
      exact FlockVBridge.OfVStar.ofBits_congr fun b hb =>
        (bitAt_eq (hupv _)).trans ((he _ ⟨hex, b, hb, rfl⟩).trans (bitAt_eq (hupv _)).symm)
    · rw [ite_eq_right hex, ite_eq_right hex]
  · have hn : ∀ Lx, (wRead Vs sh M.sess M.un Lx ρ)[i]! = default := fun Lx =>
      getElem!_neg _ i (by rw [wRead_size]; exact hi)
    rw [hn, hn]

end

end FlockC3.Reps
