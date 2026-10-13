import Proofs.Flock.C3.Reps.WRead
import Proofs.Flock.C3.Reps.ReadWhole
import Proofs.Flock.C3.Opening.Session

/-!
# R's opening's `w` read off the layers: `wReadO`, and each part's `OpeningZeroAtL` at it

As `wRead` reads `rep₀`'s `w` off the `rec-residuals` parts, `wReadO` reads R's opening's off its `rec-opening` parts
(`RecOpening.part`, `RecResiduals.groups` of the opening's S): element `i`'s bit `b` is copy 0's at column `128 a + b` of
the unit of a part `p` holding it at position `a` of its row ports, all 128 of whose columns a computed row reads and a
row-port bit keys (`WLocO`, one chosen by `wLocO`), and 0 where no part has it so.

* `partsOfO`, `wIOfO`, `nsOfO`, `nvOfO`, `rowColOfO`: the opening's parts, as `partsOf` … `rowColOf` are V*'s algebra's;
  `structureOf_nwO`, `part_nwO`: the row ports' elements are w's `nw`.
* `bitOf_wReadO`, `openingZeroAtL_wReadO`: each part's `OpeningZeroAtL` at one `w` holds at `wReadO`'s: a bit a part
  reads is of an element the part reads whole (S's `openReadCol_whole`, from the part's `RecOpening.check`), so some
  part has it so (`WLocO`), and there a good layer holds the registrant's value (`bitAt_of_regBit`), which is `w`'s in
  every part.
* `el_wReadO_acc`, `sessionAt_wReadO`: `wReadO` is `w` at each running-sum element a part holds, which the part reads
  whole (`readCol_acc`), so `SessionAt` at one `w` with every part at copy 0 holds at `wReadO`'s where some part holds
  each running sum (`AccHeldO`).

At the opening's shape not every element a part holds is read (at `m = 22`, `m24`'s first four elements are held and
never read), so `wReadO` need not be `w` there, and `OpeningZeroAtL` asks nothing of those columns.
-/

namespace FlockC3.Reps

noncomputable section

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.OfVStar (ReadCol OpeningZeroAtL rowCol partRep part_eqOpening)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.PartS (wIdx)

/-! ## The opening's parts -/

/-- The verifier's opening parts at a shape (`groups` of the opening's S), none where it refuses. -/
def partsOfO (sh : Flock.Shape) : Array (Array ℕ × Array ℕ × Array ℕ) :=
  match Flock.RecOpening.structureOf sh with
  | .ok S => match Flock.RecResiduals.groups S (Flock.RecOpening.portSizes sh) with
    | .ok gs => gs
    | .error _ => #[]
  | .error _ => #[]

/-- Opening part `p`'s `w` index (`wIdx` of its row ports). -/
def wIOfO (sh : Flock.Shape) (p : ℕ) : Array ℕ := wIdx (Flock.RecOpening.portSizes sh) (partsOfO sh)[p]!.2.1

/-- Opening part `p`'s row ports' sizes. -/
def nsOfO (sh : Flock.Shape) (p : ℕ) : List ℕ := (partsOfO sh)[p]!.2.1.toList.map ((Flock.RecOpening.portSizes sh)[·]!)

/-- Opening part `p`'s number of `v` elements. -/
def nvOfO (sh : Flock.Shape) (p : ℕ) : ℕ := (partsOfO sh)[p]!.2.2.size

/-- The input column of opening part `p`'s unit holding bit `q` of its row ports. -/
def rowColOfO (sh : Flock.Shape) (p q : ℕ) : ℕ := rowCol (partRep (nsOfO sh p) (nvOfO sh p)).ports 0 q

theorem partsOfO_eq {sh : Flock.Shape} {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs) : partsOfO sh = gs := by
  unfold partsOfO
  rw [hS]
  dsimp only
  rw [hg]

/-- The opening's S has w's `nw` elements (`build_okO`). -/
theorem structureOf_nwO {sh : Flock.Shape} {GS : Flock.RecAlgebra.GfS} (h : Flock.RecOpening.structureOf sh = .ok GS) :
    GS.nw = Flock.RecOpening.nw sh := by
  unfold Flock.RecOpening.structureOf at h
  obtain ⟨lins, -, h⟩ := FlockVBridge.PartS.bind_ok h
  obtain ⟨_, _, -, rfl⟩ := FlockC3.Opening.build_okO h
  rfl

/-- The opening's row ports' total size is S's `nw`, once `part` accepted. -/
theorem part_nwO {sh : Flock.Shape} {GS : Flock.RecAlgebra.GfS} (hS : Flock.RecOpening.structureOf sh = .ok GS) {p : ℕ}
    {x : Flock.RecOpen.GfS × List ℕ × ℕ} (h : Flock.RecOpening.part sh p = .ok x) :
    (Flock.RecOpening.portSizes sh).foldl (· + ·) 0 = GS.nw := by
  unfold Flock.RecOpening.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hS] at hS0
  cases hS0
  dsimp only at h
  split at h
  · cases h
  · rename_i hne
    simpa [bne_iff_ne] using hne

/-! ## `w` read off the opening's parts -/

/-- Position `a` of opening part `p`'s row ports (`pa = (p, a)`) is element `i` of `w`, its bits at keyed columns. -/
def WLocO {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (i : ℕ) (pa : ℕ × ℕ) : Prop :=
  pa.1 < (partsOfO sh).size ∧ (wIOfO sh pa.1)[pa.2]? = some i ∧
    ∀ b < 128, KeyedCol (Vs (sess pa.1)) (rowColOfO sh pa.1 (128 * pa.2 + b))

open Classical in
/-- Where the opening's element `i` of `w` is read: a part and position holding it (`WLocO`). -/
def wLocO {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (i : ℕ) : ℕ × ℕ :=
  if h : ∃ pa, WLocO Vs sh sess i pa then Classical.choose h else (0, 0)

/-- The session the opening's element `i` of `w` is read in. -/
def wSesO {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (i : ℕ) : Fin S :=
  sess (wLocO Vs sh sess i).1

/-- The unit the opening's element `i` of `w` is read in. -/
def wUnO {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (un : (p : ℕ) → Fin (Vs (sess p)).nv)
    (i : ℕ) : Fin (Vs (wSesO Vs sh sess i)).nv :=
  un (wLocO Vs sh sess i).1

open Classical in
/-- **R's opening's `w` read off the layers `Lb`**: element `i`'s bit `b` is the layer's at `wLocO`'s column. -/
def wReadO {S : ℕ} (Vs : Fin S → VStmt) (sh : Flock.Shape) (sess : ℕ → Fin S) (un : (p : ℕ) → Fin (Vs (sess p)).nv)
    (Lb : (j : Fin S) → (Vs j).Bits) : Array F128 :=
  Array.ofFn (n := Flock.RecOpening.nw sh) fun i => if ∃ pa, WLocO Vs sh sess i.val pa then
    FlockVBridge.ofBits fun b => FlockVBridge.Family.bitAt (Vs (wSesO Vs sh sess i.val)) (Lb (wSesO Vs sh sess i.val))
      (wUnO Vs sh sess un i.val) (rowColOfO sh (wLocO Vs sh sess i.val).1 (128 * (wLocO Vs sh sess i.val).2 + b))
  else 0

section read

variable {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S} {un : (p : ℕ) → Fin (Vs (sess p)).nv}

theorem wLocO_spec {i : ℕ} (h : ∃ pa, WLocO Vs sh sess i pa) : WLocO Vs sh sess i (wLocO Vs sh sess i) := by
  unfold wLocO
  rw [dite_eq_left h]
  exact Classical.choose_spec h

theorem wReadO_size (Lb : (j : Fin S) → (Vs j).Bits) : (wReadO Vs sh sess un Lb).size = Flock.RecOpening.nw sh := by
  rw [wReadO, Array.size_ofFn]

open Classical in
theorem getElem!_wReadO (Lb : (j : Fin S) → (Vs j).Bits) {i : ℕ} (hi : i < Flock.RecOpening.nw sh) :
    (wReadO Vs sh sess un Lb)[i]! = if ∃ pa, WLocO Vs sh sess i pa then
      FlockVBridge.ofBits fun b => FlockVBridge.Family.bitAt (Vs (wSesO Vs sh sess i)) (Lb (wSesO Vs sh sess i))
        (wUnO Vs sh sess un i) (rowColOfO sh (wLocO Vs sh sess i).1 (128 * (wLocO Vs sh sess i).2 + b))
    else 0 := by
  rw [getElem!_pos (wReadO Vs sh sess un Lb) i (by rw [wReadO_size]; exact hi)]
  simp only [wReadO, Array.getElem_ofFn]

theorem el_wReadO (Lb : (j : Fin S) → (Vs j).Bits) {i : ℕ} (hi : i < Flock.RecOpening.nw sh)
    (hex : ∃ pa, WLocO Vs sh sess i pa) :
    Flock.RecAlgebra.el (wReadO Vs sh sess un Lb) i =
      FlockVBridge.ofBits fun b => FlockVBridge.Family.bitAt (Vs (wSesO Vs sh sess i)) (Lb (wSesO Vs sh sess i))
        (wUnO Vs sh sess un i) (rowColOfO sh (wLocO Vs sh sess i).1 (128 * (wLocO Vs sh sess i).2 + b)) := by
  have hs : i < (wReadO Vs sh sess un Lb).size := by rw [wReadO_size]; exact hi
  unfold Flock.RecAlgebra.el
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs, Option.getD_some,
    ← getElem!_pos (wReadO Vs sh sess un Lb) i hs, getElem!_wReadO Lb hi, ite_eq_left hex]

/-- Opening part `p`'s maps at the part `RecOpening.part` accepts: its row ports' sizes, its `v` elements, and its `w`
index, group `p`'s, with as many elements as its row ports. -/
theorem ofO_part {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs) {p : ℕ} (hp : p < gs.size)
    {P : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : Flock.RecOpening.part sh p = .ok (P, ns, nv)) :
    nsOfO sh p = ns ∧ nvOfO sh p = nv ∧ wIOfO sh p = wIdx (Flock.RecOpening.portSizes sh) gs[p].2.1 ∧
      (wIOfO sh p).size = ns.sum := by
  obtain ⟨-, hns, hnv⟩ := part_eqOpening hS hg hp hpart
  have hgp : (partsOfO sh)[p]! = gs[p] := by rw [partsOfO_eq hS hg, getElem!_pos gs p hp]
  have hwI : wIOfO sh p = wIdx (Flock.RecOpening.portSizes sh) gs[p].2.1 := by rw [wIOfO, hgp]
  refine ⟨by rw [nsOfO, hgp, hns], by rw [nvOfO, hgp, hnv], hwI, ?_⟩
  rw [hwI, FlockVBridge.OfVStar.wIdx_size, hns]

/-- **`wReadO` holds `w`'s bit where a part's unit reads it**: where every part `p` is copy 0 of unit `un p` in
session `sess p` registering one `w` where its rows read it (`OpeningZeroAtL`) in a layer agreeing with the registrant,
bit `b` of element `(wIOfO sh p₁)[a]`, read by `p₁`'s unit, is of an element `p₁` reads whole (`openReadCol_whole`, at
the part's check), so some part reads it whole at keyed columns (`wLocO_spec`), and `wReadO` holds the layer's bit
there (`el_wReadO`), which is `w`'s (`bitAt_of_regBit`, `RegBit`). -/
theorem bitOf_wReadO {Reg' : Type} {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} (hA : ∀ s, (zs s).RegAgree (Lb s)) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs) {w v : ℕ → F128}
    (h : ∀ p < gs.size, OpeningZeroAtL (zs (sess p)) (Lb (sess p)) (un p) 0 sh gs p w v) {p₁ a b : ℕ}
    (hp₁ : p₁ < gs.size) (ha : a < (wIOfO sh p₁).size) (hb : b < 128)
    (hrd₁ : ReadCol (Rows.ofNet (Vs (sess p₁)).hU) (rowColOfO sh p₁ (128 * a + b))) :
    FlockVBridge.bitOf (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb) (wIOfO sh p₁)[a]) b =
      FlockVBridge.bitOf (w (wIOfO sh p₁)[a]) b := by
  have hP : partsOfO sh = gs := partsOfO_eq hS hg
  have hloc : ∀ p₁ (_ : p₁ < gs.size) a (_ : a < (wIOfO sh p₁).size) b, b < 128 →
      ReadCol (Rows.ofNet (Vs (sess p₁)).hU) (rowColOfO sh p₁ (128 * a + b)) →
      0 < (Vs (sess p₁)).c.upv ∧ (wIOfO sh p₁)[a] < Flock.RecOpening.nw sh ∧
      KeyedCol (Vs (sess p₁)) (rowColOfO sh p₁ (128 * a + b)) ∧
        RegBit (zs (sess p₁)) (un p₁) 0 (rowColOfO sh p₁ (128 * a + b))
          (FlockVBridge.bitOf (w (wIOfO sh p₁)[a]) b) := by
    intro p₁ hp₁ a ha b hb hrd
    obtain ⟨_, hj, _, _, _, _, _, _, _, _, ns, nv, hpart, hw, _, _⟩ := h p₁ hp₁
    obtain ⟨hns, hnv, hwI, hsz⟩ := ofO_part hS hg hp₁ hpart
    have hq : 128 * a + b < 128 * ns.sum := by rw [← hsz]; omega
    rw [rowColOfO, hns, hnv] at hrd ⊢
    have hreg := hw _ hq hrd
    have hval : FlockVBridge.wbit (FlockVBridge.Decode.sub (wIdx (Flock.RecOpening.portSizes sh) gs[p₁].2.1) w)
        (128 * a + b) = FlockVBridge.bitOf (w (wIOfO sh p₁)[a]) b := by
      unfold FlockVBridge.wbit
      rw [← hwI, show (128 * a + b) / 128 = a by omega, show (128 * a + b) % 128 = b by omega,
        FlockVBridge.Decode.sub_lt _ _ _ ha]
    rw [hval] at hreg
    have hi : (wIOfO sh p₁)[a] < Flock.RecOpening.nw sh := by
      rw [← structureOf_nwO hS, ← part_nwO hS hpart]
      exact FlockVBridge.Family.wIdx_lt (Flock.RecOpening.portSizes sh) ((partsOfO sh)[p₁]!.2.1) ha
    refine ⟨hj, hi, ⟨?_, hrd, ?_⟩, hreg⟩
    · obtain ⟨hlt, -⟩ := hreg
      exact hlt
    · obtain ⟨-, q, kq, hkq, hkey, -⟩ := hreg
      exact ⟨q, kq, hkq, hkey⟩
  have hall : ∀ b₀ < 128, ReadCol (Rows.ofNet (Vs (sess p₁)).hU) (rowColOfO sh p₁ (128 * a + b₀)) := by
    intro b₀ hb₀
    obtain ⟨_, _, _, hc, hu, m, cl, hpm, hsh, P, ns, nv, hpart, _, _, _⟩ := h p₁ hp₁
    obtain ⟨hns, hnv, -, hsz⟩ := ofO_part hS hg hp₁ hpart
    have hrd := hrd₁
    rw [rowColOfO, hns, hnv] at hrd ⊢
    exact ReadWhole.openReadCol_whole sh _ _ m cl p₁ hc hu hpm hsh P ns nv hpart a (by rw [← hsz]; exact ha) b hb
      hrd b₀ hb₀
  have hex : ∃ pa, WLocO Vs sh sess (wIOfO sh p₁)[a] pa :=
    ⟨(p₁, a), by rw [hP]; exact hp₁, Array.getElem?_eq_getElem ha,
      fun b₁ hb₁ => (hloc p₁ hp₁ a ha b₁ hb₁ (hall b₁ hb₁)).2.2.1⟩
  obtain ⟨hp₀, ha₀, hkc⟩ := wLocO_spec hex
  rw [hP] at hp₀
  obtain ⟨ha₀, he₀⟩ := Array.getElem?_eq_some_iff.mp ha₀
  obtain ⟨hlt, hrd₀, -⟩ := hkc b hb
  obtain ⟨hj₀, -, -, hreg⟩ := hloc _ hp₀ _ ha₀ b hb hrd₀
  rw [el_wReadO Lb (hloc p₁ hp₁ a ha b hb hrd₁).2.1 hex, FlockVBridge.bitOf_ofBits _ hb]
  have hdec := FlockVBridge.Family.bitAt_of_regBit (zs _) (hA _) hreg
    (FlockVBridge.Family.ioColAt_of_readCol _ hlt hrd₀) hj₀
  rw [he₀] at hdec
  exact hdec

/-- **Each opening part's `OpeningZeroAtL` at `wReadO`**: where every part `p` is copy 0 of unit `un p` in session
`sess p` registering one `w` where its rows read it (`OpeningZeroAtL`) in a layer agreeing with the registrant, each
does at `wReadO`'s, which holds `w`'s bit at every column a part's unit reads (`bitOf_wReadO`). -/
theorem openingZeroAtL_wReadO {Reg' : Type} {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} (hA : ∀ s, (zs s).RegAgree (Lb s)) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs) {w v : ℕ → F128}
    (h : ∀ p < gs.size, OpeningZeroAtL (zs (sess p)) (Lb (sess p)) (un p) 0 sh gs p w v) {p : ℕ}
    (hp : p < gs.size) :
    OpeningZeroAtL (zs (sess p)) (Lb (sess p)) (un p) 0 sh gs p (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb)) v := by
  obtain ⟨hpg, hj, hone, hc, hu, m, cl, hpm, hsh, P, ns, nv, hpart, hw, hv, hz⟩ := h p hp
  obtain ⟨hns, hnv, hwI, hsz⟩ := ofO_part hS hg hp hpart
  refine ⟨hpg, hj, hone, hc, hu, m, cl, hpm, hsh, P, ns, nv, hpart, fun q hq hrd => ?_, hv, hz⟩
  have ha : q / 128 < (wIOfO sh p).size := by rw [hsz]; omega
  have hrd' : ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOfO sh p (128 * (q / 128) + q % 128)) := by
    rw [Nat.div_add_mod, rowColOfO, hns, hnv]
    exact hrd
  have hval : FlockVBridge.wbit (FlockVBridge.Decode.sub (wIdx (Flock.RecOpening.portSizes sh) gs[p].2.1)
      (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb))) q =
      FlockVBridge.wbit (FlockVBridge.Decode.sub (wIdx (Flock.RecOpening.portSizes sh) gs[p].2.1) w) q := by
    unfold FlockVBridge.wbit
    rw [← hwI, FlockVBridge.Decode.sub_lt _ _ _ ha, FlockVBridge.Decode.sub_lt _ _ _ ha]
    exact bitOf_wReadO hA hS hg h hp ha (Nat.mod_lt _ (by norm_num)) hrd'
  rw [hval]
  exact hw q hq hrd

/-- **`wReadO` is `w` at a running-sum element a part holds**: element `e` of the opening's w, at position `q` of part
`p`'s row ports and named `acc0 r` or `acc1 r`, is read in all 128 bits there (`readCol_acc`, at the part's check),
so `wReadO` holds `w`'s bits (`bitOf_wReadO`). -/
theorem el_wReadO_acc {Reg' : Type} {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} (hA : ∀ s, (zs s).RegAgree (Lb s)) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs) {w v : ℕ → F128}
    (h : ∀ p < gs.size, OpeningZeroAtL (zs (sess p)) (Lb (sess p)) (un p) 0 sh gs p w v) {p q e r : ℕ}
    (hp : p < gs.size) (he : (wIdx (Flock.RecOpening.portSizes sh) gs[p].2.1)[q]? = some e)
    (hwi : Flock.RecOpening.wi sh (.acc0 r) = some e ∨ Flock.RecOpening.wi sh (.acc1 r) = some e) :
    Flock.RecAlgebra.el (wReadO Vs sh sess un Lb) e = w e := by
  obtain ⟨_, _, _, hc, hu, _, _, hpm, hsh, _, ns, nv, hpart, -⟩ := h p hp
  obtain ⟨hns, hnv, hwI, -⟩ := ofO_part hS hg hp hpart
  have he' : (wIOfO sh p)[q]? = some e := by rw [hwI]; exact he
  obtain ⟨hq, rfl⟩ := Array.getElem?_eq_some_iff.1 he'
  refine FlockVBridge.AccChain.eq_of_bitOf fun b hb => bitOf_wReadO hA hS hg h hp hq hb ?_
  rw [rowColOfO, hns, hnv]
  exact ReadWhole.readCol_acc _ hc hu hpm hsh hS hg hp hpart he hwi hb

/-- Obligation: discharged by the opening's S (Ligerito's final check, residuals `claims` and `claims + 1`, adds both
running sums as terms, `RecOpening.accTerms`, and a part holding a residual holds its terms' elements).
**Some part holds each running sum's element**: `acc0 r` and `acc1 r` (`r < 2`) are each element `e` of w at some
position of some part's row ports. -/
def AccHeldO (sh : Flock.Shape) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) : Prop :=
  ∀ r < 2, ∀ k : Flock.RecAlgebra.Key, (k = .acc0 r ∨ k = .acc1 r) → ∃ p e q, ∃ _ : p < gs.size,
    Flock.RecOpening.wi sh k = some e ∧ (wIdx (Flock.RecOpening.portSizes sh) gs[p].2.1)[q]? = some e

/-- **`SessionAt` at `wReadO`**: R's opening in V*'s sessions at one `w` (`SessionAt`), each part `p` of the groups `gs`
copy 0 of unit `un p` in session `sess p` (`OpeningZeroAtL` at `w`) in a layer agreeing with the registrant, is in them
at `wReadO`'s w where some part holds each running sum (`AccHeldO`): (a) is `openingZeroAtL_wReadO`, and (c)'s ends
hold as `wReadO` is `w` at each running-sum element (`el_wReadO_acc`). -/
theorem sessionAt_wReadO {Reg' : Type} {zs : (s : Fin S) → (Vs s).Outer Reg'} {Lb : (s : Fin S) → (Vs s).Bits}
    (hA : ∀ s, (zs s).RegAgree (Lb s)) {unit : String} {c3 : Flock.RecCoins.C3} {w : Array F128}
    {ops : FlockVBridge.AccChain.Openings} (hs : FlockC3.Opening.SessionAt zs Lb sh unit c3 w ops)
    {o : Flock.RecCoins.Opening} (ho : Flock.RecCoins.opening c3 unit = .ok o) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs)
    (h : ∀ p < gs.size, OpeningZeroAtL (zs (sess p)) (Lb (sess p)) (un p) 0 sh gs p (Flock.RecAlgebra.el w)
      (Flock.RecAlgebra.el o.v)) (hcov : AccHeldO sh gs) :
    FlockC3.Opening.SessionAt zs Lb sh unit c3 (wReadO Vs sh sess un Lb) ops := by
  obtain ⟨o', ho', hsh, -, -, acc, sA, uA, cA, hop, hcoef, hrd⟩ := hs
  obtain rfl : o = o' := Except.ok.inj (ho.symm.trans ho')
  have hval : ∀ r < 2, ∀ k : Flock.RecAlgebra.Key, (k = .acc0 r ∨ k = .acc1 r) →
      FlockC3.Opening.valO sh (wReadO Vs sh sess un Lb) (FlockC3.Opening.consO ops) k =
        FlockC3.Opening.valO sh w (FlockC3.Opening.consO ops) k := by
    intro r hr k hk
    obtain ⟨p, e, _, hp, hwi, he⟩ := hcov r hr k hk
    rcases hk with rfl | rfl
    · simp only [FlockC3.Opening.valO, hwi, Flock.RecAlgebra.getElem!_el, el_wReadO_acc hA hS hg h hp he (.inl hwi)]
    · simp only [FlockC3.Opening.valO, hwi, Flock.RecAlgebra.getElem!_el, el_wReadO_acc hA hS hg h hp he (.inr hwi)]
  refine ⟨o, ho, hsh, wReadO_size Lb,
    ⟨GS, gs, sess, un, fun _ => 0, hS, hg, fun p hp => openingZeroAtL_wReadO hA hS hg h hp⟩,
    acc, sA, uA, cA, hop, hcoef, fun r hr => ?_⟩
  rw [hval r hr _ (.inl rfl), hval r hr _ (.inr rfl)]
  exact hrd r hr

end read

end

end FlockC3.Reps

#print axioms FlockC3.Reps.ofO_part
#print axioms FlockC3.Reps.bitOf_wReadO
#print axioms FlockC3.Reps.openingZeroAtL_wReadO
#print axioms FlockC3.Reps.el_wReadO_acc
#print axioms FlockC3.Reps.sessionAt_wReadO
