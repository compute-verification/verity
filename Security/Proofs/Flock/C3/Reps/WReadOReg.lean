import Proofs.Flock.C3.Reps.WReadO
import Proofs.Flock.C3.Compose.RsFacts

/-!
# Each opening part's `OpeningZeroAtL` at `wReadO`, from its session's registrations

`openingZeroAtL_wReadO` carries `OpeningZeroAtL` from one `w` to `wReadO`'s. Here it holds at `wReadO`'s from what each
part's session registers, as `hResL_of` holds V*'s parts at `wOf`'s.

* `openingName_inj`, `openParams_of_check`: a unit `RecOpening.check` accepted under the name
  `rec-opening/m<m>-c<c>-p<j>` gives back `(m, c, j)`. `openUnit_of`: a session that is V*'s statement of opening part
  `p` and passes the verifier's statement check has the unit facts `OpeningZeroAtL` asks (its check, `unitLog`, and
  its name's `(m, c, p)`).
* `regBit_wReadO`: where each row-port column a part's unit reads registers some bit (`hkeys`) and two parts registering
  one bit of one element of w register one value (`hagree`), each such column registers `wReadO`'s bit. The part reads
  the element whole (`openReadCol_whole`), so some part holds it at keyed columns (`wLocO_spec`), and there a layer
  agreeing with the registrant has the registered bit (`bitAt_of_regBit`).
* `openingZeroAtL_of_regs`: each part's `OpeningZeroAtL` at `wReadO`, from the unit facts, the constant 1 (`hone`), the
  row ports' registrations (`regBit_wReadO`), its `v` port (`hv`) and its outputs (`hz`).
-/

namespace FlockC3.Reps

noncomputable section

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.OfVStar (ReadCol OpeningZeroAtL rowCol partRep vBase partBase)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.PartS (wIdx)

/-! ## The unit's name -/

/-- **The name `rec-opening/m<m>-c<c>-p<j>` determines `m`, `c` and `j`.** -/
theorem openingName_inj {m c j m' c' j' : ℕ}
    (h : Flock.RecOpening.unitName m c j = Flock.RecOpening.unitName m' c' j') : m = m' ∧ c = c' ∧ j = j' := by
  have h := congrArg String.toList h
  simp only [Flock.RecOpening.unitName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h2⟩ := FlockVBridge.digits_dash (dig m) (dig m') (by simpa using h)
  simp only [List.cons.injEq, true_and] at h2
  obtain ⟨h3, h4⟩ := FlockVBridge.digits_dash (dig c) (dig c') (by simpa using h2)
  simp only [List.cons.injEq, true_and] at h4
  exact ⟨FlockVBridge.eq_of_toDigits h1, FlockVBridge.eq_of_toDigits h3, FlockVBridge.eq_of_toDigits h4⟩

theorem startsWith_openingName (m c j : ℕ) :
    (Flock.RecOpening.unitName m c j).startsWith "rec-opening/" = true := by
  have : Flock.RecOpening.unitName m c j = "rec-opening/" ++ s!"m{m}-c{c}-p{j}" := by
    simp [Flock.RecOpening.unitName, ← String.toList_inj, List.append_assoc, FlockVBridge.toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

/-- **A unit the verifier accepted under the name `unitName m c j` gives back `(m, c, j)`**: where `params` is `none`,
`RecOpening.check` refuses a name beginning `rec-opening/`. -/
theorem openParams_of_check {c : Flock.Circuit} (h : Flock.RecOpening.check c = .ok ()) {m cl j : ℕ}
    (hn : c.unit.name = Flock.RecOpening.unitName m cl j) :
    Flock.RecOpening.params c.unit.name = some (m, cl, j) := by
  cases hp : Flock.RecOpening.params c.unit.name with
  | some x =>
    obtain ⟨m', c', j'⟩ := x
    obtain ⟨rfl, rfl, rfl⟩ := openingName_inj ((openingName_params hp).trans hn)
    rfl
  | none =>
    unfold Flock.RecOpening.check at h
    rw [hp] at h
    dsimp only at h
    rw [hn, startsWith_openingName] at h
    cases h

section unit

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **An opening part's unit facts**: where session `j` is V*'s statement of part `p` of the opening at `m`, `cl`
(`hs`) and passes the verifier's statement check (`hck`), its circuit is named `unitName m cl p` (`nameChecked`), so
the verifier's `rec-opening` check (`ParseFacts.recOpening`) and that name give the unit facts `OpeningZeroAtL` asks. -/
theorem openUnit_of (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (holoInnerAt T CT st hp).Coin)) (R₂ : Reg₂) (j : Fin (vstarCount F.ish)) {m cl p : ℕ}
    {S : Flock.RecOpen.GfS} {ports : List (String × ℕ)} {nv : ℕ}
    (hs : F.stmts[(j : ℕ)]? = some (.opening m cl p S ports nv))
    (hck : ∃ (sj : Flock.Stmt) (tags : _) (tables : _) (circuitB : _) (partition : _) (program : _) (own : _)
      (rows : _), sj.c = (F.Vs j).c ∧ Flock.RecVStar.checkStmt (some F.ish) tags tables sj circuitB partition program
        (some own) (some (j : ℕ)) (some (F.bind j)) rows = .ok ()) :
    Flock.RecOpening.check (F.Vs j).c = .ok () ∧ (F.Vs j).c.unit.unitLog ≤ 32 ∧
      Flock.RecOpening.params (F.Vs j).c.unit.name = some (m, cl, p) := by
  obtain ⟨sj, tags, tables, circuitB, partition, program, own, rows, hc, hpass⟩ := hck
  obtain ⟨s, hs', hname⟩ := FlockC3.PubRows.nameChecked F _ sj tags tables circuitB partition program own rows hc hpass
  rw [hs] at hs'
  cases hs'
  have hchk := (parse_facts (F.zs R ω t R₂ j).hc).recOpening
  have hn : (F.Vs j).c.unit.name = Flock.RecOpening.unitName m cl p := hname.symm
  exact ⟨hchk, FlockVBridge.Family.unitLog_le (F.Vs j).hscope, openParams_of_check hchk hn⟩

end unit

/-! ## The row ports at `wReadO` -/

section read

variable {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S} {un : (p : ℕ) → Fin (Vs (sess p)).nv}

/-- A read column registering a bit at copy 0 is keyed (`KeyedCol`). -/
theorem keyedCol_of_regBit {Q : VStmt} {Reg' : Type} {zo : Q.Outer Reg'} {u : Fin Q.nv} {col : ℕ} {b : Bool}
    (hb : RegBit zo u 0 col b) (hrd : ReadCol (Rows.ofNet Q.hU) col) : KeyedCol Q col := by
  obtain ⟨hlt, q, kq, hkq, hkey, -⟩ := hb
  exact ⟨hlt, hrd, q, kq, hkq, hkey⟩

/-- **A part's read row-port column registers `wReadO`'s bit**: where each row-port column a part's unit reads
registers some bit (`hkeys`) and two parts registering one bit of one element of w register one value (`hagree`), the
part reads the element whole (`openReadCol_whole`, at its unit facts `hU`), so some part holds it at keyed columns
(`wLocO_spec`), where a layer agreeing with the registrant has the registered bit (`bitAt_of_regBit`), `wReadO`'s. -/
theorem regBit_wReadO {Reg' : Type} {zs : (s : Fin S) → (Vs s).Outer Reg'} {Lb : (s : Fin S) → (Vs s).Bits}
    (hA : ∀ s, (zs s).RegAgree (Lb s)) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs) {m cl : ℕ}
    (hsh : Flock.RecOpening.shape? m cl = some sh)
    (hU : ∀ p < gs.size, Flock.RecOpening.check (Vs (sess p)).c = .ok () ∧ (Vs (sess p)).c.unit.unitLog ≤ 32 ∧
      Flock.RecOpening.params (Vs (sess p)).c.unit.name = some (m, cl, p))
    (hkeys : ∀ p < gs.size, ∀ q < 128 * (nsOfO sh p).sum, ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOfO sh p q) →
      ∃ b, RegBit (zs (sess p)) (un p) 0 (rowColOfO sh p q) b)
    (hagree : ∀ (p p' a a' e : ℕ) (b b' : Bool), p < gs.size → p' < gs.size → ∀ (ha : a < (wIOfO sh p).size)
      (ha' : a' < (wIOfO sh p').size), (wIOfO sh p)[a] = (wIOfO sh p')[a'] → e < 128 →
      RegBit (zs (sess p)) (un p) 0 (rowColOfO sh p (128 * a + e)) b →
        RegBit (zs (sess p')) (un p') 0 (rowColOfO sh p' (128 * a' + e)) b' → b = b')
    {p a e : ℕ} (hp : p < gs.size) (ha : a < (wIOfO sh p).size) (he : e < 128)
    (hrd : ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOfO sh p (128 * a + e))) :
    RegBit (zs (sess p)) (un p) 0 (rowColOfO sh p (128 * a + e))
      (FlockVBridge.bitOf (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb) (wIOfO sh p)[a]) e) := by
  have hP : partsOfO sh = gs := partsOfO_eq hS hg
  have hsz : ∀ p₁, (wIOfO sh p₁).size = (nsOfO sh p₁).sum := fun p₁ => by
    rw [wIOfO, nsOfO, FlockVBridge.OfVStar.wIdx_size]
  obtain ⟨hc, hu, hpm⟩ := hU p hp
  obtain ⟨P, ns, nv, hpart, -⟩ := FlockVBridge.OfVStar.check_opening hc hpm hsh
  obtain ⟨hns, hnv, -, hsz'⟩ := ofO_part hS hg hp hpart
  have hall : ∀ b₀ < 128, ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOfO sh p (128 * a + b₀)) := by
    intro b₀ hb₀
    have hrd' := hrd
    rw [rowColOfO, hns, hnv] at hrd' ⊢
    exact ReadWhole.openReadCol_whole sh _ _ m cl p hc hu hpm hsh P ns nv hpart a (by rw [← hsz']; exact ha) e he
      hrd' b₀ hb₀
  have hq : ∀ b₀ < 128, 128 * a + b₀ < 128 * (nsOfO sh p).sum := fun b₀ hb₀ => by rw [← hsz]; omega
  have hex : ∃ pa, WLocO Vs sh sess (wIOfO sh p)[a] pa := by
    refine ⟨(p, a), by rw [hP]; exact hp, Array.getElem?_eq_getElem ha, fun b₁ hb₁ => ?_⟩
    obtain ⟨_, hb⟩ := hkeys p hp _ (hq b₁ hb₁) (hall b₁ hb₁)
    exact keyedCol_of_regBit hb (hall b₁ hb₁)
  obtain ⟨hp₀, ha₀, hkc⟩ := wLocO_spec hex
  rw [hP] at hp₀
  obtain ⟨ha₀, he₀⟩ := Array.getElem?_eq_some_iff.mp ha₀
  obtain ⟨hlt, hrd₀, -⟩ := hkc e he
  obtain ⟨b₀, hb₀⟩ := hkeys _ hp₀ _ (by rw [← hsz]; omega) hrd₀
  have hdec := FlockVBridge.Family.bitAt_of_regBit (zs _) (hA _) hb₀
    (FlockVBridge.Family.ioColAt_of_readCol _ hlt hrd₀) (upv_of_outer (zs _))
  have hi : (wIOfO sh p)[a] < Flock.RecOpening.nw sh := by
    rw [← structureOf_nwO hS, ← part_nwO hS hpart]
    exact FlockVBridge.Family.wIdx_lt (Flock.RecOpening.portSizes sh) ((partsOfO sh)[p]!.2.1) ha
  obtain ⟨b, hb⟩ := hkeys p hp _ (hq e he) hrd
  have hbb : b = b₀ := hagree p _ a _ e b b₀ hp hp₀ ha ha₀ he₀.symm he hb hb₀
  have hbit : FlockVBridge.bitOf (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb) (wIOfO sh p)[a]) e = b₀ := by
    rw [el_wReadO Lb hi hex, FlockVBridge.bitOf_ofBits _ he]
    exact hdec
  rw [hbit, ← hbb]
  exact hb

/-- **Each opening part's `OpeningZeroAtL` at `wReadO`, from its session's registrations**: part `p` is copy 0 of unit
`un p` in session `sess p` at its unit facts (`hU`, and `check_opening`'s part), its constant 1 (`hone`), each row-port
column its unit reads registering `wReadO`'s bit (`regBit_wReadO`), its `v` port registering `v` (`hv`) and its
outputs registered zero (`hz`). -/
theorem openingZeroAtL_of_regs {Reg' : Type} {zs : (s : Fin S) → (Vs s).Outer Reg'} {Lb : (s : Fin S) → (Vs s).Bits}
    (hA : ∀ s, (zs s).RegAgree (Lb s)) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : Flock.RecOpening.structureOf sh = .ok GS)
    (hg : Flock.RecResiduals.groups GS (Flock.RecOpening.portSizes sh) = .ok gs) {m cl : ℕ}
    (hsh : Flock.RecOpening.shape? m cl = some sh)
    (hU : ∀ p < gs.size, Flock.RecOpening.check (Vs (sess p)).c = .ok () ∧ (Vs (sess p)).c.unit.unitLog ≤ 32 ∧
      Flock.RecOpening.params (Vs (sess p)).c.unit.name = some (m, cl, p))
    (hone : ∀ p < gs.size,
      Lb (sess p) (Const.Prog.oneAt (Placed.keyProg (Vs (sess p)).c (Vs (sess p)).hU (Vs (sess p)).nv) (un p)) = true)
    (hkeys : ∀ p < gs.size, ∀ q < 128 * (nsOfO sh p).sum, ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOfO sh p q) →
      ∃ b, RegBit (zs (sess p)) (un p) 0 (rowColOfO sh p q) b)
    (hagree : ∀ (p p' a a' e : ℕ) (b b' : Bool), p < gs.size → p' < gs.size → ∀ (ha : a < (wIOfO sh p).size)
      (ha' : a' < (wIOfO sh p').size), (wIOfO sh p)[a] = (wIOfO sh p')[a'] → e < 128 →
      RegBit (zs (sess p)) (un p) 0 (rowColOfO sh p (128 * a + e)) b →
        RegBit (zs (sess p')) (un p') 0 (rowColOfO sh p' (128 * a' + e)) b' → b = b')
    {v : ℕ → F128}
    (hv : ∀ (p : ℕ) (hp : p < gs.size), ∀ q < 128 * nvOfO sh p,
      RegBit (zs (sess p)) (un p) 0 (vBase (partRep (nsOfO sh p) (nvOfO sh p)) + q)
        (FlockVBridge.wbit (FlockVBridge.Decode.sub gs[p].2.2 v) q))
    (hz : ∀ p < gs.size, ∀ (P : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      Flock.RecOpening.part sh p = .ok (P, ns, nv) → ∀ t < 16 * Flock.RecResiduals.RES_OUT_WORDS,
        FlockVBridge.AccChain.RegOut (zs (sess p)) (un p) (upv_of_outer (zs (sess p))) (partBase P ns nv + t) false)
    {p : ℕ} (hp : p < gs.size) :
    OpeningZeroAtL (zs (sess p)) (Lb (sess p)) (un p) 0 sh gs p (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb)) v := by
  obtain ⟨hc, hu, hpm⟩ := hU p hp
  obtain ⟨P, ns, nv, hpart, -⟩ := FlockVBridge.OfVStar.check_opening hc hpm hsh
  obtain ⟨hns, hnv, hwI, hsz⟩ := ofO_part hS hg hp hpart
  refine ⟨hp, upv_of_outer (zs (sess p)), hone p hp, hc, hu, m, cl, hpm, hsh, P, ns, nv, hpart, fun q hq hrd => ?_,
    fun q hq => ?_, hz p hp P ns nv hpart⟩
  · have ha : q / 128 < (wIOfO sh p).size := by rw [hsz]; omega
    have hrd' : ReadCol (Rows.ofNet (Vs (sess p)).hU) (rowColOfO sh p (128 * (q / 128) + q % 128)) := by
      rw [Nat.div_add_mod, rowColOfO, hns, hnv]
      exact hrd
    have hreg := regBit_wReadO hA hS hg hsh hU hkeys hagree hp ha (Nat.mod_lt _ (by norm_num)) hrd'
    have hval : FlockVBridge.wbit (FlockVBridge.Decode.sub (wIdx (Flock.RecOpening.portSizes sh) gs[p].2.1)
        (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb))) q =
        FlockVBridge.bitOf (Flock.RecAlgebra.el (wReadO Vs sh sess un Lb) (wIOfO sh p)[q / 128]) (q % 128) := by
      unfold FlockVBridge.wbit
      rw [← hwI, FlockVBridge.Decode.sub_lt _ _ _ ha]
    rw [hval]
    rw [Nat.div_add_mod, rowColOfO, hns, hnv] at hreg
    exact hreg
  · have hv' := hv p hp q (by rw [hnv]; exact hq)
    rw [hns, hnv] at hv'
    exact hv'

end read

end

end FlockC3.Reps

#print axioms FlockC3.Reps.openingName_inj
#print axioms FlockC3.Reps.startsWith_openingName
#print axioms FlockC3.Reps.openParams_of_check
#print axioms FlockC3.Reps.openUnit_of
#print axioms FlockC3.Reps.keyedCol_of_regBit
#print axioms FlockC3.Reps.regBit_wReadO
#print axioms FlockC3.Reps.openingZeroAtL_of_regs
