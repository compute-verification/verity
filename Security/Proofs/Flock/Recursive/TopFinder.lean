import Proofs.Flock.Recursive.Registrants
import Proofs.Flock.Recursive.TopRoot
import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.C3.RegLevels.TopHold

/-!
# A one-run finder for a held top different from the registered top

The candidate-3 verifier now checks the five registered-entry fields structurally, beside its canonical-JSON
comparison.  Consequently an accepted held binding top different from the explicit registered strings gives two
equal-size frame-v3 trees with one root.  `topFinder` runs the audit once and returns `topRoot_collision`'s collision
at exactly that bad event.
-/

namespace Flock.SecurityProofs.TopFinder

open Flock

set_option maxRecDepth 20000

theorem foldl_loop_eq {β : Type} (f : β → UInt8 → β) (as : ByteArray) (stop : Nat) (h : stop ≤ as.size) :
    ∀ (i j : Nat) (b : β), i + j = stop →
      ByteArray.foldlM.loop (m := Id) (fun b a => pure (f b a)) as stop h i j b =
        ((as.data.toList.drop j).take i).foldl f b := by
  intro i
  induction i with
  | zero => intro j b _; unfold ByteArray.foldlM.loop; split <;> rfl
  | succ i ih =>
    intro j b hij
    unfold ByteArray.foldlM.loop
    have hj : j < stop := by omega
    rw [dite_eq_left_of_eq_true (eq_true hj)]
    have hjs : j < as.data.toList.length := by simp; omega
    rw [List.drop_eq_getElem_cons hjs]
    simp only [List.take_succ_cons, List.foldl_cons]
    rw [← ih (j + 1) _ (by omega)]
    simp [Array.getElem_toList]
    rfl

theorem foldl_eq {β : Type} (f : β → UInt8 → β) (init : β) (as : ByteArray) :
    as.foldl f init = as.data.toList.foldl f init := by
  unfold ByteArray.foldl ByteArray.foldlM
  rw [dite_eq_left_of_eq_true (eq_true (Nat.le_refl _))]
  dsimp only
  rw [foldl_loop_eq f as as.size (Nat.le_refl _) (as.size - 0) 0 init (by omega)]
  simp only [List.drop_zero, Nat.sub_zero]
  rw [List.take_of_length_le (by simp)]
  rfl

theorem foldl_push (d₁ d₂ : UInt8 → Char) : ∀ (l : List UInt8) (s : String),
    (l.foldl (fun s x => (s.push (d₁ x)).push (d₂ x)) s).toList =
      s.toList ++ l.flatMap fun x => [d₁ x, d₂ x]
  | [], s => by simp
  | x :: l, s => by
    rw [List.foldl_cons, foldl_push d₁ d₂ l]
    simp [String.toList_push]

theorem hexOf_toList (b : ByteArray) :
    (hexOf b).toList = b.data.toList.flatMap fun x => (hexOf ⟨#[x]⟩).toList := by
  unfold hexOf
  rw [foldl_eq, foldl_push]
  simp only [String.toList_empty, List.nil_append]
  congr 1
  funext x
  rw [foldl_eq, foldl_push]
  simp

theorem hexOf_length (x : UInt8) : (hexOf ⟨#[x]⟩).toList.length = 2 := by
  unfold hexOf
  rw [foldl_eq, foldl_push]
  simp

def nib (c : Char) : Nat := if c.toNat < 87 then c.toNat - 48 else c.toNat - 87

def dec : List Char → Nat
  | [a, b] => nib a * 16 + nib b
  | _ => 0

theorem dec_hex_nat : ∀ n < 256, dec (hexOf ⟨#[n.toUInt8]⟩).toList = n := by decide +kernel

theorem dec_hex (x : UInt8) : dec (hexOf ⟨#[x]⟩).toList = x.toNat := by
  simpa using dec_hex_nat x.toNat x.toNat_lt

theorem flatten_inj {α : Type} {k : Nat} (hk : 0 < k) :
    ∀ {L L' : List (List α)}, (∀ x ∈ L, x.length = k) → (∀ x ∈ L', x.length = k) →
      L.flatten = L'.flatten → L = L'
  | [], [], _, _, _ => rfl
  | [], x :: _, _, h', e => by
    have hl := congrArg List.length e
    simp only [List.flatten_nil, List.length_nil, List.flatten_cons, List.length_append] at hl
    have := h' x (List.mem_cons_self ..)
    omega
  | x :: _, [], h, _, e => by
    have hl := congrArg List.length e
    simp only [List.flatten_nil, List.length_nil, List.flatten_cons, List.length_append] at hl
    have := h x (List.mem_cons_self ..)
    omega
  | x :: L, y :: L', h, h', e => by
    simp only [List.flatten_cons] at e
    have hx := h x (List.mem_cons_self ..)
    have hy := h' y (List.mem_cons_self ..)
    obtain ⟨rfl, e'⟩ := List.append_inj e (hx.trans hy.symm)
    rw [flatten_inj hk (fun z hz => h z (List.mem_cons_of_mem _ hz))
      (fun z hz => h' z (List.mem_cons_of_mem _ hz)) e']

/-- Lowercase hexadecimal encoding is injective (in particular on SHA-512's 64-byte outputs). -/
theorem hexOf_inj {a b : ByteArray} (h : hexOf a = hexOf b) : a = b := by
  have e := congrArg String.toList h
  rw [hexOf_toList, hexOf_toList, List.flatMap_def, List.flatMap_def] at e
  have e' := flatten_inj (k := 2) (by decide) (by simp [hexOf_length]) (by simp [hexOf_length]) e
  have e'' := congrArg (List.map dec) e'
  simp only [List.map_map] at e''
  have hd : (dec ∘ fun x => (hexOf ⟨#[x]⟩).toList) = UInt8.toNat := funext dec_hex
  rw [hd] at e''
  have hi : a.data.toList = b.data.toList :=
    List.map_injective_iff.mpr (fun x y hxy => UInt8.toNat_inj.mp hxy) e''
  exact ByteArray.ext (Array.ext' hi)

/-- The executable structural check exposes the five field equalities it checked. -/
theorem topEntryFields_ok {program : String} {bcs : Array ByteArray} {e : Lean.Json}
    (h : RecVStar.topEntryFields program bcs e = .ok true) :
    ∃ dom,
      Registered.domain program RecVStar.STACKED RecVStar.TOP_ROW_WORDS bcs.size Registered.WORD_BITS = .ok dom ∧
      (e.getObjValAs? String "value").toOption = some RecVStar.STACKED ∧
      (e.getObjValAs? String "root").toOption =
        some (hexOf (FV3.root dom (topLeaves dom bcs) Sha512.hash)) ∧
      (e.getObjValAs? Nat "leaves").toOption = some bcs.size ∧
      (e.getObjValAs? Nat "words").toOption = some RecVStar.TOP_ROW_WORDS ∧
      (e.getObjValAs? String "domain").toOption = some (hexOf dom) := by
  cases hd : Registered.domain program RecVStar.STACKED RecVStar.TOP_ROW_WORDS bcs.size
      Registered.WORD_BITS with
  | error err =>
    rw [RecVStar.topEntryFields, hd] at h
    change (Except.error err : Except String Bool) = Except.ok true at h
    cases h
  | ok dom =>
    rw [RecVStar.topEntryFields, hd] at h
    change Except.ok (
      (e.getObjValAs? String "value").toOption == some RecVStar.STACKED &&
      (e.getObjValAs? String "root").toOption ==
        some (hexOf (FV3.root dom (topLeaves dom bcs) Sha512.hash)) &&
      (e.getObjValAs? Nat "leaves").toOption == some bcs.size &&
      (e.getObjValAs? Nat "words").toOption == some RecVStar.TOP_ROW_WORDS &&
      (e.getObjValAs? String "domain").toOption == some (hexOf dom)) = Except.ok true at h
    have hb := Except.ok.inj h
    simp only [Bool.and_eq_true, beq_iff_eq] at hb
    rcases hb with ⟨⟨⟨⟨hv, hr⟩, hl⟩, hw⟩, hd'⟩
    exact ⟨dom, rfl, hv, hr, hl, hw, hd'⟩

/-- One entry structurally checked against two top arrays gives equal sizes and equal frame-v3 roots. -/
theorem topEntryFields_bind {program : String} {e : Lean.Json} {bcs bcs' : Array ByteArray}
    (h : RecVStar.topEntryFields program bcs e = .ok true)
    (h' : RecVStar.topEntryFields program bcs' e = .ok true) :
    ∃ dom, bcs.size = bcs'.size ∧
      FV3.root dom (topLeaves dom bcs) Sha512.hash =
        FV3.root dom (topLeaves dom bcs') Sha512.hash := by
  obtain ⟨dom, hdom, -, hr, hl, -, -⟩ := topEntryFields_ok h
  obtain ⟨dom', hdom', -, hr', hl', -, -⟩ := topEntryFields_ok h'
  have hsz : bcs.size = bcs'.size := Option.some.inj (hl.symm.trans hl')
  have hdd : dom = dom' := by
    rw [hsz] at hdom
    exact Except.ok.inj (hdom.symm.trans hdom')
  subst dom'
  have hhex : hexOf (FV3.root dom (topLeaves dom bcs) Sha512.hash) =
      hexOf (FV3.root dom (topLeaves dom bcs') Sha512.hash) :=
    Option.some.inj (hr.symm.trans hr')
  exact ⟨dom, hsz, hexOf_inj hhex⟩

end Flock.SecurityProofs.TopFinder

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting StrictCR
open Flock.SecurityProofs Flock.SecurityProofs.TopFinder
open scoped ENNReal

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ : Type} {I : Inner}

/-- The held binding top at a history differs from the explicit registered top strings. -/
def TopColl (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (t : List (Cm × I.Coin)) : Prop :=
  ∃ cs s bcs, innerCommitsAt (wr t).1 (wr t).2 st reg = .ok cs ∧
    cs.stream? "stacked/rep0" = some s ∧ s.tops[0]? = some bcs ∧ bcs ≠ bcs'

/-- A bad held top yields the collision between its concrete strings and the explicit registered strings. -/
theorem TopColl.collision {wr : List (Cm × I.Coin) → Lean.Json × String} {st : Flock.Stmt} {reg : Lean.Json}
    {bcs' : Array ByteArray} {lv : FlockSoundness.Accounting.Level} {cap : Cap Refine.Dig} {salt : ℕ → ByteArray}
    (hstk : StackedTopAt lv st reg cap salt bcs') {t : List (Cm × I.Coin)}
    (h : TopColl wr st reg bcs' t) : Merkle.Collision Refine.H512 := by
  obtain ⟨cs, s, bcs, hi, hs, ht, hne⟩ := h
  obtain ⟨program, e, hreg', hfields', -⟩ := hstk
  unfold innerCommitsAt at hi
  obtain ⟨_, -, hi⟩ := FlockC3.RegLevels.TopHold.eok_bind hi
  obtain ⟨program', e', hreg, hfields, -⟩ :=
    FlockC3.RegLevels.TopHold.innerCommits_top hi hs ht
  have he : (program', e') = (program, e) := by
    simpa using Except.ok.inj (hreg.symm.trans hreg')
  obtain ⟨rfl, rfl⟩ := he
  obtain ⟨dom, hsz, hroot⟩ := topEntryFields_bind hfields hfields'
  exact topRoot_collision hsz hne hroot

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {dg : ByteArray → ByteArray} {N : ℕ}

/-- One run of the recursive audit, selecting the held-top collision when its concrete top differs from `bcs'`.
An honest `Finder.CR` budget counts the audit run plus both top recomputations: for each side, one hm96 `treeLeaf`
hash per commit string and `2 * bcs.size - 1` frame-v3 leaf/node hashes (including padding where the size is not a
power of two). -/
noncomputable def topFinder
    (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig)
    (salt : ℕ → ByteArray) (hstk : StackedTopAt lv st reg cap salt bcs')
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Finder Pair :=
  oneRun _ σ (onePick Refine.H512
    (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
      AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ TopColl wr st reg bcs' o.2.2.1)
    fun _ h => TopColl.collision hstk h.2)

/-- A differing held top occurs with probability at most `qT²/2^513` under the honest budget of `topFinder`. -/
theorem prob_topColl_le
    (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig)
    (salt : ℕ → ByteArray) (hstk : StackedTopAt lv st reg cap salt bcs')
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qT : ℝ} (hT : (topFinder wr st reg bcs' lv cap salt hstk zs σ).CR Refine.H512 qT) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ TopColl wr st reg bcs' o.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        TopColl wr st reg bcs' o.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤
      qT ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    exact (expect_one_le Refine.H512 _ σ
      (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
        AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ TopColl wr st reg bcs' o.2.2.1)
      fun _ h => TopColl.collision hstk h.2).trans hT.le
  exact (ENNReal.le_ofReal_iff_toReal_le
    (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)) (by positivity)).2 hev

/-- The separate registrant and top finders union-bound their two one-run bad events. -/
theorem prob_regOrTop_le
    (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig)
    (salt : ℕ → ByteArray) (hstk : StackedTopAt lv st reg cap salt bcs')
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qX qT : ℝ} (hX : (regFinder zs dg N σ).CR Refine.H512 qX)
    (hT : (topFinder wr st reg bcs' lv cap salt hstk zs σ).CR Refine.H512 qT) :
    prob (fun o => RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 ∨
        (AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ TopColl wr st reg bcs' o.2.2.1))
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal (qX ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) :=
  (prob_or_le _ _ _ _).trans (add_le_add
    (prob_regColl_le zs dg N σ hX) (prob_topColl_le wr st reg bcs' lv cap salt hstk zs σ hT))

end FlockSoundness.Discharge.Recursion
