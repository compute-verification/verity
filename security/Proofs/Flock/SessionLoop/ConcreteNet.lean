module

public import Proofs.Flock.Claims.Recursion.VStarNets
import Flock.Net
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.Repr

/-!
# A net's text reads back as the net

`Net.parse` of a net's lines as V\* writes them (`VStarNets.netLs name l`: `VStar.header`, then one `VStar.rowLine` per
useful row) is the net `HmNets.netOf name l` builds, whenever that build succeeds, the name is one word and the layout has
one A row and one B row per useful column (`parse_netLs`). The header's tokens are its fields (`header_tokens`), `groups`
reads each side's port groups back (`groups_ok`), each row line reads back as its row (`row_ok`), and the header checks
`parse` runs before its rows are the ones `ofRows` passed (`ofRows_checks`). Nothing here evaluates a net: the layout is
a variable, so the verifier's own `sha512x3` and `hm96` nets read back without the kernel building them.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open Flock
open Flock.HmNets (Laid netOf)
open FlockVBridge.Names (tstep tokens_eq NoWs foldl_word nat_noWs)
open RecDeploy.VStarNets (netLs rowsLs)

namespace FlockDeploy.ConcreteNet

/-- One token: nonempty, without whitespace. -/
def Tok (s : String) : Prop := s ≠ "" ∧ NoWs s

theorem word_nat (n : Nat) : Tok (toString n) := ⟨Nat.repr_ne_empty, nat_noWs n⟩

theorem foldl_intercalate (acc : Array String) : ∀ (w : String) (rest : List String), (∀ x ∈ w :: rest, Tok x) →
    (" ".intercalate (w :: rest)).toList.foldl tstep (acc, "") =
      (acc ++ (w :: rest).dropLast.toArray, (w :: rest).getLast (List.cons_ne_nil _ _))
  | w, [], h => by
    rw [String.intercalate_singleton, foldl_word _ (h w List.mem_cons_self).2, String.empty_append,
      String.ofList_toList]
    simp
  | w, w' :: rest, h => by
    rw [String.intercalate_cons_cons, String.toList_append, String.toList_append, List.foldl_append,
      List.foldl_append, foldl_word _ (h w List.mem_cons_self).2, String.empty_append, String.ofList_toList,
      show " ".toList = [' '] from rfl, List.foldl_cons, List.foldl_nil,
      show tstep (acc, w) ' ' = (acc.push w, "") by simp [tstep, (h w List.mem_cons_self).1],
      foldl_intercalate (acc.push w) w' rest fun x hx => h x (List.mem_cons_of_mem _ hx)]
    simp

theorem tokens_intercalate (ws : List String) (h : ∀ x ∈ ws, Tok x) : tokens (" ".intercalate ws) = ws.toArray := by
  cases ws with
  | nil => rfl
  | cons w rest =>
    rw [tokens_eq]
    simp only [foldl_intercalate #[] w rest h]
    have hl := (h _ (List.getLast_mem (List.cons_ne_nil w rest))).1
    simp only [hl, bne_iff_ne, ne_eq, not_false_eq_true, ↓reduceIte]
    apply Array.toList_inj.mp
    simp [List.dropLast_concat_getLast]

theorem foldl_sp : ∀ (tl : List String) (hd : String),
    tl.foldl (fun s t => s ++ " " ++ t) hd = " ".intercalate (hd :: tl)
  | [], hd => by rw [String.intercalate_singleton]; rfl
  | t :: tl, hd => by
    rw [List.foldl_cons, foldl_sp tl, String.intercalate_cons_cons]
    cases tl with
    | nil => simp [String.intercalate_singleton]
    | cons t2 tl => rw [String.intercalate_cons_cons, String.intercalate_cons_cons]; simp [String.append_assoc]

def rowNums (a b : Array Nat) : List Nat := (a.size :: a.toList) ++ (b.size :: b.toList)

theorem rowLine_eq (a b : Array Nat) : VStar.rowLine a b = " ".intercalate ((rowNums a b).map toString) := by
  unfold VStar.rowLine
  simp only [Id.run, ← Array.forIn_toList, List.forIn_pure_yield_eq_foldl, pure_bind]
  simp only [String.push_eq_append]
  have e : ∀ (l : List Nat) (s : String), l.foldl (fun s x => s ++ String.singleton ' ' ++ toString x) s =
      (l.map toString).foldl (fun s t => s ++ " " ++ t) s := by
    intro l s; rw [List.foldl_map]; rfl
  rw [e, e, rowNums, List.map_append, List.map_cons, List.map_cons, List.cons_append, ← foldl_sp,
    List.foldl_append, List.foldl_cons]
  rfl

theorem num_of {h : Array String} {i n : Nat} (hi : h[i]? = some (toString n)) : Net.num h i = .ok n := by
  unfold Net.num
  rw [hi, Option.bind_some, show (toString n).toNat? = some n from Nat.toNat?_repr n]
  rfl

/-! ## groups -/

def gNums (g : Group) : List Nat := [g.col, g.words, g.bits.size] ++ g.bits.toList

/-- Numbers `L` sit in `h` from `at0` on. -/
def At (h : Array String) (at0 : Nat) (L : List Nat) : Prop := ∀ j n, L[j]? = some n → h[at0 + j]? = some (toString n)

theorem at_append {h : Array String} {at0 : Nat} {L M : List Nat} (hh : At h at0 (L ++ M)) :
    At h at0 L ∧ At h (at0 + L.length) M := by
  refine ⟨fun j n hj => hh j n ?_, fun j n hj => ?_⟩
  · rw [List.getElem?_append_left (List.getElem?_eq_some_iff.1 hj).1]; exact hj
  · rw [Nat.add_assoc]; apply hh; rw [List.getElem?_append_right (by omega)]; simpa using hj

theorem bits_loop {h : Array String} (p : Nat) : ∀ (bs : List Nat) (k : Nat) (acc : Array Nat),
    (∀ j (hj : j < bs.length), h[p + (k + j)]? = some (toString bs[j])) →
    forIn (m := Except String) (List.range' k bs.length) acc (fun i r => do
      let x ← Net.num h (p + i); pure (ForInStep.yield (r.push x))) = .ok (acc ++ bs.toArray)
  | [], k, acc, _ => by simp <;> rfl
  | b :: bs, k, acc, hh => by
    rw [List.length_cons, List.range'_succ, List.forIn_cons]
    have h0 := hh 0 (by simp)
    simp only [Nat.add_zero, List.getElem_cons_zero] at h0
    rw [num_of h0]
    show forIn (m := Except String) (List.range' (k + 1) bs.length) (acc.push b) _ = _
    rw [bits_loop p bs (k + 1) (acc.push b) (fun j hj => by
      have := hh (j + 1) (by simp; omega)
      simpa [Nat.add_assoc, Nat.add_comm 1] using this)]
    simp

theorem range_size (n : Nat) : [:n].size = n := by simp [Std.Legacy.Range.size]

def Aligned (g : Group) : Prop := (isPow2 g.words && g.col % g.words == 0) = true

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem at_num {h : Array String} {at0 : Nat} {L : List Nat} (hh : At h at0 L) (j : Nat) (hj : j < L.length) :
    Net.num h (at0 + j) = .ok L[j] := num_of (hh j _ (List.getElem?_eq_getElem hj))

theorem groups_loop {h : Array String} : ∀ (gs : List Group) (k at_ : Nat) (acc : Array Group),
    At h at_ (gs.flatMap gNums) → (∀ g ∈ gs, Aligned g) →
    forIn (m := Except String) (List.range' k gs.length) (at_, acc) (fun x __s => do
            let col ← Net.num h __s.fst
            let words ← Net.num h (__s.fst + 1)
            let np ← Net.num h (__s.fst + 2)
            let __s_1 ←
              forIn (List.range' 0 np) #[] fun i __s_1 => do
                  let __do_lift ← Net.num h (__s.fst + 3 + i)
                  pure (ForInStep.yield (__s_1.push __do_lift))
            if (!(isPow2 words && col % words == 0)) = true then do
                throw "flock-ir-unit/v2: a port group is not an aligned power of two of words"
                pure (ForInStep.yield (__s.fst + 3 + np, __s.snd.push { col := col, words := words, bits := __s_1 }))
              else pure (ForInStep.yield (__s.fst + 3 + np, __s.snd.push { col := col, words := words, bits := __s_1 }))) =
      .ok (at_ + (gs.flatMap gNums).length, acc ++ gs.toArray)
  | [], k, at_, acc, _, _ => by simp <;> rfl
  | g :: gs, k, at_, acc, hh, hal => by
    rw [List.flatMap_cons] at hh
    obtain ⟨hg, hr⟩ := at_append hh
    have hc : Net.num h at_ = .ok g.col := by simpa [gNums] using at_num hg 0 (by simp [gNums])
    have hw : Net.num h (at_ + 1) = .ok g.words := by simpa [gNums] using at_num hg 1 (by simp [gNums])
    have hn : Net.num h (at_ + 2) = .ok g.bits.size := by simpa [gNums] using at_num hg 2 (by simp [gNums])
    have hb := bits_loop (h := h) (at_ + 3) g.bits.toList 0 #[] (fun j hj => by
      have := hg (3 + j) g.bits.toList[j] (by rw [gNums, List.getElem?_append_right (by simp)]; simp)
      simpa [Nat.add_assoc] using this)
    rw [Array.length_toList, Array.empty_append, Array.toArray_toList] at hb
    have e : at_ + (gNums g).length = at_ + 3 + g.bits.size := by simp [gNums] <;> omega
    rw [e] at hr
    have ha : (isPow2 g.words && g.col % g.words == 0) = true := hal g List.mem_cons_self
    rw [List.length_cons, List.range'_succ, List.forIn_cons]
    dsimp only
    rw [hc, ok_bind, hw, ok_bind, hn, ok_bind, hb, ok_bind, ha]
    simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, pure_bind]
    rw [groups_loop gs (k + 1) (at_ + 3 + g.bits.size) (acc.push g) hr
      (fun x hx => hal x (List.mem_cons_of_mem _ hx))]
    simp only [List.flatMap_cons, List.length_append, ← e]
    congr 2
    · omega
    · simp

def side (gs : Array Group) : List Nat := gs.size :: gs.toList.flatMap gNums

theorem groups_ok {h : Array String} {at0 : Nat} {gs : Array Group} (hh : At h at0 (side gs))
    (hal : ∀ g ∈ gs.toList, Aligned g) : Net.groups h at0 = .ok (gs, at0 + (side gs).length) := by
  unfold Net.groups
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size]
  have hcount : Net.num h at0 = .ok gs.size := by simpa [side] using at_num hh 0 (by simp [side])
  rw [hcount, ok_bind]
  obtain ⟨-, hr⟩ := at_append (L := [gs.size]) (M := gs.toList.flatMap gNums) hh
  rw [← Array.length_toList, groups_loop gs.toList 0 (at0 + 1) #[] hr hal, ok_bind]
  simp only [side, List.length_cons, Array.empty_append, Array.toArray_toList]
  rw [show at0 + 1 + (gs.toList.flatMap gNums).length = at0 + ((gs.toList.flatMap gNums).length + 1) by omega]
  rfl

theorem at_of (A B : List String) (L : List Nat) : At (A ++ L.map toString ++ B).toArray A.length L := by
  intro j n hj
  have hl := (List.getElem?_eq_some_iff.1 hj).1
  rw [List.getElem?_toArray, List.append_assoc, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left,
    List.getElem?_append_left (by simpa using hl), List.getElem?_map, hj]
  rfl

def hdr (name : String) (l : Laid) : List String :=
  [VStar.FORMAT, name, toString l.useful, toString l.constPos] ++ (side l.inGroups).map toString ++
    (side l.outGroups).map toString

theorem header_words (name : String) (l : Laid) : VStar.header name l = " ".intercalate (hdr name l) := by
  unfold VStar.header hdr side gNums
  simp [List.map_flatMap]

theorem format_word : Tok VStar.FORMAT := ⟨by decide, by simp [NoWs, VStar.FORMAT]⟩

theorem header_tokens {name : String} (hw : Tok name) (l : Laid) :
    tokens (VStar.header name l) = (hdr name l).toArray := by
  rw [header_words, tokens_intercalate]
  intro x hx
  simp only [hdr, List.mem_append, List.mem_cons, List.mem_map, List.not_mem_nil, or_false] at hx
  rcases hx with ((rfl | rfl | rfl | rfl) | ⟨n, -, rfl⟩) | ⟨n, -, rfl⟩
  · exact format_word
  · exact hw
  all_goals exact word_nat _

/-! ## The header checks `ofRows` passed -/

theorem ofRows_checks {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)}
    {n : Net} (h : Net.ofRows name useful constPos ig og ra rb = .ok n) :
    (∀ g ∈ (ig ++ og).toList, Aligned g) ∧ og.isEmpty = false ∧ (constPos + 1 != useful) = false ∧
      (ig.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD ||
          g.col + g.words > (ig.back?.map fun g => g.col + g.words).getD 0)
        || og.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD)) = false ∧
      ¬ ((ig.back?.map fun g => g.col + g.words).getD 0 * WORD > constPos) := by
  unfold Net.ofRows at h
  simp only at h
  by_cases h1 : (ig ++ og).any (fun g => !(isPow2 g.words && g.col % g.words == 0)) = true
  · rw [if_pos h1] at h; cases h
  rw [if_neg h1] at h
  by_cases h2 : og.isEmpty = true
  · rw [if_pos h2] at h; cases h
  rw [if_neg h2] at h
  by_cases h3 : (constPos + 1 != useful) = true
  · rw [if_pos h3] at h; cases h
  rw [if_neg h3] at h
  by_cases h4 : (ig.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD ||
          g.col + g.words > (ig.back?.map fun g => g.col + g.words).getD 0)
        || og.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD)) = true
  · rw [if_pos h4] at h; cases h
  rw [if_neg h4] at h
  by_cases h5 : (ig.back?.map fun g => g.col + g.words).getD 0 * WORD > constPos
  · rw [if_pos h5] at h; cases h
  refine ⟨fun g hg => ?_, by simpa using h2, by simpa using h3, by simpa using h4, h5⟩
  unfold Aligned
  cases hc : (isPow2 g.words && g.col % g.words == 0)
  · exact absurd (Array.any_eq_true'.mpr ⟨g, Array.mem_toList_iff.mp hg, by simp [hc]⟩) h1
  · rfl

/-! ## The row loop -/

theorem forIn_steps {β : Type} {f : Nat → β → Except String (ForInStep β)} {step : Nat → β → β} : ∀ (m k : Nat) (b : β),
    (∀ j < m, ∀ b, f (k + j) b = .ok (.yield (step (k + j) b))) →
    forIn (List.range' k m) b f = .ok ((List.range' k m).foldl (fun b i => step i b) b)
  | 0, _, _, _ => rfl
  | m + 1, k, b, hf => by
    rw [List.range'_succ, List.forIn_cons, List.foldl_cons]
    have h0 := hf 0 (by omega) b
    rw [Nat.add_zero] at h0
    rw [h0, ok_bind]
    exact forIn_steps m (k + 1) (step k b) fun j hj b => by
      have := hf (j + 1) (by omega) b; rwa [show k + (j + 1) = k + 1 + j by omega] at this

theorem foldl_push {F G : Nat → Array Nat} : ∀ (m k : Nat) (b : Array (Array Nat) × Array (Array Nat)),
    (List.range' k m).foldl (fun b i => (b.1.push (F i), b.2.push (G i))) b =
      (b.1 ++ ((List.range' k m).map F).toArray, b.2 ++ ((List.range' k m).map G).toArray)
  | 0, _, b => by simp
  | m + 1, k, b => by
    rw [List.range'_succ, List.foldl_cons, foldl_push m (k + 1)]
    simp

theorem map_getD_range (a : Array (Array Nat)) : ((List.range' 0 a.size).map (a.getD · #[])).toArray = a := by
  apply Array.toList_inj.mp
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_range', Nat.zero_add, Array.getD_eq_getD_getElem?]
  simp at h1
  simp [Array.getElem?_eq_getElem h1]

theorem rowLine_tokens (a b : Array Nat) : tokens (VStar.rowLine a b) = ((rowNums a b).map toString).toArray := by
  rw [rowLine_eq, tokens_intercalate]
  intro x hx
  obtain ⟨n, -, rfl⟩ := List.mem_map.mp hx
  exact word_nat n

theorem mapM_nums {f : String → Except String Nat} (hf : ∀ n : Nat, f (toString n) = .ok n) :
    ∀ L : List Nat, (L.map toString).mapM f = .ok L
  | [] => rfl
  | x :: L => by rw [List.map_cons, List.mapM_cons, hf, ok_bind, mapM_nums hf L]; rfl

theorem row_ok (a b : Array Nat) {f : String → Except String Nat} (hf : ∀ n : Nat, f (toString n) = .ok n) :
    (tokens (VStar.rowLine a b)).mapM f = .ok (rowNums a b).toArray := by
  rw [rowLine_tokens, Array.mapM_eq_mapM_toList, List.toList_toArray, mapM_nums hf]
  rfl

theorem rowNums_facts (a b : Array Nat) :
    (rowNums a b).toArray.getD 0 0 = a.size ∧ (rowNums a b).toArray.getD (1 + a.size) 0 = b.size ∧
      (rowNums a b).toArray.size = 2 + a.size + b.size ∧ (rowNums a b).toArray.extract 1 (1 + a.size) = a ∧
      (rowNums a b).toArray.extract (2 + a.size) (2 + a.size + b.size) = b := by
  let v := (rowNums a b).toArray
  show v.getD 0 0 = a.size ∧ v.getD (1 + a.size) 0 = b.size ∧ v.size = 2 + a.size + b.size ∧
      v.extract 1 (1 + a.size) = a ∧ v.extract (2 + a.size) (2 + a.size + b.size) = b
  refine ⟨by simp [v, rowNums], ?_, by simp [v, rowNums] <;> omega, ?_, ?_⟩
  · simp only [v, rowNums, Array.getD_eq_getD_getElem?, List.getElem?_toArray]
    rw [List.getElem?_append_right (by simp <;> omega)]
    simp [Nat.add_comm]
  · apply Array.toList_inj.mp
    simp [v, rowNums]
  · apply Array.toList_inj.mp
    simp only [v, rowNums]
    simp
    rw [show 2 + a.size = a.toList.length + 1 + 1 by simp; omega, List.drop_succ_cons, List.drop_append]
    simp
    rw [List.drop_eq_nil_of_le (by simp)]
    simp only [List.nil_append]
    exact List.take_of_length_le (by simp)

/-! ## The round trip -/

theorem netLs_getD0 (name : String) (l : Laid) : (netLs name l).toArray.getD 0 "" = VStar.header name l := by
  simp [netLs]

theorem netLs_size (name : String) (l : Laid) : (netLs name l).toArray.size = 1 + l.useful := by
  simp [netLs, rowsLs, Nat.add_comm]

theorem netLs_row (name : String) (l : Laid) {j : Nat} (hj : j < l.useful) :
    (netLs name l).toArray.getD (1 + j) "" = VStar.rowLine (l.ra.getD j #[]) (l.rb.getD j #[]) := by
  simp [netLs, rowsLs, Array.getD_eq_getD_getElem?, Nat.add_comm 1, hj]

theorem netLs_end (name : String) (l : Laid) : (netLs name l).toArray.getD (1 + l.useful) "" = "" := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by rw [netLs_size] <;> omega)]
  rfl

theorem parse_netLs {name : String} {l : Laid} {n : Net} (hw : Tok name) (ha : l.ra.size = l.useful)
    (hb : l.rb.size = l.useful) (h : netOf name l = .ok n) : Net.parse (netLs name l).toArray = .ok n := by
  have h' := h
  unfold netOf at h'
  obtain ⟨hal, c2, c3, c4, c5⟩ := ofRows_checks h'
  have hi : ∀ g ∈ l.inGroups.toList, Aligned g := fun g hg => hal g (by simp at hg ⊢; exact Or.inl hg)
  have ho : ∀ g ∈ l.outGroups.toList, Aligned g := fun g hg => hal g (by simp at hg ⊢; exact Or.inr hg)
  have hh : tokens ((netLs name l).toArray.getD 0 "") = (hdr name l).toArray := by
    rw [netLs_getD0, header_tokens hw]
  have e0 : ((hdr name l).toArray[0]? != some "flock-ir-unit/v2") = false := by simp [hdr, VStar.FORMAT]
  have n2 : Net.num (hdr name l).toArray 2 = .ok l.useful := num_of (by simp [hdr])
  have n3 : Net.num (hdr name l).toArray 3 = .ok l.constPos := num_of (by simp [hdr])
  have g1 : Net.groups (hdr name l).toArray 4 = .ok (l.inGroups, 4 + (side l.inGroups).length) :=
    groups_ok (at_of [VStar.FORMAT, name, toString l.useful, toString l.constPos] ((side l.outGroups).map toString)
      (side l.inGroups)) hi
  have g2 : Net.groups (hdr name l).toArray (4 + (side l.inGroups).length) =
      .ok (l.outGroups, 4 + (side l.inGroups).length + (side l.outGroups).length) := by
    have := at_of ([VStar.FORMAT, name, toString l.useful, toString l.constPos] ++ (side l.inGroups).map toString) []
      (side l.outGroups)
    rw [List.append_nil, List.length_append, List.length_map] at this
    exact groups_ok this ho
  have e1 : (4 + (side l.inGroups).length + (side l.outGroups).length != (hdr name l).toArray.size) = false := by
    simp [hdr] <;> omega
  have hL : "".startsWith "LEAVES " = false := by simp
  have hC : "".startsWith "CUT " = false := by simp
  have e2 : (1 + l.useful != (netLs name l).toArray.size) = false := by rw [netLs_size]; simp
  have e3 : (hdr name l).toArray.getD 1 "" = name := by simp [hdr]
  have hra : ((List.range' 0 l.useful).map (l.ra.getD · #[])).toArray = l.ra := by
    rw [← ha]; exact map_getD_range _
  have hrb : ((List.range' 0 l.useful).map (l.rb.getD · #[])).toArray = l.rb := by
    rw [← hb]; exact map_getD_range _
  unfold Net.parse
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size]
  rw [hh]
  simp only [e0, Bool.false_eq_true, ↓reduceIte]
  rw [n2, ok_bind, n3, ok_bind, g1, ok_bind]
  dsimp only
  rw [g2, ok_bind]
  dsimp only
  simp only [e1, c2, c3, c4, if_neg c5, Bool.false_eq_true, ↓reduceIte]
  rw [forIn_steps (step := fun i b => (b.1.push (l.ra.getD i #[]), b.2.push (l.rb.getD i #[]))) l.useful 0 (#[], #[])
    ?rows, ok_bind, foldl_push]
  case rows =>
    intro j hj b
    rw [netLs_row name l (by omega), row_ok _ _ (fun n => by
      show (match (toString n).toNat? with | some n => _ | none => _) = _
      rw [show (toString n).toNat? = some n from Nat.toNat?_repr n]
      rfl), ok_bind]
    obtain ⟨v0, v1, vs, x1, x2⟩ := rowNums_facts (l.ra.getD (0 + j) #[]) (l.rb.getD (0 + j) #[])
    rw [v0, v1, vs, x1, x2]
    simp <;> rfl
  dsimp only
  rw [Array.empty_append, Array.empty_append, hra, hrb, netLs_end, hL, hC, e2, e3]
  simp only [Bool.false_eq_true, ↓reduceIte]
  exact h'

end FlockDeploy.ConcreteNet
