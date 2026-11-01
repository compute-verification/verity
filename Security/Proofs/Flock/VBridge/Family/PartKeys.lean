import Proofs.Flock.VBridge.Family.OutCols
import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.VBridge.Family.PartReads

/-!
# A part's input keys, from V*'s META

`RecVStar.checkParsed` (`VStarFamily.hparsed`) holds a part circuit's input leaves and row ports to the ones V* wrote,
so the Δ key (`Aliased.srcKey`, copy 0) of each of the unit's input bits is the bit of V*'s port it lies in:

* `vstarRows_get`: the ports parsed as V*'s input rows have their words and bits, and start where V*'s do;
* `srcKey_laid`: at the verifier's `unitLaid`, the key of input bit `16 · (words before q) + k` is bit `k` of port `q`
  (`srcKey_inOrder`);
* `repKeys_part`: part `p`'s circuit has those keys at V*'s part `pt.S`'s ports (`RepKeys`);
* `srcKey_v`, `srcKey_row`: so the `v` column `vBase + q` is bit `q` of the port V* names `v`, and the row column
  `rowCol q` is a bit of the row port it lies in (`rowCol_at`), named as V* names it;
* `part_counts`, `part_link`: V*'s part `pt.S` has the verifier's part's row ports and v (`nsOf F.sh p`, `nvOf F.sh p`;
  `VStarFamily.hpart`, `RecVStar.checkPart`), so the keys hold at the verifier's part (`partRep`, which `rowCol` and
  `vBase` read only through those counts): `partVCols` and `partRowCols` (`PartRowCols`);
* its row ports are named the whole S's at the part's ports (`RecVStar.portNames`, `partPorts`), so a row column of
  element `a` is keyed by that element's bit of its port in w, named as that port (`wIdx_get`, `rowBit`), and two
  parts' keys of one element's bit are one bit of rows of one name (`starts_inj`, `partRowBits_holds`).
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge

/-- **V*'s input rows, port by port**: ports whose (name, words, bits) begin with V*'s input rows of words `ws` have
their names, words and 16 bits a word, and start past the words of the rows before them. -/
theorem vstarRows_get {ports : Array Flock.Port} {ins : List String} {ws : List ℕ} {rest : List (String × ℕ × ℕ)}
    (h : ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      (ins.zip ws).map (fun (nm, w) => (nm, w, 16 * w)) ++ rest)
    (hlen : ins.length = ws.length) {q : ℕ} (hq : q < ws.length) :
    ∃ hq' : q < ports.size, ports[q].name = ins[q]! ∧ ports[q].words = ws[q] ∧ ports[q].bits = 16 * ws[q] ∧
      ((ports.toList.take q).map (·.words)).sum = (ws.take q).sum := by
  have hz : (ins.zip ws).length = ws.length := by rw [List.length_zip, hlen, Nat.min_self]
  have hsz : ws.length ≤ ports.size := by
    have := congrArg List.length h
    rw [List.length_map, List.length_append, List.length_map, hz, Array.length_toList] at this
    omega
  refine ⟨by omega, ?_⟩
  have hget : ∀ i (hi : i < ws.length), (ports[i].name, ports[i].words, ports[i].bits) =
      (ins[i]!, ws[i], 16 * ws[i]) := by
    intro i hi
    have e := congrArg (·[i]?) h
    simp only at e
    have hzi : i < (ins.zip ws).length := by rw [hz]; exact hi
    have hpi : i < ports.toList.length := by rw [Array.length_toList]; omega
    rw [List.getElem?_map, List.getElem?_append_left (by rw [List.length_map, hz]; exact hi), List.getElem?_map,
      List.getElem?_eq_getElem hzi, List.getElem?_eq_getElem hpi, List.getElem_zip, Array.getElem_toList,
      Option.map_some, Option.map_some, Option.some.injEq] at e
    rw [e, getElem!_pos ins i (by omega)]
  have hh := hget q hq
  simp only [Prod.mk.injEq] at hh
  refine ⟨hh.1, hh.2.1, hh.2.2, congrArg List.sum (List.ext_getElem
    (by simp only [List.length_map, List.length_take, Array.length_toList]; omega) fun i h1 h2 => ?_)⟩
  simp only [List.length_map, List.length_take] at h1
  have hi := hget i (by omega)
  simp only [Prod.mk.injEq] at hi
  simp only [List.getElem_map, List.getElem_take, Array.getElem_toList]
  exact hi.2.1

theorem take_sum_add_le (l : List ℕ) {q : ℕ} (hq : q < l.length) : (l.take q).sum + l[q] ≤ l.sum := by
  have e : l.sum = (l.take (q + 1)).sum + (l.drop (q + 1)).sum := by rw [← List.sum_append, List.take_append_drop]
  rw [e, List.take_add_one, List.sum_append, List.getElem?_eq_getElem hq]
  simp only [Option.toList_some, List.sum_cons, List.sum_nil]
  omega

/-- **The verifier's `rec-residuals` unit keys its inputs by the rows V* wrote**: at `unitLaid` (one group of 16-bit
ports from column 0), with one unit per VU whose leaves are its input words in order and ports V*'s input rows of
words `ws`, the key of input bit `16 · (ws before q) + k` is bit `k` of port `q` (`srcKey_inOrder`). -/
theorem srcKey_laid {c : Flock.Circuit} (hU : Built c.unit) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit)
    (hleaves : c.leavesIn.size = c.upv ∧
      ∀ u < c.upv, (c.leavesIn[u]!).size = (c.unit.inGroups.getD 0 default).bits.size) (hupv : c.upv = 1)
    {ins : List String} {ws : List ℕ} {rest : List (String × ℕ × ℕ)}
    (hlv : c.leavesIn.getD 0 #[] = (Array.range (ws.foldl (· + ·) 0)).map Int.ofNat)
    (hports : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      (ins.zip ws).map (fun (nm, w) => (nm, w, 16 * w)) ++ rest)
    (hlen : ins.length = ws.length) {q : ℕ} (hq : q < ws.length) :
    ∃ hq' : q < c.ports.size, c.ports[q].name = ins[q]! ∧ c.ports[q].words = ws[q] ∧
      ∀ k < 16 * ws[q], Aliased.srcKey c (16 * (ws.take q).sum + k) = .msg q (k / 8) (k % 8) := by
  obtain ⟨hq', hnm, hw, hb, hsum⟩ := vstarRows_get hports hlen hq
  refine ⟨hq', hnm, hw, fun k hk => ?_⟩
  have hg : c.unit.inGroups = #[⟨0, groupWords ((partIn ns nv).foldl (· + ·) 0), partIn ns nv⟩] :=
    (Flock.Net.ofRows_ok hn).2.2.2.2.2.1.trans (layout_inGroups _ _ _ _)
  have h16 : ∀ b ∈ (partIn ns nv).toList, b = 16 := fun b hb => by
    unfold partIn at hb
    rw [Array.toList_replicate] at hb
    exact List.eq_of_mem_replicate hb
  have h0 : 0 < c.leavesIn.size := by rw [hleaves.1, hupv]; exact Nat.one_pos
  have hl0 := hleaves.2 0 (by rw [hupv]; exact Nat.one_pos)
  rw [hg, getElem!_pos c.leavesIn 0 h0] at hl0
  rw [Array.getD_eq_getD_getElem?, getElem?_pos c.leavesIn 0 h0, Option.getD_some] at hlv
  have hsz : ws.foldl (· + ·) 0 = (partIn ns nv).size := by
    rw [hlv, Array.size_map, Array.size_range] at hl0
    exact hl0
  have hN : 16 * (partIn ns nv).size ≤ c.unit.inWords * Flock.WORD := by
    rw [inWords_unitLaid hn, inRowsOf]
    unfold partIn
    rw [foldl_replicate, Array.size_replicate, Nat.mul_comm 16]
    exact le_groupWords _
  have hst : (FlockSoundness.Typed.starts c)[q]! = (ws.take q).sum := by
    rw [FlockSoundness.Discharge.Layout.starts_get c hq', hsum]
  have hfit : (FlockSoundness.Typed.starts c)[q]! + c.ports[q].words ≤ (partIn ns nv).size := by
    rw [hst, hw, ← hsz, foldl_add, Nat.zero_add]
    exact take_sum_add_le ws hq
  have key := FlockC3.RegLevels.srcKey_inOrder hg rfl h16 (Aliased.portBits_nodup hU) (j := 0)
    (by rw [Array.getD_eq_getD_getElem?, getElem?_pos c.leavesIn 0 h0, Option.getD_some, hlv, hsz])
    hq' (by rw [hw]; exact hk) (by rw [hb]; exact hk) hfit hN
  rw [hst, Nat.zero_mul, Nat.zero_add] at key
  exact key

/-- **A circuit keys V*'s part `S`'s input ports** (`InnerRepCheck{S}`'s, `repPorts`): port `q` has V*'s name and words,
and the key of input bit `16 · (words before q) + k` is its bit `k`. -/
def RepKeys (c : Flock.Circuit) (S : VStar.RepS) : Prop :=
  ∀ q (hq : q < (repPorts S).length), ∃ hq' : q < c.ports.size,
    c.ports[q].name = (S.ports.map (·.1) ++ ["v"])[q]! ∧ c.ports[q].words = (repPorts S)[q] ∧
      ∀ k < 16 * (repPorts S)[q], Aliased.srcKey c (16 * ((repPorts S).take q).sum + k) = .msg q (k / 8) (k % 8)

theorem repPorts_length (S : VStar.RepS) : (repPorts S).length = S.ports.length + 1 := by
  rw [repPorts, List.length_append, List.length_map, List.length_singleton]

theorem repPorts_take (S : VStar.RepS) {i : ℕ} (hi : i ≤ S.ports.length) :
    (repPorts S).take i = (S.ports.map fun p => VStar.portWordsOf p.2).take i := by
  rw [repPorts, List.take_append_of_le_length (by rw [List.length_map]; exact hi)]

theorem repPorts_v (S : VStar.RepS) (h : S.ports.length < (repPorts S).length) :
    (repPorts S)[S.ports.length] = VStar.portWordsOf S.gf.nv := by
  simp only [repPorts, List.getElem_append, List.length_map, Nat.lt_irrefl, dite_false, Nat.sub_self,
    List.getElem_singleton]

theorem names_v (S : VStar.RepS) : (S.ports.map (·.1) ++ ["v"])[S.ports.length]! = "v" := by
  rw [getElem!_pos _ _ (by rw [List.length_append, List.length_map, List.length_singleton]; omega),
    List.getElem_append_right (by simp only [List.length_map, Nat.le_refl])]
  simp only [List.length_map, Nat.sub_self, List.getElem_singleton]

/-- **`v`'s columns at V*'s part**: port `S.ports.length` is `v`, and the key of column `vBase S + q`, for `q` within
`v`'s `128 nv` bits, is its bit `q`, within its row. -/
theorem srcKey_v {c : Flock.Circuit} {S : VStar.RepS} (h : RepKeys c S) (n : ℕ) :
    ∃ qv : Fin c.ports.size, c.ports[qv].name = "v" ∧ ∀ q < 128 * S.gf.nv, ∃ _ : q < Layout.portRowW c qv,
      Aliased.srcKey c (0 * n + (vBase S + q)) = .msg qv.val (q / 8) (q % 8) := by
  have hl : S.ports.length < (repPorts S).length := by rw [repPorts_length]; omega
  obtain ⟨hq', hnm, hw, hkey⟩ := h _ hl
  have hv : 16 * ((repPorts S).take S.ports.length).sum = vBase S := by
    rw [repPorts_take S (Nat.le_refl _), List.take_of_length_le (by simp only [List.length_map, Nat.le_refl]), vBase]
  have hb : ∀ q < 128 * S.gf.nv, q < 16 * (repPorts S)[S.ports.length] := fun q hq => by
    rw [repPorts_v S hl]
    exact Nat.lt_of_lt_of_le hq (le_portWords _)
  refine ⟨⟨_, hq'⟩, hnm.trans (names_v S), fun q hq => ⟨?_, ?_⟩⟩
  · show q < 16 * c.ports[S.ports.length].words
    rw [hw]
    exact hb q hq
  · rw [Nat.zero_mul, Nat.zero_add, ← hv]
    exact hkey q (hb q hq)

theorem repPorts_get (S : VStar.RepS) {i : ℕ} (hi : i < S.ports.length) (h : i < (repPorts S).length) :
    (repPorts S)[i] = VStar.portWordsOf S.ports[i].2 := by
  simp only [repPorts, List.getElem_append, List.length_map, hi, dite_true, List.getElem_map]

theorem names_row (S : VStar.RepS) {i : ℕ} (hi : i < S.ports.length) :
    (S.ports.map (·.1) ++ ["v"])[i]! = S.ports[i].1 := by
  rw [getElem!_pos _ _ (by rw [List.length_append, List.length_map, List.length_singleton]; omega),
    List.getElem_append_left (by rw [List.length_map]; exact hi), List.getElem_map]

/-- **A row column `rowCol` lies in one row port**: bit `q` of the row ports is bit `e` of port `i`, past the ports
before it. -/
theorem rowCol_at : ∀ (ps : List (String × ℕ)) (k q : ℕ), q < 128 * (ps.map (·.2)).sum →
    ∃ i, ∃ hi : i < ps.length, ∃ e < 128 * ps[i].2, q = 128 * ((ps.map (·.2)).take i).sum + e ∧
      rowCol ps k q = k + 16 * ((ps.map fun p => VStar.portWordsOf p.2).take i).sum + e
  | [], _, _, h => by
    simp only [List.map_nil, List.sum_nil] at h
    omega
  | p :: ps, k, q, h => by
    unfold rowCol
    by_cases hq : q < 128 * p.2
    · refine ⟨0, Nat.succ_pos _, q, by simp only [List.getElem_cons_zero]; exact hq,
        by simp only [List.take_zero, List.sum_nil]; omega, ?_⟩
      rw [ite_eq_left hq]
      simp only [List.take_zero, List.sum_nil]
      omega
    · rw [ite_eq_right hq]
      have h' : q - 128 * p.2 < 128 * (ps.map (·.2)).sum := by
        simp only [List.map_cons, List.sum_cons] at h
        omega
      obtain ⟨i, hi, e, he, hqe, hr⟩ := rowCol_at ps (k + 16 * VStar.portWordsOf p.2) (q - 128 * p.2) h'
      refine ⟨i + 1, Nat.succ_lt_succ hi, e, by simp only [List.getElem_cons_succ]; exact he, ?_, ?_⟩
      · simp only [List.map_cons, List.take_succ_cons, List.sum_cons]
        omega
      · rw [hr]
        simp only [List.map_cons, List.take_succ_cons, List.sum_cons]
        omega

/-- **The row columns at V*'s part**: row column `rowCol S.ports 0 q` is keyed by bit `e` of row port `i`, named as
V* names it, where `q` is bit `e` of the `i`-th row's `128 n` bits. -/
theorem srcKey_row {c : Flock.Circuit} {S : VStar.RepS} (h : RepKeys c S) (n : ℕ) {q : ℕ}
    (hq : q < 128 * (S.ports.map (·.2)).sum) :
    ∃ (i : Fin c.ports.size) (hi : i.val < S.ports.length) (e : ℕ) (_ : e < 128 * S.ports[i.val].2),
      q = 128 * ((S.ports.map (·.2)).take i).sum + e ∧ c.ports[i].name = S.ports[i.val].1 ∧
        e < Layout.portRowW c i ∧ Aliased.srcKey c (0 * n + rowCol S.ports 0 q) = .msg i.val (e / 8) (e % 8) := by
  obtain ⟨i, hi, e, he, hqe, hr⟩ := rowCol_at S.ports 0 q hq
  have hil : i < (repPorts S).length := by rw [repPorts_length]; omega
  obtain ⟨hq', hnm, hw, hkey⟩ := h i hil
  have hb : e < 16 * (repPorts S)[i] := by
    rw [repPorts_get S hi _]
    exact Nat.lt_of_lt_of_le he (le_portWords _)
  refine ⟨⟨i, hq'⟩, hi, e, he, hqe, hnm.trans (names_row S hi), ?_, ?_⟩
  · show e < 16 * c.ports[i].words
    rw [hw]
    exact hb
  · rw [Nat.zero_mul, Nat.zero_add, hr, Nat.zero_add, ← repPorts_take S (Nat.le_of_lt hi)]
    exact hkey e hb

/-- `rowCol` reads only the ports' element counts. -/
theorem rowCol_snd : ∀ (ps ps' : List (String × ℕ)), ps.map (·.2) = ps'.map (·.2) → ∀ k q,
    rowCol ps k q = rowCol ps' k q
  | [], [], _, _, _ => rfl
  | [], _ :: _, h, _, _ => by simp at h
  | _ :: _, [], h, _, _ => by simp at h
  | p :: ps, p' :: ps', h, k, q => by
    simp only [List.map_cons, List.cons.injEq] at h
    unfold rowCol
    rw [h.1, rowCol_snd ps ps' h.2]

/-- So does `vBase`. -/
theorem vBase_snd {S S' : VStar.RepS} (h : S.ports.map (·.2) = S'.ports.map (·.2)) : vBase S = vBase S' := by
  have e : ∀ S : VStar.RepS, S.ports.map (fun p => VStar.portWordsOf p.2) = (S.ports.map (·.2)).map VStar.portWordsOf :=
    fun S => by simp only [List.map_map, Function.comp_def]
  rw [vBase, vBase, e, e, h]

theorem partRep_snd (ns : List ℕ) (nv : ℕ) : (partRep ns nv).ports.map (·.2) = ns := by
  simp only [partRep, List.map_map, Function.comp_def, List.map_id']

/-- The verifier's part `p` is one of its groups. -/
theorem part_lt {sh : Flock.Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecAlgebra.structureOf sh = .ok S) (hg : RecResiduals.groups S (RecResiduals.portSizes sh) = .ok gs) {p : ℕ}
    {x : RecOpen.GfS × List ℕ × ℕ} (h : RecResiduals.part sh p = .ok x) : p < gs.size := by
  unfold RecResiduals.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hS] at hS0
  cases hS0
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs0, hg0, h⟩ := FlockVBridge.PartS.bind_ok h
    rw [hg] at hg0
    cases hg0
    split at h
    · rename_i hj
      obtain ⟨hp, -⟩ := Array.getElem?_eq_some_iff.mp hj
      exact hp
    · cases h

/-- The verifier's part `p`'s row ports are its `p`-th group's. -/
theorem partPorts_eq {sh : Flock.Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecAlgebra.structureOf sh = .ok S) (hg : RecResiduals.groups S (RecResiduals.portSizes sh) = .ok gs) {p : ℕ}
    (hp : p < gs.size) : RecVStar.partPorts sh p = .ok gs[p].2.1 := by
  unfold RecVStar.partPorts
  rw [hS]
  dsimp only [bind, Except.bind]
  rw [hg]
  dsimp only
  rw [getElem?_pos gs p hp]
  rfl

/-- An element of a `flatMap` past the blocks before its own. -/
theorem flatMap_getElem? {α β : Type} (g : α → List β) (len : α → ℕ) (hl : ∀ x, (g x).length = len x) :
    ∀ (l : List α) (i o : ℕ) (hi : i < l.length), o < len l[i] →
      (l.flatMap g)[((l.map len).take i).sum + o]? = (g l[i])[o]?
  | [], _, _, hi, _ => absurd hi (Nat.not_lt_zero _)
  | x :: l, 0, o, _, ho => by
    simp only [List.flatMap_cons, List.map_cons, List.take_zero, List.sum_nil, Nat.zero_add,
      List.getElem_cons_zero] at ho ⊢
    exact List.getElem?_append_left (by rw [hl]; exact ho)
  | x :: l, i + 1, o, hi, ho => by
    simp only [List.flatMap_cons, List.map_cons, List.take_succ_cons, List.sum_cons, List.getElem_cons_succ] at ho ⊢
    rw [Nat.add_assoc, List.getElem?_append_right (by rw [hl]; exact Nat.le_add_right _ _), hl,
      Nat.add_sub_cancel_left]
    exact flatMap_getElem? g len hl l i o (Nat.lt_of_succ_lt_succ hi) ho

/-- **`wIdx`'s element `o` of its `i`-th port** is that port's start in w plus `o`. -/
theorem wIdx_get (sizes P : Array ℕ) {i o : ℕ} (hi : i < P.size) (ho : o < sizes[P[i]]!) {a : ℕ}
    (ha : a < (PartS.wIdx sizes P).size) (hae : a = ((P.toList.map (sizes[·]!)).take i).sum + o) :
    (PartS.wIdx sizes P)[a] = (RecResiduals.starts sizes)[P[i]]! + o := by
  have e := flatMap_getElem? (fun p => ((Array.range sizes[p]!).map ((RecResiduals.starts sizes)[p]! + ·)).toList)
    (sizes[·]!) (fun x => by simp) P.toList i o (by simpa using hi) (by simpa using ho)
  rw [← hae] at e
  have e' : (PartS.wIdx sizes P)[a]? = some ((RecResiduals.starts sizes)[P[i]]! + o) := by
    rw [← Array.getElem?_toList, PartS.wIdx, Array.toList_flatMap, e]
    simp [ho]
  rw [getElem?_pos (PartS.wIdx sizes P) a ha] at e'
  exact Option.some.inj e'

theorem take_sum_le : ∀ (ns : List ℕ) {a b : ℕ}, a ≤ b → (ns.take a).sum ≤ (ns.take b).sum
  | [], _, _, _ => by simp
  | _ :: _, 0, _, _ => by simp
  | _ :: _, _ + 1, 0, h => absurd h (Nat.not_succ_le_zero _)
  | x :: ns, _ + 1, _ + 1, h => by
    simp only [List.take_succ_cons, List.sum_cons]
    exact Nat.add_le_add_left (take_sum_le ns (Nat.le_of_succ_le_succ h)) x

theorem take_sum_lt (ns : List ℕ) {i i' e : ℕ} (hi : i < ns.length) (he : e < ns[i]) (h : i < i') :
    (ns.take i).sum + e < (ns.take i').sum := by
  have e1 : (ns.take (i + 1)).sum = (ns.take i).sum + ns[i] := by
    rw [List.take_add_one, List.sum_append, List.getElem?_eq_getElem hi]
    simp only [Option.toList_some, List.sum_cons, List.sum_nil, Nat.add_zero]
  have := take_sum_le ns (show i + 1 ≤ i' from h)
  omega

/-- **One position of blocks `ns` laid end to end** is one block's and one offset in it. -/
theorem take_sum_inj (ns : List ℕ) {i i' e e' : ℕ} (hi : i < ns.length) (hi' : i' < ns.length) (he : e < ns[i])
    (he' : e' < ns[i']) (h : (ns.take i).sum + e = (ns.take i').sum + e') : i = i' ∧ e = e' := by
  rcases Nat.lt_trichotomy i i' with hlt | rfl | hgt
  · have := take_sum_lt ns hi he hlt
    omega
  · exact ⟨rfl, by omega⟩
  · have := take_sum_lt ns hi' he' hgt
    omega

/-- **One slot of w is one port's element**: past each port's start (`starts`), within it. -/
theorem starts_inj (sizes : Array ℕ) {x y o o' : ℕ} (ho : o < sizes[x]!) (ho' : o' < sizes[y]!)
    (h : (RecResiduals.starts sizes)[x]! + o = (RecResiduals.starts sizes)[y]! + o') : x = y ∧ o = o' := by
  have hx : x < sizes.size := by
    by_cases hn : x < sizes.size
    · exact hn
    · rw [getElem!_neg sizes x hn] at ho; exact absurd ho (Nat.not_lt_zero _)
  have hy : y < sizes.size := by
    by_cases hn : y < sizes.size
    · exact hn
    · rw [getElem!_neg sizes y hn] at ho'; exact absurd ho' (Nat.not_lt_zero _)
  rw [Family.starts_get sizes (Nat.le_of_lt hx), Family.starts_get sizes (Nat.le_of_lt hy)] at h
  rw [getElem!_pos sizes x hx, ← Array.getElem_toList] at ho
  rw [getElem!_pos sizes y hy, ← Array.getElem_toList] at ho'
  exact take_sum_inj sizes.toList (by simpa using hx) (by simpa using hy) ho ho' h

/-- **A part's row column of bit `e` of its row-port element `a`, keyed to bit `b` of port `q`**: element `a` is
element `o` of the whole S's row port `x` (`wIdx_get`), `b` is bit `e` of it, and `q` is named as `x` (`srcKey_row`). -/
theorem rowBit_of {c : Flock.Circuit} {S : VStar.RepS} (h : RepKeys c S) (n : ℕ) {sizes P : Array ℕ}
    {names : Array String} (hns : S.ports.map (·.2) = P.toList.map (sizes[·]!))
    (hnm : S.ports.map (·.1) = P.toList.map (names[·]!)) {a e : ℕ} (ha : a < (PartS.wIdx sizes P).size) (he : e < 128)
    {q : Fin c.ports.size} {b : ℕ}
    (hk : Aliased.srcKey c (0 * n + rowCol S.ports 0 (128 * a + e)) = .msg q.val (b / 8) (b % 8)) :
    ∃ x o : ℕ, o < sizes[x]! ∧ (PartS.wIdx sizes P)[a] = (RecResiduals.starts sizes)[x]! + o ∧ b = 128 * o + e ∧
      c.ports[q].name = names[x]! := by
  have hq : 128 * a + e < 128 * (S.ports.map (·.2)).sum := by
    have := wIdx_size sizes P
    rw [hns]
    omega
  obtain ⟨i, hi, e0, he0, hqe, hname, -, hk0⟩ := srcKey_row h n hq
  rw [hk0] at hk
  simp only [Aliased.Src.msg.injEq] at hk
  obtain ⟨hiq, h1, h2⟩ := hk
  have hb : b = e0 := by omega
  have hqi : q = i := Fin.ext hiq.symm
  subst hqi
  have hlen : S.ports.length = P.size := by simpa using congrArg List.length hns
  have hiP : q.val < P.size := hlen ▸ hi
  have hpi : S.ports[q.val].2 = sizes[P[q.val]]! := by
    have h' := List.getElem_of_eq hns (by rw [List.length_map]; exact hi)
    rw [List.getElem_map, List.getElem_map, Array.getElem_toList] at h'
    exact h'
  have hni : S.ports[q.val].1 = names[P[q.val]]! := by
    have h' := List.getElem_of_eq hnm (by rw [List.length_map]; exact hi)
    rw [List.getElem_map, List.getElem_map, Array.getElem_toList] at h'
    exact h'
  refine ⟨P[q.val], e0 / 128, by rw [← hpi]; omega,
    wIdx_get sizes P hiP (by rw [← hpi]; omega) ha (by rw [← hns]; omega), by omega, hname.trans hni⟩

end FlockVBridge.OfVStar

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **Part `p`'s circuit keys V*'s part `pt.S`'s ports** (`srcKey_laid`): its unit is the verifier's `unitLaid`
(`check_part`, at `HUnitName`), and its leaves and row ports are the ones V* wrote (`hparsed`, `checkParsed_ok`), whose
input rows are `pt.S`'s (`innerUnit_ok`). -/
theorem repKeys_part (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) : OfVStar.RepKeys (F.Vs j).c pt.S := by
  have hname := F.hUnitName_of hne
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname j p nm pt hj).trans (statements_name F.hstmts _ hj)
  have P := parse_facts (F.zs R ω [] R₂ j).hc
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := P.recResiduals
  obtain ⟨S, ns, nv, -, hn⟩ :=
    FlockVBridge.OfVStar.check_part hchk (resParams_of_check hchk hnm) F.hsh (F.hplain j)
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨u, hu, hupv, -, hlv, hports⟩ := checkParsed_ok (F.hparsed j)
  rw [hj] at hu hports
  obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.innerUnit_ok (show VStar.innerUnit pt.S nm = .ok u from hu)
  unfold RecVStar.vstarRows at hports
  rw [VStar.UnitC.inWords, hin] at hlv
  rw [hin] at hports
  intro q hq
  exact OfVStar.srcKey_laid (F.Vs j).hU hn (check_facts P.typed hls).leaves hupv hlv hports
    (by
      show (pt.S.ports.map (·.1) ++ ["v"]).length = (OfVStar.repPorts pt.S).length
      rw [OfVStar.repPorts_length, List.length_append, List.length_map, List.length_singleton]) hq

/-- **V*'s part is the verifier's part**: part `p`'s statement's row ports' elements and v elements are those of the
verifier's part `p` at its shape, and its row ports are named the whole S's at the part's ports (`hpart`,
`checkPart_ok`, at the unit name and plain META `check_part` reads). -/
theorem part_counts (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part F.sh p = .ok (S, ns, nv)) :
    pt.S.ports.map (·.2) = ns ∧ pt.S.gf.nv = nv ∧ ∀ P, RecVStar.partPorts F.sh p = .ok P →
      pt.S.ports.map (·.1) = P.toList.map ((RecVStar.portNames F.sh)[·]!) := by
  have hname := F.hUnitName_of hne
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname j p nm pt hj).trans (statements_name F.hstmts _ hj)
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := (parse_facts (F.zs R ω [] R₂ j).hc).recResiduals
  have e : ∀ s : Flock.Shape, s.holo = false → { s with holo := false } = s := by
    intro s h0
    cases s
    simp only at h0
    subst h0
    rfl
  have h := checkPart_ok (F.hpart j) hj (resParams_of_check hchk hnm) F.hsh (F.hplain j)
    (by rw [e F.sh (OfVStar.shape?_holo F.hsh)]; exact hpart)
  rw [e F.sh (OfVStar.shape?_holo F.hsh)] at h
  exact h

/-- **The verifier's part `p`'s row ports and v are V*'s part's**, and V*'s part's row ports are named the whole S's
at `p`'s ports (`part_counts` at the part `check_part` gives, which is `nsOf` and `nvOf`'s, `part_eq`, and whose ports
are `partsOf`'s, `partPorts_eq`). -/
theorem part_link (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) :
    nsOf F.sh p = pt.S.ports.map (·.2) ∧ nvOf F.sh p = pt.S.gf.nv ∧
      pt.S.ports.map (·.1) = (partsOf F.sh)[p]!.2.1.toList.map ((RecVStar.portNames F.sh)[·]!) := by
  have hname := F.hUnitName_of hne
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname j p nm pt hj).trans (statements_name F.hstmts _ hj)
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := (parse_facts (F.zs R ω [] R₂ j).hc).recResiduals
  obtain ⟨S, ns, nv, hpart, -⟩ := OfVStar.check_part hchk (resParams_of_check hchk hnm) F.hsh (F.hplain j)
  obtain ⟨h1, h2, h3⟩ := F.part_counts ⟨(R, ω, R₂)⟩ hj hpart
  obtain ⟨GS, gs, hS, hg⟩ := FlockC3.Reps.ReadWhole.part_ok hpart
  have hp := OfVStar.part_lt hS hg hpart
  obtain ⟨-, hns, hnv⟩ := OfVStar.part_eq hS hg hp hpart
  have hgp : (partsOf F.sh)[p]! = gs[p] := by rw [partsOf_eq hS hg, getElem!_pos gs p hp]
  exact ⟨by rw [nsOf, hgp, ← hns, h1], by rw [nvOf, hgp, ← hnv, h2],
    by rw [hgp]; exact h3 _ (OfVStar.partPorts_eq hS hg hp)⟩

/-- **The `v` columns' keys at part `p`'s statement**: `srcKey_v` at V*'s part (`repKeys_part`), which is the verifier's
(`part_link`). -/
theorem partVCols (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) :
    ∃ qv : Fin (F.Vs j).c.ports.size, (F.Vs j).c.ports[qv].name = "v" ∧
      ∀ q < 128 * nvOf F.sh p, ∃ _ : q < Layout.portRowW (F.Vs j).c qv,
        Aliased.srcKey (F.Vs j).c
            (0 * (Rows.ofNet (F.Vs j).hU).nIn + (OfVStar.vBase (OfVStar.partRep (nsOf F.sh p) (nvOf F.sh p)) + q)) =
          .msg qv.val (q / 8) (q % 8) := by
  obtain ⟨hns, hnv, -⟩ := F.part_link hne hj
  rw [hns, hnv, OfVStar.vBase_snd (OfVStar.partRep_snd _ _)]
  exact OfVStar.srcKey_v (F.repKeys_part hne hj) _

/-- **`PartRowCols` at part `p`'s statement**: `srcKey_row` at V*'s part (`repKeys_part`), which is the verifier's
(`part_link`). -/
theorem partRowCols (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) :
    ∀ q < 128 * (nsOf F.sh p).sum,
      ∃ (q' : Fin (F.Vs j).c.ports.size) (k' : ℕ) (_ : k' < Layout.portRowW (F.Vs j).c q'),
        Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + rowColOf F.sh p q) =
          .msg q'.val (k' / 8) (k' % 8) := by
  obtain ⟨hns, -, -⟩ := F.part_link hne hj
  intro q hq
  rw [hns] at hq
  obtain ⟨i, -, e, -, -, -, he, hk⟩ := OfVStar.srcKey_row (F.repKeys_part hne hj) (Rows.ofNet (F.Vs j).hU).nIn hq
  refine ⟨i, e, he, ?_⟩
  rw [rowColOf, hns, OfVStar.rowCol_snd _ pt.S.ports (OfVStar.partRep_snd _ _)]
  exact hk

/-- **`PartRowCols` holds** at every family with a history. -/
theorem partRowCols_holds (hne : Nonempty (Reg × L.Ω × Reg₂)) : F.PartRowCols :=
  fun _ _ _ _ hj => F.partRowCols hne hj

/-- **A row column of part `p`'s statement is keyed by its element's bit of its port in w**: `rowBit_of` at V*'s part
(`repKeys_part`), which is the verifier's, named the whole S's at `p`'s ports (`part_link`). -/
theorem rowBit (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) {a e : ℕ} (ha : a < (wIOf F.sh p).size) (he : e < 128)
    {q : Fin (F.Vs j).c.ports.size} {b : ℕ}
    (hk : Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + rowColOf F.sh p (128 * a + e)) =
      .msg q.val (b / 8) (b % 8)) :
    ∃ x o : ℕ, o < (RecResiduals.portSizes F.sh)[x]! ∧
      (wIOf F.sh p)[a] = (RecResiduals.starts (RecResiduals.portSizes F.sh))[x]! + o ∧ b = 128 * o + e ∧
        (F.Vs j).c.ports[q].name = (RecVStar.portNames F.sh)[x]! := by
  obtain ⟨hns, -, hnm⟩ := F.part_link hne hj
  rw [rowColOf, hns, OfVStar.rowCol_snd _ pt.S.ports (OfVStar.partRep_snd _ _)] at hk
  exact OfVStar.rowBit_of (F.repKeys_part hne hj) _ hns.symm hnm ha he hk

/-- **`PartRowBits` holds** at every family with a history: two parts' keys of one element's bit are that element's
bit of its port in w (`rowBit`), one port and one element of it (`starts_inj`). -/
theorem partRowBits_holds (hne : Nonempty (Reg × L.Ω × Reg₂)) : F.PartRowBits := by
  intro p p' j j' nm pt nm' pt' hj hj' a a' e ha ha' hw he q b q' b' hk hk' _ _
  obtain ⟨x, o, ho, hx, hb, hn⟩ := F.rowBit hne hj ha he hk
  obtain ⟨x', o', ho', hx', hb', hn'⟩ := F.rowBit hne hj' ha' he hk'
  obtain ⟨rfl, rfl⟩ := OfVStar.starts_inj _ ho ho' (hx.symm.trans (hw.trans hx'))
  exact ⟨hb.trans hb'.symm, hn.trans hn'.symm⟩

end VStarFamily

end family

end FlockVBridge.Family
