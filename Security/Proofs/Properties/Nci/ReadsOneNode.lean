import Proofs.Properties.Nci.ReadsExact

/-!
# `ReadsExact.RootOneNode`, proved: a root read's gates are one root node's

A segment keeps its gates by node (`NodeW`) when each piece's gates are one node's (`InNode`) and, when its copies
shift, all its copies' gates are one node's. As `ReadsExact`'s widths hold `down`, `fromCallee`, `throughArg` and
`carry` to name gates with their widths at every fuel (`widths_of`), `oneNode_of` holds them to keep gates by node, by
the same induction on fuel, wherever the callees' summaries name their gates with their widths (`RetW`): a callee's
gate is one of its gates (`locW_lt`), so shifted to a member's first gate it is the member's node's (`inNode_at`).

* `down` keeps a gate's or parameter's piece as it is (`downN_gate`, `downN_param`), and a primitive node's and a
  batch of a primitive's name the node's gates (`downN_prim`, `downN_batchPrim`);
* `fromCallee` shifts its callee's return into one member, a segment whole (`nodeW_shift`) or piece by piece
  (`nodeW_oneShift`), alike copies of a member's pieces merged (`nodeW_alike`);
* a batch's copies of a member's return, a copy a member, are all its node's (`nodeW_copies`), and so is its one-run
  piece (`batchAffine_n`).

`rootN` reads it at the root, through `carriesRead_of`, and `rootOneNode_of` turns its nodes into the root's
(`nodeGates_of_inNode`): with the decoder's facts, `RootOneNode` holds (`BodiesOk.rootOneNode`).
-/

namespace NciCheck.Properties.SecurityProofs.ReadsExact
open NciCheck.Desc (Piece Seg PS Tgt)
open Flock.Program (Prog)
open DescAgree (nodeCount rootOffs)

/-- The gates `xs` names are node `k`'s of Definition `d`'s body: each at or past the node's first and before the next
node's. -/
def InNode (P : Prog) (d k : Nat) (xs : List (Tgt × Int)) : Prop :=
  ∀ w p, (Tgt.gate w, p) ∈ xs →
    (((P.defs[d]!).body.offsets[k]! : Nat) : Int) ≤ p ∧ p < (((P.defs[d]!).body.offsets[k + 1]! : Nat) : Int)

/-- A segment of Definition `d`'s body keeping its gates by node: each piece's gates are one node's, and, when its
copies shift, all its copies' gates are one node's. -/
def NodeW (P : Prog) (d : Nat) (s : Seg) : Prop :=
  (∀ q ∈ s.ps.toList, ∃ k, InNode P d k (pos q)) ∧ (1 < s.reps → s.shift ≠ 0 → ∃ k, InNode P d k (segPos s))

theorem inNode_mono {P : Prog} {d k : Nat} {xs ys : List (Tgt × Int)} (h : InNode P d k ys)
    (hs : ∀ x ∈ xs, x ∈ ys) : InNode P d k xs :=
  fun w p hx => h w p (hs _ hx)

/-- A piece of references names no gate, of any node. -/
theorem inNode_ref {P : Prog} {d k : Nat} {q : Piece} {b : Bool} {i : Nat} (ht : q.tgt = .ref b i) :
    InNode P d k (pos q) := by
  intro w p hx
  have := mem_pos_tgt hx
  rw [ht] at this
  cases this

/-- A piece's positions are its segment's first copy's, of a copy at least. -/
theorem pos_sub_segPos {s : Seg} (h : s.reps ≠ 0) {q : Piece} (hq : q ∈ s.ps.toList) : ∀ x ∈ pos q, x ∈ segPos s := by
  intro x hx
  unfold segPos
  refine List.mem_flatMap.2 ⟨0, List.mem_range.2 (Nat.pos_of_ne_zero h), ?_⟩
  rw [copy_pos, Nat.cast_zero, Int.zero_mul, shiftPos_zero, List.map_id]
  exact List.mem_flatMap.2 ⟨q, hq, hx⟩

/-- A segment of a copy at least whose gates are all one node's keeps its gates by node. -/
theorem nodeW_of_whole {P : Prog} {d k : Nat} {s : Seg} (h : s.reps ≠ 0) (hk : InNode P d k (segPos s)) :
    NodeW P d s :=
  ⟨fun _ hq => ⟨k, inNode_mono hk (pos_sub_segPos h hq)⟩, fun _ _ => ⟨k, hk⟩⟩

/-- One piece whose gates are one node's keeps its gates by node. -/
theorem nodeW_one {P : Prog} {d k : Nat} {q : Piece} (h : InNode P d k (pos q)) : NodeW P d (Seg.one q) := by
  refine ⟨fun g hg => ⟨k, ?_⟩, fun h1 => absurd h1 (Nat.lt_irrefl 1)⟩
  have e : g = q := by simpa [Seg.one] using hg
  subst e
  exact h

/-- **A member's gate is its node's**: in a laid-out body, gate `j` of member `m` of node `k`'s activation,
`offsets[k] + m * gates + j`, is node `k`'s. -/
theorem inNode_child (P : Prog) {d k m j : Nat} (hoff : Laid P d) (hk : k < (P.defs[d]!).body.nodes.size)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members) (hj : j < (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates) :
    (((P.defs[d]!).body.offsets[k]! : Nat) : Int) ≤
        (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j : Nat) : Int) ∧
      (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + j : Nat) : Int) <
        (((P.defs[d]!).body.offsets[k + 1]! : Nat) : Int) := by
  rw [hoff k hk]
  have hle := Nat.mul_le_mul_right (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates hm
  rw [Nat.succ_mul] at hle
  constructor <;> omega

/-- `inNode_child` at an integer position `o + v`, `o` member `m`'s first gate and `v` one of its activation's. -/
theorem inNode_at (P : Prog) {d k m : Nat} {o v : Int} (hoff : Laid P d) (hk : k < (P.defs[d]!).body.nodes.size)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members)
    (ho : o = (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Nat) : Int))
    (h0 : 0 ≤ v) (h1 : v < ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int)) :
    (((P.defs[d]!).body.offsets[k]! : Nat) : Int) ≤ v + o ∧
      v + o < (((P.defs[d]!).body.offsets[k + 1]! : Nat) : Int) := by
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le h0
  have := inNode_child P hoff hk hm (j := n) (by omega)
  rw [ho, ← Int.natCast_add, Nat.add_comm n]
  exact this

/-- `down` keeps a gate's piece as it is, of one node. -/
theorem downN_gate (c : Desc.Ctx) (fuel : Nat) (q : Piece) {w : Nat} (ht : q.tgt = .gate w)
    (hq : ∃ k', InNode c.P c.d k' (pos q)) {segs : Array Seg}
    (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  rw [down_gate c fuel q ht] at h
  injection h with h
  subst h
  intro s hs
  have e : s = Seg.one q := by simpa using hs
  subst e
  obtain ⟨k', hk'⟩ := hq
  exact nodeW_one hk'

/-- `down` keeps a parameter's piece as it is, naming no gate. -/
theorem downN_param (c : Desc.Ctx) (fuel : Nat) (q : Piece) {i : Nat} (ht : q.tgt = .ref true i)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  rw [down_param c fuel q ht] at h
  injection h with h
  subst h
  intro s hs
  have e : s = Seg.one q := by simpa using hs
  subst e
  exact nodeW_one (k := 0) (inNode_ref ht)

/-- A primitive node's piece names the node's gate, in a laid-out body. -/
theorem downN_prim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .prim) {w : Nat}
    (hret : (c.P.defs[node.fn]!).ret = .value w) (hg1 : (c.P.defs[node.fn]!).gates = 1)
    (hoff : ∀ k' < c.body.nodes.size, c.body.offsets[k' + 1]! =
      c.body.offsets[k']! + (c.body.nodes[k']!).members * (c.P.defs[(c.body.nodes[k']!).fn]!).gates)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  have hk := node_lt hn
  have hg := node_get hn
  rw [down_prim c fuel q ht hff hn hform hret] at h
  injection h with h
  subst h
  split
  · intro s hs
    simp at hs
  · intro s hs
    have e : s = Seg.one { tgt := .gate w, base := (c.body.offsets[k]! : Int), stride := 0, len := q.count } := by
      simpa using hs
    subst e
    refine nodeW_one (k := k) ?_
    intro w' p hx
    unfold pos at hx
    simp only [List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq, Tgt.gate.injEq] at hx
    obtain ⟨r, -, t, -, rfl, rfl⟩ := hx
    simp only [Int.mul_zero, Int.add_zero]
    have hm1 : node.members = 1 := by
      unfold Flock.Program.Node.members
      rw [hform]
      rfl
    have hx : c.body.offsets[k]! < c.body.offsets[k + 1]! := by
      rw [hoff k hk, hg, hm1, hg1]
      omega
    have hx' : ((c.body.offsets[k]! : Nat) : Int) < ((c.body.offsets[k + 1]! : Nat) : Int) := by exact_mod_cast hx
    exact ⟨le_rfl, hx'⟩

/-- A batch of a primitive's piece names its members' gates, node `k`'s, in a laid-out body: its positions at or past
0 and within the node's output, of no more leaves than members. -/
theorem downN_batchPrim (c : Desc.Ctx) (fuel : Nat) (q : Piece) {k : Nat} (ht : q.tgt = .ref false k)
    (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node) (hform : node.form = .batch)
    (hp : (c.P.defs[node.fn]!).prim = true) {w : Nat} (hret : (c.P.defs[node.fn]!).ret = .value w)
    (hg1 : (c.P.defs[node.fn]!).gates = 1)
    (hoff : ∀ k' < c.body.nodes.size, c.body.offsets[k' + 1]! =
      c.body.offsets[k']! + (c.body.nodes[k']!).members * (c.P.defs[(c.body.nodes[k']!).fn]!).gates)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hF : ∀ x ∈ pos q, Fits c x) (hout : node.out.leaves ≤ node.members)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  have hk := node_lt hn
  have hg := node_get hn
  rw [down_batchPrim c fuel q ht hff hn hform hp hret] at h
  injection h with h
  subst h
  split
  · intro s hs
    simp at hs
  · intro s hs
    have e : s = Seg.one { q with tgt := .gate w, base := (c.body.offsets[k]! : Int) + q.base } := by
      simpa using hs
    subst e
    refine nodeW_one (k := k) ?_
    intro w' p hx
    unfold pos at hx
    simp only [List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq, Tgt.gate.injEq] at hx
    obtain ⟨r, hr, t, htl, rfl, rfl⟩ := hx
    have hmem : (q.tgt, q.base + (r : Int) * q.rowStride + (t : Int) * q.stride) ∈ pos q := by
      unfold pos
      simp only [List.mem_flatMap, List.mem_map, List.mem_range]
      exact ⟨r, hr, t, htl, rfl⟩
    have h0 := hq _ hmem
    have h1 := hF _ hmem k ht
    rw [hg] at h1
    obtain ⟨n, hn'⟩ := Int.eq_ofNat_of_zero_le h0
    have hpn : (c.body.offsets[k]! : Int) + q.base + (r : Int) * q.rowStride + (t : Int) * q.stride =
        ((c.body.offsets[k]! + n : Nat) : Int) := by
      push_cast
      simp only at hn'
      linarith
    rw [hpn]
    have hlt : n < node.members := by
      have : (n : Int) < node.out.leaves := by simp only at hn' h1; rw [← hn']; exact h1
      omega
    have hx : c.body.offsets[k]! ≤ c.body.offsets[k]! + n ∧ c.body.offsets[k]! + n < c.body.offsets[k + 1]! := by
      rw [hoff k hk, hg, hg1]
      omega
    have hx' : ((c.body.offsets[k]! : Nat) : Int) ≤ ((c.body.offsets[k]! + n : Nat) : Int) ∧
        ((c.body.offsets[k]! + n : Nat) : Int) < ((c.body.offsets[k + 1]! : Nat) : Int) := by
      constructor <;> omega
    exact hx'

/-- **A callee's segment as member `m`'s** (`fromCallee`'s `shiftGates`): of a copy at least, its gates the callee's,
shifted to the member's first gate they are node `k`'s, in a laid-out body. -/
theorem nodeW_shift (P : Prog) {d k m : Nat} {o : Int} (hoff : Laid P d) (hk : k < (P.defs[d]!).body.nodes.size)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members)
    (ho : o = (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Nat) : Int))
    {s : Seg} (hr : s.reps ≠ 0)
    (hin : ∀ w p, (Tgt.gate w, p) ∈ segPos s → 0 ≤ p ∧ p < ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int)) :
    NodeW P d { ps := Desc.shiftGates o s.ps, len := s.len, reps := s.reps, shift := s.shift } := by
  refine nodeW_of_whole (k := k) hr ?_
  intro w p hx
  rw [segPos_shift] at hx
  obtain ⟨⟨t, v⟩, hxs, he⟩ := List.mem_map.1 hx
  cases t with
  | ref b i => cases he
  | gate w0 =>
    simp only [shiftPos, Prod.mk.injEq, Tgt.gate.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    obtain ⟨h0, h1⟩ := hin w0 v hxs
    exact inNode_at P hoff hk hm ho h0 h1

/-- **A callee's piece of gates as member `m`'s** (`fromCallee`'s `{ g with base := o + g.base }`): its gates the
callee's, they are node `k`'s, in a laid-out body. -/
theorem nodeW_oneShift (P : Prog) {d k m : Nat} {o : Int} (hoff : Laid P d) (hk : k < (P.defs[d]!).body.nodes.size)
    (hm : m < ((P.defs[d]!).body.nodes[k]!).members)
    (ho : o = (((P.defs[d]!).body.offsets[k]! + m * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Nat) : Int))
    {g : Piece}
    (hin : ∀ w p, (Tgt.gate w, p) ∈ pos g → 0 ≤ p ∧ p < ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int)) :
    NodeW P d (Seg.one { g with base := o + g.base }) := by
  refine nodeW_one (k := k) ?_
  intro w p hx
  unfold pos at hx
  simp only [List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq] at hx
  obtain ⟨r, hr, t, htl, hgt, rfl⟩ := hx
  have hmem : (Tgt.gate w, g.base + (r : Int) * g.rowStride + (t : Int) * g.stride) ∈ pos g := by
    unfold pos
    simp only [List.mem_flatMap, List.mem_map, List.mem_range]
    exact ⟨r, hr, t, htl, by rw [hgt]⟩
  obtain ⟨h0, h1⟩ := hin w _ hmem
  have := inNode_at P hoff hk hm ho h0 h1
  rw [show o + g.base + (r : Int) * g.rowStride + (t : Int) * g.stride =
    g.base + (r : Int) * g.rowStride + (t : Int) * g.stride + o by ring]
  exact this

/-- **Copies of a callee's gates, a copy a member** (`shiftGates` to member `m0`'s first gate, each next copy `dm`
members on): of a copy at least, of the callee's gates at or past 0, each copy a member of node `k`, they are node
`k`'s, in laid-out bodies. -/
theorem nodeW_copies (P : Prog) {d k : Nat}
    (hoff : Laid P d) (hk : k < (P.defs[d]!).body.nodes.size)
    (hLc : Laid P ((P.defs[d]!).body.nodes[k]!).fn)
    (hlast : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.offsets[
        (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.nodes.size]! =
      (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates)
    {ps : Array Piece} {m0 : Nat} {δ dm S : Int} {len reps : Nat} (hr : reps ≠ 0)
    (hδ : δ = ((P.defs[d]!).body.offsets[k]! : Int) +
      (m0 : Int) * ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int))
    (hS : S = dm * ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates : Int))
    (hW : ∀ w v, (Tgt.gate w, v) ∈ posAll ps → 0 ≤ v ∧ LocW P ((P.defs[d]!).body.nodes[k]!).fn v.toNat w)
    (hin : ∀ u < reps, 0 ≤ (m0 : Int) + (u : Int) * dm ∧
      (m0 : Int) + (u : Int) * dm < (((P.defs[d]!).body.nodes[k]!).members : Int)) :
    NodeW P d { ps := Desc.shiftGates δ ps, len := len, reps := reps, shift := S } := by
  refine nodeW_of_whole (k := k) hr ?_
  intro w p hx
  unfold segPos at hx
  obtain ⟨u, hu, hx⟩ := List.mem_flatMap.1 hx
  rw [List.mem_range] at hu
  simp only [copy_pos, shiftGates_pos, List.map_map, shiftPos_shiftPos, List.mem_map] at hx
  obtain ⟨⟨t, v⟩, hy, he⟩ := hx
  cases t with
  | ref b i => simp [shiftPos] at he
  | gate w0 =>
    simp only [shiftPos, Prod.mk.injEq, Tgt.gate.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    obtain ⟨hv0, hl⟩ := hW w0 v hy
    obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le hv0
    rw [Int.toNat_natCast] at hl
    have hnl := locW_lt hl hLc hlast
    obtain ⟨h0, h1⟩ := hin u hu
    obtain ⟨mm, hmm⟩ := Int.eq_ofNat_of_zero_le h0
    have hmlt : mm < ((P.defs[d]!).body.nodes[k]!).members := by omega
    have key := inNode_child P hoff hk hmlt hnl
    have e : (n : Int) + (δ + (u : Int) * S) = (((P.defs[d]!).body.offsets[k]! +
        mm * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates + n : Nat) : Int) := by
      rw [hδ, hS]
      push_cast
      rw [← hmm]
      ring
    rw [e]
    exact key

/-- Copies of one member's segments, each keeping the body's gates by node, alike: a segment keeping them by node, its
copies not shifting. -/
theorem nodeW_alike (P : Prog) (d : Nat) {one : Array Seg} (hW : ∀ s ∈ one.toList, NodeW P d s) (len reps : Nat) :
    NodeW P d { ps := one.flatMap (·.ps), len := len, reps := reps, shift := 0 } := by
  refine ⟨fun q hq => ?_, fun _ h => absurd rfl h⟩
  rw [Array.toList_flatMap, List.mem_flatMap] at hq
  obtain ⟨s, hs, hq⟩ := hq
  exact (hW s hs).1 q hq

/-- What `throughArg` gives of member `m`'s parameter `i`'s leaves `g`, each in range, keeps the body's gates by
node. -/
def ThroughN (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m i g segs, m < (c.body.nodes[k]!).members → g.tgt = .ref true i →
    (∀ x ∈ pos g, 0 ≤ x.2 ∧ i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size ∧
      x.2 < ((((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves : Int))) →
    Desc.throughArg c fuel k m i g = .ok segs → ∀ s ∈ segs.toList, NodeW c.P c.d s

/-- What `fromCallee` gives of member `m` of a composite node keeps the body's gates by node. -/
def CalleeN (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m a b segs, m < (c.body.nodes[k]!).members → (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false →
    Desc.fromCallee c fuel k m a b = .ok segs → ∀ s ∈ segs.toList, NodeW c.P c.d s

/-- **`fromCallee`'s node step**: one step up, `fromCallee`'s segments keep the body's gates by node wherever its
callees' summaries hold `RetOk` and name their gates with their widths (`RetW`), the bodies are laid out, and
`throughArg`'s segments one step down keep the body's gates by node, each of a copy at least (`ThroughReps`). -/
theorem calleeN_succ (c : Desc.Ctx) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn) (hL : Laid c.P c.d)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates)
    (hTW : ThroughN c fuel) : CalleeN c (fuel + 1) := by
  intro k m a b segs hm hp h
  rw [Desc.fromCallee.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  have hnode : c.body.nodes[k]! = node := node_get hn
  have hk : k < c.body.nodes.size := node_lt hn
  subst hnode
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  obtain ⟨hps, hpos, hgo, -, hpp⟩ := hR k _ hn
  have hrw := hRW k _ hn
  obtain ⟨hLf, hlast⟩ := hLF k _ hn hp
  have hslice := psSlice_pos _ hps a b
  have hmemS : ∀ s ∈ ((c.memo[(c.body.nodes[k]!).fn]!).ret.slice a b).toList, ∀ x ∈ segPos s,
      x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret := by
    intro s hs x hx
    have : x ∈ segsPos ((c.memo[(c.body.nodes[k]!).fn]!).ret.slice a b) := List.mem_flatMap.2 ⟨s, hs, hx⟩
    rw [hslice] at this
    exact List.mem_of_mem_drop (List.mem_of_mem_take this)
  have ho : ((c.body.offsets[k]! : Int) + (m : Int) * ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int)) =
      ((c.body.offsets[k]! + m * (c.P.defs[(c.body.nodes[k]!).fn]!).gates : Nat) : Int) := by push_cast; rfl
  have hin : ∀ w p, (Tgt.gate w, p) ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret →
      0 ≤ p ∧ p < ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int) := by
    intro w p hx
    have h0 := (hpos _ hx).1
    refine ⟨h0, ?_⟩
    have := locW_lt (hrw w p hx) hLf hlast
    omega
  have hfa := forIn_app (NodeW c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨out, rfl, hout⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hout s)
  intro s hs r y hy
  obtain ⟨hrep1, hgates⟩ := retOk_slice hgo hs
  have hx := hmemS s hs
  have hone1 : ∀ g, (∀ x ∈ pos g, x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret) →
      NodeW c.P c.d (Seg.one { g with base := ((c.body.offsets[k]! : Int) +
        (m : Int) * ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int)) + g.base }) := fun g hg =>
    nodeW_oneShift c.P hL hk hm ho (fun w p hxm => hin w p (hg _ hxm))
  split at hy
  · rename_i hr
    split at hy
    · rename_i hall
      have hgs := gates_of_all hall
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
      simp only [List.mem_singleton] at hs'
      subst hs'
      exact nodeW_shift c.P hL hk hm ho (by omega) (fun w p hxm => hin w p (hx _ hxm))
    rename_i hnot
    have hsh : s.shift = 0 := by
      rcases hgates hr with hg | hg
      · refine absurd ?_ hnot
        rw [Array.all_eq_true]
        intro i hi
        obtain ⟨w, hw⟩ := hg _ (Array.getElem_mem_toList hi)
        simp only [hw]
      · exact hg
    rw [if_neg (by simp [hsh])] at hy
    obtain ⟨one, hone, hy⟩ := Walk.except_bind_ok hy
    rw [← Array.forIn_toList] at hone
    have hxg : ∀ g ∈ s.ps.toList, ∀ x ∈ pos g, x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret := fun g hg x hxm =>
      hx x (by
        rw [segPos_alike hsh]
        exact List.mem_flatMap.2 ⟨0, List.mem_range.2 (by omega), List.mem_flatMap.2 ⟨g, hg, hxm⟩⟩)
    obtain ⟨one'', h1'', h2''⟩ := forIn_app (NodeW c.P c.d) _ s.ps.toList (fun g hg r' y' hy' => by
        split at hy'
        · simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
          simp only [List.mem_singleton] at hs'
          subst hs'
          exact hone1 g (hxg g hg)
        · exact absurd hy' (by intro h; cases h)
        · rename_i i hi
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          refine hTW k m i g v' hm hi (fun x hxm => ?_) hv'
          obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
          obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
          exact ⟨h0, h1, h2⟩) #[] one hone
    have hone_w : ∀ s' ∈ one.toList, NodeW c.P c.d s' := by
      rw [h1'', Array.empty_append]; exact h2''
    split at hy
    · simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
      simp only [List.mem_singleton] at hs'
      subst hs'
      exact nodeW_alike c.P c.d hone_w s.len s.reps
    · have hy := walkAtMost_bind _ _ _ hy
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
      obtain ⟨segs', hsegs', hA2⟩ := forIn_app (NodeW c.P c.d) _ _ (fun _ _ r' y' hy' => by
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          exact ⟨one, hy'.symm, hone_w⟩) r v hv
      exact ⟨segs', by rw [hsegs'], hA2⟩
  · rename_i hr
    have h1 : s.reps = 1 := by omega
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    rw [← Array.forIn_toList] at hv
    have hxg : ∀ g ∈ s.ps.toList, ∀ x ∈ pos g, x ∈ psPos (c.memo[(c.body.nodes[k]!).fn]!).ret := fun g hg x hxm =>
      hx x (by rw [segPos_one_rep h1]; exact List.mem_flatMap.2 ⟨g, hg, hxm⟩)
    obtain ⟨segs', rfl, hA'⟩ := forIn_app (NodeW c.P c.d) _ s.ps.toList (fun g hg r' y' hy' => by
        split at hy'
        · simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨#[_], congrArg _ Array.push_eq_append, fun s' hs' => ?_⟩
          simp only [List.mem_singleton] at hs'
          subst hs'
          exact hone1 g (hxg g hg)
        · exact absurd hy' (by intro h; cases h)
        · rename_i i hi
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          refine hTW k m i g v' hm hi (fun x hxm => ?_) hv'
          obtain ⟨h0, -, hp⟩ := hpos x (hxg g hg x hxm)
          obtain ⟨h1, h2⟩ := hp i ((mem_pos_tgt hxm).trans hi)
          exact ⟨h0, h1, h2⟩) r v hv
    exact ⟨segs', rfl, hA'⟩

/-- `down` by node at one fuel: of a piece of positive count, its positions at or past 0, naming its nodes' output
leaves and its parameters' leaves in range and its gates one node's, each segment `down` gives keeps the body's gates by
node. -/
def DownNAt (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs, 0 < q.count → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) →
    (∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int)) →
    (∃ k', InNode c.P c.d k' (pos q)) →
    Desc.down c fuel q = .ok segs → ∀ s ∈ segs.toList, NodeW c.P c.d s

/-- What `carry` gives of scan iteration `m`'s return leaves `[a, b)`, within its step's return when not empty, keeps
the body's gates by node. -/
def CarryN (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ k m a b segs, (c.body.nodes[k]!).form = .scan → m < (c.body.nodes[k]!).members →
    (a < b → b ≤ (c.P.defs[(c.body.nodes[k]!).fn]!).body.ret.count) →
    Desc.carry c fuel k m a b = .ok segs → ∀ s ∈ segs.toList, NodeW c.P c.d s

/-- Node `k`'s argument `i`'s positions `[lo, hi)` through `down`: segments keeping the body's gates by node,
the argument naming references. -/
theorem one_n (c : Desc.Ctx) (fuel : Nat) (hD : DownNAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {lo hi : Nat} (hhi : lo < hi → hi ≤ ((c.body.nodes[k]!).args[i]!).seq.count) {out : Array Seg}
    (h : oneOf c fuel k i lo hi = .ok out) : ∀ s ∈ out.toList, NodeW c.P c.d s := by
  unfold oneOf at h
  dsimp only at h
  rw [← Array.forIn_toList] at h
  have hp := pieces_pos _ hA.1 lo hi
  rw [hA.2, sl_map_range' _ _ _ _ hhi] at hp
  have hmem : ∀ q ∈ ((c.arg k i).pieces lo hi).toList, ∀ x ∈ pos q,
      ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count, x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro q hq x hx
    have : x ∈ posAll ((c.arg k i).pieces lo hi) := List.mem_flatMap.2 ⟨q, hq, hx⟩
    rw [hp] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, by rw [List.mem_range'_1] at ht; omega, rfl⟩
  have hcnt : ∀ q ∈ ((c.arg k i).pieces lo hi).toList, 0 < q.count := by
    intro q hq
    unfold Desc.PS.pieces at hq
    rw [Array.toList_flatMap, List.mem_flatMap] at hq
    obtain ⟨g, hg, hq⟩ := hq
    rw [Array.toList_flatMap, List.mem_flatMap] at hq
    obtain ⟨m, -, hq⟩ := hq
    exact shiftGates_count _ _ (psSlice_ok _ hA.1 hP lo hi g hg).2.2 q hq
  exact loop_mem _ (Desc.down c fuel) _ (fun q hq segs hs => hD q segs (hcnt q hq)
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem q hq x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem q hq x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem q hq x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩)
    ⟨0, fun w p hx => by
      obtain ⟨t, -, he⟩ := hmem q hq _ hx
      simp [refPos] at he⟩ hs) out h

/-- **`throughArg`'s node step**: one step up, what member `m`'s parameter leaves read through node `k`'s argument `i`
give keeps the body's gates by node wherever `down`'s and `carry`'s segments do (`throughShape_succ`'s argument). -/
theorem throughN_succ (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (fuel : Nat) (hD : DownNAt c fuel) (hK : CarryN c fuel) : ThroughN c (fuel + 1) := by
  intro k m i g segs hm hg hx h
  rw [Desc.throughArg.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  rcases ha : node.args[i]? with _ | a
  · rw [ha] at h; exact absurd h (by intro h; cases h)
  rw [ha] at h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hnode : c.body.nodes[k]! = node := node_get hn
  have hk : k < c.body.nodes.size := node_lt hn
  obtain ⟨hia, hai⟩ := Array.getElem?_eq_some_iff.1 ha
  have hai' : node.args[i]! = a := by rw [getElem!_pos node.args i hia]; exact hai
  have hUp := upOk_of hB hn
  have hAk : ArgOk c k i := hA k i hk (by rw [hnode]; exact hia)
  have hPk := hAP k i hk (by rw [hnode]; exact hia)
  have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
  have hone : ∀ lo hi v, (lo < hi → hi ≤ a.seq.count) → oneOf c fuel k i lo hi = .ok v →
      ∀ s ∈ v.toList, NodeW c.P c.d s := fun lo hi v hb hv =>
    one_n c fuel hD hAk hPk hRk (by rw [hnode, hai']; exact hb) hv
  rw [hnode] at hm hx
  have hsp := elems_pos g (fun x hx' => (hx x hx').1)
  have hj : ∀ x ∈ (g.rowsOf.flatMap fun g =>
      if (g.stride == 1) = true then #[(g.base.toNat, g.base.toNat + g.len)]
      else (Array.range g.len).map fun t => ((g.at t).toNat, (g.at t).toNat + 1)).toList, ∀ j, x.1 ≤ j → j < x.2 →
      i < (c.P.defs[node.fn]!).body.params.size ∧ j < ((c.P.defs[node.fn]!).body.params[i]!).2.leaves := by
    intro x hxm j h1 h2
    have : j ∈ spansPos _ := List.mem_flatMap.2 ⟨x, hxm, List.mem_range'_1.2 ⟨h1, by omega⟩⟩
    rw [hsp] at this
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 this
    have := hx y hy
    exact ⟨this.2.1, by omega⟩
  have hfa := forIn_app (NodeW c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨segs', rfl, hfin⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hfin s)
  intro x hxm r y hy
  obtain ⟨lo, hi⟩ := x
  have hjx := hj (lo, hi) hxm
  rcases hf : node.form with _ | _ | _ | _
  case prim | call =>
    simp only [hf] at hy
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
    obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
    have := hUp i hi0 (hi - 1) hjl m hm
    simp only [hf, hai'] at this
    omega
  case batch =>
    simp only [hf] at hy
    obtain ⟨w, hw, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    rw [← Array.forIn_toList] at hw
    have hLz : ∀ d L, a.dims = some (d, L) → L = 0 → hi ≤ lo := by
      intro d L hdl hL
      by_contra hlt
      have h1 := (hjx lo le_rfl (by omega)).2
      have h2 := hL0 k i d L (by rw [hnode, hai']; exact hdl) hL
        (by rw [hnode]; exact (hjx lo le_rfl (by omega)).1)
      rw [hnode] at h2
      omega
    have hms := memberSpans_idx a m lo hi hLz
    have hfa2 := forIn_app (NodeW c.P c.d) _ _ ?_ r w hw
    · obtain ⟨segs2, rfl, hag2⟩ := hfa2
      exact ⟨segs2, hy.symm, hag2⟩
    intro x hxm2 r2 y2 hy2
    obtain ⟨s, e⟩ := x
    obtain ⟨v, hv, hy2⟩ := Walk.except_bind_ok hy2
    simp only [pure, Except.pure, Except.ok.injEq] at hy2
    refine ⟨v, hy2.symm, hone s e v (fun hlt => ?_) hv⟩
    have hmem : e - 1 ∈ spansPos (Desc.memberSpans a m lo hi) :=
      List.mem_flatMap.2 ⟨(s, e), hxm2, List.mem_range'_1.2 ⟨by omega, by omega⟩⟩
    rw [hms] at hmem
    obtain ⟨j, hjm, hje⟩ := List.mem_map.1 hmem
    rw [List.mem_range'_1] at hjm
    obtain ⟨hi0, hjl⟩ := hjx j hjm.1 (by omega)
    have := hUp i hi0 j hjl m hm
    simp only [hf, hai', hje] at this
    omega
  case scan =>
    simp only [hf] at hy
    by_cases hi0 : i = 0
    · subst hi0
      rw [if_pos (show ((0 : Nat) == 0) = true from rfl)] at hy
      by_cases hm0 : m = 0
      · subst hm0
        rw [if_pos (show ((0 : Nat) == 0) = true from rfl)] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
        obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp 0 hi0 (hi - 1) hjl 0 hm
        simp only [hf, hai', ↓reduceIte] at this
        omega
      · rw [if_neg (show ¬ ((m == 0) = true) by simpa using hm0)] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        have hm1 : m - 1 < (c.body.nodes[k]!).members := by rw [hnode]; omega
        refine ⟨v, hy.symm, hK k (m - 1) lo hi v (by rw [hnode]; exact hf) hm1 (fun hlt => ?_) hv⟩
        obtain ⟨hi0, hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp 0 hi0 (hi - 1) hjl m hm
        simp only [hf, hm0, ↓reduceIte] at this
        rw [hnode]
        omega
    · rw [if_neg (show ¬ ((i == 0) = true) by simpa using hi0)] at hy
      by_cases hx0 : i ≤ node.nxs
      · rw [if_pos hx0] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        generalize hL : (Option.map (fun x => x.2) a.dims).getD 0 = L at hv
        refine ⟨v, hy.symm, hone (m * L + lo) (m * L + hi) v (fun hlt => ?_) hv⟩
        obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp i hi0' (hi - 1) hjl m hm
        simp only [hf, hi0, hx0, hai', hL, ↓reduceIte] at this
        omega
      · rw [if_neg hx0] at hy
        obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        refine ⟨v, hy.symm, hone lo hi v (fun hlt => ?_) hv⟩
        obtain ⟨hi0', hjl⟩ := hjx (hi - 1) (by omega) (by omega)
        have := hUp i hi0' (hi - 1) hjl m hm
        simp only [hf, hi0, hx0, hai', ↓reduceIte] at this
        omega

/-- What `down` gives a batch of a composite's piece of positive count, one step up, keeps the body's gates by node. -/
def BatchN (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs k node, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → q.count ≠ 0 → q.flat = q → (∀ x ∈ pos q, 0 ≤ x.2) →
    (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs → ∀ s ∈ segs.toList, NodeW c.P c.d s

/-- What `down` gives a scan's piece of one row, one step up, keeps the body's gates by node. -/
def ScanN (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q segs k node (carry y : Flock.Program.Ty), q.tgt = .ref false k → c.node k = .ok node → node.form = .scan →
    node.n ≠ 0 → q.count ≠ 0 → q.flat = q → ¬ q.rows > 1 → (c.P.defs[node.fn]!).ret = .tuple #[carry, y] →
    (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → Desc.down c (fuel + 1) q = .ok segs →
    ∀ s ∈ segs.toList, NodeW c.P c.d s

/-- A piece's rows, each through `down`, keep the body's gates by node. -/
theorem byRows_n (c : Desc.Ctx) (fuel : Nat) (hD : DownNAt c fuel) (q : Piece) (h0 : 0 < q.count)
    (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    (hp : ∀ x ∈ pos q, ∀ i, x.1 = .ref true i → i < c.body.params.size ∧ x.2 < ((c.body.params[i]!).2.leaves : Int))
    (hg : ∃ k', InNode c.P c.d k' (pos q))
    {segs : Array Seg} (h : byRows c fuel q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  unfold byRows at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  rw [← Array.forIn_toList] at h
  have hm : ∀ x ∈ q.rowsOf.toList, ∀ y ∈ pos x, y ∈ pos q := fun x hx y hy => by
    rw [← posAll_rowsOf q]
    exact List.mem_flatMap.2 ⟨x, hx, hy⟩
  have hlen : q.len ≠ 0 := fun hl => by
    unfold Desc.Piece.count at h0
    rw [hl, Nat.mul_zero] at h0
    exact absurd h0 (Nat.lt_irrefl 0)
  have hc : ∀ x ∈ q.rowsOf.toList, 0 < x.count := by
    intro x hx
    unfold Desc.Piece.rowsOf at hx
    simp only [Array.toList_map, List.mem_map] at hx
    obtain ⟨r, -, rfl⟩ := hx
    show 0 < 1 * q.len
    omega
  exact loop_mem _ (Desc.down c fuel) q.rowsOf.toList (fun x hx segs hs => hD x segs (hc x hx)
    (fun y hy => hq y (hm x hx y hy)) (fun y hy => hf y (hm x hx y hy)) (fun y hy => hp y (hm x hx y hy))
    (hg.imp fun _ h => inNode_mono h (hm x hx)) hs) segs h

/-- A Call's piece of one row and stride 1, of a composite: what `fromCallee` gives, by node. -/
theorem downN_callOne (c : Desc.Ctx) (fuel : Nat) (hC : CalleeN c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride = 1)
    (hpr : (c.P.defs[node.fn]!).prim = false)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  rw [down_call_one c fuel q ht hff hn hform h0 hr hs] at h
  exact hC k 0 q.base.toNat (q.base.toNat + q.len) segs
    (by rw [node_get hn, members_call hform]; omega) (by rw [node_get hn]; exact hpr) h

/-- A Call's piece of one row and another stride, of a composite: what `fromCallee` gives at each position, by node. -/
theorem downN_callPts (c : Desc.Ctx) (fuel : Nat) (hC : CalleeN c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (hff : q.flat = q) {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hform : node.form = .call) (h0 : q.count ≠ 0) (hr : ¬ q.rows > 1) (hs : q.stride ≠ 1)
    (hpr : (c.P.defs[node.fn]!).prim = false)
    {segs : Array Seg} (h : Desc.down c (fuel + 1) q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  rw [down_call_pts c fuel q ht hff hn hform h0 hr hs] at h
  unfold callPts at h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  exact loop_mem _ (fun t => Desc.fromCallee c fuel k 0 (q.at t).toNat ((q.at t).toNat + 1)) (List.range q.len)
    (fun _ _ segs hs => hC k 0 _ _ segs (by rw [node_get hn, members_call hform]; omega)
      (by rw [node_get hn]; exact hpr) hs) segs h

/-- **`down`'s node step**: one step up, `down`'s segments keep the body's gates by node wherever `down`'s and
`fromCallee`'s do and a batch of a composite's and a scan's piece of one row give such segments, in a laid-out body
whose primitives are one gate of one leaf and whose nodes call what the decoder lets them (`downShape_succ`'s
argument). -/
theorem downN_succ (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hL : Laid c.P c.d)
    (hG1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).gates = 1)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hCP : ∀ k node, c.node k = .ok node → (node.form = .prim → (c.P.defs[node.fn]!).prim = true) ∧
      (node.form = .call ∨ node.form = .scan → (c.P.defs[node.fn]!).prim = false))
    (fuel : Nat) (hD : DownNAt c fuel) (hC : CalleeN c fuel) (hBatch : BatchN c fuel) (hScan : ScanN c fuel) :
    DownNAt c (fuel + 1) := by
  intro q segs h0 hq hf hp hg h
  cases ht : q.tgt with
  | gate w => exact downN_gate c fuel q ht hg h
  | ref b k =>
  cases b with
  | true => exact downN_param c fuel q ht h
  | false =>
  rw [down_flat c fuel q ht] at h
  rw [← pos_flat q] at hq hf hp hg
  have ht' : q.flat.tgt = .ref false k := by rw [flat_tgt, ht]
  have h0' : q.flat.count ≠ 0 := by rw [flat_count]; omega
  have hff := flat_flat q
  generalize q.flat = p at ht' h0' hff hq hf hp hg h
  rcases hn : c.node k with e | node
  · exact absurd h (down_node c fuel p ht' h0' hn segs)
  have hdo := downOk_of hB hn
  cases hform : node.form with
  | prim =>
    have hpr := (hCP k node hn).1 hform
    cases hr : (c.P.defs[node.fn]!).ret with
    | value w => exact downN_prim c fuel p ht' hff hn hform hr (hG1 _ hpr) hL h
    | _ => exact absurd h (down_prim_ret c fuel p ht' hff hn (Or.inl hform) h0' (fun w e => by rw [hr] at e; cases e) segs)
  | call =>
    have hpr := (hCP k node hn).2 (Or.inl hform)
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inl hform) h0' hr] at h
      exact byRows_n c fuel hD p (Nat.pos_of_ne_zero h0') hq hf hp hg h
    by_cases hs : p.stride = 1
    · exact downN_callOne c fuel hC p ht' hff hn hform h0' hr hs hpr h
    · exact downN_callPts c fuel hC p ht' hff hn hform h0' hr hs hpr h
  | scan =>
    by_cases hn0 : node.n = 0
    · exact absurd h (down_scan0 c fuel p ht' hff hn hform hn0 h0' segs)
    by_cases hr : p.rows > 1
    · rw [down_rows c fuel p ht' hff hn (Or.inr ⟨hform, hn0⟩) h0' hr] at h
      exact byRows_n c fuel hD p (Nat.pos_of_ne_zero h0') hq hf hp hg h
    by_cases hex : ∃ carry y, (c.P.defs[node.fn]!).ret = .tuple #[carry, y]
    · obtain ⟨carry, y, hret⟩ := hex
      exact hScan p segs k node carry y ht' hn hform hn0 h0' hff hr hret hq hf h
    · exact absurd h
        (down_scan_ret c fuel p ht' hff hn hform hn0 h0' hr (fun carry y e => hex ⟨carry, y, e⟩) segs)
  | batch =>
    by_cases hpr : (c.P.defs[node.fn]!).prim = true
    · cases hr : (c.P.defs[node.fn]!).ret with
      | value w =>
        have hout : node.out.leaves ≤ node.members := by
          rw [members_batch hform]
          by_contra hlt
          have := hdo node.n (by omega)
          simp only [hform, hP1 _ hpr, Nat.div_one] at this
          exact absurd this.1 (Nat.lt_irrefl _)
        exact downN_batchPrim c fuel p ht' hff hn hform hpr hr (hG1 _ hpr) hL hq hf hout h
      | _ =>
        exact absurd h
          (down_prim_ret c fuel p ht' hff hn (Or.inr ⟨hform, hpr⟩) h0' (fun w e => by rw [hr] at e; cases e) segs)
    · exact hBatch p segs k node ht' hn hform (by simpa using hpr) h0' hff hq hf h

/-- A scan's initial carry's leaves `[lo, hi)`, each in range, through `throughArg`: segments keeping the body's gates
by node. -/
theorem init_n (c : Desc.Ctx) {fuel k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node)
    (hTW : ThroughN c fuel) (hm0 : 0 < node.members) {lo hi : Nat}
    (hlv : ∀ j, lo ≤ j → j < hi → 0 < (c.P.defs[node.fn]!).body.params.size ∧
      j < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves) {segs : Array Seg}
    (h : Desc.throughArg c fuel k 0 0 { tgt := .ref true 0, base := lo, stride := 1, len := hi - lo } = .ok segs) :
    ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  have hnode := node_get hn
  exact hTW k 0 0 _ segs (by rw [hnode]; exact hm0) rfl (fun x hx => by
    rw [pos_carry, List.mem_map] at hx
    obtain ⟨j, hj, rfl⟩ := hx
    rw [List.mem_range'_1] at hj
    rw [hnode]
    have := hlv j hj.1 (by omega)
    exact ⟨by simp, this.1, by show (j : Int) < _; exact_mod_cast this.2⟩) h

/-- **`carry`'s node step**: one step up, `carry`'s segments keep the body's gates by node wherever the step's summary
holds `RetOk`, the step is a composite, and `fromCallee`'s and `throughArg`'s segments one step down do
(`carryShape_succ`'s argument). -/
theorem carryN_succ (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (fuel : Nat)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hCP : ∀ k node, c.node k = .ok node → node.form = .scan → (c.P.defs[node.fn]!).prim = false)
    (hCW : CalleeN c fuel) (hTW : ThroughN c fuel) : CarryN c (fuel + 1) := by
  intro k m a b segs hform hm hb h
  rw [Desc.carry.eq_2] at h
  obtain ⟨node, hn, h⟩ := Walk.except_bind_ok h
  dsimp only at h
  have h' := ok_bind_pure h
  rw [← Array.forIn_toList] at h'
  have hnode : c.body.nodes[k]! = node := node_get hn
  rw [hnode] at hform hm hb
  have hsp := hCP k node hn hform
  have hRk := hR k node hn
  obtain ⟨hps, hpos, -, hag, hpp⟩ := hRk
  have hlenR : (psPos (c.memo[node.fn]!).ret).length = (c.P.defs[node.fn]!).body.ret.count := by
    have := (hag (c.P.child sc k 0) (by rw [child_eq c sc hd hn 0]; rfl)).length_eq
    simpa using this
  obtain ⟨-, hid⟩ := carryRuns_cover _ hps hpp a b (fun hab => by rw [hlenR]; exact hb hab)
  have hm0 : 0 < node.members := by omega
  have hcarry : ∀ h j' : Nat, (psPos (c.memo[node.fn]!).ret)[h]? = some (.ref true 0, (j' : Int)) →
      0 < (c.P.defs[node.fn]!).body.params.size ∧ j' < ((c.P.defs[node.fn]!).body.params[0]!).2.leaves := by
    intro h j' hx
    have := (hpos _ (List.mem_of_getElem? hx)).2.2 0 rfl
    exact ⟨this.1, by have h2 : ((j' : Nat) : Int) < _ := this.2; exact_mod_cast h2⟩
  have hfa := forIn_app (NodeW c.P c.d) _ _ ?_ #[] segs h'
  · obtain ⟨out, rfl, hout⟩ := hfa
    intro s hs
    rw [Array.toList_append, List.mem_append] at hs
    exact hs.elim (fun hs => by simp at hs) (hout s)
  intro x hx r' y hy
  split at hy
  · rename_i hxid
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    exact ⟨v, rfl, init_n c hn hTW hm0 (fun j h1 h2 => hcarry j j (hid x hx hxid j h1 h2)) hv⟩
  · have hy := walkAtMost_bind _ _ _ hy
    obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
    obtain ⟨segs', hsegs', hA2⟩ := forIn_app (NodeW c.P c.d) _ _ (fun j _ r'' y' hy' => by
        obtain ⟨ch, hch, hy'⟩ := Walk.except_bind_ok hy'
        have hok := chainOf_ok _ j ch hch
        split at hy'
        · obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          exact ⟨v', rfl, hCW k (m - (ch.hist.size - 1)) _ _ v' (by rw [hnode]; omega) (by rw [hnode]; exact hsp) hv'⟩
        · rename_i hcond
          obtain ⟨v', hv', hy'⟩ := Walk.except_bind_ok hy'
          simp only [pure, Except.pure, Except.ok.injEq] at hy'
          subst hy'
          refine ⟨v', rfl, ?_⟩
          have hs : ∀ s ≤ m, (psPos (c.memo[node.fn]!).ret)[ch.at s]? = some (.ref true 0, (ch.at (s + 1) : Int)) := by
            intro s hs
            rcases hex : ch.exit with _ | q0
            · obtain ⟨q, hq, ht, hb⟩ := chainAt_cyc hok hex s
              exact carryPos_of_retAt (hR k node hn) hq ht hb
            · have he : ¬ ch.hist.size - 1 ≤ m := fun he => hcond (by simp [hex, he])
              obtain ⟨-, -, hsc, -, -⟩ := hok
              obtain ⟨q, hq, ht, hb⟩ := hsc s (by omega)
              rw [chainAt_lt ch (show s < ch.hist.size by omega), chainAt_lt ch (show s + 1 < ch.hist.size by omega)]
              exact carryPos_of_retAt (hR k node hn) hq ht hb
          exact init_n c hn hTW hm0 (lo := ch.at (m + 1)) (hi := ch.at (m + 1) + 1)
            (fun j' h1 h2 => by rw [show j' = ch.at (m + 1) by omega]; exact hcarry _ _ (hs m le_rfl)) hv') r' v hv
    exact ⟨segs', by rw [hsegs'], hA2⟩

/-- **`down`, `fromCallee`, `throughArg` and `carry` keep the body's gates by node, at every fuel**, in a laid-out body
whose callees are laid out and name their gates with their widths (`RetW`), whose primitives are one gate of one leaf,
whose nodes call what the decoder lets them, wherever a batch of a composite's piece and a scan's piece of one row give
such segments one step up (`shapes`'s argument). -/
theorem oneNode (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hL : Laid c.P c.d)
    (hG1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).gates = 1)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hCP : ∀ k node, c.node k = .ok node → (node.form = .prim → (c.P.defs[node.fn]!).prim = true) ∧
      (node.form = .call ∨ node.form = .scan → (c.P.defs[node.fn]!).prim = false))
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates)
    (hBatch : ∀ fuel, DownNAt c fuel → CalleeN c fuel → BatchN c fuel)
    (hScan : ∀ fuel, CalleeN c fuel → CarryN c fuel → ScanN c fuel) :
    ∀ fuel, DownNAt c fuel ∧ CalleeN c fuel ∧ ThroughN c fuel ∧ CarryN c fuel
  | 0 => by
    refine ⟨fun q segs _ _ _ _ _ h => absurd h (down_zero c q segs), ?_, ?_, ?_⟩
    · intro _ _ _ _ _ _ _ h; rw [Desc.fromCallee.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.throughArg.eq_1] at h; cases h
    · intro _ _ _ _ _ _ _ _ h; rw [Desc.carry.eq_1] at h; cases h
  | fuel + 1 =>
    have ⟨hD, hC, hTW, hK⟩ := oneNode c sc hd hB hL hG1 hP1 hCP hA hAP hL0 hR hRW hLF hBatch hScan fuel
    ⟨downN_succ c hB hL hG1 hP1 hCP fuel hD hC (hBatch fuel hD hC) (hScan fuel hC hK),
      calleeN_succ c fuel hR hRW hL hLF hTW, throughN_succ c hB hA hAP hL0 fuel hD hK,
      carryN_succ c sc hd fuel hR (fun k node hn hf => (hCP k node hn).2 (Or.inr hf)) hC hTW⟩

/-- A scan's ys iterations `[lo / yl, (hi - 1) / yl]`, each through `fromCallee`, of a composite step: segments keeping
the body's gates by node. -/
theorem ys_n (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeN c fuel)
    {k cl yl lo hi : Nat} (hm : (hi - 1) / yl < (c.body.nodes[k]!).members)
    (hsp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false)
    {out0 segs : Array Seg}
    (h : forIn (List.range' (lo / yl) ((hi - 1) / yl + 1 - lo / yl)) out0 (fun m (s : Array Seg) => do
        let v ← Desc.fromCallee c fuel k m (cl + max lo (m * yl) - m * yl) (cl + min hi ((m + 1) * yl) - m * yl)
        pure (ForInStep.yield (s ++ v))) = .ok segs) :
    ∃ segs2, segs = out0 ++ segs2 ∧ ∀ s ∈ segs2.toList, NodeW c.P c.d s :=
  forIn_app (NodeW c.P c.d) _ _ (fun m hmm r y hy => by
      rw [List.mem_range'_1] at hmm
      obtain ⟨v, hv, hy⟩ := Walk.except_bind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨v, hy.symm, hCW k m _ _ v (by omega) hsp hv⟩) out0 segs h

/-- A scan's piece of stride 1: its ys past the carry's `cl` leaves, as `scanOne` reads them, of a composite step, give
segments keeping the body's gates by node. -/
theorem scanYs_n (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeN c fuel)
    {k cl yl a b : Nat} (hm : cl < b → (b - 1 - cl) / yl < (c.body.nodes[k]!).members)
    (hsp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false)
    {out0 segs : Array Seg}
    (h : (if (decide (cl < b) && decide (yl > 0)) = true then do
        Desc.walkAtMost "iterations of a scan whose ys are read one by one" ((b - cl - 1) / yl + 1 - (max a cl - cl) / yl)
        (forIn [(max a cl - cl) / yl:(b - cl - 1) / yl + 1] out0 fun m s => do
          let v ← Desc.fromCallee c fuel k m (cl + max (max a cl - cl) (m * yl) - m * yl)
            (cl + min (b - cl) ((m + 1) * yl) - m * yl)
          pure (ForInStep.yield (s ++ v))) >>= fun s => pure s
      else pure out0) = .ok segs) :
    ∃ s2, segs = out0 ++ s2 ∧ ∀ s ∈ s2.toList, NodeW c.P c.d s := by
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel,
      Nat.div_one] at hv
    exact ys_n c fuel hCW (k := k) (cl := cl) (yl := yl) (lo := max a cl - cl) (hi := b - cl)
      (by rw [show b - cl - 1 = b - 1 - cl by omega]; exact hm hc.1) hsp hv
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨#[], by simp, fun s hs => by simp at hs⟩

/-- **A scan's piece of one row**, one step up: what `carry` and `fromCallee` give of its final carry's leaves and its
ys keeps the body's gates by node wherever their segments one step down do (`scanShape_of`'s argument). -/
theorem scanN_of (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat)
    (hCW : CalleeN c fuel) (hK : CarryN c fuel) : ScanN c fuel := by
  intro q segs k node carry y ht hn hform hn0 h0 hff hr hret hq hf h
  have hdo := downOk_of hB hn
  have h1 := rows_one h0 hr
  have hnode : c.body.nodes[k]! = node := node_get hn
  let M : Nat → Nat × Nat := fun j => if j < carry.leaves then (node.n - 1, j)
    else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)
  have hbd : ∀ j, j < node.out.leaves →
      (M j).1 < node.n ∧ (M j).2 < (c.P.defs[node.fn]!).body.ret.count := by
    intro j hj
    have hdk := hdo j hj
    simp only [hform] at hdk
    obtain ⟨carry', y', hr', hmn, hj'⟩ := hdk.2 hn0
    rw [hret] at hr'
    injection hr' with e
    obtain ⟨rfl, rfl⟩ : carry = carry' ∧ y = y' := by simpa using e
    by_cases hlt : j < carry.leaves
    · simp only [M, hlt, ↓reduceIte]; simp only [hlt, ↓reduceIte] at hmn hj'; exact ⟨hmn, hj'⟩
    · simp only [M, hlt, ↓reduceIte]; simp only [hlt, ↓reduceIte] at hmn hj'; exact ⟨hmn, hj'⟩
  have hpt : ∀ t < q.len, 0 ≤ q.at t ∧ (q.at t).toNat < node.out.leaves := by
    intro t ht'
    have hm := at_mem_pos h1 ht'
    have hb := hq _ hm
    have hfl := hf _ hm k ht
    simp only [hnode] at hb hfl
    exact ⟨hb, by omega⟩
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hsp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false := by
    have := hdo _ (hpt 0 hlen).2
    simp only [hform] at this
    rw [hnode]
    exact this.1
  rw [down_scan_head c fuel q ht hff hn hform hn0 h0 hr hret] at h
  split at h
  · rename_i hs
    unfold scanOne at h
    dsimp only at h
    have hat : ∀ t, (q.at t).toNat = q.base.toNat + t := fun t => by
      have := (hpt 0 hlen).1
      simp only [Desc.Piece.at, hs] at this ⊢
      omega
    have hbl : q.base.toNat + q.len ≤ node.out.leaves := by
      have := (hpt (q.len - 1) (by omega)).2
      rw [hat] at this
      omega
    have hmB : carry.leaves < q.base.toNat + q.len →
        (q.base.toNat + q.len - 1 - carry.leaves) / y.leaves < (c.body.nodes[k]!).members := by
      intro hcb
      have := (hbd (q.base.toNat + q.len - 1) (by omega)).1
      simp only [M, show ¬ (q.base.toNat + q.len - 1 < carry.leaves) by omega, ↓reduceIte] at this
      rw [hnode, members_scan hform]
      exact this
    split at h
    · obtain ⟨v1, hv1, h⟩ := Walk.except_bind_ok h
      obtain ⟨s2, rfl, hA2⟩ := scanYs_n c fuel hCW hmB hsp h
      have hA1 := hK k (node.n - 1) _ _ v1 (by rw [hnode]; exact hform)
        (by rw [hnode, members_scan hform]; omega) (fun hlt => by
          have := (hbd (min (q.base.toNat + q.len) carry.leaves - 1) (by omega)).2
          simp only [M, show min (q.base.toNat + q.len) carry.leaves - 1 < carry.leaves by omega,
            ↓reduceIte] at this
          rw [hnode]
          omega) hv1
      intro s hs
      rw [Array.empty_append, Array.toList_append, List.mem_append] at hs
      exact hs.elim (hA1 s) (hA2 s)
    · obtain ⟨s2, rfl, hA2⟩ := scanYs_n c fuel hCW hmB hsp h
      rw [Array.empty_append]
      exact hA2
  · unfold scanPts at h
    dsimp only at h
    have h := walkAtMost_bind _ _ _ h
    obtain ⟨v, hv, h⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at hv
    rw [range'_size] at hv
    obtain ⟨segs', hsegs, hA⟩ := forIn_app (NodeW c.P c.d) _ (List.range q.len)
      (fun t htm r y' hy => by
        obtain ⟨-, hlt⟩ := hpt t (List.mem_range.1 htm)
        obtain ⟨hmn, hjr⟩ := hbd _ hlt
        split at hy
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, hc, ↓reduceIte] at hmn hjr
          have := hK k (node.n - 1) (q.at t).toNat ((q.at t).toNat + 1) v' (by rw [hnode]; exact hform)
            (by rw [hnode, members_scan hform]; exact hmn) (fun _ => by rw [hnode]; omega) hv'
          simpa using this
        · rename_i hc
          obtain ⟨v', hv', hy⟩ := Walk.except_bind_ok hy
          simp only [pure, Except.pure, Except.ok.injEq] at hy
          refine ⟨v', hy.symm, ?_⟩
          simp only [M, hc, ↓reduceIte] at hmn
          have := hCW k _ _ _ v' (by rw [hnode, members_scan hform]; exact hmn) hsp hv'
          simpa using this) #[] v hv
    rw [hsegs, Array.empty_append]
    exact hA

/-- **A batch of a composite's piece in its one-run case** keeps the body's gates by node: leaf `e` of the
node is member `e / L`'s return leaf `e % L`, a gate of the run its callee names with its width (`RetW`), in laid-out
bodies. `batchN_of`'s `hAff`. -/
theorem batchAffine_n (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hL : Laid c.P c.d)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    ∀ q k node p, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → affineOf c k node q = some p →
      NodeW c.P c.d (Seg.one p) := by
  intro q k node p ht hn hform hp hL0 h0 hq hf ha
  obtain ⟨hok, hpos, -, -, hpp⟩ := hR k node hn
  have hRWk := hRW k node hn
  obtain ⟨hLc, hlast⟩ := hLF k node hn hp
  have hdo := downOk_of hB hn
  have hk := node_lt hn
  have hnode := node_get hn
  unfold affineOf at ha
  dsimp only at ha
  split at ha
  swap
  · cases ha
  rename_i hc
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
  obtain ⟨⟨hsz, hreps⟩, hps⟩ := hc
  obtain ⟨s0, hs0⟩ := Array.size_eq_one_iff.1 hsz
  obtain ⟨g0, hg0⟩ := Array.size_eq_one_iff.1 (by rw [hs0] at hps; exact hps)
  rw [hs0] at ha hreps
  simp only [hg0] at ha
  have e1 : (#[s0] : Array Seg)[0]! = s0 := rfl
  have e2 : (#[g0] : Array Piece)[0]! = g0 := rfl
  rw [e1] at hreps
  simp only [e2] at ha
  rcases hgt : g0.tgt with _ | w
  · rw [hgt] at ha; cases ha
  rw [hgt] at ha
  dsimp only at ha
  split at ha
  swap
  · cases ha
  rename_i hc
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at hc
  obtain ⟨⟨hrows, hlen⟩, hFG⟩ := hc
  simp only [Option.some.injEq] at ha
  subst ha
  rw [e1] at hg0
  have hflat : psPos (c.memo[node.fn]!).ret = pos g0 := by
    rw [psPos_flat _ hok (fun x hx => by
      rw [hs0] at hx
      simp at hx
      subst hx; exact hreps), hs0]
    simp [hg0, posAll_singleton]
  have hg0c : 0 < g0.count := hpp s0 (by rw [hs0]; simp) g0 (by rw [hg0]; simp)
  have hr1 : g0.rows = 1 := by
    unfold Desc.Piece.count at hg0c
    rcases Nat.lt_or_ge g0.rows 1 with h | h
    · rw [show g0.rows = 0 by omega, Nat.zero_mul] at hg0c; omega
    · omega
  refine nodeW_one (k := k) ?_
  intro w' x hx
  rw [pos_eq_map, List.mem_map] at hx
  obtain ⟨t, ht', he⟩ := hx
  simp only [Prod.mk.injEq] at he
  obtain ⟨-, rfl⟩ := he
  rw [ppt_affine]
  have ht'' : t < q.count := List.mem_range.1 ht'
  have hmem : (q.tgt, ppt q t) ∈ pos q := by
    rw [pos_eq_map]; exact List.mem_map.2 ⟨t, List.mem_range.2 ht'', rfl⟩
  have he0 : 0 ≤ ppt q t := hq _ hmem
  have hfe : ppt q t < ((c.body.nodes[k]!).out.leaves : Int) := hf _ hmem k ht
  rw [hnode] at hfe
  obtain ⟨e, he⟩ := Int.eq_ofNat_of_zero_le he0
  rw [he] at hfe ⊢
  have hel : e < node.out.leaves := by exact_mod_cast hfe
  have hdk := hdo e hel
  simp only [hform] at hdk
  have hmn := hdk.1
  have hLpos : 0 < (c.P.defs[node.fn]!).outLeaves := Nat.pos_of_ne_zero hL0
  have hjL : e % (c.P.defs[node.fn]!).outLeaves < g0.len := by rw [hlen]; exact Nat.mod_lt _ hLpos
  have hgm := at_mem_pos hr1 hjL
  rw [hgt] at hgm
  have hRj := hRWk w _ (by rw [hflat]; exact hgm)
  have hj0 : 0 ≤ g0.at (e % (c.P.defs[node.fn]!).outLeaves) := (hpos _ (by rw [hflat]; exact hgm)).1
  obtain ⟨n', hn'⟩ := Int.eq_ofNat_of_zero_le hj0
  rw [hn', Int.toNat_natCast] at hRj
  have hnl := locW_lt hRj hLc hlast
  subst hnode
  have hm' : e / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves < (c.body.nodes[k]!).members := by
    rw [members_batch hform]; exact hmn
  have hpe := affine_pos (e := e) (B := (c.body.offsets[k]! : Int) + g0.base) (s := g0.stride)
    (G := ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int)) hFG
  have hat : g0.at (e % (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves) =
      g0.base + ((e % (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) * g0.stride := rfl
  have hfin : (c.body.offsets[k]! : Int) + g0.base + (e : Int) *
      (if ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves == 1) = true then
        ((c.P.defs[(c.body.nodes[k]!).fn]!).gates : Int) else g0.stride) =
      ((c.body.offsets[k]! + e / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves *
        (c.P.defs[(c.body.nodes[k]!).fn]!).gates + n' : Nat) : Int) := by
    rw [hpe, Int.add_assoc, ← hat, hn']
    push_cast
    ring
  rw [hfin]
  exact inNode_child c.P hL hk hm' hnl

/-- A batch's piece of one row and any other stride, position by position through `down`: segments keeping the body's
gates by node. -/
theorem batchPts_n (c : Desc.Ctx) (fuel : Nat) (hD : DownNAt c fuel) (q : Piece) {k : Nat}
    (ht : q.tgt = .ref false k) (h1 : q.rows = 1) (hq : ∀ x ∈ pos q, 0 ≤ x.2) (hf : ∀ x ∈ pos q, Fits c x)
    {segs : Array Seg} (h : batchPts c fuel q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  unfold batchPts at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  rw [range'_size] at h
  have hpt : ∀ t, pos { q with base := q.at t, stride := 1, len := 1 } = [(q.tgt, q.at t)] := by
    intro t
    simp [pos, h1]
  exact loop_mem _ (fun t => Desc.down c fuel { q with base := q.at t, stride := 1, len := 1 }) (List.range q.len)
    (fun t htm segs hs => by
      have hm := at_mem_pos h1 (List.mem_range.1 htm)
      exact hD _ segs (by simp [Desc.Piece.count, h1])
        (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hq _ hm)
        (by rw [hpt t]; intro x hx; simp only [List.mem_singleton] at hx; subst hx; exact hf _ hm)
        (by rw [hpt t]; intro x hx i hi; simp only [List.mem_singleton] at hx; subst hx; rw [ht] at hi; cases hi)
        ⟨0, inNode_ref (q := { q with base := q.at t, stride := 1, len := 1 }) ht⟩
        hs) segs h

/-- What `batchOne` gives a batch of a composite's piece of one row and stride 1, past its one-run case, keeps the
body's gates by node. -/
def BatchOneN (c : Desc.Ctx) (fuel : Nat) : Prop :=
  ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
    (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 → q.rows = 1 →
    q.stride = 1 → (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchOne c fuel k node q = .ok segs →
    ∀ s ∈ segs.toList, NodeW c.P c.d s

/-- **`BatchN` from its cases**: a batch of a composite's piece row by row and position by position through `down`
(`byRows_n`, `batchPts_n`), and in its one-run, stride-1 and whole-member-stride cases wherever those keep the body's
gates by node (`batchShape_of`'s argument). -/
theorem batchN_of (c : Desc.Ctx) (fuel : Nat) (hD : DownNAt c fuel)
    (hAff : ∀ q k node p, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.count ≠ 0 →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → affineOf c k node q = some p → NodeW c.P c.d (Seg.one p))
    (hOne : BatchOneN c fuel)
    (hStride : ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.rows = 1 →
      (q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0) →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchStride c fuel k node q = .ok segs →
      ∀ s ∈ segs.toList, NodeW c.P c.d s) :
    BatchN c fuel := by
  intro q segs k node ht hn hform hp h0 hff hq hf h
  by_cases hL : (c.P.defs[node.fn]!).outLeaves = 0
  · exact absurd h (down_batch0 c fuel q ht hff hn hform hp hL h0 segs)
  rcases ha : affineOf c k node q with _ | p
  · by_cases hr : q.rows > 1
    · rw [down_batch_rows c fuel q ht hff hn hform hp hL h0 ha hr] at h
      exact byRows_n c fuel hD q (Nat.pos_of_ne_zero h0) hq hf
        (fun x hx i hi => by rw [mem_pos_tgt hx, ht] at hi; cases hi)
        ⟨0, inNode_ref ht⟩ h
    by_cases hs : q.stride = 1
    · rw [down_batch_one c fuel q ht hff hn hform hp hL h0 ha hr hs] at h
      exact hOne q k node segs ht hn hform hp hL h0 (rows_one h0 hr) hs hq hf h
    by_cases hcp : q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0
    · rw [down_batch_stride c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
      exact hStride q k node segs ht hn hform hp hL (rows_one h0 hr) hcp hq hf h
    rw [down_batch_pts c fuel q ht hff hn hform hp hL h0 ha hr hs hcp] at h
    exact batchPts_n c fuel hD q ht (rows_one h0 hr) hq hf h
  · rw [down_batch_affine c fuel q ht hff hn hform hp hL h0 ha] at h
    simp only [Except.ok.injEq] at h
    subst h
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact hAff q k node p ht hn hform hp hL h0 hq hf ha

/-- **A batch of a composite's piece of one row and a stride of whole members**, as `batchStride` reads it, keeps the
body's gates by node: copies of a member's leaf of gates, a copy a member (`nodeW_copies`); copies of what `fromCallee`
gives of a leaf of arguments the members share (`nodeW_alike`); or position by position (`batchPts_n`). `batchN_of`'s
`hStride`. -/
theorem batchStride_n (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hL : Laid c.P c.d) (fuel : Nat)
    (hD : DownNAt c fuel) (hCW : CalleeN c fuel)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    ∀ q k node segs, q.tgt = .ref false k → c.node k = .ok node → node.form = .batch →
      (c.P.defs[node.fn]!).prim = false → (c.P.defs[node.fn]!).outLeaves ≠ 0 → q.rows = 1 →
      (q.len > 1 ∧ q.stride % ((c.P.defs[node.fn]!).outLeaves : Int) = 0 ∧ q.base ≥ 0) →
      (∀ x ∈ pos q, 0 ≤ x.2) → (∀ x ∈ pos q, Fits c x) → batchStride c fuel k node q = .ok segs →
      ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  intro q k node segs ht hn hform hp hL0 h1 hcp hq hf h
  obtain ⟨hok, hnn, hgo, -, hpp⟩ := hR k node hn
  have hRWk := hRW k node hn
  obtain ⟨hLc, hlast⟩ := hLF k node hn hp
  have hdo := downOk_of hB hn
  have hk := node_lt hn
  have hnode := node_get hn
  have hl0 : 0 < q.len := by omega
  have hfu : q.at 0 < ((c.body.nodes[k]!).out.leaves : Int) := hf _ (at_mem_pos h1 hl0) k ht
  rw [hnode] at hfu
  have hbl : q.base.toNat < node.out.leaves := by
    have : q.at 0 = q.base := by simp [Desc.Piece.at]
    rw [this] at hfu
    omega
  have hdo0 := hdo q.base.toNat hbl
  simp only [hform] at hdo0
  have hmn := hdo0.1
  unfold batchStride at h
  dsimp only at h
  split at h
  · rename_i hrep
    split at h
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      have hleaf1 : ∀ x ∈ ((c.memo[node.fn]!).ret.slice (q.base.toNat % (c.P.defs[node.fn]!).outLeaves)
          (q.base.toNat % (c.P.defs[node.fn]!).outLeaves + 1)).toList, x.reps = 1 := fun x hx => by
        have h1' := (retOk_slice hgo hx).1
        have h2' := (Array.all_eq_true_iff_forall_mem.1 hrep) x (Array.mem_toList_iff.1 hx)
        simp only [decide_eq_true_eq] at h2'
        omega
      have hpsPos := posAll_flatMap_ps _ hleaf1
      rw [psSlice_pos _ hok] at hpsPos
      have hLp : 0 < (c.P.defs[node.fn]!).outLeaves := Nat.pos_of_ne_zero hL0
      have hLI : ((c.P.defs[node.fn]!).outLeaves : Int) ≠ 0 := by exact_mod_cast hL0
      obtain ⟨dd, hdd⟩ : ∃ dd, q.stride = ((c.P.defs[node.fn]!).outLeaves : Int) * dd :=
        ⟨q.stride / (c.P.defs[node.fn]!).outLeaves, (Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero hcp.2.1)).symm⟩
      have hdiv : q.stride / ((c.P.defs[node.fn]!).outLeaves : Int) = dd := by
        rw [hdd]; exact Int.mul_ediv_cancel_left _ hLI
      have hB' : (q.base.toNat : Int) = q.base := Int.toNat_of_nonneg hcp.2.2
      have h2 : (q.base : Int) = ((q.base.toNat / (c.P.defs[node.fn]!).outLeaves : Nat) : Int) *
          ((c.P.defs[node.fn]!).outLeaves : Int) + ((q.base.toNat % (c.P.defs[node.fn]!).outLeaves : Nat) : Int) := by
        rw [← hB']
        exact_mod_cast (Nat.div_add_mod' q.base.toNat (c.P.defs[node.fn]!).outLeaves).symm
      have hjL : ((q.base.toNat % (c.P.defs[node.fn]!).outLeaves : Nat) : Int) <
          ((c.P.defs[node.fn]!).outLeaves : Int) := by exact_mod_cast Nat.mod_lt _ hLp
      subst hnode
      refine nodeW_copies c.P hL hk hLc hlast (m0 := q.base.toNat / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves)
        (dm := q.stride / ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int)) (by have := hcp.1; omega) rfl rfl ?_ ?_
      · intro w v hx
        rw [hpsPos] at hx
        have hx' := List.mem_of_mem_drop (List.mem_of_mem_take hx)
        exact ⟨(hnn _ hx').1, hRWk w v hx'⟩
      · intro u hu
        rw [hdiv]
        have hat : q.at u = (((q.base.toNat / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) +
            (u : Int) * dd) * ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int) +
            ((q.base.toNat % (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) := by
          unfold Desc.Piece.at
          rw [hdd]
          linear_combination h2
        have hEq : q.at u / ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int) =
            ((q.base.toNat / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) + (u : Int) * dd := by
          rw [hat, Int.add_comm, Int.add_mul_ediv_right _ _ hLI,
            Int.ediv_eq_zero_of_lt (Int.natCast_nonneg _) hjL, Int.zero_add]
        have hm := at_mem_pos h1 hu
        have hE0 : 0 ≤ q.at u := hq _ hm
        have hEl : q.at u < ((c.body.nodes[k]!).out.leaves : Int) := hf _ hm k ht
        obtain ⟨E, hE⟩ := Int.eq_ofNat_of_zero_le hE0
        rw [hE] at hEq hEl
        have hEl' : E < (c.body.nodes[k]!).out.leaves := by exact_mod_cast hEl
        have hdE := (hdo E hEl')
        simp only [hform] at hdE
        have hcast : ((E / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Nat) : Int) =
            (E : Int) / ((c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves : Int) := by norm_cast
        rw [← hEq, ← hcast]
        refine ⟨Int.natCast_nonneg _, ?_⟩
        have hmb : ((c.P.defs[c.d]!).body.nodes[k]!).members = (c.body.nodes[k]!).n := members_batch hform
        rw [hmb]
        exact_mod_cast hdE.1
    · split at h
      · obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
        split at h
        · simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          intro s hs
          simp only [List.mem_singleton] at hs
          subst hs
          have hmem : q.base.toNat / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members := by
            rw [hnode, members_batch hform]; exact hmn
          exact nodeW_alike c.P c.d (hCW k _ _ _ one hmem (by rw [hnode]; exact hp) hone) 1 q.len
        · exact batchPts_n c fuel hD q ht h1 hq hf h
      · exact batchPts_n c fuel hD q ht h1 hq hf h
  · exact batchPts_n c fuel hD q ht h1 hq hf h

/-- A batch's output leaves `[a, b)` from `mid` and what `fromCallee` keeps of the cut members at either end, of a
composite: keeping the body's gates by node wherever `mid`'s segments do (`ends_shape`'s argument). -/
theorem ends_n (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeN c fuel) {k L a b : Nat} (hL : L ≠ 0)
    (hab : (a + L - 1) / L ≤ b / L) (hp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false)
    (hm : ∀ j, a ≤ j → j < b → j / L < (c.body.nodes[k]!).members)
    {mid segs : Array Seg} (hmid : ∀ s ∈ mid.toList, NodeW c.P c.d s)
    (h : batchEnds c fuel k L a b mid = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  unfold batchEnds at h
  dsimp only at h
  generalize hm0 : (a + L - 1) / L = m0 at *
  generalize hm1 : b / L = m1 at *
  have hLp : 0 < L := Nat.pos_of_ne_zero hL
  have ha0 : a ≤ m0 * L := by
    rw [← hm0]
    have := Nat.div_add_mod (a + L - 1) L
    have := Nat.mod_lt (a + L - 1) hLp
    have : L * ((a + L - 1) / L) = (a + L - 1) / L * L := Nat.mul_comm _ _
    omega
  have hb0 : m1 * L ≤ b := by rw [← hm1]; exact Nat.div_mul_le_self _ _
  have hmm : m0 * L ≤ m1 * L := Nat.mul_le_mul_right _ hab
  have hfirst : ∀ x, a < m0 * L → Desc.fromCallee c fuel k (a / L) (a % L) L = .ok x →
      ∀ s ∈ x.toList, NodeW c.P c.d s := fun x hlt hx =>
    hCW k (a / L) (a % L) L x (hm a le_rfl (by omega)) hp hx
  have hlast : ∀ y, m1 * L < b → Desc.fromCallee c fuel k m1 0 (b - m1 * L) = .ok y →
      ∀ s ∈ y.toList, NodeW c.P c.d s := by
    intro y hlt hy
    have hmB := hm (m1 * L) (by omega) hlt
    rw [Nat.mul_div_cancel _ hLp] at hmB
    exact hCW k m1 0 (b - m1 * L) y hmB hp hy
  obtain ⟨A, B, rfl, hA, hB⟩ : ∃ A B, segs = A ++ mid ++ B ∧ (∀ s ∈ A.toList, NodeW c.P c.d s) ∧
      ∀ s ∈ B.toList, NodeW c.P c.d s := by
    split at h
    · rename_i hlt
      obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
      split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨x, y, by rw [← h]; try simp, hfirst x hlt hx, hlast y hlt' hy⟩
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨x, #[], by rw [← h]; try simp, hfirst x hlt hx, by simp⟩
    · split at h
      · rename_i hlt'
        obtain ⟨y, hy, h⟩ := Walk.except_bind_ok h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨#[], y, by rw [← h]; try simp, by simp, hlast y hlt' hy⟩
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        exact ⟨#[], #[], by rw [← h]; try simp, by simp, by simp⟩
  intro s hs
  simp only [Array.toList_append, List.mem_append] at hs
  rcases hs with (hs | hs) | hs
  · exact hA s hs
  · exact hmid s hs
  · exact hB s hs

/-- A batch's output leaves `[a, b)` member by member, of a composite: keeping the body's gates by node. -/
theorem batchMembers_n (c : Desc.Ctx) (fuel : Nat) (hCW : CalleeN c fuel) {k L a b : Nat}
    (hp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false) (hm : (b - 1) / L < (c.body.nodes[k]!).members)
    {segs : Array Seg} (h : batchMembers c fuel k L a b = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  unfold batchMembers at h
  have h := walkAtMost_bind _ _ _ h
  dsimp only at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one] at h
  exact loop_mem (NodeW c.P c.d)
    (fun m => Desc.fromCallee c fuel k m (max a (m * L) - m * L) (min b ((m + 1) * L) - m * L)) _
    (fun m hmm segs hs => by
      rw [List.mem_range'_1] at hmm
      exact hCW k m _ _ segs (by omega) hp hs) segs h

/-- `down` of the piece `throughPS` gives of an argument whose pieces are of positive count: segments keeping the body's
gates by node, its positions being the argument's references. -/
theorem throughPS_n (c : Desc.Ctx) (fuel : Nat) (hD : DownNAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {p bp : Piece} (h : Desc.throughPS (c.arg k i) p = some bp) {segs : Array Seg}
    (hs : Desc.down c fuel bp = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  obtain ⟨hcnt, hmem⟩ := throughPS_mem h hA.1 hP
  have hmem' : ∀ x ∈ pos bp, ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro x hx
    have := hmem x hx
    rw [hA.2] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, List.mem_range.1 ht, rfl⟩
  exact hD bp segs hcnt
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem' x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem' x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem' x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩)
    ⟨0, fun w v hx => by
      obtain ⟨t, -, he⟩ := hmem' _ hx
      simp [refPos] at he⟩ hs

/-- A batch's output leaves `[a, b)` through `throughPS`, else member by member, of a composite: keeping the body's
gates by node. -/
theorem batchThrough_n (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (fuel : Nat) (hD : DownNAt c fuel)
    (hCW : CalleeN c fuel)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    {k : Nat} {node : Flock.Program.Node} (hn : c.node k = .ok node) (hL : (c.P.defs[node.fn]!).outLeaves ≠ 0)
    (hp : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false) {a b : Nat}
    (hm : ∀ j, a ≤ j → j < b → j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members)
    (hmB : (b - 1) / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members)
    {segs : Array Seg} (h : batchThrough c fuel k node a b = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  have hmem := fun (hh : batchMembers c fuel k (c.P.defs[node.fn]!).outLeaves a b = .ok segs) =>
    batchMembers_n c fuel hCW hp hmB hh
  unfold batchThrough at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    split at h
    · rename_i i hti
      split at h
      · rename_i d Li hdl
        split at h
        · split at h
          · rename_i bp hbp
            obtain ⟨x, hx, h⟩ := Walk.except_bind_ok h
            obtain ⟨a', ha, -⟩ := Option.bind_eq_some_iff.1 hdl
            obtain ⟨hia, -⟩ := Array.getElem?_eq_some_iff.1 ha
            have hnode := node_get hn
            have hk : k < c.body.nodes.size := node_lt hn
            have hia' : i < (c.body.nodes[k]!).args.size := by rw [hnode]; exact hia
            have hRk : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
                Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t) := (hB.1 k hk).2.1 i
            exact ends_n c fuel hCW hL (by omega) hp hm
              (throughPS_n c fuel hD (hA k i hk hia') (hAP k i hk hia') hRk hbp hx) h
          · exact hmem h
        · exact hmem h
      · exact hmem h
    · exact hmem h
  · exact hmem h

/-- **A batch of a composite's piece of one row and stride 1 past its one-run case**, as `batchOne` reads it, keeps the
body's gates by node: whole members as copies of a return of gates, a copy a member (`nodeW_copies`), or of what
`fromCallee` keeps of a member's return of shared leaves (`nodeW_alike`), through `throughPS`, or member by member, the
cut members at either end each through its return (`ends_n`, `batchThrough_n`). `batchN_of`'s `hOne`. -/
theorem batchOne_n (c : Desc.Ctx) (hB : Resolve.BodyOk c.P c.d) (hLd : Laid c.P c.d) (fuel : Nat)
    (hD : DownNAt c fuel) (hCW : CalleeN c fuel)
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    BatchOneN c fuel := by
  intro q k node segs ht hn hform hp hL h0 h1 hs hq hf h
  have hlen : 0 < q.len := by rw [← count_one h1]; omega
  have hdo := downOk_of hB hn
  have hnode := node_get hn
  have hp' : (c.P.defs[(c.body.nodes[k]!).fn]!).prim = false := by rw [hnode]; exact hp
  have hbl : q.base.toNat + q.len ≤ node.out.leaves := by
    have hm := at_mem_pos h1 (show q.len - 1 < q.len by omega)
    have hb := hq _ (at_mem_pos h1 hlen)
    have := hf _ hm k ht
    simp only [hnode, Desc.Piece.at, hs] at this hb
    omega
  have hm : ∀ j, q.base.toNat ≤ j → j < q.base.toNat + q.len →
      j / (c.P.defs[node.fn]!).outLeaves < (c.body.nodes[k]!).members := by
    intro j _ hj
    have := hdo j (by omega)
    simp only [hform] at this
    rw [hnode, members_batch hform]
    exact this.1
  have hmB := hm (q.base.toNat + q.len - 1) (by omega) (by omega)
  have hm1 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves ≤ (c.body.nodes[k]!).members := by
    have hLp := Nat.pos_of_ne_zero hL
    have e1 := Nat.div_mul_le_self (q.base.toNat + q.len) (c.P.defs[node.fn]!).outLeaves
    have e5 := Nat.lt_div_mul_add (a := q.base.toNat + q.len - 1) hLp
    have e6 : (q.base.toNat + q.len) / (c.P.defs[node.fn]!).outLeaves * (c.P.defs[node.fn]!).outLeaves ≤
        ((q.base.toNat + q.len - 1) / (c.P.defs[node.fn]!).outLeaves + 1) * (c.P.defs[node.fn]!).outLeaves := by
      rw [Nat.add_mul, Nat.one_mul]; omega
    have := Nat.le_of_mul_le_mul_right e6 hLp
    omega
  unfold batchOne at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    have hgo : ∀ x ∈ (c.memo[node.fn]!).ret.segs.toList, x.reps ≤ 1 := by
      intro x hx
      have := (Array.all_eq_true_iff_forall_mem.1 hc.1) x (Array.mem_toList_iff.1 hx)
      simp only [Bool.and_eq_true, decide_eq_true_eq] at this
      exact this.1
    refine ends_n c fuel hCW hL (by omega) hp' hm ?_ h
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    obtain ⟨hok, hnn, -, -, -⟩ := hR k node hn
    have hRWk := hRW k node hn
    obtain ⟨hLc, hlast⟩ := hLF k node hn hp
    have hflat := psPos_flat _ hok hgo
    have hk := node_lt hn
    subst hnode
    have hm1' : (q.base.toNat + q.len) / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves ≤
        ((c.P.defs[c.d]!).body.nodes[k]!).members := hm1
    refine nodeW_copies c.P hLd hk hLc hlast
      (m0 := (q.base.toNat + (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves - 1) /
        (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves) (dm := 1) (by have := hc.2; omega) rfl (Int.one_mul _).symm ?_ ?_
    · intro w v hx
      rw [← hflat] at hx
      exact ⟨(hnn _ hx).1, hRWk w v hx⟩
    · intro u hu
      generalize (q.base.toNat + (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves - 1) /
        (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves = m0 at hu ⊢
      generalize (q.base.toNat + q.len) / (c.P.defs[(c.body.nodes[k]!).fn]!).outLeaves = m1 at hu hm1'
      constructor <;> omega
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨one, hone, h⟩ := Walk.except_bind_ok h
    split at h
    · refine ends_n c fuel hCW hL (by omega) hp' hm ?_ h
      intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      exact nodeW_alike c.P c.d (hCW k _ _ _ one (by omega) hp' hone) _ _
    · exact batchThrough_n c hB fuel hD hCW hA hAP hn hL hp' hm hmB h
  · exact batchThrough_n c hB fuel hD hCW hA hAP hn hL hp' hm hmB h

/-- **`down`, `fromCallee`, `throughArg` and `carry` keep the body's gates by node, at every fuel**: `oneNode` with a
batch's cases (`batchN_of`) and a scan's (`scanN_of`). -/
theorem oneNode_of (c : Desc.Ctx) (sc : Flock.Program.Scope) (hd : sc.d = c.d) (hB : Resolve.BodyOk c.P c.d)
    (hL : Laid c.P c.d)
    (hG1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).gates = 1)
    (hP1 : ∀ f : Nat, (c.P.defs[f]!).prim = true → (c.P.defs[f]!).outLeaves = 1)
    (hCP : ∀ k node, c.node k = .ok node → (node.form = .prim → (c.P.defs[node.fn]!).prim = true) ∧
      (node.form = .call ∨ node.form = .scan → (c.P.defs[node.fn]!).prim = false))
    (hA : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size → ArgOk c k i)
    (hAP : ∀ k i, k < c.body.nodes.size → i < (c.body.nodes[k]!).args.size →
      ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hL0 : ∀ k i d L : Nat, ((c.body.nodes[k]!).args[i]!).dims = some (d, L) → L = 0 →
      i < (c.P.defs[(c.body.nodes[k]!).fn]!).body.params.size →
      ((c.P.defs[(c.body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0)
    (hR : ∀ k node, c.node k = .ok node → RetOk c.P c.memo node.fn)
    (hRW : ∀ k node, c.node k = .ok node → RetW c.P c.memo node.fn)
    (hLF : ∀ k node, c.node k = .ok node → (c.P.defs[node.fn]!).prim = false → Laid c.P node.fn ∧
      (c.P.defs[node.fn]!).body.offsets[(c.P.defs[node.fn]!).body.nodes.size]! = (c.P.defs[node.fn]!).gates) :
    ∀ fuel, DownNAt c fuel ∧ CalleeN c fuel ∧ ThroughN c fuel ∧ CarryN c fuel :=
  oneNode c sc hd hB hL hG1 hP1 hCP hA hAP hL0 hR hRW hLF
    (fun fuel hD hC => batchN_of c fuel hD (batchAffine_n c hB hL hR hRW hLF)
      (batchOne_n c hB hL fuel hD hC hA hAP hR hRW hLF) (batchStride_n c hB hL fuel hD hC hR hRW hLF))
    (fun fuel hC hK => scanN_of c hB fuel hC hK)

/-- **`down`, `fromCallee`, `throughArg` and `carry` keep a decoded composite's gates by node**, from `oneNode_of` with
the decoder's facts (`Decoded`, `WidthFacts`), wherever its callees' summaries hold `RetOk` and name their gates with
their widths (`RetW`). -/
theorem oneNode_decoded (P : Prog) (memo : Array Desc.Summary) {d : Nat} (hdP : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) (hDec : Decoded P) (hWF : WidthFacts P) (hB : Resolve.BodyOk P d)
    (hR : ∀ k node, (ctxOf P memo d).node k = .ok node → RetOk P memo node.fn)
    (hRW : ∀ k node, (ctxOf P memo d).node k = .ok node → RetW P memo node.fn) :
    ∀ fuel, DownNAt (ctxOf P memo d) fuel ∧ CalleeN (ctxOf P memo d) fuel ∧
      ThroughN (ctxOf P memo d) fuel ∧ CarryN (ctxOf P memo d) fuel := by
  have hfn : ∀ k node, (ctxOf P memo d).node k = .ok node → node.fn < P.defs.size := by
    intro k node hn
    have hk := node_lt hn
    have := (hB.1 k hk).1
    have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
    rwa [hg] at this
  have hpr : ∀ f : Nat, (P.defs[f]!).prim = true → f < P.defs.size := by
    intro f hf
    by_contra hlt
    rw [getElem!_neg P.defs f hlt] at hf
    cases hf
  obtain ⟨hLd, -, hnodes⟩ := (hWF d hdP).2 hp
  refine oneNode_of (ctxOf P memo d) (.top d 0 none) rfl hB hLd
    (fun f hf => (hWF f (hpr f hf)).1 hf) (fun f hf => (hDec f (hpr f hf)).1 hf)
    (fun k node hn => by
      have hk := node_lt hn
      have hg : (P.defs[d]!).body.nodes[k]! = node := node_get hn
      rw [← hg]
      exact hnodes k hk)
    (argOk_decoded P memo hdP hDec) (fun k i hk hi => ctxOf_argCount P memo d k i hk hi) ?_ hR hRW
    (fun k node hn hp' => ⟨((hWF node.fn (hfn k node hn)).2 hp').1, ((hWF node.fn (hfn k node hn)).2 hp').2.1⟩)
  intro k i e L hdims hL hip
  change ((((P.defs[d]!).body.nodes[k]!).args[i]!).dims = some (e, L)) at hdims
  by_cases hk : k < (P.defs[d]!).body.nodes.size
  · by_cases hi : i < ((P.defs[d]!).body.nodes[k]!).args.size
    · exact ((hDec d hdP).2.2 k hk i hi).2.2 e L hdims hL hip
    · rw [getElem!_neg _ i hi] at hdims; cases hdims
  · rw [getElem!_neg _ k hk, getElem!_neg _ i (Nat.not_lt_zero i)] at hdims; cases hdims

/-- A piece argument `i` of node `k` gives (`PS.pieces`), through `down`: segments keeping the body's gates by node, the
argument naming references. -/
theorem piece_n (c : Desc.Ctx) (fuel : Nat) (hD : DownNAt c fuel) {k i : Nat} (hA : ArgOk c k i)
    (hP : ∀ s ∈ (c.arg k i).segs.toList, ∀ q ∈ s.ps.toList, 0 < q.count)
    (hR : ∀ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      Resolve.RefOk c.P c.d k (((c.body.nodes[k]!).args[i]!).seq.get t))
    {lo hi : Nat} {q : Piece} (hq : q ∈ (c.arg k i).pieces lo hi) {segs : Array Seg}
    (h : Desc.down c fuel q = .ok segs) : ∀ s ∈ segs.toList, NodeW c.P c.d s := by
  have hq' := Array.mem_toList_iff.2 hq
  have hmem : ∀ x ∈ pos q, ∃ t < ((c.body.nodes[k]!).args[i]!).seq.count,
      x = refPos (((c.body.nodes[k]!).args[i]!).seq.get t) := by
    intro x hx
    have : x ∈ posAll ((c.arg k i).pieces lo hi) := List.mem_flatMap.2 ⟨q, hq', hx⟩
    rw [pieces_pos _ hA.1 lo hi] at this
    have := mem_sl this
    rw [hA.2] at this
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 this
    exact ⟨t, List.mem_range.1 ht, rfl⟩
  have hcnt : 0 < q.count := by
    unfold Desc.PS.pieces at hq'
    rw [Array.toList_flatMap, List.mem_flatMap] at hq'
    obtain ⟨g, hg, hq'⟩ := hq'
    rw [Array.toList_flatMap, List.mem_flatMap] at hq'
    obtain ⟨m, -, hq'⟩ := hq'
    exact shiftGates_count _ _ (psSlice_ok _ hA.1 hP lo hi g hg).2.2 q hq'
  exact hD q segs hcnt
    (fun x hx => by obtain ⟨t, -, rfl⟩ := hmem x hx; simp [refPos])
    (fun x hx k' hk' => by
      obtain ⟨t, ht, rfl⟩ := hmem x hx
      simp only [refPos, Tgt.ref.injEq] at hk'
      obtain ⟨hp', rfl⟩ := hk'
      have := ((hR t ht).2 hp').2
      simp only [refPos]
      exact_mod_cast this)
    (fun x hx i' hi' => by
      obtain ⟨t, ht, rfl⟩ := hmem x hx
      simp only [refPos, Tgt.ref.injEq] at hi'
      obtain ⟨hp', rfl⟩ := hi'
      have := (hR t ht).1 hp'
      simp only [refPos]
      exact ⟨this.1, by exact_mod_cast this.2⟩)
    ⟨0, fun w p hx => by
      obtain ⟨t, -, he⟩ := hmem _ hx
      simp [refPos] at he⟩ h

/-- **Every segment `down` gives a piece of a root node's argument, one `Desc.memberRows` gives, or one
`Desc.carriesRead` gives, keeps the root's gates by node** (`piece_n`, `throughPS_n`, `carriesRead_of`). -/
theorem rootN (P : Prog) (memo : Array Desc.Summary) (hm : memoOf P = .ok memo) (hF : RootFacts P)
    (hWF : WidthFacts P) {k : Nat} {node : Flock.Program.Node} (hn : (rootCtx P memo).node k = .ok node) {s : Seg}
    (hs : (∃ i, FromDown (rootCtx P memo) k i s) ∨ FromRows (rootCtx P memo) k node s ∨
      FromCarry (rootCtx P memo) k node s) : NodeW P P.root s := by
  have hR := root_retOk P memo hm hF
  have hRW := root_retW P memo hm hF hWF
  obtain ⟨hr, hp, -, hDec, -, -, -, -, hB⟩ := hF
  have hBr := hB P.root hr (reached_root P hr)
  rw [rootCtx_eq] at hn hs hR hRW
  have hW := oneNode_decoded P memo hr hp hDec hWF hBr hR hRW Flock.Steps.FUEL
  have hA := argOk_decoded P memo hr hDec
  have hPS : ∀ k node i p bp segs, (ctxOf P memo P.root).node k = .ok node →
      Desc.throughPS ((ctxOf P memo P.root).arg k i) p = some bp →
      Desc.down (ctxOf P memo P.root) Flock.Steps.FUEL bp = .ok segs → ∀ s ∈ segs.toList, NodeW P P.root s := by
    intro k node i p bp segs hn hp hd
    have hk := node_lt hn
    by_cases hi' : i < ((ctxOf P memo P.root).body.nodes[k]!).args.size
    · exact throughPS_n _ _ hW.1 (hA k i hk hi') (ctxOf_argCount P memo P.root k i hk hi') ((hBr.1 k hk).2.1 i)
        hp hd
    · rw [throughPS_empty (ctxOf_arg_ge P memo P.root k i hi')] at hp
      cases hp
  have hk := node_lt hn
  rcases hs with ⟨i, lo, hi, q, hq, segs, hd, hs⟩ | ⟨i, u, ps, hmr, p, hp, segs, hd, hs⟩ | ⟨hfs, hn2, segs, hc, hs⟩
  · by_cases hi' : i < ((ctxOf P memo P.root).body.nodes[k]!).args.size
    · exact piece_n _ _ hW.1 (hA k i hk hi') (ctxOf_argCount P memo P.root k i hk hi') ((hBr.1 k hk).2.1 i) hq hd
        s (Array.mem_toList_iff.2 hs)
    · rw [pieces_empty (ctxOf_arg_ge P memo P.root k i hi')] at hq
      simp at hq
  · obtain ⟨p', hp'⟩ := memberRows_mem _ node k i u hmr p (Array.mem_toList_iff.2 hp)
    exact hPS k node i p' p segs hn hp' hd s (Array.mem_toList_iff.2 hs)
  · refine carriesRead_of (NodeW P P.root) _ hR ?_ ?_ hPS k node segs hn hfs hn2 hc s (Array.mem_toList_iff.2 hs)
    · intro k' node' hn' hf' m a b segs' hm' _ h'
      refine hW.2.1 k' m a b segs' hm' ?_ h'
      have hk' := node_lt hn'
      have hg : (P.defs[P.root]!).body.nodes[k']! = node' := node_get hn'
      exact (((hWF P.root hr).2 hp).2.2 k' hk').2 (Or.inr (by rw [hg]; exact hf'))
    · intro k' node' hn' hm0 lo hi' segs' hlv h'
      exact init_n _ hn' hW.2.2.1 hm0 hlv h'

/-- Gates of one node of the root's body (`InNode`), in a laid-out root, are one of its nodes' (`nodeGates`): an
index past its nodes has no gate, and wherever `xs` names none any node is. -/
theorem nodeGates_of_inNode (P : Prog) (hl : Desc.rootLaidOut P = true) {k : Nat} (hk : k < nodeCount P)
    {xs : List (Tgt × Int)} (h : ∃ k', InNode P P.root k' xs) :
    ∃ k' < nodeCount P, ∀ w p, (Tgt.gate w, p) ∈ xs → p.toNat ∈ nodeGates P k' := by
  obtain ⟨hsz, -, -, -⟩ := DescAgree.rootLaidOut_spec P hl
  obtain ⟨k', hk'⟩ := h
  by_cases hg : ∃ w p, (Tgt.gate w, p) ∈ xs
  · obtain ⟨w, p, hx⟩ := hg
    have hlt : k' < nodeCount P := by
      by_contra hge
      have hb := hk' w p hx
      have e : (P.defs[P.root]!).body.offsets[k' + 1]! = 0 := by
        apply getElem!_neg
        change ¬ (k' + 1 < (rootOffs P).size)
        omega
      rw [e] at hb
      omega
    refine ⟨k', hlt, fun w p hx => ?_⟩
    obtain ⟨h0, h1⟩ := hk' w p hx
    rw [DescAgree.mem_nodeGates]
    have h0' : (((rootOffs P)[k']! : Nat) : Int) ≤ p := h0
    have h1' : p < (((rootOffs P)[k' + 1]! : Nat) : Int) := h1
    omega
  · exact ⟨k, hk, fun w p hx => absurd ⟨w, p, hx⟩ hg⟩

/-- **`RootOneNode` from the decoder's facts** (`RootFacts`, `WidthFacts`): `rootN`'s segments keep the root's gates
by node, each node one of its nodes (`nodeGates_of_inNode`). -/
theorem rootOneNode_of
    (hF : ∀ bytes P, Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.walkable P = true →
      RootFacts P ∧ WidthFacts P) : RootOneNode := by
  intro bytes P memo c hd ho hl _ hw hm _ k hk node hn s hs
  obtain ⟨hRF, hWF⟩ := hF bytes P hd ho hw
  have hN := rootN P memo hm hRF hWF hn hs
  exact ⟨fun q hq _ => nodeGates_of_inNode P hl hk (hN.1 q hq),
    fun h1 h2 _ => nodeGates_of_inNode P hl hk (hN.2 h1 h2)⟩

end NciCheck.Properties.SecurityProofs.ReadsExact
