import Proofs.Properties.Nci.BodiesOk.ForReads
import Proofs.Properties.Nci.ReadsSet

/-!
# `ReadsExact.Decoded`, `ReadsExact.RetParts`, `ReadsExact.ScanOut` and `ReadsExact.RootFacts`, proved

`ReadsExact.PartNonneg` is `BodiesOk.PartNonneg` (`partNonneg_iff`), so `ForReads.decoded` is `ReadsExact.Decoded` of
every decoded Program (`decoded_reads`), and `BodyFacts`' return is `ReadsExact.RetParts` (`retParts_reads`). `buildNode`'s `Shape` of a scan node is `ReadsExact.ScanOut` of every body
(`scanOut_of_ok`, `scanOut_reads`).

With the root a composite without parameters (`decode_root`), every callee decoded before its caller (`BodyFacts`)
and `bodyOk_reached`, those are `ReadsExact.RootFacts` of every decoded Program whose order passes (`rootFacts`).
`CalleeOk`, `NodeOk` and the bodies `Desc.walkable` lays out are `ReadsExact.WidthFacts` (`widthFacts`), so the gates
every root read names have their widths and are one root node's (`rootOneNode`, from `ReadsExact.rootOneNode_of`),
and `Novel.ReadsExact` holds wherever `ReadsExact.RootSet` does (`readsExact_of_set`). Every
reference of a node calling a primitive is read by a member (`cover_reads`) and a node calling a composite has the
shape `ReadsExact.compNodeSet` reads (`compShape_reads`), so `ReadsExact.RootSet` holds (`rootSet`).
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program

theorem partNonneg_iff (q : Part) : ReadsExact.PartNonneg q ↔ PartNonneg q := by
  cases q <;> exact Iff.rfl

/-- **The decoder's facts the core reads hold of every decoded Program.** -/
theorem decoded_reads (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P) :
    ReadsExact.Decoded P := by
  intro d hd
  obtain ⟨h1, h2, h3⟩ := decoded bytes P hdec d hd
  refine ⟨h1, h2, fun k hk i hi => ?_⟩
  obtain ⟨a, b, c⟩ := h3 k hk i hi
  exact ⟨a, fun q hq => (partNonneg_iff q).2 (b q hq), c⟩

/-- **The decoder's facts on a Definition's return hold of every decoded Program.** -/
theorem retParts_reads (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P) :
    ReadsExact.RetParts P := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro d hd
  obtain ⟨hc, -, -, ps, hps, hnn⟩ := hok d hd
  refine ⟨fun hp => by rw [(hc.1 hp).1]; rfl, fun _ => ?_⟩
  have hparts : (P.defs[d]!).body.ret.parts = ps := by rw [hps]; exact (Walk.ofParts_spec ps).1
  exact ⟨by rw [hparts]; exact hps,
    fun q hq => (partNonneg_iff q).2 (hnn q (by rw [hparts] at hq; exact Array.mem_def.2 hq))⟩

/-- A scan node of a Program whose Definitions are all `DefOk` outputs its final carry's leaves, then its ys'. -/
theorem scanOut_of_ok (P : Prog) (hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]!)
    (memo : Array NciCheck.Desc.Summary) (d : ℕ) : ReadsExact.ScanOut (ReadsExact.ctxOf P memo d) := by
  intro k node carry y hn hform hret
  have hk := ReadsExact.node_lt hn
  have hg := ReadsExact.node_get hn
  change k < (P.defs[d]!).body.nodes.size at hk
  change (P.defs[d]!).body.nodes[k]! = node at hg
  change (P.defs[node.fn]!).ret = _ at hret
  by_cases hd : d < P.defs.size
  · obtain ⟨-, hnd, -⟩ := (hok d hd).2.1 k hk
    rw [hg] at hnd
    obtain ⟨-, -, -, -, hsh, -⟩ := hnd
    obtain ⟨form, fn, args, out, axes, n, nxs⟩ := node
    dsimp only at hform hsh hret ⊢
    subst hform
    obtain ⟨carry', y', hr, hout, -⟩ := hsh
    rw [hret] at hr
    injection hr with e
    obtain ⟨rfl, rfl⟩ : carry = carry' ∧ y = y' := by simpa using e
    rw [hout, leaves_tuple2, leaves_array]
  · rw [getElem!_neg P.defs d hd] at hk
    exact absurd hk (Nat.not_lt_zero _)

/-- **A scan node's output is its final carry's leaves, then its ys'**, in every body of a decoded Program. -/
theorem scanOut_reads (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (memo : Array NciCheck.Desc.Summary) (d : ℕ) : ReadsExact.ScanOut (ReadsExact.ctxOf P memo d) := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  exact scanOut_of_ok P hok memo d

/-- **The decoder's facts `ReadsExact.RootSegs` reads hold of every decoded Program whose order passes**: no reached
Definition calls the root, each reached one being at most the root (`reached_spec`) and each callee before its caller. -/
theorem rootFacts (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) : ReadsExact.RootFacts P := by
  obtain ⟨hr, hp, hp0⟩ := decode_root bytes P hdec
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hcf : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ((P.defs[d]!).body.nodes[k]!).fn < d :=
    fun d hd k hk => ((hok d hd).2.1 k hk).1
  refine ⟨hr, hp, hp0, decoded_reads bytes P hdec, retParts_reads bytes P hdec,
    fun m d => scanOut_reads bytes P hdec m d, hcf, fun d hd hrd k hk => ?_,
    fun d _ hrd => (bodyOk_reached bytes P hdec hov d hrd).2⟩
  have hle : d ≤ P.root := (reached_spec P (fun d => d ≤ P.root) (Nat.le_refl _)
    (fun d' hd' _ k' hk' => by
      show _ ≤ P.root
      have hd'' : d' ≤ P.root := hd'
      have := hcf d' (by omega) k' hk'
      omega)).2 d hrd
  have := hcf d hd k hk
  omega

/-- **`Novel.ReadsExact` from `ReadsExact.RootNodes` and `ReadsExact.RootSet`**, with the decoder's facts
(`rootFacts`). -/
theorem readsExact_of_nodes_set (hN : ReadsExact.RootNodes) (hS : ReadsExact.RootSet) : Novel.ReadsExact :=
  ReadsExact.readsExact_of_rootReads (ReadsExact.rootReads_of
    (ReadsExact.rootSegs_of (fun bytes P hd ho => rootFacts bytes P hd ho) (ReadsExact.rootGates_of hN)) hS)

/-- **The decoder's facts the widths read hold of every decoded Program `Desc.walkable` passes**: a primitive is one
gate (`CalleeOk`), every composite's body is laid out (`Walk.bodiesLaidOut_spec`), and its nodes call what `NodeOk`
lets them. -/
theorem widthFacts (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hwk : NciCheck.Desc.walkable P = true) : ReadsExact.WidthFacts P := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hwk' := hwk
  unfold NciCheck.Desc.walkable at hwk'
  simp only [Bool.and_eq_true] at hwk'
  obtain ⟨-, hlay, -, -⟩ := Walk.bodiesLaidOut_spec P hwk'.1.2
  intro d hd
  refine ⟨fun hp => ((hok d hd).1.1 hp).2.1, fun hp => ?_⟩
  obtain ⟨-, hlast, hstep⟩ := hlay d hd hp
  refine ⟨hstep, hlast, fun k hk => ?_⟩
  obtain ⟨-, hnd, -⟩ := (hok d hd).2.1 k hk
  exact ⟨hnd.2.2.1, hnd.2.2.2.1⟩

/-- **`ReadsExact.RootOneNode`**: a root read's gates are one root node's, with the decoder's facts (`rootFacts`,
`widthFacts`). -/
theorem rootOneNode : ReadsExact.RootOneNode :=
  ReadsExact.rootOneNode_of fun bytes P hd ho hw => ⟨rootFacts bytes P hd ho, widthFacts bytes P hd hw⟩

/-- **`Novel.ReadsExact` from `ReadsExact.RootSet`**, with the decoder's facts (`rootFacts`,
`widthFacts`): the reads' widths are `ReadsExact.rootW_circuit`'s and their nodes `rootOneNode`'s. -/
theorem readsExact_of_set (hS : ReadsExact.RootSet) : Novel.ReadsExact :=
  readsExact_of_nodes_set
    (ReadsExact.rootNodes_of (fun bytes P hd ho hw => ⟨rootFacts bytes P hd ho, widthFacts bytes P hd hw⟩)
      rootOneNode) hS

/-- **Every reference of a node calling a primitive is read by a member**, unless it is a batch of none, in every body
of a decoded Program (`cover_of_nodeOk`). -/
theorem cover_reads (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P) :
    ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ∀ node, (P.defs[d]!).body.nodes[k]! = node →
      (P.defs[node.fn]!).prim = true → ¬ (node.form == .batch && node.n == 0) = true →
      ∀ i < node.args.size, ∀ t < (node.args[i]!).seq.count, ∃ m < node.members,
        ∃ t' < (if (node.form == .batch && (node.args[i]!).dims.isSome) = true then
          ((P.defs[node.fn]!).params[i]!).2.leaves else (node.args[i]!).seq.count),
        (if (node.form == .batch) = true then Resolve.memberIdx (node.args[i]!) m t' else t') = t := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro d hd k hk node hg hp hb
  obtain ⟨-, hnd, -⟩ := (hok d hd).2.1 k hk
  rw [hg] at hnd
  exact cover_of_nodeOk _ node hnd hp hb

/-- **Every reference of a node calling a composite is some member's parameter leaf** (`ReadsExact.upIdx`), at a node
with members: a call's leaf for leaf, a batch's as `cover_of_nodeOk` covers it, a scan's carry and unmapped arguments
leaf for leaf and its `xs` row by row. -/
theorem tile_of_nodeOk (c : Defn) (nd : Node) (hc : CalleeOk c) (hn : NodeOk c nd) (hp : c.prim = false)
    (hm : nd.members ≠ 0) :
    ∀ i < nd.args.size, ∀ t < (nd.args[i]!).seq.count, ∃ m < nd.members,
      ∃ j < (c.body.params[i]!).2.leaves, ReadsExact.upIdx nd i m j = t := by
  obtain ⟨hbp, -, -⟩ := hc.2 hp
  obtain ⟨-, hcnt, hpr, -, hsh, -⟩ := hn
  intro i hi t ht
  rw [hbp, hcnt i hi] at *
  obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
  dsimp only at *
  cases form
  · exact absurd (hpr rfl) (by simp [hp])
  · refine ⟨0, by simp [Node.members, beq_call_batch, show (Form.call == Form.scan) = false from rfl], t, ?_, rfl⟩
    rw [← hsh.2 i hi]; exact ht
  · have hmem : (Node.mk .batch fn args out axes n nxs).members = n := by simp [Node.members, beq_batch_batch]
    rw [hmem] at hm ⊢
    have hn0 : 0 < n := Nat.pos_of_ne_zero hm
    obtain ⟨-, hargs⟩ := hsh
    rcases hargs i hi with ⟨hd, hty⟩ | ⟨ax, mt, hrm, hmt, hd⟩
    · refine ⟨0, hn0, t, by rw [← hty]; exact ht, ?_⟩
      simp [ReadsExact.upIdx, Resolve.memberIdx, hd]
    · obtain ⟨L, hax, hlv, ⟨s, hs⟩, -⟩ := axis_spec ax _ _ _ hrm
      rw [hlv, hs] at ht
      have hL : 0 < L := by
        rcases Nat.eq_zero_or_pos L with h0 | h0
        · rw [h0] at ht; simp at ht
        · exact h0
      have hq : t / L < n * s := by
        rw [Nat.div_lt_iff_lt_mul hL]
        calc t < n * (L * s) := ht
          _ = n * s * L := by ring
      have hq' : t / L / n < s := by
        rw [Nat.div_lt_iff_lt_mul hn0]
        calc t / L < n * s := hq
          _ = s * n := by ring
      have hr := Nat.mod_lt t hL
      refine ⟨t / L % n, Nat.mod_lt _ hn0, t / L / n * L + t % L, ?_, ?_⟩
      · rw [← hmt, hs]
        calc t / L / n * L + t % L < t / L / n * L + L := by omega
          _ = (t / L / n + 1) * L := by ring
          _ ≤ s * L := Nat.mul_le_mul_right _ hq'
          _ = L * s := by ring
      · simp only [ReadsExact.upIdx, Resolve.memberIdx, hd, hax]
        have h1 : (t / L / n * L + t % L) / L = t / L / n := by
          rw [show t / L / n * L + t % L = t % L + t / L / n * L by ring, Nat.add_mul_div_right _ _ hL,
            Nat.div_eq_of_lt hr, Nat.zero_add]
        have h2 : (t / L / n * L + t % L) % L = t % L := by
          rw [show t / L / n * L + t % L = t % L + t / L / n * L by ring, Nat.add_mul_mod_self_right,
            Nat.mod_eq_of_lt hr]
        rw [h1, h2]
        have e1 := Nat.div_add_mod (t / L) n
        have e2 := Nat.div_add_mod t L
        calc (t / L / n * n + t / L % n) * L + t % L = (n * (t / L / n) + t / L % n) * L + t % L := by ring
          _ = t := by rw [e1, Nat.mul_comm]; exact e2
  · have hmem : (Node.mk .scan fn args out axes n nxs).members = n := by
      simp [Node.members, show (Form.scan == Form.scan) = true from rfl]
    rw [hmem] at hm ⊢
    have hn0 : 0 < n := Nat.pos_of_ne_zero hm
    obtain ⟨carry, y, -, -, hp0, ha0, hrest⟩ := hsh
    by_cases hi0 : i = 0
    · subst hi0
      refine ⟨0, hn0, t, by rw [hp0, ← ha0]; exact ht, ?_⟩
      simp [ReadsExact.upIdx]
    · by_cases hx : i ≤ nxs
      · obtain ⟨e, hty, he, hd⟩ := (hrest i hi hi0).1 hx
        rw [hty, leaves_array] at ht
        have hE : 0 < e.leaves := by
          rcases Nat.eq_zero_or_pos e.leaves with h0 | h0
          · rw [h0] at ht; simp at ht
          · exact h0
        refine ⟨t / e.leaves, by rw [Nat.div_lt_iff_lt_mul hE]; exact ht, t % e.leaves,
          by rw [← he]; exact Nat.mod_lt t hE, ?_⟩
        simp only [ReadsExact.upIdx, hd, Option.map_some, Option.getD_some, ne_eq, hi0, not_false_eq_true,
          hx, and_self, ↓reduceIte]
        have := Nat.div_add_mod t e.leaves
        rw [Nat.mul_comm]; exact this
      · refine ⟨0, hn0, t, by rw [← (hrest i hi hi0).2 hx]; exact ht, ?_⟩
        simp [ReadsExact.upIdx, hx]

/-- **Each parameter leaf of a member of a node calling a composite is in range** (`ReadsExact.upIdx`, from `up_ok`),
a scan's carry past the first iteration aside. -/
theorem lo_of_nodeOk (c : Defn) (nd : Node) (hc : CalleeOk c) (hn : NodeOk c nd) (hp : c.prim = false) :
    ∀ i < nd.args.size, ∀ m < nd.members, ∀ j < (c.body.params[i]!).2.leaves,
      (nd.form = .scan → i = 0 → m = 0) → ReadsExact.upIdx nd i m j < (nd.args[i]!).seq.count := by
  have hup := up_ok c nd hc hn
  have hsz := hn.1
  have hbp := (hc.2 hp).1
  intro i hi m hm j hj hsc
  have h := hup i (by rw [hbp]; omega) j hj m hm
  obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
  dsimp only at *
  cases form
  · simpa [ReadsExact.upIdx] using h
  · simpa [ReadsExact.upIdx] using h
  · simpa [ReadsExact.upIdx] using h
  · by_cases hi0 : i = 0
    · subst hi0
      obtain rfl := hsc rfl rfl
      simpa [ReadsExact.upIdx] using h
    · simpa [ReadsExact.upIdx, hi0] using h

/-- **A scan's carry is within its callee's return**: the callee returns `(carry, y)` and takes the carry first
(`Shape`). -/
theorem carry_le_of_nodeOk (c : Defn) (nd : Node) (hc : CalleeOk c) (hn : NodeOk c nd) (hp : c.prim = false)
    (hs : nd.form = .scan) : (c.body.params[0]!).2.leaves ≤ c.body.ret.count := by
  obtain ⟨hbp, hbr, -⟩ := hc.2 hp
  obtain ⟨-, -, -, -, hsh, -⟩ := hn
  obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
  dsimp only at hs hsh
  subst hs
  obtain ⟨carry, y, hret, -, hp0, -⟩ := hsh
  rw [hbp, hp0, hbr, hret, leaves_tuple2]
  omega

/-- **A node calling a composite has the shape `ReadsExact.compNodeSet` reads** (`ReadsExact.CompShape`), in every body
of a decoded Program: its callee's parameters are its body's, one per argument, a dimension of no leaves goes to a
parameter of none (`DimsOk`), each member's parameter leaf is in range (`lo_of_nodeOk`), at a node with members every
reference is one (`tile_of_nodeOk`), and a scan's carry is within its callee's return (`carry_le_of_nodeOk`). -/
theorem compShape_reads (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    {d : ℕ} (hd : d < P.defs.size) {k : ℕ} (hk : k < (P.defs[d]!).body.nodes.size) {node : Node}
    (hg : (P.defs[d]!).body.nodes[k]! = node) (hp : (P.defs[node.fn]!).prim = false) :
    ReadsExact.CompShape P node := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hfn, hnd, -⟩ := (hok d hd).2.1 k hk
  rw [hg] at hfn hnd
  have hc : CalleeOk P.defs[node.fn]! := (hok node.fn (by omega)).1
  obtain ⟨hbp, -, -⟩ := hc.2 hp
  unfold ReadsExact.CompShape
  exact ⟨hbp.symm, by rw [hbp]; exact hnd.1.symm,
    fun i hi e L h h0 => by rw [hbp]; exact hnd.2.2.2.2.2 i hi e L h h0,
    lo_of_nodeOk _ node hc hnd hp, tile_of_nodeOk _ node hc hnd hp, carry_le_of_nodeOk _ node hc hnd hp⟩

/-- **`ReadsExact.RootSet`**, from the decoder's facts (`rootFacts`, `widthFacts`, `cover_reads`, `compShape_reads`):
`ReadsExact.rootSet_of`. -/
theorem rootSet : ReadsExact.RootSet :=
  ReadsExact.rootSet_of (fun bytes P hd ho => rootFacts bytes P hd ho) (fun bytes P hd hw => widthFacts bytes P hd hw)
    (fun bytes P hd => cover_reads bytes P hd)
    (fun bytes P hd _ hd' _ hk _ hg hp => compShape_reads bytes P hd hd' hk hg hp)

end NciCheck.Properties.SecurityProofs.BodiesOk
