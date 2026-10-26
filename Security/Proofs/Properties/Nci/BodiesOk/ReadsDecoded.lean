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
and `Novel.ReadsExact` holds wherever its open statement does, `ReadsExact.RootSet` (`readsExact_of_set`).
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

/-- **`Novel.ReadsExact` from its open statement**, `ReadsExact.RootSet`, with the decoder's facts (`rootFacts`,
`widthFacts`): the reads' widths are `ReadsExact.rootW_circuit`'s and their nodes `rootOneNode`'s. -/
theorem readsExact_of_set (hS : ReadsExact.RootSet) : Novel.ReadsExact :=
  readsExact_of_nodes_set
    (ReadsExact.rootNodes_of (fun bytes P hd ho hw => ⟨rootFacts bytes P hd ho, widthFacts bytes P hd hw⟩)
      rootOneNode) hS

end NciCheck.Properties.SecurityProofs.BodiesOk
