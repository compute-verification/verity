import Properties.Proofs.Nci.BodiesOk.ForReads
import Properties.Proofs.Nci.ReadsExact

/-!
# `ReadsExact.Decoded`, `ReadsExact.RetParts` and `ReadsExact.ScanOut`, proved

`ReadsExact.PartNonneg` is `BodiesOk.PartNonneg` (`partNonneg_iff`), so `ForReads.decoded` is `ReadsExact.Decoded` of
every decoded Program (`decoded_reads`), and `BodyFacts`' return is `ReadsExact.RetParts` (`retParts_reads`). `buildNode`'s `Shape` of a scan node is `ReadsExact.ScanOut` of every body
(`scanOut_of_ok`, `scanOut_reads`).
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

end NciCheck.Properties.SecurityProofs.BodiesOk
