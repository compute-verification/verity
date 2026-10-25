import Properties.Pouw.LiveSalt
import Properties.Proofs.Pouw.Commit
import Properties.Proofs.NetworkRecords.Bytes

/-!
# Proofs of the Pearl-C live salt's properties (`Properties.Pouw.LiveSalt`)

The epoch's root is `PouwCheck.epochOpenRoot`, the port's own (`epochOpenRoot_eq`). Its message is the tag, the 64-byte
weights' root and u64be of the epoch, so two equal messages split at fixed offsets into equal parts (`epochRootBinding`).
A row seed's message ends in u64le of `u`·2^32 + `i`, so two equal messages have one call index and row
(`rowSeedBinding`). The ledger opens a session once (`LedgerShows`' last clause), so two acceptances by one session read
one open and its one root (`oneRoot_of`), and so for `PouwCheck.liveSaltCheck` (`oneRoot`).
-/

namespace Properties.Pouw.LiveSalt

open Lean (Json)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj)

/-- The epoch's root is the port's: `PouwCheck.epochOpenRoot`. -/
theorem epochOpenRoot_eq (w : ByteArray) (k : Nat) : epochOpenRoot w k = PouwCheck.epochOpenRoot w k := rfl

theorem epochRootBinding : EpochRootBinding := by
  intro w w' k k' hw hw' hk hk' h
  by_cases hm : epochMessage w k = epochMessage w' k'
  · left
    have e := congrArg bs hm
    simp only [epochMessage, bs_append, List.append_assoc] at e
    have e1 := List.append_cancel_left e
    obtain ⟨e2, e3⟩ := List.append_inj e1 (by rw [bs_length, bs_length, hw, hw'])
    exact ⟨bs_inj e2, Properties.Pouw.Commit.be8_inj k k' hk hk' e3⟩
  · right
    exact ⟨_, List.mem_singleton_self _, _, List.mem_singleton_self _, hm, h⟩

theorem le64_data (x : Nat) : (le64 x).data.toList =
    [(x >>> 0).toUInt8, (x >>> 8).toUInt8, (x >>> 16).toUInt8, (x >>> 24).toUInt8,
     (x >>> 32).toUInt8, (x >>> 40).toUInt8, (x >>> 48).toUInt8, (x >>> 56).toUInt8] := by
  simp [le64, List.range_succ]

theorem le64_inj (x y : Nat) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64)
    (h : (le64 x).data.toList = (le64 y).data.toList) : x = y := by
  simp only [le64_data, List.cons.injEq] at h
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, -⟩ := h
  simp only [← UInt8.toNat_inj] at h0 h1 h2 h3 h4 h5 h6 h7
  simp [Nat.shiftRight_eq_div_pow] at h0 h1 h2 h3 h4 h5 h6 h7
  omega

theorem le64_length (x : Nat) : (bs (le64 x)).length = 8 := by
  simp [bs, le64_data]

theorem rowSeedBinding : RowSeedBinding := by
  intro salt leaf rootB salt' leaf' rootB' u i u' i' hu hi hu' hi' hne h
  refine ⟨_, List.mem_singleton_self _, _, List.mem_singleton_self _, fun hm => hne ?_, h⟩
  have e := congrArg bs hm
  simp only [rowSeedMessage, bs_append] at e
  have e8 := (List.append_inj' e (by rw [le64_length, le64_length])).2
  have hx := le64_inj _ _ (by omega) (by omega) e8
  obtain ⟨rfl, rfl⟩ : u = u' ∧ i = i' := by omega
  rfl

/-- One session salts one epoch, for any check: the ledger opens a session once, so two acceptances by one session read
one open, and its root's hex names one root. -/
theorem oneRoot_of (check : Check) : OneRoot check := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ w k a _ _ w' k' a' _ _ h h' hs
  obtain ⟨_, _, _, -, -, -, -, -, -, hl, -⟩ := h
  obtain ⟨_, _, _, -, -, -, -, -, -, hl', -⟩ := h'
  obtain ⟨_, op, _, -, hop, -, -, ⟨hk, hss, -, -, -, -, hr⟩, -, huniq⟩ := hl
  obtain ⟨_, op', _, -, hop', -, -, ⟨hk', hss', -, -, -, -, hr'⟩, -, -⟩ := hl'
  have heq : op' = op := huniq op' hop' hk' (Or.inr (hss'.trans hs.symm))
  subst heq
  exact Properties.NetworkRecords.Bytes.hexOf_inj (Json.str.inj (hr.symm.trans hr'))

theorem oneRoot : OneRoot PouwCheck.liveSaltCheck := oneRoot_of _

end Properties.Pouw.LiveSalt
