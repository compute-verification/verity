import FlockSoundness.Discharge.HiddenExec.SiteH
import FlockSoundness.Discharge.Layout.Keyed
import FlockSoundness.Protocol.Discharge.HiddenExec.KeyH

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProg`'s places in hidden-output tables

`Layout.keyReaders` (`Layout/KeyProg.lean`) at `SitesH.progPlaces_keyProg`, for the hidden-output statement. Its rows
are every port of the circuit: the input rows, as `Layout.keyRowOf` reads them, and the output rows `nIn + jo`, which
hold the returned outputs. So a unit's computed committed gate, which `Layout.keyReaders` reads as a single gate whose
bit region "out" pins (`Layout.outBitOf`), is here a bit of an output row whose `b ‖ c` the public file registers:

* `outBitRow`, `outRowOf`, `rowOfH`: the gate on copy `u'`'s output `j`, bit `t` (`Layout.OutColAt`, read back by
  `Layout.outDecode`) is bit `16 k + t` of output row `nIn + jo`, where the copy's output word `lo[j]` is word `k` of
  output port `jo` (`outPortOf`);
* `keyReadersH`: `Layout.Readers.ofRows` at those rows, their message columns (`keyRowAddrH`), `b ‖ c` (`keyBcAddrH`)
  and salt (`keySaltAddrH`) at each drawn unit's site (`siteAddrH`), the registration's `b ‖ c` the population's public
  file's (`popOf R`), and single gates with `Layout.keyGate (fun _ _ => false)`;
* `keyProgH_copies`: Δ's copies, from `copy_siteH` at an input row and `out_copyH` at an output row;
* `keyProgH_bcPinned`: each row's `b ‖ c` at a drawn unit's site is the population's (`bc_site_ofH`, at a `BcSiteH`);
* `keyProgH_valPinned`: the single gates are the constant and the zero (`Layout.single_gate`, with every computed
  committed gate an output row's bit, `OutsOut`), so they hold 1 and 0 (`constCols_keyProgH`, `zeroCols_keyProgH`);
* `keyLayoutH`, `keyRowsH` (`rowEnc` for the rows, `bitCode` elsewhere) and `keyRowsH_computes`, `HmRowComputes` at
  them (`bc_site_commitH`, `keyRowH_enc`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutColAt outDecode OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

/-! ## Output rows' bits -/

/-! ## Output-row copies at an accepted statement -/

/-! ## The readers -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : SitesH plan tab)

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n)))

variable (popOf : Reg → Public)

variable (ho : c.outNet = c.unitNet) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
  (houts : OutsOut c hU n outs)

/-- **The bytes `sha512/row/v1` hashes for a row at `keyProg`'s places in hidden-output tables** (`Layout.keyRow_enc`):
at a drawn unit `u`'s site, the value a message holds at row `q` of unit `u'`, `16 · words` bits, has `rowEnc` the row
prefix of `words` and the `2 · words` bytes the witness holds at the unit's VU's message columns, in its block. -/
theorem keyRowH_enc {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (u' : Fin n) (q : Fin c.ports.size) {words : ℕ} (hrw : portRowW c q = 16 * words) :
    Layout.rowEnc ((keyLayoutH hparse outs popOf ho hv1 houts).row S R u M (.inl (u', q))) =
      (HmRow.rowPrefix words).data.toList ++
        List.ofFn fun j : Fin (2 * words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          (siteAddrH (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val))) := by
  have hle : portRowW c q ≤ Finset.univ.sup (portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  show Layout.rowEnc (W := Finset.univ.sup (portRowW c) + 1)
    (Layout.widthOf (Finset.univ.sup (portRowW c) + 1) (portRowW c q),
      Layout.padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) =>
        keyRowAddrH si S R u q k.val)) = _
  rw [Layout.rowEnc_row hrw hle]
  congr 1
  refine List.ext_getElem (by simp) fun j h1 h2 => ?_
  simp only [List.getElem_ofFn]
  congr 1
  funext t
  have hj : (8 * j + t.val) / 8 = j := by omega
  have ht : (8 * j + t.val) % 8 = t.val := by omega
  show bit (Binding.wit plan tab S R u M (keyRowAddrH si S R u q (8 * j + t.val))) = _
  simp only [keyRowAddrH, hu, ↓reduceDIte, hj, ht]

/-- **`HmRowComputes` at `keyProg`'s rows in hidden-output tables** (`hHm` for `keyRowsH`): at every drawn unit's
committed wire held by a row, input or output, in a satisfying witness of the unit's table, the `b ‖ c` at the row's
`hm96` output is the commit string of the row's bytes (`rowEnc`) and the salt the witness holds at `Hm.saltCol`
(`bc_site_commitH`, `keyRowH_enc`). -/
theorem keyRowsH_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    (keyRowsH hparse outs popOf ho hv1 houts hsite hkeys).Computes := by
  unfold Binding.HmRows.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyLayoutH hparse outs popOf ho hv1 houts).pos g = pos
  rcases pos with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hwq : portRowW c q = 16 * (si.acc S R u hu).st.c.ports[q.val]!.words := by
      rw [hce, getElem!_pos c.ports q.val q.isLt]
      rfl
    have hb := bc_site_commitH (si.acc S R u hu) (by rw [hce]; exact ho) (by rw [hce]; exact hv1) hM
      (si.o S R u hu) (si.hvu S R u hu) hq
    show Binding.bytesAt _ (keyBcAddrH si S R u q) = Binding.commitString
      (Layout.rowEnc ((keyLayoutH hparse outs popOf ho hv1 houts).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (keySaltAddrH si S R u (.inl (u', q)) (hU := hU)))
    have e1 : keyBcAddrH si S R u q = fun (j : Fin 128) (t : Fin 8) => siteAddrH (si.acc S R u hu) (si.o S R u hu)
        (Layout.bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keyBcAddrH, hu, dite_true]
    have e2 : keySaltAddrH si S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        siteAddrH (si.acc S R u hu) (si.o S R u hu)
          (Hm.saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keySaltAddrH, hu, dite_true]
    rw [keyRowH_enc hparse outs popOf ho hv1 houts hu M u' q hwq, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end places

end FlockSoundness.Discharge.HiddenExec
