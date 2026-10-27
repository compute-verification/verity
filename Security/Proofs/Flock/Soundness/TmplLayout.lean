import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.TmplUnitAt
import Proofs.Flock.Soundness.TmplNetOK

/-!
# A templated statement has the layout the placement reads

`layout_of`'s templated form: a statement whose circuit `HmOut.parseTmpl` accepted has `Layout`, its unit the read
templated unit's logical net placed at `posMap` (`UnitAt.tmpl`). Every other fact is the flat file's, on the circuit
`HmRow.parse` accepted with the unit's committed net first (`TmplValue.parseTmpl_facts`), which only `Circuit.tmpl`
tells apart from the statement's.
-/

namespace FlockSoundness

open Flock

/-- **A statement with the circuit `HmOut.parseTmpl` accepted has the layout the placement reads** (`Layout`): its nets,
`HmRow.check`'s layout, `HmRow.pin` and `HmRow.delta` as `layout_of` reads them, and its unit placed at `posMap`. -/
theorem layout_of_tmpl {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {st : Stmt}
    (h : HmOut.parseTmpl tags bytes tables = .ok st.c) (hpin : HmRow.pin st.c = .ok st.pin)
    (hda : st.da = (HmRow.delta st.c st.pin).1) (hdb : st.db = (HmRow.delta st.c st.pin).2)
    (hm : st.m = st.c.kLog + st.nbl) (hk : st.c.kLog ≤ 27) : Layout st := by
  obtain ⟨tu, c₀, P, -, he, hu0, hn0⟩ := TmplValue.parseTmpl_facts h
  have hok := parseTmpl_nets_ok h
  obtain ⟨ls, hcheck⟩ := P.check
  have hty : st.c.typed = none := by rw [he]; exact P.typed
  have C : CheckFacts st.c := check_facts (tags := tags) (ls := ls) hty (by rw [he]; exact hcheck)
  have hun : st.c.unitNet < st.c.nets.size := by rw [he]; exact P.unitNet
  have hsh : st.c.shaNet < st.c.nets.size := by rw [he]; exact P.shaNet
  have hhm : st.c.hmNet < st.c.nets.size := by rw [he]; exact P.hmNet
  have hkinds : ∀ r ∈ st.c.ranges.toList, ∀ i, r.kind = .net i → i < st.c.nets.size := by rw [he]; exact P.kinds
  obtain ⟨hum, huk, huc⟩ := unit_range_facts_of hun C
  have ht : st.c.tmpl = some tu := by rw [he]
  have hu : st.c.unit = tu.net := by
    rw [he]
    show (c₀.nets.getD c₀.unitNet default).2 = tu.net
    rw [hu0, hn0]
  have A := UnitAt.tmpl ht hu
  obtain ⟨-, -, hUlk, -⟩ := unrollAt_width tu.unrolled
  obtain ⟨rp, hrp, ip, hkp, hcp, hpv⟩ := pin_spec hpin
  have hip := hkinds rp hrp ip hkp
  have hsl := slotLog_le C hrp hcp
  have hsl' : rp.slotLog = (st.c.nets.getD ip default).2.unitLog := ((C.bound_slotLog rp hrp).2 ip hkp).symm
  obtain ⟨h7, hcr⟩ := (hok ip).2 hip
  exact {
    disj := C.disj
    bound := fun r hr => (C.bound_slotLog r hr).1
    slotLog := fun r hr i hi => ((C.bound_slotLog r hr).2 i hi).symm
    fit := fun i => ⟨(hok i).1.1, (hok i).1.2.1⟩
    unit_mem := hum
    unit_kind := huk
    unit_count := huc
    unit_built := built_tmpl ht
    unit_at := A
    unit_lookup := by rw [hu]; exact hUlk
    unit_in := by rw [← A.in_eq]; exact le_trans A.in_le A.const_le
    unit_const := by rw [hu]; exact tu.netOK.2.2
    pin := ⟨rp, hrp, ip, hkp, hcp, hpv, hcr (by omega)⟩
    delta_b := by rw [hdb, hda, delta_snd _ _ hty]
    delta := by
      obtain ⟨rest, h1, h2⟩ := delta_split st.c st.pin hty
      exact ⟨rest, by rw [hda]; exact h1, fun p hp => deltaIn_inBit_of (fun n => (hok n).1.2.2) hun hsh hhm C (h2 p hp)⟩
    kLog := by omega
    kLog6 := by omega
    m := hm }

end FlockSoundness
