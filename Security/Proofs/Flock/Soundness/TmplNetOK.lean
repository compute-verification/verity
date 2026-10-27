import Proofs.Flock.Soundness.TmplUnitNet
import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.Discharge.Integrate.Bridge

/-!
# A templated file's nets fit their slots

The setup facts `nets_ok` gives a flat file's nets (`NetOK`: the constant and input rows inside the slot, the ports
inside the input rows; at least a word of rows; and the constant row reading itself, `ConstRow`) hold of a net
`Net.ofRowsAt` built (`ofRowsAt_netOK`): its constant is placed at `pos[constPos]`, past the input rows, which `pos`
keeps in place, and its committed constant row is the logical one read through `pos`. So they hold of a read templated
unit's committed net (`TmplUnit.netOK`), and of every net of a templated file's circuit (`parseTmpl_nets_ok`, through
`nets_okPre`, `nets_ok` over `ParseFactsPre`). So do the facts `Discharge.Integrate.net_width` and
`Discharge.Inst.portBits_nodup_net` give a flat file's nets: `A` and `B` stored at the slot's width, at most 32 lookup
bits, distinct port bits (`ofRowsAt_width`, `nets_widthPre`, `parseTmpl_net_width`). With them every pair Δ writes
after its first loop is at an input bit of a net slot (`deltaIn_inBitPre`, `parseTmpl_deltaIn`). They are facts of the committed nets the lincheck reads, whatever the
audit's unit for a templated statement. No verifier path reads this file.
-/

open Flock Flock.Tmpl

namespace FlockSoundness

/-- **A net committed at a position map fits its slot and its constant row reads itself.** -/
theorem ofRowsAt_netOK {name : String} {useful constPos : ℕ} {ig og : Array Group} {ra rb : Array (Array ℕ)}
    {pos : Array ℕ} {net : Net} (h : Net.ofRowsAt name useful constPos ig og ra rb pos = .ok net) :
    NetOK net ∧ 7 ≤ net.unitLog ∧ (net.unitLog ≤ 32 → ConstRow net) := by
  obtain ⟨n, hn, -, hcp, hul, ha, hb, -, hig, hiw, hlk, -, hpos, -, -⟩ := Net.ofRowsAt_ok h
  obtain ⟨-, -, -, -, -, -, -, hle, hca, hcb, -, -, hlk0, -⟩ := Net.ofRows_ok hn
  have hK : constPos < useful := by have := (ofRows_groups hn).1; omega
  have hin : n.inWords * WORD ≤ pos.getD constPos 0 := by
    refine Nat.le_of_not_lt fun hlt => ?_
    have e := Net.posOk_inputs hpos hlt
    have := Net.posOk_inj hpos (by omega) hK e
    omega
  have h7 : 7 ≤ net.unitLog := by rw [hul]; exact le_max_left _ _
  have h1 : 2 ^ log2ceil (pos.getD constPos 0 + 1) ≤ 2 ^ net.unitLog :=
    Nat.pow_le_pow_right (by norm_num) (by rw [hul]; exact le_max_right _ _)
  have h2 := le_two_pow_log2ceil (pos.getD constPos 0 + 1)
  refine ⟨⟨by rw [hcp]; omega, by rw [hiw]; omega, ?_⟩, h7, fun hu => ?_⟩
  · rw [hig, hiw]
    exact built_ports ⟨_, _, _, _, _, _, _, hn⟩
  · have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hu
    have hpa := Net.placed_at hpos ra hK
    have hpb := Net.placed_at hpos rb hK
    rw [hca] at hpa
    rw [hcb] at hpb
    refine ⟨by rw [hiw, hcp]; exact hin, fun j => ?_, fun j => ?_⟩
    · rw [sparseEntry_ofRows ha, ofRows_row ha hw, hcp, hpa]
      by_cases hj : j = pos.getD constPos 0
      · subst hj; simp
      · rw [Array.getD_eq_getD_getElem?] at hj
        simp [hj, Ne.symm hj]
    · rw [sparseEntry_ofRows hb, ofRows_row hb hw, hcp, hpb, hlk, hlk0]
      show _ + 0 = _
      by_cases hj : j = pos.getD constPos 0
      · subst hj; simp
      · rw [Array.getD_eq_getD_getElem?] at hj
        simp [hj, Ne.symm hj]

/-- **A template's committed net fits its slot and its constant row reads itself.** -/
theorem unrollAt_netOK {t : Tmpl} {lays : Array Lay} {net : Net} (h : Tmpl.unrollAt t lays = .ok net) :
    NetOK net ∧ 7 ≤ net.unitLog ∧ (net.unitLog ≤ 32 → ConstRow net) := by
  unfold Tmpl.unrollAt at h
  obtain ⟨_, -, h2⟩ := Flock.TmplOrder.bind_ok h
  exact ofRowsAt_netOK h2

/-- **`nets_ok` with nets `pre` first, of any kind**: every net of a circuit `HmRow.parse` accepted after `pre` fits its
slot, and has its constant row, when `pre`'s nets do. -/
theorem nets_okPre {tags : Flock.Tags} {pre : Array (String × Net)} {c : Flock.Circuit} (P : ParseFactsPre tags pre c)
    (hpre : ∀ x ∈ pre.toList, NetOK x.2 ∧ 7 ≤ x.2.unitLog ∧ (x.2.unitLog ≤ 32 → ConstRow x.2)) (i : ℕ) :
    NetOK (c.nets.getD i default).2 ∧ (i < c.nets.size → 7 ≤ (c.nets.getD i default).2.unitLog ∧
      ((c.nets.getD i default).2.unitLog ≤ 32 → ConstRow (c.nets.getD i default).2)) := by
  by_cases hi : i < c.nets.size
  · rcases P.nets _ (getD_mem_toList hi) with hp | hp | ⟨_, _, _, _, _, hb⟩
    · exact ⟨(hpre _ hp).1, fun _ => (hpre _ hp).2⟩
    · exact ⟨(netOK_built hp).1, fun _ => (netOK_built hp).2⟩
    · exact ⟨(netOK_lookup hb).1, fun _ => (netOK_lookup hb).2⟩
  · have hd : (c.nets.getD i default).2 = default := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]
      rfl
    rw [hd]
    refine ⟨⟨by decide, by decide, fun q hq => ?_⟩, fun h => absurd h hi⟩
    have : portCols (default : Net).inGroups = #[] := rfl
    rw [this] at hq
    cases hq

/-- **A net committed at a position map stores `A` and `B` at its slot's width**, has no lookup table, and its port bits
are distinct. -/
theorem ofRowsAt_width {name : String} {useful constPos : ℕ} {ig og : Array Group} {ra rb : Array (Array ℕ)}
    {pos : Array ℕ} {net : Net} (h : Net.ofRowsAt name useful constPos ig og ra rb pos = .ok net) :
    net.a.width = 2 ^ net.unitLog ∧ net.b.width = 2 ^ net.unitLog ∧ net.lookup = none ∧
      (portBits net.inGroups).Nodup := by
  obtain ⟨n, hn, -, -, -, ha, hb, -, hig, -, hlk, -⟩ := Net.ofRowsAt_ok h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, hlk0, -⟩ := Net.ofRows_ok hn
  refine ⟨Discharge.Integrate.ofRows_width ha, Discharge.Integrate.ofRows_width hb, hlk.trans hlk0, ?_⟩
  rw [hig]
  exact Discharge.Aliased.portBits_nodup ⟨_, _, _, _, _, _, _, hn⟩

theorem unrollAt_width {t : Tmpl} {lays : Array Lay} {net : Net} (h : Tmpl.unrollAt t lays = .ok net) :
    net.a.width = 2 ^ net.unitLog ∧ net.b.width = 2 ^ net.unitLog ∧ net.lookup = none ∧
      (portBits net.inGroups).Nodup := by
  unfold Tmpl.unrollAt at h
  obtain ⟨_, -, h2⟩ := Flock.TmplOrder.bind_ok h
  exact ofRowsAt_width h2

/-- **`Discharge.Integrate.net_width` and `Discharge.Inst.portBits_nodup_net` with nets `pre` first, of any kind**:
every net of a circuit `HmRow.parse` accepted after `pre` stores `A` and `B` at its slot's width, reads at most 32
lookup bits and has distinct port bits, when `pre`'s nets do. -/
theorem nets_widthPre {tags : Flock.Tags} {pre : Array (String × Net)} {c : Flock.Circuit}
    (P : ParseFactsPre tags pre c)
    (hpre : ∀ x ∈ pre.toList, x.2.a.width = 2 ^ x.2.unitLog ∧ x.2.b.width = 2 ^ x.2.unitLog ∧ x.2.lookup = none ∧
      (portBits x.2.inGroups).Nodup) {i : ℕ} (hi : i < c.nets.size) :
    (c.nets.getD i default).2.a.width = 2 ^ (c.nets.getD i default).2.unitLog ∧
      (c.nets.getD i default).2.b.width = 2 ^ (c.nets.getD i default).2.unitLog ∧
      (∀ lk, (c.nets.getD i default).2.lookup = some lk → lk.bits ≤ 32) ∧
      (portBits (c.nets.getD i default).2.inGroups).Nodup := by
  rcases P.nets _ (getD_mem_toList hi) with hp | hp | ⟨_, _, _, _, _, hl⟩
  · obtain ⟨ha, hb, hlk, hn⟩ := hpre _ hp
    exact ⟨ha, hb, (fun lk h => by rw [hlk] at h; cases h), hn⟩
  · obtain ⟨hlk, -, -, -, -, _, _, -, -, ha, hb⟩ := built_spec hp
    exact ⟨Discharge.Integrate.ofRows_width ha, Discharge.Integrate.ofRows_width hb,
      (fun lk h => by rw [hlk] at h; cases h), Discharge.Aliased.portBits_nodup hp⟩
  · obtain ⟨h1, h2, h3⟩ := Discharge.Integrate.build_width hl
    refine ⟨h1, h2, h3, ?_⟩
    rw [(build_facts hl).2.1, Discharge.Aliased.portBits_eq]
    simpa using List.nodup_range'

end FlockSoundness

namespace Flock

/-- **A read templated unit's committed net fits its slot and its constant row reads itself.** -/
theorem TmplUnit.netOK (tu : TmplUnit) :
    FlockSoundness.NetOK tu.net ∧ 7 ≤ tu.net.unitLog ∧ (tu.net.unitLog ≤ 32 → FlockSoundness.ConstRow tu.net) :=
  FlockSoundness.unrollAt_netOK tu.unrolled

end Flock

namespace FlockSoundness

/-- **Every net of a templated file's circuit fits its slot and has its constant row**: its unit, the read unit's
committed net (`TmplUnit.netOK`), and the file's own nets, as a flat file's (`nets_ok`). -/
theorem parseTmpl_nets_ok {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {c : Flock.Circuit} (h : HmOut.parseTmpl tags bytes tables = .ok c) (i : ℕ) :
    NetOK (c.nets.getD i default).2 ∧ (i < c.nets.size → 7 ≤ (c.nets.getD i default).2.unitLog ∧
      ((c.nets.getD i default).2.unitLog ≤ 32 → ConstRow (c.nets.getD i default).2)) := by
  obtain ⟨tu, c₀, P, -, rfl, -, -⟩ := TmplValue.parseTmpl_facts h
  refine nets_okPre P (fun x hx => ?_) i
  obtain rfl : x = ("unit", tu.net) := by simpa using hx
  exact tu.netOK

/-- **Every net of a templated file's circuit stores `A` and `B` at its slot's width**, reads at most 32 lookup bits and
has distinct port bits: its unit, the read unit's committed net, and the file's own nets, as a flat file's. -/
theorem parseTmpl_net_width {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {c : Flock.Circuit} (h : HmOut.parseTmpl tags bytes tables = .ok c) {i : ℕ} (hi : i < c.nets.size) :
    (c.nets.getD i default).2.a.width = 2 ^ (c.nets.getD i default).2.unitLog ∧
      (c.nets.getD i default).2.b.width = 2 ^ (c.nets.getD i default).2.unitLog ∧
      (∀ lk, (c.nets.getD i default).2.lookup = some lk → lk.bits ≤ 32) ∧
      (portBits (c.nets.getD i default).2.inGroups).Nodup := by
  obtain ⟨tu, c₀, P, -, rfl, -, -⟩ := TmplValue.parseTmpl_facts h
  refine nets_widthPre P (fun x hx => ?_) hi
  obtain rfl : x = ("unit", tu.net) := by simpa using hx
  exact unrollAt_width tu.unrolled

/-- **`deltaIn_inBit` with nets `pre` first, of any kind**: every pair Δ writes after its first loop is at an input bit of
a net slot, when `pre`'s nets fit their slots. -/
theorem deltaIn_inBitPre {tags : Flock.Tags} {pre : Array (String × Net)} {c : Flock.Circuit}
    (P : ParseFactsPre tags pre c)
    (hpre : ∀ x ∈ pre.toList, NetOK x.2 ∧ 7 ≤ x.2.unitLog ∧ (x.2.unitLog ≤ 32 → ConstRow x.2)) (C : CheckFacts c)
    {x : ℕ} (hx : DeltaIn c x) : InBit c x :=
  deltaIn_inBit_of (fun n => (nets_okPre P hpre n).1.2.2) P.unitNet P.shaNet P.hmNet C hx

/-- **A templated file's parse passed `HmRow.check`, and every pair Δ writes after its first loop is at an input bit of
a net slot**, on the circuit before the unit is carried on it. -/
theorem parseTmpl_deltaIn {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {c : Flock.Circuit} (h : HmOut.parseTmpl tags bytes tables = .ok c) :
    ∃ (tu : TmplUnit) (c₀ : Flock.Circuit), c = { c₀ with tmpl := some tu } ∧ CheckFacts c₀ ∧
      ∀ x, DeltaIn c₀ x → InBit c₀ x := by
  obtain ⟨tu, c₀, P, -, he, -, -⟩ := TmplValue.parseTmpl_facts h
  obtain ⟨ls, hck⟩ := P.check
  have C := check_facts P.typed hck
  refine ⟨tu, c₀, he, C, fun x hx => deltaIn_inBitPre P (fun y hy => ?_) C hx⟩
  obtain rfl : y = ("unit", tu.net) := by simpa using hy
  exact tu.netOK

end FlockSoundness
