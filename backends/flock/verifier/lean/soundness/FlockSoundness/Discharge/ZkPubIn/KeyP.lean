import FlockSoundness.Discharge.ZkPubIn.SiteP
import FlockSoundness.Discharge.HiddenExec.KeyH

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProgP`'s places, with the verifier's public inputs

HiddenExec's `keyReadersH`, `keyLayoutH`, `keyRowsH` and `keyRowsH_computes` (`HiddenExec/KeyH.lean`) at
`SitesHP.progPlaces_keyProgP`. The rows are `keyProg`'s (`HiddenExec.rowOfH`: `keyProgP` puts every input that is a
row bit on `keyProg`'s gate for it, and computes `keyProg`'s gates); a public gate is a single gate, and the
registration's bit there is the verifier's own (`Layout.keyGate` at `pubBit`):

* `siteAddrHP`, `keyRowAddrHP`, `keyBcAddrHP`, `keySaltAddrHP`: `siteAddrH`'s addresses at `AcceptedHP` sites;
* `keyReadersHP`: `Layout.Readers.ofRows` at those rows, single gates with `Layout.keyGate (fun _ g => pubBit …)`;
* `keyProgHP_copies`, `keyProgHP_bcPinned`, `keyProgHP_valPinned`: Δ's copies, the registered `b ‖ c`, and the single
  gates' values: the constant, the zero, and at a public gate the `in` region's bit, the verifier's own word's
  (`OwnWords`, at the unit's rank in the draw);
* `keyLayoutHP`, `keyRowsHP`, `keyRowsHP_computes`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model Audit Binding Placed HiddenExec
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

variable {T : Model.Statement}

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : SitesHP plan tab)

/-- **Bit `k` of row `q` at a drawn unit's site** (`keyRowAddrH`). -/
noncomputable def keyRowAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (k : ℕ) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrHP (si.acc S R u h) (si.o S R u h)
      (Typed.msgCol (si.acc S R u h).st.c (si.vu S R u h) q.val (k / 8) (k % 8))
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s `b ‖ c` at a drawn unit's site** (`keyBcAddrH`). -/
noncomputable def keyBcAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (j : Fin 128)
    (t : Fin 8) : Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrHP (si.acc S R u h) (si.o S R u h)
      (Layout.bcAddr (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s salt at a drawn unit's site** (`keySaltAddrH`). -/
noncomputable def keySaltAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (q : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n)) (j : Fin 192) (t : Fin 8) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    match q with
    | .inl (_, q) =>
      siteAddrHP (si.acc S R u h) (si.o S R u h) (Hm.saltCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
    | .inr _ => ⟨0, Nat.two_pow_pos _⟩
  else ⟨0, Nat.two_pow_pos _⟩

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (pi : HmIn.Ports) (hpi : HmIn.parse c = .ok pi) (own : ByteArray) (outs : Finset (Fin (keyN c hU n)))

/-- `SitesHP.progPlaces_keyProgP`'s places. -/
noncomputable abbrev keyPlaceHP := (SitesHP.progPlaces_keyProgP hU hparse pi hpi outs).derived.place

variable (popOf : Reg → Public)

/-- **The readers at `keyProgP`'s sites**: `keyReadersH`'s rows (`rowOfH`), addresses at the `AcceptedHP` sites, the
registration's `b ‖ c` the population's public file's (`popOf R`), and single gates with `Layout.keyGate` at the
public gates' bits (`pubBit`, the verifier's own words). -/
noncomputable def keyReadersHP : Layout.Readers ((keyProgP c hU n pi).circuit outs) plan tab :=
  Layout.Readers.ofRows (keyPlaceHP hparse pi hpi outs) (Layout.keyN_pos c hU n) (portRowW c)
    (W := Finset.univ.sup (portRowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (rowOfH c hU n outs)
    (keyRowAddrHP si) (keySaltAddrHP si (hU := hU)) (keyBcAddrHP si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (Layout.keyGate (c := c) (hU := hU) (n := n) fun _ g => (pubBit c hU n pi own g).getD false)

variable (ho : c.outNet = c.unitNet) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
  (houts : OutsOut c hU n outs)

include hparse ho hv1 houts in
/-- **Δ's copies at `keyProgP`'s sites** (`keyProgH_copies`): a column on an input row's bit holds the witness's bit
at its key's message column; a column on an output row's bit, the bit at the output row's message column. -/
theorem keyProgHP_copies : (keyReadersHP hparse pi hpi own outs popOf).Copies (keyPlaceHP hparse pi hpi outs) := by
  sorry

include hparse in
/-- **The rows' `b ‖ c` at `keyProgP`'s sites is the registration's** (`keyProgH_bcPinned`), at every site whose
rank is the unit (`BcSiteHP`). -/
theorem keyProgHP_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHP hparse pi hpi own outs popOf).BcPinned (P := (keyProgP c hU n pi).partition outs) := by
  sorry

include hparse ho hv1 houts in
/-- **The single gates at `keyProgP`'s sites hold the registration's bits** (`keyProgH_valPinned`): the constant 1,
the zero 0, and a public gate the bit of the `in` region at its column (`HmIn.regions`), which is the verifier's own
word's at the unit's instance (`OwnWords`, at the unit's rank in the draw, `BcSiteHP`). -/
theorem keyProgHP_valPinned (hkeys : KeysRow c hU (portRowW c))
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHP hparse pi hpi own outs popOf).ValPinned (P := (keyProgP c hU n pi).partition outs) := by
  sorry

/-- **`lay` at `keyProgP`'s places** (`keyLayoutH`). -/
noncomputable def keyLayoutHP :
    Binding.Layout ((keyProgP c hU n pi).partition outs) plan tab
      (((keyProgP c hU n pi).partition outs).placeDecoder plan tab (keyPlaceHP hparse pi hpi outs)) :=
  (keyReadersHP hparse pi hpi own outs popOf).layout (keyPlaceHP hparse pi hpi outs) Layout.bitCode
    (keyProgHP_copies hparse pi hpi own outs popOf ho hv1 houts)

/-- **`rs` at `keyProgP`'s places** (`keyRowsH`). -/
noncomputable def keyRowsHP
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    Binding.HmRows (keyLayoutHP hparse pi hpi own outs popOf ho hv1 houts) :=
  (keyReadersHP hparse pi hpi own outs popOf).rows (keyPlaceHP hparse pi hpi outs) Layout.rowEnc Layout.rowEnc_inj
    Layout.bitCode Layout.bitCode_inj (keyProgHP_copies hparse pi hpi own outs popOf ho hv1 houts)
    (keyProgHP_bcPinned hparse pi hpi own outs popOf hsite)
    (keyProgHP_valPinned hparse pi hpi own outs popOf ho hv1 houts hkeys hsite)

/-- **`HmRowComputes` at `keyProgP`'s rows** (`keyRowsH_computes`). -/
theorem keyRowsHP_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    (keyRowsHP hparse pi hpi own outs popOf ho hv1 houts hsite hkeys).Computes := by
  sorry

end places

end FlockSoundness.Discharge.ZkPubIn
