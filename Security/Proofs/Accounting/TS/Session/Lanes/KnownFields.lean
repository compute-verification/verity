import Proofs.Accounting.TS.Session.Lanes.KnownOf
import Proofs.Accounting.TS.Session.Lanes.KnownBits

/-!
# A lane's known fields hold at the opened rows

Where a session's known rows hold at the known content of the public inputs `P` (`KnownHeld` at `knownOf`, which
`Accounting.Guarantees.TS.LeafLanesSound` hands each session), the opened rows' bits at its `admit`, `pub` and `o0`
ports are `P`'s, the lane's index and `ok = 1`: the fields of `LaneHolds` that the known content fixes
(`LaneKnownHolds`: admit, salt, position, roots, lane and ok), and `CombHolds.ok` at the combine.

* `bitsNat_testBit`, `rowWords_testBit`: a known row's word `t / 16`'s bit `t % 16` is the row's bit `t`
  (`PouwWindow.Lanes.rowWords`, `bits_words`);
* `pubBits_salt`, `pubBits_position`, `pubBits_roots`, `pubBits_lane`: the `pub` row's bits at each field are the
  field's, at its first bit in `lanePorts` (`fieldBit_salt` …, by evaluation);
* `laneKnownHolds`, `combKnownOk`: through `holds_known` and `portBit_known` at each port.

Each port is taken as a known value's row (`KnownPort`: a `sha512/row/v2` port without segments; a one-row value is read
at row 0, `known_readAt_zero`) and each of its bits as on a wire of the unit (`RowBitRead`). Flock-verify's known-read
check gives both (`Lanes.KnownRead`'s `laneKnownHolds_of_check`, `combKnownOk_of_check`).
-/

namespace Accounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge Accounting.TS.Checked FlockSoundness.Discharge

/-- **The known content's fields of lane `l`'s view**, at the reader `rd` of its ports' row bits: the admit bits, the
salt, position and roots, the index `l` and `ok = 1`, each at its place in the layout. -/
structure LaneKnownHolds (T : Tile) (rd : String → ℕ → Option Bool) (P : LanePub) (l : Fin 32) : Prop where
  admit : ∀ i : Fin 32, rd "admit" (fieldBit (lanePorts T 0) "admit" + i.val) = some (P.admit i)
  salt : Holds (rd "pub") (fieldBit (lanePorts T 0) "salt") digestBits P.salt
  position : Holds (rd "pub") (fieldBit (lanePorts T 0) "position") positionBits P.position
  roots : Holds (rd "pub") (fieldBit (lanePorts T 0) "roots") (2 * digestBits) P.roots
  lane : Holds (rd "pub") (fieldBit (lanePorts T 0) "lane") laneBits l.val
  ok : rd "o0" 0 = some true

/-- **`LaneHolds` is its registered fields and its known ones**: the A and B rows and the record, with
`LaneKnownHolds`. -/
theorem LaneKnownHolds.laneHolds {K : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {L : Leaf K} {P : LanePub}
    {l : Fin 32} (hk : LaneKnownHolds T rd P l)
    (ha : ∀ (s : Fin 4) (k : Fin K), Holds (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].codes" + 8 * k.val) 8
      (encS 8 ((laneA L l s).codes k)))
    (haS : ∀ s : Fin 4, Holds (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].scale") 32 (laneA L l s).scale.toNat)
    (hb : ∀ (t : Fin 8) (k : Fin K), Holds (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].codes" + 8 * k.val) 8
      (encS 8 ((laneB L l t).codes k)))
    (hbS : ∀ t : Fin 8, Holds (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].scale") 32 (laneB L l t).scale.toNat)
    (hz : ∀ i j : Fin 32, Holds (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32 (encS 32 (L.z i j)))
    (hy : ∀ i j : Fin 32, Holds (rd "zy") (fieldBit (lanePorts T 0) "zy.y" + 16 * (i.val * T.c + j.val)) 16 (L.y i j).toNat) :
    LaneHolds K T rd L P l :=
  ⟨ha, haS, hb, hbS, hz, hy, hk.admit, hk.salt, hk.position, hk.roots, hk.lane, hk.ok⟩

theorem fieldBit_admit (T : Tile) : fieldBit (lanePorts T 0) "admit" = 0 := rfl
theorem fieldBit_salt (T : Tile) : fieldBit (lanePorts T 0) "salt" = 0 := rfl
theorem fieldBit_position (T : Tile) : fieldBit (lanePorts T 0) "position" = 512 := rfl
theorem fieldBit_roots (T : Tile) : fieldBit (lanePorts T 0) "roots" = 576 := rfl
theorem fieldBit_lane (T : Tile) : fieldBit (lanePorts T 0) "lane" = 1600 := rfl

/-- The number bits spell has those bits. -/
theorem bitsNat_testBit (bs : List Bool) (i : ℕ) : (bitsNat bs).testBit i = (bs[i]?).getD false := by
  induction bs generalizing i with
  | nil => simp [bitsNat]
  | cons b bs ih =>
    cases i with
    | zero => cases b <;> simp [bitsNat, Nat.testBit_zero] <;> omega
    | succ i =>
      rw [Nat.testBit_add_one]
      have e : bitsNat (b :: bs) / 2 = bitsNat bs := by
        cases b <;> simp [bitsNat] <;> omega
      rw [e, ih]
      simp

/-- **A row's bit from its words**: bit `t % 16` of word `t / 16` of `rowWords bs` is bit `t` of `bs`. -/
theorem rowWords_testBit (bs : List Bool) {t : ℕ} (ht : t < bs.length) :
    ((rowWords bs)[t / 16]!).testBit (t % 16) = (bs[t]?).getD false := by
  have hw : t / 16 < (bs.length + 15) / 16 := by omega
  have e : (rowWords bs)[t / 16]! = bitsNat ((bs.drop (16 * (t / 16))).take 16) := by
    unfold rowWords
    rw [getElem!_pos _ _ (by simpa using hw)]
    simp
  rw [e, bitsNat_testBit, List.getElem?_take, ite_eq_left (Nat.mod_lt _ (by norm_num)), List.getElem?_drop,
    Nat.div_add_mod]

theorem length_natBits (n x : ℕ) : (natBits n x).length = n := by simp [natBits]

theorem natBits_get {n x i : ℕ} (h : i < n) : (natBits n x)[i]? = some (x.testBit i) := by
  simp [natBits, h]

theorem length_pubBits (s p r l : ℕ) : (pubBits s p r l).length = 1616 := by
  simp only [pubBits, List.length_append, length_natBits, digestBits, positionBits, laneBits]

theorem getElem?_append_at {α : Type} {l₁ l₂ : List α} {n : ℕ} (hn : l₁.length = n) (i : ℕ) :
    (l₁ ++ l₂)[n + i]? = l₂[i]? := by
  rw [List.getElem?_append_right (by omega), show n + i - l₁.length = i by omega]

theorem pubBits_salt (s p r l : ℕ) {i : ℕ} (h : i < digestBits) : (pubBits s p r l)[0 + i]? = some (s.testBit i) := by
  have h' : i < 512 := h
  unfold pubBits
  rw [List.getElem?_append_left (by simp only [List.length_append, length_natBits, digestBits, positionBits]; omega),
    List.getElem?_append_left (by simp only [List.length_append, length_natBits, digestBits, positionBits]; omega),
    List.getElem?_append_left (by simp only [length_natBits, digestBits]; omega), Nat.zero_add, natBits_get h]

theorem pubBits_position (s p r l : ℕ) {i : ℕ} (h : i < positionBits) :
    (pubBits s p r l)[512 + i]? = some (p.testBit i) := by
  have h' : i < 64 := h
  unfold pubBits
  rw [List.getElem?_append_left (by simp only [List.length_append, length_natBits, digestBits, positionBits]; omega),
    List.getElem?_append_left (by simp only [List.length_append, length_natBits, digestBits, positionBits]; omega),
    getElem?_append_at (by simp only [length_natBits, digestBits]), natBits_get h]

theorem pubBits_roots (s p r l : ℕ) {i : ℕ} (h : i < 2 * digestBits) :
    (pubBits s p r l)[576 + i]? = some (r.testBit i) := by
  have h' : i < 1024 := h
  unfold pubBits
  rw [List.getElem?_append_left (by simp only [List.length_append, length_natBits, digestBits, positionBits]; omega),
    getElem?_append_at (by simp only [List.length_append, length_natBits, digestBits, positionBits]), natBits_get h]

theorem pubBits_lane (s p r l : ℕ) {i : ℕ} (h : i < laneBits) : (pubBits s p r l)[1600 + i]? = some (l.testBit i) := by
  unfold pubBits
  rw [getElem?_append_at (by simp only [List.length_append, length_natBits, digestBits, positionBits]), natBits_get h]

/-- **Port `q` is a known value's row**: a `sha512/row/v2` port without segments. -/
def KnownPort (s : RowsSession) (q : Fin s.c.ports.size) : Prop :=
  s.c.ports[q.val]!.segs = none ∧ s.c.ports[q.val]!.v2 = true

variable {s : RowsSession}

/-- **`o0` reads `ok = 1`** where the known content names `o0` with the row 1 and its bit 0 is read. -/
theorem ok_of_known (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    (hq : s.c.ports[q].name = "o0") (hP : KnownPort s q) {salt : ByteArray}
    (hm : ("o0", #[(okRow, salt)]) ∈ known) (hr : RowBitRead s u q 0) :
    portBit s (openedReg s) "o0" 0 = some true := by
  have h := portBit_known hN hK hu hS hcomp hP.1 hP.2 (ks := #[(okRow, salt)]) (by rw [hq]; exact hm) hr
  rw [hq, known_readAt_zero hK hS hcomp (r := (okRow, salt)) (by rw [hq]; exact hm)] at h
  rw [h]
  show some (((rowWords [true])[0 / 16]!).testBit (0 % 16)) = _
  rw [rowWords_testBit [true] (by simp)]
  rfl

/-- **Lane `l`'s known fields hold at the opened rows**: where its known rows hold at `knownOf P salts (some l)`
(`KnownHeld`), under flock-verify's complete check, in a circuit that names each port once, at unit 0 whose served roots
are registered, with its `admit`, `pub` and `o0` ports each a known value's one row (`KnownPort`) and every bit of them
read (`RowBitRead`), the view's admit bits, salt, position, roots and index are `P`'s and `l`, and `ok = 1`. -/
theorem laneKnownHolds (T : Tile) (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {P : LanePub} {salts : String → ByteArray} {l : Fin 32} (hK : KnownHeld s (knownOf P salts (some l)))
    {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ())
    {qa qp qo : Fin s.c.ports.size} (ha : s.c.ports[qa].name = "admit") (hp : s.c.ports[qp].name = "pub")
    (ho : s.c.ports[qo].name = "o0") (hPa : KnownPort s qa) (hPp : KnownPort s qp) (hPo : KnownPort s qo)
    (hra : ∀ k < 32, RowBitRead s u qa k) (hrp : ∀ k < 1616, RowBitRead s u qp k) (hro : RowBitRead s u qo 0) :
    LaneKnownHolds T (portBit s (openedReg s)) P l := by
  have hma : (s.c.ports[qa].name, #[(admitRow (List.ofFn P.admit), salts "admit")]) ∈ knownOf P salts (some l) := by
    rw [ha]; simp [knownOf, knownRows]
  have hmp : (s.c.ports[qp].name, #[(pubRow P.salt P.position P.roots l.val, salts "pub")]) ∈
      knownOf P salts (some l) := by
    rw [hp]; simp [knownOf, knownRows]
  have hpub : ∀ {lo n x : ℕ}, lo + n ≤ 1616 →
      (∀ i < n, (pubBits P.salt P.position P.roots l.val)[lo + i]? = some (x.testBit i)) →
      Holds (portBit s (openedReg s) "pub") lo n x := by
    intro lo n x hn hx
    have h := holds_known hN hK hu hS hcomp hPp.1 hPp.2 hmp (lo := lo) (n := n) (x := x)
      (fun i hi => hrp _ (by omega)) (fun i hi => by
        rw [known_readAt_zero hK hS hcomp hmp]
        show ((rowWords (pubBits P.salt P.position P.roots l.val))[(lo + i) / 16]!).testBit ((lo + i) % 16) = _
        rw [rowWords_testBit _ (by rw [length_pubBits]; omega), hx i hi]
        rfl)
    rwa [hp] at h
  refine ⟨fun i => ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have h := portBit_known hN hK hu hS hcomp hPa.1 hPa.2 hma
      (hra (fieldBit (lanePorts T 0) "admit" + i.val) (by rw [fieldBit_admit, Nat.zero_add]; exact i.isLt))
    rw [ha, known_readAt_zero hK hS hcomp hma] at h
    rw [h, fieldBit_admit, Nat.zero_add]
    show some (((rowWords (List.ofFn P.admit))[i.val / 16]!).testBit (i.val % 16)) = _
    rw [rowWords_testBit _ (by simp), List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
    rfl
  · rw [fieldBit_salt]
    exact hpub (by simp [digestBits]) fun i hi => pubBits_salt _ _ _ _ hi
  · rw [fieldBit_position]
    exact hpub (by simp [positionBits]) fun i hi => pubBits_position _ _ _ _ hi
  · rw [fieldBit_roots]
    exact hpub (by simp [digestBits]) fun i hi => pubBits_roots _ _ _ _ hi
  · rw [fieldBit_lane]
    exact hpub (by simp [laneBits]) fun i hi => pubBits_lane _ _ _ _ hi
  · exact ok_of_known hN hK hu hS hcomp ho hPo (salt := salts "o0") (by simp [knownOf, knownRows]) hro

/-- **The combine reads `ok = 1`** where its known rows hold at `knownOf P salts none`, its `o0` a known value's one
row whose bit 0 is read. -/
theorem combKnownOk (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {P : LanePub} {salts : String → ByteArray} (hK : KnownHeld s (knownOf P salts none)) {u : Fin s.n}
    (hu : u.val = 0) (hS : s.ServedRegistered u) (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ())
    {qo : Fin s.c.ports.size} (ho : s.c.ports[qo].name = "o0") (hPo : KnownPort s qo) (hro : RowBitRead s u qo 0) :
    portBit s (openedReg s) "o0" 0 = some true :=
  ok_of_known hN hK hu hS hcomp ho hPo (salt := salts "o0") (by simp [knownOf, knownRows]) hro

end Accounting.TS.Lanes
