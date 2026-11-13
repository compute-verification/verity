module

public import Properties.Pouw.PearlPlus
public import Definitions.Pouw.PearlC.TileCheck
public import Proofs.Properties.Pouw.PearlPlus.Open
public import Proofs.Pouw.PearlPlus.Semantics

/-!
# Pearl+'s bridge, part (d): a passed tile is good in the game

`TileAcceptMeans` says what `checkTile` establishes about a tile that passes, or is only over its inert-work bound, in
the code's terms: its openings, the words its replay forms, its leaf and its rows' rules. This part reads it in the
game's: a tile of a loaded pass that the check passes or finds only over the bound, whose openings are its unit's commitment, has `TileWordsGood` at the deployed semantics, or SHA-512 collides on
two different messages, one hashed in digesting the words the check forms and one in digesting the commitment's words on
the tile (the leaf binds the tile's words through `sha512/row-seg/v1`). The check refuses a tile whose unit is outside its
scheme's domain, `k ≤ K_MAX` among its bounds, which `WordsAreTheSemantics` reads. `TileGood` adds the cap and every weight row's rules, which wait on
the restated credit set and acceptance event (`Rules.lean`).
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- Tile `tid` of pass `p` is audit tile `g` of `U`'s layout: the same unit, shape and weight, rows and columns. -/
def TileOf (p : Pass) (tid : ℕ) (U : Pouw.PearlC.Workload) (g : ℕ) : Prop :=
  ∃ pos rows cols, p.tile? tid = some (pos, rows, cols) ∧
    g < (auditTiling 64 64 U.layout).NT ∧
    (auditTiling 64 64 U.layout).unit g = p.records[pos]!.matmul.index ∧
    (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m = p.records[pos]!.matmul.m ∧
    (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k = p.records[pos]!.matmul.k ∧
    (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n = p.records[pos]!.matmul.n ∧
    U.layout.fixedId ((auditTiling 64 64 U.layout).unit g) = p.records[pos]!.matmul.weight ∧
    (auditTiling 64 64 U.layout).rows g = rows.toFinset ∧ (auditTiling 64 64 U.layout).cols g = cols.toFinset

/-- Commitment `c`'s ticket and useful words on a tile's rows and columns, row by row, as the check's `Checked`. -/
def commitWords (c : Pouw.PearlC.Commit) (rows cols : List ℕ) : Words.Checked :=
  ⟨(rows.map fun i => (cols.map fun j => c.ticket i j).toArray).toArray,
    (rows.map fun i => (cols.map fun j => c.useful i j).toArray).toArray⟩

/-- The bytes `Words.leaf` digests: per row, each 32-column chunk's C̃ words, then its U words, little-endian. -/
def leafBytes (ch : Words.Checked) : ByteArray :=
  (ch.c.zip ch.u).foldl (fun acc (crow, urow) =>
    (List.range ((crow.size + Words.CHUNK - 1) / Words.CHUNK)).foldl (fun acc h =>
      acc ++ Words.le32s (crow.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK)) ++
        Words.le32s (urow.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK)))
      acc) ByteArray.empty

/-- The messages SHA-512 hashes in digesting the words the check forms for tile `tid` from the openings `o` (none where
it forms none). -/
def leafHashed (p : Pass) (tid : ℕ) (o : TileOpening) : List ByteArray :=
  match p.tile? tid with
  | some (pos, rows, cols) =>
    match Properties.Pouw.PearlPlus.formed p (Properties.Pouw.PearlPlus.noiseOf p tid) rows cols p.records[pos]!.matmul
        o with
    | .ok (_, ch) => Properties.Pouw.PearlPlus.segHashed (leafBytes ch)
    | .error _ => []
  | none => []

/-- The messages SHA-512 hashes in digesting commitment `c`'s words on tile `tid`. -/
def commitHashed (p : Pass) (tid : ℕ) (c : Pouw.PearlC.Commit) : List ByteArray :=
  match p.tile? tid with
  | some (_, rows, cols) => Properties.Pouw.PearlPlus.segHashed (leafBytes (commitWords c rows cols))
  | none => []

/-- The tile's openings are unit `u`'s commitment `c`: its A rows are `c`'s activation rows and its B rows the weight's
(each word below `2^32`), and its leaf is the digest of `c`'s ticket and useful words on the tile. -/
def OpensCommitted (p : Pass) (tid : ℕ) (o : TileOpening) (U : Pouw.PearlC.Workload) (u : ℕ) (c : Pouw.PearlC.Commit) : Prop :=
  ∀ pos rows cols, p.tile? tid = some (pos, rows, cols) →
    o.a.map (fun x => openedValue x.value) = rows.map (rowBytesOf c.act (U.layout.shape u).k) ∧
    o.b.map (fun x => openedValue x.value) = cols.map (rowBytesOf (U.fixed (U.layout.fixedId u)) (U.layout.shape u).k) ∧
    openedValue o.leaf.value = Words.leaf (commitWords c rows cols) ∧
    (∀ i ∈ rows, ∀ l < (U.layout.shape u).k, c.act i l < 2 ^ 32) ∧
    (∀ j ∈ cols, ∀ l < (U.layout.shape u).k, U.fixed (U.layout.fixedId u) j l < 2 ^ 32) ∧
    ∀ i ∈ rows, ∀ j ∈ cols, c.ticket i j < 2 ^ 32 ∧ c.useful i j < 2 ^ 32

/-- **(d) A passed tile's words are good**: a tile of a pass `load` gives, which is audit tile `g` of `U`, under the
semantics' salt and weight root, whose weight rows all pass the rules (`TileGood`'s second conjunct), whose openings are
its unit's commitment `c`, and that `checkTile` passes on its weight's noise or finds only over its inert-work bound, has
every word of its credited rows at the
reference values, or SHA-512 collides on two different messages, one hashed in digesting the words the check forms and
one in digesting `c`'s words on the tile. -/
def TilePassGivesTheWords : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (ρ : ℚ) (H : Q → R) (s : S)
    (coins : PouwCheck.PearlPlus.EpochCoins) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit),
    PouwCheck.PearlPlus.load coins man = .ok p → TileOf p tid U g → bytesNat p.salt = saltOf s →
    bytesNat ((p.weightRoots.lookup (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) =
      rootOf (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)) →
    (∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf ρ : Pouw.PearlC.Protocol Q R S).fixedOK (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))) j) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c →
    (checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass ∨
      ∃ why, checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .overCap why) →
      TileWordsGood (protocolH3 saltOf rootOf ρ) (tilesH3 saltOf rootOf ρ) U H s g c ∨
        ∃ x ∈ leafHashed p tid o, ∃ y ∈ commitHashed p tid c, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

end Pouw.PearlPlus.Bridge

namespace Pouw.PearlPlus.Assembly

open PouwCheck.PearlPlus

/-- A tile that passes or is only over its inert-work bound is `TileAcceptMeans`' outcome at some bound. -/
theorem passOrOver_bound {t : TileOutcome} (h : t = .pass ∨ ∃ why, t = .overCap why) :
    ∃ bound : Option String, t = match bound with | none => .pass | some why => .overCap why := by
  rcases h with h | ⟨why, h⟩
  exacts [⟨none, h⟩, ⟨some why, h⟩]

/-! ## `Except`, `mapM` and `for` -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem list_mapM_ok {α β ε : Type} [Inhabited α] [Inhabited β] (f : α → Except ε β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys → ys.length = l.length ∧ ∀ n < l.length, f l[n]! = .ok ys[n]!
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; simp
  | x :: l, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨y, hx, h⟩ := bind_ok h
    obtain ⟨cs, hl, h⟩ := bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨hlen, hget⟩ := list_mapM_ok f l cs hl
    refine ⟨by simp [hlen], fun n hn => ?_⟩
    cases n with
    | zero => simpa using hx
    | succ n =>
      have := hget n (by simp at hn; omega)
      simpa using this

theorem array_mapM_ok {α β ε : Type} [Inhabited α] [Inhabited β] (f : α → Except ε β) (xs : Array α)
    (ys : Array β) (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ n < xs.size, f xs[n]! = .ok ys[n]! := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hl : xs.toList.mapM f with e | l <;> simp only [hl, Functor.map, Except.map] at h
  · cases h
  simp only [Except.ok.injEq] at h
  subst h
  obtain ⟨hlen, hget⟩ := list_mapM_ok f _ l hl
  refine ⟨by simp [hlen], fun n hn => ?_⟩
  have := hget n (by simpa using hn)
  have e1 : xs.toList[n]! = xs[n]! := by
    rw [getElem!_pos xs.toList n (by simpa using hn), getElem!_pos xs n hn]; simp
  have e2 : l[n]! = l.toArray[n]! := by
    rw [getElem!_pos l n (by rw [hlen]; simpa using hn), getElem!_pos l.toArray n (by simp [hlen]; simpa using hn)]
    simp
  rw [← e1, ← e2]; exact this

theorem forIn_inv {α σ ε : Type} (I : ℕ → σ → Prop) (body : α → σ → Except ε (ForInStep σ))
    (hb : ∀ n a s r, I n s → body a s = .ok r → ∃ t, r = .yield t ∧ I (n + 1) t) :
    ∀ (L : List α) (n : ℕ) (s out : σ), I n s → forIn L s body = .ok out → I (n + L.length) out
  | [], n, s, out, hs, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simpa using hs
  | a :: L, n, s, out, hs, h => by
    rw [List.forIn_cons] at h
    obtain ⟨r, hr, h⟩ := bind_ok h
    obtain ⟨t, rfl, ht⟩ := hb n a s r hs hr
    have := forIn_inv I body hb L (n + 1) t out ht h
    simpa [List.length_cons, Nat.add_assoc, Nat.add_comm 1] using this

theorem mapM_ok_mem {ε α β : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → ∀ b ∈ l', ∃ a ∈ l, f a = .ok b
  | [], l', h, b, hb => by cases h; cases hb
  | a :: l, l', h, b, hb => by
    rw [List.mapM_cons] at h
    obtain ⟨b0, h0, h⟩ := bind_ok h
    obtain ⟨bs, hbs, h⟩ := bind_ok h
    cases h
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨a, List.mem_cons_self, h0⟩
    · obtain ⟨a', ha', h'⟩ := mapM_ok_mem hbs b hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

/-! ## The domain a passed tile keeps -/

theorem domain_k {s : PouwCheck.PearlPlus.Scheme} {u : PouwCheck.PearlPlus.Matmul} (h : s.domainReason u = none) :
    u.k ≤ PouwCheck.PearlPlus.K_MAX ∧ u.k % PouwCheck.PearlPlus.WINDOW = 0 := by
  unfold PouwCheck.PearlPlus.Scheme.domainReason at h
  try dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    exact ⟨hc.1.1.1.1.1.2, hc.1.1.1.1.2⟩
  · cases h

/-- **A tile `checkTile` passes, or finds only over its inert-work bound, is in its scheme's domain**: the check refuses
a unit outside it (`domainReason`) before it reads an opening. -/
theorem tile_domain {p : Pass} {wn} {tid pos : ℕ} {rows cols : List ℕ} {o : TileOpening}
    (ht : p.tile? tid = some (pos, rows, cols))
    (hc : checkTile p wn tid o = .pass ∨ ∃ why, checkTile p wn tid o = .overCap why) :
    p.scheme.domainReason p.records[pos]!.matmul = none := by
  cases hd : p.scheme.domainReason p.records[pos]!.matmul with
  | none => rfl
  | some w =>
    have hf : checkTile p wn tid o = .fail s!"DomainError: {w}" := by
      unfold checkTile
      simp only [ht, hd]
      rfl
    rcases hc with h | ⟨why, h⟩ <;> rw [hf] at h <;> cases h

/-- A tile `checkTile` passes, or finds only over its inert-work bound, has `k ≤ K_MAX` and whole promotion windows
(`128 ∣ k`). -/
theorem record_domain {p : Pass} {wn} {tid pos : ℕ} {rows cols : List ℕ} {o : TileOpening}
    (ht : p.tile? tid = some (pos, rows, cols))
    (hc : checkTile p wn tid o = .pass ∨ ∃ why, checkTile p wn tid o = .overCap why) :
    p.records[pos]!.matmul.k ≤ 2 ^ 16 ∧ p.records[pos]!.matmul.k % 128 = 0 :=
  domain_k (tile_domain ht hc)

/-! ## The shape of the formed words -/

theorem formV1_shape {rows eRows f : Array (Array ℕ)} {out : Words.Formed} (h : Words.formV1 rows eRows f = .ok out) :
    out.codes.size = (rows.zip eRows).size ∧ out.ep.size = (rows.zip eRows).size := by
  unfold Words.formV1 at h
  obtain ⟨r, hr, h⟩ := bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [← Array.forIn_toList] at hr
  have := forIn_inv (fun n (o : Words.Formed) => o.codes.size = n ∧ o.ep.size = n) _ ?_ _ 0 _ _ ⟨rfl, rfl⟩ hr
  · simpa using this
  · intro n a s r hs hr
    obtain ⟨x, e⟩ := a
    obtain ⟨st, -, hr⟩ := bind_ok hr
    obtain ⟨fst, rho, alpha, beta⟩ := st
    obtain ⟨ep, -, hr⟩ := bind_ok hr
    obtain ⟨codes, -, hr⟩ := bind_ok hr
    simp only [pure, Except.pure, Except.ok.injEq] at hr
    subst hr
    exact ⟨_, rfl, by simp [hs.1], by simp [hs.2]⟩

theorem checked_shape {ops : Words.Ops} {ch : Words.Checked} (h : Words.checked ops = .ok ch) :
    ch.c.size = ops.a.codes.size ∧ ch.u.size = ops.a.codes.size ∧
      ∀ x < ops.a.codes.size, ch.c[x]!.size = ops.b.codes.size ∧ ch.u[x]!.size = ops.pb.size := by
  unfold Words.checked at h
  obtain ⟨c, hc, h⟩ := bind_ok h
  obtain ⟨u, hu, h⟩ := bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨hcs, hcg⟩ := array_mapM_ok _ _ _ hc
  obtain ⟨hus, hug⟩ := array_mapM_ok _ _ _ hu
  refine ⟨hcs, by simpa [hcs] using hus, fun x hx => ⟨?_, ?_⟩⟩
  · exact (array_mapM_ok _ _ _ (hcg x hx)).1
  · have := (array_mapM_ok _ _ _ (hug x (by simpa [hcs] using hx))).1
    simpa using this

theorem encode_shape {salt rootB : ByteArray} {wn : Words.FixedNoise} {index k : ℕ} {rows : List ℕ}
    {aRows : List ByteArray} {cols : List ℕ} {bRows : List ByteArray} {ops : Words.Ops}
    (ha : aRows.length = rows.length) (hb : bRows.length = cols.length)
    (h : Words.encode salt rootB wn index k rows aRows cols bRows = .ok ops) :
    ops.a.codes.size = rows.length ∧ ops.b.codes.size = cols.length ∧ ops.pb.size = cols.length := by
  unfold Words.encode at h
  obtain ⟨ea, hea, h⟩ := bind_ok h
  obtain ⟨xa, hxa, h⟩ := bind_ok h
  obtain ⟨a, hfa, h⟩ := bind_ok h
  obtain ⟨eb, heb, h⟩ := bind_ok h
  obtain ⟨xb, hxb, h⟩ := bind_ok h
  obtain ⟨b, hfb, h⟩ := bind_ok h
  obtain ⟨pa, -, h⟩ := bind_ok h
  obtain ⟨pb, hpb, h⟩ := bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  unfold Words.peelB at hpb
  have sea := (array_mapM_ok _ _ _ hea).1
  have sxa := (array_mapM_ok _ _ _ hxa).1
  have seb := (array_mapM_ok _ _ _ heb).1
  have sxb := (array_mapM_ok _ _ _ hxb).1
  have spb := (array_mapM_ok _ _ _ hpb).1
  obtain ⟨fa1, -⟩ := formV1_shape hfa
  obtain ⟨fb1, fb2⟩ := formV1_shape hfb
  simp only [Array.size_zip, List.size_toArray, List.length_zip] at sea sxa seb sxb spb fa1 fb1 fb2
  try dsimp only
  refine ⟨by omega, by omega, by omega⟩

/-- The check's words of a formed tile: its noise, operands and checked words, as `formed` computes them. -/
theorem formed_ok {p : Pass} {wn : Words.Reject Words.FixedNoise} {rows cols : List ℕ}
    {u : PouwCheck.PearlPlus.Matmul} {o : TileOpening} {ops : Words.Ops} {ch : Words.Checked}
    (h : Properties.Pouw.PearlPlus.formed p wn rows cols u o = .ok (ops, ch)) :
    ∃ wn', wn = .ok wn' ∧
      Words.encode p.salt ((p.weightRoots.lookup u.weight).getD ByteArray.empty) wn' u.index u.k rows
        (o.a.map (fun x => openedValue x.value)) cols (o.b.map (fun x => openedValue x.value)) = .ok ops ∧
      Words.checked ops = .ok ch := by
  unfold Properties.Pouw.PearlPlus.formed at h
  obtain ⟨wn', hwn, h⟩ := bind_ok h
  obtain ⟨ops', hops, h⟩ := bind_ok h
  obtain ⟨ch', hch, h⟩ := bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact ⟨wn', hwn, hops, hch⟩

theorem noiseOf_eq {p : Pass} {tid pos : ℕ} {rows cols : List ℕ} (h : p.tile? tid = some (pos, rows, cols)) :
    Properties.Pouw.PearlPlus.noiseOf p tid =
      Words.fixedNoise p.salt ((p.weightRoots.lookup p.records[pos]!.matmul.weight).getD ByteArray.empty)
        p.records[pos]!.matmul.k := by
  simp only [Properties.Pouw.PearlPlus.noiseOf, h]

theorem leafHashed_eq {p : Pass} {tid pos : ℕ} {rows cols : List ℕ} {o : TileOpening} {ops : Words.Ops}
    {ch : Words.Checked} (h : p.tile? tid = some (pos, rows, cols))
    (hf : Properties.Pouw.PearlPlus.formed p (Properties.Pouw.PearlPlus.noiseOf p tid) rows cols p.records[pos]!.matmul o =
      .ok (ops, ch)) :
    Bridge.leafHashed p tid o = Properties.Pouw.PearlPlus.segHashed (Bridge.leafBytes ch) := by
  simp only [Bridge.leafHashed, h, hf]

theorem commitHashed_eq {p : Pass} {tid pos : ℕ} {rows cols : List ℕ} {c : Pouw.PearlC.Commit}
    (h : p.tile? tid = some (pos, rows, cols)) :
    Bridge.commitHashed p tid c = Properties.Pouw.PearlPlus.segHashed (Bridge.leafBytes (Bridge.commitWords c rows cols)) := by
  simp only [Bridge.commitHashed, h]

/-! ## The leaf's bytes -/

/-- A word's four little-endian bytes, as `le32s` writes them. -/
def bytes4 (x : ℕ) : List UInt8 := (Words.le 4 x).data.toList

theorem bytes4_eq (x : ℕ) : bytes4 x = (List.range 4).map fun i => (x / 256 ^ i % 256).toUInt8 := by
  simp [bytes4, Words.le]

theorem bytes4_length (x : ℕ) : (bytes4 x).length = 4 := by simp [bytes4_eq]

theorem toUInt8_inj {a b : ℕ} (ha : a < 256) (hb : b < 256) (h : a.toUInt8 = b.toUInt8) : a = b := by
  have := congrArg UInt8.toNat h
  simp only [Nat.toUInt8_eq, UInt8.toNat_ofNat'] at this
  omega

theorem bytes4_inj {x y : ℕ} (h : bytes4 x = bytes4 y) : x % 2 ^ 32 = y % 2 ^ 32 := by
  rw [bytes4_eq, bytes4_eq, show List.range 4 = [0, 1, 2, 3] from rfl] at h
  simp only [List.map_cons, List.map_nil, List.cons.injEq] at h
  obtain ⟨h0, h1, h2, h3, -⟩ := h
  have e0 := toUInt8_inj (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide)) h0
  have e1 := toUInt8_inj (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide)) h1
  have e2 := toUInt8_inj (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide)) h2
  have e3 := toUInt8_inj (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide)) h3
  try simp only [Nat.pow_zero, Nat.div_one] at e0
  try simp only [Nat.pow_one] at e1
  try simp only [show (256 : ℕ) ^ 2 = 65536 from rfl] at e2
  try simp only [show (256 : ℕ) ^ 3 = 16777216 from rfl] at e3
  show x % 4294967296 = y % 4294967296
  omega

theorem length_flatMap_bytes4 : ∀ (L : List ℕ), (L.flatMap bytes4).length = 4 * L.length
  | [] => rfl
  | x :: L => by
    rw [List.flatMap_cons, List.length_append, bytes4_length, length_flatMap_bytes4 L, List.length_cons]
    omega

theorem foldl_le_data : ∀ (L : List ℕ) (a : ByteArray),
    (L.foldl (fun acc x => acc ++ Words.le 4 x) a).data.toList = a.data.toList ++ L.flatMap bytes4
  | [], a => by simp
  | x :: L, a => by
    rw [List.foldl_cons, foldl_le_data L]
    simp [bytes4, ByteArray.data_append, List.append_assoc]

theorem le32s_data (xs : Array ℕ) : (Words.le32s xs).data.toList = xs.toList.flatMap bytes4 := by
  unfold Words.le32s
  rw [← Array.foldl_toList, foldl_le_data]
  rfl

/-- One 32-column chunk of a row's bytes: its C̃ words, then its U words. -/
def chunkL (crow urow : Array ℕ) (h : ℕ) : List UInt8 :=
  (crow.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK)).toList.flatMap bytes4 ++
    (urow.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK)).toList.flatMap bytes4

/-- One row's bytes: its chunks in order. -/
def rowL (crow urow : Array ℕ) : List UInt8 :=
  (List.range ((crow.size + Words.CHUNK - 1) / Words.CHUNK)).flatMap (chunkL crow urow)

theorem inner_data (crow urow : Array ℕ) : ∀ (L : List ℕ) (a : ByteArray),
    (L.foldl (fun acc h => acc ++ Words.le32s (crow.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK)) ++
      Words.le32s (urow.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK))) a).data.toList =
        a.data.toList ++ L.flatMap (chunkL crow urow)
  | [], a => by simp
  | h :: L, a => by
    rw [List.foldl_cons, inner_data crow urow L]
    simp [chunkL, ByteArray.data_append, le32s_data, List.append_assoc]

theorem outer_data : ∀ (L : List (Array ℕ × Array ℕ)) (a : ByteArray),
    (L.foldl (fun acc (p : Array ℕ × Array ℕ) =>
      (List.range ((p.1.size + Words.CHUNK - 1) / Words.CHUNK)).foldl (fun acc h =>
        acc ++ Words.le32s (p.1.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK)) ++
          Words.le32s (p.2.extract (Words.CHUNK * h) (Words.CHUNK * h + Words.CHUNK)))
        acc) a).data.toList = a.data.toList ++ L.flatMap fun p => rowL p.1 p.2
  | [], a => by simp
  | p :: L, a => by
    rw [List.foldl_cons, outer_data L, inner_data, List.flatMap_cons, List.append_assoc]
    rfl

theorem leafBytes_data (ch : Words.Checked) :
    (Bridge.leafBytes ch).data.toList = (ch.c.zip ch.u).toList.flatMap fun p => rowL p.1 p.2 := by
  unfold Bridge.leafBytes
  rw [← Array.foldl_toList]
  exact outer_data _ _

theorem flatMap_inj {α β γ : Type} {F : α → List γ} {G : β → List γ} :
    ∀ (L : List α) (M : List β), L.length = M.length →
      (∀ i (h₁ : i < L.length) (h₂ : i < M.length), (F L[i]).length = (G M[i]).length) →
      L.flatMap F = M.flatMap G → ∀ i (h₁ : i < L.length) (h₂ : i < M.length), F L[i] = G M[i]
  | [], _, _, _, _, i, h₁, _ => absurd h₁ (Nat.not_lt_zero _)
  | _ :: _, [], hl, _, _, _, _, _ => by simp at hl
  | a :: L, b :: M, hl, hlen, h, i, h₁, h₂ => by
    rw [List.flatMap_cons, List.flatMap_cons] at h
    obtain ⟨e1, e2⟩ := List.append_inj h (hlen 0 (by simp) (by simp))
    cases i with
    | zero => simpa using e1
    | succ i =>
      have := flatMap_inj L M (by simpa using hl)
        (fun j h₁ h₂ => by
          have := hlen (j + 1) (by simpa using h₁) (by simpa using h₂)
          simpa only [List.getElem_cons_succ] using this) e2 i
        (by simpa using h₁) (by simpa using h₂)
      simpa using this

theorem flatMap_length_eq {α β γ : Type} {F : α → List γ} {G : β → List γ} :
    ∀ (L : List α) (M : List β), L.length = M.length →
      (∀ i (h₁ : i < L.length) (h₂ : i < M.length), (F L[i]).length = (G M[i]).length) →
      (L.flatMap F).length = (M.flatMap G).length
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl
  | a :: L, b :: M, hl, hlen => by
    have h0 := hlen 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    rw [List.flatMap_cons, List.flatMap_cons, List.length_append, List.length_append, h0,
      flatMap_length_eq L M (by simpa using hl)
        (fun j h₁ h₂ => by
          have := hlen (j + 1) (by simpa using h₁) (by simpa using h₂)
          simpa only [List.getElem_cons_succ] using this)]

theorem chunkL_length (crow urow : Array ℕ) (h : ℕ) : (chunkL crow urow h).length =
    4 * (min (Words.CHUNK * h + Words.CHUNK) crow.size - Words.CHUNK * h) +
      4 * (min (Words.CHUNK * h + Words.CHUNK) urow.size - Words.CHUNK * h) := by
  simp only [chunkL, List.length_append, length_flatMap_bytes4, Array.length_toList, Array.size_extract]

theorem rowL_length_eq {c₁ u₁ c₂ u₂ : Array ℕ} (hc : c₁.size = c₂.size) (hu : u₁.size = u₂.size) :
    (rowL c₁ u₁).length = (rowL c₂ u₂).length := by
  unfold rowL
  rw [hc]
  exact flatMap_length_eq _ _ rfl fun i _ _ => by rw [chunkL_length, chunkL_length, hc, hu]

/-- Two rows' bytes of one shape are equal only where their words agree below `2^32`. -/
theorem rowL_inj {c₁ u₁ c₂ u₂ : Array ℕ} {C : ℕ} (h1c : c₁.size = C) (h1u : u₁.size = C) (h2c : c₂.size = C)
    (h2u : u₂.size = C) (h : rowL c₁ u₁ = rowL c₂ u₂) :
    ∀ z < C, c₁[z]! % 2 ^ 32 = c₂[z]! % 2 ^ 32 ∧ u₁[z]! % 2 ^ 32 = u₂[z]! % 2 ^ 32 := by
  intro z hz
  unfold rowL at h
  rw [h1c, h2c] at h
  have hz' : z / Words.CHUNK < ((C + Words.CHUNK - 1) / Words.CHUNK) := by unfold Words.CHUNK; omega
  have hch := flatMap_inj _ _ rfl (fun i _ _ => by rw [chunkL_length, chunkL_length, h1c, h1u, h2c, h2u]) h
    (z / Words.CHUNK) (by simpa using hz') (by simpa using hz')
  simp only [List.getElem_range] at hch
  unfold chunkL at hch
  have hlen : ∀ (a b : Array ℕ), a.size = b.size →
      ((a.extract (Words.CHUNK * (z / Words.CHUNK)) (Words.CHUNK * (z / Words.CHUNK) + Words.CHUNK)).toList.flatMap
        bytes4).length =
      ((b.extract (Words.CHUNK * (z / Words.CHUNK)) (Words.CHUNK * (z / Words.CHUNK) + Words.CHUNK)).toList.flatMap
        bytes4).length := by
    intro a b hab
    simp only [length_flatMap_bytes4, Array.length_toList, Array.size_extract, hab]
  obtain ⟨hc, hu⟩ := List.append_inj hch (hlen _ _ (by rw [h1c, h2c]))
  have key : ∀ (a b : Array ℕ), a.size = C → b.size = C →
      (a.extract (Words.CHUNK * (z / Words.CHUNK)) (Words.CHUNK * (z / Words.CHUNK) + Words.CHUNK)).toList.flatMap bytes4 =
        (b.extract (Words.CHUNK * (z / Words.CHUNK)) (Words.CHUNK * (z / Words.CHUNK) + Words.CHUNK)).toList.flatMap
          bytes4 →
      a[z]! % 2 ^ 32 = b[z]! % 2 ^ 32 := by
    intro a b ha hb he
    have hta : z % Words.CHUNK <
        (a.extract (Words.CHUNK * (z / Words.CHUNK)) (Words.CHUNK * (z / Words.CHUNK) + Words.CHUNK)).toList.length := by
      simp only [Array.length_toList, Array.size_extract, ha]; unfold Words.CHUNK; omega
    have htb : z % Words.CHUNK <
        (b.extract (Words.CHUNK * (z / Words.CHUNK)) (Words.CHUNK * (z / Words.CHUNK) + Words.CHUNK)).toList.length := by
      simp only [Array.length_toList, Array.size_extract, hb]; unfold Words.CHUNK; omega
    have := flatMap_inj _ _ (by simp [ha, hb]) (fun i _ _ => by rw [bytes4_length, bytes4_length]) he (z % Words.CHUNK)
      hta htb
    have := bytes4_inj this
    simp only [Array.getElem_toList, Array.getElem_extract] at this
    rw [getElem!_pos a z (by omega), getElem!_pos b z (by omega)]
    have hz2 : Words.CHUNK * (z / Words.CHUNK) + z % Words.CHUNK = z := Nat.div_add_mod z Words.CHUNK
    simp only [hz2] at this
    exact this
  exact ⟨key _ _ h1c h2c hc, key _ _ h1u h2u hu⟩

theorem zip_get {ch : Words.Checked} {R : ℕ} (hc : ch.c.size = R) (hu : ch.u.size = R) (x : ℕ)
    (h : x < (ch.c.zip ch.u).toList.length) : (ch.c.zip ch.u).toList[x] = (ch.c[x]!, ch.u[x]!) := by
  have hx : x < R := by simp [hc, hu] at h; omega
  simp [Array.getElem_toList, getElem!_pos ch.c x (by omega), getElem!_pos ch.u x (by omega)]

theorem leafBytes_shape_len {ch₁ ch₂ : Words.Checked} {R C : ℕ}
    (h1c : ch₁.c.size = R) (h1u : ch₁.u.size = R) (h2c : ch₂.c.size = R) (h2u : ch₂.u.size = R)
    (h1 : ∀ x < R, ch₁.c[x]!.size = C ∧ ch₁.u[x]!.size = C) (h2 : ∀ x < R, ch₂.c[x]!.size = C ∧ ch₂.u[x]!.size = C) :
    ∀ i (h₁ : i < (ch₁.c.zip ch₁.u).toList.length) (h₂ : i < (ch₂.c.zip ch₂.u).toList.length),
      (rowL (ch₁.c.zip ch₁.u).toList[i].1 (ch₁.c.zip ch₁.u).toList[i].2).length =
        (rowL (ch₂.c.zip ch₂.u).toList[i].1 (ch₂.c.zip ch₂.u).toList[i].2).length := by
  intro i h₁ h₂
  have hi : i < R := by simp [h1c, h1u] at h₁; omega
  rw [zip_get h1c h1u i h₁, zip_get h2c h2u i h₂]
  exact rowL_length_eq (by rw [(h1 i hi).1, (h2 i hi).1]) (by rw [(h1 i hi).2, (h2 i hi).2])

/-- Two tiles' words of one shape give leaf bytes of one length. -/
theorem leafBytes_size {ch₁ ch₂ : Words.Checked} {R C : ℕ}
    (h1c : ch₁.c.size = R) (h1u : ch₁.u.size = R) (h2c : ch₂.c.size = R) (h2u : ch₂.u.size = R)
    (h1 : ∀ x < R, ch₁.c[x]!.size = C ∧ ch₁.u[x]!.size = C) (h2 : ∀ x < R, ch₂.c[x]!.size = C ∧ ch₂.u[x]!.size = C) :
    (Bridge.leafBytes ch₁).size = (Bridge.leafBytes ch₂).size := by
  have e : (Bridge.leafBytes ch₁).data.toList.length = (Bridge.leafBytes ch₂).data.toList.length := by
    rw [leafBytes_data, leafBytes_data]
    exact flatMap_length_eq _ _ (by simp [h1c, h1u, h2c, h2u]) (leafBytes_shape_len h1c h1u h2c h2u h1 h2)
  simpa using e

/-- **The leaf's bytes fix the words**: two tiles' words of one shape with one leaf byte string agree below `2^32`. -/
theorem leafBytes_words {ch₁ ch₂ : Words.Checked} {R C : ℕ}
    (h1c : ch₁.c.size = R) (h1u : ch₁.u.size = R) (h2c : ch₂.c.size = R) (h2u : ch₂.u.size = R)
    (h1 : ∀ x < R, ch₁.c[x]!.size = C ∧ ch₁.u[x]!.size = C) (h2 : ∀ x < R, ch₂.c[x]!.size = C ∧ ch₂.u[x]!.size = C)
    (he : Bridge.leafBytes ch₁ = Bridge.leafBytes ch₂) :
    ∀ x < R, ∀ z < C, (ch₁.c[x]!)[z]! % 2 ^ 32 = (ch₂.c[x]!)[z]! % 2 ^ 32 ∧
      (ch₁.u[x]!)[z]! % 2 ^ 32 = (ch₂.u[x]!)[z]! % 2 ^ 32 := by
  intro x hx
  have he := congrArg (fun b : ByteArray => b.data.toList) he
  simp only [leafBytes_data] at he
  have hx1 : x < (ch₁.c.zip ch₁.u).toList.length := by simp [h1c, h1u]; omega
  have hx2 : x < (ch₂.c.zip ch₂.u).toList.length := by simp [h2c, h2u]; omega
  have := flatMap_inj _ _ (by simp [h1c, h1u, h2c, h2u]) (leafBytes_shape_len h1c h1u h2c h2u h1 h2) he x hx1 hx2
  rw [zip_get h1c h1u x hx1, zip_get h2c h2u x hx2] at this
  exact rowL_inj (h1 x hx).1 (h1 x hx).2 (h2 x hx).1 (h2 x hx).2 this

theorem leaf_eq (ch : Words.Checked) :
    Words.leaf ch = Flock.HmRow.segDigest (8 * (Bridge.leafBytes ch).size) (Bridge.leafBytes ch) := rfl

/-- **`sha512/row-seg/v1` binds a row of a known length**: two rows of one length with one digest are one row, or two
different messages hashed in computing the two digests have one SHA-512 digest. -/
theorem segDigest_binds_eq (row row' : ByteArray) (hN : row.size = row'.size)
    (h : Flock.HmRow.segDigest (8 * row.size) row = Flock.HmRow.segDigest (8 * row'.size) row') :
    row = row' ∨ Properties.Pouw.Commit.Break (Properties.Pouw.PearlPlus.segHashed row)
      (Properties.Pouw.PearlPlus.segHashed row') := by
  rw [Properties.Pouw.PearlPlus.segHashed_eq, Properties.Pouw.PearlPlus.segHashed_eq]
  rw [Properties.Pouw.PearlPlus.segDigest_out, Properties.Pouw.PearlPlus.segDigest_out] at h
  rcases Properties.Pouw.Commit.step (List.mem_append_right _ (List.mem_singleton_self _))
    (List.mem_append_right _ (List.mem_singleton_self _)) h with e | b
  case inr => exact .inr b
  have e := congrArg (fun b : ByteArray => b.data.toList) e
  simp only [Properties.Pouw.PearlPlus.segOut_data] at e
  obtain ⟨-, hd⟩ := List.append_inj e
    (by rw [Properties.Pouw.PearlPlus.rowPrefixSeg_length, Properties.Pouw.PearlPlus.rowPrefixSeg_length])
  rw [← hN] at hd ⊢
  have h64 : ∀ m, (Flock.Sha512.hash m).data.toList.length = 64 := fun m => by
    rw [Array.length_toList]; exact Properties.Pouw.Commit.sha512_size m
  have hblk := Properties.Pouw.PearlPlus.flatMap_range_inj (fun j => h64 _) (fun j => h64 _) _ hd
  by_cases hall : ∀ j < (Flock.HmRow.segShape (8 * row.size)).1.size,
      Properties.Pouw.PearlPlus.segIn (8 * row.size) row j = Properties.Pouw.PearlPlus.segIn (8 * row.size) row' j
  · exact .inl (Properties.Pouw.PearlPlus.row_eq_of_segIn hN hall)
  · obtain ⟨j, hj, hne⟩ : ∃ j, j < (Flock.HmRow.segShape (8 * row.size)).1.size ∧
        Properties.Pouw.PearlPlus.segIn (8 * row.size) row j ≠ Properties.Pouw.PearlPlus.segIn (8 * row.size) row' j :=
      Classical.byContradiction fun hc => hall fun j hj => Classical.byContradiction fun hne => hc ⟨j, hj, hne⟩
    exact .inr ⟨_, List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩),
      _, List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩), hne,
      ByteArray.ext (Array.toList_inj.1 (hblk j hj))⟩

/-! ## The commitment's words -/

theorem toArray_map_get! {α : Type} [Inhabited α] (f : ℕ → α) {l : List ℕ} {x : ℕ} (hx : x < l.length) :
    (l.map f).toArray[x]! = f l[x] := by
  rw [getElem!_pos (l.map f).toArray x (by simpa using hx)]
  all_goals simp only [List.getElem_toArray, List.getElem_map]

theorem commitWords_shape (c : Pouw.PearlC.Commit) (rows cols : List ℕ) :
    (Bridge.commitWords c rows cols).c.size = rows.length ∧ (Bridge.commitWords c rows cols).u.size = rows.length ∧
      ∀ x < rows.length, (Bridge.commitWords c rows cols).c[x]!.size = cols.length ∧
        (Bridge.commitWords c rows cols).u[x]!.size = cols.length := by
  refine ⟨by simp [Bridge.commitWords], by simp [Bridge.commitWords], fun x hx => ?_⟩
  simp only [Bridge.commitWords]
  rw [toArray_map_get! (α := Array ℕ) _ hx, toArray_map_get! (α := Array ℕ) _ hx]
  simp

theorem commitWords_get (c : Pouw.PearlC.Commit) {rows cols : List ℕ} {x z : ℕ} (hx : x < rows.length)
    (hz : z < cols.length) :
    ((Bridge.commitWords c rows cols).c[x]!)[z]! = c.ticket rows[x] cols[z] ∧
      ((Bridge.commitWords c rows cols).u[x]!)[z]! = c.useful rows[x] cols[z] := by
  simp only [Bridge.commitWords]
  rw [toArray_map_get! (α := Array ℕ) _ hx, toArray_map_get! (α := Array ℕ) _ hx,
    toArray_map_get! (α := ℕ) _ hz, toArray_map_get! (α := ℕ) _ hz]
  exact ⟨rfl, rfl⟩

/-- A word's value reads only its low 32 bits. -/
theorem wordQ_mod (w : ℕ) : Pouw.Fp8Atom.wordQ (w % 2 ^ 32) = Pouw.Fp8Atom.wordQ w := by
  have he : ((w % 2 ^ 32) >>> 23) % 256 = (w >>> 23) % 256 := by
    rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
    show w % 4294967296 / 8388608 % 256 = w / 8388608 % 256
    omega
  have hm : w % 2 ^ 32 % 2 ^ 23 = w % 2 ^ 23 := by omega
  have hs : ((w % 2 ^ 32) >>> 31) % 2 = (w >>> 31) % 2 := by
    rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
    show w % 4294967296 / 2147483648 % 2 = w / 2147483648 % 2
    omega
  have hv : Pouw.Fp8Atom.fp32Value (w % 2 ^ 32) = Pouw.Fp8Atom.fp32Value w := by
    unfold Pouw.Fp8Atom.fp32Value
    rw [he, hm, hs]
  unfold Pouw.Fp8Atom.wordQ
  rw [hv]

theorem rows_lt {L : Pouw.PearlC.Layout} {g i : ℕ} (h : i ∈ (Pouw.PearlC.auditTiling 64 64 L).rows g) :
    i < (L.shape ((Pouw.PearlC.auditTiling 64 64 L).unit g)).m :=
  lt_of_lt_of_le (Finset.mem_Ico.1 h).2 (min_le_right _ _)

theorem cols_lt {L : Pouw.PearlC.Layout} {g j : ℕ} (h : j ∈ (Pouw.PearlC.auditTiling 64 64 L).cols g) :
    j < (L.shape ((Pouw.PearlC.auditTiling 64 64 L).unit g)).n :=
  lt_of_lt_of_le (Finset.mem_Ico.1 h).2 (min_le_right _ _)

end Pouw.PearlPlus.Assembly

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- **(d)** from what a passed tile establishes (`TileAcceptMeans`) and what its words are (`wordsAreTheSemantics`):
the leaf binds the check's words to the commitment's, or SHA-512 collides on the two digests' messages. -/
theorem tilePassGivesTheWords (hT : Properties.Pouw.PearlPlus.TileAcceptMeans) : TilePassGivesTheWords := by
  intro Q R S saltOf rootOf ρ H s coins man p tid o U g c _ hof hsalt hroot hwOK hopen hpass
  obtain ⟨pos, rows, cols, htile, -, hunit, -, hk, -, hwid, hrows, hcols⟩ := hof
  obtain ⟨bound, hb⟩ := Assembly.passOrOver_bound hpass
  obtain ⟨pos', rows', cols', ops, ch, htile', hrest⟩ :=
    hT p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o bound (by cases bound <;> exact hb)
  simp only [htile, Option.some.injEq, Prod.mk.injEq] at htile'
  obtain ⟨rfl, rfl, rfl⟩ := htile'
  try dsimp only at hrest
  obtain ⟨-, -, -, -, -, -, hform, hleaf, -, -, -⟩ := hrest
  obtain ⟨wn', hwn, henc, hchk⟩ := Assembly.formed_ok hform
  rw [Assembly.noiseOf_eq htile, ← hk, ← hwid] at hwn
  obtain ⟨hoa, hob, hol, hact, hwt, hcw⟩ := hopen pos rows cols htile
  rw [← hunit, ← hk, ← hwid, hoa, hob] at henc
  have hK : (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k ≤ 2 ^ 16 := by
    rw [hk]; exact (Assembly.record_domain htile hpass).1
  obtain ⟨hA, hB, hPB⟩ := Assembly.encode_shape (by simp) (by simp) henc
  obtain ⟨hcs, hus, hrow⟩ := Assembly.checked_shape hchk
  have h1c : ch.c.size = rows.length := by omega
  have h1u : ch.u.size = rows.length := by omega
  have hchS : ∀ x < rows.length, ch.c[x]!.size = cols.length ∧ ch.u[x]!.size = cols.length := fun x hx => by
    obtain ⟨e1, e2⟩ := hrow x (by omega)
    exact ⟨by omega, by omega⟩
  have hcwS := Assembly.commitWords_shape c rows cols
  have hsz := Assembly.leafBytes_size h1c h1u hcwS.1 hcwS.2.1 hchS hcwS.2.2
  have hlf := (Assembly.leaf_eq ch).symm.trans (hleaf.trans (hol.trans (Assembly.leaf_eq _)))
  rcases Assembly.segDigest_binds_eq _ _ hsz hlf with he | ⟨x, hx, y, hy, hne, hxy⟩
  · left
    have hw := Assembly.leafBytes_words h1c h1u hcwS.1 hcwS.2.1 hchS hcwS.2.2 he
    intro i hi hiok j hj
    change i ∈ (auditTiling 64 64 U.layout).rows g at hi
    change j ∈ (auditTiling 64 64 U.layout).cols g at hj
    rw [hrows, List.mem_toFinset] at hi
    rw [hcols, List.mem_toFinset] at hj
    obtain ⟨x, hx, rfl⟩ := List.getElem_of_mem hi
    obtain ⟨z, hz, rfl⟩ := List.getElem_of_mem hj
    have hx! := getElem!_pos rows x hx
    have hz! := getElem!_pos cols z hz
    have hWx := wordsAreTheSemantics saltOf rootOf ρ H s U ((auditTiling 64 64 U.layout).unit g) c.act p.salt
      ((p.weightRoots.lookup (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) wn' rows cols
      ops ch hsalt hroot hK
      (fun i hi => Assembly.rows_lt (by rw [hrows]; exact List.mem_toFinset.2 hi))
      (fun j hj => Assembly.cols_lt (by rw [hcols]; exact List.mem_toFinset.2 hj))
      hact hwt (fun j hj => hwOK j (by rw [hcols]; exact List.mem_toFinset.2 hj)) hwn henc hchk x hx
      (by rw [hx!]; exact hiok) z hz
    rw [hx!, hz!] at hWx
    obtain ⟨hwc, hwu⟩ := hw x hx z hz
    obtain ⟨hcc, hcu⟩ := Assembly.commitWords_get c hx hz
    rw [hcc] at hwc
    rw [hcu] at hwu
    obtain ⟨hlt1, hlt2⟩ := hcw _ (List.getElem_mem hx) _ (List.getElem_mem hz)
    have t1 : wordQ (c.ticket rows[x] cols[z]) = wordQ ((ch.c[x]!)[z]!) := by
      rw [← Assembly.wordQ_mod ((ch.c[x]!)[z]!), hwc, Nat.mod_eq_of_lt hlt1]
    have t2 : wordQ (c.useful rows[x] cols[z]) = wordQ ((ch.u[x]!)[z]!) := by
      rw [← Assembly.wordQ_mod ((ch.u[x]!)[z]!), hwu, Nat.mod_eq_of_lt hlt2]
    exact ⟨t1.trans hWx.1, t2.trans hWx.2⟩
  · right
    refine ⟨x, ?_, y, ?_, hne, hxy⟩
    · rw [Assembly.leafHashed_eq htile hform]; exact hx
    · rw [Assembly.commitHashed_eq htile]; exact hy

end Pouw.PearlPlus.Bridge
