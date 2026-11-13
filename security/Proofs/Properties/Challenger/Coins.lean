module

public import Properties.Challenger
public import Proofs.Properties.Pouw.LiveSalt.Coin
public import Proofs.Properties.Pous.MemoryChallenger.Attest

/-!
# The coins a sample reads are the ones the run released

`challengerStream` reads the segment's openings as its family's coins `start`, `start + 1`, … of the commitment it is
given (`liveReason`), concatenates their coins, and requires their indices to be exactly those the family's roots, as
the run's release log shows them, released after `after`, every one of those roots `after` (`stream_ok`). So when the commitment and the roots are a
run's (`s.commitment`, `releasedRootsOf`), the stream is `Run.released`, the run's own coins at those indices
(`stream_released`), or the fixed finder outputs a break. `drawCoins` reads the stream under the commitment and roots
of the session the ledger opens (`drawCoins_ok`), and that commitment is its record's (`trustedSession_record`), so the
coins a sample reads are any such run's released coins, or a break (`drawn_released`). `coinsReason` runs the same
four checks under the key its evidence names, then holds the bytes read to the start of the stream (`coinsReason_none`). `drawCoins` compares the segment's
label by `jsonStrIs`, a match the kernel reduces, so its check gives the label (`jsonStrIs_eq`).

Each opening `liveReason` accepts climbs its path to the commitment's root (`Coin.reason_ok`). The run's root is
`Flock.FV3.root` over its leaves, the top of `FrameV3Sha512`'s tree (`Tree.lev`) over them under the run's domain
(`root_eq`: frame-v3's integers are `natBytes`, `uint_eq`), and `Registered.climbL` is `Attest`'s `climbRec`
(`climbL_eq`). The device hashes every node of that tree (`node_mem`) and every leaf's strings (`leaf_mem`); the
reader hashes each opening's (`opening_mem`). So under no collision among what the two hash (`CF`), `Attest.climb_bind`
makes the opening's leaf digest the run's leaf at its position, and `Coin.coin_eq` makes its coin the run's coin there
(`coin_of_opening`). A collision among them is a break the finder outputs (`cf_of_not_break`).

`collisionIn_valid`, `collisionIn_ne_none` and `findBreak_valid` are the finder's two outputs, read over
`Properties.Challenger`'s definitions.
-/

@[expose] public section

namespace Properties.Challenger.Coins

open Lean (Json)
open Spot Spot.Challenger
open MemoryChallenger.Properties.SecurityProofs.Tree (CF next base lev le_depth lev_size hash_size)
open MemoryChallenger.Attest.Properties.SecurityProofs (climbRec climb_bind)
open Properties.Pouw.LiveSalt (jsonStrIs_eq)
open Properties.Pouw.LiveSalt.Coin (leafMsg coin_eq reason_ok)

local instance lawfulBEqByteArrayCoins : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-! ## The finder -/

/-- `collisionIn`'s output is two different strings with one SHA-512 digest. -/
theorem collisionIn_valid {xs : List ByteArray} {x y : ByteArray} (h : collisionIn xs = some (x, y)) :
    x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  unfold collisionIn at h
  obtain ⟨⟨x', hx'⟩, hmx, hfs⟩ := List.exists_of_findSome?_eq_some h
  obtain ⟨⟨y', hy'⟩, hfind, he⟩ := Option.map_eq_some_iff.1 hfs
  have hp := List.find?_some hfind
  have hmy := List.mem_of_find?_eq_some hfind
  obtain ⟨a, -, ha⟩ := List.mem_map.1 hmx
  obtain ⟨b, -, hb⟩ := List.mem_map.1 hmy
  simp only [Prod.mk.injEq] at ha hb
  obtain ⟨rfl, rfl⟩ := ha
  obtain ⟨rfl, rfl⟩ := hb
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, beq_eq_false_iff_ne, ne_eq] at hp
  exact ⟨fun e => hp.2 e.symm, hp.1.symm⟩

/-- Two different strings of `xs` with one SHA-512 digest: `collisionIn` outputs a collision. -/
theorem collisionIn_ne_none {xs : List ByteArray} {x y : ByteArray} (hx : x ∈ xs) (hy : y ∈ xs) (hne : x ≠ y)
    (he : Flock.Sha512.hash x = Flock.Sha512.hash y) : collisionIn xs ≠ none := by
  intro h
  unfold collisionIn at h
  have hmx : (x, Flock.Sha512.hash x) ∈ xs.map fun x => (x, Flock.Sha512.hash x) := List.mem_map_of_mem hx
  have hmy : (y, Flock.Sha512.hash y) ∈ xs.map fun x => (x, Flock.Sha512.hash x) := List.mem_map_of_mem hy
  have h1 := List.findSome?_eq_none_iff.1 h _ hmx
  simp only [Option.map_eq_none_iff] at h1
  apply List.find?_eq_none.1 h1 _ hmy
  simp [he, Ne.symm hne]

/-- `findBreak` outputs only breaks. -/
theorem findBreak_valid {w : Finding} (h : findBreak dev key label coins = some w) : w.Valid dev key := by
  unfold findBreak at h
  split at h
  · next m sig hf =>
    cases h
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hp
    refine ⟨hp.1, fun hs => ?_⟩
    have hc : (signedBy dev key.pub).contains m = true := List.contains_iff_mem.2 hs
    rw [hc] at hp
    exact absurd hp.2 (by decide)
  · obtain ⟨⟨x, y⟩, hc, he⟩ := Option.map_eq_some_iff.1 h
    subst he
    exact collisionIn_valid hc

/-- **No break, no collision**: when the finder outputs no break, no two different strings the reader or the device
hashes share a SHA-512 digest. -/
theorem cf_of_not_break (hb : ¬ Break dev key label coins) :
    CF (readHashed key label coins ++ deviceHashed dev key.pub) := by
  intro x hx y hy he
  by_contra hne
  apply hb
  cases hf : findBreak dev key label coins with
  | some w => exact ⟨w, hf, findBreak_valid hf⟩
  | none =>
    exfalso
    unfold findBreak at hf
    split at hf
    · cases hf
    · exact collisionIn_ne_none hx hy hne he (Option.map_eq_none_iff.1 hf)

/-! ## frame-v3's integers, climb and tree -/

/-- frame-v3's `uint` is `natBytes`: minimal big-endian, one zero byte for 0. -/
theorem uint_eq (x : Nat) : Flock.FV3.uint x = FrameV3Sha512.uint x := by
  have hn : (Nat.log2 x + 8) / 8 = max 1 (((if x = 0 then 0 else Nat.log2 x + 1) + 7) / 8) := by
    split
    · subst x; decide
    · omega
  unfold Flock.FV3.uint FrameV3Sha512.uint FlockSoundness.Randomness.natBytes
  simp only [hn]

theorem shift_eq (c : Nat) : c >>> 1 = c / 2 := by rw [Nat.shiftRight_eq_div_pow, Nat.pow_one]

theorem node_eq (d : FrameV3Sha512.Domain) (l i : Nat) (a b : ByteArray) :
    Flock.Registered.node d.id l i a b = d.node l i a b := by
  simp only [Flock.FV3.Sha512.node, FrameV3Sha512.Domain.node, FrameV3Sha512.hash, uint_eq]

/-- `Registered.climbL` is `Attest`'s `climbRec`. -/
theorem climbL_eq (d : FrameV3Sha512.Domain) : ∀ (path : List ByteArray) (cur : ByteArray) (idx l : Nat),
    Flock.Registered.climbL d.id cur idx l path = climbRec d l idx cur path
  | [], _, _, _ => rfl
  | sib :: rest, cur, idx, l => by
    rw [Flock.FV3.Sha512.climbL, climbRec, climbL_eq d rest, shift_eq, node_eq, node_eq]

/-- The node frames `Registered.climbL` hashes are `Attest`'s `climbInputs`. -/
theorem frames_eq (d : FrameV3Sha512.Domain) : ∀ (path : List ByteArray) (cur : ByteArray) (idx l : Nat),
    climbFrames d.id cur idx l path = MemoryChallenger.Attest.Properties.climbInputs d l idx cur path
  | [], _, _, _ => rfl
  | sib :: rest, cur, idx, l => by
    simp only [climbFrames, MemoryChallenger.Attest.Properties.climbInputs, shift_eq, uint_eq]
    split <;> exact congrArg _ (frames_eq d rest _ _ _)

theorem log2ceil_pow (D : Nat) : Flock.log2ceil (2 ^ D) = D := by
  unfold Flock.log2ceil
  rcases D with _ | D
  · simp
  · have h1 : ¬ 2 ^ (D + 1) ≤ 1 := by have := Nat.one_lt_two_pow (n := D + 1) (by omega); omega
    simp only [h1, ↓reduceIte]
    have hne : 2 ^ (D + 1) - 1 ≠ 0 := by omega
    have hp : 2 ^ D ≤ 2 ^ (D + 1) - 1 := by rw [Nat.pow_succ]; have := Nat.one_le_two_pow (n := D); omega
    have hl := (Nat.le_log2 hne).mpr hp
    have hu := (Nat.log2_lt hne).mpr (show 2 ^ (D + 1) - 1 < 2 ^ (D + 1) by omega)
    omega

theorem log2ceil_nextPow2 (n : Nat) : Flock.log2ceil (Flock.nextPow2 n) = FrameV3Sha512.depth n :=
  log2ceil_pow _

theorem fold_next (d : FrameV3Sha512.Domain) (L : Array ByteArray) : ∀ k s,
    (List.range' s k).foldl (fun a l => next d l a) (lev d L s) = lev d L (s + k)
  | 0, _ => rfl
  | k + 1, s => by
    rw [List.range'_succ, List.foldl_cons, show s + (k + 1) = s + 1 + k by omega]
    exact fold_next d L k (s + 1)

/-- **`FV3.root` is the top of `FrameV3Sha512`'s tree** over the same leaves, under the domain `dom`. -/
theorem root_eq (dom : ByteArray) (n : Nat) (L : Array ByteArray) :
    Flock.FV3.root dom L Flock.Sha512.hash = (lev ⟨dom, n⟩ L (FrameV3Sha512.depth L.size))[0]! := by
  unfold Flock.FV3.root
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind, uint_eq]
  have hs : [:Flock.log2ceil (Flock.nextPow2 L.size)].size = FrameV3Sha512.depth L.size := by
    simp [Std.Legacy.Range.size, log2ceil_nextPow2]
  rw [hs]
  have h := fold_next ⟨dom, n⟩ L (FrameV3Sha512.depth L.size) 0
  simp only [Nat.zero_add] at h
  show ((List.range' 0 (FrameV3Sha512.depth L.size)).foldl (fun a l => next ⟨dom, n⟩ l a)
    (lev ⟨dom, n⟩ L 0)).getD 0 ByteArray.empty = _
  rw [h]
  have h1 := lev_size ⟨dom, n⟩ L _ le_rfl
  simp only [Nat.sub_self, Nat.pow_zero] at h1
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), getElem!_pos _ 0 (by omega)]
  rfl

/-! ## What the device and the reader hash -/

theorem getElem!_toList (a : Array ByteArray) (i : Nat) : a.toList[i]! = a[i]! := by
  simp only [getElem!_def, Array.getElem?_toList]

/-- The node frames of level `l` of the tree over `L` under `d`. -/
def levFrames (d : FrameV3Sha512.Domain) (L : Array ByteArray) (l : Nat) : List ByteArray :=
  (List.range ((lev d L l).size / 2)).map fun i =>
    MemoryChallenger.Properties.frameInput "node"
      [d.id, FrameV3Sha512.uint l, FrameV3Sha512.uint i, (lev d L l)[2 * i]!, (lev d L l)[2 * i + 1]!]

/-- One step of the device's node fold: the node frames over a level, and the level above. -/
def nodeStep (dom : ByteArray) (acc : List ByteArray × List ByteArray) (depth : Nat) :
    List ByteArray × List ByteArray :=
  (List.map Flock.Sha512.hash (List.map (fun i =>
      fv3Message "node" [dom, Flock.FV3.uint depth, Flock.FV3.uint i, acc.1[2 * i]!, acc.1[2 * i + 1]!])
      (List.range (acc.1.length / 2))),
    acc.2 ++ List.map (fun i =>
      fv3Message "node" [dom, Flock.FV3.uint depth, Flock.FV3.uint i, acc.1[2 * i]!, acc.1[2 * i + 1]!])
      (List.range (acc.1.length / 2)))

/-- The device's node fold over the tree's levels from `s` gives the levels above and their frames. -/
theorem fold_frames (dom : ByteArray) (n : Nat) (L : Array ByteArray) : ∀ k s (acc : List ByteArray),
    (List.range' s k).foldl (nodeStep dom) ((lev ⟨dom, n⟩ L s).toList, acc) =
      ((lev ⟨dom, n⟩ L (s + k)).toList, acc ++ (List.range' s k).flatMap (levFrames ⟨dom, n⟩ L))
  | 0, s, acc => by simp
  | k + 1, s, acc => by
    rw [List.range'_succ, List.foldl_cons]
    have hf : (List.range ((lev ⟨dom, n⟩ L s).toList.length / 2)).map (fun i =>
          fv3Message "node" [dom, Flock.FV3.uint s, Flock.FV3.uint i, (lev ⟨dom, n⟩ L s).toList[2 * i]!,
            (lev ⟨dom, n⟩ L s).toList[2 * i + 1]!]) = levFrames ⟨dom, n⟩ L s := by
      simp only [levFrames, Array.length_toList, getElem!_toList, uint_eq]
      rfl
    have hl : (levFrames ⟨dom, n⟩ L s).map Flock.Sha512.hash = (lev ⟨dom, n⟩ L (s + 1)).toList := by
      simp only [levFrames, List.map_map, lev, next, Array.toList_map, Array.toList_range]
      rfl
    simp only [nodeStep, hf, hl]
    rw [fold_frames dom n L k (s + 1), List.flatMap_cons, List.append_assoc, show s + 1 + k = s + (k + 1) by omega]

theorem nodes_eq (dom : ByteArray) (n : Nat) (L : Array ByteArray) (k : Nat) :
    (List.range k).foldl (nodeStep dom) ((lev ⟨dom, n⟩ L 0).toList, []) =
      ((lev ⟨dom, n⟩ L k).toList, (List.range k).flatMap (levFrames ⟨dom, n⟩ L)) := by
  rw [List.range_eq_range', fold_frames, List.nil_append, Nat.zero_add]

theorem leaves_size (s : Run) (dom : ByteArray) : (s.leaves dom).size = total s.families := by
  simp [Run.leaves]

theorem leaves_width (s : Run) (dom : ByteArray) : ∀ x ∈ s.leaves dom, x.size = 64 := by
  intro x hx
  simp only [Run.leaves, Array.mem_map] at hx
  obtain ⟨p, -, rfl⟩ := hx
  exact hash_size _

theorem leaves_get (s : Run) (dom : ByteArray) {p : Nat} (hp : p < total s.families) :
    (s.leaves dom)[p]! =
      Flock.FV3.leaf dom p s.scheme (liveLeaf s.scheme (s.coins p) (s.salts p)) Flock.Sha512.hash := by
  rw [getElem!_pos _ p (by rw [leaves_size]; exact hp)]
  simp [Run.leaves]

/-- What the device hashes for a run whose commitment is `c` under the domain `dom`. -/
theorem mem_deviceHashed {pk dom : ByteArray} {s : Run} {c : LiveCommitment} {x : ByteArray} (hs : s ∈ dev.runs pk)
    (hd : liveDomain s.scheme s.epoch s.families = .ok dom) (hc : s.commitment = some c)
    (hx : x ∈ ((List.range (total s.families)).flatMap fun p => leafHashed s.scheme dom p (s.coins p) (s.salts p)) ∨
      x ∈ (List.range (FrameV3Sha512.depth (total s.families))).flatMap
        (levFrames ⟨dom, total s.families⟩ (s.leaves dom))) :
    x ∈ deviceHashed dev pk := by
  have hb : (lev ⟨dom, total s.families⟩ (s.leaves dom) 0).toList =
      (s.leaves dom).toList ++ ((List.range (Flock.nextPow2 (total s.families) - total s.families)).map fun k =>
        fv3Message "pad" [dom, Flock.FV3.uint (total s.families + k)]).map Flock.Sha512.hash := by
    simp only [lev, base, leaves_size, Array.toList_append, Array.toList_map, Array.toList_range, List.map_map,
      uint_eq]
    rfl
  have hn := nodes_eq dom (total s.families) (s.leaves dom) (FrameV3Sha512.depth (total s.families))
  rw [hb] at hn
  unfold deviceHashed
  simp only [List.mem_append, List.mem_flatMap]
  refine .inl ⟨s, hs, .inr ?_⟩
  simp only [hd, hc, List.mem_append]
  rw [log2ceil_nextPow2]
  rcases hx with hx | hx
  · exact .inl (.inl (.inl (.inl hx)))
  · refine .inl (.inl (.inr ?_))
    have e := congrArg Prod.snd hn
    dsimp only at e
    rw [← e] at hx
    exact hx

/-- The device hashes every node of its run's tree. -/
theorem node_mem {pk dom : ByteArray} {s : Run} {c : LiveCommitment} (hs : s ∈ dev.runs pk)
    (hd : liveDomain s.scheme s.epoch s.families = .ok dom) (hc : s.commitment = some c) {l i : Nat}
    (hl : l < FrameV3Sha512.depth (s.leaves dom).size)
    (hi : i < (lev ⟨dom, total s.families⟩ (s.leaves dom) l).size / 2) :
    MemoryChallenger.Properties.frameInput "node" [dom, FrameV3Sha512.uint l, FrameV3Sha512.uint i,
      (lev ⟨dom, total s.families⟩ (s.leaves dom) l)[2 * i]!,
      (lev ⟨dom, total s.families⟩ (s.leaves dom) l)[2 * i + 1]!] ∈ deviceHashed dev pk := by
  refine mem_deviceHashed hs hd hc (.inr ?_)
  rw [leaves_size] at hl
  simp only [List.mem_flatMap, List.mem_range]
  exact ⟨l, hl, List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩⟩

/-- The device hashes every leaf's strings. -/
theorem leaf_mem {pk dom : ByteArray} {s : Run} {c : LiveCommitment} (hs : s ∈ dev.runs pk)
    (hd : liveDomain s.scheme s.epoch s.families = .ok dom) (hc : s.commitment = some c) {p : Nat}
    (hp : p < total s.families) {x : ByteArray} (hx : x ∈ leafHashed s.scheme dom p (s.coins p) (s.salts p)) :
    x ∈ deviceHashed dev pk :=
  mem_deviceHashed hs hd hc (.inl (List.mem_flatMap.2 ⟨p, List.mem_range.2 hp, hx⟩))

/-- The reader hashes each opening of the segment that parses, checked as its family's coin `start + i`. -/
theorem opening_mem {r : ChallengerRecord} {start : Nat} {os : Array Json} {doc : Json} {i : Nat} {o : LiveOpening}
    (hr : ChallengerRecord.ofJson (getD (getD coins "session") "record") = .ok r)
    (hst : nat? (getD (getD coins "segment") "start") = some start)
    (hos : getD (getD coins "segment") "openings" = .arr os) (hm : (doc, i) ∈ os.toList.zipIdx)
    (ho : LiveOpening.ofJson doc = .ok o) {x : ByteArray}
    (hx : x ∈ openingHashed r.commitment o label ((start + i : Nat) : Int)) :
    x ∈ readHashed key label coins := by
  unfold readHashed
  simp only [hr, hst, hos, List.mem_append]
  refine .inl (.inl (.inr (List.mem_flatten.2 ⟨_, List.mem_filterMap.2 ⟨(doc, i), hm, ?_⟩, hx⟩)))
  simp [ho, Except.toOption]

theorem climbFrames_eq (dom : ByteArray) : ∀ (path : List ByteArray) (cur : ByteArray) (idx l : Nat),
    Properties.Pouw.LiveSalt.climbFrames dom cur idx l path = climbFrames dom cur idx l path
  | [], _, _, _ => rfl
  | sib :: rest, cur, idx, l => by
    simp only [Properties.Pouw.LiveSalt.climbFrames, climbFrames, climbFrames_eq dom rest]
    rfl

theorem liveHashed_eq (c : LiveCommitment) (dom : ByteArray) (pos : Nat) (o : LiveOpening) :
    Properties.Pouw.LiveSalt.liveHashed c dom pos o = leafHashed c.scheme dom pos o.coin o.salt ++
      climbFrames dom (Flock.FV3.leaf dom pos c.scheme (liveLeaf c.scheme o.coin o.salt) Flock.Sha512.hash) pos 0
        o.path := by
  simp only [Properties.Pouw.LiveSalt.liveHashed, climbFrames_eq]
  rfl

/-- The strings `liveReason` hashes in checking an opening are `openingHashed`'s. -/
theorem liveHashed_mem {c : LiveCommitment} {o : LiveOpening} {index : Int} {dom : ByteArray} {pos : Nat}
    (hd : liveDomain c.scheme c.epoch c.families = .ok dom) (hp : c.position label index = some pos) {x : ByteArray}
    (hx : x ∈ Properties.Pouw.LiveSalt.liveHashed c dom pos o) : x ∈ openingHashed c o label index := by
  unfold openingHashed
  rw [liveHashed_eq] at hx
  simp only [hd, hp, List.mem_append]
  exact .inr (List.mem_append.1 hx)

/-! ## One opening -/

theorem position_lt {c : LiveCommitment} {index : Int} {pos : Nat} (h : c.position label index = some pos) :
    pos < c.count := by
  have key : ∀ (fs : Families) (off : Nat), LiveCommitment.position.go label index fs off = some pos →
      pos < off + total fs := by
    intro fs
    induction fs with
    | nil => intro _ h; cases h
    | cons f rest ih =>
      intro off h
      obtain ⟨name, n⟩ := f
      simp only [LiveCommitment.position.go] at h
      have ht : total ((name, n) :: rest) = n + total rest := by simp [total]
      rw [ht]
      split at h
      · split at h
        · rename_i hb
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
          cases h
          omega
        · cases h
      · have := ih _ h
        omega
  have := key c.families 0 h
  simp only [Nat.zero_add] at this
  exact this

/-- **One opening reads the run's coin**: under no collision among `S`, which holds what the device hashes for the run
`s` whose commitment is `c` and what `liveReason` hashes in checking `o`, an opening `liveReason` accepts as coin
(`label`, `index`) of `c` carries the run's coin at that coin's position. -/
theorem coin_of_opening {pk : ByteArray} {S : List ByteArray} (cf : CF S) {s : Run} {c : LiveCommitment}
    {o : LiveOpening} {index : Int} {after : ByteArray} (hs : s ∈ dev.runs pk) (hsc : s.commitment = some c)
    (hD : ∀ x ∈ deviceHashed dev pk, x ∈ S) (hR : ∀ x ∈ openingHashed c o label index, x ∈ S)
    (h : liveReason c o label index after = .ok none) :
    ∃ pos, c.position label index = some pos ∧ o.coin = s.coins pos := by
  obtain ⟨pos, dom, hp, hd, hlen, hw, hroot⟩ := reason_ok h
  have hsc' := hsc
  unfold Run.commitment at hsc'
  split at hsc'
  · next dom' hd' =>
    cases hsc'
    have hdd : dom = dom' := by
      have := hd.symm.trans hd'
      cases this
      rfl
    subst hdd
    have hpos : pos < total s.families := position_lt hp
    set L := s.leaves dom
    set d : FrameV3Sha512.Domain := ⟨dom, total s.families⟩
    have hL : L.size = total s.families := leaves_size s dom
    have hdg : (Flock.Sha512.hash (leafMsg dom pos ⟨s.scheme, s.epoch, s.families, Flock.FV3.root dom L Flock.Sha512.hash⟩
        o)) = (lev d L 0)[pos]! := by
      refine climb_bind cf (leaves_width s dom) ?_ o.path 0 pos _ ?_ ?_ (hash_size _) hw ?_ ?_
      · intro l i hl hi
        exact hD _ (node_mem hs hd hsc hl hi)
      · rw [Nat.zero_add, hlen, hL]
        rfl
      · rw [Nat.sub_zero, hL]
        exact Nat.lt_of_lt_of_le hpos (le_depth _)
      · intro x hx
        rw [← frames_eq d] at hx
        refine hR x (liveHashed_mem (o := o) hd hp ?_)
        rw [liveHashed_eq]
        exact List.mem_append_right _ hx
      · rw [← climbL_eq d, ← root_eq dom (total s.families) L]
        exact hroot
    have hlev : (lev d L 0)[pos]! = L[pos]! := by
      simp only [lev, base]
      rw [getElem!_pos _ pos (by rw [Array.size_append]; omega), Array.getElem_append_left (by omega),
        ← getElem!_pos]
    rw [hlev, leaves_get s dom hpos] at hdg
    let o' : LiveOpening := { o with coin := s.coins pos, salt := s.salts pos, path := [] }
    rcases coin_eq (o' := o') hdg with e | ⟨x, hx, y, hy, hne, he⟩
    · exact ⟨pos, hp, e⟩
    · refine absurd (cf x (hR x (liveHashed_mem hd hp hx)) y ?_ he) hne
      apply hD
      refine leaf_mem hs hd hsc hpos ?_
      rw [liveHashed_eq] at hy
      simpa [o', climbFrames] using hy
  · cases hsc'

/-! ## The stream -/

theorem forall₂_imp {α β : Type} {R R' : α → β → Prop} : ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' →
    (∀ a ∈ l, ∀ b, R a b → R' a b) → List.Forall₂ R' l l'
  | [], [], .nil, _ => .nil
  | _ :: _, _ :: _, .cons h t, f => .cons (f _ List.mem_cons_self _ h)
      (forall₂_imp t fun a ha b hb => f a (List.mem_cons_of_mem _ ha) b hb)

/-- The loop of `challengerStream` that accepts concatenates the coins its openings carry, each accepted. -/
theorem forIn_ok {P : Json × Nat → LiveOpening → Prop} {f : Json × Nat → ByteArray → Flock.V (ForInStep ByteArray)}
    (hf : ∀ p b r, f p b = .ok r → ∃ o, P p o ∧ r = .yield (b ++ o.coin)) :
    ∀ (ps : List (Json × Nat)) (init out : ByteArray), forIn ps init f = .ok out →
      ∃ os, List.Forall₂ P ps os ∧ out = os.foldl (fun acc o => acc ++ o.coin) init
  | [], init, out, h => by
    simp only [List.forIn_nil] at h
    cases h
    exact ⟨[], .nil, rfl⟩
  | p :: ps, init, out, h => by
    rw [List.forIn_cons] at h
    cases hp : f p init with
    | error e => rw [hp] at h; cases h
    | ok r =>
      rw [hp] at h
      obtain ⟨o, ho, rfl⟩ := hf p init r hp
      obtain ⟨os, hos, hout⟩ := forIn_ok hf ps _ out h
      exact ⟨o :: os, .cons ho hos, hout⟩

/-- **What `challengerStream` reads when it accepts**: the segment's label, start and openings, each opening read as
the family's coin `start + i` and accepted, the stream their coins concatenated, their indices those the roots show
released after `after`, and every root `after`. -/
theorem stream_ok {c : LiveCommitment} {segment : Json} {after ξ : ByteArray} {rs : List ByteArray}
    (h : challengerStream c segment after (some rs) = .ok ξ) :
    ∃ (l : String) (start : Nat) (openings : Array Json) (os : List LiveOpening),
      getD segment "label" = .str l ∧ nat? (getD segment "start") = some start ∧
      getD segment "openings" = .arr openings ∧
      List.Forall₂ (fun (p : Json × Nat) (o : LiveOpening) => LiveOpening.ofJson p.1 = .ok o ∧
        liveReason c o l ((start + p.2 : Nat) : Int) after = .ok none) openings.toList.zipIdx os ∧
      ξ = os.foldl (fun acc o => acc ++ o.coin) .empty ∧
      (rs.zipIdx.filter (·.1 == after)).map (·.2) = List.range' start openings.size ∧
      ∀ r ∈ rs, r = after := by
  unfold challengerStream at h
  simp only [bind, Except.bind] at h
  split at h
  · split at h
    · rename_i l start openings heq
      simp only [Prod.mk.injEq] at heq
      obtain ⟨hl, hst, hop⟩ := heq
      split at h
      · split at h
        · cases h
        · rename_i v hv
          split at h
          · split at h
            · rename_i hall hord
              cases h
              refine (forIn_ok (P := fun (p : Json × Nat) (o : LiveOpening) =>
                LiveOpening.ofJson p.1 = .ok o ∧ liveReason c o l ((start + p.2 : Nat) : Int) after = .ok none)
                ?_ _ _ _ hv).elim fun os ⟨hos, hout⟩ =>
                  ⟨l, start, openings, os, hl, hst, hop, hos, hout, beq_iff_eq.1 hord,
                    fun r hr => beq_iff_eq.1 (List.all_eq_true.1 hall r hr)⟩
              intro ⟨doc, i⟩ b r hr
              simp only at hr
              split at hr
              · cases hr
              · rename_i o ho
                split at hr
                · cases hr
                · rename_i w hw
                  split at hr
                  · cases hr
                  · rename_i hn
                    cases w with
                    | some why => exact absurd rfl (hn why)
                    | none =>
                      cases hr
                      exact ⟨o, ⟨ho, hw⟩, rfl⟩
            · cases h
          · cases h
      · cases h
    · cases h
  · cases h

/-- Coins at the positions of their family's indices from `start + m`, concatenated: `Run.released`'s fold over those
indices. -/
theorem fold_coins {s : Run} {c : LiveCommitment} {start : Nat} :
    ∀ (ds : List Json) (m : Nat) (os : List LiveOpening) (init : ByteArray),
      List.Forall₂ (fun (p : Json × Nat) (o : LiveOpening) => ∃ pos,
        c.position label ((start + p.2 : Nat) : Int) = some pos ∧ o.coin = s.coins pos) (ds.zipIdx m) os →
      os.foldl (fun acc o => acc ++ o.coin) init =
        (List.range' (start + m) ds.length).foldl (fun acc (i : Nat) => match c.position label (i : Int) with
          | some p => acc ++ s.coins p
          | none => acc) init
  | [], _, _, _, h => by cases h; rfl
  | d :: ds, m, os, init, h => by
    rw [List.zipIdx_cons] at h
    cases h with
    | cons ho hrest =>
      obtain ⟨pos, hp, hc⟩ := ho
      simp only at hp
      simp only [List.foldl_cons, List.length_cons, List.range'_succ, hp, hc]
      rw [fold_coins ds (m + 1) _ _ hrest, show start + m + 1 = start + (m + 1) by omega]

/-- `Run.released` over the indices its roots show released after `after`. -/
theorem released_eq {s : Run} {c : LiveCommitment} {roots : Roots} {after : ByteArray} {start n : Nat}
    (hsc : s.commitment = some c) (hroots : releasedRootsOf c s.log = .ok roots)
    (hord : ((((roots.find? (·.1 == label)).map (·.2)).getD []).zipIdx.filter (·.1 == after)).map (·.2) =
      List.range' start n) :
    s.released label after = (List.range' start n).foldl (fun acc (i : Nat) => match c.position label (i : Int) with
      | some p => acc ++ s.coins p
      | none => acc) .empty := by
  rw [← hord, List.foldl_map]
  simp only [Run.released, hsc, hroots]
  congr 1

/-- **The stream is the run's released coins**: when the commitment and roots `challengerStream` reads are the run `s`'s
and the segment is of family `label`, the stream it accepts is the coins `s` released after `after`, or the fixed finder
outputs a break. -/
theorem stream_released {s : Run} {c : LiveCommitment} {roots : Roots} {r : ChallengerRecord} {after ξ : ByteArray}
    (hs : s ∈ dev.runs key.pub) (hsc : s.commitment = some c) (hroots : releasedRootsOf c s.log = .ok roots)
    (hr : ChallengerRecord.ofJson (getD (getD coins "session") "record") = .ok r) (hrc : r.commitment = c)
    (hl : getD (getD coins "segment") "label" = Json.str label)
    (h : challengerStream c (getD coins "segment") after
      (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) :
    ξ = s.released label after ∨ Break dev key label coins := by
  by_cases hb : Break dev key label coins
  · exact .inr hb
  refine .inl ?_
  have cf := cf_of_not_break hb
  obtain ⟨l, start, openings, os, hl', hst, hop, hos, rfl, hord, -⟩ := stream_ok h
  rw [hl] at hl'
  cases hl'
  subst hrc
  rw [released_eq hsc hroots hord]
  have hos' : List.Forall₂ (fun (p : Json × Nat) (o : LiveOpening) => ∃ pos,
      r.commitment.position label ((start + p.2 : Nat) : Int) = some pos ∧ o.coin = s.coins pos)
      (openings.toList.zipIdx 0) os :=
    forall₂_imp hos fun ⟨_, _⟩ hp _ ho => coin_of_opening cf hs hsc (fun _ hx => List.mem_append_right _ hx)
      (fun _ hx => List.mem_append_left _ (opening_mem hr hst hop hp ho.1 hx)) ho.2
  rw [fold_coins openings.toList 0 os .empty hos', Nat.add_zero, Array.length_toList]

/-! ## The sample -/

open Std.Do
open Properties.Pouw.LiveSalt.Ledger (bind_ok of_triple except_spec throw_spec ok_spec mapM_spec)

set_option mvcgen.warning false

theorem triple_of {ε α : Type} {x : Except ε α} {Q : α → Prop} (h : ∀ a, x = .ok a → Q a) :
    ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜Q a⌝⦄ := by
  cases hx : x with
  | error e => exact fun _ => True.intro
  | ok a => exact fun _ => h a hx

theorem recordReason_spec (record : Json) : ⦃⌜True⌝⦄ recordReason key record ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem closeMessage_spec (I : ByteArray) (host : String) (c : LiveCommitment) (session : String) (lines : Nat)
    (chain : ByteArray) (ended : Option String) :
    ⦃⌜True⌝⦄ closeMessage I host c session lines chain ended ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem releasedRootsOf_spec (c : LiveCommitment) (log : List String) :
    ⦃⌜True⌝⦄ releasedRootsOf c log ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem recordOfJson_ok (j : Json) :
    ⦃⌜True⌝⦄ ChallengerRecord.ofJson j ⦃⇓? r => ⌜ChallengerRecord.ofJson j = .ok r⌝⦄ := ok_spec _

set_option maxHeartbeats 1000000 in
/-- `sessionOf` returns its record's commitment. -/
theorem sessionOf_record {record log close : Json} {res : LiveCommitment × Roots × String}
    (h : sessionOf key record log close = .ok res) :
    ∃ r, ChallengerRecord.ofJson record = .ok r ∧ r.commitment = res.1 := by
  refine of_triple (Q := fun res => ∃ r, ChallengerRecord.ofJson record = .ok r ∧ r.commitment = res.1) ?_ h
  clear h
  mvcgen [sessionOf, -Spec.throw_Except, throw_spec, recordReason_spec, recordOfJson_ok, closeMessage_spec,
    mapM_spec, releasedRootsOf_spec]
  all_goals exact ⟨_, by assumption, rfl⟩

theorem sessionOf_record_spec (record log close : Json) :
    ⦃⌜True⌝⦄ sessionOf key record log close
      ⦃⇓? res => ⌜∃ r, ChallengerRecord.ofJson record = .ok r ∧ r.commitment = res.1⌝⦄ :=
  triple_of fun _ => sessionOf_record

set_option maxHeartbeats 1000000 in
/-- `sessionNamed` returns its session record's commitment. -/
theorem sessionNamed_record {a : Audited} {session : Json} {res : LiveCommitment × Roots}
    (h : sessionNamed a key session = .ok res) :
    ∃ r, ChallengerRecord.ofJson (getD session "record") = .ok r ∧ r.commitment = res.1 := by
  refine of_triple
    (Q := fun res => ∃ r, ChallengerRecord.ofJson (getD session "record") = .ok r ∧ r.commitment = res.1) ?_ h
  clear h
  mvcgen [sessionNamed, -Spec.throw_Except, throw_spec, sessionOf_record_spec, recordOfJson_ok, mapM_spec]

/-- `trustedSessionOf` returns its session record's commitment. -/
theorem trustedSession_record {ledger window pins main session : Json} {res : LiveCommitment × Roots}
    (h : trustedSessionOf ledger window key pins main session = .ok res) :
    ∃ r, ChallengerRecord.ofJson (getD session "record") = .ok r ∧ r.commitment = res.1 := by
  unfold trustedSessionOf at h
  obtain ⟨a, -, ha⟩ := bind_ok.1 h
  exact sessionNamed_record ha

/-- What `drawCoins` reads when it accepts: the session the ledger opens on `window` under `key`, a segment of family
`label`, and the stream of its openings under that session's commitment and `label`'s roots. -/
theorem drawCoins_ok {window : String} {after ξ : ByteArray} (h : drawCoins key window label after coins = .ok ξ) :
    ∃ c roots, trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key
        (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots) ∧
      getD (getD coins "segment") "label" = Json.str label ∧
      challengerStream c (getD coins "segment") after
        (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ := by
  unfold drawCoins at h
  simp only [bind, Except.bind] at h
  split at h
  · split at h
    · cases h
    · rename_i v hv
      split at h
      · split at h
        · rename_i hl
          exact ⟨v.1, v.2, hv, jsonStrIs_eq hl, h⟩
        · cases h
      · cases h
  · cases h

/-- What `coinsReason` reads when it gives no reason: the key its evidence names, the session the ledger opens on
`window` under that key, a withheld one, a segment of family `label`, the stream of its openings under that session's
commitment and `label`'s roots, and `read` the start of that stream, ending inside its last coin. `drawCoins_ok`'s
four checks, under the evidence's key. -/
theorem coinsReason_none {window : String} {after read : ByteArray}
    (h : coinsReason coins window label after read = none) :
    ∃ k c roots ξ, ChallengerKey.ofJson (getD coins "key") = .ok k ∧
      trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) k
        (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots) ∧
      withheldOn (getD coins "ledger") window = true ∧
      getD (getD coins "segment") "label" = Json.str label ∧
      challengerStream c (getD coins "segment") after
        (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ ∧
      ξ.extract 0 read.size = read ∧ (read.size + COIN_BYTES - 1) / COIN_BYTES = ξ.size / COIN_BYTES := by
  unfold coinsReason at h
  split at h
  · cases h
  · simp only [] at h
    split at h
    · cases h
    · rename_i c roots heq
      obtain ⟨k, hk, ht⟩ := bind_ok.1 heq
      split at h
      · cases h
      · rename_i hw
        split at h
        · cases h
        · rename_i hl
          unfold readReason at h
          split at h
          · cases h
          · rename_i ξ hst
            split at h
            · cases h
            · rename_i hpre
              split at h
              · cases h
              · rename_i hall
                exact ⟨k, c, roots, ξ, hk, ht, by simpa using hw, jsonStrIs_eq (by simpa using hl), hst,
                  by simpa using hpre, by simpa using hall⟩

/-- **The coins a sample reads are the ones its session released**: when `drawCoins` reads `ξ` from a segment of family
`label`, `ξ` is the coins of `label` released after `after` by any run of the key's device whose commitment and roots
are the session's, or the fixed finder outputs a break. -/
theorem drawn_released {window : String} {after ξ : ByteArray} (h : drawCoins key window label after coins = .ok ξ) :
    ∃ c roots, trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key
        (getD coins "pins") (getD coins "main") (getD coins "session") = .ok (c, roots) ∧
      ∀ s ∈ dev.runs key.pub, s.commitment = some c → releasedRootsOf c s.log = .ok roots →
        ξ = s.released label after ∨ Break dev key label coins := by
  obtain ⟨c, roots, hts, hl, hst⟩ := drawCoins_ok h
  obtain ⟨r, hr, hrc⟩ := trustedSession_record hts
  exact ⟨c, roots, hts, fun s hs hsc hroots => stream_released hs hsc hroots hr hrc hl hst⟩

end Properties.Challenger.Coins
