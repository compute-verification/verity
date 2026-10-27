import Proofs.Properties.Pouw.LiveSalt
import Proofs.Properties.Pouw.LiveSalt.Canon

/-!
# A live root binds its commitment and its coins (`liveCoinBinding`)

`liveReason` accepts an opening only when the coin's tree leaf climbs its path to the commitment's root (`reason_ok`).
Every frame on that climb is a `node` or `leaf` frame, whose first part, after a 4-byte tag, is the tree's 64-byte domain,
so one root names one domain unless two of the frames collide (`climb_top`, `framed_dom`). The domain's frame begins with
the digest of the binding `{epoch, families, scheme}` (`domain_ok`), whose canonical form names all three
(`Canon.binding_inj`). So two commitments with one root are one, or SHA-512 collides on two messages hashed in checking
them (`same_commitment`). With one commitment, the two climbs start at one position and have one length, so equal roots
give equal leaf digests, level by level (`climb_eq`), and the leaf's frame ends in the coin's hm96 leaf, which opens to its
salt and coin (`coin_eq`). Frames are split only at their first part and their last parts, each 64 bytes: the `uint` parts
between them are minimal big-endian, of varying length.
-/

namespace Properties.Pouw.LiveSalt.Coin

open Lean (Json)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes hash_size be64_size be32_size)
open SampledProofs.Challenger (LiveCommitment LiveOpening liveReason liveDomain liveLeaf taggedSha512 familiesJson total
  BINDING)

/-! ## Collisions -/

theorem eq_or_collides {x y : ByteArray} {xs ys : List ByteArray} (hx : x ∈ xs) (hy : y ∈ ys)
    (h : Flock.Sha512.hash x = Flock.Sha512.hash y) : x = y ∨ Collides xs ys := by
  by_cases e : x = y
  · exact Or.inl e
  · exact Or.inr ⟨x, hx, y, hy, e, h⟩

theorem collides_mono {xs ys xs' ys' : List ByteArray} (hx : ∀ a ∈ xs, a ∈ xs') (hy : ∀ a ∈ ys, a ∈ ys')
    (h : Collides xs ys) : Collides xs' ys' := by
  obtain ⟨x, h1, y, h2, h3, h4⟩ := h
  exact ⟨x, hx x h1, y, hy y h2, h3, h4⟩

theorem cancel_left {a b b' : ByteArray} (h : a ++ b = a ++ b') : b = b' := by
  have e := congrArg bs h
  rw [bs_append, bs_append] at e
  exact bs_inj (List.append_cancel_left e)

/-! ## Frames -/

/-- A part as a frame-v3 frame carries it: its size as u64be, then its bytes. -/
def enc (p : ByteArray) : List UInt8 := bs (Flock.be64 p.size.toUInt64) ++ bs p

theorem enc_length (p : ByteArray) : (enc p).length = 8 + p.size := by
  simp only [enc, List.length_append, bs_length, be64_size]

theorem enc_split {p p' : ByteArray} {r r' : List UInt8} (h : enc p ++ r = enc p' ++ r') (hs : p.size = p'.size) :
    p = p' ∧ r = r' := by
  obtain ⟨he, hr⟩ := List.append_inj h (by rw [enc_length, enc_length, hs])
  unfold enc at he
  rw [hs] at he
  exact ⟨bs_inj (List.append_cancel_left he), hr⟩

theorem enc_inj {p p' : ByteArray} (h : enc p = enc p') : p = p' := by
  unfold enc at h
  exact bs_inj (List.append_inj h (by rw [bs_length, bs_length, be64_size, be64_size])).2

theorem fv3_bs (t : String) (ps : List ByteArray) :
    bs (fv3Message t ps) = bs (Flock.FV3.FRAME ++ Flock.be32 t.toUTF8.size.toUInt32 ++ t.toUTF8) ++ ps.flatMap enc := by
  show bs (ps.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p)
    (Flock.FV3.FRAME ++ Flock.be32 t.toUTF8.size.toUInt32 ++ t.toUTF8)) = _
  generalize Flock.FV3.FRAME ++ Flock.be32 t.toUTF8.size.toUInt32 ++ t.toUTF8 = acc
  induction ps generalizing acc with
  | nil => simp
  | cons p ps ih => simp only [List.foldl_cons, ih, bs_append, List.flatMap_cons, enc, List.append_assoc]

/-- Two frames whose tags have one length and whose first parts have one size have one first part. -/
theorem first_eq {t t' : String} {d d' : ByteArray} {ps ps' : List ByteArray} (ht : t.toUTF8.size = t'.toUTF8.size)
    (hd : d.size = d'.size) (h : fv3Message t (d :: ps) = fv3Message t' (d' :: ps')) : d = d' := by
  have e := congrArg bs h
  rw [fv3_bs, fv3_bs, List.flatMap_cons, List.flatMap_cons] at e
  have hl : (bs (Flock.FV3.FRAME ++ Flock.be32 t.toUTF8.size.toUInt32 ++ t.toUTF8)).length =
      (bs (Flock.FV3.FRAME ++ Flock.be32 t'.toUTF8.size.toUInt32 ++ t'.toUTF8)).length := by
    simp only [bs_append, List.length_append, bs_length, be32_size, ht]
  exact (enc_split (List.append_inj e hl).2 hd).1

theorem fv3_cancel {t : String} {ps qs qs' : List ByteArray}
    (h : fv3Message t (ps ++ qs) = fv3Message t (ps ++ qs')) : qs.flatMap enc = qs'.flatMap enc := by
  have e := congrArg bs h
  rw [fv3_bs, fv3_bs, List.flatMap_append, List.flatMap_append] at e
  exact List.append_cancel_left (List.append_cancel_left e)

/-- Two frames with one tag and the same leading parts, each ending in two parts whose first ones have one size, end in
the same two parts. -/
theorem pair_eq {t : String} {ps : List ByteArray} {l r l' r' : ByteArray} (hl : l.size = l'.size)
    (h : fv3Message t (ps ++ [l, r]) = fv3Message t (ps ++ [l', r'])) : l = l' ∧ r = r' := by
  have e := fv3_cancel h
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at e
  obtain ⟨h1, h2⟩ := enc_split e hl
  exact ⟨h1, enc_inj h2⟩

theorem last_eq {t : String} {ps : List ByteArray} {x x' : ByteArray}
    (h : fv3Message t (ps ++ [x]) = fv3Message t (ps ++ [x'])) : x = x' := by
  have e := fv3_cancel h
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at e
  exact enc_inj e

/-! ## The climb -/

/-- The node frame `climbL` hashes at index `idx` of level `d`, over `cur` and its sibling `sib`. -/
def nodeMsg (dom cur : ByteArray) (idx d : Nat) (sib : ByteArray) : ByteArray :=
  if idx % 2 = 0 then fv3Message "node" [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2), cur, sib]
  else fv3Message "node" [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2), sib, cur]

theorem frames_cons (dom cur sib : ByteArray) (idx d : Nat) (rest : List ByteArray) :
    climbFrames dom cur idx d (sib :: rest) =
      nodeMsg dom cur idx d sib ::
        climbFrames dom (Flock.Sha512.hash (nodeMsg dom cur idx d sib)) (idx / 2) (d + 1) rest :=
  rfl

theorem climb_cons (dom cur sib : ByteArray) (idx d : Nat) (rest : List ByteArray) :
    Flock.Registered.climbL dom cur idx d (sib :: rest) =
      Flock.Registered.climbL dom (Flock.Sha512.hash (nodeMsg dom cur idx d sib)) (idx / 2) (d + 1) rest := by
  rw [Flock.Registered.climbL]
  unfold nodeMsg
  by_cases hi : idx % 2 = 0
  · rw [ite_eq_left hi, ite_eq_left hi]
    rfl
  · rw [ite_eq_right hi, ite_eq_right hi]
    rfl

/-- `m` is a `node` or `leaf` frame of the tree whose domain is `dom`: its first part is `dom`. -/
def Framed (dom m : ByteArray) : Prop :=
  ∃ t ps, (t = "node" ∨ t = "leaf") ∧ m = fv3Message t (dom :: ps)

theorem nodeMsg_framed (dom cur : ByteArray) (idx d : Nat) (sib : ByteArray) :
    Framed dom (nodeMsg dom cur idx d sib) := by
  unfold nodeMsg
  split
  · exact ⟨"node", _, Or.inl rfl, rfl⟩
  · exact ⟨"node", _, Or.inl rfl, rfl⟩

theorem framed_dom {dom dom' m : ByteArray} (hd : dom.size = 64) (hd' : dom'.size = 64) (h : Framed dom m)
    (h' : Framed dom' m) : dom = dom' := by
  obtain ⟨t, ps, ht, rfl⟩ := h
  obtain ⟨t', ps', ht', he⟩ := h'
  have hs : ∀ {u : String}, (u = "node" ∨ u = "leaf") → u.toUTF8.size = 4 := by
    intro u hu
    rcases hu with rfl | rfl <;> decide
  exact first_eq (by rw [hs ht, hs ht']) (by rw [hd, hd']) he

/-- The root a climb reaches from a frame of its tree is the digest of a frame of its tree: the first, or one hashed
on the way. -/
theorem climb_top (dom : ByteArray) : ∀ (path : List ByteArray) (m0 : ByteArray) (idx d : Nat), Framed dom m0 →
    ∃ m, Framed dom m ∧ (m = m0 ∨ m ∈ climbFrames dom (Flock.Sha512.hash m0) idx d path) ∧
      Flock.Registered.climbL dom (Flock.Sha512.hash m0) idx d path = Flock.Sha512.hash m
  | [], m0, _, _, hf => ⟨m0, hf, Or.inl rfl, rfl⟩
  | sib :: rest, m0, idx, d, _ => by
    obtain ⟨m, hf, hm, he⟩ := climb_top dom rest (nodeMsg dom (Flock.Sha512.hash m0) idx d sib) (idx / 2) (d + 1)
      (nodeMsg_framed _ _ _ _ _)
    refine ⟨m, hf, Or.inr ?_, ?_⟩
    · rw [frames_cons]
      rcases hm with rfl | hm
      · exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ hm
    · rw [climb_cons]
      exact he

/-- Two climbs of one tree from one index and level, through paths of one length of 64-byte siblings, from 64-byte
digests, reach one root only from one digest, unless two of the node frames they hash collide. -/
theorem climb_eq (dom : ByteArray) : ∀ (path path' : List ByteArray) (cur cur' : ByteArray) (idx d : Nat),
    path.length = path'.length → (∀ s ∈ path, s.size = 64) → (∀ s ∈ path', s.size = 64) →
    cur.size = 64 → cur'.size = 64 →
    Flock.Registered.climbL dom cur idx d path = Flock.Registered.climbL dom cur' idx d path' →
    cur = cur' ∨ Collides (climbFrames dom cur idx d path) (climbFrames dom cur' idx d path')
  | [], [], _, _, _, _, _, _, _, _, _, h => Or.inl h
  | [], _ :: _, _, _, _, _, hl, _, _, _, _, _ => by simp at hl
  | _ :: _, [], _, _, _, _, hl, _, _, _, _, _ => by simp at hl
  | sib :: rest, sib' :: rest', cur, cur', idx, d, hl, hs, hs', hc, hc', h => by
    rw [climb_cons, climb_cons] at h
    rw [frames_cons, frames_cons]
    rcases climb_eq dom rest rest' _ _ (idx / 2) (d + 1) (by simpa using hl)
        (fun s hm => hs s (List.mem_cons_of_mem _ hm)) (fun s hm => hs' s (List.mem_cons_of_mem _ hm))
        (hash_size _) (hash_size _) h with he | ⟨x, hx, y, hy, hxy, hh⟩
    · refine (eq_or_collides List.mem_cons_self List.mem_cons_self he).elim (fun hm => Or.inl ?_) Or.inr
      unfold nodeMsg at hm
      by_cases hi : idx % 2 = 0
      · rw [ite_eq_left hi, ite_eq_left hi] at hm
        exact (pair_eq (ps := [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2)]) (by rw [hc, hc']) hm).1
      · rw [ite_eq_right hi, ite_eq_right hi] at hm
        exact (pair_eq (ps := [dom, Flock.FV3.uint d, Flock.FV3.uint (idx / 2)])
          (by rw [hs sib List.mem_cons_self, hs' sib' List.mem_cons_self]) hm).2
    · exact Or.inr ⟨x, List.mem_cons_of_mem _ hx, y, List.mem_cons_of_mem _ hy, hxy, hh⟩

/-! ## The domain and the binding -/

/-- The bytes `taggedSha512 tag` hashes when the canonical form is `s`. -/
def taggedBytes (tag s : String) : ByteArray :=
  "verity/tagged-sha512/v1".toUTF8.push 0 ++ Flock.be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8 ++
    Flock.be64 s.toUTF8.size.toUInt64 ++ s.toUTF8

theorem tagged_eq {tag s : String} {j : Json} (hc : Flock.canon j = .ok s) :
    taggedSha512 tag j = .ok (Flock.Sha512.hash (taggedBytes tag s)) := by
  unfold taggedSha512
  rw [hc]
  rfl

theorem taggedMessage_eq {tag s : String} {j : Json} (hc : Flock.canon j = .ok s) :
    taggedMessage tag j = [taggedBytes tag s] := by
  unfold taggedMessage
  rw [hc]
  rfl

theorem tagged_ok {tag : String} {j : Json} {b : ByteArray} (h : taggedSha512 tag j = .ok b) :
    ∃ s, Flock.canon j = .ok s ∧ b = Flock.Sha512.hash (taggedBytes tag s) := by
  cases hc : Flock.canon j with
  | error e =>
    unfold taggedSha512 at h
    rw [hc] at h
    cases h
  | ok s =>
    rw [tagged_eq hc] at h
    exact ⟨s, rfl, (Except.ok.inj h).symm⟩

theorem taggedBytes_inj {tag s s' : String} (h : taggedBytes tag s = taggedBytes tag s') : s = s' := by
  have e := congrArg bs h
  simp only [taggedBytes, bs_append, List.append_assoc, List.append_cancel_left_eq] at e
  exact String.toByteArray_inj.1 (bs_inj (List.append_inj e (by rw [bs_length, bs_length, be64_size, be64_size])).2)

/-- The positions object `liveDomain` hashes. -/
def positions (fs : List (String × Nat)) : Json :=
  Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson (total fs))]

/-- The domain's frame, over the binding's digest `b` and the positions' digest `p`. -/
def domMsg (b p : ByteArray) (fs : List (String × Nat)) : ByteArray :=
  fv3Message "domain" [b, Flock.FV3.uint 1, p, Flock.FV3.uint (total fs)]

theorem domain_ok {s : String} {e : Nat} {fs : List (String × Nat)} {dom : ByteArray}
    (h : liveDomain s e fs = .ok dom) :
    ∃ b p, taggedSha512 BINDING (Canon.binding e fs s) = .ok b ∧
      taggedSha512 "veritor/indexed-domain/range/v1" (positions fs) = .ok p ∧
      dom = Flock.Sha512.hash (domMsg b p fs) := by
  unfold liveDomain at h
  cases hb : taggedSha512 BINDING (Json.mkObj [("epoch", Lean.toJson e), ("families", familiesJson fs),
      ("scheme", Json.str s)]) with
  | error _ => simp [hb] at h; cases h
  | ok b =>
    cases hp : taggedSha512 "veritor/indexed-domain/range/v1" (Json.mkObj [("start", Lean.toJson (0 : Nat)),
        ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson (total fs))]) with
    | error _ => simp [hb, hp] at h; cases h
    | ok p =>
      simp only [hb, hp, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨b, p, hb, hp, h.symm⟩

theorem domainHashed_eq {c : LiveCommitment} {b p : ByteArray}
    (hb : taggedSha512 BINDING (Canon.binding c.epoch c.families c.scheme) = .ok b)
    (hp : taggedSha512 "veritor/indexed-domain/range/v1" (positions c.families) = .ok p) :
    domainHashed c = taggedMessage BINDING (Canon.binding c.epoch c.families c.scheme) ++
      taggedMessage "veritor/indexed-domain/range/v1" (positions c.families) ++ [domMsg b p c.families] := by
  have hb' : taggedSha512 BINDING (Json.mkObj [("epoch", Lean.toJson c.epoch),
      ("families", familiesJson c.families), ("scheme", Json.str c.scheme)]) = .ok b := hb
  have hp' : taggedSha512 "veritor/indexed-domain/range/v1" (Json.mkObj [("start", Lean.toJson (0 : Nat)),
      ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson (total c.families))]) = .ok p := hp
  simp only [domainHashed, hb', hp']
  rfl

theorem domain_mem {c : LiveCommitment} {b p : ByteArray} {s : String} (L : List ByteArray)
    (hb : taggedSha512 BINDING (Canon.binding c.epoch c.families c.scheme) = .ok b)
    (hp : taggedSha512 "veritor/indexed-domain/range/v1" (positions c.families) = .ok p)
    (hs : Flock.canon (Canon.binding c.epoch c.families c.scheme) = .ok s) :
    domMsg b p c.families ∈ domainHashed c ++ L ∧ taggedBytes BINDING s ∈ domainHashed c ++ L := by
  rw [domainHashed_eq hb hp, taggedMessage_eq hs]
  simp

/-! ## The coin's leaf -/

/-- The tree leaf's frame of the coin `o` at position `pos` of `c`'s tree, whose domain is `dom`. -/
def leafMsg (dom : ByteArray) (pos : Nat) (c : LiveCommitment) (o : LiveOpening) : ByteArray :=
  fv3Message "leaf" [dom, Flock.FV3.uint pos, Flock.FV3.uint pos, c.scheme.toUTF8, liveLeaf c.scheme o.coin o.salt]

theorem liveHashed_eq (c : LiveCommitment) (dom : ByteArray) (pos : Nat) (o : LiveOpening) :
    liveHashed c dom pos o = [c.scheme.toUTF8.push 0 ++ o.coin, Flock.Hm96.Default512.saltPrefix ++ o.salt,
      Flock.Hm96.Default512.leafPrefix ++
        Flock.Hm96.Default512.commitString (Flock.Sha512.hash (c.scheme.toUTF8.push 0 ++ o.coin)) o.salt,
      leafMsg dom pos c o] ++ climbFrames dom (Flock.Sha512.hash (leafMsg dom pos c o)) pos 0 o.path :=
  rfl

theorem top_mem {c : LiveCommitment} {dom : ByteArray} {pos : Nat} {o : LiveOpening} {m : ByteArray}
    (hm : m = leafMsg dom pos c o ∨ m ∈ climbFrames dom (Flock.Sha512.hash (leafMsg dom pos c o)) pos 0 o.path) :
    m ∈ domainHashed c ++ liveHashed c dom pos o := by
  apply List.mem_append_right
  rw [liveHashed_eq]
  rcases hm with rfl | hm
  · simp
  · simp [hm]

theorem commit_split {x x' salt salt' : ByteArray}
    (h : Flock.Hm96.Default512.commitString x salt = Flock.Hm96.Default512.commitString x' salt') :
    ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ (Flock.Hm96.Default512.mask salt).get! i) =
      ByteArray.mk ((Array.range 64).map fun i => x'.get! i ^^^ (Flock.Hm96.Default512.mask salt').get! i) ∧
    Flock.Sha512.hash (Flock.Hm96.Default512.saltPrefix ++ salt) =
      Flock.Sha512.hash (Flock.Hm96.Default512.saltPrefix ++ salt') := by
  have e := congrArg bs h
  simp only [Flock.Hm96.Default512.commitString, bs_append] at e
  obtain ⟨e1, e2⟩ := split_bytes e (by show ((Array.range 64).map _).size = ((Array.range 64).map _).size; simp)
  exact ⟨e1, bs_inj e2⟩

theorem xor_inj {x x' m : ByteArray} (hx : x.size = 64) (hx' : x'.size = 64)
    (h : ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i) =
      ByteArray.mk ((Array.range 64).map fun i => x'.get! i ^^^ m.get! i)) : x = x' := by
  obtain ⟨d⟩ := x
  obtain ⟨d'⟩ := x'
  have ha := ByteArray.mk.inj h
  congr 1
  apply Array.ext
  · exact hx.trans hx'.symm
  · intro i h1 h2
    have hi : i < 64 := by
      have : d.size = 64 := hx
      omega
    have e : (ByteArray.mk d).get! i ^^^ m.get! i = (ByteArray.mk d').get! i ^^^ m.get! i := by
      have e := congrArg (fun a : Array UInt8 => a[i]?) ha
      simpa [hi] using e
    have e2 := congrArg (· ^^^ m.get! i) e
    simp only [UInt8.xor_assoc, UInt8.xor_self, UInt8.xor_zero] at e2
    rw [← getElem!_pos d i h1, ← getElem!_pos d' i h2]
    exact e2

/-- Two openings of one coin of one commitment whose tree leaves have one digest carry one coin, unless two messages
hashed in checking them collide. -/
theorem coin_eq {c : LiveCommitment} {dom : ByteArray} {pos : Nat} {o o' : LiveOpening}
    (h : Flock.Sha512.hash (leafMsg dom pos c o) = Flock.Sha512.hash (leafMsg dom pos c o')) :
    o.coin = o'.coin ∨ Collides (liveHashed c dom pos o) (liveHashed c dom pos o') := by
  refine (eq_or_collides (xs := liveHashed c dom pos o) (ys := liveHashed c dom pos o')
    (by rw [liveHashed_eq]; simp) (by rw [liveHashed_eq]; simp) h).elim (fun h1 => ?_) Or.inr
  have h2 : liveLeaf c.scheme o.coin o.salt = liveLeaf c.scheme o'.coin o'.salt :=
    last_eq (ps := [dom, Flock.FV3.uint pos, Flock.FV3.uint pos, c.scheme.toUTF8]) h1
  have h3 : Flock.Sha512.hash (Flock.Hm96.Default512.leafPrefix ++
        Flock.Hm96.Default512.commitString (Flock.Sha512.hash (c.scheme.toUTF8.push 0 ++ o.coin)) o.salt) =
      Flock.Sha512.hash (Flock.Hm96.Default512.leafPrefix ++
        Flock.Hm96.Default512.commitString (Flock.Sha512.hash (c.scheme.toUTF8.push 0 ++ o'.coin)) o'.salt) := h2
  refine (eq_or_collides (xs := liveHashed c dom pos o) (ys := liveHashed c dom pos o')
    (by rw [liveHashed_eq]; simp) (by rw [liveHashed_eq]; simp) h3).elim (fun h4 => ?_) Or.inr
  obtain ⟨h5, h6⟩ := commit_split (cancel_left h4)
  refine (eq_or_collides (xs := liveHashed c dom pos o) (ys := liveHashed c dom pos o')
    (by rw [liveHashed_eq]; simp) (by rw [liveHashed_eq]; simp) h6).elim (fun h7 => ?_) Or.inr
  rw [cancel_left h7] at h5
  refine (eq_or_collides (xs := liveHashed c dom pos o) (ys := liveHashed c dom pos o')
    (by rw [liveHashed_eq]; simp) (by rw [liveHashed_eq]; simp) (xor_inj (hash_size _) (hash_size _) h5)).elim
    (fun h8 => Or.inl (cancel_left h8)) Or.inr

/-! ## The acceptance -/

theorem reason_ok {c : LiveCommitment} {o : LiveOpening} {label : String} {index : Int} {after : ByteArray}
    (h : liveReason c o label index after = .ok none) :
    ∃ pos dom, c.position label index = some pos ∧ liveDomain c.scheme c.epoch c.families = .ok dom ∧
      o.path.length = Flock.log2ceil c.count ∧ (∀ s ∈ o.path, s.size = 64) ∧
      Flock.Registered.climbL dom (Flock.Sha512.hash (leafMsg dom pos c o)) pos 0 o.path = c.root := by
  have ne {s : String} (h : (pure (some s) : Except String (Option String)) = .ok none) : False := by
    cases h
  unfold liveReason at h
  cases hp : c.position label index with
  | none => simp [hp] at h
  | some pos =>
    simp only [hp] at h
    split at h
    · exact (ne h).elim
    split at h
    · exact (ne h).elim
    split at h
    · exact (ne h).elim
    cases hd : liveDomain c.scheme c.epoch c.families with
    | error e => simp [hd] at h; cases h
    | ok dom =>
      simp only [hd, bind, Except.bind] at h
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at hc
        obtain ⟨⟨hlen, hall⟩, hroot⟩ := hc
        exact ⟨pos, dom, rfl, rfl, hlen, hall, hroot⟩
      · exact (ne h).elim

theorem hashedBy_eq {c : LiveCommitment} {o : LiveOpening} {label : String} {index : Int} {dom : ByteArray}
    {pos : Nat} (hd : liveDomain c.scheme c.epoch c.families = .ok dom) (hp : c.position label index = some pos) :
    hashedBy c o label index = domainHashed c ++ liveHashed c dom pos o := by
  simp only [hashedBy, hd, hp]

theorem commitment_eq {c c' : LiveCommitment} (hs : c.scheme = c'.scheme) (he : c.epoch = c'.epoch)
    (hf : c.families = c'.families) (hr : c.root = c'.root) : c = c' := by
  cases c
  cases c'
  simp only at hs he hf hr
  subst hs he hf hr
  rfl

/-- Two commitments with one root, each with a coin whose climb reaches it, are one commitment, unless two messages
hashed in checking them collide. -/
theorem same_commitment {c c' : LiveCommitment} {o o' : LiveOpening} {dom dom' : ByteArray} {pos pos' : Nat}
    (hd : liveDomain c.scheme c.epoch c.families = .ok dom)
    (hd' : liveDomain c'.scheme c'.epoch c'.families = .ok dom')
    (hc : Flock.Registered.climbL dom (Flock.Sha512.hash (leafMsg dom pos c o)) pos 0 o.path = c.root)
    (hc' : Flock.Registered.climbL dom' (Flock.Sha512.hash (leafMsg dom' pos' c' o')) pos' 0 o'.path = c'.root)
    (hr : c.root = c'.root) :
    c = c' ∨ Collides (domainHashed c ++ liveHashed c dom pos o) (domainHashed c' ++ liveHashed c' dom' pos' o') := by
  obtain ⟨b, p, hb, hp, hdom⟩ := domain_ok hd
  obtain ⟨b', p', hb', hp', hdom'⟩ := domain_ok hd'
  obtain ⟨m, hf, hm, he⟩ := climb_top dom o.path (leafMsg dom pos c o) pos 0 ⟨"leaf", _, Or.inr rfl, rfl⟩
  obtain ⟨m', hf', hm', he'⟩ := climb_top dom' o'.path (leafMsg dom' pos' c' o') pos' 0 ⟨"leaf", _, Or.inr rfl, rfl⟩
  have hmm : Flock.Sha512.hash m = Flock.Sha512.hash m' := by rw [← he, ← he', hc, hc', hr]
  refine (eq_or_collides (top_mem hm) (top_mem hm') hmm).elim (fun e => ?_) Or.inr
  subst e
  have edom : dom = dom' := framed_dom (by rw [hdom]; exact hash_size _) (by rw [hdom']; exact hash_size _) hf hf'
  subst edom
  have hdm : Flock.Sha512.hash (domMsg b p c.families) = Flock.Sha512.hash (domMsg b' p' c'.families) := by
    rw [← hdom, ← hdom']
  obtain ⟨s, hs, hbs⟩ := tagged_ok hb
  obtain ⟨s', hs', hbs'⟩ := tagged_ok hb'
  have dm := domain_mem (liveHashed c dom pos o) hb hp hs
  have dm' := domain_mem (liveHashed c' dom pos' o') hb' hp' hs'
  refine (eq_or_collides dm.1 dm'.1 hdm).elim (fun e => ?_) Or.inr
  have eb : b = b' := first_eq rfl (by rw [hbs, hbs', hash_size, hash_size]) e
  refine (eq_or_collides dm.2 dm'.2 (by rw [← hbs, ← hbs', eb])).elim (fun e2 => ?_) Or.inr
  replace hs := Canon.canonLine_of_canon hs
  replace hs' := Canon.canonLine_of_canon hs'
  rw [Canon.canon_binding] at hs hs'
  obtain ⟨he, hfam, hsch⟩ :=
    Canon.binding_inj ((Except.ok.inj hs).trans ((taggedBytes_inj e2).trans (Except.ok.inj hs').symm))
  exact Or.inl (commitment_eq hsch he hfam hr)

end Properties.Pouw.LiveSalt.Coin

namespace Properties.Pouw.LiveSalt

open Properties.NetworkCertifier.Bytes (hash_size)
open SampledProofs.Challenger (LiveCommitment LiveOpening liveReason)
open Coin (reason_ok hashedBy_eq same_commitment climb_eq coin_eq collides_mono top_mem leafMsg)

/-- `LiveCoinBinding` of two commitments with one root. -/
theorem liveCoinBinding_root (c c' : LiveCommitment) (o o' : LiveOpening) (label : String) (index : Int)
    (after after' : ByteArray) (hr : c.root = c'.root) (h : liveReason c o label index after = .ok none)
    (h' : liveReason c' o' label index after' = .ok none) :
    (c = c' ∧ o.coin = o'.coin) ∨ Collides (hashedBy c o label index) (hashedBy c' o' label index) := by
  obtain ⟨pos, dom, hp, hd, hlen, hsib, hcl⟩ := reason_ok h
  obtain ⟨pos', dom', hp', hd', hlen', hsib', hcl'⟩ := reason_ok h'
  rw [hashedBy_eq hd hp, hashedBy_eq hd' hp']
  refine (same_commitment hd hd' hcl hcl' hr).elim (fun e => ?_) Or.inr
  subst e
  have epos : pos = pos' := Option.some.inj (hp.symm.trans hp')
  have edom : dom = dom' := Except.ok.inj (hd.symm.trans hd')
  subst epos edom
  refine (climb_eq dom o.path o'.path (Flock.Sha512.hash (leafMsg dom pos c o))
    (Flock.Sha512.hash (leafMsg dom pos c o')) pos 0 (hlen.trans hlen'.symm) hsib hsib' (hash_size _) (hash_size _)
    (hcl.trans hcl'.symm)).elim
    (fun e => (coin_eq e).elim (fun e2 => Or.inl ⟨rfl, e2⟩)
      (fun e2 => Or.inr (collides_mono (fun _ ha => List.mem_append_right _ ha)
        (fun _ ha => List.mem_append_right _ ha) e2)))
    (fun e => Or.inr (collides_mono (fun _ ha => top_mem (Or.inr ha)) (fun _ ha => top_mem (Or.inr ha)) e))

theorem liveCoinBinding : LiveCoinBinding := by
  intro s s' e e' fs fs' r o o' label index after after'
  exact liveCoinBinding_root ⟨s, e, fs, r⟩ ⟨s', e', fs', r⟩ o o' label index after after' rfl

theorem onePair : OnePair PouwCheck.liveSaltCheck := onePair_of_binding liveCoinBinding

end Properties.Pouw.LiveSalt
