import Proofs.Flock.Bridge.Deployed
import Proofs.Flock.Recursive.TopRoot
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# Two ports with one registered root and different domains are a SHA-512 collision

The deployed check opens a position at the first registered port whose root is the position's (`regPath`'s `find?`
over `portsOf`), so a row of port `q` is checked in port `q'`'s tree when an earlier `q'` has `q`'s root. Each
port's root is `FV3.root` over its own domain, its port name (`domOf`), and that root is SHA-512 of one top frame (the
top node's, or with one leaf that leaf's, or with none the pad's; `root_top`), whose first part is the domain. With one
leaf the frame ends in the port's leaf schema and its payload, and two frames at different row schemas differ
(`step_schema`), so two frames at one domain's place are at one schema. So two ports with one root and different
domains give two different inputs with one SHA-512 digest (`rootsOf_collides`).

* `RootFixesDomain`: an earlier port with a port's root has its domain. Off that collision it holds
  (`fixes_of_none`). Two ports of one name may share a root with no collision, and may have different schemas, so what
  the deployed check needs is the distinct-roots form, at a circuit that names each port once (`Registered.dupName`,
  which the verifier's registered reads check): `rootsDistinct_of_fixes`. Then the check opens a row of `q` in `q`'s
  own tree, at `q`'s schema.
* `rootsFinder`: a finder with no coins that recomputes every port's root (`rootsCost` SHA-512 calls) and outputs the
  first pair of top frames that `rootsOf_collides` finds (`rootsColl`). Under `ecr/sha-512` (`Claims.ExpectedCR`, read
  through `Claims.ecrBound`) at that finder, a file that holds such a collision has `ecrBound ≥ 1`
  (`one_le_ecrBound`), and one that holds none has `ecrBound = rootsCost / 2^256` (`ecrBound_of_none`). So a
  probability bounded at every file with `RootFixesDomain` is bounded at every file with that term added
  (`le_add_ecrBound`).
-/

namespace Flock.Bridge.PortRoots

open Claims Claims.Sampling
open Flock.SecurityProofs
open Properties.NetworkCertifier.Bytes (bs bs_inj be64_size)
open scoped ENNReal

/-! ## Frames -/

theorem foldl_step_size : ∀ (ps : List ByteArray) (a : ByteArray),
    (ps.foldl Registered.step a).size = a.size + (ps.map fun p => 8 + p.size).sum
  | [], a => by simp
  | p :: ps, a => by
    rw [List.foldl_cons, foldl_step_size ps, Registered.step_size, be64_size, List.map_cons, List.sum_cons]
    omega

theorem sum_size_eq {ps ps' : List ByteArray} (h : List.Forall₂ (fun p p' : ByteArray => p.size = p'.size) ps ps') :
    (ps.map fun p => 8 + p.size).sum = (ps'.map fun p => 8 + p.size).sum := by
  induction h with
  | nil => rfl
  | cons hp _ ih => simp only [List.map_cons, List.sum_cons, hp, ih]

theorem foldl_step_inj {ps ps' : List ByteArray} (h : List.Forall₂ (fun p p' : ByteArray => p.size = p'.size) ps ps') :
    ∀ {a a' : ByteArray}, a.size = a'.size → ps.foldl Registered.step a = ps'.foldl Registered.step a' →
      a = a' ∧ ps = ps' := by
  induction h with
  | nil => intro a a' _ he; exact ⟨he, rfl⟩
  | cons hp _ ih =>
    intro a a' ha he
    obtain ⟨h1, rfl⟩ := ih (by rw [Registered.step_size, Registered.step_size, ha, hp]) he
    obtain ⟨rfl, rfl⟩ := Registered.step_inj ha hp h1
    exact ⟨rfl, rfl⟩

/-- **A frame binds its first part**, when its other parts have sizes part by part. -/
theorem frame_head_inj {tag : String} {d d' : ByteArray} {ps ps' : List ByteArray}
    (h : Registered.frame tag (d :: ps) = Registered.frame tag (d' :: ps'))
    (hps : List.Forall₂ (fun p p' : ByteArray => p.size = p'.size) ps ps') : d = d' := by
  have hs := congrArg ByteArray.size h
  simp only [Registered.frame, foldl_step_size, List.map_cons, List.sum_cons, sum_size_eq hps] at hs
  have hd : d.size = d'.size := by omega
  exact (List.cons.inj (foldl_step_inj (.cons hd hps) rfl h).2).1

/-- A frame's last part, of one size, and the rest. -/
theorem step_inj' {a a' p p' : ByteArray} (hp : p.size = p'.size) (h : Registered.step a p = Registered.step a' p') :
    a = a' ∧ p = p' := by
  have hs := congrArg ByteArray.size h
  rw [Registered.step_size, Registered.step_size, hp] at hs
  exact Registered.step_inj (by omega) hp h

theorem drop_of_append_eq {l₁ l₂ s t : List UInt8} (h : l₁ ++ s = l₂ ++ t) (hl : t.length ≤ s.length) :
    s.drop (s.length - t.length) = t := by
  have hs : s = s.take (s.length - t.length) ++ s.drop (s.length - t.length) := (List.take_append_drop _ _).symm
  rw [hs, ← List.append_assoc] at h
  exact (List.append_inj' h (by simp only [List.length_drop]; omega)).2

/-- The row schemas a port's leaves are framed at. -/
def schemas : List String := [HmRow.ROW_SCHEMA, HmRow.ROW_SCHEMA_V2, HmRow.ROW_SCHEMA_SEG]

theorem schemaOf_mem (c : Flock.Circuit) (q : ℕ) : schemaOf c q ∈ schemas := by
  unfold schemaOf schemas
  cases c.ports[q]? with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.getD_some, Registered.portSchema]
    split_ifs <;> simp

/-- **A frame's schema part ends it apart from its payload**: two frames that end in two row schemas are one only at
one schema (the schemas' last bytes differ). -/
theorem step_schema {a a' : ByteArray} {s s' : String} (hs : s ∈ schemas) (hs' : s' ∈ schemas)
    (h : Registered.step a s.toUTF8 = Registered.step a' s'.toUTF8) : s = s' := by
  have e : ∀ (b : ByteArray) (t : String), (Registered.step b t.toUTF8).data.toList =
      (b ++ Flock.be64 t.toUTF8.size.toUInt64).data.toList ++ t.toUTF8.data.toList := fun b t => by
    simp [Registered.step, ByteArray.data_append]
  have hl := (e a s).symm.trans ((congrArg (fun b : ByteArray => b.data.toList) h).trans (e a' s'))
  simp only [schemas, List.mem_cons, List.not_mem_nil, or_false] at hs hs'
  rcases hs with rfl | rfl | rfl <;> rcases hs' with rfl | rfl | rfl
  all_goals first
    | rfl
    | exact absurd (drop_of_append_eq hl (by decide)) (by decide)
    | exact absurd (drop_of_append_eq hl.symm (by decide)) (by decide)

/-! ## A port's root and its top frame -/

/-- Port `q`'s registered tree leaves, as `rootsOf` builds them. -/
def leaves (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : Array ByteArray :=
  (Array.range nl).map fun u => Flock.Registered.leafAs (schemaOf c q) (domOf c q) u (regLeafOf c pub q u)

theorem leaves_size (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : (leaves c pub nl q).size = nl := by
  simp [leaves]

theorem leaves_width (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : ∀ x ∈ leaves c pub nl q, x.size = 64 := by
  intro x hx
  simp only [leaves, Array.mem_map] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact Flock.Sha512.hash_size _

theorem regLeafOf_size (c : Flock.Circuit) (pub : Flock.Public) (q u : ℕ) : (regLeafOf c pub q u).size = 64 := by
  unfold regLeafOf Flock.Hm96.Default512.treeLeaf
  exact Flock.Sha512.hash_size _

theorem rootsOf_getD (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) {q : ℕ} (hq : q < c.ports.size) :
    (rootsOf c pub nl).getD q [] = bs (Flock.FV3.root (domOf c q) (leaves c pub nl q) Flock.Sha512.hash) := by
  simp [rootsOf, leaves, List.getD_eq_getElem?_getD, hq]

theorem one_le_log2ceil {n : ℕ} (h : 2 ≤ n) : 1 ≤ Flock.log2ceil n := by
  unfold Flock.log2ceil
  rw [ite_eq_right (by omega)]
  omega

/-- **What port `q`'s root hashes last**: the frame of its tree's top node; with one leaf, that leaf's frame; with
none, the lone pad's. -/
def topIn (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : ByteArray :=
  if nl = 0 then Registered.frame "pad" [domOf c q, Flock.FV3.uint 0]
  else if nl = 1 then Registered.leafInAs (schemaOf c q) (domOf c q) 0 (regLeafOf c pub q 0)
  else Registered.nodeIn (domOf c q) (Flock.log2ceil nl - 1) 0
    (TopRoot.lev (domOf c q) (leaves c pub nl q) (Flock.log2ceil nl - 1))[0]!
    (TopRoot.lev (domOf c q) (leaves c pub nl q) (Flock.log2ceil nl - 1))[1]!

theorem lev_top_size (c : Flock.Circuit) (pub : Flock.Public) {nl : ℕ} (h2 : 2 ≤ nl) (q : ℕ) {i : ℕ} (hi : i < 2) :
    ((TopRoot.lev (domOf c q) (leaves c pub nl q) (Flock.log2ceil nl - 1))[i]!).size = 64 := by
  have h1 := one_le_log2ceil h2
  refine TopRoot.lev_width _ _ (leaves_width c pub nl q) _ _ (TopRoot.get!_mem ?_)
  rw [TopRoot.lev_size _ _ _ (by rw [leaves_size]; omega), leaves_size,
    show Flock.log2ceil nl - (Flock.log2ceil nl - 1) = 1 by omega]
  simpa using hi

/-- **A port's root is SHA-512 of its top frame.** -/
theorem root_top (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) :
    Flock.FV3.root (domOf c q) (leaves c pub nl q) Flock.Sha512.hash = Flock.Sha512.hash (topIn c pub nl q) := by
  rw [TopRoot.root_eq, leaves_size]
  rcases Nat.lt_or_ge nl 2 with hlt | h2
  · interval_cases nl
    · simp [TopRoot.lev, TopRoot.base, leaves, topIn, Flock.nextPow2, Flock.log2ceil, Registered.hash_frame]
    · simp [TopRoot.lev, TopRoot.base, leaves, topIn, Flock.nextPow2, Flock.log2ceil, Registered.leafAs_eq]
  · obtain ⟨k, hk⟩ : ∃ k, Flock.log2ceil nl = k + 1 := ⟨Flock.log2ceil nl - 1, by have := one_le_log2ceil h2; omega⟩
    have hk' : Flock.log2ceil nl - 1 = k := by omega
    have hsz : 0 < (TopRoot.lev (domOf c q) (leaves c pub nl q) k).size / 2 := by
      rw [TopRoot.lev_size _ _ _ (by rw [leaves_size]; omega), leaves_size, hk, show k + 1 - k = 1 by omega]
      decide
    rw [hk, TopRoot.lev, TopRoot.next_get _ _ _ hsz]
    simp only [topIn, ite_eq_right (show nl ≠ 0 by omega), ite_eq_right (show nl ≠ 1 by omega), hk', Nat.mul_zero,
      Nat.zero_add]

/-- **Two top frames over different domains differ.** -/
theorem topIn_ne {c : Flock.Circuit} {pub : Flock.Public} {nl q q' : ℕ} (hd : domOf c q' ≠ domOf c q) :
    topIn c pub nl q' ≠ topIn c pub nl q := by
  intro h
  apply hd
  unfold topIn at h
  split_ifs at h with h0 h1
  · exact frame_head_inj h (.cons rfl .nil)
  · have hs : schemaOf c q' = schemaOf c q := by
      have h' := h
      simp only [Registered.leafInAs, Registered.frame, List.foldl] at h'
      exact step_schema (schemaOf_mem c q') (schemaOf_mem c q)
        (step_inj' (by rw [regLeafOf_size, regLeafOf_size]) h').1
    rw [hs] at h
    exact frame_head_inj h (.cons rfl (.cons rfl (.cons rfl (.cons (by rw [regLeafOf_size, regLeafOf_size]) .nil))))
  · exact frame_head_inj h (.cons rfl (.cons rfl
      (.cons (by rw [lev_top_size c pub (by omega) q' (by omega), lev_top_size c pub (by omega) q (by omega)])
      (.cons (by rw [lev_top_size c pub (by omega) q' (by omega), lev_top_size c pub (by omega) q (by omega)])
        .nil))))

/-- **Two ports with one registered root and different domains give an explicit SHA-512 collision**: their top
frames (`topIn`). -/
theorem rootsOf_collides {c : Flock.Circuit} {pub : Flock.Public} {nl q q' : ℕ} (hq' : q' < c.ports.size)
    (hq : q < c.ports.size) (hd : domOf c q' ≠ domOf c q)
    (hr : (rootsOf c pub nl).getD q' [] = (rootsOf c pub nl).getD q []) :
    Registered.Collides (topIn c pub nl q', topIn c pub nl q) := by
  rw [rootsOf_getD c pub nl hq', rootsOf_getD c pub nl hq] at hr
  have hr := bs_inj hr
  rw [root_top, root_top] at hr
  exact ⟨topIn_ne hd, hr⟩

/-! ## The guard the deployed check needs -/

/-- **An earlier port with a port's registered root has its domain** (its name), in the form of `TransportsLiveCS`'s
port-roots guard. -/
def RootFixesDomain (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : Prop :=
  ∀ q q' : ℕ, q' < q → q < c.ports.size →
    (rootsOf c pub nl).getD q' [] = (rootsOf c pub nl).getD q [] → domOf c q' = domOf c q

/-- **The distinct-roots form of the guard**, `TransportsLiveCS`'s hypothesis: from `RootFixesDomain`, at a circuit
that names each port once (`Registered.dupName`, which the verifier's registered reads check). -/
theorem rootsDistinct_of_fixes {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ}
    (hn : Flock.Registered.dupName c = none) (hf : RootFixesDomain c pub nl) :
    ∀ q q' : ℕ, q' < q → q < c.ports.size → (rootsOf c pub nl).getD q' [] ≠ (rootsOf c pub nl).getD q [] := by
  intro q q' hlt hq he
  have hd := hf q q' hlt hq he
  have hname : (c.ports[q']?.map (·.name)).getD "" = (c.ports[q]?.map (·.name)).getD "" :=
    String.toByteArray_inj.1 hd
  have h1 : c.ports[(⟨q', by omega⟩ : Fin c.ports.size)].name = c.ports[(⟨q, hq⟩ : Fin c.ports.size)].name := by
    simpa [show q' < c.ports.size by omega, hq] using hname
  have := congrArg Fin.val (FlockSoundness.Discharge.ZkReg.names_once_of_dupName hn _ _ h1)
  simp at this
  omega

/-! ## The finder, and the term in `ecr/sha-512` -/

/-- Pairs `(q', q)` of ports, `q' < q < m`. -/
def pairs (m : ℕ) : List (ℕ × ℕ) := (List.range m).flatMap fun q => (List.range q).map fun q' => (q', q)

theorem mem_pairs {m q' q : ℕ} : (q', q) ∈ pairs m ↔ q' < q ∧ q < m := by
  simp only [pairs, List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨a, ha, b, hb, rfl, rfl⟩
    exact ⟨hb, ha⟩
  · rintro ⟨h1, h2⟩
    exact ⟨q, h2, q', h1, rfl, rfl⟩

/-- The first pair of ports, earlier first, with one registered root and different domains. -/
def collPair (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : Option (ℕ × ℕ) :=
  (pairs c.ports.size).find? fun p =>
    decide ((rootsOf c pub nl).getD p.1 [] = (rootsOf c pub nl).getD p.2 [] ∧ domOf c p.1 ≠ domOf c p.2)

/-- **The collision a registration's roots expose**: `collPair`'s two top frames. -/
def rootsColl (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : Option (Msg × Msg) :=
  (collPair c pub nl).map fun p => (bs (topIn c pub nl p.1), bs (topIn c pub nl p.2))

/-- **SHA-512 calls to recompute every port's root**: per port, `nl` hm96 tree leaves, `nl` frame-v3 leaves, and at
most `2 · nextPow2 nl` pads and nodes. -/
def rootsCost (c : Flock.Circuit) (nl : ℕ) : ℕ := c.ports.size * (2 * nl + 2 * Flock.nextPow2 nl)

/-- **The roots finder**: no coins; it recomputes every port's registered root and outputs `rootsColl`. -/
def rootsFinder (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : Claims.Finder where
  S := Unit
  out _ := rootsColl c pub nl
  cost _ := rootsCost c nl

theorem rootsColl_collides {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ} {x y : Msg}
    (h : rootsColl c pub nl = some (x, y)) : Claims.Collision sha512Msg x y := by
  obtain ⟨⟨q', q⟩, hp, he⟩ := Option.map_eq_some_iff.mp h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
  have hmem := mem_pairs.mp (List.mem_of_find?_eq_some hp)
  have hpr := List.find?_some hp
  simp only [decide_eq_true_eq] at hpr
  obtain ⟨hne, hh⟩ := rootsOf_collides (hmem.1.trans hmem.2) hmem.2 hpr.2 hpr.1
  refine ⟨fun e => hne (bs_inj e), ?_⟩
  simp only [sha512Msg, bytes_bs, hh]

theorem fixes_of_none {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ} (h : rootsColl c pub nl = none) :
    RootFixesDomain c pub nl := by
  intro q q' hlt hq he
  have hc : collPair c pub nl = none := Option.map_eq_none_iff.mp h
  have := List.find?_eq_none.mp hc (q', q) (mem_pairs.mpr ⟨hlt, hq⟩)
  simp only [decide_eq_true_eq, not_and, not_not] at this
  exact this he

/-- **The decidable form of the guard**: no colliding pair of ports (`rootsColl c pub nl = none`) iff an earlier port
with a port's root has its domain. -/
theorem rootsColl_none_iff {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ} :
    rootsColl c pub nl = none ↔ RootFixesDomain c pub nl := by
  refine ⟨fixes_of_none, fun h => Option.map_eq_none_iff.mpr (List.find?_eq_none.mpr fun ⟨a, b⟩ hp => ?_)⟩
  obtain ⟨hlt, hq⟩ := mem_pairs.mp hp
  simp only [decide_eq_true_eq, not_and, not_not]
  exact h b a hlt hq

theorem finderPr_one {H : Msg → Msg} {F : Claims.Finder}
    (h : ∀ s, ∃ u v, F.out s = some (u, v) ∧ Claims.Collision H u v) : F.Pr H = 1 := by
  have hc : (Fintype.card F.S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  unfold Claims.Finder.Pr Claims.Pr
  simp [h, hc]

theorem finderPr_zero {H : Msg → Msg} {F : Claims.Finder}
    (h : ∀ s, ¬ ∃ u v, F.out s = some (u, v) ∧ Claims.Collision H u v) : F.Pr H = 0 := by
  unfold Claims.Finder.Pr Claims.Pr
  simp [h]

theorem ex_cost (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) :
    Claims.Ex (fun s => ((rootsFinder c pub nl).cost s : ℝ)) = rootsCost c nl := by
  have hc : (Fintype.card (rootsFinder c pub nl).S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  unfold Claims.Ex
  change (∑ _s : (rootsFinder c pub nl).S, (rootsCost c nl : ℝ)) / _ = _
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm, mul_div_assoc, div_self hc, mul_one]

/-- **A registration whose roots expose a collision has `ecr/sha-512`'s term at least 1** at the roots finder, at
any budget of at least its calls. -/
theorem one_le_ecrBound {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ} (hc : rootsColl c pub nl ≠ none) {q : ℝ}
    (hq : (rootsCost c nl : ℝ) ≤ q) : 1 ≤ Claims.ecrBound sha512Msg (rootsFinder c pub nl) q := by
  obtain ⟨⟨x, y⟩, hxy⟩ := Option.ne_none_iff_exists'.mp hc
  have hPr : (rootsFinder c pub nl).Pr sha512Msg = 1 :=
    finderPr_one fun _ => ⟨x, y, hxy, rootsColl_collides hxy⟩
  unfold Claims.ecrBound
  split_ifs with h
  · rw [ENNReal.one_le_ofReal]
    have h1 := h.1
    unfold Claims.ExpectedCR at h1
    rw [hPr, ex_cost] at h1
    exact h1.trans (div_le_div_of_nonneg_right hq (by positivity))
  · exact le_top

/-- **A registration whose roots expose no collision has `ecr/sha-512`'s term `rootsCost / 2^256`** at the roots
finder. -/
theorem ecrBound_of_none {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ} (hc : rootsColl c pub nl = none) :
    Claims.ecrBound sha512Msg (rootsFinder c pub nl) (rootsCost c nl) =
      ENNReal.ofReal ((rootsCost c nl : ℝ) / 2 ^ (256 : ℝ)) := by
  have hPr : (rootsFinder c pub nl).Pr sha512Msg = 0 :=
    finderPr_zero fun _ ⟨_, _, he, _⟩ => Option.some_ne_none _ (he.symm.trans hc)
  unfold Claims.ecrBound
  rw [ite_eq_left]
  refine ⟨?_, (ex_cost c pub nl).le⟩
  unfold Claims.ExpectedCR
  rw [hPr, ex_cost]
  positivity

/-- **The port-roots guard costs one `ecr/sha-512` term**: a probability `p ≤ 1` that is at most `x` at every
registration whose roots fix their domains (`RootFixesDomain`) is at most `x` plus `ecr/sha-512`'s term at the roots
finder, at every registration. -/
theorem le_add_ecrBound {c : Flock.Circuit} {pub : Flock.Public} {nl : ℕ} {p x : ℝ≥0∞} (hp : p ≤ 1)
    (h : RootFixesDomain c pub nl → p ≤ x) :
    p ≤ x + Claims.ecrBound sha512Msg (rootsFinder c pub nl) (rootsCost c nl) := by
  cases hc : rootsColl c pub nl with
  | none => exact (h (fixes_of_none hc)).trans le_self_add
  | some z => exact hp.trans ((one_le_ecrBound (by simp [hc]) le_rfl).trans le_add_self)

end Flock.Bridge.PortRoots
