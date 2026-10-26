import Proofs.Flock.Bridge.Deployed
import Proofs.Flock.Recursive.TopRoot
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# Two ports with one registered root and different domains are a SHA-512 collision

The deployed check opens a position at the first registered port whose root is the position's (`regPath`'s `find?`
over `portsOf`), so a row of port `q` is checked in port `q'`'s tree when an earlier `q'` has `q`'s root. Each
port's root is `FV3.root` over its own domain, its port name (`domOf`), and that root is SHA-512 of one top frame (the
top node's, or with one leaf that leaf's, or with none the pad's; `root_top`), whose first part is the domain. So two
ports with one root and different domains give two different inputs with one SHA-512 digest (`rootsOf_collides`).

* `RootFixesDomain`: an earlier port with a port's root has its domain. Off that collision it holds
  (`fixes_of_none`), and it is what the deployed check needs: two ports of one name and one root are one registered
  port record, so the check opens a row of `q` in `q`'s own tree (`find?_portsOf_fixes`). Two ports of one name may
  share a root with no collision, so the distinct-roots form needs the circuit to name each port once
  (`Registered.dupName`, which the verifier's registered reads check): `rootsDistinct_of_fixes`.
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

/-! ## A port's root and its top frame -/

/-- Port `q`'s registered tree leaves, as `rootsOf` builds them. -/
def leaves (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : Array ByteArray :=
  (Array.range nl).map fun u => Flock.Registered.leafAs Flock.HmRow.ROW_SCHEMA_V2 (domOf c q) u (regLeafOf c pub q u)

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
  else if nl = 1 then Registered.leafInAs Flock.HmRow.ROW_SCHEMA_V2 (domOf c q) 0 (regLeafOf c pub q 0)
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
  · exact frame_head_inj h (.cons rfl (.cons rfl (.cons rfl (.cons (by rw [regLeafOf_size, regLeafOf_size]) .nil))))
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

/-- Port `q`'s registered record (`portsOf`'s `q`-th). -/
def portRec (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) (q : ℕ) : Flock.Registered.Port :=
  { name := (c.ports[q]?.map (·.name)).getD "", value := "", root := bytes (roots.getD q []), leaves := nl,
    words := none, wordBits := 0, domain := domOf c q, readsRel := #[], schema := Flock.HmRow.ROW_SCHEMA_V2 }

theorem portsOf_get (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) {q : ℕ} (hq : q < (portsOf c nl roots).size) :
    (portsOf c nl roots)[q] = portRec c nl roots q := by
  simp [portsOf, portRec]

theorem portsOf_size (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) : (portsOf c nl roots).size = c.ports.size := by
  simp [portsOf]

/-- **The port the deployed check opens a row of port `q` at is `q`'s own record**, when every earlier port with `q`'s
root has `q`'s domain: two ports of one name and one root are one record. This is `find?_portsOf` with its
distinct-roots hypothesis weakened to the guard that holds off a collision. -/
theorem find?_portsOf_fixes (c : Flock.Circuit) (nl : ℕ) (roots : List Msg) {q : ℕ} (hq : q < c.ports.size)
    (hfix : ∀ q' < q, roots.getD q' [] = roots.getD q [] → domOf c q' = domOf c q) :
    (portsOf c nl roots).find? (fun p => decide (bs p.root = roots.getD q [])) =
      some (portRec c nl roots q) := by
  have hqs : q < (portsOf c nl roots).size := by rw [portsOf_size]; exact hq
  have hpq : decide (bs ((portsOf c nl roots)[q]'hqs).root = roots.getD q []) = true := by
    simp [portsOf_get, portRec, bs_bytes]
  obtain ⟨b, hb⟩ : ∃ b, (portsOf c nl roots).find? (fun p => decide (bs p.root = roots.getD q [])) = some b :=
    Option.isSome_iff_exists.mp (Array.find?_isSome.mpr ⟨_, Array.getElem_mem hqs, hpq⟩)
  rw [hb]
  obtain ⟨hpb, i, hi, hib, hmin⟩ := Array.find?_eq_some_iff_getElem.mp hb
  subst hib
  have hiq : i ≤ q := by
    by_contra hlt
    have := hmin q (by omega)
    simp [portsOf_get, portRec, bs_bytes] at this
  rw [portsOf_get]
  rcases Nat.lt_or_ge i q with hlt | hge
  · have hr : roots.getD i [] = roots.getD q [] := by simpa [portsOf_get, portRec, bs_bytes] using hpb
    have hd := hfix i hlt hr
    have hn : (c.ports[i]?.map (·.name)).getD "" = (c.ports[q]?.map (·.name)).getD "" := String.toByteArray_inj.1 hd
    unfold portRec
    rw [hn, hr, hd]
  · obtain rfl : i = q := by omega
    rfl

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
