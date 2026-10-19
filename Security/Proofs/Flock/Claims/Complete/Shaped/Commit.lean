import Proofs.Flock.Claims.Complete.Shaped.Hoare
import Proofs.Flock.Soundness.Refine.Ligerito
import Proofs.Flock.Soundness.LigeritoMid
import Proofs.Flock.Soundness.CompiledLock
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Verifier.Hash

/-!
# A level's commitment and opening, by their shapes

* `commitP_shape`: a level's commitment `commitP` returned through `Hashes` whose node and leaf digests are `hl` bytes
  has the shape the opening reads (`CommitOk`): `2^d` rows of `2^lanes` elements, `2^d` salts of the scheme's size, the
  Merkle layers below the cap, layer `h` of `2^(d − h)` digests, and a cap of `2^cap` digests, each digest `hl` bytes.
* `opsWith_sizes`: every registered scheme's digests are its `digestLen` bytes through `Hashes.ofSha512` at an oracle of
  64-byte outputs (`Claims.Out64`).
* `positions_facts`: a level's query positions are `q` rows of its `2^d`.
* `openP_ok`: an opening at such positions of a commitment of that shape has the shape the decoder reads
  (`ZkView.LevelOk`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy FlockSoundness.Refine FlockSoundness.Discharge

theorem mapM_V {α β : Type} {f : α → V β} {Q : β → Prop} (hf : ∀ a b, f a = .ok b → Q b) {xs : Array α}
    {ys : Array β} (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ y ∈ ys, Q y := by
  have hs := Array.SatisfiesM_mapM (as := xs) (f := f) (motive := fun _ => True) (p := fun _ b => Q b) trivial
    fun i _ => SatisfiesM_Except_eq.2 fun b hb => ⟨hf _ _ hb, trivial⟩
  obtain ⟨-, e, hq⟩ := SatisfiesM_Except_eq.1 hs ys h
  refine ⟨e, fun y hy => ?_⟩
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hy
  exact hq i (e ▸ hi)

theorem forIn_upto_V {β : Type} (n : ℕ) (b : β) {f : ℕ → β → V (ForInStep β)} (I : ℕ → β → Prop)
    (h0 : I 0 b) (h : ∀ i < n, ∀ b r, I i b → f i b = .ok r → ∃ b', r = .yield b' ∧ I (i + 1) b') {r : β}
    (hr : forIn [0:n] b f = .ok r) : I n r := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at hr
  suffices ∀ j s (b : β), s + j = n → I s b → forIn (List.range' s j 1) b f = .ok r → I n r from
    this n 0 b (by omega) h0 hr
  intro j
  induction j with
  | zero =>
    intro s b hs hb e
    rw [List.range'_zero, List.forIn_nil] at e
    cases e
    rw [← hs]; exact hb
  | succ j ih =>
    intro s b hs hb e
    rw [List.range'_succ, List.forIn_cons] at e
    obtain ⟨r', hr', e⟩ := ebind_ok e
    obtain ⟨b', rfl, hb'⟩ := h s (by omega) b r' hb hr'
    exact ih (s + 1) b' (by omega) hb' e

/-- **A level's commitment, by its shape**: `2^d` rows of `2^lanes` elements and `2^d` salts of the scheme's size,
the Merkle layers below the cap, layer `h` of `2^(d − h)` digests, and a cap of `2^cap` digests, each of `hl`
bytes. -/
def CommitOk (hl sb : ℕ) (lv : Level) (cm : Commit) : Prop :=
  cm.rows.size = 2 ^ lv.d ∧ (∀ r ∈ cm.rows, r.size = 2 ^ lv.lanes) ∧ cm.salts.size = 2 ^ lv.d ∧
    (∀ x ∈ cm.salts, x.size = sb) ∧
    (∀ h < lv.d - lv.cap, ∃ hh : h < cm.layers.size, cm.layers[h].size = 2 ^ (lv.d - h) ∧
      ∀ x ∈ cm.layers[h], x.size = hl) ∧
    cm.cap.size = 2 ^ lv.cap ∧ ∀ x ∈ cm.cap, x.size = hl

theorem fitBytes_size (n : ℕ) (b : ByteArray) : (fitBytes n b).size = n := by
  simp only [fitBytes, ByteArray.size_append, ByteArray.size_extract]
  show _ + (Array.replicate _ _).size = n
  rw [Array.size_replicate]
  omega

theorem commitP_shape {ms : MerkleScheme} {hs : Hashes V} {hl : ℕ}
    (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = hl)
    (hlf : ∀ row salt v, (ms.opsWith hs).leaf row salt = .ok v → v.size = hl) {lv : Level} (hcap : lv.cap ≤ lv.d)
    {g : Array F128} {salt : ℕ → ByteArray} {cm : Commit} (h : commitP hs ms lv g salt = .ok cm) :
    CommitOk hl ms.saltBytes lv cm := by
  unfold commitP at h
  obtain ⟨leaves, hlv, h⟩ := ebind_ok h
  obtain ⟨hls, hld⟩ := mapM_V (fun _ _ e => hlf _ _ _ e) hlv
  obtain ⟨r, hr, h⟩ := ebind_ok h
  cases h
  have hI := forIn_upto_V (lv.d - lv.cap) (#[leaves], leaves)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => p.1.size = i + 1 ∧
      (∀ j ≤ i, ∃ hj : j < p.1.size, p.1[j].size = 2 ^ (lv.d - j) ∧ ∀ x ∈ p.1[j], x.size = hl) ∧
      p.2.size = 2 ^ (lv.d - i) ∧ ∀ x ∈ p.2, x.size = hl)
    ⟨rfl, fun j hj => by obtain rfl : j = 0 := by omega
                         exact ⟨by simp, by simpa using hls, hld⟩, by simpa using hls, hld⟩
    (fun i hi p r' hp e => by
      obtain ⟨cur', hc, e⟩ := ebind_ok e
      cases e
      obtain ⟨hcs, hcd⟩ := mapM_V (fun _ _ e => hn _ _ _ e) hc
      obtain ⟨h1, h2, h3, h4⟩ := hp
      have hsz : cur'.size = 2 ^ (lv.d - (i + 1)) := by
        rw [hcs, Array.size_range, h3, show lv.d - i = lv.d - (i + 1) + 1 by omega, Nat.pow_succ,
          Nat.mul_div_cancel _ (by norm_num)]
      refine ⟨_, rfl, by simp [h1], fun j hj => ?_, hsz, hcd⟩
      rcases Nat.lt_or_eq_of_le hj with hj | rfl
      · obtain ⟨hj', e1, e2⟩ := h2 j (by omega)
        refine ⟨by simp; omega, ?_⟩
        simp only [Array.getElem_push, hj', ↓reduceDIte]
        exact ⟨e1, e2⟩
      · refine ⟨by simp [h1], ?_⟩
        simp only [Array.getElem_push, h1, lt_self_iff_false, ↓reduceDIte]
        exact ⟨hsz, hcd⟩) hr
  obtain ⟨-, hL, hC, hCd⟩ := hI
  refine ⟨by simp [encodeRows], fun row hrow => ?_, by simp, fun x hx => ?_, fun h hh => hL h hh.le,
    by rw [hC]; congr 1; omega, hCd⟩
  · simp only [encodeRows, Array.mem_map, Array.mem_range] at hrow
    obtain ⟨_, -, rfl⟩ := hrow
    simp
  · simp only [Array.mem_map, Array.mem_range] at hx
    obtain ⟨_, -, rfl⟩ := hx
    exact fitBytes_size _ _

theorem opsWith_of_lookup {M : Type → Type} [Monad M] [MonadExceptOf String M] {hs : Hashes M} {ms : MerkleScheme}
    {k : UInt32} (h : MerkleScheme.oracleIds.lookup ms.id = some k) :
    ms.opsWith hs = if k = 0 then MerkleOps.sha256With hs else if k = 2 then MerkleOps.sha512With hs
      else if k = 3 then MerkleOps.hm96With hs else MerkleOps.ofScheme ms := by
  unfold MerkleScheme.opsWith
  rw [h]

/-- **Every registered scheme's digests have its width** through `Hashes.ofSha512` at an oracle of 64-byte outputs. -/
theorem opsWith_sizes {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {ms : MerkleScheme}
    (hms : ms ∈ MerkleScheme.all) :
    (∀ l r v, (ms.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V)).node l r = .ok v → v.size = ms.digestLen) ∧
    ∀ row salt v, (ms.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V)).leaf row salt = .ok v →
      v.size = ms.digestLen := by
  have h512 : ∀ b, (Claims.hOf H b).size = 64 := hOf_size h64
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at hms
  rcases hms with rfl | rfl | rfl | rfl
  · have e : MerkleScheme.oracleIds.lookup (MerkleScheme.sha256 "flock-leaf/sha256-unsalted/v0").id = some 0 := by
      decide +kernel
    rw [opsWith_of_lookup e, ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _, fun row salt v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _⟩
  · have e : MerkleScheme.oracleIds.lookup (MerkleScheme.sha256 "flock-leaf/sha256-unsalted").id = some 0 := by
      decide +kernel
    rw [opsWith_of_lookup e, ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _, fun row salt v e => by
      simp only [MerkleOps.sha256With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact Sha256.hash_size _⟩
  · have e : MerkleScheme.oracleIds.lookup (MerkleScheme.sha512 "flock-leaf/sha512-unsalted").id = some 2 := by
      decide +kernel
    rw [opsWith_of_lookup e, ite_eq_right (by decide), ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.sha512With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact h512 _, fun row salt v e => by
      simp only [MerkleOps.sha512With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact h512 _⟩
  · have e : MerkleScheme.oracleIds.lookup MerkleScheme.hm96Sha512.id = some 3 := by decide +kernel
    rw [opsWith_of_lookup e, ite_eq_right (by decide), ite_eq_right (by decide), ite_eq_left rfl]
    exact ⟨fun l r v e => by
      simp only [MerkleOps.hm96With, Hashes.ofSha512, pure, Except.pure, Except.ok.injEq] at e
      rw [← e]; exact h512 _, fun row salt v e => by
      simp only [MerkleOps.hm96With, Hm96.Default512.leafWith, Hashes.ofSha512, bind, Except.bind, pure, Except.pure,
        Except.ok.injEq] at e
      rw [← e]; exact h512 _⟩

/-- **The query positions**: `q` of them, each a row of the level's `2^d`. -/
theorem positions_facts (lv : Level) (hl : LevelSeam lv) (coins : Array F128) :
    (positions lv coins).size = lv.q ∧ ∀ p ∈ positions lv coins, p < 2 ^ lv.d := by
  have e := positions_eq lv hl coins
  have hd : (levelOf lv).logLen = lv.d := hl.2.2.2.2.symm
  constructor
  · have := FlockSoundness.positions_length FlockSoundness.execArith (levelOf lv).queries (levelOf lv).logLen
      (vecOf lv.q coins)
    rw [e] at this
    simpa [levelOf] using this
  · intro p hp
    have := FlockSoundness.positions_lt FlockSoundness.execArith (levelOf lv).queries (levelOf lv).logLen
      (vecOf lv.q coins) p
      (by rw [e]; exact Array.mem_toList_iff.2 hp)
    rwa [hd] at this

theorem getElem!_mem {α : Type} [Inhabited α] {a : Array α} {i : ℕ} (h : i < a.size) : a[i]! ∈ a := by
  rw [getElem!_pos a i h]
  exact Array.getElem_mem h

theorem sibling_size {hl : ℕ} {lv : Level} {cm : Commit}
    (hc : ∀ h < lv.d - lv.cap, ∃ hh : h < cm.layers.size, cm.layers[h].size = 2 ^ (lv.d - h) ∧
      ∀ x ∈ cm.layers[h], x.size = hl) {p h : ℕ} (hp : p < 2 ^ lv.d) (hh : h < lv.d - lv.cap) :
    ((cm.layers[h]!)[(p >>> h) ^^^ 1]!).size = hl := by
  obtain ⟨hh', hs, he⟩ := hc h hh
  rw [getElem!_pos cm.layers h hh'] at *
  refine he _ (getElem!_mem ?_)
  rw [hs]
  have hd : 1 ≤ lv.d - h := by omega
  refine Nat.xor_lt_two_pow ?_ (Nat.one_lt_two_pow (by omega))
  rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (Nat.two_pow_pos h), ← Nat.pow_add,
    Nat.sub_add_cancel (by omega)]
  exact hp

theorem siblings_facts {hl : ℕ} {lv : Level} {cm : Commit}
    (hc : ∀ h < lv.d - lv.cap, ∃ hh : h < cm.layers.size, cm.layers[h].size = 2 ^ (lv.d - h) ∧
      ∀ x ∈ cm.layers[h], x.size = hl) :
    ∀ (l : List ℕ) (acc : Array ByteArray), (∀ p ∈ l, p < 2 ^ lv.d) → (∀ x ∈ acc, x.size = hl) →
      (l.foldl (fun acc p => acc ++ (Array.range (lv.d - lv.cap)).map fun h => (cm.layers[h]!)[(p >>> h) ^^^ 1]!)
        acc).size = acc.size + l.length * (lv.d - lv.cap) ∧
      ∀ x ∈ l.foldl (fun acc p => acc ++ (Array.range (lv.d - lv.cap)).map fun h => (cm.layers[h]!)[(p >>> h) ^^^ 1]!)
        acc, x.size = hl
  | [], acc, _, ha => ⟨by simp, ha⟩
  | p :: l, acc, hp, ha => by
    rw [List.foldl_cons]
    obtain ⟨e1, e2⟩ := siblings_facts hc l _ (fun q hq => hp q (List.mem_cons_of_mem _ hq)) (by
      intro x hx
      rcases Array.mem_append.1 hx with hx | hx
      · exact ha x hx
      · obtain ⟨h, hh, rfl⟩ := Array.mem_map.1 hx
        exact sibling_size hc (hp p List.mem_cons_self) (Array.mem_range.1 hh))
    refine ⟨?_, e2⟩
    rw [e1, Array.size_append, Array.size_map, Array.size_range, List.length_cons]
    ring

/-- **A level's opening, by its commitment's shape**: at `q` positions below `2^d`, rows of the commitment's width,
`q·(d − cap)` siblings of `hashLen` bytes, and the salts the shape reads. -/
theorem openP_ok {sh : Shape} {ms : MerkleScheme} {lv : Level} {cm : Commit} {pos : Array ℕ}
    (hc : CommitOk sh.hashLen ms.saltBytes lv cm) (hq : pos.size = lv.q) (hp : ∀ p ∈ pos, p < 2 ^ lv.d)
    (hsl : (sh.saltLen = none ∧ ms.saltBytes = 0) ∨ sh.saltLen = some ms.saltBytes) :
    ZkView.LevelOk sh lv (2 ^ lv.lanes) (openP ms cm lv pos) := by
  obtain ⟨hr, hrw, hs, hsw, hL, -, -⟩ := hc
  have hsib := siblings_facts hL pos.toList #[] (fun p h => hp p (Array.mem_toList_iff.1 h)) (by simp)
  rw [Array.foldl_toList] at hsib
  refine ⟨by simp [openP, hq], ?_, ?_, hsib.2, ?_⟩
  · intro r h
    simp only [openP, Array.mem_map] at h
    obtain ⟨p, hpm, rfl⟩ := h
    exact hrw _ (getElem!_mem (by rw [hr]; exact hp p hpm))
  · have e := hsib.1
    rw [Array.length_toList, Array.size_empty, Nat.zero_add] at e
    rw [← hq]
    exact e
  · unfold ZkView.SaltsOk
    rcases hsl with ⟨e, e0⟩ | e
    · rw [e]
      simp [openP, e0]
    · rw [e]
      by_cases e0 : ms.saltBytes = 0
      · simp [openP, e0]
      · obtain ⟨s, hs'⟩ : ∃ s, ms.saltBytes = s + 1 := ⟨ms.saltBytes - 1, by omega⟩
        simp only [openP, hs', Nat.add_one_ne_zero, ↓reduceIte, Array.size_map, hq, Array.mem_map, true_and]
        rintro _ ⟨p, hpm, rfl⟩
        rw [← hs']
        exact hsw _ (getElem!_mem (by rw [hs]; exact hp p hpm))

end FlockComplete.JsonRT
