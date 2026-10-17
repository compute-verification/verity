import Proofs.Flock.Claims.KS.ConflictOC
import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Verifier.Merkle

/-!
# The shipped Merkle check logs every node of its paths

`hm96-sha512/v1`'s Merkle check over `OC` (`merkleCheckWith (hm96With Hashes.oc)`), run against `sha512` within `L`,
passed the compiled check, and every SHA-512 input of each opening is a message of `L` (`merkleCheck_logs`): the
row's digest input, the salt digest's, the leaf's, and each node's two children on the climb (`opensIns`). Two
accepted openings of one position against one cap with different rows collide (`Flock.opens_binding`), and the
colliding pair's inputs are inputs of the two openings (`opensPair_mem`), so the two logs hold a SHA-512 collision
(`tiedColl_of_opens`).
-/

namespace FlockDeploy.KS

open Flock

/- `opensPair`'s branch decides an equality of two leaves; unfolding SHA-512 there evaluates it on symbolic input. -/
attribute [local irreducible] Sha512.hash

/-- The SHA-512 inputs of a climb under `hm96-sha512/v1`: each node's two children, leaf side first. -/
def climbIns : ByteArray → ℕ → List ByteArray → List ByteArray
  | _, _, [] => []
  | acc, idx, s :: rest =>
    (if idx % 2 = 0 then acc ++ s else s ++ acc) ::
      climbIns (Sha512.hash (if idx % 2 = 0 then acc ++ s else s ++ acc)) (idx / 2) rest

/-- The SHA-512 inputs of a leaf under `hm96-sha512/v1`: the row's encoding, the salt's, and the leaf's
(`leaf_prefix ‖ b ‖ c`, in `Hm96.leafPair`'s form). -/
def leafIns (row : Array F128) (salt : ByteArray) : List ByteArray :=
  [MerkleScheme.rowBytes row, Hm96.Default512.saltPrefix ++ salt,
    Hm96.Default512.leafPrefix ++ (Hm96.xor64 (Sha512.hash (MerkleScheme.rowBytes row)) (Hm96.Default512.mask salt) ++
      Sha512.hash (Hm96.Default512.saltPrefix ++ salt))]

/-- **The SHA-512 inputs of an opening** under `hm96-sha512/v1`: its leaf's, then its climb's. -/
def opensIns (row : Array F128) (salt : ByteArray) (pos : ℕ) (sibs : List ByteArray) : List ByteArray :=
  leafIns row salt ++ climbIns (MerkleScheme.hm96Sha512.leaf row salt) pos sibs

theorem opsWith_hm96 (hs : Hashes Claims.OCV) : MerkleScheme.hm96Sha512.opsWith hs = MerkleOps.hm96With hs := rfl

/-! ## Runs of the Merkle check's steps -/

theorem Runs.throw {H : Claims.Msg → Claims.Msg} {α : Type} {e : String} {a : α} {L : List Claims.Msg}
    (h : Runs H (throw e : Claims.OCV α) a L) : False := by
  have := h.1
  cases this

theorem Runs.hash {b y : ByteArray} {L : List Claims.Msg} (h : Runs sha512 (Claims.Hashes.oc.sha512 b) y L) :
    y = Sha512.hash b ∧ Claims.toMsg b ∈ L := by
  obtain ⟨h1, h2⟩ := h
  refine ⟨?_, h2 (Claims.toMsg b, sha512 (Claims.toMsg b)) ?_⟩
  · have e : Except.ok (Claims.hOf sha512 b) = (Except.ok y : Except String ByteArray) := h1
    rw [hOf_sha512] at e
    exact (Except.ok.inj e).symm
  · show (Claims.toMsg b, sha512 (Claims.toMsg b)) ∈
      (Claims.OC.eval sha512 (Claims.OC.query (Claims.toMsg b) fun a => Claims.OC.ret (Except.ok (Claims.ofMsg a)))).2
    exact List.Mem.head _

theorem Runs.leafW {x salt y : ByteArray} {L : List Claims.Msg}
    (h : Runs sha512 (Hm96.Default512.leafWith Claims.Hashes.oc x salt) y L) :
    y = Hm96.Default512.leaf x salt ∧ Claims.toMsg (Hm96.Default512.saltPrefix ++ salt) ∈ L ∧
      Claims.toMsg (Hm96.Default512.leafPrefix ++ (Hm96.xor64 x (Hm96.Default512.mask salt) ++
        Sha512.hash (Hm96.Default512.saltPrefix ++ salt))) ∈ L := by
  unfold Hm96.Default512.leafWith at h
  obtain ⟨c, hc, h⟩ := Runs.bind h
  obtain ⟨rfl, hcm⟩ := Runs.hash hc
  obtain ⟨rfl, hlm⟩ := Runs.hash h
  exact ⟨rfl, hcm, hlm⟩

theorem Runs.leafW' {row : Array F128} {salt y : ByteArray} {L : List Claims.Msg}
    (h : Runs sha512 ((MerkleOps.hm96With Claims.Hashes.oc).leaf row salt) y L) :
    y = MerkleScheme.hm96Sha512.leaf row salt ∧ ∀ b ∈ leafIns row salt, Claims.toMsg b ∈ L := by
  obtain ⟨x, hx, h⟩ := Runs.bind h
  obtain ⟨rfl, hxm⟩ := Runs.hash hx
  obtain ⟨rfl, hs, hl⟩ := Runs.leafW h
  refine ⟨rfl, fun b hb => ?_⟩
  simp only [leafIns, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl
  · exact hxm
  · exact hs
  · exact hl

theorem Runs.climbW {acc : ByteArray} {idx : ℕ} {sibs : List ByteArray} {top : ByteArray} {L : List Claims.Msg}
    (h : Runs sha512 (climbWith (MerkleOps.hm96With Claims.Hashes.oc) acc idx sibs) top L) :
    top = climb MerkleScheme.hm96Sha512 acc idx sibs ∧ ∀ b ∈ climbIns acc idx sibs, Claims.toMsg b ∈ L := by
  induction sibs generalizing acc idx with
  | nil => exact ⟨Runs.pure h, fun b hb => by simp [climbIns] at hb⟩
  | cons s rest ih =>
    simp only [climbWith] at h
    by_cases hp : idx % 2 = 0
    · rw [ite_eq_left hp] at h
      obtain ⟨n, hn, h⟩ := Runs.bind h
      obtain ⟨rfl, hl⟩ := Runs.hash (b := acc ++ s) hn
      obtain ⟨htop, hrest⟩ := ih h
      refine ⟨htop.trans ?_, fun b hb => ?_⟩
      · show _ = climb MerkleScheme.hm96Sha512 (if idx % 2 = 0 then _ else _) (idx / 2) rest
        rw [ite_eq_left hp]
        rfl
      · simp only [climbIns, ite_eq_left hp, List.mem_cons] at hb
        rcases hb with rfl | hb
        · exact hl
        · exact hrest b hb
    · rw [ite_eq_right hp] at h
      obtain ⟨n, hn, h⟩ := Runs.bind h
      obtain ⟨rfl, hl⟩ := Runs.hash (b := s ++ acc) hn
      obtain ⟨htop, hrest⟩ := ih h
      refine ⟨htop.trans ?_, fun b hb => ?_⟩
      · show _ = climb MerkleScheme.hm96Sha512 (if idx % 2 = 0 then _ else _) (idx / 2) rest
        rw [ite_eq_right hp]
        rfl
      · simp only [climbIns, ite_eq_right hp, List.mem_cons] at hb
        rcases hb with rfl | hb
        · exact hl
        · exact hrest b hb

theorem Runs.opensW {cap : Array ByteArray} {c d lanes pos : ℕ} {row : Array F128} {salt : ByteArray}
    {sibs : List ByteArray} {L : List Claims.Msg}
    (h : Runs sha512 (opensWith (MerkleOps.hm96With Claims.Hashes.oc) MerkleScheme.hm96Sha512 cap c d lanes pos row
      salt sibs) true L) :
    opens MerkleScheme.hm96Sha512 cap c d lanes pos row salt sibs = true ∧
      ∀ b ∈ opensIns row salt pos sibs, Claims.toMsg b ∈ L := by
  unfold opensWith at h
  split at h
  · next hs =>
    obtain ⟨y, hy, h⟩ := Runs.bind h
    obtain ⟨rfl, hly⟩ := Runs.leafW' hy
    obtain ⟨top, ht, h⟩ := Runs.bind h
    obtain ⟨rfl, hlt⟩ := Runs.climbW ht
    have he := Runs.pure h
    refine ⟨?_, fun b hb => ?_⟩
    · unfold opens
      rw [hs, ← he]
      rfl
    · rcases List.mem_append.1 hb with hb | hb
      · exact hly b hb
      · exact hlt b hb
  · cases Runs.pure h

theorem Runs.findM?_none {α : Type} {p : α → Claims.OCV Bool} {L : List Claims.Msg} :
    ∀ {l : List α}, Runs sha512 (l.findM? p) none L → ∀ a ∈ l, Runs sha512 (p a) false L
  | [], _, a, ha => by cases ha
  | x :: xs, h, a, ha => by
    simp only [List.findM?] at h
    obtain ⟨b, hb, h⟩ := Runs.bind h
    cases b with
    | true => cases Runs.pure h
    | false =>
      rcases List.mem_cons.1 ha with rfl | ha
      · exact hb
      · exact Runs.findM?_none h a ha

/-- **The shipped `hm96-sha512/v1` Merkle check logs every opening**: run against `sha512` within `L`, the compiled
check passed, and every SHA-512 input of query `j`'s opening is a message of `L`. -/
theorem merkleCheck_logs {cap : Array ByteArray} {c d : ℕ} {positions : Array ℕ} {rows : Array (Array F128)}
    {siblings salts : Array ByteArray} {lanes : ℕ} {L : List Claims.Msg}
    (h : Runs sha512 (merkleCheckWith (MerkleOps.hm96With Claims.Hashes.oc) MerkleScheme.hm96Sha512 cap c d positions
      rows siblings salts lanes) () L) :
    merkleCheck MerkleScheme.hm96Sha512 cap c d positions rows siblings salts lanes = .ok () ∧
      ∀ j < positions.size, ∀ b ∈ opensIns rows[j]! (salts.getD j ByteArray.empty) positions[j]!
        (sibsOf siblings (d - c) j), Claims.toMsg b ∈ L := by
  have hpure : merkleCheck MerkleScheme.hm96Sha512 cap c d positions rows siblings salts lanes = .ok () := by
    have hR := With.merkleCheckWith_sim (shadow sha512) (With.opsWith_sim (shadow sha512) shadow_std
      MerkleScheme.hm96Sha512) MerkleScheme.hm96Sha512 cap c d positions rows siblings salts lanes
    have named : MerkleScheme.hm96Sha512.Named := fun s hs => by
      have e : MerkleScheme.find? MerkleScheme.hm96Sha512.id = some MerkleScheme.hm96Sha512 := rfl
      rw [e] at hs
      exact (Option.some.inj hs).symm
    rw [opsWith_hm96, With.opsWith_std _ named, With.ofScheme_merkleCheck] at hR
    exact Runs.std hR h
  refine ⟨hpure, fun j hj b hb => ?_⟩
  unfold merkleCheckWith at h
  split at h
  · exact (Runs.throw h).elim
  split at h
  · exact (Runs.throw h).elim
  obtain ⟨bad, hbad, h⟩ := Runs.bind h
  cases bad with
  | some k => exact (Runs.throw h).elim
  | none =>
    have hj' := Runs.findM?_none hbad j (List.mem_range.2 hj)
    obtain ⟨o, ho, hj'⟩ := Runs.bind hj'
    have hno : false = !o := Runs.pure hj'
    cases o with
    | false => cases hno
    | true => exact (Runs.opensW ho).2 b hb

/-! ## Two openings of one position, collided -/

theorem climbPair_mem : ∀ (s1 s2 : List ByteArray) (a b : ByteArray) (idx : ℕ),
    (climbPair MerkleScheme.hm96Sha512 s1 s2 a b idx).Collides MerkleScheme.hm96Sha512 →
      (climbPair MerkleScheme.hm96Sha512 s1 s2 a b idx).hm96Inputs.1 ∈ climbIns a idx s1 ∧
        (climbPair MerkleScheme.hm96Sha512 s1 s2 a b idx).hm96Inputs.2 ∈ climbIns b idx s2
  | s :: rest, t :: rest2, a, b, idx, h => by
    by_cases hp : idx % 2 = 0
    · have ec : ∀ x y : ByteArray, (if idx % 2 = 0 then x ++ y else y ++ x) = x ++ y := fun _ _ => ite_eq_left hp
      by_cases he : MerkleScheme.hm96Sha512.node a s = MerkleScheme.hm96Sha512.node b t
      · have e : climbPair MerkleScheme.hm96Sha512 (s :: rest) (t :: rest2) a b idx = .node a s b t := by
          show (if idx % 2 = 0 then _ else _) = _
          rw [ite_eq_left hp, ite_eq_left he]
        rw [e]
        refine ⟨?_, ?_⟩
        · show a ++ s ∈ (if idx % 2 = 0 then a ++ s else s ++ a) :: _
          rw [ec]
          exact List.Mem.head _
        · show b ++ t ∈ (if idx % 2 = 0 then b ++ t else t ++ b) :: _
          rw [ec]
          exact List.Mem.head _
      · have e : climbPair MerkleScheme.hm96Sha512 (s :: rest) (t :: rest2) a b idx =
            climbPair MerkleScheme.hm96Sha512 rest rest2 (MerkleScheme.hm96Sha512.node a s)
              (MerkleScheme.hm96Sha512.node b t) (idx / 2) := by
          show (if idx % 2 = 0 then _ else _) = _
          rw [ite_eq_left hp, ite_eq_right he]
        rw [e] at h ⊢
        obtain ⟨h1, h2⟩ := climbPair_mem rest rest2 _ _ (idx / 2) h
        refine ⟨List.Mem.tail _ ?_, List.Mem.tail _ ?_⟩
        · show _ ∈ climbIns (Sha512.hash (if idx % 2 = 0 then a ++ s else s ++ a)) (idx / 2) rest
          rw [ec]
          exact h1
        · show _ ∈ climbIns (Sha512.hash (if idx % 2 = 0 then b ++ t else t ++ b)) (idx / 2) rest2
          rw [ec]
          exact h2
    · have ec : ∀ x y : ByteArray, (if idx % 2 = 0 then x ++ y else y ++ x) = y ++ x := fun _ _ => ite_eq_right hp
      by_cases he : MerkleScheme.hm96Sha512.node s a = MerkleScheme.hm96Sha512.node t b
      · have e : climbPair MerkleScheme.hm96Sha512 (s :: rest) (t :: rest2) a b idx = .node s a t b := by
          show (if idx % 2 = 0 then _ else _) = _
          rw [ite_eq_right hp, ite_eq_left he]
        rw [e]
        refine ⟨?_, ?_⟩
        · show s ++ a ∈ (if idx % 2 = 0 then a ++ s else s ++ a) :: _
          rw [ec]
          exact List.Mem.head _
        · show t ++ b ∈ (if idx % 2 = 0 then b ++ t else t ++ b) :: _
          rw [ec]
          exact List.Mem.head _
      · have e : climbPair MerkleScheme.hm96Sha512 (s :: rest) (t :: rest2) a b idx =
            climbPair MerkleScheme.hm96Sha512 rest rest2 (MerkleScheme.hm96Sha512.node s a)
              (MerkleScheme.hm96Sha512.node t b) (idx / 2) := by
          show (if idx % 2 = 0 then _ else _) = _
          rw [ite_eq_right hp, ite_eq_right he]
        rw [e] at h ⊢
        obtain ⟨h1, h2⟩ := climbPair_mem rest rest2 _ _ (idx / 2) h
        refine ⟨List.Mem.tail _ ?_, List.Mem.tail _ ?_⟩
        · show _ ∈ climbIns (Sha512.hash (if idx % 2 = 0 then a ++ s else s ++ a)) (idx / 2) rest
          rw [ec]
          exact h1
        · show _ ∈ climbIns (Sha512.hash (if idx % 2 = 0 then b ++ t else t ++ b)) (idx / 2) rest2
          rw [ec]
          exact h2
  | [], _, a, b, _, h => (h.2.2.2.2.1 rfl).elim
  | _ :: _, [], a, b, _, h => (h.2.2.2.2.1 rfl).elim

/-- A leaf pair's inputs are its two leaves' inputs. -/
theorem leafPair_mem (r1 r2 : Array F128) (y1 y2 : ByteArray) :
    (MerklePair.leaf r1 r2 y1 y2).hm96Inputs.1 ∈ leafIns r1 y1 ∧
      (MerklePair.leaf r1 r2 y1 y2).hm96Inputs.2 ∈ leafIns r2 y2 := by
  unfold MerklePair.hm96Inputs
  dsimp only
  split
  · exact ⟨List.Mem.head _, List.Mem.head _⟩
  · unfold Hm96.leafPair
    dsimp only
    split
    · exact ⟨List.Mem.tail _ (List.Mem.head _), List.Mem.tail _ (List.Mem.head _)⟩
    · exact ⟨List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _)),
        List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _))⟩

/-- **The colliding pair's inputs are the openings' inputs.** -/
theorem opensPair_mem (pos : ℕ) (r1 : Array F128) (y1 : ByteArray) (s1 : List ByteArray) (r2 : Array F128)
    (y2 : ByteArray) (s2 : List ByteArray)
    (h : (opensPair MerkleScheme.hm96Sha512 pos r1 y1 s1 r2 y2 s2).Collides MerkleScheme.hm96Sha512) :
    (opensPair MerkleScheme.hm96Sha512 pos r1 y1 s1 r2 y2 s2).hm96Inputs.1 ∈ opensIns r1 y1 pos s1 ∧
      (opensPair MerkleScheme.hm96Sha512 pos r1 y1 s1 r2 y2 s2).hm96Inputs.2 ∈ opensIns r2 y2 pos s2 := by
  by_cases hl : MerkleScheme.hm96Sha512.leaf r1 y1 = MerkleScheme.hm96Sha512.leaf r2 y2
  · have e : opensPair MerkleScheme.hm96Sha512 pos r1 y1 s1 r2 y2 s2 = .leaf r1 r2 y1 y2 := ite_eq_left hl
    rw [e]
    obtain ⟨m1, m2⟩ := leafPair_mem r1 r2 y1 y2
    exact ⟨List.mem_append_left _ m1, List.mem_append_left _ m2⟩
  · have e : opensPair MerkleScheme.hm96Sha512 pos r1 y1 s1 r2 y2 s2 =
        climbPair MerkleScheme.hm96Sha512 s1 s2 (MerkleScheme.hm96Sha512.leaf r1 y1)
          (MerkleScheme.hm96Sha512.leaf r2 y2) pos := ite_eq_right hl
    rw [e] at h ⊢
    obtain ⟨h1, h2⟩ := climbPair_mem s1 s2 _ _ pos h
    exact ⟨List.mem_append_right _ h1, List.mem_append_right _ h2⟩

theorem toMsg_injective : Function.Injective Claims.toMsg := fun a b h => by
  rw [← Claims.ofMsg_toMsg a, ← Claims.ofMsg_toMsg b, h]

/-- **Two accepted openings of one position against one cap, with different rows, collide in the two logs.** -/
theorem tiedColl_of_opens {cap : Array ByteArray} {c d lanes pos : ℕ} {r1 r2 : Array F128} {y1 y2 : ByteArray}
    {s1 s2 : List ByteArray} {l : List (Claims.Msg × Claims.Msg)}
    (hw1 : ∀ x ∈ s1, x.size = MerkleScheme.hm96Sha512.digestLen)
    (hw2 : ∀ x ∈ s2, x.size = MerkleScheme.hm96Sha512.digestLen)
    (h1 : opens MerkleScheme.hm96Sha512 cap c d lanes pos r1 y1 s1 = true)
    (h2 : opens MerkleScheme.hm96Sha512 cap c d lanes pos r2 y2 s2 = true) (hr : r1 ≠ r2)
    (hl1 : ∀ b ∈ opensIns r1 y1 pos s1, Claims.toMsg b ∈ l.map Prod.fst)
    (hl2 : ∀ b ∈ opensIns r2 y2 pos s2, Claims.toMsg b ∈ l.map Prod.fst) : Claims.TiedColl sha512 l := by
  have hc : (opensPair MerkleScheme.hm96Sha512 pos r1 y1 s1 r2 y2 s2).Collides MerkleScheme.hm96Sha512 :=
    opens_binding MerkleScheme.hm96Sha512 MerkleScheme.hm96Sha512_sized cap c d lanes pos r1 r2 y1 y2 s1 s2
      hw1 hw2 h1 h2 hr
  obtain ⟨hne, heq⟩ := MerklePair.hm96_inputs _ hc
  obtain ⟨m1, m2⟩ := opensPair_mem pos r1 y1 s1 r2 y2 s2 hc
  refine ⟨_, _, ⟨fun e => hne (toMsg_injective e), ?_⟩, hl1 _ m1, hl2 _ m2⟩
  show Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _))) =
    Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _)))
  rw [Claims.ofMsg_toMsg, Claims.ofMsg_toMsg, heq]

end FlockDeploy.KS
